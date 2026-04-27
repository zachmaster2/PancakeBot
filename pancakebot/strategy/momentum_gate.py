"""Multi-timeframe momentum gate over OKX 1s klines for BTC with independent ETH/SOL signals.

Requires BTC multi-TF returns (configured via ``GateConfig.mtf_lookbacks``)
to all agree in direction and exceed ``GateConfig.mtf_threshold``, and emits
independent ETH and SOL multi-TF directions plus confirmation strengths for
use by the pipeline's sizing and regime-2 logic.

Live data path: per-round parallel REST fetch of BTC/ETH/SOL/BNB
``/history-candles`` windows. The window is sized to exactly what the
strategy consumes: cutoff-aware (newest = ``lock_at - cutoff_seconds*1000
- 1000``) and lookback-aware (``max_lookback + 1`` candles), so a default
cutoff=2 + lookbacks=(3,7,15) fetches the 16-candle window
``[lock_at - 18_000, lock_at - 3_000]`` rather than the prior cutoff-blind
300-candle window. Critically, this avoids requesting candles that
OKX hasn't published yet at fetch time -- the older "request 300 ending
at lock-2s" path crashed ~67% of rounds with
``kline_fetch_integrity_violation`` because OKX's `after`-only filter
slid the window when the newest candle was unavailable.

Empirical (2026-04-27) OKX REST staleness is median ~410ms / p99 ~1.7s,
materially fresher than candle1s WSS push (median ~954ms / p99 ~1.5s+).
The wake-time offset is configured via ``RuntimeConfig.kline_fetch_offset_ms``
and applied by the engine; the gate itself just consumes ``lock_at_ms``.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from pancakebot.log import warn
from pancakebot.market_data.okx_client import (
    OkxClient,
    RETRY_GATE,
    okx_rate_acquire,
)
from pancakebot.util import InvariantError, TransientOkxError


# Maximum consecutive rounds skipped on TransientOkxError before the gate
# escalates to InvariantError. Three rounds at 5min each = ~15 min of
# OKX REST unreachability before the bot fail-louds — long enough to ride
# out an isolated network blip, short enough that the operator notices
# before half a day's worth of opportunity is silently lost.
_MAX_CONSECUTIVE_FETCH_FAILURES = 3

_SYMBOLS_FETCHED = ("btc", "eth", "sol", "bnb")


@dataclass(frozen=True, slots=True)
class MomentumGateConfig:
    """Live/dry-mode gate configuration.

    The ``mtf_lookbacks`` / ``mtf_threshold`` carry the per-strategy
    multi-TF parameters. ``cutoff_seconds`` defines the data window the
    gate consumes (cutoff_ts_ms = lock_at_ms - cutoff_seconds*1000) and
    is the sole source of truth for the strategy-side cutoff. ``candle_count``
    is derived from ``mtf_lookbacks`` at gate-construction time.
    """
    enabled: bool
    bnb_symbol: str          # "BNB-USDT"
    btc_symbol: str          # "BTC-USDT"
    cutoff_seconds: int
    mtf_lookbacks: tuple[int, ...]
    mtf_threshold: float
    eth_symbol: str = "ETH-USDT"
    sol_symbol: str = "SOL-USDT"


@dataclass(frozen=True, slots=True)
class MomentumGateResult:
    signal: str | None       # "Bull", "Bear", or None
    tier: str | None         # "multi_tf"
    skip_reason: str | None
    signal_strength: float = 0.0  # min(|r_lookback|) for adaptive sizing
    eth_confirmation_strength: float = 0.0  # ETH min(|r|) when confirming BTC direction
    sol_confirmation_strength: float = 0.0  # SOL min(|r|) when confirming BTC direction
    # Independent ETH/SOL multi-TF (for regime-2: fires when BTC is silent)
    eth_signal: str | None = None        # ETH's own multi-TF direction
    eth_signal_strength: float = 0.0     # ETH min(|r|) when its own multi-TF fires
    sol_signal: str | None = None        # SOL's own multi-TF direction
    sol_signal_strength: float = 0.0     # SOL min(|r|) when its own multi-TF fires


class MomentumGate:
    """Multi-asset momentum gate: fetches BTC + ETH + SOL + BNB 1s klines
    in parallel via OKX ``/history-candles`` REST per round (live mode).

    Decision-time work: 4 parallel HTTP GETs to OKX (median ~250ms,
    p99 ~500ms, sharing the global 8/s rate budget). The wake-time offset
    is set by the engine via ``RuntimeConfig.kline_fetch_offset_ms`` so
    the fetch lands a configurable margin before ``lock_at``.

    Error handling:
    - Any ``InvariantError`` from any symbol → reraise (bot crashes →
      supervisor restart). These indicate OKX returned a malformed window
      (length/contiguity/boundary violation) and should never silently
      skip.
    - Any ``TransientOkxError`` (any subset of symbols) → emit one
      ``warn("GATE", sym.upper(), "FETCH_FAIL", ...)`` per failed symbol
      and skip the round with reason ``kline_fetch_transient_failure``.
      Increments ``_consecutive_fetch_failures``; reset to 0 on a fully
      successful 4-symbol fetch (regardless of whether downstream
      signal-computation produces a BET or SKIP).
    - Streak >= ``_MAX_CONSECUTIVE_FETCH_FAILURES`` → escalate to
      ``InvariantError("kline_fetch_failure_streak_max_reached: ...")``.
    """

    def __init__(
        self,
        *,
        config: MomentumGateConfig,
        okx_client: OkxClient,
    ) -> None:
        self._cfg = config
        self._client = okx_client
        # Derived: candle_count = max(mtf_lookbacks) + 1. Computed once at
        # construction; used for validation post-fetch and for the live
        # signal computation slice.
        self._candle_count = max(config.mtf_lookbacks) + 1
        # Cached after each evaluate() so the pipeline can use data
        # for auxiliary signals and regime-adaptive sizing without re-fetching.
        self.last_btc_closes: list[float] | None = None
        # Per-pair fetch timing (ms) -- set each evaluate(), logged by caller
        # AFTER timing guard so file I/O doesn't delay bet submission.
        self.last_fetch_timing: dict[str, int] | None = None
        # Raw kline arrays (dict-shaped) from the most recent evaluate().
        # Consumed by pancakebot.runtime.kline_capture for off-critical-path
        # observability writes. Schema unchanged from REST-era: capture
        # serializes [ts, o, h, l, c, v] arrays as dicts via the existing
        # _kline_dict_to_array path.
        self.last_btc_klines_raw: list[dict] | None = None
        self.last_eth_klines_raw: list[dict] | None = None
        self.last_sol_klines_raw: list[dict] | None = None
        self.last_returns: dict[str, float | None] | None = None
        # Consecutive-failure escalation state (TransientOkxError streak).
        self._consecutive_fetch_failures: int = 0
        # ThreadPoolExecutor lives for the gate's lifetime so we don't pay
        # thread-spawn cost every round. max_workers=4 -- one per symbol.
        self._executor = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="kline-fetch",
        )

    @property
    def enabled(self) -> bool:
        return self._cfg.enabled

    def evaluate(
        self,
        *,
        lock_at_ms: int,
    ) -> MomentumGateResult:
        """Fetch BTC/ETH/SOL/BNB 1s klines in parallel and compute the signal.

        ``lock_at_ms`` is the round's lock_at in milliseconds (from
        ``int(round_t.lock_at) * 1000``). The fetch window is sized to
        exactly what the strategy consumes:
            newest_open_ms = lock_at_ms - cutoff_seconds*1000 - 1000
            oldest_open_ms = newest_open_ms - max(mtf_lookbacks) * 1000
            expected_count = max(mtf_lookbacks) + 1
        The newest candle (``open_ts == newest_open_ms``) closes at
        ``lock_at - cutoff_seconds`` -- one second before the strategy's
        cutoff filter at ``open_ts >= cutoff_ts_ms`` would drop it.
        Requesting one less candle than the cutoff-blind window prevents
        OKX from being asked for a candle it hasn't published yet at
        fetch time, which was the root cause of the
        ``kline_fetch_integrity_violation`` crash epidemic (2026-04-27).

        Returns a MomentumGateResult with one of:
        - ``signal`` set + ``skip_reason=None`` on a fired signal
        - ``signal=None`` + ``skip_reason="gate_no_signal"`` on multi-TF miss
        - ``signal=None`` + ``skip_reason="kline_fetch_transient_failure"``
          on any subset of 4-symbol transient failures
        - raises InvariantError on any contract violation OR on
          ``_MAX_CONSECUTIVE_FETCH_FAILURES`` consecutive transient rounds
        """
        if not self._cfg.enabled:
            return MomentumGateResult(
                signal=None, tier=None, skip_reason=None,
            )

        cutoff_ts_ms = lock_at_ms - self._cfg.cutoff_seconds * 1000
        max_lookback = max(self._cfg.mtf_lookbacks)
        newest_open_ms = lock_at_ms - self._cfg.cutoff_seconds * 1000 - 1000
        oldest_open_ms = newest_open_ms - max_lookback * 1000

        # ----- Parallel REST fetch (4 symbols) -----
        # Submit all four upfront so the network round-trips overlap.
        # ThreadPoolExecutor + okx_rate_acquire share the global 8/s budget.
        # ``as_completed`` reaps futures in finish-order so per-symbol
        # ``durations`` reflect true individual latency (not iteration order).
        symbols = {
            "btc": self._cfg.btc_symbol,
            "eth": self._cfg.eth_symbol,
            "sol": self._cfg.sol_symbol,
            "bnb": self._cfg.bnb_symbol,
        }
        submit_time = time.perf_counter()
        futures = {
            self._executor.submit(
                self._client.kline_fetch_window,
                symbol=symbols[sym_short],
                oldest_open_ms=oldest_open_ms,
                newest_open_ms_inclusive=newest_open_ms,
                retry_policy=RETRY_GATE,
                rate_acquire_fn=okx_rate_acquire,
                # Pin both window bounds: prevents OKX from sliding the
                # window when the newest candle isn't yet published, which
                # would otherwise return expected_count older rows and
                # trip the boundary check into a false InvariantError.
                send_before_bound=True,
            ): sym_short
            for sym_short in _SYMBOLS_FETCHED
        }

        results: dict[str, list[list]] = {}
        errors: dict[str, Exception] = {}
        durations: dict[str, int] = {}
        # Block until ALL futures complete (same wait-for-all behaviour as
        # the prior submission-order loop) but timestamp each one at its
        # actual completion, not when the iterator reaches it.
        for fut in as_completed(futures):
            sym_short = futures[fut]
            durations[sym_short] = int((time.perf_counter() - submit_time) * 1000)
            try:
                results[sym_short] = fut.result()
            except (InvariantError, TransientOkxError) as e:
                errors[sym_short] = e
        self.last_fetch_timing = {f"{sym}_ms": ms for sym, ms in durations.items()}

        invariant_errors = [
            (s, e) for s, e in errors.items() if isinstance(e, InvariantError)
        ]
        transient_errors = [
            (s, e) for s, e in errors.items() if isinstance(e, TransientOkxError)
        ]

        # ----- Error handling: InvariantError dominates -----
        # Aggregate every InvariantError detail into a single raise so the
        # operator sees the full picture (rather than only the first). The
        # inner messages already begin with ``kline_fetch_integrity_violation:``
        # so we don't re-prefix the aggregate -- per-symbol-tagged entries
        # are self-explanatory.
        if invariant_errors:
            details = "; ".join(
                f"{sym_short}={e}" for sym_short, e in invariant_errors
            )
            raise InvariantError(details)

        # ----- Error handling: TransientOkxError (any subset) -----
        if transient_errors:
            for sym_short, e in transient_errors:
                # Per-symbol detail goes into a warn log so the operator
                # reconciles the generic skip with the warn-cluster around it.
                warn(
                    "GATE", sym_short.upper(), "FETCH_FAIL",
                    msg=f"reason={e}",
                )
            self._consecutive_fetch_failures += 1
            if self._consecutive_fetch_failures >= _MAX_CONSECUTIVE_FETCH_FAILURES:
                # Capture the latest detail for the fail-loud message.
                latest_sym, latest_err = transient_errors[-1]
                raise InvariantError(
                    f"kline_fetch_failure_streak_max_reached: "
                    f"streak={self._consecutive_fetch_failures} "
                    f"max={_MAX_CONSECUTIVE_FETCH_FAILURES} "
                    f"latest={latest_sym}={latest_err}"
                )
            return self._skip("kline_fetch_transient_failure")

        # ----- All 4 fetched cleanly -- reset streak -----
        # Reset BEFORE signal computation so quiet-market `gate_no_signal`
        # rounds don't falsely escalate the streak. The streak measures
        # OKX-fetch health, not signal availability.
        self._consecutive_fetch_failures = 0

        btc_arr = results["btc"]
        eth_arr = results["eth"]
        sol_arr = results["sol"]
        # ``bnb_arr`` is fetched for BNB-first-class parity (the bot bets on
        # BNB/USD) but the BTC-driven signal does not consume BNB closes.
        # Keeping the fetch in scope guarantees a future BNB-aware strategy
        # works without re-plumbing the data path.
        _bnb_arr = results["bnb"]

        # No trim needed: the cutoff+lookback-aware fetch returns exactly
        # ``candle_count`` candles ending at ``cutoff_ts_ms - 1000`` per
        # symbol, which is the same window the strategy consumes. The
        # ``_validate_klines_raw`` checks below are now the sole shape
        # gate on the live decision path. (Backtest's
        # ``compute_signal_from_klines`` still uses ``_trim_to_window`` to
        # absorb the wider ``candle_count=31`` history-fallback slack
        # produced by the capture-mode merge.)

        # ----- Capture snapshot (BEFORE signal computation) -----
        self.last_btc_klines_raw = _arr_to_dicts(btc_arr)
        self.last_eth_klines_raw = _arr_to_dicts(eth_arr)
        self.last_sol_klines_raw = _arr_to_dicts(sol_arr)
        btc_closes_snap = [float(k[4]) for k in btc_arr] if btc_arr else None
        eth_closes_snap = [float(k[4]) for k in eth_arr] if eth_arr else None
        sol_closes_snap = [float(k[4]) for k in sol_arr] if sol_arr else None
        self.last_returns = _snapshot_returns(
            btc_closes_snap, eth_closes_snap, sol_closes_snap,
            mtf_lookbacks=self._cfg.mtf_lookbacks,
        )

        # ----- Validate fetch shape (defense in depth) -----
        # The fetch primitive guarantees length/contiguity for the
        # exact-sized window; any drift here is a bug in the cutoff math,
        # not OKX -- raise to surface it loudly.
        for label, arr in (("btc", btc_arr), ("eth", eth_arr), ("sol", sol_arr)):
            reason = _validate_klines_raw(
                arr, cutoff_ts_ms, label, candle_count=self._candle_count,
            )
            if reason is not None:
                raise InvariantError(
                    f"kline_fetch_window_validation_failed: {reason}"
                )

        # ----- Compute signal -----
        btc_closes = [float(k[4]) for k in btc_arr]
        eth_closes = [float(k[4]) for k in eth_arr]
        sol_closes = [float(k[4]) for k in sol_arr]
        self.last_btc_closes = btc_closes
        return _compute_signal(
            btc_closes, eth_closes, sol_closes,
            mtf_lookbacks=self._cfg.mtf_lookbacks,
            mtf_threshold=self._cfg.mtf_threshold,
        )

    def shutdown(self) -> None:
        """Shut down the per-gate executor. Best-effort; safe to call multiple times."""
        try:
            self._executor.shutdown(wait=False, cancel_futures=True)
        except Exception:  # noqa: BLE001 -- never block bot shutdown
            pass

    @staticmethod
    def _skip(reason: str) -> MomentumGateResult:
        return MomentumGateResult(
            signal=None, tier=None, skip_reason=reason,
        )


def _arr_to_dicts(arrs: list[list] | None) -> list[dict] | None:
    """Convert kline arrays to the dict-shape the capture module expects.

    The capture worker uses _kline_dict_to_array as its reader contract;
    this is the writer side of the same schema.
    """
    if arrs is None:
        return None
    return [
        {"open_time_ms": int(k[0]), "open": float(k[1]), "high": float(k[2]),
         "low": float(k[3]), "close_price": float(k[4]),
         "volume": float(k[5]) if len(k) > 5 else 0.0}
        for k in arrs
    ]


def _validate_klines_raw(
    klines: list[list] | None, cutoff_ms: int, label: str, *, candle_count: int,
) -> str | None:
    """Validate raw kline arrays (live + backtest paths)."""
    if not klines or len(klines) < candle_count:
        n = 0 if not klines else len(klines)
        return f"gate_{label}_insufficient:got={n}"

    newest_ts = int(klines[-1][0])
    expected_ts = cutoff_ms - 1000
    if newest_ts != expected_ts:
        return f"gate_{label}_unexpected_newest:got={newest_ts},expected={expected_ts}"

    return None


def _snapshot_returns(
    btc_closes: list[float] | None,
    eth_closes: list[float] | None,
    sol_closes: list[float] | None,
    *,
    mtf_lookbacks: tuple[int, ...],
) -> dict[str, float | None]:
    """Compute per-pair, per-lookback returns for capture.

    Used only by ``pancakebot.runtime.kline_capture`` -- this is not on
    the decision path. None values mean the closes list was missing or
    too short for that lookback; the gate's own no-signal logic handles
    those cases independently.
    """
    out: dict[str, float | None] = {}
    pairs = (("btc", btc_closes), ("eth", eth_closes), ("sol", sol_closes))
    for name, closes in pairs:
        for lb in mtf_lookbacks:
            key = f"{name}_r{lb}"
            out[key] = _get_return(closes, lb) if closes is not None else None
    return out


def _get_return(closes: list[float], lookback: int) -> float | None:
    """Return over *lookback* seconds using direct indexing.

    closes[-1] is the price at cutoff.  closes[-(lookback+1)] is the
    price *lookback* seconds before cutoff.
    """
    if len(closes) < lookback + 1:
        return None
    now = closes[-1]
    ago = closes[-(lookback + 1)]
    if ago <= 0:
        return None
    return (now / ago) - 1.0


def compute_signal_from_klines(
    btc_klines: list[list] | None,
    cutoff_ms: int,
    *,
    mtf_lookbacks: tuple[int, ...],
    mtf_threshold: float,
    candle_count: int,
    eth_klines: list[list] | None = None,
    sol_klines: list[list] | None = None,
) -> MomentumGateResult:
    """Compute signal from raw kline arrays (backtest path).

    Trims klines to the same window the live path fetches, validates
    BTC (the signal source), then computes the signal.

    In the post-2026-04-26 backtest pipeline, the loader pre-slices
    each per-round record to exactly ``candle_count`` candles already
    aligned with ``cutoff_ms``, so ``_trim_to_window`` here is a no-op
    in practice. It's retained as defense-in-depth for any caller that
    passes a wider window (e.g. live mode bridging a custom diagnostic).
    """
    if btc_klines is not None:
        btc_klines = _trim_to_window(btc_klines, cutoff_ms, candle_count=candle_count)
    if eth_klines is not None:
        eth_klines = _trim_to_window(eth_klines, cutoff_ms, candle_count=candle_count)
    if sol_klines is not None:
        sol_klines = _trim_to_window(sol_klines, cutoff_ms, candle_count=candle_count)

    # Validate BTC klines (same gate as live path).
    btc_reason = _validate_klines_raw(btc_klines, cutoff_ms, "btc", candle_count=candle_count)
    if btc_reason is not None or btc_klines is None:
        return MomentumGateResult(
            signal=None, tier=None,
            skip_reason=btc_reason or "gate_no_btc_klines",
        )

    btc_closes = [k[4] for k in btc_klines]
    eth_closes = None
    if eth_klines and len(eth_klines) >= candle_count:
        eth_closes = [k[4] for k in eth_klines]
    sol_closes = None
    if sol_klines and len(sol_klines) >= candle_count:
        sol_closes = [k[4] for k in sol_klines]

    return _compute_signal(
        btc_closes, eth_closes, sol_closes,
        mtf_lookbacks=mtf_lookbacks,
        mtf_threshold=mtf_threshold,
    )


def _trim_to_window(klines: list[list], cutoff_ms: int, *, candle_count: int) -> list[list]:
    """Keep only the *candle_count* completed candles before cutoff_ms."""
    before = [k for k in klines if int(k[0]) < cutoff_ms]
    return before[-candle_count:] if len(before) > candle_count else before


def _compute_pair_multi_tf(
    closes: list[float] | None,
    *,
    mtf_lookbacks: tuple[int, ...],
) -> tuple[str | None, float]:
    """Compute multi-TF for a single pair. Returns (direction, min_abs)."""
    if closes is None:
        return None, 0.0
    rets_opt = [_get_return(closes, lb) for lb in mtf_lookbacks]
    if any(r is None for r in rets_opt):
        return None, 0.0
    rets: list[float] = [r for r in rets_opt if r is not None]
    if all(r > 0 for r in rets):
        return "Bull", min(abs(r) for r in rets)
    if all(r < 0 for r in rets):
        return "Bear", min(abs(r) for r in rets)
    return None, 0.0


def _compute_signal(
    btc_closes: list[float] | None,
    eth_closes: list[float] | None = None,
    sol_closes: list[float] | None = None,
    *,
    mtf_lookbacks: tuple[int, ...],
    mtf_threshold: float,
) -> MomentumGateResult:
    """Core signal logic shared by live and backtest paths.

    Multi-TF BTC: all lookbacks must agree in direction and min(|return|)
    must exceed *mtf_threshold*.

    ETH/SOL confirmation: if ETH or SOL multi-TF also fires in the same
    direction, their confirmation strengths are set for sizing boost.
    """
    # Always compute independent ETH/SOL multi-TF (used by regime-2).
    eth_sig, eth_sig_str = _compute_pair_multi_tf(eth_closes, mtf_lookbacks=mtf_lookbacks)
    sol_sig, sol_sig_str = _compute_pair_multi_tf(sol_closes, mtf_lookbacks=mtf_lookbacks)

    def _no_btc_result(skip_reason: str) -> MomentumGateResult:
        return MomentumGateResult(
            signal=None, tier=None,
            skip_reason=skip_reason,
            eth_signal=eth_sig, eth_signal_strength=eth_sig_str,
            sol_signal=sol_sig, sol_signal_strength=sol_sig_str,
        )

    if btc_closes is None:
        return _no_btc_result("gate_no_btc_klines")

    returns = []
    for lb in mtf_lookbacks:
        r = _get_return(btc_closes, lb)
        if r is None:
            return _no_btc_result("gate_no_signal")
        returns.append(r)

    # All must agree in direction
    if not (all(r > 0 for r in returns) or all(r < 0 for r in returns)):
        return _no_btc_result("gate_no_signal")

    min_abs = min(abs(r) for r in returns)
    if min_abs < mtf_threshold:
        return _no_btc_result("gate_no_signal")

    direction = "Bull" if returns[0] > 0 else "Bear"

    # Check ETH/SOL confirmation for sizing boost (reuse pre-computed signals)
    btc_positive = returns[0] > 0

    eth_confirm = 0.0
    if eth_sig is not None and (eth_sig == "Bull") == btc_positive:
        eth_confirm = eth_sig_str

    sol_confirm = 0.0
    if sol_sig is not None and (sol_sig == "Bull") == btc_positive:
        sol_confirm = sol_sig_str

    return MomentumGateResult(
        signal=direction, tier="multi_tf",
        skip_reason=None,
        signal_strength=min_abs,
        eth_confirmation_strength=eth_confirm,
        sol_confirmation_strength=sol_confirm,
        eth_signal=eth_sig, eth_signal_strength=eth_sig_str,
        sol_signal=sol_sig, sol_signal_strength=sol_sig_str,
    )
