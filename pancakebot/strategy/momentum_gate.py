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

Empirical OKX REST behavior (research/p4c_canonical_loop_probe.py): per-
symbol first-try success ~96.9% at 800ms post-close, ~98.3% at 1150ms.
Pooled fetch RTT p95=289ms, p99=363ms. Probe-derived empirical
constants in pancakebot/timing_constants.py.

Live decision path uses ``RETRY_NONE`` (max_attempts=1). A first-try
fail on any symbol -> TransientOkxError -> gate skips with
``kline_fetch_transient_failure``. Retries are reserved for sync (bulk
historical fetch via sync.py) where there's no time pressure.

The wake-time offset is derived from timing_constants.py and applied
by the engine via ``RuntimeConfig.kline_fetch_wakeup_offset_ms``; the
gate itself just consumes ``lock_at_ms``.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from pancakebot.log import warn
from pancakebot.market_data.okx_client import (
    OkxClient,
    RETRY_NONE,
    okx_rate_acquire,
)
from pancakebot.util import InvariantError, TransientOkxError


# Streak counter limit is configured per-instance via
# ``MomentumGateConfig.max_consecutive_fetch_failures`` (threaded from
# ``cfg.max_consecutive_fetch_failures`` at config load). The constant
# below is the canonical default + the value used by tests that don't
# construct a custom MomentumGateConfig.
#
# At default max=5 with empirical per-round transient-failure rate ~12%
# (research/p4c_canonical_loop_probe.py n=1000 at 800ms post-close):
# P(5 consecutive transient) = 0.12^5 = 2.5e-5 -> expected crash once
# per ~40,000 rounds = ~14 weeks at 12 rounds/h.
_MAX_CONSECUTIVE_FETCH_FAILURES = 5

_SYMBOLS_FETCHED = ("btc", "eth", "sol", "bnb")


@dataclass(frozen=True, slots=True)
class MomentumGateConfig:
    """Live/dry-mode gate configuration.

    The ``mtf_lookbacks`` / ``mtf_threshold`` carry the per-strategy
    multi-TF parameters. ``cutoff_seconds`` defines the data window the
    gate consumes (cutoff_ts_ms = lock_at_ms - cutoff_seconds*1000) and
    is the sole source of truth for the strategy-side cutoff. ``candle_count``
    is derived from ``mtf_lookbacks`` at gate-construction time.

    ``max_consecutive_fetch_failures`` is the streak counter: after this
    many consecutive ``kline_fetch_transient_failure`` rounds, the gate
    raises InvariantError -> bot crashes -> supervisor restart + Discord
    alert. Threaded from ``cfg.max_consecutive_fetch_failures``.
    """
    enabled: bool
    bnb_symbol: str          # "BNB-USDT"
    btc_symbol: str          # "BTC-USDT"
    cutoff_seconds: int
    mtf_lookbacks: tuple[int, ...]
    mtf_threshold: float
    max_consecutive_fetch_failures: int = 5
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

    Decision-time work: 4 parallel HTTP GETs to OKX (pooled RTT
    p50=258ms, p95=289ms, p99=363ms per probe n=1000 2026-05-03).
    The wake-time offset is set by the engine via
    ``RuntimeConfig.kline_fetch_wakeup_offset_ms``; the gate consumes
    ``lock_at_ms``.

    Error handling:
    - Any ``InvariantError`` from any symbol → reraise (bot crashes →
      supervisor restart). These indicate OKX returned a malformed window
      (length/contiguity/boundary violation) and should never silently
      skip.
    - Any ``TransientOkxError`` (any subset of symbols) → emit one
      ``warn("GATE", sym.upper(), "FETCH_FAIL", ...)`` per failed symbol
      and skip the round with reason ``kline_fetch_transient_failure``.
      Increments ``_consecutive_fetch_failures``; reset to 0 on a fully
      successful 4-symbol fetch.
    - Streak >= ``MomentumGateConfig.max_consecutive_fetch_failures`` →
      escalate to ``InvariantError("kline_fetch_failure_streak_max_reached: ...")``.
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
        # Per-pair true HTTP RTT (ms), keyed ``{btc,eth,sol,bnb}_ms`` --
        # the duration of the successful ``session.get(...)`` call only
        # (excludes rate-limiter wait, excludes thread-pool scheduling,
        # excludes retry backoff). Set each evaluate(), logged by the
        # caller AFTER the timing guard so file I/O doesn't delay bet
        # submission.
        self.last_fetch_timing: dict[str, int] | None = None
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
          ``max_consecutive_fetch_failures`` consecutive transient rounds
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
        # ThreadPoolExecutor + okx_rate_acquire share the global token-bucket
        # budget (capacity 8, refill 8/s) -- a 4-symbol burst once per round
        # fits inside the bucket and fires with no FIFO stagger.
        symbols = {
            "btc": self._cfg.btc_symbol,
            "eth": self._cfg.eth_symbol,
            "sol": self._cfg.sol_symbol,
            "bnb": self._cfg.bnb_symbol,
        }
        futures = {
            self._executor.submit(
                self._client.kline_fetch_window,
                symbol=symbols[sym_short],
                oldest_open_ms=oldest_open_ms,
                newest_open_ms_inclusive=newest_open_ms,
                retry_policy=RETRY_NONE,
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
        # Reap futures in finish-order. ``rtt_ms`` is the per-symbol HTTP
        # RTT measured INSIDE ``kline_fetch_window`` around the
        # ``session.get(...)`` call only -- excludes rate-limiter wait,
        # retry-backoff sleeps, and thread-pool scheduling. So this
        # reflects true network round-trip per symbol, not "wall-clock
        # since submit-loop start" (which was the prior measurement).
        for fut in as_completed(futures):
            sym_short = futures[fut]
            try:
                rows, rtt_ms = fut.result()
                results[sym_short] = rows
                durations[sym_short] = rtt_ms
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
            if self._consecutive_fetch_failures >= self._cfg.max_consecutive_fetch_failures:
                # Capture the latest detail for the fail-loud message.
                latest_sym, latest_err = transient_errors[-1]
                raise InvariantError(
                    f"kline_fetch_failure_streak_max_reached: "
                    f"streak={self._consecutive_fetch_failures} "
                    f"max={self._cfg.max_consecutive_fetch_failures} "
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
        # ``_validate_klines_raw`` checks below are the sole shape gate
        # on the live decision path.

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

    Validates BTC (the signal source), then computes the signal.
    All callers pre-slice each per-round record to exactly ``candle_count``
    candles aligned with ``cutoff_ms`` (see
    ``pancakebot.backtest.runner._load_klines_from`` and
    ``research.in_process_runner._slice_per_entry``).
    """
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
