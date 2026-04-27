"""Multi-timeframe momentum gate over OKX 1s klines for BTC with independent ETH/SOL signals.

Requires BTC multi-TF returns (configured via ``GateConfig.mtf_lookbacks``)
to all agree in direction and exceed ``GateConfig.mtf_threshold``, and emits
independent ETH and SOL multi-TF directions plus confirmation strengths for
use by the pipeline's sizing and regime-2 logic.

Lookbacks and threshold live in ``pancakebot.config.GateConfig`` (TOML
``[strategy.gate]``); previously module-level constants
``_MTF_LOOKBACKS = (3, 7, 15)`` / ``_MTF_THRESH = 0.0001``. The candle
count consumed by the gate is derived: ``candle_count = max(mtf_lookbacks) + 1``.
"""

from __future__ import annotations

from dataclasses import dataclass

from pancakebot.log import warn
from pancakebot.market_data.okx_client import OkxClient


# Single skip reason emitted for ANY WSS data-availability failure across
# any of the four subscribed symbols (BTC/ETH/SOL/BNB). Per-failure detail
# is logged via per-symbol ``warn()`` calls so the operator reconciles the
# generic histogram entry with the warn-log cluster around it.
#
# Phase 2 spec item 16 (2026-04-27) -- supersedes the per-reason +
# per-symbol-suffix variants from items 1B / 9 / 15. Rationale:
#   - Histogram cardinality stays low (1 variant, not 5*4=20).
#   - Multi-failure rounds aren't masked by a single arbitrary "first" reason.
#   - All-4-failing rounds become very visible (4 clustered warns + 1 skip).
_WSS_FAILURE_SKIP_REASON = "risk_kline_wss_failure"


@dataclass(frozen=True, slots=True)
class MomentumGateConfig:
    """Live/dry-mode gate configuration.

    The ``mtf_lookbacks`` / ``mtf_threshold`` carry the per-strategy
    multi-TF parameters (formerly module constants). ``candle_count`` is
    derived from ``mtf_lookbacks`` at gate-construction time.
    """
    enabled: bool
    bnb_symbol: str          # "BNB-USDT"
    btc_symbol: str          # "BTC-USDT"
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
    """Multi-asset momentum gate: reads BTC + ETH + SOL + BNB 1s klines
    from OKX WSS in-memory ring buffers (live mode).

    Per the WSS migration: live mode reads from ``OkxWssClient`` ring
    buffers; the per-round REST fetch path is REMOVED. Bootstrap REST and
    reconnect gap-fill REST survive inside ``OkxWssClient`` for
    initialization-only purposes.

    The OkxClient reference is preserved purely so the WSS client (which
    holds the OKX HTTP session) is reachable from the gate for
    diagnostics; the gate itself no longer issues OKX HTTP requests.
    """

    def __init__(
        self,
        *,
        config: MomentumGateConfig,
        okx_client: OkxClient,
        wss_client=None,
    ) -> None:
        self._cfg = config
        self._client = okx_client
        # Derived: candle_count = max(mtf_lookbacks) + 1. Computed once at
        # construction; used for OKX ring-window length, validation, and
        # backtest pre-trim.
        self._candle_count = max(config.mtf_lookbacks) + 1
        # WSS client is REQUIRED for live runtime use (engine constructs
        # it before instantiating the gate). It's accepted as Optional
        # so backtest-side code paths can construct a "no-op" gate via
        # MomentumOnlyPipeline(gate=None) without setting up WSS.
        # If ``evaluate()`` is called with wss_client=None it raises --
        # the live runtime should never reach that state.
        self._wss = wss_client
        # Cached after each evaluate() so the pipeline can use data
        # for auxiliary signals and regime-adaptive sizing without re-fetching.
        self.last_btc_closes: list[float] | None = None
        # Per-pair fetch timing (ms) -- set each evaluate(), logged by caller
        # AFTER timing guard so file I/O doesn't delay bet submission.
        # In WSS path, "fetch" is an in-memory dict lookup (sub-ms), so this
        # field is set to {} per evaluate to keep the capture-format
        # invariant. Pre-fix REST path tracked actual TLS+fetch latency.
        self.last_fetch_timing: dict[str, int] | None = None
        # Raw kline arrays + computed returns from the most recent evaluate().
        # Consumed by pancakebot.runtime.kline_capture for off-critical-path
        # observability writes. Schema unchanged from REST-era: capture
        # serializes [ts, o, h, l, c, v] arrays as dicts via the existing
        # _kline_dict_to_array path, but in WSS mode these are ALREADY
        # arrays (the ring stores arrays). We materialize lightweight dict
        # wrappers below to keep capture's reader contract stable.
        self.last_btc_klines_raw: list[dict] | None = None
        self.last_eth_klines_raw: list[dict] | None = None
        self.last_sol_klines_raw: list[dict] | None = None
        self.last_returns: dict[str, float | None] | None = None

    @property
    def enabled(self) -> bool:
        return self._cfg.enabled

    def evaluate(
        self,
        *,
        cutoff_ts_ms: int,
    ) -> MomentumGateResult:
        """Compute signal from the WSS in-memory ring buffers.

        Live mode: reads ``self._wss.get_window(symbol, cutoff_ms, candle_count)``
        for BTC, ETH, SOL, BNB. BTC drives the primary signal; ETH/SOL feed
        regime-2 / BTC confirmation; BNB is the asset we bet on -- its window
        is read for parity (data-availability check + ``needs_reconnect``
        signal side effect) but the kline values are not consumed by signal
        computation. Sub-millisecond decision-time read.

        WSS failure handling (Phase 2 spec item 16, 2026-04-27): if ANY of
        the four ``get_window`` calls returns ``None``, the round is
        skipped with a SINGLE generic ``risk_kline_wss_failure`` reason
        AND one ``warn()`` is emitted per failed symbol naming the specific
        WSS reason (so the operator reconciles a histogram entry with the
        warn-log cluster around it). All four calls fire upfront so
        ``needs_reconnect`` side effects propagate even when an earlier
        symbol fails.
        """
        if not self._cfg.enabled:
            return MomentumGateResult(
                signal=None, tier=None, skip_reason=None,
            )

        if self._wss is None:
            # Live runtime should always provide wss_client. This branch
            # protects test paths that construct MomentumGate without WSS;
            # they get a clean skip rather than a confusing AttributeError.
            return self._skip("gate_no_wss_client")

        cc = self._candle_count
        # ----- Read rings (all four upfront) -----
        # Returns (klines, skip_reason). klines is list of
        # [ts_ms, o, h, l, c, v] arrays oldest-first; None on any WSS skip.
        # All four reads happen BEFORE any None-check so each call's side
        # effect (notably ``needs_reconnect`` from item 13) fires regardless
        # of which symbol(s) ultimately fail.
        btc_arr, btc_reason = self._wss.get_window(self._cfg.btc_symbol, cutoff_ts_ms, cc)
        eth_arr, eth_reason = self._wss.get_window(self._cfg.eth_symbol, cutoff_ts_ms, cc)
        sol_arr, sol_reason = self._wss.get_window(self._cfg.sol_symbol, cutoff_ts_ms, cc)
        bnb_arr, bnb_reason = self._wss.get_window(self._cfg.bnb_symbol, cutoff_ts_ms, cc)

        # In-memory read; no fetch-timing meaningful (sub-ms).
        self.last_fetch_timing = {"btc_ms": 0, "eth_ms": 0, "sol_ms": 0}

        # ----- Capture snapshot (BEFORE validation early-return) -----
        # Materialize dict-shaped wrappers for the capture module's
        # _kline_dict_to_array reader. This preserves the existing
        # captured.jsonl schema unchanged from the REST era.
        def _arr_to_dicts(arrs: list[list] | None) -> list[dict] | None:
            if arrs is None:
                return None
            return [
                {"open_time_ms": int(k[0]), "open": float(k[1]), "high": float(k[2]),
                 "low": float(k[3]), "close_price": float(k[4]),
                 "volume": float(k[5]) if len(k) > 5 else 0.0}
                for k in arrs
            ]

        self.last_btc_klines_raw = _arr_to_dicts(btc_arr)
        self.last_eth_klines_raw = _arr_to_dicts(eth_arr)
        self.last_sol_klines_raw = _arr_to_dicts(sol_arr)
        # Returns snapshot (uses available closes; partial data still produces
        # something for capture, even on stale paths).
        _btc_closes_snap = [float(k[4]) for k in btc_arr] if btc_arr else None
        _eth_closes_snap = [float(k[4]) for k in eth_arr] if eth_arr else None
        _sol_closes_snap = [float(k[4]) for k in sol_arr] if sol_arr else None
        self.last_returns = _snapshot_returns(
            _btc_closes_snap, _eth_closes_snap, _sol_closes_snap,
            mtf_lookbacks=self._cfg.mtf_lookbacks,
        )

        # ----- WSS failure handling (Phase 2 spec item 16) -----
        # Collect every symbol that returned a skip reason from get_window.
        # If ANY failed, log one warn() per failed symbol naming the
        # specific reason, then skip the round with the single generic
        # ``risk_kline_wss_failure`` reason.
        failures: list[tuple[str, str]] = []
        if btc_arr is None:
            failures.append(("btc", btc_reason or "unknown"))
        if eth_arr is None:
            failures.append(("eth", eth_reason or "unknown"))
        if sol_arr is None:
            failures.append(("sol", sol_reason or "unknown"))
        if bnb_arr is None:
            failures.append(("bnb", bnb_reason or "unknown"))
        if failures:
            for sym, reason in failures:
                # Note: event tag is "FAIL" (NOT "SKIP") -- the generic
                # skip reason already conveys the round was skipped; this
                # log row is the per-symbol detail for operator triage.
                warn("WSS_GATE", sym.upper(), "FAIL", msg=f"reason={reason}")
            return self._skip(_WSS_FAILURE_SKIP_REASON)

        # ----- Compute signal -----
        # Past the failures-collection block above, all four arrays are
        # non-None and length-exactly-``cc`` (post-item-9+10 contract:
        # ``get_window`` returns ``wss_insufficient`` when too few candles
        # exist before cutoff, and ``wss_newest_lagging`` when the newest
        # is behind expected -- both surface as failures-collection skips
        # before we get here). All previously-defensive ``len(<x>_arr) < cc``
        # / ``if x_arr and len(x_arr) >= cc else None`` branches were
        # provably dead and removed in Phase 2 spec item 17 part D
        # (2026-04-27).
        btc_closes = [float(k[4]) for k in btc_arr]
        eth_closes = [float(k[4]) for k in eth_arr]
        sol_closes = [float(k[4]) for k in sol_arr]
        self.last_btc_closes = btc_closes
        return _compute_signal(
            btc_closes, eth_closes, sol_closes,
            mtf_lookbacks=self._cfg.mtf_lookbacks,
            mtf_threshold=self._cfg.mtf_threshold,
        )

    @staticmethod
    def _skip(reason: str) -> MomentumGateResult:
        return MomentumGateResult(
            signal=None, tier=None, skip_reason=reason,
        )


def _validate_klines_raw(
    klines: list[list] | None, cutoff_ms: int, label: str, *, candle_count: int,
) -> str | None:
    """Validate raw kline arrays (backtest path, list-of-lists format)."""
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
