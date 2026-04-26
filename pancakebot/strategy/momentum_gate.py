"""Multi-timeframe momentum gate over OKX 1s klines for BTC with independent ETH/SOL signals.

Requires BTC 3s/7s/15s returns to all agree and exceed a strength threshold,
and emits independent ETH and SOL multi-TF directions plus confirmation
strengths for use by the pipeline's sizing and regime-2 logic.
"""

from __future__ import annotations

from dataclasses import dataclass

from pancakebot.market_data.okx_client import OkxClient

# Number of 1s candles fetched and used by all modes (live, sync, backtest).
_CANDLE_COUNT = 31

# Multi-TF BTC lookbacks -- all must agree in direction.
_MTF_LOOKBACKS = (3, 7, 15)
# Signal fires at this threshold; the pipeline may apply a stricter
# threshold for small pools (pool-adaptive logic in momentum_pipeline.py).
_MTF_THRESH = 0.0001


@dataclass(frozen=True, slots=True)
class MomentumGateConfig:
    enabled: bool
    bnb_symbol: str          # "BNB-USDT"
    btc_symbol: str          # "BTC-USDT"
    eth_symbol: str = "ETH-USDT"
    sol_symbol: str = "SOL-USDT"


@dataclass(frozen=True, slots=True)
class MomentumGateResult:
    signal: str | None       # "Bull", "Bear", or None
    tier: str | None         # "multi_tf"
    btc_agrees: bool         # kept for interface compat (always True for MTF)
    btc_disagrees: bool      # kept for interface compat (always False for MTF)
    skip_reason: str | None
    signal_strength: float = 0.0  # min(|r3|, |r7|, |r15|) for adaptive sizing
    eth_confirmation_strength: float = 0.0  # ETH min(|r|) when confirming BTC direction
    sol_confirmation_strength: float = 0.0  # SOL min(|r|) when confirming BTC direction
    # Independent ETH/SOL multi-TF (for regime-2: fires when BTC is silent)
    eth_signal: str | None = None        # ETH's own multi-TF direction
    eth_signal_strength: float = 0.0     # ETH min(|r|) when its own multi-TF fires
    sol_signal: str | None = None        # SOL's own multi-TF direction
    sol_signal_strength: float = 0.0     # SOL min(|r|) when its own multi-TF fires


class MomentumGate:
    """Multi-asset momentum gate: reads BTC + ETH + SOL 1s klines from
    OKX WSS in-memory ring buffers (live mode).

    Per the WSS migration (research/okx_wss_migration_design.md): live
    mode reads from ``OkxWssClient`` ring buffers; the per-round REST
    fetch path is REMOVED. Bootstrap REST and reconnect gap-fill REST
    survive inside ``OkxWssClient`` for initialization-only purposes.

    The OkxClient reference is preserved for those bootstrap/gap-fill
    calls and for the capture-time response_headers diagnostic
    (which is no longer applicable in WSS path -- left for backwards
    compat with the captured.jsonl schema; always None now).
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
        # Diagnostic HTTP response headers were used for the REST-era
        # cache investigation (commit 1cb2b20). With WSS there are no
        # per-round HTTP responses, so these are always None now.
        # Kept for capture schema compatibility; readers tolerate None.
        self.last_btc_response_headers: dict[str, str] | None = None
        self.last_eth_response_headers: dict[str, str] | None = None
        self.last_sol_response_headers: dict[str, str] | None = None

    @property
    def enabled(self) -> bool:
        return self._cfg.enabled

    def evaluate(
        self,
        *,
        cutoff_ts_ms: int,
    ) -> MomentumGateResult:
        """Compute signal from the WSS in-memory ring buffers.

        Live mode: reads ``self._wss.get_window(symbol, cutoff_ms)`` for
        BTC, ETH, SOL. ETH/SOL are independent feeds (regime-2 signal /
        BTC-confirmation source). Sub-millisecond decision-time read.

        Multi-instrument independence (per design doc):
          - BTC stale -> _skip("risk_kline_wss_stale")
          - ETH+SOL both stale -> _skip("risk_kline_wss_correlated_stale")
          - One of ETH/SOL stale alone -> degraded BTC-primary
            (no regime-2 signal that round)
          - BNB never read here (capture-only, not in decision path)
        """
        if not self._cfg.enabled:
            return MomentumGateResult(
                signal=None, tier=None, btc_agrees=False, btc_disagrees=False,
                skip_reason=None,
            )

        if self._wss is None:
            # Live runtime should always provide wss_client. This branch
            # protects test paths that construct MomentumGate without WSS;
            # they get a clean skip rather than a confusing AttributeError.
            return self._skip("gate_no_wss_client")

        # ----- Read rings -----
        # Returns (klines, skip_reason). klines is list of
        # [ts_ms, o, h, l, c, v] arrays oldest-first; None on stale or
        # insufficient.
        btc_arr, btc_reason = self._wss.get_window(self._cfg.btc_symbol, cutoff_ts_ms, _CANDLE_COUNT)
        eth_arr, eth_reason = self._wss.get_window(self._cfg.eth_symbol, cutoff_ts_ms, _CANDLE_COUNT)
        sol_arr, sol_reason = self._wss.get_window(self._cfg.sol_symbol, cutoff_ts_ms, _CANDLE_COUNT)

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
        # response_headers no longer apply (no per-round HTTP). Always None.
        self.last_btc_response_headers = None
        self.last_eth_response_headers = None
        self.last_sol_response_headers = None
        # Returns snapshot (uses available closes; partial data still produces
        # something for capture, even on stale paths).
        _btc_closes_snap = [float(k[4]) for k in btc_arr] if btc_arr else None
        _eth_closes_snap = [float(k[4]) for k in eth_arr] if eth_arr else None
        _sol_closes_snap = [float(k[4]) for k in sol_arr] if sol_arr else None
        self.last_returns = _snapshot_returns(
            _btc_closes_snap, _eth_closes_snap, _sol_closes_snap,
        )

        # ----- Multi-instrument independence policy -----
        # BTC stale = hard block.
        if btc_arr is None:
            return self._skip("risk_kline_wss_stale" if btc_reason == "wss_stale"
                              else f"gate_btc_{btc_reason}")
        # ETH+SOL both stale (correlated outage indicator) = block.
        if eth_arr is None and sol_arr is None and eth_reason == "wss_stale" and sol_reason == "wss_stale":
            return self._skip("risk_kline_wss_correlated_stale")
        # Either ETH or SOL alone stale: degrade -- pass None so
        # _compute_signal treats them as silent and we lose only the
        # regime-2 / confirmation paths.
        # (No additional code needed; _compute_signal accepts None closes.)

        # ----- Validate (legacy-equivalent) -----
        # Validation now amounts to: BTC has >= _CANDLE_COUNT candles AND
        # the newest candle's open_time matches cutoff_ms - 1000. WSS commits
        # candles aligned to UTC seconds via the confirm=="1" gate, so the
        # newest-ts check is a sanity assertion rather than a reliability
        # gate (the ring won't include in-progress candles in the first
        # place).
        if len(btc_arr) < _CANDLE_COUNT:
            return self._skip("gate_btc_wss_insufficient")
        if int(btc_arr[-1][0]) != cutoff_ts_ms - 1000:
            return self._skip(
                f"gate_btc_unexpected_newest:got={btc_arr[-1][0]},expected={cutoff_ts_ms - 1000}"
            )

        # ----- Compute signal -----
        btc_closes = [float(k[4]) for k in btc_arr]
        eth_closes = (
            [float(k[4]) for k in eth_arr] if eth_arr and len(eth_arr) >= _CANDLE_COUNT else None
        )
        sol_closes = (
            [float(k[4]) for k in sol_arr] if sol_arr and len(sol_arr) >= _CANDLE_COUNT else None
        )
        self.last_btc_closes = btc_closes
        return _compute_signal(btc_closes, eth_closes, sol_closes)

    @staticmethod
    def _skip(reason: str) -> MomentumGateResult:
        return MomentumGateResult(
            signal=None, tier=None, btc_agrees=False, btc_disagrees=False,
            skip_reason=reason,
        )


def _validate_klines_raw(
    klines: list[list] | None, cutoff_ms: int, label: str,
) -> str | None:
    """Validate raw kline arrays (backtest path, list-of-lists format)."""
    if not klines or len(klines) < _CANDLE_COUNT:
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
        for lb in _MTF_LOOKBACKS:
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
    eth_klines: list[list] | None = None,
    sol_klines: list[list] | None = None,
) -> MomentumGateResult:
    """Compute signal from raw kline arrays (backtest path).

    Trims klines to the same window the live path fetches, validates
    BTC (the signal source), then computes the signal.
    """
    if btc_klines is not None:
        btc_klines = _trim_to_window(btc_klines, cutoff_ms)
    if eth_klines is not None:
        eth_klines = _trim_to_window(eth_klines, cutoff_ms)
    if sol_klines is not None:
        sol_klines = _trim_to_window(sol_klines, cutoff_ms)

    # Validate BTC klines (same gate as live path).
    btc_reason = _validate_klines_raw(btc_klines, cutoff_ms, "btc")
    if btc_reason is not None or btc_klines is None:
        return MomentumGateResult(
            signal=None, tier=None, btc_agrees=False, btc_disagrees=False,
            skip_reason=btc_reason or "gate_no_btc_klines",
        )

    btc_closes = [k[4] for k in btc_klines]
    eth_closes = None
    if eth_klines and len(eth_klines) >= _CANDLE_COUNT:
        eth_closes = [k[4] for k in eth_klines]
    sol_closes = None
    if sol_klines and len(sol_klines) >= _CANDLE_COUNT:
        sol_closes = [k[4] for k in sol_klines]

    return _compute_signal(btc_closes, eth_closes, sol_closes)


def _trim_to_window(klines: list[list], cutoff_ms: int) -> list[list]:
    """Keep only the *_CANDLE_COUNT* completed candles before cutoff_ms."""
    before = [k for k in klines if int(k[0]) < cutoff_ms]
    return before[-_CANDLE_COUNT:] if len(before) > _CANDLE_COUNT else before


def _compute_pair_multi_tf(
    closes: list[float] | None,
) -> tuple[str | None, float]:
    """Compute multi-TF(3,7,15) for a single pair. Returns (direction, min_abs)."""
    if closes is None:
        return None, 0.0
    rets_opt = [_get_return(closes, lb) for lb in _MTF_LOOKBACKS]
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
) -> MomentumGateResult:
    """Core signal logic shared by live and backtest paths.

    Multi-TF BTC: all lookbacks (3, 7, 15) must agree in direction
    and min(|return|) must exceed _MTF_THRESH.

    ETH/SOL confirmation: if ETH or SOL multi-TF also fires in the same
    direction, their confirmation strengths are set for sizing boost.
    """
    # Always compute independent ETH/SOL multi-TF (used by regime-2).
    eth_sig, eth_sig_str = _compute_pair_multi_tf(eth_closes)
    sol_sig, sol_sig_str = _compute_pair_multi_tf(sol_closes)

    def _no_btc_result(skip_reason: str) -> MomentumGateResult:
        return MomentumGateResult(
            signal=None, tier=None, btc_agrees=False, btc_disagrees=False,
            skip_reason=skip_reason,
            eth_signal=eth_sig, eth_signal_strength=eth_sig_str,
            sol_signal=sol_sig, sol_signal_strength=sol_sig_str,
        )

    if btc_closes is None:
        return _no_btc_result("gate_no_btc_klines")

    returns = []
    for lb in _MTF_LOOKBACKS:
        r = _get_return(btc_closes, lb)
        if r is None:
            return _no_btc_result("gate_no_signal")
        returns.append(r)

    # All must agree in direction
    if not (all(r > 0 for r in returns) or all(r < 0 for r in returns)):
        return _no_btc_result("gate_no_signal")

    min_abs = min(abs(r) for r in returns)
    if min_abs < _MTF_THRESH:
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
        btc_agrees=True, btc_disagrees=False,
        skip_reason=None,
        signal_strength=min_abs,
        eth_confirmation_strength=eth_confirm,
        sol_confirmation_strength=sol_confirm,
        eth_signal=eth_sig, eth_signal_strength=eth_sig_str,
        sol_signal=sol_sig, sol_signal_strength=sol_sig_str,
    )
