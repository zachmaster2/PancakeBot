"""OKX multi-timeframe BTC momentum gate.

Signal architecture:

  Multi-Timeframe BTC Agreement
    BTC 3s, 7s, and 15s returns must ALL agree in direction AND
    min(|r3|, |r7|, |r15|) >= threshold (default 0.0002).

    The signal strength (min_abs_return) is exposed for adaptive
    bet sizing in the pipeline: stronger moves = larger bets.

  BNB klines are still fetched (required by sync/backtest infrastructure)
  but are NOT used for signal generation.

Uses OKX public 1s candles (no auth required).

Kline window: 31 contiguous 1s candles covering [lockAt-32, lockAt-2).
The newest candle (index -1) has open_time = cutoff - 1 and its
close_price is the price at cutoff.  Lookbacks use direct indexing:
the price N seconds before cutoff is ``closes[-(N+1)]``.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

from pancakebot.infra.okx_client import OkxClient
from pancakebot.core.logging import info, warn

# Number of 1s candles fetched and used by all modes (live, sync, backtest).
_CANDLE_COUNT = 31

# Multi-TF BTC lookbacks — all must agree in direction.
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


class MomentumGate:
    """Multi-asset momentum gate: fetches BNB + BTC + ETH + SOL 1s klines."""

    def __init__(self, *, config: MomentumGateConfig, okx_client: OkxClient) -> None:
        self._cfg = config
        self._client = okx_client
        # Cached after each evaluate() so the pipeline can use data
        # for auxiliary signals and regime-adaptive sizing without re-fetching.
        self.last_btc_closes: list[float] | None = None
        self.last_bnb_closes: list[float] | None = None

    @property
    def enabled(self) -> bool:
        return self._cfg.enabled

    # ------------------------------------------------------------------
    # Async fetch / evaluate split
    # ------------------------------------------------------------------
    # The timing budget between wakeup and the lock safety guard is
    # only ~1 s (cutoff_seconds − safety_margin = 4 − 3).  Several RPC
    # calls (epoch handshake, lock_ts, round_data) consume ~450 ms of
    # that budget *before* the gate is even called.
    #
    # To avoid exceeding the budget, the runtime loop calls
    # fetch_klines_async() immediately after waking — the two OKX HTTP
    # requests run in background threads while the RPC work proceeds.
    # By the time evaluate() is called the data is already waiting.
    # ------------------------------------------------------------------

    def fetch_klines_async(self, *, cutoff_ts_ms: int) -> tuple | None:
        """Kick off BNB + BTC + ETH + SOL kline fetches in parallel.

        Call this immediately after waking from sleep, *before* the
        RPC calls (epoch handshake, lock_ts, round_data).  Returns a
        4-tuple of Futures that evaluate() will collect, or None when
        the gate is disabled.

        *cutoff_ts_ms* is passed to OKX as the ``after`` parameter so
        only completed candles (open_time < cutoff) are returned —
        the in-progress bar is never fetched.
        """
        if not self._cfg.enabled:
            return None
        pool = ThreadPoolExecutor(max_workers=4)
        bnb_fut = pool.submit(self._fetch_klines, self._cfg.bnb_symbol, _CANDLE_COUNT, cutoff_ts_ms)
        btc_fut = pool.submit(self._fetch_klines, self._cfg.btc_symbol, _CANDLE_COUNT, cutoff_ts_ms)
        eth_fut = pool.submit(self._fetch_klines, self._cfg.eth_symbol, _CANDLE_COUNT, cutoff_ts_ms)
        sol_fut = pool.submit(self._fetch_klines, self._cfg.sol_symbol, _CANDLE_COUNT, cutoff_ts_ms)
        pool.shutdown(wait=False)   # let threads finish on their own
        return bnb_fut, btc_fut, eth_fut, sol_fut

    def evaluate(
        self,
        *,
        cutoff_ts_ms: int,
        kline_futures: tuple | None = None,
    ) -> MomentumGateResult:
        """Compute signal from current OKX data.

        If *kline_futures* is provided (from fetch_klines_async()),
        the already-completed futures are collected instantly.  Otherwise
        the klines are fetched inline (parallel) as a fallback.

        If BNB klines fail validation because OKX is exactly 1 second
        behind (stale candle), retries once after a short delay.
        """
        if not self._cfg.enabled:
            return MomentumGateResult(
                signal=None, tier=None, btc_agrees=False, btc_disagrees=False,
                skip_reason=None,
            )

        # Collect klines — from async fetch or inline fallback
        if kline_futures is not None:
            bnb_klines = kline_futures[0].result()
            btc_klines = kline_futures[1].result()
            eth_klines = kline_futures[2].result() if len(kline_futures) > 2 else None
            sol_klines = kline_futures[3].result() if len(kline_futures) > 3 else None
        else:
            with ThreadPoolExecutor(max_workers=4) as pool:
                bnb_fut = pool.submit(self._fetch_klines, self._cfg.bnb_symbol, _CANDLE_COUNT, cutoff_ts_ms)
                btc_fut = pool.submit(self._fetch_klines, self._cfg.btc_symbol, _CANDLE_COUNT, cutoff_ts_ms)
                eth_fut = pool.submit(self._fetch_klines, self._cfg.eth_symbol, _CANDLE_COUNT, cutoff_ts_ms)
                sol_fut = pool.submit(self._fetch_klines, self._cfg.sol_symbol, _CANDLE_COUNT, cutoff_ts_ms)
                bnb_klines = bnb_fut.result()
                btc_klines = btc_fut.result()
                eth_klines = eth_fut.result()
                sol_klines = sol_fut.result()

        # Validate: we expect exactly _CANDLE_COUNT contiguous 1s candles
        # ending at cutoff - 1.  Reject if the response is incomplete.
        bnb_reason = _validate_klines(bnb_klines, cutoff_ts_ms, "bnb")
        if bnb_reason is not None:
            return self._skip(bnb_reason)

        bnb_closes = [k["close_price"] for k in bnb_klines]
        btc_closes = [k["close_price"] for k in btc_klines] if btc_klines and len(btc_klines) >= _CANDLE_COUNT else None
        eth_closes = [k["close_price"] for k in eth_klines] if eth_klines and len(eth_klines) >= _CANDLE_COUNT else None
        sol_closes = [k["close_price"] for k in sol_klines] if sol_klines and len(sol_klines) >= _CANDLE_COUNT else None

        # Cache closes for pipeline auxiliary signals and regime-adaptive sizing.
        self.last_bnb_closes = bnb_closes
        self.last_btc_closes = btc_closes

        result = _compute_signal(bnb_closes, btc_closes, eth_closes, sol_closes)

        # Diagnostic: log signal details so dry-run progress is visible
        if result.signal is not None:
            info("GATE", "SIGNAL", "FIRE",
                 side=result.signal,
                 strength=f"{result.signal_strength:.6f}",
                 eth_confirm=f"{result.eth_confirmation_strength:.6f}",
                 sol_confirm=f"{result.sol_confirmation_strength:.6f}",
                 bnb_ok=bnb_klines is not None,
                 btc_ok=btc_closes is not None,
                 eth_ok=eth_closes is not None,
                 sol_ok=sol_closes is not None)
        else:
            _log_no_signal(bnb_closes, btc_closes, eth_closes, sol_closes)

        return result

    def _fetch_klines(self, symbol: str, count: int, after_ms: int | None = None) -> list[dict] | None:
        try:
            return self._client.fetch_1s_klines(symbol=symbol, count=count, after_ms=after_ms)
        except Exception as e:
            warn("GATE", "OKX", "FETCH_FAIL", symbol=symbol, reason=str(e))
            return None

    @staticmethod
    def _skip(reason: str) -> MomentumGateResult:
        return MomentumGateResult(
            signal=None, tier=None, btc_agrees=False, btc_disagrees=False,
            skip_reason=reason,
        )


def _validate_klines(
    klines: list[dict] | None, cutoff_ts_ms: int, label: str,
) -> str | None:
    """Verify we received the expected kline window.

    Returns a skip reason string on failure, or None if valid.
    """
    if klines is None or len(klines) < _CANDLE_COUNT:
        n = 0 if klines is None else len(klines)
        return f"gate_{label}_fetch_failed:got={n}"

    # Newest candle must be at cutoff - 1 (the last completed second)
    newest_ts = int(klines[-1]["open_time_ms"])
    expected_ts = cutoff_ts_ms - 1000
    if newest_ts != expected_ts:
        return f"gate_{label}_unexpected_newest:got={newest_ts},expected={expected_ts}"

    return None


def _validate_klines_raw(
    klines: list[list], cutoff_ms: int, label: str,
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


def _log_no_signal(
    bnb_closes: list[float],
    btc_closes: list[float] | None,
    eth_closes: list[float] | None = None,
    sol_closes: list[float] | None = None,
) -> None:
    """Log diagnostic when signal doesn't fire — shows why."""
    btc_rets = {}
    if btc_closes:
        for lb in _MTF_LOOKBACKS:
            r = _get_return(btc_closes, lb)
            btc_rets[lb] = f"{r:.7f}" if r is not None else "N/A"

    info("GATE", "DIAG", "NO_SIGNAL",
         btc_r3=btc_rets.get(3, "N/A"),
         btc_r7=btc_rets.get(7, "N/A"),
         btc_r15=btc_rets.get(15, "N/A"),
         thresh=f"{_MTF_THRESH}",
         bnb_ok=bnb_closes is not None and len(bnb_closes) > 0,
         btc_ok=btc_closes is not None,
         eth_ok=eth_closes is not None,
         sol_ok=sol_closes is not None)


def compute_signal_from_klines(
    bnb_klines: list[list],
    btc_klines: list[list] | None,
    cutoff_ms: int,
    eth_klines: list[list] | None = None,
    sol_klines: list[list] | None = None,
) -> MomentumGateResult:
    """Compute signal from raw kline arrays (backtest path).

    Trims klines to the same window the live path fetches, validates
    the result, then computes the signal using direct indexing.
    """
    bnb_klines = _trim_to_window(bnb_klines, cutoff_ms)
    if btc_klines is not None:
        btc_klines = _trim_to_window(btc_klines, cutoff_ms)
    if eth_klines is not None:
        eth_klines = _trim_to_window(eth_klines, cutoff_ms)
    if sol_klines is not None:
        sol_klines = _trim_to_window(sol_klines, cutoff_ms)

    bnb_reason = _validate_klines_raw(bnb_klines, cutoff_ms, "bnb")
    if bnb_reason is not None:
        return MomentumGateResult(
            signal=None, tier=None, btc_agrees=False, btc_disagrees=False,
            skip_reason=bnb_reason,
        )

    bnb_closes = [k[4] for k in bnb_klines]
    btc_closes = None
    if btc_klines and len(btc_klines) >= _CANDLE_COUNT:
        btc_closes = [k[4] for k in btc_klines]
    eth_closes = None
    if eth_klines and len(eth_klines) >= _CANDLE_COUNT:
        eth_closes = [k[4] for k in eth_klines]
    sol_closes = None
    if sol_klines and len(sol_klines) >= _CANDLE_COUNT:
        sol_closes = [k[4] for k in sol_klines]

    return _compute_signal(bnb_closes, btc_closes, eth_closes, sol_closes)


def _trim_to_window(klines: list[list], cutoff_ms: int) -> list[list]:
    """Keep only the *_CANDLE_COUNT* completed candles before cutoff_ms."""
    before = [k for k in klines if int(k[0]) < cutoff_ms]
    return before[-_CANDLE_COUNT:] if len(before) > _CANDLE_COUNT else before


def _compute_signal(
    bnb_closes: list[float],
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
    if btc_closes is None:
        return MomentumGateResult(
            signal=None, tier=None, btc_agrees=False, btc_disagrees=False,
            skip_reason="gate_no_btc_klines",
        )

    returns = []
    for lb in _MTF_LOOKBACKS:
        r = _get_return(btc_closes, lb)
        if r is None:
            return MomentumGateResult(
                signal=None, tier=None, btc_agrees=False, btc_disagrees=False,
                skip_reason="gate_no_signal",
            )
        returns.append(r)

    # All must agree in direction
    if not (all(r > 0 for r in returns) or all(r < 0 for r in returns)):
        return MomentumGateResult(
            signal=None, tier=None, btc_agrees=False, btc_disagrees=False,
            skip_reason="gate_no_signal",
        )

    min_abs = min(abs(r) for r in returns)
    if min_abs < _MTF_THRESH:
        return MomentumGateResult(
            signal=None, tier=None, btc_agrees=False, btc_disagrees=False,
            skip_reason="gate_no_signal",
        )

    direction = "Bull" if returns[0] > 0 else "Bear"

    # Check ETH/SOL confirmation for sizing boost
    btc_positive = returns[0] > 0

    eth_strength = 0.0
    if eth_closes is not None:
        eth_rets = [_get_return(eth_closes, lb) for lb in _MTF_LOOKBACKS]
        if all(r is not None for r in eth_rets):
            if all((r > 0) == btc_positive for r in eth_rets):
                eth_strength = min(abs(r) for r in eth_rets)

    sol_strength = 0.0
    if sol_closes is not None:
        sol_rets = [_get_return(sol_closes, lb) for lb in _MTF_LOOKBACKS]
        if all(r is not None for r in sol_rets):
            if all((r > 0) == btc_positive for r in sol_rets):
                sol_strength = min(abs(r) for r in sol_rets)

    return MomentumGateResult(
        signal=direction, tier="multi_tf",
        btc_agrees=True, btc_disagrees=False,
        skip_reason=None,
        signal_strength=min_abs,
        eth_confirmation_strength=eth_strength,
        sol_confirmation_strength=sol_strength,
    )
