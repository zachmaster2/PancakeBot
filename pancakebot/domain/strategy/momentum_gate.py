"""OKX dual-asset momentum gate.

Signal architecture:

  BNB Acceleration
    Two BNB 1s-return lookback pairs agree on direction and
    max(|ret|) >= 0.0002.  Pairs tried in order: (7,10), (5,10), (5,7).

  BTC is used for confirmation/sizing only (btc_agrees / btc_disagrees),
  not as a standalone signal trigger.

Uses OKX public 1s candles (no auth required).

Kline window: 40 contiguous 1s candles covering [lockAt-44, lockAt-4).
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
_CANDLE_COUNT = 40

# Tuned constants — these were optimized together; do not change independently.
_ACCEL_PAIRS: list[tuple[int, int]] = [(7, 10), (5, 10), (5, 7)]
_ACCEL_THRESH = 0.0002
_BTC_LOOKBACK = 30
_BTC_THRESH = 0.0003


@dataclass(frozen=True, slots=True)
class MomentumGateConfig:
    enabled: bool
    symbol: str              # "BNB-USDT"
    btc_symbol: str          # "BTC-USDT"


@dataclass(frozen=True, slots=True)
class MomentumGateResult:
    signal: str | None       # "Bull", "Bear", or None
    tier: str | None         # "accel" or "any+btc"
    btc_agrees: bool
    btc_disagrees: bool
    skip_reason: str | None


class MomentumGate:
    """Stateless dual-asset momentum gate: fetches BNB + BTC 1s klines."""

    def __init__(self, *, config: MomentumGateConfig, okx_client: OkxClient) -> None:
        self._cfg = config
        self._client = okx_client
        # Cached after each evaluate() so the pipeline can use BTC/BNB data
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

    def fetch_klines_async(self, *, cutoff_ts_ms: int) -> tuple[Future, Future] | None:
        """Kick off BNB + BTC kline fetches in parallel background threads.

        Call this immediately after waking from sleep, *before* the
        RPC calls (epoch handshake, lock_ts, round_data).  Returns a
        pair of Futures that evaluate() will collect, or None when the
        gate is disabled.

        *cutoff_ts_ms* is passed to OKX as the ``after`` parameter so
        only completed candles (open_time < cutoff) are returned —
        the in-progress bar is never fetched.
        """
        if not self._cfg.enabled:
            return None
        pool = ThreadPoolExecutor(max_workers=2)
        bnb_fut = pool.submit(self._fetch_klines, self._cfg.symbol, _CANDLE_COUNT, cutoff_ts_ms)
        btc_fut = pool.submit(self._fetch_klines, self._cfg.btc_symbol, _CANDLE_COUNT, cutoff_ts_ms)
        pool.shutdown(wait=False)   # let threads finish on their own
        return bnb_fut, btc_fut

    def evaluate(
        self,
        *,
        cutoff_ts_ms: int,
        kline_futures: tuple[Future, Future] | None = None,
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
        else:
            with ThreadPoolExecutor(max_workers=2) as pool:
                bnb_fut = pool.submit(self._fetch_klines, self._cfg.symbol, _CANDLE_COUNT, cutoff_ts_ms)
                btc_fut = pool.submit(self._fetch_klines, self._cfg.btc_symbol, _CANDLE_COUNT, cutoff_ts_ms)
                bnb_klines = bnb_fut.result()
                btc_klines = btc_fut.result()

        # Validate: we expect exactly _CANDLE_COUNT contiguous 1s candles
        # ending at cutoff - 1.  Reject if the response is incomplete.
        bnb_reason = _validate_klines(bnb_klines, cutoff_ts_ms, "bnb")
        if bnb_reason is not None:
            return self._skip(bnb_reason)

        bnb_closes = [k["close_price"] for k in bnb_klines]
        btc_closes = [k["close_price"] for k in btc_klines] if btc_klines and len(btc_klines) >= _CANDLE_COUNT else None

        # Cache closes for pipeline auxiliary signals and regime-adaptive sizing.
        self.last_bnb_closes = bnb_closes
        self.last_btc_closes = btc_closes

        result = _compute_signal(bnb_closes, btc_closes)

        # Diagnostic: log why signal did/didn't fire so dry-run progress is visible
        if result.signal is not None:
            info("GATE", "SIGNAL", "FIRE", tier=result.tier, side=result.signal,
                 btc_ag=result.btc_agrees)
        else:
            _log_no_signal(bnb_closes, btc_closes)

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
) -> None:
    """Log diagnostic when signal doesn't fire — shows why."""
    bnb_price = bnb_closes[-1] if bnb_closes else 0.0
    uniq = len(set(bnb_closes)) if bnb_closes else 0

    max_ret = 0.0
    for short, long in _ACCEL_PAIRS:
        for lb in (short, long):
            r = _get_return(bnb_closes, lb)
            if r is not None:
                max_ret = max(max_ret, abs(r))

    btc_ret = _get_return(btc_closes, _BTC_LOOKBACK) if btc_closes else None
    btc_str = f"{abs(btc_ret):.6f}" if btc_ret is not None else "N/A"

    info("GATE", "DIAG", "NO_SIGNAL",
         bnb=f"${bnb_price:.1f}",
         max_ret=f"{max_ret:.7f}",
         thresh=f"{_ACCEL_THRESH}",
         uniq=uniq,
         btc30=btc_str)


def compute_signal_from_klines(
    bnb_klines: list[list],
    btc_klines: list[list] | None,
    cutoff_ms: int,
) -> MomentumGateResult:
    """Compute signal from raw kline arrays (backtest path).

    Trims klines to the same window the live path fetches, validates
    the result, then computes the signal using direct indexing.
    """
    bnb_klines = _trim_to_window(bnb_klines, cutoff_ms)
    if btc_klines is not None:
        btc_klines = _trim_to_window(btc_klines, cutoff_ms)

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

    return _compute_signal(bnb_closes, btc_closes)


def _trim_to_window(klines: list[list], cutoff_ms: int) -> list[list]:
    """Keep only the *_CANDLE_COUNT* completed candles before cutoff_ms."""
    before = [k for k in klines if int(k[0]) < cutoff_ms]
    return before[-_CANDLE_COUNT:] if len(before) > _CANDLE_COUNT else before


def _compute_signal(
    bnb_closes: list[float],
    btc_closes: list[float] | None,
) -> MomentumGateResult:
    """Core signal logic shared by live and backtest paths.

    Takes pre-extracted close price arrays.  Lookbacks use direct
    indexing: closes[-(N+1)] is the price N seconds before cutoff.
    """

    # --- Tier 1: BNB Acceleration ---
    for short, long in _ACCEL_PAIRS:
        rs = _get_return(bnb_closes, short)
        rl = _get_return(bnb_closes, long)
        if rs and rl and rs != 0 and rl != 0 and (rs > 0) == (rl > 0):
            if max(abs(rs), abs(rl)) >= _ACCEL_THRESH:
                d = "Bull" if rs > 0 else "Bear"
                btc_ag, btc_dis = _check_btc(btc_closes, d)
                return MomentumGateResult(
                    signal=d, tier="accel",
                    btc_agrees=btc_ag, btc_disagrees=btc_dis,
                    skip_reason=None,
                )

    return MomentumGateResult(
        signal=None, tier=None, btc_agrees=False, btc_disagrees=False,
        skip_reason="gate_no_signal",
    )


def _check_btc(
    btc_closes: list[float] | None, bnb_dir: str,
) -> tuple[bool, bool]:
    """Check if BTC 30s return agrees/disagrees with BNB direction."""
    if btc_closes is None:
        return False, False
    btc_r = _get_return(btc_closes, _BTC_LOOKBACK)
    if btc_r is None or abs(btc_r) < _BTC_THRESH:
        return False, False
    btc_dir = "Bull" if btc_r > 0 else "Bear"
    return (btc_dir == bnb_dir, btc_dir != bnb_dir)
