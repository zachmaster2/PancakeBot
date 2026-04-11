"""OKX dual-asset momentum gate.

Signal architecture:

  Tier 1 — BNB Acceleration
    Two BNB 1s-return lookback pairs agree on direction and
    max(|ret|) >= 0.0002.  Pairs tried in order: (7,10), (5,10), (5,7).

  Tier 2 — BNB + BTC Confirmation
    Any nonzero BNB 7s return confirmed by BTC 30s return in the same
    direction with |BTC ret| >= 0.0003.

Both tiers use OKX public 1s candles (no auth required).
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

from pancakebot.infra.okx_client import OkxClient
from pancakebot.core.errors import InvariantError
from pancakebot.core.logging import info, warn

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
    kline_age_seconds: float | None


class MomentumGate:
    """Stateless dual-asset momentum gate: fetches BNB + BTC 1s klines."""

    def __init__(self, *, config: MomentumGateConfig, okx_client: OkxClient) -> None:
        self._cfg = config
        self._client = okx_client

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
        bnb_fut = pool.submit(self._fetch_klines, self._cfg.symbol, _LIVE_FETCH_COUNT, cutoff_ts_ms)
        btc_fut = pool.submit(self._fetch_klines, self._cfg.btc_symbol, _LIVE_FETCH_COUNT, cutoff_ts_ms)
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

        All klines are guaranteed completed (open_time < cutoff_ts_ms)
        because the OKX ``after`` parameter excludes the in-progress bar
        at fetch time.  The anchor is simply the newest kline.
        """
        if not self._cfg.enabled:
            return MomentumGateResult(
                signal=None, tier=None, btc_agrees=False, btc_disagrees=False,
                skip_reason=None, kline_age_seconds=None,
            )

        # Collect klines — from async fetch or inline fallback
        if kline_futures is not None:
            bnb_klines = kline_futures[0].result()
            btc_klines = kline_futures[1].result()
        else:
            with ThreadPoolExecutor(max_workers=2) as pool:
                bnb_fut = pool.submit(self._fetch_klines, self._cfg.symbol, _LIVE_FETCH_COUNT, cutoff_ts_ms)
                btc_fut = pool.submit(self._fetch_klines, self._cfg.btc_symbol, _LIVE_FETCH_COUNT, cutoff_ts_ms)
                bnb_klines = bnb_fut.result()
                btc_klines = btc_fut.result()

        if bnb_klines is None or len(bnb_klines) < 12:
            return self._skip("gate_bnb_fetch_failed")

        # All klines have open_time < cutoff (the after= parameter
        # excluded the in-progress bar at fetch time).  The anchor is
        # the newest candle, which should be at cutoff - 1 (covering
        # the interval [cutoff-1, cutoff)).
        anchor_ts_ms = int(bnb_klines[-1]["open_time_ms"])

        # Staleness guard: the anchor must be at cutoff - 1.  Anything
        # older means we're missing recent data and the signal cannot
        # be trusted.
        if anchor_ts_ms < cutoff_ts_ms - 1000:
            age_s = (cutoff_ts_ms - anchor_ts_ms) / 1000
            return self._skip(f"gate_stale_kline:age={age_s:.0f}s")

        age_seconds = (cutoff_ts_ms - anchor_ts_ms) / 1000

        result = _compute_signal(bnb_klines, btc_klines, anchor_ts_ms, age_seconds)

        # Diagnostic: log why signal did/didn't fire so dry-run progress is visible
        if result.signal is not None:
            info("GATE", "SIGNAL", "FIRE", tier=result.tier, side=result.signal,
                 btc_ag=result.btc_agrees, age=f"{age_seconds:.1f}s")
        else:
            _log_no_signal(bnb_klines, btc_klines, anchor_ts_ms, age_seconds)

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
            skip_reason=reason, kline_age_seconds=None,
        )


def _log_no_signal(
    bnb_klines: list[dict],
    btc_klines: list[dict] | None,
    cutoff_ms: int,
    age_seconds: float,
) -> None:
    """Log diagnostic when signal doesn't fire — shows why."""
    # BNB price and unique count
    prices = {k["close_price"] for k in bnb_klines} if bnb_klines else set()
    bnb_price = bnb_klines[-1]["close_price"] if bnb_klines else 0.0

    # Best BNB return across all lookback pairs
    max_ret = 0.0
    for short, long in _ACCEL_PAIRS:
        for lb in (short, long):
            r = _get_return(bnb_klines, cutoff_ms, lb)
            if r is not None:
                max_ret = max(max_ret, abs(r))

    # BTC 30s return
    btc_ret = _get_return(btc_klines, cutoff_ms, _BTC_LOOKBACK) if btc_klines else None
    btc_str = f"{abs(btc_ret):.6f}" if btc_ret is not None else "N/A"

    info("GATE", "DIAG", "NO_SIGNAL",
         bnb=f"${bnb_price:.1f}",
         max_ret=f"{max_ret:.7f}",
         thresh=f"{_ACCEL_THRESH}",
         uniq=len(prices),
         btc30=btc_str,
         age=f"{age_seconds:.1f}s")


def _find_closest_price(klines: list, target_ms: int) -> float | None:
    """Find close price of kline closest to target_ms (within 2s)."""
    best_price = None
    best_dist = float("inf")
    for k in klines:
        ts = int(k[0]) if isinstance(k, list) else int(k["open_time_ms"])
        dist = abs(ts - target_ms)
        if dist < best_dist:
            best_dist = dist
            best_price = float(k[4]) if isinstance(k, list) else float(k["close_price"])
    if best_dist <= 2000:
        return best_price
    return None


def _get_return(klines: list, cutoff_ms: int, lookback_s: int) -> float | None:
    """Compute return between cutoff and cutoff - lookback_s."""
    now = _find_closest_price(klines, cutoff_ms)
    ago = _find_closest_price(klines, cutoff_ms - lookback_s * 1000)
    if now is None or ago is None or ago <= 0:
        return None
    return (now / ago) - 1.0


def compute_signal_from_klines(
    bnb_klines: list[list],
    btc_klines: list[list] | None,
    cutoff_ms: int,
) -> MomentumGateResult:
    """Compute signal from raw kline arrays (backtest path).

    Simulates the live environment by:
      1. Trimming klines to only completed candles before *cutoff_ms*
         (the candle AT cutoff_ms would be in-progress during a live
         fetch, so it's excluded — matching ``evaluate()``'s drop).
      2. Keeping only the last 20 BNB / 40 BTC candles to match the
         live fetch window.
      3. Anchoring on the newest remaining BNB kline.
    """
    # Trim to match live fetch window (completed candles only, strict <)
    bnb_klines = _trim_to_live_window(bnb_klines, cutoff_ms, _USABLE_CANDLE_COUNT)
    if btc_klines is not None:
        btc_klines = _trim_to_live_window(btc_klines, cutoff_ms, _USABLE_CANDLE_COUNT)

    if not bnb_klines:
        return MomentumGateResult(
            signal=None, tier=None, btc_agrees=False, btc_disagrees=False,
            skip_reason="gate_no_spot_klines", kline_age_seconds=None,
        )

    # Anchor on newest completed BNB kline (mirrors live anchor logic)
    anchor_ms = int(bnb_klines[-1][0]) if isinstance(bnb_klines[-1], list) else int(bnb_klines[-1]["open_time_ms"])

    # Same guard as live path: anchor must be at cutoff - 1
    if anchor_ms < cutoff_ms - 1000:
        age_s = (cutoff_ms - anchor_ms) / 1000
        return MomentumGateResult(
            signal=None, tier=None, btc_agrees=False, btc_disagrees=False,
            skip_reason=f"gate_stale_kline:age={age_s:.0f}s", kline_age_seconds=age_s,
        )

    return _compute_signal(bnb_klines, btc_klines, anchor_ms, age_seconds=0.0)


# Candle window — 40 completed candles for both BNB and BTC.
# Live path uses OKX after= parameter to exclude the in-progress bar,
# so all 40 fetched candles are completed with final close prices.
# Backtest path trims with strict < to match.
_LIVE_FETCH_COUNT = 40
_USABLE_CANDLE_COUNT = 40


def _trim_to_live_window(klines: list[list], cutoff_ms: int, count: int) -> list[list]:
    """Keep only the *count* completed candles before cutoff_ms.

    Uses strict ``<`` because the candle at *cutoff_ms* would still be
    in-progress when fetched live — its close price is an ephemeral
    mid-second snapshot.  The live path drops the in-progress bar
    (see ``evaluate()``), so the backtest must exclude it too.
    """
    before = [k for k in klines if int(k[0]) < cutoff_ms]
    return before[-count:] if len(before) > count else before


def _compute_signal(
    bnb_klines: list,
    btc_klines: list | None,
    cutoff_ms: int,
    age_seconds: float,
) -> MomentumGateResult:
    """Core signal logic shared by live and backtest paths."""

    # --- Tier 1: BNB Acceleration ---
    for short, long in _ACCEL_PAIRS:
        rs = _get_return(bnb_klines, cutoff_ms, short)
        rl = _get_return(bnb_klines, cutoff_ms, long)
        if rs and rl and rs != 0 and rl != 0 and (rs > 0) == (rl > 0):
            if max(abs(rs), abs(rl)) >= _ACCEL_THRESH:
                d = "Bull" if rs > 0 else "Bear"
                btc_ag, btc_dis = _check_btc(btc_klines, cutoff_ms, d)
                return MomentumGateResult(
                    signal=d, tier="accel",
                    btc_agrees=btc_ag, btc_disagrees=btc_dis,
                    skip_reason=None, kline_age_seconds=age_seconds,
                )

    # --- Tier 2: BNB Any Move + BTC Confirmation ---
    if btc_klines is not None:
        bnb_r = _get_return(bnb_klines, cutoff_ms, 7)
        if bnb_r is not None and bnb_r != 0:
            btc_r = _get_return(btc_klines, cutoff_ms, _BTC_LOOKBACK)
            if btc_r is not None and abs(btc_r) >= _BTC_THRESH:
                bnb_dir = "Bull" if bnb_r > 0 else "Bear"
                btc_dir = "Bull" if btc_r > 0 else "Bear"
                if bnb_dir == btc_dir:
                    return MomentumGateResult(
                        signal=bnb_dir, tier="any+btc",
                        btc_agrees=True, btc_disagrees=False,
                        skip_reason=None, kline_age_seconds=age_seconds,
                    )

    return MomentumGateResult(
        signal=None, tier=None, btc_agrees=False, btc_disagrees=False,
        skip_reason="gate_no_signal", kline_age_seconds=age_seconds,
    )


def _check_btc(
    btc_klines: list | None, cutoff_ms: int, bnb_dir: str,
) -> tuple[bool, bool]:
    """Check if BTC 30s return agrees/disagrees with BNB direction."""
    if btc_klines is None:
        return False, False
    btc_r = _get_return(btc_klines, cutoff_ms, _BTC_LOOKBACK)
    if btc_r is None or abs(btc_r) < _BTC_THRESH:
        return False, False
    btc_dir = "Bull" if btc_r > 0 else "Bear"
    return (btc_dir == bnb_dir, btc_dir != bnb_dir)
