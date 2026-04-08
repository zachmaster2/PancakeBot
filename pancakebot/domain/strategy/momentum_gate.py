"""OKX dual-asset momentum gate.

Signal architecture (validated +10.2 BNB / 5000 rounds):

  Tier 1 — BNB Acceleration
    Two BNB 1s-return lookback pairs agree on direction and
    max(|ret|) >= 0.0002.  Pairs tried in order: (7,10), (5,10), (5,7).

  Tier 2 — BNB + BTC Confirmation
    Any nonzero BNB 7s return confirmed by BTC 30s return in the same
    direction with |BTC ret| >= 0.0003.

Both tiers use OKX public 1s candles (no auth required).
"""

from __future__ import annotations

from dataclasses import dataclass

from pancakebot.infra.okx_client import OkxClient
from pancakebot.core.errors import InvariantError
from pancakebot.core.logging import warn

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
    max_staleness_seconds: int


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
        if int(config.max_staleness_seconds) <= 0:
            raise InvariantError("momentum_gate_max_staleness_must_be_positive")
        self._cfg = config
        self._client = okx_client

    @property
    def enabled(self) -> bool:
        return bool(self._cfg.enabled)

    def evaluate(self, *, cutoff_ts_ms: int) -> MomentumGateResult:
        """Fetch BNB + BTC 1s klines and compute signal at cutoff time."""
        if not bool(self._cfg.enabled):
            return MomentumGateResult(
                signal=None, tier=None, btc_agrees=False, btc_disagrees=False,
                skip_reason=None, kline_age_seconds=None,
            )

        # Fetch BNB 1s klines (need up to 10s lookback + buffer)
        bnb_klines = self._fetch_klines(str(self._cfg.symbol), count=15)
        if bnb_klines is None or len(bnb_klines) < 12:
            return self._skip("gate_bnb_fetch_failed")

        # Check staleness
        newest_ts_ms = int(bnb_klines[-1]["open_time_ms"])
        age_seconds = float((int(cutoff_ts_ms) - newest_ts_ms) / 1000)
        if age_seconds > float(self._cfg.max_staleness_seconds):
            return self._skip(f"gate_stale_kline:age={age_seconds:.1f}s")

        # Fetch BTC 1s klines (need 30s lookback + buffer)
        btc_klines = self._fetch_klines(str(self._cfg.btc_symbol), count=35)

        # Compute signal
        return _compute_signal(bnb_klines, btc_klines, cutoff_ts_ms, age_seconds)

    def _fetch_klines(self, symbol: str, count: int) -> list[dict] | None:
        try:
            return self._client.fetch_1s_klines(symbol=symbol, count=count)
        except Exception as e:
            warn("GATE", "OKX", "FETCH_FAIL", symbol=symbol, reason=str(e))
            return None

    @staticmethod
    def _skip(reason: str) -> MomentumGateResult:
        return MomentumGateResult(
            signal=None, tier=None, btc_agrees=False, btc_disagrees=False,
            skip_reason=reason, kline_age_seconds=None,
        )


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
    """Compute signal from raw kline arrays (backtest path)."""
    return _compute_signal(bnb_klines, btc_klines, cutoff_ms, age_seconds=0.0)


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
