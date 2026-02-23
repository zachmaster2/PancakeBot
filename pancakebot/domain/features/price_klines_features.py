from __future__ import annotations

import math

from pancakebot.domain.types import Kline
from pancakebot.core.errors import InvariantError


def compute_price_klines_features(
    *,
    context_klines: list[Kline],
) -> dict[str, float]:
    """Compute external price features from 1m klines.

    All inputs MUST be fully closed klines with strictly increasing open_time_ms.
    """

    if not context_klines:
        raise InvariantError("context_klines_empty")

    # Basic ordering invariant.
    for i in range(1, len(context_klines)):
        if int(context_klines[i].open_time_ms) <= int(context_klines[i - 1].open_time_ms):
            raise InvariantError("context_klines_not_strictly_increasing")

    feats: dict[str, float] = {}

    for n in (15, 30, 60, 120):
        feats.update(_window_feats(context_klines=context_klines, n=int(n)))

    return feats


def _window_feats(*, context_klines: list[Kline], n: int) -> dict[str, float]:
    # For close-to-close returns over n minutes, we need n+1 closes.
    need = int(n) + 1
    if len(context_klines) < need:
        raise InvariantError("context_klines_insufficient_for_window")

    window = context_klines[-need:]

    closes = [float(k.close_price) for k in window]
    last_n = window[-n:]
    vols = [float(k.volume) for k in last_n]
    trades = [int(k.number_of_trades) for k in last_n]
    ranges = [float(k.high_price) - float(k.low_price) for k in last_n]

    rets: list[float] = []
    for i in range(1, len(closes)):
        prev = float(closes[i - 1])
        cur = float(closes[i])
        if prev <= 0.0 or cur <= 0.0:
            raise InvariantError("kline_close_non_positive")
        rets.append(math.log(cur / prev))

    if len(rets) != int(n):
        raise InvariantError("kline_returns_len_mismatch")

    log_ret_mean = _mean(rets)
    log_ret_std = _std(rets, log_ret_mean)
    abs_rets = [abs(float(x)) for x in rets]
    abs_ret_mean = _mean(abs_rets)
    abs_ret_max = _max(abs_rets)

    range_mean = _mean(ranges)
    range_max = _max(ranges)

    vol_mean = _mean(vols)
    vol_std = _std(vols, vol_mean)
    vol_max = _max(vols)

    trade_mean = _mean([float(x) for x in trades])
    trade_std = _std([float(x) for x in trades], trade_mean)
    trade_max = _max([float(x) for x in trades])

    return {
        f"price_log_return_mean_k_{n}": float(log_ret_mean),
        f"price_log_return_std_k_{n}": float(log_ret_std),
        f"price_log_return_abs_mean_k_{n}": float(abs_ret_mean),
        f"price_log_return_abs_max_k_{n}": float(abs_ret_max),
        f"price_range_mean_k_{n}": float(range_mean),
        f"price_range_max_k_{n}": float(range_max),
        f"price_volume_mean_k_{n}": float(vol_mean),
        f"price_volume_std_k_{n}": float(vol_std),
        f"price_volume_max_k_{n}": float(vol_max),
        f"price_trade_count_mean_k_{n}": float(trade_mean),
        f"price_trade_count_std_k_{n}": float(trade_std),
        f"price_trade_count_max_k_{n}": float(trade_max),
    }


def _mean(xs: list[float]) -> float:
    if not xs:
        raise InvariantError("mean_empty")
    return float(sum(xs)) / float(len(xs))


def _std(xs: list[float], mean: float) -> float:
    if len(xs) < 2:
        return 0.0
    var = 0.0
    for x in xs:
        d = float(x) - float(mean)
        var += float(d) * float(d)
    var = float(var) / float(len(xs) - 1)
    return math.sqrt(float(var))


def _max(xs: list[float]) -> float:
    if not xs:
        raise InvariantError("max_empty")
    return float(max(xs))
