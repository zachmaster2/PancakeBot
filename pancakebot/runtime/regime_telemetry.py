"""Rolling-window regime-divergence monitors for critical-path constants.

Several timing constants (``OKX_KLINE_FETCH_RTT_P99_MS``,
``BSC_BLOCK_TIME_MS``, the anchor-poll budget) were tuned against a
point-in-time measurement regime. If the live distribution drifts away
from the tuned value, the bot keeps using a stale threshold with no
signal. These monitors accumulate live observations in a bounded window
and return an alert string when the live distribution has diverged past
a tolerance.

Observability ONLY: a monitor never changes a decision. Callers log the
returned message at ``WARN ALERT`` and otherwise ignore it. Every method
is pure given its inputs and total-ordered — no clock reads — so the
monitors are deterministic and unit-testable.

All monitors are **edge-triggered**: they return an alert string on the
transition from in-tolerance to out-of-tolerance (and a one-shot
recovery notice on the way back), NOT on every observation while
breached. This keeps a sustained regime shift from emitting an alert
every round.
"""
from __future__ import annotations

from collections import deque


def _nearest_rank(sorted_vals: list[float], quantile: float) -> float:
    """Nearest-rank percentile of a non-empty, ascending-sorted list.

    quantile in [0, 1]. Matches the convention used when the timing
    constants were measured (rank = ceil(q * n), 1-indexed).
    """
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    # 1-indexed nearest-rank; clamp to [1, n].
    rank = max(1, min(n, int(-(-quantile * n // 1))))  # ceil(q*n)
    return sorted_vals[rank - 1]


class RollingP99Monitor:
    """Tracks a rolling high-quantile of a latency series vs a constant.

    Alerts (edge-triggered) when the windowed quantile exceeds
    ``constant_ms + tolerance_ms``. Used for OKX kline fetch RTT vs
    ``OKX_KLINE_FETCH_RTT_P99_MS`` and similar walk-back inputs whose
    stale-LOW drift directly causes missed lock-block inclusions.
    """

    def __init__(
        self,
        *,
        name: str,
        constant_ms: float,
        tolerance_ms: float,
        window: int,
        min_samples: int,
        quantile: float = 0.99,
    ) -> None:
        self.name = name
        self.constant_ms = float(constant_ms)
        self.tolerance_ms = float(tolerance_ms)
        self.quantile = quantile
        self.min_samples = min_samples
        self._window: deque[float] = deque(maxlen=window)
        self._breached = False

    def observe(self, value_ms: float) -> str | None:
        self._window.append(float(value_ms))
        if len(self._window) < self.min_samples:
            return None
        observed = _nearest_rank(sorted(self._window), self.quantile)
        threshold = self.constant_ms + self.tolerance_ms
        if observed > threshold and not self._breached:
            self._breached = True
            return (
                f"REGIME_DRIFT monitor={self.name} "
                f"observed_p{int(self.quantile * 100)}={observed:.0f}ms "
                f"constant={self.constant_ms:.0f}ms tolerance={self.tolerance_ms:.0f}ms "
                f"n={len(self._window)} "
                f"reason=live_latency_exceeds_constant"
            )
        if observed <= self.constant_ms and self._breached:
            self._breached = False
            return (
                f"REGIME_DRIFT_RECOVERED monitor={self.name} "
                f"observed_p{int(self.quantile * 100)}={observed:.0f}ms "
                f"constant={self.constant_ms:.0f}ms n={len(self._window)}"
            )
        return None


class RollingRateMonitor:
    """Tracks a rolling fraction of boolean events vs a max-rate threshold.

    Alerts (edge-triggered) when the windowed True-rate exceeds
    ``max_rate``. Used for the anchor-poll static-fallback rate: a single
    fallback is fine, a sustained majority means the dynamic-wake
    optimization is silently inert.
    """

    def __init__(
        self,
        *,
        name: str,
        max_rate: float,
        window: int,
        min_samples: int,
    ) -> None:
        self.name = name
        self.max_rate = float(max_rate)
        self.min_samples = min_samples
        self._window: deque[bool] = deque(maxlen=window)
        self._breached = False

    def observe(self, hit: bool, *, detail: str = "") -> str | None:
        self._window.append(bool(hit))
        if len(self._window) < self.min_samples:
            return None
        rate = sum(1 for h in self._window if h) / len(self._window)
        if rate > self.max_rate and not self._breached:
            self._breached = True
            suffix = f" {detail}" if detail else ""
            return (
                f"REGIME_DRIFT monitor={self.name} "
                f"rate={rate:.2f} max_rate={self.max_rate:.2f} "
                f"n={len(self._window)} reason=event_rate_exceeds_threshold{suffix}"
            )
        if rate <= self.max_rate and self._breached:
            self._breached = False
            return (
                f"REGIME_DRIFT_RECOVERED monitor={self.name} "
                f"rate={rate:.2f} max_rate={self.max_rate:.2f} n={len(self._window)}"
            )
        return None


class RollingMedianDriftMonitor:
    """Tracks the rolling median of a series vs an expected constant.

    Alerts (edge-triggered) when ``abs(median - expected) > tolerance``.
    Used to confirm the observed block time still clusters at
    ``BSC_BLOCK_TIME_MS`` — the load-bearing assumption behind the
    no-margin predecessor extrapolation.
    """

    def __init__(
        self,
        *,
        name: str,
        expected: float,
        tolerance: float,
        window: int,
        min_samples: int,
    ) -> None:
        self.name = name
        self.expected = float(expected)
        self.tolerance = float(tolerance)
        self.min_samples = min_samples
        self._window: deque[float] = deque(maxlen=window)
        self._breached = False

    @staticmethod
    def _median(vals: list[float]) -> float:
        s = sorted(vals)
        n = len(s)
        mid = n // 2
        if n % 2 == 1:
            return s[mid]
        return (s[mid - 1] + s[mid]) / 2.0

    def observe(self, value: float) -> str | None:
        self._window.append(float(value))
        if len(self._window) < self.min_samples:
            return None
        median = self._median(list(self._window))
        drift = median - self.expected
        if abs(drift) > self.tolerance and not self._breached:
            self._breached = True
            return (
                f"REGIME_DRIFT monitor={self.name} "
                f"observed_median={median:.0f} expected={self.expected:.0f} "
                f"drift={drift:+.0f} tolerance={self.tolerance:.0f} "
                f"n={len(self._window)} reason=median_diverged_from_constant"
            )
        if abs(drift) <= self.tolerance and self._breached:
            self._breached = False
            return (
                f"REGIME_DRIFT_RECOVERED monitor={self.name} "
                f"observed_median={median:.0f} expected={self.expected:.0f} n={len(self._window)}"
            )
        return None
