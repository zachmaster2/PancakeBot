"""Consecutive pool-gate-blocked detector: the "enabled but not trading" alarm.

The engine asks ``RpcPoller.is_pool_ready`` once per round. A ``False``
answer means the bot could not trade that round no matter what the
strategy wanted — stale pool coverage (``pool_uncovered``), a predicted
catch-up shortfall (``catchup_infeasible_for_round``), or an unfinished
cold start. Any single blocked round is normal; a RUN of them means the
bot is up, enabled, and silently unable to place a bet.

Why a blocked-round streak and NOT a no-bet streak: at the live fire rate
(~0.3% of rounds) a no-bet run of several hundred rounds is ordinary
behaviour and carries no information. Readiness is the signal — the
counter therefore resets whenever ``is_pool_ready`` returns True, whether
or not the strategy gate then fired.

Emits at most three shapes of event, all advisory (the caller alerts and
CONTINUES; nothing here raises, skips a round, or touches systemd):
  * first crossing of ``threshold`` consecutive blocked rounds,
  * a re-alert every ``realert_interval_s`` while the run persists (so a
    multi-day outage costs ~one message an hour, not one per round),
  * a RECOVERED event on the first ready round after an alerting run, so
    silence is never ambiguous.

Pure state machine: no I/O, no clock reads (``now`` is injected), no
dependency on the poller or the notifier.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# 12 rounds/hour: 6 consecutive blocked rounds is ~30 minutes of not
# trading — long enough that transient endpoint blips have cleared, short
# enough to catch a sustained outage the same morning it starts.
DEFAULT_THRESHOLD = 6
DEFAULT_REALERT_INTERVAL_S = 3600.0

KIND_BLOCKED = "POOL_GATE_BLOCKED"
KIND_RECOVERED = "POOL_GATE_RECOVERED"
# Second instance: OKX served partial candles so the gate could not
# evaluate. Same machine, own threshold, own kinds.
KIND_KLINE_BLOCKED = "KLINE_GATE_BLOCKED"
KIND_KLINE_RECOVERED = "KLINE_GATE_RECOVERED"
# Third instance: GENUINE fetch failures (unreachable / HTTP error /
# middle-gap response). Kept at the original sensitivity, because
# excluding publish delays means this streak means what it always did.
KIND_FETCH_FAILING = "KLINE_FETCH_FAILING"
KIND_FETCH_RECOVERED = "KLINE_FETCH_RECOVERED"

# The reasons ``is_pool_ready`` returns when the pool path itself is the
# blocker. Listed for the operator-facing histogram; the counter itself
# deliberately advances on ANY not-ready reason, so a future reason added
# to the poller cannot open a silent hole in this alarm.
POOL_BLOCKED_REASONS = frozenset({
    "pool_uncovered",
    "catchup_infeasible_for_round",
    "cold_start_in_progress",
})


def format_duration(seconds: float) -> str:
    """Compact operator-facing duration: 45s / 30m / 2h05m / 4d19h."""
    s = max(0, int(seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m"
    if s < 86400:
        return f"{s // 3600}h{(s % 3600) // 60:02d}m"
    return f"{s // 86400}d{(s % 86400) // 3600:02d}h"


@dataclass(frozen=True)
class PoolGateEvent:
    """One advisory alert the caller should dispatch. ``fields`` is the
    notifier payload; ``detail`` is the single structured kv line."""
    kind: str
    detail: str
    fields: dict[str, Any] = field(default_factory=dict)


class PoolGateAlarm:
    def __init__(
        self,
        *,
        threshold: int = DEFAULT_THRESHOLD,
        realert_interval_s: float = DEFAULT_REALERT_INTERVAL_S,
        kind_blocked: str = KIND_BLOCKED,
        kind_recovered: str = KIND_RECOVERED,
    ) -> None:
        """``kind_*`` let a SECOND instance watch a different condition on
        its own threshold and its own notification kinds. The mechanism is
        deliberately not generalised beyond that: each condition wants an
        independently tunable threshold, and conflating them would hide one
        outage behind the other."""
        if threshold < 1:
            raise ValueError(f"pool_gate_alarm_threshold_invalid: {threshold}")
        self.threshold = int(threshold)
        self.realert_interval_s = float(realert_interval_s)
        self.kind_blocked = kind_blocked
        self.kind_recovered = kind_recovered
        self._streak = 0
        self._first_blocked_at: float | None = None
        self._last_alert_at: float | None = None
        self._alerting = False
        self._reason_counts: dict[str, int] = {}
        self._last_ok_epoch: int | None = None
        self._last_blocks_short: int | None = None
        self._last_getlogs_p99_ms: int | None = None

    @property
    def streak(self) -> int:
        return self._streak

    @property
    def alerting(self) -> bool:
        return self._alerting

    def _dominant_reason(self) -> str:
        if not self._reason_counts:
            return "unknown"
        return max(self._reason_counts.items(), key=lambda kv: kv[1])[0]

    def _blocked_fields(self, *, epoch: int, now: float) -> dict[str, Any]:
        elapsed = now - self._first_blocked_at if self._first_blocked_at else 0.0
        out: dict[str, Any] = {
            "consecutive": self._streak,
            "blocked_for": format_duration(elapsed),
            "reason": self._dominant_reason(),
            "epoch": epoch,
        }
        if self._last_blocks_short is not None:
            out["blocks_short"] = self._last_blocks_short
        if self._last_getlogs_p99_ms is not None:
            # Censored at the timeout bound, so ">=" is the honest reading.
            out["getlogs_p99_ms"] = f">={self._last_getlogs_p99_ms}"
        out["last_ok_epoch"] = (
            self._last_ok_epoch if self._last_ok_epoch is not None else "none-since-start"
        )
        return out

    def record(
        self,
        *,
        ready: bool,
        reason: str,
        epoch: int,
        now: float,
        blocks_short: int | None = None,
        getlogs_p99_ms: int | None = None,
    ) -> PoolGateEvent | None:
        """Fold one round's readiness verdict in. Returns an event to
        dispatch, or None (the overwhelmingly common case)."""
        if ready:
            return self._record_ready(epoch=epoch, now=now)
        return self._record_blocked(
            reason=reason, epoch=epoch, now=now, blocks_short=blocks_short,
            getlogs_p99_ms=getlogs_p99_ms,
        )

    def _record_ready(self, *, epoch: int, now: float) -> PoolGateEvent | None:
        was_alerting = self._alerting
        streak = self._streak
        dominant = self._dominant_reason()
        elapsed = now - self._first_blocked_at if self._first_blocked_at else 0.0
        prev_ok = self._last_ok_epoch

        self._streak = 0
        self._first_blocked_at = None
        self._last_alert_at = None
        self._alerting = False
        self._reason_counts = {}
        self._last_blocks_short = None
        self._last_getlogs_p99_ms = None
        self._last_ok_epoch = epoch

        if not was_alerting:
            return None
        fields: dict[str, Any] = {
            "recovered_after": streak,
            "blocked_for": format_duration(elapsed),
            "reason": dominant,
            "epoch": epoch,
            "last_ok_epoch": prev_ok if prev_ok is not None else "none-since-start",
        }
        return PoolGateEvent(
            kind=self.kind_recovered,
            detail=_kv_line(fields),
            fields=fields,
        )

    def _record_blocked(
        self, *, reason: str, epoch: int, now: float, blocks_short: int | None,
        getlogs_p99_ms: int | None = None,
    ) -> PoolGateEvent | None:
        if self._streak == 0:
            self._first_blocked_at = now
        self._streak += 1
        key = reason or "unknown"
        self._reason_counts[key] = self._reason_counts.get(key, 0) + 1
        if blocks_short is not None:
            self._last_blocks_short = int(blocks_short)
        if getlogs_p99_ms is not None:
            self._last_getlogs_p99_ms = int(getlogs_p99_ms)

        if self._streak < self.threshold:
            return None
        if self._alerting and self._last_alert_at is not None:
            if (now - self._last_alert_at) < self.realert_interval_s:
                return None
        self._alerting = True
        self._last_alert_at = now
        fields = self._blocked_fields(epoch=epoch, now=now)
        return PoolGateEvent(
            kind=self.kind_blocked,
            detail=_kv_line(fields),
            fields=fields,
        )


def _kv_line(fields: dict[str, Any]) -> str:
    """``k=v`` in insertion order — the repo's structured-line convention."""
    return " ".join(f"{k}={v}" for k, v in fields.items())
