"""Per-round NTP clock-sync manager.

Replaces the prior OKX-server-time skew measurement (Cristian's
algorithm against an exchange API) with direct NTP queries against
public stratum-1/2 pools. Called from the per-round ``ntp_sync_wake``
(pre-critical-path); the freshest measured offset is applied via
``current_offset()`` and consumed by ``engine._utc_now()``.

Why per-round NTP instead of OS-cached state:

The Windows Time Service polls upstream NTP at ~1h intervals by
default (configurable via SpecialPollInterval, but most hosts run the
default). Between OS polls, the local clock drifts ~1-10ms/min on a
stable host, more under load. A round 1 minute after the last OS
poll has a fresh clock; a round 50 minutes after has accumulated
drift in the tens of milliseconds.

Critical-path timing is tight enough that a few ms can flip a round's
bet decision (e.g. epoch 478372 was 40ms inside the timing-guard
margin in the 2026-05-04 soak). A per-round NTP query resets the
applied offset to the freshest measurement on each wake, so the
pre-cutoff timing budget is uniform across all rounds regardless of
where they fall between OS-NTP intervals.

Server rotation: cloudflare -> google -> pool.ntp.org, advancing one
slot per round so a single transient server outage doesn't fail every
round in a row. Each round queries exactly ONE server (no fallback
chain on the critical path); the wake budget is sized for one query.
The ``ntp_sync_wake`` budget (5000 ms via
``NTP_WAKE_OFFSET_PRE_BANKROLL_MS``) is generous enough that even a
full 3-server rotation fall-through (worst case ~306 ms p99) lands
well before the bankroll wake.

Failure handling:
- Query timeout / network error / glitch (offset > +/- 250ms): keep
  prior cached offset, increment ``consecutive_failures``. Engine
  decides skip-vs-bet based on consecutive_failures and last-query
  age.
- Glitch is logged WARN; transient timeouts are silent (per-server
  outage is normal background).

Engine pre-flight: at bot bootstrap, attempt one query per server
until one succeeds. If all fail or the initial offset is > 1s,
refuse to start (``InvariantError("ntp_bootstrap_failed")``) so the
operator sees the failure immediately rather than after a round of
silent stale-clock skipping.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import ntplib

from pancakebot.log import info, warn


# Default server rotation. Cloudflare and Google return in ~20-30ms in
# the probe; pool.ntp.org rotates across stratum-2 backends and lands
# at ~110-125ms. Order matters: cloudflare first because it's the
# fastest healthy server most of the time.
_DEFAULT_SERVERS: tuple[str, ...] = (
    "time.cloudflare.com",
    "time.google.com",
    "pool.ntp.org",
)

# UDP timeout for ``ntplib.request``. Tighter than the empirical p99
# (~155ms) of the probe so a slow server can't blow the wake budget;
# the timeout-then-skip path is the right outcome when one server
# stalls, since the next round rotates to a different server anyway.
_DEFAULT_TIMEOUT_S: float = 1.5

# Hard cap on applied skew. Real NTP-synced clock drift between
# polls is sub-millisecond per minute; anything beyond 250ms in a
# single measurement is a glitch (asymmetric routing, server
# overload, malformed response). Cap rejects the sample while still
# allowing genuinely-broken-clock recovery via the bootstrap path.
_OFFSET_GLITCH_CAP_SECONDS: float = 0.250

# Last-good offset is stale after this duration; engine refuses to
# bet if no successful query has landed within this window. Sized
# generously: 1 hour covers one full server outage cycle (each pool
# rotates faster than this) without flapping.
_LAST_GOOD_MAX_AGE_SECONDS: float = 3600.0

# Engine consults consecutive_failures vs this threshold; bet is
# skipped if the streak meets or exceeds it. 3 covers single-server
# outages (each round rotates to the next server, so streak only
# grows if multiple servers are down simultaneously -- a real
# network problem worth surfacing as a skip).
_DEFAULT_MAX_CONSECUTIVE_FAILURES: int = 3


@dataclass(slots=True)
class NtpSyncState:
    """Module-level state mirrored by NtpSync for the engine to read."""
    last_offset_seconds: float = 0.0
    last_query_ts: float = 0.0
    consecutive_failures: int = 0
    last_server: str = ""
    server_index: int = 0  # round-robin pointer
    successful_queries: int = 0
    glitch_rejections: int = 0


class NtpSync:
    """Per-round NTP clock offset manager.

    Engine calls ``force_resync()`` from the per-round ntp_sync_wake.
    On a successful query, the cached offset is updated and consumed
    by ``engine._utc_now()`` until the next wake. On failure, prior
    offset persists; engine reads ``is_healthy()`` to decide whether
    to skip the round on a stale-state guard.

    Thread-unsafe by design -- the engine's per-round loop is
    single-threaded; no other code path calls ``force_resync``.
    """

    def __init__(
        self,
        *,
        servers: tuple[str, ...] = _DEFAULT_SERVERS,
        timeout_seconds: float = _DEFAULT_TIMEOUT_S,
    ) -> None:
        if not servers:
            raise ValueError("NtpSync requires at least one server")
        self._servers = servers
        self._timeout_s = timeout_seconds
        self._state = NtpSyncState()
        self._client = ntplib.NTPClient()

    # -- public read API -------------------------------------------------

    def current_offset(self) -> float:
        """Cached (local - ntp) offset in seconds. Subtract from
        ``time.time()`` to get NTP-frame time. Returns 0.0 when no
        successful query has landed (initial state)."""
        return self._state.last_offset_seconds

    def consecutive_failures(self) -> int:
        return self._state.consecutive_failures

    def successful_queries(self) -> int:
        return self._state.successful_queries

    def glitch_rejections(self) -> int:
        return self._state.glitch_rejections

    def last_server(self) -> str:
        return self._state.last_server

    def last_query_age_seconds(self) -> float:
        """Seconds since the last successful NTP query landed. Returns
        infinity when no successful query has happened yet."""
        if self._state.last_query_ts == 0.0:
            return float("inf")
        return time.time() - self._state.last_query_ts

    def is_healthy(
        self,
        *,
        max_consecutive_failures: int = _DEFAULT_MAX_CONSECUTIVE_FAILURES,
        max_age_seconds: float = _LAST_GOOD_MAX_AGE_SECONDS,
    ) -> bool:
        """Engine consults this at the wake to decide skip-vs-bet.

        Healthy iff:
        - ``consecutive_failures`` is below the streak threshold AND
        - last successful query is younger than ``max_age_seconds``.
        """
        if self._state.consecutive_failures >= max_consecutive_failures:
            return False
        if self.last_query_age_seconds() > max_age_seconds:
            return False
        return True

    # -- write API: engine calls this from the wake ---------------------

    def force_resync(self) -> bool:
        """Single NTP query against the next server in rotation. Updates
        cached offset on success.

        Returns True iff a fresh offset was applied. Returns False on:
        - Timeout / network error
        - Other ``ntplib`` exception
        - Offset exceeds the +/- 250ms glitch cap

        On any failure, the prior cached offset is preserved (so a
        single bad query doesn't lose the last known good correction).
        ``consecutive_failures`` increments on every False return.
        """
        server_index = self._state.server_index % len(self._servers)
        server = self._servers[server_index]
        self._state.server_index += 1
        try:
            response = self._client.request(
                server, version=3, timeout=self._timeout_s,
            )
        except Exception:  # noqa: BLE001 -- never crash the round
            self._state.consecutive_failures += 1
            return False
        offset = float(response.offset)
        if abs(offset) > _OFFSET_GLITCH_CAP_SECONDS:
            self._state.glitch_rejections += 1
            self._state.consecutive_failures += 1
            warn(
                "CLOCK", "NTP", "GLITCH",
                msg=(
                    f"NTP offset {offset:+.3f}s exceeds glitch cap "
                    f"+/- {_OFFSET_GLITCH_CAP_SECONDS:.3f}s; rejected. "
                    f"server={server}"
                ),
            )
            return False
        prev_offset = self._state.last_offset_seconds
        self._state.last_offset_seconds = offset
        self._state.last_query_ts = time.time()
        self._state.last_server = server
        self._state.consecutive_failures = 0
        self._state.successful_queries += 1
        delta = abs(offset - prev_offset)
        if delta >= 0.010 or self._state.successful_queries == 1:
            info(
                "CLOCK", "NTP", "UPDATE",
                msg=(
                    f"NTP offset (local - ntp): "
                    f"{prev_offset * 1000:+.2f}ms -> {offset * 1000:+.2f}ms "
                    f"server={server}"
                ),
            )
        return True

    # -- bootstrap helper ------------------------------------------------

    def bootstrap(self, *, max_attempts_per_server: int = 1) -> bool:
        """Try each server in rotation until one succeeds. Caller (engine
        bootstrap) raises InvariantError if this returns False so the
        operator sees the failure at startup rather than after silent
        round skipping.

        Returns True iff at least one server responded with an offset
        within the glitch cap.
        """
        for _ in range(len(self._servers) * max_attempts_per_server):
            if self.force_resync():
                return True
        return False
