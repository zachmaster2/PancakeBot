"""OKX public REST client with centralized retry + error classification.

Single canonical fetch primitive: ``kline_fetch_window``. Always uses
``/history-candles`` (per memory ``project_pancakebot_okx_endpoint_divergence.md``,
this endpoint has lower lag than ``/candles``). Caller passes the explicit
``[oldest_open_ms, newest_open_ms_inclusive]`` range; retry behaviour is
parameterised via ``RetryPolicy``:

- ``RETRY_SYNC``: 5 attempts with exponential backoff (used by sync.py for
  bulk historical fetch — needs robustness over 38k+ rounds).
- ``RETRY_NONE``: single attempt, no retry (used by the per-round live-
  decision gate). Retry on the live path adds 2.5s+ wall-clock that
  always pushes decision past lock_at, contributing zero bet-placement
  value. The gate's ``max_consecutive_fetch_failures`` streak counter
  (in MomentumGateConfig) handles transient-failure escalation.

Returns oldest-first list of ``[ts_ms, open, high, low, close, volume]``
arrays. Length is exactly ``(newest - oldest)/1000 + 1``. The query always
sends ``after`` (exclusive upper). When ``send_before_bound=True`` the
query also sends ``before`` (exclusive lower), pinning the window so OKX
cannot slide it when the newest requested candle isn't yet published — in
that case OKX returns fewer rows (classified ``INSUFFICIENT``, retried,
surfaced as ``TransientOkxError``) rather than silently filling with older
candles. The live decision-path gate uses ``send_before_bound=True``; sync
keeps the default ``False`` to preserve canonical-baseline request shape
(its historical data is fully published, so window-sliding cannot occur).
Boundary is also verified: the returned oldest/newest open_times must
equal the requested values, else ``InvariantError`` (catches OKX data
holes that ``len == expected_count`` alone would miss).

Error contract:
- Contract violations (contiguity gap, boundary mismatch, inverted/unaligned
  range, row malformed, OKX permanent-class response) raise
  ``InvariantError``. These indicate OKX returned data that violates our
  shape invariants, or the caller passed a malformed window.
- Retry-exhausted transient failures (network exception, HTTP 429/5xx,
  retryable OKX codes, repeated short/empty responses including
  ``len < expected_count`` when newest is unpublished) raise
  ``TransientOkxError``. Callers on the live decision path catch this to
  skip the round; sync's bulk loader treats it as fail-loud (it shouldn't
  happen with RETRY_SYNC's 5-attempt budget on already-published history).
"""
from __future__ import annotations

import json
import random
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable

import requests

from pancakebot.log import error, info, warn
from pancakebot.util import InvariantError, TransientOkxError


_OKX_BASE_URL = "https://www.okx.com"

_JITTER_MIN = 0.5
_JITTER_MAX = 1.5


# ---------------------------------------------------------------------------
# Shared OKX rate budget (single canonical source for all OKX REST callers).
# OKX allows 20 req/2s per endpoint per IP = 10/s. Cap at 8/s to leave margin.
# All in-process callers (sync.py bulk fetch, per-round live-gate fetch)
# use ``okx_rate_acquire`` so a single token bucket throttles total OKX
# load across the live system.
#
# Token-bucket shape (capacity = refill_rate = 8): burst up to 8 requests
# instantly, then 1 token / 125 ms refill. Live gate fires 4 symbols per
# round (every 5 min) -- well within burst capacity, so all 4 acquire
# immediately with zero stagger. Sync.py's bulk fetch sustains close to
# 8 req/s long-run (refill-rate-limited, no burst gain after the first
# 8 requests). The previous "leaky bucket with no burst" implementation
# slept while holding the rate-limiter lock, which serialized concurrent
# parallel-symbol acquires at FIFO 125 ms intervals; the token bucket
# never sleeps under the lock, so concurrent acquires don't serialize
# unless the bucket is genuinely empty.
# ---------------------------------------------------------------------------

_OKX_RATE_LIMIT_PER_SEC = 8
_OKX_RATE_BUCKET_CAPACITY = 8
_okx_rate_lock = threading.Lock()
# Bucket starts FULL so the first burst after process start fires
# unconstrained (no synthetic startup penalty).
_okx_rate_tokens: float = float(_OKX_RATE_BUCKET_CAPACITY)
_okx_rate_last_refill_t: float = 0.0


def _okx_rate_reset_for_tests() -> None:
    """Reset the global rate-limiter state (test-only helper).

    Tests need a clean bucket per case so behavior is deterministic.
    Safe to call between tests; not thread-safe with concurrent acquires.
    """
    global _okx_rate_tokens, _okx_rate_last_refill_t
    with _okx_rate_lock:
        _okx_rate_tokens = float(_OKX_RATE_BUCKET_CAPACITY)
        _okx_rate_last_refill_t = time.monotonic()


def okx_rate_acquire() -> None:
    """Acquire one OKX-rate token. Blocks only when the bucket is empty.

    Token bucket semantics:
      - Capacity = ``_OKX_RATE_BUCKET_CAPACITY`` tokens (default 8).
      - Refill = ``_OKX_RATE_LIMIT_PER_SEC`` tokens/sec (default 8/s),
        i.e. one token every 125 ms.
      - Lazy refill: each call computes ``elapsed * refill_rate`` new
        tokens since the last refill timestamp, clamped to capacity.
      - If a token is available, decrement and return immediately.
      - If empty, sleep ``time-until-next-token`` OUTSIDE the lock,
        re-enter, and retake. The lock is never held across ``time.sleep``.

    Process-wide shared budget. Pass as ``rate_acquire_fn`` to
    ``OkxClient.kline_fetch_window``.
    """
    global _okx_rate_tokens, _okx_rate_last_refill_t
    refill_rate = float(_OKX_RATE_LIMIT_PER_SEC)
    capacity = float(_OKX_RATE_BUCKET_CAPACITY)
    while True:
        with _okx_rate_lock:
            now = time.monotonic()
            elapsed = now - _okx_rate_last_refill_t
            if elapsed > 0.0:
                _okx_rate_tokens = min(
                    capacity, _okx_rate_tokens + elapsed * refill_rate,
                )
                _okx_rate_last_refill_t = now
            if _okx_rate_tokens >= 1.0:
                _okx_rate_tokens -= 1.0
                return
            # Bucket empty: compute wait time for the next token.
            tokens_short = 1.0 - _okx_rate_tokens
            wait_s = tokens_short / refill_rate
        # Sleep OUTSIDE the lock so other threads can refill/acquire
        # in parallel if the bucket happens to refill from another
        # source (e.g. test resets) or simply observe their own wait
        # times concurrently.
        time.sleep(wait_s)

# ---------------------------------------------------------------------------
# Retry policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetryPolicy:
    """Retry config for ``kline_fetch_window``.

    ``backoff_seconds[i]`` is the sleep BEFORE attempt i+1 (jittered by
    [0.5x, 1.5x]). Length must be ``max_attempts - 1``.
    """
    max_attempts: int
    backoff_seconds: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if len(self.backoff_seconds) != max(0, self.max_attempts - 1):
            raise ValueError(
                f"backoff_seconds len={len(self.backoff_seconds)} "
                f"must equal max_attempts-1={self.max_attempts - 1}"
            )


# Sync.py iterates 38k+ rounds; needs robust retry over slow OKX backends.
RETRY_SYNC = RetryPolicy(max_attempts=5, backoff_seconds=(2.0, 4.0, 8.0, 16.0))

# Per-round live-decision gate fetch: NO retry. Single attempt; on
# insufficient/error -> TransientOkxError -> gate skips the round with
# ``kline_fetch_transient_failure``. Retry on the live path always
# pushes decision past lock_at (2.5s+ backoff > pre-lock budget), where
# the engine's timing guard would skip anyway -- so retry just adds
# wall-clock + rate-budget cost for no operational benefit. The gate's
# ``max_consecutive_fetch_failures`` streak counter (configured per
# MomentumGateConfig) handles escalation: N consecutive -> bot crashes
# with InvariantError -> supervisor restart + Discord alert.
RETRY_NONE = RetryPolicy(max_attempts=1, backoff_seconds=())

# Test fixture only (tests/test_okx_warmup_transient.py exercises the
# retry path with this policy). Not used by the production live
# decision path, which uses RETRY_NONE above.
RETRY_GATE = RetryPolicy(max_attempts=2, backoff_seconds=(2.5,))

# OKX API-level error codes that are transient and warrant retry.
# See https://www.okx.com/docs-v5/en/#error-code
_RETRYABLE_OKX_CODES = frozenset({
    "50011",  # Request too frequent
    "50013",  # System is busy, please try again later
    "50014",  # Parameter error (sometimes returned transiently)
    "50061",  # The request is too fast and failed risk control
})


class _OkxErrorClass(Enum):
    SUCCESS = "success"            # valid response with full expected data
    RETRYABLE = "retryable"        # network/5xx/429/retryable-code — retry applies
    PERMANENT = "permanent"        # 4xx/permanent-code — don't retry
    INSUFFICIENT = "insufficient"  # code=0 but empty or short data — retry, then raise


def _classify_response(resp: requests.Response, expected_count: int) -> tuple[_OkxErrorClass, str]:
    """Classify an HTTP response into one of the four outcome classes."""
    status = resp.status_code
    if status == 429 or (500 <= status <= 599):
        return (_OkxErrorClass.RETRYABLE, f"http_{status}")
    if status in (400, 404, 451):
        return (_OkxErrorClass.PERMANENT, f"http_{status}")
    if status != 200:
        return (_OkxErrorClass.PERMANENT, f"http_{status}")

    try:
        body = resp.json()
    except (ValueError, json.JSONDecodeError):
        return (_OkxErrorClass.RETRYABLE, "json_parse_error")
    if not isinstance(body, dict):
        return (_OkxErrorClass.PERMANENT, "body_not_dict")

    code = str(body.get("code", ""))
    if code != "0":
        if code in _RETRYABLE_OKX_CODES:
            return (_OkxErrorClass.RETRYABLE, f"okx_code_{code}")
        return (_OkxErrorClass.PERMANENT, f"okx_code_{code}")

    data = body.get("data")
    if not isinstance(data, list) or len(data) == 0:
        return (_OkxErrorClass.INSUFFICIENT, "empty_data")
    if expected_count > 0 and len(data) != expected_count:
        return (_OkxErrorClass.INSUFFICIENT, f"got_{len(data)}_expected_{expected_count}")
    return (_OkxErrorClass.SUCCESS, "")


def _classify_exception(exc: Exception) -> tuple[_OkxErrorClass, str]:
    """Pre-response exceptions (network/timeout) are always retryable."""
    return (_OkxErrorClass.RETRYABLE, f"{type(exc).__name__}")


class OkxClient:
    """OKX Spot public REST client (unauthenticated) with retry and classification."""

    def __init__(self, *, timeout_seconds: float) -> None:
        self._timeout_seconds = timeout_seconds
        self._session = requests.Session()

    def warmup(self, connections: int = 4) -> None:
        """Fill the connection pool so parallel fetches hit warm connections.

        Default ``connections=4`` matches the live decision-path gate's
        4-symbol concurrent fetch (BTC/ETH/SOL/BNB), so every parallel
        request finds a pre-established TLS connection. With only 3 warm
        sockets, the 4th symbol used to pay an ~80-150ms TLS handshake
        on the critical path every cold round.

        Per-round freshness: BEFORE filling the pool, close the existing
        ``self._session`` and replace it with a brand-new
        ``requests.Session()``. This breaks any connection-reuse-induced
        backend affinity carried over from prior rounds. Without this,
        kline fetches in subsequent rounds get sticky-routed to the same
        OKX backend via TCP keep-alive, and that backend can serve stale
        data (1-3s lag in 75% of dry-bot rounds, occasional stuck-cache
        for 5.8%). A fresh session per round breaks the affinity.

        Verified by A/B probe 2026-04-25
        (research/okx_connection_ab.py): fresh_conn cuts mean lag
        458ms -> 95ms (first run) and 256ms -> -3ms (extended run).
        Definitively NOT CDN caching -- all 270 probe responses had
        ``cf-cache-status: DYNAMIC`` and ``age: 0``. The fix targets
        connection-reuse affinity below the IP layer.

        Best-effort. Transient network errors (``requests.RequestException``
        and raw ``OSError``/``ConnectionResetError`` that can occasionally
        escape the requests/urllib3 wrapping) are logged WARN and swallowed:

        - A cold/broken connection here just means the next real fetch pays
          the TLS-handshake latency itself; the round's signal logic still
          works.
        - If the underlying network is genuinely down, the next
          ``kline_fetch_window`` call (sync bulk fetch or live-gate
          per-round fetch) will raise ``TransientOkxError`` on its own
          after retry exhaustion. No reason to kill the bot from a warmup
          hiccup.

        Pre-fix history: OKX sporadically resets established sockets
        (`WinError 10054 / ConnectionResetError`). The uncaught re-raise
        here killed the dry bot twice in 24h (2026-04-22 22:58 EDT and
        2026-04-22 23:59 EDT). Supervisor reported CRASHED correctly both
        times but the process was still dead until manual restart.

        Note on error contract: a fully-failed network underneath warmup
        does NOT raise here -- the next ``kline_fetch_window`` call will
        surface a ``TransientOkxError`` (live gate path) or exhaust to
        ``InvariantError`` via PERMANENT classification (rare; only on
        explicit OKX rejection rather than network down).
        """
        # Step 1: close the stale session (breaks any backend affinity from
        # prior rounds). Best-effort -- a failed close just leaks a few
        # idle sockets that the OS will reap anyway. Don't crash on cleanup.
        try:
            self._session.close()
        except Exception:  # noqa: BLE001 -- swallow cleanup errors
            pass
        # Step 2: brand-new session for this round's pool.
        self._session = requests.Session()
        # Step 3: pre-warm the new pool with parallel GETs.
        from concurrent.futures import ThreadPoolExecutor
        url = f"{_OKX_BASE_URL}/api/v5/public/time"
        with ThreadPoolExecutor(max_workers=connections) as pool:
            futs = [pool.submit(self._session.get, url, timeout=self._timeout_seconds)
                    for _ in range(connections)]
            for i, f in enumerate(futs):
                # Catch both the requests-wrapped exceptions (ConnectionError,
                # Timeout, etc.) and raw OSError/ConnectionResetError that
                # Python 3.13 can surface directly from the ssl/socket layer
                # before requests wraps them. Any non-network bug still
                # propagates so we don't mask real errors.
                try:
                    f.result()
                except requests.RequestException as e:
                    warn("NET", "OKX", "WARM_SKIP",
                         msg=f"warmup conn {i} failed (transient, ignored): "
                             f"{type(e).__name__}: {e}")
                except (ConnectionResetError, OSError) as e:
                    warn("NET", "OKX", "WARM_SKIP",
                         msg=f"warmup conn {i} raw socket error (transient, "
                             f"ignored): {type(e).__name__}: {e}")

    # ----------------------------------------------------------------
    # Clock skew measurement
    # ----------------------------------------------------------------

    def measure_clock_skew(self, samples: int = 5, timeout_seconds: float | None = None) -> float | None:
        """Measure (local - OKX) clock skew via /api/v5/public/time.

        Returns the median observed skew in SECONDS (positive = local clock
        ahead of OKX). Each sample takes the midpoint of the local-time
        bracket around the request as the local-clock value at the moment
        OKX timestamped its response, RTT-correcting naturally.

        Returns None if every sample fails (network down, OKX unreachable,
        invalid response). Caller should fall back to skew=0 with a logged
        warning -- the bot must keep running on a best-effort basis.

        Cost: typically 50-200ms per sample, run sequentially. With 5
        samples and ~100ms each, total ~500ms. Designed for off-critical-
        path use (housekeeping phase, not the bet-submission window).
        """
        import requests
        import statistics
        timeout = float(timeout_seconds) if timeout_seconds is not None else self._timeout_seconds
        url = f"{_OKX_BASE_URL}/api/v5/public/time"
        skews_s: list[float] = []
        for _ in range(max(1, int(samples))):
            try:
                t0 = time.time()
                resp = requests.get(url, timeout=timeout)
                t1 = time.time()
                if resp.status_code != 200:
                    continue
                body = resp.json()
                if not isinstance(body, dict):
                    continue
                data = body.get("data") or []
                if not data or not isinstance(data[0], dict):
                    continue
                ts_str = data[0].get("ts")
                if not ts_str:
                    continue
                okx_ms = int(ts_str)
                # Local time at the moment OKX served the response: use the
                # midpoint of the local-bracket around the request to
                # roughly RTT-correct (assumes symmetric network latency).
                local_ms_at_response = (t0 + t1) / 2.0 * 1000.0
                skews_s.append((local_ms_at_response - okx_ms) / 1000.0)
            except (requests.RequestException, ValueError, KeyError, OSError):
                # Best-effort. Keep going for the remaining samples.
                continue
        if not skews_s:
            return None
        return float(statistics.median(skews_s))

    # ----------------------------------------------------------------
    # Canonical primitive: explicit-window /history-candles fetch.
    # ----------------------------------------------------------------

    def kline_fetch_window(
        self,
        *,
        symbol: str,
        oldest_open_ms: int,
        newest_open_ms_inclusive: int,
        retry_policy: RetryPolicy = RETRY_SYNC,
        rate_acquire_fn: Callable[[], None] | None = None,
        send_before_bound: bool = False,
    ) -> tuple[list[list], int]:
        """Fetch the inclusive 1s-candle window ``[oldest_open_ms, newest_open_ms_inclusive]``
        from OKX ``/api/v5/market/history-candles``.

        Returns ``(rows, rtt_ms)`` where:
          - ``rows`` is the oldest-first list of
            ``[ts_ms, open, high, low, close, volume]`` arrays. Length
            is exactly ``(newest - oldest)/1000 + 1``.
          - ``rtt_ms`` is the wall-clock duration of the SUCCESSFUL
            ``self._session.get(...)`` call only -- excludes rate-limiter
            wait, retry backoff sleeps, JSON parse, and shape validation.
            On a retry-then-success sequence, ``rtt_ms`` reports just the
            successful attempt's network roundtrip (failed attempts are
            reflected in the operator-facing ``RETRY``/``RECOVER`` log
            lines rather than this metric). Callers that don't care about
            timing can discard via ``rows, _ = client.kline_fetch_window(...)``.

        The query always sends ``after = newest + 1000``; when
        ``send_before_bound=True`` it additionally sends
        ``before = oldest - 1000`` (both exclusive in OKX semantics),
        pinning the window so OKX cannot slide it when the newest candle
        is unpublished. The returned oldest/newest open_times are verified
        to equal the requested values; any deviation raises
        ``InvariantError`` (catches OKX data holes that a length-only check
        would miss).

        Error contract:
        - Contract violations (contiguity, boundary, malformed input)
          raise ``InvariantError`` — fail-loud; data shape is wrong.
        - Retry-exhausted transient failures (network, HTTP 429/5xx,
          retryable OKX codes, repeatedly empty/short responses — the
          short-response case fires when the newest requested candle
          hasn't been published yet at fetch time) raise
          ``TransientOkxError`` — fail-soft on the live decision path.

        ``rate_acquire_fn`` is invoked before EVERY attempt (including
        retries) so a shared rate budget can be threaded through. Pass None
        to bypass rate limiting (rarely useful; almost every caller in this
        codebase shares the global ``okx_rate_acquire`` budget).
        """
        if newest_open_ms_inclusive < oldest_open_ms:
            raise InvariantError(
                f"kline_fetch_window_inverted_range: oldest={oldest_open_ms} "
                f"newest={newest_open_ms_inclusive}"
            )
        if (newest_open_ms_inclusive - oldest_open_ms) % 1000 != 0:
            raise InvariantError(
                f"kline_fetch_window_unaligned_ms: oldest={oldest_open_ms} "
                f"newest={newest_open_ms_inclusive} (must be 1s-aligned)"
            )
        expected_count = (newest_open_ms_inclusive - oldest_open_ms) // 1000 + 1
        if expected_count > 300:
            # OKX /history-candles caps at 300 per request. Caller must
            # paginate (none of our current callers do; sync's per-round
            # window is exactly 300, the live gate's is much smaller).
            raise InvariantError(
                f"kline_fetch_window_exceeds_max: count={expected_count} max=300 "
                f"symbol={symbol}"
            )
        # Window bounds:
        # - OKX `after` is exclusive (returns rows with open_time < after).
        #   To include newest_open_ms_inclusive, set after = newest + 1000.
        # - OKX `before` is also exclusive (returns rows with open_time >
        #   before). To include oldest_open_ms, set before = oldest - 1000.
        # Setting BOTH bounds (``send_before_bound=True``) prevents OKX
        # from sliding the window when the newest requested candle hasn't
        # been published yet -- in that case OKX returns FEWER than
        # expected_count rows and the response is classified as
        # INSUFFICIENT (retry, then TransientOkxError on exhaustion).
        # Without ``before``, OKX would silently return expected_count
        # older rows, tripping the boundary check below into a false
        # InvariantError -- the root cause of the
        # ``kline_fetch_integrity_violation`` epidemic that crashed the
        # live bot ~67% of rounds (2026-04-27).
        # Sync (bulk historical) leaves ``send_before_bound=False`` so its
        # request shape is bit-identical to the canonical baseline; OKX
        # historical data is fully published by sync time, so the
        # window-sliding failure mode the bound prevents cannot fire there.
        after_ms = newest_open_ms_inclusive + 1000
        url = f"{_OKX_BASE_URL}/api/v5/market/history-candles"
        params: dict[str, str] = {
            "instId": symbol,
            "bar": "1s",
            "limit": str(expected_count),
            "after": str(after_ms),
        }
        if send_before_bound:
            params["before"] = str(oldest_open_ms - 1000)

        last_detail = ""
        last_class: _OkxErrorClass | None = None
        rtt_ms = 0
        for attempt in range(retry_policy.max_attempts):
            if rate_acquire_fn is not None:
                rate_acquire_fn()
            # Time only around the .get() call -- excludes rate wait and
            # any prior retry backoff. On retry, this overwrites the
            # previous attempt's measurement; on SUCCESS the value is the
            # successful attempt's true HTTP RTT.
            t_get_start = time.perf_counter()
            try:
                resp = self._session.get(url, params=params, timeout=self._timeout_seconds)
            except requests.RequestException as e:
                rtt_ms = int((time.perf_counter() - t_get_start) * 1000)
                cls, detail = _classify_exception(e)
            else:
                rtt_ms = int((time.perf_counter() - t_get_start) * 1000)
                cls, detail = _classify_response(resp, expected_count)

            if cls == _OkxErrorClass.SUCCESS:
                if attempt > 0:
                    info("NET", "OKX", "RECOVER",
                         attempts=attempt + 1, endpoint="history-candles",
                         symbol=symbol)
                body = resp.json()
                rows = body["data"]  # OKX returns newest-first
                arrays: list[list] = []
                for row in reversed(rows):
                    if not isinstance(row, list) or len(row) < 6:
                        raise InvariantError(
                            f"kline_fetch_integrity_violation: symbol={symbol} "
                            f"detail=row_invalid"
                        )
                    arrays.append([
                        int(row[0]), float(row[1]), float(row[2]),
                        float(row[3]), float(row[4]), float(row[5]),
                    ])
                # Boundary verification: requested oldest/newest must match.
                # _classify_response already ensured len == expected_count.
                if arrays[0][0] != oldest_open_ms or arrays[-1][0] != newest_open_ms_inclusive:
                    raise InvariantError(
                        f"kline_fetch_integrity_violation: symbol={symbol} "
                        f"detail=boundary_mismatch "
                        f"requested=[{oldest_open_ms}..{newest_open_ms_inclusive}] "
                        f"got=[{arrays[0][0]}..{arrays[-1][0]}] count={len(arrays)}"
                    )
                # Contiguity verification: every adjacent pair must be exactly
                # 1000 ms apart. Catches mid-window holes / duplicates /
                # out-of-order rows that the boundary check alone would miss.
                # Required for the live decision path AND for sync's on-disk
                # records (downstream slice math assumes contiguity).
                for i in range(1, len(arrays)):
                    delta = arrays[i][0] - arrays[i - 1][0]
                    if delta != 1000:
                        raise InvariantError(
                            f"kline_fetch_integrity_violation: symbol={symbol} "
                            f"detail=noncontiguous "
                            f"idx={i} prev_ts={arrays[i - 1][0]} cur_ts={arrays[i][0]} "
                            f"delta_ms={delta} (expected 1000)"
                        )
                return arrays, rtt_ms

            if cls == _OkxErrorClass.PERMANENT:
                raise InvariantError(
                    f"kline_fetch_permanent: symbol={symbol} detail={detail}"
                )

            # RETRYABLE or INSUFFICIENT — retry per policy if attempts remain.
            last_detail = detail
            last_class = cls
            is_last = (attempt == retry_policy.max_attempts - 1)
            if is_last:
                error("NET", "OKX", "EXHAUST",
                      attempts=attempt + 1, endpoint="history-candles",
                      symbol=symbol, error_class=cls.value, error_detail=detail)
                raise TransientOkxError(
                    f"kline_fetch_exhausted: symbol={symbol} "
                    f"class={cls.value} detail={detail}"
                )
            base_delay = retry_policy.backoff_seconds[attempt]
            delay = base_delay * random.uniform(_JITTER_MIN, _JITTER_MAX)
            warn("NET", "OKX", "RETRY",
                 attempt=attempt + 1, delay_s=f"{delay:.2f}",
                 endpoint="history-candles", symbol=symbol,
                 error_class=cls.value, error_detail=detail)
            time.sleep(delay)

        # Defensive: loop above always returns or raises.
        raise InvariantError(
            f"kline_fetch_unreachable: symbol={symbol} "
            f"last_class={last_class} last_detail={last_detail}"
        )
