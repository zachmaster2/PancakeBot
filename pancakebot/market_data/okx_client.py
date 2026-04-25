"""OKX public REST client with centralized retry + error classification.

Two fetch modes:

- `fetch_raw` (sync mode): retries transient failures per-call with exponential
  backoff + jitter. Raises `InvariantError` on permanent error or retry
  exhaustion. Caller is expected to accept hard failure of the whole sync on
  unrecoverable errors.

- `fetch_1s_klines` (live mode): single attempt. Raises `OkxTransientError` on
  transient failure (so caller can skip the round gracefully) and
  `InvariantError` on permanent failure (hard stop).
"""
from __future__ import annotations

import json
import random
import time
from enum import Enum
from typing import Callable

import requests

from pancakebot.log import error, info, warn
from pancakebot.util import InvariantError


_OKX_BASE_URL = "https://www.okx.com"

# Retry policy (sync mode only).
_MAX_ATTEMPTS_SYNC = 5  # initial + 4 retries
_BACKOFF_BASE_SYNC = [2.0, 4.0, 8.0, 16.0]  # 4 delays between 5 attempts
_JITTER_MIN = 0.5
_JITTER_MAX = 1.5

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


class OkxTransientError(Exception):
    """Raised by live-mode fetchers on retryable errors. Caller skips the round."""


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

    # Headers to extract from each OKX response when header capture is enabled.
    # These are the upstream-cache diagnostic fields: cache-control, age (CDN
    # response age in seconds), x-cache (varnish-style HIT/MISS), cf-cache-status
    # (CloudFlare HIT/MISS/EXPIRED), x-served-by (which edge node), via, server.
    # We don't want to capture EVERY header (privacy + bloat), just the ones
    # diagnostic for "is this a cached response" investigations.
    _CAPTURED_HEADER_NAMES: tuple[str, ...] = (
        "cache-control", "age", "x-cache", "cf-cache-status",
        "x-served-by", "via", "server", "date", "x-amz-cf-id",
        "x-amz-cf-pop", "expires", "etag",
    )

    def __init__(self, *, timeout_seconds: float) -> None:
        self._timeout_seconds = timeout_seconds
        self._session = requests.Session()
        # Per-symbol cache of the most-recent response headers, populated
        # only when capture is enabled. Read by MomentumGate to forward
        # into the kline_capture JSONL. Keyed by symbol (e.g. "BTC-USDT").
        # Empty dict on each new request to that symbol.
        self._last_response_headers: dict[str, dict[str, str]] = {}

    def last_response_headers(self, symbol: str) -> dict[str, str] | None:
        """Return the most-recent captured headers for *symbol*, or None.

        Always returns None when ``OkxClient._capture_headers_enabled()``
        was False at fetch time (i.e. the env var was unset). Otherwise
        returns the diagnostic header subset.
        """
        return self._last_response_headers.get(symbol)

    @staticmethod
    def _capture_headers_enabled() -> bool:
        """Opt-in via env var. Default OFF to avoid bloating capture file."""
        import os
        return os.environ.get("PANCAKEBOT_CAPTURE_OKX_HEADERS", "").lower() in ("1", "true", "yes")

    def warmup(self, connections: int = 3) -> None:
        """Fill the connection pool so parallel fetches hit warm connections.

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
          ``fetch_1s_klines`` call will raise ``OkxTransientError`` on its
          own, which the pipeline handles with a ``gate_no_btc_klines``
          skip. No reason to kill the bot from a warmup hiccup.

        Pre-fix history: OKX sporadically resets established sockets
        (`WinError 10054 / ConnectionResetError`). The uncaught re-raise
        here killed the dry bot twice in 24h (2026-04-22 22:58 EDT and
        2026-04-22 23:59 EDT). Supervisor reported CRASHED correctly both
        times but the process was still dead until manual restart.
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
        # NOTE: per-symbol _last_response_headers dict is preserved across
        # this swap (it's an OkxClient instance attr, not on Session).
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
    # Sync mode: retry-enabled fetch. Raises on permanent/exhaustion.
    # ----------------------------------------------------------------

    def fetch_raw(
        self,
        *,
        endpoint: str,
        params: dict[str, str],
        expected_count: int,
        rate_acquire_fn: Callable[[], None] | None = None,
    ) -> dict:
        """GET a /market/<endpoint> URL with retry + classification.

        Returns the parsed JSON body on SUCCESS (code='0' with exactly
        `expected_count` rows in `data`). Raises InvariantError on permanent
        failure or exhausted retries.

        *rate_acquire_fn* is called before every attempt (including retries)
        so shared rate-limit policy holds across retries too.
        """
        url = f"{_OKX_BASE_URL}/api/v5/market/{endpoint}"

        for attempt in range(_MAX_ATTEMPTS_SYNC):
            if rate_acquire_fn is not None:
                rate_acquire_fn()

            try:
                resp = self._session.get(url, params=params, timeout=self._timeout_seconds)
            except requests.RequestException as e:
                cls, detail = _classify_exception(e)
            else:
                cls, detail = _classify_response(resp, expected_count)

            if cls == _OkxErrorClass.SUCCESS:
                if attempt > 0:
                    info("NET", "OKX", "RECOVER",
                         attempts=attempt + 1, endpoint=endpoint)
                return resp.json()

            if cls == _OkxErrorClass.PERMANENT:
                raise InvariantError(
                    f"okx_permanent: endpoint={endpoint} detail={detail}"
                )

            # RETRYABLE or INSUFFICIENT — retry if attempts remain.
            is_last = (attempt == _MAX_ATTEMPTS_SYNC - 1)
            if is_last:
                error("NET", "OKX", "EXHAUST",
                      attempts=attempt + 1, endpoint=endpoint,
                      error_class=cls.value, error_detail=detail)
                raise InvariantError(
                    f"okx_retries_exhausted: endpoint={endpoint} "
                    f"class={cls.value} detail={detail}"
                )

            delay = _BACKOFF_BASE_SYNC[attempt] * random.uniform(_JITTER_MIN, _JITTER_MAX)
            warn("NET", "OKX", "RETRY",
                 attempt=attempt + 1, delay_s=f"{delay:.2f}",
                 endpoint=endpoint, error_class=cls.value, error_detail=detail)
            time.sleep(delay)

        # Unreachable — loop always returns or raises.
        raise InvariantError(f"okx_unreachable_code_path: endpoint={endpoint}")

    # ----------------------------------------------------------------
    # Live mode: single-attempt. Transient → OkxTransientError.
    # ----------------------------------------------------------------

    def fetch_1s_klines(
        self,
        *,
        symbol: str,
        count: int = 25,
        after_ms: int | None = None,
    ) -> list[dict[str, float | int]]:
        """Fetch the most recent `count` 1s klines from OKX (live-mode, no retry).

        When *after_ms* is provided, only candles with open_time < after_ms
        are returned (OKX ``after`` pagination parameter). This excludes
        the in-progress bar whose open_time equals the current second, so
        all returned candles are completed with final close prices.

        Returns oldest-first list of dicts with keys:
          open_time_ms, close_price

        Raises:
          OkxTransientError on retryable errors (caller skips the round).
          InvariantError on permanent errors (hard stop).
        """
        url = f"{_OKX_BASE_URL}/api/v5/market/candles"
        params: dict[str, str] = {
            "instId": symbol,
            "bar": "1s",
            "limit": str(count),
        }
        if after_ms is not None:
            params["after"] = str(after_ms)

        # Per-call fresh session to defeat WITHIN-round connection
        # affinity. The per-round warmup() reset addresses BETWEEN-round
        # affinity (eliminated stuck-cache mode entirely, 0/17 post-fix
        # vs 8/145 pre-fix), but n=17 measurement showed 100% any-lag
        # because warmup connections from the same round are re-used by
        # the kline fetch via urllib3's PoolManager. urllib3 keys the
        # pool on (scheme, host, port) so /public/time warmup connections
        # and /market/candles fetch connections share the same pool.
        # Standalone A/B (research/okx_connection_ab.py) confirmed:
        # `fresh_conn` (one Session per fetch) achieves mean lag ~0ms
        # vs `session_reuse` ~250ms even on quiet network conditions.
        # Cost: ~100-200ms TLS handshake per fetch, parallelised across
        # the 3 BTC/ETH/SOL fetches in fetch_klines_async via the gate's
        # ThreadPoolExecutor. Comfortably under the 1.75s budget between
        # cutoff+0.25s fetch and lock-1s safety margin.
        resp = None
        fresh_session = requests.Session()
        try:
            try:
                resp = fresh_session.get(url, params=params, timeout=self._timeout_seconds)
            except requests.RequestException as e:
                cls, detail = _classify_exception(e)
            else:
                cls, detail = _classify_response(resp, count)
        finally:
            try:
                fresh_session.close()
            except Exception:  # noqa: BLE001 -- never crash the fetch on cleanup
                pass

        # Capture diagnostic headers if env-var-enabled. Always update the
        # per-symbol slot (overwrite the previous response's headers) so
        # the gate sees the headers from THIS request. When capture is
        # disabled or no response was received, leave the slot untouched.
        if resp is not None and self._capture_headers_enabled():
            try:
                hdrs = {
                    name: resp.headers.get(name, "")
                    for name in self._CAPTURED_HEADER_NAMES
                    if resp.headers.get(name) is not None
                }
                self._last_response_headers[symbol] = hdrs
            except Exception:  # noqa: BLE001 -- never fail the fetch on header capture
                pass

        if cls == _OkxErrorClass.SUCCESS:
            body = resp.json()
            rows = body["data"]  # newest first
            result: list[dict[str, float | int]] = []
            for row in reversed(rows):
                if not isinstance(row, list) or len(row) < 6:
                    raise InvariantError("okx_client_1s_row_invalid")
                # Full OHLCV is captured for replay analysis (see
                # pancakebot.runtime.kline_capture). Strategy code path
                # only reads ``close_price`` so the extra fields are
                # passive observability data.
                result.append({
                    "open_time_ms": int(row[0]),
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close_price": float(row[4]),
                    "volume": float(row[5]),
                })
            return result

        if cls == _OkxErrorClass.PERMANENT:
            raise InvariantError(
                f"okx_live_permanent: symbol={symbol} detail={detail}"
            )

        # RETRYABLE or INSUFFICIENT — live mode surfaces both as transient skip.
        raise OkxTransientError(
            f"okx_live_transient: symbol={symbol} class={cls.value} detail={detail}"
        )
