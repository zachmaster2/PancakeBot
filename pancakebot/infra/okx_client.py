"""Minimal OKX public REST client for BNB/USDT 1m klines.

OKX is accessible from US IPs for unauthenticated market data.
No API key required.
"""

from __future__ import annotations

import json
import time

import requests

from pancakebot.core.errors import InvariantError


_OKX_BASE_URL = "https://www.okx.com"


class OkxClient:
    """Minimal OKX Spot public REST client (unauthenticated)."""

    def __init__(self, *, timeout_seconds: float) -> None:
        self._timeout_seconds = float(timeout_seconds)

    def fetch_current_1m_kline(
        self,
        *,
        symbol: str,
        before_ts_ms: int,
    ) -> dict[str, float | int] | None:
        """Return the most recent 1m kline (confirmed or in-progress) whose open_time_ms < before_ts_ms.

        Signal semantics
        ----------------
        The momentum signal is computed as:

            ret = (close_price / open_price) - 1

        For a confirmed kline close_price is the final settled price.
        For an in-progress kline close_price is the last traded price at the
        moment this API call is made — i.e. the live spot price at decision time.
        That is exactly what we want: the return from the start of the current
        minute up to *right now*, giving a momentum reading based on current
        market price rather than data that is up to 60 s stale.

        Endpoint choice
        ---------------
        Uses /candles (live feed) instead of /history-candles so that the
        current in-progress bar is always present in the response.  No confirm
        flag filter is applied — the most recent bar before before_ts_ms is
        returned regardless of whether it has closed.

        Returns a dict with keys:
          open_time_ms, close_time_ms, open_price, high_price, low_price,
          close_price, volume, quote_asset_volume

        Returns None if no kline is found before the given timestamp.
        """
        url = f"{_OKX_BASE_URL}/api/v5/market/candles"
        params = {
            "instId": str(symbol),
            "bar": "1m",
            "limit": "5",
        }

        try:
            r = requests.get(url, params=params, timeout=self._timeout_seconds)
        except requests.RequestException as e:
            raise InvariantError(f"okx_client_request_failed: {e}") from e

        if r.status_code != 200:
            raise InvariantError(
                f"okx_client_http_error: status={r.status_code} body={r.text[:200]}"
            )

        try:
            payload = r.json()
        except json.JSONDecodeError as e:
            raise InvariantError(f"okx_client_json_decode_error: {e}") from e

        if not isinstance(payload, dict):
            raise InvariantError("okx_client_response_not_dict")

        code = payload.get("code")
        if str(code) != "0":
            raise InvariantError(f"okx_client_api_error: code={code} msg={payload.get('msg')}")

        rows = payload.get("data")
        if not isinstance(rows, list):
            raise InvariantError("okx_client_data_not_list")

        # Rows are newest-first.  Take the first row whose open_time_ms is
        # strictly before before_ts_ms.  This is the current in-progress bar
        # (or the most recently closed bar if we happen to land exactly on a
        # minute boundary).  Its close_price = last traded price = spot price
        # at the time of this API call.
        for row in rows:
            if not isinstance(row, list) or len(row) < 9:
                raise InvariantError("okx_client_row_invalid")
            ts_ms = int(row[0])
            if ts_ms >= int(before_ts_ms):
                continue  # too recent
            return {
                "open_time_ms": ts_ms,
                "close_time_ms": ts_ms + 60_000 - 1,
                "open_price": float(row[1]),
                "high_price": float(row[2]),
                "low_price": float(row[3]),
                "close_price": float(row[4]),
                "volume": float(row[5]),
                "quote_asset_volume": float(row[6]),
            }

        return None

    def fetch_1s_klines(
        self,
        *,
        symbol: str,
        count: int = 25,
    ) -> list[dict[str, float | int]] | None:
        """Fetch the most recent `count` confirmed 1s klines from OKX.

        Uses /candles (live feed) so the response includes the current
        in-progress 1s bar.  Returns oldest-first list of dicts with keys:
          open_time_ms, close_price

        Returns None on failure.
        """
        url = f"{_OKX_BASE_URL}/api/v5/market/candles"
        params = {
            "instId": str(symbol),
            "bar": "1s",
            "limit": str(int(count)),
        }

        try:
            r = requests.get(url, params=params, timeout=self._timeout_seconds)
        except requests.RequestException as e:
            raise InvariantError(f"okx_client_1s_request_failed: {e}") from e

        if r.status_code != 200:
            raise InvariantError(
                f"okx_client_1s_http_error: status={r.status_code} body={r.text[:200]}"
            )

        try:
            payload = r.json()
        except json.JSONDecodeError as e:
            raise InvariantError(f"okx_client_1s_json_decode_error: {e}") from e

        if not isinstance(payload, dict):
            raise InvariantError("okx_client_1s_response_not_dict")

        code = payload.get("code")
        if str(code) != "0":
            raise InvariantError(f"okx_client_1s_api_error: code={code} msg={payload.get('msg')}")

        rows = payload.get("data")
        if not isinstance(rows, list) or len(rows) == 0:
            return None

        # Rows are newest-first; reverse to oldest-first.
        result: list[dict[str, float | int]] = []
        for row in reversed(rows):
            if not isinstance(row, list) or len(row) < 6:
                raise InvariantError("okx_client_1s_row_invalid")
            result.append({
                "open_time_ms": int(row[0]),
                "close_price": float(row[4]),
            })
        return result
