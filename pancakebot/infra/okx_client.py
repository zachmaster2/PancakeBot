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

    def fetch_last_confirmed_1m_kline(
        self,
        *,
        symbol: str,
        before_ts_ms: int,
    ) -> dict[str, float | int] | None:
        """Return the most recent confirmed closed 1m kline whose open_time_ms < before_ts_ms.

        OKX /history-candles returns rows newest-first with a confirm flag:
          0 = open (in progress), 1 = closed/confirmed.

        Returns a dict with keys:
          open_time_ms, close_time_ms, open_price, high_price, low_price,
          close_price, volume, quote_asset_volume

        Returns None if no confirmed kline is found before the given timestamp.
        """
        url = f"{_OKX_BASE_URL}/api/v5/market/history-candles"
        params = {
            "instId": str(symbol),
            "bar": "1m",
            "limit": "10",
            "after": str(int(before_ts_ms)),
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

        # Rows are newest-first. Find the most recent confirmed (flag=1) row
        # whose open_time_ms is strictly before before_ts_ms.
        for row in rows:
            if not isinstance(row, list) or len(row) < 9:
                raise InvariantError("okx_client_row_invalid")
            ts_ms = int(row[0])
            confirm = int(row[8])
            if ts_ms >= int(before_ts_ms):
                continue  # too recent
            if confirm != 1:
                continue  # not yet closed
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
