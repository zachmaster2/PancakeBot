"""Minimal OKX public REST client for 1s klines.

OKX is accessible from US IPs for unauthenticated market data.
No API key required.
"""

from __future__ import annotations

import json

import requests

from pancakebot.core.errors import InvariantError


_OKX_BASE_URL = "https://www.okx.com"


class OkxClient:
    """Minimal OKX Spot public REST client (unauthenticated)."""

    def __init__(self, *, timeout_seconds: float) -> None:
        self._timeout_seconds = timeout_seconds

    def fetch_1s_klines(
        self,
        *,
        symbol: str,
        count: int = 25,
        after_ms: int | None = None,
    ) -> list[dict[str, float | int]] | None:
        """Fetch the most recent `count` 1s klines from OKX.

        When *after_ms* is provided, only candles with open_time < after_ms
        are returned (OKX ``after`` pagination parameter).  This excludes
        the in-progress bar whose open_time equals the current second,
        so all returned candles are completed with final close prices.

        Without *after_ms*, the response includes the current in-progress
        1s bar (whose close price is an ephemeral mid-second snapshot).

        Returns oldest-first list of dicts with keys:
          open_time_ms, close_price

        Returns None on failure.
        """
        url = f"{_OKX_BASE_URL}/api/v5/market/candles"
        params: dict[str, str] = {
            "instId": symbol,
            "bar": "1s",
            "limit": str(count),
        }
        if after_ms is not None:
            params["after"] = str(after_ms)

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
            raise InvariantError(f"okx_client_1s_api_error: code={code} msg={payload.get('msg', '')}")

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
