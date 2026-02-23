from __future__ import annotations

import json
from typing import Any

import requests

from pancakebot.domain.types import Kline
from pancakebot.core.errors import InvariantError


_BINANCE_US_BASE_URL = "https://api.binance.us"


class BinanceUsClient:
    """Minimal Binance US Spot REST client (public endpoints only)."""

    def __init__(self, *, timeout_seconds: float) -> None:
        self._timeout_seconds = float(timeout_seconds)

    def fetch_1m_klines(
        self,
        *,
        symbol: str,
        start_time_ms: int,
        end_time_ms: int,
        limit: int = 1000,
    ) -> list[Kline]:
        if limit <= 0 or limit > 1000:
            raise InvariantError("binance_us_limit_invalid")
        if int(end_time_ms) <= int(start_time_ms):
            return []

        url = f"{_BINANCE_US_BASE_URL}/api/v3/klines"
        params = {
            "symbol": str(symbol),
            "interval": "1m",
            "startTime": int(start_time_ms),
            "endTime": int(end_time_ms),
            "limit": int(limit),
        }

        r = requests.get(url, params=params, timeout=self._timeout_seconds)
        if r.status_code != 200:
            raise InvariantError(f"binance_us_http_error: status={r.status_code} body={r.text[:200]}")

        try:
            data: Any = r.json()
        except json.JSONDecodeError as e:
            raise InvariantError(f"binance_us_json_decode_error: {e}")

        if not isinstance(data, list):
            raise InvariantError("binance_us_klines_not_list")

        out: list[Kline] = []
        for row in data:
            # Expected array shape from Binance US:
            # 0 open_time
            # 1 open
            # 2 high
            # 3 low
            # 4 close
            # 5 volume
            # 6 close_time
            # 7 quote_asset_volume
            # 8 number_of_trades
            # 9 taker_buy_base_volume
            # 10 taker_buy_quote_volume
            if not isinstance(row, list) or len(row) < 11:
                raise InvariantError("binance_us_kline_row_invalid")
            out.append(
                Kline(
                    open_time_ms=int(row[0]),
                    close_time_ms=int(row[6]),
                    open_price=float(row[1]),
                    high_price=float(row[2]),
                    low_price=float(row[3]),
                    close_price=float(row[4]),
                    volume=float(row[5]),
                    quote_asset_volume=float(row[7]),
                    number_of_trades=int(row[8]),
                    taker_buy_base_volume=float(row[9]),
                    taker_buy_quote_volume=float(row[10]),
                )
            )

        return out
