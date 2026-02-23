from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

from pancakebot.core.errors import InvariantError


@dataclass(frozen=True, slots=True)
class Bet:
    """A bet object as returned by The Graph, normalized at ingestion.

    - amount_wei is always an integer wei amount (BNB has 18 decimals).
    - position is normalized to the canonical capitalized form used by the subgraph.
    - bet.user.id is mapped to wallet_address.
    """

    wallet_address: str
    amount_wei: int
    position: Literal["Bull", "Bear"]
    created_at: int

    @staticmethod
    def from_json(obj: dict[str, Any]) -> "Bet":
        position = str(obj["position"])
        if position not in ("Bull", "Bear"):
            raise InvariantError("bet_position_invalid")
        return Bet(
            wallet_address=str(obj["wallet"]),
            amount_wei=int(obj["amountWei"]),
            position=position,
            created_at=int(obj["createdAt"]),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "wallet": str(self.wallet_address),
            "amountWei": int(self.amount_wei),
            "position": str(self.position),
            "createdAt": int(self.created_at),
        }


@dataclass(frozen=True, slots=True)
class Round:
    """A round object as returned by The Graph (normalized).

    Pool amounts (total/bull/bear) are intentionally NOT fetched from Graph.
    They are computed from bets in feature-building code.

    State shapes (Graph invariants; enforced by state-specific queries):
      - Current (aka open): lockAt/lockPrice/closeAt/closePrice/failed/position are null
      - Locked: lockAt/lockPrice non-null; closeAt/closePrice/failed/position null
      - Closed usable: failed == false; startAt/lockAt/closeAt and prices non-null; position in Bull/Bear/House
    """

    epoch: int
    start_at: int
    lock_at: int | None
    close_at: int | None
    lock_price: float | None
    close_price: float | None
    position: str | None
    failed: bool | None
    bets: Sequence[Bet]

    @staticmethod
    def from_json(obj: dict[str, Any]) -> "Round":
        bets_raw = obj.get("bets") or []
        if not isinstance(bets_raw, list):
            raise InvariantError("round_bets_not_list")
        bets = tuple(Bet.from_json(b) for b in bets_raw)
        return Round(
            epoch=int(obj["epoch"]),
            start_at=int(obj["startAt"]),
            lock_at=None if obj.get("lockAt") is None else int(obj["lockAt"]),
            close_at=None if obj.get("closeAt") is None else int(obj["closeAt"]),
            lock_price=None if obj.get("lockPrice") is None else float(obj["lockPrice"]),
            close_price=None if obj.get("closePrice") is None else float(obj["closePrice"]),
            position=obj.get("position"),
            failed=obj.get("failed"),
            bets=bets,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "epoch": int(self.epoch),
            "startAt": int(self.start_at),
            "lockAt": int(self.lock_at) if self.lock_at is not None else None,
            "closeAt": int(self.close_at) if self.close_at is not None else None,
            "lockPrice": self.lock_price,
            "closePrice": self.close_price,
            "position": self.position,
            "failed": self.failed,
            "bets": [b.to_json() for b in self.bets],
        }


@dataclass(frozen=True, slots=True)
class Kline:
    """A Binance US Spot kline (candle) record (fully CLOSED).

    Times are in milliseconds since epoch.
    """

    open_time_ms: int
    close_time_ms: int
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float
    quote_asset_volume: float
    number_of_trades: int
    taker_buy_base_volume: float
    taker_buy_quote_volume: float

    @staticmethod
    def from_json(rec: dict[str, Any]) -> "Kline":
        try:
            return Kline(
                open_time_ms=int(rec["open_time_ms"]),
                close_time_ms=int(rec["close_time_ms"]),
                open_price=float(rec["open_price"]),
                high_price=float(rec["high_price"]),
                low_price=float(rec["low_price"]),
                close_price=float(rec["close_price"]),
                volume=float(rec["volume"]),
                quote_asset_volume=float(rec["quote_asset_volume"]),
                number_of_trades=int(rec["number_of_trades"]),
                taker_buy_base_volume=float(rec["taker_buy_base_volume"]),
                taker_buy_quote_volume=float(rec["taker_buy_quote_volume"]),
            )
        except Exception as e:
            raise InvariantError(f"kline_parse_error: {e}")

    def to_json(self) -> dict[str, Any]:
        return {
            "open_time_ms": int(self.open_time_ms),
            "close_time_ms": int(self.close_time_ms),
            "open_price": float(self.open_price),
            "high_price": float(self.high_price),
            "low_price": float(self.low_price),
            "close_price": float(self.close_price),
            "volume": float(self.volume),
            "quote_asset_volume": float(self.quote_asset_volume),
            "number_of_trades": int(self.number_of_trades),
            "taker_buy_base_volume": float(self.taker_buy_base_volume),
            "taker_buy_quote_volume": float(self.taker_buy_quote_volume),
        }
