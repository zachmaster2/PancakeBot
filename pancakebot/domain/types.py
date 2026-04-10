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
            "wallet": self.wallet_address,
            "amountWei": self.amount_wei,
            "position": self.position,
            "createdAt": self.created_at,
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
            "epoch": self.epoch,
            "startAt": self.start_at,
            "lockAt": self.lock_at,
            "closeAt": self.close_at,
            "lockPrice": self.lock_price,
            "closePrice": self.close_price,
            "position": self.position,
            "failed": self.failed,
            "bets": [b.to_json() for b in self.bets],
        }


