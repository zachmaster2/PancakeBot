from __future__ import annotations

from dataclasses import dataclass

from pancakebot.domain.types import Round
from pancakebot.domain.models.walk_forward import WalkForwardState, ensure_state
from pancakebot.core.errors import InvariantError


@dataclass(slots=True)
class ModelManager:
    """Thin runtime wrapper around the walk-forward owner."""

    state: WalkForwardState | None = None

    def step(self, *, cfg, closed_rounds: list[Round], current_epoch: int) -> WalkForwardState:
        if current_epoch <= 0:
            raise InvariantError("current_epoch_invalid")

        self.state = ensure_state(cfg=cfg, closed_rounds=closed_rounds, current_epoch=int(current_epoch), state=self.state)
        if self.state.models is None:
            raise InvariantError("model_manager_models_missing")
        return self.state
