from __future__ import annotations

from dataclasses import dataclass

from pancakebot.core.errors import InvariantError

_BACKTEST_RESET_MODES = ("continuous", "chunk_reset")
_BACKTEST_ROUTER_MODES = ("selector_max_score", "skip_only", "oracle_skip")


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    """Backtest configuration."""

    simulation_size: int
    initial_bankroll_bnb: float
    reset_mode: str = "continuous"
    reset_every_rounds: int = 0
    router_mode: str = "selector_max_score"
    router_score_threshold_bnb: float = -1e9

    def validate(self) -> None:
        if not isinstance(self.simulation_size, int):
            raise InvariantError("backtest_simulation_size_not_int")
        if self.simulation_size <= 0:
            raise InvariantError("backtest_simulation_size_must_be_positive")

        if not isinstance(self.initial_bankroll_bnb, (int, float)):
            raise InvariantError("backtest_initial_bankroll_bnb_not_number")
        if float(self.initial_bankroll_bnb) <= 0.0:
            raise InvariantError("backtest_initial_bankroll_bnb_must_be_positive")

        if not isinstance(self.reset_mode, str):
            raise InvariantError("backtest_reset_mode_not_str")
        mode = str(self.reset_mode).strip()
        if mode not in _BACKTEST_RESET_MODES:
            raise InvariantError("backtest_reset_mode_invalid")

        if not isinstance(self.reset_every_rounds, int):
            raise InvariantError("backtest_reset_every_rounds_not_int")
        if int(self.reset_every_rounds) < 0:
            raise InvariantError("backtest_reset_every_rounds_negative")
        if mode == "chunk_reset" and int(self.reset_every_rounds) <= 0:
            raise InvariantError("backtest_chunk_reset_every_rounds_must_be_positive")

        if not isinstance(self.router_mode, str):
            raise InvariantError("backtest_router_mode_not_str")
        if str(self.router_mode).strip() not in _BACKTEST_ROUTER_MODES:
            raise InvariantError("backtest_router_mode_invalid")

        if not isinstance(self.router_score_threshold_bnb, (int, float)):
            raise InvariantError("backtest_router_score_threshold_bnb_not_number")
