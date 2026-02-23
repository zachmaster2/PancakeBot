from __future__ import annotations

from dataclasses import dataclass

from pancakebot.core.errors import InvariantError


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    """Backtest configuration."""

    simulation_size: int
    initial_bankroll_bnb: float

    def validate(self) -> None:
        if not isinstance(self.simulation_size, int):
            raise InvariantError("backtest_simulation_size_not_int")
        if self.simulation_size <= 0:
            raise InvariantError("backtest_simulation_size_must_be_positive")

        if not isinstance(self.initial_bankroll_bnb, (int, float)):
            raise InvariantError("backtest_initial_bankroll_bnb_not_number")
        if float(self.initial_bankroll_bnb) <= 0.0:
            raise InvariantError("backtest_initial_bankroll_bnb_must_be_positive")
