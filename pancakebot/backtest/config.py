from __future__ import annotations

from dataclasses import dataclass

from pancakebot.errors import InvariantError

_BACKTEST_RESET_MODES = ("continuous", "chunk_reset")


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    """Backtest configuration."""

    simulation_size: int
    initial_bankroll_bnb: float
    reset_mode: str = "continuous"
    reset_every_rounds: int = 0
    tail_offset_rounds: int = 0
    # Optional epoch range filter. When set, overrides simulation_size windowing.
    epoch_start: int | None = None
    epoch_end: int | None = None

    def validate(self) -> None:
        if not isinstance(self.simulation_size, int):
            raise InvariantError("backtest_simulation_size_not_int")
        if self.simulation_size <= 0:
            raise InvariantError("backtest_simulation_size_must_be_positive")

        if not isinstance(self.initial_bankroll_bnb, (int, float)):
            raise InvariantError("backtest_initial_bankroll_bnb_not_number")
        if self.initial_bankroll_bnb <= 0.0:
            raise InvariantError("backtest_initial_bankroll_bnb_must_be_positive")

        if not isinstance(self.reset_mode, str):
            raise InvariantError("backtest_reset_mode_not_str")
        mode = self.reset_mode.strip()
        if mode not in _BACKTEST_RESET_MODES:
            raise InvariantError("backtest_reset_mode_invalid")

        if not isinstance(self.reset_every_rounds, int):
            raise InvariantError("backtest_reset_every_rounds_not_int")
        if self.reset_every_rounds < 0:
            raise InvariantError("backtest_reset_every_rounds_negative")
        if mode == "chunk_reset" and self.reset_every_rounds <= 0:
            raise InvariantError("backtest_chunk_reset_every_rounds_must_be_positive")

        if not isinstance(self.tail_offset_rounds, int):
            raise InvariantError("backtest_tail_offset_rounds_not_int")
        if self.tail_offset_rounds < 0:
            raise InvariantError("backtest_tail_offset_rounds_negative")

        if self.epoch_start is not None and self.epoch_end is not None:
            if self.epoch_start > self.epoch_end:
                raise InvariantError("backtest_epoch_start_after_epoch_end")
