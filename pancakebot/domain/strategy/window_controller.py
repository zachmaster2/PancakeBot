"""Causal window-level profile controller for shared pipeline modes."""

from __future__ import annotations

from dataclasses import dataclass, replace

from pancakebot.config.strategy_config import WindowControllerConfig
from pancakebot.core.errors import InvariantError
from pancakebot.domain.strategy.candidate_signal import StrategyCandidateSignal
from pancakebot.domain.types import Round

_VALID_MODES = frozenset(
    {
        "trailing_best_vs_baseline",
        "trailing_best_vs_baseline_with_skip",
    }
)


@dataclass(frozen=True, slots=True)
class WindowControllerDecision:
    """One causal window-controller choice for a target round."""

    enabled: bool
    mode: str | None
    window_index: int | None
    baseline_profile_name: str | None
    alternate_profile_name: str | None
    selected_profile_name: str | None
    selected_action: str
    estimated_baseline_per_500: float | None
    estimated_alternate_per_500: float | None
    estimated_selected_per_500: float | None
    estimated_selected_bet_rate: float | None
    lookback_windows_used: int


class WindowController:
    """Shared window-controller state used by live/dry/backtest pipeline."""

    def __init__(self, *, config: WindowControllerConfig) -> None:
        if int(config.window_rounds) <= 0:
            raise InvariantError("window_controller_window_rounds_nonpositive")
        if int(config.lookback_windows) <= 0:
            raise InvariantError("window_controller_lookback_windows_nonpositive")
        if str(config.mode) not in _VALID_MODES:
            raise InvariantError("window_controller_mode_invalid")
        if str(config.baseline_profile_name) == str(config.alternate_profile_name):
            raise InvariantError("window_controller_profiles_must_differ")

        self._config = config
        self._anchor_epoch: int | None = None
        self._round_counts_by_window: dict[int, int] = {}
        self._profit_by_window: dict[int, dict[str, float]] = {}
        self._bet_counts_by_window: dict[int, dict[str, int]] = {}

    @property
    def enabled(self) -> bool:
        return bool(self._config.enabled)

    @property
    def mode(self) -> str:
        return str(self._config.mode)

    def export_bootstrap_state(self) -> dict[str, object]:
        return {
            "anchor_epoch": (None if self._anchor_epoch is None else int(self._anchor_epoch)),
            "round_counts_by_window": {str(k): int(v) for k, v in self._round_counts_by_window.items()},
            "profit_by_window": {
                str(window_idx): {str(name): float(value) for name, value in profits.items()}
                for window_idx, profits in self._profit_by_window.items()
            },
            "bet_counts_by_window": {
                str(window_idx): {str(name): int(value) for name, value in counts.items()}
                for window_idx, counts in self._bet_counts_by_window.items()
            },
        }

    def import_bootstrap_state(self, *, state: dict[str, object]) -> None:
        anchor_epoch = state.get("anchor_epoch")
        self._anchor_epoch = None if anchor_epoch is None else int(anchor_epoch)
        self._round_counts_by_window = {
            int(window_idx): int(count)
            for window_idx, count in dict(state.get("round_counts_by_window", {})).items()
        }
        self._profit_by_window = {
            int(window_idx): {
                str(name): float(value)
                for name, value in dict(profits).items()
            }
            for window_idx, profits in dict(state.get("profit_by_window", {})).items()
        }
        self._bet_counts_by_window = {
            int(window_idx): {
                str(name): int(value)
                for name, value in dict(counts).items()
            }
            for window_idx, counts in dict(state.get("bet_counts_by_window", {})).items()
        }

    def observe_round_settlement(
        self,
        *,
        round_t: Round,
        candidate_signals: dict[str, StrategyCandidateSignal],
        realized_profit_by_candidate: dict[str, float],
    ) -> None:
        if not bool(self.enabled):
            return
        epoch = int(round_t.epoch)
        window_idx = self.window_index_for_epoch(epoch)
        self._ensure_profile_signals(candidate_signals=candidate_signals)
        round_count = int(self._round_counts_by_window.get(int(window_idx), 0)) + 1
        self._round_counts_by_window[int(window_idx)] = int(round_count)
        if int(round_count) > int(self._config.window_rounds):
            raise InvariantError("window_controller_round_count_exceeded")

        profit_row = self._profit_by_window.setdefault(int(window_idx), {})
        bet_row = self._bet_counts_by_window.setdefault(int(window_idx), {})
        for profile_name in self._profile_names():
            signal = candidate_signals[str(profile_name)]
            if str(profile_name) not in realized_profit_by_candidate:
                raise InvariantError("window_controller_realized_profit_missing")
            profit_row[str(profile_name)] = float(profit_row.get(str(profile_name), 0.0)) + float(
                realized_profit_by_candidate[str(profile_name)]
            )
            if str(signal.action) == "BET" and float(signal.bet_size_bnb) > 0.0:
                bet_row[str(profile_name)] = int(bet_row.get(str(profile_name), 0)) + 1
            else:
                bet_row.setdefault(str(profile_name), int(bet_row.get(str(profile_name), 0)))

    def decision_for_round(
        self,
        *,
        round_t: Round,
        candidate_signals: dict[str, StrategyCandidateSignal],
    ) -> WindowControllerDecision:
        if not bool(self.enabled):
            return WindowControllerDecision(
                enabled=False,
                mode=None,
                window_index=None,
                baseline_profile_name=None,
                alternate_profile_name=None,
                selected_profile_name=None,
                selected_action="off",
                estimated_baseline_per_500=None,
                estimated_alternate_per_500=None,
                estimated_selected_per_500=None,
                estimated_selected_bet_rate=None,
                lookback_windows_used=0,
            )

        self._ensure_profile_signals(candidate_signals=candidate_signals)
        window_idx = self.window_index_for_epoch(int(round_t.epoch))
        completed_window_indices = [
            int(idx)
            for idx, count in self._round_counts_by_window.items()
            if int(idx) < int(window_idx) and int(count) == int(self._config.window_rounds)
        ]
        completed_window_indices.sort()

        baseline_name = str(self._config.baseline_profile_name)
        alternate_name = str(self._config.alternate_profile_name)
        lookback = min(int(self._config.lookback_windows), int(len(completed_window_indices)))

        baseline_mean = 0.0
        alternate_mean = 0.0
        selected_profile_name = str(baseline_name)
        selected_action = "profile"
        selected_per_500 = 0.0
        selected_bet_rate = 0.0

        if int(lookback) > 0:
            hist = completed_window_indices[-int(lookback) :]
            baseline_mean = self._mean_per_500(profile_name=str(baseline_name), window_indices=hist)
            alternate_mean = self._mean_per_500(profile_name=str(alternate_name), window_indices=hist)
            if (
                str(self._config.mode) == "trailing_best_vs_baseline_with_skip"
                and max(float(baseline_mean), float(alternate_mean)) <= float(self._config.skip_threshold_per_500)
            ):
                selected_profile_name = None
                selected_action = "skip"
                selected_per_500 = 0.0
                selected_bet_rate = 0.0
            elif float(alternate_mean - baseline_mean) > float(self._config.margin_per_500):
                selected_profile_name = str(alternate_name)
                selected_per_500 = float(alternate_mean)
                selected_bet_rate = self._mean_bet_rate(profile_name=str(alternate_name), window_indices=hist)
            else:
                selected_profile_name = str(baseline_name)
                selected_per_500 = float(baseline_mean)
                selected_bet_rate = self._mean_bet_rate(profile_name=str(baseline_name), window_indices=hist)

        return WindowControllerDecision(
            enabled=True,
            mode=str(self._config.mode),
            window_index=int(window_idx),
            baseline_profile_name=str(baseline_name),
            alternate_profile_name=str(alternate_name),
            selected_profile_name=(None if selected_profile_name is None else str(selected_profile_name)),
            selected_action=str(selected_action),
            estimated_baseline_per_500=float(baseline_mean),
            estimated_alternate_per_500=float(alternate_mean),
            estimated_selected_per_500=float(selected_per_500),
            estimated_selected_bet_rate=float(selected_bet_rate),
            lookback_windows_used=int(lookback),
        )

    def apply_to_candidate_signals(
        self,
        *,
        candidate_signals: dict[str, StrategyCandidateSignal],
        decision: WindowControllerDecision,
    ) -> dict[str, StrategyCandidateSignal]:
        if not bool(decision.enabled):
            return dict(candidate_signals)
        self._ensure_profile_signals(candidate_signals=candidate_signals)
        if str(decision.selected_action) == "skip":
            return {
                str(name): self._veto_signal(
                    signal=signal,
                    skip_reason="window_controller_skip",
                )
                for name, signal in candidate_signals.items()
            }
        if decision.selected_profile_name is None:
            raise InvariantError("window_controller_selected_profile_missing")
        selected_profile_name = str(decision.selected_profile_name)
        out: dict[str, StrategyCandidateSignal] = {}
        for candidate_name, signal in candidate_signals.items():
            if str(candidate_name) == str(selected_profile_name):
                out[str(candidate_name)] = signal
            else:
                out[str(candidate_name)] = self._veto_signal(
                    signal=signal,
                    skip_reason="window_controller_profile_masked",
                )
        return out

    def window_index_for_epoch(self, epoch: int) -> int:
        if self._anchor_epoch is None:
            self._anchor_epoch = int(epoch)
        if int(epoch) < int(self._anchor_epoch):
            raise InvariantError("window_controller_epoch_before_anchor")
        return int((int(epoch) - int(self._anchor_epoch)) // int(self._config.window_rounds))

    def _profile_names(self) -> tuple[str, str]:
        return (
            str(self._config.baseline_profile_name),
            str(self._config.alternate_profile_name),
        )

    def _ensure_profile_signals(
        self,
        *,
        candidate_signals: dict[str, StrategyCandidateSignal],
    ) -> None:
        for profile_name in self._profile_names():
            if str(profile_name) not in candidate_signals:
                raise InvariantError(f"window_controller_profile_signal_missing: {profile_name}")

    def _mean_per_500(self, *, profile_name: str, window_indices: list[int]) -> float:
        if not window_indices:
            return 0.0
        vals: list[float] = []
        for window_idx in window_indices:
            rounds = int(self._round_counts_by_window.get(int(window_idx), 0))
            if int(rounds) <= 0:
                raise InvariantError("window_controller_window_rounds_missing")
            profit = float(self._profit_by_window[int(window_idx)].get(str(profile_name), 0.0))
            vals.append(float(profit) * 500.0 / float(rounds))
        return float(sum(vals) / float(len(vals)))

    def _mean_bet_rate(self, *, profile_name: str, window_indices: list[int]) -> float:
        if not window_indices:
            return 0.0
        vals: list[float] = []
        for window_idx in window_indices:
            rounds = int(self._round_counts_by_window.get(int(window_idx), 0))
            if int(rounds) <= 0:
                raise InvariantError("window_controller_window_rounds_missing")
            bet_count = int(self._bet_counts_by_window[int(window_idx)].get(str(profile_name), 0))
            vals.append(float(bet_count) / float(rounds))
        return float(sum(vals) / float(len(vals)))

    @staticmethod
    def _veto_signal(*, signal: StrategyCandidateSignal, skip_reason: str) -> StrategyCandidateSignal:
        if str(signal.action) != "BET":
            return replace(signal, skip_reason=str(skip_reason))
        return replace(
            signal,
            action="SKIP",
            bet_side=None,
            bet_size_bnb=0.0,
            expected_profit_bnb=None,
            selector_score_bnb=None,
            skip_reason=str(skip_reason),
        )
