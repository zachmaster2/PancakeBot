"""Causal absolute window-level profile controller for shared pipeline modes."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
import json
import math

from pancakebot.config.strategy_config import WindowControllerConfig
from pancakebot.core.errors import InvariantError
from pancakebot.domain.strategy.candidate_signal import StrategyCandidateSignal
from pancakebot.domain.types import Round

_VALID_MODES = frozenset({"absolute_best_with_skip"})
_VALID_ESTIMATOR_MODES = frozenset({"trailing_mean", "ewm_mean"})


@dataclass(frozen=True, slots=True)
class WindowControllerDecision:
    """One causal window-controller choice for a target round."""

    enabled: bool
    mode: str | None
    estimator_mode: str | None
    window_index: int | None
    profile_names_json: str | None
    selected_profile_name: str | None
    selected_action: str
    estimated_selected_per_500: float | None
    estimated_selected_score_per_500: float | None
    estimated_selected_bet_rate: float | None
    estimated_profiles_per_500_json: str | None
    estimated_profiles_score_per_500_json: str | None
    estimated_profiles_bet_rate_json: str | None
    lookback_windows_used: int


class WindowController:
    """Shared window-controller state used by live/dry/backtest pipeline."""

    def __init__(self, *, config: WindowControllerConfig) -> None:
        if int(config.window_rounds) <= 0:
            raise InvariantError("window_controller_window_rounds_nonpositive")
        if int(config.lookback_windows) <= 0:
            raise InvariantError("window_controller_lookback_windows_nonpositive")
        if int(config.min_history_windows) <= 0:
            raise InvariantError("window_controller_min_history_windows_nonpositive")
        if int(config.min_history_windows) > int(config.lookback_windows):
            raise InvariantError("window_controller_min_history_windows_exceeds_lookback")
        if str(config.mode) not in _VALID_MODES:
            raise InvariantError("window_controller_mode_invalid")
        if str(config.estimator_mode) not in _VALID_ESTIMATOR_MODES:
            raise InvariantError("window_controller_estimator_mode_invalid")
        if float(config.ewm_alpha) <= 0.0 or float(config.ewm_alpha) > 1.0:
            raise InvariantError("window_controller_ewm_alpha_out_of_range")
        if float(config.stability_penalty_per_500) < 0.0:
            raise InvariantError("window_controller_stability_penalty_negative")
        if float(config.activity_target_bet_rate) < 0.0 or float(config.activity_target_bet_rate) > 1.0:
            raise InvariantError("window_controller_activity_target_bet_rate_out_of_range")
        if float(config.activity_shortfall_penalty_per_500) < 0.0:
            raise InvariantError("window_controller_activity_shortfall_penalty_negative")
        if not tuple(config.profile_names):
            raise InvariantError("window_controller_profile_names_empty")
        if len(set(str(name) for name in config.profile_names)) != len(tuple(config.profile_names)):
            raise InvariantError("window_controller_profile_names_duplicate")
        if str(config.cold_start_profile_name) not in {str(name) for name in config.profile_names}:
            raise InvariantError("window_controller_cold_start_profile_missing")

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
            int(window_idx): {str(name): float(value) for name, value in dict(profits).items()}
            for window_idx, profits in dict(state.get("profit_by_window", {})).items()
        }
        self._bet_counts_by_window = {
            int(window_idx): {str(name): int(value) for name, value in dict(counts).items()}
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
                estimator_mode=None,
                window_index=None,
                profile_names_json=None,
                selected_profile_name=None,
                selected_action="off",
                estimated_selected_per_500=None,
                estimated_selected_score_per_500=None,
                estimated_selected_bet_rate=None,
                estimated_profiles_per_500_json=None,
                estimated_profiles_score_per_500_json=None,
                estimated_profiles_bet_rate_json=None,
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

        lookback_used = min(int(self._config.lookback_windows), int(len(completed_window_indices)))
        if int(lookback_used) < int(self._config.min_history_windows):
            return WindowControllerDecision(
                enabled=True,
                mode=str(self._config.mode),
                estimator_mode=str(self._config.estimator_mode),
                window_index=int(window_idx),
                profile_names_json=json.dumps(list(self._profile_names()), separators=(",", ":")),
                selected_profile_name=str(self._config.cold_start_profile_name),
                selected_action="profile",
                estimated_selected_per_500=0.0,
                estimated_selected_score_per_500=0.0,
                estimated_selected_bet_rate=0.0,
                estimated_profiles_per_500_json="{}",
                estimated_profiles_score_per_500_json="{}",
                estimated_profiles_bet_rate_json="{}",
                lookback_windows_used=int(lookback_used),
            )

        hist = completed_window_indices[-int(lookback_used) :]
        estimates: dict[str, float] = {}
        score_estimates: dict[str, float] = {}
        bet_rate_estimates: dict[str, float] = {}
        for profile_name in self._profile_names():
            estimate, bet_rate_estimate = self._estimate_profile(
                profile_name=str(profile_name),
                window_indices=hist,
            )
            estimates[str(profile_name)] = float(estimate)
            bet_rate_estimates[str(profile_name)] = float(bet_rate_estimate)
            score_estimates[str(profile_name)] = float(
                self._score_profile(
                    estimate=float(estimate),
                    bet_rate_estimate=float(bet_rate_estimate),
                )
            )

        selected_profile_name = None
        selected_action = "skip"
        selected_per_500 = 0.0
        selected_score_per_500 = 0.0
        selected_bet_rate = 0.0
        if score_estimates:
            best_profile_name, best_score = max(score_estimates.items(), key=lambda item: float(item[1]))
            if float(best_score) > float(self._config.skip_threshold_per_500):
                selected_profile_name = str(best_profile_name)
                selected_action = "profile"
                selected_per_500 = float(estimates[str(best_profile_name)])
                selected_score_per_500 = float(best_score)
                selected_bet_rate = float(bet_rate_estimates[str(best_profile_name)])

        return WindowControllerDecision(
            enabled=True,
            mode=str(self._config.mode),
            estimator_mode=str(self._config.estimator_mode),
            window_index=int(window_idx),
            profile_names_json=json.dumps(list(self._profile_names()), separators=(",", ":")),
            selected_profile_name=(None if selected_profile_name is None else str(selected_profile_name)),
            selected_action=str(selected_action),
            estimated_selected_per_500=float(selected_per_500),
            estimated_selected_score_per_500=float(selected_score_per_500),
            estimated_selected_bet_rate=float(selected_bet_rate),
            estimated_profiles_per_500_json=json.dumps(
                {str(name): float(value) for name, value in sorted(estimates.items())},
                separators=(",", ":"),
                sort_keys=True,
            ),
            estimated_profiles_score_per_500_json=json.dumps(
                {str(name): float(value) for name, value in sorted(score_estimates.items())},
                separators=(",", ":"),
                sort_keys=True,
            ),
            estimated_profiles_bet_rate_json=json.dumps(
                {str(name): float(value) for name, value in sorted(bet_rate_estimates.items())},
                separators=(",", ":"),
                sort_keys=True,
            ),
            lookback_windows_used=int(lookback_used),
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
                str(name): self._veto_signal(signal=signal, skip_reason="window_controller_skip")
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

    def _profile_names(self) -> tuple[str, ...]:
        return tuple(str(name) for name in self._config.profile_names)

    def _ensure_profile_signals(
        self,
        *,
        candidate_signals: dict[str, StrategyCandidateSignal],
    ) -> None:
        for profile_name in self._profile_names():
            if str(profile_name) not in candidate_signals:
                raise InvariantError(f"window_controller_profile_signal_missing: {profile_name}")

    def _estimate_profile(
        self,
        *,
        profile_name: str,
        window_indices: Iterable[int],
    ) -> tuple[float, float]:
        values = [self._window_per_500(profile_name=str(profile_name), window_idx=int(window_idx)) for window_idx in window_indices]
        bet_rates = [self._window_bet_rate(profile_name=str(profile_name), window_idx=int(window_idx)) for window_idx in window_indices]
        if not values:
            return 0.0, 0.0
        if str(self._config.estimator_mode) == "trailing_mean":
            raw_estimate = float(sum(values) / float(len(values)))
            bet_rate_estimate = float(sum(bet_rates) / float(len(bet_rates)))
        elif str(self._config.estimator_mode) == "ewm_mean":
            weights = self._ewm_weights(length=int(len(values)), alpha=float(self._config.ewm_alpha))
            raw_estimate = float(sum(float(value) * float(weight) for value, weight in zip(values, weights)))
            bet_rate_estimate = float(sum(float(value) * float(weight) for value, weight in zip(bet_rates, weights)))
        else:
            raise InvariantError("window_controller_estimator_mode_invalid")
        stability_penalty = 0.0
        if len(values) > 1 and float(self._config.stability_penalty_per_500) > 0.0:
            stability_penalty = float(self._config.stability_penalty_per_500) * self._std(values)
        return float(raw_estimate - stability_penalty), float(bet_rate_estimate)

    def _score_profile(self, *, estimate: float, bet_rate_estimate: float) -> float:
        penalty = 0.0
        if (
            float(self._config.activity_target_bet_rate) > 0.0
            and float(self._config.activity_shortfall_penalty_per_500) > 0.0
        ):
            shortfall = max(0.0, float(self._config.activity_target_bet_rate) - float(bet_rate_estimate))
            penalty = float(self._config.activity_shortfall_penalty_per_500) * float(shortfall)
        return float(estimate) - float(penalty)

    def _window_per_500(self, *, profile_name: str, window_idx: int) -> float:
        rounds = int(self._round_counts_by_window.get(int(window_idx), 0))
        if int(rounds) <= 0:
            raise InvariantError("window_controller_window_rounds_missing")
        profit = float(self._profit_by_window[int(window_idx)].get(str(profile_name), 0.0))
        return float(profit) * 500.0 / float(rounds)

    def _window_bet_rate(self, *, profile_name: str, window_idx: int) -> float:
        rounds = int(self._round_counts_by_window.get(int(window_idx), 0))
        if int(rounds) <= 0:
            raise InvariantError("window_controller_window_rounds_missing")
        bet_count = int(self._bet_counts_by_window[int(window_idx)].get(str(profile_name), 0))
        return float(bet_count) / float(rounds)

    @staticmethod
    def _ewm_weights(*, length: int, alpha: float) -> list[float]:
        weights = [float(alpha) ** float(int(length) - 1 - idx) for idx in range(int(length))]
        total = float(sum(weights))
        if float(total) <= 0.0:
            raise InvariantError("window_controller_ewm_weights_nonpositive")
        return [float(weight / total) for weight in weights]

    @staticmethod
    def _std(values: list[float]) -> float:
        mean_value = float(sum(values) / float(len(values)))
        variance = float(sum((float(value) - float(mean_value)) ** 2 for value in values) / float(len(values)))
        return math.sqrt(float(variance))

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
