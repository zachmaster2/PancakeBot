"""Frozen direct-action runtime policy."""

from __future__ import annotations

from dataclasses import dataclass

from pancakebot.core.constants import GAS_COST_BET_BNB
from pancakebot.core.errors import InvariantError
from pancakebot.domain.features.schema import max_required_prior_context_rounds_size
from pancakebot.domain.strategy.direct_action_policy_model import (
    DirectActionModelBundle,
    DirectActionSpec,
    _base_feature_vector_for_round,
    _direct_action_feature_row_values,
    action_spec_by_id,
    direct_action_score_values,
    direct_action_required_history_rounds,
    load_direct_action_bundle,
    summarize_top_action_predictions,
)
from pancakebot.domain.strategy.candidate_signal import StrategyCandidateSignal
from pancakebot.domain.types import Round


@dataclass(frozen=True, slots=True)
class DirectActionPolicyDecision:
    """One direct-action policy choice for a target round."""

    enabled: bool
    action_id: str
    action_label: str
    action: str
    bet_side: str | None
    bet_size_bnb: float
    score_bnb: float
    q50_net_bnb: float
    top_actions_json: str
    skip_reason: str | None


class DirectActionPolicy:
    """Frozen runtime policy that scores `Skip` plus discrete bet actions."""

    def __init__(
        self,
        *,
        cutoff_seconds: int,
        treasury_fee_fraction: float,
        klines_store_like: object,
        feature_cache_store: object | None = None,
        model_bundle_path: str | None = None,
        bundle: DirectActionModelBundle | None = None,
        top_k_actions: int = 3,
    ) -> None:
        if int(cutoff_seconds) <= 0:
            raise InvariantError("direct_action_cutoff_seconds_nonpositive")
        if float(treasury_fee_fraction) < 0.0 or float(treasury_fee_fraction) >= 1.0:
            raise InvariantError("direct_action_treasury_fee_out_of_range")
        if bundle is None:
            if model_bundle_path is None or str(model_bundle_path).strip() == "":
                raise InvariantError("direct_action_model_bundle_path_missing")
            bundle = load_direct_action_bundle(str(model_bundle_path))
        self._bundle = bundle
        self._cutoff_seconds = int(cutoff_seconds)
        self._treasury_fee_fraction = float(treasury_fee_fraction)
        self._klines_store_like = klines_store_like
        self._feature_cache_store = feature_cache_store
        self._top_k_actions = int(top_k_actions)
        if int(self._top_k_actions) <= 0:
            raise InvariantError("direct_action_top_k_actions_nonpositive")
        self._history_rounds: list[Round] = []
        self._required_history_rounds = int(
            self._bundle.metadata.get("required_history_rounds", direct_action_required_history_rounds())
        )
        if int(self._required_history_rounds) <= 0:
            raise InvariantError("direct_action_bundle_required_history_invalid")
        self._score_mode = str(self._bundle.metadata.get("score_mode", "q10"))
        self._score_risk_lambda = float(self._bundle.metadata.get("score_risk_lambda", 0.0))
        self._legacy_candidate_names = tuple(
            str(name) for name in self._bundle.metadata.get("legacy_candidate_names", [])
        )

    @property
    def enabled(self) -> bool:
        return True

    @property
    def bundle(self) -> DirectActionModelBundle:
        return self._bundle

    def export_bootstrap_state(self) -> dict[str, object]:
        return {
            "history_rounds_json": [round_t.to_json() for round_t in self._history_rounds],
        }

    def import_bootstrap_state(self, *, state: dict[str, object]) -> None:
        raw = list(state.get("history_rounds_json", []))
        history = [Round.from_json(obj) for obj in raw]
        history.sort(key=lambda round_t: int(round_t.epoch))
        deduped: list[Round] = []
        for round_t in history:
            if deduped and int(round_t.epoch) <= int(deduped[-1].epoch):
                continue
            deduped.append(round_t)
        self._history_rounds = list(deduped)
        self._prune_history_rounds()

    def bootstrap_from_closed_rounds(self, *, rounds: list[Round]) -> None:
        self.observe_closed_rounds(rounds=rounds)

    def observe_closed_rounds(self, *, rounds: list[Round]) -> None:
        if not rounds:
            return
        for round_t in sorted(rounds, key=lambda value: int(value.epoch)):
            if self._history_rounds and int(round_t.epoch) <= int(self._history_rounds[-1].epoch):
                continue
            self._history_rounds.append(round_t)
        self._prune_history_rounds()

    def decide_open_round(
        self,
        *,
        round_t: Round,
        bankroll_bnb: float,
        legacy_candidate_signals: dict[str, StrategyCandidateSignal] | None = None,
    ) -> DirectActionPolicyDecision:
        if float(bankroll_bnb) < 0.0:
            raise InvariantError("direct_action_bankroll_negative")
        if len(self._history_rounds) < int(self._required_history_rounds):
            return DirectActionPolicyDecision(
                enabled=True,
                action_id="skip",
                action_label="Skip",
                action="SKIP",
                bet_side=None,
                bet_size_bnb=0.0,
                score_bnb=0.0,
                q50_net_bnb=0.0,
                top_actions_json="[]",
                skip_reason="direct_action_history_insufficient",
            )

        prior_required = int(max_required_prior_context_rounds_size())
        prior_context_rounds = list(self._history_rounds[-int(prior_required) :])
        if len(prior_context_rounds) != int(prior_required):
            raise InvariantError("direct_action_prior_context_rounds_insufficient")
        summary_rounds = list(self._history_rounds[-int(self._required_history_rounds) :])
        base_vector = _base_feature_vector_for_round(
            round_t=round_t,
            prior_context_rounds=prior_context_rounds,
            klines_store_like=self._klines_store_like,
            cutoff_seconds=int(self._cutoff_seconds),
            feature_cache_store=self._feature_cache_store,
        )

        feasible_specs = self._feasible_action_specs(bankroll_bnb=float(bankroll_bnb))
        feature_rows = [
            _direct_action_feature_row_values(
                round_t=round_t,
                summary_rounds=summary_rounds,
                base_vector=base_vector,
                action_spec=spec,
                feature_action_specs=self._bundle.action_specs,
                cutoff_seconds=int(self._cutoff_seconds),
                treasury_fee_fraction=float(self._treasury_fee_fraction),
                legacy_candidate_signals=legacy_candidate_signals,
                legacy_candidate_names=self._legacy_candidate_names,
            )
            for spec in feasible_specs
        ]
        q10_values, q50_values = self._bundle.predict_quantiles(
            feature_rows,
            action_ids=[str(spec.action_id) for spec in feasible_specs],
        )
        score_values = direct_action_score_values(
            q10_values=q10_values,
            q50_values=q50_values,
            score_mode=str(self._score_mode),
            score_risk_lambda=float(self._score_risk_lambda),
        )
        top_actions_json = summarize_top_action_predictions(
            action_specs=feasible_specs,
            score_values=score_values,
            q10_values=q10_values,
            q50_values=q50_values,
            top_k=int(self._top_k_actions),
        )
        best_idx = max(
            range(len(feasible_specs)),
            key=lambda idx: (
                float(score_values[idx]),
                float(q50_values[idx]),
                float(q10_values[idx]),
                str(feasible_specs[idx].action_id),
            ),
        )
        best_spec = feasible_specs[int(best_idx)]
        best_score = float(score_values[int(best_idx)])
        best_q10 = float(q10_values[int(best_idx)])
        best_q50 = float(q50_values[int(best_idx)])

        if str(best_spec.action) == "SKIP":
            reason = "direct_action_best_action_is_skip"
        elif float(best_score) <= 0.0:
            reason = "direct_action_nonpositive_score"
        else:
            reason = None

        if reason is not None:
            skip_spec = action_spec_by_id(self._bundle.action_specs, "skip")
            return DirectActionPolicyDecision(
                enabled=True,
                action_id=str(skip_spec.action_id),
                action_label=str(skip_spec.label),
                action="SKIP",
                bet_side=None,
                bet_size_bnb=0.0,
                score_bnb=float(best_score),
                q50_net_bnb=float(best_q50),
                top_actions_json=str(top_actions_json),
                skip_reason=str(reason),
            )

        return DirectActionPolicyDecision(
            enabled=True,
            action_id=str(best_spec.action_id),
            action_label=str(best_spec.label),
            action="BET",
            bet_side=(None if best_spec.bet_side is None else str(best_spec.bet_side)),
            bet_size_bnb=float(best_spec.bet_size_bnb),
            score_bnb=float(best_score),
            q50_net_bnb=float(best_q50),
            top_actions_json=str(top_actions_json),
            skip_reason=None,
        )

    def _feasible_action_specs(self, *, bankroll_bnb: float) -> list[DirectActionSpec]:
        out: list[DirectActionSpec] = []
        for spec in self._bundle.action_specs:
            if str(spec.action) == "SKIP":
                out.append(spec)
                continue
            total_cost = float(spec.bet_size_bnb) + float(GAS_COST_BET_BNB)
            if float(total_cost) <= float(bankroll_bnb):
                out.append(spec)
        if not out:
            raise InvariantError("direct_action_feasible_specs_empty")
        return out

    def _prune_history_rounds(self) -> None:
        keep = max(int(self._required_history_rounds), int(max_required_prior_context_rounds_size()))
        if len(self._history_rounds) <= int(keep):
            return
        self._history_rounds = list(self._history_rounds[-int(keep) :])
