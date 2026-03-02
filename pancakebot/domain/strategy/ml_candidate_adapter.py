"""ML strategy candidate adapter for the shared router pipeline.

This adapter reuses the active feature builder and walk-forward model stack to
emit one router-compatible candidate signal per round.
"""

from __future__ import annotations

from dataclasses import dataclass

from pancakebot.config.strategy_config import MlCandidateConfig
from pancakebot.core.constants import BNB_WEI, GAS_COST_BET_BNB, GAS_COST_CLAIM_BNB
from pancakebot.core.errors import InvariantError
from pancakebot.domain.features.feature_builder import build_features, vectorize
from pancakebot.domain.features.pool_amounts import compute_pool_amounts_wei_at_or_before
from pancakebot.domain.features.schema import FEATURE_SCHEMA, max_required_prior_context_rounds_size
from pancakebot.domain.models.walk_forward import (
    WalkForwardState,
    ensure_state,
    predict_probabilities,
    predict_tradeable_probability,
)
from pancakebot.domain.strategy.candidate_signal import StrategyCandidateSignal
from pancakebot.domain.types import Kline, Round


@dataclass(frozen=True, slots=True)
class _MlWalkForwardRuntimeConfig:
    """Runtime config object consumed by walk-forward model owner."""

    klines_store: object
    cutoff_seconds: int
    train_size: int
    calibrate_size: int
    retrain_interval: int
    recalibrate_interval: int
    price_alpha: float
    pool_alpha_total: float
    pool_alpha_ratio: float
    random_seed: int
    recency_weight_floor: float
    recency_weight_power: float
    predictability_baseline_bet_bnb: float
    treasury_fee_fraction: float
    feature_cache_store: object | None


def _expected_net_from_predicted_final(
    *,
    p_bull: float,
    side: str,
    stake_bnb: float,
    final_bull_bnb: float,
    final_bear_bnb: float,
    treasury_fee_fraction: float,
) -> float:
    """Compute impact-aware expected net including bet/claim gas."""

    side_u = str(side).upper()
    if side_u not in ("BULL", "BEAR"):
        raise InvariantError("ml_side_invalid")
    if not (0.0 <= float(p_bull) <= 1.0):
        raise InvariantError("ml_p_bull_out_of_range")
    if float(stake_bnb) <= 0.0:
        raise InvariantError("ml_stake_nonpositive")
    if float(final_bull_bnb) <= 0.0 or float(final_bear_bnb) <= 0.0:
        return float("-inf")
    if not (0.0 <= float(treasury_fee_fraction) < 1.0):
        raise InvariantError("ml_treasury_fee_out_of_range")

    final_total_bnb = float(final_bull_bnb) + float(final_bear_bnb)
    if side_u == "BULL":
        adj_side_bnb = float(final_bull_bnb) + float(stake_bnb)
        p_win = float(p_bull)
    else:
        adj_side_bnb = float(final_bear_bnb) + float(stake_bnb)
        p_win = 1.0 - float(p_bull)
    adj_total_bnb = float(final_total_bnb) + float(stake_bnb)
    if float(adj_side_bnb) <= 0.0 or float(adj_total_bnb) <= 0.0:
        return float("-inf")

    payout_multiple = (float(adj_total_bnb) * (1.0 - float(treasury_fee_fraction))) / float(adj_side_bnb)
    win_credit_bnb = float(stake_bnb) * float(payout_multiple) - float(GAS_COST_CLAIM_BNB)
    expected_credit_bnb = float(p_win) * float(win_credit_bnb)
    expected_net_bnb = float(expected_credit_bnb) - (float(stake_bnb) + float(GAS_COST_BET_BNB))
    return float(expected_net_bnb)


class MlCandidateAdapter:
    """Emit one ML candidate signal compatible with the shared router."""

    def __init__(
        self,
        *,
        config: MlCandidateConfig,
        cutoff_seconds: int,
        treasury_fee_fraction: float,
        klines_store_like: object,
        feature_cache_store: object | None = None,
    ) -> None:
        self._config = config
        self._history_rounds: list[Round] = []
        self._state: WalkForwardState | None = None
        self._wf_cfg = _MlWalkForwardRuntimeConfig(
            klines_store=klines_store_like,
            cutoff_seconds=int(cutoff_seconds),
            train_size=int(config.train_size),
            calibrate_size=int(config.calibrate_size),
            retrain_interval=int(config.retrain_interval),
            recalibrate_interval=int(config.recalibrate_interval),
            price_alpha=float(config.price_alpha),
            pool_alpha_total=float(config.pool_alpha_total),
            pool_alpha_ratio=float(config.pool_alpha_ratio),
            random_seed=int(config.random_seed),
            recency_weight_floor=float(config.recency_weight_floor),
            recency_weight_power=float(config.recency_weight_power),
            predictability_baseline_bet_bnb=float(config.predictability_baseline_bet_bnb),
            treasury_fee_fraction=float(treasury_fee_fraction),
            feature_cache_store=feature_cache_store,
        )
        self._validate_config()

    @property
    def enabled(self) -> bool:
        """Return whether this adapter is enabled."""

        return bool(self._config.enabled)

    @property
    def candidate_name(self) -> str:
        """Return ML candidate name emitted to the router."""

        return str(self._config.name)

    def refresh_klines(self, *, klines: list[Kline]) -> None:
        """No-op hook for pipeline parity with dislocation providers."""

        _ = klines

    def bootstrap_from_closed_rounds(self, *, rounds: list[Round]) -> None:
        """Seed closed-history context consumed by walk-forward training."""

        self.settle_closed_rounds(rounds=rounds)

    def settle_closed_rounds(self, *, rounds: list[Round]) -> None:
        """Append new closed rounds to ML history cache."""

        if not rounds:
            return
        for round_t in sorted(rounds, key=lambda x: int(x.epoch)):
            if self._history_rounds and int(round_t.epoch) <= int(self._history_rounds[-1].epoch):
                continue
            self._history_rounds.append(round_t)

    def export_bootstrap_state(self) -> dict[str, object]:
        """Export ML walk-forward state snapshot for backtest bootstrap cache."""

        return {
            "history_rounds_json": [r.to_json() for r in self._history_rounds],
            "walk_forward_state": self._state,
        }

    def import_bootstrap_state(self, *, state: dict[str, object]) -> None:
        """Restore ML walk-forward state snapshot for backtest bootstrap cache."""

        history_raw = list(state.get("history_rounds_json", []))
        history: list[Round] = [Round.from_json(obj) for obj in history_raw]
        history.sort(key=lambda r: int(r.epoch))
        deduped: list[Round] = []
        for round_t in history:
            if deduped and int(round_t.epoch) <= int(deduped[-1].epoch):
                continue
            deduped.append(round_t)
        self._history_rounds = list(deduped)
        self._state = state.get("walk_forward_state")

    def candidate_signal_for_open_round(self, *, round_t: Round) -> StrategyCandidateSignal:
        """Generate one ML candidate signal for the target round."""

        if not bool(self._config.enabled):
            return self._skip_signal(skip_reason="ml_candidate_disabled")
        if round_t.lock_at is None:
            return self._skip_signal(skip_reason="round_lock_at_missing")

        k = int(max_required_prior_context_rounds_size())
        if len(self._history_rounds) < int(k):
            return self._skip_signal(skip_reason="ml_history_insufficient")

        try:
            self._state = ensure_state(
                cfg=self._wf_cfg,
                closed_rounds=list(self._history_rounds),
                current_epoch=int(round_t.epoch),
                state=self._state,
            )
        except InvariantError:
            return self._skip_signal(skip_reason="ml_state_not_ready")

        if self._state is None or self._state.models is None or self._state.calibrator_final is None:
            return self._skip_signal(skip_reason="ml_state_not_ready")

        cutoff_ts = int(round_t.lock_at) - int(self._wf_cfg.cutoff_seconds)
        pools = compute_pool_amounts_wei_at_or_before(
            bets=round_t.bets,
            cutoff_ts=int(cutoff_ts),
        )
        pool_total_bnb = float(pools.total_wei) / float(BNB_WEI)
        pool_bull_bnb = float(pools.bull_wei) / float(BNB_WEI)
        pool_bear_bnb = float(pools.bear_wei) / float(BNB_WEI)
        if float(pool_total_bnb) <= 0.0:
            return self._skip_signal(skip_reason="cutoff_pool_empty")
        if float(pool_total_bnb) < float(self._config.cutoff_pool_total_min_bnb):
            return self._skip_signal(skip_reason="cutoff_pool_below_min_total")

        prior_context_rounds = list(self._history_rounds[-int(k):])
        x_row = self._feature_vector_for_round(
            round_t=round_t,
            prior_context_rounds=prior_context_rounds,
        )

        mu = float(self._state.models.price_model.predict([list(x_row)])[0])
        p_bull = float(predict_probabilities(state=self._state, mu=float(mu)))
        p_tradeable = float(
            predict_tradeable_probability(
                state=self._state,
                x_row=list(x_row),
            )
        )
        p_market_bull = float(pool_bull_bnb / pool_total_bnb)
        dislocation_bull = float(p_bull) - float(p_market_bull)

        if float(p_tradeable) < float(self._config.min_tradeable_prob):
            return self._skip_signal(
                skip_reason="predictability_below_min",
                p_bull=float(p_bull),
                dislocation_bull=float(dislocation_bull),
            )
        if abs(float(p_bull) - 0.5) < float(self._config.min_prob_edge):
            return self._skip_signal(
                skip_reason="p_bull_edge_below_min",
                p_bull=float(p_bull),
                dislocation_bull=float(dislocation_bull),
            )

        late_total_bnb, late_bull_frac = self._state.models.pool_model.predict([list(x_row)])[0]
        late_total_bnb = max(0.0, float(late_total_bnb))
        late_bull_frac = min(1.0, max(0.0, float(late_bull_frac)))

        final_total_bnb = float(pool_total_bnb) + float(late_total_bnb)
        final_bull_bnb = float(pool_bull_bnb) + float(late_total_bnb) * float(late_bull_frac)
        final_bear_bnb = float(pool_bear_bnb) + float(late_total_bnb) * (1.0 - float(late_bull_frac))
        if float(final_total_bnb) <= 0.0 or float(final_bull_bnb) <= 0.0 or float(final_bear_bnb) <= 0.0:
            return self._skip_signal(
                skip_reason="predicted_pool_invalid",
                p_bull=float(p_bull),
                dislocation_bull=float(dislocation_bull),
            )

        ev_bull = _expected_net_from_predicted_final(
            p_bull=float(p_bull),
            side="BULL",
            stake_bnb=float(self._config.fixed_bet_bnb),
            final_bull_bnb=float(final_bull_bnb),
            final_bear_bnb=float(final_bear_bnb),
            treasury_fee_fraction=float(self._wf_cfg.treasury_fee_fraction),
        )
        ev_bear = _expected_net_from_predicted_final(
            p_bull=float(p_bull),
            side="BEAR",
            stake_bnb=float(self._config.fixed_bet_bnb),
            final_bull_bnb=float(final_bull_bnb),
            final_bear_bnb=float(final_bear_bnb),
            treasury_fee_fraction=float(self._wf_cfg.treasury_fee_fraction),
        )
        if float(ev_bull) >= float(ev_bear):
            side = "Bull"
            best_ev = float(ev_bull)
        else:
            side = "Bear"
            best_ev = float(ev_bear)

        if float(best_ev) < float(self._config.expected_net_min_bnb):
            return self._skip_signal(
                skip_reason="expected_net_below_min",
                p_bull=float(p_bull),
                dislocation_bull=float(dislocation_bull),
            )

        return StrategyCandidateSignal(
            candidate_name=str(self._config.name),
            action="BET",
            bet_side=str(side),
            bet_size_bnb=float(self._config.fixed_bet_bnb),
            expected_profit_bnb=float(best_ev),
            selector_score_bnb=float(best_ev),
            skip_reason=None,
            p_bull=float(p_bull),
            dislocation_bull=float(dislocation_bull),
        )

    def _skip_signal(
        self,
        *,
        skip_reason: str,
        p_bull: float | None = None,
        dislocation_bull: float | None = None,
    ) -> StrategyCandidateSignal:
        return StrategyCandidateSignal(
            candidate_name=str(self._config.name),
            action="SKIP",
            bet_side=None,
            bet_size_bnb=0.0,
            expected_profit_bnb=None,
            selector_score_bnb=None,
            skip_reason=str(skip_reason),
            p_bull=float(p_bull) if p_bull is not None else None,
            dislocation_bull=(
                float(dislocation_bull) if dislocation_bull is not None else None
            ),
        )

    def _validate_config(self) -> None:
        if str(self._config.name).strip() == "":
            raise InvariantError("ml_candidate_name_empty")
        if float(self._config.fixed_bet_bnb) <= 0.0:
            raise InvariantError("ml_candidate_fixed_bet_nonpositive")

    def _feature_vector_for_round(
        self,
        *,
        round_t: Round,
        prior_context_rounds: list[Round],
    ) -> list[float]:
        if round_t.lock_at is None:
            raise InvariantError("ml_round_lock_at_missing")
        if not prior_context_rounds:
            raise InvariantError("ml_prior_context_rounds_empty")

        prior_last_epoch = int(prior_context_rounds[-1].epoch)
        anchor_close_time_ms = self._anchor_close_time_ms(round_t=round_t)
        lock_at = int(round_t.lock_at)
        feature_cache = self._wf_cfg.feature_cache_store

        if feature_cache is not None:
            if not hasattr(feature_cache, "get_vector") or not hasattr(feature_cache, "put_vector"):
                raise InvariantError("ml_feature_cache_store_invalid")
            cached = feature_cache.get_vector(
                epoch=int(round_t.epoch),
                cutoff_seconds=int(self._wf_cfg.cutoff_seconds),
                schema_name=str(FEATURE_SCHEMA.name),
                start_at=int(round_t.start_at),
                lock_at=int(lock_at),
                prior_last_epoch=int(prior_last_epoch),
                anchor_close_time_ms=int(anchor_close_time_ms),
            )
            if cached is not None:
                return list(cached)

        context_klines = self._context_klines(round_t=round_t)
        if not context_klines:
            raise InvariantError("ml_context_klines_empty")

        features = build_features(
            target_round=round_t,
            prior_context_rounds=prior_context_rounds,
            context_klines=context_klines,
            cutoff_seconds=int(self._wf_cfg.cutoff_seconds),
        )
        x_row = vectorize(features=features, schema=FEATURE_SCHEMA)

        if feature_cache is not None:
            feature_cache.put_vector(
                epoch=int(round_t.epoch),
                cutoff_seconds=int(self._wf_cfg.cutoff_seconds),
                schema_name=str(FEATURE_SCHEMA.name),
                start_at=int(round_t.start_at),
                lock_at=int(lock_at),
                prior_last_epoch=int(prior_last_epoch),
                anchor_close_time_ms=int(anchor_close_time_ms),
                vector=list(x_row),
            )
        return list(x_row)

    def _anchor_close_time_ms(self, *, round_t: Round) -> int:
        if not hasattr(self._wf_cfg.klines_store, "latest_close_time_ms"):
            raise InvariantError("ml_klines_store_missing_latest_close_time_ms")
        if round_t.lock_at is None:
            raise InvariantError("ml_round_lock_at_missing")
        cutoff_ts = int(round_t.lock_at) - int(self._wf_cfg.cutoff_seconds)
        anchor_ms = int(cutoff_ts) * 1000
        latest_close_ms = self._wf_cfg.klines_store.latest_close_time_ms()
        if latest_close_ms is None:
            raise InvariantError("ml_klines_store_empty")
        if int(latest_close_ms) < int(anchor_ms):
            anchor_ms = int(latest_close_ms)
        return int(anchor_ms)

    def _context_klines(self, *, round_t: Round) -> list[Kline]:
        """Load cutoff-anchored context klines from the shared kline source."""

        if not hasattr(self._wf_cfg.klines_store, "latest_close_time_ms"):
            raise InvariantError("ml_klines_store_missing_latest_close_time_ms")
        if not hasattr(self._wf_cfg.klines_store, "get_context_klines"):
            raise InvariantError("ml_klines_store_missing_get_context_klines")
        if round_t.lock_at is None:
            raise InvariantError("ml_round_lock_at_missing")

        from pancakebot.domain.features.schema import max_required_context_klines_size

        kk = int(max_required_context_klines_size())
        anchor_ms = self._anchor_close_time_ms(round_t=round_t)
        out = self._wf_cfg.klines_store.get_context_klines(
            anchor_close_time_ms=int(anchor_ms),
            size=int(kk),
        )
        return list(out)
