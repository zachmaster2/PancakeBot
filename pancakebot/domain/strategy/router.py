"""Shared strategy-router contract for candidate selection.

The router consumes per-candidate signals and emits one normalized decision
(`BET` or `SKIP`). Router state is updated via `observe_settlement(...)` so
online cell-mean routing can adapt from realized candidate outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass

from pancakebot.core.errors import InvariantError
from pancakebot.domain.strategy.candidate_signal import StrategyCandidateSignal

_ROUTER_MODES = (
    "selector_max_score",
    "skip_only",
    "oracle_skip",
    "online_cellmean",
    "online_cellmean_side_gap",
    "online_cellmean_backoff",
    "online_cellmean_selector_fallback",
)
_VALID_BET_SIDES = ("Bull", "Bear")


def _quantile_edges(values: list[float], n_bins: int) -> list[float]:
    if int(n_bins) <= 1:
        raise InvariantError("router_online_num_quantile_bins_invalid")
    if not values:
        return [0.0 for _ in range(int(n_bins) + 1)]
    vv = sorted(float(x) for x in values)
    out: list[float] = []
    for i in range(int(n_bins) + 1):
        q = float(i) / float(n_bins)
        idx = int(round((len(vv) - 1) * q))
        idx = max(0, min(len(vv) - 1, idx))
        out.append(float(vv[idx]))
    return out


def _bin_index(x: float, edges: list[float]) -> int:
    n_bins = int(len(edges) - 1)
    if int(n_bins) <= 1:
        return 0
    for i in range(int(n_bins)):
        lo = float(edges[i])
        hi = float(edges[i + 1])
        if float(x) >= float(lo) and (float(x) < float(hi) or int(i) == int(n_bins - 1)):
            return int(i)
    return int(n_bins - 1)


def _side_idx(side: str) -> int:
    if str(side) == "Bull":
        return 0
    if str(side) == "Bear":
        return 1
    raise InvariantError("router_candidate_bet_side_invalid")


@dataclass(frozen=True, slots=True)
class _OnlineCellRow:
    """Internal online cell-mean row for one candidate-round observation."""

    expected_profit_bnb: float
    abs_dislocation_bull: float
    side_idx: int
    realized_profit_bnb: float


@dataclass(frozen=True, slots=True)
class StrategyRouterConfig:
    """Configuration for one router instance."""

    mode: str = "selector_max_score"
    score_threshold_bnb: float = -1e9
    online_warmup_rounds: int = 50_000
    online_num_quantile_bins: int = 12
    online_min_cell_obs: int = 5
    online_score_threshold_bnb: float = 0.0
    online_use_direction_split: bool = True

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate router configuration values."""

        if str(self.mode) not in _ROUTER_MODES:
            raise InvariantError("router_mode_invalid")
        if not isinstance(self.score_threshold_bnb, (int, float)):
            raise InvariantError("router_score_threshold_bnb_not_number")
        if int(self.online_warmup_rounds) <= 0:
            raise InvariantError("router_online_warmup_rounds_must_be_positive")
        if int(self.online_num_quantile_bins) <= 1:
            raise InvariantError("router_online_num_quantile_bins_invalid")
        if int(self.online_min_cell_obs) <= 0:
            raise InvariantError("router_online_min_cell_obs_must_be_positive")
        if not isinstance(self.online_score_threshold_bnb, (int, float)):
            raise InvariantError("router_online_score_threshold_bnb_not_number")
        if not isinstance(self.online_use_direction_split, bool):
            raise InvariantError("router_online_use_direction_split_not_bool")


@dataclass(frozen=True, slots=True)
class StrategyRouterDecision:
    """One routed strategy decision for an open round."""

    action: str
    selected_strategy: str | None
    bet_side: str | None
    bet_size_bnb: float
    expected_profit_bnb: float
    selector_score_bnb: float | None
    skip_reason: str | None
    p_bull: float | None


class StrategyRouter:
    """Select one strategy candidate signal for the current round."""

    def __init__(self, *, config: StrategyRouterConfig) -> None:
        self._config = config
        self._online_ready = False
        self._online_settled_round_count = 0
        self._online_warmup_rows_by_candidate: dict[str, list[_OnlineCellRow]] = {}
        self._online_edges_by_candidate: dict[str, tuple[list[float], list[float]]] | None = None
        self._online_sum_profit_by_candidate: dict[str, dict[tuple[int, int, int], float]] = {}
        self._online_count_by_candidate: dict[str, dict[tuple[int, int, int], int]] = {}

    @property
    def mode(self) -> str:
        """Return router mode name."""

        return str(self._config.mode)

    def export_bootstrap_state(self) -> dict[str, object]:
        """Export online router state snapshot for backtest bootstrap cache."""

        return {
            "online_ready": bool(self._online_ready),
            "online_settled_round_count": int(self._online_settled_round_count),
            "online_warmup_rows_by_candidate": self._online_warmup_rows_by_candidate,
            "online_edges_by_candidate": self._online_edges_by_candidate,
            "online_sum_profit_by_candidate": self._online_sum_profit_by_candidate,
            "online_count_by_candidate": self._online_count_by_candidate,
        }

    def import_bootstrap_state(self, *, state: dict[str, object]) -> None:
        """Restore online router state snapshot for backtest bootstrap cache."""

        self._online_ready = bool(state.get("online_ready", False))
        self._online_settled_round_count = int(state.get("online_settled_round_count", 0))
        self._online_warmup_rows_by_candidate = dict(state.get("online_warmup_rows_by_candidate", {}))
        self._online_edges_by_candidate = state.get("online_edges_by_candidate")
        self._online_sum_profit_by_candidate = dict(state.get("online_sum_profit_by_candidate", {}))
        self._online_count_by_candidate = dict(state.get("online_count_by_candidate", {}))

    def route_round(
        self,
        *,
        candidate_signals: dict[str, StrategyCandidateSignal],
        bankroll_bnb: float,
        bet_gas_cost_bnb: float,
        selector_ready: bool,
        realized_profit_by_candidate: dict[str, float] | None = None,
    ) -> StrategyRouterDecision:
        """Route one round from candidate signals to one action.

        Args:
            candidate_signals: Candidate signal table keyed by candidate name.
            bankroll_bnb: Current real bankroll used for affordability gates.
            bet_gas_cost_bnb: Gas cost added to bet-size affordability checks.
            selector_ready: Selector readiness signal from candidate pipeline.
            realized_profit_by_candidate: Optional hindsight per-candidate PnL.
                Required by `oracle_skip` mode.
        """

        self._validate_candidate_signals(candidate_signals)
        self._ensure_online_candidate_tables(candidate_signals=candidate_signals)
        if float(bankroll_bnb) < 0.0:
            raise InvariantError("router_bankroll_negative")
        if float(bet_gas_cost_bnb) < 0.0:
            raise InvariantError("router_bet_gas_cost_negative")

        mode = str(self._config.mode)
        if mode == "skip_only":
            return self._skip_decision(skip_reason="router_skip_only")
        if mode == "oracle_skip":
            return self._route_oracle_skip(
                candidate_signals=candidate_signals,
                bankroll_bnb=float(bankroll_bnb),
                bet_gas_cost_bnb=float(bet_gas_cost_bnb),
                realized_profit_by_candidate=realized_profit_by_candidate,
            )
        if mode == "selector_max_score":
            return self._route_selector_max_score(
                candidate_signals=candidate_signals,
                bankroll_bnb=float(bankroll_bnb),
                bet_gas_cost_bnb=float(bet_gas_cost_bnb),
                selector_ready=bool(selector_ready),
            )
        if mode == "online_cellmean":
            return self._route_online_cellmean(
                candidate_signals=candidate_signals,
                bankroll_bnb=float(bankroll_bnb),
                bet_gas_cost_bnb=float(bet_gas_cost_bnb),
                selector_ready=bool(selector_ready),
                use_candidate_backoff=False,
                allow_selector_fallback=False,
                require_side_gap=False,
            )
        if mode == "online_cellmean_side_gap":
            return self._route_online_cellmean(
                candidate_signals=candidate_signals,
                bankroll_bnb=float(bankroll_bnb),
                bet_gas_cost_bnb=float(bet_gas_cost_bnb),
                selector_ready=bool(selector_ready),
                use_candidate_backoff=False,
                allow_selector_fallback=False,
                require_side_gap=True,
            )
        if mode == "online_cellmean_backoff":
            return self._route_online_cellmean(
                candidate_signals=candidate_signals,
                bankroll_bnb=float(bankroll_bnb),
                bet_gas_cost_bnb=float(bet_gas_cost_bnb),
                selector_ready=bool(selector_ready),
                use_candidate_backoff=True,
                allow_selector_fallback=False,
                require_side_gap=False,
            )
        if mode == "online_cellmean_selector_fallback":
            return self._route_online_cellmean(
                candidate_signals=candidate_signals,
                bankroll_bnb=float(bankroll_bnb),
                bet_gas_cost_bnb=float(bet_gas_cost_bnb),
                selector_ready=bool(selector_ready),
                use_candidate_backoff=False,
                allow_selector_fallback=True,
                require_side_gap=False,
            )
        raise InvariantError("router_mode_unreachable")

    def observe_settlement(
        self,
        *,
        candidate_signals: dict[str, StrategyCandidateSignal],
        realized_profit_by_candidate: dict[str, float],
    ) -> None:
        """Consume realized candidate outcomes for online router adaptation."""

        if str(self._config.mode) not in (
            "online_cellmean",
            "online_cellmean_side_gap",
            "online_cellmean_backoff",
            "online_cellmean_selector_fallback",
        ):
            return

        self._validate_candidate_signals(candidate_signals)
        self._ensure_online_candidate_tables(candidate_signals=candidate_signals)

        rows_by_candidate: dict[str, _OnlineCellRow] = {}
        for candidate_name, signal in candidate_signals.items():
            if str(signal.action) != "BET":
                continue
            side = str(signal.bet_side or "")
            if side not in _VALID_BET_SIDES:
                raise InvariantError("router_candidate_bet_side_invalid")
            if str(candidate_name) not in realized_profit_by_candidate:
                raise InvariantError("router_observe_missing_realized_profit")
            if signal.expected_profit_bnb is None or signal.dislocation_bull is None:
                continue
            rows_by_candidate[str(candidate_name)] = _OnlineCellRow(
                expected_profit_bnb=float(signal.expected_profit_bnb),
                abs_dislocation_bull=abs(float(signal.dislocation_bull)),
                side_idx=int(_side_idx(str(side))),
                realized_profit_bnb=float(realized_profit_by_candidate[str(candidate_name)]),
            )

        if not bool(self._online_ready):
            for candidate_name, row in rows_by_candidate.items():
                self._online_warmup_rows_by_candidate[str(candidate_name)].append(row)
            self._online_settled_round_count += 1
            if int(self._online_settled_round_count) >= int(self._config.online_warmup_rounds):
                self._freeze_online_cells()
            return

        if self._online_edges_by_candidate is None:
            raise InvariantError("router_online_edges_missing")
        for candidate_name, row in rows_by_candidate.items():
            ev_edges, dis_edges = self._online_edges_by_candidate[str(candidate_name)]
            key = self._online_cell_key(row=row, ev_edges=ev_edges, dis_edges=dis_edges)
            sums = self._online_sum_profit_by_candidate[str(candidate_name)]
            counts = self._online_count_by_candidate[str(candidate_name)]
            sums[key] = float(sums.get(key, 0.0) + float(row.realized_profit_bnb))
            counts[key] = int(counts.get(key, 0) + 1)
        self._online_settled_round_count += 1

    def _route_selector_max_score(
        self,
        *,
        candidate_signals: dict[str, StrategyCandidateSignal],
        bankroll_bnb: float,
        bet_gas_cost_bnb: float,
        selector_ready: bool,
    ) -> StrategyRouterDecision:
        if not bool(selector_ready):
            return self._skip_decision(skip_reason="selector_warmup")

        best_name: str | None = None
        best_score = float("-inf")
        for candidate_name, signal in candidate_signals.items():
            if str(signal.action) != "BET":
                continue
            if signal.selector_score_bnb is None:
                continue
            side = str(signal.bet_side or "")
            if side not in _VALID_BET_SIDES:
                raise InvariantError("router_candidate_bet_side_invalid")
            if float(signal.bet_size_bnb) <= 0.0:
                raise InvariantError("router_candidate_bet_size_nonpositive")
            score = float(signal.selector_score_bnb)
            if float(score) > float(best_score):
                best_score = float(score)
                best_name = str(candidate_name)

        if best_name is None:
            return self._skip_decision(skip_reason="selector_no_candidate")

        if float(best_score) < float(self._config.score_threshold_bnb):
            return self._skip_decision(skip_reason="router_score_below_threshold")

        signal = candidate_signals[str(best_name)]
        return self._to_affordability_checked_decision(
            candidate_name=str(best_name),
            signal=signal,
            bankroll_bnb=float(bankroll_bnb),
            bet_gas_cost_bnb=float(bet_gas_cost_bnb),
            selector_score_bnb=float(best_score),
        )

    def _route_online_cellmean(
        self,
        *,
        candidate_signals: dict[str, StrategyCandidateSignal],
        bankroll_bnb: float,
        bet_gas_cost_bnb: float,
        selector_ready: bool,
        use_candidate_backoff: bool,
        allow_selector_fallback: bool,
        require_side_gap: bool,
    ) -> StrategyRouterDecision:
        if not bool(self._online_ready):
            return self._skip_decision(skip_reason="router_online_warmup")
        if self._online_edges_by_candidate is None:
            raise InvariantError("router_online_edges_missing")
        if bool(require_side_gap) and not bool(self._config.online_use_direction_split):
            return self._skip_decision(skip_reason="router_online_side_gap_requires_direction_split")

        best_name: str | None = None
        best_estimate = float("-inf")
        for candidate_name, signal in candidate_signals.items():
            if str(signal.action) != "BET":
                continue
            side = str(signal.bet_side or "")
            if side not in _VALID_BET_SIDES:
                raise InvariantError("router_candidate_bet_side_invalid")
            if float(signal.bet_size_bnb) <= 0.0:
                raise InvariantError("router_candidate_bet_size_nonpositive")
            if signal.expected_profit_bnb is None or signal.dislocation_bull is None:
                continue

            ev_edges, dis_edges = self._online_edges_by_candidate[str(candidate_name)]
            key = self._online_cell_key_for_signal(
                signal=signal,
                ev_edges=ev_edges,
                dis_edges=dis_edges,
            )
            counts = self._online_count_by_candidate[str(candidate_name)]
            count = int(counts.get(key, 0))
            sums = self._online_sum_profit_by_candidate[str(candidate_name)]
            if int(count) < int(self._config.online_min_cell_obs):
                if not bool(use_candidate_backoff):
                    continue
                total_count = int(sum(int(v) for v in counts.values()))
                if int(total_count) <= 0:
                    continue
                total_sum = float(sum(float(v) for v in sums.values()))
                estimated = float(total_sum) / float(total_count)
            else:
                estimated = float(sums.get(key, 0.0)) / float(count)
            if float(estimated) < float(self._config.online_score_threshold_bnb):
                continue
            if bool(require_side_gap):
                ev_bin, dis_bin, side_bin = key
                opposite_key = (int(ev_bin), int(dis_bin), int(1 - int(side_bin)))
                opposite_count = int(counts.get(opposite_key, 0))
                if int(opposite_count) < int(self._config.online_min_cell_obs):
                    continue
                opposite_estimated = float(sums.get(opposite_key, 0.0)) / float(opposite_count)
                if float(estimated) <= float(opposite_estimated):
                    continue
                estimated = float(estimated) - float(opposite_estimated)
            if float(estimated) > float(best_estimate):
                best_estimate = float(estimated)
                best_name = str(candidate_name)

        if best_name is None:
            if not bool(allow_selector_fallback):
                return self._skip_decision(skip_reason="router_online_no_candidate")

            fallback = self._route_selector_max_score(
                candidate_signals=candidate_signals,
                bankroll_bnb=float(bankroll_bnb),
                bet_gas_cost_bnb=float(bet_gas_cost_bnb),
                selector_ready=bool(selector_ready),
            )
            if str(fallback.action) != "BET":
                skip_reason = str(fallback.skip_reason or "router_fallback_selector_no_candidate")
                if skip_reason == "selector_warmup":
                    skip_reason = "router_fallback_selector_warmup"
                elif skip_reason == "selector_no_candidate":
                    skip_reason = "router_fallback_selector_no_candidate"
                elif skip_reason == "router_score_below_threshold":
                    skip_reason = "router_fallback_selector_score_below_threshold"
                return self._skip_decision(
                    skip_reason=str(skip_reason),
                    selected_strategy=fallback.selected_strategy,
                    selector_score_bnb=fallback.selector_score_bnb,
                    p_bull=fallback.p_bull,
                )

            selector_score = (
                float(fallback.selector_score_bnb)
                if fallback.selector_score_bnb is not None
                else float("-inf")
            )
            if float(selector_score) < 0.0 or float(fallback.expected_profit_bnb) <= 0.0:
                return self._skip_decision(
                    skip_reason="router_fallback_selector_rejected",
                    selected_strategy=fallback.selected_strategy,
                    selector_score_bnb=fallback.selector_score_bnb,
                    p_bull=fallback.p_bull,
                )
            return fallback

        signal = candidate_signals[str(best_name)]
        return self._to_affordability_checked_decision(
            candidate_name=str(best_name),
            signal=signal,
            bankroll_bnb=float(bankroll_bnb),
            bet_gas_cost_bnb=float(bet_gas_cost_bnb),
            selector_score_bnb=float(best_estimate),
        )

    def _route_oracle_skip(
        self,
        *,
        candidate_signals: dict[str, StrategyCandidateSignal],
        bankroll_bnb: float,
        bet_gas_cost_bnb: float,
        realized_profit_by_candidate: dict[str, float] | None,
    ) -> StrategyRouterDecision:
        if realized_profit_by_candidate is None:
            raise InvariantError("router_oracle_profit_table_missing")

        best_name: str | None = None
        best_profit = 0.0
        for candidate_name, signal in candidate_signals.items():
            if str(signal.action) != "BET":
                continue
            side = str(signal.bet_side or "")
            if side not in _VALID_BET_SIDES:
                raise InvariantError("router_candidate_bet_side_invalid")
            if float(signal.bet_size_bnb) <= 0.0:
                raise InvariantError("router_candidate_bet_size_nonpositive")
            if str(candidate_name) not in realized_profit_by_candidate:
                raise InvariantError("router_oracle_profit_candidate_missing")
            profit = float(realized_profit_by_candidate[str(candidate_name)])
            if float(profit) > float(best_profit):
                best_profit = float(profit)
                best_name = str(candidate_name)

        if best_name is None:
            return self._skip_decision(skip_reason="oracle_no_positive_profit")

        signal = candidate_signals[str(best_name)]
        return self._to_affordability_checked_decision(
            candidate_name=str(best_name),
            signal=signal,
            bankroll_bnb=float(bankroll_bnb),
            bet_gas_cost_bnb=float(bet_gas_cost_bnb),
            selector_score_bnb=None,
        )

    def _to_affordability_checked_decision(
        self,
        *,
        candidate_name: str,
        signal: StrategyCandidateSignal,
        bankroll_bnb: float,
        bet_gas_cost_bnb: float,
        selector_score_bnb: float | None,
    ) -> StrategyRouterDecision:
        side = str(signal.bet_side or "")
        if side not in _VALID_BET_SIDES:
            raise InvariantError("router_candidate_bet_side_invalid")
        bet_size = float(signal.bet_size_bnb)
        total_cost = float(bet_size) + float(bet_gas_cost_bnb)
        if float(total_cost) > float(bankroll_bnb):
            return self._skip_decision(
                skip_reason="insufficient_bankroll_real",
                selected_strategy=str(candidate_name),
                selector_score_bnb=selector_score_bnb,
                p_bull=signal.p_bull,
            )
        return StrategyRouterDecision(
            action="BET",
            selected_strategy=str(candidate_name),
            bet_side=str(side),
            bet_size_bnb=float(bet_size),
            expected_profit_bnb=float(signal.expected_profit_bnb or 0.0),
            selector_score_bnb=(
                float(selector_score_bnb) if selector_score_bnb is not None else None
            ),
            skip_reason=None,
            p_bull=float(signal.p_bull) if signal.p_bull is not None else None,
        )

    def _freeze_online_cells(self) -> None:
        edges_by_candidate: dict[str, tuple[list[float], list[float]]] = {}
        for candidate_name, rows in self._online_warmup_rows_by_candidate.items():
            ev_values = [float(r.expected_profit_bnb) for r in rows]
            dis_values = [float(r.abs_dislocation_bull) for r in rows]
            ev_edges = _quantile_edges(ev_values, int(self._config.online_num_quantile_bins))
            dis_edges = _quantile_edges(dis_values, int(self._config.online_num_quantile_bins))
            edges_by_candidate[str(candidate_name)] = (ev_edges, dis_edges)

        self._online_edges_by_candidate = edges_by_candidate
        for candidate_name, rows in self._online_warmup_rows_by_candidate.items():
            if str(candidate_name) not in self._online_sum_profit_by_candidate:
                self._online_sum_profit_by_candidate[str(candidate_name)] = {}
            if str(candidate_name) not in self._online_count_by_candidate:
                self._online_count_by_candidate[str(candidate_name)] = {}
            sums = self._online_sum_profit_by_candidate[str(candidate_name)]
            counts = self._online_count_by_candidate[str(candidate_name)]
            ev_edges, dis_edges = edges_by_candidate[str(candidate_name)]
            for row in rows:
                key = self._online_cell_key(row=row, ev_edges=ev_edges, dis_edges=dis_edges)
                sums[key] = float(sums.get(key, 0.0) + float(row.realized_profit_bnb))
                counts[key] = int(counts.get(key, 0) + 1)

        self._online_warmup_rows_by_candidate = {}
        self._online_ready = True

    def _ensure_online_candidate_tables(
        self,
        *,
        candidate_signals: dict[str, StrategyCandidateSignal],
    ) -> None:
        for candidate_name in candidate_signals.keys():
            key = str(candidate_name)
            if key not in self._online_warmup_rows_by_candidate:
                self._online_warmup_rows_by_candidate[key] = []
            if key not in self._online_sum_profit_by_candidate:
                self._online_sum_profit_by_candidate[key] = {}
            if key not in self._online_count_by_candidate:
                self._online_count_by_candidate[key] = {}

    def _online_cell_key_for_signal(
        self,
        *,
        signal: StrategyCandidateSignal,
        ev_edges: list[float],
        dis_edges: list[float],
    ) -> tuple[int, int, int]:
        if signal.expected_profit_bnb is None:
            raise InvariantError("router_online_expected_profit_missing")
        if signal.dislocation_bull is None:
            raise InvariantError("router_online_dislocation_missing")
        if signal.bet_side is None:
            raise InvariantError("router_online_bet_side_missing")
        row = _OnlineCellRow(
            expected_profit_bnb=float(signal.expected_profit_bnb),
            abs_dislocation_bull=abs(float(signal.dislocation_bull)),
            side_idx=int(_side_idx(str(signal.bet_side))),
            realized_profit_bnb=0.0,
        )
        return self._online_cell_key(row=row, ev_edges=ev_edges, dis_edges=dis_edges)

    def _online_cell_key(
        self,
        *,
        row: _OnlineCellRow,
        ev_edges: list[float],
        dis_edges: list[float],
    ) -> tuple[int, int, int]:
        ev_bin = _bin_index(float(row.expected_profit_bnb), ev_edges)
        dis_bin = _bin_index(float(row.abs_dislocation_bull), dis_edges)
        side_bin = int(row.side_idx) if bool(self._config.online_use_direction_split) else 0
        return int(ev_bin), int(dis_bin), int(side_bin)

    @staticmethod
    def _validate_candidate_signals(candidate_signals: dict[str, StrategyCandidateSignal]) -> None:
        if not candidate_signals:
            raise InvariantError("router_candidate_signals_empty")
        for candidate_name, signal in candidate_signals.items():
            if str(candidate_name) != str(signal.candidate_name):
                raise InvariantError("router_candidate_signal_key_mismatch")

    @staticmethod
    def _skip_decision(
        *,
        skip_reason: str,
        selected_strategy: str | None = None,
        selector_score_bnb: float | None = None,
        p_bull: float | None = None,
    ) -> StrategyRouterDecision:
        return StrategyRouterDecision(
            action="SKIP",
            selected_strategy=str(selected_strategy) if selected_strategy is not None else None,
            bet_side=None,
            bet_size_bnb=0.0,
            expected_profit_bnb=0.0,
            selector_score_bnb=(
                float(selector_score_bnb) if selector_score_bnb is not None else None
            ),
            skip_reason=str(skip_reason),
            p_bull=float(p_bull) if p_bull is not None else None,
        )
