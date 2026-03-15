"""Focused tests for ML/dislocation coupling in the shared strategy pipeline."""

from __future__ import annotations

import unittest

from pancakebot.domain.strategy.candidate_signal import StrategyCandidateSignal
from pancakebot.domain.strategy.pipeline import StrategyPipeline
from pancakebot.domain.types import Round


def _signal(
    *,
    candidate_name: str,
    action: str,
    bet_side: str | None,
    bet_size_bnb: float,
    expected_profit_bnb: float | None,
    selector_score_bnb: float | None,
    skip_reason: str | None,
    p_bull: float | None = 0.5,
    dislocation_bull: float | None = 0.0,
) -> StrategyCandidateSignal:
    return StrategyCandidateSignal(
        candidate_name=str(candidate_name),
        action=str(action),
        bet_side=bet_side,
        bet_size_bnb=float(bet_size_bnb),
        expected_profit_bnb=(
            float(expected_profit_bnb) if expected_profit_bnb is not None else None
        ),
        selector_score_bnb=(
            float(selector_score_bnb) if selector_score_bnb is not None else None
        ),
        skip_reason=skip_reason,
        p_bull=float(p_bull) if p_bull is not None else None,
        dislocation_bull=(
            float(dislocation_bull) if dislocation_bull is not None else None
        ),
    )


def _round(*, epoch: int = 1) -> Round:
    return Round(
        epoch=int(epoch),
        start_at=1000,
        lock_at=1300,
        close_at=None,
        lock_price=None,
        close_price=None,
        position=None,
        failed=None,
        bets=(),
    )


class _FakeDislocationEngine:
    def __init__(self, *, signals: dict[str, StrategyCandidateSignal]) -> None:
        self._signals = dict(signals)

    def candidate_signals_for_open_round(self, *, round_t: Round) -> dict[str, StrategyCandidateSignal]:
        return dict(self._signals)

    def refresh_klines(self, klines: list[object]) -> None:
        return None

    def export_kline_index_state(self) -> dict[str, object]:
        return {}

    def import_kline_index_state(self, *, state: dict[str, object]) -> None:
        return None

    def export_bootstrap_state(self) -> dict[str, object]:
        return {}

    def import_bootstrap_state(self, *, state: dict[str, object]) -> None:
        return None

    def settle_closed_rounds(self, rounds: list[Round]) -> None:
        return None

    def selector_ready(self) -> bool:
        return True


class _FakeRouter:
    mode = "selector_max_score"

    def export_bootstrap_state(self) -> dict[str, object]:
        return {}

    def import_bootstrap_state(self, *, state: dict[str, object]) -> None:
        return None

    def observe_settlement(
        self,
        *,
        candidate_signals: dict[str, StrategyCandidateSignal],
        realized_profit_by_candidate: dict[str, float],
    ) -> None:
        return None


class _FakeMlAdapter:
    def __init__(
        self,
        *,
        signal: StrategyCandidateSignal,
        emit_candidate: bool,
        veto_opposite_side_candidates: bool,
        veto_untradeable_candidates: bool,
        veto_candidate_expected_net_below_min: bool = False,
        candidate_veto_names: tuple[str, ...] = (),
        rescore_baseline_candidates_with_expected_net: bool = False,
        candidate_expected_net_by_name: dict[str, float] | None = None,
    ) -> None:
        self._signal = signal
        self.enabled = True
        self.emit_candidate = bool(emit_candidate)
        self.veto_opposite_side_candidates = bool(veto_opposite_side_candidates)
        self.veto_untradeable_candidates = bool(veto_untradeable_candidates)
        self.veto_candidate_expected_net_below_min = bool(veto_candidate_expected_net_below_min)
        self._candidate_veto_names = {str(x) for x in candidate_veto_names}
        self.rescore_baseline_candidates_with_expected_net = bool(
            rescore_baseline_candidates_with_expected_net
        )
        self._candidate_expected_net_by_name = {
            str(k): float(v) for k, v in (candidate_expected_net_by_name or {}).items()
        }
        self.observed_settlements: list[tuple[int, dict[str, float]]] = []

    def candidate_signal_for_open_round(self, *, round_t: Round) -> StrategyCandidateSignal:
        return self._signal

    def refresh_klines(self, *, klines: list[object]) -> None:
        return None

    def export_bootstrap_state(self) -> dict[str, object]:
        return {}

    def import_bootstrap_state(self, *, state: dict[str, object]) -> None:
        return None

    def settle_closed_rounds(self, *, rounds: list[Round]) -> None:
        return None

    def observe_baseline_candidate_settlement(
        self,
        *,
        round_t: Round,
        candidate_signals: dict[str, StrategyCandidateSignal],
        realized_profit_by_candidate: dict[str, float],
    ) -> None:
        _ = candidate_signals
        self.observed_settlements.append((int(round_t.epoch), dict(realized_profit_by_candidate)))

    def candidate_veto_skip_reason_for_open_round(
        self,
        *,
        round_t: Round,
        candidate_signal: StrategyCandidateSignal,
    ) -> str | None:
        _ = round_t
        if str(candidate_signal.candidate_name) in self._candidate_veto_names:
            return "ml_veto_candidate_expected_net_below_min"
        return None

    def candidate_expected_net_for_open_round(
        self,
        *,
        round_t: Round,
        candidate_signal: StrategyCandidateSignal,
    ) -> float | None:
        _ = round_t
        return self._candidate_expected_net_by_name.get(str(candidate_signal.candidate_name))


class StrategyPipelineMlCouplingTests(unittest.TestCase):
    def test_filter_only_mode_omits_ml_candidate(self) -> None:
        pipeline = StrategyPipeline(
            dislocation_engine=_FakeDislocationEngine(
                signals={
                    "altA": _signal(
                        candidate_name="altA",
                        action="BET",
                        bet_side="Bull",
                        bet_size_bnb=0.2,
                        expected_profit_bnb=0.03,
                        selector_score_bnb=0.08,
                        skip_reason=None,
                    )
                }
            ),
            router=_FakeRouter(),
            treasury_fee_fraction=0.03,
            ml_candidate_adapter=_FakeMlAdapter(
                signal=_signal(
                    candidate_name="mlwf_bestset_adapt_v1",
                    action="BET",
                    bet_side="Bull",
                    bet_size_bnb=0.2,
                    expected_profit_bnb=0.01,
                    selector_score_bnb=0.004,
                    skip_reason=None,
                ),
                emit_candidate=False,
                veto_opposite_side_candidates=False,
                veto_untradeable_candidates=False,
            ),
        )

        signals = pipeline.candidate_signals_for_open_round(round_t=_round())

        self.assertEqual({"altA"}, set(signals))
        self.assertEqual("BET", signals["altA"].action)

    def test_ml_opposite_side_veto_skips_conflicting_baseline_candidates(self) -> None:
        pipeline = StrategyPipeline(
            dislocation_engine=_FakeDislocationEngine(
                signals={
                    "altA": _signal(
                        candidate_name="altA",
                        action="BET",
                        bet_side="Bull",
                        bet_size_bnb=0.2,
                        expected_profit_bnb=0.03,
                        selector_score_bnb=0.08,
                        skip_reason=None,
                    ),
                    "altB": _signal(
                        candidate_name="altB",
                        action="BET",
                        bet_side="Bear",
                        bet_size_bnb=0.2,
                        expected_profit_bnb=0.02,
                        selector_score_bnb=0.07,
                        skip_reason=None,
                    ),
                }
            ),
            router=_FakeRouter(),
            treasury_fee_fraction=0.03,
            ml_candidate_adapter=_FakeMlAdapter(
                signal=_signal(
                    candidate_name="mlwf_bestset_adapt_v1",
                    action="BET",
                    bet_side="Bull",
                    bet_size_bnb=0.2,
                    expected_profit_bnb=0.01,
                    selector_score_bnb=0.004,
                    skip_reason=None,
                ),
                emit_candidate=False,
                veto_opposite_side_candidates=True,
                veto_untradeable_candidates=False,
            ),
        )

        signals = pipeline.candidate_signals_for_open_round(round_t=_round())

        self.assertEqual("BET", signals["altA"].action)
        self.assertEqual("SKIP", signals["altB"].action)
        self.assertEqual("ml_veto_opposite_side", signals["altB"].skip_reason)

    def test_ml_untradeable_veto_skips_all_baseline_candidates(self) -> None:
        pipeline = StrategyPipeline(
            dislocation_engine=_FakeDislocationEngine(
                signals={
                    "altA": _signal(
                        candidate_name="altA",
                        action="BET",
                        bet_side="Bull",
                        bet_size_bnb=0.2,
                        expected_profit_bnb=0.03,
                        selector_score_bnb=0.08,
                        skip_reason=None,
                    )
                }
            ),
            router=_FakeRouter(),
            treasury_fee_fraction=0.03,
            ml_candidate_adapter=_FakeMlAdapter(
                signal=_signal(
                    candidate_name="mlwf_bestset_adapt_v1",
                    action="SKIP",
                    bet_side=None,
                    bet_size_bnb=0.0,
                    expected_profit_bnb=None,
                    selector_score_bnb=None,
                    skip_reason="predictability_below_min",
                ),
                emit_candidate=False,
                veto_opposite_side_candidates=False,
                veto_untradeable_candidates=True,
            ),
        )

        signals = pipeline.candidate_signals_for_open_round(round_t=_round())

        self.assertEqual("SKIP", signals["altA"].action)
        self.assertEqual("ml_veto_untradeable", signals["altA"].skip_reason)

    def test_ml_expected_net_below_min_also_vetoes_baseline_candidates(self) -> None:
        pipeline = StrategyPipeline(
            dislocation_engine=_FakeDislocationEngine(
                signals={
                    "altA": _signal(
                        candidate_name="altA",
                        action="BET",
                        bet_side="Bull",
                        bet_size_bnb=0.2,
                        expected_profit_bnb=0.03,
                        selector_score_bnb=0.08,
                        skip_reason=None,
                    )
                }
            ),
            router=_FakeRouter(),
            treasury_fee_fraction=0.03,
            ml_candidate_adapter=_FakeMlAdapter(
                signal=_signal(
                    candidate_name="mlwf_bestset_adapt_v1",
                    action="SKIP",
                    bet_side=None,
                    bet_size_bnb=0.0,
                    expected_profit_bnb=None,
                    selector_score_bnb=None,
                    skip_reason="expected_net_below_min",
                ),
                emit_candidate=False,
                veto_opposite_side_candidates=False,
                veto_untradeable_candidates=True,
            ),
        )

        signals = pipeline.candidate_signals_for_open_round(round_t=_round())

        self.assertEqual("SKIP", signals["altA"].action)
        self.assertEqual("ml_veto_untradeable", signals["altA"].skip_reason)

    def test_ml_candidate_expected_net_veto_skips_only_rejected_candidates(self) -> None:
        pipeline = StrategyPipeline(
            dislocation_engine=_FakeDislocationEngine(
                signals={
                    "altA": _signal(
                        candidate_name="altA",
                        action="BET",
                        bet_side="Bull",
                        bet_size_bnb=0.2,
                        expected_profit_bnb=0.03,
                        selector_score_bnb=0.08,
                        skip_reason=None,
                    ),
                    "altB": _signal(
                        candidate_name="altB",
                        action="BET",
                        bet_side="Bear",
                        bet_size_bnb=0.2,
                        expected_profit_bnb=0.02,
                        selector_score_bnb=0.07,
                        skip_reason=None,
                    ),
                }
            ),
            router=_FakeRouter(),
            treasury_fee_fraction=0.03,
            ml_candidate_adapter=_FakeMlAdapter(
                signal=_signal(
                    candidate_name="mlwf_bestset_adapt_v1",
                    action="SKIP",
                    bet_side=None,
                    bet_size_bnb=0.0,
                    expected_profit_bnb=None,
                    selector_score_bnb=None,
                    skip_reason="predictability_below_min",
                ),
                emit_candidate=False,
                veto_opposite_side_candidates=False,
                veto_untradeable_candidates=False,
                veto_candidate_expected_net_below_min=True,
                candidate_veto_names=("altB",),
            ),
        )

        signals = pipeline.candidate_signals_for_open_round(round_t=_round())

        self.assertEqual("BET", signals["altA"].action)
        self.assertEqual("SKIP", signals["altB"].action)
        self.assertEqual("ml_veto_candidate_expected_net_below_min", signals["altB"].skip_reason)

    def test_ml_candidate_expected_net_rescores_baseline_candidates(self) -> None:
        pipeline = StrategyPipeline(
            dislocation_engine=_FakeDislocationEngine(
                signals={
                    "altA": _signal(
                        candidate_name="altA",
                        action="BET",
                        bet_side="Bull",
                        bet_size_bnb=0.2,
                        expected_profit_bnb=0.03,
                        selector_score_bnb=0.08,
                        skip_reason=None,
                    ),
                    "altB": _signal(
                        candidate_name="altB",
                        action="BET",
                        bet_side="Bear",
                        bet_size_bnb=0.2,
                        expected_profit_bnb=0.02,
                        selector_score_bnb=0.07,
                        skip_reason=None,
                    ),
                }
            ),
            router=_FakeRouter(),
            treasury_fee_fraction=0.03,
            ml_candidate_adapter=_FakeMlAdapter(
                signal=_signal(
                    candidate_name="mlwf_bestset_adapt_v1",
                    action="SKIP",
                    bet_side=None,
                    bet_size_bnb=0.0,
                    expected_profit_bnb=None,
                    selector_score_bnb=None,
                    skip_reason="predictability_below_min",
                ),
                emit_candidate=False,
                veto_opposite_side_candidates=False,
                veto_untradeable_candidates=False,
                rescore_baseline_candidates_with_expected_net=True,
                candidate_expected_net_by_name={"altA": 0.01, "altB": 0.09},
            ),
        )

        signals = pipeline.candidate_signals_for_open_round(round_t=_round())

        self.assertEqual(0.01, float(signals["altA"].expected_profit_bnb))
        self.assertEqual(0.01, float(signals["altA"].selector_score_bnb))
        self.assertEqual(0.09, float(signals["altB"].expected_profit_bnb))
        self.assertEqual(0.09, float(signals["altB"].selector_score_bnb))

    def test_pipeline_feeds_realized_candidate_outcomes_back_to_ml_adapter(self) -> None:
        ml_adapter = _FakeMlAdapter(
            signal=_signal(
                candidate_name="mlwf_bestset_adapt_v1",
                action="SKIP",
                bet_side=None,
                bet_size_bnb=0.0,
                expected_profit_bnb=None,
                selector_score_bnb=None,
                skip_reason="predictability_below_min",
            ),
            emit_candidate=False,
            veto_opposite_side_candidates=False,
            veto_untradeable_candidates=False,
        )
        pipeline = StrategyPipeline(
            dislocation_engine=_FakeDislocationEngine(
                signals={
                    "altA": _signal(
                        candidate_name="altA",
                        action="BET",
                        bet_side="Bull",
                        bet_size_bnb=0.2,
                        expected_profit_bnb=0.03,
                        selector_score_bnb=0.08,
                        skip_reason=None,
                    )
                }
            ),
            router=_FakeRouter(),
            treasury_fee_fraction=0.03,
            ml_candidate_adapter=ml_adapter,
        )

        closed_round = Round(
            epoch=7,
            start_at=1000,
            lock_at=1300,
            close_at=1600,
            lock_price=600.0,
            close_price=601.0,
            position="Bull",
            failed=False,
            bets=(),
        )

        pipeline.candidate_signals_for_open_round(round_t=closed_round)
        pipeline.settle_closed_rounds(rounds=[closed_round])

        self.assertEqual(1, len(ml_adapter.observed_settlements))
        epoch, realized = ml_adapter.observed_settlements[0]
        self.assertEqual(7, int(epoch))
        self.assertIn("altA", realized)


if __name__ == "__main__":
    unittest.main()
