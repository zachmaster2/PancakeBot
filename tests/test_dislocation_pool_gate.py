"""Deterministic tests for dislocation pool-total gate parsing and logic."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from pancakebot.config.load_config import load_app_config
from pancakebot.core.errors import InvariantError
from pancakebot.domain.strategy.dislocation_engine import (
    _anti_martingale_next_scale,
    _circuit_breaker_skip_rounds_for_level,
    _drawdown_stake_scale,
    _effective_ev_pools,
    _expected_net_min_for_side,
    _flow_gate_relaxed_for_dislocation,
    _late_model_conflict_flip_side,
    _late_model_neutral_filter_triggers,
    _late_support_ev_adjustment,
    _late_side_support_skip_reason,
    _late_model_veto_triggers,
    _pool_total_gate_skip_reason,
    _precutoff_shock_filter_triggers,
    _robust_selected_ev_min,
    _side_allowed,
    _stake_mode_uses_projected_pool_ev,
    _to_candidate_config,
)
from pancakebot.domain.types import Bet, Round


class DislocationPoolGateTests(unittest.TestCase):
    def test_candidate_defaults_use_cutoff_only_gate(self) -> None:
        cfg = load_app_config("config.toml")
        candidate = cfg.strategy.dislocation.candidates[0]
        self.assertEqual("cutoff_only", candidate.pool_total_gate_mode)
        self.assertAlmostEqual(1.0, float(candidate.projected_final_pool_multiplier))
        self.assertAlmostEqual(0.0, float(candidate.projected_final_pool_total_min_bnb))
        self.assertFalse(bool(candidate.robust_ev_veto_enabled))
        self.assertEqual(200, int(candidate.robust_ev_veto_min_history))
        self.assertEqual(4000, int(candidate.robust_ev_veto_window))
        self.assertAlmostEqual(0.5, float(candidate.robust_ev_veto_low_inflow_mult))
        self.assertAlmostEqual(0.2, float(candidate.robust_ev_veto_extreme_inflow_mult))
        self.assertAlmostEqual(0.15, float(candidate.robust_ev_veto_adverse_skew))
        self.assertAlmostEqual(0.0, float(candidate.robust_ev_veto_min_expected_net_bnb))
        self.assertFalse(bool(candidate.shock_filter_enabled))
        self.assertEqual(20, int(candidate.shock_filter_window_seconds))
        self.assertAlmostEqual(0.25, float(candidate.shock_filter_min_window_total_bnb))
        self.assertAlmostEqual(0.8, float(candidate.shock_filter_min_abs_imbalance))
        self.assertAlmostEqual(2.5, float(candidate.shock_filter_min_surge_ratio))
        self.assertAlmostEqual(0.0, float(candidate.nowcast_market_gap_min))
        self.assertEqual("both", str(candidate.allowed_sides))
        self.assertAlmostEqual(1.0, float(candidate.flow_gate_relax_dislocation_min))
        self.assertFalse(bool(candidate.late_model_conflict_flip_enabled))
        self.assertFalse(bool(candidate.late_model_veto_enabled))
        self.assertAlmostEqual(0.45, float(candidate.late_model_veto_min_late_ratio))
        self.assertAlmostEqual(0.35, float(candidate.late_model_veto_min_abs_imbalance))
        self.assertFalse(bool(candidate.late_model_neutral_filter_enabled))
        self.assertAlmostEqual(0.0, float(candidate.late_model_neutral_min_late_ratio))
        self.assertAlmostEqual(1.0, float(candidate.late_model_neutral_max_abs_imbalance))
        self.assertFalse(bool(candidate.drawdown_stake_guard_enabled))
        self.assertAlmostEqual(0.0, float(candidate.drawdown_stake_guard_start_bnb))
        self.assertAlmostEqual(0.0, float(candidate.drawdown_stake_guard_full_bnb))
        self.assertAlmostEqual(1.0, float(candidate.drawdown_stake_guard_min_scale))
        self.assertFalse(bool(candidate.anti_martingale_enabled))
        self.assertAlmostEqual(1.15, float(candidate.anti_martingale_win_multiplier))
        self.assertAlmostEqual(0.9, float(candidate.anti_martingale_loss_multiplier))
        self.assertAlmostEqual(0.5, float(candidate.anti_martingale_min_scale))
        self.assertAlmostEqual(1.5, float(candidate.anti_martingale_max_scale))
        self.assertFalse(bool(candidate.circuit_breaker_enabled))
        self.assertAlmostEqual(0.0, float(candidate.circuit_breaker_drawdown_trigger_bnb))
        self.assertEqual(0, int(candidate.circuit_breaker_base_skip_rounds))
        self.assertAlmostEqual(1.5, float(candidate.circuit_breaker_escalation_multiplier))
        self.assertEqual(200, int(candidate.circuit_breaker_escalation_window_rounds))
        self.assertEqual(6, int(candidate.circuit_breaker_max_level))
        self.assertEqual(0, int(candidate.circuit_breaker_max_skip_rounds))
        self.assertEqual(0, int(candidate.circuit_breaker_reentry_rounds))
        self.assertAlmostEqual(1.0, float(candidate.circuit_breaker_reentry_scale))
        self.assertAlmostEqual(0.0, float(candidate.bull_expected_net_extra_min_bnb))
        self.assertAlmostEqual(0.0, float(candidate.bear_expected_net_extra_min_bnb))
        self.assertAlmostEqual(0.0, float(candidate.bull_late_min_ratio))
        self.assertAlmostEqual(-1.0, float(candidate.bull_late_min_imbalance))
        self.assertAlmostEqual(0.0, float(candidate.bear_late_min_ratio))
        self.assertAlmostEqual(1.0, float(candidate.bear_late_max_imbalance))
        self.assertAlmostEqual(0.0, float(candidate.late_support_ev_scale_bnb))

    def test_parse_projected_gate_fields(self) -> None:
        base_text = Path("config.toml").read_text(encoding="utf-8")
        patched = base_text.replace(
            'name = "disloc_altA_20260227_x80"\n',
            (
                'name = "disloc_altA_20260227_x80"\n'
                'pool_total_gate_mode = "projected_final_only"\n'
                "projected_final_pool_multiplier = 1.8\n"
                "projected_final_pool_total_min_bnb = 2.0\n"
            ),
            1,
        )

        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "config_projected.toml"
            cfg_path.write_text(patched, encoding="utf-8")
            cfg = load_app_config(str(cfg_path))

        candidate = cfg.strategy.dislocation.candidates[0]
        self.assertEqual("projected_final_only", candidate.pool_total_gate_mode)
        self.assertAlmostEqual(1.8, float(candidate.projected_final_pool_multiplier))
        self.assertAlmostEqual(2.0, float(candidate.projected_final_pool_total_min_bnb))

    def test_invalid_pool_total_gate_mode_rejected(self) -> None:
        base_text = Path("config.toml").read_text(encoding="utf-8")
        patched = base_text.replace(
            'name = "disloc_altA_20260227_x80"\n',
            (
                'name = "disloc_altA_20260227_x80"\n'
                'pool_total_gate_mode = "bad_mode"\n'
            ),
            1,
        )

        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "config_invalid.toml"
            cfg_path.write_text(patched, encoding="utf-8")
            with self.assertRaises(InvariantError):
                load_app_config(str(cfg_path))

    def test_invalid_drawdown_guard_min_scale_rejected(self) -> None:
        base_text = Path("config.toml").read_text(encoding="utf-8")
        patched = base_text.replace(
            'name = "disloc_altA_20260227_x80"\n',
            (
                'name = "disloc_altA_20260227_x80"\n'
                "drawdown_stake_guard_min_scale = 1.5\n"
            ),
            1,
        )

        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "config_invalid_drawdown_scale.toml"
            cfg_path.write_text(patched, encoding="utf-8")
            with self.assertRaises(InvariantError):
                load_app_config(str(cfg_path))

    def test_invalid_anti_martingale_scale_bounds_rejected(self) -> None:
        base_text = Path("config.toml").read_text(encoding="utf-8")
        patched = base_text.replace(
            'name = "disloc_altA_20260227_x80"\n',
            (
                'name = "disloc_altA_20260227_x80"\n'
                "anti_martingale_min_scale = 1.2\n"
                "anti_martingale_max_scale = 1.1\n"
            ),
            1,
        )

        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "config_invalid_anti_scale.toml"
            cfg_path.write_text(patched, encoding="utf-8")
            with self.assertRaises(InvariantError):
                load_app_config(str(cfg_path))

    def test_circuit_breaker_requires_positive_trigger_and_skip_when_enabled(self) -> None:
        base_text = Path("config.toml").read_text(encoding="utf-8")
        patched = base_text.replace(
            'name = "disloc_altA_20260227_x80"\n',
            (
                'name = "disloc_altA_20260227_x80"\n'
                "circuit_breaker_enabled = true\n"
            ),
            1,
        )

        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "config_invalid_cb_enabled.toml"
            cfg_path.write_text(patched, encoding="utf-8")
            with self.assertRaises(InvariantError):
                load_app_config(str(cfg_path))

    def test_invalid_allowed_sides_rejected(self) -> None:
        base_text = Path("config.toml").read_text(encoding="utf-8")
        patched = base_text.replace(
            'name = "disloc_altA_20260227_x80"\n',
            (
                'name = "disloc_altA_20260227_x80"\n'
                'allowed_sides = "invalid_mode"\n'
            ),
            1,
        )

        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "config_invalid_allowed_sides.toml"
            cfg_path.write_text(patched, encoding="utf-8")
            with self.assertRaises(InvariantError):
                load_app_config(str(cfg_path))

    def test_invalid_bear_expected_net_extra_min_rejected(self) -> None:
        base_text = Path("config.toml").read_text(encoding="utf-8")
        patched = base_text.replace(
            'name = "disloc_altA_20260227_x80"\n',
            (
                'name = "disloc_altA_20260227_x80"\n'
                "bear_expected_net_extra_min_bnb = -0.01\n"
            ),
            1,
        )

        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "config_invalid_bear_expected_extra.toml"
            cfg_path.write_text(patched, encoding="utf-8")
            with self.assertRaises(InvariantError):
                load_app_config(str(cfg_path))

    def test_invalid_bull_expected_net_extra_min_rejected(self) -> None:
        base_text = Path("config.toml").read_text(encoding="utf-8")
        patched = base_text.replace(
            'name = "disloc_altA_20260227_x80"\n',
            (
                'name = "disloc_altA_20260227_x80"\n'
                "bull_expected_net_extra_min_bnb = -0.01\n"
            ),
            1,
        )

        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "config_invalid_bull_expected_extra.toml"
            cfg_path.write_text(patched, encoding="utf-8")
            with self.assertRaises(InvariantError):
                load_app_config(str(cfg_path))

    def test_invalid_bull_late_min_ratio_rejected(self) -> None:
        base_text = Path("config.toml").read_text(encoding="utf-8")
        patched = base_text.replace(
            'name = "disloc_altA_20260227_x80"\n',
            (
                'name = "disloc_altA_20260227_x80"\n'
                "bull_late_min_ratio = 1.2\n"
            ),
            1,
        )

        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "config_invalid_bull_late_ratio.toml"
            cfg_path.write_text(patched, encoding="utf-8")
            with self.assertRaises(InvariantError):
                load_app_config(str(cfg_path))

    def test_invalid_bear_late_max_imbalance_rejected(self) -> None:
        base_text = Path("config.toml").read_text(encoding="utf-8")
        patched = base_text.replace(
            'name = "disloc_altA_20260227_x80"\n',
            (
                'name = "disloc_altA_20260227_x80"\n'
                "bear_late_max_imbalance = 1.2\n"
            ),
            1,
        )

        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "config_invalid_bear_late_imbalance.toml"
            cfg_path.write_text(patched, encoding="utf-8")
            with self.assertRaises(InvariantError):
                load_app_config(str(cfg_path))

    def test_invalid_late_support_ev_scale_rejected(self) -> None:
        base_text = Path("config.toml").read_text(encoding="utf-8")
        patched = base_text.replace(
            'name = "disloc_altA_20260227_x80"\n',
            (
                'name = "disloc_altA_20260227_x80"\n'
                "late_support_ev_scale_bnb = -0.01\n"
            ),
            1,
        )

        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "config_invalid_late_support_scale.toml"
            cfg_path.write_text(patched, encoding="utf-8")
            with self.assertRaises(InvariantError):
                load_app_config(str(cfg_path))

    def test_invalid_flow_gate_relax_dislocation_min_rejected(self) -> None:
        base_text = Path("config.toml").read_text(encoding="utf-8")
        patched = base_text.replace(
            'name = "disloc_altA_20260227_x80"\n',
            (
                'name = "disloc_altA_20260227_x80"\n'
                "flow_gate_relax_dislocation_min = 1.5\n"
            ),
            1,
        )

        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "config_invalid_flow_gate_relax.toml"
            cfg_path.write_text(patched, encoding="utf-8")
            with self.assertRaises(InvariantError):
                load_app_config(str(cfg_path))

    def test_pool_total_gate_logic_cutoff_and_projected_modes(self) -> None:
        app_cfg = load_app_config("config.toml")
        candidate_cfg = app_cfg.strategy.dislocation.candidates[0]
        engine_cfg = _to_candidate_config(
            cfg=candidate_cfg,
            cutoff_seconds=int(app_cfg.cutoff_seconds),
        )

        cutoff_cfg = replace(
            engine_cfg,
            cutoff_pool_total_min_bnb=2.0,
            pool_total_gate_mode="cutoff_only",
        )
        self.assertEqual(
            "cutoff_pool_below_min_total",
            _pool_total_gate_skip_reason(
                cfg=cutoff_cfg,
                cutoff_pool_total_bnb=1.5,
                projected_final_pool_total_bnb=None,
            ),
        )
        self.assertIsNone(
            _pool_total_gate_skip_reason(
                cfg=cutoff_cfg,
                cutoff_pool_total_bnb=2.0,
                projected_final_pool_total_bnb=None,
            )
        )

        projected_cfg = replace(
            engine_cfg,
            pool_total_gate_mode="projected_final_only",
            projected_final_pool_multiplier=1.8,
            projected_final_pool_total_min_bnb=2.0,
        )
        self.assertEqual(
            "projected_final_pool_below_min_total",
            _pool_total_gate_skip_reason(
                cfg=projected_cfg,
                cutoff_pool_total_bnb=1.0,
                projected_final_pool_total_bnb=None,
            ),
        )
        self.assertIsNone(
            _pool_total_gate_skip_reason(
                cfg=projected_cfg,
                cutoff_pool_total_bnb=1.2,
                projected_final_pool_total_bnb=None,
            )
        )

    def test_pool_total_gate_logic_projected_model_mode(self) -> None:
        app_cfg = load_app_config("config.toml")
        candidate_cfg = app_cfg.strategy.dislocation.candidates[0]
        engine_cfg = _to_candidate_config(
            cfg=candidate_cfg,
            cutoff_seconds=int(app_cfg.cutoff_seconds),
        )
        model_cfg = replace(
            engine_cfg,
            pool_total_gate_mode="projected_final_model_only",
            projected_final_pool_total_min_bnb=2.0,
        )

        self.assertEqual(
            "projected_final_pool_model_unavailable",
            _pool_total_gate_skip_reason(
                cfg=model_cfg,
                cutoff_pool_total_bnb=1.0,
                projected_final_pool_total_bnb=None,
            ),
        )
        self.assertEqual(
            "projected_final_pool_below_min_total",
            _pool_total_gate_skip_reason(
                cfg=model_cfg,
                cutoff_pool_total_bnb=1.0,
                projected_final_pool_total_bnb=1.7,
            ),
        )
        self.assertIsNone(
            _pool_total_gate_skip_reason(
                cfg=model_cfg,
                cutoff_pool_total_bnb=1.0,
                projected_final_pool_total_bnb=2.1,
            )
        )

    def test_projected_ev_stake_mode_flag(self) -> None:
        self.assertTrue(_stake_mode_uses_projected_pool_ev(stake_mode="ev_scaled_projected"))
        self.assertTrue(_stake_mode_uses_projected_pool_ev(stake_mode="ev_optimal_projected"))
        self.assertFalse(_stake_mode_uses_projected_pool_ev(stake_mode="ev_scaled"))
        self.assertFalse(_stake_mode_uses_projected_pool_ev(stake_mode="fixed"))

    def test_side_allowed_modes(self) -> None:
        self.assertTrue(_side_allowed(side="BULL", allowed_sides="both"))
        self.assertTrue(_side_allowed(side="BEAR", allowed_sides="both"))
        self.assertTrue(_side_allowed(side="BULL", allowed_sides="bull_only"))
        self.assertFalse(_side_allowed(side="BEAR", allowed_sides="bull_only"))
        self.assertTrue(_side_allowed(side="BEAR", allowed_sides="bear_only"))
        self.assertFalse(_side_allowed(side="BULL", allowed_sides="bear_only"))

    def test_expected_net_min_for_side_applies_bear_extra(self) -> None:
        app_cfg = load_app_config("config.toml")
        candidate_cfg = app_cfg.strategy.dislocation.candidates[0]
        engine_cfg = _to_candidate_config(
            cfg=candidate_cfg,
            cutoff_seconds=int(app_cfg.cutoff_seconds),
        )
        cfg = replace(
            engine_cfg,
            expected_net_min_bnb=0.04,
            bull_expected_net_extra_min_bnb=0.02,
            bear_expected_net_extra_min_bnb=0.03,
        )
        self.assertAlmostEqual(0.06, _expected_net_min_for_side(cfg=cfg, side="BULL"))
        self.assertAlmostEqual(0.07, _expected_net_min_for_side(cfg=cfg, side="BEAR"))

    def test_flow_gate_relaxed_for_large_dislocation(self) -> None:
        app_cfg = load_app_config("config.toml")
        candidate_cfg = app_cfg.strategy.dislocation.candidates[0]
        engine_cfg = _to_candidate_config(
            cfg=candidate_cfg,
            cutoff_seconds=int(app_cfg.cutoff_seconds),
        )
        cfg = replace(engine_cfg, flow_gate_relax_dislocation_min=0.06)
        self.assertFalse(_flow_gate_relaxed_for_dislocation(cfg=cfg, dislocation_bull=0.04))
        self.assertTrue(_flow_gate_relaxed_for_dislocation(cfg=cfg, dislocation_bull=0.08))

    def test_drawdown_stake_scale_linear_haircut(self) -> None:
        app_cfg = load_app_config("config.toml")
        candidate_cfg = app_cfg.strategy.dislocation.candidates[0]
        engine_cfg = _to_candidate_config(
            cfg=candidate_cfg,
            cutoff_seconds=int(app_cfg.cutoff_seconds),
        )
        guarded_cfg = replace(
            engine_cfg,
            drawdown_stake_guard_enabled=True,
            drawdown_stake_guard_start_bnb=1.0,
            drawdown_stake_guard_full_bnb=2.0,
            drawdown_stake_guard_min_scale=0.3,
        )
        self.assertAlmostEqual(
            1.0,
            _drawdown_stake_scale(
                cfg=guarded_cfg,
                shadow_bankroll_bnb=49.1,
                shadow_peak_bankroll_bnb=50.0,
            ),
        )
        self.assertAlmostEqual(
            0.65,
            _drawdown_stake_scale(
                cfg=guarded_cfg,
                shadow_bankroll_bnb=48.5,
                shadow_peak_bankroll_bnb=50.0,
            ),
            places=6,
        )
        self.assertAlmostEqual(
            0.3,
            _drawdown_stake_scale(
                cfg=guarded_cfg,
                shadow_bankroll_bnb=47.8,
                shadow_peak_bankroll_bnb=50.0,
            ),
            places=6,
        )

    def test_anti_martingale_next_scale_clamps_inside_bounds(self) -> None:
        app_cfg = load_app_config("config.toml")
        candidate_cfg = app_cfg.strategy.dislocation.candidates[0]
        engine_cfg = _to_candidate_config(
            cfg=candidate_cfg,
            cutoff_seconds=int(app_cfg.cutoff_seconds),
        )
        am_cfg = replace(
            engine_cfg,
            anti_martingale_enabled=True,
            anti_martingale_win_multiplier=1.2,
            anti_martingale_loss_multiplier=0.85,
            anti_martingale_min_scale=0.5,
            anti_martingale_max_scale=1.5,
        )
        self.assertAlmostEqual(
            1.2,
            _anti_martingale_next_scale(
                cfg=am_cfg,
                prev_scale=1.0,
                realized_profit_bnb=0.01,
            ),
        )
        self.assertAlmostEqual(
            1.02,
            _anti_martingale_next_scale(
                cfg=am_cfg,
                prev_scale=1.2,
                realized_profit_bnb=-0.01,
            ),
            places=6,
        )
        self.assertAlmostEqual(
            1.5,
            _anti_martingale_next_scale(
                cfg=am_cfg,
                prev_scale=2.0,
                realized_profit_bnb=0.01,
            ),
            places=6,
        )
        self.assertAlmostEqual(
            0.5,
            _anti_martingale_next_scale(
                cfg=am_cfg,
                prev_scale=0.1,
                realized_profit_bnb=-0.01,
            ),
            places=6,
        )

    def test_circuit_breaker_skip_rounds_respects_cap(self) -> None:
        app_cfg = load_app_config("config.toml")
        candidate_cfg = app_cfg.strategy.dislocation.candidates[0]
        engine_cfg = _to_candidate_config(
            cfg=candidate_cfg,
            cutoff_seconds=int(app_cfg.cutoff_seconds),
        )
        cb_cfg = replace(
            engine_cfg,
            circuit_breaker_enabled=True,
            circuit_breaker_drawdown_trigger_bnb=1.0,
            circuit_breaker_base_skip_rounds=100,
            circuit_breaker_escalation_multiplier=1.5,
            circuit_breaker_max_skip_rounds=180,
        )
        self.assertEqual(100, _circuit_breaker_skip_rounds_for_level(cfg=cb_cfg, level=1))
        self.assertEqual(150, _circuit_breaker_skip_rounds_for_level(cfg=cb_cfg, level=2))
        self.assertEqual(180, _circuit_breaker_skip_rounds_for_level(cfg=cb_cfg, level=3))

    def test_effective_ev_pools_scales_model_late_inflow(self) -> None:
        app_cfg = load_app_config("config.toml")
        candidate_cfg = app_cfg.strategy.dislocation.candidates[0]
        engine_cfg = _to_candidate_config(
            cfg=candidate_cfg,
            cutoff_seconds=int(app_cfg.cutoff_seconds),
        )
        projected_cfg = replace(
            engine_cfg,
            stake_mode="ev_optimal_projected",
            projected_final_pool_multiplier=0.5,
        )
        bull_ev, bear_ev = _effective_ev_pools(
            cfg=projected_cfg,
            bull_pool_cutoff_bnb=1.0,
            bear_pool_cutoff_bnb=1.0,
            projected_final_pool_bull_bnb=2.0,
            projected_final_pool_bear_bnb=1.4,
        )
        self.assertAlmostEqual(1.5, float(bull_ev))
        self.assertAlmostEqual(1.2, float(bear_ev))

    def test_effective_ev_pools_fallback_when_not_projected_mode(self) -> None:
        app_cfg = load_app_config("config.toml")
        candidate_cfg = app_cfg.strategy.dislocation.candidates[0]
        engine_cfg = _to_candidate_config(
            cfg=candidate_cfg,
            cutoff_seconds=int(app_cfg.cutoff_seconds),
        )
        plain_cfg = replace(engine_cfg, stake_mode="ev_optimal")
        bull_ev, bear_ev = _effective_ev_pools(
            cfg=plain_cfg,
            bull_pool_cutoff_bnb=1.0,
            bear_pool_cutoff_bnb=1.0,
            projected_final_pool_bull_bnb=2.0,
            projected_final_pool_bear_bnb=1.4,
        )
        self.assertAlmostEqual(1.0, float(bull_ev))
        self.assertAlmostEqual(1.0, float(bear_ev))

    def test_robust_selected_ev_min_is_conservative(self) -> None:
        app_cfg = load_app_config("config.toml")
        candidate_cfg = app_cfg.strategy.dislocation.candidates[0]
        engine_cfg = _to_candidate_config(
            cfg=candidate_cfg,
            cutoff_seconds=int(app_cfg.cutoff_seconds),
        )
        robust_cfg = replace(
            engine_cfg,
            robust_ev_veto_enabled=True,
            robust_ev_veto_low_inflow_mult=0.5,
            robust_ev_veto_extreme_inflow_mult=0.2,
            robust_ev_veto_adverse_skew=0.15,
        )
        robust_ev = _robust_selected_ev_min(
            cfg=robust_cfg,
            side="BULL",
            p_nowcast_bull=0.55,
            bull_pool_cutoff_bnb=1.2,
            bear_pool_cutoff_bnb=1.0,
            robust_late_inflow_ratio=0.8,
            robust_late_bull_share=0.5,
            treasury_fee_fraction=0.03,
        )
        self.assertTrue(float(robust_ev) < 0.2)

    def test_precutoff_shock_filter_triggers_on_one_sided_spike(self) -> None:
        app_cfg = load_app_config("config.toml")
        candidate_cfg = app_cfg.strategy.dislocation.candidates[0]
        engine_cfg = _to_candidate_config(
            cfg=candidate_cfg,
            cutoff_seconds=int(app_cfg.cutoff_seconds),
        )
        shock_cfg = replace(
            engine_cfg,
            shock_filter_enabled=True,
            shock_filter_window_seconds=20,
            shock_filter_min_window_total_bnb=0.2,
            shock_filter_min_abs_imbalance=0.8,
            shock_filter_min_surge_ratio=2.0,
        )
        round_t = Round(
            epoch=1,
            start_at=1_000_000,
            lock_at=1_000_300,
            close_at=1_000_600,
            lock_price=1.0,
            close_price=1.0,
            position="Bull",
            failed=False,
            bets=(
                Bet(wallet_address="0x1", amount_wei=int(1e17), position="Bull", created_at=1_000_050),
                Bet(wallet_address="0x2", amount_wei=int(1e17), position="Bear", created_at=1_000_080),
                Bet(wallet_address="0x3", amount_wei=int(6e17), position="Bull", created_at=1_000_279),
            ),
        )
        self.assertTrue(_precutoff_shock_filter_triggers(round_t=round_t, cfg=shock_cfg))

    def test_late_model_veto_triggers_only_when_opposite_flow_is_strong(self) -> None:
        app_cfg = load_app_config("config.toml")
        candidate_cfg = app_cfg.strategy.dislocation.candidates[0]
        engine_cfg = _to_candidate_config(
            cfg=candidate_cfg,
            cutoff_seconds=int(app_cfg.cutoff_seconds),
        )
        veto_cfg = replace(
            engine_cfg,
            late_model_veto_enabled=True,
            late_model_veto_min_late_ratio=0.5,
            late_model_veto_min_abs_imbalance=0.6,
        )

        self.assertTrue(
            _late_model_veto_triggers(
                cfg=veto_cfg,
                side="BULL",
                bull_pool_cutoff_bnb=1.0,
                bear_pool_cutoff_bnb=1.0,
                projected_final_pool_bull_bnb=1.1,
                projected_final_pool_bear_bnb=2.5,
            )
        )
        self.assertFalse(
            _late_model_veto_triggers(
                cfg=veto_cfg,
                side="BEAR",
                bull_pool_cutoff_bnb=1.0,
                bear_pool_cutoff_bnb=1.0,
                projected_final_pool_bull_bnb=1.1,
                projected_final_pool_bear_bnb=2.5,
            )
        )

    def test_parse_late_model_neutral_fields(self) -> None:
        base_text = Path("config.toml").read_text(encoding="utf-8")
        patched = base_text.replace(
            'name = "disloc_altA_20260227_x80"\n',
            (
                'name = "disloc_altA_20260227_x80"\n'
                "late_model_conflict_flip_enabled = true\n"
                "late_model_neutral_filter_enabled = true\n"
                "late_model_neutral_min_late_ratio = 0.12\n"
                "late_model_neutral_max_abs_imbalance = 0.18\n"
            ),
            1,
        )

        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "config_late_neutral.toml"
            cfg_path.write_text(patched, encoding="utf-8")
            cfg = load_app_config(str(cfg_path))

        candidate = cfg.strategy.dislocation.candidates[0]
        self.assertTrue(bool(candidate.late_model_conflict_flip_enabled))
        self.assertTrue(bool(candidate.late_model_neutral_filter_enabled))
        self.assertAlmostEqual(0.12, float(candidate.late_model_neutral_min_late_ratio))
        self.assertAlmostEqual(0.18, float(candidate.late_model_neutral_max_abs_imbalance))

    def test_invalid_late_model_neutral_abs_imbalance_rejected(self) -> None:
        base_text = Path("config.toml").read_text(encoding="utf-8")
        patched = base_text.replace(
            'name = "disloc_altA_20260227_x80"\n',
            (
                'name = "disloc_altA_20260227_x80"\n'
                "late_model_neutral_max_abs_imbalance = 1.5\n"
            ),
            1,
        )

        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "config_invalid_late_neutral.toml"
            cfg_path.write_text(patched, encoding="utf-8")
            with self.assertRaises(InvariantError):
                load_app_config(str(cfg_path))

    def test_late_model_neutral_filter_triggers_only_for_balanced_late_flow(self) -> None:
        app_cfg = load_app_config("config.toml")
        candidate_cfg = app_cfg.strategy.dislocation.candidates[0]
        engine_cfg = _to_candidate_config(
            cfg=candidate_cfg,
            cutoff_seconds=int(app_cfg.cutoff_seconds),
        )
        neutral_cfg = replace(
            engine_cfg,
            late_model_neutral_filter_enabled=True,
            late_model_neutral_min_late_ratio=0.5,
            late_model_neutral_max_abs_imbalance=0.2,
        )

        self.assertTrue(
            _late_model_neutral_filter_triggers(
                cfg=neutral_cfg,
                bull_pool_cutoff_bnb=1.0,
                bear_pool_cutoff_bnb=1.0,
                projected_final_pool_bull_bnb=1.8,
                projected_final_pool_bear_bnb=1.6,
            )
        )
        self.assertFalse(
            _late_model_neutral_filter_triggers(
                cfg=neutral_cfg,
                bull_pool_cutoff_bnb=1.0,
                bear_pool_cutoff_bnb=1.0,
                projected_final_pool_bull_bnb=2.6,
                projected_final_pool_bear_bnb=1.2,
            )
        )

    def test_late_model_conflict_flip_side_flips_only_on_strong_opposing_late_flow(self) -> None:
        app_cfg = load_app_config("config.toml")
        candidate_cfg = app_cfg.strategy.dislocation.candidates[0]
        engine_cfg = _to_candidate_config(
            cfg=candidate_cfg,
            cutoff_seconds=int(app_cfg.cutoff_seconds),
        )
        cfg = replace(
            engine_cfg,
            late_model_conflict_flip_enabled=True,
            late_model_veto_enabled=True,
            late_model_veto_min_late_ratio=0.1,
            late_model_veto_min_abs_imbalance=0.2,
        )

        self.assertEqual(
            "BEAR",
            _late_model_conflict_flip_side(
                cfg=cfg,
                side="BULL",
                bull_pool_cutoff_bnb=1.0,
                bear_pool_cutoff_bnb=1.0,
                projected_final_pool_bull_bnb=1.1,
                projected_final_pool_bear_bnb=1.3,
            ),
        )
        self.assertEqual(
            "BULL",
            _late_model_conflict_flip_side(
                cfg=cfg,
                side="BULL",
                bull_pool_cutoff_bnb=1.0,
                bear_pool_cutoff_bnb=1.0,
                projected_final_pool_bull_bnb=1.24,
                projected_final_pool_bear_bnb=1.16,
            ),
        )

    def test_late_side_support_skip_reason_respects_side_specific_thresholds(self) -> None:
        app_cfg = load_app_config("config.toml")
        candidate_cfg = app_cfg.strategy.dislocation.candidates[0]
        engine_cfg = _to_candidate_config(
            cfg=candidate_cfg,
            cutoff_seconds=int(app_cfg.cutoff_seconds),
        )
        cfg = replace(
            engine_cfg,
            bull_late_min_ratio=0.2,
            bull_late_min_imbalance=0.1,
            bear_late_min_ratio=0.15,
            bear_late_max_imbalance=-0.1,
        )

        self.assertEqual(
            "projected_late_ratio_below_bull_min",
            _late_side_support_skip_reason(
                cfg=cfg,
                side="BULL",
                bull_pool_cutoff_bnb=1.0,
                bear_pool_cutoff_bnb=1.0,
                projected_final_pool_bull_bnb=1.2,
                projected_final_pool_bear_bnb=1.1,
            ),
        )
        self.assertEqual(
            "projected_late_bull_imbalance_below_min",
            _late_side_support_skip_reason(
                cfg=cfg,
                side="BULL",
                bull_pool_cutoff_bnb=1.0,
                bear_pool_cutoff_bnb=1.0,
                projected_final_pool_bull_bnb=1.265,
                projected_final_pool_bear_bnb=1.235,
            ),
        )
        self.assertEqual(
            "projected_late_ratio_below_bear_min",
            _late_side_support_skip_reason(
                cfg=cfg,
                side="BEAR",
                bull_pool_cutoff_bnb=1.0,
                bear_pool_cutoff_bnb=1.0,
                projected_final_pool_bull_bnb=1.05,
                projected_final_pool_bear_bnb=1.15,
            ),
        )
        self.assertEqual(
            "projected_late_bear_imbalance_above_max",
            _late_side_support_skip_reason(
                cfg=cfg,
                side="BEAR",
                bull_pool_cutoff_bnb=1.0,
                bear_pool_cutoff_bnb=1.0,
                projected_final_pool_bull_bnb=1.22,
                projected_final_pool_bear_bnb=1.18,
            ),
        )
        self.assertIsNone(
            _late_side_support_skip_reason(
                cfg=cfg,
                side="BULL",
                bull_pool_cutoff_bnb=1.0,
                bear_pool_cutoff_bnb=1.0,
                projected_final_pool_bull_bnb=1.32,
                projected_final_pool_bear_bnb=1.08,
            )
        )
        self.assertIsNone(
            _late_side_support_skip_reason(
                cfg=cfg,
                side="BEAR",
                bull_pool_cutoff_bnb=1.0,
                bear_pool_cutoff_bnb=1.0,
                projected_final_pool_bull_bnb=1.06,
                projected_final_pool_bear_bnb=1.34,
            )
        )

    def test_late_support_ev_adjustment_tracks_side_aligned_late_flow(self) -> None:
        app_cfg = load_app_config("config.toml")
        candidate_cfg = app_cfg.strategy.dislocation.candidates[0]
        engine_cfg = _to_candidate_config(
            cfg=candidate_cfg,
            cutoff_seconds=int(app_cfg.cutoff_seconds),
        )
        cfg = replace(engine_cfg, late_support_ev_scale_bnb=0.02)

        self.assertAlmostEqual(
            0.002,
            _late_support_ev_adjustment(
                cfg=cfg,
                side="BULL",
                bull_pool_cutoff_bnb=1.0,
                bear_pool_cutoff_bnb=1.0,
                projected_final_pool_bull_bnb=1.3,
                projected_final_pool_bear_bnb=1.1,
            ),
        )
        self.assertAlmostEqual(
            -0.002,
            _late_support_ev_adjustment(
                cfg=cfg,
                side="BEAR",
                bull_pool_cutoff_bnb=1.0,
                bear_pool_cutoff_bnb=1.0,
                projected_final_pool_bull_bnb=1.3,
                projected_final_pool_bear_bnb=1.1,
            ),
        )
        self.assertAlmostEqual(
            0.0,
            _late_support_ev_adjustment(
                cfg=cfg,
                side="BULL",
                bull_pool_cutoff_bnb=1.0,
                bear_pool_cutoff_bnb=1.0,
                projected_final_pool_bull_bnb=None,
                projected_final_pool_bear_bnb=1.2,
            ),
        )


if __name__ == "__main__":
    unittest.main()
