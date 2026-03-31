from __future__ import annotations

from types import SimpleNamespace
import unittest

from pancakebot.runtime.runtime_loop import _controller_decision_log_suffix


class RuntimeControllerLoggingTests(unittest.TestCase):
    def test_controller_decision_log_suffix_formats_profile_decision(self) -> None:
        decision = SimpleNamespace(
            controller_mode="absolute_best_with_skip",
            controller_estimator_mode="ewm_mean",
            controller_window_index=7,
            controller_lookback_windows_used=2,
            controller_selected_profile="disloc_stageG2_bullonly_recent5pct_v1",
            controller_selected_action="profile",
            controller_estimated_per_500=0.0834,
            controller_estimated_score_per_500=0.0811,
            controller_estimated_selected_bet_rate=0.057,
            controller_estimated_profiles_per_500_json=(
                '{"disloc_stageB_bullonly_recent8pct_v1":0.0123,'
                '"disloc_stageG2_bullonly_recent5pct_v1":0.0834,'
                '"disloc_altB_20260227_x80":-0.1010}'
            ),
            controller_estimated_profiles_score_per_500_json=(
                '{"disloc_stageB_bullonly_recent8pct_v1":0.0123,'
                '"disloc_stageG2_bullonly_recent5pct_v1":0.0811,'
                '"disloc_altB_20260227_x80":-0.1010}'
            ),
            controller_estimated_profiles_bet_rate_json=(
                '{"disloc_stageB_bullonly_recent8pct_v1":0.066,'
                '"disloc_stageG2_bullonly_recent5pct_v1":0.057,'
                '"disloc_altB_20260227_x80":0.041}'
            ),
        )

        suffix = _controller_decision_log_suffix(
            decision=decision,
            final_action="SKIP",
            final_skip_reason="selector_no_candidate",
        )

        self.assertIn("mode=absolute_best_with_skip", suffix)
        self.assertIn("estimator=ewm_mean", suffix)
        self.assertIn("win=7", suffix)
        self.assertIn("hist=2", suffix)
        self.assertIn("ctrl=profile", suffix)
        self.assertIn("pick=stageG2", suffix)
        self.assertIn("final=SKIP", suffix)
        self.assertIn("reason=selector_no_candidate", suffix)
        self.assertIn("*stageG2:+0.0834/+0.0811/5.7%", suffix)

    def test_controller_decision_log_suffix_empty_without_controller_mode(self) -> None:
        decision = SimpleNamespace(controller_mode="")
        suffix = _controller_decision_log_suffix(
            decision=decision,
            final_action="SKIP",
            final_skip_reason="selector_no_candidate",
        )
        self.assertEqual("", suffix)


if __name__ == "__main__":
    unittest.main()
