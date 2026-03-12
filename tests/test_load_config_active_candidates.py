"""Tests for dislocation active candidate allowlist parsing."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pancakebot.config.load_config import load_app_config
from pancakebot.core.errors import InvariantError

_ALT_A_NAME = "disloc_altA_20260227_x80"
_ALT_B_NAME = "disloc_altB_20260227_x80"


def _patch_dislocation_active_names(base_text: str, names: list[str]) -> str:
    names_toml = ", ".join(f'"{x}"' for x in names)
    section_start = base_text.index("[strategy.dislocation]\n")
    selector_start = base_text.index("[strategy.dislocation.selector]\n")
    before = base_text[:section_start]
    after = base_text[selector_start:]
    dislocation_section = (
        "[strategy.dislocation]\n"
        f"active_candidate_names = [{names_toml}]\n\n"
    )
    return before + dislocation_section + after


class LoadConfigActiveCandidateTests(unittest.TestCase):
    def test_active_candidate_names_filters_and_preserves_order(self) -> None:
        base_text = Path("config.toml").read_text(encoding="utf-8")
        patched = _patch_dislocation_active_names(base_text, [_ALT_B_NAME, _ALT_A_NAME])

        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "config_active_candidates.toml"
            cfg_path.write_text(patched, encoding="utf-8")
            cfg = load_app_config(str(cfg_path))

        names = [str(c.name) for c in cfg.strategy.dislocation.candidates]
        self.assertEqual([_ALT_B_NAME, _ALT_A_NAME], names)

    def test_active_candidate_names_missing_rejected(self) -> None:
        base_text = Path("config.toml").read_text(encoding="utf-8")
        patched = _patch_dislocation_active_names(
            base_text,
            [_ALT_A_NAME, "disloc_candidate_that_does_not_exist"],
        )

        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "config_active_missing.toml"
            cfg_path.write_text(patched, encoding="utf-8")
            with self.assertRaises(InvariantError):
                load_app_config(str(cfg_path))

    def test_active_candidate_names_duplicate_rejected(self) -> None:
        base_text = Path("config.toml").read_text(encoding="utf-8")
        patched = _patch_dislocation_active_names(base_text, [_ALT_A_NAME, _ALT_A_NAME])

        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "config_active_dupe.toml"
            cfg_path.write_text(patched, encoding="utf-8")
            with self.assertRaises(InvariantError):
                load_app_config(str(cfg_path))


if __name__ == "__main__":
    unittest.main()
