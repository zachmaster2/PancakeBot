"""The backfilled 2026-08-23 fingerprint must make 08-30 legible.

The point of the seed is that the CONFIG CHANGED banner fires on
2026-08-30 — the first Sunday whose fire stream runs under the 1.25 pool
filter while every prior week ran under 1.5. These tests drive the
monitor's own comparison rather than restating it.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import seed_monitor_fingerprint as seed  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "wm_for_seed_test", REPO / "research" / "weekly_monitor_state_machine.py")
wm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wm)


def _state(tmp_path, **extra):
    p = tmp_path / "state.json"
    p.write_text(json.dumps(dict(
        consecutive_weak=0, last_week="2026-08-23", last_action="none",
        history=[], **extra)), encoding="utf-8")
    return p


# ---- the seed does what it claims ---------------------------------------

def test_seeding_writes_the_fingerprint_and_its_provenance(tmp_path):
    p = _state(tmp_path)
    changed, msg = seed.seed(p, now="2026-08-24T15:00:00Z")
    assert changed, msg
    st = json.loads(p.read_text(encoding="utf-8"))
    assert st["strategy_fingerprint"] == seed.FINGERPRINT_2026_08_23
    prov = st["strategy_fingerprint_backfill"]
    assert prov["backfilled"] is True
    assert prov["derived_from_commit"] == "d750771"
    assert prov["backfilled_at_utc"] == "2026-08-24T15:00:00Z"
    assert "risk_off_config.toml" in prov["derived_from_artifact"]


def test_provenance_is_a_sibling_not_part_of_the_fingerprint(tmp_path):
    """Load-bearing. The monitor compares fingerprints with plain dict
    inequality, so a provenance key INSIDE the fingerprint would make
    config_changed true every week, forever."""
    p = _state(tmp_path)
    seed.seed(p)
    st = json.loads(p.read_text(encoding="utf-8"))
    assert "backfilled" not in st["strategy_fingerprint"]
    assert set(st["strategy_fingerprint"]) == set(seed.FINGERPRINT_2026_08_23)


def test_seeding_is_idempotent_and_refuses_to_overwrite(tmp_path):
    p = _state(tmp_path)
    assert seed.seed(p)[0] is True
    changed, msg = seed.seed(p)
    assert changed is False
    assert "refusing to overwrite" in msg


def test_seeding_refuses_once_a_later_run_has_happened(tmp_path):
    """If a monitor run has booked a newer week, seeding would misdescribe
    the comparison — re-derive instead of forcing."""
    p = _state(tmp_path, )
    st = json.loads(p.read_text(encoding="utf-8"))
    st["last_week"] = "2026-08-30"
    p.write_text(json.dumps(st), encoding="utf-8")
    changed, msg = seed.seed(p)
    assert changed is False
    assert "2026-08-23" in msg


def test_missing_state_file_is_reported_not_created(tmp_path):
    changed, msg = seed.seed(tmp_path / "nope.json")
    assert changed is False and "not found" in msg
    assert not (tmp_path / "nope.json").exists()


# ---- it makes 08-30 fire, which is the whole point ----------------------

def test_the_seeded_fingerprint_differs_from_today_in_exactly_three_keys():
    """The three real changes: the pool filter and both stake caps.
    Everything else must match, or the banner would name spurious keys."""
    current = wm.strategy_fingerprint()
    seeded = seed.FINGERPRINT_2026_08_23
    assert set(seeded) == set(current), "key sets must match to diff cleanly"
    changed = {k for k in current if current[k] != seeded[k]}
    assert changed == {
        "min_pool_bnb_at_cutoff",
        "max_bet_bnb_btc_primary",
        "max_bet_bnb_eth_sol_fallback",
    }, changed


def test_the_seed_makes_config_changed_true_against_today():
    """Drives the monitor's own comparison expression."""
    current = wm.strategy_fingerprint()
    prev = seed.FINGERPRINT_2026_08_23
    config_changed = prev is not None and prev != current
    assert config_changed is True


def test_without_the_seed_the_comparison_is_a_no_op():
    """The failure this seed exists to prevent: absent prior fingerprint
    means config_changed is False and 08-30 reads as an ordinary week."""
    prev = None
    current = wm.strategy_fingerprint()
    assert (prev is not None and prev != current) is False


@pytest.mark.parametrize("key,expected", [
    ("min_pool_bnb_at_cutoff", 1.5),
    ("min_payout_multiple_at_cutoff", 1.5),
    ("max_bet_bnb_btc_primary", 0.1),
    ("max_bet_bnb_eth_sol_fallback", 0.1),
    ("max_bet_fraction_of_bankroll", 0.05),
    ("min_bet_threshold_bnb", 0.01),
    ("kline_cutoff_seconds", 2),
    ("pool_cutoff_seconds", 6),
    ("treasury_fee_fraction", 0.03),
    ("mtf_lookbacks_used_for_slicing", [3, 7, 15]),
    ("mtf_lookbacks_deployed", [3, 7, 15]),
])
def test_each_seeded_value_is_the_one_the_records_show(key, expected):
    """Pinned individually so a future edit to the table has to be
    deliberate. Sources: the 08-23 run's own risk_off_config.toml copy and
    the repo at d750771 — see the script docstring."""
    assert seed.FINGERPRINT_2026_08_23[key] == expected
