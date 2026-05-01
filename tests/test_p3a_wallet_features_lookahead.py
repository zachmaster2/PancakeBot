"""p3a wallet-aware features: 4-test look-ahead invariance suite.

Per orchestrator v2.1 R10: ALL FOUR tests must pass before main feature
compute runs.

  Test 1: within-round contamination — synthetic bet for wallet W in
          round N must not change features for round N.
  Test 2: outcome-flip — flipping round N's position (Bull <-> Bear)
          must not change features for round N.
  Test 3: forward-event injection — adding bets in rounds N+1..N+5 must
          not change features for round N.
  Test 4: round-order shuffle — permuting prior rounds (R<N) order must
          not change history-derived features for round N.

Slice: 100 rounds from canonical CV5 (epochs 437562..437661 — start of f1).
"""
from __future__ import annotations

import copy
import json
import random
import sys
from pathlib import Path

import pytest

REPO = Path(r"C:\Users\zking\Documents\GitHub\PancakeBot")
WORKTREE = REPO / ".claude" / "worktrees" / "stupefied-bell-4d955c"
sys.path.insert(0, str(WORKTREE))

from research.p3a_wallet_features import (  # noqa: E402
    FEATURE_NAMES, build_features_chronological, compute_features,
    WalletHistoryStore,
)


SLICE_LO = 437562
SLICE_HI = 437661  # 100 rounds


@pytest.fixture(scope="module")
def slice_rounds():
    """Load 100 canonical-data rounds with full bet event records."""
    out = []
    with open(REPO / "var" / "closed_rounds.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ep = int(rec["epoch"])
            if not (SLICE_LO <= ep <= SLICE_HI):
                continue
            out.append(rec)
    out.sort(key=lambda r: int(r["epoch"]))
    assert len(out) >= 80, f"expected >=80 rounds in slice, got {len(out)}"
    return out


@pytest.fixture(scope="module")
def baseline(slice_rounds):
    """Compute baseline features for the entire slice (chronological pass)."""
    feats, outs = build_features_chronological(slice_rounds)
    return feats, outs


def _features_equal(a: dict, b: dict) -> tuple[bool, str | None]:
    """Compare two feature dicts. NaN == NaN returns True. Returns
    (equal, first_difference_description_or_None)."""
    import math
    for k in FEATURE_NAMES:
        va = a.get(k)
        vb = b.get(k)
        if va is None and vb is None:
            continue
        if va is None or vb is None:
            return False, f"{k}: one is None ({va} vs {vb})"
        # NaN equality
        if isinstance(va, float) and isinstance(vb, float):
            if math.isnan(va) and math.isnan(vb):
                continue
            if math.isnan(va) or math.isnan(vb):
                return False, f"{k}: NaN mismatch ({va} vs {vb})"
            if va != vb:
                return False, f"{k}: {va} vs {vb}"
        else:
            if va != vb:
                return False, f"{k}: {va} vs {vb}"
    return True, None


def _victim_round_indices(rounds_list: list, n_victims: int = 5, seed: int = 1) -> list[int]:
    """Pick n_victims indices among the middle of the slice (so they have
    history before AND data after — needed for forward-injection test)."""
    rng = random.Random(seed)
    # Avoid first 20 (no history) and last 10 (no future rounds for forward test)
    pool = list(range(20, len(rounds_list) - 10))
    return sorted(rng.sample(pool, n_victims))


# ============================================================
# Test 1: within-round contamination
# ============================================================

def test_within_round_contamination_invariance(slice_rounds, baseline):
    """Data-horizon enforcement: a bet placed AT OR AFTER the cutoff
    (lock_at - 2s) must not change ANY feature for round N. Tests that
    the feature compute correctly filters by `created_at < cutoff`.

    A bet placed BEFORE the cutoff legitimately changes within-round
    aggregations like pool_size_bnb — that's expected behavior, not
    look-ahead. So we inject AFTER the cutoff to test the filter."""
    base_feats, _ = baseline
    victim_idx = _victim_round_indices(slice_rounds, seed=1)

    for vi in victim_idx:
        # Mutate: add a synthetic bet to round N AT lock_at (right at the
        # boundary, past the cutoff at lock_at - 2). Should be filtered out
        # by `created_at < cutoff`.
        mutated = copy.deepcopy(slice_rounds)
        round_n = mutated[vi]
        lock_at = int(round_n["startAt"]) + 300
        synthetic_bet = {
            "wallet": "0xdead000000000000000000000000000000000001",
            "amountWei": 5_000_000_000_000_000_000,  # 5 BNB whale
            "position": "Bull",
            "createdAt": lock_at,  # exactly at lock — past cutoff (lock - 2)
        }
        round_n["bets"] = list(round_n.get("bets") or []) + [synthetic_bet]

        # Recompute the slice
        new_feats, _ = build_features_chronological(mutated)
        eq, diff = _features_equal(base_feats[vi], new_feats[vi])
        assert eq, (
            f"Test 1 FAIL: round-{slice_rounds[vi]['epoch']} features changed after "
            f"injecting synthetic bet at lock_at (past data horizon at lock_at-2s): {diff}"
        )


# ============================================================
# Test 2: outcome-flip
# ============================================================

def test_outcome_flip_invariance(slice_rounds, baseline):
    """Flipping round N's position from Bull <-> Bear must not change
    features for round N (no outcome leakage into features)."""
    base_feats, _ = baseline
    victim_idx = _victim_round_indices(slice_rounds, seed=2)

    for vi in victim_idx:
        mutated = copy.deepcopy(slice_rounds)
        old_pos = mutated[vi].get("position")
        if old_pos not in ("Bull", "Bear"):
            continue  # House / failed
        new_pos = "Bear" if old_pos == "Bull" else "Bull"
        mutated[vi]["position"] = new_pos
        # Also flip closePrice/lockPrice ordering to be consistent (defensive
        # — features shouldn't read price either, but be belt-and-suspenders)
        cp = mutated[vi].get("closePrice")
        lp = mutated[vi].get("lockPrice")
        if cp is not None and lp is not None:
            mutated[vi]["closePrice"] = lp
            mutated[vi]["lockPrice"] = cp

        new_feats, _ = build_features_chronological(mutated)
        eq, diff = _features_equal(base_feats[vi], new_feats[vi])
        assert eq, (
            f"Test 2 FAIL: round-{slice_rounds[vi]['epoch']} features changed after "
            f"flipping outcome ({old_pos}->{new_pos}): {diff}"
        )


# ============================================================
# Test 3: forward-event injection
# ============================================================

def test_forward_event_injection_invariance(slice_rounds, baseline):
    """Adding bets in rounds N+1..N+5 must not change features for round N
    (no forward-looking history)."""
    base_feats, _ = baseline
    victim_idx = _victim_round_indices(slice_rounds, seed=3)

    for vi in victim_idx:
        mutated = copy.deepcopy(slice_rounds)
        for k in range(1, 6):
            if vi + k >= len(mutated):
                break
            future_round = mutated[vi + k]
            synthetic_bet = {
                "wallet": "0xdead000000000000000000000000000000000002",
                "amountWei": 1_000_000_000_000_000_000,
                "position": "Bull",
                "createdAt": int(future_round["startAt"]) + 60,
            }
            future_round["bets"] = list(future_round.get("bets") or []) + [synthetic_bet]
        new_feats, _ = build_features_chronological(mutated)
        eq, diff = _features_equal(base_feats[vi], new_feats[vi])
        assert eq, (
            f"Test 3 FAIL: round-{slice_rounds[vi]['epoch']} features changed after "
            f"injecting bets in 5 future rounds: {diff}"
        )


# ============================================================
# Test 4: round-order shuffle (history aggregation is order-independent)
# ============================================================

def test_round_order_shuffle_invariance(slice_rounds, baseline):
    """Permuting prior rounds (R<N) order must not change history-derived
    features for round N. History should be set-aggregated, not order-
    dependent."""
    base_feats, _ = baseline
    victim_idx = _victim_round_indices(slice_rounds, seed=4)

    for vi in victim_idx:
        # Build a custom-ordered version: shuffle indices < vi, keep vi onwards in order.
        rng = random.Random(vi * 7 + 42)
        prior_indices = list(range(vi))
        rng.shuffle(prior_indices)
        shuffled_priors = [slice_rounds[i] for i in prior_indices]
        recombined = shuffled_priors + slice_rounds[vi:]

        # Recompute. Note: the shuffled priors will produce DIFFERENT pre-vi
        # features (because they're computed in a wrong order, where rounds
        # see "future" history they shouldn't). That's fine — we only check
        # that the round at the original-vi position (which is now at index
        # vi after the shuffle) gets the correct features.
        new_feats, _ = build_features_chronological(recombined)

        # The victim round is still at index vi in `recombined`.
        history_features = (
            "smart_bias", "fresh_ratio", "repeat_loser_ratio", "consensus_ratio",
        )
        for k in history_features:
            v_old = base_feats[vi].get(k)
            v_new = new_feats[vi].get(k)
            import math
            both_nan = (isinstance(v_old, float) and isinstance(v_new, float)
                        and math.isnan(v_old) and math.isnan(v_new))
            if both_nan:
                continue
            assert v_old == v_new, (
                f"Test 4 FAIL: round-{slice_rounds[vi]['epoch']} feature {k} changed "
                f"after shuffling prior rounds: {v_old} -> {v_new}"
            )
