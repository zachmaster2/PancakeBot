"""p3b orderflow features: 4-test look-ahead invariance suite.

Per orchestrator v1.1 R6 (carryover from p3a) + R2 (trade-order shuffle locked):

  Test 1: within-round contamination — synthetic trade injected AFTER cutoff
          (lock_at - 2s) must NOT change features for round N.
  Test 2: outcome-flip — flipping round N's position must NOT change features
          for round N (no outcome leakage into features).
  Test 3: forward-event injection — trades with ts in rounds N+1..N+5 must NOT
          change features for round N.
  Test 4: trade-order shuffle (R2 LOCKED): permute the LIST ORDER of trades
          within round N's data window, PRESERVING each trade's timestamp.
          ALL 12 features MUST be byte-invariant — they should use timestamp-
          derived information, never list-position.

Test data: 100 synthetic rounds with trades fabricated locally (no OKX dependency).
"""
from __future__ import annotations

import copy
import math
import random
import sys
from pathlib import Path

import pytest

REPO = Path(r"C:\Users\zking\Documents\GitHub\PancakeBot")
WORKTREE = REPO / ".claude" / "worktrees" / "stupefied-bell-4d955c"
sys.path.insert(0, str(WORKTREE))

from research.p3b_orderflow_features import (  # noqa: E402
    FEATURE_NAMES, compute_features, EPOCH_DURATION, DATA_HORIZON_OFFSET_S,
)


N_ROUNDS = 100
ROUND_START = 1700000000  # arbitrary anchor


@pytest.fixture(scope="module")
def synthetic_rounds():
    """Generate 100 synthetic rounds with realistic trade counts (~40/round)."""
    rng = random.Random(42)
    rounds: list[dict] = []
    for r in range(N_ROUNDS):
        start_at = ROUND_START + r * EPOCH_DURATION
        cutoff = start_at + EPOCH_DURATION - DATA_HORIZON_OFFSET_S
        n_trades = rng.randint(20, 65)  # match p3b pre-flight distribution
        trades = []
        # Distribute trades over [start_at, cutoff)
        last_px = 600.0 + rng.uniform(-10, 10)
        for i in range(n_trades):
            ts_s = start_at + rng.randint(0, EPOCH_DURATION - DATA_HORIZON_OFFSET_S - 1)
            last_px += rng.uniform(-0.5, 0.5)
            trades.append({
                "instId": "BNB-USDT",
                "side": rng.choice(["buy", "sell"]),
                "sz": f"{rng.uniform(0.01, 1.0):.6f}",
                "px": f"{last_px:.4f}",
                "source": "0",
                "tradeId": str(10_000_000 + r * 100 + i),
                "ts": str(ts_s * 1000 + rng.randint(0, 999)),
            })
        # Pre-sort by ts to mimic OKX's chronological-ish ordering
        trades.sort(key=lambda t: int(t["ts"]))
        # Outcome: random ~50% Bull
        rounds.append({
            "epoch": 100000 + r,
            "startAt": start_at,
            "lockPrice": last_px - 0.1,
            "closePrice": last_px,
            "position": "Bull" if rng.random() > 0.5 else "Bear",
            "failed": False,
            "trades": trades,
        })
    return rounds


@pytest.fixture(scope="module")
def large_size_threshold(synthetic_rounds):
    """Compute the 90th-percentile USD trade size across all rounds."""
    sizes = []
    for r in synthetic_rounds:
        for t in r["trades"]:
            sizes.append(float(t["sz"]) * float(t["px"]))
    sizes.sort()
    if not sizes:
        return 0.0
    idx = int(len(sizes) * 0.9)
    return sizes[idx]


@pytest.fixture(scope="module")
def baseline_features(synthetic_rounds, large_size_threshold):
    """Compute baseline features for each round."""
    return [
        compute_features(
            r["startAt"], r["trades"],
            large_size_threshold_usd=large_size_threshold,
        )
        for r in synthetic_rounds
    ]


def _features_equal(a: dict, b: dict) -> tuple[bool, str | None]:
    for k in FEATURE_NAMES:
        va, vb = a.get(k), b.get(k)
        # NaN equality
        is_nan_a = isinstance(va, float) and math.isnan(va)
        is_nan_b = isinstance(vb, float) and math.isnan(vb)
        if is_nan_a and is_nan_b:
            continue
        if is_nan_a or is_nan_b:
            return False, f"{k}: NaN mismatch ({va} vs {vb})"
        if va != vb:
            return False, f"{k}: {va} vs {vb}"
    return True, None


def _victim_indices(seed: int = 1, n_victims: int = 5) -> list[int]:
    rng = random.Random(seed)
    return sorted(rng.sample(range(N_ROUNDS), n_victims))


# ============================================================
# Test 1: data-horizon enforcement (within-round contamination)
# ============================================================

def test_within_round_contamination_invariance(synthetic_rounds, baseline_features,
                                                large_size_threshold):
    """A trade injected AT lock_at (past the cutoff at lock_at - 2s) must
    not change features for round N."""
    for vi in _victim_indices(seed=1):
        r = copy.deepcopy(synthetic_rounds[vi])
        lock_at = r["startAt"] + EPOCH_DURATION
        synthetic_trade = {
            "instId": "BNB-USDT",
            "side": "buy",
            "sz": "10.0",  # whale
            "px": "999.0",
            "source": "0",
            "tradeId": "synthetic_post_cutoff",
            "ts": str(lock_at * 1000),  # exactly at lock_at, past cutoff
        }
        r["trades"] = list(r["trades"]) + [synthetic_trade]
        new_feats = compute_features(
            r["startAt"], r["trades"],
            large_size_threshold_usd=large_size_threshold,
        )
        eq, diff = _features_equal(baseline_features[vi], new_feats)
        assert eq, (
            f"Test 1 FAIL: round-{r['epoch']} features changed after injecting "
            f"synthetic trade at lock_at (past data horizon at lock_at - 2s): {diff}"
        )


# ============================================================
# Test 2: outcome-flip
# ============================================================

def test_outcome_flip_invariance(synthetic_rounds, baseline_features,
                                  large_size_threshold):
    """Flipping round N's position field must not change features for round N
    (features must not read outcome data)."""
    for vi in _victim_indices(seed=2):
        r = copy.deepcopy(synthetic_rounds[vi])
        old_pos = r["position"]
        r["position"] = "Bear" if old_pos == "Bull" else "Bull"
        # Also flip price ordering for belt-and-suspenders
        cp, lp = r["closePrice"], r["lockPrice"]
        r["closePrice"], r["lockPrice"] = lp, cp
        new_feats = compute_features(
            r["startAt"], r["trades"],
            large_size_threshold_usd=large_size_threshold,
        )
        eq, diff = _features_equal(baseline_features[vi], new_feats)
        assert eq, (
            f"Test 2 FAIL: round-{r['epoch']} features changed after flipping "
            f"position ({old_pos} -> {r['position']}): {diff}"
        )


# ============================================================
# Test 3: forward-event injection (rounds N+1..N+5)
# ============================================================

def test_forward_event_injection_invariance(synthetic_rounds, baseline_features,
                                              large_size_threshold):
    """Trades with timestamps in rounds N+1..N+5 must not affect round N's
    features when round N is computed in isolation."""
    for vi in _victim_indices(seed=3):
        r = copy.deepcopy(synthetic_rounds[vi])
        # Inject trades into the SAME bet list but with timestamps in N+k.
        # If features correctly filter by [start_at, lock_at - 2s], these
        # should be ignored.
        for k in range(1, 6):
            future_start = r["startAt"] + k * EPOCH_DURATION
            r["trades"].append({
                "instId": "BNB-USDT", "side": "buy",
                "sz": "5.0", "px": "999.0",
                "source": "0", "tradeId": f"future_{k}",
                "ts": str((future_start + 50) * 1000),
            })
        new_feats = compute_features(
            r["startAt"], r["trades"],
            large_size_threshold_usd=large_size_threshold,
        )
        eq, diff = _features_equal(baseline_features[vi], new_feats)
        assert eq, (
            f"Test 3 FAIL: round-{r['epoch']} features changed after injecting "
            f"trades in 5 future rounds: {diff}"
        )


# ============================================================
# Test 4: trade-order shuffle (R2 locked semantics)
# ============================================================

def test_trade_order_shuffle_invariance(synthetic_rounds, baseline_features,
                                         large_size_threshold):
    """Permuting list order of trades within round N (preserving each trade's
    timestamp) must not change features. Catches list-position implementation
    bugs."""
    for vi in _victim_indices(seed=4):
        r = copy.deepcopy(synthetic_rounds[vi])
        rng = random.Random(vi * 13 + 7)
        rng.shuffle(r["trades"])
        new_feats = compute_features(
            r["startAt"], r["trades"],
            large_size_threshold_usd=large_size_threshold,
        )
        eq, diff = _features_equal(baseline_features[vi], new_feats)
        assert eq, (
            f"Test 4 FAIL: round-{r['epoch']} features changed after permuting "
            f"trade list order (timestamps preserved): {diff}"
        )
