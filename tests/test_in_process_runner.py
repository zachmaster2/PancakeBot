"""Regression test: in-process backtest driver reproduces the canonical
cutoff=2 + lookbacks=(3,7,15) baseline bit-identically.

The known baseline (per ``project_holdout_slice.md`` /
``project_experiment_rules.md``) was committed by every prior backtest
generation and is the ground truth that every refactor must preserve:

  - 5-fold CV total: 1446 bets / 884 wins / 0.6113 WR / +50.4953 BNB
  - Holdout: 9 bets / 6 wins / 0.6667 WR / +0.2282 BNB
  - 5-fold aggregated md5 hash: aa39a3a73f4e4cb718beeffaa72a22ca
  - Per-fold PnL: f1=+4.2602, f2=+7.3128, f3=+20.2876, f4=+17.0644, f5=+1.5703

This test rebuilds those numbers exactly via
``research.in_process_runner.run_experiment(...)`` against the
post-rebuild 300-candle dataset (``var/{btc,eth,sol,bnb}_spot_prices.jsonl``).
Any code change that breaks bit-identity here is either a bug or a
behavior change that needs explicit acknowledgement.

Originally this lived as ``research/cutoff_experiment.py`` and was
phrased as one-time refactor verification. Promoted to a permanent
pytest in 2026-04-26 lean&clean to catch future regressions.

Run::
    python -m pytest tests/test_in_process_runner.py -v
    python tests/test_in_process_runner.py        # standalone CLI
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from research.in_process_runner import FoldSpec, run_experiment  # noqa: E402


# ---------- canonical baseline ----------

_FOLDS = [
    {"name": "f1", "epoch_start": 437562, "epoch_end": 444866},
    {"name": "f2", "epoch_start": 444867, "epoch_end": 452171},
    {"name": "f3", "epoch_start": 452172, "epoch_end": 459476},
    {"name": "f4", "epoch_start": 459477, "epoch_end": 466781},
    {"name": "f5", "epoch_start": 466782, "epoch_end": 474086},
]
_HOLDOUT = {"name": "holdout", "epoch_start": 474880, "epoch_end": 475311}

_EXPECTED_PER_FOLD_PNL = {
    "f1": 4.2602,
    "f2": 7.3128,
    "f3": 20.2876,
    "f4": 17.0644,
    "f5": 1.5703,
}
_EXPECTED_PER_FOLD_BETS = {
    "f1": 129, "f2": 196, "f3": 473, "f4": 411, "f5": 237,
}
_EXPECTED_PER_FOLD_WINS = {
    "f1": 85, "f2": 120, "f3": 291, "f4": 251, "f5": 137,
}
_EXPECTED_5FOLD_TOTAL_PNL = 50.4953
_EXPECTED_HOLDOUT_PNL = 0.2282
_EXPECTED_HOLDOUT_BETS = 9
_EXPECTED_HOLDOUT_WINS = 6
_EXPECTED_5FOLD_HASH = "9eec23adceca7fbbe44cfae5245dfc83"

_RUN_LABEL = "in_process_baseline_verification"


def _content_hash(summary: dict) -> str:
    """Hash summary dict content excluding the elapsed_sim_seconds field."""
    obj = dict(summary)
    obj.pop("elapsed_sim_seconds", None)
    return hashlib.md5(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def _build_spec(output_base_dir: Path) -> list[FoldSpec]:
    """Build the canonical 6-fold cutoff=2 + (3,7,15)-defaults spec."""
    out: list[FoldSpec] = []
    for fold in _FOLDS + [_HOLDOUT]:
        out.append(FoldSpec(
            name=f"{_RUN_LABEL}/{fold['name']}",
            cutoff_seconds=2,
            epoch_start=fold["epoch_start"],
            epoch_end=fold["epoch_end"],
            strategy_overrides={},  # all defaults — canonical baseline
            plot=False,
        ))
    return out


def _data_files_present() -> bool:
    """Return True iff the required dataset is on disk."""
    needed = [
        _REPO_ROOT / "var" / "closed_rounds.jsonl",
        _REPO_ROOT / "var" / "btc_spot_prices.jsonl",
        _REPO_ROOT / "var" / "eth_spot_prices.jsonl",
        _REPO_ROOT / "var" / "sol_spot_prices.jsonl",
    ]
    return all(p.exists() for p in needed)


@pytest.mark.skipif(
    not _data_files_present(),
    reason="canonical dataset not on disk (var/closed_rounds.jsonl + 3 kline files)",
)
def test_canonical_baseline_bit_identical(tmp_path: Path) -> None:
    """Run the canonical 6-fold spec and assert every metric matches baseline.

    Uses a ``tmp_path`` for output so the test never pollutes
    ``var/sweep/`` and consecutive runs are isolated.
    """
    specs = _build_spec(tmp_path)
    summaries = run_experiment(
        experiment_specs=specs,
        output_base_dir=tmp_path,
    )

    # Map summaries by fold name suffix for cleaner asserts.
    by_name: dict[str, dict] = {}
    for entry in summaries:
        suffix = entry["spec_name"].split("/")[-1]
        by_name[suffix] = entry["summary"]

    # Per-fold (5-fold CV) bit-identity.
    fold_hashes: list[str] = []
    total_pnl_5fold = 0.0
    for name in ("f1", "f2", "f3", "f4", "f5"):
        s = by_name[name]
        fold_hashes.append(_content_hash(s))
        assert s["num_bets"] == _EXPECTED_PER_FOLD_BETS[name], (
            f"{name} bets: got {s['num_bets']}, expected {_EXPECTED_PER_FOLD_BETS[name]}"
        )
        assert s["num_wins"] == _EXPECTED_PER_FOLD_WINS[name], (
            f"{name} wins: got {s['num_wins']}, expected {_EXPECTED_PER_FOLD_WINS[name]}"
        )
        pnl = float(s["net_pnl_bnb"])
        total_pnl_5fold += pnl
        assert abs(pnl - _EXPECTED_PER_FOLD_PNL[name]) < 1e-3, (
            f"{name} PnL: got {pnl:+.4f}, expected {_EXPECTED_PER_FOLD_PNL[name]:+.4f}"
        )

    # 5-fold total + aggregated hash.
    assert abs(total_pnl_5fold - _EXPECTED_5FOLD_TOTAL_PNL) < 1e-3, (
        f"5-fold PnL: got {total_pnl_5fold:+.4f}, "
        f"expected {_EXPECTED_5FOLD_TOTAL_PNL:+.4f}"
    )
    agg_hash = hashlib.md5(",".join(fold_hashes).encode()).hexdigest()
    assert agg_hash == _EXPECTED_5FOLD_HASH, (
        f"5-fold aggregated hash: got {agg_hash}, expected {_EXPECTED_5FOLD_HASH}"
    )

    # Holdout slice.
    h = by_name["holdout"]
    assert h["num_bets"] == _EXPECTED_HOLDOUT_BETS, (
        f"holdout bets: got {h['num_bets']}, expected {_EXPECTED_HOLDOUT_BETS}"
    )
    assert h["num_wins"] == _EXPECTED_HOLDOUT_WINS, (
        f"holdout wins: got {h['num_wins']}, expected {_EXPECTED_HOLDOUT_WINS}"
    )
    assert abs(float(h["net_pnl_bnb"]) - _EXPECTED_HOLDOUT_PNL) < 1e-3, (
        f"holdout PnL: got {h['net_pnl_bnb']:+.4f}, "
        f"expected {_EXPECTED_HOLDOUT_PNL:+.4f}"
    )


# ---------- standalone CLI (preserves the previous research/cutoff_experiment.py UX) ----------

def main() -> int:
    """Standalone runner with human-readable table output.

    Prefer ``python -m pytest tests/test_in_process_runner.py -v`` for
    CI-style invocation. This entry point exists for ad-hoc invocation
    when you want the per-fold output table without pytest's harness.
    """
    print("=" * 78)
    print("REGRESSION TEST  cutoff=2 + (3,7,15) bit-identity check")
    print("=" * 78)

    if not _data_files_present():
        print("[SKIP] canonical dataset not on disk")
        return 0

    with tempfile.TemporaryDirectory(prefix="in_process_runner_test_") as tmp:
        out_dir = Path(tmp)
        specs = _build_spec(out_dir)
        summaries = run_experiment(
            experiment_specs=specs,
            output_base_dir=out_dir,
        )

    by_name: dict[str, dict] = {
        entry["spec_name"].split("/")[-1]: entry["summary"]
        for entry in summaries
    }

    print(f"\n{'fold':<10} {'bets':>6} {'wins':>6} {'wr':>8} "
          f"{'pnl_bnb':>12} {'expected':>12} {'delta':>10}")
    print("-" * 78)
    fold_hashes: list[str] = []
    total_pnl_5fold = 0.0
    all_match = True
    for name in ("f1", "f2", "f3", "f4", "f5"):
        s = by_name[name]
        fold_hashes.append(_content_hash(s))
        pnl = float(s["net_pnl_bnb"])
        total_pnl_5fold += pnl
        expected = _EXPECTED_PER_FOLD_PNL[name]
        delta = pnl - expected
        match = abs(delta) < 1e-3
        if not match:
            all_match = False
        print(f"{name:<10} {s['num_bets']:>6} {s['num_wins']:>6} "
              f"{s['win_rate']:>8.4f} {pnl:>+12.4f} "
              f"{expected:>+12.4f} {delta:>+10.4f}"
              f"{'' if match else '  MISMATCH'}")

    h = by_name["holdout"]
    h_pnl = float(h["net_pnl_bnb"])
    h_delta = h_pnl - _EXPECTED_HOLDOUT_PNL
    h_match = abs(h_delta) < 1e-3
    if not h_match:
        all_match = False
    print(f"{'holdout':<10} {h['num_bets']:>6} {h['num_wins']:>6} "
          f"{h['win_rate']:>8.4f} {h_pnl:>+12.4f} "
          f"{_EXPECTED_HOLDOUT_PNL:>+12.4f} {h_delta:>+10.4f}"
          f"{'' if h_match else '  MISMATCH'}")
    print("-" * 78)
    total_delta = total_pnl_5fold - _EXPECTED_5FOLD_TOTAL_PNL
    total_match = abs(total_delta) < 1e-3
    if not total_match:
        all_match = False
    print(f"{'5-fold':<10} {'':>6} {'':>6} {'':>8} {total_pnl_5fold:>+12.4f} "
          f"{_EXPECTED_5FOLD_TOTAL_PNL:>+12.4f} {total_delta:>+10.4f}"
          f"{'' if total_match else '  MISMATCH'}")

    agg_hash = hashlib.md5(",".join(fold_hashes).encode()).hexdigest()
    hash_match = (agg_hash == _EXPECTED_5FOLD_HASH)
    print(f"\n5-fold aggregated hash: {agg_hash}")
    print(f"Expected:                {_EXPECTED_5FOLD_HASH}")
    if not hash_match:
        all_match = False

    print("\n" + "=" * 78)
    if all_match:
        print("[OK] BIT-IDENTICAL  refactor preserved canonical baseline.")
        return 0
    print("[FAIL] DEVIATION DETECTED  refactor introduced a behavior change.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
