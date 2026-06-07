"""Zero-confirmation candidate gauntlet (read-only research).

Compares the CANDIDATE (require_cross_asset_confirmation=True: skip BTC-primary
bets with zero ETH/SOL confirmation) against the CANONICAL baseline across
CV5 (f1-f5), the frozen holdout, and extension_v2 — re-running the full
in-process decision path for both arms so the bankroll ripple is captured
(NOT a naive subtraction of the removed bets' PnL).

The only difference between arms is the strategy override
{"btc_primary": {"require_cross_asset_confirmation": True}}.

The canonical arm doubles as a validity gate: CV5 must reproduce the hash-locked
baseline (1446 bets / 884 wins / +50.4953 BNB).

Run:  cd <repo> && .venv/Scripts/python.exe research/zero_confirmation_gauntlet_2026_06_06.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.in_process_runner import FoldSpec, run_experiment  # noqa: E402

SECTIONS = [
    ("f1", 437562, 444866),
    ("f2", 444867, 452171),
    ("f3", 452172, 459476),
    ("f4", 459477, 466781),
    ("f5", 466782, 474086),
    ("holdout", 474880, 475311),
    ("extension_v2", 475312, 479952),
]
CAND_OVERRIDE = {"btc_primary": {"require_cross_asset_confirmation": True}}
# Canonical validity-gate expectations (hash-locked baseline).
CANON_CV5 = {"bets": 1446, "wins": 884, "pnl": 50.4953}


def _specs() -> list[FoldSpec]:
    specs: list[FoldSpec] = []
    for name, lo, hi in SECTIONS:
        specs.append(FoldSpec(
            name=f"canon/{name}", kline_cutoff_seconds=2, epoch_start=lo,
            epoch_end=hi, strategy_overrides={}, plot=False,
        ))
        specs.append(FoldSpec(
            name=f"cand/{name}", kline_cutoff_seconds=2, epoch_start=lo,
            epoch_end=hi, strategy_overrides=CAND_OVERRIDE, plot=False,
        ))
    return specs


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="zeroconf_gauntlet_") as tmp:
        summaries = run_experiment(
            experiment_specs=_specs(), output_base_dir=Path(tmp),
        )
    by: dict[str, dict] = {e["spec_name"]: e["summary"] for e in summaries}

    def row(name: str):
        c = by.get(f"canon/{name}")
        d = by.get(f"cand/{name}")
        if c is None or d is None:
            return None
        return (
            int(c["num_bets"]), int(c["num_wins"]), float(c["net_pnl_bnb"]),
            int(d["num_bets"]), int(d["num_wins"]), float(d["net_pnl_bnb"]),
        )

    print("=" * 96)
    print("ZERO-CONFIRMATION GAUNTLET  candidate(require_cross_asset_confirmation=True) vs canonical")
    print("=" * 96)
    hdr = f"{'section':<14}{'canon bets/wins/pnl':>28}{'cand bets/wins/pnl':>28}{'dPnL':>10}{'dBets':>8}"
    print(hdr)
    print("-" * 96)

    cv5 = {"cb": 0, "cw": 0, "cp": 0.0, "db": 0, "dw": 0, "dp": 0.0}
    for name, _, _ in SECTIONS:
        r = row(name)
        if r is None:
            print(f"{name:<14}{'(no data / fold empty)':>28}")
            continue
        cb, cw, cp, db, dw, dp = r
        dpnl = dp - cp
        dbets = db - cb
        canon_s = f"{cb}/{cw}/{cp:+.4f}"
        cand_s = f"{db}/{dw}/{dp:+.4f}"
        print(f"{name:<14}{canon_s:>28}{cand_s:>28}{dpnl:>+10.4f}{dbets:>8}")
        if name.startswith("f"):
            cv5["cb"] += cb; cv5["cw"] += cw; cv5["cp"] += cp
            cv5["db"] += db; cv5["dw"] += dw; cv5["dp"] += dp

    print("-" * 96)
    cv5_lift = cv5["dp"] - cv5["cp"]
    cv5_canon_s = f"{cv5['cb']}/{cv5['cw']}/{cv5['cp']:+.4f}"
    cv5_cand_s = f"{cv5['db']}/{cv5['dw']}/{cv5['dp']:+.4f}"
    label = "CV5 (f1-f5)"
    print(f"{label:<14}{cv5_canon_s:>28}{cv5_cand_s:>28}{cv5_lift:>+10.4f}{cv5['db'] - cv5['cb']:>8}")
    print("=" * 96)

    # Validity gate on the canonical arm.
    ok = (cv5["cb"] == CANON_CV5["bets"] and cv5["cw"] == CANON_CV5["wins"]
          and abs(cv5["cp"] - CANON_CV5["pnl"]) < 1e-3)
    print(f"VALIDITY GATE (canon CV5 == {CANON_CV5['bets']}/{CANON_CV5['wins']}/"
          f"+{CANON_CV5['pnl']}): {'PASS' if ok else 'FAIL'}")
    if not ok:
        print(f"  got canon CV5 {cv5['cb']}/{cv5['cw']}/{cv5['cp']:+.4f} — ABORT, baseline mismatch")
        return 1
    print(f"CANDIDATE CV5 LIFT: {cv5_lift:+.4f} BNB  ({cv5['db'] - cv5['cb']} fewer bets)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
