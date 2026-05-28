"""Reconciliation: why does canonical extension PnL differ between 2026-05-08
sweep (-14.97 BNB / 500 bets at 50 BNB scale) and Step 5 (-2.36 BNB / 108 bets
at 5 BNB scale)?

Runs canonical (3, 7, 15) cs=2 on the extension cohort (epochs 422298..437561)
with both bankroll scales using current code, to isolate whether the
discrepancy is purely a bankroll-scale artifact or whether code/data has
drifted between 2026-05-08 and now.
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

EXT_DIR = Path(r"C:\Users\zking\AppData\Local\Temp\ext\extended")
import research.in_process_runner as ipr  # noqa: E402
ipr._EXT_CLOSED_ROUNDS_PATH = EXT_DIR / "closed_rounds.jsonl"
ipr._EXT_BTC_KLINES_PATH = EXT_DIR / "btc_spot_prices.jsonl"
ipr._EXT_ETH_KLINES_PATH = EXT_DIR / "eth_spot_prices.jsonl"
ipr._EXT_SOL_KLINES_PATH = EXT_DIR / "sol_spot_prices.jsonl"


def main() -> None:
    out_root = Path(tempfile.mkdtemp(prefix="step5_recon_"))
    print(f"output: {out_root}\n")

    overrides = {"gate": {"mtf_lookbacks": [3, 7, 15]}}

    for bankroll, label in [(50.0, "50bnb_scale"), (5.0, "5bnb_scale")]:
        spec = ipr.FoldSpec(
            name=f"extension_{label}",
            kline_cutoff_seconds=2,
            epoch_start=422298,
            epoch_end=437561,
            strategy_overrides=overrides,
        )
        print(f"=== Run: initial_bankroll={bankroll} BNB ===")
        t0 = time.time()
        summaries = ipr.run_experiment(
            experiment_specs=[spec],
            output_base_dir=out_root,
            initial_bankroll_bnb=bankroll,
            treasury_fee_fraction=0.03,
            min_bet_amount_bnb=0.001,
            use_extended_data=True,
        )
        s = summaries[0]["summary"]
        print(f"  elapsed: {time.time()-t0:.1f}s")
        print(f"  bets={s['num_bets']} wins={s['num_wins']} WR={s['win_rate']:.4f}")
        print(f"  PnL={s['net_pnl_bnb']:+.4f} BNB  final={s['final_bankroll_bnb']:.4f}")
        print(f"  gross_profit={s['gross_profit_bnb']:+.4f}  gross_loss={s['gross_loss_bnb']:+.4f}")
        skips = s.get("skip_counts_by_reason", {})
        relevant = {k: v for k, v in skips.items()
                    if k in ("risk_cooldown_active", "risk_drawdown_breaker_fired",
                             "risk_bankroll_below_min", "gate_no_signal",
                             "pool_below_minimum", "payout_below_floor",
                             "bet_size_below_min")}
        print(f"  key skips: {json.dumps(relevant, indent=4)}")
        print()


if __name__ == "__main__":
    main()
