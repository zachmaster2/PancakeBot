"""Permutation null for the zero-confirmation candidate (read-only research).

Tests whether the 'keeps' subset (BTC-primary bets with ZERO ETH/SOL
confirmation) is significantly worse than a random same-size subset of all CV5
bets — i.e. whether the lift from removing them is real or luck.

Statistic : net PnL of the keeps subset (canonical CV5 run, ALL bets placed).
Null      : net PnL of N random same-size subsets of the 1446 CV5 bets.
p (left)  : P(random-subset net <= keeps net). Small p => keeps are
            non-randomly bad => lift is real. Large p => consistent with luck.

Reuses the canonical decision + settlement path (same per-bet profit the
backtest produces). Validity gate: CV5 must reproduce 1446 bets / +50.4953 BNB.

Run:  cd <repo> && .venv/Scripts/python.exe research/zero_confirmation_permutation_2026_06_06.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pancakebot.bankroll_tracker import InMemoryBankrollTracker  # noqa: E402
from pancakebot.config import _DEFAULT_STRATEGY  # noqa: E402
from pancakebot.constants import MAX_GAS_COST_BET_BNB  # noqa: E402
from pancakebot.settlement import settle_bet_against_closed_round  # noqa: E402
from pancakebot.strategy.momentum_gate import MomentumGateConfig  # noqa: E402
import research.in_process_runner as ipr  # noqa: E402
from research.btc_only_degrade_edge import (  # noqa: E402
    _CapturePipeline,
    _classify_bet,
    _pool_total_for_round,
)

CV5 = [
    ("f1", 437562, 444866), ("f2", 444867, 452171), ("f3", 452172, 459476),
    ("f4", 459477, 466781), ("f5", 466782, 474086),
]
KCUT, PCUT, FEE, BANK = 2, 6, 0.03, 50.0
N_PERM = 1000
SEED = 20260606


def _collect_cv5_bets() -> list[tuple[float, bool]]:
    """Per-bet (profit, is_keeps) across CV5, canonical config, all bets placed."""
    strat = _DEFAULT_STRATEGY
    try:
        from pancakebot.market_data.contract_constants import load_contract_constants
        min_bet = float(load_contract_constants().min_bet_amount_bnb)
    except Exception:  # noqa: BLE001
        min_bet = 0.001

    all_rounds = ipr._load_all_rounds(use_extended_data=False)
    resolved = [(ipr.FoldSpec(name="x", kline_cutoff_seconds=KCUT,
                              epoch_start=None, epoch_end=None), strat)]
    eo, lo, _ = ipr._compute_load_extent(resolved)
    btc_u = ipr._load_klines_unified(ipr._BTC_KLINES_PATH, earliest_offset=eo, latest_offset=lo)
    eth_u = ipr._load_klines_unified(ipr._ETH_KLINES_PATH, earliest_offset=eo, latest_offset=lo)
    sol_u = ipr._load_klines_unified(ipr._SOL_KLINES_PATH, earliest_offset=eo, latest_offset=lo)
    max_lb = max(strat.gate.mtf_lookbacks)

    def slice_all(u):
        return {ep: ipr._slice_per_entry(kl, kline_cutoff_seconds=KCUT,
                                         max_lookback=max_lb, earliest_offset=eo)
                for ep, kl in u.items()}

    btc_k, eth_k, sol_k = slice_all(btc_u), slice_all(eth_u), slice_all(sol_u)
    bets: list[tuple[float, bool]] = []

    for _name, es, ee in CV5:
        sim_rounds = [r for r in all_rounds if es <= r.epoch <= ee]
        gate_config = MomentumGateConfig(
            enabled=True, bnb_symbol="BNB-USDT", btc_symbol="BTC-USDT",
            eth_symbol="ETH-USDT", sol_symbol="SOL-USDT", kline_cutoff_seconds=KCUT,
            mtf_lookbacks=strat.gate.mtf_lookbacks,
            mtf_min_return_threshold=strat.gate.mtf_min_return_threshold,
        )
        tracker = InMemoryBankrollTracker(
            initial_bankroll=BANK,
            drawdown_peak_window_days=strat.risk.drawdown_peak_window_days,
            peak_mode=strat.risk.drawdown_peak_mode,
        )
        pipe = _CapturePipeline(
            config=gate_config, strategy_config=strat, gate=None,
            kline_cutoff_seconds=KCUT, pool_cutoff_seconds=PCUT,
            min_bet_amount_bnb=min_bet, treasury_fee_fraction=FEE,
            bankroll_tracker=tracker,
        )
        pipe.refresh_btc_klines(btc_klines_by_epoch=btc_k)
        pipe.refresh_eth_klines(eth_klines_by_epoch=eth_k)
        pipe.refresh_sol_klines(sol_klines_by_epoch=sol_k)
        pipe.refresh_bnb_klines(bnb_klines_by_epoch={})

        bankroll = BANK
        for round_t in sim_rounds:
            pipe.last_result = None
            decision = pipe.decide_open_round(round_t=round_t)
            if decision.action == "BET" and decision.bet_size_bnb > 0.0:
                bankroll -= decision.bet_size_bnb + MAX_GAS_COST_BET_BNB
                outcome = settle_bet_against_closed_round(
                    bet_bnb=decision.bet_size_bnb, bet_side=decision.bet_side,
                    round_closed=round_t, treasury_fee_fraction=FEE,
                )
                bankroll += outcome.credit_bnb
                profit = outcome.credit_bnb - decision.bet_size_bnb - MAX_GAS_COST_BET_BNB
                pool_total = _pool_total_for_round(round_t, PCUT)
                label = _classify_bet(result=pipe.last_result, pool_total=pool_total, strategy=strat)
                bets.append((profit, label == "keeps"))
            pipe.record_settlement(bankroll=bankroll, start_at=int(round_t.start_at))
            pipe.settle_closed_rounds(rounds=[round_t])
    return bets


def main() -> int:
    bets = _collect_cv5_bets()
    pnls = np.array([p for p, _ in bets])
    keeps_mask = np.array([k for _, k in bets])
    n_total, n_keeps = len(bets), int(keeps_mask.sum())
    keeps_net = float(pnls[keeps_mask].sum())
    total_net = float(pnls.sum())
    print(f"CV5: bets={n_total} total_net={total_net:+.4f} keeps={n_keeps} "
          f"keeps_net={keeps_net:+.4f}", flush=True)

    if n_total != 1446 or abs(total_net - 50.4953) > 1e-2:
        print(f"VALIDITY GATE FAIL: expected 1446 / +50.4953, "
              f"got {n_total} / {total_net:+.4f} -> ABORT")
        return 1
    print("VALIDITY GATE: PASS (canonical CV5 reproduced)")

    rng = np.random.default_rng(SEED)
    null = np.empty(N_PERM)
    for i in range(N_PERM):
        idx = rng.choice(n_total, size=n_keeps, replace=False)
        null[i] = pnls[idx].sum()
    p_left = (int(np.sum(null <= keeps_net)) + 1) / (N_PERM + 1)
    lift = -keeps_net
    print(f"\nkeeps_net = {keeps_net:+.4f} BNB  (naive lift if removed = {lift:+.4f})")
    print(f"null (random {n_keeps}-subset net, N={N_PERM}): "
          f"mean={null.mean():+.4f} std={null.std():.4f} "
          f"min={null.min():+.4f} max={null.max():+.4f}")
    print(f"PERMUTATION p_left = P(random net <= keeps net) = {p_left:.4f}")
    verdict = ("PASS — keeps subset is non-randomly bad (lift is real)"
               if p_left < 0.05 else
               "NULL NOT REJECTED — lift consistent with luck -> STOP")
    print(f"VERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
