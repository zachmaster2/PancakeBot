from __future__ import annotations

import csv
import json
from collections import deque
from pathlib import Path
from typing import Deque

from pancakebot.backtest.config import BacktestConfig
from pancakebot.core.constants import GAS_COST_BET_BNB
from pancakebot.domain.closed_rounds_cache import RollingClosedRoundsCache
from pancakebot.domain.contiguity import check_klines_contiguous, check_rounds_contiguous
from pancakebot.domain.types import Kline, Round
from pancakebot.domain.features.schema import max_required_context_klines_size, max_required_prior_context_rounds_size
from pancakebot.domain.strategy.planner import build_inputs, predict, size_bet
from pancakebot.runtime.cache_policy import compute_required_cache_size
from pancakebot.runtime.model_manager import ModelManager
from pancakebot.runtime.settlement import settle_bet_against_closed_round
from pancakebot.core.errors import InvariantError
from pancakebot.core.logging import info


def _tail_rounds(store, *, n: int) -> list[Round]:
    if n <= 0:
        raise InvariantError("tail_rounds_n_invalid")

    dq: Deque[Round] = deque(maxlen=n)
    for r in store.iter_closed_rounds():
        dq.append(r)
    return list(dq)


def _build_context_klines(*, klines_store, target_round: Round, cutoff_seconds: int) -> list[Kline]:
    kk = int(max_required_context_klines_size())
    if target_round.lock_at is None:
        raise InvariantError("backtest_round_missing_lock_at")
    lock_ts = int(target_round.lock_at)
    cutoff_ts = int(lock_ts) - int(cutoff_seconds)
    anchor_ms = int(cutoff_ts) * 1000

    latest_close_ms = klines_store.latest_close_time_ms()
    if latest_close_ms is None:
        raise InvariantError("klines_store_empty")

    # If we haven't synced up to cutoff yet (dry/live), anchor to what we actually have.
    if int(latest_close_ms) < int(anchor_ms):
        anchor_ms = int(latest_close_ms)

    return klines_store.get_context_klines(anchor_close_time_ms=int(anchor_ms), size=int(kk))


def _safe_rate(num: int, den: int) -> float:
    if int(den) <= 0:
        return 0.0
    return float(num) / float(den)


def run_backtest(*, runtime_cfg, backtest_cfg: BacktestConfig, out_dir: Path) -> None:
    """Run a deterministic replay over the most recent closed rounds.

    Backtest MUST NOT fetch any data. It consumes the on-disk round store and kline store.

    Artifacts are written incrementally to `{out_dir}/`:
      - backtest_trades.csv
      - backtest_summary.json
    """

    store = runtime_cfg.round_store

    train_size = int(runtime_cfg.train_size)
    calibrate_size = int(runtime_cfg.calibrate_size)
    k = int(max_required_prior_context_rounds_size())
    if k <= 0:
        context_desc = "target_only"
    else:
        context_desc = f"prior_context_rounds[{int(k)}]"

    simulation_size = int(backtest_cfg.simulation_size)

    required_cache = compute_required_cache_size(
        train_size=train_size,
        calibrate_size=calibrate_size,
    )
    total_n = int(required_cache) + int(simulation_size)

    closed_rounds = _tail_rounds(store, n=total_n)
    if len(closed_rounds) < total_n:
        raise InvariantError("backtest_insufficient_closed_rounds")

    warmup_rounds = closed_rounds[:required_cache]
    sim_rounds = closed_rounds[required_cache:]
    if len(sim_rounds) != int(simulation_size):
        raise InvariantError("backtest_sim_rounds_len_mismatch")

    cache = RollingClosedRoundsCache(rounds=warmup_rounds, capacity=int(required_cache))
    info(
        "BACK",
        "INIT",
        "CACHE",
        msg=(
            "Loaded closed rounds: "
            f"warmup_n={len(warmup_rounds)} sim_n={len(sim_rounds)} k={k} context={context_desc}"
        ),
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    trades_path = out_dir / "backtest_trades.csv"
    summary_path = out_dir / "backtest_summary.json"

    trades_f = open(trades_path, "w", newline="")
    try:
        trades_w = csv.writer(trades_f)

        trades_w.writerow(
            [
                "epoch",
                "action",
                "skip_reason",
                "direction",
                "bet_size_bnb",
                "p_final",
                "final_total_bnb",
                "final_bull_bnb",
                "final_bear_bnb",
                "ev_bnb",
                "profit_bnb",
                "bankroll_bnb",
            ]
        )
        initial_bankroll_bnb = float(backtest_cfg.initial_bankroll_bnb)
        bankroll = float(initial_bankroll_bnb)

        model_manager = ModelManager()

        num_bets = 0
        num_bets_bull = 0
        num_bets_bear = 0
        num_wins = 0
        num_wins_bull = 0
        num_wins_bear = 0
        gross_profit_bnb = 0.0
        gross_loss_bnb = 0.0
        skip_counts_by_reason: dict[str, int] = {}

        def _count_skip(reason: str) -> None:
            key = str(reason).strip()
            if key == "":
                key = "unknown_skip_reason"
            skip_counts_by_reason[key] = int(skip_counts_by_reason.get(key, 0)) + 1

        for idx, round_t in enumerate(sim_rounds):
            if round_t.lock_at is None:
                raise InvariantError("backtest_round_missing_lock_at")

            if int(k) <= 0:
                prior_context_rounds: list[Round] = []
            else:
                prior_context_rounds = list(cache.rounds[-int(k):])
            if len(prior_context_rounds) != int(k):
                raise InvariantError("backtest_prior_context_len_mismatch")

            rounds_ok, rounds_reason = check_rounds_contiguous(
                prior_context_rounds=prior_context_rounds,
                target_round=round_t,
                buffer_seconds=int(runtime_cfg.buffer_seconds),
            )
            if not rounds_ok:
                _count_skip(f"rounds:{str(rounds_reason)}")
                info("BACK", "ACT", "SKIP", msg=f"Skip epoch {int(round_t.epoch)}: {rounds_reason}")
                cache.extend([round_t])
                continue

            wf_state = model_manager.step(
                cfg=runtime_cfg,
                closed_rounds=cache.rounds,
                current_epoch=int(round_t.epoch),
            )

            context_klines = _build_context_klines(
                klines_store=runtime_cfg.klines_store,
                target_round=round_t,
                cutoff_seconds=int(runtime_cfg.cutoff_seconds),
            )
            klines_ok, klines_reason = check_klines_contiguous(context_klines=context_klines)
            if not klines_ok:
                _count_skip(f"klines:{str(klines_reason)}")
                info("BACK", "ACT", "SKIP", msg=f"Skip epoch {int(round_t.epoch)}: {klines_reason}")
                cache.extend([round_t])
                continue

            feats = build_inputs(
                cfg=runtime_cfg,
                prior_context_rounds=prior_context_rounds,
                context_klines=context_klines,
                target_round=round_t,
            )

            pred = predict(state=wf_state, feats=feats)
            decision = size_bet(
                cfg=runtime_cfg,
                pred=pred,
                bankroll_bnb=float(bankroll),
            )

            profit = 0.0
            ev = float(decision.expected_profit_bnb)

            if decision.action == "BET" and decision.amount_bnb > 0.0:
                bet_side = str(decision.bet_side)
                if bet_side not in ("Bull", "Bear"):
                    raise InvariantError("backtest_bet_side_invalid")

                # Bet placement accounting.
                bankroll -= float(decision.amount_bnb) + float(GAS_COST_BET_BNB)

                # Settlement.
                outcome = settle_bet_against_closed_round(
                    bet_bnb=float(decision.amount_bnb),
                    bet_side=str(decision.bet_side),
                    round_closed=round_t,
                    treasury_fee_fraction=float(runtime_cfg.treasury_fee_fraction),
                )
                credit_bnb = float(outcome.credit_bnb)
                bankroll += float(credit_bnb)
                profit = float(credit_bnb) - float(decision.amount_bnb) - float(GAS_COST_BET_BNB)

                num_bets += 1
                if bet_side == "Bull":
                    num_bets_bull += 1
                else:
                    num_bets_bear += 1

                if str(outcome.outcome) == "win":
                    num_wins += 1
                    if bet_side == "Bull":
                        num_wins_bull += 1
                    else:
                        num_wins_bear += 1

                if float(profit) > 0.0:
                    gross_profit_bnb += float(profit)
                elif float(profit) < 0.0:
                    gross_loss_bnb += float(-float(profit))
            else:
                _count_skip(str(decision.skip_reason or "unknown_skip_reason"))

            trades_w.writerow(
                [
                    int(round_t.epoch),
                    str(decision.action),
                    str(decision.skip_reason or ""),
                    str(decision.bet_side or ""),
                    float(decision.amount_bnb),
                    float(pred.p_final),
                    float(pred.final_total_bnb),
                    float(pred.final_bull_bnb),
                    float(pred.final_bear_bnb),
                    float(ev),
                    float(profit),
                    float(bankroll),
                ]
            )

            cache.extend([round_t])

            if (idx + 1) % 250 == 0:
                info(
                    "BACK",
                    "PROG",
                    "SIM",
                    msg=f"idx={idx+1}/{len(sim_rounds)} bankroll={float(bankroll):.4f} BNB",
                )

        num_rounds = int(len(sim_rounds))
        num_skips = int(num_rounds) - int(num_bets)
        classified_skips = int(sum(skip_counts_by_reason.values()))
        unclassified_skips = int(num_skips) - int(classified_skips)
        if unclassified_skips > 0:
            _count_skip("unclassified_skip")

        summary = {
            "num_rounds": int(num_rounds),
            "num_bets": int(num_bets),
            "num_skips": int(num_skips),
            "num_wins": int(num_wins),
            "num_bets_bull": int(num_bets_bull),
            "num_bets_bear": int(num_bets_bear),
            "num_wins_bull": int(num_wins_bull),
            "num_wins_bear": int(num_wins_bear),
            "bet_rate": float(_safe_rate(num_bets, num_rounds)),
            "bet_rate_bull": float(_safe_rate(num_bets_bull, num_bets)),
            "bet_rate_bear": float(_safe_rate(num_bets_bear, num_bets)),
            "win_rate": float(_safe_rate(num_wins, num_bets)),
            "win_rate_bull": float(_safe_rate(num_wins_bull, num_bets_bull)),
            "win_rate_bear": float(_safe_rate(num_wins_bear, num_bets_bear)),
            "initial_bankroll_bnb": float(initial_bankroll_bnb),
            "final_bankroll_bnb": float(bankroll),
            "gross_profit_bnb": float(gross_profit_bnb),
            "gross_loss_bnb": float(gross_loss_bnb),
            "net_profit_bnb": float(bankroll - float(initial_bankroll_bnb)),
            "num_skips_by_reason": {str(k): int(v) for k, v in sorted(skip_counts_by_reason.items())},
        }
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

        info("BACK", "DONE", "SIM", msg=f"Final bankroll={float(bankroll):.4f} BNB")

    finally:
        trades_f.close()

