"""Generate block artifacts for a walk-forward ML strategy.

This inspection runner reuses the active feature + model stack and writes
`dislocation_trades.csv` + `dislocation_summary.json` per block so the existing
router tooling can include the ML strategy alongside dislocation strategies.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pancakebot.core.constants import BNB_WEI, GAS_COST_BET_BNB, GAS_COST_CLAIM_BNB
from pancakebot.core.errors import InvariantError
from pancakebot.domain.features.feature_builder import build_features, vectorize
from pancakebot.domain.features.pool_amounts import compute_pool_amounts_wei_at_or_before
from pancakebot.domain.features.schema import (
    FEATURE_SCHEMA,
    max_required_context_klines_size,
    max_required_prior_context_rounds_size,
)
from pancakebot.domain.models.walk_forward import (
    WalkForwardState,
    ensure_state,
    predict_probabilities,
    predict_tradeable_probability,
)
from pancakebot.domain.types import Kline, Round
from pancakebot.infra.closed_rounds_store import ClosedRoundsStore
from pancakebot.infra.klines_store import KlinesStore
from pancakebot.runtime.contract_constants_cache import load_contract_constants
from pancakebot.runtime.settlement import settle_bet_against_closed_round


@dataclass(frozen=True, slots=True)
class MlWalkForwardConfig:
    """Configuration consumed by the restored walk-forward model owner."""

    # Data source required by walk_forward._context_klines_for_round().
    klines_store: KlinesStore

    # Time boundary for cutoff features.
    cutoff_seconds: int

    # Walk-forward windowing.
    train_size: int
    calibrate_size: int
    retrain_interval: int
    recalibrate_interval: int

    # Model regularization and deterministic seed.
    price_alpha: float
    pool_alpha_total: float
    pool_alpha_ratio: float
    random_seed: int

    # Recency weighting knobs for training/calibration rows.
    recency_weight_floor: float
    recency_weight_power: float

    # Predictability-label baseline bet sizing.
    predictability_baseline_bet_bnb: float

    # Settlement semantics.
    treasury_fee_fraction: float


@dataclass(frozen=True, slots=True)
class MlDecision:
    """One round decision emitted by the ML strategy runner."""

    action: str
    direction: str | None
    skip_reason: str | None
    p_bull: float | None
    p_market_bull: float | None
    dislocation_bull: float | None
    expected_net_bull: float | None
    expected_net_bear: float | None
    expected_net_selected: float | None
    pool_total_bnb_cutoff: float | None
    bet_size_bnb: float


def _safe_rate(num: int, den: int) -> float:
    if int(den) <= 0:
        return 0.0
    return float(num) / float(den)


def _offsets(*, block_size: int, num_blocks: int, skip_most_recent_blocks: int) -> list[int]:
    return [
        int(block_size) * i
        for i in range(
            int(num_blocks) + int(skip_most_recent_blocks) - 1,
            int(skip_most_recent_blocks) - 1,
            -1,
        )
    ]


def _scenario_name(*, name_prefix: str, block_idx: int, num_blocks: int, offset: int) -> str:
    return f"{name_prefix}_b{int(block_idx)}of{int(num_blocks)}_off{int(offset)}"


def _load_reference_block_epochs(*, scenario_dir: Path) -> list[int]:
    """Load epoch sequence from an existing block trade artifact."""

    path = scenario_dir / "dislocation_trades.csv"
    if not path.exists():
        raise FileNotFoundError(f"ml_align_missing_dislocation_trades: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        out = [int(row["epoch"]) for row in reader]
    if not out:
        raise InvariantError("ml_align_reference_epochs_empty")
    return out


def _valid_rounds_from_store(store: ClosedRoundsStore) -> list[Round]:
    rounds: list[Round] = []
    for round_t in store.iter_closed_rounds():
        if bool(round_t.failed):
            continue
        if round_t.lock_at is None or round_t.close_at is None:
            continue
        if round_t.lock_price is None or round_t.close_price is None:
            continue
        if float(round_t.lock_price) <= 0.0 or float(round_t.close_price) <= 0.0:
            continue
        rounds.append(round_t)
    if not rounds:
        raise InvariantError("ml_blocks_rounds_empty")
    return rounds


def _tail_block_rounds(*, rounds: list[Round], block_size: int, sim_offset_rounds: int) -> tuple[int, int]:
    end = len(rounds) - int(sim_offset_rounds)
    if int(end) <= 0:
        raise InvariantError("ml_blocks_offset_out_of_range")
    start = int(end) - int(block_size)
    if int(start) < 0:
        raise InvariantError("ml_blocks_insufficient_rounds_for_block")
    return int(start), int(end)


def _context_klines(*, klines_store: KlinesStore, round_t: Round, cutoff_seconds: int) -> list[Kline]:
    if round_t.lock_at is None:
        raise InvariantError("ml_round_lock_at_missing")
    kk = int(max_required_context_klines_size())
    cutoff_ts = int(round_t.lock_at) - int(cutoff_seconds)
    anchor_ms = int(cutoff_ts) * 1000
    latest_close_ms = klines_store.latest_close_time_ms()
    if latest_close_ms is None:
        raise InvariantError("ml_klines_store_empty")
    if int(latest_close_ms) < int(anchor_ms):
        anchor_ms = int(latest_close_ms)
    return list(
        klines_store.get_context_klines(
            anchor_close_time_ms=int(anchor_ms),
            size=int(kk),
        )
    )


def _expected_net_from_predicted_final(
    *,
    p_bull: float,
    side: str,
    stake_bnb: float,
    final_bull_bnb: float,
    final_bear_bnb: float,
    treasury_fee_fraction: float,
) -> float:
    """Compute impact-aware expected net (including bet and claim gas)."""

    side_u = str(side).upper()
    if side_u not in ("BULL", "BEAR"):
        raise InvariantError("ml_side_invalid")

    if not (0.0 <= float(p_bull) <= 1.0):
        raise InvariantError("ml_p_bull_out_of_range")
    if float(stake_bnb) <= 0.0:
        raise InvariantError("ml_stake_nonpositive")
    if float(final_bull_bnb) <= 0.0 or float(final_bear_bnb) <= 0.0:
        return float("-inf")
    if not (0.0 <= float(treasury_fee_fraction) < 1.0):
        raise InvariantError("ml_treasury_fee_out_of_range")

    final_total_bnb = float(final_bull_bnb) + float(final_bear_bnb)
    if side_u == "BULL":
        adj_side_bnb = float(final_bull_bnb) + float(stake_bnb)
        p_win = float(p_bull)
    else:
        adj_side_bnb = float(final_bear_bnb) + float(stake_bnb)
        p_win = 1.0 - float(p_bull)
    adj_total_bnb = float(final_total_bnb) + float(stake_bnb)
    if float(adj_side_bnb) <= 0.0 or float(adj_total_bnb) <= 0.0:
        return float("-inf")

    payout_multiple = (float(adj_total_bnb) * (1.0 - float(treasury_fee_fraction))) / float(adj_side_bnb)
    win_credit_bnb = float(stake_bnb) * float(payout_multiple) - float(GAS_COST_CLAIM_BNB)
    expected_credit_bnb = float(p_win) * float(win_credit_bnb)
    expected_net_bnb = float(expected_credit_bnb) - (float(stake_bnb) + float(GAS_COST_BET_BNB))
    return float(expected_net_bnb)


def _predict_round_decision(
    *,
    cfg: MlWalkForwardConfig,
    state: WalkForwardState,
    history_rounds: list[Round],
    round_t: Round,
    fixed_bet_bnb: float,
    min_tradeable_prob: float,
    min_prob_edge: float,
    cutoff_pool_total_min_bnb: float,
    expected_net_min_bnb: float,
) -> MlDecision:
    k = int(max_required_prior_context_rounds_size())
    if len(history_rounds) < int(k):
        raise InvariantError("ml_history_insufficient_for_features")

    cutoff_ts = int(round_t.lock_at) - int(cfg.cutoff_seconds)
    pools = compute_pool_amounts_wei_at_or_before(bets=round_t.bets, cutoff_ts=int(cutoff_ts))
    pool_total_bnb = float(pools.total_wei) / float(BNB_WEI)
    pool_bull_bnb = float(pools.bull_wei) / float(BNB_WEI)
    pool_bear_bnb = float(pools.bear_wei) / float(BNB_WEI)
    if float(pool_total_bnb) <= 0.0:
        return MlDecision(
            action="SKIP",
            direction=None,
            skip_reason="cutoff_pool_empty",
            p_bull=None,
            p_market_bull=None,
            dislocation_bull=None,
            expected_net_bull=None,
            expected_net_bear=None,
            expected_net_selected=None,
            pool_total_bnb_cutoff=float(pool_total_bnb),
            bet_size_bnb=0.0,
        )
    if float(pool_total_bnb) < float(cutoff_pool_total_min_bnb):
        return MlDecision(
            action="SKIP",
            direction=None,
            skip_reason="cutoff_pool_below_min_total",
            p_bull=None,
            p_market_bull=None,
            dislocation_bull=None,
            expected_net_bull=None,
            expected_net_bear=None,
            expected_net_selected=None,
            pool_total_bnb_cutoff=float(pool_total_bnb),
            bet_size_bnb=0.0,
        )

    prior_context_rounds = list(history_rounds[-int(k):])
    features = build_features(
        target_round=round_t,
        prior_context_rounds=prior_context_rounds,
        context_klines=_context_klines(
            klines_store=cfg.klines_store,
            round_t=round_t,
            cutoff_seconds=int(cfg.cutoff_seconds),
        ),
        cutoff_seconds=int(cfg.cutoff_seconds),
    )
    x_row = vectorize(features=features, schema=FEATURE_SCHEMA)

    mu = float(state.models.price_model.predict([list(x_row)])[0])  # type: ignore[union-attr]
    p_bull = float(predict_probabilities(state=state, mu=float(mu)))
    p_tradeable = float(predict_tradeable_probability(state=state, x_row=list(x_row)))
    p_market_bull = float(pool_bull_bnb / pool_total_bnb)
    dislocation_bull = float(p_bull) - float(p_market_bull)

    if float(p_tradeable) < float(min_tradeable_prob):
        return MlDecision(
            action="SKIP",
            direction=None,
            skip_reason="predictability_below_min",
            p_bull=float(p_bull),
            p_market_bull=float(p_market_bull),
            dislocation_bull=float(dislocation_bull),
            expected_net_bull=None,
            expected_net_bear=None,
            expected_net_selected=None,
            pool_total_bnb_cutoff=float(pool_total_bnb),
            bet_size_bnb=0.0,
        )

    if abs(float(p_bull) - 0.5) < float(min_prob_edge):
        return MlDecision(
            action="SKIP",
            direction=None,
            skip_reason="p_bull_edge_below_min",
            p_bull=float(p_bull),
            p_market_bull=float(p_market_bull),
            dislocation_bull=float(dislocation_bull),
            expected_net_bull=None,
            expected_net_bear=None,
            expected_net_selected=None,
            pool_total_bnb_cutoff=float(pool_total_bnb),
            bet_size_bnb=0.0,
        )

    late_total_bnb, late_bull_frac = state.models.pool_model.predict([list(x_row)])[0]  # type: ignore[union-attr]
    late_total_bnb = max(0.0, float(late_total_bnb))
    late_bull_frac = min(1.0, max(0.0, float(late_bull_frac)))

    final_total_bnb = float(pool_total_bnb) + float(late_total_bnb)
    final_bull_bnb = float(pool_bull_bnb) + float(late_total_bnb) * float(late_bull_frac)
    final_bear_bnb = float(pool_bear_bnb) + float(late_total_bnb) * (1.0 - float(late_bull_frac))
    if float(final_total_bnb) <= 0.0 or float(final_bull_bnb) <= 0.0 or float(final_bear_bnb) <= 0.0:
        return MlDecision(
            action="SKIP",
            direction=None,
            skip_reason="predicted_pool_invalid",
            p_bull=float(p_bull),
            p_market_bull=float(p_market_bull),
            dislocation_bull=float(dislocation_bull),
            expected_net_bull=None,
            expected_net_bear=None,
            expected_net_selected=None,
            pool_total_bnb_cutoff=float(pool_total_bnb),
            bet_size_bnb=0.0,
        )

    ev_bull = _expected_net_from_predicted_final(
        p_bull=float(p_bull),
        side="BULL",
        stake_bnb=float(fixed_bet_bnb),
        final_bull_bnb=float(final_bull_bnb),
        final_bear_bnb=float(final_bear_bnb),
        treasury_fee_fraction=float(cfg.treasury_fee_fraction),
    )
    ev_bear = _expected_net_from_predicted_final(
        p_bull=float(p_bull),
        side="BEAR",
        stake_bnb=float(fixed_bet_bnb),
        final_bull_bnb=float(final_bull_bnb),
        final_bear_bnb=float(final_bear_bnb),
        treasury_fee_fraction=float(cfg.treasury_fee_fraction),
    )
    if float(ev_bull) >= float(ev_bear):
        direction = "BULL"
        best_ev = float(ev_bull)
    else:
        direction = "BEAR"
        best_ev = float(ev_bear)

    if float(best_ev) < float(expected_net_min_bnb):
        return MlDecision(
            action="SKIP",
            direction=None,
            skip_reason="expected_net_below_min",
            p_bull=float(p_bull),
            p_market_bull=float(p_market_bull),
            dislocation_bull=float(dislocation_bull),
            expected_net_bull=float(ev_bull),
            expected_net_bear=float(ev_bear),
            expected_net_selected=float(best_ev),
            pool_total_bnb_cutoff=float(pool_total_bnb),
            bet_size_bnb=0.0,
        )

    return MlDecision(
        action="BET",
        direction=str(direction),
        skip_reason=None,
        p_bull=float(p_bull),
        p_market_bull=float(p_market_bull),
        dislocation_bull=float(dislocation_bull),
        expected_net_bull=float(ev_bull),
        expected_net_bear=float(ev_bear),
        expected_net_selected=float(best_ev),
        pool_total_bnb_cutoff=float(pool_total_bnb),
        bet_size_bnb=float(fixed_bet_bnb),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name-prefix", type=str, required=True)
    parser.add_argument("--align-to-prefix", type=str, default=None)
    parser.add_argument("--closed-rounds-path", type=str, default="var/closed_rounds.jsonl")
    parser.add_argument("--klines-path", type=str, default="var/klines.jsonl")
    parser.add_argument("--block-size", type=int, default=500)
    parser.add_argument("--num-blocks", type=int, default=20)
    parser.add_argument("--skip-most-recent-blocks", type=int, default=0)
    parser.add_argument("--cutoff-seconds", type=int, default=17)
    parser.add_argument("--initial-bankroll-bnb", type=float, default=500.0)
    parser.add_argument("--fixed-bet-bnb", type=float, default=0.2)
    parser.add_argument("--min-tradeable-prob", type=float, default=0.55)
    parser.add_argument("--min-prob-edge", type=float, default=0.02)
    parser.add_argument("--cutoff-pool-total-min-bnb", type=float, default=1.2)
    parser.add_argument("--expected-net-min-bnb", type=float, default=0.0)
    parser.add_argument("--train-size", type=int, default=20000)
    parser.add_argument("--calibrate-size", type=int, default=2000)
    parser.add_argument("--retrain-interval", type=int, default=500)
    parser.add_argument("--recalibrate-interval", type=int, default=250)
    parser.add_argument("--price-alpha", type=float, default=1.0)
    parser.add_argument("--pool-alpha-total", type=float, default=1.0)
    parser.add_argument("--pool-alpha-ratio", type=float, default=1.0)
    parser.add_argument("--recency-weight-floor", type=float, default=0.7)
    parser.add_argument("--recency-weight-power", type=float, default=1.0)
    parser.add_argument("--predictability-baseline-bet-bnb", type=float, default=0.05)
    parser.add_argument("--random-seed", type=int, default=1337)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if int(args.block_size) <= 0:
        raise InvariantError("ml_block_size_nonpositive")
    if int(args.num_blocks) <= 0:
        raise InvariantError("ml_num_blocks_nonpositive")
    if int(args.skip_most_recent_blocks) < 0:
        raise InvariantError("ml_skip_most_recent_blocks_negative")
    if float(args.fixed_bet_bnb) <= 0.0:
        raise InvariantError("ml_fixed_bet_nonpositive")
    if float(args.initial_bankroll_bnb) <= 0.0:
        raise InvariantError("ml_initial_bankroll_nonpositive")

    constants = load_contract_constants()
    cfg = MlWalkForwardConfig(
        klines_store=KlinesStore(str(args.klines_path)),
        cutoff_seconds=int(args.cutoff_seconds),
        train_size=int(args.train_size),
        calibrate_size=int(args.calibrate_size),
        retrain_interval=int(args.retrain_interval),
        recalibrate_interval=int(args.recalibrate_interval),
        price_alpha=float(args.price_alpha),
        pool_alpha_total=float(args.pool_alpha_total),
        pool_alpha_ratio=float(args.pool_alpha_ratio),
        random_seed=int(args.random_seed),
        recency_weight_floor=float(args.recency_weight_floor),
        recency_weight_power=float(args.recency_weight_power),
        predictability_baseline_bet_bnb=float(args.predictability_baseline_bet_bnb),
        treasury_fee_fraction=float(constants.treasury_fee_fraction),
    )

    rounds = _valid_rounds_from_store(ClosedRoundsStore(str(args.closed_rounds_path)))
    rounds_by_epoch: dict[int, Round] = {int(r.epoch): r for r in rounds}
    round_index_by_epoch: dict[int, int] = {int(r.epoch): idx for idx, r in enumerate(rounds)}
    k = int(max_required_prior_context_rounds_size())
    min_history = int(k + cfg.train_size + cfg.calibrate_size)

    out_dir = Path("var/exp")
    out_dir.mkdir(parents=True, exist_ok=True)

    block_rows: list[dict[str, Any]] = []
    block_nets: list[float] = []
    bets_total = 0
    wins_total = 0

    for block_idx, offset in enumerate(
        _offsets(
            block_size=int(args.block_size),
            num_blocks=int(args.num_blocks),
            skip_most_recent_blocks=int(args.skip_most_recent_blocks),
        ),
        start=1,
    ):
        scenario_name = _scenario_name(
            name_prefix=str(args.name_prefix),
            block_idx=int(block_idx),
            num_blocks=int(args.num_blocks),
            offset=int(offset),
        )
        scenario_dir = out_dir / scenario_name
        scenario_dir.mkdir(parents=True, exist_ok=True)

        if args.align_to_prefix is None:
            start, end = _tail_block_rounds(
                rounds=rounds,
                block_size=int(args.block_size),
                sim_offset_rounds=int(offset),
            )
            if int(start) < int(min_history):
                raise InvariantError("ml_block_insufficient_warm_history")
            history_rounds = list(rounds[: int(start)])
            block_rounds = list(rounds[int(start) : int(end)])
            if len(block_rounds) != int(args.block_size):
                raise InvariantError("ml_block_size_mismatch")
        else:
            ref_name = _scenario_name(
                name_prefix=str(args.align_to_prefix),
                block_idx=int(block_idx),
                num_blocks=int(args.num_blocks),
                offset=int(offset),
            )
            ref_epochs = _load_reference_block_epochs(scenario_dir=out_dir / ref_name)
            first_epoch = int(ref_epochs[0])
            first_idx = round_index_by_epoch.get(int(first_epoch))
            if first_idx is None:
                raise InvariantError("ml_align_first_epoch_not_found_in_round_store")
            if int(first_idx) < int(min_history):
                raise InvariantError("ml_align_insufficient_warm_history")

            history_rounds = list(rounds[: int(first_idx)])
            block_rounds = []
            for ep in ref_epochs:
                r = rounds_by_epoch.get(int(ep))
                if r is None:
                    raise InvariantError(f"ml_align_epoch_not_found_in_round_store: {int(ep)}")
                block_rounds.append(r)
            if len(block_rounds) != len(ref_epochs):
                raise InvariantError("ml_align_block_size_mismatch")

        state: WalkForwardState | None = None
        bankroll_bnb = float(args.initial_bankroll_bnb)
        block_net = 0.0
        num_bets = 0
        num_wins = 0
        skip_counts: dict[str, int] = {}
        trade_rows: list[list[Any]] = [
            [
                "epoch",
                "action",
                "skip_reason",
                "direction",
                "p_nowcast_bull",
                "p_market_bull",
                "dislocation_bull",
                "expected_net_bull",
                "expected_net_bear",
                "expected_net_selected",
                "pool_total_bnb_cutoff",
                "bet_size_bnb",
                "profit_bnb",
                "bankroll_bnb",
            ]
        ]

        for round_t in block_rounds:
            state = ensure_state(
                cfg=cfg,
                closed_rounds=history_rounds,
                current_epoch=int(round_t.epoch),
                state=state,
            )
            decision = _predict_round_decision(
                cfg=cfg,
                state=state,
                history_rounds=history_rounds,
                round_t=round_t,
                fixed_bet_bnb=float(args.fixed_bet_bnb),
                min_tradeable_prob=float(args.min_tradeable_prob),
                min_prob_edge=float(args.min_prob_edge),
                cutoff_pool_total_min_bnb=float(args.cutoff_pool_total_min_bnb),
                expected_net_min_bnb=float(args.expected_net_min_bnb),
            )

            profit_bnb = 0.0
            if str(decision.action) == "BET":
                if decision.direction is None:
                    raise InvariantError("ml_bet_missing_direction")
                settle = settle_bet_against_closed_round(
                    bet_bnb=float(decision.bet_size_bnb),
                    bet_side=str(decision.direction),
                    round_closed=round_t,
                    treasury_fee_fraction=float(cfg.treasury_fee_fraction),
                )
                profit_bnb = float(settle.credit_bnb) - float(decision.bet_size_bnb) - float(GAS_COST_BET_BNB)
                bankroll_bnb += float(profit_bnb)
                block_net += float(profit_bnb)
                num_bets += 1
                if float(profit_bnb) > 0.0:
                    num_wins += 1
            else:
                reason = str(decision.skip_reason or "unknown_skip_reason")
                skip_counts[reason] = int(skip_counts.get(reason, 0) + 1)

            trade_rows.append(
                [
                    int(round_t.epoch),
                    str(decision.action),
                    str(decision.skip_reason or ""),
                    str(decision.direction or ""),
                    "" if decision.p_bull is None else float(decision.p_bull),
                    "" if decision.p_market_bull is None else float(decision.p_market_bull),
                    "" if decision.dislocation_bull is None else float(decision.dislocation_bull),
                    "" if decision.expected_net_bull is None else float(decision.expected_net_bull),
                    "" if decision.expected_net_bear is None else float(decision.expected_net_bear),
                    "" if decision.expected_net_selected is None else float(decision.expected_net_selected),
                    "" if decision.pool_total_bnb_cutoff is None else float(decision.pool_total_bnb_cutoff),
                    float(decision.bet_size_bnb),
                    float(profit_bnb),
                    float(bankroll_bnb),
                ]
            )

            history_rounds.append(round_t)

        trades_path = scenario_dir / "dislocation_trades.csv"
        with trades_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerows(trade_rows)

        summary = {
            "scenario": {
                "name": str(scenario_name),
                "block_index": int(block_idx),
                "sim_offset_rounds": int(offset),
                "block_size": int(args.block_size),
                "strategy_family": "ml_walkforward",
            },
            "config": {
                "cutoff_seconds": int(cfg.cutoff_seconds),
                "fixed_bet_bnb": float(args.fixed_bet_bnb),
                "min_tradeable_prob": float(args.min_tradeable_prob),
                "min_prob_edge": float(args.min_prob_edge),
                "cutoff_pool_total_min_bnb": float(args.cutoff_pool_total_min_bnb),
                "expected_net_min_bnb": float(args.expected_net_min_bnb),
                "train_size": int(cfg.train_size),
                "calibrate_size": int(cfg.calibrate_size),
                "retrain_interval": int(cfg.retrain_interval),
                "recalibrate_interval": int(cfg.recalibrate_interval),
                "price_alpha": float(cfg.price_alpha),
                "pool_alpha_total": float(cfg.pool_alpha_total),
                "pool_alpha_ratio": float(cfg.pool_alpha_ratio),
                "recency_weight_floor": float(cfg.recency_weight_floor),
                "recency_weight_power": float(cfg.recency_weight_power),
                "predictability_baseline_bet_bnb": float(cfg.predictability_baseline_bet_bnb),
            },
            "initial_bankroll_bnb": float(args.initial_bankroll_bnb),
            "final_bankroll_bnb": float(bankroll_bnb),
            "net_profit_bnb": float(block_net),
            "num_rounds": int(args.block_size),
            "num_bets": int(num_bets),
            "num_wins": int(num_wins),
            "bet_rate": float(_safe_rate(num_bets, int(args.block_size))),
            "win_rate": float(_safe_rate(num_wins, num_bets)),
            "num_skips_by_reason": {str(k): int(v) for k, v in sorted(skip_counts.items())},
        }
        (scenario_dir / "dislocation_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        print(
            "BLOCK_DONE "
            + f"block={int(block_idx)}/{int(args.num_blocks)} offset={int(offset)} "
            + f"net={float(block_net):.6f} bets={int(num_bets)} win={float(_safe_rate(num_wins, num_bets)):.4f}"
        )
        block_rows.append(
            {
                "scenario": str(scenario_name),
                "block_index": int(block_idx),
                "sim_offset_rounds": int(offset),
                "net": float(block_net),
                "bets": int(num_bets),
                "wins": int(num_wins),
                "bet_rate": float(_safe_rate(num_bets, int(args.block_size))),
                "win_rate": float(_safe_rate(num_wins, num_bets)),
            }
        )
        block_nets.append(float(block_net))
        bets_total += int(num_bets)
        wins_total += int(num_wins)

    agg = {
        "blocks": int(len(block_rows)),
        "net_total": float(sum(block_nets)),
        "net_mean": float(sum(block_nets) / len(block_nets)),
        "net_median": float(statistics.median(block_nets)),
        "net_worst": float(min(block_nets)),
        "net_best": float(max(block_nets)),
        "positive_blocks": int(sum(1 for x in block_nets if float(x) > 0.0)),
        "positive_block_frac": float(sum(1 for x in block_nets if float(x) > 0.0) / len(block_nets)),
        "net_per_500": float(sum(block_nets) / float(int(args.block_size) * int(args.num_blocks)) * 500.0),
        "bets_total": int(bets_total),
        "win_rate_weighted": float(_safe_rate(wins_total, bets_total)),
    }
    agg_path = out_dir / f"{args.name_prefix}_aggregate.json"
    agg_path.write_text(
        json.dumps(
            {
                "name_prefix": str(args.name_prefix),
                "block_size": int(args.block_size),
                "num_blocks": int(args.num_blocks),
                "skip_most_recent_blocks": int(args.skip_most_recent_blocks),
                "rows": block_rows,
                "aggregate": agg,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"AGG={agg_path}")
    print(f"NET_PER_500={agg['net_per_500']}")


if __name__ == "__main__":
    main()
