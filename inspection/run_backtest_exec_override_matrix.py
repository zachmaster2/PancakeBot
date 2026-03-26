from __future__ import annotations

import argparse
import csv
import json
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Iterator

from pancakebot.core.errors import InvariantError
from pancakebot.runtime.settlement import settle_bet_against_closed_round

from inspection.backtest_harness_common import (
    load_cfg,
    max_drawdown_bnb,
    render_table,
    resolve_exp_root,
    run_backtest_case,
    top_skip_reasons,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.toml")
    parser.add_argument("--name-prefix", type=str, default="exec_override_matrix")
    parser.add_argument("--sim-size", type=int, default=None)
    parser.add_argument("--execution-stakes", type=str, default="0.3,0.1,0.05")
    parser.add_argument("--initial-bankrolls", type=str, default="0.2,0.5,1.0")
    parser.add_argument("--top-skip-limit", type=int, default=3)
    return parser


def _parse_positive_floats(*, raw: str, label: str) -> list[float]:
    tokens = [x.strip() for x in str(raw).split(",") if x.strip()]
    if not tokens:
        raise InvariantError(f"{label}_empty")
    out: list[float] = []
    for token in tokens:
        try:
            value = float(token)
        except ValueError as e:
            raise InvariantError(f"{label}_not_number: {token}") from e
        if float(value) <= 0.0:
            raise InvariantError(f"{label}_nonpositive")
        out.append(float(value))
    return out


def _token(value: float) -> str:
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def _min_bankroll_bnb(*, trades_csv_path: Path) -> float:
    with Path(trades_csv_path).open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return 0.0
    return float(min(float(row["bankroll_bnb"]) for row in rows))


def _insufficient_bankroll_skips(*, summary: dict[str, object]) -> int:
    raw = dict(summary.get("num_skips_by_reason", {}))
    return int(raw.get("insufficient_bankroll_real", 0))


@contextmanager
def _execution_bet_override(*, execution_bet_bnb: float) -> Iterator[None]:
    import pancakebot.backtest.runner as backtest_runner
    import pancakebot.domain.strategy.dislocation_engine as dislocation_engine

    cap_bet_bnb = float(execution_bet_bnb)
    if float(cap_bet_bnb) <= 0.0:
        raise InvariantError("exec_override_bet_bnb_nonpositive")

    original_snapshot_key = backtest_runner._snapshot_key
    original_candidate_signal_raw = dislocation_engine.DislocationEngine.__dict__["_candidate_signal_from_decision"]
    original_candidate_signal_fn = dislocation_engine.DislocationEngine._candidate_signal_from_decision
    original_settle_candidate_decision = dislocation_engine.DislocationEngine._settle_candidate_decision
    original_shadow_profit_for_decision = dislocation_engine._shadow_profit_for_decision

    def _actual_bet_bnb(raw_bet_bnb: float) -> float:
        return float(max(0.0, min(float(raw_bet_bnb), float(cap_bet_bnb))))

    def _snapshot_key_ignore_initial_bankroll(
        *,
        runtime_cfg,
        backtest_cfg,
        reset_mode: str,
        warmup_rounds,
        sim_rounds,
        phase: str,
    ) -> str:
        cache_bt_cfg = replace(backtest_cfg, initial_bankroll_bnb=1.0)
        return original_snapshot_key(
            runtime_cfg=runtime_cfg,
            backtest_cfg=cache_bt_cfg,
            reset_mode=str(reset_mode),
            warmup_rounds=list(warmup_rounds),
            sim_rounds=list(sim_rounds),
            phase=str(phase),
        )

    def _candidate_signal_from_decision_with_exec_override(
        *,
        candidate_name: str,
        dec,
        selector_score_bnb: float | None,
    ):
        signal = original_candidate_signal_fn(
            candidate_name=str(candidate_name),
            dec=dec,
            selector_score_bnb=selector_score_bnb,
        )
        if str(signal.action) != "BET" or float(signal.bet_size_bnb) <= 0.0:
            return signal
        return replace(
            signal,
            bet_size_bnb=float(_actual_bet_bnb(float(signal.bet_size_bnb))),
        )

    def _settle_candidate_decision_with_exec_override(
        self,
        *,
        state,
        dec,
        round_t,
    ):
        effective_dec = dec
        if str(dec.action) == "BET" and float(dec.bet_bnb) > 0.0:
            effective_dec = replace(
                dec,
                bet_bnb=float(_actual_bet_bnb(float(dec.bet_bnb))),
            )
        return original_settle_candidate_decision(
            self,
            state=state,
            dec=effective_dec,
            round_t=round_t,
        )

    def _shadow_profit_for_decision_with_exec_override(
        *,
        round_t,
        dec,
        cfg,
        treasury_fee_fraction: float,
        bull_pool_cutoff_bnb: float | None,
        bear_pool_cutoff_bnb: float | None,
    ) -> tuple[float, bool, bool]:
        if dec.side is None:
            return 0.0, False, False
        if dec.p_nowcast_bull is None:
            return 0.0, False, False
        if bull_pool_cutoff_bnb is None or bear_pool_cutoff_bnb is None:
            return 0.0, False, False

        ev_bull_pool = (
            float(dec.ev_pool_bull_bnb)
            if dec.ev_pool_bull_bnb is not None and dislocation_engine.math.isfinite(float(dec.ev_pool_bull_bnb))
            else float(bull_pool_cutoff_bnb)
        )
        ev_bear_pool = (
            float(dec.ev_pool_bear_bnb)
            if dec.ev_pool_bear_bnb is not None and dislocation_engine.math.isfinite(float(dec.ev_pool_bear_bnb))
            else float(bear_pool_cutoff_bnb)
        )
        if float(ev_bull_pool) <= 0.0 or float(ev_bear_pool) <= 0.0:
            ev_bull_pool = float(bull_pool_cutoff_bnb)
            ev_bear_pool = float(bear_pool_cutoff_bnb)

        raw_bet_bnb = dislocation_engine._stake_bnb_for_decision(
            stake_mode=str(cfg.stake_mode),
            fixed_bet_bnb=float(cfg.fixed_bet_bnb),
            expected_net_selected=dec.expected_net_selected,
            stake_min_bnb=float(cfg.stake_min_bnb),
            stake_max_bnb=float(cfg.stake_max_bnb),
            stake_ev_ref_bnb=float(cfg.stake_ev_ref_bnb),
            side=str(dec.side),
            p_nowcast_bull=dec.p_nowcast_bull,
            bull_pool_cutoff_bnb=float(bull_pool_cutoff_bnb),
            bear_pool_cutoff_bnb=float(bear_pool_cutoff_bnb),
            bull_pool_ev_bnb=float(ev_bull_pool),
            bear_pool_ev_bnb=float(ev_bear_pool),
            treasury_fee_fraction=float(treasury_fee_fraction),
            stake_max_side_pool_frac=float(cfg.stake_max_side_pool_frac),
        )
        bet_bnb = float(_actual_bet_bnb(float(raw_bet_bnb)))
        if float(bet_bnb) <= 0.0:
            return 0.0, False, False

        selected_ev_actual = dislocation_engine._expected_net_from_cutoff(
            p_nowcast_bull=float(dec.p_nowcast_bull),
            bull_pool_cutoff_bnb=float(ev_bull_pool),
            bear_pool_cutoff_bnb=float(ev_bear_pool),
            side=str(dec.side),
            fixed_bet_bnb=float(bet_bnb),
            treasury_fee_fraction=float(treasury_fee_fraction),
        )
        expected_net_min_side = dislocation_engine._expected_net_min_for_side(cfg=cfg, side=str(dec.side))
        if float(selected_ev_actual) < float(expected_net_min_side):
            return 0.0, False, False

        total_cost = float(bet_bnb) + float(dislocation_engine.GAS_COST_BET_BNB)
        outcome = settle_bet_against_closed_round(
            bet_bnb=float(bet_bnb),
            bet_side=str(dec.side),
            round_closed=round_t,
            treasury_fee_fraction=float(treasury_fee_fraction),
        )
        profit = -float(total_cost) + float(outcome.credit_bnb)
        is_win = str(outcome.outcome) == "win"
        return float(profit), True, bool(is_win)

    backtest_runner._snapshot_key = _snapshot_key_ignore_initial_bankroll
    dislocation_engine.DislocationEngine._candidate_signal_from_decision = staticmethod(
        _candidate_signal_from_decision_with_exec_override
    )
    dislocation_engine.DislocationEngine._settle_candidate_decision = _settle_candidate_decision_with_exec_override
    dislocation_engine._shadow_profit_for_decision = _shadow_profit_for_decision_with_exec_override
    try:
        yield
    finally:
        backtest_runner._snapshot_key = original_snapshot_key
        dislocation_engine.DislocationEngine._candidate_signal_from_decision = original_candidate_signal_raw
        dislocation_engine.DislocationEngine._settle_candidate_decision = original_settle_candidate_decision
        dislocation_engine._shadow_profit_for_decision = original_shadow_profit_for_decision


def main() -> None:
    args = _build_parser().parse_args()
    cfg = load_cfg(config_path=str(args.config))
    exp_root = resolve_exp_root()
    exp_root.mkdir(parents=True, exist_ok=True)

    sim_size = int(cfg.backtest.simulation_size) if args.sim_size is None else int(args.sim_size)
    if int(sim_size) <= 0:
        raise InvariantError("exec_override_sim_size_nonpositive")

    top_skip_limit = int(args.top_skip_limit)
    if int(top_skip_limit) <= 0:
        raise InvariantError("exec_override_top_skip_limit_nonpositive")

    execution_stakes = _parse_positive_floats(
        raw=str(args.execution_stakes),
        label="exec_override_execution_stakes",
    )
    initial_bankrolls = _parse_positive_floats(
        raw=str(args.initial_bankrolls),
        label="exec_override_initial_bankrolls",
    )

    rows_raw: list[dict[str, object]] = []
    for execution_stake_bnb in execution_stakes:
        cfg_for_exec = replace(
            cfg,
            backtest_state_cache_dir=str(
                Path(str(cfg.backtest_state_cache_dir)) / f"exec_override_{_token(float(execution_stake_bnb))}"
            ),
        )
        with _execution_bet_override(execution_bet_bnb=float(execution_stake_bnb)):
            for initial_bankroll_bnb in initial_bankrolls:
                name = (
                    f"{str(args.name_prefix)}"
                    f"_exec{_token(float(execution_stake_bnb))}"
                    f"_bankroll{_token(float(initial_bankroll_bnb))}"
                )
                result = run_backtest_case(
                    cfg=cfg_for_exec,
                    name=str(name),
                    simulation_size=int(sim_size),
                    reset_mode=str(cfg.backtest.reset_mode),
                    reset_every_rounds=int(cfg.backtest.reset_every_rounds),
                    tail_offset_rounds=int(cfg.backtest.tail_offset_rounds),
                    initial_bankroll_bnb=float(initial_bankroll_bnb),
                    exp_root=exp_root,
                )
                summary = dict(result.summary)
                net_profit_bnb = float(summary.get("net_profit_bnb", 0.0))
                final_bankroll_bnb = float(summary.get("final_bankroll_bnb", 0.0))
                rows_raw.append(
                    {
                        "execution_stake_bnb": float(execution_stake_bnb),
                        "signal_fixed_stake_bnb": 0.3,
                        "initial_bankroll_bnb": float(initial_bankroll_bnb),
                        "final_bankroll_bnb": float(final_bankroll_bnb),
                        "net_profit_bnb": float(net_profit_bnb),
                        "roi_pct": (
                            100.0 * float(net_profit_bnb) / float(initial_bankroll_bnb)
                            if float(initial_bankroll_bnb) > 0.0
                            else 0.0
                        ),
                        "max_drawdown_bnb": float(max_drawdown_bnb(trades_csv_path=result.trades_path)),
                        "min_bankroll_bnb": float(_min_bankroll_bnb(trades_csv_path=result.trades_path)),
                        "num_bets": int(summary.get("num_bets", 0)),
                        "win_rate": float(summary.get("win_rate", 0.0)),
                        "bet_rate": float(summary.get("bet_rate", 0.0)),
                        "insufficient_bankroll_skips": int(_insufficient_bankroll_skips(summary=summary)),
                        "top_skip_reasons": str(top_skip_reasons(summary=summary, limit=int(top_skip_limit))),
                        "summary_path": str(result.summary_path),
                        "trades_path": str(result.trades_path),
                    }
                )

    rows_raw.sort(
        key=lambda row: (
            float(row["execution_stake_bnb"]),
            float(row["initial_bankroll_bnb"]),
        )
    )

    table_rows = [
        {
            "exec_stake": f"{float(row['execution_stake_bnb']):.3f}",
            "bankroll": f"{float(row['initial_bankroll_bnb']):.3f}",
            "net_profit_bnb": f"{float(row['net_profit_bnb']):.6f}",
            "final_bankroll": f"{float(row['final_bankroll_bnb']):.6f}",
            "roi_pct": f"{float(row['roi_pct']):.2f}",
            "max_dd": f"{float(row['max_drawdown_bnb']):.6f}",
            "min_bankroll": f"{float(row['min_bankroll_bnb']):.6f}",
            "num_bets": int(row["num_bets"]),
            "win_rate": f"{100.0 * float(row['win_rate']):.2f}%",
            "insuff_skips": int(row["insufficient_bankroll_skips"]),
        }
        for row in rows_raw
    ]

    print(f"EXP_ROOT={exp_root}")
    print("SEMANTICS=signal_generation_keeps_configured_fixed_stake; execution_and_settlement_use_exec_stake_cap")
    print(
        render_table(
            columns=[
                ("exec_stake", "exec_stake"),
                ("bankroll", "bankroll"),
                ("net_profit_bnb", "net_profit_bnb"),
                ("final_bankroll", "final_bankroll"),
                ("roi_pct", "roi_pct"),
                ("max_dd", "max_dd"),
                ("min_bankroll", "min_bankroll"),
                ("num_bets", "num_bets"),
                ("win_rate", "win_rate"),
                ("insuff_skips", "insuff_skips"),
            ],
            rows=table_rows,
        )
    )

    json_path = exp_root / f"{str(args.name_prefix)}_table.json"
    csv_path = exp_root / f"{str(args.name_prefix)}_table.csv"
    json_path.write_text(
        json.dumps(
            {
                "semantics": (
                    "signal_generation_keeps_configured_fixed_stake;"
                    " execution_and_settlement_use_exec_stake_cap"
                ),
                "rows": rows_raw,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "execution_stake_bnb",
                "signal_fixed_stake_bnb",
                "initial_bankroll_bnb",
                "final_bankroll_bnb",
                "net_profit_bnb",
                "roi_pct",
                "max_drawdown_bnb",
                "min_bankroll_bnb",
                "num_bets",
                "win_rate",
                "bet_rate",
                "insufficient_bankroll_skips",
                "top_skip_reasons",
                "summary_path",
                "trades_path",
            ],
        )
        writer.writeheader()
        for row in rows_raw:
            writer.writerow(row)

    print(f"TABLE_JSON={json_path}")
    print(f"TABLE_CSV={csv_path}")


if __name__ == "__main__":
    main()
