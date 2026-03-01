from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class TradeRow:
    action: str
    ev_selected: float
    abs_dislocation: float
    side_idx: int
    profit_bnb: float


@dataclass(frozen=True, slots=True)
class EvalConfig:
    num_bins_ev: int
    num_bins_dis: int
    prior_strength: float
    prior_mean: float
    use_global_prior: bool
    explore_coef: float
    score_threshold: float
    ev_low_q: float
    ev_high_q: float
    dis_low_q: float
    dis_high_q: float
    min_total_obs: int
    use_side_split: bool


def _safe_float(x: Any) -> float | None:
    if x is None:
        return None
    s = str(x).strip()
    if s == "":
        return None
    try:
        v = float(s)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return float(v)


def _quantile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    qq = max(0.0, min(1.0, float(q)))
    idx = int(round((len(sorted_vals) - 1) * qq))
    return float(sorted_vals[idx])


def _offsets(*, block_size: int, num_blocks: int, skip_most_recent_blocks: int) -> list[int]:
    return [
        int(block_size) * i
        for i in range(int(num_blocks) + int(skip_most_recent_blocks) - 1, int(skip_most_recent_blocks) - 1, -1)
    ]


def _scenario_name(prefix: str, idx: int, num_blocks: int, offset: int) -> str:
    return f"{prefix}_b{idx}of{num_blocks}_off{offset}"


def _parse_side_idx(direction: str) -> int:
    d = str(direction).strip().upper()
    if d == "BULL":
        return 0
    if d == "BEAR":
        return 1
    return 0


def _load_trades(path: Path) -> dict[int, TradeRow]:
    if not path.exists():
        raise FileNotFoundError(f"missing_trades_csv: {path}")
    out: dict[int, TradeRow] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for raw in rd:
            ep = int(raw["epoch"])
            action = str(raw.get("action", "")).strip().upper()
            if action not in ("BET", "SKIP"):
                raise ValueError("trade_row_action_invalid")
            ev = _safe_float(raw.get("expected_net_selected"))
            dis = _safe_float(raw.get("dislocation_bull"))
            profit = _safe_float(raw.get("profit_bnb"))
            if profit is None:
                raise ValueError("trade_row_profit_missing")
            out[ep] = TradeRow(
                action=str(action),
                ev_selected=float(ev) if ev is not None else 0.0,
                abs_dislocation=abs(float(dis)) if dis is not None else 0.0,
                side_idx=int(_parse_side_idx(str(raw.get("direction", "")))),
                profit_bnb=float(profit),
            )
    if not out:
        raise ValueError(f"empty_trades_csv: {path}")
    return out


def _load_dataset(
    *,
    strategy_prefixes: list[str],
    block_size: int,
    num_blocks: int,
    skip_most_recent_blocks: int,
) -> tuple[list[list[dict[str, TradeRow | None]]], list[float], list[float]]:
    offsets = _offsets(
        block_size=int(block_size),
        num_blocks=int(num_blocks),
        skip_most_recent_blocks=int(skip_most_recent_blocks),
    )
    out_dir = Path("var/exp")
    blocks: list[list[dict[str, TradeRow | None]]] = []
    all_ev: list[float] = []
    all_abs_dis: list[float] = []

    for block_idx, offset in enumerate(offsets, start=1):
        rows_by_strategy: dict[str, dict[int, TradeRow]] = {}
        epochs: set[int] = set()
        for s in strategy_prefixes:
            name = _scenario_name(str(s), int(block_idx), int(num_blocks), int(offset))
            rows = _load_trades(out_dir / name / "dislocation_trades.csv")
            rows_by_strategy[str(s)] = rows
            epochs |= set(rows.keys())

        block_epochs: list[dict[str, TradeRow | None]] = []
        for ep in sorted(epochs):
            ep_map: dict[str, TradeRow | None] = {}
            for s in strategy_prefixes:
                r = rows_by_strategy[s].get(int(ep))
                if r is not None and str(r.action) == "BET":
                    ep_map[s] = r
                    all_ev.append(float(r.ev_selected))
                    all_abs_dis.append(float(r.abs_dislocation))
                else:
                    ep_map[s] = None
            block_epochs.append(ep_map)
        blocks.append(block_epochs)

    return blocks, all_ev, all_abs_dis


def _bin_index(x: float, x_min: float, x_max: float, n_bins: int) -> int:
    if int(n_bins) <= 1 or float(x_max) <= float(x_min):
        return 0
    t = (float(x) - float(x_min)) / (float(x_max) - float(x_min))
    t = max(0.0, min(0.999999, float(t)))
    return int(float(t) * int(n_bins))


def _flat_idx(*, side_idx: int, ev_bin: int, dis_bin: int, num_bins_ev: int, num_bins_dis: int) -> int:
    return int(side_idx) * int(num_bins_ev) * int(num_bins_dis) + int(ev_bin) * int(num_bins_dis) + int(dis_bin)


def _eval_cfg(
    *,
    blocks: list[list[dict[str, TradeRow | None]]],
    strategy_prefixes: list[str],
    block_size: int,
    num_blocks: int,
    ev_min: float,
    ev_max: float,
    dis_min: float,
    dis_max: float,
    cfg: EvalConfig,
) -> dict[str, Any]:
    n_ev = int(cfg.num_bins_ev)
    n_dis = int(cfg.num_bins_dis)
    side_mult = 2 if bool(cfg.use_side_split) else 1
    vec_size = int(side_mult) * int(n_ev) * int(n_dis)

    sum_profit: dict[str, list[float]] = {s: [0.0 for _ in range(vec_size)] for s in strategy_prefixes}
    cnt_profit: dict[str, list[int]] = {s: [0 for _ in range(vec_size)] for s in strategy_prefixes}

    total_obs = 0
    global_sum = 0.0
    global_cnt = 0

    block_nets: list[float] = []
    bets_total = 0
    wins_total = 0
    picks_total: dict[str, int] = {s: 0 for s in strategy_prefixes}

    for block_epochs in blocks:
        block_net = 0.0
        for ep_map in block_epochs:
            best_s: str | None = None
            best_score = float("-inf")
            best_profit = 0.0

            if int(total_obs) >= int(cfg.min_total_obs):
                for s in strategy_prefixes:
                    row = ep_map.get(s)
                    if row is None:
                        continue
                    ev_bin = _bin_index(float(row.ev_selected), float(ev_min), float(ev_max), int(n_ev))
                    dis_bin = _bin_index(float(row.abs_dislocation), float(dis_min), float(dis_max), int(n_dis))
                    side_idx = int(row.side_idx) if bool(cfg.use_side_split) else 0
                    idx = _flat_idx(
                        side_idx=int(side_idx),
                        ev_bin=int(ev_bin),
                        dis_bin=int(dis_bin),
                        num_bins_ev=int(n_ev),
                        num_bins_dis=int(n_dis),
                    )

                    cnt = int(cnt_profit[s][idx])
                    sm = float(sum_profit[s][idx])
                    if bool(cfg.use_global_prior) and int(global_cnt) > 0:
                        prior_mean = float(global_sum) / float(global_cnt)
                    else:
                        prior_mean = float(cfg.prior_mean)
                    est = (float(sm) + float(cfg.prior_strength) * float(prior_mean)) / (
                        float(cnt) + float(cfg.prior_strength)
                    )
                    est += float(cfg.explore_coef) / math.sqrt(float(cnt) + 1.0)
                    if float(est) < float(cfg.score_threshold):
                        continue

                    if float(est) > float(best_score):
                        best_score = float(est)
                        best_s = str(s)
                        best_profit = float(row.profit_bnb)

            if best_s is not None:
                block_net += float(best_profit)
                bets_total += 1
                if float(best_profit) > 0.0:
                    wins_total += 1
                picks_total[best_s] += 1

            for s in strategy_prefixes:
                row = ep_map.get(s)
                if row is None:
                    continue
                ev_bin = _bin_index(float(row.ev_selected), float(ev_min), float(ev_max), int(n_ev))
                dis_bin = _bin_index(float(row.abs_dislocation), float(dis_min), float(dis_max), int(n_dis))
                side_idx = int(row.side_idx) if bool(cfg.use_side_split) else 0
                idx = _flat_idx(
                    side_idx=int(side_idx),
                    ev_bin=int(ev_bin),
                    dis_bin=int(dis_bin),
                    num_bins_ev=int(n_ev),
                    num_bins_dis=int(n_dis),
                )
                p = float(row.profit_bnb)
                sum_profit[s][idx] += float(p)
                cnt_profit[s][idx] += 1
                total_obs += 1
                global_sum += float(p)
                global_cnt += 1

        block_nets.append(float(block_net))

    total_rounds = int(block_size) * int(num_blocks)
    net_total = float(sum(block_nets))
    return {
        "net_per_500": float(net_total / float(total_rounds) * 500.0),
        "net_total": float(net_total),
        "net_median": float(statistics.median(block_nets)),
        "positive_block_frac": float(sum(1 for x in block_nets if float(x) > 0.0) / len(block_nets)),
        "bets_total": int(bets_total),
        "win_rate_weighted": float(float(wins_total) / float(bets_total)) if int(bets_total) > 0 else 0.0,
        "bet_rate": float(float(bets_total) / float(total_rounds)),
        "picks_total": {str(k): int(v) for k, v in picks_total.items()},
    }


def _run(args: argparse.Namespace) -> None:
    random.seed(int(args.seed))
    strategy_prefixes = [str(x).strip() for x in str(args.strategy_prefixes).split(",") if str(x).strip()]
    if len(strategy_prefixes) < 2:
        raise ValueError("strategy_prefixes_requires_at_least_two")

    blocks, all_ev, all_abs_dis = _load_dataset(
        strategy_prefixes=strategy_prefixes,
        block_size=int(args.block_size),
        num_blocks=int(args.num_blocks),
        skip_most_recent_blocks=int(args.skip_most_recent_blocks),
    )
    if not all_ev:
        raise ValueError("no_ev_values_found")
    if not all_abs_dis:
        raise ValueError("no_abs_dis_values_found")
    ev_sorted = sorted(float(x) for x in all_ev)
    dis_sorted = sorted(float(x) for x in all_abs_dis)

    num_bins_ev_vals = [4, 6, 8, 12, 16]
    num_bins_dis_vals = [4, 6, 8, 12, 16]
    prior_strength_vals = [5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0]
    prior_mean_vals = [-0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02]
    use_global_prior_vals = [False, True]
    explore_vals = [0.0, 0.002, 0.005, 0.01, 0.02, 0.05]
    score_thr_vals = [-0.05, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.05]
    ev_low_q_vals = [0.0, 0.01, 0.05]
    ev_high_q_vals = [0.95, 0.99, 1.0]
    dis_low_q_vals = [0.0, 0.01, 0.05]
    dis_high_q_vals = [0.95, 0.99, 1.0]
    min_obs_vals = [0, 20, 50, 100, 200, 400]
    side_split_vals = [False, True]

    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    while len(rows) < int(args.num_samples):
        cfg = EvalConfig(
            num_bins_ev=int(random.choice(num_bins_ev_vals)),
            num_bins_dis=int(random.choice(num_bins_dis_vals)),
            prior_strength=float(random.choice(prior_strength_vals)),
            prior_mean=float(random.choice(prior_mean_vals)),
            use_global_prior=bool(random.choice(use_global_prior_vals)),
            explore_coef=float(random.choice(explore_vals)),
            score_threshold=float(random.choice(score_thr_vals)),
            ev_low_q=float(random.choice(ev_low_q_vals)),
            ev_high_q=float(random.choice(ev_high_q_vals)),
            dis_low_q=float(random.choice(dis_low_q_vals)),
            dis_high_q=float(random.choice(dis_high_q_vals)),
            min_total_obs=int(random.choice(min_obs_vals)),
            use_side_split=bool(random.choice(side_split_vals)),
        )
        if float(cfg.ev_low_q) >= float(cfg.ev_high_q):
            continue
        if float(cfg.dis_low_q) >= float(cfg.dis_high_q):
            continue

        ev_min = _quantile(ev_sorted, float(cfg.ev_low_q))
        ev_max = _quantile(ev_sorted, float(cfg.ev_high_q))
        dis_min = _quantile(dis_sorted, float(cfg.dis_low_q))
        dis_max = _quantile(dis_sorted, float(cfg.dis_high_q))
        if float(ev_max) <= float(ev_min):
            continue
        if float(dis_max) <= float(dis_min):
            continue

        key = "|".join(
            str(x)
            for x in (
                cfg.num_bins_ev,
                cfg.num_bins_dis,
                cfg.prior_strength,
                cfg.prior_mean,
                cfg.use_global_prior,
                cfg.explore_coef,
                cfg.score_threshold,
                cfg.ev_low_q,
                cfg.ev_high_q,
                cfg.dis_low_q,
                cfg.dis_high_q,
                cfg.min_total_obs,
                cfg.use_side_split,
            )
        )
        if key in seen:
            continue
        seen.add(key)

        m = _eval_cfg(
            blocks=blocks,
            strategy_prefixes=strategy_prefixes,
            block_size=int(args.block_size),
            num_blocks=int(args.num_blocks),
            ev_min=float(ev_min),
            ev_max=float(ev_max),
            dis_min=float(dis_min),
            dis_max=float(dis_max),
            cfg=cfg,
        )
        rows.append(
            {
                "cfg": {
                    "num_bins_ev": int(cfg.num_bins_ev),
                    "num_bins_dis": int(cfg.num_bins_dis),
                    "prior_strength": float(cfg.prior_strength),
                    "prior_mean": float(cfg.prior_mean),
                    "use_global_prior": bool(cfg.use_global_prior),
                    "explore_coef": float(cfg.explore_coef),
                    "score_threshold": float(cfg.score_threshold),
                    "ev_low_q": float(cfg.ev_low_q),
                    "ev_high_q": float(cfg.ev_high_q),
                    "dis_low_q": float(cfg.dis_low_q),
                    "dis_high_q": float(cfg.dis_high_q),
                    "min_total_obs": int(cfg.min_total_obs),
                    "use_side_split": bool(cfg.use_side_split),
                    "ev_min": float(ev_min),
                    "ev_max": float(ev_max),
                    "dis_min": float(dis_min),
                    "dis_max": float(dis_max),
                },
                **m,
            }
        )
        if len(rows) % 100 == 0:
            best = max(rows, key=lambda x: float(x["net_per_500"]))
            print(f"SEARCH_PROGRESS n={len(rows)} best={best['net_per_500']}")

    rows_sorted = sorted(rows, key=lambda x: float(x["net_per_500"]), reverse=True)
    out_dir = Path("var/exp")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{args.name_prefix}.json"
    csv_path = out_dir / f"{args.name_prefix}.csv"

    json_path.write_text(
        json.dumps(
            {
                "name_prefix": str(args.name_prefix),
                "strategy_prefixes": strategy_prefixes,
                "block_size": int(args.block_size),
                "num_blocks": int(args.num_blocks),
                "skip_most_recent_blocks": int(args.skip_most_recent_blocks),
                "num_samples": int(args.num_samples),
                "results": rows_sorted,
            },
            indent=2,
            sort_keys=True,
        )
    )

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "net_per_500",
            "net_total",
            "net_median",
            "positive_block_frac",
            "bets_total",
            "win_rate_weighted",
            "bet_rate",
            "cfg",
            "picks_total",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows_sorted:
            w.writerow(r)

    print(f"CSV={csv_path}")
    print(f"JSON={json_path}")
    for i, r in enumerate(rows_sorted[:20], start=1):
        print(
            "TOP "
            + f"rank={i} net500={r['net_per_500']} bets={r['bets_total']} "
            + f"cfg={r['cfg']}"
        )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--name-prefix", type=str, required=True)
    p.add_argument("--strategy-prefixes", type=str, required=True)
    p.add_argument("--block-size", type=int, default=500)
    p.add_argument("--num-blocks", type=int, default=40)
    p.add_argument("--skip-most-recent-blocks", type=int, default=0)
    p.add_argument("--num-samples", type=int, default=1500)
    p.add_argument("--seed", type=int, default=1337)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    _run(args)


if __name__ == "__main__":
    main()
