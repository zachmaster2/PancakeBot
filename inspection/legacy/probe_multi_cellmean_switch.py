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
    warmup_blocks: int
    num_quantile_bins: int
    min_cell_obs: int
    score_threshold: float
    use_direction_split: bool


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


def _parse_int_list(raw: str) -> list[int]:
    out: list[int] = []
    for tok in str(raw).split(","):
        t = str(tok).strip()
        if not t:
            continue
        out.append(int(t))
    return out


def _parse_float_list(raw: str) -> list[float]:
    out: list[float] = []
    for tok in str(raw).split(","):
        t = str(tok).strip()
        if not t:
            continue
        out.append(float(t))
    return out


def _parse_bool_list(raw: str) -> list[bool]:
    out: list[bool] = []
    for tok in str(raw).split(","):
        t = str(tok).strip().lower()
        if t in ("1", "true", "t", "yes", "y"):
            out.append(True)
        elif t in ("0", "false", "f", "no", "n"):
            out.append(False)
        else:
            raise ValueError(f"bool_list_token_invalid: {tok}")
    return out


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


def _load_blocks(
    *,
    strategy_prefixes: list[str],
    block_size: int,
    num_blocks: int,
    skip_most_recent_blocks: int,
) -> list[list[dict[str, TradeRow | None]]]:
    offsets = _offsets(
        block_size=int(block_size),
        num_blocks=int(num_blocks),
        skip_most_recent_blocks=int(skip_most_recent_blocks),
    )
    out_dir = Path("var/exp")
    blocks: list[list[dict[str, TradeRow | None]]] = []
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
                else:
                    ep_map[s] = None
            block_epochs.append(ep_map)
        blocks.append(block_epochs)
    return blocks


def _quantile_edges(values: list[float], n_bins: int) -> list[float]:
    if not values:
        return [0.0 for _ in range(int(n_bins) + 1)]
    vv = sorted(float(x) for x in values)
    out: list[float] = []
    for i in range(int(n_bins) + 1):
        q = float(i) / float(n_bins)
        idx = int(round((len(vv) - 1) * q))
        idx = max(0, min(len(vv) - 1, idx))
        out.append(float(vv[idx]))
    return out


def _bin_index(x: float, edges: list[float]) -> int:
    n_bins = int(len(edges) - 1)
    if int(n_bins) <= 1:
        return 0
    for i in range(int(n_bins)):
        lo = float(edges[i])
        hi = float(edges[i + 1])
        if float(x) >= float(lo) and (float(x) < float(hi) or int(i) == int(n_bins - 1)):
            return int(i)
    return int(n_bins - 1)


def _fit_edges(
    *,
    warmup_blocks: list[list[dict[str, TradeRow | None]]],
    strategy_prefixes: list[str],
    num_quantile_bins: int,
) -> dict[str, tuple[list[float], list[float]]]:
    out: dict[str, tuple[list[float], list[float]]] = {}
    for s in strategy_prefixes:
        ev_vals: list[float] = []
        dis_vals: list[float] = []
        for block in warmup_blocks:
            for ep_map in block:
                row = ep_map.get(s)
                if row is None:
                    continue
                ev_vals.append(float(row.ev_selected))
                dis_vals.append(float(row.abs_dislocation))
        ev_edges = _quantile_edges(ev_vals, int(num_quantile_bins))
        dis_edges = _quantile_edges(dis_vals, int(num_quantile_bins))
        out[s] = (ev_edges, dis_edges)
    return out


def _cell_key(
    *,
    row: TradeRow,
    ev_edges: list[float],
    dis_edges: list[float],
    use_direction_split: bool,
) -> tuple[int, int, int]:
    eb = _bin_index(float(row.ev_selected), ev_edges)
    db = _bin_index(float(row.abs_dislocation), dis_edges)
    sb = int(row.side_idx) if bool(use_direction_split) else 0
    return int(eb), int(db), int(sb)


def _eval_cfg(
    *,
    blocks: list[list[dict[str, TradeRow | None]]],
    strategy_prefixes: list[str],
    block_size: int,
    cfg: EvalConfig,
) -> dict[str, Any]:
    warmup = int(cfg.warmup_blocks)
    if int(warmup) <= 0:
        raise ValueError("warmup_blocks_must_be_positive")
    if int(warmup) >= int(len(blocks)):
        raise ValueError("warmup_blocks_must_be_less_than_num_blocks")

    warmup_blocks = blocks[:warmup]
    eval_blocks = blocks[warmup:]
    edges_by_strategy = _fit_edges(
        warmup_blocks=warmup_blocks,
        strategy_prefixes=strategy_prefixes,
        num_quantile_bins=int(cfg.num_quantile_bins),
    )

    sum_profit: dict[str, dict[tuple[int, int, int], float]] = {s: {} for s in strategy_prefixes}
    cnt_profit: dict[str, dict[tuple[int, int, int], int]] = {s: {} for s in strategy_prefixes}

    # Seed with warmup observations.
    for block in warmup_blocks:
        for ep_map in block:
            for s in strategy_prefixes:
                row = ep_map.get(s)
                if row is None:
                    continue
                ev_edges, dis_edges = edges_by_strategy[s]
                key = _cell_key(
                    row=row,
                    ev_edges=ev_edges,
                    dis_edges=dis_edges,
                    use_direction_split=bool(cfg.use_direction_split),
                )
                sum_profit[s][key] = float(sum_profit[s].get(key, 0.0) + float(row.profit_bnb))
                cnt_profit[s][key] = int(cnt_profit[s].get(key, 0) + 1)

    block_nets: list[float] = []
    bets_total = 0
    wins_total = 0
    picks_total: dict[str, int] = {s: 0 for s in strategy_prefixes}

    for block in eval_blocks:
        block_net = 0.0
        for ep_map in block:
            best_s: str | None = None
            best_score = float("-inf")
            best_profit = 0.0

            for s in strategy_prefixes:
                row = ep_map.get(s)
                if row is None:
                    continue
                ev_edges, dis_edges = edges_by_strategy[s]
                key = _cell_key(
                    row=row,
                    ev_edges=ev_edges,
                    dis_edges=dis_edges,
                    use_direction_split=bool(cfg.use_direction_split),
                )
                c = int(cnt_profit[s].get(key, 0))
                if int(c) < int(cfg.min_cell_obs):
                    continue
                est = float(sum_profit[s][key]) / float(c)
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

            # Online shadow updates after round closes.
            for s in strategy_prefixes:
                row = ep_map.get(s)
                if row is None:
                    continue
                ev_edges, dis_edges = edges_by_strategy[s]
                key = _cell_key(
                    row=row,
                    ev_edges=ev_edges,
                    dis_edges=dis_edges,
                    use_direction_split=bool(cfg.use_direction_split),
                )
                sum_profit[s][key] = float(sum_profit[s].get(key, 0.0) + float(row.profit_bnb))
                cnt_profit[s][key] = int(cnt_profit[s].get(key, 0) + 1)

        block_nets.append(float(block_net))

    eval_blocks_n = int(len(eval_blocks))
    eval_rounds = int(eval_blocks_n) * int(block_size)
    net_total = float(sum(block_nets))
    return {
        "net_per_500_eval": float(net_total / float(eval_rounds) * 500.0),
        "net_total": float(net_total),
        "net_median": float(statistics.median(block_nets)),
        "positive_block_frac": float(sum(1 for x in block_nets if float(x) > 0.0) / len(block_nets)),
        "bets_total": int(bets_total),
        "win_rate_weighted": float(float(wins_total) / float(bets_total)) if int(bets_total) > 0 else 0.0,
        "bet_rate": float(float(bets_total) / float(eval_rounds)),
        "eval_blocks": int(eval_blocks_n),
        "warmup_blocks": int(cfg.warmup_blocks),
        "picks_total": {str(k): int(v) for k, v in picks_total.items()},
    }


def _build_configs(args: argparse.Namespace) -> list[EvalConfig]:
    warmups = _parse_int_list(str(args.warmup_blocks_list))
    bins = _parse_int_list(str(args.num_quantile_bins_list))
    mins = _parse_int_list(str(args.min_cell_obs_list))
    thrs = _parse_float_list(str(args.score_threshold_list))
    dirs = _parse_bool_list(str(args.use_direction_split_list))

    out: list[EvalConfig] = []
    for w in warmups:
        for b in bins:
            for m in mins:
                for t in thrs:
                    for d in dirs:
                        out.append(
                            EvalConfig(
                                warmup_blocks=int(w),
                                num_quantile_bins=int(b),
                                min_cell_obs=int(m),
                                score_threshold=float(t),
                                use_direction_split=bool(d),
                            )
                        )
    return out


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--name-prefix", type=str, required=True)
    p.add_argument("--strategy-prefixes", type=str, required=True)
    p.add_argument("--block-size", type=int, default=500)
    p.add_argument("--num-blocks", type=int, default=60)
    p.add_argument("--skip-most-recent-blocks", type=int, default=0)
    p.add_argument("--warmup-blocks-list", type=str, default="10,15,20,25,30")
    p.add_argument("--num-quantile-bins-list", type=str, default="6,8,10,12")
    p.add_argument("--min-cell-obs-list", type=str, default="3,5,8,12")
    p.add_argument("--score-threshold-list", type=str, default="-0.02,-0.01,0.0,0.01,0.02,0.03,0.05")
    p.add_argument("--use-direction-split-list", type=str, default="1")
    p.add_argument("--max-candidates", type=int, default=0)
    p.add_argument("--seed", type=int, default=1337)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    random.seed(int(args.seed))

    strategy_prefixes = [str(x).strip() for x in str(args.strategy_prefixes).split(",") if str(x).strip()]
    if len(strategy_prefixes) < 2:
        raise ValueError("strategy_prefixes_requires_at_least_two")

    blocks = _load_blocks(
        strategy_prefixes=strategy_prefixes,
        block_size=int(args.block_size),
        num_blocks=int(args.num_blocks),
        skip_most_recent_blocks=int(args.skip_most_recent_blocks),
    )
    if int(len(blocks)) <= 1:
        raise ValueError("insufficient_blocks")

    cfgs = _build_configs(args)
    if int(args.max_candidates) > 0 and int(args.max_candidates) < len(cfgs):
        random.shuffle(cfgs)
        cfgs = cfgs[: int(args.max_candidates)]

    rows: list[dict[str, Any]] = []
    for i, cfg in enumerate(cfgs, start=1):
        if int(cfg.warmup_blocks) >= int(len(blocks)):
            continue
        m = _eval_cfg(
            blocks=blocks,
            strategy_prefixes=strategy_prefixes,
            block_size=int(args.block_size),
            cfg=cfg,
        )
        rows.append(
            {
                "cfg": {
                    "warmup_blocks": int(cfg.warmup_blocks),
                    "num_quantile_bins": int(cfg.num_quantile_bins),
                    "min_cell_obs": int(cfg.min_cell_obs),
                    "score_threshold": float(cfg.score_threshold),
                    "use_direction_split": bool(cfg.use_direction_split),
                },
                **m,
            }
        )
        if i % 25 == 0:
            best = max(rows, key=lambda x: float(x["net_per_500_eval"]))
            print(f"SEARCH_PROGRESS n={i} best={best['net_per_500_eval']}")

    if not rows:
        raise ValueError("no_valid_candidates")

    rows_sorted = sorted(rows, key=lambda x: float(x["net_per_500_eval"]), reverse=True)
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
                "results": rows_sorted,
            },
            indent=2,
            sort_keys=True,
        )
    )

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "net_per_500_eval",
            "net_total",
            "net_median",
            "positive_block_frac",
            "bets_total",
            "win_rate_weighted",
            "bet_rate",
            "eval_blocks",
            "warmup_blocks",
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
            + f"rank={i} net500={r['net_per_500_eval']} bets={r['bets_total']} "
            + f"cfg={r['cfg']}"
        )


if __name__ == "__main__":
    main()
