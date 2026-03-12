from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

from pancakebot.core.errors import InvariantError

from inspection.backtest_harness_common import render_table, resolve_exp_root


@dataclass(frozen=True, slots=True)
class CaseSpec:
    variant: str
    trades_path: Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--name-prefix", type=str, default="negative_period_ceiling_20260306")
    p.add_argument("--matrix-csv", type=str, required=True)
    p.add_argument(
        "--variants",
        type=str,
        default="baseline_modelgate_selector_p0p5,baseline_core_online",
    )
    p.add_argument("--skip-budgets", type=str, default="0.01,0.02,0.05,0.10,0.20,0.30")
    p.add_argument("--window-sizes", type=str, default="250,500,1000")
    p.add_argument("--initial-bankroll-bnb", type=float, default=50.0)
    return p


def _parse_float_list(raw: str) -> list[float]:
    out: list[float] = []
    for token in [x.strip() for x in str(raw).split(",") if x.strip() != ""]:
        try:
            out.append(float(token))
        except ValueError as e:
            raise InvariantError(f"negative_ceiling_float_list_invalid: {token}") from e
    out = sorted(set(out))
    if not out:
        raise InvariantError("negative_ceiling_float_list_empty")
    return out


def _parse_int_list(raw: str) -> list[int]:
    out: list[int] = []
    for token in [x.strip() for x in str(raw).split(",") if x.strip() != ""]:
        try:
            out.append(int(token))
        except ValueError as e:
            raise InvariantError(f"negative_ceiling_int_list_invalid: {token}") from e
    out = sorted(set(out))
    if not out:
        raise InvariantError("negative_ceiling_int_list_empty")
    if any(int(x) <= 0 for x in out):
        raise InvariantError("negative_ceiling_int_list_nonpositive")
    return out


def _parse_name_list(raw: str) -> list[str]:
    out = [str(x).strip() for x in str(raw).split(",") if str(x).strip() != ""]
    if not out:
        raise InvariantError("negative_ceiling_variants_empty")
    return out


def _read_cases(*, matrix_csv: Path, variants: list[str]) -> list[CaseSpec]:
    rows: dict[str, dict[str, str]] = {}
    with Path(matrix_csv).open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = str(row.get("variant", "")).strip()
            if key == "" or key in rows:
                continue
            rows[key] = dict(row)

    out: list[CaseSpec] = []
    for variant in variants:
        row = rows.get(str(variant))
        if row is None:
            raise InvariantError(f"negative_ceiling_variant_missing: {variant}")
        trades_raw = str(row.get("trades_path", "")).strip()
        if trades_raw == "":
            raise InvariantError(f"negative_ceiling_trades_path_missing: {variant}")
        trades_path = Path(trades_raw)
        if not trades_path.exists():
            raise InvariantError(f"negative_ceiling_trades_path_not_found: {trades_path}")
        out.append(CaseSpec(variant=str(variant), trades_path=Path(trades_path)))
    return out


def _load_profit_series(*, trades_path: Path) -> tuple[list[float], int]:
    profits: list[float] = []
    num_bets = 0
    with Path(trades_path).open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            profits.append(float(row.get("profit_bnb", 0.0)))
            if str(row.get("action", "")).strip() == "BET":
                num_bets += 1
    if not profits:
        raise InvariantError("negative_ceiling_empty_trades")
    return profits, int(num_bets)


def _per_500(*, profits: list[float]) -> float:
    return float(sum(profits)) * 500.0 / float(len(profits))


def _max_drawdown_bnb(*, profits: list[float], initial_bankroll_bnb: float) -> float:
    equity = float(initial_bankroll_bnb)
    peak = float(initial_bankroll_bnb)
    max_dd = 0.0
    for p in profits:
        equity += float(p)
        if float(equity) > float(peak):
            peak = float(equity)
        dd = float(peak) - float(equity)
        if float(dd) > float(max_dd):
            max_dd = float(dd)
    return float(max_dd)


def _oracle_individual_skip(*, profits: list[float], skip_budget: float) -> tuple[list[float], int]:
    out = list(profits)
    n = len(out)
    k = int(math.floor(float(skip_budget) * float(n)))
    if int(k) <= 0:
        return out, 0
    ranked = sorted(range(n), key=lambda idx: float(out[idx]))
    changed = 0
    for idx in ranked[: int(k)]:
        if float(out[idx]) < 0.0:
            out[idx] = 0.0
            changed += 1
    return out, int(changed)


def _oracle_window_skip(
    *,
    profits: list[float],
    skip_budget: float,
    window_size: int,
) -> tuple[list[float], int, int]:
    out = list(profits)
    n = len(out)
    max_skip_rounds = int(math.floor(float(skip_budget) * float(n)))
    if int(max_skip_rounds) < int(window_size):
        return out, 0, 0

    prefix = [0.0]
    for p in out:
        prefix.append(float(prefix[-1]) + float(p))
    sums: list[tuple[float, int]] = []
    for start in range(0, n - int(window_size) + 1):
        total = float(prefix[start + int(window_size)] - prefix[start])
        sums.append((float(total), int(start)))
    sums.sort(key=lambda x: float(x[0]))

    used = [False] * n
    skipped_rounds = 0
    blocks = 0
    for total, start in sums:
        if float(total) >= 0.0:
            break
        if int(skipped_rounds) + int(window_size) > int(max_skip_rounds):
            continue
        end = int(start) + int(window_size)
        if any(used[start:end]):
            continue
        for idx in range(start, end):
            used[idx] = True
        skipped_rounds += int(window_size)
        blocks += 1
        if int(skipped_rounds) >= int(max_skip_rounds):
            break

    changed = 0
    for idx, flag in enumerate(used):
        if flag and float(out[idx]) < 0.0:
            out[idx] = 0.0
            changed += 1
    return out, int(skipped_rounds), int(blocks)


def main() -> None:
    args = _build_parser().parse_args()
    exp_root = resolve_exp_root()
    exp_root.mkdir(parents=True, exist_ok=True)

    budgets = _parse_float_list(str(args.skip_budgets))
    if any(not (0.0 <= float(x) <= 1.0) for x in budgets):
        raise InvariantError("negative_ceiling_skip_budget_out_of_range")
    windows = _parse_int_list(str(args.window_sizes))
    variants = _parse_name_list(str(args.variants))
    matrix_csv = Path(str(args.matrix_csv))
    if not matrix_csv.exists():
        raise InvariantError(f"negative_ceiling_matrix_missing: {matrix_csv}")
    if float(args.initial_bankroll_bnb) <= 0.0:
        raise InvariantError("negative_ceiling_initial_bankroll_nonpositive")

    cases = _read_cases(matrix_csv=matrix_csv, variants=variants)

    rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for case in cases:
        profits, num_bets = _load_profit_series(trades_path=case.trades_path)
        base_per_500 = _per_500(profits=profits)
        base_max_dd = _max_drawdown_bnb(
            profits=profits,
            initial_bankroll_bnb=float(args.initial_bankroll_bnb),
        )
        rounds = len(profits)

        best_window_rows: list[dict[str, object]] = []
        best_individual: dict[str, object] | None = None
        for budget in budgets:
            ind_profits, ind_changed = _oracle_individual_skip(profits=profits, skip_budget=float(budget))
            ind_row = {
                "variant": str(case.variant),
                "method": "oracle_individual_loss_skip",
                "window_size": None,
                "skip_budget": float(budget),
                "rounds": int(rounds),
                "num_bets": int(num_bets),
                "changed_negative_rounds": int(ind_changed),
                "skipped_rounds": int(math.floor(float(budget) * float(rounds))),
                "per_500": _per_500(profits=ind_profits),
                "max_drawdown_bnb": _max_drawdown_bnb(
                    profits=ind_profits,
                    initial_bankroll_bnb=float(args.initial_bankroll_bnb),
                ),
                "base_per_500": float(base_per_500),
                "base_max_drawdown_bnb": float(base_max_dd),
                "trades_path": str(case.trades_path),
            }
            rows.append(ind_row)
            if best_individual is None or float(ind_row["per_500"]) > float(best_individual["per_500"]):
                best_individual = dict(ind_row)

            for window_size in windows:
                win_profits, win_skipped, win_blocks = _oracle_window_skip(
                    profits=profits,
                    skip_budget=float(budget),
                    window_size=int(window_size),
                )
                win_row = {
                    "variant": str(case.variant),
                    "method": "oracle_contiguous_window_skip",
                    "window_size": int(window_size),
                    "skip_budget": float(budget),
                    "rounds": int(rounds),
                    "num_bets": int(num_bets),
                    "changed_negative_rounds": None,
                    "skipped_rounds": int(win_skipped),
                    "window_blocks": int(win_blocks),
                    "per_500": _per_500(profits=win_profits),
                    "max_drawdown_bnb": _max_drawdown_bnb(
                        profits=win_profits,
                        initial_bankroll_bnb=float(args.initial_bankroll_bnb),
                    ),
                    "base_per_500": float(base_per_500),
                    "base_max_drawdown_bnb": float(base_max_dd),
                    "trades_path": str(case.trades_path),
                }
                rows.append(win_row)
                best_window_rows.append(dict(win_row))

        top_window = max(best_window_rows, key=lambda r: float(r["per_500"])) if best_window_rows else None
        summary_rows.append(
            {
                "variant": str(case.variant),
                "base_per_500": float(base_per_500),
                "base_max_dd": float(base_max_dd),
                "best_individual_per_500": (
                    None if best_individual is None else float(best_individual["per_500"])
                ),
                "best_window_per_500": (
                    None if top_window is None else float(top_window["per_500"])
                ),
                "best_window_size": (
                    None if top_window is None else int(top_window["window_size"])
                ),
                "best_window_budget": (
                    None if top_window is None else float(top_window["skip_budget"])
                ),
            }
        )

    out_json = exp_root / f"{str(args.name_prefix)}.json"
    out_csv = exp_root / f"{str(args.name_prefix)}.csv"
    out_json.write_text(
        json.dumps(
            {
                "matrix_csv": str(matrix_csv),
                "variants": [str(v) for v in variants],
                "skip_budgets": [float(x) for x in budgets],
                "window_sizes": [int(x) for x in windows],
                "initial_bankroll_bnb": float(args.initial_bankroll_bnb),
                "summary_rows": summary_rows,
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "variant",
            "method",
            "window_size",
            "skip_budget",
            "rounds",
            "num_bets",
            "changed_negative_rounds",
            "skipped_rounds",
            "window_blocks",
            "per_500",
            "max_drawdown_bnb",
            "base_per_500",
            "base_max_drawdown_bnb",
            "trades_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    table_rows = []
    for row in summary_rows:
        table_rows.append(
            {
                "variant": str(row["variant"]),
                "base_p500": f"{float(row['base_per_500']):+.6f}",
                "base_dd": f"{float(row['base_max_dd']):.6f}",
                "best_ind_p500": (
                    "" if row["best_individual_per_500"] is None else f"{float(row['best_individual_per_500']):+.6f}"
                ),
                "best_win_p500": (
                    "" if row["best_window_per_500"] is None else f"{float(row['best_window_per_500']):+.6f}"
                ),
                "best_win_w": ("" if row["best_window_size"] is None else int(row["best_window_size"])),
                "best_win_budget": (
                    ""
                    if row["best_window_budget"] is None
                    else f"{100.0 * float(row['best_window_budget']):.0f}%"
                ),
            }
        )

    print(
        render_table(
            columns=[
                ("variant", "variant"),
                ("base_p500", "base_p500"),
                ("base_dd", "base_dd"),
                ("best_ind_p500", "best_ind_p500"),
                ("best_win_p500", "best_win_p500"),
                ("best_win_w", "best_win_w"),
                ("best_win_budget", "best_win_budget"),
            ],
            rows=table_rows,
        )
    )
    print(f"TABLE_JSON={out_json}")
    print(f"TABLE_CSV={out_csv}")


if __name__ == "__main__":
    main()
