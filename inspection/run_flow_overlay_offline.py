from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from pancakebot.core.errors import InvariantError

_DEFAULT_EXP_ROOT = "../PancakeBot_var_exp"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--primary-trades", type=str, required=True)
    parser.add_argument("--overlay-trades", type=str, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        choices=("primary_only", "fallback_only", "margin_override", "max_effective_score"),
        default="margin_override",
    )
    parser.add_argument("--overlay-score-penalty-bnb", type=float, default=0.0)
    parser.add_argument("--override-margin-bnb", type=float, default=0.0)
    return parser


def _read_trades(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_float(row: dict[str, str], key: str) -> float:
    raw = str(row.get(key, "")).strip()
    if raw == "":
        return 0.0
    return float(raw)


def _epoch_map(rows: list[dict[str, str]]) -> dict[int, dict[str, str]]:
    out: dict[int, dict[str, str]] = {}
    for row in rows:
        epoch = int(row["epoch"])
        out[int(epoch)] = row
    return out


def _effective_overlay_score(*, row: dict[str, str], score_penalty_bnb: float) -> float:
    if str(row.get("action", "")) != "BET":
        return float("-inf")
    return _to_float(row, "selector_score_bnb") - float(score_penalty_bnb)


def _score(row: dict[str, str]) -> float:
    if str(row.get("action", "")) != "BET":
        return float("-inf")
    return _to_float(row, "selector_score_bnb")


def _select_row(
    *,
    primary_row: dict[str, str],
    overlay_row: dict[str, str],
    mode: str,
    overlay_score_penalty_bnb: float,
    override_margin_bnb: float,
) -> tuple[str, dict[str, str]]:
    primary_action = str(primary_row.get("action", ""))
    overlay_action = str(overlay_row.get("action", ""))
    primary_score = _score(primary_row)
    overlay_score = _effective_overlay_score(
        row=overlay_row,
        score_penalty_bnb=float(overlay_score_penalty_bnb),
    )

    if str(mode) == "primary_only":
        return "primary", primary_row

    if str(mode) == "fallback_only":
        if primary_action == "BET":
            return "primary", primary_row
        if overlay_action == "BET" and float(overlay_score) > 0.0:
            return "overlay", overlay_row
        return "skip", primary_row

    if str(mode) == "margin_override":
        if primary_action == "BET":
            if overlay_action == "BET" and float(overlay_score) > float(primary_score) + float(
                override_margin_bnb
            ):
                return "overlay", overlay_row
            return "primary", primary_row
        if overlay_action == "BET" and float(overlay_score) > 0.0:
            return "overlay", overlay_row
        return "skip", primary_row

    if str(mode) == "max_effective_score":
        if primary_action != "BET" and not (overlay_action == "BET" and float(overlay_score) > 0.0):
            return "skip", primary_row
        if overlay_action == "BET" and float(overlay_score) > max(0.0, float(primary_score)):
            return "overlay", overlay_row
        if primary_action == "BET":
            return "primary", primary_row
        return "skip", primary_row

    raise InvariantError("flow_overlay_mode_invalid")


def main() -> None:
    args = _build_parser().parse_args()
    if float(args.overlay_score_penalty_bnb) < 0.0:
        raise InvariantError("flow_overlay_score_penalty_negative")
    if float(args.override_margin_bnb) < 0.0:
        raise InvariantError("flow_overlay_override_margin_negative")

    exp_root = Path(os.environ.get("PANCAKEBOT_EXP_DIR", _DEFAULT_EXP_ROOT))
    out_dir = exp_root / str(args.name)
    out_dir.mkdir(parents=True, exist_ok=True)

    primary_rows = _read_trades(Path(str(args.primary_trades)))
    overlay_rows = _read_trades(Path(str(args.overlay_trades)))
    primary_by_epoch = _epoch_map(primary_rows)
    overlay_by_epoch = _epoch_map(overlay_rows)
    epochs = sorted(set(primary_by_epoch.keys()) & set(overlay_by_epoch.keys()))
    if not epochs:
        raise InvariantError("flow_overlay_no_shared_epochs")

    selected_rows: list[dict[str, object]] = []
    bankroll_bnb = 50.0
    net_profit_bnb = 0.0
    num_bets = 0
    pick_counts = {"primary": 0, "overlay": 0, "skip": 0}

    for epoch in epochs:
        primary_row = primary_by_epoch[int(epoch)]
        overlay_row = overlay_by_epoch[int(epoch)]
        pick_source, chosen = _select_row(
            primary_row=primary_row,
            overlay_row=overlay_row,
            mode=str(args.mode),
            overlay_score_penalty_bnb=float(args.overlay_score_penalty_bnb),
            override_margin_bnb=float(args.override_margin_bnb),
        )
        profit_bnb = 0.0
        action = "SKIP"
        selected_strategy = ""
        direction = ""
        bet_size_bnb = 0.0
        selector_score_bnb = 0.0
        skip_reason = "overlay_skip"
        if str(pick_source) != "skip" and str(chosen.get("action", "")) == "BET":
            action = "BET"
            profit_bnb = _to_float(chosen, "profit_bnb")
            selected_strategy = str(chosen.get("selected_strategy", ""))
            direction = str(chosen.get("direction", ""))
            bet_size_bnb = _to_float(chosen, "bet_size_bnb")
            skip_reason = ""
            if str(pick_source) == "overlay":
                selector_score_bnb = _effective_overlay_score(
                    row=chosen,
                    score_penalty_bnb=float(args.overlay_score_penalty_bnb),
                )
            else:
                selector_score_bnb = _score(chosen)
            num_bets += 1
        bankroll_bnb += float(profit_bnb)
        net_profit_bnb += float(profit_bnb)
        pick_counts[str(pick_source)] = int(pick_counts.get(str(pick_source), 0) + 1)
        selected_rows.append(
            {
                "epoch": int(epoch),
                "pick_source": str(pick_source),
                "action": str(action),
                "skip_reason": str(skip_reason),
                "direction": str(direction),
                "bet_size_bnb": float(bet_size_bnb),
                "profit_bnb": float(profit_bnb),
                "bankroll_bnb": float(bankroll_bnb),
                "selected_strategy": str(selected_strategy),
                "selector_score_bnb": float(selector_score_bnb),
                "primary_action": str(primary_row.get("action", "")),
                "primary_selected_strategy": str(primary_row.get("selected_strategy", "")),
                "primary_selector_score_bnb": float(_score(primary_row))
                if str(primary_row.get("action", "")) == "BET"
                else "",
                "overlay_action": str(overlay_row.get("action", "")),
                "overlay_selected_strategy": str(overlay_row.get("selected_strategy", "")),
                "overlay_selector_score_bnb": float(_score(overlay_row))
                if str(overlay_row.get("action", "")) == "BET"
                else "",
                "overlay_effective_score_bnb": float(
                    _effective_overlay_score(
                        row=overlay_row,
                        score_penalty_bnb=float(args.overlay_score_penalty_bnb),
                    )
                )
                if str(overlay_row.get("action", "")) == "BET"
                else "",
            }
        )

    trades_path = out_dir / "offline_overlay_trades.csv"
    with trades_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(selected_rows[0].keys()))
        writer.writeheader()
        writer.writerows(selected_rows)

    summary = {
        "name": str(args.name),
        "mode": str(args.mode),
        "primary_trades": str(args.primary_trades),
        "overlay_trades": str(args.overlay_trades),
        "overlay_score_penalty_bnb": float(args.overlay_score_penalty_bnb),
        "override_margin_bnb": float(args.override_margin_bnb),
        "rounds": int(len(epochs)),
        "net_profit_bnb": float(net_profit_bnb),
        "net_profit_per_500_rounds": float(net_profit_bnb) * 500.0 / float(len(epochs)),
        "num_bets": int(num_bets),
        "bet_rate": float(num_bets) / float(len(epochs)),
        "pick_counts": {str(k): int(v) for k, v in pick_counts.items()},
        "final_bankroll_bnb": float(bankroll_bnb),
    }
    summary_path = out_dir / "offline_overlay_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"SUMMARY={summary_path}")
    print(f"TRADES={trades_path}")


if __name__ == "__main__":
    main()
