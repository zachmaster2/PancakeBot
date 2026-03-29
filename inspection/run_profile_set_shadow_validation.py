from __future__ import annotations

import argparse
import csv
from collections import Counter
import json
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recommendation-json", type=str, required=True)
    parser.add_argument("--cycle-audit-csv", type=str, default="var/runtime/dry_cycle_audit.csv")
    parser.add_argument("--bankroll-json", type=str, default="var/runtime/dry_bankroll_state.json")
    parser.add_argument("--audit-trades-csv", type=str, default="var/runtime/dry_audit_trades.csv")
    parser.add_argument("--name-prefix", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="../PancakeBot_var_exp")
    parser.add_argument("--recent-cycles", type=int, default=12)
    return parser


def _load_cycle_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _load_trade_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _safe_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _recent_summary(rows: list[dict[str, str]], recent_cycles: int) -> dict[str, object]:
    recent = rows[-int(recent_cycles) :] if recent_cycles > 0 else list(rows)
    actions = Counter(str(row.get("action", "")) for row in recent)
    skips = Counter(str(row.get("skip_reason", "")) for row in recent if str(row.get("action", "")) == "SKIP")
    strategies = Counter(str(row.get("selected_strategy", "")) for row in recent if str(row.get("selected_strategy", "")) != "")
    return {
        "recent_cycle_count": int(len(recent)),
        "recent_action_counts": dict(sorted(actions.items())),
        "recent_skip_reason_counts": dict(sorted(skips.items())),
        "recent_strategy_counts": dict(sorted(strategies.items())),
        "recent_bet_count": int(actions.get("BET", 0)),
        "recent_skip_count": int(actions.get("SKIP", 0)),
    }


def _coherence_status(*, recommendation: dict[str, object], cycle_rows: list[dict[str, str]], recent_cycles: int) -> tuple[str, str]:
    chosen_profile = str(recommendation.get("chosen_profile", ""))
    if not cycle_rows:
        return "unknown", "no_dry_cycles"
    recent = cycle_rows[-int(recent_cycles) :] if int(recent_cycles) > 0 else list(cycle_rows)
    recent_bets = sum(1 for row in recent if str(row.get("action", "")) == "BET")
    if str(chosen_profile) == "skip":
        if int(recent_bets) == 0:
            return "coherent", "shadow_skip_and_recent_dry_is_all_skip"
        return "mixed", "shadow_skip_but_recent_dry_has_bets"
    if str(chosen_profile) == "stageb":
        return "coherent", "shadow_stageb_matches_contained_runtime"
    return "divergent_by_design", "shadow_alt_profile_cannot_match_contained_stageb_runtime"


def main() -> None:
    args = _build_parser().parse_args()
    output_dir = Path(str(args.output_dir)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    recommendation_path = Path(str(args.recommendation_json)).resolve()
    cycle_audit_path = Path(str(args.cycle_audit_csv)).resolve()
    bankroll_path = Path(str(args.bankroll_json)).resolve()
    audit_trades_path = Path(str(args.audit_trades_csv)).resolve()

    recommendation = _safe_json(recommendation_path)
    cycle_rows = _load_cycle_rows(cycle_audit_path)
    trade_rows = _load_trade_rows(audit_trades_path)
    bankroll_state = _safe_json(bankroll_path)
    recent_summary = _recent_summary(cycle_rows, int(args.recent_cycles))
    coherence, coherence_reason = _coherence_status(
        recommendation=recommendation,
        cycle_rows=cycle_rows,
        recent_cycles=int(args.recent_cycles),
    )

    overall_actions = Counter(str(row.get("action", "")) for row in cycle_rows)
    overall_skips = Counter(
        str(row.get("skip_reason", ""))
        for row in cycle_rows
        if str(row.get("action", "")) == "SKIP"
    )
    overall_strategies = Counter(
        str(row.get("selected_strategy", ""))
        for row in cycle_rows
        if str(row.get("selected_strategy", "")) != ""
    )
    settled_trade_count = sum(1 for row in trade_rows if str(row.get("trade_status", "")).strip() != "")

    summary = {
        "recommendation_json": str(recommendation_path),
        "cycle_audit_csv": str(cycle_audit_path),
        "bankroll_json": str(bankroll_path),
        "audit_trades_csv": str(audit_trades_path),
        "shadow_chosen_profile": str(recommendation.get("chosen_profile", "")),
        "shadow_chosen_predicted_per_500": float(recommendation.get("chosen_predicted_per_500", 0.0)),
        "shadow_estimated_selected_bet_rate": float(recommendation.get("estimated_selected_bet_rate", 0.0)),
        "coherence_status": str(coherence),
        "coherence_reason": str(coherence_reason),
        "cycle_count": int(len(cycle_rows)),
        "action_counts": dict(sorted(overall_actions.items())),
        "skip_reason_counts": dict(sorted(overall_skips.items())),
        "strategy_counts": dict(sorted(overall_strategies.items())),
        "recent_summary": recent_summary,
        "settled_trade_count": int(settled_trade_count),
        "bankroll_state": bankroll_state,
        "latest_cycle": (cycle_rows[-1] if cycle_rows else {}),
    }
    out_path = output_dir / f"{args.name_prefix}_profile_set_shadow_validation_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
