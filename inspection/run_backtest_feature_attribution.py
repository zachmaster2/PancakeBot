from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from pancakebot.core.errors import InvariantError
from pancakebot.domain.features.feature_builder import build_features, vectorize
from pancakebot.domain.features.schema import FEATURE_SCHEMA, max_required_prior_context_rounds_size
from pancakebot.domain.types import Round
from pancakebot.infra.closed_rounds_store import ClosedRoundsStore
from pancakebot.infra.feature_cache_store import FeatureCacheStore
from pancakebot.infra.market_data_db import MarketDataDb, SqliteKlinesStore

from inspection.backtest_harness_common import load_cfg, resolve_exp_root


_DEFAULT_FEATURES = (
    "regime_bull_frac_r_20",
    "regime_flip_rate_r_20",
    "regime_bull_frac_r_60",
    "regime_flip_rate_r_60",
    "regime_streak_len",
    "late_log_imb",
    "log_imb_w_p_0_to_p_100",
    "log_imb_w_p_80_to_p_100",
    "bet_count_w_p_80_to_p_100",
    "bet_top1_share_w_p_80_to_p_100",
    "total_sum_w_p_0_to_p_100",
    "price_log_return_mean_k_15",
    "price_log_return_abs_mean_k_15",
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config.toml")
    p.add_argument("--name-prefix", type=str, required=True)
    p.add_argument("--trades-csv", type=str, required=True)
    p.add_argument("--feature-deciles", type=int, default=10)
    p.add_argument("--features", type=str, default=",".join(_DEFAULT_FEATURES))
    p.add_argument("--top-limit", type=int, default=8)
    return p


def _parse_feature_list(raw: str) -> list[str]:
    out = [str(x).strip() for x in str(raw).split(",") if str(x).strip()]
    if not out:
        raise InvariantError("feature_attribution_features_empty")
    known = set(FEATURE_SCHEMA.columns)
    for name in out:
        if str(name) not in known:
            raise InvariantError(f"feature_attribution_feature_unknown: {name}")
    return list(out)


def _safe_float(raw: Any, *, default: float = 0.0) -> float:
    text = str(raw).strip()
    if text == "":
        return float(default)
    return float(text)


def _load_trade_rows(*, trades_csv_path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with Path(trades_csv_path).open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out.append(
                {
                    "epoch": int(row["epoch"]),
                    "action": str(row.get("action", "")).strip(),
                    "skip_reason": str(row.get("skip_reason", "")).strip(),
                    "direction": str(row.get("direction", "")).strip(),
                    "selected_strategy": str(row.get("selected_strategy", "")).strip(),
                    "profit_bnb": float(row["profit_bnb"]),
                    "ev_bnb": _safe_float(row.get("ev_bnb", ""), default=0.0),
                    "bet_size_bnb": _safe_float(row.get("bet_size_bnb", ""), default=0.0),
                    "selector_score_bnb": _safe_float(row.get("selector_score_bnb", ""), default=float("nan")),
                }
            )
    if not out:
        raise InvariantError("feature_attribution_trades_empty")
    return out


def _anchor_close_time_ms(*, klines_store: SqliteKlinesStore, round_t: Round, cutoff_seconds: int) -> int:
    if round_t.lock_at is None:
        raise InvariantError("feature_attribution_round_lock_at_missing")
    anchor_ms = (int(round_t.lock_at) - int(cutoff_seconds)) * 1000
    latest_close_ms = klines_store.latest_close_time_ms()
    if latest_close_ms is None:
        raise InvariantError("feature_attribution_klines_empty")
    if int(latest_close_ms) < int(anchor_ms):
        anchor_ms = int(latest_close_ms)
    return int(anchor_ms)


def _feature_map_for_round(
    *,
    round_t: Round,
    prior_context_rounds: list[Round],
    klines_store: SqliteKlinesStore,
    feature_cache_store: FeatureCacheStore | None,
    cutoff_seconds: int,
) -> dict[str, float]:
    if not prior_context_rounds:
        raise InvariantError("feature_attribution_prior_context_empty")

    prior_last_epoch = int(prior_context_rounds[-1].epoch)
    anchor_close_time_ms = _anchor_close_time_ms(
        klines_store=klines_store,
        round_t=round_t,
        cutoff_seconds=int(cutoff_seconds),
    )

    vector: list[float] | None = None
    if feature_cache_store is not None:
        vector = feature_cache_store.get_vector(
            epoch=int(round_t.epoch),
            cutoff_seconds=int(cutoff_seconds),
            schema_name=str(FEATURE_SCHEMA.name),
            start_at=int(round_t.start_at),
            lock_at=int(round_t.lock_at),
            prior_last_epoch=int(prior_last_epoch),
            anchor_close_time_ms=int(anchor_close_time_ms),
        )
    if vector is None:
        context_klines = klines_store.get_context_klines(
            anchor_close_time_ms=int(anchor_close_time_ms),
            size=int(FEATURE_SCHEMA.required_context_klines_size),
        )
        features = build_features(
            target_round=round_t,
            prior_context_rounds=list(prior_context_rounds),
            context_klines=list(context_klines),
            cutoff_seconds=int(cutoff_seconds),
        )
        vector = vectorize(features=features, schema=FEATURE_SCHEMA)
        if feature_cache_store is not None:
            feature_cache_store.put_vector(
                epoch=int(round_t.epoch),
                cutoff_seconds=int(cutoff_seconds),
                schema_name=str(FEATURE_SCHEMA.name),
                start_at=int(round_t.start_at),
                lock_at=int(round_t.lock_at),
                prior_last_epoch=int(prior_last_epoch),
                anchor_close_time_ms=int(anchor_close_time_ms),
                vector=list(vector),
            )
    return {str(col): float(vector[idx]) for idx, col in enumerate(FEATURE_SCHEMA.columns)}


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = int(len(rows))
    if n <= 0:
        return {
            "num_rows": 0,
            "num_bets": 0,
            "bet_rate": 0.0,
            "net_profit_bnb": 0.0,
            "profit_per_500_rounds_bnb": 0.0,
            "avg_profit_per_bet_bnb": 0.0,
            "gross_profit_bnb": 0.0,
            "gross_loss_bnb": 0.0,
            "win_rate": 0.0,
            "avg_ev_bnb_on_bets": 0.0,
        }

    bet_rows = [r for r in rows if str(r.get("action", "")) == "BET"]
    num_bets = int(len(bet_rows))
    net_profit = float(sum(float(r.get("profit_bnb", 0.0)) for r in rows))
    gross_profit = float(sum(max(0.0, float(r.get("profit_bnb", 0.0))) for r in rows))
    gross_loss = float(sum(max(0.0, -float(r.get("profit_bnb", 0.0))) for r in rows))
    wins = int(sum(1 for r in bet_rows if float(r.get("profit_bnb", 0.0)) > 0.0))
    avg_ev = (
        float(sum(float(r.get("ev_bnb", 0.0)) for r in bet_rows)) / float(num_bets)
        if int(num_bets) > 0
        else 0.0
    )
    return {
        "num_rows": int(n),
        "num_bets": int(num_bets),
        "bet_rate": float(num_bets) / float(n),
        "net_profit_bnb": float(net_profit),
        "profit_per_500_rounds_bnb": float(net_profit) * 500.0 / float(n),
        "avg_profit_per_bet_bnb": (float(net_profit) / float(num_bets) if int(num_bets) > 0 else 0.0),
        "gross_profit_bnb": float(gross_profit),
        "gross_loss_bnb": float(gross_loss),
        "win_rate": (float(wins) / float(num_bets) if int(num_bets) > 0 else 0.0),
        "avg_ev_bnb_on_bets": float(avg_ev),
    }


def _group_summaries(*, rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        label = str(row.get(key, "")).strip()
        if label == "":
            label = "unknown"
        groups.setdefault(label, []).append(row)

    out: list[dict[str, Any]] = []
    for label, group_rows in groups.items():
        summary = _summarize_rows(list(group_rows))
        summary[str(key)] = str(label)
        out.append(summary)
    out.sort(
        key=lambda r: (
            -float(r["net_profit_bnb"]),
            -float(r["profit_per_500_rounds_bnb"]),
            -int(r["num_bets"]),
            str(r[key]),
        )
    )
    return out


def _decile_summaries(
    *,
    rows: list[dict[str, Any]],
    feature: str,
    buckets: int,
) -> list[dict[str, Any]]:
    if int(buckets) <= 0:
        raise InvariantError("feature_attribution_buckets_nonpositive")

    finite_rows = [r for r in rows if math.isfinite(float(r[feature]))]
    finite_rows.sort(key=lambda r: float(r[feature]))
    out: list[dict[str, Any]] = []

    if finite_rows:
        n = int(len(finite_rows))
        for idx in range(int(buckets)):
            lo = (idx * n) // int(buckets)
            hi = ((idx + 1) * n) // int(buckets)
            if int(hi) <= int(lo):
                continue
            bucket_rows = finite_rows[int(lo): int(hi)]
            summary = _summarize_rows(list(bucket_rows))
            vals = [float(r[feature]) for r in bucket_rows]
            summary.update(
                {
                    "feature": str(feature),
                    "bucket_kind": "decile",
                    "bucket_index": int(idx) + 1,
                    "feature_min": float(vals[0]),
                    "feature_max": float(vals[-1]),
                    "feature_mean": float(sum(vals) / len(vals)),
                }
            )
            out.append(summary)

    non_finite_rows = [r for r in rows if not math.isfinite(float(r[feature]))]
    if non_finite_rows:
        summary = _summarize_rows(list(non_finite_rows))
        summary.update(
            {
                "feature": str(feature),
                "bucket_kind": "non_finite",
                "bucket_index": 0,
                "feature_min": float("nan"),
                "feature_max": float("nan"),
                "feature_mean": float("nan"),
            }
        )
        out.append(summary)

    return out


def _flatten_feature_tables(
    *,
    feature_tables: dict[str, dict[str, list[dict[str, Any]]]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for feature, populations in feature_tables.items():
        for population, rows in populations.items():
            for row in rows:
                flat = dict(row)
                flat["feature"] = str(feature)
                flat["population"] = str(population)
                out.append(flat)
    return out


def _top_and_bottom_rows(
    *,
    rows: list[dict[str, Any]],
    top_limit: int,
) -> dict[str, list[dict[str, Any]]]:
    ranked = sorted(
        rows,
        key=lambda r: (
            -float(r["profit_per_500_rounds_bnb"]),
            -float(r["net_profit_bnb"]),
            -int(r["num_bets"]),
            str(r.get("feature", "")),
            int(r.get("bucket_index", 0)),
        ),
    )
    worst = sorted(
        rows,
        key=lambda r: (
            float(r["profit_per_500_rounds_bnb"]),
            float(r["net_profit_bnb"]),
            -int(r["num_bets"]),
            str(r.get("feature", "")),
            int(r.get("bucket_index", 0)),
        ),
    )
    return {
        "best": list(ranked[: int(top_limit)]),
        "worst": list(worst[: int(top_limit)]),
    }


def main() -> None:
    args = _build_parser().parse_args()
    if int(args.feature_deciles) <= 0:
        raise InvariantError("feature_attribution_feature_deciles_nonpositive")
    if int(args.top_limit) <= 0:
        raise InvariantError("feature_attribution_top_limit_nonpositive")

    cfg = load_cfg(config_path=str(args.config))
    features_to_analyze = _parse_feature_list(str(args.features))
    trades_csv_path = Path(str(args.trades_csv))
    trade_rows = _load_trade_rows(trades_csv_path=trades_csv_path)

    closed_rounds = list(ClosedRoundsStore(str(cfg.closed_rounds_path)).iter_closed_rounds())
    if not closed_rounds:
        raise InvariantError("feature_attribution_closed_rounds_empty")
    round_index_by_epoch = {int(r.epoch): idx for idx, r in enumerate(closed_rounds)}

    market_data_db = MarketDataDb(str(cfg.market_data_db_path))
    market_data_db.ensure_sources_synced(
        rounds_jsonl_path=str(cfg.closed_rounds_path),
        klines_jsonl_path=str(cfg.klines_path),
    )
    klines_store = SqliteKlinesStore(market_data_db=market_data_db)
    feature_cache_store = FeatureCacheStore(str(cfg.feature_cache_path))

    try:
        prior_needed = int(max_required_prior_context_rounds_size())
        enriched_rows: list[dict[str, Any]] = []
        for idx, row in enumerate(trade_rows, start=1):
            epoch = int(row["epoch"])
            round_idx = round_index_by_epoch.get(int(epoch))
            if round_idx is None:
                raise InvariantError(f"feature_attribution_epoch_missing_from_round_store: {epoch}")
            if int(round_idx) < int(prior_needed):
                raise InvariantError(f"feature_attribution_epoch_has_insufficient_history: {epoch}")
            round_t = closed_rounds[int(round_idx)]
            prior_context_rounds = closed_rounds[int(round_idx) - int(prior_needed): int(round_idx)]
            feat_map = _feature_map_for_round(
                round_t=round_t,
                prior_context_rounds=list(prior_context_rounds),
                klines_store=klines_store,
                feature_cache_store=feature_cache_store,
                cutoff_seconds=int(cfg.cutoff_seconds),
            )
            merged = dict(row)
            for feature in features_to_analyze:
                merged[str(feature)] = float(feat_map[str(feature)])
            enriched_rows.append(merged)
            if int(idx) % 5000 == 0:
                print(f"progress rows={idx}/{len(trade_rows)}")

        bet_rows = [r for r in enriched_rows if str(r["action"]) == "BET"]
        selected_strategy_rows = [r for r in bet_rows if str(r["selected_strategy"]) != ""]

        feature_tables: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for feature in features_to_analyze:
            feature_tables[str(feature)] = {
                "all_rounds": _decile_summaries(
                    rows=list(enriched_rows),
                    feature=str(feature),
                    buckets=int(args.feature_deciles),
                ),
                "bet_rounds": _decile_summaries(
                    rows=list(bet_rows),
                    feature=str(feature),
                    buckets=int(args.feature_deciles),
                ),
            }

        output_dir = resolve_exp_root() / str(args.name_prefix)
        output_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "name_prefix": str(args.name_prefix),
            "trades_csv_path": str(trades_csv_path),
            "num_trade_rows": int(len(enriched_rows)),
            "num_bet_rows": int(len(bet_rows)),
            "features": list(features_to_analyze),
            "overall": _summarize_rows(list(enriched_rows)),
            "bets_only": _summarize_rows(list(bet_rows)),
            "by_selected_strategy": _group_summaries(rows=list(selected_strategy_rows), key="selected_strategy"),
            "by_direction": _group_summaries(rows=list(bet_rows), key="direction"),
            "by_strategy_side": _group_summaries(
                rows=[
                    dict(r, strategy_side=f"{str(r['selected_strategy'])}|{str(r['direction'])}")
                    for r in selected_strategy_rows
                ],
                key="strategy_side",
            ),
            "feature_deciles": feature_tables,
        }

        summary_path = Path(output_dir) / "feature_attribution_summary.json"
        summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=True), encoding="utf-8")

        feature_rows = _flatten_feature_tables(feature_tables=feature_tables)
        feature_csv_path = Path(output_dir) / "feature_attribution_deciles.csv"
        with feature_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "feature",
                    "population",
                    "bucket_kind",
                    "bucket_index",
                    "feature_min",
                    "feature_max",
                    "feature_mean",
                    "num_rows",
                    "num_bets",
                    "bet_rate",
                    "net_profit_bnb",
                    "profit_per_500_rounds_bnb",
                    "avg_profit_per_bet_bnb",
                    "gross_profit_bnb",
                    "gross_loss_bnb",
                    "win_rate",
                    "avg_ev_bnb_on_bets",
                ],
            )
            writer.writeheader()
            for row in feature_rows:
                writer.writerow(row)

        strategy_csv_path = Path(output_dir) / "feature_attribution_strategies.csv"
        with strategy_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "selected_strategy",
                    "num_rows",
                    "num_bets",
                    "bet_rate",
                    "net_profit_bnb",
                    "profit_per_500_rounds_bnb",
                    "avg_profit_per_bet_bnb",
                    "gross_profit_bnb",
                    "gross_loss_bnb",
                    "win_rate",
                    "avg_ev_bnb_on_bets",
                ],
            )
            writer.writeheader()
            for row in payload["by_selected_strategy"]:
                writer.writerow(row)

        best_worst = _top_and_bottom_rows(rows=list(feature_rows), top_limit=int(args.top_limit))
        print(f"SUMMARY_JSON={summary_path}")
        print(f"DECILES_CSV={feature_csv_path}")
        print(f"STRATEGIES_CSV={strategy_csv_path}")
        print(json.dumps({"overall": payload["overall"], "bets_only": payload["bets_only"]}, indent=2, sort_keys=True))
        print(json.dumps(best_worst, indent=2, sort_keys=True, allow_nan=True))
    finally:
        feature_cache_store.close()
        market_data_db.close()


if __name__ == "__main__":
    main()
