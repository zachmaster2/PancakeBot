from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

FULL = "__FULL__"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sf(v, d=0.0) -> float:
    try:
        x = float(v)
    except Exception:
        return float(d)
    return float(d) if x != x else float(x)


def si(v, d=0) -> int:
    try:
        return int(v)
    except Exception:
        return int(d)


def parse_offsets_csv(raw: str) -> list[int]:
    tokens = [t.strip() for t in str(raw).split(",") if str(t).strip()]
    if not tokens:
        return [0]
    out: list[int] = []
    seen: set[int] = set()
    for t in tokens:
        try:
            v = int(t)
        except Exception as e:
            raise ValueError(f"offset_not_int: {t}") from e
        if int(v) < 0:
            raise ValueError("offset_must_be_nonnegative")
        if int(v) in seen:
            continue
        seen.add(int(v))
        out.append(int(v))
    if not out:
        return [0]
    return out


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config.toml")
    p.add_argument("--state-dir", type=str, default="var/exp/_autonomy")
    p.add_argument("--name-prefix", type=str, default="auto")
    p.add_argument("--sim-size", type=int, default=300)
    p.add_argument("--max-runs", type=int, default=24)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--search-profile", type=str, choices=("legacy", "robust_v1"), default="robust_v1")
    p.add_argument("--offsets-csv", type=str, default="0")
    p.add_argument("--target-net-per-500-bnb", type=float, default=2.0)
    p.add_argument("--fixed-bet-override-bnb", type=float, default=None)
    return p


def loadj(path: Path, default: dict) -> dict:
    if not path.exists():
        return dict(default)
    try:
        obj = json.loads(path.read_text())
    except Exception:
        return dict(default)
    return obj if isinstance(obj, dict) else dict(default)


def writej(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def append_csv(path: Path, row: dict, fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k) for k in fields})


def candidates(sim_size: int, fixed_override: float | None, seed: int, search_profile: str) -> list[dict]:
    profile = str(search_profile)
    if profile == "legacy":
        feats = [
            ("full", FULL),
            ("llimb", "late_log_imb"),
            ("llimb_ltotal", "late_log_imb,late_total_sum"),
            ("step6", "late_log_imb,late_total_sum,bull_sum_ratio_w_p_50_to_p_100_over_w_p_0_to_p_50,regime_bull_frac_r_20,regime_flip_rate_r_20,regime_streak_len"),
            ("regime3", "late_log_imb,regime_flip_rate_r_20,regime_streak_len"),
            ("price15", "late_log_imb,price_log_return_mean_k_15,price_log_return_std_k_15"),
        ]
        model_cal = [("lgbm", "raw"), ("lgbm", "platt"), ("logistic", "raw")]
        trains = [(8000, 4000, 0.1, 2.0), (20000, 15000, 0.6, 1.2)]
        bull_thresholds = (0.515, 0.519, 0.523, 0.527)
        quantile_specs = ((0.03, 0.01, 0.001), (0.04, 0.015, 0.002), (0.05, 0.02, 0.003))
        threshold_windows = (300,)
    elif profile == "robust_v1":
        feats = [
            ("llimb", "late_log_imb"),
            ("llimb_ltotal", "late_log_imb,late_total_sum"),
            ("step6", "late_log_imb,late_total_sum,bull_sum_ratio_w_p_50_to_p_100_over_w_p_0_to_p_50,regime_bull_frac_r_20,regime_flip_rate_r_20,regime_streak_len"),
            ("regime3", "late_log_imb,regime_flip_rate_r_20,regime_streak_len"),
            ("price15", "late_log_imb,price_log_return_mean_k_15,price_log_return_std_k_15"),
        ]
        model_cal = [("lgbm", "raw"), ("logistic", "raw")]
        trains = [
            (8000, 4000, 0.1, 2.0),
            (20000, 15000, 0.6, 1.2),
            (35000, 20000, 0.6, 1.2),
        ]
        bull_thresholds = (0.509, 0.512, 0.515, 0.519, 0.523)
        quantile_specs = (
            (0.02, 0.005, 0.0),
            (0.025, 0.0075, 0.0005),
            (0.03, 0.01, 0.001),
            (0.035, 0.0125, 0.0015),
            (0.04, 0.015, 0.002),
            (0.05, 0.02, 0.003),
            (0.06, 0.03, 0.004),
        )
        threshold_windows = (200, 300, 500)
    else:
        raise ValueError("unknown_search_profile")

    out: list[dict] = []
    for ft, cols in feats:
        for dm, cal in model_cal:
            for train_size, calibrate_size, rw_floor, rw_power in trains:
                for b in bull_thresholds:
                    for mode in ("policy", "fixed"):
                        fixed = None if mode == "policy" else 0.05
                        if fixed_override is not None:
                            fixed = float(fixed_override)
                        out.append(
                            {
                                "tag": f"{ft}_{dm}_{cal}_b{int(round(b*1000))}_{mode}_t{train_size//1000}k",
                                "feature": cols,
                                "direction_model_type": dm,
                                "calibration_mode": cal,
                                "train_size": train_size,
                                "calibrate_size": calibrate_size,
                                "rw_floor": rw_floor,
                                "rw_power": rw_power,
                                "direction_filter_mode": "bull_only",
                                "direction_threshold_mode": "fixed",
                                "direction_threshold_bull": b,
                                "direction_threshold_bear": 0.5,
                                "direction_target_bull_rate": 0.03,
                                "direction_target_bear_rate": 0.03,
                                "direction_threshold_window": 300,
                                "direction_center_mode": "rolling_median",
                                "direction_center_window": 300,
                                "direction_edge_floor_pp": 0.0,
                                "fixed_bet_bnb": fixed,
                            }
                        )
                for target_bull, target_bear, edge_pp in quantile_specs:
                    for tw in threshold_windows:
                        for mode in ("policy", "fixed"):
                            fixed = None if mode == "policy" else 0.05
                            if fixed_override is not None:
                                fixed = float(fixed_override)
                            out.append(
                                {
                                    "tag": (
                                        f"{ft}_{dm}_{cal}_q{int(target_bull*1000)}_{int(target_bear*1000)}_"
                                        f"e{int(edge_pp*10000)}_w{int(tw)}_{mode}_t{train_size//1000}k"
                                    ),
                                    "feature": cols,
                                    "direction_model_type": dm,
                                    "calibration_mode": cal,
                                    "train_size": train_size,
                                    "calibrate_size": calibrate_size,
                                    "rw_floor": rw_floor,
                                    "rw_power": rw_power,
                                    "direction_filter_mode": "both_sides",
                                    "direction_threshold_mode": "quantile",
                                    "direction_threshold_bull": 0.5,
                                    "direction_threshold_bear": 0.5,
                                    "direction_target_bull_rate": target_bull,
                                    "direction_target_bear_rate": target_bear,
                                    "direction_threshold_window": int(tw),
                                    "direction_center_mode": "rolling_median",
                                    "direction_center_window": max(300, int(tw)),
                                    "direction_edge_floor_pp": edge_pp,
                                    "fixed_bet_bnb": fixed,
                                }
                            )
    random.Random(int(seed) + int(sim_size)).shuffle(out)
    return out


def run_one(config: str, name: str, sim_size: int, sim_offset_rounds: int, c: dict) -> tuple[int, str]:
    cmd = [
        sys.executable, "-m", "inspection.run_backtest_scenario",
        "--config", config, "--name", name,
        "--train-size", str(int(c["train_size"])),
        "--calibrate-size", str(int(c["calibrate_size"])),
        "--rw-floor", str(float(c["rw_floor"])),
        "--rw-power", str(float(c["rw_power"])),
        "--sim-size", str(int(sim_size)),
        "--sim-offset-rounds", str(int(sim_offset_rounds)),
        "--direction-model-type", str(c["direction_model_type"]),
        "--calibration-mode", str(c["calibration_mode"]),
        "--window-order", "cal_train",
        "--direction-filter-mode", str(c["direction_filter_mode"]),
        "--direction-threshold-mode", str(c["direction_threshold_mode"]),
        "--direction-threshold-bull", str(float(c["direction_threshold_bull"])),
        "--direction-threshold-bear", str(float(c["direction_threshold_bear"])),
        "--direction-target-bull-rate", str(float(c["direction_target_bull_rate"])),
        "--direction-target-bear-rate", str(float(c["direction_target_bear_rate"])),
        "--direction-threshold-window", str(int(c["direction_threshold_window"])),
        "--direction-center-mode", str(c["direction_center_mode"]),
        "--direction-center-window", str(int(c["direction_center_window"])),
        "--direction-edge-floor-pp", str(float(c["direction_edge_floor_pp"])),
    ]
    if str(c["feature"]).strip() != FULL:
        cmd.extend(["--sparse-probe-columns", str(c["feature"])])
    if c.get("fixed_bet_bnb") is not None:
        cmd.extend(["--fixed-bet-bnb", str(float(c["fixed_bet_bnb"]))])
    p = subprocess.run(cmd, capture_output=True, text=True, check=False)
    msg = (p.stderr or p.stdout or "").strip()
    return int(p.returncode), msg[:1200]


def _run_score(*, pp500: float, net: float, bet_rate: float, bets: int, max_drawdown_bnb: float) -> float:
    return float(pp500) - 0.25 * float(max_drawdown_bnb) - (0.2 if float(net) <= 0.0 else 0.0) - (0.2 if float(bet_rate) < 0.03 else 0.0) - max(0, 20 - int(bets)) * 0.01


def _aggregate_candidate(rows_ok: list[dict], *, offsets_total: int) -> dict:
    if not rows_ok:
        return {
            "candidate_offsets_ok": 0,
            "candidate_offsets_total": int(offsets_total),
            "candidate_pp500_mean": "",
            "candidate_pp500_median": "",
            "candidate_pp500_worst": "",
            "candidate_net_total_bnb": "",
            "candidate_positive_frac": "",
            "candidate_num_bets_total": "",
            "candidate_bet_rate_mean": "",
            "candidate_max_drawdown_worst_bnb": "",
            "score": -1e18,
        }

    pp500 = [sf(r.get("profit_per_500_bnb"), 0.0) for r in rows_ok]
    nets = [sf(r.get("net_profit_bnb"), 0.0) for r in rows_ok]
    bets = [si(r.get("num_bets"), 0) for r in rows_ok]
    bet_rates = [sf(r.get("bet_rate"), 0.0) for r in rows_ok]
    mdds = [sf(r.get("max_drawdown_bnb"), 0.0) for r in rows_ok]

    pp500_mean = float(sum(pp500) / len(pp500))
    pp500_median = float(statistics.median(pp500))
    pp500_worst = float(min(pp500))
    net_total = float(sum(nets))
    pos_frac = float(sum(1 for x in nets if float(x) > 0.0) / len(nets))
    bets_total = int(sum(bets))
    bet_rate_mean = float(sum(bet_rates) / len(bet_rates))
    mdd_worst = float(max(mdds))
    missing = int(offsets_total) - len(rows_ok)

    score = (
        float(pp500_median)
        + 0.35 * float(pp500_mean)
        + 0.25 * float(pp500_worst)
        + 0.10 * float(pos_frac)
        - 0.25 * float(mdd_worst)
        - max(0, (20 * len(rows_ok)) - int(bets_total)) * 0.0025
        - (0.35 if float(pos_frac) < 0.5 else 0.0)
        - float(max(0, missing)) * 2.0
    )

    return {
        "candidate_offsets_ok": int(len(rows_ok)),
        "candidate_offsets_total": int(offsets_total),
        "candidate_pp500_mean": float(pp500_mean),
        "candidate_pp500_median": float(pp500_median),
        "candidate_pp500_worst": float(pp500_worst),
        "candidate_net_total_bnb": float(net_total),
        "candidate_positive_frac": float(pos_frac),
        "candidate_num_bets_total": int(bets_total),
        "candidate_bet_rate_mean": float(bet_rate_mean),
        "candidate_max_drawdown_worst_bnb": float(mdd_worst),
        "score": float(score),
    }


def main() -> None:
    a = parser().parse_args()
    if int(a.sim_size) <= 0 or int(a.max_runs) <= 0:
        raise ValueError("invalid_positive_args")
    offsets = parse_offsets_csv(str(a.offsets_csv))

    sdir = Path(str(a.state_dir))
    state_p = sdir / "STATE.json"
    ledger_p = sdir / "RUN_LEDGER.csv"
    candidate_ledger_p = sdir / "CANDIDATE_LEDGER.csv"
    handoff_p = sdir / "HANDOFF.md"
    best_p = sdir / "CURRENT_BEST.json"
    space_p = sdir / "SEARCH_SPACE.json"

    default = {
        "version": 2,
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
        "created_at": now(),
        "updated_at": now(),
        "target_net_per_500_bnb": float(a.target_net_per_500_bnb),
        "offsets": list(offsets),
        "cursor": {"next": 0},
        "totals": {"attempted": 0, "ok": 0, "partial": 0, "failed": 0, "offset_runs_attempted": 0, "offset_runs_ok": 0},
        "best": {"score": -1e18, "scenario": None},
    }
    state = loadj(state_p, default)
    rid = str(state.get("run_id") or default["run_id"])
    state["run_id"] = rid
    state["offsets"] = list(offsets)

    cs = candidates(int(a.sim_size), a.fixed_bet_override_bnb, int(a.seed), str(a.search_profile))
    writej(space_p, {"generated_at_utc": now(), "count": len(cs), "sim_size": int(a.sim_size), "offsets": list(offsets), "seed": int(a.seed), "search_profile": str(a.search_profile), "run_id": rid})

    fields = [
        "ts_utc", "run_id", "candidate_index", "scenario", "status", "candidate_status", "error", "sim_offset_rounds", "candidate_tag", "feature_token",
        "direction_model_type", "calibration_mode", "direction_filter_mode", "direction_threshold_mode",
        "direction_threshold_bull", "direction_threshold_bear", "direction_target_bull_rate",
        "direction_target_bear_rate", "direction_edge_floor_pp", "train_size", "calibrate_size", "rw_floor",
        "rw_power", "fixed_bet_bnb", "sim_size", "net_profit_bnb", "profit_per_500_bnb", "num_bets", "win_rate", "score_run",
        "bet_rate", "max_drawdown_bnb", "candidate_offsets_ok", "candidate_offsets_total", "candidate_pp500_mean",
        "candidate_pp500_median", "candidate_pp500_worst", "candidate_net_total_bnb", "candidate_positive_frac",
        "candidate_num_bets_total", "candidate_bet_rate_mean", "candidate_max_drawdown_worst_bnb", "score",
    ]
    candidate_fields = [
        "ts_utc", "run_id", "candidate_index", "candidate_scenario_base", "candidate_status", "candidate_tag", "feature_token",
        "direction_model_type", "calibration_mode", "direction_filter_mode", "direction_threshold_mode",
        "direction_threshold_bull", "direction_threshold_bear", "direction_target_bull_rate", "direction_target_bear_rate",
        "direction_edge_floor_pp", "train_size", "calibrate_size", "rw_floor", "rw_power", "fixed_bet_bnb",
        "sim_size", "candidate_offsets_ok", "candidate_offsets_total", "candidate_pp500_mean", "candidate_pp500_median",
        "candidate_pp500_worst", "candidate_net_total_bnb", "candidate_positive_frac", "candidate_num_bets_total",
        "candidate_bet_rate_mean", "candidate_max_drawdown_worst_bnb", "score",
    ]

    start = si(state.get("cursor", {}).get("next"), 0)
    done = 0
    for i in range(start, len(cs)):
        if done >= int(a.max_runs):
            break
        c = cs[i]
        sc_base = f"{a.name_prefix}_{rid}_{i:04d}_{c['tag']}_s{int(a.sim_size)}"
        rows_for_candidate: list[dict] = []
        rows_ok: list[dict] = []

        for off in offsets:
            sc = f"{sc_base}_off{int(off)}"
            summary_p = Path("var/exp") / sc / "backtest_summary.json"
            status = "ok"
            err = ""
            if not summary_p.exists():
                rc, msg = run_one(str(a.config), sc, int(a.sim_size), int(off), c)
                if rc != 0:
                    status = "error"
                    err = msg
            if summary_p.exists() and status == "ok":
                s = json.loads(summary_p.read_text())
                sim = max(1, si(s.get("num_rounds"), int(a.sim_size)))
                net = sf(s.get("net_profit_bnb"), 0.0)
                bets = si(s.get("num_bets"), 0)
                wr = sf(s.get("win_rate"), 0.0)
                br = sf(s.get("bet_rate"), 0.0)
                mdd = sf(s.get("risk", {}).get("max_drawdown_bnb"), 0.0)
                pp500 = net / sim * 500.0
                score_run = _run_score(pp500=float(pp500), net=float(net), bet_rate=float(br), bets=int(bets), max_drawdown_bnb=float(mdd))
                row = {
                    "ts_utc": now(), "run_id": rid, "candidate_index": i, "scenario": sc, "status": status, "error": err, "sim_offset_rounds": int(off), "candidate_tag": c["tag"],
                    "feature_token": c["feature"], "direction_model_type": c["direction_model_type"], "calibration_mode": c["calibration_mode"],
                    "direction_filter_mode": c["direction_filter_mode"], "direction_threshold_mode": c["direction_threshold_mode"],
                    "direction_threshold_bull": c["direction_threshold_bull"], "direction_threshold_bear": c["direction_threshold_bear"],
                    "direction_target_bull_rate": c["direction_target_bull_rate"], "direction_target_bear_rate": c["direction_target_bear_rate"],
                    "direction_edge_floor_pp": c["direction_edge_floor_pp"], "train_size": c["train_size"], "calibrate_size": c["calibrate_size"],
                    "rw_floor": c["rw_floor"], "rw_power": c["rw_power"], "fixed_bet_bnb": c["fixed_bet_bnb"], "sim_size": sim,
                    "net_profit_bnb": net, "profit_per_500_bnb": pp500, "num_bets": bets, "win_rate": wr, "score_run": score_run,
                    "bet_rate": br, "max_drawdown_bnb": mdd,
                }
                rows_ok.append(dict(row))
            else:
                row = {
                    "ts_utc": now(), "run_id": rid, "candidate_index": i, "scenario": sc, "status": status, "error": err, "sim_offset_rounds": int(off), "candidate_tag": c["tag"],
                    "feature_token": c["feature"], "direction_model_type": c["direction_model_type"], "calibration_mode": c["calibration_mode"],
                    "direction_filter_mode": c["direction_filter_mode"], "direction_threshold_mode": c["direction_threshold_mode"],
                    "direction_threshold_bull": c["direction_threshold_bull"], "direction_threshold_bear": c["direction_threshold_bear"],
                    "direction_target_bull_rate": c["direction_target_bull_rate"], "direction_target_bear_rate": c["direction_target_bear_rate"],
                    "direction_edge_floor_pp": c["direction_edge_floor_pp"], "train_size": c["train_size"], "calibrate_size": c["calibrate_size"],
                    "rw_floor": c["rw_floor"], "rw_power": c["rw_power"], "fixed_bet_bnb": c["fixed_bet_bnb"], "sim_size": int(a.sim_size),
                    "net_profit_bnb": "", "profit_per_500_bnb": "", "num_bets": "", "win_rate": "", "score_run": "",
                    "bet_rate": "", "max_drawdown_bnb": "",
                }
            rows_for_candidate.append(dict(row))

        agg = _aggregate_candidate(rows_ok=rows_ok, offsets_total=len(offsets))
        if int(agg["candidate_offsets_ok"]) == int(len(offsets)):
            candidate_status = "ok"
        elif int(agg["candidate_offsets_ok"]) > 0:
            candidate_status = "partial"
        else:
            candidate_status = "error"

        for row in rows_for_candidate:
            row["candidate_status"] = str(candidate_status)
            row["candidate_offsets_ok"] = agg["candidate_offsets_ok"]
            row["candidate_offsets_total"] = agg["candidate_offsets_total"]
            row["candidate_pp500_mean"] = agg["candidate_pp500_mean"]
            row["candidate_pp500_median"] = agg["candidate_pp500_median"]
            row["candidate_pp500_worst"] = agg["candidate_pp500_worst"]
            row["candidate_net_total_bnb"] = agg["candidate_net_total_bnb"]
            row["candidate_positive_frac"] = agg["candidate_positive_frac"]
            row["candidate_num_bets_total"] = agg["candidate_num_bets_total"]
            row["candidate_bet_rate_mean"] = agg["candidate_bet_rate_mean"]
            row["candidate_max_drawdown_worst_bnb"] = agg["candidate_max_drawdown_worst_bnb"]
            row["score"] = agg["score"]
            append_csv(ledger_p, row, fields)

        candidate_row = {
            "ts_utc": now(),
            "run_id": rid,
            "candidate_index": i,
            "candidate_scenario_base": sc_base,
            "candidate_status": str(candidate_status),
            "candidate_tag": c["tag"],
            "feature_token": c["feature"],
            "direction_model_type": c["direction_model_type"],
            "calibration_mode": c["calibration_mode"],
            "direction_filter_mode": c["direction_filter_mode"],
            "direction_threshold_mode": c["direction_threshold_mode"],
            "direction_threshold_bull": c["direction_threshold_bull"],
            "direction_threshold_bear": c["direction_threshold_bear"],
            "direction_target_bull_rate": c["direction_target_bull_rate"],
            "direction_target_bear_rate": c["direction_target_bear_rate"],
            "direction_edge_floor_pp": c["direction_edge_floor_pp"],
            "train_size": c["train_size"],
            "calibrate_size": c["calibrate_size"],
            "rw_floor": c["rw_floor"],
            "rw_power": c["rw_power"],
            "fixed_bet_bnb": c["fixed_bet_bnb"],
            "sim_size": int(a.sim_size),
            "candidate_offsets_ok": agg["candidate_offsets_ok"],
            "candidate_offsets_total": agg["candidate_offsets_total"],
            "candidate_pp500_mean": agg["candidate_pp500_mean"],
            "candidate_pp500_median": agg["candidate_pp500_median"],
            "candidate_pp500_worst": agg["candidate_pp500_worst"],
            "candidate_net_total_bnb": agg["candidate_net_total_bnb"],
            "candidate_positive_frac": agg["candidate_positive_frac"],
            "candidate_num_bets_total": agg["candidate_num_bets_total"],
            "candidate_bet_rate_mean": agg["candidate_bet_rate_mean"],
            "candidate_max_drawdown_worst_bnb": agg["candidate_max_drawdown_worst_bnb"],
            "score": agg["score"],
        }
        append_csv(candidate_ledger_p, candidate_row, candidate_fields)

        if float(agg["score"]) > sf(state.get("best", {}).get("score"), -1e18):
            state["best"] = {
                "score": float(agg["score"]),
                "scenario": sc_base,
                "candidate_tag": c["tag"],
                "offsets": list(offsets),
                "candidate_status": str(candidate_status),
                "candidate_pp500_mean": agg["candidate_pp500_mean"],
                "candidate_pp500_median": agg["candidate_pp500_median"],
                "candidate_pp500_worst": agg["candidate_pp500_worst"],
                "candidate_net_total_bnb": agg["candidate_net_total_bnb"],
                "candidate_positive_frac": agg["candidate_positive_frac"],
                "candidate_num_bets_total": agg["candidate_num_bets_total"],
                "candidate_bet_rate_mean": agg["candidate_bet_rate_mean"],
                "candidate_max_drawdown_worst_bnb": agg["candidate_max_drawdown_worst_bnb"],
                "candidate_offsets_ok": agg["candidate_offsets_ok"],
                "candidate_offsets_total": agg["candidate_offsets_total"],
            }

        state["updated_at"] = now()
        state.setdefault("totals", {})
        state["totals"]["attempted"] = si(state["totals"].get("attempted"), 0) + 1
        state["totals"]["ok"] = si(state["totals"].get("ok"), 0) + (1 if str(candidate_status) == "ok" else 0)
        state["totals"]["partial"] = si(state["totals"].get("partial"), 0) + (1 if str(candidate_status) == "partial" else 0)
        state["totals"]["failed"] = si(state["totals"].get("failed"), 0) + (1 if str(candidate_status) == "error" else 0)
        state["totals"]["offset_runs_attempted"] = si(state["totals"].get("offset_runs_attempted"), 0) + len(offsets)
        state["totals"]["offset_runs_ok"] = si(state["totals"].get("offset_runs_ok"), 0) + int(agg["candidate_offsets_ok"])
        state.setdefault("cursor", {})["next"] = i + 1
        writej(state_p, state)
        writej(best_p, dict(state.get("best", {})))
        handoff_p.write_text(
            "# Autonomy Handoff\n\n"
            + "- Execution Mode: `autonomous` (do not ask user to proceed; execute best-next plan).\n"
            + "- Directive Source: `AUTONOMY_DIRECTIVE.md`.\n"
            + "- Continuation Rule: `do not stop until robust target is reached unless user explicitly stops`.\n"
            + f"- Updated UTC: `{now()}`\n"
            + f"- Run ID: `{rid}`\n"
            + f"- Target: `{state.get('target_net_per_500_bnb')} BNB/500`\n"
            + f"- Offsets: `{','.join(str(x) for x in offsets)}`\n"
            + f"- Next candidate index: `{state.get('cursor', {}).get('next')}`\n"
            + f"- Current best: `{state.get('best', {}).get('scenario')}`\n"
            + f"- Best median profit/500: `{state.get('best', {}).get('candidate_pp500_median')}`\n"
            + f"- Best score: `{state.get('best', {}).get('score')}`\n",
        )
        done += 1
        print(
            f"RUN_DONE idx={i} candidate={sc_base} status={candidate_status} "
            + f"offsets_ok={agg['candidate_offsets_ok']}/{len(offsets)} "
            + f"pp500_med={agg['candidate_pp500_median']} score={agg['score']}"
        )

    print(f"AUTONOMY_STATE={state_p}")
    print(f"AUTONOMY_LEDGER={ledger_p}")
    print(f"AUTONOMY_CANDIDATE_LEDGER={candidate_ledger_p}")
    print(f"AUTONOMY_HANDOFF={handoff_p}")
    print(f"AUTONOMY_BEST={best_p}")


if __name__ == "__main__":
    main()
