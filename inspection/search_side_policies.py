from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Row:
    epoch: int
    p_final: float
    direction: str
    profit_bnb: float
    is_correct: bool


def _load_closed_positions(path: Path) -> dict[int, str]:
    if not path.exists():
        raise FileNotFoundError(f"missing_closed_rounds_jsonl: {path}")
    out: dict[int, str] = {}
    for line in path.read_text().splitlines():
        s = str(line).strip()
        if not s:
            continue
        obj = json.loads(s)
        ep = int(obj["epoch"])
        pos = str(obj.get("position") or "")
        out[int(ep)] = str(pos)
    return out


def _read_rows(path: Path, *, closed_position_by_epoch: dict[int, str]) -> list[Row]:
    with path.open(newline="") as f:
        rd = csv.DictReader(f)
        out: list[Row] = []
        for r in rd:
            ep = int(r["epoch"])
            direction = str(r["direction"])
            pos = str(closed_position_by_epoch.get(ep, ""))
            is_correct = bool((direction == "Bull" and pos == "Bull") or (direction == "Bear" and pos == "Bear"))
            out.append(
                Row(
                    epoch=int(ep),
                    p_final=float(r["p_final"]),
                    direction=str(direction),
                    profit_bnb=float(r["profit_bnb"]),
                    is_correct=bool(is_correct),
                )
            )
    out.sort(key=lambda x: int(x.epoch))
    return out


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    vv = sorted(values)
    idx = int(round((len(vv) - 1) * float(q)))
    idx = max(0, min(len(vv) - 1, idx))
    return float(vv[idx])


def _drawdown(cum: list[float]) -> float:
    peak = float("-inf")
    mdd = 0.0
    for v in cum:
        if v > peak:
            peak = float(v)
        dd = float(peak - v)
        if dd > mdd:
            mdd = float(dd)
    return float(mdd)


def _metrics(profits: list[float], correct_flags: list[bool]) -> dict[str, float | int]:
    n = int(len(profits))
    if n != len(correct_flags):
        raise ValueError("profits_correct_flags_len_mismatch")
    if n <= 0:
        return {
            "num_bets": 0,
            "win_rate_directional": float("nan"),
            "win_rate_profitable": float("nan"),
            "net_profit_bnb": 0.0,
            "avg_profit_bnb": float("nan"),
            "max_drawdown_bnb": 0.0,
            "profit_factor": float("nan"),
            "split1_net_bnb": 0.0,
            "split2_net_bnb": 0.0,
            "split1_win_rate_directional": float("nan"),
            "split2_win_rate_directional": float("nan"),
        }

    wins_dir = int(sum(1 for x in correct_flags if bool(x)))
    wins_profit = int(sum(1 for p in profits if float(p) > 0.0))
    net = float(sum(profits))
    avg = float(net / n)

    gp = float(sum(p for p in profits if float(p) > 0.0))
    gl = float(sum(-float(p) for p in profits if float(p) < 0.0))
    pf = float(gp / gl) if gl > 0.0 else float("inf")

    cum: list[float] = []
    s = 0.0
    for p in profits:
        s += float(p)
        cum.append(float(s))
    mdd = float(_drawdown(cum))

    mid = int(n // 2)
    p1 = profits[:mid]
    p2 = profits[mid:]
    c1 = correct_flags[:mid]
    c2 = correct_flags[mid:]
    if not p1:
        p1 = profits
        c1 = correct_flags
    if not p2:
        p2 = profits
        c2 = correct_flags
    w1_dir = int(sum(1 for x in c1 if bool(x)))
    w2_dir = int(sum(1 for x in c2 if bool(x)))

    return {
        "num_bets": int(n),
        "win_rate_directional": float(wins_dir / n),
        "win_rate_profitable": float(wins_profit / n),
        "net_profit_bnb": float(net),
        "avg_profit_bnb": float(avg),
        "max_drawdown_bnb": float(mdd),
        "profit_factor": float(pf),
        "split1_net_bnb": float(sum(p1)),
        "split2_net_bnb": float(sum(p2)),
        "split1_win_rate_directional": float(w1_dir / len(c1)),
        "split2_win_rate_directional": float(w2_dir / len(c2)),
    }


def _score(m: dict[str, float | int]) -> float:
    n = int(m["num_bets"])
    if n <= 0:
        return float("-inf")
    net = float(m["net_profit_bnb"])
    mdd = float(m["max_drawdown_bnb"])
    wr_dir = float(m["win_rate_directional"])
    wr_prof = float(m["win_rate_profitable"])
    s1 = float(m["split1_net_bnb"])
    s2 = float(m["split2_net_bnb"])
    # Risk-adjusted score with split stability bonus/penalty.
    stability = 0.0
    if s1 > 0.0 and s2 > 0.0:
        stability = 0.02
    elif s1 < 0.0 and s2 < 0.0:
        stability = -0.02
    return float(net - 0.5 * mdd + 0.06 * wr_dir + 0.02 * wr_prof + stability)


def _threshold_grid(rows: list[Row]) -> tuple[list[float], list[float]]:
    p = [float(r.p_final) for r in rows]
    bull_q = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99]
    bear_q = [0.50, 0.40, 0.30, 0.20, 0.10, 0.05, 0.02, 0.01]
    bull_t = sorted({float(_quantile(p, q)) for q in bull_q if math.isfinite(_quantile(p, q))})
    bear_t = sorted({float(_quantile(p, q)) for q in bear_q if math.isfinite(_quantile(p, q))})
    return bull_t, bear_t


def _eval_single(
    *,
    name: str,
    rows: list[Row],
    bull_t: list[float],
    bear_t: list[float],
    min_bets: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    # Bull-only
    for tb in bull_t:
        sel = [r for r in rows if str(r.direction) == "Bull" and float(r.p_final) >= float(tb)]
        profits = [float(r.profit_bnb) for r in sel]
        correct = [bool(r.is_correct) for r in sel]
        m = _metrics(profits, correct)
        if int(m["num_bets"]) < int(min_bets):
            continue
        out.append(
            {
                "family": "bull_only",
                "scenario": str(name),
                "threshold_bull": float(tb),
                "threshold_bear": None,
                "metrics": m,
                "score": float(_score(m)),
            }
        )

    # Bear-only
    for tr in bear_t:
        sel = [r for r in rows if str(r.direction) == "Bear" and float(r.p_final) <= float(tr)]
        profits = [float(r.profit_bnb) for r in sel]
        correct = [bool(r.is_correct) for r in sel]
        m = _metrics(profits, correct)
        if int(m["num_bets"]) < int(min_bets):
            continue
        out.append(
            {
                "family": "bear_only",
                "scenario": str(name),
                "threshold_bull": None,
                "threshold_bear": float(tr),
                "metrics": m,
                "score": float(_score(m)),
            }
        )

    # Both-sided threshold
    for tb in bull_t:
        for tr in bear_t:
            sel = [
                r
                for r in rows
                if (str(r.direction) == "Bull" and float(r.p_final) >= float(tb))
                or (str(r.direction) == "Bear" and float(r.p_final) <= float(tr))
            ]
            profits = [float(r.profit_bnb) for r in sel]
            correct = [bool(r.is_correct) for r in sel]
            m = _metrics(profits, correct)
            if int(m["num_bets"]) < int(min_bets):
                continue
            out.append(
                {
                    "family": "both_sides",
                    "scenario": str(name),
                    "threshold_bull": float(tb),
                    "threshold_bear": float(tr),
                    "metrics": m,
                    "score": float(_score(m)),
                }
            )
    return out


def _rows_by_epoch(rows: list[Row]) -> dict[int, Row]:
    out: dict[int, Row] = {}
    for r in rows:
        out[int(r.epoch)] = r
    return out


def _eval_two_pass(
    *,
    first_name: str,
    first_rows: list[Row],
    second_name: str,
    second_rows: list[Row],
    bull_t_first: list[float],
    bear_t_first: list[float],
    bull_t_second: list[float],
    bear_t_second: list[float],
    min_bets: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    f_map = _rows_by_epoch(first_rows)
    s_map = _rows_by_epoch(second_rows)
    epochs = sorted(set(f_map.keys()) & set(s_map.keys()))
    if not epochs:
        return out

    # Ordered policy:
    # pass 1 -> act if first scenario confident (both sides thresholds),
    # else pass 2 -> act if second scenario confident.
    for tfb in bull_t_first:
        for tfr in bear_t_first:
            for tsb in bull_t_second:
                for tsr in bear_t_second:
                    profits: list[float] = []
                    correct: list[bool] = []
                    for ep in epochs:
                        fr = f_map[ep]
                        acted = False
                        if str(fr.direction) == "Bull" and float(fr.p_final) >= float(tfb):
                            profits.append(float(fr.profit_bnb))
                            correct.append(bool(fr.is_correct))
                            acted = True
                        elif str(fr.direction) == "Bear" and float(fr.p_final) <= float(tfr):
                            profits.append(float(fr.profit_bnb))
                            correct.append(bool(fr.is_correct))
                            acted = True

                        if acted:
                            continue

                        sr = s_map[ep]
                        if str(sr.direction) == "Bull" and float(sr.p_final) >= float(tsb):
                            profits.append(float(sr.profit_bnb))
                            correct.append(bool(sr.is_correct))
                        elif str(sr.direction) == "Bear" and float(sr.p_final) <= float(tsr):
                            profits.append(float(sr.profit_bnb))
                            correct.append(bool(sr.is_correct))

                    m = _metrics(profits, correct)
                    if int(m["num_bets"]) < int(min_bets):
                        continue
                    out.append(
                        {
                            "family": "two_pass_ordered",
                            "first_scenario": str(first_name),
                            "second_scenario": str(second_name),
                            "threshold_bull_first": float(tfb),
                            "threshold_bear_first": float(tfr),
                            "threshold_bull_second": float(tsb),
                            "threshold_bear_second": float(tsr),
                            "metrics": m,
                            "score": float(_score(m)),
                        }
                    )
    return out


def _eval_consensus(
    *,
    first_name: str,
    first_rows: list[Row],
    second_name: str,
    second_rows: list[Row],
    bull_t_first: list[float],
    bear_t_first: list[float],
    bull_t_second: list[float],
    bear_t_second: list[float],
    min_bets: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    f_map = _rows_by_epoch(first_rows)
    s_map = _rows_by_epoch(second_rows)
    epochs = sorted(set(f_map.keys()) & set(s_map.keys()))
    if not epochs:
        return out

    # Consensus policy:
    # - both models must signal the same side with confidence thresholds.
    # - choose first model row for profit/correct accounting (identical side-level payout).
    for tfb in bull_t_first:
        for tfr in bear_t_first:
            for tsb in bull_t_second:
                for tsr in bear_t_second:
                    profits: list[float] = []
                    correct: list[bool] = []
                    for ep in epochs:
                        fr = f_map[ep]
                        sr = s_map[ep]

                        f_sig = None
                        s_sig = None
                        if str(fr.direction) == "Bull" and float(fr.p_final) >= float(tfb):
                            f_sig = "Bull"
                        elif str(fr.direction) == "Bear" and float(fr.p_final) <= float(tfr):
                            f_sig = "Bear"

                        if str(sr.direction) == "Bull" and float(sr.p_final) >= float(tsb):
                            s_sig = "Bull"
                        elif str(sr.direction) == "Bear" and float(sr.p_final) <= float(tsr):
                            s_sig = "Bear"

                        if f_sig is None or s_sig is None:
                            continue
                        if str(f_sig) != str(s_sig):
                            continue

                        profits.append(float(fr.profit_bnb))
                        correct.append(bool(fr.is_correct))

                    m = _metrics(profits, correct)
                    if int(m["num_bets"]) < int(min_bets):
                        continue
                    out.append(
                        {
                            "family": "two_model_consensus",
                            "first_scenario": str(first_name),
                            "second_scenario": str(second_name),
                            "threshold_bull_first": float(tfb),
                            "threshold_bear_first": float(tfr),
                            "threshold_bull_second": float(tsb),
                            "threshold_bear_second": float(tsr),
                            "metrics": m,
                            "score": float(_score(m)),
                        }
                    )
    return out


def _load_scenarios(names: list[str], *, closed_position_by_epoch: dict[int, str]) -> dict[str, list[Row]]:
    out: dict[str, list[Row]] = {}
    for nm in names:
        p = Path("var/exp") / str(nm) / "backtest_trades.csv"
        if not p.exists():
            raise FileNotFoundError(f"missing_trades_csv: {p}")
        out[str(nm)] = _read_rows(p, closed_position_by_epoch=closed_position_by_epoch)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", type=str, required=True, help="comma-separated scenario names")
    ap.add_argument("--min-bets", type=int, default=30)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--closed-rounds-jsonl", type=str, default="var/closed_rounds.jsonl")
    ap.add_argument("--out-json", type=str, required=True)
    args = ap.parse_args()

    scenario_names = [x.strip() for x in str(args.scenarios).split(",") if x.strip()]
    if not scenario_names:
        raise ValueError("empty_scenarios")
    min_bets = int(args.min_bets)
    top_k = int(args.top_k)
    if min_bets < 1:
        raise ValueError("min_bets_must_be_positive")
    if top_k < 1:
        raise ValueError("top_k_must_be_positive")

    closed_position_by_epoch = _load_closed_positions(Path(args.closed_rounds_jsonl))
    scen_rows = _load_scenarios(scenario_names, closed_position_by_epoch=closed_position_by_epoch)
    grids: dict[str, tuple[list[float], list[float]]] = {}
    for nm, rows in scen_rows.items():
        grids[str(nm)] = _threshold_grid(rows)

    single: list[dict[str, Any]] = []
    for nm, rows in scen_rows.items():
        bt, rt = grids[str(nm)]
        single.extend(_eval_single(name=nm, rows=rows, bull_t=bt, bear_t=rt, min_bets=min_bets))

    ordered: list[dict[str, Any]] = []
    consensus: list[dict[str, Any]] = []
    names = list(scen_rows.keys())
    for i in range(len(names)):
        for j in range(len(names)):
            if i == j:
                continue
            n1 = names[i]
            n2 = names[j]
            bt1, rt1 = grids[n1]
            bt2, rt2 = grids[n2]
            ordered.extend(
                _eval_two_pass(
                    first_name=n1,
                    first_rows=scen_rows[n1],
                    second_name=n2,
                    second_rows=scen_rows[n2],
                    bull_t_first=bt1,
                    bear_t_first=rt1,
                    bull_t_second=bt2,
                    bear_t_second=rt2,
                    min_bets=min_bets,
                )
            )
            consensus.extend(
                _eval_consensus(
                    first_name=n1,
                    first_rows=scen_rows[n1],
                    second_name=n2,
                    second_rows=scen_rows[n2],
                    bull_t_first=bt1,
                    bear_t_first=rt1,
                    bull_t_second=bt2,
                    bear_t_second=rt2,
                    min_bets=min_bets,
                )
            )

    def _top(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(rows, key=lambda x: float(x["score"]), reverse=True)[:top_k]

    out = {
        "scenarios": scenario_names,
        "min_bets": min_bets,
        "top_k": top_k,
        "single_total": len(single),
        "ordered_total": len(ordered),
        "consensus_total": len(consensus),
        "top_single": _top(single),
        "top_ordered": _top(ordered),
        "top_consensus": _top(consensus),
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True))

    print(f"OUT={out_path}")
    print(f"SINGLE_TOTAL={len(single)}")
    print(f"ORDERED_TOTAL={len(ordered)}")
    print(f"CONSENSUS_TOTAL={len(consensus)}")
    if out["top_single"]:
        t = out["top_single"][0]
        m = t["metrics"]
        print(
            "BEST_SINGLE="
            + f"{t['family']}:{t.get('scenario','')} "
            + f"score={t['score']:.6f} net={float(m['net_profit_bnb']):.6f} "
            + f"wr_dir={float(m['win_rate_directional']):.4f} "
            + f"wr_prof={float(m['win_rate_profitable']):.4f} "
            + f"bets={int(m['num_bets'])} mdd={float(m['max_drawdown_bnb']):.6f}"
        )
    if out["top_ordered"]:
        t = out["top_ordered"][0]
        m = t["metrics"]
        print(
            "BEST_ORDERED="
            + f"{t.get('first_scenario','')}->{t.get('second_scenario','')} "
            + f"score={t['score']:.6f} net={float(m['net_profit_bnb']):.6f} "
            + f"wr_dir={float(m['win_rate_directional']):.4f} "
            + f"wr_prof={float(m['win_rate_profitable']):.4f} "
            + f"bets={int(m['num_bets'])} mdd={float(m['max_drawdown_bnb']):.6f}"
        )
    if out["top_consensus"]:
        t = out["top_consensus"][0]
        m = t["metrics"]
        print(
            "BEST_CONSENSUS="
            + f"{t.get('first_scenario','')}~{t.get('second_scenario','')} "
            + f"score={t['score']:.6f} net={float(m['net_profit_bnb']):.6f} "
            + f"wr_dir={float(m['win_rate_directional']):.4f} "
            + f"wr_prof={float(m['win_rate_profitable']):.4f} "
            + f"bets={int(m['num_bets'])} mdd={float(m['max_drawdown_bnb']):.6f}"
        )


if __name__ == "__main__":
    main()
