"""Regime characterization Step 1 — 2026-05-25.

Computes time/vol/pool/outcome/payout/correlation features per cohort.
Cohorts: CV5, holdout, ext_v2, fresh_oos (+ post-fresh tail). Extension
cohort (422298..437561) raw data unavailable locally — features that
need extension raw data are deferred; BTC vol stats sourced from the
archived regime_phase0_vol_distribution.json (prior research).

Writes intermediate JSON to var/strategy_review/regime_step1_data.json
and the markdown report at
var/strategy_review/2026_05_25_regime_characterization_step1.md.

Read-only with respect to var/live/. Streaming reads of large jsonl
files so memory stays bounded.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]

# Cohort epoch ranges (inclusive)
COHORTS = {
    "extension": (422298, 437561),
    "cv5":       (437562, 474086),
    "holdout":   (474880, 475311),
    "ext_v2":    (475312, 479952),
    "fresh_oos": (479953, 483191),
    "post_fresh":(483192, 999999),  # live-extension since 2026-05-22 backtest
}

# Bet treasury fee (canonical 3%)
TREASURY_FEE = 0.03

# Canonical jsonl paths (CV5 onwards)
ROUNDS_PATH = REPO / "var" / "closed_rounds.jsonl"
BTC_KLINES = REPO / "var" / "btc_spot_prices.jsonl"
ETH_KLINES = REPO / "var" / "eth_spot_prices.jsonl"
SOL_KLINES = REPO / "var" / "sol_spot_prices.jsonl"

# Extension archive (extracted from 2026_05_21_var_archive/extended.tar.gz).
# Contains the epoch 422298..437561 cohort raw data — required for
# extension-cohort features (canonical var/ only covers 437562+).
EXT_DIR = Path(r"C:\Users\zking\AppData\Local\Temp\ext\extended")
EXT_ROUNDS = EXT_DIR / "closed_rounds.jsonl"
EXT_BTC = EXT_DIR / "btc_spot_prices.jsonl"
EXT_ETH = EXT_DIR / "eth_spot_prices.jsonl"
EXT_SOL = EXT_DIR / "sol_spot_prices.jsonl"

PRIOR_VOL_JSON = Path(r"C:\Users\zking\Downloads\OLD\pancakebot\2026_05_23_var_cleanup\incident_reports\regime_phase0_vol_distribution.json")
PRIOR_PERM_JSON = Path(r"C:\Users\zking\Downloads\OLD\pancakebot\2026_05_23_var_cleanup\incident_reports\regime_phase0_permutation.json")


def cohort_of(epoch: int) -> str | None:
    for name, (lo, hi) in COHORTS.items():
        if lo <= epoch <= hi:
            return name
    return None


def _percentiles(values: list[float], pcts: list[float]) -> dict[str, float]:
    if not values:
        return {f"p{int(p)}": float("nan") for p in pcts}
    s = sorted(values)
    out = {}
    for p in pcts:
        idx = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
        out[f"p{int(p)}"] = s[idx]
    return out


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0, "mean": float("nan"), "median": float("nan"), "std": float("nan"),
                "p10": float("nan"), "p25": float("nan"), "p75": float("nan"), "p90": float("nan")}
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        **_percentiles(values, [10, 25, 75, 90]),
    }


def _ingest_rounds_file(path: Path, out: dict[str, list[dict]]) -> int:
    """Stream a closed_rounds jsonl, bucket per-cohort, count adds."""
    n_added = 0
    with path.open(encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            r = json.loads(ln)
            ep = r["epoch"]
            c = cohort_of(ep)
            if c is None:
                continue
            if r.get("failed"):
                continue
            bets = r.get("bets") or []
            bull_amount_wei = sum(b["amountWei"] for b in bets if b.get("position") == "Bull")
            bear_amount_wei = sum(b["amountWei"] for b in bets if b.get("position") == "Bear")
            total_amount_wei = bull_amount_wei + bear_amount_wei
            out[c].append({
                "epoch": ep,
                "startAt": r["startAt"],
                "position": r.get("position"),
                "bull_bnb": bull_amount_wei / 1e18,
                "bear_bnb": bear_amount_wei / 1e18,
                "total_bnb": total_amount_wei / 1e18,
                "lockPrice": r.get("lockPrice"),
                "closePrice": r.get("closePrice"),
            })
            n_added += 1
    return n_added


def load_round_aggregates() -> dict[str, list[dict]]:
    """Stream both canonical AND extended closed_rounds jsonl, dedup by epoch."""
    out: dict[str, list[dict]] = {k: [] for k in COHORTS}
    seen_epochs: set[int] = set()
    if EXT_ROUNDS.exists():
        # Extended file first (contains the pre-CV5 extension cohort)
        _ingest_rounds_file(EXT_ROUNDS, out)
        for rs in out.values():
            for r in rs:
                seen_epochs.add(r["epoch"])
    # Canonical file — adds CV5 + holdout + ext_v2 + fresh + post_fresh
    # (extended file may also contain some of these epochs — dedup)
    canonical_only: dict[str, list[dict]] = {k: [] for k in COHORTS}
    _ingest_rounds_file(ROUNDS_PATH, canonical_only)
    for c, rs in canonical_only.items():
        for r in rs:
            if r["epoch"] in seen_epochs:
                continue
            out[c].append(r)
            seen_epochs.add(r["epoch"])
    return out


def payout_multiple(bull_bnb: float, bear_bnb: float, winner: str | None) -> float | None:
    """Round payout multiple = total_pool * (1 - treasury_fee) / winning_side_pool."""
    if winner not in ("Bull", "Bear"):
        return None
    total = bull_bnb + bear_bnb
    win_side = bull_bnb if winner == "Bull" else bear_bnb
    if win_side <= 0 or total <= 0:
        return None
    return (total * (1.0 - TREASURY_FEE)) / win_side


def time_features(rounds: list[dict]) -> dict[str, Any]:
    if not rounds:
        return {"n": 0}
    months: Counter[int] = Counter()
    dows: Counter[int] = Counter()
    hours: Counter[int] = Counter()
    start_min, start_max = float("inf"), float("-inf")
    for r in rounds:
        ts = r["startAt"]
        u = dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc)
        months[u.month] += 1
        dows[u.weekday()] += 1  # 0=Mon
        hours[u.hour] += 1
        start_min = min(start_min, ts)
        start_max = max(start_max, ts)
    n = len(rounds)
    return {
        "n": n,
        "date_start": dt.datetime.fromtimestamp(start_min, tz=dt.timezone.utc).isoformat(),
        "date_end":   dt.datetime.fromtimestamp(start_max, tz=dt.timezone.utc).isoformat(),
        "months_pct": {str(m): round(100.0 * months[m] / n, 1) for m in sorted(months)},
        "dow_pct":    {str(d): round(100.0 * dows[d] / n, 1) for d in sorted(dows)},
        "hours_pct":  {str(h): round(100.0 * hours[h] / n, 1) for h in sorted(hours)},
    }


def pool_features(rounds: list[dict]) -> dict[str, Any]:
    pools = [r["total_bnb"] for r in rounds if r["total_bnb"] > 0]
    below_min = sum(1 for p in pools if p < 1.5)
    return {
        **_stats(pools),
        "pct_below_1_5_bnb": round(100.0 * below_min / max(1, len(pools)), 2),
    }


def outcome_features(rounds: list[dict]) -> dict[str, Any]:
    n = len(rounds)
    if n == 0:
        return {"n": 0}
    bulls = sum(1 for r in rounds if r["position"] == "Bull")
    bears = sum(1 for r in rounds if r["position"] == "Bear")
    others = n - bulls - bears
    # Streaks (consecutive same-side wins)
    seq = [r["position"] for r in sorted(rounds, key=lambda r: r["epoch"]) if r["position"] in ("Bull", "Bear")]
    max_bull_streak = max_bear_streak = cur = 0
    cur_side = None
    for s in seq:
        if s == cur_side:
            cur += 1
        else:
            cur_side = s
            cur = 1
        if s == "Bull":
            max_bull_streak = max(max_bull_streak, cur)
        else:
            max_bear_streak = max(max_bear_streak, cur)
    return {
        "n": n,
        "bull_pct": round(100.0 * bulls / n, 2),
        "bear_pct": round(100.0 * bears / n, 2),
        "null_or_unknown_pct": round(100.0 * others / n, 2),
        "max_bull_streak": max_bull_streak,
        "max_bear_streak": max_bear_streak,
    }


def payout_features(rounds: list[dict]) -> dict[str, Any]:
    pms = []
    for r in rounds:
        pm = payout_multiple(r["bull_bnb"], r["bear_bnb"], r["position"])
        if pm is not None:
            pms.append(pm)
    below_1_5 = sum(1 for pm in pms if pm < 1.5)
    return {
        **_stats(pms),
        "pct_below_1_5x": round(100.0 * below_1_5 / max(1, len(pms)), 2),
    }


# --- kline-based features --------------------------------------------------

def load_kline_close_per_epoch(path: Path, epochs_needed: set[int]) -> dict[int, list[float]]:
    """Return {epoch: [close_price_per_second, ...]} for each requested epoch.

    Each round's klines_1s field has 1-second OHLCV (typically 300 samples =
    5 minutes covering the lock-window). Keeping full 1s resolution gives
    enough samples per round for stable std/correlation estimates. Empty
    list returned for rounds with missing kline data (extension cohort
    has ~31% MISSING_VERIFIED_BY_PROBE_RESULT per prior research).
    """
    out: dict[int, list[float]] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            r = json.loads(ln)
            ep = r["epoch"]
            if ep not in epochs_needed:
                continue
            klines = r.get("klines_1s") or []
            # k = [ts_ms, open, high, low, close, vol] — keep close only
            out[ep] = [k[4] for k in klines]
    return out


def load_klines_unified(canonical_path: Path, extended_path: Path, epochs_needed: set[int]) -> dict[int, list[tuple[int, float]]]:
    """Merge canonical + extended kline files into one dict by epoch."""
    out = load_kline_close_per_epoch(canonical_path, epochs_needed)
    if extended_path.exists():
        ext = load_kline_close_per_epoch(extended_path, epochs_needed)
        for ep, samples in ext.items():
            if ep not in out:
                out[ep] = samples
    return out


def _log_returns(prices: list[float]) -> list[float]:
    """Per-step log returns ignoring non-positive prices."""
    rets = []
    for i in range(1, len(prices)):
        p0, p1 = prices[i - 1], prices[i]
        if p0 > 0 and p1 > 0:
            rets.append(math.log(p1 / p0))
    return rets


def realized_vol_per_round(prices: list[float]) -> float | None:
    """Std of 1-second log-returns across the round's 5-minute kline window.
    Returns std dev of returns — effectively per-second realized volatility.
    """
    if len(prices) < 60:  # need >=1 minute of samples
        return None
    rets = _log_returns(prices)
    if len(rets) < 50:
        return None
    return statistics.pstdev(rets)


def round_total_move(prices: list[float]) -> float | None:
    """|close - open| / open over the round's entire 5-min window."""
    if len(prices) < 2:
        return None
    p0, p1 = prices[0], prices[-1]
    if p0 <= 0:
        return None
    return abs(p1 - p0) / p0


def vol_features_for_cohort(rounds: list[dict], btc_klines_by_ep: dict[int, list[float]]) -> dict[str, Any]:
    vols, total_moves, big_move_flags = [], [], []
    for r in rounds:
        prices = btc_klines_by_ep.get(r["epoch"])
        if not prices:
            continue
        v = realized_vol_per_round(prices)
        if v is not None:
            vols.append(v)
        m = round_total_move(prices)
        if m is not None:
            total_moves.append(m)
            big_move_flags.append(1 if m > 0.001 else 0)
    pct_big = (100.0 * sum(big_move_flags) / len(big_move_flags)) if big_move_flags else float("nan")
    return {
        "btc_realized_vol_per_round": _stats(vols),
        "btc_round_total_move": _stats(total_moves),
        "btc_pct_rounds_move_above_0_1pct": pct_big,
    }


def _pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 30:
        return None
    mx, my = sum(x) / len(x), sum(y) / len(y)
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    dx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    dy = math.sqrt(sum((yi - my) ** 2 for yi in y))
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def correlation_features(rounds: list[dict],
                          btc_kl: dict[int, list[float]],
                          eth_kl: dict[int, list[float]],
                          sol_kl: dict[int, list[float]]) -> dict[str, Any]:
    """Per-round Pearson r of 1-second log returns between asset pairs."""
    pearson_btc_eth, pearson_btc_sol, pearson_eth_sol = [], [], []
    for r in rounds:
        ep = r["epoch"]
        b, e, s = btc_kl.get(ep), eth_kl.get(ep), sol_kl.get(ep)
        if not (b and e and s):
            continue
        n = min(len(b), len(e), len(s))
        if n < 60:
            continue
        rb = _log_returns(b[:n])
        re_ = _log_returns(e[:n])
        rs = _log_returns(s[:n])
        # All three must have the same length after trimming
        m = min(len(rb), len(re_), len(rs))
        if m < 50:
            continue
        rb, re_, rs = rb[:m], re_[:m], rs[:m]
        rbe = _pearson(rb, re_)
        rbs = _pearson(rb, rs)
        res = _pearson(re_, rs)
        if rbe is not None: pearson_btc_eth.append(rbe)
        if rbs is not None: pearson_btc_sol.append(rbs)
        if res is not None: pearson_eth_sol.append(res)
    return {
        "btc_eth_pearson_per_round": _stats(pearson_btc_eth),
        "btc_sol_pearson_per_round": _stats(pearson_btc_sol),
        "eth_sol_pearson_per_round": _stats(pearson_eth_sol),
    }


def load_prior_extension_stats() -> dict[str, Any]:
    out: dict[str, Any] = {}
    if PRIOR_VOL_JSON.exists():
        with PRIOR_VOL_JSON.open(encoding="utf-8") as f:
            vol = json.load(f)
        out["btc_vol_prior_research"] = vol
    if PRIOR_PERM_JSON.exists():
        with PRIOR_PERM_JSON.open(encoding="utf-8") as f:
            perm = json.load(f)
        out["permutation_null_prior_research"] = perm
    return out


def main() -> None:
    print("loading rounds...")
    by_cohort = load_round_aggregates()
    for name in COHORTS:
        print(f"  {name}: {len(by_cohort[name])} rounds")

    print("collecting epochs needed for kline features...")
    epochs_needed: set[int] = set()
    for rs in by_cohort.values():
        for r in rs:
            epochs_needed.add(r["epoch"])
    print(f"  total epochs to lookup klines for: {len(epochs_needed)}")

    print("loading BTC klines (canonical + extended)...")
    btc_kl = load_klines_unified(BTC_KLINES, EXT_BTC, epochs_needed)
    print(f"  matched: {len(btc_kl)}")
    print("loading ETH klines (canonical + extended)...")
    eth_kl = load_klines_unified(ETH_KLINES, EXT_ETH, epochs_needed)
    print(f"  matched: {len(eth_kl)}")
    print("loading SOL klines (canonical + extended)...")
    sol_kl = load_klines_unified(SOL_KLINES, EXT_SOL, epochs_needed)
    print(f"  matched: {len(sol_kl)}")

    print("computing features per cohort...")
    result: dict[str, Any] = {"cohorts": {}}
    for name, rounds in by_cohort.items():
        print(f"  {name}...")
        result["cohorts"][name] = {
            "epoch_range": COHORTS[name],
            "n_rounds": len(rounds),
            "time": time_features(rounds),
            "pool": pool_features(rounds),
            "outcome": outcome_features(rounds),
            "payout": payout_features(rounds),
            "btc_vol": vol_features_for_cohort(rounds, btc_kl),
            "correlation": correlation_features(rounds, btc_kl, eth_kl, sol_kl),
        }

    print("loading prior extension stats...")
    result["extension_prior"] = load_prior_extension_stats()
    result["extension_status"] = "data_unavailable_locally - var/extended/ archived 2026-05-23; refetch via research/backfill_okx_extended.py for full extraction"

    out_path = REPO / "var" / "strategy_review" / "regime_step1_data.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True, default=str)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
