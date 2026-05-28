"""Regime characterization Step 2 — pool-microstructure features.

Computes 4 features per cohort and tests whether any cleanly separates
the extension cohort (NEG) from the union of positive cohorts:

  1. dormant_revival_bull_share_size
     Wallets that bet in the current round AND haven't bet in the prior
     ``_DORMANT_LOOKBACK_ROUNDS`` (30 rounds = ~2.5h). Among those
     "revival" bets, fraction (by amountWei) on the Bull side.

  2. agreement_count
     At decision time (lock_at - 2s), count of {BTC, ETH, SOL} whose
     15-second log return is > 0 ("votes Bull"). Range {0,1,2,3}.

  3. late_bull_share_size
     Among bets placed in the last 30s before lock_at, fraction (by
     amountWei) on the Bull side.

  4. oracle_velocity_1800s_bps
     BNB Chainlink price change over the last 1800s window, expressed
     in basis points. Approximated using closed_rounds[ep].lockPrice
     vs closed_rounds[ep-6].lockPrice (6 rounds × 300s = 1800s).

Per-cohort distribution stats + Kolmogorov-Smirnov 2-sample test
between extension and union(CV5, holdout, ext_v2, fresh_oos).

Writes per-feature JSON + a markdown report.
"""
from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

REPO = Path(__file__).resolve().parents[1]
EXT_DIR = Path(r"C:\Users\zking\AppData\Local\Temp\ext\extended")

COHORTS = {
    "extension": (422298, 437561),
    "cv5":       (437562, 474086),
    "holdout":   (474880, 475311),
    "ext_v2":    (475312, 479952),
    "fresh_oos": (479953, 483191),
    "post_fresh":(483192, 999999),
}
POSITIVE_COHORTS = ("cv5", "holdout", "ext_v2", "fresh_oos", "post_fresh")

_DORMANT_LOOKBACK_ROUNDS = 30   # ~2.5 hours
_LATE_WINDOW_SECONDS = 30
_AGREEMENT_RETURN_WINDOW_S = 15
_AGREEMENT_DECISION_OFFSET_S = 2  # lock_at - 2 (matches p4c convention)
_ORACLE_LOOKBACK_ROUNDS = 6      # 6 * 300s = 1800s


def cohort_of(epoch: int) -> str | None:
    for name, (lo, hi) in COHORTS.items():
        if lo <= epoch <= hi:
            return name
    return None


def iter_rounds_chronologically() -> Iterator[dict]:
    """Stream both extended + canonical closed_rounds, dedup by epoch, sort."""
    seen: set[int] = set()
    all_rounds: list[dict] = []
    for path in (EXT_DIR / "closed_rounds.jsonl", REPO / "var" / "closed_rounds.jsonl"):
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                r = json.loads(ln)
                if r["epoch"] in seen:
                    continue
                if r.get("failed"):
                    continue
                seen.add(r["epoch"])
                all_rounds.append(r)
    all_rounds.sort(key=lambda r: r["epoch"])
    for r in all_rounds:
        yield r


def lock_at(round_rec: dict) -> int:
    """startAt + 300 (the round lock occurs 5 min after start)."""
    return int(round_rec["startAt"]) + 300


# ---------------------------------------------------------------------------
# Feature 1: dormant_revival_bull_share_size
# ---------------------------------------------------------------------------

def compute_dormant_revival(rounds: list[dict]) -> dict[int, float | None]:
    """Per epoch, returns the bull-share (by amountWei) of revival bets.

    A wallet is "revived" in round R if its last prior bet was in a round
    >= _DORMANT_LOOKBACK_ROUNDS rounds before R. Walks chronologically with
    strict no-look-ahead (computes the feature for R using only data from
    rounds prior to R).
    """
    last_bet_epoch: dict[str, int] = {}
    out: dict[int, float | None] = {}
    for r in rounds:
        ep = r["epoch"]
        bets = r.get("bets") or []
        # Detect revivals BEFORE updating last_bet_epoch
        rev_bull_amt = 0
        rev_total_amt = 0
        for b in bets:
            wallet = b.get("wallet")
            if not wallet:
                continue
            prior = last_bet_epoch.get(wallet)
            is_revival = (
                prior is None  # never seen — treat first-touch as "revival" too
                or (ep - prior) >= _DORMANT_LOOKBACK_ROUNDS
            )
            if is_revival:
                amt = int(b.get("amountWei") or 0)
                rev_total_amt += amt
                if b.get("position") == "Bull":
                    rev_bull_amt += amt
        # Compute feature
        if rev_total_amt > 0:
            out[ep] = rev_bull_amt / rev_total_amt
        else:
            out[ep] = None
        # Update history with THIS round's bettors AFTER computing feature
        for b in bets:
            w = b.get("wallet")
            if w:
                last_bet_epoch[w] = ep
    return out


# ---------------------------------------------------------------------------
# Feature 2: agreement_count (BTC/ETH/SOL 15s return votes at lock-2)
# ---------------------------------------------------------------------------

def load_kline_close_arrays(canonical: Path, ext: Path, epochs: set[int]) -> dict[int, list[float]]:
    """Per-epoch 1-second close-price list from canonical + extended files."""
    out: dict[int, list[float]] = {}
    for path in (ext, canonical):
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                r = json.loads(ln)
                ep = r["epoch"]
                if ep not in epochs:
                    continue
                if ep in out:
                    continue  # extended first; dedup
                kls = r.get("klines_1s") or []
                out[ep] = [k[4] for k in kls]
    return out


def ret_at_decision(closes: list[float], offset_decision: int, window: int) -> float | None:
    """log(close[lock-offset_decision] / close[lock-offset_decision-window])"""
    if len(closes) < offset_decision + window + 1:
        return None
    # closes are 1s spaced, length usually 300 (5min). Last element is
    # close at lock_at; closes[-1-offset_decision] is at lock_at - offset.
    idx_now = len(closes) - 1 - offset_decision
    idx_then = idx_now - window
    if idx_then < 0 or idx_now >= len(closes):
        return None
    p1 = closes[idx_now]
    p0 = closes[idx_then]
    if p0 <= 0 or p1 <= 0:
        return None
    return math.log(p1 / p0)


def compute_agreement_count(rounds: list[dict],
                             btc: dict[int, list[float]],
                             eth: dict[int, list[float]],
                             sol: dict[int, list[float]]) -> dict[int, int | None]:
    """Per epoch: count of assets with positive 15s return at lock-2."""
    out: dict[int, int | None] = {}
    for r in rounds:
        ep = r["epoch"]
        b_ret = ret_at_decision(btc.get(ep, []), _AGREEMENT_DECISION_OFFSET_S, _AGREEMENT_RETURN_WINDOW_S)
        e_ret = ret_at_decision(eth.get(ep, []), _AGREEMENT_DECISION_OFFSET_S, _AGREEMENT_RETURN_WINDOW_S)
        s_ret = ret_at_decision(sol.get(ep, []), _AGREEMENT_DECISION_OFFSET_S, _AGREEMENT_RETURN_WINDOW_S)
        rets = [b_ret, e_ret, s_ret]
        if any(x is None for x in rets):
            out[ep] = None
        else:
            out[ep] = sum(1 for x in rets if x > 0)
    return out


# ---------------------------------------------------------------------------
# Feature 3: late_bull_share_size
# ---------------------------------------------------------------------------

def compute_late_bull_share(rounds: list[dict]) -> dict[int, float | None]:
    """Bets in [lock_at - 30s, lock_at): fraction by amountWei on Bull."""
    out: dict[int, float | None] = {}
    for r in rounds:
        ep = r["epoch"]
        la = lock_at(r)
        bull_amt = 0
        total_amt = 0
        for b in r.get("bets") or []:
            ts = b.get("createdAt")
            if ts is None:
                continue
            if la - _LATE_WINDOW_SECONDS <= ts < la:
                amt = int(b.get("amountWei") or 0)
                total_amt += amt
                if b.get("position") == "Bull":
                    bull_amt += amt
        if total_amt > 0:
            out[ep] = bull_amt / total_amt
        else:
            out[ep] = None
    return out


# ---------------------------------------------------------------------------
# Feature 4: oracle_velocity_1800s_bps (BNB Chainlink price velocity proxy)
# ---------------------------------------------------------------------------

def compute_oracle_velocity_1800s_bps(rounds: list[dict]) -> dict[int, float | None]:
    """Velocity = (lockPrice[ep] - lockPrice[ep - 6]) / lockPrice[ep - 6] * 10000.

    Uses the round's recorded Chainlink BNB lockPrice as a per-epoch sample
    of the BNB/USD oracle price. Each round is 300s; 6 rounds = 1800s.
    Result in basis points (1 bp = 0.01%).
    """
    by_ep: dict[int, float] = {}
    for r in rounds:
        lp = r.get("lockPrice")
        if isinstance(lp, (int, float)) and lp > 0:
            by_ep[r["epoch"]] = float(lp)
    out: dict[int, float | None] = {}
    for r in rounds:
        ep = r["epoch"]
        if ep not in by_ep:
            out[ep] = None
            continue
        prior_ep = ep - _ORACLE_LOOKBACK_ROUNDS
        if prior_ep not in by_ep:
            out[ep] = None
            continue
        p_now = by_ep[ep]
        p_then = by_ep[prior_ep]
        out[ep] = (p_now - p_then) / p_then * 10000.0
    return out


# ---------------------------------------------------------------------------
# Aggregation + KS test
# ---------------------------------------------------------------------------

def _percentiles(values: list[float], pcts: list[float]) -> dict[str, float]:
    if not values:
        return {f"p{int(p)}": float("nan") for p in pcts}
    s = sorted(values)
    out = {}
    for p in pcts:
        idx = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
        out[f"p{int(p)}"] = s[idx]
    return out


def stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        **_percentiles(values, [10, 50, 90]),
    }


def ks_2samp(a: list[float], b: list[float]) -> dict[str, float]:
    """Two-sample Kolmogorov-Smirnov test (vanilla; no scipy needed).

    Returns the test statistic D and an approximate two-sided p-value via
    the asymptotic Kolmogorov distribution. Adequate for n_a, n_b > 100.
    """
    if not a or not b:
        return {"D": float("nan"), "p_value": float("nan"), "n_a": len(a), "n_b": len(b)}
    a_s = sorted(a); b_s = sorted(b)
    n_a, n_b = len(a_s), len(b_s)
    i = j = 0
    cdf_a = cdf_b = 0.0
    D = 0.0
    while i < n_a and j < n_b:
        if a_s[i] <= b_s[j]:
            i += 1
            cdf_a = i / n_a
        else:
            j += 1
            cdf_b = j / n_b
        D = max(D, abs(cdf_a - cdf_b))
    # Asymptotic p-value approximation (Kolmogorov)
    en = math.sqrt(n_a * n_b / (n_a + n_b))
    lam = (en + 0.12 + 0.11 / en) * D
    # P-value = Q_KS(lam) = 2 * sum_{k=1..} (-1)^{k-1} exp(-2 k^2 lam^2)
    s = 0.0
    for k in range(1, 101):
        s += (-1) ** (k - 1) * math.exp(-2.0 * k * k * lam * lam)
    p = max(0.0, min(1.0, 2.0 * s))
    return {"D": D, "p_value": p, "n_a": n_a, "n_b": n_b}


def main() -> None:
    print("loading rounds chronologically...")
    rounds = list(iter_rounds_chronologically())
    print(f"  total rounds: {len(rounds)}")

    print("computing dormant_revival_bull_share_size (chronological pass)...")
    drv = compute_dormant_revival(rounds)

    print("computing late_bull_share_size...")
    lbs = compute_late_bull_share(rounds)

    print("computing oracle_velocity_1800s_bps...")
    ovb = compute_oracle_velocity_1800s_bps(rounds)

    print("loading klines for agreement_count...")
    epochs_needed = {r["epoch"] for r in rounds if cohort_of(r["epoch"])}
    btc = load_kline_close_arrays(
        REPO / "var" / "btc_spot_prices.jsonl",
        EXT_DIR / "btc_spot_prices.jsonl",
        epochs_needed,
    )
    eth = load_kline_close_arrays(
        REPO / "var" / "eth_spot_prices.jsonl",
        EXT_DIR / "eth_spot_prices.jsonl",
        epochs_needed,
    )
    sol = load_kline_close_arrays(
        REPO / "var" / "sol_spot_prices.jsonl",
        EXT_DIR / "sol_spot_prices.jsonl",
        epochs_needed,
    )
    print(f"  BTC klines: {len(btc)}  ETH: {len(eth)}  SOL: {len(sol)}")

    print("computing agreement_count...")
    agc = compute_agreement_count(rounds, btc, eth, sol)

    print("aggregating per cohort...")
    per_cohort: dict[str, dict] = {}
    for name in COHORTS:
        per_cohort[name] = {
            "epoch_range": COHORTS[name],
            "dormant_revival_bull_share": [],
            "agreement_count": [],
            "late_bull_share": [],
            "oracle_velocity_bps": [],
        }
    for r in rounds:
        ep = r["epoch"]
        c = cohort_of(ep)
        if c is None:
            continue
        v = drv.get(ep)
        if v is not None: per_cohort[c]["dormant_revival_bull_share"].append(v)
        v = agc.get(ep)
        if v is not None: per_cohort[c]["agreement_count"].append(float(v))
        v = lbs.get(ep)
        if v is not None: per_cohort[c]["late_bull_share"].append(v)
        v = ovb.get(ep)
        if v is not None: per_cohort[c]["oracle_velocity_bps"].append(v)

    print("computing per-cohort stats + KS tests...")
    result = {"features": {}, "cohort_sizes": {}}
    feat_names = [
        "dormant_revival_bull_share",
        "agreement_count",
        "late_bull_share",
        "oracle_velocity_bps",
    ]
    for fname in feat_names:
        result["features"][fname] = {"by_cohort": {}}
        # Per-cohort stats
        for cname in COHORTS:
            vals = per_cohort[cname][fname]
            result["features"][fname]["by_cohort"][cname] = stats(vals)
        # KS: extension vs union(positives)
        ext_vals = per_cohort["extension"][fname]
        pos_vals = []
        for pname in POSITIVE_COHORTS:
            pos_vals.extend(per_cohort[pname][fname])
        result["features"][fname]["ks_extension_vs_positives"] = ks_2samp(ext_vals, pos_vals)
        # Also: KS extension vs CV5 alone (the largest individual positive cohort)
        result["features"][fname]["ks_extension_vs_cv5"] = ks_2samp(
            ext_vals, per_cohort["cv5"][fname],
        )

    for c in COHORTS:
        result["cohort_sizes"][c] = sum(1 for r in rounds if cohort_of(r["epoch"]) == c)

    out_path = REPO / "var" / "strategy_review" / "regime_step2_data.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
