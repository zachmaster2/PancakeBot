"""Edge-decay post-mortem (2026-06-11). READ-ONLY analysis.

Answers, end-to-end from the synced dataset (51,248 rounds through 488832):
  A. WHEN the canonical edge decayed (rolling-window WR/PnL + CUSUM
     inflection over the RISK-FREE canonical signal stream, cross-referenced
     against BTC volatility, pool size, and gate fire-rate).
  B. HOW it decayed (signal-strength calibration per era, primary vs
     regime-2 split, fire-rate drift).
  C. Regime shift vs model-prior drift (per-feature per-cohort
     signal-to-noise: did anything EMERGE on the dead cohorts, or did the
     old features just collapse with nothing replacing them?).
  D. Unexplored-data check (pool-state features computable today;
     orderbook/funding flagged as new-collection).

Method caveats (adversarial-review verified, 2026-06-11 panel):
  - SnR = r*sqrt(n) is a z-score; cohorts differ ~30x in n. COMPARE EFFECT
    SIZES via feature_r.csv (r units), not feature_snr.csv, across cohorts.
  - The CUSUM peak is referenced to the in-sample golden mean and is
    structurally confined to the golden era; its date is NOT a supported
    decay onset (null p=0.28, 20k sims). The only statistically supported
    break is the dead-era boundary (epoch 484409, 2026-05-26).
  - Flat-stake economics are the infinitesimal-stake limit: a real 1 BNB
    stake in the ~3 BNB median bet-round pool self-dilutes the ~1.84
    multiple to ~1.5 (breakeven ~65%). Era-RELATIVE comparisons stand.

Method notes:
  - The bet stream is the CANONICAL pipeline replayed with
    ``bankroll_tracker=None`` (risk gates are a no-op by design), so the
    decay analysis sees the raw signal, not breaker/cooldown suppression.
  - Flat-stake settlement (stake=1, 3% treasury fee on final pools, no
    gas): isolates signal quality from sizing and execution costs.
  - Eras for B/C: golden = 437562..479952 (CV5+holdout+ext_v2, the
    validated edge), fade = 479953..484408, dead = 484409..488832 (the
    permutation-CONSISTENT_WITH_LUCK cohorts).

Outputs (var/strategy_review/post_mortem_2026_06_11/):
  rolling_wr.png, calibration.png, feature_snr.csv, regime_correlates.csv,
  findings.json
Findings document: research/post_mortem_2026_06_11_findings.md (committed).

Run:  cd <repo> && .venv/Scripts/python.exe research/post_mortem_2026_06_11.py
"""
from __future__ import annotations

import csv
import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import research.in_process_runner as ipr  # noqa: E402
from pancakebot.config import load_strategy_config_from_dict  # noqa: E402
from pancakebot.constants import BNB_WEI  # noqa: E402
from pancakebot.pool_amounts import compute_pool_amounts_wei  # noqa: E402
from pancakebot.strategy.momentum_gate import MomentumGateConfig  # noqa: E402
from pancakebot.strategy.momentum_pipeline import (  # noqa: E402
    MomentumOnlyPipeline,
    _pools_from_bets,
)

OUT = REPO / "var" / "strategy_review" / "post_mortem_2026_06_11"
CUTOFF = 2
LOOKBACKS = (3, 7, 15)
FEE = 0.03

ERAS = [
    ("golden", 437562, 479952),   # CV5 + holdout + ext_v2 (validated edge)
    ("fade", 479953, 484408),     # fresh_oos + post_fresh
    ("dead", 484409, 488832),     # latest + vm_live_era (consistent-with-luck)
]
COHORTS = [
    ("f1-f5_cv5", 437562, 474086),
    ("holdout+gap", 474087, 475311),
    ("ext_v2", 475312, 479952),
    ("fresh_oos", 479953, 483191),
    ("post_fresh", 483192, 484408),
    ("latest", 484409, 487686),
    ("vm_live_era", 487687, 488832),
]


def era_of(epoch: int) -> str:
    for name, lo, hi in ERAS:
        if lo <= epoch <= hi:
            return name
    return "other"


def cohort_of(epoch: int) -> str:
    for name, lo, hi in COHORTS:
        if lo <= epoch <= hi:
            return name
    return "other"


# ---------------------------------------------------------------------------
# PART 0 — load (mirrors research/post_cv5_to_current_step10b_2026_05_26)
# ---------------------------------------------------------------------------

def load_everything():
    print("--- loading rounds ---", flush=True)
    all_rounds = ipr._load_all_rounds(use_extended_data=False)
    all_rounds = [r for r in all_rounds if r.epoch >= ERAS[0][1]]
    print(f"  {len(all_rounds)} rounds [{all_rounds[0].epoch}..{all_rounds[-1].epoch}]")

    max_lb = max(LOOKBACKS)
    earliest_offset = CUTOFF + max_lb + 1
    latest_offset = CUTOFF + 1
    print("--- loading klines ---", flush=True)
    uni = {}
    for sym, path in (("btc", ipr._BTC_KLINES_PATH), ("eth", ipr._ETH_KLINES_PATH),
                      ("sol", ipr._SOL_KLINES_PATH)):
        uni[sym] = ipr._load_klines_unified(
            path, earliest_offset=earliest_offset, latest_offset=latest_offset)
        print(f"  {sym}={len(uni[sym])}")

    sliced = {
        sym: {
            ep: ipr._slice_per_entry(
                kl, kline_cutoff_seconds=CUTOFF, max_lookback=max_lb,
                earliest_offset=earliest_offset)
            for ep, kl in uni[sym].items()
        }
        for sym in uni
    }
    return all_rounds, sliced


# ---------------------------------------------------------------------------
# PART 1 — canonical risk-free signal stream
# ---------------------------------------------------------------------------

def replay_canonical(all_rounds, sliced) -> list[dict]:
    strategy_cfg = load_strategy_config_from_dict({})
    gate_cfg = MomentumGateConfig(
        enabled=True, bnb_symbol="BNB-USDT", btc_symbol="BTC-USDT",
        eth_symbol="ETH-USDT", sol_symbol="SOL-USDT",
        kline_cutoff_seconds=CUTOFF,
        mtf_lookbacks=strategy_cfg.gate.mtf_lookbacks,
        mtf_min_return_threshold=strategy_cfg.gate.mtf_min_return_threshold,
    )
    pipeline = MomentumOnlyPipeline(
        config=gate_cfg, strategy_config=strategy_cfg, gate=None,
        kline_cutoff_seconds=CUTOFF, pool_cutoff_seconds=6,
        min_bet_amount_bnb=0.001, treasury_fee_fraction=FEE,
        bankroll_tracker=None,  # risk gates OFF: pure signal stream
    )
    pipeline.refresh_btc_klines(btc_klines_by_epoch=sliced["btc"])
    pipeline.refresh_eth_klines(eth_klines_by_epoch=sliced["eth"])
    pipeline.refresh_sol_klines(sol_klines_by_epoch=sliced["sol"])
    pipeline.refresh_bnb_klines(bnb_klines_by_epoch={})

    bets: list[dict] = []
    t0 = time.time()
    for r in all_rounds:
        d = pipeline.decide_open_round(round_t=r)
        if d.action != "BET":
            continue
        winner = r.position
        if winner not in ("Bull", "Bear"):
            continue
        pools = compute_pool_amounts_wei(bets=r.bets)
        bull = pools.bull_wei / BNB_WEI
        bear = pools.bear_wei / BNB_WEI
        total = bull + bear
        side_pool = bull if d.bet_side == "Bull" else bear
        if side_pool <= 0 or total <= 0:
            continue
        win = d.bet_side == winner
        payout_mult = total * (1.0 - FEE) / side_pool
        pnl_flat = (payout_mult - 1.0) if win else -1.0
        c_bull, c_bear = _pools_from_bets(r, int(r.lock_at) - 6)
        with_crowd = (
            (d.bet_side == "Bull") == (c_bull >= c_bear)
            if (c_bull + c_bear) > 0 else None)
        bets.append(dict(
            epoch=int(r.epoch), start_at=int(r.start_at), side=d.bet_side,
            win=bool(win), pnl_flat=float(pnl_flat),
            pool_total=float(total), payout_mult=float(payout_mult),
            with_crowd=with_crowd,
        ))
    print(f"--- replay: {len(bets)} risk-free bets in {time.time()-t0:.1f}s ---")
    return bets


# ---------------------------------------------------------------------------
# PART 2 — per-round features (numpy, all rounds)
# ---------------------------------------------------------------------------

FEATURE_NAMES: list[str] = []


def compute_features(all_rounds, sliced) -> dict[int, dict]:
    """Per-round features; signed features are direction-aligned so
    corr(feature, outcome_signed) is the signal measure."""
    feats: dict[int, dict] = {}
    lock_by_epoch = {}
    prev_lock = None
    for r in all_rounds:
        f: dict[str, float] = {}
        # kline-derived (per asset): signed returns at the gate's lookbacks
        ok = True
        for sym in ("btc", "eth", "sol"):
            entry = sliced[sym].get(r.epoch)
            if not entry or len(entry) < max(LOOKBACKS) + 1:
                ok = False
                break
            closes = np.array([float(k[4]) for k in entry])
            for lb in LOOKBACKS:
                f[f"{sym}_r{lb}"] = float(closes[-1] / closes[-1 - lb] - 1.0)
            rets1s = np.diff(closes) / closes[:-1]
            f[f"{sym}_vol1s"] = float(np.std(rets1s))
        if not ok:
            continue
        # cross-asset agreement (signed: sum of signs)
        for lb in LOOKBACKS:
            f[f"agree_sign_r{lb}"] = float(
                np.sign(f[f"btc_r{lb}"]) + np.sign(f[f"eth_r{lb}"])
                + np.sign(f[f"sol_r{lb}"]))
        f["btc_strength"] = float(min(abs(f[f"btc_r{lb}"]) for lb in LOOKBACKS))
        f["btc_mtf_agree"] = float(
            len({np.sign(f[f"btc_r{lb}"]) for lb in LOOKBACKS}) == 1
            and np.sign(f["btc_r3"]) != 0)
        # BNB momentum from round lock prices (5-min spacing)
        lp = float(r.lock_price) if r.lock_price else None
        f["bnb_ret_prev_round"] = (
            (lp / prev_lock - 1.0) if (lp and prev_lock) else 0.0)
        if lp:
            prev_lock = lp
        # pool features at FINAL pools (the unexplored family)
        pools = compute_pool_amounts_wei(bets=r.bets)
        bull = pools.bull_wei / BNB_WEI
        bear = pools.bear_wei / BNB_WEI
        total = bull + bear
        f["pool_final_total"] = float(total)
        f["pool_final_imbalance"] = float((bull - bear) / total) if total > 0 else 0.0
        # ACTIONABLE variant: pools as the bot sees them at the decision
        # cutoff (lock - 6s), via the canonical reconstruction helper.
        c_bull, c_bear = _pools_from_bets(r, int(r.lock_at) - 6)
        c_total = c_bull + c_bear
        f["pool_cutoff_total"] = float(c_total)
        f["pool_cutoff_imbalance"] = (
            float((c_bull - c_bear) / c_total) if c_total > 0 else 0.0)
        f["pool_late_share"] = (
            float(1.0 - c_total / total) if total > 0 else 0.0)
        f["pool_n_bets"] = float(len(r.bets))
        if r.bets and total > 0:
            amounts = sorted((float(b.amount_wei) / BNB_WEI for b in r.bets),
                             reverse=True)
            f["pool_top_bet_frac"] = float(amounts[0] / total)
        else:
            f["pool_top_bet_frac"] = 0.0
        f["hour_utc"] = float(
            datetime.fromtimestamp(int(r.start_at), tz=timezone.utc).hour)
        # outcome
        winner = r.position
        if winner not in ("Bull", "Bear"):
            continue
        f["outcome_signed"] = 1.0 if winner == "Bull" else -1.0
        feats[int(r.epoch)] = f
    global FEATURE_NAMES
    FEATURE_NAMES = [k for k in next(iter(feats.values())) if k != "outcome_signed"]
    print(f"--- features: {len(feats)} rounds x {len(FEATURE_NAMES)} features ---")
    return feats


# ---------------------------------------------------------------------------
# PART 3 — analyses
# ---------------------------------------------------------------------------

def rolling_wr(bets, window=100, step=10):
    wins = np.array([1.0 if b["win"] else 0.0 for b in bets])
    pnls = np.array([b["pnl_flat"] for b in bets])
    rows = []
    for i in range(0, max(1, len(bets) - window + 1), step):
        seg = slice(i, i + window)
        mid = bets[min(i + window // 2, len(bets) - 1)]
        rows.append(dict(
            bet_idx=i + window // 2, epoch=mid["epoch"],
            date=datetime.fromtimestamp(mid["start_at"], tz=timezone.utc)
                 .strftime("%Y-%m-%d"),
            wr=float(wins[seg].mean()), pnl=float(pnls[seg].mean()),
        ))
    return rows


def cusum_inflection(bets):
    """CUSUM of (win - golden-era mean); the maximum marks decay onset."""
    golden = [b for b in bets if era_of(b["epoch"]) == "golden"]
    p0 = float(np.mean([b["win"] for b in golden]))
    s, peak_s, peak_i = 0.0, -1e9, 0
    for i, b in enumerate(bets):
        s += (1.0 if b["win"] else 0.0) - p0
        if s > peak_s:
            peak_s, peak_i = s, i
    b = bets[peak_i]
    return dict(
        golden_wr=p0, peak_bet_idx=peak_i, peak_epoch=b["epoch"],
        peak_date=datetime.fromtimestamp(b["start_at"], tz=timezone.utc)
                  .strftime("%Y-%m-%d"),
        bets_after_peak=len(bets) - peak_i - 1,
        wr_after_peak=float(np.mean([x["win"] for x in bets[peak_i + 1:]]))
        if peak_i + 1 < len(bets) else None,
    )


def calibration(bets, feats):
    """Signal-strength terciles x era -> WR; fire-rate per era; regime split."""
    rows = []
    by_era: dict[str, list] = defaultdict(list)
    for b in bets:
        f = feats.get(b["epoch"])
        if f is None:
            continue
        by_era[era_of(b["epoch"])].append((f["btc_strength"], b["win"],
                                           f["btc_mtf_agree"]))
    for era, vals in by_era.items():
        if len(vals) < 30:
            continue
        strengths = np.array([v[0] for v in vals])
        wins = np.array([1.0 if v[1] else 0.0 for v in vals])
        primary = np.array([v[2] > 0 for v in vals])
        terc = np.quantile(strengths, [1 / 3, 2 / 3])
        lo = wins[strengths <= terc[0]]
        mid = wins[(strengths > terc[0]) & (strengths <= terc[1])]
        hi = wins[strengths > terc[1]]
        rows.append(dict(
            era=era, n=len(vals),
            wr_all=float(wins.mean()),
            wr_weak_third=float(lo.mean()), wr_mid_third=float(mid.mean()),
            wr_strong_third=float(hi.mean()),
            wr_primary=float(wins[primary].mean()) if primary.any() else None,
            n_primary=int(primary.sum()),
            wr_regime2=float(wins[~primary].mean()) if (~primary).any() else None,
            n_regime2=int((~primary).sum()),
        ))
    return rows


def fire_rates(bets, all_rounds):
    n_rounds = defaultdict(int)
    n_bets = defaultdict(int)
    for r in all_rounds:
        n_rounds[era_of(int(r.epoch))] += 1
    for b in bets:
        n_bets[era_of(b["epoch"])] += 1
    return {e: dict(rounds=n_rounds[e], bets=n_bets[e],
                    fire_rate=n_bets[e] / n_rounds[e] if n_rounds[e] else 0.0)
            for e in n_rounds if e != "other"}


def era_bet_economics(bets):
    """Per-era payout + crowd-alignment stats: the crowding test. If the
    payout multiple on our side compressed era-over-era while alignment
    with the cutoff-pool majority rose, the crowd is pricing the same
    signal in."""
    by_era = defaultdict(list)
    for b in bets:
        by_era[era_of(b["epoch"])].append(b)
    out = {}
    for era, bs in by_era.items():
        if era == "other" or not bs:
            continue
        payouts = np.array([b["payout_mult"] for b in bs])
        wins = np.array([b["win"] for b in bs])
        crowd = [b["with_crowd"] for b in bs if b["with_crowd"] is not None]
        out[era] = dict(
            n=len(bs),
            mean_payout=float(payouts.mean()),
            mean_payout_on_wins=float(payouts[wins].mean()) if wins.any() else None,
            mean_pnl_flat=float(np.mean([b["pnl_flat"] for b in bs])),
            frac_with_crowd=float(np.mean(crowd)) if crowd else None,
            breakeven_wr=float(np.mean(1.0 / payouts)),
        )
    return out


def feature_snr(feats):
    """Per-feature per-cohort corr(feature, outcome_signed) + SnR."""
    by_cohort: dict[str, list[dict]] = defaultdict(list)
    for ep, f in feats.items():
        by_cohort[cohort_of(ep)].append(f)
    out = []
    for name in FEATURE_NAMES:
        row = {"feature": name}
        for coh, _, _ in [(c[0], c[1], c[2]) for c in COHORTS]:
            fs = by_cohort.get(coh, [])
            if len(fs) < 200:
                row[coh] = None
                continue
            x = np.array([f[name] for f in fs])
            y = np.array([f["outcome_signed"] for f in fs])
            if np.std(x) == 0:
                row[coh] = 0.0
                continue
            r = float(np.corrcoef(x, y)[0, 1])
            row[coh] = round(r * math.sqrt(len(fs)), 2)  # SnR = r*sqrt(n)
        out.append(row)
    return out


def regime_correlates(bets, feats, window=100, step=10):
    """Rolling WR vs same-window medians of vol / pool size."""
    rows = []
    for i in range(0, max(1, len(bets) - window + 1), step):
        seg = bets[i:i + window]
        wr = float(np.mean([b["win"] for b in seg]))
        vols, pools_ = [], []
        for b in seg:
            f = feats.get(b["epoch"])
            if f:
                vols.append(f["btc_vol1s"])
                pools_.append(f["pool_final_total"])
        rows.append((wr, float(np.median(vols)), float(np.median(pools_))))
    arr = np.array(rows)
    return dict(
        n_windows=len(rows),
        corr_wr_vs_btc_vol=float(np.corrcoef(arr[:, 0], arr[:, 1])[0, 1]),
        corr_wr_vs_pool_total=float(np.corrcoef(arr[:, 0], arr[:, 2])[0, 1]),
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    all_rounds, sliced = load_everything()
    bets = replay_canonical(all_rounds, sliced)
    feats = compute_features(all_rounds, sliced)

    # A — rolling + inflection + correlates
    roll = rolling_wr(bets)
    infl = cusum_inflection(bets)
    correlates = regime_correlates(bets, feats)
    fire = fire_rates(bets, all_rounds)
    economics = era_bet_economics(bets)

    fig, ax = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    xs = [r["bet_idx"] for r in roll]
    ax[0].plot(xs, [r["wr"] for r in roll], lw=1.2)
    ax[0].axhline(0.5, color="grey", ls="--", lw=0.8)
    ax[0].axvline(infl["peak_bet_idx"], color="red", ls=":",
                  label=f"CUSUM peak {infl['peak_date']} (ep {infl['peak_epoch']})")
    ax[0].set_ylabel("rolling WR (100 bets)")
    ax[0].legend()
    ax[1].plot(xs, np.array([r["pnl"] for r in roll]), lw=1.2, color="darkorange")
    ax[1].axhline(0.0, color="grey", ls="--", lw=0.8)
    ax[1].set_ylabel("rolling mean flat PnL / bet")
    ax[1].set_xlabel("bet index (risk-free canonical stream)")
    tick_idx = list(range(0, len(roll), max(1, len(roll) // 10)))
    ax[1].set_xticks([roll[i]["bet_idx"] for i in tick_idx])
    ax[1].set_xticklabels([roll[i]["date"] for i in tick_idx], rotation=45,
                          fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "rolling_wr.png", dpi=120)

    # B — calibration
    cal = calibration(bets, feats)
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for row in cal:
        ax2.plot(["weak", "mid", "strong"],
                 [row["wr_weak_third"], row["wr_mid_third"], row["wr_strong_third"]],
                 marker="o", label=f"{row['era']} (n={row['n']})")
    ax2.axhline(0.5, color="grey", ls="--", lw=0.8)
    ax2.set_ylabel("WR")
    ax2.set_title("WR by signal-strength tercile, per era")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(OUT / "calibration.png", dpi=120)

    # C/D — feature SnR table
    snr = feature_snr(feats)
    with open(OUT / "feature_snr.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["feature"] + [c[0] for c in COHORTS])
        w.writeheader()
        w.writerows(snr)
    # Effect-size units (r = SnR / sqrt(n_cohort)) — the cross-cohort
    # comparable table; SnR alone conflates effect with cohort size.
    coh_n = defaultdict(int)
    for ep in feats:
        coh_n[cohort_of(ep)] += 1
    r_rows = []
    for row in snr:
        rr = {"feature": row["feature"]}
        for c, _, _ in COHORTS:
            v = row.get(c)
            rr[c] = (round(v / math.sqrt(coh_n[c]), 4)
                     if (v is not None and coh_n[c] > 0) else None)
        r_rows.append(rr)
    with open(OUT / "feature_r.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["feature"] + [c[0] for c in COHORTS])
        w.writeheader()
        w.writerows(r_rows)

    findings = dict(
        n_rounds=len(all_rounds), n_riskfree_bets=len(bets),
        inflection=infl, fire_rates=fire, calibration=cal,
        regime_correlates=correlates,
        economics=economics,
        feature_snr=snr,
        feature_r=r_rows,
        rolling=roll,
        bets=bets,
    )
    (OUT / "findings.json").write_text(
        json.dumps(findings, indent=2), encoding="utf-8")

    # console digest
    print("\n=== A: inflection ===")
    print(json.dumps(infl, indent=2))
    print("=== fire rates ===")
    print(json.dumps(fire, indent=2))
    print("=== B: calibration ===")
    for row in cal:
        print(row)
    print("=== regime correlates ===")
    print(json.dumps(correlates, indent=2))
    print("=== era economics (crowding test) ===")
    print(json.dumps(economics, indent=2))
    print("=== C/D: feature SnR (|SnR|>3 in any cohort) ===")
    _always = {"pool_cutoff_imbalance", "pool_cutoff_total", "pool_late_share",
               "bnb_ret_prev_round", "pool_final_imbalance"}
    for row in snr:
        vals = [v for k, v in row.items() if k != "feature" and v is not None]
        if row["feature"] in _always or (vals and max(abs(v) for v in vals) >= 3.0):
            print(row)
    print(f"\n[done] artifacts in {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
