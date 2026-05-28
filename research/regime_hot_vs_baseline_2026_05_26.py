"""Test user hypothesis: does the late-Jan-to-early-Mar hot window coincide
with a multi-asset bear run?

Three analyses:
  1. For each of BTC/ETH/SOL/BNB, compute net %-change + max drawdown +
     annualized vol over: hot 7d/14d/30d, extension cohort, full range,
     recent quiet 30d.
  2. Of the 601 bets in the 30d hot window, BULL vs BEAR breakdown + WR.
  3. Coarse correlation: bucket each calendar day in full range by BTC
     daily %-change quintile; report mean per-bet PnL per quintile.

Uses streaming JSONL reads for kline price tracking (no full kline load)
+ cached trades.csv from Step 11 bonus.
"""
from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import research.in_process_runner as ipr
# Override extended-data paths so _load_all_rounds picks up extension cohort.
EXT_DIR = Path(r"C:\Users\zking\AppData\Local\Temp\ext\extended")
ipr._EXT_CLOSED_ROUNDS_PATH = EXT_DIR / "closed_rounds.jsonl"

KLINE_PATHS = {
    "BTC": REPO / "var" / "btc_spot_prices.jsonl",
    "ETH": REPO / "var" / "eth_spot_prices.jsonl",
    "SOL": REPO / "var" / "sol_spot_prices.jsonl",
    "BNB": REPO / "var" / "bnb_spot_prices.jsonl",
}
EXT_KLINE_PATHS = {
    "BTC": Path(r"C:\Users\zking\AppData\Local\Temp\ext\extended\btc_spot_prices.jsonl"),
    "ETH": Path(r"C:\Users\zking\AppData\Local\Temp\ext\extended\eth_spot_prices.jsonl"),
    "SOL": Path(r"C:\Users\zking\AppData\Local\Temp\ext\extended\sol_spot_prices.jsonl"),
    "BNB": Path(r"C:\Users\zking\AppData\Local\Temp\ext\extended\bnb_spot_prices.jsonl"),
}

BONUS_TRADES_5BNB = Path(r"C:\Users\zking\AppData\Local\Temp\step11_k3ub14zf\bonus_5bnb\trades.csv")
BONUS_TRADES_50BNB = Path(r"C:\Users\zking\AppData\Local\Temp\step11_k3ub14zf\bonus_50bnb\trades.csv")


def stream_last_close(path: Path) -> dict[int, float]:
    """Stream-read JSONL, extract {epoch: last_close_price}. Skips records
    without klines_1s or with errors.
    """
    out: dict[int, float] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("error") or rec.get("klines_1s") is None:
                continue
            kl = rec["klines_1s"]
            if not kl:
                continue
            last_close = kl[-1][4]
            if last_close is None or float(last_close) <= 0:
                continue
            out[int(rec["epoch"])] = float(last_close)
    return out


def fmt_dt(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def fmt_date(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


def compute_window_metrics(prices_by_ts: list[tuple[int, float]]) -> dict:
    """prices_by_ts is sorted (ts, price). Returns net %-change, max
    drawdown, max run-up, and annualized vol from 5-min log returns
    (each round is 300s).
    """
    if len(prices_by_ts) < 2:
        return {"n": len(prices_by_ts), "net_pct": None, "max_dd_pct": None,
                "max_ru_pct": None, "vol_ann_pct": None}
    prices = [p for _, p in prices_by_ts]
    start = prices[0]; end = prices[-1]
    net_pct = (end - start) / start * 100.0
    # Max drawdown from peak; max run-up from trough
    peak = prices[0]; trough = prices[0]
    max_dd = 0.0; max_ru = 0.0
    for p in prices:
        if p > peak: peak = p
        if p < trough: trough = p
        dd = (peak - p) / peak * 100.0
        ru = (p - trough) / trough * 100.0
        if dd > max_dd: max_dd = dd
        if ru > max_ru: max_ru = ru
    # Annualized vol from 5-min log returns
    log_returns = []
    for i in range(1, len(prices)):
        if prices[i-1] > 0:
            log_returns.append(math.log(prices[i] / prices[i-1]))
    if len(log_returns) > 1:
        sd = statistics.stdev(log_returns)
        # 288 5-min periods/day * 365 days/yr
        vol_ann = sd * math.sqrt(288 * 365) * 100.0
    else:
        vol_ann = 0.0
    return {"n": len(prices_by_ts), "net_pct": net_pct,
            "max_dd_pct": -max_dd, "max_ru_pct": max_ru,
            "vol_ann_pct": vol_ann,
            "start_price": start, "end_price": end}


def main():
    print("--- loading rounds for epoch -> start_at lookup ---", flush=True)
    all_rounds = ipr._load_all_rounds(use_extended_data=True)
    epoch_to_ts = {int(r.epoch): int(r.start_at) for r in all_rounds}
    ts_to_epoch = {ts: ep for ep, ts in epoch_to_ts.items()}
    print(f"  {len(epoch_to_ts)} rounds; epoch range [{min(epoch_to_ts)}..{max(epoch_to_ts)}]", flush=True)

    print("--- streaming last-close per round from kline files (canonical + extended) ---", flush=True)
    asset_prices: dict[str, dict[int, float]] = {}
    for asset, path in KLINE_PATHS.items():
        print(f"  loading {asset}...", flush=True)
        merged: dict[int, float] = {}
        # Extended first (older epochs), canonical wins on overlap
        ext_path = EXT_KLINE_PATHS.get(asset)
        if ext_path and ext_path.exists():
            ext = stream_last_close(ext_path)
            merged.update(ext)
            print(f"    ext: {len(ext)} epochs", flush=True)
        if path.exists():
            can = stream_last_close(path)
            merged.update(can)
            print(f"    canonical: {len(can)} epochs", flush=True)
        asset_prices[asset] = merged
        print(f"    {asset} merged: {len(merged)} epochs", flush=True)

    # Build (ts, price) timelines per asset
    asset_timeline: dict[str, list[tuple[int, float]]] = {}
    for asset, ep_prices in asset_prices.items():
        tl = []
        for ep, price in ep_prices.items():
            ts = epoch_to_ts.get(ep)
            if ts is not None:
                tl.append((ts, price))
        tl.sort()
        asset_timeline[asset] = tl

    # Define windows
    def slice_window(asset: str, ts_start: int, ts_end: int) -> list[tuple[int, float]]:
        tl = asset_timeline.get(asset, [])
        return [(ts, p) for ts, p in tl if ts_start <= ts <= ts_end]

    # Window boundaries (UTC unix)
    def utc_unix(date_str: str) -> int:
        return int(datetime.fromisoformat(date_str.replace("Z", "+00:00")).timestamp())

    full_ts_min = min(ts_to_epoch.keys())
    full_ts_max = max(ts_to_epoch.keys())
    recent_ts_start = full_ts_max - 30 * 86400

    windows = [
        ("Hot 7d",       utc_unix("2026-01-30 18:32+00:00"), utc_unix("2026-02-06 18:28+00:00")),
        ("Hot 14d",      utc_unix("2026-01-30 16:30+00:00"), utc_unix("2026-02-13 16:21+00:00")),
        ("Hot 30d",      utc_unix("2026-02-03 18:46+00:00"), utc_unix("2026-03-05 16:44+00:00")),
        ("Extension",    epoch_to_ts.get(422298, 0),         epoch_to_ts.get(437561, 0)),
        ("Full range",   full_ts_min,                        full_ts_max),
        ("Recent 30d",   recent_ts_start,                    full_ts_max),
    ]

    print("\n=== Q1: Market regime per window per asset ===")
    header = f"{'Window':>12s} {'Range':>26s}  "
    for asset in ("BTC", "ETH", "SOL", "BNB"):
        header += f"{asset+'_pct':>9s} {asset+'_dd':>8s} {asset+'_ru':>7s} {asset+'_vol':>8s}  "
    print(header)
    for name, ts_s, ts_e in windows:
        date_range = f"{fmt_date(ts_s)}..{fmt_date(ts_e)}"
        line = f"{name:>12s} {date_range:>26s}  "
        for asset in ("BTC", "ETH", "SOL", "BNB"):
            sl = slice_window(asset, ts_s, ts_e)
            m = compute_window_metrics(sl)
            if m["net_pct"] is None:
                line += f"{'n/a':>9s} {'n/a':>8s} {'n/a':>7s} {'n/a':>8s}  "
            else:
                line += (f"{m['net_pct']:+8.2f}% {m['max_dd_pct']:+7.2f}% "
                         f"{m['max_ru_pct']:+6.2f}% {m['vol_ann_pct']:>7.1f}%  ")
        print(line)

    # ----- Q2: Bet direction in 30d hot window -----
    print("\n=== Q2: 30d hot window bet direction breakdown ===")
    hot30_s = utc_unix("2026-02-03 18:46+00:00")
    hot30_e = utc_unix("2026-03-05 16:44+00:00")

    def load_bets(trades_csv: Path) -> list[dict]:
        out = []
        with open(trades_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("action") != "BET":
                    continue
                ep = int(row["epoch"])
                ts = epoch_to_ts.get(ep)
                if ts is None: continue
                out.append({
                    "epoch": ep, "ts": ts,
                    "side": row["direction"].strip(),
                    "profit": float(row["profit_bnb"]),
                    "win": float(row["profit_bnb"]) > 0,
                })
        return out

    bets_50 = load_bets(BONUS_TRADES_50BNB)
    in_window = [b for b in bets_50 if hot30_s <= b["ts"] <= hot30_e]
    overall = bets_50

    def side_stats(bets, label):
        bull = [b for b in bets if b["side"] == "Bull"]
        bear = [b for b in bets if b["side"] == "Bear"]
        print(f"{label}:")
        print(f"  total bets: {len(bets)}")
        for name, lst in (("Bull", bull), ("Bear", bear)):
            if not lst:
                print(f"  {name}: 0 bets")
                continue
            wins = sum(1 for b in lst if b["win"])
            pnl = sum(b["profit"] for b in lst)
            print(f"  {name}: {len(lst)} bets ({len(lst)/len(bets)*100:.1f}%)  "
                  f"WR={wins/len(lst)*100:.2f}%  PnL={pnl:+.4f} BNB  "
                  f"mean/bet={pnl/len(lst):+.5f}")
    side_stats(in_window, "Hot 30d (n={})".format(len(in_window)))
    side_stats(overall,   "Full range (n={})".format(len(overall)))

    # ----- Q3: BTC daily %-change quintile correlation with per-bet PnL -----
    print("\n=== Q3: Per-day BTC %-change quintile vs per-bet PnL ===")
    # Build BTC per-day %-change. Group by calendar UTC day.
    btc_tl = asset_timeline.get("BTC", [])
    if not btc_tl:
        print("  BTC missing; skipping")
    else:
        # Group BTC prices by UTC date string -> list of (ts, price)
        day_prices: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for ts, p in btc_tl:
            day = fmt_date(ts)
            day_prices[day].append((ts, p))
        # For each day, %-change = (last - first) / first
        day_chg: dict[str, float] = {}
        for day, lst in day_prices.items():
            if len(lst) >= 2:
                lst.sort()
                first = lst[0][1]; last = lst[-1][1]
                day_chg[day] = (last - first) / first * 100.0

        # Build per-day bet aggregations using 50 BNB bets (more samples)
        day_bets: dict[str, list[float]] = defaultdict(list)
        for b in bets_50:
            day = fmt_date(b["ts"])
            day_bets[day].append(b["profit"])

        # Join days that have both BTC %-change AND ≥1 bet
        joined = [(day, day_chg[day], day_bets[day]) for day in day_chg if day in day_bets]
        if not joined:
            print("  no days with both BTC chg + bets")
        else:
            # Sort by BTC %-change, then quintile-bucket
            joined.sort(key=lambda x: x[1])
            n = len(joined)
            q_size = max(1, n // 5)
            print(f"  {n} days with bets + BTC pct-change. Quintile size ~{q_size}.")
            print(f"  {'Quintile':>9s}  {'BTC pct range':>18s}  {'Days':>4s}  "
                  f"{'Bets':>5s}  {'Wins':>4s}  {'WR':>7s}  {'PnL':>10s}  {'Mean/bet':>10s}")
            for q in range(5):
                lo = q * q_size
                hi = (q + 1) * q_size if q < 4 else n
                slice_ = joined[lo:hi]
                btc_lo = slice_[0][1]; btc_hi = slice_[-1][1]
                bets_in_q = [p for _, _, profits in slice_ for p in profits]
                n_bets = len(bets_in_q)
                if n_bets == 0:
                    print(f"  Q{q+1:>1d}    {btc_lo:+7.2f}%..{btc_hi:+7.2f}%  "
                          f"{len(slice_):>4d}  {'0':>5s}  {'-':>4s}  {'-':>7s}  "
                          f"{'-':>10s}  {'-':>10s}")
                    continue
                wins_in_q = sum(1 for p in bets_in_q if p > 0)
                pnl_in_q = sum(bets_in_q)
                mean_in_q = pnl_in_q / n_bets
                wr_in_q = wins_in_q / n_bets * 100.0
                print(f"  Q{q+1:>1d}    {btc_lo:+7.2f}%..{btc_hi:+7.2f}%  "
                      f"{len(slice_):>4d}  {n_bets:>5d}  {wins_in_q:>4d}  "
                      f"{wr_in_q:>6.2f}%  {pnl_in_q:>+10.4f}  {mean_in_q:>+10.5f}")


if __name__ == "__main__":
    main()
