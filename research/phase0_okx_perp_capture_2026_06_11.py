"""Phase-0 OKX perp probe — data capture (read-only, public endpoints).

Captures, to var/extended/ (gitignored data; this script is the provenance):
  1. okx_bnb_swap_funding.jsonl   — full available funding history for
     BNB-USDT-SWAP (8h cadence; OKX retains ~3 months -> reaches 2026-03-11).
  2. okx_bnb_oi_1d.jsonl          — daily open-interest history (rubik,
     ~180 days -> reaches 2025-12-14).
  3. okx_swap_trades_BNB-USDT-SWAP.jsonl — perp trade tape walk-back via
     /api/v5/market/history-trades tradeId pagination (4 req/s, checkpointed,
     resumable) until ts <= TAPE_TARGET_S (fade-era start, 2026-05-10).
  4. okx_trades_BNB-USDT_gap.jsonl — SPOT tape gap-fill from now back to
     2026-05-01 (the archived Feb-25..May-01 capture's end), so the spot
     tape is continuous Feb-25 -> now for cross-era comparison.

Rate limit: 4 req/s total (sleep 0.25s between requests), far under OKX's
public budget; the bot is paused so there is no contention.

Run:  cd <repo> && .venv/Scripts/python.exe research/phase0_okx_perp_capture_2026_06_11.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "var" / "extended"
B = "https://www.okx.com"
RATE_SLEEP_S = 0.25

# fade-era start round (479953) started 2026-05-10; cover fade+dead on perp.
TAPE_TARGET_S = 1778371200          # 2026-05-10 00:00:00 UTC
SPOT_GAP_TARGET_S = 1777593600      # 2026-05-01 00:00:00 UTC (archive end)

S = requests.Session()
S.headers["User-Agent"] = "phase0-capture/1.0"


def get(path: str, params: dict) -> list:
    for attempt in range(5):
        try:
            r = S.get(B + path, params=params, timeout=15)
            d = r.json()
            if d.get("code") == "0":
                return d.get("data", [])
            time.sleep(1.0 + attempt)
        except Exception:
            time.sleep(1.0 + attempt)
    return []


def capture_funding() -> None:
    path = OUT / "okx_bnb_swap_funding.jsonl"
    rows, after = [], None
    while True:
        params = {"instId": "BNB-USDT-SWAP", "limit": "100"}
        if after:
            params["after"] = after
        d = get("/api/v5/public/funding-rate-history", params)
        if not d:
            break
        rows.extend(d)
        after = d[-1]["fundingTime"]
        time.sleep(RATE_SLEEP_S)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
    print(f"[funding] {len(rows)} records -> {path.name}", flush=True)


def capture_oi() -> None:
    path = OUT / "okx_bnb_oi_1d.jsonl"
    d = get("/api/v5/rubik/stat/contracts/open-interest-volume",
            {"ccy": "BNB", "period": "1D"})
    with open(path, "w", encoding="utf-8") as f:
        for r in d:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
    print(f"[oi-1d] {len(d)} records -> {path.name}", flush=True)


def walk_tape(inst_id: str, out_name: str, target_s: int) -> None:
    """Walk history-trades backward by tradeId until ts <= target_s.
    Checkpointed + resumable; appends raw records."""
    path = OUT / out_name
    ckpt_path = OUT / (out_name + ".checkpoint.json")
    after = None
    n_total = 0
    if ckpt_path.exists():
        ck = json.loads(ckpt_path.read_text(encoding="utf-8"))
        if ck.get("completed"):
            print(f"[{inst_id}] already complete ({ck['n_trades']} trades)", flush=True)
            return
        after = ck.get("oldest_tradeId")
        n_total = int(ck.get("n_trades", 0))
        print(f"[{inst_id}] resuming from tradeId={after} ({n_total} trades)", flush=True)
    t0 = time.time()
    mode = "a" if after else "w"
    with open(path, mode, encoding="utf-8") as f:
        pages = 0
        while True:
            params = {"instId": inst_id, "limit": "100", "type": "1"}
            if after:
                params["after"] = after
            d = get("/api/v5/market/history-trades", params)
            if not d:
                # Ambiguous: retention floor OR 5x persistent fetch failure.
                # Only the target-reached exit may stamp completed=True — a
                # false 'completed' checkpoint would permanently mask an
                # interior tape hole (adversarial-review finding 2026-06-12).
                print(f"[{inst_id}] empty page BEFORE target — leaving "
                      f"checkpoint incomplete (retention floor or persistent "
                      f"fetch failure; resume to retry)", flush=True)
                completed = False
                break
            for r in d:
                f.write(json.dumps(r, separators=(",", ":")) + "\n")
            n_total += len(d)
            after = d[-1]["tradeId"]
            oldest_ts = int(d[-1]["ts"]) / 1000
            pages += 1
            if pages % 200 == 0:
                f.flush()
                ckpt_path.write_text(json.dumps(dict(
                    oldest_tradeId=after, n_trades=n_total,
                    oldest_ts=oldest_ts, completed=False)), encoding="utf-8")
                rate = n_total / max(1, time.time() - t0)
                print(f"[{inst_id}] {n_total} trades, at "
                      f"{time.strftime('%Y-%m-%d %H:%M', time.gmtime(oldest_ts))}, "
                      f"{rate:.0f} tr/s", flush=True)
            if oldest_ts <= target_s:
                completed = True
                break
            time.sleep(RATE_SLEEP_S)
    ckpt_path.write_text(json.dumps(dict(
        oldest_tradeId=after, n_trades=n_total, completed=completed)),
        encoding="utf-8")
    print(f"[{inst_id}] DONE: {n_total} trades in {time.time()-t0:.0f}s", flush=True)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    capture_funding()
    capture_oi()
    walk_tape("BNB-USDT-SWAP", "okx_swap_trades_BNB-USDT-SWAP.jsonl", TAPE_TARGET_S)
    walk_tape("BNB-USDT", "okx_trades_BNB-USDT_gap.jsonl", SPOT_GAP_TARGET_S)
    print("[capture] ALL DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
