"""Backfill historical 1s OKX klines + extended closed-rounds into var/extended/.

Lenient: never raises on insufficient/boundary-mismatch/empty.  Persists what
we got with a `data_status` field.  Single-attempt during initial pass; a
separate verification pass retries holes 5x at 30s spacing to confirm whether
they are real or transient.

Resumable: per-round fsync after each successful append; epoch checkpoint
state at var/extended/<symbol>.checkpoint.json.

Phases (CLI subcommands):
    --probe                 # determine oldest reachable epoch per symbol
    --fetch-rounds          # extended closed_rounds via Graph (Phase B2)
    --fetch-klines [SYM]    # extended kline fetch for SYM or all (Phase B3)
                            #   --order=week-major-symbol-minor (default)
    --verify-holes          # re-probe MISSING/PARTIAL entries 5x@30s (Phase B4)
    --report                # coverage report (Phase B5)
    --all                   # rounds + klines + verify + report end-to-end

Storage layout:
    var/extended/closed_rounds.jsonl    [oldest_reachable..437561]
    var/extended/btc_spot_prices.jsonl  per-round records, lenient
    var/extended/eth_spot_prices.jsonl
    var/extended/sol_spot_prices.jsonl
    var/extended/bnb_spot_prices.jsonl
    var/extended/<sym>.checkpoint.json  {last_completed_epoch, n_ok_full, n_partial, n_missing}
    var/extended/coverage_report.json   summary
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Iterable

import requests

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pancakebot.market_data.okx_client import OkxClient, okx_rate_acquire  # noqa: E402
from pancakebot.market_data.graph_client import GraphClient  # noqa: E402
from pancakebot.constants import PREDICTION_V2_GRAPH_ENDPOINT  # noqa: E402
from pancakebot.app import load_env, require_env  # noqa: E402
from pancakebot.util import InvariantError, TransientGraphError  # noqa: E402

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

EXTENDED_DIR = REPO / "var" / "extended"
EXTENDED_CR = EXTENDED_DIR / "closed_rounds.jsonl"

CANONICAL_FLOOR = 437562  # canonical store starts here; stop backfill at floor-1
GRAPH_PAGE_SIZE = 1000  # matches GraphClient default; 50 was a debug fallback
INTERVAL_SECONDS = 300  # round duration

# Per-symbol extended file paths
SYMBOL_TO_FILE = {
    "BTC-USDT": EXTENDED_DIR / "btc_spot_prices.jsonl",
    "ETH-USDT": EXTENDED_DIR / "eth_spot_prices.jsonl",
    "SOL-USDT": EXTENDED_DIR / "sol_spot_prices.jsonl",
    "BNB-USDT": EXTENDED_DIR / "bnb_spot_prices.jsonl",
}
SYMBOL_TO_CHECKPOINT = {
    sym: EXTENDED_DIR / f"{sym.split('-')[0].lower()}.checkpoint.json"
    for sym in SYMBOL_TO_FILE
}

OKX_HISTORY_CANDLES = "https://www.okx.com/api/v5/market/history-candles"

# Status values written to each kline record's `data_status` field
STATUS_OK_FULL = "OK_FULL"            # 300 contig rows, exact boundary match
STATUS_OK_PARTIAL = "OK_PARTIAL"      # got rows, but not perfect (initial pass)
STATUS_MISSING = "MISSING"            # 0 rows (initial pass)
STATUS_ERROR = "ERROR"                # network/parse failure (initial pass)
STATUS_MISSING_VERIFIED = "MISSING_VERIFIED"  # 5x retry confirmed
STATUS_PARTIAL_VERIFIED = "PARTIAL_VERIFIED"  # 5x retry confirmed partial


# ----------------------------------------------------------------------------
# Lenient OKX fetch (single attempt, never raises)
# ----------------------------------------------------------------------------

def lenient_fetch_kline_window(
    symbol: str,
    oldest_open_ms: int,
    newest_open_ms_inclusive: int,
    *,
    timeout: float = 10.0,
    rate_acquire_fn=None,
) -> tuple[list[list], str, str]:
    """Single-attempt OKX /history-candles fetch.  Never raises.

    Returns (rows_oldest_first, status, detail).  status is one of:
      OK_FULL    - len(rows) == expected, contiguous, boundary matches
      OK_PARTIAL - got rows but length/contig/boundary off
      MISSING    - empty data
      ERROR      - HTTP/network/parse error
    """
    if newest_open_ms_inclusive < oldest_open_ms:
        return ([], STATUS_ERROR, "inverted_range")
    if (newest_open_ms_inclusive - oldest_open_ms) % 1000 != 0:
        return ([], STATUS_ERROR, "unaligned_ms")
    expected = (newest_open_ms_inclusive - oldest_open_ms) // 1000 + 1
    if expected > 300:
        return ([], STATUS_ERROR, "exceeds_max_300")

    after_ms = newest_open_ms_inclusive + 1000
    before_ms = oldest_open_ms - 1000
    params = {
        "instId": symbol, "bar": "1s",
        "limit": str(expected),
        "after": str(after_ms),
        "before": str(before_ms),
    }
    if rate_acquire_fn is not None:
        rate_acquire_fn()
    try:
        resp = requests.get(OKX_HISTORY_CANDLES, params=params, timeout=timeout)
    except Exception as e:
        return ([], STATUS_ERROR, f"net:{type(e).__name__}:{str(e)[:80]}")

    if resp.status_code != 200:
        return ([], STATUS_ERROR, f"http_{resp.status_code}")
    try:
        body = resp.json()
        rows_raw = body.get("data") or []
    except Exception as e:
        return ([], STATUS_ERROR, f"parse:{type(e).__name__}")

    if not rows_raw:
        return ([], STATUS_MISSING, "empty_data")

    arrays: list[list] = []
    for row in reversed(rows_raw):
        if not isinstance(row, list) or len(row) < 6:
            return ([], STATUS_ERROR, "row_invalid")
        try:
            arrays.append([
                int(row[0]), float(row[1]), float(row[2]),
                float(row[3]), float(row[4]), float(row[5]),
            ])
        except Exception:
            return ([], STATUS_ERROR, "row_parse")

    if (len(arrays) == expected
            and arrays[0][0] == oldest_open_ms
            and arrays[-1][0] == newest_open_ms_inclusive):
        for i in range(1, len(arrays)):
            if arrays[i][0] - arrays[i - 1][0] != 1000:
                return (arrays, STATUS_OK_PARTIAL, f"non_contig_at_{i}")
        return (arrays, STATUS_OK_FULL, "")

    return (arrays, STATUS_OK_PARTIAL,
            f"got_{len(arrays)}_first_{arrays[0][0]}_last_{arrays[-1][0]}")


# ----------------------------------------------------------------------------
# Probe: per-symbol oldest reachable epoch
# ----------------------------------------------------------------------------

def _build_kline_window(start_at: int) -> tuple[int, int]:
    """Compute (oldest_open_ms, newest_open_ms_inclusive) for a round."""
    lock_at = start_at + INTERVAL_SECONDS
    oldest_ms = (lock_at - 301) * 1000
    newest_ms = (lock_at - 2) * 1000
    return oldest_ms, newest_ms


def probe_oldest_reachable_epoch(
    symbol: str,
    *,
    known_floor_epoch: int,
    known_floor_start_at: int,
    days_back_max: int = 200,
    coarse_step_days: int = 5,
) -> dict:
    """Coarse-then-fine bisect: find the oldest epoch where OKX returns OK_FULL."""
    floor_la = known_floor_start_at + INTERVAL_SECONDS
    day_seconds = 86400

    # Coarse scan back from floor in 5-day steps.
    last_ok_days = 0
    first_fail_days: Optional[int] = None
    coarse: list[tuple[int, str, int]] = []  # (days_back, status, n)
    for d in range(0, days_back_max + 1, coarse_step_days):
        la = floor_la - d * day_seconds
        oldest_ms = (la - 301) * 1000
        newest_ms = (la - 2) * 1000
        rows, status, _detail = lenient_fetch_kline_window(symbol, oldest_ms, newest_ms)
        coarse.append((d, status, len(rows)))
        if status == STATUS_OK_FULL:
            last_ok_days = d
        else:
            first_fail_days = d
            break
        time.sleep(0.05)

    if first_fail_days is None:
        return {
            "symbol": symbol,
            "last_ok_days_back": last_ok_days,
            "first_fail_days_back": None,
            "coarse_scan": coarse,
        }

    # Fine bisect day-resolution between last_ok and first_fail.
    lo, hi = last_ok_days, first_fail_days
    while hi - lo > 1:
        mid = (lo + hi) // 2
        la = floor_la - mid * day_seconds
        oldest_ms = (la - 301) * 1000
        newest_ms = (la - 2) * 1000
        rows, status, _ = lenient_fetch_kline_window(symbol, oldest_ms, newest_ms)
        if status == STATUS_OK_FULL:
            lo = mid
        else:
            hi = mid
        time.sleep(0.05)

    return {
        "symbol": symbol,
        "last_ok_days_back": lo,
        "first_fail_days_back": hi,
        "coarse_scan": coarse,
    }


# ----------------------------------------------------------------------------
# Phase B2: extended closed_rounds via The Graph
# ----------------------------------------------------------------------------

def fetch_extended_closed_rounds(graph: GraphClient, *, oldest_epoch: int) -> dict:
    """Fetch closed rounds [oldest_epoch..CANONICAL_FLOOR-1] from the Graph.

    Single-attempt per page during initial pass.  Failed pages -> to_retry.
    Second pass with exponential backoff on to_retry list.
    """
    out_path = EXTENDED_CR
    end_epoch = CANONICAL_FLOOR - 1
    if end_epoch < oldest_epoch:
        return {"n_rounds": 0, "msg": "empty range"}

    # Page through in 1000-epoch chunks.
    # Initial pass.
    to_retry: list[tuple[int, int]] = []
    n_persisted = 0
    pages_attempted = 0
    pages_succeeded = 0
    t_start = time.time()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Write fresh (caller can re-run; we always overwrite for closed_rounds).
    with open(out_path, "w", encoding="utf-8") as out_f:
        # Iterate from oldest forward.
        epoch_start = oldest_epoch
        while epoch_start <= end_epoch:
            epoch_chunk_end = min(epoch_start + GRAPH_PAGE_SIZE - 1, end_epoch)
            pages_attempted += 1
            page_t0 = time.time()
            try:
                rounds = graph.fetch_closed_rounds(
                    order="asc",
                    epoch_gte=epoch_start,
                    epoch_lte=epoch_chunk_end,
                    first=GRAPH_PAGE_SIZE,
                    skip=0,
                )
                pages_succeeded += 1
                for r in rounds:
                    rec = {
                        "epoch": int(r.epoch),
                        "startAt": int(r.start_at),
                        "lockPrice": r.lock_price,
                        "closePrice": r.close_price,
                        "position": r.position,
                        "failed": bool(r.failed) if r.failed is not None else False,
                        "bets": [b.to_json() for b in r.bets],
                    }
                    out_f.write(json.dumps(rec) + "\n")
                    n_persisted += 1
                out_f.flush()
                os.fsync(out_f.fileno())
                if pages_attempted % 5 == 0 or len(rounds) > 0:
                    print(f"  [page ok ] [{epoch_start}..{epoch_chunk_end}] "
                          f"got={len(rounds)} total={n_persisted} "
                          f"elapsed={time.time() - page_t0:.1f}s", flush=True)
            except (TransientGraphError, Exception) as e:
                to_retry.append((epoch_start, epoch_chunk_end))
                print(f"  [page fail] epoch [{epoch_start}..{epoch_chunk_end}]: "
                      f"{type(e).__name__}: {str(e)[:100]}", flush=True)
            epoch_start = epoch_chunk_end + 1

    # Retry pass
    retried_succeeded = 0
    if to_retry:
        print(f"\n  retrying {len(to_retry)} failed pages with exponential backoff...", flush=True)
        with open(out_path, "a", encoding="utf-8") as out_f:
            for attempt_idx in range(3):
                still_retry: list[tuple[int, int]] = []
                backoff = 2 ** attempt_idx  # 1, 2, 4
                for ep_lo, ep_hi in to_retry:
                    time.sleep(backoff)
                    try:
                        rounds = graph.fetch_closed_rounds(
                            order="asc", epoch_gte=ep_lo, epoch_lte=ep_hi,
                            first=GRAPH_PAGE_SIZE, skip=0,
                        )
                        for r in rounds:
                            rec = {
                                "epoch": int(r.epoch),
                                "startAt": int(r.start_at),
                                "lockPrice": r.lock_price,
                                "closePrice": r.close_price,
                                "position": r.position,
                                "failed": bool(r.failed) if r.failed is not None else False,
                                "bets": [b.to_json() for b in r.bets],
                            }
                            out_f.write(json.dumps(rec) + "\n")
                            n_persisted += 1
                        out_f.flush()
                        os.fsync(out_f.fileno())
                        retried_succeeded += 1
                    except Exception as e:
                        still_retry.append((ep_lo, ep_hi))
                        print(f"    retry attempt {attempt_idx+1} failed [{ep_lo}..{ep_hi}]: "
                              f"{type(e).__name__}", flush=True)
                to_retry = still_retry
                if not to_retry:
                    break

    elapsed = time.time() - t_start
    summary = {
        "out_path": str(out_path),
        "oldest_epoch": oldest_epoch,
        "newest_epoch": end_epoch,
        "n_rounds_persisted": n_persisted,
        "pages_attempted": pages_attempted,
        "pages_succeeded": pages_succeeded,
        "pages_retried_succeeded": retried_succeeded,
        "pages_still_failing": len(to_retry),
        "still_failing_pages": [(lo, hi) for lo, hi in to_retry],
        "elapsed_seconds": round(elapsed, 2),
    }
    return summary


# ----------------------------------------------------------------------------
# Phase B3: extended klines (week-major-symbol-minor)
# ----------------------------------------------------------------------------

def _load_extended_rounds() -> list[dict]:
    """Load all rounds from var/extended/closed_rounds.jsonl, sorted by epoch asc."""
    rounds = []
    if not EXTENDED_CR.exists():
        raise FileNotFoundError(f"{EXTENDED_CR} not found.  Run --fetch-rounds first.")
    with open(EXTENDED_CR, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rounds.append(json.loads(s))
    rounds.sort(key=lambda r: int(r["epoch"]))
    return rounds


def _load_checkpoint(symbol: str) -> dict:
    p = SYMBOL_TO_CHECKPOINT[symbol]
    if not p.exists():
        return {"last_completed_epoch": None, "n_ok_full": 0, "n_partial": 0, "n_missing": 0, "n_error": 0}
    return json.loads(p.read_text())


def _save_checkpoint(symbol: str, ckpt: dict) -> None:
    p = SYMBOL_TO_CHECKPOINT[symbol]
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(ckpt))
    tmp.replace(p)


def _append_kline_record(symbol: str, record: dict) -> None:
    p = SYMBOL_TO_FILE[symbol]
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()
        os.fsync(f.fileno())


def fetch_extended_klines(
    symbols: list[str] | None = None,
    *,
    days_per_week: int = 7,
) -> dict:
    """Fetch klines for all rounds in var/extended/closed_rounds.jsonl, week-major.

    Iteration order:
        for week in oldest_week..newest_week:
            for sym in symbols (in order):
                fetch all rounds in this week for this symbol.

    Resumability: if a symbol's checkpoint indicates a last_completed_epoch
    >= the round being considered, skip it.
    """
    symbols = symbols or list(SYMBOL_TO_FILE.keys())
    rounds = _load_extended_rounds()
    if not rounds:
        return {"n_rounds": 0, "msg": "no extended rounds"}

    week_seconds = days_per_week * 86400
    earliest_start = rounds[0]["startAt"]
    week_buckets: dict[int, list[dict]] = {}
    for r in rounds:
        bucket_idx = (r["startAt"] - earliest_start) // week_seconds
        week_buckets.setdefault(int(bucket_idx), []).append(r)

    checkpoints = {sym: _load_checkpoint(sym) for sym in symbols}

    summary_per_sym = {sym: {"OK_FULL": 0, "OK_PARTIAL": 0, "MISSING": 0, "ERROR": 0,
                              "first_epoch": None, "last_epoch": None}
                       for sym in symbols}

    t_start = time.time()
    for week_idx in sorted(week_buckets.keys()):
        bucket = week_buckets[week_idx]
        first_epoch = bucket[0]["epoch"]
        last_epoch = bucket[-1]["epoch"]
        print(f"\n=== Week {week_idx} (epochs {first_epoch}..{last_epoch}, {len(bucket)} rounds) ===", flush=True)
        for sym in symbols:
            ckpt = checkpoints[sym]
            last_done = ckpt.get("last_completed_epoch")
            n_attempts = 0
            t_sym_start = time.time()
            for r in bucket:
                ep = int(r["epoch"])
                if last_done is not None and ep <= last_done:
                    continue
                start_at = int(r["startAt"])
                lock_at = start_at + INTERVAL_SECONDS
                oldest_ms, newest_ms = _build_kline_window(start_at)
                rows, status, detail = lenient_fetch_kline_window(
                    sym, oldest_ms, newest_ms,
                    rate_acquire_fn=okx_rate_acquire,
                )
                rec = {
                    "epoch": ep,
                    "lock_at": lock_at,
                    "klines_1s": rows,
                    "data_status": status,
                    "detail": detail,
                }
                _append_kline_record(sym, rec)
                # bump counts
                if status == STATUS_OK_FULL:
                    ckpt["n_ok_full"] = ckpt.get("n_ok_full", 0) + 1
                elif status == STATUS_OK_PARTIAL:
                    ckpt["n_partial"] = ckpt.get("n_partial", 0) + 1
                elif status == STATUS_MISSING:
                    ckpt["n_missing"] = ckpt.get("n_missing", 0) + 1
                else:
                    ckpt["n_error"] = ckpt.get("n_error", 0) + 1
                ckpt["last_completed_epoch"] = ep
                _save_checkpoint(sym, ckpt)
                summary_per_sym[sym][status] = summary_per_sym[sym].get(status, 0) + 1
                if summary_per_sym[sym]["first_epoch"] is None:
                    summary_per_sym[sym]["first_epoch"] = ep
                summary_per_sym[sym]["last_epoch"] = ep
                n_attempts += 1
            t_sym = time.time() - t_sym_start
            print(f"  {sym}: {n_attempts} fetches in {t_sym:.1f}s "
                  f"(OK_FULL={ckpt.get('n_ok_full',0)} P={ckpt.get('n_partial',0)} "
                  f"M={ckpt.get('n_missing',0)} E={ckpt.get('n_error',0)})", flush=True)

    elapsed = time.time() - t_start
    return {
        "elapsed_seconds": round(elapsed, 2),
        "weeks_processed": len(week_buckets),
        "per_symbol": summary_per_sym,
    }


# ----------------------------------------------------------------------------
# Phase B4: hole verification (5x retry, 30s spacing)
# ----------------------------------------------------------------------------

def verify_holes() -> dict:
    """For each MISSING/PARTIAL/ERROR record across all 4 symbols, retry 5x@30s.

    Updates each record's data_status in-place.  If all 5 retries produce the
    same shortfall, mark MISSING_VERIFIED or PARTIAL_VERIFIED.  If any retry
    succeeds, replace with the OK_FULL record.

    Modifies existing var/extended/<sym>.jsonl files via tmp+rename.
    """
    rounds_by_epoch_by_sym: dict[str, dict[int, dict]] = {}
    for sym, p in SYMBOL_TO_FILE.items():
        if not p.exists():
            continue
        rounds_by_epoch_by_sym[sym] = {}
        with open(p, encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                rec = json.loads(s)
                rounds_by_epoch_by_sym[sym][int(rec["epoch"])] = rec

    summary = {sym: {"holes_total": 0, "verified_missing": 0, "verified_partial": 0,
                      "recovered_ok_full": 0}
               for sym in rounds_by_epoch_by_sym}

    for sym, recs in rounds_by_epoch_by_sym.items():
        holes = [(ep, rec) for ep, rec in recs.items()
                 if rec.get("data_status") in (STATUS_OK_PARTIAL, STATUS_MISSING, STATUS_ERROR)]
        summary[sym]["holes_total"] = len(holes)
        if not holes:
            continue
        print(f"\n[verify {sym}] {len(holes)} holes to verify (5x retry @ 30s spacing)...", flush=True)
        for ep, rec in holes:
            lock_at = int(rec["lock_at"])
            oldest_ms = (lock_at - 301) * 1000
            newest_ms = (lock_at - 2) * 1000
            results = []
            recovered: Optional[dict] = None
            for attempt in range(5):
                if attempt > 0:
                    time.sleep(30)
                rows, status, _ = lenient_fetch_kline_window(
                    sym, oldest_ms, newest_ms, rate_acquire_fn=okx_rate_acquire,
                )
                results.append((status, len(rows)))
                if status == STATUS_OK_FULL:
                    recovered = {"epoch": ep, "lock_at": lock_at,
                                 "klines_1s": rows, "data_status": STATUS_OK_FULL,
                                 "detail": "verified_recovery"}
                    break
            if recovered:
                recs[ep] = recovered
                summary[sym]["recovered_ok_full"] += 1
            else:
                final_statuses = {s for s, _ in results}
                if final_statuses == {STATUS_MISSING}:
                    rec["data_status"] = STATUS_MISSING_VERIFIED
                    summary[sym]["verified_missing"] += 1
                else:
                    rec["data_status"] = STATUS_PARTIAL_VERIFIED
                    rec["detail"] = f"5x_retry_results={results}"
                    summary[sym]["verified_partial"] += 1

        # Rewrite the file
        p = SYMBOL_TO_FILE[sym]
        tmp = p.with_suffix(p.suffix + ".tmp_verify")
        with open(tmp, "w", encoding="utf-8") as f:
            for ep in sorted(recs.keys()):
                f.write(json.dumps(recs[ep]) + "\n")
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(p)
        print(f"  {sym}: rewrote {len(recs)} records  "
              f"(recovered={summary[sym]['recovered_ok_full']}, "
              f"verified_missing={summary[sym]['verified_missing']}, "
              f"verified_partial={summary[sym]['verified_partial']})", flush=True)

    return summary


# ----------------------------------------------------------------------------
# Phase B5: coverage report
# ----------------------------------------------------------------------------

def coverage_report() -> dict:
    rep: dict = {"closed_rounds": {}, "symbols": {}}
    if EXTENDED_CR.exists():
        n_cr = 0
        first_ep = last_ep = None
        with open(EXTENDED_CR, encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                ep = int(json.loads(s)["epoch"])
                if first_ep is None:
                    first_ep = ep
                last_ep = ep
                n_cr += 1
        rep["closed_rounds"] = {"n": n_cr, "first": first_ep, "last": last_ep,
                                 "path": str(EXTENDED_CR)}
    for sym, p in SYMBOL_TO_FILE.items():
        if not p.exists():
            rep["symbols"][sym] = {"n": 0, "path": str(p), "status_counts": {}}
            continue
        n = 0
        first_ep = last_ep = None
        status_counts: dict[str, int] = {}
        with open(p, encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                rec = json.loads(s)
                ep = int(rec["epoch"])
                if first_ep is None:
                    first_ep = ep
                last_ep = ep
                n += 1
                st = rec.get("data_status", "UNKNOWN")
                status_counts[st] = status_counts.get(st, 0) + 1
        rep["symbols"][sym] = {"n": n, "first": first_ep, "last": last_ep,
                                "path": str(p), "status_counts": status_counts}
    out = EXTENDED_DIR / "coverage_report.json"
    out.write_text(json.dumps(rep, indent=2))
    return rep


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def cmd_probe() -> dict:
    """Per-symbol oldest reachable.  Reads canonical floor from closed_rounds."""
    # Need the start_at for canonical floor epoch 437562
    cr_floor = REPO / "var" / "closed_rounds.jsonl"
    floor_start_at: Optional[int] = None
    with open(cr_floor, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            if int(obj["epoch"]) == CANONICAL_FLOOR:
                floor_start_at = int(obj["startAt"])
                break
    if floor_start_at is None:
        raise InvariantError("canonical_floor_not_in_closed_rounds")

    out = {}
    for sym in SYMBOL_TO_FILE:
        print(f"\n=== probe {sym} ===", flush=True)
        result = probe_oldest_reachable_epoch(
            sym, known_floor_epoch=CANONICAL_FLOOR, known_floor_start_at=floor_start_at,
        )
        out[sym] = result
        # Translate days_back to epoch estimate
        last_ok_epoch = CANONICAL_FLOOR - result["last_ok_days_back"] * 288
        print(f"  oldest_reachable ~ epoch {last_ok_epoch} ({result['last_ok_days_back']}d back)", flush=True)
    return out


def cmd_fetch_rounds(*, oldest_epoch: int) -> dict:
    load_env()
    graph = GraphClient(endpoint=PREDICTION_V2_GRAPH_ENDPOINT,
                        api_key=require_env("THE_GRAPH_API_KEY"))
    return fetch_extended_closed_rounds(graph, oldest_epoch=oldest_epoch)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true",
                    help="Probe per-symbol oldest reachable epoch")
    ap.add_argument("--fetch-rounds", action="store_true",
                    help="Fetch extended closed_rounds via Graph")
    ap.add_argument("--fetch-klines", action="store_true",
                    help="Fetch extended klines (4 symbols)")
    ap.add_argument("--verify-holes", action="store_true",
                    help="5x@30s retry on MISSING/PARTIAL records")
    ap.add_argument("--report", action="store_true",
                    help="Coverage report")
    ap.add_argument("--all", action="store_true",
                    help="Run rounds + klines + verify + report end-to-end")
    ap.add_argument("--oldest-epoch", type=int, default=None,
                    help="Override oldest closed_rounds epoch to fetch")
    ap.add_argument("--symbols", default="BTC-USDT,ETH-USDT,SOL-USDT,BNB-USDT",
                    help="Comma-separated symbols (default all 4)")
    args = ap.parse_args()

    EXTENDED_DIR.mkdir(parents=True, exist_ok=True)
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    if args.probe or args.all:
        probe_summary = cmd_probe()
        (EXTENDED_DIR / "probe_summary.json").write_text(json.dumps(probe_summary, indent=2, default=str))
        print(f"\n[probe] wrote {EXTENDED_DIR / 'probe_summary.json'}")

    if args.fetch_rounds or args.all:
        oldest = args.oldest_epoch
        if oldest is None:
            # Use BTC's last_ok_days from probe (BTC has the deepest reach)
            probe_path = EXTENDED_DIR / "probe_summary.json"
            if not probe_path.exists():
                raise InvariantError("must --probe first or pass --oldest-epoch")
            ps = json.loads(probe_path.read_text())
            btc_days = ps["BTC-USDT"]["last_ok_days_back"]
            oldest = CANONICAL_FLOOR - btc_days * 288
            print(f"[fetch-rounds] derived oldest_epoch={oldest} from BTC probe ({btc_days}d back)")
        rs = cmd_fetch_rounds(oldest_epoch=oldest)
        (EXTENDED_DIR / "fetch_rounds_summary.json").write_text(json.dumps(rs, indent=2, default=str))
        print(f"\n[fetch-rounds] {rs}")

    if args.fetch_klines or args.all:
        ks = fetch_extended_klines(symbols)
        (EXTENDED_DIR / "fetch_klines_summary.json").write_text(json.dumps(ks, indent=2, default=str))
        print(f"\n[fetch-klines] {ks}")

    if args.verify_holes or args.all:
        vh = verify_holes()
        (EXTENDED_DIR / "verify_holes_summary.json").write_text(json.dumps(vh, indent=2, default=str))
        print(f"\n[verify-holes] {vh}")

    if args.report or args.all:
        rep = coverage_report()
        print(f"\n[report] {json.dumps(rep, indent=2, default=str)}")


if __name__ == "__main__":
    main()
