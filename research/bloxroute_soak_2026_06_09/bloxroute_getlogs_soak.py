#!/usr/bin/env python3
"""bloXroute eth_getLogs read-path SOAK (read-only; does NOT touch rpc_poller.py
or live config).

Goal: build confidence that bsc.rpc.blxrbdn.com (bloXroute) can serve as a live
read-path endpoint for eth_getLogs before wiring it in. Polls at the live bot's
cadence (~8s) over the live bot's near-head window (most recent ~18 blocks),
writing nothing back.

Per 8s poll:
  - bloXroute eth_getLogs over [cursor+1, head-SETTLE]  (EVERY poll: the load test)
  - capture latency / HTTP status / error class / response bytes / near-head dist
  - record bloXroute head vs reference head (throttle/stall signal)
Every PARITY_EVERY-th poll (~60s):
  - fetch the SAME range via the live 3-endpoint hedged getBlockReceipts path
  - assert byte-identical bet-event set on
        (tx_hash, logIndex, epoch, side, amount_wei, block_number)
  - (sampled — not per-poll — to bound receipts data to ~4 GB over 12h, since the
    receipts path is the ~6 MB/poll firehose the bot was stopped to avoid)

First-hour guard: if cumulative bloXroute error rate > 5% (>=20 polls) OR ANY
receipts-parity divergence, write a STOP sentinel + final summary and exit so the
operator can debug before committing more time. After the first hour it runs to
completion.

Outputs (research/bloxroute_soak_2026_06_09/):
  poll_log.csv   one row per poll
  summary.json   running aggregates (rewritten every poll)
  STOP           sentinel written only on early-stop, with the reason

  ./.venv/bin/python research/bloxroute_soak_2026_06_09/bloxroute_getlogs_soak.py [hours=12]
"""
import json
import os
import sys
import time
import urllib.request as R
from collections import Counter

# --- constants (hardcoded from the verified live config so the soak is self-
# contained and cannot die on an import error; provenance in comments) ---
BLOXROUTE = "https://bsc.rpc.blxrbdn.com"                         # candidate read endpoint
# READ_PATH_HEDGED_ENDPOINTS, rpc_poller.py:115-119 (the live poll/receipts pool)
RECEIPT_ENDPOINTS = ["https://bsc-dataseed1.binance.org",
                     "https://bsc-dataseed1.defibit.io",
                     "https://bsc-rpc.publicnode.com"]
# PREDICTION_V2_CONTRACT_ADDRESS, constants.py:9
CONTRACT = "0x18B2A687610328590Bc8F2e5fEdDe3b582A49cdA".lower()
# _BET_BULL_TOPIC / _BET_BEAR_TOPIC, rpc_poller.py:78-79
BULL = "0x438122d8cff518d18388099a5181f0d17a12b4f1b55faedf6e4a6acee0060c12"
BEAR = "0x0d8c1fe3e67ab767116a81f122b83c2557a8c2564019cb7c4f83de1aeb1f1f0d"

SETTLE = 3                 # query to head-3 (~1.35s back): near-head but past the
                           # racing tip, so endpoint head-sync can't fake a diff
POLL_INTERVAL_S = 8        # live periodic-poll cadence
SEED_WINDOW = 18           # first-poll window (~8s of blocks @ 450ms)
PARITY_EVERY = 8           # receipts parity on every Nth poll (~64s)
RANGE_CLAMP = 1000         # safety: never query a giant range if the soak stalls
FIRST_HOUR_S = 3600
ERR_RATE_TRIP = 0.05
MIN_POLLS_FOR_TRIP = 20
GETLOGS_TIMEOUT = 15
RECEIPTS_TIMEOUT = 20
HEAD_TIMEOUT = 5

OUTDIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(OUTDIR, "poll_log.csv")
SUMMARY_PATH = os.path.join(OUTDIR, "summary.json")
STOP_PATH = os.path.join(OUTDIR, "STOP")


def iso(t):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


def pct(xs, p):
    if not xs:
        return 0.0
    s = sorted(xs)
    k = (len(s) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def http_post(url, payload, timeout):
    """Returns dict(ok_http, raw, bytes, ms, err). Never raises."""
    body = json.dumps(payload).encode()
    req = R.Request(url, data=body, headers={"Content-Type": "application/json",
                                             "User-Agent": "pancakebot-soak/1.0"})
    t = time.time()
    try:
        with R.urlopen(req, timeout=timeout) as r:
            raw = r.read()
        return {"ok_http": True, "raw": raw, "bytes": len(raw), "ms": (time.time() - t) * 1000, "err": None}
    except R.HTTPError as e:
        try:
            raw = e.read()
        except Exception:  # noqa: BLE001
            raw = b""
        return {"ok_http": False, "raw": raw, "bytes": len(raw), "ms": (time.time() - t) * 1000,
                "err": f"http{e.code}"}
    except Exception as e:  # noqa: BLE001
        msg = str(e).lower()
        cls = "timeout" if "timed out" in msg or "timeout" in msg else type(e).__name__
        return {"ok_http": False, "raw": b"", "bytes": 0, "ms": (time.time() - t) * 1000, "err": cls}


def extract_bet(log):
    if (log.get("address") or "").lower() != CONTRACT:
        return None
    topics = log.get("topics") or []
    if len(topics) < 3:
        return None
    t0 = topics[0]
    if t0 != BULL and t0 != BEAR:
        return None
    try:
        return (log["transactionHash"], int(log["logIndex"], 16), int(topics[2], 16),
                "Bull" if t0 == BULL else "Bear", int(log.get("data", "0x0"), 16),
                int(log["blockNumber"], 16))
    except (ValueError, KeyError, IndexError):
        return None


def blocknumber(url):
    r = http_post(url, {"jsonrpc": "2.0", "id": 1, "method": "eth_blockNumber", "params": []}, HEAD_TIMEOUT)
    if not r["ok_http"]:
        return None
    try:
        return int(json.loads(r["raw"])["result"], 16)
    except Exception:  # noqa: BLE001
        return None


def ref_head():
    for ep in RECEIPT_ENDPOINTS:
        h = blocknumber(ep)
        if h is not None:
            return h
    return None


def getlogs_bloxroute(frm, to):
    flt = {"fromBlock": hex(frm), "toBlock": hex(to), "address": CONTRACT, "topics": [[BULL, BEAR]]}
    r = http_post(BLOXROUTE, {"jsonrpc": "2.0", "id": 1, "method": "eth_getLogs", "params": [flt]}, GETLOGS_TIMEOUT)
    if not r["ok_http"]:
        return {"status": r["err"], "ms": r["ms"], "bytes": r["bytes"], "bets": None, "nlogs": 0}
    try:
        p = json.loads(r["raw"])
    except Exception:  # noqa: BLE001
        return {"status": "badjson", "ms": r["ms"], "bytes": r["bytes"], "bets": None, "nlogs": 0}
    if "error" in p:
        return {"status": f"rpc{p['error'].get('code', '?')}", "ms": r["ms"], "bytes": r["bytes"],
                "bets": None, "nlogs": 0}
    logs = p.get("result") or []
    bets = {b for b in (extract_bet(x) for x in logs) if b}
    return {"status": "ok", "ms": r["ms"], "bytes": r["bytes"], "bets": bets, "nlogs": len(logs)}


def receipts_bets(frm, to):
    """Live read pool, hedged first-success (serialized). Ground-truth oracle."""
    payload = [{"jsonrpc": "2.0", "id": j, "method": "eth_getBlockReceipts", "params": [hex(bn)]}
               for j, bn in enumerate(range(frm, to + 1))]
    t = time.time()
    last = "all_failed"
    nbytes = 0
    for ep in RECEIPT_ENDPOINTS:
        r = http_post(ep, payload, RECEIPTS_TIMEOUT)
        nbytes = r["bytes"]
        if not r["ok_http"]:
            last = r["err"]
            continue
        try:
            arr = json.loads(r["raw"])
        except Exception:  # noqa: BLE001
            last = "badjson"
            continue
        if not isinstance(arr, list):
            last = "notlist"
            continue
        byid = {x.get("id"): x for x in arr}
        bets, good = set(), True
        for j in range(to - frm + 1):
            x = byid.get(j)
            if not x or "result" not in x or x.get("result") is None:
                good = False
                break
            for rcpt in x["result"]:
                if isinstance(rcpt, dict):
                    for log in (rcpt.get("logs") or []):
                        b = extract_bet(log)
                        if b:
                            bets.add(b)
        if good:
            return {"status": "ok", "ms": (time.time() - t) * 1000, "bytes": nbytes, "bets": bets}
        last = "partial"
    return {"status": last, "ms": (time.time() - t) * 1000, "bytes": nbytes, "bets": None}


def write_summary(st, final=False):
    elapsed = time.time() - st["start"]
    ok_lat = st["ok_lat"]
    summ = {
        "soak": "bloxroute_getlogs_2026_06_09",
        "final": final,
        "stopped_early": st["stopped_early"],
        "started_utc": iso(st["start"]),
        "updated_utc": iso(time.time()),
        "elapsed_hours": round(elapsed / 3600.0, 3),
        "config": {"poll_interval_s": POLL_INTERVAL_S, "settle_blocks": SETTLE,
                   "parity_every": PARITY_EVERY, "endpoint": BLOXROUTE},
        "blxr_getlogs": {
            "polls": st["polls"],
            "ok": st["polls"] - st["errors"],
            "errors": st["errors"],
            "error_rate": round(st["errors"] / max(1, st["polls"]), 4),
            "error_breakdown": dict(st["err_counter"]),
            "latency_ms_ok": {"p50": round(pct(ok_lat, 50), 1), "p95": round(pct(ok_lat, 95), 1),
                              "p99": round(pct(ok_lat, 99), 1), "max": round(max(ok_lat), 1) if ok_lat else 0},
            "bytes_total": st["blxr_bytes"],
            "bytes_per_day_est": int(st["blxr_bytes"] / max(1.0, elapsed) * 86400),
            "total_bets_seen": st["blxr_bets_seen"],
        },
        "parity_vs_receipts": {
            "checks": st["parity_checks"],
            "clean": st["parity_checks"] - len(st["parity_diffs"]),
            "divergences": len(st["parity_diffs"]),
            "oracle_failures": st["oracle_fail"],
            "diff_detail": st["parity_diffs"][:20],
            "receipts_bytes_total": st["rcpt_bytes"],
        },
        "head_lag_blxr_minus_ref": {
            "samples": len(st["head_lags"]),
            "min": min(st["head_lags"]) if st["head_lags"] else None,
            "p50": round(pct(st["head_lags"], 50), 1) if st["head_lags"] else None,
            "max": max(st["head_lags"]) if st["head_lags"] else None,
        },
    }
    tmp = SUMMARY_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(summ, f, indent=2)
    os.replace(tmp, SUMMARY_PATH)


def main():
    hours = float(sys.argv[1]) if len(sys.argv) > 1 else 12.0
    os.makedirs(OUTDIR, exist_ok=True)
    if os.path.exists(STOP_PATH):
        os.remove(STOP_PATH)
    header = ["ts_utc", "poll", "ref_head", "blxr_head", "head_lag", "from_block", "to_block",
              "range", "near_head_dist", "blxr_status", "blxr_ms", "blxr_bytes", "blxr_nlogs",
              "blxr_nbets", "parity", "parity_ms", "parity_ok", "parity_diff_n"]
    csv_f = open(CSV_PATH, "w", buffering=1)
    csv_f.write(",".join(header) + "\n")

    st = {"start": time.time(), "polls": 0, "errors": 0, "err_counter": Counter(), "ok_lat": [],
          "blxr_bytes": 0, "rcpt_bytes": 0, "blxr_bets_seen": 0, "parity_checks": 0,
          "parity_diffs": [], "oracle_fail": 0, "head_lags": [], "stopped_early": None}
    end_time = st["start"] + hours * 3600.0
    cursor = None
    print(f"[soak] start {iso(st['start'])}  duration={hours}h  endpoint={BLOXROUTE}", flush=True)
    print(f"[soak] cadence={POLL_INTERVAL_S}s  settle={SETTLE}  parity_every={PARITY_EVERY}", flush=True)

    while time.time() < end_time:
        idx = st["polls"]
        try:
            H = ref_head()
            if H is None:
                print(f"[soak] poll {idx}: ref head unavailable, skipping", flush=True)
                time.sleep(POLL_INTERVAL_S)
                continue
            Hb = blocknumber(BLOXROUTE)
            to_block = H - SETTLE
            from_block = (to_block - SEED_WINDOW + 1) if cursor is None else (cursor + 1)
            if to_block < from_block:
                time.sleep(POLL_INTERVAL_S)
                continue
            if to_block - from_block + 1 > RANGE_CLAMP:
                from_block = to_block - RANGE_CLAMP + 1

            g = getlogs_bloxroute(from_block, to_block)
            st["polls"] += 1
            st["blxr_bytes"] += g["bytes"]
            if g["status"] == "ok":
                st["ok_lat"].append(g["ms"])
                if g["bets"]:
                    st["blxr_bets_seen"] += len(g["bets"])
            else:
                st["errors"] += 1
                st["err_counter"][g["status"]] += 1
            head_lag = (Hb - H) if Hb is not None else ""
            if Hb is not None:
                st["head_lags"].append(Hb - H)

            parity_label, parity_ms, parity_ok, diff_n = "", "", "", ""
            if idx % PARITY_EVERY == 0 and g["status"] == "ok":
                rc = receipts_bets(from_block, to_block)
                st["rcpt_bytes"] += rc["bytes"]
                parity_label, parity_ms = rc["status"], round(rc["ms"], 1)
                if rc["status"] == "ok" and g["bets"] is not None:
                    st["parity_checks"] += 1
                    only_b = sorted(g["bets"] - rc["bets"])
                    only_r = sorted(rc["bets"] - g["bets"])
                    diff_n = len(only_b) + len(only_r)
                    parity_ok = (diff_n == 0)
                    if diff_n:
                        st["parity_diffs"].append({"poll": idx, "from": from_block, "to": to_block,
                                                   "only_blxr": [list(x) for x in only_b[:10]],
                                                   "only_receipts": [list(x) for x in only_r[:10]]})
                else:
                    st["oracle_fail"] += 1
                    parity_ok = f"oracle_{rc['status']}"

            csv_f.write(",".join(str(v) for v in [
                iso(time.time()), idx, H, (Hb if Hb is not None else ""), head_lag, from_block, to_block,
                to_block - from_block + 1, H - from_block, g["status"], round(g["ms"], 1), g["bytes"],
                g["nlogs"], (len(g["bets"]) if g["bets"] is not None else ""),
                parity_label, parity_ms, parity_ok, diff_n]) + "\n")
            cursor = to_block
            write_summary(st)

            if idx % 25 == 0:
                er = st["errors"] / max(1, st["polls"])
                print(f"[soak] poll {idx} elapsed={int(time.time()-st['start'])}s "
                      f"blxr={g['status']}({round(g['ms'])}ms,{g['nlogs']}logs) "
                      f"err_rate={er:.1%} parity_ok={st['parity_checks']-len(st['parity_diffs'])}/"
                      f"{st['parity_checks']} head_lag={head_lag}", flush=True)

            # ---- first-hour guard ----
            if time.time() - st["start"] < FIRST_HOUR_S:
                er = st["errors"] / max(1, st["polls"])
                if st["polls"] >= MIN_POLLS_FOR_TRIP and er > ERR_RATE_TRIP:
                    st["stopped_early"] = f"blxr error_rate {er:.1%} > {ERR_RATE_TRIP:.0%} at poll {idx}"
                    break
                if st["parity_diffs"]:
                    d = st["parity_diffs"][-1]
                    st["stopped_early"] = f"parity divergence at poll {d['poll']} (blocks {d['from']}..{d['to']})"
                    break
        except Exception as e:  # noqa: BLE001  -- one bad poll must not kill a 12h soak
            st["polls"] += 1
            st["errors"] += 1
            st["err_counter"][f"exc_{type(e).__name__}"] += 1
            print(f"[soak] poll {idx} EXCEPTION {type(e).__name__}: {e}", flush=True)

        nxt = st["start"] + st["polls"] * POLL_INTERVAL_S
        slp = nxt - time.time()
        if slp > 0:
            time.sleep(slp)

    write_summary(st, final=True)
    if st["stopped_early"]:
        with open(STOP_PATH, "w") as f:
            f.write(st["stopped_early"] + "\n")
        print(f"[soak] STOPPED EARLY: {st['stopped_early']}", flush=True)
    else:
        print(f"[soak] DONE {iso(time.time())}  polls={st['polls']} "
              f"err_rate={st['errors']/max(1,st['polls']):.1%} "
              f"parity_clean={st['parity_checks']-len(st['parity_diffs'])}/{st['parity_checks']}", flush=True)
    csv_f.close()


if __name__ == "__main__":
    main()
