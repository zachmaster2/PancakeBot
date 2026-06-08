"""Analyze the inclusion-offset experiment logs.

NOTE: the in-run inclusion check logs `inclusion_pending` because the wallet stored
tx_hash WITHOUT the 0x prefix (.hex() drops it on this web3 build) -> receipt lookup
missed. Bets place fine; outcomes are reconstructed here POST-HOC from the broadcast
tx_hashes (prepend 0x -> receipt -> block -> BEP-520 ms).

Metrics per (wallet, round):
  margin   = included_block_ms - lock_ms        (<0 on-time, >=0 LATE)
  on_time  = margin < 0  (corroborated by receipt status: 1=placed, 0=reverted/LATE)
  incl_lat = margin + broadcast_off_lock_ms      (broadcast -> included-block latency)
  slip     = incl_lat > 550  (bimodal ~313 made / ~760 slipped; NOTE over-flags very
             early broadcasts that simply waited for a far block -> use BLOCK NUMBER
             within-round as the clean slip signal, shown in the paired table)

Headline = WITHIN-ROUND PAIRED comparison: same round = same builder conditions, so
W0->W4 differences isolate the offset's causal effect (A: earlier prevents slip;
B: flat, only absorption). Usage: python analyze_inclusion_experiment.py <logdir>
"""
import glob
import json
import os
import sys
import urllib.request as R

sys.path.insert(0, "/root/pancakebot")
from pancakebot.chain.rpc_poller import compute_milli_ts  # noqa: E402

E = "https://bsc-dataseed1.binance.org"
SLIP_MS = 550
_cache = {}


def rpc(m, p):
    try:
        req = R.Request(E, data=json.dumps({"jsonrpc": "2.0", "id": 1, "method": m, "params": p}).encode(),
                        headers={"Content-Type": "application/json"})
        return json.load(R.urlopen(req, timeout=6)).get("result")
    except Exception:
        return None


def block_ms(bn):
    if bn not in _cache:
        b = rpc("eth_getBlockByNumber", [hex(bn), False])
        _cache[bn] = compute_milli_ts(b) if b else None
    return _cache[bn]


def load(logdir):
    coord = {}
    cpath = os.path.join(logdir, "coordinator.jsonl")
    if os.path.exists(cpath):
        for line in open(cpath):
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("ev") == "deadline_published":
                coord[d["epoch"]] = d
    wallets = {}
    for f in sorted(glob.glob(os.path.join(logdir, "wallet_*.jsonl"))):
        idx = int(f.split("wallet_")[1].split(".")[0])
        bcasts, ipc_late, meta = {}, set(), {}
        for line in open(f):
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("ev") == "wallet_start":
                meta = d
            elif d.get("ev") == "broadcast" and d.get("tx_hash"):
                bcasts[d["epoch"]] = d
            elif d.get("ev") == "ipc_late":
                ipc_late.add(d["epoch"])
        wallets[idx] = dict(meta=meta, bcasts=bcasts, ipc_late=ipc_late)
    return coord, wallets


def reconstruct(bc):
    h = bc["tx_hash"]
    h = h if str(h).startswith("0x") else "0x" + h
    rc = rpc("eth_getTransactionReceipt", [h])
    if not rc:
        return None
    bn = int(rc["blockNumber"], 16)
    st = int(rc["status"], 16)
    ms = block_ms(bn)
    if ms is None:
        return None
    margin = ms - int(bc["lock_ms"])
    incl = margin + bc["broadcast_off_lock_ms"]
    return dict(bn=bn, status=st, margin=margin, on_time=margin < 0, incl=incl, slip=incl > SLIP_MS)


def main():
    logdir = sys.argv[1] if len(sys.argv) > 1 else "/root/pancakebot/var/experiment_20260608/live1"
    coord, wallets = load(logdir)
    offmap = {i: wallets[i]["meta"].get("offset") for i in sorted(wallets)}
    rec = {i: {} for i in wallets}                    # idx -> epoch -> reconstruction
    for i in sorted(wallets):
        for ep, bc in wallets[i]["bcasts"].items():
            r = reconstruct(bc)
            if r:
                rec[i][ep] = r

    print(f"=== {logdir} ===")
    print(f"rounds published: {len(coord)}   epochs {min(coord) if coord else '-'}..{max(coord) if coord else '-'}")
    print(f"\n{'W':>2} {'off_vs_dl':>9} {'bets':>4} {'on_time':>7} {'LATE':>4} {'revert':>6} {'slip*':>5} {'ipc_late':>8}")
    for i in sorted(wallets):
        rs = rec[i].values()
        n = len(rs)
        ot = sum(1 for r in rs if r["on_time"])
        late = sum(1 for r in rs if not r["on_time"])
        rev = sum(1 for r in rs if r["status"] == 0)
        slip = sum(1 for r in rs if r["slip"])
        ipl = len(wallets[i]["ipc_late"])
        print(f"{i:>2} {-offmap[i]:>9} {n:>4} {ot:>7} {late:>4} {rev:>6} {slip:>5} {ipl:>8}")

    print("\n=== LATE rate + on-time rate vs off_vs_deadline (the bottom line) ===")
    for i in sorted(wallets):
        rs = list(rec[i].values())
        n = len(rs)
        if n:
            late = sum(1 for r in rs if not r["on_time"])
            print(f"  off_vs_dl={-offmap[i]:>3}:  n={n:>2}   LATE {late}/{n} ({100*late/n:>3.0f}%)   "
                  f"on_time {n-late}/{n} ({100*(n-late)/n:>3.0f}%)")

    print("\n=== WITHIN-ROUND PAIRED: included block (last 4) + margin; L=LATE, *=incl>550 ===")
    print("  " + "epoch".rjust(7) + "dl_off".rjust(7) + "".join(f"W{i}/{-offmap[i]}".rjust(14) for i in sorted(wallets)))
    for ep in sorted(coord):
        dloff = coord[ep].get("deadline_off_lock_ms", "?")
        cells = []
        for i in sorted(wallets):
            if ep in wallets[i]["ipc_late"]:
                cells.append("ipc_late")
            elif ep in rec[i]:
                r = rec[i][ep]
                tag = ("L" if not r["on_time"] else "") + ("*" if r["slip"] else "")
                cells.append(f"{r['bn']%10000:04d}:{r['margin']:+d}{tag}")
            else:
                cells.append("-")
        print("  " + f"{ep}".rjust(7) + f"{dloff}".rjust(7) + "".join(c.rjust(14) for c in cells))
    print("\n  (same round = same builder; W0 in a LATER block than W1-W4 => offset prevented a slip => A)")


if __name__ == "__main__":
    main()
