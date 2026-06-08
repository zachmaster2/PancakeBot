#!/usr/bin/env python3
"""Inclusion-offset experiment: does broadcasting EARLIER reduce the one-block slip?

Architecture (per spec):
  - 1 COORDINATOR process: per round, sleep to lock-1300, anchor-poll the head
    block (BEP-520 ms), compute dynamic_deadline_ms via the engine SSOT
    (predict_predecessor_milli_ts -> compute_submit_deadline_ms), publish to the
    5 wallets over a Unix socket. Publishes TWICE per round: a "round" msg (epoch
    + lock) early so wallets pre-sign, then a "deadline" msg after the anchor.
  - 5 WALLET processes: pre-sign a min-stake betBull/betBear on the "round" msg;
    on the "deadline" msg, broadcast at deadline + offset (offset <= 0 => EARLIER
    than the dynamic deadline). All on defibit-1. If the deadline msg arrives
    after the target broadcast time -> SKIP, log ipc_late (no own anchor poll).
  - POST-BET: after lock+2s, each wallet checks inclusion (receipt -> block ->
    BEP-520 ms vs lock_ts) -> on_time = included_block_ts < lock_ts (strict).

OFFSET CONVENTION: broadcast_target = deadline_ms + offset (offset < 0 => broadcast
EARLIER than the dynamic deadline); off_vs_deadline = -offset. Offsets
{-200,-300,-350,-400,-450} => off_vs_deadline {200,300,350,400,450}, concentrated
around the absorption knee. W0 (-200) anchors the live bot's operating point (it
broadcasts ~186 ms before its own deadline); W1-W4 probe progressively earlier.
Tests whether broadcasting earlier reduces the one-block slip RATE (hypothesis A)
or only ABSORBS slips that still occur (hypothesis B) -- the within-round paired
comparison across the 5 wallets discriminates. The dynamic deadline (per-round,
predecessor-anchored) is the live bot's broadcast target.

Real money. --dry-run signs real TXs but sends NONE (validates plumbing).
Run:  python research/inclusion_experiment_2026_06_08.py --rounds 10 [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import socket
import struct
import sys
import time
import urllib.request as R
from multiprocessing import Process

REPO = "/root/pancakebot"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from web3 import Web3                                              # noqa: E402
from eth_account import Account                                   # noqa: E402
from pancakebot.constants import (                                # noqa: E402
    PREDICTION_V2_CONTRACT_ADDRESS, MAX_GAS_PRICE_WEI,
    EXPECTED_CHAIN_ID, BNB_WEI, GAS_LIMIT_BET,
)
from pancakebot.chain.rpc_poller import (                         # noqa: E402
    compute_milli_ts, predict_predecessor_milli_ts, compute_submit_deadline_ms,
)

SOCK_PATH = "/tmp/pancakebot_experiment.sock"
SEND_ENDPOINT = "https://bsc-dataseed1.defibit.io"   # defibit-1 (all wallets, sends)
READ_ENDPOINT = "https://bsc-dataseed1.binance.org"  # reads (coordinator + inclusion)
ABI_PATH = f"{REPO}/abi/prediction_v2_abi.json"
WALLET_FILE = "/etc/pancakebot/experiment_wallets.env"
LOGDIR = f"{REPO}/var/experiment_20260608"
OFFSETS = {0: -200, 1: -300, 2: -350, 3: -400, 4: -450}  # wallet idx -> ms; off_vs_deadline = -offset
GAS_PRICE_WEI = int(MAX_GAS_PRICE_WEI)                  # 1 gwei
ANCHOR_LEAD_MS = 1500  # anchor at lock-1500: headroom for X up to 450 even in earliest-deadline (lock-625) phase
POST_LOCK_WAIT_MS = 2000
BLOCK_TIME_MS = 450


def now_ms() -> float:
    return time.time() * 1000.0


def log(path, rec):
    rec = dict(rec)
    rec["wall_ms"] = now_ms()
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def load_wallets():
    kv = {}
    for line in open(WALLET_FILE):
        line = line.strip()
        if line and "=" in line:
            k, v = line.split("=", 1)
            kv[k] = v
    return {i: (kv[f"WALLET_{i}_KEY"], kv[f"WALLET_{i}_ADDR"]) for i in range(5)}


def raw_rpc(endpoint, method, params, timeout=5):
    req = R.Request(endpoint, data=json.dumps(
        {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode(),
        headers={"Content-Type": "application/json"})
    with R.urlopen(req, timeout=timeout) as r:
        return json.load(r).get("result")


def _send_msg(conn, obj):
    b = json.dumps(obj).encode()
    conn.sendall(struct.pack(">I", len(b)) + b)


def _recv_msg(conn):
    hdr = b""
    while len(hdr) < 4:
        c = conn.recv(4 - len(hdr))
        if not c:
            return None
        hdr += c
    n = struct.unpack(">I", hdr)[0]
    buf = b""
    while len(buf) < n:
        c = conn.recv(n - len(buf))
        if not c:
            return None
        buf += c
    return json.loads(buf.decode())


# ---------------------------------------------------------------- coordinator
def coordinator(n_rounds):
    logp = f"{LOGDIR}/coordinator.jsonl"
    w3 = Web3(Web3.HTTPProvider(READ_ENDPOINT, request_kwargs={"timeout": 6}))
    abi = json.load(open(ABI_PATH))
    contract = w3.eth.contract(
        address=Web3.to_checksum_address(PREDICTION_V2_CONTRACT_ADDRESS), abi=abi)

    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    if os.path.exists(SOCK_PATH):
        os.unlink(SOCK_PATH)
    srv.bind(SOCK_PATH)
    srv.listen(8)
    log(logp, {"ev": "coordinator_listening", "sock": SOCK_PATH})
    conns = []
    for _ in range(5):
        c, _a = srv.accept()
        conns.append(c)
    log(logp, {"ev": "all_wallets_connected", "n": len(conns)})

    last_epoch = None
    for rnd in range(n_rounds):
        while True:                                  # find round with >=18s lead
            epoch = int(contract.functions.currentEpoch().call())
            lock_ts = int(contract.functions.rounds(epoch).call()[2])
            if epoch != last_epoch and (lock_ts - time.time()) >= 18:
                break
            time.sleep(2)
        last_epoch = epoch
        lock_ms = lock_ts * 1000
        for c in conns:                              # msg1: round (wallets pre-sign)
            try:
                _send_msg(c, {"type": "round", "rnd": rnd, "epoch": epoch,
                              "lock_ms": lock_ms, "lock_ts": lock_ts})
            except Exception as e:
                log(logp, {"ev": "publish_round_fail", "err": str(e)})
        log(logp, {"ev": "round_detected", "rnd": rnd, "epoch": epoch,
                   "lock_ts": lock_ts, "lock_ms": lock_ms, "lead_s": lock_ts - time.time()})

        target = (lock_ms - ANCHOR_LEAD_MS) / 1000.0  # sleep to anchor time
        while time.time() < target:
            time.sleep(min(0.05, max(0.0, target - time.time())))
        a0 = now_ms()
        try:
            head = raw_rpc(READ_ENDPOINT, "eth_getBlockByNumber", ["latest", False], timeout=4)
        except Exception as e:
            head = None
            log(logp, {"ev": "anchor_rpc_fail", "rnd": rnd, "err": str(e)})
        a1 = now_ms()
        anchor_milli = compute_milli_ts(head) if head else None
        if anchor_milli is None:                      # degrade: publish null deadline
            for c in conns:
                try:
                    _send_msg(c, {"type": "deadline", "rnd": rnd, "epoch": epoch,
                                  "deadline_ms": None, "lock_ms": lock_ms})
                except Exception:
                    pass
            log(logp, {"ev": "anchor_decode_fail", "rnd": rnd, "epoch": epoch})
            time.sleep(max(0, (lock_ms + 4000) / 1000.0 - time.time()))
            continue
        pred_pred = predict_predecessor_milli_ts(anchor_milli_ts=anchor_milli, lock_ms=lock_ms)
        deadline_ms = compute_submit_deadline_ms(
            predicted_predecessor_milli_ts=pred_pred, lock_ms=lock_ms)
        pub = now_ms()
        msg = {"type": "deadline", "rnd": rnd, "epoch": epoch, "lock_ms": lock_ms,
               "deadline_ms": deadline_ms, "anchor_milli": anchor_milli,
               "anchor_block": int(head["number"], 16), "pred_pred_ms": pred_pred,
               "pred_lock_ms": pred_pred + BLOCK_TIME_MS, "ipc_publish_ts": pub}
        for c in conns:
            try:
                _send_msg(c, msg)
            except Exception as e:
                log(logp, {"ev": "publish_deadline_fail", "err": str(e)})
        log(logp, {"ev": "deadline_published", "rnd": rnd, "epoch": epoch, "lock_ms": lock_ms,
                   "anchor_block": int(head["number"], 16), "anchor_milli": anchor_milli,
                   "pred_pred_ms": pred_pred, "pred_lock_ms": pred_pred + BLOCK_TIME_MS,
                   "deadline_ms": deadline_ms, "deadline_off_lock_ms": lock_ms - deadline_ms,
                   "anchor_poll_start_ts": a0, "anchor_poll_complete_ts": a1,
                   "anchor_rtt_ms": a1 - a0, "ipc_publish_ts": pub})
        time.sleep(max(0, (lock_ms + 4000) / 1000.0 - time.time()))  # wait past lock
    for c in conns:
        try:
            c.close()
        except Exception:
            pass
    log(logp, {"ev": "coordinator_done"})


# -------------------------------------------------------------------- wallet
def wallet(idx, key, addr, offset, dry_run):
    logp = f"{LOGDIR}/wallet_{idx}.jsonl"
    w3s = Web3(Web3.HTTPProvider(SEND_ENDPOINT, request_kwargs={"timeout": 8}))
    abi = json.load(open(ABI_PATH))
    contract = w3s.eth.contract(
        address=Web3.to_checksum_address(PREDICTION_V2_CONTRACT_ADDRESS), abi=abi)
    acct = Account.from_key(key)
    assert acct.address == addr, "key/addr mismatch"
    try:
        stake = int(contract.functions.minBetAmount().call())
    except Exception:
        stake = 10 ** 15                              # 0.001 BNB fallback
    nonce = int(raw_rpc(READ_ENDPOINT, "eth_getTransactionCount", [addr, "pending"]), 16)
    log(logp, {"ev": "wallet_start", "idx": idx, "addr": addr, "offset": offset,
               "start_nonce": nonce, "stake_bnb": stake / BNB_WEI, "dry": dry_run})

    conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    for _ in range(50):
        try:
            conn.connect(SOCK_PATH)
            break
        except Exception:
            time.sleep(0.2)
    else:
        log(logp, {"ev": "connect_fail"})
        return
    log(logp, {"ev": "connected"})

    presigned = None
    while True:
        m = _recv_msg(conn)
        if m is None:
            log(logp, {"ev": "socket_closed"})
            break

        if m["type"] == "round":
            epoch = int(m["epoch"])
            lock_ms = float(m["lock_ms"])
            side = random.choice(["Bull", "Bear"])
            fn = contract.functions.betBull(epoch) if side == "Bull" else contract.functions.betBear(epoch)
            tx = fn.build_transaction({                # all fields => zero RPCs
                "from": addr, "value": int(stake), "nonce": int(nonce),
                "gas": int(GAS_LIMIT_BET), "gasPrice": int(GAS_PRICE_WEI),
                "chainId": int(EXPECTED_CHAIN_ID)})
            signed = acct.sign_transaction(tx)
            presigned = {"epoch": epoch, "side": side, "lock_ms": lock_ms,
                         "raw": signed.raw_transaction, "nonce": int(nonce)}
            log(logp, {"ev": "presigned", "rnd": m["rnd"], "epoch": epoch, "side": side,
                       "nonce": int(nonce), "stake_bnb": stake / BNB_WEI,
                       "sign_bytes": len(signed.raw_transaction), "tx_to": tx["to"],
                       "tx_value": int(tx["value"]), "tx_gas": int(tx["gas"]),
                       "tx_gasprice": int(tx["gasPrice"]), "tx_chainid": int(tx["chainId"]),
                       "tx_data_selector": tx["data"][:10]})
            continue

        if m["type"] != "deadline":
            continue
        rnd = m["rnd"]
        epoch = int(m["epoch"])
        ipc_read_ts = now_ms()
        deadline_ms = m.get("deadline_ms")
        if deadline_ms is None or presigned is None or presigned["epoch"] != epoch:
            log(logp, {"ev": "skip_no_deadline_or_mismatch", "rnd": rnd, "epoch": epoch,
                       "have_deadline": deadline_ms is not None,
                       "presigned_epoch": (presigned or {}).get("epoch")})
            presigned = None
            continue
        target_broadcast = float(deadline_ms) + offset
        ipc_latency = ipc_read_ts - float(m.get("ipc_publish_ts", ipc_read_ts))
        if ipc_read_ts > target_broadcast:            # IPC late -> SKIP (no fallback)
            log(logp, {"ev": "ipc_late", "rnd": rnd, "epoch": epoch, "offset": offset,
                       "ipc_read_ts": ipc_read_ts, "target_broadcast": target_broadcast,
                       "ipc_latency_ms": ipc_latency, "miss_ms": ipc_read_ts - target_broadcast})
            presigned = None
            continue
        while now_ms() < target_broadcast:            # spin-sleep to broadcast time
            time.sleep(min(0.002, max(0.0, (target_broadcast - now_ms()) / 1000.0)))
        ab0 = now_ms()
        txh = None
        send_err = None
        if dry_run:
            send_err = "DRY_RUN"
        else:
            try:
                txh = w3s.eth.send_raw_transaction(presigned["raw"]).hex()
            except Exception as e:
                send_err = str(e)
        ab1 = now_ms()
        log(logp, {"ev": "broadcast", "rnd": rnd, "epoch": epoch, "side": presigned["side"],
                   "nonce": presigned["nonce"], "lock_ms": float(m["lock_ms"]),
                   "deadline_ms": float(deadline_ms), "offset": offset,
                   "target_broadcast_ts": target_broadcast, "actual_broadcast_ts": ab0,
                   "broadcast_off_lock_ms": float(m["lock_ms"]) - ab0,
                   "ipc_publish_ts": m.get("ipc_publish_ts"), "ipc_read_ts": ipc_read_ts,
                   "ipc_latency_ms": ipc_latency, "send_raw_rtt_ms": ab1 - ab0,
                   "sign_bytes": len(presigned["raw"]), "tx_hash": txh, "send_err": send_err})
        if txh and not dry_run:
            nonce += 1
            wait_until = (float(m["lock_ms"]) + POST_LOCK_WAIT_MS) / 1000.0
            time.sleep(max(0, wait_until - time.time()))
            _log_inclusion(logp, txh, epoch, rnd, int(m["lock_ms"]) // 1000, m)
        presigned = None
    try:
        conn.close()
    except Exception:
        pass


def _log_inclusion(logp, txh, epoch, rnd, lock_ts, m):
    try:
        rcpt = raw_rpc(READ_ENDPOINT, "eth_getTransactionReceipt", [txh])
        if not rcpt:
            log(logp, {"ev": "inclusion_pending", "rnd": rnd, "epoch": epoch, "tx_hash": txh})
            return
        bn = int(rcpt["blockNumber"], 16)
        status = int(rcpt["status"], 16)
        blk = raw_rpc(READ_ENDPOINT, "eth_getBlockByNumber", [hex(bn), False])
        bms = compute_milli_ts(blk)
        bts = int(blk["timestamp"], 16)
        on_time = bts < lock_ts
        log(logp, {"ev": "inclusion", "rnd": rnd, "epoch": epoch, "tx_hash": txh,
                   "included_block_number": bn, "included_block_ts": bts,
                   "included_block_ms": bms, "lock_ts": lock_ts, "lock_ms": int(m["lock_ms"]),
                   "margin_ms": (bms - int(m["lock_ms"])) if bms is not None else None,
                   "status": status, "on_time": on_time,
                   "pred_pred_ms": m.get("pred_pred_ms"), "pred_lock_ms": m.get("pred_lock_ms"),
                   "anchor_block": m.get("anchor_block")})
    except Exception as e:
        log(logp, {"ev": "inclusion_error", "rnd": rnd, "epoch": epoch, "tx_hash": txh, "err": str(e)})


def main():
    global LOGDIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--logdir", default=LOGDIR)
    args = ap.parse_args()
    LOGDIR = args.logdir                              # children inherit via fork
    os.makedirs(LOGDIR, exist_ok=True)
    wallets = load_wallets()
    procs = [Process(target=coordinator, args=(args.rounds,))]
    for i in range(5):
        key, addr = wallets[i]
        procs.append(Process(target=wallet, args=(i, key, addr, OFFSETS[i], args.dry_run)))
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    print("experiment finished; logs in", LOGDIR)


if __name__ == "__main__":
    main()
