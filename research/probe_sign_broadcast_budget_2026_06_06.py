"""Decompose the unbudgeted decision_ready -> mempool-arrival latency on BSC.

The LATE diagnosis (2026-06-06) showed the deadline guard fires at
``t_decision_ready`` (engine.py:1122) but the bot then spends ~265ms it never
budgeted: build-tx (incl. an INLINE ``get_transaction_count`` nonce RPC,
prediction_contract.py:839) -> ECDSA sign -> ``send_raw_transaction`` broadcast.
This probe measures each component against the bot's PRIMARY WRITE endpoint
(``WRITE_PATH_RPC_URLS[0]``) so we can set a defensible ``BSC_BET_SIGN_BROADCAST_MS``.

Components:
  1. nonce WARM  - back-to-back get_transaction_count on a reused, hot client
  2. nonce COLD  - fresh client each call (TLS handshake included). The bot's
                   write client is idle for MINUTES between bets, so the real
                   per-bet nonce fetch is a cold/reconnecting call, not warm.
  3. build_tx    - real betBull(...).build_transaction({inline nonce, gas, gasPrice})
                   = warm nonce + ABI encode (+ chainId if uncached); the faithful
                   full pre-sign cost.
  4. sign        - local account.sign_transaction(built_tx) (ECDSA; no network)
  5. broadcast   - NOT re-sent (would burn gas). Cited from the 4 VM production
                   bets (latency_broadcast_ms 28-31ms) + 2026-05-20 send_raw probe.

SAFETY: zero gas. Steps 1-3 are read-only RPC; step 4 is local. No TX is sent.
Uses a FIXED DUMMY key (timings are address/key-independent) -- the production
key is never loaded or printed.

Run on the VM:
  cd /root/pancakebot && PYTHONPATH=/root/pancakebot ./.venv/bin/python \\
      research/probe_sign_broadcast_budget_2026_06_06.py --n 200 --n-cold 100
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from eth_account import Account
from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware

from pancakebot.constants import (
    WRITE_PATH_RPC_URLS, EXPECTED_CHAIN_ID, GAS_LIMIT_BET, MAX_GAS_PRICE_WEI,
    PREDICTION_V2_CONTRACT_ADDRESS, BNB_WEI,
)

PRIMARY = WRITE_PATH_RPC_URLS[0]
DUMMY_KEY = "0x" + "11" * 32           # fixed, non-zero; timings are key-independent
ABI_PATH = Path("abi/prediction_v2_abi.json")
DUMMY_EPOCH = 999_999                   # no send -> any epoch is fine
BET_AMOUNT_WEI = int(0.001 * BNB_WEI)


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def _pct(xs, p):
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, int(len(s) * p))]


def _stats(label, xs):
    if not xs:
        return {"label": label, "n": 0}
    return {
        "label": label, "n": len(xs), "mean": sum(xs) / len(xs),
        "p50": _pct(xs, .50), "p75": _pct(xs, .75), "p95": _pct(xs, .95),
        "p99": _pct(xs, .99), "max": max(xs),
    }


def _print_stats(s):
    if not s.get("n"):
        print(f"  {s['label']:<22} n=0 (no samples)")
        return
    print(f"  {s['label']:<22} n={s['n']:<4} "
          f"p50={s['p50']:6.1f}  p75={s['p75']:6.1f}  p95={s['p95']:6.1f}  "
          f"p99={s['p99']:6.1f}  max={s['max']:6.1f}  mean={s['mean']:6.1f}")


def _build_w3(url):
    w3 = Web3(Web3.HTTPProvider(url, request_kwargs={"timeout": 20}))
    w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
    return w3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="warm-call sample size")
    ap.add_argument("--n-cold", type=int, default=100, help="cold-call sample size")
    args = ap.parse_args()

    acct = Account.from_key(DUMMY_KEY)
    addr = acct.address
    print(f"primary write endpoint: {PRIMARY}")
    print(f"probe address (ephemeral, not production): {addr[:6]}...{addr[-4:]}")
    print(f"gas_limit={GAS_LIMIT_BET}  gas_price_wei={MAX_GAS_PRICE_WEI}\n")

    # Warm client; prime chain_id so build_transaction won't re-fetch it.
    w3 = _build_w3(PRIMARY)
    cid = int(w3.eth.chain_id)
    assert cid == int(EXPECTED_CHAIN_ID), f"chain_id {cid} != {EXPECTED_CHAIN_ID}"
    contract = w3.eth.contract(
        address=Web3.to_checksum_address(PREDICTION_V2_CONTRACT_ADDRESS),
        abi=json.loads(ABI_PATH.read_text()),
    )
    for _ in range(5):  # warm the TLS/keepalive connection
        w3.eth.get_transaction_count(addr)

    # 1. nonce WARM (reused hot connection, back-to-back)
    warm = []
    for _ in range(args.n):
        t0 = _now_ms()
        w3.eth.get_transaction_count(addr)
        warm.append(_now_ms() - t0)

    # 2. nonce COLD (fresh client each call -> TLS handshake)
    cold = []
    for _ in range(args.n_cold):
        cw3 = _build_w3(PRIMARY)
        t0 = _now_ms()
        cw3.eth.get_transaction_count(addr)
        cold.append(_now_ms() - t0)
        del cw3
        time.sleep(0.2)  # avoid hammering / rate-limit confound

    # 3. build_tx (warm nonce + ABI encode (+chainId if uncached))
    builds = []
    built_tx = None
    for _ in range(min(args.n, 100)):
        t0 = _now_ms()
        tx = contract.functions.betBull(DUMMY_EPOCH).build_transaction({
            "from": addr,
            "value": BET_AMOUNT_WEI,
            "nonce": Web3.to_int(w3.eth.get_transaction_count(addr)),
            "gas": int(GAS_LIMIT_BET),
            "gasPrice": int(MAX_GAS_PRICE_WEI),
        })
        builds.append(_now_ms() - t0)
        built_tx = tx

    # 4. sign (local ECDSA)
    signs = []
    for _ in range(args.n):
        t0 = _now_ms()
        acct.sign_transaction(built_tx)
        signs.append(_now_ms() - t0)

    print("=== component latency (ms) ===")
    s_warm = _stats("nonce_warm", warm)
    s_cold = _stats("nonce_cold", cold)
    s_build = _stats("build_tx(warm+nonce)", builds)
    s_sign = _stats("sign_ecdsa(local)", signs)
    for s in (s_warm, s_cold, s_build, s_sign):
        _print_stats(s)

    build_overhead_p95 = max(0.0, s_build["p95"] - s_warm["p95"])
    print(f"\n  build encode overhead (build_p95 - nonce_warm_p95): "
          f"~{build_overhead_p95:.1f} ms")

    print("\n=== modeled decision_ready -> mempool arrival (the unbudgeted gap) ===")
    BCAST_REAL = 31.0  # max of the 4 VM production bets (latency_broadcast_ms)
    warm_total = s_warm["p95"] + build_overhead_p95 + s_sign["p95"] + BCAST_REAL
    cold_total = s_cold["p95"] + build_overhead_p95 + s_sign["p95"] + BCAST_REAL
    print(f"  WARM-nonce path p95: {s_warm['p95']:.0f}(nonce) + "
          f"{build_overhead_p95:.0f}(encode) + {s_sign['p95']:.1f}(sign) + "
          f"{BCAST_REAL:.0f}(bcast) = {warm_total:.0f} ms")
    print(f"  COLD-nonce path p95: {s_cold['p95']:.0f}(nonce) + "
          f"{build_overhead_p95:.0f}(encode) + {s_sign['p95']:.1f}(sign) + "
          f"{BCAST_REAL:.0f}(bcast) = {cold_total:.0f} ms")
    print(f"\n  cross-check vs 4 VM real bets (sign_bucket+broadcast): "
          f"265 / 267 / 257 / 314 ms  (mean ~276)")

    out = {
        "primary": PRIMARY, "chain_id": cid,
        "nonce_warm": s_warm, "nonce_cold": s_cold,
        "build_tx": s_build, "sign": s_sign,
        "build_overhead_p95_ms": build_overhead_p95,
        "broadcast_real_ms": BCAST_REAL,
        "warm_path_p95_ms": warm_total, "cold_path_p95_ms": cold_total,
    }
    Path("research/probe_sign_broadcast_budget_2026_06_06_summary.json").write_text(
        json.dumps(out, indent=2))
    print("\nsummary -> research/probe_sign_broadcast_budget_2026_06_06_summary.json")


if __name__ == "__main__":
    main()
