"""Exact sub-component breakdown of the decision_ready -> mempool critical path,
to validate the 2-RPC-elimination hypothesis (Zach, 2026-06-06).

Measures the FOUR things on the path, the way the bot actually does them
(rotating across the 3 WRITE_PATH_RPC_URLS endpoints, prediction_contract.py
_rotate_rpc):
  - gas_price RPC   (eth_gasPrice)            -- cap-breach check, CACHEABLE
  - nonce RPC       (eth_getTransactionCount) -- CACHEABLE
  - ECDSA sign      (local account.sign)      -- stays on path (~local)
  - send_raw RPC    (broadcast)               -- stays on path; cited from real bets

For each RPC: COLD (fresh client each call -> the bot's real per-bet state, idle
write connection) and WARM (reused hot client -> the ceiling if we kept+warmed it).
The pre-cache design ELIMINATES the two RPCs from the path, so the realistic floor
is sign + send_raw (+ encode unless templated).

ZERO gas: gas_price/nonce are read-only; sign is local. No TX sent. Fixed dummy key.

Run on VM:
  cd /root/pancakebot && PYTHONPATH=/root/pancakebot ./.venv/bin/python \\
      research/probe_critpath_rpc_breakdown_2026_06_06.py
"""
from __future__ import annotations

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

DUMMY_KEY = "0x" + "11" * 32
N_COLD = 60          # rotated across the 3 endpoints, 20 each
N_WARM = 100
N_SIGN = 200


def _now_ms():
    return time.perf_counter() * 1000.0


def _pct(xs, p):
    s = sorted(xs)
    return s[min(len(s) - 1, int(len(s) * p))]


def _row(label, xs):
    if not xs:
        return f"  {label:<26} n=0"
    return (f"  {label:<26} n={len(xs):<4} p50={_pct(xs,.5):6.1f} "
            f"p75={_pct(xs,.75):6.1f} p95={_pct(xs,.95):6.1f} "
            f"p99={_pct(xs,.99):6.1f} max={max(xs):6.1f}")


def _w3(url):
    w = Web3(Web3.HTTPProvider(url, request_kwargs={"timeout": 20}))
    w.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
    return w


def main():
    acct = Account.from_key(DUMMY_KEY)
    addr = acct.address
    print(f"write endpoints (rotated): {WRITE_PATH_RPC_URLS}")
    print(f"probe addr (ephemeral): {addr[:6]}...{addr[-4:]}\n")

    # Warm clients, one per endpoint (primed).
    warm_clients = []
    for u in WRITE_PATH_RPC_URLS:
        w = _w3(u)
        assert int(w.eth.chain_id) == int(EXPECTED_CHAIN_ID)
        for _ in range(3):
            w.eth.gas_price; w.eth.get_transaction_count(addr)
        warm_clients.append(w)

    # COLD: fresh client each call, rotating endpoint like the bot.
    cold_gas, cold_nonce = [], []
    for i in range(N_COLD):
        url = WRITE_PATH_RPC_URLS[i % 3]
        w = _w3(url)
        t0 = _now_ms(); w.eth.gas_price; cold_gas.append(_now_ms() - t0)
        del w; time.sleep(0.15)
        w = _w3(url)
        t0 = _now_ms(); w.eth.get_transaction_count(addr); cold_nonce.append(_now_ms() - t0)
        del w; time.sleep(0.15)

    # WARM: reused hot clients, rotating.
    warm_gas, warm_nonce = [], []
    for i in range(N_WARM):
        w = warm_clients[i % 3]
        t0 = _now_ms(); w.eth.gas_price; warm_gas.append(_now_ms() - t0)
        t0 = _now_ms(); w.eth.get_transaction_count(addr); warm_nonce.append(_now_ms() - t0)

    # SIGN: local ECDSA on a realistic bet tx.
    contract = warm_clients[0].eth.contract(
        address=Web3.to_checksum_address(PREDICTION_V2_CONTRACT_ADDRESS),
        abi=json.loads(Path("abi/prediction_v2_abi.json").read_text()),
    )
    built = contract.functions.betBull(999999).build_transaction({
        "from": addr, "value": int(0.001 * BNB_WEI),
        "nonce": Web3.to_int(warm_clients[0].eth.get_transaction_count(addr)),
        "gas": int(GAS_LIMIT_BET), "gasPrice": int(MAX_GAS_PRICE_WEI),
    })
    signs = []
    for _ in range(N_SIGN):
        t0 = _now_ms(); acct.sign_transaction(built); signs.append(_now_ms() - t0)

    print("=== CRITICAL-PATH SUB-COMPONENTS (ms) ===")
    print("  -- the two RPCs the pre-cache ELIMINATES from the path --")
    print(_row("gas_price RPC  COLD", cold_gas))
    print(_row("gas_price RPC  warm", warm_gas))
    print(_row("nonce RPC      COLD", cold_nonce))
    print(_row("nonce RPC      warm", warm_nonce))
    print("  -- what STAYS on the path --")
    print(_row("ECDSA sign (local)", signs))
    print("  send_raw RPC               n=4   ~28-31 (4 VM production bets; warm, same")
    print("                                   endpoint as nonce)")

    cold_total = _pct(cold_gas, .95) + _pct(cold_nonce, .95) + _pct(signs, .95) + 31
    floor = _pct(signs, .95) + 31
    print(f"\n  CURRENT p95 (2 cold RPCs + sign + send_raw): ~{cold_total:.0f} ms")
    print(f"  FLOOR after eliminate (sign + send_raw):     ~{floor:.0f} ms")
    print(f"  => realistic critical-path improvement: ~{cold_total-floor:.0f} ms")

    Path("research/probe_critpath_rpc_breakdown_2026_06_06_summary.json").write_text(json.dumps({
        "cold_gas": {"p50": _pct(cold_gas,.5), "p95": _pct(cold_gas,.95), "max": max(cold_gas)},
        "cold_nonce": {"p50": _pct(cold_nonce,.5), "p95": _pct(cold_nonce,.95), "max": max(cold_nonce)},
        "warm_gas": {"p50": _pct(warm_gas,.5), "p95": _pct(warm_gas,.95)},
        "warm_nonce": {"p50": _pct(warm_nonce,.5), "p95": _pct(warm_nonce,.95)},
        "sign": {"p50": _pct(signs,.5), "p95": _pct(signs,.95)},
        "send_raw_real_ms": 31, "current_p95_ms": cold_total, "floor_ms": floor,
    }, indent=2))
    print("\nsummary -> research/probe_critpath_rpc_breakdown_2026_06_06_summary.json")


if __name__ == "__main__":
    main()
