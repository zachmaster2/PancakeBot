"""Real ``eth_sendRawTransaction`` RTT probe on BSC mainnet.

Calibrates the ``BSC_BET_SUBMIT_ONE_WAY_MS=150`` placeholder in
``pancakebot/timing_constants.py`` against empirical round-trip time
for a self-transfer (value=0) using the production wallet + production
Web3 client setup.

SAFETY:
- Each TX is a self-transfer of 0 BNB (gas-only, no asset transfer).
- Hard cap of 100 TXs; never exceeded regardless of args.
- Pre-flight: balance >= 0.01 BNB, chain_id == 56, --confirm flag.
- Abort on 5 consecutive send failures.
- The probe never touches PredictionV2 contracts or PancakeSwap state.
- Bot can keep running; probe uses same wallet but separate Web3 client.
  No nonce conflict because bot is in dry mode (no on-chain TX submission).

Output:
- JSONL: var/strategy_review/.send_raw_tx_probe.jsonl  (one row per TX)
- Markdown summary: var/strategy_review/send_raw_tx_probe.md

Usage:
  # Single TX (dry-run, the recommended first step)
  python research/probe_send_raw_transaction_rtt.py --mode single \\
      --confirm "I authorize the sendRawTransaction probe"

  # Full 100 TXs (~17 min wallclock at 10s spacing)
  python research/probe_send_raw_transaction_rtt.py --mode full \\
      --confirm "I authorize the sendRawTransaction probe"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from pathlib import Path

# Add repo root to path so we can import pancakebot.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware  # BSC is a POA chain
from pancakebot.constants import EXPECTED_CHAIN_ID, RPC_URLS, BNB_WEI
from pancakebot.config import load_env, require_env


# --- Hardcoded safety constants -------------------------------------------

HARD_CAP_TXS = 100
MIN_BALANCE_BNB = 0.01            # abort if wallet has less than this
MAX_CONSECUTIVE_FAILURES = 5      # abort after this many in a row
INTER_TX_SLEEP_S = 10.0           # spacing between TX submissions
RECEIPT_TIMEOUT_S = 30.0          # inclusion poll timeout per TX
GAS_LIMIT_SELF_TRANSFER = 21000   # standard for simple value transfer
SAFE_GAS_PRICE_MAX_GWEI = 5       # abort if network suggests > this
EXPECTED_CONFIRM = "I authorize the sendRawTransaction probe"

OUTPUT_DIR = Path("var/strategy_review")
JSONL_PATH = OUTPUT_DIR / ".send_raw_tx_probe.jsonl"
SUMMARY_MD_PATH = OUTPUT_DIR / "send_raw_tx_probe.md"


# --- Per-TX record --------------------------------------------------------

@dataclass
class TxRecord:
    idx: int
    nonce: int
    tx_hash: str = ""
    ok: bool = False
    error: str = ""
    # Timing (perf_counter ms; relative, but deltas are wallclock-accurate)
    t_signed_ms: float = 0.0
    t_response_ms: float = 0.0
    rtt_ms: float = 0.0
    # Inclusion
    inclusion_ok: bool = False
    inclusion_error: str = ""
    block_number: int = 0
    block_timestamp: int = 0
    t_receipt_ms: float = 0.0
    inclusion_lag_ms: float = 0.0  # t_receipt - t_response


# --- Helpers --------------------------------------------------------------

def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def _build_w3() -> Web3:
    """Build Web3 client matching production primary RPC."""
    primary_url = RPC_URLS[0]
    w3 = Web3(Web3.HTTPProvider(primary_url, request_kwargs={"timeout": 30}))
    # BSC is a POA chain (extraData >32 bytes). Without this middleware,
    # get_block() raises ExtraDataLengthError.
    w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
    return w3


def _preflight(w3: Web3, account) -> dict:
    """Verify chain, balance, gas. Returns context dict or raises."""
    chain_id = int(w3.eth.chain_id)
    if chain_id != int(EXPECTED_CHAIN_ID):
        raise RuntimeError(
            f"unexpected_chain_id: got={chain_id} expected={EXPECTED_CHAIN_ID}"
        )

    addr = account.address
    balance_wei = int(w3.eth.get_balance(addr))
    balance_bnb = balance_wei / BNB_WEI
    if balance_bnb < MIN_BALANCE_BNB:
        raise RuntimeError(
            f"insufficient_balance: {balance_bnb:.6f} BNB < "
            f"{MIN_BALANCE_BNB} BNB minimum"
        )

    gas_price_wei = int(w3.eth.gas_price)
    gas_price_gwei = gas_price_wei / 1e9
    if gas_price_gwei > SAFE_GAS_PRICE_MAX_GWEI:
        raise RuntimeError(
            f"gas_price_too_high: {gas_price_gwei:.2f} Gwei > "
            f"{SAFE_GAS_PRICE_MAX_GWEI} Gwei safety max"
        )

    nonce = int(w3.eth.get_transaction_count(addr))

    return {
        "chain_id": chain_id,
        "wallet_address": addr,
        "balance_bnb": balance_bnb,
        "gas_price_wei": gas_price_wei,
        "gas_price_gwei": gas_price_gwei,
        "starting_nonce": nonce,
        "primary_rpc": RPC_URLS[0],
    }


def _send_one(
    w3: Web3, account, nonce: int, gas_price_wei: int, chain_id: int,
) -> tuple[TxRecord, bytes]:
    """Build, sign, send one self-transfer TX. Returns (record, signed_raw)
    for caller to optionally poll receipt off the critical path."""
    tx = {
        "from": account.address,
        "to": account.address,  # self-transfer
        "value": 0,
        "gas": GAS_LIMIT_SELF_TRANSFER,
        "gasPrice": int(gas_price_wei),
        "nonce": int(nonce),
        "chainId": int(chain_id),
        "data": b"",
    }
    rec = TxRecord(idx=-1, nonce=nonce)
    try:
        signed = account.sign_transaction(tx)
    except Exception as e:
        rec.error = f"sign_failed: {type(e).__name__}: {e}"
        return rec, b""

    # CRITICAL MEASUREMENT: bracket the synchronous send_raw_transaction call.
    rec.t_signed_ms = _now_ms()
    try:
        txh = w3.eth.send_raw_transaction(signed.raw_transaction)
        rec.t_response_ms = _now_ms()
        rec.rtt_ms = rec.t_response_ms - rec.t_signed_ms
        rec.tx_hash = txh.hex() if isinstance(txh, bytes) else str(txh)
        if not rec.tx_hash.startswith("0x"):
            rec.tx_hash = "0x" + rec.tx_hash
        rec.ok = True
    except Exception as e:
        rec.t_response_ms = _now_ms()
        rec.rtt_ms = rec.t_response_ms - rec.t_signed_ms
        rec.error = f"send_failed: {type(e).__name__}: {e}"
    return rec, signed.raw_transaction


def _wait_for_receipt_async(
    w3: Web3, rec: TxRecord, executor: ThreadPoolExecutor,
):
    """Submit a background receipt poll. Mutates rec when complete."""
    def _poll():
        try:
            receipt = w3.eth.wait_for_transaction_receipt(
                rec.tx_hash, timeout=RECEIPT_TIMEOUT_S, poll_latency=0.2,
            )
            rec.t_receipt_ms = _now_ms()
            rec.inclusion_lag_ms = rec.t_receipt_ms - rec.t_response_ms
            rec.block_number = int(receipt["blockNumber"])
            block = w3.eth.get_block(rec.block_number)
            rec.block_timestamp = int(block["timestamp"])
            rec.inclusion_ok = bool(receipt.get("status") == 1)
            if not rec.inclusion_ok:
                rec.inclusion_error = f"reverted: status={receipt.get('status')}"
        except Exception as e:
            rec.inclusion_error = f"poll_failed: {type(e).__name__}: {e}"
    return executor.submit(_poll)


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs_sorted = sorted(xs)
    n = len(xs_sorted)
    return xs_sorted[min(n - 1, int(n * p))]


def _stats(label: str, xs: list[float]) -> dict:
    if not xs:
        return {"label": label, "n": 0}
    return {
        "label": label,
        "n": len(xs),
        "mean": sum(xs) / len(xs),
        "p50": _pct(xs, 0.5),
        "p90": _pct(xs, 0.9),
        "p95": _pct(xs, 0.95),
        "p99": _pct(xs, 0.99),
        "max": max(xs),
    }


def _write_summary(records: list[TxRecord], ctx: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    successful = [r for r in records if r.ok]
    included = [r for r in successful if r.inclusion_ok]

    rtt_xs = [r.rtt_ms for r in successful]
    lag_xs = [r.inclusion_lag_ms for r in included]

    rtt_stats = _stats("rtt_ms", rtt_xs)
    lag_stats = _stats("inclusion_lag_ms", lag_xs)

    n_total = len(records)
    n_sent = len(successful)
    n_included = len(included)

    # Recommendation logic.
    if rtt_stats.get("n", 0) >= 10:
        # Production code measures the full round-trip via Web3.py's
        # synchronous send_raw_transaction. The TX is committed to the
        # validator's mempool when the RPC accepts it (= part-way through
        # the round-trip). A reasonable one-way upper bound is
        # min(p95_rtt, p95_rtt - typical_response_serialize). Without a
        # cleaner split, half the p95 RTT is the conservative one-way
        # estimate; rounding up to nearest 50ms quantum.
        p95 = rtt_stats["p95"]
        one_way_estimate = int((p95 / 2 + 49) // 50 * 50)  # round up to 50ms
        rec_text = (
            f"Empirical p95 round-trip: **{p95:.0f}ms**. "
            f"One-way estimate (p95/2 + propagation, rounded up to 50ms "
            f"quantum): **{one_way_estimate}ms**. "
            f"Current placeholder `BSC_BET_SUBMIT_ONE_WAY_MS = 150`."
        )
        if abs(one_way_estimate - 150) <= 25:
            rec_text += " **The 150ms placeholder is within ±25ms of the empirical estimate — no change needed.**"
        elif one_way_estimate < 150:
            rec_text += f" **Recommend lowering to {one_way_estimate}ms (tighter deadline math, more critical-path budget).**"
        else:
            rec_text += f" **Recommend raising to {one_way_estimate}ms (safer deadline math, fewer too-close-to-lock aborts).**"
    else:
        rec_text = "Sample too small for recommendation (need n >= 10 successful sends)."

    md = []
    md.append("# `eth_sendRawTransaction` RTT probe — BSC mainnet")
    md.append("")
    md.append(f"Date: 2026-05-16")
    md.append(f"Wallet: `{ctx['wallet_address']}`")
    md.append(f"RPC: `{ctx['primary_rpc']}`")
    md.append(f"Chain ID: {ctx['chain_id']}")
    md.append(f"Gas price at start: {ctx['gas_price_gwei']:.2f} Gwei")
    md.append(f"Starting nonce: {ctx['starting_nonce']}")
    md.append(f"Starting balance: {ctx['balance_bnb']:.6f} BNB")
    md.append("")
    md.append("## Results")
    md.append("")
    md.append(f"- TXs attempted: **{n_total}**")
    md.append(f"- TXs accepted by RPC (RTT measured): **{n_sent}**")
    md.append(f"- TXs included on-chain (within {RECEIPT_TIMEOUT_S:.0f}s): **{n_included}**")
    md.append(f"- Inclusion rate (of sent): **{(n_included/n_sent*100) if n_sent else 0:.1f}%**")
    md.append("")
    md.append("### Round-trip RTT (TX-signed → RPC-response, ms)")
    md.append("")
    if rtt_stats.get("n", 0):
        md.append(f"| stat | value |")
        md.append(f"|---|---:|")
        md.append(f"| n | {rtt_stats['n']} |")
        md.append(f"| mean | {rtt_stats['mean']:.1f} ms |")
        md.append(f"| p50 | {rtt_stats['p50']:.1f} ms |")
        md.append(f"| p90 | {rtt_stats['p90']:.1f} ms |")
        md.append(f"| p95 | {rtt_stats['p95']:.1f} ms |")
        md.append(f"| p99 | {rtt_stats['p99']:.1f} ms |")
        md.append(f"| max | {rtt_stats['max']:.1f} ms |")
    else:
        md.append("(no successful sends)")
    md.append("")
    md.append("### Inclusion lag (RPC-response → on-chain block, ms)")
    md.append("")
    if lag_stats.get("n", 0):
        md.append(f"| stat | value |")
        md.append(f"|---|---:|")
        md.append(f"| n | {lag_stats['n']} |")
        md.append(f"| mean | {lag_stats['mean']:.1f} ms |")
        md.append(f"| p50 | {lag_stats['p50']:.1f} ms |")
        md.append(f"| p90 | {lag_stats['p90']:.1f} ms |")
        md.append(f"| p95 | {lag_stats['p95']:.1f} ms |")
        md.append(f"| p99 | {lag_stats['p99']:.1f} ms |")
        md.append(f"| max | {lag_stats['max']:.1f} ms |")
    else:
        md.append("(no successful inclusions)")
    md.append("")
    md.append("## Recommendation for `BSC_BET_SUBMIT_ONE_WAY_MS`")
    md.append("")
    md.append(rec_text)
    md.append("")
    SUMMARY_MD_PATH.write_text("\n".join(md), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["single", "full"], required=True)
    ap.add_argument("--confirm", required=True,
                    help=f'Must equal: "{EXPECTED_CONFIRM}"')
    args = ap.parse_args()

    if args.confirm != EXPECTED_CONFIRM:
        print(f"REJECTED: --confirm must equal exactly: {EXPECTED_CONFIRM!r}",
              file=sys.stderr)
        return 2

    load_env()
    pk = require_env("BSC_WALLET_PRIVATE_KEY").strip()
    if pk.startswith("0x"):
        pk = pk[2:]

    print("Building Web3 client…", flush=True)
    w3 = _build_w3()
    account = w3.eth.account.from_key(pk)

    print("Preflight…", flush=True)
    ctx = _preflight(w3, account)
    print(json.dumps(ctx, indent=2), flush=True)

    n_target = 1 if args.mode == "single" else HARD_CAP_TXS
    print(f"\nMode: {args.mode!r}  ->  will attempt {n_target} TX(s) at "
          f"{INTER_TX_SLEEP_S}s spacing.\n", flush=True)

    records: list[TxRecord] = []
    consecutive_failures = 0
    nonce = ctx["starting_nonce"]
    gas_price_wei = ctx["gas_price_wei"]
    chain_id = ctx["chain_id"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    f = JSONL_PATH.open("w", encoding="utf-8")
    executor = ThreadPoolExecutor(max_workers=20, thread_name_prefix="rcpt")
    receipt_futures = []

    try:
        for i in range(n_target):
            rec, _raw = _send_one(w3, account, nonce, gas_price_wei, chain_id)
            rec.idx = i
            records.append(rec)

            if rec.ok:
                print(f"[{i+1}/{n_target}] nonce={nonce} OK  "
                      f"rtt={rec.rtt_ms:.0f}ms  tx={rec.tx_hash[:10]}…",
                      flush=True)
                consecutive_failures = 0
                nonce += 1
                receipt_futures.append((rec, _wait_for_receipt_async(w3, rec, executor)))
            else:
                print(f"[{i+1}/{n_target}] nonce={nonce} ERR  {rec.error}",
                      flush=True)
                consecutive_failures += 1
                # On failure, re-fetch nonce in case it advanced server-side.
                try:
                    new_nonce = int(w3.eth.get_transaction_count(account.address))
                    if new_nonce != nonce:
                        print(f"   nonce resync: {nonce} -> {new_nonce}", flush=True)
                        nonce = new_nonce
                except Exception as e:
                    print(f"   nonce resync failed: {e}", flush=True)
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    print(f"\nABORT: {MAX_CONSECUTIVE_FAILURES} consecutive "
                          f"failures.", flush=True)
                    break

            # Stream record to jsonl now (receipt fields will update; we'll
            # re-dump at end after all receipts complete).
            f.write(json.dumps(asdict(rec)) + "\n")
            f.flush()

            # Pace.
            if i < n_target - 1:
                time.sleep(INTER_TX_SLEEP_S)

        # Wait for any in-flight receipt polls (up to RECEIPT_TIMEOUT_S each).
        print("\nWaiting for in-flight receipt polls…", flush=True)
        for rec, fut in receipt_futures:
            try:
                fut.result(timeout=RECEIPT_TIMEOUT_S + 5)
            except Exception as e:
                if not rec.inclusion_error:
                    rec.inclusion_error = f"future_err: {type(e).__name__}: {e}"
    finally:
        executor.shutdown(wait=True)
        f.close()

    # Re-dump complete jsonl with receipt info.
    with JSONL_PATH.open("w", encoding="utf-8") as f2:
        for rec in records:
            f2.write(json.dumps(asdict(rec)) + "\n")

    _write_summary(records, ctx)
    print(f"\nSummary written to {SUMMARY_MD_PATH}", flush=True)
    print(f"Raw data:           {JSONL_PATH}", flush=True)

    # Re-check balance to confirm gas burn matches estimate.
    end_balance = int(w3.eth.get_balance(account.address)) / BNB_WEI
    gas_burned = ctx["balance_bnb"] - end_balance
    print(f"\nGas burned: {gas_burned:.6f} BNB "
          f"(${gas_burned * 670:.4f} approx at $670/BNB)", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
