#!/usr/bin/env python3
"""Claim winnings + sweep the 5 inclusion-experiment test wallets to production.

Run ON THE VM (reads test keys from /etc/pancakebot/experiment_wallets.env).
DO NOT auto-run — Zach inspects with --dry-run, then sends with --execute himself.

For each of the 5 test wallets:
  1. verify the derived address matches the env's WALLET_<i>_ADDR (and is non-zero),
  2. enumerate the wallet's bet rounds on-chain (getUserRounds),
  3. find unclaimed winners (claimable(epoch,user) == True; already-claimed/lost → skipped),
  4. claim(epochs) to pull winnings (skipped if none),
  5. sweep the remaining BNB (minus 1-gwei sweep gas) to the production wallet,
  6. print a per-wallet summary: start -> winnings -> final -> swept.

SAFETY
  - DRY-RUN is the DEFAULT: prints every TX it would build, signs/sends NOTHING.
  - --execute is required to send; --dry-run forces dry even if --execute is also passed.
  - In --execute, EACH TX is printed and requires typing 'yes' to send.
  - All TXs use 1 gwei gas (bot-consistent).
  - Production address is hardcoded + re-verified against the expected 0xaF96…A5E7;
    the script NEVER reads the production private key, and refuses if the destination
    equals any test wallet.

  python sweep_and_claim.py              # DRY-RUN (default)
  python sweep_and_claim.py --dry-run    # explicit dry-run
  python sweep_and_claim.py --execute    # send (prompts 'yes' per TX)
"""
import argparse
import json
import sys

sys.path.insert(0, "/root/pancakebot")
from web3 import Web3                                                    # noqa: E402
from eth_account import Account                                          # noqa: E402
from pancakebot.constants import (                                       # noqa: E402
    PREDICTION_V2_CONTRACT_ADDRESS, EXPECTED_CHAIN_ID, BNB_WEI,
)

RPC = "https://bsc-dataseed1.binance.org"
ABI_PATH = "/root/pancakebot/abi/prediction_v2_abi.json"
WALLET_FILE = "/etc/pancakebot/experiment_wallets.env"
# Production sweep destination (derived from /etc/pancakebot/pancakebot.env's
# BSC_WALLET_PRIVATE_KEY on 2026-06-08; hardcoded so this script never reads the
# production key). Re-verified at runtime against the 0xaF96…A5E7 abbreviation.
PROD_WALLET = "0xaF966D00698F92DeBe2127136D5159c5a51dA5E7"
GAS_PRICE_WEI = 1_000_000_000          # 1 gwei
SWEEP_GAS = 21_000                     # plain BNB transfer


def out(msg):
    print(msg, flush=True)


def load_test_wallets():
    kv = {}
    for line in open(WALLET_FILE):
        line = line.strip()
        if line and "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            kv[k] = v
    return {i: (kv["WALLET_%d_KEY" % i], kv["WALLET_%d_ADDR" % i]) for i in range(5)}


def confirm():
    return input("    >>> SEND this TX? type 'yes' to confirm: ").strip() == "yes"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true", help="actually send TXs (default: dry-run)")
    ap.add_argument("--dry-run", action="store_true", help="explicit dry-run (forces dry even with --execute)")
    args = ap.parse_args()
    execute = args.execute and not args.dry_run

    out("=" * 72)
    out("sweep_and_claim — %s" % ("EXECUTE — REAL TXs WILL BE SENT" if execute else "DRY-RUN — nothing sent"))
    out("=" * 72)
    if not execute:
        out("DRY-RUN: no signing, no sends. Re-run with --execute to send (prompts per TX).")

    w3 = Web3(Web3.HTTPProvider(RPC, request_kwargs={"timeout": 15}))
    abi = json.load(open(ABI_PATH))
    c = w3.eth.contract(address=Web3.to_checksum_address(PREDICTION_V2_CONTRACT_ADDRESS), abi=abi)

    prod = Web3.to_checksum_address(PROD_WALLET)
    if not (prod.lower().startswith("0xaf96") and prod.lower().endswith("a5e7")):
        raise SystemExit("ABORT: PROD_WALLET %s does not match expected 0xaF96…A5E7" % prod)
    out("sweep destination (production): %s" % prod)

    wallets = load_test_wallets()
    test_addrs = {Web3.to_checksum_address(a) for _, a in wallets.values()}
    if prod in test_addrs:
        raise SystemExit("ABORT: production address equals a test wallet — refusing")

    tot_start = tot_swept = 0
    for i in range(5):
        key, expected = wallets[i]
        acct = Account.from_key(key)
        if int(acct.address, 16) == 0:
            raise SystemExit("ABORT: W%d derived an all-zeros address" % i)
        if acct.address.lower() != expected.lower():
            raise SystemExit("ABORT: W%d derived %s != env %s" % (i, acct.address, expected))
        addr = acct.address

        out("\n" + "-" * 64)
        out("W%d  %s" % (i, addr))
        start = w3.eth.get_balance(addr)
        tot_start += start
        out("  start balance: %.6f BNB" % (start / BNB_WEI))

        length = c.functions.getUserRoundsLength(addr).call()
        epochs, cur = [], 0
        while cur < length:
            size = min(100, length - cur)
            res = c.functions.getUserRounds(addr, cur, size).call()
            epochs.extend(int(e) for e in res[0])
            cur += size
        out("  bet rounds on-chain: %d" % len(epochs))

        claim_eps, est_win = [], 0
        for ep in epochs:
            try:
                if not c.functions.claimable(ep, addr).call():
                    continue
            except Exception as e:                       # noqa: BLE001
                out("    claimable(%d) error: %s" % (ep, e))
                continue
            claim_eps.append(ep)
            try:
                bet = int(c.functions.ledger(ep, addr).call()[1])
                rd = c.functions.rounds(ep).call()       # [11]=rewardBaseCalAmount [12]=rewardAmount
                rbase, ramt = int(rd[11]), int(rd[12])
                est_win += (bet * ramt // rbase) if rbase > 0 else 0
            except Exception:                            # noqa: BLE001
                pass
        if not epochs:
            out("  no bets on-chain for this wallet")
        elif not claim_eps:
            out("  no winnings to claim (all rounds lost or already claimed)")
        else:
            out("  unclaimed winning rounds: %d  epochs=%s  est winnings ~%.6f BNB"
                % (len(claim_eps), claim_eps, est_win / BNB_WEI))

        nonce0 = w3.eth.get_transaction_count(addr)

        # ---- TX 1: CLAIM ----
        if claim_eps:
            try:
                gas = int(c.functions.claim(claim_eps).estimate_gas({"from": addr}) * 1.3)
            except Exception:                            # noqa: BLE001
                gas = 80_000 + 60_000 * len(claim_eps)
            tx = c.functions.claim(claim_eps).build_transaction(
                {"from": addr, "nonce": nonce0, "gas": gas,
                 "gasPrice": GAS_PRICE_WEI, "chainId": EXPECTED_CHAIN_ID})
            out("  [TX 1 CLAIM] from=%s to=%s epochs=%s value=0 gas=%d gasPrice=1gwei nonce=%d"
                % (addr, tx["to"], claim_eps, gas, nonce0))
            out("              data=%s..." % tx["data"][:26])
            if execute and confirm():
                signed = acct.sign_transaction(tx)
                h = w3.eth.send_raw_transaction(signed.raw_transaction).hex()
                h = h if str(h).startswith("0x") else "0x" + h
                out("              sent %s — waiting receipt..." % h)
                r = w3.eth.wait_for_transaction_receipt(h, timeout=180)
                out("              mined block=%d status=%d" % (r["blockNumber"], r["status"]))
            elif execute:
                out("              NOT confirmed — claim skipped")

        # ---- TX 2: SWEEP ----
        if execute:
            bal = w3.eth.get_balance(addr)
            nonce = w3.eth.get_transaction_count(addr)
        else:
            bal = start + est_win                        # dry projection
            nonce = nonce0 + (1 if claim_eps else 0)
        sweep_amt = bal - SWEEP_GAS * GAS_PRICE_WEI
        if sweep_amt <= 0:
            out("  [sweep] balance after gas <= 0 (%.6f BNB) — nothing to sweep" % (sweep_amt / BNB_WEI))
            sweep_amt = 0
        else:
            tx = {"from": addr, "to": prod, "value": int(sweep_amt), "nonce": nonce,
                  "gas": SWEEP_GAS, "gasPrice": GAS_PRICE_WEI, "chainId": EXPECTED_CHAIN_ID}
            out("  [TX 2 SWEEP] from=%s to=%s value=%.6f BNB gas=%d gasPrice=1gwei nonce=%d"
                % (addr, prod, sweep_amt / BNB_WEI, SWEEP_GAS, nonce))
            if execute and confirm():
                signed = acct.sign_transaction(tx)
                h = w3.eth.send_raw_transaction(signed.raw_transaction).hex()
                h = h if str(h).startswith("0x") else "0x" + h
                out("              sent %s — waiting receipt..." % h)
                r = w3.eth.wait_for_transaction_receipt(h, timeout=180)
                out("              mined block=%d status=%d" % (r["blockNumber"], r["status"]))
                tot_swept += sweep_amt
            elif execute:
                out("              NOT confirmed — sweep skipped")
                sweep_amt = 0

        final = w3.eth.get_balance(addr) if execute else None
        out("  SUMMARY W%d: start=%.6f  winning_rounds=%d  est_winnings~%.6f  %s  %s"
            % (i, start / BNB_WEI, len(claim_eps), est_win / BNB_WEI,
               ("final=%.6f" % (final / BNB_WEI)) if execute else "final=(dry)",
               ("swept=%.6f" % (sweep_amt / BNB_WEI)) if execute else ("would_sweep~%.6f" % (max(0, sweep_amt) / BNB_WEI))))

    out("\n" + "=" * 72)
    out("TOTAL: start=%.6f BNB across 5 wallets   %s"
        % (tot_start / BNB_WEI,
           ("swept=%.6f BNB -> %s" % (tot_swept / BNB_WEI, prod)) if execute else "(dry-run — nothing sent)"))
    if not execute:
        out("Re-run with --execute to send. Each TX prompts for 'yes'. Tip: `... --execute | tee sweep.log`.")


if __name__ == "__main__":
    main()
