"""Probe RPC pool data around cutoff and lock_at for several rounds.

Polls round_data() every ~200ms from cutoff-2s through lock_at+2s,
recording bull/bear pool amounts at each sample. Shows exactly when
and how much pools change relative to what backtest would see.

Run while dry mode is NOT running (they'd compete for the same epochs).
"""
from __future__ import annotations

import sys
import time

sys.path.insert(0, ".")

from pancakebot.config.load_config import load_app_config
from pancakebot.core.constants import BNB_WEI, EXPECTED_CHAIN_ID, RPC_TIMEOUT_SECONDS, RPC_URLS
from pancakebot.config.env import load_env, require_env
from pancakebot.infra.rpc_pool import choose_rpc_url
from pancakebot.infra.onchain.web3_contract_config import Web3ContractConfig
from pancakebot.infra.onchain.web3_prediction_contract import Web3PredictionContract

NUM_ROUNDS = 3          # how many rounds to probe
POLL_INTERVAL = 0.2     # seconds between polls
PRE_CUTOFF_LEAD = 2.0   # start polling this many seconds before cutoff
POST_LOCK_TRAIL = 2.0   # keep polling this many seconds after lock_at


def main():
    load_env()
    cfg = load_app_config("config.toml")
    private_key = require_env("BSC_WALLET_PRIVATE_KEY")

    rpc_url = choose_rpc_url(
        RPC_URLS,
        expected_chain_id=int(EXPECTED_CHAIN_ID),
        timeout_seconds=int(RPC_TIMEOUT_SECONDS),
    )
    contract = Web3PredictionContract(Web3ContractConfig(
        rpc_url=rpc_url,
        abi_json_path=cfg.abi_json_path,
        private_key=private_key,
    ))

    cutoff_seconds = int(cfg.cutoff_seconds)

    print(f"Pool Probe: polling every {POLL_INTERVAL}s, {NUM_ROUNDS} rounds")
    print(f"cutoff_seconds={cutoff_seconds}")
    print()

    for round_i in range(NUM_ROUNDS):
        # Find current epoch and its lock_at
        current_epoch = contract.current_epoch()
        rd = contract.round_data(current_epoch)
        lock_at = rd.lock_ts

        if lock_at <= 0:
            print(f"Round {round_i+1}: epoch {current_epoch} has no lock_ts yet, waiting...")
            time.sleep(5)
            continue

        cutoff_ts = lock_at - cutoff_seconds
        now = time.time()

        # Wait until PRE_CUTOFF_LEAD seconds before cutoff
        wait_until = cutoff_ts - PRE_CUTOFF_LEAD
        if now < wait_until:
            sleep_dur = wait_until - now
            print(f"Round {round_i+1}: epoch {current_epoch}, lock_at={lock_at}, "
                  f"cutoff={cutoff_ts}, sleeping {sleep_dur:.0f}s until probe window...")
            time.sleep(sleep_dur)

        print(f"\n{'='*70}")
        print(f"PROBING epoch {current_epoch} | cutoff={cutoff_ts} | lock_at={lock_at}")
        print(f"{'='*70}")
        print(f"{'Time':>10s}  {'Rel':>7s}  {'Bull BNB':>10s}  {'Bear BNB':>10s}  "
              f"{'Total BNB':>10s}  {'Bull PM':>8s}  {'Bear PM':>8s}")
        print("-" * 70)

        stop_at = lock_at + POST_LOCK_TRAIL
        samples = []

        while time.time() < stop_at:
            t0 = time.time()
            try:
                rd = contract.round_data(current_epoch)
                t1 = time.time()
                bull = rd.bull_amount_wei / BNB_WEI
                bear = rd.bear_amount_wei / BNB_WEI
                total = bull + bear
                bull_pm = total * 0.97 / bull if bull > 0 else 0
                bear_pm = total * 0.97 / bear if bear > 0 else 0

                rel = t1 - cutoff_ts
                label = ""
                if abs(rel) < 0.3:
                    label = " <-- CUTOFF"
                elif abs(t1 - lock_at) < 0.3:
                    label = " <-- LOCK_AT"

                print(f"{t1:.1f}  {rel:+7.2f}s  {bull:10.4f}  {bear:10.4f}  "
                      f"{total:10.4f}  {bull_pm:8.3f}  {bear_pm:8.3f}{label}")

                samples.append((rel, bull, bear, total, t1 - t0))
            except Exception as e:
                print(f"  RPC error: {e}")

            # Sleep remainder of interval
            elapsed = time.time() - t0
            if elapsed < POLL_INTERVAL:
                time.sleep(POLL_INTERVAL - elapsed)

        # Summary
        if samples:
            pre_cutoff = [(r, b, br, t) for r, b, br, t, _ in samples if r < 0]
            at_cutoff = [(r, b, br, t) for r, b, br, t, _ in samples if 0 <= r < 1]
            at_lock = [(r, b, br, t) for r, b, br, t, _ in samples if abs(r - cutoff_seconds) < 1]
            final = samples[-1]

            print(f"\nSummary epoch {current_epoch}:")
            if pre_cutoff:
                print(f"  Pre-cutoff:  bull={pre_cutoff[-1][1]:.4f} bear={pre_cutoff[-1][2]:.4f} total={pre_cutoff[-1][3]:.4f}")
            if at_cutoff:
                print(f"  At cutoff:   bull={at_cutoff[0][1]:.4f} bear={at_cutoff[0][2]:.4f} total={at_cutoff[0][3]:.4f}")
            if at_lock:
                print(f"  At lock_at:  bull={at_lock[0][1]:.4f} bear={at_lock[0][2]:.4f} total={at_lock[0][3]:.4f}")
            print(f"  Final:       bull={final[1]:.4f} bear={final[2]:.4f} total={final[3]:.4f}")

            # Delta between cutoff and final
            if at_cutoff:
                d_bull = final[1] - at_cutoff[0][1]
                d_bear = final[2] - at_cutoff[0][2]
                d_total = final[3] - at_cutoff[0][3]
                print(f"  Delta (cutoff->final): bull={d_bull:+.4f} bear={d_bear:+.4f} total={d_total:+.4f}")
                if at_cutoff[0][3] > 0:
                    print(f"  Delta as % of pool: {abs(d_total)/at_cutoff[0][3]*100:.1f}%")

            avg_rpc = sum(l for _, _, _, _, l in samples) / len(samples)
            print(f"  Avg RPC latency: {avg_rpc*1000:.0f}ms ({len(samples)} samples)")

        print()

        # Wait for next round
        if round_i < NUM_ROUNDS - 1:
            next_lock = lock_at + 306  # ~5 min rounds
            wait = max(1, next_lock - cutoff_seconds - PRE_CUTOFF_LEAD - time.time())
            print(f"Waiting {wait:.0f}s for next round...")
            time.sleep(wait)


if __name__ == "__main__":
    main()
