"""p4c offline-replay diagnostic: re-fire canonical gate on epochs 477568-477601.

Goal: determine whether the live dry bot's 0-firings-in-33-rounds streak
post-p4c-fix is (a) genuine quiet market, or (b) a secondary issue distinct
from the timing-guard regression.

Method: for each epoch in the replay range, fetch the lock_ts from on-chain
``rounds(epoch)`` and re-call the SAME ``MomentumGate.evaluate(lock_at_ms)``
the live bot uses. Compare per-epoch result to the live bot's recorded
``gate_no_signal``.

Disagreement metric:
  - 0 disagreements -> hypothesis (a) confirmed; gate truly didn't fire
  - >0 disagreements -> hypothesis (b); secondary issue worth investigating

Constraints respected:
  - Gate's data horizon: ``newest_open_ms = lock_at - cutoff*1000 - 1000``
    means data through ``lock_at - kline_cutoff_seconds`` only. Re-using
    ``MomentumGate.evaluate`` ensures bit-identical horizon enforcement.
  - Throttle: 1s sleep between epochs to avoid OKX rate-budget contention
    with the running live bot in a different process.
  - Read-only: no on-chain bets, no state mutation, no log file writes.

Usage::

    py research/p4c_offline_replay.py 477568 477601
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pancakebot.config import load_app_config  # noqa: E402
from pancakebot.chain.prediction_contract import (  # noqa: E402
    Web3PredictionContract,
    Web3ContractConfig,
)
from pancakebot.market_data.okx_client import OkxClient  # noqa: E402
from pancakebot.strategy.momentum_gate import (  # noqa: E402
    MomentumGate,
    MomentumGateConfig,
)
from pancakebot.constants import WRITE_PATH_RPC_URLS, EXPECTED_CHAIN_ID, WRITE_PATH_RPC_TIMEOUT_SECONDS  # noqa: E402
from pancakebot import paths as _paths  # noqa: E402
from pancakebot.chain.rpc_pool import choose_rpc_url  # noqa: E402


def main(epoch_start: int, epoch_end_inclusive: int) -> int:
    cfg = load_app_config(str(_REPO_ROOT / "config.toml"))

    rpc_url = choose_rpc_url(
        WRITE_PATH_RPC_URLS,
        expected_chain_id=int(EXPECTED_CHAIN_ID),
        timeout_seconds=int(WRITE_PATH_RPC_TIMEOUT_SECONDS),
    )
    contract = Web3PredictionContract(Web3ContractConfig(
        rpc_url=rpc_url,
        rpc_urls=tuple(WRITE_PATH_RPC_URLS),
        abi_json_path=_paths.ABI_JSON_PATH,
        private_key="",
    ))

    okx_client = OkxClient(timeout_seconds=10.0)

    momentum_gate_cfg = MomentumGateConfig(
        enabled=True,
        bnb_symbol="BNB-USDT",
        btc_symbol="BTC-USDT",
        eth_symbol="ETH-USDT",
        sol_symbol="SOL-USDT",
        kline_cutoff_seconds=cfg.kline_cutoff_seconds,
        mtf_lookbacks=cfg.strategy.gate.mtf_lookbacks,
        mtf_min_return_threshold=cfg.strategy.gate.mtf_min_return_threshold,
    )
    gate = MomentumGate(config=momentum_gate_cfg, okx_client=okx_client)

    r2_min_strength = cfg.strategy.eth_sol_fallback.signal.min_signal_strength

    print(f"# p4c offline-replay over epochs [{epoch_start}..{epoch_end_inclusive}]")
    print(f"# config: cutoff={cfg.kline_cutoff_seconds}s "
          f"lookbacks={cfg.strategy.gate.mtf_lookbacks} "
          f"btc_threshold={cfg.strategy.gate.mtf_min_return_threshold} "
          f"r2_min_strength={r2_min_strength}")
    print()
    header = f"{'epoch':>8} {'btc_sig':>8} {'eth_sig':>8} {'eth_str':>10} {'sol_sig':>8} {'sol_str':>10} {'r2_str':>10} {'r2?':>4} {'final':>16}"
    print(header)
    print("-" * len(header))

    btc_fired_count = 0
    r2_fired_count = 0
    error_count = 0
    skip_reasons: dict[str, int] = {}

    for epoch in range(epoch_start, epoch_end_inclusive + 1):
        try:
            rd = contract.round_data(epoch)
        except Exception as e:  # noqa: BLE001
            print(f"{epoch:>8} {'<rpc_fail>':>12}  err={e}")
            error_count += 1
            time.sleep(1.0)
            continue

        if rd.lock_ts <= 0:
            print(f"{epoch:>8} {rd.lock_ts:>12} <future_or_unset>")
            error_count += 1
            time.sleep(1.0)
            continue

        lock_at_ms = int(rd.lock_ts) * 1000
        try:
            res = gate.evaluate(lock_at_ms=lock_at_ms)
        except Exception as e:  # noqa: BLE001
            print(f"{epoch:>8} {rd.lock_ts:>12}  err={type(e).__name__}: {e}")
            error_count += 1
            time.sleep(1.0)
            continue

        btc_sig = str(res.signal) if res.signal is not None else "-"
        eth_sig = str(res.eth_signal) if res.eth_signal is not None else "-"
        sol_sig = str(res.sol_signal) if res.sol_signal is not None else "-"
        eth_str_v = res.eth_signal_strength or 0.0
        sol_str_v = res.sol_signal_strength or 0.0

        # Apply pipeline regime-2 logic offline:
        #   regime-2 fires iff (BTC silent) AND (ETH+SOL same direction)
        #   AND min(eth_str, sol_str) >= r2_min_strength.
        r2_candidate = (
            res.signal is None
            and res.eth_signal is not None
            and res.sol_signal is not None
            and res.eth_signal == res.sol_signal
        )
        if r2_candidate:
            r2_str = min(eth_str_v, sol_str_v)
            r2_would_fire = r2_str >= r2_min_strength
        else:
            r2_str = 0.0
            r2_would_fire = False

        if res.signal is not None:
            final = f"{res.signal}_btc"
            btc_fired_count += 1
        elif r2_would_fire:
            final = f"{res.eth_signal}_r2"
            r2_fired_count += 1
        else:
            final = "gate_no_signal"

        r2_marker = "Y" if r2_would_fire else ("?" if r2_candidate else "-")
        r2_str_display = f"{r2_str:.6f}" if r2_candidate else "-"

        print(f"{epoch:>8} {btc_sig:>8} {eth_sig:>8} {eth_str_v:>10.6f} "
              f"{sol_sig:>8} {sol_str_v:>10.6f} {r2_str_display:>10} {r2_marker:>4} {final:>16}")

        if final not in skip_reasons:
            skip_reasons[final] = 0
        skip_reasons[final] += 1

        # Throttle to avoid contending with live bot's OKX rate budget.
        time.sleep(1.0)

    total_fired = btc_fired_count + r2_fired_count
    print()
    print(f"# RESULTS")
    print(f"# epochs replayed: {epoch_end_inclusive - epoch_start + 1}")
    print(f"# btc-primary firings: {btc_fired_count}")
    print(f"# regime-2 firings: {r2_fired_count}")
    print(f"# total pipeline-level firings: {total_fired}")
    print(f"# rpc/fetch errors: {error_count}")
    print(f"# final-decision distribution: {skip_reasons}")
    print()

    # Live bot recorded gate_no_signal for ALL rounds in the replay range
    # (per p4c implementer at T+165min). Disagreement = offline says
    # Bull/Bear (BTC primary OR regime-2) but live said gate_no_signal.
    if total_fired == 0:
        print("# VERDICT: 0 firings offline = 0 firings live -> AGREEMENT.")
        print("#   Hypothesis (a) confirmed: genuine quiet-market overlap.")
        print("#   Gate is working as designed; just unlucky in this window.")
        return 0
    else:
        print(f"# VERDICT: {total_fired} firings offline vs 0 firings live "
              f"-> DISAGREEMENT.")
        print(f"#   ({btc_fired_count} BTC-primary + {r2_fired_count} regime-2)")
        print("#   Hypothesis (b): secondary issue distinct from p4c timing")
        print("#   guard regression. Surface to user for investigation.")
        return 2  # non-zero exit signals disagreement


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <epoch_start> <epoch_end_inclusive>")
        sys.exit(1)
    sys.exit(main(int(sys.argv[1]), int(sys.argv[2])))
