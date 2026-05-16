"""p4c canonical-loop staleness probe.

A thin instrumented wrapper around the bot's actual fetch+decide loop.
NO new fetch logic. NO new skew logic. The probe imports and calls the
canonical helpers (``engine._refresh_clock_skew``, ``engine._utc_now``,
``engine._sleep_until_ts``, ``MomentumGate.evaluate``) exactly as
``engine._run_one_iteration`` does -- minus the BSC pool-watcher and
the bet/settle paths.

Per round we:
  1. Refresh clock skew via ``engine._refresh_clock_skew(gate)`` (the
     canonical housekeeping refresh; uses ``client.measure_clock_skew(samples=3)``).
  2. Pick a synthetic ``lock_at_okx_ms`` = next 5s boundary in OKX-time.
  3. Sleep until wake = ``lock_at - kline_fetch_offset_ms`` using
     ``engine._sleep_until_ts`` (skew-corrected, via ``engine._utc_now``).
  4. Call ``gate.evaluate(lock_at_ms=lock_at_okx_ms)`` -- THE canonical
     decision-time call (4-parallel REST GETs, ``RETRY_GATE`` policy).
  5. Record per-sample timing.

Output: ``var/extended/okx_staleness_probe.jsonl``.

Usage::

    py research/p4c_canonical_loop_probe.py [N_rounds]

Default N_rounds = 50 (each round measures all 4 symbols via the
canonical 4-parallel orchestration). Each round takes ~5s wall-clock.
n=200 -> ~17 minutes.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pancakebot.config import load_app_config  # noqa: E402
from pancakebot.market_data.okx_client import OkxClient  # noqa: E402
from pancakebot.strategy.momentum_gate import (  # noqa: E402
    MomentumGate,
    MomentumGateConfig,
)
from pancakebot.runtime import engine as canonical_engine  # noqa: E402

OUT_PATH = _REPO_ROOT / "var" / "extended" / "okx_staleness_probe.jsonl"
SYNTHETIC_LOCK_BOUNDARY_S = 5  # synthetic lock_at lands on OKX-time multiples of this


def main(n_rounds: int, offset_ms_override: int | None = None) -> int:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    cfg = load_app_config(str(_REPO_ROOT / "config.toml"))
    # Optional CLI override on the canonical wake offset. Used to sweep
    # different post-close fetch points without modifying config.toml.
    # We monkey-patch the loaded config object's value -- the probe loop
    # below reads cfg.kline_fetch_offset_ms when computing wake_okx_s.
    if offset_ms_override is not None:
        # AppConfig is a frozen dataclass; rebuild via dataclasses.replace.
        from dataclasses import replace
        cfg = replace(cfg, kline_fetch_offset_ms=int(offset_ms_override))

    # Build canonical OKX client + gate, exact same construction as app.py:
    #   - OkxClient(timeout_seconds=10.0)
    #   - client.warmup() to pre-establish connections
    #   - MomentumGateConfig matching cfg.kline_cutoff_seconds + cfg.strategy.gate
    #   - MomentumGate(config, okx_client) wires the canonical 4-parallel orchestrator.
    client = OkxClient(timeout_seconds=10.0)
    client.warmup()  # canonical warmup -- 4 parallel /public/time GETs
    gate_cfg = MomentumGateConfig(
        enabled=True,
        bnb_symbol="BNB-USDT",
        btc_symbol="BTC-USDT",
        eth_symbol="ETH-USDT",
        sol_symbol="SOL-USDT",
        cutoff_seconds=cfg.kline_cutoff_seconds,
        mtf_lookbacks=cfg.strategy.gate.mtf_lookbacks,
        mtf_threshold=cfg.strategy.gate.mtf_threshold,
    )
    gate = MomentumGate(config=gate_cfg, okx_client=client)

    print(f"# canonical-loop probe: kline_fetch_offset_ms={cfg.kline_fetch_offset_ms} "
          f"cutoff_seconds={cfg.kline_cutoff_seconds} n_rounds={n_rounds}")

    samples_recorded = 0
    timeouts_or_errors = 0
    with OUT_PATH.open("a", encoding="utf-8") as f:
        for round_idx in range(n_rounds):
            # --- Step 1: Canonical housekeeping skew refresh --------------------
            # Calls engine._refresh_clock_skew(gate) which calls
            # client.measure_clock_skew(samples=3) and updates the engine
            # module's _clock_skew_seconds. engine._utc_now() will then return
            # time.time() - this skew. Identical to engine.py:500.
            canonical_engine._refresh_clock_skew(gate)
            skew_s = canonical_engine._clock_skew_seconds  # for logging

            # --- Step 2: Pick synthetic lock_at in OKX-time ---------------------
            # OKX-now via _utc_now() = time.time() - skew (canonical math).
            # Round up to next SYNTHETIC_LOCK_BOUNDARY_S boundary, with a small
            # buffer so wake-time is comfortably in the future.
            okx_now_s = canonical_engine._utc_now()
            lock_at_okx_s = float(int(okx_now_s / SYNTHETIC_LOCK_BOUNDARY_S) + 1) \
                * SYNTHETIC_LOCK_BOUNDARY_S
            wake_okx_s = lock_at_okx_s - cfg.kline_fetch_offset_ms / 1000.0
            if wake_okx_s - okx_now_s < 0.5:  # too close, skip to next boundary
                lock_at_okx_s += SYNTHETIC_LOCK_BOUNDARY_S
                wake_okx_s = lock_at_okx_s - cfg.kline_fetch_offset_ms / 1000.0
            lock_at_okx_ms = int(lock_at_okx_s * 1000)
            candle_close_okx_ms = lock_at_okx_ms - cfg.kline_cutoff_seconds * 1000

            # --- Step 3: Canonical _sleep_until_ts ------------------------------
            # Same helper engine._run_one_iteration uses at line 554. Internally
            # uses _utc_now() so the skew correction applies.
            canonical_engine._sleep_until_ts(
                wake_okx_s, reason="probe_wait_for_kline_fetch",
                epoch=lock_at_okx_ms // 1000,
            )

            # --- Step 4: Canonical gate.evaluate --------------------------------
            # Same call site engine._run_one_iteration's
            # closed.strategy_pipeline.decide_open_round delegates to. Fires 4
            # parallel /history-candles GETs via ThreadPoolExecutor, with
            # RETRY_GATE retry policy (max_attempts=2, 2.5s backoff).
            t_call_start = time.perf_counter()
            try:
                result = gate.evaluate(lock_at_ms=lock_at_okx_ms)
            except Exception as e:  # noqa: BLE001 -- canonical InvariantError on shape violation
                timeouts_or_errors += 1
                f.write(json.dumps({
                    "round_idx": round_idx,
                    "lock_at_okx_ms": lock_at_okx_ms,
                    "skew_s": skew_s,
                    "error": f"{type(e).__name__}: {str(e)[:200]}",
                }) + "\n")
                f.flush()
                continue
            t_call_end = time.perf_counter()
            gate_elapsed_ms = int((t_call_end - t_call_start) * 1000)

            # gate.last_fetch_timing is {"btc_ms": N, "eth_ms": N, ...} from
            # the SUCCESSFUL attempts only (per okx_client.py docstring).
            timing = gate.last_fetch_timing or {}
            # First-try proxy: RETRY_GATE has 2.5s backoff before retry, so
            # gate_elapsed_ms < 2500ms strongly indicates no symbol needed
            # a retry (all 4 succeeded on first attempt).
            first_try_round = gate_elapsed_ms < 2500

            record = {
                "round_idx": round_idx,
                "lock_at_okx_ms": lock_at_okx_ms,
                "candle_close_okx_ms": candle_close_okx_ms,
                "wake_okx_ms": int(wake_okx_s * 1000),
                "skew_s": skew_s,
                "gate_elapsed_ms": gate_elapsed_ms,
                "btc_ms": timing.get("btc_ms"),
                "eth_ms": timing.get("eth_ms"),
                "sol_ms": timing.get("sol_ms"),
                "bnb_ms": timing.get("bnb_ms"),
                "skip_reason": result.skip_reason,
                "signal": str(result.signal) if result.signal is not None else None,
                "first_try_round": bool(first_try_round),
            }
            f.write(json.dumps(record) + "\n")
            f.flush()
            samples_recorded += 1

            if (round_idx + 1) % 10 == 0:
                print(f"# round {round_idx+1}/{n_rounds} "
                      f"elapsed={gate_elapsed_ms}ms "
                      f"first_try={first_try_round} "
                      f"skew={skew_s*1000:.0f}ms "
                      f"skip={result.skip_reason or 'ok'}",
                      flush=True)

    print()
    print(f"# DONE: {samples_recorded} rounds recorded, "
          f"{timeouts_or_errors} errors")
    print(f"# output: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    offset = int(sys.argv[2]) if len(sys.argv) > 2 else None
    sys.exit(main(n, offset))
