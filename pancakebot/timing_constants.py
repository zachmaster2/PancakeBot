"""Empirical timing constants for the per-round wake/fetch/decide/submit chain.

This module is the SINGLE SOURCE OF TRUTH for the timing-budget constants
that the live decision-path config derives from. Each constant cites its
empirical provenance: the probe script that measured it, the date, the
percentile, and the sample size.

Co-update discipline: if any environment-dependent constant changes
(network latency, BSC behavior, OKX publishing characteristics), the
corresponding probe script in ``research/`` must be re-run, and the
value here must be updated co-locked with a new measurement date.

Derivation chain (computed at config-load time in ``pancakebot/config.py``):

    lock_safety_margin_ms      = BSC_SUBMIT_RTT_P95_MS + BSC_BLOCK_TIME_MS
                                 + SAFETY_BUFFER_MS
    kline_wakeup_offset_ms     = lock_safety_margin_ms + OKX_FETCH_RTT_P95_MS
                                 + GATE_COMPUTE_MS
    pool_wakeup_offset_ms      = kline_wakeup_offset_ms + POOL_READ_BUFFER_MS
    skew_sync_wakeup_offset_ms = pool_wakeup_offset_ms + SKEW_SYNC_TIME_P99_MS
                                 + SKEW_SAFETY_BUFFER_MS

Cross-validations enforced at config load:

    kline_cutoff_seconds * 1000 >= OKX_PUBLISH_DELAY_P99_MS
        (otherwise the gate asks OKX for a candle still being published)

    pool_cutoff_seconds * 1000 >= WSS_ARRIVAL_DELAY_P99_MS
        (otherwise the pool aggregate excludes bets still en route via WSS)
"""
from __future__ import annotations


# --- OKX REST timing -------------------------------------------------------

# OKX /history-candles publishing latency (= time between candle close and
# the candle being available via REST /history-candles for our 4-symbol
# parallel fetch). Per-symbol p99.
#
# Source: research/p4c_canonical_loop_probe.py at varying wake offsets,
#         n=1000 + n=200 prior, 2026-05-02..2026-05-03.
# Method: probe @ wake=850ms (1150ms post-close) → 98.3% per-symbol
#         first-try; probe @ wake=1200ms (800ms post-close) → 96.9%.
#         Implied per-symbol p99 staleness ~1200-1300ms.
# Last measured: 2026-05-03
OKX_PUBLISH_DELAY_P99_MS: int = 1300

# OKX REST round-trip time for /history-candles fetches. Pooled p95 over
# the 4 symbols (BTC, ETH, SOL, BNB).
#
# Source: research/p4c_canonical_loop_probe.py n=1000, 2026-05-03.
# Pooled (n=4000 fetches): p50=258, p90=277, p95=289, p99=363, max=981.
# Last measured: 2026-05-03
OKX_FETCH_RTT_P95_MS: int = 290


# --- Strategy compute ------------------------------------------------------

# Time the gate's signal-computation logic takes after kline data arrives.
# In practice this is sub-ms (numpy ops on 16-element arrays), but we
# allow a small headroom for variance.
#
# Source: code inspection of momentum_gate._compute_signal,
#         pancakebot/strategy/momentum_gate.py:415-... (numpy diff/sum
#         over 16-row arrays, no I/O).
# Last measured: 2026-05-03 (engineering judgment, not empirical probe)
GATE_COMPUTE_MS: int = 50


# --- BSC chain timing ------------------------------------------------------

# Average BSC block production interval. Used for worst-case TX-inclusion
# math: a TX broadcast at time T lands in the next block sometime within
# [T, T + BSC_BLOCK_TIME_MS].
#
# Source: research/p4c_bsc_block_probe.py n=200 consecutive blocks,
#         2026-05-03. Mean inter-block delta = 0.452s. Conservative
#         rounding to 500ms.
# Last measured: 2026-05-03
BSC_BLOCK_TIME_MS: int = 500

# BSC RPC submit RTT (eth_sendRawTransaction). EMPIRICALLY MEASURED
# AS A LOWER BOUND via eth_blockNumber proxy (p99=57ms in
# research/p4c_bsc_rpc_probe.py n=200). Production sendRawTransaction
# involves mempool insertion and may have higher RTT than a cached
# read. Use 200ms as a conservative interim estimate; revisit when a
# dedicated sendRawTransaction probe is feasible.
#
# Source: research/p4c_bsc_rpc_probe.py n=200, 2026-05-03 (lower bound).
#         Production estimate: 200ms (no direct measurement; would
#         require gas-spending probe to be precise).
# Last measured: 2026-05-03 (proxy + estimate)
BSC_SUBMIT_RTT_P95_MS: int = 200


# --- WSS subscriber timing ------------------------------------------------

# Time between a BSC block being mined (= block.timestamp) and the
# corresponding BetBull/BetBear event being received by our WSS
# subscriber. Per-event p99 over the 30-min sample window.
#
# Source: research/p4c_wss_arrival_probe.py n=219 events over 30min,
#         2026-05-03. p50=1041ms, p90=1723, p95=1777, p99=3106,
#         p99.9=3834.
# Last measured: 2026-05-03
WSS_ARRIVAL_DELAY_P99_MS: int = 3500  # p99=3106 + ~400ms buffer


# --- Skew sync ------------------------------------------------------------

# Wall-clock duration of one canonical clock-skew refresh
# (OkxClient.measure_clock_skew(samples=3), which fires 3 sequential
# GETs to OKX /api/v5/public/time). Per-call p99.
#
# Source: research/p4c_skew_probe.py n=100, 2026-05-03.
#         min=1635, p50=1890, p90=2236, p95=2294, p99=2409, max=2409.
# Last measured: 2026-05-03
SKEW_SYNC_TIME_P99_MS: int = 2500  # p99=2409 + ~100ms buffer


# --- Static buffers --------------------------------------------------------

# Small constant for the in-memory pool data read (engine.py:531-534).
# The PoolEventWatcher's get_pool() is a dict + list filter under a
# lock — sub-ms in practice. Buffer kept as a placeholder for the
# wake-schedule arithmetic and any future GIL-contention budget.
#
# Source: code inspection of pool_watcher.PoolEventWatcher.get_pool.
# Last measured: 2026-05-03 (engineering judgment)
POOL_READ_BUFFER_MS: int = 5

# Headroom on the lock_safety_margin_ms derivation. Beyond
# BSC_SUBMIT_RTT_P95 + BSC_BLOCK_TIME, this absorbs second-order
# variance in TX submission (signing time, sign-to-send dispatch) and
# any clock-jitter on the engine's timing-guard check.
#
# Source: engineering judgment.
# Last measured: 2026-05-03
SAFETY_BUFFER_MS: int = 50

# Headroom on the skew_sync_wakeup_offset_ms derivation. Beyond
# SKEW_SYNC_TIME_P99, this absorbs dispatch overhead between the
# skew refresh completing and the next wake firing.
#
# Source: engineering judgment.
# Last measured: 2026-05-03
SKEW_SAFETY_BUFFER_MS: int = 50
