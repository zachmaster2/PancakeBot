"""Empirical timing constants for the per-round wake/fetch/decide/submit chain.

This module is the SINGLE SOURCE OF TRUTH for the timing-budget constants
that the live decision-path config derives from. Each constant cites its
empirical provenance: the probe script that measured it, the date, the
percentile, and the sample size.

Naming convention: ``<subsystem>_<event>_<measurement>``. Constants are
named for the subsystem they characterize (OKX, BSC, WSS), the specific
event being measured (kline publish, bet submit, bet event arrival), and
the statistic (P99/P95/TIME/SAFETY_BUFFER).

Co-update discipline: if any environment-dependent constant changes
(network latency, BSC behavior, OKX publishing characteristics), the
corresponding probe script in ``research/`` must be re-run, and the
value here must be updated co-locked with a new measurement date.

Derivation chain (computed at config-load time in ``pancakebot/config.py``):

    bet_submit_deadline_offset_ms  = (BSC_BET_SUBMIT_RTT_P95_MS
                                      + BSC_BLOCK_TIME_MS
                                      + BET_SUBMIT_SAFETY_BUFFER_MS)
    critical_path_wakeup_offset_ms = (bet_submit_deadline_offset_ms
                                      + OKX_KLINE_FETCH_RTT_P95_MS
                                      + SIGNAL_COMPUTE_TIME_MS
                                      + POOL_READ_TIME_MS)
    bankroll_wakeup_offset_ms      = (critical_path_wakeup_offset_ms
                                      + BANKROLL_WAKE_OFFSET_PRE_CRITICAL_MS)
    ntp_sync_wakeup_offset_ms      = (bankroll_wakeup_offset_ms
                                      + NTP_WAKE_OFFSET_PRE_BANKROLL_MS)

Inside the critical-path wake the engine sequences pool snapshot ->
kline fetch -> signal compute -> bet submit. The 5ms POOL_READ_TIME_MS
is a cushion for the in-memory pool aggregate read; it does NOT need
its own wake (the prior architecture used a separate pool_read_wake
5ms ahead of kline_fetch_wake, which conceptually overstated as
"two scheduled events" what is really sequential operation time).
A single ``critical_path_wakeup_offset_ms`` keeps the wake schedule
honest about what's a scheduled event vs what's intra-wake sequencing.

The bankroll- and ntp-sync wake offsets above the critical path are
deliberately LITERAL 5-second gaps rather than derived from tightly
measured query budgets. Non-critical-path wakes are sized for
robustness against environmental drift (network spikes, Windows
update kicks, OKX RPC pauses) -- a 5s gap dwarfs every observed
worst case (~125ms NTP roundtrip, ~50-200ms wallet RPC). See
``BANKROLL_WAKE_OFFSET_PRE_CRITICAL_MS`` and
``NTP_WAKE_OFFSET_PRE_BANKROLL_MS`` for the rationale.
``NTP_QUERY_TIME_P99_MS`` is exposed for cross-validation
(N_servers x P99 << 5000) but NOT in the derivation.

Cross-validations enforced at config load. The CUTOFFS are fixed
inputs (set by strategy / data-horizon requirements); the OFFSETS
must fit within the cutoff windows.

    (critical_path_wakeup_offset_ms - POOL_READ_TIME_MS)
        <= (kline_cutoff_seconds * 1000 - OKX_KLINE_PUBLISH_DELAY_P95_MS)
        (the kline fetch fires inside the critical path AFTER the pool
        snapshot, i.e. at lock - (critical_path_wakeup_offset_ms -
        POOL_READ_TIME_MS); the cutoff candle has typically been
        published by then; rare publish-delay tail misses are
        absorbed by the streak counter)

    critical_path_wakeup_offset_ms <= (pool_cutoff_seconds * 1000
                                       - WSS_BET_EVENT_ARRIVAL_DELAY_P99_MS)
        (the pool snapshot fires at the START of the critical path,
        i.e. at lock - critical_path_wakeup_offset_ms; the cutoff bet
        event has typically arrived via WSS by then)
"""
from __future__ import annotations


# --- OKX REST timing -------------------------------------------------------

# OKX /history-candles publishing latency (= time between candle close and
# the candle being available via REST /history-candles for our 3-symbol
# parallel fetch).
#
# Source: research/p4c_canonical_loop_probe.py at varying wake offsets,
#         n=1000 + n=200 prior, 2026-05-02..2026-05-03.
# Method: probe @ wake=850ms (1150ms post-close) → 98.3% per-symbol
#         first-try; probe @ wake=1200ms (800ms post-close) → 96.9%.
#
# Two percentile values are exposed:
#   - P95: the budget the cutoff cross-validation uses. The kline-fetch
#     wake fires at lock - kline_fetch_wakeup_offset_ms; the cutoff
#     candle has had (cutoff_ms - kline_fetch_wakeup_offset_ms) time
#     to publish by then. Validation: that gap >= P95 publish delay
#     (~5% of fetches will hit a still-unpublished cutoff candle and
#     skip via the streak-counter path -- acceptable tail).
#   - P99: documented strict tail (extrapolated from probe). NOT used
#     in validation: at the canonical operating point (cutoff=2,
#     kline_fetch_wakeup_offset_ms=1090) the gap is 910ms < 1300ms,
#     so a strict-P99 validation would block a known-good config.
#     The strategy tolerates P99 misses; the streak counter is the
#     safety net.
# Last measured: 2026-05-03
OKX_KLINE_PUBLISH_DELAY_P95_MS: int = 700
OKX_KLINE_PUBLISH_DELAY_P99_MS: int = 1300

# OKX REST round-trip time for /history-candles fetches. Pooled p95 over
# the 4 symbols (BTC, ETH, SOL, BNB).
#
# Source: research/p4c_canonical_loop_probe.py n=1000, 2026-05-03.
# Pooled (n=4000 fetches): p50=258, p90=277, p95=289, p99=363, max=981.
# Last measured: 2026-05-03
OKX_KLINE_FETCH_RTT_P95_MS: int = 290


# --- Strategy compute ------------------------------------------------------

# Time the gate's signal-computation logic takes after kline data arrives.
# In practice this is sub-ms (numpy ops on 16-element arrays), but we
# allow a small headroom for variance.
#
# Source: code inspection of momentum_gate._compute_signal,
#         pancakebot/strategy/momentum_gate.py:415-... (numpy diff/sum
#         over 16-row arrays, no I/O).
# Last measured: 2026-05-03 (engineering judgment, not empirical probe)
SIGNAL_COMPUTE_TIME_MS: int = 50


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

# BSC RPC bet-TX submit RTT (eth_sendRawTransaction). EMPIRICALLY MEASURED
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
BSC_BET_SUBMIT_RTT_P95_MS: int = 200


# --- WSS subscriber timing ------------------------------------------------

# Time between a BSC block being mined (= block.timestamp) and the
# corresponding BetBull/BetBear event being received by our WSS
# subscriber. Per-event p99 over the 30-min sample window.
#
# Source: research/p4c_wss_arrival_probe.py n=219 events over 30min,
#         2026-05-03. p50=1041ms, p90=1723, p95=1777, p99=3106,
#         p99.9=3834.
# Last measured: 2026-05-03
WSS_BET_EVENT_ARRIVAL_DELAY_P99_MS: int = 3500  # p99=3106 + ~400ms buffer


# --- NTP clock sync -------------------------------------------------------

# Wall-clock duration of one ``ntplib.NTPClient.request(...)`` call against
# a public stratum-2 pool server (cloudflare/google/pool.ntp.org), end to
# end including DNS + UDP roundtrip. Per-call worst-server p99 across the
# rotation -- not the pooled p99, since the rotation visits each server
# in turn so the slowest server's tail dominates the wake budget.
#
# This constant is INFORMATIONAL / cross-validation only -- the engine's
# ntp_sync_wake budget is the literal NTP_WAKE_OFFSET_PRE_BANKROLL_MS
# (5000 ms), which dwarfs even N_servers x P99 worst case
# (3 x 102 = 306 ms per the 2026-05-06 probe). The cross-validation
# assertion in pancakebot/config.py confirms the 5000ms budget covers
# N_servers x p99 + a small dispatch buffer.
#
# Source: research/p4c_ntp_probe.py n=150, 2026-05-06.
#         servers rotated across cloudflare/google/pool.ntp.org with
#         200ms gap between calls.
#         per-server p99: cloudflare=54.7, google=69.7, pool.ntp.org=102.0
#         worst-server p99 = 102.0 ms.
# Last measured: 2026-05-06
NTP_QUERY_TIME_P99_MS: int = 125  # worst-server p99=102 + ~25ms buffer


# --- Non-critical-path wake gaps ------------------------------------------

# Gap between bankroll_wake and the critical_path entry. The bankroll
# wake fires at critical_path_wakeup_offset_ms + this offset
# (= ~lock-6.095s); the engine uses the budget to read live wallet
# balance via BSC RPC (~50-200ms p99) or, in dry mode, the in-memory
# simulated bankroll (sub-ms). 5s is deliberately generous: it covers
# any plausible RPC stall (even a slow fallback to a backup endpoint),
# and small RPC variance can't bleed into the critical path.
#
# Source: engineering judgment. Robustness > micro-optimization for
# non-critical-path wakes; if RPC p99 ever drifts to 4s the bot still
# bets on time. If it drifts to 6s the cross-validation gate in
# config.py fires and the operator notices before production breaks.
# Last measured: 2026-05-06
BANKROLL_WAKE_OFFSET_PRE_CRITICAL_MS: int = 5000

# Gap between ntp_sync_wake and bankroll_wake. The ntp_sync wake fires
# at bankroll_wakeup_offset_ms + this offset (= ~lock-11.095s); the
# engine uses the budget for one (or up to N_SERVERS rotated) NTP
# query. 5s is deliberately generous: 3 x P99 worst case = ~306ms;
# a 5000ms budget covers even a multi-second pool.ntp.org stall plus
# the rotation fall-through.
#
# Source: engineering judgment. Same robustness rationale as
# BANKROLL_WAKE_OFFSET_PRE_CRITICAL_MS.
# Last measured: 2026-05-06
NTP_WAKE_OFFSET_PRE_BANKROLL_MS: int = 5000


# --- Static buffers --------------------------------------------------------

# Time for the in-memory pool data read (engine.py reads
# PoolEventWatcher.get_pool, a dict + list filter under a lock).
# Sub-ms in practice; held as a small placeholder for the wake-schedule
# arithmetic and any future GIL-contention budget.
#
# Source: code inspection of pool_watcher.PoolEventWatcher.get_pool.
# Last measured: 2026-05-03 (engineering judgment)
POOL_READ_TIME_MS: int = 5

# Headroom on the bet_submit_deadline_offset_ms derivation. Beyond
# BSC_BET_SUBMIT_RTT_P95 + BSC_BLOCK_TIME, this absorbs second-order
# variance in bet-TX submission (signing time, sign-to-send dispatch)
# and any clock-jitter on the engine's timing-guard check.
#
# Source: engineering judgment.
# Last measured: 2026-05-03
BET_SUBMIT_SAFETY_BUFFER_MS: int = 50



# --- Module-load sanity checks --------------------------------------------

# Percentile order must hold: P95 <= P99 (probe noise that inverted them
# would silently break the tier-fallback validation in config.py).
assert OKX_KLINE_PUBLISH_DELAY_P95_MS <= OKX_KLINE_PUBLISH_DELAY_P99_MS, (
    f"OKX_KLINE_PUBLISH_DELAY_P95_MS ({OKX_KLINE_PUBLISH_DELAY_P95_MS}) "
    f"must be <= OKX_KLINE_PUBLISH_DELAY_P99_MS "
    f"({OKX_KLINE_PUBLISH_DELAY_P99_MS}); probe ordering violated"
)
