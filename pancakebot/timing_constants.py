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

    bet_submit_deadline_offset_before_lock_ms  = (BSC_QUANTUM_MS              # 50ms — quantum-shift buffer
                                      + BSC_BLOCK_TIME_MS         # 450ms — one full slot back-off
                                      + VALIDATOR_ASSEMBLY_WINDOW_MS  # 50ms — validator TX-list freeze
                                      + BSC_BET_SUBMIT_ONE_WAY_MS)  # 150ms — one-way RPC submit
    # = 700ms STATIC FALLBACK (Bundle 4 2026-05-14). Used when Lorentz ms-encoding
    # is unavailable (pre-Lorentz chain or detection failed). Live decision path
    # under Lorentz uses ``RpcPoller.compute_dynamic_submit_deadline_ms()`` for
    # per-round prediction that's typically 250-300ms tighter than this fallback.
    critical_path_wakeup_offset_before_lock_ms = (bet_submit_deadline_offset_before_lock_ms
                                      + OKX_KLINE_FETCH_RTT_P95_MS
                                      + MOMENTUM_GATE_COMPUTE_TIME_MS
                                      + POOL_READ_TIME_MS)
    final_rpc_poll_wakeup_offset_before_lock_ms = (
        pool_cutoff_seconds * 1000
        - BSC_BLOCK_TIME_MS
        - RPC_BLOCK_AVAILABILITY_DELAY_P99_MS
        - RPC_POLL_FINAL_TO_CRITICAL_PATH_SAFETY_MS)
    ramp_poll_2_wakeup_offset_before_lock_ms   = (final_rpc_poll_wakeup_offset_before_lock_ms
                                      + RPC_RAMP_2_TO_FINAL_INTERVAL_MS)
    ramp_poll_1_wakeup_offset_before_lock_ms   = (ramp_poll_2_wakeup_offset_before_lock_ms
                                      + RPC_RAMP_1_TO_RAMP_2_INTERVAL_MS)
    bankroll_wakeup_offset_before_lock_ms      = (critical_path_wakeup_offset_before_lock_ms
                                      + BANKROLL_WAKEUP_OFFSET_BEFORE_CRITICAL_PATH_MS)

Inside the critical-path wake the engine sequences pool snapshot ->
kline fetch -> signal compute -> bet submit. The 5ms POOL_READ_TIME_MS
is a cushion for the in-memory pool aggregate read; it does NOT need
its own wake (the prior architecture used a separate pool_read_wake
5ms ahead of kline_fetch_wake, which conceptually overstated as
"two scheduled events" what is really sequential operation time).
A single ``critical_path_wakeup_offset_before_lock_ms`` keeps the wake schedule
honest about what's a scheduled event vs what's intra-wake sequencing.

The bankroll wake offset above the critical path is deliberately a
LITERAL 5-second gap rather than derived from tightly measured query
budgets. Non-critical-path wakes are sized for robustness against
environmental drift (network spikes, Windows update kicks, OKX RPC
pauses) — a 5s gap dwarfs every observed worst case (~50-200ms wallet
RPC). See ``BANKROLL_WAKEUP_OFFSET_BEFORE_CRITICAL_PATH_MS`` for the rationale.

Bundle 5 v2 (2026-05-14): the prior ``ntp_sync_wakeup_offset_ms`` is
retired. The bot trusts the OS clock directly (Windows Time Service
kept tight via MaxPollInterval=5; see README "W32Time prerequisite").
``NTP_WAKE_OFFSET_PRE_BANKROLL_MS`` and ``NTP_QUERY_TIME_P99_MS``
constants are deleted along with this retirement.

Cross-validations enforced at config load. The CUTOFFS are fixed
inputs (set by strategy / data-horizon requirements); the OFFSETS
must fit within the cutoff windows.

    (critical_path_wakeup_offset_before_lock_ms - POOL_READ_TIME_MS)
        <= (kline_cutoff_seconds * 1000 - OKX_KLINE_PUBLISH_DELAY_P95_MS)
        (the kline fetch fires inside the critical path AFTER the pool
        snapshot, i.e. at lock - (critical_path_wakeup_offset_before_lock_ms -
        POOL_READ_TIME_MS); the cutoff candle has typically been
        published by then; rare publish-delay tail misses are
        absorbed by the streak counter)

    (final_rpc_poll_wakeup_offset_before_lock_ms
        - rpc_rtt_p99_for_batch(EXPECTED_FINAL_POLL_BATCH_SIZE)
        - RPC_POLL_DEADLINE_SAFETY_BUFFER_MS
        >= critical_path_wakeup_offset_before_lock_ms)
        (the final RPC poll must fire AND complete (at empirical p99
        RTT) before the critical-path wake reads the pool snapshot;
        pool_cutoff_seconds too small OR an upward drift in
        rpc_rtt_p99_for_batch raises ``final_rpc_poll_rtt_budget_insufficient``
        at config load — refactored 2026-05-12 to be strictly stronger
        than the prior ``final > critical_path + safety`` check.)

    RPC_RAMP_1_TO_RAMP_2_INTERVAL_MS >= (
        rpc_rtt_p99_for_batch(EXPECTED_RAMP_POLL_1_BATCH_SIZE)
        + RPC_POLL_DEADLINE_SAFETY_BUFFER_MS)
    RPC_RAMP_2_TO_FINAL_INTERVAL_MS >= (
        rpc_rtt_p99_for_batch(EXPECTED_RAMP_POLL_2_BATCH_SIZE)
        + RPC_POLL_DEADLINE_SAFETY_BUFFER_MS)
        (each per-leg interval must accommodate its corresponding ramp
        poll's actual rtt_p99 + safety; upward drift in
        rpc_rtt_p99_for_batch raises ``ramp_poll_1_to_ramp_2_interval_insufficient``
        or ``ramp_poll_2_to_final_interval_insufficient`` at config load.)
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

# Same fetch, p99 measurement. Used by the Bundle 5 (2026-05-14) dynamic
# critical_path_wakeup_offset_before_lock_ms: ``RpcPoller.compute_dynamic_critical_path_wake_ts()``
# walks back from the predicted predecessor block by
#   (OKX_KLINE_FETCH_RTT_P99_MS + MOMENTUM_GATE_COMPUTE_TIME_MS
#    + POOL_READ_TIME_MS + BSC_BET_SUBMIT_ONE_WAY_MS)
# = 352 + 50 + 5 + 150 = 557ms.
#
# Why 352 (not the 363 from the pooled canonical probe): the production
# decision-path effective p99 is "max-of-3" across the 3 OKX symbols
# fetched in parallel (BTC, ETH, SOL — BNB is fetched but not on the
# strategy-critical path). The Bundle 4 timing harness measured the
# round-trip of the slowest of three parallel fetches at 352ms p99 over
# n=1000 rounds, 2026-05-14. The pooled-per-symbol p99 (363ms) is a
# strictly weaker statistic for the deadline math (a single symbol could
# be in its tail while the max-of-3 is still on the central body).
#
# Source: research/bundle4_timing_harness.py n=1000, 2026-05-14
#         (max-of-3 parallel symbol fetch RTT).
# Last measured: 2026-05-14
OKX_KLINE_FETCH_RTT_P99_MS: int = 352


# --- Strategy compute ------------------------------------------------------

# Time the gate's signal-computation logic takes after kline data arrives.
# In practice this is sub-ms (numpy ops on 16-element arrays), but we
# allow a small headroom for variance.
#
# Source: code inspection of momentum_gate._compute_signal,
#         pancakebot/strategy/momentum_gate.py:415-... (numpy diff/sum
#         over 16-row arrays, no I/O).
# Last measured: 2026-05-03 (engineering judgment, not empirical probe)
MOMENTUM_GATE_COMPUTE_TIME_MS: int = 50


# --- Bundle 5 v2 anchor poll (2026-05-14) ---------------------------------

# Single anchor poll fires at lock - ANCHOR_POLL_OFFSET_BEFORE_LOCK_MS. The response
# is awaited for at most ANCHOR_POLL_TIMEOUT_MS. If the response arrives
# in time, the engine extracts BEP-520 mixHash to get a fresh ms-precise
# chain anchor, then computes a dynamic critical-path wake (typically
# closer to lock than the static fallback). If the poll times out, the
# engine falls back to the static critical-path wake +
# bet_submit_deadline.
#
# Design rationale:
# - Single poll, not continuous (Bundle 4 ran a 200ms-interval fine-phase
#   poller across the last ~3.5s before critical_path; Bundle 5 v2 drops
#   that for one well-timed poll, saving ~15 RPC calls per round).
# - Fire at lock-1300ms = static_wake_offset (1045) + ANCHOR_POLL_TIMEOUT_MS
#   (200) + small slack (55ms). Worst-case completion at lock-1100ms,
#   leaving 5ms slack before static_wake would otherwise fire at lock-1045.
# - Anchor lifetime is one round: no persistent state on RpcPoller. The
#   engine stores the AnchorState in a local variable and passes it to
#   the wake + deadline math.
ANCHOR_POLL_OFFSET_BEFORE_LOCK_MS: int = 1300
ANCHOR_POLL_TIMEOUT_MS: int = 200


# --- BSC chain timing ------------------------------------------------------

# Exact post-Lorentz BSC block production interval. Verified 2026-05-13
# (Bundle 4 reconnaissance): 4951/4951 consecutive-block transitions in a
# 5000-block sample had delta=450ms exactly (machine-precision regularity).
# 47/4999 transitions showed delta=500ms (50ms quantum-shift, see
# BSC_QUANTUM_MS); 1/4999 was a 2000ms multi-slot miss. Zero transitions
# < 450ms observed — i.e., misses can only DELAY the chain, never advance
# it. This load-bearing property lets Bundle 4 ship with no safety margin
# on the predict-by-block arithmetic.
#
# Source: ad-hoc Bundle 4 reconnaissance via eth_getBlockByNumber batches
#         over blocks 98133347..98138346 (~37.5 min of chain), 2026-05-13.
#         Mean=450.78ms, median=450, min=450, max=2000.
# Prior value: 500ms (conservative rounding before Lorentz ms-encoding was
#         empirically verified). Replaced 2026-05-14 (Bundle 4).
# Last measured: 2026-05-13
BSC_BLOCK_TIME_MS: int = 450

# BEP-520 millisecond-encoding quantum. Post-Lorentz BSC encodes block
# milliseconds in mixHash[-2:] (uint16 big-endian) at 50ms granularity.
# All observed ms values in the Bundle 4 sample fell on the {0, 50, 100,
# ..., 950} grid with uniform distribution. Validator clocks deemed
# accurate to 50ms; quantum-shift events (delta=500ms = block_time + 1
# quantum) are 0.94% per block.
#
# Used by Bundle 4 dynamic deadline math: a predicted block boundary
# within one quantum of lock_ms must back off one full block to absorb
# the possibility of a quantum-shift pushing the predecessor across.
#
# Source: BEP-520 spec + empirical Bundle 4 reconnaissance 2026-05-13.
# Last measured: 2026-05-13
BSC_QUANTUM_MS: int = 50

# Validator TX-list freeze window: time before block publication when
# the in-turn validator stops accepting new TXs into the candidate
# block. A bet TX that arrives at the validator's mempool LESS than
# this window before the validator publishes will miss inclusion in
# that block (and slip to the next slot, which is the lock block —
# definite revert).
#
# Source: BSC Parlia consensus literature + community probes. Conservative
#         50ms estimate; precise value is implementation-defined per
#         validator and not directly measurable from RPC.
# Last measured: 2026-05-13 (engineering judgment; not empirically probed)
VALIDATOR_ASSEMBLY_WINDOW_MS: int = 50

# One-way TCP submit time from this host to a BSC validator/RPC mempool.
# Used in the dynamic deadline math: the bet TX must REACH the validator
# mempool by ``predecessor_block.milli_ts - VALIDATOR_ASSEMBLY_WINDOW_MS``,
# so the LOCAL deadline for ``eth_sendRawTransaction(...)`` is that minus
# this one-way budget.
#
# Empirically validated 2026-05-16 via probe:
# ``research/probe_send_raw_transaction_rtt.py --mode full`` ran 100
# self-transfer TXs from the production wallet via the primary write-
# path RPC ``bsc-dataseed1.defibit.io``. 100/100 sent, 100/100 included
# within 30s. Round-trip RTT: p50=37ms, p95=188ms, p99=277ms, max=277ms.
# One-way ≈ RTT/2 + propagation; the 50ms-quantum cover for p99/2 (139ms)
# is 150ms with 11ms safety margin. Lowering to 100ms would cover p95/2
# but leave ~1-2% of TXs arriving slightly past the dynamic submit
# deadline (clean BET TIMING ABORT, but lost bet opportunity); the
# critical-path budget gain (50ms = ~5% of static-wake budget) is not
# worth the asymmetric cost.
#
# Source: research/probe_send_raw_transaction_rtt.py n=100, 2026-05-16.
#         Full distribution + reasoning in
#         var/strategy_review/send_raw_tx_probe.md.
# Last measured: 2026-05-16
BSC_BET_SUBMIT_ONE_WAY_MS: int = 150

# --- RPC poll timing (Era 11: 2026-05-07 pivot) ---------------------------
#
# Replaces the WSS-subscription pool watcher with deterministic
# batched-RPC polling. WSS arrival timing is no longer relevant on the
# decision path; the pool aggregate is built from periodic + ramp +
# final polls of `eth_getBlockReceipts(blockHash)` over HTTP. See
# var/design/rpc_polling_architecture_2026_05_07.md for the full
# architecture and var/incident_reports/2026_05_07_rpc_polling_spike_results.md
# for the empirical provenance.

# Per-batch p99 round-trip time, indexed by batch size. The wake
# offsets look up this table by EXPECTED_*_BATCH_SIZE constants below.
#
# Sizes 2-15: research/probe_rpc_polling.py n=50 per size, 2026-05-07,
# publicnode single-endpoint. Wake-offset derivation uses sizes 10
# (EXPECTED_FINAL_POLL_BATCH_SIZE) and 15 (EXPECTED_RAMP_POLL_{1,2}_BATCH_SIZE);
# these are preserved at the original publicnode baseline to keep the
# canonical wake-offset schedule (pinned by
# test_canonical_pool_cutoff_6_produces_expected_offsets) stable across
# the 2026-05-11 transport switch. Re-measuring sizes 10/15 under the
# new fire-to-all transport would force a wake-schedule shift the
# wider system isn't asking for. drpc.org rejected batched JSON-RPC
# arrays with HTTP 500 at every tested size. size=15's raw measurement
# was 2285ms but came from 50-sample p99-as-max noise; the monotonic
# interpolation 1213ms (between size=10 at 910 and the OLD size=20
# publicnode baseline of 1533ms) is the correct provisioning value.
#
# Size=20: research/probe_fire_to_all_p99_batch20_clean_2026_05_11.py
# n=30, fire-to-all-pool (6 endpoints), urllib3 PoolManager, 30s
# inter-call spacing, BOT STOPPED. 30/30 successes. The bot-stopped
# measurement matters because the 2026-05-11 transport switch
# (urllib3 PoolManager + fire-to-all) means production now hedges
# across the configured endpoints; running the probe alongside the
# bot inflated RTTs ~3.5x due to same-IP / Windows-TCP / urllib3-pool
# contention between the two processes. The 1319ms is the bot's
# actual operating value (no concurrent caller in production).
#
# Bundle 6 caveat (2026-05-15): the 1319ms value was measured under
# the prior 6-endpoint hedged pool (min-of-6 fastest-response wins).
# Bundle 6 trimmed the pool to 3 endpoints (one per fault-domain
# family — see READ_PATH_HEDGED_ENDPOINTS in chain/rpc_poller.py).
# Under min-of-3, the operating p99 could rise modestly vs min-of-6,
# but the per-endpoint probe (research/probe_per_endpoint_isolated_2026_05_15.py)
# showed bsc-dataseed1.binance.org dominates: its single-endpoint
# batch p95 is 2717ms, well below the 5s timeout, and it wins
# production hedged races by a wide margin (23/40 in I3). The
# downstream consumer (``_estimated_catchup_ms`` feasibility check)
# absorbs modest under-measurement via the per-batch deadline check
# and per-batch try/except in ``RpcPoller._poll_now``. If post-trim
# cold-start observations show the estimate is too tight, re-measure
# under the 3-endpoint pool.
#
# CAVEAT: n=30 is statistically thin for a P99 estimate (~2nd-highest
# sample out of 30). The true population P99 could realistically sit
# anywhere from p93 to p100 of this sample; rule of thumb for a stable
# P99 is n>=100. The value is used by ``_estimated_catchup_ms`` for
# the catch-up feasibility check; runtime graceful degradation absorbs
# undermeasurement via the per-batch deadline check + per-batch
# try/except in ``RpcPoller._poll_now`` (which downgrades partial
# failures to ``pool_not_ready`` rather than crashing). If a tighter
# estimate is needed (e.g. INFEAS rate diagnostics still suggest the
# value is wrong), re-measure with n>=100.
RPC_BATCH_RECEIPTS_RTT_P99_MS_BY_SIZE: dict[int, int] = {
    2: 421,
    5: 771,
    10: 910,
    15: 1213,   # interpolated against OLD size=20 publicnode baseline 1533
    20: 1319,   # fire-to-all p99 n=30, 2026-05-11 (was 1533 single-publicnode)
}

# Block availability delay — newhead arrival to first successful
# eth_getBlockReceipts(block_hash). drpc.org p99 596ms, publicnode
# p99 436ms (n=133 each, 2026-05-07). Lock 600ms = drpc.org worst-case
# + small buffer to absorb either endpoint.
#
# Source: research/probe_rpc_polling.py n=133 per endpoint, 2026-05-07.
RPC_BLOCK_AVAILABILITY_DELAY_P99_MS: int = 600

# Hard cap on batch size. publicnode tested up to 100 (p50 3092ms;
# unusable for deadline path) and 200 (response-too-large rejection).
# 20 is the operating cap for both deadline-driven polls and cold-start
# (a larger batch is fine for cold-start latency-wise but giving it the
# same cap simplifies the implementation).
RPC_BATCH_MAX_BLOCKS: int = 20

# Per-request HTTP timeout for batched JSON-RPC (eth_getBlockReceipts +
# eth_getBlockByNumber bundles, batch_size <= RPC_BATCH_MAX_BLOCKS).
# 5s detects unreachable-endpoint scenarios fast while staying well above
# the empirical p99 single-batch RTT (RPC_BATCH_RECEIPTS_RTT_P99_MS_BY_SIZE:
# batch=20 -> 1319ms fire-to-all, 2026-05-11). Was 30s; reduced 2026-05-08
# after a publicnode outage where 30s hangs grew the catch-up backlog by
# ~60 blocks per failed poll.
RPC_HTTP_BATCH_TIMEOUT_SECONDS: int = 5

# Per-request HTTP timeout for single (non-batched) RPC calls
# (eth_blockNumber, eth_getBlockByNumber). Smaller request, same 5s
# timeout for consistency.
RPC_HTTP_SINGLE_TIMEOUT_SECONDS: int = 5

# Periodic poll cadence during the round. Was 30s (giving ~10 polls/round)
# until 2026-05-12. Lowered to 8s after INFEAS rate analysis: at 30s
# cadence, ~67 BSC blocks accumulate between polls (BSC block ~0.45s,
# rounded to BSC_BLOCK_TIME_MS=500). 67 blocks forces multi-batch
# catch-up that often exceeds available time before lock. 8s cadence
# = ~17.8 blocks per poll = single batch at batch_size=20 with margin.
# Any poll failure costs less cursor advance; periodic poll's job is
# keeping cursor close to head.
RPC_PERIODIC_POLL_INTERVAL_SECONDS: int = 8

# Final-poll wake derivation safety cushion (cross-RPC variance the
# spike didn't capture, etc.). Engineering judgment.
RPC_POLL_FINAL_TO_CRITICAL_PATH_SAFETY_MS: int = 200

# Per-poll deadline cushion — if a poll's RTT exceeds (next_wake_offset
# - this), the poll is marked stale (logged, _last_poll_too_slow=True
# for diagnostics). The critical-path readiness gate no longer skips
# on this alone; the round-aware feasibility check
# (pool_not_ready_catchup_infeasible_for_round) is the canonical
# integrating signal. Engineering judgment.
RPC_POLL_DEADLINE_SAFETY_BUFFER_MS: int = 200

# Per-leg fixed intervals between successive RPC polls. Replaces the
# prior uniform ``RPC_RAMP_POLL_INTERVAL_MS=1500`` (2026-05-12) which
# coupled both ramp gaps to the same constant and was sized for a
# stale 30s-periodic-cadence assumption (batch=15 for both ramps).
# With 8s periodic cadence (post 740328f), the actual expected ramp
# workloads diverge sharply:
#
#   ramp_1: catches up since the last periodic poll. Worst case = one
#           periodic interval (8s) at BSC 0.45s blocks ≈ 17.8 blocks
#           → batch=20 (clamped at RPC_BATCH_MAX_BLOCKS).
#           Needs rtt_p99(20)=1319ms + safety + margin → 1700ms.
#   ramp_2: catches up since ramp_1 cursor advance. Wall gap is
#           bounded by RPC_RAMP_1_TO_RAMP_2_INTERVAL_MS plus bankroll
#           work; ≈ 1700-2000ms / 450ms/block ≈ 4 blocks → batch=5.
#           Needs rtt_p99(5)=771ms + safety + margin → 1100ms.
#
# Startup invariants in pancakebot/config.py validate each interval
# covers its corresponding rtt_p99 + safety; future drift in
# RPC_BATCH_RECEIPTS_RTT_P99_MS_BY_SIZE that violates either invariant
# fails-fast at config-load time rather than silently producing a
# too-tight schedule.
RPC_RAMP_1_TO_RAMP_2_INTERVAL_MS: int = 1700
RPC_RAMP_2_TO_FINAL_INTERVAL_MS: int = 1100

# Expected batch sizes used to derive the deadline-driven wake offsets'
# rtt_p99 lookups (in the startup invariants). Runtime batches are
# dynamic (= blocks since last poll); these are sized for the worst-case
# blocks-behind each poll could plausibly see at the canonical 8s
# periodic cadence (BSC 0.45s blocks):
#
#   ramp_1: catches up since last periodic poll. Worst case = full
#           periodic interval (8s) → ~17.8 blocks → batch=20 (clamped at
#           RPC_BATCH_MAX_BLOCKS).
#   ramp_2: catches up since ramp_1 cursor advance (1.7-2s wall gap) →
#           ~4 blocks → batch=5.
#   final:  catches up since ramp_2 cursor advance (1.1s wall gap) →
#           ~3 blocks → batch=5.
#
# Refactored 2026-05-12 from the prior 30s-periodic-era values (10/15/15).
EXPECTED_FINAL_POLL_BATCH_SIZE: int = 5
EXPECTED_RAMP_POLL_2_BATCH_SIZE: int = 5
EXPECTED_RAMP_POLL_1_BATCH_SIZE: int = 20


def rpc_rtt_p99_for_batch(batch_size: int) -> int:
    """Return P99 RTT for a batch of ``batch_size``, linearly interpolated
    between adjacent measured keys in RPC_BATCH_RECEIPTS_RTT_P99_MS_BY_SIZE.

    Behavior spec (refactored 2026-05-12 from a ceiling-only lookup):
      - ``batch_size <= 0``          -> ``0``
      - ``batch_size <= smallest``   -> ``table[smallest]`` (ceiling at small end)
      - ``batch_size`` is a key      -> ``table[batch_size]`` (exact passthrough)
      - between keys k_lo < n < k_hi -> linear interpolation:
            round(table[k_lo] + (table[k_hi] - table[k_lo])
                  * (n - k_lo) / (k_hi - k_lo))
      - ``batch_size > largest``     -> ``table[largest]`` (ceiling at large end)

    All current callers (config.py invariants + rpc_poller._estimated_catchup_ms
    with ``_batch_size=20``) pass exact measured keys, so the change is
    pure-passthrough at canonical config. Interpolation only matters when
    ``_estimated_catchup_ms`` calls with a per-batch remainder (e.g. 7, 12, 18)
    after the batch-size-aware refactor; there it tightens the estimate vs
    the prior ceiling lookup, reducing false-INFEAS at small backlogs.

    For sizes above the largest measured key, the ceiling fallback is
    informational; if it fires, the calling code is provisioning a batch
    above the measured range and should be re-checked.
    """
    if batch_size <= 0:
        return 0
    table = RPC_BATCH_RECEIPTS_RTT_P99_MS_BY_SIZE
    keys = sorted(table.keys())
    # Ceiling at small end: batch_size <= smallest measured key.
    if batch_size <= keys[0]:
        return table[keys[0]]
    # Ceiling at large end: above the largest measured key.
    if batch_size >= keys[-1]:
        return table[keys[-1]]
    # Exact-key passthrough (covers interior measured points).
    if batch_size in table:
        return table[batch_size]
    # Linear interpolation between the bracketing adjacent keys.
    for k_lo, k_hi in zip(keys, keys[1:]):
        if k_lo < batch_size < k_hi:
            rtt_lo = table[k_lo]
            rtt_hi = table[k_hi]
            return int(round(
                rtt_lo + (rtt_hi - rtt_lo) * (batch_size - k_lo) / (k_hi - k_lo)
            ))
    # Defensive: unreachable given the bounds checks above.
    return table[keys[-1]]


# --- Non-critical-path wake gaps ------------------------------------------

# Gap between bankroll_wake and the critical_path entry. The bankroll
# wake fires at critical_path_wakeup_offset_before_lock_ms + this offset
# (= ~lock-6.045s); the engine uses the budget to read live wallet
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
BANKROLL_WAKEUP_OFFSET_BEFORE_CRITICAL_PATH_MS: int = 5000

# Bundle 5 v2 (2026-05-14): ``NTP_QUERY_TIME_P99_MS`` and
# ``NTP_WAKE_OFFSET_PRE_BANKROLL_MS`` are retired alongside the
# application-level NTP layer. The bot trusts the OS clock (W32Time
# tightening per README).


# --- Static buffers --------------------------------------------------------

# Time for the in-memory pool data read (engine.py reads
# PoolEventWatcher.get_pool, a dict + list filter under a lock).
# Sub-ms in practice; held as a small placeholder for the wake-schedule
# arithmetic and any future GIL-contention budget.
#
# Source: code inspection of pool_watcher.PoolEventWatcher.get_pool.
# Last measured: 2026-05-03 (engineering judgment)
POOL_READ_TIME_MS: int = 5

# --- Module-load sanity checks --------------------------------------------

# Percentile order must hold: P95 <= P99 (probe noise that inverted them
# would silently break the tier-fallback validation in config.py).
assert OKX_KLINE_PUBLISH_DELAY_P95_MS <= OKX_KLINE_PUBLISH_DELAY_P99_MS, (
    f"OKX_KLINE_PUBLISH_DELAY_P95_MS ({OKX_KLINE_PUBLISH_DELAY_P95_MS}) "
    f"must be <= OKX_KLINE_PUBLISH_DELAY_P99_MS "
    f"({OKX_KLINE_PUBLISH_DELAY_P99_MS}); probe ordering violated"
)
assert OKX_KLINE_FETCH_RTT_P95_MS <= OKX_KLINE_FETCH_RTT_P99_MS, (
    f"OKX_KLINE_FETCH_RTT_P95_MS ({OKX_KLINE_FETCH_RTT_P95_MS}) "
    f"must be <= OKX_KLINE_FETCH_RTT_P99_MS "
    f"({OKX_KLINE_FETCH_RTT_P99_MS}); probe ordering violated"
)
