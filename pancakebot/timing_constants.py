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
                                      + BSC_BET_SUBMIT_ONE_WAY_MS)  # 75ms — one-way RPC submit
    # = 625ms STATIC FALLBACK (Bundle 4 2026-05-14). Used when Lorentz ms-encoding
    # is unavailable (pre-Lorentz chain or detection failed). Live decision path
    # under Lorentz uses ``RpcPoller.compute_dynamic_submit_deadline_ms()`` for
    # per-round prediction that's typically 250-300ms tighter than this fallback.
    critical_path_wakeup_offset_before_lock_ms = (bet_submit_deadline_offset_before_lock_ms
                                      + OKX_KLINE_FETCH_RTT_P95_MS
                                      + SIGNAL_COMPUTE_TIME_MS
                                      + POOL_READ_TIME_MS)
    single_poll_wakeup_offset_before_lock_ms   = (pool_cutoff_seconds * 1000
                                      - BSC_BLOCK_TIME_MS
                                      - RPC_BLOCK_AVAILABILITY_DELAY_P99_MS
                                      - RPC_POLL_FINAL_TO_CRITICAL_PATH_SAFETY_MS)
    preflight_wakeup_offset_before_lock_ms     = (critical_path_wakeup_offset_before_lock_ms
                                      + PREFLIGHT_WAKEUP_OFFSET_BEFORE_CRITICAL_PATH_MS)

Inside the critical-path wake the engine sequences pool snapshot ->
kline fetch -> signal compute -> bet submit. The 5ms POOL_READ_TIME_MS
is a cushion for the in-memory pool aggregate read; it does NOT need
its own wake (the prior architecture used a separate pool_read_wake
5ms ahead of kline_fetch_wake, which conceptually overstated as
"two scheduled events" what is really sequential operation time).
A single ``critical_path_wakeup_offset_before_lock_ms`` keeps the wake schedule
honest about what's a scheduled event vs what's intra-wake sequencing.

The preflight wake offset above the critical path is deliberately a
LITERAL 5-second gap rather than derived from tightly measured query
budgets. Non-critical-path wakes are sized for robustness against
environmental drift (network spikes, Windows update kicks, OKX RPC
pauses) — a 5s gap dwarfs every observed worst case (~50-200ms wallet
RPC). See ``PREFLIGHT_WAKEUP_OFFSET_BEFORE_CRITICAL_PATH_MS`` for the rationale.

Bundle 5 v2 (2026-05-14): the prior ``ntp_sync_wakeup_offset_ms`` is
retired. The bot trusts the OS clock directly (Windows Time Service
kept tight via MaxPollInterval=5; see README "W32Time prerequisite").
``NTP_WAKE_OFFSET_PRE_BANKROLL_MS`` and ``NTP_QUERY_TIME_P99_MS``
constants are deleted along with this retirement.

Cross-validations enforced at config load. The CUTOFFS are fixed
inputs (set by strategy / data-horizon requirements); the OFFSETS
must fit within the cutoff windows.

    (single_poll_wakeup_offset_before_lock_ms
        - rpc_rtt_p99_for_batch(EXPECTED_SINGLE_POLL_BATCH_SIZE)
        - RPC_POLL_DEADLINE_SAFETY_BUFFER_MS
        >= critical_path_wakeup_offset_before_lock_ms)
        (Candidate C, 2026-06-06: the single RPC poll must fire AND
        complete (at empirical p99 RTT for the worst-case full-interval
        batch) before the critical-path wake reads the pool snapshot;
        pool_cutoff_seconds too small OR an upward drift in
        rpc_rtt_p99_for_batch raises ``single_poll_rtt_budget_insufficient``
        at config load. The prior 3-leg ramp ladder + its per-leg interval
        invariants are retired — the retained 8s periodic poll bounds the
        single poll's catch-up batch.)
"""
from __future__ import annotations


# --- OKX REST timing -------------------------------------------------------

# OKX /history-candles publishing latency reference (informational only,
# not gated by the runtime). The bot always fetches at the latest moment
# the wake schedule allows, so the per-round publish-tail risk depends on
# how late dynamic mode fires -- not on a configured "budget." The only
# runtime tolerance is the streak counter
# (``max_consecutive_kline_fetch_failures``), which absorbs up to N
# consecutive ``got_<N-1>_expected_N`` outcomes before crashing the bot
# (-> supervisor restart + Discord alert).
#
# Empirically measured distribution (kept here so future operators don't
# re-derive what we've already paid the cost to measure):
#   P95 ≈ 700ms, P99 ≈ 1300ms
# Source: research/p4c_canonical_loop_probe.py at varying wake offsets,
#         n=1000 + n=200 prior, 2026-05-02..2026-05-03.
# Method: probe @ wake=850ms (1150ms post-close) → 98.3% per-symbol
#         first-try; probe @ wake=1200ms (800ms post-close) → 96.9%.
# Last measured: 2026-05-03.
#
# Removed 2026-05-17: the prior ``OKX_KLINE_PUBLISH_DELAY_P95_MS`` and
# ``OKX_KLINE_PUBLISH_DELAY_P99_MS`` constants + their config-load tier
# ladder + the ``kline_publish_tier`` label they emitted. The tier check
# was a one-shot config-load gate, never re-consulted at runtime; with
# the Bundle 5 v2 dynamic anchor-driven wake, the actual fetch fires at
# whatever offset the anchor dictates anyway, making the static-wake
# tier label misleading.

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
#   (OKX_KLINE_FETCH_RTT_P99_MS + SIGNAL_COMPUTE_TIME_MS
#    + POOL_READ_TIME_MS + BSC_BET_SUBMIT_ONE_WAY_MS)
# = 352 + 50 + 5 + 75 = 482ms.
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
SIGNAL_COMPUTE_TIME_MS: int = 50


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
# Re-measured 2026-05-20 via 4 independent 100-TX probes (n=400 total)
# across two UTC hours (~04:50 and ~19:18) and two spacings (1s, 10s):
# p95 RTT clustered at 46-53ms across all four runs (vs. 188ms on
# 2026-05-16 — ~4× network/RPC speedup). p99 RTT 65-127ms with the
# single 127ms sample being a one-off outlier; modal p99 ~80ms.
# One-way ≈ RTT/2 + propagation. p99/2 across runs: 33-63ms. The 75ms
# value covers p99/2 with ≥12ms margin in all 4 runs and absorbs the
# Phase-2 outlier's RTT/2 (63ms) with 12ms margin. Choosing 75ms over
# the script's auto-recommendation of 50ms preserves a half-quantum
# tail-outlier buffer.
#
# Source: research/probe_send_raw_tx_rtt_2026_05_20*.py (n=400 across
#         4 runs). Full distributions in
#         var/strategy_review/2026_05_20_send_raw_tx_probe_100_at_{1s,10s}{,_hour2}.{jsonl,md}.
# Last measured: 2026-05-20
BSC_BET_SUBMIT_ONE_WAY_MS: int = 75

# --- RPC poll timing (Era 11: 2026-05-07 pivot) ---------------------------
#
# Replaces the WSS-subscription pool watcher with deterministic
# batched-RPC polling. WSS arrival timing is no longer relevant on the
# decision path; the pool aggregate is built from periodic + single
# polls of `eth_getBlockReceipts(blockHash)` over HTTP. See
# var/design/rpc_polling_architecture_2026_05_07.md for the full
# architecture and var/incident_reports/2026_05_07_rpc_polling_spike_results.md
# for the empirical provenance.

# Per-batch p99 round-trip time, indexed by batch size. The wake
# offsets look up this table by EXPECTED_*_BATCH_SIZE constants below.
#
# Sizes 2-15: research/probe_rpc_polling.py n=50 per size, 2026-05-07,
# publicnode single-endpoint. The single-poll startup invariant looks up
# size 20 (EXPECTED_SINGLE_POLL_BATCH_SIZE); the table is preserved at the
# original publicnode baseline to keep the canonical wake-offset schedule
# (pinned by test_canonical_pool_cutoff_6_produces_expected_offsets) stable
# across
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

# How long either TX path (bet OR claim) waits for its receipt before giving
# up. Decoupled from buffer_seconds + padding (the refund-eligibility math) —
# a TX receipt has no reason to be tied to that. After ~10s (~22 BSC blocks at
# 0.45s) without a receipt, the TX is realistically dropped from the mempool;
# waiting longer just delays the signal. Bet and claim share this single value.
TX_RECEIPT_WAIT_TIMEOUT_SECONDS: int = 10

# Periodic poll cadence during the round. Was 30s (giving ~10 polls/round)
# until 2026-05-12. Lowered to 8s after INFEAS rate analysis: at 30s
# cadence, ~67 BSC blocks accumulate between polls (BSC block ~0.45s,
# rounded to BSC_BLOCK_TIME_MS=500). 67 blocks forces multi-batch
# catch-up that often exceeds available time before lock. 8s cadence
# = ~17.8 blocks per poll = single batch at batch_size=20 with margin.
# Any poll failure costs less cursor advance; periodic poll's job is
# keeping cursor close to head.
RPC_PERIODIC_POLL_INTERVAL_SECONDS: int = 8

# Defense-in-depth pad on top of RPC_HTTP_BATCH_TIMEOUT_SECONDS when
# deciding the latest safe time a periodic poll may fire before the
# single-poll window opens.
#
# What it protects: the race where a periodic poll's ``_poll_lock``
# release barely overlaps with the engine-driven single poll's wake and
# non-blocking acquire. If a periodic poll anchored just before
# single_poll_window_start hits its full HTTP timeout (5s), it completes
# IN-FLIGHT against the single poll's wake at lock_at − single_poll_offset.
# Even a few ms of overlap causes the single poll's
# ``_poll_lock.acquire(blocking=False)`` to fail and silently drop it.
#
# What it accounts for:
#   - OS scheduler jitter on thread wake (~1 ms typical, up to a few
#     ms worst case under load)
#   - urllib3 HTTP read-after-timeout cleanup time (~few ms to surface
#     the timeout exception and release the connection back to the pool)
#   - Python GIL release latency between request completion in the
#     network thread and lock release in the poller thread (microseconds
#     typically, but adversarial GC pauses can extend it)
#
# Why 50 ms and not less: cushion against pathological cases — loaded
# system, GC pause, antivirus scan interrupt — while staying small
# enough to be operationally invisible (this pad effectively shortens
# the periodic cadence by ~50 ms once per round, which is negligible
# against the 8 s base cadence).
#
# Why not bigger: the suspend-on-overrun branch in
# ``_compute_periodic_timeout`` is the actual correctness fix for this
# race. This buffer is purely defense-in-depth for the rare
# exact-timeout-with-bad-jitter case where a periodic completed within
# the buffer of the single poll's wake; a larger buffer would needlessly skip
# more periodic ticks per round without proportionate safety gain.
RPC_PERIODIC_TO_SINGLE_POLL_SAFETY_BUFFER_SECONDS: float = 0.05

# Final-poll wake derivation safety cushion (cross-RPC variance the
# spike didn't capture, etc.). Engineering judgment.
RPC_POLL_FINAL_TO_CRITICAL_PATH_SAFETY_MS: int = 200

# Per-poll deadline cushion — if a poll's RTT exceeds (next_wake_offset
# - this), the poll is marked slow (logged, _last_poll_too_slow=True
# for diagnostics). The critical-path readiness gate no longer skips
# on this alone; the round-aware feasibility check
# (pool_not_ready_catchup_infeasible_for_round) is the canonical
# integrating signal. Engineering judgment.
RPC_POLL_DEADLINE_SAFETY_BUFFER_MS: int = 200

# Expected batch size used to derive the single-poll wake offset's rtt_p99
# lookup (in the startup invariant). Candidate C (2026-06-06) replaced the
# 3-leg ramp ladder with ONE batched poll before the critical path; the
# retained 8s periodic poll keeps the cursor within one interval of head, so
# the single poll's worst case = one periodic interval (8s) at BSC 0.45s
# blocks ≈ 17.8 blocks → batch=20 (clamped at RPC_BATCH_MAX_BLOCKS). Runtime
# batches are dynamic (= blocks since last periodic poll).
EXPECTED_SINGLE_POLL_BATCH_SIZE: int = 20


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

# Gap between preflight_wake and the critical_path entry. The preflight
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
PREFLIGHT_WAKEUP_OFFSET_BEFORE_CRITICAL_PATH_MS: int = 5000

# OKX warmup wake (2026-05-21): pre-bet TLS-handshake warmup for the
# OkxClient's HTTPS connection pool. Fires BEFORE preflight_wake so any
# OKX-side keep-alive expiry (typical nginx default ~75s) from a long
# idle window (e.g. a catchup_infeasible streak) is paid OUT of the
# critical path. Without this, the first kline fetch after a long idle
# pays a 500-800ms TLS handshake during the bet-decision window —
# observed 2026-05-21 in live mode when 5 consecutive
# pool_not_ready_catchup_infeasible_for_round skips left OKX idle for
# ~25 minutes; on the recovery round, fetch_ms ran 734-850ms vs typical
# 270ms, contributing to a missed lock-block inclusion.
#
# Slot rationale: the first pre-lock wake (lock - 7000ms), ~1030ms before
# preflight (lock - 5970ms), leaving headroom for a slow warmup (~270ms
# typical, ~800ms worst-case cold). Doesn't touch the
# critical path; even a fully-failed warmup (rare; OkxClient.warmup
# swallows transient errors) just means the bet round pays the cold
# fetch like before.
#
# Source: engineering judgment + 2026-05-21 incident.
# Last measured: 2026-05-21
OKX_WARMUP_WAKEUP_OFFSET_BEFORE_LOCK_MS: int = 7000

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

assert OKX_KLINE_FETCH_RTT_P95_MS <= OKX_KLINE_FETCH_RTT_P99_MS, (
    f"OKX_KLINE_FETCH_RTT_P95_MS ({OKX_KLINE_FETCH_RTT_P95_MS}) "
    f"must be <= OKX_KLINE_FETCH_RTT_P99_MS "
    f"({OKX_KLINE_FETCH_RTT_P99_MS}); probe ordering violated"
)
