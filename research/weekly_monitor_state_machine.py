"""Weekly state-machine monitor — sync, evaluate, and (safely) toggle the bot.

Runs the wait-and-monitor protocol on a weekly cadence and drives the live
bot's enable/disable state through a deliberately ASYMMETRIC, fail-safe
state machine:

  * AUTO-DISABLE (protect capital) is fully autonomous — the negative
    trigger stops+disables the live unit with `--apply` alone.
  * AUTO-ENABLE (2026-07-09 user decision, re-affirmed 2026-07-17): the
    positive trigger acts under `--apply` alone — raw p<0.10 on the
    trailing-1-week window, single test, no multiple-comparison gate.
    When the bot is DISABLED and its persisted pause state shows an
    active suspension, enabling ALSO writes the cooldown override flag
    first, so the restarted bot releases on its very first paused round
    (one-shot re-enable; without the flag it would boot into the stale
    suspension and need a second consecutive positive Sunday). If the
    bot is enabled but breaker-suspended, the positive trigger writes
    the override flag alone.

Default is DRY-RUN: it computes + reports the decision and writes the
artifact but touches NOTHING. Pass `--apply` to let it act (systemd
enable/disable + override-flag writes).

Unattended-safety (2026-07-17/18 hardening):
  * evidence gate: positive actions require BOTH a clean sync exit AND
    fresh data (newest closed ROUND <= 36h old — a stalled indexer can
    exit 0 without advancing the stores). ROUND freshness is the yardstick
    here because the round stream is dense; the sparse FIRE stream lags
    days in normal droughts and would false-trip it. Blind runs (either
    check failing) block enable/release, freeze the weekly counters, and
    alert loudly; the protective disable may still act on last-synced
    data.
  * frozen-window gate (2026-08-21): the fire stream has its OWN, separate
    staleness bound (FIRE_STALE_MAX_AGE_S) because the evaluation windows
    are keyed to the newest FIRE — a bot that stops betting freezes them
    while --sync keeps the ROUND stream fresh, so the run above would
    report healthy stats for a week it never traded in. A frozen window
    blocks POSITIVE actions and is reported loudly, but never freezes
    counters, never arms retries, and is never itself a disable trigger.
  * daily retries (2026-07-18): a blind applied run writes an atomic
    retry_pending marker; cron fires DAILY and the wrapper runs Mon-Sat
    only while a marker exists. A recovered retry is keyed to the MISSED
    Sunday (Sundays are the last ISO day — calendar keying would steal
    the next Sunday's state advance), runs the full evaluation (triggers
    included), clears the marker, and reports "recovered after N failed
    attempts". Blind retries alert one line each. The next Sunday
    supersedes any unresolved marker.
  * blindness escalation: sync_fail_streak counts FULLY-blind ISO weeks
    (Sunday + every retry failed); at 3 — counting the currently-blind
    attempt — with the bot enabled or running, the monitor disables it.
    Never bet for months while the evaluator cannot see performance.
  * systemctl not answering blocks ALL actions with a ❌ alert (a
    failed `is-enabled` read must not masquerade as "already safe").
  * enable failure removes the just-written override flag (no 8-day
    release grenade for a later manual `systemctl start` to consume)
    and alerts ❌; an enabled-but-dead unit is restarted weekly with a
    ⚠️ alert (operators who want it stopped must DISABLE it).
  * dry runs (no --apply) never advance weekly state and never touch
    systemd — pure previews. Weekly state is booked per ISO week; an
    applied same-week re-run overwrites that week's booking (recomputed
    from the prior-week baseline), never double-advancing it. This holds
    for EVERY per-week counter — consecutive_weak and
    evidence_gap_streak — which is why both go through book_streak().
  * every completed run VERIFIES Discord delivery (HTTP < 400, retry);
    undelivered -> rc=3 so the cron wrapper curls a fallback. Any crash
    Discords a ❌ CRASHED alert with the traceback tail and exits
    nonzero. A Sunday with NO message therefore means the box, cron, or
    webhook itself is dead — nothing else fails silently.

Steps each run:
  1. sync (`run.py --sync`) unless --no-sync
  2. canonical gate flat-stake bet stream (risk-free) + trailing 2w/1w windows
  3. standard backtest (risk breaker OFF) @5BNB -> gas-inclusive net PnL on
     the 1w window (+ the 2w window iff the positive fallback engages)
  4. positive/negative trigger evaluation (+ Šidák correction, + consecutive-weak counter)
  5. read live bot state (systemctl), decide action, act iff permitted
  6. Discord alert (state change or weekly summary) + artifact + persistent state

Idempotent: artifact dirs are per-day; weekly STATE books once per ISO
week — a same-ISO-week re-run recomputes the week's weak booking from the
persisted prior-week baseline and OVERWRITES it (plus the week's history
entry): corrected code can fix the current week's booking, and nothing
ever double-advances the counter.

Triggers (pinned; 2026-07-09 redesign + 2026-08-16 fallback, per user
decisions):
  POSITIVE:  evaluated on exactly ONE window — the trailing-1w window
             when it clears the n_fires >= 10 information floor, else
             falling back to the trailing-2w window (the floor is an
             information floor, not a time floor: a starved 1w week must
             not mask a strong fortnight). The SPENT window must pass
             all four legs: WR > BREAKEVEN(0.55) AND raw p_upper < 0.10
             (single test) AND n_fires >= 10 AND the standard risk-off
             backtest net PnL (after gas) > 0, with the backtest run
             over that same window — never rescaled from the other one.
             Both windows starved -> the positive trigger cannot fire.
             Action when bot DISABLED: enable + start (under --apply).
             Action when bot ENABLED and breaker-suspended: write the
             cooldown override flag (var/live/cooldown_override.json),
             which the pipeline consumes to release the suspension
             immediately (ignoring extend-while-bleeding).
  EVIDENCE:  the windows are keyed to the newest FIRE, so a bot that stops
             betting freezes them. A newest fire older than
             FIRE_STALE_MAX_AGE_S makes the window FROZEN evidence: it is
             reported loudly and may not be spent on any POSITIVE action.
             It is NOT a disable trigger and never blocks the protective
             disable — see fire_evidence_fresh.
             DENSITY: the point check above is generalised by the
             evidence-gap rule -- no drought longer than
             FIRE_STALE_MAX_AGE_S anywhere INSIDE the spent window, taking
             the max over every consecutive-fire gap plus the trailing gap
             to now. Same constant, same units, no new threshold. It gates
             the ACTION only; window selection is untouched, so the whole
             negative path is unchanged. See window_gap_profile.
  NEGATIVE:  WR leg on the SPENT positive window (the same window the
             positive legs read — a starved-1w regime must not make the
             WR floor unreachable while the 2w window shows a losing
             edge): spent WR < 0.45 -> disable. With no spendable window
             the WR leg is unevaluable; weak-booking covers total
             starvation.  OR  3 consecutive weekly runs weak — weak is
             judged on the SPENT positive window: an evaluable spent
             window with p_upper > 0.5, or both windows starved (n < 10
             fires even across 14 days = a genuine fire collapse). A
             1w-starved week with a strong 2w window is NOT weak.
             Action: disable + stop entirely.
  The 2w permutation stats + latest-100 WR + Šidák are computed and
  reported every week; the 2w risk-off backtest runs only when the
  fallback is engaged. decision.json records the spent window as
  triggers.trigger_window ("1w" | "2w_fallback" | "none").

Artifacts: var/strategy_review/weekly_monitors/<YYYY-MM-DD>/{decision.json},
persistent state var/strategy_review/weekly_monitors/state.json.

Run (VM, real):  .venv/bin/python research/weekly_monitor_state_machine.py --apply
Run (dry, safe): .venv/Scripts/python.exe research/weekly_monitor_state_machine.py
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import research.in_process_runner as ipr  # noqa: E402
from pancakebot.config import (  # noqa: E402
    load_app_config,
    load_strategy_config_from_dict,
)
from pancakebot.constants import BNB_WEI, MAX_GAS_COST_BET_BNB  # noqa: E402
from pancakebot.pool_amounts import compute_pool_amounts_wei  # noqa: E402
from pancakebot.strategy.momentum_gate import MomentumGateConfig  # noqa: E402
from pancakebot.strategy.momentum_pipeline import MomentumOnlyPipeline  # noqa: E402

ROOT = REPO / "var" / "strategy_review" / "weekly_monitors"
STATE_PATH = ROOT / "state.json"
LIVE_UNIT = "pancakebot-live"
CUTOFF, LOOKBACKS, FEE = 2, (3, 7, 15), 0.03
# Named so the fingerprint can report them instead of literals buried in
# the pipeline construction below (MED-A2).
POOL_CUTOFF = 6
MIN_BET_AMOUNT_BNB = 0.001
BREAKEVEN_WR = 0.55
POS_RAW_P = 0.10           # raw permutation p_upper, single test (user decision)
POS_MIN_FIRES = 10         # per-WINDOW information floor; 1w starved -> positive falls back to 2w
NEG_WR_FLOOR = 0.45        # spent-window WR below this -> disable
NEG_CONSECUTIVE_WEAK = 3
NEG_WEAK_P = 0.5           # weak week: SPENT-window p_upper above this (or both windows starved)
N_PERM = 10_000
SEED = 20260630


# --------------------------------------------------------------------------
# canonical gate flat-stake bet stream
# --------------------------------------------------------------------------

# `n` (canonical fire stream) and `backtest.num_bets` sit side by side in
# each window block and are NOT the same count. Before 2026-08-24 they were
# not even measuring the same bot: `n` came from code defaults while the
# backtest read config.toml, so 2026-07-26 recorded n=10 next to
# num_bets~18 over an IDENTICAL epoch range, with nothing in the artifact
# saying why. Both now read the deployed config, so they should track
# closely; what remains is that the backtest simulates every round in the
# range under its own bankroll and stake caps, while `n` counts fires in a
# flat-stake risk-free replay. A small residual gap is expected. A LARGE
# one means they have diverged again -- check strategy_fingerprint first.
_N_VS_BACKTEST_NOTE = (
    "n = fires in the canonical flat-stake risk-free stream; "
    "backtest.num_bets = bets the risk-off backtest placed over the same "
    "epoch range under real bankroll/stake constraints. Both read the "
    "deployed config (see strategy_fingerprint); a large gap means they "
    "have diverged again."
)


def deployed_strategy():
    """``(strategy_config, error_or_None)`` — the DEPLOYED config, falling
    back to CODE DEFAULTS if config.toml cannot be read.

    MED-A1, a safety regression introduced by the first version of this
    change and caught in review. Reading the deployed config made the
    monitor describe the right bot, but doing it UNGUARDED made an
    unreadable config.toml raise inside build_canonical_bets: rc=1, no
    decision.json, no systemd action, no protective disable — with the bot
    enabled and trading. Before that change an unreadable config was
    invisible here (this ran on defaults), so capital protection was
    intact. The commit therefore converted "unreadable config -> evaluates
    on defaults, protection intact" into "unreadable config -> crashes,
    takes no action, bot keeps trading". config.toml is HAND-EDITED for
    exactly the threshold changes this function exists to track, so an
    invalid one is a plausible Sunday, not a hypothetical.

    The fix follows the asymmetry the rest of this file is built on:
    degraded evidence may never ENABLE, but must never stop a DISABLE.
    Falling back to defaults keeps the negative path running on the same
    stream it always had; the returned error blocks every positive action
    and raises a loud banner.
    """
    try:
        return load_app_config(str(REPO / "config.toml")).strategy, None
    except Exception as e:  # noqa: BLE001
        return (load_strategy_config_from_dict({}),
                f"{type(e).__name__}: {e}")


def strategy_fingerprint() -> dict:
    """The deployed knobs this report depends on, recorded in every
    decision.json so a reader can tell WHICH BOT a week describes.

    ``min_pool_bnb_at_cutoff`` is the one that changes which rounds fire,
    so it changes n / wr / p_upper. The stake caps do NOT change the
    canonical stream (it is flat-stake and risk-free) but they do scale
    every btPnL the risk-off backtest reports, which is the same class of
    silent drift -- a reader comparing btPnL across a cap change needs to
    know the cap moved. Recorded together for that reason.
    """
    s, err = deployed_strategy()
    if err is not None:
        # DEGRADE, do not crash: a reporting field must never cost a
        # Sunday against the dead-man contract. deployed_strategy() has
        # already fallen back to code defaults, so record BOTH the error
        # and the fact that the numbers in this report describe the
        # default-config bot rather than the deployed one.
        return dict(error=err, fell_back_to_defaults=True)
    return dict(
        min_pool_bnb_at_cutoff=s.pool_filter.min_pool_bnb_at_cutoff,
        min_payout_multiple_at_cutoff=s.pool_filter.min_payout_multiple_at_cutoff,
        max_bet_bnb_btc_primary=s.risk.max_bet_bnb_btc_primary,
        max_bet_bnb_eth_sol_fallback=s.risk.max_bet_bnb_eth_sol_fallback,
        max_bet_fraction_of_bankroll=s.risk.max_bet_fraction_of_bankroll,
        # MED-A2. These are not read from config by build_canonical_bets --
        # they are hardcoded module constants HERE -- which is exactly why
        # they belong in the fingerprint: a drift between them and the
        # deployed gate is invisible otherwise. The sharpest is
        # mtf_lookbacks: the gate is handed the DEPLOYED value while the
        # kline slice is cut with max(LOOKBACKS) from this module, so
        # raising the deployed lookbacks past this constant silently
        # starves the gate of history instead of failing.
        kline_cutoff_seconds=CUTOFF,
        mtf_lookbacks_used_for_slicing=list(LOOKBACKS),
        mtf_lookbacks_deployed=list(s.gate.mtf_lookbacks),
        pool_cutoff_seconds=POOL_CUTOFF,
        treasury_fee_fraction=FEE,
        # The stake caps do not change WHICH rounds fire, but that
        # inertness is CONTINGENT, not structural: it holds because
        # min_bet_threshold_bnb (0.01) >= min_bet_amount_bnb (0.001), so a
        # capped stake still clears the contract minimum. Drop the caps
        # below the contract minimum and they would start removing fires.
        min_bet_threshold_bnb=s.tier2_sizing.min_bet_threshold_bnb,
    )


def build_canonical_bets():
    """Return (bets, newest_round_lock). Freshness must be judged from the
    newest closed ROUND in the store — the newest FIRE can lag days behind
    during normal signal droughts (~1-2% fire rate) and tripped the
    2026-07-19 Sunday run as a false DATA STALE."""
    rounds = [r for r in ipr._load_all_rounds(use_extended_data=False)
              if r.position in ("Bull", "Bear")]
    rounds.sort(key=lambda r: r.epoch)
    newest_round_lock = max(
        (int(r.lock_at) for r in rounds if r.lock_at is not None), default=0)
    max_lb = max(LOOKBACKS)
    sliced = {}
    for sym, path in (("btc", ipr._BTC_KLINES_PATH), ("eth", ipr._ETH_KLINES_PATH),
                      ("sol", ipr._SOL_KLINES_PATH)):
        uni = ipr._load_klines_unified(
            path, earliest_offset=CUTOFF + max_lb + 1, latest_offset=CUTOFF + 1)
        sliced[sym] = {ep: ipr._slice_per_entry(
            kl, kline_cutoff_seconds=CUTOFF, max_lookback=max_lb,
            earliest_offset=CUTOFF + max_lb + 1) for ep, kl in uni.items()}
    # THE DEPLOYED strategy config, not code defaults. Until 2026-08-24
    # this read load_strategy_config_from_dict({}), so the canonical fire
    # stream described a bot with whatever the code defaults happened to
    # be while a differently-configured bot traded. That went unnoticed
    # until the pool filter moved 1.5 -> 1.25 and the two diverged by
    # +29.3% of fires all-history, +51.4% over the last 70 days.
    #
    # Why that was sharp rather than cosmetic:
    #   * The POSITIVE trigger became a conjunction across TWO bots --
    #     n/wr/p_upper from the default-config stream, bt.net_pnl_bnb from
    #     a deployed-config backtest over the same declared epochs. No
    #     single bot's configuration makes that conjunction the right test.
    #   * The DISABLE path is the exposed one. Replaying window selection
    #     with the real stream: on 2026-07-26 and 2026-08-23 this stream
    #     sat at EXACTLY POS_MIN_FIRES (n=10) while the deployed filter
    #     gave 18 and 11. One fire lower and trigger_window becomes
    #     "none", which books weak=True, and three consecutive weak weeks
    #     auto-disable the live unit. The monitor could reach that on
    #     evidence starvation the bot was not experiencing.
    #   * The WR bias is small and FLIPS SIGN by regime (+0.0071
    #     all-time, -0.0174 over 70d), so there is no direction to
    #     correct for -- only an unknown.
    #
    # Chosen over freezing the stream at a pinned config. A frozen stream
    # is comparable across history and wrong about the present; a
    # config-tracking stream that RECORDS its config is right about the
    # present and still comparable, because the frozen view can be
    # re-derived from the append-only store while the live view cannot be
    # re-derived from an artifact that never recorded which bot it
    # described. Freezing discards information that recording keeps --
    # hence strategy_fingerprint() below, written to every decision.json.
    #
    # COST, stated because it is real: n, WR and p_upper are now
    # DISCONTINUOUS at any config change, so consecutive_weak and the
    # permutation p-values are not comparable across that boundary. Treat
    # a change in the fingerprint as an explicit epoch boundary when
    # reading artifact history.
    sc, _cfg_err = deployed_strategy()
    gate_cfg = MomentumGateConfig(
        enabled=True, bnb_symbol="BNB-USDT", btc_symbol="BTC-USDT",
        eth_symbol="ETH-USDT", sol_symbol="SOL-USDT", kline_cutoff_seconds=CUTOFF,
        mtf_lookbacks=sc.gate.mtf_lookbacks,
        mtf_min_return_threshold=sc.gate.mtf_min_return_threshold)
    pipe = MomentumOnlyPipeline(
        config=gate_cfg, strategy_config=sc, gate=None, kline_cutoff_seconds=CUTOFF,
        pool_cutoff_seconds=POOL_CUTOFF, min_bet_amount_bnb=MIN_BET_AMOUNT_BNB,
        treasury_fee_fraction=FEE,
        bankroll_tracker=None)
    pipe.refresh_btc_klines(btc_klines_by_epoch=sliced["btc"])
    pipe.refresh_eth_klines(eth_klines_by_epoch=sliced["eth"])
    pipe.refresh_sol_klines(sol_klines_by_epoch=sliced["sol"])
    pipe.refresh_bnb_klines(bnb_klines_by_epoch={})
    bets = []
    for r in rounds:
        d = pipe.decide_open_round(round_t=r)
        if d.action != "BET":
            continue
        pools = compute_pool_amounts_wei(bets=r.bets)
        fb, fbe = pools.bull_wei / BNB_WEI, pools.bear_wei / BNB_WEI
        if fb <= 0 or fbe <= 0:
            continue
        tot = fb + fbe
        bull = d.bet_side == "Bull"
        outcome_bull = r.position == "Bull"
        pay = (tot * (1 - FEE) / fb) if bull else (tot * (1 - FEE) / fbe)
        win = bull == outcome_bull
        bets.append(dict(epoch=int(r.epoch), lock=int(r.lock_at), side_bull=bull,
                         outcome_bull=outcome_bull, payout_bull=tot * (1 - FEE) / fb,
                         payout_bear=tot * (1 - FEE) / fbe, win=win,
                         pnl=(pay - 1.0) if win else -1.0))
    return bets, newest_round_lock


def perm(bets, n_iter=N_PERM, seed=SEED):
    if len(bets) < POS_MIN_FIRES:
        return dict(n=len(bets), insufficient=True)
    obs = float(np.mean([b["pnl"] for b in bets]))
    out = np.array([b["outcome_bull"] for b in bets])
    pb = np.array([b["payout_bull"] for b in bets])
    pr = np.array([b["payout_bear"] for b in bets])
    side = np.array([b["side_bull"] for b in bets])
    rng = np.random.default_rng(seed)
    null = np.empty(n_iter)
    for i in range(n_iter):
        p = rng.permutation(len(out))
        null[i] = np.where(out[p] == side, np.where(side, pb[p], pr[p]) - 1.0, -1.0).mean()
    return dict(n=len(bets), wr=round(float(np.mean([b["win"] for b in bets])), 4),
                obs_mean_pnl=round(obs, 4), null_mean=round(float(null.mean()), 4),
                p_upper=round(float((null >= obs).mean()), 5))


def fire_evidence_fresh(newest_fire_lock: int, now: float) -> bool:
    """True when the newest FIRE is recent enough for its window to count
    as live evidence (see FIRE_STALE_MAX_AGE_S).

    Gates POSITIVE actions only. A stale fire stream never blocks the
    protective disable: if the last evidence we have says the strategy is
    losing, a drought must not rescue the bot from being switched off.
    Nor is staleness itself a disable trigger - a genuinely quiet week is
    legitimate, the bot self-heals from infrastructure faults, and the
    runtime pool-gate alarm is what pages for "up but unable to trade".
    """
    return (now - float(newest_fire_lock)) <= FIRE_STALE_MAX_AGE_S


def window_gap_profile(window: list, now: float) -> tuple[float | None, float]:
    """``(largest gap BETWEEN consecutive fires, trailing gap to now)``, in
    hours. ``None`` for the first when the window holds fewer than 2 fires.

    Feeds the evidence-gap rule: no drought longer than
    FIRE_STALE_MAX_AGE_S anywhere inside the SPENT window. The shipped
    ``fire_evidence_fresh`` tests exactly ONE gap -- the trailing one -- so
    18 stale fires plus a single fresh one passes it. Taking the max over
    every gap is a strict generalisation of that test using the same
    constant and the same units, which is why it needs no new threshold: a
    count rule (>=K fresh) or a fraction rule (>=X% fresh) would have to be
    fitted to seven Sunday observations, whereas 96h came from 304 real
    inter-fire gaps. Both alternatives were swept and are dominated -- no K
    separates the one bad decision from the good ones (K=6 misses
    2026-08-16, K=8 also blocks 07-12 and 07-26), and X=0.30 misses
    2026-08-16 outright.

    KNOWN LIMITATIONS, recorded rather than implied away:

    * EFFICACY IS n=1. Across all nine archived Sundays this rule changes
      exactly one decision (2026-08-16, max gap 117.5h). The "derived from
      304 real gaps" framing describes where the 96h CONSTANT came from,
      not how much evidence supports the rule catching anything: that
      rests on a single case, and it clears the bound by 22.4% while
      sitting only 2.1% below 120h, the smallest bound that would drop the
      false-positive rate under ~5%.

    * THE SHAPE IS NARROWED, NOT CLOSED. A mostly-stale window that is
      sparsely BRIDGED still passes: 2026-08-09's spent window was 86%
      stale by composition and its largest gap was 57.0h, comfortably
      inside the bound. This rule removes the "one fresh fire rescues 18
      stale ones" hole; it does not guarantee the window is dense.

    * EXPOSURE IS REAL AND MEASURED. Replaying the monitor's own window
      selection once per day against the live stream: 15.7% of days in the
      current regime are blocked at 96h, against 0.0% over the preceding
      180 days. At half the fire rate the median is 21.4% and the worst of
      nine seeds is 37.1%. See FIRE_STALE_MAX_AGE_S for why 96h is kept
      anyway and why no single constant satisfies both efficacy and
      false-positive rate.
    """
    locks = sorted(int(b["lock"]) for b in window)
    if not locks:
        return None, float("inf")
    trailing_h = (now - locks[-1]) / 3600.0
    if len(locks) < 2:
        return None, trailing_h
    gaps_h = [(locks[i + 1] - locks[i]) / 3600.0 for i in range(len(locks) - 1)]
    return max(gaps_h), trailing_h


def window_fire_composition(window: list, now: float) -> dict:
    """How much of a window's evidence is RECENT, not merely present.

    ``fire_evidence_fresh`` is a POINT check on the newest fire, so a
    single recovery bet un-freezes an otherwise all-stale window: 18 fires
    aged 9-13 days plus one fire 2h old passes it. The evidence-gap rule
    (``window_gap_profile``) now blocks that shape, so this composition is
    no longer the only defence -- it remains the human-readable summary,
    reported every week so a mixed window cannot be MISREAD even when the
    gap rule lets it through.
    """
    n = len(window)
    fresh = sum(1 for b in window
                if (now - b["lock"]) <= FIRE_STALE_MAX_AGE_S)
    d7 = sum(1 for b in window if (now - b["lock"]) <= 7 * 86400)
    oldest_h = ((now - min(b["lock"] for b in window)) / 3600.0) if window else None
    return dict(
        n=n, fresh_within_bound=fresh, in_last_7d=d7,
        stale=n - fresh,
        oldest_fire_age_h=round(oldest_h, 1) if oldest_h is not None else None,
    )


def fire_gap_p99_h(bets: list, now: float, lookback_days: int = 70) -> float | None:
    """Trailing p99 inter-fire gap in hours, same definition as the one
    FIRE_STALE_MAX_AGE_S was derived from. Reported every week so the
    threshold's shrinking margin is visible as the fire rate falls."""
    locks = sorted(int(b["lock"]) for b in bets)
    cut = now - lookback_days * 86400
    gaps = [(locks[i + 1] - locks[i]) / 3600.0
            for i in range(len(locks) - 1) if locks[i] > cut]
    if len(gaps) < 20:
        return None
    gaps.sort()
    return round(gaps[min(len(gaps) - 1, int(0.99 * len(gaps)))], 1)


def positive_window(p1: dict, p2: dict) -> str:
    """Select the window the POSITIVE trigger spends: '1w' when it clears
    the POS_MIN_FIRES information floor, else '2w_fallback' when the 2w
    window clears it (the floor is an information floor, not a time floor
    — a starved 1w week must not mask a strong fortnight), else 'none'
    (both windows starved: the positive trigger cannot fire). The
    NEGATIVE trigger never consults this — it is 1w-only."""
    if not p1.get("insufficient"):
        return "1w"
    if not p2.get("insufficient"):
        return "2w_fallback"
    return "none"


def spent_stats(trigger_window: str, p1: dict, p2: dict) -> dict:
    """The permutation stats handed to the trigger legs: the 2w stats
    only when the fallback is spent; '1w' and 'none' both read the 1w
    stats (for 'none' they are insufficient, keeping every leg off)."""
    return p2 if trigger_window == "2w_fallback" else p1


def negative_wr_leg(trigger_window: str, stats: dict) -> bool:
    """The disable WR leg reads the SPENT window, same source as the
    positive legs — a starved-1w regime must not make the WR floor
    unreachable while the 2w window shows a losing edge. With no
    spendable window ('none') the leg is unevaluable; weak-booking
    covers total starvation."""
    return bool(trigger_window != "none"
                and stats.get("wr") is not None
                and stats["wr"] < NEG_WR_FLOOR)


def evaluate_positive(stats: dict, bt: dict) -> bool:
    """The four positive legs, applied to the SPENT window's permutation
    stats and the risk-off backtest run over that SAME window. A missing
    or errored backtest (no net_pnl_bnb key) fails the PnL leg."""
    return bool(
        not stats.get("insufficient") and stats.get("wr", 0) > BREAKEVEN_WR
        and stats.get("p_upper", 1) < POS_RAW_P
        and stats.get("n", 0) >= POS_MIN_FIRES
        and bt.get("net_pnl_bnb", -1) > 0)


def weak_week(trigger_window: str, stats: dict) -> bool:
    """Weak is judged on the SPENT positive window: an evaluable spent
    window whose p_upper exceeds NEG_WEAK_P, or no spendable window at
    all ('none' — not even the 2w window produced POS_MIN_FIRES fires, a
    fortnight-wide fire collapse with zero statistical evidence to defend
    betting on). A 1w-starved week with a strong 2w window is NOT weak.
    `stats` is the spent window's perm stats (ignored for 'none')."""
    if trigger_window == "none":
        return True
    return bool(stats.get("p_upper") is not None
                and stats["p_upper"] > NEG_WEAK_P)


def book_streak(st: dict, key: str, *, same_week_rerun: bool,
                hit: bool) -> tuple[int, int]:
    """Return (baseline, counter) for a per-ISO-week consecutive counter.

    First run of the week: baseline = last week's persisted counter,
    advance from it. Same-ISO-week re-run: RECOMPUTE from the persisted
    baseline — overwrite semantics, so a re-run under corrected code can
    fix the week's booking, and nothing ever double-advances. The
    baseline fallback for state files that predate the key assumes the
    last booking was a hit (stored-1); a non-hit booking stores 0 either
    way, so the fallback can only over-forgive, never over-count.

    Shared rather than duplicated: evidence_gap_streak was written as a
    bare `stored + 1` and DID double-advance on an applied same-week
    re-run (a single blocked Sunday re-run wrote 2 and fired the
    "2 CONSECUTIVE SUNDAYS" banner), contradicting the overwrite
    contract in this module's docstring two lines from correct code."""
    stored = int(st.get(key, 0))
    if same_week_rerun:
        baseline = int(st.get(f"{key}_baseline", max(stored - 1, 0)))
    else:
        baseline = stored
    return baseline, (baseline + 1 if hit else 0)


def book_weak_week(st: dict, *, same_week_rerun: bool, weak: bool) -> tuple[int, int]:
    """(baseline, consec) for this ISO week's weak booking."""
    return book_streak(st, "consecutive_weak",
                       same_week_rerun=same_week_rerun, hit=weak)


def _window_desc(tag: str, stats: dict, bt: dict | None = None,
                 comp: dict | None = None) -> str:
    """One window, one clause: 'tag: n=.. WR=.. p=..' when evaluable, or
    'tag: n=..<10 insufficient' when starved — a starved window must read
    as starved, never as 'WR=None p=None'. Appends btPnL when a backtest
    dict is provided (and non-empty)."""
    if stats.get("insufficient"):
        s = f"{tag}: n={stats.get('n')}<{POS_MIN_FIRES} insufficient"
    else:
        s = (f"{tag}: n={stats.get('n')} WR={stats.get('wr')} "
             f"p={stats.get('p_upper')}")
    if bt:
        s += f" btPnL={bt.get('net_pnl_bnb')}"
    if comp:
        # Composition always shown: a window can pass the point-in-time
        # freshness gate while being almost entirely pre-outage evidence.
        s += (f" [fresh {comp['fresh_within_bound']}/{comp['n']}"
              f" last7d={comp['in_last_7d']}"
              f" oldest={comp['oldest_fire_age_h']}h]")
    return s


# --------------------------------------------------------------------------
# standard backtest (risk breaker OFF) on a window -> gas-inclusive net PnL
# --------------------------------------------------------------------------

BACKTEST_TIMEOUT_S = 1800   # a hung backtest must not eat the weekly slot
SYNC_TIMEOUT_S = 3600       # observed healthy sync ~14 min; 60 min = hung
FRESH_MAX_AGE_S = 36 * 3600  # newest closed ROUND older than this = stale

# Newest FIRE older than this -> the evaluation windows are FROZEN: they
# re-score bets that already happened and describe a week the bot did not
# trade in. Distinct from FRESH_MAX_AGE_S above, which watches the dense
# ROUND stream and is refreshed by every --sync even while the bot places
# no bets at all (the 2026-08-17 pool outage: rounds fresh, fires frozen
# since 08-14, monitor would have reported healthy stats and action=none).
#
# Derived from the fire stream itself. REPRODUCIBLE DEFINITION: fires are
# the `lock` timestamps of build_canonical_bets() (non-failed closed rounds
# the canonical gate would have BET, both pools > 0); gaps are between
# consecutive fires; the sample is every gap whose EARLIER endpoint is
# strictly newer than (newest_fire - 70d), with newest_fire = 2026-08-14
# 18:06Z. On that definition n=272: p50 2.0h, p90 17.5h, p95 24.1h,
# p99 57.0h, max 117.5h. (A re-computation using a different window edge
# got n=304 / p50 1.5 / p90 16.1 / p95 23.0 / p99 56.7 — the max, the
# exceedance count and the time-weighted figure below were identical, so
# the choice below is insensitive to the edge; state the edge when
# re-deriving.)
#
# SUPERSEDED DERIVATION, kept because it is easy to re-derive by mistake:
# 96h was originally justified as "the LARGEST bound that still catches the
# worst drought ever observed", on time-weighted TRAILING exceedance
# (48h -> 7.79%, 72h -> 3.86%, 96h -> ~1.3%, 120h -> 0.00%). That was true
# of the trailing-only framing and is FALSE under the density rule: the
# worst observed drought is a 117.5h INTERNAL gap, so EVERY bound below
# 117.5h catches it -- 116h included. 96h is not the largest such bound, it
# is one choice among many. Those exceedance figures are also not
# commensurable with the block rates below: one is the time-weighted share
# of wall clock spent inside a trailing gap, the other is the share of DAYS
# on which the density rule would block. Do not read them as one series.
#
# RE-DERIVED 2026-08-24 against the live 1,982-fire stream, replaying the
# monitor's own spent-window selection once per day and asking how often
# the rule would block. The band the efficacy criterion actually points at
# is measured here rather than jumped over:
#
#   bound  catches 08-16?  current regime  half rate (med / worst, 9 seeds)
#    96h        yes            15.7%            15.9%  /  34.4%
#   104h        yes            15.7%            12.1%  /  34.4%
#   112h        yes            14.3%            10.6%  /  31.2%
#   116h        yes            14.3%            10.6%  /  31.2%
#   120h        NO              4.3%             6.7%  /  31.2%
#   144h        NO              2.9%             5.0%  /  15.6%
#   168h        NO              0.0%             3.3%  /  14.1%
#
# "Current regime" = the last 70 evaluable days, which is the honest
# denominator: over the preceding 180 days the rule blocks 0.0% of days,
# because the gap distribution has MOVED (last 70d p99 117.5h max 161.5h;
# days 70-250 p99 26.3h max 53.6h). Measuring across both regimes averages
# a world that no longer exists into the answer.
#
# THE CHOICE, made on those numbers instead of asserted:
#   * EFFICACY needs bound < 117.5h -- the max internal gap of 2026-08-16,
#     the one decision this rule changes. At 120h and above the rule is
#     decoration: it stops catching the only case it was built for. So
#     FALSE POSITIVES, which need >= 120h to fall under ~5%, cannot be
#     bought at all without giving up the rule.
#   * INSIDE the efficacy-preserving band the false-positive rate moves by
#     exactly ONE DAY: 11 blocked days out of 70 at 96h, 10 at 116h. That
#     is not a dominance result. It is a single day at the edge of a single
#     episode (see COST below) and is not a basis for choosing.
#   * What does separate them is EFFICACY MARGIN. 117.5h sits 22.4% above
#     96h but only 1.3% above 116h -- a 116h bound would catch the one case
#     it exists for by 1.5 hours.
#
# 96h is KEPT, and explicitly NOT because it measures better: 116h is
# weakly better on false positives (one day today, ~5pp under a simulated
# halving of the fire rate). It is kept to cover droughts in the 96-116h
# band, which have never been observed and cannot be ruled out, and to keep
# the efficacy leg robust to measurement noise rather than resting it on a
# 1.5h margin. The half-rate column argues for a wider bound but is
# inconsistent as a tuning input: halving the fire rate lengthens the
# droughts we want to CATCH as much as the ones we want to forgive, so it
# cannot move the bound while the efficacy anchor stays pinned to a
# full-rate event. Blocking is also the safe direction -- it refuses to
# ENABLE, never to disable.
#
# COST OF THE CHOICE, stated correctly: 15.7% of days is 11 blocked days
# out of 70, but they are ONE CONTIGUOUS EPISODE -- 2026-08-12 through
# 2026-08-22, the Aug 7 -> Aug 12 drought aging through the window. Not 11
# independent blocks, and NOT a ~1-in-6 per-Sunday probability; framing it
# as a rate implies an independence that is not there. The real cost is
# roughly ONE blocked episode straddling one or two Sundays, which is
# materially cheaper than the rate suggests and cuts in this rule's favour.
# The flip side is that a single future drought blocks a RUN of consecutive
# Sundays -- which is why consecutive evidence-gap Sundays escalate
# (evidence_gap_streak): a slow strangulation must announce itself instead
# of looking like a quiet run of ordinary weeks.
#
# What a constant CANNOT do is distinguish "evidence is thin because the
# market is thin" from "evidence is thin because we are broken". If this
# keeps firing, the fix is the fire rate, not the bound.
#
# A RELATIVE threshold (k x trailing mean gap) was rejected deliberately:
# it inflates as the fire rate collapses, so it goes quiet exactly during
# a slow strangulation - the same blind-spot class as a gate that cannot
# fire when it matters most.
FIRE_STALE_MAX_AGE_S = 96 * 3600
SYNC_FAIL_DISABLE_STREAK = 3  # blind weeks in a row before protective disable


def risk_off_backtest(epoch_start: int, epoch_end: int, out_dir: Path,
                      bankroll: float = 5.0,
                      cfg_name: str = "risk_off_config.toml") -> dict:
    section = None
    lines = []
    for raw in (REPO / "config.toml").read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if s.startswith("[") and s.endswith("]"):
            section = s
        line = raw
        if section == "[backtest]":
            if s.startswith("initial_bankroll_bnb"):
                line = f"initial_bankroll_bnb = {bankroll}"
            elif s.startswith("# epoch_start") or s.startswith("epoch_start"):
                line = f"epoch_start = {epoch_start}"
            elif s.startswith("# epoch_end") or s.startswith("epoch_end"):
                line = f"epoch_end = {epoch_end}"
        elif section == "[strategy.risk]":
            if s.startswith("max_drawdown_fraction_from_peak"):
                line = "max_drawdown_fraction_from_peak = 1.0"
            elif s.startswith("min_bankroll_bnb_to_bet"):
                line = "min_bankroll_bnb_to_bet = 0.001"
            elif s.startswith("cooldown_rounds"):
                line = "cooldown_rounds = 0"
        lines.append(line)
    cfg = out_dir / cfg_name
    cfg.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        r = subprocess.run([sys.executable, str(REPO / "run.py"), "--backtest",
                            "--config", str(cfg)], cwd=REPO, capture_output=True,
                           text=True, timeout=BACKTEST_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return dict(error=f"backtest timed out after {BACKTEST_TIMEOUT_S}s")
    if r.returncode != 0:
        return dict(error=r.stderr[-800:])
    summ = json.loads((REPO / "var" / "backtest" / "summary.json").read_text(encoding="utf-8"))
    # Window-identity guard: summary.json is a SHARED artifact and this
    # helper runs once per window — a summary whose boundary rounds don't
    # match the requested range is another window's result (or a config
    # substitution that failed to apply) and must fail the PnL leg, never
    # silently stand in. Equality is exact: the requested boundaries are
    # canonical fire epochs, i.e. non-failed closed rounds in the same
    # store the backtest loads, so they are always its boundary sim rounds.
    if (summ.get("first_epoch") != epoch_start
            or summ.get("last_epoch") != epoch_end):
        return dict(error=(
            f"backtest summary window mismatch: requested "
            f"[{epoch_start}..{epoch_end}], summary "
            f"[{summ.get('first_epoch')}..{summ.get('last_epoch')}]"))
    return dict(net_pnl_bnb=summ["net_pnl_bnb"], num_bets=summ["num_bets"],
                win_rate=summ["win_rate"], gas_per_bet=MAX_GAS_COST_BET_BNB)


# --------------------------------------------------------------------------
# systemd state + actions (guarded)
# --------------------------------------------------------------------------

def _systemctl(*args) -> tuple[int, str]:
    try:
        r = subprocess.run(["systemctl", *args], capture_output=True, text=True, timeout=30)
        return r.returncode, (r.stdout + r.stderr).strip()
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return -1, f"systemctl unavailable: {e}"


def read_bot_state() -> dict:
    ac_rc, ac = _systemctl("is-active", LIVE_UNIT)
    en_rc, en = _systemctl("is-enabled", LIVE_UNIT)
    available = not ac.startswith("systemctl unavailable")
    return dict(available=available, active=ac, enabled=en,
                is_running=(ac == "active"), is_enabled=(en == "enabled"))


def do_enable() -> tuple[bool, str]:
    rc, out = _systemctl("enable", "--now", LIVE_UNIT)
    return rc == 0, f"enable --now rc={rc}: {out}"


def do_disable() -> str:
    rc1, o1 = _systemctl("disable", LIVE_UNIT)
    rc2, o2 = _systemctl("stop", LIVE_UNIT)
    return f"disable rc={rc1} / stop rc={rc2}: {o1} {o2}"


# --------------------------------------------------------------------------
# Discord (best-effort)
# --------------------------------------------------------------------------

def discord(msg: str) -> bool:
    """Post + VERIFY delivery (HTTP < 400). Returns False on any failure so
    the caller can exit nonzero and the cron wrapper can fire its own
    fallback — an undelivered weekly alert must never look like success."""
    url = os.environ.get("PANCAKEBOT_GENERAL_DISCORD_WEBHOOK_URL", "")
    if not url:
        return False
    for attempt in (1, 2):
        try:
            import requests
            r = requests.post(url, json={"content": msg[:1900]}, timeout=10)
            if r.status_code < 400:
                return True
        except Exception:
            pass
        if attempt == 1:
            time.sleep(5)
    return False


# --------------------------------------------------------------------------
# state persistence (idempotency + consecutive-weak counter)
# --------------------------------------------------------------------------

def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return dict(consecutive_weak=0, last_week=None, last_action=None, history=[])


def save_state(st: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(st, indent=2), encoding="utf-8")


def _date_key(s: str) -> str:
    """argparse validator: the week key is a YYYY-MM-DD date (despite the
    legacy --iso-week flag name), fed to _iso_week_key for idempotency."""
    import datetime as _dt
    try:
        _dt.date.fromisoformat(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected YYYY-MM-DD, got {s!r}")
    return s


def _iso_week_key(day: str) -> str:
    """'2026-07-19' -> '2026-W29'. Same-week idempotency compares ISO weeks
    (a mid-week manual re-run must not double-advance the weak counter, per
    the module docstring; the raw date comparison only caught same-DAY)."""
    import datetime as _dt
    y, w, _ = _dt.date.fromisoformat(day).isocalendar()
    return f"{y}-W{w:02d}"


RETRY_MARKER_PATH = ROOT / "retry_pending.json"


def _load_retry_marker(path: Path | None = None) -> dict | None:
    """Read the pending-retry marker; a corrupt or malformed marker is
    deleted and treated as absent (garbage must not wedge the daily gate)."""
    p = path if path is not None else RETRY_MARKER_PATH
    if not p.exists():
        return None
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(doc, dict):
            raise ValueError("marker not a dict")
        _date_key(str(doc["sunday_key"]))
        doc["attempts"] = int(doc.get("attempts", 1))
        return doc
    except (OSError, ValueError, TypeError, KeyError,
            json.JSONDecodeError, argparse.ArgumentTypeError):
        try:
            p.unlink()
        except OSError:
            pass
        return None


def _write_retry_marker(*, sunday_key: str, attempts: int, reason: str,
                        path: Path | None = None) -> None:
    """Atomic (tmp+rename): the wrapper's daily existence check and a
    concurrent manual run must never see a torn marker."""
    p = path if path is not None else RETRY_MARKER_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(
        ts=time.time(), sunday_key=sunday_key, attempts=int(attempts),
        reason=reason), indent=2), encoding="utf-8")
    tmp.replace(p)


def _clear_retry_marker(path: Path | None = None) -> None:
    p = path if path is not None else RETRY_MARKER_PATH
    try:
        p.unlink()
    except OSError:
        pass


def _resolve_run_context(week: str, marker: dict | None) -> tuple[str, bool, bool]:
    """Return (effective_week, retry_mode, completed_blind_week).

    Mon-Sat with a pending marker = makeup attempt for that blind Sunday.
    The run is keyed to the MISSED Sunday: Sundays are the LAST ISO day,
    so the retry days after a blind Sunday fall in the NEXT ISO week —
    keying a recovery by calendar date would consume the FOLLOWING
    Sunday's once-per-ISO-week state advance. A Sunday run that finds a
    marker from a previous Sunday means that whole week stayed blind
    (Sunday + every retry failed): report it, supersede the marker.
    """
    import datetime as _dt
    dow = _dt.date.fromisoformat(week).isoweekday()
    if marker is None:
        return week, False, False
    if dow == 7:
        return week, False, str(marker.get("sunday_key")) != week
    return str(marker["sunday_key"]), True, False


def write_override_flag(*, week: str, reason: str, stats: dict,
                        trigger_window: str) -> Path:
    """Write the cooldown-override flag the pipeline consumes on its next
    paused round (fresh <= 8 days; `_consume_override_flag`). Atomic
    tmp+rename: the running bot's reader DELETES the flag on a parse error,
    so a torn write would silently discard the release. The window block is
    forensic only — the pipeline reader checks nothing but ts freshness —
    and records the SPENT positive window's legs."""
    flag = REPO / "var" / "live" / "cooldown_override.json"
    flag.parent.mkdir(parents=True, exist_ok=True)
    tmp = flag.with_suffix(flag.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(
        ts=time.time(), week=week, reason=reason,
        trigger_window=trigger_window,
        window=dict(wr=stats.get("wr"), p_upper=stats.get("p_upper"),
                    n=stats.get("n")),
    ), indent=2), encoding="utf-8")
    tmp.replace(flag)
    return flag


# --------------------------------------------------------------------------
def _main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="actually act on systemd (else dry-run)")
    ap.add_argument("--no-sync", action="store_true", help="skip run.py --sync")
    ap.add_argument("--iso-week", type=_date_key, default=None,
                    help="override the week key with a YYYY-MM-DD date (testing)")
    args = ap.parse_args()

    week = args.iso_week or time.strftime("%Y-%m-%d", time.gmtime())
    marker = _load_retry_marker()
    week, retry_mode, completed_blind_week = _resolve_run_context(week, marker)
    attempts_so_far = int(marker["attempts"]) if retry_mode else 0
    if args.apply and marker is not None and not retry_mode:
        # Sunday supersedes any pending retry (same-day rerun or a fully
        # blind previous week — completed_blind_week reports the latter).
        _clear_retry_marker()
    out_dir = ROOT / week
    out_dir.mkdir(parents=True, exist_ok=True)
    st = load_state()
    last_week = st.get("last_week")
    same_week_rerun = (
        last_week is not None
        and _iso_week_key(last_week) == _iso_week_key(week))

    sync_ok = True
    if not args.no_sync:
        print("--- sync ---", flush=True)
        try:
            r = subprocess.run([sys.executable, str(REPO / "run.py"), "--sync"],
                               cwd=REPO, timeout=SYNC_TIMEOUT_S)
            sync_ok = (r.returncode == 0)
        except subprocess.TimeoutExpired:
            sync_ok = False
        if not sync_ok:
            print("!!! sync FAILED — evaluating on last-synced data; "
                  "positive actions blocked this week", flush=True)

    print("--- canonical bet stream ---", flush=True)
    bets, newest_round_lock = build_canonical_bets()
    if not bets:
        # Unusable stores (empty/corrupt). Raise -> the crash handler
        # Discords it; silence is never an outcome.
        raise RuntimeError("canonical bet stream is EMPTY — data stores unusable")
    max_lock = max(b["lock"] for b in bets)

    # Evidence gate (2026-07-18, fixed 2026-07-19): a zero exit from sync
    # is not enough — a stalled indexer can exit 0 without advancing the
    # stores, and --no-sync skips it entirely. Freshness is judged from the
    # newest closed ROUND in the store; the newest FIRE (max_lock, which
    # keys the evaluation windows) lags days behind in normal signal
    # droughts and must not trip this gate.
    # ONE clock for the whole evidence evaluation. The gap check lands on
    # the far side of the risk-off backtests (BACKTEST_TIMEOUT_S = 1800s
    # each), so re-reading time.time() there drifted 5-60 minutes against a
    # 96h bound and could disagree with the label, the reason text and the
    # banner at the boundary.
    now_ts = time.time()
    data_fresh = (now_ts - newest_round_lock) <= FRESH_MAX_AGE_S
    evidence_ok = sync_ok and data_fresh
    # Fire-stream freshness is a SEPARATE gate, deliberately NOT folded into
    # evidence_ok: a quiet week is not a sync failure, so it must not freeze
    # the weekly counters or arm the daily retry machinery. It gates the
    # POSITIVE actions only (see fire_evidence_fresh).
    fire_age_h = (now_ts - max_lock) / 3600.0
    fire_fresh = fire_evidence_fresh(max_lock, now_ts)
    # Named _base because it is NOT the final verdict: the evidence-gap
    # rule below ANDs into it once the spent window is known. The rename
    # makes the two-stage assignment structurally impossible to misread as
    # a finished value.
    positive_evidence_base = evidence_ok and fire_fresh
    if not fire_fresh:
        print(f"!!! fire stream STALE: newest fire "
              f"{time.strftime('%Y-%m-%d %H:%M', time.gmtime(max_lock))}Z "
              f"({fire_age_h:.1f}h > {FIRE_STALE_MAX_AGE_S / 3600:.0f}h) — the "
              "evaluation windows are FROZEN; positive actions blocked",
              flush=True)
    if not data_fresh:
        print("!!! data STALE: newest closed round lock "
              f"{time.strftime('%Y-%m-%d %H:%M', time.gmtime(newest_round_lock))}Z "
              "— positive actions blocked", flush=True)

    # Blindness streak: consecutive FULLY-blind ISO weeks (Sunday + every
    # daily retry failed — detected by the next Sunday superseding an
    # unresolved marker). Any fresh evidence resets it. Persisted
    # immediately (the weekly advance below is deliberately frozen on
    # blind runs, so this needs its own write); the disable check adds
    # the currently-blind attempt so escalation timing matches the old
    # per-Sunday counting.
    streak = int(st.get("sync_fail_streak", 0))
    if args.apply and not args.no_sync:
        new_streak = 0 if evidence_ok else (
            streak + 1 if completed_blind_week else streak)
        if new_streak != streak:
            streak = new_streak
            st["sync_fail_streak"] = streak
            save_state(st)

    # Retry marker lifecycle: a blind applied run (Sunday OR retry day)
    # arms/extends daily retries; fresh evidence on a retry clears them.
    # Dry runs never touch the marker.
    if args.apply and not args.no_sync:
        if evidence_ok:
            if retry_mode:
                _clear_retry_marker()
        else:
            _write_retry_marker(
                sunday_key=week, attempts=attempts_so_far + 1,
                reason="sync_failed" if not sync_ok else "data_stale")

    def window(days):
        cut = max_lock - days * 86400
        return [b for b in bets if b["lock"] >= cut]

    w2, w1 = window(14), window(7)
    # Composition of each window's evidence (see window_fire_composition):
    # reported always, so a window that passes the point-in-time freshness
    # gate on one recent fire cannot be misread as a live window.
    comp1 = window_fire_composition(w1, now_ts)
    comp2 = window_fire_composition(w2, now_ts)
    gap_p99_h = fire_gap_p99_h(bets, now_ts)
    e1 = (min(b["epoch"] for b in w1), max(b["epoch"] for b in w1)) if w1 else (0, 0)
    e2 = (min(b["epoch"] for b in w2), max(b["epoch"] for b in w2)) if w2 else (0, 0)
    p2, p1 = perm(w2), perm(w1)

    print("--- risk-off standard backtest (1w @5BNB) ---", flush=True)
    bt = risk_off_backtest(e1[0], e1[1], out_dir, bankroll=5.0) if w1 else {}

    # Positive-trigger window selection: the 2w risk-off backtest is REAL
    # and runs only when the fallback is engaged (1w starved, 2w evaluable)
    # — the PnL leg is never rescaled from the 1w run or simulated.
    trigger_window = positive_window(p1, p2)
    bt2 = {}
    if trigger_window == "2w_fallback":
        print("--- risk-off standard backtest (2w fallback @5BNB) ---", flush=True)
        bt2 = risk_off_backtest(e2[0], e2[1], out_dir, bankroll=5.0,
                                cfg_name="risk_off_config_2w.toml")
    pos_stats = spent_stats(trigger_window, p1, p2)
    pos_bt = bt2 if trigger_window == "2w_fallback" else bt

    # Evidence-gap rule: no drought longer than the staleness bound
    # ANYWHERE inside the spent window (see window_gap_profile). This is an
    # ADDITIONAL condition on positive_evidence_ok and deliberately does
    # NOT touch positive_window(): window selection feeds spent_stats,
    # negative_wr_leg and weak_week, and trigger_window == "none" already
    # books weak=True, so a density-driven "none" would start advancing the
    # DISABLE counter on evidence-quality grounds. Gate the action, not the
    # window -- that keeps the whole negative machinery bit-for-bit
    # unchanged. Positive-only for the same reason HIGH-1 was: degraded
    # evidence must never rescue a losing bot from being switched off.
    spent_bets = w2 if trigger_window == "2w_fallback" else w1
    max_internal_gap_h, trailing_gap_h = window_gap_profile(spent_bets, now_ts)
    gap_bound_h = FIRE_STALE_MAX_AGE_S / 3600.0
    _gaps = [g for g in (max_internal_gap_h, trailing_gap_h) if g is not None]
    max_window_gap_h = max(_gaps) if _gaps else None
    # FAIL CLOSED: no measurable gap means no evidence that the window is
    # continuous, and this is a fail-safe gate. Defaulting an unmeasurable
    # window to "fine" would be the gate declining to do its job in exactly
    # the case it cannot see.
    evidence_gap_ok = (max_window_gap_h is not None
                       and max_window_gap_h <= gap_bound_h)
    # MED-A1: an unreadable config means the stream fell back to code
    # defaults, so it no longer describes the deployed bot. That is
    # degraded evidence: it may not ENABLE. It deliberately does NOT gate
    # the negative path below -- a protective disable on default-config
    # evidence is the same disable this monitor made for months, and
    # withholding it would be the unsafe direction.
    _, config_error = deployed_strategy()
    config_ok = config_error is None
    positive_evidence_ok = (positive_evidence_base and evidence_gap_ok
                            and config_ok)
    if not evidence_gap_ok and fire_fresh:
        print(f"!!! evidence GAP in the spent {trigger_window} window: "
              f"largest drought {max_window_gap_h:.1f}h > {gap_bound_h:.0f}h "
              "— positive actions blocked", flush=True)

    latest100 = bets[-100:]
    wr100 = float(np.mean([b["win"] for b in latest100])) if len(latest100) >= 50 else None

    # ---- trigger evaluation (positive on the spent window; negative 1w) ----
    # Šidák over the two computed windows is still REPORTED (informational);
    # the positive trigger is the raw single-test p per the user's decision.
    raw_best_p = min([p for p in (p2.get("p_upper"), p1.get("p_upper")) if p is not None],
                     default=1.0)
    sidak_p = 1 - (1 - raw_best_p) ** 2

    pos_trigger = evaluate_positive(pos_stats, pos_bt)

    # weak week: judged on the SPENT positive window (see weak_week).
    weak_this_week = weak_week(trigger_window, pos_stats)
    # Book the week: first run of the ISO week advances from last week's
    # counter; a same-week re-run RECOMPUTES from the persisted baseline
    # (overwrite — never freeze, never double-advance). Blind weeks keep
    # the stored value untouched (stale data says nothing about THIS
    # week; it still feeds the negative trigger below). Persistence is
    # additionally gated on --apply — dry runs preview, never persist.
    consec = int(st.get("consecutive_weak", 0))
    baseline = consec
    if evidence_ok:
        baseline, consec = book_weak_week(
            st, same_week_rerun=same_week_rerun, weak=weak_this_week)
    neg_wr_leg = negative_wr_leg(trigger_window, pos_stats)
    neg_trigger = bool(neg_wr_leg or consec >= NEG_CONSECUTIVE_WEAK)

    state = read_bot_state()
    pause_path = REPO / "var" / "live" / "pause_state.json"
    in_cooldown = False
    try:
        if pause_path.exists():
            in_cooldown = bool(json.loads(
                pause_path.read_text(encoding="utf-8")).get("paused", False))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        in_cooldown = False

    # ---- decide action ----
    # Fail-safe asymmetry on blind weeks (failed sync / stale data): the
    # protective disable may act on last-synced data, but a positive
    # trigger on stale evidence must never enable/release.
    action, reason, acted = "none", "", ""
    if not state["available"]:
        # systemctl itself did not answer: NO action is trustworthy (a
        # "disabled" read here is just the error string). Scream, act next
        # week — do not let a wedged systemd read as "already safe".
        action = "systemctl_UNAVAILABLE"
        reason = f"systemctl did not respond ({state['active']}) — no action possible"
    elif (streak + (0 if evidence_ok else 1)) >= SYNC_FAIL_DISABLE_STREAK \
            and (state["is_enabled"] or state["is_running"]):
        # streak counts COMPLETED fully-blind weeks; the current blind
        # attempt adds one so the 3rd consecutive blind week disables on
        # its Sunday, not a week later.
        reason = (f"FLYING BLIND: {streak} completed blind weeks + current "
                  "blind attempt — protective disable")
        if args.apply:
            action = "disable"
            acted = do_disable()
        else:
            action = "disable_DRYRUN"
    elif neg_trigger and (state["is_enabled"] or state["is_running"]):
        # is_running covers a running-but-disabled unit (manual start
        # without enable): do_disable() stops it either way.
        neg_desc = (_window_desc("2w(fallback SPENT)", p2)
                    if trigger_window == "2w_fallback"
                    else _window_desc("1w", p1))
        reason = (f"NEGATIVE: {neg_desc} (WR<{NEG_WR_FLOOR}: {neg_wr_leg}) "
                  f"or consecutive_weak={consec}>={NEG_CONSECUTIVE_WEAK}")
        if args.apply:
            action = "disable"
            acted = do_disable()
        else:
            action = "disable_DRYRUN"
    elif pos_trigger and not state["is_enabled"]:
        reason = (f"POSITIVE ({trigger_window}): WR={pos_stats.get('wr')}>"
                  f"{BREAKEVEN_WR}, p={pos_stats.get('p_upper')}<{POS_RAW_P}, "
                  f"n={pos_stats.get('n')}>={POS_MIN_FIRES}, "
                  f"btPnL={pos_bt.get('net_pnl_bnb')}>0")
        if not positive_evidence_ok:
            # PRECEDENCE IS LOAD-BEARING, not stylistic. The spent window
            # always ends at the newest fire, so its trailing gap IS
            # `now - newest_fire` -- the quantity fire_evidence_fresh
            # tests. evidence_gap_ok therefore IMPLIES fire_fresh, and
            # checking the gap rule first would make
            # *_BLOCKED_frozen_window unreachable. Order: stale -> frozen
            # -> gap, weakest precondition first.
            assert not (evidence_gap_ok and not fire_fresh), (
                "gap rule must imply freshness; precedence below relies on it")
            if not config_ok:
                action = "enable_BLOCKED_config_unreadable"
                reason += (f" — config.toml unreadable ({config_error}); the "
                           "fire stream fell back to CODE DEFAULTS and does "
                           "not describe the deployed bot; refusing to enable")
            elif not evidence_ok:
                action = "enable_BLOCKED_stale_evidence"
                reason += " — sync failed or data stale; refusing to enable"
            elif not fire_fresh:
                action = "enable_BLOCKED_frozen_window"
                reason += (f" — newest fire {fire_age_h:.1f}h old: this window "
                           "is FROZEN evidence, not this week's; refusing to "
                           "enable")
            else:
                action = "enable_BLOCKED_evidence_gap"
                reason += (f" — {max_window_gap_h:.1f}h drought inside the "
                           f"spent {trigger_window} window (> {gap_bound_h:.0f}h): "
                           "the fires are recent but not CONTINUOUS; refusing "
                           "to enable")
        elif args.apply:
            action = "enable"
            # One-shot re-enable (2026-07-17): if the bot went down mid-
            # suspension, write the override flag BEFORE starting it so the
            # first paused round releases (unpause + peak reseed + shadow
            # clear) instead of resuming a months-stale `bleeding` ledger
            # that would extend until a second positive Sunday.
            flag = None
            if in_cooldown:
                flag = write_override_flag(week=week, reason=reason,
                                           stats=pos_stats,
                                           trigger_window=trigger_window)
                acted = f"wrote {flag}; "
            ok, msg = do_enable()
            acted += msg
            if not ok:
                # A failed enable must not leave an 8-day release grenade:
                # any later manual `systemctl start` would consume the flag
                # and bet without an enable decision.
                action = "enable_FAILED"
                if flag is not None:
                    try:
                        flag.unlink()
                        acted += " (override flag removed)"
                    except OSError:
                        acted += " (override flag REMOVAL FAILED — delete var/live/cooldown_override.json manually)"
        else:
            action = "enable_DRYRUN"
    elif pos_trigger and state["is_enabled"] and in_cooldown:
        # Bot is enabled but breaker-suspended: release via the override
        # flag, which the pipeline consumes on its next paused round
        # (ignores extend-while-bleeding by design).
        reason = f"POSITIVE ({trigger_window}) while breaker-suspended -> override flag"
        if not positive_evidence_ok:
            if not config_ok:
                action = "cooldown_override_BLOCKED_config_unreadable"
                reason += (f" — config.toml unreadable ({config_error}); the "
                           "fire stream fell back to CODE DEFAULTS and does "
                           "not describe the deployed bot; refusing to release")
            elif not evidence_ok:
                action = "cooldown_override_BLOCKED_stale_evidence"
                reason += " — sync failed or data stale; refusing to release"
            elif not fire_fresh:
                action = "cooldown_override_BLOCKED_frozen_window"
                reason += (f" — newest fire {fire_age_h:.1f}h old: this window "
                           "is FROZEN evidence, not this week's; refusing to "
                           "release")
            else:
                action = "cooldown_override_BLOCKED_evidence_gap"
                reason += (f" — {max_window_gap_h:.1f}h drought inside the "
                           f"spent {trigger_window} window (> {gap_bound_h:.0f}h): "
                           "the fires are recent but not CONTINUOUS; refusing "
                           "to release")
        elif args.apply:
            action = "cooldown_override"
            flag = write_override_flag(week=week, reason=reason,
                                       stats=pos_stats,
                                       trigger_window=trigger_window)
            acted = f"wrote {flag}"
        else:
            action = "cooldown_override_DRYRUN"
    elif state["is_enabled"] and not state["is_running"]:
        # Reconcile enabled-but-dead (start-limit-hit residue, manual stop
        # without disable): weekly restart + alert — otherwise a dead bot
        # reads as healthy in every summary for months. Operators who WANT
        # it stopped must disable it (that is what enabled means here).
        reason = "unit enabled but not running — starting it"
        if args.apply:
            action = "restart_dead_unit"
            rc, out = _systemctl("start", LIVE_UNIT)
            acted = f"start rc={rc}: {out}"
            if rc != 0:
                action = "restart_dead_unit_FAILED"
        else:
            action = "restart_dead_unit_DRYRUN"

    # Consecutive Sundays whose positive action was blocked by the
    # evidence-gap rule. A slow strangulation otherwise looks exactly like
    # a quiet run of ordinary weeks -- the blind-spot class the bound's own
    # comment says it is guarding against.
    # Booked through the SAME overwrite guard as consecutive_weak: an
    # applied same-week re-run recomputes from the persisted baseline
    # instead of advancing again.
    # MED-A3(b). The fingerprint was written to decision.json and NOWHERE
    # ELSE: not state.json, not the history entry, never compared week over
    # week. So consecutive_weak advanced across a config discontinuity
    # blind, mitigated only by a source comment asking a human to notice --
    # the same shape as the defect this whole change exists to fix.
    #
    # RECORD AND ALERT, DO NOT AUTO-RESET. Resetting consecutive_weak on a
    # config change would be WORSE than the status quo: it would delay the
    # protective disable, turning a visible discontinuity into a silent
    # weakening of the path that stops a losing bot. The bound on the harm
    # of not resetting is small and worth stating -- one config change can
    # contribute at most ONE spurious weak week, and three are needed to
    # disable.
    fingerprint = strategy_fingerprint()
    prev_fingerprint = st.get("strategy_fingerprint")
    config_changed = (prev_fingerprint is not None
                      and prev_fingerprint != fingerprint)

    _gap_blocked = action.endswith("_BLOCKED_evidence_gap")
    gap_baseline, evidence_gap_streak = book_streak(
        st, "evidence_gap_streak", same_week_rerun=same_week_rerun,
        hit=_gap_blocked)

    decision = dict(
        week=week, run_at_utc=time.strftime("%Y-%m-%d %H:%M", time.gmtime()),
        data_newest_lock=time.strftime(
            "%Y-%m-%d %H:%M", time.gmtime(newest_round_lock)),
        newest_fire_lock=time.strftime("%Y-%m-%d %H:%M", time.gmtime(max_lock)),
        window_1w=dict(epochs=list(e1), **p1, backtest=bt,
                       fire_composition=comp1,
                       n_vs_backtest=_N_VS_BACKTEST_NOTE),
        window_2w=dict(epochs=list(e2), **p2, backtest=bt2,
                       fire_composition=comp2,
                       n_vs_backtest=_N_VS_BACKTEST_NOTE), latest100_wr=wr100,
        # WHICH BOT this week describes. A change here makes n / wr /
        # p_upper / btPnL discontinuous against previous weeks -- treat it
        # as an epoch boundary rather than a trend.
        strategy_fingerprint=fingerprint,
        prev_strategy_fingerprint=prev_fingerprint,
        config_changed=config_changed,
        triggers=dict(positive=pos_trigger, trigger_window=trigger_window,
                      negative=neg_trigger,
                      neg_wr_leg=neg_wr_leg, weak_this_week=weak_this_week,
                      raw_best_p=round(raw_best_p, 5),
                      sidak_p_informational=round(sidak_p, 5),
                      consecutive_weak=consec),
        bot_state=state, in_cooldown=in_cooldown, sync_ok=sync_ok,
        data_fresh=data_fresh, fire_fresh=fire_fresh,
        newest_fire_age_h=round(fire_age_h, 1),
        fire_stale_max_age_h=FIRE_STALE_MAX_AGE_S // 3600,
        max_internal_gap_h=(round(max_internal_gap_h, 1)
                            if max_internal_gap_h is not None else None),
        # The value the gate ACTUALLY tests. Without it a trailing-driven
        # block writes max_internal_gap_h=8.0 beside evidence_gap_ok=false,
        # which is self-contradictory to any future replay.
        max_window_gap_h=(round(max_window_gap_h, 1)
                          if max_window_gap_h is not None else None),
        trailing_gap_h=round(trailing_gap_h, 1),
        gap_bound_h=round(gap_bound_h, 1),
        evidence_gap_ok=evidence_gap_ok,
        evidence_gap_streak=evidence_gap_streak,
        fire_gap_p99_h=gap_p99_h,
        sync_fail_streak=streak,
        retry_mode=retry_mode, retry_attempts=attempts_so_far,
        completed_blind_week=completed_blind_week,
        action=action, reason=reason, acted=acted,
        applied=args.apply)
    (out_dir / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")

    # ---- persist state (booked once per ISO week; only real, fresh,
    # applied runs count — dry runs preview without persisting, blind
    # weeks retry). A same-week applied re-run OVERWRITES the week's
    # booking and history entry (recomputed from the baseline above). ----
    if args.apply and evidence_ok:
        st["consecutive_weak"] = consec
        st["consecutive_weak_baseline"] = baseline
        st["evidence_gap_streak"] = evidence_gap_streak
        st["evidence_gap_streak_baseline"] = gap_baseline
        st["last_week"] = week
        st["last_action"] = action
        # Persisted so NEXT week can detect the discontinuity, and stamped
        # on the history entry so a later reader can see which bot each
        # booked week described without re-reading old artifacts.
        st["strategy_fingerprint"] = fingerprint
        hist = st.setdefault("history", [])
        entry = dict(week=week, action=action, wr_1w=p1.get("wr"),
                     p_1w=p1.get("p_upper"), trigger_window=trigger_window,
                     sidak=round(sidak_p, 4),
                     strategy_fingerprint=fingerprint,
                     config_changed=config_changed)
        if same_week_rerun and hist and hist[-1].get("week") == last_week:
            hist[-1] = entry
        else:
            hist.append(entry)
        save_state(st)

    # ---- alert (fires on EVERY completed run — the dead-man's switch) ----
    head = f"[weekly-monitor {week}] action={action}"
    if not args.apply:
        head = f"[DRY RUN] {head}"
    if retry_mode and evidence_ok:
        head += f" — recovered after {attempts_so_far} failed attempt(s)"
    if not evidence_ok:
        what = "SYNC FAILED" if not sync_ok else "DATA STALE"
        head = (f"⚠️ {what} — stale-data evaluation; will retry daily "
                f"until Sunday\n{head}")
    if not fire_fresh:
        # Loudest banner available: the numbers below describe bets that
        # already happened, not the week being reported on.
        head = (f"⚠️ FROZEN WINDOW — newest fire is {fire_age_h:.1f}h old "
                f"(> {FIRE_STALE_MAX_AGE_S // 3600}h): the stats below re-score "
                f"PAST bets and say nothing about this week. Positive actions "
                f"blocked; the bot has placed no bet since "
                f"{time.strftime('%Y-%m-%d %H:%M', time.gmtime(max_lock))}Z.\n"
                f"{head}")
    if not evidence_gap_ok and fire_fresh:
        # H1: without this the suppressed enable fell through to the
        # plain `else` below and read as a routine no-op week -- a
        # silenced live-money action looking like nothing happened.
        head = (f"⚠️ EVIDENCE GAP — a {max_window_gap_h:.1f}h drought "
                f"inside the spent {trigger_window} window "
                f"(> {gap_bound_h:.0f}h). The fires are recent but NOT "
                f"continuous, so the stats below rest on a window with a "
                f"hole in it. Positive actions blocked; the negative path "
                f"is unaffected.\n{head}")
    if config_changed:
        changed_keys = sorted(
            k for k in set(fingerprint) | set(prev_fingerprint or {})
            if (prev_fingerprint or {}).get(k) != fingerprint.get(k))
        detail = ", ".join(
            f"{k}: {(prev_fingerprint or {}).get(k)} -> {fingerprint.get(k)}"
            for k in changed_keys)
        head = (f"⚠️ CONFIG CHANGED — n/WR/p_upper are NOT comparable to "
                f"previous weeks ({detail}). consecutive_weak is "
                f"deliberately NOT reset: resetting would delay the "
                f"protective disable. Treat this week as an epoch boundary "
                f"when reading the series.\n{head}")
    if evidence_gap_streak >= 2:
        # The counter tracks SUPPRESSED ENABLES, not sparse evidence: it
        # only advances on a *_BLOCKED_evidence_gap action, and any week
        # whose positive trigger did not fire resets it to 0. Saying "the
        # fire stream has been too sparse" overclaimed — a run of quiet
        # weeks does not raise this, and a quiet week interleaved between
        # two blocked ones lowers it.
        head = (f"⚠️ EVIDENCE GAP x{evidence_gap_streak} CONSECUTIVE "
                f"SUNDAYS — on {evidence_gap_streak} Sundays running a "
                f"positive trigger FIRED and was suppressed by the gap "
                f"rule (the counter follows suppressed enables, and resets "
                f"on any week whose triggers did not fire — so consecutive "
                f"here means consecutive SUPPRESSIONS). Check the fire "
                f"RATE, not the bound.\n{head}")
    if completed_blind_week:
        head += "\n(previous week ended fully blind — Sunday and every retry failed)"
    if trigger_window == "2w_fallback":
        w2_desc = _window_desc("2w(fallback SPENT)", p2, bt2, comp2)
    else:
        w2_desc = _window_desc("2w(info)", p2, None, comp2)
    body = (f"{_window_desc('1w', p1, bt, comp1)}; {w2_desc}; "
            f"fire_age={fire_age_h:.1f}h fresh={fire_fresh} "
            f"max_gap={'-' if max_window_gap_h is None else format(max_window_gap_h, '.1f')}h"
            f"/{gap_bound_h:.0f}h "
            f"gap_p99={gap_p99_h}h/{FIRE_STALE_MAX_AGE_S // 3600}h; "
            f"neg={neg_trigger} weak={weak_this_week} consec_weak={consec} "
            f"blind_streak={streak} gap_streak={evidence_gap_streak}; "
            f"enabled={state.get('is_enabled')} "
            f"running={state.get('is_running')} in_cooldown={in_cooldown}")
    # `head` may already open with its own ⚠️ banner (stale data, frozen
    # window, evidence gap, escalation). The severity prefixes below then
    # rendered "⚠️ ⚠️ ..." in Discord.
    warn = "" if head.startswith("⚠️") else "⚠️ "
    if retry_mode and not evidence_ok and action == "none":
        # Daily retry still blind, nothing actionable: one line, no spam.
        delivered = discord(
            f"⚠️ [weekly-monitor retry] week {week} still blind (attempt "
            f"{attempts_so_far + 1}: "
            f"{'sync failed' if not sync_ok else 'data stale'}) — retrying "
            "daily; next full run Sunday")
    elif action in ("enable", "disable", "cooldown_override", "restart_dead_unit"):
        delivered = discord(f"{warn}{head} — STATE CHANGED\n{reason}\n{acted}\n{body}")
    elif "_BLOCKED_" in action:
        # A suppressed live-money action is never a routine no-op.
        delivered = discord(
            f"{warn}{head} — POSITIVE ACTION SUPPRESSED\n{reason}\n{body}")
    elif action.endswith("_FAILED") or action == "systemctl_UNAVAILABLE":
        delivered = discord(f"❌ {head} — ACTION FAILED / DEGRADED\n{reason}\n{acted}\n{body}")
    else:
        delivered = discord(f"{head}\n{reason or 'neutral / no-op'}\n{body}")

    print("\n=== WEEKLY MONITOR DECISION ===")
    print(head); print(reason or "neutral / no-op"); print(body)
    print(f"(applied={args.apply})")
    print(f"artifacts -> {out_dir}")
    if not delivered:
        # Evaluation completed but the alert did not land: exit 3 so the
        # wrapper attempts its curl fallback; if Discord itself is down,
        # cron.log carries the explanation and next week retries.
        print("!!! Discord delivery FAILED (rc=3)", file=sys.stderr)
        return 3
    return 0


def main() -> int:
    """Crash containment: any unhandled exception still produces a Discord
    alert (the walk-away contract: a silent Sunday can only mean the box,
    cron, or webhook is dead — never a swallowed error)."""
    try:
        return _main()
    except Exception:
        import traceback
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        discord("❌ [weekly-monitor] CRASHED mid-run — an action may already "
                "have been taken (check the decision artifact + systemctl "
                f"state); will retry next Sunday\n```{tb[-1200:]}```")
        return 1


if __name__ == "__main__":
    sys.exit(main())
