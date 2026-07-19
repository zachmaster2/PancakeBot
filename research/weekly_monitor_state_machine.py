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
    exit 0 without advancing the stores; the newest FIRE is deliberately
    not the yardstick, it lags days in normal signal droughts). Blind
    runs (either check failing)
    block enable/release, freeze the weekly counters, and alert loudly;
    the protective disable may still act on last-synced data.
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
    systemd — pure previews. State advances at most once per ISO week.
  * every completed run VERIFIES Discord delivery (HTTP < 400, retry);
    undelivered -> rc=3 so the cron wrapper curls a fallback. Any crash
    Discords a ❌ CRASHED alert with the traceback tail and exits
    nonzero. A Sunday with NO message therefore means the box, cron, or
    webhook itself is dead — nothing else fails silently.

Steps each run:
  1. sync (`run.py --sync`) unless --no-sync
  2. canonical gate flat-stake bet stream (risk-free) + trailing 2w/1w windows
  3. standard backtest (risk breaker OFF) on the 1w window @5BNB -> gas-inclusive PnL
  4. positive/negative trigger evaluation (+ Šidák correction, + consecutive-weak counter)
  5. read live bot state (systemctl), decide action, act iff permitted
  6. Discord alert (state change or weekly summary) + artifact + persistent state

Idempotent: artifact dirs are per-day, but weekly STATE advances once per
ISO week — a re-run any day of the same ISO week re-computes and re-reports
without double-advancing the consecutive-weak counter.

Triggers (pinned; 2026-07-09 redesign — 1-WEEK window only, per user decision):
  POSITIVE:  on the trailing-1w window: WR > BREAKEVEN(0.55) AND raw
             p_upper < 0.10 (single test) AND n_fires >= 10 AND the
             standard risk-off backtest net PnL (after gas) > 0.
             Action when bot DISABLED: enable + start (under --apply).
             Action when bot ENABLED and breaker-suspended: write the
             cooldown override flag (var/live/cooldown_override.json),
             which the pipeline consumes to release the suspension
             immediately (ignoring extend-while-bleeding).
  NEGATIVE:  trailing-1w WR < 0.45  OR  3 consecutive weekly runs weak
             (weak = 1w p_upper > 0.5, or insufficient fires n < 10).
             Action: disable + stop entirely.
  The 2w window + latest-100 WR + Šidák are still computed and reported
  (informational only — they no longer gate actions).

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
from pancakebot.config import load_strategy_config_from_dict  # noqa: E402
from pancakebot.constants import BNB_WEI, MAX_GAS_COST_BET_BNB  # noqa: E402
from pancakebot.pool_amounts import compute_pool_amounts_wei  # noqa: E402
from pancakebot.strategy.momentum_gate import MomentumGateConfig  # noqa: E402
from pancakebot.strategy.momentum_pipeline import MomentumOnlyPipeline  # noqa: E402

ROOT = REPO / "var" / "strategy_review" / "weekly_monitors"
STATE_PATH = ROOT / "state.json"
LIVE_UNIT = "pancakebot-live"
CUTOFF, LOOKBACKS, FEE = 2, (3, 7, 15), 0.03
BREAKEVEN_WR = 0.55
POS_RAW_P = 0.10           # raw permutation p_upper, single test (user decision)
POS_MIN_FIRES = 10         # 2026-07-09: halved window (2w->1w) -> halved floor
NEG_WR_1W = 0.45           # trailing-1w WR below this -> disable
NEG_CONSECUTIVE_WEAK = 3
NEG_WEAK_P = 0.5           # weak week: 1w p_upper above this (or n < POS_MIN_FIRES)
N_PERM = 10_000
SEED = 20260630


# --------------------------------------------------------------------------
# canonical gate flat-stake bet stream
# --------------------------------------------------------------------------

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
    sc = load_strategy_config_from_dict({})
    gate_cfg = MomentumGateConfig(
        enabled=True, bnb_symbol="BNB-USDT", btc_symbol="BTC-USDT",
        eth_symbol="ETH-USDT", sol_symbol="SOL-USDT", kline_cutoff_seconds=CUTOFF,
        mtf_lookbacks=sc.gate.mtf_lookbacks,
        mtf_min_return_threshold=sc.gate.mtf_min_return_threshold)
    pipe = MomentumOnlyPipeline(
        config=gate_cfg, strategy_config=sc, gate=None, kline_cutoff_seconds=CUTOFF,
        pool_cutoff_seconds=6, min_bet_amount_bnb=0.001, treasury_fee_fraction=FEE,
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


# --------------------------------------------------------------------------
# standard backtest (risk breaker OFF) on a window -> gas-inclusive net PnL
# --------------------------------------------------------------------------

BACKTEST_TIMEOUT_S = 1800   # a hung backtest must not eat the weekly slot
SYNC_TIMEOUT_S = 3600       # observed healthy sync ~14 min; 60 min = hung
FRESH_MAX_AGE_S = 36 * 3600  # newest closed ROUND older than this = stale
SYNC_FAIL_DISABLE_STREAK = 3  # blind weeks in a row before protective disable


def risk_off_backtest(epoch_start: int, epoch_end: int, out_dir: Path,
                      bankroll: float = 5.0) -> dict:
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
    cfg = out_dir / "risk_off_config.toml"
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


def write_override_flag(*, week: str, reason: str, p1: dict) -> Path:
    """Write the cooldown-override flag the pipeline consumes on its next
    paused round (fresh <= 8 days; `_consume_override_flag`). Atomic
    tmp+rename: the running bot's reader DELETES the flag on a parse error,
    so a torn write would silently discard the release."""
    flag = REPO / "var" / "live" / "cooldown_override.json"
    flag.parent.mkdir(parents=True, exist_ok=True)
    tmp = flag.with_suffix(flag.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(
        ts=time.time(), week=week, reason=reason,
        window_1w=dict(wr=p1.get("wr"), p_upper=p1.get("p_upper"),
                       n=p1.get("n")),
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
    data_fresh = (time.time() - newest_round_lock) <= FRESH_MAX_AGE_S
    evidence_ok = sync_ok and data_fresh
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
    e1 = (min(b["epoch"] for b in w1), max(b["epoch"] for b in w1)) if w1 else (0, 0)
    p2, p1 = perm(w2), perm(w1)

    print("--- risk-off standard backtest (1w @5BNB) ---", flush=True)
    bt = risk_off_backtest(e1[0], e1[1], out_dir, bankroll=5.0) if w1 else {}

    latest100 = bets[-100:]
    wr100 = float(np.mean([b["win"] for b in latest100])) if len(latest100) >= 50 else None

    # ---- trigger evaluation (2026-07-09: 1-WEEK window governs) ----
    # Šidák over the two computed windows is still REPORTED (informational);
    # the positive trigger is the raw single-test p per the user's decision.
    raw_best_p = min([p for p in (p2.get("p_upper"), p1.get("p_upper")) if p is not None],
                     default=1.0)
    sidak_p = 1 - (1 - raw_best_p) ** 2

    pos_trigger = bool(
        not p1.get("insufficient") and p1.get("wr", 0) > BREAKEVEN_WR
        and p1.get("p_upper", 1) < POS_RAW_P and p1.get("n", 0) >= POS_MIN_FIRES
        and bt.get("net_pnl_bnb", -1) > 0)

    # weak week: 1w p_upper above the bar, or not enough fires to know.
    weak_this_week = bool(
        p1.get("insufficient")
        or (p1.get("p_upper") is not None and p1["p_upper"] > NEG_WEAK_P))
    # advance the consecutive-weak counter only once per week, and never on
    # a blind week (stale data says nothing about THIS week; the stored
    # value still feeds the negative trigger below). Persistence is
    # additionally gated on --apply — dry runs preview, never advance.
    consec = st.get("consecutive_weak", 0)
    if not same_week_rerun and evidence_ok:
        consec = consec + 1 if weak_this_week else 0
    neg_wr_leg = bool(
        not p1.get("insufficient") and p1.get("wr") is not None
        and p1["wr"] < NEG_WR_1W)
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
        reason = (f"NEGATIVE: 1w WR={p1.get('wr')} (<{NEG_WR_1W}: {neg_wr_leg}) "
                  f"or consecutive_weak={consec}>={NEG_CONSECUTIVE_WEAK}")
        if args.apply:
            action = "disable"
            acted = do_disable()
        else:
            action = "disable_DRYRUN"
    elif pos_trigger and not state["is_enabled"]:
        reason = (f"POSITIVE (1w): WR={p1.get('wr')}>{BREAKEVEN_WR}, "
                  f"p={p1.get('p_upper')}<{POS_RAW_P}, n={p1.get('n')}>="
                  f"{POS_MIN_FIRES}, btPnL={bt.get('net_pnl_bnb')}>0")
        if not evidence_ok:
            action = "enable_BLOCKED_stale_evidence"
            reason += " — sync failed or data stale; refusing to enable"
        elif args.apply:
            action = "enable"
            # One-shot re-enable (2026-07-17): if the bot went down mid-
            # suspension, write the override flag BEFORE starting it so the
            # first paused round releases (unpause + peak reseed + shadow
            # clear) instead of resuming a months-stale `bleeding` ledger
            # that would extend until a second positive Sunday.
            flag = None
            if in_cooldown:
                flag = write_override_flag(week=week, reason=reason, p1=p1)
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
        reason = "POSITIVE (1w) while breaker-suspended -> override flag"
        if not evidence_ok:
            action = "cooldown_override_BLOCKED_stale_evidence"
            reason += " — sync failed or data stale; refusing to release"
        elif args.apply:
            action = "cooldown_override"
            acted = f"wrote {write_override_flag(week=week, reason=reason, p1=p1)}"
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

    decision = dict(
        week=week, run_at_utc=time.strftime("%Y-%m-%d %H:%M", time.gmtime()),
        data_newest_lock=time.strftime(
            "%Y-%m-%d %H:%M", time.gmtime(newest_round_lock)),
        newest_fire_lock=time.strftime("%Y-%m-%d %H:%M", time.gmtime(max_lock)),
        window_1w=dict(epochs=list(e1), **p1, backtest=bt),
        window_2w=p2, latest100_wr=wr100,
        triggers=dict(positive=pos_trigger, negative=neg_trigger,
                      neg_wr_leg=neg_wr_leg, weak_this_week=weak_this_week,
                      raw_best_p=round(raw_best_p, 5),
                      sidak_p_informational=round(sidak_p, 5),
                      consecutive_weak=consec),
        bot_state=state, in_cooldown=in_cooldown, sync_ok=sync_ok,
        data_fresh=data_fresh, sync_fail_streak=streak,
        retry_mode=retry_mode, retry_attempts=attempts_so_far,
        completed_blind_week=completed_blind_week,
        action=action, reason=reason, acted=acted,
        applied=args.apply)
    (out_dir / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")

    # ---- persist state (once per ISO week; only real, fresh, applied
    # runs count — dry runs preview without advancing, blind weeks retry) --
    if args.apply and evidence_ok and not same_week_rerun:
        st["consecutive_weak"] = consec
        st["last_week"] = week
        st["last_action"] = action
        st.setdefault("history", []).append(
            dict(week=week, action=action, wr_1w=p1.get("wr"), p_1w=p1.get("p_upper"),
                 sidak=round(sidak_p, 4)))
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
    if completed_blind_week:
        head += "\n(previous week ended fully blind — Sunday and every retry failed)"
    body = (f"1w: n={p1.get('n')} WR={p1.get('wr')} p={p1.get('p_upper')} "
            f"btPnL={bt.get('net_pnl_bnb')}; 2w(info): WR={p2.get('wr')} "
            f"p={p2.get('p_upper')}; neg={neg_trigger} consec_weak={consec} "
            f"blind_streak={streak}; enabled={state.get('is_enabled')} "
            f"running={state.get('is_running')} in_cooldown={in_cooldown}")
    if retry_mode and not evidence_ok and action == "none":
        # Daily retry still blind, nothing actionable: one line, no spam.
        delivered = discord(
            f"⚠️ [weekly-monitor retry] week {week} still blind (attempt "
            f"{attempts_so_far + 1}: "
            f"{'sync failed' if not sync_ok else 'data stale'}) — retrying "
            "daily; next full run Sunday")
    elif action in ("enable", "disable", "cooldown_override", "restart_dead_unit"):
        delivered = discord(f"⚠️ {head} — STATE CHANGED\n{reason}\n{acted}\n{body}")
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
