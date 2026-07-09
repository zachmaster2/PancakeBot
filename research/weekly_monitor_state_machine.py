"""Weekly state-machine monitor — sync, evaluate, and (safely) toggle the bot.

Runs the wait-and-monitor protocol on a weekly cadence and drives the live
bot's enable/disable state through a deliberately ASYMMETRIC, fail-safe
state machine:

  * AUTO-DISABLE (protect capital) is fully autonomous — the negative
    trigger stops+disables the live unit with `--apply` alone.
  * AUTO-ENABLE (2026-07-09 user decision): the positive trigger acts
    under `--apply` alone — raw p<0.10 on the trailing-1-week window,
    single test, no multiple-comparison gate. If the bot is enabled but
    breaker-suspended, the positive trigger instead writes the cooldown
    override flag that the pipeline consumes to release the suspension.

Default is DRY-RUN: it computes + reports the decision and writes the
artifact but touches NOTHING. Pass `--apply` to let it act (systemd
enable/disable + override-flag writes).

Steps each run:
  1. sync (`run.py --sync`) unless --no-sync
  2. canonical gate flat-stake bet stream (risk-free) + trailing 2w/1w windows
  3. standard backtest (risk breaker OFF) on the 1w window @5BNB -> gas-inclusive PnL
  4. positive/negative trigger evaluation (+ Šidák correction, + consecutive-weak counter)
  5. read live bot state (systemctl), decide action, act iff permitted
  6. Discord alert (state change or weekly summary) + artifact + persistent state

Idempotent: one artifact dir per ISO week; a second run in the same week
re-computes but does not double-advance the consecutive-weak counter or
re-fire a state change already recorded this week.

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
    rounds = [r for r in ipr._load_all_rounds(use_extended_data=False)
              if r.position in ("Bull", "Bear")]
    rounds.sort(key=lambda r: r.epoch)
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
    return bets


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
    r = subprocess.run([sys.executable, str(REPO / "run.py"), "--backtest",
                        "--config", str(cfg)], cwd=REPO, capture_output=True, text=True)
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


def do_enable() -> str:
    rc, out = _systemctl("enable", "--now", LIVE_UNIT)
    return f"enable --now rc={rc}: {out}"


def do_disable() -> str:
    rc1, o1 = _systemctl("disable", LIVE_UNIT)
    rc2, o2 = _systemctl("stop", LIVE_UNIT)
    return f"disable rc={rc1} / stop rc={rc2}: {o1} {o2}"


# --------------------------------------------------------------------------
# Discord (best-effort)
# --------------------------------------------------------------------------

def discord(msg: str) -> None:
    url = os.environ.get("PANCAKEBOT_GENERAL_DISCORD_WEBHOOK_URL", "")
    if not url:
        return
    try:
        import requests
        requests.post(url, json={"content": msg[:1900]}, timeout=10)
    except Exception:
        pass


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


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="actually act on systemd (else dry-run)")
    ap.add_argument("--no-sync", action="store_true", help="skip run.py --sync")
    ap.add_argument("--iso-week", type=str, default=None, help="override week key (testing)")
    args = ap.parse_args()

    week = args.iso_week or time.strftime("%Y-%m-%d", time.gmtime())
    out_dir = ROOT / week
    out_dir.mkdir(parents=True, exist_ok=True)
    st = load_state()
    same_week_rerun = (st.get("last_week") == week)

    if not args.no_sync:
        print("--- sync ---", flush=True)
        subprocess.run([sys.executable, str(REPO / "run.py"), "--sync"], cwd=REPO)

    print("--- canonical bet stream ---", flush=True)
    bets = build_canonical_bets()
    max_lock = max(b["lock"] for b in bets)

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
    # advance the consecutive-weak counter only once per week
    consec = st.get("consecutive_weak", 0)
    if not same_week_rerun:
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
    action, reason, acted = "none", "", ""
    if neg_trigger and state["is_enabled"]:
        action = "disable"
        reason = (f"NEGATIVE: 1w WR={p1.get('wr')} (<{NEG_WR_1W}: {neg_wr_leg}) "
                  f"or consecutive_weak={consec}>={NEG_CONSECUTIVE_WEAK}")
        if args.apply:
            acted = do_disable()
    elif pos_trigger and not state["is_enabled"]:
        action = "enable"
        reason = (f"POSITIVE (1w): WR={p1.get('wr')}>{BREAKEVEN_WR}, "
                  f"p={p1.get('p_upper')}<{POS_RAW_P}, n={p1.get('n')}>="
                  f"{POS_MIN_FIRES}, btPnL={bt.get('net_pnl_bnb')}>0")
        if args.apply:
            acted = do_enable()
        else:
            action = "enable_DRYRUN"
    elif pos_trigger and state["is_enabled"] and in_cooldown:
        # Bot is enabled but breaker-suspended: release via the override
        # flag, which the pipeline consumes on its next paused round
        # (ignores extend-while-bleeding by design).
        action = "cooldown_override"
        reason = "POSITIVE (1w) while breaker-suspended -> override flag"
        if args.apply:
            flag = REPO / "var" / "live" / "cooldown_override.json"
            flag.parent.mkdir(parents=True, exist_ok=True)
            flag.write_text(json.dumps(dict(
                ts=time.time(), week=week,
                reason=reason,
                window_1w=dict(wr=p1.get("wr"), p_upper=p1.get("p_upper"),
                               n=p1.get("n")),
            ), indent=2), encoding="utf-8")
            acted = f"wrote {flag}"
        else:
            action = "cooldown_override_DRYRUN"

    decision = dict(
        week=week, run_at_utc=time.strftime("%Y-%m-%d %H:%M", time.gmtime()),
        data_newest_lock=time.strftime("%Y-%m-%d %H:%M", time.gmtime(max_lock)),
        window_1w=dict(epochs=list(e1), **p1, backtest=bt),
        window_2w=p2, latest100_wr=wr100,
        triggers=dict(positive=pos_trigger, negative=neg_trigger,
                      neg_wr_leg=neg_wr_leg, weak_this_week=weak_this_week,
                      raw_best_p=round(raw_best_p, 5),
                      sidak_p_informational=round(sidak_p, 5),
                      consecutive_weak=consec),
        bot_state=state, in_cooldown=in_cooldown,
        action=action, reason=reason, acted=acted,
        applied=args.apply)
    (out_dir / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")

    # ---- persist state (once per week) ----
    if not same_week_rerun:
        st["consecutive_weak"] = consec
        st["last_week"] = week
        st["last_action"] = action
        st.setdefault("history", []).append(
            dict(week=week, action=action, wr_1w=p1.get("wr"), p_1w=p1.get("p_upper"),
                 sidak=round(sidak_p, 4)))
        save_state(st)

    # ---- alert ----
    head = f"[weekly-monitor {week}] action={action}"
    body = (f"1w: n={p1.get('n')} WR={p1.get('wr')} p={p1.get('p_upper')} "
            f"btPnL={bt.get('net_pnl_bnb')}; 2w(info): WR={p2.get('wr')} "
            f"p={p2.get('p_upper')}; neg={neg_trigger} consec_weak={consec}; "
            f"enabled={state.get('is_enabled')} in_cooldown={in_cooldown}")
    if action in ("enable", "disable", "cooldown_override"):
        discord(f"⚠️ {head} — STATE CHANGED\n{reason}\n{acted}\n{body}")
    else:
        discord(f"{head}\n{reason or 'neutral / no-op'}\n{body}")

    print("\n=== WEEKLY MONITOR DECISION ===")
    print(head); print(reason or "neutral / no-op"); print(body)
    print(f"(applied={args.apply})")
    print(f"artifacts -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
