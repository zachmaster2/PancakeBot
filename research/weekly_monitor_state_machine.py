"""Weekly state-machine monitor — sync, evaluate, and (safely) toggle the bot.

Runs the wait-and-monitor protocol on a weekly cadence and drives the live
bot's enable/disable state through a deliberately ASYMMETRIC, fail-safe
state machine:

  * AUTO-DISABLE (protect capital) is fully autonomous — the negative
    trigger stops+disables the live unit with `--apply` alone.
  * AUTO-ENABLE (risk capital) is gated: the positive trigger must clear
    the MULTIPLE-COMPARISON-CORRECTED bar (not just raw p), AND the run
    must be explicitly `--arm`ed, AND `--apply` given. Rationale: the
    2026-06-30 gauntlet showed the loose raw-p<0.10 trigger fires on the
    exact noise it ruled out (Šidák 0.54). Automating protection is safe;
    automating real-money entry on an uncorrected p-value is not.

Default is DRY-RUN: it computes + reports the decision and writes the
artifact but touches NOTHING. Pass `--apply` to let it act on systemd,
`--arm` to additionally permit an enable.

Steps each run:
  1. sync (`run.py --sync`) unless --no-sync
  2. canonical gate flat-stake bet stream (risk-free) + trailing 2w/1w windows
  3. standard backtest (risk breaker OFF) on the 2w window @5BNB -> gas-inclusive PnL
  4. positive/negative trigger evaluation (+ Šidák correction, + consecutive-weak counter)
  5. read live bot state (systemctl), decide action, act iff permitted
  6. Discord alert (state change or weekly summary) + artifact + persistent state

Idempotent: one artifact dir per ISO week; a second run in the same week
re-computes but does not double-advance the consecutive-weak counter or
re-fire a state change already recorded this week.

Triggers (pinned):
  POSITIVE (loose, per dispatch):   WR > BREAKEVEN(0.55) AND raw p_upper < 0.10
                                    AND n_fires >= 20 AND standard-backtest
                                    net PnL (after gas) > 0, on the 2w window.
  POSITIVE ENABLE-GATE (strict):    additionally Šidák-adjusted p < 0.05 over
                                    the windows examined this run.
  NEGATIVE:  latest-100-fire WR < 0.45  OR  3 consecutive weekly runs with
             2w p_upper > 0.5.

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
POS_RAW_P = 0.10          # loose positive-trigger p bar
POS_ENABLE_SIDAK_P = 0.05  # strict enable-gate (corrected)
POS_MIN_FIRES = 20
NEG_WR_100 = 0.45
NEG_CONSECUTIVE_WEAK = 3
NEG_WEAK_P = 0.5
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
    ap.add_argument("--arm", action="store_true", help="permit an AUTO-ENABLE (still needs strict gate)")
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
    e2 = (min(b["epoch"] for b in w2), max(b["epoch"] for b in w2)) if w2 else (0, 0)
    p2, p1 = perm(w2), perm(w1)

    print("--- risk-off standard backtest (2w @5BNB) ---", flush=True)
    bt = risk_off_backtest(e2[0], e2[1], out_dir, bankroll=5.0) if w2 else {}

    latest100 = bets[-100:]
    wr100 = float(np.mean([b["win"] for b in latest100])) if len(latest100) >= 50 else None

    # ---- trigger evaluation ----
    n_windows_examined = 2
    raw_best_p = min([p for p in (p2.get("p_upper"), p1.get("p_upper")) if p is not None],
                     default=1.0)
    sidak_p = 1 - (1 - raw_best_p) ** n_windows_examined

    pos_loose = bool(
        not p2.get("insufficient") and p2.get("wr", 0) > BREAKEVEN_WR
        and p2.get("p_upper", 1) < POS_RAW_P and p2.get("n", 0) >= POS_MIN_FIRES
        and bt.get("net_pnl_bnb", -1) > 0)
    pos_strict_gate = bool(pos_loose and sidak_p < POS_ENABLE_SIDAK_P)

    weak_this_week = bool(p2.get("p_upper", 0) is not None and p2.get("p_upper", 0) > NEG_WEAK_P)
    # advance the consecutive-weak counter only once per week
    consec = st.get("consecutive_weak", 0)
    if not same_week_rerun:
        consec = consec + 1 if weak_this_week else 0
    neg_trigger = bool((wr100 is not None and wr100 < NEG_WR_100)
                       or consec >= NEG_CONSECUTIVE_WEAK)

    state = read_bot_state()

    # ---- decide action ----
    action, reason, acted = "none", "", ""
    if neg_trigger and state["is_enabled"]:
        action = "disable"
        reason = (f"NEGATIVE: latest100 WR={wr100} (<{NEG_WR_100}) "
                  f"or consecutive_weak={consec}>={NEG_CONSECUTIVE_WEAK}")
        if args.apply:
            acted = do_disable()
    elif pos_loose and not state["is_enabled"]:
        if pos_strict_gate and args.arm and args.apply:
            action = "enable"
            reason = f"POSITIVE passed strict gate (Šidák {sidak_p:.3f}<{POS_ENABLE_SIDAK_P}) + armed"
            acted = do_enable()
        else:
            action = "enable_BLOCKED"
            blockers = []
            if not pos_strict_gate:
                blockers.append(f"strict gate NOT met (Šidák {sidak_p:.3f} >= {POS_ENABLE_SIDAK_P})")
            if not args.arm:
                blockers.append("not --arm'ed")
            if not args.apply:
                blockers.append("dry-run (no --apply)")
            reason = ("POSITIVE loose trigger met but ENABLE withheld: "
                      + "; ".join(blockers))

    decision = dict(
        week=week, run_at_utc=time.strftime("%Y-%m-%d %H:%M", time.gmtime()),
        data_newest_lock=time.strftime("%Y-%m-%d %H:%M", time.gmtime(max_lock)),
        window_2w=dict(epochs=list(e2), **p2, backtest=bt),
        window_1w=p1, latest100_wr=wr100,
        triggers=dict(positive_loose=pos_loose, positive_strict_gate=pos_strict_gate,
                      negative=neg_trigger, raw_best_p=round(raw_best_p, 5),
                      sidak_p=round(sidak_p, 5), consecutive_weak=consec),
        bot_state=state, action=action, reason=reason, acted=acted,
        applied=args.apply, armed=args.arm)
    (out_dir / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")

    # ---- persist state (once per week) ----
    if not same_week_rerun:
        st["consecutive_weak"] = consec
        st["last_week"] = week
        st["last_action"] = action
        st.setdefault("history", []).append(
            dict(week=week, action=action, wr_2w=p2.get("wr"), p_2w=p2.get("p_upper"),
                 sidak=round(sidak_p, 4)))
        save_state(st)

    # ---- alert ----
    head = f"[weekly-monitor {week}] action={action}"
    body = (f"2w: n={p2.get('n')} WR={p2.get('wr')} p={p2.get('p_upper')} "
            f"Šidák={sidak_p:.3f} btPnL={bt.get('net_pnl_bnb')}; "
            f"neg={neg_trigger} consec_weak={consec}; bot enabled={state.get('is_enabled')}")
    if action in ("enable", "disable"):
        discord(f"⚠️ {head} — STATE CHANGED\n{reason}\n{acted}\n{body}")
    else:
        discord(f"{head}\n{reason or 'neutral / no-op'}\n{body}")

    print("\n=== WEEKLY MONITOR DECISION ===")
    print(head); print(reason or "neutral / no-op"); print(body)
    print(f"(applied={args.apply} armed={args.arm})")
    print(f"artifacts -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
