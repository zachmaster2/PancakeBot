"""Step 3: Feature enhancements on top of the BTC lead baseline.

All features are tested as ADDITIONS to the BTC lead signal,
not standalone signals (standalone scan was done in v2_step2).

Features explored:
  1. BNB-BTC spread (catch-up effect)
  2. BTC move magnitude sizing
  3. BTC volume confirmation
  4. Previous round outcome filter
  5. Pool imbalance as contrarian signal
  6. BTC return acceleration

Walk-forward: train on first 70%, validate on last 30%.
Cutoffs: klines at lock_at - 2, pools at lock_at - 6.
"""
from __future__ import annotations

import json, sys, math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pancakebot.core.constants import (
    GAS_COST_BET_BNB, INTERVAL_SECONDS, BNB_WEI, POOL_CUTOFF_SECONDS,
)
from pancakebot.infra.closed_rounds_store import ClosedRoundsStore
from pancakebot.domain.strategy.momentum_gate import _trim_to_window, _get_return
from pancakebot.runtime.settlement import settle_bet_against_closed_round

CUTOFF_S = 4          # TODO: change to 2 once klines are resynced
POOL_CUTOFF_S = POOL_CUTOFF_SECONDS  # pool cutoff: lock_at - 6
CANDLE_COUNT = 31
TREASURY_FEE = 0.03


def load_data():
    store = ClosedRoundsStore("var/closed_rounds.jsonl")
    rounds = list(store.iter_closed_rounds())

    def load_kl(p):
        out = {}
        for line in Path(p).read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("klines_1s") is not None:
                out[int(rec["epoch"])] = rec["klines_1s"]
        return out

    return rounds, load_kl("var/bnb_spot_prices.jsonl"), load_kl("var/btc_spot_prices.jsonl")


def get_closes(raw, cutoff_ms):
    trimmed = _trim_to_window(raw, cutoff_ms)
    if len(trimmed) < CANDLE_COUNT:
        return None
    return [k[4] for k in trimmed]


def get_closes_and_volumes(raw, cutoff_ms):
    trimmed = _trim_to_window(raw, cutoff_ms)
    if len(trimmed) < CANDLE_COUNT:
        return None, None
    return [k[4] for k in trimmed], [k[5] for k in trimmed]


def get_pool(rnd, lock_at):
    """Pool amounts at decision time (lock_at - POOL_CUTOFF_S)."""
    pool_cutoff_ts = lock_at - POOL_CUTOFF_S
    bull_wei = 0
    bear_wei = 0
    for bet in rnd.bets:
        if int(bet.created_at) > pool_cutoff_ts:
            continue
        if bet.position == "Bull":
            bull_wei += int(bet.amount_wei)
        else:
            bear_wei += int(bet.amount_wei)
    return bull_wei / 1e18, bear_wei / 1e18


def payout_mult(pool_bull, pool_bear, signal):
    pool_total = pool_bull + pool_bear
    if pool_total <= 0:
        return 0.0
    our_side = pool_bull if signal == "Bull" else pool_bear
    if our_side <= 0:
        return 0.0
    return pool_total * (1.0 - TREASURY_FEE) / our_side


def simulate(rounds, spot, btc_kl, *, config):
    """Flexible simulation with feature enhancements."""
    btc_lb = config["btc_lb"]
    btc_thresh = config["btc_thresh"]
    min_payout = config.get("min_payout", 0.0)
    base_bet = config.get("base_bet", 0.10)
    payout_sizing = config.get("payout_sizing", False)

    # Feature enhancement flags
    spread_filter = config.get("spread_filter", False)
    spread_lb = config.get("spread_lb", 7)
    spread_min = config.get("spread_min", 0.0)

    magnitude_sizing = config.get("magnitude_sizing", False)
    magnitude_base = config.get("magnitude_base", 1.0)
    magnitude_scale = config.get("magnitude_scale", 1000.0)

    volume_filter = config.get("volume_filter", False)
    volume_lb = config.get("volume_lb", 10)
    volume_min_ratio = config.get("volume_min_ratio", 1.0)

    prev_outcome_filter = config.get("prev_outcome_filter", None)  # "streak" or "reversal"

    pool_imbalance_filter = config.get("pool_imbalance_filter", False)
    pool_imbalance_min = config.get("pool_imbalance_min", 0.0)

    btc_accel_filter = config.get("btc_accel_filter", False)
    btc_accel_short = config.get("btc_accel_short", 3)

    skip_hours = config.get("skip_hours", ())

    trades = []
    prev_outcome = None  # "Bull" or "Bear" outcome of previous round

    for rnd in rounds:
        lock_at = int(rnd.lock_at)
        epoch = int(rnd.epoch)
        cutoff_ms = (lock_at - CUTOFF_S) * 1000

        # Track previous round outcome for filters
        current_outcome = rnd.position  # "Bull", "Bear", or None

        # Hour filter
        if skip_hours:
            hour_utc = (lock_at % 86400) // 3600
            if hour_utc in skip_hours:
                prev_outcome = current_outcome
                continue

        # BTC signal (baseline)
        btc_raw = btc_kl.get(epoch)
        if not btc_raw:
            prev_outcome = current_outcome
            continue
        btc_closes, btc_vols = get_closes_and_volumes(btc_raw, cutoff_ms)
        if btc_closes is None:
            prev_outcome = current_outcome
            continue

        btc_r = _get_return(btc_closes, btc_lb)
        if btc_r is None or abs(btc_r) < btc_thresh:
            prev_outcome = current_outcome
            continue
        signal = "Bull" if btc_r > 0 else "Bear"

        # --- Feature 1: BNB-BTC spread filter ---
        if spread_filter:
            bnb_raw = spot.get(epoch)
            if not bnb_raw:
                prev_outcome = current_outcome
                continue
            bnb_closes = get_closes(bnb_raw, cutoff_ms)
            if bnb_closes is None:
                prev_outcome = current_outcome
                continue
            bnb_r = _get_return(bnb_closes, spread_lb)
            if bnb_r is None:
                prev_outcome = current_outcome
                continue
            # Spread = BTC move - BNB move (how much BNB lags)
            spread = abs(btc_r) - abs(bnb_r) if (btc_r > 0) == (bnb_r > 0) else abs(btc_r) + abs(bnb_r)
            if spread < spread_min:
                prev_outcome = current_outcome
                continue

        # --- Feature 3: BTC volume confirmation ---
        if volume_filter:
            if btc_vols is None:
                prev_outcome = current_outcome
                continue
            recent_vol = sum(btc_vols[-volume_lb:])
            baseline_vol = sum(btc_vols[:-volume_lb]) / max(1, len(btc_vols) - volume_lb) * volume_lb
            if baseline_vol > 0 and recent_vol / baseline_vol < volume_min_ratio:
                prev_outcome = current_outcome
                continue

        # --- Feature 4: Previous round outcome ---
        if prev_outcome_filter == "streak" and prev_outcome is not None:
            if signal != prev_outcome:
                prev_outcome = current_outcome
                continue
        elif prev_outcome_filter == "reversal" and prev_outcome is not None:
            if signal == prev_outcome:
                prev_outcome = current_outcome
                continue

        # --- Feature 6: BTC acceleration filter ---
        if btc_accel_filter:
            btc_r_short = _get_return(btc_closes, btc_accel_short)
            if btc_r_short is None:
                prev_outcome = current_outcome
                continue
            # Acceleration = short return in same direction as long return
            # AND short return is proportionally stronger
            if (btc_r_short > 0) != (btc_r > 0):
                prev_outcome = current_outcome
                continue

        # Pool / payout
        pool_bull, pool_bear = get_pool(rnd, lock_at)
        pool_total = pool_bull + pool_bear
        if pool_total <= 0:
            prev_outcome = current_outcome
            continue

        pm = payout_mult(pool_bull, pool_bear, signal)
        if pm < min_payout:
            prev_outcome = current_outcome
            continue

        # --- Feature 5: Pool imbalance filter ---
        if pool_imbalance_filter:
            imbalance = abs(pool_bull - pool_bear) / pool_total
            if imbalance < pool_imbalance_min:
                prev_outcome = current_outcome
                continue

        # Sizing
        bet = base_bet
        if payout_sizing:
            bet *= max(0.3, 0.1 + 1.0 * (pm - 1.0))
            bet = min(2.0, bet)

        # --- Feature 2: Magnitude sizing ---
        if magnitude_sizing:
            strength = abs(btc_r) * magnitude_scale
            bet *= max(0.5, min(3.0, magnitude_base + strength))
            bet = min(2.0, bet)

        out = settle_bet_against_closed_round(
            bet_bnb=bet, bet_side=signal,
            round_closed=rnd, treasury_fee_fraction=TREASURY_FEE,
        )
        profit = out.credit_bnb - bet - GAS_COST_BET_BNB
        trades.append(profit)

        prev_outcome = current_outcome

    n = len(trades)
    wins = sum(1 for p in trades if p > 0)
    wr = wins / max(1, n) * 100
    pnl = sum(trades)
    return n, wins, wr, pnl


def show(label, train, valid, spot, btc, config, *, min_train=50, min_valid=20):
    nt, _, wt, pt = simulate(train, spot, btc, config=config)
    if nt < min_train:
        return False
    nv, _, wv, pv = simulate(valid, spot, btc, config=config)
    if nv < min_valid:
        return False
    flag = " ***" if pv > 0 else ""
    per_bet = pv / max(1, nv)
    print(f"  {label:60s} T={wt:5.1f}%({nt:4d}) V={wv:5.1f}%({nv:4d}) PnL={pv:+7.2f} /bet={per_bet:+.4f}{flag}")
    return True


def main():
    rounds, spot, btc = load_data()
    split = int(len(rounds) * 0.70)
    train = rounds[:split]
    valid = rounds[split:]
    print(f"Rounds: {len(rounds)} total, {len(train)} train, {len(valid)} validate")
    print(f"Kline cutoff: lock_at - {CUTOFF_S}s, Pool cutoff: lock_at - {POOL_CUTOFF_S}s\n")

    # Baseline configs from v2_step2c
    baselines = [
        {"btc_lb": 7, "btc_thresh": 0.0007},
        {"btc_lb": 10, "btc_thresh": 0.0005},
        {"btc_lb": 10, "btc_thresh": 0.0007},
        {"btc_lb": 15, "btc_thresh": 0.0010},
    ]

    # =====================================================================
    print("=" * 90)
    print("BASELINE: BTC lead signal with corrected cutoffs (cutoff_s=2, pool_cutoff=6)")
    print("=" * 90)

    for bc in baselines:
        lb, th = bc["btc_lb"], bc["btc_thresh"]
        for min_pm in [0.0, 1.85, 2.0, 2.5]:
            for ps in [False, True]:
                ps_str = "payout_sz" if ps else "fixed"
                cfg = {**bc, "min_payout": min_pm, "base_bet": 0.10, "payout_sizing": ps}
                show(f"btc({lb},{th}) pm>={min_pm:.2f} {ps_str}", train, valid, spot, btc, cfg)

    # =====================================================================
    print(f"\n{'=' * 90}")
    print("FEATURE 1: BNB-BTC spread (catch-up effect)")
    print("  BTC moved but BNB hasn't caught up -> BNB will follow")
    print("=" * 90)

    for bc in baselines[:2]:
        lb, th = bc["btc_lb"], bc["btc_thresh"]
        for spread_lb in [3, 5, 7]:
            for spread_min in [0.0001, 0.0002, 0.0003, 0.0005]:
                for min_pm in [0.0, 1.85]:
                    cfg = {**bc, "min_payout": min_pm, "base_bet": 0.10,
                           "spread_filter": True, "spread_lb": spread_lb,
                           "spread_min": spread_min}
                    show(f"btc({lb},{th}) spread(lb={spread_lb},min={spread_min}) pm>={min_pm:.2f}",
                         train, valid, spot, btc, cfg)

    # =====================================================================
    print(f"\n{'=' * 90}")
    print("FEATURE 2: BTC move magnitude sizing")
    print("  Bigger BTC move -> higher confidence -> bigger bet")
    print("=" * 90)

    for bc in baselines[:2]:
        lb, th = bc["btc_lb"], bc["btc_thresh"]
        for min_pm in [0.0, 1.85, 2.0]:
            for mag_base, mag_scale in [(0.5, 500), (0.5, 1000), (1.0, 500), (1.0, 1000)]:
                cfg = {**bc, "min_payout": min_pm, "base_bet": 0.10,
                       "magnitude_sizing": True, "magnitude_base": mag_base,
                       "magnitude_scale": mag_scale}
                show(f"btc({lb},{th}) mag(b={mag_base},s={mag_scale}) pm>={min_pm:.2f}",
                     train, valid, spot, btc, cfg)

    # =====================================================================
    print(f"\n{'=' * 90}")
    print("FEATURE 3: BTC volume confirmation")
    print("  Higher BTC volume during signal window -> more reliable move")
    print("=" * 90)

    for bc in baselines[:2]:
        lb, th = bc["btc_lb"], bc["btc_thresh"]
        for vol_lb in [5, 10, 15]:
            for vol_ratio in [1.2, 1.5, 2.0, 3.0]:
                for min_pm in [0.0, 1.85]:
                    cfg = {**bc, "min_payout": min_pm, "base_bet": 0.10,
                           "volume_filter": True, "volume_lb": vol_lb,
                           "volume_min_ratio": vol_ratio}
                    show(f"btc({lb},{th}) vol(lb={vol_lb},r={vol_ratio}) pm>={min_pm:.2f}",
                         train, valid, spot, btc, cfg)

    # =====================================================================
    print(f"\n{'=' * 90}")
    print("FEATURE 4: Previous round outcome")
    print("  Streak: only bet if signal matches last round outcome")
    print("  Reversal: only bet if signal opposes last round outcome")
    print("=" * 90)

    for bc in baselines[:2]:
        lb, th = bc["btc_lb"], bc["btc_thresh"]
        for mode in ["streak", "reversal"]:
            for min_pm in [0.0, 1.85, 2.0]:
                cfg = {**bc, "min_payout": min_pm, "base_bet": 0.10,
                       "prev_outcome_filter": mode}
                show(f"btc({lb},{th}) prev={mode} pm>={min_pm:.2f}",
                     train, valid, spot, btc, cfg)

    # =====================================================================
    print(f"\n{'=' * 90}")
    print("FEATURE 5: Pool imbalance filter")
    print("  Only bet when pools are meaningfully imbalanced")
    print("=" * 90)

    for bc in baselines[:2]:
        lb, th = bc["btc_lb"], bc["btc_thresh"]
        for imb_min in [0.1, 0.2, 0.3, 0.4, 0.5]:
            for min_pm in [0.0, 1.85]:
                cfg = {**bc, "min_payout": min_pm, "base_bet": 0.10,
                       "pool_imbalance_filter": True, "pool_imbalance_min": imb_min}
                show(f"btc({lb},{th}) imb>={imb_min:.1f} pm>={min_pm:.2f}",
                     train, valid, spot, btc, cfg)

    # =====================================================================
    print(f"\n{'=' * 90}")
    print("FEATURE 6: BTC acceleration (short return in same direction)")
    print("  BTC move is accelerating -> more reliable signal")
    print("=" * 90)

    for bc in baselines[:2]:
        lb, th = bc["btc_lb"], bc["btc_thresh"]
        for accel_short in [2, 3, 5]:
            if accel_short >= lb:
                continue
            for min_pm in [0.0, 1.85, 2.0]:
                cfg = {**bc, "min_payout": min_pm, "base_bet": 0.10,
                       "btc_accel_filter": True, "btc_accel_short": accel_short}
                show(f"btc({lb},{th}) accel(s={accel_short}) pm>={min_pm:.2f}",
                     train, valid, spot, btc, cfg)

    # =====================================================================
    print(f"\n{'=' * 90}")
    print("COMBINED: Best payout sizing + best enhancement features")
    print("=" * 90)

    # Try combining payout sizing with each feature that showed promise
    for bc in baselines[:2]:
        lb, th = bc["btc_lb"], bc["btc_thresh"]
        for min_pm in [1.85, 2.0, 2.5]:
            # Payout sizing + spread
            for spread_min in [0.0001, 0.0002]:
                cfg = {**bc, "min_payout": min_pm, "base_bet": 0.10,
                       "payout_sizing": True,
                       "spread_filter": True, "spread_lb": 7, "spread_min": spread_min}
                show(f"btc({lb},{th}) ps + spread(min={spread_min}) pm>={min_pm:.2f}",
                     train, valid, spot, btc, cfg)

            # Payout sizing + acceleration
            for accel_s in [2, 3]:
                if accel_s >= lb:
                    continue
                cfg = {**bc, "min_payout": min_pm, "base_bet": 0.10,
                       "payout_sizing": True,
                       "btc_accel_filter": True, "btc_accel_short": accel_s}
                show(f"btc({lb},{th}) ps + accel(s={accel_s}) pm>={min_pm:.2f}",
                     train, valid, spot, btc, cfg)

            # Payout sizing + volume
            for vol_ratio in [1.2, 1.5]:
                cfg = {**bc, "min_payout": min_pm, "base_bet": 0.10,
                       "payout_sizing": True,
                       "volume_filter": True, "volume_lb": 10, "volume_min_ratio": vol_ratio}
                show(f"btc({lb},{th}) ps + vol(r={vol_ratio}) pm>={min_pm:.2f}",
                     train, valid, spot, btc, cfg)

            # Payout sizing + magnitude
            cfg = {**bc, "min_payout": min_pm, "base_bet": 0.10,
                   "payout_sizing": True,
                   "magnitude_sizing": True, "magnitude_base": 0.5, "magnitude_scale": 1000}
            show(f"btc({lb},{th}) ps + mag(0.5,1000) pm>={min_pm:.2f}",
                 train, valid, spot, btc, cfg)

    print("\nDone.")


if __name__ == "__main__":
    main()
