"""Simple flow/LGBM candidate adapter for the shared strategy pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor, log

import lightgbm as lgb
import numpy as np
import pandas as pd

from pancakebot.config.strategy_config import FlowCandidateConfig
from pancakebot.core.constants import BNB_WEI, GAS_COST_BET_BNB, GAS_COST_CLAIM_BNB
from pancakebot.core.errors import InvariantError
from pancakebot.domain.strategy.candidate_signal import StrategyCandidateSignal
from pancakebot.domain.types import Kline, Round
from pancakebot.runtime.settlement import settle_bet_against_closed_round

_EPS = 1e-12
_WINDOWS_SECONDS = (10, 30, 60, 120)
_RET_LAGS = (1, 3, 5)
_VOL_WINDOWS = (5, 10, 20)
_TOPK_FOR_SHARE = 5
_IMPACT_ITERS = 3
_MAX_EXTRA_HISTORY_ROWS = 64
_MIN_REGIME_HISTORY = 50
_FEATURE_DROP_COLS = frozenset(
    {
        "position",
        "winner_is_bull",
        "bullAmount",
        "bearAmount",
        "totalAmount",
        "startAt",
        "lockAt",
        "closeAt",
        "cutoff_ts",
        "lockPrice",
        "closePrice",
    }
)


def _side_allowed(*, side: str, allowed_sides: str) -> bool:
    mode = str(allowed_sides)
    if mode == "both":
        return True
    if mode == "bull_only":
        return str(side) == "Bull"
    if mode == "bear_only":
        return str(side) == "Bear"
    raise InvariantError("flow_candidate_allowed_sides_invalid")


def _topk_share(amounts: list[float], k: int) -> float:
    if not amounts:
        return float("nan")
    total = float(sum(float(x) for x in amounts))
    if float(total) <= 0.0:
        return float("nan")
    kk = max(1, min(int(k), len(amounts)))
    return float(sum(sorted(float(x) for x in amounts)[-kk:]) / total)


def _full_pool_amounts(round_t: Round) -> tuple[float, float, float]:
    bull = 0.0
    bear = 0.0
    for bet in round_t.bets:
        amount_bnb = float(int(bet.amount_wei)) / float(BNB_WEI)
        if str(bet.position) == "Bull":
            bull += float(amount_bnb)
        elif str(bet.position) == "Bear":
            bear += float(amount_bnb)
        else:
            raise InvariantError(f"flow_candidate_unexpected_bet_side: {bet.position}")
    total = float(bull + bear)
    return float(bull), float(bear), float(total)


def _impact_unit_profits(
    *,
    total_c: float,
    side_c: float,
    bet_bnb: float,
    treasury_fee_rate: float,
) -> tuple[float, float]:
    payout = ((float(total_c) + float(bet_bnb)) * (1.0 - float(treasury_fee_rate))) / max(
        float(side_c) + float(bet_bnb),
        _EPS,
    )
    unit_gas_bet = float(GAS_COST_BET_BNB) / max(float(bet_bnb), _EPS)
    unit_gas_claim = float(GAS_COST_CLAIM_BNB) / max(float(bet_bnb), _EPS)
    win_profit = float(payout - 1.0 - float(unit_gas_bet) - float(unit_gas_claim))
    lose_profit = float(-1.0 - float(unit_gas_bet))
    return float(win_profit), float(lose_profit)


def _impact_ev_per_unit(*, p_win: float, win_profit: float, lose_profit: float) -> float:
    prob = float(np.clip(float(p_win), 0.0, 1.0))
    return float(float(prob) * float(win_profit) + (1.0 - float(prob)) * float(lose_profit))


def _kelly_fraction_binary(*, p_win: float, win_profit: float, loss_amount: float) -> float:
    prob = float(np.clip(float(p_win), 0.0, 1.0))
    win = float(win_profit)
    loss = float(loss_amount)
    if float(win) <= 0.0 or float(loss) <= 0.0:
        return 0.0
    numer = float(prob) * float(win) - (1.0 - float(prob)) * float(loss)
    denom = float(loss) * float(win)
    return float(max(0.0, float(numer) / max(float(denom), _EPS)))


def _build_round_flow_row(
    *,
    round_t: Round,
    cutoff_seconds: int,
    prior_rows: list[dict[str, float | int | str | None]],
) -> dict[str, float | int | str | None]:
    if round_t.lock_at is None:
        raise InvariantError("flow_candidate_round_lock_at_missing")
    cutoff_ts = int(round_t.lock_at) - int(cutoff_seconds)

    bull_amt_c = 0.0
    bear_amt_c = 0.0
    bull_n_c = 0
    bear_n_c = 0
    bull_amts: list[float] = []
    bear_amts: list[float] = []
    all_cut_amounts: list[float] = []
    all_cut_ts: list[int] = []
    all_cut_side_is_bull: list[bool] = []

    for bet in round_t.bets:
        created_at = int(bet.created_at)
        if int(created_at) > int(cutoff_ts) or int(created_at) > int(round_t.lock_at):
            continue
        amount_bnb = float(int(bet.amount_wei)) / float(BNB_WEI)
        if str(bet.position) == "Bull":
            bull_amt_c += float(amount_bnb)
            bull_n_c += 1
            bull_amts.append(float(amount_bnb))
            all_cut_side_is_bull.append(True)
        elif str(bet.position) == "Bear":
            bear_amt_c += float(amount_bnb)
            bear_n_c += 1
            bear_amts.append(float(amount_bnb))
            all_cut_side_is_bull.append(False)
        else:
            raise InvariantError(f"flow_candidate_unexpected_bet_side: {bet.position}")
        all_cut_amounts.append(float(amount_bnb))
        all_cut_ts.append(int(created_at))

    total_amt_c = float(bull_amt_c + bear_amt_c)
    bull_ratio_c = float(bull_amt_c / (float(total_amt_c) + _EPS))
    log_imbalance_c = float(log((float(bull_amt_c) + _EPS) / (float(bear_amt_c) + _EPS)))
    payout_bull_est_c = float(float(total_amt_c) / (float(bull_amt_c) + _EPS))
    payout_bear_est_c = float(float(total_amt_c) / (float(bear_amt_c) + _EPS))
    payout_ratio_est_c = float(float(payout_bull_est_c) / (float(payout_bear_est_c) + _EPS))
    max_bull = float(max(bull_amts)) if bull_amts else 0.0
    max_bear = float(max(bear_amts)) if bear_amts else 0.0
    whale_time_to_cutoff = float("nan")
    if all_cut_amounts:
        whale_idx = int(np.argmax(np.asarray(all_cut_amounts, dtype=float)))
        whale_time_to_cutoff = float(int(cutoff_ts) - int(all_cut_ts[whale_idx]))

    row: dict[str, float | int | str | None] = {
        "epoch": int(round_t.epoch),
        "startAt": int(round_t.start_at),
        "lockAt": int(round_t.lock_at),
        "closeAt": None if round_t.close_at is None else int(round_t.close_at),
        "cutoff_ts": int(cutoff_ts),
        "lockPrice": None if round_t.lock_price is None else float(round_t.lock_price),
        "closePrice": None if round_t.close_price is None else float(round_t.close_price),
        "position": None if round_t.position is None else str(round_t.position),
        "seconds_into_round": float(int(cutoff_ts) - int(round_t.start_at)),
        "round_duration": float(int(round_t.lock_at) - int(round_t.start_at)),
        "gap_seconds": float("nan"),
        "bull_amt_c": float(bull_amt_c),
        "bear_amt_c": float(bear_amt_c),
        "total_amt_c": float(total_amt_c),
        "bull_n_c": float(bull_n_c),
        "bear_n_c": float(bear_n_c),
        "total_n_c": float(int(bull_n_c + bear_n_c)),
        "bull_ratio_c": float(bull_ratio_c),
        "log_imbalance_c": float(log_imbalance_c),
        "payout_bull_est_c": float(payout_bull_est_c),
        "payout_bear_est_c": float(payout_bear_est_c),
        "payout_ratio_est_c": float(payout_ratio_est_c),
        "max_bet_bull_c": float(max_bull),
        "max_bet_bear_c": float(max_bear),
        "top1_share_bull_c": float(max_bull / (float(bull_amt_c) + _EPS))
        if float(bull_amt_c) > 0.0
        else float("nan"),
        "top1_share_bear_c": float(max_bear / (float(bear_amt_c) + _EPS))
        if float(bear_amt_c) > 0.0
        else float("nan"),
        "topk_share_bull_c": float(_topk_share(bull_amts, _TOPK_FOR_SHARE)),
        "topk_share_bear_c": float(_topk_share(bear_amts, _TOPK_FOR_SHARE)),
        "whale_side_bull": 1.0 if float(max_bull) > float(max_bear) else 0.0,
        "whale_time_to_cutoff": float(whale_time_to_cutoff)
        if np.isfinite(whale_time_to_cutoff)
        else float("nan"),
    }

    for window_seconds in _WINDOWS_SECONDS:
        bull_amt_window = 0.0
        bear_amt_window = 0.0
        bull_n_window = 0
        bear_n_window = 0
        window_start = int(cutoff_ts) - int(window_seconds)
        for idx, created_at in enumerate(all_cut_ts):
            if int(created_at) <= int(window_start):
                continue
            amount_bnb = float(all_cut_amounts[idx])
            if bool(all_cut_side_is_bull[idx]):
                bull_amt_window += float(amount_bnb)
                bull_n_window += 1
            else:
                bear_amt_window += float(amount_bnb)
                bear_n_window += 1
        row[f"bull_amt_{int(window_seconds)}s"] = float(bull_amt_window)
        row[f"bear_amt_{int(window_seconds)}s"] = float(bear_amt_window)
        row[f"bull_n_{int(window_seconds)}s"] = float(bull_n_window)
        row[f"bear_n_{int(window_seconds)}s"] = float(bear_n_window)
        row[f"late_share_{int(window_seconds)}s"] = float(
            (float(bull_amt_window) + float(bear_amt_window)) / (float(total_amt_c) + _EPS)
        )
        row[f"late_log_imb_{int(window_seconds)}s"] = float(
            log((float(bull_amt_window) + _EPS) / (float(bear_amt_window) + _EPS))
        )

    net10 = float(row.get("bull_amt_10s", 0.0) or 0.0) - float(row.get("bear_amt_10s", 0.0) or 0.0)
    net60 = float(row.get("bull_amt_60s", 0.0) or 0.0) - float(row.get("bear_amt_60s", 0.0) or 0.0)
    row["net_accel_10_vs_60"] = float(float(net10) - float(net60) * (10.0 / 60.0))

    if prior_rows:
        last_close_at = prior_rows[-1].get("closeAt")
        if isinstance(last_close_at, (int, float)):
            row["gap_seconds"] = float(int(round_t.start_at) - int(last_close_at))

    prior_lock_prices = [
        float(x["lockPrice"])
        for x in prior_rows
        if isinstance(x.get("lockPrice"), (int, float)) and np.isfinite(float(x["lockPrice"]))
    ]
    current_ret_1 = float("nan")
    for lag in _RET_LAGS:
        key = f"ret_{int(lag)}"
        if len(prior_lock_prices) >= int(lag) + 1:
            row[key] = float(
                log(
                    (float(prior_lock_prices[-1]) + _EPS)
                    / (float(prior_lock_prices[-1 - int(lag)]) + _EPS)
                )
            )
            if int(lag) == 1:
                current_ret_1 = float(row[key])
        else:
            row[key] = float("nan")
    prior_ret_1 = [
        float(x["ret_1"])
        for x in prior_rows
        if isinstance(x.get("ret_1"), (int, float)) and np.isfinite(float(x["ret_1"]))
    ]
    for window in _VOL_WINDOWS:
        series = list(prior_ret_1[-max(0, int(window) - 1) :])
        if np.isfinite(current_ret_1):
            series.append(float(current_ret_1))
        if len(series) >= 2:
            row[f"vol_{int(window)}"] = float(np.std(np.asarray(series, dtype=float), ddof=1))
        else:
            row[f"vol_{int(window)}"] = float("nan")

    bull_final, bear_final, total_final = _full_pool_amounts(round_t)
    row["bullAmount"] = float(bull_final)
    row["bearAmount"] = float(bear_final)
    row["totalAmount"] = float(total_final)
    if round_t.position is not None:
        row["winner_is_bull"] = 1 if str(round_t.position).lower() == "bull" else 0
    return row


@dataclass(slots=True)
class _PendingFlowSignal:
    epoch: int
    side: str
    bet_size_bnb: float


class FlowCandidateAdapter:
    """Emit one flow/LGBM candidate signal compatible with the shared router."""

    def __init__(
        self,
        *,
        config: FlowCandidateConfig,
        cutoff_seconds: int,
        treasury_fee_fraction: float,
    ) -> None:
        self._config = config
        self._cutoff_seconds = int(cutoff_seconds)
        self._treasury_fee_fraction = float(treasury_fee_fraction)
        self._history_rows: list[dict[str, float | int | str | None]] = []
        self._pending_signals_by_epoch: dict[int, _PendingFlowSignal] = {}
        self._realized_unit_pnls: list[float] = []
        self._realized_wins: list[int] = []
        self._cooldown_remaining = 0
        self._shadow_bankroll_bnb = float(config.shadow_initial_bankroll_bnb)
        self._shadow_peak_bankroll_bnb = float(config.shadow_initial_bankroll_bnb)
        self._model: lgb.LGBMClassifier | None = None
        self._feature_columns: list[str] = []
        self._last_trained_epoch: int | None = None
        self._constant_p_bull: float | None = None
        self._validate_config()

    @property
    def enabled(self) -> bool:
        return bool(self._config.enabled)

    @property
    def candidate_name(self) -> str:
        return str(self._config.name)

    def refresh_klines(self, *, klines: list[Kline]) -> None:
        _ = klines

    def export_bootstrap_state(self) -> dict[str, object]:
        return {
            "history_rows": list(self._history_rows),
            "pending_signals_by_epoch": {
                str(epoch): {
                    "epoch": int(signal.epoch),
                    "side": str(signal.side),
                    "bet_size_bnb": float(signal.bet_size_bnb),
                }
                for epoch, signal in self._pending_signals_by_epoch.items()
            },
            "realized_unit_pnls": list(self._realized_unit_pnls),
            "realized_wins": list(self._realized_wins),
            "cooldown_remaining": int(self._cooldown_remaining),
            "shadow_bankroll_bnb": float(self._shadow_bankroll_bnb),
            "shadow_peak_bankroll_bnb": float(self._shadow_peak_bankroll_bnb),
            "last_trained_epoch": self._last_trained_epoch,
        }

    def import_bootstrap_state(self, *, state: dict[str, object]) -> None:
        history_rows = state.get("history_rows", [])
        if not isinstance(history_rows, list):
            raise InvariantError("flow_candidate_state_history_rows_invalid")
        self._history_rows = [dict(x) for x in history_rows if isinstance(x, dict)]
        pending_raw = state.get("pending_signals_by_epoch", {})
        if not isinstance(pending_raw, dict):
            raise InvariantError("flow_candidate_state_pending_invalid")
        pending: dict[int, _PendingFlowSignal] = {}
        for key, value in pending_raw.items():
            if not isinstance(value, dict):
                raise InvariantError("flow_candidate_state_pending_row_invalid")
            epoch = int(value.get("epoch", int(key)))
            pending[int(epoch)] = _PendingFlowSignal(
                epoch=int(epoch),
                side=str(value.get("side", "")),
                bet_size_bnb=float(value.get("bet_size_bnb", 0.0)),
            )
        self._pending_signals_by_epoch = pending
        self._realized_unit_pnls = [float(x) for x in state.get("realized_unit_pnls", [])]
        self._realized_wins = [int(x) for x in state.get("realized_wins", [])]
        self._cooldown_remaining = int(state.get("cooldown_remaining", 0))
        self._shadow_bankroll_bnb = float(
            state.get("shadow_bankroll_bnb", self._config.shadow_initial_bankroll_bnb)
        )
        self._shadow_peak_bankroll_bnb = float(
            state.get("shadow_peak_bankroll_bnb", self._shadow_bankroll_bnb)
        )
        last_trained = state.get("last_trained_epoch")
        self._last_trained_epoch = None if last_trained is None else int(last_trained)
        self._model = None
        self._feature_columns = []
        self._constant_p_bull = None
        self._prune_state()

    def bootstrap_from_closed_rounds(self, *, rounds: list[Round]) -> None:
        self.settle_closed_rounds(rounds=rounds)

    def settle_closed_rounds(self, *, rounds: list[Round]) -> None:
        if not rounds:
            return
        for round_t in sorted(rounds, key=lambda x: int(x.epoch)):
            epoch = int(round_t.epoch)
            pending = self._pending_signals_by_epoch.pop(int(epoch), None)
            if pending is not None and float(pending.bet_size_bnb) > 0.0:
                outcome = settle_bet_against_closed_round(
                    bet_bnb=float(pending.bet_size_bnb),
                    bet_side=str(pending.side),
                    round_closed=round_t,
                    treasury_fee_fraction=float(self._treasury_fee_fraction),
                )
                pnl_bnb = (
                    float(outcome.credit_bnb)
                    - float(pending.bet_size_bnb)
                    - float(GAS_COST_BET_BNB)
                )
                self._shadow_bankroll_bnb = max(0.0, float(self._shadow_bankroll_bnb) + float(pnl_bnb))
                self._shadow_peak_bankroll_bnb = max(
                    float(self._shadow_peak_bankroll_bnb),
                    float(self._shadow_bankroll_bnb),
                )
                unit_pnl = float(pnl_bnb) / max(float(pending.bet_size_bnb), _EPS)
                self._realized_unit_pnls.append(float(unit_pnl))
                self._realized_wins.append(1 if float(pnl_bnb) > 0.0 else 0)
            row = _build_round_flow_row(
                round_t=round_t,
                cutoff_seconds=int(self._cutoff_seconds),
                prior_rows=list(self._history_rows),
            )
            self._history_rows.append(row)
            self._prune_state()

    def candidate_signal_for_open_round(self, *, round_t: Round) -> StrategyCandidateSignal:
        if not bool(self._config.enabled):
            return self._skip_signal(skip_reason="flow_candidate_disabled")
        if round_t.lock_at is None:
            return self._skip_signal(skip_reason="flow_candidate_lock_at_missing")
        if len(self._history_rows) < int(self._config.train_size):
            return self._skip_signal(skip_reason="flow_candidate_warmup_history")

        latest_epoch = int(self._history_rows[-1]["epoch"])
        if (
            self._model is None
            or self._last_trained_epoch is None
            or int(latest_epoch) - int(self._last_trained_epoch) >= int(self._config.retrain_interval)
        ):
            self._train_model()
        if self._model is None and self._constant_p_bull is None:
            return self._skip_signal(skip_reason="flow_candidate_model_not_ready")

        row = _build_round_flow_row(
            round_t=round_t,
            cutoff_seconds=int(self._cutoff_seconds),
            prior_rows=list(self._history_rows),
        )

        if self._shadow_bankroll_bnb <= 0.0:
            return self._skip_signal(skip_reason="flow_candidate_shadow_bankroll_depleted")
        dd_pct = 0.0
        if float(self._shadow_peak_bankroll_bnb) > 0.0:
            dd_pct = float(
                (float(self._shadow_bankroll_bnb) - float(self._shadow_peak_bankroll_bnb))
                / float(self._shadow_peak_bankroll_bnb)
            )
        if float(dd_pct) <= -abs(float(self._config.drawdown_stop_pct)):
            return self._skip_signal(skip_reason="flow_candidate_drawdown_stop")
        if int(self._cooldown_remaining) > 0:
            self._cooldown_remaining -= 1
            return self._skip_signal(skip_reason="flow_candidate_cooldown")

        total_c = float(row.get("total_amt_c", 0.0) or 0.0)
        if float(total_c) <= 0.0:
            return self._skip_signal(skip_reason="flow_candidate_pool_bad")
        if float(total_c) < float(self._config.min_total_pool_c):
            return self._skip_signal(skip_reason="flow_candidate_pool_small")
        bull_ratio_c = row.get("bull_ratio_c")
        if isinstance(bull_ratio_c, (int, float)) and np.isfinite(float(bull_ratio_c)):
            if float(bull_ratio_c) < float(self._config.min_bull_ratio) or float(bull_ratio_c) > float(
                self._config.max_bull_ratio
            ):
                return self._skip_signal(skip_reason="flow_candidate_pool_imbalance")
        vol_20 = row.get("vol_20")
        if isinstance(vol_20, (int, float)) and np.isfinite(float(vol_20)):
            if float(vol_20) > float(self._config.vol_mid):
                return self._skip_signal(skip_reason="flow_candidate_high_volatility")
        if len(self._realized_unit_pnls) >= max(_MIN_REGIME_HISTORY, int(self._config.roll_window * 0.5)):
            recent = self._realized_unit_pnls[-int(self._config.roll_window) :]
            recent_wins = self._realized_wins[-int(self._config.roll_window) :]
            roll_edge = float(np.mean(np.asarray(recent, dtype=float))) if recent else 0.0
            roll_wr = float(np.mean(np.asarray(recent_wins, dtype=float))) if recent_wins else 0.0
            if float(roll_edge) < float(self._config.roll_edge_min) or float(roll_wr) < float(
                self._config.roll_winrate_min
            ):
                self._cooldown_remaining = int(self._config.cooldown_trades)
                return self._skip_signal(skip_reason="flow_candidate_regime_cooldown")

        p_bull = self._predict_p_bull(row=row)
        ev_bull, ev_bear = self._current_odds_ev(row=row, p_bull=float(p_bull))
        choose_bull = float(ev_bull) >= float(ev_bear)
        bet_side = "Bull" if bool(choose_bull) else "Bear"
        if not _side_allowed(side=str(bet_side), allowed_sides=str(self._config.allowed_sides)):
            return self._skip_signal(skip_reason="flow_candidate_side_not_allowed", p_bull=float(p_bull))
        p_win = float(p_bull) if bool(choose_bull) else float(1.0 - float(p_bull))
        side_c = float(row.get("bull_amt_c", 0.0) or 0.0) if bool(choose_bull) else float(
            row.get("bear_amt_c", 0.0) or 0.0
        )
        if float(side_c) <= 0.0:
            return self._skip_signal(skip_reason="flow_candidate_side_pool_bad", p_bull=float(p_bull))

        bet_bnb = min(float(self._config.max_bet_abs), float(self._shadow_bankroll_bnb))
        bet_bnb = max(float(bet_bnb), float(self._config.min_bet_size))
        ev_unit = float("-inf")
        for _ in range(int(_IMPACT_ITERS)):
            win_profit_unit, lose_profit_unit = _impact_unit_profits(
                total_c=float(total_c),
                side_c=float(side_c),
                bet_bnb=float(bet_bnb),
                treasury_fee_rate=float(self._treasury_fee_fraction),
            )
            ev_unit = _impact_ev_per_unit(
                p_win=float(p_win),
                win_profit=float(win_profit_unit),
                lose_profit=float(lose_profit_unit),
            )
            if float(ev_unit) < float(self._config.ev_threshold):
                return self._skip_signal(skip_reason="flow_candidate_ev_below_threshold", p_bull=float(p_bull))
            loss_amount = 1.0 + float(GAS_COST_BET_BNB) / max(float(bet_bnb), _EPS)
            kelly_frac = _kelly_fraction_binary(
                p_win=float(p_win),
                win_profit=float(win_profit_unit),
                loss_amount=float(loss_amount),
            )
            frac = float(np.clip(float(self._config.kelly_fraction) * float(kelly_frac), 0.0, float(self._config.max_fraction)))
            throttle_scale = 1.0
            if float(dd_pct) < -abs(float(self._config.drawdown_throttle_start_pct)):
                start = abs(float(self._config.drawdown_throttle_start_pct))
                stop = abs(float(self._config.drawdown_stop_pct))
                dd_abs = abs(float(dd_pct))
                if float(dd_abs) >= float(stop):
                    throttle_scale = float(self._config.drawdown_throttle_min_scale)
                else:
                    t = (float(dd_abs) - float(start)) / max(float(stop) - float(start), _EPS)
                    throttle_scale = float(1.0 - float(t) * (1.0 - float(self._config.drawdown_throttle_min_scale)))
                throttle_scale = float(
                    np.clip(float(throttle_scale), float(self._config.drawdown_throttle_min_scale), 1.0)
                )
            frac *= float(throttle_scale)
            bet_bnb = min(
                float(frac) * float(self._shadow_bankroll_bnb),
                float(self._config.max_bet_abs),
                float(self._shadow_bankroll_bnb),
            )
            bet_bnb = max(float(bet_bnb), float(self._config.min_bet_size))
            bet_bnb = float(
                floor(float(bet_bnb) / float(self._config.round_to)) * float(self._config.round_to)
            )

        if float(bet_bnb) <= 0.0:
            return self._skip_signal(skip_reason="flow_candidate_bet_nonpositive", p_bull=float(p_bull))
        pool_cap = min(
            float(self._config.max_total_pool_share) * float(total_c),
            float(self._config.max_side_pool_share) * max(float(side_c), _EPS),
        )
        if float(bet_bnb) > float(pool_cap):
            bet_capped = float(
                floor(float(pool_cap) / float(self._config.round_to)) * float(self._config.round_to)
            )
            if float(bet_capped) < float(self._config.min_bet_size):
                return self._skip_signal(skip_reason="flow_candidate_pool_cap", p_bull=float(p_bull))
            bet_bnb = float(bet_capped)

        win_profit_unit, lose_profit_unit = _impact_unit_profits(
            total_c=float(total_c),
            side_c=float(side_c),
            bet_bnb=float(bet_bnb),
            treasury_fee_rate=float(self._treasury_fee_fraction),
        )
        ev_unit = _impact_ev_per_unit(
            p_win=float(p_win),
            win_profit=float(win_profit_unit),
            lose_profit=float(lose_profit_unit),
        )
        expected_profit_bnb = float(ev_unit) * float(bet_bnb)
        if float(expected_profit_bnb) <= 0.0:
            return self._skip_signal(skip_reason="flow_candidate_expected_profit_nonpositive", p_bull=float(p_bull))
        selector_score_bnb = float(expected_profit_bnb) - float(self._config.selector_score_penalty_bnb)
        if float(selector_score_bnb) <= 0.0:
            return self._skip_signal(
                skip_reason="flow_candidate_selector_score_below_penalty",
                p_bull=float(p_bull),
            )

        signal = StrategyCandidateSignal(
            candidate_name=str(self._config.name),
            action="BET",
            bet_side=str(bet_side),
            bet_size_bnb=float(bet_bnb),
            expected_profit_bnb=float(expected_profit_bnb),
            selector_score_bnb=float(selector_score_bnb),
            skip_reason=None,
            p_bull=float(p_bull),
            dislocation_bull=float(row.get("log_imbalance_c", 0.0) or 0.0),
        )
        self._pending_signals_by_epoch[int(round_t.epoch)] = _PendingFlowSignal(
            epoch=int(round_t.epoch),
            side=str(bet_side),
            bet_size_bnb=float(bet_bnb),
        )
        return signal

    def _skip_signal(self, *, skip_reason: str, p_bull: float | None = None) -> StrategyCandidateSignal:
        return StrategyCandidateSignal(
            candidate_name=str(self._config.name),
            action="SKIP",
            bet_side=None,
            bet_size_bnb=0.0,
            expected_profit_bnb=None,
            selector_score_bnb=None,
            skip_reason=str(skip_reason),
            p_bull=None if p_bull is None else float(p_bull),
            dislocation_bull=None,
        )

    def _train_model(self) -> None:
        if len(self._history_rows) < int(self._config.train_size):
            self._model = None
            self._feature_columns = []
            self._constant_p_bull = None
            return
        df = pd.DataFrame(self._history_rows[-int(self._config.train_size) :])
        feature_columns = [
            str(col)
            for col in df.columns
            if str(col) not in _FEATURE_DROP_COLS and pd.api.types.is_numeric_dtype(df[str(col)])
        ]
        if "winner_is_bull" not in df.columns:
            raise InvariantError("flow_candidate_training_label_missing")
        y = df["winner_is_bull"].astype(int)
        if int(y.nunique()) < 2:
            self._model = None
            self._feature_columns = list(feature_columns)
            self._constant_p_bull = float(y.iloc[-1])
            self._last_trained_epoch = int(df["epoch"].iloc[-1])
            return
        clf = lgb.LGBMClassifier(
            n_estimators=int(self._config.n_estimators),
            learning_rate=float(self._config.learning_rate),
            num_leaves=int(self._config.num_leaves),
            subsample=float(self._config.subsample),
            colsample_bytree=float(self._config.colsample_bytree),
            random_state=int(self._config.random_seed),
            verbose=-1,
        )
        clf.fit(df.loc[:, feature_columns], y)
        self._model = clf
        self._feature_columns = list(feature_columns)
        self._constant_p_bull = None
        self._last_trained_epoch = int(df["epoch"].iloc[-1])

    def _predict_p_bull(self, *, row: dict[str, float | int | str | None]) -> float:
        if self._constant_p_bull is not None:
            return float(np.clip(float(self._constant_p_bull), 0.0, 1.0))
        if self._model is None or not self._feature_columns:
            raise InvariantError("flow_candidate_model_not_ready")
        frame = pd.DataFrame([{col: row.get(col, np.nan) for col in self._feature_columns}])
        return float(np.clip(float(self._model.predict_proba(frame)[:, 1][0]), 0.0, 1.0))

    def _current_odds_ev(self, *, row: dict[str, float | int | str | None], p_bull: float) -> tuple[float, float]:
        bull_c = float(row.get("bull_amt_c", 0.0) or 0.0)
        bear_c = float(row.get("bear_amt_c", 0.0) or 0.0)
        total_c = float(bull_c + bear_c)
        total_p = float(total_c + 1.0)
        bull_p = float(bull_c + 1.0)
        bear_p = float(bear_c + 1.0)
        payout_bull = (float(total_p) * (1.0 - float(self._treasury_fee_fraction))) / max(float(bull_p), _EPS)
        payout_bear = (float(total_p) * (1.0 - float(self._treasury_fee_fraction))) / max(float(bear_p), _EPS)
        lose_profit = -1.0
        win_profit_bull = float(payout_bull - 1.0)
        win_profit_bear = float(payout_bear - 1.0)
        ev_bull = float(float(p_bull) * float(win_profit_bull) + (1.0 - float(p_bull)) * float(lose_profit))
        ev_bear = float((1.0 - float(p_bull)) * float(win_profit_bear) + float(p_bull) * float(lose_profit))
        return float(ev_bull), float(ev_bear)

    def _prune_state(self) -> None:
        keep_rows = int(self._config.train_size) + int(_MAX_EXTRA_HISTORY_ROWS)
        if len(self._history_rows) > int(keep_rows):
            self._history_rows = self._history_rows[-int(keep_rows) :]
        keep_realized = max(int(self._config.roll_window), _MIN_REGIME_HISTORY)
        if len(self._realized_unit_pnls) > int(keep_realized):
            self._realized_unit_pnls = self._realized_unit_pnls[-int(keep_realized) :]
        if len(self._realized_wins) > int(keep_realized):
            self._realized_wins = self._realized_wins[-int(keep_realized) :]

    def _validate_config(self) -> None:
        if float(self._treasury_fee_fraction) < 0.0 or float(self._treasury_fee_fraction) >= 1.0:
            raise InvariantError("flow_candidate_treasury_fee_out_of_range")
        if float(self._config.shadow_initial_bankroll_bnb) <= 0.0:
            raise InvariantError("flow_candidate_shadow_initial_bankroll_nonpositive")
        if int(self._config.train_size) <= 0:
            raise InvariantError("flow_candidate_train_size_nonpositive")
        if int(self._config.retrain_interval) <= 0:
            raise InvariantError("flow_candidate_retrain_interval_nonpositive")
        if int(self._config.n_estimators) <= 0:
            raise InvariantError("flow_candidate_n_estimators_nonpositive")
        if float(self._config.learning_rate) <= 0.0:
            raise InvariantError("flow_candidate_learning_rate_nonpositive")
        if int(self._config.num_leaves) <= 1:
            raise InvariantError("flow_candidate_num_leaves_invalid")
        if not (0.0 < float(self._config.subsample) <= 1.0):
            raise InvariantError("flow_candidate_subsample_out_of_range")
        if not (0.0 < float(self._config.colsample_bytree) <= 1.0):
            raise InvariantError("flow_candidate_colsample_bytree_out_of_range")
        if float(self._config.max_bet_abs) <= 0.0:
            raise InvariantError("flow_candidate_max_bet_abs_nonpositive")
        if float(self._config.min_bet_size) <= 0.0:
            raise InvariantError("flow_candidate_min_bet_size_nonpositive")
        if float(self._config.round_to) <= 0.0:
            raise InvariantError("flow_candidate_round_to_nonpositive")
        if str(self._config.allowed_sides) not in ("both", "bull_only", "bear_only"):
            raise InvariantError("flow_candidate_allowed_sides_invalid")
        if float(self._config.selector_score_penalty_bnb) < 0.0:
            raise InvariantError("flow_candidate_selector_score_penalty_negative")
