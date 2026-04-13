"""Momentum-only strategy pipeline.

Signal: dual-asset BNB+BTC momentum gate (OKX 1s candles).
Sizing: pool-proportional with linear payout-proportional scaling.

Pre-signal filters (applied before sizing):
  - Low-liquidity hour exclusion: skip hours with <52% WR on 34k rounds.
  - Payout floor: skip rounds where the payout multiplier on our
    signal's side is below 1.85.

Live/dry mode: MomentumGate fetches live 1s klines from OKX at cutoff.
Backtest mode: uses pre-fetched 1s kline arrays from JSONL caches.
"""

from __future__ import annotations

from dataclasses import dataclass

from pancakebot.core.constants import BNB_WEI
from pancakebot.core.errors import InvariantError
from pancakebot.domain.strategy.momentum_gate import (
    MomentumGate,
    MomentumGateConfig,
    MomentumGateResult,
    compute_signal_from_klines,
)
from pancakebot.domain.types import Round

# Sizing constants — tuned together on 34k clean rounds; do not change independently.
_BASE_FRAC = 0.06
_FLOOR_BNB = 0.10
_CAP_BNB = 2.0
_BTC_AGREE_MULT = 2.0
_MAX_PAYOUT_DILUTION = 0.12     # cap bet if it would dilute our payout by > 12%
_BTC_DISAGREE_MULT = 0.7

# Linear payout-proportional sizing:
#   bet_mult = _PAYOUT_LINEAR_BASE + _PAYOUT_LINEAR_SLOPE * (payout - 1.0)
# Higher payout → bigger bet.  Replaces threshold-based payout bands.
_PAYOUT_LINEAR_BASE = 0.1
_PAYOUT_LINEAR_SLOPE = 1.0

# Pre-signal filters — tuned on 20k rounds alongside sizing constants.
_LOW_LIQ_SKIP_HOURS = (3, 4, 7, 10, 18, 19)  # skip hours with <52% WR on 34k rounds
_MIN_OUR_PAYOUT = 1.85           # skip if payout on our side < 1.85



@dataclass(frozen=True, slots=True)
class StrategyPipelineDecision:
    """Normalized open-round strategy pipeline decision (momentum-only variant)."""

    action: str
    selected_strategy: str | None
    bet_side: str | None
    bet_size_bnb: float
    expected_profit_bnb: float
    selector_score_bnb: float | None
    skip_reason: str | None
    p_bull: float | None
    controller_mode: str | None = None
    controller_estimator_mode: str | None = None
    controller_window_index: int | None = None
    controller_lookback_windows_used: int | None = None
    controller_selected_profile: str | None = None
    controller_selected_action: str | None = None


def _compute_bet_size(
    *,
    signal: str,
    tier: str,
    btc_agrees: bool,
    btc_disagrees: bool,
    pool_bull_bnb: float,
    pool_bear_bnb: float,
    treasury_fee_fraction: float,
) -> float:
    """Pool-proportional sizing with linear payout-proportional scaling."""
    pool_bnb = pool_bull_bnb + pool_bear_bnb
    if pool_bnb <= 0:
        return _FLOOR_BNB

    bet = max(_FLOOR_BNB, pool_bnb * _BASE_FRAC)

    # Linear payout-proportional adjustment: bet more when payout is higher.
    our_side = pool_bull_bnb if signal == "Bull" else pool_bear_bnb
    if our_side > 0:
        pm = pool_bnb * (1.0 - treasury_fee_fraction) / our_side
        mult = max(0.3, _PAYOUT_LINEAR_BASE + _PAYOUT_LINEAR_SLOPE * (pm - 1.0))
        bet *= mult

    # BTC confirmation boost
    if btc_agrees:
        bet *= _BTC_AGREE_MULT
    elif btc_disagrees:
        bet *= _BTC_DISAGREE_MULT

    return min(_CAP_BNB, bet)


class MomentumOnlyPipeline:
    """Momentum-only pipeline: satisfies the StrategyPipeline interface.

    In live/dry mode pass `gate` (a MomentumGate backed by OKX client).
    In backtest mode leave `gate=None`; the pipeline uses cached 1s klines.
    """

    def __init__(
        self,
        *,
        config: MomentumGateConfig,
        gate: MomentumGate | None,
        cutoff_seconds: int,
        min_bet_amount_bnb: float,
        treasury_fee_fraction: float,
    ) -> None:
        self._cfg = config
        self._gate = gate
        self._cutoff_seconds = int(cutoff_seconds)
        self._min_bet_amount_bnb = float(min_bet_amount_bnb)
        self._treasury_fee_fraction = float(treasury_fee_fraction)
        self._last_settled_epoch: int | None = None
        # Backtest: 1s klines per epoch {epoch: [[ts_ms, o, h, l, c, vol], ...]}
        self._spot_klines_by_epoch: dict[int, list[list]] = {}
        self._btc_klines_by_epoch: dict[int, list[list]] = {}

    # ------------------------------------------------------------------
    # Required interface: StrategyPipeline-compatible
    # ------------------------------------------------------------------

    @property
    def last_settled_epoch(self) -> int | None:
        return self._last_settled_epoch

    @property
    def router_mode(self) -> str:
        return "momentum_gate"

    def selector_ready(self) -> bool:
        return True

    def refresh_spot_klines(self, *, spot_klines_by_epoch: dict[int, list[list]]) -> None:
        """Load pre-fetched BNB 1s kline arrays keyed by epoch (backtest mode)."""
        self._spot_klines_by_epoch = dict(spot_klines_by_epoch)

    def refresh_btc_klines(self, *, btc_klines_by_epoch: dict[int, list[list]]) -> None:
        """Load pre-fetched BTC 1s kline arrays keyed by epoch (backtest mode)."""
        self._btc_klines_by_epoch = dict(btc_klines_by_epoch)

    def settle_closed_rounds(self, *, rounds: list[Round]) -> None:
        """Track the last settled epoch (no ML state to update)."""
        for r in sorted(rounds, key=lambda x: int(x.epoch)):
            epoch = int(r.epoch)
            if self._last_settled_epoch is None or epoch > int(self._last_settled_epoch):
                self._last_settled_epoch = epoch

    def bootstrap_from_closed_rounds(self, *, rounds: list[Round]) -> None:
        """Set last_settled_epoch from the warmup batch. No ML state."""
        self.settle_closed_rounds(rounds=rounds)

    def export_bootstrap_state(self) -> dict:
        return {
            "last_settled_epoch": self._last_settled_epoch,
        }

    def import_bootstrap_state(self, *, state: dict) -> None:
        raw = state.get("last_settled_epoch")
        self._last_settled_epoch = None if raw is None else int(raw)

    # ------------------------------------------------------------------
    # Core decision
    # ------------------------------------------------------------------

    def decide_open_round(
        self,
        *,
        round_t: Round,
        bankroll_bnb: float,
        allow_oracle_mode: bool,
        pool_bull_bnb: float = 0.0,
        pool_bear_bnb: float = 0.0,
        okx_kline_futures: object | None = None,
    ) -> StrategyPipelineDecision:
        """Return BET or SKIP based on dual-asset momentum signal."""

        lock_at = int(round_t.lock_at)
        cutoff_ts_ms = (lock_at - self._cutoff_seconds) * 1000

        # Pre-signal filter: skip low-liquidity hours (03:00–04:59 UTC).
        hour_utc = (lock_at % 86400) // 3600
        if hour_utc in _LOW_LIQ_SKIP_HOURS:
            return self._skip("low_liquidity_hour_skip")

        if self._gate is not None:
            # Live/dry: fetch from OKX (use async-fetched data if available)
            result = self._gate.evaluate(
                cutoff_ts_ms=int(cutoff_ts_ms),
                kline_futures=okx_kline_futures,
            )
        else:
            # Backtest: use cached 1s klines
            result = self._evaluate_from_cache(
                epoch=int(round_t.epoch),
                cutoff_ts_ms=int(cutoff_ts_ms),
            )
            # Compute pools from round bets if not provided externally
            if pool_bull_bnb <= 0.0 and pool_bear_bnb <= 0.0 and round_t.bets:
                pool_bull_bnb, pool_bear_bnb = _pools_from_bets(round_t, lock_at)
        pool_total = pool_bull_bnb + pool_bear_bnb

        if result.skip_reason is not None and result.signal is None:
            return self._skip(str(result.skip_reason))
        if result.signal is None:
            return self._skip("gate_no_signal")

        # Payout floor filter: skip if payout on our side is too low.
        if pool_total > 0:
            our_side = pool_bull_bnb if result.signal == "Bull" else pool_bear_bnb
            if our_side > 0:
                pm = pool_total * (1.0 - self._treasury_fee_fraction) / our_side
                if pm < _MIN_OUR_PAYOUT:
                    return self._skip("payout_below_floor")

        bet_size = _compute_bet_size(
            signal=result.signal,
            tier=result.tier or "accel",
            btc_agrees=result.btc_agrees,
            btc_disagrees=result.btc_disagrees,
            pool_bull_bnb=pool_bull_bnb,
            pool_bear_bnb=pool_bear_bnb,
            treasury_fee_fraction=self._treasury_fee_fraction,
        )

        # Dilution cap: reduce bet if it would dilute our payout by > 12%.
        # Exact formula: max_bet = D * pool * our_side / ((1-D)*pool - our_side)
        if pool_total > 0:
            our_side = pool_bull_bnb if result.signal == "Bull" else pool_bear_bnb
            if our_side > 0:
                d = _MAX_PAYOUT_DILUTION
                denom = (1.0 - d) * pool_total - our_side
                if denom > 0:
                    max_bet = d * pool_total * our_side / denom
                    if bet_size > max_bet:
                        bet_size = max_bet

        if bet_size < self._min_bet_amount_bnb:
            return self._skip("bet_size_below_min")

        return self._bet(side=str(result.signal), size_bnb=float(bet_size))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evaluate_from_cache(
        self,
        *,
        epoch: int,
        cutoff_ts_ms: int,
    ) -> MomentumGateResult:
        """Run signal logic on cached klines (backtest path)."""
        bnb_klines = self._spot_klines_by_epoch.get(epoch)
        if bnb_klines is None or len(bnb_klines) == 0:
            return MomentumGateResult(
                signal=None, tier=None, btc_agrees=False, btc_disagrees=False,
                skip_reason="gate_no_spot_klines",
            )
        btc_klines = self._btc_klines_by_epoch.get(epoch)
        return compute_signal_from_klines(bnb_klines, btc_klines, cutoff_ts_ms)

    @staticmethod
    def _skip(reason: str) -> StrategyPipelineDecision:
        return StrategyPipelineDecision(
            action="SKIP",
            selected_strategy=None,
            bet_side=None,
            bet_size_bnb=0.0,
            expected_profit_bnb=0.0,
            selector_score_bnb=None,
            skip_reason=reason,
            p_bull=None,
        )

    @staticmethod
    def _bet(*, side: str, size_bnb: float) -> StrategyPipelineDecision:
        if side not in ("Bull", "Bear"):
            raise InvariantError(f"momentum_pipeline_invalid_side: {side}")
        return StrategyPipelineDecision(
            action="BET",
            selected_strategy="momentum_gate",
            bet_side=side,
            bet_size_bnb=float(size_bnb),
            expected_profit_bnb=0.0,
            selector_score_bnb=None,
            skip_reason=None,
            p_bull=None,
        )

    # ------------------------------------------------------------------
    # candidate_signals_for_open_round stub (called by audit/logging code)
    # ------------------------------------------------------------------

    def candidate_signals_for_open_round(self, *, round_t: Round) -> dict:
        return {}


def _pools_from_bets(round_t: Round, lock_at: int) -> tuple[float, float]:
    """Compute bull/bear pool BNB from round bets at or before lock_at."""
    bull_wei = 0
    bear_wei = 0
    for bet in round_t.bets:
        if int(bet.created_at) > lock_at:
            continue
        if bet.position == "Bull":
            bull_wei += int(bet.amount_wei)
        else:
            bear_wei += int(bet.amount_wei)
    return float(bull_wei) / float(BNB_WEI), float(bear_wei) / float(BNB_WEI)
