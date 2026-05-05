"""MomentumOnlyPipeline: BTC primary signal plus ETH+SOL regime-2 with adaptive sizing.

Drives bet/skip decisions from MomentumGate output (live/dry) or cached klines
(backtest), applies pool-adaptive thresholds, a payout floor, and continuous
sizing scaled by signal strength.

All strategy parameters live in ``pancakebot.config.StrategyConfig``
(TOML ``[strategy.*]``). Module-level constants here are intentionally
NONE: the previous _SIZING_SLOPE / _MAX_FRAC / _REGIME2_ENABLED constants
were extracted into config in the 2026-04-26 lean&clean refactor.
"""

from __future__ import annotations

from dataclasses import dataclass

from pancakebot.bankroll_tracker import BankrollTracker
from pancakebot.config import StrategyConfig
from pancakebot.constants import BNB_WEI
from pancakebot.util import InvariantError
from pancakebot.strategy.momentum_gate import (
    MomentumGate,
    MomentumGateConfig,
    MomentumGateResult,
    compute_signal_from_klines,
)
from pancakebot.types import Round


@dataclass(frozen=True, slots=True)
class StrategyPipelineDecision:
    """Normalized open-round strategy pipeline decision (momentum-only variant).

    Slim post-2026-04-26 schema: removed dead fields ``selected_strategy``,
    ``expected_profit_bnb``, ``selector_score_bnb``, ``p_bull``, and the
    six ``controller_*`` carryovers from the prior multi-strategy
    architecture. None of those were read in any decision path or written
    to a meaningful CSV column.
    """

    action: str
    bet_side: str | None
    bet_size_bnb: float
    skip_reason: str | None


def _compute_bet_size(
    *,
    signal_strength: float,
    pool_bnb: float,
    our_side_bnb: float,
    base_frac: float,
    cap_bnb: float,
    sizing_slope: float,
    max_frac: float,
    treasury_fee_fraction: float,
    min_bet_threshold_bnb: float,
    current_bankroll: float | None = None,
    max_bet_frac_of_bankroll: float = 1.0,
) -> float:
    """Continuous adaptive sizing with payout-proportional boost + bankroll cap.

    1. Signal-strength sizing: frac = base_frac + sizing_slope * signal_strength
    2. Payout boost (hardcoded slope=1.0 since 2026-04-26 lean&clean):
       payout = pool * (1 - treasury_fee_fraction) / our_side
       payout_mult = max(0.5, payout - 1.0)   # = max(0.5, 1.0 + (payout - 2.0))
       frac = min(frac * payout_mult, max_frac)
    3. bet = pool_bnb * frac
    4. Bankroll cap (when current_bankroll is not None): bet <=
       max_bet_frac_of_bankroll * current_bankroll.
    5. Absolute BNB cap (cap_bnb), then floor by min_bet_threshold_bnb.

    base_frac, cap_bnb are per-regime (btc_primary or eth_sol_fallback).
    sizing_slope, max_frac come from BtcPrimarySizingConfig (apply uniformly
    to both regimes). treasury_fee_fraction comes from on-chain contract
    constants (matched to the settlement code path). min_bet_threshold_bnb
    comes from Tier2SizingConfig. current_bankroll defaults to None
    (no bankroll cap).

    Removed parameters (2026-04-26):
      - payout_slope (hardcoded to 1.0; was misleading because cutoff-time
        payout != settlement-time payout).

    Renamed (2026-04-26):
      - floor_bnb -> min_bet_threshold_bnb (clearer semantics; the
        on-chain ``min_bet_amount_bnb`` checked at the call site is a
        separate concept).
    """
    if pool_bnb <= 0:
        return 0.0

    frac = min(base_frac + sizing_slope * signal_strength, max_frac)

    # Payout-proportional boost: bet more when our side has high payout.
    # Equivalent to the old payout_slope=1.0 default: max(0.5, 1.0 + (payout - 2.0)).
    if our_side_bnb > 0:
        payout = pool_bnb * (1.0 - treasury_fee_fraction) / our_side_bnb
        payout_mult = max(0.5, payout - 1.0)
        frac = min(frac * payout_mult, max_frac)

    bet = pool_bnb * frac

    # Bankroll cap (risk control). When current_bankroll is None, no-op.
    if current_bankroll is not None and current_bankroll > 0:
        bet = min(bet, max_bet_frac_of_bankroll * current_bankroll)

    return max(min_bet_threshold_bnb, min(cap_bnb, bet))


class MomentumOnlyPipeline:
    """Momentum-only pipeline: satisfies the StrategyPipeline interface.

    In live/dry mode pass `gate` (a MomentumGate backed by OKX client).
    In backtest mode leave `gate=None`; the pipeline uses cached 1s klines.
    """

    def __init__(
        self,
        *,
        config: MomentumGateConfig,
        strategy_config: StrategyConfig,
        gate: MomentumGate | None,
        cutoff_seconds: int,
        min_bet_amount_bnb: float,
        treasury_fee_fraction: float,
        bankroll_tracker: BankrollTracker | None = None,
    ) -> None:
        self._cfg = config
        self._strategy = strategy_config
        self._gate = gate
        self._cutoff_seconds = int(cutoff_seconds)
        self._min_bet_amount_bnb = float(min_bet_amount_bnb)
        self._treasury_fee_fraction = float(treasury_fee_fraction)
        self._last_settled_epoch: int | None = None
        # Derived candle count used for backtest validation.
        self._candle_count = max(strategy_config.gate.mtf_lookbacks) + 1
        # Backtest: 1s klines per epoch {epoch: [[ts_ms, o, h, l, c, vol], ...]}
        self._bnb_klines_by_epoch: dict[int, list[list]] = {}
        self._btc_klines_by_epoch: dict[int, list[list]] = {}
        self._eth_klines_by_epoch: dict[int, list[list]] = {}
        self._sol_klines_by_epoch: dict[int, list[list]] = {}
        # Risk tracker: None disables risk checks (backward-compatible). When
        # present, decide_open_round runs pre-signal gates (min_bankroll,
        # cooldown, drawdown-from-peak) and _compute_bet_size applies the
        # max_bet_frac_of_bankroll cap. Callers record settlements via
        # pipeline.record_settlement(bankroll, start_at).
        self._bankroll_tracker: BankrollTracker | None = bankroll_tracker

    # ------------------------------------------------------------------
    # Required interface: StrategyPipeline-compatible
    # ------------------------------------------------------------------

    @property
    def last_settled_epoch(self) -> int | None:
        return self._last_settled_epoch

    @property
    def router_mode(self) -> str:
        return "momentum_gate"

    def refresh_bnb_klines(self, *, bnb_klines_by_epoch: dict[int, list[list]]) -> None:
        """Load pre-fetched BNB 1s kline arrays keyed by epoch (backtest mode)."""
        self._bnb_klines_by_epoch = dict(bnb_klines_by_epoch)

    def refresh_btc_klines(self, *, btc_klines_by_epoch: dict[int, list[list]]) -> None:
        """Load pre-fetched BTC 1s kline arrays keyed by epoch (backtest mode)."""
        self._btc_klines_by_epoch = dict(btc_klines_by_epoch)

    def refresh_eth_klines(self, *, eth_klines_by_epoch: dict[int, list[list]]) -> None:
        """Load pre-fetched ETH 1s kline arrays keyed by epoch (backtest mode)."""
        self._eth_klines_by_epoch = dict(eth_klines_by_epoch)

    def refresh_sol_klines(self, *, sol_klines_by_epoch: dict[int, list[list]]) -> None:
        """Load pre-fetched SOL 1s kline arrays keyed by epoch (backtest mode)."""
        self._sol_klines_by_epoch = dict(sol_klines_by_epoch)

    def settle_closed_rounds(self, *, rounds: list[Round]) -> None:
        """Track the last settled epoch (no ML state to update)."""
        for r in sorted(rounds, key=lambda x: int(x.epoch)):
            epoch = int(r.epoch)
            if self._last_settled_epoch is None or epoch > int(self._last_settled_epoch):
                self._last_settled_epoch = epoch

    def bootstrap_from_closed_rounds(self, *, rounds: list[Round]) -> None:
        """Set last_settled_epoch from the warmup batch. No ML state."""
        self.settle_closed_rounds(rounds=rounds)

    def record_settlement(self, *, bankroll: float, start_at: int) -> None:
        """Forward a post-settlement bankroll snapshot to the tracker (if wired).

        Caller responsibility: call this AFTER the round's bankroll has been
        updated (post bet-debit + settle-credit), using the round's start_at
        as the timestamp anchor for the rolling window.
        """
        if self._bankroll_tracker is not None:
            self._bankroll_tracker.record_settlement(bankroll, start_at)

    def set_bankroll_tracker(self, tracker: BankrollTracker | None) -> None:
        """Wire (or rewire) the bankroll tracker after pipeline construction.

        Needed for dry/live runtime where the initial bankroll is resolved
        AFTER the pipeline is built (bankroll depends on on-chain wallet
        balance or persisted dry state). Pass None to disable risk checks.
        """
        self._bankroll_tracker = tracker

    # ------------------------------------------------------------------
    # Core decision
    # ------------------------------------------------------------------

    def decide_open_round(
        self,
        *,
        round_t: Round,
        pool_bull_bnb: float = 0.0,
        pool_bear_bnb: float = 0.0,
    ) -> StrategyPipelineDecision:
        """Return BET or SKIP from BTC primary / ETH+SOL regime-2 signals."""

        if round_t.lock_at is None:
            raise InvariantError("round_lock_at_missing")
        lock_at = int(round_t.lock_at)
        cutoff_ts_ms = (lock_at - self._cutoff_seconds) * 1000

        # Risk checks (when tracker is wired). Runs BEFORE signal computation so
        # we don't waste kline fetches on paused / low-bankroll rounds.
        # When tracker is None, this block is a complete no-op.
        if self._bankroll_tracker is not None:
            risk = self._strategy.risk
            start_at = int(round_t.start_at)
            # Check 1: cooldown paused. Tick the counter on every paused round
            # observed by the pipeline so the cooldown actually winds down.
            if self._bankroll_tracker.is_paused(start_at):
                self._bankroll_tracker.tick_cooldown()
                return self._skip("risk_cooldown_active")
            # Check 2: bankroll below minimum -- skip without firing cooldown.
            current = self._bankroll_tracker.current_bankroll()
            if current < risk.min_bankroll_bnb:
                return self._skip("risk_bankroll_below_min")
            # Check 3: drawdown from peak. If >= threshold, fire cooldown.
            peak = self._bankroll_tracker.peak_bankroll(start_at)
            if peak > 0:
                dd_frac = (peak - current) / peak
                if dd_frac >= risk.max_drawdown_frac_from_peak:
                    self._bankroll_tracker.set_paused(risk.cooldown_rounds, start_at)
                    return self._skip("risk_drawdown_breaker_fired")

        if self._gate is not None:
            # Live/dry: gate fetches BTC/ETH/SOL/BNB klines in parallel via
            # OKX REST and computes the signal off the returned arrays.
            result = self._gate.evaluate(
                lock_at_ms=lock_at * 1000,
            )
        else:
            # Backtest: use cached 1s klines
            result = self._evaluate_from_cache(
                epoch=int(round_t.epoch),
                cutoff_ts_ms=int(cutoff_ts_ms),
            )
            # Compute pools from round bets if not provided externally.
            if pool_bull_bnb <= 0.0 and pool_bear_bnb <= 0.0 and round_t.bets:
                from pancakebot.constants import POOL_CUTOFF_SECONDS
                pool_cutoff_ts = lock_at - POOL_CUTOFF_SECONDS
                pool_bull_bnb, pool_bear_bnb = _pools_from_bets(round_t, pool_cutoff_ts)
        pool_total = pool_bull_bnb + pool_bear_bnb

        # Determine signal source: primary (BTC) or regime-2 (ETH+SOL)
        signal_dir = None
        effective_strength = 0.0
        is_regime2 = False
        t2_w = self._strategy.tier2_sizing.eth_sol_sizing_weight

        if result.signal is not None:
            # Primary: BTC multi-TF fires
            # Pool-adaptive admission: small-pool stricter threshold
            # (large pools rely on the gate's mtf_threshold which already fired).
            bt = self._strategy.btc_primary.threshold
            small_pool_ok = (
                pool_total >= bt.pool_size_boundary_bnb
                or result.signal_strength >= bt.small_pool
            )
            if small_pool_ok:
                signal_dir = result.signal
                effective_strength = result.signal_strength
                if result.eth_confirmation_strength > 0:
                    effective_strength += result.eth_confirmation_strength * t2_w
                if result.sol_confirmation_strength > 0:
                    effective_strength += result.sol_confirmation_strength * t2_w

        if signal_dir is None:
            # Regime-2: ETH+SOL both fire same direction, BTC silent.
            if (result.eth_signal is not None
                    and result.sol_signal is not None
                    and result.eth_signal == result.sol_signal):
                r2_str = min(result.eth_signal_strength, result.sol_signal_strength)
                if r2_str >= self._strategy.eth_sol_fallback.signal.min_strength:
                    signal_dir = result.eth_signal
                    effective_strength = (
                        result.eth_signal_strength * t2_w
                        + result.sol_signal_strength * t2_w
                    )
                    is_regime2 = True

        if signal_dir is None:
            # Propagate the gate's specific skip reason if it set one
            # (e.g. ``kline_fetch_transient_failure`` when ETH/SOL fetch
            # returned ``got_15_expected_16`` and the regime-2 fallback
            # couldn't fire either). Falling through to ``gate_no_signal``
            # silently rebrands transient kline failures and hides
            # data-availability issues from the cycle_audit.
            return self._skip(result.skip_reason or "gate_no_signal")

        # Pool filter: skip if visible pool is too small (dilution kills edge).
        if pool_total < self._strategy.pool_filter.min_pool_bnb:
            return self._skip("pool_below_minimum")

        our_side = pool_bull_bnb if signal_dir == "Bull" else pool_bear_bnb

        # Payout floor: skip if payout on our side is too low.
        if our_side > 0 and pool_total > 0:
            payout = pool_total * (1.0 - self._treasury_fee_fraction) / our_side
            if payout < self._strategy.pool_filter.min_payout:
                return self._skip("payout_below_floor")

        # Bankroll for the bankroll cap kwarg (None when no tracker -> cap disabled).
        br_current = (
            self._bankroll_tracker.current_bankroll()
            if self._bankroll_tracker is not None else None
        )
        br_cap_frac = self._strategy.risk.max_bet_frac_of_bankroll
        bt_sz = self._strategy.btc_primary.sizing
        t2 = self._strategy.tier2_sizing

        if is_regime2:
            es_sizing = self._strategy.eth_sol_fallback.sizing
            bet_size = _compute_bet_size(
                signal_strength=effective_strength,
                pool_bnb=pool_total,
                our_side_bnb=our_side,
                base_frac=es_sizing.base_fraction,
                cap_bnb=self._strategy.risk.max_bet_bnb_eth_sol_fallback,
                sizing_slope=bt_sz.sizing_slope,
                max_frac=bt_sz.max_frac,
                treasury_fee_fraction=self._treasury_fee_fraction,
                min_bet_threshold_bnb=t2.min_bet_threshold_bnb,
                current_bankroll=br_current,
                max_bet_frac_of_bankroll=br_cap_frac,
            )
        else:
            bet_size = _compute_bet_size(
                signal_strength=effective_strength,
                pool_bnb=pool_total,
                our_side_bnb=our_side,
                base_frac=bt_sz.base_fraction,
                cap_bnb=self._strategy.risk.max_bet_bnb_btc_primary,
                sizing_slope=bt_sz.sizing_slope,
                max_frac=bt_sz.max_frac,
                treasury_fee_fraction=self._treasury_fee_fraction,
                min_bet_threshold_bnb=t2.min_bet_threshold_bnb,
                current_bankroll=br_current,
                max_bet_frac_of_bankroll=br_cap_frac,
            )

        if bet_size < self._min_bet_amount_bnb:
            return self._skip("bet_size_below_min")

        return self._bet(side=str(signal_dir), size_bnb=float(bet_size))

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
        btc_klines = self._btc_klines_by_epoch.get(epoch)
        if btc_klines is None or len(btc_klines) == 0:
            return MomentumGateResult(
                signal=None, tier=None,
                skip_reason="gate_no_btc_klines",
            )
        eth_klines = self._eth_klines_by_epoch.get(epoch)
        sol_klines = self._sol_klines_by_epoch.get(epoch)
        gate = self._strategy.gate
        return compute_signal_from_klines(
            btc_klines, cutoff_ts_ms,
            mtf_lookbacks=gate.mtf_lookbacks,
            mtf_threshold=gate.mtf_threshold,
            candle_count=self._candle_count,
            eth_klines=eth_klines, sol_klines=sol_klines,
        )

    @staticmethod
    def _skip(reason: str) -> StrategyPipelineDecision:
        return StrategyPipelineDecision(
            action="SKIP",
            bet_side=None,
            bet_size_bnb=0.0,
            skip_reason=reason,
        )

    @staticmethod
    def _bet(*, side: str, size_bnb: float) -> StrategyPipelineDecision:
        if side not in ("Bull", "Bear"):
            raise InvariantError(f"momentum_pipeline_invalid_side: {side}")
        return StrategyPipelineDecision(
            action="BET",
            bet_side=side,
            bet_size_bnb=float(size_bnb),
            skip_reason=None,
        )


def _pools_from_bets(round_t: Round, cutoff_ts: int) -> tuple[float, float]:
    """Compute bull/bear pool BNB from bets placed strictly before cutoff_ts.

    Uses cutoff_ts (lock_at - POOL_CUTOFF_SECONDS) to match what the live
    bot sees.  Strict < avoids boundary ambiguity between The Graph's
    createdAt and BSC block timestamps.
    """
    bull_wei = 0
    bear_wei = 0
    for bet in round_t.bets:
        if int(bet.created_at) >= cutoff_ts:
            continue
        if bet.position == "Bull":
            bull_wei += int(bet.amount_wei)
        else:
            bear_wei += int(bet.amount_wei)
    return float(bull_wei) / float(BNB_WEI), float(bear_wei) / float(BNB_WEI)
