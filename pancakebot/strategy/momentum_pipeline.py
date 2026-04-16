"""MomentumOnlyPipeline: BTC primary signal plus ETH+SOL regime-2 with adaptive sizing.

Drives bet/skip decisions from MomentumGate output (live/dry) or cached klines
(backtest), applies pool-adaptive thresholds, a strong-signal pool bypass, a
payout floor, and continuous sizing scaled by signal strength.
"""

from __future__ import annotations

from dataclasses import dataclass

from pancakebot.constants import BNB_WEI
from pancakebot.util import InvariantError
from pancakebot.strategy.momentum_gate import (
    MomentumGate,
    MomentumGateConfig,
    MomentumGateResult,
    compute_signal_from_klines,
)
from pancakebot.types import Round

# Continuous adaptive sizing with payout boost.
# Signal strength sizing: frac = BASE_FRAC + SIZING_SLOPE * signal_strength
# Payout boost: frac *= max(0.5, 1.0 + PAYOUT_SLOPE * (payout - 2.0))
# Payout floor: skip rounds where payout on our side < MIN_PAYOUT.
# Validated: 5-fold +2.21/2k (5/5 positive), nested CV +1.59/2k (4/5 positive).
_BASE_FRAC = 0.04
_SIZING_SLOPE = 100        # scales with min(|r3|, |r7|, |r15|)
_PAYOUT_SLOPE = 1.0        # bet more when our side has high payout
_ETH_SIZING_WEIGHT = 0.3   # add ETH confirming strength * weight to signal_strength
_SOL_SIZING_WEIGHT = 0.3   # add SOL confirming strength * weight to signal_strength
_MIN_PAYOUT = 1.5          # skip if payout on our side < this
_MAX_FRAC = 0.30           # cap the pool fraction
_FLOOR_BNB = 0.01
_CAP_BNB = 2.0

# Pool filter: skip rounds with small pools (lose money due to dilution).
# Sweep showed pools >= 1.5 BNB are profitable with adaptive threshold.
# 5-fold: +2.75/2k (5/5), 48% more bets than pool>=2.0.
_MIN_POOL_BNB = 1.5

# Pool-adaptive threshold: stricter signal on small pools (lower WR),
# relaxed on large pools (higher WR, less dilution).
# Nested CV: +2.31/2k (4/5), consistent param selection across folds.
_SMALL_POOL_THRESH = 0.0002  # pool < _POOL_THRESH_BOUNDARY
_LARGE_POOL_THRESH = 0.0001  # pool >= _POOL_THRESH_BOUNDARY
_POOL_THRESH_BOUNDARY = 3.0

# Regime-2: ETH+SOL multi-TF agreement when BTC is silent.
# Fills flat periods where primary BTC signal doesn't fire.
# 5-fold: +2.83/2k (5/5) with separate sizing, 37% more bets.
# Regime-2 bets have 58.6% WR -- lower than primary, so bet smaller.
_REGIME2_ENABLED = True
_REGIME2_MIN_STRENGTH = 0.00015  # min(eth_strength, sol_strength) threshold
_REGIME2_BASE_FRAC = 0.02       # smaller base (regime-2 WR is lower)
_REGIME2_CAP_BNB = 0.5          # cap regime-2 bets at 0.5 BNB

# Strong-signal bypass: allow bets on pools below _MIN_POOL_BNB if the
# BTC signal is very strong. Pools must still exceed this floor.
# 5-fold: +2.85/2k (5/5), 113 extra bets at 58.4% WR.
_STRONG_BYPASS_STRENGTH = 0.0004  # BTC min(|r3|,|r7|,|r15|) threshold
_STRONG_BYPASS_POOL_FLOOR = 1.0   # minimum pool even for strong signals
_STRONG_BYPASS_BASE_FRAC = 0.03   # conservative sizing on small pools
_STRONG_BYPASS_CAP_BNB = 0.3      # small cap to limit dilution


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
    signal_strength: float,
    pool_bnb: float,
    our_side_bnb: float,
    base_frac: float = _BASE_FRAC,
    cap_bnb: float = _CAP_BNB,
) -> float:
    """Continuous adaptive sizing with payout-proportional boost.

    1. Signal-strength sizing: frac = base_frac + SIZING_SLOPE * signal_strength
    2. Payout boost: when our side has high payout (fewer bettors on our side),
       multiply frac by up to 2x. This is Kelly reasoning -- bet more when
       the odds are favorable.

    Regime-2 bets use smaller base_frac and cap (lower WR -> bet less).
    """
    if pool_bnb <= 0:
        return _FLOOR_BNB

    frac = min(base_frac + _SIZING_SLOPE * signal_strength, _MAX_FRAC)

    # Payout-proportional boost: bet more when our side has high payout.
    if our_side_bnb > 0:
        payout = pool_bnb * 0.97 / our_side_bnb  # 3% treasury fee
        payout_mult = max(0.5, 1.0 + _PAYOUT_SLOPE * (payout - 2.0))
        frac = min(frac * payout_mult, _MAX_FRAC)

    bet = pool_bnb * frac
    return max(_FLOOR_BNB, min(cap_bnb, bet))


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
        self._bnb_klines_by_epoch: dict[int, list[list]] = {}
        self._btc_klines_by_epoch: dict[int, list[list]] = {}
        self._eth_klines_by_epoch: dict[int, list[list]] = {}
        self._sol_klines_by_epoch: dict[int, list[list]] = {}

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

    # ------------------------------------------------------------------
    # Core decision
    # ------------------------------------------------------------------

    def decide_open_round(
        self,
        *,
        round_t: Round,
        pool_bull_bnb: float = 0.0,
        pool_bear_bnb: float = 0.0,
        okx_kline_futures: object | None = None,
    ) -> StrategyPipelineDecision:
        """Return BET or SKIP from BTC primary / ETH+SOL regime-2 signals."""

        if round_t.lock_at is None:
            raise InvariantError("round_lock_at_missing")
        lock_at = int(round_t.lock_at)
        cutoff_ts_ms = (lock_at - self._cutoff_seconds) * 1000

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

        if result.signal is not None:
            # Primary: BTC multi-TF fires
            # Pool-adaptive threshold
            min_thresh = _LARGE_POOL_THRESH if pool_total >= _POOL_THRESH_BOUNDARY \
                else _SMALL_POOL_THRESH
            if result.signal_strength >= min_thresh:
                signal_dir = result.signal
                effective_strength = result.signal_strength
                if result.eth_confirmation_strength > 0:
                    effective_strength += result.eth_confirmation_strength * _ETH_SIZING_WEIGHT
                if result.sol_confirmation_strength > 0:
                    effective_strength += result.sol_confirmation_strength * _SOL_SIZING_WEIGHT

        if signal_dir is None and _REGIME2_ENABLED:
            # Regime-2: ETH+SOL both fire same direction, BTC silent
            if (result.eth_signal is not None
                    and result.sol_signal is not None
                    and result.eth_signal == result.sol_signal):
                r2_str = min(result.eth_signal_strength, result.sol_signal_strength)
                if r2_str >= _REGIME2_MIN_STRENGTH:
                    signal_dir = result.eth_signal
                    effective_strength = (
                        result.eth_signal_strength * _ETH_SIZING_WEIGHT
                        + result.sol_signal_strength * _SOL_SIZING_WEIGHT
                    )
                    is_regime2 = True

        if signal_dir is None:
            return self._skip("gate_no_signal")

        # Pool filter: skip if visible pool is too small (dilution kills edge).
        # Exception: very strong primary signals can bypass on pools >= floor.
        is_strong_bypass = False
        if pool_total < _MIN_POOL_BNB:
            if (not is_regime2
                    and result.signal_strength >= _STRONG_BYPASS_STRENGTH
                    and pool_total >= _STRONG_BYPASS_POOL_FLOOR):
                is_strong_bypass = True
            else:
                return self._skip("pool_below_minimum")

        our_side = pool_bull_bnb if signal_dir == "Bull" else pool_bear_bnb

        # Payout floor: skip if payout on our side is too low.
        if our_side > 0 and pool_total > 0:
            payout = pool_total * (1.0 - self._treasury_fee_fraction) / our_side
            if payout < _MIN_PAYOUT:
                return self._skip("payout_below_floor")

        if is_regime2:
            bet_size = _compute_bet_size(
                signal_strength=effective_strength,
                pool_bnb=pool_total,
                our_side_bnb=our_side,
                base_frac=_REGIME2_BASE_FRAC,
                cap_bnb=_REGIME2_CAP_BNB,
            )
        elif is_strong_bypass:
            bet_size = _compute_bet_size(
                signal_strength=effective_strength,
                pool_bnb=pool_total,
                our_side_bnb=our_side,
                base_frac=_STRONG_BYPASS_BASE_FRAC,
                cap_bnb=_STRONG_BYPASS_CAP_BNB,
            )
        else:
            bet_size = _compute_bet_size(
                signal_strength=effective_strength,
                pool_bnb=pool_total,
                our_side_bnb=our_side,
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
                signal=None, tier=None, btc_agrees=False, btc_disagrees=False,
                skip_reason="gate_no_btc_klines",
            )
        eth_klines = self._eth_klines_by_epoch.get(epoch)
        sol_klines = self._sol_klines_by_epoch.get(epoch)
        return compute_signal_from_klines(
            btc_klines, cutoff_ts_ms,
            eth_klines=eth_klines, sol_klines=sol_klines,
        )

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
