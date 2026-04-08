"""Momentum-only strategy pipeline.

Replaces the complex dislocation/router/ML pipeline with a single signal:
the last confirmed 1m BNB/USDT return on OKX at cutoff time.

  ret_1m = (close / open) - 1 of the last closed 1m kline before cutoff

  If ret_1m >  threshold  → BET Bull
  If ret_1m < -threshold  → BET Bear
  Otherwise               → SKIP

In live/dry mode the kline is fetched live from OKX via MomentumGate.
In backtest mode the kline is looked up from the in-memory klines cache
(which must be populated from OKX historical data, i.e. klines_okx.jsonl).
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass

from pancakebot.core.errors import InvariantError
from pancakebot.core.logging import warn
from pancakebot.domain.strategy.momentum_gate import MomentumGate, MomentumGateConfig
from pancakebot.domain.types import Kline, Round


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


@dataclass(frozen=True, slots=True)
class _BacktestKlineResult:
    ret_1m: float | None
    signal: str | None      # "Bull", "Bear", or None
    skip_reason: str | None
    kline_age_seconds: float | None


def _compute_from_kline(kline: dict | Kline, threshold: float) -> _BacktestKlineResult:
    """Compute ret_1m signal from a kline object."""
    if isinstance(kline, Kline):
        open_price = float(kline.open_price)
        close_price = float(kline.close_price)
        close_time_ms = int(kline.close_time_ms)
    else:
        open_price = float(kline["open_price"])
        close_price = float(kline["close_price"])
        close_time_ms = int(kline["close_time_ms"])

    if open_price <= 0:
        return _BacktestKlineResult(ret_1m=None, signal=None,
                                    skip_reason="momentum_gate_invalid_open_price",
                                    kline_age_seconds=None)

    ret_1m = float((close_price / open_price) - 1.0)

    if ret_1m > float(threshold):
        signal: str | None = "Bull"
    elif ret_1m < -float(threshold):
        signal = "Bear"
    else:
        signal = None

    return _BacktestKlineResult(ret_1m=ret_1m, signal=signal, skip_reason=None, kline_age_seconds=None)


class MomentumOnlyPipeline:
    """Momentum-only pipeline: satisfies the StrategyPipeline interface.

    In live/dry mode pass `gate` (a MomentumGate backed by OKX client).
    In backtest mode leave `gate=None`; the pipeline uses `klines_cache` instead.
    """

    def __init__(
        self,
        *,
        config: MomentumGateConfig,
        gate: MomentumGate | None,
        cutoff_seconds: int,
        min_bet_amount_bnb: float,
    ) -> None:
        self._cfg = config
        self._gate = gate  # None in backtest
        self._cutoff_seconds = int(cutoff_seconds)
        self._min_bet_amount_bnb = float(min_bet_amount_bnb)
        self._last_settled_epoch: int | None = None
        # In backtest, klines are refreshed via refresh_klines.
        self._klines: list[Kline] = []
        self._kline_open_times: list[int] = []

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

    def refresh_klines(self, *, klines: list[Kline]) -> None:
        """Update the local kline cache (used in backtest mode)."""
        self._klines = list(klines)
        self._kline_open_times = [int(k.open_time_ms) for k in self._klines]

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
        return {"last_settled_epoch": self._last_settled_epoch}

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
    ) -> StrategyPipelineDecision:
        """Return BET or SKIP based on OKX 1m momentum signal."""

        bet_size_bnb = float(self._cfg.bet_size_bnb)
        if bet_size_bnb < float(self._min_bet_amount_bnb):
            return self._skip("bet_size_below_min_bet_amount")

        lock_at = int(round_t.lock_at)
        cutoff_ts_s = lock_at - self._cutoff_seconds
        cutoff_ts_ms = cutoff_ts_s * 1000

        if self._gate is not None:
            # Live/dry: fetch from OKX
            result = self._gate.evaluate(
                cutoff_ts_ms=int(cutoff_ts_ms),
                pipeline_bet_side=None,
            )
            if result.skip_reason is not None:
                return self._skip(str(result.skip_reason))
            if result.signal is None:
                return self._skip(
                    f"momentum_gate_no_signal:ret_1m={result.ret_1m:.6f}"
                    if result.ret_1m is not None
                    else "momentum_gate_no_signal"
                )
            return self._bet(side=str(result.signal), size_bnb=bet_size_bnb)

        # Backtest: look up from klines cache
        return self._decide_from_klines_cache(
            cutoff_ts_ms=int(cutoff_ts_ms),
            bet_size_bnb=bet_size_bnb,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _decide_from_klines_cache(
        self,
        *,
        cutoff_ts_ms: int,
        bet_size_bnb: float,
    ) -> StrategyPipelineDecision:
        if not self._klines:
            return self._skip("momentum_gate_no_klines_in_cache")

        # Find last kline whose open_time_ms < cutoff_ts_ms
        # (strictly before cutoff — the kline must be fully closed)
        idx = bisect_right(self._kline_open_times, cutoff_ts_ms - 1) - 1
        if idx < 0:
            return self._skip("momentum_gate_no_kline_before_cutoff")

        kline = self._klines[idx]
        close_time_ms = int(kline.close_time_ms)
        age_s = float((int(cutoff_ts_ms) - close_time_ms) / 1000)

        if age_s > float(self._cfg.max_staleness_seconds):
            warn("GATE", "BACKTEST", "STALE_KLINE", age_seconds=age_s)
            return self._skip(f"momentum_gate_stale_kline:age={age_s:.1f}s")

        result = _compute_from_kline(kline, float(self._cfg.threshold))
        if result.skip_reason is not None:
            return self._skip(result.skip_reason)
        if result.signal is None:
            return self._skip(
                f"momentum_gate_no_signal:ret_1m={result.ret_1m:.6f}"
                if result.ret_1m is not None
                else "momentum_gate_no_signal"
            )
        return self._bet(side=str(result.signal), size_bnb=bet_size_bnb)

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
