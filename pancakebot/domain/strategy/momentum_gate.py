"""OKX 1m momentum gate.

At decision time (cutoff), fetches the last confirmed 1m BNB/USDT kline from
OKX and computes a simple return signal:

  ret_1m = (close_price / open_price) - 1

If ret_1m > threshold  -> signal "Bull"
If ret_1m < -threshold -> signal "Bear"
Otherwise              -> signal None (no opinion, pass through)

The gate is applied AFTER the strategy pipeline decision. Two modes:

  filter  (default): if the gate signal disagrees with the pipeline's bet_side,
                     veto the bet (return skip_reason). If gate has no opinion
                     (|ret_1m| < threshold), allow the bet through unchanged.

  override:          the gate signal replaces the pipeline's bet_side entirely.
                     Bet only when gate has a strong opinion (|ret_1m| >= threshold).

The gate never changes bet size, only direction or veto.
"""

from __future__ import annotations

from dataclasses import dataclass

from pancakebot.infra.okx_client import OkxClient
from pancakebot.core.errors import InvariantError
from pancakebot.core.logging import warn


@dataclass(frozen=True, slots=True)
class MomentumGateConfig:
    enabled: bool
    symbol: str                  # e.g. "BNB-USDT"
    threshold: float             # minimum |ret_1m| to act, e.g. 0.0001
    mode: str                    # "filter" or "override"
    max_staleness_seconds: int   # reject kline if older than this before cutoff
    bet_size_bnb: float = 0.05  # fixed stake per bet


@dataclass(frozen=True, slots=True)
class MomentumGateResult:
    signal: str | None     # "Bull", "Bear", or None
    ret_1m: float | None
    skip_reason: str | None   # set when gate vetoes
    kline_age_seconds: float | None


class MomentumGate:
    """Stateless OKX 1m momentum gate."""

    def __init__(self, *, config: MomentumGateConfig, okx_client: OkxClient) -> None:
        if str(config.mode) not in ("filter", "override"):
            raise InvariantError(f"momentum_gate_invalid_mode: {config.mode}")
        if float(config.threshold) <= 0:
            raise InvariantError("momentum_gate_threshold_must_be_positive")
        if int(config.max_staleness_seconds) <= 0:
            raise InvariantError("momentum_gate_max_staleness_must_be_positive")
        self._cfg = config
        self._client = okx_client

    @property
    def enabled(self) -> bool:
        return bool(self._cfg.enabled)

    def evaluate(
        self,
        *,
        cutoff_ts_ms: int,
        pipeline_bet_side: str | None,
    ) -> MomentumGateResult:
        """Evaluate the gate at cutoff time.

        Args:
            cutoff_ts_ms: cutoff timestamp in milliseconds (lock_at_ms - cutoff_seconds * 1000)
            pipeline_bet_side: the pipeline's proposed bet side ("Bull"/"Bear"), or None if SKIP

        Returns MomentumGateResult. If skip_reason is set, the runtime should skip.
        """
        if not bool(self._cfg.enabled):
            return MomentumGateResult(signal=None, ret_1m=None, skip_reason=None, kline_age_seconds=None)

        try:
            kline = self._client.fetch_last_confirmed_1m_kline(
                symbol=str(self._cfg.symbol),
                before_ts_ms=int(cutoff_ts_ms),
            )
        except Exception as e:
            warn("GATE", "OKX", "FETCH_FAILED", reason=str(e))
            # On fetch failure in filter mode: pass through (don't veto)
            # In override mode: skip (no signal to act on)
            if str(self._cfg.mode) == "override":
                return MomentumGateResult(
                    signal=None,
                    ret_1m=None,
                    skip_reason=f"momentum_gate_fetch_failed:{e}",
                    kline_age_seconds=None,
                )
            return MomentumGateResult(signal=None, ret_1m=None, skip_reason=None, kline_age_seconds=None)

        if kline is None:
            warn("GATE", "OKX", "NO_KLINE", cutoff_ts_ms=int(cutoff_ts_ms))
            if str(self._cfg.mode) == "override":
                return MomentumGateResult(
                    signal=None,
                    ret_1m=None,
                    skip_reason="momentum_gate_no_kline",
                    kline_age_seconds=None,
                )
            return MomentumGateResult(signal=None, ret_1m=None, skip_reason=None, kline_age_seconds=None)

        kline_close_ms = int(kline["close_time_ms"])
        age_seconds = float((int(cutoff_ts_ms) - int(kline_close_ms)) / 1000)

        if age_seconds > float(self._cfg.max_staleness_seconds):
            warn("GATE", "OKX", "STALE_KLINE", age_seconds=age_seconds,
                 max_staleness=int(self._cfg.max_staleness_seconds))
            if str(self._cfg.mode) == "override":
                return MomentumGateResult(
                    signal=None,
                    ret_1m=None,
                    skip_reason=f"momentum_gate_stale_kline:age={age_seconds:.1f}s",
                    kline_age_seconds=age_seconds,
                )
            return MomentumGateResult(signal=None, ret_1m=None, skip_reason=None, kline_age_seconds=age_seconds)

        open_price = float(kline["open_price"])
        close_price = float(kline["close_price"])

        if open_price <= 0:
            raise InvariantError(f"momentum_gate_kline_open_price_invalid: {open_price}")

        ret_1m = float((close_price / open_price) - 1.0)
        threshold = float(self._cfg.threshold)

        if ret_1m > threshold:
            signal: str | None = "Bull"
        elif ret_1m < -threshold:
            signal = "Bear"
        else:
            signal = None  # within threshold band, no opinion

        if str(self._cfg.mode) == "filter":
            # Veto only if signal disagrees with pipeline. No opinion = pass through.
            if signal is not None and pipeline_bet_side is not None:
                if str(signal) != str(pipeline_bet_side):
                    return MomentumGateResult(
                        signal=signal,
                        ret_1m=ret_1m,
                        skip_reason=(
                            f"momentum_gate_disagrees:"
                            f"gate={signal},pipeline={pipeline_bet_side},"
                            f"ret_1m={ret_1m:.6f}"
                        ),
                        kline_age_seconds=age_seconds,
                    )
            return MomentumGateResult(
                signal=signal, ret_1m=ret_1m, skip_reason=None, kline_age_seconds=age_seconds
            )

        # override mode: only bet when gate has a strong opinion
        if signal is None:
            return MomentumGateResult(
                signal=None,
                ret_1m=ret_1m,
                skip_reason=f"momentum_gate_no_signal:ret_1m={ret_1m:.6f}",
                kline_age_seconds=age_seconds,
            )

        return MomentumGateResult(
            signal=signal, ret_1m=ret_1m, skip_reason=None, kline_age_seconds=age_seconds
        )
