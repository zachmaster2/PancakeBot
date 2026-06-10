"""Load AppConfig and BacktestConfig from a TOML file and read required env vars."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomllib
from dotenv import load_dotenv

from pancakebot.util import InvariantError


# -- Environment helpers ------------------------------------------------------

def load_env() -> None:
    """Load .env into process environment."""
    load_dotenv()


def require_env(name: str) -> str:
    v = os.getenv(name)
    if v is None or not str(v).strip():
        raise InvariantError(f"missing_env_var: {name}")
    return str(v).strip()


# -- Backtest config ----------------------------------------------------------

@dataclass(frozen=True, slots=True)
class BacktestConfig:
    """Backtest configuration."""

    backtest_round_count: int
    initial_bankroll_bnb: float
    epoch_start: int | None = None
    epoch_end: int | None = None

    def validate(self) -> None:
        if not isinstance(self.backtest_round_count, int):
            raise InvariantError("backtest_round_count_not_int")
        if self.backtest_round_count <= 0:
            raise InvariantError("backtest_round_count_must_be_positive")

        if not isinstance(self.initial_bankroll_bnb, (int, float)):
            raise InvariantError("backtest_initial_bankroll_bnb_not_number")
        if self.initial_bankroll_bnb <= 0.0:
            raise InvariantError("backtest_initial_bankroll_bnb_must_be_positive")

        if self.epoch_start is not None and self.epoch_end is not None:
            if self.epoch_start > self.epoch_end:
                raise InvariantError("backtest_epoch_start_after_epoch_end")


# -- Strategy config ----------------------------------------------------------
#
# 14 knobs exposed for TOML configuration. Four sections match the strategy's
# logical layers: pool-admission filters, BTC primary signal, ETH+SOL fallback,
# and the strong-signal bypass. Other strategy values (multi-TF lookbacks,
# sizing slopes, cross-pair weights) remain as module-level constants in
# pancakebot/strategy/momentum_pipeline.py — they're more like algorithm
# identity than experiment knobs.


@dataclass(frozen=True, slots=True)
class PoolFilterConfig:
    """Pool-admission filters applied before any signal evaluation."""
    min_pool_bnb_at_cutoff: float
    min_payout_multiple_at_cutoff: float


@dataclass(frozen=True, slots=True)
class BtcPrimaryThresholdConfig:
    """BTC primary small-pool admission threshold (large pools use gate threshold).

    For pools >= ``pool_size_boundary_bnb`` the gate's
    ``GateConfig.mtf_min_return_threshold`` is the binding constraint (it already
    fired). For pools below, signal_strength must additionally exceed
    ``small_pool_min_signal_strength`` (a stricter cutoff that excludes weak signals on
    small / dilution-prone pools).
    """
    small_pool_min_signal_strength: float
    pool_size_boundary_bnb: float


@dataclass(frozen=True, slots=True)
class BtcPrimarySizingConfig:
    """BTC primary bet sizing.

    - base_pool_fraction: starting fraction-of-pool stake.
    - pool_fraction_slope: gain on signal_strength; final frac =
      base_pool_fraction + pool_fraction_slope * signal_strength, capped at max_pool_fraction.
    - max_pool_fraction: hard cap on the computed pool fraction, applied
      before per-bet absolute caps (in RiskConfig).
    """
    base_pool_fraction: float
    pool_fraction_slope: float
    max_pool_fraction: float


@dataclass(frozen=True, slots=True)
class BtcPrimaryConfig:
    threshold: BtcPrimaryThresholdConfig
    sizing: BtcPrimarySizingConfig


@dataclass(frozen=True, slots=True)
class EthSolFallbackSignalConfig:
    """ETH+SOL fallback signal (fires when BTC primary is silent)."""
    min_signal_strength: float


@dataclass(frozen=True, slots=True)
class EthSolFallbackSizingConfig:
    """ETH+SOL fallback bet sizing. (max_bet_bnb moved to RiskConfig.)"""
    base_pool_fraction: float


@dataclass(frozen=True, slots=True)
class EthSolFallbackConfig:
    signal: EthSolFallbackSignalConfig
    sizing: EthSolFallbackSizingConfig


@dataclass(frozen=True, slots=True)
class GateConfig:
    """Multi-TF momentum gate parameters.

    - mtf_lookbacks: tuple of integer-second lookbacks; all must agree
      in direction for the gate to fire. Default (3, 7, 15) — 3-second,
      7-second, 15-second returns.
    - mtf_min_return_threshold: minimum min(|return|) across the lookbacks for
      the gate to fire. The pipeline may apply a stricter pool-adaptive
      threshold via BtcPrimaryThresholdConfig.

    Previously module constants (_MTF_LOOKBACKS, _MTF_THRESH) in
    pancakebot/strategy/momentum_gate.py.
    """
    mtf_lookbacks: tuple[int, ...]
    mtf_min_return_threshold: float


@dataclass(frozen=True, slots=True)
class Tier2SizingConfig:
    """Cross-regime sizing knobs.

    - eth_sol_signal_weight: coefficient applied to BOTH ETH and SOL
      signal strengths in the BTC-primary confirmation boost AND in the
      regime-2 (ETH+SOL-only) effective-strength composition.
      Symmetric by design (collapsed from former separate eth/sol
      weights, which were always swept together at identical values).

    - min_bet_threshold_bnb: lower bound applied at the end of
      ``_compute_bet_size`` (``max(min_bet_threshold_bnb, min(cap_bnb, bet))``).
      Default 0.01 BNB. Bit-identical with prior behaviour because every
      observed computed bet on the canonical baseline already exceeds
      0.01 (Phase-2 ablation 2026-04-23). Distinct from the on-chain
      ``min_bet_amount_bnb`` (~0.001) which is the contract floor checked
      at the call-site.

    Removals (2026-04-26 lean&clean refactor):
      - payout_slope: removed; payout-proportional boost is now
        hardcoded with slope=1.0 (the default). The "payout at cutoff"
        signal was misleading because settlement-time payout differs.
      - eth_sizing_weight / sol_sizing_weight: collapsed into the single
        symmetric eth_sol_signal_weight above.
    """
    eth_sol_signal_weight: float
    min_bet_threshold_bnb: float


@dataclass(frozen=True, slots=True)
class RiskConfig:
    """Risk controls applied by MomentumOnlyPipeline when a BankrollTracker is active.

    - max_bet_fraction_of_bankroll: stake <= this fraction of current bankroll.
    - min_bankroll_bnb_to_bet: skip all bets while bankroll < this.
    - max_drawdown_fraction_from_peak: fire cooldown when (peak-current)/peak >= this.
      Set to 1.0 to disable the circuit breaker.
    - cooldown_rounds: number of rounds to pause after drawdown breaker fires.
    - drawdown_peak_window_days: rolling window (days) for the drawdown-from-peak calculation
      when ``drawdown_peak_mode == "rolling_7d"``.
    - drawdown_peak_mode: peak-tracking semantics for the drawdown breaker.
      ``"rolling_7d"`` (default) uses a rolling window of ``drawdown_peak_window_days``.
      ``"absolute_ratchet"`` uses an absolute-since-launch peak that
      monotonically only goes up — catches slow drains the rolling
      window misses.
    - max_bet_bnb_btc_primary: absolute BNB cap for BTC primary bets.
    - max_bet_bnb_eth_sol_fallback: absolute BNB cap for regime-2 ETH+SOL bets.
    """
    max_bet_fraction_of_bankroll: float
    min_bankroll_bnb_to_bet: float
    max_drawdown_fraction_from_peak: float
    cooldown_rounds: int
    drawdown_peak_window_days: int
    max_bet_bnb_btc_primary: float
    max_bet_bnb_eth_sol_fallback: float
    drawdown_peak_mode: str = "rolling_7d"


@dataclass(frozen=True, slots=True)
class StrategyConfig:
    pool_filter: PoolFilterConfig
    gate: GateConfig
    btc_primary: BtcPrimaryConfig
    eth_sol_fallback: EthSolFallbackConfig
    tier2_sizing: Tier2SizingConfig
    risk: RiskConfig

    def validate(self) -> None:
        """Assert invariants; raise InvariantError on any violation."""
        pf = self.pool_filter
        if pf.min_pool_bnb_at_cutoff <= 0.0:
            raise InvariantError("strategy_pool_filter_min_pool_bnb_at_cutoff_must_be_positive")
        if pf.min_payout_multiple_at_cutoff < 1.0:
            raise InvariantError("strategy_pool_filter_min_payout_multiple_at_cutoff_must_be_at_least_1")

        g = self.gate
        if not g.mtf_lookbacks:
            raise InvariantError("strategy_gate_mtf_lookbacks_must_be_non_empty")
        if any(not isinstance(lb, int) or lb <= 0 for lb in g.mtf_lookbacks):
            raise InvariantError("strategy_gate_mtf_lookbacks_must_be_positive_ints")
        if g.mtf_min_return_threshold <= 0.0:
            raise InvariantError("strategy_gate_mtf_min_return_threshold_must_be_positive")

        bt = self.btc_primary.threshold
        if bt.small_pool_min_signal_strength <= 0.0:
            raise InvariantError("strategy_btc_primary_threshold_small_pool_min_signal_strength_must_be_positive")
        if bt.pool_size_boundary_bnb <= 0.0:
            raise InvariantError("strategy_btc_primary_threshold_pool_size_boundary_bnb_must_be_positive")

        bs = self.btc_primary.sizing
        if not (0.0 < bs.base_pool_fraction < 1.0):
            raise InvariantError("strategy_btc_primary_sizing_base_pool_fraction_out_of_range")
        if bs.pool_fraction_slope < 0.0:
            raise InvariantError("strategy_btc_primary_sizing_slope_must_be_non_negative")
        if not (0.0 < bs.max_pool_fraction <= 1.0):
            raise InvariantError("strategy_btc_primary_sizing_max_pool_fraction_out_of_range")

        es = self.eth_sol_fallback.signal
        if es.min_signal_strength <= 0.0:
            raise InvariantError("strategy_eth_sol_fallback_signal_min_signal_strength_must_be_positive")

        ez = self.eth_sol_fallback.sizing
        if not (0.0 < ez.base_pool_fraction < 1.0):
            raise InvariantError("strategy_eth_sol_fallback_sizing_base_pool_fraction_out_of_range")

        t2 = self.tier2_sizing
        if t2.eth_sol_signal_weight < 0.0:
            raise InvariantError("strategy_tier2_sizing_eth_sol_signal_weight_must_be_non_negative")
        if t2.min_bet_threshold_bnb < 0.0:
            raise InvariantError("strategy_tier2_sizing_min_bet_threshold_bnb_must_be_non_negative")

        rk = self.risk
        if not (0.0 < rk.max_bet_fraction_of_bankroll <= 1.0):
            raise InvariantError("strategy_risk_max_bet_fraction_of_bankroll_out_of_range")
        if rk.min_bankroll_bnb_to_bet < 0.0:
            raise InvariantError("strategy_risk_min_bankroll_bnb_to_bet_must_be_non_negative")
        if not (0.0 < rk.max_drawdown_fraction_from_peak <= 1.0):
            raise InvariantError("strategy_risk_max_drawdown_fraction_from_peak_out_of_range")
        if rk.cooldown_rounds < 0:
            raise InvariantError("strategy_risk_cooldown_rounds_must_be_non_negative")
        if rk.drawdown_peak_window_days <= 0:
            raise InvariantError("strategy_risk_drawdown_peak_window_days_must_be_positive")
        if rk.drawdown_peak_mode not in ("rolling_7d", "absolute_ratchet"):
            raise InvariantError(
                f"strategy_risk_drawdown_peak_mode_invalid: {rk.drawdown_peak_mode!r} "
                "(expected 'rolling_7d' or 'absolute_ratchet')"
            )
        if rk.max_bet_bnb_btc_primary <= 0.0:
            raise InvariantError("strategy_risk_max_bet_bnb_btc_primary_must_be_positive")
        if rk.max_bet_bnb_eth_sol_fallback <= 0.0:
            raise InvariantError("strategy_risk_max_bet_bnb_eth_sol_fallback_must_be_positive")


# Default strategy values — match the module-level constants they replaced in
# pancakebot/strategy/momentum_pipeline.py, so a config.toml without any
# [strategy.*] sections reproduces the pre-refactor behavior exactly.
_DEFAULT_STRATEGY = StrategyConfig(
    pool_filter=PoolFilterConfig(min_pool_bnb_at_cutoff=1.5, min_payout_multiple_at_cutoff=1.5),
    gate=GateConfig(
        mtf_lookbacks=(3, 7, 15),
        mtf_min_return_threshold=0.0001,
    ),
    btc_primary=BtcPrimaryConfig(
        threshold=BtcPrimaryThresholdConfig(
            small_pool_min_signal_strength=0.0002,
            pool_size_boundary_bnb=3.0,
        ),
        sizing=BtcPrimarySizingConfig(
            base_pool_fraction=0.04,
            pool_fraction_slope=100.0,
            max_pool_fraction=0.30,
        ),
    ),
    eth_sol_fallback=EthSolFallbackConfig(
        signal=EthSolFallbackSignalConfig(min_signal_strength=0.00015),
        sizing=EthSolFallbackSizingConfig(base_pool_fraction=0.02),
    ),
    tier2_sizing=Tier2SizingConfig(
        eth_sol_signal_weight=0.3,
        min_bet_threshold_bnb=0.01,
    ),
    risk=RiskConfig(
        max_bet_fraction_of_bankroll=0.05,
        min_bankroll_bnb_to_bet=0.20,
        max_drawdown_fraction_from_peak=0.15,
        cooldown_rounds=72,
        drawdown_peak_window_days=7,
        max_bet_bnb_btc_primary=2.0,
        max_bet_bnb_eth_sol_fallback=0.5,
        drawdown_peak_mode="rolling_7d",
    ),
)


# -- App config ---------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class AppConfig:
    """User-facing configuration loaded from config.toml.

    User-tunable knobs:
    - ``kline_cutoff_seconds``: data horizon for the strategy (gate
      consumes candles closing at-or-before lock_at - this).
    - ``pool_cutoff_seconds``: data horizon for the pool aggregate (only
      bets with block_ts < lock_at - this are counted).
    - ``max_consecutive_kline_fetch_failures``: streak counter before the bot
      crashes with InvariantError + supervisor restart.

    Derived (computed at config-load time from
    ``pancakebot/timing_constants.py``; not user-tunable):
    - ``bet_submit_deadline_offset_before_lock_ms``
    - ``critical_path_wakeup_offset_before_lock_ms``
    - ``preflight_wakeup_offset_before_lock_ms``
    """

    # User-tunable
    kline_cutoff_seconds: int
    pool_cutoff_seconds: int
    max_consecutive_kline_fetch_failures: int

    # Derived (from timing_constants.py at load time)
    bet_submit_deadline_offset_before_lock_ms: int
    critical_path_wakeup_offset_before_lock_ms: int
    single_poll_wakeup_offset_before_lock_ms: int
    preflight_wakeup_offset_before_lock_ms: int
    okx_warmup_wakeup_offset_before_lock_ms: int

    # Other
    dry_initial_bankroll_bnb: float
    live_min_bet_only: bool
    backtest_round_count: int
    backtest_initial_bankroll_bnb: float

    # Full BacktestConfig with validation (kept as inner dataclass).
    backtest: BacktestConfig
    # Full StrategyConfig.
    strategy: StrategyConfig


# -- TOML parsing helpers -----------------------------------------------------

def _coerce_strict_int(v: Any, key: str) -> int:
    """Return *v* as a Python int with strict type semantics.

    Accepts integer literals (TOML ``int``) and integer-shaped strings
    (e.g. for env-var-overridden config). Rejects floats (no silent
    truncation -- ``2.5`` for an int field is a config error, not a 2),
    bools (``bool`` is a subclass of ``int`` in Python; accepting
    ``true``/``false`` for an int field is wrong), and everything else.

    Raises ``InvariantError`` with the field name + received type/value.
    """
    if isinstance(v, str):
        try:
            return int(v)
        except ValueError as e:
            raise InvariantError(
                f"config_key_not_int: {key}={v!r} (str parse failed: {e})"
            ) from e
    # Order matters: bool BEFORE int (bool is a subclass of int).
    if isinstance(v, bool):
        raise InvariantError(
            f"config_key_not_int: {key}={v!r} "
            f"(bool not allowed; expected integer literal)"
        )
    if isinstance(v, float):
        raise InvariantError(
            f"config_key_not_int: {key}={v!r} "
            f"(float not allowed; use an integer literal in TOML)"
        )
    if isinstance(v, int):
        return v
    raise InvariantError(
        f"config_key_not_int: {key}={v!r} "
        f"(got {type(v).__name__}; expected int)"
    )


def _req_int(obj: dict[str, Any], key: str) -> int:
    if key not in obj:
        raise InvariantError(f"missing_config_key: {key}")
    return _coerce_strict_int(obj[key], key)


def _opt_int(obj: dict[str, Any], key: str, default: int) -> int:
    if key not in obj:
        return int(default)
    return _coerce_strict_int(obj[key], key)


def _opt_int_or_none(obj: dict[str, Any], key: str) -> int | None:
    """Return an int if key exists and value is an int, otherwise None.

    Bools are explicitly rejected (``bool`` is a subclass of ``int`` in
    Python; accepting ``true``/``false`` for an int field is wrong).
    """
    v = obj.get(key)
    if isinstance(v, bool):
        return None
    return v if isinstance(v, int) else None


def _opt_float(obj: dict[str, Any], key: str, default: float) -> float:
    """Return *key* as a float, accepting int (no truncation risk) but
    rejecting bool (``bool`` is a subclass of ``int`` in Python).
    """
    if key not in obj:
        return float(default)
    v = obj[key]
    if isinstance(v, bool):
        raise InvariantError(
            f"config_key_not_number: {key}={v!r} "
            f"(bool not allowed; expected number)"
        )
    if not isinstance(v, (int, float)):
        raise InvariantError(
            f"config_key_not_number: {key}={v!r} "
            f"(got {type(v).__name__}; expected number)"
        )
    return float(v)


def _opt_bool(obj: dict[str, Any], key: str, default: bool) -> bool:
    if key not in obj:
        return default
    v = obj[key]
    if not isinstance(v, bool):
        raise InvariantError(f"config_key_not_bool: {key}")
    return v


def _opt_str(obj: dict[str, Any], key: str, default: str) -> str:
    if key not in obj:
        return default
    v = obj[key]
    if not isinstance(v, str):
        raise InvariantError(f"config_key_not_str: {key}")
    return v


def _opt_section(parent: dict[str, Any], key: str) -> dict[str, Any]:
    """Return parent[key] as a dict, or an empty dict if absent.

    Used for nested [strategy.*] sections in config.toml — missing sections
    fall back to defaults instead of raising.
    """
    v = parent.get(key)
    if v is None:
        return {}
    if not isinstance(v, dict):
        raise InvariantError(f"config_section_not_dict: {key}")
    return v


def _opt_int_tuple(obj: dict[str, Any], key: str, default: tuple[int, ...]) -> tuple[int, ...]:
    if key not in obj:
        return default
    v = obj[key]
    if not isinstance(v, (list, tuple)):
        raise InvariantError(f"config_key_not_int_list: {key} (not a list)")
    # Reject bool elements (bool is a subclass of int in Python; accepting
    # [true, false] in an int-list field would be wrong).
    for i, x in enumerate(v):
        if isinstance(x, bool) or not isinstance(x, int):
            raise InvariantError(
                f"config_key_not_int_list: {key}[{i}]={x!r} "
                f"(got {type(x).__name__}; expected int)"
            )
    return tuple(int(x) for x in v)


def load_strategy_config(cfg_toml: dict[str, Any]) -> StrategyConfig:
    """Build StrategyConfig from the parsed TOML root dict.

    Missing [strategy.*] sections or missing keys fall back to the defaults
    in _DEFAULT_STRATEGY.
    """
    strat_sec = _opt_section(cfg_toml, "strategy")
    pf_sec = _opt_section(strat_sec, "pool_filter")
    gate_sec = _opt_section(strat_sec, "gate")
    btc_sec = _opt_section(strat_sec, "btc_primary")
    btc_thresh_sec = _opt_section(btc_sec, "threshold")
    btc_sizing_sec = _opt_section(btc_sec, "sizing")
    es_sec = _opt_section(strat_sec, "eth_sol_fallback")
    es_signal_sec = _opt_section(es_sec, "signal")
    es_sizing_sec = _opt_section(es_sec, "sizing")
    t2_sec = _opt_section(strat_sec, "tier2_sizing")
    risk_sec = _opt_section(strat_sec, "risk")

    d = _DEFAULT_STRATEGY
    cfg = StrategyConfig(
        pool_filter=PoolFilterConfig(
            min_pool_bnb_at_cutoff=_opt_float(pf_sec, "min_pool_bnb_at_cutoff", d.pool_filter.min_pool_bnb_at_cutoff),
            min_payout_multiple_at_cutoff=_opt_float(pf_sec, "min_payout_multiple_at_cutoff", d.pool_filter.min_payout_multiple_at_cutoff),
        ),
        gate=GateConfig(
            mtf_lookbacks=_opt_int_tuple(gate_sec, "mtf_lookbacks", d.gate.mtf_lookbacks),
            mtf_min_return_threshold=_opt_float(gate_sec, "mtf_min_return_threshold", d.gate.mtf_min_return_threshold),
        ),
        btc_primary=BtcPrimaryConfig(
            threshold=BtcPrimaryThresholdConfig(
                small_pool_min_signal_strength=_opt_float(btc_thresh_sec, "small_pool_min_signal_strength", d.btc_primary.threshold.small_pool_min_signal_strength),
                pool_size_boundary_bnb=_opt_float(
                    btc_thresh_sec, "pool_size_boundary_bnb",
                    d.btc_primary.threshold.pool_size_boundary_bnb,
                ),
            ),
            sizing=BtcPrimarySizingConfig(
                base_pool_fraction=_opt_float(btc_sizing_sec, "base_pool_fraction", d.btc_primary.sizing.base_pool_fraction),
                pool_fraction_slope=_opt_float(btc_sizing_sec, "pool_fraction_slope", d.btc_primary.sizing.pool_fraction_slope),
                max_pool_fraction=_opt_float(btc_sizing_sec, "max_pool_fraction", d.btc_primary.sizing.max_pool_fraction),
            ),
        ),
        eth_sol_fallback=EthSolFallbackConfig(
            signal=EthSolFallbackSignalConfig(
                min_signal_strength=_opt_float(
                    es_signal_sec, "min_signal_strength",
                    d.eth_sol_fallback.signal.min_signal_strength,
                ),
            ),
            sizing=EthSolFallbackSizingConfig(
                base_pool_fraction=_opt_float(
                    es_sizing_sec, "base_pool_fraction",
                    d.eth_sol_fallback.sizing.base_pool_fraction,
                ),
            ),
        ),
        tier2_sizing=Tier2SizingConfig(
            eth_sol_signal_weight=_opt_float(
                t2_sec, "eth_sol_signal_weight", d.tier2_sizing.eth_sol_signal_weight,
            ),
            min_bet_threshold_bnb=_opt_float(
                t2_sec, "min_bet_threshold_bnb", d.tier2_sizing.min_bet_threshold_bnb,
            ),
        ),
        risk=RiskConfig(
            max_bet_fraction_of_bankroll=_opt_float(
                risk_sec, "max_bet_fraction_of_bankroll",
                d.risk.max_bet_fraction_of_bankroll,
            ),
            min_bankroll_bnb_to_bet=_opt_float(
                risk_sec, "min_bankroll_bnb_to_bet", d.risk.min_bankroll_bnb_to_bet,
            ),
            max_drawdown_fraction_from_peak=_opt_float(
                risk_sec, "max_drawdown_fraction_from_peak",
                d.risk.max_drawdown_fraction_from_peak,
            ),
            cooldown_rounds=_opt_int(
                risk_sec, "cooldown_rounds", d.risk.cooldown_rounds,
            ),
            drawdown_peak_window_days=_opt_int(
                risk_sec, "drawdown_peak_window_days", d.risk.drawdown_peak_window_days,
            ),
            max_bet_bnb_btc_primary=_opt_float(
                risk_sec, "max_bet_bnb_btc_primary",
                d.risk.max_bet_bnb_btc_primary,
            ),
            max_bet_bnb_eth_sol_fallback=_opt_float(
                risk_sec, "max_bet_bnb_eth_sol_fallback",
                d.risk.max_bet_bnb_eth_sol_fallback,
            ),
            drawdown_peak_mode=_opt_str(
                risk_sec, "drawdown_peak_mode", d.risk.drawdown_peak_mode,
            ),
        ),
    )
    cfg.validate()
    return cfg


def load_strategy_config_from_dict(d: dict[str, Any]) -> StrategyConfig:
    """Build StrategyConfig from a Python dict in [strategy.*] TOML shape.

    Used by the in-process experiment driver to construct StrategyConfig
    directly without round-tripping through a temp .toml file. The dict
    has the same shape as the parsed TOML root (i.e. the top level is
    expected to be a wrapper with a "strategy" key, or you can pass the
    inner [strategy] dict directly via {"strategy": d}).

    Missing keys/sections fall back to _DEFAULT_STRATEGY just like the
    TOML loader.
    """
    if "strategy" in d:
        wrapped = d
    else:
        # Caller passed the inner [strategy] dict directly.
        wrapped = {"strategy": d}
    return load_strategy_config(wrapped)


# -- Strict-mode schema -------------------------------------------------------
#
# Allow-list of every recognized section and its keys. Walked by
# ``_validate_strict_schema`` after parsing the TOML; any key or section
# not listed here raises InvariantError with a difflib-based "did you
# mean" suggestion. The dataclass fields above ARE the truth — keep this
# table in sync when adding/removing knobs.

_CONFIG_SCHEMA: dict[str, set[str]] = {
    "runtime": {
        "kline_cutoff_seconds",
        "pool_cutoff_seconds",
        "max_consecutive_kline_fetch_failures",
    },
    "dry": {"initial_bankroll_bnb"},
    "live": {"min_bet_only"},
    "backtest": {
        "backtest_round_count",
        "initial_bankroll_bnb",
        "epoch_start",
        "epoch_end",
    },
    "strategy": set(),
    "strategy.pool_filter": {
        "min_pool_bnb_at_cutoff",
        "min_payout_multiple_at_cutoff",
    },
    "strategy.gate": {"mtf_lookbacks", "mtf_min_return_threshold"},
    "strategy.btc_primary": set(),
    "strategy.btc_primary.threshold": {
        "small_pool_min_signal_strength",
        "pool_size_boundary_bnb",
    },
    "strategy.btc_primary.sizing": {
        "base_pool_fraction",
        "pool_fraction_slope",
        "max_pool_fraction",
    },
    "strategy.eth_sol_fallback": set(),
    "strategy.eth_sol_fallback.signal": {"min_signal_strength"},
    "strategy.eth_sol_fallback.sizing": {"base_pool_fraction"},
    "strategy.tier2_sizing": {
        "eth_sol_signal_weight",
        "min_bet_threshold_bnb",
    },
    "strategy.risk": {
        "max_bet_fraction_of_bankroll",
        "min_bankroll_bnb_to_bet",
        "max_drawdown_fraction_from_peak",
        "cooldown_rounds",
        "drawdown_peak_window_days",
        "max_bet_bnb_btc_primary",
        "max_bet_bnb_eth_sol_fallback",
        "drawdown_peak_mode",
    },
}


def _suggest(name: str, candidates) -> str:
    """Return a ' (did you mean X?)' fragment, or '' if no close match."""
    import difflib
    matches = difflib.get_close_matches(name, list(candidates), n=1, cutoff=0.6)
    return f" (did you mean {matches[0]!r}?)" if matches else ""


def _validate_strict_schema(raw: dict, schema: dict[str, set[str]]) -> None:
    """Walk ``raw`` (parsed TOML root) and raise on any key/section not in
    ``schema``. Suggests a close match via difflib when one exists.

    Sections are paths like ``"strategy.risk"``; keys are leaf names.
    Anything in ``raw`` that doesn't match the schema is a config error —
    typo'd renamed key, obsolete knob from a removed feature, or section
    misplaced under the wrong parent.
    """
    # Top-level: every key must be a known section (we have no top-level
    # scalars in this schema).
    for k, v in raw.items():
        if not isinstance(v, dict):
            sug = _suggest(k, [p for p in schema if "." not in p])
            raise InvariantError(
                f"config_unknown_top_level_key: {k!r}"
                f" (expected a section header){sug}"
            )
        _validate_strict_section(k, v, schema)


def _validate_strict_section(path: str, section: dict, schema: dict[str, set[str]]) -> None:
    if path not in schema:
        sug = _suggest(path, schema.keys())
        raise InvariantError(f"config_unknown_section: [{path}]{sug}")
    valid_keys = schema[path]
    for k, v in section.items():
        sub_path = f"{path}.{k}"
        if isinstance(v, dict):
            _validate_strict_section(sub_path, v, schema)
        else:
            if k not in valid_keys:
                sug = _suggest(k, valid_keys)
                raise InvariantError(
                    f"config_unknown_key: {k!r} in [{path}]{sug}"
                )


# -- Main loader --------------------------------------------------------------

def load_app_config(path: str) -> AppConfig:
    p = Path(path)
    if not p.exists():
        raise InvariantError(f"config_file_missing: {path}")

    try:
        raw = tomllib.loads(p.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as e:
        raise InvariantError(f"config_toml_parse_failed: {e}") from e

    if not isinstance(raw, dict):
        raise InvariantError("config_root_not_dict")

    # Fail-fast on unknown keys/sections before per-field reads. The
    # loader's _opt_* helpers silently ignore unrecognized keys; this
    # check is the safety floor that catches typos, obsolete renames, and
    # operator-side misconfigurations that would otherwise go unnoticed.
    _validate_strict_schema(raw, _CONFIG_SCHEMA)

    runtime = raw.get("runtime", {})
    dry_sec = raw.get("dry", {})
    live_sec = raw.get("live", {})
    backtest_sec = raw.get("backtest", {})

    if not isinstance(runtime, dict):
        raise InvariantError("config_section_not_dict: runtime")
    if not isinstance(dry_sec, dict):
        raise InvariantError("config_section_not_dict: dry")
    if not isinstance(live_sec, dict):
        raise InvariantError("config_section_not_dict: live")
    if not isinstance(backtest_sec, dict):
        raise InvariantError("config_section_not_dict: backtest")

    # [runtime]
    # ``kline_cutoff_seconds`` is the strategy's data horizon for OKX
    # klines (FIXED by strategy: the kline closing at lock - cutoff is
    # required; without that kline the strategy falls apart). The gate's
    # newest candle CLOSES at ``lock_at - cutoff*1000``. The wake offset
    # adjusts around this fixed cutoff -- the cross-validation below
    # asserts the wake offset fits the cutoff window.
    kline_cutoff_seconds = _req_int(runtime, "kline_cutoff_seconds")
    if not (1 <= kline_cutoff_seconds <= 30):
        raise InvariantError(
            f"runtime_kline_cutoff_seconds_out_of_range: "
            f"got={kline_cutoff_seconds} valid=[1..30]"
        )

    # ``pool_cutoff_seconds`` is the strategy's data horizon for BSC
    # BetBull/BetBear events. Only bets with on-chain block_timestamp
    # < lock_at - this are counted in the pool aggregate. The pool-read
    # wake offset must fit the cutoff window (cross-validated below).
    pool_cutoff_seconds = _opt_int(runtime, "pool_cutoff_seconds", 6)
    if not (1 <= pool_cutoff_seconds <= 30):
        raise InvariantError(
            f"runtime_pool_cutoff_seconds_out_of_range: "
            f"got={pool_cutoff_seconds} valid=[1..30]"
        )

    # ``max_consecutive_kline_fetch_failures``: streak counter for OKX
    # transient failures on the live decision path. After this many in a
    # row, the gate raises InvariantError -> bot crashes -> supervisor
    # restart + Discord alert.
    max_consecutive_kline_fetch_failures = _opt_int(
        runtime, "max_consecutive_kline_fetch_failures", 5,
    )
    if not (1 <= max_consecutive_kline_fetch_failures <= 100):
        raise InvariantError(
            f"runtime_max_consecutive_fetch_failures_out_of_range: "
            f"got={max_consecutive_kline_fetch_failures} valid=[1..100]"
        )

    # --- Derived timing constants (NOT user-tunable) ---
    # All four wake offsets and the bet-submit deadline offset are computed
    # from empirical constants in pancakebot/timing_constants.py. To change
    # any value, re-run the corresponding probe and update the constant
    # there. See timing_constants.py for the derivation formulas.
    from pancakebot import timing_constants as _tc

    # Bundle 4 (2026-05-14): static fallback only. Used when the chain is
    # pre-Lorentz OR ms-encoding detection fails. Live decision path on a
    # post-Lorentz chain uses RpcPoller.compute_dynamic_submit_deadline_ms()
    # for per-round prediction (typically 250-300ms tighter than this
    # static value). Derivation breakdown:
    #   BSC_QUANTUM_MS             (50)  — guard against 50ms quantum shift
    #   BSC_BLOCK_TIME_MS          (450) — back off one full slot if needed
    #   VALIDATOR_ASSEMBLY_WINDOW_MS (50) — validator's TX-list freeze window
    #   BSC_BET_SUBMIT_ONE_WAY_MS  (75)  — one-way RPC send to validator mempool
    # = 625ms total.
    bet_submit_deadline_offset_before_lock_ms = (
        _tc.BSC_QUANTUM_MS
        + _tc.BSC_BLOCK_TIME_MS
        + _tc.VALIDATOR_ASSEMBLY_WINDOW_MS
        + _tc.BSC_BET_SUBMIT_ONE_WAY_MS
    )
    # Static critical-path fallback offset, used only when the per-round anchor
    # poll times out. Uses OKX P99 — the SAME statistic the dynamic wake walks
    # back by (engine.py) — so static and dynamic agree; the prior separate P95
    # tier is retired (2026-06-06 VM re-baseline).
    critical_path_wakeup_offset_before_lock_ms = (
        bet_submit_deadline_offset_before_lock_ms
        + _tc.OKX_KLINE_FETCH_RTT_P99_MS
        + _tc.SIGNAL_COMPUTE_TIME_MS
        + _tc.POOL_READ_TIME_MS
    )
    preflight_wakeup_offset_before_lock_ms = (
        critical_path_wakeup_offset_before_lock_ms
        + _tc.PREFLIGHT_WAKEUP_OFFSET_BEFORE_CRITICAL_PATH_MS
    )
    # OKX session warmup wake (2026-05-21): fires before preflight_wake so
    # any TLS handshake cost on an expired OkxClient connection is paid OUT
    # of the bet-decision critical path.
    okx_warmup_wakeup_offset_before_lock_ms = (
        _tc.OKX_WARMUP_WAKEUP_OFFSET_BEFORE_LOCK_MS
    )
    # Bundle 5 v2 (2026-05-14): ``ntp_sync_wakeup_offset_ms`` retired.
    # The bot trusts the OS clock directly (W32Time tightening per
    # README); no application-level NTP wake is scheduled.

    # --- RPC poll wake schedule (Era 11: 2026-05-07 pivot; Candidate C
    # single-poll: 2026-06-06) ---
    # See var/design/rpc_polling_architecture_2026_05_07.md. The single-poll
    # offset doesn't depend on the rtt_p99 lookup; the startup invariant below
    # checks the chosen offset still accommodates the actual p99 + safety,
    # failing-fast with a clear InvariantError if rtt_p99 drifts past what the
    # schedule can absorb.
    #
    # Candidate C (2026-06-06): ONE batched poll before the critical path,
    # replacing the 3-leg ramp ladder. The 2026-06-06 VM re-baseline pins it to
    # a fixed rail (SINGLE_POLL_WAKEUP_OFFSET_BEFORE_LOCK_MS = 2500ms) fired
    # closer to lock for a fresher pool snapshot; the VM RPC-RTT table makes the
    # completion budget comfortable. The retained 8s periodic poll keeps the
    # catch-up batch bounded (~1 interval). Two startup invariants below bracket
    # the rail: CAPTURE (don't fire before the cutoff block is available) then
    # COMPLETION (fire early enough to finish before critical_path).
    single_poll_wakeup_offset_before_lock_ms = (
        _tc.SINGLE_POLL_WAKEUP_OFFSET_BEFORE_LOCK_MS
    )

    # --- Startup invariant (CAPTURE): the single poll must NOT fire before the
    # cutoff block (block_ts < lock - pool_cutoff) has its receipts available
    # (2026-06-06). The latest safe offset = the old final-poll derivation: the
    # pool_cutoff window minus one block, the receipt-availability delay, and
    # the final-to-critical-path cushion. If the rail is set EARLIER than this
    # (a LARGER offset), it polls before the cutoff block is fetchable and
    # misses late bets — normally a sign pool_cutoff_seconds is too small for
    # the chosen rail. ---
    _single_poll_max_capture_offset = (
        pool_cutoff_seconds * 1000
        - _tc.BSC_BLOCK_TIME_MS
        - _tc.RPC_BLOCK_AVAILABILITY_DELAY_P99_MS
        - _tc.RPC_POLL_FINAL_TO_CRITICAL_PATH_SAFETY_MS
    )
    if single_poll_wakeup_offset_before_lock_ms > _single_poll_max_capture_offset:
        raise InvariantError(
            f"single_poll_fires_before_cutoff_available: "
            f"single_poll_wakeup={single_poll_wakeup_offset_before_lock_ms}ms "
            f"> max_capture={_single_poll_max_capture_offset}ms "
            f"(pool_cutoff*1000={pool_cutoff_seconds * 1000} "
            f"- block_time={_tc.BSC_BLOCK_TIME_MS} "
            f"- availability={_tc.RPC_BLOCK_AVAILABILITY_DELAY_P99_MS} "
            f"- safety={_tc.RPC_POLL_FINAL_TO_CRITICAL_PATH_SAFETY_MS}). "
            f"pool_cutoff_seconds={pool_cutoff_seconds}. Either raise "
            f"pool_cutoff_seconds or lower "
            f"SINGLE_POLL_WAKEUP_OFFSET_BEFORE_LOCK_MS."
        )

    # --- Startup invariant (COMPLETION): the single batched poll must complete
    # before critical_path reads the pool snapshot (Candidate C, 2026-06-06) ---
    # At the worst-case batch (one 8s periodic interval ≈ 20 blocks) and
    # empirical p99 RTT, the single poll must fire AND complete before
    # critical_path, with the deadline safety cushion. If this fails, the rail
    # (SINGLE_POLL_WAKEUP_OFFSET_BEFORE_LOCK_MS) is too small, critical_path has
    # grown, or the rtt_p99 table value has grown past what the budget absorbs.
    # (The prior 3-leg ladder's per-leg interval invariants are retired.)
    _single_poll_rtt = _tc.rpc_rtt_p99_for_batch(_tc.EXPECTED_SINGLE_POLL_BATCH_SIZE)
    _single_poll_completion_offset = (
        single_poll_wakeup_offset_before_lock_ms
        - _single_poll_rtt
        - _tc.RPC_POLL_DEADLINE_SAFETY_BUFFER_MS
    )
    if _single_poll_completion_offset < critical_path_wakeup_offset_before_lock_ms:
        raise InvariantError(
            f"single_poll_rtt_budget_insufficient: "
            f"single_poll_wakeup={single_poll_wakeup_offset_before_lock_ms}ms "
            f"- rtt_p99({_tc.EXPECTED_SINGLE_POLL_BATCH_SIZE})={_single_poll_rtt}ms "
            f"- safety={_tc.RPC_POLL_DEADLINE_SAFETY_BUFFER_MS}ms "
            f"= {_single_poll_completion_offset}ms "
            f"< critical_path={critical_path_wakeup_offset_before_lock_ms}ms. "
            f"Either raise SINGLE_POLL_WAKEUP_OFFSET_BEFORE_LOCK_MS or investigate "
            f"RPC_BATCH_RECEIPTS_RTT_P99_MS_BY_SIZE."
        )

    # --- Startup invariant (ANCHOR CLEARANCE, Era 12b 2026-06-10): a
    # wall-cap-bound single poll must release the engine thread before the
    # anchor poll fires. The single poll runs SYNCHRONOUSLY on the engine
    # thread from its wake (lock - single_poll_wakeup) until at most the
    # wall cap + the post-cap processing tail; the next event on that
    # thread is the anchor poll at lock - ANCHOR_POLL_OFFSET. Every ms a
    # capped poll runs past that gap delays the anchor 1:1 and eats the
    # critical path's slack. Keyed to the ANCHOR offset — the actual next
    # downstream event — NOT critical_path: the engine's deadline was
    # historically derived from critical_path (1105ms budget vs the real
    # 1000ms anchor gap), which masked exactly this adjacency until the
    # Era 12b pre-deploy review. The engine derives single_poll
    # deadline_ms from this same expression; fires if anyone moves the
    # rail, the anchor offset, the wall cap, or the tail margin without
    # co-updating the others. ---
    _single_poll_anchor_gap = (
        single_poll_wakeup_offset_before_lock_ms
        - _tc.ANCHOR_POLL_OFFSET_BEFORE_LOCK_MS
    )
    if (
        _tc.RPC_POLL_WALL_CAP_SINGLE_MS + _tc.RPC_POLL_TAIL_MARGIN_MS
        > _single_poll_anchor_gap
    ):
        raise InvariantError(
            f"single_poll_wall_cap_collides_with_anchor: "
            f"wall_cap={_tc.RPC_POLL_WALL_CAP_SINGLE_MS}ms "
            f"+ tail={_tc.RPC_POLL_TAIL_MARGIN_MS}ms "
            f"> anchor_gap={_single_poll_anchor_gap}ms "
            f"(single_poll_wakeup={single_poll_wakeup_offset_before_lock_ms}ms "
            f"- anchor_offset={_tc.ANCHOR_POLL_OFFSET_BEFORE_LOCK_MS}ms). "
            f"A cap-bound single poll would still be running when the anchor "
            f"poll fires. Lower RPC_POLL_WALL_CAP_SINGLE_MS or move the "
            f"rail/anchor offset consciously (timing co-update discipline)."
        )

    # Candidate C (2026-06-06): the 3-leg ramp ladder + its per-leg interval
    # invariants are retired. The first scheduled wake in the round is now
    # okx_warmup, then preflight, then the single poll — all strictly
    # decreasing offsets, validated by the engine's wake-ladder ordering
    # check (_assert_critical_path_timing_sane) at startup.

    # Bundle 5 v2 (2026-05-14): the NTP wake budget cross-validation
    # is retired alongside the application-level NTP layer itself.
    #
    # 2026-05-17: the prior P95/P99 publish-tier ladder is retired too.
    # That check was a one-shot config-load gate; under Bundle 5 v2 the
    # actual fetch fires at whatever offset the per-round anchor dictates,
    # which is independent of the static wake's relation to the
    # OKX publish percentiles. The streak counter
    # (``max_consecutive_kline_fetch_failures``) remains the only runtime
    # tolerance mechanism for publish-delay tails.

    # [dry]
    dry_initial_bankroll_bnb = _opt_float(dry_sec, "initial_bankroll_bnb", 50.0)
    if dry_initial_bankroll_bnb <= 0.0:
        raise InvariantError("dry_initial_bankroll_bnb_must_be_positive")

    # [live]
    live_min_bet_only = _opt_bool(live_sec, "min_bet_only", True)

    # [backtest]
    backtest_round_count = _opt_int(backtest_sec, "backtest_round_count", 5000)
    if backtest_round_count <= 0:
        raise InvariantError("backtest_round_count_must_be_positive")

    bt_bankroll = _opt_float(backtest_sec, "initial_bankroll_bnb", 50.0)
    if bt_bankroll <= 0.0:
        raise InvariantError("backtest_initial_bankroll_bnb_must_be_positive")

    epoch_start = _opt_int_or_none(backtest_sec, "epoch_start")
    epoch_end = _opt_int_or_none(backtest_sec, "epoch_end")

    backtest_cfg = BacktestConfig(
        backtest_round_count=backtest_round_count,
        initial_bankroll_bnb=bt_bankroll,
        epoch_start=epoch_start,
        epoch_end=epoch_end,
    )
    backtest_cfg.validate()

    strategy_cfg = load_strategy_config(raw)

    return AppConfig(
        kline_cutoff_seconds=kline_cutoff_seconds,
        pool_cutoff_seconds=pool_cutoff_seconds,
        max_consecutive_kline_fetch_failures=max_consecutive_kline_fetch_failures,
        bet_submit_deadline_offset_before_lock_ms=bet_submit_deadline_offset_before_lock_ms,
        critical_path_wakeup_offset_before_lock_ms=critical_path_wakeup_offset_before_lock_ms,
        single_poll_wakeup_offset_before_lock_ms=single_poll_wakeup_offset_before_lock_ms,
        preflight_wakeup_offset_before_lock_ms=preflight_wakeup_offset_before_lock_ms,
        okx_warmup_wakeup_offset_before_lock_ms=okx_warmup_wakeup_offset_before_lock_ms,
        dry_initial_bankroll_bnb=dry_initial_bankroll_bnb,
        live_min_bet_only=live_min_bet_only,
        backtest_round_count=backtest_round_count,
        backtest_initial_bankroll_bnb=bt_bankroll,
        backtest=backtest_cfg,
        strategy=strategy_cfg,
    )
