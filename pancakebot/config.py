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

    simulation_size: int
    initial_bankroll_bnb: float
    epoch_start: int | None = None
    epoch_end: int | None = None

    def validate(self) -> None:
        if not isinstance(self.simulation_size, int):
            raise InvariantError("backtest_simulation_size_not_int")
        if self.simulation_size <= 0:
            raise InvariantError("backtest_simulation_size_must_be_positive")

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
    min_pool_bnb: float
    min_payout: float


@dataclass(frozen=True, slots=True)
class BtcPrimaryThresholdConfig:
    """BTC primary small-pool admission threshold (large pools use gate threshold).

    For pools >= ``pool_size_boundary_bnb`` the gate's
    ``GateConfig.mtf_threshold`` is the binding constraint (it already
    fired). For pools below, signal_strength must additionally exceed
    ``small_pool`` (a stricter cutoff that excludes weak signals on
    small / dilution-prone pools).
    """
    small_pool: float
    pool_size_boundary_bnb: float


@dataclass(frozen=True, slots=True)
class BtcPrimarySizingConfig:
    """BTC primary bet sizing.

    - base_fraction: starting fraction-of-pool stake.
    - sizing_slope: gain on signal_strength; final frac =
      base_fraction + sizing_slope * signal_strength, capped at max_frac.
    - max_frac: hard cap on the computed pool fraction, applied
      before per-bet absolute caps (in RiskConfig).
    """
    base_fraction: float
    sizing_slope: float
    max_frac: float


@dataclass(frozen=True, slots=True)
class BtcPrimaryConfig:
    threshold: BtcPrimaryThresholdConfig
    sizing: BtcPrimarySizingConfig


@dataclass(frozen=True, slots=True)
class EthSolFallbackSignalConfig:
    """ETH+SOL fallback signal (fires when BTC primary is silent)."""
    min_strength: float


@dataclass(frozen=True, slots=True)
class EthSolFallbackSizingConfig:
    """ETH+SOL fallback bet sizing. (max_bet_bnb moved to RiskConfig.)"""
    base_fraction: float


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
    - mtf_threshold: minimum min(|return|) across the lookbacks for
      the gate to fire. The pipeline may apply a stricter pool-adaptive
      threshold via BtcPrimaryThresholdConfig.

    Previously module constants (_MTF_LOOKBACKS, _MTF_THRESH) in
    pancakebot/strategy/momentum_gate.py.
    """
    mtf_lookbacks: tuple[int, ...]
    mtf_threshold: float


@dataclass(frozen=True, slots=True)
class Tier2SizingConfig:
    """Cross-regime sizing knobs.

    - eth_sol_sizing_weight: coefficient applied to BOTH ETH and SOL
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
        symmetric eth_sol_sizing_weight above.
    """
    eth_sol_sizing_weight: float
    min_bet_threshold_bnb: float


@dataclass(frozen=True, slots=True)
class RiskConfig:
    """Risk controls applied by MomentumOnlyPipeline when a BankrollTracker is active.

    - max_bet_frac_of_bankroll: stake <= this fraction of current bankroll.
    - min_bankroll_bnb: skip all bets while bankroll < this.
    - max_drawdown_frac_from_peak: fire cooldown when (peak-current)/peak >= this.
      Set to 1.0 to disable the circuit breaker.
    - cooldown_rounds: number of rounds to pause after drawdown breaker fires.
    - window_days: rolling window (days) for the drawdown-from-peak calculation
      when ``dd_peak_mode == "rolling_7d"``.
    - dd_peak_mode: peak-tracking semantics for the drawdown breaker.
      ``"rolling_7d"`` (default) uses a rolling window of ``window_days``.
      ``"absolute_ratchet"`` uses an absolute-since-launch peak that
      monotonically only goes up — catches slow drains the rolling
      window misses.
    - max_bet_bnb_btc_primary: absolute BNB cap for BTC primary bets.
    - max_bet_bnb_eth_sol_fallback: absolute BNB cap for regime-2 ETH+SOL bets.
    """
    max_bet_frac_of_bankroll: float
    min_bankroll_bnb: float
    max_drawdown_frac_from_peak: float
    cooldown_rounds: int
    window_days: int
    max_bet_bnb_btc_primary: float
    max_bet_bnb_eth_sol_fallback: float
    dd_peak_mode: str = "rolling_7d"


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
        if pf.min_pool_bnb <= 0.0:
            raise InvariantError("strategy_pool_filter_min_pool_bnb_must_be_positive")
        if pf.min_payout < 1.0:
            raise InvariantError("strategy_pool_filter_min_payout_must_be_at_least_1")

        g = self.gate
        if not g.mtf_lookbacks:
            raise InvariantError("strategy_gate_mtf_lookbacks_must_be_non_empty")
        if any(not isinstance(lb, int) or lb <= 0 for lb in g.mtf_lookbacks):
            raise InvariantError("strategy_gate_mtf_lookbacks_must_be_positive_ints")
        if g.mtf_threshold <= 0.0:
            raise InvariantError("strategy_gate_mtf_threshold_must_be_positive")

        bt = self.btc_primary.threshold
        if bt.small_pool <= 0.0:
            raise InvariantError("strategy_btc_primary_threshold_small_pool_must_be_positive")
        if bt.pool_size_boundary_bnb <= 0.0:
            raise InvariantError("strategy_btc_primary_threshold_pool_size_boundary_bnb_must_be_positive")

        bs = self.btc_primary.sizing
        if not (0.0 < bs.base_fraction < 1.0):
            raise InvariantError("strategy_btc_primary_sizing_base_fraction_out_of_range")
        if bs.sizing_slope < 0.0:
            raise InvariantError("strategy_btc_primary_sizing_slope_must_be_non_negative")
        if not (0.0 < bs.max_frac <= 1.0):
            raise InvariantError("strategy_btc_primary_sizing_max_frac_out_of_range")

        es = self.eth_sol_fallback.signal
        if es.min_strength <= 0.0:
            raise InvariantError("strategy_eth_sol_fallback_signal_min_strength_must_be_positive")

        ez = self.eth_sol_fallback.sizing
        if not (0.0 < ez.base_fraction < 1.0):
            raise InvariantError("strategy_eth_sol_fallback_sizing_base_fraction_out_of_range")

        t2 = self.tier2_sizing
        if t2.eth_sol_sizing_weight < 0.0:
            raise InvariantError("strategy_tier2_sizing_eth_sol_sizing_weight_must_be_non_negative")
        if t2.min_bet_threshold_bnb < 0.0:
            raise InvariantError("strategy_tier2_sizing_min_bet_threshold_bnb_must_be_non_negative")

        rk = self.risk
        if not (0.0 < rk.max_bet_frac_of_bankroll <= 1.0):
            raise InvariantError("strategy_risk_max_bet_frac_of_bankroll_out_of_range")
        if rk.min_bankroll_bnb < 0.0:
            raise InvariantError("strategy_risk_min_bankroll_bnb_must_be_non_negative")
        if not (0.0 < rk.max_drawdown_frac_from_peak <= 1.0):
            raise InvariantError("strategy_risk_max_drawdown_frac_from_peak_out_of_range")
        if rk.cooldown_rounds < 0:
            raise InvariantError("strategy_risk_cooldown_rounds_must_be_non_negative")
        if rk.window_days <= 0:
            raise InvariantError("strategy_risk_window_days_must_be_positive")
        if rk.dd_peak_mode not in ("rolling_7d", "absolute_ratchet"):
            raise InvariantError(
                f"strategy_risk_dd_peak_mode_invalid: {rk.dd_peak_mode!r} "
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
    pool_filter=PoolFilterConfig(min_pool_bnb=1.5, min_payout=1.5),
    gate=GateConfig(
        mtf_lookbacks=(3, 7, 15),
        mtf_threshold=0.0001,
    ),
    btc_primary=BtcPrimaryConfig(
        threshold=BtcPrimaryThresholdConfig(
            small_pool=0.0002,
            pool_size_boundary_bnb=3.0,
        ),
        sizing=BtcPrimarySizingConfig(
            base_fraction=0.04,
            sizing_slope=100.0,
            max_frac=0.30,
        ),
    ),
    eth_sol_fallback=EthSolFallbackConfig(
        signal=EthSolFallbackSignalConfig(min_strength=0.00015),
        sizing=EthSolFallbackSizingConfig(base_fraction=0.02),
    ),
    tier2_sizing=Tier2SizingConfig(
        eth_sol_sizing_weight=0.3,
        min_bet_threshold_bnb=0.01,
    ),
    risk=RiskConfig(
        max_bet_frac_of_bankroll=0.05,
        min_bankroll_bnb=0.20,
        max_drawdown_frac_from_peak=0.15,
        cooldown_rounds=72,
        window_days=7,
        max_bet_bnb_btc_primary=2.0,
        max_bet_bnb_eth_sol_fallback=0.5,
        dd_peak_mode="rolling_7d",
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
    - ``max_consecutive_fetch_failures``: streak counter before the bot
      crashes with InvariantError + supervisor restart.

    Derived (computed at config-load time from
    ``pancakebot/timing_constants.py``; not user-tunable):
    - ``bet_submit_deadline_offset_ms``
    - ``kline_fetch_wakeup_offset_ms``
    - ``pool_read_wakeup_offset_ms``
    - ``skew_sync_wakeup_offset_ms``
    - ``kline_publish_tier``: ``"P99"`` (strict, full-inclusion guarantee)
      or ``"P95"`` (operating budget; ~5% tail absorbed by streak counter).
      Selected by tier-ladder cross-validation: P99 first, P95 fallback.
    """

    # User-tunable
    kline_cutoff_seconds: int
    pool_cutoff_seconds: int
    max_consecutive_fetch_failures: int

    # Derived (from timing_constants.py at load time)
    bet_submit_deadline_offset_ms: int
    kline_fetch_wakeup_offset_ms: int
    pool_read_wakeup_offset_ms: int
    skew_sync_wakeup_offset_ms: int
    kline_publish_tier: str

    # Other
    dry_initial_bankroll_bnb: float
    live_min_bet_only: bool
    backtest_simulation_size: int
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
            min_pool_bnb=_opt_float(pf_sec, "min_pool_bnb", d.pool_filter.min_pool_bnb),
            min_payout=_opt_float(pf_sec, "min_payout", d.pool_filter.min_payout),
        ),
        gate=GateConfig(
            mtf_lookbacks=_opt_int_tuple(gate_sec, "mtf_lookbacks", d.gate.mtf_lookbacks),
            mtf_threshold=_opt_float(gate_sec, "mtf_threshold", d.gate.mtf_threshold),
        ),
        btc_primary=BtcPrimaryConfig(
            threshold=BtcPrimaryThresholdConfig(
                small_pool=_opt_float(btc_thresh_sec, "small_pool", d.btc_primary.threshold.small_pool),
                pool_size_boundary_bnb=_opt_float(
                    btc_thresh_sec, "pool_size_boundary_bnb",
                    d.btc_primary.threshold.pool_size_boundary_bnb,
                ),
            ),
            sizing=BtcPrimarySizingConfig(
                base_fraction=_opt_float(btc_sizing_sec, "base_fraction", d.btc_primary.sizing.base_fraction),
                sizing_slope=_opt_float(btc_sizing_sec, "sizing_slope", d.btc_primary.sizing.sizing_slope),
                max_frac=_opt_float(btc_sizing_sec, "max_frac", d.btc_primary.sizing.max_frac),
            ),
        ),
        eth_sol_fallback=EthSolFallbackConfig(
            signal=EthSolFallbackSignalConfig(
                min_strength=_opt_float(
                    es_signal_sec, "min_strength",
                    d.eth_sol_fallback.signal.min_strength,
                ),
            ),
            sizing=EthSolFallbackSizingConfig(
                base_fraction=_opt_float(
                    es_sizing_sec, "base_fraction",
                    d.eth_sol_fallback.sizing.base_fraction,
                ),
            ),
        ),
        tier2_sizing=Tier2SizingConfig(
            eth_sol_sizing_weight=_opt_float(
                t2_sec, "eth_sol_sizing_weight", d.tier2_sizing.eth_sol_sizing_weight,
            ),
            min_bet_threshold_bnb=_opt_float(
                t2_sec, "min_bet_threshold_bnb", d.tier2_sizing.min_bet_threshold_bnb,
            ),
        ),
        risk=RiskConfig(
            max_bet_frac_of_bankroll=_opt_float(
                risk_sec, "max_bet_frac_of_bankroll",
                d.risk.max_bet_frac_of_bankroll,
            ),
            min_bankroll_bnb=_opt_float(
                risk_sec, "min_bankroll_bnb", d.risk.min_bankroll_bnb,
            ),
            max_drawdown_frac_from_peak=_opt_float(
                risk_sec, "max_drawdown_frac_from_peak",
                d.risk.max_drawdown_frac_from_peak,
            ),
            cooldown_rounds=_opt_int(
                risk_sec, "cooldown_rounds", d.risk.cooldown_rounds,
            ),
            window_days=_opt_int(
                risk_sec, "window_days", d.risk.window_days,
            ),
            max_bet_bnb_btc_primary=_opt_float(
                risk_sec, "max_bet_bnb_btc_primary",
                d.risk.max_bet_bnb_btc_primary,
            ),
            max_bet_bnb_eth_sol_fallback=_opt_float(
                risk_sec, "max_bet_bnb_eth_sol_fallback",
                d.risk.max_bet_bnb_eth_sol_fallback,
            ),
            dd_peak_mode=_opt_str(
                risk_sec, "dd_peak_mode", d.risk.dd_peak_mode,
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

    # ``max_consecutive_fetch_failures``: streak counter for OKX
    # transient failures on the live decision path. After this many in a
    # row, the gate raises InvariantError -> bot crashes -> supervisor
    # restart + Discord alert.
    max_consecutive_fetch_failures = _opt_int(
        runtime, "max_consecutive_fetch_failures", 5,
    )
    if not (1 <= max_consecutive_fetch_failures <= 100):
        raise InvariantError(
            f"runtime_max_consecutive_fetch_failures_out_of_range: "
            f"got={max_consecutive_fetch_failures} valid=[1..100]"
        )
    # --- Derived timing constants (NOT user-tunable) ---
    # All four wake offsets and the bet-submit deadline offset are computed
    # from empirical constants in pancakebot/timing_constants.py. To change
    # any value, re-run the corresponding probe and update the constant
    # there. See timing_constants.py for the derivation formulas.
    from pancakebot import timing_constants as _tc

    bet_submit_deadline_offset_ms = (
        _tc.BSC_BET_SUBMIT_RTT_P95_MS
        + _tc.BSC_BLOCK_TIME_MS
        + _tc.BET_SUBMIT_SAFETY_BUFFER_MS
    )
    kline_fetch_wakeup_offset_ms = (
        bet_submit_deadline_offset_ms
        + _tc.OKX_KLINE_FETCH_RTT_P95_MS
        + _tc.SIGNAL_COMPUTE_TIME_MS
    )
    pool_read_wakeup_offset_ms = (
        kline_fetch_wakeup_offset_ms
        + _tc.POOL_READ_TIME_MS
    )
    skew_sync_wakeup_offset_ms = (
        pool_read_wakeup_offset_ms
        + _tc.OKX_SKEW_SYNC_TIME_P99_MS
        + _tc.SKEW_SYNC_SAFETY_BUFFER_MS
    )

    # Tier-based publish-delay validation: prefer strict P99
    # (full-inclusion guarantee that the cutoff candle is published at
    # fetch time); fall back to P95 (operating budget; ~5% publish-delay
    # tail absorbed by the streak counter). InvariantError fires only if
    # even the looser P95 budget is exceeded.
    #
    # Behavior across cutoffs at the locked offsets:
    #   - cutoff=2 (canonical): P99 budget=700ms < wake=1090ms. P95
    #     budget=1300ms >= 1090ms -> tier="P95".
    #   - cutoff=3+: P99 budget=1700ms+ >= 1090ms -> tier="P99".
    #     Auto-strict; no code change needed when a future user opts
    #     into a larger cutoff.
    p99_budget_ms = (
        kline_cutoff_seconds * 1000 - _tc.OKX_KLINE_PUBLISH_DELAY_P99_MS
    )
    p95_budget_ms = (
        kline_cutoff_seconds * 1000 - _tc.OKX_KLINE_PUBLISH_DELAY_P95_MS
    )
    if kline_fetch_wakeup_offset_ms <= p99_budget_ms:
        kline_publish_tier = "P99"
    elif kline_fetch_wakeup_offset_ms <= p95_budget_ms:
        kline_publish_tier = "P95"
    else:
        raise InvariantError(
            f"config_kline_fetch_wakeup_exceeds_cutoff_publish_budget: "
            f"kline_fetch_wakeup_offset_ms={kline_fetch_wakeup_offset_ms}ms "
            f"exceeds P95 budget ({p95_budget_ms}ms) at "
            f"kline_cutoff_seconds={kline_cutoff_seconds}s "
            f"(P99 budget={p99_budget_ms}ms; "
            f"P95={_tc.OKX_KLINE_PUBLISH_DELAY_P95_MS}ms; "
            f"P99={_tc.OKX_KLINE_PUBLISH_DELAY_P99_MS}ms). "
            f"Increase kline_cutoff_seconds or reduce wake offset."
        )

    # Cross-validation: the pool-read wake offset must fit inside the
    # pool-cutoff window minus WSS bet-event arrival delay (P99). Same
    # framing as klines: the cutoff is fixed; the wake offset adjusts
    # around it.
    #
    # No tier fallback here -- WSS arrival is much more uniform than
    # REST publishing (single subscriber stream vs. per-symbol
    # publishing pipeline), so the P99 figure IS the operating budget;
    # there's no looser percentile to fall back to. If a P95 WSS
    # arrival probe constant is added later, this could be tier-ified
    # symmetrically with klines. For now: P99-strict is the only check.
    if pool_read_wakeup_offset_ms > (
        pool_cutoff_seconds * 1000 - _tc.WSS_BET_EVENT_ARRIVAL_DELAY_P99_MS
    ):
        raise InvariantError(
            f"config_pool_read_wakeup_exceeds_cutoff_arrival_budget: "
            f"pool_read_wakeup_offset_ms={pool_read_wakeup_offset_ms} "
            f"> pool_cutoff_seconds*1000={pool_cutoff_seconds * 1000} "
            f"- WSS_BET_EVENT_ARRIVAL_DELAY_P99_MS"
            f"={_tc.WSS_BET_EVENT_ARRIVAL_DELAY_P99_MS}"
        )

    # [dry]
    dry_initial_bankroll_bnb = _opt_float(dry_sec, "initial_bankroll_bnb", 50.0)
    if dry_initial_bankroll_bnb <= 0.0:
        raise InvariantError("dry_initial_bankroll_bnb_must_be_positive")

    # [live]
    live_min_bet_only = _opt_bool(live_sec, "min_bet_only", True)

    # [backtest]
    simulation_size = _opt_int(backtest_sec, "simulation_size", 5000)
    if simulation_size <= 0:
        raise InvariantError("backtest_simulation_size_must_be_positive")

    bt_bankroll = _opt_float(backtest_sec, "initial_bankroll_bnb", 50.0)
    if bt_bankroll <= 0.0:
        raise InvariantError("backtest_initial_bankroll_bnb_must_be_positive")

    epoch_start = _opt_int_or_none(backtest_sec, "epoch_start")
    epoch_end = _opt_int_or_none(backtest_sec, "epoch_end")

    backtest_cfg = BacktestConfig(
        simulation_size=simulation_size,
        initial_bankroll_bnb=bt_bankroll,
        epoch_start=epoch_start,
        epoch_end=epoch_end,
    )
    backtest_cfg.validate()

    strategy_cfg = load_strategy_config(raw)

    return AppConfig(
        kline_cutoff_seconds=kline_cutoff_seconds,
        pool_cutoff_seconds=pool_cutoff_seconds,
        max_consecutive_fetch_failures=max_consecutive_fetch_failures,
        bet_submit_deadline_offset_ms=bet_submit_deadline_offset_ms,
        kline_fetch_wakeup_offset_ms=kline_fetch_wakeup_offset_ms,
        pool_read_wakeup_offset_ms=pool_read_wakeup_offset_ms,
        skew_sync_wakeup_offset_ms=skew_sync_wakeup_offset_ms,
        kline_publish_tier=kline_publish_tier,
        dry_initial_bankroll_bnb=dry_initial_bankroll_bnb,
        live_min_bet_only=live_min_bet_only,
        backtest_simulation_size=simulation_size,
        backtest_initial_bankroll_bnb=bt_bankroll,
        backtest=backtest_cfg,
        strategy=strategy_cfg,
    )
