"""Load AppConfig and BacktestConfig from a TOML file and read required env vars."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomllib
from dotenv import load_dotenv

from pancakebot.errors import InvariantError


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

_BACKTEST_RESET_MODES = ("continuous", "chunk_reset")


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    """Backtest configuration."""

    simulation_size: int
    initial_bankroll_bnb: float
    reset_mode: str = "continuous"
    reset_every_rounds: int = 0
    tail_offset_rounds: int = 0
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

        if not isinstance(self.reset_mode, str):
            raise InvariantError("backtest_reset_mode_not_str")
        mode = self.reset_mode.strip()
        if mode not in _BACKTEST_RESET_MODES:
            raise InvariantError("backtest_reset_mode_invalid")

        if not isinstance(self.reset_every_rounds, int):
            raise InvariantError("backtest_reset_every_rounds_not_int")
        if self.reset_every_rounds < 0:
            raise InvariantError("backtest_reset_every_rounds_negative")
        if mode == "chunk_reset" and self.reset_every_rounds <= 0:
            raise InvariantError("backtest_chunk_reset_every_rounds_must_be_positive")

        if not isinstance(self.tail_offset_rounds, int):
            raise InvariantError("backtest_tail_offset_rounds_not_int")
        if self.tail_offset_rounds < 0:
            raise InvariantError("backtest_tail_offset_rounds_negative")

        if self.epoch_start is not None and self.epoch_end is not None:
            if self.epoch_start > self.epoch_end:
                raise InvariantError("backtest_epoch_start_after_epoch_end")


# -- App config ---------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class AppConfig:
    """User-facing configuration loaded from config.toml."""

    kline_cutoff_seconds: int
    prefetch_offset_seconds: int
    dry_initial_bankroll_bnb: float
    live_min_bet_only: bool
    backtest_simulation_size: int
    backtest_initial_bankroll_bnb: float

    # Full BacktestConfig with validation (kept as inner dataclass).
    backtest: BacktestConfig


# -- TOML parsing helpers -----------------------------------------------------

def _req_int(obj: dict[str, Any], key: str) -> int:
    if key not in obj:
        raise InvariantError(f"missing_config_key: {key}")
    v = obj[key]
    try:
        i = int(v)
    except (TypeError, ValueError) as e:
        raise InvariantError(f"config_key_not_int: {key} err={e}") from e
    return i


def _opt_int(obj: dict[str, Any], key: str, default: int) -> int:
    if key not in obj:
        return int(default)
    v = obj[key]
    try:
        i = int(v)
    except (TypeError, ValueError) as e:
        raise InvariantError(f"config_key_not_int: {key} err={e}") from e
    return i


def _opt_float(obj: dict[str, Any], key: str, default: float) -> float:
    if key not in obj:
        return float(default)
    v = obj[key]
    if not isinstance(v, (int, float)):
        raise InvariantError(f"config_key_not_number: {key}")
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
        return str(default)
    v = obj[key]
    if not isinstance(v, str) or not v.strip():
        raise InvariantError(f"config_key_not_nonempty_str: {key}")
    return v.strip()


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
    kline_cutoff_seconds = _req_int(runtime, "kline_cutoff_seconds")
    if kline_cutoff_seconds <= 0:
        raise InvariantError("kline_cutoff_seconds_must_be_positive")
    prefetch_offset_seconds = _req_int(runtime, "prefetch_offset_seconds")
    if prefetch_offset_seconds <= 0:
        raise InvariantError("prefetch_offset_seconds_must_be_positive")

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

    reset_mode = _opt_str(backtest_sec, "reset_mode", "continuous")
    reset_every_rounds = _opt_int(backtest_sec, "reset_every_rounds", 0)
    tail_offset_rounds = _opt_int(backtest_sec, "tail_offset_rounds", 0)

    epoch_start_raw = backtest_sec.get("epoch_start")
    epoch_end_raw = backtest_sec.get("epoch_end")
    epoch_start = None if epoch_start_raw is None else int(epoch_start_raw)
    epoch_end = None if epoch_end_raw is None else int(epoch_end_raw)

    backtest_cfg = BacktestConfig(
        simulation_size=simulation_size,
        initial_bankroll_bnb=bt_bankroll,
        reset_mode=str(reset_mode),
        reset_every_rounds=int(reset_every_rounds),
        tail_offset_rounds=int(tail_offset_rounds),
        epoch_start=epoch_start,
        epoch_end=epoch_end,
    )
    backtest_cfg.validate()

    return AppConfig(
        kline_cutoff_seconds=kline_cutoff_seconds,
        prefetch_offset_seconds=prefetch_offset_seconds,
        dry_initial_bankroll_bnb=dry_initial_bankroll_bnb,
        live_min_bet_only=live_min_bet_only,
        backtest_simulation_size=simulation_size,
        backtest_initial_bankroll_bnb=bt_bankroll,
        backtest=backtest_cfg,
    )
