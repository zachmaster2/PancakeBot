"""Load and save on-chain contract constants (min bet, treasury fee, interval, buffer) to disk."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pancakebot.util import InvariantError
from pancakebot import paths as _paths


_DEFAULT_PATH = Path(_paths.CONTRACT_CONSTANTS_PATH)


@dataclass(frozen=True, slots=True)
class ContractConstants:
    min_bet_amount_bnb: float
    treasury_fee_fraction: float
    round_interval_seconds: int
    round_close_buffer_seconds: int


def load_contract_constants(*, path: Path | None = None) -> ContractConstants:
    """Load cached contract constants from disk. Raises if missing."""
    cache_path = _DEFAULT_PATH if path is None else Path(path)
    if not cache_path.exists():
        raise InvariantError(f"contract_constants_cache_missing: {cache_path} (run --sync first)")
    try:
        obj = json.loads(cache_path.read_text())
    except Exception as e:
        raise InvariantError(f"contract_constants_cache_parse_failed: {cache_path} err={e}") from e

    if not isinstance(obj, dict):
        raise InvariantError("contract_constants_cache_not_object")

    try:
        min_bet_amount_bnb = float(obj["min_bet_amount_bnb"])
        treasury_fee_fraction = float(obj["treasury_fee_fraction"])
        round_interval_seconds = int(obj["round_interval_seconds"])
        round_close_buffer_seconds = int(obj["round_close_buffer_seconds"])
    except Exception as e:
        raise InvariantError(f"contract_constants_cache_missing_fields: err={e}") from e

    if min_bet_amount_bnb <= 0.0:
        raise InvariantError("contract_constants_min_bet_nonpositive")
    if not (0.0 <= treasury_fee_fraction < 1.0):
        raise InvariantError("contract_constants_treasury_fee_out_of_range")
    if round_interval_seconds <= 0:
        raise InvariantError("contract_constants_interval_nonpositive")
    if round_close_buffer_seconds < 0:
        raise InvariantError("contract_constants_buffer_negative")

    return ContractConstants(
        min_bet_amount_bnb=min_bet_amount_bnb,
        treasury_fee_fraction=treasury_fee_fraction,
        round_interval_seconds=round_interval_seconds,
        round_close_buffer_seconds=round_close_buffer_seconds,
    )


def save_contract_constants(*, constants: ContractConstants, path: Path | None = None) -> Path:
    """Save contract constants to disk."""
    cache_path = _DEFAULT_PATH if path is None else Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "min_bet_amount_bnb": constants.min_bet_amount_bnb,
        "treasury_fee_fraction": constants.treasury_fee_fraction,
        "round_interval_seconds": constants.round_interval_seconds,
        "round_close_buffer_seconds": constants.round_close_buffer_seconds,
    }
    cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return cache_path
