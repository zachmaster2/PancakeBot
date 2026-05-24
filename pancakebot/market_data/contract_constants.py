"""Load and save on-chain contract constants (min bet, treasury fee, interval, buffer) to disk."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pancakebot.util import InvariantError
from pancakebot import paths as _paths


_DEFAULT_PATH = Path(_paths.CONTRACT_CONSTANTS_PATH)


@dataclass(frozen=True, slots=True)
class ContractConstants:
    min_bet_amount_bnb: float
    treasury_fee_fraction: float
    interval_seconds: int
    buffer_seconds: int


def load_contract_constants(*, path: Path | None = None) -> ContractConstants:
    """Load cached contract constants from disk. Raises if missing."""
    cache_path = _DEFAULT_PATH if path is None else Path(path)
    if not cache_path.exists():
        # Populated by app.run_from_config in --sync, --dry, and --live
        # branches (each calls fetch_and_save_contract_constants at startup).
        # If it's still missing after --sync ran, the on-chain reads or
        # disk write failed — not user error.
        raise InvariantError(
            f"contract_constants_cache_missing: {cache_path} "
            f"(populated by --sync / --dry / --live startup; if --sync ran "
            f"successfully and this file is still missing, check RPC "
            f"connectivity to BSC mainnet and disk-write permissions on {cache_path.parent})"
        )
    try:
        obj = json.loads(cache_path.read_text())
    except Exception as e:
        raise InvariantError(f"contract_constants_cache_parse_failed: {cache_path} err={e}") from e

    if not isinstance(obj, dict):
        raise InvariantError("contract_constants_cache_not_object")

    try:
        min_bet_amount_bnb = float(obj["min_bet_amount_bnb"])
        treasury_fee_fraction = float(obj["treasury_fee_fraction"])
        interval_seconds = int(obj["interval_seconds"])
        buffer_seconds = int(obj["buffer_seconds"])
    except Exception as e:
        raise InvariantError(f"contract_constants_cache_missing_fields: err={e}") from e

    if min_bet_amount_bnb <= 0.0:
        raise InvariantError("contract_constants_min_bet_nonpositive")
    if not (0.0 <= treasury_fee_fraction < 1.0):
        raise InvariantError("contract_constants_treasury_fee_out_of_range")
    if interval_seconds <= 0:
        raise InvariantError("contract_constants_interval_nonpositive")
    if buffer_seconds < 0:
        raise InvariantError("contract_constants_buffer_negative")

    return ContractConstants(
        min_bet_amount_bnb=min_bet_amount_bnb,
        treasury_fee_fraction=treasury_fee_fraction,
        interval_seconds=interval_seconds,
        buffer_seconds=buffer_seconds,
    )


def save_contract_constants(*, constants: ContractConstants, path: Path | None = None) -> Path:
    """Save contract constants to disk."""
    cache_path = _DEFAULT_PATH if path is None else Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "min_bet_amount_bnb": constants.min_bet_amount_bnb,
        "treasury_fee_fraction": constants.treasury_fee_fraction,
        "interval_seconds": constants.interval_seconds,
        "buffer_seconds": constants.buffer_seconds,
    }
    cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return cache_path


def fetch_and_save_contract_constants(
    contract: Any,
    *,
    path: Path | None = None,
) -> ContractConstants:
    """Fetch contract constants from chain and persist to disk.

    Single source of truth for the "read 4 contract constants + save"
    pattern. Called by ``app.run_from_config`` for all three modes that
    need the cache populated:

      --sync : called before sync_runtime_market_data so the Graph
               parser (graph_client._parse_round → load_contract_constants)
               sees a fresh cache. Fixes the bootstrap loop where
               --sync depended on a cache it didn't populate.
      --dry  : called at startup; refresh on every run keeps the cache
               in lockstep with current chain state.
      --live : same as --dry.

    Reads (4 chain calls, ~hundreds of ms total):
      min_bet_amount()       (wei)
      treasury_fee_rate()    (fraction in [0, 1))
      interval_seconds()     (int)
      buffer_seconds()       (int)

    Args:
        contract: a ``Web3PredictionContract``-shaped object. Duck-typed
            with ``Any`` to avoid a chain → market_data import cycle.
        path: optional override of the default cache path.

    Returns:
        The ``ContractConstants`` dataclass that was persisted.
    """
    # Imported locally so this module stays free of chain-package
    # dependencies (chain.constants is fine — it's a leaf module).
    from pancakebot.constants import BNB_WEI

    constants = ContractConstants(
        min_bet_amount_bnb=float(contract.min_bet_amount()) / float(BNB_WEI),
        treasury_fee_fraction=float(contract.treasury_fee_rate()),
        interval_seconds=int(contract.interval_seconds()),
        buffer_seconds=int(contract.buffer_seconds()),
    )
    save_contract_constants(constants=constants, path=path)
    return constants
