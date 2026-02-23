from __future__ import annotations

import json
from typing import Any, Sequence

from web3 import Web3

from pancakebot.core.constants import (
    BNB_WEI,
    EXPECTED_CHAIN_ID,
    PREDICTION_V2_CONTRACT_ADDRESS,
    TREASURY_FEE_DIVISOR,
)
from pancakebot.infra.onchain.web3_contract_config import Web3ContractConfig
from pancakebot.core.errors import InvariantError


def _load_abi_list(path: str) -> list[dict[str, Any]]:
    """Load ABI JSON from a file.

    Requirement (locked): the file must contain a JSON *list* (the ABI array).
    We intentionally do NOT accept {'abi': [...]} to keep inputs unambiguous.
    """
    try:
        with open(path, 'r') as f:
            obj = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise InvariantError(f'abi_load_failed: {e}') from e

    if not isinstance(obj, list):
        raise InvariantError('abi_json_must_be_list')

    for i, item in enumerate(obj):
        if not isinstance(item, dict):
            raise InvariantError(f'abi_item_not_object: idx={i}')
    return obj  # type: ignore[return-value]


class Web3PredictionContract:
    """Thin Web3 wrapper for Pancake PredictionV2.

    Hardcoded (locked):
      - contract address (BNB mainnet)
      - expected chain id
      - treasury fee divisor (bps)

    Configuration:
      - rpc_url selected by RpcPool
      - abi_json_path from config.toml
      - private_key from .env
    """

    def __init__(self, cfg: Web3ContractConfig):
        self._cfg = cfg

        w3 = Web3(Web3.HTTPProvider(cfg.rpc_url))
        chain_id = int(w3.eth.chain_id)
        if chain_id != int(EXPECTED_CHAIN_ID):
            raise InvariantError(f'unexpected_chain_id: got={chain_id} expected={EXPECTED_CHAIN_ID}')

        pk = cfg.private_key.strip()
        if pk.startswith('0x'):
            pk = pk[2:]
        if len(pk) != 64:
            raise InvariantError('private_key_must_be_32_bytes')

        self._w3 = w3
        self._account = w3.eth.account.from_key(pk)

        abi = _load_abi_list(cfg.abi_json_path)
        self._contract = w3.eth.contract(
            address=Web3.to_checksum_address(PREDICTION_V2_CONTRACT_ADDRESS),
            abi=abi,
        )

    @property
    def wallet_address(self) -> str:
        return str(self._account.address)

    def wallet_balance_bnb(self, wallet_address: str) -> float:
        """Return native BNB balance for a wallet address (as BNB float)."""
        wei = Web3.to_int(self._w3.eth.get_balance(Web3.to_checksum_address(str(wallet_address))))
        return float(wei) / float(BNB_WEI)

    # ---- Read calls ----

    def current_epoch(self) -> int:
        return int(self._contract.functions.currentEpoch().call())

    def min_bet_amount(self) -> int:
        return int(self._contract.functions.minBetAmount().call())

    def treasury_fee_rate(self) -> float:
        fee_bps = int(self._contract.functions.treasuryFee().call())
        return float(fee_bps) / float(TREASURY_FEE_DIVISOR)

    def buffer_seconds(self) -> int:
        """Protocol bufferSeconds constant (seconds)."""
        value = self._contract.functions.bufferSeconds().call()
        return int(value)

    def lock_ts(self, epoch: int) -> int:
        r = self._contract.functions.rounds(int(epoch)).call()
        # Indices stable for PredictionV2 rounds tuple:
        # (epoch, startTimestamp, lockTimestamp, closeTimestamp, ...)
        return int(r[2])

    def close_ts(self, epoch: int) -> int:
        r = self._contract.functions.rounds(int(epoch)).call()
        return int(r[3])

    def suggest_gas_price_wei(self) -> int:
        """Return the node-suggested gas price (wei)."""
        return Web3.to_int(self._w3.eth.gas_price)

    def get_user_rounds_length(self, wallet_address: str) -> int:
        return int(self._contract.functions.getUserRoundsLength(Web3.to_checksum_address(str(wallet_address))).call())

    def get_user_rounds(self, *, wallet_address: str, cursor: int, size: int) -> Sequence[int]:
        values = self._contract.functions.getUserRounds(
            Web3.to_checksum_address(str(wallet_address)),
            int(cursor),
            int(size),
        ).call()
        epochs = values[0]
        return [int(x) for x in epochs]

    def claimable(self, *, epoch: int, wallet_address: str) -> bool:
        return bool(
            self._contract.functions.claimable(int(epoch), Web3.to_checksum_address(str(wallet_address))).call()
        )

    def refundable(self, *, epoch: int, wallet_address: str) -> bool:
        return bool(
            self._contract.functions.refundable(int(epoch), Web3.to_checksum_address(str(wallet_address))).call()
        )

    # ---- Write calls ----

    def bet_bull(self, *, epoch: int, amount_wei: int, gas_limit: int, gas_price_wei: int) -> str:
        fn = self._contract.functions.betBull(int(epoch))
        tx = fn.build_transaction(
            {
                'from': self._account.address,
                'value': int(amount_wei),
                'nonce': Web3.to_int(self._w3.eth.get_transaction_count(self._account.address)),
                'gas': int(gas_limit),
                'gasPrice': int(gas_price_wei),
            }
        )
        signed = self._account.sign_transaction(tx)
        txh = self._w3.eth.send_raw_transaction(signed.raw_transaction)
        return str(txh.hex())

    def bet_bear(self, *, epoch: int, amount_wei: int, gas_limit: int, gas_price_wei: int) -> str:
        fn = self._contract.functions.betBear(int(epoch))
        tx = fn.build_transaction(
            {
                'from': self._account.address,
                'value': int(amount_wei),
                'nonce': Web3.to_int(self._w3.eth.get_transaction_count(self._account.address)),
                'gas': int(gas_limit),
                'gasPrice': int(gas_price_wei),
            }
        )
        signed = self._account.sign_transaction(tx)
        txh = self._w3.eth.send_raw_transaction(signed.raw_transaction)
        return str(txh.hex())

    def claim(self, *, epochs: Sequence[int], gas_limit: int, gas_price_wei: int) -> str:
        fn = self._contract.functions.claim([int(e) for e in epochs])
        tx = fn.build_transaction(
            {
                'from': self._account.address,
                'nonce': Web3.to_int(self._w3.eth.get_transaction_count(self._account.address)),
                'gas': int(gas_limit),
                'gasPrice': int(gas_price_wei),
            }
        )
        signed = self._account.sign_transaction(tx)
        txh = self._w3.eth.send_raw_transaction(signed.raw_transaction)
        return str(txh.hex())
