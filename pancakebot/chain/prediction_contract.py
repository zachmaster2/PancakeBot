"""Web3 wrapper for PancakeSwap Prediction V2: round reads, claim/bet writes, batched RPC calls."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Literal, Sequence, TypeVar

from web3 import Web3
from web3.exceptions import TimeExhausted

from pancakebot.constants import (
    BNB_WEI,
    EXPECTED_CHAIN_ID,
    MAX_GAS_PRICE_WEI,
    PREDICTION_V2_CONTRACT_ADDRESS,
    TREASURY_FEE_DIVISOR,
)
from pancakebot.chain.contract_config import Web3ContractConfig
from pancakebot.util import GasPriceCapBreachedError, InvariantError, TransientRpcError

_T = TypeVar("_T")

# Chainlink oracle price scale: BNB/USD feed uses 8 decimal places.
_ORACLE_PRICE_SCALE = 1e8


@dataclass(frozen=True, slots=True)
class RoundData:
    """Structured round data fetched from the on-chain rounds() call."""
    epoch: int
    start_ts: int
    lock_ts: int
    close_ts: int
    lock_price_usd: float   # Chainlink oracle price / 1e8
    close_price_usd: float
    bull_amount_wei: int
    bear_amount_wei: int
    oracle_called: bool


def _load_abi_list(path: str) -> list[dict[str, Any]]:
    """Load ABI JSON from a file.

    Requirement (locked): the file must contain a JSON *list* (the ABI array).
    We intentionally do NOT accept {'abi': [...]} to keep inputs unambiguous.
    """
    try:
        with open(path, "r") as f:
            obj = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise InvariantError(f"abi_load_failed: {e}") from e

    if not isinstance(obj, list):
        raise InvariantError("abi_json_must_be_list")

    for i, item in enumerate(obj):
        if not isinstance(item, dict):
            raise InvariantError(f"abi_item_not_object: idx={i}")
    return obj  # type: ignore[return-value]


def _canonical_abi_type(component: dict[str, Any]) -> str:
    """Build the canonical eth_abi type string for one ABI input/output entry.

    Handles the tuple-inlining that ``eth_abi.codec.decode`` requires:
    a JSON ABI entry of ``{"type": "tuple[]", "components": [...]}`` must
    be passed to the codec as ``"(uint8,uint256,bool)[]"`` (the
    components inlined inside parentheses, with the array suffix
    preserved).
    """
    t = component["type"]
    if t.startswith("tuple"):
        inner = ",".join(_canonical_abi_type(c) for c in component.get("components", []))
        suffix = t[len("tuple"):]  # "", "[]", "[N]" — preserve array suffix verbatim
        return f"({inner}){suffix}"
    return t


def derive_abi_output_types(abi: list[dict[str, Any]], function_name: str) -> list[str]:
    """Return the ABI output type strings for the named function.

    Single source of truth: types live in the JSON ABI file only; Python
    code derives them at runtime via this helper rather than re-declaring
    them in local tuples. Eliminates the drift class of bug caught
    2026-05-23 when ``close_ts_batch`` hand-wrote an 11-field tuple for
    a 14-field on-chain ``rounds()`` struct and crashed at the first
    bear-amount that ended on a non-{0x00,0x01} byte.

    Tuple outputs are returned in canonical eth_abi form (components
    inlined inside parentheses), so the result is directly usable as
    the ``types`` argument to ``codec.decode``.

    Args:
        abi: parsed JSON ABI list (one entry per function/event/error).
        function_name: the function's ``name`` field as declared in ABI.

    Returns:
        Output type strings in declaration order. For ``rounds()`` this is
        14 entries ending in ``"bool"``. For ``getUserRounds()`` this is
        ``["uint256[]", "(uint8,uint256,bool)[]", "uint256"]``.

    Raises:
        InvariantError: if ``function_name`` isn't found as a function in
            the ABI. (Typos in call sites surface loudly rather than
            silently producing an empty type list.)
    """
    for entry in abi:
        if entry.get("type") == "function" and entry.get("name") == function_name:
            return [_canonical_abi_type(out) for out in entry.get("outputs", [])]
    raise InvariantError(f"abi_function_not_found: {function_name}")


@dataclass(frozen=True, slots=True)
class BetEvent:
    wallet_address: str
    epoch: int
    amount_wei: int
    position: Literal["Bull", "Bear"]
    block_number: int
    block_timestamp: int
    tx_hash: str
    log_index: int


@dataclass(frozen=True, slots=True)
class TxSubmitResult:
    tx_hash: str
    t_tx_signed_mono_ms: float
    t_tx_hash_received_mono_ms: float
    t_receipt_confirmed_mono_ms: float | None
    included_block_number: int | None
    included_block_timestamp: int | None


@dataclass(frozen=True, slots=True)
class ClaimSubmitResult:
    """Outcome of a ``claim()`` submission.

    ``status`` is one of:
      - ``"success"``: receipt arrived with status=1 (chain accepted the claim;
        bankroll credit reflects on next ``wallet_balance_bnb`` read).
      - ``"revert"``: receipt arrived with status=0 (chain rejected; gas was
        burned, no bankroll credit). Caller logs + alerts; no retry.
      - ``"timeout"``: ``wait_for_transaction_receipt`` exceeded its timeout
        (TX may still mine later). Caller logs + alerts; no retry. Next
        iteration's ``claim_scan_cursor`` will re-detect the still-claimable
        epochs and try again.

    ``total_amount_wei`` is the sum of ``amount`` fields from the
    ``Claim(sender, epoch, amount)`` events emitted by the TX. Populated
    only on ``"success"`` (where the receipt's logs are decodable); ``None``
    for ``"revert"`` (no events emitted) and ``"timeout"`` (no receipt yet).
    The operator-facing CLAIM log line consumes this to lead with the actual
    BNB received rather than just the epoch count.
    """
    tx_hash: str
    status: Literal["success", "revert", "timeout"]
    included_block_number: int | None
    included_block_timestamp: int | None
    total_amount_wei: int | None = None


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
        self._rpc_urls = list(cfg.rpc_urls) if cfg.rpc_urls else [cfg.rpc_url]
        self._rpc_index = 0

        # Create a web3 instance + contract per RPC URL so each keeps
        # its own persistent session (warm TLS connection).
        pk = cfg.private_key.strip()
        if pk.startswith("0x"):
            pk = pk[2:]

        abi = _load_abi_list(cfg.abi_json_path)
        # Keep the raw JSON list around as the SSOT for codec.decode type
        # derivation (see derive_abi_output_types). web3.py v6+ wraps the
        # ABI in typed objects inside the Contract; we prefer the raw list
        # to stay decoupled from web3.py-version-dependent internals.
        self._abi_raw: list[dict[str, Any]] = abi
        contract_addr = Web3.to_checksum_address(PREDICTION_V2_CONTRACT_ADDRESS)

        # BSC is a POA chain (Lorentz hardfork): block.extraData is 280
        # bytes (validator signatures), but stock web3.py expects 32. Without
        # this middleware, any ``eth.get_block(...)`` raises ExtraDataLengthError
        # — caught 2026-05-21 when the live bot's ``block_timestamp`` call
        # inside ``_submit_tx_with_timing`` crashed after a successfully-submitted
        # bet TX. The send path (``send_raw_transaction``) didn't trip it
        # because raw tx submission doesn't decode the block header.
        from web3.middleware import ExtraDataToPOAMiddleware
        self._providers: list[tuple[Web3, Any]] = []
        for url in self._rpc_urls:
            w3 = Web3(Web3.HTTPProvider(url))
            w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
            c = w3.eth.contract(address=contract_addr, abi=abi)
            self._providers.append((w3, c))

        # Verify chain ID on the primary URL.
        w3_primary = self._providers[0][0]
        chain_id = int(w3_primary.eth.chain_id)
        if chain_id != int(EXPECTED_CHAIN_ID):
            raise InvariantError(f"unexpected_chain_id: got={chain_id} expected={EXPECTED_CHAIN_ID}")

        self._w3 = w3_primary
        self._contract = self._providers[0][1]
        # Account is optional -- only needed for live mode (signing transactions).
        self._account: Any = w3_primary.eth.account.from_key(pk) if len(pk) == 64 else None

    def _rotate_rpc(self) -> None:
        """Round-robin to next RPC URL, switching to its warm session."""
        self._rpc_index = (self._rpc_index + 1) % len(self._providers)
        self._w3, self._contract = self._providers[self._rpc_index]

    def _require_account(self) -> Any:
        """Return self._account or raise if unset (live-only operations)."""
        account = self._account
        if account is None:
            raise InvariantError("account_required_for_signing")
        return account

    @property
    def wallet_address(self) -> str:
        account: Any = self._account
        if account is None:
            return ""
        return str(account.address)

    def _rpc_call(self, *, op: str, fn: Callable[[], _T]) -> _T:
        self._rotate_rpc()
        try:
            return fn()
        except (InvariantError, TransientRpcError):
            raise
        except Exception as e:
            raise TransientRpcError(f"{str(op)}_failed: {e}") from e

    def _batch_eth_calls(self, encoded_calls: list[str]) -> list[bytes | None]:
        """Send multiple eth_call requests in one JSON-RPC batch.

        Each entry in *encoded_calls* is a hex-encoded calldata string
        (from ``_encode_transaction_data()``) for the prediction contract.
        Returns raw response bytes (or None for failed calls) in order.
        """
        contract_addr = str(self._contract.address)
        batch = [
            {
                "jsonrpc": "2.0",
                "id": i,
                "method": "eth_call",
                "params": [{"to": contract_addr, "data": data}, "latest"],
            }
            for i, data in enumerate(encoded_calls)
        ]
        import urllib.request
        self._rotate_rpc()
        rpc_url = self._rpc_urls[self._rpc_index]
        req = urllib.request.Request(
            rpc_url,
            data=json.dumps(batch).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            resp = urllib.request.urlopen(req, timeout=30)
            body = json.loads(resp.read())
        except Exception as e:
            raise TransientRpcError(f"batch_eth_call_failed: {e}") from e

        if not isinstance(body, list):
            raise TransientRpcError(f"batch_eth_call_bad_response: {type(body)}")

        by_id: dict[int, str | None] = {}
        for item in body:
            by_id[item.get("id")] = item.get("result")

        out: list[bytes | None] = []
        for i in range(len(encoded_calls)):
            raw_hex = by_id.get(i)
            if raw_hex is None or not isinstance(raw_hex, str):
                out.append(None)
            else:
                out.append(bytes.fromhex(raw_hex[2:] if raw_hex.startswith("0x") else raw_hex))
        return out

    def wallet_balance_bnb(self, wallet_address: str) -> float:
        """Return native BNB balance for a wallet address (as BNB float)."""
        checksum_address = Web3.to_checksum_address(str(wallet_address))
        wei = self._rpc_call(
            op="wallet_balance_bnb",
            fn=lambda: Web3.to_int(self._w3.eth.get_balance(checksum_address)),
        )
        return float(wei) / float(BNB_WEI)

    # ---- Read calls ----

    def current_epoch(self) -> int:
        return int(self._rpc_call(op="current_epoch", fn=lambda: self._contract.functions.currentEpoch().call()))

    def min_bet_amount(self) -> int:
        return int(self._rpc_call(op="min_bet_amount", fn=lambda: self._contract.functions.minBetAmount().call()))

    def treasury_fee_rate(self) -> float:
        fee_bps = int(self._rpc_call(op="treasury_fee_rate", fn=lambda: self._contract.functions.treasuryFee().call()))
        return float(fee_bps) / float(TREASURY_FEE_DIVISOR)

    def interval_seconds(self) -> int:
        return int(self._rpc_call(op="interval_seconds", fn=lambda: self._contract.functions.intervalSeconds().call()))

    def buffer_seconds(self) -> int:
        return int(self._rpc_call(op="buffer_seconds", fn=lambda: self._contract.functions.bufferSeconds().call()))

    def lock_ts(self, epoch: int) -> int:
        r = self._rpc_call(op="lock_ts", fn=lambda: self._contract.functions.rounds(int(epoch)).call())
        # Indices stable for PredictionV2 rounds tuple:
        # (epoch, startTimestamp, lockTimestamp, closeTimestamp, ...)
        return int(r[2])

    def close_ts(self, epoch: int) -> int:
        r = self._rpc_call(op="close_ts", fn=lambda: self._contract.functions.rounds(int(epoch)).call())
        return int(r[3])

    def round_data(self, epoch: int) -> "RoundData":
        """Fetch structured round data for a given epoch.

        Tuple indices for PredictionV2 rounds():
          [0]=epoch  [1]=startTs  [2]=lockTs  [3]=closeTs
          [4]=lockPrice  [5]=closePrice  [6]=lockOracleId  [7]=closeOracleId
          [8]=totalAmount  [9]=bullAmount  [10]=bearAmount
          [11]=rewardBaseCalAmount  [12]=rewardAmount  [13]=oracleCalled
        """
        r = self._rpc_call(op="round_data", fn=lambda: self._contract.functions.rounds(int(epoch)).call())
        return RoundData(
            epoch=int(r[0]),
            start_ts=int(r[1]),
            lock_ts=int(r[2]),
            close_ts=int(r[3]),
            lock_price_usd=float(r[4]) / _ORACLE_PRICE_SCALE,
            close_price_usd=float(r[5]) / _ORACLE_PRICE_SCALE,
            bull_amount_wei=int(r[9]),
            bear_amount_wei=int(r[10]),
            oracle_called=bool(r[13]),
        )

    def latest_block_number(self) -> int:
        try:
            return int(Web3.to_int(self._w3.eth.block_number))
        except Exception as e:
            raise TransientRpcError(f"latest_block_number_failed: {e}") from e

    def block_timestamp(self, block_number: int) -> int:
        if int(block_number) < 0:
            raise InvariantError("block_number_negative")
        try:
            b = self._w3.eth.get_block(int(block_number))
        except Exception as e:
            raise TransientRpcError(f"block_timestamp_failed: block={int(block_number)} err={e}") from e
        return int(b["timestamp"])

    def fetch_bet_events_for_epoch(
        self,
        *,
        epoch: int,
        from_block: int,
        to_block: int,
    ) -> list[BetEvent]:
        if int(epoch) <= 0:
            raise InvariantError("event_epoch_nonpositive")
        if int(from_block) <= 0:
            raise InvariantError("event_from_block_nonpositive")
        if int(to_block) < int(from_block):
            raise InvariantError("event_block_range_invalid")

        try:
            bull_logs = list(
                self._contract.events.BetBull().get_logs(
                    argument_filters={"epoch": int(epoch)},
                    from_block=int(from_block),
                    to_block=int(to_block),
                )
            )
            bear_logs = list(
                self._contract.events.BetBear().get_logs(
                    argument_filters={"epoch": int(epoch)},
                    from_block=int(from_block),
                    to_block=int(to_block),
                )
            )
        except Exception as e:
            raise TransientRpcError(f"event_log_fetch_failed: {e}") from e

        out: list[BetEvent] = []
        block_ts_cache: dict[int, int] = {}

        def _mk(log_obj: Any, side: Literal["Bull", "Bear"]) -> BetEvent:
            args = log_obj["args"]
            bn = int(log_obj["blockNumber"])
            if bn not in block_ts_cache:
                block_ts_cache[bn] = int(self.block_timestamp(int(bn)))
            return BetEvent(
                wallet_address=str(args["sender"]),
                epoch=int(args["epoch"]),
                amount_wei=int(args["amount"]),
                position=side,
                block_number=int(bn),
                block_timestamp=int(block_ts_cache[int(bn)]),
                tx_hash=str(log_obj["transactionHash"].hex()),
                log_index=int(log_obj["logIndex"]),
            )

        for ev in bull_logs:
            out.append(_mk(ev, "Bull"))
        for ev in bear_logs:
            out.append(_mk(ev, "Bear"))

        out.sort(key=lambda x: (int(x.block_number), int(x.log_index)))
        return out

    def suggest_gas_price_wei(self) -> int:
        """Return the node-suggested gas price (wei)."""
        return int(self._rpc_call(op="suggest_gas_price_wei", fn=lambda: Web3.to_int(self._w3.eth.gas_price)))

    def assert_gas_cap_not_breached(self) -> None:
        """Validates that eth.gas_price <= MAX_GAS_PRICE_WEI.

        Live bet/claim TXs are posted at MAX_GAS_PRICE_WEI (the worst-case
        ceiling). If the node-suggested gas price exceeds the cap, the
        ceiling is below current network reality and live TXs paid at MAX
        would land at the back of the priority queue (likely to miss the
        lock-block inclusion window).

        Raises:
            GasPriceCapBreachedError: when ``eth.gas_price > MAX_GAS_PRICE_WEI``.
                The caller should skip the bet/claim, alert the operator,
                and continue running. The operator must lift the cap and
                review before resuming.

        Logs warning and returns (does NOT raise) on:
            - RPC failure fetching ``eth.gas_price``. Proceed with MAX
              (single-round transient hiccups shouldn't block the bet).
            - ``eth.gas_price`` == 0. Misbehaving node; proceed with MAX.
        """
        try:
            suggested = self.suggest_gas_price_wei()
        except TransientRpcError:
            # Don't crash the round on a transient RPC hiccup; proceed with MAX.
            # The check repeats next round; sustained outages surface elsewhere
            # (bankroll wake fetch, etc.).
            return
        if suggested == 0:
            # Misbehaving node returned 0 — can't validate. Proceed with MAX;
            # the bet still goes out at the ceiling.
            return
        if suggested > MAX_GAS_PRICE_WEI:
            raise GasPriceCapBreachedError(
                f"eth.gas_price={suggested} > MAX_GAS_PRICE_WEI={MAX_GAS_PRICE_WEI}; "
                f"raise the cap and review before resuming"
            )

    def get_user_rounds_length(self, wallet_address: str) -> int:
        checksum_address = Web3.to_checksum_address(str(wallet_address))
        return int(
            self._rpc_call(
                op="get_user_rounds_length",
                fn=lambda: self._contract.functions.getUserRoundsLength(checksum_address).call(),
            )
        )

    def get_user_rounds(self, *, wallet_address: str, cursor: int, size: int) -> Sequence[int]:
        checksum_address = Web3.to_checksum_address(str(wallet_address))
        values = self._rpc_call(
            op="get_user_rounds",
            fn=lambda: self._contract.functions.getUserRounds(
                checksum_address,
                int(cursor),
                int(size),
            ).call(),
        )
        epochs = values[0]
        return [int(x) for x in epochs]

    def get_user_rounds_all_batched(
        self, *, wallet_address: str, cursor: int, total: int, page_size: int = 100,
    ) -> list[int]:
        """Fetch all user round epochs from cursor to total in one RPC batch."""
        checksum = Web3.to_checksum_address(str(wallet_address))
        encoded: list[str] = []
        for offset in range(cursor, total, page_size):
            size = min(page_size, total - offset)
            # noinspection PyProtectedMember
            encoded.append(
                self._contract.functions.getUserRounds(checksum, offset, size)._encode_transaction_data()
            )
        if not encoded:
            return []

        results = self._batch_eth_calls(encoded)
        # Derive output types from the ABI (SSOT) rather than hand-writing
        # them here. getUserRounds returns (uint256[], uint256).
        user_rounds_types = derive_abi_output_types(self._abi_raw, "getUserRounds")
        all_epochs: list[int] = []
        for raw in results:
            if raw is None:
                continue
            decoded = self._w3.codec.decode(user_rounds_types, raw)
            all_epochs.extend(int(x) for x in decoded[0])
        return all_epochs

    def close_ts_batch(self, epochs: list[int]) -> dict[int, int | None]:
        """Fetch close_ts for multiple epochs in one RPC batch."""
        if not epochs:
            return {}
        # noinspection PyProtectedMember
        encoded = [
            self._contract.functions.rounds(int(e))._encode_transaction_data()
            for e in epochs
        ]
        # Derive output types from the ABI (SSOT). rounds() returns a 14-
        # field tuple per abi/prediction_v2_abi.json:
        #   epoch, startTimestamp, lockTimestamp, closeTimestamp,
        #   lockPrice, closePrice,
        #   lockOracleId, closeOracleId,
        #   totalAmount, bullAmount, bearAmount,
        #   rewardBaseCalAmount, rewardAmount,
        #   oracleCalled
        # close_ts lives at index 3.
        round_types = derive_abi_output_types(self._abi_raw, "rounds")
        # Batch in chunks of 100 (BSC free RPC limit).
        out: dict[int, int | None] = {}
        for chunk_start in range(0, len(encoded), 100):
            chunk_encoded = encoded[chunk_start:chunk_start + 100]
            chunk_epochs = epochs[chunk_start:chunk_start + 100]
            results = self._batch_eth_calls(chunk_encoded)
            for e, raw in zip(chunk_epochs, results):
                if raw is None:
                    out[int(e)] = None
                else:
                    decoded = self._w3.codec.decode(round_types, raw)
                    out[int(e)] = int(decoded[3])
        return out

    def claimable(self, *, epoch: int, wallet_address: str) -> bool:
        checksum_address = Web3.to_checksum_address(str(wallet_address))
        return bool(
            self._rpc_call(
                op="claimable",
                fn=lambda: self._contract.functions.claimable(int(epoch), checksum_address).call(),
            )
        )

    def refundable(self, *, epoch: int, wallet_address: str) -> bool:
        checksum_address = Web3.to_checksum_address(str(wallet_address))
        return bool(
            self._rpc_call(
                op="refundable",
                fn=lambda: self._contract.functions.refundable(int(epoch), checksum_address).call(),
            )
        )

    def claimable_refundable_batch(
        self, *, epochs: list[int], wallet_address: str,
    ) -> dict[int, tuple[bool, bool]]:
        """Batch-check claimable and refundable for multiple epochs.

        Returns {epoch: (claimable, refundable)} for each epoch.
        Both checks are packed into a single RPC batch (2 calls per epoch).
        """
        if not epochs:
            return {}
        checksum = Web3.to_checksum_address(str(wallet_address))
        encoded: list[str] = []
        for e in epochs:
            # noinspection PyProtectedMember
            encoded.append(
                self._contract.functions.claimable(int(e), checksum)._encode_transaction_data()
            )
            # noinspection PyProtectedMember
            encoded.append(
                self._contract.functions.refundable(int(e), checksum)._encode_transaction_data()
            )
        # Batch in chunks of 100 calls.
        all_results: list[bytes | None] = []
        for chunk_start in range(0, len(encoded), 100):
            chunk = encoded[chunk_start:chunk_start + 100]
            all_results.extend(self._batch_eth_calls(chunk))

        # Derive output types from the ABI (SSOT). claimable() and
        # refundable() both return a single bool; hand-coded ``["bool"]``
        # would be functionally equivalent today but goes through the same
        # SSOT helper as the wider-tuple decoders for consistency, so
        # nothing in this module hand-declares types anymore.
        claimable_types = derive_abi_output_types(self._abi_raw, "claimable")
        refundable_types = derive_abi_output_types(self._abi_raw, "refundable")

        out: dict[int, tuple[bool, bool]] = {}
        for i, e in enumerate(epochs):
            c_raw = all_results[i * 2]
            r_raw = all_results[i * 2 + 1]
            c = bool(self._w3.codec.decode(claimable_types, c_raw)[0]) if c_raw else False
            r = bool(self._w3.codec.decode(refundable_types, r_raw)[0]) if r_raw else False
            out[int(e)] = (c, r)
        return out

    # ---- Write calls ----

    def _submit_tx_with_timing(
        self,
        *,
        tx: dict[str, Any],
        wait_receipt: bool,
        receipt_timeout_seconds: int,
    ) -> TxSubmitResult:
        # Bundle 4 reviewer note (2026-05-14): the bracketing of
        # ``send_raw_transaction(...)`` between ``t_tx_signed`` (line below)
        # and ``t_tx_hash`` (below the try block) measures the FULL
        # round-trip — request serialize → wire-out → RPC ingest +
        # mempool insert → wire-back with txh → deserialize. Web3.py
        # implements ``send_raw_transaction`` as a synchronous JSON-RPC
        # POST that blocks until the server returns the transaction
        # hash. There is no one-way path here.
        #
        # The TX is committed to the validator's mempool at the moment
        # the RPC accepts it. Bundle 4 budgets ``BSC_BET_SUBMIT_ONE_WAY_MS=75``
        # for that one-way path — re-measured 2026-05-20 against the
        # production write-path RPC (4×100-TX probe, n=400). Modal p99
        # RTT ~80ms → ~40ms one-way; 75ms covers p99/2 with ≥12ms margin
        # in all 4 runs. See
        # var/strategy_review/2026_05_20_send_raw_tx_probe_100_at_*.md
        # for the full distributions.
        signed = self._require_account().sign_transaction(tx)
        t_tx_signed = float(time.perf_counter() * 1000.0)
        try:
            txh = self._w3.eth.send_raw_transaction(signed.raw_transaction)
        except Exception as e:
            raise TransientRpcError(f"tx_send_failed: {e}") from e
        t_tx_hash = float(time.perf_counter() * 1000.0)

        tx_hash = str(txh.hex())
        t_receipt = None
        block_number = None
        block_timestamp = None

        if bool(wait_receipt):
            if int(receipt_timeout_seconds) <= 0:
                raise InvariantError("receipt_timeout_seconds_nonpositive")
            try:
                receipt = self._w3.eth.wait_for_transaction_receipt(
                    txh,
                    timeout=float(receipt_timeout_seconds),
                    poll_latency=0.2,
                )
                t_receipt = float(time.perf_counter() * 1000.0)
                block_number = int(receipt["blockNumber"])
                block_timestamp = int(self.block_timestamp(int(block_number)))
            except TimeExhausted:
                t_receipt = None
                block_number = None
                block_timestamp = None
            except Exception as e:
                raise TransientRpcError(f"tx_receipt_wait_failed: {e}") from e

        return TxSubmitResult(
            tx_hash=str(tx_hash),
            t_tx_signed_mono_ms=float(t_tx_signed),
            t_tx_hash_received_mono_ms=float(t_tx_hash),
            t_receipt_confirmed_mono_ms=float(t_receipt) if t_receipt is not None else None,
            included_block_number=int(block_number) if block_number is not None else None,
            included_block_timestamp=int(block_timestamp) if block_timestamp is not None else None,
        )

    def _build_bet_tx(
        self,
        *,
        side: Literal["Bull", "Bear"],
        epoch: int,
        amount_wei: int,
        gas_limit: int,
        gas_price_wei: int,
    ) -> dict[str, Any]:
        if int(epoch) <= 0:
            raise InvariantError("bet_epoch_nonpositive")
        if int(amount_wei) <= 0:
            raise InvariantError("bet_amount_wei_nonpositive")
        if int(gas_limit) <= 0:
            raise InvariantError("bet_gas_limit_nonpositive")
        if int(gas_price_wei) <= 0:
            raise InvariantError("bet_gas_price_nonpositive")

        if str(side) == "Bull":
            fn = self._contract.functions.betBull(int(epoch))
        elif str(side) == "Bear":
            fn = self._contract.functions.betBear(int(epoch))
        else:
            raise InvariantError("bet_side_invalid")

        return self._rpc_call(
            op="build_bet_tx",
            fn=lambda: fn.build_transaction(
                {
                    "from": self._require_account().address,
                    "value": int(amount_wei),
                    "nonce": Web3.to_int(self._w3.eth.get_transaction_count(self._require_account().address)),
                    "gas": int(gas_limit),
                    "gasPrice": int(gas_price_wei),
                }
            ),
        )

    def bet_bull_timed(
        self,
        *,
        epoch: int,
        amount_wei: int,
        gas_limit: int,
        gas_price_wei: int,
        wait_receipt: bool,
        receipt_timeout_seconds: int,
    ) -> TxSubmitResult:
        tx = self._build_bet_tx(
            side="Bull",
            epoch=int(epoch),
            amount_wei=int(amount_wei),
            gas_limit=int(gas_limit),
            gas_price_wei=int(gas_price_wei),
        )
        return self._submit_tx_with_timing(
            tx=tx,
            wait_receipt=bool(wait_receipt),
            receipt_timeout_seconds=int(receipt_timeout_seconds),
        )

    def bet_bear_timed(
        self,
        *,
        epoch: int,
        amount_wei: int,
        gas_limit: int,
        gas_price_wei: int,
        wait_receipt: bool,
        receipt_timeout_seconds: int,
    ) -> TxSubmitResult:
        tx = self._build_bet_tx(
            side="Bear",
            epoch=int(epoch),
            amount_wei=int(amount_wei),
            gas_limit=int(gas_limit),
            gas_price_wei=int(gas_price_wei),
        )
        return self._submit_tx_with_timing(
            tx=tx,
            wait_receipt=bool(wait_receipt),
            receipt_timeout_seconds=int(receipt_timeout_seconds),
        )

    def bet_bull(self, *, epoch: int, amount_wei: int, gas_limit: int, gas_price_wei: int) -> str:
        out = self.bet_bull_timed(
            epoch=int(epoch),
            amount_wei=int(amount_wei),
            gas_limit=int(gas_limit),
            gas_price_wei=int(gas_price_wei),
            wait_receipt=False,
            receipt_timeout_seconds=1,
        )
        return str(out.tx_hash)

    def bet_bear(self, *, epoch: int, amount_wei: int, gas_limit: int, gas_price_wei: int) -> str:
        out = self.bet_bear_timed(
            epoch=int(epoch),
            amount_wei=int(amount_wei),
            gas_limit=int(gas_limit),
            gas_price_wei=int(gas_price_wei),
            wait_receipt=False,
            receipt_timeout_seconds=1,
        )
        return str(out.tx_hash)

    def claim(
        self,
        *,
        epochs: Sequence[int],
        gas_limit: int,
        gas_price_wei: int,
        wait_receipt: bool,
        receipt_timeout_seconds: int,
    ) -> ClaimSubmitResult:
        """Submit a claim() TX and (optionally) wait for receipt.

        Returns a ``ClaimSubmitResult`` whose ``status`` distinguishes
        chain success / chain revert / receipt-poll timeout. When
        ``wait_receipt`` is False the result has status="success" with
        block fields None (used only by tests); production live code path
        always passes ``wait_receipt=True``.
        """
        fn = self._contract.functions.claim([int(e) for e in epochs])
        tx = fn.build_transaction(
            {
                "from": self._require_account().address,
                "nonce": Web3.to_int(self._w3.eth.get_transaction_count(self._require_account().address)),
                "gas": int(gas_limit),
                "gasPrice": int(gas_price_wei),
            }
        )
        signed = self._require_account().sign_transaction(tx)
        try:
            txh = self._w3.eth.send_raw_transaction(signed.raw_transaction)
        except Exception as e:
            raise TransientRpcError(f"claim_tx_send_failed: {e}") from e
        tx_hash = str(txh.hex())

        if not bool(wait_receipt):
            return ClaimSubmitResult(
                tx_hash=tx_hash,
                status="success",
                included_block_number=None,
                included_block_timestamp=None,
            )

        if int(receipt_timeout_seconds) <= 0:
            raise InvariantError("claim_receipt_timeout_seconds_nonpositive")

        try:
            receipt = self._w3.eth.wait_for_transaction_receipt(
                txh,
                timeout=float(receipt_timeout_seconds),
                poll_latency=0.2,
            )
        except TimeExhausted:
            return ClaimSubmitResult(
                tx_hash=tx_hash,
                status="timeout",
                included_block_number=None,
                included_block_timestamp=None,
            )
        except Exception as e:
            raise TransientRpcError(f"claim_tx_receipt_wait_failed: {e}") from e

        block_number = int(receipt["blockNumber"])
        block_timestamp = int(self.block_timestamp(block_number))
        chain_status = int(receipt.get("status", 0))

        # Extract total BNB claimed by summing ``amount`` across the
        # PredictionV2 Claim(sender, epoch, amount) events in the receipt.
        # Only meaningful for status=success (revert emits no events).
        total_amount_wei: int | None = None
        if chain_status == 1:
            try:
                # noinspection PyProtectedMember
                claim_events = self._contract.events.Claim().process_receipt(receipt)
                total_amount_wei = sum(int(ev["args"]["amount"]) for ev in claim_events)
            except Exception:
                # Best-effort: if event decode fails for any reason, leave
                # total_amount_wei=None. The caller's log line tolerates this
                # by omitting the BNB amount rather than crashing.
                total_amount_wei = None

        return ClaimSubmitResult(
            tx_hash=tx_hash,
            status="success" if chain_status == 1 else "revert",
            included_block_number=block_number,
            included_block_timestamp=block_timestamp,
            total_amount_wei=total_amount_wei,
        )
