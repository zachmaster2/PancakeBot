from __future__ import annotations

from pathlib import Path

from pancakebot.backtest.runner import run_backtest
from pancakebot.config.env import load_env, require_env
from pancakebot.config.load_config import load_app_config
from pancakebot.core.constants import (
    BNB_WEI,
    EXPECTED_CHAIN_ID,
    PREDICTION_V2_GRAPH_ENDPOINT,
    RPC_TIMEOUT_SECONDS,
    RPC_URLS,
)
from pancakebot.infra.okx_client import OkxClient
from pancakebot.domain.strategy.momentum_gate import MomentumGate
from pancakebot.infra.closed_rounds_store import ClosedRoundsStore
from pancakebot.infra.market_data_db import MarketDataDb, SqliteKlinesStore
from pancakebot.infra.graph_client import GraphClient
from pancakebot.infra.rpc_pool import choose_rpc_url
from pancakebot.infra.onchain.web3_contract_config import Web3ContractConfig
from pancakebot.infra.onchain.web3_prediction_contract import Web3PredictionContract
from pancakebot.runtime.contract_constants_cache import ContractConstants, load_contract_constants, save_contract_constants
from pancakebot.runtime.runtime_loop import RuntimeConfig, run_live_loop
from pancakebot.core.determinism import set_global_determinism
from pancakebot.integration.sync_mode import sync_runtime_market_data
from pancakebot.core.errors import InvariantError
from pancakebot.core.logging import info






def run_from_config(*, config_path: str, dry: bool, backtest: bool, sync_only: bool) -> None:
    cfg = load_app_config(config_path)
    set_global_determinism(seed=int(cfg.random_seed))

    selected_modes = int(bool(dry)) + int(bool(backtest)) + int(bool(sync_only))
    if selected_modes > 1:
        raise InvariantError("run_modes_mutually_exclusive")

    if backtest:
        round_store = ClosedRoundsStore(cfg.closed_rounds_path)
        constants = load_contract_constants()
        market_data_store = MarketDataDb(cfg.market_data_db_path)
        try:
            market_data_store.ensure_sources_synced(
                rounds_jsonl_path=str(cfg.closed_rounds_path),
                klines_jsonl_path=str(cfg.klines_path),
            )
            runtime_cfg = RuntimeConfig(
                round_store=round_store,
                klines_store=SqliteKlinesStore(market_data_db=market_data_store),
                contract=None,
                wallet_address="",
                cutoff_seconds=cfg.cutoff_seconds,
                latency_log_path=cfg.latency_log_path,
                dry_initial_bankroll_bnb=cfg.dry_initial_bankroll_bnb,
                wait_for_bet_receipt=False,
                bet_receipt_timeout_seconds=cfg.bet_receipt_timeout_seconds,
                momentum_gate_config=cfg.momentum_gate,
                momentum_gate=None,
                dry=dry,
                runtime_state_paths=cfg.runtime_state_paths,
                min_bet_amount_bnb=float(constants.min_bet_amount_bnb),
                treasury_fee_fraction=float(constants.treasury_fee_fraction),
                buffer_seconds=int(constants.buffer_seconds),
            )
            run_backtest(runtime_cfg=runtime_cfg, backtest_cfg=cfg.backtest, out_dir=Path("var"))
        finally:
            try:
                market_data_store.close()
            except Exception:
                pass
        return

    if sync_only:
        load_env()
        graph_api_key = require_env("THE_GRAPH_API_KEY")
        graph = GraphClient(endpoint=PREDICTION_V2_GRAPH_ENDPOINT, api_key=graph_api_key)
        round_store = ClosedRoundsStore(cfg.closed_rounds_path)
        summary = sync_runtime_market_data(
            cfg=cfg,
            graph=graph,
            round_store=round_store,
        )
        info(
            "CORE",
            "SYNC",
            "DONE",
            msg=(
                f"closed_rounds={int(summary.stored_closed_round_count)} "
                f"epochs=[{int(summary.earliest_closed_epoch)}..{int(summary.latest_closed_epoch)}]"
            ),
        )
        return

    load_env()
    private_key = require_env("BSC_WALLET_PRIVATE_KEY")
    rpc_url = choose_rpc_url(
        RPC_URLS,
        expected_chain_id=int(EXPECTED_CHAIN_ID),
        timeout_seconds=int(RPC_TIMEOUT_SECONDS),
    )

    contract_cfg = Web3ContractConfig(
        rpc_url=rpc_url,
        abi_json_path=cfg.abi_json_path,
        private_key=private_key,
    )
    contract = Web3PredictionContract(contract_cfg)

    treasury_fee_fraction = contract.treasury_fee_rate()
    buffer_seconds = contract.buffer_seconds()
    min_bet_amount_bnb = float(contract.min_bet_amount()) / float(BNB_WEI)
    save_contract_constants(
        constants=ContractConstants(
            min_bet_amount_bnb=float(min_bet_amount_bnb),
            treasury_fee_fraction=float(treasury_fee_fraction),
            buffer_seconds=int(buffer_seconds),
        )
    )

    momentum_gate = None
    if bool(cfg.momentum_gate.enabled):
        okx_client = OkxClient(timeout_seconds=10.0)
        momentum_gate = MomentumGate(config=cfg.momentum_gate, okx_client=okx_client)

    runtime_cfg = RuntimeConfig(
        round_store=None,
        klines_store=None,
        contract=contract,
        wallet_address=contract.wallet_address,
        cutoff_seconds=cfg.cutoff_seconds,
        latency_log_path=cfg.latency_log_path,
        dry_initial_bankroll_bnb=cfg.dry_initial_bankroll_bnb,
        wait_for_bet_receipt=cfg.wait_for_bet_receipt,
        bet_receipt_timeout_seconds=cfg.bet_receipt_timeout_seconds,
        momentum_gate_config=cfg.momentum_gate,
        dry=dry,
        runtime_state_paths=cfg.runtime_state_paths,
        min_bet_amount_bnb=float(min_bet_amount_bnb),
        treasury_fee_fraction=treasury_fee_fraction,
        buffer_seconds=buffer_seconds,
        momentum_gate=momentum_gate,
    )

    run_live_loop(runtime_cfg)
