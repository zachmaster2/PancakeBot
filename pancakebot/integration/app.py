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
from pancakebot.infra.graph_client import GraphClient
from pancakebot.infra.rpc_pool import choose_rpc_url
from pancakebot.infra.onchain.web3_contract_config import Web3ContractConfig
from pancakebot.infra.onchain.web3_prediction_contract import Web3PredictionContract
from pancakebot.runtime.contract_constants_cache import ContractConstants, save_contract_constants
from pancakebot.runtime.runtime_loop import RuntimeConfig, run_live_loop
from pancakebot.integration.sync_mode import sync_runtime_market_data
from pancakebot.core.errors import InvariantError
from pancakebot.core.logging import info
from pancakebot.infra.pool_event_watcher import PoolEventWatcher






def run_from_config(*, config_path: str, dry: bool, backtest: bool, sync: bool) -> None:
    cfg = load_app_config(config_path)

    selected_modes = int(dry) + int(backtest) + int(sync)
    if selected_modes > 1:
        raise InvariantError("run_modes_mutually_exclusive")

    if backtest:
        round_store = ClosedRoundsStore(cfg.closed_rounds_path)
        runtime_cfg = RuntimeConfig(
            round_store=round_store,
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
            min_bet_amount_bnb=float(cfg.min_bet_amount_bnb),
            treasury_fee_fraction=float(cfg.treasury_fee_fraction),
        )
        run_backtest(runtime_cfg=runtime_cfg, backtest_cfg=cfg.backtest, out_dir=Path("var"))
        return

    if sync:
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
                f"epochs=[{int(summary.earliest_closed_epoch)}..{int(summary.latest_closed_epoch)}] "
                f"spot_klines_synced={int(summary.spot_klines_synced)} "
                f"btc_klines_synced={int(summary.btc_klines_synced)}"
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
    min_bet_amount_bnb = float(contract.min_bet_amount()) / float(BNB_WEI)
    save_contract_constants(
        constants=ContractConstants(
            min_bet_amount_bnb=min_bet_amount_bnb,
            treasury_fee_fraction=treasury_fee_fraction,
        )
    )

    momentum_gate = None
    if cfg.momentum_gate.enabled:
        okx_client = OkxClient(timeout_seconds=10.0)
        momentum_gate = MomentumGate(config=cfg.momentum_gate, okx_client=okx_client)

    # Pool event watcher: subscribes to confirmed BetBull/BetBear events
    # via public WSS for accurate pool tracking (no signup required).
    pool_watcher = PoolEventWatcher()
    pool_watcher.start()

    runtime_cfg = RuntimeConfig(
        round_store=None,
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
        momentum_gate=momentum_gate,
        pool_watcher=pool_watcher,
    )

    try:
        run_live_loop(runtime_cfg)
    finally:
        pool_watcher.stop()
