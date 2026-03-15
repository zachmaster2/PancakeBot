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
from pancakebot.infra.binance_us_client import BinanceUsClient
from pancakebot.infra.klines_store import KlinesStore
from pancakebot.infra.closed_rounds_store import ClosedRoundsStore
from pancakebot.infra.feature_cache_store import FeatureCacheStore
from pancakebot.infra.market_data_db import MarketDataDb, SqliteKlinesStore
from pancakebot.infra.projection_cache_store import ProjectionCacheStore
from pancakebot.infra.run_registry_store import RunRegistryStore
from pancakebot.infra.graph_client import GraphClient
from pancakebot.infra.rpc_pool import choose_rpc_url
from pancakebot.infra.onchain.web3_contract_config import Web3ContractConfig
from pancakebot.infra.onchain.web3_prediction_contract import Web3PredictionContract
from pancakebot.runtime.contract_constants_cache import ContractConstants, load_contract_constants, save_contract_constants
from pancakebot.runtime.runtime_loop import RuntimeConfig, run_live_loop
from pancakebot.core.determinism import set_global_determinism


_BINANCE_US_SYMBOL = "BNBUSDT"


def run_from_config(*, config_path: str, dry: bool, backtest: bool) -> None:
    cfg = load_app_config(config_path)
    set_global_determinism(seed=int(cfg.random_seed))

    round_store = ClosedRoundsStore(cfg.closed_rounds_path)
    klines_store = KlinesStore(cfg.klines_path)
    binance_us_client = BinanceUsClient(timeout_seconds=10.0)

    if backtest:
        constants = load_contract_constants()
        market_data_store = MarketDataDb(cfg.market_data_db_path)
        feature_cache_store = FeatureCacheStore(cfg.feature_cache_path)
        projection_cache_store = ProjectionCacheStore(cfg.projection_cache_db_path)
        run_registry_store = RunRegistryStore(cfg.run_registry_db_path)
        try:
            market_data_store.ensure_sources_synced(
                rounds_jsonl_path=str(cfg.closed_rounds_path),
                klines_jsonl_path=str(cfg.klines_path),
            )
            runtime_cfg = RuntimeConfig(
                graph_client=None,
                round_store=round_store,
                klines_store=SqliteKlinesStore(market_data_db=market_data_store),
                binance_us_client=binance_us_client,
                binance_us_symbol=_BINANCE_US_SYMBOL,
                contract=None,
                wallet_address="",
                cutoff_seconds=cfg.cutoff_seconds,
                use_onchain_event_bets=False,
                event_lookback_blocks=cfg.event_lookback_blocks,
                latency_log_path=cfg.latency_log_path,
                wait_for_bet_receipt=False,
                bet_receipt_timeout_seconds=cfg.bet_receipt_timeout_seconds,
                strategy_cfg=cfg.strategy,
                dry=dry,
                feature_cache_store=feature_cache_store,
                market_data_store=market_data_store,
                projection_cache_store=projection_cache_store,
                run_registry_store=run_registry_store,
                backtest_state_cache_dir=cfg.backtest_state_cache_dir,
                runtime_state_paths=cfg.runtime_state_paths,
                min_bet_amount_bnb=float(constants.min_bet_amount_bnb),
                treasury_fee_fraction=float(constants.treasury_fee_fraction),
                buffer_seconds=int(constants.buffer_seconds),
            )
            run_backtest(runtime_cfg=runtime_cfg, backtest_cfg=cfg.backtest, out_dir=Path("var"))
        finally:
            try:
                feature_cache_store.close()
            except Exception:
                pass
            try:
                projection_cache_store.close()
            except Exception:
                pass
            try:
                run_registry_store.close()
            except Exception:
                pass
            try:
                market_data_store.close()
            except Exception:
                pass
        return

    load_env()
    graph_api_key = require_env("THE_GRAPH_API_KEY")
    private_key = require_env("BSC_WALLET_PRIVATE_KEY")
    rpc_url = choose_rpc_url(
        RPC_URLS,
        expected_chain_id=int(EXPECTED_CHAIN_ID),
        timeout_seconds=int(RPC_TIMEOUT_SECONDS),
    )

    graph = GraphClient(endpoint=PREDICTION_V2_GRAPH_ENDPOINT, api_key=graph_api_key)

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

    runtime_cfg = RuntimeConfig(
        graph_client=graph,
        round_store=round_store,
        klines_store=klines_store,
        binance_us_client=binance_us_client,
        binance_us_symbol=_BINANCE_US_SYMBOL,
        contract=contract,
        wallet_address=contract.wallet_address,
        cutoff_seconds=cfg.cutoff_seconds,
        use_onchain_event_bets=cfg.use_onchain_event_bets,
        event_lookback_blocks=cfg.event_lookback_blocks,
        latency_log_path=cfg.latency_log_path,
        wait_for_bet_receipt=cfg.wait_for_bet_receipt,
        bet_receipt_timeout_seconds=cfg.bet_receipt_timeout_seconds,
        strategy_cfg=cfg.strategy,
        dry=dry,
        feature_cache_store=None,
        market_data_store=None,
        projection_cache_store=None,
        run_registry_store=None,
        backtest_state_cache_dir=cfg.backtest_state_cache_dir,
        runtime_state_paths=cfg.runtime_state_paths,
        min_bet_amount_bnb=float(min_bet_amount_bnb),
        treasury_fee_fraction=treasury_fee_fraction,
        buffer_seconds=buffer_seconds,
    )

    run_live_loop(runtime_cfg)
