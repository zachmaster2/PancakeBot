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
        runtime_cfg = RuntimeConfig(
            graph_client=None,
            round_store=round_store,
            klines_store=klines_store,
            binance_us_client=binance_us_client,
            binance_us_symbol=_BINANCE_US_SYMBOL,
            contract=None,
            wallet_address="",
            cutoff_seconds=cfg.cutoff_seconds,
            train_size=cfg.train_size,
            retrain_interval=cfg.retrain_interval,
            calibrate_size=cfg.calibrate_size,
            recalibrate_interval=cfg.recalibrate_interval,
            recency_weight_floor=cfg.recency_weight_floor,
            recency_weight_power=cfg.recency_weight_power,
            policy_cfg=cfg.policy,
            dry=dry,
            treasury_fee_fraction=float(constants.treasury_fee_fraction),
            buffer_seconds=int(constants.buffer_seconds),
            min_bet_amount_bnb=float(constants.min_bet_amount_bnb),
            price_alpha=cfg.price_alpha,
            pool_alpha_total=cfg.pool_alpha_total,
            pool_alpha_ratio=cfg.pool_alpha_ratio,
            random_seed=cfg.random_seed,
        )
        run_backtest(runtime_cfg=runtime_cfg, backtest_cfg=cfg.backtest, out_dir=Path("var"))
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
        train_size=cfg.train_size,
        retrain_interval=cfg.retrain_interval,
        calibrate_size=cfg.calibrate_size,
        recalibrate_interval=cfg.recalibrate_interval,
        recency_weight_floor=cfg.recency_weight_floor,
        recency_weight_power=cfg.recency_weight_power,
        policy_cfg=cfg.policy,
        dry=dry,
        treasury_fee_fraction=treasury_fee_fraction,
        buffer_seconds=buffer_seconds,
        min_bet_amount_bnb=float(min_bet_amount_bnb),
        price_alpha=cfg.price_alpha,
        pool_alpha_total=cfg.pool_alpha_total,
        pool_alpha_ratio=cfg.pool_alpha_ratio,
        random_seed=cfg.random_seed,
    )

    run_live_loop(runtime_cfg)
