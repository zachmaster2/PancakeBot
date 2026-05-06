"""Entrypoint dispatcher for backtest, sync, dry, and live run modes."""
from __future__ import annotations

from pathlib import Path

from pancakebot.backtest.runner import run_backtest
from pancakebot.config import BacktestConfig, load_app_config, load_env, require_env
from pancakebot.constants import (
    BNB_WEI,
    EXPECTED_CHAIN_ID,
    PREDICTION_V2_GRAPH_ENDPOINT,
    RPC_TIMEOUT_SECONDS,
    RPC_URLS,
)
from pancakebot.market_data.contract_constants import load_contract_constants
from pancakebot.market_data.okx_client import OkxClient
from pancakebot.strategy.momentum_gate import MomentumGate, MomentumGateConfig
from pancakebot.market_data.round_store import ClosedRoundsStore
from pancakebot.market_data.graph_client import GraphClient
from pancakebot.chain.rpc_pool import choose_rpc_url
from pancakebot.chain.contract_config import Web3ContractConfig
from pancakebot.chain.prediction_contract import Web3PredictionContract
from pancakebot.market_data.contract_constants import ContractConstants, save_contract_constants
from pancakebot.runtime.config import RuntimeConfig
from pancakebot.runtime import engine
from pancakebot.market_data.sync import sync_runtime_market_data
from pancakebot.util import InvariantError
from pancakebot.log import info
from pancakebot.chain.pool_watcher import PoolEventWatcher
from pancakebot import paths


def run_from_config(
    *,
    config_path: str,
    dry: bool,
    backtest: bool,
    sync: bool,
    live: bool = False,
    fresh: bool = False,
    no_archive: bool = False,
    use_extended_data: bool = False,
) -> None:
    cfg = load_app_config(config_path)

    selected_modes = int(dry) + int(backtest) + int(sync) + int(live)
    if selected_modes > 1:
        raise InvariantError("run_modes_mutually_exclusive")

    # Inline MomentumGateConfig -- symbols are hardcoded project constants;
    # mtf_lookbacks and mtf_threshold come from [strategy.gate] config.
    # cutoff_seconds threads ``[runtime] kline_cutoff_seconds`` to the gate
    # so the gate is the single source of truth for the data window.
    momentum_gate_cfg = MomentumGateConfig(
        enabled=True,
        bnb_symbol="BNB-USDT",
        btc_symbol="BTC-USDT",
        eth_symbol="ETH-USDT",
        sol_symbol="SOL-USDT",
        cutoff_seconds=cfg.kline_cutoff_seconds,
        mtf_lookbacks=cfg.strategy.gate.mtf_lookbacks,
        mtf_threshold=cfg.strategy.gate.mtf_threshold,
        max_consecutive_fetch_failures=cfg.max_consecutive_fetch_failures,
    )

    if backtest:
        cc = load_contract_constants()
        round_store = ClosedRoundsStore(paths.CLOSED_ROUNDS_PATH)
        # noinspection PyTypeChecker
        backtest_cfg: BacktestConfig = cfg.backtest
        # Receipt timeouts derived from chain-loaded buffer_seconds +
        # _CLAIM_CHECK_PADDING_SECONDS (≈35s on canonical chain constants).
        # Both bet and claim TX receipts share this timeout sizing.
        from pancakebot.runtime.engine import _CLAIM_CHECK_PADDING_SECONDS
        _bt_receipt_timeout = (
            int(cc.buffer_seconds) + int(_CLAIM_CHECK_PADDING_SECONDS)
        )
        # noinspection PyTypeChecker
        runtime_cfg = RuntimeConfig(
            round_store=round_store,
            contract=None,
            wallet_address="",
            cutoff_seconds=cfg.kline_cutoff_seconds,
            ntp_sync_wakeup_offset_ms=cfg.ntp_sync_wakeup_offset_ms,
            bankroll_wakeup_offset_ms=cfg.bankroll_wakeup_offset_ms,
            critical_path_wakeup_offset_ms=cfg.critical_path_wakeup_offset_ms,
            bet_submit_deadline_offset_ms=cfg.bet_submit_deadline_offset_ms,
            bet_tx_receipt_timeout_seconds=_bt_receipt_timeout,
            claim_tx_receipt_timeout_seconds=_bt_receipt_timeout,
            kline_publish_tier=cfg.kline_publish_tier,
            max_consecutive_fetch_failures=cfg.max_consecutive_fetch_failures,
            pool_cutoff_seconds=cfg.pool_cutoff_seconds,
            dry_initial_bankroll_bnb=cfg.dry_initial_bankroll_bnb,
            momentum_gate_config=momentum_gate_cfg,
            momentum_gate=None,
            dry=dry,
            live_min_bet_only=False,
            dry_fresh_start=False,
            dry_no_archive=False,
            min_bet_amount_bnb=cc.min_bet_amount_bnb,
            treasury_fee_fraction=cc.treasury_fee_fraction,
            interval_seconds=cc.interval_seconds,
            buffer_seconds=cc.buffer_seconds,
            strategy=cfg.strategy,
        )
        run_backtest(
            runtime_cfg=runtime_cfg,
            backtest_cfg=backtest_cfg,
            out_dir=Path("var/backtest"),
            use_extended_data=use_extended_data,
        )
        return

    if sync:
        load_env()
        graph_api_key = require_env("THE_GRAPH_API_KEY")
        graph = GraphClient(endpoint=PREDICTION_V2_GRAPH_ENDPOINT, api_key=graph_api_key)
        round_store = ClosedRoundsStore(paths.CLOSED_ROUNDS_PATH)
        okx_client = OkxClient(timeout_seconds=10.0)
        # Sync mode fetches all 4 symbols (BNB/BTC/ETH/SOL) in parallel even
        # though the live/dry hot-path gate only consumes 3 (BTC/ETH/SOL).
        # BNB klines are kept synced to disk so future strategies that want
        # BNB closes have the historical data already on hand. Pass
        # connections=4 explicitly so all 4 sockets are pre-warmed; the
        # default (3) tracks the live/dry gate's symbol count.
        okx_client.warmup(connections=4)
        summary = sync_runtime_market_data(
            cfg=cfg,
            graph=graph,
            round_store=round_store,
            okx_client=okx_client,
        )
        info(
            "CORE",
            "SYNC",
            "DONE",
            msg=(
                f"closed_rounds={int(summary.stored_closed_round_count)} "
                f"epochs=[{int(summary.earliest_closed_epoch)}..{int(summary.latest_closed_epoch)}] "
                f"bnb_synced={int(summary.bnb_klines_synced)} "
                f"btc_synced={int(summary.btc_klines_synced)} "
                f"eth_synced={int(summary.eth_klines_synced)} "
                f"sol_synced={int(summary.sol_klines_synced)}"
            ),
        )
        return

    # Dry or live mode -- both need RPC, live also needs private key.
    load_env()
    private_key = require_env("BSC_WALLET_PRIVATE_KEY") if live else ""
    rpc_url = choose_rpc_url(
        RPC_URLS,
        expected_chain_id=int(EXPECTED_CHAIN_ID),
        timeout_seconds=int(RPC_TIMEOUT_SECONDS),
    )

    contract_cfg = Web3ContractConfig(
        rpc_url=rpc_url,
        rpc_urls=tuple(RPC_URLS),
        abi_json_path=paths.ABI_JSON_PATH,
        private_key=private_key,
    )
    contract = Web3PredictionContract(contract_cfg)

    treasury_fee_fraction = contract.treasury_fee_rate()
    min_bet_amount_bnb = float(contract.min_bet_amount()) / float(BNB_WEI)
    interval_seconds = contract.interval_seconds()
    buffer_seconds = contract.buffer_seconds()
    save_contract_constants(
        constants=ContractConstants(
            min_bet_amount_bnb=min_bet_amount_bnb,
            treasury_fee_fraction=treasury_fee_fraction,
            interval_seconds=interval_seconds,
            buffer_seconds=buffer_seconds,
        )
    )

    okx_client = OkxClient(timeout_seconds=10.0)
    okx_client.warmup()

    # Per-round REST kline fetch path: the gate fires 3 parallel
    # ``/history-candles`` GETs each round (BTC/ETH/SOL) anchored to
    # ``lock_at_ms``. Triggered inside the critical_path wake
    # (configured via ``RuntimeConfig.critical_path_wakeup_offset_ms``)
    # after the in-memory pool snapshot. BNB fetch is currently
    # disabled (the strategy doesn't consume BNB closes for signal
    # computation, and chain-supplied lock_price covers display/USD
    # conversion); see ``MomentumGate._SYMBOLS_FETCHED`` for re-enable
    # steps if a future strategy needs BNB klines.
    momentum_gate = MomentumGate(
        config=momentum_gate_cfg,
        okx_client=okx_client,
    )
    import atexit as _atexit
    _atexit.register(momentum_gate.shutdown)

    # Pool event watcher: subscribes to confirmed BetBull/BetBear events
    # via public WSS for accurate pool tracking (no signup required).
    pool_watcher = PoolEventWatcher(interval_seconds=interval_seconds)
    pool_watcher.start()

    # Receipt timeouts derived from chain-loaded buffer_seconds +
    # _CLAIM_CHECK_PADDING_SECONDS (≈35s on canonical chain constants).
    # Both bet and claim TX receipts share this timeout sizing.
    from pancakebot.runtime.engine import _CLAIM_CHECK_PADDING_SECONDS
    _runtime_receipt_timeout = (
        int(buffer_seconds) + int(_CLAIM_CHECK_PADDING_SECONDS)
    )

    runtime_cfg = RuntimeConfig(
        round_store=None,
        contract=contract,
        wallet_address=contract.wallet_address,
        cutoff_seconds=cfg.kline_cutoff_seconds,
        ntp_sync_wakeup_offset_ms=cfg.ntp_sync_wakeup_offset_ms,
        bankroll_wakeup_offset_ms=cfg.bankroll_wakeup_offset_ms,
        critical_path_wakeup_offset_ms=cfg.critical_path_wakeup_offset_ms,
        bet_submit_deadline_offset_ms=cfg.bet_submit_deadline_offset_ms,
        bet_tx_receipt_timeout_seconds=_runtime_receipt_timeout,
        claim_tx_receipt_timeout_seconds=_runtime_receipt_timeout,
        kline_publish_tier=cfg.kline_publish_tier,
        max_consecutive_fetch_failures=cfg.max_consecutive_fetch_failures,
        pool_cutoff_seconds=cfg.pool_cutoff_seconds,
        dry_initial_bankroll_bnb=cfg.dry_initial_bankroll_bnb,
        momentum_gate_config=momentum_gate_cfg,
        dry=dry,
        live_min_bet_only=cfg.live_min_bet_only,
        dry_fresh_start=fresh,
        dry_no_archive=no_archive,
        min_bet_amount_bnb=float(min_bet_amount_bnb),
        treasury_fee_fraction=treasury_fee_fraction,
        interval_seconds=interval_seconds,
        buffer_seconds=buffer_seconds,
        strategy=cfg.strategy,
        momentum_gate=momentum_gate,
        pool_watcher=pool_watcher,
    )

    try:
        engine.run_realtime_loop(runtime_cfg)
    finally:
        pool_watcher.stop()
