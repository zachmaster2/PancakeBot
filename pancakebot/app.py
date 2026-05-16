"""Entrypoint dispatcher for backtest, sync, dry, and live run modes."""
from __future__ import annotations

from pathlib import Path

from pancakebot.backtest.runner import run_backtest
from pancakebot.config import BacktestConfig, load_app_config, load_env, require_env
from pancakebot.constants import (
    BNB_WEI,
    EXPECTED_CHAIN_ID,
    PREDICTION_V2_GRAPH_ENDPOINT,
    WRITE_PATH_RPC_TIMEOUT_SECONDS,
    WRITE_PATH_RPC_URLS,
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
from pancakebot.log import configure_file_logging, info
from pancakebot.chain.rpc_poller import RpcPoller, READ_PATH_HEDGED_ENDPOINTS
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
    # mtf_lookbacks and mtf_min_return_threshold come from [strategy.gate] config.
    # kline_cutoff_seconds threads ``[runtime] kline_cutoff_seconds`` to the gate
    # so the gate is the single source of truth for the data window.
    momentum_gate_cfg = MomentumGateConfig(
        enabled=True,
        bnb_symbol="BNB-USDT",
        btc_symbol="BTC-USDT",
        eth_symbol="ETH-USDT",
        sol_symbol="SOL-USDT",
        kline_cutoff_seconds=cfg.kline_cutoff_seconds,
        mtf_lookbacks=cfg.strategy.gate.mtf_lookbacks,
        mtf_min_return_threshold=cfg.strategy.gate.mtf_min_return_threshold,
        max_consecutive_kline_fetch_failures=cfg.max_consecutive_kline_fetch_failures,
    )

    if backtest:
        cc = load_contract_constants()
        round_store = ClosedRoundsStore(paths.CLOSED_ROUNDS_PATH)
        # noinspection PyTypeChecker
        backtest_cfg: BacktestConfig = cfg.backtest
        # Receipt timeouts derived from chain-loaded round_close_buffer_seconds +
        # _CLAIM_RECEIPT_TIMEOUT_PADDING_SECONDS (≈35s on canonical chain constants).
        # Both bet and claim TX receipts share this timeout sizing.
        from pancakebot.runtime.engine import _CLAIM_RECEIPT_TIMEOUT_PADDING_SECONDS
        _bt_receipt_timeout = (
            int(cc.round_close_buffer_seconds) + int(_CLAIM_RECEIPT_TIMEOUT_PADDING_SECONDS)
        )
        # noinspection PyTypeChecker
        runtime_cfg = RuntimeConfig(
            round_store=round_store,
            contract=None,
            wallet_address="",
            kline_cutoff_seconds=cfg.kline_cutoff_seconds,
            bankroll_wakeup_offset_before_lock_ms=cfg.bankroll_wakeup_offset_before_lock_ms,
            critical_path_wakeup_offset_before_lock_ms=cfg.critical_path_wakeup_offset_before_lock_ms,
            bet_submit_deadline_offset_before_lock_ms=cfg.bet_submit_deadline_offset_before_lock_ms,
            bet_tx_receipt_timeout_seconds=_bt_receipt_timeout,
            claim_tx_receipt_timeout_seconds=_bt_receipt_timeout,
            kline_publish_tier=cfg.kline_publish_tier,
            max_consecutive_kline_fetch_failures=cfg.max_consecutive_kline_fetch_failures,
            pool_cutoff_seconds=cfg.pool_cutoff_seconds,
            dry_initial_bankroll_bnb=cfg.dry_initial_bankroll_bnb,
            momentum_gate_config=momentum_gate_cfg,
            momentum_gate=None,
            dry=dry,
            live_clamp_bet_to_contract_minimum=False,
            dry_fresh_start=False,
            dry_no_archive=False,
            min_bet_amount_bnb=cc.min_bet_amount_bnb,
            treasury_fee_fraction=cc.treasury_fee_fraction,
            round_interval_seconds=cc.round_interval_seconds,
            round_close_buffer_seconds=cc.round_close_buffer_seconds,
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

    # Bundle 5 2026-05-14: persist every structured log line to a
    # rotating file under ``var/{mode}/runtime.log``. The stdout writer
    # (consumed by the Windows pythonw redirect into the supervisor's
    # stdout capture) is preserved; the file sink is purely additive
    # and survives a crash that takes the stdout consumer down with it.
    _runtime_log_path = (
        paths.DRY_RUNTIME_LOG_PATH if dry else paths.LIVE_RUNTIME_LOG_PATH
    )
    configure_file_logging(_runtime_log_path)
    rpc_url = choose_rpc_url(
        WRITE_PATH_RPC_URLS,
        expected_chain_id=int(EXPECTED_CHAIN_ID),
        timeout_seconds=int(WRITE_PATH_RPC_TIMEOUT_SECONDS),
    )

    contract_cfg = Web3ContractConfig(
        rpc_url=rpc_url,
        rpc_urls=tuple(WRITE_PATH_RPC_URLS),
        abi_json_path=paths.ABI_JSON_PATH,
        private_key=private_key,
    )
    contract = Web3PredictionContract(contract_cfg)

    treasury_fee_fraction = contract.treasury_fee_rate()
    min_bet_amount_bnb = float(contract.min_bet_amount()) / float(BNB_WEI)
    round_interval_seconds = contract.round_interval_seconds()
    round_close_buffer_seconds = contract.round_close_buffer_seconds()
    save_contract_constants(
        constants=ContractConstants(
            min_bet_amount_bnb=min_bet_amount_bnb,
            treasury_fee_fraction=treasury_fee_fraction,
            round_interval_seconds=round_interval_seconds,
            round_close_buffer_seconds=round_close_buffer_seconds,
        )
    )

    okx_client = OkxClient(timeout_seconds=10.0)
    okx_client.warmup()

    # Per-round REST kline fetch path: the gate fires 3 parallel
    # ``/history-candles`` GETs each round (BTC/ETH/SOL) anchored to
    # ``lock_at_ms``. Triggered inside the critical_path wake
    # (configured via ``RuntimeConfig.critical_path_wakeup_offset_before_lock_ms``)
    # after the in-memory pool snapshot. BNB fetch is currently
    # disabled (the strategy doesn't consume BNB closes for signal
    # computation, and chain-supplied lock_price covers display/USD
    # conversion); see ``MomentumGate._OKX_SYMBOLS_FETCHED`` for re-enable
    # steps if a future strategy needs BNB klines.
    momentum_gate = MomentumGate(
        config=momentum_gate_cfg,
        okx_client=okx_client,
    )
    import atexit as _atexit
    _atexit.register(momentum_gate.shutdown)

    # RPC poller: deterministic poll schedule (cold-start backfill +
    # periodic + ramp + final) over batched eth_getBlockReceipts.
    # Replaces the WSS-subscription pool watcher (Era 11, 2026-05-07);
    # see var/design/rpc_polling_architecture_2026_05_07.md.
    #
    # Every JSON-RPC call fires in parallel to every endpoint in
    # READ_PATH_HEDGED_ENDPOINTS via a shared urllib3.PoolManager (persistent
    # HTTP/1.1 connections); first 200 response wins. No selection logic
    # — if an endpoint misbehaves, remove it from the constant. See
    # var/incident_reports/2026_05_11_parallel_request_transport_bottleneck.md.
    rpc_poller = RpcPoller(
        round_interval_seconds=round_interval_seconds,
        endpoint_pool=READ_PATH_HEDGED_ENDPOINTS,
        ramp_poll_1_wakeup_offset_before_lock_ms=cfg.ramp_poll_1_wakeup_offset_before_lock_ms,
    )
    rpc_poller.start()

    # Receipt timeouts derived from chain-loaded round_close_buffer_seconds +
    # _CLAIM_RECEIPT_TIMEOUT_PADDING_SECONDS (≈35s on canonical chain constants).
    # Both bet and claim TX receipts share this timeout sizing.
    from pancakebot.runtime.engine import _CLAIM_RECEIPT_TIMEOUT_PADDING_SECONDS
    _runtime_receipt_timeout = (
        int(round_close_buffer_seconds) + int(_CLAIM_RECEIPT_TIMEOUT_PADDING_SECONDS)
    )

    runtime_cfg = RuntimeConfig(
        round_store=None,
        contract=contract,
        wallet_address=contract.wallet_address,
        kline_cutoff_seconds=cfg.kline_cutoff_seconds,
        ramp_poll_1_wakeup_offset_before_lock_ms=cfg.ramp_poll_1_wakeup_offset_before_lock_ms,
        ramp_poll_2_wakeup_offset_before_lock_ms=cfg.ramp_poll_2_wakeup_offset_before_lock_ms,
        final_rpc_poll_wakeup_offset_before_lock_ms=cfg.final_rpc_poll_wakeup_offset_before_lock_ms,
        bankroll_wakeup_offset_before_lock_ms=cfg.bankroll_wakeup_offset_before_lock_ms,
        critical_path_wakeup_offset_before_lock_ms=cfg.critical_path_wakeup_offset_before_lock_ms,
        bet_submit_deadline_offset_before_lock_ms=cfg.bet_submit_deadline_offset_before_lock_ms,
        bet_tx_receipt_timeout_seconds=_runtime_receipt_timeout,
        claim_tx_receipt_timeout_seconds=_runtime_receipt_timeout,
        kline_publish_tier=cfg.kline_publish_tier,
        max_consecutive_kline_fetch_failures=cfg.max_consecutive_kline_fetch_failures,
        pool_cutoff_seconds=cfg.pool_cutoff_seconds,
        dry_initial_bankroll_bnb=cfg.dry_initial_bankroll_bnb,
        momentum_gate_config=momentum_gate_cfg,
        dry=dry,
        live_clamp_bet_to_contract_minimum=cfg.live_clamp_bet_to_contract_minimum,
        dry_fresh_start=fresh,
        dry_no_archive=no_archive,
        min_bet_amount_bnb=float(min_bet_amount_bnb),
        treasury_fee_fraction=treasury_fee_fraction,
        round_interval_seconds=round_interval_seconds,
        round_close_buffer_seconds=round_close_buffer_seconds,
        strategy=cfg.strategy,
        momentum_gate=momentum_gate,
        rpc_poller=rpc_poller,
    )

    try:
        engine.run_realtime_loop(runtime_cfg)
    finally:
        rpc_poller.stop()
