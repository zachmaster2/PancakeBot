"""Filesystem paths for ABI, closed rounds, spot price klines, and dry/live/backtest outputs."""

# -- Shared --
ABI_JSON_PATH = "abi/prediction_v2_abi.json"
CLOSED_ROUNDS_PATH = "var/closed_rounds.jsonl"
CONTRACT_CONSTANTS_PATH = "var/contract_constants.json"

# Spot price klines (synced by --sync, consumed by --backtest).
BNB_SPOT_PRICES_PATH = "var/bnb_spot_prices.jsonl"
BTC_SPOT_PRICES_PATH = "var/btc_spot_prices.jsonl"
ETH_SPOT_PRICES_PATH = "var/eth_spot_prices.jsonl"
SOL_SPOT_PRICES_PATH = "var/sol_spot_prices.jsonl"

# -- Dry mode --
DRY_BANKROLL_STATE_PATH = "var/dry/bankroll.json"
DRY_BANKROLL_HISTORY_PATH = "var/dry/bankroll_history.jsonl"
DRY_PENDING_BETS_PATH = "var/dry/pending_bets.jsonl"
DRY_CYCLE_AUDIT_PATH = "var/dry/cycle_audit.csv"
DRY_TRADES_PATH = "var/dry/trades.csv"
DRY_SETTLED_EPOCHS_PATH = "var/dry/settled_epochs.txt"
DRY_ARCHIVE_ROOT = "var/dry/archive"

# -- Live mode --
LIVE_BANKROLL_HISTORY_PATH = "var/live/bankroll_history.jsonl"
LIVE_CLAIM_CURSOR_PATH = "var/live/claim_cursor.txt"
LIVE_CYCLE_AUDIT_PATH = "var/live/cycle_audit.csv"
LIVE_TRADES_PATH = "var/live/trades.csv"

# -- Backtest mode --
BACKTEST_TRADES_PATH = "var/backtest/trades.csv"
BACKTEST_SUMMARY_PATH = "var/backtest/summary.json"
BACKTEST_EQUITY_PATH = "var/backtest/equity_curves.png"
