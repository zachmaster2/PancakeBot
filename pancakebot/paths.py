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

# Extended dataset (older epochs, lenient/with status; written by
# research/backfill_okx_extended.py, consumed by --backtest --use-extended-data).
# Files at the same paths under var/extended/ have epochs strictly older than
# the canonical floor (epoch 437562); Phase B's lenient fetcher tolerates
# partial/missing data and tags each record with a ``data_status`` field.
EXTENDED_CLOSED_ROUNDS_PATH = "var/extended/closed_rounds.jsonl"
EXTENDED_BNB_SPOT_PRICES_PATH = "var/extended/bnb_spot_prices.jsonl"
EXTENDED_BTC_SPOT_PRICES_PATH = "var/extended/btc_spot_prices.jsonl"
EXTENDED_ETH_SPOT_PRICES_PATH = "var/extended/eth_spot_prices.jsonl"
EXTENDED_SOL_SPOT_PRICES_PATH = "var/extended/sol_spot_prices.jsonl"

# -- Dry mode --
DRY_BANKROLL_STATE_PATH = "var/dry/bankroll.json"
DRY_BANKROLL_HISTORY_PATH = "var/dry/bankroll_history.jsonl"
DRY_PENDING_BETS_PATH = "var/dry/pending_bets.jsonl"
DRY_CYCLE_AUDIT_PATH = "var/dry/cycle_audit.csv"
DRY_TRADES_PATH = "var/dry/trades.csv"
DRY_SETTLED_EPOCHS_PATH = "var/dry/settled_epochs.txt"
DRY_ARCHIVE_ROOT = "var/dry/archive"

# Process-health artifacts (crash.json + PID file).
# Supervisor uses Popen.poll() for liveness.
DRY_CRASH_PATH = "var/dry/crash.json"
DRY_PID_PATH = "var/dry/bot.pid"

# RotatingFileHandler sink (Bundle 5 2026-05-14). The runtime mirrors
# every ``pancakebot.log`` line into this file via a Python ``logging``
# handler. Stdout writer is preserved. 25MB rotation × 7 backups.
DRY_RUNTIME_LOG_PATH = "var/dry/runtime.log"

# -- Live mode --
LIVE_BANKROLL_HISTORY_PATH = "var/live/bankroll_history.jsonl"
LIVE_CLAIM_CURSOR_PATH = "var/live/claim_cursor.txt"
LIVE_CYCLE_AUDIT_PATH = "var/live/cycle_audit.csv"
LIVE_TRADES_PATH = "var/live/trades.csv"

# Process-health artifacts (live mirror of the dry pair above).
LIVE_CRASH_PATH = "var/live/crash.json"
LIVE_PID_PATH = "var/live/bot.pid"

# RotatingFileHandler sink (Bundle 5 2026-05-14). Live mirror of the dry
# runtime log path; see DRY_RUNTIME_LOG_PATH for rationale.
LIVE_RUNTIME_LOG_PATH = "var/live/runtime.log"

# -- Backtest mode --
BACKTEST_TRADES_PATH = "var/backtest/trades.csv"
BACKTEST_SUMMARY_PATH = "var/backtest/summary.json"
BACKTEST_EQUITY_PATH = "var/backtest/equity_curves.png"
