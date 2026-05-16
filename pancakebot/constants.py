"""Chain, RPC, Graph, and gas constants for BNB Chain and PancakeSwap Prediction V2."""

from __future__ import annotations

# --- Chain / contract (BNB Chain mainnet) ---

EXPECTED_CHAIN_ID = 56

PREDICTION_V2_CONTRACT_ADDRESS = "0x18B2A687610328590Bc8F2e5fEdDe3b582A49cdA"

# The contract's treasury fee is expressed in basis points (bps).
TREASURY_FEE_DIVISOR = 10_000

# Protocol constants (treasury_fee, min_bet, round_interval_seconds, round_close_buffer_seconds)
# are synced from chain by --sync and cached in var/contract_constants.json.
# See pancakebot/market_data/contract_constants.py.

# --- RPC (hardcoded list; failover is handled by RPC chooser) ---

WRITE_PATH_RPC_URLS: list[str] = [
    "https://bsc-dataseed1.defibit.io",
    "https://bsc-dataseed2.defibit.io",
    "https://bsc-dataseed3.defibit.io",
]

WRITE_PATH_RPC_TIMEOUT_SECONDS = 20

# --- The Graph (gateway + subgraph id) ---

# Pancake Prediction V2 subgraph id (locked).
PREDICTION_V2_SUBGRAPH_ID = "4kRuZVKCR9dsG2ePXhLSiKw5oaw3YMJo4nAwxZbUaqVY"

# Gateway base (locked).
THE_GRAPH_GATEWAY_BASE = "https://gateway.thegraph.com/api"

# Endpoint used for GraphQL POSTs.
# Auth is provided via: Authorization: Bearer {THE_GRAPH_API_KEY}
PREDICTION_V2_GRAPH_ENDPOINT = f"{THE_GRAPH_GATEWAY_BASE}/subgraphs/id/{PREDICTION_V2_SUBGRAPH_ID}"

# --- Math constants ---

BNB_WEI = 1_000_000_000_000_000_000

# --- Gas foundation (v1.0 frozen) ---

# Deterministic gas *cost* accounting used for EV/backtest/dry bankroll.
# Runtime transaction submission still uses on-chain gas suggestions.
BACKTEST_GAS_PRICE_WEI = 1_000_000_000

# Deterministic gas limits (used for cost accounting; may also be used as tx gas limits).
BACKTEST_GAS_LIMIT_BET = 200_000
BACKTEST_GAS_LIMIT_CLAIM = 250_000

# Deterministic gas costs (BNB). These are costs, not limits.
BACKTEST_GAS_COST_BET_BNB = float(BACKTEST_GAS_LIMIT_BET) * float(BACKTEST_GAS_PRICE_WEI) / float(BNB_WEI)
BACKTEST_GAS_COST_CLAIM_BNB = float(BACKTEST_GAS_LIMIT_CLAIM) * float(BACKTEST_GAS_PRICE_WEI) / float(BNB_WEI)

# --- Runtime retry ---

RETRY_BACKOFF_SECONDS = [2, 4, 8, 16, 32, 58]  # locked
