"""Chain, RPC, Graph, and gas constants for BNB Chain and PancakeSwap Prediction V2."""

from __future__ import annotations

# --- Chain / contract (BNB Chain mainnet) ---

EXPECTED_CHAIN_ID = 56

PREDICTION_V2_CONTRACT_ADDRESS = "0x18B2A687610328590Bc8F2e5fEdDe3b582A49cdA"

# The contract's treasury fee is expressed in basis points (bps).
TREASURY_FEE_DIVISOR = 10_000

# Protocol constants (treasury_fee, min_bet, interval_seconds, buffer_seconds)
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

# Worst-case gas-price ceiling shared across all execution modes:
#   - backtest / dry: assumed-paid for EV/PnL cost accounting (charge
#     the worst case so simulated bankrolls aren't optimistic)
#   - live: cap on the gas price paid for bet/claim TXs. The bot's
#     sanity check raises if eth.gas_price exceeds this — the operator
#     must lift the cap before resuming.
# 1 Gwei is comfortably above today's BSC mainnet floor (~0.05 Gwei)
# yet bounds operator cost per TX.
MAX_GAS_PRICE_WEI = 1_000_000_000

# Gas limits posted on bet/claim TXs (also used to compute the worst-case
# BNB cost below). Sized from on-chain receipt observation; the headroom
# above typical usage absorbs occasional contract-internal branches.
GAS_LIMIT_BET = 200_000
GAS_LIMIT_CLAIM = 250_000

# Worst-case BNB cost per bet/claim TX (= limit × MAX_GAS_PRICE_WEI).
# Used by backtest+dry to debit simulated bankrolls. Live mode burns
# the real on-chain cost (typically less) and re-reads bankroll from
# chain at the next bankroll-wake.
MAX_GAS_COST_BET_BNB = float(GAS_LIMIT_BET) * float(MAX_GAS_PRICE_WEI) / float(BNB_WEI)
MAX_GAS_COST_CLAIM_BNB = float(GAS_LIMIT_CLAIM) * float(MAX_GAS_PRICE_WEI) / float(BNB_WEI)

# --- Runtime retry ---

RETRY_BACKOFF_SECONDS = [2, 4, 8, 16, 32, 58]  # locked
