from __future__ import annotations

import os

from dotenv import load_dotenv

from pancakebot.core.errors import InvariantError


def load_env() -> None:
    """Load .env into process environment.

    Per user approval:
      - use python-dotenv
      - simply call load_dotenv()
      - both THE_GRAPH_API_KEY and BSC_WALLET_PRIVATE_KEY are required (all modes)
    """
    load_dotenv()


def require_env(name: str) -> str:
    v = os.getenv(name)
    if v is None or not str(v).strip():
        raise InvariantError(f"missing_env_var: {name}")
    return str(v).strip()
