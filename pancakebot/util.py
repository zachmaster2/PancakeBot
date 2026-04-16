"""General utility helpers: exceptions, time, paths, money formatting."""
from __future__ import annotations

import time
from pathlib import Path


# ── Exceptions ────────────────────────────────────────────────────

class InvariantError(Exception):
    pass


class TransientRpcError(Exception):
    pass


class TransientGraphError(Exception):
    pass


# ── Time ──────────────────────────────────────────────────────────

def now_ts() -> int:
    return int(time.time())


# ── Paths ─────────────────────────────────────────────────────────

def ensure_parent_dir(path: str) -> None:
    p = Path(path)
    parent = p.parent
    if parent and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)


# ── Money ─────────────────────────────────────────────────────────

def usd_value(*, amount_bnb: float, bnbusd_price: float) -> float:
    return amount_bnb * bnbusd_price


def usd_suffix(*, amount_bnb: float, bnbusd_price: float) -> str:
    usd = usd_value(amount_bnb=amount_bnb, bnbusd_price=bnbusd_price)
    return f" (${usd:.2f} USD)"


def bankroll_suffix(*, bankroll_bnb: float, bnbusd_price: float) -> str:
    usd = usd_value(amount_bnb=bankroll_bnb, bnbusd_price=bnbusd_price)
    return f" (Bankroll is now {bankroll_bnb:.4f} BNB (${usd:.2f} USD))"


def format_bankroll(*, bankroll_bnb: float, bnbusd_price: float) -> str:
    usd = usd_value(amount_bnb=bankroll_bnb, bnbusd_price=bnbusd_price)
    return f"{bankroll_bnb:.4f} BNB (${usd:.2f} USD)"
