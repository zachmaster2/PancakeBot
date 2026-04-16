"""BNB-to-USD conversion and human-readable bankroll/amount formatting helpers."""
from __future__ import annotations


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
