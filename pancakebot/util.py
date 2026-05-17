"""General utility helpers: exceptions, paths, money formatting."""
from __future__ import annotations

from pathlib import Path


# -- Exceptions ----------------------------------------------------

class InvariantError(Exception):
    pass


class TransientRpcError(Exception):
    pass


class TransientGraphError(Exception):
    pass


class TransientOkxError(Exception):
    """OKX fetch failed after the retry budget exhausted (or single-attempt
    on the live decision path). Optionally carries structured detail so
    callers can render per-symbol result codes for observability without
    parsing the str message.

    ``error_class`` mirrors ``_OkxErrorClass.value`` from okx_client
    (``"retryable" | "permanent" | "insufficient"``); ``error_detail`` is
    the same short string the EXHAUST log line emits
    (``"got_15_expected_16" | "http_429" | "okx_code_50011" | ...``).
    ``rtt_ms`` is the HTTP round-trip time of the final attempt when a
    response was actually received from OKX (INSUFFICIENT/RETRYABLE-from-
    HTTP/PERMANENT-from-HTTP). It is ``None`` when no response was
    received (pre-response network exception: DNS fail, connect refused,
    pre-bytes timeout) -- the time-to-failure of those paths isn't a
    meaningful "OKX RTT" for downstream analysis.

    All three default to None for backwards compatibility with bare
    ``raise TransientOkxError("msg")`` callsites.
    """

    def __init__(
        self,
        message: str,
        *,
        error_class: str | None = None,
        error_detail: str | None = None,
        rtt_ms: int | None = None,
    ) -> None:
        super().__init__(message)
        self.error_class = error_class
        self.error_detail = error_detail
        self.rtt_ms = rtt_ms


# -- Paths ---------------------------------------------------------

def ensure_parent_dir(path: str) -> None:
    p = Path(path)
    parent = p.parent
    if parent and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)


# -- Money ---------------------------------------------------------

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
