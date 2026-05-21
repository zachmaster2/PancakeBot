"""General utility helpers: exceptions, paths, money formatting."""
from __future__ import annotations

from pathlib import Path


# -- Exceptions ----------------------------------------------------

class InvariantError(Exception):
    pass


class TransientRpcError(Exception):
    pass


class GasPriceCapBreachedError(Exception):
    """Raised when eth.gas_price exceeds the operator-set MAX_GAS_PRICE_WEI
    ceiling. Indicates the cap is below current network reality; the
    operator must lift the cap and review before resuming bets/claims.

    The bot does NOT crash on this — callers in the bet/claim paths catch,
    alert, skip the action, and continue running. The next round retries
    naturally; if the network is sustained above the cap the alerts will
    repeat until the operator intervenes.
    """
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
    the same short string the EXHAUST log line used to emit
    (``"got_15_expected_16" | "http_429" | "okx_code_50011" | ...``).
    ``rtt_ms`` is the HTTP round-trip time of the final attempt when a
    response was actually received from OKX (INSUFFICIENT/RETRYABLE-from-
    HTTP/PERMANENT-from-HTTP). It is ``None`` when no response was
    received (pre-response network exception: DNS fail, connect refused,
    pre-bytes timeout) -- the time-to-failure of those paths isn't a
    meaningful "OKX RTT" for downstream analysis.

    ``received_count`` / ``requested_count`` carry the partial-response
    shape: how many rows OKX returned vs how many were asked for. Both
    are ``None`` when the request didn't reach the response-parsing path
    (pre-response failure).

    All fields default to None for backwards compatibility with bare
    ``raise TransientOkxError("msg")`` callsites.
    """

    def __init__(
        self,
        message: str,
        *,
        error_class: str | None = None,
        error_detail: str | None = None,
        rtt_ms: int | None = None,
        received_count: int | None = None,
        requested_count: int | None = None,
    ) -> None:
        super().__init__(message)
        self.error_class = error_class
        self.error_detail = error_detail
        self.rtt_ms = rtt_ms
        self.received_count = received_count
        self.requested_count = requested_count


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
