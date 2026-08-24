#!/usr/bin/env python3
"""Gate a ``pancakebot-live`` restart. Exit 0 = safe, non-zero = blocked.

Written because the pre-restart checks kept being a PROMISE rather than a
guard, and twice on 2026-08-24 the promise did not bind: one restart went
out before the deploy had been pulled, and one went out 17 seconds after a
bet was submitted, while that position was open.

Use it as a gate, not as information::

    .venv/bin/python scripts/preflight_restart.py && systemctl restart pancakebot-live

Three conditions, all of which must hold:

  1. NO OPEN POSITION. "Open" is `bet_ledger._OPEN_STATUSES`, imported
     rather than re-implemented -- that is the whole point. Two ad-hoc
     versions of this test were written by hand on 2026-08-24 and BOTH
     were wrong in different ways: the first treated ``LATE`` as open
     (it is terminal -- the TX reverted, no position was ever taken),
     and the second tested ``SUBMITTED`` minus terminal, which misses
     ``CONFIRMED`` -- exactly the state of the bet that was live during
     the restart it was supposed to prevent. The ledger module already
     knows the answer; ask it.

  2. THE LAST DECISION WAS A SKIP. A restart immediately after a BET
     decision races the submit/confirm path. After a SKIP the bot has
     nothing in flight for that round.

  3. ENOUGH SLACK BEFORE THE NEXT LOCK. This is the calculation that was
     being done by hand every time ("settles at T, restart at T+n, next
     lock at T+m"), which is the part most likely to be got wrong at 4am.
     Measured cold-start to first completed poll is 12-13s (03:30:18 ->
     :30, 13:44:37 -> :50), so the default 90s floor leaves a wide margin.

Read-only: it queries systemd, the chain and the ledger, and changes
nothing.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pancakebot import paths  # noqa: E402
from pancakebot.constants import PREDICTION_V2_CONTRACT_ADDRESS  # noqa: E402
from pancakebot.runtime.bet_ledger import (  # noqa: E402
    _OPEN_STATUSES,
    load_ledger,
)

DEFAULT_MIN_SLACK_S = 90.0
DEFAULT_UNIT = "pancakebot-live"

# Fallback chain, deliberately more than one entry: publicnode returned
# HTTP 403 rate-limits during read bursts on 2026-08-24, and a preflight
# gate that fails open because one endpoint throttled would be worse than
# no gate at all. A read failure BLOCKS (see main); the fallbacks exist so
# that blocking is rare and honest rather than routine.
DEFAULT_RPCS = (
    "https://bsc-dataseed.bnbchain.org",
    "https://bsc-dataseed1.defibit.io",
    "https://bsc-dataseed1.ninicoin.io",
    "https://bsc-rpc.publicnode.com",
)


# ---- 1. open positions ---------------------------------------------------

def open_positions(ledger_path: str = paths.LIVE_BETS_LEDGER_PATH,
                   *, loader=load_ledger) -> list[tuple[int, str]]:
    """``[(epoch, status)]`` for every epoch whose MERGED latest status is
    still open. Empty list means nothing is at risk."""
    merged = loader(ledger_path)
    return sorted(
        (int(epoch), str(rec.get("status", "?")))
        for epoch, rec in merged.items()
        if rec.get("status") in _OPEN_STATUSES
    )


# ---- 2. the last decision ------------------------------------------------

def _journal(unit: str, lines: int) -> str:
    return subprocess.run(
        ["journalctl", "-u", unit, "-n", str(lines), "--no-pager", "-o", "cat"],
        capture_output=True, text=True, timeout=30,
    ).stdout


def last_decision(unit: str = DEFAULT_UNIT, *, lines: int = 200,
                  journal=_journal) -> tuple[str | None, str]:
    """``("SKIP"|"BET"|None, line)`` for the most recent round decision."""
    for line in reversed((journal(unit, lines) or "").splitlines()):
        if "Skipped epoch" in line:
            return "SKIP", line.strip()
        if " BET " in line and "Bet " in line:
            return "BET", line.strip()
    return None, ""


# ---- 3. slack before the next lock ---------------------------------------

def _chain_next_lock(rpcs=DEFAULT_RPCS) -> tuple[int, int]:
    """``(epoch, lock_ts)`` for the currently OPEN round, off the chain."""
    from web3 import Web3

    abi = json.loads(Path(paths.ABI_JSON_PATH).read_text(encoding="utf-8"))
    if isinstance(abi, dict):
        abi = abi.get("abi", abi)
    last_err: Exception | None = None
    for url in rpcs:
        try:
            w3 = Web3(Web3.HTTPProvider(url, request_kwargs={"timeout": 15}))
            c = w3.eth.contract(
                address=Web3.to_checksum_address(PREDICTION_V2_CONTRACT_ADDRESS),
                abi=abi)
            epoch = int(c.functions.currentEpoch().call())
            return epoch, int(c.functions.rounds(epoch).call()[2])
        except Exception as e:  # noqa: BLE001 — try the next endpoint
            last_err = e
    raise RuntimeError(f"every RPC endpoint failed; last error: {last_err!r}")


def lock_slack(*, now: float | None = None,
               next_lock=_chain_next_lock) -> tuple[int, float]:
    """``(epoch, seconds_until_lock)`` for the open round."""
    epoch, lock_ts = next_lock()
    return epoch, lock_ts - (time.time() if now is None else now)


# ---- gate ----------------------------------------------------------------

def check(*, unit: str, ledger_path: str, min_slack_s: float) -> list[str]:
    """Return the list of blockers. Empty list means safe to restart."""
    blockers: list[str] = []

    positions = open_positions(ledger_path)
    if positions:
        detail = ", ".join(f"epoch {e} status={s}" for e, s in positions)
        blockers.append(f"OPEN POSITION: {detail} — wait for settlement")

    kind, line = last_decision(unit)
    if kind == "BET":
        blockers.append(
            f"LAST DECISION WAS A BET, not a skip — a submit may be in "
            f"flight: {line}")
    elif kind is None:
        blockers.append(
            "NO DECISION FOUND in the recent journal — cannot confirm the "
            "bot is cycling; check the unit before restarting")

    try:
        epoch, slack = lock_slack()
    except Exception as e:  # noqa: BLE001
        blockers.append(f"CHAIN READ FAILED, cannot size the window: {e}")
    else:
        if slack < min_slack_s:
            blockers.append(
                f"ONLY {slack:.0f}s BEFORE THE NEXT LOCK (epoch {epoch}, "
                f"need {min_slack_s:.0f}s) — restart would race the next "
                f"decision; wait ~{slack + 60:.0f}s for the round to turn over")
        else:
            print(f"  slack: {slack:.0f}s before epoch {epoch} locks "
                  f"(need {min_slack_s:.0f}s)")
    return blockers


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--unit", default=DEFAULT_UNIT)
    ap.add_argument("--ledger", default=paths.LIVE_BETS_LEDGER_PATH)
    ap.add_argument("--min-slack-s", type=float, default=DEFAULT_MIN_SLACK_S)
    args = ap.parse_args(argv)

    print(f"preflight_restart: unit={args.unit} ledger={args.ledger}")
    blockers = check(unit=args.unit, ledger_path=args.ledger,
                     min_slack_s=args.min_slack_s)
    if blockers:
        print("\nBLOCKED — do not restart:")
        for b in blockers:
            print(f"  * {b}")
        return 1
    print("\nOK — no open position, last decision was a skip, window is wide "
          "enough. Safe to restart.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
