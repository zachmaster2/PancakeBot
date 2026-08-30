"""The stores must write a bare LF on every platform, Windows included.

WHY THIS EXISTS. Every write in kline_store/round_store used Python's
default text mode, which translates "\\n" -> "\\r\\n" on Windows. The
terminator in the file was therefore whichever platform happened to append
last. As of 2026-08-29 all five canonical stores are internally mixed, and
the two largest carry TWO transitions each -- LF->CRLF at line 39686 and
CRLF->LF at line 56753 -- a record of the writer changing hands twice.

It was harmless only by luck. Reads use the universal-newline default, so
"\\r\\n", "\\r" and "\\n" all normalise to "\\n" before json.loads ever
sees them; the canonical 5-fold hash was measured to reproduce
bit-for-bit over an all-CRLF copy of the stores. But that tolerance is
CPython's default, not a property this codebase asked for, and a mixed
store defeats byte-level diffing, costs one byte per CRLF line, and makes
"is this file intact?" unanswerable by any simple check.

SEQUENCING. This fix has to land BEFORE the stores are normalised. Ordered
the other way, the next sync run from Windows re-mixes a 4.37 GiB file
that was just rewritten -- the rewrite would be undone by the first append.

READS ARE DELIBERATELY LEFT ALONE. Their universal-newline tolerance is
what allows the existing mixed files to load at all; pinning newline="" on
a read would make "\\r" a literal character inside the parsed line.
"""
from __future__ import annotations

import io
import json
import re
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pancakebot.market_data.kline_store import KlineStore  # noqa: E402
from pancakebot.market_data.round_store import ClosedRoundsStore  # noqa: E402
from pancakebot.types import Bet, Round  # noqa: E402


def _rec(epoch: int) -> dict:
    return {"epoch": epoch, "closes": [[1787000000000, "700.1"]]}


def _round(epoch: int) -> Round:
    return Round(
        epoch=epoch, start_at=1787000000 + epoch,
        lock_at=1787000300 + epoch, lock_price=700.0,
        close_price=701.0, position="Bull", failed=False,
        bets=[Bet(wallet_address="0xabc", amount_wei=10**15,
                  position="Bull", created_at=1787000010 + epoch)],
    )


def _terminators(path) -> set[bytes]:
    """The DISTINCT line terminators actually present in the file bytes."""
    out: set[bytes] = set()
    with io.open(path, "rb") as f:
        for raw in f:
            if raw.endswith(b"\r\n"):
                out.add(b"\r\n")
            elif raw.endswith(b"\n"):
                out.add(b"\n")
            elif raw.endswith(b"\r"):
                out.add(b"\r")
    return out


# ---- the kline store ------------------------------------------------------

def test_kline_append_writes_bare_lf(tmp_path):
    """THE regression, on the hot path: every sync appends here."""
    p = str(tmp_path / "k.jsonl")
    s = KlineStore(p)
    s.write_new([_rec(1), _rec(2)])
    s.append_after(2, [_rec(3), _rec(4)])
    assert _terminators(p) == {b"\n"}, (
        f"kline store wrote {_terminators(p)}; on Windows a default-text-mode "
        f"open emits \\r\\n and re-mixes the store on the next sync")


def test_kline_write_new_writes_bare_lf(tmp_path):
    p = str(tmp_path / "k2.jsonl")
    KlineStore(p).write_new([_rec(i) for i in range(1, 5)])
    assert _terminators(p) == {b"\n"}


# ---- the round store ------------------------------------------------------

def test_round_append_writes_bare_lf(tmp_path):
    p = str(tmp_path / "r.jsonl")
    s = ClosedRoundsStore(p)
    s.write_new_store([_round(1), _round(2)])
    s.append_rounds([_round(3), _round(4)])
    assert _terminators(p) == {b"\n"}


def test_round_write_new_store_writes_bare_lf(tmp_path):
    p = str(tmp_path / "r2.jsonl")
    ClosedRoundsStore(p).write_new_store([_round(i) for i in range(1, 5)])
    assert _terminators(p) == {b"\n"}


# ---- appending to a legacy CRLF file must not extend the mixture ---------

def test_appending_to_a_crlf_file_writes_lf_and_still_reads(tmp_path):
    """The real migration case. A store already carrying CRLF gets new LF
    records; the file is transiently mixed and MUST still parse, because
    that is the state every store is in until normalisation runs."""
    p = tmp_path / "legacy.jsonl"
    with io.open(p, "wb") as f:
        for e in (1, 2):
            f.write((json.dumps(_round(e).to_json(), separators=(",", ":"))
                     + "\r\n").encode())
    s = ClosedRoundsStore(str(p))
    s.append_rounds([_round(3)])
    assert _terminators(p) == {b"\r\n", b"\n"}, "expected a transiently mixed file"
    got = [r.epoch for r in s.iter_closed_rounds()]
    assert got == [1, 2, 3], f"mixed file did not read back cleanly: {got}"


# ---- structural: no write may fall back to the platform default ---------

@pytest.mark.parametrize("module", ["kline_store", "round_store"])
def test_every_write_open_pins_the_newline(module):
    """Catches a NEW write site added later without newline="". Reads are
    excluded on purpose -- their tolerance is load-bearing."""
    src = (_REPO_ROOT / "pancakebot" / "market_data" / f"{module}.py").read_text(
        encoding="utf-8")
    opens = re.findall(r'open\([^)]*?"(?:w|a|r\+)"[^)]*\)', src, re.S)
    assert opens, f"no write opens found in {module} — did it move?"
    bad = [" ".join(o.split()) for o in opens if "newline=" not in o]
    assert not bad, (
        f"{module} has write open(s) with no explicit newline: {bad}. On "
        f"Windows these emit \\r\\n and re-mix the store on the next append.")
