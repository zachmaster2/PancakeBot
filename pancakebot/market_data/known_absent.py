"""Epochs that are permanently absent from the kline stores, by fact.

WHY THIS EXISTS. Without it the contiguity report flags the same eight
epochs as gaps on every single run, forever. That is how a real finding
becomes background noise and then gets ignored -- and a report nobody reads
is worth less than no report, because it also carries the false comfort of
having one.

PROVENANCE. These are the `unfetchable_klines` recorded in the 2026-08-24
repair MANIFEST. The repair fetched all 23 missing closed rounds from The
Graph successfully, but OKX serves 1s klines for only ~171.6 days and these
eight epochs had already aged out. They cannot be refetched by anyone, ever.
The whole-store byte scan independently rediscovered exactly this set on
2026-08-30 and nothing else, which is what confirms the list is complete.

THIS IS AN EXPLANATION, NEVER A PERMISSION. Listing an epoch here changes
how the REPORT describes it. It does not license the sync to skip anything,
and it does not suppress a gap anywhere else in the range. An epoch absent
from a store and absent from this list is still a gap and is still reported.

closed_rounds.jsonl is NOT covered: it is fully contiguous
(437562..511578, zero missing) and needs no exceptions.
"""
from __future__ import annotations

# The 2026-08-24 repair, MANIFEST.json -> provenance.unfetchable_klines
KNOWN_ABSENT_KLINE_EPOCHS: frozenset[int] = frozenset({
    445330, 445331,
    447533, 447534,
    449665, 449666,
    452486, 452487,
})

# Kept as a mapping so a future store-specific exception does not have to
# reuse this one. Absent keys mean "no known exceptions", which is the
# correct default for anything not listed.
KNOWN_ABSENT_BY_STORE: dict[str, frozenset[int]] = {
    "bnb_spot_prices.jsonl": KNOWN_ABSENT_KLINE_EPOCHS,
    "btc_spot_prices.jsonl": KNOWN_ABSENT_KLINE_EPOCHS,
    "eth_spot_prices.jsonl": KNOWN_ABSENT_KLINE_EPOCHS,
    "sol_spot_prices.jsonl": KNOWN_ABSENT_KLINE_EPOCHS,
}


def known_absent_for(path: str) -> frozenset[int]:
    """Known-absent epochs for a store path, empty when none are recorded."""
    name = path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    return KNOWN_ABSENT_BY_STORE.get(name, frozenset())
