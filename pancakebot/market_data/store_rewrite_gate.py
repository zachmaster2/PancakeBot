"""One switch controlling every path that can REPLACE a whole store.

WHY THIS EXISTS. On 2026-08-30 the project moved to data-collection-only
and the Frankfurt VM was destroyed. The five store files on the operator's
Windows machine became the ONLY surviving copy of history that cannot be
refetched: OKX serves 1s klines for ~171.6 days, and everything older than
that exists nowhere else in the world.

At that moment an append-only file stopped being a convenience and became
the safety property. Two code paths violate it:

  * KlineStore.rewrite()                     -- via _prepend_staging_to_store
  * ClosedRoundsStore older-backfill         -- tmp_path.replace(store_path)

Both write a complete new file and atomically swap it over the original.
Atomicity protects against a TORN write; it does not protect against a
WRONG one. If the merge produces bad output, the swap makes it permanent
and the original is gone. That is an acceptable risk for an operator
running a backfill and watching it. It is not acceptable for an unattended
scheduled job.

THE DAILY SYNC NEVER NEEDS EITHER PATH. It only moves forward, appending
epochs newer than what is on disk. A prepend is only ever wanted when a
human has decided to backfill.

REFUSAL MUST NOT ABORT THE RUN. The forward append always completes first
and its data is already on disk. Skipping a backfill loses nothing that a
later deliberate run cannot recover; aborting on a discovered gap would
stop ALL forward collection indefinitely, turning a backfill need into a
permanent silent stop -- the precise failure this design exists to
prevent. So the gate SKIPS, warns loudly, and lets the sync exit 0.

To backfill deliberately:  ALLOW_STORE_REWRITE=1 python run.py --sync
"""
from __future__ import annotations

import os

ENV_VAR = "ALLOW_STORE_REWRITE"

_TRUE = frozenset({"1", "true", "yes", "on"})


def store_rewrite_allowed() -> bool:
    """True only when the operator has explicitly opted in, this run.

    Default is False. Anything unset, empty, or unrecognised is False --
    a typo must fail CLOSED, toward preserving data.
    """
    return str(os.environ.get(ENV_VAR, "")).strip().lower() in _TRUE
