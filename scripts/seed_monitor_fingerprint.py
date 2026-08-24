#!/usr/bin/env python3
"""Backfill the weekly monitor's strategy fingerprint for 2026-08-23.

WHY THIS EXISTS. The monitor compares each Sunday's ``strategy_fingerprint``
against the one persisted from the previous run, and raises a CONFIG CHANGED
banner when they differ. The field was introduced on 2026-08-24, so
``state.json`` held no fingerprint at all — which would have made the
comparison a no-op on **2026-08-30**, the very first Sunday whose fire
stream runs under the 1.25 pool filter while every prior week ran under
1.5. The banner would have stayed silent on the one run it was built to
annotate, and a genuine epoch boundary would have read as a trend.

WHAT IS BEING WRITTEN. Not a guess: the configuration the 2026-08-23 run
actually evaluated under, established from TWO independent records that
agree on every value.

  1. That run's own copy of the config, which the monitor writes beside its
     decision: ``weekly_monitors/2026-08-23/risk_off_config.toml``. It is a
     verbatim copy of config.toml with only six keys overridden
     (initial_bankroll_bnb, epoch_start, epoch_end,
     max_drawdown_fraction_from_peak, min_bankroll_bnb_to_bet,
     cooldown_rounds) — none of which appear in the fingerprint.
       min_pool_bnb_at_cutoff 1.5, min_payout_multiple_at_cutoff 1.5,
       max_bet_bnb_btc_primary 0.1, max_bet_bnb_eth_sol_fallback 0.1,
       max_bet_fraction_of_bankroll 0.05, min_bet_threshold_bnb 0.01

  2. The repository at ``d750771``, the newest commit preceding that run
     (2026-08-23T05:15Z, run at 06:26Z), for the values the fingerprint
     takes from module constants rather than config:
       CUTOFF, LOOKBACKS, FEE = 2, (3, 7, 15), 0.03   (:161)
       pool_cutoff_seconds=6                          (:203)
     and config's own ``mtf_lookbacks = [3, 7, 15]`` (:65) for the
     deployed-lookbacks field.

Every fingerprint key is covered by at least one of those, and where both
speak they agree, so nothing is inferred. If that had not held, the right
answer would have been to leave the key out — a partial fingerprint still
detects a change in the keys it does carry.

PROVENANCE. The backfill marker is written as a SIBLING key,
``strategy_fingerprint_backfill``, never inside the fingerprint itself:
the monitor's comparison is a plain dict inequality, so an extra key
inside the fingerprint would make ``config_changed`` true forever.

Idempotent and refuses to overwrite: if a fingerprint is already present,
it exits without touching anything.
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DEFAULT_STATE = ("var/strategy_review/weekly_monitors/state.json")

# The configuration in effect at the 2026-08-23 monitor run. See the module
# docstring for the two records each value was read from.
FINGERPRINT_2026_08_23 = {
    "min_pool_bnb_at_cutoff": 1.5,
    "min_payout_multiple_at_cutoff": 1.5,
    "max_bet_bnb_btc_primary": 0.1,
    "max_bet_bnb_eth_sol_fallback": 0.1,
    "max_bet_fraction_of_bankroll": 0.05,
    "kline_cutoff_seconds": 2,
    "mtf_lookbacks_used_for_slicing": [3, 7, 15],
    "mtf_lookbacks_deployed": [3, 7, 15],
    "pool_cutoff_seconds": 6,
    "treasury_fee_fraction": 0.03,
    "min_bet_threshold_bnb": 0.01,
}

PROVENANCE = {
    "backfilled": True,
    "backfilled_for_week": "2026-08-23",
    "derived_from_commit": "d750771",
    "derived_from_artifact":
        "var/strategy_review/weekly_monitors/2026-08-23/risk_off_config.toml",
    "reason": (
        "strategy_fingerprint was introduced 2026-08-24, after the 2026-08-23 "
        "run. Without this the first comparison would have fallen on "
        "2026-09-06 and the CONFIG CHANGED banner would have stayed silent "
        "on 2026-08-30 — the week the 1.5 -> 1.25 pool-filter discontinuity "
        "actually lands."),
    "not_written_by_the_monitor": (
        "This fingerprint was reconstructed from records, not emitted by a "
        "monitor run. Do not read it as evidence that the 2026-08-23 run "
        "computed a fingerprint; it did not."),
}


def seed(state_path: Path, *, now: str | None = None,
         force: bool = False) -> tuple[bool, str]:
    """Returns ``(changed, message)``."""
    if not state_path.exists():
        return False, f"state file not found: {state_path}"
    st = json.loads(state_path.read_text(encoding="utf-8"))

    existing = st.get("strategy_fingerprint")
    if existing is not None and not force:
        return False, (
            f"state.json already carries a fingerprint ({existing}); refusing "
            f"to overwrite — pass --force only if you mean to replace it")

    last_week = st.get("last_week")
    if last_week != "2026-08-23" and not force:
        return False, (
            f"last_week is {last_week!r}, not '2026-08-23' — a monitor run "
            f"has happened since this backfill was designed, so seeding it "
            f"now would misdescribe the comparison. Re-derive instead.")

    st["strategy_fingerprint"] = dict(FINGERPRINT_2026_08_23)
    st["strategy_fingerprint_backfill"] = dict(
        PROVENANCE,
        backfilled_at_utc=now or datetime.datetime.now(
            datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    tmp = state_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(st, indent=2), encoding="utf-8")
    tmp.replace(state_path)
    return True, f"seeded 2026-08-23 fingerprint into {state_path}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--state", default=DEFAULT_STATE)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    path = Path(args.state)
    if args.dry_run:
        print(f"would seed {path} with:")
        print(json.dumps(FINGERPRINT_2026_08_23, indent=2))
        return 0
    changed, msg = seed(path, force=args.force)
    print(msg)
    return 0 if changed else 1


if __name__ == "__main__":
    raise SystemExit(main())
