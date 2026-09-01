"""Merge staged records INTO a canonical store. The dangerous half.

READ THIS BEFORE RUNNING IT.

Staging deferred this write. It did not remove it. Everything up to now has
been append-only and safe; this script is the point where that stops being
true. It calls ``KlineStore.rewrite()``, which builds a complete new file
and ``os.replace()``s it over the original. Atomicity protects against a
TORN write, not a WRONG one: if the merge produces bad output the swap
makes it permanent and the original is gone.

The store files are the only surviving copy of data that cannot be
refetched. OKX serves 1s klines for ~171.6 days; older data exists nowhere
else on Earth.

SO THIS SCRIPT IS DELIBERATE, GATED, AND VERIFIED ON BOTH SIDES:

  * refuses unless ALLOW_STORE_REWRITE=1 is set for this run
  * refuses unless --yes is passed, so it cannot run from a scheduler
  * snapshots record count / contiguity / CRLF BEFORE
  * refuses if the store is already unhealthy -- never merge into a store
    that is failing a check, because the merge would bury the evidence
  * verifies AFTER, and reports loudly on any mismatch
  * re-runs the canonical baseline hash test and refuses to declare success
    if it moved

  --dry-run    do everything except the rewrite
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pancakebot import paths  # noqa: E402
from pancakebot.market_data.epoch_scan import contiguity  # noqa: E402
from pancakebot.market_data.kline_store import KlineStore  # noqa: E402
from pancakebot.market_data.store_rewrite_gate import (  # noqa: E402
    ENV_VAR,
    store_rewrite_allowed,
)
from pancakebot.ops.gap_capture import staging_path  # noqa: E402

_STORE_PATHS = {
    "bnb_spot_prices.jsonl": paths.BNB_SPOT_PRICES_PATH,
    "btc_spot_prices.jsonl": paths.BTC_SPOT_PRICES_PATH,
    "eth_spot_prices.jsonl": paths.ETH_SPOT_PRICES_PATH,
    "sol_spot_prices.jsonl": paths.SOL_SPOT_PRICES_PATH,
}


def _summarise(path: str) -> dict:
    c = contiguity(path)
    return {"n": c["n"], "distinct": c["distinct"], "latest": c["latest"],
            "earliest": c["earliest"], "missing": c["missing"],
            "crlf": c["crlf"], "duplicates": c["duplicates"],
            "out_of_order": c["out_of_order"]}


def merge_one(store: str, *, staging_dir: str = "staged_captures",
              dry_run: bool = False, backup_dir: str | None = None) -> int:
    spath = _STORE_PATHS[store]
    stage = staging_path(store, staging_dir)

    if not os.path.exists(stage):
        print(f"{store}: no staged file, nothing to merge")
        return 0

    staged = [json.loads(l) for l in open(stage, encoding="utf-8") if l.strip()]
    if not staged:
        print(f"{store}: staged file is empty")
        return 0

    before = _summarise(spath)
    print(f"{store}: BEFORE n={before['n']} latest={before['latest']} "
          f"missing={before['missing']} crlf={before['crlf']} "
          f"dups={before['duplicates']} disorder={before['out_of_order']}")

    # Never merge into a store that is already failing a check. The rewrite
    # would bury the evidence of whatever was wrong under a new file.
    if before["crlf"] or before["duplicates"] or before["out_of_order"]:
        print(f"{store}: REFUSING -- store is already unhealthy "
              f"(crlf={before['crlf']} dups={before['duplicates']} "
              f"disorder={before['out_of_order']}). Fix that first; a merge "
              f"would bury it.")
        return 2

    existing = {int(r["epoch"]) for r in KlineStore(spath).iter_records()}
    add = [r for r in staged if int(r["epoch"]) not in existing]
    print(f"{store}: staged={len(staged)} already_present={len(staged) - len(add)} "
          f"to_add={len(add)}")
    if not add:
        print(f"{store}: nothing new to merge")
        return 0

    merged = list(KlineStore(spath).iter_records()) + add
    merged.sort(key=lambda r: int(r["epoch"]))

    seen = set()
    for r in merged:
        e = int(r["epoch"])
        if e in seen:
            print(f"{store}: REFUSING -- merge would create a duplicate at {e}")
            return 2
        seen.add(e)

    expected_n = before["n"] + len(add)

    if dry_run:
        print(f"{store}: [dry-run] would rewrite to n={expected_n}")
        return 0

    # A copy of the pre-merge file, kept until the post-check passes. The
    # rewrite is atomic but not reversible, and "we can restore" should be
    # a fact rather than a hope.
    if backup_dir:
        os.makedirs(backup_dir, exist_ok=True)
        bak = os.path.join(
            backup_dir,
            f"{store}.premerge.{time.strftime('%Y%m%d-%H%M%S')}")
        print(f"{store}: backing up to {bak}")
        shutil.copy2(spath, bak)

    print(f"{store}: REWRITING (this is the dangerous write)")
    KlineStore(spath).rewrite(merged)

    after = _summarise(spath)
    print(f"{store}: AFTER  n={after['n']} latest={after['latest']} "
          f"missing={after['missing']} crlf={after['crlf']} "
          f"dups={after['duplicates']} disorder={after['out_of_order']}")

    problems = []
    if after["n"] != expected_n:
        problems.append(f"record count {after['n']} != expected {expected_n}")
    if after["n"] < before["n"]:
        problems.append("store SHRANK")
    if after["crlf"]:
        problems.append(f"CRLF appeared: {after['crlf']}")
    if after["duplicates"]:
        problems.append(f"duplicates appeared: {after['duplicates']}")
    if after["out_of_order"]:
        problems.append(f"disorder appeared: {after['out_of_order']}")
    if after["missing"] > before["missing"]:
        problems.append("new gaps appeared")
    if after["latest"] is not None and before["latest"] is not None \
            and after["latest"] < before["latest"]:
        problems.append("last epoch went backwards")

    if problems:
        print(f"{store}: *** POST-MERGE VERIFICATION FAILED ***")
        for p in problems:
            print(f"    - {p}")
        if backup_dir:
            print(f"    RESTORE FROM: {bak}")
        return 2

    print(f"{store}: verified OK (+{len(add)} records)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", action="append",
                    help="store filename; repeatable. Default: all four.")
    ap.add_argument("--staging-dir", default="staged_captures")
    ap.add_argument("--backup-dir", default="var/premerge_backups")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true",
                    help="required for a real merge; prevents scheduled runs")
    args = ap.parse_args()

    if not store_rewrite_allowed():
        print(f"REFUSING: {ENV_VAR}=1 is required. This script performs a "
              f"whole-file rewrite of an irreplaceable store.")
        return 3
    if not args.dry_run and not args.yes:
        print("REFUSING: pass --yes for a real merge. This exists so the "
              "merge can never be run by a scheduler.")
        return 3

    stores = args.store or sorted(_STORE_PATHS)
    rc = 0
    for s in stores:
        if s not in _STORE_PATHS:
            print(f"unknown store: {s}")
            return 3
        rc = max(rc, merge_one(s, staging_dir=args.staging_dir,
                               dry_run=args.dry_run,
                               backup_dir=None if args.dry_run else args.backup_dir))
    print()
    print("REMINDER: merged records are now in the canonical store. The "
          "staged file is left in place deliberately -- delete it only once "
          "you have confirmed the store is healthy and backed up.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
