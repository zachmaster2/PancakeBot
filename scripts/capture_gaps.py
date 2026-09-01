"""Fetch missing kline records into a staging file. Never touches a store.

Run by the watchdog (which is a SEPARATE task from the sync, so a failing
capture can never block or slow data collection -- that ordering is
deliberate and collection always wins).

STAGING DEFERS THE DANGEROUS WRITE, IT DOES NOT REMOVE IT. Records captured
here are held, not repaired. Integrating them still calls store.rewrite(),
a whole-file 4.5 GB replace -- see scripts/merge_staged.py. Staged data is
a preserved problem, not a solved one.

  --dry-run   plan only: classify every gap and print, fetch nothing.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pancakebot import paths  # noqa: E402
from pancakebot.market_data.epoch_scan import contiguity  # noqa: E402
from pancakebot.ops.gap_capture import (  # noqa: E402
    OKX_KLINE_HORIZON_DAYS,
    append_staged,
    plan_capture,
    promote_permanent,
    staged_epochs,
    staging_path,
    validate_kline_record,
)

# store filename -> OKX instrument
KLINE_STORES = {
    "bnb_spot_prices.jsonl": "BNB-USDT",
    "btc_spot_prices.jsonl": "BTC-USDT",
    "eth_spot_prices.jsonl": "ETH-USDT",
    "sol_spot_prices.jsonl": "SOL-USDT",
}

_STORE_PATHS = {
    "bnb_spot_prices.jsonl": paths.BNB_SPOT_PRICES_PATH,
    "btc_spot_prices.jsonl": paths.BTC_SPOT_PRICES_PATH,
    "eth_spot_prices.jsonl": paths.ETH_SPOT_PRICES_PATH,
    "sol_spot_prices.jsonl": paths.SOL_SPOT_PRICES_PATH,
}


def build_round_index() -> dict[int, int]:
    """epoch -> start_at, for every closed round on disk.

    A kline fetch window is anchored to the round's lock_at, so an epoch
    with no round on disk cannot be captured at all. That is reported as
    BLOCKED rather than silently skipped.
    """
    idx: dict[int, int] = {}
    try:
        with open(paths.CLOSED_ROUNDS_PATH, "rb") as f:
            for raw in f:
                s = raw.strip()
                if not s:
                    continue
                try:
                    o = json.loads(s)
                    idx[int(o["epoch"])] = int(o["startAt"])
                except Exception:
                    continue
    except FileNotFoundError:
        pass
    return idx


def run(*, dry_run: bool = False, staging_dir: str | None = None,
        now: float | None = None) -> dict:
    """Returns a summary dict. Never raises for a fetch failure."""
    now = time.time() if now is None else now
    sdir = staging_dir or "staged_captures"
    summary: dict = {"captured": 0, "unrecoverable": 0, "blocked": 0,
                     "failed": 0, "already_staged": 0, "stores": {}}

    round_index = build_round_index()
    interval = 300  # PCS V2 frozen round interval; lock_at = start_at + 300

    okx = None
    for store, inst in sorted(KLINE_STORES.items()):
        spath = _STORE_PATHS[store]
        c = contiguity(spath)
        # The gap list comes from the WHOLE-STORE report, not the sync's
        # 35,000-round tail. That is the entire point of this module.
        missing: list[int] = []
        for a, b in c["runs"]:
            missing.extend(range(a, b + 1))

        stage = staging_path(store, sdir)
        plan = plan_capture(
            store=store, missing=missing, round_start_at=round_index,
            staged=staged_epochs(stage), now=now,
        )

        summary["stores"][store] = {
            "missing": len(missing),
            "fetchable": len(plan.fetchable),
            "unrecoverable": len(plan.unrecoverable),
            "blocked": len(plan.blocked),
            "already_staged": len(plan.already_staged),
            "captured": 0,
            "failed": 0,
        }
        summary["unrecoverable"] += len(plan.unrecoverable)
        summary["blocked"] += len(plan.blocked)
        summary["already_staged"] += len(plan.already_staged)

        if plan.unrecoverable and not dry_run:
            promote_permanent(
                store, plan.unrecoverable,
                reason=f"age exceeds OKX 1s kline horizon of "
                       f"{OKX_KLINE_HORIZON_DAYS} days; unfetchable by anyone")

        if not plan.fetchable or dry_run:
            continue

        if okx is None:
            from pancakebot.market_data.okx_client import OkxClient
            okx = OkxClient(timeout_seconds=10.0)
            okx.warmup(connections=4)

        from pancakebot.market_data.sync import _fetch_one_kline

        class _R:
            __slots__ = ("epoch", "lock_at")

            def __init__(self, e, la):
                self.epoch = e
                self.lock_at = la

        good: list[dict] = []
        for e in plan.fetchable:
            lock_at = round_index[e] + interval
            try:
                rec = _fetch_one_kline(_R(e, lock_at), inst, okx)
                # Validate HERE, while a refetch is still possible. Finding
                # out at merge time -- months later, past the horizon --
                # means the real data is gone and the staged copy is junk.
                validate_kline_record(rec, epoch=e, lock_at=lock_at)
                good.append(rec)
            except Exception as ex:      # noqa: BLE001
                summary["stores"][store]["failed"] += 1
                summary["failed"] += 1
                print(f"  CAPTURE FAILED {store} epoch={e}: "
                      f"{type(ex).__name__}: {str(ex)[:120]}")

        n = append_staged(stage, good)
        summary["stores"][store]["captured"] = n
        summary["captured"] += n
        if n:
            print(f"  CAPTURED {store}: {n} record(s) -> {stage}")

    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--staging-dir", default="staged_captures")
    args = ap.parse_args()

    s = run(dry_run=args.dry_run, staging_dir=args.staging_dir)
    print(json.dumps(s, indent=2, sort_keys=True))
    # Non-zero only when a fetch FAILED -- an unrecoverable epoch is not a
    # failure of this run, it is a fact about the world, and reporting it
    # as a failure every day is what trains people to ignore alarms.
    return 1 if s["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
