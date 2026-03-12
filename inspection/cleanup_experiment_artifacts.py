from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import time
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--exp-root", type=str, default="../PancakeBot_var_exp")
    p.add_argument("--state-cache-dir", type=str, default="../PancakeBot_var_exp/backtest_state_cache")
    p.add_argument("--run-registry-db", type=str, default="../PancakeBot_var_exp/run_registry_v1.sqlite")
    p.add_argument("--feature-cache-db", type=str, default="../PancakeBot_var_exp/feature_cache_v8.sqlite")
    p.add_argument("--projection-cache-db", type=str, default="../PancakeBot_var_exp/projection_cache_v1.sqlite")
    p.add_argument("--market-data-db", type=str, default="../PancakeBot_var_exp/market_data_v1.sqlite")
    p.add_argument("--delete-failed-older-days", type=int, default=14)
    p.add_argument("--prune-state-cache-older-days", type=int, default=14)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--vacuum-dbs", action="store_true")
    return p


def _prune_state_cache(*, state_cache_dir: Path, older_than_days: int, dry_run: bool) -> int:
    if int(older_than_days) < 0:
        return 0
    if not state_cache_dir.exists():
        return 0
    cutoff_ts = float(time.time()) - float(int(older_than_days) * 86400)
    deleted = 0
    for p in state_cache_dir.rglob("*.pkl.gz"):
        try:
            mtime = float(p.stat().st_mtime)
        except OSError:
            continue
        if float(mtime) > float(cutoff_ts):
            continue
        deleted += 1
        if not bool(dry_run):
            try:
                p.unlink()
            except OSError:
                continue
    return int(deleted)


def _failed_runs_to_remove(*, run_registry_db: Path, older_than_days: int) -> list[str]:
    if int(older_than_days) < 0:
        return []
    if not run_registry_db.exists():
        return []
    cutoff_ts = int(time.time()) - int(older_than_days) * 86400
    conn = sqlite3.connect(str(run_registry_db))
    try:
        rows = conn.execute(
            """
            SELECT run_name
            FROM runs
            WHERE status = 'failed'
              AND finished_at_ts IS NOT NULL
              AND finished_at_ts <= ?
            ORDER BY finished_at_ts ASC
            """,
            (int(cutoff_ts),),
        ).fetchall()
    finally:
        conn.close()
    return [str(r[0]) for r in rows]


def _remove_failed_run_dirs(*, exp_root: Path, run_names: list[str], dry_run: bool) -> int:
    removed = 0
    for run_name in run_names:
        d = exp_root / str(run_name)
        if not d.exists():
            continue
        if not d.is_dir():
            continue
        removed += 1
        if not bool(dry_run):
            shutil.rmtree(d, ignore_errors=True)
    return int(removed)


def _vacuum_db(path: Path) -> bool:
    if not path.exists():
        return False
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("VACUUM")
        conn.commit()
    finally:
        conn.close()
    return True


def main() -> None:
    args = _build_parser().parse_args()
    exp_root = Path(str(args.exp_root))
    state_cache_dir = Path(str(args.state_cache_dir))
    run_registry_db = Path(str(args.run_registry_db))

    deleted_cache_files = _prune_state_cache(
        state_cache_dir=state_cache_dir,
        older_than_days=int(args.prune_state_cache_older_days),
        dry_run=bool(args.dry_run),
    )
    failed_runs = _failed_runs_to_remove(
        run_registry_db=run_registry_db,
        older_than_days=int(args.delete_failed_older_days),
    )
    removed_failed_dirs = _remove_failed_run_dirs(
        exp_root=exp_root,
        run_names=list(failed_runs),
        dry_run=bool(args.dry_run),
    )

    vacuumed = 0
    if bool(args.vacuum_dbs) and not bool(args.dry_run):
        for db_path in (
            Path(str(args.run_registry_db)),
            Path(str(args.feature_cache_db)),
            Path(str(args.projection_cache_db)),
            Path(str(args.market_data_db)),
        ):
            if _vacuum_db(path=db_path):
                vacuumed += 1

    print(f"DRY_RUN={bool(args.dry_run)}")
    print(f"STATE_CACHE_FILES_PRUNED={int(deleted_cache_files)}")
    print(f"FAILED_RUNS_MATCHED={int(len(failed_runs))}")
    print(f"FAILED_RUN_DIRS_REMOVED={int(removed_failed_dirs)}")
    print(f"DBS_VACUUMED={int(vacuumed)}")


if __name__ == "__main__":
    main()
