from __future__ import annotations

import gzip
import hashlib
import json
import os
import pickle
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from pancakebot.core.logging import warn
from pancakebot.domain.types import Round

_RUNTIME_PIPELINE_SNAPSHOT_VERSION = "runtime_pipeline_snapshot_v1"


def _stable_hash(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return str(hashlib.sha256(data).hexdigest())


def _round_fingerprint(*, round_t: Round) -> dict[str, object]:
    return {
        "epoch": int(round_t.epoch),
        "hash": str(_stable_hash(round_t.to_json())),
    }


def runtime_pipeline_snapshot_compatibility_key(
    *,
    strategy_cfg: object,
    cutoff_seconds: int,
    treasury_fee_fraction: float,
    round_store_path: str,
    klines_store_path: str,
) -> str:
    payload = {
        "version": str(_RUNTIME_PIPELINE_SNAPSHOT_VERSION),
        "cutoff_seconds": int(cutoff_seconds),
        "treasury_fee_fraction": float(treasury_fee_fraction),
        "round_store_path": str(round_store_path),
        "klines_store_path": str(klines_store_path),
        "strategy_cfg": asdict(strategy_cfg),
    }
    return str(_stable_hash(payload))


def load_runtime_pipeline_snapshot(
    *,
    path: str,
    compatibility_key: str,
    rounds_by_epoch: dict[int, Round],
) -> dict[str, object] | None:
    snapshot_path = Path(str(path))
    if not snapshot_path.exists():
        return None

    try:
        with gzip.open(snapshot_path, "rb") as f:
            raw = pickle.load(f)
    except Exception as e:
        warn("RUN", "CACHE", "LOAD", msg=f"path={snapshot_path} err={e}")
        return None

    if not isinstance(raw, dict):
        warn("RUN", "CACHE", "LOAD", msg=f"path={snapshot_path} invalid_snapshot_type=1")
        return None
    if str(raw.get("version", "")) != str(_RUNTIME_PIPELINE_SNAPSHOT_VERSION):
        return None
    if str(raw.get("compatibility_key", "")) != str(compatibility_key):
        return None

    pipeline_state = raw.get("pipeline_state")
    if not isinstance(pipeline_state, dict):
        warn("RUN", "CACHE", "LOAD", msg=f"path={snapshot_path} pipeline_state_missing=1")
        return None

    last_settled_epoch = raw.get("last_settled_epoch")
    if last_settled_epoch is None:
        return dict(raw)

    try:
        settled_epoch = int(last_settled_epoch)
    except Exception:
        warn("RUN", "CACHE", "LOAD", msg=f"path={snapshot_path} last_settled_epoch_invalid=1")
        return None

    current_round = rounds_by_epoch.get(int(settled_epoch))
    if current_round is None:
        return None

    saved_fingerprint = raw.get("last_settled_round_fingerprint")
    if not isinstance(saved_fingerprint, dict):
        warn("RUN", "CACHE", "LOAD", msg=f"path={snapshot_path} last_round_fingerprint_missing=1")
        return None
    if dict(saved_fingerprint) != _round_fingerprint(round_t=current_round):
        return None

    return dict(raw)


def save_runtime_pipeline_snapshot(
    *,
    path: str,
    compatibility_key: str,
    pipeline_state: dict[str, object],
    last_settled_round: Round | None,
) -> None:
    snapshot_path = Path(str(path))
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = snapshot_path.with_name(
        f"{snapshot_path.name}.{int(os.getpid())}.{str(uuid.uuid4().hex)}.tmp"
    )
    payload = {
        "version": str(_RUNTIME_PIPELINE_SNAPSHOT_VERSION),
        "compatibility_key": str(compatibility_key),
        "saved_ts": int(time.time()),
        "last_settled_epoch": (
            None
            if pipeline_state.get("last_settled_epoch") is None
            else int(pipeline_state["last_settled_epoch"])
        ),
        "last_settled_round_fingerprint": (
            None if last_settled_round is None else _round_fingerprint(round_t=last_settled_round)
        ),
        "pipeline_state": dict(pipeline_state),
    }

    with gzip.open(tmp_path, "wb", compresslevel=3) as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    last_err: Exception | None = None
    for attempt in range(8):
        try:
            tmp_path.replace(snapshot_path)
            return
        except PermissionError as e:
            last_err = e
            time.sleep(0.03 * float(attempt + 1))

    if snapshot_path.exists():
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        warn("RUN", "CACHE", "SAVE", msg=f"path={snapshot_path} concurrent_replace_used_existing=1")
        return

    if last_err is not None:
        raise last_err
