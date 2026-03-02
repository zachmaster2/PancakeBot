from __future__ import annotations

import gzip
import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pancakebot.core.errors import InvariantError
from pancakebot.core.logging import warn


def stable_hash(payload: dict[str, Any]) -> str:
    """Return deterministic content hash for a JSON-serializable payload."""

    try:
        data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError) as e:
        raise InvariantError("backtest_state_cache_payload_not_json_serializable") from e
    return str(hashlib.sha256(data).hexdigest())


@dataclass(frozen=True, slots=True)
class BacktestStateCache:
    """Compressed pickle cache for reusable backtest warmup snapshots."""

    root_dir: str

    def __post_init__(self) -> None:
        if str(self.root_dir).strip() == "":
            raise InvariantError("backtest_state_cache_root_empty")
        Path(self.root_dir).mkdir(parents=True, exist_ok=True)

    def load(self, *, namespace: str, key: str) -> object | None:
        path = self._path(namespace=namespace, key=key)
        if not path.exists():
            return None
        try:
            with gzip.open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            warn("BACK", "CACHE", "LOAD", msg=f"path={path} err={e}")
            return None

    def save(self, *, namespace: str, key: str, value: object) -> Path:
        path = self._path(namespace=namespace, key=key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with gzip.open(tmp, "wb", compresslevel=3) as f:
            pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)
        return path

    def _path(self, *, namespace: str, key: str) -> Path:
        ns = str(namespace).strip()
        kk = str(key).strip()
        if ns == "" or kk == "":
            raise InvariantError("backtest_state_cache_namespace_or_key_empty")
        return Path(self.root_dir) / str(ns) / f"{kk}.pkl.gz"
