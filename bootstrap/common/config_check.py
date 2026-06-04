"""Validate that config + secrets are present before installing the service.

Checks (fail-fast, no mutation):
  - ``config.toml`` exists at the repo root and parses, with the expected
    top-level sections.
  - ``.env`` exists with the required secret KEYS present and non-empty
    (values are never printed).
  - The Discord webhook env vars are set in the environment (machine-scope on
    Windows / EnvironmentFile on Linux). Missing webhooks are a WARNING, not a
    hard failure — the bot runs without alerts.

Exit code 0 = ready to install; non-zero = blockers found.
"""
from __future__ import annotations

import os
import sys
import tomllib
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

_REQUIRED_SECRET_KEYS = ("THE_GRAPH_API_KEY", "BSC_WALLET_PRIVATE_KEY")
_EXPECTED_CONFIG_SECTIONS = ("runtime", "live", "dry", "strategy")
_WEBHOOK_ENV_VARS = (
    "PANCAKEBOT_LIVE_ALERTS_DISCORD_WEBHOOK_URL",
    "PANCAKEBOT_DRY_ALERTS_DISCORD_WEBHOOK_URL",
    "PANCAKEBOT_GENERAL_DISCORD_WEBHOOK_URL",
)


def _log(msg: str) -> None:
    print(f"[config_check] {msg}", flush=True)


def _parse_env_keys(env_path: Path) -> set[str]:
    keys: set[str] = set()
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if v.strip():
            keys.add(k.strip())
    return keys


def check(*, repo_root: Path = _REPO_ROOT, env: dict | None = None) -> list[str]:
    """Return a list of blocker strings (empty = ready). Warnings are logged but
    not returned as blockers."""
    env = os.environ if env is None else env
    blockers: list[str] = []

    cfg = repo_root / "config.toml"
    if not cfg.exists():
        blockers.append("config.toml missing at repo root")
    else:
        try:
            data = tomllib.loads(cfg.read_text(encoding="utf-8"))
            missing = [s for s in _EXPECTED_CONFIG_SECTIONS if s not in data]
            if missing:
                blockers.append(f"config.toml missing sections: {missing}")
            else:
                _log("config.toml OK")
        except Exception as e:  # noqa: BLE001
            blockers.append(f"config.toml failed to parse: {e}")

    env_path = repo_root / ".env"
    if not env_path.exists():
        blockers.append(".env missing (needs " + ", ".join(_REQUIRED_SECRET_KEYS) + ")")
    else:
        keys = _parse_env_keys(env_path)
        miss = [k for k in _REQUIRED_SECRET_KEYS if k not in keys]
        if miss:
            blockers.append(f".env missing/empty keys: {miss}")
        else:
            _log(".env has required secret keys")

    set_webhooks = [w for w in _WEBHOOK_ENV_VARS if env.get(w)]
    if len(set_webhooks) < len(_WEBHOOK_ENV_VARS):
        missing_w = [w for w in _WEBHOOK_ENV_VARS if not env.get(w)]
        _log(f"WARNING: Discord webhook env vars not set: {missing_w} "
             f"(bot runs without those alerts)")
    else:
        _log("all 3 Discord webhook env vars set")
    return blockers


def main(argv: list[str] | None = None) -> int:
    blockers = check()
    if blockers:
        for b in blockers:
            _log(f"BLOCKER: {b}")
        return 1
    _log("OK: config + secrets ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
