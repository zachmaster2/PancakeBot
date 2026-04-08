from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from pancakebot.config.env import load_env
from pancakebot.config.load_config import load_app_config
from pancakebot.core.constants import GAS_PRICE_WEI


@dataclass(frozen=True, slots=True)
class PreflightCheck:
    name: str
    passed: bool
    detail: str


def _file_exists_check(*, name: str, path: str) -> PreflightCheck:
    p = Path(str(path))
    return PreflightCheck(
        name=str(name),
        passed=bool(p.exists() and p.is_file()),
        detail=str(p),
    )


def _parent_writable_check(*, name: str, path: str) -> PreflightCheck:
    p = Path(str(path))
    parent = p.parent if str(p.parent) not in ("", ".") else Path(".")
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return PreflightCheck(
            name=str(name),
            passed=False,
            detail=f"{parent} err={e}",
        )
    return PreflightCheck(name=str(name), passed=True, detail=str(parent))


def _env_present_check(*, name: str, env: Mapping[str, str]) -> PreflightCheck:
    value = str(env.get(str(name), "")).strip()
    return PreflightCheck(
        name=f"env:{name}",
        passed=bool(value != ""),
        detail="present" if value != "" else "missing",
    )


def collect_preflight_checks(
    *,
    config_path: str,
    check_env: bool,
    env: Mapping[str, str] | None = None,
) -> tuple[object, list[PreflightCheck]]:
    cfg = load_app_config(str(config_path))
    env_map = dict(os.environ if env is None else env)
    checks = [
        _file_exists_check(name="abi_json_path", path=str(cfg.abi_json_path)),
        _parent_writable_check(name="latency_log_parent", path=str(cfg.latency_log_path)),
        _parent_writable_check(
            name="claim_scan_cursor_parent",
            path=str(cfg.runtime_state_paths.claim_scan_cursor_path),
        ),
        _parent_writable_check(
            name="dry_bets_parent",
            path=str(cfg.runtime_state_paths.dry_bets_path),
        ),
        _parent_writable_check(
            name="dry_settled_parent",
            path=str(cfg.runtime_state_paths.dry_settled_epochs_path),
        ),
        _parent_writable_check(
            name="dry_audit_parent",
            path=str(cfg.runtime_state_paths.dry_audit_trades_path),
        ),
        _parent_writable_check(
            name="dry_cycle_audit_parent",
            path=str(cfg.runtime_state_paths.dry_cycle_audit_path),
        ),
        _parent_writable_check(
            name="dry_bankroll_parent",
            path=str(cfg.runtime_state_paths.dry_bankroll_state_path),
        ),
        _parent_writable_check(
            name="dry_pipeline_snapshot_parent",
            path=str(cfg.runtime_state_paths.dry_pipeline_bootstrap_state_path),
        ),
        _parent_writable_check(
            name="live_pipeline_snapshot_parent",
            path=str(cfg.runtime_state_paths.live_pipeline_bootstrap_state_path),
        ),
        PreflightCheck(
            name="momentum_gate_enabled",
            passed=bool(cfg.momentum_gate.enabled),
            detail=f"symbol={cfg.momentum_gate.symbol} threshold={cfg.momentum_gate.threshold}",
        ),
    ]
    if bool(check_env):
        checks.extend(
            [
                _env_present_check(name="BSC_WALLET_PRIVATE_KEY", env=env_map),
            ]
        )
    return cfg, checks


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config.toml")
    p.add_argument(
        "--check-env",
        action="store_true",
        help="Also require THE_GRAPH_API_KEY and BSC_WALLET_PRIVATE_KEY to be present after loading .env.",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    if bool(args.check_env):
        load_env()
    cfg, checks = collect_preflight_checks(
        config_path=str(args.config),
        check_env=bool(args.check_env),
    )

    print(f"CONFIG={args.config}")
    print(f"ACCOUNTING_GAS_WEI={int(GAS_PRICE_WEI)}")
    print(f"MOMENTUM_GATE_ENABLED={cfg.momentum_gate.enabled}")
    print(f"MOMENTUM_GATE_SYMBOL={cfg.momentum_gate.symbol}")
    for check in checks:
        status = "PASS" if bool(check.passed) else "FAIL"
        print(f"{status} {check.name}: {check.detail}")

    failed = [c for c in checks if not bool(c.passed)]
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
