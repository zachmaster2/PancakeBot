"""Smoke-test the Phase 2c+2d wiring's routing logic.

Direct function-level tests verify env-var routing without needing to fake
UNINSTRUMENTED end-to-end (which fights the live bot's per-second heartbeat).

End-to-end tests still use the local catcher for the paths where we can
force the classification (CRASHED by writing crash.json + removing hb, etc.)

Run with the env already set up by the caller -- this script doesn't manage
env vars. Expects:
    PANCAKEBOT_DRY_ALERTS_DISCORD_WEBHOOK_URL   -> set to catcher URL per test
    PANCAKEBOT_LIVE_ALERTS_DISCORD_WEBHOOK_URL  -> set to catcher URL per test
    PANCAKEBOT_GENERAL_DISCORD_WEBHOOK_URL      -> set to catcher URL per test
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts"))

from supervisor import (  # noqa: E402
    GENERAL_WEBHOOK_ENV,
    _env_var_for_mode,
    _resolve_webhook_env,
    _maybe_send_discord,
    _maybe_send_supervisor_error_alert,
    _artifacts_for_mode,
)


FAILED = []


def _eq(label: str, got, want) -> None:
    if got == want:
        print(f"  [OK] {label}: {got}")
    else:
        print(f"  [FAIL] {label}: got={got!r} want={want!r}")
        FAILED.append(label)


def test_direct_routing() -> None:
    print("=== direct routing tests ===")
    _eq("dry CRASHED",       _resolve_webhook_env("dry", "CRASHED"),       "PANCAKEBOT_DRY_ALERTS_DISCORD_WEBHOOK_URL")
    _eq("dry STALE",         _resolve_webhook_env("dry", "STALE"),         "PANCAKEBOT_DRY_ALERTS_DISCORD_WEBHOOK_URL")
    _eq("dry DOWN",          _resolve_webhook_env("dry", "DOWN"),          "PANCAKEBOT_DRY_ALERTS_DISCORD_WEBHOOK_URL")
    _eq("dry UNINSTRUMENTED",_resolve_webhook_env("dry", "UNINSTRUMENTED"),GENERAL_WEBHOOK_ENV)
    _eq("live CRASHED",      _resolve_webhook_env("live", "CRASHED"),      "PANCAKEBOT_LIVE_ALERTS_DISCORD_WEBHOOK_URL")
    _eq("live STALE",        _resolve_webhook_env("live", "STALE"),        "PANCAKEBOT_LIVE_ALERTS_DISCORD_WEBHOOK_URL")
    _eq("live UNINSTRUMENTED",_resolve_webhook_env("live", "UNINSTRUMENTED"),GENERAL_WEBHOOK_ENV)
    _eq("env for dry",       _env_var_for_mode("dry"),                     "PANCAKEBOT_DRY_ALERTS_DISCORD_WEBHOOK_URL")
    _eq("env for live",      _env_var_for_mode("live"),                    "PANCAKEBOT_LIVE_ALERTS_DISCORD_WEBHOOK_URL")


def test_end_to_end_uninstr_routes_general() -> None:
    """Fake UNINSTRUMENTED fields; verify _maybe_send_discord reads GENERAL env,
    not DRY_ALERTS env. Uses catcher URL in GENERAL to confirm which was chosen."""
    print("\n=== end-to-end: UNINSTRUMENTED goes to GENERAL ===")
    mode = "dry"
    art = _artifacts_for_mode(mode)
    fields = {"pid": 999999, "note": "synthetic"}
    outcome = _maybe_send_discord(
        mode=mode,
        status="UNINSTRUMENTED",
        fields=fields,
        art=art,
        escalation=None,
    )
    # If GENERAL is set to the catcher URL, outcome should be SENT.
    _eq("UNINSTRUMENTED outcome", outcome, "SENT")


def test_end_to_end_uninstr_disabled_when_general_unset() -> None:
    """Same as above but with GENERAL env unset -> DISABLED (proves the
    router is NOT falling back to DRY_ALERTS)."""
    print("\n=== end-to-end: UNINSTRUMENTED with GENERAL unset -> DISABLED ===")
    # Remove GENERAL from env; DRY_ALERTS may still be set from prior test.
    saved = os.environ.pop(GENERAL_WEBHOOK_ENV, None)
    try:
        mode = "dry"
        art = _artifacts_for_mode(mode)
        fields = {"pid": 999999, "note": "synthetic"}
        # Also need to reset the rate-limit state so the first attempt isn't
        # suppressed from the prior test.
        (art["last_alert"]).unlink(missing_ok=True)
        outcome = _maybe_send_discord(
            mode=mode,
            status="UNINSTRUMENTED",
            fields=fields,
            art=art,
            escalation=None,
        )
        _eq("UNINSTRUMENTED (GENERAL unset) outcome", outcome, "DISABLED")
    finally:
        if saved is not None:
            os.environ[GENERAL_WEBHOOK_ENV] = saved


def test_end_to_end_crashed_routes_dry_alerts() -> None:
    """Fake CRASHED fields; verify routing to DRY_ALERTS not GENERAL."""
    print("\n=== end-to-end: CRASHED goes to DRY_ALERTS ===")
    mode = "dry"
    art = _artifacts_for_mode(mode)
    # Write a fake crash.json so _build_discord_message has something to read.
    import json, time as _t
    art["crash"].parent.mkdir(parents=True, exist_ok=True)
    art["crash"].write_text(json.dumps({
        "ts_wall": _t.time(),
        "exc_type": "SyntheticError",
        "exc_repr": "SyntheticError('routing smoke')",
        "traceback_str": "Traceback:\n  File synthetic\n    raise SyntheticError\nSyntheticError: routing smoke",
        "last_epoch": 123456,
    }))
    # Reset rate-limit state to ensure this attempt isn't blocked.
    (art["last_alert"]).unlink(missing_ok=True)
    outcome = _maybe_send_discord(
        mode=mode,
        status="CRASHED",
        fields={"last_epoch": 123456, "exc": "SyntheticError"},
        art=art,
        escalation=None,
    )
    _eq("CRASHED outcome (expects DRY_ALERTS catcher hit)", outcome, "SENT")
    art["crash"].unlink(missing_ok=True)


def test_supervisor_error_goes_to_general() -> None:
    """Simulate a supervisor-itself exception; verify alert goes via GENERAL."""
    print("\n=== end-to-end: supervisor-self error goes to GENERAL ===")
    mode = "dry"
    art = _artifacts_for_mode(mode)
    # Reset rate-limit bucket for SUPERVISOR_ERROR.
    (art["last_alert"]).unlink(missing_ok=True)
    try:
        raise RuntimeError("synthetic classify failure for routing smoke")
    except RuntimeError as e:
        # _maybe_send_supervisor_error_alert swallows its own exceptions;
        # it has no return value. Verify indirectly via catcher receiving.
        _maybe_send_supervisor_error_alert(mode=mode, exc=e, art=art)
    print("  (check catcher captured message; routing verified if received)")


def main() -> int:
    test_direct_routing()
    if FAILED:
        print(f"\nDIRECT ROUTING FAILED: {FAILED}")
        return 2
    # The end-to-end tests depend on env vars the caller sets up.
    test_end_to_end_uninstr_routes_general()
    test_end_to_end_uninstr_disabled_when_general_unset()
    test_end_to_end_crashed_routes_dry_alerts()
    test_supervisor_error_goes_to_general()
    if FAILED:
        print(f"\nFAILED: {FAILED}")
        return 1
    print("\nALL ROUTING SMOKES PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
