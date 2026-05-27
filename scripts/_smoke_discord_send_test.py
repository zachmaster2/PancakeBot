"""One-shot test-message poster for the 3 PancakeBot Discord webhooks.

Reads the three Machine-level env vars the supervisor uses and POSTs a
distinguishable test payload to each. Run manually once after setting
up the webhooks; user confirms each channel receives its test message
before we flip schtasks to --alert.

Usage:
    python scripts/_smoke_discord_send_test.py

Exit codes:
    0 - all three sent (HTTP 2xx each)
    1 - one or more failed (details printed)
    2 - one or more env vars missing (details printed)
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone


CHANNELS = [
    {
        "label": "DRY_ALERTS",
        "env": "PANCAKEBOT_DRY_ALERTS_DISCORD_WEBHOOK_URL",
        "username": "PancakeBot-dry",
        "emoji": ":herb:",
        "position": "Test 1/3",
        "expected_channel": "pancakebot-dry-alerts",
        "purpose": "dry-mode alerts (CRASHED / DOWN + escalations)",
    },
    {
        "label": "LIVE_ALERTS",
        "env": "PANCAKEBOT_LIVE_ALERTS_DISCORD_WEBHOOK_URL",
        "username": "PancakeBot-live",
        "emoji": ":money_with_wings:",
        "position": "Test 2/3",
        "expected_channel": "pancakebot-live-alerts",
        "purpose": "live-mode alerts (CRASHED / DOWN + escalations)",
    },
    {
        "label": "GENERAL",
        "env": "PANCAKEBOT_GENERAL_DISCORD_WEBHOOK_URL",
        "username": "PancakeBot-general",
        "emoji": ":pancakes:",
        "position": "Test 3/3",
        "expected_channel": "pancakebot-general",
        "purpose": "UNINSTRUMENTED + supervisor-self errors",
    },
]


def main() -> int:
    import requests  # in requirements.txt

    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    missing = [c for c in CHANNELS if not os.environ.get(c["env"], "").strip()]
    if missing:
        for c in missing:
            print(f"[MISSING] env var not set: {c['env']}")
        print("\nHint: `setx /M <NAME> \"<URL>\"` in an elevated PowerShell,")
        print("then restart the shell so the current process inherits the new env.")
        # Keep going for any channels that ARE set -- partial coverage is
        # still informative and we don't want to hide a send failure behind
        # an unrelated missing var.

    failures = []
    for c in CHANNELS:
        webhook = os.environ.get(c["env"], "").strip()
        if not webhook:
            continue
        payload = {
            "content": (
                f"{c['emoji']} **{c['position']} -- Webhook test for `{c['label']}`** at `{now_iso}`\n"
                f"Expected channel: `#{c['expected_channel']}`\n"
                f"Purpose: {c['purpose']}\n"
                f"Env var: `{c['env']}`\n"
                f"\n"
                f"If you see this, the wiring for this channel is correct."
            ),
            "username": c["username"],
        }
        try:
            r = requests.post(webhook, json=payload, timeout=10)
        except Exception as e:
            print(f"[FAIL] {c['label']:12}  post_exception: {type(e).__name__}: {e}")
            failures.append(c["label"])
            continue
        if 200 <= r.status_code < 300:
            print(f"[SENT] {c['label']:12}  HTTP {r.status_code}  (check channel)")
        else:
            body = (r.text or "")[:200]
            print(f"[FAIL] {c['label']:12}  HTTP {r.status_code}  body={body}")
            failures.append(c["label"])

    if missing:
        return 2
    if failures:
        return 1
    print("\nAll three sent. Confirm receipt in Discord, then proceed to schtasks /change.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
