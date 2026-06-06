"""Lint: every info/warn/error ACTION literal must fit _ACTION_W.

The logger raises InvariantError('log_action_too_long') at runtime when an
action exceeds the width — which crash-loops whatever round path emits it. This
static scan catches an over-length action at test time instead of in production.

(Near-miss 2026-06-06: ``info("PREFLIGHT", ...)`` — 9 chars vs _ACTION_W=8 —
crash-looped the dry round loop; caught in dry before reaching live.)
"""
from __future__ import annotations

import re
from pathlib import Path

from pancakebot.log import _ACTION_W

_PKG = Path(__file__).resolve().parent.parent / "pancakebot"
# Bare info/warn/error("ACTION", ...) — the negative lookbehind excludes
# attribute calls like logging's ``logger.info(...)`` (no _ACTION_W constraint).
_CALL = re.compile(r'(?<![.\w])(?:info|warn|error)\(\s*"([A-Za-z0-9_]+)"')


def test_all_log_actions_fit_action_width():
    offenders = []
    for py in _PKG.rglob("*.py"):
        text = py.read_text(encoding="utf-8", errors="replace")
        for m in _CALL.finditer(text):
            action = m.group(1)
            if len(action) > _ACTION_W:
                offenders.append(
                    f"{py.relative_to(_PKG.parent)}: {action!r} "
                    f"({len(action)} > _ACTION_W={_ACTION_W})"
                )
    assert not offenders, "log actions exceeding _ACTION_W:\n" + "\n".join(offenders)
