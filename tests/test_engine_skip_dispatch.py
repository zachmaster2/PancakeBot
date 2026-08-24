"""The engine's skip-reason dispatch must agree with the skip-context schema.

Every branch reads `decision.skip_context` by direct subscript and the
schema is exact-match enforced in StrategyPipelineDecision.__post_init__,
so a branch that reads a key its own reason does not carry raises KeyError
on the pre-lock path — where engine.py documents that there is NO
top-level catch, so it takes the process down into a systemd restart.

That is not hypothetical. Adding `risk_worst_case_exposure` ABOVE the
breaker branch relocated the D3 COOLDOWN ENTERED block into the new branch,
which (a) silenced the suspension alert entirely, (b) made a merely
declined round announce itself as a suspension, and (c) read
`_ctx["drawdown_pct"]` from a context that does not have it. The whole
suite was green because all 30 assertions on that change were at the
pipeline level and nothing drove the engine branch.

These tests read the dispatch structurally, so they hold under reflow and
do not need an engine harness.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pancakebot.runtime import engine  # noqa: E402
from pancakebot.strategy.base import _SKIP_CONTEXT_SCHEMA  # noqa: E402


def _dispatch_branches():
    """{reason: If-node} for every `reason == "..."` branch in the round
    loop's skip dispatch."""
    tree = ast.parse(Path(engine.__file__).read_text(encoding="utf-8"))
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
               and n.name == "_run_one_iteration"), None)
    assert fn is not None, "engine._run_one_iteration not found"
    def reason_of(test) -> str | None:
        """The reason a branch test selects, for ANY shape that contains a
        `reason == "..."` comparison -- a bare Compare, or one inside a
        BoolOp such as
        `reason == "kline_fetch_transient_failure" and gate is not None`.
        Matching only the bare Compare made that branch invisible, and an
        invisible branch is exempt from every check in this file."""
        for n in ast.walk(test):
            if (isinstance(n, ast.Compare)
                    and isinstance(n.left, ast.Name) and n.left.id == "reason"
                    and len(n.ops) == 1 and isinstance(n.ops[0], ast.Eq)
                    and isinstance(n.comparators[0], ast.Constant)
                    and isinstance(n.comparators[0].value, str)):
                return n.comparators[0].value
        return None

    out: dict[str, ast.If] = {}
    for node in ast.walk(fn):
        if not isinstance(node, ast.If):
            continue
        reason = reason_of(node.test)
        if reason is not None:
            out[reason] = node
    assert out, "no `reason == ...` branches found — dispatch was restructured"

    # COMPLETENESS. The matcher above only understands a bare
    # `reason == "..."` Compare, so ANY other shape is invisible to it and
    # every assertion in this file passes vacuously for that branch. One
    # such branch already exists -- `if reason == "kline_fetch_transient_failure"
    # and gate is not None:` is a BoolOp -- and injecting
    # `closed.in_cooldown = True` into it was proven to leave this file at
    # 8 passed.
    #
    # Rather than enumerate shapes (which only ever covers the ones someone
    # thought of), count every If in the round loop whose test mentions
    # `reason` at all and require the matcher to have understood all of
    # them. A shape it cannot parse now fails LOUDLY here instead of
    # silently exempting itself from the checks below.
    # Keyed on the NAME `reason`, not the substring: `_cd_reason` and
    # `ready_reason` are different variables in this function and are not
    # part of the skip dispatch.
    def references_reason(test) -> bool:
        return any(isinstance(n, ast.Name) and n.id == "reason"
                   for n in ast.walk(test))

    mentions_reason = [
        n for n in ast.walk(fn)
        if isinstance(n, ast.If) and references_reason(n.test)
    ]
    unparsed = [ast.unparse(n.test) for n in mentions_reason
                if n not in out.values()]
    assert not unparsed, (
        f"{len(unparsed)} branch test(s) mention `reason` but this parser "
        f"does not understand them, so they are exempt from every check in "
        f"this file: {unparsed}. Extend _dispatch_branches rather than "
        f"deleting this assertion.")
    return out


def _body_nodes(node: ast.If):
    """Walk ONLY this branch's own body.

    `elif` is a nested If inside `orelse`, so walking the If node itself
    would sweep in every later branch of the chain and make each branch
    look like it reads every other branch's keys."""
    for stmt in node.body:
        yield from ast.walk(stmt)


def _ctx_keys(node: ast.If) -> set[str]:
    """Every `_ctx["key"]` read directly inside this branch."""
    return {
        n.slice.value
        for n in _body_nodes(node)
        if (isinstance(n, ast.Subscript)
            and isinstance(n.value, ast.Name) and n.value.id == "_ctx"
            and isinstance(n.slice, ast.Constant)
            and isinstance(n.slice.value, str))
    }


def test_the_dispatch_covers_the_reasons_that_require_context():
    branches = _dispatch_branches()
    for reason in _SKIP_CONTEXT_SCHEMA:
        assert reason in branches, (
            f"{reason} requires skip_context but the engine has no branch "
            f"for it — its context would never be rendered")


@pytest.mark.parametrize("reason", sorted(_SKIP_CONTEXT_SCHEMA))
def test_each_branch_only_reads_keys_its_own_reason_carries(reason):
    """THE regression. `risk_worst_case_exposure` reading
    `_ctx["drawdown_pct"]` is a KeyError on the pre-lock path, and there is
    no top-level catch."""
    branch = _dispatch_branches()[reason]
    allowed = set(_SKIP_CONTEXT_SCHEMA[reason])
    read = _ctx_keys(branch)
    extra = read - allowed
    assert not extra, (
        f"engine branch for {reason} reads {sorted(extra)} but that reason's "
        f"schema is {sorted(allowed)} — this raises KeyError on the pre-lock "
        f"path and crashes the round loop")


def test_the_cooldown_alert_belongs_to_the_breaker_and_only_the_breaker():
    """(a) and (b) together. A genuine ~24h suspension MUST alert — the
    absence of that alert is why the 2026-08-24 trip was found a day late,
    from a summary. And a merely declined round must NOT claim to be one."""
    branches = _dispatch_branches()

    def calls(node):
        return {n.func.id for n in _body_nodes(node)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}

    breaker = branches["risk_drawdown_breaker_fired"]
    assert "send_cooldown_entered_alert" in calls(breaker), (
        "the breaker branch no longer sends COOLDOWN ENTERED — a real "
        "suspension would produce a log line and nothing else")
    for reason, node in branches.items():
        if reason == "risk_drawdown_breaker_fired":
            continue
        assert "send_cooldown_entered_alert" not in calls(node), (
            f"{reason} sends COOLDOWN ENTERED but is not a suspension")


def test_only_the_suspension_paths_set_in_cooldown():
    """`closed.in_cooldown = True` drives the later COOLDOWN LIFTED edge.
    A declined round setting it would emit a spurious LIFTED alert."""
    branches = _dispatch_branches()
    allowed = {"risk_drawdown_breaker_fired", "risk_cooldown_active"}

    def sets_in_cooldown(node):
        for n in _body_nodes(node):
            if isinstance(n, ast.Assign):
                for tgt in n.targets:
                    if (isinstance(tgt, ast.Attribute)
                            and tgt.attr == "in_cooldown"
                            and isinstance(n.value, ast.Constant)
                            and n.value.value is True):
                        return True
        return False

    for reason, node in branches.items():
        if reason in allowed:
            continue
        assert not sets_in_cooldown(node), (
            f"{reason} sets in_cooldown=True but is not a suspension — this "
            f"produces a spurious COOLDOWN LIFTED alert later")


def test_the_worst_case_branch_says_it_is_not_a_suspension():
    """Operator-facing wording: the whole point of the gate is that
    declining a round is not a stand-down."""
    src = Path(engine.__file__).read_text(encoding="utf-8")
    branch = src.split('elif reason == "risk_worst_case_exposure":', 1)[1]
    branch = branch.split("elif reason ==", 1)[0]
    assert "NOT suspended" in branch
    assert "worst-case exposure" in branch
