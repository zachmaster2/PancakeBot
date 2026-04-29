"""Detect other dry/live bot instances to prevent duplicate-process clashes."""
import os
import os.path

_MODE_TOKENS = frozenset(("--dry", "--live"))
_SCRIPT_NAME = "run.py"


def _is_run_py_token(token: str) -> bool:
    """True iff *token* refers to ``run.py`` as a standalone token or the
    basename of a path token. Handles ``run.py``, ``./run.py``,
    ``C:\\path\\run.py``, ``/abs/path/run.py``."""
    if token == _SCRIPT_NAME:
        return True
    # Basename strip handles both POSIX and Windows path separators because
    # ``os.path.basename`` is platform-aware. Cross-platform safety net:
    # also try the alternate separator since psutil cmdlines may carry
    # POSIX paths on Windows or vice versa.
    if os.path.basename(token) == _SCRIPT_NAME:
        return True
    if token.replace("\\", "/").rsplit("/", 1)[-1] == _SCRIPT_NAME:
        return True
    return False


def _cmdline_is_bot(cmdline: list[str]) -> bool:
    """True iff *cmdline* (psutil's argv list) is a PancakeBot dry/live invocation.

    Two independent token checks (both must pass):
      - Some token IS ``run.py`` (or a path ending in ``/run.py``).
      - Some token IS exactly ``--dry`` or ``--live``.

    Token-based matching protects against argument-order bugs (the prior
    substring check ``"run.py --dry"`` failed when ``--config <path>`` was
    interposed) and against false positives like ``--config dry_run.toml``
    (where ``--dry`` only appears as a substring of an unrelated arg).
    """
    if not cmdline:
        return False
    has_run_py = any(_is_run_py_token(t) for t in cmdline)
    if not has_run_py:
        return False
    has_mode = any(t in _MODE_TOKENS for t in cmdline)
    return has_mode


def _ancestor_pids(self_pid: int) -> set[int]:
    """Return the set of ancestor PIDs (parents up the chain) for ``self_pid``.

    When the bot is launched via ``py run.py --dry``, the ``py.exe`` launcher
    becomes the parent and ``python.exe`` the child.  Both processes share
    the same cmdline (``run.py --dry``), so a naive duplicate scan would
    flag the launcher as a duplicate of its own child.  Walking the parent
    chain and excluding every ancestor is the most robust fix -- it handles
    arbitrary launcher chains (``py.exe`` -> ``python.exe``,
    ``cmd.exe`` -> ``py.exe`` -> ``python.exe``, etc.) without requiring
    a hard-coded launcher name list.

    Walks until: a parent is None (root), an exception fires (process
    already gone or access denied), or 16 hops are walked (safety against
    pathological cycles).
    """
    import psutil
    ancestors: set[int] = set()
    try:
        proc = psutil.Process(self_pid)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return ancestors
    for _ in range(16):
        try:
            parent = proc.parent()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            break
        if parent is None:
            break
        ancestors.add(parent.pid)
        proc = parent
    return ancestors


def find_duplicate_bots():
    """Return list of {pid, cmdline, started_at} for other python processes
    running ``run.py --dry`` or ``run.py --live``, excluding self and the
    self process's ancestors.  Uses psutil.

    Both ``--dry`` and ``--live`` count as duplicates regardless of the
    caller's own mode -- they share dry/live state file paths in
    ``var/`` (bankroll, heartbeat, captures) and would clash on writes.

    Ancestor PIDs are excluded because the ``py.exe`` launcher (or any
    similar wrapper) becomes the parent of the actual ``python.exe``
    interpreter, and both processes carry the same cmdline -- without
    ancestor exclusion, the launcher would be flagged as a duplicate
    of its own child.  See ``_ancestor_pids`` for the walk strategy.
    """
    import psutil
    self_pid = os.getpid()
    excluded = {self_pid} | _ancestor_pids(self_pid)
    results = []
    for proc in psutil.process_iter(["pid", "cmdline", "create_time"]):
        try:
            if proc.info["pid"] in excluded:
                continue
            cmdline = proc.info.get("cmdline") or []
            if not _cmdline_is_bot(cmdline):
                continue
            results.append({
                "pid": proc.info["pid"],
                "cmdline": " ".join(cmdline),
                "started_at": proc.info.get("create_time"),
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return results
