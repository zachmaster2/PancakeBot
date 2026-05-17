"""Fixed-width structured logger with info/warn/error levels and typed field formatting."""
from __future__ import annotations

import datetime
import logging
import math
import numbers
import os
import sys
from logging.handlers import RotatingFileHandler
from typing import Any

from pancakebot.util import InvariantError

# Logging goals:
# - Fixed-width columns for readability.
# - Keep logs concise and stable (no debug mode).
# - INFO for normal progress; WARN for retry attempts; ERROR for transient failures.

_SYS_NAME_W: int = 8
_SUB_W: int = 6
# Event column width (left-padded, hard-enforced by ``_emit``). Sized
# to fit the longest event name in the codebase (currently
# ``ROTATE_FAIL`` and ``IMPORT_FAIL`` at 11 chars). Convention:
# event names describe what HAPPENED (an action / outcome) and should
# leverage the SUB column to avoid redundancy -- e.g. SUB=KLINE +
# EVENT=PARTIAL reads cleaner than SUB=KLINE + EVENT=KLINE_PARTIAL.
# If a future event name exceeds 11, ``InvariantError`` fires at emit
# time so the dev can either shorten the name or bump this width.
_EVENT_W: int = 11

# Bundle 5 2026-05-14: a RotatingFileHandler-backed Python ``logging``
# sink mirrors every ``_emit`` call into ``var/{mode}/runtime.log``. The
# stdout writer is preserved; the file sink is purely additive.
#
# ``_FILE_LOGGER`` is the ``pancakebot`` namespaced logger that is
# attached to the file handler in ``configure_file_logging``. We mirror
# log lines to it via ``_FILE_LOGGER.log(level, line)`` inside
# ``_emit``. If the file sink hasn't been configured (backtest, sync,
# unit-test contexts) the logger has no handlers and the call is a
# no-op (propagate is also disabled to avoid duplicating to any caller-
# installed root handler).
_FILE_LOGGER: logging.Logger = logging.getLogger("pancakebot")
_FILE_LOGGER.propagate = False
_FILE_LOGGER.setLevel(logging.DEBUG)

# Level mapping: our "INFO"/"WARN"/"ERROR" strings → logging module ints.
# Bundle 5 logging levels stay coarse — we don't yet emit DEBUG from the
# structured logger, but configure the file handler at DEBUG so future
# debug events (e.g. from third-party libs that log to the pancakebot
# namespace) are captured without a config change.
_LEVEL_TO_LOGGING: dict[str, int] = {
    "INFO": logging.INFO,
    "WARN": logging.WARNING,
    "ERROR": logging.ERROR,
}


def configure_file_logging(log_path: str) -> RotatingFileHandler:
    """Attach a RotatingFileHandler to the pancakebot namespace logger.

    Called from ``pancakebot.app.run_from_config`` at startup of dry and
    live modes; backtest/sync runs do not configure the sink and the
    stdout writer is the sole output. Returns the handler so callers can
    keep a reference for graceful flush/close on shutdown (the bot
    relies on Python's interpreter-exit flush, but tests close
    explicitly to avoid file-handle leaks).

    Idempotent: re-calling with the same path is a no-op (the handler
    is recognized by its ``baseFilename``). Re-calling with a different
    path removes prior pancakebot-attached file handlers first.

    File rotation: 25MB threshold, 7 backups, parent dir auto-created.
    Format: ``HH:MM:SS.mmm message``. The format intentionally drops
    %(levelname)s and %(name)s because the rendered ``_emit`` line
    (passed as %(message)s) already carries level + sys_name columns;
    duplicating them would (a) waste bytes per line and (b) cause a
    level-string drift between stdout (``WARN``) and file
    (Python's ``WARNING``). The handler's own ``asctime`` captures
    the moment of write — ms-difference vs ``_emit``'s wall-clock
    capture at worst.

    Relative ``log_path`` is resolved against the repo root anchor
    (the directory that contains the ``pancakebot/`` package), NOT
    ``os.getcwd()``. This prevents log-file misplacement when an
    operator launches the bot from a non-repo-root working directory
    (the supervisor + Windows scheduled task already use repo root,
    but ad-hoc operator launches drift if we resolve via cwd).
    Absolute paths pass through unchanged.
    """
    if os.path.isabs(log_path):
        target = log_path
    else:
        # Repo root = parent of the pancakebot/ package dir =
        # parent of this file's parent. log.py lives at
        # ``<repo>/pancakebot/log.py`` → parent.parent is ``<repo>``.
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        target = os.path.join(repo_root, log_path)
    target = os.path.abspath(target)
    parent = os.path.dirname(target)
    if parent:
        os.makedirs(parent, exist_ok=True)

    # Idempotency: skip if a handler for this path is already attached.
    for h in list(_FILE_LOGGER.handlers):
        if isinstance(h, RotatingFileHandler) and os.path.abspath(h.baseFilename) == target:
            return h
        # Different path → detach the old one to avoid dual file outputs.
        if isinstance(h, RotatingFileHandler):
            _FILE_LOGGER.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass

    handler = RotatingFileHandler(
        target,
        maxBytes=25 * 1024 * 1024,
        backupCount=7,
        encoding="utf-8",
    )
    handler.setLevel(logging.DEBUG)
    # NB (reviewer flag, Bundle 5): we deliberately drop %(levelname)s
    # from the file format because the rendered ``_emit`` line (passed
    # in as %(message)s) ALREADY includes the level column ("INFO",
    # "WARN", "ERROR" — note "WARN", not Python's "WARNING"). Including
    # %(levelname)s would cause a level-string drift: stdout shows
    # "WARN" while file shows "WARNING", breaking operator-side
    # ``\bWARN\b`` grep on the file. The handler's level filter still
    # applies via setLevel(DEBUG).
    formatter = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)
    _FILE_LOGGER.addHandler(handler)
    return handler


def _ts_hundredths() -> str:
    # Local time; hundredths of a second.
    now = datetime.datetime.now()
    base = now.strftime("%Y-%m-%d %H:%M:%S")
    hundredths = int(now.microsecond / 10_000)  # 0..99
    return f"{base}.{hundredths:02d}"


def _fmt_float_by_key(key: str, x: float) -> str:
    # Consistent, human-readable numeric formatting.
    # Rules (locked):
    # - *_bnb               -> 4 decimals (Pancake UI style)
    # - *probability*       -> 5 decimals
    # - *_fraction          -> 4 decimals
    # - *_multiple          -> 4 decimals
    # - default floats      -> 4 decimals
    if not math.isfinite(x):
        return "inf" if x > 0 else "-inf" if x < 0 else "nan"

    k = key.lower()
    if k.endswith("_bnb"):
        return f"{x:.4f}"
    if "probability" in k:
        return f"{x:.5f}"
    if k.endswith("_fraction"):
        return f"{x:.4f}"
    if k.endswith("_multiple"):
        return f"{x:.4f}"
    return f"{x:.4f}"


def _fmt_value(key: str, v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    if isinstance(v, numbers.Real) and not isinstance(v, (bool, int)):
        return _fmt_float_by_key(key, float(v))
    return str(v)


def _shorten_long(k: str, v: str) -> str:
    if k == "tx" and len(v) > 24:
        return f"{v[:8]}...{v[-8:]}"
    return v


def _fmt_fields(fields: dict[str, Any]) -> str:
    if not fields:
        return ""
    parts: list[str] = []
    for k, v in fields.items():
        val = _shorten_long(k, _fmt_value(k, v))
        parts.append(f"{k}={val}")
    return " " + " ".join(parts)


def _emit(level: str, sys_name: str, sub: str, event: str, *, msg: str | None = None, **fields: Any) -> None:
    ts = _ts_hundredths()

    if len(sys_name) > _SYS_NAME_W:
        raise InvariantError("log_label_too_long_sys")
    if len(sub) > _SUB_W:
        raise InvariantError("log_label_too_long_sub")
    if len(event) > _EVENT_W:
        raise InvariantError("log_label_too_long_event")

    # Fixed-width columns for readability (spaces only; no tabs).
    tail = ""
    if msg is not None and msg != "":
        tail = " " + str(msg)
    elif fields:
        tail = _fmt_fields(fields)

    line = (
        f"{ts}  "
        f"{level:<5} "
        f"{sys_name:<{_SYS_NAME_W}} "
        f"{sub:<{_SUB_W}} "
        f"{event:<{_EVENT_W}}"
        f"{tail}"
    )

    rendered = line.rstrip()
    sys.stdout.write(rendered + "\n")
    sys.stdout.flush()

    # Bundle 5 2026-05-14: dual-write into the namespaced
    # ``pancakebot`` logger so a RotatingFileHandler (configured at
    # startup by ``configure_file_logging``) can persist every line to
    # ``var/{mode}/runtime.log``. When no handler is attached (backtest,
    # sync, unit-test contexts) ``Logger.log`` is a near-no-op and the
    # stdout path is unaffected. Wrap in try/except so any logging-
    # subsystem failure does NOT propagate up and abort the caller —
    # the stdout write above already succeeded and is the source of
    # truth; the file mirror is a strictly additive sink.
    try:
        _FILE_LOGGER.log(_LEVEL_TO_LOGGING.get(level, logging.INFO), rendered)
    except Exception:
        pass


def info(sys_name: str, sub: str, event: str, *, msg: str | None = None, **fields: Any) -> None:
    _emit("INFO", sys_name, sub, event, msg=msg, **fields)


def warn(sys_name: str, sub: str, event: str, *, msg: str | None = None, **fields: Any) -> None:
    _emit("WARN", sys_name, sub, event, msg=msg, **fields)


def error(sys_name: str, sub: str, event: str, *, msg: str | None = None, **fields: Any) -> None:
    _emit("ERROR", sys_name, sub, event, msg=msg, **fields)
