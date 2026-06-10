"""Fixed-width structured logger with info/warn/error levels and a single ACTION column.

Phase B v2 (2026-05-18): collapsed the prior 3-column hierarchy
(SYSTEM/SUB/EVENT, each with its own width + naming convention) into a
single ACTION column. Operator-facing messages are now plain English
sentences passed positionally; there is no `**fields` kv rendering and
no `msg=` short-circuit. Callers pick an ACTION verb from the canonical
vocabulary (see ``docs/logging.md``) and compose the message as prose.
"""
from __future__ import annotations

import datetime
import logging
import os
import sys
from logging.handlers import RotatingFileHandler

from pancakebot.util import InvariantError

# Action column width (left-padded, hard-enforced by ``_emit``). Sized to
# fit the longest action verb in the canonical vocabulary (PROGRESS = 8,
# RECOVER = 7, REFUND = 6). Exceeding 8 chars raises ``InvariantError``
# at emit time so a typo or a stray verb-not-in-vocab is caught
# immediately rather than silently misaligning the column.
_ACTION_W: int = 8

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
    (passed as %(message)s) already carries the level column;
    duplicating them would (a) waste bytes per line and (b) cause a
    level-string drift between stdout (``WARN``) and file
    (Python's ``WARNING``). The handler's own ``asctime`` captures
    the moment of write — ms-difference vs ``_emit``'s wall-clock
    capture at worst.

    Relative ``log_path`` is resolved against the repo root anchor
    (the directory that contains the ``pancakebot/`` package), NOT
    ``os.getcwd()``. This prevents log-file misplacement when an
    operator launches the bot from a non-repo-root working directory
    (the supervisor's systemd unit already sets WorkingDirectory to the
    repo root, but ad-hoc operator launches drift if we resolve via cwd).
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


def _emit(level: str, action: str, message: str) -> None:
    """Render and emit a structured log line to stdout + file sink.

    ``action`` must be from the canonical vocabulary (see
    ``docs/logging.md``); ≤ ``_ACTION_W`` chars enforced here.
    ``message`` is the operator-facing English sentence — no kv
    rendering, no formatting helpers; callers compose prose directly.
    """
    if len(action) > _ACTION_W:
        raise InvariantError("log_action_too_long")

    ts = _ts_hundredths()
    line = (
        f"{ts}  "
        f"{level:<5}  "
        f"{action:<{_ACTION_W}}  "
        f"{message}"
    )

    rendered = line.rstrip()
    sys.stdout.write(rendered + "\n")
    sys.stdout.flush()

    # Bundle 5 2026-05-14: dual-write into the namespaced ``pancakebot``
    # logger. When no handler is attached (backtest, sync, unit-test
    # contexts) ``Logger.log`` is a near-no-op. Wrap in try/except so
    # any logging-subsystem failure does NOT abort the caller — the
    # stdout write above is the source of truth.
    try:
        _FILE_LOGGER.log(_LEVEL_TO_LOGGING.get(level, logging.INFO), rendered)
    except Exception:
        pass


def info(action: str, message: str) -> None:
    _emit("INFO", action, message)


def warn(action: str, message: str) -> None:
    _emit("WARN", action, message)


def error(action: str, message: str) -> None:
    _emit("ERROR", action, message)
