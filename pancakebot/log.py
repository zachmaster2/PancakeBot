"""Fixed-width structured logger with info/warn/error levels and typed field formatting."""
from __future__ import annotations

import datetime
import math
import numbers
import sys
from typing import Any

from pancakebot.util import InvariantError

# Logging goals:
# - Fixed-width columns for readability.
# - Keep logs concise and stable (no debug mode).
# - INFO for normal progress; WARN for retry attempts; ERROR for transient failures.

_SYS_NAME_W: int = 8
_SUB_W: int = 6
_EVENT_W: int = 11


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
        f"{sys_name:<8} "
        f"{sub:<6} "
        f"{event:<11}"
        f"{tail}"
    )

    sys.stdout.write(line.rstrip() + "\n")
    sys.stdout.flush()


def info(sys_name: str, sub: str, event: str, *, msg: str | None = None, **fields: Any) -> None:
    _emit("INFO", sys_name, sub, event, msg=msg, **fields)


def warn(sys_name: str, sub: str, event: str, *, msg: str | None = None, **fields: Any) -> None:
    _emit("WARN", sys_name, sub, event, msg=msg, **fields)


def error(sys_name: str, sub: str, event: str, *, msg: str | None = None, **fields: Any) -> None:
    _emit("ERROR", sys_name, sub, event, msg=msg, **fields)
