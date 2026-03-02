"""Legacy overlay package used only by inspection/legacy tooling.

This package is intentionally legacy-first and extends its module search path
with the canonical package so missing modules can fall back safely.
"""

from __future__ import annotations

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)