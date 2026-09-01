"""A program that writes to BOTH stdout and stderr, then exits non-zero.

Used by test_wrapper_cannot_die_silently to prove that the wrappers survive
a stderr line even under $ErrorActionPreference = 'Stop'. Kept as a real
file rather than a python -c string so that no layer of PowerShell/shell
quoting can distort what is being tested -- the 2026-09-01 incident was
itself a quoting/stream-semantics problem, and a test for it should not
depend on getting quoting right.
"""
import sys

print("a")
sys.stderr.write("W\n")
sys.stderr.flush()
print("b")
sys.exit(5)
