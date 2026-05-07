"""WSS forensic diagnostics: raw-frame logger + system-stats poller.

Background instrumentation for diagnosing WSS silent-stall scenarios
(see ``var/incident_reports/2026_05_06_wss_silent_stall_root_cause.md``
and ``var/incident_reports/2026_05_06_phase0_spike_results.md``).

Two independent components, both opt-in via constructor on
``PoolEventWatcher``:

1. ``RawWssLogger`` — daily-rolling file writer that records every
   WSS frame the recv loop receives (timestamp + length + truncated
   payload). Lets us reconstruct exactly what the upstream sent vs.
   what we processed when post-mortem analysis is needed.

2. ``SystemStatsPoller`` — background daemon thread polling Windows
   NIC adapter statistics and active-connection counts every 60s.
   Detects packet-level errors (``ReceivedDiscardedPackets``,
   ``ReceivedErrors``) and logs deltas between polls. Output goes to
   a separate file so the main log stays readable.

Both components write to ``var/dry/logs/`` by default. Files are
ASCII text (line-delimited, one frame or stat-snapshot per line) for
easy ``grep``/``awk`` analysis.

Failure mode: if a diagnostic component fails (e.g., disk full,
PowerShell unavailable), it MUST NOT take down the bot. Each
component swallows-and-logs at the WARN level; the recv loop and
engine continue regardless.
"""
from __future__ import annotations

import datetime
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

from pancakebot.log import info, warn


_DEFAULT_LOG_DIR = Path("var/dry/logs")
_RAW_FRAME_TRUNCATE_BYTES = 8192  # truncate payloads larger than this
_SYSTEM_STATS_POLL_INTERVAL_S = 60.0


class RawWssLogger:
    """Daily-rolled raw WSS frame logger.

    Each frame (the JSON string the recv loop pulled from
    ``await ws.recv()``, before parsing) is written as one line:

        {ISO_timestamp}  {kind}  len={N}  {payload_or_truncated}

    where ``kind`` is ``"recv"`` for received frames or ``"meta"`` for
    bookkeeping events (session start/end, subscription confirms).

    The logger handles its own file lifecycle: opens lazily, rolls at
    midnight (local time), flushes after every write so a process
    crash loses at most the last in-flight write.

    Thread-safe: a single ``threading.Lock`` serializes writes. All
    log_frame calls from the recv loop go through this lock; lock
    contention is negligible because writes are short and the recv
    loop is the only writer in steady state.
    """

    def __init__(self, log_dir: Path = _DEFAULT_LOG_DIR) -> None:
        self._log_dir: Path = Path(log_dir)
        self._lock = threading.Lock()
        self._current_date: str = ""
        self._fp: Optional[object] = None
        self._frame_count: int = 0
        self._dropped_count: int = 0
        try:
            self._log_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            warn("DIAG", "RAW", "MKDIR_FAIL",
                 msg=f"{type(e).__name__}: {e}; raw-frame logging disabled")

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def _ensure_open_locked(self) -> None:
        """Caller must hold ``self._lock``. Opens or rolls the log file
        based on today's date."""
        today = datetime.date.today().isoformat()
        if today == self._current_date and self._fp is not None:
            return
        # Roll: close prior, open new.
        if self._fp is not None:
            try:
                self._fp.close()  # type: ignore[union-attr]
            except OSError:
                pass
            self._fp = None
        path = self._log_dir / f"wss_raw_{today}.log"
        try:
            self._fp = open(path, "a", encoding="utf-8")
            self._current_date = today
        except OSError as e:
            warn("DIAG", "RAW", "OPEN_FAIL",
                 msg=f"{path}: {type(e).__name__}: {e}; "
                     f"raw-frame logging disabled this turn")
            self._dropped_count += 1
            self._fp = None

    def log_frame(self, kind: str, length: int, payload: str | None = None) -> None:
        """Log a single frame. Errors swallowed-and-counted; never
        propagate."""
        with self._lock:
            self._ensure_open_locked()
            if self._fp is None:
                self._dropped_count += 1
                return
            ts = datetime.datetime.now().isoformat(timespec="milliseconds")
            if payload is None:
                line = f"{ts}  {kind}  len={length}\n"
            else:
                if length > _RAW_FRAME_TRUNCATE_BYTES:
                    payload = (
                        payload[:_RAW_FRAME_TRUNCATE_BYTES]
                        + f"...<truncated {length - _RAW_FRAME_TRUNCATE_BYTES} bytes>"
                    )
                line = f"{ts}  {kind}  len={length}  {payload}\n"
            try:
                self._fp.write(line)  # type: ignore[union-attr]
                self._fp.flush()  # type: ignore[union-attr]
                self._frame_count += 1
            except OSError as e:
                warn("DIAG", "RAW", "WRITE_FAIL",
                     msg=f"{type(e).__name__}: {e}")
                self._dropped_count += 1

    def close(self) -> None:
        with self._lock:
            if self._fp is not None:
                try:
                    self._fp.close()  # type: ignore[union-attr]
                except OSError:
                    pass
                self._fp = None


class SystemStatsPoller:
    """Daemon thread polling Windows NIC + TCP-connection stats.

    Every ``poll_interval_s`` seconds:
      - Run ``Get-NetAdapterStatistics`` (PowerShell) and capture
        per-adapter received/sent bytes + error counts.
      - Run ``Get-NetTCPConnection`` (PowerShell) and count
        established connections to the WSS endpoint hostnames.
      - Compute deltas vs prior poll; log to a daily-rolled file.
      - WARN-log if any error counter incremented since last poll.

    All errors are swallowed and logged. The poller never crashes the
    parent thread.

    Linux/Mac fallback: PowerShell is Windows-only; on other platforms
    the poller logs a one-shot "skipped: not Windows" line and exits.
    """

    def __init__(
        self,
        log_dir: Path = _DEFAULT_LOG_DIR,
        poll_interval_s: float = _SYSTEM_STATS_POLL_INTERVAL_S,
    ) -> None:
        self._log_dir: Path = Path(log_dir)
        self._poll_interval_s = poll_interval_s
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._prior_nic_stats: dict[str, dict[str, int]] = {}
        self._poll_count: int = 0
        try:
            self._log_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            warn("DIAG", "SYS", "MKDIR_FAIL",
                 msg=f"{type(e).__name__}: {e}; system-stats poll disabled")

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        if os.name != "nt":
            info("DIAG", "SYS", "SKIP_NONWIN",
                 msg=f"system-stats poller is Windows-only (os.name={os.name})")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="wss-diag-stats",
        )
        self._thread.start()
        info("DIAG", "SYS", "START",
             msg=f"system-stats poller started "
                 f"(interval={self._poll_interval_s:.0f}s, dir={self._log_dir})")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._poll_once()
            except Exception as e:  # noqa: BLE001
                warn("DIAG", "SYS", "POLL_ERR",
                     msg=f"{type(e).__name__}: {e}")
            if self._stop_event.wait(timeout=self._poll_interval_s):
                break

    def _poll_once(self) -> None:
        self._poll_count += 1
        ts = datetime.datetime.now().isoformat(timespec="milliseconds")
        today = datetime.date.today().isoformat()
        path = self._log_dir / f"wss_sys_{today}.log"
        nic_lines = self._fetch_nic_stats()
        tcp_lines = self._fetch_tcp_connection_count()
        try:
            with open(path, "a", encoding="utf-8") as fp:
                fp.write(f"{ts}  poll#{self._poll_count}\n")
                for line in nic_lines:
                    fp.write(f"{ts}  NIC  {line}\n")
                for line in tcp_lines:
                    fp.write(f"{ts}  TCP  {line}\n")
        except OSError as e:
            warn("DIAG", "SYS", "WRITE_FAIL",
                 msg=f"{path}: {type(e).__name__}: {e}")

    def _run_powershell(self, command: str, timeout_s: float = 10.0) -> str:
        """Run a PowerShell one-liner. Returns stdout or empty on failure.
        Suppresses console window via CREATE_NO_WINDOW for pythonw.exe
        compatibility."""
        creationflags = 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW
        try:
            result = subprocess.run(
                ["powershell.exe", "-NonInteractive", "-NoProfile", "-Command", command],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                creationflags=creationflags,
            )
            if result.returncode != 0:
                return ""
            return result.stdout
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return ""

    def _fetch_nic_stats(self) -> list[str]:
        """Returns per-adapter NIC stats lines, one per adapter, including
        error-counter deltas vs prior poll."""
        cmd = (
            "Get-NetAdapterStatistics | "
            "Where-Object { $_.ReceivedBytes -gt 0 } | "
            "Select-Object Name, ReceivedBytes, ReceivedUnicastPackets, "
            "ReceivedDiscardedPackets, ReceivedPacketErrors, "
            "OutboundDiscardedPackets, OutboundPacketErrors | "
            "ConvertTo-Json -Compress"
        )
        out = self._run_powershell(cmd)
        if not out.strip():
            return ["nic_fetch_failed"]
        try:
            import json
            data = json.loads(out)
            if isinstance(data, dict):
                data = [data]
            elif not isinstance(data, list):
                return ["nic_parse_unexpected_shape"]
        except (ValueError, TypeError) as e:
            return [f"nic_parse_fail:{type(e).__name__}"]
        lines: list[str] = []
        for entry in data:
            try:
                name = str(entry.get("Name", "?"))
                rxb = int(entry.get("ReceivedBytes", 0))
                rxp = int(entry.get("ReceivedUnicastPackets", 0))
                rxd = int(entry.get("ReceivedDiscardedPackets", 0))
                rxe = int(entry.get("ReceivedPacketErrors", 0))
                txd = int(entry.get("OutboundDiscardedPackets", 0))
                txe = int(entry.get("OutboundPacketErrors", 0))
            except (ValueError, TypeError):
                continue
            prior = self._prior_nic_stats.get(name, {})
            delta_rxd = rxd - prior.get("rxd", rxd)
            delta_rxe = rxe - prior.get("rxe", rxe)
            delta_txd = txd - prior.get("txd", txd)
            delta_txe = txe - prior.get("txe", txe)
            self._prior_nic_stats[name] = {
                "rxb": rxb, "rxp": rxp, "rxd": rxd, "rxe": rxe,
                "txd": txd, "txe": txe,
            }
            line = (
                f"name={name!r} rxb={rxb} rxp={rxp} "
                f"rxd={rxd}(d{delta_rxd}) rxe={rxe}(d{delta_rxe}) "
                f"txd={txd}(d{delta_txd}) txe={txe}(d{delta_txe})"
            )
            lines.append(line)
            if delta_rxd > 0 or delta_rxe > 0 or delta_txd > 0 or delta_txe > 0:
                warn("DIAG", "SYS", "NIC_ERR_DELTA",
                     msg=f"adapter {name!r} new errors: "
                         f"rxd+{delta_rxd} rxe+{delta_rxe} "
                         f"txd+{delta_txd} txe+{delta_txe}")
        return lines

    def _fetch_tcp_connection_count(self) -> list[str]:
        """Counts established TCP connections grouped by remote port.
        Useful for verifying the WSS sockets are still established. The
        443/wss endpoints are remote port 443 by default."""
        cmd = (
            "Get-NetTCPConnection -State Established -ErrorAction SilentlyContinue | "
            "Group-Object -Property RemotePort | "
            "Select-Object Name, Count | "
            "ConvertTo-Json -Compress"
        )
        out = self._run_powershell(cmd)
        if not out.strip():
            return ["tcp_fetch_failed"]
        try:
            import json
            data = json.loads(out)
            if isinstance(data, dict):
                data = [data]
            elif not isinstance(data, list):
                return ["tcp_parse_unexpected_shape"]
        except (ValueError, TypeError) as e:
            return [f"tcp_parse_fail:{type(e).__name__}"]
        lines: list[str] = []
        for entry in data:
            try:
                port = str(entry.get("Name", "?"))
                count = int(entry.get("Count", 0))
            except (ValueError, TypeError):
                continue
            lines.append(f"established remote_port={port} count={count}")
        return lines
