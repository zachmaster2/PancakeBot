"""Discord alert executor for bot lifecycle events.

Pure logic plus a single HTTP call (``requests.post``). Fired by
``pancakebot.ops.notify_lifecycle`` — the ``pancakebot-notify@`` oneshot
that the systemd bot units trigger on start/stop edges (docs/SUPERVISOR.md).

Channels:
  PANCAKEBOT_LIVE_ALERTS_DISCORD_WEBHOOK_URL  -> live mode events
  PANCAKEBOT_DRY_ALERTS_DISCORD_WEBHOOK_URL   -> dry mode events
  PANCAKEBOT_GENERAL_DISCORD_WEBHOOK_URL      -> cross-cutting kinds

Rate limit: one alert per (mode, kind) per 5 minutes.
Unset env var: silent fallback (no HTTP, no crash). HTTP failure: logged
to stderr, never raises.

The mode mutex routes MODE_TRANSITION (Live started, Dry stopped) to the
**live** channel since that's the actionable event. MODE_TRANSITION_REFUSED
(Dry refused to start because Live is up) routes to the **dry** channel
since dry-watchers care that their start attempt was vetoed.
"""
from __future__ import annotations

import datetime
import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants & routing tables
# ---------------------------------------------------------------------------

_ALERT_COOLDOWN_S: float = 300.0  # 5 min per (mode, kind)

LIVE_WEBHOOK_ENV = "PANCAKEBOT_LIVE_ALERTS_DISCORD_WEBHOOK_URL"
DRY_WEBHOOK_ENV = "PANCAKEBOT_DRY_ALERTS_DISCORD_WEBHOOK_URL"
GENERAL_WEBHOOK_ENV = "PANCAKEBOT_GENERAL_DISCORD_WEBHOOK_URL"

# Channel routing per notification kind. "mode" = use mode-specific webhook;
# "general" = use general channel; "live" / "dry" = always that channel
# regardless of the firing service's mode.
_CHANNEL_BY_KIND: dict[str, str] = {
    # Bot-health states.
    "CRASHED": "mode",
    "DOWN": "mode",
    "UNINSTRUMENTED": "general",
    # Crashloop limiter outcomes.
    "SUPPRESSED_FAST_CRASHLOOP": "mode",
    "SLOW_CRASHLOOP_WARNING": "mode",
    "SPAWN_FAILED": "mode",
    # New service-only states.
    "STARTED": "mode",
    "REBOOTED": "mode",
    "RECOVERY_AFTER_CRASH": "mode",
    "STOPPED": "mode",
    "MODE_TRANSITION": "live",  # Live started, Dry was stopped
    "MODE_TRANSITION_REFUSED": "dry",  # Dry refused, Live is up
    "SERVICE_CRASHED": "general",
    # Pool-gate alarm (raised in-process by the runtime, not by the
    # systemd notifier): the bot is up and enabled but has been unable
    # to trade for a run of rounds. Mode channel, alongside CRASHED/DOWN.
    "POOL_GATE_BLOCKED": "mode",
    "POOL_GATE_RECOVERED": "mode",
    # Kline-gate alarm: the bot is up but cannot EVALUATE (OKX served
    # partial candles for a run of rounds). Same channel, same shape.
    "KLINE_GATE_BLOCKED": "mode",
    "KLINE_GATE_RECOVERED": "mode",
    "KLINE_FETCH_FAILING": "mode",
    "ENDPOINT_MOVE_TRIGGERED": "mode",
    "ENDPOINT_MOVE_CLEARED": "mode",
    "KLINE_FETCH_RECOVERED": "mode",
}

# Severity tag per kind. ASCII-only, monospace-friendly. Replaces the
# earlier emoji-shortcode header that rendered as Discord emoji glyphs
# (visually noisy and hard to filter / grep in alert pipelines).
_SEVERITY_BY_KIND: dict[str, str] = {
    "CRASHED": "CRIT",
    "DOWN": "CRIT",
    "STOPPED": "CRIT",
    "SPAWN_FAILED": "CRIT",
    "SERVICE_CRASHED": "CRIT",
    "UNINSTRUMENTED": "WARN",
    "MODE_TRANSITION_REFUSED": "WARN",
    "SUPPRESSED_FAST_CRASHLOOP": "WARN",
    "SLOW_CRASHLOOP_WARNING": "WARN",
    "STARTED": "INFO",
    "REBOOTED": "INFO",
    "RECOVERY_AFTER_CRASH": "INFO",
    "MODE_TRANSITION": "INFO",
    "POOL_GATE_BLOCKED": "CRIT",
    "POOL_GATE_RECOVERED": "INFO",
    "KLINE_GATE_BLOCKED": "CRIT",
    "KLINE_GATE_RECOVERED": "INFO",
    "KLINE_FETCH_FAILING": "CRIT",
    "ENDPOINT_MOVE_TRIGGERED": "CRIT",
    "ENDPOINT_MOVE_CLEARED": "INFO",
    "KLINE_FETCH_RECOVERED": "INFO",
}


def _local_time_str() -> str:
    """America/New_York wall time for Discord human-readability.

    Deliberately whole-second and local-only. A millisecond + dual-UTC
    stamp was added to make a concurrent STOPPED/STARTED pair orderable
    when Discord's arrival order could not be trusted; ordering is now
    structural (the started hook waits on its stopped sibling), so the
    precision bought nothing and cost a timestamp that wrapped onto two
    lines on a phone. Do not re-add it without a consumer that needs it.
    """
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo("America/New_York")
        return datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception:
        return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


# ---------------------------------------------------------------------------
# Channel resolution
# ---------------------------------------------------------------------------

def resolve_webhook_env(mode: str, kind: str) -> str:
    """Return the env-var name whose value is the target webhook URL."""
    channel = _CHANNEL_BY_KIND.get(kind, "mode")
    if channel == "general":
        return GENERAL_WEBHOOK_ENV
    if channel == "live":
        return LIVE_WEBHOOK_ENV
    if channel == "dry":
        return DRY_WEBHOOK_ENV
    # "mode" -> mode-specific
    if mode == "live":
        return LIVE_WEBHOOK_ENV
    if mode == "dry":
        return DRY_WEBHOOK_ENV
    raise ValueError(f"unknown_mode_for_routing: {mode!r}")


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

def _safe_read_json(path: Path) -> dict | None:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except (PermissionError, OSError):
        return None
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def rate_limit_ok(last_alert_path: Path, key: str, now: float, cooldown_s: float = _ALERT_COOLDOWN_S) -> bool:
    """True if we haven't alerted for ``key`` within ``cooldown_s``.

    Updates last_alert.json on every attempt (success OR failure) so a flapping
    endpoint doesn't get hammered with retries.
    """
    data = _safe_read_json(last_alert_path) or {}
    last_ts = 0.0
    raw = data.get(key)
    if isinstance(raw, (int, float)):
        last_ts = float(raw)
    if (now - last_ts) < cooldown_s:
        return False
    data[key] = now
    try:
        last_alert_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = last_alert_path.parent / (last_alert_path.name + ".tmp")
        tmp.write_text(
            json.dumps(data, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        tmp.replace(last_alert_path)
    except Exception:
        # Best-effort. Still allow the send if we can't persist state.
        pass
    return True


# ---------------------------------------------------------------------------
# Message building
# ---------------------------------------------------------------------------

def _clip_text(text: str, max_lines: int, max_chars: int) -> str:
    lines = text.splitlines()
    result = "\n".join(lines[:max_lines])
    if len(result) > max_chars:
        result = result[:max_chars] + "\n... [truncated]"
    return result


def _tail_latest_err_log(logs_dir: Path, max_lines: int = 20) -> str | None:
    """Return the last *max_lines* of the most-recent ``*_err.log`` file."""
    if not logs_dir.exists():
        return None
    try:
        candidates = [p for p in logs_dir.glob("*_err.log") if p.is_file()]
    except OSError:
        return None
    if not candidates:
        return None
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    try:
        content = latest.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    lines = content.splitlines()
    return "\n".join(lines[-max_lines:]) if lines else None


def build_message(
    *,
    mode: str,
    kind: str,
    fields: dict[str, Any] | None = None,
    art: dict[str, Path] | None = None,
    detail: str | None = None,
) -> str:
    """Compose a Discord message body for the given notification kind."""
    fields = fields or {}
    hostname = socket.gethostname()
    local = _local_time_str()

    severity = _SEVERITY_BY_KIND.get(kind, "INFO")
    # D4: STOPPED is INFO when intentional (systemctl stop, mode mutex, deploy)
    # and CRIT otherwise. notify_lifecycle passes ``intentional=True`` on the
    # Result=success stop path; any unintentional STOPPED stays CRIT.
    if kind == "STOPPED":
        severity = "INFO" if fields.get("intentional") else "CRIT"
    # D5: [MODE] prefix on EVERY alert (matches the bet-alert channel prefix) —
    # consistent visual scan + a guard against webhook misconfiguration (a
    # misrouted alert is obvious by its mode tag).
    mode_tag = f"[{mode.upper()}] "
    header = f"{mode_tag}[{severity}] **{kind}** `PancakeBot-{mode}` on `{hostname}` at `{local}`"
    lines: list[str] = [header]

    if detail:
        lines.append(f"detail: `{detail}`")

    # Common context fields (best-effort).
    for k in ("pid", "bankroll", "iterations", "last_epoch"):
        if k in fields:
            lines.append(f"{k}: `{fields[k]}`")

    if kind == "CRASHED" and art is not None:
        crash = _safe_read_json(art["crash"])
        if crash is not None:
            exc_type = crash.get("exc_type", "?")
            exc_repr = crash.get("exc_repr", "?")
            lines.append(f"exc: `{exc_type}`")
            lines.append(f"repr: `{exc_repr}`")
            tb_raw = str(crash.get("traceback_str", ""))
            tb = _clip_text(tb_raw, max_lines=20, max_chars=1500)
            if tb:
                lines.append("```\n" + tb + "\n```")
    elif kind == "UNINSTRUMENTED":
        lines.append("note: legacy bot detected outside service control")
    elif kind == "SPAWN_FAILED":
        spawn_err = fields.get("spawn_error")
        if isinstance(spawn_err, str) and spawn_err:
            lines.append(f"spawn_error: `{spawn_err}`")
        lines.append("note: service failed to spawn a bot child; manual intervention required")
    elif kind in ("POOL_GATE_BLOCKED", "POOL_GATE_RECOVERED",
                  "KLINE_GATE_BLOCKED", "KLINE_GATE_RECOVERED",
                  "KLINE_FETCH_FAILING", "KLINE_FETCH_RECOVERED",
                  "ENDPOINT_MOVE_TRIGGERED", "ENDPOINT_MOVE_CLEARED"):
        # Structured kv rendering ordered for phone triage: how bad, how
        # long, why, how far behind, where coverage last held. The same
        # field set also arrives on the `detail` line.
        for k in ("signal", "rate", "trigger", "static_wake_share",
                  "header_failure_rate", "pool_uncovered_rate",
                  "window_rounds", "consecutive",
                  "recovered_after", "blocked_for", "reason",
                  "blocks_short", "getlogs_p99_ms", "last_ok_epoch", "epoch"):
            if k in fields:
                lines.append(f"{k}: `{fields[k]}`")
        if kind == "POOL_GATE_BLOCKED":
            lines.append(
                "note: bot is UP and ENABLED but cannot place bets — pool "
                "coverage unavailable. No capital at risk; it resumes "
                "automatically when coverage returns."
            )
        elif kind == "KLINE_GATE_BLOCKED":
            lines.append(
                "note: bot is UP and ENABLED but cannot EVALUATE — OKX "
                "returned partial candles for every round in this run. Each "
                "round was SKIPPED before any signal was computed, so no bet "
                "was ever sized on incomplete data. Resumes automatically."
            )
        elif kind == "KLINE_GATE_RECOVERED":
            lines.append("note: OKX publishing on time again; evaluation resumed.")
        elif kind == "KLINE_FETCH_FAILING":
            lines.append(
                "note: GENUINE kline fetch failures (unreachable / HTTP error "
                "/ gapped response) \u2014 not the benign publish delay. Rounds "
                "are skipped, no bet is sized on partial data, but this is a "
                "real fetch fault worth investigating."
            )
        elif kind == "KLINE_FETCH_RECOVERED":
            lines.append("note: kline fetches succeeding again.")
        elif kind == "ENDPOINT_MOVE_TRIGGERED":
            # THE DISCRIMINATOR INSTRUCTION LIVES HERE, NOT IN THE DESIGN
            # DOC. An alarm that fires without it wastes the window, which
            # is exactly what happened between 2026-08-23 and 08-24: the
            # condition was active for four days and the ours-vs-theirs
            # question is still circumstantial because nobody ran the test
            # while it could still be run.
            lines.append(
                "ACTION: execute the endpoint move for the header/anchor "
                "path. Both metrics are above; either one alone is "
                "sufficient to act."
            )
            lines.append(
                "RUN THE DISCRIMINATOR FIRST — IT ONLY WORKS WHILE THIS IS "
                "ACTIVE. From the VM, issue the SAME eth_getBlockByNumber "
                "request to bloXroute and to an independent host "
                "back-to-back, n>=30, and compare p50/p99. Both slow = "
                "network/VM (ours). bloXroute slow, independent host fast = "
                "provider (theirs). Save the paired output; once the "
                "condition clears this question cannot be answered."
            )
            lines.append(
                "BANKED LATENCY (anchor peak-window measurement 2026-08-24, "
                "13:00-18:00 UTC, n=35, production cadence — 8s idle, warm "
                "pooled). p50/p90/p99/max ms. bloXroute: head 4/5/15/15, "
                "header 4/8/30/30, block_ts 6/10/25/25. publicnode: head "
                "19/29/52/52, header 17/25/32/32, block_ts 18/24/67/67."
            )
            lines.append(
                "TIMEOUTS — NO CONSTANT NEEDS TO CHANGE. Derived at 3.5x "
                "the CANDIDATE host's p99 (publicnode, unrounded samples; "
                "the printed p99s are rounded to ms) vs current: "
                "_BLX_HEAD_TIMEOUT_MS 250 -> 182, _BLX_HEADER_TIMEOUT_MS "
                "250 -> 114, _BLX_BLOCK_TS_TIMEOUT_MS 250 -> 236 (the "
                "LARGEST derived value, i.e. the least slack under 250), "
                "ANCHOR_POLL_TIMEOUT_MS 200 -> 114. Every derived value "
                "sits BELOW the current one, so the move needs no "
                "increase. KEEP the current values: tightening buys "
                "nothing and only adds wake_mode=static risk."
            )
            lines.append(
                "ALREADY VERIFIED, do not re-run: fan-out — 8 concurrent "
                "getBlockByNumber + head + header, 18 iterations (~180 "
                "requests), wall p50 34 / p90 53 / max 110ms, ZERO failures "
                "and zero 429s. Header parity — 200/200 byte-identical on "
                "(hash, timestamp, mixHash, derived milli_ts). Head skew — "
                "min -1 / p50 0 / max 0; publicnode was NEVER behind "
                "bloXroute across 35 samples."
            )
            lines.append(
                "CAVEATS — these make it ready to DECIDE, not ready to "
                "EXECUTE. (1) n=35 is thin for a p99: treat the numbers as "
                "confirming 250/200 are adequate, NOT as a replacement "
                "derivation; a real move should re-soak. (2) Both hosts "
                "were sampled while NEITHER was degraded, so these are "
                "healthy-state numbers for publicnode too, and the fan-out "
                "shows BURST tolerance, not multi-day sustained tolerance. "
                "A permanent move roughly DOUBLES sustained load on an "
                "endpoint already serving our getLogs — and that host "
                "returned HTTP 403 to a burst of ad-hoc reads on "
                "2026-08-24, which is the same caveat from the other side."
            )
            lines.append(
                "DO NOT LEAN ON THE SECOND-ORDER BENEFIT: the static-"
                "fallback effect on partial-kline rate is only 0.6-1.4pp. "
                "Settled by natural experiment — a 0% static-wake day moved "
                "the partial rate 33.3% -> 29.2% only, because that rate is "
                "dominated by OKX publish latency, which is structural. "
                "Argue the move on the header path, not on this."
            )
            lines.append(
                "SUPPORTING (separate, earlier measurement of the same "
                "host — 2026-08-21 getLogs split): publicnode eth_getLogs "
                "18-block p50 17 / p95 35 / p99 41 / max 51ms; 660-block "
                "p50 41 / p95 61 / p99 103ms; timeout 250ms. The regression "
                "that motivated that split: bloXroute 2,865ms vs publicnode "
                "11ms on an identical 1-block filtered call, per-METHOD not "
                "per-host."
            )
        elif kind == "ENDPOINT_MOVE_CLEARED":
            # NOT a normal recovery, and the wording must not read like one.
            lines.append(
                "note: THE DIAGNOSTIC WINDOW HAS CLOSED. This is not "
                "\"fine now\" — the discriminator can no longer be run "
                "until the condition returns. If it was not run while the "
                "condition was active, the ours-vs-theirs question stays "
                "CIRCUMSTANTIAL and the next occurrence starts from the "
                "same place."
            )
        else:
            lines.append("note: pool coverage restored; betting resumed.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Discord HTTP send
# ---------------------------------------------------------------------------

# Discord rejects a webhook `content` longer than this with HTTP 400 — it
# does NOT truncate. An over-long alert is therefore not a degraded alert,
# it is a SILENT one, and a re-alert loop regenerates the same oversized
# body forever. ENDPOINT_MOVE_TRIGGERED rendered 2,990-3,310 chars and was
# permanently undeliverable, while its own RECOVERED message (708) sent
# fine — so the alarm would have produced exactly the outcome its
# RECOVERED text exists to warn about. Chunking lives in the transport
# because deliverability is a property of the notification system, not of
# any one alert.
_DISCORD_CONTENT_LIMIT = 2000
_DISCORD_PART_MARKER_ROOM = 12          # "(nn/nn) " plus slack


def _chunk_for_discord(
    message: str, limit: int = _DISCORD_CONTENT_LIMIT,
) -> list[str]:
    """Split ``message`` into parts that each fit Discord's content cap.

    Splits on LINE boundaries so a rendered alert never breaks mid-fact;
    a single line longer than the budget is hard-split as a last resort.
    Part markers are added only when there is more than one part, so the
    overwhelming majority of alerts are byte-identical to before.
    """
    if len(message) <= limit:
        return [message]
    body = limit - _DISCORD_PART_MARKER_ROOM
    parts: list[str] = []
    cur: list[str] = []

    def _flush() -> None:
        if cur:
            parts.append("\n".join(cur))
            cur.clear()

    for line in message.split("\n"):
        while len(line) > body:            # unsplittable single line
            _flush()
            parts.append(line[:body])
            line = line[body:]
        projected = sum(len(x) + 1 for x in cur) + len(line)
        if cur and projected > body:
            _flush()
        cur.append(line)
    _flush()
    if len(parts) <= 1:
        return parts or [message]
    n = len(parts)
    return [f"({i + 1}/{n}) {p}" for i, p in enumerate(parts)]


def _send_discord(webhook_url: str, mode: str, message: str) -> tuple[bool, str]:
    """POST a Discord message. Returns ``(ok, detail)``. Never raises.

    Sends sequentially when the body exceeds the content cap. The first
    part carries the most important lines because ``build_message``
    already orders them that way (ACTION and the discriminator first), so
    a partial delivery still delivers the actionable part.
    """
    try:
        import requests
    except Exception as e:
        return False, f"requests_import_failed:{e}"
    chunks = _chunk_for_discord(message)
    n = len(chunks)
    details: list[str] = []

    def _failed(i: int, why: str) -> tuple[bool, str]:
        # LEAD WITH THE PART, because a later part failing is the one
        # failure mode that still LOOKS delivered from the phone: part 1
        # arrives and the operator sees an alert, while the constants and
        # caveats in part 2 never land. `notify` already returns
        # SEND_FAILED, but the only place that is visible is this line, so
        # it has to be greppable on its own -- a systematically failing
        # part 2 must be obvious from the journal without correlation.
        details.append(f"part{i}/{n}:{why}")
        return False, (f"FAILED_AT_PART_{i}_OF_{n} delivered={i - 1}/{n} "
                       + " ".join(details))

    for i, chunk in enumerate(chunks, 1):
        payload = {"content": chunk, "username": f"PancakeBot-{mode}"}
        try:
            r = requests.post(webhook_url, json=payload, timeout=10)
        except Exception as e:
            return _failed(i, f"post_exception:{type(e).__name__}:{e}")
        if not (200 <= r.status_code < 300):
            return _failed(i, f"http_{r.status_code}:{(r.text or '')[:200]}")
        details.append(f"part{i}/{n}:http_{r.status_code}")
    return True, " ".join(details)


# ---------------------------------------------------------------------------
# Public entry: notify(...)
# ---------------------------------------------------------------------------

def notify(
    *,
    mode: str,
    kind: str,
    fields: dict[str, Any] | None = None,
    art: dict[str, Path] | None = None,
    detail: str | None = None,
) -> str:
    """Dispatch a notification. Returns an outcome tag for the caller's log.

    Outcomes: SENT, DISABLED (env var unset), RATE_LIMITED, SEND_FAILED.

    Never raises. If ``art`` is None, rate-limit state is not persisted
    (used by the service crash path where we may not have an art dict
    available); the alert still tries to send.
    """
    env_var = resolve_webhook_env(mode, kind)
    webhook = os.environ.get(env_var, "").strip()
    if not webhook:
        return "DISABLED"

    now = time.time()
    if art is not None:
        if not rate_limit_ok(art["last_alert"], kind, now):
            return "RATE_LIMITED"

    msg = build_message(mode=mode, kind=kind, fields=fields, art=art, detail=detail)
    ok, send_detail = _send_discord(webhook, mode, msg)
    if ok:
        return "SENT"
    # stderr -> journald under the systemd-hosted callers. Guarded so the
    # diagnostic write can never escalate a SEND_FAILED into a crash of the
    # alerting process (2026-05-23: an unguarded stderr write here took the
    # alerter down when the first-run Discord POST failed on unresolved DNS).
    try:
        print(
            f"discord_send_failed mode={mode} kind={kind} detail={send_detail}",
            file=sys.stderr,
        )
    except Exception:
        pass
    return "SEND_FAILED"

