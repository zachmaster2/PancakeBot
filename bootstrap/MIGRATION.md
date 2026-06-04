# Windows → Linux cutover playbook (Phase 4)

**Document only — no execution in Phase 2.** The cutover happens after the
Phase 3 dry soak (≥6h + a few real bets) passes on the VM.

## ⚠️ HIGH RISK: shared-wallet cross-OS nonce collision

The production wallet is reused on the VM, and the Live/Dry mutex is
**per-OS only** (SCM on Windows, systemd `Conflicts=` on Linux) — there is
**no cross-OS mutex**. If Windows-live and Linux-live run simultaneously, both
sign from the same address and **nonce-collide**, breaking bets on both sides
(the exact hazard seen in the send-raw-tx probe).

**Rule:** there must be **zero overlap** between Windows `STOPPED` and Linux
`STARTED`. Serialize strictly (steps 1–2 fully complete before step 7). The
dry soak is safe to run concurrently with Windows-live because **dry mode sends
no transactions** (no nonce use).

## What state actually migrates

Live mode rebuilds the RPC pool aggregate from chain on startup, so only the
**persistent ledger/bookkeeping** state must move. Authoritative live state
(see `pancakebot/paths.py`):

| File | Why it must move |
|---|---|
| `var/live/bets.jsonl` | **Critical** — the bet ledger (SUBMITTED/CONFIRMED/SETTLED/CLAIMED). Losing it re-fires settled alerts / mis-attributes claims. |
| `var/live/bankroll_history.jsonl` | Bankroll tracker history → drawdown-from-peak gate. |
| `var/live/claim_cursor.txt` | Claim-scan cursor (avoids re-scanning old epochs). |
| `var/live/cycle_audit.csv`, `var/live/trades.csv` | History/audit (not load-bearing; keep for continuity). |
| `config.toml`, `.env` | Config + secrets (config.toml is tracked; `.env` is not). |

Do **not** migrate: `bot.pid`, `crash.json`, `runtime.log` (transient,
regenerated). Note: live mode has **no** `bankroll.json` or `settled_epochs.txt`
— those are dry-mode files; the live equivalents are the ledger + history above.

## Pre-cutover checklist

- [ ] Phase 3 dry soak passed on the VM (≥6h, 0 Tracebacks, restart works, READY, webhooks fire).
- [ ] A few **real** bets observed end-to-end in a temporary VM-live window **with Windows stopped** (or deferred to the cutover itself).
- [ ] `pancakebot-live` unit installed on the VM but **disabled**.
- [ ] `/etc/pancakebot/pancakebot.env` has the 3 webhook URLs; `.env` has the 2 secrets (chmod 600).
- [ ] `chronyc tracking` offset within ±0.25s.
- [ ] Operator on hand to watch Discord through the first bet.

## Cutover sequence

```
1. Windows:  sc stop PancakeBotLive          (or scripts\disable_live.ps1)
2. WAIT for the "BET ... / STOPPED" Discord alert confirming a CLEAN exit
   AND `Get-Service PancakeBotLive` = Stopped AND no python bot child left.
   --- from here until step 7, NEITHER side is live (zero-overlap window) ---
3. Windows:  tar/zip the live state:
       var/live/bets.jsonl  var/live/bankroll_history.jsonl
       var/live/claim_cursor.txt  var/live/cycle_audit.csv  var/live/trades.csv
       config.toml  .env
4. scp the archive to the VM.
5. VM:  extract into <repo>/var/live/  +  config.toml/.env at repo root.
6. VM:  verify integrity — file sizes match the source, and
        `sha256sum` of bets.jsonl/bankroll_history.jsonl matches the Windows
        side (compute on both, compare).
7. VM:  systemctl enable --now pancakebot-live
8. WAIT for STARTED + BOT READY Discord alerts; `journalctl -u pancakebot-live`
   shows the supervisor spawn + the bot's "Starting bankroll" matching the
   pre-cutover figure.
9. Observe the first full BET_SUBMITTED -> (CONFIRMED) -> settle/BET_WON cycle
   end-to-end on the VM. Confirm the bankroll moves correctly post-claim
   (the read-your-writes fix, commit 390436a).
```

The zero-overlap window (steps 2→7) is a few minutes of no betting — acceptable
(the bot already skips ~most rounds). Do NOT shortcut it.

## Rollback

Trigger if anything looks wrong at steps 7–9 (bot won't start, bankroll
mismatch, repeated crashes, LATE/again, no READY):

```
R1. VM:  systemctl disable --now pancakebot-live     (stop the Linux bot first)
R2. VM:  confirm inactive (`systemctl is-active` = inactive, no python child)
         --- zero-overlap again before bringing Windows back ---
R3. If the VM advanced the ledger (placed/settled bets in steps 8–9), copy the
    VM's var/live/{bets.jsonl,bankroll_history.jsonl,claim_cursor.txt} BACK to
    Windows so the Windows ledger reflects on-chain reality. (If the VM placed
    no bets, the Windows state is already current — skip.)
R4. Windows:  sc start PancakeBotLive   (or scripts\enable_live.ps1)
R5. WAIT for STARTED + BOT READY; confirm bankroll matches chain.
```

Because state is file-based and the pool is chain-derived, rollback is clean as
long as the zero-overlap rule holds in BOTH directions and the ledger is
reconciled (R3) when the VM placed any bets.

## Post-cutover (once stable)

- Decommission: `scripts\disable_live.ps1` permanently on Windows (keep it
  installed as a warm rollback target for a while).
- Long-term, consider a **dedicated VM wallet** to remove the shared-wallet
  hazard entirely (eliminates the zero-overlap constraint for future work).
