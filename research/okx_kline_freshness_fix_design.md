# OKX kline freshness fix — design doc

## Diagnostic summary (Phase 1)

The dry bot's `gate_no_signal` epidemic (262 consecutive skips before
restart, then continuing) was traced to the OKX `/api/v5/market/candles`
endpoint returning **lagged windows** in 75% of fetches. Root cause:
**connection-reuse-induced backend affinity** — the precise mechanism
(LB sticky routing, backend-pinned keep-alive, urllib3 pool state, TLS
session resumption) is undetermined but irrelevant. Closing the
connection breaks the affinity. Definitively NOT CDN caching
(verified by `cf-cache-status: DYNAMIC`, `age: 0` on 100% of probe
responses).

### Evidence

**A/B probe**, 90 fetches across 3 variants on a fresh script process
(no bot involvement), 2026-04-25 first run:

| variant         | n  | mean lag | p50 | p95   | max   | >1s | >3s |
|-----------------|---:|---------:|----:|------:|------:|----:|----:|
| session_reuse   | 30 |   +458ms |+329 | +1999 | +2186 |   6 |   0 |
| **fresh_conn**  | 30 |   **+95ms** |**+38** |  **+762** | **+1299** | **1** |   **0** |
| cache_buster    | 30 |   +426ms |+413 | +1672 | +1699 |   6 |   0 |

**Extended A/B**, 180 fetches (60/variant) at 2s intervals:

| variant         |  n | mean | p50 | p95   | max   | >1s | >3s |
|-----------------|---:|-----:|----:|------:|------:|----:|----:|
| session_reuse   | 60 | +256 |+276 | +1418 | +1637 |   6 |   0 |
| **fresh_conn**  | 60 |  **-3** | **-23** | +1495 | +1833 |   4 |   0 |
| cache_buster    | 60 |  +72 | +12 | +1035 | +1572 |   3 |   0 |

`fresh_conn` (new `requests.Session()` per call + `Connection: close`)
**reduces mean lag 80-100%** (down to essentially zero in OKX frame).
`cache_buster` (unique query param) helps slightly less on mean
but better on tail — rules out pure URL-keyed CDN caching as the
mechanism (the cache-control headers are explicit no-cache anyway).

**Caveat: the tail isn't fully fixed.** All three variants still show
3-6 outliers >1s in a 60-fetch window, with max lag 1.5-1.8s. Fresh
connections fix the BULK case (most rounds get valid data) but not
every round. Acceptable: pre-fix the bot validates on every round and
75% fail; post-fix we expect 5-10% to still fail (the residual tail).
That's a >10× improvement in valid-round rate.

**Verified that fresh TCP socket reaches a different backend even when
DNS resolves to the same IP**: the A/B used the same Python process
across all 3 variants (same DNS cache state), and fresh_conn still
won 5× on mean lag. So the affinity is below the IP layer — per-TCP-
connection state on either client (urllib3 pool) or server (LB
session token, keep-alive backend pin).

**Cache headers (all 90 responses):**
- `cf-cache-status: DYNAMIC` (CloudFlare did NOT cache)
- `age: 0` (no upstream age)
- `cache-control: no-cache, no-store, max-age=0, must-revalidate`
- `server: cloudflare`

CDN caching is definitively ruled out. The lag is in the OKX API
backend's response selection: when our session reuses a TCP
connection, we appear to be sticky-routed to a backend that has
slightly stale data. Closing the connection forces a fresh route
to a (typically) fresher backend.

### Stuck-cache mode (306s+, 5.8% of dry rounds)

Couldn't reproduce in the A/B probe (max lag 2.2s across 90 fetches).
But captured-bot data showed **3 different request URLs returning the
EXACT SAME response across 2-3 consecutive rounds**. With CDN caching
ruled out, the only consistent explanation is the bot stayed on a
single bad backend for 10-15 minutes via session reuse. Fresh-conn
likely fixes this too -- needs verification post-fix.

### Stuck-cache mode header data (PENDING)

Bot was restarted at 15:50 UTC with `PANCAKEBOT_CAPTURE_OKX_HEADERS=1`.
After 30+ min of capture, we'll know whether stuck-cache responses have
distinguishing header signatures (e.g. specific `x-amz-cf-pop` for the
sticky backend). [Will update this section after data accumulates.]

## Chosen path: A — Per-round connection reset

Smallest fix that addresses root cause. Reject Path B (full WSS migration)
for now: complexity vs benefit isn't justified by the diagnostic evidence
when a 5× improvement is achievable with 10 lines of code.

If post-fix verification shows residual lag (>5% of rounds with
unexpected_newest), revisit and consider WSS as Path B follow-up.

### Specific code change

**File: `pancakebot/market_data/okx_client.py`**

Modify `OkxClient.warmup()`:
- Currently: opens N GET requests on the existing `self._session` to
  fill the connection pool.
- New: BEFORE the warmup requests, close the existing session and
  create a fresh one. This forces all subsequent fetches in this
  round to use brand-new TCP/TLS connections.

```python
def warmup(self, connections: int = 3) -> None:
    # Close any existing connection pool. Without this, kline fetches
    # in subsequent rounds get sticky-routed to the same OKX backend
    # via TCP keep-alive, and that backend can serve stale data
    # (1-3s lag in 75% of rounds, 306s+ stuck for 6%). A fresh session
    # at the start of each round breaks the affinity. Verified by A/B
    # probe 2026-04-25: fresh_conn cuts mean lag 458ms -> 95ms.
    try:
        self._session.close()
    except Exception:
        pass
    self._session = requests.Session()
    # ... existing warmup logic ...
```

The existing call site (`MomentumGate.warmup_session()` invoked once
per round in the housekeeping phase) means we get exactly one session
reset per round -- no extra latency on the critical bet path.

**Per-round granularity (not per-fetch) is sufficient because urllib3's
PoolManager hands each thread its own socket up to `pool_maxsize`.**
The 3 parallel BTC/ETH/SOL fetches in `fetch_klines_async` each
acquire a separate TCP connection from the freshly-created pool, not
a shared one. So one session reset per round breaks all three
threads' affinity simultaneously. Per-fetch reset (Connection: close
on every request) would force 3 fresh handshakes on the critical
path -- unnecessary cost for the same affinity break.

### Backwards compatibility

- No signature changes. `warmup()` still accepts `connections` int.
- No new dependencies.
- Behaviour change: connection-pool affinity is broken every round.
  Cost: each round's first fetch pays a fresh TLS handshake (~50-150ms).
  But: the warmup itself runs in the housekeeping phase (~4-5s before
  cutoff), so the handshake cost is absorbed there, not on the
  critical path. Subsequent same-round fetches (the parallel
  BTC/ETH/SOL fetches at cutoff+0.25s) still benefit from
  keep-alive within that round.
- `--kline-source captured` interaction: none. The new behaviour
  affects live OKX fetches only; capture-source backtests don't
  call `warmup()`.

### Verification plan

1. **Unit test**: assert `OkxClient.warmup()` replaces `self._session`
   with a fresh instance.
2. **Backtest equivalence**: `python run.py --backtest` produces hash
   `b8518b6b724a93e2ba339eeff59da58d` (current post-sync baseline).
   No backtest behaviour change expected.
3. **Live measurement post-fix**:
   - Restart dry bot
   - Wait 30+ minutes
   - Compare lag distribution from new captures vs pre-fix:
     * Target: mean lag <200ms (vs 1-2s pre-fix)
     * Target: 0 stuck-cache events
     * Target: gate_btc_unexpected_newest rate <5% (vs 75% pre-fix)
   - Re-run captured-vs-history A/B on post-fix capture: divergence
     should be ≤2.24% (and ideally close to 0%)
4. **Soak**: 12-24 hour bot run post-fix. Expect:
   - No new error modes
   - Capture file growing cleanly
   - bot_state continues to track real bankroll
   - Supervisor classifies UP every cycle

### Rollback plan

Single commit, easily revertable. The change is in one method of
`OkxClient`. If verification surfaces an issue (e.g. TLS handshake
overhead pushes us past the timing guard), revert with:
```
git revert <fix-hash>
# restart dry bot
```

The code that landed before the fix (`session_reuse` mode) was
running for many months without crashing -- just with the lag
issue. Reverting brings back the lag but doesn't introduce new
risk.

### Reviewer checklist

- [ ] `warmup()` cleanly closes the old session (no resource leak)
- [ ] No race condition: warmup runs synchronously before any
      fetch_klines_async; the gate's existing futures pattern
      still works (verify previous round's futures fully drained
      before next round's warmup_session())
- [ ] No new exceptions on the hot path: `session.close()` in a
      try/except so a malformed close doesn't crash the round
- [ ] `_last_response_headers` dict on OkxClient is preserved across
      session swap (it's an OkxClient instance attr, not on
      requests.Session, but explicitly verify no clearing happens)
- [ ] New `requests.Session()` doesn't lose any per-session config
      (User-Agent, default headers, adapters). Current code uses
      default Session everywhere so this is OK; explicit verification
      that no caller has set custom adapters on the OkxClient session
- [ ] `app.py` paths that call `OkxClient.warmup()` directly (sync
      mode) still work after the session-swap change
- [ ] Backtest hash unchanged with `--kline-source history`
- [ ] Tests pass
- [ ] Header capture instrumentation still works (independent feature)
- [ ] Capture worker still functioning (independent feature)
- [ ] No effect on `--sync` (uses a different code path)
- [ ] Live mode safe (live mode uses the same warmup; same fix
      applies, no live-specific risk)

### Stuck-cache fallback trigger

The 5.8% stuck-cache mode (306s+ lag, same response served across
2-3 consecutive rounds) was NOT reproducible in the A/B probe
(max lag 2.2s in 90 fetches, max 1.8s in 180 fetches). Mechanism
may differ from the 1-3s lag mode. Per-round session reset is
expected but NOT proven to fix it.

**Hard fallback trigger**: if post-fix bot captures show ANY
stuck-cache events (≥1 round with lag >300s) within the first
12-hour soak, escalate to **Path B (WSS migration)** -- do not
iterate further on Path A. The simple mitigation is necessary
but not sufficient if the stuck-cache mode persists.

## Out of scope for this fix

- WSS migration (revisit if Path A doesn't fully solve)
- Lowering `_MTF_THRESH` (rejected -- validation rejects fetch
  before threshold check)
- Cutoff change (rejected -- cuts off the most-edge-relevant
  candles)
- Clock-skew correction (separate issue; the user system clock has
  1.2s skew, which compounds the lag perception. Not in scope here
  because Windows clock sync needs admin privileges and is the
  user's environment, not bot code.)
