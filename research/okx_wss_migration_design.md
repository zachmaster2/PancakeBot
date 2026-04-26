# OKX WSS migration — design doc

## Context

The dry bot's kline-fetch path is REST-based: every round, three parallel
HTTPS GETs to `https://www.okx.com/api/v5/market/candles`. Two recent
finds make WSS the right next step:

1. **Clock-skew fix (commit `4dda6cc`)** brought lag from 75% any-lag /
   ~1000ms median to **0% any-lag / 0ms median** by anchoring scheduling
   to `_utc_now()`. REST works correctly, but the path still costs
   ~250-500ms per round in TLS handshake + fetch + JSON parse.
2. **Live data has known divergence vs history** at OKX response level
   (~3% rounds had ≠ data on stuck-cache path). Even with the fix, REST
   responses are point-in-time snapshots — small windows of "OKX hadn't
   yet published the cutoff-1 candle" still occur on the tail.

WSS subscription gives us:
- Continuous push-based candle stream — no per-round fetch latency
- No stuck-cache (different transport, different backend)
- Decision-time read becomes an in-memory dict lookup (~µs)
- Survives slow networks better (heartbeat-based, not request-based)

This migration **is not load-bearing** — the REST path with skew fix
already works. WSS is a robustness + latency upgrade.

## Architecture

### Component layout

New file `pancakebot/market_data/okx_wss_client.py`. The class
`OkxWssClient` mirrors `pool_watcher.WssPoolWatcher`'s pattern: an
asyncio loop running in a daemon thread, exposing **synchronous
accessors** under a lock. Strategy-side code (`MomentumGate`) gets the
same kline shape it already gets from REST.

```
┌─ Main thread (engine.py) ─┐         ┌─ Daemon thread ──────────────┐
│  gate.evaluate(...)       │         │  asyncio loop                │
│   └─ wss.get_window(BTC)  │ ◄─────► │   ws.recv() → ring_buffer    │
│      ↑ in-memory read     │  lock   │   reconnect on drop          │
│      µs latency           │         │   refill from REST on gap    │
└───────────────────────────┘         └──────────────────────────────┘
```

### Per-instrument state

```python
class _InstrumentRing:
    symbol: str                                   # "BTC-USDT" etc.
    klines: collections.deque[list]               # [[ts_ms, o, h, l, c, v], ...]
    max_size: int = 300                           # 300s = 5min sliding window
    last_received_ms: int                         # wall-clock of last push
    last_candle_ts_ms: int                        # newest open_time_ms in ring
```

Subscribe to BTC, ETH, SOL, BNB independently. Four `_InstrumentRing`s,
each with own stale clock and ring buffer. Stored in
`OkxWssClient._rings: dict[symbol, _InstrumentRing]` guarded by
`OkxWssClient._lock`.

### OKX WSS endpoint + subscription

OKX public WSS: `wss://ws.okx.com:8443/ws/v5/business`.

**Empirically verified 2026-04-26 via `research/okx_wss_endpoint_probe.py`:**
- `/public` REJECTS `candle1s` subscription with error code 60018
  ("Wrong URL or channel:candle1s,instId:BTC-USDT doesn't exist").
- `/business` ACCEPTS and pushes (80 push messages over 30s, 31 unique
  open_times — confirmed mid-bar update behaviour).
- Subscribe ack on `/business`:
  `{'event': 'subscribe', 'arg': {'channel': 'candle1s', 'instId':
  'BTC-USDT'}}`. Returned within 250ms of connect.

Subscribe message:
```json
{
  "op": "subscribe",
  "args": [
    {"channel": "candle1s", "instId": "BTC-USDT"},
    {"channel": "candle1s", "instId": "ETH-USDT"},
    {"channel": "candle1s", "instId": "SOL-USDT"},
    {"channel": "candle1s", "instId": "BNB-USDT"}
  ]
}
```

Push messages have shape (per OKX docs):
```json
{
  "arg": {"channel": "candle1s", "instId": "BTC-USDT"},
  "data": [["1704067200000", "44000", "44100", "43950", "44050", "1.234", "..."]]
}
```

Each push is one candle update with shape (verified by probe):
`[ts_ms, o, h, l, c, vol, volCcy, volCcyQuote, confirm]`

The 9th field `confirm` is the OKX-canonical "candle closed" flag:
- `confirm = "0"`: mid-bar update (price still moving within this 1s window)
- `confirm = "1"`: candle closed, OHLC final

Probe captured 80 pushes / 31 unique open_times over 30s, with 49
same-ts repeats. About 2-3 mid-bar updates per closed candle.

**Commit policy**: a candle is appended to the ring **only when
`confirm == "1"` is received** for that open_time. This gives us
"fully closed" semantics matching REST `after=cutoff`, AND it commits
immediately on close (no need to wait for the next candle to start —
OKX pushes the closing update within ~50ms of the second boundary).

### Bootstrap (REST → WSS handoff)

At `OkxWssClient.start()`:

1. **Phase 1 — REST initial fill**: For each instrument, call
   `OkxClient.fetch_raw(method='history-candles', symbol, limit=300,
   after=now_ms)`. Populates ring with 300 historical candles.
2. **Phase 2 — WSS connect + subscribe**: Open WSS, send subscribe op,
   wait for confirmation messages.
3. **Phase 3 — Live append**: Each push writes to ring's pending slot;
   on next push (= candle finalized), append to ring deque (drops
   oldest if at max_size).

Bridge correctness: at the moment WSS push starts arriving, we may
overlap with the REST data. Dedupe by `open_time_ms` — if a WSS
candle's ts is `<=` newest ring ts, skip. That handles the gap
cleanly.

Bootstrap-readiness ordering (per-ring), explicit:

1. REST initial fill complete (ring has ≥ 31 candles).
2. WSS subscription confirmation received for this instrument.
3. At least one push received whose `open_time > newest REST ring ts`
   AND with `confirm == "1"` (i.e. one fully-closed WSS candle bridges
   cleanly past the REST tail).

Only when **all 4 rings have satisfied conditions 1+2+3 simultaneously**
does `is_ready()` return True. Until then, the gate's `evaluate()`
short-circuits with `risk_kline_wss_stale` (or a more specific
`risk_kline_wss_bootstrap_pending` if we want to differentiate startup
from steady-state outage — bias toward fewer skip reasons unless
investigation needs it).

This ordering eliminates the race where REST has filled to ts=N but
WSS's first push lands on ts=N (mid-bar update of the candle REST
already gave us). The "strictly newer + confirmed" gate guarantees
no stale data + no missed candles in the bridge.

### Reconnect / gap-detection

WSS connection can drop. Reconnect machinery follows
`pool_watcher.py` exactly:

- **Endpoints**: primary `wss://ws.okx.com:8443/ws/v5/business`,
  fallback `wss://wsaws.okx.com:8443/ws/v5/business` (AWS fanout, OKX docs).
  Round-robin on each disconnect.
- **Backoff**: exponential per-cycle `[5, 10, 20, 40, 80, 120s]` (line
  48-49 of pool_watcher).
- **Health threshold**: a session lasting >60s resets the failure
  streak (line 302-305). Same as pool_watcher.
- **Heartbeat**: OKX WSS sends `ping`/`pong` every 25s. We respond.
  If no pong within 30s, assume dead and reconnect.

On reconnect:
- Re-subscribe.
- For each ring, if the gap (new ts vs old newest ts) > 5 candles,
  call REST `/history-candles` to fill the gap. **Gap-fill REST calls
  share the existing `OkxClient` rate budget** (`_rate_acquire` token
  bucket in `pancakebot/market_data/sync.py`) to avoid hammering OKX
  during a regional incident that might also be causing the WSS drop.
  If we can't acquire a token within 30s, give up and skip rounds
  with `risk_kline_wss_stale` until the next successful push.
- If gap > 100 candles, **re-run the full 3-step bootstrap** for that
  ring (REST fill → WSS sub-confirm → first strictly-newer-confirmed
  push). Same gating as initial bootstrap; `is_ready()` returns False
  until satisfied.
- During reconnect window, `is_ready()` returns False → bot skips with
  `risk_kline_wss_stale`.

### Stale-data refusal

Per-ring stale check inside `get_window(symbol, cutoff_ms)`:

```python
def get_window(self, symbol, cutoff_ms, expected_count=31, stale_threshold_ms=None):
    ring = self._rings[symbol]
    threshold = stale_threshold_ms if stale_threshold_ms is not None else self._cfg.wss_stale_threshold_ms
    with self._lock:
        # Both last_received_ms and now use LOCAL clock (set/read on
        # the same machine) -- skew cancels naturally. Do NOT skew-
        # correct here; that would double-correct.
        now_ms = int(time.time() * 1000)
        if now_ms - ring.last_received_ms > threshold:
            return None, "wss_stale"
        # Filter ring to candles with open_time < cutoff_ms.
        # cutoff_ms IS chain-anchored true UTC, but ring open_times
        # come from OKX (also true UTC), so direct comparison is
        # correct frame-wise.
        valid = [k for k in ring.klines if k[0] < cutoff_ms]
        if len(valid) < expected_count:
            return None, "wss_insufficient"
        return valid[-expected_count:], None
```

Default 5s threshold: at 1s candle cadence, missing 5 consecutive
pushes is a clear signal something's wrong. Bot refuses to bet that
round.

**Configurable via `[okx] wss_stale_threshold_ms = 5000`** in
`config.toml` so we can tune without a code change. Adaptive
thresholding (tighter during high-volatility) is deferred — premature
optimization. During the soak verification, if `risk_kline_wss_stale`
rate exceeds 0.5%, revisit the threshold.

**Frame correctness (Q10 from reviewer):** `last_received_ms` is set
to `int(time.time() * 1000)` at the moment the daemon thread's
`recv()` returns — pure LOCAL clock. The threshold compares LOCAL
now to LOCAL last-received; skew cancels (same frame on both sides).
We DO NOT apply skew correction to this comparison. Skew correction
is only relevant for `cutoff_ms` filtering (which uses chain-anchored
timestamps that ARE in true UTC frame).

### Multi-instrument independence

**Decision: BTC stale → skip. ETH+SOL both stale within ≤5s of each
other → skip (correlated-stale safeguard). One of ETH/SOL stale
alone → degraded (BTC-primary only, no regime-2). BNB stale → log
only, don't block.**

Rationale:
- BTC primary signal drives the gate. Without it, no signal.
- ETH/SOL feed `regime-2` (the secondary fallback). Either one
  alone going stale is a single-stream issue; BTC-primary still safe.
- **Both ETH and SOL stale together is strong evidence of a
  region-wide OKX incident** (same TCP, same backend pool). BTC may
  be next or already silently lagging. Skip with
  `risk_kline_wss_correlated_stale`.
- BNB price klines are not part of `_compute_signal` or any decision
  filter (verified: `momentum_gate.py:120-122` only fetches BTC/ETH/
  SOL; `kline_capture.py:151-153` capture builder reads only btc/eth/
  sol). Recorded for capture/replay only.

Implementation: gate's `evaluate()` queries each ring; on BTC-stale
returns `_skip("risk_kline_wss_stale")`; on ETH+SOL both stale,
returns `_skip("risk_kline_wss_correlated_stale")`; on either one
stale alone, sets that side's `*_signal=None` and proceeds with
BTC-only; on BNB-stale, no impact on decision.

### Integration with MomentumGate

**Per user mandate (2026-04-26): no backward compatibility. WSS is
the canonical live-mode kline path. Per-round REST fetch in live
mode is REMOVED, not gated by a flag.**

`MomentumGate` constructor takes `wss_client` as a required arg in
live mode. The class no longer accepts a "REST per-round" mode for
live operation:

```python
class MomentumGate:
    def __init__(self, *, config, okx_client, wss_client):
        ...
        self._wss = wss_client    # required for live runtime
        # _client is kept ONLY for bootstrap REST /history-candles
        # and reconnect gap-fill paths inside OkxWssClient.
        self._client = okx_client
```

Live-mode behaviour:

- **`fetch_klines_async()` is DELETED** (not made a no-op — physically
  removed from the class). Its previous call site in `engine.py` is
  also removed: no kline_futures variable, no future-collection code.
- `evaluate()` reads from `self._wss.get_window(symbol, cutoff_ms)`
  directly. Sub-millisecond in-memory dict lookup + range filter +
  validation.
- Validation logic stays the same — same shape, same checks. The
  "newest_ts == cutoff - 1000" assertion holds because WSS commits
  candles at the second boundary (verified by `confirm == "1"` push).
- Capture-snapshot fields (`last_btc_klines_raw` etc.) populated from
  the WSS ring read. Schema unchanged.

REST `OkxClient.fetch_1s_klines` survives, but with two narrower
purposes:
1. **Bootstrap REST initial fill** — startup-only, used by
   `OkxWssClient.start()` to populate the rings before WSS catches up.
   Required for initialization, not "backward compat."
2. **Reconnect gap-fill** — when WSS drops and reconnects, REST fetches
   the gap. Bounded by the same rate budget as the existing client.

The sync-mode `OkxClient.fetch_raw` and the backtest sweep harness's
REST history path are **untouched** — those are research/research-replay
flows, not live runtime.

There is **no `use_wss` config flag**. WSS is the single canonical
live path; there is no live-mode toggle. Bot bootstraps WSS at
startup or refuses to run.

### Engine integration

In `run_realtime_loop`, **WSS construction is mandatory** (no flag):

```python
wss_client = OkxWssClient(
    okx_client=client,
    instruments=("BTC-USDT", "ETH-USDT", "SOL-USDT", "BNB-USDT"),
)
wss_client.start()  # spawns daemon thread, blocks until is_ready()
info("OKX_WSS", "INIT", "READY", msg=f"WSS subscribed; rings populated")
atexit.register(wss_client.stop)  # graceful shutdown (Q8)
```

If WSS bootstrap fails (REST history unreachable, all WSS endpoints
down), `start()` raises. Bot startup fails loudly — fail-fast rather
than silently degrade.

Engine timing changes:
- `gate.warmup_session()` is **deleted** (no REST fetch to warm).
- The skew refresh in housekeeping still runs (we still need accurate
  `_utc_now()` for `_sleep_until_ts` scheduling against `lock_at`,
  for `_LOCK_SAFETY_MARGIN_SECONDS` guard, and for stale-data
  threshold).
- `gate.fetch_klines_async()` call site in engine.py is **deleted**
  (no futures, no waiting).
- `_OKX_PUBLISH_DELAY_SECONDS` reduced from 0.25 to 0.05 — WSS commits
  on `confirm == "1"` within ~50ms of the second boundary, so we just
  need a small grace period after cutoff to ensure the cutoff-1 candle
  is already in the ring.

Critical-path read: `evaluate()` does an in-memory dict lookup +
range filter + validation. Sub-millisecond. Frees up timing budget
that previously went to TLS handshake + fetch.

### Backtest equivalence preservation

- `--kline-source history`: unchanged. Reads from disk
  (`var/btc_spot_prices.jsonl` etc.). No WSS involved.
- `--kline-source captured`: unchanged. Reads from disk
  (`captured_klines.jsonl`). Captures from a WSS-fed bot have
  identical shape to REST-fed captures, so this path "just works"
  with new captures.
- Backtest path constructs `MomentumOnlyPipeline(gate=None, ...)`
  (line 245-260 of `runner.py`). WSS isn't constructed here.

**Explicit isolation proof (Q5 from reviewer):**
The backtest entrypoint is `pancakebot/backtest/runner.py:run_backtest`
called directly by `pancakebot/app.py` for `--backtest` mode. The
function never invokes `engine.run_realtime_loop`, where
`OkxWssClient` is constructed. The entire backtest pipeline is
disk-bound (kline_store → MomentumOnlyPipeline(gate=None)). Therefore
**`OkxWssClient` is never instantiated in backtest mode regardless
of `[okx] use_wss` config flag**. Hash equivalence guaranteed.

### Capture continuity

`record_round_decision` reads from gate's `last_btc_klines_raw` etc.
attrs. These get populated the same way in WSS path. Capture format
unchanged (schema_version stays at 1). Backwards-compatible: a
post-WSS bot's captured.jsonl is consumable by every existing tool.

## Verification plan

### Unit tests

- `_InstrumentRing` — append, dedupe by ts, max_size enforcement
- `OkxWssClient.bootstrap` — REST fill + WSS subscribe + bridge
- `OkxWssClient.reconnect` — endpoint cycling, backoff, gap-fill
- `OkxWssClient.get_window` — stale detection, insufficient-count
  detection, cutoff filtering
- `MomentumGate.evaluate` with mocked WSS client — same outputs
  as REST path on identical kline data

### Integration tests

- Full WSS lifecycle: connect, subscribe, receive 5 push messages,
  ring populated, get_window returns 31-element window matching
  expected newest_ts.
- Reconnect: simulate disconnect (close socket from inside test),
  observe reconnect + re-subscribe + ring continuity.
- Bootstrap with mocked REST + WSS: ring filled correctly from REST,
  subsequent WSS pushes bridge cleanly.
- Stale threshold: pause pushes for 6s, observe `get_window` returns
  None with `wss_stale`. Resume pushes, observe recovery.

### Live verification (when bot resumes)

- Lag distribution: should be ~0ms (in-memory read; no fetch latency).
- Backtest hash equivalence at default `--kline-source history`:
  `c8e53948bc6182faf939c90a82d92154` (current post-sync baseline).
- A/B captured-vs-history rerun: divergence should be ≤2.24% (the
  pre-fix figure). Ideally ~0%.
- 12-24h soak: zero stuck-cache events, zero `risk_kline_wss_stale`
  spike events under normal network conditions.

### Backtest hash check (CI / pre-merge)

`python run.py --backtest` produces hash
`c8e53948bc6182faf939c90a82d92154` (current post-sync baseline) with
`use_wss = false` AND with `use_wss = true` (since backtest doesn't
construct WSS — proves no inadvertent coupling).

## Reviewer's blocker checklist

Required for sign-off:

- [ ] **Bootstrap correctness**: 3-step ordering (REST fill → WSS
      sub-ack → ≥1 strictly-newer push with `confirm == "1"`) holds
      per-ring before `is_ready()` returns True.
- [ ] **Reconnect robustness**: endpoint cycling + exponential backoff
      matches `pool_watcher` pattern. Health threshold (60s) resets
      failure streak. Reconnect doesn't lose pending candle.
- [ ] **Reconnect rate-limit budget**: REST gap-fill calls share
      existing `OkxClient`'s `_rate_acquire` token bucket — no
      hammering OKX during a regional incident.
- [ ] **Re-bootstrap path on >100 candle gap**: full 3-step bootstrap
      runs again for that ring; `is_ready()` returns False until
      satisfied.
- [ ] **Stale-data refusal**: 5s default threshold (configurable via
      `[okx] wss_stale_threshold_ms`). `last_received_ms` uses LOCAL
      clock; threshold compares LOCAL-vs-LOCAL (skew cancels). No
      double-correction.
- [ ] **Multi-instrument independence**: BTC stale blocks; ETH+SOL
      both stale → `risk_kline_wss_correlated_stale`; one of ETH/SOL
      alone → degraded BTC-primary mode; BNB stale → log only.
      Documented + tested.
- [ ] **Backtest entrypoint isolation**: `runner.py:run_backtest`
      never invokes `engine.run_realtime_loop`; `OkxWssClient` cannot
      be constructed in backtest mode. Hash equivalence guaranteed.
- [ ] **Capture continuity**: post-WSS captured.jsonl readable by
      `load_klines_from_capture()`, schema_version unchanged.
- [ ] **Gate-result invariants**: same `MomentumGateResult` fields
      populated, same `last_*_klines_raw` semantics for capture.
      Skip reasons preserved, plus new `risk_kline_wss_stale` and
      `risk_kline_wss_correlated_stale`.
- [ ] **Live REST path REMOVED, not gated**: per-round REST fetch
      machinery deleted from `MomentumGate.fetch_klines_async` (and
      its engine call site). No config flag, no toggle. `OkxClient`'s
      `fetch_1s_klines` survives only as a callable for bootstrap +
      reconnect gap-fill internal use.
- [ ] **Bootstrap REST preserved**: `OkxClient.fetch_1s_klines` still
      callable for `/history-candles` initial fill. Sweep harness's
      REST history path untouched (research/research-replay isn't
      "live mode").
- [ ] **Threading safety — zero I/O under lock**: all ring mutations
      and reads under `OkxWssClient._lock`, but the locked section
      contains ONLY in-memory operations (deque append, list filter,
      timestamp compare). No `info()`/`warn()`, no `json.dumps`, no
      file I/O, no `requests.get` inside the locked block.
- [ ] **Daemon-thread lifecycle (Q8)**: `OkxWssClient.stop()` mirrors
      `pool_watcher.stop()` exactly: `_stop_event.set()` + WSS
      `close()` + `thread.join(timeout=10)`. Registered via
      `atexit.register(wss.stop)` at engine startup. No lingering
      threads on bot shutdown.
- [ ] **Skew-aware integration**: `_utc_now()` continues to work
      unchanged for `_sleep_until_ts` and `_LOCK_SAFETY_MARGIN_SECONDS`
      guard. WSS path does not bypass any other skew-correction site.
- [ ] **Heartbeat / supervisor**: bot heartbeat continues writing
      under WSS path; supervisor sees UP. WSS daemon thread doesn't
      starve the main loop (asyncio in daemon, blocking deque under
      lock — main loop work runs at full speed between gate reads).
- [ ] **Memory bounded**: 4 rings × 300 candles × ~80 bytes = ~96 KB
      total. No unbounded growth.
- [ ] **Burst handling**: WSS rate is ~3 push/sec/instrument under
      mid-bar updates × 4 instruments = ~12 push/sec. Daemon thread
      handles each in <1ms. No queue backpressure needed (we only
      hold one pending candle per ring at a time; finalized candles
      flow into the bounded deque).
- [ ] **Endpoint + push semantics**: `/business` confirmed serves
      `candle1s`; `confirm == "1"` indicates closed candle. Verified
      empirically by `research/okx_wss_endpoint_probe.py`
      (2026-04-26).

## Out of scope (later refinements)

- WSS for BSC pool data (already exists in `pool_watcher`)
- Replacing REST `/sync` with WSS history fetch (sync runs offline,
  doesn't need real-time)
- Compression (`compress=true` query param) — defer; raw JSON is
  small enough
- Authenticated channels (we only need public market data)

## Rollback

Per user mandate, **there is no fallback flag**. WSS is the live path,
period. If a deep issue surfaces post-deploy, rollback is a `git revert`
of the implementation commit (single commit if landed clean). The
revert restores the per-round REST fetch path that was deleted.

Pre-deploy mitigations:
- Verify-before-deploy via the verification plan above.
- Soak (12-24h) before declaring stable.
- Atexit-registered shutdown so a `git revert` + supervisor restart
  brings the bot back cleanly with no zombie WSS connections.

## Estimated implementation scope

- `okx_wss_client.py`: ~400 LOC
- `momentum_gate.py` integration: ~30 LOC delta (mostly wiring the
  optional `wss_client` argument)
- `engine.py` integration: ~10 LOC delta (construct + start WSS at
  startup)
- Tests: ~250 LOC
- Total: ~700 LOC. ~3-5h implementation + 1h test + 1h reviewer iteration.

Backtest hash equivalence preserved by construction — backtest path
literally does not touch this code.
