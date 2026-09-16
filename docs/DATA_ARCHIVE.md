# The data archive — collection stopped 2026-09-16

**Read the OKX section before anything else.** It is the one fact about this archive that cannot be
fixed later.

---

## 1. Status

Data collection **stopped deliberately on 2026-09-16**, after a final clean sync. The archive ends at
**epoch 516226**. The two scheduled tasks that ran collection (`PancakeBotDailySync` and
`PancakeBotSyncWatchdog`) were removed as the last closing step, after this note was committed.
Nothing collects any more.

**The archive is not in this repository.** The stores live under `var/`, which is gitignored. They
exist only on the operator's machine: in `var/`, and in a verified zip of the closed state (section 7). This file is the record of
what they contain and the state they were closed in.

## 2. Why it stopped

- **The strategy thread is closed.** [FINDINGS.md](FINDINGS.md) explains what the bot's edge was
  (a stale settlement oracle plus a short-horizon lead-lag that has since decayed), and that it no
  longer pays.
- **No open question needs new data.** Every analysis still worth running can be run on what is
  already here.
- **The one forward test that would have needed new data was declined.**
  [prereg_A3_gap_persistence_2026_09_15.md](prereg_A3_gap_persistence_2026_09_15.md) registered a
  test to be judged on rounds after epoch 515944, then declined it: seven months of waiting for an
  expected null, on a venue whose pools were still shrinking. Stopping collection follows from that
  decision, and the registration's status section records it.

## 3. ⚠️ OKX serves 1-second klines for only ~171.6 days — restarting later does NOT backfill

**This is the single most important fact about this archive.**

- OKX only serves 1-second candles for roughly the most recent **171.6 days** (measured). Older
  seconds are simply not available from OKX, to anyone, at any price.
- At closure, that meant every second **before about 2026-03-28** was already unrecoverable from OKX.
  The archive's klines from 2025-12-11 to 2026-03-28 exist **only here** (and in the backup, up to its
  own date).
- **Every day after closure, one more day of history ages out.** By about 2027-03-07, *none* of the
  kline data in this archive can be re-fetched from OKX.
- **Restarting collection later does not fill the gap.** A restart begins a new, disconnected series.
  Rounds between epoch 516226 and whenever collection resumes will have no klines once they are more
  than ~171.6 days old, permanently.
- By contrast, `closed_rounds.jsonl` (the rounds and their bets) is **not** horizon-bound: it comes
  from on-chain data and can be rebuilt from the chain later. Only the klines are irreplaceable.

## 4. What the archive contains, and its exact boundaries

All five files are in `var/`, newline-delimited JSON, one record per round, keyed by `epoch`.

| file | contents |
|---|---|
| `closed_rounds.jsonl` | every settled PancakeSwap Prediction V2 (BNB) round: start time, lock/close price, winning side, every bet (wallet, amount, side, timestamp) |
| `bnb_spot_prices.jsonl` | OKX BNB-USDT 1-second candles for each round: 300 candles covering `startAt-1 .. startAt+298` |
| `btc_spot_prices.jsonl` | the same for BTC-USDT |
| `eth_spot_prices.jsonl` | the same for ETH-USDT |
| `sol_spot_prices.jsonl` | the same for SOL-USDT |

- **First epoch:** 437562, started 2025-12-11 09:17 UTC.
- **Last epoch:** 516226, started 2026-09-16 10:20 UTC.
- **78,665 rounds** in `closed_rounds.jsonl`; **78,657** in each kline store.

## 5. `closed_rounds` is exactly 8 rounds ahead of every kline store — this is correct

**Someone checking this archive will see the count mismatch and suspect corruption. It is not.**

Eight epochs have no klines in any of the four kline stores, and never will:

    445330, 445331, 447533, 447534, 449665, 449666, 452486, 452487

These rounds were missing from the stores and were repaired on 2026-08-24. The rounds themselves were
re-fetched successfully, but by then these eight had already passed OKX's 171.6-day horizon (section
3), so their klines could not be fetched by anyone. A whole-store byte scan on 2026-08-30 independently
found exactly these eight and nothing else. They are listed in code as
`KNOWN_ABSENT_KLINE_EPOCHS` in
[`pancakebot/market_data/known_absent.py`](../pancakebot/market_data/known_absent.py).

So: `closed_rounds` has 8 more lines than each kline store, every kline store is missing exactly
those 8 epochs, and nothing else is missing anywhere.

## 6. Integrity at closure (measured 2026-09-16, after the last write)

| store | bytes | lines | first | last | CRLF | duplicates | out of order | gaps (known / unexplained) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| closed_rounds.jsonl | 388,019,189 | 78,665 | 437562 | 516226 | 0 | 0 | 0 | 0 / 0 |
| bnb_spot_prices.jsonl | 1,056,202,490 | 78,657 | 437562 | 516226 | 0 | 0 | 0 | 8 / 0 |
| btc_spot_prices.jsonl | 1,321,510,554 | 78,657 | 437562 | 516226 | 0 | 0 | 0 | 8 / 0 |
| eth_spot_prices.jsonl | 1,270,740,223 | 78,657 | 437562 | 516226 | 0 | 0 | 0 | 8 / 0 |
| sol_spot_prices.jsonl | 1,097,552,244 | 78,657 | 437562 | 516226 | 0 | 0 | 0 | 8 / 0 |

**Cross-store:** in every kline store the last epoch matches `closed_rounds` exactly (drift +0), the
only rounds missing are the 8 known-absent epochs of section 5, and no store holds an epoch that
`closed_rounds` lacks. `closed_rounds` minus each kline store = 8 lines.

**SHA-256 of each store at closure** — if these match, the files are byte-identical to what this note
describes (matching counts alone would not prove that):

    closed_rounds.jsonl    4cd71b4e4ab903668d51812b2edbd611ce1e8af36242eeaf7c0f93290bcb9063
    bnb_spot_prices.jsonl  4d708869c61bb7d6048cf8b32313cd9bfcb0cbcb7a5e99da6ed8b617f3e09c73
    btc_spot_prices.jsonl  9e2b630a47b97f12d59abe3f81087ef976148ab4c1790b3f7b4d26f0553d201f
    eth_spot_prices.jsonl  11da7414c442395696f25338c00a01911e38dabbae128cd70c23192854dee366
    sol_spot_prices.jsonl  d5d70d1c1b305aa2a43d52290f97d0bf34c8eac06187fd65326f2dc7a752b003

**Canonical replay:** `tests/test_in_process_runner.py::test_canonical_baseline_bit_identical` ran
(not skipped) against these exact files and **passed**, reproducing the canonical 5-fold hash
`ac3fb12ab299930fa9f2a9fcac45cfbb` bit for bit. That test replays tens of thousands of rounds through
the real engine, so it is the strongest single check that the archive is sound.

To re-check later: `sha256sum var/*.jsonl` against the table above, and
`python -m pytest tests/test_in_process_runner.py -k canonical`.

## 7. Copies, and the single point of failure

**The checksums in section 6 describe files that exist in two places**, both verified on 2026-09-16:

1. **`var/`**, the working copy.
2. **`pancakebot_data_archive_CLOSED_epoch516226_20260916.zip`** (674,535,415 bytes) in the operator's
   Downloads folder, under `pancakebot_archive_closed_20260916\`, with a README written for someone who
   has never seen this project. SHA-256 of the zip:

       c74d4ab20081c839106a4d04f2a3d553e5442fb1b5f07caf3348d3f012b9d8db

   It holds the five stores, the collection logs and health records, and the records recovered from
   the live bot's server before that server was destroyed. It is a zip64 container, confirmed in its
   raw bytes. It was verified by **extracting every member and re-hashing it**, not only by testing
   the container: all 174 files matched their checksums from before zipping, and the five extracted
   stores match section 6 exactly. A checksum list for every file is inside the zip.

**Both copies are on the same physical disk.** The zip protects against corruption or an accidental
change to `var/`. It does **not** protect against losing the drive. The kline history before about
2026-03-28 exists nowhere else, so if it matters, copy the zip to another device and check its
SHA-256 there.

Other copies that are **not** the archive:

- **`pancakebot_stores_pre_normalization_20260829.zip`** (617 MB), under
  `pancakebot_store_backup_20260829\` in the same Downloads folder. It is an **earlier, different
  state**: all five stores as of 2026-08-29, before their line endings were normalized from CRLF to LF.
  It lacks the final ~2.5 weeks, and its checksums will not match section 6. Its README originally
  said it was written as zip64; it was not. It never needed zip64 (no member reaches 4 GiB), so its
  data is not in question, and the README now carries that correction.
- `var/` also holds older working snapshots (`pre_repair_20260824/`, `stale_20260823/` and others),
  bringing it to about 15 GB in total. Every round in those snapshots is also in the final stores.
  Only the five files in section 6 are the archive.

## 8. How it closed

Collection ran daily under Windows Task Scheduler (a sync at 06:30 local, a watchdog at 09:00, both
also triggered at boot). The final days were healthy, with one incident recorded plainly:

- The manual closing sync (2026-09-16 04:20 UTC) and both scheduled syncs that ran to completion after
  it ended in `RESULT: SUCCESS`, and the watchdog reported healthy.
- **At about 09:30 UTC on 2026-09-16, a Windows Update restarted the machine three times in two
  minutes.** That killed four boot-triggered runs mid-flight (return code 1067, "process terminated
  unexpectedly"). Two sync logs and two watchdog logs from that minute have no result line. The same
  pattern happened once before, on 2026-09-10.
- Recovery was complete both times: the next sync succeeded within ten minutes and the regular daily
  sync after it succeeded too. No rounds were lost (section 6 shows no gaps beyond the known eight).
- At closure the health record read 29 attempts, 24 successes, **0 consecutive failures**, and no
  alert marker was outstanding.

Operational logs remain in `var/sync_logs/` and `var/watchdog_logs/`.

## 9. If anyone restarts collection

`python run.py --sync` with `THE_GRAPH_API_KEY` set extends all five stores from wherever they end.
Re-read section 3 first: the rounds between 516226 and the restart can be rebuilt from the chain, but
their klines cannot once they are more than ~171.6 days old.
