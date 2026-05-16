"""Parameter-sweep harness for PancakeBot backtest.

Runs the production `python run.py --backtest` in a subprocess, one configuration
at a time, against N epoch-range folds. Captures each run's summary.json and
trades.csv under `var/sweep/<config_name>/fold_<fold_name>/`.

Configs are specified as a list of `{name, overrides}` dicts where overrides is
a flat dict of dotted-path TOML keys (e.g. `"strategy.btc_2of3.enabled"`) to
their new values. Folds are a list of `{name, epoch_start, epoch_end}` dicts.

Uses tomlkit to preserve formatting and comments across override writes so the
generated configs remain human-readable.

Usage (from repo root):
    python research/sweep_harness.py

The module exports:
    apply_overrides(base_config_path, overrides, output_path) -> None
    run_one(name, overrides, epoch_start, epoch_end, output_base_dir) -> dict
    run_sweep(configs, folds, output_base_dir) -> list
    equivalence_check() -> bool

`main()` runs `equivalence_check()` as a standalone self-test.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import psutil
import tomlkit

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_CONFIG_PATH = REPO_ROOT / "config.toml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "var/sweep"
BACKTEST_OUTPUT_DIR = REPO_ROOT / "var/backtest"
SUBPROCESS_TIMEOUT_S = 600  # 10 minutes is generous; default backtest runs in ~1s

# RSS watchdog (added 2026-04-26 after Phase-7 OOM incident).
# Single-fold dry test on f1/cutoff=2 measured peak RSS = 1.85 GB
# (the closed_rounds.jsonl load + 3 trimmed kline dicts + Python
# interpreter sum higher than the 1.0–1.3 GB initial forecast — the
# closed_rounds bets-per-round overhead was under-budgeted). 3 GB
# ceiling gives ~1.15 GB headroom over observed peak while staying
# well below the ~10 GB OOM trigger that bricked the host this
# morning. Polling at 0.5s catches spikes before they overshoot by
# more than ~1 GB at worst-case Python allocation rates.
_RSS_CEILING_BYTES = 3 * 1024 * 1024 * 1024
_RSS_POLL_INTERVAL_S = 0.5


def _spawn_with_rss_watchdog(
    cmd: list[str],
    *,
    cwd: str,
    name: str,
    run_dir: Path,
    timeout_s: int,
) -> int:
    """Spawn `cmd` as a subprocess with an RSS-watchdog daemon thread.

    Replaces `subprocess.run(capture_output=True)` for backtest
    invocations: stdout+stderr stream to disk (so the parent never
    buffers child output), and a daemon thread polls
    `psutil.Process(child_pid).memory_info().rss` (sum-with-children)
    every `_RSS_POLL_INTERVAL_S`. If RSS exceeds `_RSS_CEILING_BYTES`,
    the watchdog kills the child and the main thread raises
    `RuntimeError`. Always persists `run_dir/rss_peak.json` for
    post-run review (even on RSS-kill, timeout, or normal exit).

    Returns the child returncode on normal exit. Raises:
      - `RuntimeError("RSS limit exceeded: ...")` on RSS-kill
      - `RuntimeError("backtest timed out ...")` on wall-clock timeout
    """
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"

    # Mutable cells so the daemon thread can write back to main thread
    peak_rss_bytes = [0]
    killed_for_rss = [False]
    killed_for_timeout = [False]

    def _watchdog(child_pid: int) -> None:
        # Returns when the child exits, gets killed, or is no longer
        # accessible. Daemon thread, so dies with the parent.
        try:
            p = psutil.Process(child_pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return
        while True:
            try:
                rss = p.memory_info().rss
                # Defensive: sum any grandchildren too. Backtest doesn't
                # fork, but cheap insurance against future refactors.
                for c in p.children(recursive=True):
                    try:
                        rss += c.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                if rss > peak_rss_bytes[0]:
                    peak_rss_bytes[0] = rss
                if rss > _RSS_CEILING_BYTES:
                    killed_for_rss[0] = True
                    try:
                        p.kill()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                    return
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                return
            time.sleep(_RSS_POLL_INTERVAL_S)

    rc: int = -1
    try:
        with open(stdout_path, "w", encoding="utf-8") as out_f, \
             open(stderr_path, "w", encoding="utf-8") as err_f:
            proc = subprocess.Popen(
                cmd,
                cwd=cwd,
                stdout=out_f,
                stderr=err_f,
                text=True,
            )
            wd = threading.Thread(
                target=_watchdog, args=(proc.pid,), daemon=True,
            )
            wd.start()
            try:
                rc = proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                killed_for_timeout[0] = True
                try:
                    proc.kill()
                except Exception:  # noqa: BLE001 -- best-effort cleanup
                    pass
                try:
                    rc = proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    rc = -1
    finally:
        # Always persist rss_peak.json so the user can review even on failure.
        peak_bytes = peak_rss_bytes[0]
        (run_dir / "rss_peak.json").write_text(
            json.dumps({
                "fold": name,
                "peak_rss_bytes": int(peak_bytes),
                "peak_rss_gb": round(peak_bytes / (1024 ** 3), 4),
                "killed_for_rss": bool(killed_for_rss[0]),
                "killed_for_timeout": bool(killed_for_timeout[0]),
                "ceiling_bytes": int(_RSS_CEILING_BYTES),
                "ceiling_gb": round(_RSS_CEILING_BYTES / (1024 ** 3), 4),
                "poll_interval_s": _RSS_POLL_INTERVAL_S,
                "returncode": int(rc) if rc is not None else -1,
            }, indent=2),
            encoding="utf-8",
        )

    if killed_for_rss[0]:
        raise RuntimeError(
            f"RSS limit exceeded: peak={peak_rss_bytes[0] / (1024**3):.2f} GB > "
            f"{_RSS_CEILING_BYTES / (1024**3):.2f} GB on fold={name}"
        )
    if killed_for_timeout[0]:
        raise RuntimeError(
            f"backtest timed out after {timeout_s}s: name={name}"
        )
    return rc


def _set_dotted(doc: Any, dotted_key: str, value: Any) -> None:
    """Set doc[a][b][c] = value for dotted_key 'a.b.c'.

    Creates intermediate tables if missing. Raises KeyError if intermediate
    exists but isn't a table.
    """
    parts = dotted_key.split(".")
    target = doc
    for p in parts[:-1]:
        if p not in target:
            target[p] = tomlkit.table()
        sub = target[p]
        if not isinstance(sub, (dict, tomlkit.items.Table, tomlkit.items.InlineTable)):
            raise KeyError(f"override_path_not_table: {dotted_key} at {p}")
        target = sub
    target[parts[-1]] = value


def apply_overrides(
    base_config_path: Path,
    overrides: dict[str, Any],
    output_path: Path,
) -> None:
    """Read the TOML at base_config_path, apply overrides, write to output_path.

    overrides is a flat dict where keys are dotted paths and values are the new
    scalar values. Missing intermediate tables are created. Comments and
    formatting in the base file are preserved (via tomlkit).
    """
    text = base_config_path.read_text(encoding="utf-8")
    doc = tomlkit.parse(text)
    for key, value in overrides.items():
        _set_dotted(doc, key, value)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(tomlkit.dumps(doc), encoding="utf-8")


def run_one(
    name: str,
    overrides: dict[str, Any],
    epoch_start: int | None,
    epoch_end: int | None,
    output_base_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Run one backtest with the given overrides + epoch range.

    name is the output-subdirectory path under output_base_dir, e.g.
    "myconfig/fold_a". The harness writes a temp config, invokes
    `python run.py --backtest --config <temp>`, and moves the outputs.

    Returns the parsed summary.json as a dict.
    """
    run_dir = output_base_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    temp_config = run_dir / "config.toml"

    # Build the full override set: user overrides + epoch-range injection.
    full_overrides = dict(overrides)
    if epoch_start is not None:
        full_overrides["backtest.epoch_start"] = int(epoch_start)
    if epoch_end is not None:
        full_overrides["backtest.epoch_end"] = int(epoch_end)

    apply_overrides(BASE_CONFIG_PATH, full_overrides, temp_config)

    # Invoke the production backtest via subprocess. Output streams
    # directly to disk (no parent-RAM buffering), and an RSS watchdog
    # kills the child if it exceeds _RSS_CEILING_BYTES (2 GB). Peak
    # RSS is persisted to run_dir/rss_peak.json for post-run review.
    cmd = [
        sys.executable,
        str(REPO_ROOT / "run.py"),
        "--config", str(temp_config),
        "--backtest",
    ]
    rc = _spawn_with_rss_watchdog(
        cmd,
        cwd=str(REPO_ROOT),
        name=name,
        run_dir=run_dir,
        timeout_s=SUBPROCESS_TIMEOUT_S,
    )
    if rc != 0:
        # Read tails of the on-disk logs for the error message.
        stdout_path = run_dir / "stdout.log"
        stderr_path = run_dir / "stderr.log"
        stdout_tail = (
            stdout_path.read_text(encoding="utf-8")[-2000:]
            if stdout_path.exists() else ""
        )
        stderr_tail = (
            stderr_path.read_text(encoding="utf-8")[-2000:]
            if stderr_path.exists() else ""
        )
        raise RuntimeError(
            f"backtest_subprocess_failed: name={name} rc={rc}\n"
            f"stdout:\n{stdout_tail}\n"
            f"stderr:\n{stderr_tail}"
        )

    # Move summary.json and trades.csv from var/backtest to the sweep dir.
    for fname in ("summary.json", "trades.csv", "equity_curves.png"):
        src = BACKTEST_OUTPUT_DIR / fname
        if src.exists():
            dst = run_dir / fname
            shutil.move(str(src), str(dst))

    summary_path = run_dir / "summary.json"
    with summary_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_sweep(
    configs: list[dict[str, Any]],
    folds: list[dict[str, Any]],
    output_base_dir: Path = DEFAULT_OUTPUT_DIR,
) -> list[dict[str, Any]]:
    """Serial sweep over the cross-product of configs x folds.

    configs: [{"name": str, "overrides": dict}, ...]
    folds:   [{"name": str, "epoch_start": int|None, "epoch_end": int|None}, ...]

    Returns a list of result dicts with {config_name, fold_name, summary}.
    """
    results: list[dict[str, Any]] = []
    total = len(configs) * len(folds)
    idx = 0
    for cfg in configs:
        for fold in folds:
            idx += 1
            cname = cfg["name"]
            fname = fold["name"]
            run_name = f"{cname}/fold_{fname}"
            print(f"[{idx}/{total}] running {run_name} ...", flush=True)
            summary = run_one(
                name=run_name,
                overrides=cfg.get("overrides", {}),
                epoch_start=fold.get("epoch_start"),
                epoch_end=fold.get("epoch_end"),
                output_base_dir=output_base_dir,
            )
            results.append({
                "config_name": cname,
                "fold_name": fname,
                "summary": summary,
            })
            print(
                f"  [{idx}/{total}] {run_name}: "
                f"bets={summary.get('num_bets')} "
                f"wr={summary.get('win_rate'):.3%} "
                f"net_pnl={summary.get('net_pnl_bnb'):+.4f}",
                flush=True,
            )
    return results


def _content_hash(summary_path: Path) -> str:
    """Hash summary.json content excluding the elapsed_sim_seconds timing field."""
    with summary_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    obj.pop("elapsed_sim_seconds", None)
    return hashlib.md5(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def equivalence_check() -> bool:
    """Prove that harness-driven baseline == direct-invocation baseline.

    Step A: invoke `python run.py --backtest` directly (using config.toml);
            hash the resulting summary.json (minus timing field).
    Step B: run through the harness with empty overrides and no epoch range
            (identical semantics to a default backtest); hash that summary.
    Compare hashes. Return True iff identical.
    """
    print("=== equivalence_check ===", flush=True)

    # Step A: direct invocation.
    print("step A: direct `python run.py --backtest` ...", flush=True)
    direct_cmd = [sys.executable, str(REPO_ROOT / "run.py"), "--backtest"]
    proc = subprocess.run(
        direct_cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT_S,
    )
    if proc.returncode != 0:
        print(f"  direct invocation failed rc={proc.returncode}", flush=True)
        print(proc.stderr[-1000:], flush=True)
        return False

    direct_hash = _content_hash(BACKTEST_OUTPUT_DIR / "summary.json")
    print(f"  direct content hash: {direct_hash}", flush=True)

    # Step B: harness-driven baseline with empty overrides.
    print("step B: harness-driven baseline (empty overrides) ...", flush=True)
    harness_run_dir = DEFAULT_OUTPUT_DIR / "_equivalence_check"
    if harness_run_dir.exists():
        shutil.rmtree(harness_run_dir)
    summary = run_one(
        name="_equivalence_check",
        overrides={},
        epoch_start=None,
        epoch_end=None,
    )
    harness_hash = _content_hash(harness_run_dir / "summary.json")
    print(f"  harness content hash: {harness_hash}", flush=True)

    identical = (direct_hash == harness_hash)
    if identical:
        print(f"\n[OK] EQUIVALENCE PASSED: hashes match ({direct_hash})", flush=True)
    else:
        print("\n[FAIL] EQUIVALENCE FAILED: hashes differ", flush=True)
        print(f"  direct   : {direct_hash}", flush=True)
        print(f"  harness  : {harness_hash}", flush=True)
        # Dump a diff of key fields for debugging.
        with (BACKTEST_OUTPUT_DIR / "summary.json").open() as f:
            pass  # direct already moved? No — direct run wrote to var/backtest, harness did not touch it between the two runs.
        # Actually, step B may have moved files. Skip diff for now.
        print("  (summary files present at:", flush=True)
        print(f"    direct  -> {BACKTEST_OUTPUT_DIR / 'summary.json'}", flush=True)
        print(f"    harness -> {harness_run_dir / 'summary.json'}", flush=True)
        print("   strip elapsed_sim_seconds and diff to debug)", flush=True)
    print(f"num_bets={summary.get('num_bets')} net_pnl={summary.get('net_pnl_bnb'):+.4f}",
          flush=True)
    return identical


_BTC2OF3_R3_FOLDS: list[dict[str, Any]] = [
    {"name": "fold1", "epoch_start": 437562, "epoch_end": 444866},
    {"name": "fold2", "epoch_start": 444867, "epoch_end": 452171},
    {"name": "fold3", "epoch_start": 452172, "epoch_end": 459476},
    {"name": "fold4", "epoch_start": 459477, "epoch_end": 466781},
    {"name": "fold5", "epoch_start": 466782, "epoch_end": 474086},
]

_BTC2OF3_R3_CONFIGS: list[dict[str, Any]] = [
    {"name": "baseline", "overrides": {
        "strategy.btc_2of3.enabled": False,
    }},
    {"name": "default", "overrides": {
        "strategy.btc_2of3.enabled": True,
        "strategy.btc_2of3.signal.threshold": 0.0003,
        "strategy.btc_2of3.filter.min_pool_bnb": 2.0,
        "strategy.btc_2of3.filter.min_payout": 2.0,
    }},
    {"name": "thresh_0.0002", "overrides": {
        "strategy.btc_2of3.enabled": True,
        "strategy.btc_2of3.signal.threshold": 0.0002,
        "strategy.btc_2of3.filter.min_pool_bnb": 2.0,
        "strategy.btc_2of3.filter.min_payout": 2.0,
    }},
    {"name": "thresh_0.0005", "overrides": {
        "strategy.btc_2of3.enabled": True,
        "strategy.btc_2of3.signal.threshold": 0.0005,
        "strategy.btc_2of3.filter.min_pool_bnb": 2.0,
        "strategy.btc_2of3.filter.min_payout": 2.0,
    }},
    {"name": "pool_1.5", "overrides": {
        "strategy.btc_2of3.enabled": True,
        "strategy.btc_2of3.signal.threshold": 0.0003,
        "strategy.btc_2of3.filter.min_pool_bnb": 1.5,
        "strategy.btc_2of3.filter.min_payout": 2.0,
    }},
    {"name": "pool_3.0", "overrides": {
        "strategy.btc_2of3.enabled": True,
        "strategy.btc_2of3.signal.threshold": 0.0003,
        "strategy.btc_2of3.filter.min_pool_bnb": 3.0,
        "strategy.btc_2of3.filter.min_payout": 2.0,
    }},
    {"name": "payout_1.5", "overrides": {
        "strategy.btc_2of3.enabled": True,
        "strategy.btc_2of3.signal.threshold": 0.0003,
        "strategy.btc_2of3.filter.min_pool_bnb": 2.0,
        "strategy.btc_2of3.filter.min_payout": 1.5,
    }},
    {"name": "payout_2.5", "overrides": {
        "strategy.btc_2of3.enabled": True,
        "strategy.btc_2of3.signal.threshold": 0.0003,
        "strategy.btc_2of3.filter.min_pool_bnb": 2.0,
        "strategy.btc_2of3.filter.min_payout": 2.5,
    }},
]


def _sign(x: float) -> str:
    if x > 0: return "+"
    if x < 0: return "-"
    return "0"


def _classify_config(
    per_fold: list[dict[str, Any]],
    baseline_total_pnl: float | None,
) -> str:
    """Return one of [PROMOTE] / [NESTED_CV] / [REJECT_UNDERTRADE] / [REJECT_PNL]."""
    # Per-fold trade-count requirement (memory: 100+ trades minimum per fold).
    min_bets = min(f["num_bets"] for f in per_fold)
    if min_bets < 100:
        return "[REJECT_UNDERTRADE]"
    total_pnl = sum(f["net_pnl_bnb"] for f in per_fold)
    if baseline_total_pnl is not None and total_pnl <= baseline_total_pnl:
        return "[REJECT_PNL]"
    pos_count = sum(1 for f in per_fold if f["net_pnl_bnb"] > 0)
    if pos_count == 5:
        return "[PROMOTE]"
    if pos_count == 4:
        return "[NESTED_CV]"
    return "[REJECT_PNL]"


def run_btc2of3_r3_sweep() -> int:
    """R3: 8 configs × 5 folds = 40 runs against the 2-of-3 parameter space.

    Output: var/sweep/btc2of3_r3/. Prints summary table with promotion flags.
    """
    import time
    t0 = time.perf_counter()
    out_dir = DEFAULT_OUTPUT_DIR / "btc2of3_r3"
    if out_dir.exists():
        shutil.rmtree(out_dir)

    print(f"=== btc2of3 R3 sweep: {len(_BTC2OF3_R3_CONFIGS)} configs × "
          f"{len(_BTC2OF3_R3_FOLDS)} folds = "
          f"{len(_BTC2OF3_R3_CONFIGS) * len(_BTC2OF3_R3_FOLDS)} runs ===",
          flush=True)

    results = run_sweep(_BTC2OF3_R3_CONFIGS, _BTC2OF3_R3_FOLDS, output_base_dir=out_dir)
    elapsed = time.perf_counter() - t0

    # Aggregate per config.
    per_config: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        per_config.setdefault(r["config_name"], []).append({
            "fold": r["fold_name"],
            "num_bets": r["summary"].get("num_bets", 0),
            "num_wins": r["summary"].get("num_wins", 0),
            "net_pnl_bnb": r["summary"].get("net_pnl_bnb", 0.0),
        })

    baseline_total = sum(
        f["net_pnl_bnb"] for f in per_config.get("baseline", [])
    )

    print("\n" + "=" * 110, flush=True)
    print(f"R3 SWEEP SUMMARY  (baseline total PnL: {baseline_total:+.4f} BNB, "
          f"elapsed {elapsed:.1f}s)", flush=True)
    print("=" * 110, flush=True)
    hdr = f"{'config':<18} {'total_bets':>10} {'min_bets':>9} {'WR':>7} {'total_pnl':>11}  {'f1':>4}{'f2':>4}{'f3':>4}{'f4':>4}{'f5':>4}  {'+cnt':>5}  flag"
    print(hdr, flush=True)
    print("-" * 110, flush=True)

    summary_rows = []
    for cfg in _BTC2OF3_R3_CONFIGS:
        name = cfg["name"]
        folds = per_config.get(name, [])
        # Keep order matching _BTC2OF3_R3_FOLDS
        folds_by_name = {f["fold"]: f for f in folds}
        ordered = [folds_by_name[fd["name"]] for fd in _BTC2OF3_R3_FOLDS]
        total_bets = sum(f["num_bets"] for f in ordered)
        total_wins = sum(f["num_wins"] for f in ordered)
        total_pnl = sum(f["net_pnl_bnb"] for f in ordered)
        wr = (total_wins / total_bets) if total_bets > 0 else 0.0
        min_bets = min(f["num_bets"] for f in ordered)
        pos_count = sum(1 for f in ordered if f["net_pnl_bnb"] > 0)
        signs = "".join(f"{_sign(f['net_pnl_bnb']):>4}" for f in ordered)

        if name == "baseline":
            flag = "[BASELINE]"
        else:
            flag = _classify_config(ordered, baseline_total)

        print(f"{name:<18} {total_bets:>10} {min_bets:>9} {wr:>6.1%} "
              f"{total_pnl:>+11.4f}  {signs}  {pos_count:>5}  {flag}",
              flush=True)
        summary_rows.append({
            "config": name,
            "total_bets": total_bets,
            "min_bets_per_fold": min_bets,
            "win_rate": wr,
            "total_net_pnl_bnb": total_pnl,
            "positive_folds": pos_count,
            "per_fold": ordered,
            "flag": flag,
        })

    print("=" * 110, flush=True)

    # Any configs flagged for promotion or nested CV?
    hits = [r for r in summary_rows if r["flag"] in ("[PROMOTE]", "[NESTED_CV]")]
    if hits:
        print(f"\n{len(hits)} candidate(s):")
        for r in hits:
            print(f"  {r['flag']}  {r['config']}: "
                  f"{r['total_bets']} bets, "
                  f"WR {r['win_rate']:.1%}, "
                  f"PnL {r['total_net_pnl_bnb']:+.4f} BNB, "
                  f"{r['positive_folds']}/5 positive folds")
    else:
        print("\nNo candidates meet promotion criteria. All configs rejected.")

    # Persist aggregated summary.
    summary_path = out_dir / "sweep_summary.json"
    summary_path.write_text(
        json.dumps({
            "elapsed_seconds": round(elapsed, 2),
            "baseline_total_pnl_bnb": baseline_total,
            "rows": summary_rows,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"\nPersisted aggregate summary: {summary_path}")
    return 0


_VOLATILITY_R4_FOLDS: list[dict[str, Any]] = [
    {"name": "fold1", "epoch_start": 437562, "epoch_end": 444866},
    {"name": "fold2", "epoch_start": 444867, "epoch_end": 452171},
    {"name": "fold3", "epoch_start": 452172, "epoch_end": 459476},
    {"name": "fold4", "epoch_start": 459477, "epoch_end": 466781},
    {"name": "fold5", "epoch_start": 466782, "epoch_end": 474086},
]

_VOLATILITY_R4_CONFIGS: list[dict[str, Any]] = [
    {"name": "baseline", "overrides": {
        "strategy.btc_primary.volatility_filter.enabled": False,
    }},
    # window_candles=16 (strict-15s semantics)
    {"name": "w16_mr_0.0001", "overrides": {
        "strategy.btc_primary.volatility_filter.enabled": True,
        "strategy.btc_primary.volatility_filter.window_candles": 16,
        "strategy.btc_primary.volatility_filter.min_range": 0.0001,
    }},
    {"name": "w16_mr_0.0002", "overrides": {
        "strategy.btc_primary.volatility_filter.enabled": True,
        "strategy.btc_primary.volatility_filter.window_candles": 16,
        "strategy.btc_primary.volatility_filter.min_range": 0.0002,
    }},
    {"name": "w16_mr_0.0003", "overrides": {
        "strategy.btc_primary.volatility_filter.enabled": True,
        "strategy.btc_primary.volatility_filter.window_candles": 16,
        "strategy.btc_primary.volatility_filter.min_range": 0.0003,
    }},
    {"name": "w16_mr_0.0005", "overrides": {
        "strategy.btc_primary.volatility_filter.enabled": True,
        "strategy.btc_primary.volatility_filter.window_candles": 16,
        "strategy.btc_primary.volatility_filter.min_range": 0.0005,
    }},
    {"name": "w16_mr_0.001", "overrides": {
        "strategy.btc_primary.volatility_filter.enabled": True,
        "strategy.btc_primary.volatility_filter.window_candles": 16,
        "strategy.btc_primary.volatility_filter.min_range": 0.001,
    }},
    # window_candles=31 (full close buffer)
    {"name": "w31_mr_0.0001", "overrides": {
        "strategy.btc_primary.volatility_filter.enabled": True,
        "strategy.btc_primary.volatility_filter.window_candles": 31,
        "strategy.btc_primary.volatility_filter.min_range": 0.0001,
    }},
    {"name": "w31_mr_0.0002", "overrides": {
        "strategy.btc_primary.volatility_filter.enabled": True,
        "strategy.btc_primary.volatility_filter.window_candles": 31,
        "strategy.btc_primary.volatility_filter.min_range": 0.0002,
    }},
    {"name": "w31_mr_0.0003", "overrides": {
        "strategy.btc_primary.volatility_filter.enabled": True,
        "strategy.btc_primary.volatility_filter.window_candles": 31,
        "strategy.btc_primary.volatility_filter.min_range": 0.0003,
    }},
    {"name": "w31_mr_0.0005", "overrides": {
        "strategy.btc_primary.volatility_filter.enabled": True,
        "strategy.btc_primary.volatility_filter.window_candles": 31,
        "strategy.btc_primary.volatility_filter.min_range": 0.0005,
    }},
    {"name": "w31_mr_0.001", "overrides": {
        "strategy.btc_primary.volatility_filter.enabled": True,
        "strategy.btc_primary.volatility_filter.window_candles": 31,
        "strategy.btc_primary.volatility_filter.min_range": 0.001,
    }},
]


def run_volatility_r4_sweep() -> int:
    """R4: 11 configs × 5 folds = 55 runs against the BTC volatility filter.

    Output: var/sweep/volatility_r4/. Prints summary table with promotion flags
    plus a w16 vs w31 comparison block.
    """
    import time
    t0 = time.perf_counter()
    out_dir = DEFAULT_OUTPUT_DIR / "volatility_r4"
    if out_dir.exists():
        shutil.rmtree(out_dir)

    print(f"=== volatility R4 sweep: {len(_VOLATILITY_R4_CONFIGS)} configs × "
          f"{len(_VOLATILITY_R4_FOLDS)} folds = "
          f"{len(_VOLATILITY_R4_CONFIGS) * len(_VOLATILITY_R4_FOLDS)} runs ===",
          flush=True)

    results = run_sweep(_VOLATILITY_R4_CONFIGS, _VOLATILITY_R4_FOLDS, output_base_dir=out_dir)
    elapsed = time.perf_counter() - t0

    # Aggregate per config.
    per_config: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        per_config.setdefault(r["config_name"], []).append({
            "fold": r["fold_name"],
            "num_bets": r["summary"].get("num_bets", 0),
            "num_wins": r["summary"].get("num_wins", 0),
            "net_pnl_bnb": r["summary"].get("net_pnl_bnb", 0.0),
        })

    baseline_total = sum(
        f["net_pnl_bnb"] for f in per_config.get("baseline", [])
    )

    print("\n" + "=" * 110, flush=True)
    print(f"R4 SWEEP SUMMARY  (baseline total PnL: {baseline_total:+.4f} BNB, "
          f"elapsed {elapsed:.1f}s)", flush=True)
    print("=" * 110, flush=True)
    hdr = f"{'config':<18} {'total_bets':>10} {'min_bets':>9} {'WR':>7} {'total_pnl':>11}  {'f1':>4}{'f2':>4}{'f3':>4}{'f4':>4}{'f5':>4}  {'+cnt':>5}  flag"
    print(hdr, flush=True)
    print("-" * 110, flush=True)

    summary_rows = []
    for cfg in _VOLATILITY_R4_CONFIGS:
        name = cfg["name"]
        folds = per_config.get(name, [])
        folds_by_name = {f["fold"]: f for f in folds}
        ordered = [folds_by_name[fd["name"]] for fd in _VOLATILITY_R4_FOLDS]
        total_bets = sum(f["num_bets"] for f in ordered)
        total_wins = sum(f["num_wins"] for f in ordered)
        total_pnl = sum(f["net_pnl_bnb"] for f in ordered)
        wr = (total_wins / total_bets) if total_bets > 0 else 0.0
        min_bets = min(f["num_bets"] for f in ordered)
        pos_count = sum(1 for f in ordered if f["net_pnl_bnb"] > 0)
        signs = "".join(f"{_sign(f['net_pnl_bnb']):>4}" for f in ordered)

        if name == "baseline":
            flag = "[BASELINE]"
        else:
            flag = _classify_config(ordered, baseline_total)

        print(f"{name:<18} {total_bets:>10} {min_bets:>9} {wr:>6.1%} "
              f"{total_pnl:>+11.4f}  {signs}  {pos_count:>5}  {flag}",
              flush=True)
        summary_rows.append({
            "config": name,
            "total_bets": total_bets,
            "min_bets_per_fold": min_bets,
            "win_rate": wr,
            "total_net_pnl_bnb": total_pnl,
            "positive_folds": pos_count,
            "per_fold": ordered,
            "flag": flag,
        })

    print("=" * 110, flush=True)

    # w16 vs w31 paired comparison -- same min_range, different window.
    print("\nw16 vs w31 paired comparison (total PnL, BNB):")
    print("-" * 72)
    print(f"{'min_range':>10}  {'w16_pnl':>10}  {'w31_pnl':>10}  {'delta':>10}  {'w16_bets':>9} {'w31_bets':>9}")
    print("-" * 72)
    by_name = {r["config"]: r for r in summary_rows}
    for mr_tag in ["0.0001", "0.0002", "0.0003", "0.0005", "0.001"]:
        a = by_name.get(f"w16_mr_{mr_tag}")
        b = by_name.get(f"w31_mr_{mr_tag}")
        if a and b:
            delta = a["total_net_pnl_bnb"] - b["total_net_pnl_bnb"]
            print(f"{mr_tag:>10}  {a['total_net_pnl_bnb']:>+10.4f}  "
                  f"{b['total_net_pnl_bnb']:>+10.4f}  {delta:>+10.4f}  "
                  f"{a['total_bets']:>9} {b['total_bets']:>9}")

    hits = [r for r in summary_rows if r["flag"] in ("[PROMOTE]", "[NESTED_CV]")]
    if hits:
        print(f"\n{len(hits)} candidate(s):")
        for r in hits:
            print(f"  {r['flag']}  {r['config']}: "
                  f"{r['total_bets']} bets, "
                  f"WR {r['win_rate']:.1%}, "
                  f"PnL {r['total_net_pnl_bnb']:+.4f} BNB, "
                  f"{r['positive_folds']}/5 positive folds")
    else:
        print("\nNo candidates meet promotion criteria. All configs rejected.")

    summary_path = out_dir / "sweep_summary.json"
    summary_path.write_text(
        json.dumps({
            "elapsed_seconds": round(elapsed, 2),
            "baseline_total_pnl_bnb": baseline_total,
            "rows": summary_rows,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"\nPersisted aggregate summary: {summary_path}")
    return 0


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sweep",
        choices=["equivalence", "btc2of3_r3", "volatility_r4"],
        default="equivalence",
        help="Which sweep/self-test to run.",
    )
    args = parser.parse_args()
    if args.sweep == "equivalence":
        ok = equivalence_check()
        return 0 if ok else 1
    if args.sweep == "btc2of3_r3":
        return run_btc2of3_r3_sweep()
    if args.sweep == "volatility_r4":
        return run_volatility_r4_sweep()
    return 1


if __name__ == "__main__":
    sys.exit(main())
