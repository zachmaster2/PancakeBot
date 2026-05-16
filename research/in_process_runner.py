"""In-process load-once + iterate-folds backtest driver.

Replaces the per-fold ``subprocess.run(python run.py --backtest)`` model
in ``research/sweep_harness.run_one``. Loads ``closed_rounds.jsonl`` and
the BTC/ETH/SOL kline files exactly once, then iterates fold-or-entry
specs in-process with a fresh ``InMemoryBankrollTracker`` +
``MomentumOnlyPipeline`` per fold.

Wall-clock target: ~70 s for the canonical 6 cutoff=2 folds + holdout
(load ~46 s + 6×0.3 s sim ≈ 50 s, vs 4m30 s under the per-fold subprocess
model). Memory: one ~1.0–1.3 GB process for the duration.

Driver semantics:

  experiment_specs : list[FoldSpec]

A FoldSpec is a dict with keys:

    name          : str                      # output sub-path
    kline_cutoff_seconds: int                      # 1, 2, ...
    epoch_start   : int | None
    epoch_end     : int | None
    strategy_overrides : dict (optional)     # nested-dict shape, see config.load_strategy_config_from_dict
    plot          : bool (optional, default False)  # generate equity_curves.png

The driver computes a unified kline-load extent that covers every
spec's ``(kline_cutoff_seconds, max_lookback)`` pair, then for each spec
slices the loaded array to that spec's exact window before running the
fold. Per-fold output (``trades.csv``, ``summary.json``) lands under
``<output_base_dir>/<name>/``.

CLI: ``python research/in_process_runner.py --spec <spec_json_path>``
where the JSON file holds ``{"experiment_specs": [...]}``. The CLI
mode is what the watchdog wraps via ``sweep_harness._spawn_with_rss_watchdog``.

This driver does NOT touch the production sync code or live-mode runtime;
it only consumes the post-rebuild dataset under ``var/*.jsonl``.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pancakebot import paths as _paths
from pancakebot.bankroll_tracker import InMemoryBankrollTracker
from pancakebot.config import (
    StrategyConfig,
    _DEFAULT_STRATEGY,
    load_strategy_config_from_dict,
)
from pancakebot.constants import BACKTEST_GAS_COST_BET_BNB
from pancakebot.market_data.round_store import ClosedRoundsStore
from pancakebot.settlement import settle_bet_against_closed_round
from pancakebot.strategy.momentum_gate import MomentumGateConfig
from pancakebot.strategy.momentum_pipeline import MomentumOnlyPipeline, _pools_from_bets
from pancakebot.types import Round
from pancakebot.util import InvariantError


_BTC_KLINES_PATH = REPO_ROOT / _paths.BTC_SPOT_PRICES_PATH
_ETH_KLINES_PATH = REPO_ROOT / _paths.ETH_SPOT_PRICES_PATH
_SOL_KLINES_PATH = REPO_ROOT / _paths.SOL_SPOT_PRICES_PATH
_CLOSED_ROUNDS_PATH = REPO_ROOT / _paths.CLOSED_ROUNDS_PATH

# Extended-data paths (consumed when use_extended_data=True, see run_experiment).
_EXT_BTC_KLINES_PATH = REPO_ROOT / _paths.EXTENDED_BTC_SPOT_PRICES_PATH
_EXT_ETH_KLINES_PATH = REPO_ROOT / _paths.EXTENDED_ETH_SPOT_PRICES_PATH
_EXT_SOL_KLINES_PATH = REPO_ROOT / _paths.EXTENDED_SOL_SPOT_PRICES_PATH
_EXT_CLOSED_ROUNDS_PATH = REPO_ROOT / _paths.EXTENDED_CLOSED_ROUNDS_PATH


# ---------- spec dataclasses ----------

@dataclass(frozen=True)
class FoldSpec:
    name: str
    kline_cutoff_seconds: int
    epoch_start: int | None
    epoch_end: int | None
    strategy_overrides: dict[str, Any] = field(default_factory=dict)
    plot: bool = False
    pool_cutoff_seconds: int = 6


# ---------- one-time load ----------

def _load_all_rounds(*, use_extended_data: bool = False) -> list[Round]:
    store = ClosedRoundsStore(str(_CLOSED_ROUNDS_PATH))
    rounds = list(store.iter_closed_rounds())
    rounds = [r for r in rounds if not r.failed]
    if use_extended_data and _EXT_CLOSED_ROUNDS_PATH.exists():
        ext_store = ClosedRoundsStore(str(_EXT_CLOSED_ROUNDS_PATH))
        ext_rounds = [r for r in ext_store.iter_closed_rounds() if not r.failed]
        # Extended rounds have epochs strictly older than canonical floor;
        # by construction no overlap with canonical. Prepend (older first).
        existing_epochs = {int(r.epoch) for r in rounds}
        ext_only = [r for r in ext_rounds if int(r.epoch) not in existing_epochs]
        ext_only.sort(key=lambda r: int(r.epoch))
        rounds = ext_only + rounds
    return rounds


def _compute_load_extent(
    resolved: list[tuple[FoldSpec, StrategyConfig]],
) -> tuple[int, int, int]:
    """Compute earliest_offset, latest_offset, load_count covering all specs.

    Takes the (spec, resolved_strategy_config) pairs already constructed
    once in run_experiment so we don't re-resolve per fold.

    For each spec:
      - cutoff seconds (e.g. 2)
      - max_lookback = max(strategy.gate.mtf_lookbacks) under the spec's overrides
      - per-spec offset range: [cutoff+1, cutoff+max_lookback+1]
    Unified:
      - earliest_offset = max(cutoff + max_lookback + 1)
      - latest_offset   = min(cutoff + 1)
      - load_count      = earliest_offset - latest_offset + 1
    """
    if not resolved:
        raise InvariantError("in_process_runner_empty_spec_list")
    cutoffs: list[int] = []
    max_endpoints: list[int] = []
    for spec, sc in resolved:
        ml = max(sc.gate.mtf_lookbacks)
        cutoffs.append(spec.kline_cutoff_seconds)
        max_endpoints.append(spec.kline_cutoff_seconds + ml + 1)
    earliest_offset = max(max_endpoints)
    latest_offset = min(c + 1 for c in cutoffs)
    load_count = earliest_offset - latest_offset + 1
    return earliest_offset, latest_offset, load_count


def _load_klines_unified(
    path: Path,
    *,
    earliest_offset: int,
    latest_offset: int,
    extended_path: Path | None = None,
) -> dict[int, list[list]]:
    """Load + slice each per-round record to the unified
    [open_ts: lock_at - earliest_offset, lock_at - latest_offset] window.

    Stored records: 300 candles oldest-first, position p has
    open_ts = lock_at - (301 - p) seconds.

    Slice (negative indexing):
      start_neg = -(earliest_offset - 1)
      end_neg   = -(latest_offset - 2)   if latest_offset >= 3 else None
                  (latest_offset == 2 means we want through the newest stored
                   candle; Python slice end-of-array is None.)

    If ``extended_path`` is provided and exists, also loads from that file.
    Extended records may carry a ``data_status`` field (``OK_FULL``, ``OK_PARTIAL``,
    ``MISSING``, etc.); records with empty or partial ``klines_1s`` are loaded
    as-is and the strategy's ``_validate_klines_raw`` call naturally skips
    rounds with insufficient data via ``gate_<sym>_insufficient`` skip reasons.
    Canonical records take precedence on epoch collisions (none expected
    by construction; extended fully precedes canonical's epoch range).
    """
    if not path.exists() and (extended_path is None or not extended_path.exists()):
        return {}
    if latest_offset < 2:
        raise InvariantError(f"in_process_runner_latest_offset_must_be_ge_2: {latest_offset}")
    start_neg = -(earliest_offset - 1)
    end_neg: int | None = None if latest_offset == 2 else -(latest_offset - 2)
    result: dict[int, list[list]] = {}

    def _ingest(p: Path) -> None:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("error") or rec.get("klines_1s") is None:
                    continue
                kl = rec["klines_1s"]
                if end_neg is None:
                    kl = kl[start_neg:]
                else:
                    kl = kl[start_neg:end_neg]
                # Don't overwrite a previously-loaded record (canonical wins).
                ep = int(rec["epoch"])
                if ep not in result:
                    result[ep] = kl

    # Canonical first (so it wins on any epoch collision).
    if path.exists():
        _ingest(path)
    if extended_path is not None and extended_path.exists():
        _ingest(extended_path)
    return result


def _slice_per_entry(
    kl_unified: list[list],
    *,
    kline_cutoff_seconds: int,
    max_lookback: int,
    earliest_offset: int,
) -> list[list]:
    """Extract the per-entry window from the unified loaded array.

    end_idx = earliest_offset - kline_cutoff_seconds  (exclusive end)
    start_idx = end_idx - (max_lookback + 1)
    Returns exactly (max_lookback + 1) candles.
    """
    end_idx = earliest_offset - kline_cutoff_seconds
    start_idx = end_idx - (max_lookback + 1)
    return kl_unified[start_idx:end_idx]


# ---------- strategy config resolution ----------

def _resolve_strategy_config(spec: FoldSpec) -> StrategyConfig:
    """Build StrategyConfig for a fold by merging spec.strategy_overrides
    into the canonical _DEFAULT_STRATEGY shape.

    The overrides dict has the same nested shape as the [strategy.*]
    TOML sections, e.g. ``{"gate": {"mtf_lookbacks": [4, 8, 16]}}``.
    Missing keys retain their _DEFAULT_STRATEGY values.
    """
    if not spec.strategy_overrides:
        return _DEFAULT_STRATEGY
    # The loader merges with defaults section-by-section already.
    return load_strategy_config_from_dict(spec.strategy_overrides)


# ---------- per-fold execution ----------

@dataclass(slots=True)
class _BacktestStats:
    num_bets: int = 0
    num_bets_bull: int = 0
    num_bets_bear: int = 0
    num_wins: int = 0
    num_wins_bull: int = 0
    num_wins_bear: int = 0
    gross_profit_bnb: float = 0.0
    gross_loss_bnb: float = 0.0
    skip_counts_by_reason: dict[str, int] = field(default_factory=dict)


def _safe_rate(num: int, den: int) -> float:
    return num / den if den > 0 else 0.0


def run_fold(
    *,
    spec: FoldSpec,
    strategy_cfg: StrategyConfig,
    all_rounds: list[Round],
    btc_unified: dict[int, list[list]],
    eth_unified: dict[int, list[list]],
    sol_unified: dict[int, list[list]],
    earliest_offset: int,
    output_base_dir: Path,
    initial_bankroll_bnb: float,
    treasury_fee_fraction: float,
    min_bet_amount_bnb: float,
) -> dict[str, Any]:
    """Run one fold in-process. Returns the summary dict.

    ``strategy_cfg`` is pre-resolved by ``run_experiment`` and passed in
    (vs re-resolving per fold).

    Per-iteration init creates a fresh InMemoryBankrollTracker and
    MomentumOnlyPipeline so no state leaks across folds.
    """
    max_lookback = max(strategy_cfg.gate.mtf_lookbacks)

    sim_rounds = [
        r for r in all_rounds
        if (spec.epoch_start is None or r.epoch >= spec.epoch_start)
        and (spec.epoch_end is None or r.epoch <= spec.epoch_end)
    ]
    if not sim_rounds:
        raise InvariantError(
            f"in_process_runner_no_rounds: {spec.name} "
            f"start={spec.epoch_start} end={spec.epoch_end}"
        )
    first_epoch = sim_rounds[0].epoch
    last_epoch = sim_rounds[-1].epoch

    # Slice per-fold kline windows from the unified load.
    # Each list comprehension references the loaded inner lists by ID --
    # no copy. The slice itself returns a new outer list per record.
    btc_klines = {
        ep: _slice_per_entry(
            kl, kline_cutoff_seconds=spec.kline_cutoff_seconds,
            max_lookback=max_lookback, earliest_offset=earliest_offset,
        )
        for ep, kl in btc_unified.items()
    }
    eth_klines = {
        ep: _slice_per_entry(
            kl, kline_cutoff_seconds=spec.kline_cutoff_seconds,
            max_lookback=max_lookback, earliest_offset=earliest_offset,
        )
        for ep, kl in eth_unified.items()
    }
    sol_klines = {
        ep: _slice_per_entry(
            kl, kline_cutoff_seconds=spec.kline_cutoff_seconds,
            max_lookback=max_lookback, earliest_offset=earliest_offset,
        )
        for ep, kl in sol_unified.items()
    }

    # Fresh per-fold pipeline state.
    gate_config = MomentumGateConfig(
        enabled=True,
        bnb_symbol="BNB-USDT",
        btc_symbol="BTC-USDT",
        eth_symbol="ETH-USDT",
        sol_symbol="SOL-USDT",
        kline_cutoff_seconds=spec.kline_cutoff_seconds,
        mtf_lookbacks=strategy_cfg.gate.mtf_lookbacks,
        mtf_min_return_threshold=strategy_cfg.gate.mtf_min_return_threshold,
    )
    bankroll_tracker = InMemoryBankrollTracker(
        initial_bankroll=initial_bankroll_bnb,
        drawdown_peak_window_days=strategy_cfg.risk.drawdown_peak_window_days,
        peak_mode=strategy_cfg.risk.drawdown_peak_mode,
    )
    pipeline = MomentumOnlyPipeline(
        config=gate_config,
        strategy_config=strategy_cfg,
        gate=None,
        kline_cutoff_seconds=spec.kline_cutoff_seconds,
        pool_cutoff_seconds=spec.pool_cutoff_seconds,
        min_bet_amount_bnb=min_bet_amount_bnb,
        treasury_fee_fraction=treasury_fee_fraction,
        bankroll_tracker=bankroll_tracker,
    )
    pipeline.refresh_btc_klines(btc_klines_by_epoch=btc_klines)
    pipeline.refresh_eth_klines(eth_klines_by_epoch=eth_klines)
    pipeline.refresh_sol_klines(sol_klines_by_epoch=sol_klines)
    # BNB never read by strategy; pass empty dict to satisfy the interface.
    pipeline.refresh_bnb_klines(bnb_klines_by_epoch={})

    # Output dir + trades.csv writer.
    run_dir = output_base_dir / spec.name
    run_dir.mkdir(parents=True, exist_ok=True)
    trades_path = run_dir / "trades.csv"
    summary_path = run_dir / "summary.json"

    bankroll = initial_bankroll_bnb
    stats = _BacktestStats()

    t_sim_start = time.perf_counter()
    with open(trades_path, "w", newline="", encoding="utf-8") as trades_f:
        trades_w = csv.writer(trades_f)
        trades_w.writerow([
            "epoch", "action", "skip_reason", "direction",
            "bet_size_bnb", "profit_bnb", "bankroll_bnb",
        ])
        for round_t in sim_rounds:
            decision = pipeline.decide_open_round(round_t=round_t)

            profit = 0.0
            if decision.action == "BET" and decision.bet_size_bnb > 0.0:
                bet_side = decision.bet_side
                if bet_side not in ("Bull", "Bear"):
                    raise InvariantError("backtest_bet_side_invalid")

                bankroll -= decision.bet_size_bnb + BACKTEST_GAS_COST_BET_BNB
                outcome = settle_bet_against_closed_round(
                    bet_bnb=decision.bet_size_bnb,
                    bet_side=bet_side,
                    round_closed=round_t,
                    treasury_fee_fraction=treasury_fee_fraction,
                )
                bankroll += outcome.credit_bnb
                profit = outcome.credit_bnb - decision.bet_size_bnb - BACKTEST_GAS_COST_BET_BNB

                stats.num_bets += 1
                if bet_side == "Bull":
                    stats.num_bets_bull += 1
                else:
                    stats.num_bets_bear += 1

                if outcome.outcome == "win":
                    stats.num_wins += 1
                    if bet_side == "Bull":
                        stats.num_wins_bull += 1
                    else:
                        stats.num_wins_bear += 1

                if profit > 0.0:
                    stats.gross_profit_bnb += profit
                elif profit < 0.0:
                    stats.gross_loss_bnb += -profit

            else:
                key = decision.skip_reason or "unknown_skip_reason"
                stats.skip_counts_by_reason[key] = stats.skip_counts_by_reason.get(key, 0) + 1

            trades_w.writerow([
                round_t.epoch,
                decision.action,
                decision.skip_reason or "",
                decision.bet_side or "",
                decision.bet_size_bnb,
                profit,
                bankroll,
            ])

            pipeline.record_settlement(bankroll=bankroll, start_at=int(round_t.start_at))
            pipeline.settle_closed_rounds(rounds=[round_t])

    elapsed_sim = time.perf_counter() - t_sim_start

    total_rounds = len(sim_rounds)
    total_skips = total_rounds - stats.num_bets
    net_pnl = bankroll - initial_bankroll_bnb
    win_rate = _safe_rate(stats.num_wins, stats.num_bets)
    bet_rate = _safe_rate(stats.num_bets, total_rounds)

    skip_detail = dict(sorted(stats.skip_counts_by_reason.items(), key=lambda x: -x[1]))
    summary = {
        "backtest_round_count": total_rounds,
        "first_epoch": first_epoch,
        "last_epoch": last_epoch,
        "initial_bankroll_bnb": initial_bankroll_bnb,
        "final_bankroll_bnb": bankroll,
        "net_pnl_bnb": net_pnl,
        "num_bets": stats.num_bets,
        "num_bets_bull": stats.num_bets_bull,
        "num_bets_bear": stats.num_bets_bear,
        "num_wins": stats.num_wins,
        "num_wins_bull": stats.num_wins_bull,
        "num_wins_bear": stats.num_wins_bear,
        "num_skips": total_skips,
        "win_rate": win_rate,
        "bet_rate": bet_rate,
        "gross_profit_bnb": stats.gross_profit_bnb,
        "gross_loss_bnb": stats.gross_loss_bnb,
        "skip_counts_by_reason": skip_detail,
        "elapsed_sim_seconds": elapsed_sim,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    # Optional equity-curve plot. Default OFF in experiments to avoid
    # the ~150 MB matplotlib import on the hot path.
    if spec.plot:
        _plot_equity_curve(trades_path, run_dir / "equity_curves.png",
                           initial_bankroll_bnb, total_rounds)

    return summary


def _plot_equity_curve(
    trades_path: Path,
    out_path: Path,
    initial_bankroll: float,
    total_rounds: int,
) -> None:
    """Generate equity curve PNG from trades CSV. Lazy-import matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bankrolls = []
    with open(trades_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            bankrolls.append(float(row["bankroll_bnb"]))
    pnl = [b - initial_bankroll for b in bankrolls]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})
    axes[0].plot(range(len(pnl)), pnl, color="#E91E63", linewidth=1.5)
    axes[0].set_ylabel("Cumulative PnL (BNB)")
    axes[0].set_title(f"Backtest Equity Curve ({total_rounds} rounds)")
    axes[0].axhline(y=0, color="black", linewidth=0.5, linestyle="--")
    axes[0].grid(True, alpha=0.3)

    window = 500
    if len(bankrolls) > window:
        rolling = []
        for i in range(window, len(bankrolls)):
            rate = (bankrolls[i] - bankrolls[i - window]) / window * 2000
            rolling.append(rate)
        axes[1].plot(range(window, len(bankrolls)), rolling,
                     color="#E91E63", linewidth=1.0, alpha=0.8)
    axes[1].set_ylabel("Rolling PnL/2k (500-round window)")
    axes[1].set_xlabel("Round index")
    axes[1].axhline(y=0, color="red", linewidth=0.5, linestyle="--")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150)
    plt.close(fig)
    plt.close("all")


# ---------- experiment driver ----------

def run_experiment(
    *,
    experiment_specs: list[FoldSpec],
    output_base_dir: Path,
    initial_bankroll_bnb: float = 50.0,
    treasury_fee_fraction: float = 0.03,
    min_bet_amount_bnb: float | None = None,
    use_extended_data: bool = False,
) -> list[dict[str, Any]]:
    """Run all folds in one process. Loads klines once.

    Returns a list of per-fold summary dicts. Output files land under
    ``output_base_dir/<spec.name>/{trades.csv,summary.json}``.

    When ``use_extended_data=True`` (default False), the loader also reads
    rounds + klines from ``var/extended/`` (older epochs not in the canonical
    store; written by ``research/backfill_okx_extended.py`` with possibly
    partial/missing data). Default OFF preserves canonical bit-identical
    behavior. The strategy's per-symbol ``_validate_klines_raw`` check
    naturally skips rounds whose extended klines are insufficient (empty,
    too short, or boundary-misaligned).
    """
    if min_bet_amount_bnb is None:
        # Default to the on-chain contract minimum from cached constants.
        # Fall back to 0.001 BNB if the constants file isn't present
        # (research-only path; live always loads contract_constants.json).
        try:
            from pancakebot.market_data.contract_constants import load_contract_constants
            cc = load_contract_constants()
            min_bet_amount_bnb = float(cc.min_bet_amount_bnb)
        except Exception:  # noqa: BLE001 -- fall back for offline use
            min_bet_amount_bnb = 0.001

    print(f"Loading closed_rounds + klines (one-time)..."
          f"{' (use_extended_data=True)' if use_extended_data else ''}", flush=True)
    t0 = time.perf_counter()
    all_rounds = _load_all_rounds(use_extended_data=use_extended_data)
    # Resolve StrategyConfig once per spec (not per fold-iteration).
    resolved: list[tuple[FoldSpec, StrategyConfig]] = [
        (spec, _resolve_strategy_config(spec)) for spec in experiment_specs
    ]
    earliest_offset, latest_offset, load_count = _compute_load_extent(resolved)
    print(f"  {len(all_rounds)} valid rounds; load_extent="
          f"[{latest_offset}..{earliest_offset}] ({load_count} candles/record)",
          flush=True)

    btc_ext = _EXT_BTC_KLINES_PATH if use_extended_data else None
    eth_ext = _EXT_ETH_KLINES_PATH if use_extended_data else None
    sol_ext = _EXT_SOL_KLINES_PATH if use_extended_data else None

    btc_unified = _load_klines_unified(
        _BTC_KLINES_PATH, earliest_offset=earliest_offset, latest_offset=latest_offset,
        extended_path=btc_ext,
    )
    print(f"  BTC: {len(btc_unified)} epochs", flush=True)
    eth_unified = _load_klines_unified(
        _ETH_KLINES_PATH, earliest_offset=earliest_offset, latest_offset=latest_offset,
        extended_path=eth_ext,
    )
    print(f"  ETH: {len(eth_unified)} epochs", flush=True)
    sol_unified = _load_klines_unified(
        _SOL_KLINES_PATH, earliest_offset=earliest_offset, latest_offset=latest_offset,
        extended_path=sol_ext,
    )
    print(f"  SOL: {len(sol_unified)} epochs", flush=True)
    elapsed_load = time.perf_counter() - t0
    print(f"  load_elapsed={elapsed_load:.1f}s", flush=True)

    summaries: list[dict[str, Any]] = []
    for spec, strategy_cfg in resolved:
        t_fold = time.perf_counter()
        print(f"\nfold {spec.name} cutoff={spec.kline_cutoff_seconds} "
              f"epochs=[{spec.epoch_start}..{spec.epoch_end}]", flush=True)
        summary = run_fold(
            spec=spec,
            strategy_cfg=strategy_cfg,
            all_rounds=all_rounds,
            btc_unified=btc_unified,
            eth_unified=eth_unified,
            sol_unified=sol_unified,
            earliest_offset=earliest_offset,
            output_base_dir=output_base_dir,
            initial_bankroll_bnb=initial_bankroll_bnb,
            treasury_fee_fraction=treasury_fee_fraction,
            min_bet_amount_bnb=min_bet_amount_bnb,
        )
        summaries.append({"spec_name": spec.name, "summary": summary})
        print(f"  bets={summary['num_bets']} wins={summary['num_wins']} "
              f"wr={summary['win_rate']:.4f} pnl={summary['net_pnl_bnb']:+.4f} "
              f"elapsed={time.perf_counter()-t_fold:.2f}s", flush=True)

    return summaries


# ---------- CLI entry ----------

def _spec_from_dict(d: dict[str, Any]) -> FoldSpec:
    return FoldSpec(
        name=str(d["name"]),
        kline_cutoff_seconds=int(d["kline_cutoff_seconds"]),
        epoch_start=d.get("epoch_start"),
        epoch_end=d.get("epoch_end"),
        strategy_overrides=d.get("strategy_overrides", {}) or {},
        plot=bool(d.get("plot", False)),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=str, required=True,
                        help="Path to JSON file with {experiment_specs: [...]}")
    parser.add_argument("--output-base-dir", type=str,
                        default=str(REPO_ROOT / "var" / "sweep"),
                        help="Where per-fold output dirs land")
    parser.add_argument("--initial-bankroll-bnb", type=float, default=50.0)
    parser.add_argument("--treasury-fee-fraction", type=float, default=0.03)
    parser.add_argument("--use-extended-data", action="store_true",
                        help="Also load older epochs from var/extended/ "
                             "(default OFF preserves canonical bit-identical behavior)")
    args = parser.parse_args()

    spec_path = Path(args.spec)
    if not spec_path.exists():
        print(f"[FAIL] spec file missing: {spec_path}", file=sys.stderr)
        return 1
    raw = json.loads(spec_path.read_text(encoding="utf-8"))
    specs = [_spec_from_dict(d) for d in raw["experiment_specs"]]
    if not specs:
        print("[FAIL] empty experiment_specs", file=sys.stderr)
        return 1

    summaries = run_experiment(
        experiment_specs=specs,
        output_base_dir=Path(args.output_base_dir),
        initial_bankroll_bnb=args.initial_bankroll_bnb,
        treasury_fee_fraction=args.treasury_fee_fraction,
        use_extended_data=args.use_extended_data,
    )

    # Persist aggregated summary alongside the per-fold outputs.
    Path(args.output_base_dir).mkdir(parents=True, exist_ok=True)
    print(f"\n=== Aggregate ===", flush=True)
    print(json.dumps([
        {"spec_name": s["spec_name"], **{
            k: s["summary"].get(k) for k in
            ("num_bets", "num_wins", "win_rate", "bet_rate", "net_pnl_bnb",
             "first_epoch", "last_epoch")
        }} for s in summaries
    ], indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
