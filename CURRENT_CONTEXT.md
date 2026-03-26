# Current Context

## Objective

See [PROJECT_GOALS_AND_WORKFLOW.md](/C:/Users/zking/Documents/GitHub/PancakeBot/PROJECT_GOALS_AND_WORKFLOW.md).

Short version:

1. Maximize net `BNB` per `500` rounds on recent history.
2. Expect regime drift.
3. Assume smaller windows can have different winners than the overall baseline.
4. Treat the strategy as continuously evolving, not frozen.

## Repo State

1. V1 (`PancakeBot/`) is the current reference repo for behavior, research tooling, and promoted baselines.
2. V2 (`../PancakeBotV2/`) is a cleaner redesign attempt, but it is not yet parity-safe.
3. V2 should move toward approved design specs, not more patch-driven fixes.

## Working Assumptions

1. Overall best historical strategy is only a reference point, not the final answer.
2. Recent-window winners matter.
3. Stability and adaptability both matter.
4. Runtime and logging design need to be coherent, not just functional.
5. New threads should bootstrap from startup docs, not from broad rediscovery.
6. Block-based meta-strategy selection is now implemented offline in `inspection/`; see [META_STRATEGY_PROBLEM.md](/C:/Users/zking/Documents/GitHub/PancakeBot/docs/META_STRATEGY_PROBLEM.md), [build_meta_strategy_dataset.py](/C:/Users/zking/Documents/GitHub/PancakeBot/inspection/build_meta_strategy_dataset.py), and [run_meta_strategy_probe.py](/C:/Users/zking/Documents/GitHub/PancakeBot/inspection/run_meta_strategy_probe.py).
7. The current candidate-generation constraint is a hard minimum selected bet rate of `5%`; there is no current hard maximum.
8. Recent candidate-generation work found the first viable `5%+` dislocation-family lead via a side-biased `stageG2` branch: `market_contra`, `bull_only`, `temperature_bps=5.0`, `dislocation_threshold_pp=0.25`, `expected_net_min_bnb=0.04`, `fixed_bet_bnb=0.2`, `cutoff_pool_total_min_bnb=1.2`, `market_extreme_min=0.02`, `bull_expected_net_extra_min_bnb=0.01`, `perf_adapt_mode=off`, which reached `+0.025820 / 500` at `5.07%` bet rate over the latest `30,000` rounds.
9. A second viable `5%+` lead now exists from a looser bull-only `stageB` branch: `allowed_sides=bull_only`, `dislocation_threshold_pp=0.05`, `expected_net_min_bnb=0.02`, `fixed_bet_bnb=0.1`, `cutoff_pool_total_min_bnb=0.6`, `market_extreme_min=0.02`, `bull_expected_net_extra_min_bnb=0.01`, `temperature_bps=5.0`, `perf_adapt_mode=off`, with the base `adaptive_shadow` side selection and `flow_gate_mode=off`. On the latest `30,000` rounds, the best verified chunk result reached `+0.037466 / 500` at `8.32%` bet rate.
10. Nearby-family results are now asymmetric: `stageH` stayed far too sparse even after side-biasing, and broad `stageB` bear-only variants reached `5%+` activity but remained decisively negative.
11. On the x80 / recent-40k block framing, neither new `5%+` candidate is a static replacement for the promoted baseline, but both add hindsight oracle value as pocket candidates. `stageG2` bull-only remains the stronger marginal-pocket candidate, while `stageB` bull-only adds a second, more persistent but smaller pocket source.
12. The archived `good-results-codebase` flow/LGBM family is now a live candidate family again under current `1 GWEI` strict-gas semantics. The active-repo port lives in [inspection/run_flow_backtest_scenario.py](/C:/Users/zking/Documents/GitHub/PancakeBot/inspection/run_flow_backtest_scenario.py) backed by [inspection/flow_strategy_common.py](/C:/Users/zking/Documents/GitHub/PancakeBot/inspection/flow_strategy_common.py), and the runner now supports `tail_offset_rounds` so the same flow setup can be replayed on older recent windows and aligned exactly to block-level meta datasets via `backtest_trades.csv`.
13. The flow-family tooling had a duplicate-`epoch` column bug in `build_flow_table()` that was fixed by deduplicating the concatenated feature table in [inspection/flow_strategy_common.py](/C:/Users/zking/Documents/GitHub/PancakeBot/inspection/flow_strategy_common.py). That bug affected the earlier flow research numbers, so older `+0.311285 / 500` claims should be treated as superseded by the corrected reruns.
14. A direct warmup sweep on the current promoted dislocation baseline showed that lowering selector/router warmup is not the fix for the live sparsity problem. On the recent `30,000`-round window, `warmup=10000` was best at `+0.105031 / 500` with `0.743%` bet rate, `warmup=20000` fell to `+0.073450 / 500` with `0.947%` bet rate, `warmup=5000` dropped to `+0.015923 / 500` with `0.543%` bet rate, and `warmup=50000` turned negative at `-0.047832 / 500` despite a higher `1.27%` bet rate. This points back to candidate EV supply, not warmup, as the main bottleneck.
15. After the duplicate-column fix, the best current flow-family recent setup is narrower than first thought but still meaningful. Using [inspection/run_flow_backtest_scenario.py](/C:/Users/zking/Documents/GitHub/PancakeBot/inspection/run_flow_backtest_scenario.py) with `train=15000`, `val=1000`, `step=1000`, `min_total_pool_c=1.2`, and `ev_threshold=0.005`, the corrected latest-`30,000` source-window run (`tail_offset_rounds=0`) produced `+0.167333 / 500` on its evaluated latest `15,000` rounds at `6.07%` bet rate. The matched promoted baseline on that same latest `15,000` window, with `warmup=10000`, was negative at about `-0.041177 / 500`, so the flow family still clearly beats baseline on the most recent evaluated tail.
16. The corrected flow-family rolling-window stability check is mixed but still favorable versus the promoted baseline. Over six overlapping latest-`30,000` source windows (`tail_offset_rounds = 0, 5000, 10000, 15000, 20000, 25000`), the flow setup was positive on `5/6` windows and beat the matched promoted baseline on `5/6` windows. Mean result was about `+0.036247 / 500` for flow versus `-0.014792 / 500` for the matched baseline, with mean bet rate `2.52%` for flow versus `0.53%` for baseline. The worst flow window was `-0.018392 / 500`, so the edge is real but clearly regime-sensitive rather than robust across all recent subwindows.
17. The same corrected flow setup does not generalize well to the broader recent-`40,000` x80 meta window. An aligned run using a `55,000`-round source window plus `tail_offset_rounds=104` (to match the current recent-40k block universe exactly) was slightly negative at about `-0.000784 / 500` with `0.375%` bet rate. When added to the current recent-40k meta dataset as an extra series, the flow candidate won only `1` oracle block and increased oracle value only marginally (about `+0.000618 / 500`). So, at the moment, the flow family looks like a localized latest-tail candidate, not a strong recent-40k pocket source.
18. Dry-mode observability was upgraded on `2026-03-26`. In addition to the persistent dry bet/settlement files (`dry_bets.jsonl`, `dry_audit_trades.csv`, `dry_bankroll_state.json`), dry startup now resets and writes `var/runtime/dry_cycle_audit.csv`, a per-cycle decision audit with epoch state, open-round pool snapshot, router mode, selected strategy, skip reason, and bankroll-before/after-action fields. This is the primary artifact to inspect when terminal output is repetitive or sparse.
19. The shared runtime/backtest warmup calculation was corrected on `2026-03-26` to include the most demanding active provider instead of only the dislocation selector. The helper lives in [pipeline.py](/C:/Users/zking/Documents/GitHub/PancakeBot/pancakebot/domain/strategy/pipeline.py) as `required_pipeline_warmup_rounds(...)`. This fixed a real integration gap where a live flow overlay can be enabled without silently staying unready.
20. The current promoted runtime profile changed again on `2026-03-26` after isolating the real source of recent-window lift. The active `config.toml` profile now uses:
   - router mode `selector_max_score`
   - only `disloc_stageB_bullonly_recent8pct_v1` on the dislocation side
   - flow overlay `flow_lgbm_recent_t12k_r1k_regime40_v1`
   - `flow.train_size = 12000`
   - `flow.retrain_interval = 1000`
   - `flow.ev_threshold = 0.0025`
   - `flow.min_total_pool_c = 1.0`
   - `flow.roll_window = 40`
   - `flow.roll_winrate_min = 0.48`
   - `flow.cooldown_trades = 40`
   Shared-pipeline verification on `2026-03-26` showed:
   - previous promoted runtime (`stageB` only, `online_selector_score_fallback`) latest `10000`: `-0.130542 BNB`, `8.21%` bet rate
   - `stageB` only with `selector_max_score` latest `10000`: `+0.119605 BNB`, `9.67%` bet rate
   - new promoted hybrid (`stageB + tuned flow`, `selector_max_score`) latest `10000`: `+3.094839 BNB`, `9.38%` bet rate
   - previous promoted runtime latest `6000`: `+1.291108 BNB`, `8.02%` bet rate
   - `stageB` only with `selector_max_score` latest `6000`: `+0.724201 BNB`, `9.90%` bet rate
   - new promoted hybrid latest `6000`: `+1.635637 BNB`, `10.20%` bet rate
   On the latest `10000`, the hybrid is not just a router improvement; the tuned flow overlay materially improves the selected trade set.
21. A background dry monitor now exists in [run_dry_cycle_monitor.py](/C:/Users/zking/Documents/GitHub/PancakeBot/inspection/run_dry_cycle_monitor.py). It tails `var/runtime/dry_cycle_audit.csv`, writes periodic summaries to `../PancakeBot_var_exp/*.jsonl`, and flags obvious anomalies such as unexpected strategy selection, unexpected bet side, zero bets for too long, or prolonged idle streaks. Its default allowlists now match the promoted hybrid runtime: `disloc_stageB_bullonly_recent8pct_v1` plus `flow_lgbm_recent_t12k_r1k_regime40_v1`, with both `Bull` and `Bear` sides allowed.

## Operational Rules

1. Use the V1 repo to verify historical/runtime behavior before claiming V2 parity.
2. Keep artifacts out of the repo:
   - experiments: `../PancakeBot_var_exp/`
   - repo archive: `../PancakeBot_repo_archive/`
3. Use the workspace venv only.

## Current Cleanup

On `2026-03-23`, obvious legacy/scratch material was archived out of V1. See [REPO_ARCHIVE.md](/C:/Users/zking/Documents/GitHub/PancakeBot/REPO_ARCHIVE.md).
Obsolete build/cache artifacts should be deleted rather than archived.

## Next Likely Work

1. Write V2 design specs, starting with project goals/workflow, then runtime/logging.
2. Rebuild V2 around an explicit runtime contract and operator logging contract.
3. Expand the offline meta-strategy selector beyond the first block-level V1 in [META_STRATEGY_PROBLEM.md](/C:/Users/zking/Documents/GitHub/PancakeBot/docs/META_STRATEGY_PROBLEM.md).
4. Re-run parity validation only after the relevant contracts are approved.
