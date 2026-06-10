# Phase 3a Grand Audit — 2026-06-10

**Method.** Inventory of every file at every visibility level on BOTH hosts: Windows (canonical git clone, `master` @ `b3b06db`) and the Frankfurt VM (`/root/pancakebot`, now a git worktree of `/srv/pancakebot.git` via push-to-deploy). Tracked sync is verified by git itself (worktree status); untracked/ignored cross-host comparisons use CR-normalized SHA-1. READ-ONLY: nothing was moved, deleted, or committed by this audit.

**Action legend.** KEEP = current, no action. COMMIT = untracked file that belongs in git (KEEP-equivalent for untracked). ARCHIVE = move to `Downloads/OLD/pancakebot/2026_06_10_phase3_repo_archive/` (preserve directory structure). UPDATE = content fix needed (flagged, separate work). CONSOLIDATE = merge/move within the repo docs structure. DELETE = no archival value. TBD = owner decision.

## Executive summary

| Inventory | Count | Breakdown |
|---|---|---|
| Tracked (Windows == VM, **zero drift**) | 244 | per-file review of 162 non-core files: KEEP 121, ARCHIVE 31, UPDATE 8, CONSOLIDATE 2; the remaining 82 core files (`pancakebot/` + `tests/` outside the review set) are KEEP by default — importer analysis found **zero dead modules** |
| Untracked (34 unique across hosts) | W-only 14, both 9 (all 9 hash-match), VM-only 11 | COMMIT 12, ARCHIVE 13, DELETE 8, TBD 1 |
| Ignored outside `var/`+`.venv`+caches | Windows 31, VM 5 | listed in section 3 |
| Ignored runtime dirs | `var/`, `.venv/`, caches (both hosts) | KEEP (`var/`, `.venv/`) / DELETE (caches) |

**End-state delta after executing this plan:** zero untracked on both hosts; ignored = `var/` + `.venv/` + `.env` (Windows) + IDE/caches per user call — i.e. the Phase 3 goal.

## 1. Tracked files — sync status

`git status --porcelain` on the VM worktree: **zero tracked modifications** (guaranteed going forward by push-to-deploy — drift is structurally impossible while deploys go through `git push vm master`).

### 1a. ARCHIVE (31) — retired-era probes/installers; conclusions live in memory, a cited successor, or markdown

| File | Reason |
|---|---|
| `bootstrap/MIGRATION.md` | Completed one-time Windows→VM cutover runbook (Phase 4 done, live on VM). Historical value only; its Windows-rollback recipe depends on the service cluster being archived anyway — keep them together in the archive. |
| `bootstrap/install.ps1` | Windows-BOT fresh-clone installer (venv + config check + SCM service registration via windows/setup_service.py). Note: also the only orchestrated -IncludeOperatorUI entry to boot_survival.ps1, which runs standalone after archive. |
| `bootstrap/uninstall.ps1` | Windows-BOT uninstaller (stops/removes PancakeBotLive/Dry SCM services); pair of install.ps1. |
| `bootstrap/windows/setup_service.py` | Windows-BOT service registration orchestrator; imports WindowsServicePlatform and shells out to scripts/install_services.ps1 — coupling confirmed. |
| `pancakebot/service/common.py` | pywin32 ServiceFramework host shell (imports servicemanager/win32serviceutil/windows_platform — coupling confirmed); Windows-BOT only. |
| `pancakebot/service/dry_service.py` | PancakeBotDry SCM service entry (imports win32serviceutil + common — coupling confirmed). |
| `pancakebot/service/live_service.py` | PancakeBotLive SCM service entry (imports win32serviceutil + common — coupling confirmed). |
| `pancakebot/service/windows_platform.py` | Windows SCM/pywin32 adapter (win32event/win32job/win32service imports confirmed); VM-only is permanent. |
| `research/p4c_ntp_probe.py` | Measured NTP_QUERY_TIME_* constants removed in Bundle 5 v2 (timing_constants.py:548); NTP wake retired, narrative in timing-architecture-history memory |
| `research/p4c_offline_replay.py` | One-shot diagnostic for the resolved p4c-era 0-firings streak; conclusion lives in resolved-issues/timing-history memory |
| `research/probe_batch_shape_compare_2026_05_14.py` | I3 probe of retired receipts batching on retired home host; superseded by probe_batch_receipts_p99_3ep_2026_06_03.py + Era 12 getLogs migration; conclusions in decision-log memory |
| `research/probe_batch_size_2026_05_14.py` | I4 batch-size probe for retired receipts backfill; superseded by probe_batch_receipts_p99_3ep_2026_06_03.py (VM re-baseline) |
| `research/probe_fire_to_all_p99_2026_05_11.py` | Measured retired hedged fire-to-all transport on retired home host (1319ms value replaced by VM 240ms); successor probe_batch_receipts_p99_3ep_2026_06_03.py is the cited source |
| `research/probe_fire_to_all_p99_batch20_clean_2026_05_11.py` | Clean-spacing rerun for retired hedged transport; superseded by VM re-baseline (probe_batch_receipts_p99_3ep_2026_06_03.py); executor-queue lesson preserved in its docstring offline |
| `research/probe_fire_to_all_warmup_diversity_2026_05_12.py` | Warmup/diversity diagnostic for the retired 6-endpoint hedged pool (Era 12b is single-source bloXroute) |
| `research/probe_get_block_receipts.py` | Spike validating eth_getBlockReceipts on drpc/publicnode — endpoints and receipts read path both retired; pivot narrative in memory + var/design docs |
| `research/probe_kline_reliability_2026_05_13.py` | Bundle-3 home-host OKX latency/failure analysis pinned to a 2026-05-12 commit window; superseded by the 2026-06-06 VM re-baseline of all OKX constants |
| `research/probe_methodology_verify_2026_05_11.py` | Methodology check for the retired fire-to-all probes; archive together with that cluster |
| `research/probe_per_endpoint_isolated_2026_05_15.py` | I5 per-endpoint RTT on the retired 6-endpoint hedged pool; host and transport both gone |
| `research/probe_rpc_polling.py` | 2026-05-07 pivot spike for batched eth_getBlockReceipts viability — method retired by Era 12 getLogs; timing_constants cites the var/design architecture doc, not this spike |
| `research/probe_send_raw_transaction_rtt.py` | Predecessor of the cited probe_send_raw_tx_rtt_2026_05_20*.py reruns (near-identical docstring); timing_constants cites only the 2026-05-20 n=400 series |
| `research/probe_w32time_drift_2026_05_14.py` | I6 Windows-host clock-drift probe; bot is VM-only so W32Time discipline no longer gates the bot; conclusion preserved in reference_w32time_config memory |
| `research/probe_wss_ordering.py` | WSS Phase 0 spike for the abandoned WSS approach; findings preserved in project_wss_phase0_findings memory |
| `research/step28_kline_lookback_expansion_2026_05_27.py` | Superseded partial rerun: step28_kline_lookback_expansion_2026_05_28.py (successor, named in its docstring) re-covers these 4 variants in the full 13-variant numpy-loader run |
| `research/zero_bet_investigation_2026_04_24.md` | Resolved benign incident (verdict: keep running, none-needed); streak analysis predates and is partially superseded by the 2026-04-26 clock-skew root cause; conclusions in resolved-issues memory |
| `scripts/disable_dry.ps1` | Windows-BOT SCM control (stop + disable PancakeBotDry); SCM-only. |
| `scripts/disable_live.ps1` | Windows-BOT SCM control (stop + disable PancakeBotLive); SCM-only. |
| `scripts/enable_dry.ps1` | Windows-BOT SCM control for PancakeBotDry; SCM-only, confirmed by content. |
| `scripts/enable_live.ps1` | Windows-BOT SCM control (start type Automatic + start PancakeBotLive); SCM-only, confirmed by content. |
| `scripts/install_services.ps1` | Windows-BOT SCM installer (pywin32 registration + pythonservice DLL/registry fixups); VM-only is permanent. Coupling confirmed: invoked by bootstrap/windows/setup_service.py. |
| `scripts/uninstall_services.ps1` | Windows-BOT SCM uninstaller; pair of install_services.ps1. (Contains a stale ref to nonexistent scripts/uninstall_old_supervisor.ps1 — moot once archived.) |

### 1b. UPDATE (8) — content fixes, flagged for follow-up work

| File | Reason |
|---|---|
| `.gitignore` | Four dead entries — old_experiments_scripts_tests.zip, AUTONOMY_DIRECTIVE.md, QuickQuestion.txt, new_idea.txt — none exist anywhere on disk (Test-Path verified, gitignore-blind). Remove the four lines; preventive ignores (.idea/, .aiassistant/, var/, etc.) are fine. |
| `bootstrap/README.md` | Documents Windows install as a current first-class path ('one-script setup (Windows + Linux)', enable_dry.ps1 etc.); rewrite Linux-only + new tree after the archive/operator-move; drop or redirect the MIGRATION pointer. |
| `config.toml` | Values current, but the pool_cutoff_seconds comment still cites the retired '(ramp_1, ramp_2, final)' poll schedule — Candidate C (2026-06-06) collapsed it to one single_poll (engine.py:542, runtime/config.py:44 confirm). One-line comment fix. |
| `requirements.txt` | torch, catboost, lightgbm, scikit-learn have ZERO imports anywhere in the repo (pancakebot/, tests/, research/, scripts/; loose pattern incl. indented imports) — dead deps (torch alone is GBs on the VM). All other lines verified in use (matplotlib via backtest/runner.py; pandas-stubs is type-check-only, plausibly intentional). |
| `scripts/_smoke_discord_send_test.py` | Tool is current (POSTs to the 3 PANCAKEBOT_*_DISCORD_WEBHOOK_URL channels used on the VM) but docstring is stale Windows-era ('Machine-level env vars', 'flip schtasks to --alert') — refresh wording, keep the tool. |
| `tests/test_bootstrap_scripts.py` | Outside my name-pattern scope but directly coupled (flagging to avoid a silent break): test_all_expected_files_exist + _PS1_SCRIPTS/_PY_HELPERS lists assert existence of install.ps1, uninstall.ps1, windows/setup_service.py, setup_autologon.ps1, boot_survival.ps1, launch vbs, AUMID README — must drop archived entries and re-path operator files after the tools/claude_desktop/ move. |
| `tests/test_service_lifecycle.py` | Mostly OS-agnostic (supervision/notifications/SupervisorCore tests — keep), but split 3 Windows tests with the archive: test_common_module_imports (L330, imports common+windows_platform), test_live_and_dry_service_classes_importable (L349), test_job_object_kills_child_on_supervisor_death (L490, win32job). File docstring 'Windows-Service-based' also needs refresh. |
| `tests/test_service_platform.py` | Linux-adapter tests (mocked systemctl) are current — keep; split the ~9 @skipif(not _IS_WIN) WindowsServicePlatform tests (L125-292) with the archive or they fail on this Windows dev box once windows_platform.py is gone. |

### 1c. CONSOLIDATE (2)

| File | Reason |
|---|---|
| `bootstrap/windows/setup_autologon.ps1` | NOT a stale duplicate — thin delegation wrapper around scripts/setup_autologon.ps1 (the implementation). Fold into the single operator copy during the tools/claude_desktop/ move; keeping both is pointless once co-located. |
| `research/holdout_2026_04_24.md` | Still-binding promotion gate (frozen holdout 474880..475311); belongs in docs/ during Phase 3 doc consolidation; in-doc hash note already flags stale snapshots |

### 1d. KEEP (121 reviewed; 82 core files KEEP by default, zero dead modules)

<details><summary>121 reviewed KEEP files (expand)</summary>

| File | Reason |
|---|---|
| `.gitattributes` | Sound for the Windows-dev/Linux-deploy reality: index stores all text LF (git ls-files --eol shows zero i/crlf or i/mixed), *.sh eol=lf protects VM bootstrap + post-receive hook shebangs (4 .sh files), *.vbs eol=crlf covers the operator launcher; globs survive the planned Phase 3 dir moves. |
| `README.md` | Refreshed 2026-06-10 (b3b06db push-to-deploy + c817e53 Era 12b); wake table matches current constants (6195/2500/1500/1195/789), VM/systemd + chrony reality, and the NtpSync-retirement note is accurate (file confirmed gone). |
| `abi/prediction_v2_abi.json` | Single source of truth for contract types; referenced by pancakebot/paths.py (ABI_JSON_PATH), prediction_contract.py, tests/test_abi_type_derivation.py, and multiple research probes. |
| `bootstrap/common/config_check.py` | Cross-platform config/secrets preflight used by install.sh; current. |
| `bootstrap/common/health_check.py` | Post-install validation via ServicePlatform (chrony + registered + RUNNING + READY); current Linux path, recently touched (2026-06-10). |
| `bootstrap/common/python_setup.py` | Cross-platform venv+deps installer used by bootstrap/install.sh; current Linux install path. |
| `bootstrap/common/service_specs.py` | Shared live/dry ServiceSpec builder; 3c-2 systemd-direct candidate. Minor: still renders SCM PancakeBotLive/Dry names via sys.platform branch — clean with the win32-branch sweep. |
| `bootstrap/install.sh` | Current Linux fresh-clone installer (the production bootstrap path). |
| `bootstrap/linux/git_post_receive.sh` | Phase 3c-0 push-to-deploy hook (tracked copy of the VM's /srv/pancakebot.git post-receive, applied 2026-06-10); current. |
| `bootstrap/linux/install_python313.sh` | Current pyenv-based Python 3.13 build for the AlmaLinux VM; consumed by install.sh. |
| `bootstrap/linux/setup_service.py` | Installs the systemd live/dry units via LinuxServicePlatform; current production install path; 3c-2 systemd-direct candidate. |
| `bootstrap/uninstall.sh` | Current Linux uninstaller; pair of install.sh. |
| `bootstrap/windows/AUMID_stamper/README.md` | Claude-OPERATOR scaffolding: rebuild instructions for the out-of-repo C:\Tools stamp_claude_aumid.exe that boot_survival.ps1 verifies. Flag move to tools/claude_desktop/. |
| `bootstrap/windows/boot_survival.ps1` | Claude-OPERATOR-desktop boot-survival chain (autologon + ClaudeLaunchElevated task + AUMID stamping); header explicitly states the bot does not need it. Flag move to tools/claude_desktop/. |
| `bootstrap/windows/launch_claude_admin_direct.vbs` | Claude-OPERATOR elevated launcher (tracked copy of C:\Tools deployment; also keepalive /keepalive mode per memory). Flag move to tools/claude_desktop/. |
| `docs/SUPERVISOR.md` | Refreshed 2026-06-10 (c817e53); top half current (systemd units, 4 classifications, Step 27a aggregation). NOTE: body from '## Install' down is still the Windows-era guide (pywin32/SCM, setx /M, Get-Service, PS troubleshooting) — flag UPDATE when 3c-2 systemd-direct lands / Phase 3 archives the Windows-Service cluster; those sections need the systemd rewrite then. |
| `docs/architecture.html` | Refreshed 2026-06-10 (c817e53); spot-checked wake offsets are post-off350 current (preflight ~6.20s, single_poll 2.50s fixed rail, critical path ~1195ms static fallback). |
| `docs/logging.md` | Verified against pancakebot/log.py: format '{ts}  {LEVEL:<5}  {ACTION:<8}  {message}', _ACTION_W=8 emit-time enforcement, info/warn/error signatures, var/<mode>/runtime.log + cycle_audit.csv layers, no-production-DEBUG — all match reality. |
| `pancakebot/runtime/supervisor_artifacts.py` | Real path of 'supervisor_artifacts' (under runtime/, not service/). Used by run.py + supervisor_core on Linux; 3c-2 systemd-direct candidate. Docstring 'consumed by the Windows Service supervisor' is stale — Phase 2 wording refresh. |
| `pancakebot/service/__init__.py` | Package facade + get_platform(); 3c-2 systemd-direct candidate. Clean the win32 adapter-selection branch when windows_platform.py is archived (per Phase 3 plan). |
| `pancakebot/service/linux_platform.py` | systemd adapter (sd_notify, systemctl, fcntl mutex) — the production platform; 3c-2 systemd-direct candidate. |
| `pancakebot/service/notifications.py` | Discord alert taxonomy used by the live Linux supervisor; 3c-2 systemd-direct candidate. |
| `pancakebot/service/platform_base.py` | Core ServicePlatform ABC (plan says untouched); 3c-2 systemd-direct candidate. |
| `pancakebot/service/supervise.py` | systemd ExecStart entry point; 3c-2 systemd-direct candidate. Clean its win32 service-name branch (_service_names) with the archive. |
| `pancakebot/service/supervision.py` | Pure health classification + restart history (no Win32 deps, verified); 3c-2 systemd-direct candidate. Docstring still says 'used by the Windows Service' — Phase 2 wording refresh. |
| `pancakebot/service/supervisor_core.py` | OS-agnostic supervision loop run under systemd via supervise.py; 3c-2 systemd-direct candidate. Imports verified clean of pywin32. |
| `research/2026_05_20_bsc_rtt_probe_data/2026_05_20_send_raw_tx_probe_100_at_10s.jsonl` | Cited measurement data (n=400 across 4 probes) behind BSC_BET_SUBMIT_RTT in timing_constants.py:250-264 |
| `research/2026_05_20_bsc_rtt_probe_data/2026_05_20_send_raw_tx_probe_100_at_10s.md` | Companion results memo for cited RTT probe data |
| `research/2026_05_20_bsc_rtt_probe_data/2026_05_20_send_raw_tx_probe_100_at_10s_hour2.jsonl` | Cited measurement data behind BSC_BET_SUBMIT_RTT in timing_constants.py |
| `research/2026_05_20_bsc_rtt_probe_data/2026_05_20_send_raw_tx_probe_100_at_10s_hour2.md` | Companion results memo for cited RTT probe data |
| `research/2026_05_20_bsc_rtt_probe_data/2026_05_20_send_raw_tx_probe_100_at_1s.jsonl` | Cited measurement data behind BSC_BET_SUBMIT_RTT in timing_constants.py |
| `research/2026_05_20_bsc_rtt_probe_data/2026_05_20_send_raw_tx_probe_100_at_1s.md` | Companion results memo for cited RTT probe data |
| `research/2026_05_20_bsc_rtt_probe_data/2026_05_20_send_raw_tx_probe_100_at_1s_hour2.jsonl` | Cited measurement data behind BSC_BET_SUBMIT_RTT in timing_constants.py |
| `research/2026_05_20_bsc_rtt_probe_data/2026_05_20_send_raw_tx_probe_100_at_1s_hour2.md` | Companion results memo for cited RTT probe data |
| `research/adaptive_dd_step13_2026_05_26.py` | Signal-research-v2 ledger step 13; 13b/13c document its bugs, arc retained deliberately |
| `research/adaptive_dd_step13b_2026_05_26.py` | Ledger step 13b (peak-semantics fix); part of deliberate dead-end record |
| `research/adaptive_dd_step13c_2026_05_26.py` | Ledger step 13c (pre-decision timing fix); final word of the adaptive-dd arc |
| `research/backfill_okx_extended.py` | Cited in pancakebot/paths.py:15 and backtest/runner.py:159; produces var/extended data still consumed by --use-extended-data |
| `research/bankroll_scale_rerun_step7_2026_05_26.py` | Ledger step 7 (5 vs 50 BNB scale baseline) |
| `research/bloxroute_soak_2026_06_09/bloxroute_getlogs_soak.py` | Active soak gating the VM restart greenlight for the getLogs migration (Phase 1) |
| `research/breaker_anchor_robustness_2026_05_08.py` | Provenance for the live drawdown-breaker design (E3 gate, restart-fragility analysis) |
| `research/breaker_threshold_cooldown_sweep_2026_05_08.py` | Provenance for live breaker config (dd=0.15/cd=72); cv5_followup reuses its patches |
| `research/breaker_threshold_cv5_followup_2026_05_08.py` | Companion CV5 spot-check for the breaker sweep; same provenance cluster |
| `research/canonical_cv5_permutation_null_2026_05_22.py` | Significance evidence (permutation null) for the canonical strategy's CV5 edge |
| `research/cooldown_sweep_step15_2026_05_26.py` | Ledger step 15 (cooldown sweep + permutation null) |
| `research/creative_risk_step14_2026_05_26.py` | Ledger step 14; 14c documents its structural problems, arc retained |
| `research/creative_risk_step14c_2026_05_26.py` | Ledger step 14c (shadow-exit diagnostic fix) |
| `research/cutoff_3_preliminary.py` | Dead-end ledger for the cutoff=2 canonical invariant (memory rule cites this exploration) |
| `research/cutoff_3_retune_2_6_14.py` | Dead-end ledger companion; references cutoff_3_preliminary.py by name |
| `research/cutoff_lookback_sweep_2026_05_08.py` | 153-variant grid provenance: 'only canonical passes 4 gates' (memory project_cutoff_lookback_sweep) |
| `research/cutoff_lookback_sweep_2026_05_08_reagg.py` | Corrected aggregation (net_pnl_bnb key fix) + Phase 4 extension run; not a duplicate of the sweep |
| `research/diag_getlogs_endpoints_2026_06_09.py` | Current-era getLogs migration validation arc (commit f9b376d); explains v1-parity zero-result |
| `research/extension_v2_permutation.py` | Promotion-gate provenance: extension_v2 permutation null p=0.002 (still-binding gate per memory) |
| `research/feature_characterization_step17_2026_05_26.py` | Ledger step 17 (feature characterization F1-F5) |
| `research/hot_windows_step11_bonus_2026_05_26.py` | Ledger step 11 bonus (hot-window scan) |
| `research/hour_filter_step19_2026_05_26.py` | Ledger step 19 (hour-of-day filter test) |
| `research/in_process_runner.py` | Shared harness: imported by tests/test_in_process_runner.py + test_extended_data_loader.py and ~50 research scripts; cited in momentum_gate.py:488 |
| `research/inclusion_experiment_2026_06_08/README.md` | Cited in timing_constants.py:234/241 — provenance for live VALIDATOR_ASSEMBLY_WINDOW_MS=214 (off350 fix) |
| `research/inclusion_experiment_2026_06_08/analyze.py` | Part of cited off350 experiment dir; post-hoc outcome reconstruction |
| `research/inclusion_experiment_2026_06_08/run_experiment.py` | Part of cited off350 experiment dir; the live A/B coordinator |
| `research/inclusion_experiment_2026_06_08/slip_vs_predecessor.py` | Part of cited off350 experiment dir; deadline-normalized slip analysis |
| `research/inclusion_experiment_2026_06_08/sweep_and_claim.py` | Part of cited experiment dir; operationally reusable wallet claim+sweep tool |
| `research/inclusion_experiment_2026_06_08/watch.py` | Part of cited off350 experiment dir; progress checkpoint tool |
| `research/max_bet_cap_step9_2026_05_26.py` | Ledger step 9 (0.5 BNB max-bet cap test) |
| `research/numpy_kline_loader.py` | Shared harness cited in momentum_gate.py:440; required by step28 full run and memory-bound backtests |
| `research/okx_artificial_delay_probe.py` | Provenance for the still-live skew-aware _utc_now fix (test b of the clock-skew investigation; companion to kept root-cause md) |
| `research/okx_connection_ab.py` | Cited in pancakebot/market_data/okx_client.py:268 (fresh_conn behavior) |
| `research/okx_lag_root_cause_clock_skew.md` | Root-cause record for the live clock-skew fix (commit 4dda6cc); memory points operators back to this diagnosis |
| `research/p3a_wallet_features.py` | Imported by tests/test_p3a_wallet_features_lookahead.py |
| `research/p3b_orderflow_features.py` | Imported by tests/test_p3b_orderflow_features_lookahead.py |
| `research/p4c_canonical_loop_aggregate.py` | Companion aggregator for the cited p4c_canonical_loop_probe output |
| `research/p4c_canonical_loop_probe.py` | Cited in timing_constants.py:110 and momentum_gate.py:20,55 — provenance for OKX publish-delay constants |
| `research/p4c_phase0_feature_builder.py` | Origin of the regime/phase0 feature methodology later steps build on (regime_step1 sources its archived output); in-doubt default KEEP |
| `research/parity_getlogs_v2_blxr_2026_06_09.py` | Cited in pancakebot/chain/rpc_poller.py:96,1785 — byte-identical parity proof for the live getLogs path |
| `research/parity_getlogs_vs_receipts_2026_06_09.py` | v1 parity documenting WHY production endpoints can't serve getLogs (load-bearing for endpoint choice); self-marked regression-keepable, part of committed validation arc |
| `research/phase1_pool_gap_analysis.py` | Partial-vs-final pool gap evidence underpinning the accumulated-pool mental model and F0 pool-coverage framing |
| `research/post_cv5_to_current_step10b_2026_05_26.py` | Ledger step 10b (full post-CV5 cohort breakdown, distinct scope from step 10) |
| `research/post_fresh_backtest_step10_2026_05_26.py` | Ledger step 10 (post_fresh recent-regime characterization) |
| `research/probe_batch_receipts_p99_3ep_2026_06_03.py` | Cited in timing_constants.py:294 — source of the VM RPC_BATCH_RECEIPTS_RTT table {20: 240} |
| `research/probe_batch_receipts_p99_3ep_FROM_VM_2026_06_03_summary.json` | Cited by name in timing_constants.py:295 as the VM measurement output |
| `research/probe_block_availability_vm_2026_06_06.py` | Cited in timing_constants.py:318 — source of RPC_BLOCK_AVAILABILITY_DELAY_P99_MS=625 |
| `research/probe_send_raw_tx_rtt_2026_05_20.py` | Cited via wildcard in timing_constants.py:261 (probe_send_raw_tx_rtt_2026_05_20*.py, n=400) |
| `research/probe_send_raw_tx_rtt_2026_05_20_at_10s.py` | Cited via wildcard in timing_constants.py:261; one of the 4 independent 100-TX probes |
| `research/probe_send_raw_tx_rtt_2026_05_20_at_10s_hour2.py` | Cited via wildcard in timing_constants.py:261; one of the 4 independent 100-TX probes |
| `research/probe_send_raw_tx_rtt_2026_05_20_hour2.py` | Cited via wildcard in timing_constants.py:261; one of the 4 independent 100-TX probes |
| `research/random_shuffle_cv_step4_2026_05_25.py` | Ledger step 4 (fold-structure sensitivity of the CV5 edge) |
| `research/regime_hot_vs_baseline_2026_05_26.py` | Ledger: hot-window/bear-run regime hypothesis test, part of the step arc |
| `research/regime_step1_2026_05_25.py` | Ledger step 1 (per-cohort regime characterization) |
| `research/regime_step2_2026_05_25.py` | Ledger step 2 (pool-microstructure features) |
| `research/safety_extraction_step11_2026_05_26.py` | Ledger step 11 (drawdown/absolute-threshold/anti-martingale experiments) |
| `research/safety_extraction_step11_dual_2026_05_26.py` | Ledger step 11 dual-scale rerun (distinct scope: 5 BNB deployment scale) |
| `research/safety_extraction_step11c_2026_05_26.py` | Ledger step 11c (sizing-cap-while-compounding refinement) |
| `research/sizing_variants_step6_2026_05_26.py` | Ledger step 6 (6 bet-sizing variants incl. Kelly) |
| `research/static_bankroll_step8_2026_05_26.py` | Ledger step 8 (pure-edge static-bankroll baseline) |
| `research/step16_perm_test_50bnb_2026_05_26.py` | Ledger step 16 (permutation null on Step 11 50 BNB findings) |
| `research/step18a_f2_alignment_check_2026_05_26.py` | Ledger step 18a (F2 timestamp alignment sanity check) |
| `research/step20_external_signals_2026_05_26.py` | Ledger step 20 (external signal sources, incl. documented not-viable paths) |
| `research/step21_funding_buckets_and_social_2026_05_26.py` | Ledger step 21 (funding-rate buckets + social proxy) |
| `research/step22a_volume_filter_2026_05_26.py` | Ledger step 22a (volume-Z filter backtest) |
| `research/step23_intra_cooldown_bet_density_2026_05_26.py` | Ledger step 23 (intra-cooldown bet density) |
| `research/step24_loss_cooldown_timing_fix_2026_05_26.py` | Ledger step 24 — production-faithful settlement-timing fix reused by steps 25/26 |
| `research/step25_loss_offset_diagnostics_2026_05_26.py` | Ledger step 25 (loss-offset EV diagnostics feeding step 26) |
| `research/step26_selective_flip_veto_2026_05_26.py` | Ledger step 26 (selective flip-veto backtest) |
| `research/step28_kline_lookback_expansion_2026_05_28.py` | Ledger step 28 final (full 13-variant lookback-expansion run) |
| `research/step5_reconciliation_2026_05_26.py` | Ledger: reconciles 2026-05-08 sweep vs Step 5 extension PnL discrepancy (scale artifact) |
| `research/survey_getlogs_endpoints_2026_06_09.py` | Cited in pancakebot/chain/rpc_poller.py:94 — basis for the bloXroute endpoint choice |
| `research/sweep_harness.py` | Shared harness infrastructure referenced by in_process_runner and the breaker/cutoff sweeps |
| `research/verify_numpy_loader_equivalence.py` | Shared-harness equivalence proof legitimizing numpy_kline_loader results (step28 full run depends on it) |
| `research/vol_filter_step12_2026_05_26.py` | Ledger step 12; 12b documents what it conflated, arc retained |
| `research/vol_filter_step12b_disentangled_2026_05_26.py` | Ledger step 12b (disentangled vol-filter analysis, source of the dd=0.08+vol winner tested in step 16) |
| `research/walk_forward_step3_2026_05_25.py` | Ledger step 3 (walk-forward retune vs fixed canonical) |
| `research/wr_meta_filter_step5_2026_05_25.py` | Ledger step 5 (rolling-WR pause meta-filter) |
| `research/zero_confirmation_gauntlet_2026_06_06.py` | Recent rejected-candidate gauntlet (REJECTED 2026-06-06) — deliberate dead-end ledger backing the do-not-ship decision |
| `research/zero_confirmation_permutation_2026_06_06.py` | Permutation null (p=0.1359) that drove the zero-confirmation rejection; provenance for a binding decision |
| `run.py` | Current CLI entrypoint; mode flags, single-instance check, PID/crash.json supervisor artifacts all consistent with app.py and SUPERVISOR.md. |
| `scripts/_smoke_discord_catcher.py` | OS-agnostic localhost webhook catcher for supervisor notification smoke tests; exercises the live Linux notification path (notifications.py POSTs). |
| `scripts/_smoke_write_artifacts.py` | Writes the cross-platform supervision artifacts (crash.json, bot.pid, restart_history.jsonl) that SupervisorCore consumes on Linux; current smoke glue. |
| `scripts/notify_user_followup.py` | Claude-operator desktop helper (Discord follow-up ping for pending coordinator messages, Task-Scheduler-invoked). Flag move to tools/claude_desktop/. |
| `scripts/notify_user_mark_answered.py` | Claude-operator desktop helper (marks pending notifications answered); pair of notify_user_followup.py. Flag move to tools/claude_desktop/. |
| `scripts/setup_autologon.ps1` | REAL autologon implementation (operator-desktop: restores session so Claude app relaunches). Flag move to tools/claude_desktop/; fix .NOTES ref to install_services.ps1's Autologon.exe auto-download (that installer is being archived). |
| `tests/test_supervisor_core.py` | OS-agnostic SupervisorCore tests via FakePlatform (no pywin32, verified); guards the live Linux supervision logic; 3c-2 systemd-direct candidate alongside the module. |

</details>


## 2. Untracked files (34 unique) — the action core

### 2a. COMMIT (12) — provenance for shipped decisions / active workstreams
| File | Where | Reason |
|---|---|---|
| `research/probe_critpath_rpc_breakdown_2026_06_06.py` | both (match) | pre-cache fix provenance (decision log: latency_sign 265 -> 23ms) |
| `research/probe_critpath_rpc_breakdown_2026_06_06_summary.json` | VM only — pull back | results behind the pre-cache decision |
| `research/probe_keepalive_decay_2026_06_06.py` | both (match) | keep-alive >=30s preflight-warm design source |
| `research/probe_sign_broadcast_budget_2026_06_06.py` | both (match) | pre-cache provenance |
| `research/probe_sign_broadcast_budget_2026_06_06_summary.json` | VM only — pull back | results |
| `research/probe_block_grid_late_2026_06_06.py` | both (match) | LATE-bet/off350 analysis chain |
| `research/inclusion_latency_2026_06_07.py` | both (match) | off350 precursor (shipped VALIDATOR_ASSEMBLY_WINDOW fix) |
| `research/late_margin_analysis_2026_06_07.py` | both (match) | off350 precursor |
| `research/bloxroute_soak_2026_06_09/poll_log.csv` (512K) | both (match) | 12h soak evidence (5400 polls / 0 errors) gating the Era 12 restart; script already tracked |
| `research/bloxroute_soak_2026_06_09/summary.json` | both (match) | soak verdict |
| `research/probe_batch_receipts_p99_3ep_FROM_VM_2026_06_03.jsonl` (215K) | Windows only | raw data behind the TRACKED summary + the live RTT table |
| `research/post_cv5_to_current_2026_06_06.py` | Windows only | ACTIVE workstream (pending cohort-refresh backtests) |

### 2b. ARCHIVE (13) — rejected-experiment artifacts; verdicts recorded in memory
| File | Where | Reason |
|---|---|---|
| `research/ab_inclusion_latency_2026_06_01.py` | Windows only | relay A/B harness — relays REJECTED (48Club/BlockRazor) |
| `research/ab_inclusion_results_*.jsonl` (x4) | Windows only | rejected-experiment data |
| `research/ab_inclusion_summary_*.json` (x3) | Windows only | rejected-experiment summaries |
| `tests/test_ab_inclusion_harness_dryrun.py` | both (match) | tests the archived harness — goes with it |
| `research/btc_only_degrade_edge.py` | Windows only | BTC-only degrade = HOLD verdict (memory) |
| `research/_cutoff_sweep_234.py` | Windows only | cutoff 2/3/4 exploration; cutoff=2 invariant + sweep verdicts in memory |
| `research/probe_batch_receipts_p99_3ep_2026_06_03.jsonl` | Windows only | HOME-host raw RTT data (superseded by VM re-baseline) |
| `research/probe_batch_receipts_p99_3ep_2026_06_03_summary.json` | Windows only | home-era summary (FROM_VM summary is the tracked, cited one) |

### 2c. DELETE (8) — zero information loss, all on the VM
| File | Why safe |
|---|---|
| `research/inclusion_experiment_2026_06_08.py` | EXACT dup (CR-norm SHA-1) of tracked `research/inclusion_experiment_2026_06_08/run_experiment.py` |
| `research/analyze_inclusion_experiment.py` | EXACT dup of tracked `.../analyze.py` |
| `research/_experiment_watch.py` | EXACT dup of tracked `.../watch.py` |
| `research/slip_vs_predecessor_2026_06_08.py` | EXACT dup of tracked `.../slip_vs_predecessor.py` |
| `research/bloxroute_soak_2026_06_09/soak.stdout` | console dup; poll_log.csv + summary.json carry the data |
| `pancakebot/config.py.bak_pre_rebaseline` | warm-rollback copy, superseded by git push-deploy (instant rollback) |
| `pancakebot/timing_constants.py.bak_pre_rebaseline` | same |
| `.env.bak.20260605` | **SENSITIVE** stale secrets backup at the VM repo root; live secrets correctly live in `/etc/pancakebot/pancakebot.env` (verified: wallet key + Graph key + 3 webhooks present; repo-root `.env` correctly absent on the VM). Recommend `shred -u`. |

### 2d. TBD (1) — needs your call
| File | Question |
|---|---|
| `experiment_wallets.env.bak.20260608` (VM) | **SENSITIVE**: the 5 off350 experiment-wallet keys. Wallets may still hold dust BNB. Recommend: sweep remaining balances to the main wallet (tracked `inclusion_experiment_2026_06_08/sweep_and_claim.py` does this), store keys off-repo if wanted, then `shred -u`. |

## 3. Ignored files (list-only per scope)

### Windows, outside var/.venv/caches (31)
| Path | On VM? | Proposed |
|---|---|---|
| `.env` | no (VM secrets live in `/etc/pancakebot` — better layout) | KEEP (Windows research/sync needs it) |
| `.idea/` (9 files) | no | TBD — JetBrains IDE state; keep-ignored if PyCharm is used, else DELETE |
| `.pytest_cache/` (5) | yes (5 on VM) | DELETE both (regenerable) |
| `research/b1_realistic_home.log` | no | ARCHIVE (home-era probe log) |
| `research/probe_batch_receipts_p99_3ep_2026_06_03.log` | no | ARCHIVE (home-era probe log) |
| `research/data/ping_experiment_*.log` (15) | no | ARCHIVE (May WSS/ping-era experiment logs; verdicts in memory) |
| `var/` (285 files) | yes (runtime state; backed up to /root/backups 2026-06-10) | KEEP — the intended ignored dir |
| `.venv/` (~28k files) | yes | KEEP (machine artifact) |
| `__pycache__`/`*.pyc` (1453) | yes | DELETE (regenerable) |

### .gitignore hygiene
Four dead entries (`old_experiments_scripts_tests.zip`, `AUTONOMY_DIRECTIVE.md`, `QuickQuestion.txt`, `new_idea.txt`) — the files no longer exist anywhere -> remove the lines (UPDATE, flagged in 1b).

## 4. Directory restructure proposal (for approval before any move)

**3-way split:**

1. **`bootstrap/` -> Linux bot deploy only**: keeps `common/`, `linux/`, `install.sh`, `uninstall.sh`, `README.md` (UPDATE: rewrite Linux-only). `MIGRATION.md` -> archive (completed one-time cutover runbook).
2. **NEW `tools/claude_desktop/` -> Claude operator-desktop scaffolding** (KEEP, relocated):
   - `bootstrap/windows/boot_survival.ps1`
   - `bootstrap/windows/launch_claude_admin_direct.vbs`
   - `bootstrap/windows/AUMID_stamper/` (README)
   - `scripts/setup_autologon.ps1` — the real implementation; the `bootstrap/windows/setup_autologon.ps1` WRAPPER folds into it (they are wrapper+implementation, NOT duplicates — verified)
   - `scripts/notify_user_followup.py` + `scripts/notify_user_mark_answered.py`
   - NEW note needed: Autologon.exe provisioning loses its home (was auto-downloaded by the archived `install_services.ps1`)
3. **Archive `Downloads/OLD/pancakebot/2026_06_10_phase3_repo_archive/`** (preserve structure): the Windows-bot-service cluster (13 files, import-coupling verified): `pancakebot/service/{windows_platform,common,live_service,dry_service}.py`, `bootstrap/windows/setup_service.py`, `bootstrap/{install,uninstall}.ps1`, `scripts/{install,uninstall}_services.ps1`, `scripts/{enable,disable}_{live,dry}.ps1`; plus `bootstrap/MIGRATION.md`, the Windows test splits, the 31 ARCHIVE-tracked files (1a), the 2b untracked set, and the section-3 old logs.

**Code/test follow-ups the moves force** (must land in the SAME commit): `service/__init__.py` + `service/supervise.py` win32 branches; `tests/test_service_lifecycle.py` (3 Windows tests: L330/L349/L490); `tests/test_service_platform.py` (~9 skipif-Windows tests, L125-292); `tests/test_bootstrap_scripts.py` (existence-asserts the exact archived/moved paths — WILL fail post-move otherwise).

## 5. Cross-cutting findings

1. **`requirements.txt` dead deps**: `torch`, `catboost`, `lightgbm`, `scikit-learn` have ZERO imports across pancakebot/, tests/, research/, scripts/ — torch alone is GBs in every venv (including the VM's). Remove after a clean-venv suite run confirms.
2. **Dangling citations** (comment fixes): `timing_constants.py:146` cites missing `research/bundle4_timing_harness.py`; `timing_constants.py:263` cites `var/strategy_review/...` paths for data whose tracked home is `research/2026_05_20_bsc_rtt_probe_data/`; `tests/test_okx_warmup_transient.py:40` cites missing `okx_kline_freshness_fix_design.md`; `research/okx_lag_root_cause_clock_skew.md:57` cites missing `okx_parallel_probe.py`; `research/extension_v2_permutation.py` cites missing `regime_phase0_permutation.py`.
3. **Stale wording sweep-ups** (cosmetic, batch with 3c): `supervision.py` + `runtime/supervisor_artifacts.py` docstrings say "Windows Service"; `scripts/_smoke_discord_send_test.py` docstring is Windows-era; `pancakebot/log.py:79` mentions "Windows scheduled task"; `docs/SUPERVISOR.md` body below "## Install" is the Windows-era guide (flagged for the 3c-2 rewrite).
4. **Secrets layout confirmed healthy**: VM has NO repo-root `.env`; all 5 secrets live in `/etc/pancakebot/pancakebot.env` (0600, systemd EnvironmentFile). Only secret-hygiene issues: the two VM `.bak` files (2c/2d).

## 6. Proposed execution order (3b/3c — NOT executed)

1. COMMIT batch (2a, 12 files — pull the 2 VM-only summaries back first).
2. DELETE batch on the VM (2c, 8 files; `shred -u` the .env.bak) + resolve the TBD wallet file (2d).
3. ARCHIVE batch (1a + 2b + section-3 logs) -> `Downloads/OLD/pancakebot/2026_06_10_phase3_repo_archive/` preserving structure, then `git rm` the tracked ones.
4. Restructure (section 4) + the forced test/code updates in the SAME commit.
5. UPDATE batch (1b + findings 1-3).
6. 3d verification: `git status` clean on both hosts; ignored = var/.venv (+.env/.idea per call).
