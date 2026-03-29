# PancakeBot Agent Guidelines

## Startup Routine

1. Read [PROJECT_GOALS_AND_WORKFLOW.md](/C:/Users/zking/Documents/GitHub/PancakeBot/PROJECT_GOALS_AND_WORKFLOW.md) first.
2. Read [CURRENT_CONTEXT.md](/C:/Users/zking/Documents/GitHub/PancakeBot/CURRENT_CONTEXT.md) second.
3. Do not start with a broad repo rediscovery unless the user asks for it or the context is stale.
4. If the task touches V2 runtime, logging, or parity, use V1 as the behavioral reference until an approved V2 design spec says otherwise.
5. Read only the specific V1/V2 files needed for the task. Do not re-read the entire `NEXT_CHAT_HANDOFF.md` by default.

## Project Priorities

1. Follow [PROJECT_GOALS_AND_WORKFLOW.md](/C:/Users/zking/Documents/GitHub/PancakeBot/PROJECT_GOALS_AND_WORKFLOW.md) as the durable project policy.
2. Treat regime drift as a first-class constraint.
3. Keep code, runtime behavior, and logging coherent rather than patch-driven.

## Repo Roles

1. `PancakeBot/` is the active V1 reference repo.
2. `PancakeBotV2/` is a redesign repo, not yet safe to assume parity-complete.
3. Experiment outputs belong outside the repo, normally under `../PancakeBot_var_exp/`.
4. Archived repo clutter belongs outside the repo, under `../PancakeBot_repo_archive/`.

## Operating Rules

1. Use the workspace venv only:
   `C:\Users\zking\Documents\GitHub\PancakeBot\.venv\Scripts\python.exe`
2. Keep repo trees lean. Do not leave experiment artifacts, scratch logs, copied crash dumps, or ad hoc notes in the repo.
3. Delete build/cache artifacts instead of archiving them:
   `__pycache__/`, `.pytest_cache/`, `*.pyc`, `*.pyo`, `*.egg-info/`
4. Never use PowerShell `Remove-Item` for cleanup. If cleanup is needed, use `cmd /c del /f /q` for files and `cmd /c rmdir /s /q` for directories, and do not retry PowerShell variants.
5. Distinguish clearly between:
   - parity work
   - cleanup work
   - intentional redesign
6. For V2, do not stack tactical patches onto a confused runtime shape. Prefer a written contract/spec first, then implement against it.
7. Logging must be operator-centric:
   - concise
   - stable
   - semantically named
   - emitted at a cadence a human can follow while the bot runs
8. When a new thread starts, catch up from this file, [PROJECT_GOALS_AND_WORKFLOW.md](/C:/Users/zking/Documents/GitHub/PancakeBot/PROJECT_GOALS_AND_WORKFLOW.md), and [CURRENT_CONTEXT.md](/C:/Users/zking/Documents/GitHub/PancakeBot/CURRENT_CONTEXT.md), then move directly to the task.
9. For profile/controller research, use rolling causal backtests as the primary evidence source. Treat shadow recommendations as a secondary, final sanity check before any runtime-control rollout, not as the main comparison method.

## Decision Standard

1. If a term, log field, or control flow name is ambiguous, change it or reject it.
2. If a behavior differs from V1, decide whether that is:
   - required parity
   - acceptable redesign
   - unverified drift
3. If it is unverified drift, do not present it as complete.
