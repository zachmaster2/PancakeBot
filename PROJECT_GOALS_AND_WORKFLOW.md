# Project Goals And Workflow

## Core Goal

Maximize net `BNB` per `500` rounds on recent history.

That goal is not the same as maximizing all-history performance. The current overall baseline can still be useful, but smaller or newer windows may have different winners, and a strategy that works now may degrade quickly in the next regime.

## Strategic Reality

1. The strategy is expected to keep evolving.
2. Different windows can have materially different best candidates, router settings, or profiles.
3. Recent performance matters more than frozen all-history leadership.
4. Robustness still matters. A strong recent result that is obviously fragile is not enough.

## Secondary Goals

1. Keep the code coherent, working, and intentionally designed.
2. Keep runtime behavior understandable to an operator while the bot is running.
3. Keep the repo lean and focused on active code and durable context.
4. Use V1 as the behavioral reference until V2 has approved design specs and verified parity where needed.

## Evaluation Standard

When evaluating a strategy or code change, prefer explicit comparison on:

1. recent-window `net_bnb_per_500`
2. overall/frozen-window baseline comparison
3. drawdown and bankroll-floor behavior when relevant
4. regime sensitivity, not just aggregate average

Do not assume the all-history winner is the right next step.

## Working Workflow

### 1. Start Narrow

1. Read [AGENTS.md](/C:/Users/zking/Documents/GitHub/PancakeBot/AGENTS.md).
2. Read this file.
3. Read [CURRENT_CONTEXT.md](/C:/Users/zking/Documents/GitHub/PancakeBot/CURRENT_CONTEXT.md).
4. Read only the specific code/docs needed for the current task.
5. Do not start with broad repo rediscovery unless the context is stale or the user asks for it.

### 2. Classify The Task

Every meaningful task should be treated as one of:

1. strategy research
2. parity verification
3. cleanup/repo hygiene
4. intentional redesign/spec work
5. runtime/operator experience work

Do not mix these casually. State which one is being done.

### 3. For Strategy Research

1. Define the evaluation window or windows before running experiments.
2. Say what metric actually matters for the decision.
3. Compare against the relevant baseline, not a random prior result.
4. If results differ by window, treat that as signal, not noise.
5. Capture the takeaway in durable context if it changes project direction.
6. For controller/profile-selection work, prefer rolling causal backtests over shadow validation as the main evidence source.
7. Use shadow only as a thin final sanity check before any runtime-controller rollout.
8. The target controller design is multi-profile absolute local estimation plus `skip`, not a structurally privileged baseline-versus-alternate controller.

### 4. For Parity Work

1. Treat V1 as the behavioral reference unless the user explicitly wants redesign instead.
2. Verify concrete behavior before claiming parity.
3. If V2 differs from V1, label it as one of:
   - required parity
   - intentional redesign
   - unverified drift
4. Do not describe unverified drift as finished work.

### 5. For V2 Design Work

1. Write or update a spec before stacking more tactical fixes.
2. Keep the runtime contract, logging contract, and operator model explicit.
3. Prefer a small number of deliberate abstractions over patch-driven helpers.
4. A cleaner design is good, but only if the resulting behavior is still coherent and testable.

### 6. Keep Operator Perspective In View

Logs and runtime behavior should answer:

1. What state is the bot in?
2. What is it waiting for?
3. Why did it skip, bet, settle, retry, or exit?
4. Is the behavior expected, late, degraded, or broken?

The operator should not need to reverse-engineer the code from the log stream.

### 7. Keep Context Durable

When a task changes the project direction, assumptions, or operating rules:

1. update [CURRENT_CONTEXT.md](/C:/Users/zking/Documents/GitHub/PancakeBot/CURRENT_CONTEXT.md)
2. update [AGENTS.md](/C:/Users/zking/Documents/GitHub/PancakeBot/AGENTS.md) if startup behavior should change
3. avoid burying durable guidance only in chat history

### 8. Cleanup Commands

1. Never use PowerShell `Remove-Item` for cleanup.
2. If cleanup is needed, use `cmd /c del /f /q` for files and `cmd /c rmdir /s /q` for directories.
3. Do not retry PowerShell cleanup variants after they fail once.

## Anti-Patterns

1. Chasing a local symptom without checking whether the system model itself is wrong.
2. Treating V2 as patch-first when the correct next step is a design/spec.
3. Optimizing for all-history averages while ignoring recent-window deterioration.
4. Leaving scratch artifacts or obsolete code in the active repo.
5. Claiming completion when behavior is still unverified.
