# PancakeBot Refactor Anchor

This document is the non-negotiable anchor for the long-running refactor.
All future iterations must align with these directives unless the user
explicitly changes them.

## User Directives

1. The codebase must stay clean and lean.
2. There is exactly one production strategy pipeline shared by live, dry, and
   backtest modes.
3. Backtest probe tooling is separate and independent from production runtime
   code, but must run the same shared production pipeline.
4. No backward compatibility layers.
5. Legacy modules are retained under `inspection/legacy` for one transition
   cycle only.
6. Knobs should be tunable from config when meaningful.
7. Use consistent terminology across code, config, docs, logs, and artifacts.
8. Document config settings, modules, classes, functions, and complex logic.
9. Refactor continuously and reevaluate after each conceptual iteration.
10. Commit frequently with small rollback units.

## Conceptual Iteration Policy

1. An iteration is conceptual, not pre-exhaustively knowable.
2. Unknown discoveries are expected during implementation.
3. Full codebase reevaluation is the final step of each iteration.
4. After reevaluation, continue implementation directly unless user input is
   required by a blocker.

## Engineering Rules

1. Favor deletion over abstraction when code is obsolete.
2. Avoid duplicate logic between production and probe paths.
3. Keep configuration explicit and validated.
4. Prefer deterministic and reproducible behavior.
5. Keep modules small, coherent, and single-purpose.

## Required Persistent Artifacts

1. `docs/TERMINOLOGY.md` for canonical terms.
2. `docs/REFRACTOR_LOG.md` for decisions, tradeoffs, and follow-ups.
3. `NEXT_CHAT_HANDOFF.md` updated at meaningful checkpoints for context
   compaction resilience.
