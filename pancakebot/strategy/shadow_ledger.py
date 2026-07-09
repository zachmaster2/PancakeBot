"""Shadow ledger for the composite cooldown state machine (2026-07-09).

While the drawdown breaker has real betting suspended, MomentumOnlyPipeline
keeps evaluating the gate every round and records the bet it WOULD have
made here — sized off the HYPOTHETICAL bankroll (suspension-start bankroll
plus cumulative shadow PnL), subject to the same pool/payout filters and
sizing caps as real bets. Shadow bets settle against realized closed-round
pools via the same impact-aware settlement function used by dry/backtest,
minus bet gas, so the ledger is a faithful counterfactual of continuing to
bet through the suspension.

At cooldown expiry the pipeline consults :meth:`release_ok`; the suspension
is extended unless the shadow shows genuine recovery (enough settled fires,
non-negative cumulative PnL, and hypothetical bankroll above a fraction of
the hypothetical rolling-window peak).

Persistence: one small JSON document rewritten atomically on every mutation
(the ``pause_state.json`` durability pattern), so a crash-restart
mid-suspension resumes with the ledger intact. Pass ``path=None`` for a
purely in-memory ledger (backtest).
"""
from __future__ import annotations

import json
from pathlib import Path

from pancakebot.settlement import settle_bet_against_closed_round
from pancakebot.types import Round
from pancakebot.util import InvariantError

_TRAJECTORY_WINDOW_DAYS = 7   # hypothetical-peak window (mirrors rolling_7d)


class ShadowLedger:
    """Counterfactual bet ledger for one breaker suspension."""

    __slots__ = (
        "_path", "active", "suspension_start_at", "suspension_bankroll",
        "cum_pnl", "n_settled", "n_wins", "extensions", "pending",
        "_trajectory",
    )

    def __init__(self, *, path: Path | None) -> None:
        self._path = Path(path) if path is not None else None
        self.active: bool = False
        self.suspension_start_at: int | None = None
        self.suspension_bankroll: float = 0.0
        self.cum_pnl: float = 0.0
        self.n_settled: int = 0
        self.n_wins: int = 0
        self.extensions: int = 0
        self.pending: dict[int, dict] = {}
        self._trajectory: list[tuple[int, float]] = []
        self._load()

    # -- persistence -------------------------------------------------------

    def _load(self) -> None:
        if self._path is None or not self._path.exists():
            return
        try:
            doc = json.loads(self._path.read_text(encoding="utf-8"))
            self.active = bool(doc.get("active", False))
            sa = doc.get("suspension_start_at")
            self.suspension_start_at = int(sa) if sa is not None else None
            self.suspension_bankroll = float(doc.get("suspension_bankroll", 0.0))
            self.cum_pnl = float(doc.get("cum_pnl", 0.0))
            self.n_settled = int(doc.get("n_settled", 0))
            self.n_wins = int(doc.get("n_wins", 0))
            self.extensions = int(doc.get("extensions", 0))
            self.pending = {int(k): v for k, v in doc.get("pending", {}).items()}
            self._trajectory = [
                (int(a), float(b)) for a, b in doc.get("trajectory", [])
            ]
        except (OSError, json.JSONDecodeError, TypeError, ValueError, KeyError) as e:
            raise InvariantError(
                f"shadow_ledger_state_parse_failed: {self._path} err={e}"
            ) from e

    def _persist(self) -> None:
        if self._path is None:
            return
        doc = {
            "active": self.active,
            "suspension_start_at": self.suspension_start_at,
            "suspension_bankroll": self.suspension_bankroll,
            "cum_pnl": self.cum_pnl,
            "n_settled": self.n_settled,
            "n_wins": self.n_wins,
            "extensions": self.extensions,
            "pending": {str(k): v for k, v in self.pending.items()},
            "trajectory": [[a, b] for a, b in self._trajectory],
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        tmp.replace(self._path)

    # -- lifecycle ----------------------------------------------------------

    def start(self, *, bankroll: float, start_at: int) -> None:
        """Begin tracking a new suspension (called when the breaker fires)."""
        self.active = True
        self.suspension_start_at = int(start_at)
        self.suspension_bankroll = float(bankroll)
        self.cum_pnl = 0.0
        self.n_settled = 0
        self.n_wins = 0
        self.extensions = 0
        self.pending = {}
        self._trajectory = [(int(start_at), float(bankroll))]
        self._persist()

    def extend(self) -> None:
        self.extensions += 1
        self._persist()

    def clear(self) -> None:
        """End tracking (suspension released)."""
        self.active = False
        self.suspension_start_at = None
        self.pending = {}
        self._persist()

    # -- bets ----------------------------------------------------------------

    def record_fire(self, *, epoch: int, side: str, size_bnb: float) -> None:
        if not self.active:
            return
        epoch = int(epoch)
        if epoch in self.pending:
            return
        self.pending[epoch] = {"side": str(side), "size_bnb": float(size_bnb)}
        self._persist()

    def settle_round(
        self,
        *,
        round_t: Round,
        treasury_fee_fraction: float,
        bet_gas_bnb: float,
    ) -> float | None:
        """Settle a pending shadow bet against a closed round; return pnl."""
        if not self.active:
            return None
        epoch = int(round_t.epoch)
        if epoch not in self.pending:
            return None
        # Settleability guard: the live engine passes epoch-tracking STUB
        # rounds (position=None, bets=()) through settle_closed_rounds; a
        # stub is not an outcome. Leave the bet pending (do NOT pop) until a
        # real closed round arrives — popping-then-raising here crashed the
        # live bot 5x on 2026-07-09.
        if round_t.position is None and not round_t.failed:
            return None
        # Settle BEFORE mutating any ledger state: if the settle math raises
        # (malformed round data, corrupt pending entry), the bet must remain
        # pending in memory AND on disk so a retry is clean — pop-then-raise
        # was the crash-loop-with-poisoned-state pattern (2026-07-09 review).
        bet = self.pending[epoch]
        outcome = settle_bet_against_closed_round(
            bet_bnb=float(bet["size_bnb"]),
            bet_side=str(bet["side"]),
            round_closed=round_t,
            treasury_fee_fraction=treasury_fee_fraction,
        )
        pnl = float(outcome.credit_bnb) - float(bet["size_bnb"]) - float(bet_gas_bnb)
        self.pending.pop(epoch)
        self.cum_pnl += pnl
        self.n_settled += 1
        if outcome.outcome == "win":
            self.n_wins += 1
        self._trajectory.append((int(round_t.start_at), self.hypo_bankroll()))
        self._prune(int(round_t.start_at))
        self._persist()
        return pnl

    # -- stats / release -----------------------------------------------------

    def hypo_bankroll(self) -> float:
        return self.suspension_bankroll + self.cum_pnl

    def hypo_peak(self, as_of_start_at: int) -> float:
        """Peak of the hypothetical trajectory within the trailing window."""
        floor_ts = int(as_of_start_at) - _TRAJECTORY_WINDOW_DAYS * 86400
        vals = [b for a, b in self._trajectory if a >= floor_ts]
        return max(vals) if vals else self.hypo_bankroll()

    def _prune(self, as_of_start_at: int) -> None:
        floor_ts = int(as_of_start_at) - _TRAJECTORY_WINDOW_DAYS * 86400
        # Keep at least the most recent point regardless of age.
        kept = [(a, b) for a, b in self._trajectory if a >= floor_ts]
        self._trajectory = kept if kept else self._trajectory[-1:]

    def release_ok(
        self,
        *,
        min_fires: int,
        recovery_frac: float,
        as_of_start_at: int,
    ) -> tuple[bool, str]:
        """Both-must-hold release test (plus a small-n evidence floor)."""
        if self.n_settled < int(min_fires):
            return False, f"insufficient_fires:{self.n_settled}<{int(min_fires)}"
        if self.cum_pnl < 0.0:
            return False, f"bleeding:cum_pnl={self.cum_pnl:.4f}"
        hypo = self.hypo_bankroll()
        threshold = self.hypo_peak(as_of_start_at) * float(recovery_frac)
        if hypo <= threshold:
            return False, f"below_recovery:{hypo:.4f}<={threshold:.4f}"
        return True, (
            f"recovered:n={self.n_settled},cum_pnl={self.cum_pnl:.4f},"
            f"hypo={hypo:.4f}>{threshold:.4f}"
        )

    def stats(self) -> dict:
        return {
            "shadow_n_settled": int(self.n_settled),
            "shadow_n_wins": int(self.n_wins),
            "shadow_cum_pnl": round(float(self.cum_pnl), 6),
            "shadow_hypo_bankroll": round(self.hypo_bankroll(), 6),
            "shadow_extensions": int(self.extensions),
            "shadow_pending": len(self.pending),
        }
