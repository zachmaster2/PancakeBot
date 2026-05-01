"""p3a wallet-aware features computation.

Per orchestrator v2.1 (locked):

13 features per round, each computed using ONLY data with timestamp <
lock_at - 2s (data horizon discipline). Wallet history is set-aggregated
over rounds R such that R.lock_at < N.lock_at - 2s AND R has been
settled (settled_at = R+1.close_time ≈ R.lock_at + 600s).

Tier 0 (confounder-mediated):
  1. n_bettors
  2. pool_size_bnb
  5. bull_pool_ratio
  12. bet_count_velocity

Tier 1 (microstructure-specific):
  3. top1_wallet_share
  4. top5_concentration
  6. consensus_ratio (majority_count / n_bettors)
  7. smart_bias
  8. fresh_ratio
  9. repeat_loser_ratio
  10. late_flow_ratio (window [lock_at - 30, lock_at - 2))
  11. whale_presence (binary)
  13. late_flow_directional_bias

Wallet history is built up by chronological iteration. For each round N
processed in epoch order:
  1. Compute features for N using wallet_history-as-of-cutoff (lock_at - 2)
  2. AFTER round N's features are computed, settle: each wallet in N
     gets its history updated when N's settle_time has passed (in
     practice: as soon as N's outcome is known, which is at N's close_at).
     For a chronological pass over the dataset, we update wallet_history
     after processing N's features.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

DATA_HORIZON_OFFSET_S = 2
LATE_FLOW_WINDOW_S = 30        # window [lock_at - 30, lock_at - 2)
BET_VELOCITY_WINDOW_S = 60     # window [lock_at - 60, lock_at - 2)
MIN_PRIOR_BETS = 10            # wallets with <10 prior settled bets are "fresh"
WHALE_THRESHOLD = 0.10         # wallet bet > 10% of pool
REPEAT_LOSER_WR_THRESHOLD = 0.40
BNB_WEI = 1_000_000_000_000_000_000


@dataclass
class WalletHistory:
    """Aggregated history for one wallet. Set-style: order doesn't matter."""
    n_settled_bets: int = 0
    n_wins: int = 0
    total_volume_bnb: float = 0.0

    def win_rate(self) -> float:
        if self.n_settled_bets == 0:
            return 0.0
        return self.n_wins / self.n_settled_bets

    def is_fresh(self) -> bool:
        return self.n_settled_bets < MIN_PRIOR_BETS

    def is_repeat_loser(self) -> bool:
        return (not self.is_fresh()) and self.win_rate() < REPEAT_LOSER_WR_THRESHOLD


@dataclass
class WalletHistoryStore:
    """Cumulative history for ALL wallets seen so far."""
    by_wallet: dict[str, WalletHistory] = field(default_factory=dict)

    def get(self, wallet: str) -> WalletHistory:
        h = self.by_wallet.get(wallet)
        if h is None:
            h = WalletHistory()
            self.by_wallet[wallet] = h
        return h

    def settle_round(self, bets: Sequence[dict], winning_position: str | None) -> None:
        """Update each wallet's history after a round settles. Idempotent if
        a round is settled twice with the same args (but caller should not
        do that)."""
        if winning_position is None:
            return  # House: refund, no settled bets recorded
        for b in bets:
            w = b["wallet"]
            h = self.get(w)
            h.n_settled_bets += 1
            if b["position"] == winning_position:
                h.n_wins += 1
            h.total_volume_bnb += int(b["amountWei"]) / BNB_WEI


def compute_features(round_rec: dict, history: WalletHistoryStore) -> dict:
    """Compute 13 wallet-aware features for one round.

    `round_rec` follows the closed_rounds.jsonl schema (epoch, startAt,
    lockPrice, closePrice, position, failed, bets[]).

    `history` reflects wallet histories as-of the round's data horizon
    (lock_at - 2s). Caller is responsible for not having settled THIS
    round into history yet.

    Returns a dict of all 13 features (some may be NaN if undefined,
    e.g. no bets pre-cutoff).
    """
    import math
    start_at = int(round_rec["startAt"])
    lock_at = start_at + 300  # canonical round duration
    cutoff = lock_at - DATA_HORIZON_OFFSET_S
    late_lo = lock_at - LATE_FLOW_WINDOW_S
    velocity_lo = lock_at - BET_VELOCITY_WINDOW_S

    # Pre-cutoff bets
    bets = [b for b in (round_rec.get("bets") or []) if int(b["createdAt"]) < cutoff]
    nan = float("nan")

    if not bets:
        # No bets observable pre-cutoff; all features undefined except trivially zero ones
        return {f: nan for f in (
            "n_bettors", "pool_size_bnb", "top1_wallet_share", "top5_concentration",
            "bull_pool_ratio", "consensus_ratio", "smart_bias", "fresh_ratio",
            "repeat_loser_ratio", "late_flow_ratio", "whale_presence",
            "bet_count_velocity", "late_flow_directional_bias",
        )}

    # Per-wallet aggregation (sum of all bets by that wallet pre-cutoff)
    by_wallet_size: dict[str, int] = {}
    by_wallet_position_size: dict[str, dict[str, int]] = {}
    bull_wei = 0
    bear_wei = 0
    for b in bets:
        w = b["wallet"]
        amt = int(b["amountWei"])
        pos = b["position"]
        by_wallet_size[w] = by_wallet_size.get(w, 0) + amt
        if w not in by_wallet_position_size:
            by_wallet_position_size[w] = {"Bull": 0, "Bear": 0}
        by_wallet_position_size[w][pos] += amt
        if pos == "Bull":
            bull_wei += amt
        else:
            bear_wei += amt

    total_wei = bull_wei + bear_wei
    pool_size_bnb = total_wei / BNB_WEI
    n_bettors = len(by_wallet_size)
    bull_pool_ratio = bull_wei / total_wei if total_wei > 0 else nan

    # Top-share metrics use per-wallet aggregated size (not per-bet)
    sizes = sorted(by_wallet_size.values(), reverse=True)
    top1 = sizes[0] if sizes else 0
    top5_sum = sum(sizes[:5])
    top1_share = top1 / total_wei if total_wei > 0 else nan
    top5_concentration = top5_sum / total_wei if total_wei > 0 else nan
    whale_presence = 1.0 if (top1 / total_wei >= WHALE_THRESHOLD) else 0.0 if total_wei > 0 else nan

    # Consensus ratio (Tier 1, reformulated per v2.1 Q1)
    bull_count = sum(1 for w, ps in by_wallet_position_size.items() if ps["Bull"] >= ps["Bear"])
    bear_count = n_bettors - bull_count
    majority_count = max(bull_count, bear_count)
    consensus_ratio = majority_count / n_bettors if n_bettors > 0 else nan

    # Smart bias: Σ(wallet_net_signed_size × wallet_WR_signed) / pool_size_bnb
    # where wallet_net_signed_size = bull_size - bear_size for that wallet (signed),
    # and wallet_WR_signed weights the contribution.
    # Per v2.1 R4: only wallets with ≥10 prior settled bets contribute.
    smart_num = 0.0
    n_smart_eligible = 0
    n_repeat_losers = 0
    n_fresh = 0
    for w, ps in by_wallet_position_size.items():
        h = history.by_wallet.get(w)
        if h is None or h.is_fresh():
            n_fresh += 1
            continue
        n_smart_eligible += 1
        if h.is_repeat_loser():
            n_repeat_losers += 1
        # Net direction-weighted signed size
        wallet_net_signed_bnb = (ps["Bull"] - ps["Bear"]) / BNB_WEI
        wr_signed = (h.win_rate() - 0.5) * 2.0  # in [-1, +1]
        smart_num += wallet_net_signed_bnb * wr_signed
    smart_bias = smart_num / pool_size_bnb if pool_size_bnb > 0 else nan
    fresh_ratio = n_fresh / n_bettors if n_bettors > 0 else nan
    repeat_loser_ratio = (
        n_repeat_losers / n_smart_eligible if n_smart_eligible > 0 else 0.0
    )

    # Late-flow features
    late_bull_wei = 0
    late_bear_wei = 0
    velocity_count = 0
    for b in bets:
        ts = int(b["createdAt"])
        if late_lo <= ts < cutoff:
            amt = int(b["amountWei"])
            if b["position"] == "Bull":
                late_bull_wei += amt
            else:
                late_bear_wei += amt
        if velocity_lo <= ts < cutoff:
            velocity_count += 1
    late_total_wei = late_bull_wei + late_bear_wei
    late_flow_ratio = late_total_wei / total_wei if total_wei > 0 else nan
    bet_count_velocity = velocity_count / (BET_VELOCITY_WINDOW_S - DATA_HORIZON_OFFSET_S)
    if late_total_wei > 0:
        late_flow_directional_bias = late_bull_wei / late_total_wei
    else:
        late_flow_directional_bias = nan

    return {
        "n_bettors": float(n_bettors),
        "pool_size_bnb": pool_size_bnb,
        "top1_wallet_share": top1_share,
        "top5_concentration": top5_concentration,
        "bull_pool_ratio": bull_pool_ratio,
        "consensus_ratio": consensus_ratio,
        "smart_bias": smart_bias,
        "fresh_ratio": fresh_ratio,
        "repeat_loser_ratio": repeat_loser_ratio,
        "late_flow_ratio": late_flow_ratio,
        "whale_presence": whale_presence,
        "bet_count_velocity": bet_count_velocity,
        "late_flow_directional_bias": late_flow_directional_bias,
    }


FEATURE_NAMES = [
    "n_bettors", "pool_size_bnb", "top1_wallet_share", "top5_concentration",
    "bull_pool_ratio", "consensus_ratio", "smart_bias", "fresh_ratio",
    "repeat_loser_ratio", "late_flow_ratio", "whale_presence",
    "bet_count_velocity", "late_flow_directional_bias",
]

TIER0_FEATURES = {"n_bettors", "pool_size_bnb", "bull_pool_ratio", "bet_count_velocity"}
TIER1_FEATURES = {n for n in FEATURE_NAMES if n not in TIER0_FEATURES}


def round_outcome(round_rec: dict) -> int | None:
    """Return 1 if Bull won, 0 if Bear won, None for House (close==lock) or
    failed/unsettled rounds."""
    if round_rec.get("failed"):
        return None
    pos = round_rec.get("position")
    if pos == "Bull":
        return 1
    if pos == "Bear":
        return 0
    return None  # House or unknown


def build_features_chronological(rounds: Iterable[dict]) -> tuple[list[dict], list[int]]:
    """Walk rounds in epoch order. For each, compute features as-of round's
    horizon, then settle the round into wallet history.

    Returns (features_list, outcomes_list). features_list[i] is the feature
    dict for the i-th round; outcomes_list[i] is its binary outcome (1=Bull,
    0=Bear, or -1 for House/failed/unsettled — caller should filter).
    """
    history = WalletHistoryStore()
    feats: list[dict] = []
    outs: list[int] = []
    for r in rounds:
        f = compute_features(r, history)
        feats.append(f)
        o = round_outcome(r)
        outs.append(o if o is not None else -1)
        # Settle this round into history (only if it has a valid winning position).
        # Per the temporal rule: this round's bets affect future rounds' history,
        # NOT the current round's features.
        winning = r.get("position") if not r.get("failed") else None
        if winning in ("Bull", "Bear"):
            history.settle_round(r.get("bets") or [], winning)
    return feats, outs
