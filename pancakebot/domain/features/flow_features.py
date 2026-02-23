"""Within-round and late-phase pool/flow features (canonical v8).

This module emits ONLY canonical v8 feature names. It does not depend on any schema
objects and does not perform any column-order logic.

Definitions
- cutoff_ts = lock_ts - cutoff_seconds
- Within-round windows are defined over the pre-cutoff interval [start_at, cutoff_ts].
  Windows are expressed as percent ranges of that interval:
    - w_p_0_to_p_50
    - w_p_50_to_p_100
    - w_p_0_to_p_100

- Late-phase features are lagged to the target and computed from the most recent
  prior context round (prior_context_rounds[-1]) using bets in (cutoff_ts, lock_ts].

Missingness
- If required inputs are missing for a feature, the feature value MUST be NaN.
- log_imb uses epsilon smoothing and MUST be defined for all numeric bull/bear sums.

Imbalance sign convention (v8)
- log_imb = log((bull_value + eps) / (bear_value + eps))
"""

from __future__ import annotations

import math
from typing import Iterable

from pancakebot.domain.types import Bet


_WEI_PER_BNB = 1_000_000_000_000_000_000
_EPS = 1e-12


def _to_bnb(amount_wei: int) -> float:
    return float(amount_wei) / float(_WEI_PER_BNB)


def _cutoff_ts(lock_ts: int, cutoff_seconds: int) -> int:
    return int(lock_ts) - int(cutoff_seconds)


def _window_bucket(*, start_ts: int, cutoff_ts: int, created_at: int) -> str | None:
    """Return the canonical window label for a bet within the pre-cutoff interval.

    Returns:
      - "w_p_0_to_p_50" if created_at is in the first half
      - "w_p_50_to_p_100" if created_at is in the second half
      - None if created_at is outside [start_ts, cutoff_ts]
    """
    if int(created_at) < int(start_ts) or int(created_at) > int(cutoff_ts):
        return None
    span = int(cutoff_ts) - int(start_ts)
    if span <= 0:
        return None
    # Use integer arithmetic to avoid float drift.
    # First half is [0, span/2), second half is [span/2, span].
    if (int(created_at) - int(start_ts)) * 2 < span:
        return "w_p_0_to_p_50"
    return "w_p_50_to_p_100"


def _gini(values: list[float]) -> float:
    """Gini coefficient for non-negative values."""
    if not values:
        return float("nan")
    v = [x for x in values if x >= 0.0]
    if not v:
        return float("nan")
    total = float(sum(v))
    if total <= 0.0:
        return float("nan")
    v_sorted = sorted(v)
    n = len(v_sorted)
    cum = 0.0
    for i, x in enumerate(v_sorted, start=1):
        cum += float(i) * float(x)
    return (2.0 * cum) / (n * total) - (n + 1.0) / n


def _hhi(values: list[float]) -> float:
    if not values:
        return float("nan")
    v = [x for x in values if x >= 0.0]
    if not v:
        return float("nan")
    total = float(sum(v))
    if total <= 0.0:
        return float("nan")
    return float(sum((x / total) ** 2 for x in v))


def _max(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(max(values))


def _log_imb(*, bull: float, bear: float) -> float:
    return float(math.log((float(bull) + _EPS) / (float(bear) + _EPS)))


def compute_within_round_features(
    *,
    bets: Iterable[Bet],
    start_ts: int,
    lock_ts: int,
    cutoff_seconds: int,
) -> dict[str, float]:
    """Compute canonical within-round window features for the target round."""
    cutoff = _cutoff_ts(lock_ts, cutoff_seconds)
    if int(start_ts) <= 0 or int(lock_ts) <= 0:
        # Missing inputs
        return {}

    # Collect per-window per-side amounts/counts.
    sums: dict[str, dict[str, float]] = {
        "w_p_0_to_p_50": {"Bull": 0.0, "Bear": 0.0},
        "w_p_50_to_p_100": {"Bull": 0.0, "Bear": 0.0},
        "w_p_0_to_p_100": {"Bull": 0.0, "Bear": 0.0},
    }
    counts: dict[str, dict[str, float]] = {
        "w_p_0_to_p_50": {"Bull": 0.0, "Bear": 0.0},
        "w_p_50_to_p_100": {"Bull": 0.0, "Bear": 0.0},
        "w_p_0_to_p_100": {"Bull": 0.0, "Bear": 0.0},
    }
    bull_vals: dict[str, list[float]] = {"w_p_0_to_p_50": [], "w_p_50_to_p_100": [], "w_p_0_to_p_100": []}
    bear_vals: dict[str, list[float]] = {"w_p_0_to_p_50": [], "w_p_50_to_p_100": [], "w_p_0_to_p_100": []}

    for b in bets:
        if int(b.created_at) > int(cutoff):
            continue
        bucket = _window_bucket(start_ts=int(start_ts), cutoff_ts=int(cutoff), created_at=int(b.created_at))
        if bucket is None:
            continue

        amt = _to_bnb(int(b.amount_wei))
        pos = str(b.position)
        if pos not in ("Bull", "Bear"):
            continue

        # window-specific
        sums[bucket][pos] += amt
        counts[bucket][pos] += 1.0
        if pos == "Bull":
            bull_vals[bucket].append(amt)
        else:
            bear_vals[bucket].append(amt)

        # full window
        sums["w_p_0_to_p_100"][pos] += amt
        counts["w_p_0_to_p_100"][pos] += 1.0
        if pos == "Bull":
            bull_vals["w_p_0_to_p_100"].append(amt)
        else:
            bear_vals["w_p_0_to_p_100"].append(amt)

    out: dict[str, float] = {}

    for w in ("w_p_0_to_p_50", "w_p_50_to_p_100", "w_p_0_to_p_100"):
        bull = float(sums[w]["Bull"])
        bear = float(sums[w]["Bear"])
        total = bull + bear

        bull_n = float(counts[w]["Bull"])
        bear_n = float(counts[w]["Bear"])
        total_n = bull_n + bear_n

        out[f"bull_sum_{w}"] = bull
        out[f"bear_sum_{w}"] = bear
        out[f"total_sum_{w}"] = total

        out[f"bull_n_{w}"] = bull_n
        out[f"bear_n_{w}"] = bear_n
        out[f"total_n_{w}"] = total_n

        out[f"has_any_bets_{w}"] = 1.0 if total_n > 0.0 else 0.0
        out[f"has_bull_bets_{w}"] = 1.0 if bull_n > 0.0 else 0.0
        out[f"has_bear_bets_{w}"] = 1.0 if bear_n > 0.0 else 0.0

        out[f"log_imb_{w}"] = _log_imb(bull=bull, bear=bear)

        # concentration (NaN if side has no bets)
        out[f"max_bet_bull_{w}"] = _max(bull_vals[w])
        out[f"max_bet_bear_{w}"] = _max(bear_vals[w])
        out[f"hhi_bull_{w}"] = _hhi(bull_vals[w])
        out[f"hhi_bear_{w}"] = _hhi(bear_vals[w])
        out[f"gini_bull_{w}"] = _gini(bull_vals[w])
        out[f"gini_bear_{w}"] = _gini(bear_vals[w])

    # dynamics ratios
    lo = "w_p_0_to_p_50"
    hi = "w_p_50_to_p_100"
    for side in ("bull", "bear", "total"):
        num = float(out[f"{side}_sum_{hi}"])
        den = float(out[f"{side}_sum_{lo}"])
        key = f"{side}_sum_ratio_{hi}_over_{lo}"
        out[key] = float("nan") if den == 0.0 else float(num / den)

    return out


def compute_late_phase_features(
    *,
    bets: Iterable[Bet],
    lock_ts: int | None,
    cutoff_seconds: int,
) -> dict[str, float]:
    """Compute lagged late-phase features for the prior context round.

    Late-phase is the post-cutoff, pre-lock interval: (cutoff_ts, lock_ts].
    """
    if lock_ts is None or int(lock_ts) <= 0:
        return {
            "late_bull_sum": float("nan"),
            "late_bear_sum": float("nan"),
            "late_total_sum": float("nan"),
            "late_bull_n": float("nan"),
            "late_bear_n": float("nan"),
            "late_total_n": float("nan"),
            "late_log_imb": float("nan"),
        }
    cutoff = _cutoff_ts(int(lock_ts), int(cutoff_seconds))

    bull = 0.0
    bear = 0.0
    bull_n = 0.0
    bear_n = 0.0

    for b in bets:
        if int(b.created_at) <= int(cutoff) or int(b.created_at) > int(lock_ts):
            continue
        amt = _to_bnb(int(b.amount_wei))
        pos = str(b.position)
        if pos == "Bull":
            bull += amt
            bull_n += 1.0
        elif pos == "Bear":
            bear += amt
            bear_n += 1.0

    total = bull + bear
    total_n = bull_n + bear_n

    return {
        "late_bull_sum": float(bull),
        "late_bear_sum": float(bear),
        "late_total_sum": float(total),
        "late_bull_n": float(bull_n),
        "late_bear_n": float(bear_n),
        "late_total_n": float(total_n),
        "late_log_imb": _log_imb(bull=bull, bear=bear),
    }
