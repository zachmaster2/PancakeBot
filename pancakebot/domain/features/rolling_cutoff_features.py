"""Cutoff-only rolling regime features (Group H).

Locked spec: specs/FEATURES.md

Computed over cutoff snapshots of rounds j-1 ... j-N (includes t-1).
"""

from __future__ import annotations

from typing import Sequence

from pancakebot.domain.features._math import NAN
from pancakebot.domain.features.rolling_stats import mean_defined, sample_std_defined, tail


def compute_group_h(*, prior_cutoff_snapshots: Sequence[dict[str, float]]) -> dict[str, float]:
    """Compute Group H features.

    prior_cutoff_snapshots must be in epoch order ascending and contain only
    rounds j-1 ...; the caller provides the correct window.

    Each snapshot must include: total_amt_c, total_n_c, log_imbalance_c.
    """
    out: dict[str, float] = {}

    # for n in (20, 50, 100):
    for n in (5,):
        window = tail(prior_cutoff_snapshots, n)

        total_amt_vals = [float(s.get("total_amt_c", NAN)) for s in window]
        total_n_vals = [float(s.get("total_n_c", NAN)) for s in window]
        log_imb_vals = [float(s.get("log_imbalance_c", NAN)) for s in window]

        # Means require >= 1 defined value
        mean_total_amt, mean_total_amt_def = mean_defined(total_amt_vals)
        mean_total_n, mean_total_n_def = mean_defined(total_n_vals)
        mean_log_imb, mean_log_imb_def = mean_defined(log_imb_vals)

        out[f"roll_mean_total_amt_c_{n}"] = mean_total_amt
        out[f"roll_mean_total_n_c_{n}"] = mean_total_n
        out[f"roll_mean_log_imbalance_c_{n}"] = mean_log_imb

        # Sample std requires >= 2 defined values
        std_total_amt, std_total_amt_def = sample_std_defined(total_amt_vals)
        std_total_n, std_total_n_def = sample_std_defined(total_n_vals)
        std_log_imb, std_log_imb_def = sample_std_defined(log_imb_vals)

        out[f"roll_std_total_amt_c_{n}"] = std_total_amt
        out[f"roll_std_total_n_c_{n}"] = std_total_n
        out[f"roll_std_log_imbalance_c_{n}"] = std_log_imb

        # Defined masks are 1 iff at least 1 defined value exists for the series.
        out[f"roll_total_amt_c_{n}_defined"] = float(mean_total_amt_def)
        out[f"roll_total_n_c_{n}_defined"] = float(mean_total_n_def)
        out[f"roll_log_imbalance_c_{n}_defined"] = float(mean_log_imb_def)

        # Note: std defined masks are not part of the schema; the schema uses the
        # "at least one defined" masks above. We compute std per spec rules and
        # leave it as NaN when undefined.
        _ = std_total_amt_def
        _ = std_total_n_def
        _ = std_log_imb_def

    return out
