from __future__ import annotations

from pancakebot.domain.types import Round

_LOCK_INTERVAL_SECONDS = 300


def check_rounds_contiguous(
    *,
    prior_context_rounds: list[Round],
    target_round: Round,
    buffer_seconds: int,
) -> tuple[bool, str]:
    if not prior_context_rounds:
        return True, ""

    if target_round.lock_at is None:
        return False, "target_lock_at_missing"

    rounds = list(prior_context_rounds)
    for i in range(1, len(rounds)):
        prev = rounds[i - 1]
        cur = rounds[i]

        if prev.lock_at is None or cur.lock_at is None:
            return False, "round_lock_at_missing"

        if cur.epoch - prev.epoch != 1:
            return False, "round_epoch_gap"

        lock_delta = cur.lock_at - prev.lock_at
        if abs(lock_delta - _LOCK_INTERVAL_SECONDS) > buffer_seconds:
            return False, "round_lock_gap"

    prev = rounds[-1]
    if prev.lock_at is None:
        return False, "round_lock_at_missing"

    if target_round.epoch - prev.epoch != 1:
        return False, "round_epoch_gap"

    lock_delta = target_round.lock_at - prev.lock_at
    if abs(lock_delta - _LOCK_INTERVAL_SECONDS) > buffer_seconds:
        return False, "round_lock_gap"

    return True, ""
