from __future__ import annotations

from pancakebot.domain.types import Kline, Round

_ONE_MINUTE_MS = 60_000
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
    target_lock_ts = int(target_round.lock_at)

    rounds = list(prior_context_rounds)
    for i in range(1, len(rounds)):
        prev = rounds[i - 1]
        cur = rounds[i]

        if prev.lock_at is None or cur.lock_at is None:
            return False, "round_lock_at_missing"

        prev_epoch = int(prev.epoch)
        cur_epoch = int(cur.epoch)
        if int(cur_epoch - prev_epoch) != 1:
            return False, "round_epoch_gap"

        lock_delta = int(cur.lock_at) - int(prev.lock_at)
        if abs(int(lock_delta) - int(_LOCK_INTERVAL_SECONDS)) > int(buffer_seconds):
            return False, "round_lock_gap"

    prev = rounds[-1]
    if prev.lock_at is None:
        return False, "round_lock_at_missing"

    prev_epoch = int(prev.epoch)
    target_epoch = int(target_round.epoch)
    if int(target_epoch - prev_epoch) != 1:
        return False, "round_epoch_gap"

    lock_delta = int(target_lock_ts) - int(prev.lock_at)
    if abs(int(lock_delta) - int(_LOCK_INTERVAL_SECONDS)) > int(buffer_seconds):
        return False, "round_lock_gap"

    return True, ""


def check_klines_contiguous(*, context_klines: list[Kline]) -> tuple[bool, str]:
    if len(context_klines) < 2:
        return True, ""

    prev_open = int(context_klines[0].open_time_ms)
    for i in range(1, len(context_klines)):
        cur_open = int(context_klines[i].open_time_ms)
        if int(cur_open - prev_open) != int(_ONE_MINUTE_MS):
            return False, "kline_open_gap"
        prev_open = int(cur_open)

    return True, ""
