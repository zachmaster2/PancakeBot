def cutoff_ts(lock_ts: int, cutoff_seconds: int) -> int:
    return lock_ts - cutoff_seconds
