"""Integer-seconds wall-clock timestamp helper."""
import time


def now_ts() -> int:
    return int(time.time())
