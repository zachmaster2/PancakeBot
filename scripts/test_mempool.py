"""Test the mempool watcher standalone — connects to WSS and logs pending bets.

Usage:
    python scripts/test_mempool.py

Requires BSC_WSS_URL in .env (e.g., wss://xxx.bsc.quiknode.pro/your-key/)
Runs for 2 minutes, printing every pending PancakeSwap bet it sees.
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from pancakebot.infra.mempool_watcher import MempoolWatcher

load_dotenv()

wss_url = os.getenv("BSC_WSS_URL", "").strip()
if not wss_url:
    print("ERROR: BSC_WSS_URL not set in .env")
    print("Add: BSC_WSS_URL=wss://your-quicknode-endpoint")
    sys.exit(1)

print(f"Connecting to: {wss_url[:40]}...")
watcher = MempoolWatcher(wss_url=wss_url)
watcher.start()

# Wait for connection
for i in range(30):
    if watcher.connected:
        break
    time.sleep(0.5)

if not watcher.connected:
    print("ERROR: Failed to connect after 15 seconds")
    watcher.stop()
    sys.exit(1)

print(f"Connected! Watching for pending PancakeSwap bets...")
print(f"(will run for 2 minutes)\n")

start = time.time()
last_stats = None

while time.time() - start < 120:
    stats = watcher.stats
    if stats != last_stats:
        elapsed = time.time() - start
        print(f"[{elapsed:6.1f}s] seen={stats['total_seen']} "
              f"matched={stats['total_matched']} "
              f"pending={stats['pending_count']}")
        last_stats = stats
    time.sleep(2)

print(f"\nFinal stats: {watcher.stats}")
watcher.stop()
print("Done.")
