"""Progress checkpoint for the running inclusion experiment. Blocks until >=TARGET
rounds published (or coordinator_done), then prints analysis + procs + balances.
Run via run_in_background; completes -> surfaces a checkpoint. Halt-watch fields:
proc count, balances, ipc_late/no_incl in the analysis.

Usage: python research/_experiment_watch.py <logdir> <target_rounds>
"""
import json
import os
import sys
import time
import urllib.request as R

logdir = sys.argv[1]
target = int(sys.argv[2])
cpath = os.path.join(logdir, "coordinator.jsonl")


def count(ev):
    if not os.path.exists(cpath):
        return 0
    return sum(1 for l in open(cpath) if f'"{ev}"' in l)


for _ in range(120):                       # <= 40 min safety cap
    if count("coordinator_done") >= 1 or count("deadline_published") >= target:
        break
    time.sleep(20)

print("=== ANALYSIS ===", flush=True)
os.system(f"PYTHONPATH=/root/pancakebot {sys.executable} "
          f"/root/pancakebot/research/analyze_inclusion_experiment.py {logdir}")

print("\n=== EXPERIMENT PROCS (0 => finished) ===", flush=True)
os.system("ps -eo cmd | grep -c '[i]nclusion_experiment_2026'")

print("\n=== BALANCES ===", flush=True)
kv = {}
for line in open("/etc/pancakebot/experiment_wallets.env"):
    line = line.strip()
    if line and "=" in line:
        k, v = line.split("=", 1)
        kv[k] = v
E = "https://bsc-dataseed1.binance.org"
tot = 0.0
for i in range(5):
    a = kv[f"WALLET_{i}_ADDR"]
    try:
        req = R.Request(E, data=json.dumps({"jsonrpc": "2.0", "id": 1, "method": "eth_getBalance",
                        "params": [a, "latest"]}).encode(), headers={"Content-Type": "application/json"})
        bal = int(json.load(R.urlopen(req, timeout=5))["result"], 16) / 1e18
    except Exception as e:
        bal = -1.0
    tot += max(0.0, bal)
    print(f"  W{i} {a}: {bal:.5f} BNB")
print(f"  total: {tot:.5f} BNB  (started 0.10000)")
print(f"\n=== done={count('coordinator_done') >= 1}  rounds_published={count('deadline_published')} ===")
