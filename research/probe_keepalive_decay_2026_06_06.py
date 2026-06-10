"""Map the write-endpoint keep-alive decay: after warming a connection, how
long can it sit idle before the next call pays a cold (reconnect) penalty?

Decides the pre-cache design: if the bet send (cached nonce+gas => send_raw is
the only write RPC) happens ~4.5s after the preflight-wake warm, is the
connection still hot? If keep-alive >> 4.5s, warming at the preflight wake keeps
send_raw ~30ms. If << 4.5s, send_raw goes cold (~110ms) and we need a later warm.

Read-only (eth_chainId). Run on VM:
  cd /root/pancakebot && PYTHONPATH=/root/pancakebot ./.venv/bin/python \\
      research/probe_keepalive_decay_2026_06_06.py
"""
from __future__ import annotations

import time

from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware

from pancakebot.constants import WRITE_PATH_RPC_URLS

URL = WRITE_PATH_RPC_URLS[0]
IDLES = [0.0, 1.0, 2.0, 3.0, 4.0, 4.5, 5.0, 6.0, 8.0, 10.0, 15.0, 30.0]
REPEATS = 3  # per idle, take the median to cut jitter


def _w3():
    w = Web3(Web3.HTTPProvider(URL, request_kwargs={"timeout": 20}))
    w.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
    return w


def _call_ms(w):
    t0 = time.perf_counter()
    w.eth.gas_price          # same RPC class the send endpoint serves
    return (time.perf_counter() - t0) * 1000.0


print(f"endpoint: {URL}")
print(f"{'idle_s':>7} {'after-idle ms (median of 3)':>30}")
for idle in IDLES:
    samples = []
    for _ in range(REPEATS):
        w = _w3()
        _call_ms(w)            # warm: first call pays handshake
        _call_ms(w)            # ensure hot
        time.sleep(idle)
        samples.append(_call_ms(w))  # the measured post-idle call
        del w
        time.sleep(0.3)
    samples.sort()
    med = samples[len(samples) // 2]
    tag = "  <- hot" if med < 55 else ("  <- COLD (reconnect)" if med > 80 else "")
    print(f"{idle:>7.1f} {med:>26.1f}{tag}")
