# `eth_sendRawTransaction` RTT probe — BSC mainnet

Date: 2026-05-16
Wallet: `0xaF966D00698F92DeBe2127136D5159c5a51dA5E7`
RPC: `https://bsc-dataseed1.defibit.io`
Chain ID: 56
Gas price at start: 0.05 Gwei
Starting nonce: 681
Starting balance: 0.233025 BNB

## Results

- TXs attempted: **100**
- TXs accepted by RPC (RTT measured): **100**
- TXs included on-chain (within 30s): **100**
- Inclusion rate (of sent): **100.0%**

### Round-trip RTT (TX-signed → RPC-response, ms)

| stat | value |
|---|---:|
| n | 100 |
| mean | 33.6 ms |
| p50 | 32.5 ms |
| p90 | 46.9 ms |
| p95 | 49.5 ms |
| p99 | 78.0 ms |
| max | 78.0 ms |

### Inclusion lag (RPC-response → on-chain block, ms)

| stat | value |
|---|---:|
| n | 100 |
| mean | 937.3 ms |
| p50 | 953.1 ms |
| p90 | 1203.8 ms |
| p95 | 1409.9 ms |
| p99 | 1789.4 ms |
| max | 1789.4 ms |

## Recommendation for `BSC_BET_SUBMIT_ONE_WAY_MS`

Empirical p95 round-trip: **49ms**. One-way estimate (p95/2 + propagation, rounded up to 50ms quantum): **50ms**. Current placeholder `BSC_BET_SUBMIT_ONE_WAY_MS = 150`. **Recommend lowering to 50ms (tighter deadline math, more critical-path budget).**
