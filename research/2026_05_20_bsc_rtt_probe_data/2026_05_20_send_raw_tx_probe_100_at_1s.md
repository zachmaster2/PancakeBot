# `eth_sendRawTransaction` RTT probe — BSC mainnet

Date: 2026-05-16
Wallet: `0xaF966D00698F92DeBe2127136D5159c5a51dA5E7`
RPC: `https://bsc-dataseed1.defibit.io`
Chain ID: 56
Gas price at start: 0.05 Gwei
Starting nonce: 481
Starting balance: 0.233235 BNB

## Results

- TXs attempted: **100**
- TXs accepted by RPC (RTT measured): **100**
- TXs included on-chain (within 30s): **100**
- Inclusion rate (of sent): **100.0%**

### Round-trip RTT (TX-signed → RPC-response, ms)

| stat | value |
|---|---:|
| n | 100 |
| mean | 33.8 ms |
| p50 | 33.8 ms |
| p90 | 41.3 ms |
| p95 | 45.9 ms |
| p99 | 65.2 ms |
| max | 65.2 ms |

### Inclusion lag (RPC-response → on-chain block, ms)

| stat | value |
|---|---:|
| n | 100 |
| mean | 1000.2 ms |
| p50 | 962.8 ms |
| p90 | 1197.7 ms |
| p95 | 1425.7 ms |
| p99 | 2072.8 ms |
| max | 2072.8 ms |

## Recommendation for `BSC_BET_SUBMIT_ONE_WAY_MS`

Empirical p95 round-trip: **46ms**. One-way estimate (p95/2 + propagation, rounded up to 50ms quantum): **50ms**. Current placeholder `BSC_BET_SUBMIT_ONE_WAY_MS = 150`. **Recommend lowering to 50ms (tighter deadline math, more critical-path budget).**
