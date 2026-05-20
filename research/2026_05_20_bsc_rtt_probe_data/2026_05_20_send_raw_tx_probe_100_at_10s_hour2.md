# `eth_sendRawTransaction` RTT probe — BSC mainnet

Date: 2026-05-16
Wallet: `0xaF966D00698F92DeBe2127136D5159c5a51dA5E7`
RPC: `https://bsc-dataseed1.defibit.io`
Chain ID: 56
Gas price at start: 0.05 Gwei
Starting nonce: 781
Starting balance: 0.232920 BNB

## Results

- TXs attempted: **100**
- TXs accepted by RPC (RTT measured): **100**
- TXs included on-chain (within 30s): **100**
- Inclusion rate (of sent): **100.0%**

### Round-trip RTT (TX-signed → RPC-response, ms)

| stat | value |
|---|---:|
| n | 100 |
| mean | 37.8 ms |
| p50 | 35.4 ms |
| p90 | 48.3 ms |
| p95 | 52.5 ms |
| p99 | 126.5 ms |
| max | 126.5 ms |

### Inclusion lag (RPC-response → on-chain block, ms)

| stat | value |
|---|---:|
| n | 100 |
| mean | 906.5 ms |
| p50 | 935.9 ms |
| p90 | 1179.0 ms |
| p95 | 1191.1 ms |
| p99 | 1628.2 ms |
| max | 1628.2 ms |

## Recommendation for `BSC_BET_SUBMIT_ONE_WAY_MS`

Empirical p95 round-trip: **52ms**. One-way estimate (p95/2 + propagation, rounded up to 50ms quantum): **50ms**. Current placeholder `BSC_BET_SUBMIT_ONE_WAY_MS = 150`. **Recommend lowering to 50ms (tighter deadline math, more critical-path budget).**
