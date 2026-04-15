from __future__ import annotations

import argparse
import sys

from pancakebot.app import run_from_config
from pancakebot.log import info


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run.py",
        description="PancakeBot — automated trading for PancakeSwap Prediction V2",
    )
    p.add_argument("--config", type=str, default="config.toml",
                   help="Path to config file (default: config.toml)")
    p.add_argument("--sync", action="store_true",
                   help="Fetch rounds + klines + contract constants from chain/OKX/Graph")
    p.add_argument("--backtest", action="store_true",
                   help="Replay historical data and compute PnL")
    p.add_argument("--dry", action="store_true",
                   help="Real-time paper trading (no on-chain transactions)")
    p.add_argument("--live", action="store_true",
                   help="Real-time trading with real BNB")
    args = p.parse_args(argv)

    selected = [args.sync, args.backtest, args.dry, args.live]
    if not any(selected):
        p.print_help()
        sys.exit(1)
    if sum(bool(s) for s in selected) > 1:
        p.error("--sync, --backtest, --dry, and --live are mutually exclusive")
    return args


def main() -> None:
    args = _parse_args()
    try:
        run_from_config(
            config_path=args.config,
            dry=bool(args.dry),
            backtest=bool(args.backtest),
            sync=bool(args.sync),
            live=bool(args.live),
        )
    except KeyboardInterrupt:
        info("CORE", "RUN", "EXIT", msg="Caught KeyboardInterrupt: shutting down")


if __name__ == "__main__":
    main()
