from __future__ import annotations

import argparse

from pancakebot.integration.app import run_from_config
from pancakebot.core.logging import info


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="run.py")
    p.add_argument("--config", type=str, default="config.toml")
    p.add_argument("--dry", action="store_true", default=False, help="Run in dry mode (no on-chain tx)")
    p.add_argument("--backtest", action="store_true", default=False, help="Run backtest (no sleeping, simulated betting)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        run_from_config(config_path=args.config, dry=bool(args.dry), backtest=bool(args.backtest))
    except KeyboardInterrupt:
        info("CORE", "RUN", "EXIT", msg="Caught KeyboardInterrupt: shutting down")


if __name__ == "__main__":
    main()
