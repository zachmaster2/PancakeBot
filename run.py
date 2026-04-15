from __future__ import annotations

import argparse

from pancakebot.app import run_from_config
from pancakebot.log import info


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="run.py")
    p.add_argument("--config", type=str, default="config.toml")
    p.add_argument("--dry", action="store_true", default=False, help="Run in dry mode (no on-chain tx)")
    p.add_argument("--backtest", action="store_true", default=False, help="Run backtest (no sleeping, simulated betting)")
    p.add_argument("--sync", action="store_true", default=False, help="Sync closed rounds and OKX klines, then exit")
    args = p.parse_args(argv)
    selected_modes = int(bool(args.dry)) + int(bool(args.backtest)) + int(bool(args.sync))
    if selected_modes > 1:
        p.error("--dry, --backtest, and --sync are mutually exclusive")
    return args


def main() -> None:
    args = _parse_args()
    try:
        run_from_config(
            config_path=args.config,
            dry=bool(args.dry),
            backtest=bool(args.backtest),
            sync=bool(args.sync),
        )
    except KeyboardInterrupt:
        info("CORE", "RUN", "EXIT", msg="Caught KeyboardInterrupt: shutting down")


if __name__ == "__main__":
    main()
