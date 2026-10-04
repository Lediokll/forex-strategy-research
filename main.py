"""
Entry point.

    py main.py --download                 # fetch M15 history + symbol spec from MT5
    py main.py --mode backtest            # walk-forward backtest + report
    py main.py                            # demo (default mode from config.yaml)
    py main.py --mode live --confirm-live # live: also needs live.enabled + account_number
"""

from __future__ import annotations

import argparse
import logging
import sys

from src import config as cfg_module

logger = logging.getLogger(__name__)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EURUSD M15 pullback bot (MT5)")
    parser.add_argument("--mode", choices=cfg_module.MODES, help="overrides `mode` in config.yaml")
    parser.add_argument("--config", help="path to a config file (default: config.yaml)")
    parser.add_argument("--download", action="store_true", help="download history from MT5 and exit")
    parser.add_argument("--confirm-live", action="store_true", help="required to run in live mode")
    parser.add_argument("--reset-virtual", action="store_true", help="start a new virtual account")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg_module.setup_logging()

    try:
        overrides = {"mode": args.mode} if args.mode else None
        cfg = cfg_module.load_config(args.config, overrides)
    except cfg_module.ConfigError as error:
        logger.error("Invalid configuration: %s", error)
        return 2

    if args.download:
        from src.data.mt5_data import download_history

        download_history(cfg)
        return 0

    if cfg.mode == "backtest":
        from src.backtest.runner import run_backtest

        run_backtest(cfg)
        return 0

    logger.error("Mode %r: the live/demo loop is not built yet (phase 2).", cfg.mode)
    return 1


if __name__ == "__main__":
    sys.exit(main())
