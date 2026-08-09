#!/usr/bin/env python3
"""CLI wrapper retained for an obvious server GPU-smoke entry point."""

from __future__ import annotations

import argparse

from beeid.cli import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    arguments = parser.parse_args()
    raise SystemExit(main(["gpu-smoke", "--config", arguments.config]))
