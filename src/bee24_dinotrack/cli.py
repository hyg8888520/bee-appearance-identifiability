"""Command-line entry point for the H1 experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .config import ConfigurationError, load_config
from .experiment import ExperimentError, run_h1
from .validation import StartupValidationError, format_summary, validate_startup


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the portable BEE24 DINOv3 H1 experiment.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the server-local YAML config.")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate paths/device and print the summary without loading the model.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config)
        validated = validate_startup(config)
        print(format_summary(validated), flush=True)
        if args.validate_only:
            print("Validation completed; model inference was not started.")
            return 0
        metadata = run_h1(validated)
    except (ConfigurationError, StartupValidationError, ExperimentError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(f"H1 completed in {metadata['duration_seconds']:.1f}s")
    print(f"Results: {config.output.query_results}")
    print(f"Metadata: {config.output.metadata}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

