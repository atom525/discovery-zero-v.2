#!/usr/bin/env python3
"""
Run a production benchmark suite for Discovery Zero.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a Discovery Zero benchmark suite.")
    parser.add_argument("--suite", required=True, help="Path to benchmark suite JSON config")
    parser.add_argument("--repeats", type=int, default=None, help="Optional override for repeat count")
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional evaluation root directory (defaults to Zero/evaluation)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root / "src"))

    from discovery_zero.benchmark import run_suite

    result = run_suite(
        Path(args.suite).resolve(),
        repeats_override=args.repeats,
        output_root=Path(args.output_root).resolve() if args.output_root else None,
    )
    print(f"Suite run directory: {result.suite_run_dir}")
    print(f"Suite summary: {result.suite_summary_path}")
    print(f"Suite scorecard: {result.suite_scorecard_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
