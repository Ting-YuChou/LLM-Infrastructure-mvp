#!/usr/bin/env python3
"""CLI entrypoint for the aggregated training pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from training.pipeline import TrainingPipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SFT -> reward -> RLHF training pipeline")
    parser.add_argument("--config", required=True, help="Pipeline YAML config")
    parser.add_argument(
        "--stages",
        default="all",
        help="all or comma-separated stages: sft,reward,rlhf,eval,promotion,manifest,awq,handoff",
    )
    parser.add_argument("--output-dir", help="Override pipeline output root")
    parser.add_argument("--baseline-metrics", help="Baseline metrics JSON for final delta gates")
    parser.add_argument(
        "--skip-promotion",
        action="store_true",
        help="Register final aligned model as pending without setting champion",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse deterministic final checkpoints when they already exist",
    )
    parser.add_argument(
        "--launcher",
        choices=["auto", "torchrun", "inprocess"],
        help="How to run sft/reward/rlhf: one torchrun job per stage, or in this process "
        "(default: distributed.launcher in the pipeline YAML)",
    )
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        help="Processes (GPUs) per training stage (default: every visible GPU)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse config and print the deterministic execution plan without training",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pipeline = TrainingPipeline(
        config_path=args.config,
        stages=args.stages,
        output_dir=args.output_dir,
        baseline_metrics=args.baseline_metrics,
        skip_promotion=args.skip_promotion,
        resume=args.resume,
        dry_run=args.dry_run,
        launcher=args.launcher,
        nproc_per_node=args.nproc_per_node,
    )
    result = pipeline.run()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
