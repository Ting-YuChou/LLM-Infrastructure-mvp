"""
Run one training stage of the pipeline as a torchrun job.

    torchrun --standalone --nproc_per_node 8 -m training.stage_runner \
        --pipeline-config config/training_pipeline.yaml --stage sft

TrainingPipeline launches this for each GPU stage; rank 0 writes the stage
result to <stage output>/stage_result.json for the launcher to read back.
"""

from __future__ import annotations

import argparse

try:
    from .distributed import cleanup_distributed, is_main_process
    from .pipeline import GPU_STAGES, TrainingPipeline, write_stage_result
except ImportError:  # pragma: no cover - supports direct script execution
    from distributed import cleanup_distributed, is_main_process  # type: ignore[no-redef]
    from pipeline import GPU_STAGES, TrainingPipeline, write_stage_result  # type: ignore[no-redef]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-config", required=True, help="Pipeline YAML config")
    parser.add_argument("--stage", required=True, choices=GPU_STAGES)
    parser.add_argument("--output-dir", help="Override pipeline output root")
    parser.add_argument("--baseline-metrics", help="Baseline metrics JSON for final delta gates")
    parser.add_argument("--skip-promotion", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pipeline = TrainingPipeline(
        config_path=args.pipeline_config,
        stages=args.stage,
        output_dir=args.output_dir,
        baseline_metrics=args.baseline_metrics,
        skip_promotion=args.skip_promotion,
        launcher="inprocess",
    )
    # The trainers join the process group with their YAML timeout.
    result = pipeline.run_gpu_stage(args.stage)
    if is_main_process():
        write_stage_result(pipeline.stage_result_path(args.stage), result)
    cleanup_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
