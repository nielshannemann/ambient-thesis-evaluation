"""Canonical command-line interface for the AmbiEnt-style experiment suite."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Callable

from ambient.constants import (
    DEFAULT_DATA_PATH,
    LLADA_BASE_MODEL_ID,
    LLADA_INSTRUCT_MODEL_ID,
    LLAMA_BASE_MODEL_ID,
    LLAMA_INSTRUCT_MODEL_ID,
    MODEL_FAMILY_CHOICES,
    TASK2_EMBED_MODEL_ID,
    TASK3_NLI_MODEL_ID,
    TASK6_JUDGE_MODEL_ID,
    TASK6_SECONDARY_JUDGE_MODEL_ID,
)
from ambient.paths import (
    dataset_similarity_default_paths,
    plots_root,
    task4_output_path,
    task5_plot_dir,
    task6_judge_output_path,
)


Handler = Callable[[argparse.Namespace], int | None]


def _lazy_handler(import_path: str) -> Handler:
    module_name, func_name = import_path.split(":")

    def _run(args: argparse.Namespace) -> int | None:
        module = importlib.import_module(module_name)
        handler = getattr(module, func_name)
        return handler(args)

    return _run


def _set_handler(parser: argparse.ArgumentParser, import_path: str) -> None:
    parser.set_defaults(handler=_lazy_handler(import_path))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ambient",
        description="Unified paper-ready CLI for the AmbiEnt-style experiment suite.",
    )
    top_level = parser.add_subparsers(dest="command_group", required=True)

    _add_task1_commands(top_level)
    _add_task2_commands(top_level)
    _add_task3_commands(top_level)
    _add_task4_commands(top_level)
    _add_task5_commands(top_level)
    _add_task6_commands(top_level)
    _add_plot_commands(top_level)
    _add_dataset_commands(top_level)
    _add_diagnostic_commands(top_level)
    _add_robustness_commands(top_level)

    return parser


def _add_task1_commands(top_level: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    task1 = top_level.add_parser("task1", help="Task 1: reading-specific continuation ranking.")
    subparsers = task1.add_subparsers(dest="task1_command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the Task-1 reading-ranking benchmark.")
    run_parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    run_parser.add_argument("--model-family", choices=MODEL_FAMILY_CHOICES, required=True)
    run_parser.add_argument("--model-name", type=str, required=True)
    run_parser.add_argument("--model-id", type=str, default=None)
    run_parser.add_argument("--num-generations", type=int, default=100)
    run_parser.add_argument("--batch-size", type=int, default=25)
    run_parser.add_argument("--seed", type=int, default=42)
    run_parser.add_argument("--diffusion-steps", type=int, default=64)
    run_parser.add_argument("--mc-num", type=str, default="128")
    run_parser.add_argument("--mc-batch-size", type=int, default=16)
    run_parser.add_argument("--cfg-scale", type=float, default=0.0)
    run_parser.add_argument("--top-p", type=float, default=1.0)
    run_parser.add_argument("--top-k", type=int, default=0)
    run_parser.add_argument("--temperature", type=float, default=1.0)
    run_parser.add_argument("--output-dir", type=Path, default=None)
    _set_handler(run_parser, "ambient.evaluation.run_ambient_experiments:run")

    metrics_parser = subparsers.add_parser("metrics", help="Aggregate Task-1 metrics for one summary file.")
    metrics_parser.add_argument("results_path", type=Path, help="Path to a Task-1 summary JSONL file.")
    metrics_parser.add_argument("--dedupe", choices=["instance", "row"], default="instance")
    _set_handler(metrics_parser, "ambient.evaluation.task1_compute_results_metrics:run")


def _add_task6_commands(top_level: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    task6 = top_level.add_parser("task6", help="Task 6: auxiliary explicit generative disambiguation.")
    subparsers = task6.add_subparsers(dest="task6_command", required=True)

    generate_parser = subparsers.add_parser("generate", help="Generate Task-6 disambiguations.")
    generate_parser.add_argument("--model-family", choices=MODEL_FAMILY_CHOICES, required=True)
    generate_parser.add_argument("--model-name", type=str, required=True)
    generate_parser.add_argument("--model-id", type=str, default=None)
    generate_parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    generate_parser.add_argument("--max-examples", type=int, default=580)
    generate_parser.add_argument("--num-continuations", type=int, default=1)
    generate_parser.add_argument("--batch-size", type=int, default=25)
    generate_parser.add_argument("--seed", type=int, default=42)
    generate_parser.add_argument("--temperature", type=float, default=1.0)
    generate_parser.add_argument("--top-p", type=float, default=1.0)
    generate_parser.add_argument("--top-k", type=int, default=0)
    generate_parser.add_argument("--cfg-scale", type=float, default=0.0)
    generate_parser.add_argument("--diffusion-steps", type=int, default=128)
    generate_parser.add_argument("--output-path", type=Path, default=None)
    _set_handler(generate_parser, "ambient.generation.task6_disambiguation:run")

    judge_parser = subparsers.add_parser("judge", help="Run the Task-6 LLM-as-a-judge evaluation.")
    judge_parser.add_argument("--llada-file", type=Path, default=Path("results/task6/llada8b_n100.json"))
    judge_parser.add_argument("--llama-file", type=Path, default=Path("results/task6/llama8b_n100.json"))
    judge_parser.add_argument("--judge-models", nargs="+", type=str, default=None)
    judge_parser.add_argument("--judge-model", type=str, default=None)
    judge_parser.add_argument("--seed", type=int, default=42)
    judge_parser.add_argument("--disable-4bit", action="store_true")
    judge_parser.add_argument(
        "--output-path",
        type=Path,
        default=task6_judge_output_path(),
    )
    judge_parser.set_defaults(
        default_judge_models=[TASK6_JUDGE_MODEL_ID, TASK6_SECONDARY_JUDGE_MODEL_ID]
    )
    _set_handler(judge_parser, "ambient.evaluation.task6_evaluation:run")


def _add_task2_commands(top_level: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    task2 = top_level.add_parser("task2", help="Task 2: quality and diversity evaluation.")
    subparsers = task2.add_subparsers(dest="task2_command", required=True)

    evaluate_parser = subparsers.add_parser("evaluate", help="Compute Task-2 metrics over result folders.")
    evaluate_parser.add_argument("--model-dirs", nargs="+", type=Path, required=True)
    evaluate_parser.add_argument("--ppl-model", type=str, default=LLAMA_BASE_MODEL_ID)
    evaluate_parser.add_argument("--ppl-batch-size", type=int, default=16)
    evaluate_parser.add_argument("--ppl-max-length", type=int, default=None)
    evaluate_parser.add_argument("--max-ppl-texts-per-dir", type=int, default=None)
    evaluate_parser.add_argument("--embed-model", type=str, default=TASK2_EMBED_MODEL_ID)
    evaluate_parser.add_argument("--skip-diversity", action="store_true")
    evaluate_parser.add_argument("--skip-ppl", action="store_true")
    evaluate_parser.add_argument("--output-suffix", type=str, default=None)
    evaluate_parser.add_argument("--summary-output", type=Path, default=None)
    evaluate_parser.add_argument("--use-4bit", action="store_true")
    evaluate_parser.add_argument("--seed", type=int, default=42)
    _set_handler(evaluate_parser, "ambient.evaluation.task2_semantic_diversity:run")


def _add_task3_commands(top_level: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    task3 = top_level.add_parser("task3", help="Task 3: generative semantic clustering.")
    subparsers = task3.add_subparsers(dest="task3_command", required=True)

    generate_parser = subparsers.add_parser("generate", help="Sample Task-3 continuations.")
    generate_parser.add_argument("--model-family", choices=MODEL_FAMILY_CHOICES, required=True)
    generate_parser.add_argument("--model-name", type=str, required=True)
    generate_parser.add_argument("--model-id", type=str, default=None)
    generate_parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    generate_parser.add_argument("--prompt-type", choices=["ambiguous", "disambiguated_control"], default="ambiguous")
    generate_parser.add_argument("--max-examples", type=int, default=580)
    generate_parser.add_argument("--num-continuations", type=int, default=10)
    generate_parser.add_argument("--batch-size", type=int, default=25)
    generate_parser.add_argument("--seed", type=int, default=42)
    generate_parser.add_argument("--temperature", type=float, default=1.0)
    generate_parser.add_argument("--top-p", type=float, default=1.0)
    generate_parser.add_argument("--top-k", type=int, default=0)
    generate_parser.add_argument("--cfg-scale", type=float, default=0.0)
    generate_parser.add_argument("--diffusion-steps", type=int, default=128)
    generate_parser.add_argument("--output-path", type=Path, default=None)
    _set_handler(generate_parser, "ambient.generation.task3_silhouette_generate:run")

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate Task-3 semantic clusters.")
    evaluate_parser.add_argument("--results-path", type=Path, required=True)
    evaluate_parser.add_argument("--embed-model", type=str, default=TASK2_EMBED_MODEL_ID)
    evaluate_parser.add_argument("--nli-model", type=str, default=TASK3_NLI_MODEL_ID)
    evaluate_parser.add_argument("--nli-thresholds", type=str, default="argmax")
    evaluate_parser.add_argument("--nli-batch-size", type=int, default=16)
    evaluate_parser.add_argument("--skip-nli", action="store_true")
    evaluate_parser.add_argument("--progress-every", type=int, default=25)
    evaluate_parser.add_argument("--output-path", type=Path, default=None)
    _set_handler(evaluate_parser, "ambient.evaluation.task3_silhouette_evaluate:run")


def _add_task4_commands(top_level: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    task4 = top_level.add_parser("task4", help="Task 4: probing and von Neumann entropy analysis.")
    subparsers = task4.add_subparsers(dest="task4_command", required=True)

    evaluate_parser = subparsers.add_parser("evaluate", help="Run Task-4 probing and entropy analysis.")
    evaluate_parser.add_argument("--llama-model-id", type=str, default=LLAMA_BASE_MODEL_ID)
    evaluate_parser.add_argument("--llada-model-id", type=str, default=LLADA_BASE_MODEL_ID)
    evaluate_parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    evaluate_parser.add_argument("--max-examples", type=int, default=580)
    evaluate_parser.add_argument("--batch-size", type=int, default=16)
    evaluate_parser.add_argument("--seed", type=int, default=42)
    evaluate_parser.add_argument("--use-4bit", action="store_true")
    evaluate_parser.add_argument("--include-embedding-layer", action="store_true")
    evaluate_parser.add_argument(
        "--dataset-modes",
        nargs="+",
        default=["side_reconstructed", "fully_disambiguated"],
        choices=["side_reconstructed", "fully_disambiguated"],
    )
    evaluate_parser.add_argument(
        "--probe-control-modes",
        nargs="+",
        default=[],
        choices=["unambiguous_length_matched"],
    )
    evaluate_parser.add_argument("--vne-input-mode", type=str, default="sentence_only", choices=["sentence_only", "pair_prompt"])
    evaluate_parser.add_argument(
        "--vne-control-conditions",
        nargs="+",
        default=[],
        choices=["distractor_rewrite", "random_matched_rewrite"],
    )
    evaluate_parser.add_argument("--vne-center-tokens", action="store_true")
    evaluate_parser.add_argument("--skip-vne", action="store_true")
    evaluate_parser.add_argument("--output-path", type=Path, default=task4_output_path())
    _set_handler(evaluate_parser, "ambient.evaluation.task4_linear_probing:run")


def _add_task5_commands(top_level: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    task5 = top_level.add_parser("task5", help="Task 5: temporal semantic commitment.")
    subparsers = task5.add_subparsers(dest="task5_command", required=True)

    generate_parser = subparsers.add_parser("generate", help="Generate Task-5 commitment trajectories.")
    generate_parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    generate_parser.add_argument("--model-family", choices=MODEL_FAMILY_CHOICES, required=True)
    generate_parser.add_argument("--model-name", type=str, required=True)
    generate_parser.add_argument("--model-id", type=str, default=None)
    generate_parser.add_argument("--seed", type=int, default=42)
    generate_parser.add_argument("--max-examples", type=int, default=580)
    generate_parser.add_argument("--max-steps", type=int, default=20)
    generate_parser.add_argument("--mc-num", type=int, default=8)
    generate_parser.add_argument("--cfg-scale", type=float, default=0.0)
    generate_parser.add_argument(
        "--condition",
        type=str,
        default="gold_disambiguation",
        choices=["gold_disambiguation", "distractor_rewrite", "random_matched_rewrite"],
    )
    generate_parser.add_argument("--output-path", type=Path, default=None)
    _set_handler(generate_parser, "ambient.generation.task5_superposition_decay:run")

    metrics_parser = subparsers.add_parser("metrics", help="Aggregate Task-5 scalar metrics.")
    metrics_parser.add_argument("--llama-file", type=Path, required=True)
    metrics_parser.add_argument("--llada-file", type=Path, required=True)
    metrics_parser.add_argument("--bootstrap-reps", type=int, default=5000)
    metrics_parser.add_argument("--ci-level", type=float, default=95.0)
    metrics_parser.add_argument("--seed", type=int, default=42)
    metrics_parser.add_argument("--output-path", type=Path, default=None)
    _set_handler(metrics_parser, "ambient.evaluation.task5_compute_decay_metrics:run")


def _add_plot_commands(top_level: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    plots = top_level.add_parser("plots", help="Plot existing experiment outputs without recomputing them.")
    subparsers = plots.add_subparsers(dest="plots_command", required=True)

    sweep_parser = subparsers.add_parser("sweep-overview", help="Plot Task-1/2 result sweeps.")
    sweep_parser.add_argument("--results-dir", type=Path, default=Path("results"))
    sweep_parser.add_argument("--output-dir", type=Path, default=None)
    sweep_parser.add_argument("--llada-pattern", type=str, default=r"llada8b-n10-d(\d+)")
    sweep_parser.add_argument("--llama-dir", type=str, default="llama8b-n100")
    sweep_parser.add_argument("--llama-extra-dirs", nargs="*", type=str, default=[])
    sweep_parser.add_argument("--paper-mc", type=int, default=256)
    sweep_parser.add_argument("--skip-paper-figures", action="store_true")
    sweep_parser.add_argument("--paper-graphics-dir", type=Path, default=None)
    _set_handler(sweep_parser, "ambient.visualization.task1_plot_results:run")

    task4_parser = subparsers.add_parser("task4", help="Plot Task-4 probing and entropy outputs.")
    task4_parser.add_argument("--input", type=Path, required=True)
    task4_parser.add_argument("--output-dir", type=Path, required=True)
    task4_parser.add_argument("--no-error-band", action="store_true")
    _set_handler(task4_parser, "ambient.visualization.task4_layerwise:run")

    task5_parser = subparsers.add_parser("task5", help="Plot Task-5 trajectory summaries.")
    task5_parser.add_argument("--llama-file", type=Path, default=None)
    task5_parser.add_argument("--llada-file", type=Path, default=None)
    task5_parser.add_argument("--llama-distractor-file", type=Path, default=None)
    task5_parser.add_argument("--llama-random-file", type=Path, default=None)
    task5_parser.add_argument("--llada-distractor-file", type=Path, default=None)
    task5_parser.add_argument("--llada-random-file", type=Path, default=None)
    task5_parser.add_argument("--band", choices=["std", "sem", "none"], default="std")
    task5_parser.add_argument("--num-points", type=int, default=101)
    task5_parser.add_argument("--output-dir", type=Path, default=task5_plot_dir())
    task5_parser.add_argument("--paper-graphics-dir", type=Path, default=None)
    _set_handler(task5_parser, "ambient.visualization.task5_plot_decay:run")


def _add_dataset_commands(top_level: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    dataset = top_level.add_parser("dataset", help="Dataset preparation and analysis utilities.")
    subparsers = dataset.add_subparsers(dest="dataset_command", required=True)

    bake_parser = subparsers.add_parser("bake-distractors", help="Bake distractors into the AMBIENT dataset.")
    bake_parser.add_argument("--data-path", type=Path, default=Path("external/ambient/AmbiEnt/test.jsonl"))
    bake_parser.add_argument("--output-path", type=Path, default=Path("external/ambient/AmbiEnt/test_baked.jsonl"))
    _set_handler(bake_parser, "ambient.bake_distractors:run")

    defaults = dataset_similarity_default_paths()
    similarity_parser = subparsers.add_parser("disambiguation-similarity", help="Analyze the gold disambiguation geometry.")
    similarity_parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    similarity_parser.add_argument("--embed-model", type=str, default=TASK2_EMBED_MODEL_ID)
    similarity_parser.add_argument("--batch-size", type=int, default=128)
    similarity_parser.add_argument("--max-examples", type=int, default=None)
    similarity_parser.add_argument("--seed", type=int, default=42)
    similarity_parser.add_argument("--output-json", type=Path, default=defaults["json"])
    similarity_parser.add_argument("--output-csv", type=Path, default=defaults["csv"])
    similarity_parser.add_argument("--output-agg-csv", type=Path, default=defaults["agg_csv"])
    _set_handler(similarity_parser, "ambient.evaluation.task_dataset_disambiguation_similarity:run")


def _add_diagnostic_commands(top_level: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    diagnostics = top_level.add_parser("diagnostics", help="Read-only diagnostics over existing outputs.")
    subparsers = diagnostics.add_subparsers(dest="diagnostics_command", required=True)

    lengths_parser = subparsers.add_parser("continuation-lengths", help="Analyze continuation length statistics.")
    lengths_parser.add_argument("--roots", nargs="+", type=Path, required=True)
    lengths_parser.add_argument("--output-dir", type=Path, default=Path("."))
    _set_handler(lengths_parser, "ambient.evaluate_example_dirs:run")


def _add_robustness_commands(top_level: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    robustness = top_level.add_parser("robustness", help="Read-only robustness analyses over existing outputs.")
    subparsers = robustness.add_subparsers(dest="robustness_command", required=True)

    task1_parser = subparsers.add_parser(
        "task1-ranking",
        help="Task-1 ranking robustness: bootstrap CIs, K sensitivity, matched-count subsampling, and filter sensitivity.",
    )
    task1_parser.add_argument("--run-dirs", nargs="+", type=Path, required=True)
    task1_parser.add_argument("--summary-glob", type=str, default="summary*.jsonl")
    task1_parser.add_argument("--dedupe", choices=["instance", "row"], default="instance")
    task1_parser.add_argument(
        "--metric-key",
        type=str,
        default="normalized_cleaned",
        help="Metric key or alias: normalized_cleaned, normalized_unfiltered, unnormalized_cleaned, unnormalized_unfiltered.",
    )
    task1_parser.add_argument("--bootstrap-reps", type=int, default=5000)
    task1_parser.add_argument("--ci-level", type=float, default=95.0)
    task1_parser.add_argument("--seed", type=int, default=42)
    task1_parser.add_argument("--example-dir-name", type=str, default="example_dirs")
    task1_parser.add_argument("--subsample-n", type=int, default=None)
    task1_parser.add_argument("--subsample-reps", type=int, default=1000)
    task1_parser.add_argument("--subsample-with-replacement", action="store_true")
    task1_parser.add_argument("--filter-non-alnum-ratios", type=str, default="0.25,0.35,0.45")
    task1_parser.add_argument("--filter-repeat-thresholds", type=str, default="8,12,20")
    task1_parser.add_argument("--write-replicates", action="store_true")
    task1_parser.add_argument("--output-dir", type=Path, default=Path("results/robustness/task1"))
    _set_handler(task1_parser, "ambient.evaluation.task1_ranking_robustness:run")


def main(argv: list[str] | None = None) -> int | None:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.error("No command selected.")
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
