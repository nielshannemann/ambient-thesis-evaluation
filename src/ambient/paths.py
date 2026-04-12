"""Path helpers that preserve the current on-disk results contract."""

from pathlib import Path


RESULTS_ROOT = Path("results")

TASK4_RESULT_FILENAME = "layerwise_probe_results_with_vne.json"
TASK5_PLOT_FILENAME = "temporal_semantic_commitment_comparison.png"


def task0_run_dir(model_name: str, num_generations: int, model_family: str, diffusion_steps: int) -> Path:
    directory_name = f"{model_name}-n{num_generations}"
    if model_family == "llada":
        directory_name += f"-d{diffusion_steps}"
    return RESULTS_ROOT / directory_name


def task1_output_path(model_name: str, num_continuations: int) -> Path:
    return RESULTS_ROOT / "task1" / f"{model_name}_n{num_continuations}.json"


def task3_output_path(model_name: str, prompt_type: str) -> Path:
    return RESULTS_ROOT / "task3" / f"{model_name}_{prompt_type}.json"


def task4_output_path() -> Path:
    return RESULTS_ROOT / "task4" / TASK4_RESULT_FILENAME


def task5_output_path(model_name: str) -> Path:
    return RESULTS_ROOT / "task5" / f"{model_name}.json"


def task5_plot_dir() -> Path:
    return RESULTS_ROOT / "task5"


def plots_root(results_dir: Path | None = None) -> Path:
    root = results_dir or RESULTS_ROOT
    return root / "plots"


def task2_metrics_path(model_root_dir: Path) -> Path:
    return model_root_dir / "task2_semantic_metrics.json"


def dataset_similarity_default_paths() -> dict[str, Path]:
    return {
        "json": RESULTS_ROOT / "task_disambiguation_similarity_summary.json",
        "csv": RESULTS_ROOT / "task_disambiguation_similarity_instances.csv",
        "agg_csv": RESULTS_ROOT / "task_disambiguation_similarity_aggregates.csv",
    }


def continuation_length_output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "summary": output_dir / "continuation_length_summary.csv",
        "details": output_dir / "continuation_length_details.csv",
        "mismatches": output_dir / "continuation_token_mismatches.csv",
    }
