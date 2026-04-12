import json
from pathlib import Path

from ambient.paths import (
    TASK4_RESULT_FILENAME,
    TASK5_PLOT_FILENAME,
    task0_run_dir,
    task1_output_path,
    task4_output_path,
    task5_output_path,
    task5_plot_dir,
)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def test_core_paths_match_existing_results() -> None:
    assert task0_run_dir("llama8b", 100, "llama", 64) == Path("results/llama8b-n100")
    assert task0_run_dir("llada8b", 10, "llada", 64) == Path("results/llada8b-n10-d64")
    assert task1_output_path("llama8b", 1) == Path("results/task1/llama8b_n1.json")
    assert task4_output_path() == Path(f"results/task4/{TASK4_RESULT_FILENAME}")
    assert task5_output_path("llada") == Path("results/task5/llada.json")

    assert task0_run_dir("llama8b", 100, "llama", 64).exists()
    assert task0_run_dir("llada8b", 10, "llada", 64).exists()
    assert task1_output_path("llama8b", 1).exists()
    assert task4_output_path().exists()
    assert task5_output_path("llada").exists()
    assert (task5_plot_dir() / TASK5_PLOT_FILENAME).exists()


def test_existing_task_artifacts_keep_expected_json_contracts() -> None:
    task1 = load_json(Path("results/task1/llama8b_n1.json"))
    assert {"metadata", "results"} <= set(task1.keys())
    assert {"task", "model_type", "model_id", "hyperparameters"} <= set(task1["metadata"].keys())
    assert {"id", "generated_raw", "generated_clean"} <= set(task1["results"][0].keys())

    task3 = load_json(Path("results/task3/llama8b_without_ambiguous.json"))
    assert {"metadata", "results"} <= set(task3.keys())
    assert {"task", "model_name", "model_type", "prompt_type", "hyperparameters"} <= set(task3["metadata"].keys())
    assert {"id", "prompt_text", "gold_disambiguations", "continuations"} <= set(task3["results"][0].keys())

    task4 = load_json(task4_output_path())
    assert {"config", "datasets", "results", "von_neumann_entropy"} <= set(task4.keys())
    assert {"llama_model", "llada_model", "data_path"} <= set(task4["config"].keys())

    task5 = load_json(task5_output_path("llada"))
    assert {"metadata", "results"} <= set(task5.keys())
    assert {"task", "model_name", "model_type", "model_id"} <= set(task5["metadata"].keys())
    first_instance = next(iter(task5["results"].values()))
    assert {"id", "prompt", "trajectory"} <= set(first_instance.keys())
