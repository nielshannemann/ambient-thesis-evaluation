from pathlib import Path

import pytest

from ambient.cli import build_parser, main
from ambient.paths import plots_root, task4_output_path, task5_plot_dir


def test_help_entrypoints_exit_cleanly() -> None:
    commands = [
        [],
        ["task0", "run"],
        ["task0", "metrics"],
        ["task1", "generate"],
        ["task1", "judge"],
        ["task2", "evaluate"],
        ["task3", "generate"],
        ["task3", "evaluate"],
        ["task4", "evaluate"],
        ["task5", "generate"],
        ["task5", "metrics"],
        ["plots", "sweep-overview"],
        ["plots", "task4"],
        ["plots", "task5"],
        ["dataset", "bake-distractors"],
        ["dataset", "disambiguation-similarity"],
        ["diagnostics", "continuation-lengths"],
    ]

    for command in commands:
        with pytest.raises(SystemExit) as excinfo:
            main([*command, "--help"])
        assert excinfo.value.code == 0


def test_task0_run_parser_uses_new_flag_names() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "task0",
            "run",
            "--model-family",
            "llada",
            "--model-name",
            "llada8b",
        ]
    )

    assert args.model_family == "llada"
    assert args.batch_size == 25
    assert args.output_dir is None
    assert callable(args.handler)


def test_task5_and_plot_commands_use_llama_llada_file_flags() -> None:
    parser = build_parser()

    metrics_args = parser.parse_args(
        [
            "task5",
            "metrics",
            "--llama-file",
            "results/task5/llama.json",
            "--llada-file",
            "results/task5/llada.json",
        ]
    )
    assert metrics_args.llama_file == Path("results/task5/llama.json")
    assert metrics_args.llada_file == Path("results/task5/llada.json")
    assert metrics_args.output_path is None

    plot_args = parser.parse_args(["plots", "task5"])
    assert plot_args.llama_file is None
    assert plot_args.llada_file is None
    assert plot_args.output_dir == task5_plot_dir()


def test_task4_and_plot_defaults_match_contract_paths() -> None:
    parser = build_parser()

    task4_args = parser.parse_args(["task4", "evaluate"])
    assert task4_args.output_path == task4_output_path()

    sweep_args = parser.parse_args(["plots", "sweep-overview"])
    assert sweep_args.results_dir == Path("results")
    assert sweep_args.output_dir is None
    assert plots_root(sweep_args.results_dir) == Path("results/plots")
