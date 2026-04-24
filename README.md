# AMBIENT-style ambiguity evaluation artifact

Anonymous artifact repository for the AMBIENT-style ambiguity experiments in
the accompanying EMNLP submission. The repository contains the experiment CLI,
evaluation code, tests, and selected generated outputs used by the paper.

## CLI

The canonical interface is the `ambient` CLI. The refactor keeps the current
`results/` contract intact: existing directory names, file names, and JSON
payloads stay unchanged unless you explicitly choose a different output path.

Core commands:

```bash
ambient task0 run --model-family llada --model-name llada8b
ambient task0 metrics results/llada8b-n10-d64/summary_mc128.jsonl

ambient task1 generate --model-family llama --model-name llama8b
ambient task1 judge --llama-file results/task1/llama8b_n1.json --llada-file results/task1/llada8b_n1.json

ambient task2 evaluate --model-dirs results/llama8b-n100 results/llada8b-n10-d64

ambient task3 generate --model-family llada --model-name llada8b --prompt-type ambiguous
ambient task3 evaluate --results-path results/task3/llada8b_ambiguous.json

ambient task4 evaluate

ambient task5 generate --model-family llama --model-name llama
ambient task5 metrics --llama-file results/task5/llama.json --llada-file results/task5/llada.json

ambient plots sweep-overview
ambient plots task4 --input results/task4/layerwise_probe_results_with_vne.json --output-dir results/task4
ambient plots task5 --llama-file results/task5/llama.json --llada-file results/task5/llada.json
```

Additional utilities:

```bash
ambient dataset bake-distractors
ambient dataset disambiguation-similarity
ambient diagnostics continuation-lengths --roots results/llama8b-n100 results/llada8b-n10-d64
```

## Packaging

The package uses a `src/` layout and exposes one console entrypoint:

```bash
ambient --help
```

If you are running directly from the repository without installation, set
`PYTHONPATH=src` before calling `python -m ambient.cli`.
