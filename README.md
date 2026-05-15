# AmbiEnt-style ambiguity evaluation artifact

Anonymous artifact repository for the AmbiEnt-style ambiguity experiments in
the accompanying ARR/EMNLP submission. The repository contains the experiment
CLI, evaluation code, tests, and selected generated outputs used by the paper.

The paper refers to the main reported diagnostics as T1--T5. The repository
uses the same public task names and keeps the auxiliary explicit-disambiguation
experiment as T6:

| Paper diagnostic | Repository entrypoint | Notes |
| --- | --- | --- |
| T1: reading-specific continuation ranking | `ambient task1` | Core AmbiEnt ranking sweep. |
| T2: generation-quality controls | `ambient task2` | External-LM perplexity, embedding dispersion, overlap, artifact rate. |
| T3: reading coverage | `ambient task3` | Free-continuation embedding and NLI coverage metrics. |
| T4: activation-based controls | `ambient task4` | Linear probes and representation-dispersion controls. |
| T5: incremental commitment | `ambient task5` | Entropy-based commitment trajectories and controls. |
| T6: auxiliary explicit disambiguation | `ambient task6` | Side experiment; not part of the main T1--T5 paper narrative. |

## CLI

The canonical interface is the `ambient` CLI. The main Task-1 sweep directories
remain unchanged, while the auxiliary explicit-disambiguation outputs now live
under `results/task6/` to match the paper-facing task numbering.

Core commands:

```bash
# Paper T1: reading-specific continuation ranking
ambient task1 run --model-family llada --model-name llada8b
ambient task1 metrics results/llada8b-n10-d64/summary_mc128.jsonl

# Auxiliary Task 6: explicit disambiguation
ambient task6 generate --model-family llama --model-name llama8b
ambient task6 judge --llama-file results/task6/llama8b_n1.json --llada-file results/task6/llada8b_n1.json

# Paper T2--T5
ambient task2 evaluate --model-dirs results/llama8b-n100 results/llada8b-n10-d64

ambient task3 generate --model-family llada --model-name llada8b --prompt-type ambiguous
ambient task3 evaluate --results-path results/task3/llada8b_ambiguous.json

ambient task4 evaluate

ambient task5 generate --model-family llama --model-name llama
ambient task5 metrics --llama-file results/task5/llama.json --llada-file results/task5/llada.json

ambient plots sweep-overview --paper-graphics-dir paper/graphics
ambient plots task4 --input results/task4/layerwise_probe_results_with_vne.json --output-dir results/task4
ambient plots task5 --llama-file results/task5/llama.json --llada-file results/task5/llada.json --paper-graphics-dir paper/graphics
```

Additional utilities:

```bash
ambient dataset bake-distractors
ambient dataset disambiguation-similarity
ambient diagnostics continuation-lengths --roots results/llama8b-n100 results/llada8b-n10-d64
```

## Licenses and terms

This repository is intended for research reproducibility and anonymous review.

The original AmbiEnt dataset is licensed under CC BY 4.0 by its creators. This
artifact uses a benchmark-derived subset for offline evaluation and cites the
original dataset paper in the accompanying submission.

Pretrained model weights are not redistributed in this repository. Users must
obtain them from the original providers and follow their licenses and terms of
use. The main models and scorers used by the paper are:

- `meta-llama/Llama-3.1-8B`: Llama 3.1 Community License.
- `GSAI-ML/LLaDA-8B-Base`: MIT license according to the Hugging Face model card.
- `meta-llama/Meta-Llama-3.1-70B-Instruct` and `Qwen/Qwen2.5-72B-Instruct`:
  used as optional LLM-as-a-judge models for the auxiliary explicit
  disambiguation experiment.
- `all-MiniLM-L6-v2`, `all-mpnet-base-v2`, `roberta-large-mnli`,
  `microsoft/deberta-large-mnli`, and the Pythia external-LM scorer: loaded
  from their original package or Hugging Face sources and subject to the
  respective upstream licenses.

The original code in this anonymized artifact is provided for review and
research reproducibility. If this repository is made public after review, add a
formal project license before broad reuse.

## Data and privacy

The experiments use an existing benchmark-derived dataset and model-generated
continuations. No new personal data was collected, and the artifact is not
intended to identify individuals. Some generated continuations may contain named
entities or web-like snippets because they are raw model outputs. We did not
perform a dedicated PII/offensive-content audit beyond the heuristic artifact
filtering described in the paper and implemented in the evaluation code.

## Compute

No model training was performed; all reported experiments are inference-only.
The main GPU experiments were run on a single NVIDIA RTX A6000 GPU. Runtime
varied substantially by task and setting; the longest individual LLaDA runs took
up to approximately 36 GPU-hours, while most runs were shorter. This is an
approximate compute description rather than an exact total GPU-hour accounting.

## Packaging

The package uses a `src/` layout and exposes one console entrypoint:

```bash
ambient --help
```

If you are running directly from the repository without installation, set
`PYTHONPATH=src` before calling `python -m ambient.cli`.
