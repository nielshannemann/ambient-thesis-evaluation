# Evaluating Linguistic Ambiguity in Autoregressive and Masked Diffusion Language Models

This repository contains the evaluation code and selected result artifacts for
the master's thesis *Evaluating Linguistic Ambiguity in Autoregressive and
Masked Diffusion Language Models* by Niels Hannemann (University of Hamburg,
2026).

The primary comparison evaluates `meta-llama/Meta-Llama-3.1-8B` and
`GSAI-ML/LLaDA-8B-Base` on the AMBIENT benchmark. It distinguishes three
questions: whether a model ranks annotated readings ahead of a distractor,
which readings appear in free continuations, and how the relative scores of
two readings change as more context becomes available. Generation-quality and
probing experiments put those comparisons into perspective. A supplementary
experiment tests whether instruction-tuned checkpoints can explain the
ambiguity explicitly.

No language-model checkpoint is trained or fine-tuned in this repository. The
probing experiment trains lightweight linear classifiers on frozen hidden
states.

## Thesis map

The command-line interface retains stable `task1` to `task6` names from the
original implementation. The thesis uses descriptive experiment names:

| Thesis component | CLI entry point | Role |
| --- | --- | --- |
| Ranking Readings Against a Distractor | `ambient task1` | Generate from each reading and distractor, rescore the continuations, and aggregate ranking accuracy. |
| Generation Quality and Ranking Reliability | `ambient task2` | Measure artifact rate, reference-model perplexity, embedding dispersion, and lexical overlap for the ranking continuations. |
| Reading Coverage in Free Continuations | `ambient task3` | Generate without a reading-specific prompt and compare the resulting pool with the annotated readings. |
| Supporting Probing Experiments | `ambient task4` | Run grouped linear probes and token-state spectral analyses. |
| Reading-Score Profiles | `ambient task5` | Construct autoregressive prefix and diffusion mask-ratio profiles for two fixed readings. |
| Prompted Ambiguity Explanation | `ambient task6` | Generate explicit explanations and evaluate them with an LLM judge. |

Code comments refer to these section titles rather than page numbers so that
the references remain stable across rendered versions of the thesis. The
thesis appendix section **Implementation and Artifact Reference** provides the
corresponding methodological and implementation details.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/ambient/` | Package, CLI, model adapters, generation, evaluation, and plotting code. |
| `data/` | Processed AMBIENT input and fixed split files used by the experiments. |
| `results/` | Selected saved generations, scores, aggregate tables, and figures. |
| `tests/` | Tests for deterministic preprocessing, output contracts, metrics, controls, and CLI routing. |
| `scripts/` | Auxiliary figure and manual-audit scripts. |
| `docs/artifact_guide.md` | Result map, terminology, and reproduction levels. |

## Environment

The recorded experiment environment used Python 3.12.3, PyTorch 2.7.1 with
CUDA 11.8, and Transformers 4.57.1. `requirements.txt` is a complete snapshot
of that workstation environment, including development tools and CUDA-specific
packages; it is intentionally more extensive than a minimal runtime dependency
list.

For the closest environment reconstruction on a compatible Linux/CUDA system:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m pytest -q
ambient --help
```

The editable install uses `--no-deps` because the exact dependencies are kept
in `requirements.txt`. To run directly from a checkout instead, set
`PYTHONPATH=src` and invoke `python -m ambient.cli`.

Model downloads require access to the relevant Hugging Face repositories.
LLaMA access may additionally require accepting Meta's license terms. The
masked-diffusion checkpoints load model-specific code with
`trust_remote_code=True`; review upstream code and model cards before use.

## Reproduction levels

The artifact supports three different levels of verification:

1. **Code and contract checks.** Run `python -m pytest -q`. These tests do not
   reproduce the scientific results, but they check deterministic data
   handling, metric calculations, control construction, and CLI routing.
2. **Analysis from saved outputs.** Recompute aggregates, robustness checks,
   and figures from files already under `results/`. Some semantic analyses
   still download frozen embedding or NLI evaluators.
3. **Full model inference.** Regenerate continuations, reconstruction scores,
   hidden states, or LLM judgments. These commands require the original model
   weights and substantial GPU time.

Representative commands and the authoritative result paths are documented in
[`docs/artifact_guide.md`](docs/artifact_guide.md). Every CLI group also has
local help, for example:

```bash
ambient task1 --help
ambient task3 evaluate --help
ambient robustness --help
ambient plots --help
```

## Data and result provenance

`data/test_baked.jsonl` is the processed AMBIENT input consumed by the CLI. It
contains the benchmark records together with fixed distractors used by the
ranking experiment. See [`data/README.md`](data/README.md) and the original
[AMBIENT paper](https://aclanthology.org/2023.emnlp-main.51/) for provenance.

The serialized files under `results/` are the authoritative basis for the
reported aggregates. Run metadata records the model identifier, scientific
settings, random seed, software environment, and, for newer runs, input hashes.
Do not combine files from different run directories unless the corresponding
analysis explicitly supports that comparison.

Local exploratory runs, model caches, and temporary analysis outputs are
excluded through `.gitignore`; the tracked artifacts are the files referenced
by the thesis and the artifact guide.

## Models and compute

The primary thesis experiments use:

- [`meta-llama/Meta-Llama-3.1-8B`](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B)
- [`GSAI-ML/LLaDA-8B-Base`](https://huggingface.co/GSAI-ML/LLaDA-8B-Base)

Frozen auxiliary evaluators include `all-MiniLM-L6-v2`,
`all-mpnet-base-v2`, `roberta-large-mnli`,
`microsoft/deberta-large-mnli`, and
`EleutherAI/pythia-410m-deduped`; the LLaMA checkpoint also serves as one
autoregressive perplexity reference. The supplementary prompted-explanation
experiment uses instruction-tuned generator checkpoints recorded in the saved
generation files. Its judge configuration is documented in the thesis.

The pretrained language models are used without updating their parameters.
GPU-heavy runs were carried out on one NVIDIA RTX A6000. Runtime varies
substantially by experiment: the longest individual LLaDA runs took
approximately 36 GPU-hours, while analysis of saved outputs is considerably
cheaper.

## Licensing and generated text

Original code in this repository is released under the MIT License. Files
adapted from the official [LLaDA repository](https://github.com/ML-GSAI/LLaDA)
retain upstream attribution in their module docstrings. The AMBIENT-derived
data remains subject to the dataset's CC BY 4.0 license. Pretrained model
weights are not redistributed and remain subject to their providers' terms.

Some result files contain raw model continuations. These may include named
entities, web-like fragments, malformed text, or other unfiltered content.
No new personal data was collected, and the repository is not intended to
identify individuals.

## Citation

Please cite the thesis when using this evaluation design or its reported
results:

> Niels Hannemann. *Evaluating Linguistic Ambiguity in Autoregressive and
> Masked Diffusion Language Models*. Master's thesis, University of Hamburg,
> 2026.

Machine-readable software citation metadata is provided in `CITATION.cff`.
