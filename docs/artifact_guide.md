# Thesis artifact guide

This guide maps the final thesis terminology to the repository and separates
checks that reuse saved outputs from commands that invoke language models. The
methodological definitions remain in the thesis; this document focuses on
where the corresponding implementation and artifacts are located.

## Scope

The primary thesis comparison uses Meta-Llama-3.1-8B and LLaDA-8B-Base on
AMBIENT. The tracked results contain the primary experiments, their controls,
and the robustness analyses referenced by the thesis.

The CLI task numbers are stable implementation identifiers. They should not be
interpreted as an ordering of evidential importance.

## Component map

| Thesis section | CLI | Main implementation |
| --- | --- | --- |
| Ranking Readings Against a Distractor | `task1 run`, `task1 metrics` | `evaluation/run_ambient_experiments.py`, `evaluation/continuation_evaluation_adapted.py`, `evaluation/task1_compute_results_metrics.py` |
| Generation Quality and Ranking Reliability | `task2 evaluate` | `evaluation/task2_semantic_diversity.py` |
| Reading Coverage in Free Continuations | `task3 generate`, `task3 evaluate` | `generation/task3_silhouette_generate.py`, `evaluation/task3_silhouette_evaluate.py` |
| Supporting Probing Experiments | `task4 evaluate`, `plots task4` | `evaluation/task4_linear_probing.py`, `visualization/task4_layerwise.py` |
| Reading-Score Profiles | `task5 generate`, `task5 metrics`, `plots task5` | `generation/task5_temporal_semantic_commitment.py`, `evaluation/task5_compute_decay_metrics.py`, `visualization/task5_plot_decay.py` |
| Prompted Ambiguity Explanation | `task6 generate`, `task6 judge` | `generation/task6_disambiguation.py`, `evaluation/task6_evaluation.py` |

Model loading is centralized in `modeling.py`. `adapters.py` exposes the common
generation and sequence-scoring interface, while `llada_loader.py` and
`evaluation/get_log_likelihood.py` contain the LLaDA-specific generation and
reconstruction-scoring adaptations.

## Authoritative thesis artifacts

| Component | Saved artifact location |
| --- | --- |
| Reading ranking | `results/llama8b-n100/`, `results/llada8b-n10-d*/`, `results/robustness/task1/` |
| Generation quality | `task2_semantic_metrics*.json` inside the ranking run directories and `results/robustness/task2_external_ppl/` |
| Free-continuation coverage | `results/task3/no_trailing_quotes/` and `results/robustness/task3_*` |
| Probing experiments | `results/task4/layerwise_probe_results_with_vne_controls.json` |
| Reading-score profiles | `results/task5/llama.json`, `results/task5/llada.json`, and `results/task5/*_metrics.json` |
| Prompted explanation | `results/task6/llama8b_n1.json`, `results/task6/llada8b_n1.json`, and the archived judge aggregate described in the thesis |

The serialized result files, rather than regenerated figures, are the
authoritative inputs to the reported aggregates. Newer run directories also
contain `run_meta.json` files recording model IDs, parameters, software and GPU
information, seeds, and input hashes.

## Verification from saved outputs

The following examples avoid regenerating continuations. They still require
the Python dependencies, and semantic evaluation may download frozen evaluator
models.

```bash
mkdir -p tmp/verification

# Recompute reading-ranking aggregates.
ambient task1 metrics results/llama8b-n100/summary.jsonl
ambient task1 metrics results/llada8b-n10-d64/summary_mc256.jsonl

# Re-evaluate one saved free-continuation file.
ambient task3 evaluate \
  --results-path results/task3/no_trailing_quotes/llada8b_ambiguous.json \
  --output-path tmp/verification/llada8b_ambiguous_evaluation.json

# Recompute reading-score profile summaries.
ambient task5 metrics \
  --llama-file results/task5/llama.json \
  --llada-file results/task5/llada.json \
  --output-path tmp/verification/reading_score_profile_metrics.json

# Recreate the probing figure from saved hidden-state analyses.
ambient plots task4 \
  --input results/task4/layerwise_probe_results_with_vne_controls.json \
  --output-dir tmp/verification/task4_plots
```

The example paths deliberately keep regenerated files outside the tracked
artifact directories.
The `robustness`, `dataset`, `diagnostics`, and `plots` command groups expose
additional saved-output analyses.

## Full inference

Generation and scoring commands under `task1 run`, `task3 generate`, `task4
evaluate`, `task5 generate`, and `task6 generate` load one or more model
checkpoints. `task6 judge` additionally loads an instruction-tuned judge. Full
commands and scientific settings are given in the thesis appendix section
**Representative Reproduction Commands** and are also discoverable through
the relevant `--help` pages.

The primary LLaDA ranking sweep is particularly expensive because every saved
continuation is rescored under multiple Monte Carlo budgets. Preserve each run
directory's metadata and do not resume a directory with different scientific
settings. The newer task runners reject incompatible resume metadata where
that contract is available.

## Naming and compatibility

Several serialized keys and command flags predate the final thesis wording.
They remain unchanged to preserve compatibility with saved outputs and scripts:

- `side_reconstructed` corresponds to the thesis's *single-side
  disambiguated* probing condition.
- `fully_disambiguated` corresponds to the *complete-pair disambiguated*
  condition.
- `empirical_KL_div*` fields store the historical reading-ranking score names;
  the thesis defines the reported quantity as a loss-gap statistic rather than
  exact KL divergence.
- `--paper-graphics-dir`, `--paper-mc`, and related plotting names are retained
  as public CLI options even though the repository now accompanies a thesis.
- Modules containing `silhouette` or `temporal_semantic_commitment` in their
  filenames are wrapped by the clearer Task-3 and Task-5 entry-point modules.

## Reproducibility boundary

Seeds and deterministic preprocessing make repeated runs comparable within a
fixed environment. Exact floating-point outputs can still depend on hardware,
CUDA kernels, remote model code, and library versions. For this reason, the
tracked results remain the reference for the thesis tables and figures.

After the release commit is frozen, record its full Git SHA alongside the
thesis artifact reference. Do not write a final SHA into documentation before
the release commit exists, because doing so changes the commit being recorded.
