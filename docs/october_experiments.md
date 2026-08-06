# October revision experiment protocol

This runbook extends the paper experiments without changing the thesis snapshot.
All extension outputs go below `results/october_revision/`, which is ignored by
Git. Historical model-family names (`llama`, `llada`) remain valid. The new
generic backends are `ar` for Hugging Face causal LMs and `dream` for Dream's
official masked-diffusion API.

## Current status (2026-08-04)

No confirmatory October experiment has been completed yet. The work so far
covers implementation, smoke tests, and Task-3 continuation calibration only;
calibration outputs are excluded from confirmatory analyses.

| Work package | Status | Completed evidence | Next gate |
| --- | --- | --- | --- |
| P0: smoke tests | Complete | Historical LLaMA, generic Qwen, Dream scoring/generation, and Task-3 resume paths ran successfully | Keep the tested code revision with all outputs |
| P1: human validation | Prepared, annotation pending | Protocol 1.1 and the 32-row disjoint pilot package are prepared | Two annotators complete the pilot; then freeze the guidance and regenerate the untouched main package |
| P2: second-pair Tasks 1 and 2 | Complete | Equal-count Qwen/Dream T1 and both generation-quality oracles are complete | Preserve outputs and proceed to P3 |
| P3: four-model Task 3 | Generation complete; semantic evaluation pending | All four 15,000-slot artifacts pass structural and quality audits | Run the four-model primary semantic evaluation |
| P4: second dataset | Ready, not started | Experiment-2B loader and scoring implementation are available | Obtain the official repository and run the four specified checkpoints |
| P5: PLL triangulation | Ready, not started | Rescoring and scorer-comparison commands are implemented | Reuse the complete historical `example_dirs` on the workstation |
| P6: matched Task-1 budget | Ready, not started | Run and analysis commands are implemented | Run the primary `T=64`, `N=100` condition |
| P7: second-pair Task 5 | Optional, not started | Commands are available | Reconsider only after reviewing P1--P6 |

The work packages use the following checkpoint sets. In particular, "four
models" does not denote the same set in P3 and P4.

| Checkpoint | P2: T1/T2 | P3: Task 3 | P4: Scope 2B | P5: PLL | P6: matched budget | P7: Task 5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Meta-Llama-3.1-8B |  | Yes | Yes |  | Historical reference |  |
| LLaDA-8B-Base |  | Yes | Yes | Primary | Primary |  |
| Qwen2.5-7B | Yes |  | Yes |  |  | Yes |
| Dream-v0-Base-7B | Yes | Yes | Yes | Optional follow-up |  | Yes |
| Mistral-7B-v0.3 |  | Yes |  |  |  |  |

## 1. What each work package addresses

| Priority | Work package | Main reviewer concern | Required for revision |
| --- | --- | --- | --- |
| P0 | Smoke tests and frozen protocol | Reproducibility | Yes |
| P1 | Blinded human validation of Task 3 | Automatic proxies and no human validation | Yes |
| P2 | Qwen2.5-7B versus Dream-v0-Base-7B on Tasks 1 and 2 | One model per family | Yes |
| P3 | Fresh four-base-model Task-3 comparison on a held-out subset | One model per family and judge-dependent coverage | Yes |
| P4 | Scope Ambiguities Experiment 2B on all four models | One dataset | Yes |
| P5 | Single-token PLL triangulation of Task 1 | Reconstruction-proxy validity | Yes |
| P6 | Full matched continuation budget for LLaDA | 100 AR versus 10 diffusion continuations | Yes, start with T=64 |
| P7 | Qwen/Dream Task 5 and qualitative failure cases | Architecture mechanism and interpretability | Optional |

The primary revision should not present all outputs as independent evidence of
one architecture-wide claim. P2 to P4 are replications. P5 is scorer
triangulation. P6 isolates one budget confound. P7 remains an optional scoped
diagnostic rather than a direct trace of internal denoising dynamics.

## 2. Branch and environment

Run all commands from the repository root on the workstation.

Publish the branch once from the development machine:

```bash
git push -u origin paper-october-experiments
```

Then obtain it on the workstation:

```bash
git fetch origin
git switch --track origin/paper-october-experiments
```

If the local workstation branch already exists, use
`git switch paper-october-experiments` followed by `git pull --ff-only`.

```bash
git switch paper-october-experiments
git status --short --branch
git rev-parse HEAD
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python src/ambient/cli.py --help
mkdir -p results/october_revision
git rev-parse HEAD > results/october_revision/CODE_REVISION.txt
```

The commands below use the workstation convention requested for this project:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python src/ambient/cli.py ...
```

The new metadata records Python, PyTorch, Transformers, CUDA, and the visible
GPU. Task-1 metadata additionally records the dataset hash, continuation
length/termination policy, precision choice, and decoding controls; resuming
with incompatible scientific settings is rejected. Keep the
`CODE_REVISION.txt` file with the outputs. Do not mix outputs from different
commits in one run directory.

Dream's upstream implementation uses `trust_remote_code=True` and its official
`diffusion_generate` method. Try the existing project environment first. The
upstream Dream release reported compatibility with PyTorch 2.5.1 and
Transformers 4.46.2, while this repository currently pins newer versions. If
the Dream smoke tests fail because of a remote-code API incompatibility, create
a separate Dream environment instead of downgrading the established LLaMA and
LLaDA environment. Record that environment as a separate protocol decision.
The Mistral tokenizer additionally requires the pinned `protobuf` package from
`requirements.txt`; install it into older workstation environments before
loading `mistralai/Mistral-7B-v0.3`.

## 3. Mandatory smoke tests

These tests intentionally write to disposable extension directories. They do
not modify the historical result folders.

### 3.1 Historical LLaMA path

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python src/ambient/cli.py task1 run \
  --model-family llama \
  --model-name llama8b-smoke \
  --model-id meta-llama/Meta-Llama-3.1-8B \
  --max-examples 1 \
  --num-generations 2 \
  --batch-size 2 \
  --mc-batch-size 2 \
  --seed 42 \
  --no-use-4bit \
  --output-dir results/october_revision/smoke/llama-task1
```

### 3.2 Generic Qwen AR path

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python src/ambient/cli.py task1 run \
  --model-family ar \
  --model-name qwen25-7b-smoke \
  --model-id Qwen/Qwen2.5-7B \
  --max-examples 1 \
  --num-generations 2 \
  --batch-size 2 \
  --mc-batch-size 2 \
  --seed 42 \
  --no-use-4bit \
  --output-dir results/october_revision/smoke/qwen-task1
```

### 3.3 Dream generation and reconstruction scoring

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python src/ambient/cli.py task1 run \
  --model-family dream \
  --model-name dream7b-smoke \
  --model-id Dream-org/Dream-v0-Base-7B \
  --max-examples 1 \
  --num-generations 2 \
  --batch-size 1 \
  --diffusion-steps 8 \
  --diffusion-alg entropy \
  --diffusion-alg-temp 0.0 \
  --mc-num 4 \
  --mc-batch-size 2 \
  --score-progress-every 1 \
  --seed 42 \
  --no-use-4bit \
  --output-dir results/october_revision/smoke/dream-task1
```

### 3.4 Task-3 checkpoint and resume path

Run this command twice. The second call should report that the existing item is
complete and perform no generation.

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python src/ambient/cli.py task3 generate \
  --model-family dream \
  --model-name dream7b-smoke \
  --model-id Dream-org/Dream-v0-Base-7B \
  --sample-size 1 \
  --selection-seed 2026 \
  --num-continuations 2 \
  --batch-size 1 \
  --diffusion-steps 8 \
  --resume \
  --checkpoint-every 1 \
  --no-use-4bit \
  --output-path results/october_revision/smoke/dream-task3.json
```

Do not start full runs until all four smoke tests finish, their `run_meta.json`
or JSON metadata says `finished`, and the Dream outputs contain non-empty text.

### 3.5 Task-3 continuation calibration

An eight-item development sample (`selection_seed=2027`, eight continuations
per item) showed that raw Qwen2.5-7B completion is not suitable for Task 3:
QA/cloze continuations persisted at temperatures 0.2, 0.5, and 0.7. A shared
instruction prompt corrected Qwen's format but Dream-v0-Instruct-7B produced
many early-EOS, fragmentary, or contextually unsupported outputs at
temperatures 0.5, 0.7, and 1.0. These failed configurations are calibration
evidence only and must not enter the confirmatory analysis.

Dream-v0-Base-7B produced mostly natural raw continuations at temperature 0.7
and top-p 0.95, with one heuristic artifact and no exact duplicates among 64
development outputs. Mistral-7B-v0.3 passed a separate raw-completion smoke
test at the same setting. Its subsequent nine-item calibration produced 72/72
non-empty outputs, no heuristic artifacts, and no exact within-item duplicates.
Manual inspection found generally natural continuations, with occasional
context drift on generic prompts and one visibly truncated output; no further
decoding search was performed. P3 therefore uses LLaMA, LLaDA, Mistral, and
Dream as four base models with a shared raw prompt and matched decoding
parameters. This calibration establishes mechanical suitability for the shared
generation protocol; it is not evidence of semantic adequacy or comparative
model performance.

The eight-item development sample and the shared one-item smoke prompt are
frozen as nine excluded IDs in
`data/splits/task3_october_calibration_ids.txt` and excluded before the
confirmatory sample is drawn.

## 4. Historical commands remain available

These commands document the old public interface. Existing historical outputs
should normally be reused rather than overwritten.

### 4.1 Task 1

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python src/ambient/cli.py task1 run \
  --model-family llama \
  --model-name llama8b \
  --model-id meta-llama/Meta-Llama-3.1-8B \
  --num-generations 100 \
  --batch-size 100 \
  --temperature 1.0 \
  --top-p 1.0 \
  --top-k 0 \
  --seed 42 \
  --output-dir results/llama8b-n100

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python src/ambient/cli.py task1 run \
  --model-family llada \
  --model-name llada8b \
  --model-id GSAI-ML/LLaDA-8B-Base \
  --num-generations 10 \
  --batch-size 25 \
  --diffusion-steps 64 \
  --mc-num 2,4,8,16,32,64,128,256 \
  --mc-batch-size 128 \
  --cfg-scale 0.0 \
  --temperature 1.0 \
  --seed 42 \
  --output-dir results/llada8b-n10-d64

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python src/ambient/cli.py task1 metrics \
  results/llada8b-n10-d64/summary_mc256.jsonl
```

Task 1 resumes from completed instance IDs when the same output directory is
used. Stop it with `Ctrl+C`; the current item is allowed to finish before exit.

### 4.2 Tasks 2 and 3

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python src/ambient/cli.py task2 evaluate \
  --model-dirs results/llama8b-n100 results/llada8b-n10-d64 \
  --ppl-model meta-llama/Meta-Llama-3.1-8B \
  --embed-model all-MiniLM-L6-v2 \
  --seed 42

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python src/ambient/cli.py task3 generate \
  --model-family llama \
  --model-name llama8b \
  --model-id meta-llama/Meta-Llama-3.1-8B \
  --prompt-type ambiguous \
  --max-examples 580 \
  --num-continuations 100 \
  --batch-size 100 \
  --seed 42 \
  --output-path results/task3/no_trailing_quotes/llama8b_ambiguous.json

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python src/ambient/cli.py task3 generate \
  --model-family llada \
  --model-name llada8b \
  --model-id GSAI-ML/LLaDA-8B-Base \
  --prompt-type ambiguous \
  --max-examples 580 \
  --num-continuations 100 \
  --batch-size 100 \
  --diffusion-steps 128 \
  --cfg-scale 0.0 \
  --seed 42 \
  --output-path results/task3/no_trailing_quotes/llada8b_ambiguous.json

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python src/ambient/cli.py task3 evaluate \
  --results-path results/task3/no_trailing_quotes/llada8b_ambiguous.json \
  --embed-model all-MiniLM-L6-v2 \
  --nli-model roberta-large-mnli \
  --nli-thresholds argmax \
  --output-path results/task3/no_trailing_quotes/llada8b_ambiguous_evaluation.json
```

### 4.3 Tasks 4 to 6

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python src/ambient/cli.py task4 evaluate \
  --llama-model-id meta-llama/Meta-Llama-3.1-8B \
  --llada-model-id GSAI-ML/LLaDA-8B-Base \
  --data-path data/test_baked.jsonl \
  --max-examples 580 \
  --output-path results/task4/layerwise_probe_results_with_vne.json

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python src/ambient/cli.py task5 generate \
  --model-family llama \
  --model-name llama8b \
  --model-id meta-llama/Meta-Llama-3.1-8B \
  --condition gold_disambiguation \
  --max-examples 50 \
  --max-steps 20 \
  --seed 42 \
  --output-path results/task5/llama8b.json

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python src/ambient/cli.py task5 generate \
  --model-family llada \
  --model-name llada8b \
  --model-id GSAI-ML/LLaDA-8B-Base \
  --condition gold_disambiguation \
  --max-examples 50 \
  --max-steps 20 \
  --mc-num 8 \
  --seed 42 \
  --output-path results/task5/llada8b.json

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python src/ambient/cli.py task5 metrics \
  --llama-file results/task5/llama8b.json \
  --llada-file results/task5/llada8b.json \
  --output-path results/task5/temporal_semantic_commitment_metrics.json

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python src/ambient/cli.py task6 generate \
  --model-family llama \
  --model-name llama8b \
  --model-id meta-llama/Meta-Llama-3.1-8B-Instruct \
  --max-examples 580 \
  --output-path results/task6/llama8b_n1.json
```

## 5. P1: blinded human validation

The primary human sample should use the original LLaMA/LLaDA Task-3 artifacts,
not the later model-extension subset. Recommended design: 50 shared AMBIENT
items, two continuations per model and item, two independent annotators. This
produces 200 blind rows. The model key must remain hidden until annotation and
adjudication are complete.

### 5.1 Prepare a disjoint annotation pilot

Before touching the main sheets, both annotators independently label the same
eight pilot items. These items do not overlap the frozen 50-item main sample
and never enter reported results. After both pilot sheets are complete, discuss
disagreements to align the written rules, not to establish preferred model
outcomes. Keep the pilot key hidden until both independent passes are finished.

```bash
PYTHONPATH=src python src/ambient/cli.py human-eval prepare \
  --model-file llama=results/task3/no_trailing_quotes/llama8b_ambiguous.json llada=results/task3/no_trailing_quotes/llada8b_ambiguous.json \
  --id-file data/splits/human_eval_pilot_ids.txt \
  --num-instances 8 \
  --continuations-per-model 2 \
  --num-annotators 2 \
  --seed 42 \
  --stratum-label pilot \
  --output-dir results/october_revision/human_eval/pilot
```

The preparation command samples all continuation slots uniformly, including
empty outputs. Annotator order is randomized independently, while `blind_id`
keeps rows alignable for agreement analysis.

### 5.2 Freeze the guidance and prepare the main sheets

Only prepare or distribute the main package after the pilot discussion and any
resulting written clarification. Protocol version and sampled IDs are stored in
`manifest.json`.

```bash
PYTHONPATH=src python src/ambient/cli.py human-eval prepare \
  --model-file llama=results/task3/no_trailing_quotes/llama8b_ambiguous.json llada=results/task3/no_trailing_quotes/llada8b_ambiguous.json \
  --num-instances 50 \
  --continuations-per-model 2 \
  --num-annotators 2 \
  --seed 42 \
  --stratum-label random \
  --output-dir results/october_revision/human_eval/random_sample
```

Distribute only `annotation_annotator_1.csv`,
`annotation_annotator_2.csv`, and `instructions.md`. Keep `private_key.csv`
private. Each annotator labels support for every available gold reading,
invalidity, fluency, and confidence.

`human-eval prepare` refuses to replace an existing package. `--overwrite` is
reserved for a package that is known to be untouched, for example one created
before the pilot protocol was frozen. Never overwrite a sheet after annotation
has started.

### 5.3 Apply the automatic NLI judge to the same rows

Run this independently of the human annotation process so annotators never see
the automatic labels.

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python src/ambient/cli.py human-eval nli-label \
  --annotation-sheet results/october_revision/human_eval/random_sample/annotation_annotator_1.csv \
  --nli-model roberta-large-mnli \
  --threshold argmax \
  --batch-size 32 \
  --output-path results/october_revision/human_eval/random_sample/nli_labels_roberta.csv
```

### 5.4 Evaluate completed annotations

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python src/ambient/cli.py human-eval evaluate \
  --annotations results/october_revision/human_eval/random_sample/annotation_annotator_1.csv results/october_revision/human_eval/random_sample/annotation_annotator_2.csv \
  --key-file results/october_revision/human_eval/random_sample/private_key.csv \
  --nli-labels results/october_revision/human_eval/random_sample/nli_labels_roberta.csv \
  --bootstrap-reps 5000 \
  --seed 42 \
  --output-path results/october_revision/human_eval/random_sample/evaluation.json
```

The output reports per-reading agreement, Cohen's kappa, fluency agreement,
strict-consensus model summaries, paired bootstrap differences, and NLI-human
precision/recall/F1. Least-reading support is reported only for items with at
least one resolved judgment for every displayed gold reading. Use the unblinded
consensus CSV to select a small number of success, failure, and NLI-human
disagreement cases for the paper. Do not infer social rarity from AMBIENT's
least-covered reading.

## 6. P2: second controlled model pair on Tasks 1 and 2

Dream-v0-Base-7B is initialized from the Qwen2.5-7B family, so this pair is more
controlled than adding unrelated checkpoints. It is still not an architecture-
only comparison because Dream underwent substantial diffusion training after
initialization. Use equal continuation counts and the same generation chunk
size for the pair. The primary setting is `N=10`, `T=64`, and `K=256`.
Dream records the same intermediate `K=2,4,8,16,32,64,128,256` estimates as
the historical LLaDA run. Since all estimates are prefixes of the same
`K=256` sample stream, this sweep does not require additional model forwards.
MC scoring uses batch size 128, matching the historical stratified masking
schedule; a one-item Dream rescore confirmed that this fits on the A6000.

The numerical sampling controls (`temperature=1`, `top_p=1`, `top_k=0`) and
64-token first-sentence continuation policy reproduce the primary Task-1
protocol. They do not imply equal sampling entropy across model families;
Task 2 measures the resulting quality and dispersion. Batch size is matched at
two because the adapters advance their random seed once per generation chunk.

### 6.1 Final-setting P2 smoke

Before the full pair, run one item with the final generation length, matched
two-sequence chunks, `T=64`, and `K=256`. These outputs verify execution only
and are never reported.

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python src/ambient/cli.py task1 run \
  --model-family ar --model-name qwen25-7b-p2-final-smoke \
  --model-id Qwen/Qwen2.5-7B \
  --max-examples 1 --num-generations 4 --batch-size 2 \
  --mc-batch-size 16 \
  --temperature 1.0 --top-p 1.0 --top-k 0 \
  --max-new-tokens 64 --stop-at-sentence \
  --progress-every-chunks 1 --score-progress-every 10 \
  --seed 42 --no-use-4bit \
  --output-dir results/october_revision/smoke/qwen-p2-final

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python src/ambient/cli.py task1 run \
  --model-family dream --model-name dream7b-p2-final-smoke \
  --model-id Dream-org/Dream-v0-Base-7B \
  --max-examples 1 --num-generations 4 --batch-size 2 \
  --diffusion-steps 64 --diffusion-alg entropy --diffusion-alg-temp 0.0 \
  --mc-num 256 --mc-batch-size 128 --cfg-scale 0.0 \
  --temperature 1.0 --top-p 1.0 --top-k 0 \
  --max-new-tokens 64 --stop-at-sentence \
  --progress-every-chunks 1 --score-progress-every 10 \
  --seed 42 --no-use-4bit \
  --output-dir results/october_revision/smoke/dream-p2-final
```

Both `run_meta.json` files must report `finished`, the expected model ID,
`max_new_tokens=64`, `stop_at_sentence=true`, and `use_4bit=false`. Do not
change scientific settings in response to the smoke outputs unless the change
is documented as a new calibration decision.

### 6.2 Full Task 1 generation and scoring

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python src/ambient/cli.py task1 run \
  --model-family ar \
  --model-name qwen25-7b \
  --model-id Qwen/Qwen2.5-7B \
  --num-generations 10 \
  --batch-size 2 \
  --mc-batch-size 16 \
  --temperature 1.0 \
  --top-p 1.0 \
  --top-k 0 \
  --max-new-tokens 64 \
  --stop-at-sentence \
  --progress-every-chunks 1 \
  --score-progress-every 10 \
  --seed 42 \
  --no-use-4bit \
  --output-dir results/october_revision/second_pair/qwen25-7b-n10

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python src/ambient/cli.py task1 run \
  --model-family dream \
  --model-name dream7b \
  --model-id Dream-org/Dream-v0-Base-7B \
  --num-generations 10 \
  --batch-size 2 \
  --diffusion-steps 64 \
  --diffusion-alg entropy \
  --diffusion-alg-temp 0.0 \
  --mc-num 2,4,8,16,32,64,128,256 \
  --mc-batch-size 128 \
  --cfg-scale 0.0 \
  --temperature 1.0 \
  --top-p 1.0 \
  --top-k 0 \
  --max-new-tokens 64 \
  --stop-at-sentence \
  --progress-every-chunks 1 \
  --score-progress-every 10 \
  --seed 42 \
  --no-use-4bit \
  --output-dir results/october_revision/second_pair/dream7b-n10-d64
```

### 6.3 Task 1 summaries and Task 2 controls

The Qwen run completed on 4 August 2026. Its 580 ambiguity-side records reduce
to the same 543 source rows used by the historical T1 aggregate. One empty
generation caused one three-condition instance to retain nine rather than ten
continuations per condition (3/17,530 balanced slots, 0.0171%). The
normalized-cleaned aggregate evaluates 532/543 rows, with ranking accuracy
0.6898 and mean condition-level artifact rate 0.0444. These values remain an
execution record; the paired comparison is reported below.

Task-1 resume state is tracked by ambiguity side (and, for historical summaries
without side metadata, by ambiguous sentence). Row-level IDs and the default
543-row aggregation remain unchanged.

The Dream run contains all 580 ambiguity-side records at every requested K,
with no duplicate sides. It retains 17,416/17,530 balanced continuation slots;
at K=256, 17,411 are scoreable and 17,020 pass the artifact heuristic. Its
normalized-cleaned all-readings accuracy rises from 0.3812 at K=2 to 0.6188 at
K=256, with all 543 rows evaluable and mean condition-level artifact rate
0.0224. On the 532 rows evaluable for both models, Qwen scores 0.6898 and Dream
0.6222. The paired Dream-minus-Qwen difference is -0.0677 (5,000-replicate 95%
bootstrap CI [-0.1203, -0.0150]); the outcome table contains 118 Qwen-only and
82 Dream-only successes. This comparison concerns binary ranking outcomes
only: exact AR NLL and reconstruction-loss margins are not commensurate.

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python src/ambient/cli.py task1 metrics \
  results/october_revision/second_pair/qwen25-7b-n10/summary.jsonl

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python src/ambient/cli.py task1 metrics \
  results/october_revision/second_pair/dream7b-n10-d64/summary_mc256.jsonl

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python src/ambient/cli.py task2 evaluate \
  --model-dirs results/october_revision/second_pair/qwen25-7b-n10 results/october_revision/second_pair/dream7b-n10-d64 \
  --ppl-model meta-llama/Meta-Llama-3.1-8B \
  --embed-model all-MiniLM-L6-v2 \
  --output-suffix llama_oracle \
  --summary-output results/october_revision/second_pair/task2_llama_oracle.json \
  --seed 42

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python src/ambient/cli.py task2 evaluate \
  --model-dirs results/october_revision/second_pair/qwen25-7b-n10 results/october_revision/second_pair/dream7b-n10-d64 \
  --ppl-model EleutherAI/pythia-410m-deduped \
  --embed-model all-MiniLM-L6-v2 \
  --output-suffix pythia_oracle \
  --summary-output results/october_revision/second_pair/task2_pythia_oracle.json \
  --seed 42
```

Task 2 accepts either each run root or its nested `example_dirs` path. Before
loading the embedding and perplexity models, the preflight must resolve both
inputs to `.../example_dirs` and report 543 instances for each. A summary with
one evaluated instance and null metrics indicates the pre-resolution loader
bug and is not an experimental result; rerunning with the corrected loader
overwrites that summary.

The Pythia-410M oracle evaluates 543 instances for each model. Qwen has lower
external-LM perplexity than Dream (median 45.09 versus 68.75; mean 111.08
versus 290.89). Dream has slightly higher within-item embedding dispersion
(MCD 0.8080 versus 0.7759) and lower prompt overlap (0.0961 versus 0.1156).
The LLaMA-8B oracle agrees in direction: Qwen's median perplexity is 51.79 and
Dream's is 87.25 (means 166.91 and 5,407.71). The large Dream mean indicates a
long outlier tail, so median perplexity is the primary summary. Absolute values
are not compared across the two oracle models. Both oracles are causal LMs and
therefore triangulate an automatic fluency proxy rather than provide a
model-family-neutral or human quality judgment. MCD and overlap are identical
between oracle runs by construction because their encoder and texts do not
change.

Interpret this as a replication across a second pair, not as a factorial test
that isolates architecture from training data and objective.

## 7. P3: matched four-base-model Task-3 subset

Use one held-out set of 150 AMBIENT IDs for LLaMA, LLaDA, Mistral, and Dream.
Generate every continuation file afresh with the historical raw-prompt
operationalization and matched temperature/top-p settings. This is a model
breadth robustness analysis; it does not replace the full-data primary result.
Qwen remains part of P2 and P4 but is excluded here because its raw Task-3
outputs failed the format calibration.

### 7.1 Freeze the held-out IDs

Use the full historical LLaMA artifact only as the 543-source-row ID universe.
The 580 ambiguity-side record count belongs to Task 1 and is not the Task-3
sampling unit. The resulting reference subset is not one of the four newly
evaluated files.

```bash
PYTHONPATH=src python src/ambient/cli.py task3 subset \
  --results-path results/task3/no_trailing_quotes/llama8b_ambiguous.json \
  --exclude-id-file data/splits/task3_october_calibration_ids.txt \
  --sample-size 150 \
  --selection-seed 2026 \
  --output-path results/october_revision/task3/id_selection_reference.json \
  --id-output results/october_revision/task3/shared_confirmatory_ids_150.txt
```

The frozen file contains 150 unique IDs sampled from 534 eligible source rows
after excluding all nine calibration IDs; it has no calibration overlap. Its
SHA-256 digest is
`ec1afb9d864929e6cda1b75b59eec7ce4521e36e496913fa54b73f795e863c28`.

### 7.2 Generate all four matched continuation files

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python src/ambient/cli.py task3 generate \
  --model-family llama \
  --model-name llama8b-task3-t07 \
  --model-id meta-llama/Meta-Llama-3.1-8B \
  --prompt-type ambiguous --prompt-mode raw \
  --id-file results/october_revision/task3/shared_confirmatory_ids_150.txt \
  --num-continuations 100 --batch-size 25 \
  --temperature 0.7 --top-p 0.95 --top-k 0 \
  --seed 42 --resume --checkpoint-every 1 --no-use-4bit \
  --output-path results/october_revision/task3/llama8b_raw_t07_150.json

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python src/ambient/cli.py task3 generate \
  --model-family llada \
  --model-name llada8b-task3-t07 \
  --model-id GSAI-ML/LLaDA-8B-Base \
  --prompt-type ambiguous --prompt-mode raw \
  --id-file results/october_revision/task3/shared_confirmatory_ids_150.txt \
  --num-continuations 100 --batch-size 25 \
  --diffusion-steps 128 --cfg-scale 0.0 \
  --temperature 0.7 --top-p 0.95 --top-k 0 \
  --seed 42 --resume --checkpoint-every 1 --no-use-4bit \
  --output-path results/october_revision/task3/llada8b_raw_t07_150.json

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python src/ambient/cli.py task3 generate \
  --model-family ar \
  --model-name mistral7b-task3-t07 \
  --model-id mistralai/Mistral-7B-v0.3 \
  --prompt-type ambiguous --prompt-mode raw \
  --id-file results/october_revision/task3/shared_confirmatory_ids_150.txt \
  --num-continuations 100 --batch-size 25 \
  --temperature 0.7 --top-p 0.95 --top-k 0 \
  --seed 42 --resume --checkpoint-every 1 --no-use-4bit \
  --output-path results/october_revision/task3/mistral7b_raw_t07_150.json

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python src/ambient/cli.py task3 generate \
  --model-family dream \
  --model-name dream7b-task3-t07 \
  --model-id Dream-org/Dream-v0-Base-7B \
  --prompt-type ambiguous --prompt-mode raw \
  --id-file results/october_revision/task3/shared_confirmatory_ids_150.txt \
  --num-continuations 100 --batch-size 2 \
  --diffusion-steps 128 --diffusion-alg entropy --diffusion-alg-temp 0.0 \
  --cfg-scale 0.0 \
  --temperature 0.7 --top-p 0.95 --top-k 0 \
  --seed 42 --resume --checkpoint-every 1 --no-use-4bit \
  --output-path results/october_revision/task3/dream7b_raw_t07_150.json
```

### 7.3 Record lightweight generation-quality diagnostics

Run these before semantic evaluation. Empty, heuristic-artifact, and exact
duplicate rates are descriptive controls and must not be presented as human
fluency judgments.

```bash
PYTHONPATH=src python src/ambient/cli.py task3 quality \
  --results-path results/october_revision/task3/llama8b_raw_t07_150.json
PYTHONPATH=src python src/ambient/cli.py task3 quality \
  --results-path results/october_revision/task3/llada8b_raw_t07_150.json
PYTHONPATH=src python src/ambient/cli.py task3 quality \
  --results-path results/october_revision/task3/mistral7b_raw_t07_150.json
PYTHONPATH=src python src/ambient/cli.py task3 quality \
  --results-path results/october_revision/task3/dream7b_raw_t07_150.json
```

The LLaMA artifact contains all 15,000 requested non-empty outputs. The frozen
heuristic flags 2,475 (16.5%), and there are 2,225 within-item exact duplicate
excesses. Inspection shows that repeated short numbered outputs account for
much of both counts: 1,660 duplicate excesses occur in suspicious-text groups,
versus 565 in non-suspicious groups. These are generated outcomes, not missing
data, so the artifact is retained rather than regenerated.

The LLaDA artifact returns all 15,000 slots, of which 14,988 are non-empty.
Its surface-artifact rate is lower than LLaMA's (6.94% versus 16.5%), while its
within-item exact duplicate excess is substantially higher (28.02% versus
14.83%). This difference is descriptive evidence about sampled concentration,
but it can also affect geometry-based metrics. Section 7.6 therefore adds a
unique-output sensitivity before any semantic result is inspected.

The Mistral artifact also contains all 15,000 requested non-empty outputs. It
has substantially lower heuristic-artifact and exact-duplicate-excess rates
than the first two artifacts (0.46% and 5.06%, respectively), and its outputs
are somewhat longer (median 13 words, versus 11 for LLaMA and 10 for LLaDA).
These model-specific quality differences are reported as controls rather than
treated as ambiguity effects.

The Dream artifact returns all 15,000 requested non-empty outputs. Its
surface-artifact rate is 1.48%, its exact-duplicate-excess rate is 5.79%, and
its median output length is 12 words. Dream is therefore much closer to
Mistral than to LLaDA on these controls. The resulting variation does not
support treating generation quality or repetition as a simple property of the
AR-versus-diffusion distinction.

### 7.4 Apply one frozen semantic evaluation to all four files

The historical all-nonempty analysis remains primary. Set
`--artifact-policy keep` explicitly and retain exact duplicate multiplicities,
which represent probability mass in the sampled output distribution. Section
7.5 repeats the same evaluation after dropping only texts identified by the
pre-existing surface-artifact heuristic. This sensitivity protocol was frozen
after the LLaMA quality audit and before any four-model semantic result was
computed. Filtering produces model- and item-specific effective sample counts,
so it is a robustness check rather than a replacement for the matched
100-sample primary analysis.

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python src/ambient/cli.py task3 evaluate \
  --results-path results/october_revision/task3/llama8b_raw_t07_150.json \
  --embed-model all-MiniLM-L6-v2 --nli-model roberta-large-mnli \
  --nli-thresholds argmax,0.5,0.8 \
  --artifact-policy keep --duplicate-policy keep \
  --output-path results/october_revision/task3/llama8b_t07_evaluation.json

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python src/ambient/cli.py task3 evaluate \
  --results-path results/october_revision/task3/llada8b_raw_t07_150.json \
  --embed-model all-MiniLM-L6-v2 --nli-model roberta-large-mnli \
  --nli-thresholds argmax,0.5,0.8 \
  --artifact-policy keep --duplicate-policy keep \
  --output-path results/october_revision/task3/llada8b_t07_evaluation.json

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python src/ambient/cli.py task3 evaluate \
  --results-path results/october_revision/task3/mistral7b_raw_t07_150.json \
  --embed-model all-MiniLM-L6-v2 --nli-model roberta-large-mnli \
  --nli-thresholds argmax,0.5,0.8 \
  --artifact-policy keep --duplicate-policy keep \
  --output-path results/october_revision/task3/mistral7b_t07_evaluation.json

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python src/ambient/cli.py task3 evaluate \
  --results-path results/october_revision/task3/dream7b_raw_t07_150.json \
  --embed-model all-MiniLM-L6-v2 --nli-model roberta-large-mnli \
  --nli-thresholds argmax,0.5,0.8 \
  --artifact-policy keep --duplicate-policy keep \
  --output-path results/october_revision/task3/dream7b_t07_evaluation.json
```

The paper already contains MiniLM/MPNet and RoBERTa/DeBERTa robustness checks.
Do not multiply judge combinations unless the four-model result changes
direction and needs diagnosis.

### 7.5 Artifact-filter sensitivity

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python src/ambient/cli.py task3 evaluate \
  --results-path results/october_revision/task3/llama8b_raw_t07_150.json \
  --embed-model all-MiniLM-L6-v2 --nli-model roberta-large-mnli \
  --nli-thresholds argmax,0.5,0.8 \
  --artifact-policy drop --duplicate-policy keep \
  --output-path results/october_revision/task3/llama8b_t07_evaluation_clean.json

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python src/ambient/cli.py task3 evaluate \
  --results-path results/october_revision/task3/llada8b_raw_t07_150.json \
  --embed-model all-MiniLM-L6-v2 --nli-model roberta-large-mnli \
  --nli-thresholds argmax,0.5,0.8 \
  --artifact-policy drop --duplicate-policy keep \
  --output-path results/october_revision/task3/llada8b_t07_evaluation_clean.json

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python src/ambient/cli.py task3 evaluate \
  --results-path results/october_revision/task3/mistral7b_raw_t07_150.json \
  --embed-model all-MiniLM-L6-v2 --nli-model roberta-large-mnli \
  --nli-thresholds argmax,0.5,0.8 \
  --artifact-policy drop --duplicate-policy keep \
  --output-path results/october_revision/task3/mistral7b_t07_evaluation_clean.json

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python src/ambient/cli.py task3 evaluate \
  --results-path results/october_revision/task3/dream7b_raw_t07_150.json \
  --embed-model all-MiniLM-L6-v2 --nli-model roberta-large-mnli \
  --nli-thresholds argmax,0.5,0.8 \
  --artifact-policy drop --duplicate-policy keep \
  --output-path results/october_revision/task3/dream7b_t07_evaluation_clean.json
```

### 7.6 Clean unique-output sensitivity

This third view first applies the same surface-artifact filter and then keeps
only the first NFKC-normalized, whitespace-normalized, case-insensitive exact
match per item. Unlike the primary analysis, it estimates geometry and reading
support among unique sampled texts rather than weighting modes by empirical
frequency. Effective sample counts vary, so this view is reported only as a
sensitivity analysis.

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python src/ambient/cli.py task3 evaluate \
  --results-path results/october_revision/task3/llama8b_raw_t07_150.json \
  --embed-model all-MiniLM-L6-v2 --nli-model roberta-large-mnli \
  --nli-thresholds argmax,0.5,0.8 \
  --artifact-policy drop --duplicate-policy drop \
  --output-path results/october_revision/task3/llama8b_t07_evaluation_clean_unique.json

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python src/ambient/cli.py task3 evaluate \
  --results-path results/october_revision/task3/llada8b_raw_t07_150.json \
  --embed-model all-MiniLM-L6-v2 --nli-model roberta-large-mnli \
  --nli-thresholds argmax,0.5,0.8 \
  --artifact-policy drop --duplicate-policy drop \
  --output-path results/october_revision/task3/llada8b_t07_evaluation_clean_unique.json

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python src/ambient/cli.py task3 evaluate \
  --results-path results/october_revision/task3/mistral7b_raw_t07_150.json \
  --embed-model all-MiniLM-L6-v2 --nli-model roberta-large-mnli \
  --nli-thresholds argmax,0.5,0.8 \
  --artifact-policy drop --duplicate-policy drop \
  --output-path results/october_revision/task3/mistral7b_t07_evaluation_clean_unique.json

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python src/ambient/cli.py task3 evaluate \
  --results-path results/october_revision/task3/dream7b_raw_t07_150.json \
  --embed-model all-MiniLM-L6-v2 --nli-model roberta-large-mnli \
  --nli-thresholds argmax,0.5,0.8 \
  --artifact-policy drop --duplicate-policy drop \
  --output-path results/october_revision/task3/dream7b_t07_evaluation_clean_unique.json
```

## 8. P4: second dataset, Scope Ambiguities Experiment 2B

The implementation follows the official Experiment-2 contrast:

`alpha = -[(log P(F1|S) - log P(F2|S)) - (log P(F1|Sc) - log P(F2|Sc))]`.

It reports mean alpha with bootstrap CI, proportion of positive alphas, the
paired test between ambiguous and control differences, operator-type summaries,
and Pearson correlation with the published human proxy alphas. AR models use
exact continuation NLL. Masked-diffusion models use the stated MC reconstruction
proxy, so raw alpha magnitudes remain architecture-dependent.

### 8.1 Obtain the official data once

```bash
git clone https://github.com/McGill-NLP/scope-ambiguity.git external/scope-ambiguity
```

The repository is already ignored through `/external/`.

### 8.2 Run LLaMA, LLaDA, Qwen, and Dream

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python src/ambient/cli.py scope score \
  --data-path external/scope-ambiguity/datasets/exp2b_base_dataset.csv \
  --human-results external/scope-ambiguity/human_results/exp2b_human_results_cleaned.csv \
  --model-family llama --model-name llama8b \
  --model-id meta-llama/Meta-Llama-3.1-8B \
  --scoring-method exact --batch-size 16 --progress-every 20 \
  --seed 42 --no-use-4bit \
  --output-path results/october_revision/scope/llama8b_exp2b.json

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python src/ambient/cli.py scope score \
  --data-path external/scope-ambiguity/datasets/exp2b_base_dataset.csv \
  --human-results external/scope-ambiguity/human_results/exp2b_human_results_cleaned.csv \
  --model-family llada --model-name llada8b \
  --model-id GSAI-ML/LLaDA-8B-Base \
  --scoring-method mc --mc-num 256 --batch-size 16 --progress-every 10 \
  --seed 42 --no-use-4bit \
  --output-path results/october_revision/scope/llada8b_exp2b_mc256.json

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python src/ambient/cli.py scope score \
  --data-path external/scope-ambiguity/datasets/exp2b_base_dataset.csv \
  --human-results external/scope-ambiguity/human_results/exp2b_human_results_cleaned.csv \
  --model-family ar --model-name qwen25-7b \
  --model-id Qwen/Qwen2.5-7B \
  --scoring-method exact --batch-size 16 --progress-every 20 \
  --seed 42 --no-use-4bit \
  --output-path results/october_revision/scope/qwen25-7b_exp2b.json

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python src/ambient/cli.py scope score \
  --data-path external/scope-ambiguity/datasets/exp2b_base_dataset.csv \
  --human-results external/scope-ambiguity/human_results/exp2b_human_results_cleaned.csv \
  --model-family dream --model-name dream7b \
  --model-id Dream-org/Dream-v0-Base-7B \
  --scoring-method mc --mc-num 256 --batch-size 16 --progress-every 10 \
  --seed 42 --no-use-4bit \
  --output-path results/october_revision/scope/dream7b_exp2b_mc256.json
```

Combine the completed outputs into one paper-facing table source:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python src/ambient/cli.py scope summarize \
  --model-result llama=results/october_revision/scope/llama8b_exp2b.json llada=results/october_revision/scope/llada8b_exp2b_mc256.json qwen=results/october_revision/scope/qwen25-7b_exp2b.json dream=results/october_revision/scope/dream7b_exp2b_mc256.json \
  --output-path results/october_revision/scope/comparison.json \
  --csv-path results/october_revision/scope/comparison.csv
```

The implementation has been checked against all 110 complete Experiment-2B
items and all 110 corresponding published human proxy scores.

## 9. P5: Task-1 PLL triangulation

This experiment reuses saved continuations from the workstation's complete
`example_dirs`. It does not regenerate text. The alternative score masks one
continuation token at a time while all other prompt and continuation tokens are
visible, then sums token reconstruction cross-entropies. It is a deterministic
pseudo-NLL, not a calibrated sequence likelihood.

### 9.1 Rescore the primary LLaDA T=64 run

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python src/ambient/cli.py task1 rescore \
  --run-dir results/llada8b-n10-d64 \
  --model-family llada \
  --model-id GSAI-ML/LLaDA-8B-Base \
  --scoring-method pll \
  --batch-size 16 \
  --progress-every 10 \
  --seed 42 \
  --no-use-4bit \
  --output-dir results/october_revision/pll/llada8b-n10-d64

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python src/ambient/cli.py task1 metrics \
  results/october_revision/pll/llada8b-n10-d64/summary_pll.jsonl

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python src/ambient/cli.py task1 compare-scorers \
  --reference-summary results/llada8b-n10-d64/summary_mc256.jsonl \
  --alternative-summary results/october_revision/pll/llada8b-n10-d64/summary_pll.jsonl \
  --metric-key empirical_KL_div_normalized_clean \
  --dedupe instance \
  --bootstrap-reps 5000 \
  --seed 42 \
  --output-path results/october_revision/pll/llada8b_d64_mc_vs_pll.json
```

### 9.2 Repeat at the low-quality T=4 setting

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python src/ambient/cli.py task1 rescore \
  --run-dir results/llada8b-n10-d4 \
  --model-family llada \
  --model-id GSAI-ML/LLaDA-8B-Base \
  --scoring-method pll \
  --batch-size 16 \
  --progress-every 10 \
  --seed 42 \
  --no-use-4bit \
  --output-dir results/october_revision/pll/llada8b-n10-d4

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python src/ambient/cli.py task1 compare-scorers \
  --reference-summary results/llada8b-n10-d4/summary_mc256.jsonl \
  --alternative-summary results/october_revision/pll/llada8b-n10-d4/summary_pll.jsonl \
  --metric-key empirical_KL_div_normalized_clean \
  --dedupe instance \
  --bootstrap-reps 5000 \
  --seed 42 \
  --output-path results/october_revision/pll/llada8b_d4_mc_vs_pll.json
```

After P2, the same command can rescore Dream's saved continuations by changing
the family, model ID, and run directory. Agreement in direction across MC and
PLL strengthens robustness; disagreement should be reported rather than hidden.

## 10. P6: full matched continuation budget

The existing LLaMA run already has N=100. Generate LLaDA with N=100 at the
primary T=64 setting first. T=4 is a secondary matched-budget quality-stress
setting and can be skipped if compute becomes disproportionate.

### 10.1 Primary T=64

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python src/ambient/cli.py task1 run \
  --model-family llada \
  --model-name llada8b-matched \
  --model-id GSAI-ML/LLaDA-8B-Base \
  --num-generations 100 \
  --batch-size 25 \
  --diffusion-steps 64 \
  --mc-num 256 \
  --mc-batch-size 128 \
  --cfg-scale 0.0 \
  --temperature 1.0 \
  --seed 42 \
  --no-use-4bit \
  --score-progress-every 10 \
  --output-dir results/october_revision/matched_budget/llada8b-n100-d64
```

### 10.2 Secondary T=4

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python src/ambient/cli.py task1 run \
  --model-family llada \
  --model-name llada8b-matched \
  --model-id GSAI-ML/LLaDA-8B-Base \
  --num-generations 100 \
  --batch-size 25 \
  --diffusion-steps 4 \
  --mc-num 256 \
  --mc-batch-size 128 \
  --cfg-scale 0.0 \
  --temperature 1.0 \
  --seed 42 \
  --no-use-4bit \
  --score-progress-every 10 \
  --output-dir results/october_revision/matched_budget/llada8b-n100-d4
```

### 10.3 Summaries, uncertainty, and quality

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python src/ambient/cli.py task1 metrics \
  results/october_revision/matched_budget/llada8b-n100-d64/summary_mc256.jsonl

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python src/ambient/cli.py robustness task1-ranking \
  --run-dirs results/llama8b-n100 results/october_revision/matched_budget/llada8b-n100-d64 \
  --metric-key normalized_cleaned \
  --bootstrap-reps 5000 \
  --seed 42 \
  --output-dir results/october_revision/matched_budget/robustness_d64

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python src/ambient/cli.py task2 evaluate \
  --model-dirs results/llama8b-n100 results/october_revision/matched_budget/llada8b-n100-d64 \
  --ppl-model EleutherAI/pythia-410m-deduped \
  --embed-model all-MiniLM-L6-v2 \
  --output-suffix matched_pythia \
  --summary-output results/october_revision/matched_budget/task2_d64_pythia.json \
  --seed 42
```

Equal N removes the continuation-count mismatch but does not equalize total
inference compute, denoising steps, or scorer semantics. State that distinction
explicitly in the revision.

## 11. P7: optional second-pair Task 5

Run this only after P1 to P6. It tests whether the original checkpoint-level
pattern replicates, but still compares AR prefix growth with fixed mask-ratio
probes rather than logged Dream denoising states.

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python src/ambient/cli.py task5 generate \
  --model-family ar --model-name qwen25-7b \
  --model-id Qwen/Qwen2.5-7B \
  --condition gold_disambiguation \
  --max-examples 50 --max-steps 20 --seed 42 \
  --no-use-4bit \
  --output-path results/october_revision/task5/qwen25-7b_gold.json

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python src/ambient/cli.py task5 generate \
  --model-family dream --model-name dream7b \
  --model-id Dream-org/Dream-v0-Base-7B \
  --condition gold_disambiguation \
  --max-examples 50 --max-steps 20 --mc-num 8 --seed 42 \
  --no-use-4bit \
  --output-path results/october_revision/task5/dream7b_gold.json

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python src/ambient/cli.py task5 metrics \
  --llama-file results/october_revision/task5/qwen25-7b_gold.json \
  --llada-file results/october_revision/task5/dream7b_gold.json \
  --ar-label Qwen2.5-7B \
  --diffusion-label Dream-v0-Base-7B \
  --bootstrap-reps 5000 \
  --seed 42 \
  --output-path results/october_revision/task5/qwen_dream_metrics.json
```

## 12. Recommended execution order

1. Freeze the branch revision and run all smoke tests.
2. Prepare the disjoint human-annotation pilot and arrange two annotators. P1
   may remain open while independent GPU experiments proceed.
3. Start P2 Task 1. In parallel, complete the pilot, freeze the guidance, and
   regenerate the untouched main sheets before annotation begins.
4. After P2, start the four fresh P3 generations with checkpointing.
5. Run the P5 PLL rescoring in parallel only if another GPU is available.
6. Run P4 Scope Experiment 2B for all four models.
7. Run P6 LLaDA N=100 at T=64; decide later whether T=4 is worth the cost.
8. Complete annotation analysis and qualitative examples.
9. Review all primary outcomes with the supervisors before starting optional P7.

## 13. Pre-paper decision rules

Record these decisions before inspecting results:

- Human evaluation is primary validation of Task-3 NLI coverage, not a search
  for favorable examples.
- Pilot rows train the annotation protocol and are excluded from every reported
  human result.
- The random human sample is the main sample. Any targeted stress sample is
  separate and labeled exploratory.
- The four-model Task-3 comparison uses exactly
  `shared_confirmatory_ids_150.txt`, after excluding the frozen calibration IDs.
- Experiment 2B is the primary second dataset. Experiment 2A is not substituted
  after seeing results.
- T=64 is the primary matched-budget setting. T=4 is a quality-stress control.
- MC versus PLL is a triangulation. Neither estimator is called ground truth.
- Architecture-wide language is used only if both model pairs agree; otherwise
  report checkpoint-specific patterns.
- Failed replications, judge disagreement, and human/NLI disagreement remain in
  the paper and determine the final claim strength.

## 14. Output checklist

Before drafting the October revision, verify that the extension folder contains:

- `CODE_REVISION.txt` and runtime metadata for every model run;
- completed human annotation sheets, private key, agreement output, and
  qualitative examples, plus the separately labeled pilot package;
- Qwen/Dream Task-1 summaries and Task-2 controls;
- four Task-3 generation artifacts on the exact same 150 IDs and four
  corresponding evaluation files;
- four Scope Experiment-2B JSON/CSV outputs;
- MC/PLL item-level agreement files at T=64 and T=4;
- LLaDA N=100 T=64 output, bootstrap summary, and quality controls;
- a short experiment ledger with command, start/end time, GPU, exit status,
  and any deviation from this protocol.
