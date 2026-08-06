import csv
import json
import math
import random
from pathlib import Path
from types import SimpleNamespace

import torch
import pandas as pd
import pytest

from ambient.adapters import ARAdapter
from ambient.dream_loader import _decode_dream_suffix
from ambient.evaluation.get_log_likelihood import get_pseudo_log_likelihood
from ambient.evaluation.continuation_evaluation_adapted import (
    create_test_instances,
    task1_instance_was_processed,
    task1_resume_sentence_key,
)
from ambient.evaluation.run_ambient_experiments import task1_resume_mismatches
from ambient.evaluation.human_evaluation import (
    binary_kappa,
    binary_value,
    bootstrap_mean_ci,
    choose_continuations,
    paired_bootstrap_difference,
    prepare as prepare_human_evaluation,
)
from ambient.evaluation.scope_ambiguity import build_summary, load_scope_items
from ambient.evaluation.task1_compare_scorers import index_rows, validate_row_alignment
from ambient.evaluation.task2_semantic_diversity import (
    resolve_task2_model_input,
    task2_continuation_files,
)
from ambient.evaluation.task3_generation_quality import summarize_task3_quality
from ambient.evaluation.task3_compare import build_comparison
from ambient.evaluation.task3_silhouette_evaluate import select_task3_continuations
from ambient.evaluation.task3_subset import run as run_task3_subset
from ambient.generation.task3_silhouette_generate import (
    TASK3_CHAT_SYSTEM_PROMPT,
    TASK3_CHAT_USER_TEMPLATE,
    build_task3_generation_prompt,
    run as run_task3_generation,
    select_examples,
)
from ambient.modeling import (
    canonical_backend,
    default_base_model_id,
    is_autoregressive_family,
    is_masked_diffusion_family,
)


class FakeTokenizer:
    mask_token_id = 9

    def __call__(self, text, add_special_tokens=True, **_kwargs):
        tokens = [2 + index for index, _token in enumerate(str(text).split())]
        if add_special_tokens:
            tokens = [1] + tokens
        return {"input_ids": tokens}


class UniformMaskedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(10, 4)
        self.config = SimpleNamespace(mask_token_id=9)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids=None, attention_mask=None):
        assert attention_mask is None
        batch, length = input_ids.shape
        return SimpleNamespace(logits=torch.zeros(batch, length, 10, device=input_ids.device))


class MovableBatch(dict):
    def to(self, device):
        return MovableBatch({key: value.to(device) for key, value in self.items()})


class FakeARTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, _text, return_tensors=None):
        return MovableBatch(
            {
                "input_ids": torch.tensor([[1]], dtype=torch.long),
                "attention_mask": torch.tensor([[1]], dtype=torch.long),
            }
        )

    def batch_decode(self, sequences, skip_special_tokens=True):
        return ["output."] * len(sequences)


class FakeARGenerationModel:
    device = torch.device("cpu")

    def __init__(self):
        self.kwargs = None

    def generate(self, **kwargs):
        self.kwargs = kwargs
        suffix = torch.full((kwargs["input_ids"].shape[0], 1), 2, dtype=torch.long)
        return torch.cat([kwargs["input_ids"], suffix], dim=1)


class FakeChatTokenizer:
    def __init__(self):
        self.messages = None

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        self.messages = messages
        assert tokenize is False
        assert add_generation_prompt is True
        return "FORMATTED CHAT PROMPT"


class FakeDreamDecodeTokenizer:
    eos_token = "<eos>"

    def decode(self, token_ids, skip_special_tokens, clean_up_tokenization_spaces):
        assert token_ids == [11, 12, 13]
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is True
        return "Useful continuation.<eos>Tokens after EOS must be ignored."


def test_model_family_aliases_preserve_old_names_and_add_second_pair() -> None:
    assert canonical_backend("llama") == "ar"
    assert canonical_backend("llada") == "llada"
    assert canonical_backend("dream") == "dream"
    assert is_autoregressive_family("ar")
    assert is_masked_diffusion_family("llada")
    assert is_masked_diffusion_family("dream")
    assert default_base_model_id("ar") == "Qwen/Qwen2.5-7B"
    assert default_base_model_id("dream") == "Dream-org/Dream-v0-Base-7B"


def test_single_token_pll_sums_continuation_cross_entropy() -> None:
    model = UniformMaskedModel()
    tokenizer = FakeTokenizer()
    scores = get_pseudo_log_likelihood(
        model,
        tokenizer,
        prompts=["prompt"],
        continuations=["first second"],
        batch_size=2,
    )
    assert math.isclose(scores[0], 2.0 * math.log(10.0), rel_tol=1e-6)


def test_scope_loader_and_alpha_summary(tmp_path: Path) -> None:
    data_path = tmp_path / "scope.csv"
    fieldnames = ["idx", "sentence", "followup", "stype", "ftype", "OP1", "OP1_type", "OP2", "OP2_type"]
    rows = [
        {"idx": 1, "sentence": "amb", "followup": "f1", "stype": "S", "ftype": "F1", "OP1": "not", "OP1_type": "NEG", "OP2": "a", "OP2_type": "IND"},
        {"idx": 1, "sentence": "amb", "followup": "f2", "stype": "S", "ftype": "F2", "OP1": "not", "OP1_type": "NEG", "OP2": "a", "OP2_type": "IND"},
        {"idx": 1, "sentence": "control", "followup": "f1", "stype": "Sc", "ftype": "F1", "OP1": "NA", "OP1_type": "NA", "OP2": "NA", "OP2_type": "NA"},
        {"idx": 1, "sentence": "control", "followup": "f2", "stype": "Sc", "ftype": "F2", "OP1": "NA", "OP1_type": "NA", "OP2": "NA", "OP2_type": "NA"},
    ]
    with data_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    items = load_scope_items(data_path)
    assert len(items) == 1
    assert items[0]["followup_f2"] == "f2"

    result_rows = [
        {
            "idx": "1",
            "alpha": 2.0,
            "logp_sf1": -1.0,
            "logp_sf2": -2.0,
            "logp_scf1": -1.0,
            "logp_scf2": -4.0,
            "op1_type": "NEG",
            "op2_type": "IND",
        },
        {
            "idx": "2",
            "alpha": 1.0,
            "logp_sf1": -1.0,
            "logp_sf2": -2.0,
            "logp_scf1": -1.0,
            "logp_scf2": -3.0,
            "op1_type": "NEG",
            "op2_type": "IND",
        },
    ]
    summary = build_summary(result_rows, None, bootstrap_reps=20, seed=3)
    assert summary["alpha"]["mean"] == 1.5
    assert summary["proportion_positive_alpha"] == 1.0


def test_human_label_parsing_and_kappa() -> None:
    assert binary_value("yes") == 1
    assert binary_value("No") == 0
    assert binary_value("uncertain") is None
    assert binary_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0


def test_human_bootstrap_summaries_are_paired() -> None:
    summary = bootstrap_mean_ci([0.0, 1.0], reps=100, seed=3)
    assert summary["mean"] == 0.5
    paired = paired_bootstrap_difference([0.0, 0.5], [0.5, 1.0], reps=100, seed=3)
    assert paired["difference_b_minus_a"] == 0.5
    assert paired["n"] == 2


def test_human_continuation_sampling_does_not_filter_empty_slots() -> None:
    selected = choose_continuations(
        {"id": "x", "continuations": ["", "usable continuation"]},
        count=2,
        rng=random.Random(42),
    )
    assert {index for index, _text in selected} == {0, 1}
    assert any(not text for _index, text in selected)


def test_human_package_is_versioned_blinded_and_overwrite_protected(tmp_path: Path) -> None:
    model_paths = []
    for model in ("a", "b"):
        path = tmp_path / f"{model}.json"
        path.write_text(
            json.dumps(
                {
                    "results": [
                        {
                            "id": f"item-{item}",
                            "ambiguity_side": "hypothesis",
                            "gold_disambiguations": [
                                {"hypothesis": f"reading {item} one"},
                                {"hypothesis": f"reading {item} two"},
                            ],
                            "continuations": [
                                f"{model} item {item} continuation {continuation}"
                                for continuation in range(4)
                            ],
                        }
                        for item in range(2)
                    ]
                }
            ),
            encoding="utf-8",
        )
        model_paths.append(f"{model}={path}")

    output_dir = tmp_path / "human"
    args = SimpleNamespace(
        model_file=model_paths,
        id_file=None,
        num_instances=2,
        continuations_per_model=2,
        num_annotators=2,
        seed=42,
        stratum_label="test",
        output_dir=output_dir,
        overwrite=False,
    )
    assert prepare_human_evaluation(args) == 0

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["annotation_protocol_version"] == "1.1"
    assert manifest["num_instances_included"] == 2

    with (output_dir / "annotation_annotator_1.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        first = list(csv.DictReader(handle))
    with (output_dir / "annotation_annotator_2.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        second = list(csv.DictReader(handle))
    assert {row["blind_id"] for row in first} == {row["blind_id"] for row in second}
    assert [row["blind_id"] for row in first] != [row["blind_id"] for row in second]
    assert "actual_model" not in first[0]
    assert "Mere grammatical compatibility" in (
        output_dir / "instructions.md"
    ).read_text(encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        prepare_human_evaluation(args)

    args.overwrite = True
    assert prepare_human_evaluation(args) == 0


def test_task1_scorer_comparison_preserves_composite_instance_ids(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.jsonl"
    summary_path.write_text(
        '{"id":"12_abcd","row_id":12,"options":{}}\n',
        encoding="utf-8",
    )
    assert set(index_rows(summary_path, "instance")) == {"12_abcd"}
    assert set(index_rows(summary_path, "row")) == {"12"}


def test_task1_scorer_comparison_keeps_last_repeated_instance(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.jsonl"
    summary_path.write_text(
        '{"id":"12","ambiguous_sentence":"first"}\n'
        '{"id":"12","ambiguous_sentence":"last"}\n',
        encoding="utf-8",
    )
    indexed = index_rows(summary_path, "instance")
    assert indexed["12"]["ambiguous_sentence"] == "last"


def test_task1_scorer_comparison_validates_prompt_and_candidate_alignment() -> None:
    reference = {
        "12": {
            "ambiguous_sentence": "Ambiguous.",
            "options": {
                "y0": {"sentence": "Reading one."},
                "d": {"sentence": "Distractor."},
            },
        }
    }
    matching = json.loads(json.dumps(reference))
    validate_row_alignment(reference, matching, ["12"])

    mismatched = json.loads(json.dumps(reference))
    mismatched["12"]["options"]["y0"]["sentence"] = "Different reading."
    with pytest.raises(ValueError, match="prompt or candidate readings"):
        validate_row_alignment(reference, mismatched, ["12"])


def test_task3_id_selection_handles_zero(tmp_path: Path) -> None:
    id_path = tmp_path / "ids.txt"
    id_path.write_text("0\n", encoding="utf-8")
    selected = select_examples(
        [{"id": 0}, {"id": 1}],
        id_file=id_path,
        sample_size=None,
        selection_seed=2026,
    )
    assert selected == [{"id": 0}]


def test_task3_subset_excludes_development_ids_before_sampling(tmp_path: Path) -> None:
    source_path = tmp_path / "source.json"
    source_path.write_text(
        json.dumps({"metadata": {}, "results": [{"id": value} for value in "abcd"]}),
        encoding="utf-8",
    )
    exclude_path = tmp_path / "exclude.txt"
    exclude_path.write_text("b\n", encoding="utf-8")
    output_path = tmp_path / "subset.json"
    ids_path = tmp_path / "subset.ids.txt"
    args = SimpleNamespace(
        results_path=source_path,
        id_file=None,
        exclude_id_file=exclude_path,
        sample_size=3,
        selection_seed=2026,
        output_path=output_path,
        id_output=ids_path,
    )

    assert run_task3_subset(args) == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert [item["id"] for item in payload["results"]] == ["a", "c", "d"]
    assert ids_path.read_text(encoding="utf-8").splitlines() == ["a", "c", "d"]
    assert payload["metadata"]["subset"]["num_excluded_items"] == 1


def test_task3_quality_counts_empty_missing_artifact_and_duplicate_outputs() -> None:
    summary = summarize_task3_quality(
        {
            "metadata": {"hyperparameters": {"num_continuations": 3}},
            "results": [
                {"id": "x", "continuations": ["A sentence.", "A sentence.", ""]},
                {"id": "y", "continuations": ["2.", "Valid output."]},
            ],
        }
    )
    assert summary["expected_slots"] == 6
    assert summary["returned_slots"] == 5
    assert summary["nonempty_count"] == 4
    assert summary["empty_or_missing_count"] == 2
    assert summary["heuristic_artifact_count"] == 1
    assert summary["nonempty_nonartifact_count"] == 3
    assert summary["exact_duplicate_excess_count"] == 1
    assert summary["items_with_empty_or_missing"] == 2


def test_task3_raw_prompt_mode_preserves_historical_whitespace_suffix() -> None:
    assert build_task3_generation_prompt(None, "Ambiguous text.", "raw") == "Ambiguous text. "


def test_task3_chat_prompt_uses_tokenizer_template_and_records_instruction() -> None:
    tokenizer = FakeChatTokenizer()
    prompt = build_task3_generation_prompt(tokenizer, "Ambiguous text. ", "chat_continuation")
    assert prompt == "FORMATTED CHAT PROMPT"
    assert tokenizer.messages == [
        {"role": "system", "content": TASK3_CHAT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": TASK3_CHAT_USER_TEMPLATE.format(text="Ambiguous text."),
        },
    ]


def test_dream_suffix_decoding_stops_at_first_eos_before_cleaning() -> None:
    sequence = torch.tensor([1, 2, 11, 12, 13])
    decoded = _decode_dream_suffix(FakeDreamDecodeTokenizer(), sequence, prompt_len=2)
    assert decoded == "Useful continuation."


def test_ar_adapter_removes_diffusion_only_generation_arguments() -> None:
    model = FakeARGenerationModel()
    adapter = ARAdapter("fake", model, FakeARTokenizer(), ar_score_fn=None)
    output = adapter.generate(
        "prompt",
        num_return_sequences=1,
        batch_size=1,
        diffusion_alg="entropy",
        diffusion_alg_temp=0.0,
    )
    assert output == ["output."]
    assert "diffusion_alg" not in model.kwargs
    assert "diffusion_alg_temp" not in model.kwargs


def test_task3_completed_resume_returns_before_model_loading(tmp_path: Path) -> None:
    data_path = tmp_path / "data.jsonl"
    data_path.write_text(
        json.dumps(
            {
                "id": 0,
                "premise": "Ambiguous premise.",
                "hypothesis": "Hypothesis.",
                "premise_ambiguous": True,
                "hypothesis_ambiguous": False,
                "disambiguations": [
                    {"premise": "Reading one."},
                    {"premise": "Reading two."},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "task3.json"
    output_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "model_id": "Qwen/Qwen2.5-7B",
                    "prompt_type": "ambiguous",
                },
                "results": [{"id": 0, "continuations": ["Done."]}],
            }
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        seed=42,
        model_family="ar",
        model_name="qwen",
        model_id="Qwen/Qwen2.5-7B",
        prompt_type="ambiguous",
        num_continuations=1,
        batch_size=1,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        cfg_scale=0.0,
        diffusion_steps=8,
        diffusion_alg="entropy",
        diffusion_alg_temp=0.0,
        progress_every_chunks=1,
        data_path=data_path,
        max_examples=1,
        id_file=None,
        sample_size=None,
        selection_seed=2026,
        resume=True,
        checkpoint_every=1,
        use_4bit=False,
        output_path=output_path,
    )
    assert run_task3_generation(args) == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["metadata"]["status"] == "finished"
    assert len(payload["results"]) == 1


def test_task3_resume_rejects_changed_generation_hyperparameters(tmp_path: Path) -> None:
    data_path = tmp_path / "data.jsonl"
    data_path.write_text(
        json.dumps(
            {
                "id": 0,
                "premise": "Ambiguous premise.",
                "hypothesis": "Hypothesis.",
                "premise_ambiguous": True,
                "hypothesis_ambiguous": False,
                "disambiguations": [
                    {"premise": "Reading one."},
                    {"premise": "Reading two."},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "task3.json"
    output_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "model_id": "Qwen/Qwen2.5-7B",
                    "prompt_type": "ambiguous",
                    "prompt_mode": "raw",
                    "hyperparameters": {"temperature": 0.5},
                },
                "results": [{"id": 0, "continuations": ["Done."]}],
            }
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        seed=42,
        model_family="ar",
        model_name="qwen",
        model_id="Qwen/Qwen2.5-7B",
        prompt_type="ambiguous",
        prompt_mode="raw",
        num_continuations=1,
        batch_size=1,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        cfg_scale=0.0,
        diffusion_steps=8,
        diffusion_alg="entropy",
        diffusion_alg_temp=0.0,
        progress_every_chunks=1,
        data_path=data_path,
        max_examples=1,
        id_file=None,
        sample_size=None,
        selection_seed=2026,
        resume=True,
        checkpoint_every=1,
        use_4bit=False,
        output_path=output_path,
    )
    with pytest.raises(ValueError, match="hyperparameters.temperature"):
        run_task3_generation(args)


def test_task1_readings_keep_benchmark_order_while_deduplicating() -> None:
    frame = pd.DataFrame(
        [
            {
                "id": 1,
                "premise": "Ambiguous.",
                "hypothesis": "Fixed.",
                "premise_ambiguous": True,
                "hypothesis_ambiguous": False,
                "distractor_premise": "Distractor.",
                "disambiguations": [
                    {"premise": "Reading B."},
                    {"premise": "Reading A."},
                    {"premise": "Reading B."},
                ],
            }
        ]
    )
    instances = create_test_instances(frame)
    assert instances[0]["disambiguations"] == ["Reading B.", "Reading A."]


def test_task2_resolves_run_root_and_direct_example_dirs(tmp_path: Path) -> None:
    run_root = tmp_path / "model-run"
    examples_root = run_root / "example_dirs"
    instance_dir = examples_root / "item-1"
    instance_dir.mkdir(parents=True)
    (instance_dir / "y0.jsonl").write_text("{}\n", encoding="utf-8")

    expected = ("model-run", run_root, examples_root)
    assert resolve_task2_model_input(run_root) == expected
    assert resolve_task2_model_input(examples_root) == expected
    assert task2_continuation_files(instance_dir) == [instance_dir / "y0.jsonl"]


def test_task2_preserves_direct_historical_layout(tmp_path: Path) -> None:
    direct_root = tmp_path / "custom-results"
    instance_dir = direct_root / "item-1"
    instance_dir.mkdir(parents=True)
    (instance_dir / "reading.jsonl").write_text("{}\n", encoding="utf-8")
    (instance_dir / "d.jsonl").write_text("{}\n", encoding="utf-8")
    (instance_dir / "prompts.jsonl").write_text("{}\n", encoding="utf-8")

    assert resolve_task2_model_input(direct_root) == (
        "custom-results",
        direct_root,
        direct_root,
    )
    assert task2_continuation_files(instance_dir) == [
        instance_dir / "reading.jsonl"
    ]


def test_task3_artifact_sensitivity_preserves_clean_duplicates() -> None:
    item = {
        "continuations": [
            "A coherent continuation.",
            "A coherent continuation.",
            "1.",
            "",
        ]
    }

    kept, nonempty_count, artifact_count, duplicate_count = (
        select_task3_continuations(item, "keep", "keep")
    )
    assert kept == ["A coherent continuation.", "A coherent continuation.", "1."]
    assert nonempty_count == 3
    assert artifact_count == 0
    assert duplicate_count == 0

    cleaned, nonempty_count, artifact_count, duplicate_count = (
        select_task3_continuations(item, "drop", "keep")
    )
    assert cleaned == ["A coherent continuation.", "A coherent continuation."]
    assert nonempty_count == 3
    assert artifact_count == 1
    assert duplicate_count == 0

    unique, nonempty_count, artifact_count, duplicate_count = (
        select_task3_continuations(item, "drop", "drop")
    )
    assert unique == ["A coherent continuation."]
    assert nonempty_count == 3
    assert artifact_count == 1
    assert duplicate_count == 1


def test_task3_comparison_uses_aligned_paired_item_differences() -> None:
    def payload(offset: float) -> dict:
        return {
            "artifact_policy": "keep",
            "duplicate_policy": "keep",
            "item_metrics": [
                {
                    "id": "a",
                    "mcd": 1.0 + offset,
                    "silhouette": 0.1 + offset,
                    "mtc_cos_percent": 10.0 + offset,
                    "mtc_nli_percent": {
                        "argmax": 1.0 + offset,
                        "0.5": 0.8 + offset,
                        "0.8": 0.5 + offset,
                    },
                },
                {
                    "id": "b",
                    "mcd": 2.0 + offset,
                    "silhouette": 0.2 + offset,
                    "mtc_cos_percent": 20.0 + offset,
                    "mtc_nli_percent": {
                        "argmax": 2.0 + offset,
                        "0.5": 1.8 + offset,
                        "0.8": 1.5 + offset,
                    },
                },
            ],
        }

    comparison = build_comparison(
        {"first": payload(0.0), "second": payload(0.5)},
        bootstrap_reps=100,
        ci_level=95.0,
        seed=42,
    )

    assert comparison["num_aligned_items"] == 2
    paired = comparison["paired_differences"]["second_minus_first"]
    assert paired["mcd"]["mean_difference"] == pytest.approx(0.5)
    assert paired["mcd"]["n"] == 2
    assert paired["mtc_nli_argmax_percent"]["mean_difference"] == pytest.approx(0.5)


def test_task1_resume_preserves_unfinished_side_of_dual_ambiguous_row() -> None:
    frame = pd.DataFrame(
        [
            {
                "id": "dual_1",
                "premise": "Ambiguous premise.",
                "hypothesis": "Ambiguous hypothesis.",
                "premise_ambiguous": True,
                "hypothesis_ambiguous": True,
                "distractor_premise": "Premise distractor.",
                "distractor_hypothesis": "Hypothesis distractor.",
                "disambiguations": [
                    {
                        "premise": "Premise reading one.",
                        "hypothesis": "Hypothesis reading one.",
                    },
                    {
                        "premise": "Premise reading two.",
                        "hypothesis": "Hypothesis reading two.",
                    },
                ],
            }
        ]
    )
    premise, hypothesis = create_test_instances(frame)
    historical_completed = {
        task1_resume_sentence_key("dual_1", '"Ambiguous premise.')
    }

    assert task1_instance_was_processed(
        premise, set(), historical_completed, set()
    )
    assert not task1_instance_was_processed(
        hypothesis, set(), historical_completed, set()
    )


def test_task1_resume_rejects_changed_scientific_settings_but_accepts_old_metadata() -> None:
    current = {
        "model_name": "qwen25-7b",
        "model_id": "Qwen/Qwen2.5-7B",
        "model_type": "ar",
        "model_family": "ar",
        "backend": "ar",
        "reading_order": "benchmark_order_with_stable_deduplication",
        "hyperparameters": {
            "seed": 42,
            "num_generations": 10,
            "gen_batch_size": 2,
            "max_new_tokens": 64,
            "stop_at_sentence": True,
        },
        "data_selection": {
            "data_path": "data/test_baked.jsonl",
            "data_sha256": "abc",
            "id_file": None,
        },
    }
    assert task1_resume_mismatches({}, current) == {}
    assert task1_resume_mismatches(current, current) == {}

    changed = json.loads(json.dumps(current))
    changed["hyperparameters"]["gen_batch_size"] = 10
    mismatches = task1_resume_mismatches(current, changed)
    assert mismatches["hyperparameters.gen_batch_size"] == (2, 10)

    changed_data = json.loads(json.dumps(current))
    changed_data["data_selection"]["data_sha256"] = "def"
    mismatches = task1_resume_mismatches(current, changed_data)
    assert mismatches["data_selection.data_sha256"] == ("abc", "def")
