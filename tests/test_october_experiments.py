import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
import pandas as pd

from ambient.adapters import ARAdapter
from ambient.evaluation.get_log_likelihood import get_pseudo_log_likelihood
from ambient.evaluation.continuation_evaluation_adapted import create_test_instances
from ambient.evaluation.human_evaluation import (
    binary_kappa,
    binary_value,
    bootstrap_mean_ci,
    paired_bootstrap_difference,
)
from ambient.evaluation.scope_ambiguity import build_summary, load_scope_items
from ambient.evaluation.task1_compare_scorers import index_rows
from ambient.generation.task3_silhouette_generate import run as run_task3_generation
from ambient.generation.task3_silhouette_generate import select_examples
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


def test_task1_scorer_comparison_preserves_composite_instance_ids(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.jsonl"
    summary_path.write_text(
        '{"id":"12_abcd","row_id":12,"options":{}}\n',
        encoding="utf-8",
    )
    assert set(index_rows(summary_path, "instance")) == {"12_abcd"}
    assert set(index_rows(summary_path, "row")) == {"12"}


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
