import json

from ambient.evaluation.task4_linear_probing import (
    build_probe_records_from_examples,
    build_unambiguous_length_matched_probe_control,
    build_vne_control_pairs,
)
from ambient.visualization.task4_layerwise import run as run_task4_plotter


def _examples() -> list[dict]:
    return [
        {
            "id": "amb-1",
            "premise": "Workers may strike.",
            "hypothesis": "Workers will strike.",
            "premise_ambiguous": True,
            "hypothesis_ambiguous": False,
            "disambiguations": [
                {"premise": "Workers are permitted to strike.", "hypothesis": "Workers will strike.", "label": "contradiction"},
                {"premise": "Workers might strike.", "hypothesis": "Workers will strike.", "label": "entailment"},
            ],
            "distractor_premise": "Workers may stripe.",
        },
        {
            "id": "amb-2",
            "premise": "The board considered the offer.",
            "hypothesis": "The board accepted the offer.",
            "premise_ambiguous": True,
            "hypothesis_ambiguous": False,
            "disambiguations": [
                {"premise": "The board thought about the offer.", "hypothesis": "The board accepted the offer.", "label": "contradiction"},
                {"premise": "The board carefully considered the offer before accepting it.", "hypothesis": "The board accepted the offer.", "label": "entailment"},
            ],
            "distractor_premise": "The plank considered the offer.",
        },
        {
            "id": "plain-ent",
            "premise": "All swans are birds.",
            "hypothesis": "Swans are birds.",
            "premise_ambiguous": False,
            "hypothesis_ambiguous": False,
            "labels": "entailment",
            "disambiguations": [],
        },
        {
            "id": "plain-con",
            "premise": "No swans are mammals.",
            "hypothesis": "Swans are mammals.",
            "premise_ambiguous": False,
            "hypothesis_ambiguous": False,
            "labels": "contradiction",
            "disambiguations": [],
        },
    ]


def test_unambiguous_probe_control_matches_labels_without_reuse() -> None:
    reference_records, _ = build_probe_records_from_examples(_examples(), mode="fully_disambiguated", max_examples=2)
    matched_records, metadata = build_unambiguous_length_matched_probe_control(_examples(), reference_records)

    assert metadata["num_pairs"] == len(matched_records)
    assert all(record["source_type"] == "none" for record in matched_records)
    assert len({record["group"] for record in matched_records}) == len(matched_records)
    assert {record["label"] for record in matched_records} <= {"entailment", "contradiction"}


def test_vne_control_builders_return_deterministic_non_empty_pairs() -> None:
    distractor_pairs, distractor_meta = build_vne_control_pairs(
        _examples(),
        input_mode="sentence_only",
        condition="distractor_rewrite",
        max_examples=2,
    )
    random_pairs, random_meta = build_vne_control_pairs(
        _examples(),
        input_mode="sentence_only",
        condition="random_matched_rewrite",
        max_examples=2,
    )

    assert distractor_pairs
    assert random_pairs
    assert distractor_meta["condition"] == "distractor_rewrite"
    assert random_meta["condition"] == "random_matched_rewrite"
    assert all("length_gap_tokens" in pair for pair in distractor_pairs)
    assert all("control_source_instance_id" in pair for pair in random_pairs)


def test_task4_plotter_reads_legacy_json_without_control_blocks(tmp_path) -> None:
    payload = {
        "config": {"llama_model": "x", "llada_model": "y", "data_path": "data/test_baked.jsonl"},
        "datasets": {
            "side_reconstructed": {"num_pairs": 2},
            "von_neumann_entropy": {"num_pairs": 1},
        },
        "results": {
            "side_reconstructed": {
                "llama": {"1": {"mean_accuracy": 0.5, "std_accuracy": 0.0, "mean_probe_entropy_bits": 0.2, "std_probe_entropy_bits": 0.0}},
                "llada": {"1": {"mean_accuracy": 0.6, "std_accuracy": 0.0, "mean_probe_entropy_bits": 0.3, "std_probe_entropy_bits": 0.0}},
            }
        },
        "von_neumann_entropy": {
            "llama": {"1": {"ambiguous_raw_entropy_bits_mean": 0.2, "ambiguous_raw_entropy_bits_std": 0.0, "ambiguous_normalized_entropy_mean": 0.2, "ambiguous_normalized_entropy_std": 0.0, "disambiguated_raw_entropy_bits_mean": 0.1, "disambiguated_raw_entropy_bits_std": 0.0, "disambiguated_normalized_entropy_mean": 0.1, "disambiguated_normalized_entropy_std": 0.0, "delta_raw_entropy_bits_mean": -0.1, "delta_raw_entropy_bits_std": 0.0, "delta_normalized_entropy_mean": -0.1, "delta_normalized_entropy_std": 0.0}},
            "llada": {"1": {"ambiguous_raw_entropy_bits_mean": 0.4, "ambiguous_raw_entropy_bits_std": 0.0, "ambiguous_normalized_entropy_mean": 0.4, "ambiguous_normalized_entropy_std": 0.0, "disambiguated_raw_entropy_bits_mean": 0.2, "disambiguated_raw_entropy_bits_std": 0.0, "disambiguated_normalized_entropy_mean": 0.2, "disambiguated_normalized_entropy_std": 0.0, "delta_raw_entropy_bits_mean": -0.2, "delta_raw_entropy_bits_std": 0.0, "delta_normalized_entropy_mean": -0.2, "delta_normalized_entropy_std": 0.0}},
        },
    }
    input_path = tmp_path / "task4.json"
    output_dir = tmp_path / "plots"
    input_path.write_text(json.dumps(payload), encoding="utf-8")

    args = type("Args", (), {"input": input_path, "output_dir": output_dir, "no_error_band": True})()
    assert run_task4_plotter(args) == 0
    assert (output_dir / "combined_summary.txt").exists()
