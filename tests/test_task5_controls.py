from ambient.generation.task5_temporal_semantic_commitment import build_task5_instances
from ambient.evaluation.task5_compute_decay_metrics import _extract_trajectory_map
from ambient.evaluation.task5_compute_decay_metrics import _trajectory_to_instance_metrics
from ambient.paths import task5_output_path
from ambient.visualization.task5_plot_decay import _extract_trajectory_records


def _examples() -> list[dict]:
    return [
        {
            "id": "a1",
            "premise": "The workers can leave early.",
            "hypothesis": "The workers will leave early.",
            "premise_ambiguous": True,
            "hypothesis_ambiguous": False,
            "disambiguations": [
                {"premise": "The workers are allowed to leave early.", "label": "contradiction"},
                {"premise": "The workers might leave early.", "label": "entailment"},
            ],
            "distractor_premise": "The workers can leaf early.",
        },
        {
            "id": "a2",
            "premise": "The board considered the offer.",
            "hypothesis": "The board accepted the offer.",
            "premise_ambiguous": True,
            "hypothesis_ambiguous": False,
            "disambiguations": [
                {"premise": "The board thought about the offer.", "label": "contradiction"},
                {"premise": "The board carefully considered the offer before accepting it.", "label": "entailment"},
            ],
            "distractor_premise": "The plank considered the offer.",
        },
    ]


def test_task5_gold_condition_preserves_default_path_contract() -> None:
    gold_instances = build_task5_instances(_examples(), condition="gold_disambiguation", max_examples=2)
    assert gold_instances
    assert gold_instances[0]["condition"] == "gold_disambiguation"
    assert task5_output_path("llama8b") == task5_output_path("llama8b", condition="gold_disambiguation")


def test_task5_control_conditions_add_metadata_and_suffix_paths() -> None:
    distractor_instances = build_task5_instances(_examples(), condition="distractor_rewrite", max_examples=2)
    random_instances = build_task5_instances(_examples(), condition="random_matched_rewrite", max_examples=2)

    assert distractor_instances
    assert random_instances
    assert all(instance["target_b_source"] == "distractor_rewrite" for instance in distractor_instances)
    assert all(instance["target_b_source"] == "random_matched_rewrite" for instance in random_instances)
    assert all("length_gap_tokens" in instance for instance in distractor_instances + random_instances)
    assert task5_output_path("llama8b", condition="distractor_rewrite").name == "llama8b_distractor_rewrite.json"


def test_task5_metrics_and_plot_helpers_accept_current_trajectory_shape() -> None:
    payload = {
        "metadata": {"condition": "gold_disambiguation"},
        "results": {
            "a1": {
                "trajectory": [
                    {"step": 0, "entropy": 1.0},
                    {"step": 1, "entropy": 0.2},
                ]
            }
        },
    }

    trajectory_map = _extract_trajectory_map(payload)
    trajectory_records = _extract_trajectory_records(payload)

    assert list(trajectory_map.keys()) == ["a1"]
    assert len(trajectory_records) == 1


def test_task5_instance_metrics_compute_auc_across_numpy_versions() -> None:
    trajectory = [
        {"step": 0, "entropy": 0.0},
        {"step": 1, "entropy": 1.0},
        {"step": 2, "entropy": 0.0},
    ]

    metrics = _trajectory_to_instance_metrics(trajectory)

    assert metrics is not None
    assert metrics["Mean Start Entropy (H_0)"] == 0.0
    assert metrics["Mean End Entropy (H_100)"] == 0.0
    assert metrics["Mean Peak Entropy (H_max)"] == 1.0
    assert metrics["Area Under Entropy Curve (AUC)"] == 0.5
