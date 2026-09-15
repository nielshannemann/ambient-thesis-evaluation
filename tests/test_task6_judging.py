from types import SimpleNamespace

from ambient.evaluation.task6_evaluation import (
    build_consensus_label,
    build_pairwise_agreement,
    compute_cohens_kappa,
    parse_judge_response,
    resolve_judge_model_ids,
)


def test_single_judge_parse_keeps_backward_compatible_tie_fallback() -> None:
    winner, valid = parse_judge_response("Model A", llada_is_model_a=True)
    assert (winner, valid) == ("LLaDA", True)

    winner, valid = parse_judge_response("Something else", llada_is_model_a=False)
    assert (winner, valid) == ("Tie", False)


def test_multi_judge_agreement_and_consensus_helpers() -> None:
    rows = [
        {
            "judges": {
                "judge_a": {"winner_model": "LLaDA"},
                "judge_b": {"winner_model": "LLaDA"},
            }
        },
        {
            "judges": {
                "judge_a": {"winner_model": "Tie"},
                "judge_b": {"winner_model": "LLaMA-8B"},
            }
        },
    ]

    pairwise = build_pairwise_agreement(rows, ["judge_a", "judge_b"])
    assert len(pairwise) == 1
    assert pairwise[0]["num_instances"] == 2
    assert pairwise[0]["raw_agreement"] == 0.5

    assert build_consensus_label(["LLaDA", "LLaDA"]) == "LLaDA"
    assert build_consensus_label(["LLaDA", "Tie"]) == "Tie"


def test_cohens_kappa_and_default_judge_resolution() -> None:
    kappa = compute_cohens_kappa(
        ["LLaDA", "Tie", "LLaMA-8B"],
        ["LLaDA", "Tie", "Tie"],
    )
    assert -1.0 <= kappa <= 1.0

    args = SimpleNamespace(
        judge_models=None,
        judge_model=None,
        default_judge_models=["judge_primary", "judge_secondary"],
    )
    assert resolve_judge_model_ids(args) == ["judge_primary", "judge_secondary"]
