from ambient.utils import is_suspicious


def test_mcq_filter_keeps_sentences_starting_with_indefinite_article() -> None:
    assert not is_suspicious("A few of the lights were out.")


def test_mcq_filter_rejects_explicit_option_labels() -> None:
    assert is_suspicious("A. First option")
    assert is_suspicious("B) Second option")
