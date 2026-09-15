# Data provenance

`test_baked.jsonl` is the processed benchmark input used by the experiment
CLI. It contains AMBIENT natural-language-inference records, their annotated
readings, ambiguity-side indicators, and the fixed distractors required by the
reading-ranking experiment.

The source benchmark is described in:

> Alisa Liu et al. "We're Afraid Language Models Aren't Modeling Ambiguity."
> EMNLP 2023, pages 790-807.
> https://aclanthology.org/2023.emnlp-main.51/

AMBIENT is distributed by its creators under CC BY 4.0. This repository does
not replace the original dataset documentation. Users should consult the
source release for the complete license, intended use, and dataset statement.

The default CLI path is `data/test_baked.jsonl`. Dataset preparation and
descriptive-analysis commands are available through `ambient dataset --help`.
