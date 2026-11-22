# Information Extraction Lab

A small, reproducible educational baseline for **information extraction (IE)**:
recognizing entity mentions, extracting typed relations, and evaluating both
with exact-match precision, recall, and F1.

The repository provides annotated examples, executable extraction rules, and
explicit evaluation criteria for studying the behavior of a simple baseline.

## What the example does

`core.py` finds names in a fixed gazetteer and applies a few explicit relation
patterns between adjacent entity mentions. It emits:

- Entity spans with `PERSON`, `ORG`, or `LOC` labels.
- Directed `WORKS_FOR`, `FOUNDED`, or `BASED_IN` relations between spans.
- Character offsets using the half-open convention `[start, end)`.

For example, `Amara Okafor works at Cedar Labs in Kigali.` contains a person,
an organization, and a location. The supported relation is
`(Amara Okafor, WORKS_FOR, Cedar Labs)`. The text does not explicitly say that
Cedar Labs is based in Kigali, so the example does not add that relation.

## Data and evaluation

`data/examples.json` contains **12 hand-authored synthetic sentences** and
separately authored gold entity/relation annotations. The annotations are not
produced by the extractor. All people and organizations in the sentences are
fictional examples.

The cases deliberately include supported patterns, negation, uncertainty,
unseen names, and paraphrases the baseline cannot handle. This fixture was
created alongside the baseline for explanation and testing. It is **not a
held-out benchmark** and provides no evidence of generalization to new text.

Entity matching requires the exact start offset, end offset, and entity type.
Relation matching requires the exact relation type, direction, and both endpoint
spans. Counts are aggregated across documents before computing micro PRF:

```text
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
F1        = 2 TP / (2 TP + FP + FN)
```

Undefined ratios are set to zero. Duplicates within a document count once, but
identical spans in different documents are evaluated separately. The CSV and
JSON include raw counts as well as scores.

## Run

Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python demo.py
```

The extractor and scorer use only Python's standard library. Matplotlib is
needed solely to generate the chart. The experiment is deterministic and does
not download a model or call an API.

## Files

- `core.py`: entity and relation extraction, exact-match scoring.
- `data/examples.json`: fixed synthetic examples and gold annotations.
- `demo.py`: inference, evaluation, and chart generation.
- `results/predictions.json`: predictions for every sentence, including failures.
- `results/metrics.csv`: exact-match scores and TP/FP/FN counts.
- `results/summary.json`: data hash, settings, limitations, and results.
- `tests/test_core.py`: offsets, abstention, direction, scoring, and fixture checks.

## Limits

A fixed gazetteer misses unseen names. Rules miss paraphrases and do not perform
coreference resolution. Surface names alone do not disambiguate entities.
A rule failing to match is abstention, not a calibrated probability. No score
here should be interpreted as production-quality NLP performance.

## Links

- Repository: https://github.com/km1120-code/information-extraction-lab
- Homepage: https://km1120-code.github.io/
- Project page: https://km1120-code.github.io/projects/information-extraction-lab.html

Code is provided under the MIT license.
