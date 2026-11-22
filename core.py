"""Transparent gazetteer and relation-pattern baseline, with exact-match scoring."""
import re

LEXICON = {
    'PERSON': ('Amara Okafor', 'Kofi Mensah', 'Leila Njeri'),
    'ORG': ('Cedar Labs', 'River AI'),
    'LOC': ('Kigali', 'Accra'),
}
RELATION_PATTERNS = (
    ('PERSON', 'ORG', 'WORKS_FOR', re.compile(r'\s+(?:works at|joined)\s+', re.I)),
    ('PERSON', 'ORG', 'FOUNDED', re.compile(r'\s+founded\s+', re.I)),
    ('ORG', 'LOC', 'BASED_IN', re.compile(r'\s+is based in\s+', re.I)),
)


def extract(text):
    """Return entity mentions and relations; offsets are half-open characters.

    This baseline recognizes a fixed lexicon and three explicit verb patterns.
    It is neither a pretrained model nor a general-purpose language parser.
    """
    if not isinstance(text, str):
        raise TypeError('text must be a string')
    candidates = []
    for label, names in LEXICON.items():
        for name in names:
            for match in re.finditer(r'(?<!\w)' + re.escape(name) + r'(?!\w)', text, re.I):
                candidates.append({'start': match.start(), 'end': match.end(),
                                   'label': label, 'text': match.group()})
    # Longest mention first at the same start; do not emit overlapping entities.
    entities = []
    for entity in sorted(candidates, key=lambda x: (x['start'], -x['end'], x['label'])):
        if not entities or entity['start'] >= entities[-1]['end']:
            entities.append(entity)
    relations = []
    for head, tail in zip(entities, entities[1:]):
        between = text[head['end']:tail['start']]
        for head_type, tail_type, label, pattern in RELATION_PATTERNS:
            if head['label'] == head_type and tail['label'] == tail_type and pattern.fullmatch(between):
                relations.append({'type': label,
                                  'head': [head['start'], head['end']],
                                  'tail': [tail['start'], tail['end']]})
    return {'entities': entities, 'relations': relations}


def entity_keys(result):
    return {(x['start'], x['end'], x['label']) for x in result['entities']}


def relation_keys(result):
    return {(x['type'], *x['head'], *x['tail']) for x in result['relations']}


def score_pairs(pairs, key_fn):
    """Micro exact-match PRF over (prediction, gold) pairs, document by document.

    A mention must match both character boundaries and its label. A relation
    must match its type, direction, and both endpoint spans. Duplicate records
    within a document count once. Undefined ratios are reported as zero.
    """
    tp = fp = fn = 0
    for predicted, gold in pairs:
        pred_set, gold_set = key_fn(predicted), key_fn(gold)
        tp += len(pred_set & gold_set)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)
    precision = tp / (tp + fp) if tp + fp else 0.
    recall = tp / (tp + fn) if tp + fn else 0.
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.
    return {'tp': tp, 'fp': fp, 'fn': fn, 'precision': precision, 'recall': recall, 'f1': f1}
