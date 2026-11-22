import json
from pathlib import Path
import unittest
from core import extract, entity_keys, relation_keys, score_pairs

class ExtractionTests(unittest.TestCase):
    def test_exact_offsets(self):
        text='Amara Okafor works at Cedar Labs in Kigali.'
        result=extract(text)
        self.assertEqual(entity_keys(result), {(0, 12, 'PERSON'), (22, 32, 'ORG'), (36, 42, 'LOC')})
        for entity in result['entities']:
            self.assertEqual(text[entity['start']:entity['end']],entity['text'])
    def test_direct_employment(self):
        r=extract('Amara Okafor works at Cedar Labs.')
        self.assertEqual([v['type'] for v in r['relations']],['WORKS_FOR'])
    def test_negation_abstains(self):
        self.assertEqual(extract('Amara Okafor did not join River AI.')['relations'],[])
    def test_uncertainty_abstains(self):
        self.assertEqual(extract('Leila Njeri may join Cedar Labs.')['relations'],[])
    def test_whole_entity_boundaries(self):
        self.assertEqual(extract('Cedar LabsX')['entities'],[])
    def test_known_paraphrase_limitation(self):
        r=extract('Leila Njeri co-founded River AI.')
        self.assertEqual(len(r['entities']),2)
        self.assertEqual(r['relations'],[])
    def test_relation_direction_matters(self):
        gold={'entities':[],'relations':[{'type':'R','head':[0,2],'tail':[3,5]}]}
        pred={'entities':[],'relations':[{'type':'R','head':[3,5],'tail':[0,2]}]}
        self.assertEqual(score_pairs([(pred,gold)],relation_keys),{'tp':0,'fp':1,'fn':1,'precision':0.,'recall':0.,'f1':0.})
    def test_metric_counts_and_documents(self):
        correct={'entities':[{'start':0,'end':2,'label':'PERSON'}],'relations':[]}
        wrong={'entities':[{'start':0,'end':3,'label':'PERSON'}],'relations':[]}
        score=score_pairs([(correct,correct),(wrong,correct)],entity_keys)
        self.assertEqual((score['tp'],score['fp'],score['fn']),(1,1,1))
        self.assertEqual(score['f1'],.5)
    def test_empty_metric(self):
        self.assertEqual(score_pairs([],entity_keys)['f1'],0.)
    def test_fixture_integrity(self):
        docs=json.loads((Path(__file__).resolve().parents[1]/'data/examples.json').read_text())
        self.assertEqual(len(docs),12)
        self.assertEqual(len({d['id'] for d in docs}),12)
        for d in docs:
            spans=set()
            for e in d['entities']:
                self.assertEqual(d['text'][e['start']:e['end']],e['text'])
                spans.add((e['start'],e['end']))
            for r in d['relations']:
                self.assertIn(tuple(r['head']),spans);self.assertIn(tuple(r['tail']),spans)
