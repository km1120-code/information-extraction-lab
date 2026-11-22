from pathlib import Path
import csv
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from core import extract, entity_keys, relation_keys, score_pairs

ROOT = Path(__file__).parent
OUT = ROOT / 'results'
OUT.mkdir(exist_ok=True)
fixture = ROOT / 'data/examples.json'
documents = json.loads(fixture.read_text())
predictions = [dict(id=d['id'], title=d['title'], text=d['text'], **extract(d['text'])) for d in documents]
pairs = list(zip(predictions, documents))
metrics = {'entities':score_pairs(pairs,entity_keys),'relations':score_pairs(pairs,relation_keys)}
(OUT/'predictions.json').write_text(json.dumps(predictions,indent=2,ensure_ascii=False)+'\n')
with (OUT/'metrics.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=['task','tp','fp','fn','precision','recall','f1'])
    writer.writeheader()
    for name,values in metrics.items():writer.writerow(dict(task=name,**values))
summary={'experiment':'Gazetteer and pattern-based information extraction',
         'dataset':'12 hand-authored synthetic examples, including known failure cases',
         'documents':len(documents),'entity_types':['PERSON','ORG','LOC'],
         'relation_types':['WORKS_FOR','FOUNDED','BASED_IN'],
         'evaluation':'Micro exact-match precision, recall and F1; half-open character offsets',
         'fixture_sha256':hashlib.sha256(fixture.read_bytes()).hexdigest(),
         'metrics':metrics,
         'limitations':'Illustrative fixture designed alongside the baseline; not a held-out benchmark. No generalization claim.'}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
 'axes.spines.right':False,'axes.labelcolor':'#34433e','text.color':'#34433e',
 'axes.edgecolor':'#b6beb8','xtick.color':'#64726b','ytick.color':'#64726b',
 'figure.facecolor':'#faf9f5','axes.facecolor':'#faf9f5','savefig.facecolor':'#faf9f5'})
fig,ax=plt.subplots(figsize=(7.4,4.4),layout='constrained')
labels=['Precision','Recall','F1'];x=[0,1,2];width=.32
for offset,task,color,label in [(-width/2,'entities','#235b48','Entity mentions'),(width/2,'relations','#b77c45','Relations')]:
 vals=[metrics[task][m.lower()] for m in labels]
 bars=ax.bar([v+offset for v in x],vals,width,color=color,label=label,zorder=3)
 ax.bar_label(bars,labels=[f'{v:.2f}' for v in vals],padding=4,fontsize=8)
ax.set_xticks(x,labels);ax.set(ylim=(0,1.18),ylabel='Micro exact-match score',title='A small information extraction baseline')
ax.set_yticks([0,.25,.5,.75,1]);ax.grid(axis='y',alpha=.15,zorder=0)
ax.legend(frameon=False,loc='upper right',fontsize=8)
fig.text(.5,.005,'12 synthetic examples · Illustrative fixture, not a held-out benchmark',ha='center',fontsize=8,color='#64726b')
fig.savefig(OUT/'extraction-evaluation.png',dpi=180)
print(json.dumps(summary))
