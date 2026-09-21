"""Development-only scope check. Official test set is never evaluated."""
import hashlib, json, re, statistics, time, urllib.request
from collections import defaultdict
from pathlib import Path
from system1 import ChoiceField, DecisionSchema, SystemOneCompiler, System1Engine
from system1.core.text import TfidfProjector
url='https://raw.githubusercontent.com/clinc/oos-eval/828f8093932c8fe6ca7936c3d2e52903b1c523de/data/data_full.json'
raw=urllib.request.urlopen(url, timeout=30).read()
assert hashlib.sha256(raw).hexdigest()=='36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0'
data=json.loads(raw)
def group(text):return hashlib.sha256(' '.join(re.findall(r'\w+',text.casefold())).encode()).hexdigest()
# Prefer validation over teaching when normalized duplicates cross splits.
val=data['val']+data['oos_val']; forbidden={group(t) for t,y in val}
by_label=defaultdict(list)
for text,label in data['train']+data['oos_train']:
    if group(text) not in forbidden:by_label[label].append((text,label))
fit=[];checks=[]
for label,rows in sorted(by_label.items()):
    rows=sorted({group(t):(t,y) for t,y in rows}.values(),key=lambda r:group(r[0]))
    checks+=rows[:20];fit+=rows[20:]
class Route(DecisionSchema):intent=ChoiceField(options=sorted(by_label))
started=time.perf_counter();projector=TfidfProjector.fit([t for t,y in fit],max_features=2048)
model=SystemOneCompiler(Route,projector=projector,regularization=.1).compile({'intent':fit},augment=False,calibration_exemplars={'intent':checks})
compile_ms=(time.perf_counter()-started)*1000
engine=System1Engine(Route,model=model,strict_mode=True,use_cache=False)
rows=[]
for text,label in val:
    start=time.perf_counter();answer=engine.decide(text,record_receipt=False)
    rows.append({'label':label,'predicted':answer.values['intent'],'review':bool(answer.is_ambiguous),'ms':(time.perf_counter()-start)*1000})
normal=[r for r in rows if r['label']!='oos'];accepted=[r for r in normal if not r['review'] and r['predicted']!='oos'];ood=[r for r in rows if r['label']=='oos']
report={'scope':'development validation only, 150 intents plus explicit out-of-scope label; not an n8n integration or LLM teacher result','fit':len(fit),'calibration':len(checks),'validation':len(normal),'correct':sum(r['label']==r['predicted'] for r in normal),'accepted':len(accepted),'accepted_correct':sum(r['label']==r['predicted'] for r in accepted),'ood_cases':len(ood),'ood_review_or_other':sum(r['review'] or r['predicted']=='oos' for r in ood),'compile_ms':compile_ms,'median_decision_ms':statistics.median(r['ms'] for r in rows)}
Path('/tmp/system1-gauntlet-feasibility.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
