from __future__ import annotations
import csv, importlib.util, sys, tempfile
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
# Windows does not ship the SocioVerse runtime; stub imports before tajfel package init.
import types
for name in ["socioverse", "socioverse.abc", "socioverse.engine", "socioverse.schemas"]:
    sys.modules[name] = types.ModuleType(name)
class Dummy:
    def __init__(self,*a,**kw):
        for k,v in kw.items(): setattr(self,k,v)
for n in ("DecisionModel","EnvironmentProvider","MetricCollector","PopulationProvider"): setattr(sys.modules["socioverse.abc"],n,Dummy)
sys.modules["socioverse.engine"].register=lambda kind,key: (lambda cls: cls)
for n in ("Action","Observation","Broadcast","EnvironmentBundle","EnvironmentLayer","InformationProgram","InteractionStructure","Persona","PopulationBundle","PropagationMode","ScheduledEvent","SimulationConfig","StudySpec"): setattr(sys.modules["socioverse.schemas"],n,Dummy)
sys.path.insert(0,str(ROOT))
from tajfel.feedback import ARMS, FeedbackEnvironmentProvider, FeedbackDecisionModel, FeedbackMetricCollector, build_feedback_prompt, run_arm

def check(ok,msg):
 if not ok: raise AssertionError(msg)
def main():
 for seed in range(8):
  e=FeedbackEnvironmentProvider(seed=seed)
  for t in range(1,6):
   e.advance_to(t); ps=e.pairings[t]; check(len(ps)==5,f"seed {seed} step {t}: five pairs")
   check(sum(p[2]=='AA' for p in ps)==1 and sum(p[2]=='BB' for p in ps)==1 and sum(p[2]=='AB' for p in ps)==3,"pair structure")
   flat=[x for a,b,_ in ps for x in (a,b)]; check(len(set(flat))==10,"perfect matching")
 for arm in ARMS:
  e=FeedbackEnvironmentProvider(arm=arm,seed=17); d=FeedbackDecisionModel(); ids=list(e.groups)
  for t in range(1,4):
   e.advance_to(t); acts=d.decide_batch(e.observe_batch(ids,t,0),{}); e.apply_allocations(acts,t)
   for r in (1,2): e.apply_messages(d.decide_batch(e.observe_batch(ids,t,r),{}),t,r)
  check(all(len([x for x in h if x['kind']=='allocation'])==3 for h in e.history.values()),f"{arm} allocations history")
  check(len(e.messages)==3*10*2,f"{arm} message count")
  check(all(len(x['content'])<=200 for x in e.messages),"message limit")
  if arm=='d1': check(all(x['has_cross'] for x in e.announcements),"D1 complete announcements")
  if arm=='d3': check(all(x['is_sum_only'] and not any(z in x['text'].lower() for z in ('group a','group b','ingroup','outgroup')) for x in e.announcements),"D3 sum-only")
  if arm in ('d2','d4'): check(not e.announcements,"no-announcement arm")
  obs=e.observe_batch(ids,3,0)
  # memories argument is a functional source if an Observation omits local history.
  bare=obs[0]; bare.local_physical=dict(bare.local_physical); bare.local_physical["personal_history"]=[]
  mem={bare.agent_id:{"personal_history":[{"kind":"allocation","step":1,"row":0,"gap":10}]}}
  mem_act=FeedbackDecisionModel().decide_batch([bare],mem)
  check(len(mem_act)==1,"framework memories consumed")
  if arm in ('d3','d4'):
   for o in obs:
    for r in (0,1,2): _=build_feedback_prompt(o,r)
  # each pair has two messages in each communication round
  for t in range(1,4):
   for a,b,k in e.pairings[t]:
    for r in (1,2): check(sum(m['step']==t and m['round']==r and m['sender'] in (a,b) and m['recipient'] in (a,b) for m in e.messages)==2,"bidirectional chat")
 with tempfile.TemporaryDirectory() as d:
  for arm in ARMS:
   env,met=run_arm(arm,5,9,d); check(len(met)==5 and all(x['fallback_rate']==0 for x in met),"end-to-end metrics")
   check((Path(d)/arm/'export'/'messages.csv').exists(),"messages export")
  # repeat determinism
  d1=Path(d)/'repeat1'; d2=Path(d)/'repeat2'; run_arm('d1',5,33,d1); run_arm('d1',5,33,d2)
  x=(d1/'d1/export/allocations.csv').read_bytes(); y=(d2/'d1/export/allocations.csv').read_bytes(); check(x==y,"allocation export reproducibility")
 print('all local checks passed'); return 0
if __name__=='__main__': raise SystemExit(main())
