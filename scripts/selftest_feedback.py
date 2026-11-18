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
  check(len({g for g in e.groups.values()})==2,"randomized groups contain both labels")
  check(any((int(a.split("-")[1]) % 2 == 0) != (g == "A") for a,g in e.groups.items()),"agent ID parity must not encode group")
  for t in range(1,6):
   e.advance_to(t); ps=e.pairings[t]; check(len(ps)==5,f"seed {seed} step {t}: five pairs")
   check(sum(p[2]=='AA' for p in ps)==1 and sum(p[2]=='BB' for p in ps)==1 and sum(p[2]=='AB' for p in ps)==3,"pair structure")
   flat=[x for a,b,_ in ps for x in (a,b)]; check(len(set(flat))==10,"perfect matching")
 for arm in ARMS:
  e=FeedbackEnvironmentProvider(arm=arm,seed=17); d=FeedbackDecisionModel(); ids=list(e.groups)
  chat_rounds = () if arm == "d5" else (1,2)
  for t in range(1,4):
   e.advance_to(t); acts=d.decide_batch(e.observe_batch(ids,t,0),{}); e.apply_allocations(acts,t)
   for r in chat_rounds: e.apply_messages(d.decide_batch(e.observe_batch(ids,t,r),{}),t,r)
  check(all(len([x for x in h if x['kind']=='allocation'])==3 for h in e.history.values()),f"{arm} allocations history")
  check(len(e.messages)==len(chat_rounds)*3*10,f"{arm} message count")
  check(all(len(x['content'])<=200 for x in e.messages),"message limit")
  if arm in ('d1','d5'): check(all(x['has_cross'] for x in e.announcements),"D1/D5 complete announcements")
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
    for r in chat_rounds: check(sum(m['step']==t and m['round']==r and m['sender'] in (a,b) and m['recipient'] in (a,b) for m in e.messages)==2,"bidirectional chat")
  # Oversized incoming LLM text is clipped and explicitly marked.
  tx=FeedbackEnvironmentProvider(arm="d1",seed=5); tx.advance_to(1)
  from types import SimpleNamespace as NS
  long_action=NS(agent_id="tj-000",step=1,kind="chat",payload={"content":"中"*240,"recipient":"tj-001","round_idx":1},source="llm")
  tx.apply_messages([long_action],1,1)
  check(len(tx.messages[0]["content"])==200 and tx.messages[0]["truncated"],"incoming text truncation flag")
  # The LLM mode calls the client and distinguishes valid replies from fallback.
  e=FeedbackEnvironmentProvider(arm="d1",seed=4); e.advance_to(1); o=e.observe_batch(["tj-000"],1,0)[0]
  llm=FeedbackDecisionModel(llm_kind="openai"); llm._call_llm=lambda prompt:'{"row": 2}'
  valid=llm.decide_batch([o],{})[0]; check(valid.source=="llm" and valid.payload["row"]==2,"OpenAI allocation parsing")
  llm._call_llm=lambda prompt:'not json'; bad=llm.decide_batch([o],{})[0]; check(bad.source=="fallback" and llm.fallback_count==1,"LLM fallback accounting")
  om=FeedbackEnvironmentProvider(arm="d4",seed=3); om.advance_to(1); chat=om.observe_batch(["tj-000"],1,1)[0]
  llm2=FeedbackDecisionModel(llm_kind="openai"); llm2._call_llm=lambda prompt:'{"message":"你好，我们继续讨论公平。"}'
  ca=llm2.decide_batch([chat],{})[0]; check(ca.source=="llm" and "你好" in ca.payload["content"],"OpenAI chat parsing")
 with tempfile.TemporaryDirectory() as d:
  for arm in ARMS:
   env,met=run_arm(arm,5,9,d); check(len(met)==5 and all(x['fallback_rate']==0 for x in met),"end-to-end metrics")
   check((Path(d)/arm/'export'/'messages.csv').exists(),"messages export")
   with (Path(d)/arm/'export'/'allocations.csv').open(encoding='utf-8',newline='') as f:
    hdr=next(csv.reader(f)); check("arm" in hdr and "pair_type" in hdr,"allocation export schema")
   with (Path(d)/arm/'export'/'metrics.csv').open(encoding='utf-8',newline='') as f:
    check(sum(1 for _ in csv.DictReader(f))==5,"metrics export step count")
  # repeat determinism
  d1=Path(d)/'repeat1'; d2=Path(d)/'repeat2'; run_arm('d1',5,33,d1); run_arm('d1',5,33,d2)
  x=(d1/'d1/export/allocations.csv').read_bytes(); y=(d2/'d1/export/allocations.csv').read_bytes(); check(x==y,"allocation export reproducibility")
 print('all local checks passed'); return 0
if __name__=='__main__': raise SystemExit(main())
