#!/usr/bin/env python3
"""Run the four-arm Tajfel feedback study through SocioVerse2's longitudinal engine."""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import tajfel  # registers the parent population provider
from tajfel.model import make_tajfel_bundles, LLM_PROVIDERS, is_llm_kind
import tajfel.feedback as feedback  # registers D-series providers
from socioverse.engine import build_simulator

def _dump(path, rows, fields):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); w.writeheader(); w.writerows(rows)
def run_one(arm,llm,steps,seed,out,model=None,n_agents=10,matrices=None):
 matrices=list(matrices or ["mif"])
 study, env_bundle, pop_bundle, sim=make_tajfel_bundles(study_id=f'tajfel_{arm}',condition='c1',n_agents=n_agents,n_steps=steps,seed=seed,matrices=matrices)
 env_bundle=env_bundle.model_copy(update={'provider_ref':'tajfel_feedback.env','provider_args':{'n_agents':n_agents,'arm':arm,'matrices':matrices,'seed':seed}})
 pop_bundle=pop_bundle.model_copy(update={'provider_ref':'tajfel_feedback.population','provider_args':{'n_agents':n_agents}})
 interaction_rounds = 1 if arm in feedback.CHATLESS_ARMS else 3
 sim=sim.model_copy(update={'decision_ref':'tajfel_feedback.decision','decision_args':{'llm_kind':llm,'seed':seed,**({'model':model} if model else {})},'collector_ref':'tajfel_feedback.collector','interaction_rounds':interaction_rounds,'engine_args':{'memory_window':max(steps*interaction_rounds,16)}})
 dest=Path(out)/arm; (dest/'trajectory').mkdir(parents=True,exist_ok=True)
 runner=build_simulator(env_bundle=env_bundle,pop_bundle=pop_bundle,sim_config=sim,store_path=dest/'trajectory'/'study.duckdb')
 history=runner.run(); env=runner.env; rows=[r for r in history.rows if r.get('step',0)>0]
 ex=dest/'export'
 _dump(ex/'allocations.csv',env.allocations,['arm','step','agent_id','row','matrix','own_group','pair_type','reason','gap','ingroup_points','outgroup_points'])
 _dump(ex/'messages.csv',env.messages,['step','round','sender','recipient','pair_type','content','truncated','source'])
 _dump(ex/'scores.csv',env.scores,['step','group_a_received','group_b_received','a_to_a','a_to_b','b_to_a','b_to_b','total_sum'])
 _dump(ex/'announcements.csv',env.announcements,['step','arm','text','is_sum_only','has_cross'])
 _dump(ex/'pairings.csv',[{'step':t,'agent_a':a,'agent_b':b,'pair_type':k,'round_rotation':t%2} for t,ps in env.pairings.items() for a,b,k in ps],['step','agent_a','agent_b','pair_type','round_rotation'])
 _dump(ex/'metrics.csv',rows,list(rows[0]) if rows else ['step'])
 return rows

def main(argv=None):
 p=argparse.ArgumentParser(); p.add_argument('--arm',choices=[*feedback.ARMS,'all'],default='all'); p.add_argument('--llm',choices=['scripted',*LLM_PROVIDERS],default='scripted'); p.add_argument('--steps',type=int,default=5); p.add_argument('--seed',type=int,default=42); p.add_argument('--out',default='runs_feedback'); p.add_argument('--model',default=None); p.add_argument('--n-agents',type=int,default=10); p.add_argument('--matrices',default=None,help='comma-separated matrix names to rotate, e.g. mip_sym,parity_sym (default: mif)'); a=p.parse_args(argv)
 mats=[m for m in (a.matrices.replace(' ','').split(',') if a.matrices else []) if m]
 arms=feedback.ARMS if a.arm=='all' else (a.arm,)
 for arm in arms:
  if is_llm_kind(a.llm): print(f'WARNING: real LLM run ({a.llm}): {arm}, {a.steps} steps, {a.n_agents} agents x {1 if arm in feedback.CHATLESS_ARMS else 3} rounds')
  rows=run_one(arm,a.llm,a.steps,a.seed,a.out,a.model,a.n_agents,mats or None); last=rows[-1] if rows else {}; print(f"{arm}: final mean_gap={last.get('mean_gap')}")
 return 0
if __name__=='__main__': raise SystemExit(main())
