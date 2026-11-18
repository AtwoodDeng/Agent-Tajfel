from __future__ import annotations
import argparse, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import importlib.util
_spec=importlib.util.spec_from_file_location("tajfel_feedback", ROOT / "tajfel" / "feedback.py")
_mod=importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_mod)
ARMS, run_arm = _mod.ARMS, _mod.run_arm

def main(argv=None):
 p=argparse.ArgumentParser(); p.add_argument('--arm',choices=[*ARMS,'all'],default='all'); p.add_argument('--llm',choices=['scripted','openai'],default='scripted'); p.add_argument('--steps',type=int,default=5); p.add_argument('--seed',type=int,default=0); p.add_argument('--out',default='runs'); a=p.parse_args(argv)
 # D-P0 scripted mode validates wiring. LLM routing is added in the confirmatory phase.
 arms=ARMS if a.arm=='all' else (a.arm,)
 for arm in arms:
  _,m=run_arm(arm,a.steps,a.seed,a.out); print(f"{arm}: final mean_gap={m[-1]['mean_gap'] if m else None}")
 return 0
if __name__=='__main__': raise SystemExit(main())
