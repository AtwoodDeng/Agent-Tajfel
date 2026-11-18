"""D-series Tajfel feedback and communication study.

This module is intentionally study-layer code: it keeps the classic C1-C3 allocator untouched and
adds an explicit personal-history channel for the multi-round feedback manipulation.
"""
from __future__ import annotations
import csv, hashlib, json, random, re, sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

try:
    from socioverse.engine import register
    from socioverse.schemas import Action, Observation
except Exception:  # Windows selftest imports this module with framework stubs
    def register(_kind, _name):
        return lambda cls: cls
    Action = SimpleNamespace
    Observation = SimpleNamespace

try:
    from tajfel.model import MATRIX_MIF, agent_id, agent_index, group_of, group_sizes, other_group
except Exception:
    MATRIX_MIF=[(15,5),(14,8),(13,11),(12,14),(11,17),(10,20)]
    def agent_id(i): return f"tj-{i:03d}"
    def agent_index(a): return int(str(a).split("-")[-1])
    def group_of(i,n,condition="c1"): return "A" if i < n//2 else "B"
    def other_group(g): return "B" if g=="A" else "A"
    def group_sizes(n,condition): return (n//2,n-n//2)

ARMS=("d1","d2","d3","d4")
N_AGENTS=10
STEPS=5

def _groups(n=10): return {agent_id(i): group_of(i,n,"c1") for i in range(n)}

def _mk_action(**kw):
    try: return Action(**kw)
    except TypeError: return SimpleNamespace(**kw)

def _history_text(history):
    return json.dumps(history, ensure_ascii=False, sort_keys=True)

def _leak_check(prompt, hidden=True):
    if hidden and re.search(r"your group|ingroup|outgroup|group\s+[ab]", prompt, re.I):
        raise AssertionError("hidden-arm prompt leaks group identity")
    return prompt

def build_feedback_prompt(ob, round_idx=None):
    lp=ob.local_physical or {}; mp=ob.macro_physical or {}; arm=mp.get("arm", "d1")
    r=lp.get("round_idx", 0) if round_idx is None else round_idx
    hist=lp.get("personal_history", []) or []
    rows=mp.get("rows", [])
    lines=["You are a participant in a multi-round allocation and communication study."]
    visible=bool(mp.get("label_visible", arm in ("d1","d2")))
    if r==0:
        lines.append("Choose exactly one row of the matrix.")
        for row in rows:
            a,b=row["a_points"],row["b_points"]
            if visible:
                own,oth=(a,b) if lp.get("own_group")=="A" else (b,a)
                lines.append(f"row {row['row']}: ingroup {own} outgroup {oth}")
            else: lines.append(f"row {row['row']}: Person A {a} Person B {b}")
    else:
        lines.append("Write one Chinese message of at most 200 characters.")
        partner=lp.get("partner_id")
        if partner: lines.append(f"Your partner is participant {partner}.")
        if lp.get("partner_message"): lines.append("Partner's message: "+str(lp["partner_message"]))
    if hist:
        lines.append("Your past choices:")
        for h in hist:
            if h.get("kind")=="allocation": lines.append(f"step {h['step']}: row {h['row']}, gap {h['gap']}")
        pubs=[h for h in hist if h.get("kind")=="announcement"]
        if pubs:
            lines.append("Published results so far:")
            lines.extend(str(h["text"]) for h in pubs)
        msgs=[h for h in hist if h.get("kind")=="message"]
        if msgs:
            lines.append("Your conversations:")
            lines.extend(f"step {h['step']} {h['direction']} {h['partner_id']}: {h['content']}" for h in msgs)
    prompt="\n".join(lines)
    return _leak_check(prompt, not visible)

@register("environment", "tajfel_feedback.env")
class FeedbackEnvironmentProvider:
    def __init__(self, bundle=None, n_agents=10, arm="d1", matrices=None, seed=0, **kw):
        args=getattr(bundle,"provider_args",{}) if bundle is not None else {}
        self.n=int(args.get("n_agents",n_agents)); self.arm=str(args.get("arm",arm))
        self.matrices=list(args.get("matrices",matrices or ["mif"]))
        self.seed=int(args.get("seed",seed)); self.groups=_groups(self.n)
        self.label_visible=self.arm in ("d1","d2")
        self.history={agent_id(i):[] for i in range(self.n)}
        self.pairings={}; self.messages=[]; self.allocations=[]; self.scores=[]; self.announcements=[]
        self._pending_partner={}
    def reset(self, seed=0): self.seed=int(seed); self.history={a:[] for a in self.groups}; self.pairings={}; self.messages=[]; self.allocations=[]; self.scores=[]; self.announcements=[]
    def matrix_at(self,t): return self.matrices[(max(1,t)-1)%len(self.matrices)]
    def advance_to(self,t):
        ids=list(self.groups); rng=random.Random((self.seed*1000003+t*9176)&0xffffffff); A=[a for a in ids if self.groups[a]=="A"]; B=[a for a in ids if self.groups[a]=="B"]; rng.shuffle(A); rng.shuffle(B)
        pairs=[(A[0],A[1],"AA"),(B[0],B[1],"BB")]+[(A[i],B[i],"AB") for i in range(2,5)]
        rng.shuffle(pairs); self.pairings[t]=[(a,b,k) for a,b,k in pairs]
        self._pending_partner={x:(y,k) for x,y,k in pairs for y,x in [(x,y)]} if False else {}
        for a,b,k in pairs: self._pending_partner[a]=(b,k); self._pending_partner[b]=(a,k)
        return []
    def _announcement(self,t):
        if not self.scores: return ""
        s=self.scores[-1]
        if self.arm=="d1": return (f"Previous results: group A total {s['group_a_received']}; group B total {s['group_b_received']}; " f"A-to-A {s['a_to_a']}; A-to-B {s['a_to_b']}; B-to-A {s['b_to_a']}; B-to-B {s['b_to_b']}.")
        if self.arm=="d3": return f"Previous results: total points received by the two groups combined were {s['total_sum']}."
        return ""
    def observe_batch(self, agent_ids, t, round_idx=0):
        if t not in self.pairings: self.advance_to(t)
        ann=self._announcement(t)
        if round_idx == 0 and ann and not any(x["step"] == t for x in self.announcements):
            self.announcements.append({"step":t,"arm":self.arm,"text":ann,"is_sum_only":self.arm=="d3","has_cross":self.arm=="d1"})
            for aid in self.history: self.history[aid].append({"kind":"announcement","step":t,"text":ann})
        out=[]; rows=[{"row":i,"a_points":a,"b_points":b} for i,(a,b) in enumerate(MATRIX_MIF)]
        for aid in agent_ids:
            partner,ptype=self._pending_partner.get(aid,(None,"unknown")); partner_msg=""
            if round_idx>=2:
                vals=[m for m in self.messages if m["step"]==t and m["round"]==1 and m["sender"]==partner and m["recipient"]==aid]
                partner_msg=vals[-1]["content"] if vals else ""
            lp={"own_group":self.groups[aid] if self.label_visible else None,"partner_id":partner,"pair_type":ptype,"round_idx":round_idx,"personal_history":list(self.history[aid]),"partner_message":partner_msg}
            mp={"arm":self.arm,"matrix_kind":"mif","label_visible":self.label_visible,"rows":rows,"announcement":ann}
            out.append(_mk_action(agent_id=aid,step=t,kind="observation",payload={}) if False else SimpleNamespace(agent_id=aid,step=t,macro_physical=mp,macro_information={"announcement":ann},local_physical=lp))
        return out
    def apply_allocations(self, actions,t):
        by_group={"A":[],"B":[]}; cross={"AA":0,"AB":0,"BA":0,"BB":0}
        for a in actions:
            p=a.payload; g=self.groups[a.agent_id]; rec={"kind":"allocation","step":t,"agent_id":a.agent_id,"row":int(p["row"]),"own_group":g,"ingroup_points":int(p["ingroup_points"]),"outgroup_points":int(p["outgroup_points"]),"gap":int(p["gap"])}; self.history[a.agent_id].append(rec); self.allocations.append(rec); by_group[g].append(rec)
            if g=="A": cross["AA"]+=rec["ingroup_points"]; cross["AB"]+=rec["outgroup_points"]
            else: cross["BB"]+=rec["ingroup_points"]; cross["BA"]+=rec["outgroup_points"]
        s={"step":t,"group_a_received":cross["AA"]+cross["BA"],"group_b_received":cross["AB"]+cross["BB"],**{f"a_to_a":cross["AA"],"a_to_b":cross["AB"],"b_to_a":cross["BA"],"b_to_b":cross["BB"]},"total_sum":sum(cross.values())}; self.scores.append(s)
    def apply_messages(self, actions,t,round_idx):
        for a in actions:
            p=a.payload; content=str(p.get("content", "")); trunc=len(content)>200; content=content[:200]; partner,ptype=self._pending_partner.get(a.agent_id,(p.get("recipient"),"unknown")); rec={"step":t,"round":round_idx,"sender":a.agent_id,"recipient":partner,"pair_type":ptype,"content":content,"truncated":trunc,"source":getattr(a,"source","scripted")}; self.messages.append(rec); self.history[a.agent_id].append({"kind":"message","step":t,"direction":"sent","partner_id":partner,"content":content}); self.history[partner].append({"kind":"message","step":t,"direction":"received","partner_id":a.agent_id,"content":content})

@register("decision", "tajfel_feedback.decision")
class FeedbackDecisionModel:
    def __init__(self,llm_kind="scripted",seed=0,**kw): self.llm_kind=llm_kind; self.seed=seed
    def _row(self,ob):
        hist=ob.local_physical.get("personal_history",[]); h=hashlib.sha256((ob.agent_id+str(ob.step)+_history_text(hist)).encode()).digest(); return int.from_bytes(h[:4],"big")%len(ob.macro_physical["rows"])
    def decide_batch(self,obs,memories=None):
        out=[]
        memories = memories or {}
        for ob in obs:
            # Observation is the primary source; the framework memory map is the durable fallback.
            lp = dict(ob.local_physical or {})
            if not lp.get("personal_history"):
                mem = memories.get(ob.agent_id, {}) if isinstance(memories, dict) else {}
                if isinstance(mem, dict):
                    lp["personal_history"] = mem.get("personal_history", mem.get("history", []))
            ob_for_prompt = SimpleNamespace(agent_id=ob.agent_id,step=ob.step,
                macro_physical=ob.macro_physical,macro_information=ob.macro_information,
                local_physical=lp)
            r=lp.get("round_idx",0); prompt=build_feedback_prompt(ob_for_prompt,r); partner=lp.get("partner_id")
            if r==0:
                row=self._row(ob); a,b=MATRIX_MIF[row]; own,other=(a,b) if lp.get("own_group")=="A" else (b,a); out.append(_mk_action(agent_id=ob.agent_id,step=ob.step,kind="allocate_in_group_matrix",payload={"row":row,"ingroup_points":own,"outgroup_points":other,"gap":own-other,"matrix":"mif"},source="rule",rationale=["history-sensitive scripted choice"]))
            else:
                h=hashlib.sha256((ob.agent_id+str(ob.step)+str(r)+prompt).encode()).hexdigest(); content=["我觉得上一轮的分配值得继续讨论。","我们可以比较一下刚才的结果。","我重视公平，也会参考之前的交流。"][int(h[:2],16)%3]; out.append(_mk_action(agent_id=ob.agent_id,step=ob.step,kind="chat",payload={"content":content,"recipient":partner},source="rule",rationale=["scripted chat"]))
        return out

@register("collector", "tajfel_feedback.collector")
class FeedbackMetricCollector:
    def collect(self,env,actions,t):
        aa=[x for x in env.allocations if x["step"]==t]
        msgs=[x for x in env.messages if x["step"]==t]
        gaps=[x["gap"] for x in aa]
        pair_type={agent: kind for a,b,kind in env.pairings.get(t,[]) for agent in (a,b)}
        def avg(xs): return round(sum(xs)/len(xs),4) if xs else None
        return {"step":t,"mean_gap":avg(gaps),
                "mif_favour_share":round(sum(x>0 for x in gaps)/len(gaps),4) if gaps else None,
                "n_messages":len(msgs),"mean_message_len":avg([len(x["content"]) for x in msgs]),
                "fallback_rate":0.0,"n_allocations":len(aa),
                "mean_gap_samepair":avg([x["gap"] for x in aa if pair_type.get(x["agent_id"]) in ("AA","BB")]),
                "mean_gap_crosspair":avg([x["gap"] for x in aa if pair_type.get(x["agent_id"])=="AB"])}

def run_arm(arm="d1",steps=5,seed=0,out="runs"):
    env=FeedbackEnvironmentProvider(n_agents=10,arm=arm,seed=seed); dec=FeedbackDecisionModel(seed=seed); col=FeedbackMetricCollector(); env.reset(seed); metrics=[]
    for t in range(1,steps+1):
        env.advance_to(t); a=dec.decide_batch(env.observe_batch(list(env.groups),t,0),{}); env.apply_allocations(a,t)
        for r in (1,2): env.apply_messages(dec.decide_batch(env.observe_batch(list(env.groups),t,r),{}),t,r)
        metrics.append(col.collect(env,[],t))
    d=Path(out)/arm/"export"; d.mkdir(parents=True,exist_ok=True)
    def dump(name,rows,fields):
        with (d/name).open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    dump("allocations.csv",env.allocations,["step","agent_id","row","own_group","gap","ingroup_points","outgroup_points"]); dump("messages.csv",env.messages,["step","round","sender","recipient","pair_type","content","truncated","source"]); dump("scores.csv",env.scores,["step","group_a_received","group_b_received","a_to_a","a_to_b","b_to_a","b_to_b","total_sum"]); dump("announcements.csv",env.announcements,["step","arm","text","is_sum_only","has_cross"]); dump("pairings.csv",[{"step":t,"agent_a":a,"agent_b":b,"pair_type":k,"round_rotation":t%2} for t,ps in env.pairings.items() for a,b,k in ps],["step","agent_a","agent_b","pair_type","round_rotation"]); dump("metrics.csv",metrics,list(metrics[0]) if metrics else ["step"])
    return env,metrics
