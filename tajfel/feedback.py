"""D-series Tajfel feedback and communication study.

This module is intentionally study-layer code: it keeps the classic C1-C3 allocator untouched and
adds an explicit personal-history channel for the multi-round feedback manipulation.
"""
from __future__ import annotations
import csv, hashlib, json, os, random, re, sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

try:
    from socioverse.engine import register
    from socioverse.abc import DecisionModel, EnvironmentProvider, MetricCollector, PopulationProvider
    from socioverse.schemas import Action, Observation, Persona
except Exception:  # Windows selftest imports this module with framework stubs
    def register(_kind, _name):
        return lambda cls: cls
    Action = SimpleNamespace
    Observation = SimpleNamespace
    DecisionModel = EnvironmentProvider = MetricCollector = PopulationProvider = object
    Persona = SimpleNamespace

try:
    from tajfel.model import MATRIX_MIF, GROUP_NAMES, matrix_for, agent_id, agent_index, group_of, group_sizes, other_group, is_llm_kind
except Exception:
    MATRIX_MIF=[(15,5),(14,8),(13,11),(12,14),(11,17),(10,20)]
    GROUP_NAMES={"A":"BLOTQW","B":"BTSLAB"}
    def matrix_for(kind): return MATRIX_MIF
    def agent_id(i): return f"tj-{i:03d}"
    def agent_index(a): return int(str(a).split("-")[-1])
    def group_of(i,n,condition="c1"): return "A" if i < n//2 else "B"
    def other_group(g): return "B" if g=="A" else "A"
    def group_sizes(n,condition): return (n//2,n-n//2)
    def is_llm_kind(kind): return kind == "openai"

try:
    from socioverse.engine.messaging import InMemoryMessageBus
except Exception:
    class InMemoryMessageBus:
        def __init__(self,carry_previous_step=True): self.carry_previous_step=carry_previous_step; self._messages=[]
        def post(self,message): self._messages.append(dict(message))
        def visible_to(self,recipient_id,neighbors,step,round_idx=0):
            nb=set(neighbors)
            return [m for m in self._messages if m.get("author_id") in nb and
                    ((m.get("step")==step and m.get("round",0)<round_idx) or
                     (self.carry_previous_step and m.get("step")==step-1))]
        def drain(self): return []

def participant_no(aid: str) -> int:
    """1-indexed participant number shown to agents as their name."""
    return agent_index(aid) + 1


ARMS=("d1","d2","d3","d4","d5","d6")
# d6 is the missing cell of the announcement x chat 2x2: labels visible (like d1/d2/d5) but no
# announcement and no chat. Without it D4 is the only "no announcement, no chat" arm and it hides
# the label, so the 2x2 interaction was confounded with label visibility.
CHATLESS_ARMS=("d5","d6")
N_AGENTS=10
STEPS=5

def _groups(n=10,seed=0):
    ids=[agent_id(i) for i in range(n)]
    random.Random((int(seed)*2654435761+7919)&0xffffffff).shuffle(ids)
    n_a=n//2
    return {aid:("A" if aid in set(ids[:n_a]) else "B") for aid in ids}

@register("population", "tajfel_feedback.population")
class FeedbackPopulationProvider(PopulationProvider):
    def __init__(self,bundle):
        self.bundle=bundle; self.n=int(bundle.provider_args.get("n_agents",10))
    def build(self,seed):
        groups=_groups(self.n,seed); n_a=sum(g=="A" for g in groups.values()); n_b=self.n-n_a
        return [Persona(agent_id=aid,group_key=groups[aid],
            attributes={"group":groups[aid],"pool_size":self.n,"group_size":n_a if groups[aid]=="A" else n_b,
                        "other_group_size":n_b if groups[aid]=="A" else n_a,"condition":"d_feedback"},
            init_state={"group":groups[aid],"steps_played":0,"cum_gap":0}) for aid in groups]
    def neighbors(self,agent_id_): return []

def _mk_action(**kw):
    try: return Action(**kw)
    except TypeError: return SimpleNamespace(**kw)

def _history_text(history):
    return json.dumps(history, ensure_ascii=False, sort_keys=True)

def _leak_check(prompt, hidden=True):
    if hidden and re.search(r"your group|ingroup|outgroup|group\s+[ab]|BLOTQW|BTSLAB", prompt, re.I):
        raise AssertionError("hidden-arm prompt leaks group identity")
    return prompt

def build_feedback_prompt(ob, round_idx=None):
    lp=ob.local_physical or {}; mp=ob.macro_physical or {}; arm=mp.get("arm", "d1")
    r=lp.get("round_idx", 0) if round_idx is None else round_idx
    hist=lp.get("personal_history", []) or []
    rows=mp.get("rows", [])
    visible=bool(mp.get("label_visible", arm in ("d1","d2","d5","d6")))
    own_group=lp.get("own_group") or "A"
    own_name=lp.get("own_name") or GROUP_NAMES.get(own_group, own_group)
    other_name=GROUP_NAMES.get(other_group(own_group), other_group(own_group))
    own_no=lp.get("own_no")
    lines=[f"You are Participant #{own_no} in a multi-round allocation and communication study."]
    if visible:
        lines.append(f"You belong to the {own_name} group.")
    if r==0:
        if visible:
            ing=lp.get("ingroup_recipient_no"); out=lp.get("outgroup_recipient_no")
            lines.append(f"You must allocate points to two specific participants: "
                         f"Participant #{ing} ({own_name}, your group) and "
                         f"Participant #{out} ({other_name}, other group).")
        lines.append('Choose exactly one row of the matrix. Reply as JSON only: {"row": 0, "reason": "one short sentence explaining why you chose this row"}')
        for row in rows:
            a,b=row["a_points"],row["b_points"]
            if visible:
                own,oth=(a,b) if own_group=="A" else (b,a)
                lines.append(f"row {row['row']}: {own_name} {own} {other_name} {oth}")
            else: lines.append(f"row {row['row']}: Person A {a} Person B {b}")
    else:
        lines.append("Write one Chinese message of at most 200 characters.")
        partner=lp.get("partner_id")
        if partner:
            pn=lp.get("partner_no"); pname=lp.get("partner_name","")
            if visible and pname:
                lines.append(f"Your partner is Participant #{pn} ({pname} group).")
            else:
                lines.append(f"Your partner is Participant #{pn}.")
        if lp.get("partner_message"): lines.append("Partner's message: "+str(lp["partner_message"]))
    if hist:
        lines.append("Your past choices:")
        for h in hist:
            if h.get("kind")=="allocation":
                past=f"step {h['step']}: row {h['row']}, gap {h['gap']}"
                if h.get("reason"): past+=f" ({h['reason']})"
                lines.append(past)
        pubs=[h for h in hist if h.get("kind")=="announcement"]
        if pubs:
            lines.append("Published results so far:")
            lines.extend(str(h["text"]) for h in pubs)
        msgs=[h for h in hist if h.get("kind")=="message"]
        if msgs:
            lines.append("Your conversations:")
            for h in msgs:
                pno=participant_no(h["partner_id"]) if h.get("partner_id") else "?"
                lines.append(f"step {h['step']} {h['direction']} Participant #{pno}: {h['content']}")
    prompt="\n".join(lines)
    return _leak_check(prompt, not visible)

@register("environment", "tajfel_feedback.env")
class FeedbackEnvironmentProvider(EnvironmentProvider):
    def __init__(self, bundle=None, n_agents=10, arm="d1", matrices=None, seed=0, **kw):
        args=getattr(bundle,"provider_args",{}) if bundle is not None else {}
        self.n=int(args.get("n_agents",n_agents)); self.arm=str(args.get("arm",arm))
        self.matrices=list(args.get("matrices",matrices or ["mif"]))
        self.seed=int(args.get("seed",seed)); self.groups=_groups(self.n,self.seed)
        self.label_visible=self.arm in ("d1","d2","d5","d6")
        self.history={agent_id(i):[] for i in range(self.n)}
        self.pairings={}; self.messages=[]; self.allocations=[]; self.scores=[]; self.announcements=[]
        self._pending_partner={}; self.fallback_count=0; self.call_count=0; self.call_events=[]
        self.message_bus=InMemoryMessageBus(carry_previous_step=True)
    def reset(self, seed=0): self.seed=int(seed); self.groups=_groups(self.n,self.seed); self.history={a:[] for a in self.groups}; self.pairings={}; self.messages=[]; self.allocations=[]; self.scores=[]; self.announcements=[]; self.fallback_count=0; self.call_count=0; self.call_events=[]; self._pending_partner={}; self.message_bus=InMemoryMessageBus(carry_previous_step=True)
    def matrix_at(self,t): return self.matrices[(max(1,t)-1)%len(self.matrices)]
    def advance_to(self,t):
        ids=list(self.groups); rng=random.Random((self.seed*1000003+t*9176)&0xffffffff)
        A=[a for a in ids if self.groups[a]=="A"]; B=[a for a in ids if self.groups[a]=="B"]
        rng.shuffle(A); rng.shuffle(B)
        # ~60% cross-group pairs (scaled from the original 3 AB : 1 AA : 1 BB), covering all agents
        n_ab=min(len(A),len(B))*3//5
        if (len(A)-n_ab)%2: n_ab-=1
        if (len(B)-n_ab)%2: n_ab-=1
        n_ab=max(0,n_ab)
        pairs=[(A[i],B[i],"AB") for i in range(n_ab)]
        pairs += [(A[i],A[i+1],"AA") for i in range(n_ab,len(A)-1,2)]
        pairs += [(B[i],B[i+1],"BB") for i in range(n_ab,len(B)-1,2)]
        rng.shuffle(pairs); self.pairings[t]=[(a,b,k) for a,b,k in pairs]
        self._pending_partner={}
        for a,b,k in pairs: self._pending_partner[a]=(b,k); self._pending_partner[b]=(a,k)
        return []
    def agent_state(self, agent_id_):
        records=[x for x in self.allocations if x["agent_id"]==agent_id_]
        return {"group":self.groups[agent_id_],"steps_played":len(records),
                "cum_gap":sum(x["gap"] for x in records),
                "last_gap":records[-1]["gap"] if records else None}
    def snapshot(self):
        return {"arm":self.arm,"n_allocations":len(self.allocations),
                "n_messages":len(self.messages)}
    def _announcement(self,t):
        if not self.scores: return ""
        s=self.scores[-1]
        if self.arm in ("d1","d5"): return (f"Previous results: group A total {s['group_a_received']}; group B total {s['group_b_received']}; " f"A-to-A {s['a_to_a']}; A-to-B {s['a_to_b']}; B-to-A {s['b_to_a']}; B-to-B {s['b_to_b']}.")
        if self.arm=="d3": return f"Previous results: total points received by the two groups combined were {s['total_sum']}."
        return ""
    def observe_batch(self, agent_ids, t, round_idx=0):
        if t not in self.pairings: self.advance_to(t)
        ann=self._announcement(t)
        if round_idx == 0 and ann and not any(x["step"] == t for x in self.announcements):
            self.announcements.append({"step":t,"arm":self.arm,"text":ann,"is_sum_only":self.arm=="d3","has_cross":self.arm in ("d1","d5")})
            for aid in self.history: self.history[aid].append({"kind":"announcement","step":t,"text":ann})
        out=[]; kind=self.matrix_at(t); rows=[{"row":i,"a_points":a,"b_points":b} for i,(a,b) in enumerate(matrix_for(kind))]
        for aid in agent_ids:
            g=self.groups[aid]
            partner,ptype=self._pending_partner.get(aid,(None,"unknown")); partner_msg=""
            if round_idx>=2 and partner:
                vals=self.message_bus.visible_to(aid,[partner],t,round_idx)
                vals=[m for m in vals if m.get("step")==t and m.get("round")==1]
                partner_msg=vals[-1].get("content","") if vals else ""
            same=sorted(a for a,gg in self.groups.items() if gg==g and a!=aid)
            other=sorted(a for a,gg in self.groups.items() if gg!=g)
            ing_rec=same[(t-1)%len(same)] if same else None
            out_rec=other[(t-1)%len(other)] if other else None
            if ptype in ("AA","BB") and partner: ing_rec=partner
            elif ptype=="AB" and partner: out_rec=partner
            lp={"own_group":g if self.label_visible else None,
                "own_no":participant_no(aid),
                "own_name":GROUP_NAMES.get(g,g),
                "partner_id":partner,
                "partner_no":participant_no(partner) if partner else None,
                "partner_name":GROUP_NAMES.get(self.groups.get(partner,""),"") if partner else None,
                "pair_type":ptype,
                "ingroup_recipient":ing_rec,
                "ingroup_recipient_no":participant_no(ing_rec) if ing_rec else None,
                "outgroup_recipient":out_rec,
                "outgroup_recipient_no":participant_no(out_rec) if out_rec else None,
                "round_idx":round_idx,
                "personal_history":list(self.history[aid]),
                "partner_message":partner_msg}
            mp={"arm":self.arm,"matrix_kind":kind,"label_visible":self.label_visible,"rows":rows,"announcement":ann}
            out.append(Observation(agent_id=aid,step=t,macro_physical=mp,macro_information={"announcement":ann},local_physical=lp))
        return out
    def apply(self, actions):
        if not actions: return
        for action in actions:
            self.call_count += 1
            is_fallback=getattr(action,"source",None)=="fallback"
            if is_fallback: self.fallback_count += 1
            self.call_events.append({"step":int(action.step),"fallback":is_fallback})
        t=int(actions[0].step)
        if actions[0].kind == "allocate_in_group_matrix": self.apply_allocations(actions,t)
        elif actions[0].kind == "chat": self.apply_messages(actions,t,int(actions[0].payload.get("round_idx",1)))
    def apply_allocations(self, actions,t):
        by_group={"A":[],"B":[]}; cross={"AA":0,"AB":0,"BA":0,"BB":0}
        for a in actions:
            p=a.payload; g=self.groups[a.agent_id]
            ap=int(p.get("a_points",p.get("ingroup_points",0))); bp=int(p.get("b_points",p.get("outgroup_points",0)))
            own,other=(ap,bp) if g=="A" else (bp,ap)
            ptype=next((kind for x,y,kind in self.pairings.get(t,[]) if a.agent_id in (x,y)),"unknown")
            rec={"kind":"allocation","arm":self.arm,"step":t,"agent_id":a.agent_id,"row":int(p["row"]),"matrix":str(p.get("matrix","mif")),"own_group":g,"pair_type":ptype,"reason":str(p.get("reason",""))[:200],"ingroup_points":own,"outgroup_points":other,"gap":own-other}; self.history[a.agent_id].append(rec); self.allocations.append(rec); by_group[g].append(rec)
            if g=="A": cross["AA"]+=rec["ingroup_points"]; cross["AB"]+=rec["outgroup_points"]
            else: cross["BB"]+=rec["ingroup_points"]; cross["BA"]+=rec["outgroup_points"]
        s={"step":t,"group_a_received":cross["AA"]+cross["BA"],"group_b_received":cross["AB"]+cross["BB"],**{f"a_to_a":cross["AA"],"a_to_b":cross["AB"],"b_to_a":cross["BA"],"b_to_b":cross["BB"]},"total_sum":sum(cross.values())}; self.scores.append(s)
    def apply_messages(self, actions,t,round_idx):
        for a in actions:
            p=a.payload; content=str(p.get("content", "")); trunc=len(content)>200; content=content[:200]; partner,ptype=self._pending_partner.get(a.agent_id,(p.get("recipient"),"unknown")); rec={"step":t,"round":round_idx,"sender":a.agent_id,"recipient":partner,"pair_type":ptype,"content":content,"truncated":trunc,"source":getattr(a,"source","scripted")}; self.messages.append(rec); self.history[a.agent_id].append({"kind":"message","step":t,"direction":"sent","partner_id":partner,"content":content}); self.history[partner].append({"kind":"message","step":t,"direction":"received","partner_id":a.agent_id,"content":content})

@register("decision", "tajfel_feedback.decision")
class FeedbackDecisionModel(DecisionModel):
    def __init__(self,llm_kind="scripted",seed=0,model=None,temperature=0.7,max_tokens=180,**kw):
        self.llm_kind=llm_kind; self.seed=seed; self.model=model; self.temperature=temperature
        self.max_tokens=max_tokens; self.fallback_count=0; self.call_count=0
        self._llm=None
    def _ensure_llm(self):
        if self._llm is None:
            from tajfel.model import _OpenAILLM
            self._llm=_OpenAILLM(model=self.model,temperature=self.temperature,max_tokens=self.max_tokens,provider=self.llm_kind)
        return self._llm
    def _call_llm(self,prompt):
        return self._ensure_llm()(prompt)
    def _scripted_message(self,ob,r,prompt):
        h=hashlib.sha256((ob.agent_id+str(ob.step)+str(r)+prompt).encode()).hexdigest()
        return ["我觉得上一轮的分配值得继续讨论。","我们可以比较一下刚才的结果。","我重视公平，也会参考之前的交流。"][int(h[:2],16)%3]
    def _row(self,ob):
        hist=ob.local_physical.get("personal_history",[]); h=hashlib.sha256((ob.agent_id+str(ob.step)+_history_text(hist)).encode()).digest(); return int.from_bytes(h[:4],"big")%len(ob.macro_physical["rows"])
    def _choose_one(self,ob,memories):
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
        source="rule"; row=None; content=""; reason=""
        if is_llm_kind(self.llm_kind):
            try:
                reply=str(self._call_llm(prompt) or "")
                if r==0:
                    match=re.search(r'"row"\s*:\s*(\d+)',reply)
                    row=int(match.group(1)) if match else -1
                    if not 0 <= row < len(ob.macro_physical["rows"]): raise ValueError("unparseable allocation")
                    rm=re.search(r'"reason"\s*:\s*"([^"]*)"',reply)
                    reason=rm.group(1)[:200] if rm else ""
                else:
                    match=re.search(r'"message"\s*:\s*"((?:[^"\\]|\\.)*)"',reply,re.S)
                    content=json.loads('"'+match.group(1)+'"') if match else reply.strip()
                    if not content: raise ValueError("empty message")
                source="llm"
            except Exception:
                source="fallback"
                if r==0: row=self._row(ob)
                else: content=self._scripted_message(ob,r,prompt)
        elif r==0: row=self._row(ob)
        else: content=self._scripted_message(ob,r,prompt)
        if r==0:
            if not is_llm_kind(self.llm_kind): row=self._row(ob_for_prompt)
            rr=ob.macro_physical["rows"][row]; a,b=int(rr["a_points"]),int(rr["b_points"])
            if lp.get("own_group") in ("A","B"):
                own,other=(a,b) if lp.get("own_group")=="A" else (b,a)
            else:
                own,other=a,b  # hidden-label arms do not reveal this orientation to the agent
            return _mk_action(agent_id=ob.agent_id,step=ob.step,kind="allocate_in_group_matrix",
                payload={"row":row,"a_points":a,"b_points":b,"ingroup_points":own,
                         "outgroup_points":other,"gap":own-other,"matrix":ob.macro_physical.get("matrix_kind","mif"),"reason":reason},
                source=source,rationale=[reason or "allocation choice"])
        return _mk_action(agent_id=ob.agent_id,step=ob.step,kind="chat",
            payload={"content":content,"recipient":partner,"round_idx":r},
            source=source,rationale=["chat message"])

    def decide_batch(self,obs,memories=None):
        memories = memories or {}
        if is_llm_kind(self.llm_kind) and len(obs) > 1:
            from concurrent.futures import ThreadPoolExecutor
            self._ensure_llm()  # create the shared client once, before worker threads race on it
            with ThreadPoolExecutor(max_workers=min(8, len(obs))) as ex:
                actions = list(ex.map(lambda o: self._choose_one(o, memories), obs))
        else:
            actions = [self._choose_one(o, memories) for o in obs]
        # Each agent makes exactly ONE LLM call per decide_batch in openai mode; count once here
        # instead of racing on a shared counter inside the worker threads.
        if is_llm_kind(self.llm_kind):
            self.call_count += len(obs)
            self.fallback_count += sum(1 for a in actions if getattr(a, "source", None) == "fallback")
        return actions

@register("collector", "tajfel_feedback.collector")
class FeedbackMetricCollector(MetricCollector):
    def collect(self,env,actions,t):
        aa=[x for x in env.allocations if x["step"]==t]
        msgs=[x for x in env.messages if x["step"]==t]
        gaps=[x["gap"] for x in aa]
        pair_type={agent: kind for a,b,kind in env.pairings.get(t,[]) for agent in (a,b)}
        def avg(xs): return round(sum(xs)/len(xs),4) if xs else None
        if aa:
            joints=[a+b for a,b in matrix_for(env.matrix_at(t))]; lo,hi=min(joints),max(joints)
            total=sum(int(x["ingroup_points"])+int(x["outgroup_points"]) for x in aa)
            group_welfare=round((total-lo*len(aa))/((hi-lo)*len(aa)),4) if hi>lo else None
        else:
            group_welfare=None
        return {"step":t,"mean_gap":avg(gaps),
                "mif_favour_share":round(sum(x>0 for x in gaps)/len(gaps),4) if gaps else None,
                "group_welfare":group_welfare,
                "n_messages":len(msgs),"mean_message_len":avg([len(x["content"]) for x in msgs]),
                "fallback_rate":round(sum(x["fallback"] for x in env.call_events if x["step"]==t) / sum(1 for x in env.call_events if x["step"]==t),4) if any(x["step"]==t for x in env.call_events) else 0.0,"n_allocations":len(aa),
                "mean_gap_samepair":avg([x["gap"] for x in aa if pair_type.get(x["agent_id"]) in ("AA","BB")]),
                "mean_gap_crosspair":avg([x["gap"] for x in aa if pair_type.get(x["agent_id"])=="AB"])}

def run_arm(arm="d1",steps=5,seed=0,out="runs"):
    env=FeedbackEnvironmentProvider(n_agents=10,arm=arm,seed=seed); dec=FeedbackDecisionModel(seed=seed); col=FeedbackMetricCollector(); env.reset(seed); metrics=[]
    chat_rounds = () if arm in CHATLESS_ARMS else (1, 2)
    for t in range(1,steps+1):
        env.advance_to(t); a=dec.decide_batch(env.observe_batch(list(env.groups),t,0),{}); env.apply(a)
        for r in chat_rounds: env.apply(dec.decide_batch(env.observe_batch(list(env.groups),t,r),{}))
        metrics.append(col.collect(env,[],t))
    d=Path(out)/arm/"export"; d.mkdir(parents=True,exist_ok=True)
    def dump(name,rows,fields):
        with (d/name).open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    dump("allocations.csv",env.allocations,["arm","step","agent_id","row","matrix","own_group","pair_type","reason","gap","ingroup_points","outgroup_points"]); dump("messages.csv",env.messages,["step","round","sender","recipient","pair_type","content","truncated","source"]); dump("scores.csv",env.scores,["step","group_a_received","group_b_received","a_to_a","a_to_b","b_to_a","b_to_b","total_sum"]); dump("announcements.csv",env.announcements,["step","arm","text","is_sum_only","has_cross"]); dump("pairings.csv",[{"step":t,"agent_a":a,"agent_b":b,"pair_type":k,"round_rotation":t%2} for t,ps in env.pairings.items() for a,b,k in ps],["step","agent_a","agent_b","pair_type","round_rotation"]); dump("metrics.csv",metrics,list(metrics[0]) if metrics else ["step"])
    return env,metrics
