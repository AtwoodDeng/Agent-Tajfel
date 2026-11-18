#!/usr/bin/env python3
"""Local validation of the E-series communication study (no framework, no LLM needed).

Run from the repository root:

    python scripts/selftest_communication.py

It checks the parts a real LLM run cannot check for us: who is allowed to read which message,
that routing cannot be talked around, that E3's one-way asymmetry really is one-way, that E4's
six stances map onto distinct rows, and that the exports are reproducible.
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# Windows does not ship the SocioVerse runtime; stub imports before the tajfel package init.
for name in ["socioverse", "socioverse.abc", "socioverse.engine", "socioverse.schemas"]:
    sys.modules[name] = types.ModuleType(name)


class Dummy:
    def __init__(self, *a, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


for n in ("DecisionModel", "EnvironmentProvider", "MetricCollector", "PopulationProvider"):
    setattr(sys.modules["socioverse.abc"], n, Dummy)
sys.modules["socioverse.engine"].register = lambda kind, key: (lambda cls: cls)
for n in ("Action", "Observation", "Broadcast", "EnvironmentBundle", "EnvironmentLayer",
          "InformationProgram", "InteractionStructure", "Persona", "PopulationBundle",
          "PropagationMode", "ScheduledEvent", "SimulationConfig", "StudySpec"):
    setattr(sys.modules["socioverse.schemas"], n, Dummy)
sys.path.insert(0, str(ROOT))

from tajfel.communication import (  # noqa: E402
    CHAT_SUBROUNDS, CONDITIONS, E2_CONDITIONS, E3_CONDITIONS, E4_CONDITIONS, E4_MATRIX,
    CommunicationDecisionModel, CommunicationEnvironmentProvider, CommunicationMetricCollector,
    build_allocation_prompt, build_chat_prompt, e4_targets, groups_for, leaders_for,
    run_condition, stance_of, target_rows, _parse_row,
)

CHECKS = {"n": 0}


def check(ok, msg):
    CHECKS["n"] += 1
    if not ok:
        raise AssertionError(msg)


def _fail(fn, msg):
    try:
        fn()
    except (PermissionError, ValueError):
        return True
    raise AssertionError(msg)


class _NS(types.SimpleNamespace):
    pass


def _act(agent, step, kind, payload, source="llm"):
    return _NS(agent_id=agent, step=step, kind=kind, payload=payload, source=source)


# --- 1. routing table --------------------------------------------------------------------------
def test_routing():
    for seed in range(6):
        groups = groups_for(10, seed)
        check(len(set(groups.values())) == 2, "both groups present")
        check(sum(g == "A" for g in groups.values()) == 5, "balanced groups")
        for aid, g in groups.items():
            check(g in ("A", "B"), "every agent has a group")
        # id parity must not encode the group
        by_parity = {}
        for aid, g in groups.items():
            by_parity.setdefault(int(aid.split("-")[1]) % 2, set()).add(g)
        check(any(len(v) == 2 for v in by_parity.values()), f"seed {seed}: id parity encodes group")

    # E2 pair: mutual, complete matching, D-compatible mix
    env = CommunicationEnvironmentProvider(condition="e2-pair", seed=3)
    for t in range(1, 6):
        env.advance_to(t)
        route = env._route_for(t, 1)
        pairs = env.pairings[t]
        flat = [x for p in pairs for x in p]
        check(len(flat) == 10 and len(set(flat)) == 10, f"step {t}: perfect matching")
        route = env._route_for(t, 1)
        for a, b in pairs:
            check(route.recipients(a) == [b] and route.recipients(b) == [a], "pair is mutual and private")

    # E2 ingroup / public / silent
    for cond, kind in (("e2-ingroup", "ingroup"), ("e2-public", "public"), ("e2-silent", "silent")):
        env = CommunicationEnvironmentProvider(condition=cond, seed=3)
        env.advance_to(1)
        route = env._route_for(1, 1)
        for aid, g in env.groups.items():
            if kind == "silent":
                check(not route.allows(aid), "silent has no senders")
                continue
            check(route.allows(aid), f"{cond}: every member posts once")
            rec = route.recipients(aid)
            check(aid not in rec, f"{cond}: never delivers to itself")
            if kind == "ingroup":
                check(all(env.groups[r] == g for r in rec) and len(rec) == 4,
                      f"{cond}: only the own group reads, 4 readers")
            else:
                check(len(rec) == 9 and set(rec) == set(env.groups) - {aid}, f"{cond}: 9 readers")

    # E3: every A speaks to exactly one B, every B is spoken to, nobody replies
    for seed in range(6):
        env = CommunicationEnvironmentProvider(condition="e3-a-to-b", seed=seed)
        for t in range(1, 6):
            env.advance_to(t)
            env._route_for(t, 1)
            pairs = env.pairings[t]
            senders = [a for a, _ in pairs]
            receivers = {b for _, b in pairs}
            check(all(env.groups[a] == "A" for a in senders), "E3 senders are all group A")
            check(all(env.groups[b] == "B" for b in receivers), "E3 receivers are all group B")
            check(len(receivers) == len(pairs) == 5, "E3 pairs are injective and complete")
            route = env._route_for(t, 1)
            for a, b in pairs:
                check(route.allows(a) and not route.allows(b), f"seed {seed}: B may not speak")
                check(route.recipients(a) == [b], "A reaches exactly its partner")

    # E4: one leader per group, leaders only, same leaders across all six arms at a given seed
    for seed in range(6):
        leaders = leaders_for(groups_for(10, seed), seed)
        check(set(leaders) == {"A", "B"} and leaders["A"] != leaders["B"], "two distinct leaders")
        for cond in E4_CONDITIONS:
            env = CommunicationEnvironmentProvider(condition=cond, seed=seed)
            check(env.leaders == leaders, f"{cond}: leaders must not vary with the channel/stance")
            env.advance_to(1)
            route = env._route_for(1, 1)
            check(sorted(route.senders) == sorted(leaders.values()), f"{cond}: only leaders speak")
            for aid in env.groups:
                if aid not in route.senders:
                    check(not route.allows(aid), f"{cond}: ordinary members never speak")

    # leaders are not a function of the group's first member (they must be drawn, not fixed)
    picks = {leaders_for(groups_for(10, s), s)["A"] for s in range(40)}
    check(len(picks) > 1, "leader draw is not degenerate")

    # E4 private stays inside the ingroup; E4 public reaches everyone
    pv = CommunicationEnvironmentProvider(condition="e4-private-welfare", seed=2)
    pv.advance_to(1)
    r_priv = pv._route_for(1, 1)
    for lead in r_priv.senders:
        g = pv.groups[lead]
        rec = r_priv.recipients(lead)
        check(len(rec) == 4 and all(pv.groups[r] == g for r in rec), "private reaches the 4 ingroup peers")
    pb = CommunicationEnvironmentProvider(condition="e4-public-welfare", seed=2)
    pb.advance_to(1)
    r_pub = pb._route_for(1, 1)
    for lead in r_pub.senders:
        check(len(r_pub.recipients(lead)) == 9, "public reaches all 9 others")


# --- 2. enforcement ----------------------------------------------------------------------------
def test_enforcement():
    env = CommunicationEnvironmentProvider(condition="e3-a-to-b", seed=1)
    env.advance_to(1)
    env._route_for(1, 1)
    pairs = dict(env.pairings[1])
    a = sorted(pairs)[0]
    b = pairs[a]
    # the receiver may not speak
    check(_fail(lambda: env.apply_messages([_act(b, 1, "chat", {"content": "hi", "round_idx": 1})], 1, 1),
                ""), "E3 receiver speaking must raise")
    # a may not address someone else
    other = next(x for x in env.groups if env.groups[x] == "B" and x != b)
    check(_fail(lambda: env.apply_messages([_act(a, 1, "chat", {"recipient": other, "content": "hi", "round_idx": 1})], 1, 1),
                ""), "E3 mis-addressed message must raise")
    # a may not send twice
    env.apply_messages([_act(a, 1, "chat", {"recipient": b, "content": "one", "round_idx": 1})], 1, 1)
    check(_fail(lambda: env.apply_messages([_act(a, 1, "chat", {"recipient": b, "content": "two", "round_idx": 1})], 1, 1),
                ""), "duplicate send must raise")
    # silence from a non-permitted sender is dropped, not delivered
    env.apply_messages([_act(b, 1, "chat", {"content": "", "round_idx": 1})], 1, 1)
    check(len(env.messages) == 1, "a blank action from a non-sender is not a message")
    check(len(env.deliveries) == 1 and env.deliveries[0]["recipient"] == b, "exactly one delivery")

    # silent condition refuses chat outright
    sil = CommunicationEnvironmentProvider(condition="e2-silent", seed=1)
    sil.advance_to(1)
    check(_fail(lambda: sil.apply_messages([_act("tj-000", 1, "chat", {"content": "x", "round_idx": 1})], 1, 1),
                ""), "silent condition must refuse chat")

    # E4: an ordinary member may not speak
    e4 = CommunicationEnvironmentProvider(condition="e4-private-equal", seed=4)
    e4.advance_to(1)
    member = next(a for a in e4.groups if a not in e4.leader_of)
    check(_fail(lambda: e4.apply_messages([_act(member, 1, "chat", {"content": "x", "round_idx": 1})], 1, 1),
                ""), "ordinary member must not speak in E4")

    # no chats after the final allocation
    last = CommunicationEnvironmentProvider(condition="e2-ingroup", seed=1, steps=3)
    check(_fail(lambda: last.observe_batch(sorted(last.groups), 3, 1), ""),
          "sub-round after the final step must raise")

    # oversized messages are clipped and flagged
    tx = CommunicationEnvironmentProvider(condition="e2-ingroup", seed=5)
    tx.advance_to(1)
    tx.apply_messages([_act("tj-000", 1, "chat", {"content": "x" * 240, "round_idx": 1})], 1, 1)
    check(len(tx.messages[0]["content"]) == 200 and tx.messages[0]["truncated"], "200-char cap + flag")

    # duplicate allocation for one step is refused
    dup = CommunicationEnvironmentProvider(condition="e2-ingroup", seed=5)
    dup.advance_to(1)
    dup.apply_allocations([_act("tj-000", 1, "allocate_in_group_matrix", {"row": 0})], 1)
    check(_fail(lambda: dup.apply_allocations([_act("tj-000", 1, "allocate_in_group_matrix", {"row": 1})], 1),
                ""), "duplicate allocation must raise")


# --- 3. privacy: no message ever reaches a non-recipient ---------------------------------------
def test_privacy():
    for cond in CONDITIONS:
        env, _ = run_condition(cond, steps=3, seed=8)
        allowed = {}
        for rec in env.messages:
            recipients = set(rec["recipients"].split(",")) if rec["recipients"] else set()
            allowed[rec["message_id"]] = recipients
        for d in env.deliveries:
            check(d["recipient"] in allowed[d["message_id"]], f"{cond}: delivery to a non-recipient")
        for aid, hist in env.history.items():
            for h in hist:
                if h.get("kind") != "message" or h.get("direction") != "received":
                    continue
                check(h["peer"] in env.groups, f"{cond}: received row from an unknown peer")
        # an agent's received rows must match exactly the deliveries addressed to it
        for aid in env.groups:
            got = sum(1 for h in env.history[aid]
                      if h.get("kind") == "message" and h.get("direction") == "received")
            want = sum(1 for d in env.deliveries if d["recipient"] == aid)
            check(got == want, f"{cond}: {aid} received rows ({got}) != deliveries ({want})")


# --- 4. E2 channel arithmetic ------------------------------------------------------------------
def test_e2_counts():
    steps = 3
    expect_per_subround = {"e2-pair": (10, 10), "e2-ingroup": (10, 40), "e2-public": (10, 90),
                           "e2-silent": (0, 0)}
    for cond, (msgs, deliv) in expect_per_subround.items():
        env, metrics = run_condition(cond, steps=steps, seed=4)
        chat_steps = steps - 1  # no chat after the final allocation
        n_sub = len(CHAT_SUBROUNDS)
        check(len(env.messages) == msgs * chat_steps * n_sub,
              f"{cond}: expected {msgs * chat_steps * n_sub} messages, got {len(env.messages)}")
        check(len(env.deliveries) == deliv * chat_steps * n_sub,
              f"{cond}: expected {deliv * chat_steps * n_sub} deliveries, got {len(env.deliveries)}")
        check(len(env.allocations) == 10 * steps, f"{cond}: one allocation per agent per step")
        check(all(m["fallback_rate"] == 0 for m in metrics), f"{cond}: scripted runs never fall back")


# --- 5. E3 one-way asymmetry -------------------------------------------------------------------
def test_e3_asymmetry():
    env, metrics = run_condition("e3-a-to-b", steps=4, seed=6)
    a_ids = {a for a, g in env.groups.items() if g == "A"}
    b_ids = {a for a, g in env.groups.items() if g == "B"}
    check({m["sender"] for m in env.messages} == a_ids, "only group A sends")
    check(all(len(m["recipients"].split(",")) == 1 for m in env.messages), "one recipient per message")
    check(len(env.messages) == 5 * 3 * 2, "5 A-members x 3 chat steps x 2 sub-rounds")
    check(len(env.deliveries) == len(env.messages), "one delivery per message")
    # B agents accumulate received messages and never send; both sides are told the rule
    for b in b_ids:
        recvd = [h for h in env.history[b] if h.get("kind") == "message"]
        check(recvd, "every B receives its partner's messages")
        check(all(h["direction"] == "received" for h in recvd), "B has no sent rows")
    env._route_for(3, 1)
    obs = env.observe_batch(sorted(env.groups), 3, 0)
    for o in obs:
        lp = o.local_physical
        check(lp["partner_id"] in env.groups, "both sides know their partner")
        check(lp["partner_group"] == ("B" if lp["own_group"] == "A" else "A"), "partner group known")
        check(lp["can_send"] == (lp["own_group"] == "A"), "send permission differs by group")
        check(lp["can_reply"] is False, "replies are off in the one-way arm")
    check(all(not m["fallback_rate"] for m in metrics), "E3 scripted run has no fallbacks")


# --- 6. E4 stances -----------------------------------------------------------------------------
def test_e4_stances():
    rows = {c: e4_targets(c) for c in E4_CONDITIONS}
    for stance in ("ingroup", "welfare", "equal"):
        check(rows[f"e4-private-{stance}"] == rows[f"e4-public-{stance}"],
              f"{stance}: the channel must not move the recommended row")
    check(rows["e4-private-ingroup"] != rows["e4-private-welfare"], "ingroup != welfare rows")
    check(rows["e4-private-equal"] == {"A": 4, "B": 4}, "equality row is the only zero-gap row")
    # ingroup and welfare disagree for BOTH groups -- that is why E4 needs its own matrix
    for g in ("A", "B"):
        check(rows["e4-private-ingroup"][g] != rows["e4-private-welfare"][g],
              f"group {g}: ingroup and welfare must not coincide")
    # the MIF matrix collapses them for group B, which is the documented reason for E4_MATRIX
    mif = [(15, 5), (14, 8), (13, 11), (12, 14), (11, 17), (10, 20)]
    check(target_rows(mif, "ingroup")[1] == target_rows(mif, "welfare")[1],
          "MIF is on record as collapsing B's favour and welfare rows")
    # E4 rows are the dedicated matrix and the allocations honour it
    env, metrics = run_condition("e4-private-ingroup", steps=3, seed=0)
    check(env.matrix == [tuple(r) for r in E4_MATRIX], "E4 uses the dedicated symmetric matrix")
    check(all((x["a_points"], x["b_points"]) in E4_MATRIX for x in env.allocations),
          "every allocation comes from the dedicated matrix")
    # allocations never carry the stance instruction, the speaking prompt does
    obs = env.observe_batch(sorted(env.groups), 3, 0)
    for o in obs:
        prompt = build_allocation_prompt(o)
        check("Your goal is to" not in prompt and "On this matrix that is row" not in prompt,
              "allocation prompt must not carry the leader's stance instruction")
        check(o.local_physical.get("commitment_target") is None,
              "allocation observation must not expose the target row")
    leaders = set(env.leaders.values())
    env.advance_to(1)
    chat_obs = env.observe_batch(sorted(leaders), 1, 1)
    for o in chat_obs:
        prompt = build_chat_prompt(o)
        check("commit to row" in prompt, "leader chat prompt carries the commitment instruction")
        check(str(o.local_physical["commitment_target"]) in prompt, "leader is told its target row")
    # ordinary members are never told a stance
    member = next(a for a in env.groups if a not in env.leader_of)
    mo = env.observe_batch([member], 1, 1)[0]
    check(mo.local_physical.get("commitment_target") is None, "members get no target row")

    # every message is a commitment with a resolved follow-through after the next allocation
    all_scored: list = []
    for cond in E4_CONDITIONS:
        env, _ = run_condition(cond, steps=4, seed=2)
        env2, _ = run_condition(cond, steps=4, seed=5)
        check(len(env.commitments) == len(env.messages), f"{cond}: every message is a commitment")
        check(all(m["stance"] == stance_of(cond) for m in env.commitments), f"{cond}: stance recorded")
        for e in (env, env2):
            all_scored.extend(e.commitments)
        check(all(c["target_step"] == c["step"] + 1 for c in env.commitments),
              f"{cond}: a commitment targets the following step")
        check(all(c["commitment_row"] is not None for c in env.commitments),
              f"{cond}: the scripted leader always states a row")
        check(all(env.messages[0].get("expected_row") is not None for _ in [0]), f"{cond}: expected row")
    check(all(c.get("fulfilled") is not None for c in all_scored),
          "every commitment across seeds resolves against the next step")
    check(any(c["fulfilled"] for c in all_scored) and not all(c["fulfilled"] for c in all_scored),
          "the follow-through column is not degenerate (scripted leaders keep and break promises)")


# --- 7. deterministic pairing / leaders --------------------------------------------------------
def test_determinism():
    a, _ = run_condition("e3-a-to-b", steps=3, seed=11)
    b, _ = run_condition("e3-a-to-b", steps=3, seed=11)
    check(a.pairings == b.pairings, "pairing reproducible for a seed")
    c, _ = run_condition("e3-a-to-b", steps=3, seed=12)
    check(a.pairings != c.pairings, "different seeds pair differently")
    check(a.leaders == b.leaders, "leader draw reproducible")
    la = [x["agent_id"] for x in a.allocations]
    lb = [x["agent_id"] for x in b.allocations]
    check(la == lb, "allocation order reproducible")


# --- 8. exports --------------------------------------------------------------------------------
def test_exports():
    # no TemporaryDirectory: the sandbox blocks its chmod-based cleanup on Windows
    base = ROOT / ".tmp-selftest"
    base.mkdir(exist_ok=True)
    dest = base / "export_check"
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    try:
        env, metrics = run_condition("e4-public-ingroup", steps=3, seed=9, out=dest / "run")
        for name in ("config.json", "roster.csv", "allocations.csv", "messages.csv", "deliveries.csv",
                     "metrics.csv", "announcements.csv", "scores.csv", "rejections.csv"):
            check((dest / "run" / name).exists(), f"export {name} missing")
        with (dest / "run" / "messages.csv").open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        check(len(rows) == 2 * 2 * 2, "2 leaders x 2 chat steps x 2 sub-rounds")
        check(all(int(r["n_recipients"]) == 9 for r in rows), "public message has 9 recipients")
        check("fulfilled" in rows[0], "commitment follow-through column exported")
        check(json.loads((dest / "run" / "config.json").read_text(encoding="utf-8"))["condition"]
              == "e4-public-ingroup", "config.json records the condition")
        check(sum(1 for _ in (dest / "run" / "metrics.csv").open(encoding="utf-8")) == 4,
              "metrics export has one header + one row per step")
        check(not env.rejections, "a clean run records no routing rejections")
    finally:
        shutil.rmtree(dest, ignore_errors=True)


# --- 9. row parsing ----------------------------------------------------------------------------
def test_row_parsing():
    n = len(E4_MATRIX)
    check(_parse_row("I commit to row 3 next round.", n) == 3, "row N is read as the label")
    check(_parse_row("第2行，我认为这最公平。", n) == 2, "Chinese 第2行 is read as the label")
    check(_parse_row("In step 2 I will pick row 4.", n) == 4, "the row reference beats the step number")
    check(_parse_row("no commitment here", n) is None, "a message without a row yields None")
    check(_parse_row("row 9", n) is None, "an out-of-range row yields None")
    check(_parse_row("row 1", n) == 1, "row 1 is unambiguous")


def main() -> int:
    for fn in (test_routing, test_enforcement, test_privacy, test_e2_counts, test_e3_asymmetry,
               test_e4_stances, test_determinism, test_exports, test_row_parsing):
        fn()
        print(f"  OK  {fn.__name__}")
    print(f"all local checks passed ({CHECKS['n']} assertions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
