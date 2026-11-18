"""E-series Tajfel communication study (E2 / E3 / E4).

Three independent communication *forms* on top of the D-series allocation task:

  E2  who hears whom      — pair private chat / ingroup channel / public broadcast / silent
  E3  one-way directed     — each A privately messages one randomly matched B; B cannot reply
  E4  leader commitment    — one leader per group, ingroup channel vs public bulletin x
                             (favour own group / maximise joint profit / maximise equality)

Design invariants shared by all three series:

  * Routing is decided by the ENVIRONMENT (`CommunicationEnvironmentProvider`), never by asking
    the model nicely. A message is delivered to exactly the recipients computed from the
    condition; `apply_messages` rejects out-of-scope senders, wrong recipients, duplicate sends
    and stale sub-rounds instead of silently accepting them.
  * Visibility never leaks through history. Each agent's `history[aid]` only ever receives
    messages it is entitled to read, and the chat prompt renders those rows verbatim — there is
    no shared log, summary or framework memory the model can read around the routing.
  * Within one chat sub-round every message is generated against a snapshot taken BEFORE any of
    that sub-round's messages exist, then all are delivered at once. Re-ordering the batch cannot
    change what any agent sees.
  * Allocation prompts never carry the leader's stance instruction; the stance only reaches the
    leader's *speaking* prompt, so a leader choosing the recommended row is the model's own
    compliance, not an instruction injected at decision time.
  * The D-series (`tajfel/feedback.py`) is untouched. These providers register under
    `tajfel_communication.*` and are only reached through `scripts/run_communication.py`.

Scripted mode is a routing/schema validator only; it is NOT evidence about real communication
effects. Confirmatory claims require `--llm openai` runs with a documented model and seed list.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

try:  # the Windows selftest imports this module with framework stubs installed
    from socioverse.engine import register
    from socioverse.abc import DecisionModel, EnvironmentProvider, MetricCollector, PopulationProvider
    from socioverse.schemas import Action, Observation, Persona
except Exception:  # pragma: no cover - stub path
    def register(_kind, _name):
        return lambda cls: cls

    Action = Observation = Persona = SimpleNamespace
    DecisionModel = EnvironmentProvider = MetricCollector = PopulationProvider = object

try:
    from tajfel.model import MATRIX_MIF
except Exception:  # pragma: no cover
    MATRIX_MIF = [(15, 5), (14, 8), (13, 11), (12, 14), (11, 17), (10, 20)]

# --- shared protocol defaults ------------------------------------------------------------------
N_AGENTS = 10
STEPS = 5
CHAT_SUBROUNDS = (1, 2)
MESSAGE_CHAR_LIMIT = 200
POPULATION_REF = "tajfel_communication.population"
ENV_REF = "tajfel_communication.env"
DECISION_REF = "tajfel_communication.decision"
COLLECTOR_REF = "tajfel_communication.collector"

# --- conditions --------------------------------------------------------------------------------
E2_CONDITIONS = ("e2-pair", "e2-ingroup", "e2-public", "e2-silent")
E3_CONDITIONS = ("e3-a-to-b",)
E4_CONDITIONS = (
    "e4-private-ingroup", "e4-private-welfare", "e4-private-equal",
    "e4-public-ingroup", "e4-public-welfare", "e4-public-equal",
)
CONDITIONS = E2_CONDITIONS + E3_CONDITIONS + E4_CONDITIONS
SERIES_OF = {**{c: "e2" for c in E2_CONDITIONS},
             **{c: "e3" for c in E3_CONDITIONS},
             **{c: "e4" for c in E4_CONDITIONS}}
CONDITION_LABELS = {
    "e2-pair": "mixed pair private chat (both directions), the D-style topology baseline",
    "e2-ingroup": "shared ingroup channel: each member posts once, only the own group reads it",
    "e2-public": "public bulletin: each member posts once, every other member reads it",
    "e2-silent": "no communication (supplementary baseline)",
    "e3-a-to-b": "one-way directed private message from each A member to one random B member; B cannot reply",
    "e4-private-ingroup": "ingroup channel, both leaders commit to favouring their own group",
    "e4-private-welfare": "ingroup channel, both leaders commit to maximising the joint total",
    "e4-private-equal": "ingroup channel, both leaders commit to minimising the A-B difference",
    "e4-public-ingroup": "public bulletin, both leaders commit to favouring their own group",
    "e4-public-welfare": "public bulletin, both leaders commit to maximising the joint total",
    "e4-public-equal": "public bulletin, both leaders commit to minimising the A-B difference",
}

# E4 uses its own symmetric matrix: on the D MIF table the B group's "favour own group" row and
# its "maximise joint" row coincide at (10, 20), which would collapse two conditions into one
# recommendation. Rows are (A points, B points).
E4_MATRIX: list[tuple[int, int]] = [(16, 4), (4, 16), (15, 13), (13, 15), (13, 13)]
E4_TIE_RULE = "maximise the joint total; if rows tie, prefer the row with more own-group points"


# --- small helpers -----------------------------------------------------------------------------
def agent_id(i: int) -> str:
    return f"tj-{i:03d}"


def agent_index(a: str) -> int:
    return int(str(a).split("-")[-1])


def other_group(g: str) -> str:
    return "B" if g == "A" else "A"


def condition_series(condition: str) -> str:
    return SERIES_OF[condition]


def _mk_action(**kw):
    try:
        return Action(**kw)
    except TypeError:
        return SimpleNamespace(**kw)


def groups_for(n: int = N_AGENTS, seed: int = 0) -> dict[str, str]:
    """Deterministic A/B split. Agent id parity never encodes the group (D-series invariant)."""
    ids = [agent_id(i) for i in range(n)]
    random.Random((int(seed) * 2654435761 + 7919) & 0xFFFFFFFF).shuffle(ids)
    n_a = n // 2
    a_set = set(ids[:n_a])
    return {aid: ("A" if aid in a_set else "B") for aid in ids}


def leaders_for(groups: dict[str, str], seed: int) -> dict[str, str]:
    """One leader per group, drawn from a stream independent of pairing and model randomness.

    The same (seed, roster) therefore yields the same two leaders in all six E4 arms, which is what
    makes the seed-paired channel comparison a within-leader comparison.
    """
    rng = random.Random((int(seed) * 7654321 + 104729) & 0xFFFFFFFF)
    picks: dict[str, str] = {}
    for g in ("A", "B"):
        members = sorted(a for a, gg in groups.items() if gg == g)
        if not members:
            raise ValueError(f"group {g} has no members; cannot draw a leader")
        picks[g] = rng.choice(members)
    return picks


def validate_condition(condition: str, n_agents: int) -> None:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}; choose from {CONDITIONS}")
    series = condition_series(condition)
    if n_agents % 2:
        raise ValueError(f"{series}: n_agents must be even so both groups are the same size, got {n_agents}")
    if n_agents < 4:
        raise ValueError(f"{series}: need at least 4 agents, got {n_agents}")


def matrix_for(condition: str) -> list[tuple[int, int]]:
    return list(E4_MATRIX) if condition_series(condition) == "e4" else list(MATRIX_MIF)


def stance_of(condition: str) -> str | None:
    if condition_series(condition) != "e4":
        return None
    return condition.rsplit("-", 1)[1]


def channel_of(condition: str) -> str | None:
    series = condition_series(condition)
    if series == "e4":
        return condition.split("-")[1]
    if series == "e2":
        return condition.split("-", 1)[1]
    return None


def target_rows(matrix: list[tuple[int, int]], stance: str) -> tuple[int, int]:
    """Return (row for group A, row for group B) implementing `stance` on `matrix`.

    ingroup : maximise own - other  (A wants the largest A-B row, B the largest B-A row)
    welfare : maximise A+B, ties broken toward the leader's own group (E4_TIE_RULE)
    equal   : minimise |A-B|, ties broken by the joint total then the own-group points
    """
    def key_ingroup(own: int):
        def k(i: int) -> tuple:
            a, b = matrix[i]
            me, you = (a, b) if own == 0 else (b, a)
            return (me - you, me, a + b)
        return k

    def key_welfare(own: int):
        def k(i: int) -> tuple:
            a, b = matrix[i]
            me = a if own == 0 else b
            return (a + b, me)
        return k

    def key_equal(own: int):
        def k(i: int) -> tuple:
            a, b = matrix[i]
            me = a if own == 0 else b
            return (-abs(a - b), a + b, me)
        return k

    keys = {"ingroup": key_ingroup, "welfare": key_welfare, "equal": key_equal}
    if stance not in keys:
        raise ValueError(f"unknown stance {stance!r}")
    return (max(range(len(matrix)), key=keys[stance](0)),
            max(range(len(matrix)), key=keys[stance](1)))


def e4_targets(condition: str, matrix: list[tuple[int, int]] | None = None) -> dict[str, int]:
    matrix = matrix or E4_MATRIX
    a_row, b_row = target_rows(matrix, stance_of(condition) or "ingroup")
    return {"A": a_row, "B": b_row}


# --- routing: the single source of truth for who may read what ---------------------------------
class Route:
    """Describes one chat sub-round: who may speak, and who reads each message."""

    __slots__ = ("kind", "senders", "recipients_of", "rejects_replies")

    def __init__(self, kind: str, senders: list[str], recipients_of, rejects_replies: bool = False):
        self.kind = kind
        self.senders = list(senders)
        self.recipients_of = recipients_of  # callable(sender) -> list[recipient]
        self.rejects_replies = rejects_replies

    def allows(self, sender: str) -> bool:
        return sender in self.senders

    def recipients(self, sender: str) -> list[str]:
        return [r for r in self.recipients_of(sender) if r != sender]


# --- population --------------------------------------------------------------------------------
@register("population", POPULATION_REF)
class CommunicationPopulationProvider(PopulationProvider):
    def __init__(self, bundle=None):
        args = getattr(bundle, "provider_args", {}) if bundle is not None else {}
        self.n = int(args.get("n_agents", N_AGENTS))

    def build(self, seed: int) -> list[Persona]:
        groups = groups_for(self.n, seed)
        out = []
        for aid, g in groups.items():
            out.append(Persona(
                agent_id=aid, group_key=g,
                attributes={"group": g, "pool_size": self.n,
                            "group_size": sum(1 for x in groups.values() if x == g),
                            "other_group_size": sum(1 for x in groups.values() if x != g),
                            "condition": "e_communication"},
                init_state={"group": g, "steps_played": 0, "cum_gap": 0}))
        return out

    def neighbors(self, agent_id_: str) -> list[str]:
        return []


# --- environment -------------------------------------------------------------------------------
class CommunicationEnvironmentProvider(EnvironmentProvider):
    """Shared E-series environment: allocation rounds interleaved with routed chat sub-rounds."""

    def __init__(self, bundle=None, condition: str = "e2-pair", n_agents: int = N_AGENTS,
                 seed: int = 0, steps: int = STEPS, matrix: list[tuple[int, int]] | None = None, **kw):
        args = getattr(bundle, "provider_args", {}) if bundle is not None else {}
        self.condition = str(args.get("condition", condition))
        self.n = int(args.get("n_agents", n_agents))
        self.seed = int(args.get("seed", seed))
        self.steps = int(args.get("steps", steps))
        validate_condition(self.condition, self.n)
        self.series = condition_series(self.condition)
        self.matrix = [tuple(r) for r in (args.get("matrix") or matrix or matrix_for(self.condition))]
        self.label_visible = True  # every E condition shows group labels and stable row numbers
        self.reset(self.seed)

    # -- state -----------------------------------------------------------------------------------
    def reset(self, seed: int = 0):
        self.seed = int(seed)
        self.groups = groups_for(self.n, self.seed)
        self.leaders = leaders_for(self.groups, self.seed) if self.series == "e4" else {}
        self.leader_of = {aid: g for g, aid in self.leaders.items()}
        self.targets = e4_targets(self.condition, self.matrix) if self.series == "e4" else {}
        self.history: dict[str, list[dict]] = {a: [] for a in self.groups}
        self.routes: dict[tuple[int, int], Route] = {}
        self.pairings: dict[int, list[tuple[str, str]]] = {}
        self.messages: list[dict] = []
        self.deliveries: list[dict] = []
        self.rejections: list[dict] = []
        self.commitments: list[dict] = []
        self.allocations: list[dict] = []
        self.scores: list[dict] = []
        self.announcements: list[dict] = []
        self.call_events: list[dict] = []
        self._current_step = 0
        self._built_steps: set[int] = set()

    # -- round bookkeeping -----------------------------------------------------------------------
    def matrix_at(self, t: int) -> str:
        return "e4_symmetric" if self.series == "e4" else "mif"

    def _route_for(self, t: int, r: int) -> Route:
        key = (t, r)
        if key in self.routes:
            return self.routes[key]
        route = self._build_route(t, r)
        self.routes[key] = route
        return route

    def _build_route(self, t: int, r: int) -> Route:
        ids = sorted(self.groups)
        by_group = {g: [a for a in ids if self.groups[a] == g] for g in ("A", "B")}
        if self.series == "e2":
            kind = channel_of(self.condition) or "pair"
            if kind == "pair":
                if t not in self.pairings:
                    rng = random.Random((self.seed * 1000003 + t * 9176) & 0xFFFFFFFF)
                    pairs = self._mixed_pairs(rng)
                    self.pairings[t] = pairs
                partner = {a: b for a, b in self.pairings[t]}
                partner.update({b: a for a, b in self.pairings[t]})
                return Route("pair", ids, lambda s: [partner[s]])
            if kind == "ingroup":
                return Route("ingroup", ids, lambda s: [a for a in by_group[self.groups[s]] if a != s])
            if kind == "public":
                return Route("public", ids, lambda s: [a for a in ids if a != s])
            if kind == "silent":
                return Route("silent", [], lambda s: [])
            raise ValueError(f"unknown E2 channel {kind!r}")
        if self.series == "e3":
            if t not in self.pairings:
                rng = random.Random((self.seed * 15485863 + t * 7919) & 0xFFFFFFFF)
                self.pairings[t] = self._directed_pairs(rng)
            receiver = {a: b for a, b in self.pairings[t]}
            return Route("directed", sorted(receiver), lambda s: [receiver[s]], rejects_replies=True)
        # e4: leaders speak, ordinary members only read
        kind = channel_of(self.condition) or "private"
        speakers = [self.leaders["A"], self.leaders["B"]]
        if kind == "private":
            return Route("private", speakers,
                         lambda s: [a for a in by_group[self.groups[s]] if a != s])
        return Route("public", speakers, lambda s: [a for a in ids if a != s])

    def _mixed_pairs(self, rng: random.Random) -> list[tuple[str, str]]:
        """D-compatible mixed pairing: ~60% cross-group, every agent in exactly one pair."""
        a = [x for x in self.groups if self.groups[x] == "A"]
        b = [x for x in self.groups if self.groups[x] == "B"]
        rng.shuffle(a)
        rng.shuffle(b)
        n_ab = min(len(a), len(b)) * 3 // 5
        if (len(a) - n_ab) % 2:
            n_ab -= 1
        if (len(b) - n_ab) % 2:
            n_ab -= 1
        n_ab = max(0, n_ab)
        pairs = [(a[i], b[i]) for i in range(n_ab)]
        pairs += [(a[i], a[i + 1]) for i in range(n_ab, len(a) - 1, 2)]
        pairs += [(b[i], b[i + 1]) for i in range(n_ab, len(b) - 1, 2)]
        rng.shuffle(pairs)
        return pairs

    def _directed_pairs(self, rng: random.Random) -> list[tuple[str, str]]:
        """E3: shuffle the B side, then match it one-to-one against the ordered A side."""
        a = sorted(x for x in self.groups if self.groups[x] == "A")
        b = sorted(x for x in self.groups if self.groups[x] == "B")
        if len(a) != len(b):
            raise ValueError("E3 requires equal group sizes")
        rng.shuffle(b)
        return list(zip(a, b))

    def advance_to(self, t: int) -> list:
        t = int(t)
        if t not in self._built_steps:
            self._built_steps.add(t)
        self._current_step = max(self._current_step, t)
        return []

    def resolve_commitment(self, target_step: int | None = None) -> None:
        """Score outstanding promises against the allocations now on record.

        Call after a step's allocations land. `advance_to` does this automatically for every step
        the caller actually visits, so this only matters for the final window.
        """
        t = self._current_step if target_step is None else int(target_step)
        for c in self.commitments:
            if c.get("target_step") != t or c.get("fulfilled") is not None:
                continue
            realised = next((x for x in self.allocations if x["step"] == t and x["agent_id"] == c["sender"]), None)
            c["realised_row"] = None if realised is None else int(realised["row"])
            if realised is None or c.get("commitment_row") is None:
                c["fulfilled"] = None
            else:
                c["fulfilled"] = int(realised["row"]) == int(c["commitment_row"])
            c["matched_expected"] = (None if realised is None or c.get("expected_row") is None
                                     else int(realised["row"]) == int(c["expected_row"]))

    # -- observation -----------------------------------------------------------------------------
    def _announcement_text(self) -> str:
        if not self.scores:
            return ""
        s = self.scores[-1]
        return (f"Previous results: group A total {s['group_a_received']}; group B total "
                f"{s['group_b_received']}; A-to-A {s['AA']}; A-to-B {s['AB']}; "
                f"B-to-A {s['BA']}; B-to-B {s['BB']}.")

    def observe_batch(self, agent_ids, t: int, round_idx: int = 0) -> list:
        t = int(t)
        ann = self._announcement_text()
        if round_idx == 0 and ann and not any(x["step"] == t for x in self.announcements):
            rec = {"step": t, "condition": self.condition, "text": ann}
            self.announcements.append(rec)
            for aid in self.history:
                self.history[aid].append({"kind": "announcement", "step": t, "text": ann})
        if round_idx and t >= self.steps:
            raise ValueError("no chat sub-rounds are allowed after the final allocation")
        route = None if round_idx == 0 else self._route_for(t, round_idx)
        rows = [{"row": i, "a_points": a, "b_points": b} for i, (a, b) in enumerate(self.matrix)]
        out = []
        for aid in agent_ids:
            lp = {"own_group": self.groups[aid],
                  "own_leader": self.leaders.get(self.groups[aid]),
                  "is_leader": aid in self.leader_of,
                  "leader_of_group": self.leader_of.get(aid),
                  "round_idx": int(round_idx),
                  "personal_history": list(self.history[aid]),
                  "partner_id": None,
                  "partner_group": None,
                  "can_send": False,
                  "can_reply": False,
                  "message_limit": MESSAGE_CHAR_LIMIT,
                  "condition": self.condition,
                  "series": self.series}
            if route is not None:
                lp["can_send"] = route.allows(aid)
                lp["can_reply"] = False  # replies are a separate condition, never implicit
            desired = aid
            if self.series == "e3" and round_idx == 0 and t in self.pairings:
                pair = dict(self.pairings[t])
                rev = {v: k for k, v in pair.items()}
                if aid in pair:
                    lp["partner_id"], lp["partner_group"] = pair[aid], "B"
                    lp["can_send"] = True
                    lp["rule"] = ("You may send one private message to your partner. Your partner cannot "
                                  "send anything back to you. Your partner knows this rule too.")
                elif aid in rev:
                    lp["partner_id"], lp["partner_group"] = rev[aid], "A"
                    desired = rev[aid]  # a receiver is told who it is allowed to hear from
                    lp["rule"] = ("Your partner may send you one private message. You cannot send "
                                  "anything back to them. Your partner knows this rule too.")
            if self.series == "e4":
                lp["stance_instruction"] = None  # only the speaking prompt carries the stance
                if round_idx:
                    lp["commitment_target"] = self.targets.get(self.groups[aid]) if lp["is_leader"] else None
            mp = {"condition": self.condition, "series": self.series, "matrix_kind": self.matrix_at(t),
                  "label_visible": self.label_visible, "rows": rows, "announcement": ann,
                  "n_agents": self.n, "tie_rule": E4_TIE_RULE if self.series == "e4" else None,
                  "expected_senders": desired}
            out.append(Observation(agent_id=aid, step=t, macro_physical=mp,
                                   macro_information={"announcement": ann}, local_physical=lp))
        return out

    # -- allocation ------------------------------------------------------------------------------
    def apply(self, actions) -> None:
        if not actions:
            return
        for action in actions:
            self.call_events.append({"step": int(action.step), "kind": str(action.kind),
                                     "source": getattr(action, "source", "scripted"),
                                     "fallback": getattr(action, "source", None) == "fallback"})
        t = int(actions[0].step)
        if actions[0].kind == "allocate_in_group_matrix":
            self.apply_allocations(actions, t)
        elif actions[0].kind == "chat":
            self.apply_messages(actions, t, int(actions[0].payload.get("round_idx", 1)))

    def apply_allocations(self, actions, t: int) -> None:
        t = int(t)
        if any(x["step"] == t and x["agent_id"] == a.agent_id for a in actions for x in self.allocations):
            raise ValueError(f"duplicate allocation for step {t}")
        cross = {"AA": 0, "AB": 0, "BA": 0, "BB": 0}
        for a in actions:
            p = a.payload
            g = self.groups[a.agent_id]
            row = int(p["row"])
            if not 0 <= row < len(self.matrix):
                raise ValueError(f"row {row} outside the {len(self.matrix)}-row matrix")
            ap, bp = self.matrix[row]
            own, other = (ap, bp) if g == "A" else (bp, ap)
            rec = {"condition": self.condition, "series": self.series, "step": t, "agent_id": a.agent_id,
                   "own_group": g, "row": row, "a_points": ap, "b_points": bp,
                   "ingroup_points": own, "outgroup_points": other, "gap": own - other,
                   "is_leader": a.agent_id in self.leader_of,
                   "reason": str(p.get("reason", ""))[:200],
                   "source": getattr(a, "source", "scripted")}
            self.history[a.agent_id].append({"kind": "allocation", "step": t, "row": row,
                                             "gap": own - other, "reason": rec["reason"]})
            self.allocations.append(rec)
            if g == "A":
                cross["AA"] += own
                cross["AB"] += other
            else:
                cross["BB"] += own
                cross["BA"] += other
        self.scores.append({"step": t, "group_a_received": cross["AA"] + cross["BA"],
                            "group_b_received": cross["AB"] + cross["BB"], **cross,
                            "total_sum": sum(cross.values())})
        if self.series == "e4":
            self.resolve_commitment(t)  # promises made at t-1 become scoreable the moment t lands

    # -- chat ------------------------------------------------------------------------------------
    def apply_messages(self, actions, t: int, round_idx: int) -> None:
        t, round_idx = int(t), int(round_idx)
        route = self._route_for(t, round_idx)
        if self.condition == "e2-silent":
            raise ValueError("the silent condition must not accept any chat message")
        seen: set[str] = {m["sender"] for m in self.messages
                          if int(m["step"]) == t and int(m["round"]) == round_idx}
        for a in actions:
            sender = a.agent_id
            if not route.allows(sender):
                # The engine may hand the whole roster to a sub-round. Silence from an agent that
                # may not speak is expected, not a message: record it and drop it. An action that
                # is NOT silence is a routing violation and must fail loudly.
                if not str(a.payload.get("content", "")).strip():
                    self.rejections.append({"step": t, "round": round_idx, "sender": sender,
                                            "reason": "not a permitted sender (empty action ignored)"})
                    continue
                self.rejections.append({"step": t, "round": round_idx, "sender": sender,
                                        "reason": "sender is not allowed to speak in this condition"})
                raise PermissionError(f"{sender} may not speak in {self.condition} sub-round {round_idx}")
            if sender in seen:
                raise ValueError(f"{sender} sent more than one message in step {t} sub-round {round_idx}")
            seen.add(sender)
            recipients = route.recipients(sender)
            claimed = a.payload.get("recipient")
            if claimed not in (None, "", "public") and [claimed] != recipients:
                raise PermissionError(f"{sender} addressed {claimed!r} but the route allows {recipients}")
            text = str(a.payload.get("content", "")).strip()
            if not text:
                continue  # a blank message is not a message
            truncated = len(text) > MESSAGE_CHAR_LIMIT
            text = text[:MESSAGE_CHAR_LIMIT]
            message_id = f"m{t:03d}-{round_idx}-{sender}"
            rec = {"message_id": message_id, "condition": self.condition, "series": self.series,
                   "step": t, "round": round_idx, "sender": sender, "sender_group": self.groups[sender],
                   "channel": route.kind, "recipients": ",".join(recipients),
                   "n_recipients": len(recipients), "content": text,
                   "char_len": len(text), "truncated": truncated,
                   "source": getattr(a, "source", "scripted")}
            if self.series == "e4":
                committed = _parse_row(text, len(self.matrix))
                rec.update(stance=stance_of(self.condition),
                           issued_step=t, target_step=t + 1,
                           expected_row=self.targets.get(self.groups[sender]),
                           commitment_row1=(None if committed is None else committed + 1),
                           commitment_row=committed,
                           is_leader=True, leader_group=self.groups[sender])
            self.messages.append(rec)
            self.history[sender].append({"kind": "message", "step": t, "round": round_idx,
                                         "direction": "sent", "channel": route.kind,
                                         "peer": ", ".join(recipients), "content": text})
            for rid in recipients:
                self.deliveries.append({"message_id": message_id, "step": t, "round": round_idx,
                                        "recipient": rid, "recipient_group": self.groups[rid],
                                        "delivered_step": t})
                self.history[rid].append({"kind": "message", "step": t, "round": round_idx,
                                          "direction": "received", "channel": route.kind,
                                          "peer": sender, "peer_group": self.groups[sender],
                                          "is_leader": sender in self.leader_of,
                                          "content": text})
            if self.series == "e4":
                self.commitments.append(rec)

    def finalize(self) -> None:
        """Resolve the last commitment window; call once after the final allocation."""
        self.resolve_commitment(self.steps)

    def agent_state(self, agent_id_: str) -> dict:
        recs = [x for x in self.allocations if x["agent_id"] == agent_id_]
        return {"group": self.groups[agent_id_], "steps_played": len(recs),
                "cum_gap": sum(x["gap"] for x in recs),
                "last_gap": recs[-1]["gap"] if recs else None,
                "is_leader": agent_id_ in self.leader_of}

    def snapshot(self) -> dict:
        return {"condition": self.condition, "series": self.series, "n_allocations": len(self.allocations),
                "n_messages": len(self.messages), "n_deliveries": len(self.deliveries),
                "leaders": dict(self.leaders), "targets": dict(self.targets)}


def _parse_row(text: str, n_rows: int) -> int | None:
    """Extract the row a message explicitly commits to.

    Both conventions are accepted, because a model told "row 0..4" may reasonably answer "the
    2nd row". A numbered reference wins over a bare digit, since messages also contain step
    numbers. Returns a ZERO-BASED row, or None when the message names no usable row.
    """
    patterns = (r"(?:第|row|Row|rows)\s*#?\s*(\d+)",
                r"\b(\d+)\s*(?:st|nd|rd|th)\s+row\b",
                r"行\s*号?\s*[:：]?\s*(\d+)")
    for pat in patterns:
        for m in re.finditer(pat, text):
            raw = int(m.group(1))
            for candidate in _row_conventions(raw, n_rows):
                if candidate is not None:
                    return candidate
    return None


def _row_conventions(raw: int, n_rows: int) -> tuple[int | None, ...]:
    """Rows are labelled 0..n-1; a reply of 1..n is read as the 1-based ordinal when 0 is invalid."""
    zero_based = raw if 0 <= raw < n_rows else None
    one_based = raw - 1 if 1 <= raw <= n_rows else None
    if zero_based is None:
        return (one_based,)
    if one_based is None or raw == 0:
        return (zero_based,)
    # ambiguous mid-range value: both readings are legal, prefer the literal row label
    return (zero_based, one_based)


@register("environment", ENV_REF)
class _RegisteredCommunicationEnvironment(CommunicationEnvironmentProvider):
    pass


# --- prompts -----------------------------------------------------------------------------------
def _stance_text(stance: str, own: str, target_row: int) -> str:
    if stance == "ingroup":
        return (f"You strongly favour your own group ({own}). Your goal is to make the gap between "
                f"your group and the other group as large as possible in your group's favour, even "
                f"if that costs total points. On this matrix that is row {target_row}.")
    if stance == "welfare":
        return ("Your goal is to maximise the total points the two groups receive together, even if "
                f"that means your own group does not come out ahead. On this matrix that is row {target_row}.")
    return ("Your goal is to make the two groups come out as equal as possible, keeping the "
            f"difference between them as small as possible. On this matrix that is row {target_row}.")


def build_allocation_prompt(ob) -> str:
    """Allocation prompt. Identical across E2/E3/E4 apart from the matrix and the permission line."""
    lp = ob.local_physical or {}
    mp = ob.macro_physical or {}
    own = lp.get("own_group") or "A"
    rows = mp.get("rows", [])
    lines = ["You are a participant in a multi-round allocation study.",
             "Choose exactly one row of the matrix below.",
             'Reply as JSON only: {"row": 0, "reason": "one short sentence explaining why you chose this row"}']
    for r in rows:
        a, b = int(r["a_points"]), int(r["b_points"])
        mine, theirs = (a, b) if own == "A" else (b, a)
        lines.append(f"row {r['row']}: your group {mine}  other group {theirs}   (joint {a + b})")
    if lp.get("is_leader"):
        lines.append(f"You are the opinion leader of group {own}, but your own choice is still private.")
    lines.extend(_history_block(ob))
    return "\n".join(lines)


def build_chat_prompt(ob, round_idx: int | None = None) -> str:
    lp = ob.local_physical or {}
    mp = ob.macro_physical or {}
    r = int(lp.get("round_idx", 0) if round_idx is None else round_idx)
    lines = ["You are a participant in a multi-round allocation study."]
    if not lp.get("can_send"):
        lines.append("You do not speak in this round. You only read what others say.")
    if lp.get("own_group"):
        lines.append(f"You belong to group {lp['own_group']}.")
    if lp.get("partner_id"):
        lines.append(f"Your partner this round is {lp['partner_id']} (group {lp.get('partner_group')}).")
    if lp.get("rule"):
        lines.append(str(lp["rule"]))
    condition = mp.get("condition", "")
    if condition.startswith("e4") and lp.get("is_leader") and r:
        stance = stance_of(condition) or "equal"
        target = lp.get("commitment_target")
        own = lp.get("own_group") or "A"
        lines.append("You are the opinion leader of group " + str(own) + ".")
        lines.append(_stance_text(stance, str(own), int(target)))
        lines.append('Say clearly which row you commit to choosing in the next round: "I commit to row X next round", '
                     "then justify it in one sentence and invite your listeners to consider it.")
    else:
        lines.append(f"Write one message of at most {lp.get('message_limit', MESSAGE_CHAR_LIMIT)} characters.")
    lines.append('Reply as JSON only: {"message": "<your message>"}')
    lines.extend(_history_block(ob))
    return "\n".join(lines)


def _history_block(ob) -> list[str]:
    lp = ob.local_physical or {}
    hist = lp.get("personal_history") or []
    out: list[str] = []
    mine = [h for h in hist if h.get("kind") == "allocation"]
    if mine:
        out.append("Your past choices:")
        for h in mine:
            line = f"step {h['step']}: row {h['row']}, gap {h['gap']}"
            if h.get("reason"):
                line += f" ({h['reason']})"
            out.append(line)
    pubs = [h for h in hist if h.get("kind") == "announcement"]
    if pubs:
        out.append("Published results so far:")
        out.extend(str(h["text"]) for h in pubs)
    msgs = [h for h in hist if h.get("kind") == "message"]
    if msgs:
        out.append("Messages you sent or received:")
        for h in msgs:
            if h.get("direction") == "sent":
                out.append(f"step {h['step']} you -> {h.get('peer')}: {h['content']}")
            else:
                tag = " (leader)" if h.get("is_leader") else ""
                out.append(f"step {h['step']} group {h.get('peer_group')} {h.get('peer')}{tag} -> you: {h['content']}")
    return out


# --- decision ----------------------------------------------------------------------------------
@register("decision", DECISION_REF)
class CommunicationDecisionModel(DecisionModel):
    def __init__(self, llm_kind: str = "scripted", seed: int = 0, model: str | None = None,
                 temperature: float = 0.7, max_tokens: int = 220, **kw):
        self.llm_kind = llm_kind
        self.seed = int(seed)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.fallback_count = 0
        self.call_count = 0
        self._llm = None

    def _call_llm(self, prompt: str) -> str:
        if self._llm is None:
            from tajfel.model import _OpenAILLM
            self._llm = _OpenAILLM(model=self.model, temperature=self.temperature, max_tokens=self.max_tokens)
        return self._llm(prompt)

    # -- scripted helpers (validation only) ------------------------------------------------------
    def _scripted_row(self, ob) -> int:
        rows = ob.macro_physical["rows"]
        hist = ob.local_physical.get("personal_history") or []
        key = ob.agent_id + str(ob.step) + json.dumps(hist, ensure_ascii=False, sort_keys=True)
        return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big") % len(rows)

    def _scripted_message(self, ob) -> str:
        lp = ob.local_physical or {}
        key = ob.agent_id + str(ob.step) + str(lp.get("round_idx"))
        digest = hashlib.sha256(key.encode()).hexdigest()
        if lp.get("is_leader") and lp.get("commitment_target") is not None:
            # Scripted leaders are NOT evidence: they state the target row half the time and a
            # rival row otherwise, which keeps the follow-through column exercised end to end.
            row = int(lp["commitment_target"])
            if int(digest[:2], 16) % 2:
                row = (row + 1) % len((ob.macro_physical or {}).get("rows", E4_MATRIX))
            return f"I commit to row {row} next round, that is the fair choice for us."
        pool = ["I think the previous round is worth discussing.",
                "Let's compare the results we just saw.",
                "I value fairness and will consider what we discussed."]
        return pool[int(digest[:2], 16) % len(pool)]

    # -- one agent, one action -------------------------------------------------------------------
    def _choose_one(self, ob, memories=None):
        lp = dict(ob.local_physical or {})
        memories = memories or {}
        if not lp.get("personal_history"):
            mem = memories.get(ob.agent_id, {}) if isinstance(memories, dict) else {}
            if isinstance(mem, dict):
                lp["personal_history"] = mem.get("personal_history", mem.get("history", []))
        ob = SimpleNamespace(agent_id=ob.agent_id, step=ob.step, macro_physical=ob.macro_physical,
                             macro_information=ob.macro_information, local_physical=lp)
        r = int(lp.get("round_idx", 0))
        if r == 0:
            return self._decide_allocation(ob)
        return self._decide_message(ob, r)

    def _decide_allocation(self, ob):
        prompt = build_allocation_prompt(ob)
        row, reason, source = None, "", "rule"
        if self.llm_kind == "openai":
            try:
                reply = str(self._call_llm(prompt) or "")
                m = re.search(r'"row"\s*:\s*(\d+)', reply)
                if not m:
                    raise ValueError("unparseable allocation")
                row = int(m.group(1))
                if not 0 <= row < len(ob.macro_physical["rows"]):
                    raise ValueError("row out of range")
                rm = re.search(r'"reason"\s*:\s*"((?:[^"\\]|\\.)*)"', reply, re.S)
                reason = json.loads('"' + rm.group(1) + '"')[:200] if rm else ""
                source = "llm"
            except Exception:
                row, source, reason = self._scripted_row(ob), "fallback", ""
        else:
            row = self._scripted_row(ob)
        return _mk_action(agent_id=ob.agent_id, step=ob.step, kind="allocate_in_group_matrix",
                          payload={"row": row, "reason": reason}, source=source,
                          rationale=[reason or "allocation choice"])

    def _decide_message(self, ob, r: int):
        prompt = build_chat_prompt(ob, r)
        content, source = "", "rule"
        if self.llm_kind == "openai":
            try:
                reply = str(self._call_llm(prompt) or "")
                m = re.search(r'"message"\s*:\s*"((?:[^"\\]|\\.)*)"', reply, re.S)
                content = json.loads('"' + m.group(1) + '"') if m else reply.strip()
                if not content.strip():
                    raise ValueError("empty message")
                source = "llm"
            except Exception:
                content, source = self._scripted_message(ob), "fallback"
        else:
            content = self._scripted_message(ob)
        payload = {"content": content, "round_idx": r}
        if ob.local_physical.get("series") == "e3":
            payload["recipient"] = ob.local_physical.get("partner_id")
        return _mk_action(agent_id=ob.agent_id, step=ob.step, kind="chat", payload=payload,
                          source=source, rationale=["chat message"])

    def decide_batch(self, obs, memories=None):
        obs = list(obs)
        if not obs:
            return []
        if self.llm_kind == "openai" and len(obs) > 1:
            from concurrent.futures import ThreadPoolExecutor
            llm = self._llm
            with ThreadPoolExecutor(max_workers=min(8, len(obs))) as ex:
                actions = list(ex.map(lambda o: self._choose_one(o, memories), obs))
        else:
            actions = [self._choose_one(o, memories) for o in obs]
        if self.llm_kind == "openai":
            self.call_count += len(obs)
            self.fallback_count += sum(1 for a in actions if getattr(a, "source", None) == "fallback")
        return actions


# --- metrics -----------------------------------------------------------------------------------
def _avg(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


@register("collector", COLLECTOR_REF)
class CommunicationMetricCollector(MetricCollector):
    def collect(self, env, actions, t: int):
        t = int(t)
        aa = [x for x in env.allocations if x["step"] == t]
        msgs = [x for x in env.messages if x["step"] == t]
        deliv = [x for x in env.deliveries if x["step"] == t]
        members = [x for x in aa if not x["is_leader"]]
        leaders = [x for x in aa if x["is_leader"]]
        gaps = [x["gap"] for x in aa]
        step_calls = [x for x in env.call_events if x["step"] == t]
        joints = [a + b for a, b in env.matrix]
        lo, hi = min(joints), max(joints)
        total = sum(x["a_points"] + x["b_points"] for x in aa)
        welfare = round((total - lo * len(aa)) / ((hi - lo) * len(aa)), 4) if aa and hi > lo else None
        return {
            "condition": env.condition, "series": env.series, "step": t,
            "mean_gap": _avg(gaps),
            "mif_favour_share": round(sum(g > 0 for g in gaps) / len(gaps), 4) if gaps else None,
            "group_welfare": welfare,
            "mean_abs_gap": _avg([abs(x["a_points"] - x["b_points"]) for x in aa]),
            "mean_gap_member": _avg([x["gap"] for x in members]),
            "mean_gap_leader": _avg([x["gap"] for x in leaders]),
            "mean_gap_group_a": _avg([x["gap"] for x in aa if x["own_group"] == "A"]),
            "mean_gap_group_b": _avg([x["gap"] for x in aa if x["own_group"] == "B"]),
            "n_messages": len(msgs),
            "n_deliveries": len(deliv),
            "mean_message_len": _avg([x["char_len"] for x in msgs]),
            "n_truncated": sum(1 for x in msgs if x["truncated"]),
            "n_allocations": len(aa),
            "fallback_rate": (round(sum(1 for x in step_calls if x["fallback"]) / len(step_calls), 4)
                              if step_calls else 0.0),
        }

    def columns(self):
        return ["condition", "series", "step", "mean_gap", "mif_favour_share", "group_welfare",
                "mean_abs_gap", "mean_gap_member", "mean_gap_leader", "mean_gap_group_a",
                "mean_gap_group_b", "n_messages", "n_deliveries", "mean_message_len", "n_truncated",
                "n_allocations", "fallback_rate"]


# --- standalone runner (no engine required; used by the selftest and scripted smoke runs) -------
def run_condition(condition: str, steps: int = STEPS, seed: int = 0, n_agents: int = N_AGENTS,
                  llm_kind: str = "scripted", model: str | None = None, out: str | Path | None = None,
                  chat_subrounds: tuple[int, ...] = CHAT_SUBROUNDS):
    validate_condition(condition, n_agents)
    env = CommunicationEnvironmentProvider(condition=condition, n_agents=n_agents, seed=seed, steps=steps)
    dec = CommunicationDecisionModel(llm_kind=llm_kind, seed=seed, model=model)
    col = CommunicationMetricCollector()
    env.reset(seed)
    ids = sorted(env.groups)
    metrics = []
    for t in range(1, steps + 1):
        env.advance_to(t)
        env.apply(dec.decide_batch(env.observe_batch(ids, t, 0), {}))
        if t < steps and condition != "e2-silent":
            for r in chat_subrounds:
                route = env._route_for(t, r)
                if not route.senders:
                    continue
                env.apply(dec.decide_batch(env.observe_batch(route.senders, t, r), {}))
        metrics.append(col.collect(env, [], t))
    env.finalize()
    if out is not None:
        export_run(env, metrics, Path(out), config={"condition": condition, "steps": steps,
                                                    "seed": seed, "n_agents": n_agents,
                                                    "llm_kind": llm_kind, "model": model})
    return env, metrics


def export_run(env, metrics, dest: Path, config: dict | None = None) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    if config is not None:
        (dest / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    def dump(name, rows, fields):
        with (dest / name).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    dump("roster.csv", [{"agent_id": a, "group": g, "is_leader": a in env.leader_of}
                        for a, g in sorted(env.groups.items())], ["agent_id", "group", "is_leader"])
    dump("pairings.csv", [{"step": t, "sender": a, "recipient": b, "sender_group": env.groups[a],
                           "recipient_group": env.groups[b]}
                          for t, ps in sorted(env.pairings.items()) for a, b in ps],
         ["step", "sender", "recipient", "sender_group", "recipient_group"])
    dump("allocations.csv", env.allocations,
         ["condition", "series", "step", "agent_id", "own_group", "is_leader", "row", "a_points",
          "b_points", "ingroup_points", "outgroup_points", "gap", "reason", "source"])
    dump("messages.csv", env.messages,
         ["message_id", "condition", "series", "step", "round", "sender", "sender_group", "channel",
          "recipients", "n_recipients", "char_len", "truncated", "content", "source",
          "stance", "issued_step", "target_step", "expected_row", "commitment_row", "realised_row",
          "fulfilled"])
    dump("deliveries.csv", env.deliveries,
         ["message_id", "step", "round", "recipient", "recipient_group", "delivered_step"])
    dump("rejections.csv", env.rejections, ["step", "round", "sender", "reason"])
    dump("scores.csv", env.scores,
         ["step", "group_a_received", "group_b_received", "AA", "AB", "BA", "BB", "total_sum"])
    dump("announcements.csv", env.announcements, ["step", "condition", "text"])
    dump("metrics.csv", metrics, list(metrics[0]) if metrics else ["step"])
    return dest
