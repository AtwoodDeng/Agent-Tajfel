"""tajfel — Tajfel (1971) minimal-group paradigm as a FROM-SCRATCH SocioVerse2 study.

The classic minimal-group experiment: participants are assigned to one of two groups on a
trivial criterion, then allocate a small money matrix to an anonymous ingroup member and an
anonymous outgroup member. No face-to-face contact, no self-interest -- the only thing that
varies is whether the group membership is salient. The robust finding (Tajfel et al. 1971;
reviewed by Balliet et al. 2014, meta-analytic r ~ .30 favouring the ingroup) is that people
give the ingroup more even at a cost to their own absolute payoff: the "pull of ingroup
favouritism".

This study implements the paradigm as one LongitudinalSimulator run with a runtime condition
switch, so the SAME registered providers serve all three arms (no per-arm study directory):

  c1  label visible  — the classic minimal group: the allocation matrix shows the ingroup and
                       outgroup columns explicitly. Favourite the ingroup.
  c2  label hidden   — same matrices, same payoffs, but the columns are presented as
                       "Person A" / "Person B" with the group labels withheld. The minimal-group
                       claim predicts the ingroup bias DISAPPEARS here (memory-based, not
                       payoff-based; the bias is social identity, not a payoff asymmetry).
  c3  rarity         — label visible, but the agent's own group is the minority one (10 of 40 /
                       10 of 50). Tests whether the effect tracks mere categorization or
                       distinctiveness/majority status.

Design notes (what a reader of the trajectory should know):
  - P: a FIXED pool of agents with persistent ids (`tj-000`, ...) split into two groups by a
    trivial criterion. Group assignment is deterministic in (seed, n_agents) so a re-run with the
    same seed reproduces the same pool.
  - E (4 quadrants): the current allocation matrix (macro_physical: the 13-point payoff matrix
    type and its rows; local_physical: this agent's own group, the ingroup/outgroup targets and
    the condition's visibility flag), a macro_information line restating the condition's framing.
  - B: each agent, each step, picks one row of the matrix (the classic single-choice response).
    The loop folds B back into E: the agent's chosen row updates its own payoff state and the
    population-level ingroup-bias record.
  - Metrics: the ingroup-minus-outgroup allocation gap, split by condition-relevant subsets.

Two decision modes behind one registered ref (`tajfel.decision`), exactly like
`studies/opinion_diffusion`:
  - `llm_kind="scripted"` (DEFAULT): a NOISY FORMULAIC allocator. It is a PIPELINE-VALIDATION
    stub only -- it invents a choice from a fixed bias formula plus per-agent noise. It will NOT
    reproduce the real minimal-group effect and must not be read as a confirmatory result.
  - `llm_kind="openai"`: each agent role-plays the matrix choice in its own LLM call, one
    concurrent fan-out per step, and the SAME formulaic choice is applied per agent as a
    guaranteed fallback whenever the reply does not parse (`source="fallback"`, so the
    fallback_rate metric makes an unreliable run visible).

Reference: Tajfel, H., Billig, M. G., Bundy, R. P., & Flament, C. (1971). Social categorization
and intergroup behaviour. European Journal of Social Psychology, 1(2), 149-178.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from statistics import mean
from typing import Any, Callable

from socioverse.abc import (
    DecisionModel,
    EnvironmentProvider,
    MetricCollector,
    PopulationProvider,
)
from socioverse.engine import register
from socioverse.schemas import (
    Action,
    Broadcast,
    EnvironmentBundle,
    EnvironmentLayer,
    InformationProgram,
    InteractionStructure,
    Observation,
    Persona,
    PopulationBundle,
    PropagationMode,
    ScheduledEvent,
    SimulationConfig,
    StudySpec,
)

_log = logging.getLogger(__name__)

# --- the Tajfel allocation matrices (the classic 13-point, 6-row response format) ---------------
# Each matrix is a list of ROWS; a row is (ingroup_points, outgroup_points) for the two anonymous
# recipients. The 6 rows run from "maximum joint profit" style anchors to "maximum ingroup
# difference". We only need the payoff pairs -- the agents choose ONE row index.
#
# Type 1 = parity/integrity: joint profit is constant across rows, so the choice is purely
#          distributive (who gets more). Type 2 = maximum joint profit (MJP): joint profit peaks
#          at one row, so a payoff-seeking agent has a clear target. Type 3 = maximum ingroup
#          difference (MIF): the ingroup-vs-outgroup GAP peaks at one end, opposite the joint
#          profit end -- the classic trade-off that reveals ingroup favouritism.

# Reading the response correctly matters more than it looks. On a type-1 (parity) matrix the
# own-group-minus-other-group GAP varies row to row on every matrix, so that gap IS the
# dependent variable throughout. On a type-3 (MIF) matrix the gap varies in the OPPOSITE
# direction to the joint profit: walking up the table buys the ingroup a bigger relative
# advantage while shrinking the total paid out. That sacrifice is what the matrix measures.

MATRIX_PARITY: list[tuple[int, int]] = [
    (19, 1), (17, 3), (15, 5), (13, 7), (11, 9),
    (9, 11), (7, 13), (5, 15), (3, 17), (1, 19),
]
# Type 3 (MIF): the gap GROWS as the joint profit FALLS, which is the real maximum-difference
# trade-off: row 0 is the maximum-difference row (biggest gap, lowest joint profit), the last
# row is the maximum-joint row (biggest total, own group behind). No row dominates another --
# own-group points fall monotonically while the other group's rise -- so every one of the four
# Tajfel strategies (max joint / max own / max difference / fair) lands on a different row.
MATRIX_MIF: list[tuple[int, int]] = [
    (15, 5), (14, 8), (13, 11), (12, 14), (11, 17), (10, 20),
]
# Type 2 (MJP): joint profit peaks INTERIOR to the table and the gap falls away at the high end,
# so a joint-profit maximiser lands mid-table while an ingroup-favouring agent drifts toward the
# low rows (accepting a smaller joint profit to keep the ingroup ahead). The interior peak is what
# makes this a trade-off matrix rather than a monotone one.
MATRIX_MJP: list[tuple[int, int]] = [
    (7, 1), (10, 5), (11, 10), (12, 15), (10, 19), (8, 21),
]

MATRICES: dict[str, list[tuple[int, int]]] = {
    "parity": MATRIX_PARITY,
    "mjp": MATRIX_MJP,   # maximum joint profit
    "mif": MATRIX_MIF,   # maximum ingroup difference
}

# Register the extended matrix library (tajfel/matrices.py: the original 1971 matrix, AB-symmetric
# MIP/MJP/P and their loss-domain variants) so `matrix_for` / `row_count` / the env rotation can
# resolve them by name. setdefault keeps the three canonical names above taking precedence.
try:
    from .matrices import MATRIX_LIBRARY as _MATRIX_LIBRARY
    for _name, _entry in _MATRIX_LIBRARY.items():
        MATRICES.setdefault(_name, _entry["rows"])
except Exception:  # pragma: no cover - keep model.py importable without matrices.py
    pass

# Anchor rows (what "favouring the ingroup" means on each matrix type).
ROW_MIF_FAVOUR = 0                          # maximum difference / lowest joint profit
ROW_MJP_MAX_JOINT = max(range(len(MATRIX_MJP)), key=lambda r: sum(MATRIX_MJP[r]))
ROW_PARITY_FAIR = min(range(len(MATRIX_PARITY)),
                      key=lambda r: abs(MATRIX_PARITY[r][0] - MATRIX_PARITY[r][1]))

# --- conditions --------------------------------------------------------------------------------

CONDITIONS = ("c1", "c2", "c3")
DEFAULT_CONDITION = "c1"

CONDITION_LABELS = {
    "c1": "label visible (classic minimal group)",
    "c2": "label hidden (Person A / Person B, group labels withheld)",
    "c3": "rarity (label visible, own group is the 10/40 minority)",
}

CONDITION_FRAMING = {
    "c1": ("You and another participant were assigned to groups by a trivial criterion. "
           "The matrix below shows your group's column and the other group's column."),
    "c2": ("You and another participant were assigned to groups by a trivial criterion, but the "
           "matrix below hides which column is which group -- both columns are labelled only "
           "Person A and Person B."),
    "c3": ("You and another participant were assigned to groups by a trivial criterion. Your own "
           "group is much smaller than the other group. The matrix below shows both columns."),
}
# The c3 cover story has to be TRUE for the agent hearing it: only the minority group is rare,
# so a majority-group agent gets the "larger" wording instead of being told its group is small.
CONDITION_FRAMING["c3_majority"] = (
    "You and another participant were assigned to groups by a trivial criterion. Your own group "
    "is much larger than the other group. The matrix below shows both columns.")

# --- shared, PURE seed / index functions (no mutable shared state -> generic cls(bundle) works) ---


def agent_id(i: int) -> str:
    return f"tj-{i:03d}"


def agent_index(aid: str) -> int:
    """Inverse of agent_id; tolerant of the row-suffix variants authors sometimes use."""
    return int(aid.split("-")[1])


GROUPS = ("A", "B")
# Concrete, meaningless group names shown to agents in place of "ingroup"/"outgroup".
GROUP_NAMES = {"A": "BLOTQW", "B": "BTSLAB"}


def group_of(i: int, n: int, condition: str = DEFAULT_CONDITION) -> str:
    """Which of the two groups agent i belongs to. c3 makes group A the rare minority."""
    if n <= 1:
        return "A"
    if condition == "c3":
        n_minority = max(1, min(n - 1, int(round(n * 0.2))))
        return "A" if i < n_minority else "B"
    return "A" if i % 2 == 0 else "B"


def other_group(g: str) -> str:
    return "B" if g == "A" else "A"


def group_sizes(n: int, condition: str = DEFAULT_CONDITION) -> tuple[int, int]:
    """(n_A, n_B) for the given pool size and condition."""
    n_a = sum(1 for i in range(n) if group_of(i, n, condition) == "A")
    return n_a, n - n_a


def own_points(group: str, a: int, b: int) -> int:
    """Points a (group-A, group-B) row gives to the DECIDING agent's own group."""
    return int(a) if group == "A" else int(b)


def own_gap(group: str, a: int, b: int) -> int:
    """> 0 means this agent gave their OWN group more than the other group."""
    return (int(a) - int(b)) if group == "A" else (int(b) - int(a))


def matrix_for(kind: str) -> list[tuple[int, int]]:
    if kind not in MATRICES:
        raise ValueError(f"unknown matrix kind {kind!r}; expected one of {sorted(MATRICES)}")
    return MATRICES[kind]


def allocation_gap(kind: str, row: int) -> int:
    """ingroup minus outgroup points for a chosen row (the raw points difference)."""
    pairs = matrix_for(kind)
    row = max(0, min(len(pairs) - 1, int(row)))
    a, b = pairs[row]
    return int(a) - int(b)


def ingroup_favour(kind: str, row: int, own_group: str = "A") -> float:
    """How much a choice favours the DECIDING agent's own group over the other group.

    Always the oriented points gap (own-group points - other-group points), so higher is always
    more ingroup-favouring and the three matrix types pool cleanly. Each agent's "ingroup" is
    their OWN group: on the c3 arm a majority-group agent who favours their own group scores
    positive too, which is what makes the per-allocator stratification of C3 possible.
    `kind` is accepted for the call sites and to bounds-check the row.
    """
    pairs = matrix_for(kind)
    row = max(0, min(len(pairs) - 1, int(row)))
    a, b = pairs[row]
    return own_gap(own_group, a, b)


def row_count(kind: str) -> int:
    return len(matrix_for(kind))


# --- P: the fixed population pool ---------------------------------------------------------------


@register("population", "tajfel.pop")
class TajfelPopulationProvider(PopulationProvider):
    def __init__(self, bundle: PopulationBundle):
        self.bundle = bundle
        self.n = int(bundle.provider_args.get("n_agents", 10))
        self.condition = str(bundle.provider_args.get("condition", DEFAULT_CONDITION))
        if self.condition not in CONDITIONS:
            raise ValueError(f"condition must be one of {CONDITIONS}, got {self.condition!r}")

    def build(self, seed: int) -> list[Persona]:
        n_a, n_b = group_sizes(self.n, self.condition)
        out = []
        for i in range(self.n):
            g = group_of(i, self.n, self.condition)
            trivial = ["over-estimated the number of dots", "preferred the painting"]
            n_own = n_a if g == "A" else n_b

            out.append(Persona(
                agent_id=agent_id(i),
                group_key=g,
                attributes={
                    "group": g,
                    "condition": self.condition,
                    "trivial_criterion": trivial[i % len(trivial)],
                    "group_size": n_own,
                    "other_group_size": self.n - n_own,
                    "pool_size": self.n,
                },
                init_state={"group": g, "steps_played": 0, "cum_gap": 0},
            ))
        return out

    def neighbors(self, aid: str) -> list[str]:
        """No interaction: the Tajfel paradigm is deliberately INDEPENDENT decisions."""
        return []


# --- E: the dynamic environment (owns the per-agent allocation history) -------------------------


@register("environment", "tajfel.env")
class TajfelEnvironmentProvider(EnvironmentProvider):
    """Carries the matrix currently on the agent's desk + each agent's accumulated allocation.

    `provider_args`:
      - n_agents (int)          pool size, must match the population bundle
      - condition (str)         c1 | c2 | c3
      - matrices (list[str])    which matrix types to rotate through by step (default all three)
      - label_visible (bool)    explicit override; otherwise derived from the condition
    """

    def __init__(self, bundle: EnvironmentBundle):
        self.bundle = bundle
        self.n = int(bundle.provider_args.get("n_agents", 10))
        self.condition = str(bundle.provider_args.get("condition", DEFAULT_CONDITION))
        self.matrices = list(bundle.provider_args.get("matrices") or ["parity", "mjp", "mif"])
        self.label_visible = bool(bundle.provider_args.get(
            "label_visible", self.condition in ("c1", "c3")))
        self.history: dict[str, list[dict[str, Any]]] = {}
        self._fired: set[int] = set()

    # -- life cycle --
    def reset(self, seed: int) -> None:
        self.history = {agent_id(i): [] for i in range(self.n)}
        self._fired = set()

    def matrix_at(self, t: int) -> str:
        """Rotate the matrix types by step, so a multi-step run samples all three bands."""
        return self.matrices[(max(1, t) - 1) % len(self.matrices)]

    def advance_to(self, t: int) -> list[ScheduledEvent]:
        fired: list[ScheduledEvent] = []
        for ev in self.bundle.scheduled_events:
            if ev.at_step == t and id(ev) not in self._fired:
                self._fired.add(id(ev))
                fired.append(ev)
        return fired

    def apply(self, actions: list[Action]) -> None:
        for a in actions:
            if a.kind != "allocate_in_group_matrix":
                continue
            rec = {
                "t": int(a.step),
                "matrix": a.payload.get("matrix"),
                "row": int(a.payload.get("row", 0)),
                "ingroup_points": int(a.payload.get("ingroup_points", 0)),
                "outgroup_points": int(a.payload.get("outgroup_points", 0)),
                "gap": int(a.payload.get("gap", 0)),
                "favour": float(a.payload.get("favour", 0.0)),
            }
            self.history.setdefault(a.agent_id, []).append(rec)

    def observe_batch(self, agent_ids: list[str], t: int, round_idx: int = 0) -> list[Observation]:
        kind = self.matrix_at(t)
        pairs = matrix_for(kind)
        # rows are stored canonically as (group A, group B); each agent is shown them from its
        # own point of view, so "the left column is YOUR group" stays true for every agent
        rows = [{"row": r, "a_points": a, "b_points": b, "gap": a - b, "joint": a + b}
                for r, (a, b) in enumerate(pairs)]
        n_a, n_b = group_sizes(self.n, self.condition)

        out: list[Observation] = []
        for aid in agent_ids:
            i = agent_index(aid)
            g = group_of(i, self.n, self.condition)
            n_own = n_a if g == "A" else n_b
            n_other = self.n - n_own
            # only the MINORITY is rare, so a majority agent must not be told its group is small
            if self.condition == "c3" and n_own > n_other:
                framing = CONDITION_FRAMING["c3_majority"]
            else:
                framing = CONDITION_FRAMING.get(self.condition,
                                                CONDITION_FRAMING[DEFAULT_CONDITION])
            records = self.history.get(aid, [])
            out.append(Observation(
                agent_id=aid, step=t,
                macro_physical={
                    "matrix_kind": kind,
                    "rows": rows,
                    "label_visible": self.label_visible,
                },
                local_physical={
                    "own_group": g,
                    "other_group": other_group(g),
                    "own_group_size": n_own,
                    "other_group_size": n_other,
                    "ingroup_column": "left" if self.label_visible else None,
                    "outgroup_column": "right" if self.label_visible else None,
                    "steps_played": len(records),
                    "cum_gap": sum(int(r["gap"]) for r in records),
                    "last_choice_gap": records[-1]["gap"] if records else None,
                },
                macro_information={"framing": framing, "condition": self.condition},
            ))
        return out

    def agent_state(self, agent_id_: str) -> dict:
        records = self.history.get(agent_id_, [])
        i = agent_index(agent_id_)
        return {
            "group": group_of(i, self.n, self.condition),
            "steps_played": len(records),
            "cum_gap": sum(r["gap"] for r in records),
            "mean_favour": (round(mean([r["favour"] for r in records]), 4) if records else None),
            "last_gap": records[-1]["gap"] if records else None,
            "last_matrix": records[-1]["matrix"] if records else None,
        }

    def snapshot(self) -> dict:
        favours = [r["favour"] for recs in self.history.values() for r in recs]
        gaps = [r["gap"] for recs in self.history.values() for r in recs]
        return {
            "condition": self.condition,
            "label_visible": self.label_visible,
            "n_allocations": len(favours),
            "mean_favour": round(mean(favours), 4) if favours else None,
            "mean_gap": round(mean(gaps), 4) if gaps else None,
        }

    # -- convenience for the collector / runner (not part of the ABC) --
    def groups(self) -> dict[str, str]:
        return {agent_id(i): group_of(i, self.n, self.condition) for i in range(self.n)}


# --- B: the decision model (formulaic stub + opt-in per-agent LLM behind one ref) --------------


def _load_dotenv(root: Path | None = None) -> None:
    from socioverse.external_events import _load_dotenv as load

    load(root)


# --- LLM providers ------------------------------------------------------------------------------
# One OpenAI-compatible client behind each `llm_kind` value. `openai` keeps the legacy SV_LLM_*
# environment variables; additional slots read SV_LLM2_* / SV_LLM3_* so several providers can be
# configured side by side (each with its own base URL, model and API key). Add a slot here plus the
# matching SV_LLM<N>_* lines in .env to register another provider.
LLM_PROVIDERS: dict[str, dict[str, str]] = {
    "openai": {"prefix": "SV_LLM", "default_base_url": "https://api.openai.com/v1",
               "default_model": "gpt-4o"},
    "llm2": {"prefix": "SV_LLM2", "default_base_url": "", "default_model": ""},
    "llm3": {"prefix": "SV_LLM3", "default_base_url": "", "default_model": ""},
}


def is_llm_kind(kind: str) -> bool:
    """True for any kind that resolves to a real LLM client (as opposed to `scripted`)."""
    return kind in LLM_PROVIDERS


def _provider_env(provider: str) -> tuple[str, str, str]:
    """(api_key, base_url, model) resolved from the environment for one provider slot."""
    cfg = LLM_PROVIDERS[provider]
    prefix = cfg["prefix"]
    api_key = os.environ.get(f"{prefix}_API_KEY") or ""
    if provider == "openai":
        api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get(f"{prefix}_BASE_URL") or cfg["default_base_url"]
    model = os.environ.get(f"{prefix}_MODEL") or cfg["default_model"]
    return api_key, base_url, model


class _OpenAILLM:
    """Thin OpenAI-compatible client over one registered provider slot, lazily connected."""

    def __init__(self, model: str | None = None, temperature: float = 0.7, max_tokens: int = 160,
                 provider: str = "openai"):
        _load_dotenv()
        self.provider = provider
        api_key, base_url, default_model = _provider_env(provider)
        self.model = model or default_model
        self.api_key = api_key
        self.base_url = base_url
        self.temperature, self.max_tokens = temperature, max_tokens
        self._client = None

    def _ensure(self):
        if self._client is None:
            prefix = LLM_PROVIDERS[self.provider]["prefix"]
            if not self.api_key:
                raise RuntimeError(
                    f"no API key for LLM provider {self.provider!r}; set {prefix}_API_KEY in .env")
            if not self.base_url:
                raise RuntimeError(
                    f"no base URL for LLM provider {self.provider!r}; set {prefix}_BASE_URL in .env")
            if not self.model:
                raise RuntimeError(
                    f"no model for LLM provider {self.provider!r}; set {prefix}_MODEL in .env")
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def __call__(self, prompt: str) -> str:
        resp = self._ensure().chat.completions.create(
            model=self.model, temperature=self.temperature, max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}])
        return resp.choices[0].message.content or ""


def _make_llm(kind: str, **kw) -> Callable[[str], str] | None:
    if kind == "scripted":
        return None
    if is_llm_kind(kind):
        return _OpenAILLM(provider=kind, **kw)
    raise ValueError(f"unknown llm_kind: {kind!r}")


_PROMPTS: dict[str, dict[str, str]] = {
    "en": {
        "head": ("You are a participant in a short decision study. {framing}"),
        "task_labelled": ("You are a member of the {own_name} group. You must allocate points to "
                          "two anonymous people: the {own_name} member (listed in the left column) "
                          "and the {other_name} member (listed in the right column). Pick exactly "
                          "ONE row of the matrix below. Each row shows ({own_name} points, "
                          "{other_name} points)."),
        "task_hidden": ("You must allocate money between two anonymous people -- Person A in the "
                        "left column and Person B in the right column. Pick exactly ONE row of "
                        "the matrix below. Each row shows (Person A's points, Person B's points)."),
        "tail": ('Answer with one line of JSON only: {{"row": <a row number from 0 to {max_row}>, '
                 '"reason": "<at most 12 words>"}}'),
        "note_visible": "- Left column = the {own_name} member, right column = the {other_name} member.",
        "note_hidden": ("- You are not told which of Person A and Person B belongs to which group."),
    },
    "zh": {
        "head": ("你是参加一个简短决策研究的被试。{framing}"),
        "task_labelled": ("你是 {own_name} 组的成员。你需要把积分分配给两个匿名的人：{own_name} 组成员（在左列）"
                          "和 {other_name} 组成员（在右列）。请在下面的矩阵中只选一行。每一行是（{own_name} 的点数, "
                          "{other_name} 的点数）。"),
        "task_hidden": ("你需要把金额分配给两个匿名的人——左列的甲与右列的乙。请在下面的矩阵中只选一行。"
                        "每一行是（甲的点数, 乙的点数）。"),
        "tail": ('只输出一行 JSON: {{"row": <0 到 {max_row} 之间的行号>, '
                 '"reason": "<不超过20字的理由>"}}'),
        "note_visible": "- 左列 = {own_name} 组成员，右列 = {other_name} 组成员。",
        "note_hidden": "- 你并不知道甲和乙分别属于哪个群体。",
    },
}
PROMPT_LANGS = tuple(_PROMPTS)


def build_tajfel_prompt(ob: Observation, lang: str = "en") -> str:
    """Render one agent's allocation prompt for the matrix currently on its desk.

    The c2 arm must not leak the category: when `label_visible` is false the columns are named
    Person A / Person B and the task text never mentions ingroup or outgroup. This is the whole
    manipulation, so it is enforced here rather than left to the caller.

    DELIBERATELY MEMORYLESS (C1-C3): the classic Tajfel paradigm treats every allocation as an
    INDEPENDENT single-shot decision -- real participants never see their own past matrices --
    so the history fields the environment puts in local_physical (steps_played / cum_gap /
    last_choice_gap) are intentionally NOT rendered into this prompt, and decide_batch
    intentionally ignores its `memories` argument. The D-series feedback study (see
    Docs/Specs/tajfel_feedback_study_spec.md) is the arm where cross-round memory is the
    construct under test; it gets its own prompt section, not a change to this one.
    """
    t = _PROMPTS[lang]
    framing = ob.macro_information.get("framing", "")
    rows = ob.macro_physical.get("rows", [])
    label_visible = bool(ob.macro_physical.get("label_visible", True))
    # rows are stored canonically (A, B); on a labelled arm the agent sees the two columns from
    # ITS OWN point of view, so "the left column is your group" is true for every agent
    own = (ob.local_physical or {}).get("own_group", "A")
    own_name = GROUP_NAMES.get(own, own)
    other_name = GROUP_NAMES.get(other_group(own), other_group(own))

    lines = [t["head"].format(framing=framing)]
    if label_visible:
        lines.append(t["note_visible"].format(own_name=own_name, other_name=other_name))
        lines.append(t["task_labelled"].format(own_name=own_name, other_name=other_name))
    else:
        lines.append(t["note_hidden"])
        lines.append(t["task_hidden"])
    lines.append("")
    for r in rows:
        a, b = int(r["a_points"]), int(r["b_points"])
        if label_visible:
            own_pts, other_pts = (a, b) if own == "A" else (b, a)
            lines.append(f"  row {r['row']}:  {own_name} {own_pts}  {other_name} {other_pts}   "
                         f"(joint {a + b})")
        else:
            # a hidden label must not disclose which column is the agent's own group
            lines.append(f"  row {r['row']}:  A {a}  B {b}   (joint {a + b})")
    lines.append("")
    lines.append(t["tail"].format(max_row=len(rows) - 1))
    return "\n".join(lines)


_ROW_RE = re.compile(r'"row"\s*:\s*(-?\d+)')
_REASON_RE = re.compile(r'"reason"\s*:\s*"([^"]*)"')


def parse_tajfel_reply(resp: str, n_rows: int) -> dict | None:
    """Pull {row, reason} out of the model's reply; None when no usable row index is present."""
    if not resp:
        return None
    m = _ROW_RE.search(resp)
    if m is None:
        m = re.search(r'\brow\s*#?\s*(\d+)\b', resp, re.IGNORECASE)
    if m is None:
        return None
    try:
        row = int(m.group(1))
    except (TypeError, ValueError):
        return None
    if not (0 <= row < n_rows):
        return None
    r = _REASON_RE.search(resp)
    return {"row": row, "reason": (r.group(1) if r else "").strip()}


@register("decision", "tajfel.decision")
class TajfelDecisionModel(DecisionModel):
    """Tajfel matrix choice. `llm_kind="scripted"` (default) is a NOISY FORMULAIC STUB -- a
    pipeline smoke-test that fills every metric without spending a single LLM call. It is NOT
    a model of the phenomenon: its output is regenerated from a fixed bias parameter plus
    seeded per-agent noise. `llm_kind="openai"` role-plays each agent in its own LLM call (one
    concurrent fan-out per step), falling back to the identical formula for any agent whose
    reply does not parse.

    `decision_args`:
      - matrix_bias (float)   how far up the ingroup-difference end a scripted agent drifts (0..1)
      - noise (float)         per-agent scripted randomness (0..1)
      - label_gain (float)    how strongly the SCRIPTED stub responds to a visible label
      - rarity_gain (float)   extra scripted drift when the agent's own group is the minority
      - llm_kind              `scripted` (default) or any key in LLM_PROVIDERS
                              (`openai`, `llm2`, `llm3`); each provider reads its own
                              SV_LLM* / SV_LLM2* / SV_LLM3* env vars from .env
      - model, temperature, max_tokens, max_workers, prompt_lang
    """

    def __init__(self, matrix_bias: float = 0.65, noise: float = 0.5, label_gain: float = 0.3,
                 rarity_gain: float = 0.3,
                 llm_kind: str = "scripted", model: str | None = None,
                 temperature: float = 0.7, max_tokens: int = 160, max_workers: int = 8,
                 prompt_lang: str = "en"):
        if prompt_lang not in PROMPT_LANGS:
            raise ValueError(f"prompt_lang must be one of {PROMPT_LANGS}, got {prompt_lang!r}")
        self.prompt_lang = prompt_lang
        self.matrix_bias = float(matrix_bias)
        self.noise = float(noise)
        self.label_gain = float(label_gain)
        self.rarity_gain = float(rarity_gain)
        self.llm_kind = llm_kind
        self.max_workers = max(1, int(max_workers))
        self.llm = _make_llm(llm_kind, model=model, temperature=temperature, max_tokens=max_tokens)

    def _scripted_row(self, ob: Observation, n_rows: int) -> int:
        """A deterministic-but-noisy row pick.

        Deterministic in (agent, step, matrix kind, condition, own group) so a re-run reproduces
        the same choices, but spread across agents rather than pinned to one row: `noise` sets
        how wide the individual draw is around the group-level centre. The centre is the row
        that would favour THIS agent's own group on THIS matrix type; a visible label tilts the
        centre further that way and, on c3, being the rare minority tilts it a little further
        still.

        This is a PIPELINE PROBE, not a model of the phenomenon. Reading the condition here is
        what makes the c1/c3 arms distinguishable at all -- without it a broken c3 wiring would
        pass the smoke test silently -- so the tilt direction is a wiring assertion, not
        evidence about minimal-group behaviour.
        """
        rows = ob.macro_physical.get("rows", []) or []
        if not rows:
            return 0
        lpb = ob.local_physical or {}
        own = lpb.get("own_group", "A")
        cond = str((ob.macro_information or {}).get("condition", ""))
        kind = ob.macro_physical.get("matrix_kind", "")
        h = 0
        for ch in f"{ob.agent_id}|{ob.step}|{kind}|{cond}|{own}":
            h = (h * 131 + ord(ch)) % 100003
        u = (h % 997) / 997.0                          # u in [0, 1), well spread over agents

        visible = bool(ob.macro_physical.get("label_visible"))
        # rank 0.0 = the row that most favours this agent's own group, 1.0 = the opposite end.
        # A hidden label carries no own-group cue, so it is ranked on the canonical A/B gap.
        gaps = []
        for r in rows:
            a, b = int(r["a_points"]), int(r["b_points"])
            gaps.append((a - b) if (own == "A" or not visible) else (b - a))
        lo, hi = min(gaps), max(gaps)

        centre = 0.5
        if visible:
            centre -= self.label_gain * self.matrix_bias     # drift toward the own-group end
            n_own = lpb.get("own_group_size")
            n_oth = lpb.get("other_group_size")
            if cond == "c3" and n_own is not None and n_oth is not None and n_own < n_oth:
                centre -= self.rarity_gain * self.matrix_bias
        # noise widens the draw around the centre (noise=0 -> everyone picks the centre row)
        pos = min(1.0, max(0.0, centre + (u - 0.5) * self.noise))
        if hi == lo:
            return int(round(pos * (n_rows - 1)))
        return min(range(n_rows), key=lambda r: abs((hi - gaps[r]) / (hi - lo) - pos))

    def _choose(self, ob: Observation) -> tuple[int, str, list[str]]:
        rows = ob.macro_physical.get("rows", [])
        n_rows = len(rows)
        if n_rows == 0:
            return 0, "", ["empty matrix"]
        rule_row = self._scripted_row(ob, n_rows)
        if self.llm is None:
            return rule_row, "rule", [f"scripted~{rule_row}/{n_rows - 1}"]

        try:
            resp = self.llm(build_tajfel_prompt(ob, self.prompt_lang))
        except Exception as exc:                              # noqa: BLE001 - never fail a step on one agent
            _log.warning("LLM call failed for %s: %s", ob.agent_id, exc)
            return rule_row, "fallback", ["llm error -> scripted"]
        dec = parse_tajfel_reply(resp, n_rows)
        if dec is None:
            _log.warning("unparsed reply from %s, using the scripted row: %r", ob.agent_id, resp[:300])
            return rule_row, "fallback", [f"unparsed~scripted~{rule_row}"]
        return int(dec["row"]), "llm", [dec.get("reason") or f"row={dec['row']}"]

    def decide_batch(self, obs: list[Observation], memories: dict[str, Any]) -> list[Action]:
        # `memories` is intentionally unused for C1-C3: each allocation is an independent
        # single-shot decision (see build_tajfel_prompt). The D-series study reuses this model
        # with a memory-aware prompt; do NOT "fix" this without that spec.
        if self.llm is not None and len(obs) > 1:
            from concurrent.futures import ThreadPoolExecutor
            if hasattr(self.llm, "_ensure"):
                self.llm._ensure()
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(obs))) as ex:
                results = list(ex.map(lambda o: (o, self._choose(o)), obs))
        else:
            results = [(ob, self._choose(ob)) for ob in obs]

        actions: list[Action] = []
        for ob, (row, source, rationale) in results:
            rows = ob.macro_physical.get("rows", [])
            if not rows:
                continue
            r = rows[min(max(0, row), len(rows) - 1)]
            kind = ob.macro_physical.get("matrix_kind")
            own = (ob.local_physical or {}).get("own_group", "A")
            a, b = int(r["a_points"]), int(r["b_points"])
            # ingroup_points/outgroup_points are the points this choice gives the DECIDING
            # agent's own group and the other group -- not fixed matrix columns
            own_pts, other_pts = (a, b) if own == "A" else (b, a)
            actions.append(Action(
                agent_id=ob.agent_id, step=ob.step, kind="allocate_in_group_matrix",
                payload={
                    "matrix": kind,
                    "row": int(r["row"]),
                    "own_group": own,
                    "a_points": a,
                    "b_points": b,
                    "ingroup_points": own_pts,
                    "outgroup_points": other_pts,
                    "gap": own_pts - other_pts,
                    "favour": float(own_pts - other_pts),
                    "reason": (rationale[0] if rationale else ""),
                },
                source=source, rationale=rationale,
            ))
        return actions


# --- metrics ------------------------------------------------------------------------------------

TAJFEL_METRICS = [
    "mean_favour", "favour_parity", "favour_mjp", "favour_mif",
    "mean_gap", "gap_parity", "mif_favour_share",
    "mean_joint", "mean_ingroup_points", "mean_outgroup_points",
    "n_allocations", "n_agents", "fallback_rate",
]


@register("collector", "tajfel.collector")
class TajfelMetricCollector(MetricCollector):
    """The paper's dependent variable, computed per matrix type.

    Every action payload already carries the own-group-oriented gap, so `mean_favour` is the
    mean of (own-group points - other-group points) and higher is always more ingroup-favouring
    -- on every matrix type, and for every agent whichever group they were assigned to.
    `mif_favour_share` is the share of MIF choices that kept the agent's OWN group ahead; the
    uniform baseline is the share of own-favouring rows in the table (0.5 for the current one).
    """

    def collect(self, env: Any, actions: list[Action], t: int) -> dict[str, Any]:
        acts = [a for a in (actions or []) if a.kind == "allocate_in_group_matrix"]
        n = len(acts)

        def _favour(pred) -> float | None:
            sub = [float(a.payload.get("gap", 0)) for a in acts if pred(a)]
            return round(mean(sub), 4) if sub else None

        def _gap(pred) -> float | None:
            sub = [int(a.payload.get("gap", 0)) for a in acts if pred(a)]
            return round(mean(sub), 4) if sub else None

        by_mif = [a for a in acts if a.payload.get("matrix") == "mif"]
        mif_favour = sum(1 for a in by_mif if int(a.payload.get("gap", 0)) > 0)

        return {
            "mean_favour": _favour(lambda a: True),
            "favour_parity": _favour(lambda a: a.payload.get("matrix") == "parity"),
            "favour_mjp": _favour(lambda a: a.payload.get("matrix") == "mjp"),
            "favour_mif": _favour(lambda a: a.payload.get("matrix") == "mif"),
            "mean_gap": _gap(lambda a: True),
            "gap_parity": _gap(lambda a: a.payload.get("matrix") == "parity"),
            "mif_favour_share": round(mif_favour / len(by_mif), 4) if by_mif else None,
            "mean_joint": round(mean([int(a.payload.get("ingroup_points", 0))
                                      + int(a.payload.get("outgroup_points", 0))
                                      for a in acts]), 4) if acts else None,
            "mean_ingroup_points": round(mean([int(a.payload.get("ingroup_points", 0))
                                               for a in acts]), 4) if acts else None,
            "mean_outgroup_points": round(mean([int(a.payload.get("outgroup_points", 0))
                                                for a in acts]), 4) if acts else None,
            "n_allocations": n,
            "n_agents": len({a.agent_id for a in acts}),
            "fallback_rate": round(sum(1 for a in acts if a.source == "fallback") / n, 4) if n else 0.0,
        }

    def columns(self) -> list[str]:
        return list(TAJFEL_METRICS)


# --- artifact factory (what the runner writes to disk) ------------------------------------------


def make_tajfel_bundles(
    study_id: str = "tajfel",
    *,
    condition: str = DEFAULT_CONDITION,
    n_agents: int = 10,
    n_steps: int = 3,
    seed: int = 42,
    matrices: list[str] | None = None,
) -> tuple[StudySpec, EnvironmentBundle, PopulationBundle, SimulationConfig]:
    """Build the four artifacts for one condition arm at one pool size.

    The condition only reaches the providers as `provider_args` -- one study directory serves all
    three arms, exactly as intended.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}, got {condition!r}")
    matrices = list(matrices or ["parity", "mjp", "mif"])
    label_visible = condition in ("c1", "c3")

    study = StudySpec(
        study_id=study_id,
        title=f"Tajfel minimal group paradigm ({condition}: {CONDITION_LABELS[condition]})",
        research_question=(
            "Does mere social categorization produce ingroup favouritism in resource "
            "allocation, and does it depend on the category being salient / the group being a "
            "minority?"),
        hypothesis=(
            "H1: with a visible label (c1) agents favour the ingroup (mean_favour > 0 on the "
            "parity matrix, and mif_favour_share > 0.5 -- they give up joint profit to keep the "
            "ingroup ahead). H2: hiding the label (c2) removes or shrinks the bias. "
            "H3: the rarity arm (c3) does not change the direction, only the size, of the bias."),
        study_type="longitudinal", n_steps=n_steps, seed=seed,
        time_unit="abstract",
        step_meaning="one allocation decision over one matrix type; steps rotate parity -> MJP -> MIF",
        metrics=list(TAJFEL_METRICS),
        metric_descriptions={
            "mean_favour": "mean (own-group points - other-group points) over all choices; > 0 = the agent's OWN group was favoured",
            "favour_parity": "own-group points gap on the parity (equal joint-profit) matrix",
            "favour_mjp": "own-group points gap on the maximum-joint-profit matrix",
            "favour_mif": "own-group points gap on the maximum-ingroup-difference matrix (favouritism costs joint profit here)",
            "mean_gap": "alias of mean_favour, retained for readability in the exported CSV",
            "gap_parity": "own-group points gap on the parity matrix",
            "mif_favour_share": "share of MIF choices that kept the agent's OWN group ahead; uniform baseline 0.5",
            "mean_joint": "mean total points allocated to the pair (own-payoff cost of favouritism)",
            "fallback_rate": "share of LLM replies that failed to parse and used the scripted rule; >0.10 invalidates the run",
            "n_allocations": "allocations recorded this step (= agents that answered)",
            "n_agents": "distinct agents that acted this step",
        },
        display_metrics=["mean_favour", "favour_parity", "favour_mif", "mif_favour_share",
                         "fallback_rate"],
        domain="social-psychology",
        tags=["social-psychology", "minimal-group", "tajfel", "ingroup-bias", "from-scratch",
              f"condition-{condition}"],
        legacy_simulator="from_scratch",
        provider_refs=["tajfel.env", "tajfel.pop", "tajfel.decision", "tajfel.collector"],
        adjustable_params=["condition", "n_agents", "n_steps", "matrices", "matrix_bias",
                           "noise", "label_gain", "rarity_gain", "llm_kind", "prompt_lang"],
        status="draft",
        maintainer="dsh",
    )

    env_bundle = EnvironmentBundle(
        study_id=study_id, provider_ref="tajfel.env",
        provider_args={"n_agents": n_agents, "condition": condition,
                       "matrices": matrices, "label_visible": label_visible},
        layers=[
            EnvironmentLayer(name="allocation_matrix", modality="physical", scope="macro",
                             dynamics="scheduled",
                             description="the 13-point payoff matrix currently on the desk"),
            EnvironmentLayer(name="own_group", modality="physical", scope="local",
                             dynamics="static",
                             description="this agent's group + the condition's label visibility"),
            EnvironmentLayer(name="framing", modality="information", scope="macro",
                             dynamics="scheduled",
                             description="the condition's one-line restatement of the cover story"),
        ],
        scheduled_events=[
            ScheduledEvent(at_step=1, target_layer="allocation_matrix", op="set",
                           property_name="matrix_kind", value=matrices[0],
                           note="t=1 opens with the first matrix type"),
        ],
        information_program=InformationProgram(broadcasts=[
            Broadcast(message_id=f"{condition}-framing", channel="instructions", at_step=0, ttl=99,
                      audience="all", content=CONDITION_FRAMING[condition],
                      note=f"condition {condition}: {CONDITION_LABELS[condition]}"),
        ]),
    )

    pop_bundle = PopulationBundle(
        study_id=study_id, provider_ref="tajfel.pop",
        provider_args={"n_agents": n_agents, "condition": condition},
        interaction=InteractionStructure(kind="none"),   # independent decisions by design
        propagation=PropagationMode.INDEPENDENT,
    )

    sim_config = SimulationConfig(
        study_id=study_id, n_steps=n_steps, seed=seed,
        decision_ref="tajfel.decision",
        decision_args={
            "matrix_bias": 0.65, "noise": 0.5, "label_gain": 0.3, "rarity_gain": 0.3,
            "llm_kind": "scripted", "prompt_lang": "en", "max_workers": 8,
        },
        collector_ref="tajfel.collector",
        interaction_rounds=1,
        engine_args={"memory_window": 4},
    )
    return study, env_bundle, pop_bundle, sim_config
