"""Local unit test for the pure logic in tajfel/model.py.

Socioverse is NOT importable on Windows, so the framework imports are stubbed out; only the
matrix maths, group assignment, the scripted chooser and the prompt renderer are exercised.
Run with:  python scripts/selftest_tajfel.py
"""

from __future__ import annotations

import collections
import importlib.util
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _stub_socioverse() -> None:
    for name in ["socioverse", "socioverse.abc", "socioverse.engine", "socioverse.schemas"]:
        sys.modules[name] = types.ModuleType(name)

    class Dummy:
        def __init__(self, *a, **k):
            for key, val in k.items():
                setattr(self, key, val)
            if a:
                self._args = a

    abc = sys.modules["socioverse.abc"]
    for n in ("DecisionModel", "EnvironmentProvider", "MetricCollector", "PopulationProvider"):
        setattr(abc, n, Dummy)
    sys.modules["socioverse.engine"].register = lambda kind, key: (lambda cls: cls)
    schemas = sys.modules["socioverse.schemas"]
    for n in ("Action", "Broadcast", "EnvironmentBundle", "EnvironmentLayer",
              "InformationProgram", "InteractionStructure", "Observation", "Persona",
              "PopulationBundle", "PropagationMode", "ScheduledEvent", "SimulationConfig",
              "StudySpec"):
        setattr(schemas, n, Dummy)


def load_model():
    _stub_socioverse()
    spec = importlib.util.spec_from_file_location("tajfel_model", HERE.parent / "tajfel" / "model.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["tajfel_model"] = mod
    spec.loader.exec_module(mod)
    return mod


class FakeObs:
    """Stand-in for a socioverse Observation carrying one Tajfel matrix."""

    def __init__(self, model, aid: str, step: int, kind: str, visible: bool,
                 condition: str = "c1", own_group: str = "A"):
        self.agent_id = aid
        self.step = step
        pairs = model.matrix_for(kind)
        self.macro_physical = {
            "matrix_kind": kind,
            "label_visible": visible,
            "rows": [{"row": r, "a_points": a, "b_points": b,
                      "gap": a - b, "joint": a + b} for r, (a, b) in enumerate(pairs)],
        }
        n_a, n_b = model.group_sizes(10, condition)
        n_own = n_a if own_group == "A" else n_b
        framing = model.CONDITION_FRAMING[condition]
        if condition == "c3" and n_own > 10 - n_own:
            framing = model.CONDITION_FRAMING["c3_majority"]
        self.macro_information = {"framing": framing, "condition": condition}
        self.local_physical = {"own_group": own_group,
                               "other_group": model.other_group(own_group),
                               "own_group_size": n_own,
                               "other_group_size": 10 - n_own,
                               "steps_played": 0, "cum_gap": 0}


def check_runner_args(failures: list[str], taj) -> None:
    """Exercise run_tajfel's arg parsing and output-dir naming (no socioverse needed if we stop
    before the framework imports are used)."""
    print("\n== runner args ==")
    src = (HERE / "run_tajfel.py").read_text(encoding="utf-8")
    ns: dict = {"__file__": str(HERE / "run_tajfel.py"), "__name__": "run_tajfel_probe",
                "Path": Path, "__builtins__": __builtins__,
                # the names run_tajfel imports from the study module, supplied directly
                "CONDITIONS": taj.CONDITIONS,
                "CONDITION_LABELS": taj.CONDITION_LABELS,
                "LLM_PROVIDERS": taj.LLM_PROVIDERS,
                "matrix_for": taj.matrix_for,
                "row_count": taj.row_count,
                "STUDIES_ROOT": HERE, "REF_STUDY_ID": "tajfel"}
    start = src.index("def parse_args")
    end = src.index("def build_config")
    body = "import argparse\n" + src[start:end]
    try:
        exec(compile(body, "run_tajfel_args", "exec"), ns)
    except Exception as exc:                              # noqa: BLE001
        failures.append(f"runner: cannot exec arg block: {exc}")
        return

    cases = [
        ([], ("c1", "scripted", 10, 3, 42)),
        (["--llm", "openai", "--n-agents", "10"], ("c1", "openai", 10, 3, 42)),
        (["--condition", "c3", "--n-steps", "5"], ("c3", "scripted", 10, 5, 42)),
    ]
    for argv, want in cases:
        p = ns["parse_args"](argv)
        got = (p.condition, p.llm, p.n_agents, p.n_steps, p.seed)
        ok = got == want
        print(f"  {'OK ' if ok else 'FAIL'} {argv} -> {got}")
        if not ok:
            failures.append(f"runner args: {argv} -> {got}, want {want}")

    # --all must give each arm its own directory
    p_all = ns["parse_args"](["--all", "--out", str(HERE / "_probe_out")])
    c1 = ns["run_dir_for"](p_all, "c1")
    c2 = ns["run_dir_for"](p_all, "c2")
    print(f"  --all c1 -> {c1.name} / c2 -> {c2.name}")
    if c1 == c2:
        failures.append("runner: --all must allocate a separate dir per condition")
    p_tag = ns["parse_args"](["--llm", "openai", "--tag", "smoke"])
    tagged = ns["run_dir_for"](p_tag, "c1").name
    print(f"  tagged dir -> {tagged}")
    if "smoke" not in tagged or "openai" not in tagged:
        failures.append(f"runner: tagged dir {tagged!r} should show llm+tag")


def main() -> int:
    taj = load_model()
    failures = []
    check_runner_args(failures, taj)

    # --- matrix shape ---
    print("== matrices ==")
    for k, v in taj.MATRICES.items():
        gaps = [a - b for a, b in v]
        joints = [a + b for a, b in v]
        print(f"  {k:7} rows={len(v):2}  gaps={gaps}")
        print(f"  {'':7}              joints={joints}")
        if len(v) < 2:
            failures.append(f"{k}: needs >= 2 rows")
        valence = taj.MATRIX_LIBRARY.get(k, {}).get("valence", "gain")
        if valence == "gain" and any(a < 0 or b < 0 for a, b in v):
            failures.append(f"{k}: negative payoff in a gain matrix")
    # parity: joint profit constant, gap sweeps sign
    if len({a + b for a, b in taj.MATRIX_PARITY}) != 1:
        failures.append("parity: joint profit must be constant across rows")
    # MIF: the gap must MOVE (a constant gap makes "max difference" and "fairness" the same
    # choice) and must run OPPOSITE to the joint profit, so favouritism costs absolute payoff
    mif_gaps = [a - b for a, b in taj.MATRIX_MIF]
    mif_joints = [a + b for a, b in taj.MATRIX_MIF]
    if len(set(mif_gaps)) < 2:
        failures.append("mif: the gap must vary across rows or max-difference is unmeasurable")
    if not all(y < x for x, y in zip(mif_gaps, mif_gaps[1:])):
        failures.append(f"mif: the gap must fall from the max-difference row, got {mif_gaps}")
    if not all(y > x for x, y in zip(mif_joints, mif_joints[1:])):
        failures.append(f"mif: joint profit must rise toward the last row, got {mif_joints}")
    # a row that is >= another on BOTH columns makes the matrix a pure payoff-maximiser test
    dominated = [i for i, (a, b) in enumerate(taj.MATRIX_MIF)
                 if any(c >= a and d >= b and (c, d) != (a, b) for c, d in taj.MATRIX_MIF)]
    if dominated:
        failures.append(f"mif: rows {dominated} are weakly dominated on both columns -- the "
                        f"matrix would measure 'take the most points', not favouritism")
    if taj.ingroup_favour("mif", 0, "A") <= taj.ingroup_favour("mif", len(taj.MATRIX_MIF) - 1, "A"):
        failures.append("mif: the top row (max difference) must count as MORE ingroup-favouring")
    # perspective: the same table must read the other way round for a group-B agent
    if taj.ingroup_favour("mif", 0, "B") >= taj.ingroup_favour("mif", len(taj.MATRIX_MIF) - 1, "B"):
        failures.append("mif: for a group-B agent the bottom row must be the favouring end")

    # every matrix must let the four Tajfel strategies choose different rows
    def _strategy_rows(pairs):
        n = len(pairs)
        return {"MJP": max(range(n), key=lambda r: pairs[r][0] + pairs[r][1]),
                "MIP": max(range(n), key=lambda r: pairs[r][0]),
                "MD": max(range(n), key=lambda r: pairs[r][0] - pairs[r][1]),
                "Fair": min(range(n), key=lambda r: abs(pairs[r][0] - pairs[r][1]))}

    for k, v in taj.MATRICES.items():
        s = _strategy_rows(v)
        print(f"  {k:7} strategies={s} distinct={len(set(s.values()))}")
        if k == "mif" and len(set(s.values())) < 3:
            failures.append(f"mif: max-joint / max-own / max-diff / fair must land on >= 3 "
                            f"distinct rows, got {s}")
    if taj.allocation_gap("parity", 0) <= 0:
        failures.append("parity: first row must favour the ingroup")
    # MJP: joint profit peaks somewhere strictly inside the table
    peak = max(range(len(taj.MATRIX_MJP)), key=lambda r: sum(taj.MATRIX_MJP[r]))
    if peak in (0, len(taj.MATRIX_MJP) - 1):
        failures.append(f"mjp: joint-profit peak should be interior, got row {peak}")
    print(f"  anchors: mif_favour=row {taj.ROW_MIF_FAVOUR}, "
          f"mjp_max_joint=row {taj.ROW_MJP_MAX_JOINT}, "
          f"parity_fair=row {taj.ROW_PARITY_FAIR}")

    # --- group assignment ---
    print("\n== group sizes ==")
    for n in (10, 50):
        for c in taj.CONDITIONS:
            n_in, n_out = taj.group_sizes(n, c)
            seq = "".join(taj.group_of(i, n, c)[0] for i in range(n))
            print(f"  n={n:2} {c}: in={n_in:2} out={n_out:2}  {seq[:20]}{'...' if n > 20 else ''}")
            if n_in + n_out != n:
                failures.append(f"n={n} {c}: group sizes do not sum to n")
            if n_in == 0 or n_out == 0:
                failures.append(f"n={n} {c}: empty group")
    n_in_c3, _ = taj.group_sizes(10, "c3")
    if n_in_c3 != 2:
        failures.append(f"c3 with n=10 should make a 2-person minority, got {n_in_c3}")

    # --- condition wiring: the condition must actually reach the decision layer ------------
    print("\n== condition wiring ==")
    env = taj.TajfelEnvironmentProvider(types.SimpleNamespace(
        provider_args={"n_agents": 10, "condition": "c3"}, scheduled_events=[]))
    seen = {o.agent_id: o for o in env.observe_batch(["tj-000", "tj-005"], 1)}
    mnr, mjr = seen["tj-000"], seen["tj-005"]
    print(f"  minority framing: {mnr.macro_information['framing'][-90:]}")
    print(f"  majority framing: {mjr.macro_information['framing'][-90:]}")
    if mnr.local_physical["own_group_size"] != 2:
        failures.append("c3: tj-000 must sit in the 2-person minority")
    if "group_size" in mjr.local_physical and mjr.local_physical["own_group_size"] != 8:
        failures.append("c3: tj-005 must sit in the 8-person majority")
    if mnr.macro_information["framing"] == mjr.macro_information["framing"]:
        failures.append("c3: minority and majority agents must not get the same size framing")
    if "small" not in mnr.macro_information["framing"]:
        failures.append("c3: the minority agent must be told its group is the small one")
    if "larger" not in mjr.macro_information["framing"]:
        failures.append("c3: the majority agent must be told its group is the large one")

    # the stub must DISTINGUISH c1 from c3; when it cannot, both arms export identical data
    dm_cond = taj.TajfelDecisionModel(llm_kind="scripted")
    picks_by_cond = {}
    for cond in taj.CONDITIONS:
        picks_by_cond[cond] = [
            dm_cond._choose(FakeObs(taj, f"tj-{i:03d}", 2, "mif", cond in ("c1", "c3"),
                                    condition=cond,
                                    own_group=taj.group_of(i, 10, cond)))[0]
            for i in range(10)]
        print(f"  {cond} rows={picks_by_cond[cond]}")
    if picks_by_cond["c1"] == picks_by_cond["c3"]:
        failures.append("scripted stub cannot tell c1 from c3 -- the rarity condition never "
                        "reaches the decision layer")
    if picks_by_cond["c1"] == picks_by_cond["c2"]:
        failures.append("scripted stub cannot tell c1 from c2 -- label_visible is not wired")

    # --- prompt render ---
    print("\n== prompt (c2, label hidden) ==")
    ob_hidden = FakeObs(taj, "tj-004", 3, "mif", False, condition="c2")
    prompt = taj.build_tajfel_prompt(ob_hidden, "en")
    print(prompt)
    if "Person A" not in prompt:
        failures.append("c2 prompt must present the anonymous labels")
    # the c2 arm leaks the manipulation if any of these words reach the agent
    for leaked in ("ingroup", "outgroup", "Left column = your"):
        if leaked in prompt:
            failures.append(f"c2 prompt leaks the category via {leaked!r}")
    zh = taj.build_tajfel_prompt(ob_hidden, "zh")
    if "行" not in zh:
        failures.append("zh prompt missing")
    if "内群体" in zh:
        failures.append("zh c2 prompt leaks the category")

    ob_shown = FakeObs(taj, "tj-004", 3, "mif", True)
    prompt_v = taj.build_tajfel_prompt(ob_shown, "en")
    if "BLOTQW" not in prompt_v or "BTSLAB" not in prompt_v:
        failures.append("c1/c3 prompt must name the two concrete group columns")
    # the visible prompt must present the matrix from THIS agent's own point of view
    p_a = taj.build_tajfel_prompt(FakeObs(taj, "tj-000", 3, "mif", True,
                                          condition="c1", own_group="A"), "en")
    p_b = taj.build_tajfel_prompt(FakeObs(taj, "tj-001", 3, "mif", True,
                                          condition="c1", own_group="B"), "en")
    if p_a == p_b:
        failures.append("c1/c3 prompt must differ for group-A and group-B agents (own-group "
                        "perspective)")
    if "BLOTQW 15  BTSLAB 5" not in p_a:
        failures.append("c1: a group-A agent must see row 0 as BLOTQW 15 / BTSLAB 5")
    if "BTSLAB 5  BLOTQW 15" not in p_b:
        failures.append("c1: a group-B agent must see row 0 as BTSLAB 5 / BLOTQW 15 "
                        "(own-group perspective)")
    print("\n== prompt (c1, label visible) ==")
    print(prompt_v)

    # --- reply parser ---
    print("\n== reply parser ==")
    cases = [('{"row": 3, "reason": "fair"}', 3), ('row 5', 5), ('I choose row #2.', 2),
             ('no idea', None), ('{"row": 99}', None), ("", None)]
    for text, want in cases:
        got = taj.parse_tajfel_reply(text, 6)
        got_row = None if got is None else got["row"]
        ok = got_row == want
        print(f"  {'OK ' if ok else 'FAIL'} {text!r:40} -> {got_row} (want {want})")
        if not ok:
            failures.append(f"parser: {text!r} -> {got_row}, want {want}")

    # --- scripted chooser: the stub must show a label effect and must not be a constant ---
    # Note: the draw is deterministic in (agent, step, matrix), so a single agent must always land
    # on the same row -- spread has to be measured across the POOL, not by re-sampling one id.
    print("\n== scripted chooser (10 agents x 3 steps, mif matrix) ==")
    dm = taj.TajfelDecisionModel(llm_kind="scripted")
    favours, distinct = {}, {}
    for visible in (True, False):
        rows, fav = collections.Counter(), []
        for i in range(10):
            own = taj.group_of(i, 10, "c1")
            for step in (1, 2, 3):
                ob = FakeObs(taj, f"tj-{i:03d}", step, "mif", visible,
                             condition="c1", own_group=own)
                row, src, _r = dm._choose(ob)
                if src != "rule":
                    failures.append(f"scripted mode must emit source='rule', got {src!r}")
                rows[row] += 1
                fav.append(taj.ingroup_favour("mif", row, own))
        favours[visible] = sum(fav) / len(fav)
        distinct[visible] = len(rows)
        print(f"  visible={visible}: rows={dict(sorted(rows.items()))} "
              f"mean_favour={favours[visible]:.2f}")
    if favours[True] <= favours[False]:
        failures.append("scripted stub: a visible label should raise ingroup favouring "
                        f"(got {favours[True]:.2f} vs {favours[False]:.2f})")
    if abs(favours[True] - favours[False]) < 0.3:
        failures.append("scripted stub: label effect too small to be visible in a smoke test")
    for visible in (True, False):
        if distinct[visible] < 2:
            failures.append(f"scripted stub collapsed to a single row across the pool "
                            f"(label_visible={visible}) -- noise is not reaching the draw")

    # a single agent is stable across instances -> re-runs reproduce the same choices
    solo = [taj.TajfelDecisionModel(llm_kind="scripted")._choose(
        FakeObs(taj, "tj-004", 2, "parity", True))[0] for _ in range(5)]
    print(f"\n== single-agent stability ==\n  tj-004 picks={solo}")
    if len(set(solo)) != 1:
        failures.append("the same agent must pick the same row across re-runs")

    pool_rows = [dm._choose(FakeObs(taj, f"tj-{i:03d}", 2, "parity", True))[0] for i in range(10)]
    print(f"  pool spread (10 agents, parity, c1)={pool_rows}")
    if len(set(pool_rows)) < 2:
        failures.append("a 10-agent pool must not land on one single row")

    # --- determinism: same seed -> same choice ---
    a = taj.TajfelDecisionModel(llm_kind="scripted")
    picks1 = [a._choose(FakeObs(taj, f"tj-{i:03d}", 2, "parity", True))[0] for i in range(10)]
    picks2 = [taj.TajfelDecisionModel(llm_kind="scripted")._choose(
        FakeObs(taj, f"tj-{i:03d}", 2, "parity", True))[0] for i in range(10)]
    print(f"\n== determinism ==\n  picks={picks1}\n  {'OK' if picks1 == picks2 else 'FAIL'}")
    if picks1 != picks2:
        failures.append("scripted chooser is not deterministic across instances")

    # --- multi-round integration: the full env->decision->collector loop over 5 steps ---------
    # The single-probe checks above cannot catch a broken matrix_at rotation, a history that
    # never accumulates, or a collector that drifts across steps -- so run the real chain.
    print("\n== multi-round integration (c1, mif only, 5 steps x 10 agents) ==")
    import types as _t
    env2 = taj.TajfelEnvironmentProvider(_t.SimpleNamespace(
        provider_args={"n_agents": 10, "condition": "c1", "matrices": ["mif"]},
        scheduled_events=[]))
    dm2 = taj.TajfelDecisionModel(llm_kind="scripted")
    coll = taj.TajfelMetricCollector()
    agents10 = [f"tj-{i:03d}" for i in range(10)]
    steps_gap_sum = []
    for t in range(1, 6):
        obs_t = env2.observe_batch(agents10, t)
        # the environment must show each agent its own accumulated history
        for ob in obs_t:
            sp = ob.local_physical.get("steps_played")
            if sp != t - 1:
                failures.append(f"step {t}: {ob.agent_id} sees steps_played={sp}, want {t - 1}")
        acts_t = dm2.decide_batch(obs_t, {})
        if len(acts_t) != 10:
            failures.append(f"step {t}: decide_batch returned {len(acts_t)} actions, want 10")
        env2.apply(acts_t)
        met = coll.collect(env2, acts_t, t)
        steps_gap_sum.append(met["mean_gap"])
        print(f"  step {t}: matrix={obs_t[0].macro_physical['matrix_kind']} "
              f"mean_gap={met['mean_gap']} mif_favour_share={met['mif_favour_share']} "
              f"n_allocations={met['n_allocations']}")
        if met["n_allocations"] != 10:
            failures.append(f"step {t}: collector counted {met['n_allocations']} allocations")
        if met["fallback_rate"] not in (0.0, 0):
            failures.append(f"step {t}: scripted run must have fallback_rate 0, got "
                            f"{met['fallback_rate']}")
    # history must have accumulated exactly 5 records per agent, cum_gap consistent with gaps
    hist = env2.history
    if sorted(hist) != sorted(agents10):
        failures.append("multi-round: history does not cover exactly the 10 agents")
    for aid in agents10:
        recs = hist.get(aid, [])
        if len(recs) != 5:
            failures.append(f"multi-round: {aid} has {len(recs)} records, want 5")
            continue
        if any(r["t"] != i + 1 for i, r in enumerate(recs)):
            failures.append(f"multi-round: {aid} steps are {([r['t'] for r in recs])}")
        cg = sum(int(r["gap"]) for r in recs)
        ob_end = env2.observe_batch([aid], 6)[0]
        if ob_end.local_physical.get("cum_gap") != cg:
            failures.append(f"multi-round: {aid} cum_gap mismatch "
                            f"({ob_end.local_physical.get('cum_gap')} vs {cg})")
        # every action payload must stay internally consistent (own-perspective orientation)
        for r in recs:
            if r["gap"] != r["ingroup_points"] - r["outgroup_points"]:
                failures.append(f"multi-round: {aid} step {r['t']} payload gap inconsistent")
    print(f"  per-step mean_gap series: {steps_gap_sum}")
    # the stub legitimately re-draws each step (step is hashed into the pick), so cross-step
    # differences are fine; the real invariant is END-TO-END REPRODUCIBILITY: a brand-new
    # env + decision model must replay the exact same 5-step series from the same seed inputs.
    import types as _t2
    env3 = taj.TajfelEnvironmentProvider(_t2.SimpleNamespace(
        provider_args={"n_agents": 10, "condition": "c1", "matrices": ["mif"]},
        scheduled_events=[]))
    dm3 = taj.TajfelDecisionModel(llm_kind="scripted")
    replay = []
    for t in range(1, 6):
        acts_t = dm3.decide_batch(env3.observe_batch(agents10, t), {})
        env3.apply(acts_t)
        replay.append(coll.collect(env3, acts_t, t)["mean_gap"])
    if replay != steps_gap_sum:
        failures.append(f"multi-round: the 5-step chain is not reproducible: "
                        f"{steps_gap_sum} vs replay {replay}")
    else:
        print("  5-step chain replays identically with fresh instances: OK")
    # and the history-based draws really are per-step (not frozen by the first step's pick)
    frozen = [recs for recs in hist.values() if len({r["row"] for r in recs}) < 2]
    if len(frozen) == len(agents10):
        failures.append("multi-round: every agent repeated its step-1 row at all 5 steps -- "
                        "the draw is frozen on history and never re-samples")

    print("\n" + "=" * 70)
    if failures:
        print(f"{len(failures)} FAILURE(S):")
        for f in failures:
            print("  -", f)
        return 1
    print("all local checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
