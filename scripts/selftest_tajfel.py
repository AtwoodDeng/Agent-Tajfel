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

    def __init__(self, model, aid: str, step: int, kind: str, visible: bool):
        self.agent_id = aid
        self.step = step
        pairs = model.matrix_for(kind)
        self.macro_physical = {
            "matrix_kind": kind,
            "label_visible": visible,
            "rows": [{"row": r, "ingroup_points": a, "outgroup_points": b,
                      "gap": a - b, "joint": a + b} for r, (a, b) in enumerate(pairs)],
        }
        self.macro_information = {"framing": "test framing"}
        self.local_physical = {"own_group": "ingroup", "steps_played": 0, "cum_gap": 0}


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
        if any(a < 0 or b < 0 for a, b in v):
            failures.append(f"{k}: negative payoff")
    # parity: joint profit constant, gap sweeps sign
    if len({a + b for a, b in taj.MATRIX_PARITY}) != 1:
        failures.append("parity: joint profit must be constant across rows")
    # MIF: gap constant, joint profit rising -> favouring ingroup = low joint profit
    if len({a - b for a, b in taj.MATRIX_MIF}) != 1:
        failures.append("mif: the ingroup/outgroup gap must be held CONSTANT across rows")
    if taj.ingroup_favour("mif", 0) <= taj.ingroup_favour("mif", len(taj.MATRIX_MIF) - 1):
        failures.append("mif: the first row (lowest joint profit) must count as MORE "
                        "ingroup-favouring")
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

    # --- prompt render ---
    print("\n== prompt (c2, label hidden) ==")
    ob_hidden = FakeObs(taj, "tj-004", 3, "mif", False)
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
    if "ingroup" not in prompt_v:
        failures.append("c1/c3 prompt must name the ingroup and outgroup columns")
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
            for step in (1, 2, 3):
                ob = FakeObs(taj, f"tj-{i:03d}", step, "mif", visible)
                row, src, _r = dm._choose(ob)
                if src != "rule":
                    failures.append(f"scripted mode must emit source='rule', got {src!r}")
                rows[row] += 1
                fav.append(taj.ingroup_favour("mif", row))
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
