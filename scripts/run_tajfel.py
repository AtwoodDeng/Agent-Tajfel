#!/usr/bin/env python3
"""Run the Tajfel minimal-group study for one condition arm and export its data.

One study, three arms: the condition is passed to the providers at run time, so this script
mutates `provider_args` / `decision_args` in memory and re-builds the simulator -- no engine
change, no per-arm study directory.

    python scripts/run_tajfel.py --condition c1 --llm scripted --n-agents 10
    python scripts/run_tajfel.py --condition c1 --llm openai   --n-agents 10 --n-steps 3
    python scripts/run_tajfel.py --all --llm scripted --n-agents 10      # all three arms

Outputs land under `--out` (default `runs/<condition>[_<tag>]/`):
    study.yaml, environment/environment.json, population/population.json,
    simulation/simulation.json, population/roster.jsonl,
    trajectory/study.duckdb, trajectory/metrics_history.json,
    export/allocations.csv, export/agents.csv, export/metrics.csv, export/summary.json

`--dedicated-study` runs inside its own copy of the study dir (`studies/tajfel_<condition>`) so
each arm has an independent trajectory store and the standard `study_paths()` layout.
By default every arm writes its own subdirectory under `<repo>/studies/tajfel/runs/`, so the
three arms never overwrite each other.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

# This is a STANDALONE study project: it consumes the installed `socioverse` library
# (pip install -e /path/to/SocioVerse2) rather than living inside the SocioVerse2 repo.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tajfel as tajfel_study  # noqa: E402,F401  (registers the four providers)
from tajfel.model import (  # noqa: E402
    CONDITION_LABELS,
    CONDITIONS,
    matrix_for,
    row_count,
)
from socioverse.engine import build_simulator  # noqa: E402
from socioverse.schemas import SimulationConfig  # noqa: E402
from socioverse.validation import write_artifact  # noqa: E402

# Outputs land under this project's own runs/ directory (not a sibling repo's studies/ tree).
STUDIES_ROOT = PROJECT_ROOT / "runs"
REF_STUDY_ID = "tajfel"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--condition", choices=CONDITIONS, default="c1",
                    help="c1 = label visible, c2 = label hidden, c3 = rarity (minority group)")
    ap.add_argument("--all", action="store_true", help="run every condition in sequence")
    ap.add_argument("--llm", choices=["scripted", "openai"], default="scripted",
                    help="scripted = offline noisy stub (pipeline smoke test); openai = real LLM calls")
    ap.add_argument("--n-agents", type=int, default=10,
                    help="pool size (10 = smoke test, 50 = full design)")
    ap.add_argument("--n-steps", type=int, default=3,
                    help="steps = allocation rounds; steps rotate parity -> MJP -> MIF")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default=None, help="LLM model id (default: SV_LLM_MODEL)")
    ap.add_argument("--prompt-lang", choices=["en", "zh"], default="en")
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--matrix-bias", type=float, default=0.65,
                    help="scripted stub's drift toward the ingroup-difference end (0..1)")
    ap.add_argument("--noise", type=float, default=0.5, help="scripted stub per-agent noise (0..1)")
    ap.add_argument("--out", default=None,
                    help="output dir (default studies/tajfel/runs/<condition>[_<llm>]<tag>)")
    ap.add_argument("--tag", default=None, help="suffix for the output dir, e.g. 'smoke'")
    ap.add_argument("--no-export", action="store_true", help="skip the CSV/summary export")
    return ap.parse_args(argv)


def run_dir_for(args: argparse.Namespace, condition: str) -> Path:
    if args.out:
        base = Path(args.out).expanduser().resolve()
        return base / condition if args.all else base
    name = condition if args.llm == "scripted" else f"{condition}_{args.llm}"
    if args.tag:
        name = f"{name}_{args.tag}"
    return STUDIES_ROOT / name


def build_config(args: argparse.Namespace, condition: str
                 ) -> tuple[object, object, object, SimulationConfig]:
    """Fresh bundles for one arm, with the run-time switches folded into the args."""
    study, env_bundle, pop_bundle, sim = tajfel_study.make_tajfel_bundles(
        study_id=REF_STUDY_ID,
        condition=condition,
        n_agents=args.n_agents,
        n_steps=args.n_steps,
        seed=args.seed,
    )
    # arm-specific provider args (the condition flag the providers read)
    sim = sim.model_copy(update={
        "decision_args": {
            **sim.decision_args,
            "matrix_bias": args.matrix_bias,
            "noise": args.noise,
            "llm_kind": args.llm,
            "prompt_lang": args.prompt_lang,
            "max_workers": args.max_workers,
            **({"model": args.model} if args.model else {}),
        }
    })
    return study, env_bundle, pop_bundle, sim


def write_study_artifacts(out: Path, study, env_bundle, pop_bundle, sim: SimulationConfig) -> None:
    write_artifact(out / "study.yaml", study)
    write_artifact(out / "environment" / "environment.json", env_bundle)
    write_artifact(out / "population" / "population.json", pop_bundle)
    write_artifact(out / "simulation" / "simulation.json", sim)


def export_rows(out: Path, pairs: list[tuple[int, str, str, int, int, int, int]]
                ) -> tuple[int, Path]:
    """Write allocations.csv: one row per (step, agent, matrix) allocation choice."""
    dest = out / "export" / "allocations.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step", "agent_id", "matrix", "row", "ingroup_points", "outgroup_points", "gap"])
        for rec in pairs:
            w.writerow(rec)
    return len(pairs), dest


def read_metrics(out: Path) -> list[dict]:
    p = out / "trajectory" / "metrics_history.json"
    if not p.exists():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    return data.get("rows", []) if isinstance(data, dict) else []


def export_metrics_csv(out: Path, rows: list[dict]) -> Path | None:
    if not rows:
        return None
    cols: list[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    if "step" in cols:
        cols = ["step"] + [c for c in cols if c != "step"]
    dest = out / "export" / "metrics.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})
    return dest


def read_store(db_path: Path) -> tuple[list[dict], list[dict]]:
    """Pull the panel + metrics tables out of the DuckDB store (read-only)."""
    rows, metrics = [], []
    try:
        import duckdb  # type: ignore
    except ImportError:
        print("  (duckdb not importable; skipping CSV export)")
        return rows, metrics
    if not db_path.exists():
        print(f"  (no store at {db_path}; skipping CSV export)")
        return rows, metrics
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        tables = {r[0] for r in con.execute(
            "SELECT table_name FROM information_schema.tables").fetchall()}
        for name in tables:
            if "panel" in name or "trajector" in name:
                df = con.execute(f'SELECT * FROM "{name}" ORDER BY step').df()
                rows = df.to_dict("records")
            if "metric" in name:
                metrics = con.execute(f'SELECT * FROM "{name}" ORDER BY step').df().to_dict("records")
    finally:
        con.close()
    return rows, metrics


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    conds = list(CONDITIONS) if args.all else [args.condition]
    t0 = time.time()

    if args.llm == "openai":
        print("=" * 78)
        print("NOTE: openai mode spends real LLM calls "
              f"({args.n_agents} agents x {args.n_steps} steps x {len(conds)} arm(s))")
        print("=" * 78)

    results: dict[str, dict] = {}
    for cond in conds:
        out = run_dir_for(args, cond)
        print(f"\n--- {cond}: {CONDITION_LABELS[cond]} ---")
        study, env_bundle, pop_bundle, sim = build_config(args, cond)
        out.mkdir(parents=True, exist_ok=True)
        write_study_artifacts(out, study, env_bundle, pop_bundle, sim)
        print(f"  artifacts -> {out}")

        store_path = out / "trajectory" / "study.duckdb"
        if store_path.exists():
            store_path.unlink()
        simr = build_simulator(
            env_bundle=env_bundle, pop_bundle=pop_bundle, sim_config=sim, store_path=store_path)

        personas, panel0 = simr.materialize_initial()
        roster = out / "population" / "roster.jsonl"
        roster.parent.mkdir(parents=True, exist_ok=True)
        with roster.open("w", encoding="utf-8") as f:
            for p in personas:
                f.write(json.dumps({"agent_id": p.agent_id, "step": 0,
                                    "state": p.init_state,
                                    "attributes": p.attributes}, ensure_ascii=False) + "\n")
        print(f"  agents: {len(personas)}  -> {roster.name}")

        hist = simr.run()
        rows = hist.rows
        rows_p = out / "trajectory" / "metrics_history.json"
        rows_p.write_text(json.dumps({"study_id": hist.study_id, "rows": rows,
                                      "schema_columns": hist.schema_columns},
                                     indent=2, ensure_ascii=False), encoding="utf-8")

        # ---- console table ----
        print(f"  {'step':>4}  {'mean_favour':>11}  {'favour_parity':>13}  {'favour_mif':>10}  "
              f"{'mif_share':>9}  {'fallback':>8}")
        for r in rows:
            print(f"  {r.get('step', 0):>4}  {_fmt(r.get('mean_favour')):>11}  "
                  f"{_fmt(r.get('favour_parity')):>13}  {_fmt(r.get('favour_mif')):>10}  "
                  f"{_fmt(r.get('mif_favour_share')):>9}  {_fmt(r.get('fallback_rate')):>8}")

        n_alloc, csv_path = 0, None
        summary: dict = {"condition": cond, "label": CONDITION_LABELS[cond],
                         "llm": args.llm, "n_agents": args.n_agents, "n_steps": args.n_steps,
                         "seed": args.seed, "out": str(out)}
        if not args.no_export:
            panel, _ = read_store(store_path)
            allocs = []
            for rec in panel:
                pl = rec.get("action_payload") or {}
                if isinstance(pl, str):
                    pl = json.loads(pl) if pl else {}
                if rec.get("action_kind") != "allocate_in_group_matrix":
                    continue
                allocs.append((rec.get("step"), rec.get("agent_id"), pl.get("matrix"),
                               pl.get("row"), pl.get("ingroup_points"), pl.get("outgroup_points"),
                               pl.get("gap")))
            n_alloc, csv_path = export_rows(out, allocs)
            mpath = export_metrics_csv(out, rows)
            summary.update({"n_allocations": n_alloc,
                            "allocations_csv": str(csv_path),
                            "metrics_csv": str(mpath) if mpath else None})
            last = rows[-1] if rows else {}
            summary["last_step"] = {k: last.get(k) for k in
                                    ("mean_favour", "favour_parity", "favour_mjp", "favour_mif",
                                     "mean_gap", "mif_favour_share", "fallback_rate")}
            (out / "export" / "summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"  exported: {n_alloc} allocations -> {csv_path}")

        results[cond] = summary

    print(f"\nDone in {time.time() - t0:.1f} s.")
    if len(results) > 1:
        print(f"\n{'condition':<10}  {'mean_favour':>11}  {'favour_parity':>13}  "
              f"{'mif_share':>9}  {'fallback':>8}")
        for cond, s in results.items():
            ls = s.get("last_step") or {}
            print(f"{cond:<10}  {_fmt(ls.get('mean_favour')):>11}  "
                  f"{_fmt(ls.get('favour_parity')):>13}  "
                  f"{_fmt(ls.get('mif_favour_share')):>9}  {_fmt(ls.get('fallback_rate')):>8}")
    if args.llm == "scripted":
        print("\nReminder: --llm scripted is a NOISY STUB for pipeline validation only. "
              "It does not reproduce the minimal-group effect; use --llm openai for results.")
    elif any((s.get("last_step") or {}).get("fallback_rate", 0) > 0.10 for s in results.values()):
        print("\nWARNING: fallback_rate > 10% in at least one arm -> that arm is INVALID per the "
              "study's pre-registered criteria; inspect the warnings and re-run.")
    return 0


def _fmt(v) -> str:
    if v is None:
        return "-"
    try:
        return f"{float(v):.3f}"
    except (TypeError, ValueError):
        return str(v)


if __name__ == "__main__":
    raise SystemExit(main())
