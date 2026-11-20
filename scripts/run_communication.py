#!/usr/bin/env python3
"""Run the E-series Tajfel communication study through SocioVerse2's longitudinal engine.

    python scripts/run_communication.py --condition all --llm scripted
    python scripts/run_communication.py --series e3 --llm openai --seeds 0-9 --out runs_communication

Output layout (one directory per condition / seed / run):

    <out>/<condition>/seed_<seed>/<run_id>/
        config.json        study parameters of this run
        roster.csv         agent -> group, is_leader
        pairings.csv       per-step sender -> recipient map (E2-pair / E3 only)
        allocations.csv    one row per agent per step (+ is_leader)
        messages.csv       one row per message (+ recipients, stance, commitment_row, fulfilled)
        deliveries.csv     one row per (message, recipient) pair -- E2-pair therefore has 2 per message
        rejections.csv     routing violations refused by the environment (must stay empty)
        scores.csv         per-step group totals
        announcements.csv  the results broadcast shown before the next round
        metrics.csv        per-step metrics from the collector
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --local-stub lets an interpreter without the SocioVerse runtime still exercise this CLI; the
# stub only installs registration plumbing, never a decision rule or a routing shortcut.
if "--local-stub" in sys.argv:
    sys.argv.remove("--local-stub")
    from commshim import install  # noqa: E402  (scripts/ is on sys.path as this file's dir)
    install()

import tajfel  # noqa: F401  (registers the C-series providers the bundles reference)
from tajfel.communication import (
    CHAT_SUBROUNDS, CONDITIONS, DECISION_REF, E2_CONDITIONS, E3_CONDITIONS, E4_CONDITIONS,
    ENV_REF, POPULATION_REF, SERIES_OF, STEPS, N_AGENTS, CommunicationEnvironmentProvider,
    CommunicationDecisionModel, CommunicationMetricCollector, export_run, run_condition,
)
from tajfel.model import make_tajfel_bundles, LLM_PROVIDERS, is_llm_kind

LLM_KINDS = ("scripted", *LLM_PROVIDERS)


def _parse_seeds(text: str) -> list[int]:
    """"0" | "3,5,7" | "0-9" | "0,4-6" -> list of ints."""
    seeds: list[int] = []
    for chunk in str(text).replace(" ", "").split(","):
        if not chunk:
            continue
        if "-" in chunk[1:]:
            lo, hi = chunk.split("-", 1)
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(chunk))
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def run_via_engine(condition: str, llm: str, steps: int, seed: int, n_agents: int,
                   model: str | None, dest: Path) -> list[dict]:
    """Drive the real engine so the condition also survives SocioVerse2's own orchestration."""
    from socioverse.engine import build_simulator

    series = SERIES_OF[condition]
    study, env_bundle, pop_bundle, sim = make_tajfel_bundles(
        study_id=f"tajfel_{condition}_{series}", condition="c1", n_agents=n_agents,
        n_steps=steps, seed=seed, matrices=["mif"])
    del study
    env_bundle = env_bundle.model_copy(update={
        "provider_ref": ENV_REF,
        "provider_args": {"condition": condition, "n_agents": n_agents, "seed": seed, "steps": steps}})
    pop_bundle = pop_bundle.model_copy(update={
        "provider_ref": POPULATION_REF, "provider_args": {"n_agents": n_agents}})
    # every step has 1 allocation round + the chat sub-rounds that exist before the last step
    interaction_rounds = 1 + len(CHAT_SUBROUNDS)
    sim = sim.model_copy(update={
        "decision_ref": DECISION_REF,
        "decision_args": {"llm_kind": llm, "seed": seed, **({"model": model} if model else {})},
        "collector_ref": "tajfel_communication.collector",
        "interaction_rounds": interaction_rounds,
        "engine_args": {"memory_window": max(steps * interaction_rounds, 16)}})
    (dest / "trajectory").mkdir(parents=True, exist_ok=True)
    runner = build_simulator(env_bundle=env_bundle, pop_bundle=pop_bundle, sim_config=sim,
                             store_path=dest / "trajectory" / "study.duckdb")
    history = runner.run()
    env = runner.env
    if hasattr(env, "finalize"):
        env.finalize()
    rows = [r for r in history.rows if r.get("step", 0) > 0]
    export_run(env, rows, dest, config={"condition": condition, "series": series, "steps": steps,
                                        "seed": seed, "n_agents": n_agents, "llm_kind": llm,
                                        "model": model, "driver": "engine",
                                        "chat_subrounds": list(CHAT_SUBROUNDS)})
    return rows


def run_standalone(condition: str, llm: str, steps: int, seed: int, n_agents: int,
                   model: str | None, dest: Path) -> tuple[list[dict], object]:
    """Fallback driver: identical round loop without the engine (used when it cannot be built)."""
    env, metrics = run_condition(condition, steps=steps, seed=seed, n_agents=n_agents, llm_kind=llm,
                                 model=model, out=None)
    export_run(env, metrics, dest, config={"condition": condition, "series": SERIES_OF[condition],
                                           "steps": steps, "seed": seed, "n_agents": n_agents,
                                           "llm_kind": llm, "model": model, "driver": "standalone",
                                           "chat_subrounds": list(CHAT_SUBROUNDS)})
    return metrics, env


def _run_prefix(seed: int, llm: str, model: str | None) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = llm if not model else f"{llm}_{str(model).replace('/', '-')}"
    return f"{stamp}_{tag}_seed{seed}"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Run the E2/E3/E4 Tajfel communication conditions.")
    p.add_argument("--condition", default="all",
                   help="a condition name, 'all', or one of the series: e2 / e3 / e4")
    p.add_argument("--series", choices=["e2", "e3", "e4"], default=None,
                   help="shorthand for --condition <series>")
    p.add_argument("--llm", choices=LLM_KINDS, default="scripted")
    p.add_argument("--model", default=None, help="LLM model id override (real LLM modes only)")
    p.add_argument("--steps", type=int, default=STEPS)
    p.add_argument("--seed", type=int, default=None, help="single seed (kept for compatibility)")
    p.add_argument("--seeds", default=None, help="seed list, e.g. 0-9 or 1,4,7")
    p.add_argument("--n-agents", type=int, default=N_AGENTS)
    p.add_argument("--out", default="runs_communication")
    p.add_argument("--driver", choices=["engine", "standalone"], default="engine")
    p.add_argument("--force", action="store_true", help="allow re-running a condition/seed that exists")
    a = p.parse_args(argv)

    if a.series:
        conditions = tuple({"e2": E2_CONDITIONS, "e3": E3_CONDITIONS, "e4": E4_CONDITIONS}[a.series])
    elif a.condition == "all":
        conditions = CONDITIONS
    elif a.condition in CONDITIONS:
        conditions = (a.condition,)
    else:
        p.error(f"unknown condition {a.condition!r}; choose from {CONDITIONS}, all, e2, e3, e4")

    seeds = _parse_seeds(a.seeds) if a.seeds else [42 if a.seed is None else a.seed]
    out = Path(a.out)
    planned = [(c, s) for c in conditions for s in seeds]
    if is_llm_kind(a.llm):
        print(f"WARNING: real LLM run ({a.llm}) -- {len(planned)} condition/seed pairs, {a.steps} steps, "
              f"{a.n_agents} agents, model={a.model or 'env default'}")
    print(f"conditions={len(conditions)} seeds={seeds} driver={a.driver} llm={a.llm} "
          f"-> {out}/<condition>/seed_<seed>/")

    summary, failures = [], []
    for condition, seed in planned:
        base = out / condition / f"seed_{seed}"
        run_id = _run_prefix(seed, a.llm, a.model)
        dest = base / run_id
        if dest.exists() and not a.force:
            print(f"{condition} seed {seed}: {dest} already exists (use --force to overwrite)")
            continue
        try:
            if a.driver == "engine":
                rows = run_via_engine(condition, a.llm, a.steps, seed, a.n_agents, a.model, dest)
                env = None
            else:
                rows, env = run_standalone(condition, a.llm, a.steps, seed, a.n_agents, a.model, dest)
        except Exception as exc:  # keep the batch going; report at the end
            failures.append((condition, seed, f"{type(exc).__name__}: {exc}"))
            print(f"{condition} seed {seed}: FAILED {type(exc).__name__}: {exc}")
            continue
        last = rows[-1] if rows else {}
        summary.append({"condition": condition, "series": SERIES_OF[condition], "seed": seed,
                        "run_id": run_id, "steps": a.steps, "llm_kind": a.llm, "model": a.model,
                        "driver": a.driver,
                        "final_mean_gap": last.get("mean_gap"),
                        "final_group_welfare": last.get("group_welfare"),
                        "final_fallback_rate": last.get("fallback_rate"),
                        "n_messages": last.get("n_messages")})
        print(f"{condition} seed {seed}: mean_gap={last.get('mean_gap')} "
              f"welfare={last.get('group_welfare')} fallback={last.get('fallback_rate')}")

    if summary:
        index = out / "index.json"
        existing = []
        if index.exists():
            try:
                existing = json.loads(index.read_text(encoding="utf-8"))
            except Exception:
                existing = []
        index.write_text(json.dumps(existing + summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"summary -> {index}")
    if failures:
        print(f"{len(failures)} run(s) failed:")
        for condition, seed, msg in failures:
            print(f"  {condition} seed {seed}: {msg}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
