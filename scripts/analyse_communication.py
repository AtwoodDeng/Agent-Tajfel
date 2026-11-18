#!/usr/bin/env python3
"""Seed-paired summary of E-series runs produced by scripts/run_communication.py.

    python scripts/analyse_communication.py --runs runs_communication
    python scripts/analyse_communication.py --runs runs_communication --compare e2-public/e2-silent

Two things this refuses to do, on purpose:

* it never pools conditions across seeds into one regression -- seeds are paired blocks and
  agents are nested inside seeds, so the seed is the unit of resampling;
* it never reports a "significant" difference from a scripted run. Scripted output is rule
  generated (see CommunicationDecisionModel._scripted_*), so it validates the pipeline and
  nothing else. The `--llm` column is printed next to every number for exactly that reason.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

METRIC_ROWS = ("mean_gap", "mif_favour_share", "group_welfare", "mean_abs_gap")


def load_runs(root: Path) -> list[dict]:
    """Discover every <condition>/seed_<n>/<timestamp>_<llm>_.../metrics.csv under `root`."""
    runs = []
    for metrics_path in sorted(root.glob("*/seed_*/**/metrics.csv")):
        d = metrics_path.parent
        cfg_path = d / "config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
        with metrics_path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        seed = int(cfg.get("seed", d.parent.name.split("_")[-1]))
        runs.append({"condition": cfg.get("condition", d.parent.parent.name), "seed": seed,
                     "llm": cfg.get("llm_kind", "scripted"), "model": cfg.get("model"),
                     "steps": int(cfg.get("steps", len(rows))), "rows": rows, "dir": d})
    return runs


def summarise(runs: list[dict]) -> list[dict]:
    """Average each metric over steps within a run, then over seeds; keep the seed count."""
    per_condition = defaultdict(lambda: defaultdict(list))
    for run in runs:
        for key in METRIC_ROWS:
            vals = [float(r[key]) for r in run["rows"] if r.get(key) not in (None, "")]
            if vals:
                per_condition[run["condition"]][key].append(statistics.fmean(vals))
    out = []
    for cond in sorted(per_condition):
        rec = {"condition": cond, "n_seeds": len(next(iter(per_condition[cond].values()))),
               "llm": sorted({r["llm"] for r in runs if r["condition"] == cond})}
        for key in METRIC_ROWS:
            vals = per_condition[cond].get(key, [])
            rec[key] = round(statistics.fmean(vals), 4) if vals else None
            rec[f"{key}_sd"] = round(statistics.stdev(vals), 4) if len(vals) > 1 else None
        out.append(rec)
    return out


def paired_compare(runs: list[dict], a: str, b: str, key: str = "mean_gap",
                   n_perm: int = 5000, rng: random.Random | None = None) -> dict:
    """Paired difference b - a over seeds, with a sign-flip permutation p-value.

    Both conditions must have been run on the same seeds; otherwise the comparison is refused
    rather than silently degraded to an unpaired one.
    """
    per = defaultdict(dict)
    for run in runs:
        vals = [float(r[key]) for r in run["rows"] if r.get(key) not in (None, "")]
        if vals:
            per[run["condition"]][run["seed"]] = statistics.fmean(vals)
    if a not in per or b not in per:
        raise KeyError(f"missing condition: {a if a not in per else b}")
    shared = sorted(set(per[a]) & set(per[b]))
    if not shared:
        raise ValueError(f"{a} and {b} share no seeds; cannot pair")
    diffs = [per[b][s] - per[a][s] for s in shared]
    rng = rng or random.Random(0)
    observed = statistics.fmean(diffs)
    hits = 0
    for _ in range(n_perm):
        stat = statistics.fmean(d if rng.random() < 0.5 else -d for d in diffs)
        if abs(stat) >= abs(observed) - 1e-12:
            hits += 1
    return {"a": a, "b": b, "metric": key, "n_pairs": len(shared), "mean_diff": round(observed, 4),
            "p_two_sided": round(hits / n_perm, 4),
            "caveat": "sign-flip permutation on " + str(len(shared)) + " seed pairs; "
                      "with so few blocks only a large effect can reach p<0.05"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Seed-paired summary of E-series runs.")
    ap.add_argument("--runs", default="runs_communication", help="directory written by run_communication.py")
    ap.add_argument("--compare", action="append", default=[],
                    metavar="A/B", help="paired comparison, e.g. --compare e2-public/e2-silent (repeatable)")
    ap.add_argument("--metric", default="mean_gap", choices=METRIC_ROWS)
    ap.add_argument("--json", action="store_true", help="emit JSON instead of tables")
    args = ap.parse_args(argv)

    root = Path(args.runs)
    runs = load_runs(root)
    if not runs:
        print(f"no runs found under {root}/ -- run scripts/run_communication.py first")
        return 1

    table = summarise(runs)
    comparisons = []
    for spec in args.compare:
        a, _, b = spec.partition("/")
        comparisons.append(paired_compare(runs, a, b, args.metric))

    if args.json:
        print(json.dumps({"summary": table, "comparisons": comparisons}, indent=2, ensure_ascii=False))
        return 0

    llms = sorted({r["llm"] for r in runs})
    print(f"{len(runs)} run(s) under {root}/   llm={','.join(llms)}")
    for metric in METRIC_ROWS:
        if metric != args.metric and metric != "mean_gap":
            continue
        print(f"\n== {metric} (mean over steps, then over seeds) ==")
        print(f"{'condition':<24}{'seeds':>6}{'mean':>9}{'sd':>9}")
        for rec in table:
            sd = "n/a" if rec[f"{metric}_sd"] is None else f"{rec[f'{metric}_sd']:.4f}"
            mean = "n/a" if rec[metric] is None else f"{rec[metric]:.4f}"
            print(f"{rec['condition']:<24}{rec['n_seeds']:>6}{mean:>9}{sd:>9}")

    for cmp_ in comparisons:
        print(f"\n== paired {cmp_['metric']}: {cmp_['b']} - {cmp_['a']} ==")
        print(f"seed pairs={cmp_['n_pairs']}  mean_diff={cmp_['mean_diff']:+.4f}  "
              f"p_perm={cmp_['p_two_sided']}")
        print(f"note: {cmp_['caveat']}")

    if any(l == "scripted" for l in llms):
        print("\nWARNING: scripted runs are rule-generated pipeline checks. "
              "Do not read any number above as behavioural evidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
