# tajfel-study

The **Tajfel (1971) minimal-group paradigm** implemented as a *standalone* SocioVerse2 study, in
three cumulative series:

| series | question | arms | module | docs |
|---|---|---|---|---|
| **C** | does mere categorisation produce ingroup bias? | `c1` `c2` `c3` | `tajfel/model.py` | this file |
| **D** | does repeated feedback and peer chat change it? | `d1`–`d5` | `tajfel/feedback.py` | [docs/D-series.md](docs/D-series.md) |
| **E** | does the *form* of communication change it? | 11 conditions | `tajfel/communication.py` | [docs/E-series.md](docs/E-series.md) |

The D-series keeps the C-series allocator untouched and adds a public announcement channel and
paired private chat. The E-series keeps both untouched and varies only the **shape** of the
communication channel — who hears whom, whether the flow is one-way, and whether a leader's
commitment reaches the ingroup or everyone.

This project consumes **SocioVerse2 as a library** (`from socioverse import ...`); it does not
live inside the SocioVerse2 repository. Each series registers its own provider set and is
switched at run time.

## Install

```bash
# The runtime (pick one):
pip install socioverse                 # published package
pip install -e /path/to/SocioVerse2    # editable local checkout, for development

pip install -e ".[dev]"                # this project (optional dev extras)
```

LLM mode (`--llm openai`) also needs an API key: copy `.env.example` to `.env` and set
`SV_LLM_API_KEY` (and optionally `SV_LLM_BASE_URL` / `SV_LLM_MODEL`).

## C-series: single-shot allocation

Three arms share one registered provider set, switched at run time:

| arm | condition | expectation |
|---|---|---|
| `c1` | label **visible** | favour the ingroup (classic minimal-group effect) |
| `c2` | label **hidden** ("Person A"/"Person B") | ingroup bias *disappears* (identity, not payoff) |
| `c3` | rarity (own group is the minority) | direction unchanged, size may shift |

```bash
# pipeline-only smoke test (no LLM, deterministic noisy stub):
python scripts/run_tajfel.py --condition c1 --llm scripted --n-agents 10

# real LLM run (spends API calls):
python scripts/run_tajfel.py --condition c1 --llm openai --n-agents 10 --n-steps 3

# all three arms:
python scripts/run_tajfel.py --all --llm scripted --n-agents 10
```

Outputs land under `runs/<condition>[_<tag>]/`: study/environment/population/simulation
artifacts, `roster.jsonl`, the DuckDB trajectory store, `metrics_history.json`, and CSV/summary
exports under `export/`.

## D-series: feedback and peer chat

Five arms on a fixed `mif` matrix, five steps each, most with two chat sub-rounds per step.
`d1`/`d3` vs `d2`/`d4` isolates the public announcement, `d1`/`d2` vs `d3`/`d4` isolates label
visibility, and `d5` vs `d1` isolates chat (it runs with no conversation at all).

```bash
python scripts/run_feedback.py --arm all --llm scripted --steps 5 --seed 42
python scripts/run_feedback.py --arm d1  --llm openai  --steps 5 --seed 42 --n-agents 50
python scripts/selftest_feedback.py          # no SocioVerse needed
```

Outputs land under `runs_feedback/<arm>/export/` (`allocations`, `messages`, `scores`,
`announcements`, `pairings`, `metrics`). The D-series adds `group_welfare` (payoff normalised
0..1 within the matrix) and a `fallback_rate` column you should check before trusting any real
run.

Full arm table, pairing ratio, metric definitions and limitations: **[docs/D-series.md](docs/D-series.md)**.

## E-series: the form of communication

Eleven conditions across three sub-studies, all on the same allocation task:

| series | conditions | what varies |
|---|---|---|
| **E2** | `e2-pair` `e2-ingroup` `e2-public` `e2-silent` | who hears whom (labels held constant) |
| **E3** | `e3-a-to-b` | one-way flow: every A messages a random B, B cannot reply |
| **E4** | `e4-{private,public}-{ingroup,welfare,equal}` | one leader per group × ingroup vs public audience × three stances |

Routing is decided by the environment, not by prompt wording: a message reaches exactly the
recipients the condition implies, and out-of-scope sends are rejected rather than ignored. In E4
the leader's stance appears only in the *speaking* prompt, never in the allocation prompt, so
compliance is the model's own choice — and each leader's row commitment is scored against what it
actually allocated.

```bash
python scripts/run_communication.py --condition all --llm scripted --steps 3 --seeds 0-2
python scripts/run_communication.py --series e4 --llm openai --steps 5 --seeds 0-9
python scripts/run_communication.py --condition e2-public --llm openai --driver engine --seeds 0-9
python scripts/selftest_communication.py     # no SocioVerse needed

# seed-paired summary plus a sign-flip permutation test:
python scripts/analyse_communication.py --runs runs_communication \
    --compare e2-public/e2-silent --metric mean_gap
```

Outputs land under `runs_communication/<condition>/seed_<n>/`, including `deliveries.csv` (the
audit trail proving no message reached an unentitled reader) and `rejections.csv`.

Two things worth knowing before comparing numbers across series: E4 uses its **own symmetric
matrix**, because on the D table group B's "favour own group" and "maximise joint welfare" rows
coincide — so E4 results are comparable to each other, not to D/E2/E3. And `e2-pair` is not a
re-run of `d1`; it exposes speaker group labels.

Full condition table, matrices and limitations: **[docs/E-series.md](docs/E-series.md)**.

## Self-tests (no SocioVerse needed, they stub the framework imports)

```bash
python scripts/selftest_tajfel.py          # C-series
python scripts/selftest_feedback.py        # D-series
python scripts/selftest_communication.py   # E-series
```

## How to read any result here

`--llm scripted` generates **rule-based** allocations and messages. It validates that the
providers, routing, scoring and exports work. It is not evidence about group behaviour, and no
number from a scripted run should be reported as a finding — `scripts/analyse_communication.py`
prints that warning on every invocation. Confirmatory claims need `--llm openai` with a
documented model, a seed list, and a power analysis; the unit of analysis is the agent nested in
the randomisation seed, not the individual allocation.

Run artifacts (`runs/`, `runs_feedback/`, `runs_communication/`) are gitignored. The E-series
implementation plans live in `Plan/`, which stays local and is not part of this repository.

## Reference

Tajfel, H., Billig, M. G., Bundy, R. P., & Flament, C. (1971). Social categorization and
intergroup behaviour. *European Journal of Social Psychology, 1*(2), 149–178.

See `tajfel/model.py`, `tajfel/feedback.py` and `tajfel/communication.py` for design notes.
