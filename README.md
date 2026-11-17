# tajfel-study

The **Tajfel (1971) minimal-group paradigm** implemented as a *standalone* SocioVerse2 study.

This project consumes **SocioVerse2 as a library** (`from socioverse import ...`); it does not
live inside the SocioVerse2 repository. Three experimental arms share one registered provider
set, switched at run time:

| arm | condition | expectation |
|---|---|---|
| `c1` | label **visible** | favour the ingroup (classic minimal-group effect) |
| `c2` | label **hidden** ("Person A"/"Person B") | ingroup bias *disappears* (identity, not payoff) |
| `c3` | rarity (own group is the minority) | direction unchanged, size may shift |

## Install

```bash
# The runtime (pick one):
pip install socioverse                 # published package
pip install -e /path/to/SocioVerse2    # editable local checkout, for development

pip install -e ".[dev]"                # this project (optional dev extras)
```

LLM mode (`--llm openai`) also needs an API key: copy `.env.example` to `.env` and set
`SV_LLM_API_KEY` (and optionally `SV_LLM_BASE_URL` / `SV_LLM_MODEL`).

## Run

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

## Self-test (no Socioverse needed, stubs the framework imports)

```bash
python scripts/selftest_tajfel.py
```

## Reference

Tajfel, H., Billig, M. G., Bundy, R. P., & Flament, C. (1971). Social categorization and
intergroup behaviour. *European Journal of Social Psychology, 1*(2), 149–178.

See `tajfel/model.py` for the full design notes, and always read `llm_kind="scripted"` as a
pipeline-validation stub, not a confirmatory result.
