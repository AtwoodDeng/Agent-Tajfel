# D-series: multi-round feedback and peer communication

The C-series (`tajfel/model.py`) is a **single-shot** minimal-group allocator. The D-series
(`tajfel/feedback.py`) keeps that allocator untouched and adds the two channels a repeated
interaction needs: a public announcement of last round's group totals, and private message
exchange between paired participants.

Run it with `scripts/run_feedback.py`; the code registers as `tajfel_feedback.*`.

## What changes relative to C

| | C-series | D-series |
|---|---|---|
| rounds | one allocation, no history | 5 steps, each agent allocates once per step |
| feedback | none | per-agent history; optional public announcement |
| communication | none | paired private chat, 2 sub-rounds per step (except `d5`) |
| pairing | nobody is paired | mostly cross-group pairs, rotated every step |
| labels | `c1` visible / `c2` hidden | visible on `d1`,`d2`,`d5`; hidden on `d3`,`d4` |
| matrix | `mif` / `parity` / `mjp` selectable | `mif` only, fixed across arms |

Group membership is randomised by seed rather than being "first half is A", so the D-series
cannot reuse the C-series' deterministic group split.

## Arms

All six arms use `MATRIX_MIF` — the rows are the six Tajfel payoff pairs, and `gap` is always
`own_group_points - other_group_points`.

| arm | label visible | public announcement | peer chat | chat sub-rounds per step |
|---|---|---|---|---|
| `d1` | yes | full breakdown (group totals **and** the four cross cells) | yes | 2 |
| `d2` | yes | none | yes | 2 |
| `d3` | no | combined total only | yes | 2 |
| `d4` | no | none | yes | 2 |
| `d5` | yes | full breakdown | **no** | — |
| `d6` | yes | none | **no** | — |

`d5` and `d6` run with `interaction_rounds=1` instead of 3, which is what makes them the
"no conversation" arms. The four-cell breakdown (`a_to_a`, `a_to_b`, `b_to_a`, `b_to_b`) is what
`d1` and `d5` receive; `d3` deliberately gets only the pooled total, so it can see that points
exist without seeing who gave them to whom.

### The 2×2 these arms define

`d1`/`d2`/`d6`/`d5` form a complete **announcement × peer-chat** 2×2 with the label held visible
throughout, which makes it the clean factorial reading of the series:

| | peer chat | no chat |
|---|---|---|
| **announcement** | `d1` | `d5` |
| **no announcement** | `d2` | `d6` |

That gives both main effects plus an unconfounded interaction: `d1` − `d2` − `d5` + `d6`.

This is why `d6` exists. The same 2×2 can also be filled by `d1`/`d2`/`d4`/`d5`, but `d4` hides
the label, so its "no announcement, no chat" cell would differ from the others in two respects at
once and the interaction would be confounded with label visibility. Prefer `d6`; keep `d4` for
the label-hidden contrast it is actually for.

`d3` and `d4` sit in the same announcement × chat cells as `d1` and `d2` but with the label
hidden, so they are the label-visibility contrast rather than part of the clean 2×2. `d3`
additionally swaps the full breakdown for a pooled total, so read it as its own condition —
"hidden label, sum-only feedback, with chat" — not as a plain announcement cell.

## Pairing

Each step, the seeded RNG shuffles the two groups and builds pairs at roughly a **3 AB : 1 AA :
1 BB** ratio, covering every agent. The ratio is scaled to arbitrary `n_agents`, so `--n-agents`
can exceed 10 — the earlier "first half is A" convention is gone.

Pairing rotates every step, which means an agent's partner changes; the C-series has no analogue
of this, and it is the main thing to hold fixed when comparing D to C.

## Metrics

`FeedbackMetricCollector` emits per step:

| column | meaning |
|---|---|
| `mean_gap` | mean `own - other` over that step's allocations |
| `mif_favour_share` | share of allocations with `gap > 0` |
| `group_welfare` | total payoff normalised into 0..1 against the matrix' min/max joint total |
| `mean_gap_samepair` | mean gap for agents whose partner is in their own group (AA/BB) |
| `mean_gap_crosspair` | mean gap for agents in a cross-group (AB) pair |
| `n_messages`, `mean_message_len` | chat volume |
| `fallback_rate` | share of that step's LLM calls that fell back to the scripted rule |
| `n_allocations` | allocations recorded this step |

`group_welfare` was added because `mean_gap` alone cannot distinguish "everybody favours their
own group" from "everybody picked a low-payoff row". It is normalised **within the matrix used**,
so it is comparable across D arms and not across series that use a different matrix (see the E4
note in [E-series](E-series.md)).

`fallback_rate` is the column to check first on any real run: a high value means the model's
replies were unparseable and the numbers largely reflect the scripted rule, not the model.

## Outputs

`runs_feedback/<arm>/` plus `export/`:

```
allocations.csv   arm, step, agent_id, row, own_group, pair_type, reason, gap, ...
messages.csv      step, round, sender, recipient, pair_type, content, truncated, source
scores.csv        step, group_a_received, group_b_received, a_to_a, a_to_b, b_to_a, b_to_b, total_sum
announcements.csv step, arm, text, is_sum_only, has_cross
pairings.csv      step, agent_a, agent_b, pair_type, round_rotation
metrics.csv       the collector columns above
```

`own_group` is `null` on the hidden-label arms by design — the environment withholds it rather
than asking the prompt to ignore it.

## Reading a scripted run

`--llm scripted` generates rule-based choices and rule-based messages. It exercises the schema,
the pairing and the export path. **It says nothing about ingroup bias.** `FeedbackDecisionModel`
falls back to the same rule when a real reply fails to parse, which is why `fallback_rate` is
exported next to the metrics.

## Commands

```bash
# all six arms, pipeline only
python scripts/run_feedback.py --arm all --llm scripted --steps 5 --seed 42

# one arm, real LLM (spends API calls; a warning line prints the call budget)
python scripts/run_feedback.py --arm d1 --llm openai --steps 5 --seed 42 --n-agents 50

# schema and routing checks, no SocioVerse needed
python scripts/selftest_feedback.py
```

`runs_feedback/` is gitignored; run artifacts never go into the repository.

## Known limits

* **Scripted is not evidence.** Same rule as the C-series: pipeline validation only.
* **`mif` only.** Rarity (`c3`) and the parity/mjp matrices have no D-series counterpart, so
  the D-series cannot say anything about how the payoff table shapes the effect.
* **5 steps, 10 agents by default.** That is smoke-test scale. A confirmatory claim needs a
  seed-paired design with a power analysis, and the unit of analysis is the agent nested in the
  randomisation seed, not the individual allocation.
* **Messages are Chinese, capped at 200 characters**, and truncation is flagged rather than
  prevented. Long-message behaviour is therefore partly an artifact of the cap.
* **Pairing ratio is fixed**, not varied, so "does mixing groups in conversation matter?" is not
  answered here — that is what E2 is for.
