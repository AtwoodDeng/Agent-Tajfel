# E-series: does the *form* of communication move the allocation?

The D-series established that agents can talk and that the environment can announce results. The
E-series asks a different question: **holding the group labels, the payoff table and the
allocation task fixed, does the shape of the communication channel change what people allocate?**

Three sub-studies, eleven conditions, one shared module (`tajfel/communication.py`, registered as
`tajfel_communication.*`). Run everything through `scripts/run_communication.py`.

## Design invariants

These hold across all eleven conditions, and they are the reason the results can be attributed to
channel shape rather than to prompt phrasing:

* **Routing is enforced by the environment.** A single `Route` object decides who may speak and
  who reads each message. `apply_messages` *rejects* out-of-scope senders, wrong recipients,
  duplicate sends and stale sub-rounds. Nothing relies on asking the model to please not talk to
  the wrong person.
* **Visibility never leaks through history.** Each agent's history only ever receives messages it
  is entitled to read, and the chat prompt renders those rows verbatim. There is no shared log or
  framework memory an agent could read around the routing.
* **Within a chat sub-round, every message is generated against a snapshot taken before any of
  that sub-round's messages exist**, then all are delivered at once. Re-ordering the batch cannot
  change what anyone sees.
* **The allocation prompt never carries the leader's stance.** In E4 the stance reaches only the
  leader's *speaking* prompt. A leader who then picks the recommended row has complied on its
  own; it was not instructed at decision time.
* **The D-series is untouched.** `tajfel/feedback.py` is unchanged; E providers register
  separately and are reached only via the E script.

## E2 — who hears whom

Labels are visible in all four conditions. Group assignment, partner rotation and the allocation
task are identical; only the channel differs.

| condition | channel | readers per message |
|---|---|---|
| `e2-pair` | mixed private pairs, both directions (the D-style topology) | 1 |
| `e2-ingroup` | shared ingroup channel, each member posts once | own group only |
| `e2-public` | public bulletin, each member posts once | every other member |
| `e2-silent` | no communication at all (supplementary baseline) | 0 |

This is the primary comparison in the series: private-and-mixed vs ingroup-only vs fully public
vs silence, with the number of speakers held constant at one post per member per sub-round.

**Caveat.** `e2-pair` is *not* identical to unmodified `d1`: it exposes speaker group labels,
which `d1` does not surface in the same way. Treat `e2-pair` as "the D-style topology expressed
in E2's prompt format", not as a re-run of `d1`.

## E3 — one-way, asymmetric

Single condition, `e3-a-to-b`. Each round, every A member privately messages one randomly matched
B member. Critically, **both sides know the labels and both know that A can message B but B
cannot reply**. Group A is privileged by construction; group B can only listen.

This is the only condition in the series with a structural power asymmetry, so it is the one that
speaks to whether the *direction* of a channel — not its mere presence — changes allocations.

## E4 — leader commitment

Each group picks one opinion leader (seeded, and deliberately held fixed across all six arms so
the same individuals lead throughout). The leader pushes one of three stances, through one of two
channels:

| stance | instruction given to the leader |
|---|---|
| `ingroup` | favour your own group |
| `welfare` | maximise the joint total |
| `equal` | minimise the A–B difference |

| channel | who reads the leader's message |
|---|---|
| `private` | the leader's own group only (4 readers) |
| `public` | everyone except the leader (9 readers) |

That is 2 × 3 = 6 conditions: `e4-{private,public}-{ingroup,welfare,equal}`.

The leader also makes an **explicit row commitment**, and every commitment is scored: when the
target step's allocations land, `resolve_commitment` records the realised row and sets
`fulfilled = realised_row == commitment_row`. So E4 measures both *what the leader said* and
*whether the leader did it* — and lets you separate "the channel transmitted the message" from
"the audience complied".

### Why E4 has its own matrix

E4 uses a dedicated symmetric table `(16,4), (4,16), (15,13), (13,15), (13,13)` instead of
`MATRIX_MIF`, because on the D table **group B's "favour own group" row and its "maximise joint"
row coincide at `(10, 20)`** — verified target rows on `MATRIX_MIF` are B→row 5 for *both*
stances, which would collapse two conditions into one recommendation and make the stance
contrast meaningless. On the E4 matrix the stances map to distinct rows:

| stance | row for A | row for B | payoff |
|---|---|---|---|
| `ingroup` | 0 | 1 | `(16,4)` / `(4,16)` |
| `welfare` | 2 | 3 | `(15,13)` / `(13,15)` |
| `equal` | 4 | 4 | `(13,13)` |

**Consequence: E4 numbers are not directly comparable to D, E2 or E3.** Different payoff table,
different `group_welfare` normalisation. Compare E4 arms to each other, not to the rest.

### What E4 does not identify

Because all six arms include a commitment, E4 identifies **channel × stance**, not
"commitment vs no commitment". Answering the latter needs an additional no-commitment cell,
which this series does not have. The public arms also leak one extra cross-group message (the
leader's own audience is larger), so "public" differs from "private" in both audience size and
cross-group exposure.

## Convention shared with D

Same metric definitions as the D-series: `gap = own_group_points - other_group_points`,
`mean_gap`, `mif_favour_share`, and `group_welfare` normalised within the matrix in use.
200-character message cap with a `truncated` flag. Chat runs in sub-rounds 1 and 2 of each step;
there is no chat on the final step.

## Outputs

`runs_communication/<condition>/seed_<n>/<timestamp>_<llm>_.../`:

```
config.json       condition / steps / seed / n_agents / llm_kind / model
roster.csv        agent_id, group, is_leader
pairings.csv      step, sender, recipient, sender_group, recipient_group
allocations.csv   condition, series, step, agent_id, own_group, is_leader, row, a_points, b_points,
                  ingroup_points, outgroup_points, gap, reason, source
messages.csv      message_id, condition, series, step, round, sender, sender_group, channel,
                  recipients, n_recipients, char_len, truncated, content, source,
                  stance, issued_step, target_step, expected_row, commitment_row,
                  realised_row, fulfilled
deliveries.csv    message_id, step, round, recipient, recipient_group, delivered_step
rejections.csv    step, round, sender, reason
scores.csv        step, group_a_received, group_b_received, AA, AB, BA, BB, total_sum
announcements.csv step, condition, text
metrics.csv       the collector columns
```

`deliveries.csv` is the audit trail for the routing invariant: a message that reached someone it
should not have is visible there as an extra row. `rejections.csv` is the converse — attempted
sends the environment refused. On a clean run every rejection reason should be a genuine
out-of-scope attempt, never a silently accepted leak.

## Commands

```bash
# everything, pipeline only, three seeds
python scripts/run_communication.py --condition all --llm scripted --steps 3 --seeds 0-2

# one series
python scripts/run_communication.py --series e4 --llm openai --steps 5 --seeds 0-9 --n-agents 10

# one condition, real engine driver
python scripts/run_communication.py --condition e2-public --llm openai --driver engine \
    --steps 5 --seeds 0-9 --out runs_communication

# checks that need no SocioVerse install
python scripts/selftest_communication.py

# seed-paired summary + permutation test
python scripts/analyse_communication.py --runs runs_communication \
    --compare e2-public/e2-silent --metric mean_gap
```

Real runs need the `socioverse` package importable (`--driver engine`); `--driver standalone`
runs the same provider set without the engine, which is what the selftest uses.

`runs_communication/` is gitignored.

## Reading these results honestly

* **Scripted mode is a routing validator, nothing more.** `--llm scripted` generates rule-based
  rows and rule-based messages. Every number from a scripted run is an artifact of
  `_scripted_row`/`_scripted_message`. `scripts/analyse_communication.py` prints a warning on
  every invocation for exactly this reason, and refuses to pool conditions across seeds.
* **The seed is the unit of resampling**, and agents are nested inside seeds. With a handful of
  seeds only a large effect can reach significance; the permutation test reports the pair count
  alongside the p-value so a five-seed run is visibly weak rather than quietly over-claimed.
* **10 agents is smoke-test scale.** A confirmatory design needs an `n_agents` and seed count
  justified by a power analysis, not the defaults.
* **`fallback_rate` first.** On any real run, check that column before interpreting anything
  else: high fallback means the numbers are largely the scripted rule.
* **Where the design has a built-in confound, it is written down above** — `e2-pair` vs `d1`,
  E4's own matrix, E4's public arm leaking one extra message, and E4's inability to identify
  "commitment vs none".

## Relationship to the plans

The three implementation plans (E2/E3/E4) live in `Plan/` alongside this document's source
material and are **not** part of the repository — see the note in the top-level README.
