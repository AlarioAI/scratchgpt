# memory/

Scientific lab notebook for ScratchGPT. Committed to the repo so the record is shared, permanent, and versioned alongside the code.

This directory answers questions the code and `git log` can't:

- **Why** did we run this experiment?
- **What** did we expect to find vs. what we actually found?
- **What** does the reader (future us, a collaborator, an LLM agent, a blog writer) need to know that isn't obvious from the metrics?

If you're ever asked to write a blog post or paper from this work, start here.

---

## Layout

```
memory/
  README.md           # this file (conventions + templates)
  LOG.md              # append-only one-liner per entry, chronological
  phases/             # per-phase retrospectives (what we built, why, lessons)
  experiments/        # individual experiment entries with hypothesis + result
  decisions/          # architectural / methodological choices and their reasoning
  findings/           # non-obvious discoveries worth remembering independently
```

### How they relate

- A **phase** is a multi-week body of work (Phase 1 = harness, Phase 2 = Tier-1 improvements, etc.). One retrospective per phase, written at the end.
- An **experiment** is a single A/B run or short sequence of runs. Every code change that could affect numbers gets an experiment entry. Links to `runs/` dir(s).
- A **decision** is a choice that shapes future work (e.g., "we commit baselines only at the standard contract"). Decisions don't report numbers; they report reasoning.
- A **finding** is a discovery we'd want to remember even if it didn't originate from a formal experiment (e.g., "B1 is 4× slower per token than B2/B3 because of the lm_head").

---

## Conventions

- **Filenames**: `YYYY-MM-DD-<kebab-slug>.md`. The date is when the entry was *written*, not when the work happened — a retrospective written today about last week still gets today's date.
- **One topic per file.** If an entry starts sprawling, split it.
- **Link to `runs/` by exact path.** `runs/baseline-p1-b1-tinystories/summary.json` is a permanent reference once committed. Drive-by references to `runs/20260504-*-whatever/` are fine for unfinished work but get canonicalized before the entry is considered complete.
- **Numbers with units.** Loss values carry 4 decimals. Throughput in tokens/sec or it/s. VRAM in MiB or GiB, not bytes. Wallclock in mm:ss or hours.
- **Prose over prose-style bullet lists.** A five-sentence paragraph with real content beats ten hollow bullets. Bullets for genuinely list-shaped data.
- **No "see above" or "as mentioned earlier".** Each entry stands on its own — a reader might jump in at any entry.

---

## The LOG.md

`LOG.md` is a flat, append-only, one-line-per-entry chronological index. One entry per day is typical. Example:

```
2026-05-04  phases/2026-05-04-phase1-harness.md          Phase 1 retrospective: measurement harness shipped
2026-05-04  experiments/2026-05-04-phase1-baselines.md   B1/B2/B3 baselines at standard contract
2026-05-04  findings/2026-05-04-b1-vocab-throughput.md   B1 is 4x slower b/c GPT-2 lm_head dominates
2026-05-05  experiments/2026-05-05-gelu-vs-relu.md       Phase 2 #1: GELU in FFN on all three benchmarks
```

Use `LOG.md` as a scannable table of contents. Direct links to files are cheap; prose in LOG.md is waste.

---

## Templates

### Experiment

```markdown
# <one-line title: what was tested>

**Date:** YYYY-MM-DD
**Phase:** N
**Branch / commit:** <sha-range or branch name>

## Hypothesis

One paragraph. What did we expect and why? Cite sources if this is a known technique. Predict direction (expected to reduce val_loss on B1, no effect on throughput, etc.) and rough magnitude if possible.

## Setup

What changed, expressed as a delta from the baseline. If this is a config-only change, paste the diff. If code changed, reference the commit SHA and list the files. Include:
- Benchmark(s) run (B1 / B2 / B3 / all three)
- Contract: confirm the contract fields match the baseline, OR call out the deliberate deviation (and how compare.py was invoked with --allow-contract-mismatch)
- Compute used (GPU, wallclock)

## Result

Numbers. Paste the compare.py table. Link to the exact run dirs under runs/.

## Interpretation

Why did we see what we saw? Was the hypothesis confirmed? If not, what's the mechanism for the deviation? Surface surprises explicitly.

## Conclusion

Do we ship this change? Yes / no / needs-more-investigation. Concrete next step.

## Open questions

What's unanswered? What would we need to run next to answer it?
```

### Decision

```markdown
# <one-line title: what was decided>

**Date:** YYYY-MM-DD
**Phase:** N

## Context

What problem are we solving? Why is this decision necessary now?

## Options considered

Enumerate with trade-offs. Include the ones we rejected — explicitly saying *why* something was rejected is half the value.

## Chosen option

What we went with and why.

## Expected consequences

What does this lock in for future work? What does it leave flexible?

## How we'd know if this was wrong

Concrete signals that would make us revisit.
```

### Finding

```markdown
# <one-line title: what was discovered>

**Date:** YYYY-MM-DD
**Source:** <experiment entry, phase retrospective, or "ad-hoc observation">

## Observation

One paragraph. What did we notice?

## Evidence

Point to run dirs, metrics, or specific commits. Numbers with units.

## Mechanism

Why does this happen? Physics, architecture, data property, implementation detail?

## Implications

How does this change what we do next? Does it invalidate any prior assumption? Does it predict something about future phases?
```

### Phase retrospective

```markdown
# Phase N retrospective: <theme>

**Phase dates:** YYYY-MM-DD to YYYY-MM-DD
**Commit range:** <base-sha>..<head-sha>

## Goal recap

What were we trying to accomplish? Quote or paraphrase from the roadmap.

## What shipped

High-level inventory, not a diff summary. The reader should understand the shape of what's now in the repo.

## What we learned

The non-obvious stuff. What worked better than expected, what didn't, what the harness surfaced that we didn't anticipate.

## Numbers

If the phase produced measurable outcomes, a summary table.

## What's different in the next phase because of this

Concrete implications.

## Outstanding debt

Known issues or shortcuts that will need attention. Be specific.
```

---

## Rules

1. **Every PR that changes training outcomes includes at least one experiment entry.** No exceptions. If a PR is purely infrastructure, an entry isn't required, but a note in the relevant phase retrospective is.
2. **Entries are not edited after the next phase starts.** If you need to correct something, add a new entry that references and supersedes the old one. The historical record is append-only.
3. **Decisions cite their rationale at the time of decision.** If the rationale turns out to be wrong, that's a new entry — not a rewrite.
4. **When in doubt, write the entry.** A slightly redundant entry is much cheaper than a lost insight.
