# Evaluator Fairness Audit

Audit date: 2026-06-24

## Scope

Audited files:

- `evaluator/evaluator.py`
- `evaluator/metrics.py`
- `evaluator/dashboard.py`
- `evaluator/research_summary.py`
- `evaluator/research/benchmarks.py`
- `evaluator/research/scoring.py`
- `evaluator/research/statistics.py`
- `evaluator/research/run_clean_experiment.py`
- `evaluator/research/worker.py`
- `tests/test_research_harness.py`

## Direct System-Name Scoring

No explicit rule was found that awards a bonus because a response came from
System C or penalizes a response because it came from System A/B.

System names are used for routing, labeling, paired comparisons, and selecting
which callable to run. That is expected.

Remaining risks:

- Legacy `run_systems` always executes A, then B, then C. This can bias latency
  through warm-up or queue effects.
- Legacy per-answer judge fallback seeds are tied to A/B/C order if batch
  judging fails.
- The clean runner also iterates systems in a fixed order, although it records
  this in metadata.

## Legacy Evaluator Problems

The legacy evaluator in `evaluator/evaluator.py` is not sufficient for the
research claim.

Key problems:

- Mixed-domain heuristic accuracy can become 10 when any separation phrase is
  present, even if the answer is shallow.
- Mixed-domain contamination can be suppressed by words such as `indirectly` or
  `conditional`, even when the answer asserts a false relationship.
- Safe explanatory answers can be marked contaminated when they mention both
  domains without matching a hardcoded separation phrase.
- Personal recall checks for the expected substring and can miss contradiction.
- Daily and robotics fallback accuracy partly rewards length and domain-keyword
  presence.
- Hallucination detection only checks a few fixed phrases.
- Leakage detection only checks a few fixed strings and misses many meta leaks.
- Cross-domain robustness in `evaluator/metrics.py` reuses `accuracy` and
  `contamination`, so it is not an independent observation.

The blind Mistral judge is better than keyword-only scoring, but it still uses
the same local model family as the assistants and needs gold-conditioned,
case-level checks.

## Historical Results Are Not Controlled Evidence

`evaluator/research_summary.py` loads every JSON file in
`evaluator/results/runs`. This mixes old historical results with newer runs.

Observed risks:

- Multiple architecture versions can be aggregated together.
- Different benchmark sizes can be averaged together.
- Different evaluator methods can be mixed.
- Duplicate or near-duplicate historical runs can be counted more than once.
- Confidence intervals are run-level normal intervals, not question-level
  bootstrap intervals or paired tests.

Conclusion:

Historical summaries may be useful for project history, but they must not be
used to prove the final claim.

## Current Dashboard Problems

`evaluator/dashboard.py` currently reads:

- `evaluator/results/results.json`

It does not primarily display:

- `fresh_run_results.json`
- `category_summary.csv`
- `intent_confusion_matrix.csv`
- `statistical_tests.json`
- `ablation_summary.csv`

It displays separated metrics from the legacy result shape and states there is
no composite score, which is good. However, it does not yet show the requested
fresh clean benchmark statistics, latency median/P95, confidence intervals, or
ablation chart.

## Intent Evaluation Problems

The production classifier exposes only four labels:

- robotics
- daily
- personal
- mixed

The requested benchmark includes six labels:

- robotics
- daily
- personal
- mixed
- general
- unknown

The newer clean scoring correctly leaves System A as `N/A` when no classifier
output exists. It also preserves general/unknown gold labels so B/C can show
confusion rather than being silently excused.

Interpretation caution:

Because B and C use the same classifier instance, intent accuracy is a shared
classifier diagnostic, not a B-versus-C differentiator.

## Clean Research Harness Strengths

The newer `evaluator/research/` path fixes several legacy fairness issues:

- Generates explicit balanced benchmark files before running.
- Uses experiment-local state directories.
- Writes fresh machine-readable outputs separately from historical archives.
- Uses N/A for absent classifier output.
- Computes sample count, mean, median, standard deviation, bootstrap 95 percent
  confidence intervals.
- Uses McNemar's test for paired binary comparisons.
- Reports raw latency in milliseconds with median and P95.
- Separates context contamination, memory, knowledge, cross-domain robustness,
  intent, and latency instead of producing a composite final score.

## Clean Research Harness Risks

Issues found before measurement changes:

- Stable System C semantic embeddings are disabled in
  `evaluator/research/state.py`, so the stable C benchmark does not test the
  promised MiniLM+FAISS feature.
- Production structured memory does not extract preferred language or operating
  system, even though the memory suite tests them.
- Production System C does not have the same C6 audit/revision pipeline as the
  ablation implementation.
- Relationship scoring still relies on a Mistral judge. Malformed judgments are
  retried and unresolved cases are marked for human review, but the judge is not
  an independent human oracle.
- Full relationship scoring requires local Ollama access. In this environment,
  sandboxed Python cannot reach the endpoint; elevated local-network execution
  is required.

## Benchmark Design Audit

The generated benchmark suites satisfy the requested counts:

- Context contamination: 100 cases.
- Memory recall: 7 save/recall pairs.
- Knowledge growth: 10 taught fictional concepts.
- Intent classification: 60 balanced questions, 10 per label.
- Cross-domain robustness: 50 cases, 10 per relationship type.

Context contamination distribution:

- robotics to daily: 20
- daily to robotics: 20
- misleading shared word: 15
- incompatible cross-domain claim: 15
- legitimate relationship: 15
- unrelated dual question: 5
- adversarial wording: 5
- pronoun context carryover: 5

Cross-domain robustness distribution:

- direct: 10
- indirect: 10
- conditional: 10
- incompatible: 10
- uncertain: 10

No duplicate IDs were found in generated suites.

## Specific Hypotheses Supported By Audit

- Same Mistral model causes similar general knowledge: supported.
- System B prompt is already too advanced: supported.
- System A has too many contamination-prevention instructions: not supported.
  A's prompt is minimal.
- Evaluator rewards generic answers: supported for legacy fallback.
- System C answers can be shorter but safer: plausible; must be measured.
- System C retrieval can miss known concepts: plausible when embeddings are
  disabled or learned concepts are not matched lexically.
- System C can over-reject valid indirect relationships: plausible because its
  guard and relationship rules are deterministic; must be measured.
- Benchmark has too many easy single-domain questions: supported for legacy
  `results.json` and default auto runs; less true for generated clean suites.
- Historical results mixed with new runs: supported in `research_summary.py`.
- Latency rounded to zero: raw clean harness uses ms and avoids this; old
  summaries can hide deterministic fast paths.
- Memory tests after previous systems save same facts: clean harness isolates
  per-system state; legacy sequential result file can confuse evaluator facts.
- Shared files allow state leakage: normal A/B/C memories are separated; legacy
  evaluator state is shared.
- RAG documents equally accessible to multiple systems: not in assistant
  prompts; only C uses document retrieval, while evaluator references may use
  documents for judging.
- Expected-answer scoring does not measure task completion: supported for
  legacy fallback; clean relationship scoring improves this.

## Fairness Conclusion

No artificial System C bonus was found. The bigger problem is that the old
measurement path can hide real differences, and the current production
architecture does not fully match the claimed System C design.

Use the clean benchmark and ablation outputs for claims. Do not use historical
aggregate summaries or a composite final score to claim that C wins.
