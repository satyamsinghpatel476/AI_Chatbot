# Final Research Findings

## Current status

The evaluator and dashboard now report the project through separate contamination-focused metrics rather than a composite final score. The current generated artifacts are from a small smoke run and should be treated as a pipeline validation run, not as the final research conclusion.

Generated outputs:

- `evaluator/results/fresh_run_results.json`
- `evaluator/results/fresh_run_results.csv`
- `evaluator/results/category_summary.csv`
- `evaluator/results/ablation_summary.csv`
- `evaluator/results/intent_confusion_matrix.csv`
- `evaluator/results/statistical_tests.json`

## Why results previously looked similar

The earlier evaluator mixed broad answer quality, contamination, memory, and intent into comparable-looking summaries. With only a small benchmark, Systems A, B, and C could all receive similar marks for plausible answers even when they differed in memory, relationship handling, and contamination behavior.

## Why System C was slow

System C had advanced components that could lead to repeated work: classifier use, retrieval, memory lookup, relationship analysis, generation, and optional audit/revision behavior. The clean runner also used model-backed judging for relationship metrics. Fast mode keeps the research value but removes avoidable model calls.

## What now proves the project better

The stronger metrics are:

- Context Contamination Rate
- False Rejection Rate
- Memory Recall
- Knowledge Growth
- Cross-Domain Robustness
- Intent Accuracy
- Intent Macro F1
- Median Latency
- P95 Latency

These metrics directly test whether an assistant keeps robotics support separate from daily-life assistance while still remembering useful information and handling valid indirect relationships.

## How to run

Fast smoke benchmark:

```bash
env/bin/python evaluator/evaluator.py --mode research --fast --limit 30
```

Context-only research run:

```bash
env/bin/python evaluator/evaluator.py --mode research --suite context --systems A,B,C
```

Full fast research run:

```bash
env/bin/python evaluator/evaluator.py --mode research --fast
```

Full planner/audit mode:

```bash
FULL_RESEARCH_MODE=1 env/bin/python evaluator/evaluator.py --mode research
```

Dashboard:

```bash
env/bin/streamlit run evaluator/dashboard.py
```
