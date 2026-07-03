# Research design

## Research question

How much do intent-conditioned context selection, structured and semantic
memory, domain-specific retrieval, and relationship analysis reduce context
contamination while improving memory recall, knowledge growth, cross-domain
robustness, and intent classification accuracy in a local multi-domain
assistant for beginners?

## Controlled systems

All systems use the same local `mistral:latest` model through Ollama.

- **A — naive stateful baseline:** unfiltered recent interaction history and a
  minimal conversational prompt.
- **B — intent-conditioned assistant:** BERT intent classification,
  intent-filtered recent context, relevant structured personal facts, and one
  Mistral generation. B has no RAG or verification.
- **C — advanced assistant:** structured personal memory, user-taught
  knowledge, lightweight semantic memory, deterministic full-sentence task
  analysis, hybrid domain-scoped RAG, cross-domain relationship guidance, and
  one grounded generation with an integrated self-check.

Ordinary factual answers are generated rather than selected from
benchmark-specific answer tables. Deterministic handling is limited to explicit
memory writes/recalls and an evidence-based guardrail for unsupported named
technologies.

## Benchmark structure

`testing_1.py` defines 50 questions with gold category metadata:

- 15 robotics reasoning and troubleshooting questions;
- 10 daily-life digital-service questions;
- 10 intentionally ambiguous questions;
- 10 cross-domain relationship questions;
- 5 deliberately unverifiable named-technology questions.

The category labels describe the expected behavior, not a reference answer.
They prevent the evaluator from incorrectly inferring benchmark strata from a
few keywords.

## Evaluation

The system comparison uses one primary metric and four secondary metrics:

1. **Context Contamination Rate (primary, lower is better):** the percentage of
   robotics, daily-life, and mixed-domain responses containing an incorrect
   cross-domain connection.
2. **Memory Recall (higher is better):** objective correctness on delayed
   personal-fact recall questions.
3. **Knowledge Growth (higher is better):** objective correctness on delayed
   recall of concepts explicitly taught by the user earlier in the run.
4. **Cross-Domain Robustness (higher is better):** normalized task quality on
   mixed-domain questions, with contaminated answers receiving zero.
5. **Intent Classification Accuracy (higher is better):** agreement between a
   system's reported classifier label and gold labels covered by the classifier
   taxonomy. Systems without an intent classifier report `N/A`.

Responses remain anonymized for the internal quality judge. Its correctness,
task-fulfillment, relevance, completeness, clarity, and calibration dimensions
support Cross-Domain Robustness and behavioral constraints, but they are not
separate comparison metrics or combined into an overall score.

Deliberately unverifiable items use a gold constraint: unsupported confident
details are penalized, while calibrated uncertainty and a request for
documentation are rewarded.

Intentionally ambiguous items also use a gold behavioral constraint. Because
there is no single factual answer to score for correctness, every system is
assessed on whether it identifies missing context, provides useful conditional
guidance, and asks a focused clarifying question.

## Recommended paper protocol

1. Freeze the exact Ollama model digest and software environment.
2. Keep system development prompts separate from held-out benchmark questions.
3. Use at least one additional held-out benchmark not used during development.
4. Run at least five random seeds and archive every response.
5. Randomize anonymous answer order for judging.
6. Validate the model judge with at least two human annotators.
7. Report inter-annotator agreement and paired confidence intervals.
8. Include C ablations without retrieval, relationship analysis, and
   verification.
9. Publish failure examples and refusal errors, not only averages.

## Result integrity

Desired target ranges for any metric are engineering hypotheses, not values to
force. Encoding target ranges into prompts, answer tables, benchmark weights,
or scoring logic would invalidate the comparison. The acceptable claim is the
ordering and effect size actually measured across multiple seeds and
human-validated evaluation.

Generate archived-run statistics with:

```bash
env/bin/python evaluator/research_summary.py
```

## Important limitation

The development judge currently uses the same model family as the assistants.
That is useful for iteration but insufficient for a strong factuality claim.
The paper should use independent human evaluation or a separately sourced judge
and should report disagreements.

For laptop-friendly development runs, the evaluator scores all three answers in
one randomized anonymous batch request. Set `EVALUATOR_BATCH_JUDGE=0` to restore
one independent judge call per answer for the final paper protocol.
