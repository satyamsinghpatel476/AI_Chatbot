# Assistant architectures

## System A — naive stateful baseline

```text
User
  ↓
Unfiltered recent conversation transcript
  ↓
Mistral 7B
  ↓
Answer
```

System A receives the latest ten interactions with no intent filtering,
retrieval, structured memory, ambiguity analysis, or answer verification. Its
minimal prompt represents ordinary local-LLM conversation. Earlier irrelevant
turns can influence later answers, making A the contamination-prone baseline.

## System B — intent-conditioned selective-context assistant

```text
User
  ↓
BERT Intent Classifier
  ↓
Intent-Filtered Recent Context
  ↓
Relevant Structured Personal Facts
  ↓
One Mistral 7B Generation
  ↓
Answer
```

System B uses the classifier to select compatible earlier turns before
generation. It also supplies explicitly stored personal facts when relevant.
The classifier is treated as a hint rather than ground truth when confidence is
low. B has no document retrieval, task-planning model, semantic memory,
cross-domain relationship analyzer, or answer verifier. It therefore isolates
the value of intent-conditioned context selection over A.

## System C — fast grounded reasoning, hybrid RAG, and memory

```text
User
  ↓
Structured Personal Memory + User-Taught Knowledge
  ↓
BERT Intent Hint + Deterministic Full-Sentence Task Analysis
  ↓
Domain-Scoped BM25 Retrieval + Lightweight Semantic Memory
  ↓
One Grounded Mistral Generation with Integrated Self-Check
  ↓
Answer
```

System C analyzes the complete request with lightweight deterministic rules
before answering. It identifies causal, procedural, comparative,
recommendation, troubleshooting, ambiguity, memory, and cross-domain tasks
without spending a separate LLM call on planning. Retrieval uses separate
robotics and daily-digital reference material.

The final Mistral call receives the task type, ambiguity requirement,
cross-domain relationship hint, selected memory, and retrieved passages. It is
instructed to reason and self-check in one pass. Unsupported named technologies
are handled by an evidence-based entity guardrail. This reduces ordinary System
C generation from three to five LLM calls to one.

## Experimental interpretation

The systems form a controlled capability ladder:

- A tests unfiltered conversational generation.
- B adds intent-conditioned selective context.
- C adds task analysis, multi-source memory, RAG, relationship reasoning, and
  an integrated generation-time self-check.

This architecture is designed to make C more robust, but measured ordering must
still be reported honestly. Exact score ranges must not be encoded in prompts,
benchmark answers, or scoring rules.
