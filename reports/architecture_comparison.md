# Architecture Comparison

Audit date: 2026-06-24

## File Layout Finding

Current System C is implemented at:

- `chatbot_system_c.py`

There is no current `system_c/chatbot_system_c.py`. The evaluator adds a
nonexistent `system_c` directory to `sys.path`, but imports the root module
because the project root is also on `sys.path`.

## Actual Feature Matrix

| Feature | System A | System B | System C |
|---|---|---|---|
| Mistral through Ollama | Yes | Yes | Yes |
| Recent conversation history | Unfiltered 10 turns | Intent-filtered selected turns | No general transcript |
| Prior assistant answers in prompt | Yes | No | No |
| Intent classifier | No | Yes | Yes |
| Structured personal memory | No | Yes | Yes |
| Cross-domain prompt constraints | No | Yes | Yes |
| Unsupported named-entity guard | No | Yes | Yes |
| User-taught knowledge database | No | No | Yes |
| Semantic memory | No | No | Yes, but lexical unless embeddings enabled |
| MiniLM embeddings | No | No | Not active in stable runner |
| FAISS semantic index | No | No | Available in module, disabled in stable runner |
| Domain-scoped RAG | No | No | Yes, lexical BM25-style documents |
| `rag/index.faiss` retriever | No | No | No active import |
| Deterministic task/domain analysis | No | Limited prompt guidance | Yes |
| Relationship analyzer | No | Deterministic guidance | Yes |
| Independent answer audit | No | No | No in production C |
| Final revision call | No | No | No in production C |
| Meta-leakage cleanup | No | Limited cleanup | Limited cleanup |
| Can skip Mistral | Runtime error only | Unsupported entity guard | Memory, learning, unsupported entity guard |

## System A

Implementation: `system_a/chatbot_system_a.py`

Actual flow:

1. Load `memory/system_a_interactions.json`.
2. Include up to ten prior user/assistant turns without filtering.
3. Send a short system prompt plus the transcript to Mistral.
4. Trim the answer and append the interaction back to A history.

What A uses:

- Mistral generation.
- Recent unfiltered conversation history.
- The same global runtime seed policy.

What A does not use:

- Intent classifier.
- Structured personal facts.
- Filtered context.
- RAG.
- Semantic memory.
- Relationship analysis.
- Unsupported named-entity guard.
- Answer audit.
- Contamination protection.

Audit conclusion:

System A is a plausible weak stateful baseline. Its main strength is Mistral's
general pretrained knowledge and immediate transcript memory.

## System B

Implementation: `system_b/chatbot_system_b.py`,
`system_b/grounding.py`

Actual flow:

1. Run the shared BERT intent classifier.
2. Update `memory/system_b_facts.json`.
3. Select relevant structured personal facts.
4. Load `memory/system_b_interactions.json`.
5. Select prior context by classifier label.
6. Detect unsupported named technologies.
7. Generate deterministic cross-domain relationship guidance.
8. Send a detailed prompt to Mistral.
9. Post-process mixed-domain answers to remove follow-up/API drift.

What B uses:

- Shared intent classifier.
- Intent-filtered conversation context.
- Structured personal memory.
- Cross-domain prompt constraints.
- Unsupported named-entity guardrail.
- Mistral generation.

What B does not use:

- FAISS semantic memory.
- Domain-scoped RAG document retrieval.
- User-taught knowledge database.
- Independent answer audit.
- Final revision.

Audit conclusion:

System B is stronger than the intended intermediate baseline. It already has
task-completion rules, ambiguity instructions, relationship taxonomy,
unsupported-entity protection, and contamination constraints. This is a major
reason B and C can look too similar.

Risk:

The classifier has only four labels: robotics, daily, personal, mixed. It has
no general or unknown class. B relies on this classifier for context selection,
so classifier mistakes can route the wrong history into the prompt.

## System C

Implementation: `chatbot_system_c.py`

Actual flow:

1. Update `memory/system_c_facts.json`.
2. Parse explicit `Learn that ... means ...` or `Teaching: ... = ...` statements
   into `learning.json`.
3. Add taught facts to semantic memory.
4. Run the same shared classifier as B.
5. Resolve domain and task deterministically.
6. Detect ambiguity and relationship requirements.
7. Retrieve domain-scoped local document chunks with `research_core.py`.
8. Search semantic memory.
9. Make one grounded Mistral call for ordinary questions.
10. Clean length and limited meta-leakage phrases.

What C uses:

- Structured personal memory.
- User-taught knowledge database.
- Semantic-memory module.
- Intent classifier.
- Deterministic domain/task analysis.
- Domain-scoped local document retrieval.
- Relationship guidance.
- Unsupported entity guard.
- Grounded Mistral generation for ordinary questions.
- Limited meta-leakage cleanup.

What C does not currently use in production:

- Separate Mistral draft followed by a distinct answer audit.
- Separate final revision call.
- The `rag/retrieve.py` FAISS document retriever.
- Guaranteed MiniLM embeddings in controlled stable runs.

Mistral-skipping paths:

- Personal memory write/recall.
- Learned concept write/recall.
- Unsupported named-entity guard.

These deterministic paths are valid product behavior, but they must be recorded
because they affect latency and make some C answers not comparable to B/A
generation calls.

## Semantic Memory And RAG

`memory/semantic_memory.py` can use FAISS, but MiniLM embeddings only activate
when embedding mode is enabled. In the current controlled runner
`evaluator/research/state.py`, stable System C is explicitly configured with:

- `semantic.EMBEDDINGS_ENABLED = False`
- `semantic.MODEL = None`
- a fresh `faiss.IndexFlatIP`

That means stable C clean experiments currently use lexical semantic recall,
not MiniLM+FAISS semantic recall.

`rag/retrieve.py` and `rag/index.faiss` exist, but production C does not import
that retriever. Production C uses deterministic lexical document scoring in
`research_core.retrieve_local_knowledge`.

## Ablation Implementation

`evaluator/research/ablation.py` implements C0-C6 as a separate experimental
ladder:

- C0: Mistral only.
- C1: classifier.
- C2: structured personal memory.
- C3: MiniLM+FAISS semantic memory and learning.
- C4: local document retrieval.
- C5: relationship analyzer.
- C6: answer audit, final revision, meta-leakage cleanup.

This ablation path is closer to the expected advanced System C design than the
production System C file. It should be reported as an experimental ablation,
not silently treated as the production architecture.

## Shared Advanced Features

No evidence was found that System A receives advanced C features.

System B shares or approximates several advanced C behaviors:

- Shared classifier.
- Structured memory.
- Cross-domain constraints.
- Named-entity guard.
- Relationship-specific guidance.
- Output cleanup for mixed-domain answers.

These shared protections make B substantially closer to C and reduce measured
separation.

## Shared Mutable State

Normal assistant state:

- A: `memory/system_a_interactions.json`
- B: `memory/system_b_interactions.json`, `memory/system_b_facts.json`
- C: `memory/system_c_facts.json`, `learning.json`, `memory/texts.json`,
  `memory/faiss.index`

No normal A/B/C cross-system personal-memory file sharing was found.

Evaluator risks:

- Legacy evaluator derives established facts from shared `results.json`.
- Historical result summaries read archived runs from multiple old versions.
- Controlled runner uses experiment-local state directories, which is the safer
  design.

## Prompt Similarity Diagnosis

The three systems do not receive identical prompts:

- A receives a short system prompt plus raw transcript.
- B receives classifier label, classifier guidance, selected topics, personal
  facts, and optional relationship constraints.
- C receives deterministic request analysis, relationship/ambiguity guidance,
  personal facts, learned facts, semantic memory, and local references.

They still produce similar results because:

- B and C both contain strong contamination-prevention instructions.
- B and C use the same classifier.
- C's advertised separate audit/revision pipeline is not active in production.
- Stable C experiments currently disable MiniLM embeddings.
- Easy single-domain questions are mostly solved by the shared Mistral model.

## Architecture Conclusion

System A is genuinely different and weak.

System B is not merely intermediate; it already contains several advanced
protections.

System C is more advanced than B, but the production implementation is missing
some claimed components. The project claim should therefore be tested as:

- stable A/B/C comparison of the current actual systems; and
- C0-C6 ablation showing which advanced components add measurable value.
