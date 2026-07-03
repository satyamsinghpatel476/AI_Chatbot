# AI Robotics Assistant - Version 3 Restoration Prompt

Created: June 23, 2026, 14:14 IST

Use the prompt below with Codex if later modifications damage this project.
The archive is the authoritative byte-for-byte backup; this prompt explains
what must be restored and verified.

## Restoration Prompt

```text
Restore my AI Robotics Assistant project to the saved Version 3 state.

Workspace:
/home/satyam/ai_robotics_assistant

Authoritative backup:
/home/satyam/ai_robotics_assistant/backups/ai_robotics_assistant_v3_20260623_141427.tar.gz

Integrity file:
/home/satyam/ai_robotics_assistant/backups/ai_robotics_assistant_v3_20260623_141427.tar.gz.sha256

Before changing anything:
1. Inspect the current workspace and preserve any newer files or modifications
   in a separate timestamped archive.
2. Verify the Version 3 archive with:
   sha256sum -c backups/ai_robotics_assistant_v3_20260623_141427.tar.gz.sha256
3. List the archive and confirm it contains the expected project files.
4. Restore Version 3 from the archive. Do not restore `env`, `.git`,
   `__pycache__`, or old training optimizer checkpoints because they were
   intentionally excluded.

Version 3 architecture:
- System A: naive stateful Mistral baseline using unfiltered conversation
  history.
- System B: shared intent classifier, intent-filtered context, structured
  personal memory, cross-domain constraints, and an unsupported named-entity
  guardrail.
- System C: structured personal memory, user-taught knowledge, semantic
  memory, deterministic task/domain analysis, domain-scoped RAG, relationship
  guidance, unsupported-entity protection, and grounded Mistral generation.
- Shared Ollama runtime: `llm_runtime.py`, default model `mistral:latest`.
- Intent classifier: `models/intent_classifier/model.safetensors` and its
  tokenizer/configuration files.
- Evaluator: `evaluator/evaluator.py`.
- Dashboard: `evaluator/dashboard.py`.
- Shared comparison metrics: `evaluator/metrics.py`.

Version 3 comparison metrics:
Primary:
1. Context Contamination Rate, lower is better.

Secondary:
2. Memory Recall, higher is better.
3. Knowledge Growth, higher is better.
4. Cross-Domain Robustness, higher is better.
5. Intent Classification Accuracy, higher is better.

There must be no composite final score. Old Accuracy, Hallucination, Leakage,
Task Fulfillment, Latency, or penalty-based Final Score fields may remain as
internal evaluator evidence, but they must not be used as the system comparison
scorecard.

Important expected behavior:
- System B must not treat earlier assistant answers as evidence.
- Duplicate current questions and runtime-error turns must be excluded from
  System B context selection.
- Consumer applications must not be presented as robot sensors, controllers,
  SLAM systems, localization algorithms, or direct replacements.
- Qualified indirect cross-domain use must name the exact authorized fields
  required and retain onboard perception and safety sensing.
- Unfamiliar named technologies must receive calibrated uncertainty instead
  of invented specifications, unless the user explicitly taught the definition.
- The dashboard must display the five metrics separately and show `N/A` when
  a run has no applicable questions or a system exposes no classifier output.

After restoration:
1. Run Python syntax checks for all project modules.
2. Run the evaluator metric-definition regression checks.
3. Run the Streamlit dashboard render test.
4. Regenerate archived statistics with:
   env/bin/python evaluator/research_summary.py
5. Start the dashboard with:
   env/bin/streamlit run evaluator/dashboard.py --server.headless true
6. Report every restored file, verification result, and anything that could
   not be restored.

Do not redesign or upgrade Version 3 during restoration. Restore it faithfully
first, verify it, and only then discuss optional improvements.
```

## Backup Scope

Included:

- all project Python source and Markdown documentation;
- System A, B, and C implementations;
- benchmark and intent datasets;
- evaluator source, current results, archived runs, and research summaries;
- memory files and semantic indexes;
- RAG documents, chunks, and FAISS index;
- trained classifier final weights and tokenizer/configuration;
- project PDFs and JSON state files.

Intentionally excluded:

- `env/` because it is a recreatable 7.9 GB virtual environment;
- `models/intent_classifier/checkpoint-*` because they contain approximately
  2.5 GB of duplicate training checkpoints and optimizer state;
- `.git/` because the repository metadata is empty/broken in this workspace;
- `__pycache__/` and compiled bytecode;
- the `backups/` directory itself.

The final classifier weights needed at runtime are included.
