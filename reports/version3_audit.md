# Version 3 Audit

Audit date: 2026-06-24

Project:
Local Multi-Domain Assistant for Beginners: Evaluating Strategies to Prevent
Context Contamination Between Robotics Support and Daily-Life Assistance

## Preservation

Before source-code inspection or edits, the current workspace was archived:

- Fresh archive: `backups/current_workspace_20260624_035313.tar.gz`
- Fresh checksum: `backups/current_workspace_20260624_035313.tar.gz.sha256`
- Verification: `sha256sum -c` passed
- Size: 3.3G

The archive command excluded `.git`, `env`, `venv`, `.venv`, `__pycache__`
directories, common checkpoint folders, and the archive being created.

## Authoritative Version 3 Backup

Integrity check:

```text
sha256sum -c backups/ai_robotics_assistant_v3_20260623_141427.tar.gz.sha256
backups/ai_robotics_assistant_v3_20260623_141427.tar.gz: OK
```

Archive listing was generated at `/tmp/ai_robotics_assistant_v3_contents.txt`.
The archive contains 78 entries.

Expected files found:

- `./system_a/chatbot_system_a.py`
- `./system_b/chatbot_system_b.py`
- `./intent_model.py`
- `./llm_runtime.py`
- `./research_core.py`
- `./system_b/grounding.py`
- `./memory/semantic_memory.py`
- `./evaluator/evaluator.py`
- `./evaluator/metrics.py`
- `./evaluator/dashboard.py`
- `./evaluator/research_summary.py`

Important layout mismatch:

- Expected by the request: `system_c/chatbot_system_c.py`
- Present in Version 3 archive: `./chatbot_system_c.py`
- Present in current workspace: `chatbot_system_c.py`

The archive also includes RAG documents and indexes, evaluator historical
results, memory files, `.hf_cache`, `.agents`, PDFs, and the intent classifier
model files.

## Current Workspace State

The current workspace already contained these generated or experimental files
before this audit pass:

- `reports/version3_audit.md`
- `reports/architecture_comparison.md`
- `reports/evaluator_fairness_audit.md`
- `reports/benchmark_design.md`
- `evaluator/research/`
- `evaluator/benchmarks/`
- fresh-result CSV/JSON files under `evaluator/results/`

The directory is not currently a Git repository from
`/home/satyam/ai_robotics_assistant`; `git status --short` returns:

```text
fatal: not a git repository (or any of the parent directories): .git
```

## Verified Runtime Preconditions

Static checks performed before source edits:

- `python -m py_compile ...` over the audited source files: passed
- `python -m unittest tests/test_research_harness.py`: 8 tests passed

Ollama:

- Sandboxed localhost access failed.
- Elevated localhost check succeeded.
- Available model: `mistral:latest`
- Digest: `6577803aa9a036369e481d648a2baebb381ebc6e897f2bb9a766a2aa7bfbc1cf`
- Parameter size: 7.2B
- Quantization: Q4_K_M

Any model-backed benchmark must be run with elevated local-network access in
this environment; otherwise Python cannot reach `127.0.0.1:11434`.

## Main Audit Findings

1. System A, B, and C are not identical, but B is much stronger than the
   intended intermediate design.
2. System B already contains strong contamination-prevention rules, relationship
   taxonomy, unsupported named-entity handling, and structured memory.
3. Production System C is advanced relative to B, but it does not implement the
   full claimed draft, independent answer audit, final revision pipeline.
4. Production System C can skip Mistral generation for personal-memory,
   learning, and unsupported-entity paths.
5. Production System C imports semantic memory, but the controlled stable
   runner currently disables embeddings and uses lexical fallback for C.
6. The ablation implementation under `evaluator/research/ablation.py` is closer
   to the intended C0-C6 design than production `chatbot_system_c.py`.
7. The legacy evaluator has no direct system-name bonus, but its heuristic
   fallback can reward shallow separation phrases and can mis-score memory and
   contamination.
8. Historical run aggregation mixes old architectures, old evaluators, and
   different benchmark sizes, so it cannot prove the project claim.
9. The newer clean experiment harness is the right direction, but dashboard
   display and stable System C semantic-memory configuration still need repair.

## Why Previous Results Looked Too Similar

- All systems use the same local Mistral model, so easy single-domain answers
  converge naturally.
- System B has many of the same safety and relationship constraints expected
  from System C.
- Several current metrics reuse the same response properties repeatedly.
- The old dashboard reads `evaluator/results/results.json`, which can represent
  a narrow or historical run rather than the clean balanced suites.
- Intent accuracy for B and C measures the same shared classifier.
- If MiniLM/FAISS and separate audit/revision are not active in production C,
  the measurable gap between B and C shrinks.
- Immediate save/recall or shared benchmark state can make A look stronger than
  it should.
- Historical summaries mix fresh and old result histories.

## Audit Boundary

No claim that System C wins is supported by this audit alone. The next valid
step is a clean, balanced, state-isolated benchmark plus C0-C6 ablation, with no
system-name scoring or hidden bonuses.
