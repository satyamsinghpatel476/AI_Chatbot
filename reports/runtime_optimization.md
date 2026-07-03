# Runtime Optimization

## What changed

- `FAST_RESEARCH_MODE` is enabled by default unless `FULL_RESEARCH_MODE=1`.
- System C now answers personal-memory writes/recalls, learned-knowledge writes/recalls, unsupported named technologies, and incompatible mixed-domain relationship questions without calling Mistral.
- Normal single-domain System C questions still use one grounded Mistral call.
- Audit/revision work is reserved for full mode or ablation cases that need it.
- Classifier predictions are cached in-process.
- Local RAG retrieval results are cached in-process.
- Identical Mistral prompts are cached in `/tmp/ai_robotics_assistant_mistral_cache.json`.
- Worker calls are wrapped with per-call timeouts.
- Worker JSON outputs are written after each case so interrupted runs leave partial results.

## Modes

Fast mode:

```bash
env/bin/python evaluator/evaluator.py --mode research --fast --limit 30
```

Full mode:

```bash
FULL_RESEARCH_MODE=1 env/bin/python evaluator/evaluator.py --mode research
```

Limited runs skip ablation by default to stay useful as smoke tests. To include the C0-C6 ablation during a limited run:

```bash
env/bin/python evaluator/evaluator.py --mode research --fast --limit 30 --run-ablation
```

## Smoke check

The smoke command used for validation was:

```bash
env/bin/python evaluator/evaluator.py --mode research --fast --limit 6 --timeout 5 --fresh
```

It completed and wrote the standard result files. Some generation rows reached the 5-second timeout, which is recorded in response-level results instead of hidden.
