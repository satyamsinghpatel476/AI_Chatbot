# AI Chatbot

Local multi-domain chatbot project focused on robotics, daily digital
assistance, personal memory, and research evaluation. The repository contains
the important source code, benchmark definitions, reports, tests, and dashboard
assets. Heavy local artifacts such as trained model weights, caches, virtual
environments, FAISS indexes, and full run archives are intentionally not
committed.

## Contents

- `chatbot_system_c.py`, `system_a/`, `system_b/`: chatbot systems and CLI
  entry points.
- `research_core.py`, `llm_runtime.py`, `intent_model.py`: shared runtime,
  retrieval, memory, and intent-classifier integration.
- `evaluator/`, `benchmarks/`: benchmark and
  research evaluation tools.
- `scripts/`: standalone generators, checks, and maintenance scripts.
- `docs/`, `reports/`: design notes, deployment notes, and research reports.
- `data/intent_dataset.json`, `training/train_intent.py`: intent-classifier
  dataset and training script.
- `rag/documents/`: local knowledge documents used by retrieval.
- `tests/`: regression and research harness tests.
- `ui_app/`: Streamlit dashboard for local evaluator results.
- `web_page/`: static dashboard and GitHub Pages deployment assets.

## Setup

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

The chatbot expects a local Ollama chat model by default:

```bash
ollama pull mistral
ollama serve
python chatbot_system_c.py
```

You can override the runtime with environment variables such as
`OLLAMA_URL`, `OLLAMA_MODEL`, `FAST_RESEARCH_MODE`, and `ENABLE_CUDA`.

## Model Artifacts

The trained classifier under `models/intent_classifier/` is not committed
because the weights and checkpoints are large. Rebuild it locally with:

```bash
python training/train_intent.py
```

Generated files such as `models/`, `.hf_cache/`, `rag/index.faiss`,
`rag/chunks.pkl`, `memory/*.json`, and full evaluator run archives are ignored
by git.

## Useful Commands

```bash
python -m unittest discover tests
python evaluator/evaluator.py --mode research --fast --limit 5 --smoke
python scripts/prebenchmark_check.py
streamlit run ui_app/app.py
python -m http.server 8000 --directory web_page
```

The repository includes a GitHub Actions workflow for publishing `web_page/`
through GitHub Pages. See `web_page/README.md` and
`docs/GITHUB_PAGES_DEPLOYMENT.md` for details.
