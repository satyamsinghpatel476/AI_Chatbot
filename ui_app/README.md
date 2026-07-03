# Python Dashboard

Purpose: Streamlit dashboard for viewing the latest local benchmark results without changing the evaluator, benchmark files, systems, prompts, or result schema.

## Installation

From the project root:

```bash
pip install -r ui_app/requirements.txt
```

## Run

```bash
streamlit run ui_app/app.py
```

## Results Source

The dashboard reads directly from:

```text
evaluator/results/results.json
```

No copy of `results.json` is stored inside `ui_app/`.

## Refresh Behavior

The app loads `evaluator/results/results.json` from disk on each Streamlit rerun. Click **Refresh latest results** or refresh the browser tab after a benchmark run to display the newest values.

The loader validates that:

- `results.json` exists.
- At least one benchmark entry exists.
- System `A`, `B`, and `C` results exist.
- Summary metrics can be computed.
