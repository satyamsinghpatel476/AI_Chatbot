# Static Web Dashboard

Purpose: static HTML/CSS/JavaScript dashboard for presenting the latest benchmark results from the existing evaluator output.

## Export Latest Results

From the project root:

```bash
python web_page/export_results.py
```

This reads:

```text
evaluator/results/results.json
```

and writes:

```text
web_page/results_summary.json
```

## Serve Webpage

```bash
python -m http.server 8000 --directory web_page
```

Open:

```text
http://localhost:8000
```

## Refresh Behavior

Browser JavaScript loads `web_page/results_summary.json` using `fetch()`. When `evaluator/results/results.json` changes, rerun:

```bash
python web_page/export_results.py
```

Then refresh the browser page.

## Publish on GitHub Pages

This project includes a GitHub Actions workflow:

```text
.github/workflows/deploy-web-page.yml
```

After you push the project to GitHub:

1. Open your repository on GitHub.
2. Go to **Settings** -> **Pages**.
3. Under **Build and deployment**, set **Source** to **GitHub Actions**.
4. Push to `main` or `master`, or run the workflow manually from the **Actions** tab.
5. Open the Pages URL shown by the deploy workflow.

The workflow publishes the contents of `web_page/`. If `evaluator/results/results.json` is present in the repository, the workflow regenerates `web_page/results_summary.json` before publishing. If `results.json` is not committed, it uses the committed `web_page/results_summary.json`.

Note: GitHub Pages sites are public on the internet. Review `results_summary.json` before publishing if benchmark questions or responses contain private information.

The export script validates that:

- `results.json` exists.
- At least one benchmark entry exists.
- System `A`, `B`, and `C` results exist.
- Summary metrics can be computed.
