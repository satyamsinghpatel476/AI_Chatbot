# GitHub Pages Deployment

This project can publish the static dashboard in `web_page/` using GitHub Pages.

## 1. Export the Latest Dashboard Data

From the project root:

```bash
python web_page/export_results.py
```

This writes:

```text
web_page/results_summary.json
```

The webpage reads this file in the browser.

## 2. Initialize Git Locally

Run this only once if the project is not already a Git repository:

```bash
git init
git branch -M main
```

## 3. Commit the GitHub Pages Files

For a dashboard-only GitHub Pages repository:

```bash
git add .gitignore .github web_page GITHUB_PAGES_DEPLOYMENT.md
git commit -m "Add GitHub Pages dashboard"
```

If you want GitHub Actions to regenerate `results_summary.json` from the original evaluator output, also commit:

```bash
git add -f evaluator/results/results.json
git commit -m "Add dashboard source results"
```

Only do this if the benchmark questions and responses are safe to publish.

## 4. Connect to Your GitHub Repository

Replace `USERNAME` and `REPOSITORY` with your GitHub account and repository name:

```bash
git remote add origin https://github.com/USERNAME/REPOSITORY.git
git push -u origin main
```

If the remote already exists:

```bash
git remote set-url origin https://github.com/USERNAME/REPOSITORY.git
git push -u origin main
```

## 5. Enable GitHub Pages

In GitHub:

1. Open your repository.
2. Go to **Settings** -> **Pages**.
3. Under **Build and deployment**, set **Source** to **GitHub Actions**.
4. Open the **Actions** tab and confirm the deploy workflow completed.

The site URL will look like:

```text
https://USERNAME.github.io/REPOSITORY/
```

## Updating Published Results

Each time `evaluator/results/results.json` changes:

```bash
python web_page/export_results.py
git add web_page/results_summary.json
git commit -m "Update dashboard results"
git push
```

GitHub Actions will redeploy the static dashboard.
