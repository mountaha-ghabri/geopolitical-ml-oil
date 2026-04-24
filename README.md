# Replication + ML Contribution

Notebook-first replication and extension workflow for Saadaoui (2026, JCE).

Reference package: [Saadaoui_2026 folder](https://github.com/JamelSaadaoui/ResearchPapers/tree/e27ebc3b65f4e120a8287900daa4f65a017ce758/Saadaoui_2026)

## Clean Folder Layout

- `notebooks/`
  - `01_replication_workflow.ipynb`: baseline replication with Stata-log parity checks
  - `02_ml_benchmark.ipynb`: ML benchmark (Lasso/Ridge/RF) with weak-IV diagnostics
- `replication/`
  - `saadaoui_replication.py`: calibrated core implementation used by notebooks
- `original/`
  - original `.do` and `.log` from author
- `data/`
  - place `Saadaoui_2026_JCE.dta` here
  - `data/cache/` stores parquet cache
- `figures/`
  - generated plots
- `results/`
  - generated CSV outputs and diagnostics
- `docs/`
  - transparent troubleshooting notes (`ML_EXTENSION_PLAYBOOK.md`)

## Usage

Run in order:
1. Open `notebooks/01_replication_workflow.ipynb`
2. Then run `notebooks/02_ml_benchmark.ipynb`

## Current Replication Quality

- Figure 3 lead IRF: near-identical to Stata log
- Figure 4 IV-LP path: very close, minor residual differences
- Baseline first-stage F: matched Stata step-0 value

## Transparency on Non-Identity

Exact identity may fail because:
- Stata `locproj` internals are not public-equivalent to Python loops
- `ivregress gmm` and Python IV routines differ in numerical/covariance details
- `ivqregress` has no exact one-command Python clone
- plotting defaults differ across Stata and matplotlib

## ML Extension Status

Documented in `docs/ML_EXTENSION_PLAYBOOK.md`:
- what failed
- why it failed
- what is retained as robust
- what should be reported as robustness only

## Contribution Log

- Date:
- Change:
- Rationale:
- Validation: