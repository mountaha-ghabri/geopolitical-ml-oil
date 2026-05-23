# Geopolitical Turning Points and Macroeconomic Volatility

### Causal Implementation Guide (Extension of Saadaoui, 2026 JCE)

**Author:** Montaha Ghabri ([montahaghabry@gmail.com]())

**Supervisor:** Dr. Naceur Khraief

**Institution:** Tunis Business School (TBS), University of Tunis

**Program:** Master of Research in Business Analytics (M2)

## 📌 Project Overview & Disclaimer

To the best of our knowledge, this repository represents the first comprehensive framework attempting to integrate deep Natural Language Processing (NLP) extensions and regularized Double/Debiased Machine Learning (DML) nuisances directly into the empirical architecture of  **[Saadaoui (2026, JCE)](https://crawford.anu.edu.au/sites/default/files/2026-02/08_2026_Saadaoui.pdf)** .

This codebook is an academic attempt to stress-test the structural stability, multi-market propagation, and state-dependence of bilateral geopolitical predictability index (PRI) shocks on global energy and commodity markets.

> [!NOTE]
>
> **Replication & Support:** If you encounter any bugs, library deprecation conflicts, or data alignment anomalies while executing this playbook, please  **[open a GitHub Issue](https://github.com/mountaha-ghabri/geopolitical-ml-oil/issues)** . I actively maintain this repository to support open-source empirical reproducibility.

## Core Empirical Findings & Insights

This extension delivers four primary academic insights regarding how bilateral Geopolitical Predictability Index (PRI) shocks transmit to global energy and financial markets:

### 1. The Dynamic Transmission Channel is Non-Random

* **Insight:** Placebo data permutation checks (500 iterations) confirm that the structural PRI shock vector has a true causal signature (**$p = 0.000$**). The response pattern exhibits a highly specific temporal wave matching the original paper: a distinct **negative asset response in the short run** (demand/uncertainty shocks) followed by an **inflationary positive correction in the medium run** (supply/containment hedges).

### 2. High-Dimensional Textual Nuances Stabilize Estimators

* **Insight:** Swapping generic macro-level geopolitical risk indices for structured, bilateral GDELT narrative features (Goldstein valence scores, thematic concentration, conflict/cooperation shares) dramatically mitigates omitted variable bias. When processed through a Partially Linear IV (PLIV) Double ML framework with regularized XGBoost nuisance estimators, the residualized impulse response functions remain stable without falling prey to standard degrees-of-freedom degradation.

### 3. Structural Instability and the Post-2015 Regime Shift

* **Insight:** Parameter stability is not uniform over time. Splitting the baseline sample reveals that the structural transmission mechanism was highly potent **pre-2015** (yielding 25 out of 49 statistically significant monthly horizons). However,  **post-2015** , as the trade and diplomatic architecture between major superpowers shifted permanently, the historical impulse response collapsed completely (only 5 out of 49 horizons remaining significant). Formal Chow stability testing confirms this structural break at **$p < 0.001$** across all horizons.

### 4. Methodological Boundaries & Small-Sample Constraints (Honest Disclosures)

* **Continuous Interaction Failure:** Modeling state-dependence via continuous interaction instruments (e.g., **$d2pri \times VIX$**) is structurally invalid for this framework. Weak instrument tests indicate the interaction instrument is functionally dead (**$F_{\text{interaction}} \approx 0.36$**), because adding a continuous macro modifier provides zero incremental information over the base instrument. Splitting samples into discrete structural regimes (Ramey & Zubairy topology) is the only valid way to identify state-dependent heterogeneity.
* **Smooth-Transition (STLP) Collapse:** Attempting a Smooth Transition Local Projection (STLP-IV) introduces fatal sample scarcity. Unconstrained calibrations collapse entirely into a single state (**$G_t \equiv 1$**), while constrained variants isolate only 7% of the data (~27 months) into the high-stress state. Estimating 19 parameters from 27 observations breaks model identification. Consequently, STLP modeling is excluded from the final execution layer as methodologically invalid for an **$n=385$** data profile.

## 🛠️ Environment Setup & Installation

Before running the playbook, initialize your environment. This project requires  **Python 3.10+** .

**Bash**

```
# Clone the repository
git clone https://github.com/mountaha-ghabri/geopolitical-ml-oil.git
cd geopolitical-ml-oil

# Install exact version-controlled dependencies
pip install -r requirements.txt
```

## 📁 Project Architecture

All source datasets, code files, and exported artifacts are mapped to the following explicit structure:

**Plaintext**

```
.
├── data/
│   ├── 02_features/      # High-dimensional commodity spreads and macroeconomic time-series
│   ├── 03_nlp/           # Parsed GDELT variables and JSON structural role maps
│   ├── cache/            # Intermediate serialized Parquet data (FinBERT article scores)
│   └── final/            # Final analytical matrices (baseline, nlp_A, nlp_bert)
├── original/             # Baseline Replication files (Saadaoui_2026_JCE.dta, .log)
├── notebooks/            # The Execution Playbook (01_*.ipynb to 10_*.ipynb)
├── results/              # Exported CSV coefficient tables and Wald test statistics
└── figures/              # Exported Impulse Response Functions (IRFs) and diagnostic plots
```

## Complete Execution Playbook

The notebooks must be executed sequentially. Below is the operational run order detailing the purpose, data dependencies, outputs, and runtime profiles for every stage of the project.

**Code snippet**

```
graph TD
    A[01 Baseline Replication] --> B[02 Macro Features]
    B --> C[03 GDELT NLP]
    C --> D[04 FinBERT / BERTopic]
    D --> E[05 Merge & Validation]
    E --> F[06 Double ML & PLIV]
    F --> G[07 State-Dependent LP]
    G --> H[08 Structural Breaks]
    H --> I[09 Multi-Outcome LP-IV]
    I --> J[10 Panel Extension]
```

### Phase 1: Replication & Feature Assembly

#### 1. `01_baseline_replication_saadaoui.ipynb`

* **Purpose:** Replicates the baseline core Stata code (`Saadaoui_JCE_2026.do`) in pure Python to verify identity parity for the US-China PRI **$\rightarrow$** WTI transmission framework. Maps standard error differences (`debiased=True` in Python vs. Stata GMM corrections).
* **Inputs:** `original/Saadaoui_2026_JCE.dta`
* **Outputs:** Baseline mean IV-LP plots in `figures/01_baseline_replication_saadaoui/`
* **Estimated Runtime:** ~15 seconds

#### 2. `02_macro_controls_feature_engineering.ipynb`

* **Purpose:** Engineers a high-dimensional control matrix (Global Supply Chain Pressure Index, Baltic Dry Index, bond spreads, VIX volatility shifts, commodity differentials) extracted from raw financial feeds.
* **Inputs:** `data/02_features/raw/bdi.csv`, `data/02_features/raw/gscpi.csv`
* **Outputs:** Enriched control tensors saved to `data/02_features/`
* **Estimated Runtime:** ~45 seconds

#### 3. `03_gdelt_structured_nlp_features_nb.ipynb`

* **Purpose:** Builds bilateral US-China structural narrative indices (Goldstein means, event volume tallies, conflict vs. cooperation shares) across the entire historical baseline timeline (1990–2022).
* **Inputs:** Raw GDELT event dumps in `data/`
* **Outputs:** `data/03_nlp/feature_matrix_nlp_A.csv`, `data/03_nlp/var_roles_nlp_A.json`
* **Estimated Runtime:** ~2 minutes

#### 4. `04_finbert_bertopic_deep_nlp.ipynb`

* **Purpose:** Runs deep NLP asset extraction on media headlines over the 2015–2022 window. Uses pre-trained FinBERT transformers for sentiment classification and BERTopic clusters to measure topic entropy.
* **Inputs:** `data/cache/uschn_urls_2015_2022.parquet`
* **Outputs:** `data/cache/finbert_sentiment_monthly.csv`, `data/cache/bertopic_input.csv`
* **Estimated Runtime:** ~12 minutes (GPU dependent)

#### 5. `05_data_merge_and_instrument_validation.ipynb`

* **Purpose:** Consolidates all macro features and textual streams into unified estimation data matrices. Runs first-stage instrument audits. Defines a curated matrix of **$p=15$** high-signal features to maintain optimal sample convergence parameters (**$n/p \approx 25$**).
* **Inputs:** `original/Saadaoui_2026_JCE.dta`, engineered macro/NLP outputs from steps 02, 03, 04.
* **Outputs:** `data/final/variable_roles.json`, validated dataset layers.
* **Estimated Runtime:** ~30 seconds

### Phase 2: Causality Testing & Machine Learning Nuisances

#### 6. `06_double_ml_iv_impulse_responses.ipynb`

* **Purpose:** Estimates regularized impulse responses using Partially Linear IV (PLIV) Double ML models with XGBoost learners. Evaluates instrument validity via Angrist-Fisch Random Forest weak instrument diagnostic checks and run directional nonlinear Granger causality tests alongside placebo data permutations.
* **Inputs:** Tiered datasets in `data/final/`
* **Outputs:** DML vs OLS IRF curves in `figures/06_double_ml_iv_impulse_responses/`, tabular statistics in `results/`
* **Estimated Runtime:** ~4 minutes (due to cross-fitting iterations)

### Phase 3: Heterogeneity, Breaks, and Multi-Asset Cascades

#### 7. `07_state_dependent_lp.ipynb`

* **Purpose:** Models non-linear macro transmission profiles. Swaps out collinear continuous interaction instruments for explicit Subsample Regime Splitting (Ramey & Zubairy, 2018 topology) over high/low financial stress (VIX) regimes.
* **Inputs:** Unified data arrays in `data/final/`
* **Outputs:** `results/wald_vix_regime.csv`, `results/wald_vix_regime_nlp.csv`, split-sample IRFs in `figures/07_state_dependent_lp/`
* **Estimated Runtime:** ~1 minute

#### 8. `08_structural_breaks.ipynb`

* **Purpose:** Assesses historical parameter drift by executing Chow structural stability tests across target break dates (e.g., the 2015 trade landscape shift). Documents why smooth-transition (STLP-IV) extensions collapse on small historical macro-samples.
* **Inputs:** Target data sheets in `data/final/`
* **Outputs:** Structural break F-test results in `results/08_structural_breaks/`
* **Estimated Runtime:** ~45 seconds

#### 9. `09_multi_outcome_lp_iv.ipynb`

* **Purpose:** Maps systemic spillovers by evaluating the transmission velocity and magnitude of the structural PRI shock across alternative outcome targets (Brent crude, Spot Gold, Safe-Haven instruments, CNY/USD exchange rate volatility, and the Baltic Dry Index).
* **Inputs:** Multi-market time series in `data/final/`
* **Outputs:** `results/multi_outcome_summary.csv`, comparative asset grid plots in `figures/09_multi_outcome_lp_iv/`
* **Estimated Runtime:** ~1.5 minutes

#### 10. `10_panel.ipynb`

* **Purpose:** Pools data across eleven independent bilateral country-dyads using Inverse-Variance Weighted (IVW) panel local projections to confirm multi-country external validity and rule out idiosyncratic location biases.
* **Inputs:** Dyadic structural inputs in `data/final/`
* **Outputs:** Aggregated panel summary models in `results/10_panel/`
* **Estimated Runtime:** ~2 minutes

## Summary of Production Guidelines

* **Modularity:** Do not skip steps. `05_data_merge_and_instrument_validation.ipynb` acts as a hard checkpoint. Downstream analysis notebooks (`06` through `10`) will fail if the analytical datasets are not fully built and validated first.
* **Data Scarcity Warning:** The datasets are small sample sizes (**$n=385$**). Keep cross-fitting parameter folds (`n_folds=3` or `5`) constrained in `DoubleML` to avoid high variance or model degradation in validation splits.
