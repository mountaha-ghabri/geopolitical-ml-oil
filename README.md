# Geopolitical Turning Points and Macroeconomic Volatility

### Empirical Replication & Dyadic Extensions of Saadaoui (2026, JCE)

**Author:** Montaha Ghabri ([montahaghabry@gmail.com]())

**Supervisor:** Dr. Naceur Khraief

**Institution:** Tunis Business School (TBS), University of Tunis

**Program:** Master of Research in Business Analytics (M2)

## 📌 Project Overview & Scope Realism

This repository contains the replication code and empirical extensions for my Master's thesis, building directly upon the econometric framework of  **[Saadaoui (2026, JCE)](https://crawford.anu.edu.au/sites/default/files/2026-02/08_2026_Saadaoui.pdf)** .

To the best of our knowledge, this is an attempt to evaluate the external validity of the original US-China Geopolitical Predictability Index (PRI) shock transmission to global oil markets. We extend the baseline framework across 11 additional global bilateral country-dyads, audit historical parameter stability across key structural break thresholds, and transparently document the performance boundaries of Machine Learning (DML) and non-linear specifications when applied to small-sample macro data (**$n=385$**).

> [!NOTE]
>
> **Replication & Support:** This repository is fully open-sourced to ensure absolute empirical reproducibility. If you encounter any execution queries, package environment mismatches, or data alignment discrepancies, please  **[open a GitHub Issue](https://www.google.com/search?q=https://github.com/mountaha-ghabri/geopolitical-ml-oil/issues)** .

## Empirical Findings & Methodological Disclosures

Rather than masking statistical insignificance behind complex machine learning architectures, this thesis prioritizes empirical transparency. The actual findings from our pipeline show clear success in baseline extensions, alongside definitive statistical boundaries for high-dimensional models:

### 1. Robustness of the Core US-China Causal Signature

* **Finding:** Placebo data permutation tests (500 iterations) confirm that the baseline US-China PRI **$\rightarrow$** WTI transmission wave is non-random (**$placebo\ p = 0.000$**). The temporal response pattern matches the original paper, confirming a distinct  **sign reversal** : a significant negative asset drop in the short run (**$h = 6$**) followed by an inflationary upward correction in the medium run (**$h = 32$**). Anderson-Rubin confidence intervals confirm identification remains highly valid through **$h \approx 36$**; after that, precision gradually declines due to smaller effective sample sizes at long horizons.

### 2. The Asymmetry of Superpower Risk (GPR-USA vs. GPR-China)

* **Finding:** The transmission vector is heavily driven by Western risk perception. Domestic US perception of geopolitical risk (`GPR-USA`) heavily drives global oil prices, yielding  **41 out of 49 significant monthly horizons** . Conversely, Chinese domestic risk (`GPR-China`) exhibits near-zero predictive power, registering only  **3 out of 49 significant horizons** .

### 3. The Post-2015 Structural Collapse

* **Finding:** Parameter stability is highly regime-dependent over time. Formal Chow structural stability tests confirm a permanent structural break (**$p < 0.001$** across all horizons) around the 2015 geopolitical trade pivot. Pre-2015, the structural transmission mechanism is highly potent (25/49 significant horizons); post-2015, the historical impulse response collapses completely, leaving only 5 out of 49 horizons significant.

### 4. Dyadic Instrument Heterogeneity & Endogeneity Audits

* **Finding:** The baseline identification strategy does not automatically generalize worldwide. Out of the 11 alternative bilateral country-dyads tested via Inverse-Variance Weighted (IVW) panel pooling, instrument behavior and transmission profiles vary drastically:
  * **The Japan Null:** Japan-China displays strong first-stage instrument strength (**$F = 113$**) yet yields  **0/49 significant horizons** —a genuine null indicating that the underlying transmission mechanism is uniquely US-specific.
  * **The Anglo-Pacific Disconnect:** Australia (**$F = 54.5$**) and the United Kingdom (**$F = 46.3$**) also display robust first-stage F-statistics, yet yield near-zero significant horizons (9/49 and 1/49 respectively). **This confirms that strong instrument identification is necessary but not sufficient for structural transmission—the causal channel is US-specific.**
  * **Endogeneity Mitigation:** For France and Germany, IV estimates (0 to 3 out of 49 significant horizons) sharply contrast with baseline OLS runs (which naively report 45 to 46 significant horizons). This confirms that our IV configuration successfully strips out the severe upward endogeneity bias present in standard macro correlations. The remaining country-dyads suffer from severe weak instrument limitations (**$F < 15$**) or fail fundamental exogeneity diagnostics.

### 5. Honest Methodological Boundaries (Appendix Commitments)

* **Double/Debiased ML (PLIV) Boundaries:** Attempting to control for high-dimensional predictors using Partially Linear IV DoubleML (with an XGBoost nuisance setup) proved uninformative for our primary models. Due to small-sample constraints (**$n=385$**), cross-fitting inflated the median standard errors (**$\text{SE} > 0.6$**), rendering **$0/49$** horizons statistically significant. These regularized results are transparently moved to the **Thesis Appendix** as an exploration of small-sample boundaries.
* **Continuous Interaction & STLP-IV Failure:** Modeling state-dependence through continuous interaction instruments (e.g., **$d2pri \times VIX$**) is structurally invalid here; weak instrument audits confirm the modifier adds zero incremental power (**$F_{\text{interaction}} \approx 0.36$**). Furthermore, Smooth Transition Local Projections (STLP-IV) collapsed due to data scarcity, isolating only ~27 monthly observations into the high-stress state. Consequently, non-linear insights are derived strictly using discrete Subsample Regime Splitting (Ramey & Zubairy, 2018 topology) in the main text.

## 🛠️ Environment Setup & Installation

This project is built using a clean scientific stack in  **Python 3.10+** .

**Bash**

```
# Clone the repository
git clone https://github.com/mountaha-ghabri/geopolitical-ml-oil.git
cd geopolitical-ml-oil

# Install the exact required econometric and data libraries
pip install -r requirements.txt
```

## 📁 Project Architecture

**Plaintext**

```
.
├── data/
│   ├── 02_features/      # High-dimensional macro control variables & commodity spreads
│   ├── 03_nlp/           # Parsed bilateral US-China GDELT event matrices (1990-2022)
│   ├── cache/            # Intermediate serialized Parquet data layers
│   └── final/            # Consolidated estimation matrices (Baseline, GDELT-enriched)
├── original/             # Baseline replication source files (Saadaoui_2026_V7.dta, .log)
├── notebooks/            # The Execution Playbook (01_*.ipynb to 10_*.ipynb)
├── results/              # Exported main-text CSV tables, Wald criteria, and Appendix DML logs
└── figures/              # Exported Impulse Response Functions (IRFs) and diagnostic plots
```

## Complete Execution Playbook

The notebooks must be executed sequentially to honor intermediate file linkages. `05_data_merge_and_instrument_validation.ipynb` serves as a vital statistical checkpoint.

**Code snippet**

```
graph TD
    A[01 Baseline Replication] --> B[02 Macro Features]
    B --> C[03 GDELT NLP]
    C --> D[04 Pipeline Parameters]
    D --> E[05 Merge & Checkpoint]
    E --> F[06 Appendix DML Audits]
    E --> G[07 Discrete Regime Splits]
    E --> H[08 Structural Break Diagnostics]
    E --> I[09 Multi-Outcome Spillovers]
    E --> J[10 Multi-Dyad Panel Extension]
```

### Phase 1: Replication & Feature Assembly

#### 1. `01_baseline_replication_saadaoui.ipynb`

* **Purpose:** Replicates core Stata code (`Saadaoui_JCE_2026.do`) in Python to verify mathematical identity parity. Documents slight variance in standard errors caused by Python's `IV2SLS` debiased covariance adjustments vs. Stata’s GMM small-sample parameters.
* **Inputs:** `original/Saadaoui_2026_JCE.dta`
* **Outputs:** Replicated baseline mean IV-LP plots in `figures/01_baseline_replication/`
* **Runtime:** ~15 seconds

#### 2. `02_macro_controls_feature_engineering.ipynb`

* **Purpose:** Cleans and processes high-signal macro controls (Global Supply Chain Pressure Index, Baltic Dry Index, yield spreads, VIX volatility shifts, commodity spreads).
* **Inputs:** `data/02_features/raw/bdi.csv`, `data/02_features/raw/gscpi.csv`
* **Outputs:** Structured control arrays in `data/02_features/`
* **Runtime:** ~45 minutes

#### 3. `03_gdelt_structured_nlp_features_nb.ipynb`

* **Purpose:** Compiles bilateral US-China structural GDELT metrics (Goldstein means, event volume, conflict vs. cooperation shares) across the full 1990–2022 timeline.
* **Inputs:** Raw historical GDELT event matrix
* **Outputs:** Long-horizon narrative feature frames in `data/03_nlp/`
* **Runtime:** ~560 minutes

#### 4. `04_finbert_bertopic_deep_nlp.ipynb`

* **Purpose:** Investigates deep textual pipelines (Pre-trained FinBERT headlines and BERTopic entropy metrics). Maps out why short transformer windows (2015–2022) introduce structural look-ahead bias and sample constraints when paired with long-horizon macro models.
* **Inputs:** `data/cache/uschn_urls_2015_2022.parquet`
* **Outputs:** High-dimensional NLP control profiles (treated as exploratory text parameters)
* **Runtime:** ~200 minutes (GPU dependent)

#### 5. `05_data_merge_and_instrument_validation.ipynb`

* **Purpose:** Hard data checkpoint. Consolidates macro features and text features into clean matrices. Audits first-stage instrument performance. Constrains the final primary regression control set to a robust **$p=15$** vectors to keep optimal sample convergence limits (**$n/p \approx 25$**).
* **Inputs:** Engineered macro and text files from steps 01, 02, 03.
* **Outputs:** `data/final/variable_roles.json`, analytical estimation arrays.
* **Runtime:** ~ 3 minutes

### Phase 2: Causality Tests & Machine Learning Nuisance Limits

#### 6. `06_double_ml_iv_impulse_responses.ipynb`

* **Purpose:** Conducts instrument diagnostics via Angrist-Fisch Random Forest models and executes directional linear and non-linear XGBoost Granger causality runs. Estimates the Partially Linear IV (PLIV) Double ML models, preserving the resulting high-variance **$0/49$** significant results for the thesis  **Technical Appendix** .
* **Inputs:** Validated matrices in `data/final/`
* **Outputs:** Comparative DML vs OLS coefficients in `results/`, diagnostic plots.
* **Runtime:** ~ 20 minutes (due to cross-fitting iterations)

### Phase 3: Regime Heterogeneity, Structural Breaks, and Panel Assets

#### 7. `07_state_dependent_lp.ipynb`

* **Purpose:** Identifies structural state-dependence. Omits weak continuous interaction instruments and implements discrete Subsample Regime Splitting (Ramey & Zubairy topology) across explicit high/low financial stress (VIX) layers.
* **Inputs:** Target matrices in `data/final/`
* **Outputs:** `results/wald_vix_regime.csv`, regime split charts in `figures/07_state_dependent_lp/`
* **Runtime:** ~1 minute

#### 8. `08_structural_breaks.ipynb`

* **Purpose:** Runs formal Chow stability tests across distinct historical pivot dates (e.g., the 2015 trade landscape shift). Explicitly logs why Smooth Transition (STLP-IV) architectures mathematically fail on limited small-sample properties.
* **Inputs:** Consolidated analytical sheets in `data/final/`
* **Outputs:** Tabular F-test statistics and break horizons in `results/`
* **Runtime:** ~45 seconds

#### 9. `09_multi_outcome_lp_iv.ipynb`

* **Purpose:** Traces shock propagation across a wide financial grid, measuring spillover magnitude and response ordering across alternative targets (Brent crude, Spot Gold, Safe-Havens, CNY/USD volatility, and the Baltic Dry Index).
* **Inputs:** Multi-market time series arrays in `data/final/`
* **Outputs:** Multi-outcome baseline response frames in `results/multi_outcome_summary.csv`
* **Runtime:** ~1.5 minutes

#### 10. `10_panel.ipynb`

* **Purpose:** Tests geographic external validity by pooling transmission data across 11 separate dyadic combinations using Inverse-Variance Weighted (IVW) panel local projections, evaluating location biases and dyadic heterogeneity.
* **Inputs:** Multi-country dyadic arrays in `data/final/`
* **Outputs:** Panel pooled coefficients and heterogeneity metrics (**$I^2$**) in `results/`
* **Runtime:** ~2 minutes

## Essential Production & Estimation Rules

* **Enforced Modularity:** Notebooks cannot be run out of order. `05_data_merge_and_instrument_validation.ipynb` generates the underlying structural keys; subsequent scripts will throw fatal errors if this validation layer is omitted.
* **Sample Bounds Restriction:** Given our structural macroeconomic size (**$n=385$**), downstream cross-validation models in the Appendix folders must keep cross-fitting parameters locked tight (`n_folds=3` or `5`) to prevent total validation partition breakdown.
