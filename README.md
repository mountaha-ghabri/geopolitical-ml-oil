# Geopolitical Turning Points and Oil Price Dynamics

### Replication and ML Extension of Saadaoui (2026, Journal of Comparative Economics)

**Author:** Montaha Ghabri · moontahaghabry@gmail.com
**Supervisor:** Professor Naceur Khraief
**Institution:** Tunis Business School, University of Tunis · M2 Business Analytics

---

## Context

This is the computational pipeline for a master's thesis extending [Saadaoui (2026)](https://crawford.anu.edu.au/cama/research/geopolitical-turning-points-and-macroeconomic-volatility-bilateral-identification), who constructs a
bilateral Political Relationship Index (PRI) and uses its second difference (Δ²PRI) as an instrument
to identify abrupt diplomatic turning points and estimate their causal effect on world oil prices via
instrumental variable local projections (LP-IV), for the China-USA dyad at monthly frequency 1990–2022.

**Three extensions are implemented:**

1. **Measurement.** A composite geopolitical index aggregating five independent NLP pipelines:
   GDELT event scores, ICEWS Goldstein scores, Phoenix SWB, Phoenix NYT, and FinBERT financial
   sentiment. Used as nuisance control in DML, not as an instrument (weak F = 2.59).
2. **Single-dyad causal ML (CHN-USA).** Double/Debiased ML with XGBoost nuisance,
   Chernozhukov-Hansen (2005) Algorithm 1 IV quantile local projections, regime heterogeneity,
   wild bootstrap CIs, Anderson-Rubin confidence sets, BH multiple testing correction,
   sup-Wald structural break test.
=======
# Geopolitical Turning Points and Oil Price Dynamics
### Replication and ML Extension of Saadaoui (2026, Journal of Comparative Economics)

**Author:** Montaha Ghabri · moontahaghabry@gmail.com  
**Supervisor:** Professor Naceur Khraief  
**Institution:** Tunis Business School, University of Tunis · M2 Business Analytics

---

## Context

This is the computational pipeline for a master's thesis extending [Saadaoui (2026)](https://crawford.anu.edu.au/cama/research/geopolitical-turning-points-and-macroeconomic-volatility-bilateral-identification), who constructs a
bilateral Political Relationship Index (PRI) and uses its second difference (Δ²PRI) as an instrument
to identify abrupt diplomatic turning points and estimate their causal effect on world oil prices via
instrumental variable local projections (LP-IV), for the China-USA dyad at monthly frequency 1990–2022.

**Three extensions are implemented:**

1. **Measurement.** A composite geopolitical index aggregating five independent NLP pipelines:
   GDELT event scores, ICEWS Goldstein scores, Phoenix SWB, Phoenix NYT, and FinBERT financial
   sentiment. Used as nuisance control in DML, not as an instrument (weak F = 2.59).

2. **Single-dyad causal ML (CHN-USA).** Double/Debiased ML with XGBoost nuisance,
   Chernozhukov-Hansen (2005) Algorithm 1 IV quantile local projections, regime heterogeneity,
   wild bootstrap CIs, Anderson-Rubin confidence sets, BH multiple testing correction,
   sup-Wald structural break test.

>>>>>>> af4e5fc3b8097a03c8375cf6e8ea8f47b62d4623
3. **Panel extension (12 dyads).** LP-IV for 8 valid dyads, IVW pooled estimator, pooled DML
   with dyad fixed effects (n ≈ 3,000), causal forest with ICEWS attention-share moderator,
   Diebold-Yilmaz connectedness, Granger causality network, Graph Attention Network.

---

## Repository Structure

```
macro-geopolitics/
│
├── notebooks/                              ← Main analysis notebooks (run in order)
│   ├── 00_explore_and_validate.ipynb
│   ├── 01_baseline_replication_saadaoui.ipynb
│   ├── 02_composite_index_construction.ipynb
│   ├── 02_macro_controls_feature_engineering.ipynb
│   ├── 03_instrument_diagnostics_and_macro_merge.ipynb
│   ├── 05_dml_quantile_iv_lp_clean.ipynb       ← Core single-dyad results
│   └── 06_panel_network_causal_forest_merge.ipynb ← Panel + network extension
│
├── notebooks/data_acquisition/             ← GDELT download scripts (optional)
│   ├── 01_gdelt_event_panel.ipynb
│   └── 02_gdelt_corpus.ipynb
│
├── data/
│   ├── Saadaoui_2026_JCE.dta               ← Primary dataset (from public replication package)
│   ├── 03_nlp/                             ← Processed NLP panels (GDELT, ICEWS, Phoenix)
│   └── final/                              ← Merged master panel, variable roles JSON
│
└── outputs/
    ├── 03c/                                ← Composite GPR parquet files
    ├── 04_diagnostics/                     ← Master panel, first-stage diagnostics
    ├── 04_dml/                             ← NB05 figures and tables
    └── 05_panel/                           ← NB06 figures and tables
```

---

## How to Run

**All notebooks are already executed.** Outputs (figures, tables, printed results) are saved
inside each notebook and in the `outputs/` directory. You **do not** need to rerun anything to
see the results. Just open the notebooks in Jupyter or view them on GitHub.

To rerun from scratch:

```bash
git clone https://github.com/mountaha-ghabri/geopolitical-ml-oil.git
cd geopolitical-ml-oil
pip install -r requirements.txt
```

**Run notebooks in this order:**

<<<<<<< HEAD
| Step | Notebook                                      | Purpose                               | Runtime     |
| ---- | --------------------------------------------- | ------------------------------------- | ----------- |
| 1    | `00_explore_and_validate`                   | Data audit, source coverage           | ~90 min     |
| 2    | `01_baseline_replication_saadaoui`          | LP-IV replication matching Stata      | ~2 min      |
| 3    | `02_composite_index_construction`           | Build 5-source GPR index              | ~10 sec     |
| 4    | `03_macro_controls_feature_engineering`     | Download/proxy macro controls         | ~2 min      |
| 5    | `04_instrument_diagnostics_and_macro_merge` | First-stage F-stats, master panel     | ~1 min      |
| 6    | `05_dml_quantile_iv_lp_clean`               | **Main single-dyad estimation** | ~60–90 min |
| 7    | `06_panel_network_causal_forest_merge`      | **Panel + network extension**   | ~50–70 min |
=======
| Step | Notebook | Purpose | Runtime |
|------|----------|---------|---------|
| 1 | `00_explore_and_validate` | Data audit, source coverage | ~90 min |
| 2 | `01_baseline_replication_saadaoui` | LP-IV replication matching Stata | ~2 min |
| 3 | `02_composite_index_construction` | Build 5-source GPR index | ~10 sec |
| 4 | `03_macro_controls_feature_engineering` | Download/proxy macro controls | ~2 min |
| 5 | `04_instrument_diagnostics_and_macro_merge` | First-stage F-stats, master panel | ~1 min |
| 6 | `05_dml_quantile_iv_lp_clean` | **Main single-dyad estimation** | ~60–90 min |
| 7 | `06_panel_network_causal_forest_merge` | **Panel + network extension** | ~50–70 min |
>>>>>>> af4e5fc3b8097a03c8375cf6e8ea8f47b62d4623

NB05 and NB06 are the primary outputs. All earlier notebooks feed into them and only need to
be rerun if you want to change the data pipeline.

---

## Data Notes

**What is included in the repository:**

- `Saadaoui_2026_JCE.dta`: the replication dataset from Saadaoui's public GitHub package.
  Contains bilateral PRI series for several China dyads, macro controls, 1990–2022.
- Processed NLP output files in `data/03_nlp/`: monthly bilateral panels for GDELT,
  ICEWS, and Phoenix, already cleaned and standardised.
- All `outputs/` files: figures and CSVs from each notebook run.

**What is NOT included (file size):**

- Raw GDELT event tables (~40 GB uncompressed). The `data_acquisition/` notebooks
  reproduce them if needed, but the processed monthly aggregates in `data/03_nlp/` are
  sufficient for all downstream analysis.
- Raw ICEWS and Phoenix event-level files. Same reason, processed monthly panels are
  already in the repository.
- FinBERT model weights. Downloaded automatically from HuggingFace on first run of NB02.

---

## Key Findings

**Replication.** First-stage F = 242 for Δ²PRI (CHN-USA). LP-IV h=6 β = −0.158*,
sign reversal at h≈15, h=36 β = +0.152*. Matches Saadaoui (2026) Stata output.

**DML (XGBoost depth=3, preferred).** h=6 β = −0.224 (41% larger than baseline).
Sign pattern preserved. Depth=5 shows overfitting at h=12 and is reported as sensitivity only.

**CH-IVQR (Algorithm 1).** At h=6: τ=0.10: −0.252*, τ=0.25: −0.142*, τ=0.50: −0.037 (n.s.),
τ=0.75: −0.089*, τ=0.90: −0.185*. The median effect is near zero; the LP-IV average is driven
by tail heterogeneity. At h=24 positive effects concentrate in upper quantiles.

**Structural break.** Sup-Wald (Andrews 1993) identifies break at 2002-02-01 (EP-3 incident /
WTO accession), not 2010. Chow tests confirm at h=3, 6, 12, 24.

**Multiple testing.** 20/49 horizons significant at raw 10%; 6/49 survive BH FDR correction
(h=28–35). The h=6 result does not survive FDR correction but is confirmed by wild bootstrap.

**Panel (8 valid dyads).** Oil importers (CHN-USA, CHN-JPN, CHN-GBR, CHN-AUS) show negative
h=6 coefficients. Energy exporter (CHN-RUS) shows positive. IVW pooled h=6 = −0.165*.
Pooled DML (n≈3,000, dyad FE): h=6 = −0.343*, h=36 = +0.378*.

**Causal forest.** r(ICEWS attention share, CATE) = −0.803, p < 0.0001. Months when CHN-USA
commands high share of China's bilateral activity show near-zero CATE, saturation hypothesis
confirmed.

**Network.** DY total connectedness ≈ 30–40%. GAT attention weights converge to uniform
(≈0.083 per pair) dominant common factor, not differentiated spillover.

---

## Known Limitations

- **Multiple testing.** h=6 does not survive BH correction. Results confirmed by wild bootstrap
  and Wald joint test but should be read with this caveat.
- **Composite GPR.** Weak as instrument (F=2.59). Correctly used only as nuisance control.
- **GAT.** Uniform attention weights indicate the network structure is not informative beyond
  the common factor. Reported as honest null result.
- **Meta-regression.** n=8 dyads, severely underpowered. Scatter plots are the valid output;
  regression coefficients are reported for direction only.
- **Sample size.** n=385 for CHN-USA. Precision declines at long horizons.
- **FinBERT coverage** starts 2013; Phoenix SWB ends 2019. Composite GPR coverage varies
  by source and dyad, documented in NB02.

---

## Issues and Contact

If you encounter execution problems, missing files, or have questions about the methodology:

- **Open a GitHub Issue:** [github.com/mountaha-ghabri/geopolitical-ml-oil/issues](https://github.com/mountaha-ghabri/geopolitical-ml-oil/issues)
- **Email:** moontahaghabry@gmail.com

Please include the notebook name, the cell that failed, and the error message.
