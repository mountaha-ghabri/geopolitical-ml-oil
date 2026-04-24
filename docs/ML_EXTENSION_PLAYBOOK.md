# ML Extension Playbook (Saadaoui 2026)

This document records what failed, why it failed, and what is accepted as stable.

## 1) Stable baseline to protect

Keep the calibrated replication as the reference baseline:
- LP-IV dynamics with Stata-aligned structure (`yl(3)`, `sl(2)`)
- validated against Stata log with horizon-level parity checks

Do not replace this baseline; extend around it.

## 2) Known failure modes and operational fixes

### A. Tree marginal-effect IRF failed (flat lines)
- **Why**: in small samples, shifting one feature by +1 often does not change tree leaf assignment.
- **Fix**: do not use mean-shift delta prediction as primary causal IRF for RF/XGBoost.
- **Action**: use DML residual-on-residual IV mapping for causal estimates.

### B. Cross-fitting mismatch (in-sample Z residuals)
- **Why**: residualizing `Z` in-sample while `Y,T` are out-of-fold breaks orthogonality and kills first stage.
- **Fix**: residualize `Y,T,Z` all within the same fold loop.
- **Action**: implemented in `src/dml_iv_lp.py`.

### C. XGBoost instability on T~386
- **Why**: variance and overfitting dominate unless aggressively regularized.
- **Fix**: prefer `LassoCV` or `RidgeCV` as default nuisance learners.
- **Action**: default learner in module is `LassoCV`.

### D. Time-series leakage concerns
- **Why**: standard KFold can mix temporal structure.
- **Fix**: use `TimeSeriesSplit` for cross-fitting.
- **Action**: module uses `TimeSeriesSplit` by default.

### E. Weak structural signal despite decent F
- **Why**: Wald ratio can explode when denominator (`gamma`) is small/noisy.
- **Fix**: add weak-IV safeguards using both F-stat and minimum `|gamma|`.
- **Action**: module sets `weak_iv=True` and returns `NaN` beta when unstable.

## 3) ML Approach Search Results (2026-04-19)

Systematic search over 386 obs, 9 X cols with various learners:

| Model | Weak-IV | Median F | MAE vs Baseline | MAE vs Stata |
|-------|--------|---------|---------------|-------------|
| **Ridge_α10** | 0.0% | 52.2 | **0.1032** | **0.1015** |
| LinearRegression | 0.0% | 2115.6 | 0.1408 | 0.1391 |
| Ridge_α0.1 | 0.0% | 1368.8 | 0.1629 | 0.1612 |
| RidgeCV | 0.0% | 1938.5 | 0.1731 | 0.1714 |
| BayesianRidge | 0.0% | 1774.1 | 0.1761 | 0.1744 |
| Ridge_α1 | 0.0% | 295.7 | 0.1988 | 0.2004 |
| Ridge_α100 | N/A | N/A | no valid | no valid |

**Key finding**: Higher regularization (Ridge α=10) gives **lower MAE** despite lower F-stat.
- Counter-intuitive: lower F (52) beats very high F (2115)
- Explanation: regularization stabilizes first-stage denominator (gamma), reducing Wald ratio variance

**Best approach**: Ridge with α=10 provides best balance of stability and accuracy.

## 4) Reproducible protocol

For each dyad/outcome/horizon path:
1. Run calibrated linear baseline first.
2. Run DML-IV with Ridge(α=10) as default.
3. Report:
   - median and minimum first-stage F across horizons
   - share of horizons flagged `weak_iv`
   - IRF shape compared to baseline
4. If weak/unstable:
   - keep linear baseline as main result
   - relegate DML path to robustness appendix

## 5) Writing guidance (thesis/paper)

Use this language:
- "ML is used to relax functional-form assumptions while preserving the original identification strategy."
- "When DML diagnostics indicate weak horizon-level identification (low F or small first-stage slope), we treat those IRFs as unreliable and report linear LP-IV as the primary estimate."
- "Given sample size constraints, sparse linear nuisance learners produce more stable orthogonalization than boosted trees."

## 6) Practical next tasks

- Use Ridge(α=10) as default DML learner for robustness checks
- Test on Japan-China dyad
- Validate panel results before publishing