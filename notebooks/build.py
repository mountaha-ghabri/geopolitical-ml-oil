import json, copy

with open('./05_panel_rebuilt.ipynb', 'r', encoding='utf-8') as f:
    nb_orig = json.load(f)
C = nb_orig['cells']

_id = 0
def nid():
    global _id; _id += 1; return f"nb05f_{_id:04d}"

def md(src):
    return {"cell_type":"markdown","id":nid(),"metadata":{},"source":src}

def keep(i):
    c = copy.deepcopy(C[i]); c['id'] = nid(); return c

def new_code(src):
    lines = src.strip().split('\n')
    return {"cell_type":"code","execution_count":None,"id":nid(),
            "metadata":{},"outputs":[],
            "source":[l+'\n' for l in lines[:-1]]+[lines[-1]]}

cells = []

# HEADER
cells.append(md(
"""# Notebook 05 - Panel LP-IV, NLP Integration, Network Heterogeneity, and Causal Forest

## Research questions

1. **External validity:** does the CHN-USA geopolitical oil-price channel hold across China's other bilateral relationships?
2. **NLP contribution:** does the composite GPR index change causal estimates when used as a panel nuisance control?
3. **Multi-outcome:** which channels does the shock transmit through?
4. **Network heterogeneity:** does China's network position moderate the causal transmission?
5. **Dyad heterogeneity:** what characteristics explain why signs differ across dyads?

## Sections

| Section | Content |
|---------|---------|
| A | Setup, data loading, d2PRI construction for 12 dyads |
| B | First-stage F-statistics; instrument validity |
| C | Panel LP-IV for 8 valid dyads; sign consistency; heatmap |
| D | CHN-USA annotated IRF with bilateral event history |
| E | Composite GPR as panel nuisance control |
| F | IVW pooled estimator across 3 strong dyads |
| G | Anderson-Rubin CIs for weak-instrument dyads |
| H | Network metrics: PRI-based and ICEWS-based |
| I | Multi-outcome LP-IV: Brent, industrial production, VIX |
| J | Causal Forest with attention-share moderator + placebo |
| K | Graph Attention Network: spillover structure |
| K2 | GAT controlled regression |
| L | Meta-regression: explaining cross-dyad sign heterogeneity |
| M | Notebook summary and findings |
| App | Instrument validity diagnostics |

## Key results

| Result | Value |
|--------|-------|
| Valid dyads (F>=10) | 8 of 12 |
| Sign consistent at h=6 (cooperation -> lower WTI) | CHN-USA, CHN-JPN, CHN-AUS, CHN-GBR |
| CHN-USA composite GPR deviation at h=6 | +5.7% (stable) |
| WTI sig horizons | 21/49 |
| Brent | 11/49 sig |
| indpro | 0/49 (no real-economy channel) |
| VIX | 5/49 (reduced form) |
| Causal forest r(attention_share, CATE) | -0.803, p<0.0001 (saturation) |
| GAT centrality r with log WTI | +0.586, p<0.0001 |

## Literature

| Method | Reference |
|--------|-----------|
| LP-IV | Jorda (2005) AER; Ramey (2016) NBER |
| Stock-Yogo weak IV | Stock & Yogo (2005) |
| Anderson-Rubin CI | Anderson & Rubin (1949) |
| IVW meta-analysis | Borenstein et al. (2009) |
| Causal Forest | Wager & Athey (2018) JASA; Athey et al. (2019) AOS |
| GAT | Velickovic et al. (2018) ICLR |
| Meta-regression | Thompson & Sharp (1999) Stat Med |
"""))

# A: SETUP
cells.append(md(
"""---
## Section A - Imports and Data Loading

### A1. Libraries

`networkx` constructs the bilateral event graph. `econml.CausalForestDML` implements the doubly-robust causal forest (Athey et al. 2019). `torch` enables the Graph Attention Network (Velickovic et al. 2018). All are optional; cells degrade gracefully if absent.
"""))
cells.append(keep(2))

cells.append(md(
"""### A2. Saadaoui dataset: 12 bilateral PRI series

The published dataset contains log PRI for all 12 China bilateral dyads, 1990-01 to 2022-02, 386 observations, 100% coverage for every dyad.

**d2PRI construction:** CHN-USA and CHN-JPN have d2PRI pre-computed in the published data. For the remaining 10 dyads a 5-month rolling mean is applied before computing the second difference. Pre-smoothing prevents near-collinearity between the constructed d2PRI and the lpri treatment lags in the first stage — without it, the first-stage F-statistic for smooth PRI series (CHN-PAK, std=0.055) would reflect spurious collinearity.

**Composite GPR merge:** the composite GPR panel from NB02 is pivoted by dyad and merged as `cgpr_{dyad}` columns. CHN-KOR is absent (not among the original 11 dyads).

**ICEWS attention share:** monthly ratio of CHN-USA ICEWS event volume to total China bilateral event volume. Available for 314 of 386 months (1995 onwards).
"""))
cells.append(keep(4))

# B: FIRST STAGE
cells.append(md(
"""---
## Section B - First-Stage F-Statistics

### B1. Instrument validity for 12 dyads

Treatment lags (lpri at t-1, t-2) are excluded from the first-stage control set. Because d2PRI is the second difference of lpri, including its own lags in the first stage creates near-perfect collinearity and artificially inflates F. The lags are retained in the second-stage LP-IV.

**F-statistic by dyad:** high F reflects a PRI series with many sharp reversals. CHN-PAK (std=0.055) barely changes month-to-month, making d2PRI negligible and the instrument weak. The four excluded dyads (CHN-IDN, CHN-PAK, CHN-VNM, CHN-KOR) are reported with Anderson-Rubin intervals in Section G.

**Thresholds:** F>=10 (Stock-Yogo 10% max size distortion), F>=16.38 (5%). STRONG denotes F>=100.
"""))
cells.append(keep(6))

# C: PANEL LP-IV
cells.append(md(
"""---
## Section C - Panel LP-IV across Valid Dyads

### C1. LP-IV for 8 valid dyads

Specification follows Saadaoui (2026) Equation 5 for each dyad: `lpri_{dyad}` instrumented by `d2PRI_{dyad}`, 3 lags of log WTI, 2 lags of `lpri_{dyad}`, BASE_CONTROLS. Standard errors are kernel Newey-West HAC.

**Large coefficients at long horizons** (CHN-DEU h=12: +2.04, CHN-AUS h=36: +2.25) are IV fat-tail artefacts. With moderate instruments (F=14-33), the IV estimator is consistent but heavy-tailed. As horizon increases, effective sample shrinks and fat-tail behaviour worsens. These are directional only. The heatmap clips at +/-0.5.

**Sign pattern at h=6:**
- Negative (4/8): CHN-USA, CHN-JPN, CHN-AUS, CHN-GBR -- oil-importing economies; cooperation reduces uncertainty premia on WTI
- Positive (4/8): CHN-RUS (energy export relationship; cooperation may raise supply expectations), CHN-FRA, CHN-DEU (borderline F, IV noise), CHN-IND (near-zero +0.095)
"""))
cells.append(keep(8))

cells.append(md(
"""### C2. Panel IRF plots and sign consistency

Figure N1 (left): overlaid IRFs for all 8 valid dyads; CHN-USA highlighted with 90% CI.

Figure N2 (right): fraction of valid dyads with negative coefficient at each horizon. Horizons where more than half agree in sign provide cross-dyad external validity.

Heatmap: full coefficient matrix clipped at +/-0.5.
"""))
cells.append(keep(10))

# D: ANNOTATED IRF
cells.append(md(
"""---
## Section D - CHN-USA IRF with Bilateral Event Annotations

Annotating the IRF with known bilateral events provides qualitative validation. The short-run negative range (h=3-6) corresponds to the period immediately following a diplomatic shock, when uncertainty premia are most concentrated. The medium-run positive range (h=24-36) corresponds to the horizon over which economic cooperation translates into demand growth.

**Conflict events (red):** expected to produce deterioration (positive d2PRI shock) -> negative WTI impact at short horizons. **Cooperation events (blue):** expected improvement -> positive WTI at short horizon per the full-sample sign convention.

The annotations mark the calendar date of each event, not the impulse response horizon.
"""))
cells.append(keep(12))

# E: COMPOSITE GPR
cells.append(md(
"""---
## Section E - Composite GPR as Panel Nuisance Control

### E1. Stability test across valid dyads

The dyad-specific composite GPR (lagged one period) is appended to BASE_CONTROLS and the LP-IV is re-run for each valid dyad. Deviation from the BASE_CONTROLS-only baseline indicates whether the NLP index absorbs confounding.

**Interpretation by dyad:**
- CHN-USA (+5.7%), CHN-JPN (-2.5%), CHN-GBR (+0.2%): stable -- NLP composite does not change identification for well-identified dyads
- CHN-IND (-81.4%): fragile instrument (F=11.62); adding any regressor can destabilise borderline IV -- this reflects instrument weakness, not NLP-driven confounding
- CHN-DEU (+46.7%): same interpretation; borderline F=28 with shrinking sample

For the two strong dyads, the composite GPR is a valid panel nuisance control that does not alter causal conclusions.
"""))
cells.append(keep(14))

# F: IVW
cells.append(md(
"""---
## Section F - IVW Pooled Estimator: Three Strong Dyads

### F1. Rationale

The IVW estimator (Borenstein et al. 2009) computes a precision-weighted average of dyad-specific IRFs:

$$\\hat{\\beta}^{IVW}_{h} = \\frac{\\sum_d \\hat{\\beta}_{d,h}/\\hat{\\sigma}^2_{d,h}}{\\sum_d 1/\\hat{\\sigma}^2_{d,h}}$$

This gives more weight to more precisely estimated dyads and provides a single summary of the cross-dyad evidence without imposing homogeneity.

### F2. Dyad selection: CHN-USA, CHN-JPN, CHN-GBR

These three share the oil-import demand channel (all net oil importers, none major energy exporters) and have the three strongest instruments (F=236, 128, 81). CHN-AUS is excluded despite valid F=33 because its h=36 coefficient (+2.25) signals IV fat-tail instability at longer horizons.

The IVW recovers a precision-weighted central tendency across three independently identified dyads. The between-dyad variation is visible in the individual IRF figure.
"""))
cells.append(new_code(
"""# F: IVW pooled estimator across 3 strong dyads
STRONG_DYADS = ['CHN-USA', 'CHN-JPN', 'CHN-GBR']
palette_s = {'CHN-USA': '#C0392B', 'CHN-JPN': '#2980B9', 'CHN-GBR': '#27AE60'}

print("IVW pooled estimator:", STRONG_DYADS)
print()

ivw_h, ivw_c, ivw_se = [], [], []
for h in HORIZONS:
    wts, coefs = [], []
    for dyad in STRONG_DYADS:
        if dyad not in irf_store or h not in irf_store[dyad]['h']:
            continue
        idx = irf_store[dyad]['h'].index(h)
        se  = irf_store[dyad]['se'][idx]
        if se > 0:
            wts.append(1/se**2)
            coefs.append(irf_store[dyad]['coef'][idx])
    if len(wts) < 2:
        continue
    tw = sum(wts)
    ivw_coef  = sum(w*c for w,c in zip(wts,coefs)) / tw
    ivw_sehat = 1 / tw**0.5
    ivw_h.append(h); ivw_c.append(ivw_coef); ivw_se.append(ivw_sehat)

n_sig_ivw = sum(1 for c,se in zip(ivw_c,ivw_se) if se>0 and abs(c/se)>Z90)
print(f"IVW: {n_sig_ivw}/{len(ivw_h)} significant at 10%")
for hh in [6, 12, 24, 36]:
    if hh in ivw_h:
        idx = ivw_h.index(hh)
        c=ivw_c[idx]; se=ivw_se[idx]; t=c/se
        sig = '*' if abs(t) > Z90 else ''
        print(f"  h={hh:2d}: beta={c:+.4f}  se={se:.4f}  t={t:+.2f}{sig}")

pd.DataFrame({'h':ivw_h,'coef':ivw_c,'se':ivw_se}).to_csv(OUT/'irf_ivw_pooled.csv', index=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.axhline(0, color='black', lw=0.7)
ax.axvspan(0,6,alpha=0.05,color='grey'); ax.axvspan(18,36,alpha=0.05,color='steelblue')
for dyad in STRONG_DYADS:
    if dyad not in irf_store: continue
    irf = irf_store[dyad]
    hs=np.array(irf['h']); cs=np.array(irf['coef']); ses=np.array(irf['se'])
    ax.plot(hs, cs, color=palette_s[dyad], lw=1.8, label=dyad)
    ax.fill_between(hs, cs-Z90*ses, cs+Z90*ses, alpha=0.10, color=palette_s[dyad])
ax.set_xlabel('Horizon h (months)'); ax.set_ylabel('Causal effect on log WTI')
ax.set_title('Panel LP-IV: Three Strongest Dyads (F>=80)\\n'
             'CHN-AUS excluded: long-horizon instability (h=36: +2.25)',
             fontsize=9, fontweight='bold')
ax.legend(fontsize=9)

ax = axes[1]
ax.axhline(0, color='black', lw=0.7)
ax.axvspan(0,6,alpha=0.05,color='grey'); ax.axvspan(18,36,alpha=0.05,color='steelblue')
hs=np.array(ivw_h); cs=np.array(ivw_c); ses=np.array(ivw_se)
ax.plot(hs, cs, color='#2C3E50', lw=2.2, label='IVW pooled (USA+JPN+GBR)')
ax.fill_between(hs, cs-Z90*ses, cs+Z90*ses, alpha=0.14, color='#2C3E50')
for dyad in STRONG_DYADS:
    if dyad not in irf_store: continue
    irf = irf_store[dyad]
    ax.plot(np.array(irf['h']), np.array(irf['coef']),
            color=palette_s[dyad], lw=0.8, ls='--', alpha=0.5, label=dyad)
ax.set_xlabel('Horizon h (months)'); ax.set_ylabel('IVW causal effect on log WTI')
ax.set_title('IVW Pooled Estimator\\nPrecision-weighted average: CHN-USA, CHN-JPN, CHN-GBR',
             fontsize=9, fontweight='bold')
ax.legend(fontsize=8, ncol=2)

plt.tight_layout()
plt.savefig(OUT/'fig_strong_dyads_ivw.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: fig_strong_dyads_ivw.png, irf_ivw_pooled.csv")
"""))

# G: AR CIs
cells.append(md(
"""---
## Section G - Anderson-Rubin Confidence Sets for Weak-Instrument Dyads

For the four excluded dyads (CHN-IDN, CHN-PAK, CHN-VNM, CHN-KOR, all F<10), standard Wald LP-IV intervals are unreliable. The IV estimator has a fat-tailed finite-sample distribution when the instrument is weak, and the asymptotic normal approximation overstates precision.

Anderson-Rubin (1949) confidence sets are valid regardless of instrument strength. The procedure inverts the AR test: for a grid of candidate parameter values theta_0, it tests H0: beta=theta_0 and retains all values where the null is not rejected at level alpha. The AR statistic is:

F_AR(theta_0) = [(y - theta_0 * D)' P_Z (y - theta_0 * D)] / sigma^2

where P_Z is the projection onto the instrument.

**If the AR CI spans the full grid (-5, +5):** the data contain essentially no information about the causal effect for that dyad. This is the honest result -- the instrument is insufficient, and LP-IV point estimates should not be reported for those dyads.

Widths are compared to the Wald CI width to show how much precision is overstated by ignoring instrument weakness.
"""))
cells.append(new_code(
"""# G: Anderson-Rubin CIs for weak dyads
from scipy.stats import f as f_dist

def ar_ci(df_in, lpri_col, d2_col, controls, h,
          beta_grid=None, alpha=0.10, min_obs=50):
    if beta_grid is None:
        beta_grid = np.linspace(-5, 5, 400)
    w = df_in.copy()
    w['_y'] = w['lwti'].shift(-h)
    for l in range(1,4): w[f'_Ly{l}'] = w['lwti'].shift(l)
    lags = [f'_Ly{l}' for l in range(1,4)]
    need = ['_y', lpri_col, d2_col] + lags + controls
    sub  = w[[c for c in need if c in w.columns]].dropna()
    if len(sub) < min_obs: return None, None

    cl    = _prune(sub, lags + controls)
    y_arr = sub['_y'].values
    d_arr = sub[lpri_col].values
    z_arr = sub[d2_col].values
    Xmat  = add_constant(sub[cl], has_constant='add').values
    n, k  = len(y_arr), Xmat.shape[1]

    Q    = np.linalg.lstsq(Xmat, np.eye(n), rcond=None)[0]
    Mmat = np.eye(n) - Xmat @ Q.T
    z_dem = Mmat @ z_arr
    z2    = z_dem @ z_dem

    accept = []
    for b0 in beta_grid:
        ytilde = y_arr - b0 * d_arr
        e = Mmat @ ytilde
        num = (z_dem @ ytilde)**2 / z2
        den = (e @ e) / max(n - k - 1, 1)
        F_ar = num / den if den > 0 else np.inf
        crit = f_dist.ppf(1 - alpha, 1, n - k - 1)
        accept.append(F_ar <= crit)

    accept = np.array(accept)
    if not accept.any():
        return None, None
    bg = np.array(beta_grid)
    return float(bg[accept][0]), float(bg[accept][-1])

WEAK_DYADS = ['CHN-IDN', 'CHN-PAK', 'CHN-VNM', 'CHN-KOR']
print("Anderson-Rubin 90% Confidence Sets -- Weak Instrument Dyads")
print("="*68)
print(f"{'Dyad':10s}  {'h':>4}  {'F-stat':>8}  {'AR lower':>9}  {'AR upper':>9}  Width  Uninformative?")
print("-"*68)

ar_results = []
for dyad in WEAK_DYADS:
    lpri_col, d2_col, _ = DYADS[dyad]
    row_fs = fs_df[fs_df['dyad']==dyad]
    f_stat = float(row_fs['F'].values[0]) if len(row_fs) else float('nan')
    for h_ar in [6, 36]:
        lo, hi = ar_ci(df, lpri_col, d2_col, BASE_CONTROLS, h_ar)
        if lo is None:
            print(f"{dyad:10s}  {h_ar:>4}  {f_stat:>8.2f}  {'EMPTY':>9}  {'':>9}  ---    all rejected")
            ar_results.append({'dyad':dyad,'h':h_ar,'f_stat':f_stat,'ar_lo':None,'ar_hi':None})
        else:
            width = hi - lo
            spans = (lo <= -4.8 and hi >= 4.8)
            print(f"{dyad:10s}  {h_ar:>4}  {f_stat:>8.2f}  {lo:>+9.3f}  {hi:>+9.3f}  "
                  f"{width:>5.2f}  {'YES -- no information' if spans else ''}")
            ar_results.append({'dyad':dyad,'h':h_ar,'f_stat':f_stat,'ar_lo':lo,'ar_hi':hi})

pd.DataFrame(ar_results).to_csv(OUT/'ar_ci_weak_dyads.csv', index=False)
print()
print("Saved: ar_ci_weak_dyads.csv")
print()
print("Rule: AR width ~ 10 (grid -5 to +5) = instrument provides no causal information.")
print("These dyads are excluded from the main panel LP-IV table.")
print("Wald CIs from LP-IV for weak instruments would be misleadingly narrow.")
"""))

# H: NETWORK METRICS
cells.append(md(
"""---
## Section H - Network Metrics

### H1. Guard and construction

Three time-varying network metrics are constructed and used in Section J (causal forest) and Section K (GAT):

**`usa_d2pri_share`:** absolute CHN-USA d2PRI as a fraction of total absolute d2PRI across all 12 dyads per month (mean=0.472, std=0.342). High values indicate US-China turning points dominate China's bilateral geopolitical shock landscape.

**`usa_pri_relative`:** CHN-USA log PRI minus cross-dyad mean log PRI (mean=-1.295, std=1.090). Persistently negative values indicate the US-China relationship quality is below China's average bilateral relationship throughout the sample -- consistent with strategic competition dynamics.

**`usa_attention_share` (ICEWS):** CHN-USA event volume as a fraction of total China bilateral event volume per month (available 1995-2022, 314/386 months). A media and activity-based measure that captures how much of China's observed bilateral diplomatic activity is directed at the US.
"""))
cells.append(keep(16))
cells.append(keep(17))

# I: MULTI-OUTCOME
cells.append(md(
"""---
## Section I - Multi-Outcome LP-IV

### I1. Transmission channel identification

| Outcome | Type | Expected result |
|---------|------|----------------|
| Log WTI | LP-IV (causal) | Negative h=6 -- primary finding |
| Brent crude | LP-IV (causal) | Similar to WTI if global oil channel |
| US industrial production | LP-IV (causal) | Near-zero if oil-specific, not real-economy |
| VIX | Reduced form (not causal) | Positive if uncertainty channel |

**VIX caveat:** VIX measures financial uncertainty. A positive coefficient means geopolitical improvement (rising lpri) is associated with higher VIX at horizon h. This is a reduced-form test; reverse causality cannot be ruled out. Treat as descriptive only.

**Key diagnostic:** if WTI and Brent both respond but industrial production does not, the transmission is oil-market specific (uncertainty premium) rather than through real-economy trade flows. The zero industrial production response (0/49 significant) supports the oil-channel interpretation and is itself an informative finding.
"""))
cells.append(keep(19))

# J: CAUSAL FOREST
cells.append(md(
"""---
## Section J - Causal Forest: Network-Moderated Heterogeneity

### J1. CausalForestDML with attention-share moderator

`CausalForestDML` (Wager & Athey 2018; Athey et al. 2019) learns the conditional average treatment effect (CATE) as a function of a moderator X by combining double/debiased ML nuisance estimation with a random forest that splits observations to maximise treatment effect heterogeneity.

**Moderator:** `usa_attention_share` (z-standardised). This captures what fraction of China's bilateral diplomatic event activity is directed at the US each month.

**Saturation hypothesis:** when the US-China relationship commands a large attention share, oil markets have already priced in the geopolitical signal -- turning points carry less marginal surprise, producing weaker WTI responses. The prediction is r(attention_share, CATE) < 0.

**Nuisance W:** BASE_CONTROLS excluding `brent` and `gold`. Both commodity prices share a causal pathway with WTI through geopolitical shocks and must not enter the nuisance partialling step.

**Sample:** 308 monthly observations at h=6 (ICEWS coverage from 1995).

**Cross-fitting:** 5 folds, RandomForestRegressor(max_depth=4, n_estimators=200); 300-tree causal forest.
"""))
cells.append(keep(21))

cells.append(md("""### J2. CATE figures N5 and N6
"""))
cells.append(keep(22))

cells.append(md(
"""### J3. Placebo test for causal forest heterogeneity

The negative correlation r=-0.803 between attention_share and CATE could be driven by the correlation structure of the moderator itself rather than genuine treatment effect heterogeneity. The placebo test shuffles the moderator, breaking any genuine relationship, and re-estimates the causal forest. Under the null of no heterogeneity, the shuffled correlation should be near zero.

If the true r is far from the placebo distribution (|z| > 2), the heterogeneity finding survives this robustness check.
"""))
cells.append(new_code(
"""# J3: Causal Forest placebo test
if HAS_ECONML and 'cate' in dir() and 'X' in dir():
    print("Causal Forest Placebo Test -- shuffled moderator")
    print("="*55)
    print(f"True r(attention_share, CATE): {r_mod:+.4f}  p={p_mod:.4f}")
    print()

    N_PLACEBO = 10
    rng_pb = np.random.RandomState(0)
    placebo_rs = []

    for p_idx in range(N_PLACEBO):
        X_shuf = X.copy()
        rng_pb.shuffle(X_shuf[:, 0])

        cf_pb = CausalForestDML(
            model_y=RandomForestRegressor(n_estimators=100, max_depth=4, random_state=p_idx),
            model_t=RandomForestRegressor(n_estimators=100, max_depth=4, random_state=p_idx),
            n_estimators=200, random_state=p_idx, cv=5
        )
        cf_pb.fit(Y, T, X=X_shuf, W=W)
        cate_pb = cf_pb.effect(X_shuf)
        r_pb, _ = stats.pearsonr(X_shuf[:, 0], cate_pb)
        placebo_rs.append(r_pb)
        print(f"  Placebo {p_idx+1:2d}: r(shuffled, CATE) = {r_pb:+.4f}")

    mean_pb = np.mean(placebo_rs)
    std_pb  = np.std(placebo_rs)
    z_vs_pb = (r_mod - mean_pb) / std_pb if std_pb > 0 else float('nan')
    print(f"\n  Placebo mean: {mean_pb:+.4f}  std: {std_pb:.4f}")
    print(f"  True r = {r_mod:+.4f}  vs placebo [{min(placebo_rs):+.4f}, {max(placebo_rs):+.4f}]")
    print(f"  z-score (true vs placebo): {z_vs_pb:+.2f}")
    if abs(z_vs_pb) > 2:
        print("  True correlation is distinguishable from placebo (|z| > 2).")
        print("  Heterogeneity finding is not driven by moderator correlation structure.")
    else:
        print("  True correlation NOT clearly distinguishable from placebo.")
        print("  Interpret causal forest heterogeneity with caution.")

    pd.DataFrame({'permutation': range(1, N_PLACEBO+1),
                  'r_placebo': placebo_rs}).to_csv(OUT/'cf_placebo_test.csv', index=False)
    print("Saved: cf_placebo_test.csv")
else:
    print("Run Section J1 first.")
"""))

# K: GAT
cells.append(md(
"""---
## Section K - Graph Attention Network

### K1. Architecture and training

A two-layer Graph Attention Network (Velickovic et al. 2018, ICLR) is trained on the panel of 12 bilateral PRI series:

- **Nodes:** 12 partner countries in DYADS
- **Node features at time t:** [lpri_{China-X,t}, d2PRI_{China-X,t}]
- **Edges:** fully connected (each node attends to all 11 others)
- **Objective:** predict lpri_{t+1} for each node given the full network state at t (self-supervised)
- **Architecture:** GATLayer(in=2, hidden=16) -> GATLayer(hidden=16, out=1), LeakyReLU, dropout=0.1

The attention coefficient a_{ij,t} quantifies how much node j's bilateral state at t is informative for predicting node i's bilateral state at t+1. These weights capture dynamic spillover structure across China's bilateral relationships.

**Attention weight result:** all 12 nodes receive approximately equal attention (approximately 0.083 each). This means the GAT does not identify a specific subset of bilateral relationships as especially predictive of others. The most likely explanation is a dominant common factor (China's aggregate diplomatic posture) that all 12 bilateral series share -- the network state evolves such that all bilateral signals are roughly equally informative about any other.

**GAT centrality r=+0.586 with log WTI:** periods when the CHN-USA bilateral node occupies a more prominent position in the learned network embedding correspond to higher oil price levels. Section K2 tests whether this survives macro controls.
"""))
cells.append(keep(28))

cells.append(md("""### K2. GAT correlation -- deduplication and cross-check
"""))
cells.append(keep(29))

cells.append(md(
"""### K3. Controlled regression: does GAT centrality add beyond standard controls?

The raw r=+0.586 does not account for shared macro factors driving both GAT centrality and oil prices. This cell runs OLS and IV regressions of log WTI on GAT centrality plus standard macro controls and WTI lags. If centrality remains significant, it adds incremental information beyond what standard controls capture -- elevating the result from descriptive correlation to conditional predictive power.
"""))
cells.append(new_code(
"""# K3: Controlled regression for GAT centrality
if 'cent_df' in dir() and 'CHN-USA' in cent_df.columns:
    print("="*60)
    print("K3: GAT CENTRALITY -- CONTROLLED REGRESSION")
    print("="*60)

    gat_series = cent_df['CHN-USA'].copy()
    gat_series = gat_series.loc[~gat_series.index.duplicated(keep='first')]

    df_gat = df.copy().set_index('date')
    df_gat['gat_cent'] = gat_series.reindex(df_gat.index)
    df_gat['gat_cent_z'] = ((df_gat['gat_cent'] - df_gat['gat_cent'].mean())
                             / df_gat['gat_cent'].std())
    h_gat = 6
    df_gat['_y'] = df_gat['lwti'].shift(-h_gat)
    for l in range(1,4): df_gat[f'_Ly{l}'] = df_gat['lwti'].shift(l)
    df_gat = df_gat.reset_index()

    lags_g = [f'_Ly{l}' for l in range(1,4)]
    reg_cols = ['_y','gat_cent_z','lpri','d2pri'] + lags_g + BASE_CONTROLS
    sub_gat = df_gat[[c for c in reg_cols if c in df_gat.columns]].dropna()
    cl_g = _prune(sub_gat, lags_g + BASE_CONTROLS)

    # OLS 1: no lpri (pure descriptive)
    X_ols1 = add_constant(sub_gat[['gat_cent_z'] + cl_g])
    fit1 = sm.OLS(sub_gat['_y'], X_ols1).fit(cov_type='HC1')
    c1 = fit1.params.get('gat_cent_z', float('nan'))
    t1 = fit1.tvalues.get('gat_cent_z', float('nan'))
    p1 = fit1.pvalues.get('gat_cent_z', float('nan'))
    print(f"\nOLS-1: lwti(h=6) ~ gat_z + WTI lags + macro (no lpri)")
    print(f"  beta(gat_cent_z) = {c1:+.4f}  t={t1:+.2f}  p={p1:.4f}  "
          f"({'sig' if p1<0.10 else 'n.s.'})   R2={fit1.rsquared:.4f}  n={len(sub_gat)}")

    # OLS 2: controlling for lpri level
    X_ols2 = add_constant(sub_gat[['gat_cent_z','lpri'] + cl_g])
    fit2 = sm.OLS(sub_gat['_y'], X_ols2).fit(cov_type='HC1')
    c2 = fit2.params.get('gat_cent_z', float('nan'))
    t2 = fit2.tvalues.get('gat_cent_z', float('nan'))
    p2 = fit2.pvalues.get('gat_cent_z', float('nan'))
    print(f"\nOLS-2: lwti(h=6) ~ gat_z + lpri + WTI lags + macro")
    print(f"  beta(gat_cent_z) = {c2:+.4f}  t={t2:+.2f}  p={p2:.4f}  "
          f"({'sig' if p2<0.10 else 'n.s.'})   R2={fit2.rsquared:.4f}")

    # IV: lpri instrumented by d2pri, gat_z exogenous
    exog_iv = sub_gat[['gat_cent_z'] + cl_g].copy()
    exog_iv.insert(0,'const',1.0)
    try:
        fit_iv = IV2SLS(sub_gat['_y'], exog_iv,
                        sub_gat[['lpri']], sub_gat[['d2pri']]).fit(cov_type='kernel')
        c_lpri_iv = float(fit_iv.params.get('lpri', float('nan')))
        se_lpri_iv = float(fit_iv.std_errors.get('lpri', float('nan')))
        c_gat_iv = float(fit_iv.params.get('gat_cent_z', float('nan')))
        se_gat_iv = float(fit_iv.std_errors.get('gat_cent_z', 1.0))
        t_gat_iv = c_gat_iv / se_gat_iv if se_gat_iv > 0 else float('nan')
        from scipy.stats import norm as norm_dist
        p_gat_iv = 2*(1 - norm_dist.cdf(abs(t_gat_iv)))
        print(f"\nIV (lpri instrumented by d2PRI) + gat_z exogenous:")
        print(f"  beta(lpri/IV) = {c_lpri_iv:+.4f}  se={se_lpri_iv:.4f}")
        print(f"  beta(gat_cent_z) = {c_gat_iv:+.4f}  t={t_gat_iv:+.2f}  p={p_gat_iv:.4f}  "
              f"({'sig' if p_gat_iv<0.10 else 'n.s.'})")
    except Exception as e:
        print(f"\nIV estimation: {e}")

    print()
    print("Interpretation guide:")
    print("  Sig in OLS-2 -> GAT centrality adds predictive power beyond PRI level + controls")
    print("  n.s. in OLS-2 -> r=0.586 is explained by shared macro factors (e.g. oil trend, VIX)")
    print("  IV coefficient stable with/without gat_z -> centrality does not confound causal estimate")
else:
    print("Run Section K1 first.")
"""))

# L: META-REGRESSION
cells.append(md(
"""---
## Section L - Meta-Regression: Why Do Dyad Signs Differ?

### L1. Research design

The panel LP-IV produces 8 dyad-level h=6 coefficients. A meta-regression (Thompson & Sharp 1999) regresses these coefficients against dyad characteristics to test whether the sign variation reflects economic structure or statistical artefacts.

**Variables:**

| Variable | Rationale |
|----------|-----------|
| `first_stage_f` | Instrument strength; if F drives the sign, variation is statistical noise |
| `energy_exporter` | Russia is a major energy exporter; cooperation raises supply expectations -> positive WTI |
| `trade_bn_usd` | Higher bilateral trade -> demand channel; cooperation raises demand expectations |
| `net_oil_importer` | Direct test: countries that import oil should show the negative uncertainty-premium channel |
| `log_trade` | Log-linear trade effect |

**Caveat:** n=8 dyads, up to 4 regressors. This regression is severely underpowered. Every coefficient will have a large standard error and p-values will be uninformative. The exercise is exploratory -- it provides an economic framework for interpreting the sign pattern, not a statistically validated explanation. State this clearly in the thesis.

**Data sources (approximate 2022 values):** UN Comtrade for trade volumes; EIA for oil importer/exporter status; Correlates of War for security alignment. Values below are approximate; replace with official figures where available.
"""))
cells.append(new_code(
"""# L: Meta-regression of dyad heterogeneity
import statsmodels.api as sm

# L1: extract h=6 coefficients
meta_rows = []
for dyad, irf in irf_store.items():
    if 6 not in irf['h']: continue
    idx = irf['h'].index(6)
    meta_rows.append({'dyad': dyad, 'coef_h6': irf['coef'][idx], 'se_h6': irf['se'][idx]})
coef_df_meta = pd.DataFrame(meta_rows)
print(f"h=6 coefficients for {len(coef_df_meta)} valid dyads")

# L2: dyad characteristics (approximate 2022 values; see docstring above)
dyad_chars = pd.DataFrame([
    ('CHN-USA', 558.0, 0, 11170, 1, 1),
    ('CHN-JPN', 317.0, 0,  2100, 1, 1),
    ('CHN-AUS', 168.0, 1,  9000, 0, 1),
    ('CHN-FRA',  65.0, 0,  8200, 1, 1),
    ('CHN-DEU', 206.0, 0,  7400, 1, 1),
    ('CHN-GBR',  86.0, 0,  8200, 1, 1),
    ('CHN-RUS', 111.0, 1,  6400, 0, 0),
    ('CHN-IND',  93.0, 0,  3800, 1, 0),
], columns=['dyad','trade_bn_usd','energy_exporter','distance_km',
            'net_oil_importer','us_security_partner'])

dyad_chars = dyad_chars.merge(
    fs_df[['dyad','F']].rename(columns={'F':'first_stage_f'}), on='dyad')
df_meta = coef_df_meta.merge(dyad_chars, on='dyad')
df_meta['log_trade'] = np.log(df_meta['trade_bn_usd'])
df_meta['sign_neg'] = (df_meta['coef_h6'] < 0).astype(int)

print(f"\nMeta-regression dataset:")
print(df_meta[['dyad','coef_h6','first_stage_f','energy_exporter',
               'net_oil_importer','trade_bn_usd']].to_string(index=False))

print("\nCAVEAT: n=8, exploratory only. Standard errors are large; do not over-interpret.")

# L3: three models
models = {
    'M1 (F-stat only)': ['first_stage_f'],
    'M2 (channel structure)': ['energy_exporter','net_oil_importer'],
    'M3 (full)': ['first_stage_f','energy_exporter','log_trade'],
}
print()
for mname, regs in models.items():
    X_m = add_constant(df_meta[regs])
    m = sm.OLS(df_meta['coef_h6'], X_m).fit()
    print(f"--- {mname}  R2={m.rsquared:.3f} ---")
    for v in regs:
        print(f"  beta({v}) = {m.params[v]:+.4f}  t={m.tvalues[v]:+.2f}  p={m.pvalues[v]:.3f}")
    print()

# L4: scatter plots
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
plot_vars = [
    ('first_stage_f', 'First-stage F-statistic'),
    ('energy_exporter', 'Energy exporter (0=No, 1=Yes)'),
    ('trade_bn_usd', 'Bilateral trade (bn USD, 2022)'),
]
for ax, (xvar, xlabel) in zip(axes, plot_vars):
    colors = ['#C0392B' if c < 0 else '#2980B9' for c in df_meta['coef_h6']]
    ax.scatter(df_meta[xvar], df_meta['coef_h6'], c=colors, s=120,
               zorder=3, edgecolors='white', linewidth=1.0)
    for _, row in df_meta.iterrows():
        label = row['dyad'].replace('CHN-','')
        ax.annotate(label, (row[xvar], row['coef_h6']),
                    textcoords='offset points', xytext=(5,4), fontsize=7.5)
    ax.axhline(0, color='black', lw=0.6, ls='--')
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel('LP-IV h=6 coefficient', fontsize=9)
    ax.set_title(xlabel.split(' (')[0], fontsize=9, fontweight='bold')

fig.suptitle('Meta-Regression: Explaining Cross-Dyad Sign Heterogeneity\\n'
             'Red = negative effect (cooperation->lower WTI); Blue = positive',
             fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT/'fig_meta_regression.png', dpi=150, bbox_inches='tight')
plt.show()
df_meta.to_csv(OUT/'meta_regression_data.csv', index=False)
print("Saved: fig_meta_regression.png, meta_regression_data.csv")
"""))

# APPENDIX
cells.append(md(
"""---
## Appendix - Instrument Validity Diagnostics by Dyad

Correlation between each constructed d2PRI and the corresponding lpri level and lags. All |r| < 0.20 confirms that the second-difference transformation successfully removes the persistent diplomatic trend and isolates abrupt turning points. CHN-GBR shows slightly higher correlations with its lags (|r| up to 0.165), but remains well within acceptable range.
"""))
cells.append(keep(26))

nb_new = {
    "nbformat": nb_orig["nbformat"],
    "nbformat_minor": nb_orig["nbformat_minor"],
    "metadata": nb_orig["metadata"],
    "cells": cells
}
with open('./05_complete.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb_new, f, indent=1, ensure_ascii=False)   
print(f"Done. {len(cells)} cells.")