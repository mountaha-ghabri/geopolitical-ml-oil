import json

with open('nb06_cells_part1.json','r') as f:
    cells = json.load(f)

def uid(s):
    import hashlib, random
    return hashlib.md5((s+str(random.random())).encode()).hexdigest()[:8]
def md(s): return {"cell_type":"markdown","id":uid(s),"metadata":{},"source":s}
def code(s): return {"cell_type":"code","id":uid(s),"metadata":{},"source":s,"outputs":[],"execution_count":None}

# SECTION 6: NETWORK
cells.append(md("""---
## 6. Network Structure and Bilateral Connectedness

This section examines whether China's 12 bilateral geopolitical relationships co-move as an interconnected system rather than as independent series. Three complementary methods are applied: PRI-based scalar network metrics, Diebold-Yilmaz forecast error variance decomposition connectedness, and Granger predictive causality testing.

Network analysis relaxes the independence assumption implicit in the panel LP-IV of Section 3. If shocks to one bilateral d2PRI series consistently predict movements in others, the panel LP-IV's assumption that each observation is drawn from an independent data-generating process is an approximation that may understate uncertainty.
"""))

cells.append(md("""### 6.1 PRI-Based Network Metrics

Two scalar metrics summarise China's bilateral network position each month.

`usa_d2pri_share` measures the fraction of China's total bilateral turning-point shock intensity attributable to the CHN-USA dyad. A value of 0.40 means the US-China relationship accounts for 40 percent of all bilateral d2PRI activity across the 12 dyads in that month.

`usa_pri_relative` measures the CHN-USA log PRI relative to the cross-dyad average. Positive values indicate the US-China relationship is of higher quality than China's average bilateral relationship; negative values indicate relative hostility in the US direction.

Both metrics are constructed in Section 1.3 and plotted here for the full sample period.
"""))

cells.append(code("""fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
ax = axes[0]
ax.fill_between(df['date'], df['usa_d2pri_share'], df['usa_d2pri_share'].mean(),
                where=df['usa_d2pri_share']>=df['usa_d2pri_share'].mean(), color='#C0392B', alpha=0.3)
ax.fill_between(df['date'], df['usa_d2pri_share'], df['usa_d2pri_share'].mean(),
                where=df['usa_d2pri_share']< df['usa_d2pri_share'].mean(), color='#2980B9', alpha=0.3)
ax.plot(df['date'], df['usa_d2pri_share'], color='#2C3E50', lw=0.8)
ax.axhline(df['usa_d2pri_share'].mean(), color='black', lw=0.5, ls='--')
ax.set_ylabel('CHN-USA share of total |d2PRI|')
ax.set_title('Network Position Metric 1: US Share of China Bilateral Shock Intensity\nHigh months = US-China turning points dominate all other bilateral activity', fontsize=9, fontweight='bold')

ax2 = axes[1]
ax2.plot(df['date'], df['usa_pri_relative'], color='#8E44AD', lw=1.0)
ax2.axhline(0, color='black', lw=0.5, ls='--')
ax2.fill_between(df['date'], df['usa_pri_relative'], 0, where=df['usa_pri_relative']>=0, color='#C0392B', alpha=0.2)
ax2.fill_between(df['date'], df['usa_pri_relative'], 0, where=df['usa_pri_relative']< 0, color='#2980B9', alpha=0.2)
ax2.set_ylabel('CHN-USA lpri minus\\nmean lpri across all dyads')
ax2.set_title('Network Position Metric 2: US Relations Relative to China Average\nPositive = US better than average; Negative = US worse than average', fontsize=9, fontweight='bold')
ax2.set_xlabel('Date')
import matplotlib.dates as mdates
for ax_ in axes:
    ax_.xaxis.set_major_locator(mdates.YearLocator(4))
    ax_.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.tight_layout()
plt.savefig(OUT/'fig_network_pri_metrics.png', dpi=150, bbox_inches='tight')
plt.show(); print("Saved: fig_network_pri_metrics.png")
"""))

cells.append(md("""### 6.2 Diebold-Yilmaz Forecast Error Variance Decomposition

The Diebold-Yilmaz (2012) connectedness index is computed from the forecast error variance decomposition of a VAR estimated on the 12 bilateral d2PRI series.

**Total connectedness** is the share of the average forecast error variance that comes from other dyads' shocks. A value of 50 percent means half of any dyad's forecast uncertainty is explained by shocks originating in other bilateral relationships.

**TO connectedness** measures how much of other dyads' forecast error variance is explained by a focal dyad's shocks. High TO values identify geopolitical transmitters.

**FROM connectedness** measures how much of a focal dyad's forecast error variance is explained by shocks from other dyads. High FROM values identify receivers.

The VAR lag length is selected by AIC. Non-stationary d2PRI series are differenced once before estimation.

⏱ Approximately 3 minutes.
"""))

cells.append(code("""from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller

def dy_connectedness(data, n_lags=None, horizon=10):
    data_diff = data.copy()
    for col in data_diff.columns:
        if adfuller(data_diff[col].dropna())[1] > 0.05:
            data_diff[col] = data_diff[col].diff()
    data_diff = data_diff.dropna()
    model     = VAR(data_diff)
    if n_lags is None:
        n_lags = max(1, model.select_order(maxlags=12).get('aic', 2))
    results   = model.fit(n_lags)
    fevd_list = results.fevd(horizon).decomp
    fevd_avg  = fevd_list.mean(axis=0)
    k         = fevd_avg.shape[0]
    total     = 100 * (1 - np.trace(fevd_avg) / k)
    to_       = 100 * (fevd_avg.sum(axis=1) - np.diag(fevd_avg))
    from_     = 100 * (fevd_avg.sum(axis=0) - np.diag(fevd_avg))
    net       = to_ - from_
    pairwise  = 100 * fevd_avg.copy(); np.fill_diagonal(pairwise, 0)
    return total, pd.Series(to_, index=data_diff.columns), pd.Series(from_, index=data_diff.columns), pd.Series(net, index=data_diff.columns), pairwise

d2_series = {dyad: df.set_index('date')[v[1]].dropna()
             for dyad, v in DYADS.items() if v[1] in df.columns}
data_d2   = pd.DataFrame(d2_series).dropna()
print(f"DY input panel: {data_d2.shape}")

total, to_, from_, net_, pairwise = dy_connectedness(data_d2, n_lags=2, horizon=12)
print(f"Total connectedness: {total:.2f}%")
print("\nTO others (net transmitters, top 5):")
print(to_.sort_values(ascending=False).head(5).round(2))
print("\nNET connectedness (positive = transmitter):")
print(net_.sort_values(ascending=False).round(2))

# Pairwise heatmap
plt.figure(figsize=(10, 8))
plt.imshow(pairwise, cmap='RdBu_r', aspect='auto', vmin=0, vmax=max(float(pairwise.max()), 10))
plt.colorbar(label='Pairwise connectedness (%)')
plt.xticks(range(len(data_d2.columns)), data_d2.columns, rotation=90)
plt.yticks(range(len(data_d2.columns)), data_d2.columns)
plt.title(f'Diebold-Yilmaz Pairwise Connectedness (h=12)\nTotal connectedness = {total:.1f}%', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT/'dy_pairwise_connectedness.png', dpi=150)
plt.show(); print("Saved: dy_pairwise_connectedness.png")
"""))

cells.append(md("""### 6.3 Rolling-Window Total Connectedness

The VAR-based DY total connectedness is re-estimated on 10-year rolling windows stepped forward by 12 months at a time. This shows whether the degree of co-movement among bilateral d2PRI series has increased or decreased over the sample period and whether it spikes during known multilateral geopolitical crises.

Expected patterns include elevated connectedness around the early-2000s tensions (EP-3 incident, post-9/11 realignment), the 2008 global financial crisis, the 2018 trade war onset, and the 2020 COVID-19 period, all of which plausibly caused simultaneous shifts in multiple bilateral relationships.
"""))

cells.append(code("""def rolling_connectedness(data, window=120, step=12, horizon=12, n_lags=2):
    dates, total_roll = [], []
    for start in range(0, len(data)-window, step):
        sub = data.iloc[start:start+window]
        if len(sub) < 50: continue
        try:
            tot, *_ = dy_connectedness(sub, n_lags=n_lags, horizon=horizon)
            dates.append(data.index[start+window-1]); total_roll.append(tot)
        except: continue
    return dates, total_roll

dates_roll, total_roll = rolling_connectedness(data_d2)
plt.figure(figsize=(12, 4))
plt.plot(dates_roll, total_roll, marker='o', ms=4, linestyle='-', color='#2C3E50')
plt.axhline(total, color='#C0392B', ls='--', lw=1.2, label=f'Full-sample total: {total:.1f}%')
plt.xlabel('End of rolling window'); plt.ylabel('Total connectedness (%)')
plt.title('Rolling-Window Total Connectedness (Diebold-Yilmaz, 10-year windows)\nEvolution of bilateral d2PRI co-movement over the sample period', fontsize=9, fontweight='bold')
plt.legend(); plt.grid(alpha=0.3)
plt.savefig(OUT/'dy_rolling_connectedness.png', dpi=150)
plt.show(); print("Saved: dy_rolling_connectedness.png")
"""))

cells.append(md("""### 6.4 Granger Predictive Causality Network

Granger causality (Granger 1969) tests whether lagged values of one bilateral d2PRI series help predict another. This section tests all ordered pairs, constructs a directed network from the significant pairs, and visualises it as a graph where edge direction shows the direction of predictive causality and edge width reflects the F-statistic.

This is a predictive, not structural, concept. A significant result from dyad i to dyad j means the China-i bilateral turning-point series leads the China-j series by one to two months. Structural interpretation requires additional assumptions beyond what the Granger test provides.
"""))

cells.append(code("""from statsmodels.tsa.stattools import grangercausalitytests
import itertools

def granger_pairs(data, max_lag=2, alpha=0.05):
    results = []
    dyads   = data.columns.tolist()
    for src, tgt in itertools.permutations(dyads, 2):
        df_pair = data[[src, tgt]].dropna()
        if len(df_pair) < 30: continue
        try:
            gc = grangercausalitytests(df_pair, maxlag=max_lag, verbose=False)
            p  = gc[max_lag][0]['ssr_ftest'][1]
            f  = gc[max_lag][0]['ssr_ftest'][0]
            if p < alpha: results.append((src, tgt, p, f))
        except: continue
    return results

print("Running Granger causality tests (max_lag=2, alpha=0.05)...")
gc_pairs = granger_pairs(data_d2, max_lag=2, alpha=0.05)
print(f"Significant directed pairs: {len(gc_pairs)}")
for src, tgt, p, f in sorted(gc_pairs, key=lambda x: -x[3]):
    print(f"  {src} -> {tgt}: p={p:.4f}  F={f:.2f}")
gc_df = pd.DataFrame(gc_pairs, columns=['source','target','p_value','f_stat'])
gc_df.to_csv(OUT/'granger_significant_pairs.csv', index=False)
print("Saved: granger_significant_pairs.csv")
"""))

cells.append(code("""try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False

if HAS_NX and gc_pairs:
    region = {
        'CHN-USA':'North America','CHN-JPN':'East Asia','CHN-AUS':'Oceania',
        'CHN-GBR':'Europe','CHN-FRA':'Europe','CHN-DEU':'Europe',
        'CHN-RUS':'Eurasia','CHN-IND':'South Asia','CHN-IDN':'Southeast Asia',
        'CHN-VNM':'Southeast Asia','CHN-PAK':'South Asia','CHN-KOR':'East Asia',
    }
    region_colors = {
        'North America':'#C0392B','Europe':'#2980B9','East Asia':'#27AE60',
        'Southeast Asia':'#F39C12','South Asia':'#8E44AD','Eurasia':'#16A085','Oceania':'#E67E22',
    }
    G   = nx.DiGraph()
    for src, tgt, p, f in gc_pairs: G.add_edge(src, tgt, pval=p, fstat=f)
    pos = nx.spring_layout(G, seed=42, k=2, iterations=60)
    node_sizes   = [G.degree(n)*300+400 for n in G.nodes()]
    node_colors  = [region_colors.get(region.get(n,'Other'),'#7F8C8D') for n in G.nodes()]
    f_vals       = [e[2]['fstat'] for e in G.edges(data=True)]
    if f_vals:
        min_f, max_f = min(f_vals), max(f_vals)
        edge_widths  = [1+4*(f-min_f)/(max_f-min_f+1e-8) for f in f_vals]
    else:
        edge_widths  = [2]*len(G.edges())
    plt.figure(figsize=(14, 10))
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.85, edgecolors='white', linewidths=1.5)
    for (s,t,d_e), w in zip(G.edges(data=True), edge_widths):
        nx.draw_networkx_edges(G, pos, edgelist=[(s,t)], width=w, alpha=0.7, arrowstyle='-|>', arrowsize=18, edge_color='#2C3E50')
    nx.draw_networkx_labels(G, {n:(pos[n][0],pos[n][1]-0.03) for n in G.nodes()}, font_size=8, font_weight='bold')
    plt.title(f'Granger Causality Network (d2PRI, p<0.05, lag=2)\n{len(gc_pairs)} significant directed pairs; edge width proportional to F-statistic', fontsize=12, fontweight='bold')
    plt.axis('off')
    handles = [plt.Line2D([0],[0], marker='o', color='w', markerfacecolor=c, markersize=10, label=r)
               for r,c in region_colors.items()]
    plt.legend(handles=handles, title='Region', loc='upper left', bbox_to_anchor=(1,1), fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT/'granger_network.png', dpi=150, bbox_inches='tight')
    plt.show(); print("Saved: granger_network.png")
elif not HAS_NX:
    print("networkx not available (pip install networkx)")
else:
    print("No significant Granger pairs at p<0.05")
"""))

cells.append(md("""### 6.5 Graph Attention Network for Bilateral Connectedness

A two-layer Graph Attention Network (Velickovic et al. 2018) is trained on the 12-node bilateral PRI panel. Each node represents a bilateral relationship; node features at time t are the log PRI level and the d2PRI value for that dyad.

The model is trained self-supervisedly to predict each node's log PRI at t+1 from the full network state at t. Attention weights are learned to reflect which nodes' contemporaneous states are most informative for predicting each other node's future state. Unlike the VAR-based Diebold-Yilmaz approach, the GAT can learn non-linear co-movement patterns.

If PyTorch is available, a full GAT with learnable W and attention matrices is used. Otherwise, a NumPy softmax attention mechanism with fixed random weights is substituted. The NumPy version captures the same graph structure but cannot adapt weights to the data.

The primary outputs are (a) a time series of node centrality embeddings for each of the 12 dyads, and (b) the mean attention weight matrix (12 by 12), which shows which dyad pairs are assigned the most attention weight by the trained network.

⏱ 3 to 5 minutes with PyTorch; 30 seconds with the NumPy fallback.
"""))

cells.append(code("""try:
    import torch, torch.nn as nn, torch.nn.functional as F_torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

NODES_GAT  = list(DYADS.keys())
NODE_COLS  = [(lc, dc) for lc, dc, _ in DYADS.values()]
valid_gat  = [(name,lc,dc) for name,(lc,dc) in zip(NODES_GAT,NODE_COLS) if lc in df.columns and dc in df.columns]
V_NAMES    = [n for n,_,_ in valid_gat]
V_LCOLS    = [lc for _,lc,_ in valid_gat]
V_DCOLS    = [dc for _,_,dc in valid_gat]
N_V        = len(valid_gat)
print(f"GAT active nodes ({N_V}): {V_NAMES}")

T_len  = len(df)
X_raw  = np.zeros((T_len, N_V, 2))
for ni,(lc,dc) in enumerate(zip(V_LCOLS,V_DCOLS)):
    X_raw[:,ni,0] = df[lc].ffill().fillna(0).values
    X_raw[:,ni,1] = df[dc].fillna(0).values
X_sc = X_raw.copy()
for ni in range(N_V):
    for fi in range(2):
        s=X_raw[:,ni,fi]; X_sc[:,ni,fi]=(s-s.mean())/(s.std()+1e-8)

if HAS_TORCH:
    class GATLayer(nn.Module):
        def __init__(self,in_f,out_f,dropout=0.1):
            super().__init__()
            self.W=nn.Linear(in_f,out_f,bias=False); self.a=nn.Linear(2*out_f,1,bias=False)
            self.drop=nn.Dropout(dropout); self.lrelu=nn.LeakyReLU(0.2)
        def forward(self,x):
            h=self.W(x); N=h.size(0)
            hi=h.unsqueeze(1).expand(-1,N,-1); hj=h.unsqueeze(0).expand(N,-1,-1)
            e=self.lrelu(self.a(torch.cat([hi,hj],-1))).squeeze(-1)
            alpha=F_torch.softmax(e,dim=1); alpha=self.drop(alpha)
            return torch.matmul(alpha,h), alpha.detach()
    class TwoLayerGAT(nn.Module):
        def __init__(self,in_f=2,hid=16,out_f=1):
            super().__init__()
            self.g1=GATLayer(in_f,hid); self.g2=GATLayer(hid,out_f)
        def forward(self,x):
            h1,a1=self.g1(x); h1=F_torch.elu(h1); h2,a2=self.g2(h1)
            return h2,a1,a2

    torch.manual_seed(42)
    gat=TwoLayerGAT(in_f=2,hid=16,out_f=1)
    opt=torch.optim.Adam(gat.parameters(),lr=1e-3,weight_decay=1e-4)
    Xt =torch.tensor(X_sc[:-1],dtype=torch.float32)
    Xt1=torch.tensor(X_sc[1:,:,0:1],dtype=torch.float32)
    losses=[]; gat.train()
    for ep in range(150):
        opt.zero_grad()
        tot=sum(F_torch.mse_loss(gat(Xt[t])[0],Xt1[t]) for t in range(len(Xt)))
        (tot/len(Xt)).backward(); opt.step(); losses.append(float(tot.item())/len(Xt))
        if (ep+1)%30==0: print(f"  Epoch {ep+1:3d}  loss={losses[-1]:.5f}")
    gat.eval()
    cent_np=np.zeros((len(Xt),N_V)); attn_np=np.zeros((len(Xt),N_V,N_V))
    with torch.no_grad():
        for t in range(len(Xt)):
            out,a1,_=gat(Xt[t]); cent_np[t]=out.squeeze(-1).numpy(); attn_np[t]=a1.numpy()
    cent_dates=df['date'].values[:-1]
    print(f"Training complete. Loss: {losses[0]:.5f} (initial) to {losses[-1]:.5f} (final)")
else:
    rng=np.random.default_rng(42); W0=rng.normal(0,0.1,(2,16))
    def _softmax(x,axis=-1):
        ex=np.exp(x-x.max(axis=axis,keepdims=True)); return ex/ex.sum(axis=axis,keepdims=True)
    cent_np=np.zeros((T_len,N_V)); attn_np=np.zeros((T_len,N_V,N_V))
    for t in range(T_len):
        H=X_sc[t]@W0; alpha=_softmax(H@H.T/np.sqrt(16)); agg=alpha@H
        cent_np[t]=agg[:,0]; attn_np[t]=alpha
    cent_dates=df['date'].values; losses=None
    print("NumPy fallback used (PyTorch not available)")

cent_df  = pd.DataFrame(cent_np, columns=V_NAMES, index=pd.to_datetime(cent_dates))
usa_share= df.set_index('date')['d2pri'].abs() / df.set_index('date')[V_DCOLS].abs().sum(axis=1,min_count=1).replace(0,np.nan)

# Plot centrality time series
fig, axes = plt.subplots(2,1,figsize=(14,8),sharex=False,gridspec_kw={'hspace':0.35})
cmap_g   = plt.cm.tab10(np.linspace(0,1,N_V))
ax = axes[0]
for ni,name in enumerate(V_NAMES):
    lw=2.0 if name=='CHN-USA' else 0.9; al=1.0 if name=='CHN-USA' else 0.5
    ax.plot(cent_df.index, cent_df[name], color=cmap_g[ni], lw=lw, alpha=al, label=name.replace('CHN-',''))
ax.axhline(0, color='black', lw=0.4)
ax.legend(fontsize=7, ncol=6, loc='upper left')
ax.set_ylabel('GAT centrality embedding')
ax.set_title('GAT Node Centrality: All 12 Partner Countries (CHN-USA bold)\nAmplitude reflects bilateral network integration', fontsize=9, fontweight='bold')
ax.xaxis.set_major_locator(mdates.YearLocator(4)); ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax2 = axes[1]
ax2.fill_between(usa_share.index, 0, usa_share.values, color='#C0392B', alpha=0.35)
ax2.plot(usa_share.index, usa_share.rolling(6).mean(), color='#C0392B', lw=1.2)
ax2.axhline(usa_share.mean(), color='black', lw=0.5, ls='--')
ax2.set_ylabel('USA |d2PRI| share'); ax2.set_xlabel('Date')
ax2.set_title('USA Dominance in China Bilateral Shock Network', fontsize=9, fontweight='bold')
ax2.xaxis.set_major_locator(mdates.YearLocator(4)); ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.savefig(OUT/'fig_gat_network.png', dpi=150, bbox_inches='tight')
plt.show()

# Mean attention matrix heatmap
mean_attn = attn_np.mean(axis=0)
plt.figure(figsize=(10,8))
plt.imshow(mean_attn, cmap='Blues', aspect='auto', vmin=0)
plt.colorbar(label='Mean attention weight')
plt.xticks(range(N_V), V_NAMES, rotation=90); plt.yticks(range(N_V), V_NAMES)
plt.title('GAT Mean Attention Weight Matrix (row i attends to column j)\nUniform pattern indicates dominant common factor', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT/'fig_gat_attention_matrix.png', dpi=150)
plt.show()

cent_df.to_csv(OUT/'gat_centrality.csv')
usa_share.to_csv(OUT/'usa_d2pri_share.csv', header=True)
pd.DataFrame(mean_attn, index=V_NAMES, columns=V_NAMES).to_csv(OUT/'gat_mean_attention.csv')
print("Saved: fig_gat_network.png, fig_gat_attention_matrix.png, gat_centrality.csv, gat_mean_attention.csv")
"""))

cells.append(md("""### 6.6 GAT Controlled Regression

A raw correlation between GAT centrality and log WTI could reflect shared macro trends rather than an independent network-position effect. This cell estimates three OLS regressions at h=6 to determine whether GAT centrality adds predictive content for future WTI after controlling for the log PRI level and standard macro variables.

Regression 1 controls for macro variables and WTI lags only, without log PRI. Regression 2 additionally controls for the log PRI level, testing whether GAT centrality captures information beyond the bilateral relationship level. Regression 3 instruments log PRI with d2PRI to remove endogeneity and tests whether GAT centrality remains significant in the IV specification.

An insignificant coefficient in Regression 2 would indicate that the raw correlation is fully explained by shared macro-PRI dynamics, consistent with a common-factor interpretation of the network structure.
"""))

cells.append(code("""if 'CHN-USA' in cent_df.columns:
    gat_s    = cent_df['CHN-USA'].loc[~cent_df.index.duplicated(keep='first')]
    df_gat   = df.copy().set_index('date')
    df_gat['gat_cent_z'] = (gat_s.reindex(df_gat.index) - gat_s.mean()) / gat_s.std()
    h_gat    = 6
    df_gat['_y']  = df_gat['lwti'].shift(-h_gat)
    for l in range(1,4): df_gat[f'_Ly{l}'] = df_gat['lwti'].shift(l)
    df_gat   = df_gat.reset_index()
    lags_g   = [f'_Ly{l}' for l in range(1,4)]
    reg_cols = ['_y','gat_cent_z','lpri','d2pri'] + lags_g + BASE_CONTROLS
    sub_gat  = df_gat[[c for c in reg_cols if c in df_gat.columns]].dropna()
    cl_g     = _prune(sub_gat, lags_g + BASE_CONTROLS)

    print(f"Sample for GAT controlled regression: n={len(sub_gat)}")
    print(f"\n{'Spec':30s}  {'beta(gat_z)':>12}  {'p':>6}  {'R2':>6}")
    print('-'*60)
    for spec_name, regs, instr in [
        ('OLS-1: no lpri',           ['gat_cent_z']+cl_g,           False),
        ('OLS-2: with lpri',         ['gat_cent_z','lpri']+cl_g,    False),
        ('IV: lpri instrumented',    ['gat_cent_z']+cl_g,            True),
    ]:
        if not instr:
            X_ = add_constant(sub_gat[regs]); fit = sm.OLS(sub_gat['_y'], X_).fit(cov_type='HC1')
            b  = fit.params.get('gat_cent_z',float('nan')); p = fit.pvalues.get('gat_cent_z',float('nan'))
            print(f"{spec_name:30s}  {b:>+12.4f}  {p:>6.4f}{'*' if p<0.10 else ' '}  {fit.rsquared:>6.4f}")
        else:
            exog_ = sub_gat[regs].copy(); exog_.insert(0,'const',1.0)
            try:
                fit_ = IV2SLS(sub_gat['_y'], exog_, sub_gat[['lpri']], sub_gat[['d2pri']]).fit(cov_type='kernel')
                b    = float(fit_.params.get('gat_cent_z',float('nan')))
                se_  = float(fit_.std_errors.get('gat_cent_z',1.0))
                p    = float(2*(1-stats.norm.cdf(abs(b/se_)))) if se_>0 else float('nan')
                print(f"{spec_name:30s}  {b:>+12.4f}  {p:>6.4f}{'*' if p<0.10 else ' '}  n/a (IV)")
            except Exception as e: print(f"{spec_name}: failed ({e})")

    print("\nInterpretation: significant in OLS-2 means GAT centrality adds beyond PRI level and macro controls.")
    print("Not significant in OLS-2 means the raw correlation reflects shared macro trends.")
else:
    print("GAT centrality not available for CHN-USA. Run Section 6.5 first.")
"""))

# SECTION 7: CAUSAL FOREST (COMPREHENSIVE)
cells.append(md("""---
## 7. Heterogeneous Treatment Effects and Causal Forest

This section applies a doubly-robust causal forest to estimate conditional average treatment effects (CATEs) for the CHN-USA dyad. The causal forest allows the treatment effect to vary with observed characteristics rather than assuming a constant effect across all time periods and market conditions.

Seven cells cover the setup, model fitting, visualisation, sub-period comparison, feature importance, and permutation placebo test. The causal forest is run at the primary horizon h=6, where the LP-IV and DML results are most consistent.
"""))

cells.append(md("""### 7.1 Data Preparation and Moderator Construction

The causal forest requires specifying a moderator vector X that conditions the treatment effect estimation. The moderator here is the ICEWS attention share (z-standardised), which represents the salience of the US-China relationship relative to China's other bilateral commitments in each month.

The saturation hypothesis predicts that months when the CHN-USA relationship commands a large share of China's bilateral activity will show smaller causal effects of bilateral turning points on WTI, because markets have already priced in the geopolitical salience. Months with low attention share may show larger effects because the CHN-USA turning point arrives as a relative surprise.

If the ICEWS attention share is unavailable (coverage begins 1995), the PRI-based `usa_d2pri_share` is substituted.
"""))

cells.append(code("""mod_col = 'usa_attention_share' if df['usa_attention_share'].notna().sum()>100 else 'usa_d2pri_share'
print(f"Moderator: {mod_col}  ({df[mod_col].notna().sum()} obs available)")
df['moderator_z'] = (df[mod_col] - df[mod_col].mean()) / df[mod_col].std()

h_cf = 6
w_cf = df.copy()
w_cf['_y'] = w_cf['lwti'].shift(-h_cf)
for l in range(1,4): w_cf[f'_Ly{l}'] = w_cf['lwti'].shift(l)
for l in range(1,3): w_cf[f'_Lt{l}'] = w_cf['lpri'].shift(l)
lags_cf     = [f'_Ly{l}' for l in range(1,4)] + [f'_Lt{l}' for l in range(1,3)]
DML_NUIS_CF = [c for c in BASE_CONTROLS if c not in ('brent','gold')]
feat_cols   = lags_cf + DML_NUIS_CF
all_need    = ['_y','lpri','d2pri','moderator_z'] + feat_cols
sub_cf      = w_cf[[c for c in all_need if c in w_cf.columns]].dropna()
feat_cols   = _prune(sub_cf, feat_cols)

Y_cf   = sub_cf['_y'].values
T_cf   = sub_cf['lpri'].values
X_cf   = sub_cf[['moderator_z']].values
W_mat  = sub_cf[feat_cols].values
dates_cf = sub_cf.index

print(f"Causal forest sample: n={len(sub_cf)}")
print(f"Date range: {df.iloc[dates_cf[0]]['date'].date()} to {df.iloc[dates_cf[-1]]['date'].date()}" if len(dates_cf)>0 else "")
print(f"Moderator z-score range: [{X_cf.min():.2f}, {X_cf.max():.2f}]")
"""))

cells.append(md("""### 7.2 CausalForestDML Fitting

The CausalForestDML from Athey et al. (2019) is estimated using RandomForestRegressor for both the outcome nuisance E[Y|X,W] and the treatment nuisance E[T|X,W]. The forest uses 300 trees with 5-fold cross-validation for cross-fitting.

The key quantity estimated is the conditional average treatment effect:

CATE(x) = E[Y(t=1) minus Y(t=0) | X = x]

where x here is the moderator value (attention share z-score) for the observation. In practice, the CATE is the estimated marginal effect of a unit improvement in bilateral log PRI on log WTI, conditional on the level of US diplomatic attention in that month.

⏱ 5 to 10 minutes.
"""))

cells.append(code("""if HAS_ECONML:
    cf = CausalForestDML(
        model_y=RandomForestRegressor(n_estimators=200, max_depth=4, random_state=42),
        model_t=RandomForestRegressor(n_estimators=200, max_depth=4, random_state=42),
        n_estimators=300, random_state=42, cv=5)
    cf.fit(Y_cf, T_cf, X=X_cf, W=W_mat)
    cate     = cf.effect(X_cf)
    ate      = float(np.mean(cate))
    ate_se   = float(np.std(cate) / np.sqrt(len(cate)))
    r_mod, p_mod = stats.pearsonr(X_cf[:,0], cate)
    print(f"ATE (h={h_cf}): {ate:+.4f}  SE={ate_se:.4f}")
    print(f"CATE std: {cate.std():.4f}  range: [{cate.min():.4f}, {cate.max():.4f}]")
    print(f"r(moderator, CATE): {r_mod:+.4f}  p={p_mod:.4f}")
    pct_neg = 100*(cate<0).mean()
    print(f"Fraction of observations with negative CATE: {pct_neg:.1f}%")
    direction = "negative (saturation: higher attention = weaker effect)" if r_mod<0 else "positive (amplification: higher attention = stronger effect)"
    print(f"Direction: {direction}")
else:
    print("econml not available. Install with: pip install econml")
    cate = np.array([]); ate = float('nan'); r_mod = float('nan'); p_mod = float('nan')
"""))

cells.append(md("""### 7.3 CATE Scatter Plot and Distribution

Two figures display the estimated CATEs.

Figure 7.1 (left) plots each observation's CATE against the moderator value, with a horizontal line at the ATE. A negative slope would confirm the saturation hypothesis: when the CHN-USA relationship commands high attention, the estimated treatment effect is closer to zero. A positive slope would suggest amplification.

Figure 7.1 (right) shows the histogram of all monthly CATEs. A distribution centred below zero indicates that cooperation predominantly reduces WTI, with the spread indicating how much the magnitude varies across time periods. A bimodal distribution would suggest two distinct regimes.
"""))

cells.append(code("""if len(cate) > 0:
    sort_idx = np.argsort(X_cf[:,0]); xs=X_cf[sort_idx,0]; cs_s=cate[sort_idx]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.scatter(xs, cs_s, alpha=0.35, s=18, color='#2C3E50', zorder=3)
    m_fit = np.polyfit(xs, cs_s, 1)
    ax.plot(np.sort(xs), np.polyval(m_fit, np.sort(xs)), color='#C0392B', lw=1.5, label=f'Trend  slope={m_fit[0]:+.4f}')
    ax.axhline(ate, color='#8E44AD', ls='--', lw=1.8, label=f'ATE = {ate:+.4f}')
    ax.axhline(0, color='black', lw=0.6)
    ax.set_xlabel(f'{mod_col} (z-score)'); ax.set_ylabel(f'CATE on log WTI (h={h_cf})')
    ax.set_title(f'Causal Forest CATE vs Moderator\nr={r_mod:+.3f}  p={p_mod:.3f}', fontsize=9, fontweight='bold')
    ax.legend(fontsize=8)

    ax2 = axes[1]
    ax2.hist(cate, bins=25, color='#2980B9', edgecolor='white', alpha=0.82)
    ax2.axvline(ate, color='#C0392B', lw=2.2, ls='--', label=f'ATE = {ate:+.4f}')
    ax2.axvline(0, color='black', lw=0.8)
    ax2.set_xlabel('CATE value'); ax2.set_ylabel('Count')
    ax2.set_title(f'Distribution of Monthly CATEs (h={h_cf})\n{pct_neg:.1f}% negative  std={cate.std():.4f}', fontsize=9, fontweight='bold')
    ax2.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT/'fig_causal_forest.png', dpi=150, bbox_inches='tight')
    plt.show()
    pd.DataFrame({'moderator_z':X_cf[:,0],'cate':cate}).to_csv(OUT/'causal_forest_cate.csv',index=False)
    print(f"Saved: fig_causal_forest.png, causal_forest_cate.csv")
else:
    print("Run Section 7.2 first.")
"""))

cells.append(md("""### 7.4 CATE Time Series

This cell plots the monthly sequence of CATEs over the sample period. Time-varying patterns in the CATEs can reveal whether the treatment effect is concentrated in specific sub-periods, such as heightened bilateral tension phases, post-crisis periods, or periods of unusual oil market conditions.

A CATE time series that is persistently near zero in some years but strongly negative in others would suggest that the average LP-IV estimate conflates structurally different regimes. This would be relevant to the structural break analysis in NB05 Section 10.5, which identified a break at 2002-02.
"""))

cells.append(code("""if len(cate) > 0 and len(dates_cf) == len(cate):
    cate_dates = pd.to_datetime(df.iloc[dates_cf.tolist()]['date'].values)
    cate_series = pd.Series(cate, index=cate_dates)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(cate_series.index, cate_series.values, width=28,
           color=np.where(cate_series.values<0,'#2980B9','#C0392B'), alpha=0.75)
    ax.plot(cate_series.index, cate_series.rolling(12,center=True,min_periods=3).mean(),
            color='black', lw=1.5, label='12-month rolling mean')
    ax.axhline(0, color='black', lw=0.7)
    ax.axhline(ate, color='#8E44AD', ls='--', lw=1.2, label=f'ATE = {ate:+.4f}')
    ax.axvline(pd.Timestamp('2002-02-01'), color='#27AE60', ls=':', lw=1.5, label='Structural break 2002-02')
    ax.set_xlabel('Month'); ax.set_ylabel(f'Estimated CATE (h={h_cf})')
    ax.set_title(f'Monthly CATE Time Series (h={h_cf})\nBlue = negative effect (cooperation lowers WTI); Red = positive', fontsize=9, fontweight='bold')
    ax.legend(fontsize=8)
    ax.xaxis.set_major_locator(mdates.YearLocator(4)); ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.tight_layout()
    plt.savefig(OUT/'fig_cf_cate_timeseries.png', dpi=150, bbox_inches='tight')
    plt.show(); print("Saved: fig_cf_cate_timeseries.png")
    pre2002 = cate_series[cate_series.index < pd.Timestamp('2002-02-01')]
    post2002= cate_series[cate_series.index >= pd.Timestamp('2002-02-01')]
    print(f"\nPre-2002 CATE:  mean={pre2002.mean():+.4f}  std={pre2002.std():.4f}  n={len(pre2002)}")
    print(f"Post-2002 CATE: mean={post2002.mean():+.4f}  std={post2002.std():.4f}  n={len(post2002)}")
    t_stat, p_break = stats.ttest_ind(pre2002.dropna(), post2002.dropna())
    print(f"Two-sample t-test (pre vs post 2002): t={t_stat:.2f}  p={p_break:.4f}")
else:
    print("Run Section 7.2 first.")
"""))

cells.append(md("""### 7.5 CATE by Sub-Period

The structural break identified by the sup-Wald test in NB05 (2002-02) divides the sample into two sub-periods with distinct geopolitical and macroeconomic characteristics. This cell fits the causal forest separately on each sub-period and compares the resulting ATEs and CATE distributions.

Pre-2002 (1990-01 to 2002-01, approximately 144 months): post-Cold War diplomatic normalisation, Taiwan Strait tensions, NATO bombing.

Post-2002 (2002-02 to 2022-02, approximately 241 months): WTO integration, financial crisis, trade war, COVID.

A significant difference in ATEs across the two sub-periods would confirm structural non-stationarity in the causal mechanism, which the rolling-window LP-IV in NB05 Section 10.6 shows as time-varying coefficients.
"""))

cells.append(code("""if HAS_ECONML and len(cate) > 0:
    BREAK_DATE = pd.Timestamp('2002-02-01')
    w_sp = df.copy()
    w_sp['_y'] = w_sp['lwti'].shift(-h_cf)
    for l in range(1,4): w_sp[f'_Ly{l}'] = w_sp['lwti'].shift(l)
    for l in range(1,3): w_sp[f'_Lt{l}'] = w_sp['lpri'].shift(l)
    all_cols = ['_y','lpri','d2pri','moderator_z'] + feat_cols
    sub_sp   = w_sp[[c for c in all_cols if c in w_sp.columns]].dropna()

    ate_by_period = {}
    for period_name, mask in [
        ('Pre-2002',  sub_sp.index < df.index[df['date']>=BREAK_DATE].min() if hasattr(df.index,'min') else np.ones(len(sub_sp),dtype=bool)),
        ('Post-2002', sub_sp.index >= df.index[df['date']>=BREAK_DATE].min() if hasattr(df.index,'min') else np.ones(len(sub_sp),dtype=bool)),
    ]:
        pass

    # More direct: split by date in the original dataframe
    date_arr = df['date'].values
    for period_name, date_mask_fn in [
        ('Pre-2002',  lambda d: d < pd.Timestamp('2002-02-01')),
        ('Post-2002', lambda d: d >= pd.Timestamp('2002-02-01')),
    ]:
        valid_dates_mask = np.array([date_mask_fn(pd.Timestamp(d)) for d in date_arr])
        df_period = df[valid_dates_mask].copy()
        w_ = df_period.copy()
        w_['_y'] = w_['lwti'].shift(-h_cf)
        for l in range(1,4): w_[f'_Ly{l}'] = w_['lwti'].shift(l)
        for l in range(1,3): w_[f'_Lt{l}'] = w_['lpri'].shift(l)
        sub_ = w_[[c for c in all_cols if c in w_.columns]].dropna()
        if len(sub_) < 50:
            print(f"{period_name}: insufficient obs ({len(sub_)})")
            ate_by_period[period_name] = float('nan')
            continue
        Y_ = sub_['_y'].values; T_ = sub_['lpri'].values
        X_ = sub_[['moderator_z']].values; W_ = sub_[feat_cols].values
        cf_ = CausalForestDML(
            model_y=RandomForestRegressor(n_estimators=100,max_depth=4,random_state=42),
            model_t=RandomForestRegressor(n_estimators=100,max_depth=4,random_state=42),
            n_estimators=200, random_state=42, cv=min(5,max(2,len(sub_)//50)))
        cf_.fit(Y_, T_, X=X_, W=W_)
        cate_ = cf_.effect(X_); ate_ = float(np.mean(cate_))
        ate_by_period[period_name] = ate_
        r_, p_ = stats.pearsonr(X_[:,0], cate_) if len(cate_) > 3 else (float('nan'), float('nan'))
        print(f"{period_name}: n={len(sub_)}  ATE={ate_:+.4f}  r(mod,CATE)={r_:+.3f}  p={p_:.4f}")

    # Compare ATEs
    print(f"\nFull sample ATE: {ate:+.4f}")
    for k, v in ate_by_period.items():
        print(f"{k} ATE: {v:+.4f}")
    if not any(np.isnan(v) for v in ate_by_period.values()):
        diff = list(ate_by_period.values())[1] - list(ate_by_period.values())[0]
        print(f"Post minus Pre ATE: {diff:+.4f}")
else:
    print("Run Section 7.2 first or install econml.")
"""))

cells.append(md("""### 7.6 Feature Importance from Causal Forest

The causal forest implicitly ranks features by how much they contribute to heterogeneity in the estimated treatment effect. This can be assessed by looking at the variable importance of the nuisance models: features that are important for predicting the outcome or the treatment (the nuisance models) are candidates for driving CATE heterogeneity.

This cell extracts feature importances from the outcome nuisance model (E[log WTI | X, W]) and the treatment nuisance model (E[log PRI | X, W]) fitted during the causal forest cross-fitting. High importance for a feature in both models indicates that it explains variation in both outcomes and treatment, which is necessary (but not sufficient) for that feature to drive treatment effect heterogeneity.
"""))

cells.append(code("""if HAS_ECONML and len(cate) > 0:
    try:
        # Access nuisance model feature importances if available
        # CausalForestDML stores models in cf.models_y and cf.models_t
        all_feat_names = feat_cols  # W features (controls/nuisance)

        fig, axes = plt.subplots(1, 2, figsize=(13, max(4, len(all_feat_names)*0.4+2)))
        feat_display = [f.replace('_Ly','WTI lag ').replace('_Lt','lpri lag ') for f in all_feat_names]
        feat_display = [f.replace('llwip','Log World IP').replace('dllgop','DeltaLogGOP')
                          .replace('l2lwip','Log World IP L2').replace('dl2lgop','DeltaSqLogGOP')
                          for f in feat_display]

        for ax, (model_attr, title) in zip(axes,[('models_y','Outcome Nuisance E[WTI|X,W]'),
                                                   ('models_t','Treatment Nuisance E[lpri|X,W]')]):
            try:
                models = getattr(cf, model_attr, None)
                if models is None or not hasattr(models[0][0], 'feature_importances_'):
                    ax.text(0.5,0.5,f'{model_attr}\nfeature importances\nnot accessible\nfor this econml version',
                            ha='center',va='center',transform=ax.transAxes,fontsize=9)
                    ax.set_title(title, fontweight='bold', fontsize=8)
                    continue
                importances = np.mean([m.feature_importances_ for fold_models in models for m in fold_models], axis=0)
                imp_series = pd.Series(importances[:len(feat_display)], index=feat_display).sort_values(ascending=True)
                colors_fi = ['#C0392B' if v>=imp_series.median() else '#3498DB' for v in imp_series]
                ax.barh(range(len(imp_series)), imp_series.values, color=colors_fi, alpha=0.82, edgecolor='white')
                ax.set_yticks(range(len(imp_series))); ax.set_yticklabels(imp_series.index, fontsize=8)
                ax.set_xlabel('Mean feature importance')
                ax.set_title(title, fontweight='bold', fontsize=8)
            except Exception as e:
                ax.text(0.5,0.5,f'Feature importances\nnot available:\n{str(e)[:50]}',
                        ha='center',va='center',transform=ax.transAxes,fontsize=8)
                ax.set_title(title, fontweight='bold', fontsize=8)

        fig.suptitle('Nuisance Model Feature Importances from Causal Forest\nRed = above median importance', fontsize=9, fontweight='bold', y=1.01)
        plt.tight_layout()
        plt.savefig(OUT/'fig_cf_feature_importance.png', dpi=150, bbox_inches='tight')
        plt.show(); print("Saved: fig_cf_feature_importance.png")
    except Exception as e:
        print(f"Feature importance extraction failed: {e}")
else:
    print("Run Section 7.2 first or install econml.")
"""))

cells.append(md("""### 7.7 Placebo Test: Permutation of the Moderator

A permutation placebo test evaluates whether the correlation between the moderator and the CATE could be explained by the statistical structure of the data rather than a genuine heterogeneous treatment effect.

Under the null hypothesis of no CATE heterogeneity, permuting the moderator values (breaking any true relationship with the CATEs) and re-estimating the causal forest should produce correlations between the permuted moderator and the resulting CATEs that are centred near zero. If the true correlation is outside the 95th percentile of the placebo distribution, the heterogeneity finding is unlikely to be a statistical artefact.

Ten permutations are used for computational tractability. Each refit takes approximately 30 to 60 seconds.

⏱ Approximately 10 minutes.
"""))

cells.append(code("""if HAS_ECONML and len(cate) > 0:
    print(f"Placebo test (10 permutations)")
    print(f"True r(moderator, CATE): {r_mod:+.4f}  p={p_mod:.4f}")
    print()
    N_PLACEBO = 10; rng_pb = np.random.RandomState(0); placebo_rs = []
    for p_idx in range(N_PLACEBO):
        X_shuf = X_cf.copy(); rng_pb.shuffle(X_shuf[:,0])
        cf_pb  = CausalForestDML(
            model_y=RandomForestRegressor(n_estimators=100,max_depth=4,random_state=p_idx),
            model_t=RandomForestRegressor(n_estimators=100,max_depth=4,random_state=p_idx),
            n_estimators=200, random_state=p_idx, cv=5)
        cf_pb.fit(Y_cf, T_cf, X=X_shuf, W=W_mat)
        cate_pb  = cf_pb.effect(X_shuf)
        r_pb, _  = stats.pearsonr(X_shuf[:,0], cate_pb)
        placebo_rs.append(r_pb)
        print(f"  Permutation {p_idx+1:2d}: r(shuffled moderator, CATE) = {r_pb:+.4f}")
    mean_pb = np.mean(placebo_rs); std_pb = np.std(placebo_rs)
    z_vs_pb = (r_mod - mean_pb) / std_pb if std_pb > 0 else float('nan')
    print(f"\nPlacebo mean={mean_pb:+.4f}  std={std_pb:.4f}")
    print(f"True r = {r_mod:+.4f}  z-score vs placebo = {z_vs_pb:+.2f}")
    if abs(z_vs_pb) > 2:
        print("The true correlation is distinguishable from the permutation distribution (|z| > 2).")
    else:
        print("The true correlation is not clearly distinguishable from permutation noise.")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(placebo_rs, bins=10, color='#7F8C8D', edgecolor='white', alpha=0.80, label='Placebo correlations')
    ax.axvline(r_mod, color='#C0392B', lw=2.5, label=f'True r = {r_mod:+.4f}')
    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlabel('Pearson r (moderator vs CATE)'); ax.set_ylabel('Count')
    ax.set_title(f'Causal Forest Placebo Test ({N_PLACEBO} permutations)\nRed line = true correlation; grey = permuted distribution', fontsize=9, fontweight='bold')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT/'fig_cf_placebo.png', dpi=150, bbox_inches='tight')
    plt.show()
    pd.DataFrame({'permutation':range(1,N_PLACEBO+1),'r_placebo':placebo_rs}).to_csv(OUT/'cf_placebo_test.csv',index=False)
    print("Saved: fig_cf_placebo.png, cf_placebo_test.csv")
else:
    print("Run Section 7.2 first or install econml.")
"""))

# SECTION 8: META-REGRESSION
cells.append(md("""---
## 8. Cross-Dyad Meta-Analysis

This section extracts the eight dyad-level h=6 LP-IV coefficients and attempts to explain the cross-dyad sign variation using a meta-regression against dyad economic characteristics.

The analysis is explicitly framed as exploratory. With n=8 dyads and three to four regressors, the regression is severely underpowered (Thompson and Sharp 1999 recommend at minimum n=15 to 20 for a meta-regression with this many predictors). The scatter plots are the primary output. The regression coefficients are printed only to quantify the direction of association visible in the plots; the p-values are not interpretable.
"""))

cells.append(md("""### 8.1 Dyad Characteristics and Coefficient Extraction

Approximate 2022 values are used for dyad economic characteristics. Three characteristics are examined: first-stage instrument strength (F-statistic), net oil-importer versus energy-exporter status, and bilateral trade volume. These correspond to three candidate channels for cross-dyad heterogeneity: identification precision, oil market channel direction, and economic integration.
"""))

cells.append(code("""meta_rows = []
for dyad, irf in irf_store.items():
    if 6 not in irf['h']: continue
    idx = irf['h'].index(6)
    meta_rows.append({'dyad':dyad,'coef_h6':irf['coef'][idx],'se_h6':irf['se'][idx]})
coef_meta = pd.DataFrame(meta_rows)

dyad_chars = pd.DataFrame([
    ('CHN-USA',558.0,0,1,1),('CHN-JPN',317.0,0,1,1),
    ('CHN-AUS',168.0,1,0,1),('CHN-FRA',65.0,0,1,1),
    ('CHN-DEU',206.0,0,1,1),('CHN-GBR',86.0,0,1,1),
    ('CHN-RUS',111.0,1,0,0),('CHN-IND',93.0,0,1,0),
], columns=['dyad','trade_bn_usd','energy_exporter','net_oil_importer','us_security_partner'])
dyad_chars = dyad_chars.merge(fs_df[['dyad','F']].rename(columns={'F':'first_stage_f'}),on='dyad',how='left')
df_meta    = coef_meta.merge(dyad_chars, on='dyad')
df_meta['log_trade'] = np.log(df_meta['trade_bn_usd'])

print(f"Meta-regression dataset: n={len(df_meta)} dyads")
print(f"\n{'Dyad':10s}  {'h=6 coef':>10}  {'se':>8}  {'F-stat':>8}  {'Energy exp.':>11}  {'Trade':>8}")
print('-'*65)
for _, r in df_meta.iterrows():
    print(f"{r['dyad']:10s}  {r['coef_h6']:>+10.4f}  {r['se_h6']:>8.4f}  {r['first_stage_f']:>8.1f}  {int(r['energy_exporter']):>11}  {r['trade_bn_usd']:>8.1f}")
"""))

cells.append(md("""### 8.2 Meta-Regression and Scatter Plots

Three OLS models are estimated. All are exploratory. The printed p-values are reported for completeness but cannot be interpreted as evidence at n=8.

The scatter plots (Figure N10) are the valid output: they show the raw bivariate relationships between each dyad characteristic and the h=6 LP-IV coefficient. Energy exporters (CHN-RUS, CHN-AUS) tend toward positive coefficients; oil importers (CHN-USA, CHN-JPN, CHN-GBR) tend toward negative coefficients. This pattern is consistent with a supply-channel for energy exporters and a demand/uncertainty-channel for oil importers, but it cannot be confirmed statistically with eight dyads.
"""))

cells.append(code("""print("Statistical caveat:")
print(f"  n = {len(df_meta)} dyads with up to 4 regressors.")
print("  This regression is severely underpowered. p-values are not interpretable.")
print("  The scatter plots below are the primary output.")
print()
for mname, regs in [
    ('M1: F-statistic only',    ['first_stage_f']),
    ('M2: channel structure',   ['energy_exporter','net_oil_importer']),
    ('M3: full specification',  ['first_stage_f','energy_exporter','log_trade']),
]:
    if not all(c in df_meta.columns for c in regs): continue
    X_ = add_constant(df_meta[regs]); fit = sm.OLS(df_meta['coef_h6'], X_).fit()
    print(f"\n{mname}  R2={fit.rsquared:.3f}  n={len(df_meta)}")
    for v in regs:
        b = fit.params[v]; t = fit.tvalues[v]; p = fit.pvalues[v]
        print(f"  {v:20s}: beta={b:+.4f}  t={t:+.2f}  p={p:.3f}  (exploratory only)")

# Scatter plots
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax, (xvar, xlabel) in zip(axes,[
    ('first_stage_f','First-stage F-statistic'),
    ('energy_exporter','Energy exporter (0=No 1=Yes)'),
    ('trade_bn_usd','Bilateral trade (bn USD 2022)'),
]):
    if xvar not in df_meta.columns: ax.set_visible(False); continue
    colors = ['#C0392B' if c<0 else '#2980B9' for c in df_meta['coef_h6']]
    ax.scatter(df_meta[xvar], df_meta['coef_h6'], c=colors, s=120, zorder=3, edgecolors='white', linewidth=1.0)
    for _, row in df_meta.iterrows():
        ax.annotate(row['dyad'].replace('CHN-',''), (row[xvar], row['coef_h6']),
                    textcoords='offset points', xytext=(5,4), fontsize=7.5)
    ax.axhline(0, color='black', lw=0.6, ls='--')
    ax.set_xlabel(xlabel, fontsize=9); ax.set_ylabel('LP-IV h=6 coefficient', fontsize=9)
    ax.set_title(xlabel.split(' (')[0], fontsize=9, fontweight='bold')
fig.suptitle('Cross-Dyad Coefficient Patterns (Exploratory)\nRed = negative (cooperation lowers WTI). Formal inference not feasible at n=8.', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT/'fig_meta_regression.png', dpi=150, bbox_inches='tight')
plt.show()
df_meta.to_csv(OUT/'meta_regression_data.csv', index=False)
print("Saved: fig_meta_regression.png, meta_regression_data.csv")
"""))

# APPENDIX
cells.append(md("""---
## Appendix: Instrument Diagnostics

This appendix reports two diagnostics that verify the quality of the constructed bilateral d2PRI instruments.

**Variation in log PRI series.** Dyads with very low PRI standard deviation (such as CHN-PAK) will produce near-zero d2PRI values even after pre-smoothing, because the underlying diplomatic series has very little movement to differentiate.

**Collinearity check.** The correlation between the constructed d2PRI and the lagged log PRI levels that appear as controls in the first stage. Values above 0.20 in absolute value indicate residual collinearity that may inflate the first-stage F-statistic.
"""))

cells.append(code("""print("Log PRI variation by dyad:")
print(f"{'Dyad':12s}  {'n':>4}  {'std':>8}  {'min':>8}  {'max':>8}")
print('-'*50)
for dyad,(lpri_col,_,_) in DYADS.items():
    if lpri_col not in df.columns: continue
    s = df[lpri_col].dropna()
    print(f"{dyad:12s}  {len(s):>4}  {s.std():>8.4f}  {s.min():>8.3f}  {s.max():>8.3f}")

print("\nCorrelation between d2PRI and lagged log PRI:")
for dyad,(lpri_col,d2_col,_) in DYADS.items():
    if d2_col not in df.columns or df[d2_col].isna().all(): continue
    w = df[[d2_col,lpri_col]].copy()
    w['L1'] = w[lpri_col].shift(1); w['L2'] = w[lpri_col].shift(2)
    w = w.dropna()
    if len(w) < 30: continue
    corr = w[[d2_col,lpri_col,'L1','L2']].corr()[d2_col].drop(d2_col)
    high = any(abs(corr) > 0.20)
    flag = '  NOTE: |r| > 0.20' if high else ''
    print(f"  {dyad}: level={corr[lpri_col]:+.3f}  L1={corr['L1']:+.3f}  L2={corr['L2']:+.3f}{flag}")
"""))

# SUMMARY
cells.append(md("""---
## Notebook Summary

This section collects the key numerical outputs from each section for quick reference and documents all outputs saved to disk.

**What was estimated and found:**

Section 2 identified 8 of 12 bilateral instruments as valid (F at or above 10). CHN-USA (F near 236) and CHN-JPN (F near 128) are the strongest. Four dyads (CHN-IDN, CHN-PAK, CHN-VNM, CHN-KOR) show weak instruments and receive Anderson-Rubin confidence sets instead of Wald intervals.

Section 3 estimated separate LP-IV regressions for all 8 valid dyads. The sign at h=6 is negative for 4 of 8 dyads (CHN-USA, CHN-JPN, CHN-AUS, CHN-GBR), all net oil importers. CHN-RUS (energy exporter) shows a positive coefficient, consistent with a supply-side channel.

Section 4 combined evidence across dyads: IVW pooled (USA, JPN, GBR) and pooled panel DML both confirm the negative short-run, sign-reversal pattern. The DML estimate at h=6 is approximately twice the magnitude of the IVW estimate, consistent with non-linear confounding that the linear LP-IV does not fully absorb.

Section 5 tested transmission channels: WTI (20 of 49 sig) and Brent (10 of 49 sig) both show the sign reversal. Industrial production (1 of 49) and CNY/USD (1 of 49) are near-zero, confirming the effect is oil-market specific rather than a broad macroeconomic shock.

Section 6 examined network structure. Diebold-Yilmaz total connectedness is approximately 30 to 40 percent, indicating moderate co-movement among bilateral d2PRI series. Granger causality finds a small number of directed predictive pairs. GAT attention weights converge to near-uniform values (1/12 per pair), indicating a dominant common factor in China's bilateral diplomatic posture.

Section 7 estimated CATEs using a causal forest with the ICEWS attention share as moderator. The ATE at h=6 is near the LP-IV estimate. The correlation between the moderator and CATEs, and whether the placebo test confirms heterogeneity, depends on results from running the cells.

Section 8 documents cross-dyad sign patterns. Energy exporters tend toward positive coefficients; oil importers toward negative. Formal meta-regression is not feasible at n=8.

**Output files saved to `outputs/05_panel/`:**

| File | Description |
|---|---|
| `panel_first_stage.csv` | F-statistics for all 12 bilateral instruments |
| `panel_lp_iv_irfs.csv` | Dyad by horizon LP-IV coefficients |
| `irf_ivw_pooled.csv` | IVW pooled estimator |
| `irf_panel_dml.csv` | Pooled panel DML results |
| `ar_ci_weak_dyads.csv` | Anderson-Rubin sets for excluded dyads |
| `granger_significant_pairs.csv` | Directed Granger causality pairs |
| `gat_centrality.csv` | Monthly GAT centrality for all 12 nodes |
| `gat_mean_attention.csv` | Average attention weight matrix |
| `causal_forest_cate.csv` | Monthly CATEs and moderator values |
| `meta_regression_data.csv` | Dyad characteristics and h=6 coefficients |
"""))

cells.append(code("""# Summary diagnostics
print("="*60)
print("NOTEBOOK 06: SUMMARY STATISTICS")
print("="*60)
print(f"\nDataset: {len(df)} months  ({df['date'].min().date()} to {df['date'].max().date()})")
print(f"Bilateral dyads in DYADS dict: {len(DYADS)}")
print(f"Valid instruments (F>=10): {len(valid_dyads)}  {valid_dyads}")
print()
print("LP-IV h=6 coefficients:")
for dyad, irf in irf_store.items():
    if 6 not in irf['h']: continue
    idx=irf['h'].index(6); c,se=irf['coef'][idx],irf['se'][idx]
    print(f"  {dyad:10s}: {c:+.4f}{'*' if se>0 and abs(c/se)>Z90 else ' '}  (F={fs_df[fs_df['dyad']==dyad]['F'].values[0]:.1f})")
print()
if ivw_h and 6 in ivw_h:
    idx=ivw_h.index(6); c=ivw_c[idx]; se=ivw_se[idx]
    print(f"IVW pooled h=6: {c:+.4f}{'*' if abs(c/se)>Z90 else ''}")
if irf_panel_dml['h'] and 6 in irf_panel_dml['h']:
    idx=irf_panel_dml['h'].index(6); c=irf_panel_dml['coef'][idx]; se=irf_panel_dml['se'][idx]
    print(f"Panel DML h=6:  {c:+.4f}{'*' if abs(c/se)>Z90 else ''}")
print()
print("Network:")
print(f"  DY total connectedness: {total:.1f}%")
print(f"  Significant Granger pairs: {len(gc_pairs)}")
if HAS_TORCH: print(f"  GAT: PyTorch trained (150 epochs)")
else: print(f"  GAT: NumPy fallback used")
print()
if HAS_ECONML and len(cate)>0:
    print(f"Causal Forest ATE (h=6): {ate:+.4f}")
    print(f"CATE r(moderator): {r_mod:+.4f}  p={p_mod:.4f}")
"""))

# WRITE NOTEBOOK
nb = {
    "nbformat":4, "nbformat_minor":5,
    "metadata":{
        "kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
        "language_info":{"name":"python","version":"3.13.0"}
    },
    "cells": cells
}
with open('06_panel_network_causal_forest_merge.ipynb','w',encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"Notebook written: {len(cells)} cells")