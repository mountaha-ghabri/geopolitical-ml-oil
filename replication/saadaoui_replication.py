"""
Saadaoui (2026 JCE) replication in Python.

Goal: mirror `original/Saadaoui_JCE_2026.do` as closely as possible.
This script creates the same figure names and prints sanity checks
against values reported in `original/Saadaoui_2026_JCE.log`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.iv import IV2SLS
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.tools.tools import add_constant


HMAX = 48


@dataclass
class Paths:
    root: Path
    original: Path
    figures: Path
    results: Path
    cache: Path
    dta: Path
    log: Path


def locate_paths() -> Paths:
    root = Path(__file__).resolve().parents[1]
    original = root / "original"
    figures = root / "figures"
    results = root / "results"
    cache = root / "data" / "cache"
    dta_data = root / "data" / "Saadaoui_2026_JCE.dta"
    dta_original = original / "Saadaoui_2026_JCE.dta"
    dta = dta_data if dta_data.exists() else dta_original
    log = original / "Saadaoui_2026_JCE.log"
    return Paths(
        root=root,
        original=original,
        figures=figures,
        results=results,
        cache=cache,
        dta=dta,
        log=log,
    )


def D(s: pd.Series) -> pd.Series:
    return s.diff()


def F(s: pd.Series, h: int) -> pd.Series:
    return s.shift(-h)


def stata_month_to_datetime(period: pd.Series) -> pd.Series:
    # Some Stata readers already return datetimes for %tm variables; if so, keep them.
    if pd.api.types.is_datetime64_any_dtype(period):
        return pd.to_datetime(period)

    # Otherwise interpret as Stata monthly index where 0 == 1960m1.
    base = pd.Period("1960-01", freq="M")
    numeric_period = pd.to_numeric(period, errors="coerce")
    return numeric_period.map(
        lambda m: (base + int(m)).to_timestamp(how="end") if pd.notna(m) else pd.NaT
    )


def load_data(dta_path: Path, cache_path: Path | None = None, refresh_cache: bool = False) -> pd.DataFrame:
    if cache_path is not None and cache_path.exists() and not refresh_cache:
        df = pd.read_parquet(cache_path)
        df = df.sort_values("Period").reset_index(drop=True)
        if "Period_dt" not in df.columns:
            df["Period_dt"] = stata_month_to_datetime(df["Period"])
        return df

    if not dta_path.exists():
        raise FileNotFoundError(
            f"Missing dataset: {dta_path}\n"
            "Place `Saadaoui_2026_JCE.dta` in the `original` folder."
        )
    df = pd.read_stata(dta_path)
    df = df.sort_values("Period").reset_index(drop=True)
    df["Period_dt"] = stata_month_to_datetime(df["Period"])
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
    return df


def ensure_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dllgop"] = D(out["llgop"])
    out["dl2lgop"] = D(out["l2lgop"])
    out["F2_d2pri"] = F(out["d2pri"], 2)
    for c in out.columns:
        if c.startswith("lpri_"):
            dc = f"d_{c}"
            if dc not in out.columns:
                out[dc] = D(out[c])
    return out


def add_lagged_controls(
    df: pd.DataFrame,
    y_col: str,
    shock_col: str,
    y_lags: int = 3,
    shock_lags: int = 2,
) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    lag_cols: list[str] = []
    for l in range(1, y_lags + 1):
        c = f"L{l}_{y_col}"
        out[c] = out[y_col].shift(l)
        lag_cols.append(c)
    for l in range(1, shock_lags + 1):
        c = f"L{l}_{shock_col}"
        out[c] = out[shock_col].shift(l)
        lag_cols.append(c)
    return out, lag_cols


def lp_ols(
    df: pd.DataFrame,
    x: str,
    controls: list[str],
    y_col: str = "lwti",
    y_lags: int = 3,
    shock_lags: int = 2,
    hmax: int = HMAX,
) -> pd.DataFrame:
    work, lag_cols = add_lagged_controls(df, y_col=y_col, shock_col=x, y_lags=y_lags, shock_lags=shock_lags)
    rhs_cols = [x] + lag_cols + controls
    rows = []
    for h in range(hmax + 1):
        hdf = pd.DataFrame(
            {"y_fwd": F(work[y_col], h), **{c: work[c] for c in rhs_cols}}
        ).replace([np.inf, -np.inf], np.nan).dropna()
        if len(hdf) < 30:
            rows.append((h, np.nan, np.nan))
            continue
        X = add_constant(hdf[rhs_cols], has_constant="add")
        y = hdf["y_fwd"]
        fit = sm.OLS(y, X).fit(cov_type="HC1")
        rows.append((h, fit.params.get(x, np.nan), fit.bse.get(x, np.nan)))
    irf = pd.DataFrame(rows, columns=["h", "coef", "se"])
    irf["lo95"] = irf["coef"] - 1.96 * irf["se"]
    irf["hi95"] = irf["coef"] + 1.96 * irf["se"]
    return irf


def lp_iv(
    df: pd.DataFrame,
    endog: str,
    instr: str,
    controls: list[str],
    y_col: str = "lwti",
    y_lags: int = 3,
    shock_lags: int = 2,
    hmax: int = HMAX,
) -> pd.DataFrame:
    work, lag_cols = add_lagged_controls(
        df, y_col=y_col, shock_col=endog, y_lags=y_lags, shock_lags=shock_lags
    )
    exog_cols = lag_cols + controls
    rows = []
    for h in range(hmax + 1):
        hdf = pd.DataFrame(
            {
                "y_fwd": F(work[y_col], h),
                endog: work[endog],
                instr: work[instr],
                **{c: work[c] for c in exog_cols},
            }
        ).replace([np.inf, -np.inf], np.nan).dropna()
        if len(hdf) < 30:
            rows.append((h, np.nan, np.nan))
            continue
        fit = IV2SLS(
            dependent=hdf["y_fwd"],
            exog=add_constant(hdf[exog_cols], has_constant="add"),
            endog=hdf[endog],
            instruments=hdf[instr],
        ).fit(cov_type="robust", debiased=True)
        rows.append((h, fit.params.get(endog, np.nan), fit.std_errors.get(endog, np.nan)))
    irf = pd.DataFrame(rows, columns=["h", "coef", "se"])
    irf["lo90"] = irf["coef"] - 1.645 * irf["se"]
    irf["hi90"] = irf["coef"] + 1.645 * irf["se"]
    irf["lo95"] = irf["coef"] - 1.96 * irf["se"]
    irf["hi95"] = irf["coef"] + 1.96 * irf["se"]
    return irf


def lp_quantile(
    df: pd.DataFrame,
    endog: str,
    instrument: str,
    controls: list[str],
    q: float,
    y_col: str = "lwti",
    y_lags: int = 3,
    shock_lags: int = 2,
    hmax: int = HMAX,
) -> pd.DataFrame:
    # Approximate Stata ivqregress with a control-function approach:
    # 1) First stage for endogenous regressor.
    # 2) Quantile regression includes first-stage residual.
    work, lag_cols = add_lagged_controls(
        df, y_col=y_col, shock_col=endog, y_lags=y_lags, shock_lags=shock_lags
    )
    exog_cols = lag_cols + controls
    rows = []
    for h in range(hmax + 1):
        hdf = pd.DataFrame(
            {
                "y_fwd": F(work[y_col], h),
                endog: work[endog],
                instrument: work[instrument],
                **{c: work[c] for c in exog_cols},
            }
        ).replace([np.inf, -np.inf], np.nan).dropna()
        if len(hdf) < 30:
            rows.append((h, np.nan))
            continue
        fs_X = add_constant(hdf[[instrument] + exog_cols], has_constant="add")
        fs_fit = sm.OLS(hdf[endog], fs_X).fit()
        hdf = hdf.assign(vhat=fs_fit.resid)
        qr_X = add_constant(hdf[[endog] + exog_cols + ["vhat"]], has_constant="add")
        fit = QuantReg(hdf["y_fwd"], qr_X).fit(q=q, max_iter=20000)
        rows.append((h, fit.params.get(endog, np.nan)))
    return pd.DataFrame(rows, columns=["h", "coef"])


def plot_irf_mean_quant(mean: pd.DataFrame, q25: pd.DataFrame, q50: pd.DataFrame, q75: pd.DataFrame, title: str, out_png: Path) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(mean["h"], mean["coef"], color="green", label="IV-LP")
    plt.fill_between(mean["h"], mean["lo90"], mean["hi90"], color="green", alpha=0.2, label="90% CI")
    plt.plot(q25["h"], q25["coef"], linestyle="--", label="Low - Q25")
    plt.plot(q50["h"], q50["coef"], linestyle=":", label="Median")
    plt.plot(q75["h"], q75["coef"], linestyle="-.", label="High - Q75")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xlabel("Months")
    plt.ylabel("Response")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=400, bbox_inches="tight")
    plt.close()


def set_stata_month_axis(ax: plt.Axes, period_dt: pd.Series) -> None:
    min_year = int(period_dt.dt.year.min())
    max_year = int(period_dt.dt.year.max())
    # Stata-like sparse ticks: 1990m1, 2000m1, 2010m1, ...
    tick_years = list(range((min_year // 10) * 10, max_year + 1, 10))
    ticks = [pd.Timestamp(year=y, month=1, day=1) for y in tick_years]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{y}m1" for y in tick_years])
    ax.set_xlim(period_dt.min(), period_dt.max())
    ax.set_xlabel("Time")


def set_horizon_axis(ax: plt.Axes) -> None:
    ax.set_xlim(0, HMAX)
    ax.set_xticks(np.arange(0, HMAX + 1, 6))
    ax.set_xlabel("Months")


def plot_pri_with_d2(
    df: pd.DataFrame,
    pri: str,
    d2: str,
    events: Iterable[tuple[float, int, str]],
    out_png: Path,
    y1_lim: tuple[float, float] | None = None,
    y2_lim: tuple[float, float] | None = None,
) -> None:
    fig, ax1 = plt.subplots(figsize=(11, 6))
    ax1.plot(df["Period_dt"], df[pri], label=pri, color="tab:blue")
    ax1.set_ylabel(pri)
    ax2 = ax1.twinx()
    ax2.bar(df["Period_dt"], df[d2], width=22, alpha=0.3, color="gray", label=d2)
    ax2.set_ylabel(d2)
    if y1_lim is not None:
        ax1.set_ylim(*y1_lim)
    if y2_lim is not None:
        ax2.set_ylim(*y2_lim)
    for yval, obs, txt in events:
        idx = int(max(0, min(len(df) - 1, obs - 1)))
        ax1.text(
            df["Period_dt"].iloc[idx],
            yval,
            txt,
            rotation=90,
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.75, "pad": 2},
            ha="center",
            va="top",
        )
    set_stata_month_axis(ax1, df["Period_dt"])
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=400, bbox_inches="tight")
    plt.close()


def scatter_fit(x: pd.Series, y: pd.Series, xlabel: str, ylabel: str, out_png: Path) -> None:
    v = pd.DataFrame({"x": x, "y": y}).dropna()
    X = add_constant(v["x"], has_constant="add")
    fit = sm.OLS(v["y"], X).fit()
    xp = np.linspace(v["x"].min(), v["x"].max(), 200)
    yp = fit.predict(add_constant(xp, has_constant="add"))
    plt.figure(figsize=(7.5, 5.5))
    plt.scatter(v["x"], v["y"], s=12, alpha=0.55)
    plt.plot(xp, yp, color="crimson", linewidth=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_png, dpi=400, bbox_inches="tight")
    plt.close()


def dynamic_legend_plot(df: pd.DataFrame, cols: list[str], labels: list[str], out_png: Path) -> None:
    last_vals = {}
    for c in cols:
        s = df[["Period", c]].dropna()
        if s.empty:
            last_vals[c] = -np.inf
        else:
            last_vals[c] = float(s.iloc[-1][c])
    ordered = sorted(cols, key=lambda c: last_vals[c], reverse=True)
    plt.figure(figsize=(11, 6))
    for c in ordered:
        plt.plot(df["Period_dt"], df[c], label=labels[cols.index(c)])
    plt.axhline(0, color="black", linewidth=0.8)
    plt.title("Relations with China (Log-modulus transform)")
    plt.legend(loc="upper left")
    set_stata_month_axis(plt.gca(), df["Period_dt"])
    plt.tight_layout()
    plt.savefig(out_png, dpi=400, bbox_inches="tight")
    plt.close()


def first_stage_f(df: pd.DataFrame, x: str, z: str, controls: list[str]) -> float:
    work, lag_cols = add_lagged_controls(df, y_col="lwti", shock_col=x, y_lags=3, shock_lags=2)
    exog_cols = lag_cols + controls
    fdf = pd.DataFrame({x: work[x], z: work[z], **{c: work[c] for c in exog_cols}}).dropna()
    X = add_constant(fdf[[z] + exog_cols], has_constant="add")
    fit = sm.OLS(fdf[x], X).fit(cov_type="HC1")
    return float(fit.f_test(f"{z} = 0").fvalue)


def print_log_sanity(irf_lead: pd.DataFrame, fs_f: float) -> None:
    # From Stata log (Figure 3 / Lead test), horizon 0 coefficient:
    log_h0 = 0.00870
    py_h0 = float(irf_lead.loc[irf_lead["h"] == 0, "coef"].iloc[0])
    print("\nSanity check vs Stata log:")
    print(f"- Lead test h=0 coef, Stata: {log_h0:.5f} | Python: {py_h0:.5f} | diff: {py_h0 - log_h0:+.5f}")
    print(f"- First-stage robust F (US baseline): {fs_f:.3f} (Stata log reports ~236.2 in IV-LP stage 0)")


def parse_stata_log(log_path: Path) -> dict[str, object]:
    txt = log_path.read_text(encoding="utf-8", errors="ignore")

    # Lead-test IRF table (Figure 3)
    lead_section_match = re.search(
        r"Impulse Response Function(.*?)(?=\. graph export Figure_3\.png)",
        txt,
        flags=re.DOTALL,
    )
    lead_section = lead_section_match.group(1) if lead_section_match else txt
    lead_rows = re.findall(
        r"^\s*(\d+)\s*\|\s*([\-0-9\.Ee]+)\s+([\-0-9\.Ee]+)\s+([\-0-9\.Ee]+)\s+([\-0-9\.Ee]+)\s*$",
        lead_section,
        flags=re.MULTILINE,
    )
    lead_irf = {}
    for h, coef, se, lo, hi in lead_rows:
        hh = int(h)
        if 0 <= hh <= HMAX:
            lead_irf[hh] = float(coef)

    # IV-LP mean table coefficients for lpri in each lwti_h(h)
    iv_blocks = re.finditer(r"lwti_h\((\d+)\)(.*?)(?=IV Test Step = \d+|\. graph export Figure_4|$)", txt, flags=re.DOTALL)
    iv_lpri = {}
    for m in iv_blocks:
        h = int(m.group(1))
        block = m.group(2)
        coef_match = re.search(r"lpri\s*\|\s*\n\s*--\.\s*\|\s*([\-0-9\.Ee]+)", block)
        if coef_match and 0 <= h <= HMAX:
            iv_lpri[h] = float(coef_match.group(1))

    # First-stage F (step 0)
    f_match = re.search(r"lpri\s*\|\s*[0-9\.]+\s+[0-9\.]+\s+[0-9\.]+\s+([0-9\.]+)\s+0\.0000", txt)
    fs_f = float(f_match.group(1)) if f_match else np.nan

    return {"lead_irf": lead_irf, "iv_lpri": iv_lpri, "first_stage_f": fs_f}


def compare_series(label: str, py: pd.Series, st: dict[int, float]) -> None:
    common_h = sorted(set(py.index.astype(int)).intersection(st.keys()))
    if not common_h:
        print(f"- {label}: no overlapping horizons found in log parse.")
        return
    py_vals = np.array([float(py.loc[h]) for h in common_h], dtype=float)
    st_vals = np.array([st[h] for h in common_h], dtype=float)
    diffs = py_vals - st_vals
    mae = float(np.mean(np.abs(diffs)))
    rmse = float(np.sqrt(np.mean(diffs**2)))
    max_abs = float(np.max(np.abs(diffs)))
    print(f"- {label}: matched {len(common_h)} horizons | MAE={mae:.6f} RMSE={rmse:.6f} MAX|diff|={max_abs:.6f}")


def main() -> None:
    p = locate_paths()
    p.figures.mkdir(parents=True, exist_ok=True)
    p.results.mkdir(parents=True, exist_ok=True)
    cache_file = p.cache / "Saadaoui_2026_JCE.parquet"
    df = ensure_derived_columns(load_data(p.dta, cache_path=cache_file))

    # Controls aligned to Stata baseline specification.
    base_controls = ["llwip", "dllgop", "l2lwip", "dl2lgop"]
    lead_controls = base_controls + [c for c in df.columns if c.startswith("d_lpri_")]

    # Figure 1
    plot_pri_with_d2(
        df=df,
        pri="pri",
        d2="d2pri",
        events=[
            (-7.0, 428, "Taiwan Strait Crisis"),
            (-7.0, 448, "Jiang Zemin's visit"),
            (-8.0, 466, "NATO bombing"),
            (-6.5, 692, "Trade war"),
            (-5.0, 737, "Winter Olympics"),
        ],
        out_png=p.figures / "Figure_1.png",
        y1_lim=(-10.5, 5.5),
        y2_lim=(-2.4, 2.4),
    )

    # Figure 3
    irf_lead = lp_ols(df, "F2_d2pri", lead_controls)
    plt.figure(figsize=(10, 6))
    plt.plot(irf_lead["h"], irf_lead["coef"], label="Lead test IRF")
    plt.fill_between(irf_lead["h"], irf_lead["lo95"], irf_lead["hi95"], alpha=0.2)
    plt.axhline(0, color="black", linewidth=0.8)
    set_horizon_axis(plt.gca())
    plt.ylabel("Response")
    plt.title("Lead Test: Reaction of Oil Prices to Geopolitical Turning Points")
    plt.legend()
    plt.tight_layout()
    plt.savefig(p.figures / "Figure_3.png", dpi=400, bbox_inches="tight")
    plt.close()

    # Figure 4 / Table A1 / A2 core IV-LP
    irf_mean_us = lp_iv(df, "lpri", "d2pri", base_controls)
    plt.figure(figsize=(10, 6))
    plt.plot(irf_mean_us["h"], irf_mean_us["coef"], color="green", label="IV-LP")
    plt.fill_between(irf_mean_us["h"], irf_mean_us["lo90"], irf_mean_us["hi90"], color="green", alpha=0.2, label="90% CI")
    plt.fill_between(irf_mean_us["h"], irf_mean_us["lo95"], irf_mean_us["hi95"], color="green", alpha=0.12, label="95% CI")
    plt.axhline(0, color="black", linewidth=0.8)
    set_horizon_axis(plt.gca())
    plt.ylabel("Response")
    plt.legend()
    plt.tight_layout()
    plt.savefig(p.figures / "Figure_4.png", dpi=400, bbox_inches="tight")
    plt.close()

    # Figure 5 (quantile approximation)
    q_controls = ["llwip", "dllgop", "l2lwip", "dl2lgop"]
    q25 = lp_quantile(df, "lpri", "d2pri", q_controls, q=0.25)
    q50 = lp_quantile(df, "lpri", "d2pri", q_controls, q=0.50)
    q75 = lp_quantile(df, "lpri", "d2pri", q_controls, q=0.75)
    plot_irf_mean_quant(
        irf_mean_us,
        q25,
        q50,
        q75,
        "IV-LP and Quantile LP (US-China)",
        p.figures / "Figure_5.png",
    )

    # Figures A1 / A2
    scatter_fit(D(df["lpri"]), df["d2pri"], "D.lpri", "d2.pri", p.figures / "Figure_A1.png")
    scatter_fit(D(df["lwti"]), df["d2pri"], "D.lwti", "d2.pri", p.figures / "Figure_A2.png")

    # Figure B1
    plot_pri_with_d2(
        df=df,
        pri="pri_jp",
        d2="d2pri_jp",
        events=[
            (-3.0, 601, "Rare-earth export ban"),
            (-3.5, 625, "Senkaku Islands"),
            (1.5, 693, "Li Keqiang visit"),
        ],
        out_png=p.figures / "Figure_B1.png",
        y1_lim=(-4.5, 2.5),
        y2_lim=(-2.4, 2.4),
    )

    # Figure B2 and B3
    irf_mean_jp = lp_iv(df, "lpri_jp", "d2pri_jp", base_controls)
    plt.figure(figsize=(10, 6))
    plt.plot(irf_mean_jp["h"], irf_mean_jp["coef"], color="green", label="IV-LP_jp")
    plt.fill_between(irf_mean_jp["h"], irf_mean_jp["lo90"], irf_mean_jp["hi90"], color="green", alpha=0.2)
    plt.axhline(0, color="black", linewidth=0.8)
    set_horizon_axis(plt.gca())
    plt.legend()
    plt.tight_layout()
    plt.savefig(p.figures / "Figure_B2.png", dpi=400, bbox_inches="tight")
    plt.close()

    q25_jp = lp_quantile(df, "lpri_jp", "d2pri_jp", q_controls, q=0.25)
    q50_jp = lp_quantile(df, "lpri_jp", "d2pri_jp", q_controls, q=0.50)
    q75_jp = lp_quantile(df, "lpri_jp", "d2pri_jp", q_controls, q=0.75)
    plot_irf_mean_quant(
        irf_mean_jp,
        q25_jp,
        q50_jp,
        q75_jp,
        "IV-LP and Quantile LP (Japan-China)",
        p.figures / "Figure_B3.png",
    )

    # Figure C1a / C1b
    c1a_cols = ["lpri_jp", "lpri_aus", "lpri_fra", "lpri_ger", "lpri_uk", "lpri"]
    c1a_labels = ["Japan", "Australia", "France", "Germany", "United Kingdom", "United States"]
    dynamic_legend_plot(df, c1a_cols, c1a_labels, p.figures / "Figure_C1a.png")

    c1b_cols = ["lpri_indo", "lpri_pak", "lpri_rus", "lpri_vn", "lpri_india", "lpri_cds"]
    c1b_labels = ["Indonesia", "Pakistan", "Russia", "Vietnam", "India", "South Korea"]
    dynamic_legend_plot(df, c1b_cols, c1b_labels, p.figures / "Figure_C1b.png")

    # Figure C2
    c2_controls = base_controls + [c for c in df.columns if c.startswith("d_lpri_")]
    irf_c2 = lp_iv(df, "lpri", "d2pri", c2_controls)
    q25_c2 = lp_quantile(df, "lpri", "d2pri", c2_controls, q=0.25)
    q50_c2 = lp_quantile(df, "lpri", "d2pri", c2_controls, q=0.50)
    q75_c2 = lp_quantile(df, "lpri", "d2pri", c2_controls, q=0.75)
    plot_irf_mean_quant(
        irf_c2,
        q25_c2,
        q50_c2,
        q75_c2,
        "IV-LP and Quantile LP controlling for alliances",
        p.figures / "Figure_C2.png",
    )

    irf_lead.to_csv(p.results / "parity_figure3_python.csv", index=False)
    irf_mean_us.to_csv(p.results / "parity_figure4_python.csv", index=False)

    # Sanity checks against log
    fs_f = first_stage_f(df, x="lpri", z="d2pri", controls=base_controls)
    print_log_sanity(irf_lead=irf_lead, fs_f=fs_f)
    parsed = parse_stata_log(p.log)
    print("\nExtended parity checks vs Stata log:")
    compare_series("Figure 3 lead IRF", irf_lead.set_index("h")["coef"], parsed["lead_irf"])
    compare_series("Figure 4 IV-LP (lpri coef)", irf_mean_us.set_index("h")["coef"], parsed["iv_lpri"])
    stata_f = float(parsed["first_stage_f"]) if not pd.isna(parsed["first_stage_f"]) else np.nan
    if np.isfinite(stata_f):
        print(
            f"- First-stage F (step 0): Stata={stata_f:.3f} Python={fs_f:.3f} diff={fs_f - stata_f:+.3f}"
        )
    print("\nDone. Figures saved in `figures/` and parity tables in `results/`.")


if __name__ == "__main__":
    main()

