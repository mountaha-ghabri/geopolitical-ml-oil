"""
Bondarenko-style bilateral GPR from raw GDELT 1.0 files (free HTTP only).

GDELT file layout (see gdeltproject.org/data.html):
  1979-2005: yearly  YYYY.zip
  2006-2013-03: monthly YYYYMM.zip
  2013-04+: daily   YYYYMMDD.export.CSV.zip

Daily URLs return 404 before 2013-04-01 — that is why a daily-only pipeline leaves 1990-2013 empty.
"""
from __future__ import annotations

import io
import json
import re
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

SAMPLE_START = "1990-01-01"
SAMPLE_END = "2022-02-01"
DATE_RANGE = pd.date_range(SAMPLE_START, SAMPLE_END, freq="MS")

GDELT_BASE = "http://data.gdeltproject.org/events/"
GDELT_DAILY_URL = GDELT_BASE + "{date}.export.CSV.zip"

US_CODE, CHN_CODE = "USA", "CHN"
COL = {
    "sqldate": 1,
    "actor1_code": 5,
    "actor1_name": 6,
    "actor2_code": 15,
    "actor2_name": 16,
    "event_code": 25,
    "event_root": 26,
    "goldstein": 30,
    "sourceurl": 57,
}


def resolve_paths(notebook_dir: Path | None = None) -> dict[str, Path]:
    nb = (notebook_dir or Path.cwd()).resolve()
    root = nb.parent.parent if (nb.parent.parent / "data").exists() else nb.parent
    if not (root / "data").exists():
        root = nb.parent if (nb.parent / "data").exists() else nb
    meas = root / "data" / "measurement"
    raw = meas / "raw"
    return {
        "root": root,
        "final": root / "data" / "final",
        "nlp_legacy": root / "data" / "03_nlp",
        "meas": meas,
        "raw": raw,
        "gdelt_daily": raw / "gdelt_daily",
        "gdelt_yearly": raw / "gdelt_yearly",
        "gdelt_monthly": raw / "gdelt_monthly",
        "corpus": meas / "corpus",
        "constructed": meas / "constructed",
        "validation": meas / "validation",
        "figures": meas / "figures",
        "keywords": meas / "keywords",
    }


def align_sample(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df.index = df.index.to_period("M").to_timestamp()
    return df.reindex(DATE_RANGE)


GPR_KEYWORDS_EN = [
    "war", "wars", "warfare", "military", "armed conflict", "conflict",
    "terror", "terrorism", "terrorist", "attack", "attacks", "invasion",
    "invade", "battle", "troops", "sanction", "sanctions", "embargo",
    "hostage", "kidnap", "assassination", "coup", "rebel", "insurgent",
    "nuclear", "missile", "airstrike", "bombing", "combat", "blockade",
    "geopolitical", "tension", "crisis", "hostile", "retaliation",
    "trade war", "tariff", "espionage", "spy",
]
GPR_KEYWORDS_BILATERAL_EN = [
    "china", "chinese", "beijing", "taiwan", "sino-american", "us-china",
]
GPR_KEYWORDS_CN = [
    "战争", "军事", "冲突", "恐怖", "袭击", "入侵", "制裁", "危机",
    "紧张", "对峙", "贸易战", "间谍", "军演", "导弹", "封锁", "报复",
]
GPR_CAMEO_ROOTS = {14, 15, 17, 18, 19, 20}
GPR_CAMEO_EVENT_PREFIXES = ("17", "18", "19", "20", "14", "15", "13")


def compile_keyword_patterns(
    en: Iterable[str] | None = None,
    cn: Iterable[str] | None = None,
) -> tuple[re.Pattern, re.Pattern]:
    en = list(en or GPR_KEYWORDS_EN)
    cn = list(cn or GPR_KEYWORDS_CN)
    en_pat = re.compile(
        r"\b(" + "|".join(re.escape(w) for w in sorted(en, key=len, reverse=True)) + r")\b",
        re.I,
    )
    cn_pat = re.compile("|".join(re.escape(w) for w in cn))
    return en_pat, cn_pat


def text_has_gpr_keyword(text: str, en_pat: re.Pattern, cn_pat: re.Pattern) -> bool:
    if not isinstance(text, str) or not text.strip():
        return False
    return bool(en_pat.search(text) or cn_pat.search(text))


def _http_get(url: str, timeout: int = 180) -> bytes | None:
    try:
        r = requests.get(url, timeout=timeout)
        return r.content if r.status_code == 200 else None
    except Exception:
        return None


def _valid_source_url(series: pd.Series) -> pd.Series:
    s = series.fillna("").astype(str).str.strip()
    return s.str.startswith("http://", na=False) | s.str.startswith("https://", na=False)


def extract_uschn_events(df: pd.DataFrame) -> pd.DataFrame:
    """Filter raw GDELT TSV (no header) to US↔CHN rows — real events only."""
    a1, a2 = COL["actor1_code"], COL["actor2_code"]
    mask = ((df[a1] == US_CODE) & (df[a2] == CHN_CODE)) | ((df[a1] == CHN_CODE) & (df[a2] == US_CODE))
    sub = df.loc[mask]
    if sub.empty:
        return pd.DataFrame()

    ncols = sub.shape[1]
    out = pd.DataFrame()
    out["event_root"] = pd.to_numeric(sub[COL["event_root"]], errors="coerce")
    out["event_code"] = sub[COL["event_code"]].astype(str)
    out["actor1_name"] = sub[COL["actor1_name"]].fillna("")
    out["actor2_name"] = sub[COL["actor2_name"]].fillna("")
    if ncols > COL["sourceurl"]:
        out["sourceurl"] = sub[COL["sourceurl"]].fillna("")
    else:
        out["sourceurl"] = ""
    out["date"] = pd.to_datetime(sub[COL["sqldate"]], format="%Y%m%d", errors="coerce")
    return out.dropna(subset=["date"])


def label_events_kw(df: pd.DataFrame, en_pat: re.Pattern, cn_pat: re.Pattern) -> pd.DataFrame:
    """
    Hybrid GPR hit (same underlying news articles, two extraction channels):

    - **Has article URL** (58-column GDELT, ~2013-04+): Bondarenko-style keyword match on URL text.
    - **No URL** (57-column early GDELT): CAMEO event codes for protest/force/military — from the
      same news coding pipeline, not pre-downloaded Goldstein means.

    We do **not** use: Goldstein averages, monthly clean CSV aggregates, or invented placeholder months.
    """
    df = df.copy()
    df["has_url"] = _valid_source_url(df["sourceurl"])
    df["kw_url"] = np.where(
        df["has_url"],
        df["sourceurl"].map(lambda t: text_has_gpr_keyword(t, en_pat, cn_pat)),
        False,
    )
    df["kw_cameo"] = df["event_root"].isin(GPR_CAMEO_ROOTS) | df["event_code"].str.startswith(
        GPR_CAMEO_EVENT_PREFIXES
    )
    df["kw_hit"] = np.where(df["has_url"], df["kw_url"], df["kw_cameo"])
    df["channel"] = np.where(df["has_url"], "url_keyword", "cameo_event")
    return df


def monthly_gpr_from_events(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly shares from real labeled events only (no interpolation)."""
    df = df.copy()
    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp()
    rows = []
    for month, grp in df.groupby("month"):
        url = grp[grp["has_url"]]
        nourl = grp[~grp["has_url"]]
        rows.append(
            {
                "month": month,
                "n_events_total": len(grp),
                "n_events_kw": int(grp["kw_hit"].sum()),
                "n_events_with_url": int(grp["has_url"].sum()),
                "pct_events_with_url": float(grp["has_url"].mean()),
                "gpr_hybrid_share": float(grp["kw_hit"].mean()),
                "gpr_url_share": float(url["kw_url"].mean()) if len(url) else np.nan,
                "gpr_nourl_cameo_share": float(nourl["kw_cameo"].mean()) if len(nourl) else np.nan,
            }
        )
    out = pd.DataFrame(rows).set_index("month")
    out["gpr_kw_share"] = out["gpr_hybrid_share"]
    return align_sample(out)


def second_difference(s: pd.Series) -> pd.Series:
    return s - 2 * s.shift(1) + s.shift(2)


def _read_zip_csvs(blob: bytes) -> list[pd.DataFrame]:
    frames = []
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        for name in zf.namelist():
            if not name.lower().endswith(".csv"):
                continue
            with zf.open(name) as f:
                frames.append(pd.read_csv(f, sep="\t", header=None, dtype=str, low_memory=False))
    return frames


def process_yearly_zip(year: int, en_pat: re.Pattern, cn_pat: re.Pattern) -> pd.DataFrame:
    blob = _http_get(f"{GDELT_BASE}{year}.zip")
    if blob is None:
        return pd.DataFrame()
    parts = []
    for chunk in _read_zip_csvs(blob):
        raw = extract_uschn_events(chunk)
        if not raw.empty:
            parts.append(label_events_kw(raw, en_pat, cn_pat))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def process_monthly_zip(yyyymm: str, en_pat: re.Pattern, cn_pat: re.Pattern) -> pd.DataFrame:
    blob = _http_get(f"{GDELT_BASE}{yyyymm}.zip")
    if blob is None:
        return pd.DataFrame()
    parts = []
    for chunk in _read_zip_csvs(blob):
        raw = extract_uschn_events(chunk)
        if not raw.empty:
            parts.append(label_events_kw(raw, en_pat, cn_pat))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def fetch_gdelt_day(date_str: str, timeout: int = 90) -> pd.DataFrame | None:
    blob = _http_get(GDELT_DAILY_URL.format(date=date_str), timeout=timeout)
    if blob is None:
        return None
    try:
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            with zf.open(f"{date_str}.export.CSV") as f:
                df = pd.read_csv(f, sep="\t", header=None, dtype=str, low_memory=False)
    except Exception:
        return None
    raw = extract_uschn_events(df)
    return raw if not raw.empty else None


def gdelt_download_plan(scope: str = "saadaoui") -> list[tuple[str, str]]:
    """
    Download plan (all files are real GDELT archives on data.gdeltproject.org):

    - saadaoui: full Saadaoui window — yearly 1990-2005, monthly 2006-201303,
      **every day** 20130401-20220228 (complete months, has article URLs).
    - saadaoui_lite: same but 2013-04+ uses only the 1st of each month (faster, sparser).
    - smoke: tiny test set.
    """
    items: list[tuple[str, str]] = []
    if scope == "smoke":
        for y in [1990, 2005]:
            items.append(("year", str(y)))
        items.append(("month", "201001"))
        for ds in ["20130401", "20180301", "20220201"]:
            items.append(("day", ds))
        return items

    if scope not in ("saadaoui", "saadaoui_lite"):
        scope = "saadaoui"

    for y in range(1990, 2006):
        items.append(("year", str(y)))
    for y in range(2006, 2014):
        for m in range(1, 13):
            if y == 2013 and m > 3:
                break
            items.append(("month", f"{y}{m:02d}"))

    d = datetime(2013, 4, 1)
    end = datetime(2022, 2, 28)
    while d <= end:
        if scope == "saadaoui_lite" and d.day != 1:
            d += timedelta(days=1)
            continue
        items.append(("day", d.strftime("%Y%m%d")))
        d += timedelta(days=1)
    return items


def build_corpus(
    paths: dict[str, Path],
    *,
    scope: str = "saadaoui",
    en_pat: re.Pattern | None = None,
    cn_pat: re.Pattern | None = None,
    progress: bool = True,
) -> pd.DataFrame:
    """
    Build US-CHN labeled event corpus from raw GDELT archives (not pre-aggregated CSVs).
    """
    if en_pat is None or cn_pat is None:
        en_pat, cn_pat = compile_keyword_patterns()

    for p in (paths["gdelt_yearly"], paths["gdelt_monthly"], paths["gdelt_daily"]):
        p.mkdir(parents=True, exist_ok=True)

    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = lambda x, **k: x

    plan = gdelt_download_plan(scope)
    chunks: list[pd.DataFrame] = []

    for kind, kid in tqdm(plan, desc=f"GDELT ({scope})", disable=not progress):
        if kind == "year":
            cache = paths["gdelt_yearly"] / f"{kid}.parquet"
            if cache.exists():
                chunks.append(pd.read_parquet(cache))
                continue
            df = process_yearly_zip(int(kid), en_pat, cn_pat)
            if not df.empty:
                df.to_parquet(cache, index=False)
                chunks.append(df)
        elif kind == "month":
            cache = paths["gdelt_monthly"] / f"{kid}.parquet"
            if cache.exists():
                chunks.append(pd.read_parquet(cache))
                continue
            df = process_monthly_zip(kid, en_pat, cn_pat)
            if not df.empty:
                df.to_parquet(cache, index=False)
                chunks.append(df)
        else:
            cache = paths["gdelt_daily"] / f"{kid}.parquet"
            if cache.exists():
                chunks.append(pd.read_parquet(cache))
                continue
            raw = fetch_gdelt_day(kid)
            if raw is None or raw.empty:
                continue
            lab = label_events_kw(raw, en_pat, cn_pat)
            lab.to_parquet(cache, index=False)
            chunks.append(lab)

    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def coverage_table(corpus: pd.DataFrame) -> pd.DataFrame:
    if corpus.empty:
        return pd.DataFrame()
    c = corpus.copy()
    c["month"] = pd.to_datetime(c["date"]).dt.to_period("M").dt.to_timestamp()
    n = c.groupby("month").size().reindex(DATE_RANGE, fill_value=0)
    return pd.DataFrame({"n_events": n, "has_data": n > 0})


def download_caldara_gpr(cache: Path) -> pd.DataFrame:
    url = "https://www.matteoiacoviello.com/gpr_files/gpr_web_latest.xlsx"
    if cache.exists():
        return pd.read_csv(cache, index_col=0, parse_dates=True)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    xl = pd.ExcelFile(r.content)
    raw = xl.parse(xl.sheet_names[1])
    raw.columns = [str(c).strip() for c in raw.columns]
    date_col = raw.columns[0]
    raw["date"] = pd.to_datetime(raw[date_col], errors="coerce")
    raw = raw.dropna(subset=["date"]).set_index("date")
    gpr_col = next((c for c in raw.columns if str(c).upper() == "GPR"), raw.columns[1])
    out = pd.DataFrame({"gpr_global": raw[gpr_col]})
    out = align_sample(out)
    out.to_csv(cache)
    return out


def save_keywords(paths: dict[str, Path]) -> None:
    paths["keywords"].mkdir(parents=True, exist_ok=True)
    payload = {
        "gpr_keywords_en": GPR_KEYWORDS_EN,
        "gpr_keywords_cn": GPR_KEYWORDS_CN,
        "gpr_keywords_bilateral_en_doc_only": GPR_KEYWORDS_BILATERAL_EN,
        "note": "Do not use bilateral EN keywords when corpus is already US-CHN filtered.",
        "hybrid_rule": "If event has http(s) source URL: keyword on URL. Else: CAMEO GPR codes (same news pipeline, no Goldstein mean).",
        "gdelt_file_layout": {
            "1979-2005": "YYYY.zip",
            "2006-2013-03": "YYYYMM.zip",
            "2013-04+": "YYYYMMDD.export.CSV.zip",
        },
    }
    with open(paths["keywords"] / "gpr_keyword_config.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
