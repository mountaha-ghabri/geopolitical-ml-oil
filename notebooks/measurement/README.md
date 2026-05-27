# Bilateral GPR — one notebook

[`construct_bilateral_gpr.ipynb`](construct_bilateral_gpr.ipynb)

## Real data only

- Downloads **yearly / monthly / daily** GDELT zips (1990–2022).
- **No** fake pseudo-events, **no** interpolating empty months, **no** `gdel_events_monthly_clean.csv` as the index.

## Hybrid measurement (same news, two channels)

| Era | GDELT file | Article URL in export? | GPR hit |
|-----|------------|------------------------|---------|
| 1990–2005 | `YYYY.zip` | Usually **no** (57 cols) | CAMEO protest/force/military |
| 2006–2013-03 | `YYYYMM.zip` | Usually **no** | CAMEO |
| 2013-04–2022 | daily zip | **Yes** (58 cols) | Keywords on `http` URL |

Using **Goldstein mean** as a filler is **not** the same as Bondarenko keywords — we use **CAMEO event types** when URLs are missing (same article pipeline, different field).

## `DOWNLOAD_SCOPE`

- `smoke` — quick test
- `saadaoui_lite` — full timeline; 2013+ uses 1st-of-month daily only
- `saadaoui` — full timeline; **every day** 2013-04–2022 (complete months, overnight)

After fixing the old daily-only bug, **re-download** yearly/monthly caches; optional: delete `raw/gdelt_daily/*` from the broken pilot.
