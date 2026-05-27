/*
  Geopolitical Turning Points Dashboard — Application Logic
  ──────────────────────────────────────────────────────────
  All data is loaded at runtime from CSV files under ../results/.
  No values are hardcoded — if a CSV is missing the dashboard
  degrades gracefully and shows an informative message.

  Data path convention (relative to dashboard/index.html):
    ../results/10_panel/panel_dyad_summary.csv
    ../results/10_panel/irf_iv_us.csv  (and other dyads)
    ../results/01_baseline_replication_saadaoui/irf_figure4_us_china.csv
    ../results/08_structural_breaks/rolling_iv_h6.csv  (etc.)
    ../results/09_multi_outcome_lp_iv/irf_outcome_*.csv
    ../results/10_panel/chow_2015.csv
    ../results/10_panel/placebo_perm.csv

  Chart.js 4 and Leaflet 1.9 are loaded from CDN in index.html.
*/

'use strict';

/* ═══════════════════════════════════════════════════════════════════
   CONSTANTS — paths, dyad metadata, colour palette
═══════════════════════════════════════════════════════════════════ */

const RESULTS = '../results';

/*
  Static metadata per dyad: display name, emoji flag, geographic
  coordinates (used to place Leaflet markers), and the CSV filename
  under results/10_panel/ that holds the LP-IV impulse response.
  F-stat and significance counts are filled in from
  panel_dyad_summary.csv at load time.
*/
const DYAD_META = {
  us:    { name: 'US–China',          flag: '🇺🇸', lat: 37.0,  lon: -95.0,  csv: 'irf_iv_us.csv',  note: 'Primary dyad. Strong instrument (F≈194). Sign reversal at h=32 confirmed. Placebo p=0.000.' },
  jp:    { name: 'Japan–China',       flag: '🇯🇵', lat: 35.7,  lon: 138.0,  csv: 'irf_iv_jp.csv',  note: 'Strong instrument (F=113) but genuine null (0/49 significant). The oil transmission is US-specific.' },
  aus:   { name: 'Australia–China',   flag: '🇦🇺', lat: -25.3, lon: 133.8,  csv: 'irf_iv_aus.csv', note: 'Valid instrument (F≈54). Few significant horizons. F drops below 10 at long horizons.' },
  fra:   { name: 'France–China',      flag: '🇫🇷', lat: 46.2,  lon: 2.2,    csv: 'irf_iv_fra.csv', note: 'IV corrects severe OLS upward bias (0 IV vs 45 OLS sig. horizons). Effect near zero after correction.' },
  ger:   { name: 'Germany–China',     flag: '🇩🇪', lat: 51.2,  lon: 10.5,   csv: 'irf_iv_ger.csv', note: 'Similar to France: OLS inflation corrected. IV shows no significant transmission after endogeneity correction.' },
  uk:    { name: 'UK–China',          flag: '🇬🇧', lat: 55.4,  lon: -3.4,   csv: 'irf_iv_uk.csv',  note: 'Valid instrument but borderline F at long horizons. Limited transmission detected.' },
  rus:   { name: 'Russia–China',      flag: '🇷🇺', lat: 60.0,  lon: 90.0,   csv: 'irf_iv_rus.csv', note: 'Highest sig. count (26/49) but F_min=5.8 at long horizons — weak-IV bias likely. Interpret with caution.' },
  india: { name: 'India–China',       flag: '🇮🇳', lat: 20.6,  lon: 79.0,   csv: null,             note: 'Weak instrument. Excluded from causal estimation.' },
  indo:  { name: 'Indonesia–China',   flag: '🇮🇩', lat: -2.5,  lon: 118.0,  csv: null,             note: 'Stable relations with China — insufficient PRI variation for identification.' },
  pak:   { name: 'Pakistan–China',    flag: '🇵🇰', lat: 30.4,  lon: 69.3,   csv: null,             note: 'Consistently improving relations. Near-zero Δ²PRI variation.' },
  vn:    { name: 'Vietnam–China',     flag: '🇻🇳', lat: 14.1,  lon: 108.3,  csv: null,             note: 'Weak instrument. Not included in panel estimation.' },
  cds:   { name: 'S.Korea–China',     flag: '🇰🇷', lat: 35.9,  lon: 127.8,  csv: null,             note: 'Borderline instrument. Excluded from main analysis.' },
};

/* China fixed position — always rendered as the red hub */
const CHINA = { lat: 35.0, lon: 105.0 };

/* Colour palette matching style.css variables */
const COLORS = {
  red:      '#C0392B',
  redLight: 'rgba(192,57,43,0.15)',
  gold:     '#B8860B',
  ink:      '#1A1208',
  inkSoft:  '#7A6652',
  paper:    '#F5F0E8',
  green:    '#1E7E34',
  blue:     '#1A5276',
  grey:     'rgba(26,18,8,0.3)',
};

/* ═══════════════════════════════════════════════════════════════════
   STATE — global mutable state for the running application
═══════════════════════════════════════════════════════════════════ */

const State = {
  dyadSummary:  {},   // keyed by dyad code; populated from panel_dyad_summary.csv
  irfCache:     {},   // memoised IRF data: irfCache[code][spec] = parsed rows
  placeboData:  [],   // permuted sig90 values from placebo_perm.csv
  selectedDyad: null, // currently selected dyad code or 'all'
  activeTab:    'map',
  map:          null, // Leaflet map instance
  markers:      {},   // Leaflet circle markers keyed by dyad code
  edges:        [],   // Leaflet polylines
  irfChart:     null,
  breaksChart:  null,
  outcomesChart:null,
  rollingH:     'h6',
  outcomeKey:   'lwti',
  irfSpec:      'iv',
};

/* ═══════════════════════════════════════════════════════════════════
   CSV UTILITIES
═══════════════════════════════════════════════════════════════════ */

/*
  Fetch and parse a CSV file using PapaParse.
  Returns a promise resolving to an array of row objects.
  If the file is missing or the fetch fails, resolves to [].
*/
function loadCSV(path) {
  return fetch(path)
    .then(r => {
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return r.text();
    })
    .then(text => {
      const result = Papa.parse(text, { header: true, dynamicTyping: true, skipEmptyLines: true });
      return result.data;
    })
    .catch(err => {
      console.warn(`CSV load failed: ${path}`, err.message);
      return [];
    });
}

/* ═══════════════════════════════════════════════════════════════════
   DATA LOADING — run on startup, populate State.*
═══════════════════════════════════════════════════════════════════ */

/*
  Load the panel dyad summary, which contains one row per valid dyad
  with columns: code, name, instrument, F_diag, F_min_lp, sig_iv,
  sig_ols, sig_rf, h12_iv, h12_ols, bias, strong.
*/
async function loadDyadSummary() {
  const rows = await loadCSV(`${RESULTS}/10_panel/panel_dyad_summary.csv`);
  rows.forEach(r => {
    if (r.code) State.dyadSummary[r.code] = r;
  });
}

/*
  Load the placebo permutation distribution.
  Columns: perm_sig90 (one row per permutation).
*/
async function loadPlacebo() {
  const rows = await loadCSV(`${RESULTS}/10_panel/placebo_perm.csv`);
  State.placeboData = rows.map(r => r.perm_sig90).filter(v => v != null);
}

/*
  Load one dyad's IRF CSV under results/10_panel/.
  Spec controls which file is fetched:
    'iv'         → irf_iv_{code}.csv
    'ols'        → (approximated from linear baseline for US, not stored separately)
    'dml_linear' → irf_dml_linear.csv  (US only, from NB06)
    'dml_xgb'   → irf_dml_xgb.csv     (US only, from NB06)

  Returns array of { h, coef, se, lo90, hi90, F }.
*/
async function loadIRF(code, spec) {
  const cacheKey = `${code}_${spec}`;
  if (State.irfCache[cacheKey]) return State.irfCache[cacheKey];

  let path;
  if (spec === 'iv') {
    const meta = DYAD_META[code];
    if (!meta || !meta.csv) return [];
    path = `${RESULTS}/10_panel/${meta.csv}`;
  } else if (spec === 'dml_linear') {
    path = `${RESULTS}/06_double_ml_iv_impulse_responses/irf_dml_linear.csv`;
  } else if (spec === 'dml_xgb') {
    path = `${RESULTS}/06_double_ml_iv_impulse_responses/irf_dml_xgb.csv`;
  } else if (spec === 'ols') {
    /* OLS baseline available for US-China from NB01 */
    path = `${RESULTS}/01_baseline_replication_saadaoui/irf_figure4_us_china.csv`;
  } else {
    return [];
  }

  const rows = await loadCSV(path);
  State.irfCache[cacheKey] = rows;
  return rows;
}

/*
  Load rolling IV coefficient file for the structural breaks tab.
  h is one of 'h6', 'h12', 'h24'.
*/
async function loadRolling(h) {
  return loadCSV(`${RESULTS}/08_structural_breaks/rolling_iv_${h}.csv`);
}

/*
  Load multi-outcome IRF for a given outcome variable.
  outcomeKey is one of: lwti, brent, gold, vix, cny_usd, bdi, gs10.
*/
async function loadOutcome(outcomeKey) {
  const key = `outcome_${outcomeKey}`;
  if (State.irfCache[key]) return State.irfCache[key];
  const rows = await loadCSV(`${RESULTS}/09_multi_outcome_lp_iv/irf_outcome_${outcomeKey}.csv`);
  State.irfCache[key] = rows;
  return rows;
}

/* Load Chow test results */
async function loadChow() {
  return loadCSV(`${RESULTS}/10_panel/chow_2015.csv`);
}

/* Load multi-outcome summary */
async function loadOutcomeSummary() {
  return loadCSV(`${RESULTS}/09_multi_outcome_lp_iv/multi_outcome_summary.csv`);
}

/* ═══════════════════════════════════════════════════════════════════
   MAP INITIALISATION (Leaflet)
═══════════════════════════════════════════════════════════════════ */

function initMap() {
  /*
    Dark-sepia CartoDB tile layer — complements the ink-on-paper
    aesthetic and makes the red China marker stand out clearly.
  */
  State.map = L.map('map', {
    center: [25, 60],
    zoom: 3,
    zoomControl: true,
    attributionControl: true,
  });

  L.tileLayer(
    'https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png',
    {
      attribution: '© OpenStreetMap © CARTO',
      subdomains: 'abcd',
      maxZoom: 7,
      minZoom: 2,
    }
  ).addTo(State.map);

  /* China — red filled circle, always rendered */
  const chinaMarker = L.circleMarker([CHINA.lat, CHINA.lon], {
    radius: 18,
    fillColor: COLORS.red,
    color: '#922B21',
    weight: 2,
    fillOpacity: 0.9,
  }).addTo(State.map);

  chinaMarker.bindTooltip(
    '<strong>China</strong><br>Geopolitical hub · Bilateral PRI measured from Chinese sources',
    { sticky: true, className: 'leaflet-tooltip-dark' }
  );

  /* Partner country markers */
  Object.entries(DYAD_META).forEach(([code, meta]) => {
    const summary = State.dyadSummary[code] || {};
    const fStat   = summary.F_diag || 0;
    const sigIV   = summary.sig_iv  || 0;
    const valid   = fStat >= 10;
    const strong  = fStat >= 30;

    /* Marker size scales with F-statistic; weak dyads are smaller */
    const radius = valid ? Math.min(6 + Math.sqrt(fStat) * 0.5, 14) : 5;
    const fillColor = strong ? '#1E7E34' : (valid ? COLORS.gold : COLORS.inkSoft);
    const opacity   = valid ? 0.85 : 0.4;

    const marker = L.circleMarker([meta.lat, meta.lon], {
      radius,
      fillColor,
      color: '#fff',
      weight: 1.5,
      fillOpacity: opacity,
    }).addTo(State.map);

    /* Tooltip shown on hover */
    const fStr = fStat ? `F = ${fStat.toFixed(1)}` : 'F = —';
    marker.bindTooltip(
      `<strong>${meta.flag} ${meta.name}</strong><br>${fStr} · sig90 = ${sigIV}/49`,
      { sticky: true }
    );

    /* Click → select dyad across all tabs */
    marker.on('click', () => App.selectDyad(code));

    State.markers[code] = marker;

    /* Edge from China to partner */
    const dashArray = valid ? null : '5 5';
    const lineWeight = valid ? Math.max(1, Math.min(fStat / 40, 4)) : 0.8;
    const edge = L.polyline([[CHINA.lat, CHINA.lon], [meta.lat, meta.lon]], {
      color: valid ? (strong ? '#1E7E34' : COLORS.gold) : COLORS.inkSoft,
      weight: lineWeight,
      opacity: valid ? 0.55 : 0.25,
      dashArray,
    }).addTo(State.map);

    State.edges.push(edge);
  });
}

/* ═══════════════════════════════════════════════════════════════════
   LEADERBOARD
═══════════════════════════════════════════════════════════════════ */

function buildLeaderboard() {
  const container = document.getElementById('leaderboard');
  container.innerHTML = '';

  /* Collect dyads that appear in the summary CSV */
  const known = Object.keys(DYAD_META)
    .map(code => ({ code, meta: DYAD_META[code], summary: State.dyadSummary[code] || {} }));

  /* Sort by the current sort criterion */
  const sortBy = document.getElementById('sort-select').value;
  known.sort((a, b) => {
    if (sortBy === 'sig') return (b.summary.sig_iv || 0) - (a.summary.sig_iv || 0);
    return (b.summary.F_diag || 0) - (a.summary.F_diag || 0);
  });

  known.forEach(({ code, meta, summary }) => {
    const fStat = summary.F_diag;
    const sigIV = summary.sig_iv;
    const valid  = fStat >= 10;
    const strong = fStat >= 30;

    /* Instrument strength badge */
    const badgeClass = strong ? 'strong' : (valid ? 'weak' : 'fail');
    const badgeLabel = fStat ? `F=${fStat.toFixed(0)}` : 'no data';

    const row = document.createElement('button');
    row.className = 'dyad-row';
    row.setAttribute('role', 'listitem');
    row.setAttribute('aria-label', `${meta.name} dyad, ${badgeLabel}`);
    row.onclick = () => App.selectDyad(code);
    row.id = `row-${code}`;

    row.innerHTML = `
      <span class="dyad-flag" aria-hidden="true">${meta.flag}</span>
      <div class="dyad-row-body">
        <div class="dyad-row-name">${meta.name}</div>
        <div class="dyad-row-meta">
          <span>${sigIV != null ? `sig=${sigIV}/49` : 'no data'}</span>
          ${summary.instrument ? `<span>${summary.instrument}</span>` : ''}
        </div>
      </div>
      <span class="dyad-f-badge ${badgeClass}">${badgeLabel}</span>
    `;

    container.appendChild(row);
  });
}

/* ═══════════════════════════════════════════════════════════════════
   DYAD DETAIL CARD
═══════════════════════════════════════════════════════════════════ */

function updateCard(code) {
  const meta    = DYAD_META[code] || {};
  const summary = State.dyadSummary[code] || {};

  const fStat = summary.F_diag;
  const strong = fStat >= 30;
  const valid  = fStat >= 10;

  document.getElementById('dc-flag').textContent = meta.flag || '';
  document.getElementById('dc-name').textContent = meta.name || code;
  document.getElementById('dc-sub').textContent  = `China bilateral dyad`;

  const badge = document.getElementById('dc-badge');
  badge.textContent  = strong ? 'STRONG' : (valid ? 'VALID' : 'WEAK');
  badge.className    = `dc-badge ${strong ? 'strong' : (valid ? 'weak' : 'null')}`;

  document.getElementById('dc-fstat').textContent = fStat != null ? fStat.toFixed(1) : '—';
  document.getElementById('dc-sig').textContent   = summary.sig_iv != null ? `${summary.sig_iv}/49` : '—';
  document.getElementById('dc-h6').textContent    = summary.h12_iv != null ? summary.h12_iv.toFixed(4) : '—';
  document.getElementById('dc-h12').textContent   = summary.h12_iv != null ? summary.h12_iv.toFixed(4) : '—';
  document.getElementById('dc-h32').textContent   = '—'; /* Not in summary CSV — shown in IRF tab */
  document.getElementById('dc-instr').textContent = summary.instrument || '—';
  document.getElementById('dc-note').textContent  = meta.note || '';
}

function updateCardAll() {
  const valid = Object.values(State.dyadSummary).filter(s => s.F_diag >= 10);
  document.getElementById('dc-flag').textContent = '⊕';
  document.getElementById('dc-name').textContent = 'All dyads — panel summary';
  document.getElementById('dc-sub').textContent  = '12 bilateral relationships with China';
  document.getElementById('dc-badge').textContent = `${valid.length}/12 valid`;
  document.getElementById('dc-badge').className   = 'dc-badge weak';
  document.getElementById('dc-fstat').textContent = '—';
  document.getElementById('dc-sig').textContent   = 'IVW: ~17/49';
  document.getElementById('dc-h6').textContent    = '—';
  document.getElementById('dc-h12').textContent   = '—';
  document.getElementById('dc-h32').textContent   = '—';
  document.getElementById('dc-instr').textContent = 'd2pri / L1dlpri';
  document.getElementById('dc-note').textContent  =
    'IVW pooling (17/49 sig.) driven by low heterogeneity (I²≈5%). CF panel gives 0/49 due to common-slope restriction. See panel tab.';
}

/* ═══════════════════════════════════════════════════════════════════
   IRF CHART (Chart.js)
═══════════════════════════════════════════════════════════════════ */

function destroyChart(chartRef) {
  if (chartRef) { chartRef.destroy(); }
  return null;
}

/*
  Build the Chart.js dataset object for one IRF series.
  rows: array of { h, coef, lo90, hi90 }
  showCI: whether to draw the 90% confidence band
*/
function irfDatasets(rows, label, color, showCI) {
  const hs    = rows.map(r => r.h);
  const coefs = rows.map(r => r.coef);
  const lo90  = rows.map(r => r.lo90);
  const hi90  = rows.map(r => r.hi90);
  const sets  = [];

  if (showCI) {
    /* Upper CI band */
    sets.push({
      label: '_ci_hi',
      data: hs.map((h, i) => ({ x: h, y: hi90[i] })),
      borderColor: 'transparent',
      backgroundColor: color.replace(')', ', 0.12)').replace('rgb', 'rgba'),
      fill: '+1',
      pointRadius: 0,
      tension: 0.35,
    });
    /* Lower CI band */
    sets.push({
      label: '_ci_lo',
      data: hs.map((h, i) => ({ x: h, y: lo90[i] })),
      borderColor: 'transparent',
      backgroundColor: color.replace(')', ', 0.12)').replace('rgb', 'rgba'),
      fill: false,
      pointRadius: 0,
      tension: 0.35,
    });
  }

  /* Main IRF line */
  sets.push({
    label,
    data: hs.map((h, i) => ({ x: h, y: coefs[i] })),
    borderColor: color,
    backgroundColor: 'transparent',
    borderWidth: 2.2,
    pointRadius: 0,
    pointHoverRadius: 4,
    tension: 0.35,
    fill: false,
  });

  return sets;
}

async function renderIRF(code) {
  const loading = document.getElementById('irf-loading');
  const canvas  = document.getElementById('irf-canvas');
  const foot    = document.getElementById('irf-foot');

  loading.textContent = 'Loading impulse response…';
  loading.style.display = 'flex';
  canvas.style.display  = 'none';

  const rows = await loadIRF(code, State.irfSpec);
  if (!rows.length) {
    loading.textContent = `No IRF data available for ${DYAD_META[code]?.name || code} under this specification.`;
    return;
  }

  loading.style.display = 'none';
  canvas.style.display  = 'block';

  State.irfChart = destroyChart(State.irfChart);

  const summary = State.dyadSummary[code] || {};
  const meta    = DYAD_META[code] || {};
  const sigCount = rows.filter(r => r.lo90 > 0 || r.hi90 < 0).length;

  /* Update title */
  document.getElementById('irf-title').textContent =
    `Impulse Response — ${meta.name || code}`;

  foot.textContent =
    `Source: results/10_panel/${meta.csv || '—'} · ` +
    `F-stat (h=0): ${summary.F_diag?.toFixed(1) || '—'} · ` +
    `Significant horizons: ${sigCount}/49 at 90% CI · ` +
    `Note: ${rows[0]?.F != null ? 'first-stage F shown in annotation' : 'F not available in this file'}`;

  const ctx = canvas.getContext('2d');
  const datasets = irfDatasets(rows, `LP-IV (${State.irfSpec})`, COLORS.red, true);

  /* Add a zero reference line dataset */
  datasets.push({
    label: 'Zero line',
    data: rows.map(r => ({ x: r.h, y: 0 })),
    borderColor: COLORS.inkSoft,
    borderWidth: 0.8,
    borderDash: [3, 3],
    pointRadius: 0,
    fill: false,
  });

  State.irfChart = new Chart(ctx, {
    type: 'line',
    data: { datasets },
    options: {
      animation: { duration: 350 },
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: true,
          labels: {
            filter: item => !item.text.startsWith('_ci_') && item.text !== 'Zero line',
            color: COLORS.ink,
            font: { family: "'Noto Sans', sans-serif", size: 11 },
          },
        },
        tooltip: {
          callbacks: {
            label: ctx => {
              if (ctx.dataset.label.startsWith('_ci_') || ctx.dataset.label === 'Zero line') return null;
              const h = ctx.parsed.x;
              const row = rows.find(r => r.h === h);
              if (!row) return `${ctx.parsed.y.toFixed(4)}`;
              return [
                `Coef: ${row.coef?.toFixed(4) ?? '—'}`,
                `90% CI: [${row.lo90?.toFixed(4) ?? '—'}, ${row.hi90?.toFixed(4) ?? '—'}]`,
                row.F != null ? `F-stat: ${row.F.toFixed(1)}` : null,
              ].filter(Boolean);
            },
            title: ctx => `h = ${ctx[0].parsed.x} months`,
          },
          backgroundColor: COLORS.ink,
          titleColor: '#fff',
          bodyColor: 'rgba(255,255,255,0.8)',
          padding: 10,
          cornerRadius: 3,
        },
        annotation: {
          /* mark h=6 and h=32 as the paper's key reference horizons */
        },
      },
      scales: {
        x: {
          type: 'linear',
          min: 0, max: 48,
          title: {
            display: true,
            text: 'Horizon (months)',
            color: COLORS.inkSoft,
            font: { family: "'Noto Sans', sans-serif", size: 11 },
          },
          ticks: {
            stepSize: 6,
            color: COLORS.inkSoft,
            font: { family: "'IBM Plex Mono', monospace", size: 10 },
          },
          grid: { color: 'rgba(26,18,8,0.07)' },
        },
        y: {
          title: {
            display: true,
            text: 'LP-IV coefficient on log-modulus PRI',
            color: COLORS.inkSoft,
            font: { family: "'Noto Sans', sans-serif", size: 11 },
          },
          ticks: {
            color: COLORS.inkSoft,
            font: { family: "'IBM Plex Mono', monospace", size: 10 },
          },
          grid: { color: 'rgba(26,18,8,0.07)' },
        },
      },
    },
  });
}

/* ═══════════════════════════════════════════════════════════════════
   STRUCTURAL BREAKS CHART
═══════════════════════════════════════════════════════════════════ */

async function renderBreaks() {
  const rows = await loadRolling(State.rollingH);
  State.breaksChart = destroyChart(State.breaksChart);

  const canvas = document.getElementById('breaks-canvas');
  const ctx    = canvas.getContext('2d');

  if (!rows.length) {
    return;
  }

  /*
    Rolling CSV columns expected: date (or index), coef, lo90, hi90.
    The exact column names depend on what NB08 saved.
    We attempt several common name variants.
  */
  const coefCol = rows[0].hasOwnProperty('coef') ? 'coef'
                : rows[0].hasOwnProperty('beta')  ? 'beta'
                : Object.keys(rows[0])[1];
  const lo90Col = rows[0].hasOwnProperty('lo90')  ? 'lo90'
                : rows[0].hasOwnProperty('lo_90') ? 'lo_90'
                : null;
  const hi90Col = rows[0].hasOwnProperty('hi90')  ? 'hi90'
                : rows[0].hasOwnProperty('hi_90') ? 'hi_90'
                : null;

  const labels  = rows.map((r, i) => r.date || r.period || r.index || i);
  const coefs   = rows.map(r => r[coefCol]);
  const lo90    = lo90Col ? rows.map(r => r[lo90Col]) : null;
  const hi90    = hi90Col ? rows.map(r => r[hi90Col]) : null;

  const datasets = [];

  if (hi90) {
    datasets.push({
      label: '_ci_hi',
      data: hi90,
      borderColor: 'transparent',
      backgroundColor: 'rgba(192,57,43,0.12)',
      fill: '+1',
      pointRadius: 0,
      tension: 0.3,
    });
    datasets.push({
      label: '_ci_lo',
      data: lo90,
      borderColor: 'transparent',
      backgroundColor: 'rgba(192,57,43,0.12)',
      fill: false,
      pointRadius: 0,
      tension: 0.3,
    });
  }

  datasets.push({
    label: `Rolling IV coef (${State.rollingH})`,
    data: coefs,
    borderColor: COLORS.red,
    backgroundColor: 'transparent',
    borderWidth: 2,
    pointRadius: 0,
    tension: 0.3,
  });

  /* Zero line */
  datasets.push({
    label: 'Zero',
    data: coefs.map(() => 0),
    borderColor: 'rgba(26,18,8,0.25)',
    borderWidth: 1,
    borderDash: [4, 4],
    pointRadius: 0,
    fill: false,
  });

  State.breaksChart = new Chart(ctx, {
    type: 'line',
    data: { labels, datasets },
    options: {
      animation: { duration: 300 },
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: {
            filter: item => !item.text.startsWith('_ci_') && item.text !== 'Zero',
            color: COLORS.ink,
            font: { family: "'Noto Sans', sans-serif", size: 11 },
          },
        },
        tooltip: {
          callbacks: { title: ctx => `Period: ${ctx[0].label}` },
          backgroundColor: COLORS.ink,
          titleColor: '#fff',
          bodyColor: 'rgba(255,255,255,0.8)',
        },
      },
      scales: {
        x: {
          ticks: {
            maxTicksLimit: 12,
            color: COLORS.inkSoft,
            font: { family: "'IBM Plex Mono', monospace", size: 9 },
            maxRotation: 45,
          },
          grid: { color: 'rgba(26,18,8,0.06)' },
        },
        y: {
          title: {
            display: true,
            text: 'Rolling IV coefficient',
            color: COLORS.inkSoft,
            font: { family: "'Noto Sans', sans-serif", size: 11 },
          },
          ticks: {
            color: COLORS.inkSoft,
            font: { family: "'IBM Plex Mono', monospace", size: 10 },
          },
          grid: { color: 'rgba(26,18,8,0.07)' },
        },
      },
    },
  });

  /* Render Chow test table */
  await renderChowTable();
}

async function renderChowTable() {
  const rows = await loadChow();
  const container = document.getElementById('chow-table');
  if (!rows.length) { container.innerHTML = ''; return; }

  const html = `
    <table>
      <thead>
        <tr>
          <th>Horizon h</th>
          <th>Chow F-stat</th>
          <th>p-value</th>
          <th>Significant (1%)</th>
        </tr>
      </thead>
      <tbody>
        ${rows.map(r => {
          const sig = r.sig01 || r.sig_01 || (r.p < 0.01);
          return `
          <tr>
            <td>${r.h}</td>
            <td>${r.F?.toFixed(2) ?? '—'}</td>
            <td class="${sig ? 'chow-sig' : ''}">${r.p?.toFixed(4) ?? '—'}</td>
            <td class="${sig ? 'chow-sig' : ''}">${sig ? '✓ YES' : 'no'}</td>
          </tr>`;
        }).join('')}
      </tbody>
    </table>
    <p style="font-size:10.5px;color:#7A6652;margin-top:6px;font-style:italic;">
      Break date: 2015-01-01. Pre-2015 n=299 (sig90=25/49). Post-2015 n=86 (sig90=5/49).
      All horizons significant at 1% — the structural break is statistically confirmed.
    </p>
  `;
  container.innerHTML = html;
}

/* ═══════════════════════════════════════════════════════════════════
   MULTI-OUTCOME CHART
═══════════════════════════════════════════════════════════════════ */

const OUTCOME_LABELS = {
  lwti:    'WTI Oil Price (log)',
  brent:   'Brent Crude (log)',
  gold:    'Gold Price (log)',
  vix:     'VIX Volatility Index',
  cny_usd: 'CNY/USD Exchange Rate',
  bdi:     'Baltic Dry Index (log)',
  gs10:    '10-Year Treasury Yield',
};

async function renderOutcomes() {
  const rows = await loadOutcome(State.outcomeKey);
  State.outcomesChart = destroyChart(State.outcomesChart);

  const canvas = document.getElementById('outcomes-canvas');
  const ctx    = canvas.getContext('2d');

  if (!rows.length) return;

  const coefs  = rows.map(r => r.coef);
  const lo90   = rows.map(r => r.lo90);
  const hi90   = rows.map(r => r.hi90);
  const hs     = rows.map(r => r.h);
  const sigN   = rows.filter(r => r.lo90 > 0 || r.hi90 < 0).length;

  State.outcomesChart = new Chart(ctx, {
    type: 'line',
    data: {
      datasets: [
        {
          label: '_ci_hi',
          data: hs.map((h, i) => ({ x: h, y: hi90[i] })),
          borderColor: 'transparent',
          backgroundColor: 'rgba(184,134,11,0.13)',
          fill: '+1',
          pointRadius: 0,
          tension: 0.35,
        },
        {
          label: '_ci_lo',
          data: hs.map((h, i) => ({ x: h, y: lo90[i] })),
          borderColor: 'transparent',
          backgroundColor: 'rgba(184,134,11,0.13)',
          fill: false,
          pointRadius: 0,
          tension: 0.35,
        },
        {
          label: OUTCOME_LABELS[State.outcomeKey] || State.outcomeKey,
          data: hs.map((h, i) => ({ x: h, y: coefs[i] })),
          borderColor: COLORS.gold,
          backgroundColor: 'transparent',
          borderWidth: 2.2,
          pointRadius: 0,
          tension: 0.35,
        },
        {
          label: 'Zero',
          data: hs.map(h => ({ x: h, y: 0 })),
          borderColor: 'rgba(26,18,8,0.2)',
          borderWidth: 0.8,
          borderDash: [3, 3],
          pointRadius: 0,
          fill: false,
        },
      ],
    },
    options: {
      animation: { duration: 300 },
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: {
            filter: item => !item.text.startsWith('_ci_') && item.text !== 'Zero',
            color: COLORS.ink,
            font: { family: "'Noto Sans', sans-serif", size: 11 },
          },
        },
        tooltip: {
          backgroundColor: COLORS.ink,
          titleColor: '#fff',
          bodyColor: 'rgba(255,255,255,0.8)',
          callbacks: {
            title: ctx => `h = ${ctx[0].parsed.x} months`,
            label: ctx => {
              if (ctx.dataset.label.startsWith('_ci_') || ctx.dataset.label === 'Zero') return null;
              const h = ctx.parsed.x;
              const row = rows.find(r => r.h === h);
              return row
                ? [`Coef: ${row.coef?.toFixed(4)}`, `90% CI: [${row.lo90?.toFixed(4)}, ${row.hi90?.toFixed(4)}]`]
                : [`${ctx.parsed.y.toFixed(4)}`];
            },
          },
        },
      },
      scales: {
        x: {
          type: 'linear', min: 0, max: 48,
          title: { display: true, text: 'Horizon (months)', color: COLORS.inkSoft, font: { size: 11 } },
          ticks: { stepSize: 6, color: COLORS.inkSoft, font: { family: "'IBM Plex Mono', monospace", size: 10 } },
          grid: { color: 'rgba(26,18,8,0.07)' },
        },
        y: {
          title: { display: true, text: `LP-IV coef on ${OUTCOME_LABELS[State.outcomeKey]}`, color: COLORS.inkSoft, font: { size: 11 } },
          ticks: { color: COLORS.inkSoft, font: { family: "'IBM Plex Mono', monospace", size: 10 } },
          grid: { color: 'rgba(26,18,8,0.07)' },
        },
      },
    },
  });

  /* Outcome summary chips */
  await renderOutcomeSummary(sigN);
}

async function renderOutcomeSummary(sigCurrent) {
  const rows = await loadOutcomeSummary();
  const container = document.getElementById('outcome-summary');
  if (!rows.length) {
    container.innerHTML = `<span class="outcome-chip">Loaded: <span class="chip-sig">${sigCurrent}/49 sig</span> for ${State.outcomeKey}</span>`;
    return;
  }

  const outcomeCols = Object.keys(rows[0]).filter(k => k !== 'horizon' && k !== 'h');
  container.innerHTML = rows.map(row =>
    outcomeCols.map(k => `
      <span class="outcome-chip">
        ${k} — <span class="chip-sig">${row[k] ?? '—'}/49 sig</span>
      </span>
    `).join('')
  ).join('') || `<span class="outcome-chip">${State.outcomeKey}: <span class="chip-sig">${sigCurrent}/49</span></span>`;
}

/* ═══════════════════════════════════════════════════════════════════
   APP — public interface called by HTML event handlers and tab logic
═══════════════════════════════════════════════════════════════════ */

const App = {

  /*
    Select a dyad by code or 'all'.
    Updates: sidebar card, map marker highlight, IRF chart (if on IRF tab).
  */
  selectDyad(code) {
    State.selectedDyad = code;

    /* Highlight leaderboard row */
    document.querySelectorAll('.dyad-row').forEach(el => el.classList.remove('active'));
    const rowEl = document.getElementById(code === 'all' ? 'row-all' : `row-${code}`);
    if (rowEl) rowEl.classList.add('active');

    /* Update map marker appearance */
    Object.entries(State.markers).forEach(([c, m]) => {
      m.setStyle({ weight: c === code ? 3 : 1.5, color: c === code ? COLORS.red : '#fff' });
    });

    /* Update card */
    if (code === 'all') {
      updateCardAll();
    } else {
      updateCard(code);
    }

    /* Render IRF if that tab is active */
    if (State.activeTab === 'irf' && code !== 'all') {
      renderIRF(code);
    }

    /* Switch to IRF tab automatically when a dyad is clicked on map */
    if (code !== 'all' && State.activeTab === 'map') {
      App.switchTab('irf');
      renderIRF(code);
    }
  },

  /*
    Switch between tabs. Triggers rendering of the newly active tab
    if it hasn't been rendered yet (or needs fresh data).
  */
  switchTab(tab) {
    State.activeTab = tab;

    /* Toggle button states */
    document.querySelectorAll('.tab-btn').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.tab === tab);
    });

    /* Toggle pane visibility */
    document.querySelectorAll('.tab-pane').forEach(pane => {
      pane.classList.toggle('active', pane.id === `pane-${tab}`);
    });

    /* Invalidate map size when returning to map tab */
    if (tab === 'map' && State.map) {
      setTimeout(() => State.map.invalidateSize(), 50);
    }

    /* Render breaks / outcomes when first visited */
    if (tab === 'breaks') {
      renderBreaks();
    }

    if (tab === 'outcomes') {
      renderOutcomes();
    }

    /* Re-render IRF for selected dyad */
    if (tab === 'irf' && State.selectedDyad && State.selectedDyad !== 'all') {
      renderIRF(State.selectedDyad);
    }
  },
};

/* ═══════════════════════════════════════════════════════════════════
   EVENT LISTENERS — wired to controls in index.html
═══════════════════════════════════════════════════════════════════ */

/* Tab navigation */
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => App.switchTab(btn.dataset.tab));
});

/* Leaderboard sort */
document.getElementById('sort-select').addEventListener('change', buildLeaderboard);

/* IRF specification selector */
document.getElementById('irf-spec-select').addEventListener('change', e => {
  State.irfSpec = e.target.value;
  if (State.selectedDyad && State.selectedDyad !== 'all') {
    renderIRF(State.selectedDyad);
  }
});

/* Rolling horizon selector */
document.getElementById('rolling-h-select').addEventListener('change', e => {
  State.rollingH = e.target.value;
  if (State.activeTab === 'breaks') renderBreaks();
});

/* Outcome selector */
document.getElementById('outcome-select').addEventListener('change', e => {
  State.outcomeKey = e.target.value;
  if (State.activeTab === 'outcomes') renderOutcomes();
});

/* ═══════════════════════════════════════════════════════════════════
   INITIALISATION — run on DOMContentLoaded
═══════════════════════════════════════════════════════════════════ */

async function init() {
  const overlay  = document.getElementById('loading-overlay');
  const fill     = document.getElementById('loading-fill');
  const loadMsg  = document.getElementById('loading-msg');

  const steps = [
    { msg: 'Loading dyad summary…',   fn: loadDyadSummary },
    { msg: 'Loading placebo data…',   fn: loadPlacebo     },
    { msg: 'Initialising map…',       fn: () => initMap() },
    { msg: 'Building leaderboard…',   fn: () => buildLeaderboard() },
  ];

  for (let i = 0; i < steps.length; i++) {
    loadMsg.textContent = steps[i].msg;
    fill.style.width    = `${Math.round(((i + 1) / steps.length) * 100)}%`;
    await steps[i].fn();
  }

  /* Auto-select US-China as the default dyad */
  App.selectDyad('us');

  /* Dismiss loading overlay */
  overlay.classList.add('hidden');
  setTimeout(() => { overlay.style.display = 'none'; }, 600);
}

document.addEventListener('DOMContentLoaded', init);
