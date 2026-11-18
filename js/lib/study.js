// The shared mock study (data/study/cohort.json, built by scripts/build_study.py).
// Every rewritten concept page uses these same patients, so the learner gets to know one dataset well.
import { loadJSON, el, html } from './ui.js';

let cached = null;
/** Returns {all, discovery, validation, meta, truth}. Each patient is a plain object keyed by column name. */
export async function loadCohort() {
  if (cached) return cached;
  const d = await loadJSON('data/study/cohort.json');
  const all = d.rows.map((r) => Object.fromEntries(d.columns.map((c, j) => [c, r[j]])));
  all.forEach((p) => { p.log2tmb = Math.log2(p.tmb); p.smoker = p.smoking === 'never' ? 0 : 1; p.centreB = p.centre === 'Centre B' ? 1 : 0; p.male = p.sex === 'M' ? 1 : 0; });
  cached = { all, discovery: all.filter((p) => p.cohort === 'discovery'), validation: all.filter((p) => p.cohort === 'validation'), meta: d.meta, truth: d.truth, about: d.about };
  return cached;
}

/** Model-ready numeric features. `get` maps a patient to the number the model sees. */
export const FEATURES = {
  ifng: { label: 'IFN-γ score', unit: 'log2 expr.', get: (p) => p.ifng, fmt: (v) => v.toFixed(2) },
  log2tmb: { label: 'TMB (log2)', unit: 'log2 mut/Mb', get: (p) => p.log2tmb, fmt: (v) => v.toFixed(2), note: 'TMB is skewed (1.7 to 39 mut/Mb), so models use log2(TMB): each +1 means TMB doubled.' },
  caf: { label: 'CAF / TGF-β score', unit: 'log2 expr.', get: (p) => p.caf, fmt: (v) => v.toFixed(2) },
  pdl1: { label: 'PD-L1 TPS', unit: '%', get: (p) => p.pdl1, fmt: (v) => v.toFixed(0) },
  cd8: { label: 'CD8 T cells', unit: '% of cells', get: (p) => p.cd8, fmt: (v) => v.toFixed(1) },
  prolif: { label: 'Proliferation', unit: 'log2 expr.', get: (p) => p.prolif, fmt: (v) => v.toFixed(2) },
  age: { label: 'Age', unit: 'years', get: (p) => p.age, fmt: (v) => v.toFixed(0) },
  smoker: { label: 'Ever smoker', unit: '0/1', get: (p) => p.smoker, fmt: (v) => (v ? 'yes' : 'no') },
  male: { label: 'Male', unit: '0/1', get: (p) => p.male, fmt: (v) => (v ? 'yes' : 'no') },
  centreB: { label: 'Centre B', unit: '0/1', get: (p) => p.centreB, fmt: (v) => (v ? 'B' : 'A') },
};

/** The "meet the data" card shown near the top of every lesson page. */
export function studyCard(main, cohort, focusHtml) {
  const d = cohort.discovery, nR = d.filter((p) => p.response).length;
  const card = el('div', { class: 'study-card' });
  card.append(html(`<div class="sc-k">The running example</div>
    <div class="sc-t">${d.length} lung-cancer patients treated with an anti-PD-1 immunotherapy</div>
    <div class="sc-d">Only <b>${nR} of ${d.length} (${Math.round(100 * nR / d.length)}%)</b> responded. For each patient we have a tumour biopsy (gene expression, mutations, PD-L1 staining) and basic clinical data. A further ${cohort.validation.length} patients from a third hospital are kept aside for validation. The data are simulated, so we know the <i>true</i> rule behind response and can check every model against it.</div>
    ${focusHtml ? `<div class="sc-q">${focusHtml}</div>` : ''}`));
  const cols = ['id', 'centre', 'age', 'smoking', 'pdl1', 'tmb', 'ifng', 'cd8', 'caf', 'prolif', 'phenotype', 'response'];
  const det = el('details', {}, el('summary', {}, 'Peek at the data (first 10 patients) and what each column means'));
  const t = el('table', { class: 'simple compact' });
  t.append(el('thead', {}, el('tr', {}, cols.map((c) => el('th', { title: cohort.meta[c] ? cohort.meta[c].desc : '' }, cohort.meta[c] ? cohort.meta[c].label : c)))));
  t.append(el('tbody', {}, d.slice(0, 10).map((p) => el('tr', {}, cols.map((c) => el('td', {}, c === 'response' ? (p[c] ? 'yes' : 'no') : String(p[c])))))));
  det.append(el('div', { class: 'table-wrap' }, t));
  det.append(el('ul', { class: 'prose' }, ['ifng', 'tmb', 'caf', 'pdl1', 'cd8', 'prolif', 'phenotype'].map((c) => el('li', { html: `<b>${cohort.meta[c].label}</b>${cohort.meta[c].unit ? ` (${cohort.meta[c].unit})` : ''}: ${cohort.meta[c].desc}` }))));
  card.append(det);
  main.append(card);
  return card;
}
