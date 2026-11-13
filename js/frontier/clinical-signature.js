// Small signature + clinical score (Tempus IPS-style framing): TMB + a few RNA genes, L1-logistic, nested CV, leakage toggle,
// and a locked model scored on a separate validation cohort.
import { header, figure, slider, select, button, stats, section, steps, math, seeAlso, legend, tip, untip, el } from '../lib/ui.js';
import { frame, cat, css, ink, ink2, muted } from '../lib/plot.js';
import { rng, sigmoid } from '../lib/data.js';
import { sourceCard, toyBadge, coreIdea, realityCheck, useIt, cite } from './_shared.js';
const d3 = window.d3;

const NG = 100, NMOD = 8, NDEV = 200, NVAL = 200, TOPK = 20, LAMS = [0.01, 0.03, 0.08];

function cohort(n, { sig, tmb, seed }) {
  const r = rng(seed); const X = [], y = [];
  for (let i = 0; i < n; i++) {
    const z = r.normal(), lt = r.normal();                       // IFN-γ-like module activity, log-TMB (standardised)
    const genes = Array.from({ length: NG }, (_, g) => (g < NMOD ? 0.75 * z + 0.66 * r.normal() : r.normal()));
    X.push([lt, ...genes]);
    y.push(r() < sigmoid(-0.4 + sig * z + tmb * lt) ? 1 : 0);
  }
  return { X, y };
}
const names = ['TMB', ...d3.range(NG).map((g) => (g < NMOD ? `IFNG-mod ${g + 1}` : `gene ${g + 1}`))];

/** L1-logistic regression by proximal gradient (ISTA). Features already standardised. TMB (col 0) is always kept. */
function l1logit(X, y, lam, iters = 120) {
  const n = X.length, p = X[0].length; const w = new Float64Array(p); let b = 0; const step = 0.5;
  const gr = new Float64Array(p);
  for (let it = 0; it < iters; it++) {
    gr.fill(0); let gb = 0;
    for (let i = 0; i < n; i++) { const xi = X[i]; let z = b; for (let j = 0; j < p; j++) z += w[j] * xi[j]; const e = sigmoid(z) - y[i]; gb += e; for (let j = 0; j < p; j++) gr[j] += e * xi[j]; }
    b -= step * gb / n;
    for (let j = 0; j < p; j++) { const v = w[j] - step * gr[j] / n, t = step * lam; w[j] = Math.sign(v) * Math.max(0, Math.abs(v) - t); }
  }
  return { w, b, predict: (x) => { let z = b; for (let j = 0; j < p; j++) z += w[j] * x[j]; return sigmoid(z); } };
}
function auc(scores, y) {
  const idx = d3.range(y.length).sort((a, b) => scores[a] - scores[b]);
  let rankSum = 0, np = 0; idx.forEach((i, r) => { if (y[i]) { rankSum += r + 1; np++; } });
  const nn = y.length - np; return np && nn ? (rankSum - np * (np + 1) / 2) / (np * nn) : 0.5;
}
function roc(scores, y) {
  const idx = d3.range(y.length).sort((a, b) => scores[b] - scores[a]);
  const P = d3.sum(y), N = y.length - P; let tp = 0, fp = 0; const pts = [{ x: 0, y: 0 }];
  idx.forEach((i) => { if (y[i]) tp++; else fp++; pts.push({ x: fp / N, y: tp / P }); });
  return pts;
}
/** Rank genes (cols 1..) by |mean difference| between classes on the given rows; return TMB + top-k columns. */
function selectCols(X, y, rows) {
  const s = d3.range(1, NG + 1).map((j) => {
    const a = rows.filter((i) => y[i]).map((i) => X[i][j]), b = rows.filter((i) => !y[i]).map((i) => X[i][j]);
    return { j, t: Math.abs(d3.mean(a) - d3.mean(b)) / Math.sqrt((d3.variance(a) || 1) / a.length + (d3.variance(b) || 1) / b.length) };
  }).sort((u, v) => v.t - u.t);
  return [0, ...s.slice(0, TOPK).map((d) => d.j)];
}
const sub = (X, rows, cols) => rows.map((i) => cols.map((j) => X[i][j]));
const folds = (n, k, seed) => { const r = rng(seed); const ord = r.shuffle(d3.range(n)); return d3.range(k).map((f) => ord.filter((_, i) => i % k === f)); };

/** Fit preselect + L1-logistic with λ chosen by inner CV. Returns a predictor on full feature rows. */
function fitPipeline(X, y, rows, leakCols) {
  const cols = leakCols || selectCols(X, y, rows);
  const inner = folds(rows.length, 3, 5).map((f) => f.map((i) => rows[i]));
  const best = LAMS.map((lam) => {
    const sc = [], yy = [];
    inner.forEach((te) => {
      const tr = rows.filter((i) => !te.includes(i));
      const c2 = leakCols || selectCols(X, y, tr);
      const m = l1logit(sub(X, tr, c2), tr.map((i) => y[i]), lam, 80);
      te.forEach((i) => { sc.push(m.predict(c2.map((j) => X[i][j]))); yy.push(y[i]); });
    });
    return { lam, a: auc(sc, yy) };
  }).reduce((a, b) => (b.a > a.a ? b : a));
  const m = l1logit(sub(X, rows, cols), rows.map((i) => y[i]), best.lam);
  return { cols, lam: best.lam, w: m.w, predict: (x) => m.predict(cols.map((j) => x[j])) };
}
function outerCV(X, y, leaky) {
  const all = d3.range(y.length);
  const leakCols = leaky ? selectCols(X, y, all) : null;   // the leak: genes chosen using every patient's outcome
  const scores = new Array(y.length);
  folds(y.length, 5, 9).forEach((te) => {
    const tr = all.filter((i) => !te.includes(i));
    const m = fitPipeline(X, y, tr, leakCols);
    te.forEach((i) => { scores[i] = m.predict(X[i]); });
  });
  return scores;
}

export default async function render(main) {
  header(main, 'Small signature + clinical score',
    'Companies such as Tempus build clinical scores from <b>a handful of RNA genes plus tumour mutational burden (TMB)</b>. The model is simple. The hard part is the evaluation: choose the genes inside the cross-validation, lock the model, and only then test it on a new cohort.');

  sourceCard(main, {
    org: 'Tempus',
    when: 'medRxiv preprint (v2), 2024',
    links: ['tempus'],
    claim: 'The Immune Profile Score, an ML model on TMB plus 8 RNA genes, was developed on n=1707 patients and validated on n=1600.',
    status: 'preprint',
  });

  coreIdea(main, `<p>A clinical signature should be <b>small</b> (cheap to run, easy to lock and to explain to a regulator), <b>combine modalities</b> (DNA-level TMB with RNA-level immune activity) and be validated on a <b>separate cohort</b> with a frozen model.</p>
  <p><b>Key ML ingredient:</b> L1-penalised (lasso) logistic regression. It drives most gene weights to exactly zero, so a short signature drops out of the fit. The honesty comes from <b>nested cross-validation</b>: gene preselection and the choice of penalty λ are repeated inside each training fold, never on the full data.</p>
  <p>The omics tab's <a href="#/omics/bulk-classifier">response classifier</a> page covers nested CV in depth. This page focuses on the clinical framing: development cohort, locked model, validation cohort.</p>`);

  const fig = figure(main, 'Development CV vs the locked model on a new cohort',
    `Mock cohort: ${NDEV} development and ${NVAL} validation patients, ${NG} genes (the first ${NMOD} share an IFN-γ-like module), and log-TMB. Pipeline: keep TMB + the top ${TOPK} genes by t-statistic → L1-logistic, with λ chosen by inner 3-fold CV → 5-fold outer CV.`);
  toyBadge(fig, 'Simulated patients; not Tempus data or its model.');
  const P = { sig: 0.6, tmb: 0.5, seed: 21, mode: 'both' };
  slider(fig.controls, { label: 'RNA module effect', min: 0, max: 1.5, step: 0.1, value: P.sig, fmt: (v) => v.toFixed(1), onInput: (v) => { P.sig = v; schedule(); } });
  slider(fig.controls, { label: 'TMB effect', min: 0, max: 1.5, step: 0.1, value: P.tmb, fmt: (v) => v.toFixed(1), onInput: (v) => { P.tmb = v; schedule(); } });
  select(fig.controls, { label: 'Gene selection', value: 'both', options: [['both', 'Show both'], ['honest', 'Inside CV (honest)'], ['leaky', 'Before CV on all data (leaky)']], onChange: (v) => { P.mode = v; paint(); } });
  button(fig.controls, 'New cohorts', () => { P.seed += 2; schedule(); });
  const readout = stats(fig.root);
  const row = el('div', { class: 'fig-row' }); fig.body.append(row);
  const left = el('div'), right = el('div'); row.append(left, right);
  legend(left, [['honest nested CV (dev)', cat(0)], ['leaky CV (dev)', cat(1)], ['locked model on validation cohort', cat(2)]]);
  const R = frame(left, { width: 440, height: 400, x: [0, 1], y: [0, 1], xLabel: 'false positive rate', yLabel: 'true positive rate' });
  R.layer('diag').append('line').attr('x1', R.x(0)).attr('y1', R.y(0)).attr('x2', R.x(1)).attr('y2', R.y(1)).attr('stroke', muted()).attr('stroke-dasharray', '4 3');
  legend(right, [['TMB', cat(3)], ['true IFN-γ module gene', cat(2)], ['noise gene', cat(7)]]);
  const C = frame(right, { width: 500, height: 400, x: [0, 1], y: [0, 1], yLabel: 'L1-logistic weight (locked model)', xScale: d3.scaleBand().padding(0.2), margin: { bottom: 70 } });

  let res, pend = null;
  function schedule() { if (pend) clearTimeout(pend); pend = setTimeout(() => { pend = null; compute(); paint(); }, 0); }
  function compute() {
    const dev = cohort(NDEV, P), val = cohort(NVAL, { ...P, seed: P.seed + 1 });
    const honest = outerCV(dev.X, dev.y, false), leaky = outerCV(dev.X, dev.y, true);
    const locked = fitPipeline(dev.X, dev.y, d3.range(NDEV), null);
    const vs = val.X.map((x) => locked.predict(x));
    res = { dev, val, honest, leaky, locked, vs, aH: auc(honest, dev.y), aL: auc(leaky, dev.y), aV: auc(vs, val.y) };
  }
  function paint() {
    const curves = [];
    if (P.mode !== 'leaky') curves.push({ k: 0, pts: roc(res.honest, res.dev.y) });
    if (P.mode !== 'honest') curves.push({ k: 1, pts: roc(res.leaky, res.dev.y) });
    curves.push({ k: 2, pts: roc(res.vs, res.val.y) });
    R.layer('c').selectAll('path').data(curves, (d) => d.k).join('path').attr('fill', 'none').attr('stroke-width', 2)
      .attr('stroke', (d) => cat(d.k)).attr('stroke-dasharray', (d) => (d.k === 2 ? '6 3' : null))
      .attr('d', (d) => d3.line().curve(d3.curveStepAfter).x((p) => R.x(p.x)).y((p) => R.y(p.y))(d.pts));
    const L = res.locked;
    const bars = L.cols.map((j, q) => ({ j, w: L.w[q] })).filter((d) => d.w !== 0).sort((a, b) => Math.abs(b.w) - Math.abs(a.w));
    const lim = Math.max(0.2, d3.max(bars, (d) => Math.abs(d.w)) || 0.2) * 1.1;
    C.x.domain(bars.map((d) => names[d.j])); C.y.domain([-lim, lim]);
    C.g.select('.axis.x').call(d3.axisBottom(C.x)).selectAll('text').attr('transform', 'rotate(-45)').attr('text-anchor', 'end').attr('dx', '-0.4em').attr('dy', '0.3em');
    C.g.select('.axis.x').attr('transform', `translate(0,${C.y(-lim)})`);
    C.g.select('.axis.y').call(d3.axisLeft(C.y).ticks(5)); C.g.select('.gridline').call(d3.axisLeft(C.y).ticks(5).tickSize(-C.w).tickFormat(''));
    C.layer('z').selectAll('line').data([0]).join('line').attr('x1', 0).attr('x2', C.w).attr('y1', C.y(0)).attr('y2', C.y(0)).attr('stroke', muted());
    C.layer('b').selectAll('rect').data(bars).join('rect').attr('x', (d) => C.x(names[d.j])).attr('width', C.x.bandwidth())
      .attr('y', (d) => C.y(Math.max(0, d.w))).attr('height', (d) => Math.abs(C.y(d.w) - C.y(0))).attr('rx', 2)
      .attr('fill', (d) => (d.j === 0 ? cat(3) : d.j <= NMOD ? cat(2) : cat(7)))
      .on('mousemove', (e, d) => tip(e, `<b>${names[d.j]}</b><br>weight ${d.w.toFixed(3)}`)).on('mouseleave', untip);
    const nTrue = bars.filter((d) => d.j >= 1 && d.j <= NMOD).length, nNoise = bars.filter((d) => d.j > NMOD).length;
    readout.set({
      'honest nested-CV AUC': res.aH.toFixed(2), 'leaky CV AUC': res.aL.toFixed(2), 'validation-cohort AUC': res.aV.toFixed(2),
      'locked signature': `TMB${L.w[0] ? '' : ' (zeroed)'} + ${nTrue + nNoise} genes (${nTrue} true)`, 'chosen λ': L.lam,
    });
  }
  compute(); paint();

  section(main, 'What to notice', `<ul>
    <li>Compare the readouts: the <b>leaky CV AUC</b> is higher than the honest one, and the <b>validation cohort</b> agrees with the honest one. In a paper, the leaky number is the one that does not replicate.</li>
    <li>Set <b>RNA module effect</b> and <b>TMB effect</b> to 0, so there is no signal at all. The honest AUC and the validation AUC hover around 0.5, but the leaky CV still reports a clearly better-than-chance AUC from pure noise genes.</li>
    <li>With TMB at 0 and the RNA module on, the lasso keeps mostly IFN-γ module genes. Because module genes are correlated, it often keeps only a few of the 8: <b>which</b> correlated gene survives is partly luck, so do not over-interpret individual genes.</li>
    <li>Click <b>New cohorts</b> a few times. Validation AUC moves by several points between draws with 200 patients. That is why cohorts in the thousands (as in the Tempus preprint) matter.</li>
  </ul>`);

  steps(main, 'How it works', [
    '<b>Development cohort.</b> For each outer fold, hold out 1/5 of patients.',
    'On the remaining 4/5 only: rank genes by a t-statistic, keep TMB + the top 20.',
    'Still inside: pick the lasso penalty λ by an inner 3-fold CV (gene ranking is redone inside each inner fold too).',
    'Fit L1-logistic on the 4/5 with that λ; score the held-out 1/5. After 5 folds every patient has an honest out-of-fold score; the AUC of those is the development estimate.',
    '<b>Leaky version:</b> the gene ranking is done once on all 200 patients <i>before</i> CV. The held-out patients\' outcomes have already influenced which genes were chosen.',
    '<b>Lock</b> the pipeline (refit on all development patients), freeze genes, weights and threshold, then score the validation cohort once.',
  ]);
  math(main, 'minimise   −Σ [ y log p + (1−y) log(1−p) ] / n  +  λ Σ |w_j|      p = σ(b + w_TMB·TMB + Σ w_j·gene_j)\nproximal step:  w_j ← sign(v)·max(0, |v| − η λ)   (soft-thresholding is what zeroes genes)');

  realityCheck(main, `<ul>
    <li>The ${cite('tempus', 'Tempus IPS preprint')} reports development n=1707 and validation n=1600: large cohorts and a separate validation set, which is the right shape. Its performance numbers are the company's own, in a preprint; our brief lists no independent external validation.</li>
    <li>${cite('bulkformer', 'BulkFormer')} (a masked-expression foundation model for bulk RNA) shows the other direction of travel: big pretrained models for bulk data. For an 8-gene clinical assay, though, a locked, sparse, linear model is far easier to validate and run as a lab test.</li>
    <li>The AstraZeneca × SOPHiA multimodal stratification work (POSEIDON) is announcement-level only in our brief, so we cannot assess it.</li>
  </ul>`);

  useIt(main, `<ul>
    <li>If you are building a response signature from your own bulk or pseudobulk data, put <b>every</b> data-driven step (HVG or DE filtering, batch correction fitted on labels, λ tuning) inside the CV loop.</li>
    <li>Report the honest nested-CV AUC as the development estimate, then lock the model and test it on a cohort from a different site or batch.</li>
    <li>Include the clinical baseline (TMB alone, PD-L1 alone) as a comparator. A signature only matters if it adds to what the clinic already measures.</li>
  </ul>`);

  seeAlso(main, [['omics', 'bulk-classifier', 'Response classifier & nested CV'], ['omics', 'bulk-feature-selection', 'Biomarker feature selection'], ['omics', 'bulk-validation', 'Locking & validating the signature'],
    ['concepts', 'cross-validation', 'Cross-validation'], ['concepts', 'regularization', 'Regularization (L1 / L2)'], ['concepts', 'logistic-regression', 'Logistic regression'], ['concepts', 'metrics', 'Classification metrics']]);
}
