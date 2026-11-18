// Train/test split and overfitting, lesson format.
// Arc: what → why → where → how it works (6 steps: hold-out split → complexity curve → bias and variance
//      → 2-D complexity lab with try/see/means table → learning curves → Centre C as external test) → pitfalls → recap.
// Numbers in the prose are computed at render time from data/study/cohort.json and data/study/train_test.json
// (extra patients simulated from the same generator by scripts/build_train_test.py).
import { header, figure, slider, select, button, checkbox, stats, legend, seeAlso, animate, tip, untip, el, html, loadJSON } from '../lib/ui.js';
import { frame, decisionCanvas, cat, css, ink, ink2, muted } from '../lib/plot.js';
import { mean, solve, sigmoid, clamp } from '../lib/data.js';
import { fitLogistic, predictProb, logLoss, auc, confusion, standardize, stratifiedSplit } from '../lib/models.js';
import { loadCohort, studyCard } from '../lib/study.js';
import { lessonNav, chapter, prose, rung, worked, quiz, cards, table, code, recap } from '../lib/lesson.js';
const d3 = window.d3;

const um = (t) => t.replace(/^-/, '−');
const f2 = (v) => (Math.abs(v) < 0.005 ? '0.00' : um(v.toFixed(2)));
const f3 = (v) => um(v.toFixed(3));
const pc = (v) => Math.round(100 * v) + '%';
const sum = (a) => a.reduce((s, v) => s + v, 0);
const jit = (i) => (((i * 7919) % 101) / 101 - 0.5);
const dotv = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
const L2S = [0.001, 0.1, 1, 10];

// Legendre polynomial features. Inputs are clipped to the fitted range (IFN-γ 2.5–10, log2 TMB 0.5–5.5), as in the Python builder.
const leg = (u, d) => { const p = [1, u]; for (let k = 1; k < d; k++) p.push(((2 * k + 1) * u * p[k] - k * p[k - 1]) / (k + 1)); return p; };
const uI = (x) => clamp((x - 6.25) / 3.75, -1, 1), uT = (t) => clamp((t - 3) / 2.5, -1, 1);
const poly1 = (x, d) => leg(uI(x), d).slice(1, d + 1);
const poly2 = (x, t, d) => { const a = leg(uI(x), d), b = leg(uT(t), d), o = []; for (let s = 1; s <= d; s++) for (let j = 0; j <= s; j++) o.push(a[s - j] * b[j]); return o; };
const nPar2 = (d) => (d + 1) * (d + 2) / 2;

/** Penalised logistic regression by Newton steps with step halving (stable for many polynomial terms). */
function fitLR(X, y, l2 = 0.001, maxIter = 100) {
  const n = X.length, k = (X[0] ? X[0].length : 0) + 1, A = X.map((r) => [1, ...r]);
  const obj = (b) => { let s = 0; for (let i = 0; i < n; i++) { const z = dotv(A[i], b); s += (z > 0 ? z + Math.log1p(Math.exp(-z)) : Math.log1p(Math.exp(z))) - y[i] * z; } for (let a = 1; a < k; a++) s += 0.5 * l2 * b[a] * b[a]; return s; };
  let b = new Array(k).fill(0), f = obj(b);
  for (let it = 0; it < maxIter; it++) {
    const g = new Array(k).fill(0), H = Array.from({ length: k }, () => new Array(k).fill(0));
    for (let i = 0; i < n; i++) {
      const r = A[i], p = sigmoid(dotv(r, b)), w = Math.max(p * (1 - p), 1e-10), e = y[i] - p;
      for (let a = 0; a < k; a++) { g[a] += r[a] * e; const wa = w * r[a]; for (let c = a; c < k; c++) H[a][c] += wa * r[c]; }
    }
    for (let a = 0; a < k; a++) { for (let c = 0; c < a; c++) H[a][c] = H[c][a]; if (a) { H[a][a] += l2; g[a] -= l2 * b[a]; } }
    const s = solve(H, g);
    let t = 1, bn = b, fn = f;
    while (t > 1e-6) { bn = b.map((v, a) => v + t * s[a]); fn = obj(bn); if (fn <= f + 1e-12) break; t /= 2; }
    b = bn; const done = Math.abs(f - fn) < 1e-10 * (1 + Math.abs(f)); f = fn;
    if (done) break;
  }
  return { b0: b[0], w: b.slice(1) };
}
const pr = (fit, row) => sigmoid(fit.b0 + dotv(fit.w, row));

/** k-nearest-neighbour vote share for each query row (training queries include themselves, as in standard software). */
function knn(Ztr, ytr, Zq, k) {
  const bd = new Float64Array(k), bi = new Int32Array(k);
  return Zq.map((q) => {
    bd.fill(Infinity); bi.fill(-1);
    for (let i = 0; i < Ztr.length; i++) {
      const z = Ztr[i], d = (z[0] - q[0]) ** 2 + (z[1] - q[1]) ** 2 + (z[2] - q[2]) ** 2;
      if (d >= bd[k - 1]) continue;
      let j = k - 1; while (j > 0 && bd[j - 1] > d) { bd[j] = bd[j - 1]; bi[j] = bi[j - 1]; j--; }
      bd[j] = d; bi[j] = i;
    }
    let s = 0; for (let j = 0; j < k; j++) s += ytr[bi[j]];
    return s / k;
  });
}
/** Hanley–McNeil standard error of an AUC. */
function aucSE(A, nP, nN) {
  const q1 = A / (2 - A), q2 = 2 * A * A / (1 + A);
  return Math.sqrt((A * (1 - A) + (nP - 1) * (q1 - A * A) + (nN - 1) * (q2 - A * A)) / (nP * nN));
}

export default async function render(main) {
  const [C, T] = await Promise.all([loadCohort(), loadJSON('data/study/train_test.json')]);
  const D = C.discovery, V = C.validation;
  const y = D.map((p) => p.response), yv = V.map((p) => p.response);
  const n = D.length, nV = V.length, nR = sum(y), base = nR / n;
  const F3 = (p) => [p.ifng, p.log2tmb, p.caf];
  const acc = (p, yy) => confusion(p, yy, 0.5).acc;

  // ---- models fitted on all 180 discovery patients, scored on themselves and on Centre C
  const st3 = standardize(D.map(F3)), Z3 = st3.Z, Zv3 = V.map((p) => st3.apply(F3(p)));
  const fitL1 = fitLogistic(D.map((p) => [p.ifng]), y), fitL3 = fitLogistic(Z3, y);
  const labCache = new Map();
  const lab = (d, l2) => {
    const key = d + '|' + l2;
    if (!labCache.has(key)) {
      const Xt = D.map((p) => poly2(p.ifng, p.log2tmb, d)), fit = fitLR(Xt, y, l2);
      const pt = Xt.map((r) => pr(fit, r)), pv = V.map((p) => pr(fit, poly2(p.ifng, p.log2tmb, d)));
      labCache.set(key, { fit, pt, pv, trLL: logLoss(pt, y), vaLL: logLoss(pv, yv), trAUC: auc(pt, y), vaAUC: auc(pv, yv) });
    }
    return labCache.get(key);
  };
  const MODELS = [
    { k: 'base', lab: 'Base rate for everyone (1 parameter)', pt: D.map(() => base), pv: V.map(() => base) },
    { k: 'lr1', lab: 'Logistic, IFN-γ (2 parameters)', pt: D.map((p) => predictProb([p.ifng], fitL1.b0, fitL1.w)), pv: V.map((p) => predictProb([p.ifng], fitL1.b0, fitL1.w)) },
    { k: 'lr3', lab: 'Logistic, IFN-γ + TMB + CAF (4)', pt: Z3.map((z) => predictProb(z, fitL3.b0, fitL3.w)), pv: Zv3.map((z) => predictProb(z, fitL3.b0, fitL3.w)) },
    { k: 'p7', lab: `Polynomial degree 7, IFN-γ × TMB (${nPar2(7)})`, pt: lab(7, 0.001).pt, pv: lab(7, 0.001).pv },
    { k: 'nn15', lab: '15-nearest neighbours (stores all 180)', pt: knn(Z3, y, Z3, 15), pv: knn(Z3, y, Zv3, 15) },
    { k: 'nn1', lab: '1-nearest neighbour (stores all 180)', pt: knn(Z3, y, Z3, 1), pv: knn(Z3, y, Zv3, 1) },
  ];
  MODELS.forEach((m) => Object.assign(m, { tA: auc(m.pt, y), vA: auc(m.pv, yv), tC: acc(m.pt, y), vC: acc(m.pv, yv) }));
  const M = Object.fromEntries(MODELS.map((m) => [m.k, m]));

  header(main, 'Train/test split and overfitting',
    'A model is judged by how well it predicts <b>patients it has not seen</b>. Performance on the training data is always optimistic, and the gap widens as the model becomes more flexible: the model starts to describe the noise in its particular sample. This is <b>overfitting</b>.');
  lessonNav(main);
  studyCard(main, C, `<b>Question for this page:</b> how well will a model fitted to the ${n} discovery patients predict response in new patients, and how flexible should that model be?`);

  // =====================================================================================
  // 1. WHAT
  // =====================================================================================
  let ch = chapter(main, 'what', 'What is a train/test split?', 'In one line: <b>fit on one set of patients, measure on another</b>.');
  prose(ch, `<p>A predictive model is fitted to a set of labelled examples, the <b>training set</b>. Its purpose is to predict new cases, so its performance has to be measured on examples that played no part in fitting: a <b>test set</b>, also called a hold-out set. The ability to predict new cases is called <b>generalisation</b>.</p>
    <p>This study has the standard three-level structure. Discovery patients from Centres A and B are used for fitting and for choosing between models; the ${nV} patients from Centre C are reserved as an <b>external</b> test set and used once, at the end.</p>`);
  ch.append(splitSvg(n, nV, nR, sum(yv)));
  prose(ch, '<p>Key terms:</p>');
  table(ch, [
    { t: 'Training set', m: 'The examples the model is fitted to', s: `Discovery patients, or a part of them (${n} at most)` },
    { t: 'Validation (tuning) set', m: 'Held-out examples used to <i>choose</i> between models or settings', s: 'A random part of the discovery cohort, or cross-validation folds' },
    { t: 'Test set', m: 'Held-out examples used once, to report final performance', s: `Centre C (${nV} patients); clinical papers call this the "validation cohort"` },
    { t: 'Training error', m: 'Error measured on the training set', s: `1-nearest neighbour: training accuracy ${pc(M.nn1.tC)}` },
    { t: 'Test (generalisation) error', m: 'Error expected on new cases from the same setting', s: `1-nearest neighbour: Centre C accuracy ${pc(M.nn1.vC)}` },
    { t: 'Model complexity', m: 'How flexible the model is: number of parameters, polynomial degree, 1/k in k-nearest neighbours, tree depth', s: `From 1 parameter (base rate) to ${nPar2(7)} (degree-7 polynomial)` },
    { t: 'Underfitting', m: 'The model is too simple to capture the real pattern: training and test error are both high', s: 'Predicting the base rate for everyone' },
    { t: 'Overfitting', m: 'The model fits noise in the training set: low training error, higher test error', s: `Degree-7 polynomial: training AUC ${f2(M.p7.tA)}, Centre C ${f2(M.p7.vA)}` },
    { t: 'Bias / variance', m: 'Error from systematic misfit (bias) and from sensitivity to the particular training sample (variance)', s: 'Chapter 4, step 3' },
  ], [['t', 'Term'], ['m', 'Meaning'], ['s', 'In our study']]);

  // =====================================================================================
  // 2. WHY
  // =====================================================================================
  ch = chapter(main, 'why', 'Why do we need it?', 'Training performance rewards memorisation, so it cannot rank models.');
  prose(ch, `<p>Fitting chooses parameters that make the training data look as predictable as possible. Part of what the model learns is the real relationship, and part is the noise peculiar to those ${n} patients. On the training data both parts help; on new patients only the first does. The more parameters a model has, the more noise it can absorb, so <b>training performance increases with flexibility whether or not the model has learned anything that transfers</b>.</p>`);
  const fw = figure(ch, 'The same six models, scored on their own training patients and on Centre C',
    'Each row is a model fitted to all discovery patients. The hollow dot is its score on those patients; the filled dot is its score on the Centre C patients. Hover a dot for the exact values. AUC (area under the ROC curve) is the probability that a random responder receives a higher score than a random non-responder: 0.5 is chance, 1 is perfect ranking.');
  let wMet = 'auc';
  select(fw.controls, { label: 'Metric', value: wMet, options: [['auc', 'AUC'], ['acc', 'accuracy (threshold 0.5)']], onChange: (v) => { wMet = v; paintW(); } });
  const FWr = frame(fw.body, { width: 820, height: 280, x: [0.45, 1.0], y: [0, 6], xLabel: 'score (higher is better)', grid: false, axes: false, margin: { left: 300, right: 20, top: 8 } });
  const axW = FWr.g.append('g').attr('class', 'axis x').attr('transform', `translate(0,${FWr.h})`);
  const yW = d3.scaleBand().domain(MODELS.map((m) => m.k)).range([0, FWr.h]).padding(0.35);
  FWr.layer('lab').selectAll('text').data(MODELS).join('text').attr('x', -10).attr('y', (m) => yW(m.k) + yW.bandwidth() / 2 + 4).attr('text-anchor', 'end').attr('font-size', 12.5).attr('fill', ink2()).text((m) => m.lab);
  const dumb = FWr.layer('d');
  legend(fw.body, [['training patients (hollow)', ink2()], ['Centre C patients', cat(6)]]);
  function paintW() {
    const g = (m) => (wMet === 'auc' ? [m.tA, m.vA] : [m.tC, m.vC]);
    FWr.x.domain(wMet === 'auc' ? [0.45, 1] : [0.55, 1]); axW.call(d3.axisBottom(FWr.x).ticks(6));
    dumb.selectAll('*').remove();
    MODELS.forEach((m) => {
      const [a, b] = g(m), cy = yW(m.k) + yW.bandwidth() / 2, t = (e) => tip(e, `${m.lab}<br>training ${wMet === 'auc' ? 'AUC' : 'accuracy'}: ${f3(a)}<br>Centre C: ${f3(b)}<br>gap: ${f3(a - b)}`);
      dumb.append('line').attr('x1', FWr.x(a)).attr('x2', FWr.x(b)).attr('y1', cy).attr('y2', cy).attr('stroke', muted()).attr('stroke-width', 2);
      dumb.append('circle').attr('cx', FWr.x(a)).attr('cy', cy).attr('r', 6).attr('fill', css('--surface')).attr('stroke', ink2()).attr('stroke-width', 2).on('mousemove', t).on('mouseleave', untip);
      dumb.append('circle').attr('cx', FWr.x(b)).attr('cy', cy).attr('r', 6).attr('fill', cat(6)).attr('stroke', css('--surface')).attr('stroke-width', 1.5).on('mousemove', t).on('mouseleave', untip);
    });
  }
  paintW();
  const bestTrain = MODELS.reduce((a, m) => (m.tA > a.tA ? m : a)), bestVal = MODELS.reduce((a, m) => (m.vA > a.vA ? m : a));
  prose(ch, `<p>Ranked by training AUC, the best model is the <b>${bestTrain.lab.split(' (')[0]}</b> (${f2(bestTrain.tA)}), which on Centre C scores ${f2(bestTrain.vA)}. Ranked by Centre C AUC, the best is the <b>${bestVal.lab.split(' (')[0]}</b> (${f2(bestVal.vA)}). The degree-7 polynomial gains ${f2(M.p7.tA - M.lr3.tA)} in training AUC over the four-parameter logistic model and ${M.p7.vA < M.lr3.vA ? `loses ${f2(M.lr3.vA - M.p7.vA)}` : `gains ${f2(M.p7.vA - M.lr3.vA)}`} on Centre C.</p>`);
  worked(ch, 'A model that memorises: 1-nearest neighbour',
    `<p>The 1-nearest-neighbour rule predicts a patient's outcome as the outcome of the most similar training patient (here, in standardised IFN-γ, log2 TMB and CAF). Asked about a training patient, it finds that patient and returns the recorded outcome, so its training accuracy is <b>${pc(M.nn1.tC)}</b> by construction: it has stored the data rather than summarised it. On Centre C its accuracy is <b>${pc(M.nn1.vC)}</b>, compared with ${pc(M.lr3.vC)} for the four-parameter logistic model and ${pc(1 - mean(yv))} for predicting "no response" for every patient.</p>
     <p>${M.nn1.vC >= M.lr3.vC ? `At the 0.5 threshold the stored patients classify ${Math.round((M.nn1.vC - M.lr3.vC) * nV)} more Centre C patients correctly than the logistic model (${pc(M.nn1.vC)} against ${pc(M.lr3.vC)}), so accuracy at a single threshold can favour a memorising model. The 1-nearest-neighbour rule returns only yes or no, however, so it cannot rank patients: its Centre C AUC is ${f2(M.nn1.vA)}, against ${f2(M.lr3.vA)} for the logistic model. ` : ''}The point is the gap: the training score of ${pc(M.nn1.tC)} carries no information about the ${pc(M.nn1.vC)} that new patients will see.</p>`);

  // =====================================================================================
  // 3. WHERE
  // =====================================================================================
  ch = chapter(main, 'where', 'Where is it used?', 'Every claim that a model predicts something should be backed by performance on data not used to build it.');
  cards(ch, [
    { tag: 'Clinical biomarkers', title: 'Discovery and validation cohorts', body: 'A response or prognostic signature is developed in one cohort and tested, locked, in an independent one, as with Centres A/B and C here. Reporting guidelines (TRIPOD) distinguish internal from external validation.' },
    { tag: 'Transcriptomics', title: 'Train on TCGA, test on GEO', body: 'Subtype or outcome classifiers trained on one compendium are tested on independently generated datasets, which also exposes platform and batch differences.' },
    { tag: 'Genetics', title: 'Polygenic risk scores', body: 'Variant weights come from a discovery GWAS; the score is tuned in one sample and evaluated in a separate target sample with no overlapping or related individuals.' },
    { tag: 'Single-cell', title: 'Cell-type classifiers', body: 'Annotation models are evaluated on held-out donors or whole held-out datasets, not on random cells, because cells from one donor share technical and biological signal.' },
    { tag: 'Digital pathology', title: 'Slide-level models', body: 'Splits are made by patient and often by hospital, because tiles from one slide, or slides from one scanner, are more similar to each other than to new cases.' },
    { tag: 'Protein structure', title: 'Blind and time-split tests', body: 'CASP assesses structure prediction on structures not yet released; sequence-identity or date cut-offs keep near-duplicates of test proteins out of training.' },
    { tag: 'Model selection', title: 'Choosing complexity', body: 'Polynomial degree, the number of neighbours, tree depth, the penalty strength or the number of genes in a signature are all chosen by held-out performance.' },
    { tag: 'Outside biology', title: 'Benchmarks and competitions', body: 'Public leaderboards keep a private test set that is scored only at the end, because repeated scoring on a public test set rewards over-tuning to it.' },
  ]);
  prose(ch, `<p><b>When a single random split is not enough:</b> with small cohorts a single split leaves part of the data out of fitting and gives a noisy estimate (use cross-validation, or repeated splits); samples that are not independent (several biopsies or cells per patient) must be split by patient; data collected over time should be split by date when the model will be used prospectively. When the aim is <b>inference</b> rather than prediction, such as estimating an odds ratio with a confidence interval, the usual approach is to fit all data and report uncertainty instead of holding data out.</p>`);

  // =====================================================================================
  // 4. HOW IT WORKS
  // =====================================================================================
  ch = chapter(main, 'how', 'How it works', 'Six steps: a hold-out split, model complexity, bias and variance, a complexity lab, learning curves and the external test.');

  // ---- Step 1: hold-out split and its variability
  let rb = rung(ch, 1, 'Hold patients out before fitting',
    `<p>The simplest check is a <b>hold-out split</b>: set a random fraction of the discovery patients aside, fit on the rest, and score the fitted model on the held-out patients. The split is <b>stratified</b>, so both parts keep the ${pc(base)} response rate. The held-out score depends on which patients happen to be held out; repeating the split many times shows how much.</p>`);
  const fs = figure(rb, 'One split, and the same split repeated 80 times',
    'Left: discovery patients on two of the three features; filled dots are training patients, open circles are held-out patients. Right: held-out AUC over 80 random splits (bars), the current split (solid line) and the mean training AUC (dashed line). Hover a bar for the range it covers.');
  let sMod = 'lr3', sFrac = 0.3, sSeed = 1;
  const SM = { lr3: 'Logistic, IFN-γ + TMB + CAF', nn15: '15-nearest neighbours', nn1: '1-nearest neighbour' };
  select(fs.controls, { label: 'Model', value: sMod, options: Object.entries(SM), onChange: (v) => { sMod = v; paintS(); } });
  slider(fs.controls, { label: 'held-out fraction', min: 0.1, max: 0.5, step: 0.1, value: sFrac, fmt: pc, onInput: (v) => { sFrac = v; paintS(); } });
  button(fs.controls, 'New split', () => { sSeed = sSeed % 80 + 1; paintS(); }, true);
  const sS = stats(fs.root);
  const rowS = el('div', { class: 'fig-row' }); fs.body.append(rowS);
  const sa = el('div'), sb = el('div'); rowS.append(sa, sb);
  const FS1 = frame(sa, { width: 480, height: 320, x: [2.2, 10.3], y: [0.6, 5.5], xLabel: 'IFN-γ score', yLabel: 'log2 TMB (mut/Mb)' });
  const FS2 = frame(sb, { width: 460, height: 320, x: [0.55, 1.0], y: [0, 20], xLabel: 'held-out AUC', yLabel: 'number of splits' });
  const ptsS = FS1.layer('p'), barS = FS2.layer('b'), refS = FS2.layer('r');
  legend(fs.body, [['no response', cat(0)], ['responded', cat(1)], ['current split', ink()], ['mean training AUC (dashed)', ink2()]]);
  const splitCache = new Map();
  function runSplits(mk, frac) {
    const key = mk + frac;
    if (splitCache.has(key)) return splitCache.get(key);
    const out = d3.range(1, 81).map((seed) => {
      const { train, test } = stratifiedSplit(y, frac, seed);
      const st = standardize(train.map((i) => F3(D[i]))), Zt = st.Z, Zh = test.map((i) => st.apply(F3(D[i])));
      const yt = train.map((i) => y[i]), yh = test.map((i) => y[i]);
      let pt, ph;
      if (mk === 'lr3') { const f = fitLogistic(Zt, yt); pt = Zt.map((z) => predictProb(z, f.b0, f.w)); ph = Zh.map((z) => predictProb(z, f.b0, f.w)); }
      else { const k = mk === 'nn1' ? 1 : 15; pt = knn(Zt, yt, Zt, k); ph = knn(Zt, yt, Zh, k); }
      return { test: new Set(test), tr: auc(pt, yt), ho: auc(ph, yh) };
    });
    splitCache.set(key, out);
    return out;
  }
  function paintS() {
    const R = runSplits(sMod, sFrac), cur = R[sSeed - 1], hos = R.map((r) => r.ho).sort((a, b) => a - b);
    ptsS.selectAll('circle').data(D).join('circle').attr('cx', (d) => FS1.x(d.ifng)).attr('cy', (d) => FS1.y(d.log2tmb)).attr('r', 4.2)
      .attr('fill', (d, i) => (cur.test.has(i) ? css('--surface') : cat(d.response))).attr('stroke', (d, i) => (cur.test.has(i) ? cat(d.response) : css('--surface'))).attr('stroke-width', (d, i) => (cur.test.has(i) ? 2 : 1))
      .on('mousemove', (e, d) => tip(e, `${d.id} · ${cur.test.has(D.indexOf(d)) ? 'held out' : 'training'}<br>IFN-γ ${d.ifng}, TMB ${d.tmb} mut/Mb<br>${d.response ? 'responded' : 'no response'}`)).on('mouseleave', untip);
    const bins = d3.bin().domain([0.55, 1.0]).thresholds(d3.range(0.55, 1.0, 0.0125))(hos);
    FS2.y.domain([0, Math.max(10, d3.max(bins, (b) => b.length) + 2)]); FS2.g.select('.axis.y').call(d3.axisLeft(FS2.y).ticks(5));
    FS2.g.select('.gridline').call(d3.axisLeft(FS2.y).ticks(5).tickSize(-FS2.w).tickFormat(''));
    barS.selectAll('rect').data(bins).join('rect').attr('x', (b) => FS2.x(b.x0) + 0.5).attr('width', (b) => Math.max(0, FS2.x(b.x1) - FS2.x(b.x0) - 1)).attr('y', (b) => FS2.y(b.length)).attr('height', (b) => FS2.h - FS2.y(b.length)).attr('fill', cat(6)).attr('fill-opacity', 0.65)
      .on('mousemove', (e, b) => tip(e, `held-out AUC ${f2(b.x0)}–${f2(b.x1)}<br>${b.length} of 80 splits`)).on('mouseleave', untip);
    const mTr = mean(R.map((r) => r.tr));
    refS.selectAll('line').data([[cur.ho, ink(), null], [Math.min(0.999, mTr), ink2(), '6 4']]).join('line').attr('x1', (d) => FS2.x(d[0])).attr('x2', (d) => FS2.x(d[0])).attr('y1', 0).attr('y2', FS2.h).attr('stroke', (d) => d[1]).attr('stroke-width', 2).attr('stroke-dasharray', (d) => d[2]);
    sS.set({ 'this split: training AUC': f3(cur.tr), 'held-out AUC': f3(cur.ho), 'held-out patients': `${cur.test.size} of ${n}`, 'held-out AUC, 80 splits (10th–90th pct.)': `${f2(d3.quantile(hos, 0.1))}–${f2(d3.quantile(hos, 0.9))}` });
  }
  paintS();
  const r30 = runSplits('lr3', 0.3), h30 = r30.map((r) => r.ho).sort((a, b) => a - b), r10 = runSplits('lr3', 0.1).map((r) => r.ho).sort((a, b) => a - b);
  prose(rb, `<p>For the logistic model with 30% held out, the mean training AUC is ${f2(mean(r30.map((r) => r.tr)))} and the mean held-out AUC ${f2(mean(h30))}; the gap is small because the model has only four parameters. The individual held-out values range from ${f2(d3.quantile(h30, 0.1))} to ${f2(d3.quantile(h30, 0.9))} (10th to 90th percentile), and from ${f2(d3.quantile(r10, 0.1))} to ${f2(d3.quantile(r10, 0.9))} when only 10% is held out. A single small hold-out set gives an imprecise estimate; cross-validation averages over several splits for this reason.</p>`);

  // ---- Step 2: complexity curve (polynomial in IFN-γ)
  const DEG = T.deg1, DMAX = DEG[DEG.length - 1];
  const fits1 = DEG.map((d) => { const f = fitLR(D.map((p) => poly1(p.ifng, d)), y); const pt = D.map((p) => pr(f, poly1(p.ifng, d))), pv = V.map((p) => pr(f, poly1(p.ifng, d))); return { d, f, tr: logLoss(pt, y), va: logLoss(pv, yv) }; });
  const dBest = fits1.reduce((a, c) => (c.va < a.va ? c : a)).d;
  rb = rung(ch, 2, 'Model complexity: from underfitting to overfitting',
    `<p>Complexity can be varied on a single feature. The model is a logistic regression on a polynomial of IFN-γ: degree 0 is a constant (the base rate), degree 1 the usual S-curve, and each extra degree adds one parameter that lets the curve bend once more. Performance is measured by <b>log-loss</b>, the average of −ln(probability assigned to the observed outcome); lower is better, and it penalises confident wrong predictions heavily.</p>`);
  const fc = figure(rb, 'Polynomial degree against training and Centre C log-loss',
    'Left: discovery patients at 1 (responded) and 0 (no response), spread vertically so they do not overlap; diamonds are observed response rates in eight equal-sized IFN-γ groups; the dashed line is the true P(response | IFN-γ) in the Centre A + B population, known from the simulation. Right: log-loss at every degree; the ring marks the selected degree.');
  let deg = 1;
  slider(fc.controls, { label: 'polynomial degree', min: 0, max: DMAX, step: 1, value: deg, onInput: (v) => { deg = v; paintC(); } });
  const sC = stats(fc.root);
  const rowC = el('div', { class: 'fig-row' }); fc.body.append(rowC);
  const ca = el('div'), cb = el('div'); rowC.append(ca, cb);
  const FC = frame(ca, { width: 480, height: 320, x: [2.2, 10.3], y: [-0.1, 1.1], xLabel: 'IFN-γ score', yLabel: 'P(response)' });
  FC.layer('pts').selectAll('circle').data(D).join('circle').attr('cx', (d) => FC.x(d.ifng)).attr('cy', (d, i) => FC.y(d.response + jit(i) * 0.12)).attr('r', 3.4).attr('fill', (d) => cat(d.response)).attr('fill-opacity', 0.6);
  const curveL = (pts) => d3.line().x((q) => FC.x(q[0])).y((q) => FC.y(q[1]))(pts);
  FC.layer('truth').append('path').attr('d', curveL(T.grid.map((g, i) => [g, T.truthAB[i]]))).attr('fill', 'none').attr('stroke', ink2()).attr('stroke-width', 2).attr('stroke-dasharray', '6 4');
  FC.layer('bins').selectAll('path').data(qBins(D, (p) => p.ifng, 8)).join('path').attr('d', d3.symbol(d3.symbolDiamond, 80)).attr('transform', (b) => `translate(${FC.x(b.x)},${FC.y(b.r / b.n)})`).attr('fill', ink()).attr('stroke', css('--surface'))
    .on('mousemove', (e, b) => tip(e, `IFN-γ ${f2(b.lo)}–${f2(b.hi)}<br>${b.r} of ${b.n} responded (${pc(b.r / b.n)})`)).on('mouseleave', untip);
  const fitC = FC.layer('fit');
  const llMax = Math.min(1.2, d3.max(fits1, (f) => Math.max(f.tr, f.va)) * 1.05), llMin = d3.min(fits1, (f) => f.tr) * 0.95;
  const FC2 = frame(cb, { width: 460, height: 320, x: [0, DMAX], y: [llMin, llMax], xLabel: 'polynomial degree (parameters − 1)', yLabel: 'log-loss (lower is better)' });
  [['tr', ink2(), '6 4'], ['va', cat(6), null]].forEach(([k, c, dash]) => {
    FC2.layer('l').append('path').attr('d', d3.line().x((f) => FC2.x(f.d)).y((f) => FC2.y(Math.min(llMax, f[k])))(fits1)).attr('fill', 'none').attr('stroke', c).attr('stroke-width', 2).attr('stroke-dasharray', dash);
    FC2.layer('p').selectAll('circle.' + k).data(fits1).join('circle').attr('class', k).attr('cx', (f) => FC2.x(f.d)).attr('cy', (f) => FC2.y(Math.min(llMax, f[k]))).attr('r', 4).attr('fill', c).attr('stroke', css('--surface'))
      .on('mousemove', (e, f) => tip(e, `degree ${f.d}<br>training log-loss ${f3(f.tr)}<br>Centre C log-loss ${f3(f.va)}`)).on('mouseleave', untip);
  });
  const ringC = FC2.layer('ring');
  legend(fc.body, [['no response', cat(0)], ['responded', cat(1)], ['true curve (dashed)', ink2()], ['fitted curve', cat(2)], ['training log-loss (dashed)', ink2()], ['Centre C log-loss', cat(6)]]);
  const gridFine = d3.range(2.2, 10.31, 0.05);
  function paintC() {
    const F = fits1[deg];
    fitC.selectAll('path').data([0]).join('path').attr('fill', 'none').attr('stroke', cat(2)).attr('stroke-width', 2.6).attr('d', curveL(gridFine.map((x) => [x, pr(F.f, poly1(x, deg))])));
    ringC.selectAll('circle').data([F.tr, F.va]).join('circle').attr('cx', FC2.x(deg)).attr('cy', (v) => FC2.y(Math.min(llMax, v))).attr('r', 8).attr('fill', 'none').attr('stroke', ink()).attr('stroke-width', 1.5);
    sC.set({ degree: deg, parameters: deg + 1, 'training log-loss': f3(F.tr), 'Centre C log-loss': f3(F.va), 'gap': f3(F.va - F.tr) });
  }
  paintC();
  const trMono = fits1.every((f, i) => i === 0 || f.tr <= fits1[i - 1].tr + 1e-9);
  prose(rb, `<p>Training log-loss ${trMono ? 'falls with every added degree' : 'falls almost steadily with degree'}, from ${f3(fits1[0].tr)} at degree 0 to ${f3(fits1[DMAX].tr)} at degree ${DMAX}. Centre C log-loss is lowest at degree ${dBest} (${f3(fits1[dBest].va)}) and reaches ${f3(fits1[DMAX].va)} at degree ${DMAX}. Degree 0 <b>underfits</b>: it ignores IFN-γ entirely. Beyond degree ${dBest} the extra bends follow clusters of individual patients, which is <b>overfitting</b>. Underfitting is visible only at degree 0 here because the true curve is close to the simple S-shape that degree 1 can already represent; with a more complex true relationship the best degree would be higher. The Centre C curve is not smooth: with ${nV} patients, a few patients near the edges of the IFN-γ range move it noticeably.</p>
    <p class="fig-caption">In heavily over-parameterised models (more parameters than training examples, as in large neural networks), test error can fall again beyond the point where the training data are fitted exactly, a pattern called double descent. It does not change how performance is measured.</p>`);

  // ---- Step 3: bias and variance across fresh cohorts
  const G = T.grid, W = T.densAB, R = T.reps;
  const bv = DEG.map((d) => {
    const m = G.map((_, g) => mean(R.map((r) => r[d][g])));
    const b2 = sum(G.map((_, g) => W[g] * (m[g] - T.truthAB[g]) ** 2)), va = sum(G.map((_, g) => W[g] * mean(R.map((r) => (r[d][g] - m[g]) ** 2))));
    return { d, m, b2, va, tot: b2 + va };
  });
  const dBV = bv.reduce((a, c) => (c.tot < a.tot ? c : a)).d;
  rb = rung(ch, 3, 'Bias and variance',
    `<p>The data for this step were simulated: ${R.length} new discovery cohorts of ${n} patients from the same two centres, each fitted with the same model. Because the true curve is known, the error of the fitted probabilities can be split into two parts:</p>
     <ul><li><b>Bias²</b>: how far the <i>average</i> fitted curve (over many cohorts) is from the truth. It reflects what the model cannot represent.</li>
     <li><b>Variance</b>: how much the fitted curve changes from one cohort to the next. It reflects how much the model adapts to its particular sample.</li></ul>
     <p>Both are averaged over the IFN-γ values that occur in the population. A third component, the irreducible noise in the outcome itself, is the same for every model and is not shown.</p>`);
  const fb = figure(rb, 'The same model fitted to many cohorts',
    'Left: thin lines are fits to individual simulated cohorts, the solid line is their average and the dashed line the true curve. Right: bias², variance and their sum at every degree, on the probability scale (mean squared difference from the true probability), with a logarithmic axis so that small values remain visible.');
  let bDeg = 1, shown = R.length, stopB = null;
  slider(fb.controls, { label: 'polynomial degree', min: 0, max: DMAX, step: 1, value: bDeg, onInput: (v) => { bDeg = v; paintB(); } });
  button(fb.controls, 'Fit cohorts one by one', () => {
    if (stopB) stopB(); shown = 0; let last = 0;
    stopB = animate((dt, t) => { if (t - last < 70) return true; last = t; shown++; paintB(); return shown < R.length; });
  }, true);
  const sB = stats(fb.root);
  const rowB = el('div', { class: 'fig-row' }); fb.body.append(rowB);
  const ba = el('div'), bb = el('div'); rowB.append(ba, bb);
  const FB = frame(ba, { width: 480, height: 300, x: [2.5, 10], y: [0, 1], xLabel: 'IFN-γ score', yLabel: 'fitted P(response)' });
  const lineB = d3.line().x((q) => FB.x(q[0])).y((q) => FB.y(q[1]));
  const repsL = FB.layer('reps'), meanL = FB.layer('mean');
  FB.layer('truth').append('path').attr('d', lineB(G.map((g, i) => [g, T.truthAB[i]]))).attr('fill', 'none').attr('stroke', ink()).attr('stroke-width', 2).attr('stroke-dasharray', '6 4');
  const bvMin = Math.max(1e-5, d3.min(bv, (b) => Math.min(b.b2, b.va)) / 1.5), bvMax = d3.max(bv, (b) => b.tot) * 1.5, yb = (v) => FB2.y(Math.max(bvMin, v));
  const FB2 = frame(bb, { width: 460, height: 300, x: [0, DMAX], y: [bvMin, bvMax], yScale: d3.scaleLog(), xLabel: 'polynomial degree', yLabel: 'mean squared error (log scale)' });
  const decs = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1].filter((v) => v >= bvMin && v <= bvMax);
  FB2.g.select('.axis.y').call(d3.axisLeft(FB2.y).tickValues(decs).tickFormat(d3.format('.0e'))); FB2.g.select('.gridline').call(d3.axisLeft(FB2.y).tickValues(decs).tickSize(-FB2.w).tickFormat(''));
  [['b2', cat(3)], ['va', cat(4)], ['tot', ink()]].forEach(([k, c]) => {
    FB2.layer('l').append('path').attr('d', d3.line().x((b) => FB2.x(b.d)).y((b) => yb(b[k]))(bv)).attr('fill', 'none').attr('stroke', c).attr('stroke-width', 2);
    FB2.layer('p').selectAll('circle.' + k).data(bv).join('circle').attr('class', k).attr('cx', (b) => FB2.x(b.d)).attr('cy', (b) => yb(b[k])).attr('r', 3.8).attr('fill', c).attr('stroke', css('--surface'))
      .on('mousemove', (e, b) => tip(e, `degree ${b.d}<br>bias² ${b.b2.toFixed(4)}<br>variance ${b.va.toFixed(4)}<br>sum ${b.tot.toFixed(4)}`)).on('mouseleave', untip);
  });
  const markB = FB2.layer('m');
  legend(fb.body, [['fit to one cohort', cat(2)], ['average fit', cat(6)], ['true curve (dashed)', ink()], ['bias²', cat(3)], ['variance', cat(4)], ['bias² + variance', ink()]]);
  function paintB() {
    const pts = (arr) => G.map((g, i) => [g, arr[i]]);
    repsL.selectAll('path').data(R.slice(0, shown)).join('path').attr('d', (r) => lineB(pts(r[bDeg]))).attr('fill', 'none').attr('stroke', cat(2)).attr('stroke-width', 1).attr('stroke-opacity', 0.35);
    meanL.selectAll('path').data(shown > 1 ? [bv[bDeg].m] : []).join('path').attr('d', (m) => lineB(pts(m))).attr('fill', 'none').attr('stroke', cat(6)).attr('stroke-width', 2.6);
    markB.selectAll('line').data([bDeg]).join('line').attr('x1', (d) => FB2.x(d)).attr('x2', (d) => FB2.x(d)).attr('y1', 0).attr('y2', FB2.h).attr('stroke', muted()).attr('stroke-dasharray', '3 3');
    const b = bv[bDeg];
    sB.set({ 'cohorts shown': `${shown} of ${R.length}`, 'bias²': b.b2.toFixed(4), variance: b.va.toFixed(4), 'bias² + variance': b.tot.toFixed(4) });
  }
  paintB();
  prose(rb, `<p>At degree 0, bias² is ${bv[0].b2.toFixed(4)} and variance ${bv[0].va.toFixed(4)}: every cohort gives nearly the same flat line, and it is wrong. At degree 1, bias² drops to ${bv[1].b2.toFixed(4)}; higher degrees reduce it further only slightly (${bv[DMAX].b2.toFixed(4)} at degree ${DMAX}), because the true curve has a mild shoulder that a single S-curve cannot follow. Meanwhile variance rises, from ${bv[1].va.toFixed(4)} at degree 1 to ${bv[DMAX].va.toFixed(4)} at degree ${DMAX}; the lines fan out most at the ends of the IFN-γ range, where there are few patients. The sum is lowest at degree ${dBV}. This is the <b>bias–variance trade-off</b>: added flexibility removes bias only while there is bias to remove, and its cost in variance continues to grow.</p>`);
  quiz(rb, 'In the real study only one cohort is available. Which of the two error components can be estimated from it, and how?',
    '<p>Neither directly: bias needs the truth, and variance needs many cohorts. A held-out set or cross-validation estimates their combined effect (the test error), which is why model complexity is chosen by held-out performance. Bootstrap resampling of the one cohort gives an approximate view of the variance.</p>');

  // ---- Step 4: complexity lab in two dimensions
  rb = rung(ch, 4, 'Lab: tune a flexible model and check it on Centre C',
    `<p>A logistic regression on polynomial terms of IFN-γ and log2 TMB: degree <i>d</i> includes every product IFN-γ<sup>i</sup> × TMB<sup>j</sup> with i + j ≤ d, so the number of parameters grows quickly (${nPar2(1)} at degree 1, ${nPar2(4)} at degree 4, ${nPar2(7)} at degree 7) while the number of responders stays at ${nR}. An <b>L2 penalty</b> λ shrinks the polynomial weights towards zero and so limits flexibility without removing terms (see Regularisation).</p>`);
  const fl = figure(rb, 'Complexity lab',
    'Left: shading is the fitted probability of response; the solid line is the p = 0.5 boundary. Dots are the discovery patients used for fitting, or the Centre C patients if selected. Right: training and Centre C log-loss at every degree for the current λ; the ring marks the selected degree.');
  let lDeg = 1, lL2 = 0.001, showV = false;
  slider(fl.controls, { label: 'degree', min: 1, max: 7, step: 1, value: lDeg, onInput: (v) => { lDeg = v; paintL(); } });
  select(fl.controls, { label: 'L2 penalty λ', value: String(lL2), options: L2S.map((v) => [String(v), String(v)]), onChange: (v) => { lL2 = +v; paintL(); } });
  checkbox(fl.controls, { label: 'show Centre C patients', checked: showV, onChange: (c) => { showV = c; paintL(); } });
  const sL = stats(fl.root);
  const rowL = el('div', { class: 'fig-row' }); fl.body.append(rowL);
  const la = el('div'), lb = el('div'); rowL.append(la, lb);
  const FL = frame(la, { width: 480, height: 340, x: [2.2, 10.3], y: [0.6, 5.5], xLabel: 'IFN-γ score', yLabel: 'log2 TMB (mut/Mb)', grid: false });
  const surfL = decisionCanvas(FL, () => 0.5, { res: 5, alpha: 0.4 });
  FL.g.append('clipPath').attr('id', 'tt-lab-clip').append('rect').attr('width', FL.w).attr('height', FL.h);
  const bndL = FL.layer('bnd').attr('clip-path', 'url(#tt-lab-clip)'), ptsL = FL.layer('pts');
  const FL2 = frame(lb, { width: 460, height: 340, x: [1, 7], y: [0.25, 1.0], xLabel: 'polynomial degree', yLabel: 'log-loss (lower is better)' });
  const curL = FL2.layer('c'), ringL = FL2.layer('r');
  legend(fl.body, [['no response', cat(0)], ['responded', cat(1)], ['p = 0.5 boundary', ink()], ['training log-loss (dashed)', ink2()], ['Centre C log-loss', cat(6)]]);
  const NX = 82, NY = 50, gx = d3.range(NX).map((i) => 2.2 + (8.1 * i) / (NX - 1)), gy = d3.range(NY).map((j) => 0.6 + (4.9 * j) / (NY - 1));
  const contourPath = d3.geoPath(d3.geoTransform({ point(px, py) { this.stream.point(FL.x(2.2 + (8.1 * (px - 0.5)) / (NX - 1)), FL.y(0.6 + (4.9 * (py - 0.5)) / (NY - 1))); } }));
  function paintL() {
    const L = lab(lDeg, lL2), f = L.fit;
    surfL.draw((a, b) => pr(f, poly2(a, b, lDeg)));
    const vals = []; gy.forEach((b) => gx.forEach((a) => vals.push(pr(f, poly2(a, b, lDeg)))));
    bndL.selectAll('path').data(d3.contours().size([NX, NY]).thresholds([0.5])(vals)).join('path').attr('d', contourPath).attr('fill', 'none').attr('stroke', ink()).attr('stroke-width', 2.2);
    const P = showV ? V : D, pp = showV ? L.pv : L.pt;
    ptsL.selectAll('circle').data(P, (d) => d.id).join('circle').attr('cx', (d) => FL.x(d.ifng)).attr('cy', (d) => FL.y(d.log2tmb)).attr('r', 4).attr('fill', (d) => cat(d.response)).attr('stroke', css('--surface')).attr('stroke-width', 1)
      .on('mousemove', (e, d) => tip(e, `${d.id} (${d.centre})<br>predicted P = ${f2(pp[P.indexOf(d)])}<br>${d.response ? 'responded' : 'no response'}`)).on('mouseleave', untip);
    const curve = d3.range(1, 8).map((d) => ({ d, ...lab(d, lL2) }));
    const top = 1.0, yy = (v) => FL2.y(Math.min(top, v));
    curL.selectAll('path').data([['trLL', ink2(), '6 4'], ['vaLL', cat(6), null]]).join('path').attr('d', ([k]) => d3.line().x((c) => FL2.x(c.d)).y((c) => yy(c[k]))(curve)).attr('fill', 'none').attr('stroke', (d) => d[1]).attr('stroke-width', 2).attr('stroke-dasharray', (d) => d[2]);
    curL.selectAll('circle').data(curve.flatMap((c) => [['trLL', c, ink2()], ['vaLL', c, cat(6)]])).join('circle').attr('cx', (q) => FL2.x(q[1].d)).attr('cy', (q) => yy(q[1][q[0]])).attr('r', 4).attr('fill', (q) => q[2]).attr('stroke', css('--surface'))
      .on('mousemove', (e, q) => tip(e, `degree ${q[1].d} (${nPar2(q[1].d)} parameters), λ = ${lL2}<br>training log-loss ${f3(q[1].trLL)}<br>Centre C log-loss ${f3(q[1].vaLL)}${q[1].vaLL > top ? ' (above the axis)' : ''}`)).on('mouseleave', untip);
    ringL.selectAll('circle').data([L.trLL, L.vaLL]).join('circle').attr('cx', FL2.x(lDeg)).attr('cy', (v) => yy(v)).attr('r', 8).attr('fill', 'none').attr('stroke', ink()).attr('stroke-width', 1.5);
    sL.set({ parameters: nPar2(lDeg), 'responders per parameter': (nR / nPar2(lDeg)).toFixed(1), 'training AUC': f3(L.trAUC), 'Centre C AUC': f3(L.vaAUC), 'training log-loss': f3(L.trLL), 'Centre C log-loss': f3(L.vaLL) });
  }
  paintL();
  const a1 = lab(1, 0.001), a7 = lab(7, 0.001), r7 = lab(7, 10), a2 = lab(2, 0.001), a4 = lab(4, 0.001);
  const lamBest = L2S.reduce((a, l) => (lab(7, l).vaLL < lab(7, a).vaLL ? l : a));
  table(rb, [
    { t: 'Degree <b>1 → 7</b>, λ = 0.001', s: `Training log-loss ${f3(a1.trLL)} → ${f3(a7.trLL)}; Centre C ${f3(a1.vaLL)} → <b>${f3(a7.vaLL)}</b>. AUC ${f3(a1.vaAUC)} → ${f3(a7.vaAUC)} on Centre C`, m: 'Each added term lowers training error. On new patients the extra flexibility costs more than it gains: overfitting.' },
    { t: 'Degree 7, tick <b>show Centre C patients</b>', s: 'The boundary curls around small groups of discovery patients; the Centre C patients do not follow those curls', m: 'The curls describe noise in the training sample, not the biology. A flexible boundary is only as reliable as the data near it.' },
    { t: 'Degree 7, λ from <b>0.001 to 10</b>', s: `Centre C log-loss ${f3(a7.vaLL)} → <b>${f3(r7.vaLL)}</b> (lowest at λ = ${lamBest}); training log-loss rises to ${f3(r7.trLL)}`, m: 'A penalty reduces variance without changing the set of terms. Penalty strength is another complexity setting, chosen on held-out data.' },
    { t: 'Degree <b>1 vs 2</b>', s: `Centre C log-loss ${f3(a1.vaLL)} vs ${f3(a2.vaLL)}`, m: 'Little or nothing is gained beyond a linear boundary. In the simulation the true log-odds are linear in IFN-γ, log2 TMB and the CAF score (not in this model).' },
    { t: 'Read <b>responders per parameter</b>', s: `${(nR / nPar2(1)).toFixed(1)} at degree 1, ${(nR / nPar2(4)).toFixed(1)} at degree 4 (training AUC ${f3(a4.trAUC)}, Centre C ${f3(a4.vaAUC)})`, m: 'The information is limited by the smaller outcome class. A common rule of thumb asks for at least 10 events per parameter in an unpenalised model.' },
  ], [['t', 'Try'], ['s', 'What you see'], ['m', 'What it means']], { cls: 'compact' });

  // ---- Step 5: learning curves
  rb = rung(ch, 5, 'Learning curves: how much data a model needs',
    `<p>A <b>learning curve</b> plots training and test performance against the number of training patients. The simulation can provide more patients than the real study: each point averages 30 random training sets drawn from a large simulated Centre A + B population, and every model is scored on the real ${nV} Centre C patients.</p>`);
  const fe = figure(rb, 'Learning curves for a simple and a flexible model',
    'Dashed line: performance on the training patients; solid line: on Centre C. Bands span the 10th to 90th percentile of the 30 training sets. The vertical line marks the size of the real discovery cohort; the dotted line is the Centre C value for a model fitted to 20,000 simulated patients.');
  let eMod = '6', eMet = 'll', grown = T.ns.length;
  select(fe.controls, { label: 'Model', value: eMod, options: [['1', `Degree 1 (${nPar2(1)} parameters)`], ['6', `Degree 6 (${nPar2(6)} parameters)`]], onChange: (v) => { eMod = v; replay(); } });
  select(fe.controls, { label: 'Metric', value: eMet, options: [['ll', 'log-loss (lower is better)'], ['auc', 'AUC (higher is better)']], onChange: (v) => { eMet = v; paintE(); } });
  button(fe.controls, 'Replay: add patients', () => replay(), true);
  const sE = stats(fe.root);
  const FE = frame(fe.body, { width: 820, height: 320, xScale: d3.scaleLog(), x: [28, 1500], y: [0, 1], xLabel: 'training patients (log scale)', yLabel: 'performance' });
  FE.g.select('.axis.x').call(d3.axisBottom(FE.x).tickValues(T.ns).tickFormat(d3.format('d')));
  FE.layer('n').append('line').attr('x1', FE.x(n)).attr('x2', FE.x(n)).attr('y1', 0).attr('y2', FE.h).attr('stroke', muted()).attr('stroke-dasharray', '3 3');
  FE.layer('n').append('text').attr('x', FE.x(n) + 5).attr('y', 12).attr('font-size', 11.5).attr('fill', ink2()).text(`this study: ${n}`);
  const bandE = FE.layer('band'), lineE = FE.layer('line'), dotE = FE.layer('dot'), limE = FE.layer('lim');
  legend(fe.body, [['training patients (dashed)', ink2()], ['Centre C', cat(6)], ['large-sample limit (dotted)', ink()]]);
  let stopE = null;
  function replay() { if (stopE) stopE(); grown = 0; let last = 0; stopE = animate((dt, t) => { if (t - last < 110) return true; last = t; grown++; paintE(); return grown < T.ns.length; }); }
  function paintE() {
    const Lc = T.learning[eMod], cv = Lc.curve.slice(0, grown), tk = eMet === 'll' ? 'tr_ll' : 'tr_auc', vk = eMet === 'll' ? 'va_ll' : 'va_auc';
    const dom = eMet === 'll' ? [0, Math.min(2.4, d3.max(Lc.curve, (c) => c.va_ll[2]) * 1.05)] : [0.5, 1.0];
    FE.y.domain(dom); FE.g.select('.axis.y').call(d3.axisLeft(FE.y).ticks(6)); FE.g.select('.gridline').call(d3.axisLeft(FE.y).ticks(6).tickSize(-FE.w).tickFormat(''));
    FE.g.select('.ax-label[transform]').text(eMet === 'll' ? 'log-loss (lower is better)' : 'AUC (higher is better)');
    const cl = (v) => FE.y(clamp(v, dom[0], dom[1]));
    bandE.selectAll('path').data([[tk, ink2()], [vk, cat(6)]]).join('path').attr('fill', (d) => d[1]).attr('fill-opacity', 0.14).attr('d', ([k]) => d3.area().x((c) => FE.x(c.n)).y0((c) => cl(c[k][1])).y1((c) => cl(c[k][2]))(cv));
    lineE.selectAll('path').data([[tk, ink2(), '6 4'], [vk, cat(6), null]]).join('path').attr('fill', 'none').attr('stroke', (d) => d[1]).attr('stroke-width', 2.2).attr('stroke-dasharray', (d) => d[2]).attr('d', ([k]) => d3.line().x((c) => FE.x(c.n)).y((c) => cl(c[k][0]))(cv));
    dotE.selectAll('circle').data(cv.flatMap((c) => [[tk, c, ink2()], [vk, c, cat(6)]])).join('circle').attr('cx', (q) => FE.x(q[1].n)).attr('cy', (q) => cl(q[1][q[0]][0])).attr('r', 4).attr('fill', (q) => q[2]).attr('stroke', css('--surface'))
      .on('mousemove', (e, q) => tip(e, `${q[1].n} training patients<br>training: ${f3(q[1][tk][0])}<br>Centre C: ${f3(q[1][vk][0])} (${f3(q[1][vk][1])}–${f3(q[1][vk][2])})`)).on('mouseleave', untip);
    const lim = Lc.limit[vk];
    limE.selectAll('line').data([lim]).join('line').attr('x1', 0).attr('x2', FE.w).attr('y1', cl(lim)).attr('y2', cl(lim)).attr('stroke', ink()).attr('stroke-dasharray', '2 4').attr('stroke-width', 1.4);
    const at = Lc.curve.find((c) => c.n === n), last = cv[cv.length - 1];
    sE.set({ 'at n = 180: training': f3(at[tk][0]), 'at n = 180: Centre C': f3(at[vk][0]), [`at n = ${last ? last.n : '…'}: Centre C`]: last ? f3(last[vk][0]) : '…', 'large-sample limit': f3(lim) });
  }
  paintE();
  const LC1 = T.learning['1'], LC6 = T.learning['6'], at = (Lc, nn) => Lc.curve.find((c) => c.n === nn), nMax = T.ns[T.ns.length - 1];
  const close6 = LC6.curve.find((c) => c.va_ll[0] - LC6.limit.va_ll < 0.03);
  prose(rb, `<p>For the degree-1 model (${nPar2(1)} parameters), Centre C log-loss is ${f3(at(LC1, n).va_ll[0])} at ${n} patients and ${f3(at(LC1, nMax).va_ll[0])} at ${nMax}: it is already close to its limit, and more patients would change little. The degree-6 model (${nPar2(6)} parameters) is far from its limit at ${n} patients (training ${f3(at(LC6, n).tr_ll[0])}, Centre C ${f3(at(LC6, n).va_ll[0])})${close6 ? ` and comes within 0.03 of it only at about ${close6.n} patients` : ''}. A large gap between the curves indicates <b>variance</b>, which more data reduces; curves that meet at a poor value indicate <b>bias</b>, which only a better model or better features can reduce.</p>
    <p>For the degree-1 model the Centre C curve lies <i>below</i> the training curve at large sample sizes (${f3(at(LC1, nMax).va_ll[0])} vs ${f3(at(LC1, nMax).tr_ll[0])}). This is not an error: Centre C is a different population (step 6), with more immune-desert tumours, which are easy to classify as non-responders. Its patients are therefore on average easier to classify from these two features than patients from Centres A and B.</p>`);

  // ---- Step 6: Centre C as external test
  const { train: itr, test: ite } = stratifiedSplit(y, 0.3, 7);
  const sub = (idx, pts) => idx.map((i) => pts[i]);
  const iTrD = sub(itr, D), iTeD = sub(ite, D), iy = itr.map((i) => y[i]), iyh = ite.map((i) => y[i]);
  const cfgs = [];
  [1, 2, 3, 4, 5, 6, 7].forEach((d) => [0.001, 10].forEach((l2) => {
    const f = fitLR(iTrD.map((p) => poly2(p.ifng, p.log2tmb, d)), iy, l2);
    cfgs.push({ d, l2, ho: auc(iTeD.map((p) => pr(f, poly2(p.ifng, p.log2tmb, d))), iyh), vc: lab(d, l2).vaAUC });
  }));
  const byHo = cfgs.reduce((a, c) => (c.ho > a.ho ? c : a)), byC = cfgs.reduce((a, c) => (c.vc > a.vc ? c : a));
  const mix = ['Centre A', 'Centre B', 'Centre C'].map((c) => { const P = C.all.filter((p) => p.centre === c); return { c, n: P.length, ...Object.fromEntries(['inflamed', 'excluded', 'desert'].map((ph) => [ph, P.filter((p) => p.phenotype === ph).length / P.length])), r: mean(P.map((p) => p.response)) }; });
  const g6 = G.findIndex((g) => Math.abs(g - 6) < 1e-6), maxDiff = d3.max(G, (g, i) => Math.abs(T.truthAB[i] - T.truthC[i]));
  const ABdes = D.filter((p) => p.phenotype === 'desert').length / n;
  const seC = aucSE(M.lr3.vA, sum(yv), nV - sum(yv));
  rb = rung(ch, 6, 'The external test: Centre C',
    `<p>An internal hold-out set comes from the same centres, the same period and the same protocols as the training data. It estimates performance in that setting. An <b>external</b> test set, collected elsewhere, also tests whether the model transfers to a different population. Centre C differs from the discovery centres in its mix of immune phenotypes:</p>`);
  table(rb, mix, [['c', 'Centre'], ['n', 'Patients'], ['inflamed', 'Inflamed', pc], ['excluded', 'Excluded', pc], ['desert', 'Desert', pc], ['r', 'Response rate', pc]], { cls: 'compact' });
  prose(rb, `<p>The true relationship between IFN-γ and response is ${maxDiff < 0.05 ? 'almost the same in both populations' : 'not the same in both populations'} (at IFN-γ = 6, ${f2(T.truthAB[g6])} in Centres A + B and ${f2(T.truthC[g6])} in Centre C; the largest difference over the IFN-γ range is ${f2(maxDiff)}). What differs is the mix of patients: Centre C has ${pc(mix[2].desert)} immune-desert tumours against ${pc(ABdes)} in discovery, and these are among the easiest patients to classify as non-responders. Performance measured in one population does not automatically carry over to another with a different mix. The Centre C estimate is itself uncertain: for the four-parameter logistic model, AUC ${f3(M.lr3.vA)} has a standard error of about ${f3(seC)} with ${nV} patients.</p>
    <p>The test set must be used <b>once</b>. In the lab, 14 configurations (degree 1–7, λ = 0.001 or 10) were fitted on 70% of the discovery cohort and compared on the remaining 30%. The configuration chosen this way (degree ${byHo.d}, λ = ${byHo.l2}) has Centre C AUC ${f3(byHo.vc)} when refitted on all discovery patients. ${byC === byHo ? 'The same configuration also has the best Centre C AUC, so here choosing on the test set would not change the report' : byC.vc - byHo.vc < 0.002 ? `Choosing instead the configuration with the best Centre C AUC (degree ${byC.d}, λ = ${byC.l2}) would report essentially the same value (${f3(byC.vc)}), because most configurations perform similarly` : `Choosing instead the configuration with the best Centre C AUC (degree ${byC.d}, λ = ${byC.l2}) would report ${f3(byC.vc)}, ${f3(byC.vc - byHo.vc)} higher than the configuration chosen on the held-out split`}; with many more candidates, such as thousands of possible gene signatures, the best test score can lie well above what the next cohort will show. Once the test set has been used to choose, its score is no longer an unbiased estimate for the next cohort.</p>`);
  quiz(rb, `Why does Centre C provide stronger evidence than a further random ${nV} patients from Centres A and B would?`,
    '<p>Random patients from the same centres share their referral patterns, sample handling and phenotype mix with the training data, so they test only whether the model generalises within that setting. Centre C differs in phenotype mix (and, in real studies, often in assay batch and patient selection), so good performance there also shows that the model transfers. Poor performance there would show that it does not, which an internal split cannot reveal.</p>');

  // =====================================================================================
  // 5. PITFALLS
  // =====================================================================================
  ch = chapter(main, 'pitfalls', 'Pitfalls', 'Common problems, how to recognise them, and how to address them.');
  const h10 = runSplits('lr3', 0.1).map((r) => r.ho);
  cards(ch, [
    { tag: 'Symptom: excellent performance, reported on the training data', title: 'No held-out evaluation', body: `Training performance mainly reflects flexibility: the 1-nearest-neighbour rule reaches ${pc(M.nn1.tC)} training accuracy. <b>Fix:</b> report performance on data not used for fitting, and say which data those were.` },
    { tag: 'Symptom: the test score drops in the next cohort', title: 'Choosing on the test set', body: 'Every comparison made on the test set (features, degree, penalty, threshold) fits a little of its noise. <b>Fix:</b> make all choices on a validation set or by cross-validation inside the training data; score the test set once.' },
    { tag: 'Symptom: high held-out score, poor external score', title: 'Leakage through preprocessing', body: 'Gene selection, normalisation, batch correction or imputation computed on all samples before splitting carries information from the test set into training. <b>Fix:</b> fit every data-driven step on the training part only, inside the split (see Cross-validation).' },
    { tag: 'Symptom: near-perfect held-out accuracy', title: 'Related samples on both sides', body: 'Several biopsies, sequencing runs or cells from the same patient split at random let the model recognise the patient rather than predict the outcome. <b>Fix:</b> split by patient (grouped split), and by site or time where relevant.' },
    { tag: 'Symptom: results change with the random seed', title: 'Small test set', body: `With ${Math.round(0.1 * n)} held-out patients, held-out AUC for the same model ranged from ${f2(d3.quantile(h10.sort((a, b) => a - b), 0.1))} to ${f2(d3.quantile(h10, 0.9))} across splits. <b>Fix:</b> use cross-validation or repeated splits, and report a confidence interval.` },
    { tag: 'Symptom: a flexible model fits the training data perfectly', title: 'Too many parameters for the events', body: `Degree 7 has ${nPar2(7)} parameters for ${nR} responders. <b>Fix:</b> prefer simpler models, penalise, and choose complexity on held-out data.` },
    { tag: 'Symptom: different response rates in train and test', title: 'Unstratified split', body: `With only ${pc(base)} responders, a random split can put noticeably more responders in one part. <b>Fix:</b> stratify the split by outcome (and by centre or batch if needed).` },
    { tag: 'Symptom: good discrimination, wrong probabilities', title: 'Population shift', body: `Centre C has a different phenotype mix from discovery (${pc(mix[2].desert)} vs ${pc(ABdes)} immune deserts) and a response rate of ${pc(mix[2].r)} vs ${pc(base)}; real external cohorts also differ in assay batch and patient selection. Rankings can transfer while absolute probabilities do not. <b>Fix:</b> check calibration in each new setting, and recalibrate if needed.` },
  ]);

  // =====================================================================================
  // 6. RECAP
  // =====================================================================================
  ch = chapter(main, 'recap', 'Recap', 'Key points.');
  recap(ch, [
    'Predictive performance is measured on <b>data not used for fitting</b>. Training performance is optimistic, and more so for flexible models.',
    'Use three roles: a <b>training set</b> to fit, a <b>validation set</b> (or cross-validation) to choose, and a <b>test set</b> used once to report.',
    '<b>Underfitting</b>: too simple; training and test error both high. <b>Overfitting</b>: too flexible; training error low, test error higher.',
    'Test error combines <b>bias²</b> (systematic misfit), <b>variance</b> (sensitivity to the training sample) and irreducible noise. Flexibility trades the first for the second.',
    'Complexity settings (degree, k, depth, penalty λ, number of features) are chosen by held-out performance, never by training performance.',
    '<b>Learning curves</b> show whether more data would help: a large train–test gap indicates variance; curves meeting at a poor value indicate bias.',
    'A single small hold-out set is noisy; split by patient (not by sample), stratify by outcome, and keep every preprocessing step inside the split.',
    'An <b>external</b> cohort such as Centre C tests whether the model transfers to a different population, which an internal split cannot show.',
  ]);
  prose(ch, '<h3>In code</h3><p>A stratified hold-out split, a validation curve over model complexity, and a learning curve.</p>');
  code(ch, {
    Python: `import numpy as np
from sklearn.model_selection import train_test_split, validation_curve, learning_curve, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss

X, y = disc[["ifng", "log2tmb"]], disc["response"]
X_tr, X_ho, y_tr, y_ho = train_test_split(X, y, test_size=0.3, stratify=y, random_state=7)

def model(degree=1, C=1000):   # C = 1 / lambda
    return make_pipeline(PolynomialFeatures(degree, include_bias=False), StandardScaler(),
                         LogisticRegression(C=C, max_iter=5000))

# Choose complexity by cross-validation inside the training part
cv = StratifiedKFold(5, shuffle=True, random_state=1)
tr, va = validation_curve(model(), X_tr, y_tr, param_name="polynomialfeatures__degree",
                          param_range=range(1, 8), cv=cv, scoring="neg_log_loss")
best = 1 + int(np.argmax(va.mean(axis=1)))

print("hold-out log-loss", log_loss(y_ho, model(best).fit(X_tr, y_tr).predict_proba(X_ho)[:, 1]))

# Learning curve: training size vs performance
sizes, tr_s, va_s = learning_curve(model(best), X_tr, y_tr, cv=cv, scoring="roc_auc",
                                   train_sizes=np.linspace(0.2, 1.0, 6))

# Refit on all discovery patients; score the external cohort once
final = model(best).fit(X, y)
p = final.predict_proba(centre_c[["ifng", "log2tmb"]])[:, 1]
print("Centre C AUC", roc_auc_score(centre_c["response"], p), "log-loss", log_loss(centre_c["response"], p))`,
    R: `library(rsample)   # splitting
set.seed(7)
split <- initial_split(disc, prop = 0.7, strata = response)
train <- training(split); hold <- testing(split)

# Complexity curve: polynomial degree vs hold-out log-loss
logloss <- function(p, y) -mean(y * log(p) + (1 - y) * log(1 - p))
res <- sapply(1:7, function(d) {
  m <- glm(response ~ poly(ifng, log2tmb, degree = d), family = binomial, data = train)
  c(train = logloss(fitted(m), train$response),
    hold  = logloss(predict(m, hold, type = "response"), hold$response))
})
best <- which.min(res["hold", ])

# Refit on all discovery patients; score the external cohort once
final <- glm(response ~ poly(ifng, log2tmb, degree = best), family = binomial, data = disc)
p <- predict(final, centre_c, type = "response")
pROC::auc(centre_c$response, p)
# Grouped data (several samples per patient): rsample::group_initial_split(data, group = patient_id)`,
  });
  seeAlso(main, [['concepts', 'what-is-ml', 'What is machine learning?'], ['concepts', 'cross-validation', 'Cross-validation and data leakage'], ['concepts', 'regularization', 'Regularisation'], ['concepts', 'logistic-regression', 'Logistic regression'], ['concepts', 'metrics', 'Evaluating a classifier'], ['omics', 'bulk-classifier', 'Biomarker classifier and nested CV']]);
}

// ---------------------------------------------------------------------------------------------
function qBins(pts, key, k) {
  const s = pts.slice().sort((a, b) => key(a) - key(b)), out = [];
  for (let i = 0; i < k; i++) {
    const part = s.slice(Math.round(i * s.length / k), Math.round((i + 1) * s.length / k));
    out.push({ x: mean(part.map(key)), lo: key(part[0]), hi: key(part[part.length - 1]), n: part.length, r: part.filter((p) => p.response).length });
  }
  return out;
}

/** Diagram: discovery cohort → training / hold-out; Centre C → external test. */
function splitSvg(n, nV, nR, nRV) {
  const wrap = el('div', { class: 'figure' });
  wrap.append(el('div', { class: 'fig-title' }, 'Three roles for the data in this study'));
  const svg = d3.select(wrap).append('svg').attr('viewBox', '0 0 900 220').attr('width', 900).attr('role', 'img').attr('aria-label', 'discovery cohort split into training and hold-out; Centre C as external test');
  const box = (x, y, w, h, title, lines, accent) => {
    const g = svg.append('g').attr('transform', `translate(${x},${y})`);
    g.append('rect').attr('width', w).attr('height', h).attr('rx', 10).attr('fill', css('--surface-2')).attr('stroke', accent || css('--axis')).attr('stroke-width', accent ? 2 : 1);
    g.append('text').attr('x', 12).attr('y', 24).attr('font-weight', 700).attr('font-size', 13.5).attr('fill', ink()).text(title);
    lines.forEach((l, j) => g.append('text').attr('x', 12).attr('y', 48 + j * 20).attr('font-size', 12.5).attr('fill', ink2()).text(l));
  };
  const arrow = (x1, y1, x2, y2) => {
    const a = Math.atan2(y2 - y1, x2 - x1), h = 8;
    svg.append('path').attr('d', `M${x1},${y1} L${x2},${y2} M${x2 - h * Math.cos(a - 0.45)},${y2 - h * Math.sin(a - 0.45)} L${x2},${y2} L${x2 - h * Math.cos(a + 0.45)},${y2 - h * Math.sin(a + 0.45)}`).attr('fill', 'none').attr('stroke', ink2()).attr('stroke-width', 1.8);
  };
  box(10, 50, 230, 116, 'Discovery cohort', ['Centres A and B', `${n} patients, ${nR} responders`, 'all model building happens here']);
  box(300, 10, 260, 92, 'Training set', ['fit the parameters', '(e.g. 70% of discovery)']);
  box(300, 118, 260, 92, 'Validation (hold-out) set', ['compare models, choose settings', '(e.g. 30%, or cross-validation)']);
  box(630, 50, 260, 116, 'Test set: Centre C', [`${nV} patients, ${nRV} responders`, 'external hospital', 'scored once, after all choices'], cat(6));
  arrow(240, 95, 296, 60); arrow(240, 121, 296, 160);
  arrow(562, 108, 626, 108);
  svg.append('text').attr('x', 594).attr('y', 98).attr('text-anchor', 'middle').attr('font-size', 11).attr('fill', muted()).text('final model');
  return wrap;
}
