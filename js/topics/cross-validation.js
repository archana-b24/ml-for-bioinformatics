// Cross-validation and data leakage, lesson format.
// Arc: what → why → where → how it works (6 steps: k-fold → stratification and variance → grouped CV
//      → leakage with permuted labels → nested CV → pipeline lab with try/see/means) → pitfalls → recap.
// Every number in the prose is computed at render time from data/study/cohort.json plus seeded noise features.
import { header, figure, slider, select, button, checkbox, stats, legend, seeAlso, animate, every, tip, untip, el, html } from '../lib/ui.js';
import { frame, cat, css, ink, ink2, muted } from '../lib/plot.js';
import { rng, mean, sd } from '../lib/data.js';
import { fitLogistic, predictProb, auc, stratifiedSplit } from '../lib/models.js';
import { loadCohort, FEATURES, studyCard } from '../lib/study.js';
import { lessonNav, chapter, prose, rung, worked, quiz, cards, table, code, recap } from '../lib/lesson.js';
const d3 = window.d3;

const um = (t) => t.replace(/^-/, '−');
const f2 = (v) => (isFinite(v) ? um(v.toFixed(2)) : '—');
const f3 = (v) => (isFinite(v) ? um(v.toFixed(3)) : '—');
const pc = (v) => Math.round(100 * v) + '%';
const range = (n) => Array.from({ length: n }, (_, i) => i);
const sum = (a) => a.reduce((s, v) => s + v, 0);
const okMean = (a) => { const b = a.filter(isFinite); return b.length ? mean(b) : NaN; };
const span = (a, b) => (a === b ? String(a) : `${a}–${b}`);
/** Hanley–McNeil approximate standard error of an AUC with n1 positives and n0 negatives. */
const aucSE = (A, n1, n0) => { const q1 = A / (2 - A), q2 = 2 * A * A / (1 + A); return Math.sqrt((A * (1 - A) + (n1 - 1) * (q1 - A * A) + (n0 - 1) * (q2 - A * A)) / (n1 * n0)); };
const q = (a, p) => d3.quantile(a.slice().sort((x, z) => x - z), p);
const TRAIN = () => cat(0), TEST = () => cat(1), VAL = () => cat(2);
const BAD = () => cat(3), GOOD = () => cat(6), PLAIN = () => cat(4);
const LAB = ['ifng', 'log2tmb', 'caf', 'pdl1', 'cd8', 'prolif', 'age', 'smoker', 'male', 'centreB'];
const MG = [1, 2, 3, 5, 10, 20, 50, 100];

/** Fold id (0..k-1) for each of n items. y: deal each class round-robin (stratified); groups: whole groups per fold. */
function makeFolds(n, k, seed, { y = null, groups = null } = {}) {
  const r = rng(seed), f = new Array(n);
  if (groups) { const gf = new Map(r.shuffle([...new Set(groups)]).map((g, i) => [g, i % k])); groups.forEach((g, i) => { f[i] = gf.get(g); }); }
  else if (y) { let pos = r.int(k); [0, 1].forEach((c) => r.shuffle(range(n).filter((i) => +y[i] === c)).forEach((i) => { f[i] = pos++ % k; })); }
  else r.shuffle(range(n)).forEach((i, j) => { f[i] = j % k; });
  return f;
}
/** k-fold CV of an unpenalised logistic regression on rows X. Returns per-fold AUC, their mean and the out-of-fold predictions. */
function cvLogit(X, y, fold, k) {
  const per = [], oof = new Array(y.length);
  for (let f = 0; f < k; f++) {
    const tr = [], te = []; fold.forEach((g, i) => (g === f ? te : tr).push(i));
    const fit = fitLogistic(tr.map((i) => X[i]), tr.map((i) => y[i]));
    const p = te.map((i) => predictProb(X[i], fit.b0, fit.w)); te.forEach((i, j) => { oof[i] = p[j]; });
    per.push({ auc: auc(p, te.map((i) => y[i])), n: te.length, pos: te.filter((i) => y[i]).length });
  }
  return { per, est: okMean(per.map((d) => d.auc)), undef: per.filter((d) => !isFinite(d.auc)).length, oof };
}
/** Per-feature t-statistic (pooled SD), mean and SD over patients idx. F is feature-major. */
function featStats(F, idx, lab) {
  const n1 = idx.reduce((s, i) => s + lab[i], 0), n0 = idx.length - n1, n = idx.length;
  return F.map((col) => {
    let s0 = 0, s1 = 0, q0 = 0, q1 = 0;
    for (const i of idx) { const v = col[i]; if (lab[i]) { s1 += v; q1 += v * v; } else { s0 += v; q0 += v * v; } }
    const m0 = s0 / n0, m1 = s1 / n1, sp = Math.sqrt((q0 - n0 * m0 * m0 + q1 - n1 * m1 * m1) / (n - 2)) || 1;
    const mu = (s0 + s1) / n;
    return { t: (m1 - m0) / (sp * Math.sqrt(1 / n0 + 1 / n1)), mu, sd: Math.sqrt(Math.max(1e-12, (q0 + q1 - n * mu * mu) / (n - 1))) };
  });
}
const rankBy = (st) => range(st.length).sort((a, b) => Math.abs(st[b].t) - Math.abs(st[a].t));
/** Weighted-signature (diagonal discriminant) scores for patients te, for every m in grid: Σ over the top-m features of t × z. */
function sigScores(F, st, norm, ord, grid, te) {
  const acc = new Float64Array(te.length), out = []; let gi = 0;
  const top = Math.min(ord.length, grid[grid.length - 1]);
  for (let r = 0; r < top; r++) {
    const j = ord[r], col = F[j], w = st[j].t, mu = norm[j].mu, s = norm[j].sd;
    for (let a = 0; a < te.length; a++) acc[a] += w * (col[te[a]] - mu) / s;
    while (gi < grid.length && (grid[gi] === r + 1 || (r === top - 1 && grid[gi] > top))) { out.push(Array.from(acc)); gi++; }
  }
  return out;
}
/** CV of the signature pipeline over patients idx. leaky: feature ranking from all of idx. scaleAll: z-scores from all of idx. */
function sigCV(F, lab, idx, fold, k, grid, { leaky = false, scaleAll = false, stAll = null } = {}) {
  stAll = stAll || featStats(F, idx, lab);
  const ordAll = rankBy(stAll), per = grid.map(() => []);
  for (let f = 0; f < k; f++) {
    const tr = idx.filter((_, a) => fold[a] !== f), te = idx.filter((_, a) => fold[a] === f);
    const st = featStats(F, tr, lab), S = sigScores(F, st, scaleAll ? stAll : st, leaky ? ordAll : rankBy(st), grid, te);
    S.forEach((sc, g) => per[g].push(auc(sc, te.map((i) => lab[i]))));
  }
  return grid.map((m, g) => ({ m, per: per[g], est: okMean(per[g]) }));
}
/** Nested CV: inner stratified 4-fold CV on each outer-training set chooses m; the outer fold scores that choice once. */
function sigNested(F, lab, idx, fold, k, grid, opts = {}) {
  const stAll = featStats(F, idx, lab), ordAll = rankBy(stAll), rounds = [];
  for (let f = 0; f < k; f++) {
    const tr = idx.filter((_, a) => fold[a] !== f), te = idx.filter((_, a) => fold[a] === f);
    const inner = makeFolds(tr.length, 4, 300 + f, { y: tr.map((i) => lab[i]) });
    const ins = sigCV(F, lab, tr, inner, 4, grid, { ...opts, stAll: opts.leaky ? stAll : null });
    const best = ins.reduce((b, r) => (r.est > b.est ? r : b));
    const st = featStats(F, tr, lab);
    const sc = sigScores(F, st, opts.scaleAll ? stAll : st, opts.leaky ? ordAll : rankBy(st), [best.m], te)[0];
    rounds.push({ m: best.m, inner: best.est, auc: auc(sc, te.map((i) => lab[i])), n: te.length });
  }
  return { rounds, est: okMean(rounds.map((r) => r.auc)) };
}

export default async function render(main) {
  const C = await loadCohort();
  const D = C.discovery, V = C.validation, ALL = D.concat(V), nD = D.length;
  const y = D.map((p) => p.response), yv = V.map((p) => p.response), nR = sum(y);
  const F3 = ['ifng', 'log2tmb', 'caf'];
  const rowsOf = (pts, keys) => pts.map((p) => keys.map((k) => FEATURES[k].get(p)));
  const X3 = rowsOf(D, F3), X3v = rowsOf(V, F3);
  const fitAll = fitLogistic(X3, y);
  const aucC = auc(X3v.map((r) => predictProb(r, fitAll.b0, fitAll.w)), yv);
  const DIDX = range(nD), VIDX = range(V.length).map((i) => nD + i);
  // feature-major matrix: 10 cohort measurements, then up to 5,000 random noise features (unrelated to response)
  const COH = LAB.map((k) => Float64Array.from(ALL, (p) => FEATURES[k].get(p)));
  const rn = rng(91), NOISE = range(5000).map(() => Float64Array.from(ALL, () => rn.normal()));
  const Fof = (ng) => COH.concat(NOISE.slice(0, ng));
  const permLab = (s) => { const pl = rng(700 + s).shuffle(y); return pl.concat(yv); };
  const realLab = y.concat(yv);
  const sFold = (lab, seed = 11) => makeFolds(nD, 5, seed, { y: lab.slice(0, nD) });
  const centreC = (F, lab, m) => { const st = featStats(F, DIDX, lab); return auc(sigScores(F, st, st, rankBy(st), [m], VIDX)[0], yv); };

  header(main, 'Cross-validation and data leakage',
    'Cross-validation estimates how well a modelling procedure will predict new samples by <b>repeatedly holding out part of the data, fitting on the rest and scoring on the part held out</b>. The estimate is only valid if every step that learns from the data, including feature selection, scaling and tuning, is repeated inside each training part; otherwise information from the held-out samples <b>leaks</b> into the model.');
  lessonNav(main);
  studyCard(main, C, `<b>Question for this page:</b> with only ${nD} discovery patients, how well will a response model perform on new patients, and how can that estimate be made without wasting data or overstating performance?`);

  // =====================================================================================
  // 1. WHAT
  // =====================================================================================
  let ch = chapter(main, 'what', 'What is cross-validation?', 'In one line: <b>split into k parts → each part is held out once → average the k scores</b>.');
  const fold5 = makeFolds(nD, 5, 3, { y });
  const cv5 = cvLogit(X3, y, fold5, 5);
  prose(ch, `<p><b>Cross-validation (CV)</b> is a resampling method for estimating predictive performance. The patients are divided into <b>k folds</b> of nearly equal size. The model is fitted k times, each time on k − 1 folds, and scored on the remaining fold. Every patient is predicted exactly once, by a model that did not see that patient, and the k scores are averaged into the <b>CV estimate</b>.</p>
    <p>The diagram shows stratified 5-fold CV of a logistic regression on IFN-γ, TMB and CAF score in the ${nD} discovery patients. The CV estimate of the AUC is <b>${f2(cv5.est)}</b>; the same model fitted to all ${nD} patients reaches ${f2(aucC)} on the ${V.length} Centre C patients.</p>`);
  ch.append(foldDiagram(cv5));
  prose(ch, '<p>Key terms:</p>');
  table(ch, [
    { t: 'Fold', m: 'One of the k parts the data are divided into', s: `5 folds of ${span(d3.min(cv5.per, (d) => d.n), d3.max(cv5.per, (d) => d.n))} patients` },
    { t: 'Training / test fold', m: 'In one round, the k − 1 folds used for fitting and the fold used for scoring', s: `${nD - cv5.per[0].n} patients fit, ${cv5.per[0].n} scored in round 1` },
    { t: 'CV estimate', m: 'The mean of the k fold scores', s: `AUC ${f2(cv5.est)}` },
    { t: 'Out-of-fold prediction', m: 'The prediction for a patient from the model that did not see that patient', s: `one per patient, ${nD} in total` },
    { t: 'Stratified CV', m: 'Folds keep the class balance of the whole data set', s: `each fold ${span(d3.min(cv5.per, (d) => d.pos), d3.max(cv5.per, (d) => d.pos))} responders` },
    { t: 'Grouped CV', m: 'Related samples (same patient, donor, centre) are kept in the same fold', s: 'leave-one-centre-out: train on A, test on B' },
    { t: 'Hyperparameter', m: 'A setting chosen before fitting, e.g. the number of genes or a penalty', s: 'number of features in a signature' },
    { t: 'Nested CV', m: 'An inner CV chooses hyperparameters inside each outer training set; the outer folds score the whole procedure', s: 'chapter 4, step 5' },
    { t: 'Data leakage', m: 'Information from the test samples influences the model or its settings', s: 'choosing genes on all patients before splitting' },
  ], [['t', 'Term'], ['m', 'Meaning'], ['s', 'In our study']]);

  // =====================================================================================
  // 2. WHY
  // =====================================================================================
  ch = chapter(main, 'why', 'Why do we need it?', 'What a performance estimate should provide, and where simpler approaches fall short.');
  prose(ch, `<p>A performance estimate for a predictive model should:</p><ol>
    <li><b>Use patients the model has not seen.</b> Performance on the training patients is optimistic, because the fit has adapted to their noise.</li>
    <li><b>Be precise enough to act on.</b> With ${nD} patients and ${nR} responders, one held-out set of a few dozen patients gives a noisy estimate.</li>
    <li><b>Cover the whole procedure</b>, including feature selection and tuning, so the estimate refers to what will actually be deployed.</li></ol>
    <p>Scoring on the training data fails the first requirement; a single hold-out split often fails the second.</p>`);
  const fw = figure(ch, 'One hold-out split vs 5-fold cross-validation, each repeated 150 times',
    'Each repetition reshuffles the patients. A hold-out repetition fits the three-feature logistic regression on one stratified split and scores the held-out patients; a CV repetition averages the five fold AUCs. Narrower is more precise. The dashed line is the AUC on Centre C of the model fitted to all discovery patients. Hover a bar for counts.');
  let wFrac = 0.3, wCV = true;
  select(fw.controls, { label: 'held-out fraction', value: '0.3', options: [['0.2', '20%'], ['0.3', '30%'], ['0.5', '50%']], onChange: (v) => { wFrac = +v; paintW(); } });
  checkbox(fw.controls, { label: 'show 5-fold CV', checked: wCV, onChange: (c) => { wCV = c; paintW(); } });
  const wS = stats(fw.root);
  const FW = frame(fw.body, { width: 820, height: 270, x: [0.6, 1], y: [0, 60], xLabel: 'estimated AUC', yLabel: 'repetitions' });
  legend(fw.body, [['single hold-out split', TEST()], ['5-fold CV', TRAIN()], ['Centre C (dashed)', ink()]]);
  FW.layer('ref').append('line').attr('x1', FW.x(aucC)).attr('x2', FW.x(aucC)).attr('y1', 0).attr('y2', FW.h).attr('stroke', ink()).attr('stroke-dasharray', '5 4');
  const hold = (frac) => range(150).map((s) => { const sp = stratifiedSplit(y, frac, 500 + s); const fit = fitLogistic(sp.train.map((i) => X3[i]), sp.train.map((i) => y[i])); return auc(sp.test.map((i) => predictProb(X3[i], fit.b0, fit.w)), sp.test.map((i) => y[i])); });
  const HO = { 0.3: hold(0.3) };
  const CVR = range(150).map((s) => cvLogit(X3, y, makeFolds(nD, 5, 900 + s, { y }), 5).est);
  const BW = 0.0125, bins = d3.range(0.6, 1, BW);
  function paintW() {
    HO[wFrac] = HO[wFrac] || hold(wFrac);
    const ser = [{ k: 'hold-out', v: HO[wFrac], c: TEST(), o: 0 }].concat(wCV ? [{ k: '5-fold CV', v: CVR, c: TRAIN(), o: 1 }] : []);
    const bars = ser.flatMap((s) => bins.map((b) => ({ ...s, b, n: s.v.filter((v) => v >= b && v < b + BW).length })));
    const bw = FW.x(0.6 + BW) - FW.x(0.6), half = wCV ? bw / 2 : bw;
    FW.layer('bars').selectAll('rect').data(bars).join('rect').attr('x', (d) => FW.x(d.b) + d.o * half).attr('width', half - 0.5)
      .attr('y', (d) => FW.y(Math.min(60, d.n))).attr('height', (d) => FW.h - FW.y(Math.min(60, d.n))).attr('fill', (d) => d.c).attr('fill-opacity', 0.85)
      .on('mousemove', (e, d) => tip(e, `${d.k}<br>AUC ${f3(d.b)}–${f3(d.b + BW)}: ${d.n} of 150`)).on('mouseleave', untip);
    const h = HO[wFrac];
    FW.y.domain([0, Math.max(10, d3.max(bars, (d) => d.n) * 1.08)]); FW.g.select('.axis.y').call(d3.axisLeft(FW.y).ticks(5));
    FW.g.select('.gridline').call(d3.axisLeft(FW.y).ticks(5).tickSize(-FW.w).tickFormat(''));
    FW.layer('bars').selectAll('rect').attr('y', (d) => FW.y(d.n)).attr('height', (d) => FW.h - FW.y(d.n));
    wS.set({ 'hold-out: 10th–90th percentile': `${f2(q(h, 0.1))} – ${f2(q(h, 0.9))}`, 'hold-out: lowest / highest': `${f2(d3.min(h))} / ${f2(d3.max(h))}`, '5-fold CV: 10th–90th percentile': `${f2(q(CVR, 0.1))} – ${f2(q(CVR, 0.9))}`, 'Centre C': f2(aucC) });
  }
  paintW();
  const h30 = HO[0.3], nTest30 = Math.round(nD * 0.3);
  prose(ch, `<p>With 30% held out (about ${nTest30} patients), the hold-out AUC ranges from ${f2(d3.min(h30))} to ${f2(d3.max(h30))} across reshuffles, depending only on which patients happen to be held out; the middle 80% spans ${f2(q(h30, 0.1))}–${f2(q(h30, 0.9))}. Repeated 5-fold CV estimates span ${f2(q(CVR, 0.1))}–${f2(q(CVR, 0.9))}, because every patient is scored and each model is fitted on 80% of the data. Neither is the true performance on a new population; the Centre C value is itself an estimate from ${V.length} patients at a different centre.</p>`);
  worked(ch, 'Training-set performance',
    `<p>The same model scored on the ${nD} patients it was fitted to has AUC <b>${f2(auc(X3.map((r) => predictProb(r, fitAll.b0, fitAll.w)), y))}</b>. With three features the optimism is small. With hundreds of candidate features and a selection step, it becomes large (chapter 4, step 4), and only an estimate that repeats the selection on held-out data reveals it.</p>`);
  table(ch, [
    { a: 'Score on the training data', u: 'all', h: 'no', c: '1 fit', e: 'optimistic; not an estimate of performance on new data' },
    { a: 'Single hold-out split', u: 'part', h: 'only if tuning uses a separate validation set', c: '1 fit', e: 'unbiased for a model trained on less data, but noisy' },
    { a: '<b>k-fold CV</b>', u: '<b>all</b>', h: '<b>only if every data-driven step is inside the folds</b>', c: '<b>k fits</b>', e: '<b>the procedure, slightly pessimistic (trained on (k−1)/k of the data)</b>' },
    { a: 'Nested CV', u: 'all', h: 'yes, including tuning', c: 'k × inner folds × settings', e: 'the procedure including its tuning step' },
    { a: 'External validation (Centre C)', u: 'separate cohort', h: 'yes', c: '1 fit', e: 'the final model in a new setting; the strongest evidence' },
  ], [['a', 'Approach'], ['u', 'Data used for scoring'], ['h', 'Fair after tuning?'], ['c', 'Cost'], ['e', 'What it estimates']]);

  // =====================================================================================
  // 3. WHERE
  // =====================================================================================
  ch = chapter(main, 'where', 'Where is it used?', 'Any study that has to estimate predictive performance, or choose model settings, from a limited number of samples.');
  cards(ch, [
    { tag: 'Clinical biomarkers', title: 'Response and prognosis signatures', body: 'Cohorts of 50–300 patients are too small to set aside a large test set. Gene selection, scaling and the penalty (e.g. <code>cv.glmnet</code>) are all chosen inside the CV loop, and an independent cohort is used for the final check.' },
    { tag: 'Multi-centre studies', title: 'Leave-one-centre-out', body: 'Each centre or sequencing batch is held out in turn. This estimates performance at a hospital or batch not seen during training, which is usually the question a clinical signature has to answer.' },
    { tag: 'Single-cell', title: 'Cell-type and state classifiers', body: 'Cells from one donor are highly correlated. Folds are formed by donor (<code>GroupKFold</code>), otherwise the classifier is scored on near-copies of its training cells.' },
    { tag: 'Digital pathology', title: 'Tiles from whole-slide images', body: 'Hundreds of image tiles come from each slide. Splitting tiles at random places the same tumour in training and test folds; splits are made by patient or slide.' },
    { tag: 'Sequence models', title: 'Proteins, enhancers, variants', body: 'Homologous sequences behave alike. Folds are formed by sequence-similarity clusters or by chromosome, so a model is not rewarded for recognising near-duplicates.' },
    { tag: 'Microbiome and multi-study', title: 'Leave-one-study-out', body: 'Disease classifiers built from several public cohorts are evaluated by training on all but one study and testing on the remaining one, which exposes study-specific technical effects.' },
    { tag: 'Model choice', title: 'Tuning hyperparameters', body: 'The penalty λ, the number of genes, tree depth or the number of neighbours are chosen by their CV score; the score of the chosen setting then needs nested CV or a separate test set.' },
    { tag: 'Outside biology', title: 'Standard practice in ML', body: 'k-fold CV (often k = 5 or 10) is the default evaluation in scikit-learn, caret and tidymodels, and the basis of most hyperparameter search tools.' },
  ]);
  prose(ch, '<p><b>When not to use it (alone):</b> when the data set is large enough that a single hold-out set gives a precise estimate (CV then mainly adds computation); when the model will be used prospectively on future samples, where a split by time is closer to the real use; and as a substitute for external validation, since all CV folds share the centres, protocols and selection biases of the discovery cohort. For inference (estimating an effect with a confidence interval) a model is usually fitted to all the data instead.</p>');

  // =====================================================================================
  // 4. HOW IT WORKS
  // =====================================================================================
  ch = chapter(main, 'how', 'How it works', 'Six steps, from the basic k-fold loop to nested CV and a complete pipeline checked on Centre C.');

  // ---- Step 1: animated k-fold
  let rb = rung(ch, 1, 'The k-fold loop',
    `<p>Shuffle the patients, cut them into k folds, and for each fold in turn fit on the other folds and score the held-out one. Below, each column of the grid is one of the ${nD} discovery patients (responders on the right, marked by the bar below the grid) and each row is one round.</p>`);
  const fk = figure(rb, 'k-fold cross-validation, one round at a time',
    'Top: training patients (blue) and the test fold (orange) in each round. Bottom: the test-fold AUC of each round; after the last round the band shows mean ± SD. Hover a cell or a dot for details.');
  let K = 5, kStrat = false, kSeed = 1, kCur = -1, kStop = null, kFold, kRes;
  slider(fk.controls, { label: 'k (folds)', min: 2, max: 10, step: 1, value: K, onInput: (v) => { K = v; resetK(); } });
  checkbox(fk.controls, { label: 'stratify by response', checked: kStrat, onChange: (c) => { kStrat = c; resetK(); } });
  button(fk.controls, 'Run', () => runK(), true);
  button(fk.controls, 'Step', () => { haltK(); if (kCur < K - 1) { kCur++; paintK(); } });
  button(fk.controls, 'Reshuffle', () => { kSeed++; resetK(); });
  const kS = stats(fk.root);
  legend(fk.body, [['training patients', TRAIN()], ['test fold', TEST()], ['responder', ink()]]);
  const ord = DIDX.slice().sort((a, b) => y[a] - y[b] || a - b), col = new Array(nD); ord.forEach((i, c) => { col[i] = c; });
  const cw = 4.2, lx = 64;
  const mat = d3.select(fk.body).append('svg').attr('width', 820).attr('role', 'img');
  const FK = frame(fk.body, { width: 820, height: 210, x: [0.5, 10.5], y: [0.4, 1], xLabel: 'round (test fold)', yLabel: 'test-fold AUC' });
  function resetK() {
    haltK(); kFold = makeFolds(nD, K, 40 + kSeed, kStrat ? { y } : {}); kRes = cvLogit(X3, y, kFold, K); kCur = K - 1;
    const H = 8 + K * 15 + 16; mat.attr('viewBox', `0 0 820 ${H}`).selectAll('*').remove();
    const cells = []; for (let f = 0; f < K; f++) DIDX.forEach((i) => cells.push({ f, i }));
    mat.append('g').attr('class', 'cells').selectAll('rect').data(cells).join('rect').attr('x', (d) => lx + col[d.i] * cw).attr('y', (d) => 6 + d.f * 15).attr('width', cw - 0.6).attr('height', 13)
      .attr('fill', (d) => (kFold[d.i] === d.f ? TEST() : TRAIN()))
      .on('mousemove', (e, d) => tip(e, `round ${d.f + 1}: <b>${kFold[d.i] === d.f ? 'test' : 'training'}</b><br>${D[d.i].id} · ${y[d.i] ? 'responded' : 'no response'}`)).on('mouseleave', untip);
    range(K).forEach((f) => {
      mat.append('text').attr('class', 'rl').attr('x', lx - 6).attr('y', 6 + f * 15 + 10.5).attr('text-anchor', 'end').attr('font-size', 11).attr('fill', ink2()).text(`round ${f + 1}`);
      mat.append('text').attr('x', lx + nD * cw + 6).attr('y', 6 + f * 15 + 10.5).attr('font-size', 11).attr('fill', ink2()).text(`${kRes.per[f].pos} resp.`);
    });
    mat.append('rect').attr('x', lx + (nD - nR) * cw).attr('y', 8 + K * 15).attr('width', nR * cw).attr('height', 6).attr('fill', ink());
    FK.x.domain([0.5, K + 0.5]); FK.g.select('.axis.x').call(d3.axisBottom(FK.x).ticks(K).tickFormat(d3.format('d')));
    paintK();
  }
  function paintK() {
    mat.selectAll('g.cells rect').attr('opacity', (d) => (kCur < 0 || kCur === K - 1 || d.f === kCur ? 1 : 0.25));
    mat.selectAll('text.rl').attr('font-weight', (d, f) => (f === kCur ? 700 : 400));
    const done = kRes.per.slice(0, kCur + 1).map((d, f) => ({ ...d, f })), all = kCur === K - 1;
    const aucs = kRes.per.map((d) => d.auc).filter(isFinite), mu = mean(aucs), s = sd(aucs);
    FK.layer('band').selectAll('*').remove();
    FK.layer('band').append('line').attr('x1', 0).attr('x2', FK.w).attr('y1', FK.y(aucC)).attr('y2', FK.y(aucC)).attr('stroke', ink()).attr('stroke-dasharray', '5 4');
    FK.layer('band').append('text').attr('x', 4).attr('y', FK.y(aucC) - 5).attr('text-anchor', 'start').attr('font-size', 11).attr('fill', ink2()).text(`Centre C ${f2(aucC)}`);
    if (all) {
      FK.layer('band').append('rect').attr('x', 0).attr('width', FK.w).attr('y', FK.y(Math.min(1, mu + s))).attr('height', FK.y(mu - s) - FK.y(Math.min(1, mu + s))).attr('fill', css('--accent-soft')).attr('opacity', 0.6);
      FK.layer('band').append('line').attr('x1', 0).attr('x2', FK.w).attr('y1', FK.y(mu)).attr('y2', FK.y(mu)).attr('stroke', css('--accent')).attr('stroke-width', 2);
    }
    FK.layer('dots').selectAll('circle').data(done).join('circle').attr('cx', (d) => FK.x(d.f + 1)).attr('cy', (d) => FK.y(Math.max(0.4, d.auc))).attr('r', 6)
      .attr('fill', TEST()).attr('stroke', css('--surface')).attr('stroke-width', 1.5)
      .on('mousemove', (e, d) => tip(e, `round ${d.f + 1}: AUC ${f3(d.auc)}<br>${d.n} patients, ${d.pos} responders`)).on('mouseleave', untip);
    kS.set({ round: kCur < 0 ? `– / ${K}` : `${kCur + 1} / ${K}`, 'this round AUC': kCur < 0 ? '–' : f3(kRes.per[kCur].auc), 'responders per test fold': `${d3.min(kRes.per, (d) => d.pos)}–${d3.max(kRes.per, (d) => d.pos)}`, 'CV estimate (mean ± SD)': all ? `${f3(mu)} ± ${f3(s)}` : 'complete all rounds' });
  }
  function haltK() { if (kStop) { kStop(); kStop = null; } }
  function runK() { haltK(); kCur = -1; paintK(); kStop = every(550, () => { if (kCur < K - 1) { kCur++; paintK(); } else haltK(); }); }
  resetK();
  prose(rb, `<p>Each patient appears in exactly one orange cell. The fold AUCs vary because each test fold holds only ${span(Math.floor(nD / K), Math.ceil(nD / K))} patients at k = ${K}. The final model is fitted to all ${nD} patients; the CV estimate describes that <b>procedure</b>, not one fitted model.</p>`);

  // ---- Step 2: stratification and variance of the estimate
  const KS = [2, 3, 5, 10], REP = 40;
  const varRes = Object.fromEntries(['plain', 'strat'].map((sc) => [sc, KS.map((k) => range(REP).map((r) => cvLogit(X3, y, makeFolds(nD, k, 2000 + 37 * k + r, sc === 'strat' ? { y } : {}), k)))]));
  const posRange = (sc, k) => { const all = varRes[sc][KS.indexOf(k)].flatMap((r) => r.per.map((d) => d.pos)); return [d3.min(all), d3.max(all)]; };
  const spreadOf = (sc, k) => sd(varRes[sc][KS.indexOf(k)].map((r) => r.est));
  const foldSd = (sc, k) => mean(varRes[sc][KS.indexOf(k)].map((r) => sd(r.per.map((d) => d.auc).filter(isFinite))));
  rb = rung(ch, 2, 'Stratification, choice of k and the variance of the estimate',
    `<p>With ${pc(nR / nD)} responders, a random 10-fold split can leave a test fold with very few of them: across ${REP} reshuffles, test folds held ${span(...posRange('plain', 10))} responders without stratification and ${span(...posRange('strat', 10))} with it. <b>Stratified</b> CV deals responders and non-responders into the folds separately, so each fold keeps the overall class balance.</p>`);
  const fv = figure(rb, 'Repeating CV with different shuffles',
    `Each dot is one complete CV run (the mean of its k fold AUCs) with a different random shuffle, ${REP} per k. The dashed line is the Centre C AUC. Hover a dot for the fold details.`);
  let vMode = 'both';
  select(fv.controls, { label: 'Fold assignment', value: vMode, options: [['both', 'Both'], ['plain', 'Random'], ['strat', 'Stratified']], onChange: (v) => { vMode = v; paintV(); } });
  const vS = stats(fv.root);
  const allEst = ['plain', 'strat'].flatMap((sc) => varRes[sc].flat().map((r) => r.est)).concat([aucC]);
  const vLo = Math.floor(d3.min(allEst) * 100 - 1) / 100, vHi = Math.ceil(d3.max(allEst) * 100 + 1) / 100;
  const FV = frame(fv.body, { width: 820, height: 280, x: [0, 1], y: [vLo, vHi], xLabel: 'number of folds k', yLabel: 'CV estimate (AUC)', axes: false });
  const xb = d3.scaleBand().domain(KS).range([0, FV.w]).padding(0.25);
  FV.g.append('g').attr('class', 'axis x').attr('transform', `translate(0,${FV.h})`).call(d3.axisBottom(xb).tickFormat((k) => `k = ${k}`));
  FV.g.append('g').attr('class', 'axis y').call(d3.axisLeft(FV.y).ticks(5));
  FV.layer('ref').append('line').attr('x1', 0).attr('x2', FV.w).attr('y1', FV.y(aucC)).attr('y2', FV.y(aucC)).attr('stroke', ink()).attr('stroke-dasharray', '5 4');
  legend(fv.body, [['random folds', PLAIN()], ['stratified folds', GOOD()], ['Centre C (dashed)', ink()]]);
  function paintV() {
    const pts = ['plain', 'strat'].filter((sc) => vMode === 'both' || vMode === sc).flatMap((sc) => KS.flatMap((k, ki) => varRes[sc][ki].map((r, j) => ({ sc, k, r, j }))));
    const off = (sc) => (vMode === 'both' ? (sc === 'plain' ? -0.22 : 0.22) : 0) * xb.bandwidth();
    FV.layer('pts').selectAll('circle').data(pts).join('circle')
      .attr('cx', (d) => xb(d.k) + xb.bandwidth() / 2 + off(d.sc) + ((d.j * 7919) % 41 - 20) / 20 * xb.bandwidth() * 0.16)
      .attr('cy', (d) => FV.y(d.r.est)).attr('r', 4).attr('fill', (d) => (d.sc === 'plain' ? PLAIN() : GOOD())).attr('fill-opacity', 0.8).attr('stroke', css('--surface')).attr('stroke-width', 0.8)
      .on('mousemove', (e, d) => tip(e, `${d.sc === 'plain' ? 'random' : 'stratified'} ${d.k}-fold, shuffle ${d.j + 1}<br>CV estimate ${f3(d.r.est)}<br>fold AUCs ${f2(d3.min(d.r.per, (p) => p.auc))}–${f2(d3.max(d.r.per, (p) => p.auc))}<br>responders per fold ${d3.min(d.r.per, (p) => p.pos)}–${d3.max(d.r.per, (p) => p.pos)}`)).on('mouseleave', untip);
    const sc = vMode === 'plain' ? 'plain' : 'strat';
    vS.set(Object.fromEntries(KS.map((k) => [`k = ${k}: SD across shuffles (${sc === 'plain' ? 'random' : 'stratified'})`, f3(spreadOf(sc, k))])));
  }
  paintV();
  const m2 = mean(varRes.strat[0].map((r) => r.est)), m10 = mean(varRes.strat[3].map((r) => r.est));
  prose(rb, `<ul><li><b>Choice of k.</b> Each model is trained on (k − 1)/k of the patients: ${nD / 2} at k = 2, ${nD - nD / 10} at k = 10. Small k is therefore slightly pessimistic; here the mean stratified estimate is ${f3(m2)} at k = 2 and ${f3(m10)} at k = 10. k = 5 or 10 is the usual compromise; leave-one-out (k = n) is nearly unbiased but costly and variable.</li>
    <li><b>Variance.</b> The SD across shuffles at k = 5 is ${f3(spreadOf('strat', 5))} (stratified) and ${f3(spreadOf('plain', 5))} (random). <b>Repeated CV</b> averages several shuffles; it removes this variation but not the uncertainty of having only ${nD} patients.</li>
    <li><b>The fold SD is not a standard error.</b> At k = 5 the fold AUCs have an average SD of ${f3(foldSd('strat', 5))}, but overlapping training sets make the fold scores correlated, so dividing by √k understates the uncertainty (no unbiased variance estimator exists; Bengio and Grandvalet, 2004). Report the fold range or a bootstrap interval.</li></ul>`);
  quiz(rb, 'A study reports "10-fold CV AUC 0.81 ± 0.02 (SE)" from 120 patients. Why is the ± 0.02 likely too small?',
    '<p>The ten fold scores come from models trained on overlapping data (each pair shares 80% of training patients) and are scored on small, correlated test folds, so they are not ten independent measurements. Treating them as independent and dividing the SD by √10 understates the uncertainty. A more informative summary gives the fold range, a bootstrap interval of the out-of-fold predictions, or results on an independent cohort.</p>');

  // ---- Step 3: grouped CV with replicate biopsies
  const NB = 3, zs = F3.map((k) => { const v = D.map((p) => FEATURES[k].get(p)); return [mean(v), sd(v)]; });
  const Z = ALL.map((p) => F3.map((k, j) => (FEATURES[k].get(p) - zs[j][0]) / zs[j][1]));
  const re = rng(57), EPS = ALL.map(() => range(NB).map(() => F3.map(() => re.normal())));
  const labAll = realLab;
  const biops = (w) => ALL.flatMap((p, pi) => range(NB).map((b) => ({ pi, b, x: Z[pi].map((v, j) => v + w * EPS[pi][b][j]), y: labAll[pi] })));
  const knnScore = (train, qx, kN) => {
    const bd = new Array(kN).fill(Infinity), bl = new Array(kN).fill(0), bi = new Array(kN).fill(-1);
    for (const t of train) {
      const dd = (t.x[0] - qx[0]) ** 2 + (t.x[1] - qx[1]) ** 2 + (t.x[2] - qx[2]) ** 2;
      if (dd >= bd[kN - 1]) continue;
      let j = kN - 1; while (j > 0 && bd[j - 1] > dd) { bd[j] = bd[j - 1]; bl[j] = bl[j - 1]; bi[j] = bi[j - 1]; j--; }
      bd[j] = dd; bl[j] = t.y; bi[j] = t.pi;
    }
    return { s: mean(bl), nn: bi[0] };
  };
  const groupRun = (w, kN) => {
    const B = biops(w), Bd = B.filter((b) => b.pi < nD), out = {};
    [['random', makeFolds(Bd.length, 5, 77)], ['grouped', makeFolds(Bd.length, 5, 77, { groups: Bd.map((b) => b.pi) })]].forEach(([nm, fo]) => {
      const per = range(5).map((f) => { const tr = Bd.filter((_, i) => fo[i] !== f), te = Bd.filter((_, i) => fo[i] === f); return auc(te.map((t) => knnScore(tr, t.x, kN).s), te.map((t) => t.y)); });
      out[nm] = okMean(per); out[nm + 'Fold'] = fo;
    });
    const Bv = B.filter((b) => b.pi >= nD && b.b === 0);
    out.centre = auc(Bv.map((t) => knnScore(Bd, t.x, kN).s), Bv.map((t) => t.y)); out.B = Bd;
    return out;
  };
  const WS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1];
  const gCache = {};
  const gCurve = (kN) => (gCache[kN] = gCache[kN] || WS.map((w) => ({ w, ...groupRun(w, kN) })));
  rb = rung(ch, 3, 'Grouped CV: related samples stay together',
    `<p>CV assumes the test fold is as new to the model as a future patient, which fails when samples are related. Here each patient has ${NB} simulated biopsies with some biopsy-to-biopsy variation. With random folds most patients have biopsies on both sides, and a flexible model can score a test biopsy by recognising its sibling. <b>Grouped CV</b> keeps all biopsies of a patient in one fold. The model is k-nearest neighbours on standardised IFN-γ, log2 TMB and CAF score.</p>`);
  const fg = figure(rb, 'Random vs grouped folds with three biopsies per patient',
    'Left: biopsies of the discovery patients in one round; orange biopsies are the test fold, and each grey segment joins a test biopsy to its nearest training biopsy. Right: CV estimate against biopsy-to-biopsy variation (in units of the between-patient SD), with the AUC on one biopsy per Centre C patient. Hover points for details.');
  let gW = 0.1, gK = 5, gScheme = 'random';
  const wIdx = (v) => WS.reduce((b, w, i) => (Math.abs(w - v) < Math.abs(WS[b] - v) ? i : b), 0);
  slider(fg.controls, { label: 'biopsy-to-biopsy variation', min: 0, max: WS.length - 1, step: 1, value: wIdx(gW), fmt: (i) => WS[i].toFixed(2), onInput: (i) => { gW = WS[i]; paintG(); } });
  select(fg.controls, { label: 'neighbours', value: '5', options: [['1', '1'], ['5', '5'], ['15', '15']], onChange: (v) => { gK = +v; paintG(); } });
  select(fg.controls, { label: 'Folds (left panel)', value: gScheme, options: [['random', 'Random over biopsies'], ['grouped', 'Grouped by patient']], onChange: (v) => { gScheme = v; paintG(); } });
  const gS = stats(fg.root);
  const rowG = el('div', { class: 'fig-row' }); fg.body.append(rowG);
  const ga = el('div'), gb = el('div'); rowG.append(ga, gb);
  const FG = frame(ga, { width: 460, height: 330, x: [-2.6, 2.6], y: [-2.8, 2.8], xLabel: 'IFN-γ (standardised)', yLabel: 'log2 TMB (standardised)' });
  const FC = frame(gb, { width: 460, height: 330, x: [0, 1], y: [0.5, 1], xLabel: 'biopsy-to-biopsy variation (SD units)', yLabel: 'AUC' });
  legend(fg.body, [['training biopsy', TRAIN()], ['test biopsy', TEST()], ['random folds', BAD()], ['grouped folds', GOOD()], ['Centre C', ink()]]);
  function paintG() {
    const cur = gCurve(gK), row = cur[WS.indexOf(gW)], fo = row[gScheme + 'Fold'], Bd = row.B;
    const tr = Bd.filter((_, i) => fo[i] !== 0), te = Bd.filter((_, i) => fo[i] === 0);
    const nnOf = te.map((t) => { let best = null, bd = Infinity; tr.forEach((s) => { const dd = (s.x[0] - t.x[0]) ** 2 + (s.x[1] - t.x[1]) ** 2 + (s.x[2] - t.x[2]) ** 2; if (dd < bd) { bd = dd; best = s; } }); return { t, s: best }; });
    const sib = nnOf.filter((d) => d.s.pi === d.t.pi).length;
    FG.layer('seg').selectAll('line').data(nnOf).join('line').attr('x1', (d) => FG.x(d.t.x[0])).attr('y1', (d) => FG.y(d.t.x[1])).attr('x2', (d) => FG.x(d.s.x[0])).attr('y2', (d) => FG.y(d.s.x[1])).attr('stroke', muted()).attr('stroke-width', 1.2);
    const pts = tr.map((b) => ({ b, te: false })).concat(te.map((b) => ({ b, te: true })));
    FG.layer('pts').selectAll('circle').data(pts).join('circle').attr('cx', (d) => FG.x(Math.max(-2.6, Math.min(2.6, d.b.x[0])))).attr('cy', (d) => FG.y(Math.max(-2.8, Math.min(2.8, d.b.x[1]))))
      .attr('r', (d) => (d.te ? 4.2 : 3)).attr('fill', (d) => (d.te ? TEST() : TRAIN())).attr('fill-opacity', (d) => (d.te ? 0.95 : 0.45)).attr('stroke', css('--surface')).attr('stroke-width', 0.6)
      .on('mousemove', (e, d) => tip(e, `${ALL[d.b.pi].id}, biopsy ${d.b.b + 1} · ${d.te ? 'test' : 'training'}<br>${d.b.y ? 'responded' : 'no response'}`)).on('mouseleave', untip);
    const lines = [['random', BAD()], ['grouped', GOOD()], ['centre', ink()]];
    FC.layer('l').selectAll('path').data(lines).join('path').attr('fill', 'none').attr('stroke', (d) => d[1]).attr('stroke-width', 2.2).attr('stroke-dasharray', (d) => (d[0] === 'centre' ? '5 4' : null))
      .attr('d', (d) => d3.line().x((r) => FC.x(r.w)).y((r) => FC.y(Math.max(0.5, r[d[0]])))(cur));
    FC.layer('m').selectAll('line').data([gW]).join('line').attr('x1', (w) => FC.x(w)).attr('x2', (w) => FC.x(w)).attr('y1', 0).attr('y2', FC.h).attr('stroke', muted()).attr('stroke-dasharray', '3 3');
    FC.layer('d').selectAll('circle').data(lines).join('circle').attr('cx', FC.x(gW)).attr('cy', (d) => FC.y(Math.max(0.5, row[d[0]]))).attr('r', 5.5).attr('fill', (d) => d[1]).attr('stroke', css('--surface')).attr('stroke-width', 1.5)
      .on('mousemove', (e, d) => tip(e, `${d[0] === 'centre' ? 'Centre C' : d[0] + ' folds'}: AUC ${f3(row[d[0]])}`)).on('mouseleave', untip);
    gS.set({ 'random folds': f3(row.random), 'grouped folds': f3(row.grouped), 'Centre C': f3(row.centre), 'test biopsies whose nearest neighbour is a sibling': `${sib} / ${te.length}` });
  }
  paintG();
  const g5 = gCurve(5), g1 = gCurve(1), at = (c, w) => c[WS.indexOf(w)];
  const sibAt = (w) => { const r = at(g5, w), fo = r.randomFold, tr = r.B.filter((_, i) => fo[i] !== 0), te = r.B.filter((_, i) => fo[i] === 0);
    return [te.filter((t) => { let best = null, bd = Infinity; tr.forEach((u) => { const dd = (u.x[0] - t.x[0]) ** 2 + (u.x[1] - t.x[1]) ** 2 + (u.x[2] - t.x[2]) ** 2; if (dd < bd) { bd = dd; best = u; } }); return best.pi === t.pi; }).length, te.length]; };
  const s01 = sibAt(0.1), s1 = sibAt(1), r01 = at(g5, 0.1), r1 = at(g5, 1);
  prose(rb, `<p>At variation 0.1 with 5 neighbours, ${s01[0]} of ${s01[1]} test biopsies in round 1 have a sibling as their nearest training biopsy. Random folds give AUC <b>${f3(r01.random)}</b>, grouped folds <b>${f3(r01.grouped)}</b>, and Centre C, where every patient is new, <b>${f3(r01.centre)}</b>. With 1 neighbour and variation 0.05 the effect is extreme: ${f3(at(g1, 0.05).random)} vs ${f3(at(g1, 0.05).grouped)}. At variation 1 the two schemes agree (${f3(r1.random)} vs ${f3(r1.grouped)}). The same applies to cells from one donor, tiles from one slide, repeated visits and technical replicates.</p>`);
  const cA = D.map((p) => (p.centreB ? 1 : 0));
  const loco = [0, 1].map((tc) => { const tr = DIDX.filter((i) => cA[i] !== tc), te = DIDX.filter((i) => cA[i] === tc); const fit = fitLogistic(tr.map((i) => X3[i]), tr.map((i) => y[i])); const p = te.map((i) => predictProb(X3[i], fit.b0, fit.w)); return { tc, n: te.length, auc: auc(p, te.map((i) => y[i])), pm: mean(p), obs: mean(te.map((i) => y[i])) }; });
  worked(rb, 'Grouping by centre: leave-one-centre-out',
    `<p>The discovery patients come from two centres, so a model can be trained on one and tested on the other. For the three-feature logistic regression:</p>
     <ul>${loco.map((l) => `<li>Test on Centre ${l.tc ? 'B' : 'A'} (${l.n} patients): AUC <b>${f3(l.auc)}</b>; mean predicted probability ${pc(l.pm)} against an observed response rate of ${pc(l.obs)}.</li>`).join('')}</ul>
     <p>Stratified 5-fold CV over both centres gave ${f3(cv5.est)}. ${Math.abs(mean(loco.map((l) => l.auc)) - cv5.est) < 0.05 ? 'Ranking transfers between centres because the true response rule is the same; the base rate differs, which shows in the calibration.' : 'The difference reflects the change in patient mix between centres.'} Leave-one-centre-out estimates performance at an unseen centre, which random folds cannot.</p>`);

  // ---- Step 4: leakage with permuted labels
  const NG0 = 1000, M0 = 10, NP0 = 20;
  const leakRun = (F, lab, m) => {
    const fo = sFold(lab), stAll = featStats(F, DIDX, lab);
    const le = sigCV(F, lab, DIDX, fo, 5, [m], { leaky: true, stAll })[0].est;
    const ok = sigCV(F, lab, DIDX, fo, 5, [m], { stAll })[0].est;
    return { leaky: le, inside: ok, centre: auc(sigScores(F, stAll, stAll, rankBy(stAll), [m], VIDX)[0], yv) };
  };
  const F0 = Fof(NG0), real0 = leakRun(F0, realLab, M0), perm0 = range(NP0).map((s) => leakRun(F0, permLab(s), M0));
  rb = rung(ch, 4, 'Leakage: selecting features before splitting',
    `<p>To mimic a gene-expression study, ${NG0.toLocaleString('en-GB')} random <b>noise features</b> (standard normal values, unrelated to response) are added to the 10 cohort measurements. The classifier is a <b>weighted signature</b>: the m features with the largest |t-statistic|, standardised and summed with their t-statistics as weights. Two pipelines differ only in where the selection happens:</p>
     <ul><li><b>Leaky:</b> rank features on all ${nD} patients, then cross-validate the signature on the chosen features.</li><li><b>Correct:</b> inside each training fold, rank the features on the training patients only.</li></ul>
     <p>The decisive test is to <b>permute the labels</b>: shuffle response across patients so that no feature can predict it. Any procedure that then reports an AUC clearly above 0.5 is measuring its own leakage.</p>`);
  const fp = figure(rb, 'CV estimate with permuted labels: leaky vs correct selection',
    'Small dots: one permutation each (labels shuffled among the discovery patients). Rings: the real labels. The Centre C row scores the final model, fitted to all discovery patients, on the real Centre C outcomes. The dashed line is AUC 0.5 (no information).');
  let pNG = NG0, pM = M0, pRuns = perm0.slice(), pStop = null;
  select(fp.controls, { label: 'noise features', value: String(pNG), options: [['200', '200'], ['1000', '1,000'], ['5000', '5,000']], onChange: (v) => { pNG = +v; resetP(); } });
  select(fp.controls, { label: 'features selected m', value: String(pM), options: [5, 10, 20, 50].map((m) => [String(m), String(m)]), onChange: (v) => { pM = +v; resetP(); } });
  button(fp.controls, 'Run 20 more permutations', () => morePerm(20), true);
  const pS = stats(fp.root);
  const ROWS = [['leaky', 'selection before splitting', BAD()], ['inside', 'selection inside folds', GOOD()], ['centre', 'Centre C (final model)', ink()]];
  const FP = frame(fp.body, { width: 820, height: 210, x: [0.25, 1], y: [0, 3], xLabel: 'AUC', axes: false, grid: false, margin: { left: 180 } });
  FP.g.append('g').attr('class', 'axis x').attr('transform', `translate(0,${FP.h})`).call(d3.axisBottom(FP.x).ticks(8));
  FP.layer('ref').append('line').attr('x1', FP.x(0.5)).attr('x2', FP.x(0.5)).attr('y1', 0).attr('y2', FP.h).attr('stroke', muted()).attr('stroke-dasharray', '5 4');
  ROWS.forEach(([, lab], i) => FP.layer('ref').append('text').attr('x', -10).attr('y', FP.y(2.5 - i) + 4).attr('text-anchor', 'end').attr('font-size', 12).attr('fill', ink2()).text(lab));
  legend(fp.body, [['permuted labels (one dot per permutation)', muted()], ['real labels (ring)', ink()], ['leaky', BAD()], ['correct', GOOD()]]);
  let pReal = real0;
  function resetP() { if (pStop) { pStop(); pStop = null; } pReal = leakRun(Fof(pNG), realLab, pM); pRuns = []; paintP(); morePerm(20); }
  function morePerm(nAdd) {
    if (pStop) pStop();
    const F = Fof(pNG), target = Math.min(200, pRuns.length + nAdd);
    pStop = animate(() => { if (pRuns.length < target) pRuns.push(leakRun(F, permLab(pRuns.length), pM)); paintP(); return pRuns.length < target; });
  }
  function paintP() {
    const dots = pRuns.flatMap((r, j) => ROWS.map(([k], i) => ({ v: r[k], i, j, k })));
    FP.layer('dots').selectAll('circle').data(dots).join('circle').attr('cx', (d) => FP.x(Math.max(0.25, d.v))).attr('cy', (d) => FP.y(2.5 - d.i) + ((d.j * 7919) % 31 - 15) * 0.9)
      .attr('r', 3.2).attr('fill', (d) => ROWS[d.i][2]).attr('fill-opacity', 0.55)
      .on('mousemove', (e, d) => tip(e, `permutation ${d.j + 1}<br>${ROWS[d.i][1]}: AUC ${f3(d.v)}`)).on('mouseleave', untip);
    FP.layer('real').selectAll('circle').data(ROWS).join('circle').attr('cx', (d) => FP.x(pReal[d[0]])).attr('cy', (d, i) => FP.y(2.5 - i)).attr('r', 8)
      .attr('fill', 'none').attr('stroke', (d) => d[2]).attr('stroke-width', 2.5)
      .on('mousemove', (e, d) => tip(e, `real labels<br>${d[1]}: AUC ${f3(pReal[d[0]])}`)).on('mouseleave', untip);
    const mm = (k) => (pRuns.length ? f3(mean(pRuns.map((r) => r[k]))) : '–');
    pS.set({ permutations: pRuns.length, 'leaky, mean over permutations': mm('leaky'), 'correct, mean over permutations': mm('inside'), 'real labels: leaky / correct / Centre C': `${f2(pReal.leaky)} / ${f2(pReal.inside)} / ${f2(pReal.centre)}` });
  }
  paintP();
  const pm = (k) => mean(perm0.map((r) => r[k]));
  const perm5k = range(6).map((s0) => leakRun(Fof(5000), permLab(s0), M0)), pm5k = mean(perm5k.map((r) => r.leaky));
  const seC0 = aucSE(real0.centre, sum(yv), yv.length - sum(yv));
  const scl = (() => { const lab = permLab(0), fo = sFold(lab); return [sigCV(F0, lab, DIDX, fo, 5, [M0])[0].est, sigCV(F0, lab, DIDX, fo, 5, [M0], { scaleAll: true })[0].est]; })();
  prose(rb, `<p>With ${NG0.toLocaleString('en-GB')} noise features, m = ${M0} and permuted labels, the leaky pipeline reports a mean CV AUC of <b>${f3(pm('leaky'))}</b> over ${NP0} permutations, although nothing can be predicted; the correct pipeline reports <b>${f3(pm('inside'))}</b>, and the Centre C AUC of the final model averages ${f3(pm('centre'))}. The leaky selection used the test folds' labels, so those folds were no longer unseen. ${pm5k > pm('leaky') ? `More candidate features make the effect stronger: with 5,000 noise features the leaky mean is ${f3(pm5k)} (${perm5k.length} permutations).` : ''}</p>
    <p>With the real labels the leaky estimate (${f3(real0.leaky)}) is above the correct one (${f3(real0.inside)}), but <b>both pipelines produce the same final model</b>, so the leak changes only the estimate, not the model. ${Math.abs(real0.leaky - real0.centre) < Math.abs(real0.inside - real0.centre) ? `The leaky estimate lies closer to Centre C (${f3(real0.centre)}, standard error about ${f3(seC0)}) only by coincidence: the permutation runs show it is high whether or not there is any signal.` : `Centre C gives ${f3(real0.centre)}.`}</p>
    <p><b>Scaling on all data.</b> Standardising on all patients is also leakage in principle, but small here (permuted labels, m = ${M0}: ${f3(scl[0])} fold-wise vs ${f3(scl[1])} on all patients). Imputation, batch correction, PCA, oversampling and any step that looks at the labels can leak much more; all belong inside the folds.</p>`);

  // ---- Step 5: nested CV
  const permN = permLab(0);
  const nestOf = (lab) => {
    const fo = sFold(lab), flat = sigCV(F0, lab, DIDX, fo, 5, MG), best = flat.reduce((b, r) => (r.est > b.est ? r : b));
    return { flat, best, nested: sigNested(F0, lab, DIDX, fo, 5, MG), centre: centreC(F0, lab, best.m) };
  };
  const NEST = { real: nestOf(realLab), perm: nestOf(permN) };
  const NPN = 10, permNest = range(NPN).map((s0) => { const lb = permLab(s0), fo = sFold(lb); return { best: d3.max(sigCV(F0, lb, DIDX, fo, 5, MG), (r) => r.est), nested: sigNested(F0, lb, DIDX, fo, 5, MG).est }; });
  const pnB = mean(permNest.map((r) => r.best)), pnN = mean(permNest.map((r) => r.nested));
  rb = rung(ch, 5, 'Nested CV: tuning without scoring on the same folds',
    `<p>The number of features m is a <b>hyperparameter</b>. Running CV for each candidate m (here ${MG.join(', ')}) and reporting the best score reports the <b>maximum of ${MG.length} noisy estimates</b>, which is biased upwards. <b>Nested CV</b> moves the choice inside the loop: within each outer training set, an inner 4-fold CV picks m, and the outer test fold scores that choice once (Varma and Simon, 2006).</p>`);
  const fn = figure(rb, 'Nested cross-validation, computed on the study with 1,000 noise features',
    'Top: the outer loop; click a round to show its inner loop. Each inner round trains on blue, scores candidate settings on green, and never touches the outer test fold (orange). The text under the diagram reports what that outer round chose and scored.');
  let nLab = 'real', nSel = 0;
  select(fn.controls, { label: 'Labels', value: nLab, options: [['real', 'Real'], ['perm', 'Permuted']], onChange: (v) => { nLab = v; paintN(); } });
  const nS = stats(fn.root);
  legend(fn.body, [['training', TRAIN()], ['inner validation (choose m)', VAL()], ['outer test (score once)', TEST()]]);
  const NSV = d3.select(fn.body).append('svg').attr('viewBox', '0 0 820 262').attr('width', 820).attr('role', 'img');
  function paintN() {
    const R = NEST[nLab], rd = R.nested.rounds[nSel], bx = 150, bw = 470, NO = 5, NI = 4;
    NSV.selectAll('*').remove();
    const txt = (x, yy, t, o = {}) => NSV.append('text').attr('x', x).attr('y', yy).attr('font-size', o.size || 12).attr('text-anchor', o.anchor || 'start').attr('font-weight', o.bold ? 600 : 400).attr('fill', o.c || ink2()).text(t);
    txt(10, 20, 'Outer loop', { bold: true, c: ink() }); txt(10, 36, 'click a round', { size: 11 });
    for (let o = 0; o < NO; o++) {
      const yy = 8 + o * 17, g = NSV.append('g').style('cursor', 'pointer').on('click', () => { nSel = o; paintN(); });
      for (let f = 0; f < NO; f++) g.append('rect').attr('x', bx + f * bw / NO).attr('y', yy).attr('width', bw / NO - 2).attr('height', 14).attr('rx', 2).attr('fill', f === o ? TEST() : TRAIN()).attr('opacity', o === nSel ? 1 : 0.3);
      g.append('text').attr('x', bx + bw + 8).attr('y', yy + 11).attr('font-size', 11).attr('font-weight', o === nSel ? 600 : 400).attr('fill', o === nSel ? ink() : muted()).text(`round ${o + 1}: m = ${R.nested.rounds[o].m}, AUC ${f2(R.nested.rounds[o].auc)}`);
    }
    const y0 = 112, trW = bw * (NO - 1) / NO;
    txt(10, y0 + 12, 'Inner loop', { bold: true, c: ink() }); txt(10, y0 + 28, 'outer-training only', { size: 11 });
    for (let i = 0; i < NI; i++) {
      const yy = y0 + i * 17;
      for (let f = 0; f < NI; f++) NSV.append('rect').attr('x', bx + f * trW / NI).attr('y', yy).attr('width', trW / NI - 2).attr('height', 14).attr('rx', 2).attr('fill', f === i ? VAL() : TRAIN());
      NSV.append('rect').attr('x', bx + trW + 4).attr('y', yy).attr('width', bw / NO - 6).attr('height', 14).attr('rx', 2).attr('fill', 'none').attr('stroke', TEST()).attr('stroke-dasharray', '3 2');
    }
    txt(bx + trW + 4, y0 + NI * 17 + 12, 'not used', { size: 11, c: muted() });
    const y1 = y0 + NI * 17 + 36;
    txt(bx, y1, `Outer round ${nSel + 1}: inner CV on ${nD - rd.n} training patients picks m = ${rd.m} (inner AUC ${f3(rd.inner)}).`, { c: ink() });
    txt(bx, y1 + 19, `Refit with m = ${rd.m} on those patients; score once on ${rd.n} outer-test patients: AUC ${f3(rd.auc)}.`, { c: ink() });
    txt(bx, y1 + 38, `The nested estimate is the mean of the five outer AUCs: ${f3(R.nested.est)}.`, { c: ink() });
    nS.set({ 'best single CV score (m chosen on the same folds)': `${f3(R.best.est)} (m = ${R.best.m})`, 'nested CV estimate': f3(R.nested.est), 'Centre C, final model': f3(R.centre) });
  }
  paintN();
  const nr = NEST.real, np = NEST.perm;
  table(rb, [
    { l: 'Real', b: `${f3(nr.best.est)} (m = ${nr.best.m})`, n: f3(nr.nested.est), c: f3(nr.centre), ms: nr.nested.rounds.map((r) => r.m).join(', ') },
    { l: 'Permuted (permutation 1)', b: `${f3(np.best.est)} (m = ${np.best.m})`, n: f3(np.nested.est), c: f3(np.centre), ms: np.nested.rounds.map((r) => r.m).join(', ') },
    { l: `Permuted (mean of ${NPN} permutations)`, b: f3(pnB), n: f3(pnN), c: '', ms: '' },
  ], [['l', 'Labels'], ['b', 'Best CV score over m'], ['n', 'Nested CV estimate'], ['c', 'Centre C, final model'], ['ms', 'm chosen in the 5 outer rounds']], { cls: 'compact' });
  prose(rb, `<p>Averaged over ${NPN} permutations, the best single CV score is ${f3(pnB)} and the nested estimate ${f3(pnN)}; with no signal the correct answer is 0.5, so the difference of ${f3(pnB - pnN)} is the optimism created by picking the best of ${MG.length} candidates. With the real labels the two estimates are ${f3(nr.best.est)} and ${f3(nr.nested.est)}. The outer rounds choose different m: nested CV evaluates the <b>tuning procedure</b>, which is then run once on all discovery patients to give the final model.${nr.centre - nr.nested.est > 0.03 ? ` Centre C (${f3(nr.centre)}, approximate standard error ${f3(aucSE(nr.centre, sum(yv), yv.length - sum(yv)))}) is higher than the nested estimate: each outer model is fitted on only ${nD - Math.round(nD / 5)} patients, and the external cohort has its own sampling error.` : ''}</p>`);

  // ---- Step 6: lab
  rb = rung(ch, 6, 'Lab: build a pipeline and check it on Centre C',
    `<p>Assemble a pipeline from the choices above. The CV estimate is computed on the ${nD} discovery patients; the final model is then fitted to all of them and scored on the ${V.length} Centre C patients, which play no part in any choice.</p>`);
  const fl = figure(rb, 'Pipeline lab',
    'Top row: the AUC of each CV test fold. Middle: the CV estimate (mean ± SD of the folds). Bottom: the AUC of the final model on Centre C. Hover a marker for its value.');
  const L = { ng: '1000', where: 'inside', m: '10', folds: 'strat', scale: false, lab: 'real' };
  const lOpt = (key, label, opts) => select(fl.controls, { label, value: L[key], options: opts, onChange: (v) => { L[key] = v; paintL(); } });
  lOpt('ng', 'noise features', [['0', 'none'], ['200', '200'], ['1000', '1,000'], ['5000', '5,000']]);
  lOpt('where', 'feature selection', [['inside', 'inside each training fold'], ['before', 'on all patients, before splitting']]);
  lOpt('m', 'features m', MG.map((m) => [String(m), String(m)]).concat([['best', 'tuned: best CV score'], ['nested', 'tuned: nested CV']]));
  lOpt('folds', 'folds', [['strat', 'stratified 5-fold'], ['random', 'random 5-fold'], ['centre', 'leave-one-centre-out']]);
  lOpt('lab', 'labels', [['real', 'real'], ['perm', 'permuted']]);
  checkbox(fl.controls, { label: 'standardise on all patients', checked: L.scale, onChange: (c) => { L.scale = c; paintL(); } });
  const lS = stats(fl.root);
  const FL = frame(fl.body, { width: 820, height: 200, x: [0.25, 1], y: [0, 3], xLabel: 'AUC', axes: false, grid: false, margin: { left: 150 } });
  FL.g.append('g').attr('class', 'axis x').attr('transform', `translate(0,${FL.h})`).call(d3.axisBottom(FL.x).ticks(8));
  FL.layer('ref').append('line').attr('x1', FL.x(0.5)).attr('x2', FL.x(0.5)).attr('y1', 0).attr('y2', FL.h).attr('stroke', muted()).attr('stroke-dasharray', '5 4');
  ['test folds', 'CV estimate', 'Centre C'].forEach((t, i) => FL.layer('ref').append('text').attr('x', -10).attr('y', FL.y(2.5 - i) + 4).attr('text-anchor', 'end').attr('font-size', 12).attr('fill', ink2()).text(t));
  legend(fl.body, [['test fold', TEST()], ['CV estimate ± SD', css('--accent')], ['Centre C', ink()]]);
  const runLab = (o) => {
    const F = Fof(+o.ng), lab = o.lab === 'perm' ? permN : realLab, k = o.folds === 'centre' ? 2 : 5;
    const fo = o.folds === 'centre' ? cA : o.folds === 'random' ? makeFolds(nD, 5, 11) : sFold(lab);
    const opt = { leaky: o.where === 'before', scaleAll: o.scale };
    const grid = MG.filter((m) => m <= F.length || m === MG.find((g) => g >= F.length));
    let per, est, mFinal, mNote;
    if (o.m === 'nested') { const N = sigNested(F, lab, DIDX, fo, k, grid, opt); per = N.rounds.map((r) => r.auc); est = N.est; mNote = N.rounds.map((r) => r.m).join(', '); }
    if (o.m === 'nested' || o.m === 'best') { const fl2 = sigCV(F, lab, DIDX, fo, k, grid, opt), b = fl2.reduce((a, r) => (r.est > a.est ? r : a)); mFinal = Math.min(b.m, F.length); if (o.m === 'best') { per = b.per; est = b.est; mNote = String(mFinal); } }
    else { mFinal = Math.min(+o.m, F.length); const r = sigCV(F, lab, DIDX, fo, k, [mFinal], opt)[0]; per = r.per; est = r.est; mNote = String(mFinal); }
    const stAll = featStats(F, DIDX, lab), top = rankBy(stAll).slice(0, mFinal);
    return { per, est, sdv: sd(per.filter(isFinite)), centre: auc(sigScores(F, stAll, stAll, rankBy(stAll), [mFinal], VIDX)[0], yv), mNote, mFinal, nCoh: top.filter((j) => j < LAB.length).map((j) => FEATURES[LAB[j]].label) };
  };
  function paintL() {
    const R = runLab(L);
    FL.layer('f').selectAll('circle').data(R.per).join('circle').attr('cx', (v) => FL.x(Math.max(0.25, v))).attr('cy', (v, i) => FL.y(2.5) + (i % 2 ? 5 : -5)).attr('r', 6).attr('fill', TEST()).attr('stroke', css('--surface')).attr('stroke-width', 1.5)
      .on('mousemove', (e, v) => tip(e, `test fold: AUC ${f3(v)}`)).on('mouseleave', untip);
    FL.layer('e').selectAll('*').remove();
    FL.layer('e').append('line').attr('x1', FL.x(Math.max(0.25, R.est - R.sdv))).attr('x2', FL.x(Math.min(1, R.est + R.sdv))).attr('y1', FL.y(1.5)).attr('y2', FL.y(1.5)).attr('stroke', css('--accent')).attr('stroke-width', 2.5);
    FL.layer('e').append('path').attr('d', d3.symbol(d3.symbolDiamond, 150)).attr('transform', `translate(${FL.x(Math.max(0.25, R.est))},${FL.y(1.5)})`).attr('fill', css('--accent')).attr('stroke', css('--surface'))
      .on('mousemove', (e) => tip(e, `CV estimate ${f3(R.est)} ± ${f3(R.sdv)}`)).on('mouseleave', untip);
    FL.layer('e').append('path').attr('d', d3.symbol(d3.symbolSquare, 130)).attr('transform', `translate(${FL.x(Math.max(0.25, R.centre))},${FL.y(0.5)})`).attr('fill', ink()).attr('stroke', css('--surface'))
      .on('mousemove', (e) => tip(e, `Centre C: AUC ${f3(R.centre)}`)).on('mouseleave', untip);
    lS.set({ 'CV estimate': f3(R.est), 'Centre C': f3(R.centre), 'CV − Centre C': (R.est - R.centre >= 0 ? '+' : '−') + f3(Math.abs(R.est - R.centre)), 'm used (per fold / final)': L.m === 'nested' ? `${R.mNote} / ${R.mFinal}` : R.mNote, 'cohort measurements in final model': R.nCoh.length ? R.nCoh.join(', ') : 'none' });
  }
  paintL();
  const lab = (o) => runLab({ ...L, ...o });
  const t1 = [lab({}), lab({ where: 'before' })], t2 = [lab({ lab: 'perm', where: 'before', ng: '5000' }), lab({ lab: 'perm', ng: '5000' })];
  const t3 = [lab({ m: 'best', lab: 'perm' }), lab({ m: 'nested', lab: 'perm' })], t4 = [lab({ ng: '0' }), lab({ ng: '5000' })];
  const t5 = lab({ folds: 'centre' }), t6 = [lab({ lab: 'perm' }), lab({ lab: 'perm', scale: true })];
  table(rb, [
    { t: 'Default, then selection <b>on all patients</b>', s: `CV ${f3(t1[0].est)} → <b>${f3(t1[1].est)}</b>; Centre C ${f3(t1[0].centre)} in both`, m: 'Leaky selection raises the estimate, not the performance: the final model is identical.' },
    { t: '<b>Permuted</b> labels, 5,000 noise features: leaky vs inside', s: `CV <b>${f3(t2[0].est)}</b> vs ${f3(t2[1].est)}; Centre C ${f3(t2[0].centre)}`, m: 'A permutation check exposes leakage directly: the correct pipeline stays near 0.5.' },
    { t: 'Permuted labels, m <b>tuned: best CV score</b> → <b>nested CV</b>', s: `CV ${f3(t3[0].est)} → <b>${f3(t3[1].est)}</b>`, m: 'Reporting the best of several CV scores is optimistic; nested CV scores the tuning procedure itself and stays near 0.5 on average (step 5).' },
    { t: 'Noise features <b>none</b> → <b>5,000</b>', s: `CV ${f3(t4[0].est)} → ${f3(t4[1].est)}; Centre C ${f3(t4[0].centre)} → ${f3(t4[1].centre)}; cohort measurements among the ${t4[1].mFinal} selected: ${t4[1].nCoh.length}`, m: 'Noise features compete with real ones for the m places. Here the strong cohort measurements still rank near the top, so the loss is small; with weaker signals or smaller cohorts, noise features displace real ones.' },
    { t: 'Folds <b>leave-one-centre-out</b>', s: `CV ${f3(t5.est)} (two folds: ${t5.per.map(f2).join(', ')})`, m: 'Each fold is a centre; the estimate refers to an unseen centre, at the cost of fewer and larger folds.' },
    { t: 'Permuted, then <b>standardise on all patients</b>', s: `CV ${f3(t6[0].est)} → ${f3(t6[1].est)}`, m: 'Scaling leakage is minor for this pipeline; it is still avoided by fitting the scaler inside the folds.' },
  ], [['t', 'Try'], ['s', 'What you see'], ['m', 'What it means']], { cls: 'compact' });

  // =====================================================================================
  // 5. PITFALLS
  // =====================================================================================
  ch = chapter(main, 'pitfalls', 'Pitfalls', 'Common problems, how to recognise them, and how to address them.');
  cards(ch, [
    { tag: 'Symptom: excellent CV, poor external result', title: 'Feature selection outside the folds', body: `Differential-expression filtering, univariate screening or "top variable genes" computed on all samples before CV. With permuted labels and ${NG0.toLocaleString('en-GB')} noise features, this gave a mean CV AUC of ${f3(pm('leaky'))} here. Described for microarrays by Ambroise and McLachlan (2002) and Simon et al. (2003). <b>Fix:</b> put selection in the pipeline that is refitted in each training fold.` },
    { tag: 'Symptom: the reported score is the best of a grid', title: 'Tuning and scoring on the same folds', body: `Choosing λ, m, k or a model family by CV score and reporting that same score. Permuted labels, mean of ${NPN} permutations: ${f3(pnB)} best single score vs ${f3(pnN)} nested. <b>Fix:</b> nested CV, or a separate test set that the search never sees.` },
    { tag: 'Symptom: near-perfect CV with a flexible model', title: 'Related samples split across folds', body: 'Replicates, multiple biopsies, cells from one donor, tiles from one slide, repeated visits. <b>Fix:</b> grouped CV (<code>GroupKFold</code>, <code>StratifiedGroupKFold</code>, <code>LeaveOneGroupOut</code>) on the patient, donor or slide identifier.' },
    { tag: 'Symptom: preprocessing fitted once', title: 'Normalisation, batch correction, imputation, PCA', body: 'Any step that estimates parameters from the data (scaling factors, ComBat batch means, imputation models, principal components, SMOTE oversampling) is part of the model. <b>Fix:</b> fit these steps on the training fold and apply them to the test fold; sample-wise steps that use only one sample (e.g. log-transforming its counts) are safe.' },
    { tag: 'Symptom: undefined or erratic fold AUCs', title: 'Rare classes without stratification', body: `At 10 folds without stratification, test folds held as few as ${posRange('plain', 10)[0]} responder${posRange('plain', 10)[0] === 1 ? '' : 's'}. <b>Fix:</b> stratify, use fewer folds, or pool the out-of-fold predictions before computing AUC.` },
    { tag: 'Symptom: ± values that seem too precise', title: 'Treating fold scores as independent', body: 'Training sets overlap, so the fold SD divided by √k is not a standard error. <b>Fix:</b> report the range across folds and repeats, use a bootstrap of the out-of-fold predictions, and keep an external cohort for the final estimate.' },
    { tag: 'Symptom: the test set is "used up"', title: 'Repeatedly checking the external set', body: 'Adjusting the model after each look at Centre C turns it into a tuning set, and its estimate becomes optimistic too. <b>Fix:</b> lock the complete pipeline (features, preprocessing, model, threshold) before the external evaluation and run it once.' },
    { tag: 'Symptom: a feature that predicts too well', title: 'Label leakage through the features', body: 'A variable recorded after or because of the outcome (e.g. a post-treatment scan, the treatment arm, a sample identifier that encodes the batch of responders). No splitting scheme removes this. <b>Fix:</b> check when and how every feature was measured relative to the outcome.' },
  ]);

  // =====================================================================================
  // 6. RECAP
  // =====================================================================================
  ch = chapter(main, 'recap', 'Recap', 'Key points.');
  recap(ch, [
    '<b>k-fold CV</b> holds out each of k folds once, fits on the rest, and averages the k test scores; every sample is predicted once by a model that did not see it.',
    'It estimates the performance of a <b>procedure</b> trained on (k − 1)/k of the data; the final model is refitted on all data.',
    'Use <b>stratified</b> folds for classification, especially with a rare class; k = 5 or 10 is the usual choice.',
    'CV estimates vary with the shuffle, and fold scores are correlated: report their spread and do not treat the fold SD as a standard error.',
    'Use <b>grouped</b> folds whenever samples are related (patients, donors, slides, centres); leave-one-centre-out estimates transfer to a new site.',
    '<b>Every data-driven step</b> (feature selection, scaling, imputation, batch correction, oversampling, tuning) belongs inside the training folds.',
    'If CV also chooses hyperparameters, use <b>nested CV</b> to estimate performance; a permutation of the labels is a direct check for leakage.',
    'CV complements, and does not replace, a locked model evaluated once on an <b>independent cohort</b>.',
  ]);
  prose(ch, '<h3>In code</h3><p>Wrapping every step in a pipeline object ensures that it is refitted inside each training fold.</p>');
  code(ch, {
    Python: `from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GroupKFold, GridSearchCV, cross_val_score
from sklearn.metrics import roc_auc_score

# Scaling and feature selection are inside the pipeline, so they are refitted in every training fold
pipe = make_pipeline(StandardScaler(), SelectKBest(f_classif, k=20), LogisticRegression(max_iter=1000))
outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
print(cross_val_score(pipe, X, y, cv=outer, scoring="roc_auc"))

# Nested CV: GridSearchCV is the inner loop, cross_val_score the outer loop
inner = StratifiedKFold(n_splits=4, shuffle=True, random_state=1)
search = GridSearchCV(pipe, {"selectkbest__k": [1, 2, 5, 10, 20, 50, 100]}, cv=inner, scoring="roc_auc")
nested = cross_val_score(search, X, y, cv=outer, scoring="roc_auc")
print(nested.mean(), nested.min(), nested.max())

# Several samples per patient: keep each patient in one fold (StratifiedGroupKFold also exists)
print(cross_val_score(pipe, X, y, groups=patient_id, cv=GroupKFold(n_splits=5), scoring="roc_auc"))

# Final model: run the whole search on all discovery data, then evaluate once on the external cohort
search.fit(X, y)
print(roc_auc_score(y_ext, search.predict_proba(X_ext)[:, 1]))`,
    R: `library(glmnet); library(pROC)
set.seed(1)
# Stratified outer folds
foldid <- caret::createFolds(factor(y), k = 5, list = FALSE)

# Nested CV: feature selection and lambda are chosen inside each outer training set
auc_outer <- sapply(1:5, function(f) {
  tr <- foldid != f
  tstat <- apply(x[tr, ], 2, function(g) t.test(g[y[tr] == 1], g[y[tr] == 0])$statistic)
  keep <- order(-abs(tstat))[1:20]                       # selection on training patients only
  fit <- cv.glmnet(x[tr, keep], y[tr], family = "binomial", nfolds = 4)  # inner CV for lambda
  p <- predict(fit, x[!tr, keep], s = "lambda.min", type = "response")
  as.numeric(auc(y[!tr], as.vector(p), quiet = TRUE))
})
summary(auc_outer)

# Grouped folds (several samples per patient): one fold id per patient
# pf <- sample(rep(1:5, length.out = length(unique(pid)))); foldid <- pf[match(pid, unique(pid))]
# cv.glmnet(x, y, family = "binomial", foldid = foldid)`,
  });
  seeAlso(main, [['concepts', 'train-test', 'Train/test split and overfitting'], ['concepts', 'logistic-regression', 'Logistic regression'], ['concepts', 'regularization', 'Regularisation'], ['concepts', 'metrics', 'Evaluating a classifier'], ['omics', 'bulk-classifier', 'Biomarker classifier and nested CV']]);
}

// ---------------------------------------------------------------------------------------------
/** Static 5-fold diagram with the real fold sizes, responder counts and fold AUCs. */
function foldDiagram(cv) {
  const wrap = el('div', { class: 'figure' });
  wrap.append(el('div', { class: 'fig-title' }, 'Stratified 5-fold cross-validation of the three-feature model'));
  const W = 820, k = cv.per.length, bx = 70, bw = 560, rh = 30;
  const svg = d3.select(wrap).append('svg').attr('viewBox', `0 0 ${W} ${k * rh + 60}`).attr('width', W).attr('role', 'img').attr('aria-label', 'five rounds of cross-validation');
  const t = (x, yy, s, o = {}) => svg.append('text').attr('x', x).attr('y', yy).attr('font-size', o.size || 12.5).attr('text-anchor', o.anchor || 'start').attr('font-weight', o.bold ? 700 : 400).attr('fill', o.c || ink2()).attr('font-family', o.mono ? css('--mono') : null).text(s);
  t(bx + bw / 2, 14, 'discovery patients, split into 5 folds', { anchor: 'middle', bold: true, c: ink() });
  t(bx + bw + 20, 14, 'test-fold AUC', { bold: true, c: ink() });
  cv.per.forEach((d, r) => {
    const yy = 24 + r * rh;
    t(bx - 10, yy + 17, `round ${r + 1}`, { anchor: 'end' });
    for (let f = 0; f < k; f++) {
      svg.append('rect').attr('x', bx + f * bw / k).attr('y', yy).attr('width', bw / k - 3).attr('height', rh - 6).attr('rx', 3).attr('fill', f === r ? TEST() : TRAIN()).attr('fill-opacity', f === r ? 1 : 0.55);
      if (f === r) t(bx + f * bw / k + (bw / k - 3) / 2, yy + 16, `${d.n} (${d.pos} resp.)`, { anchor: 'middle', size: 11, c: ink() });
    }
    t(bx + bw + 20, yy + 17, d.auc.toFixed(3), { mono: true });
    svg.append('rect').attr('x', bx + bw + 70).attr('y', yy + 6).attr('height', rh - 18).attr('width', Math.max(0, (d.auc - 0.5) * 230)).attr('fill', TEST()).attr('fill-opacity', 0.8);
  });
  const ye = 24 + k * rh + 18;
  t(bx - 10, ye, 'mean', { anchor: 'end', bold: true, c: ink() });
  t(bx, ye, 'each patient is in exactly one test fold; the model is refitted in every round', { size: 12 });
  t(bx + bw + 20, ye, cv.est.toFixed(3), { mono: true, bold: true, c: ink() });
  legend(wrap, [['training folds', TRAIN()], ['test fold', TEST()]]);
  return wrap;
}
