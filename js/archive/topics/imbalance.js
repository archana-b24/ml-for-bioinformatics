// Class imbalance: what prevalence does to accuracy, ROC-AUC and PR-AUC, and what weighting/resampling do to a logistic boundary.
import { header, figure, slider, select, button, stats, section, steps, callout, math, seeAlso, legend, el, tip, untip } from '../../lib/ui.js';
import { frame, scatter, decisionCanvas, cat, css, ink, ink2, muted } from '../../lib/plot.js';
import { rng, solve, sigmoid } from '../../lib/data.js';
const d3 = window.d3;

// ---- normal distribution helpers (analytic curves, so ROC-AUC is exactly prevalence-free) ----
function erf(x) { // Abramowitz & Stegun 7.1.26, |error| < 1.5e-7
  const s = Math.sign(x); x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return s * y;
}
const Phi = (z) => 0.5 * (1 + erf(z / Math.SQRT2));
const pdf = (z) => Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI);
const fmtPct = (p) => `${+(100 * p).toFixed(1)}%`;
const SEP = 1.5; // negatives ~ N(0,1), positives ~ N(1.5,1)
const TS = d3.range(8, -6.0001, -0.01); // thresholds, high -> low
const ROC = TS.map((t) => ({ t, tpr: 1 - Phi(t - SEP), fpr: 1 - Phi(t) }));
const rocAUC = (() => { let a = 0; for (let i = 1; i < ROC.length; i++) a += (ROC[i].fpr - ROC[i - 1].fpr) * (ROC[i].tpr + ROC[i - 1].tpr) / 2; return a; })();
function prCurve(pi) {
  return ROC.map((d) => ({ t: d.t, rec: d.tpr, prec: (pi * d.tpr) / (pi * d.tpr + (1 - pi) * d.fpr + 1e-300) }));
}
function avgPrec(pr) { let a = 0; for (let i = 1; i < pr.length; i++) a += (pr[i].rec - pr[i - 1].rec) * pr[i].prec; return a; }
function precAtRecall(pr, r) { const d = pr.find((p) => p.rec >= r); return d ? d.prec : NaN; }

// ---- weighted logistic regression by IRLS (Newton), tiny ridge on the slopes ----
function fitLogistic(pts, w) {
  let b = [0, 0, 0];
  for (let it = 0; it < 30; it++) {
    const Hm = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], g = [0, 0, 0];
    pts.forEach((p, i) => {
      const x = [1, p.x, p.y], pr = sigmoid(b[0] + b[1] * p.x + b[2] * p.y), wi = w[i];
      for (let a = 0; a < 3; a++) { g[a] += wi * (p.label - pr) * x[a]; for (let c = 0; c < 3; c++) Hm[a][c] += wi * pr * (1 - pr) * x[a] * x[c]; }
    });
    for (let a = 1; a < 3; a++) { Hm[a][a] += 1e-2; g[a] -= 1e-2 * b[a]; }
    const step = solve(Hm, g);
    b = b.map((v, i) => v + step[i]);
    if (Math.hypot(...step) < 1e-8) break;
  }
  return b;
}

export default async function render(main) {
  header(main, 'Class imbalance',
    'When one class is rare, a model can look excellent while being useless: at 1% prevalence, "always say negative" is <b>99% accurate</b>. Imbalance changes which metrics you can trust (<b>ROC-AUC ignores prevalence; precision and PR-AUC do not</b>) and pulls decision boundaries toward the rare class.');

  // ================= Figure 1: prevalence vs metrics =================
  const fig = figure(main, 'Same model, different prevalence',
    'The model is fixed: negatives score N(0, 1), positives score N(1.5, 1). Only the <b>prevalence</b> (fraction of positives) changes. Top left: score distributions drawn to scale, so the positives shrink as they get rarer. Top right: both AUCs across all prevalences, with the current one marked. Bottom: the ROC curve never moves; the PR curve sinks.');
  let pi = 0.1;
  slider(fig.controls, { label: 'Prevalence', min: Math.log10(0.005), max: Math.log10(0.5), step: 0.01, value: -1, fmt: (v) => fmtPct(10 ** v), onInput: (v) => { pi = 10 ** v; update(); } });
  const readout = stats(fig.root);

  const r1 = el('div', { class: 'fig-row' }), r2 = el('div', { class: 'fig-row' });
  fig.body.append(r1, r2);
  const dDiv = el('div'), aDiv = el('div'), rocDiv = el('div'), prDiv = el('div');
  r1.append(dDiv, aDiv); r2.append(rocDiv, prDiv);
  legend(dDiv, [['negatives', cat(0)], ['positives', cat(1)]]);
  legend(aDiv, [['ROC-AUC', cat(2)], ['PR-AUC (average precision)', cat(3)], ['PR-AUC of random guessing (= prevalence)', css('--muted')]]);

  const D = frame(dDiv, { width: 460, height: 300, x: [-4, 5.5], y: [0, 0.42], xLabel: 'model score', yLabel: 'count density (to scale)' });
  D.layer('a');
  const zs = d3.range(-4, 5.501, 0.05);

  const A = frame(aDiv, { width: 460, height: 300, x: [0.005, 0.5], y: [0, 1], xScale: d3.scaleLog(), xLabel: 'prevalence (log scale)', yLabel: 'area under curve' });
  A.g.select('.axis.x').call(d3.axisBottom(A.x).tickValues([0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]).tickFormat((v) => fmtPct(v)));
  const piGrid = d3.range(Math.log10(0.005), Math.log10(0.5) + 1e-9, 0.02).map((l) => 10 ** l);
  const apGrid = piGrid.map((p) => ({ p, ap: avgPrec(prCurve(p)) }));
  const lnA = (f) => d3.line().x((d) => A.x(d.p)).y((d) => A.y(f(d)));
  A.layer('l').append('path').datum(apGrid).attr('fill', 'none').attr('stroke', muted()).attr('stroke-width', 1.5).attr('stroke-dasharray', '4 4').attr('d', lnA((d) => d.p));
  A.layer('l').append('path').datum(apGrid).attr('fill', 'none').attr('stroke', cat(2)).attr('stroke-width', 2).attr('d', lnA(() => rocAUC));
  A.layer('l').append('path').datum(apGrid).attr('fill', 'none').attr('stroke', cat(3)).attr('stroke-width', 2).attr('d', lnA((d) => d.ap));
  A.layer('m');

  const R = frame(rocDiv, { width: 460, height: 340, x: [0, 1], y: [0, 1], xLabel: 'false positive rate', yLabel: 'true positive rate (recall)' });
  R.layer('d').append('line').attr('x1', R.x(0)).attr('y1', R.y(0)).attr('x2', R.x(1)).attr('y2', R.y(1)).attr('stroke', muted()).attr('stroke-dasharray', '4 4');
  R.layer('c').append('path').datum(ROC).attr('fill', 'none').attr('stroke', cat(2)).attr('stroke-width', 2).attr('d', d3.line().x((d) => R.x(d.fpr)).y((d) => R.y(d.tpr)));
  R.layer('t').append('text').attr('x', R.x(0.3)).attr('y', R.y(0.12)).attr('fill', ink2()).attr('font-size', 12).text(`ROC-AUC = ${rocAUC.toFixed(3)} at every prevalence`);

  const P = frame(prDiv, { width: 460, height: 340, x: [0, 1], y: [0, 1], xLabel: 'recall', yLabel: 'precision' });
  P.layer('base'); P.layer('c'); P.layer('m');

  function update() {
    const pr = prCurve(pi), ap = avgPrec(pr);
    // densities to scale
    const ar = (k, w) => d3.area().x((z) => D.x(z)).y0(D.y(0)).y1((z) => D.y(w * pdf(z - k)));
    const ser = [{ c: 0, k: 0, w: 1 - pi }, { c: 1, k: SEP, w: pi }];
    D.layer('a').selectAll('path.f').data(ser).join('path').attr('class', 'f').attr('fill', (d) => cat(d.c)).attr('fill-opacity', 0.25).attr('d', (d) => ar(d.k, d.w)(zs));
    D.layer('a').selectAll('path.l').data(ser).join('path').attr('class', 'l').attr('fill', 'none').attr('stroke', (d) => cat(d.c)).attr('stroke-width', 2)
      .attr('d', (d) => d3.line().x((z) => D.x(z)).y((z) => D.y(d.w * pdf(z - d.k)))(zs));
    // AUC vs prevalence marker
    const m = A.layer('m');
    m.selectAll('line').data([0]).join('line').attr('x1', A.x(pi)).attr('x2', A.x(pi)).attr('y1', 0).attr('y2', A.h).attr('stroke', ink()).attr('stroke-width', 1).attr('stroke-dasharray', '2 3');
    m.selectAll('circle').data([[rocAUC, 2], [ap, 3], [pi, -1]]).join('circle').attr('cx', A.x(pi)).attr('cy', (d) => A.y(d[0])).attr('r', 5.5)
      .attr('fill', (d) => (d[1] < 0 ? muted() : cat(d[1]))).attr('stroke', css('--surface')).attr('stroke-width', 1.5);
    // PR curve
    P.layer('base').selectAll('line').data([0]).join('line').attr('x1', P.x(0)).attr('x2', P.x(1)).attr('y1', P.y(pi)).attr('y2', P.y(pi)).attr('stroke', muted()).attr('stroke-dasharray', '4 4');
    P.layer('base').selectAll('text').data([0]).join('text').attr('x', P.x(0.02)).attr('y', P.y(pi) - 6).attr('text-anchor', 'start').attr('fill', muted()).attr('font-size', 11).text(`random guessing (${fmtPct(pi)})`);
    P.layer('c').selectAll('path').data([pr.filter((d) => d.rec > 1e-4)]).join('path').attr('fill', 'none').attr('stroke', cat(3)).attr('stroke-width', 2)
      .attr('d', d3.line().x((d) => P.x(d.rec)).y((d) => P.y(d.prec)));
    const p80 = precAtRecall(pr, 0.8);
    P.layer('m').selectAll('circle').data([0]).join('circle').attr('cx', P.x(0.8)).attr('cy', P.y(p80)).attr('r', 6).attr('fill', cat(3)).attr('stroke', css('--surface')).attr('stroke-width', 1.5)
      .on('mousemove', (e) => tip(e, `At 80% recall, precision = ${p80.toFixed(3)}<br>i.e. ${Math.round(100 * (1 - p80))} of every 100 calls are false positives`)).on('mouseleave', untip);
    P.layer('m').selectAll('text').data([0]).join('text').attr('x', P.x(0.8) - 10).attr('y', P.y(p80) + 18).attr('text-anchor', 'end').attr('fill', ink2()).attr('font-size', 11).text(`precision at 80% recall: ${p80.toFixed(2)}`);
    const n = 1000, npos = Math.round(n * pi);
    readout.set({ prevalence: fmtPct(pi), 'positives per 1,000': npos, '"always negative" accuracy': fmtPct(1 - pi), 'ROC-AUC': rocAUC.toFixed(3), 'PR-AUC': ap.toFixed(3), 'random PR-AUC': pi.toFixed(3), 'precision @ 80% recall': p80.toFixed(3) });
  }
  update();

  // ================= Figure 2: weighting / resampling a logistic boundary =================
  const fig2 = figure(main, 'Re-weighting and resampling a logistic regression',
    'Training data: 400 majority (class 0) and 24 minority (class 1) samples, about 5.7% prevalence. Each line is the p = 0.5 decision boundary of a logistic regression trained a different way; shading shows the selected one. The table scores every option on a separate test set of 4,000 samples with the same prevalence.');
  const METHODS = {
    none: { name: 'Plain fit', c: 2 },
    weight: { name: 'Class weights (balanced)', c: 3 },
    over: { name: 'Random oversampling', c: 6, dash: '6 5' },
    under: { name: 'Random undersampling', c: 4 },
    thresh: { name: 'Plain fit, threshold = prevalence', c: 7, dash: '2 4' },
  };
  let method = 'weight', underSeed = 1;
  select(fig2.controls, { label: 'Shade', value: method, options: Object.entries(METHODS).map(([k, v]) => [k, v.name]), onChange: (k) => { method = k; paint2(); } });
  button(fig2.controls, 'Redraw undersample', () => { underSeed++; fitAll(); paint2(); });
  const readout2 = stats(fig2.root);

  const gen = (n0, n1, seed) => {
    const r = rng(seed), out = [];
    for (let i = 0; i < n0; i++) out.push({ x: r.normal(-0.25, 0.32), y: r.normal(-0.15, 0.32), label: 0 });
    for (let i = 0; i < n1; i++) out.push({ x: r.normal(0.3, 0.28), y: r.normal(0.3, 0.28), label: 1 });
    return out;
  };
  const train = gen(400, 24, 21), test = gen(3771, 229, 77);
  const n0 = 400, n1 = 24;

  const row = el('div', { class: 'fig-row' });
  fig2.body.append(row);
  const pDiv = el('div'), tDiv = el('div');
  row.append(pDiv, tDiv);
  legend(pDiv, [['class 0 (majority)', cat(0)], ['class 1 (minority)', cat(1)], ...Object.values(METHODS).map((m) => [m.name, cat(m.c)])]);
  const F = frame(pDiv, { width: 480, height: 440, x: [-1.3, 1.3], y: [-1.3, 1.3], xLabel: 'feature 1', yLabel: 'feature 2', grid: false });
  let canvas = null;
  F.layer('pts'); F.layer('lines');
  const table = el('table', { class: 'simple' });
  tDiv.append(table);
  tDiv.append(el('p', { class: 'fig-caption', html: 'Recall = fraction of test minority samples caught. Precision = fraction of minority calls that are right. Balanced accuracy = mean of recall and specificity. Click a row to shade it.' }));

  let fits = {}, kept = null;
  function fitAll() {
    fits.none = fitLogistic(train, train.map(() => 1));
    fits.weight = fitLogistic(train, train.map((p) => (p.label ? n0 / n1 : 1)));
    const k = Math.round(n0 / n1); // duplicate each minority sample k times
    const over = [...train.filter((p) => !p.label), ...train.filter((p) => p.label).flatMap((p) => Array(k).fill(p))];
    fits.over = fitLogistic(over, over.map(() => 1));
    const r = rng(1000 + underSeed);
    const maj = r.shuffle(train.filter((p) => !p.label)).slice(0, n1);
    kept = new Set(maj);
    const under = [...maj, ...train.filter((p) => p.label)];
    fits.under = fitLogistic(under, under.map(() => 1));
    // plain fit, but call positive when p >= training prevalence (shift the logit by -log(pi/(1-pi)))
    const piT = n1 / (n0 + n1);
    fits.thresh = fits.none.slice(); fits.thresh[0] -= Math.log(piT / (1 - piT));
  }
  const prob = (b, x, y) => sigmoid(b[0] + b[1] * x + b[2] * y);
  function score(b) {
    let tp = 0, fp = 0, fn = 0, tn = 0;
    test.forEach((p) => { const c = prob(b, p.x, p.y) >= 0.5; if (p.label) { if (c) tp++; else fn++; } else if (c) fp++; else tn++; });
    const rec = tp / (tp + fn), spec = tn / (tn + fp), prec = tp + fp ? tp / (tp + fp) : NaN;
    return { rec, spec, prec, f1: tp ? 2 * prec * rec / (prec + rec) : 0, bacc: (rec + spec) / 2, acc: (tp + tn) / test.length, calls: tp + fp };
  }

  function paint2() {
    const b = fits[method];
    if (!canvas) canvas = decisionCanvas(F, (x, y) => prob(b, x, y), { res: 5, alpha: 0.22 });
    else canvas.draw((x, y) => prob(b, x, y));
    const pts = [...train.filter((p) => !p.label), ...train.filter((p) => p.label)];
    scatter(F.layer('pts'), pts, F.x, F.y, { r: 4.5 })
      .attr('opacity', (d) => (method === 'under' && !d.label && !kept.has(d) ? 0.15 : 1))
      .attr('r', (d) => (d.label ? 5.5 : 4.2))
      .on('mousemove', (e, d) => tip(e, `class ${d.label}<br>P(class 1) under ${METHODS[method].name.toLowerCase()}: ${prob(b, d.x, d.y).toFixed(2)}`)).on('mouseleave', untip);
    // boundaries: b0 + b1 x + b2 y = 0
    const segs = Object.entries(METHODS).map(([k, m]) => {
      const f = fits[k], x0 = -1.3, x1 = 1.3;
      return { k, m, p: [[x0, -(f[0] + f[1] * x0) / f[2]], [x1, -(f[0] + f[1] * x1) / f[2]]] };
    });
    F.layer('lines').selectAll('path').data(segs).join('path').attr('fill', 'none').attr('stroke', (d) => cat(d.m.c))
      .attr('stroke-width', (d) => (d.k === method ? 3.5 : 2)).attr('stroke-dasharray', (d) => d.m.dash || null)
      .attr('d', (d) => d3.line().x((p) => F.x(p[0])).y((p) => F.y(p[1]))(d.p));
    F.g.selectAll('.clipr').data([0]).join('clipPath').attr('class', 'clipr').attr('id', 'imb-clip').selectAll('rect').data([0]).join('rect').attr('width', F.w).attr('height', F.h);
    F.layer('lines').attr('clip-path', 'url(#imb-clip)');
    // table
    const f = (v) => (isNaN(v) ? '—' : v.toFixed(2));
    const rows = Object.entries(METHODS).map(([k, m]) => ({ k, m, s: score(fits[k]) }));
    table.innerHTML = '';
    table.append(el('tr', {}, ['training', 'recall', 'precision', 'F1', 'bal. acc.', 'accuracy'].map((h) => el('th', {}, h))));
    rows.forEach(({ k, m, s }) => {
      const tr = el('tr', { style: `cursor:pointer;${k === method ? 'background:var(--accent-soft);font-weight:600' : ''}` },
        el('td', {}, el('span', { style: `display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;background:${cat(m.c)}` }), m.name),
        [s.rec, s.prec, s.f1, s.bacc, s.acc].map((v) => el('td', { style: 'font-variant-numeric:tabular-nums' }, f(v))));
      tr.addEventListener('click', () => { method = k; sel2.value = k; paint2(); });
      table.append(tr);
    });
    const s = score(b);
    readout2.set({ 'test recall': f(s.rec), 'test precision': f(s.prec), 'minority calls (of 4,000)': s.calls, 'test accuracy': f(s.acc), 'always-majority accuracy': (1 - 229 / 4000).toFixed(2) });
  }
  const sel2 = fig2.controls.querySelector('select');
  fitAll(); paint2();

  section(main, 'What to notice', `<ul>
    <li><b>Drag prevalence from 50% down to 0.5%.</b> The ROC curve and ROC-AUC (${rocAUC.toFixed(3)}) do not move at all, because TPR and FPR are each computed within one class. PR-AUC falls from 0.853 at 50% to 0.066 at 0.5%, because precision mixes the two classes: at low prevalence even a small false-positive rate swamps the few true positives.</li>
    <li>The useful baseline for PR-AUC is the prevalence (dashed line), not 0.5. At 1% prevalence the PR-AUC of 0.115 is more than 10× better than random, yet at 80% recall the precision is 0.030: 97 of every 100 positive calls are wrong.</li>
    <li>"Always negative" accuracy is 1 − prevalence. Any model with accuracy below that is worse than doing nothing.</li>
    <li>In the second figure the <b>plain fit</b> has the best accuracy (0.95) but that is barely above the 0.94 of always predicting the majority, and it catches only 48% of the minority test samples: its boundary sits inside the minority cloud. Class weights, oversampling and undersampling all push the boundary <b>toward the majority</b>, trading precision for recall: with class weights, recall rises to 0.87 and precision falls from 0.63 to 0.31.</li>
    <li><b>Oversampling by duplication lies on top of class weighting</b> (the dashed line on the solid one): for logistic regression, duplicating each minority sample k times is exactly the same as giving it weight k. Here k = 17 copies versus a weight of 400/24 ≈ 16.7, so the two lines are almost identical.</li>
    <li>The <b>threshold = prevalence</b> line lands close to the weighted fit. Re-weighting mostly shifts the intercept, so you can often get the same effect by keeping the plain model and moving the threshold. That also keeps its probabilities calibrated.</li>
    <li>Press <b>Redraw undersample</b> a few times: throwing away 94% of the majority makes the boundary jump around. Across the first few redraws its test precision moves between 0.25 and 0.35. That's the variance you pay for discarding data.</li>
  </ul>`);

  steps(main, 'How it works', [
    'Measure prevalence first, and report the baseline accuracy (1 − prevalence) and baseline PR-AUC (= prevalence) next to your results.',
    'Use metrics that match the question: recall at a fixed specificity, precision at a fixed recall, PR-AUC, or balanced accuracy, rather than plain accuracy.',
    '<b>Class weights</b>: multiply each sample\'s loss by a weight, usually n_total / (n_classes · n_class). Rare-class mistakes then cost more, so the boundary moves toward the majority.',
    '<b>Resampling</b>: oversample the minority (duplicate it, or synthesise new points with SMOTE) or undersample the majority. Do it <b>only inside the training folds</b>, never before splitting.',
    '<b>Threshold moving</b>: train normally, then choose the decision threshold on validation data to hit the recall or precision you need. This is often the simplest fix.',
  ]);
  math(main, 'precision = π·TPR / (π·TPR + (1 − π)·FPR)        π = prevalence\nTPR and FPR do not depend on π  ⇒  ROC curve and ROC-AUC do not either\nweighted log-loss  = − Σ wᵢ [yᵢ log pᵢ + (1 − yᵢ) log(1 − pᵢ)],   w = n₀/n₁ for class 1');

  callout(main, '', 'In biology', 'Imbalance is the rule in omics. <b>Rare cell types</b> in scRNA-seq (e.g. pDCs or ILCs at under 1% of PBMCs) are where classifiers such as CellTypist or scANVI lose accuracy, yet overall accuracy stays high. <b>Doublets</b> are roughly 5–10% of droplets. <b>Responders</b> in an immunotherapy biomarker cohort may be only 20–25% of patients. In <b>variant</b> and <b>enhancer</b> prediction, true positives can be under 1% of candidate sites, which is why papers on those report PR-AUC (auPRC) rather than only ROC-AUC.');
  callout(main, 'warn', 'Pitfalls', '<ul><li>Oversampling or SMOTE <b>before</b> the train/test split leaks copies of test samples into training and inflates scores.</li><li>Weighting or resampling distorts predicted probabilities: the model now believes the classes are balanced. Recalibrate, or move the threshold instead, if you need real probabilities.</li><li>A PR-AUC from a case–control study (50% prevalence) cannot be carried over to a screening population (1%). Recompute precision at the deployment prevalence.</li><li>With very few minority samples (24 here), every metric on the minority class is noisy. Report confidence intervals.</li></ul>');
  seeAlso(main, [['concepts', 'metrics', 'Classification metrics'], ['concepts', 'calibration', 'Calibration'], ['concepts', 'logistic-regression', 'Logistic regression'], ['concepts', 'cross-validation', 'Cross-validation']]);
}
