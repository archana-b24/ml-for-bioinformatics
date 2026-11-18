// Bulk step 7: logistic (lasso) vs random forest vs gradient boosting vs a one-feature prespecified score,
// compared under the SAME nested cross-validation folds; discrimination (ROC) and calibration (reliability).
import { header, figure, slider, select, stats, section, steps, callout, math, legend, seeAlso, tip, untip, el } from '../../lib/ui.js';
import { frame, cat, css, ink2, muted } from '../../lib/plot.js';
import { stepCard, omicsData } from '../../omics/_shared.js';
const d3 = window.d3;

const ORDER = ['ifng', 'lasso', 'rf', 'gb'];
const SHORT = { ifng: 'IFN-γ score', lasso: 'Lasso', rf: 'Random forest', gb: 'Boosting' };
const f2 = (v) => v.toFixed(2), f3 = (v) => v.toFixed(3);

function rocOf(y, p) {
  const idx = d3.range(y.length).sort((a, b) => p[b] - p[a]);
  const P = d3.sum(y), N = y.length - P; let tp = 0, fp = 0; const pts = [{ x: 0, y: 0 }];
  idx.forEach((i, k) => { if (y[i]) tp++; else fp++; if (k === idx.length - 1 || p[idx[k + 1]] !== p[i]) pts.push({ x: fp / N, y: tp / P }); });
  return pts;
}
function relBins(y, p, k) {
  const b = d3.range(k).map(() => ({ n: 0, sp: 0, sy: 0 }));
  p.forEach((v, i) => { const j = Math.min(k - 1, Math.floor(v * k)); b[j].n++; b[j].sp += v; b[j].sy += y[i]; });
  return b.map((o, j) => ({ j, n: o.n, mp: o.n ? o.sp / o.n : null, obs: o.n ? o.sy / o.n : null, lo: j / k, hi: (j + 1) / k })).filter((o) => o.n > 0);
}

export default async function render(main) {
  header(main, 'Model comparison & calibration',
    'Is a random forest or gradient boosting better than a penalised logistic regression for predicting response? Compare them on <b>exactly the same nested cross-validation folds</b>, look at the spread across repeats, and check <b>calibration</b> as well as AUC: a model can rank patients well and still give wrong probabilities.');
  const D = await omicsData(main, 'bulk_compare.json');
  if (!D) return;
  const M = D.models, y = D.y;
  const mean = (k) => d3.mean(M[k].auc), sdv = (k) => d3.deviation(M[k].auc);
  const best = ORDER.slice().sort((a, b) => mean(b) - mean(a))[0];
  const spread = d3.max(ORDER, mean) - d3.min(ORDER, mean);

  stepCard(main, {
    stage: 'Bulk RNA-seq · step 7 of 11',
    question: 'Does a flexible non-linear learner predict anti-PD-1 response better than a sparse linear model, and are its probabilities trustworthy?',
    data: `${D.n} discovery patients (${D.n_resp} responders), ${D.p.toLocaleString()} batch-corrected genes; ${D.reps} repeats of 5-fold outer CV, 3-fold inner CV for tuning.`,
    method: 'Lasso logistic, random forest, gradient boosting (each tuned in the inner loop) and a prespecified one-feature IFN-γ score; ROC, Brier score, calibration slope.',
    verdict: 'both',
    why: 'The candidates are ML models, but deciding between them is a statistical question: paired comparison on shared folds, uncertainty across repeats, and calibration checks.',
  });

  // ---------------- Figure 1: ROC + AUC across repeats
  const f1 = figure(main, 'Discrimination: ROC curves and AUC across repeats',
    'Left: out-of-fold ROC curves from repeat 1 (every patient is predicted by a model that never saw them). Right: one dot per repeat of the whole nested CV; the bar is the mean. Hover a dot for its value.');
  let hl = 'all';
  select(f1.controls, { label: 'Highlight', value: 'all', options: [['all', 'All models'], ...ORDER.map((k) => [k, M[k].label])], onChange: (v) => { hl = v; draw1(); } });
  legend(f1.root, ORDER.map((k, i) => [M[k].label, cat(i)]));
  const s1 = stats(f1.root);
  const row1 = el('div', { class: 'fig-row' }); f1.body.append(row1);
  const L1 = el('div'), R1 = el('div'); row1.append(L1, R1);
  const rocs = Object.fromEntries(ORDER.map((k) => [k, rocOf(y, M[k].p)]));
  const F = frame(L1, { width: 440, height: 400, x: [0, 1], y: [0, 1], xLabel: 'false-positive rate (1 − specificity)', yLabel: 'true-positive rate (sensitivity)' });
  F.layer('diag').append('line').attr('x1', F.x(0)).attr('y1', F.y(0)).attr('x2', F.x(1)).attr('y2', F.y(1)).attr('stroke', muted()).attr('stroke-dasharray', '4 4');
  const aLo = Math.floor((d3.min(ORDER, (k) => d3.min(M[k].auc)) - 0.02) * 50) / 50, aHi = Math.ceil((d3.max(ORDER, (k) => d3.max(M[k].auc)) + 0.02) * 50) / 50;
  const G = frame(R1, { width: 440, height: 400, x: [-0.5, 3.5], y: [aLo, aHi], yLabel: 'out-of-fold AUC (per repeat)', xTicks: 4, margin: { bottom: 46 } });
  G.g.select('.axis.x').call(d3.axisBottom(G.x).tickValues([0, 1, 2, 3]).tickFormat((i) => SHORT[ORDER[i]]));

  function draw1() {
    const op = (k) => (hl === 'all' || hl === k ? 1 : 0.15);
    F.layer('roc').selectAll('path').data(ORDER).join('path').attr('fill', 'none').attr('stroke-width', (k) => (hl === k ? 3 : 2))
      .attr('stroke', (k) => cat(ORDER.indexOf(k))).attr('opacity', op)
      .attr('d', (k) => d3.line().curve(d3.curveStepAfter).x((d) => F.x(d.x)).y((d) => F.y(d.y))(rocs[k]));
    const pts = ORDER.flatMap((k, i) => M[k].auc.map((a, r) => ({ k, i, r, a })));
    G.layer('bars').selectAll('line').data(ORDER).join('line').attr('x1', (k, i) => G.x(i) - 0.28 * (G.x(1) - G.x(0))).attr('x2', (k, i) => G.x(i) + 0.28 * (G.x(1) - G.x(0)))
      .attr('y1', (k) => G.y(mean(k))).attr('y2', (k) => G.y(mean(k))).attr('stroke', (k, i) => cat(i)).attr('stroke-width', 3).attr('opacity', op);
    G.layer('dots').selectAll('circle').data(pts).join('circle').attr('cx', (d) => G.x(d.i) + (d.r - (D.reps - 1) / 2) * 9).attr('cy', (d) => G.y(d.a)).attr('r', 5)
      .attr('fill', (d) => cat(d.i)).attr('stroke', css('--surface')).attr('stroke-width', 1.5).attr('opacity', (d) => op(d.k))
      .on('mousemove', (e, d) => tip(e, `<b>${M[d.k].label}</b><br>repeat ${d.r + 1}: AUC ${f3(d.a)}`)).on('mouseleave', untip);
    const o = {};
    (hl === 'all' ? ORDER : [hl]).forEach((k) => { o[`${SHORT[k]} AUC`] = `${f3(mean(k))} ± ${f3(sdv(k))}`; });
    if (hl !== 'all' && hl !== 'lasso') {
      const dd = D.diff_vs_lasso[hl];
      o['paired Δ vs lasso'] = `${d3.mean(dd) >= 0 ? '+' : ''}${f3(d3.mean(dd))} (${d3.sum(dd, (v) => v > 0)}/${dd.length} repeats better)`;
    }
    s1.set(o);
  }
  draw1();

  // ---------------- Figure 2: calibration + threshold
  const f2g = figure(main, 'Calibration: when a model says 60%, do 60% respond?',
    'Left: reliability diagram. Patients are binned by predicted probability; each dot is a bin (size = number of patients), placed at the mean prediction (x) and the observed response rate (y). On the diagonal = calibrated. Right: the predicted probabilities themselves, responders above the axis, non-responders below; the dashed line is the decision threshold.');
  let mk = 'rf', nb = 8, thr = 0.5;
  select(f2g.controls, { label: 'Model', value: mk, options: ORDER.map((k) => [k, M[k].label]), onChange: (v) => { mk = v; draw2(); } });
  slider(f2g.controls, { label: 'Bins', min: 4, max: 12, step: 1, value: nb, onInput: (v) => { nb = v; draw2(); } });
  slider(f2g.controls, { label: 'Threshold', min: 0.05, max: 0.95, step: 0.01, value: thr, fmt: f2, onInput: (v) => { thr = v; draw2(); } });
  const s2 = stats(f2g.root);
  const row2 = el('div', { class: 'fig-row' }); f2g.body.append(row2);
  const L2 = el('div'), R2 = el('div'); row2.append(L2, R2);
  const C = frame(L2, { width: 440, height: 400, x: [0, 1], y: [0, 1], xLabel: 'mean predicted probability (bin)', yLabel: 'observed response rate (bin)' });
  C.layer('diag').append('line').attr('x1', C.x(0)).attr('y1', C.y(0)).attr('x2', C.x(1)).attr('y2', C.y(1)).attr('stroke', muted()).attr('stroke-dasharray', '4 4');
  const H = frame(R2, { width: 440, height: 400, x: [0, 1], y: [-1, 1], xLabel: 'predicted probability of response', yLabel: '', grid: false, margin: { left: 20 } });
  H.g.select('.axis.y').remove();
  H.g.append('line').attr('x1', 0).attr('x2', H.w).attr('y1', H.y(0)).attr('y2', H.y(0)).attr('stroke', css('--axis'));
  H.g.append('text').attr('x', 4).attr('y', 14).attr('fill', ink2()).attr('font-size', 12).text('responders ↑');
  H.g.append('text').attr('x', 4).attr('y', H.h - 4).attr('fill', ink2()).attr('font-size', 12).text('non-responders ↓');

  function draw2() {
    const p = M[mk].p, ci = ORDER.indexOf(mk);
    const bins = relBins(y, p, nb);
    const rs = d3.scaleSqrt().domain([0, d3.max(bins, (b) => b.n)]).range([4, 14]);
    C.layer('line').selectAll('path').data([bins]).join('path').attr('fill', 'none').attr('stroke', cat(ci)).attr('stroke-width', 2)
      .attr('d', d3.line().x((b) => C.x(b.mp)).y((b) => C.y(b.obs)));
    C.layer('pts').selectAll('circle').data(bins).join('circle').attr('cx', (b) => C.x(b.mp)).attr('cy', (b) => C.y(b.obs)).attr('r', (b) => rs(b.n))
      .attr('fill', cat(ci)).attr('stroke', css('--surface')).attr('stroke-width', 1.5)
      .on('mousemove', (e, b) => tip(e, `bin ${f2(b.lo)}–${f2(b.hi)}<br>${b.n} patients<br>mean predicted ${f2(b.mp)}<br>observed rate ${f2(b.obs)}`)).on('mouseleave', untip);
    // dot histogram of predictions
    const w = 0.025, stackU = {}, stackD = {};
    const pts = p.map((v, i) => {
      const b = Math.min(39, Math.floor(v / w)), st = y[i] ? stackU : stackD; st[b] = (st[b] || 0) + 1;
      return { v, y: y[i], b, s: st[b], i };
    });
    const maxS = Math.max(d3.max(Object.values(stackU)), d3.max(Object.values(stackD)));
    const dy = Math.min(9, (H.h / 2 - 22) / maxS);
    H.layer('d').selectAll('circle').data(pts).join('circle').attr('cx', (d) => H.x((d.b + 0.5) * w)).attr('cy', (d) => H.y(0) + (d.y ? -1 : 1) * (6 + (d.s - 0.5) * dy))
      .attr('r', Math.max(2.2, dy / 2 - 0.3)).attr('fill', (d) => (d.y ? cat(1) : cat(0))).attr('opacity', (d) => ((d.v >= thr) ? 1 : 0.45))
      .on('mousemove', (e, d) => tip(e, `${d.y ? 'responder' : 'non-responder'}<br>predicted ${f3(d.v)}`)).on('mouseleave', untip);
    H.layer('t').selectAll('line').data([thr]).join('line').attr('x1', (t) => H.x(t)).attr('x2', (t) => H.x(t)).attr('y1', 0).attr('y2', H.h)
      .attr('stroke', ink2()).attr('stroke-dasharray', '5 4').attr('stroke-width', 1.5);
    const tp = d3.sum(pts, (d) => d.y && d.v >= thr), fp = d3.sum(pts, (d) => !d.y && d.v >= thr);
    const P = d3.sum(y), N = y.length - P, r0 = M[mk].rep0;
    s2.set({ 'AUC': f3(r0.auc), 'Brier score': f3(r0.brier), 'calibration slope': f2(r0.slope), 'mean predicted vs observed': `${f2(r0.mean_p)} vs ${f2(r0.obs)}`,
      'sensitivity': `${tp}/${P} = ${f2(tp / P)}`, 'specificity': `${N - fp}/${N} = ${f2((N - fp) / N)}`, 'PPV': tp + fp ? f2(tp / (tp + fp)) : '–' });
  }
  draw2();

  // ---------------- Figure 3: optimism of picking the best config after looking
  const f3g = figure(main, 'The winner\'s curse: picking the best configuration after looking',
    'Grey bars: plain (non-nested) CV AUC of every hyperparameter setting, averaged over repeats. Reporting the tallest bar is what happens when you tune on the same folds you report. The coloured line is the honest nested-CV AUC of the whole tuning procedure.');
  const row3 = el('div'); f3g.body.append(row3);
  const tuned = ['lasso', 'rf', 'gb'];
  const allCfg = tuned.flatMap((k) => M[k].grid.map((g, j) => ({ k, j, g, v: M[k].plain_mean[j], picked: M[k].picked_counts[j] })));
  const lo3 = Math.floor((d3.min(allCfg, (d) => d.v) - 0.03) * 20) / 20, hi3 = Math.ceil((d3.max(allCfg, (d) => d.v) + 0.02) * 20) / 20;
  const B = frame(row3, { width: 900, height: 300, x: [0, allCfg.length + tuned.length - 1], y: [lo3, hi3], yLabel: 'AUC', axes: true, margin: { bottom: 34 } });
  B.g.select('.axis.x').remove();
  let pos = 0; const xs = [];
  tuned.forEach((k, t) => { M[k].grid.forEach(() => xs.push(pos++)); pos++; });
  const bw = (B.x(1) - B.x(0)) * 0.8;
  B.layer('b').selectAll('rect').data(allCfg).join('rect').attr('x', (d, i) => B.x(xs[i]) + (B.x(1) - B.x(0)) * 0.1).attr('width', bw)
    .attr('y', (d) => B.y(d.v)).attr('height', (d) => B.y(lo3) - B.y(d.v)).attr('fill', css('--surface-2')).attr('stroke', muted())
    .on('mousemove', (e, d) => tip(e, `<b>${M[d.k].label}</b><br>${d.g}<br>plain CV AUC ${f3(d.v)}<br>chosen by the inner loop in ${d.picked} of ${D.reps * 5} outer folds`)).on('mouseleave', untip);
  tuned.forEach((k) => {
    const ii = allCfg.map((d, i) => (d.k === k ? i : -1)).filter((i) => i >= 0);
    const x0 = B.x(xs[ii[0]]), x1 = B.x(xs[ii[ii.length - 1]] + 1), ci = ORDER.indexOf(k);
    B.layer('n').append('line').attr('x1', x0).attr('x2', x1).attr('y1', B.y(mean(k))).attr('y2', B.y(mean(k))).attr('stroke', cat(ci)).attr('stroke-width', 3);
    B.layer('n').append('text').attr('x', (x0 + x1) / 2).attr('y', B.h + 22).attr('text-anchor', 'middle').attr('fill', ink2()).attr('font-size', 12).text(M[k].label);
  });
  const optim = tuned.map((k) => ({ k, best: d3.max(M[k].plain_mean), nested: mean(k) }));
  const s3 = stats(f3g.root);
  const o3 = {}; optim.forEach((o) => { o3[`${SHORT[o.k]}: best-looking vs nested`] = `${f3(o.best)} vs ${f3(o.nested)}`; }); s3.set(o3);

  // ---------------- prose (numbers computed from the data)
  const cal = (k) => M[k].rep0;
  const overlap = ORDER.filter((k) => k !== best).filter((k) => d3.max(M[k].auc) >= d3.min(M[best].auc));
  section(main, 'What to notice', `<ul>
    <li>The mean AUCs span only <b>${f3(spread)}</b> (from ${f3(d3.min(ORDER, mean))} to ${f3(d3.max(ORDER, mean))}). The highest mean belongs to the <b>${best === 'ifng' ? 'prespecified one-feature IFN-γ score, which involves no learning at all' : M[best].label.toLowerCase()}</b>, but ${overlap.length ? overlap.map((k) => M[k].label.toLowerCase()).join(', ') + ' overlap' + (overlap.length === 1 ? 's' : '') + ' with it across repeats' : 'the gaps are small relative to the repeat-to-repeat spread'}. With ${D.n} patients, differences of a few hundredths are noise.</li>
    <li>The one-feature <b>IFN-γ score</b> (the mean of the prespecified interferon-module genes, with no gene selection at all) reaches ${f3(mean('ifng'))}. When a simple, biologically motivated score is this close, it is usually the better biomarker: cheaper to measure, easier to validate, harder to overfit.</li>
    <li>Select <b>Highlight → Random forest</b>: the paired difference against the lasso is computed on identical folds, which removes fold-to-fold noise from the comparison.</li>
    <li>In the calibration panel, switch between models. The random forest has calibration slope ${f2(cal('rf').slope)} and gradient boosting ${f2(cal('gb').slope)} (ideal 1; above 1 = predictions too timid, below 1 = too extreme). The lasso has ${f2(cal('lasso').slope)}. Similar AUCs, different probabilities.</li>
    <li>Drag the <b>threshold</b>: sensitivity and specificity trade off, and the positive predictive value depends on the ${f2(D.n_resp / D.n)} response rate, not on the model alone.</li>
    <li>In the last figure the tallest grey bar is at or above the honest nested estimate for every learner (${optim.map((o) => `${SHORT[o.k]} ${f3(o.best)} vs ${f3(o.nested)}`).join('; ')}). The gap is small here because the grids are small; with dozens of configurations and many learners it grows. Comparing many models and reporting the winner is another form of the leakage in step 6.</li>
  </ul>`);

  steps(main, 'How it works', [
    'Fix the <b>outer folds</b> once per repeat (5-fold, stratified by response) and use them for every model, so the comparison is paired.',
    'Inside each outer training set, an <b>inner 3-fold CV</b> picks the hyperparameters (lasso strength; forest <code>max_features</code> and leaf size; boosting learning rate and depth). Standardisation is also learned on the training fold only.',
    'Refit the chosen configuration on the whole outer training set and predict the held-out fold. After 5 folds every patient has one honest prediction.',
    'Score the out-of-fold predictions: <b>AUC</b> (ranking), <b>Brier score</b> (mean squared error of the probability), and the <b>calibration slope</b> (logistic regression of the outcome on logit(prediction)).',
    'Repeat with new fold splits to see how much the estimate moves. Prefer the simplest model whose performance is within that noise.',
  ]);
  math(main, 'Brier = mean( (p_i − y_i)² )               lower is better; 0.25 = always predicting 0.5\ncalibration:  logit P(y=1) = a + b · logit(p)     calibrated ⇔ a = 0, b = 1\nb < 1: predictions too extreme (overfit)          b > 1: predictions too timid');

  callout(main, '', 'In biology', 'Published immunotherapy response models (for example on TCGA-like or IMvigor210 bulk RNA-seq) routinely report one AUC for the winning learner. Better practice: <code>scikit-learn</code> <code>cross_val_predict</code> inside <code>GridSearchCV</code> (nested), <code>CalibrationDisplay</code> for reliability curves, or <code>tidymodels</code>/<code>mlr3</code> in R with resampling-based benchmarking. Tree ensembles shine when there are real interactions and many samples (thousands); with ~180 patients and a mostly additive signal, a penalised linear model or a prespecified gene-set score is usually as good and easier to defend. Report calibration as the TRIPOD+AI guideline asks.');
  callout(main, 'warn', 'Pitfalls', `<ul>
    <li><b>Unpaired comparison</b>: running each model with different random folds adds noise that can flip the ranking.</li>
    <li><b>Reporting the best of many</b>: trying ten learners and reporting the top AUC is optimistic, exactly like tuning on the test set.</li>
    <li><b>AUC is blind to calibration</b>: any monotone transform of the scores gives the same AUC. If clinicians will read the probability, check the reliability diagram.</li>
    <li><b>Random forest probabilities</b> are vote fractions, often squeezed toward the middle; boosting can be over-confident. Recalibrate (Platt or isotonic) <i>inside</i> the CV if you need probabilities.</li>
    <li><b>Small test folds</b>: with ${Math.round(D.n_resp / 5)} responders per outer fold, a single fold AUC is very noisy; use pooled out-of-fold predictions and repeats.</li>
  </ul>`);
  seeAlso(main, [['omics', 'bulk-classifier', 'Response classifier & nested CV'], ['concepts', 'calibration', 'Calibration'], ['concepts', 'random-forest', 'Random forests'], ['concepts', 'gradient-boosting', 'Gradient boosting'], ['concepts', 'metrics', 'Classification metrics'], ['omics', 'bulk-explain', 'Next: explaining predictions']]);
}
