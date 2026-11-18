// Classification metrics: one threshold over two score distributions drives the confusion matrix, ROC and PR.
import { header, figure, slider, button, stats, section, steps, callout, math, seeAlso, legend, el, tip, untip } from '../../lib/ui.js';
import { frame, cat, css, ink, ink2, muted } from '../../lib/plot.js';
import { rng } from '../../lib/data.js';
const d3 = window.d3;

const N_NEG = 200, N_POS = 200;

export default async function render(main) {
  header(main, 'Classification metrics',
    'A classifier outputs a <b>score</b>; you turn it into a yes/no call by picking a <b>threshold</b>. Every metric (accuracy, precision, recall, specificity, F1) is just a different way of counting the four boxes of the confusion matrix at that one threshold. ROC and PR curves show <b>all thresholds at once</b>.');

  // Fixed standard-normal draws; positives are shifted by the separation d. Deterministic.
  const r = rng(11);
  const zNeg = Array.from({ length: N_NEG }, () => r.normal());
  const zPos = Array.from({ length: N_POS }, () => r.normal());
  let sep = 1.5, thr = 0.75;

  const fig = figure(main, 'One threshold, every metric',
    'Top left: model scores for 200 true negatives and 200 true positives. Everything right of the threshold line is called positive. <b>Drag the line</b> (or use the slider). Bottom: the ROC and PR curves trace every possible threshold; the large dot is the current one. Hover the matrix cells and curves for details.');
  const thrS = slider(fig.controls, { label: 'Threshold', min: -3, max: 5, step: 0.05, value: thr, fmt: (v) => v.toFixed(2), onInput: (v) => { thr = v; update(); } });
  slider(fig.controls, { label: 'Separation d', min: 0, max: 4, step: 0.1, value: sep, fmt: (v) => v.toFixed(1), onInput: (v) => { sep = v; recompute(); update(); } });
  button(fig.controls, 'Best F1 threshold', () => { const b = bestF1(); thr = b; thrS.set(+b.toFixed(2)); update(); });
  const readout = stats(fig.root);

  const row1 = el('div', { class: 'fig-row' }), row2 = el('div', { class: 'fig-row' });
  fig.body.append(row1, row2);
  const histDiv = el('div'), cmDiv = el('div', { style: 'flex: 0 1 340px' });
  const rocDiv = el('div'), prDiv = el('div');
  row1.append(histDiv, cmDiv); row2.append(rocDiv, prDiv);
  legend(histDiv, [['true negatives (class 0)', cat(0)], ['true positives (class 1)', cat(1)]]);

  // ---------- histogram panel ----------
  const X = [-4, 8];
  const H = frame(histDiv, { width: 580, height: 300, x: X, y: [0, 0.5], xLabel: 'model score', yLabel: 'density', margin: { top: 22 } });
  const bins = d3.range(X[0], X[1] + 1e-9, 0.4);
  const regionL = H.layer('regions');
  H.layer('hist'); H.layer('thr');
  H.svg.style('cursor', 'ew-resize').call(d3.drag().container(H.g.node()).on('start drag', (e) => {
    thr = Math.max(-3, Math.min(5, H.x.invert(e.x))); thrS.set(+thr.toFixed(2)); update();
  }));

  // ---------- confusion matrix ----------
  const CM = d3.select(cmDiv).append('svg').attr('viewBox', '0 0 340 300').attr('width', 340);
  const cmG = CM.append('g').attr('transform', 'translate(78,52)');
  const cs = 120;
  CM.append('text').attr('x', 78 + cs).attr('y', 18).attr('text-anchor', 'middle').attr('fill', ink2()).attr('font-size', 13).attr('font-weight', 600).text('Predicted');
  ['−', '+'].forEach((t, j) => CM.append('text').attr('x', 78 + cs * j + cs / 2).attr('y', 42).attr('text-anchor', 'middle').attr('fill', ink2()).attr('font-size', 13).text(t === '−' ? 'negative' : 'positive'));
  CM.append('text').attr('transform', `translate(20,${52 + cs}) rotate(-90)`).attr('text-anchor', 'middle').attr('fill', ink2()).attr('font-size', 13).attr('font-weight', 600).text('Actual');
  ['negative', 'positive'].forEach((t, i) => CM.append('text').attr('transform', `translate(44,${52 + cs * i + cs / 2}) rotate(-90)`).attr('text-anchor', 'middle').attr('fill', ink2()).attr('font-size', 13).text(t));
  const CELLS = [
    { k: 'TN', i: 0, j: 0, name: 'True negatives', def: 'negatives correctly called negative', good: true },
    { k: 'FP', i: 0, j: 1, name: 'False positives', def: 'negatives wrongly called positive (type I error)', good: false },
    { k: 'FN', i: 1, j: 0, name: 'False negatives', def: 'positives missed (type II error)', good: false },
    { k: 'TP', i: 1, j: 1, name: 'True positives', def: 'positives correctly called positive', good: true },
  ];
  const cellG = cmG.selectAll('g').data(CELLS).join('g').attr('transform', (d) => `translate(${d.j * cs},${d.i * cs})`);
  cellG.append('rect').attr('width', cs - 4).attr('height', cs - 4).attr('rx', 6).attr('stroke', (d) => (d.good ? css('--good') : css('--bad'))).attr('stroke-width', 2);
  cellG.append('text').attr('class', 'k').attr('x', (cs - 4) / 2).attr('y', 40).attr('text-anchor', 'middle').attr('font-size', 14).attr('font-weight', 700).attr('fill', ink2());
  cellG.append('text').attr('class', 'n').attr('x', (cs - 4) / 2).attr('y', 76).attr('text-anchor', 'middle').attr('font-size', 26).attr('font-weight', 700).attr('fill', ink());
  cellG.on('mousemove', (e, d) => tip(e, `<b>${d.name}</b>: ${d.def}<br>count = ${counts[d.k]}`)).on('mouseleave', untip);

  // ---------- ROC + PR ----------
  const R = frame(rocDiv, { width: 460, height: 380, x: [0, 1], y: [0, 1], xLabel: 'false positive rate (1 − specificity)', yLabel: 'true positive rate (recall)' });
  R.layer('diag').append('line').attr('x1', R.x(0)).attr('y1', R.y(0)).attr('x2', R.x(1)).attr('y2', R.y(1)).attr('stroke', muted()).attr('stroke-dasharray', '4 4');
  R.layer('diag').append('text').attr('x', R.x(0.62)).attr('y', R.y(0.55)).attr('fill', muted()).attr('font-size', 11).attr('transform', `rotate(-38,${R.x(0.62)},${R.y(0.55)})`).text('random guessing');
  R.layer('curve'); R.layer('hover'); R.layer('pt');
  const P = frame(prDiv, { width: 460, height: 380, x: [0, 1], y: [0, 1.02], xLabel: 'recall (true positive rate)', yLabel: 'precision' });
  const base = N_POS / (N_POS + N_NEG);
  P.layer('diag').append('line').attr('x1', P.x(0)).attr('y1', P.y(base)).attr('x2', P.x(1)).attr('y2', P.y(base)).attr('stroke', muted()).attr('stroke-dasharray', '4 4');
  P.layer('diag').append('text').attr('x', P.x(0.02)).attr('y', P.y(base) + 14).attr('fill', muted()).attr('font-size', 11).text(`random guessing = prevalence (${base.toFixed(2)})`);
  P.layer('curve'); P.layer('hover'); P.layer('pt');

  let neg, pos, curve, counts;
  function recompute() {
    neg = zNeg.slice(); pos = zPos.map((z) => z + sep);
    // sweep thresholds from +inf down through every score
    const all = [...neg.map((s) => ({ s, y: 0 })), ...pos.map((s) => ({ s, y: 1 }))].sort((a, b) => b.s - a.s);
    let tp = 0, fp = 0; curve = [{ t: Infinity, tpr: 0, fpr: 0, prec: 1 }];
    all.forEach((d, i) => {
      if (d.y) tp++; else fp++;
      if (i < all.length - 1 && all[i + 1].s === d.s) return;
      curve.push({ t: d.s, tpr: tp / N_POS, fpr: fp / N_NEG, prec: tp / (tp + fp) });
    });
    curve[0].prec = curve[1].prec; // PR curve starts at the first real point's precision
  }
  function metricsAt(t) {
    const TP = pos.filter((s) => s >= t).length, FN = N_POS - TP;
    const FP = neg.filter((s) => s >= t).length, TN = N_NEG - FP;
    const prec = TP + FP ? TP / (TP + FP) : NaN, rec = TP / N_POS, spec = TN / N_NEG;
    const f1 = TP ? 2 * prec * rec / (prec + rec) : 0;
    return { TP, FN, FP, TN, prec, rec, spec, f1, acc: (TP + TN) / (N_POS + N_NEG) };
  }
  function bestF1() {
    let best = -1, bt = 0;
    for (let t = -3; t <= 5; t += 0.05) { const m = metricsAt(t); if (m.f1 > best) { best = m.f1; bt = t; } }
    return bt;
  }
  const aucROC = () => { let a = 0; for (let i = 1; i < curve.length; i++) a += (curve[i].fpr - curve[i - 1].fpr) * (curve[i].tpr + curve[i - 1].tpr) / 2; return a; };
  const avgPrec = () => { let a = 0; for (let i = 1; i < curve.length; i++) a += (curve[i].tpr - curve[i - 1].tpr) * curve[i].prec; return a; };

  function update() {
    const m = metricsAt(thr); counts = m;
    // histograms (density-normalised), each class as a filled step area
    const hist = (arr) => {
      const h = d3.bin().domain(X).thresholds(bins)(arr);
      return h.filter((b) => b.x1 > b.x0).map((b) => ({ x0: b.x0, x1: b.x1, d: b.length / (arr.length * (b.x1 - b.x0)) }));
    };
    const area = (hs) => { const pts = []; hs.forEach((b) => { pts.push([b.x0, b.d], [b.x1, b.d]); }); return pts; };
    const ar = d3.area().x((p) => H.x(p[0])).y0(H.y(0)).y1((p) => H.y(Math.min(0.5, p[1]))).curve(d3.curveStepAfter);
    const ln = d3.line().x((p) => H.x(p[0])).y((p) => H.y(Math.min(0.5, p[1])));
    const series = [{ c: 0, pts: area(hist(neg)) }, { c: 1, pts: area(hist(pos)) }];
    H.layer('hist').selectAll('path.a').data(series).join('path').attr('class', 'a').attr('d', (d) => ar(d.pts)).attr('fill', (d) => cat(d.c)).attr('fill-opacity', 0.28);
    H.layer('hist').selectAll('path.l').data(series).join('path').attr('class', 'l').attr('d', (d) => ln(d.pts)).attr('fill', 'none').attr('stroke', (d) => cat(d.c)).attr('stroke-width', 2);
    // shaded "called positive" region
    regionL.selectAll('rect').data([0]).join('rect').attr('x', H.x(thr)).attr('y', 0).attr('width', Math.max(0, H.w - H.x(thr))).attr('height', H.h).attr('fill', css('--surface-2'));
    const tl = H.layer('thr');
    tl.selectAll('line').data([0]).join('line').attr('x1', H.x(thr)).attr('x2', H.x(thr)).attr('y1', -8).attr('y2', H.h).attr('stroke', ink()).attr('stroke-width', 2);
    tl.selectAll('text').data(['← called negative', 'called positive →']).join('text').attr('x', (d, i) => H.x(thr) + (i ? 6 : -6)).attr('y', -10)
      .attr('text-anchor', (d, i) => (i ? 'start' : 'end')).attr('font-size', 12).attr('fill', ink2()).text((d) => d);
    // confusion matrix
    const maxC = Math.max(N_POS, N_NEG);
    cellG.select('rect').attr('fill', (d) => (d.good ? css('--good') : css('--bad'))).attr('fill-opacity', (d) => 0.06 + 0.4 * m[d.k] / maxC);
    cellG.select('text.k').text((d) => d.k);
    cellG.select('text.n').text((d) => m[d.k]);
    // ROC / PR curves
    R.layer('curve').selectAll('path').data([curve]).join('path').attr('fill', 'none').attr('stroke', cat(0)).attr('stroke-width', 2)
      .attr('d', d3.line().x((d) => R.x(d.fpr)).y((d) => R.y(d.tpr)));
    P.layer('curve').selectAll('path').data([curve.slice(1)]).join('path').attr('fill', 'none').attr('stroke', cat(0)).attr('stroke-width', 2)
      .attr('d', d3.line().x((d) => P.x(d.tpr)).y((d) => P.y(d.prec)));
    const hov = (F, xf, yf) => F.layer('hover').selectAll('circle').data(curve.slice(1)).join('circle').attr('cx', (d) => F.x(xf(d))).attr('cy', (d) => F.y(yf(d))).attr('r', 5).attr('fill', 'transparent')
      .on('mousemove', (e, d) => tip(e, `threshold ${d.t.toFixed(2)}<br>TPR/recall ${d.tpr.toFixed(2)} · FPR ${d.fpr.toFixed(2)}<br>precision ${d.prec.toFixed(2)}`)).on('mouseleave', untip);
    hov(R, (d) => d.fpr, (d) => d.tpr); hov(P, (d) => d.tpr, (d) => d.prec);
    const fpr = 1 - m.spec;
    const dot = (F, cx, cy) => F.layer('pt').selectAll('circle').data([0]).join('circle').attr('cx', F.x(cx)).attr('cy', F.y(cy)).attr('r', 7).attr('fill', cat(1)).attr('stroke', css('--surface')).attr('stroke-width', 2);
    dot(R, fpr, m.rec); dot(P, m.rec, isNaN(m.prec) ? 1 : m.prec);
    const f = (v) => (isNaN(v) ? '—' : v.toFixed(3));
    readout.set({ accuracy: f(m.acc), 'precision (PPV)': f(m.prec), 'recall (sensitivity)': f(m.rec), specificity: f(m.spec), F1: f(m.f1), 'ROC-AUC': aucROC().toFixed(3), 'PR-AUC (avg precision)': avgPrec().toFixed(3) });
  }

  recompute(); update();

  section(main, 'What to notice', `<ul>
    <li><b>Slide the threshold right</b>: fewer calls, so false positives fall and precision and specificity rise, but recall falls. You can't get both for free; the threshold only picks a point on the trade-off.</li>
    <li><b>The ROC and PR curves don't move when you change the threshold.</b> They belong to the model (its scores). Only the dot moves. ROC-AUC and PR-AUC summarise the whole curve, so they don't depend on the threshold either.</li>
    <li><b>Set the separation to 0</b>: the two histograms overlap completely, the ROC curve hugs the diagonal and its AUC is close to 0.5. That's a coin flip, however you tune the threshold.</li>
    <li><b>Set the separation to 4</b>: the ROC curve runs into the top-left corner and almost any threshold between the two humps gives near-perfect metrics.</li>
    <li><b>Best F1 threshold</b> jumps to the threshold that maximises F1 on <i>these</i> data. Picking a threshold on your test set like this is itself a form of tuning, so do it on validation data.</li>
    <li>Here the classes are balanced (200 vs 200), so accuracy is sensible. When positives are rare, accuracy and ROC flatter the model. See <a href="#/archive/topics/imbalance">class imbalance</a>.</li>
  </ul>`);

  steps(main, 'How it works', [
    'The model gives each sample a <b>score</b> (a probability, a logit, a distance from a boundary; only the ordering matters for ROC/PR).',
    'Choose a <b>threshold</b> t: score ≥ t ⇒ predict positive.',
    'Cross the prediction with the truth to fill the <b>confusion matrix</b>: TP, FP, FN, TN.',
    '<b>Recall</b> (sensitivity, TPR) = TP / all actual positives: "of the real cases, how many did we catch?". <b>Specificity</b> = TN / all actual negatives. <b>Precision</b> (PPV) = TP / all predicted positives: "when we call it, how often are we right?". <b>F1</b> is the harmonic mean of precision and recall.',
    'Sweep t from +∞ to −∞ and plot (FPR, TPR) to get the <b>ROC curve</b>, or (recall, precision) to get the <b>PR curve</b>. The area under the ROC curve is the probability that a random positive scores higher than a random negative.',
  ]);
  math(main, 'precision = TP/(TP+FP)    recall = TPR = TP/(TP+FN)    specificity = TN/(TN+FP)    FPR = 1 − specificity\nF1 = 2·precision·recall/(precision+recall)        accuracy = (TP+TN)/N\nROC-AUC = P(score of random positive > score of random negative)');

  callout(main, '', 'In biology', 'A diagnostic <b>biomarker cohort</b> (say, a ctDNA methylation classifier for early cancer) is reported as sensitivity at a fixed specificity (e.g. 99% specificity), because false positives in screening cause unnecessary biopsies. For a <b>cell-type classifier</b> on scRNA-seq or a <b>doublet detector</b> (scDblFinder, DoubletFinder), per-class precision and recall matter more than overall accuracy, since rare types are the ones that get missed. For variant callers (GATK, DeepVariant), benchmarks against Genome in a Bottle report precision, recall and F1.');
  callout(main, 'warn', 'Pitfalls', '<ul><li>Accuracy is meaningless when classes are imbalanced: predicting "healthy" for everyone is 99% accurate at 1% prevalence.</li><li>Precision depends on prevalence; recall and specificity do not. A test validated in a 50/50 case–control study will have far lower precision in the clinic.</li><li>Choosing the threshold (or "the best of several models") on the test set inflates every metric. Choose on validation or inside cross-validation.</li><li>ROC-AUC says nothing about calibration: a model can rank perfectly and still output badly wrong probabilities. See <a href="#/archive/topics/calibration">calibration</a>.</li></ul>');
  seeAlso(main, [['concepts', 'imbalance', 'Class imbalance'], ['concepts', 'calibration', 'Calibration'], ['concepts', 'cross-validation', 'Cross-validation'], ['concepts', 'logistic-regression', 'Logistic regression']]);
}
