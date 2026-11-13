// Loss functions: what each one punishes. Regression losses vs residual, classification losses vs margin,
// and a draggable outlier showing how squared vs absolute vs Huber fits react.
import { header, figure, slider, button, stats, section, steps, callout, math, seeAlso, legend, tip, untip, el } from '../lib/ui.js';
import { frame, cat, css, ink, ink2, muted } from '../lib/plot.js';
import { rng, linspace } from '../lib/data.js';
const d3 = window.d3;

// ---- losses and their derivatives ----
const sq = (r) => r * r, dsq = (r) => 2 * r;
const ab = (r) => Math.abs(r), dab = (r) => Math.sign(r);
const hub = (r, d) => (Math.abs(r) <= d ? 0.5 * r * r : d * (Math.abs(r) - d / 2)), dhub = (r, d) => Math.max(-d, Math.min(d, r));
const LOG2 = Math.log(2);
const logl = (m) => (m > 30 ? Math.exp(-m) / LOG2 : Math.log1p(Math.exp(-m)) / LOG2); // log₂(1+e^−m): 1 at m = 0
const hinge = (m) => Math.max(0, 1 - m);
const expl = (m) => Math.exp(-m);
const zo = (m) => (m <= 0 ? 1 : 0);

const REG = [['squared  r²', cat(0)], ['absolute  |r|', cat(1)], ['Huber (δ)', cat(2)]];
const CLS = [['0–1 (what we really care about)', muted()], ['log-loss  log₂(1+e^−m)', cat(4)], ['hinge  max(0, 1−m)', cat(6)], ['exponential  e^−m', cat(3)]];

let clipN = 0;
function clipTo(F) {
  const id = 'lf-clip-' + (++clipN);
  F.svg.append('defs').append('clipPath').attr('id', id).append('rect').attr('width', F.w).attr('height', F.h);
  return id;
}

// Weighted least squares for a line y = a + b x.
function wls(pts, w) {
  let sw = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;
  pts.forEach((p, i) => { const k = w[i]; sw += k; sx += k * p.x; sy += k * p.y; sxx += k * p.x * p.x; sxy += k * p.x * p.y; });
  const b = (sw * sxy - sx * sy) / (sw * sxx - sx * sx);
  return { a: (sy - b * sx) / sw, b };
}
// Fit by iteratively reweighted least squares. weightFn(residual) -> weight.
function irls(pts, weightFn, iters) {
  let f = wls(pts, pts.map(() => 1));
  for (let t = 0; t < iters; t++) f = wls(pts, pts.map((p) => weightFn(p.y - f.a - f.b * p.x)));
  return f;
}

export default async function render(main) {
  header(main, 'Loss functions',
    'A loss function turns "how wrong is this prediction?" into a number that training then minimises. The <b>shape</b> of that function decides what the model cares about: squared error punishes big misses enormously, absolute error treats every unit of miss the same, and classification losses reward being <b>confidently right</b>.');

  // ================= figure 1: the curves =================
  const fig = figure(main, 'What each loss punishes',
    'Left: regression losses as a function of the residual r = y − ŷ (true minus predicted). Right: classification losses as a function of the <b>margin</b> m = y·f(x) with labels y ∈ {−1, +1}: m &gt; 0 means the right side of the boundary, large m means confidently right, negative m means wrong. The dots mark the slider values; the <b>slope</b> at the dot is how hard that one example pulls on the model during gradient descent.');
  let r0 = 1.5, m0 = -0.5, delta = 1;
  slider(fig.controls, { label: 'Residual r', min: -3, max: 3, step: 0.05, value: r0, fmt: (v) => v.toFixed(2), onInput: (v) => { r0 = v; paint1(); } });
  slider(fig.controls, { label: 'Huber δ', min: 0.2, max: 2.5, step: 0.05, value: delta, fmt: (v) => v.toFixed(2), onInput: (v) => { delta = v; paint1(); } });
  slider(fig.controls, { label: 'Margin m', min: -3, max: 3, step: 0.05, value: m0, fmt: (v) => v.toFixed(2), onInput: (v) => { m0 = v; paint1(); } });

  const row = el('div', { class: 'fig-row' }); fig.body.append(row);
  const left = el('div'), right = el('div'); row.append(left, right);
  const A = frame(left, { width: 480, height: 330, x: [-3, 3], y: [0, 5], xLabel: 'residual r = y − ŷ', yLabel: 'loss' });
  legend(left, REG);
  const sA = stats(left);
  const B = frame(right, { width: 480, height: 330, x: [-3, 3], y: [0, 5], xLabel: 'margin m = y·f(x)', yLabel: 'loss' });
  legend(right, CLS);
  const sB = stats(right);
  const ca = clipTo(A), cb = clipTo(B);
  const R = linspace(-3, 3, 301);
  const lnA = d3.line().x((d) => A.x(d[0])).y((d) => A.y(Math.min(d[1], 6)));
  const lnB = d3.line().x((d) => B.x(d[0])).y((d) => B.y(Math.min(d[1], 6)));
  // shaded "wrong side" region on the margin plot
  B.layer('bg').append('rect').attr('x', 0).attr('y', 0).attr('width', B.x(0)).attr('height', B.h).attr('fill', css('--surface-2'));
  B.layer('bg').append('text').attr('x', 8).attr('y', 16).attr('font-size', 11).attr('fill', muted()).text('misclassified');
  B.layer('bg').append('text').attr('x', B.w - 8).attr('y', 16).attr('text-anchor', 'end').attr('font-size', 11).attr('fill', muted()).text('correct');

  function paint1() {
    const regF = [sq, ab, (r) => hub(r, delta)];
    A.layer('lines').attr('clip-path', `url(#${ca})`).selectAll('path').data(regF).join('path').attr('fill', 'none')
      .attr('stroke', (f, i) => REG[i][1]).attr('stroke-width', 2).attr('d', (f) => lnA(R.map((r) => [r, f(r)])));
    A.layer('delta').selectAll('line').data([-delta, delta]).join('line').attr('x1', (d) => A.x(d)).attr('x2', (d) => A.x(d)).attr('y1', 0).attr('y2', A.h)
      .attr('stroke', cat(2)).attr('stroke-opacity', 0.5).attr('stroke-dasharray', '3 3');
    A.layer('dots').selectAll('circle').data(regF).join('circle').attr('cx', A.x(r0)).attr('cy', (f) => A.y(Math.min(5, f(r0)))).attr('r', 5.5)
      .attr('fill', (f, i) => REG[i][1]).attr('stroke', css('--surface')).attr('stroke-width', 1.5)
      .on('mousemove', (e, f) => tip(e, `r = ${r0.toFixed(2)}<br>loss = ${f(r0).toFixed(3)}`)).on('mouseleave', untip);
    sA.set({ [`squared (slope ${dsq(r0).toFixed(2)})`]: sq(r0).toFixed(2), [`absolute (slope ${dab(r0).toFixed(0)})`]: ab(r0).toFixed(2), [`Huber (slope ${dhub(r0, delta).toFixed(2)})`]: hub(r0, delta).toFixed(2) });

    const clsF = [zo, logl, hinge, expl];
    B.layer('lines').attr('clip-path', `url(#${cb})`).selectAll('path').data(clsF).join('path').attr('fill', 'none')
      .attr('stroke', (f, i) => CLS[i][1]).attr('stroke-width', 2).attr('stroke-dasharray', (f, i) => (i === 0 ? '5 4' : null))
      .attr('d', (f, i) => (i === 0 ? lnB([[-3, 1], [0, 1], [0, 0], [3, 0]]) : lnB(R.map((m) => [m, f(m)]))));
    B.layer('dots').selectAll('circle').data(clsF.slice(1)).join('circle').attr('cx', B.x(m0)).attr('cy', (f) => B.y(Math.min(5, f(m0)))).attr('r', 5.5)
      .attr('fill', (f, i) => CLS[i + 1][1]).attr('stroke', css('--surface')).attr('stroke-width', 1.5)
      .on('mousemove', (e, f) => tip(e, `m = ${m0.toFixed(2)}<br>loss = ${f(m0).toFixed(3)}`)).on('mouseleave', untip);
    const p = 1 / (1 + Math.exp(-m0));
    sB.set({ 'P(true class) = σ(m)': p.toFixed(2), 'log-loss': logl(m0).toFixed(2), hinge: hinge(m0).toFixed(2), exponential: expl(m0).toFixed(2) });
  }
  paint1();

  // ================= figure 2: drag the outlier =================
  const fig2 = figure(main, 'Drag the outlier: who gets pulled?',
    'Fourteen points follow y = 0.8x + 0.1 with small noise; the red point is an outlier you can <b>drag</b>. Three lines are fitted to all fifteen points, each by minimising a different loss. Right: each point\'s <b>share of the total pull</b> on its fit, i.e. |slope of the loss at that point\'s residual| divided by the sum over all points. The dotted line is an equal share (1/15). Under squared loss, a point\'s pull grows with its residual, so one bad point can dominate.');
  let hd = 0.3;
  slider(fig2.controls, { label: 'Huber δ', min: 0.05, max: 2, step: 0.05, value: hd, fmt: (v) => v.toFixed(2), onInput: (v) => { hd = v; paint2(); } });
  const r = rng(11);
  const base = d3.range(14).map((i) => { const x = -1 + 2 * (i + 0.5) / 14 + r.normal(0, 0.03); return { x, y: 0.8 * x + 0.1 + r.normal(0, 0.15) }; });
  const OUT0 = { x: 0.75, y: -2.2, outlier: true };
  let out = { ...OUT0 };
  button(fig2.controls, 'Reset outlier', () => { out = { ...OUT0 }; paint2(); });
  const s2 = stats(fig2.root);
  const row2 = el('div', { class: 'fig-row' }); fig2.body.append(row2);
  const l2 = el('div'), r2 = el('div'); row2.append(l2, r2);
  const P = frame(l2, { width: 480, height: 380, x: [-1.2, 1.2], y: [-3, 3], xLabel: 'x', yLabel: 'y' });
  const Q = frame(r2, { width: 480, height: 380, x: [0, 1], y: [0, 1], xLabel: 'point (sorted by x; red = outlier)', yLabel: 'share of total pull', grid: true });
  Q.g.select('.axis.x').remove();
  legend(fig2.body, [['least squares (MSE) fit', cat(0)], ['least absolute deviation (MAE) fit', cat(1)], ['Huber fit', cat(2)], ['true line', ink2()]]);
  const pc = clipTo(P);
  const XS = [-1.2, 1.2];
  P.layer('true').append('line').attr('x1', P.x(-1.2)).attr('x2', P.x(1.2)).attr('y1', P.y(0.8 * -1.2 + 0.1)).attr('y2', P.y(0.8 * 1.2 + 0.1))
    .attr('stroke', ink2()).attr('stroke-width', 2).attr('stroke-dasharray', '6 4');

  function fits(pts) {
    return [
      irls(pts, () => 1, 0),
      irls(pts, (e) => 1 / Math.max(Math.abs(e), 1e-5), 300),
      irls(pts, (e) => (Math.abs(e) <= hd ? 1 : hd / Math.abs(e)), 100),
    ];
  }
  function paint2() {
    const pts = [...base, out];
    const F = fits(pts);
    P.layer('lines').attr('clip-path', `url(#${pc})`).selectAll('line').data(F).join('line')
      .attr('x1', P.x(XS[0])).attr('x2', P.x(XS[1])).attr('y1', (f) => P.y(f.a + f.b * XS[0])).attr('y2', (f) => P.y(f.a + f.b * XS[1]))
      .attr('stroke', (f, i) => cat(i)).attr('stroke-width', 2.5);
    P.layer('pts').selectAll('circle').data(base).join('circle').attr('cx', (d) => P.x(d.x)).attr('cy', (d) => P.y(d.y)).attr('r', 5)
      .attr('fill', ink2()).attr('stroke', css('--surface')).attr('stroke-width', 1.5);
    const o = P.layer('out').selectAll('circle').data([out]).join('circle').attr('cx', (d) => P.x(d.x)).attr('cy', (d) => P.y(d.y)).attr('r', 8)
      .attr('fill', cat(7)).attr('stroke', css('--surface')).attr('stroke-width', 2).style('cursor', 'grab');
    o.call(d3.drag().on('drag', (e) => {
      const [mx, my] = d3.pointer(e.sourceEvent, P.g.node()); // pixel coords (the datum holds data coords, so e.x would mix the two)
      out = { x: Math.max(-1.15, Math.min(1.15, P.x.invert(mx))), y: Math.max(-2.95, Math.min(2.95, P.y.invert(my))), outlier: true };
      paint2();
    })).on('mousemove', (e, d) => tip(e, `outlier (${d.x.toFixed(2)}, ${d.y.toFixed(2)})<br>drag me`)).on('mouseleave', untip);

    // pull shares
    const grads = [(e) => Math.abs(dsq(e)), (e) => Math.min(1, Math.abs(e) / 1e-3), (e) => Math.abs(dhub(e, hd))];
    const order = pts.map((p, i) => i).sort((i, j) => pts[i].x - pts[j].x);
    const shares = F.map((f, k) => { const g = pts.map((p) => grads[k](p.y - f.a - f.b * p.x)); const s = d3.sum(g) || 1; return g.map((v) => v / s); });
    const n = pts.length, bw = Q.w / n, sub = (bw - 4) / 3;
    const maxS = Math.max(0.3, d3.max(shares.flat()));
    Q.y.domain([0, maxS]).nice(); Q.g.select('.axis.y').call(d3.axisLeft(Q.y).ticks(5, '.0%'));
    Q.g.select('.gridline').call(d3.axisLeft(Q.y).ticks(5).tickSize(-Q.w).tickFormat(''));
    const bars = order.flatMap((pi, slot) => shares.map((s, k) => ({ pi, slot, k, v: s[pi] })));
    Q.layer('hl').selectAll('rect').data([order.indexOf(n - 1)]).join('rect').attr('x', (s) => s * bw).attr('width', bw).attr('y', 0).attr('height', Q.h)
      .attr('fill', cat(7)).attr('opacity', 0.12);
    Q.layer('bars').selectAll('rect').data(bars).join('rect').attr('x', (d) => d.slot * bw + 2 + d.k * sub).attr('width', Math.max(1, sub - 0.5))
      .attr('y', (d) => Q.y(d.v)).attr('height', (d) => Q.h - Q.y(d.v)).attr('fill', (d) => cat(d.k))
      .on('mousemove', (e, d) => tip(e, `${pts[d.pi].outlier ? '<b>outlier</b>' : 'point at x = ' + pts[d.pi].x.toFixed(2)}<br>${['squared', 'absolute', 'Huber'][d.k]} loss: ${(100 * d.v).toFixed(1)}% of the pull`)).on('mouseleave', untip);
    Q.layer('even').selectAll('line').data([1 / n]).join('line').attr('x1', 0).attr('x2', Q.w).attr('y1', (d) => Q.y(d)).attr('y2', (d) => Q.y(d))
      .attr('stroke', muted()).attr('stroke-dasharray', '3 3');
    const oS = shares.map((s) => (100 * s[n - 1]).toFixed(0) + '%');
    s2.set({ 'slope: MSE fit': F[0].b.toFixed(2), 'slope: MAE fit': F[1].b.toFixed(2), 'slope: Huber fit': F[2].b.toFixed(2), 'true slope': '0.80', "outlier's pull (MSE / MAE / Huber)": oS.join(' / ') });
  }
  paint2();

  section(main, 'What to notice', `<ul>
    <li>Slide the residual to <b>r = 3</b>: squared loss is 9 and its slope is 6, while absolute loss is 3 with slope 1. A point three units off pulls six times harder than one half a unit off under squared loss, but exactly as hard under absolute loss.</li>
    <li>Huber is <b>squared inside ±δ and absolute outside</b>: smooth near zero like MSE, robust in the tails like MAE. Its slope is capped at δ.</li>
    <li>In the outlier figure, the outlier starts at (0.75, −2.2). The least-squares line tilts to a slope of 0.50 (true slope 0.80) and the outlier alone supplies 40% of the total pull on it. The MAE and Huber lines stay near the bulk of the data (slopes 0.91 and 0.85), and the outlier's share of their pull is only 8% and 13%. Drag the outlier up and down and watch mainly the blue line follow it.</li>
    <li>Drag the outlier to the bottom-right corner, about (1.15, −2.9): a point at the edge of the x-range has <b>leverage</b>, so the least-squares slope collapses to 0.18, while the MAE and Huber slopes stay at 0.91 and 0.82.</li>
    <li>Drag it back into the cloud of points: it stops being an outlier and the three lines nearly coincide.</li>
    <li>Set the margin to <b>m = 1</b>: hinge loss is exactly 0, while log-loss (0.45) and exponential loss (0.37) are still positive. Hinge stops caring once a point is beyond the margin (that is what makes SVMs depend only on a few support vectors); log-loss keeps rewarding more confidence, a little.</li>
    <li>For a badly wrong prediction (m = −3), exponential loss is 20 while log-loss is about 4.4: boosting with exponential loss (AdaBoost) is very sensitive to mislabelled samples.</li>
  </ul>`);

  steps(main, 'How it works', [
    'For each example, compare the prediction with the truth: a <b>residual</b> r for regression, or a <b>margin</b> m for classification.',
    'Push that through the loss function ℓ to get one non-negative number per example, and average over the training set.',
    'The optimiser follows the <b>gradient</b> of that average. Each example contributes ℓ′(its residual), so the slope of the loss curve is the "pull" that example exerts.',
    'Squared loss: pull ∝ residual, so the fit chases large errors (and outliers). The minimiser is the conditional <b>mean</b>.',
    'Absolute loss: constant pull, so the fit ignores how far off a point is, only which side it is on. The minimiser is the conditional <b>median</b>.',
    'Classification losses are convex stand-ins for the 0–1 loss (which has zero gradient almost everywhere). Log-loss = cross-entropy, the loss of logistic regression and of neural classifiers.',
  ]);
  math(main, 'squared   ℓ(r) = r²                       ℓ′ = 2r\nabsolute  ℓ(r) = |r|                      ℓ′ = sign(r)\nHuber     ℓ(r) = ½r² if |r| ≤ δ, else δ(|r| − δ/2)   ℓ′ = clip(r, −δ, δ)\nlog-loss  ℓ(m) = log(1 + e^−m) = −log σ(m)   (cross-entropy, y ∈ {−1,+1})\nhinge     ℓ(m) = max(0, 1 − m)');

  callout(main, '', 'In biology', 'The loss is a modelling choice about your noise. <b>scVI</b> does not use squared error on counts; it uses the negative log-likelihood of a zero-inflated negative binomial, because UMI counts are overdispersed and squared error would let a handful of highly expressed genes dominate. <b>DESeq2</b> and edgeR likewise fit negative-binomial likelihoods, not least squares. For a continuous phenotype with a few aberrant samples (a failed library, a mislabelled patient), Huber or MAE regression stops one sample from setting the slope. Cell-type classifiers and biomarker models trained with log-loss produce probabilities you can <a href="#/concepts/calibration">calibrate</a>.');
  callout(main, 'warn', 'Pitfalls', '<ul><li>The training loss and the metric you report need not match: you may train with log-loss and report AUROC. Choose the loss so that minimising it moves the metric you care about.</li><li>Squared error on raw counts or other heavy-tailed data is dominated by the largest values. Transform (log1p) or use a likelihood that fits the noise.</li><li>Huber\'s δ is in the units of the residual: rescale y and you need a new δ.</li><li>MAE has a kink at zero, so its gradient does not shrink near the optimum; plain gradient descent with a fixed step will jitter around the minimum.</li></ul>');
  seeAlso(main, [['concepts', 'gradient-descent', 'Gradient descent'], ['concepts', 'probability-mle', 'Likelihood & Bayes'], ['concepts', 'logistic-regression', 'Logistic regression'], ['concepts', 'svm', 'Support vector machines']]);
}
