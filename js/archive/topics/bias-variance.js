// Under- vs overfitting: polynomial degree slider on noisy sin data, train/test error curves,
// resampled fits overlaid (variance made visible), and a bias²/variance decomposition over 100 training sets.
import { header, figure, slider, button, stats, section, steps, callout, math, seeAlso, legend, tip, untip, el } from '../../lib/ui.js';
import { frame, cat, css, ink, ink2, muted } from '../../lib/plot.js';
import { rng, ridge, linspace, mean } from '../../lib/data.js';
const d3 = window.d3;

const TRUE = (x) => Math.sin(Math.PI * x);
const SIGMA = 0.3, DMAX = 14, NSETS = 100;
const GRID = linspace(-1, 1, 121);

// Legendre polynomials are orthogonal on [-1, 1], so high-degree fits stay well conditioned.
function legendre(x, d) {
  const p = [1];
  if (d >= 1) p.push(x);
  for (let k = 1; k < d; k++) p.push(((2 * k + 1) * x * p[k] - k * p[k - 1]) / (k + 1));
  return p;
}
function fitPoly(pts, d) {
  const w = ridge(pts.map((p) => legendre(p.x, d)), pts.map((p) => p.y), 1e-7);
  return (x) => legendre(x, d).reduce((s, v, i) => s + v * w[i], 0);
}
function sample(n, seed) {
  const r = rng(seed);
  // stratified x: one point per equal-width slot, so every sample covers the whole range
  return Array.from({ length: n }, (_, i) => { const x = -1 + 2 * (i + r()) / n; return { x, y: TRUE(x) + r.normal(0, SIGMA) }; });
}
const mseOf = (f, pts) => mean(pts.map((p) => (f(p.x) - p.y) ** 2));
const f3 = (v) => (v >= 100 ? v.toExponential(1) : v >= 10 ? v.toFixed(1) : v.toFixed(3));

let clipN = 0;
function clipTo(F) {
  const id = 'bv-clip-' + (++clipN);
  F.svg.append('defs').append('clipPath').attr('id', id).append('rect').attr('width', F.w).attr('height', F.h);
  return id;
}

export default async function render(main) {
  header(main, 'Under- vs overfitting',
    'A model that is too simple misses the pattern (<b>bias</b>); a model that is too flexible memorises the noise in its particular training sample (<b>variance</b>). The sweet spot is where the error on <b>new</b> data is lowest, and training error alone can never tell you where that is.');

  const TEST = sample(400, 999); // stratified too, so it spans [-1, 1] evenly
  let degree = 3, n = 20, seed = 1, ghosts = [];

  const fig = figure(main, 'Fit a polynomial of degree d to 20 noisy points',
    'Left: blue dots are the training sample, faint grey dots are fresh test data from the same process, the dashed line is the true function sin(πx). Right: mean squared error (MSE) on training vs test data for every degree; the dashed marker is the current degree. Press <b>Resample training data</b> a few times: each new sample leaves its fit behind as a faint line, so the spread of the faint lines <i>is</i> the variance.');
  const sD = slider(fig.controls, { label: 'Degree d', min: 0, max: DMAX, step: 1, value: degree, onInput: (v) => { degree = v; paint(); } });
  slider(fig.controls, { label: 'Training points n', min: 15, max: 100, step: 5, value: n, onInput: (v) => { n = v; ghosts = []; recompute(); paint(); } });
  button(fig.controls, 'Resample training data', () => { ghosts.push(seed); if (ghosts.length > 30) ghosts.shift(); seed += 1; recompute(); paint(); }, true);
  button(fig.controls, 'Clear overlaid fits', () => { ghosts = []; paint(); });
  const readout = stats(fig.root);

  const row = el('div', { class: 'fig-row' });
  const left = el('div'), right = el('div');
  row.append(left, right); fig.body.append(row);
  legend(fig.body, [['training points', cat(0)], ['test points', muted()], ['true function', ink2()], ['fit at degree d', cat(1)], ['fits to other training samples', d3.color(cat(1)).copy({ opacity: 0.35 }).formatRgb()]]);

  const F = frame(left, { width: 480, height: 380, x: [-1, 1], y: [-2.2, 2.2], xLabel: 'x', yLabel: 'y' });
  const fclip = clipTo(F);
  const E = frame(right, { width: 480, height: 380, x: [0, DMAX], y: [0.003, 100], yScale: d3.scaleLog(), xLabel: 'polynomial degree d', yLabel: 'MSE (log scale)' });
  E.g.select('.axis.y').call(d3.axisLeft(E.y).ticks(5, '~g'));
  const eclip = clipTo(E);
  const lineF = d3.line().x((d) => F.x(d.x)).y((d) => F.y(Math.max(-50, Math.min(50, d.y))));

  // static: test points + truth
  F.layer('test').selectAll('circle').data(TEST.filter((_, i) => i % 4 === 0)).join('circle').attr('cx', (d) => F.x(d.x)).attr('cy', (d) => F.y(d.y))
    .attr('r', 3).attr('fill', muted()).attr('opacity', 0.45);
  F.layer('truth').append('path').datum(GRID.map((x) => ({ x, y: TRUE(x) }))).attr('d', lineF).attr('fill', 'none')
    .attr('stroke', ink2()).attr('stroke-width', 2).attr('stroke-dasharray', '6 4');
  const ghostL = F.layer('ghosts').attr('clip-path', `url(#${fclip})`);
  const fitL = F.layer('fit').attr('clip-path', `url(#${fclip})`);
  const trainL = F.layer('train');

  let train, curves;
  function recompute() {
    train = sample(n, seed);
    curves = d3.range(DMAX + 1).map((d) => { const f = fitPoly(train, d); return { d, train: mseOf(f, train), test: mseOf(f, TEST) }; });
    decomp();
  }

  function paint() {
    const f = fitPoly(train, degree);
    ghostL.selectAll('path').data(ghosts).join('path').attr('fill', 'none').attr('stroke', cat(1)).attr('stroke-opacity', 0.3).attr('stroke-width', 1.3)
      .attr('d', (s) => { const g = fitPoly(sample(n, s), degree); return lineF(GRID.map((x) => ({ x, y: g(x) }))); });
    fitL.selectAll('path').data([GRID.map((x) => ({ x, y: f(x) }))]).join('path').attr('fill', 'none').attr('stroke', cat(1)).attr('stroke-width', 2.5).attr('d', lineF);
    trainL.selectAll('circle').data(train).join('circle').attr('cx', (d) => F.x(d.x)).attr('cy', (d) => F.y(d.y)).attr('r', 5)
      .attr('fill', cat(0)).attr('stroke', css('--surface')).attr('stroke-width', 1.5)
      .on('mousemove', (e, d) => tip(e, `x = ${d.x.toFixed(2)}, y = ${d.y.toFixed(2)}<br>fit = ${f(d.x).toFixed(2)}, residual = ${(d.y - f(d.x)).toFixed(2)}`)).on('mouseleave', untip);

    // error vs degree
    const yc = (v) => E.y(Math.max(0.003, Math.min(100, v)));
    const ln = (key) => d3.line().x((d) => E.x(d.d)).y((d) => yc(d[key]));
    E.layer('marker').selectAll('line').data([degree]).join('line').attr('x1', (d) => E.x(d)).attr('x2', (d) => E.x(d)).attr('y1', 0).attr('y2', E.h)
      .attr('stroke', ink()).attr('stroke-dasharray', '4 3').attr('stroke-width', 1.2);
    E.layer('noise').selectAll('line').data([SIGMA * SIGMA]).join('line').attr('x1', 0).attr('x2', E.w).attr('y1', (d) => E.y(d)).attr('y2', (d) => E.y(d))
      .attr('stroke', muted()).attr('stroke-dasharray', '2 3');
    E.layer('noise').selectAll('text').data([0]).join('text').attr('x', E.w - 4).attr('y', E.y(SIGMA * SIGMA) - 5).attr('text-anchor', 'end')
      .attr('font-size', 11).attr('fill', muted()).text('noise floor σ² = 0.09');
    const L = E.layer('lines').attr('clip-path', `url(#${eclip})`);
    L.selectAll('path').data(['train', 'test']).join('path').attr('fill', 'none').attr('stroke', (k) => cat(k === 'train' ? 0 : 1)).attr('stroke-width', 2)
      .attr('d', (k) => ln(k)(curves));
    E.layer('dots').selectAll('circle').data(curves.flatMap((c) => [{ ...c, k: 'train' }, { ...c, k: 'test' }])).join('circle')
      .attr('cx', (d) => E.x(d.d)).attr('cy', (d) => yc(d[d.k])).attr('r', (d) => (d.d === degree ? 6 : 4))
      .attr('fill', (d) => cat(d.k === 'train' ? 0 : 1)).attr('stroke', css('--surface')).attr('stroke-width', 1.5)
      .on('mousemove', (e, d) => tip(e, `degree ${d.d}<br>train MSE ${f3(d.train)}<br>test MSE ${f3(d.test)}`)).on('mouseleave', untip)
      .on('click', (e, d) => { degree = d.d; sD.set(degree); paint(); }).style('cursor', 'pointer');

    const c = curves[degree], best = curves.reduce((a, b) => (b.test < a.test ? b : a));
    readout.set({
      degree, 'train MSE': f3(c.train), 'test MSE': f3(c.test), 'best test degree': `${best.d} (${f3(best.test)})`,
      diagnosis: degree < best.d && c.test > best.test * 1.15 ? 'underfitting (bias)' : degree > best.d && c.test > best.test * 1.15 ? 'overfitting (variance)' : 'about right',
    });
    paintDecomp();
  }

  // ---- figure 2: bias² + variance decomposition over many training sets ----
  const fig2 = figure(main, 'Where test error comes from: bias² + variance + noise',
    `For each degree we draw ${NSETS} independent training sets of size n, fit each one, and ask two questions at every x: how far is the <b>average</b> fit from the truth (bias²), and how much do the individual fits <b>scatter</b> around that average (variance)? Their sum plus the irreducible noise σ² is the expected test error. The dashed marker follows the degree slider above.`);
  const readout2 = stats(fig2.root);
  const D = frame(fig2.body, { width: 900, height: 330, x: [0, DMAX], y: [1e-4, 100], yScale: d3.scaleLog(), xLabel: 'polynomial degree d', yLabel: 'error (log scale)' });
  D.g.select('.axis.y').call(d3.axisLeft(D.y).ticks(5, '~g'));
  const dclip = clipTo(D);
  legend(fig2.body, [['bias²', cat(2)], ['variance', cat(3)], ['noise σ²', muted()], ['bias² + variance + σ² (expected test MSE)', cat(1)]]);
  let dec = [];
  function decomp() {
    const preds = d3.range(DMAX + 1).map(() => []);
    for (let k = 0; k < NSETS; k++) {
      const s = sample(n, 5000 + k);
      for (let d = 0; d <= DMAX; d++) { const f = fitPoly(s, d); preds[d].push(GRID.map(f)); }
    }
    dec = preds.map((P, d) => {
      let b2 = 0, v = 0;
      GRID.forEach((x, g) => { const col = P.map((p) => p[g]); const m = mean(col); b2 += (m - TRUE(x)) ** 2; v += mean(col.map((c) => (c - m) ** 2)); });
      b2 /= GRID.length; v /= GRID.length;
      return { d, bias2: b2, variance: v, total: b2 + v + SIGMA * SIGMA };
    });
  }
  function paintDecomp() {
    const yc = (v) => D.y(Math.max(1e-4, Math.min(100, v)));
    const L = D.layer('lines').attr('clip-path', `url(#${dclip})`);
    const keys = [['bias2', cat(2)], ['variance', cat(3)], ['total', cat(1)]];
    D.layer('noise').selectAll('line').data([0]).join('line').attr('x1', 0).attr('x2', D.w).attr('y1', D.y(SIGMA * SIGMA)).attr('y2', D.y(SIGMA * SIGMA))
      .attr('stroke', muted()).attr('stroke-width', 2).attr('stroke-dasharray', '5 4');
    L.selectAll('path').data(keys).join('path').attr('fill', 'none').attr('stroke', (k) => k[1]).attr('stroke-width', 2)
      .attr('d', (k) => d3.line().x((d) => D.x(d.d)).y((d) => yc(d[k[0]]))(dec));
    D.layer('marker').selectAll('line').data([degree]).join('line').attr('x1', (d) => D.x(d)).attr('x2', (d) => D.x(d)).attr('y1', 0).attr('y2', D.h)
      .attr('stroke', ink()).attr('stroke-dasharray', '4 3').attr('stroke-width', 1.2);
    D.layer('dots').selectAll('circle').data(dec.flatMap((r) => keys.map(([k, c]) => ({ r, k, c })))).join('circle')
      .attr('cx', (d) => D.x(d.r.d)).attr('cy', (d) => yc(d.r[d.k])).attr('r', (d) => (d.r.d === degree ? 5.5 : 4)).attr('fill', (d) => d.c)
      .attr('stroke', css('--surface')).attr('stroke-width', 1.5)
      .on('mousemove', (e, d) => tip(e, `degree ${d.r.d}<br>bias² ${f3(d.r.bias2)}<br>variance ${f3(d.r.variance)}<br>expected test MSE ${f3(d.r.total)}`)).on('mouseleave', untip);
    const r = dec[degree], best = dec.reduce((a, b) => (b.total < a.total ? b : a));
    readout2.set({ degree, 'bias²': f3(r.bias2), variance: f3(r.variance), 'expected test MSE': f3(r.total), 'best degree (on average)': best.d });
  }

  recompute(); paint();

  section(main, 'What to notice', `<ul>
    <li><b>Degree 0 and 1</b> are straight lines through a wave: both training and test error sit far above the noise floor. That is <b>bias</b>, and adding more data does not fix it.</li>
    <li>At <b>degree 3</b> the fit tracks the dashed truth and the test error reaches its minimum (0.142), not far above the noise floor σ² = 0.09. Nothing can beat that floor: it is the noise in the labels.</li>
    <li>Push to <b>degree 14</b>: training error falls to 0.010, but the curve whips between the points and the test error shoots up to 83. Press <b>Resample training data</b> five times at degree 14 and then at degree 3 — the faint lines fan out wildly at high degree. That fan is the <b>variance</b>.</li>
    <li>Drag <b>n</b> up to 100: the overfitting zone moves to the right and the test curve flattens. More data buys you the right to use a more flexible model.</li>
    <li>In the decomposition, bias² collapses to almost zero by degree 3 while variance keeps rising; their sum is U-shaped, with its minimum at degree 3 for n = 20 (degree 5 for n = 100). Values off the top of the chart are capped at 100; hover a dot for the exact number. Variance explodes mostly at the edges of the x-range, where a high-degree polynomial has nothing to hold it down.</li>
  </ul>`);

  steps(main, 'How it works', [
    'Pick a family of models with a complexity knob: here, polynomials of degree d (d+1 coefficients).',
    'Fit by least squares on the <b>training</b> points only. More coefficients can always bend closer to those points, so training error can only go down as d grows.',
    'Measure error on <b>held-out</b> test points from the same process. This estimates how the model does on data it has never seen.',
    'Imagine repeating the whole experiment with fresh training samples. <b>Bias</b> is how wrong the <i>average</i> fit is; <b>variance</b> is how much individual fits disagree with each other.',
    'Choose the complexity where held-out error is lowest, using a validation set or <a href="#/archive/topics/cross-validation">cross-validation</a>, never the test set and never the training error.',
  ]);
  math(main, 'E[(y − f̂(x))²] = (E[f̂(x)] − f(x))²  +  Var[f̂(x)]  +  σ²\n                        bias²             variance      noise (irreducible)');

  callout(main, '', 'In biology', 'A biomarker study with 80 patients and 20,000 genes sits deep in the high-variance corner: a classifier can separate the training patients perfectly, and a different 80 patients would give a different gene list. That is why we regularise (<a href="#/archive/topics/regularization">lasso / ridge</a>), check stability across resamples, and report nested-CV performance. DESeq2\'s shrinkage of log fold changes is a deliberate trade: a little bias for a large cut in variance for low-count genes. In scRNA-seq, picking too many PCs or too high a clustering resolution is the same overfitting picture: you start fitting technical noise.');
  callout(main, 'warn', 'Pitfalls', '<ul><li>Training error is not an estimate of performance. A model at degree 14 can look excellent on the data it was fitted to.</li><li>"Complexity" is not just parameter count: regularisation strength, tree depth, number of boosting rounds and training epochs all move you along this curve.</li><li>Very large neural nets can show a second descent in test error past the interpolation point; see <a href="#/archive/topics/double-descent">double descent</a>. The classic U-shape is still the right picture for most omics-sized problems.</li><li>If you choose the degree by looking at test error, the test error of that choice is now optimistic. Keep a final test set untouched.</li></ul>');
  seeAlso(main, [['concepts', 'train-test', 'Train, validation, test'], ['concepts', 'regularization', 'Regularization'], ['concepts', 'cross-validation', 'Cross-validation'], ['concepts', 'double-descent', 'Double descent']]);
}
