// Double descent: random-features regression with the minimum-norm least-squares fit.
import { header, figure, slider, select, stats, section, steps, callout, math, legend, seeAlso, tip, untip, el } from '../lib/ui.js';
import { frame, cat, css, ink, muted } from '../lib/plot.js';
import { rng, solve } from '../lib/data.js';
const d3 = window.d3;

const N = 40;                                   // training points
const f = (x) => Math.sin(2.5 * x) + 0.5 * x;   // true function
const XT = d3.range(200).map((i) => -1 + (2 * i) / 199);
const PS = [...new Set([...d3.range(0, 41).map((i) => Math.round(Math.pow(10, (i / 40) * Math.log10(400)))), 40])].sort((a, b) => a - b); // 1..400, log-spaced, incl. p = n
const DRAWS = 6, PMAX = 400;

// Training x positions are fixed; only the labels change with the noise slider.
function trainSet(noise) {
  const r = rng(5), out = [];
  for (let i = 0; i < N; i++) { const x = r() * 2 - 1; out.push({ x, y: f(x) + r.normal(0, noise) }); }
  return out.sort((a, b) => a.x - b.x);
}
const XTR = trainSet(0).map((d) => d.x);

/** Random Fourier features √2·cos(ωx + b) with heavy-tailed ω (Cauchy, scale 3), b ~ U(0, 2π).
 *  The first p of a fixed list of 400 are used, so models are nested as p grows. */
function featureBank(draw) {
  const r = rng(1000 + draw), om = [], b = [];
  for (let j = 0; j < PMAX; j++) { om.push(3 * Math.tan(Math.PI * (r() - 0.5))); b.push(r() * 2 * Math.PI); }
  const phi = (x) => om.map((o, j) => Math.SQRT2 * Math.cos(o * x + b[j]));
  const Ftr = XTR.map(phi), Fte = XT.map(phi);
  // Gram matrices for every p in PS, built incrementally: ΦΦᵀ (n×n) for p > n, ΦᵀΦ (p×p) for p ≤ n.
  const gram = new Map(); const G = d3.range(N).map(() => new Array(N).fill(0));
  for (let j = 0; j < PMAX; j++) {
    for (let a = 0; a < N; a++) for (let c = 0; c < N; c++) G[a][c] += Ftr[a][j] * Ftr[c][j];
    const p = j + 1;
    if (!PS.includes(p)) continue;
    if (p > N) gram.set(p, G.map((row) => row.slice()));
    else gram.set(p, d3.range(p).map((i) => d3.range(p).map((k) => { let s = 0; for (let a = 0; a < N; a++) s += Ftr[a][i] * Ftr[a][k]; return s; })));
  }
  return { Ftr, Fte, gram, phi };
}
const BANKS = d3.range(DRAWS).map(featureBank);

/** Ridge solution on the first p features; with lam -> 0 this is the minimum-norm least-squares fit (pseudoinverse). */
function run(p, draw, data, lam) {
  const { Ftr, Fte, gram, phi } = BANKS[draw], y = data.map((d) => d.y), l = lam + 1e-10;
  const Gm = gram.get(p);
  let w;
  if (p <= N) {
    const A = Gm.map((row, i) => row.map((v, k) => v + (i === k ? l : 0)));
    w = solve(A, d3.range(p).map((i) => { let s = 0; for (let a = 0; a < N; a++) s += Ftr[a][i] * y[a]; return s; }));
  } else {
    const al = solve(Gm.map((row, i) => row.map((v, k) => v + (i === k ? l : 0))), y);
    w = d3.range(p).map((j) => { let s = 0; for (let a = 0; a < N; a++) s += Ftr[a][j] * al[a]; return s; });
  }
  const dotp = (v) => { let s = 0; for (let j = 0; j < p; j++) s += v[j] * w[j]; return s; };
  const train = d3.mean(Ftr, (v, i) => (dotp(v) - y[i]) ** 2);
  const test = d3.mean(Fte, (v, i) => (dotp(v) - f(XT[i])) ** 2);
  return { pred: (x) => dotp(phi(x)), train, test, norm: Math.sqrt(d3.sum(w, (v) => v * v)) };
}
function curve(data, lam) {
  return PS.map((p) => {
    const rs = d3.range(DRAWS).map((d) => run(p, d, data, lam));
    return { p, test: d3.median(rs, (r) => r.test), train: d3.median(rs, (r) => r.train) };
  });
}

export default async function render(main) {
  header(main, 'Double descent',
    'Classical statistics says test error is U-shaped in model size: too simple underfits, too complex overfits. Modern over-parameterised models show a <b>second descent</b>: error peaks when the model has just enough parameters to fit the training data exactly, then <b>falls again</b> as you add more, because among all the perfect fits the algorithm picks the smoothest one.');

  let noise = 0.2, lam = 0, pSel = N, data = trainSet(noise), C = curve(data, lam);
  const fig = figure(main, 'Random-features regression: test error vs number of parameters', '');
  slider(fig.controls, { label: 'Features p (selected)', min: 0, max: PS.length - 1, step: 1, value: PS.indexOf(N), fmt: (v) => PS[v], onInput: (v) => { pSel = PS[v]; paint(); } });
  slider(fig.controls, { label: 'Label noise σ', min: 0, max: 0.5, step: 0.05, value: noise, fmt: (v) => v.toFixed(2), onInput: (v) => { noise = v; data = trainSet(noise); C = curve(data, lam); paint(); } });
  select(fig.controls, { label: 'Ridge λ', value: '0', options: [['0', '0 (min-norm)'], ['0.001', '0.001'], ['0.1', '0.1']], onChange: (v) => { lam = +v; C = curve(data, lam); paint(); } });
  const readout = stats(fig.root);
  legend(fig.root, [['test error', cat(0)], ['training error', cat(1)], ['true function', muted()], ['fit at selected p', cat(0)]]);
  const row = el('div', { class: 'fig-row' }); fig.body.append(row);
  const lDiv = el('div'), rDiv = el('div'); row.append(lDiv, rDiv);
  const E = frame(lDiv, { width: 480, height: 380, x: [1, 400], y: [1e-3, 100], xScale: d3.scaleLog(), yScale: d3.scaleLog(), xLabel: 'number of random features p (log)', yLabel: 'mean squared error (log)' });
  E.g.select('.axis.x').call(d3.axisBottom(E.x).ticks(5, '~g'));
  E.g.select('.axis.y').call(d3.axisLeft(E.y).ticks(5, '~g'));
  const thr = E.layer('thr');
  thr.append('rect').attr('x', E.x(N * 0.8)).attr('width', E.x(N * 1.25) - E.x(N * 0.8)).attr('y', 0).attr('height', E.h).attr('fill', css('--surface-2'));
  thr.append('line').attr('x1', E.x(N)).attr('x2', E.x(N)).attr('y1', 0).attr('y2', E.h).attr('stroke', ink()).attr('stroke-width', 1).attr('stroke-dasharray', '3 3');
  thr.append('text').attr('x', E.x(N) + 5).attr('y', 14).attr('font-size', 11.5).attr('fill', ink()).text(`interpolation threshold p = n = ${N}`);
  E.layer('reg').append('text').attr('x', 6).attr('y', E.h - 8).attr('font-size', 11).attr('fill', ink()).text('← classical regime');
  E.layer('reg').append('text').attr('x', E.w - 6).attr('y', E.h - 8).attr('text-anchor', 'end').attr('font-size', 11).attr('fill', ink()).text('over-parameterised →');
  const Fp = frame(rDiv, { width: 480, height: 380, x: [-1, 1], y: [-2.5, 2.5], xLabel: 'x', yLabel: 'y' });
  Fp.layer('true').append('path').datum(XT).attr('fill', 'none').attr('stroke', muted()).attr('stroke-width', 2).attr('stroke-dasharray', '5 4')
    .attr('d', d3.line().x((x) => Fp.x(x)).y((x) => Fp.y(f(x))));
  Fp.layer('fit'); Fp.layer('pts');
  Fp.svg.append('defs').append('clipPath').attr('id', 'dd-clip').append('rect').attr('width', Fp.w).attr('height', Fp.h);
  Fp.layer('fit').attr('clip-path', 'url(#dd-clip)');

  function paint() {
    const cl = (v) => Math.max(1e-3, Math.min(100, v));
    const ln = (key, color) => E.layer('c-' + key).selectAll('path').data([C]).join('path').attr('fill', 'none').attr('stroke', color).attr('stroke-width', 2)
      .attr('d', d3.line().x((d) => E.x(d.p)).y((d) => E.y(cl(d[key]))));
    ln('test', cat(0)); ln('train', cat(1));
    E.layer('dots').selectAll('circle').data(C).join('circle').attr('cx', (d) => E.x(d.p)).attr('cy', (d) => E.y(cl(d.test))).attr('r', (d) => (d.p === pSel ? 6.5 : 0))
      .attr('fill', cat(0)).attr('stroke', css('--surface')).attr('stroke-width', 1.5);
    E.layer('hover').selectAll('rect').data(C).join('rect').attr('x', (d, i) => E.x(i ? Math.sqrt(C[i - 1].p * d.p) : 1)).attr('width', (d, i) => E.x(i < C.length - 1 ? Math.sqrt(C[i + 1].p * d.p) : 400) - E.x(i ? Math.sqrt(C[i - 1].p * d.p) : 1))
      .attr('y', 0).attr('height', E.h).attr('fill', 'transparent')
      .on('mousemove', (e, d) => tip(e, `p = ${d.p} features<br>test MSE ${d.test.toPrecision(3)}<br>train MSE ${d.train.toPrecision(3)}<br>(median of ${DRAWS} random feature draws)`)).on('mouseleave', untip);

    const r = run(pSel, 0, data, lam);
    Fp.layer('fit').selectAll('path').data([XT]).join('path').attr('fill', 'none').attr('stroke', cat(0)).attr('stroke-width', 2)
      .attr('d', d3.line().x((x) => Fp.x(x)).y((x) => Fp.y(Math.max(-4, Math.min(4, r.pred(x))))));
    Fp.layer('pts').selectAll('circle').data(data).join('circle').attr('cx', (d) => Fp.x(d.x)).attr('cy', (d) => Fp.y(d.y)).attr('r', 4.5)
      .attr('fill', cat(1)).attr('stroke', css('--surface')).attr('stroke-width', 1.5);

    const peak = C.reduce((m, d) => (d.test > m.test ? d : m));
    const classical = C.filter((d) => d.p < N * 0.8).reduce((m, d) => (d.test < m.test ? d : m));
    const big = C[C.length - 1];
    readout.set({
      [`selected p = ${pSel}: test MSE`]: r.test.toPrecision(3), 'train MSE': r.train.toPrecision(2), '‖w‖ (weight norm)': r.norm.toPrecision(3),
      'worst p (peak)': `${peak.p} (${peak.test.toPrecision(3)})`, 'best p < n': `${classical.p} (${classical.test.toPrecision(3)})`, 'p = 400': big.test.toPrecision(3),
    });
    fig.caption(`${N} noisy training points from a smooth curve; the model is a linear fit on p random cosine features. Left: median test error over ${DRAWS} random feature sets. Right: the fitted curve for one feature set at the selected p. ${lam === 0 ? 'With λ = 0, for p ≥ n this is the minimum-norm interpolating solution (training error ≈ 0).' : `With λ = ${lam} the fit no longer interpolates.`}`);
  }
  paint();

  // reference numbers for the default setting (σ = 0.2, λ = 0)
  const peak = C.reduce((m, d) => (d.test > m.test ? d : m));
  const classical = C.filter((d) => d.p < N * 0.8).reduce((m, d) => (d.test < m.test ? d : m));
  const big = C[C.length - 1];
  const C0 = curve(trainSet(0), 0), peak0 = C0.reduce((m, d) => (d.test > m.test ? d : m));
  const Cr = curve(data, 0.1), peakR = Cr.filter((d) => d.p >= N * 0.8 && d.p <= N * 1.25).reduce((m, d) => (d.test > m.test ? d : m));

  section(main, 'What to notice', `<ul>
    <li>The test-error curve first falls (the model gains enough features to follow the curve), then <b>explodes</b> near p = n = ${N}: the median test MSE peaks at ${peak.test.toPrecision(2)} at p = ${peak.p}, against ${classical.test.toPrecision(2)} at the best small model (p = ${classical.p}).</li>
    <li>Keep going right. With 400 features (10× more parameters than data points) test MSE is back down to ${big.test.toPrecision(2)}${big.test < classical.test ? ', <b>lower than the best classical model</b>' : ''}, even though training error is still essentially zero. Overfitting the training data perfectly did not hurt.</li>
    <li>Drag the <b>Features</b> slider to p = ${N} and look at the right panel: to pass through every noisy point with exactly enough features, the curve has to swing wildly between them, and the weight norm ‖w‖ shoots up. At p = 400 there are infinitely many interpolating fits, and the minimum-norm one is smooth.</li>
    <li>Set noise to 0: the peak shrinks (to ${peak0.test.toPrecision(2)}), because there is no noise to be forced through. The spike is noise amplified by an ill-conditioned fit.</li>
    <li>Choose ridge λ = 0.1: the peak disappears (around p = n the test MSE is now at most ${peakR.test.toPrecision(2)}). Proper regularisation removes double descent; it is a symptom of fitting with too little regularisation exactly at the threshold.</li>
  </ul>`);

  steps(main, 'How it works', [
    'Map each input x to p random features φⱼ(x) = √2·cos(ωⱼx + bⱼ), with frequencies ωⱼ and phases bⱼ drawn once at random and never trained. Only the linear weights w on top are learned.',
    'For p &lt; n there are more equations than unknowns: ordinary least squares, with a unique best fit.',
    'At p = n the system is square and (usually) exactly solvable: the fit passes through every point. If the feature matrix is nearly singular, the weights needed are huge and the curve oscillates. That is the peak.',
    'For p &gt; n there are infinitely many exact fits. Gradient descent started from zero (and the pseudoinverse) picks the one with the <b>smallest ‖w‖</b>. As p grows, that minimum norm falls and the fit becomes smoother: an implicit regulariser.',
    'Test error therefore rises toward the threshold and falls beyond it: double descent.',
  ]);
  math(main, 'p ≤ n:  w = (ΦᵀΦ)⁻¹ Φᵀ y             (ordinary least squares)\np > n:  w = Φᵀ (ΦΦᵀ)⁻¹ y             (minimum-norm interpolant = Φ⁺ y)\nridge:  w = (ΦᵀΦ + λI)⁻¹ Φᵀ y        → removes the peak at p = n');

  callout(main, '', 'In biology', `<ul>
    <li>Omics is natively <b>p ≫ n</b>: 20,000 genes and 100 patients. Classical intuition says such models must overfit, yet ridge regression and min-norm-like solutions on expression data often generalise reasonably. Double descent is part of the explanation.</li>
    <li>Foundation models such as Geneformer (tens of millions of parameters) and scGPT are deep in the over-parameterised regime, and their behaviour follows "bigger plus more data is smoother", not the classical U-curve.</li>
    <li>The dangerous zone is when parameters ≈ samples, e.g. an unregularised logistic regression on ~100 selected genes for ~100 patients. That is exactly the peak.</li>
  </ul>`);
  callout(main, 'warn', 'Pitfalls', `<ul>
    <li>Double descent is <b>not</b> "bigger is always better". The second descent depends on implicit regularisation (min-norm, early stopping, SGD) and on the data; with a tuned explicit regulariser the classical curve is often just as good.</li>
    <li>The peak location depends on the number of <b>training samples</b>, so adding data can temporarily make a fixed-size model worse (sample-wise double descent).</li>
    <li>Training error of zero is not evidence of a good model, nor proof of a bad one. Only held-out error tells you.</li>
  </ul>`);
  seeAlso(main, [['concepts', 'bias-variance', 'Under- vs overfitting'], ['concepts', 'regularization', 'Regularization'], ['concepts', 'linear-regression', 'Linear regression'], ['concepts', 'curse-dim', 'Curse of dimensionality']]);
}
