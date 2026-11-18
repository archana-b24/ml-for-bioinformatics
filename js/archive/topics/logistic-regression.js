// Logistic regression: a linear score squashed by a sigmoid into a probability, trained by gradient descent on log-loss.
import { header, figure, slider, select, button, stats, section, steps, callout, math, legend, seeAlso, animate, tip, untip, el } from '../../lib/ui.js';
import { frame, decisionCanvas, cat, css, ink, ink2, muted } from '../../lib/plot.js';
import { blobs, sigmoid } from '../../lib/data.js';
const d3 = window.d3;

const DATA = {
  overlap: { name: 'Overlapping classes', pts: () => blobs(120, [[-0.42, -0.28], [0.42, 0.28]], 0.42, 21) },
  separable: { name: 'Perfectly separable', pts: () => blobs(120, [[-0.55, -0.35], [0.55, 0.35]], 0.17, 22) },
};
const START = { w: [-1.2, 2.4], b: 0.6 };
const LEVELS = [0.1, 0.25, 0.5, 0.75, 0.9];

export default async function render(main) {
  header(main, 'Logistic regression',
    'Compute a weighted sum of the features, exactly like linear regression, then squash it through an S-shaped <b>sigmoid</b> so the output is a probability between 0 and 1. The decision boundary is where that probability crosses 0.5, which is always a straight line.');

  // ---------------- Figure 1: live training ----------------
  const fig = figure(main, 'Watch gradient descent rotate the boundary into place',
    'Left: shading is the predicted probability of class 1 (orange); the solid line is p = 0.5 and dashed lines are p = 0.1, 0.25, 0.75, 0.9. The arrow is the weight vector w, which always points across the boundary towards class 1. Right: every point projected onto w (its score z = w·x + b) against its true label, with the sigmoid that turns z into p. Hover a point for its probability and loss.');
  let key = 'overlap', pts = DATA[key].pts(), w = START.w.slice(), b = START.b, lr = 10 ** 0.5, hist = [], stop = null;

  select(fig.controls, { label: 'Data', value: key, options: Object.entries(DATA).map(([k, v]) => [k, v.name]), onChange: (k) => { key = k; pts = DATA[k].pts(); reset(); } });
  slider(fig.controls, { label: 'Learning rate η', min: -1.3, max: 0.7, step: 0.05, value: 0.5, fmt: (v) => (10 ** v).toFixed(2), onInput: (v) => { lr = 10 ** v; } });
  button(fig.controls, 'Step', () => { halt(); gdStep(); paint(); }, true);
  button(fig.controls, 'Run 200 steps', () => run());
  button(fig.controls, 'Reset', () => reset());
  const readout = stats(fig.root);

  const row = el('div', { class: 'fig-row' }); fig.body.append(row);
  const left = el('div'), right = el('div'); row.append(left, right);
  legend(fig.body, [['class 0', cat(0)], ['class 1', cat(1)], ['p = 0.5 boundary', ink()], ['sigmoid σ(z)', cat(2)]]);

  const F = frame(left, { width: 480, height: 440, x: [-1.4, 1.4], y: [-1.4, 1.4], xLabel: 'feature 1 (e.g. IFNG score)', yLabel: 'feature 2 (e.g. CD8A expr.)', grid: false });
  const prob = (px, py) => sigmoid(w[0] * px + w[1] * py + b);
  const surf = decisionCanvas(F, prob, { res: 4, alpha: 0.35 });
  F.g.append('clipPath').attr('id', 'lg-clip').append('rect').attr('width', F.w).attr('height', F.h);
  const contL = F.layer('cont').attr('clip-path', 'url(#lg-clip)');
  const ptL = F.layer('pts'), arrL = F.layer('arrow');
  const defs = F.svg.append('defs');
  defs.append('marker').attr('id', 'lg-arrow').attr('viewBox', '0 0 10 10').attr('refX', 8).attr('refY', 5).attr('markerWidth', 7).attr('markerHeight', 7).attr('orient', 'auto')
    .append('path').attr('d', 'M0,0 L10,5 L0,10 z').attr('fill', ink());

  const right1 = el('div'), right2 = el('div'); right.append(right1, right2);
  const ZD = [-8, 8];
  const G = frame(right1, { width: 480, height: 270, x: ZD, y: [-0.12, 1.12], xLabel: 'score z = w·x + b (projection onto w)', yLabel: 'P(class 1)' });
  G.g.append('clipPath').attr('id', 'lg-clip2').append('rect').attr('width', G.w).attr('height', G.h);
  const zs = d3.range(ZD[0], ZD[1] + 0.01, 0.1).map((z) => ({ x: z, y: sigmoid(z) }));
  G.layer('ref').append('line').attr('x1', G.x(0)).attr('x2', G.x(0)).attr('y1', 0).attr('y2', G.h).attr('stroke', muted()).attr('stroke-dasharray', '4 4');
  G.layer('ref').append('line').attr('x1', 0).attr('x2', G.w).attr('y1', G.y(0.5)).attr('y2', G.y(0.5)).attr('stroke', muted()).attr('stroke-dasharray', '4 4');
  G.layer('sig').append('path').datum(zs).attr('fill', 'none').attr('stroke', cat(2)).attr('stroke-width', 2.5).attr('d', d3.line().x((d) => G.x(d.x)).y((d) => G.y(d.y)));
  const gPts = G.layer('pts').attr('clip-path', 'url(#lg-clip2)');
  const H = frame(right2, { width: 480, height: 170, x: [0, 200], y: [0, 1], xLabel: 'gradient-descent step', yLabel: 'log-loss' });
  const hL = H.layer('h');

  // jitter for the projected strip plot (fixed so points do not dance)
  const jit = pts.map((_, i) => ((i * 7919) % 97) / 97 - 0.5);

  function grads() {
    let g0 = 0, g1 = 0, gb = 0;
    pts.forEach((p) => { const e = prob(p.x, p.y) - p.label; g0 += e * p.x; g1 += e * p.y; gb += e; });
    const n = pts.length; return [g0 / n, g1 / n, gb / n];
  }
  function logloss() {
    return d3.mean(pts, (p) => { const q = Math.min(1 - 1e-12, Math.max(1e-12, prob(p.x, p.y))); return -(p.label ? Math.log(q) : Math.log(1 - q)); });
  }
  function gdStep() { const g = grads(); w = [w[0] - lr * g[0], w[1] - lr * g[1]]; b -= lr * g[2]; hist.push(logloss()); }
  function halt() { if (stop) { stop(); stop = null; } }
  function run() {
    halt(); let acc = 0, k = 0;
    stop = animate((dt) => { acc += dt; if (acc < 25) return true; acc = 0; gdStep(); paint(); k++; return k < 200; });
  }
  function reset() { halt(); w = START.w.slice(); b = START.b; hist = [logloss()]; paint(); }

  function paint() {
    surf.draw(prob);
    // probability contour lines: w·x + b = logit(p)
    const [X0, X1] = F.x.domain(), [Y0, Y1] = F.y.domain();
    const segs = LEVELS.map((p) => {
      const L = Math.log(p / (1 - p)) - b;
      let a, c;
      if (Math.abs(w[1]) > Math.abs(w[0])) { a = { x: X0, y: (L - w[0] * X0) / w[1] }; c = { x: X1, y: (L - w[0] * X1) / w[1] }; }
      else { a = { x: (L - w[1] * Y0) / w[0], y: Y0 }; c = { x: (L - w[1] * Y1) / w[0], y: Y1 }; }
      return { p, a, c };
    });
    contL.selectAll('line').data(segs).join('line').attr('x1', (d) => F.x(d.a.x)).attr('y1', (d) => F.y(d.a.y)).attr('x2', (d) => F.x(d.c.x)).attr('y2', (d) => F.y(d.c.y))
      .attr('stroke', (d) => (d.p === 0.5 ? ink() : ink2())).attr('stroke-width', (d) => (d.p === 0.5 ? 2.2 : 1)).attr('stroke-dasharray', (d) => (d.p === 0.5 ? null : '5 4'));
    contL.selectAll('text').data(segs.filter((d) => d.p !== 0.5)).join('text').attr('font-size', 10.5).attr('fill', ink2())
      .attr('x', (d) => { const t = 0.86; return F.x(d.a.x + t * (d.c.x - d.a.x)) + 4; }).attr('y', (d) => { const t = 0.86; return F.y(d.a.y + t * (d.c.y - d.a.y)) - 3; }).text((d) => d.p);
    const lossOf = (p) => { const q = prob(p.x, p.y); return -(p.label ? Math.log(q) : Math.log(1 - q)); };
    ptL.selectAll('circle').data(pts).join('circle').attr('cx', (d) => F.x(d.x)).attr('cy', (d) => F.y(d.y)).attr('r', 4.5)
      .attr('fill', (d) => cat(d.label)).attr('stroke', css('--surface')).attr('stroke-width', 1.2)
      .on('mousemove', (e, d) => tip(e, `true class ${d.label}<br>P(class 1) = ${prob(d.x, d.y).toFixed(3)}<br>log-loss = ${lossOf(d).toFixed(3)}`)).on('mouseleave', untip);
    // weight-vector arrow from the boundary point closest to the origin
    const nw = Math.hypot(w[0], w[1]) || 1e-9, u = [w[0] / nw, w[1] / nw], d0 = -b / nw;
    const o = [u[0] * d0, u[1] * d0], len = 0.45;
    arrL.selectAll('line').data([0]).join('line').attr('x1', F.x(o[0])).attr('y1', F.y(o[1])).attr('x2', F.x(o[0] + u[0] * len)).attr('y2', F.y(o[1] + u[1] * len))
      .attr('stroke', ink()).attr('stroke-width', 2.2).attr('marker-end', 'url(#lg-arrow)');
    arrL.selectAll('text').data([0]).join('text').attr('x', F.x(o[0] + u[0] * (len + 0.08))).attr('y', F.y(o[1] + u[1] * (len + 0.08)) + 4).attr('font-size', 13).attr('font-weight', 700).attr('fill', ink()).attr('text-anchor', 'middle').text('w');
    // projection strip plot
    gPts.selectAll('circle').data(pts).join('circle').attr('r', 4).attr('fill', (d) => cat(d.label)).attr('fill-opacity', 0.85).attr('stroke', css('--surface')).attr('stroke-width', 1)
      .attr('cx', (d) => G.x(Math.max(ZD[0] - 0.3, Math.min(ZD[1] + 0.3, w[0] * d.x + w[1] * d.y + b))))
      .attr('cy', (d, i) => G.y(d.label + (d.label ? -1 : 1) * (0.04 + 0.05 * (jit[i] + 0.5))))
      .on('mousemove', (e, d) => tip(e, `z = ${(w[0] * d.x + w[1] * d.y + b).toFixed(2)} → p = ${prob(d.x, d.y).toFixed(3)}<br>true class ${d.label}`)).on('mouseleave', untip);
    // loss history
    const hx = Math.max(200, hist.length);
    H.x.domain([0, hx]); H.y.domain([0, Math.max(0.8, d3.max(hist) * 1.05)]);
    H.g.select('.axis.x').call(d3.axisBottom(H.x).ticks(5)); H.g.select('.axis.y').call(d3.axisLeft(H.y).ticks(3));
    hL.selectAll('path').data([hist]).join('path').attr('fill', 'none').attr('stroke', cat(0)).attr('stroke-width', 2)
      .attr('d', d3.line().x((d, i) => H.x(i)).y((d) => H.y(d)));
    const acc = d3.mean(pts, (p) => (prob(p.x, p.y) >= 0.5) === (p.label === 1));
    readout.set({ step: hist.length - 1, 'log-loss': logloss().toFixed(4), accuracy: (100 * acc).toFixed(1) + '%', 'w = (w₁, w₂)': `(${w[0].toFixed(2)}, ${w[1].toFixed(2)})`, '|w| (steepness)': nw.toFixed(2) });
  }
  hist = [logloss()]; paint();

  // ---------------- Figure 2: log-loss ----------------
  const fig2 = figure(main, 'What log-loss punishes',
    'The loss for one sample is −log of the probability the model gave to the <i>true</i> class. Slide the predicted probability. Being confidently wrong costs far more than being unsure.');
  let pp = 0.8;
  slider(fig2.controls, { label: 'predicted P(class 1)', min: 0.01, max: 0.99, step: 0.01, value: pp, fmt: (v) => v.toFixed(2), onInput: (v) => { pp = v; paint2(); } });
  const r2 = stats(fig2.root);
  const L = frame(fig2.body, { width: 720, height: 280, x: [0, 1], y: [0, 4.8], xLabel: 'predicted probability of class 1', yLabel: 'loss' });
  const ps = d3.range(0.005, 0.9951, 0.005);
  L.layer('c').append('path').datum(ps.map((p) => ({ x: p, y: -Math.log(p) }))).attr('fill', 'none').attr('stroke', cat(1)).attr('stroke-width', 2.2).attr('d', d3.line().x((d) => L.x(d.x)).y((d) => L.y(Math.min(4.8, d.y))));
  L.layer('c').append('path').datum(ps.map((p) => ({ x: p, y: -Math.log(1 - p) }))).attr('fill', 'none').attr('stroke', cat(0)).attr('stroke-width', 2.2).attr('d', d3.line().x((d) => L.x(d.x)).y((d) => L.y(Math.min(4.8, d.y))));
  legend(fig2.body, [['if the true class is 1: −log p', cat(1)], ['if the true class is 0: −log(1 − p)', cat(0)]]);
  const mk = L.layer('m');
  function paint2() {
    const l1 = -Math.log(pp), l0 = -Math.log(1 - pp);
    mk.selectAll('line').data([0]).join('line').attr('x1', L.x(pp)).attr('x2', L.x(pp)).attr('y1', 0).attr('y2', L.h).attr('stroke', muted()).attr('stroke-dasharray', '4 4');
    mk.selectAll('circle').data([[l1, 1], [l0, 0]]).join('circle').attr('cx', L.x(pp)).attr('cy', (d) => L.y(Math.min(4.8, d[0]))).attr('r', 6)
      .attr('fill', (d) => cat(d[1])).attr('stroke', css('--surface')).attr('stroke-width', 1.5);
    r2.set({ 'loss if truly class 1': l1.toFixed(3), 'loss if truly class 0': l0.toFixed(3), 'ratio wrong / right': (Math.max(l0, l1) / Math.min(l0, l1)).toFixed(1) + '×' });
  }
  paint2();

  section(main, 'What to notice', `<ul>
    <li>Press <b>Step</b> a few times. The boundary starts tilted the wrong way (56% accuracy, barely better than a coin) and swings round within about five steps, because the misclassified points contribute the largest gradients.</li>
    <li>On the right, the projection onto w turns the 2D problem into 1D: training slides the orange points right and the blue points left along the sigmoid. Points near z = 0 are the uncertain ones.</li>
    <li>On <b>Overlapping classes</b>, after about 200 steps the loss levels off at 0.227 and |w| settles near 6.6. About 10% of the points sit on the wrong side whatever you do, so the model cannot gain by becoming more confident, and the probability bands stay wide.</li>
    <li>Switch to <b>Perfectly separable</b> and press Run a few times. Accuracy hits 100% by step 5, yet the loss keeps falling and |w| keeps growing (about 9.5 after 200 steps, 13 after 1,000), and the bands squeeze together. The model can always become more confident by making the sigmoid steeper, so unregularised weights head to infinity. An L2 penalty stops this.</li>
    <li>In the log-loss figure, predicting 0.99 for a sample that is truly class 0 costs −log(0.01) ≈ 4.6, about 460 times the cost of predicting 0.99 for a true class 1.</li>
  </ul>`);

  steps(main, 'How it works', [
    'Compute a linear score per sample: <code>z = w·x + b</code>. This is the signed distance to the boundary, scaled by |w|.',
    'Turn it into a probability: <code>p = σ(z) = 1 / (1 + e<sup>−z</sup>)</code>. Equivalently, the <b>log-odds</b> log(p / (1 − p)) is linear in the features, so each coefficient is a change in log-odds per unit of that feature.',
    'Score the predictions with <b>log-loss</b> (binary cross-entropy): the mean of −log(probability given to the true class).',
    'The gradient has a lovely form: <code>∇w = mean((p − y)·x)</code>. Each point pushes w in proportion to how wrong it is.',
    'Step downhill (<code>w ← w − η∇w</code>) until the loss stops falling. Log-loss is convex here, so there is one minimum and gradient descent finds it (or a solver like L-BFGS does so faster).',
  ]);
  math(main, 'p(y=1 | x) = σ(wᵀx + b),   σ(z) = 1/(1+e⁻ᶻ)\nlog-loss = −(1/n) Σ [ yᵢ log pᵢ + (1−yᵢ) log(1−pᵢ) ]\n∂loss/∂w = (1/n) Σ (pᵢ − yᵢ) xᵢ');

  callout(main, '', 'In biology', 'Penalised logistic regression is the default first model for a binary clinical outcome from omics, for example responder vs non-responder from bulk RNA-seq, or tumour vs normal from methylation array CpGs (<code>glmnet</code> with family = "binomial"; scikit-learn <code>LogisticRegression</code>). Because the coefficients are log-odds ratios, they are easy to report. Polygenic risk scores used as covariates and many diagnostic signatures (such as multi-gene expression panels) are logistic models underneath. In single-cell work, CellTypist annotates cells with a regularised logistic regression on gene expression.');
  callout(main, 'warn', 'Pitfalls', `<ul>
    <li><b>Perfect separation</b> (common when genes ≫ samples) makes the weights blow up and the fit fail to converge. Always regularise (L2 or <a href="#/archive/topics/regularization">lasso / elastic-net</a>).</li>
    <li>The boundary is linear. Curved class structure needs engineered features, kernels or a non-linear model.</li>
    <li>Coefficients are only comparable across features if the features are on the same scale. <a href="#/archive/topics/scaling">Standardise first.</a></li>
    <li>Predicted probabilities from a heavily regularised or resampled model are not automatically trustworthy. <a href="#/archive/topics/calibration">Check calibration.</a></li>
  </ul>`);
  seeAlso(main, [['concepts', 'linear-regression', 'Linear regression'], ['concepts', 'gradient-descent', 'Gradient descent'], ['concepts', 'loss-functions', 'Loss functions'], ['concepts', 'calibration', 'Calibration'], ['concepts', 'svm', 'Support vector machines']]);
}
