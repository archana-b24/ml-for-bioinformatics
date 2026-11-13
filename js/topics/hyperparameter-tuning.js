// Hyperparameter tuning: grid vs random search on a 2D response surface, and Bayesian optimisation with a tiny GP.
import { header, figure, slider, select, button, stats, section, steps, callout, math, legend, seeAlso, every, tip, untip, el } from '../lib/ui.js';
import { frame, cat, css, ink, muted } from '../lib/plot.js';
import { rng, linspace } from '../lib/data.js';
const d3 = window.d3;

// ---------- 2D synthetic response: validation accuracy over (log10 learning rate, dropout) ----------
const LR = [-5, -1], DR = [0, 0.6];
const acc = (lr, dr) => 0.62 + 0.3 * Math.exp(-(((lr + 2.35) / 0.4) ** 2)) - 0.05 * (dr - 0.3) ** 2;
const gridPts = (b) => { const k = Math.round(Math.sqrt(b)); const xs = linspace(LR[0], LR[1], k), ys = linspace(DR[0], DR[1], k); return xs.flatMap((x) => ys.map((y) => ({ x, y }))); };
const randPts = (b, seed) => { const r = rng(seed); return d3.range(b).map(() => ({ x: LR[0] + (LR[1] - LR[0]) * r(), y: DR[0] + (DR[1] - DR[0]) * r() })); };
const best = (pts) => d3.max(pts, (p) => acc(p.x, p.y));

// ---------- 1D objective for Bayesian optimisation: a broad local bump and a narrow global peak ----------
const obj = (x) => 0.55 * Math.exp(-(((x - 0.25) / 0.13) ** 2)) + 0.85 * Math.exp(-(((x - 0.74) / 0.06) ** 2)) + 0.08 * Math.sin(9 * x);
const XS = linspace(0, 1, 301);
const TRUE_MAX = d3.max(XS, obj), X_STAR = XS[d3.maxIndex(XS, obj)];
const START = [0.08, 0.4, 0.95];

function erfc(x) {
  const z = Math.abs(x), t = 1 / (1 + 0.5 * z);
  const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 +
    t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
  return x >= 0 ? r : 2 - r;
}
const Phi = (z) => 0.5 * erfc(-z / Math.SQRT2);
const phi = (z) => Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI);

/** Gaussian-process posterior with an RBF kernel. Observations are standardised; returns mean/sd on XS in original units. */
function gp(xs, ys, ell, noise = 1e-4) {
  const n = xs.length, mu0 = d3.mean(ys), s0 = d3.deviation(ys) || 1;
  const yz = ys.map((y) => (y - mu0) / s0);
  const k = (a, b) => Math.exp(-((a - b) ** 2) / (2 * ell * ell));
  // Cholesky of K + noise I
  const L = d3.range(n).map(() => new Array(n).fill(0));
  for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) {
    let s = k(xs[i], xs[j]) + (i === j ? noise : 0);
    for (let m = 0; m < j; m++) s -= L[i][m] * L[j][m];
    L[i][j] = i === j ? Math.sqrt(Math.max(s, 1e-12)) : s / L[j][j];
  }
  const fwd = (b) => { const z = new Array(n); for (let i = 0; i < n; i++) { let s = b[i]; for (let m = 0; m < i; m++) s -= L[i][m] * z[m]; z[i] = s / L[i][i]; } return z; };
  const bwd = (z) => { const x = new Array(n); for (let i = n - 1; i >= 0; i--) { let s = z[i]; for (let m = i + 1; m < n; m++) s -= L[m][i] * x[m]; x[i] = s / L[i][i]; } return x; };
  const alpha = bwd(fwd(yz));
  return XS.map((x) => {
    const ks = xs.map((xi) => k(x, xi));
    const v = fwd(ks);
    const m = ks.reduce((s, kv, i) => s + kv * alpha[i], 0);
    const varz = Math.max(1e-10, 1 - v.reduce((s, vv) => s + vv * vv, 0));
    return { x, mu: mu0 + s0 * m, sd: s0 * Math.sqrt(varz) };
  });
}
function acquisition(post, ybest, kind) {
  return post.map((p) => {
    if (kind === 'ucb') return p.mu + 2 * p.sd;
    const imp = p.mu - ybest - 0.01, z = imp / p.sd;
    return imp * Phi(z) + p.sd * phi(z); // expected improvement
  });
}
function nextPoint(xs, ys, ell, kind) {
  const post = gp(xs, ys, ell), a = acquisition(post, d3.max(ys), kind);
  return { post, a, x: XS[d3.maxIndex(a)] };
}
/** Run BO for a budget and report the first evaluation (1-based) that lands within 0.02 of the global optimum's value. */
function boFindsAt(ell, kind, budget = 15) {
  const xs = START.slice(), ys = xs.map(obj);
  for (let t = xs.length; t < budget; t++) { const nx = nextPoint(xs, ys, ell, kind).x; xs.push(nx); ys.push(obj(nx)); if (TRUE_MAX - obj(nx) < 0.02) return t + 1; }
  return null;
}

export default async function render(main) {
  header(main, 'Hyperparameter tuning',
    'Hyperparameters (learning rate, depth, regularisation strength) are set before training, so you choose them by trying settings and scoring each on validation data. Since every try costs a full training run, the question is <b>where to look next</b>. Random search beats a grid when only a few knobs matter; Bayesian optimisation <b>learns from past tries</b> to choose the next one.');

  // ================= Figure 1: grid vs random =================
  let budget = 9, seed = 3, show = 'both';
  const fig = figure(main, 'Grid vs random search with the same budget', '');
  slider(fig.controls, { label: 'Budget (training runs)', min: 0, max: 2, step: 1, value: 0, fmt: (v) => [9, 16, 25][v], onInput: (v) => { budget = [9, 16, 25][v]; paint1(); } });
  select(fig.controls, { label: 'Show', value: 'both', options: [['both', 'Both'], ['grid', 'Grid only'], ['random', 'Random only']], onChange: (v) => { show = v; paint1(); } });
  button(fig.controls, 'New random draw', () => { seed++; paint1(); });
  const r1 = stats(fig.root);
  legend(fig.root, [['grid search', cat(0)], ['random search', cat(1)]]);
  const row = el('div', { class: 'fig-row' }); fig.body.append(row);
  const aDiv = el('div'), bDiv = el('div'); row.append(aDiv, bDiv);
  const S = frame(aDiv, { width: 480, height: 400, x: LR, y: DR, xLabel: 'log₁₀ learning rate (matters)', yLabel: 'dropout (barely matters)', grid: false });
  const P = frame(bDiv, { width: 480, height: 400, x: LR, y: [0.55, 0.95], xLabel: 'log₁₀ learning rate', yLabel: 'validation accuracy' });
  {
    const n = 80, vals = new Array(n * n);
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) vals[j * n + i] = acc(LR[0] + (LR[1] - LR[0]) * i / (n - 1), DR[1] - (DR[1] - DR[0]) * j / (n - 1));
    const levels = d3.range(0.6, 0.93, 0.025);
    const shade = d3.scaleLinear().domain([0.6, 0.92]).range([css('--surface-2'), css('--accent-soft')]);
    const proj = d3.geoTransform({ point(px, py) { this.stream.point(S.x(LR[0] + (LR[1] - LR[0]) * px / (n - 1)), S.y(DR[1] - (DR[1] - DR[0]) * py / (n - 1))); } });
    S.layer('contours').selectAll('path').data(d3.contours().size([n, n]).thresholds(levels)(vals)).join('path').attr('d', d3.geoPath(proj))
      .attr('fill', (d) => shade(d.value)).attr('stroke', css('--axis')).attr('stroke-width', 0.6);
    S.layer('note').append('text').attr('x', S.x(-2.35)).attr('y', 14).attr('text-anchor', 'middle').attr('font-size', 11.5).attr('fill', ink()).text('narrow good band');
    P.layer('curve').append('path').datum(linspace(LR[0], LR[1], 200)).attr('fill', 'none').attr('stroke', muted()).attr('stroke-width', 2)
      .attr('d', d3.line().x((x) => P.x(x)).y((x) => P.y(acc(x, 0.3))));
  }
  // how often does random beat grid? (fixed reference over 500 draws)
  const winRate = (b) => d3.mean(d3.range(500), (s) => (best(randPts(b, 1000 + s)) > best(gridPts(b)) ? 1 : 0));
  const WIN = { 9: winRate(9), 16: winRate(16), 25: winRate(25) };

  function paint1() {
    const G = gridPts(budget), R = randPts(budget, seed);
    const sets = [['grid', G, 0], ['random', R, 1]].filter(([k]) => show === 'both' || show === k);
    const all = sets.flatMap(([k, pts, c]) => pts.map((p) => ({ ...p, k, c, a: acc(p.x, p.y) })));
    const bestOf = (k) => all.filter((d) => d.k === k).reduce((m, d) => (d.a > m.a ? d : m), { a: -1 });
    const sym = (sel, F, yf) => sel.join('path')
      .attr('d', (d) => d3.symbol(d.k === 'grid' ? d3.symbolSquare : d3.symbolCircle, d.k === 'grid' ? 70 : 64)())
      .attr('transform', (d) => `translate(${F.x(d.x)},${F.y(yf(d))})`).attr('fill', (d) => cat(d.c)).attr('stroke', css('--surface')).attr('stroke-width', 1.5)
      .on('mousemove', (e, d) => tip(e, `${d.k} search<br>learning rate 10^${d.x.toFixed(2)} · dropout ${d.y.toFixed(2)}<br>accuracy ${d.a.toFixed(3)}`)).on('mouseleave', untip);
    sym(S.layer('pts').selectAll('path').data(all), S, (d) => d.y);
    sym(P.layer('pts').selectAll('path').data(all), P, (d) => d.a);
    // best markers
    const bests = sets.map(([k]) => bestOf(k));
    S.layer('best').selectAll('circle').data(bests).join('circle').attr('cx', (d) => S.x(d.x)).attr('cy', (d) => S.y(d.y)).attr('r', 10)
      .attr('fill', 'none').attr('stroke', ink()).attr('stroke-width', 1.5);
    const gb = best(G), rb = best(R);
    const distinct = (pts) => new Set(pts.map((p) => p.x.toFixed(4))).size;
    r1.set({
      'grid: best accuracy': gb.toFixed(3), 'random: best accuracy': rb.toFixed(3), 'best possible': '0.920',
      'distinct learning rates tried (grid / random)': `${distinct(G)} / ${distinct(R)}`,
      [`random beats grid (500 draws, budget ${budget})`]: Math.round(WIN[budget] * 100) + '%',
    });
    fig.caption(`Left: the validation-accuracy landscape (lighter = better) with the ${budget} configurations each method tries; rings mark each method's best. Right: the same tries projected onto the learning-rate axis. A ${Math.sqrt(budget)}×${Math.sqrt(budget)} grid spends its budget re-testing just ${distinct(G)} learning rates at different dropouts, which barely matters; random search tries ${distinct(R)} different learning rates.`);
  }
  paint1();

  // ================= Figure 2: Bayesian optimisation =================
  let ell = 0.08, kind = 'ei', xs = [], ys = [], stopAuto = null;
  const fig2 = figure(main, 'Bayesian optimisation: a Gaussian process decides where to sample next', '');
  slider(fig2.controls, { label: 'GP length-scale ℓ', min: 0.02, max: 0.3, step: 0.01, value: ell, fmt: (v) => v.toFixed(2), onInput: (v) => { ell = v; paint2(); } });
  select(fig2.controls, { label: 'Acquisition', value: 'ei', options: [['ei', 'Expected improvement'], ['ucb', 'Upper confidence bound (μ + 2σ)']], onChange: (v) => { kind = v; paint2(); } });
  button(fig2.controls, 'Next sample', () => { sampleNext(); }, true);
  button(fig2.controls, 'Auto-run', () => { if (stopAuto) stopAuto(); stopAuto = every(650, () => { if (xs.length >= 15) { stopAuto(); stopAuto = null; return; } sampleNext(); }); });
  button(fig2.controls, 'Reset', () => { if (stopAuto) { stopAuto(); stopAuto = null; } reset2(); });
  const r2 = stats(fig2.root);
  legend(fig2.root, [['true objective (hidden from the optimiser)', muted()], ['GP mean ± 2 sd', cat(0)], ['evaluated points', ink()], ['acquisition / next point', cat(1)]]);
  const T = frame(fig2.body, { width: 980, height: 300, x: [0, 1], y: [-0.4, 1.3], xLabel: '', yLabel: 'validation score', margin: { bottom: 22 } });
  const A = frame(fig2.body, { width: 980, height: 150, x: [0, 1], y: [0, 1], xLabel: 'hyperparameter (scaled to 0–1)', yLabel: 'acquisition', yTicks: 2 });
  T.layer('band'); T.layer('true').append('path').datum(XS).attr('fill', 'none').attr('stroke', muted()).attr('stroke-width', 2).attr('stroke-dasharray', '5 4')
    .attr('d', d3.line().x((x) => T.x(x)).y((x) => T.y(obj(x))));
  T.layer('mean'); T.layer('next'); T.layer('pts');

  let cur = null;
  function reset2() { xs = START.slice(); ys = xs.map(obj); paint2(); }
  function sampleNext() { if (!cur) return; xs.push(cur.x); ys.push(obj(cur.x)); paint2(); }
  function paint2() {
    cur = nextPoint(xs, ys, ell, kind);
    const { post, a } = cur;
    T.layer('band').selectAll('path').data([post]).join('path').attr('fill', cat(0)).attr('fill-opacity', 0.16)
      .attr('d', d3.area().x((p) => T.x(p.x)).y0((p) => T.y(Math.max(-0.4, p.mu - 2 * p.sd))).y1((p) => T.y(Math.min(1.3, p.mu + 2 * p.sd))));
    T.layer('mean').selectAll('path').data([post]).join('path').attr('fill', 'none').attr('stroke', cat(0)).attr('stroke-width', 2)
      .attr('d', d3.line().x((p) => T.x(p.x)).y((p) => T.y(Math.max(-0.4, Math.min(1.3, p.mu)))));
    const pts = xs.map((x, i) => ({ x, y: ys[i], i }));
    T.layer('pts').selectAll('circle').data(pts).join('circle').attr('cx', (d) => T.x(d.x)).attr('cy', (d) => T.y(d.y)).attr('r', (d) => (d.i === xs.length - 1 && xs.length > START.length ? 6.5 : 5))
      .attr('fill', ink()).attr('stroke', css('--surface')).attr('stroke-width', 1.5)
      .on('mousemove', (e, d) => tip(e, `evaluation ${d.i + 1}${d.i < START.length ? ' (initial)' : ''}<br>x = ${d.x.toFixed(3)} · score ${d.y.toFixed(3)}`)).on('mouseleave', untip);
    T.layer('next').selectAll('line').data([cur.x]).join('line').attr('x1', (x) => T.x(x)).attr('x2', (x) => T.x(x)).attr('y1', 0).attr('y2', T.h)
      .attr('stroke', cat(1)).attr('stroke-width', 1.5).attr('stroke-dasharray', '4 3');
    const amax = d3.max(a) || 1, amin = kind === 'ucb' ? d3.min(a) : 0;
    A.y.domain([amin, amax * 1.05 || 1]);
    A.g.select('.axis.y').call(d3.axisLeft(A.y).ticks(2).tickFormat(''));
    A.layer('a').selectAll('path').data([a]).join('path').attr('fill', cat(1)).attr('fill-opacity', 0.25).attr('stroke', cat(1)).attr('stroke-width', 2)
      .attr('d', d3.area().x((v, i) => A.x(XS[i])).y0(A.h).y1((v) => A.y(v)));
    A.layer('m').selectAll('circle').data([cur.x]).join('circle').attr('cx', (x) => A.x(x)).attr('cy', A.y(d3.max(a))).attr('r', 5)
      .attr('fill', cat(1)).attr('stroke', css('--surface')).attr('stroke-width', 1.5);
    const bi = d3.maxIndex(ys);
    r2.set({ evaluations: xs.length, 'best score so far': ys[bi].toFixed(3), 'at x': xs[bi].toFixed(3), 'true optimum': `${TRUE_MAX.toFixed(3)} at ${X_STAR.toFixed(3)}`, 'next sample at': cur.x.toFixed(3) });
  }
  // reference numbers for the text (deterministic)
  const found = boFindsAt(0.08, 'ei');
  const randFind = d3.mean(d3.range(500), (s) => { const r = rng(5000 + s); return d3.max(d3.range(15), () => obj(r())) > TRUE_MAX - 0.02 ? 1 : 0; });
  fig2.caption(`Top: the GP's belief about the score (line = mean, band = ±2 standard deviations) given the black evaluations. Bottom: the acquisition function; its peak (orange) is the next point to try. With the defaults (ℓ = 0.08, expected improvement) BO lands within 0.02 of the true optimum at evaluation <b>${found ?? '>15'}</b>. Random search with 15 evaluations does so in only ${Math.round(randFind * 100)}% of draws, because the good peak is narrow.`);
  reset2();

  section(main, 'What to notice', `<ul>
    <li>The grid tests only √budget distinct learning rates (3 with a budget of 9), and none of them lands in the narrow good band. Random search tries a new learning rate every time; across 500 random draws it beats the 3×3 grid ${Math.round(WIN[9] * 100)}% of the time. This is Bergstra &amp; Bengio's (2012) argument: when few hyperparameters matter, a grid wastes runs on the ones that do not.</li>
    <li>Now set the budget to <b>16</b>. The 4×4 grid happens to put a learning rate at 10<sup>−2.33</sup>, right in the band, and random search beats it in only ${Math.round(WIN[16] * 100)}% of draws. At 25 the 5×5 grid misses again (random wins ${Math.round(WIN[25] * 100)}%). Grids are not always bad; their luck depends on where the good region happens to fall, which you do not know in advance.</li>
    <li>In the BO panel, press <b>Next sample</b> repeatedly. Early samples go where the band is wide (<b>exploration</b>); once a good region is found, samples cluster there (<b>exploitation</b>). Expected improvement balances the two automatically.</li>
    <li>Set ℓ to <b>0.25</b> and press Reset, then Next sample. The GP now assumes the score varies slowly, so it smooths over the narrow peak and keeps sampling the broad bump near 0.25: after 15 evaluations it has still not found the optimum. Set ℓ = <b>0.02</b> and the opposite happens: the GP trusts only the immediate neighbourhood of each point, so it creeps up the nearest bump in tiny steps and also never reaches the peak. Real BO libraries fit ℓ to the data by maximum likelihood.</li>
    <li>Switch to <b>UCB</b>: it samples where μ + 2σ is highest. With ℓ = 0.08 it takes a slightly different route but also reaches the peak at evaluation 11.</li>
  </ul>`);

  steps(main, 'How it works', [
    'Pick the hyperparameters and a range for each. Use a <b>log scale</b> for anything spanning orders of magnitude (learning rate, regularisation λ, C).',
    'Choose a validation scheme: a held-out split or k-fold CV. Every evaluation is a full train-and-score run.',
    '<b>Grid search:</b> every combination of a few values per hyperparameter. Cost grows exponentially with the number of hyperparameters.',
    '<b>Random search:</b> sample each hyperparameter independently. With the same budget it tries more distinct values of each one.',
    '<b>Bayesian optimisation:</b> fit a surrogate model (here a Gaussian process) to the scores seen so far. It gives a mean and an uncertainty everywhere.',
    'Compute an <b>acquisition function</b> from the surrogate (expected improvement over the current best, or an upper confidence bound) and evaluate the hyperparameter where it is highest. Refit and repeat.',
    'Report the chosen model\'s performance on a test set (or the outer loop of nested CV) that the search never saw.',
  ]);
  math(main, 'GP posterior:  μ(x) = k(x)ᵀ (K + σ²I)⁻¹ y          s²(x) = k(x,x) − k(x)ᵀ (K + σ²I)⁻¹ k(x)\nRBF kernel:    k(a, b) = exp(−(a − b)² / 2ℓ²)\nExpected improvement:  EI(x) = (μ − y_best − ξ) Φ(z) + s φ(z),   z = (μ − y_best − ξ) / s');

  callout(main, '', 'In biology', `<ul>
    <li>Tuning an <b>elastic-net</b> biomarker model on bulk RNA-seq: λ (log scale) matters a lot, the mixing α much less. Random or BO search over (λ, α) beats a coarse grid.</li>
    <li><b>scVI / scANVI</b> have many knobs (latent dimension, layers, dropout, learning rate). scvi-tools integrates <b>Ray Tune</b> for exactly this search, and Optuna (tree-structured Parzen estimator, a BO variant) is common for XGBoost models on omics features.</li>
    <li>Single-cell clustering <b>resolution</b> is a hyperparameter too, but with no labels there is no validation score; people tune it to a silhouette or stability metric, which is weaker.</li>
  </ul>`);
  callout(main, 'warn', 'Pitfalls', `<ul>
    <li><b>Tuning on the test set:</b> the best of 100 configurations scored on the same data is optimistically biased. Use nested cross-validation, especially with small cohorts (n &lt; 200 patients).</li>
    <li>Searching a linear range for a scale parameter wastes nearly all samples at the large end.</li>
    <li>With small validation sets, differences between top configurations are often noise. Prefer the simplest model within one standard error of the best.</li>
    <li>BO's surrogate can be confidently wrong (see the large-ℓ case). Keep some random exploration.</li>
  </ul>`);
  seeAlso(main, [['concepts', 'cross-validation', 'Cross-validation'], ['concepts', 'bias-variance', 'Under- vs overfitting'], ['concepts', 'regularization', 'Regularization'], ['omics', 'bulk-classifier', 'Response classifier & nested CV']]);
}
