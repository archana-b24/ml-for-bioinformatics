// Regularization: L1 diamond vs L2 circle touching elliptical loss contours, and ridge vs lasso
// coefficient paths (lasso by coordinate descent) on a small regression with correlated features.
import { header, figure, slider, stats, section, steps, callout, math, seeAlso, legend, tip, untip, el } from '../../lib/ui.js';
import { frame, cat, css, ink, ink2, muted } from '../../lib/plot.js';
import { rng, solve, linspace, mean, sd } from '../../lib/data.js';
const d3 = window.d3;

// ---- figure 1: a 2-parameter quadratic loss Q(w) = (w − b̂)ᵀ A (w − b̂) ----
const BHAT = [2, 0.5];
const A = [[1, 0.5], [0.5, 0.6]];
const Q = (w1, w2) => { const a = w1 - BHAT[0], b = w2 - BHAT[1]; return A[0][0] * a * a + 2 * A[0][1] * a * b + A[1][1] * b * b; };
const gradQ = ([w1, w2]) => { const a = w1 - BHAT[0], b = w2 - BHAT[1]; return [2 * (A[0][0] * a + A[0][1] * b), 2 * (A[0][1] * a + A[1][1] * b)]; };

// Euclidean projection onto the L1 ball of radius t (Duchi et al. 2008).
function projL1(v, t) {
  if (Math.abs(v[0]) + Math.abs(v[1]) <= t) return v.slice();
  const u = v.map(Math.abs).sort((a, b) => b - a);
  let css_ = 0, theta = 0;
  for (let k = 0; k < u.length; k++) { css_ += u[k]; const th = (css_ - t) / (k + 1); if (u[k] - th > 0) theta = th; }
  return v.map((x) => Math.sign(x) * Math.max(0, Math.abs(x) - theta));
}
const projL2 = (v, t) => { const n = Math.hypot(...v); return n <= t ? v.slice() : v.map((x) => (x * t) / n); };
// Projected gradient descent: exact enough for a 2-D convex problem.
function constrainedMin(proj, t) {
  let w = proj([0, 0], t); const eta = 0.35;
  for (let i = 0; i < 3000; i++) { const g = gradQ(w); w = proj([w[0] - eta * g[0], w[1] - eta * g[1]], t); }
  return w.map((x) => (Math.abs(x) < 1e-9 ? 0 : x));
}

// ---- figure 2: synthetic "expression" regression with correlated genes ----
const GENES = ['gene A', 'gene B', 'gene C', 'gene D', 'gene E', 'gene F'];
const BETA = [1.5, 0, -1.0, 0.6, 0, 0];
function makeData(n = 60, seed = 21) {
  const r = rng(seed); const X = [], y = [];
  for (let i = 0; i < n; i++) {
    const z1 = r.normal(), z2 = r.normal();
    const x = [z1, 0.9 * z1 + Math.sqrt(1 - 0.81) * r.normal(), z2, 0.7 * z2 + Math.sqrt(1 - 0.49) * r.normal(), r.normal(), r.normal()];
    X.push(x); y.push(x.reduce((s, v, j) => s + v * BETA[j], 0) + r.normal(0, 1));
  }
  // standardise columns (population sd) and centre y, so penalties treat every gene alike
  const p = X[0].length;
  for (let j = 0; j < p; j++) { const col = X.map((row) => row[j]); const m = mean(col), s = Math.sqrt(mean(col.map((v) => (v - m) ** 2))); X.forEach((row) => { row[j] = (row[j] - m) / s; }); }
  const my = mean(y);
  return { X, y: y.map((v) => v - my), n, p };
}
function corr(X, a, b) { const ca = X.map((r) => r[a]), cb = X.map((r) => r[b]); return mean(ca.map((v, i) => v * cb[i])); } // standardised
function ridgePath({ X, y, n, p }, lam) {
  const G = d3.range(p).map((a) => d3.range(p).map((b) => mean(X.map((r) => r[a] * r[b])) + (a === b ? lam : 0)));
  return solve(G, d3.range(p).map((a) => mean(X.map((r, i) => r[a] * y[i]))));
}
// Lasso by cyclic coordinate descent on (1/2n)||y − Xb||² + λ||b||₁ (columns have mean 0, variance 1).
function lassoCD({ X, y, n, p }, lam, warm) {
  const b = warm ? warm.slice() : new Array(p).fill(0);
  const res = y.map((v, i) => v - X[i].reduce((s, x, j) => s + x * b[j], 0));
  for (let it = 0; it < 500; it++) {
    let maxd = 0;
    for (let j = 0; j < p; j++) {
      let rho = 0; for (let i = 0; i < n; i++) rho += X[i][j] * (res[i] + X[i][j] * b[j]); rho /= n;
      const nb = Math.sign(rho) * Math.max(0, Math.abs(rho) - lam), d = nb - b[j];
      if (d !== 0) { for (let i = 0; i < n; i++) res[i] -= X[i][j] * d; b[j] = nb; maxd = Math.max(maxd, Math.abs(d)); }
    }
    if (maxd < 1e-9) break;
  }
  return b;
}

let clipN = 0;
function clipTo(F) {
  const id = 'rg-clip-' + (++clipN);
  F.svg.append('defs').append('clipPath').attr('id', id).append('rect').attr('width', F.w).attr('height', F.h);
  return id;
}
const fmt = (v) => (v === 0 ? '0 (exactly)' : v.toFixed(2));

export default async function render(main) {
  header(main, 'Regularization (L1 / L2)',
    'Regularization adds a penalty on the size of the weights, so the model must trade a little training fit for simpler coefficients. <b>Ridge (L2)</b> shrinks all weights smoothly towards zero; the <b>lasso (L1)</b> sets many of them to <b>exactly</b> zero, which is why it is used to pick biomarker panels. The reason is pure geometry.');

  // ================= figure 1 =================
  const fig = figure(main, 'Why the lasso lands on a corner',
    'Two weights w₁, w₂. The ellipses are contours of the training loss (darker = lower), centred on the unpenalised least-squares solution (×). The shaded shape is the allowed region: the L1 diamond |w₁| + |w₂| ≤ t or the L2 disc w₁² + w₂² ≤ t². The regularised solution (orange) is where the lowest ellipse just touches the shape; the bold ellipse is that touching contour. A smaller t is the same as a larger penalty λ.');
  let t = 1.2;
  slider(fig.controls, { label: 'Constraint size t (smaller = stronger penalty λ)', min: 0.2, max: 2.8, step: 0.02, value: t, fmt: (v) => v.toFixed(2), onInput: (v) => { t = v; paint1(); } });
  const r1 = stats(fig.root);
  const row = el('div', { class: 'fig-row' }); fig.body.append(row);
  const panels = [['L1 (lasso): diamond', projL1], ['L2 (ridge): disc', projL2]].map(([title, proj]) => {
    const d = el('div'); row.append(d);
    d.append(el('div', { style: 'font-size:13px;color:var(--ink-2);text-align:center;font-weight:600' }, title));
    const F = frame(d, { width: 460, height: 450, x: [-1.2, 2.8], y: [-1.2, 2.8], xLabel: 'weight w₁', yLabel: 'weight w₂', grid: false });
    return { F, proj, l1: proj === projL1 };
  });
  legend(fig.body, [['allowed region', d3.color(cat(0)).copy({ opacity: 0.45 }).formatRgb()], ['unpenalised solution (×)', ink()], ['regularised solution', cat(1)], ['contour that touches the region', ink2()]]);

  const NG = 100, lo = -1.2, hi = 2.8, vals = new Array(NG * NG);
  for (let j = 0; j < NG; j++) for (let i = 0; i < NG; i++) vals[j * NG + i] = Q(lo + (hi - lo) * i / (NG - 1), hi - (hi - lo) * j / (NG - 1));
  const vmax = d3.max(vals);
  const levels = d3.range(1, 15).map((k) => vmax * (k / 15) ** 2.4);
  panels.forEach(({ F }) => {
    const proj = d3.geoTransform({ point(px, py) { this.stream.point(F.x(lo + (hi - lo) * px / (NG - 1)), F.y(hi - (hi - lo) * py / (NG - 1))); } });
    F.path = d3.geoPath(proj);
    const mix = d3.interpolateRgb(css('--surface'), css('--accent'));
    const shade = (i) => mix(0.42 * (1 - (i + 1) / (levels.length + 1)));
    F.layer('bgfill').append('rect').attr('width', F.w).attr('height', F.h).attr('fill', mix(0.42)); // lowest-loss core
    F.layer('contours').selectAll('path').data(d3.contours().size([NG, NG]).thresholds(levels)(vals)).join('path').attr('d', F.path)
      .attr('fill', (d, i) => shade(i)).attr('stroke', muted()).attr('stroke-opacity', 0.5).attr('stroke-width', 0.7);
    F.layer('axes0').selectAll('line').data([0, 1]).join('line')
      .attr('x1', (k) => (k ? F.x(0) : 0)).attr('x2', (k) => (k ? F.x(0) : F.w)).attr('y1', (k) => (k ? 0 : F.y(0))).attr('y2', (k) => (k ? F.h : F.y(0)))
      .attr('stroke', ink2()).attr('stroke-width', 1);
    F.layer('region'); F.layer('touch'); F.layer('pts');
  });

  function paint1() {
    const out = panels.map((P) => {
      const { F, proj, l1 } = P;
      const w = constrainedMin(proj, t), q = Q(...w);
      const shape = l1 ? [[t, 0], [0, t], [-t, 0], [0, -t]] : d3.range(73).map((k) => [t * Math.cos(k * Math.PI / 36), t * Math.sin(k * Math.PI / 36)]);
      F.layer('region').selectAll('path').data([shape]).join('path').attr('d', (s) => d3.line().x((d) => F.x(d[0])).y((d) => F.y(d[1]))(s) + 'Z')
        .attr('fill', cat(0)).attr('fill-opacity', 0.28).attr('stroke', cat(0)).attr('stroke-width', 2);
      F.layer('touch').selectAll('path').data(d3.contours().size([NG, NG]).thresholds([q])(vals)).join('path').attr('d', F.path)
        .attr('fill', 'none').attr('stroke', ink2()).attr('stroke-width', 2);
      const pts = F.layer('pts');
      pts.selectAll('text.ols').data([0]).join('text').attr('class', 'ols').attr('x', F.x(BHAT[0])).attr('y', F.y(BHAT[1]) + 6).attr('text-anchor', 'middle')
        .attr('font-size', 18).attr('font-weight', 700).attr('fill', ink()).text('×');
      pts.selectAll('circle').data([w]).join('circle').attr('cx', (d) => F.x(d[0])).attr('cy', (d) => F.y(d[1])).attr('r', 7)
        .attr('fill', cat(1)).attr('stroke', css('--surface')).attr('stroke-width', 2)
        .on('mousemove', (e, d) => tip(e, `w₁ = ${d[0].toFixed(3)}<br>w₂ = ${d[1].toFixed(3)}<br>loss = ${Q(...d).toFixed(3)}`)).on('mouseleave', untip);
      return w;
    });
    const [wl, wr] = out;
    r1.set({ 't': t.toFixed(2), 'lasso w₁': fmt(wl[0]), 'lasso w₂': fmt(wl[1]), 'ridge w₁': fmt(wr[0]), 'ridge w₂': fmt(wr[1]),
      'lasso status': wl[1] === 0 && wl[0] === 0 ? 'both zero' : wl[1] === 0 ? 'corner: w₂ = 0' : Math.abs(wl[0]) + Math.abs(wl[1]) < t - 1e-6 ? 'unconstrained' : 'edge: both non-zero' });
  }
  paint1();

  // ================= figure 2 =================
  const D = makeData();
  const LG = linspace(-3, 2.5, 111);
  const lamMax = d3.max(d3.range(D.p), (j) => Math.abs(mean(D.X.map((r, i) => r[j] * D.y[i]))));
  const RP = LG.map((l) => ridgePath(D, 10 ** l));
  const LP = []; { let warm = null; for (let k = LG.length - 1; k >= 0; k--) { warm = lassoCD(D, 10 ** LG[k], warm); LP[k] = warm.slice(); } }
  const entry = d3.range(D.p).map((j) => { let k = LG.length - 1; while (k >= 0 && LP[k][j] === 0) k--; return { j, at: k < 0 ? null : LG[k] }; })
    .filter((e) => e.at !== null).sort((a, b) => b.at - a.at);

  const fig2 = figure(main, 'Coefficient paths: ridge vs lasso',
    `Sixty samples, six standardised genes. The outcome truly depends on gene A (+1.5), gene C (−1.0) and gene D (+0.6). <b>Gene B is a near-copy of gene A</b> (correlation ${corr(D.X, 0, 1).toFixed(2)}) with no effect of its own; gene D is correlated with gene C (${corr(D.X, 2, 3).toFixed(2)}); genes E and F are noise. Each line is one coefficient as the penalty λ grows (left to right, log scale). The lasso was solved by coordinate descent, warm-started along the path.`);
  let lamI = 50;
  slider(fig2.controls, { label: 'Penalty log₁₀ λ', min: 0, max: LG.length - 1, step: 1, value: lamI, fmt: (i) => LG[i].toFixed(2), onInput: (v) => { lamI = v; paint2(); } });
  const r2 = stats(fig2.root);
  const row2 = el('div', { class: 'fig-row' }); fig2.body.append(row2);
  const yr = [d3.min([...RP.flat(), ...LP.flat()]) - 0.1, d3.max([...RP.flat(), ...LP.flat()]) + 0.1];
  const charts = [['Ridge (L2)', RP], ['Lasso (L1)', LP]].map(([title, paths]) => {
    const d = el('div'); row2.append(d);
    d.append(el('div', { style: 'font-size:13px;color:var(--ink-2);text-align:center;font-weight:600' }, title));
    const F = frame(d, { width: 480, height: 330, x: [LG[0], LG[LG.length - 1]], y: yr, xLabel: 'log₁₀ λ  (stronger penalty →)', yLabel: 'coefficient' });
    F.layer('zero').append('line').attr('x1', 0).attr('x2', F.w).attr('y1', F.y(0)).attr('y2', F.y(0)).attr('stroke', ink2()).attr('stroke-width', 1);
    F.layer('lines').selectAll('path').data(GENES.map((g, j) => j)).join('path').attr('fill', 'none').attr('stroke', (j) => cat(j)).attr('stroke-width', 2)
      .attr('d', (j) => d3.line().x((d, k) => F.x(LG[k])).y((d) => F.y(d[j]))(paths))
      .on('mousemove', (e, j) => { const [mx] = d3.pointer(e, F.g.node()); const k = Math.max(0, Math.min(LG.length - 1, Math.round((F.x.invert(mx) - LG[0]) / (LG[1] - LG[0])))); tip(e, `<b>${GENES[j]}</b> (true ${BETA[j]})<br>log₁₀ λ = ${LG[k].toFixed(2)}: coef ${paths[k][j].toFixed(3)}`); })
      .on('mouseleave', untip);
    return { F, paths };
  });
  legend(fig2.body, GENES.map((g, j) => [`${g} (true ${BETA[j]})`, cat(j)]));

  function paint2() {
    charts.forEach(({ F, paths }) => {
      F.layer('marker').selectAll('line').data([LG[lamI]]).join('line').attr('x1', (d) => F.x(d)).attr('x2', (d) => F.x(d)).attr('y1', 0).attr('y2', F.h)
        .attr('stroke', ink()).attr('stroke-dasharray', '4 3').attr('stroke-width', 1.2);
      F.layer('dots').selectAll('circle').data(paths[lamI].map((v, j) => ({ v, j }))).join('circle').attr('cx', F.x(LG[lamI])).attr('cy', (d) => F.y(d.v)).attr('r', 4.5)
        .attr('fill', (d) => cat(d.j)).attr('stroke', css('--surface')).attr('stroke-width', 1.5)
        .on('mousemove', (e, d) => tip(e, `<b>${GENES[d.j]}</b>: ${d.v.toFixed(3)}`)).on('mouseleave', untip);
    });
    const lb = LP[lamI], rb = RP[lamI];
    const sel = GENES.filter((g, j) => lb[j] !== 0).map((g) => g.replace('gene ', ''));
    r2.set({ 'λ': (10 ** LG[lamI]).toPrecision(2), 'λ_max (lasso all zero above)': lamMax.toFixed(2), 'lasso non-zero': `${sel.length} / 6 (${sel.join(', ') || 'none'})`, 'ridge non-zero': `${rb.filter((v) => Math.abs(v) > 1e-12).length} / 6`,
      'A, B (ridge)': `${rb[0].toFixed(2)}, ${rb[1].toFixed(2)}`, 'A, B (lasso)': `${lb[0].toFixed(2)}, ${lb[1].toFixed(2)}`,
      'lasso entry order': entry.map((e) => e.j).map((j) => GENES[j].replace('gene ', '')).join(' → ') });
  }
  paint2();

  section(main, 'What to notice', `<ul>
    <li>In the geometry figure, with t at its starting value of 1.20, the lasso solution sits on the <b>corner</b> of the diamond at (1.20, 0): w₂ is exactly zero. The ridge solution at the same t, (1.08, 0.53), keeps both weights. Ellipses touch a round disc at a generic point, but a diamond's corners stick out towards them.</li>
    <li>Slide t upward. The lasso stays on the corner until t ≈ 1.9, then slides along an edge with both weights non-zero. The constraint stops binding once t reaches the size of × itself: t ≥ 2.06 (its length) for ridge and t ≥ 2.5 (|2| + |0.5|) for the lasso. From there on the solution is just ×.</li>
    <li>In the paths, the ridge lines all shrink smoothly towards zero and never reach it. The lasso lines hit zero one by one and stay there: by λ = 1 only gene A survives, and above λ = 1.43 every coefficient is exactly zero. The readout lists the order in which genes first enter as λ decreases.</li>
    <li>Watch genes A and B at the starting λ ≈ 0.32. Ridge splits the signal between the correlated pair (A 0.73, B 0.52), so B gets a sizeable coefficient though it has no effect of its own. The lasso gives almost all of it to A (A 1.10, B 0.03), and B drops out entirely as λ shrinks further. With a different sample it might pick B instead: lasso selection among correlated genes is unstable.</li>
  </ul>`);

  steps(main, 'How it works', [
    'Start from the ordinary loss, e.g. the mean squared error of a linear model.',
    'Add a penalty: <b>ridge</b> adds λ·Σwⱼ², the <b>lasso</b> adds λ·Σ|wⱼ|. <b>Elastic net</b> adds a mix of both.',
    'Equivalently, minimise the loss subject to Σwⱼ² ≤ t² or Σ|wⱼ| ≤ t. Each λ corresponds to some t: larger λ, smaller t.',
    'Ridge has a closed form, (XᵀX + λI)⁻¹Xᵀy. Adding λ to the diagonal also makes the problem solvable when genes outnumber samples.',
    'The lasso has no closed form. <b>Coordinate descent</b> cycles through the weights; each update is a least-squares step followed by <b>soft-thresholding</b>, w ← sign(ρ)·max(0, |ρ| − λ), which snaps small values to exactly zero.',
    'Choose λ by <a href="#/archive/topics/cross-validation">cross-validation</a>. Always standardise features first, or the penalty hits small-scale features hardest.',
  ]);
  math(main, 'ridge:  min  (1/2n)‖y − Xw‖² + (λ/2)‖w‖₂²\nlasso:  min  (1/2n)‖y − Xw‖² + λ‖w‖₁\nlasso coordinate update (standardised X):\nwⱼ ← S(ρⱼ, λ),   ρⱼ = (1/n)·xⱼᵀ(residual + xⱼwⱼ),  S(ρ, λ) = sign(ρ)·max(0, |ρ| − λ)');

  callout(main, '', 'In biology', 'Biomarker discovery from bulk RNA-seq is the textbook lasso problem: 20,000 genes, 100 patients, and the clinic wants a 10-gene panel. <code>glmnet</code> (R) and scikit-learn\'s <code>LogisticRegressionCV(penalty="l1")</code> fit exactly these paths. Genes in the same pathway are highly correlated, so the lasso\'s chosen panel changes between bootstrap samples; <b>elastic net</b> and <b>stability selection</b> exist for this. Ridge-type shrinkage is everywhere too: polygenic risk scores, eQTL models, and DESeq2\'s log-fold-change shrinkage (a prior that pulls noisy estimates towards zero).');
  callout(main, 'warn', 'Pitfalls', '<ul><li>Not standardising: the penalty then depends on the units of each gene, and highly expressed genes escape it.</li><li>Reading lasso zeros as "not associated". A zero often just means a correlated partner was picked instead.</li><li>Choosing λ, or pre-filtering genes, on the full dataset before cross-validation leaks information; it must happen inside each fold (<a href="#/archive/topics/cross-validation">nested CV</a>).</li><li>Lasso coefficients are biased towards zero. If you need effect sizes, refit the selected genes without penalty, and validate on independent data.</li></ul>');
  seeAlso(main, [['concepts', 'bias-variance', 'Under- vs overfitting'], ['concepts', 'scaling', 'Feature scaling'], ['concepts', 'cross-validation', 'Cross-validation'], ['omics', 'bulk-feature-selection', 'Biomarker feature selection']]);
}
