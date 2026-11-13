// PCA: rotate a projection axis by hand and watch the captured variance; then PCA on a synthetic 50-gene dataset.
import { header, figure, slider, select, button, checkbox, stats, section, steps, callout, math, legend, seeAlso, animate, tip, untip, el } from '../lib/ui.js';
import { frame, cat, css, ink, ink2, muted } from '../lib/plot.js';
import { rng, eig2 } from '../lib/data.js';
const d3 = window.d3;

/** Symmetric eigen-decomposition (cyclic Jacobi). A: Float64Array n*n. Returns {vals, vecs[k] = Float64Array(n)} sorted desc. */
function symEig(Ain, n) {
  const A = Float64Array.from(Ain), V = new Float64Array(n * n);
  for (let i = 0; i < n; i++) V[i * n + i] = 1;
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += A[p * n + q] ** 2;
    if (off < 1e-20) break;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
      const apq = A[p * n + q]; if (Math.abs(apq) < 1e-15) continue;
      const th = (A[q * n + q] - A[p * n + p]) / (2 * apq);
      const t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1)), c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < n; k++) { // rotate columns p,q
        const akp = A[k * n + p], akq = A[k * n + q];
        A[k * n + p] = c * akp - s * akq; A[k * n + q] = s * akp + c * akq;
      }
      for (let k = 0; k < n; k++) { // rotate rows p,q
        const apk = A[p * n + k], aqk = A[q * n + k];
        A[p * n + k] = c * apk - s * aqk; A[q * n + k] = s * apk + c * aqk;
      }
      for (let k = 0; k < n; k++) {
        const vkp = V[k * n + p], vkq = V[k * n + q];
        V[k * n + p] = c * vkp - s * vkq; V[k * n + q] = s * vkp + c * vkq;
      }
    }
  }
  const order = d3.range(n).sort((a, b) => A[b * n + b] - A[a * n + a]);
  const vecs = order.map((j) => {
    const v = new Float64Array(n); for (let k = 0; k < n; k++) v[k] = V[k * n + j];
    let big = 0; for (let k = 0; k < n; k++) if (Math.abs(v[k]) > Math.abs(v[big])) big = k;
    if (v[big] < 0) for (let k = 0; k < n; k++) v[k] = -v[k]; // sign convention: largest loading positive
    return v;
  });
  return { vals: order.map((j) => Math.max(0, A[j * n + j])), vecs };
}

// ---------- synthetic "expression" data: 3 cell types x 50 genes ----------
const TYPES = ['T cell', 'B cell', 'Monocyte'];
const SIZES = [110, 70, 40];
const GENES = [
  'CD3E', 'CD3D', 'CD2', 'IL7R', 'TRAC', 'LCK', 'CD247', 'IL32',
  'MS4A1', 'CD79A', 'CD79B', 'CD19', 'BANK1', 'CD22', 'PAX5', 'IGHM',
  'LYZ', 'CD14', 'S100A8', 'S100A9', 'FCN1', 'CST3', 'LST1', 'TYROBP',
  'ACTB', 'GAPDH', 'B2M', 'RPL13', 'RPS18', 'EEF1A1',
  ...d3.range(1, 21).map((i) => 'noise' + i),
];
const G = GENES.length;
const geneGroup = (g) => (g < 24 ? Math.floor(g / 8) : g < 30 ? 3 : 4); // 0..2 markers of type, 3 housekeeping, 4 noise
function makeExpression() {
  const r = rng(2024);
  const strength = GENES.map((_, g) => (g < 24 ? 1.4 + 1.2 * r() : 0));
  const cells = [];
  SIZES.forEach((n, t) => { for (let i = 0; i < n; i++) cells.push(t); });
  const X = new Float64Array(cells.length * G);
  cells.forEach((t, i) => {
    for (let g = 0; g < G; g++) {
      const grp = geneGroup(g);
      let v;
      if (grp < 3) v = 0.4 + (grp === t ? strength[g] : 0) + r.normal(0, 0.45);
      else if (grp === 3) v = 3 + r.normal(0, 0.25);
      else v = 1.2 + r.normal(0, 0.6);
      X[i * G + g] = Math.max(0, v); // log-expression is never negative
    }
  });
  return { X, cells, n: cells.length };
}

function pcaOf(X, n, p, scale) {
  const Z = new Float64Array(n * p);
  for (let g = 0; g < p; g++) {
    let m = 0; for (let i = 0; i < n; i++) m += X[i * p + g]; m /= n;
    let s = 0; for (let i = 0; i < n; i++) s += (X[i * p + g] - m) ** 2; s = Math.sqrt(s / (n - 1)) || 1;
    for (let i = 0; i < n; i++) Z[i * p + g] = (X[i * p + g] - m) / (scale ? s : 1);
  }
  const C = new Float64Array(p * p);
  for (let a = 0; a < p; a++) for (let b = a; b < p; b++) {
    let s = 0; for (let i = 0; i < n; i++) s += Z[i * p + a] * Z[i * p + b];
    C[a * p + b] = C[b * p + a] = s / (n - 1);
  }
  const { vals, vecs } = symEig(C, p);
  const total = vals.reduce((s, v) => s + v, 0);
  const scores = (k) => { const v = vecs[k], out = new Float64Array(n); for (let i = 0; i < n; i++) { let s = 0; for (let g = 0; g < p; g++) s += Z[i * p + g] * v[g]; out[i] = s; } return out; };
  return { vals, vecs, ratio: vals.map((v) => v / total), scores };
}

export default async function render(main) {
  header(main, 'PCA: principal component analysis',
    'PCA looks for the direction along which your data <b>spread out the most</b>, projects every point onto it, then repeats at right angles. A few such directions often summarise thousands of correlated measurements.');

  // ================= Figure 1: drag the projection axis =================
  const fig = figure(main, 'Rotate the axis and watch how much variance it keeps',
    'Drag anywhere on the left plot to rotate the axis (orange line). Each point is projected onto it (small orange dots); the grey ticks are what the projection throws away. The right panel sweeps every angle: <b>PC1</b> is the angle at the top of the curve and <b>PC2</b>, at right angles to it, is the bottom.');
  const r1 = rng(11), N = 160, MAJOR = 0.55, ANG = 35 * Math.PI / 180;
  const Zn = d3.range(N).map(() => [r1.normal(), r1.normal()]);
  let minor = 0.18, theta = 100 * Math.PI / 180, pts = [], cov = null, pcAngle = 0, showPC = false, stopAnim = null;

  function makeCloud() {
    pts = Zn.map(([a, b]) => { const u = a * MAJOR, v = b * minor; return { x: u * Math.cos(ANG) - v * Math.sin(ANG), y: u * Math.sin(ANG) + v * Math.cos(ANG) }; });
    const mx = d3.mean(pts, (d) => d.x), my = d3.mean(pts, (d) => d.y);
    pts.forEach((d) => { d.x -= mx; d.y -= my; });
    const sxx = d3.sum(pts, (d) => d.x * d.x) / (N - 1), syy = d3.sum(pts, (d) => d.y * d.y) / (N - 1), sxy = d3.sum(pts, (d) => d.x * d.y) / (N - 1);
    cov = { sxx, syy, sxy, total: sxx + syy };
    const e = eig2(sxx, sxy, syy);
    pcAngle = Math.atan2(e.vecs[0][1], e.vecs[0][0]);
    cov.l1 = e.vals[0]; cov.l2 = e.vals[1];
  }
  const varAt = (t) => { const c = Math.cos(t), s = Math.sin(t); return c * c * cov.sxx + 2 * c * s * cov.sxy + s * s * cov.syy; };
  const deg = (t) => ((t * 180 / Math.PI) % 180 + 180) % 180;

  slider(fig.controls, { label: 'Spread across the main axis', min: 0.03, max: 0.55, step: 0.01, value: minor, fmt: (v) => v.toFixed(2), onInput: (v) => { minor = v; makeCloud(); paint(); } });
  button(fig.controls, 'Snap to PC1', () => snapTo(pcAngle), true);
  button(fig.controls, 'Snap to PC2', () => snapTo(pcAngle + Math.PI / 2));
  checkbox(fig.controls, { label: 'Show PC axes', checked: false, onChange: (c) => { showPC = c; paint(); } });
  const readout = stats(fig.root);

  const row = el('div', { class: 'fig-row' }); fig.body.append(row);
  const left = el('div'), right = el('div'); row.append(left, right);
  const W = 460, H = 430, M = { top: 12, right: 14, bottom: 28, left: 38 };
  const pw = W - M.left - M.right, ph = H - M.top - M.bottom, XR = 1.45, YR = XR * ph / pw; // equal aspect: angles look right
  const F = frame(left, { width: W, height: H, margin: M, x: [-XR, XR], y: [-YR, YR], ariaLabel: 'scatter with rotatable projection axis' });
  const clipId = 'pca-clip'; F.svg.append('defs').append('clipPath').attr('id', clipId).append('rect').attr('width', F.w).attr('height', F.h);
  ['pcs', 'resid', 'axis', 'pts', 'proj', 'handle'].forEach((k) => F.layer(k).attr('clip-path', `url(#${clipId})`));
  F.svg.style('cursor', 'grab').style('touch-action', 'none');
  F.svg.call(d3.drag().on('start drag', (e) => {
    if (stopAnim) { stopAnim(); stopAnim = null; }
    const [mx, my] = d3.pointer(e.sourceEvent, F.g.node());
    const px = F.x.invert(mx), py = F.y.invert(my);
    if (Math.hypot(px, py) > 0.05) { theta = Math.atan2(py, px); paint(); }
  }));
  const C = frame(right, { width: 440, height: H, x: [0, 180], y: [0, 100], xLabel: 'axis angle (degrees)', yLabel: '% of total variance captured' });
  C.g.select('.axis.x').call(d3.axisBottom(C.x).tickValues([0, 45, 90, 135, 180]));

  function snapTo(target) {
    if (stopAnim) stopAnim();
    stopAnim = animate((dt) => {
      let diff = ((target - theta) % Math.PI + 1.5 * Math.PI) % Math.PI - Math.PI / 2; // shortest turn, axis is mod 180°
      theta += diff * Math.min(1, dt / 110);
      if (Math.abs(diff) < 0.002) { theta = target; paint(); stopAnim = null; return false; }
      paint(); return true;
    });
  }

  function paint() {
    const c = Math.cos(theta), s = Math.sin(theta), L = 3;
    const proj = pts.map((d) => { const t = d.x * c + d.y * s; return { t, x: t * c, y: t * s, ox: d.x, oy: d.y }; });
    const pcLines = showPC ? [pcAngle, pcAngle + Math.PI / 2] : [];
    F.layer('pcs').selectAll('line').data(pcLines).join('line')
      .attr('x1', (a) => F.x(-L * Math.cos(a))).attr('y1', (a) => F.y(-L * Math.sin(a))).attr('x2', (a) => F.x(L * Math.cos(a))).attr('y2', (a) => F.y(L * Math.sin(a)))
      .attr('stroke', ink2()).attr('stroke-width', 1.5).attr('stroke-dasharray', (a, i) => (i ? '2 4' : '7 4'));
    F.layer('pcs').selectAll('text').data(pcLines).join('text')
      .attr('x', (a) => F.x(1.25 * Math.cos(a)) + 6).attr('y', (a) => F.y(1.25 * Math.sin(a)) - 6)
      .attr('fill', ink2()).attr('font-size', 12).text((a, i) => (i ? 'PC2' : 'PC1'));
    F.layer('resid').selectAll('line').data(proj).join('line')
      .attr('x1', (d) => F.x(d.ox)).attr('y1', (d) => F.y(d.oy)).attr('x2', (d) => F.x(d.x)).attr('y2', (d) => F.y(d.y))
      .attr('stroke', muted()).attr('stroke-opacity', 0.45).attr('stroke-width', 1);
    F.layer('axis').selectAll('line').data([0]).join('line')
      .attr('x1', F.x(-L * c)).attr('y1', F.y(-L * s)).attr('x2', F.x(L * c)).attr('y2', F.y(L * s))
      .attr('stroke', cat(1)).attr('stroke-width', 2.5);
    F.layer('pts').selectAll('circle').data(proj).join('circle')
      .attr('cx', (d) => F.x(d.ox)).attr('cy', (d) => F.y(d.oy)).attr('r', 4.5).attr('fill', cat(0)).attr('stroke', css('--surface')).attr('stroke-width', 1.2)
      .on('mousemove', (e, d) => tip(e, `point (${d.ox.toFixed(2)}, ${d.oy.toFixed(2)})<br>projection score ${d.t.toFixed(2)}`)).on('mouseleave', untip);
    F.layer('proj').selectAll('circle').data(proj).join('circle')
      .attr('cx', (d) => F.x(d.x)).attr('cy', (d) => F.y(d.y)).attr('r', 2.6).attr('fill', cat(1)).attr('opacity', 0.8);
    F.layer('handle').selectAll('circle').data([1, -1]).join('circle')
      .attr('cx', (k) => F.x(k * 1.2 * c)).attr('cy', (k) => F.y(k * 1.2 * s)).attr('r', 7).attr('fill', css('--surface')).attr('stroke', cat(1)).attr('stroke-width', 2.5);

    // variance-vs-angle curve
    const curve = d3.range(0, 181, 1).map((a) => ({ x: a, y: 100 * varAt(a * Math.PI / 180) / cov.total }));
    C.layer('curve').selectAll('path').data([curve]).join('path').attr('fill', 'none').attr('stroke', cat(0)).attr('stroke-width', 2)
      .attr('d', d3.line().x((d) => C.x(d.x)).y((d) => C.y(d.y)));
    const marks = [{ a: deg(pcAngle), t: 'PC1' }, { a: deg(pcAngle + Math.PI / 2), t: 'PC2' }];
    C.layer('marks').selectAll('line').data(marks).join('line').attr('x1', (d) => C.x(d.a)).attr('x2', (d) => C.x(d.a)).attr('y1', 0).attr('y2', C.h)
      .attr('stroke', ink2()).attr('stroke-dasharray', '4 4').attr('stroke-width', 1);
    C.layer('marks').selectAll('text').data(marks).join('text').attr('x', (d) => C.x(d.a) + 4).attr('y', 14).attr('fill', ink2()).attr('font-size', 12).text((d) => d.t);
    const cur = { x: deg(theta), y: 100 * varAt(theta) / cov.total };
    C.layer('cur').selectAll('circle').data([cur]).join('circle').attr('cx', (d) => C.x(d.x)).attr('cy', (d) => C.y(d.y)).attr('r', 7)
      .attr('fill', cat(1)).attr('stroke', css('--surface')).attr('stroke-width', 2);
    readout.set({
      'axis angle': deg(theta).toFixed(0) + '°',
      'variance along axis': varAt(theta).toFixed(3),
      'share of total': (100 * varAt(theta) / cov.total).toFixed(1) + '%',
      'PC1 angle': deg(pcAngle).toFixed(0) + '°',
      'PC1 share (max)': (100 * cov.l1 / cov.total).toFixed(1) + '%',
    });
  }
  makeCloud(); paint();

  // ================= Figure 2: 50-gene expression =================
  const fig2 = figure(main, 'PCA on a synthetic 50-gene expression matrix',
    '220 simulated cells (110 T cells, 70 B cells, 40 monocytes) × 50 genes: 8 markers per cell type, 6 housekeeping genes and 20 pure-noise genes. Hover cells and bars for values. The loadings show <i>which genes</i> build each PC.');
  const E = makeExpression();
  let scale = false, px = 0, py = 1, pl = 0, res = null;
  const pcOpts = d3.range(6).map((k) => [String(k), 'PC' + (k + 1)]);
  select(fig2.controls, { label: 'x axis', options: pcOpts, value: '0', onChange: (v) => { px = +v; drawScatter(); } });
  select(fig2.controls, { label: 'y axis', options: pcOpts, value: '1', onChange: (v) => { py = +v; drawScatter(); } });
  select(fig2.controls, { label: 'Loadings of', options: pcOpts, value: '0', onChange: (v) => { pl = +v; drawLoadings(); } });
  checkbox(fig2.controls, { label: 'Scale each gene to unit variance', checked: false, onChange: (c) => { scale = c; recompute(); } });
  const readout2 = stats(fig2.root);
  legend(fig2.root, [[TYPES[0] + ' (and its markers)', cat(0)], [TYPES[1], cat(1)], [TYPES[2], cat(2)], ['housekeeping gene', cat(3)], ['noise gene', muted()]]);
  const row2 = el('div', { class: 'fig-row' }); fig2.body.append(row2);
  const sc = el('div'), sr = el('div'); row2.append(sc, sr);
  const lo = el('div'); fig2.body.append(lo);

  const geneColor = (g) => (geneGroup(g) < 4 ? cat(geneGroup(g)) : muted());
  function recompute() { res = pcaOf(E.X, E.n, G, scale); drawScatter(); drawScree(); drawLoadings(); }
  function drawScatter() {
    sc.innerHTML = '';
    const a = res.scores(px), b = res.scores(py);
    const ea = d3.extent(a), eb = d3.extent(b), pad = (e) => { const d = (e[1] - e[0]) * 0.08; return [e[0] - d, e[1] + d]; };
    const S = frame(sc, { width: 480, height: 400, x: pad(ea), y: pad(eb), xLabel: `PC${px + 1} (${(100 * res.ratio[px]).toFixed(1)}% of variance)`, yLabel: `PC${py + 1} (${(100 * res.ratio[py]).toFixed(1)}%)` });
    const data = E.cells.map((t, i) => ({ t, a: a[i], b: b[i], i }));
    S.layer('p').selectAll('circle').data(data).join('circle').attr('cx', (d) => S.x(d.a)).attr('cy', (d) => S.y(d.b)).attr('r', 4.2)
      .attr('fill', (d) => cat(d.t)).attr('stroke', css('--surface')).attr('stroke-width', 1).attr('opacity', 0.9)
      .on('mousemove', (e, d) => tip(e, `cell ${d.i + 1}: <b>${TYPES[d.t]}</b><br>PC${px + 1} ${d.a.toFixed(2)}, PC${py + 1} ${d.b.toFixed(2)}`)).on('mouseleave', untip);
  }
  function drawScree() {
    sr.innerHTML = '';
    const K = 12, data = d3.range(K).map((k) => ({ k, v: 100 * res.ratio[k] }));
    const S = frame(sr, { width: 440, height: 400, x: [0.4, K + 0.6], y: [0, Math.ceil(d3.max(data, (d) => d.v) / 10) * 10 + 5], xLabel: 'principal component', yLabel: '% of variance explained', xTicks: K });
    S.g.select('.axis.x').call(d3.axisBottom(S.x).tickValues(d3.range(1, K + 1)).tickFormat((d) => 'PC' + d));
    S.g.selectAll('.axis.x text').attr('font-size', 9);
    const bw = S.x(1.4) - S.x(1);
    S.layer('b').selectAll('rect').data(data).join('rect').attr('x', (d) => S.x(d.k + 1) - bw / 2).attr('width', bw).attr('y', (d) => S.y(d.v)).attr('height', (d) => S.h - S.y(d.v))
      .attr('rx', 2).attr('fill', (d) => (d.k < 2 ? cat(0) : muted()))
      .on('mousemove', (e, d) => tip(e, `PC${d.k + 1}: ${d.v.toFixed(1)}% of variance<br>cumulative ${(100 * d3.sum(res.ratio.slice(0, d.k + 1))).toFixed(1)}%`)).on('mouseleave', untip);
    S.layer('t').selectAll('text').data(data.slice(0, 4)).join('text').attr('x', (d) => S.x(d.k + 1)).attr('y', (d) => S.y(d.v) - 5).attr('text-anchor', 'middle')
      .attr('font-size', 11).attr('fill', ink2()).text((d) => d.v.toFixed(1) + '%');
    const elbow = d3.sum(res.ratio.slice(0, 2)) * 100;
    readout2.set({ 'PC1': (100 * res.ratio[0]).toFixed(1) + '%', 'PC2': (100 * res.ratio[1]).toFixed(1) + '%', 'PC3': (100 * res.ratio[2]).toFixed(1) + '%', 'PC1+PC2': elbow.toFixed(1) + '%', 'genes in': G });
  }
  function drawLoadings() {
    lo.innerHTML = '';
    const v = res.vecs[pl], m = d3.max(v, Math.abs) * 1.1;
    const S = frame(lo, { width: 940, height: 250, x: [0, G], y: [-m, m], yLabel: `PC${pl + 1} loading`, margin: { bottom: 62 }, axes: false });
    S.g.append('g').attr('class', 'axis y').call(d3.axisLeft(S.y).ticks(4));
    S.g.append('line').attr('x1', 0).attr('x2', S.w).attr('y1', S.y(0)).attr('y2', S.y(0)).attr('stroke', css('--axis'));
    const bw = S.w / G;
    S.layer('b').selectAll('rect').data(Array.from(v)).join('rect').attr('x', (d, g) => g * bw + 1).attr('width', bw - 2)
      .attr('y', (d) => S.y(Math.max(0, d))).attr('height', (d) => Math.abs(S.y(d) - S.y(0))).attr('fill', (d, g) => geneColor(g))
      .on('mousemove', (e, d) => { const g = v.indexOf(d); tip(e, `<b>${GENES[g]}</b>: loading ${d.toFixed(3)}`); }).on('mouseleave', untip);
    S.g.selectAll('text.gn').data(GENES).join('text').attr('class', 'gn').attr('font-size', 9).attr('fill', ink2()).attr('text-anchor', 'end')
      .attr('transform', (d, g) => `translate(${g * bw + bw / 2 + 3},${S.h + 8}) rotate(-60)`).text((d) => d);
  }
  recompute();

  section(main, 'What to notice', `<ul>
    <li>In the first figure, drag the axis round. The captured share goes up and down like a wave, from about 91% down to about 9%. <b>PC1 is the top of that wave</b>: with this seed it sits at about 35°, the direction the cloud was stretched along. Press <i>Snap to PC1</i>.</li>
    <li>The grey ticks (the residuals) are shortest overall exactly when the variance captured is largest. Maximising projected variance and minimising squared reconstruction error are the same problem.</li>
    <li>Push <i>Spread across the main axis</i> to 0.55, so the cloud is round. The curve flattens to between about 47% and 53%, and PC1 now points wherever this particular sample of 160 points happens to be slightly longer. That direction is sampling noise, not structure. PCA has nothing to find when no direction is special.</li>
    <li>In the gene example, PC1 (42% of the variance) puts T cells at one end and B cells at the other, and its loadings say why: T-cell markers (blue) load positive, B-cell markers (orange) load negative. PC2 (24%) lifts the monocytes away from both; switch <i>Loadings of</i> to PC2 to see the monocyte markers (green) take over. Three cell types need two PCs, and after PC2 the scree plot drops to a flat floor of noise.</li>
    <li>Tick <i>Scale each gene</i>. Each noise gene now counts as much as a marker gene, so the first two PCs explain a smaller share and the noise floor rises. The cell types still separate, but scaling amplifies uninformative genes, which is one reason single-cell pipelines first pick highly variable genes.</li>
    <li>Housekeeping genes (yellow) have near-zero loadings: they are high but do not <i>vary</i>, and PCA only sees variance.</li>
  </ul>`);

  steps(main, 'How it works', [
    '<b>Centre</b> each feature (gene) by subtracting its mean. Optionally <b>scale</b> it to unit variance so that all genes count equally.',
    'Compute the <b>covariance matrix</b>: how every pair of genes varies together (50×50 here).',
    'Find its <b>eigenvectors</b>. The first eigenvector is the direction of maximum variance (PC1), the next is the maximum-variance direction at right angles to it (PC2), and so on. Each eigenvalue is the variance along its PC.',
    'The eigenvector entries are the <b>loadings</b>: how much each gene contributes to that PC.',
    '<b>Project</b> each cell onto the first k PCs. The resulting <b>scores</b> are the new coordinates, which you can plot or pass on to clustering, kNN graphs or UMAP.',
    'Choose k from the <b>scree plot</b>: keep PCs until the curve flattens into the noise floor.',
  ]);
  math(main, 'Σ = (1/(n−1)) Xᶜᵀ Xᶜ          (Xᶜ = centred data, n cells × p genes)\nΣ v_k = λ_k v_k,   λ₁ ≥ λ₂ ≥ …          PC scores = Xᶜ v_k\nvariance captured by an axis u (|u|=1) = uᵀ Σ u,  maximised by u = v₁;  share = λ_k / Σ λ');

  callout(main, '', 'In biology', `PCA is step one of nearly every single-cell workflow. <code>sc.pp.highly_variable_genes</code> → <code>sc.pp.scale</code> → <code>sc.tl.pca(n_comps=50)</code> in scanpy, or <code>FindVariableFeatures</code> (2,000 HVGs) → <code>ScaleData</code> → <code>RunPCA</code> in Seurat. The neighbour graph, Leiden clusters and UMAP are then all computed on the top 20–50 PCs, not on raw genes. <code>sc.pl.pca_variance_ratio</code> and Seurat's <code>ElbowPlot</code> are the scree plot above. In bulk RNA-seq, a PCA of the top 500 variable genes after a VST (DESeq2 <code>plotPCA</code>) is the standard first look for batch effects and outlier samples. Loadings of a PC often read as a programme, such as cell cycle, interferon response or a cell-type signature.`);
  callout(main, 'warn', 'Pitfalls', `<ul>
    <li><b>Variance is not importance.</b> PC1 is often technical: library size, batch, percent mitochondrial reads. Check what correlates with each PC before interpreting it.</li>
    <li>PCA is <b>linear</b>. Curved structure, such as a differentiation trajectory, gets spread over several PCs.</li>
    <li>Scale matters: unscaled, the highest-variance genes dominate; scaled, noise genes get a vote. Choosing HVGs first is what makes scaling safe.</li>
    <li>The sign of a PC is arbitrary. Two runs can flip PC1, so compare loadings, not orientations.</li>
    <li>Eigenvalues from noise are not zero. With few cells and many genes the noise floor is high, so do not over-read PC10 vs PC11.</li>
  </ul>`);
  seeAlso(main, [['concepts', 'tsne-umap', 't-SNE & UMAP'], ['concepts', 'graph-clustering', 'kNN graphs & Leiden'], ['concepts', 'scaling', 'Feature scaling'], ['concepts', 'curse-dim', 'Curse of dimensionality'], ['omics', 'sc-embedding', 'Normalise, HVGs, PCA']]);
}
