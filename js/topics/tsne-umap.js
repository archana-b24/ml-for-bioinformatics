// t-SNE (exact, animated, in JS) and a miniature UMAP on the same synthetic 20-D data, compared against PCA.
import { header, figure, slider, select, button, checkbox, stats, section, steps, callout, math, legend, seeAlso, animate, tip, untip, el } from '../lib/ui.js';
import { frame, cat, css, ink, ink2, muted } from '../lib/plot.js';
import { rng } from '../lib/data.js';
const d3 = window.d3;

// ---------- synthetic 20-D data with known geometry ----------
const DIM = 20;
const GROUPS = [
  { name: 'tight cluster A', n: 70 },
  { name: 'wide cluster B', n: 70 },
  { name: 'distant cluster C', n: 60 },
  { name: 'trajectory', n: 100 },
];
function makeData() {
  const r = rng(7), rows = [], lab = [], tt = [];
  const add = (vec, g, t = 0) => { rows.push(vec); lab.push(g); tt.push(t); };
  const noise = (sd) => Float64Array.from({ length: DIM }, () => r.normal(0, sd));
  for (let i = 0; i < 70; i++) add(noise(0.5), 0); // A at the origin, sd 0.5
  for (let i = 0; i < 70; i++) { const v = noise(2.0); v[0] += 16; add(v, 1); } // B: 16 from A, 4x wider
  for (let i = 0; i < 60; i++) { const v = noise(0.8); v[1] += 40; add(v, 2); } // C: 40 from A
  for (let i = 0; i < 100; i++) { // a curved 1-D path, like a differentiation trajectory
    const t = i / 99, v = noise(0.4);
    v[2] += 20; v[3] += 24 * t; v[4] += 6 * Math.sin(Math.PI * t);
    add(v, 3, t);
  }
  const n = rows.length, X = new Float64Array(n * DIM);
  rows.forEach((v, i) => X.set(v, i * DIM));
  const D2 = new Float64Array(n * n);
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
    let s = 0; for (let d = 0; d < DIM; d++) { const q = X[i * DIM + d] - X[j * DIM + d]; s += q * q; }
    D2[i * n + j] = D2[j * n + i] = s;
  }
  return { X, D2, n, lab, tt };
}

// top-2 PCA via power iteration with deflation (20x20 covariance)
function pca2(X, n, p) {
  const mu = new Float64Array(p);
  for (let i = 0; i < n; i++) for (let d = 0; d < p; d++) mu[d] += X[i * p + d] / n;
  const C = new Float64Array(p * p);
  for (let i = 0; i < n; i++) for (let a = 0; a < p; a++) { const xa = X[i * p + a] - mu[a]; for (let b = 0; b < p; b++) C[a * p + b] += xa * (X[i * p + b] - mu[b]) / (n - 1); }
  const vecs = [];
  for (let k = 0; k < 2; k++) {
    let v = Float64Array.from({ length: p }, (_, i) => 1 / Math.sqrt(p) + i * 1e-3), lam = 0;
    for (let it = 0; it < 500; it++) {
      const w = new Float64Array(p);
      for (let a = 0; a < p; a++) for (let b = 0; b < p; b++) w[a] += C[a * p + b] * v[b];
      lam = Math.hypot(...w); v = w.map((x) => x / lam);
    }
    vecs.push(v);
    for (let a = 0; a < p; a++) for (let b = 0; b < p; b++) C[a * p + b] -= lam * v[a] * v[b];
  }
  const Y = new Float64Array(n * 2);
  for (let i = 0; i < n; i++) for (let k = 0; k < 2; k++) { let s = 0; for (let d = 0; d < p; d++) s += (X[i * p + d] - mu[d]) * vecs[k][d]; Y[i * 2 + k] = s; }
  return Y;
}

// ---------- exact t-SNE ----------
function affinities(D2, n, perp) {
  const P = new Float64Array(n * n), row = new Float64Array(n), target = Math.log(perp);
  for (let i = 0; i < n; i++) {
    let mn = Infinity; for (let j = 0; j < n; j++) if (j !== i && D2[i * n + j] < mn) mn = D2[i * n + j];
    let beta = 1, lo = -Infinity, hi = Infinity;
    for (let it = 0; it < 60; it++) {
      let sum = 0, sdp = 0;
      for (let j = 0; j < n; j++) { if (j === i) { row[j] = 0; continue; } const d = D2[i * n + j] - mn, p = Math.exp(-d * beta); row[j] = p; sum += p; sdp += d * p; }
      const H = Math.log(sum) + beta * sdp / sum; // entropy of the conditional distribution (nats)
      if (Math.abs(H - target) < 1e-5) break;
      if (H > target) { lo = beta; beta = hi === Infinity ? beta * 2 : (beta + hi) / 2; } else { hi = beta; beta = lo === -Infinity ? beta / 2 : (beta + lo) / 2; }
    }
    let sum = 0; for (let j = 0; j < n; j++) sum += row[j];
    for (let j = 0; j < n; j++) P[i * n + j] = row[j] / sum;
  }
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) { const v = Math.max((P[i * n + j] + P[j * n + i]) / (2 * n), 1e-12); P[i * n + j] = P[j * n + i] = v; }
  return P;
}
function makeTsne(P, n, seed) {
  const r = rng(seed), Y = new Float64Array(n * 2), iY = new Float64Array(n * 2), gains = new Float64Array(n * 2).fill(1);
  const num = new Float64Array(n * n), grad = new Float64Array(n * 2);
  for (let i = 0; i < 2 * n; i++) Y[i] = r.normal(0, 1e-4);
  const EXAG_IT = 150, LR = 50;
  let iter = 0;
  function step() {
    const exag = iter < EXAG_IT ? 12 : 1, mom = iter < 250 ? 0.5 : 0.8;
    let Z = 0;
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
      const dx = Y[2 * i] - Y[2 * j], dy = Y[2 * i + 1] - Y[2 * j + 1], q = 1 / (1 + dx * dx + dy * dy);
      num[i * n + j] = q; Z += 2 * q;
    }
    grad.fill(0);
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
      const q = num[i * n + j], m = (exag * P[i * n + j] - q / Z) * q;
      const dx = Y[2 * i] - Y[2 * j], dy = Y[2 * i + 1] - Y[2 * j + 1];
      grad[2 * i] += 4 * m * dx; grad[2 * i + 1] += 4 * m * dy; grad[2 * j] -= 4 * m * dx; grad[2 * j + 1] -= 4 * m * dy;
    }
    let mx = 0, my = 0;
    for (let k = 0; k < 2 * n; k++) {
      gains[k] = Math.sign(grad[k]) === Math.sign(iY[k]) ? Math.max(0.01, gains[k] * 0.8) : gains[k] + 0.2;
      iY[k] = mom * iY[k] - LR * gains[k] * grad[k];
      Y[k] += iY[k];
    }
    for (let i = 0; i < n; i++) { mx += Y[2 * i]; my += Y[2 * i + 1]; }
    for (let i = 0; i < n; i++) { Y[2 * i] -= mx / n; Y[2 * i + 1] -= my / n; }
    iter++;
    return Z;
  }
  function kl() {
    let Z = 0;
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) { const dx = Y[2 * i] - Y[2 * j], dy = Y[2 * i + 1] - Y[2 * j + 1]; Z += 2 / (1 + dx * dx + dy * dy); }
    let s = 0;
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
      const dx = Y[2 * i] - Y[2 * j], dy = Y[2 * i + 1] - Y[2 * j + 1], q = 1 / (1 + dx * dx + dy * dy) / Z, p = P[i * n + j];
      s += 2 * p * Math.log(p / Math.max(q, 1e-300));
    }
    return s;
  }
  return { Y, step, kl, get iter() { return iter; }, EXAG_IT };
}

// ---------- miniature UMAP ----------
function fitAB(minDist, spread = 1) { // least-squares fit of 1/(1+a d^2b) to the min_dist-shaped target
  const xs = d3.range(1, 300).map((i) => i / 100), ys = xs.map((x) => (x < minDist ? 1 : Math.exp(-(x - minDist) / spread)));
  let best = [1, 1], bestE = Infinity;
  const search = (aLo, aHi, bLo, bHi, steps) => {
    for (let ia = 0; ia <= steps; ia++) for (let ib = 0; ib <= steps; ib++) {
      const a = Math.exp(aLo + (aHi - aLo) * ia / steps), b = bLo + (bHi - bLo) * ib / steps;
      let e = 0; for (let k = 0; k < xs.length; k++) { const f = 1 / (1 + a * Math.pow(xs[k], 2 * b)) - ys[k]; e += f * f; }
      if (e < bestE) { bestE = e; best = [a, b]; }
    }
  };
  search(Math.log(0.02), Math.log(20), 0.3, 2.5, 60);
  const [a0, b0] = best; search(Math.log(a0) - 0.15, Math.log(a0) + 0.15, b0 - 0.05, b0 + 0.05, 30);
  return { a: best[0], b: best[1] };
}
function fuzzyGraph(D2, n, k, order) {
  const W = new Map();
  for (let i = 0; i < n; i++) {
    const nb = order[i].slice(0, k), d = nb.map((j) => Math.sqrt(D2[i * n + j])), rho = d[0], target = Math.log2(k);
    let lo = 0, hi = Infinity, sig = 1;
    for (let it = 0; it < 64; it++) {
      let s = 0; for (let q = 0; q < k; q++) s += Math.exp(-Math.max(0, d[q] - rho) / sig);
      if (Math.abs(s - target) < 1e-5) break;
      if (s > target) { hi = sig; sig = (lo + hi) / 2; } else { lo = sig; sig = hi === Infinity ? sig * 2 : (lo + hi) / 2; }
    }
    nb.forEach((j, q) => {
      const w = Math.exp(-Math.max(0, d[q] - rho) / sig), key = i < j ? i * n + j : j * n + i, prev = W.get(key);
      W.set(key, prev == null ? w : prev + w - prev * w); // fuzzy union: a + b − ab
    });
  }
  const m = W.size, ei = new Int32Array(m), ej = new Int32Array(m), ew = new Float64Array(m);
  let q = 0; for (const [key, w] of W) { ei[q] = Math.floor(key / n); ej[q] = key % n; ew[q] = w; q++; }
  return { ei, ej, ew, m };
}
function makeUmap(G, n, init, ab, seed, nEpochs = 300) {
  const r = rng(seed), Y = Float64Array.from(init), { a, b } = ab;
  let epoch = 0;
  const clip = (v) => (v > 4 ? 4 : v < -4 ? -4 : v);
  function step() {
    const alpha = 1 - epoch / nEpochs;
    for (let e = 0; e < G.m; e++) {
      if (r() > G.ew[e]) continue; // sample each edge with probability = its membership strength
      const i = G.ei[e], j = G.ej[e];
      let dx = Y[2 * i] - Y[2 * j], dy = Y[2 * i + 1] - Y[2 * j + 1], d2 = dx * dx + dy * dy;
      if (d2 > 0) {
        const c = (-2 * a * b * Math.pow(d2, b - 1)) / (1 + a * Math.pow(d2, b));
        const gx = clip(c * dx), gy = clip(c * dy);
        Y[2 * i] += alpha * gx; Y[2 * i + 1] += alpha * gy; Y[2 * j] -= alpha * gx; Y[2 * j + 1] -= alpha * gy;
      }
      for (let s = 0; s < 5; s++) { // negative samples: random points are pushed away
        const k = r.int(n); if (k === i) continue;
        dx = Y[2 * i] - Y[2 * k]; dy = Y[2 * i + 1] - Y[2 * k + 1]; d2 = dx * dx + dy * dy;
        const c = (2 * b) / ((0.001 + d2) * (1 + a * Math.pow(d2, b)));
        Y[2 * i] += alpha * (c > 0 ? clip(c * dx) : 4); Y[2 * i + 1] += alpha * (c > 0 ? clip(c * dy) : 4);
      }
    }
    epoch++;
  }
  return { Y, step, get epoch() { return epoch; }, nEpochs };
}

// ---------- geometry summaries (scale-free) ----------
function summary(Y, dim, lab) {
  const cent = [0, 1, 2].map((g) => { const c = new Float64Array(dim); let m = 0; lab.forEach((l, i) => { if (l === g) { m++; for (let d = 0; d < dim; d++) c[d] += Y[i * dim + d]; } }); return c.map((v) => v / m); });
  const dist = (i, c) => { let s = 0; for (let d = 0; d < dim; d++) s += (Y[i * dim + d] - c[d]) ** 2; return Math.sqrt(s); };
  const spread = (g) => d3.mean(lab.map((l, i) => (l === g ? dist(i, cent[g]) : null)).filter((v) => v != null));
  const cd = (g, h) => Math.sqrt(d3.sum(d3.range(dim), (d) => (cent[g][d] - cent[h][d]) ** 2));
  return { spread: spread(1) / spread(0), far: cd(0, 2) / cd(0, 1) };
}

export default async function render(main) {
  header(main, 't-SNE & UMAP',
    't-SNE and UMAP squash high-dimensional data into a 2D picture by <b>keeping each point\'s nearest neighbours near</b>. They are excellent at showing which points belong together, and unreliable about <b>how big</b> groups are or <b>how far apart</b> they sit.');

  const DATA = makeData(), { n, lab, tt } = DATA;
  const PC = pca2(DATA.X, n, DIM);
  const order = d3.range(n).map((i) => d3.range(n).filter((j) => j !== i).sort((a, b) => DATA.D2[i * n + a] - DATA.D2[i * n + b]));
  const truth = summary(DATA.X, DIM, lab), pcaSum = summary(PC, 2, lab);
  const color = (i) => (lab[i] === 3 ? d3.interpolateRgb(css('--surface-2'), cat(3))(0.3 + 0.7 * tt[i]) : cat(lab[i]));
  const tipFor = (i) => `<b>${GROUPS[lab[i]].name}</b>${lab[i] === 3 ? `<br>position along path t = ${tt[i].toFixed(2)}` : ''}`;
  const legendEntries = [[GROUPS[0].name + ' (sd 0.5)', cat(0)], [GROUPS[1].name + ' (sd 2.0)', cat(1)], [GROUPS[2].name, cat(2)], ['trajectory (light → dark = start → end)', cat(3)]];

  /** An embedding panel with no axes (the coordinates have no units). */
  function panel(parent, title) {
    const Fr = frame(parent, { width: 470, height: 440, x: [-1, 1], y: [-1, 1], axes: false, grid: false, margin: { top: 26, right: 10, bottom: 10, left: 10 } });
    Fr.svg.append('rect').attr('x', 10).attr('y', 26).attr('width', Fr.w).attr('height', Fr.h).attr('fill', 'none').attr('stroke', css('--grid'));
    Fr.svg.append('text').attr('x', 14).attr('y', 17).attr('font-size', 13).attr('font-weight', 600).attr('fill', ink()).text(title);
    const data = d3.range(n);
    const sel = Fr.layer('p').selectAll('circle').data(data).join('circle').attr('r', 4).attr('fill', color).attr('stroke', css('--surface')).attr('stroke-width', 1)
      .on('mousemove', (e, i) => tip(e, tipFor(i))).on('mouseleave', untip);
    const draw = (Y) => {
      let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
      for (let i = 0; i < n; i++) { x0 = Math.min(x0, Y[2 * i]); x1 = Math.max(x1, Y[2 * i]); y0 = Math.min(y0, Y[2 * i + 1]); y1 = Math.max(y1, Y[2 * i + 1]); }
      const half = Math.max(x1 - x0, (y1 - y0) * Fr.w / Fr.h, 1e-9) / 2 * 1.08, cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
      Fr.x.domain([cx - half, cx + half]); Fr.y.domain([cy - half * Fr.h / Fr.w, cy + half * Fr.h / Fr.w]);
      sel.attr('cx', (i) => Fr.x(Y[2 * i])).attr('cy', (i) => Fr.y(Y[2 * i + 1]));
    };
    return { Fr, draw, sel };
  }

  // ================= Figure 1: t-SNE vs PCA =================
  const fig = figure(main, 't-SNE, iteration by iteration, next to PCA of the same data',
    `300 points in 20 dimensions: a tight cluster A, a cluster B that is <b>4× wider</b> and sits 16 units from A, a cluster C 40 units from A, and a curved 100-point trajectory. t-SNE starts from a tiny random blob. The first 150 iterations use <b>early exaggeration</b> (attractions ×12) so clusters form, and then it refines. PCA (right) is a fixed linear projection. Hover any point.`);
  let perp = 30, seed = 1, ts = null, playing = true, P = null, lastKL = null;
  slider(fig.controls, { label: 'Perplexity', min: 2, max: 100, step: 1, value: perp, onInput: (v) => { perp = v; restart(true); } });
  select(fig.controls, { label: 'Random start', options: [['1', 'seed 1'], ['2', 'seed 2'], ['3', 'seed 3']], value: '1', onChange: (v) => { seed = +v; restart(false); } });
  const playBtn = button(fig.controls, 'Pause', () => { playing = !playing; playBtn.textContent = playing ? 'Pause' : 'Play'; }, true);
  button(fig.controls, 'Restart', () => restart(false));
  const readout = stats(fig.root);
  legend(fig.root, legendEntries);
  const row = el('div', { class: 'fig-row' }); fig.body.append(row);
  const lp = el('div'), rp = el('div'); row.append(lp, rp);
  const T = panel(lp, 't-SNE'), Pp = panel(rp, 'PCA (PC1 vs PC2)');
  Pp.draw(PC);
  const table = el('table', { class: 'simple', style: 'margin-top:10px' }); fig.body.append(table);

  function restart(newP) {
    if (newP || !P) P = affinities(DATA.D2, n, perp);
    ts = makeTsne(P, n, seed); lastKL = null; T.draw(ts.Y); paintStats();
    if (!playing) { playing = true; playBtn.textContent = 'Pause'; }
  }
  const MAX_IT = 1000;
  function paintStats() {
    const s = summary(ts.Y, 2, lab);
    readout.set({ iteration: ts.iter, phase: ts.iter < ts.EXAG_IT ? 'early exaggeration' : ts.iter < MAX_IT ? 'refining' : 'done', 'KL divergence': lastKL == null ? '…' : lastKL.toFixed(3), perplexity: perp });
    table.innerHTML = `<tr><th></th><th>original 20-D</th><th>PCA</th><th>t-SNE (live)</th></tr>
      <tr><td>B's spread ÷ A's spread</td><td>${truth.spread.toFixed(1)}</td><td>${pcaSum.spread.toFixed(1)}</td><td><b>${s.spread.toFixed(1)}</b></td></tr>
      <tr><td>distance A→C ÷ distance A→B</td><td>${truth.far.toFixed(1)}</td><td>${pcaSum.far.toFixed(1)}</td><td><b>${s.far.toFixed(1)}</b></td></tr>`;
  }
  restart(true);
  animate(() => {
    if (!playing || ts.iter >= MAX_IT) return true;
    for (let k = 0; k < 4 && ts.iter < MAX_IT; k++) ts.step(); // 4 iterations per frame (~1 ms each): slow enough to watch
    if (ts.iter % 50 < 8 || ts.iter >= MAX_IT) lastKL = ts.kl();
    T.draw(ts.Y); paintStats();
    if (ts.iter >= MAX_IT) { playBtn.textContent = 'Play'; playing = false; }
    return true;
  });

  // ================= Figure 2: mini UMAP =================
  const fig2 = figure(main, 'UMAP in miniature: a fuzzy neighbour graph, then a layout',
    'The same 300 points. Each point connects to its <i>n_neighbors</i> nearest neighbours in 20-D, with a weight that falls off with distance (darker edges = stronger). The layout starts from PCA, and stochastic gradient descent then pulls connected points together and pushes randomly sampled points apart. The right panel shows the similarity curve the layout aims for, which <i>min_dist</i> controls, beside t-SNE\'s curve.');
  let kU = 15, minDist = 0.1, showEdges = true, um = null, G = null, ab = fitAB(0.1);
  slider(fig2.controls, { label: 'n_neighbors', min: 3, max: 50, step: 1, value: kU, onInput: (v) => { kU = v; restartU(true); } });
  slider(fig2.controls, { label: 'min_dist', min: 0, max: 0.9, step: 0.05, value: minDist, fmt: (v) => v.toFixed(2), onInput: (v) => { minDist = v; ab = fitAB(v); drawCurve(); restartU(false); } });
  checkbox(fig2.controls, { label: 'Show graph edges', checked: true, onChange: (c) => { showEdges = c; drawU(); } });
  button(fig2.controls, 'Re-run layout', () => restartU(false), true);
  const readout2 = stats(fig2.root);
  const row2 = el('div', { class: 'fig-row' }); fig2.body.append(row2);
  const ul = el('div'), ur = el('div'); row2.append(ul, ur);
  const U = panel(ul, 'mini-UMAP');
  const edgeLayer = U.Fr.g.insert('g', ':first-child');
  const CF = frame(ur, { width: 440, height: 440, x: [0, 3], y: [0, 1.05], xLabel: 'distance between two points in the 2-D layout', yLabel: 'target similarity' });
  legend(ur, [['UMAP curve (fitted to min_dist)', cat(0)], ['t-SNE Student-t: 1/(1+d²)', cat(1)]]);

  function drawCurve() {
    const xs = d3.range(0, 3.001, 0.02);
    const ser = [
      { c: cat(1), dash: '6 4', pts: xs.map((x) => ({ x, y: 1 / (1 + x * x) })) },
      { c: cat(0), dash: null, pts: xs.map((x) => ({ x, y: 1 / (1 + ab.a * Math.pow(x, 2 * ab.b)) })) },
    ];
    CF.layer('c').selectAll('path').data(ser).join('path').attr('fill', 'none').attr('stroke', (d) => d.c).attr('stroke-width', 2)
      .attr('stroke-dasharray', (d) => d.dash).attr('d', (d) => d3.line().x((p) => CF.x(p.x)).y((p) => CF.y(p.y))(d.pts));
    CF.layer('m').selectAll('line').data([minDist]).join('line').attr('x1', (d) => CF.x(d)).attr('x2', (d) => CF.x(d)).attr('y1', 0).attr('y2', CF.h)
      .attr('stroke', ink2()).attr('stroke-dasharray', '3 4');
    CF.layer('m').selectAll('text').data([minDist]).join('text').attr('x', (d) => CF.x(d) + 5).attr('y', CF.h - 8).attr('font-size', 12).attr('fill', ink2()).text('min_dist');
  }
  const initU = (() => { // UMAP rescales its (spectral) initialisation to a 0–10 box; we use PCA here
    const Y = new Float64Array(2 * n); let m = 0;
    for (let k = 0; k < 2 * n; k++) m = Math.max(m, Math.abs(PC[k]));
    for (let k = 0; k < 2 * n; k++) Y[k] = PC[k] / m * 10;
    return Y;
  })();
  function restartU(newGraph) { if (newGraph || !G) G = fuzzyGraph(DATA.D2, n, kU, order); um = makeUmap(G, n, initU, ab, 3); drawU(); }
  function drawU() {
    U.draw(um.Y);
    const buckets = [[0.66, 1.01, 0.45], [0.33, 0.66, 0.25], [0, 0.33, 0.1]];
    const paths = showEdges ? buckets.map(([lo, hi, op]) => {
      let d = '';
      for (let e = 0; e < G.m; e++) if (G.ew[e] >= lo && G.ew[e] < hi) d += `M${U.Fr.x(um.Y[2 * G.ei[e]]).toFixed(1)},${U.Fr.y(um.Y[2 * G.ei[e] + 1]).toFixed(1)}L${U.Fr.x(um.Y[2 * G.ej[e]]).toFixed(1)},${U.Fr.y(um.Y[2 * G.ej[e] + 1]).toFixed(1)}`;
      return { d, op };
    }) : [];
    edgeLayer.selectAll('path').data(paths).join('path').attr('d', (p) => p.d).attr('fill', 'none').attr('stroke', ink2()).attr('stroke-opacity', (p) => p.op).attr('stroke-width', 0.8);
    const s = summary(um.Y, 2, lab);
    readout2.set({ epoch: `${um.epoch} / ${um.nEpochs}`, 'graph edges': G.m, 'B spread ÷ A spread': s.spread.toFixed(1), 'A→C ÷ A→B': s.far.toFixed(1), 'curve a, b': `${ab.a.toFixed(2)}, ${ab.b.toFixed(2)}` });
  }
  drawCurve(); restartU(true);
  animate(() => {
    if (um.epoch >= um.nEpochs) return true;
    const t0 = performance.now();
    while (um.epoch < um.nEpochs && performance.now() - t0 < 10) um.step();
    drawU(); return true;
  });

  section(main, 'What to notice', `<ul>
    <li><b>Sizes are equalised.</b> In 20-D, cluster B is 4× wider than A (first row of the table). PCA keeps that (3.9). t-SNE ends with a ratio of 1.0, and the mini-UMAP also gives about 1, because every point gets roughly the same number of effective neighbours whatever the local density. A big blob in a t-SNE or UMAP plot is not a more heterogeneous population.</li>
    <li><b>Between-cluster distances are not to scale.</b> C is 2.5× further from A than B is. t-SNE places the three clusters roughly equidistant (ratio about 1.0 for seeds 1, 2 and 3), and the arrangement itself changes with the seed. The mini-UMAP gives 1.7. Neither number means anything biologically.</li>
    <li><b>PCA is not innocent either.</b> It keeps B's width, but it draws A <i>inside</i> B and reports C as 5.4× further away than B. The direction that separates A from B is mostly not among the top two PCs, and projecting can only shrink distances, so some shrink far more than others.</li>
    <li><b>The trajectory survives</b> as a continuous light-to-dark path, because consecutive points are each other's nearest neighbours. Set perplexity to 5 and it breaks into separate segments that could be mistaken for distinct "clusters".</li>
    <li><b>Perplexity</b> sets how many neighbours each point listens to. Low values make clusters ragged and fragment continuous structure, and high values smooth things out. There is no single correct value, so check that a conclusion holds across several.</li>
    <li>In the mini-UMAP, raise <i>min_dist</i> to 0.8 and the clusters puff up. The UMAP curve (right) now stays near 1 out to that distance, so neighbours stop being pulled closer once they are about min_dist apart. With n_neighbors = 3 only very local structure is kept, and the trajectory turns into a wiggly chain.</li>
  </ul>`);

  steps(main, 'How t-SNE works', [
    'For each point i, turn distances to every other point into probabilities <code>p(j|i)</code> with a Gaussian. Its width σᵢ is tuned per point so that the distribution has a fixed <b>perplexity</b>, roughly the number of neighbours that matter. Dense regions get small σ and sparse ones large σ, and that is exactly why sizes get equalised.',
    'Symmetrise: <code>p_ij = (p(j|i)+p(i|j)) / 2n</code>.',
    'Place points randomly in 2-D and define low-dimensional similarities with a heavy-tailed <b>Student-t</b> kernel: <code>q_ij ∝ 1/(1+|yᵢ−yⱼ|²)</code>. The heavy tail lets moderately distant points sit far apart, which opens up gaps between clusters.',
    'Move the points by gradient descent to minimise the <b>KL divergence</b> between P and Q. Neighbours in P attract, and every pair repels a little.',
    'Early exaggeration (P×12 for the first iterations) helps clusters form before the fine detail settles.',
  ]);
  steps(main, 'How UMAP differs', [
    'Build a <b>kNN graph</b> with <code>n_neighbors</code> neighbours per point. Each edge gets a fuzzy membership <code>exp(−(d − ρᵢ)/σᵢ)</code>, where ρᵢ is the distance to the nearest neighbour, so every point is connected to at least one other with weight 1.',
    'Combine the two directions of each edge as a fuzzy union, <code>w = a + b − ab</code>. This is the same kind of graph that scanpy\'s <code>pp.neighbors</code> builds and that Leiden clusters.',
    'Lay out the graph with stochastic gradient descent: sampled edges attract and random <b>negative samples</b> repel. Working only with edges, never all pairs, is why UMAP scales to millions of cells.',
    '<code>min_dist</code> sets the shape of the low-dimensional similarity curve, and with it how tightly points may pack.',
    'Real UMAP starts from a spectral embedding of the graph. Here we start from PCA, which plays the same role of providing a sensible global arrangement.',
  ]);
  math(main, 't-SNE:  minimise KL(P‖Q) = Σ p_ij log(p_ij / q_ij),   q_ij = (1+|yᵢ−yⱼ|²)⁻¹ / Σ_kl (1+|y_k−y_l|²)⁻¹\n        ∂KL/∂yᵢ = 4 Σⱼ (p_ij − q_ij)(yᵢ−yⱼ)(1+|yᵢ−yⱼ|²)⁻¹\nUMAP:   minimise Σ_edges [ w log(w/v) + (1−w) log((1−w)/(1−v)) ],   v_ij = 1/(1 + a|yᵢ−yⱼ|^(2b))');

  callout(main, '', 'In biology', `Every single-cell paper has a UMAP. In scanpy it is <code>sc.pp.neighbors(adata, n_neighbors=15, n_pcs=50)</code> → <code>sc.tl.umap(adata, min_dist=0.5)</code>, and Seurat's <code>RunUMAP(dims = 1:30)</code> does the same. Both run on the PCA, not on raw genes, and the embedding is only for <i>looking</i>: clustering, trajectory inference (PAGA, diffusion pseudotime) and differential expression all use the kNN graph or the PCA, never the 2-D coordinates. t-SNE (<code>sc.tl.tsne</code>, <code>RunTSNE</code>) was the standard before 2018 and is still used in flow and mass cytometry (viSNE, openTSNE). UMAP also embeds spatial-omics cells and spot-level expression, and colouring a UMAP by a gene (<code>sc.pl.umap(adata, color="CD8A")</code>) is how most people first look at markers.`);
  callout(main, 'warn', 'Pitfalls', `<ul>
    <li><b>Do not read cluster size, density or between-cluster distance</b> off a t-SNE or UMAP plot. A "big" cluster may just be one with many cells, and "closer" clusters may not be more similar.</li>
    <li>A gap is weak evidence of separate populations, because continuous trajectories can be cut at low perplexity or low n_neighbors. A blob without gaps can still hide distinct types.</li>
    <li>Different seeds give different pictures. Fix and report the seed, and do not over-interpret one run.</li>
    <li>Never cluster on UMAP coordinates. Cluster on the graph (Leiden) or the PCA.</li>
    <li>Chari &amp; Pachter (2023, <i>PLOS Comput Biol</i>, "The specious art of single-cell genomics") show how far these embeddings can distort the real distances. Use them as maps of neighbourhoods, not of geometry.</li>
  </ul>`);
  seeAlso(main, [['concepts', 'pca', 'PCA'], ['concepts', 'graph-clustering', 'kNN graphs & Leiden'], ['concepts', 'curse-dim', 'Curse of dimensionality'], ['omics', 'sc-clustering', 'Graph clustering & UMAP']]);
}
