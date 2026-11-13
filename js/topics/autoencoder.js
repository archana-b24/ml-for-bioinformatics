// Autoencoders: a tiny autoencoder trained live in the browser.
// Figure 1: 2-D shapes squeezed through a 1-D bottleneck (reconstructions, decoder curve, latent, PCA line).
// Figure 2: 24-gene "trajectory" profiles, bottleneck-size slider, autoencoder vs PCA reconstruction error.
import { header, figure, slider, select, button, stats, section, steps, callout, math, legend, seeAlso, animate, tip, untip, el } from '../lib/ui.js';
import { frame, cat, css, ink, ink2, muted } from '../lib/plot.js';
import { rng } from '../lib/data.js';
const d3 = window.d3;

/** Tiny fully connected net with per-sample backprop and Adam. acts[l] is 'tanh' or 'linear' for the output of layer l.
 *  forward(x) returns an internal buffer (copy it if you need to keep it). backward(dy) accumulates gradients and returns dL/dx. */
export function mlp(sizes, acts, seed = 1) {
  const r = rng(seed), L = sizes.length - 1;
  const W = [], b = [], gW = [], gb = [], mW = [], vW = [], mb = [], vb = [], A = [new Float64Array(sizes[0])], Dl = [];
  for (let l = 0; l < L; l++) {
    const ni = sizes[l], no = sizes[l + 1], s = Math.sqrt(1 / ni);
    W.push(Float64Array.from({ length: ni * no }, () => r.normal(0, s)));
    b.push(new Float64Array(no));
    [gW, mW, vW].forEach((arr) => arr.push(new Float64Array(ni * no)));
    [gb, mb, vb].forEach((arr) => arr.push(new Float64Array(no)));
    A.push(new Float64Array(no)); Dl.push(new Float64Array(no));
  }
  const dx = new Float64Array(sizes[0]);
  function forward(x) {
    A[0].set(x);
    for (let l = 0; l < L; l++) {
      const ni = sizes[l], no = sizes[l + 1], a = A[l], out = A[l + 1], w = W[l], th = acts[l] === 'tanh';
      for (let o = 0; o < no; o++) { let s = b[l][o]; const off = o * ni; for (let i = 0; i < ni; i++) s += w[off + i] * a[i]; out[o] = th ? Math.tanh(s) : s; }
    }
    return A[L];
  }
  function backward(dy) {
    const dL = Dl[L - 1], th = acts[L - 1] === 'tanh';
    for (let o = 0; o < sizes[L]; o++) dL[o] = dy[o] * (th ? 1 - A[L][o] * A[L][o] : 1);
    for (let l = L - 1; l >= 0; l--) {
      const ni = sizes[l], no = sizes[l + 1], a = A[l], d = Dl[l], w = W[l], g = gW[l];
      for (let o = 0; o < no; o++) { gb[l][o] += d[o]; const off = o * ni; for (let i = 0; i < ni; i++) g[off + i] += d[o] * a[i]; }
      const prev = l > 0 ? Dl[l - 1] : dx, pth = l > 0 && acts[l - 1] === 'tanh';
      for (let i = 0; i < ni; i++) { let s = 0; for (let o = 0; o < no; o++) s += w[o * ni + i] * d[o]; prev[i] = pth ? s * (1 - a[i] * a[i]) : s; }
    }
    return dx;
  }
  let t = 0;
  function step(lr, n = 1) {
    t++;
    const c1 = 1 - 0.9 ** t, c2 = 1 - 0.999 ** t;
    const upd = (p, g, m, v) => { for (let k = 0; k < p.length; k++) { const gg = g[k] / n; m[k] = 0.9 * m[k] + 0.1 * gg; v[k] = 0.999 * v[k] + 0.001 * gg * gg; p[k] -= lr * (m[k] / c1) / (Math.sqrt(v[k] / c2) + 1e-8); g[k] = 0; } };
    for (let l = 0; l < L; l++) { upd(W[l], gW[l], mW[l], vW[l]); upd(b[l], gb[l], mb[l], vb[l]); }
  }
  return { forward, backward, step, sizes, W, b, layerOut: (l) => A[l] };
}

/** Eigen-decomposition of a small symmetric matrix (cyclic Jacobi). Returns { vals, vecs } sorted by value, descending. */
export function symEig(Ain) {
  const n = Ain.length, M = Ain.map((row) => row.slice()), V = Ain.map((row, i) => row.map((_, j) => (i === j ? 1 : 0)));
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0; for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += M[i][j] ** 2;
    if (off < 1e-20) break;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
      if (Math.abs(M[p][q]) < 1e-15) continue;
      const th = 0.5 * Math.atan2(2 * M[p][q], M[q][q] - M[p][p]), c = Math.cos(th), s = Math.sin(th);
      for (let k = 0; k < n; k++) { const a = M[k][p], bq = M[k][q]; M[k][p] = c * a - s * bq; M[k][q] = s * a + c * bq; }
      for (let k = 0; k < n; k++) { const a = M[p][k], bq = M[q][k]; M[p][k] = c * a - s * bq; M[q][k] = s * a + c * bq; }
      for (let k = 0; k < n; k++) { const a = V[k][p], bq = V[k][q]; V[k][p] = c * a - s * bq; V[k][q] = s * a + c * bq; }
    }
  }
  const order = d3.range(n).sort((i, j) => M[j][j] - M[i][i]);
  return { vals: order.map((i) => M[i][i]), vecs: order.map((i) => V.map((row) => row[i])) };
}

/** PCA of rows X (arrays). Returns { mu, vals, vecs, recon(x, k) }. */
export function pcaFit(X) {
  const n = X.length, D = X[0].length, mu = new Array(D).fill(0);
  X.forEach((x) => x.forEach((v, j) => { mu[j] += v / n; }));
  const C = d3.range(D).map(() => new Array(D).fill(0));
  X.forEach((x) => { for (let i = 0; i < D; i++) { const a = x[i] - mu[i]; for (let j = i; j < D; j++) C[i][j] += a * (x[j] - mu[j]) / n; } });
  for (let i = 0; i < D; i++) for (let j = 0; j < i; j++) C[i][j] = C[j][i];
  const { vals, vecs } = symEig(C);
  const recon = (x, k) => {
    const out = mu.slice();
    for (let c = 0; c < k; c++) { let s = 0; for (let j = 0; j < D; j++) s += (x[j] - mu[j]) * vecs[c][j]; for (let j = 0; j < D; j++) out[j] += s * vecs[c][j]; }
    return out;
  };
  return { mu, vals, vecs, recon };
}

// ---------------- Figure 1 data: 2-D shapes that are really 1-D ----------------
const SHAPES = {
  arc: { name: 'Horseshoe (curved 1-D)', f: (t) => { const th = Math.PI * (-0.15 + 1.3 * t); return [1.05 * Math.cos(th), 1.05 * Math.sin(th) - 0.35]; } },
  line: { name: 'Straight line (linear 1-D)', f: (t) => [-1.1 + 2.2 * t, 0.45 * (-1.1 + 2.2 * t)] },
  ring: { name: 'Closed ring (no 1-D cut is perfect)', f: (t) => { const th = 2 * Math.PI * t; return [0.95 * Math.cos(th), 0.95 * Math.sin(th)]; } },
};
function shapeData(key) {
  const r = rng(11), f = SHAPES[key].f;
  return d3.range(200).map(() => { const t = r(); const [x, y] = f(t); return { t, x: [x + r.normal(0, 0.06), y + r.normal(0, 0.06)] }; });
}

// ---------------- Figure 2 data: 24-gene profiles along a trajectory ----------------
const G = 24;
function profiles(n, seed) {
  const r = rng(seed);
  return d3.range(n).map(() => {
    const p = 0.08 + 0.84 * r(), w = 0.05 + 0.08 * r();
    return { p, w, x: d3.range(G).map((g) => Math.exp(-((g / (G - 1) - p) ** 2) / (2 * w * w)) + r.normal(0, 0.03)) };
  });
}

export default async function render(main) {
  header(main, 'Autoencoders',
    'An autoencoder is a network trained to <b>copy its input to its output through a narrow bottleneck</b>. It cannot just memorise, so it must find a short code that captures the structure of the data. Nobody gives it labels: the input is its own target.');

  // =================== Figure 1 ===================
  const fig = figure(main, 'Squeeze 2-D points through a 1-number bottleneck',
    'Each point (x₁, x₂) goes into the <b>encoder</b> (2 → 16 tanh units → 1 number z) and the <b>decoder</b> (z → 16 tanh units → 2) tries to rebuild it. Left: data coloured by its true position along the shape; small dark dots are the reconstructions; grey ticks join each point to its reconstruction. The thick line is the decoder applied to every z: the 1-D curve the autoencoder has learned. Dashed: the PCA line (best straight 1-D fit). Right: the code z each point gets, against its true position. Trained live by Adam on the mean squared reconstruction error.');
  let shapeKey = 'arc', modelKey = 'nonlinear', data, net, P, stepN = 0, hist = [], stop = null;
  select(fig.controls, { label: 'Data', value: shapeKey, options: Object.entries(SHAPES).map(([k, v]) => [k, v.name]), onChange: (k) => { shapeKey = k; reset(true); } });
  select(fig.controls, { label: 'Model', value: modelKey, options: [['nonlinear', 'nonlinear (tanh hidden layers)'], ['linear', 'linear (2 → 1 → 2, no hidden layer)']], onChange: (k) => { modelKey = k; reset(true); } });
  const runBtn = button(fig.controls, 'Train', () => (stop ? pause() : run()), true);
  button(fig.controls, 'Reset', () => reset(false));
  const ro = stats(fig.root);
  const row = el('div', { class: 'fig-row' }); fig.body.append(row);
  const left = el('div'), right = el('div'); row.append(left, right);
  legend(left, [['data (colour = position along shape)', cat(0)], ['reconstruction', ink()], ['decoder curve', cat(2)], ['PCA line', ink2()]]);
  const F = frame(left, { width: 460, height: 440, x: [-1.6, 1.6], y: [-1.6, 1.6], xLabel: 'x₁', yLabel: 'x₂' });
  ['pca', 'ticks', 'curve', 'pts', 'rec'].forEach((k) => F.layer(k));
  const Z = frame(right, { width: 460, height: 260, x: [0, 1], y: [-1, 1], xLabel: 'true position along the shape', yLabel: 'code z' });
  const LC = frame(right, { width: 460, height: 200, x: [0, 2000], y: [0, 0.5], xLabel: 'training step', yLabel: 'reconstruction MSE' });
  const posColor = (u) => d3.interpolateRgb(cat(0), cat(1))(u);

  function reset(newData) {
    pause();
    if (newData || !data) { data = shapeData(shapeKey); P = pcaFit(data.map((d) => d.x)); }
    net = modelKey === 'linear' ? mlp([2, 1, 2], ['linear', 'linear'], 3) : mlp([2, 16, 1, 16, 2], ['tanh', 'linear', 'tanh', 'linear'], 3);
    stepN = 0; hist = [[0, evalLoss()]];
    paint();
  }
  const codeLayer = () => (modelKey === 'linear' ? 1 : 2);
  function evalLoss() {
    let s = 0; data.forEach((d) => { const o = net.forward(d.x); s += ((o[0] - d.x[0]) ** 2 + (o[1] - d.x[1]) ** 2) / 2; });
    return s / data.length;
  }
  const dy = new Float64Array(2);
  function trainStep() {
    data.forEach((d) => { const o = net.forward(d.x); dy[0] = o[0] - d.x[0]; dy[1] = o[1] - d.x[1]; net.backward(dy); });
    net.step(0.01, data.length); stepN++;
  }
  function run() {
    runBtn.textContent = 'Pause';
    stop = animate(() => {
      const t0 = performance.now();
      while (performance.now() - t0 < 12 && stepN < 2000) trainStep();
      hist.push([stepN, evalLoss()]); paint();
      if (stepN >= 2000) { pause(); return false; }
      return true;
    });
  }
  function pause() { if (stop) { stop(); stop = null; } runBtn.textContent = stepN >= 2000 ? 'Done' : 'Train'; }

  function decodeZ(z) {
    // run only the decoder half: feed z into the code layer by temporarily using layer weights
    const L = net.sizes.length - 1, c = codeLayer();
    let a = [z];
    for (let l = c; l < L; l++) {
      const ni = net.sizes[l], no = net.sizes[l + 1], out = new Array(no);
      for (let o = 0; o < no; o++) { let s = net.b[l][o]; for (let i = 0; i < ni; i++) s += net.W[l][o * ni + i] * a[i]; out[o] = (modelKey === 'nonlinear' && l < L - 1) ? Math.tanh(s) : s; }
      a = out;
    }
    return a;
  }
  function paint() {
    const pts = data.map((d) => { const o = net.forward(d.x); const z = net.layerOut(codeLayer())[0]; return { t: d.t, x: d.x, r: [o[0], o[1]], z }; });
    const zr = d3.extent(pts, (d) => d.z), pad = 0.15 * (zr[1] - zr[0] || 1);
    const cl = (v) => Math.max(-1.6, Math.min(1.6, v));
    // PCA line
    const v = P.vecs[0];
    F.layer('pca').selectAll('line').data([0]).join('line')
      .attr('x1', F.x(P.mu[0] - 3 * v[0])).attr('y1', F.y(P.mu[1] - 3 * v[1])).attr('x2', F.x(P.mu[0] + 3 * v[0])).attr('y2', F.y(P.mu[1] + 3 * v[1]))
      .attr('stroke', ink2()).attr('stroke-width', 1.5).attr('stroke-dasharray', '6 4');
    F.layer('ticks').selectAll('line').data(pts).join('line')
      .attr('x1', (d) => F.x(d.x[0])).attr('y1', (d) => F.y(d.x[1])).attr('x2', (d) => F.x(cl(d.r[0]))).attr('y2', (d) => F.y(cl(d.r[1])))
      .attr('stroke', muted()).attr('stroke-opacity', 0.6).attr('stroke-width', 1);
    const curve = d3.range(121).map((i) => decodeZ(zr[0] - pad + (zr[1] - zr[0] + 2 * pad) * i / 120)).map((p) => [cl(p[0]), cl(p[1])]);
    F.layer('curve').selectAll('path').data([curve]).join('path').attr('fill', 'none').attr('stroke', cat(2)).attr('stroke-width', 3).attr('stroke-linecap', 'round')
      .attr('d', d3.line().x((p) => F.x(p[0])).y((p) => F.y(p[1])));
    F.layer('pts').selectAll('circle').data(pts).join('circle').attr('cx', (d) => F.x(d.x[0])).attr('cy', (d) => F.y(d.x[1])).attr('r', 4.2)
      .attr('fill', (d) => posColor(d.t)).attr('stroke', css('--surface')).attr('stroke-width', 1)
      .on('mousemove', (e, d) => tip(e, `input (${d.x[0].toFixed(2)}, ${d.x[1].toFixed(2)})<br>code z = ${d.z.toFixed(3)}<br>reconstruction (${d.r[0].toFixed(2)}, ${d.r[1].toFixed(2)})`)).on('mouseleave', untip);
    F.layer('rec').selectAll('circle').data(pts).join('circle').attr('cx', (d) => F.x(cl(d.r[0]))).attr('cy', (d) => F.y(cl(d.r[1]))).attr('r', 1.8).attr('fill', ink()).attr('pointer-events', 'none');
    // latent vs position
    Z.y.domain([zr[0] - pad, zr[1] + pad]); Z.g.select('.axis.y').call(d3.axisLeft(Z.y).ticks(5));
    Z.layer('p').selectAll('circle').data(pts).join('circle').attr('cx', (d) => Z.x(d.t)).attr('cy', (d) => Z.y(d.z)).attr('r', 4)
      .attr('fill', (d) => posColor(d.t)).attr('stroke', css('--surface')).attr('stroke-width', 1);
    // loss curve
    const pcaMSE = P.vals[1] / 2; // per-coordinate MSE of the 1-component PCA reconstruction
    const ymax = Math.max(pcaMSE * 1.6, d3.max(hist, (h) => h[1]) * 1.05);
    LC.y.domain([0, ymax]); LC.g.select('.axis.y').call(d3.axisLeft(LC.y).ticks(4));
    LC.layer('ref').selectAll('line').data([pcaMSE]).join('line').attr('x1', 0).attr('x2', LC.w).attr('y1', (d) => LC.y(d)).attr('y2', (d) => LC.y(d))
      .attr('stroke', ink2()).attr('stroke-width', 1.5).attr('stroke-dasharray', '6 4');
    LC.layer('ref').selectAll('text').data([pcaMSE]).join('text').attr('x', LC.w - 4).attr('y', (d) => LC.y(d) - 5).attr('text-anchor', 'end').attr('font-size', 12).attr('fill', ink2()).text('PCA, 1 component');
    LC.layer('c').selectAll('path').data([hist]).join('path').attr('fill', 'none').attr('stroke', cat(2)).attr('stroke-width', 2)
      .attr('d', d3.line().x((h) => LC.x(h[0])).y((h) => LC.y(Math.min(ymax, h[1]))));
    // angle between linear AE decoder direction and PC1
    let extra = {};
    if (modelKey === 'linear') {
      const wv = [net.W[1][0], net.W[1][1]], n = Math.hypot(...wv) || 1;
      const ang = Math.acos(Math.min(1, Math.abs((wv[0] * v[0] + wv[1] * v[1]) / n))) * 180 / Math.PI;
      extra = { 'angle between AE line and PC1': ang.toFixed(1) + '°' };
    }
    const cur = hist[hist.length - 1][1];
    ro.set({ step: stepN, 'autoencoder MSE': cur.toFixed(4), 'PCA (1 comp.) MSE': pcaMSE.toFixed(4), 'AE ÷ PCA': (cur / pcaMSE).toFixed(2), ...extra });
  }
  reset(true);

  // =================== Figure 2 ===================
  const fig2 = figure(main, 'Bottleneck size: how many numbers does a profile need?',
    `${300} simulated cells, each a 24-gene profile. The genes switch on in sequence along a differentiation trajectory, so every profile is one bump: <b>where</b> the cell is (bump position) and <b>how broad</b> its programme is (bump width), plus small noise. Two hidden numbers generate 24. The autoencoder is 24 → 32 tanh → <b>k</b> → 32 tanh → 24, trained live; PCA keeps the top k components. Panels show 8 held-out cells.`);
  let k = 2, net2 = null, st2 = 0, stop2 = null;
  const train2 = profiles(300, 21), test2 = profiles(120, 22);
  const P2 = pcaFit(train2.map((d) => d.x));
  const aeErr = {};
  const ks = slider(fig2.controls, { label: 'bottleneck size k', min: 1, max: 8, step: 1, value: k, onInput: (v) => { k = v; start2(); } });
  const run2 = button(fig2.controls, 'Retrain (new init)', () => { delete aeErr[k]; delete aeNets[k]; start2(); }, true);
  const ro2 = stats(fig2.root);
  legend(fig2.body, [['original profile', ink2()], ['autoencoder reconstruction', cat(2)], ['PCA reconstruction (same k)', cat(1)]]);
  const row2 = el('div', { class: 'fig-row' }); fig2.body.append(row2);
  const l2 = el('div'), r2 = el('div'); row2.append(l2, r2);
  const TW = 520, TH = 380, cols = 2, rows = 4, cw = TW / cols, ch = TH / rows;
  const tiles = d3.select(l2).append('svg').attr('viewBox', `0 0 ${TW} ${TH}`).attr('width', TW).attr('role', 'img');
  const SHOW = [0, 7, 14, 3, 25, 41, 9, 60].map((i) => test2[i]).sort((a, b) => a.p - b.p);
  const E2 = frame(r2, { width: 440, height: 380, x: [0.5, 8.5], y: [0, 1], xLabel: 'bottleneck size k', yLabel: 'held-out MSE (log)', yScale: d3.scaleLog() });
  E2.g.select('.axis.x').call(d3.axisBottom(E2.x).ticks(8).tickFormat(d3.format('d')));

  const mse = (a, b) => { let s = 0; for (let j = 0; j < a.length; j++) s += (a[j] - b[j]) ** 2; return s / a.length; };
  const pcaErr = d3.range(1, 9).map((kk) => ({ k: kk, e: d3.mean(test2, (d) => mse(P2.recon(d.x, kk), d.x)) }));
  const STEPS2 = 3000, BATCH = 32;
  let brng = rng(5);
  const aeNets = {}, seeds = {};
  function start2() {
    if (stop2) { stop2(); stop2 = null; run2.textContent = 'Retrain (new init)'; }
    if (aeNets[k]) { net2 = aeNets[k]; st2 = STEPS2; paint2(); return; } // already trained for this k
    seeds[k] = (seeds[k] || 0) + 1;
    net2 = mlp([G, 32, k, 32, G], ['tanh', 'linear', 'tanh', 'linear'], 7 + 10 * k + seeds[k]); st2 = 0; brng = rng(5 + seeds[k]);
    const kk = k, nn = net2;
    run2.textContent = 'Training…';
    stop2 = animate(() => {
      const t0 = performance.now();
      while (performance.now() - t0 < 12 && st2 < STEPS2) step2();
      if (st2 >= STEPS2) { aeErr[kk] = aeTestErr(); aeNets[kk] = nn; }
      paint2();
      if (st2 >= STEPS2) { stop2 = null; run2.textContent = 'Retrain (new init)'; return false; }
      return true;
    });
  }
  const dy2 = new Float64Array(G);
  function step2() {
    for (let i = 0; i < BATCH; i++) { const d = train2[brng.int(train2.length)]; const o = net2.forward(d.x); for (let j = 0; j < G; j++) dy2[j] = o[j] - d.x[j]; net2.backward(dy2); }
    net2.step(0.005, BATCH); st2++;
  }
  const aeRecon = (x) => Array.from(net2.forward(x));
  const aeTestErr = () => d3.mean(test2, (d) => mse(aeRecon(d.x), d.x));
  function paint2() {
    const gx = d3.scaleLinear().domain([0, G - 1]), gy = d3.scaleLinear().domain([-0.2, 1.2]);
    const cells = SHOW.map((d, i) => ({ d, i, ae: aeRecon(d.x), pc: P2.recon(d.x, k) }));
    const g = tiles.selectAll('g.tile').data(cells).join((en) => { const gg = en.append('g').attr('class', 'tile'); gg.append('rect'); gg.append('text'); return gg; })
      .attr('transform', (c) => `translate(${(c.i % cols) * cw},${Math.floor(c.i / cols) * ch})`);
    g.select('rect').attr('x', 4).attr('y', 4).attr('width', cw - 8).attr('height', ch - 8).attr('rx', 6).attr('fill', css('--surface-2'));
    g.select('text').attr('x', 12).attr('y', 20).attr('font-size', 11).attr('fill', ink2()).text((c) => `position ${c.d.p.toFixed(2)}, width ${c.d.w.toFixed(2)}`);
    gx.range([12, cw - 12]); gy.range([ch - 10, 26]);
    const ln = d3.line().x((v, j) => gx(j)).y((v) => gy(Math.max(-0.2, Math.min(1.2, v))));
    g.selectAll('path.o').data((c) => [c.d.x]).join('path').attr('class', 'o').attr('fill', 'none').attr('stroke', ink2()).attr('stroke-width', 1.2).attr('stroke-dasharray', '2 2').attr('d', ln);
    g.selectAll('path.p').data((c) => [c.pc]).join('path').attr('class', 'p').attr('fill', 'none').attr('stroke', cat(1)).attr('stroke-width', 2).attr('d', ln);
    g.selectAll('path.a').data((c) => [c.ae]).join('path').attr('class', 'a').attr('fill', 'none').attr('stroke', cat(2)).attr('stroke-width', 2).attr('d', ln);
    // error chart
    const cur = st2 > 0 ? aeTestErr() : null;
    const aePts = Object.entries(aeErr).map(([kk, e]) => ({ k: +kk, e }));
    const all = pcaErr.map((d) => d.e).concat(aePts.map((d) => d.e)).concat(cur ? [cur] : []);
    E2.y.domain([Math.max(5e-5, d3.min(all) * 0.7), d3.max(all) * 1.4]); E2.g.select('.axis.y').call(d3.axisLeft(E2.y).ticks(4, '~g'));
    E2.layer('pl').selectAll('path').data([pcaErr]).join('path').attr('fill', 'none').attr('stroke', cat(1)).attr('stroke-width', 2)
      .attr('d', d3.line().x((d) => E2.x(d.k)).y((d) => E2.y(d.e)));
    E2.layer('pc').selectAll('circle').data(pcaErr).join('circle').attr('cx', (d) => E2.x(d.k)).attr('cy', (d) => E2.y(d.e)).attr('r', 4.5).attr('fill', cat(1)).attr('stroke', css('--surface')).attr('stroke-width', 1.5)
      .on('mousemove', (e, d) => tip(e, `PCA, k = ${d.k}<br>held-out MSE ${d.e.toExponential(2)}`)).on('mouseleave', untip);
    E2.layer('mk').selectAll('line').data([k]).join('line').attr('x1', (d) => E2.x(d)).attr('x2', (d) => E2.x(d)).attr('y1', 0).attr('y2', E2.h).attr('stroke', muted()).attr('stroke-dasharray', '3 3');
    const aeShow = aePts.filter((d) => d.k !== k || st2 >= STEPS2);
    if (st2 < STEPS2 && cur) aeShow.push({ k, e: cur, live: true });
    E2.layer('ae').selectAll('circle').data(aeShow).join('circle').attr('cx', (d) => E2.x(d.k)).attr('cy', (d) => E2.y(d.e)).attr('r', (d) => (d.live ? 5 : 6))
      .attr('fill', (d) => (d.live ? css('--surface') : cat(2))).attr('stroke', (d) => (d.live ? cat(2) : css('--surface'))).attr('stroke-width', 2)
      .on('mousemove', (e, d) => tip(e, `autoencoder, k = ${d.k}${d.live ? ' (still training)' : ''}<br>held-out MSE ${d.e.toExponential(2)}`)).on('mouseleave', untip);
    const pe = pcaErr[k - 1].e;
    ro2.set({ 'training step': `${st2} / ${STEPS2}`, 'AE held-out MSE': cur ? cur.toExponential(2) : '–', 'PCA held-out MSE': pe.toExponential(2), 'AE ÷ PCA': cur ? (cur / pe).toFixed(2) : '–' });
  }
  start2();

  section(main, 'What to notice', `<ul>
    <li><b>Horseshoe, nonlinear model.</b> Press Train. The decoder curve starts as a short squiggle, then bends to follow the horseshoe, and the grey ticks shrink. The PCA line can only cut straight across, so every PCA reconstruction lands on a straight line and the bend is lost. Watch the MSE drop below the PCA reference.</li>
    <li><b>Right panel.</b> When training has worked, code z rises (or falls) steadily with the true position: the network has discovered "where along the shape am I" without being told. The sign and scale of z are arbitrary.</li>
    <li><b>Linear model.</b> Switch to <i>linear (2 → 1 → 2)</i> and train. The decoder line swings onto the dashed PCA line and the angle readout heads to 0°. A linear autoencoder with squared error learns the same subspace as PCA: it is PCA, found by gradient descent.</li>
    <li><b>Closed ring.</b> One number cannot go round a loop and come back. The nonlinear model has to cut the ring somewhere: look for the gap in the decoder curve and the jump in the right panel, where neighbouring points get very different codes.</li>
    <li><b>Bottleneck slider (second figure).</b> The profiles are made from 2 hidden numbers. At k = 2 PCA draws wavy compromises (see the orange ripples), because a sliding bump is not a straight-line (linear) family of shapes; the autoencoder learns the curved family, and its error at k = 2 is about what PCA needs 6 components to reach. At k = 1 the autoencoder can still encode where the bump is, but not how wide. Past k = 2 it keeps improving a little, partly because a small network trained for 3,000 steps is not a perfect fit; the floor set by the added noise is 0.0009 (its variance).</li>
  </ul>`);

  steps(main, 'How it works', [
    'Build two networks: an <b>encoder</b> that maps input x to a short code z (the bottleneck), and a <b>decoder</b> that maps z back to x̂, the same shape as x.',
    'Loss = reconstruction error, here the mean of (x − x̂)². The target is the input itself, so no labels are needed: this is <b>self-supervised</b> learning.',
    'Train both networks together by backpropagation and gradient descent (Adam here). The bottleneck is what forces learning: with a code as big as the input, the network could just pass x through.',
    'Afterwards, use the encoder alone for a compressed representation (for clustering, visualisation, denoising), or the decoder alone to map codes back to data space.',
    'Variants change what the bottleneck is: <b>denoising</b> autoencoders corrupt x and reconstruct the clean x; <b>sparse</b> ones penalise active units; <a href="#/concepts/vae">variational</a> ones make the code a probability distribution.',
  ]);
  math(main, 'z = encoder(x),   x̂ = decoder(z),   loss = mean ‖x − x̂‖²\nlinear encoder + linear decoder + squared error\n  ⇒  the optimal decoder spans the top-k principal components (same subspace as PCA)');

  callout(main, '', 'In biology', 'Single-cell RNA-seq is the natural home: 20,000 genes per cell, but cells vary along far fewer programmes. <b>scVI</b> (scvi-tools) is a variational autoencoder over raw counts with a negative-binomial decoder; its latent space replaces PCA for clustering, UMAP and batch integration. <b>DCA</b> (deep count autoencoder) uses the reconstruction for denoising and imputation of dropouts, and <b>SAUCIE</b> uses a regularised autoencoder for mass cytometry. In spatial omics, autoencoders over spot or neighbourhood profiles feed domain detection.');
  callout(main, 'warn', 'Pitfalls', `<ul>
    <li><b>Low reconstruction error is not the goal.</b> A big enough autoencoder can reconstruct almost anything, including noise and batch effects. What you want is a <i>useful</i> code, so judge it on downstream tasks.</li>
    <li><b>The code axes mean nothing on their own.</b> Like the sign of z above, the latent space can be rotated, flipped or stretched with no change in loss. Do not read latent dimension 1 as "a biological axis" without checking.</li>
    <li><b>Gaps in the latent space.</b> A plain autoencoder only learns to decode codes it has seen. Decode a point between clusters and you may get nonsense: that is why <a href="#/concepts/vae">VAEs</a> exist.</li>
    <li><b>Compare with PCA first.</b> On many omics datasets a well-tuned PCA is a strong baseline. Use an autoencoder when the structure is clearly nonlinear, or when you need a count-aware decoder (as in scVI).</li>
  </ul>`);
  seeAlso(main, [['concepts', 'pca', 'PCA'], ['concepts', 'vae', 'Variational autoencoders'], ['concepts', 'neural-network', 'Neural network playground'], ['concepts', 'tsne-umap', 't-SNE & UMAP'], ['concepts', 'foundation-models', 'Self-supervision & foundation models'], ['omics', 'sc-integration', 'Batch integration']]);
}
