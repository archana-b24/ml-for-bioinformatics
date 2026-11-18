// Dropout, batch norm, early stopping, weight decay and learning-rate schedules, each shown on a small live computation.
import { header, figure, slider, select, button, checkbox, stats, section, steps, callout, math, seeAlso, animate, every, legend, tip, untip, el } from '../../lib/ui.js';
import { frame, cat, css, ink, ink2, muted } from '../../lib/plot.js';
import { rng, regression1d, linspace, mean } from '../../lib/data.js';
const d3 = window.d3;

// =====================================================================================
// Tiny 1 -> H -> H -> 1 tanh MLP with inverted dropout on both hidden layers, trained by Adam.
// =====================================================================================
const PARAMS = ['W1', 'b1', 'W2', 'b2', 'W3', 'b3'];
function makeNet(H, seed) {
  const r = rng(seed), n = { H, t: 0 };
  const g = (len, s) => Float64Array.from({ length: len }, () => r.normal(0, s));
  n.W1 = g(H, 2.5); n.b1 = g(H, 1.5); n.W2 = g(H * H, 1 / Math.sqrt(H)); n.b2 = g(H, 0.1); n.W3 = g(H, 1 / Math.sqrt(H)); n.b3 = new Float64Array(1);
  n.g = {}; n.m = {}; n.v = {};
  PARAMS.forEach((k) => { n.g[k] = new Float64Array(n[k].length); n.m[k] = new Float64Array(n[k].length); n.v[k] = new Float64Array(n[k].length); });
  n.buf = { t1: new Float64Array(H), t2: new Float64Array(H), a1: new Float64Array(H), a2: new Float64Array(H), m1: new Float64Array(H), m2: new Float64Array(H), dz2: new Float64Array(H) };
  return n;
}
// mask(i, layer) -> multiplier; null = no dropout
function forward(n, x, m1, m2) {
  const { H, W1, b1, W2, b2, W3, b3 } = n, { t1, t2, a1, a2 } = n.buf;
  for (let i = 0; i < H; i++) { t1[i] = Math.tanh(W1[i] * x + b1[i]); a1[i] = t1[i] * (m1 ? m1[i] : 1); }
  let y = b3[0];
  for (let j = 0; j < H; j++) {
    let z = b2[j]; const o = j * H;
    for (let i = 0; i < H; i++) z += W2[o + i] * a1[i];
    t2[j] = Math.tanh(z); a2[j] = t2[j] * (m2 ? m2[j] : 1); y += W3[j] * a2[j];
  }
  return y;
}
function trainStep(n, X, Y, p, r, lr = 0.01) {
  const { H, W2, W3 } = n, G = n.g, B = n.buf, N = X.length;
  PARAMS.forEach((k) => G[k].fill(0));
  let loss = 0;
  for (let s = 0; s < N; s++) {
    if (p > 0) for (let i = 0; i < H; i++) { B.m1[i] = r() < p ? 0 : 1 / (1 - p); B.m2[i] = r() < p ? 0 : 1 / (1 - p); }
    const y = forward(n, X[s], p > 0 ? B.m1 : null, p > 0 ? B.m2 : null);
    const e = y - Y[s]; loss += e * e;
    const dy = 2 * e / N;
    G.b3[0] += dy;
    for (let j = 0; j < H; j++) {
      G.W3[j] += dy * B.a2[j];
      B.dz2[j] = dy * W3[j] * (p > 0 ? B.m2[j] : 1) * (1 - B.t2[j] * B.t2[j]);
      G.b2[j] += B.dz2[j];
      const o = j * H; for (let i = 0; i < H; i++) G.W2[o + i] += B.dz2[j] * B.a1[i];
    }
    for (let i = 0; i < H; i++) {
      let da = 0; for (let j = 0; j < H; j++) da += W2[j * H + i] * B.dz2[j];
      const dz = da * (p > 0 ? B.m1[i] : 1) * (1 - B.t1[i] * B.t1[i]);
      G.W1[i] += dz * X[s]; G.b1[i] += dz;
    }
  }
  n.t++;
  const b1c = 1 - 0.9 ** n.t, b2c = 1 - 0.999 ** n.t;
  PARAMS.forEach((k) => {
    const w = n[k], g = G[k], m = n.m[k], v = n.v[k];
    for (let i = 0; i < w.length; i++) { m[i] = 0.9 * m[i] + 0.1 * g[i]; v[i] = 0.999 * v[i] + 0.001 * g[i] * g[i]; w[i] -= lr * (m[i] / b1c) / (Math.sqrt(v[i] / b2c) + 1e-8); }
  });
  return loss / N;
}

// =====================================================================================
// Seeded linear-regression problems for early stopping, weight decay and LR schedules.
// =====================================================================================
function linProblem({ n, nVal, p, wTrue, noise, seed, colScale }) {
  const r = rng(seed);
  const row = () => Float64Array.from({ length: p }, (_, j) => r.normal() * (colScale ? colScale[j] : 1));
  const w = Float64Array.from({ length: p }, (_, j) => wTrue[j] || 0);
  const mk = (m) => { const X = [], Y = []; for (let i = 0; i < m; i++) { const x = row(); X.push(x); let y = r.normal(0, noise); for (let j = 0; j < p; j++) y += w[j] * x[j]; Y.push(y); } return { X, Y }; };
  return { tr: mk(n), va: mk(nVal), p };
}
function mse(w, { X, Y }) { let s = 0; for (let i = 0; i < X.length; i++) { let y = 0; const x = X[i]; for (let j = 0; j < w.length; j++) y += w[j] * x[j]; s += (y - Y[i]) ** 2; } return s / X.length; }
function sgdEpoch(w, data, order, B, lr, lambda) {
  const p = w.length, g = new Float64Array(p);
  for (let s = 0; s < order.length; s += B) {
    g.fill(0); const idx = order.slice(s, s + B);
    idx.forEach((i) => { const x = data.X[i]; let y = 0; for (let j = 0; j < p; j++) y += w[j] * x[j]; const e = 2 * (y - data.Y[i]) / idx.length; for (let j = 0; j < p; j++) g[j] += e * x[j]; });
    for (let j = 0; j < p; j++) w[j] -= lr * (g[j] + 2 * lambda * w[j]);
  }
}

export default async function render(main) {
  header(main, 'Dropout, batch norm, early stopping',
    'A deep net has far more parameters than a typical omics dataset has samples. These practical tricks are what make such nets <b>train stably and generalise</b> anyway: dropout and weight decay limit what the net can memorise, batch norm keeps signals at a sensible scale through many layers, and early stopping and learning-rate schedules decide <b>when and how fast</b> to move.');

  // =================== Figure 1: dropout ===================
  const fig = figure(main, 'Dropout: train a different thinned network on every step', '');
  let p = 0.3, stopTrain = null;
  const pS = slider(fig.controls, { label: 'Dropout rate p', min: 0.1, max: 0.7, step: 0.05, value: 0.3, onInput: (v) => { p = v; } });
  pS.input.addEventListener('change', () => startTrain());
  button(fig.controls, 'Retrain', () => startTrain(), true);
  let showThin = true;
  checkbox(fig.controls, { label: 'show 8 random thinned networks', checked: true, onChange: (c) => { showThin = c; paintFit(); } });
  const r1 = stats(fig.root); fig.root.insertBefore(r1.root, fig.body);
  const row1 = el('div', { class: 'fig-row' }); fig.body.append(row1);
  const dl = el('div'), dr = el('div'); row1.append(dl, dr);

  // network diagram (10 of the 48 units per hidden layer drawn)
  const D = frame(dl, { width: 420, height: 360, x: [0, 3], y: [0, 11], axes: false, grid: false, margin: { top: 24, right: 20, bottom: 10, left: 20 } });
  const SHOWN = 10, layerN = [1, SHOWN, SHOWN, 1];
  const nodePos = layerN.map((k, l) => d3.range(k).map((i) => ({ l, i, x: l, y: k === 1 ? 5 : 0.5 + i * (10 - 0.5 * 2) / (SHOWN - 1) + 0.5 })));
  ['input x', 'hidden 1', 'hidden 2', 'output ŷ'].forEach((t, l) => D.g.append('text').attr('x', D.x(l)).attr('y', -8).attr('text-anchor', 'middle').attr('font-size', 12).attr('fill', ink2()).text(t));
  const flick = rng(123);
  let dmask = [[1], Array(SHOWN).fill(1), Array(SHOWN).fill(1), [1]];
  function paintDiagram() {
    const E = [];
    for (let l = 0; l < 3; l++) nodePos[l].forEach((a) => nodePos[l + 1].forEach((b) => E.push({ a, b, on: dmask[l][a.i] && dmask[l + 1][b.i] })));
    D.layer('e').selectAll('line').data(E).join('line').attr('x1', (d) => D.x(d.a.x)).attr('y1', (d) => D.y(d.a.y)).attr('x2', (d) => D.x(d.b.x)).attr('y2', (d) => D.y(d.b.y))
      .attr('stroke', (d) => (d.on ? ink2() : css('--grid'))).attr('stroke-opacity', (d) => (d.on ? 0.35 : 0.5)).attr('stroke-width', 0.8);
    D.layer('n').selectAll('circle').data(nodePos.flat()).join('circle').attr('cx', (d) => D.x(d.x)).attr('cy', (d) => D.y(d.y)).attr('r', 9)
      .attr('fill', (d) => (dmask[d.l][d.i] ? cat(1) : css('--surface'))).attr('stroke', (d) => (dmask[d.l][d.i] ? css('--surface') : muted()))
      .attr('stroke-dasharray', (d) => (dmask[d.l][d.i] ? null : '2 2')).attr('stroke-width', 1.5);
    D.layer('x').selectAll('text').data(nodePos.flat().filter((d) => !dmask[d.l][d.i])).join('text').attr('x', (d) => D.x(d.x)).attr('y', (d) => D.y(d.y) + 4)
      .attr('text-anchor', 'middle').attr('font-size', 11).attr('fill', muted()).text('×');
  }
  every(650, () => { dmask = [[1], d3.range(SHOWN).map(() => (flick() < p ? 0 : 1)), d3.range(SHOWN).map(() => (flick() < p ? 0 : 1)), [1]]; paintDiagram(); });
  paintDiagram();

  // regression fit panel
  const F = frame(dr, { width: 500, height: 360, x: [-1.05, 1.05], y: [-2, 2], xLabel: 'x', yLabel: 'y' });
  const truth = (x) => Math.sin(2.5 * x);
  const train = regression1d(16, truth, 0.3, 11);
  const X = train.map((d) => d.x), Y = train.map((d) => d.y);
  const grid = linspace(-1, 1, 120);
  legend(dr, [['no dropout', cat(0)], ['dropout (full net at test time)', cat(1)], ['true curve', muted()]]);
  F.layer('truth').append('path').datum(grid.map((x) => ({ x, y: truth(x) }))).attr('fill', 'none').attr('stroke', muted()).attr('stroke-width', 2).attr('stroke-dasharray', '5 4')
    .attr('d', d3.line().x((d) => F.x(d.x)).y((d) => F.y(d.y)));
  F.layer('thin'); F.layer('fits');
  F.layer('pts').selectAll('circle').data(train).join('circle').attr('cx', (d) => F.x(d.x)).attr('cy', (d) => F.y(d.y)).attr('r', 4.5).attr('fill', ink()).attr('stroke', css('--surface')).attr('stroke-width', 1.5)
    .on('mousemove', (e, d) => tip(e, `training point<br>x = ${d.x.toFixed(2)}, y = ${d.y.toFixed(2)}`)).on('mouseleave', untip);

  const TOTAL = 3000;
  let nets, trLoss = [0, 0], step = 0, trainR;
  function testErr(net) { return mean(grid.map((x) => (forward(net, x) - truth(x)) ** 2)); }
  function paintFit() {
    if (!nets) return;
    const curve = (net, m1, m2) => grid.map((x) => ({ x, y: Math.max(-2.4, Math.min(2.4, forward(net, x, m1, m2))) }));
    const H = nets[1].H, mr = rng(99), thin = [];
    if (showThin) for (let k = 0; k < 8; k++) {
      const m1 = Float64Array.from({ length: H }, () => (mr() < p ? 0 : 1 / (1 - p))), m2 = Float64Array.from({ length: H }, () => (mr() < p ? 0 : 1 / (1 - p)));
      thin.push(curve(nets[1], m1, m2));
    }
    F.layer('thin').selectAll('path').data(thin).join('path').attr('fill', 'none').attr('stroke', cat(1)).attr('stroke-opacity', 0.3).attr('stroke-width', 1)
      .attr('d', d3.line().x((d) => F.x(d.x)).y((d) => F.y(d.y)));
    const c0 = curve(nets[0]), c1 = curve(nets[1]);
    F.layer('fits').selectAll('path').data([c0, c1]).join('path').attr('fill', 'none').attr('stroke', (d, i) => cat(i)).attr('stroke-width', 2.5)
      .attr('d', d3.line().x((d) => F.x(d.x)).y((d) => F.y(d.y)));
    // average of the thinned networks, to compare with the weight-scaled full net
    let ensGap = '–';
    if (thin.length) { const avg = grid.map((_, i) => mean(thin.map((t) => t[i].y))); ensGap = mean(avg.map((v, i) => Math.abs(v - c1[i].y))).toFixed(3); }
    r1.set({
      'training step': `${step} / ${TOTAL}`,
      'train MSE: no dropout / dropout': `${trLoss[0].toFixed(3)} / ${trLoss[1].toFixed(3)}`,
      'error vs true curve: no dropout / dropout': `${testErr(nets[0]).toFixed(3)} / ${testErr(nets[1]).toFixed(3)}`,
      '|mean of 8 thinned − full net|': ensGap,
    });
  }
  function startTrain() {
    if (stopTrain) stopTrain();
    nets = [makeNet(48, 5), makeNet(48, 5)]; step = 0; trainR = rng(17);
    stopTrain = animate(() => {
      for (let k = 0; k < 60 && step < TOTAL; k++, step++) { trLoss[0] = trainStep(nets[0], X, Y, 0, trainR); trainStep(nets[1], X, Y, p, trainR); }
      // dropout train loss is measured with masks on; report the clean (full-net) loss instead
      trLoss[1] = mean(X.map((x, i) => (forward(nets[1], x) - Y[i]) ** 2));
      paintFit();
      return step < TOTAL;
    });
  }
  startTrain();
  fig.caption('Left: the same 1 → 48 → 48 → 1 network on every step, with each hidden unit switched off independently with probability p (10 of the 48 units per layer drawn). Right: 16 noisy training points from a sine curve. Both networks start from the same weights and are trained for 3000 Adam steps. Faint lines = the dropout network with 8 different random masks applied; thick line = the full network, which is what you use at test time.');

  // =================== Figure 2: batch norm ===================
  const fig2 = figure(main, 'Batch norm: keeping activations alive through ten layers', '');
  let act = 'tanh', scale = 1;
  select(fig2.controls, { label: 'Activation', value: 'tanh', options: [['tanh', 'tanh'], ['relu', 'ReLU']], onChange: (v) => { act = v; drawBN(); } });
  slider(fig2.controls, { label: 'Weight init scale s', min: 0.4, max: 3, step: 0.05, value: 1, fmt: (v) => v.toFixed(2), onInput: (v) => { scale = v; drawBN(); } });
  const r2 = stats(fig2.root); fig2.root.insertBefore(r2.root, fig2.body);
  const row2 = el('div', { class: 'fig-row' }); fig2.body.append(row2);
  const bl = el('div'), br = el('div'); row2.append(bl, br);
  const NL = 10, WID = 32, NB = 256;
  const br2 = rng(5);
  const X0 = d3.range(NB).map(() => Float64Array.from({ length: WID }, () => br2.normal()));
  const Wraw = d3.range(NL).map(() => d3.range(WID).map(() => Float64Array.from({ length: WID }, () => br2.normal())));
  function runNet(useBN) {
    const f = act === 'tanh' ? Math.tanh : (z) => (z > 0 ? z : 0);
    let A = X0, out = [];
    const s = scale / Math.sqrt(WID);
    for (let l = 0; l < NL; l++) {
      const Z = A.map((a) => Float64Array.from({ length: WID }, (_, j) => { let z = 0; const w = Wraw[l][j]; for (let i = 0; i < WID; i++) z += w[i] * a[i]; return z * s; }));
      if (useBN) for (let j = 0; j < WID; j++) {
        let m = 0, v = 0; Z.forEach((z) => { m += z[j]; }); m /= NB; Z.forEach((z) => { v += (z[j] - m) ** 2; }); v /= NB;
        const inv = 1 / Math.sqrt(v + 1e-5); Z.forEach((z) => { z[j] = (z[j] - m) * inv; });
      }
      A = Z.map((z) => z.map(f));
      const flat = A.flatMap((a) => Array.from(a));
      const m = mean(flat), sdv = Math.sqrt(mean(flat.map((v) => (v - m) ** 2)));
      const sat = act === 'tanh' ? flat.filter((v) => Math.abs(v) > 0.95).length / flat.length : flat.filter((v) => v === 0).length / flat.length;
      out.push({ flat, sd: sdv, sat });
    }
    return out;
  }
  function ridge(parent, res, title) {
    parent.innerHTML = '';
    const lo = act === 'tanh' ? -1 : 0, hi = act === 'tanh' ? 1 : 3;
    const R = frame(parent, { width: 480, height: 420, x: [lo, hi], y: [0, NL], xLabel: `activation value (${title})`, grid: false, margin: { left: 64, right: 74 } });
    R.g.select('.axis.y').remove();
    const bins = d3.bin().domain([lo, hi]).thresholds(d3.range(lo, hi, (hi - lo) / 40));
    const band = R.h / NL;
    res.forEach((L, l) => {
      const hs = bins(L.flat.map((v) => Math.min(hi - 1e-9, v)));
      const mx = d3.max(hs, (b) => b.length);
      const y0 = (l + 1) * band;
      const area = d3.area().curve(d3.curveStepAfter).x((b) => R.x(b.x0)).y0(y0).y1((b) => y0 - (b.length / mx) * band * 0.92);
      R.g.append('path').datum([...hs, { x0: hi, length: hs[hs.length - 1].length }]).attr('d', area).attr('fill', cat(title.startsWith('with') ? 2 : 0)).attr('fill-opacity', 0.75)
        .attr('stroke', css('--surface')).attr('stroke-width', 0.5)
        .on('mousemove', (e) => tip(e, `layer ${l + 1}<br>sd = ${L.sd.toFixed(3)}<br>${act === 'tanh' ? 'saturated (|a| > 0.95)' : 'exactly zero (dead)'}: ${(100 * L.sat).toFixed(0)}%`)).on('mouseleave', untip);
      R.g.append('text').attr('x', -8).attr('y', y0 - band * 0.3).attr('text-anchor', 'end').attr('font-size', 11).attr('fill', ink2()).text(`layer ${l + 1}`);
      R.g.append('text').attr('x', R.w + 8).attr('y', y0 - band * 0.3).attr('font-size', 11).attr('fill', ink2()).text(`sd ${L.sd < 0.01 ? L.sd.toExponential(0) : L.sd.toFixed(2)}`);
    });
  }
  function drawBN() {
    const a = runNet(false), b = runNet(true);
    ridge(bl, a, 'without batch norm'); ridge(br, b, 'with batch norm');
    const satLbl = act === 'tanh' ? 'saturated at layer 10' : 'dead (zero) at layer 10';
    r2.set({ 'layer-10 sd: without BN': a[NL - 1].sd.toFixed(4), 'layer-10 sd: with BN': b[NL - 1].sd.toFixed(3), [`${satLbl}: without / with BN`]: `${(100 * a[NL - 1].sat).toFixed(0)}% / ${(100 * b[NL - 1].sat).toFixed(0)}%` });
  }
  drawBN();
  fig2.caption(`A batch of ${NB} random inputs is pushed through ${NL} fully connected layers of ${WID} units with random weights W ~ N(0, s²/${WID}); no training, just the forward pass. Each ridge is the histogram of all activations in one layer, and the right-hand number is their standard deviation. With batch norm, each unit's pre-activation is standardised over the batch (mean 0, sd 1) before the nonlinearity. Hover a ridge for details.`);

  // =================== Figure 3: early stopping + weight decay ===================
  const fig3 = figure(main, 'Early stopping and weight decay on a "more genes than samples" problem', '');
  const EP = 300;
  const LP = linProblem({ n: 30, nVal: 300, p: 40, wTrue: [1.5, -1.2, 1, 0.8, -0.8], noise: 1, seed: 21 });
  let patience = 20, lam = 0;
  slider(fig3.controls, { label: 'Patience (epochs)', min: 1, max: 60, step: 1, value: 20, onInput: (v) => { patience = v; paint3(); } });
  slider(fig3.controls, { label: 'Weight decay λ', min: -3, max: 0, step: 0.1, value: -3, fmt: (v) => (v <= -3 ? '0' : (10 ** v).toFixed(3)), onInput: (v) => { lam = v <= -3 ? 0 : 10 ** v; curves = runLin(); paint3(); } });
  const r3 = stats(fig3.root); fig3.root.insertBefore(r3.root, fig3.body);
  const ES = frame(fig3.body, { width: 900, height: 340, x: [0, EP], y: [0, 7], xLabel: 'epoch', yLabel: 'mean squared error' });
  legend(fig3.body, [['training loss', cat(0)], ['validation loss', cat(1)], ['noise floor (σ² = 1)', muted()]]);
  ES.layer('floor').append('line').attr('x1', 0).attr('x2', ES.w).attr('y1', ES.y(1)).attr('y2', ES.y(1)).attr('stroke', muted()).attr('stroke-dasharray', '4 4');
  function runLin() {
    const w = new Float64Array(LP.p), r = rng(3), tr = [mse(w, LP.tr)], va = [mse(w, LP.va)];
    for (let e = 0; e < EP; e++) { sgdEpoch(w, LP.tr, r.shuffle(d3.range(30)), 10, 0.006, lam); tr.push(mse(w, LP.tr)); va.push(mse(w, LP.va)); }
    return { tr, va };
  }
  let curves = runLin();
  function paint3() {
    const { tr, va } = curves;
    let best = 0, stop = EP;
    for (let e = 1; e <= EP; e++) { if (va[e] < va[best]) best = e; else if (e - best >= patience) { stop = e; break; } }
    const trueBest = d3.minIndex(va);
    const ln = d3.line().x((d, i) => ES.x(i)).y((d) => ES.y(Math.min(7, d)));
    ES.layer('c').selectAll('path').data([tr, va]).join('path').attr('fill', 'none').attr('stroke', (d, i) => cat(i)).attr('stroke-width', 2).attr('d', ln);
    ES.layer('shade').selectAll('rect').data([0]).join('rect').attr('x', ES.x(best)).attr('width', Math.max(0, ES.x(stop) - ES.x(best))).attr('y', 0).attr('height', ES.h)
      .attr('fill', css('--accent-soft')).attr('opacity', 0.6);
    ES.layer('shade').lower();
    const marks = [{ e: best, t: 'best so far' }, { e: stop, t: stop < EP ? 'stop' : 'never stopped' }];
    ES.layer('m').selectAll('line').data(marks).join('line').attr('x1', (d) => ES.x(d.e)).attr('x2', (d) => ES.x(d.e)).attr('y1', 0).attr('y2', ES.h).attr('stroke', ink2()).attr('stroke-dasharray', (d, i) => (i ? '4 3' : null));
    ES.layer('m').selectAll('text').data(marks).join('text').attr('x', (d, i) => ES.x(d.e) + (i ? 4 : -4)).attr('y', 12).attr('text-anchor', (d, i) => (i ? 'start' : 'end'))
      .attr('font-size', 12).attr('fill', ink()).text((d) => `${d.t} (epoch ${d.e})`);
    ES.layer('m').selectAll('circle').data([best]).join('circle').attr('cx', ES.x(best)).attr('cy', ES.y(va[best])).attr('r', 5).attr('fill', cat(1)).attr('stroke', css('--surface')).attr('stroke-width', 1.5);
    ES.svg.on('mousemove', (e) => {
      const [mx] = d3.pointer(e, ES.g.node()); const i = Math.round(ES.x.invert(mx));
      if (i < 0 || i > EP) return untip();
      tip(e, `epoch ${i}<br>train ${tr[i].toFixed(3)} · validation ${va[i].toFixed(3)}`);
    }).on('mouseleave', untip);
    r3.set({ 'best validation epoch': best, 'stopped at epoch': stop < EP ? stop : '—', 'val. loss, restored best weights': va[best].toFixed(3), [`val. loss if trained all ${EP} epochs`]: va[EP].toFixed(3), [`train loss at epoch ${EP}`]: tr[EP].toFixed(3) });
    return { best, stop, trueBest };
  }
  const es0 = paint3();
  fig3.caption(`Linear regression with 40 features but only 30 training samples (like a gene panel larger than the cohort); only 5 features truly matter, and the noise variance is 1. Trained by mini-batch SGD from all-zero weights; validation loss is on 300 held-out samples. With λ = 0 the validation loss bottoms out at epoch ${es0.trueBest} (${curves.va[es0.trueBest].toFixed(2)}) and climbs to ${curves.va[EP].toFixed(2)} by epoch ${EP}, while the training loss heads towards zero: the model is memorising noise. The shaded band is the patience window: epochs spent waiting for an improvement that never came.`);

  // =================== Figure 4: learning-rate schedules ===================
  const fig4 = figure(main, 'Learning-rate schedules: warm-up, cosine decay, step decay', '');
  const T = 400;
  let peak = 0.12, wu = 0.1;
  slider(fig4.controls, { label: 'Peak learning rate', min: 0.02, max: 0.3, step: 0.01, value: 0.12, fmt: (v) => v.toFixed(2), onInput: (v) => { peak = v; draw4(); } });
  slider(fig4.controls, { label: 'Warm-up (% of steps)', min: 0, max: 40, step: 1, value: 10, fmt: (v) => v + '%', onInput: (v) => { wu = v / 100; draw4(); } });
  const r4 = stats(fig4.root); fig4.root.insertBefore(r4.root, fig4.body);
  const row4 = el('div', { class: 'fig-row' }); fig4.body.append(row4);
  const ll = el('div'), lr4 = el('div'); row4.append(ll, lr4);
  const SCH = [['constant', 'constant'], ['step', 'step decay (×0.1 at 50% and 75%)'], ['cosine', 'linear warm-up + cosine decay']];
  const lrAt = (k, t) => {
    if (k === 'constant') return peak;
    if (k === 'step') return peak * (t < 0.5 * T ? 1 : t < 0.75 * T ? 0.1 : 0.01);
    const W = Math.round(wu * T);
    if (t < W) return peak * (t + 1) / W;
    return peak * 0.5 * (1 + Math.cos(Math.PI * (t - W) / Math.max(1, T - W)));
  };
  const SP = linProblem({ n: 400, nVal: 1, p: 10, wTrue: [1, -1, 0.5, 2, -0.5, 1, 0, 0.8, -1.5, 0.3], noise: 0.5, seed: 8, colScale: d3.range(10).map((j) => 0.4 + 0.14 * j) });
  function runSched(k) {
    const reps = 8, out = new Float64Array(T + 1);
    for (let s = 0; s < reps; s++) {
      const w = new Float64Array(10), r = rng(100 + s);
      out[0] += mse(w, SP.tr) / reps;
      for (let t = 0; t < T; t++) {
        const idx = [r.int(400), r.int(400), r.int(400), r.int(400)];
        const g = new Float64Array(10);
        idx.forEach((i) => { const x = SP.tr.X[i]; let y = 0; for (let j = 0; j < 10; j++) y += w[j] * x[j]; const e = 2 * (y - SP.tr.Y[i]) / 4; for (let j = 0; j < 10; j++) g[j] += e * x[j]; });
        const a = lrAt(k, t); for (let j = 0; j < 10; j++) w[j] -= a * g[j];
        out[t + 1] += Math.min(1e3, mse(w, SP.tr)) / reps;
      }
    }
    return out;
  }
  const floor = (() => { // least-squares optimum = best achievable training loss
    const p = 10, A = d3.range(p).map(() => new Float64Array(p)), b = new Float64Array(p);
    SP.tr.X.forEach((x, i) => { for (let a = 0; a < p; a++) { b[a] += x[a] * SP.tr.Y[i]; for (let c = 0; c < p; c++) A[a][c] += x[a] * x[c]; } });
    // Gaussian elimination
    const M = A.map((r, i) => [...r, b[i]]);
    for (let c = 0; c < p; c++) { let q = c; for (let r = c + 1; r < p; r++) if (Math.abs(M[r][c]) > Math.abs(M[q][c])) q = r; [M[c], M[q]] = [M[q], M[c]]; for (let r = 0; r < p; r++) if (r !== c) { const f = M[r][c] / M[c][c]; for (let k = c; k <= p; k++) M[r][k] -= f * M[c][k]; } }
    return mse(M.map((r, i) => r[p] / M[i][i]), SP.tr);
  })();
  function draw4() {
    ll.innerHTML = ''; lr4.innerHTML = '';
    const A = frame(ll, { width: 460, height: 320, x: [0, T], y: [0, 0.32], xLabel: 'training step', yLabel: 'learning rate' });
    SCH.forEach(([k], i) => A.layer('l').append('path').datum(d3.range(T)).attr('fill', 'none').attr('stroke', cat(i)).attr('stroke-width', 2)
      .attr('d', d3.line().x((t) => A.x(t)).y((t) => A.y(lrAt(k, t)))));
    const Lc = SCH.map(([k]) => runSched(k));
    const B = frame(lr4, { width: 460, height: 320, x: [0, T], y: [floor * 0.95, 30], yScale: d3.scaleLog(), xLabel: 'training step', yLabel: 'training loss (log scale)' });
    B.g.select('.axis.y').call(d3.axisLeft(B.y).ticks(4, '~g'));
    B.layer('f').append('line').attr('x1', 0).attr('x2', B.w).attr('y1', B.y(floor)).attr('y2', B.y(floor)).attr('stroke', muted()).attr('stroke-dasharray', '4 4');
    Lc.forEach((c, i) => B.layer('l').append('path').datum(Array.from(c)).attr('fill', 'none').attr('stroke', cat(i)).attr('stroke-width', 2)
      .attr('d', d3.line().x((d, t) => B.x(t)).y((d) => B.y(Math.max(floor * 0.95, Math.min(30, d))))));
    const lastAvg = (c) => mean(Array.from(c.slice(T - 20)));
    r4.set(Object.fromEntries(SCH.map(([k], i) => { const d = Math.max(0, lastAvg(Lc[i]) - floor); return [`final loss: ${k}`, !(d < 20) ? 'diverged' : d.toFixed(4) + ' above best']; })));
  }
  legend(fig4.body, [...SCH.map(([, name], i) => [name, cat(i)]), ['best achievable training loss', muted()]]);
  draw4();
  fig4.caption('Left: the learning rate over 400 steps for three schedules. Right: mini-batch SGD (batch of 4) on a 10-feature regression, averaged over 8 runs; the readouts give the mean loss over the last 20 steps minus the best achievable loss (dashed line). With a constant rate, mini-batch noise keeps the weights jittering around the optimum at a height set by the learning rate; decaying the rate lets them settle.');

  section(main, 'What to notice', `<ul>
    <li><b>Dropout:</b> the faint lines (single thinned networks) disagree with each other, but their average sits close to the thick full-network line (readout <i>|mean of 8 thinned − full net|</i>). Using the full network with scaled activations at test time is a cheap approximation to averaging an ensemble of 2⁹⁶ thinned networks. Compare <i>error vs true curve</i>: at p = 0.3 the network without dropout threads through every noisy point (train MSE 0.001) and ends further from the true sine (0.097) than the dropout network (0.031). Raise p and the dropout network fits the training points less tightly, and its thinned versions disagree more.</li>
    <li><b>Batch norm:</b> with tanh and s = 1, the activations without BN shrink layer by layer; at s = 3 they pile up at ±1 (saturated, where tanh's gradient is almost zero). With BN the ridges keep the same shape at every layer whatever s is. Switch to ReLU: without BN the scale drifts with s (s ≈ 1.4, the "He" initialisation, is the value that keeps it level); with BN it does not.</li>
    <li><b>Early stopping:</b> the training curve keeps falling while the validation curve turns back up. A small patience stops close to the best epoch but risks quitting on a noisy blip; a large patience wastes epochs, but you restore the best weights anyway.</li>
    <li><b>Weight decay:</b> raise λ to 0.316 and the validation loss after all 300 epochs drops from 5.87 to 4.90, close to what early stopping gets (4.74) without any decay. Weight decay and early stopping are two routes to the same effect: both keep the weights small. For linear models, stopping gradient descent early behaves much like ridge regression.</li>
    <li><b>Schedules:</b> compare the final losses: the two decaying schedules end much closer to the best achievable loss than the constant one. Raise the peak rate and the constant schedule's plateau rises with it; at 0.20 it diverges while warm-up + cosine still finishes near the optimum, and from 0.25 every schedule blows up. In this small, well-behaved problem, a longer warm-up only costs a little (it wastes steps at a low rate); its benefit shows up in large transformers trained with Adam, where the first updates are the dangerous ones.</li>
  </ul>`);

  steps(main, 'How it works', [
    '<b>Dropout:</b> during training, set each hidden activation to zero with probability p and multiply the survivors by 1/(1−p), so the expected value is unchanged. A new mask is drawn for every example in every step. At test time, switch dropout off and use every unit. Because no unit can rely on any particular partner being present, the network learns redundant, more robust features.',
    '<b>Batch norm:</b> for each unit, subtract the mean and divide by the standard deviation of its pre-activations across the current mini-batch, then apply a learned scale γ and shift β. At test time, use running averages of the mean and variance collected during training. Layer norm does the same across units within one example instead, and is what transformers use.',
    '<b>Early stopping:</b> evaluate validation loss after each epoch, keep a copy of the best weights, and stop once there has been no improvement for <i>patience</i> epochs. Then restore the best copy.',
    '<b>Weight decay:</b> add λ‖w‖² to the loss, so each step also shrinks every weight a little towards zero. With Adam, use the decoupled version (AdamW), which applies the shrinkage directly rather than through the adaptive gradient.',
    '<b>Learning-rate schedule:</b> start with a short linear warm-up from near zero to the peak rate, then decay, most often with a cosine curve. Warm-up protects the first steps, when the weights are random, gradients can be very large and Adam\'s running variance estimates are not yet reliable. Decay lets the weights settle into a minimum instead of bouncing around it.',
  ]);
  math(main, 'dropout (train):   h̃ = h ⊙ m / (1 − p),   m ~ Bernoulli(1 − p)        (test: h̃ = h)\nbatch norm:        ẑ = γ · (z − μ_batch) / √(σ²_batch + ε) + β\nweight decay:      L(w) + λ‖w‖²   ⇒   w ← w − η(∇L + 2λw)\nwarm-up + cosine:  η(t) = η_max · t/W  for t < W;   η_max · ½(1 + cos(π (t − W)/(T − W)))  after');

  callout(main, '', 'In biology', `<ul>
    <li><b>scVI</b> and most single-cell VAEs use batch norm (or layer norm) and dropout inside the encoder and decoder, and stop training early on a held-out set of cells by default.</li>
    <li><b>Geneformer</b> and <b>scGPT</b> are trained with AdamW, weight decay and a warm-up followed by decay, the standard recipe from language models. When you fine-tune them on a small labelled set, you usually need a lower peak rate, warm-up and early stopping to avoid destroying what pretraining learned.</li>
    <li><b>Cellpose</b> and <b>pathology foundation models</b> (vision transformers trained on millions of tiles) rely on the same tricks plus heavy image augmentation, which acts as a regulariser in the same spirit as dropout.</li>
    <li>Omics datasets are the "more features than samples" case of Figure 3: 20,000 genes and tens to hundreds of patients. Regularisation is not optional there; it decides whether any signal survives.</li>
  </ul>`);
  callout(main, 'warn', 'Pitfalls', `<ul>
    <li><b>Forgetting eval mode.</b> Dropout and batch norm behave differently in training and inference. In PyTorch, predicting without <code>model.eval()</code> gives random, worse outputs.</li>
    <li><b>Batch norm with tiny or unbalanced batches.</b> With batch size 2, or a batch made entirely of one sample type, the batch statistics are noisy or biased. Use layer or group norm instead. In omics, where a batch can also mean a sequencing batch, a model can even learn batch identity through BN statistics.</li>
    <li><b>Early stopping on the test set</b> is leakage: the stopping epoch is a hyperparameter, so choose it on a validation split and keep the test set untouched.</li>
    <li><b>Too much regularisation underfits:</b> a very high dropout rate or a very large λ stops the model fitting real signal. On a problem this small even p = 0.7 is tolerated, but large models and small datasets differ. Tune them like any other hyperparameter.</li>
    <li>A learning rate that is too high for the warm-up length still diverges; warm-up delays the problem, it does not remove it.</li>
  </ul>`);
  seeAlso(main, [['concepts', 'regularization', 'Regularization (L1 / L2)'], ['concepts', 'bias-variance', 'Under- vs overfitting'], ['concepts', 'gradient-descent', 'Gradient descent'], ['concepts', 'optimizers', 'Optimizers race'], ['concepts', 'double-descent', 'Double descent']]);
}
