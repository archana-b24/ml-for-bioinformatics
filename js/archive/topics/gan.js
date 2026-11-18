// GANs: a generator and a discriminator trained against each other, live, in 2-D.
import { header, figure, slider, select, button, checkbox, stats, section, steps, callout, math, legend, seeAlso, animate, tip, untip, el } from '../../lib/ui.js';
import { frame, decisionCanvas, cat, css, ink, ink2, muted } from '../../lib/plot.js';
import { rng } from '../../lib/data.js';
const d3 = window.d3;

// ---------- a tiny MLP with manual backprop + Adam (tanh hidden layers, linear output) ----------
function makeMLP(sizes, r) {
  const L = [];
  for (let k = 0; k < sizes.length - 1; k++) {
    const nin = sizes[k], nout = sizes[k + 1], n = nin * nout, s = Math.sqrt(1 / nin);
    const W = new Float64Array(n); for (let i = 0; i < n; i++) W[i] = r.normal(0, s);
    L.push({ nin, nout, W, b: new Float64Array(nout), gW: new Float64Array(n), gb: new Float64Array(nout),
      mW: new Float64Array(n), vW: new Float64Array(n), mb: new Float64Array(nout), vb: new Float64Array(nout) });
  }
  let acts = null, B = 0, t = 0;
  const last = L.length - 1;
  function forward(X, nb) {
    B = nb; acts = [X]; let A = X;
    for (let k = 0; k <= last; k++) {
      const l = L[k], Z = new Float64Array(B * l.nout);
      for (let b = 0; b < B; b++) for (let o = 0; o < l.nout; o++) {
        let s = l.b[o]; const wo = o * l.nin, ai = b * l.nin;
        for (let i = 0; i < l.nin; i++) s += A[ai + i] * l.W[wo + i];
        Z[b * l.nout + o] = k < last ? Math.tanh(s) : s;
      }
      acts.push(Z); A = Z;
    }
    return A;
  }
  // dOut: dLoss/dOutput (B x nout). Accumulates parameter grads, returns dLoss/dInput.
  function backward(dOut) {
    let D = dOut;
    for (let k = last; k >= 0; k--) {
      const l = L[k], Ain = acts[k], Aout = acts[k + 1];
      if (k < last) for (let i = 0; i < D.length; i++) D[i] *= 1 - Aout[i] * Aout[i];
      const dIn = new Float64Array(B * l.nin);
      for (let b = 0; b < B; b++) for (let o = 0; o < l.nout; o++) {
        const g = D[b * l.nout + o]; if (g === 0) continue;
        l.gb[o] += g; const wo = o * l.nin, ai = b * l.nin;
        for (let i = 0; i < l.nin; i++) { l.gW[wo + i] += g * Ain[ai + i]; dIn[ai + i] += g * l.W[wo + i]; }
      }
      D = dIn;
    }
    return D;
  }
  function zero() { L.forEach((l) => { l.gW.fill(0); l.gb.fill(0); }); }
  function adam(lr, b1 = 0.5, b2 = 0.999) {
    t++; const c1 = 1 - b1 ** t, c2 = 1 - b2 ** t;
    const upd = (P, G, M, V) => { for (let i = 0; i < P.length; i++) { M[i] = b1 * M[i] + (1 - b1) * G[i]; V[i] = b2 * V[i] + (1 - b2) * G[i] * G[i]; P[i] -= lr * (M[i] / c1) / (Math.sqrt(V[i] / c2) + 1e-8); } };
    L.forEach((l) => { upd(l.W, l.gW, l.mW, l.vW); upd(l.b, l.gb, l.mb, l.vb); });
    zero();
  }
  // fast single-point evaluation for 2-D inputs (used for heatmaps)
  const h1 = new Float64Array(64), h2 = new Float64Array(64);
  function at(x, y) {
    let A = h1; const l0 = L[0];
    for (let o = 0; o < l0.nout; o++) { const s = l0.b[o] + x * l0.W[o * 2] + y * l0.W[o * 2 + 1]; A[o] = last === 0 ? s : Math.tanh(s); }
    let n = l0.nout, cur = A, nxt = h2;
    for (let k = 1; k <= last; k++) {
      const l = L[k];
      for (let o = 0; o < l.nout; o++) { let s = l.b[o]; for (let i = 0; i < n; i++) s += cur[i] * l.W[o * n + i]; nxt[o] = k < last ? Math.tanh(s) : s; }
      n = l.nout; [cur, nxt] = [nxt, cur];
    }
    return cur;
  }
  return { forward, backward, zero, adam, at };
}

const sig = (z) => 1 / (1 + Math.exp(-z));
const softplus = (z) => (z > 30 ? z : Math.log1p(Math.exp(z)));

const DATASETS = {
  ring8: { name: '8 clusters on a ring', modes: d3.range(8).map((k) => [Math.cos(k * Math.PI / 4), Math.sin(k * Math.PI / 4)]), sd: 0.07 },
  grid4: { name: '4 clusters (square)', modes: [[-0.7, -0.7], [0.7, -0.7], [-0.7, 0.7], [0.7, 0.7]], sd: 0.1 },
};
const SCHEDULES = {
  balanced: { name: '1 D step : 1 G step (balanced)', nD: 1, nG: 1 },
  weakD: { name: '1 D step : 15 G steps (weak critic)', nD: 1, nG: 15 },
};

export default async function render(main) {
  header(main, 'GANs (generative adversarial networks)',
    'A <b>generator</b> turns random noise into fake samples; a <b>discriminator</b> tries to tell fake from real. They are trained against each other: every time the critic finds a tell-tale difference, the generator moves its samples to erase it. When neither can improve, the fakes look like the data.');

  // ================= Figure 1: live 2-D GAN =================
  const fig = figure(main, 'A GAN trained live in your browser',
    'Left: <b>real</b> data (orange) and the generator\'s <b>fake</b> samples (blue, the same 300 noise vectors z pushed through G each frame). The background is the discriminator: orange where D says "real", blue where it says "fake", stronger colour = more confident. Short ink lines on fakes show where the generator\'s gradient is pushing them. Right, top: the two losses; dashed lines mark their values at equilibrium (D = ½ everywhere). Right, bottom: what the losses do not tell you, namely the share of fakes that land on a real cluster and the share of the 8 clusters that receive at least 2.5% of the fakes.');
  let dsKey = 'ring8', schedKey = 'balanced', seed = 3, showArrows = true, showGrid = false, stop = null;
  let G, D, r, it = 0, nDsteps = 0, nGsteps = 0, hist = [], qhist = [], phase = 0;
  const LR = 0.002, BATCH = 128, NZ = 2;

  select(fig.controls, { label: 'Real data', value: dsKey, options: Object.entries(DATASETS).map(([k, v]) => [k, v.name]), onChange: (k) => { dsKey = k; reset(); } });
  select(fig.controls, { label: 'Schedule', value: schedKey, options: Object.entries(SCHEDULES).map(([k, v]) => [k, v.name]), onChange: (k) => { schedKey = k; reset(); } });
  slider(fig.controls, { label: 'Seed', min: 1, max: 6, step: 1, value: seed, onInput: (v) => { seed = v; reset(); } });
  const runBtn = button(fig.controls, 'Train', () => toggle(), true);
  button(fig.controls, 'One phase', () => { pause(); onePhase(); paint(); });
  button(fig.controls, 'Reset', () => reset());
  checkbox(fig.controls, { label: 'G gradient arrows', checked: true, onChange: (c) => { showArrows = c; paint(); } });
  checkbox(fig.controls, { label: 'Warped noise grid', checked: false, onChange: (c) => { showGrid = c; paint(); } });

  // the training loop as chips, highlighted per phase
  const loop = el('div', { class: 'controls', style: 'gap:6px;font-size:13px' });
  const chipStyle = 'padding:3px 9px;border-radius:999px;border:1px solid var(--border);background:var(--surface-2);color:var(--ink-2)';
  const chips = ['① sample real x and fake G(z)', '② update D to label real 1, fake 0 (G frozen)', '③ sample fresh z', '④ update G so D(G(z)) → 1 (D frozen)'].map((t) => el('span', { style: chipStyle }, t));
  loop.append(el('b', { style: 'font-size:13px;margin-right:4px' }, 'Loop:'), ...chips);
  fig.body.append(loop);
  const readout = stats(fig.root);

  const row = el('div', { class: 'fig-row' }); fig.body.append(row);
  const left = el('div'), right = el('div'); row.append(left, right);
  legend(left, [['real data', cat(1)], ['fake G(z)', cat(0)]]);
  legend(right, [['discriminator loss', cat(2)], ['generator loss', cat(6)], ['fakes on a cluster', cat(0)], ['clusters covered', cat(3)]]);

  const DOM = [-1.6, 1.6];
  const F = frame(left, { width: 480, height: 440, x: DOM, y: DOM, xLabel: 'feature 1', yLabel: 'feature 2' });
  const surf = decisionCanvas(F, () => 0.5, { res: 5, alpha: 0.42 });
  ['grid', 'real', 'arrows', 'fake'].forEach((n) => F.layer(n));
  const LF = frame(right, { width: 440, height: 200, x: [0, 200], y: [0, 2.5], yLabel: 'loss', yTicks: 4 });
  const QF = frame(right, { width: 440, height: 230, x: [0, 200], y: [0, 100], xLabel: 'training iteration', yLabel: 'sample quality (%)', yTicks: 4 });
  [Math.log(4), Math.log(2)].forEach((v, i) => LF.layer('ref').append('line').attr('x1', 0).attr('x2', LF.w).attr('y1', LF.y(v)).attr('y2', LF.y(v))
    .attr('stroke', muted()).attr('stroke-dasharray', '4 4').attr('stroke-width', 1).attr('class', 'ref' + i));
  const halo = (t) => t.attr('stroke', css('--surface')).attr('stroke-width', 4).attr('paint-order', 'stroke');
  halo(LF.layer('ref').append('text').attr('class', 'reft0').attr('x', LF.w - 4).attr('text-anchor', 'end').attr('font-size', 11).attr('fill', muted()).text('2 ln 2 ≈ 1.39 (D at equilibrium)'));
  halo(LF.layer('ref').append('text').attr('class', 'reft1').attr('x', LF.w - 4).attr('text-anchor', 'end').attr('font-size', 11).attr('fill', muted()).text('ln 2 ≈ 0.69 (G at equilibrium)'));

  let real = [], zFix, zGrid;
  function sampleReal(n, rr) {
    const ds = DATASETS[dsKey], X = new Float64Array(n * 2);
    for (let i = 0; i < n; i++) { const m = ds.modes[rr.int(ds.modes.length)]; X[2 * i] = rr.normal(m[0], ds.sd); X[2 * i + 1] = rr.normal(m[1], ds.sd); }
    return X;
  }
  function sampleZ(n, rr) { const Z = new Float64Array(n * NZ); for (let i = 0; i < Z.length; i++) Z[i] = rr.normal(); return Z; }

  function reset() {
    pause();
    r = rng(1000 + seed);
    G = makeMLP([NZ, 32, 32, 2], rng(seed)); D = makeMLP([2, 32, 32, 1], rng(50 + seed));
    it = 0; nDsteps = 0; nGsteps = 0; hist = []; qhist = []; phase = 0;
    const rd = rng(7); const X = sampleReal(300, rd); real = d3.range(300).map((i) => ({ x: X[2 * i], y: X[2 * i + 1] }));
    zFix = sampleZ(300, rng(11));
    // noise grid lines in z space (−2.5..2.5)
    zGrid = [];
    const ticks = d3.range(-2.5, 2.51, 0.5), fine = d3.range(-2.5, 2.501, 0.1);
    ticks.forEach((a) => { zGrid.push(fine.map((b) => [a, b])); zGrid.push(fine.map((b) => [b, a])); });
    paint();
  }

  // --- one discriminator update (G frozen) ---
  function dStep() {
    const Xr = sampleReal(BATCH, r), Xf = G.forward(sampleZ(BATCH, r), BATCH);
    const X = new Float64Array(4 * BATCH); X.set(Xr, 0); X.set(Xf, 2 * BATCH);
    const out = D.forward(X, 2 * BATCH), dOut = new Float64Array(2 * BATCH);
    let loss = 0;
    for (let i = 0; i < 2 * BATCH; i++) { const y = i < BATCH ? 1 : 0; loss += y ? softplus(-out[i]) : softplus(out[i]); dOut[i] = (sig(out[i]) - y) / BATCH; }
    D.backward(dOut); D.adam(LR); G.zero(); nDsteps++;
    return loss / BATCH; // = mean over real + mean over fake
  }
  // --- one generator update (D frozen): non-saturating loss −log D(G(z)) ---
  function gStep() {
    const Xf = G.forward(sampleZ(BATCH, r), BATCH);
    const out = D.forward(Xf, BATCH), dOut = new Float64Array(BATCH);
    let loss = 0;
    for (let i = 0; i < BATCH; i++) { loss += softplus(-out[i]); dOut[i] = (sig(out[i]) - 1) / BATCH; }
    const dX = D.backward(dOut); D.zero();
    G.backward(dX); G.adam(LR); nGsteps++;
    return loss / BATCH;
  }
  function iteration() {
    const s = SCHEDULES[schedKey]; let dl = 0, gl = 0;
    for (let k = 0; k < s.nD; k++) dl += dStep();
    for (let k = 0; k < s.nG; k++) gl += gStep();
    it++; hist.push({ it, d: dl / s.nD, g: gl / s.nG });
  }
  function onePhase() {
    // phases: 0 = D update, 1 = G update (chips ①② then ③④)
    if (phase === 0) { const dl = dStep(); hist.push({ it: it + 0.5, d: dl, g: hist.length ? hist[hist.length - 1].g : NaN }); phase = 1; }
    else { const s = SCHEDULES[schedKey]; let gl = 0; for (let k = 0; k < s.nG; k++) gl += gStep(); it++; hist.push({ it, d: hist[hist.length - 1].d, g: gl / s.nG }); phase = 0; }
    chips.forEach((c, i) => { const on = phase === 1 ? i < 2 : i >= 2; c.style.background = on ? 'var(--accent-soft)' : 'var(--surface-2)'; c.style.color = on ? 'var(--ink)' : 'var(--ink-2)'; });
  }
  function pause() { if (stop) { stop(); stop = null; } runBtn.textContent = 'Train'; chips.forEach((c) => { c.style.background = 'var(--surface-2)'; c.style.color = 'var(--ink-2)'; }); }
  function toggle() {
    if (stop) { pause(); return; }
    runBtn.textContent = 'Pause';
    chips.forEach((c) => { c.style.background = 'var(--accent-soft)'; c.style.color = 'var(--ink)'; });
    const per = schedKey === 'weakD' ? 2 : 6;
    stop = animate(() => { for (let k = 0; k < per; k++) iteration(); paint(); if (it >= 3000) { pause(); return false; } return true; });
  }

  function coverage(pts) {
    const ds = DATASETS[dsKey], cnt = new Array(ds.modes.length).fill(0); let on = 0;
    pts.forEach((p) => { let best = 1e9, bi = -1; ds.modes.forEach((m, i) => { const d = Math.hypot(p.x - m[0], p.y - m[1]); if (d < best) { best = d; bi = i; } }); if (best < 3 * ds.sd) { cnt[bi]++; on++; } });
    return { covered: cnt.filter((c) => c >= 0.025 * pts.length).length, total: ds.modes.length, onFrac: on / pts.length };
  }

  function paint() {
    surf.draw((x, y) => sig(D.at(x, y)[0]));
    const fakeArr = G.forward(zFix, 300);
    const fake = d3.range(300).map((i) => ({ x: fakeArr[2 * i], y: fakeArr[2 * i + 1], i }));
    const cl = (v) => Math.max(DOM[0], Math.min(DOM[1], v));
    F.layer('real').selectAll('circle').data(real).join('circle').attr('cx', (d) => F.x(d.x)).attr('cy', (d) => F.y(d.y)).attr('r', 3.2)
      .attr('fill', cat(1)).attr('stroke', css('--surface')).attr('stroke-width', 0.8).attr('opacity', 0.85);
    F.layer('fake').selectAll('circle').data(fake).join('circle').attr('cx', (d) => F.x(cl(d.x))).attr('cy', (d) => F.y(cl(d.y))).attr('r', 4)
      .attr('fill', cat(0)).attr('stroke', css('--surface')).attr('stroke-width', 1)
      .on('mousemove', (e, d) => { const p = sig(D.at(d.x, d.y)[0]); tip(e, `fake sample from z = (${zFix[2 * d.i].toFixed(2)}, ${zFix[2 * d.i + 1].toFixed(2)})<br>D(x) = P(real) = <b>${p.toFixed(2)}</b>`); })
      .on('mouseleave', untip);
    // generator push direction: −∂L_G/∂x = (1−D)·∇logit
    let arrows = [];
    if (showArrows) {
      const sub = fake.filter((d) => d.i % 3 === 0), X = new Float64Array(sub.length * 2);
      sub.forEach((d, k) => { X[2 * k] = d.x; X[2 * k + 1] = d.y; });
      const out = D.forward(X, sub.length), dOut = new Float64Array(sub.length);
      for (let k = 0; k < sub.length; k++) dOut[k] = sig(out[k]) - 1;
      const dX = D.backward(dOut); D.zero();
      arrows = sub.map((d, k) => { let gx = -dX[2 * k], gy = -dX[2 * k + 1]; const n = Math.hypot(gx, gy), m = Math.min(0.22, n * 0.25) / (n || 1); return { x: d.x, y: d.y, x2: d.x + gx * m, y2: d.y + gy * m }; });
    }
    F.layer('arrows').selectAll('line').data(arrows).join('line').attr('x1', (d) => F.x(cl(d.x))).attr('y1', (d) => F.y(cl(d.y)))
      .attr('x2', (d) => F.x(cl(d.x2))).attr('y2', (d) => F.y(cl(d.y2))).attr('stroke', ink()).attr('stroke-width', 1.4).attr('stroke-linecap', 'round');
    // warped grid G(z-grid)
    let lines = [];
    if (showGrid) {
      lines = zGrid.map((ln) => { const Z = new Float64Array(ln.length * 2); ln.forEach((p, k) => { Z[2 * k] = p[0]; Z[2 * k + 1] = p[1]; }); const O = G.forward(Z, ln.length); return ln.map((_, k) => [F.x(cl(O[2 * k])), F.y(cl(O[2 * k + 1]))]); });
    }
    F.layer('grid').selectAll('path').data(lines).join('path').attr('d', d3.line()).attr('fill', 'none').attr('stroke', ink2()).attr('stroke-width', 0.7).attr('opacity', 0.55);

    // loss curves
    const xmax = Math.max(200, it);
    LF.x.domain([0, xmax]); LF.g.select('.axis.x').call(d3.axisBottom(LF.x).ticks(5));
    QF.x.domain([0, xmax]); QF.g.select('.axis.x').call(d3.axisBottom(QF.x).ticks(5));
    const step = Math.max(1, Math.floor(hist.length / 400)), hs = hist.filter((_, i) => i % step === 0 || i === hist.length - 1);
    // light smoothing so the trend is readable
    const smooth = (key) => { let m = null; return hs.filter((h) => isFinite(h[key])).map((h) => { m = m == null ? h[key] : 0.85 * m + 0.15 * h[key]; return { x: h.it, y: Math.min(2.5, m) }; }); };
    const series = [[smooth('d'), cat(2)], [smooth('g'), cat(6)]];
    LF.layer('c').selectAll('path').data(series).join('path').attr('fill', 'none').attr('stroke', (d) => d[1]).attr('stroke-width', 2)
      .attr('d', (d) => d3.line().x((p) => LF.x(p.x)).y((p) => LF.y(p.y))(d[0]));
    LF.layer('ref').select('.reft0').attr('y', LF.y(Math.log(4)) - 5);
    LF.layer('ref').select('.reft1').attr('y', LF.y(Math.log(2)) + 14);

    const cov = coverage(fake), last = hist[hist.length - 1];
    if (!qhist.length || qhist[qhist.length - 1].it !== it) qhist.push({ it, on: 100 * cov.onFrac, cov: 100 * cov.covered / cov.total });
    const qs = [[qhist.map((q) => ({ x: q.it, y: q.on })), cat(0)], [qhist.map((q) => ({ x: q.it, y: q.cov })), cat(3)]];
    QF.layer('c').selectAll('path').data(qs).join('path').attr('fill', 'none').attr('stroke', (d) => d[1]).attr('stroke-width', 2)
      .attr('d', (d) => d3.line().x((p) => QF.x(p.x)).y((p) => QF.y(p.y))(d[0]));
    let dr = 0; real.forEach((p) => { dr += sig(D.at(p.x, p.y)[0]); });
    let df = 0; fake.forEach((p) => { df += sig(D.at(p.x, p.y)[0]); });
    readout.set({
      iteration: it, 'D steps / G steps': `${nDsteps} / ${nGsteps}`,
      'mean D(real)': (dr / real.length).toFixed(2), 'mean D(fake)': (df / fake.length).toFixed(2),
      'clusters covered': `${cov.covered} / ${cov.total}`, 'fakes on a cluster': (100 * cov.onFrac).toFixed(0) + '%',
      'D loss': last ? last.d.toFixed(2) : '–',
    });
  }
  reset();

  section(main, 'What to notice', `<ul>
    <li>Press <b>Train</b> (seed 3, balanced). The fakes start as a small blob in the middle, then spread out and split towards the clusters. With this seed, the bottom chart climbs from 2 of 8 clusters covered at about iteration 100 to all 8 at around iteration 800, and by iteration 1,400 over 90% of fakes sit on a cluster. The curves are jumpy: quality in a GAN wobbles from step to step.</li>
    <li>Now look at the losses: they sit on the dashed equilibrium lines almost from the start and barely move. In a GAN <b>the loss is not a progress bar</b>; each network's loss is measured against an opponent that keeps changing. You have to look at the samples.</li>
    <li>The background never becomes strongly coloured, because the discriminator is kept close to guessing (D ≈ 0.5) by a generator that keeps up. The faint orange wedges are the gaps between fake clumps where D has spotted that real points still outnumber fakes; the arrows show fakes being pulled into them.</li>
    <li>Try other seeds with the balanced schedule. With seeds 1 and 5 the quality curves stay lower and jumpier for much longer, with many fakes stranded on bridges between clusters. Sensitivity to the seed is the everyday frustration of GAN training.</li>
    <li>Switch to <b>1 D step : 15 G steps</b>. Against a critic that hardly changes, the generator's best move is to put <i>every</i> sample at the one place D currently likes most. The fakes travel as a single clump and hop from cluster to cluster (by iteration ~380 with seed 3, almost all of them sit on <b>one</b> cluster). This is <b>mode collapse</b>. Meanwhile D(fake) ≈ D(real) ≈ 0.5 and both losses look perfect.</li>
    <li>Tick <b>Warped noise grid</b> to see the generator as a function: it takes the square grid of noise z and bends, stretches and folds it onto the data. A continuous map cannot tear one sheet into 8 separate pieces, which is why thin "bridges" of fakes between clusters are so common.</li>
    <li>Pause and use <b>One phase</b> to walk the loop: a D update changes the background, and a G update moves the blue points.</li>
  </ul>`);

  // ================= Figure 2: the optimal discriminator in 1-D =================
  const fig2 = figure(main, 'What the discriminator is really estimating',
    'For a fixed generator, the best possible discriminator is D*(x) = p_data(x) / (p_data(x) + p_G(x)): the probability that a sample at x came from the data. Move the generator\'s Gaussian and watch D* flatten to ½ only when the two densities match.');
  let mu = -1.8, sg = 0.6;
  slider(fig2.controls, { label: 'generator mean', min: -3, max: 3, step: 0.05, value: mu, fmt: (v) => v.toFixed(2), onInput: (v) => { mu = v; draw2(); } });
  slider(fig2.controls, { label: 'generator spread', min: 0.15, max: 2, step: 0.05, value: sg, fmt: (v) => v.toFixed(2), onInput: (v) => { sg = v; draw2(); } });
  const ro2 = stats(fig2.root);
  legend(fig2.body, [['p_data (real)', cat(1)], ['p_G (fake)', cat(0)], ['optimal D*(x)', ink()]]);
  const F2 = frame(fig2.body, { width: 900, height: 280, x: [-4, 4], y: [0, 1.05], xLabel: 'x', yLabel: 'density / probability' });
  const npdf = (x, m, s) => Math.exp(-0.5 * ((x - m) / s) ** 2) / (s * Math.sqrt(2 * Math.PI));
  const pdata = (x) => 0.5 * npdf(x, -0.9, 0.45) + 0.5 * npdf(x, 1.1, 0.45);
  function draw2() {
    const xs = d3.range(-4, 4.001, 0.02);
    const pts = xs.map((x) => { const a = pdata(x), b = npdf(x, mu, sg); return { x, a, b, d: a + b > 1e-12 ? a / (a + b) : 0.5 }; });
    const area = (k) => d3.area().x((p) => F2.x(p.x)).y0(F2.y(0)).y1((p) => F2.y(Math.min(1.05, p[k])));
    F2.layer('a').selectAll('path').data([['a', cat(1)], ['b', cat(0)]]).join('path').attr('d', (d) => area(d[0])(pts)).attr('fill', (d) => d[1]).attr('fill-opacity', 0.25).attr('stroke', (d) => d[1]).attr('stroke-width', 2);
    F2.layer('d').selectAll('path').data([pts]).join('path').attr('d', d3.line().x((p) => F2.x(p.x)).y((p) => F2.y(p.d))).attr('fill', 'none').attr('stroke', ink()).attr('stroke-width', 2).attr('stroke-dasharray', '6 3');
    // Jensen–Shannon divergence (nats) by numerical integration
    let js = 0; const dx = 0.02;
    pts.forEach((p) => { const m = 0.5 * (p.a + p.b); if (p.a > 1e-12) js += 0.5 * p.a * Math.log(p.a / m) * dx; if (p.b > 1e-12) js += 0.5 * p.b * Math.log(p.b / m) * dx; });
    ro2.set({ 'JS divergence': js.toFixed(3) + ' nats', 'best possible D loss': (Math.log(4) - 2 * js).toFixed(3), 'when p_G = p_data': '2 ln 2 = 1.386' });
  }
  draw2();

  steps(main, 'How it works', [
    'Pick a simple noise distribution, here a 2-D standard normal z. The <b>generator</b> G is a neural network that maps each z to a sample G(z) in data space.',
    'The <b>discriminator</b> D is a classifier that outputs P(real) for any point. Train it for a step on a batch of real samples (label 1) and fakes (label 0), with G held fixed.',
    'Then train G for a step with D held fixed: backpropagate <i>through</i> D to find how to move each fake so that D calls it more real (the arrows), and pass that gradient on into G\'s weights.',
    'Alternate. If both reach the ideal, p_G equals p_data and D outputs ½ everywhere (figure 2). In practice the game oscillates, and the balance between the two players matters a lot.',
    'Once trained, throw D away. Sampling is a single forward pass of G on fresh noise, which is very fast compared with diffusion models.',
  ]);
  math(main, 'min_G max_D  E_x~data[ log D(x) ] + E_z[ log(1 − D(G(z))) ]\nD loss = −log D(x_real) − log(1 − D(G(z)))        G loss (non-saturating) = −log D(G(z))\noptimal D*(x) = p_data / (p_data + p_G)   ⇒   best D loss = 2 ln 2 − 2·JS(p_data ‖ p_G)');

  callout(main, '', 'In biology', `<ul>
    <li><b>Simulating realistic single-cell profiles:</b> scGAN / cscGAN (Marouf et al., Nat Commun 2020) generated synthetic scRNA-seq cells, including rare cell types to augment small classes.</li>
    <li><b>Batch correction and domain translation:</b> adversarial ideas survive inside other models. Several integration methods add a discriminator that tries to predict the batch from the latent space, and the encoder is trained to fool it, so the embedding stops carrying batch identity.</li>
    <li><b>Imaging:</b> GANs were used for virtual staining (e.g. predicting H&E or IHC appearance from unstained or autofluorescence images) and for super-resolution microscopy. In pathology and microscopy, diffusion models have largely replaced them for generation, because they train stably and do not mode-collapse.</li>
    <li>Mode collapse is a real risk for biology: a generator that only produces the dominant cell states will silently drop the rare population you care about. Always check coverage (e.g. do generated cells populate every cluster?), not just how "real" individual samples look.</li></ul>`);
  callout(main, 'warn', 'Pitfalls', `<ul>
    <li><b>Mode collapse</b>: realistic samples with too little variety. The loss values do not reveal it, so look at coverage.</li>
    <li><b>Unstable training</b>: there is no single loss going down. Fixes include Wasserstein losses with gradient penalties, spectral normalisation and two-timescale learning rates.</li>
    <li><b>An overconfident discriminator</b> gives the generator vanishing gradients (the original log(1−D) loss saturates). That is why the non-saturating −log D loss is used here.</li>
    <li>There is no likelihood, so it is hard to compare GANs or to detect overfitting. Sample-based metrics (e.g. FID) depend on a reference feature extractor.</li></ul>`);
  seeAlso(main, [['concepts', 'diffusion', 'Diffusion models'], ['concepts', 'vae', 'Variational autoencoders'], ['concepts', 'neural-network', 'Neural network playground'], ['concepts', 'loss-functions', 'Loss functions'], ['omics', 'sc-integration', 'Batch integration']]);
}
