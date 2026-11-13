// Diffusion models: forward noising of a 2-D point cloud, and reverse denoising with the exact score of the data density.
import { header, figure, slider, select, button, checkbox, stats, section, steps, callout, math, legend, seeAlso, animate, tip, untip, el } from '../lib/ui.js';
import { frame, cat, css, ink, ink2, muted } from '../lib/plot.js';
import { rng } from '../lib/data.js';
const d3 = window.d3;

const T = 100;          // number of diffusion steps
const SIG0 = 0.05;      // width of each Gaussian bump in the data density

// ---- data densities: Gaussian mixtures whose centres trace a shape (radius ≤ ~1.6) ----
const SHAPES = {
  spiral: { name: 'Spiral', centres: d3.range(160).map((k) => { const th = 0.7 * Math.PI + (k / 159) * 3 * Math.PI, rad = 1.6 * th / (3.7 * Math.PI); return [rad * Math.cos(th), rad * Math.sin(th)]; }) },
  ring: { name: 'Ring', centres: d3.range(120).map((k) => { const th = 2 * Math.PI * k / 120; return [1.2 * Math.cos(th), 1.2 * Math.sin(th)]; }) },
  branches: { name: 'Branching trajectory (3 fates)', centres: (() => {
    const c = []; for (let k = 0; k < 50; k++) c.push([-1.5 + 1.5 * k / 49, -0.2 + 0.2 * k / 49]); // stem
    [[1.3, 1.2], [1.6, 0.0], [1.1, -1.3]].forEach(([ex, ey]) => { for (let k = 1; k <= 40; k++) { const u = k / 40; c.push([u * ex, (1 - u) * 0 + u * ey]); } });
    return c; })() },
};

// ---- noise schedules: return alphaBar[0..T] (alphaBar[0] = 1) and beta[1..T] ----
function schedule(kind) {
  const ab = new Float64Array(T + 1), beta = new Float64Array(T + 1); ab[0] = 1;
  if (kind === 'linear') {
    for (let t = 1; t <= T; t++) { beta[t] = 1e-3 + (0.2 - 1e-3) * (t - 1) / (T - 1); ab[t] = ab[t - 1] * (1 - beta[t]); }
  } else {
    const s = 0.008, f = (t) => Math.cos(((t / T + s) / (1 + s)) * Math.PI / 2) ** 2;
    for (let t = 1; t <= T; t++) { beta[t] = Math.min(0.999, 1 - f(t) / f(t - 1)); ab[t] = ab[t - 1] * (1 - beta[t]); }
  }
  return { ab, beta };
}

// Exact posterior quantities of the noised mixture q_t(x) = mean_i N(x; √ā μ_i, (ā σ0² + 1 − ā) I).
// Returns the predicted clean point x̂0 = E[x0 | x_t] and the score ∇ log q_t(x).
function denoise(x, y, ab, C) {
  const sa = Math.sqrt(ab), s2 = ab * SIG0 * SIG0 + (1 - ab);
  let mx = -Infinity; const n = C.length, lw = new Float64Array(n);
  for (let i = 0; i < n; i++) { const dx = x - sa * C[i][0], dy = y - sa * C[i][1]; lw[i] = -(dx * dx + dy * dy) / (2 * s2); if (lw[i] > mx) mx = lw[i]; }
  let Z = 0, ex = 0, ey = 0;
  for (let i = 0; i < n; i++) { const w = Math.exp(lw[i] - mx); Z += w; ex += w * C[i][0]; ey += w * C[i][1]; }
  ex /= Z; ey /= Z;                                  // posterior-weighted centre
  const k = sa * SIG0 * SIG0 / s2;                   // shrink towards x within a bump
  const x0 = ex + k * (x - sa * ex), y0 = ey + k * (y - sa * ey);
  const sx = (sa * ex - x) / s2, sy = (sa * ey - y) / s2;
  return [x0, y0, sx, sy];
}

export default async function render(main) {
  header(main, 'Diffusion models',
    'Destroying a picture with noise is easy; a diffusion model learns to <b>undo noise one small step at a time</b>. Start from pure noise, apply the learned "denoise a little" step a hundred times, and a new sample of the data appears.');

  // ================= Figure 1: forward process =================
  const fig = figure(main, 'Forward process: drown the data in noise',
    'Each point is mixed with its own fixed Gaussian noise: x_t = √ā_t · x₀ + √(1 − ā_t) · ε. Points are coloured by their position along the shape so you can follow them. Right: how much signal (√ā) and noise (√(1−ā)) is left at each step, for the chosen schedule (solid) and the other one (faint).');
  let shapeKey = 'spiral', schedKey = 'cosine', t1 = 0;
  select(fig.controls, { label: 'Data shape', value: shapeKey, options: Object.entries(SHAPES).map(([k, v]) => [k, v.name]), onChange: (k) => { shapeKey = k; makeData(); drawF(); resetR(); } });
  select(fig.controls, { label: 'Noise schedule', value: schedKey, options: [['cosine', 'cosine (Nichol & Dhariwal)'], ['linear', 'linear β (original DDPM)']], onChange: (k) => { schedKey = k; drawF(); resetR(); } });
  const tSl = slider(fig.controls, { label: 'timestep t', min: 0, max: T, step: 1, value: 0, onInput: (v) => { t1 = v; drawF(); } });
  const ro1 = stats(fig.root);
  const row1 = el('div', { class: 'fig-row' }); fig.body.append(row1);
  const l1 = el('div'), r1 = el('div'); row1.append(l1, r1);
  const DOM = [-3, 3];
  const F1 = frame(l1, { width: 460, height: 440, x: DOM, y: DOM, xLabel: 'x₁', yLabel: 'x₂' });
  legend(r1, [['signal √ā_t', cat(0)], ['noise √(1 − ā_t)', cat(1)]]);
  const S1 = frame(r1, { width: 460, height: 410, x: [0, T], y: [0, 1.02], xLabel: 'timestep t', yLabel: 'coefficient' });
  const posColor = (u) => d3.interpolateRgb(cat(0), cat(1))(u);

  let X0 = [], EPS = [];
  function makeData() {
    const r = rng(5), C = SHAPES[shapeKey].centres;
    X0 = d3.range(400).map((i) => { const k = r.int(C.length); return { x: r.normal(C[k][0], SIG0), y: r.normal(C[k][1], SIG0), u: k / (C.length - 1) }; });
    const re = rng(9); EPS = X0.map(() => [re.normal(), re.normal()]);
  }
  function drawF() {
    const sc = schedule(schedKey), ab = sc.ab[t1], a = Math.sqrt(ab), b = Math.sqrt(1 - ab);
    const pts = X0.map((p, i) => ({ x: a * p.x + b * EPS[i][0], y: a * p.y + b * EPS[i][1], u: p.u }));
    const cl = (v) => Math.max(DOM[0], Math.min(DOM[1], v));
    F1.layer('p').selectAll('circle').data(pts).join('circle').attr('cx', (d) => F1.x(cl(d.x))).attr('cy', (d) => F1.y(cl(d.y))).attr('r', 3.6)
      .attr('fill', (d) => posColor(d.u)).attr('stroke', css('--surface')).attr('stroke-width', 0.8);
    const other = schedule(schedKey === 'cosine' ? 'linear' : 'cosine');
    const curves = [];
    [[other, 0.25], [sc, 1]].forEach(([s, op]) => {
      curves.push({ pts: d3.range(T + 1).map((t) => [t, Math.sqrt(s.ab[t])]), c: cat(0), op });
      curves.push({ pts: d3.range(T + 1).map((t) => [t, Math.sqrt(1 - s.ab[t])]), c: cat(1), op });
    });
    S1.layer('c').selectAll('path').data(curves).join('path').attr('fill', 'none').attr('stroke', (d) => d.c).attr('stroke-opacity', (d) => d.op).attr('stroke-width', 2)
      .attr('d', (d) => d3.line().x((p) => S1.x(p[0])).y((p) => S1.y(p[1]))(d.pts));
    S1.layer('m').selectAll('line').data([t1]).join('line').attr('x1', (d) => S1.x(d)).attr('x2', (d) => S1.x(d)).attr('y1', 0).attr('y2', S1.h).attr('stroke', ink()).attr('stroke-width', 1.2).attr('stroke-dasharray', '3 3');
    S1.layer('m').selectAll('circle').data([[t1, a, cat(0)], [t1, b, cat(1)]]).join('circle').attr('cx', (d) => S1.x(d[0])).attr('cy', (d) => S1.y(d[1])).attr('r', 5).attr('fill', (d) => d[2]).attr('stroke', css('--surface')).attr('stroke-width', 1.5);
    // first t where noise outweighs signal
    let half = 0; for (let t = 0; t <= T; t++) if (sc.ab[t] < 0.5) { half = t; break; }
    ro1.set({ t: t1, 'ā_t (signal variance kept)': ab.toFixed(3), 'signal-to-noise ā/(1−ā)': ab > 0.9999 ? '∞' : (ab / (1 - ab)).toExponential(1), 'noise overtakes signal at t': half, 'β_t this step': t1 ? sc.beta[t1].toFixed(4) : '–' });
  }
  makeData(); drawF();

  // ================= Figure 2: reverse process =================
  const fig2 = figure(main, 'Reverse process: walk noise back to data',
    'Start 400 particles from pure Gaussian noise (t = 100) and apply the reverse step 100 times. Arrows show the <b>score</b> ∇log q_t(x), the direction in which noisy data becomes more likely at the current t. Right: the denoiser\'s current best guess x̂₀ = E[x₀ | x_t] for each particle. <b>Honest shortcut:</b> instead of training a network, this demo uses the exact score of the data density (a mixture of small Gaussians along the shape), which is what a trained denoiser approximates.');
  let sampler = 'ddpm', showField = true, stopR = null, tCur = T, P = [], trails = [];
  select(fig2.controls, { label: 'Sampler', value: sampler, options: [['ddpm', 'stochastic (DDPM)'], ['ddim', 'deterministic (DDIM / probability-flow ODE)']], onChange: (k) => { sampler = k; resetR(); } });
  const goBtn = button(fig2.controls, 'Denoise', () => run(), true);
  button(fig2.controls, 'One step', () => { if (stopR) { stopR(); stopR = null; goBtn.textContent = 'Denoise'; } stepR(); paintR(); });
  button(fig2.controls, 'Reset to noise', () => resetR());
  checkbox(fig2.controls, { label: 'Score arrows', checked: true, onChange: (c) => { showField = c; paintR(); } });
  const ro2 = stats(fig2.root);
  const row2 = el('div', { class: 'fig-row' }); fig2.body.append(row2);
  const l2 = el('div'), r2 = el('div'); row2.append(l2, r2);
  legend(l2, [['particle x_t', cat(0)], ['the data shape', muted()], ['score arrow', ink2()]]);
  legend(r2, [['denoiser guess x̂₀', cat(2)], ['the data shape', muted()]]);
  const F2 = frame(l2, { width: 460, height: 440, x: DOM, y: DOM, xLabel: 'x₁', yLabel: 'x₂' });
  const G2 = frame(r2, { width: 460, height: 440, x: DOM, y: DOM, xLabel: 'x₁', yLabel: 'x₂' });
  ['shape', 'field', 'trail', 'p'].forEach((n) => F2.layer(n)); ['shape', 'p'].forEach((n) => G2.layer(n));

  function resetR() {
    if (stopR) { stopR(); stopR = null; } goBtn.textContent = 'Denoise';
    const r = rng(21); tCur = T; P = d3.range(400).map(() => ({ x: r.normal(), y: r.normal() }));
    trails = d3.range(0, 400, 40).map((i) => [[P[i].x, P[i].y]]);
    paintR();
  }
  const rz = rng(77);
  function stepR() {
    if (tCur <= 0) return false;
    const sc = schedule(schedKey), C = SHAPES[shapeKey].centres, t = tCur;
    const abt = sc.ab[t], abp = sc.ab[t - 1], bt = sc.beta[t];
    P.forEach((p) => {
      const [x0, y0] = denoise(p.x, p.y, abt, C);
      if (sampler === 'ddpm') {
        // posterior q(x_{t-1} | x_t, x̂0): mean is a blend of x̂0 and x_t; then add fresh noise
        const c0 = Math.sqrt(abp) * bt / (1 - abt), ct = Math.sqrt(1 - bt) * (1 - abp) / (1 - abt);
        const sd = t > 1 ? Math.sqrt((1 - abp) / (1 - abt) * bt) : 0;
        p.x = c0 * x0 + ct * p.x + sd * rz.normal(); p.y = c0 * y0 + ct * p.y + sd * rz.normal();
      } else {
        // DDIM: keep the implied noise direction ε̂ and re-mix it at the lower noise level
        const ex = (p.x - Math.sqrt(abt) * x0) / Math.sqrt(1 - abt), ey = (p.y - Math.sqrt(abt) * y0) / Math.sqrt(1 - abt);
        p.x = Math.sqrt(abp) * x0 + Math.sqrt(1 - abp) * ex; p.y = Math.sqrt(abp) * y0 + Math.sqrt(1 - abp) * ey;
      }
    });
    trails.forEach((tr, k) => tr.push([P[k * 40].x, P[k * 40].y]));
    tCur--;
    return tCur > 0;
  }
  function run() {
    if (stopR) { stopR(); stopR = null; goBtn.textContent = 'Denoise'; return; }
    if (tCur <= 0) resetR();
    goBtn.textContent = 'Pause';
    let acc = 0;
    stopR = animate((dt) => { acc += dt; if (acc < 60) return true; acc = 0; const go = stepR(); paintR(); if (!go) { goBtn.textContent = 'Denoise'; stopR = null; } return go; });
  }
  function paintR() {
    const sc = schedule(schedKey), C = SHAPES[shapeKey].centres, ab = sc.ab[tCur];
    const cl = (v) => Math.max(DOM[0], Math.min(DOM[1], v));
    [F2, G2].forEach((F) => F.layer('shape').selectAll('circle').data(C).join('circle').attr('cx', (d) => F.x(d[0])).attr('cy', (d) => F.y(d[1])).attr('r', 2.2).attr('fill', muted()).attr('opacity', 0.5));
    // score field on a grid
    let arrows = [];
    if (showField && tCur > 0) {
      const g = d3.range(-2.75, 2.76, 0.5);
      g.forEach((gx) => g.forEach((gy) => { const [, , sx, sy] = denoise(gx, gy, ab, C); const n = Math.hypot(sx, sy), m = Math.min(0.4, 0.12 * n) / (n || 1); arrows.push({ x: gx, y: gy, x2: gx + sx * m, y2: gy + sy * m, n }); }));
    }
    F2.layer('field').selectAll('g').data(arrows).join((e) => { const gg = e.append('g'); gg.append('line'); gg.append('circle'); return gg; })
      .each(function (d) {
        const gg = d3.select(this);
        gg.select('line').attr('x1', F2.x(d.x)).attr('y1', F2.y(d.y)).attr('x2', F2.x(d.x2)).attr('y2', F2.y(d.y2)).attr('stroke', ink2()).attr('stroke-width', 1.3).attr('opacity', 0.75);
        gg.select('circle').attr('cx', F2.x(d.x2)).attr('cy', F2.y(d.y2)).attr('r', 1.8).attr('fill', ink2()).attr('opacity', 0.75);
      });
    F2.layer('trail').selectAll('path').data(trails).join('path').attr('fill', 'none').attr('stroke', ink()).attr('stroke-width', 1).attr('opacity', 0.45)
      .attr('d', (d) => d3.line()(d.map((q) => [F2.x(cl(q[0])), F2.y(cl(q[1]))])));
    F2.layer('p').selectAll('circle').data(P).join('circle').attr('cx', (d) => F2.x(cl(d.x))).attr('cy', (d) => F2.y(cl(d.y))).attr('r', 3.4)
      .attr('fill', cat(0)).attr('stroke', css('--surface')).attr('stroke-width', 0.8);
    const guesses = P.map((p) => { const [x0, y0] = denoise(p.x, p.y, ab, C); return { x: x0, y: y0 }; });
    G2.layer('p').selectAll('circle').data(guesses).join('circle').attr('cx', (d) => G2.x(cl(d.x))).attr('cy', (d) => G2.y(cl(d.y))).attr('r', 3.4)
      .attr('fill', cat(2)).attr('stroke', css('--surface')).attr('stroke-width', 0.8);
    // how close are particles to the shape?
    let near = 0, md = 0;
    P.forEach((p) => { let best = 1e9; C.forEach((c) => { const d = Math.hypot(p.x - c[0], p.y - c[1]); if (d < best) best = d; }); md += best; if (best < 0.15) near++; });
    ro2.set({ 'timestep t': tCur, 'ā_t': ab.toFixed(3), 'mean distance to shape': (md / P.length).toFixed(3), 'particles within 0.15 of shape': (100 * near / P.length).toFixed(0) + '%' });
  }
  F2.svg.on('mousemove', (e) => {
    const [mx, my] = d3.pointer(e, F2.g.node()); const x = F2.x.invert(mx), y = F2.y.invert(my);
    if (x < DOM[0] || x > DOM[1] || y < DOM[0] || y > DOM[1]) { untip(); return; }
    const [x0, y0, sx, sy] = denoise(x, y, schedule(schedKey).ab[tCur], SHAPES[shapeKey].centres);
    tip(e, `at (${x.toFixed(2)}, ${y.toFixed(2)}), t = ${tCur}<br>score = (${sx.toFixed(2)}, ${sy.toFixed(2)})<br>denoiser guess x̂₀ = (${x0.toFixed(2)}, ${y0.toFixed(2)})`);
  }).on('mouseleave', untip);
  resetR();

  section(main, 'What to notice', `<ul>
    <li>In the forward figure, drag t. With the <b>cosine</b> schedule the shape fades gradually; with the <b>linear</b> schedule (DDPM's original, rescaled to 100 steps) it is already gone by about t = 30, and the remaining steps just shuffle pure noise. Compare the "noise overtakes signal at t" readout: 50 for cosine, 15 for linear. Wasting steps like this is why the cosine schedule was introduced.</li>
    <li>At t = 100 the cloud is a standard normal blob for every shape: the forward process erases everything that made the data distinctive. That shared, known endpoint is what lets you start sampling from plain noise.</li>
    <li>Press <b>Denoise</b>. For the first few dozen steps the particles hardly change shape, but look at the right panel: the denoiser's guess x̂₀ starts as a cloud collapsed near the middle (at high noise, the best guess for <i>any</i> point is close to the average of the data) and then spreads out onto the shape. Coarse structure is decided early, fine detail late.</li>
    <li>The score arrows point towards the shape. Early on they are smooth and point roughly at the centre; near the end they become short-range and point at the nearest piece of the curve.</li>
    <li>Switch to the <b>deterministic</b> sampler and denoise again: the trails become smooth curves instead of jittery walks. Each noise vector now maps to exactly one output, by following a velocity field. Learning such a field directly is <a href="#/frontier/flow-matching">flow matching</a>.</li>
    <li>Try <b>Branching trajectory</b>: one stem splitting into three fates. Noise particles have to "decide" a branch, and the decision is made in the middle of the reverse process, when the guess x̂₀ splits apart.</li>
  </ul>`);

  steps(main, 'How it works', [
    '<b>Forward (fixed, no learning):</b> pick a schedule β₁…β_T. Adding noise step by step is equivalent to jumping straight to any t with x_t = √ā_t·x₀ + √(1−ā_t)·ε, where ā_t is the product of (1 − β) up to t.',
    '<b>Training:</b> take a real sample x₀, a random t and fresh noise ε, build x_t, and train a network ε̂(x_t, t) to predict the noise ε (plain mean squared error). Predicting the noise, the clean sample x̂₀, or the score are interchangeable: score = −ε̂ / √(1−ā_t).',
    'This simple regression secretly learns the score of the noisy data at every noise level (denoising score matching). This page skips training and plugs in the exact answer.',
    '<b>Sampling:</b> start from x_T ~ N(0, I). At each step, ask the denoiser for x̂₀, move part of the way towards it, and (for DDPM) add back a little fresh noise. Repeat down to t = 0.',
    'Conditioning (a text prompt, a cell type, a perturbation) is just an extra input to the denoiser. Classifier-free guidance amplifies it by extrapolating between conditional and unconditional predictions.',
  ]);
  math(main, 'forward:   x_t = √ā_t · x₀ + √(1 − ā_t) · ε,          ā_t = Π_{s≤t} (1 − β_s)\ntrain:     L = E_{x₀, t, ε} ‖ ε − ε̂_θ(x_t, t) ‖²\nscore:     ∇ log q_t(x) = −ε̂ / √(1 − ā_t)            x̂₀ = (x_t − √(1 − ā_t)·ε̂) / √ā_t\nDDPM step: x_{t−1} = c₀·x̂₀ + c_t·x_t + σ_t·z        DDIM: x_{t−1} = √ā_{t−1}·x̂₀ + √(1 − ā_{t−1})·ε̂');

  callout(main, '', 'In biology', `<ul>
    <li><b>Protein design:</b> RFdiffusion denoises protein backbones and AlphaFold 3 uses a diffusion module to produce atom coordinates. Here the "points" are 3-D atoms rather than 2-D dots.</li>
    <li><b>Single-cell generation and perturbation:</b> scDiffusion and related models generate expression profiles in a latent space (often an autoencoder's). Cell-state transport models such as CellFlow use the closely related <a href="#/frontier/flow-matching">flow matching</a> to carry control cells to perturbed ones.</li>
    <li><b>Imaging:</b> diffusion models now dominate microscopy and pathology image generation (virtual staining, super-resolution, synthetic training data). Cellpose 3 restores noisy and blurry images before segmentation, though with a trained restoration network rather than a diffusion sampler.</li>
    <li>The branching example is a reminder that generated "cells" are only as good as the density the model learned. Rare branches with few training cells are exactly where a learned score is least accurate.</li></ul>`);
  callout(main, 'warn', 'Pitfalls', `<ul>
    <li><b>Slow sampling:</b> each sample needs many network evaluations (here 100). DDIM, distillation and flow matching reduce the step count.</li>
    <li><b>Memorisation:</b> the exact score of a finite training set reproduces the training points. Real models generalise only because the network is an imperfect, smooth approximation; heavily over-trained models on small datasets (e.g. a few hundred patients) can leak training samples.</li>
    <li><b>Schedule and scaling matter:</b> data must be standardised to roughly unit scale, or the noise at t = T will not swamp the signal, and sampling will start from the wrong distribution.</li>
    <li>Pretty samples are not proof of a correct distribution. Check marginals, correlations and rare populations against held-out data.</li></ul>`);
  seeAlso(main, [['frontier', 'flow-matching', 'Flow matching (CellFlow)'], ['concepts', 'vae', 'Variational autoencoders'], ['concepts', 'gan', 'GANs'], ['concepts', 'autoencoder', 'Autoencoders'], ['frontier', 'state-sets', 'State: predicting distributions']]);
}
