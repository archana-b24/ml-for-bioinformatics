// CellFlow / PRiMeFlow: conditional flow matching from control to perturbed cells.
// Mini-replication: a tiny MLP velocity field v(x, t, dose) trained live with the CFM loss in 2-D; particles are integrated with Euler steps.
import { header, figure, slider, button, stats, section, steps, math, legend, seeAlso, animate, el } from '../../lib/ui.js';
import { frame, cat, css, ink2, muted } from '../../lib/plot.js';
import { rng } from '../../lib/data.js';
import { sourceCard, toyBadge, coreIdea, realityCheck, useIt, cite } from '../../frontier/_shared.js';
const d3 = window.d3;

function mlp(sizes, r) {
  const L = sizes.slice(1).map((nout, k) => {
    const nin = sizes[k], s = Math.sqrt(1 / nin), nW = nin * nout;
    return { nin, nout, W: Float64Array.from({ length: nW }, () => r.normal(0, s)), b: new Float64Array(nout),
      gW: new Float64Array(nW), gb: new Float64Array(nout), mW: new Float64Array(nW), vW: new Float64Array(nW), mb: new Float64Array(nout), vb: new Float64Array(nout) };
  });
  let t = 0;
  function forward(x) {
    const acts = [x]; let a = x;
    L.forEach((l, k) => {
      const o = new Float64Array(l.nout);
      for (let i = 0; i < l.nout; i++) { let s = l.b[i]; for (let j = 0; j < l.nin; j++) s += l.W[i * l.nin + j] * a[j]; o[i] = k < L.length - 1 ? Math.tanh(s) : s; }
      acts.push(o); a = o;
    });
    return acts;
  }
  function backward(acts, gOut) {
    let g = gOut;
    for (let k = L.length - 1; k >= 0; k--) {
      const l = L[k], a = acts[k], gin = new Float64Array(l.nin);
      for (let i = 0; i < l.nout; i++) { const gi = g[i]; l.gb[i] += gi; for (let j = 0; j < l.nin; j++) { l.gW[i * l.nin + j] += gi * a[j]; gin[j] += l.W[i * l.nin + j] * gi; } }
      if (k > 0) for (let j = 0; j < l.nin; j++) gin[j] *= 1 - a[j] * a[j];
      g = gin;
    }
  }
  function step(lr) {
    t++; const b1 = 0.9, b2 = 0.999, c1 = 1 - b1 ** t, c2 = 1 - b2 ** t;
    const upd = (P, G, M, V) => { for (let i = 0; i < P.length; i++) { M[i] = b1 * M[i] + (1 - b1) * G[i]; V[i] = b2 * V[i] + (1 - b2) * G[i] * G[i]; P[i] -= lr * (M[i] / c1) / (Math.sqrt(V[i] / c2) + 1e-8); G[i] = 0; } };
    L.forEach((l) => { upd(l.W, l.gW, l.mW, l.vW); upd(l.b, l.gb, l.mb, l.vb); });
  }
  return { forward, backward, step, predict: (x) => { const a = forward(x); return a[a.length - 1]; } };
}

// control cells and dose-dependent perturbed populations (2-D "expression space")
const ctrl = (r) => [r.normal(-1.3, 0.3), r.normal(0, 0.35)];
const pert = (r, d) => [r.normal(-1.3 + 2.6 * d, 0.3), r.normal((r() < 0.5 ? -1 : 1) * 1.3 * d, 0.3)];
const TRAIN_DOSES = [0.2, 0.5, 1.0];
const STEPS = 4000, BATCH = 64, NP = 200, EULER = 40;

export default async function render(main) {
  header(main, 'Flow matching: carry control cells to perturbed cells',
    'Instead of predicting one perturbed profile, learn a <b>velocity field</b>: an arrow at every point of expression space that says which way a cell should drift. Integrate control cells along the arrows and they arrive at the perturbed population. Give the arrows a <b>condition</b> (drug, dose, gene) and one model covers many perturbations.');

  sourceCard(main, {
    org: 'Theis lab (CellFlow); Altos Labs (PRiMeFlow, Virtual Cell Challenge 2025 generalist prize)',
    when: '2025 (CellFlow bioRxiv); 2025–26 (PRiMeFlow)',
    links: ['cellflow', 'primeflow', 'vcc2025'],
    claim: 'Conditional flow matching maps the control-cell distribution to the perturbed-cell distribution, conditioned on the perturbation.',
  });

  coreIdea(main, `<p>Take a control cell x₀ and a perturbed cell x₁, drawn independently (they are unpaired). Put a point on the straight line between them at a random time t, x_t = (1−t)·x₀ + t·x₁. Train a network to output the direction x₁ − x₀ at that point. Over many random pairs, the network learns the <b>average</b> direction at each place and time, and that average field transports the whole control cloud onto the whole perturbed cloud.</p>
  <p><b>Key ML ingredient:</b> the <b>conditional flow-matching (CFM) loss</b>: plain regression on velocities, with no adversary and no simulation during training. Generation is an ODE solve (here 40 Euler steps). It belongs to the same family as <a href="#/archive/topics/diffusion">diffusion models</a>, but with straight paths.</p>`);

  const fig = figure(main, 'A dose-conditioned velocity field, trained in your browser',
    'Left: particles start as control cells (blue) and follow the learned arrows; orange rings are cells simulated at the chosen dose. Faint arrows show the field at the current time. Right: the CFM training loss. Training uses only doses 0.2, 0.5 and 1.0 (150 cells each); move the slider to an unseen dose to test interpolation, or above 1 to test extrapolation.');
  toyBadge(fig, '2-D cells, MLP 4→32→32→2, 4,000 Adam steps');
  let dose = 0.75;
  slider(fig.controls, { label: 'Dose', min: 0, max: 1.3, step: 0.05, value: dose, fmt: (v) => v.toFixed(2), onInput: (v) => { dose = v; newTruth(); evaluate(); restartFlow(); } });
  button(fig.controls, 'Replay flow', () => restartFlow());
  button(fig.controls, 'Retrain', () => { init(); run(); }, true);
  legend(fig.root, [['control → generated particles', cat(0)], ['simulated cells at this dose (truth)', cat(1)]]);
  const readout = stats(fig.root);

  const row = el('div', { class: 'fig-row' }); fig.body.append(row);
  const left = el('div'), right = el('div'); row.append(left, right);
  const F = frame(left, { width: 500, height: 400, x: [-2.4, 2.6], y: [-2.4, 2.4], xLabel: 'expression axis 1', yLabel: 'expression axis 2' });
  const Lf = frame(right, { width: 420, height: 400, x: [0, STEPS], y: [0, 1.5], xLabel: 'training step', yLabel: 'CFM loss (smoothed)' });

  // fixed training data
  const rd = rng(17);
  const trainCtrl = d3.range(450).map(() => ctrl(rd));
  const trainPert = TRAIN_DOSES.map((d) => d3.range(150).map(() => pert(rd, d)));
  const r0 = rng(29);
  const start = d3.range(NP).map(() => ctrl(r0));
  let net, r, it, losses, ema, stop = null, truth, particles, tFlow, hold;

  function init() {
    if (stop) { stop(); stop = null; }
    net = mlp([4, 32, 32, 2], rng(7)); r = rng(41); it = 0; losses = []; ema = null;
  }
  const v = (x, y, t, d) => net.predict(Float64Array.from([x, y, t, d]));
  function trainStep() {
    let L = 0;
    for (let b = 0; b < BATCH; b++) {
      const k = r.int(3), x0 = trainCtrl[r.int(trainCtrl.length)], x1 = trainPert[k][r.int(150)], t = r();
      const xt = [(1 - t) * x0[0] + t * x1[0], (1 - t) * x0[1] + t * x1[1]], u = [x1[0] - x0[0], x1[1] - x0[1]];
      const acts = net.forward(Float64Array.from([xt[0], xt[1], t, TRAIN_DOSES[k]])), o = acts[acts.length - 1];
      const e0 = o[0] - u[0], e1 = o[1] - u[1]; L += e0 * e0 + e1 * e1;
      net.backward(acts, [2 * e0 / BATCH, 2 * e1 / BATCH]);
    }
    net.step(it < 2500 ? 0.004 : 0.0015);
    it++;
    L /= BATCH; ema = ema == null ? L : 0.97 * ema + 0.03 * L;
    if (it % 20 === 0) losses.push({ x: it, y: ema });
  }
  function newTruth() { const rt = rng(53); truth = d3.range(NP).map(() => pert(rt, dose)); }
  function integrate(P, t0, t1, n) {
    const h = (t1 - t0) / n;
    for (let s = 0; s < n; s++) { const t = t0 + s * h; P.forEach((p) => { const vv = v(p[0], p[1], t, dose); p[0] += h * vv[0]; p[1] += h * vv[1]; }); }
  }
  function evaluate() {
    const P = start.map((p) => p.slice()); integrate(P, 0, 1, EULER);
    const m = (A, j) => d3.mean(A, (p) => p[j]);
    const absY = (A) => d3.mean(A, (p) => Math.abs(p[1]));
    const status = TRAIN_DOSES.some((d) => Math.abs(d - dose) < 1e-6) ? 'seen in training' : dose > 1.0 ? 'extrapolation' : 'interpolation (unseen)';
    readout.set({
      'training step': `${it} / ${STEPS}`,
      'dose': `${dose.toFixed(2)} · ${status}`,
      'mean axis-1: generated / truth': `${m(P, 0).toFixed(2)} / ${m(truth, 0).toFixed(2)}`,
      'branch split |axis-2|: generated / truth': `${absY(P).toFixed(2)} / ${absY(truth).toFixed(2)}`,
    });
  }
  function restartFlow() { particles = start.map((p) => p.slice()); tFlow = 0; hold = 0; }

  const grid = [];
  for (let gx = -2.2; gx <= 2.45; gx += 0.36) for (let gy = -2.2; gy <= 2.25; gy += 0.36) grid.push([gx, gy]);
  function paint() {
    F.layer('field').selectAll('line').data(grid).join('line')
      .each(function (g) { const vv = v(g[0], g[1], Math.min(tFlow, 1), dose); d3.select(this).attr('x1', F.x(g[0])).attr('y1', F.y(g[1])).attr('x2', F.x(g[0] + 0.12 * vv[0])).attr('y2', F.y(g[1] + 0.12 * vv[1])); })
      .attr('stroke', muted()).attr('stroke-width', 1.2).attr('opacity', 0.55);
    F.layer('truth').selectAll('circle').data(truth).join('circle').attr('cx', (d) => F.x(d[0])).attr('cy', (d) => F.y(d[1])).attr('r', 4.5)
      .attr('fill', 'none').attr('stroke', cat(1)).attr('stroke-width', 1.4).attr('opacity', 0.7);
    F.layer('parts').selectAll('circle').data(particles).join('circle').attr('cx', (d) => F.x(d[0])).attr('cy', (d) => F.y(d[1])).attr('r', 4)
      .attr('fill', cat(0)).attr('stroke', css('--surface')).attr('stroke-width', 1);
    F.layer('tlabel').selectAll('text').data([tFlow]).join('text').attr('x', 8).attr('y', 16).attr('fill', ink2()).attr('font-size', 13).text((t) => `flow time t = ${Math.min(1, t).toFixed(2)}`);
    Lf.layer('loss').selectAll('path').data([losses]).join('path').attr('fill', 'none').attr('stroke', cat(0)).attr('stroke-width', 2)
      .attr('d', d3.line().x((d) => Lf.x(d.x)).y((d) => Lf.y(Math.min(1.5, d.y))));
  }
  function run() {
    if (stop) stop();
    restartFlow(); let frameN = 0;
    stop = animate((dt) => {
      if (it < STEPS) { for (let k = 0; k < 40 && it < STEPS; k++) trainStep(); if (it >= STEPS || ++frameN % 10 === 0) evaluate(); }
      if (tFlow < 1) { const nt = Math.min(1, tFlow + dt / 1800); integrate(particles, tFlow, nt, 1); tFlow = nt; }
      else if ((hold += dt) > 1200) restartFlow();
      paint();
      return true;
    });
  }
  init(); newTruth(); restartFlow(); evaluate(); paint(); run();

  section(main, 'What to notice', `<ul>
    <li>Press <b>Retrain</b> and watch the particles while the loss falls: once the field is good enough they <b>split into two branches</b>. No single training pair asks a cell to choose a branch; the split emerges because the averaged velocity depends on where a particle is.</li>
    <li>The <b>loss never reaches zero</b>. With random pairs, many different target velocities pass through the same point, so the best possible prediction is their average. That irreducible part is expected, and is exactly what makes the learned field a transport map rather than a memorised pairing.</li>
    <li>At an <b>unseen dose</b> between training doses (e.g. 0.75), compare the generated and true numbers in the readout: the condition is a smooth input, so interpolation is usually reasonable. Above 1.0 the model has never seen anything; extrapolation is where conditional models fail quietly.</li>
    <li>At dose 0 the true population equals the controls, but dose 0 was never trained. Watch whether the particles stay put.</li>
  </ul>`);

  steps(main, 'How it works', [
    'Collect control cells and perturbed cells for several conditions (here doses; in CellFlow and PRiMeFlow, perturbation embeddings).',
    'Sample a control cell x₀, a perturbed cell x₁ from condition c, and a time t ∈ [0,1]. Form x_t = (1−t)x₀ + t·x₁.',
    'Regress: train v(x_t, t, c) to output x₁ − x₀ with squared error. That is the whole training loop; no ODE is solved during training.',
    'To generate, start from control cells (or noise) and solve dx/dt = v(x, t, c) from t = 0 to 1 with a few dozen Euler steps.',
    'Read out the arriving cells as a predicted perturbed population: means, variances, responder fractions.',
  ]);
  math(main, 'L_CFM = E_{t, x₀, x₁, c} ‖ v_θ(x_t, t, c) − (x₁ − x₀) ‖²,    x_t = (1−t)·x₀ + t·x₁\ngenerate:  x(0) ~ control,  x(t+h) = x(t) + h·v_θ(x(t), t, c)');

  realityCheck(main, `<ul>
    <li>Altos's flow-matching entry (PRiMeFlow) won the <b>generalist prize</b> at the ${cite('vcc2025', 'Virtual Cell Challenge 2025')}; the three main winners used other approaches (scFoundation-style, an MLP on pseudobulk deltas, statistical reweighting). The challenge's lesson was that hybrids of statistics and deep learning did best, and almost nobody beat the mean baseline on MAE.</li>
    <li>CellFlow's performance claims are the <b>authors' own</b> (bioRxiv preprint). Independent perturbation benchmarks (${cite('perturbench', 'PerturBench')}, ${cite('perteval', 'PertEval-scFM')}, ${cite('vcbench', 'VCBench')}) repeatedly find that simple baselines are hard to beat on unseen perturbations.</li>
    <li>Flow matching solves the <i>distribution</i> problem (no collapse to the mean), not the <i>generalisation</i> problem. Whether the condition embedding lets it transfer to a new drug, gene or cell line is an empirical question per dataset.</li>
    <li>The ${cite('vcc2026', '2026 challenge')} tests zero-shot transfer to 6 new cell lines; winners had not been announced as of 2026-11-12.</li>
  </ul>`);

  useIt(main, `<ul>
    <li>Run flow matching in a low-dimensional space you trust (PCA or scVI latent of your Perturb-seq or drug-screen data), not on raw counts; decode or project back afterwards.</li>
    <li>Hold out whole <b>conditions</b> (a dose, a compound, a donor), not random cells, when you evaluate. Compare against "shift the controls by the mean delta of the nearest seen condition", which is a strong, cheap baseline.</li>
    <li>Useful for dose–response and time-course designs: train on a few doses or time points and inspect the interpolated populations as hypotheses, then validate the in-between conditions you care about.</li>
  </ul>`);

  seeAlso(main, [['frontier', 'state-sets', 'State: predicting distributions'], ['frontier', 'perturbation-arena', 'Perturbation baselines arena'], ['frontier', 'metric-gaming', 'VCC metrics'],
    ['concepts', 'diffusion', 'Diffusion models'], ['concepts', 'neural-network', 'Neural network playground'], ['concepts', 'vae', 'Variational autoencoders'], ['omics', 'sc-trajectory', 'Trajectories & pseudotime']]);
}
