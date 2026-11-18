// Optimizers race: SGD, momentum, RMSProp and Adam run from the same start on the same 2-D loss landscape.
import { header, figure, slider, select, button, checkbox, stats, section, steps, callout, math, seeAlso, animate, legend, tip, untip, el } from '../../lib/ui.js';
import { frame, cat, css, ink } from '../../lib/plot.js';
import { rng } from '../../lib/data.js';
const d3 = window.d3;

// Each landscape has its minimum loss at 0, so "loss" on the right panel is also "distance to optimal loss".
const SURFACES = {
  ravine: {
    name: 'Narrow ravine (ill-conditioned bowl)', f: (a, b) => 0.05 * a * a + 2 * b * b, g: (a, b) => [0.1 * a, 4 * b],
    start: [-2.7, 1.2], dom: [-3, 3], minima: [[0, 0]], lrSgd: 0.45, lrAda: 0.1, beta: 0.5, noise: 0.6,
  },
  valley: {
    name: 'Rosenbrock banana valley', f: (a, b) => (1 - a) ** 2 + 5 * (b - a * a) ** 2,
    g: (a, b) => [-2 * (1 - a) - 20 * a * (b - a * a), 10 * (b - a * a)],
    start: [-1.6, 2.2], dom: [-2.2, 2.4], minima: [[1, 1]], lrSgd: 0.03, lrAda: 0.1, beta: 0.9, noise: 2,
  },
  saddle: {
    name: 'Saddle point (flat ridge)', f: (a, b) => 0.5 * a * a + 0.25 * (b * b - 1) ** 2, g: (a, b) => [a, b * (b * b - 1)],
    start: [-2.6, 0.001], dom: [-3, 3], minima: [[0, 1], [0, -1]], lrSgd: 0.1, lrAda: 0.1, beta: 0.9, noise: 0.3,
  },
};

const OPTS = [
  { key: 'sgd', name: 'SGD' },
  { key: 'mom', name: 'Momentum' },
  { key: 'rms', name: 'RMSProp' },
  { key: 'adam', name: 'Adam' },
];
const N_STEPS = 300, TARGET = 1e-3;
let clipN = 0;
/** Clip a frame's plot group so trails that overshoot stay inside the axes. */
function clipFrame(Fr) {
  const id = `opt-clip-${++clipN}`;
  Fr.svg.append('defs').append('clipPath').attr('id', id).append('rect').attr('width', Fr.w).attr('height', Fr.h);
  return id;
}

/** Run one optimiser for N_STEPS from `start`, returning the whole trajectory. */
function trajectory(kind, surf, start, { lrSgd, lrAda, beta, noisy }) {
  const r = rng(11); // same noise sequence for every optimiser: a fair race
  const w = start.slice(), m = [0, 0], v = [0, 0];
  const out = [{ w: w.slice(), loss: surf.f(...w) }];
  let hit = out[0].loss < TARGET ? 0 : null, diverged = false;
  for (let t = 1; t <= N_STEPS; t++) {
    const gr = surf.g(...w);
    if (noisy) { gr[0] += r.normal(0, surf.noise); gr[1] += r.normal(0, surf.noise); }
    for (let i = 0; i < 2; i++) {
      if (kind === 'sgd') w[i] -= lrSgd * gr[i];
      else if (kind === 'mom') { m[i] = beta * m[i] + gr[i]; w[i] -= lrSgd * m[i]; }
      else if (kind === 'rms') { v[i] = 0.9 * v[i] + 0.1 * gr[i] ** 2; w[i] -= lrAda * gr[i] / (Math.sqrt(v[i]) + 1e-8); }
      else {
        m[i] = 0.9 * m[i] + 0.1 * gr[i]; v[i] = 0.999 * v[i] + 0.001 * gr[i] ** 2;
        w[i] -= lrAda * (m[i] / (1 - 0.9 ** t)) / (Math.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8);
      }
    }
    const loss = surf.f(...w);
    if (!isFinite(loss) || loss > 1e6) { diverged = true; out.push({ w: out[out.length - 1].w, loss: 1e6 }); break; }
    out.push({ w: w.slice(), loss });
    if (hit === null && loss < TARGET) hit = t;
  }
  return { pts: out, hit, diverged };
}

export default async function render(main) {
  header(main, 'Optimizers race',
    'Plain gradient descent only looks at the slope right here. Better optimisers also <b>remember past gradients</b> (momentum) or <b>rescale each parameter\'s step</b> by how steep it has been (RMSProp, Adam). Here all four start from the same point on the same surface.');

  const fig = figure(main, 'Four optimisers, one landscape',
    'Contours are lines of equal loss (darker = lower; × marks the minimum). Each coloured trail is one optimiser. Click anywhere on the map to restart all four from that point. The right panel shows loss on a log scale; the dashed line is the "arrived" threshold, loss = 10⁻³.');
  let surfKey = 'ravine', surf = SURFACES.ravine, start = surf.start.slice();
  const P = { lrSgd: surf.lrSgd, lrAda: surf.lrAda, beta: surf.beta, noisy: false };
  let runs = [], t = 0, stop = null;

  select(fig.controls, {
    label: 'Landscape', value: 'ravine', options: Object.entries(SURFACES).map(([k, v]) => [k, v.name]),
    onChange: (k) => {
      surfKey = k; surf = SURFACES[k]; start = surf.start.slice();
      P.lrSgd = surf.lrSgd; P.lrAda = surf.lrAda; P.beta = surf.beta;
      sSgd.set(Math.log10(P.lrSgd)); sAda.set(Math.log10(P.lrAda)); sBeta.set(P.beta);
      draw(); restart();
    },
  });
  const sSgd = slider(fig.controls, { label: 'η SGD & momentum', min: -3, max: 0, step: 0.05, value: Math.log10(P.lrSgd), fmt: (v) => (10 ** v).toPrecision(2), onInput: (v) => { P.lrSgd = 10 ** v; restart(); } });
  const sAda = slider(fig.controls, { label: 'η RMSProp & Adam', min: -3, max: 0, step: 0.05, value: Math.log10(P.lrAda), fmt: (v) => (10 ** v).toPrecision(2), onInput: (v) => { P.lrAda = 10 ** v; restart(); } });
  const sBeta = slider(fig.controls, { label: 'Momentum β', min: 0, max: 0.99, step: 0.01, value: P.beta, fmt: (v) => v.toFixed(2), onInput: (v) => { P.beta = v; restart(); } });
  checkbox(fig.controls, { label: 'Noisy (mini-batch) gradients', checked: false, onChange: (c) => { P.noisy = c; restart(); } });
  button(fig.controls, 'Replay', () => restart(), true);

  fig.root.insertBefore(legend(fig.root, OPTS.map((o, i) => [o.name, cat(i)])), fig.body);
  const readout = stats(fig.root);
  const note = el('div', { class: 'fig-caption' });
  fig.root.append(note);

  const row = d3.select(fig.body).append('div').attr('class', 'fig-row').node();
  const left = document.createElement('div'), right = document.createElement('div');
  row.append(left, right);

  let F, L;
  function draw() {
    left.innerHTML = ''; right.innerHTML = '';
    const [lo, hi] = surf.dom;
    F = frame(left, { width: 460, height: 450, x: [lo, hi], y: [lo, hi], xLabel: 'parameter w₁', yLabel: 'parameter w₂', grid: false });
    const n = 90, vals = new Array(n * n);
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) vals[j * n + i] = surf.f(lo + (hi - lo) * i / (n - 1), hi - (hi - lo) * j / (n - 1));
    const ext = d3.extent(vals);
    const levels = d3.range(16).map((k) => ext[0] + (ext[1] - ext[0]) * Math.pow(k / 16, 2.6));
    const cs = d3.contours().size([n, n]).thresholds(levels)(vals);
    const shade = d3.scaleLinear().domain([0, levels.length]).range([css('--accent'), css('--surface-2')]);
    const proj = d3.geoTransform({ point(px, py) { this.stream.point(F.x(lo + (hi - lo) * px / (n - 1)), F.y(hi - (hi - lo) * py / (n - 1))); } });
    F.layer('contours').selectAll('path').data(cs).join('path').attr('d', d3.geoPath(proj))
      .attr('fill', (d, i) => shade(i)).attr('stroke', css('--surface')).attr('stroke-width', 0.6);
    const mk = F.layer('minima').selectAll('text').data(surf.minima).join('text')
      .attr('x', (d) => F.x(d[0])).attr('y', (d) => F.y(d[1]) + 5).attr('text-anchor', 'middle')
      .attr('font-size', 16).attr('font-weight', 700).attr('fill', ink()).text('×');
    mk.style('pointer-events', 'none');
    const cf = clipFrame(F);
    F.layer('trails').attr('clip-path', `url(#${cf})`); F.layer('start'); F.layer('heads');
    F.svg.style('cursor', 'crosshair').on('click', (e) => {
      const [mx, my] = d3.pointer(e, F.g.node());
      start = [F.x.invert(mx), F.y.invert(my)];
      restart();
    });
    L = frame(right, { width: 460, height: 450, x: [0, N_STEPS], y: [1e-6, 10], xLabel: 'step', yLabel: 'loss (log scale)', yScale: d3.scaleLog() });
    const cl = clipFrame(L);
    L.layer('target');
    L.layer('curves').attr('clip-path', `url(#${cl})`);
  }

  function restart() {
    if (stop) { stop(); stop = null; }
    runs = OPTS.map((o) => trajectory(o.key, surf, start, P));
    t = 0;
    const l0 = Math.max(10, surf.f(...start) * 2);
    L.y.domain([1e-6, l0]);
    L.g.select('.gridline').call(d3.axisLeft(L.y).tickValues(d3.range(-6, Math.floor(Math.log10(l0)) + 1).map((e) => 10 ** e)).tickSize(-L.w).tickFormat(''));
    L.g.select('.axis.y').call(d3.axisLeft(L.y).ticks(5).tickFormat((v) => { const e = Math.round(Math.log10(v)); return Math.abs(Math.log10(v) - e) < 1e-6 ? (e === 0 ? '1' : e === 1 ? '10' : `1e${e}`) : ''; }));
    L.layer('target').selectAll('line').data([TARGET]).join('line')
      .attr('x1', 0).attr('x2', L.w).attr('y1', (d) => L.y(d)).attr('y2', (d) => L.y(d))
      .attr('stroke', css('--muted')).attr('stroke-dasharray', '4 4').attr('stroke-width', 1.2);
    F.layer('start').selectAll('circle').data([start]).join('circle').attr('cx', (d) => F.x(d[0])).attr('cy', (d) => F.y(d[1]))
      .attr('r', 5).attr('fill', ink()).attr('stroke', css('--surface')).attr('stroke-width', 1.5);
    paint();
    let acc = 0;
    stop = animate((dt) => {
      acc += dt; if (acc < 15) return true;
      acc = 0; t = Math.min(N_STEPS, t + 1); paint();
      return t < N_STEPS;
    });
  }

  function paint() {
    const [lo, hi] = surf.dom, cl = (v) => Math.max(lo - 5, Math.min(hi + 5, v)), ch = (v) => Math.max(lo, Math.min(hi, v));
    const shown = runs.map((r) => r.pts.slice(0, Math.min(t, r.pts.length - 1) + 1));
    F.layer('trails').selectAll('path').data(shown).join('path').attr('fill', 'none')
      .attr('stroke', (d, i) => cat(i)).attr('stroke-width', 2).attr('stroke-linejoin', 'round').attr('opacity', 0.95)
      .attr('d', d3.line().x((p) => F.x(cl(p.w[0]))).y((p) => F.y(cl(p.w[1]))));
    const heads = shown.map((s, i) => ({ i, p: s[s.length - 1], k: s.length - 1 }));
    F.layer('heads').selectAll('circle').data(heads).join('circle')
      .attr('cx', (d) => F.x(ch(d.p.w[0]))).attr('cy', (d) => F.y(ch(d.p.w[1]))).attr('r', 6)
      .attr('fill', (d) => cat(d.i)).attr('stroke', css('--surface')).attr('stroke-width', 1.5)
      .on('mousemove', (e, d) => tip(e, `<b>${OPTS[d.i].name}</b><br>step ${d.k}<br>w = (${d.p.w[0].toFixed(3)}, ${d.p.w[1].toFixed(3)})<br>loss ${d.p.loss.toExponential(2)}`))
      .on('mouseleave', untip);

    const yMin = L.y.domain()[0], yMax = L.y.domain()[1];
    L.layer('curves').selectAll('path').data(shown).join('path').attr('fill', 'none')
      .attr('stroke', (d, i) => cat(i)).attr('stroke-width', 2)
      .attr('d', d3.line().x((p, k) => L.x(k)).y((p) => L.y(Math.max(yMin, Math.min(yMax, p.loss)))));

    const obj = { step: t };
    runs.forEach((r, i) => {
      const done = r.hit !== null && r.hit <= t;
      obj[`${OPTS[i].name} → 10⁻³ at step`] = r.diverged && t >= r.pts.length - 1 ? 'diverged' : done ? r.hit : '—';
    });
    readout.set(obj);
    if (t >= N_STEPS) {
      const fin = runs.map((r, i) => `${OPTS[i].name} ${r.diverged ? 'diverged' : r.pts[r.pts.length - 1].loss.toExponential(1)}`).join(' · ');
      note.innerHTML = `Loss after ${N_STEPS} steps: ${fin}`;
    } else note.innerHTML = '&nbsp;';
  }

  draw(); restart();

  section(main, 'What to notice', `<ul>
    <li><b>Narrow ravine</b> (default settings): plain SGD zig-zags across the steep direction and creeps along the shallow one, reaching loss 10⁻³ at step 65. Momentum (β = 0.5) damps the zig-zag and builds speed along the floor, arriving at step 29. RMSProp gets there at step 31 and Adam at step 36.</li>
    <li>RMSProp arrives early but <b>never settles</b>: its step is always about η in each direction, so it rattles around the minimum (final loss about 5×10⁻³). Adam's bias-corrected averages let its steps shrink, so it keeps descending. Real training uses a decaying learning-rate schedule for exactly this reason.</li>
    <li><b>Banana valley</b>: SGD needs 239 steps because the valley floor is almost flat. Momentum builds up speed along the floor and wins (step 51); Adam takes 98 steps. RMSProp never gets below 10⁻³ at η = 0.1. Try lowering its η.</li>
    <li><b>Saddle point</b>: the start sits on a ridge where the w₂ gradient is almost zero. SGD and momentum have to wait for that tiny gradient to grow (steps 85 and 73). Adam divides by the gradient's own size, so it walks off the ridge almost at once and arrives at step 32.</li>
    <li>Set <b>momentum β</b> to 0: momentum becomes plain SGD and the two trails overlap. On the ravine, push β to 0.9 or higher and momentum overshoots: the heavy ball swings across the ravine and spirals in, arriving later than with β = 0.5. Each landscape loads its own sensible β.</li>
    <li>Tick <b>noisy gradients</b> to mimic mini-batches. Every method jitters; the adaptive ones jitter at a size set by η, not by the gradient.</li>
  </ul>`);

  steps(main, 'How it works', [
    '<b>SGD</b>: step against the current gradient, <code>w ← w − η·g</code>. One learning rate for every parameter, so the steepest direction sets the speed limit for all of them.',
    '<b>Momentum</b>: keep a running "velocity" <code>v ← β·v + g</code> and step along it. Gradients that keep pointing the same way add up (faster along valleys), while gradients that flip sign cancel (less zig-zag).',
    '<b>RMSProp</b>: keep a running average of the squared gradient per parameter, <code>s</code>, and divide the step by <code>√s</code>. Steep parameters get small steps and flat parameters get big ones, so every direction moves at roughly the same rate.',
    '<b>Adam</b>: momentum and RMSProp together, plus a <b>bias correction</b> that undoes the fact that both averages start at zero. It is the default for deep learning (scVI, transformers) because it needs little tuning.',
    'All four are "stochastic" in practice: g is computed on a random mini-batch, so it is a noisy estimate of the true gradient.',
  ]);
  math(main, 'SGD       w ← w − η g\nMomentum  v ← β v + g ;               w ← w − η v\nRMSProp   s ← 0.9 s + 0.1 g² ;         w ← w − η g / (√s + ε)\nAdam      m ← 0.9 m + 0.1 g ;  s ← 0.999 s + 0.001 g²\n          m̂ = m/(1−0.9ᵗ) ;  ŝ = s/(1−0.999ᵗ) ;  w ← w − η m̂ / (√ŝ + ε)');

  callout(main, '', 'In biology', 'scVI, totalVI and most single-cell deep models are trained with Adam (scvi-tools uses a learning rate of about 10⁻³ by default). Geneformer and scGPT use AdamW (Adam with decoupled weight decay) plus warm-up and decay schedules. Classical models like glmnet\'s lasso or DESeq2\'s GLM fits do not use these at all: they use coordinate descent or Newton-type updates (IRLS), which exploit curvature directly and are far faster on small, convex problems.');
  callout(main, 'warn', 'Pitfalls', '<ul><li>Adam\'s learning rate is in <b>parameter units per step</b>, SGD\'s is in parameter units per unit of gradient. You cannot copy a learning rate from one to the other.</li><li>An adaptive optimiser does not fix a badly scaled problem for free: it rescales per parameter, not along rotated (correlated) directions. <a href="#/archive/topics/scaling">Scaling inputs</a> still helps.</li><li>A constant learning rate leaves Adam and RMSProp bouncing near the minimum. Use a schedule or reduce-on-plateau.</li><li>Comparing optimisers at one learning rate each is unfair. Tune each one, as the default settings here do.</li></ul>');
  seeAlso(main, [['concepts', 'gradient-descent', 'Gradient descent'], ['concepts', 'scaling', 'Feature scaling'], ['concepts', 'loss-functions', 'Loss functions'], ['concepts', 'training-tricks', 'Dropout, batch norm, early stopping']]);
}
