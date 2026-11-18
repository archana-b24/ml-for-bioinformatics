// Perturbation baselines arena: simulate Perturb-seq-like pseudobulk deltas, fit mean / additive / linear / MLP,
// and score them by Pearson-delta against control and against the perturbed centroid (Systema).
import { header, figure, slider, select, button, stats, section, steps, math, seeAlso, legend, tip, untip, el } from '../../lib/ui.js';
import { frame, cat, css, ink, ink2, muted } from '../../lib/plot.js';
import { rng } from '../../lib/data.js';
import { sourceCard, toyBadge, coreIdea, realityCheck, useIt, cite, pearson, ridgeMulti } from '../../frontier/_shared.js';
const d3 = window.d3;

const G = 200, K = 6, NS = 60, ND = 40, NTEST_S = 20, R = 8;

/** Build the whole synthetic experiment. Biology is fixed by seed; sliders change stress and interaction. */
function simulate({ stress, gamma, embNoise, seed }) {
  const r = rng(seed);
  // gene programs: each gene belongs to one program (loading ±1-ish) with weak loadings elsewhere
  const U = Array.from({ length: G }, (_, g) => Array.from({ length: K }, (_, k) => (g % K === k ? (r() < 0.8 ? 1 : -1) * r.normal(1, 0.25) : r.normal(0, 0.08))));
  // generic stress vector: ~40 genes go up strongly, the rest drift slightly
  const sv = Array.from({ length: G }, () => (r() < 0.2 ? r.normal(1.0, 0.25) : r.normal(0.15, 0.1)));
  const snorm = Math.hypot(...sv); for (let g = 0; g < G; g++) sv[g] *= Math.sqrt(G) * 0.5 / snorm;
  const targets = r.shuffle(d3.range(G)).slice(0, NS);
  const singles = targets.map((t, i) => {
    const strength = i % 4 === 3 ? 0.2 : r() * 1.2 + 0.5;       // a quarter are near-duds
    const a = U[t].map((u) => u * strength);
    const c = r() * 0.8 + 0.6;                                   // how much generic stress this perturbation triggers
    const emb = U[t].map((u) => u + r.normal(0, embNoise));      // "prior-knowledge" embedding (e.g. from another screen)
    return { name: 'G' + String(t).padStart(3, '0'), t, a, c, emb, strength };
  });
  const programPart = (a) => U.map((row) => row.reduce((s, u, k) => s + u * a[k], 0));
  const noiseVec = () => Array.from({ length: G }, () => r.normal(0, 0.25));
  singles.forEach((p) => {
    const prog = programPart(p.a);
    p.truth = prog.map((v, g) => v + (g === p.t ? -2 * p.strength : 0) + stress * p.c * sv[g]);
    const nz = noiseVec();
    p.obs = p.truth.map((v, g) => v + nz[g]);
  });
  // doubles: pairs of singles; stress saturates (a cell can only be so stressed) and a program-level synergy term
  const doubles = [];
  const seen = new Set();
  while (doubles.length < ND) {
    const i = r.int(NS), j = r.int(NS); const key = Math.min(i, j) + '-' + Math.max(i, j);
    if (i === j || seen.has(key)) continue; seen.add(key);
    const p = singles[i], q = singles[j];
    const syn = programPart(d3.range(K).map(() => r.normal(0, 1.2) * Math.sqrt(p.strength * q.strength)));
    const prog = programPart(p.a.map((v, k) => v + q.a[k]));
    const truth = prog.map((v, g) => v + (g === p.t ? -2 * p.strength : 0) + (g === q.t ? -2 * q.strength : 0) + stress * Math.max(p.c, q.c) * sv[g] + gamma * syn[g]);
    const nz = noiseVec();
    doubles.push({ name: p.name + '+' + q.name, i, j, truth, obs: truth.map((v, g) => v + nz[g]), emb: p.emb.map((v, k) => v + q.emb[k]) });
  }
  return { singles, doubles, sv };
}

/** Tiny MLP from embedding (d) -> o outputs, tanh hidden layer, full-batch Adam. Flat typed arrays for speed. */
function trainMLP(X, Y, { H = 16, epochs = 200, lr = 0.02, seed = 3, l2 = 1e-3 } = {}) {
  const r = rng(seed), d = X[0].length, o = Y[0].length, n = X.length;
  const nW1 = H * d, nb1 = H, nW2 = o * H, NP = nW1 + nb1 + nW2 + o;
  const th = new Float64Array(NP), m = new Float64Array(NP), v = new Float64Array(NP), g = new Float64Array(NP);
  const oW1 = 0, ob1 = nW1, oW2 = nW1 + nb1, ob2 = oW2 + nW2;
  for (let i = 0; i < nW1; i++) th[oW1 + i] = r.normal(0, 1 / Math.sqrt(d));
  for (let i = 0; i < nW2; i++) th[oW2 + i] = r.normal(0, 0.3 / Math.sqrt(H));
  const h = new Float64Array(H), dh = new Float64Array(H), y = new Float64Array(o);
  const fwd = (x) => {
    for (let i = 0; i < H; i++) { let z = th[ob1 + i]; for (let k = 0; k < d; k++) z += th[oW1 + i * d + k] * x[k]; h[i] = Math.tanh(z); }
    for (let j = 0; j < o; j++) { let z = th[ob2 + j]; for (let i = 0; i < H; i++) z += th[oW2 + j * H + i] * h[i]; y[j] = z; }
  };
  for (let e = 1; e <= epochs; e++) {
    g.fill(0);
    for (let s = 0; s < n; s++) {
      const x = X[s]; fwd(x); dh.fill(0);
      for (let j = 0; j < o; j++) { const dy = (2 * (y[j] - Y[s][j])) / n; g[ob2 + j] += dy; for (let i = 0; i < H; i++) { g[oW2 + j * H + i] += dy * h[i]; dh[i] += dy * th[oW2 + j * H + i]; } }
      for (let i = 0; i < H; i++) { const dz = dh[i] * (1 - h[i] * h[i]); g[ob1 + i] += dz; for (let k = 0; k < d; k++) g[oW1 + i * d + k] += dz * x[k]; }
    }
    const c1 = 1 - 0.9 ** e, c2 = 1 - 0.999 ** e;
    for (let q = 0; q < NP; q++) {
      const gq = g[q] + (q < ob1 || (q >= oW2 && q < ob2) ? l2 * th[q] : 0);
      m[q] = 0.9 * m[q] + 0.1 * gq; v[q] = 0.999 * v[q] + 0.001 * gq * gq;
      th[q] -= lr * (m[q] / c1) / (Math.sqrt(v[q] / c2) + 1e-8);
    }
  }
  return (x) => { fwd(x); return Array.from(y); };
}

/** Fast PCA for few rows (n<=~100) and many columns: power iteration on the n×n Gram matrix. */
function gramPCA(X, k) {
  const n = X.length, p = X[0].length;
  const mu = new Float64Array(p); X.forEach((row) => { for (let j = 0; j < p; j++) mu[j] += row[j] / n; });
  const C = X.map((row) => Float64Array.from(row, (x, j) => x - mu[j]));
  const K2 = Array.from({ length: n }, () => new Float64Array(n));
  for (let a = 0; a < n; a++) for (let b = a; b < n; b++) { let s = 0; const ra = C[a], rb = C[b]; for (let j = 0; j < p; j++) s += ra[j] * rb[j]; K2[a][b] = K2[b][a] = s; }
  const vecs = [], vals = [];
  for (let c = 0; c < k; c++) {
    let u = Float64Array.from({ length: n }, (_, i) => Math.sin(i * 1.7 + c * 3.1) + 0.1);
    let lam = 0;
    for (let it = 0; it < 50; it++) {
      const w = new Float64Array(n);
      for (let a = 0; a < n; a++) { let s = 0; for (let b = 0; b < n; b++) s += K2[a][b] * u[b]; w[a] = s; }
      vecs.forEach((q, qi) => { let dd = 0; for (let a = 0; a < n; a++) dd += q[a] * w[a]; for (let a = 0; a < n; a++) w[a] -= dd * q[a]; });
      lam = Math.hypot(...w) || 1e-12; for (let a = 0; a < n; a++) u[a] = w[a] / lam;
    }
    vecs.push(u); vals.push(lam);
  }
  // gene axes = Cᵀu / sqrt(λ); scores = C · axis = u · sqrt(λ)
  const axes = vecs.map((u, c) => { const ax = new Array(p).fill(0); for (let a = 0; a < n; a++) for (let j = 0; j < p; j++) ax[j] += C[a][j] * u[a]; const s = Math.sqrt(vals[c]); return ax.map((x) => x / s); });
  const scores = C.map((_, a) => vecs.map((u, c) => u[a] * Math.sqrt(vals[c])));
  return { axes, scores, mean: Array.from(mu) };
}

/** Fit all models for a task and return predictions on the test set + training centroid. */
function runTask(sim, task) {
  const { singles, doubles } = sim;
  let train, test;
  if (task === 'singles') { train = singles.slice(NTEST_S); test = singles.slice(0, NTEST_S); }
  else { train = singles.concat(doubles.slice(ND / 2)); test = doubles.slice(0, ND / 2); }
  const Ytr = train.map((p) => p.obs);
  const centroid = d3.range(G).map((g) => d3.mean(Ytr, (y) => y[g]));
  // gene embedding = top-R principal axes of the training delta matrix (Ahlmann-Eltze-style linear model)
  const { axes, scores, mean: mu } = gramPCA(Ytr, R);
  const Xtr = train.map((p) => [...p.emb, 1]);
  const W = ridgeMulti(Xtr, scores, 0.5);
  const lin = (p) => { const x = [...p.emb, 1]; const s = d3.range(R).map((c) => x.reduce((acc, xi, k) => acc + xi * W[k][c], 0)); return mu.map((m0, g) => m0 + s.reduce((acc, sc, c) => acc + sc * axes[c][g], 0)); };
  const sdS = d3.range(R).map((c) => d3.deviation(scores, (s) => s[c]) || 1);
  const net = trainMLP(train.map((p) => p.emb), scores.map((s) => s.map((v, c) => v / sdS[c])));
  const mlp = (p) => { const s = net(p.emb).map((v, c) => v * sdS[c]); return mu.map((m0, g) => m0 + s.reduce((acc, sc, c) => acc + sc * axes[c][g], 0)); };
  const models = [
    { key: 'mean', label: 'Training mean', pred: () => centroid },
    ...(task === 'doubles' ? [{ key: 'add', label: 'Additive', pred: (p) => singles[p.i].obs.map((v, g) => v + singles[p.j].obs[g]) }] : []),
    { key: 'lin', label: 'Linear (embeddings)', pred: lin },
    { key: 'mlp', label: 'Small MLP', pred: mlp },
  ];
  models.forEach((m) => {
    m.preds = test.map((p) => m.pred(p));
    m.pdCtrl = d3.mean(test, (p, i) => pearson(m.preds[i], p.obs));
    m.pdCent = d3.mean(test, (p, i) => pearson(m.preds[i].map((v, g) => v - centroid[g]), p.obs.map((v, g) => v - centroid[g])));
    m.mse = d3.mean(test, (p, i) => d3.mean(p.obs, (v, g) => (v - m.preds[i][g]) ** 2));
  });
  return { models, test, centroid };
}

export default async function render(main) {
  header(main, 'Perturbation baselines arena',
    'Before trusting a "virtual cell" model, ask whether it beats <b>the dumbest possible prediction</b>. Here four models predict Perturb-seq responses on mock data, and two ways of scoring them tell very different stories.');

  sourceCard(main, {
    org: 'Ahlmann-Eltze, Huber & Anders (EMBL); Systema (Brbić lab, EPFL); Arc Institute Virtual Cell Challenge',
    when: '2025 (Nat Methods 22:1657); VCC 2025 wrap-up',
    links: ['ahlmann', 'systema', 'vcc2025', 'perteval', 'vcbench'],
    claim: 'Independent benchmarks: an additive baseline beats 5 foundation models and 2 other deep-learning models on double perturbations, and nothing beats the training mean on unseen single perturbations. A linear model on good perturbation embeddings does best.',
    selfReported: false, status: 'independent benchmark',
  });

  coreIdea(main, `<p>A Perturb-seq screen knocks down one (or two) genes per cell and reads out the whole transcriptome. The prediction task: given a perturbation the model has <i>not</i> seen, predict its <b>delta</b>, the shift of the mean expression profile away from control cells.</p>
  <p>The catch is that almost every perturbation also triggers a <b>shared, generic response</b> (stress, slower growth, selection). That shared part is easy to predict: just output the average delta. A metric that compares predicted and observed deltas <i>against control</i> rewards any model that has learned the generic part, even if it knows nothing about the specific perturbation. <b>Systema</b> fixes this by measuring deltas against the <i>centroid of all perturbed cells</i>, so only the perturbation-specific part counts.</p>
  <p><b>Key ML ingredient:</b> not a new architecture but a <b>baseline and an evaluation design</b>: mean, additive and low-rank linear models, scored with a metric that removes the shared signal.</p>`);

  // ------------------------------------------------------------ figure
  const fig = figure(main, 'Four models, two scoreboards',
    'Mock pseudobulk deltas for 200 genes. 60 single knockdowns are built from 6 gene programs, a target-gene knockdown, a generic stress vector and noise; 40 doubles add a program-level synergy. Bars are mean Pearson correlations over the held-out perturbations.');
  toyBadge(fig, 'The simulation is designed to show the mechanism. Change the sliders and the ranking changes.');
  const P = { task: 'singles', stress: 1.5, gamma: 0.6, embNoise: 0.4, seed: 11 };
  select(fig.controls, { label: 'Task', value: 'singles', options: [['singles', 'Unseen single perturbations'], ['doubles', 'Unseen doubles (singles seen)']], onChange: (v) => { P.task = v; recompute(); } });
  slider(fig.controls, { label: 'Generic stress', min: 0, max: 3, step: 0.1, value: P.stress, fmt: (v) => v.toFixed(1), onInput: (v) => { P.stress = v; recompute(); } });
  const gs = slider(fig.controls, { label: 'Interaction (doubles)', min: 0, max: 2, step: 0.1, value: P.gamma, fmt: (v) => v.toFixed(1), onInput: (v) => { P.gamma = v; recompute(); } });
  slider(fig.controls, { label: 'Embedding noise', min: 0, max: 1.5, step: 0.05, value: P.embNoise, fmt: (v) => v.toFixed(2), onInput: (v) => { P.embNoise = v; recompute(); } });
  button(fig.controls, 'New random screen', () => { P.seed += 1; recompute(); });
  const readout = stats(fig.root);

  const row = el('div', { class: 'fig-row' }); fig.body.append(row);
  const left = el('div'), right = el('div'); row.append(left, right);
  legend(left, [['Pearson-delta vs control', cat(0)], ['vs perturbed centroid (Systema)', cat(1)]]);
  const B = frame(left, { width: 520, height: 360, x: [0, 1], y: [-0.2, 1], yLabel: 'mean Pearson r (held-out)', xScale: d3.scaleBand().padding(0.25), margin: { bottom: 48 } });
  B.g.append('line').attr('class', 'zero').attr('x1', 0).attr('x2', B.w).attr('y1', B.y(0)).attr('y2', B.y(0)).attr('stroke', muted()).attr('stroke-width', 1);

  const ctl = el('div', { class: 'controls', style: 'margin:0 0 4px' }); right.append(ctl);
  let selModel = 'mean', selPert = 0, refMode = 'ctrl';
  const mSel = select(ctl, { label: 'Model', value: selModel, options: [['mean', 'Training mean'], ['add', 'Additive'], ['lin', 'Linear'], ['mlp', 'Small MLP']], onChange: (v) => { selModel = v; paintScatter(); } });
  select(ctl, { label: 'Reference', value: 'ctrl', options: [['ctrl', 'vs control'], ['cent', 'vs perturbed centroid']], onChange: (v) => { refMode = v; paintScatter(); } });
  button(ctl, 'Next perturbation ▸', () => { selPert = (selPert + 1) % res.test.length; paintScatter(); });
  const S = frame(right, { width: 460, height: 360, x: [-3, 3], y: [-3, 3], xLabel: 'observed delta (per gene)', yLabel: 'predicted delta' });
  const sTitle = S.g.append('text').attr('x', 4).attr('y', 12).attr('font-size', 12).attr('fill', ink2());
  const sR = S.g.append('text').attr('x', 4).attr('y', 28).attr('font-size', 12).attr('fill', ink());

  let sim, res, pending = null;
  function recompute() { if (pending) clearTimeout(pending); pending = setTimeout(() => { pending = null; recomputeNow(); }, 0); }
  function recomputeNow() {
    gs.input.disabled = P.task !== 'doubles';
    sim = simulate(P);
    res = runTask(sim, P.task);
    [...mSel.options].forEach((o) => { o.disabled = o.value === 'add' && P.task !== 'doubles'; });
    if (selModel === 'add' && P.task !== 'doubles') { selModel = 'mean'; mSel.value = 'mean'; }
    selPert = Math.min(selPert, res.test.length - 1);
    paintBars(); paintScatter();
    // share of observed delta variance that is the shared (centroid) direction
    const tot = d3.sum(res.test, (p) => d3.sum(p.obs, (v) => v * v));
    const spec = d3.sum(res.test, (p) => d3.sum(p.obs, (v, g) => (v - res.centroid[g]) ** 2));
    const best = res.models.reduce((a, b) => (b.pdCent > a.pdCent ? b : a));
    readout.set({
      'held-out perturbations': res.test.length,
      'signal shared by all perturbations': Math.max(0, Math.round(100 * (1 - spec / tot))) + '%',
      'mean baseline, vs control': res.models[0].pdCtrl.toFixed(2),
      'best model vs centroid': `${best.label} ${best.pdCent.toFixed(2)}`,
    });
  }

  function paintBars() {
    const ms = res.models;
    B.x.domain(ms.map((m) => m.label));
    B.g.select('.axis.x').call(d3.axisBottom(B.x)).selectAll('text').attr('font-size', 11.5);
    const bw = B.x.bandwidth() / 2;
    const data = ms.flatMap((m) => [{ m, k: 0, v: m.pdCtrl }, { m, k: 1, v: m.pdCent }]);
    B.layer('bars').selectAll('rect').data(data).join('rect')
      .attr('x', (d) => B.x(d.m.label) + d.k * bw).attr('width', bw - 2)
      .attr('y', (d) => B.y(Math.max(0, d.v))).attr('height', (d) => Math.abs(B.y(d.v) - B.y(0)))
      .attr('fill', (d) => cat(d.k)).attr('rx', 2)
      .on('mousemove', (e, d) => tip(e, `<b>${d.m.label}</b><br>${d.k ? 'vs perturbed centroid' : 'vs control'}: r = ${d.v.toFixed(3)}<br>MSE: ${d.m.mse.toFixed(3)}`)).on('mouseleave', untip);
    B.layer('lab').selectAll('text').data(data).join('text')
      .attr('x', (d) => B.x(d.m.label) + d.k * bw + bw / 2 - 1).attr('y', (d) => B.y(Math.max(0, d.v)) - 4)
      .attr('text-anchor', 'middle').attr('font-size', 11).attr('fill', ink()).text((d) => d.v.toFixed(2));
  }

  function paintScatter() {
    const m = res.models.find((x) => x.key === selModel) || res.models[0];
    const p = res.test[selPert], pred = m.preds[selPert];
    const ref = refMode === 'ctrl' ? res.centroid.map(() => 0) : res.centroid;
    const pts = p.obs.map((v, g) => ({ x: v - ref[g], y: pred[g] - ref[g], g, stress: sim.sv[g] > 0.8 }));
    const lim = Math.max(1, d3.max(pts, (d) => Math.max(Math.abs(d.x), Math.abs(d.y)))) * 1.05;
    S.x.domain([-lim, lim]); S.y.domain([-lim, lim]);
    S.g.select('.axis.x').call(d3.axisBottom(S.x).ticks(5)); S.g.select('.axis.y').call(d3.axisLeft(S.y).ticks(5));
    S.g.select('.gridline').call(d3.axisLeft(S.y).ticks(5).tickSize(-S.w).tickFormat(''));
    S.layer('diag').selectAll('line').data([0]).join('line').attr('x1', S.x(-lim)).attr('y1', S.y(-lim)).attr('x2', S.x(lim)).attr('y2', S.y(lim)).attr('stroke', muted()).attr('stroke-dasharray', '4 3');
    S.layer('pts').selectAll('circle').data(pts).join('circle')
      .attr('cx', (d) => S.x(d.x)).attr('cy', (d) => S.y(d.y)).attr('r', 4)
      .attr('fill', (d) => (d.stress ? cat(3) : cat(0))).attr('fill-opacity', 0.8).attr('stroke', css('--surface')).attr('stroke-width', 1)
      .on('mousemove', (e, d) => tip(e, `gene G${String(d.g).padStart(3, '0')}${d.stress ? ' (stress gene)' : ''}<br>observed ${d.x.toFixed(2)} · predicted ${d.y.toFixed(2)}`)).on('mouseleave', untip);
    sTitle.text(`${p.name} · ${m.label} · ${refMode === 'ctrl' ? 'vs control' : 'vs perturbed centroid'}`);
    sR.text(`Pearson r = ${pearson(pts.map((d) => d.x), pts.map((d) => d.y)).toFixed(2)}`);
  }
  legend(right, [['ordinary gene', cat(0)], ['strong stress-response gene', cat(3)]]);

  recomputeNow();

  section(main, 'What to notice', `<ul>
    <li>With the default settings, the <b>training mean</b> (which predicts the same delta for every perturbation) gets a respectable blue bar. Look at its scatter: the yellow stress genes line up along the diagonal and carry the correlation.</li>
    <li>Switch the scatter to <b>vs perturbed centroid</b>: the mean baseline collapses to exactly zero, because its prediction <i>is</i> the centroid. Only models that learned something perturbation-specific keep an orange bar.</li>
    <li>Drag <b>Generic stress</b> to 3. Every blue bar rises, including the mean's: the metric is rewarding the shared response. Drag it to 0 and the blue and orange bars converge.</li>
    <li>Raise <b>Embedding noise</b>. The linear model and MLP only know about an unseen perturbation through its embedding; once the embedding is noise, they fall back to the mean. This is why Ahlmann-Eltze et al. found the <i>embedding</i> (pretrained on perturbation data) mattered more than the architecture.</li>
    <li>On <b>doubles</b>, the additive model (sum of the two observed single deltas) is hard to beat at low interaction. Push <b>Interaction</b> up and see it weaken, although it still double-counts the saturating stress response.</li>
    <li>The small MLP rarely beats the linear model here: with 40–80 training perturbations there is too little data for extra flexibility to pay off.</li>
  </ul>`);

  steps(main, 'How it works', [
    'Simulate a screen: each knockdown target belongs to gene programs; its delta = programs it touches + its own knockdown + <code>c × stress vector</code> + noise.',
    'Hold out perturbations. <b>Singles:</b> 20 of 60 knockdowns are never seen in training. <b>Doubles:</b> all singles are seen, and half of the 40 doubles are held out.',
    '<b>Training mean:</b> predict the average training delta for everything.',
    '<b>Additive (doubles only):</b> predict delta(A+B) = delta(A) + delta(B), using the observed singles.',
    '<b>Linear:</b> take the top 8 principal axes of the training delta matrix as a <i>gene embedding</i>, and ridge-regress each perturbation\'s scores on a <i>perturbation embedding</i> (here a noisy copy of the target\'s program memberships, standing in for embeddings learned from another screen).',
    '<b>Small MLP:</b> the same pipeline with a 16-unit tanh hidden layer instead of the linear map, trained with Adam.',
    'Score each held-out perturbation with Pearson r between predicted and observed deltas, computed once relative to control and once relative to the centroid of all training perturbations (Systema).',
  ]);
  math(main, 'Pearson-delta      = corr( ŷ_p − x_ctrl ,  y_p − x_ctrl )\nSystema-style      = corr( ŷ_p − c_pert ,  y_p − c_pert ),   c_pert = mean over perturbations of y\nlinear model       Y ≈ G · W · Pᵀ     (G: gene embedding, P: perturbation embedding)');

  realityCheck(main, `<ul>
    <li>${cite('ahlmann', 'Ahlmann-Eltze, Huber & Anders (Nat Methods 2025)')}: on double perturbations, the additive baseline beat 5 foundation models and 2 other deep-learning models. On unseen single perturbations, nothing beat the training mean. A linear model with embeddings pretrained on <i>perturbation data</i> did best.</li>
    <li>${cite('systema', 'Systema')}: systematic differences between perturbed and control cells inflate standard metrics, which is the effect the blue bars show.</li>
    <li>${cite('perteval', 'PertEval-scFM')}, ${cite('perturbench', 'PerturBench')} and ${cite('vcbench', 'VCBench (2026)')} reach a similar conclusion. In VCBench, simple baselines match or beat foundation models on 4 of 5 axes.</li>
    <li>${cite('vcc2025', 'Virtual Cell Challenge 2025')} (H1 hESC, 300 CRISPRi perturbations): almost nobody beat the mean baseline on MAE, and the winners were hybrids of statistics and deep learning. The ${cite('vcc2026', '2026 round')} is zero-shot on 6 new cell lines; its winners have not been announced as of 2026-11-12.</li>
    <li>Arc describes ${cite('stateNews', 'State')} as the "first to consistently beat linear baselines". That is self-reported.</li>
  </ul>`);

  useIt(main, `<ul>
    <li>Before you report a perturbation model (yours or a vendor's), put <b>the training mean, the additive model and a linear model</b> next to it in the same table. If the fancy model does not clear them, say so.</li>
    <li>Report at least one metric that removes the shared response: Pearson-delta against the perturbed centroid, or correlation restricted to each perturbation's top DE genes.</li>
    <li>Split by perturbation (never by cell), and for doubles say whether the singles were seen. Those choices change the difficulty more than the model does.</li>
    <li>Check how much of your deltas is shared: the "signal shared by all perturbations" readout above is a two-line computation on your own pseudobulk matrix.</li>
  </ul>`);

  seeAlso(main, [['frontier', 'state-sets', 'State: predicting distributions'], ['frontier', 'metric-gaming', 'Virtual Cell Challenge metrics'], ['frontier', 'perturbation-tensor', 'Large Perturbation Model'],
    ['concepts', 'linear-regression', 'Linear regression'], ['concepts', 'regularization', 'Regularization'], ['concepts', 'pca', 'PCA'], ['concepts', 'neural-network', 'Neural networks'], ['concepts', 'foundation-models', 'Foundation models'], ['omics', 'sc-pseudobulk', 'Pseudobulk DE']]);
}
