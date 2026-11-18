// Virtual cells and perturbation prediction, lesson format (frontier tab).
// Arc: what → why → where → how it works (7 steps: delta and shared response → simple baselines → arena lab →
//      distribution loss (State) → flow matching → challenge metrics → tensor factorisation) → pitfalls → recap.
// All data are simulated in the browser from seeded generators (teaching toys, not reproductions); prose numbers are computed at render time.
import { header, figure, slider, select, button, stats, legend, seeAlso, animate, tip, untip, el } from '../lib/ui.js';
import { frame, cat, css, ink, ink2, muted, diverging, sequential } from '../lib/plot.js';
import { rng } from '../lib/data.js';
import { lessonNav, chapter, prose, rung, worked, quiz, cards, table, code, recap } from '../lib/lesson.js';
import { sourceCard, toyBadge, cite, pearson, ridgeMulti, cholSolve } from './_shared.js';
const d3 = window.d3;

const f1 = (v) => v.toFixed(1), f2 = (v) => (Math.abs(v) < 0.005 ? '0.00' : v.toFixed(2)), f3 = (v) => v.toFixed(3), pc = (v) => Math.round(100 * v) + '%';
const SUP = '⁰¹²³⁴⁵⁶⁷⁸⁹', sup = (n) => String(n).split('').map((c) => SUP[+c]).join('');
const sciTxt = (v) => { const e = Math.floor(Math.log10(v)); return `${(v / 10 ** e).toFixed(1)} × 10${sup(e)}`; };
const sci = (v) => { const e = Math.floor(Math.log10(v)); return `${(v / 10 ** e).toFixed(1)} × 10<sup>${e}</sup>`; };

// =========================================================================================================
// Simulated Perturb-seq screen (pseudobulk deltas): gene programmes + target knockdown + shared stress response + noise
// =========================================================================================================
const G = 200, K = 6, NS = 60, ND = 40, NTEST_S = 20, R = 8;
const PROG = ['IFN-γ response', 'antigen presentation', 'TGF-β / CAF-like', 'proliferation', 'hypoxia', 'translation'];
const DEF = { stress: 1.5, gamma: 0.6, embNoise: 0.4, seed: 11 };

function simulate({ stress, gamma, embNoise, seed }) {
  const r = rng(seed);
  const U = Array.from({ length: G }, (_, g) => Array.from({ length: K }, (_, k) => (g % K === k ? (r() < 0.8 ? 1 : -1) * r.normal(1, 0.25) : r.normal(0, 0.08))));
  const sv = Array.from({ length: G }, () => (r() < 0.2 ? r.normal(1.0, 0.25) : r.normal(0.15, 0.1)));
  const sn = Math.hypot(...sv); for (let g = 0; g < G; g++) sv[g] *= Math.sqrt(G) * 0.5 / sn;
  const targets = r.shuffle(d3.range(G)).slice(0, NS);
  const prog = (a) => U.map((row) => row.reduce((s, u, k) => s + u * a[k], 0));
  const noise = () => Array.from({ length: G }, () => r.normal(0, 0.25));
  const singles = targets.map((t, i) => {
    const strength = i % 4 === 3 ? 0.2 : r() * 1.2 + 0.5, c = r() * 0.8 + 0.6;
    const a = U[t].map((u) => u * strength), emb = U[t].map((u) => u + r.normal(0, embNoise));
    const truth = prog(a).map((v, g) => v + (g === t ? -2 * strength : 0) + stress * c * sv[g]), nz = noise();
    return { name: 'G' + String(t).padStart(3, '0'), t, a, c, strength, emb, truth, obs: truth.map((v, g) => v + nz[g]) };
  });
  const doubles = [], seen = new Set();
  while (doubles.length < ND) {
    const i = r.int(NS), j = r.int(NS), key = Math.min(i, j) + '-' + Math.max(i, j);
    if (i === j || seen.has(key)) continue; seen.add(key);
    const p = singles[i], q = singles[j];
    const syn = prog(d3.range(K).map(() => r.normal(0, 1.2) * Math.sqrt(p.strength * q.strength))), pr = prog(p.a.map((v, k) => v + q.a[k]));
    const truth = pr.map((v, g) => v + (g === p.t ? -2 * p.strength : 0) + (g === q.t ? -2 * q.strength : 0) + stress * Math.max(p.c, q.c) * sv[g] + gamma * syn[g]), nz = noise();
    doubles.push({ name: p.name + ' + ' + q.name, i, j, truth, obs: truth.map((v, g) => v + nz[g]), emb: p.emb.map((v, k) => v + q.emb[k]) });
  }
  return { singles, doubles, sv };
}

/** Small tanh MLP with Adam (page-local). Layers: sizes[0] → … → sizes[last]; hidden layers tanh, linear output. */
function mlp(sizes, r) {
  const L = sizes.slice(1).map((nout, k) => { const nin = sizes[k], s = Math.sqrt(1 / nin), nW = nin * nout; const z = () => new Float64Array(nW), zb = () => new Float64Array(nout);
    return { nin, nout, W: Float64Array.from({ length: nW }, () => r.normal(0, s)), b: zb(), gW: z(), gb: zb(), mW: z(), vW: z(), mb: zb(), vb: zb() }; });
  let t = 0;
  const forward = (x) => { const acts = [x]; let a = x;
    L.forEach((l, k) => { const o = new Float64Array(l.nout); for (let i = 0; i < l.nout; i++) { let s = l.b[i]; for (let j = 0; j < l.nin; j++) s += l.W[i * l.nin + j] * a[j]; o[i] = k < L.length - 1 ? Math.tanh(s) : s; } acts.push(o); a = o; });
    return acts; };
  const backward = (acts, gOut) => { let g = gOut;
    for (let k = L.length - 1; k >= 0; k--) { const l = L[k], a = acts[k], gin = new Float64Array(l.nin);
      for (let i = 0; i < l.nout; i++) { const gi = g[i]; l.gb[i] += gi; for (let j = 0; j < l.nin; j++) { l.gW[i * l.nin + j] += gi * a[j]; gin[j] += l.W[i * l.nin + j] * gi; } }
      if (k > 0) for (let j = 0; j < l.nin; j++) gin[j] *= 1 - a[j] * a[j];
      g = gin; } };
  const step = (lr) => { t++; const c1 = 1 - 0.9 ** t, c2 = 1 - 0.999 ** t;
    const upd = (P, Gr, M, V) => { for (let i = 0; i < P.length; i++) { M[i] = 0.9 * M[i] + 0.1 * Gr[i]; V[i] = 0.999 * V[i] + 0.001 * Gr[i] * Gr[i]; P[i] -= lr * (M[i] / c1) / (Math.sqrt(V[i] / c2) + 1e-8); Gr[i] = 0; } };
    L.forEach((l) => { upd(l.W, l.gW, l.mW, l.vW); upd(l.b, l.gb, l.mb, l.vb); }); };
  return { forward, backward, step, predict: (x) => { const a = forward(x); return a[a.length - 1]; } };
}
function fitMLP(X, Y, seed) {
  const net = mlp([X[0].length, 16, Y[0].length], rng(seed)), n = X.length;
  for (let e = 0; e < 200; e++) { X.forEach((x, i) => { const a = net.forward(Float64Array.from(x)), o = a[a.length - 1]; net.backward(a, Float64Array.from(o, (v, j) => 2 * (v - Y[i][j]) / n)); }); net.step(0.02); }
  return (x) => Array.from(net.predict(Float64Array.from(x)));
}

/** Top-k principal axes via the n×n Gram matrix (n rows ≪ p columns). */
function gramPCA(X, k) {
  const n = X.length, p = X[0].length, mu = new Float64Array(p);
  X.forEach((row) => { for (let j = 0; j < p; j++) mu[j] += row[j] / n; });
  const C = X.map((row) => Float64Array.from(row, (x, j) => x - mu[j])), K2 = Array.from({ length: n }, () => new Float64Array(n));
  for (let a = 0; a < n; a++) for (let b = a; b < n; b++) { let s = 0; for (let j = 0; j < p; j++) s += C[a][j] * C[b][j]; K2[a][b] = K2[b][a] = s; }
  const vecs = [], vals = [];
  for (let c = 0; c < k; c++) {
    const u = Float64Array.from({ length: n }, (_, i) => Math.sin(i * 1.7 + c * 3.1) + 0.1); let lam = 1e-12;
    for (let it = 0; it < 50; it++) {
      const w = new Float64Array(n); for (let a = 0; a < n; a++) { let s = 0; for (let b = 0; b < n; b++) s += K2[a][b] * u[b]; w[a] = s; }
      vecs.forEach((q) => { let dd = 0; for (let a = 0; a < n; a++) dd += q[a] * w[a]; for (let a = 0; a < n; a++) w[a] -= dd * q[a]; });
      lam = Math.hypot(...w) || 1e-12; for (let a = 0; a < n; a++) u[a] = w[a] / lam;
    }
    vecs.push(u); vals.push(lam);
  }
  const axes = vecs.map((u, c) => { const ax = new Array(p).fill(0); for (let a = 0; a < n; a++) for (let j = 0; j < p; j++) ax[j] += C[a][j] * u[a]; const s = Math.sqrt(vals[c]); return ax.map((x) => x / s); });
  return { axes, scores: C.map((_, a) => vecs.map((u, c) => u[a] * Math.sqrt(vals[c]))), mean: Array.from(mu) };
}

/** Fit every model on the training perturbations and score it on the held-out ones, relative to control and to the perturbed centroid. */
function runTask(sim, task) {
  const { singles, doubles } = sim;
  const train = task === 'singles' ? singles.slice(NTEST_S) : singles.concat(doubles.slice(ND / 2));
  const test = task === 'singles' ? singles.slice(0, NTEST_S) : doubles.slice(0, ND / 2);
  const Ytr = train.map((p) => p.obs), centroid = d3.range(G).map((g) => d3.mean(Ytr, (y) => y[g]));
  const { axes, scores, mean: mu } = gramPCA(Ytr, R);
  const W = ridgeMulti(train.map((p) => [...p.emb, 1]), scores, 0.5);
  const back = (s) => mu.map((m0, g) => m0 + s.reduce((acc, sc, c) => acc + sc * axes[c][g], 0));
  const lin = (p) => { const x = [...p.emb, 1]; return back(d3.range(R).map((c) => x.reduce((acc, xi, k) => acc + xi * W[k][c], 0))); };
  const sdS = d3.range(R).map((c) => d3.deviation(scores, (s) => s[c]) || 1);
  const net = fitMLP(train.map((p) => p.emb), scores.map((s) => s.map((v, c) => v / sdS[c])), 3);
  const models = [{ key: 'mean', label: 'Training mean', pred: () => centroid },
    ...(task === 'doubles' ? [{ key: 'add', label: 'Additive', pred: (p) => singles[p.i].obs.map((v, g) => v + singles[p.j].obs[g]) }] : []),
    { key: 'lin', label: 'Linear', pred: lin }, { key: 'mlp', label: 'Small MLP', pred: (p) => back(net(p.emb).map((v, c) => v * sdS[c])) },
    { key: 'ceil', label: 'True delta', pred: (p) => p.truth }];
  models.forEach((m) => {
    m.preds = test.map((p) => m.pred(p));
    m.pdCtrl = d3.mean(test, (p, i) => pearson(m.preds[i], p.obs));
    m.pdCent = d3.mean(test, (p, i) => pearson(m.preds[i].map((v, g) => v - centroid[g]), p.obs.map((v, g) => v - centroid[g])));
    m.mse = d3.mean(test, (p, i) => d3.mean(p.obs, (v, g) => (v - m.preds[i][g]) ** 2));
  });
  return { models, test, centroid, by: (k) => models.find((m) => m.key === k) };
}
const sharedShare = (res) => { const tot = d3.sum(res.test, (p) => d3.sum(p.obs, (v) => v * v)), sp = d3.sum(res.test, (p) => d3.sum(p.obs, (v, g) => (v - res.centroid[g]) ** 2)); return Math.max(0, 1 - sp / tot); };
const meanPairR = (rows) => { const rs = []; for (let a = 0; a < rows.length; a++) for (let b = a + 1; b < rows.length; b++) rs.push(pearson(rows[a], rows[b])); return d3.mean(rs); };

// =========================================================================================================
// Distribution-level toys: two 2-D populations of cells (control, perturbed), MMD loss and flow-matching field
// =========================================================================================================
const NC = 80, BW = [0.25, 0.6, 1.2];
const kern = (dx, dy) => { const q = dx * dx + dy * dy; let s = 0; for (const b of BW) s += Math.exp(-q / (2 * b * b)); return s / BW.length; };
function mmd2(A, B) { let aa = 0, bb = 0, ab = 0; for (const p of A) for (const q of A) aa += kern(p[0] - q[0], p[1] - q[1]); for (const p of B) for (const q of B) bb += kern(p[0] - q[0], p[1] - q[1]); for (const p of A) for (const q of B) ab += kern(p[0] - q[0], p[1] - q[1]); return aa / A.length ** 2 + bb / B.length ** 2 - 2 * ab / (A.length * B.length); }
const spread = (P) => { const mx = d3.mean(P, (p) => p[0]), my = d3.mean(P, (p) => p[1]); return Math.sqrt(d3.mean(P, (p) => (p[0] - mx) ** 2 + (p[1] - my) ** 2)); };
const SCEN = {
  two: { name: 'Two responder states', gen: (r, i) => [r.normal(1.3, 0.3), r.normal(i % 2 ? 1.1 : -1.1, 0.3)] },
  partial: { name: 'Partial response (40% unaffected)', gen: (r, i) => (i % 5 < 2 ? [r.normal(-1.3, 0.35), r.normal(0, 0.35)] : [r.normal(1.3, 0.3), r.normal(0.7, 0.3)]) },
};
function popData(scen) {
  const rd = rng(11), X = d3.range(NC).map(() => [rd.normal(-1.3, 0.35), rd.normal(0, 0.35)]), Z = d3.range(NC).map(() => rd() * 2 - 1);
  const rt = rng(23), Y = d3.range(NC).map((i) => SCEN[scen].gen(rt, i));
  return { X, Z, Y };
}

// ---------------------------------------------------------------------------------------------------------
// Challenge metrics: one fixed set of predictions scaled by alpha
// ---------------------------------------------------------------------------------------------------------
const MG = 150, MK = 40, ALPHAS = d3.range(0, 5.001, 0.1);
function simMetrics(stress, noise) {
  const r = rng(101), s = d3.range(MG).map(() => r.normal(0, 0.6));
  const one = () => { const mag = Math.exp(r.normal(0, 0.5)), v = new Array(MG).fill(0); for (let k = 0; k < 15; k++) v[r.int(MG)] += r.normal(0, mag); return v.map((x, g) => stress * s[g] + x + r.normal(0, 0.05)); };
  const test = d3.range(MK).map(one), train = d3.range(MK).map(one), rp = rng(202);
  return { test, pred: test.map((t) => t.map((x) => 0.4 * x + rp.normal(0, noise))), meanBase: d3.range(MG).map((g) => d3.mean(train, (t) => t[g])) };
}
const l1 = (a, b) => { let s = 0; for (let g = 0; g < MG; g++) s += Math.abs(a[g] - b[g]); return s / MG; };
function pds(P, T) {
  return d3.mean(P, (p, k) => { const dk = l1(p, T[k]); let sc = 0; T.forEach((t, j) => { if (j === k) return; const dj = l1(p, t); sc += dj > dk + 1e-12 ? 1 : Math.abs(dj - dk) <= 1e-12 ? 0.5 : 0; }); return sc / (T.length - 1); });
}
const metricsAt = (sim, a) => { const P = sim.pred.map((p) => p.map((x) => a * x)); return { pds: pds(P, sim.test), mae: d3.mean(P, (p, k) => l1(p, sim.test[k])), pr: a === 0 ? 0 : d3.mean(P, (p, k) => pearson(p, sim.test[k])) }; };
const curvesOf = (sim) => ALPHAS.map((a) => ({ a, ...metricsAt(sim, a) }));

// ---------------------------------------------------------------------------------------------------------
// Perturbation × readout × context tensor (CP factorisation, masked ALS)
// ---------------------------------------------------------------------------------------------------------
const TP = 30, TG = 50, TC = 6, TR = 3, SW = 30, LAM = 1;
function simTensor(noise) {
  const r = rng(77), A = d3.range(TP).map(() => d3.range(TR).map(() => r.normal(0, 1))), B = d3.range(TG).map(() => d3.range(TR).map(() => r.normal(0, 1))), D = d3.range(TC).map(() => d3.range(TR).map(() => 0.2 + 1.4 * r()));
  return d3.range(TP).map((p) => d3.range(TC).map((c) => d3.range(TG).map((g) => { let s = 0; for (let k = 0; k < TR; k++) s += A[p][k] * B[g][k] * D[c][k]; return s + r.normal(0, noise); })));
}
function maskTensor(frac) {
  const r = rng(91), M = d3.range(TP).map(() => d3.range(TC).map(() => r() < frac));
  M.forEach((row, p) => { if (!row.some(Boolean)) row[p % TC] = true; });
  d3.range(TC).forEach((c) => { if (!M.some((row) => row[c])) M[c][c] = true; });
  return M;
}
function fitTensor(Y, M, rank) {
  const r = rng(5), A = d3.range(TP).map(() => d3.range(rank).map(() => r.normal(0, 0.5))), B = d3.range(TG).map(() => d3.range(rank).map(() => r.normal(0, 0.5))), D = d3.range(TC).map(() => d3.range(rank).map(() => 1 + r.normal(0, 0.1)));
  const had = (u, v) => u.map((x, k) => x * v[k]);
  const solve = (F, entries) => F.forEach((row, i) => { const AtA = d3.range(rank).map(() => new Array(rank).fill(0)), Atb = d3.range(rank).map(() => [0]);
    entries(i, (z, y) => { for (let u = 0; u < rank; u++) { Atb[u][0] += z[u] * y; for (let w = 0; w < rank; w++) AtA[u][w] += z[u] * z[w]; } });
    for (let u = 0; u < rank; u++) AtA[u][u] += LAM; const x = cholSolve(AtA, Atb); for (let u = 0; u < rank; u++) row[u] = x[u][0]; });
  const predict = (p, c) => d3.range(TG).map((g) => { let s = 0; for (let k = 0; k < rank; k++) s += A[p][k] * B[g][k] * D[c][k]; return s; });
  const evalHeld = () => { const rs = []; for (let p = 0; p < TP; p++) for (let c = 0; c < TC; c++) if (!M[p][c]) rs.push({ p, c, r: pearson(predict(p, c), Y[p][c]) }); return rs; };
  const hist = [d3.mean(evalHeld(), (d) => d.r)];
  for (let s = 0; s < SW; s++) {
    solve(A, (p, add) => { for (let c = 0; c < TC; c++) if (M[p][c]) for (let g = 0; g < TG; g++) add(had(B[g], D[c]), Y[p][c][g]); });
    solve(B, (g, add) => { for (let p = 0; p < TP; p++) for (let c = 0; c < TC; c++) if (M[p][c]) add(had(A[p], D[c]), Y[p][c][g]); });
    solve(D, (c, add) => { for (let p = 0; p < TP; p++) if (M[p][c]) for (let g = 0; g < TG; g++) add(had(A[p], B[g]), Y[p][c][g]); });
    hist.push(d3.mean(evalHeld(), (d) => d.r));
  }
  let se = 0, n = 0; for (let p = 0; p < TP; p++) for (let c = 0; c < TC; c++) if (M[p][c]) predict(p, c).forEach((v, g) => { se += (v - Y[p][c][g]) ** 2; n++; });
  return { hist, held: evalHeld(), predict, rmse: Math.sqrt(se / n) };
}
function tensorBaselines(Y, M) {
  const pm = [], cm = [];
  for (let p = 0; p < TP; p++) for (let c = 0; c < TC; c++) { if (M[p][c]) continue;
    const others = d3.range(TC).filter((k) => M[p][k]), peers = d3.range(TP).filter((q) => M[q][c]);
    pm.push(pearson(d3.range(TG).map((g) => d3.mean(others, (k) => Y[p][k][g])), Y[p][c]));
    cm.push(peers.length ? pearson(d3.range(TG).map((g) => d3.mean(peers, (q) => Y[q][c][g])), Y[p][c]) : 0); }
  return { pm: d3.mean(pm), cm: d3.mean(cm) };
}

// =========================================================================================================
export default async function render(main) {
  const sim0 = simulate(DEF), res0 = runTask(sim0, 'singles');
  const memo = new Map();
  const scen = (p, task = 'singles') => { const k = JSON.stringify([p, task]); if (!memo.has(k)) memo.set(k, runTask(simulate({ ...DEF, ...p }), task)); return memo.get(k); };
  const nTrainS = NS - NTEST_S, nStress = sim0.sv.filter((v) => v > 0.6).length;

  header(main, 'Virtual cells and perturbation prediction',
    'A virtual cell is a model that predicts <b>how a cell\'s gene expression changes after a perturbation</b>, such as a gene knockdown or a drug, without running the experiment. Evidence so far indicates that the answer depends as much on the <b>baselines and metrics</b> used to judge such models as on the models themselves.');
  lessonNav(main);
  sourceCard(main, {
    org: 'Independent benchmarks (Ahlmann-Eltze, Huber &amp; Anders; Systema; PertEval-scFM; PerturBench; VCBench), the Arc Institute (State, Virtual Cell Challenge), Theis lab (CellFlow), Altos Labs (PRiMeFlow), LPM authors',
    when: '2025–2026',
    links: ['ahlmann', 'systema', 'vcc2025', 'stateNews', 'cellflow', 'lpm'],
    claim: 'Independent benchmarks report that simple baselines (training mean, additive, linear) match or beat large models on unseen perturbations; developers report gains for distribution-level models. <b>Question for this page:</b> how are perturbation predictions made and scored, and what does a model have to beat?',
    selfReported: false, status: 'mixed: independent and developer claims, labelled in the text',
  });
  prose(main, '<p class="fig-caption"><b>Every figure on this page is a teaching toy.</b> The screen is simulated in the browser (200 genes, 6 gene programmes named after the study\'s biology, a shared stress response and noise). The toys illustrate mechanisms; they do not reproduce the numbers of any paper. Gene names are placeholders.</p>');

  // ---------------------------------------------------------------- 1. WHAT
  let ch = chapter(main, 'what', 'What is perturbation prediction?', 'In one line: <b>given a perturbation never measured, predict the shift in expression it causes</b>.');
  prose(ch, `<p>In a Perturb-seq screen, each cell receives one genetic perturbation (for example a CRISPR knockdown of one gene) and its transcriptome is sequenced. Comparing perturbed cells with <b>control</b> cells gives, for each perturbation, a <b>delta</b>: the change in mean expression of every gene. A perturbation-prediction model is trained on the perturbations that were measured and asked to predict the delta of one that was not. "Virtual cell" is the name used for models intended to do this at scale and across cell types.</p>
    <p>The schematic uses the simulated screen on this page: ${NS} single knockdowns, of which ${NTEST_S} are held out.</p>`);
  ch.append(schematic(res0, nTrainS));
  table(ch, [
    { t: 'Perturbation', m: 'An intervention on the cell: a gene knockdown or knockout, a drug, or a combination', s: `${NS} single knockdowns and ${ND} pairs in the toy screen` },
    { t: 'Control', m: 'Cells that received no perturbation (or a non-targeting guide); the reference for every delta', s: 'delta = perturbed − control, per gene' },
    { t: 'Delta (<i>Δ</i>)', m: 'The shift in mean expression of each gene caused by one perturbation, a vector with one value per gene', s: `${G} numbers per perturbation` },
    { t: 'Perturbation embedding', m: 'A vector describing the perturbation (for example from a gene-gene similarity map) that lets a model predict an unseen one', s: `${K} numbers per gene in the toy; noisy` },
    { t: 'Held-out perturbation', m: 'A perturbation excluded from training and used only for scoring. Splitting must be by perturbation, not by cell', s: `${NTEST_S} of ${NS} singles; half of the ${ND} pairs` },
    { t: 'Baseline', m: 'A deliberately simple predictor that any proposed model must beat: the training mean, the sum of single deltas, a linear model', s: 'step 2 and the lab in step 3' },
    { t: 'Pearson-delta', m: 'Correlation between predicted and observed deltas across genes, averaged over held-out perturbations', s: `mean baseline: r = ${f2(res0.by('mean').pdCtrl)} (step 1)` },
  ], [['t', 'Term'], ['m', 'Meaning'], ['s', 'In the toy screen']]);

  // ---------------------------------------------------------------- 2. WHY
  ch = chapter(main, 'why', 'Why do we need it?', 'Screens cannot cover the space of perturbations, and claims about models need a fair yardstick.');
  prose(ch, `<p>A model that predicts perturbation effects is useful only if experiments cannot simply be run instead. Three constraints make that the case:</p><ol>
    <li><b>Combinatorics.</b> Single-gene screens are feasible; pairs, triples and drug combinations are not. The figure below counts what would have to be measured.</li>
    <li><b>Context.</b> An effect measured in one cell line or donor may not transfer to another, and each new context multiplies the number of experiments.</li>
    <li><b>Evaluation.</b> Because the target is a high-dimensional vector with a large shared component, a model can look accurate while learning nothing specific to the perturbation (steps 1 and 6).</li></ol>`);
  const fc = figure(ch, 'How many experiments would be needed?', 'Counts on a log scale. The challenge entry is the 300 CRISPRi perturbations of the Virtual Cell Challenge 2025 (H1 hESC, from the research brief); the other bars are arithmetic on the chosen number of genes and contexts.');
  let nGenes = 20000, nCtx = 6;
  slider(fc.controls, { label: 'genes perturbed', min: 1000, max: 20000, step: 500, value: nGenes, fmt: (v) => v.toLocaleString('en-GB'), onInput: (v) => { nGenes = v; paintC(); } });
  slider(fc.controls, { label: 'cell lines / contexts', min: 1, max: 10, step: 1, value: nCtx, onInput: (v) => { nCtx = v; paintC(); } });
  const sC = stats(fc.root);
  const FC = frame(fc.body, { width: 820, height: 250, x: [0, 10], y: [0, 5], xLabel: 'number of experimental conditions (log10 scale)', grid: false, axes: false, margin: { left: 250, right: 70, top: 8, bottom: 42 } });
  FC.g.append('g').attr('class', 'axis x').attr('transform', `translate(0,${FC.h})`).call(d3.axisBottom(FC.x).tickValues(d3.range(0, 11, 1)).tickFormat((v) => '10' + sup(v)));
  const bandC = d3.scaleBand().domain(d3.range(4)).range([0, FC.h]).padding(0.3);
  function paintC() {
    const rows = [{ l: 'Virtual Cell Challenge 2025 screen', v: 300 }, { l: 'Single perturbations, one cell line', v: nGenes }, { l: 'All gene pairs, one cell line', v: nGenes * (nGenes - 1) / 2 }, { l: `All gene pairs × ${nCtx} contexts`, v: nGenes * (nGenes - 1) / 2 * nCtx }];
    FC.layer('b').selectAll('rect').data(rows).join('rect').attr('x', 0).attr('y', (d, i) => bandC(i)).attr('height', bandC.bandwidth()).attr('width', (d) => Math.max(1, FC.x(Math.log10(d.v)))).attr('fill', (d, i) => cat(i === 0 ? 1 : 0)).attr('rx', 2)
      .on('mousemove', (e, d) => tip(e, `${d.l}<br><b>${d.v.toLocaleString('en-GB')}</b> conditions`)).on('mouseleave', untip);
    FC.layer('l').selectAll('text.lab').data(rows).join('text').attr('class', 'lab').attr('x', -8).attr('y', (d, i) => bandC(i) + bandC.bandwidth() / 2 + 4).attr('text-anchor', 'end').attr('font-size', 12).attr('fill', ink2()).text((d) => d.l);
    FC.layer('l').selectAll('text.val').data(rows).join('text').attr('class', 'val').attr('x', (d) => FC.x(Math.log10(d.v)) + 6).attr('y', (d, i) => bandC(i) + bandC.bandwidth() / 2 + 4).attr('font-size', 11.5).attr('fill', ink()).text((d) => (d.v >= 1e6 ? sciTxt(d.v) : d.v.toLocaleString('en-GB')));
    sC.set({ 'pairs per context': (nGenes * (nGenes - 1) / 2).toLocaleString('en-GB'), 'pairs ÷ challenge screen': Math.round(nGenes * (nGenes - 1) / 2 / 300).toLocaleString('en-GB') });
  }
  paintC();
  prose(ch, `<p>Measuring every pair of ${(20000).toLocaleString('en-GB')} genes would need ${sci(20000 * 19999 / 2)} conditions, about ${Math.round(20000 * 19999 / 2 / 300).toLocaleString('en-GB')} times the size of the 2025 challenge screen. A predictive model is only useful if it extrapolates from the measured fraction, which is why it has to be compared with simple extrapolation rules.</p>`);
  table(ch, [
    { a: 'Run the experiment', u: 'yes, in the tested context', h: 'yes', c: 'high per condition; does not scale to combinations' },
    { a: 'Training mean / additive / linear baselines', u: 'yes', h: 'no (mean effect only)', c: 'negligible; strong on unseen perturbations in independent benchmarks' },
    { a: 'Deep / foundation models (State, scFoundation-style, CellFlow)', u: 'claimed', h: 'some (State, flow matching)', c: 'high; benefit over baselines disputed (steps 3 and 6)' },
    { a: 'Factorised multi-experiment models (LPM)', u: 'claimed', h: 'no', c: 'moderate; pools heterogeneous screens (step 7)' },
  ], [['a', 'Approach'], ['u', 'Predicts unseen perturbations?'], ['h', 'Captures cell-to-cell variation?'], ['c', 'Cost and evidence']]);

  // ---------------------------------------------------------------- 3. WHERE
  ch = chapter(main, 'where', 'Where is it used?', 'Settings where perturbation effects must be estimated for conditions that were not, or could not be, measured.');
  cards(ch, [
    { tag: 'Target prioritisation', title: 'Which knockdowns are worth running?', body: 'Rank candidate genes by predicted effect on a programme such as IFN-γ response before committing to a focused screen. Predictions are used to choose which experiments to run, not to replace them.' },
    { tag: 'Combinations', title: 'Double knockouts and drug pairs', body: 'The additive baseline sums two single-perturbation deltas. Independent benchmarking found that it was not beaten by five foundation models on double perturbations (Ahlmann-Eltze et al., 2025).' },
    { tag: 'New contexts', title: 'Transfer to unseen cell lines', body: 'The Virtual Cell Challenge 2026 is zero-shot on six new cell lines. Factorised models such as LPM and embedding-based models address the same question.' },
    { tag: 'Challenges', title: 'Virtual Cell Challenge (Arc)', body: 'The 2025 round used H1 hESC and 300 CRISPRi perturbations, scored by PDS, DES and MAE. Winning entries combined statistical and deep-learning components.' },
    { tag: 'Heterogeneous response', title: 'Partial penetrance, resistant subclones', body: 'When only a fraction of cells respond, the population matters, not its mean. State and CellFlow predict a perturbed population from a control population.' },
    { tag: 'Data at scale', title: 'Pretraining on large perturbation atlases', body: 'Tahoe-100M (drug perturbations) and Xaira\'s 8M-cell Perturb-seq are examples of the data that models in this area are trained on.' },
    { tag: 'Benchmarking', title: 'Comparing methods fairly', body: 'PertEval-scFM, PerturBench, VCBench and Systema provide splits, baselines and metrics. They are the reference when a model is evaluated on new data.' },
    { tag: 'Inside bioinformatics workflows', title: 'Pseudobulk linear models', body: 'A ridge regression of perturbation deltas on a perturbation embedding is a few lines of code and is the baseline against which the larger models are tested.' },
  ]);
  prose(ch, `<p><b>When not to use it:</b> when the readout must be accurate for a specific gene rather than a programme-level trend (a model that predicts the shared response can still be wrong for any single gene); when the target context has no related measurements (a zero-shot prediction in a new cell type is the least-validated use); when a causal claim is required (a model reproduces associations learned from screens, not mechanisms); or when a few hundred perturbations can simply be measured.</p>`);

  // ---------------------------------------------------------------- 4. HOW
  ch = chapter(main, 'how', 'How it works', 'Seven short steps, from the delta matrix and its shared component to baselines, distribution-level models, evaluation metrics and factorisation.');

  // ---- Step 1: delta matrix and shared response
  let rb = rung(ch, 1, 'The delta matrix has a large shared component',
    `<p>Each row below is one single knockdown, each column one gene (grouped by gene programme). Almost every knockdown also triggers a <b>shared response</b> (stress, slower growth, selection), so ${nStress} of the ${G} genes move in the same direction in most rows. Genes in this block are listed first.</p>`);
  const f1g = figure(rb, 'Deltas of 30 knockdowns, relative to control or to the perturbed centroid', 'Colour: change in expression (red up, blue down). The perturbed centroid is the mean delta over all 60 knockdowns; subtracting it removes what every perturbation has in common. Hover a cell for details.');
  toyBadge(f1g, 'simulated pseudobulk deltas');
  let st1 = DEF.stress, mode1 = 'ctrl', sim1 = sim0;
  slider(f1g.controls, { label: 'shared stress response', min: 0, max: 3, step: 0.1, value: st1, fmt: f1, onInput: (v) => { st1 = v; sim1 = simulate({ ...DEF, stress: v }); paint1(); } });
  select(f1g.controls, { label: 'Reference', value: mode1, options: [['ctrl', 'delta vs control'], ['cent', 'delta vs perturbed centroid']], onChange: (v) => { mode1 = v; paint1(); } });
  const s1 = stats(f1g.root);
  const ord = d3.range(G).sort((a, b) => ((sim0.sv[b] > 0.6) - (sim0.sv[a] > 0.6)) || ((a % K) - (b % K)) || a - b);
  const F1 = frame(f1g.body, { width: 820, height: 330, x: [0, G], y: [0, 30], grid: false, axes: false, margin: { left: 14, right: 14, top: 40, bottom: 14 } });
  const cw1 = F1.w / G, ch1 = F1.h / 30, col1 = diverging(2.5);
  const blocks = [{ l: 'shared stress genes', n: nStress }, ...PROG.map((l, k) => ({ l, n: ord.filter((g) => !(sim0.sv[g] > 0.6) && g % K === k).length }))];
  let x0b = 0; blocks.forEach((b, i) => { const xa = x0b * cw1, xb = (x0b + b.n) * cw1; x0b += b.n;
    F1.layer('br').append('line').attr('x1', xa + 1).attr('x2', xb - 1).attr('y1', -8).attr('y2', -8).attr('stroke', i ? muted() : ink()).attr('stroke-width', i ? 1.5 : 3);
    F1.layer('br').append('text').attr('x', (xa + xb) / 2).attr('y', -14).attr('text-anchor', 'middle').attr('font-size', i ? 10 : 11).attr('font-weight', i ? 400 : 700).attr('fill', i ? ink2() : ink()).text(i && b.l.length > 14 ? b.l.split(' ')[0] : b.l); });
  F1.layer('br').append('text').attr('x', 0).attr('y', -28).attr('font-size', 11).attr('fill', muted()).text('gene groups (columns), 200 genes');
  const cells1 = []; for (let i = 0; i < 30; i++) for (let j = 0; j < G; j++) cells1.push({ i, j, g: ord[j] });
  const rects1 = F1.layer('c').selectAll('rect').data(cells1).join('rect').attr('x', (d) => d.j * cw1).attr('y', (d) => d.i * ch1).attr('width', cw1 + 0.3).attr('height', ch1 - 0.5)
    .on('mousemove', (e, d) => { const p = sim1.singles[d.i]; tip(e, `knockdown ${p.name} · gene G${String(d.g).padStart(3, '0')}${sim1.sv[d.g] > 0.6 ? ' (shared stress gene)' : ' · ' + PROG[d.g % K]}<br>delta ${f2(val1(d))} (${mode1 === 'ctrl' ? 'vs control' : 'vs centroid'})`); }).on('mouseleave', untip);
  F1.g.append('text').attr('transform', `translate(-4,${F1.h / 2}) rotate(-90)`).attr('text-anchor', 'middle').attr('font-size', 11).attr('fill', muted()).text('30 knockdowns');
  let cent1 = null;
  const val1 = (d) => sim1.singles[d.i].obs[d.g] - (mode1 === 'cent' ? cent1[d.g] : 0);
  legend(f1g.body, [['decrease', col1(-2.5)], ['no change', col1(0)], ['increase', col1(2.5)]]);
  function paint1() {
    cent1 = d3.range(G).map((g) => d3.mean(sim1.singles, (p) => p.obs[g]));
    rects1.attr('fill', (d) => col1(val1(d)));
    const rows = sim1.singles.slice(0, 30), tot = d3.sum(sim1.singles, (p) => d3.sum(p.obs, (v) => v * v)), sp = d3.sum(sim1.singles, (p) => d3.sum(p.obs, (v, g) => (v - cent1[g]) ** 2));
    s1.set({ 'variance shared by all knockdowns': pc(Math.max(0, 1 - sp / tot)), 'mean correlation between rows, vs control': f2(meanPairR(rows.map((p) => p.obs))), 'mean correlation between rows, vs centroid': f2(meanPairR(rows.map((p) => p.obs.map((v, g) => v - cent1[g])))) });
  }
  paint1();
  const c0 = d3.range(G).map((g) => d3.mean(sim0.singles, (p) => p.obs[g]));
  const rows0 = sim0.singles.slice(0, 30), shr0 = 1 - d3.sum(sim0.singles, (p) => d3.sum(p.obs, (v, g) => (v - c0[g]) ** 2)) / d3.sum(sim0.singles, (p) => d3.sum(p.obs, (v) => v * v));
  prose(rb, `<p>At the default setting ${pc(shr0)} of the squared delta is shared by all knockdowns, and the average pair of rows correlates at r = ${f2(meanPairR(rows0.map((p) => p.obs)))} against control but only ${f2(meanPairR(rows0.map((p) => p.obs.map((v, g) => v - c0[g]))))} once the centroid is removed. Because the shared part is easy to predict, any metric computed against control rewards a model that has learned nothing perturbation-specific. Set the slider to 0 to see the rows become nearly independent.</p>`);

  // ---- Step 2: baselines
  const dbl = scen({}, 'doubles').test[0], gS = d3.range(G).reduce((a, g) => (sim0.sv[g] > sim0.sv[a] ? g : a), 0), dsim = sim0.doubles[0], sa = sim0.singles[dsim.i], sb = sim0.singles[dsim.j];
  rb = rung(ch, 2, 'Three baselines: mean, additive and linear',
    `<p>Each baseline is a few lines of arithmetic and defines the minimum a learned model has to exceed.</p>`);
  table(rb, [
    { b: '<b>Training mean</b>', p: 'The average delta of all training perturbations, for every test perturbation', n: 'Nothing about the perturbation', c: 'Anything specific to it. Equals the perturbed centroid, so its score against the centroid is 0 by construction' },
    { b: '<b>Additive</b>', p: 'Δ(A + B) = Δ(A) + Δ(B), from the two measured single deltas', n: 'Both single perturbations measured', c: 'Interactions, and saturation of the shared response (it counts that response twice)' },
    { b: '<b>Linear</b>', p: 'Delta = low-rank gene axes × (ridge regression on a perturbation embedding)', n: 'An embedding of each perturbation', c: 'Non-linear dependence on the embedding; unseen biology absent from the embedding' },
  ], [['b', 'Baseline'], ['p', 'Prediction'], ['n', 'Needs'], ['c', 'Cannot capture']], { cls: 'compact' });
  worked(rb, `Additive prediction for the double ${dsim.name}`,
    `<p>At the strongest shared-stress gene (G${String(gS).padStart(3, '0')}), the two single knockdowns changed expression by <b>${f2(sa.obs[gS])}</b> and <b>${f2(sb.obs[gS])}</b>. The additive prediction is their sum, <b>${f2(sa.obs[gS] + sb.obs[gS])}</b>. The observed double changed by <b>${f2(dsim.obs[gS])}</b>, because the stress response saturates in the simulation (a cell can only be so stressed). Over all ${G} genes, the additive prediction correlates with the observed double at r = ${f2(pearson(sa.obs.map((v, g) => v + sb.obs[g]), dsim.obs))}, and the mean baseline at r = ${f2(pearson(scen({}, 'doubles').by('mean').preds[0], dbl.obs))} for the first held-out double.</p>`);
  prose(rb, `<p>The linear baseline follows ${cite('ahlmann', 'Ahlmann-Eltze, Huber and Anders (2025)')}: the top ${R} principal axes of the training deltas form a gene embedding, and a ridge regression maps a perturbation embedding onto the coordinates along those axes. The next step runs all three against a small neural network.</p>`);

  // ---- Step 3: arena lab
  rb = rung(ch, 3, 'Lab: baselines against a small neural network',
    `<p>Models are fitted on the training perturbations and scored on <b>held-out</b> ones (${NTEST_S} singles, or ${ND / 2} doubles). Two scores are shown for each model: Pearson-delta against control and against the perturbed centroid (the Systema-style check, ${cite('systema', 'Brbić lab')}). "True delta" predicts the noise-free simulated delta and marks the best score possible with noisy observations.</p>`);
  const fl = figure(rb, 'Baselines arena', `Left: mean Pearson r over held-out perturbations. Right: one held-out perturbation; each dot is a gene, with the strong shared-stress genes highlighted. Mock deltas for ${G} genes.`);
  toyBadge(fl, 'the simulation is designed to show the mechanism; change the sliders and the ranking changes');
  const P3 = { ...DEF, task: 'singles' };
  select(fl.controls, { label: 'Task', value: 'singles', options: [['singles', 'Unseen single perturbations'], ['doubles', 'Unseen doubles (singles seen)']], onChange: (v) => { P3.task = v; go3(); } });
  slider(fl.controls, { label: 'shared stress', min: 0, max: 3, step: 0.1, value: P3.stress, fmt: f1, onInput: (v) => { P3.stress = v; go3(); } });
  const gs3 = slider(fl.controls, { label: 'interaction (doubles)', min: 0, max: 2, step: 0.1, value: P3.gamma, fmt: f1, onInput: (v) => { P3.gamma = v; go3(); } });
  slider(fl.controls, { label: 'embedding noise', min: 0, max: 1.5, step: 0.05, value: P3.embNoise, fmt: f2, onInput: (v) => { P3.embNoise = v; go3(); } });
  button(fl.controls, 'New random screen', () => { P3.seed += 1; go3(); });
  const s3 = stats(fl.root);
  const row3 = el('div', { class: 'fig-row' }); fl.body.append(row3);
  const l3 = el('div'), r3 = el('div'); row3.append(l3, r3);
  legend(l3, [['Pearson-delta vs control', cat(0)], ['vs perturbed centroid', cat(1)]]);
  const B3 = frame(l3, { width: 520, height: 350, x: [0, 1], y: [-0.2, 1], yLabel: 'mean Pearson r (held-out)', xScale: d3.scaleBand().padding(0.25), margin: { top: 26, bottom: 46 } });
  B3.g.append('line').attr('x1', 0).attr('x2', B3.w).attr('y1', B3.y(0)).attr('y2', B3.y(0)).attr('stroke', muted());
  B3.g.append('text').attr('x', 2).attr('y', -10).attr('font-size', 11.5).attr('fill', ink2()).text('Training mean: 0 vs centroid by construction');
  const ctl = el('div', { class: 'controls', style: 'margin:0 0 4px' }); r3.append(ctl);
  let selM = 'mean', selP = 0, ref3 = 'ctrl', res3;
  const mSel = select(ctl, { label: 'Model', value: selM, options: [['mean', 'Training mean'], ['add', 'Additive'], ['lin', 'Linear'], ['mlp', 'Small MLP'], ['ceil', 'True delta']], onChange: (v) => { selM = v; paintS3(); } });
  select(ctl, { label: 'Reference', value: 'ctrl', options: [['ctrl', 'vs control'], ['cent', 'vs perturbed centroid']], onChange: (v) => { ref3 = v; paintS3(); } });
  button(ctl, 'Next perturbation ▸', () => { selP = (selP + 1) % res3.test.length; paintS3(); });
  const S3 = frame(r3, { width: 460, height: 350, x: [-3, 3], y: [-3, 3], xLabel: 'observed delta (per gene)', yLabel: 'predicted delta' });
  const t3a = S3.g.append('text').attr('x', 4).attr('y', 12).attr('font-size', 11.5).attr('fill', ink2()), t3b = S3.g.append('text').attr('x', 4).attr('y', 28).attr('font-size', 12).attr('fill', ink());
  legend(r3, [['ordinary gene', cat(0)], ['shared stress gene', cat(3)]]);
  let sim3;
  function go3() {
    gs3.input.disabled = P3.task !== 'doubles';
    sim3 = simulate(P3); res3 = runTask(sim3, P3.task);
    [...mSel.options].forEach((o) => { o.disabled = o.value === 'add' && P3.task !== 'doubles'; });
    if (selM === 'add' && P3.task !== 'doubles') { selM = 'mean'; mSel.value = 'mean'; }
    selP = Math.min(selP, res3.test.length - 1);
    paintB3(); paintS3();
    const best = res3.models.filter((m) => m.key !== 'ceil').reduce((a, b) => (b.pdCent > a.pdCent ? b : a));
    s3.set({ 'held-out perturbations': res3.test.length, 'signal shared by all perturbations': pc(sharedShare(res3)), 'training mean, vs control': f2(res3.by('mean').pdCtrl), 'best model, vs centroid': `${best.label} ${f2(best.pdCent)}` });
  }
  function paintB3() {
    const ms = res3.models; B3.x.domain(ms.map((m) => m.label));
    B3.g.select('.axis.x').call(d3.axisBottom(B3.x)).selectAll('text').attr('font-size', 11.5);
    const bw = B3.x.bandwidth() / 2, data = ms.flatMap((m) => [{ m, k: 0, v: m.pdCtrl }, { m, k: 1, v: m.pdCent }]);
    B3.layer('bars').selectAll('rect').data(data).join('rect').attr('x', (d) => B3.x(d.m.label) + d.k * bw).attr('width', bw - 2).attr('y', (d) => B3.y(Math.max(0, d.v))).attr('height', (d) => Math.abs(B3.y(d.v) - B3.y(0))).attr('fill', (d) => cat(d.k)).attr('rx', 2)
      .on('mousemove', (e, d) => tip(e, `<b>${d.m.label}</b><br>${d.k ? 'vs perturbed centroid' : 'vs control'}: r = ${f3(d.v)}<br>MSE ${f3(d.m.mse)}`)).on('mouseleave', untip);
    B3.layer('lab').selectAll('text').data(data).join('text').attr('x', (d) => B3.x(d.m.label) + d.k * bw + bw / 2 - 1).attr('y', (d) => B3.y(Math.max(0, d.v)) - 4).attr('text-anchor', 'middle').attr('font-size', 11).attr('fill', ink()).text((d) => f2(d.v));
  }
  function paintS3() {
    const m = res3.by(selM) || res3.models[0], p = res3.test[selP], pred = m.preds[selP], ref = ref3 === 'ctrl' ? res3.centroid.map(() => 0) : res3.centroid;
    const pts = p.obs.map((v, g) => ({ x: v - ref[g], y: pred[g] - ref[g], g, st: sim3.sv[g] > 0.6 })), lim = Math.max(1, d3.max(pts, (d) => Math.max(Math.abs(d.x), Math.abs(d.y)))) * 1.05;
    S3.x.domain([-lim, lim]); S3.y.domain([-lim, lim]);
    S3.g.select('.axis.x').call(d3.axisBottom(S3.x).ticks(5)); S3.g.select('.axis.y').call(d3.axisLeft(S3.y).ticks(5)); S3.g.select('.gridline').call(d3.axisLeft(S3.y).ticks(5).tickSize(-S3.w).tickFormat(''));
    S3.layer('d').selectAll('line').data([0]).join('line').attr('x1', S3.x(-lim)).attr('y1', S3.y(-lim)).attr('x2', S3.x(lim)).attr('y2', S3.y(lim)).attr('stroke', muted()).attr('stroke-dasharray', '4 3');
    S3.layer('p').selectAll('circle').data(pts).join('circle').attr('cx', (d) => S3.x(d.x)).attr('cy', (d) => S3.y(d.y)).attr('r', 4).attr('fill', (d) => (d.st ? cat(3) : cat(0))).attr('fill-opacity', 0.8).attr('stroke', css('--surface')).attr('stroke-width', 1)
      .on('mousemove', (e, d) => tip(e, `gene G${String(d.g).padStart(3, '0')}${d.st ? ' (shared stress gene)' : ''}<br>observed ${f2(d.x)} · predicted ${f2(d.y)}`)).on('mouseleave', untip);
    t3a.text(`${p.name} · ${m.label} · ${ref3 === 'ctrl' ? 'vs control' : 'vs centroid'}`); t3b.text(`Pearson r = ${f2(pearson(pts.map((d) => d.x), pts.map((d) => d.y)))}`);
  }
  go3();
  const run = (p, t) => scen(p, t);
  const sd0 = run({ stress: 0 }), sd3 = run({ stress: 3 }), en0 = run({ embNoise: 0.05 }), en1 = run({ embNoise: 1.5 }), db0 = run({ gamma: 0 }, 'doubles'), db2 = run({ gamma: 2 }, 'doubles');
  const seeds = [11, 12, 13, 14, 15].map((s) => run({ seed: s })), winLin = seeds.filter((r) => r.by('lin').pdCent > 0.02).length, mlpBeat = seeds.filter((r) => r.by('mlp').pdCent > r.by('lin').pdCent).length;
  prose(rb, '<p><b>Suggested comparisons</b> (the numbers below are computed from the same simulation):</p>');
  table(rb, [
    { t: 'Training mean: <b>vs control</b> → <b>vs centroid</b>', s: `r = ${f2(res0.by('mean').pdCtrl)} → <b>${f2(res0.by('mean').pdCent)}</b>`, m: '<b>The shared response alone earns a high score against control</b>, and the score disappears once the centroid is removed.' },
    { t: 'Shared stress <b>0</b> → <b>3</b> (training mean, vs control)', s: `${f2(sd0.by('mean').pdCtrl)} → <b>${f2(sd3.by('mean').pdCtrl)}</b>`, m: 'The stronger the shared component, the better the uninformed baseline looks. This is the effect Systema describes in real data.' },
    { t: 'Embedding noise <b>0.05</b> → <b>1.5</b> (linear, vs centroid)', s: `${f2(en0.by('lin').pdCent)} → <b>${f2(en1.by('lin').pdCent)}</b>`, m: 'A linear model knows an unseen perturbation only through its embedding; with a poor embedding it falls back towards the mean. Embedding quality matters more than the architecture.' },
    { t: 'Doubles, interaction <b>0</b> → <b>2</b> (additive, vs centroid)', s: `${f2(db0.by('add').pdCent)} → <b>${f2(db2.by('add').pdCent)}</b>; linear ${f2(db0.by('lin').pdCent)} → ${f2(db2.by('lin').pdCent)}`, m: 'Additive predictions use measured single effects and degrade as true interactions grow; the toy contains saturation of the shared response regardless.' },
    { t: 'Small MLP against linear (default screen, vs centroid)', s: `${f2(res0.by('mlp').pdCent)} vs ${f2(res0.by('lin').pdCent)}; MLP ahead in <b>${mlpBeat} of 5</b> random screens`, m: `${mlpBeat >= 3 ? 'The MLP is ahead in most screens here; the margin is small and depends on the embedding noise and the training-set size.' : 'The extra flexibility does not pay off with only ' + nTrainS + ' training perturbations, the pattern reported by independent benchmarks.'} Neither reaches the "true delta" ceiling (${f2(res0.by('ceil').pdCent)}).` },
    { t: 'New random screen (5 seeds)', s: `Linear exceeds 0.02 vs centroid in <b>${winLin} of 5</b> screens`, m: 'A single screen is one draw. Differences of a few hundredths between models are within what a new random screen changes.' },
  ], [['t', 'Try'], ['s', 'What you see'], ['m', 'What it means']], { cls: 'compact' });

  // ---- Step 4: State toy (MSE vs MMD)
  rb = rung(ch, 4, 'Predicting a distribution of cells (State)',
    `<p>Sequencing destroys the cell, so no cell is observed before and after a perturbation: control and perturbed cells are two <b>unpaired populations</b>. A regressor trained with squared error on randomly paired cells can only learn the average perturbed cell, because a random partner carries no information about which way an individual cell will go. ${cite('stateNews', 'State')} (Arc Institute) instead maps a <i>set</i> of control cells to a set of perturbed cells and is trained with a set-level loss. Secondary sources describe it as MMD (maximum mean discrepancy), which is <b>unverified</b> in the research brief; the toy below uses MMD to show the principle.</p>`);
  const f4 = figure(rb, 'Same network, two losses', 'Blue: control cells (input). Open orange rings: observed perturbed cells (target). Green: predicted perturbed cells, one per control cell. Both models are the same 3→20→20→2 tanh network that receives (control x, control y, a random number per cell); only the loss differs.');
  toyBadge(f4, 'two 2-D populations of 80 cells; State itself uses a set transformer');
  let sc4 = 'two', stop4 = null, it4 = 0, pop4, nets4, rs4;
  const MAX4 = 1000;
  select(f4.controls, { label: 'Perturbed population', value: sc4, options: Object.entries(SCEN).map(([k, v]) => [k, v.name]), onChange: (v) => { sc4 = v; reset4(); run4(); } });
  const btn4 = button(f4.controls, 'Pause', () => { if (stop4) { stop4(); stop4 = null; btn4.textContent = 'Train'; } else run4(); }, true);
  button(f4.controls, 'Reset', () => { reset4(); paint4(); });
  legend(f4.root, [['control cells', cat(0)], ['observed perturbed cells', cat(1)], ['predicted cells', cat(2)]]);
  const s4 = stats(f4.root);
  const row4 = el('div', { class: 'fig-row' }); f4.body.append(row4);
  const P4 = ['Squared error on random pairs', 'Set-level MMD loss'].map((t) => { const d = el('div'); row4.append(d); d.append(el('div', { style: 'font-size:13px;font-weight:600;color:var(--ink-2);margin:0 0 2px 40px' }, t)); return frame(d, { width: 460, height: 360, x: [-2.4, 2.6], y: [-2.2, 2.2], xLabel: 'expression axis 1', yLabel: 'expression axis 2' }); });
  function reset4() { if (stop4) { stop4(); stop4 = null; } pop4 = popData(sc4); rs4 = rng(5); nets4 = [mlp([3, 20, 20, 2], rng(3)), mlp([3, 20, 20, 2], rng(3))]; it4 = 0; }
  const in4 = () => pop4.X.map((x, i) => Float64Array.from([x[0], x[1], pop4.Z[i]]));
  function train4() {
    const U = in4(), N = NC, { Y } = pop4, perm = rs4.shuffle(d3.range(N));
    U.forEach((u, i) => { const a = nets4[0].forward(u), o = a[a.length - 1], y = Y[perm[i]]; nets4[0].backward(a, [2 * (o[0] - y[0]) / N, 2 * (o[1] - y[1]) / N]); }); nets4[0].step(0.01);
    const acts = U.map((u) => nets4[1].forward(u)), O = acts.map((a) => a[a.length - 1]);
    O.forEach((o, i) => { let gx = 0, gy = 0;
      for (const [set, w] of [[O, 2 / (N * N)], [Y, -2 / (N * N)]]) for (const q of set) { const dx = o[0] - q[0], dy = o[1] - q[1], qd = dx * dx + dy * dy; for (const b of BW) { const k = Math.exp(-qd / (2 * b * b)) / BW.length; gx += w * k * (-dx / (b * b)); gy += w * k * (-dy / (b * b)); } }
      nets4[1].backward(acts[i], [gx, gy]); });
    nets4[1].step(0.01); it4++;
  }
  function paint4() {
    const preds = nets4.map((n) => in4().map((u) => Array.from(n.predict(u))));
    P4.forEach((F, m) => { const pt = (ly, data, fn) => F.layer(ly).selectAll('circle').data(data).join('circle').attr('cx', (d) => F.x(d[0])).attr('cy', (d) => F.y(d[1])).call(fn);
      pt('c', pop4.X, (s) => s.attr('r', 4).attr('fill', cat(0)).attr('opacity', 0.35).attr('stroke', css('--surface')).attr('stroke-width', 1));
      pt('t', pop4.Y, (s) => s.attr('r', 4.5).attr('fill', 'none').attr('stroke', cat(1)).attr('stroke-width', 1.5).attr('opacity', 0.85));
      pt('p', preds[m], (s) => s.attr('r', 4.5).attr('fill', cat(2)).attr('stroke', css('--surface')).attr('stroke-width', 1.2).on('mousemove', (e, d) => tip(e, `predicted cell (${f2(d[0])}, ${f2(d[1])})`)).on('mouseleave', untip)); });
    s4.set({ 'training step': `${it4} / ${MAX4}`, 'MMD², squared-error model': f3(mmd2(preds[0], pop4.Y)), 'MMD², MMD model': f3(mmd2(preds[1], pop4.Y)), 'spread: squared error / MMD / observed': `${f2(spread(preds[0]))} / ${f2(spread(preds[1]))} / ${f2(spread(pop4.Y))}` });
  }
  function run4() { if (stop4) stop4(); btn4.textContent = 'Pause'; let k = 0;
    stop4 = animate(() => { for (let s = 0; s < 6 && it4 < MAX4; s++) train4(); if (++k % 2 === 0 || it4 >= MAX4) paint4(); if (it4 >= MAX4) { btn4.textContent = 'Train'; stop4 = null; return false; } return true; }); }
  reset4(); paint4(); run4();
  const popTwo = popData('two'), popPart = popData('partial'), mY = (Y) => [d3.mean(Y, (y) => y[0]), d3.mean(Y, (y) => y[1])];
  const nearest = (Y) => { const m = mY(Y); return d3.min(Y, (y) => Math.hypot(y[0] - m[0], y[1] - m[1])); };
  prose(rb, `<p>In the "two responder states" population, the mean perturbed cell lies at (${mY(popTwo.Y).map(f2).join(', ')}), and the closest observed cell is ${f2(nearest(popTwo.Y))} units from it; no real cell has the average phenotype. The squared-error model converges there. The MMD loss compares whole sets: each predicted cell is pulled towards nearby observed cells and pushed away from other predicted cells, which prevents collapse. Distribution losses address collapse; they do not by themselves ensure that the predicted cloud shifts the right genes, or that an unseen perturbation is predicted well.</p>`);

  // ---- Step 5: flow matching
  rb = rung(ch, 5, 'Flow matching: carrying control cells to perturbed cells',
    `<p>${cite('cellflow', 'CellFlow')} (Theis lab) and ${cite('primeflow', 'PRiMeFlow')} (Altos Labs, generalist prize in the 2025 challenge) use <b>conditional flow matching</b>. A control cell <i>x</i>₀ and a perturbed cell <i>x</i>₁ are drawn independently. The point on the straight line between them at time <i>t</i> is <i>x<sub>t</sub></i> = (1 − <i>t</i>)·<i>x</i>₀ + <i>t</i>·<i>x</i>₁, and a network is trained by plain regression to output the velocity <i>x</i>₁ − <i>x</i>₀ at (<i>x<sub>t</sub></i>, <i>t</i>, perturbation). At prediction time, control cells are moved along the learned velocity field by a numerical ODE solver.</p>`);
  const f5 = figure(rb, 'The flow-matching regression target, solved without a network', 'Grey lines: straight paths between randomly paired control and perturbed cells. Arrows: the optimal velocity field at time t, estimated by kernel-weighted averaging of the paths that pass nearby (a stand-in for the trained network). Green: control cells moved along that field up to time t.');
  toyBadge(f5, 'kernel regression in 2-D replaces the neural network');
  let t5 = 0.6, h5 = 0.25, traj5 = null;
  const pop5 = popData('two'), pr5 = rng(31).shuffle(d3.range(NC)), X1 = pr5.map((i) => pop5.Y[i]), X0 = pop5.X, ST5 = 30;
  slider(f5.controls, { label: 'time t', min: 0, max: 1, step: 1 / ST5, value: t5, fmt: f2, onInput: (v) => { t5 = v; paint5(); } });
  slider(f5.controls, { label: 'kernel width (field smoothness)', min: 0.08, max: 1.2, step: 0.02, value: h5, fmt: f2, onInput: (v) => { h5 = v; integ5(); paint5(); } });
  legend(f5.root, [['control cells', cat(0)], ['observed perturbed cells', cat(1)], ['cells moved by the field', cat(2)]]);
  const s5 = stats(f5.root);
  const F5 = frame(f5.body, { width: 820, height: 340, x: [-2.4, 2.6], y: [-2.2, 2.2], xLabel: 'expression axis 1', yLabel: 'expression axis 2' });
  const vel = (x, t) => { let sw = 0, vx = 0, vy = 0; for (let i = 0; i < NC; i++) { const px = (1 - t) * X0[i][0] + t * X1[i][0], py = (1 - t) * X0[i][1] + t * X1[i][1], w = Math.exp(-((x[0] - px) ** 2 + (x[1] - py) ** 2) / (2 * h5 * h5)); sw += w; vx += w * (X1[i][0] - X0[i][0]); vy += w * (X1[i][1] - X0[i][1]); } return sw < 1e-12 ? [0, 0] : [vx / sw, vy / sw]; };
  function integ5() { let x = X0.map((p) => p.slice()); traj5 = [x.map((p) => p.slice())]; for (let k = 0; k < ST5; k++) { const t = k / ST5; x = x.map((p) => { const v = vel(p, t); return [p[0] + v[0] / ST5, p[1] + v[1] / ST5]; }); traj5.push(x.map((p) => p.slice())); } }
  F5.layer('paths').selectAll('line').data(d3.range(NC)).join('line').attr('x1', (i) => F5.x(X0[i][0])).attr('y1', (i) => F5.y(X0[i][1])).attr('x2', (i) => F5.x(X1[i][0])).attr('y2', (i) => F5.y(X1[i][1])).attr('stroke', muted()).attr('stroke-opacity', 0.18);
  F5.layer('c').selectAll('circle').data(X0).join('circle').attr('cx', (d) => F5.x(d[0])).attr('cy', (d) => F5.y(d[1])).attr('r', 3.5).attr('fill', cat(0)).attr('opacity', 0.3);
  F5.layer('t').selectAll('circle').data(pop5.Y).join('circle').attr('cx', (d) => F5.x(d[0])).attr('cy', (d) => F5.y(d[1])).attr('r', 4.5).attr('fill', 'none').attr('stroke', cat(1)).attr('stroke-width', 1.5);
  const arrowG = d3.range(8).flatMap((i) => d3.range(5).map((j) => [-2.1 + i * 0.5, -1.6 + j * 0.8]));
  function paint5() {
    const k = Math.round(t5 * ST5), P = traj5[k], tt = k / ST5;
    F5.layer('a').selectAll('path').data(arrowG.map((p) => { const v = vel(p, tt), m = Math.hypot(v[0], v[1]); return m < 0.05 ? null : { p, v, m }; }).filter(Boolean)).join('path').attr('fill', 'none').attr('stroke', ink2()).attr('stroke-width', 1.4)
      .attr('d', (d) => { const ex = d.p[0] + d.v[0] * 0.2, ey = d.p[1] + d.v[1] * 0.2, X = F5.x(ex), Y = F5.y(ey), ang = Math.atan2(F5.y(ey) - F5.y(d.p[1]), F5.x(ex) - F5.x(d.p[0])); return `M${F5.x(d.p[0])},${F5.y(d.p[1])}L${X},${Y}M${X - 6 * Math.cos(ang - 0.45)},${Y - 6 * Math.sin(ang - 0.45)}L${X},${Y}L${X - 6 * Math.cos(ang + 0.45)},${Y - 6 * Math.sin(ang + 0.45)}`; });
    F5.layer('m').selectAll('circle').data(P).join('circle').attr('cx', (d) => F5.x(d[0])).attr('cy', (d) => F5.y(d[1])).attr('r', 4.3).attr('fill', cat(2)).attr('stroke', css('--surface')).attr('stroke-width', 1.1)
      .on('mousemove', (e, d) => tip(e, `cell at t = ${f2(tt)}: (${f2(d[0])}, ${f2(d[1])})`)).on('mouseleave', untip);
    const end = traj5[ST5];
    s5.set({ 'time t': f2(tt), 'MMD² of final cells to observed': f3(mmd2(end, pop5.Y)), 'spread: flow / observed': `${f2(spread(end))} / ${f2(spread(pop5.Y))}`, 'final cells in upper / lower state': `${end.filter((p) => p[1] > 0).length} / ${end.filter((p) => p[1] <= 0).length}` });
  }
  integ5(); paint5();
  const hSave = h5; h5 = 1.2; integ5(); const wide = mmd2(traj5[ST5], pop5.Y); h5 = 0.08; integ5(); const narrow = mmd2(traj5[ST5], pop5.Y); h5 = hSave; integ5(); paint5();
  prose(rb, `<p>At time 0.5 the straight paths cross in the middle of the plot, so the velocity at a given point is an average over several incompatible targets. The learned field is the regression solution, and integrating it separates the cells into the two observed states. At the widest kernel (1.2) the final MMD² to the observed cells is ${f3(wide)}, and at the narrowest (0.08) it is ${f3(narrow)}. The width plays the role of network smoothness: a very smooth field averages the branches, while a very narrow one reproduces the training pairs almost exactly. The narrow value gives the lowest MMD² here only because it is measured on the same training cells, so it says nothing about unseen perturbations. Compared with the MSE regressor of step 4, flow matching also needs no adversary and no simulation during training, only regression; sampling needs an ODE solve.</p>`);

  // ---- Step 6: metrics
  rb = rung(ch, 6, 'Metrics can be gamed: the Virtual Cell Challenge',
    `<p>The ${cite('vcc2025', '2025 challenge')} scored entries by <b>PDS</b> (perturbation discrimination: can a prediction be matched to its own perturbation, using L1 distance?), <b>DES</b> (are the differentially expressed genes recovered?) and <b>MAE</b> (mean absolute error). Arc's wrap-up reports that almost nobody beat the mean baseline on MAE and that L1-based PDS rewards inflating the magnitude of predictions. The figure multiplies one fixed set of predictions by a constant α; the biology of each prediction does not change.</p>`);
  const f6 = figure(rb, 'One set of predictions, rescaled by α', `${MK} test perturbations × ${MG} genes. The "model" predicts 0.4 × the true delta plus noise, a shrunk but informative stand-in. Dashed lines: the mean baseline (the average of ${MK} other perturbations, predicted for every test perturbation). Vertical marker: current α.`);
  toyBadge(f6, 'simplified PDS, not the official scoring code');
  let al6 = 1, st6 = 0.5, nz6 = 0.4, sim6, cv6, bs6;
  slider(f6.controls, { label: 'scale factor α', min: 0, max: 5, step: 0.1, value: al6, fmt: f1, onInput: (v) => { al6 = v; paint6(); } });
  slider(f6.controls, { label: 'shared stress', min: 0, max: 1.5, step: 0.05, value: st6, fmt: f2, onInput: (v) => { st6 = v; calc6(); } });
  slider(f6.controls, { label: 'prediction noise', min: 0.05, max: 0.6, step: 0.05, value: nz6, fmt: f2, onInput: (v) => { nz6 = v; calc6(); } });
  legend(f6.root, [['PDS (simplified)', cat(0)], ['Pearson-delta', cat(1)], ['MAE', cat(2)], ['mean baseline (dashed)', muted()]]);
  const s6 = stats(f6.root);
  const row6 = el('div', { class: 'fig-row' }); f6.body.append(row6);
  const a6 = el('div'), b6 = el('div'); row6.append(a6, b6);
  const F6a = frame(a6, { width: 500, height: 320, x: [0, 5], y: [0, 1], xLabel: 'scale factor α', yLabel: 'score (higher is better)', margin: { right: 60 } });
  const F6b = frame(b6, { width: 460, height: 320, x: [0, 5], y: [0, 1], xLabel: 'scale factor α', yLabel: 'MAE (lower is better)', margin: { right: 40 } });
  const ln6 = (F, data, key, color, dash) => F.layer(key).selectAll('path').data([data]).join('path').attr('fill', 'none').attr('stroke', color).attr('stroke-width', 2).attr('stroke-dasharray', dash ? '5 4' : null).attr('d', d3.line().x((d) => F.x(d.a)).y((d) => F.y(d.v)));
  const tag6 = (F, key, v, txt) => F.layer(key).selectAll('text').data([txt]).join('text').attr('x', F.w + 4).attr('y', F.y(v) + 4).attr('font-size', 11).attr('fill', ink2()).text((d) => d);
  function calc6() {
    sim6 = simMetrics(st6, nz6); cv6 = curvesOf(sim6); const MB = sim6.test.map(() => sim6.meanBase);
    bs6 = { pds: pds(MB, sim6.test), mae: d3.mean(MB, (p, k) => l1(p, sim6.test[k])), pr: d3.mean(MB, (p, k) => pearson(p, sim6.test[k])) };
    F6b.y.domain([0, Math.max(bs6.mae, d3.max(cv6, (d) => d.mae)) * 1.05]); F6b.g.select('.axis.y').call(d3.axisLeft(F6b.y).ticks(5)); paint6();
  }
  function paint6() {
    ln6(F6a, cv6.map((d) => ({ a: d.a, v: d.pds })), 'pds', cat(0)); ln6(F6a, cv6.filter((d) => d.a > 0).map((d) => ({ a: d.a, v: Math.max(0, d.pr) })), 'pr', cat(1));
    ln6(F6a, [{ a: 0, v: bs6.pds }, { a: 5, v: bs6.pds }], 'bp', cat(0), true); ln6(F6a, [{ a: 0, v: Math.max(0, bs6.pr) }, { a: 5, v: Math.max(0, bs6.pr) }], 'br', cat(1), true);
    ln6(F6b, cv6.map((d) => ({ a: d.a, v: d.mae })), 'mae', cat(2)); ln6(F6b, [{ a: 0, v: bs6.mae }, { a: 5, v: bs6.mae }], 'bm', cat(2), true);
    tag6(F6a, 'tp', cv6[cv6.length - 1].pds, 'PDS'); tag6(F6a, 'tr', Math.max(0, cv6[cv6.length - 1].pr) + (cv6[cv6.length - 1].pds - cv6[cv6.length - 1].pr > -0.06 && cv6[cv6.length - 1].pds - cv6[cv6.length - 1].pr < 0.06 ? -0.05 : 0), 'Pearson-δ'); tag6(F6b, 'tm', cv6[cv6.length - 1].mae, 'MAE');
    [F6a, F6b].forEach((F) => F.layer('mk').selectAll('line').data([al6]).join('line').attr('x1', (a) => F.x(a)).attr('x2', (a) => F.x(a)).attr('y1', 0).attr('y2', F.h).attr('stroke', ink2()).attr('stroke-width', 1.5));
    const m = metricsAt(sim6, al6), best = cv6.reduce((x, y) => (y.mae < x.mae ? y : x));
    const dots = (F, pts) => F.layer('dots').selectAll('circle').data(pts).join('circle').attr('cx', F.x(al6)).attr('cy', (d) => F.y(d.v)).attr('r', 5).attr('fill', (d) => d.c).attr('stroke', css('--surface')).attr('stroke-width', 1.5).on('mousemove', (e, d) => tip(e, `${d.n} at α = ${f1(al6)}: <b>${f3(d.v)}</b>`)).on('mouseleave', untip);
    dots(F6a, [{ v: m.pds, c: cat(0), n: 'PDS' }, ...(al6 > 0 ? [{ v: Math.max(0, m.pr), c: cat(1), n: 'Pearson-delta' }] : [])]); dots(F6b, [{ v: m.mae, c: cat(2), n: 'MAE' }]);
    s6.set({ 'PDS at α': f3(m.pds), 'Pearson-delta at α': al6 > 0 ? f3(m.pr) : 'undefined', 'MAE at α': f3(m.mae), 'α minimising MAE': f1(best.a), 'mean baseline PDS / Pearson-δ / MAE': `${f2(bs6.pds)} / ${f2(bs6.pr)} / ${f3(bs6.mae)}` });
  }
  calc6();
  const cv0 = curvesOf(simMetrics(0.5, 0.4)), at = (a) => cv0.find((d) => Math.abs(d.a - a) < 1e-9), bestMae = cv0.reduce((x, y) => (y.mae < x.mae ? y : x)), MB0 = simMetrics(0.5, 0.4), baseMae = d3.mean(MB0.test, (t) => l1(MB0.meanBase, t)), bsPr0 = d3.mean(MB0.test, (t) => pearson(MB0.meanBase, t));
  prose(rb, `<ul><li><b>Pearson-delta ignores scale:</b> it is ${f3(at(0.5).pr)} at α = 0.5 and ${f3(at(3).pr)} at α = 3.</li>
    <li><b>MAE penalises inflation:</b> it is lowest at α = ${f1(bestMae.a)} (${f3(bestMae.mae)}) and ${f3(at(3).mae)} at α = 3. The mean baseline's MAE is ${f3(baseMae)}${bestMae.mae > baseMae ? ', which this model does not beat at any α, the pattern reported for the real challenge' : ''}.</li>
    <li><b>PDS depends on α:</b> ${f3(at(0.5).pds)} at α = 0.5 and ${f3(at(3).pds)} at α = 3, although the information in the predictions is identical. With L1 distance, a large prediction is far from every true profile, and the ranking is decided by whether the <i>signs</i> agree with the right perturbation, so inflating costs little.</li>
    <li>The mean baseline has PDS = 0.5 (chance), since it predicts the same profile for every perturbation. Its Pearson-delta is ${f2(bsPr0)} against ${f2(at(1).pr)} for the model: with a shared stress response, correlation alone ranks the uninformed baseline ${bsPr0 > at(1).pr ? 'above' : 'below'} the informative model.</li></ul>
    <p>The ${cite('vcc2026', '2026 round')} is zero-shot on six new cell lines; its winners had not been announced when the research brief was compiled (2026-11-12), so how its metrics behaved is unknown.</p>`);
  quiz(rb, 'A leaderboard metric improves when every prediction is multiplied by 2. What does that indicate?', '<p>The metric rewards magnitude rather than biological content. Before trusting a headline score, rescale the predictions by 0.5 and 2 and recompute; a trustworthy metric should not move in the direction of inflation.</p>');

  // ---- Step 7: tensor factorisation
  rb = rung(ch, 7, 'Pooling experiments: perturbation × gene × context',
    `<p>A <b>Large Perturbation Model</b> (${cite('lpm', 'LPM')}) treats every experiment as a point in a three-way array: <b>what was done</b> (perturbation), <b>what was measured</b> (readout) and <b>where</b> (context, for example cell line). Each axis gets its own learned vector, and a value is predicted from the three vectors together, so experiments from different screens can be pooled and unmeasured combinations filled in. The simplest version of this idea is a <b>tensor factorisation</b>: value ≈ Σ<sub>k</sub> A[p,k]·B[g,k]·D[c,k]. LPM itself is a neural network; its results are the authors' own, and the research brief has found no independent head-to-head against the baselines of step 2.</p>`);
  const f7 = figure(rb, 'Filling in unmeasured perturbation × context combinations', `${TP} perturbations × ${TG} genes × ${TC} contexts, generated from ${TR} hidden programmes whose strength differs by context. Left: filled cells are held-out (perturbation, context) pairs, coloured by the Pearson r of the prediction; dashed outlines were measured (training). Right: held-out r against alternating-least-squares sweeps, with two baselines.`);
  toyBadge(f7, 'CP factorisation fitted by masked ALS, not LPM\'s network');
  let rk7 = 3, fr7 = 0.4, nz7 = 0.5, Y7, M7, fit7, bs7, sel7 = null;
  slider(f7.controls, { label: 'model rank', min: 1, max: 6, step: 1, value: rk7, onInput: (v) => { rk7 = v; refit7(); } });
  slider(f7.controls, { label: 'fraction measured', min: 0.2, max: 0.8, step: 0.05, value: fr7, fmt: pc, onInput: (v) => { fr7 = v; rebuild7(); } });
  slider(f7.controls, { label: 'noise', min: 0.1, max: 1.5, step: 0.1, value: nz7, fmt: f1, onInput: (v) => { nz7 = v; rebuild7(); } });
  legend(f7.root, [['factorised model', cat(0)], ['same perturbation, mean over measured contexts', cat(1)], ['same context, mean over measured perturbations', cat(2)]]);
  const s7 = stats(f7.root);
  const row7 = el('div', { class: 'fig-row' }); f7.body.append(row7);
  const a7 = el('div'), b7 = el('div'); row7.append(a7, b7);
  const cw7 = 14.5, hmW = 70 + TP * cw7 + 8, rh = 24;
  const hm7 = d3.select(a7).append('svg').attr('viewBox', `0 0 ${hmW} ${40 + TC * rh + 46}`).attr('width', hmW).attr('role', 'img');
  const hg7 = hm7.append('g').attr('transform', 'translate(70,26)');
  hm7.append('text').attr('x', 70).attr('y', 14).attr('font-size', 11).attr('fill', ink2()).text('perturbation →');
  d3.range(TC).forEach((c) => hm7.append('text').attr('x', 64).attr('y', 26 + c * rh + 16).attr('text-anchor', 'end').attr('font-size', 11).attr('fill', ink2()).text('context ' + 'ABCDEF'[c]));
  const col7 = sequential(-0.2, 1);
  [0, 0.25, 0.5, 0.75, 1].forEach((v, i) => { hm7.append('rect').attr('x', 70 + i * 40).attr('y', 26 + TC * rh + 12).attr('width', 38).attr('height', 9).attr('fill', col7(v)); hm7.append('text').attr('x', 70 + i * 40 + 19).attr('y', 26 + TC * rh + 34).attr('text-anchor', 'middle').attr('font-size', 10).attr('fill', ink2()).text(v); });
  hm7.append('text').attr('x', 70 + 5 * 40 + 6).attr('y', 26 + TC * rh + 21).attr('font-size', 10.5).attr('fill', ink2()).text('held-out r');
  const F7 = frame(b7, { width: 440, height: 300, x: [0, SW], y: [-0.2, 1], xLabel: 'ALS sweep', yLabel: 'mean held-out r', margin: { top: 26 } });
  const t7 = F7.g.append('text').attr('x', 2).attr('y', -10).attr('font-size', 11.5).attr('fill', ink2());
  function rebuild7() { Y7 = simTensor(nz7); M7 = maskTensor(fr7); bs7 = tensorBaselines(Y7, M7); sel7 = null; refit7(); }
  function refit7() {
    fit7 = fitTensor(Y7, M7, rk7);
    const rmap = new Map(fit7.held.map((d) => [d.p + ',' + d.c, d.r]));
    const cells = []; for (let p = 0; p < TP; p++) for (let c = 0; c < TC; c++) cells.push({ p, c, obs: M7[p][c], r: rmap.get(p + ',' + c) });
    hg7.selectAll('rect').data(cells).join('rect').attr('x', (d) => d.p * cw7).attr('y', (d) => d.c * rh).attr('width', cw7 - 1.5).attr('height', rh - 2).attr('rx', 2)  .attr('fill', (d) => (d.obs ? 'none' : col7(d.r))).attr('stroke', (d) => (d.obs ? muted() : 'none')).attr('stroke-dasharray', '2 2')
      .on('mousemove', (e, d) => tip(e, `perturbation ${d.p + 1}, context ${'ABCDEF'[d.c]}<br>${d.obs ? 'measured (training)' : `held out: r = <b>${f2(d.r)}</b>`}`)).on('mouseleave', untip);
    F7.layer('b').selectAll('path').data([[bs7.pm, cat(1)], [bs7.cm, cat(2)]]).join('path').attr('fill', 'none').attr('stroke', (d) => d[1]).attr('stroke-width', 2).attr('stroke-dasharray', '5 4').attr('d', (d) => d3.line()([[F7.x(0), F7.y(d[0])], [F7.x(SW), F7.y(d[0])]]));
    F7.layer('c').selectAll('path').data([fit7.hist]).join('path').attr('fill', 'none').attr('stroke', cat(0)).attr('stroke-width', 2.2).attr('d', d3.line().x((d, i) => F7.x(i)).y((d) => F7.y(Math.max(-0.2, d))));
    const fin = fit7.hist[SW]; t7.text(`after ${SW} sweeps: ${f2(fin)} (factorised) vs ${f2(bs7.pm)} / ${f2(bs7.cm)} (baselines)`);
    s7.set({ 'held-out pairs': fit7.held.length, 'factorised: held-out r': f3(fin), 'same-perturbation mean': f3(bs7.pm), 'same-context mean': f3(bs7.cm), 'training RMSE': f3(fit7.rmse) });
  }
  rebuild7();
  Y7 = simTensor(0.5); M7 = maskTensor(0.4); const b0 = tensorBaselines(Y7, M7), tr = [1, 3, 6].map((k) => fitTensor(Y7, M7, k)), lowM = (() => { const M = maskTensor(0.2); return { b: tensorBaselines(Y7, M), f: fitTensor(Y7, M, 3).hist[SW] }; })();
  Y7 = simTensor(nz7); M7 = maskTensor(fr7); bs7 = tensorBaselines(Y7, M7); fit7 = fitTensor(Y7, M7, rk7);
  prose(rb, `<p>At the default setting the factorised model reaches a held-out r of ${f2(tr[1].hist[SW])}, against ${f2(b0.pm)} for the mean of the same perturbation in other contexts and ${f2(b0.cm)} for the mean of other perturbations in the same context. ${tr[1].hist[SW] > b0.pm ? 'It beats the perturbation-mean baseline here because it also learns how strongly each context expresses each programme.' : 'It does not beat the perturbation-mean baseline here.'} Rank 1 gives ${f2(tr[0].hist[SW])}, rank 6 gives ${f2(tr[2].hist[SW])} (training RMSE ${f3(tr[1].rmse)} at rank 3 and ${f3(tr[2].rmse)} at rank 6): extra rank fits the training entries more closely, and the effect on held-out pairs is ${tr[2].hist[SW] < tr[1].hist[SW] ? 'a loss' : 'not a gain'}. With only 20% of pairs measured, the factorised model reaches ${f2(lowM.f)} and the perturbation-mean baseline ${f2(lowM.b.pm)}. The model can only fill in a context whose factors it has estimated; a context with no measurements needs another source of information.</p>`);

  // =====================================================================================
  // 5. PITFALLS
  // =====================================================================================
  ch = chapter(main, 'pitfalls', 'Pitfalls', 'Common problems when building or judging perturbation-prediction models.');
  const ceil0 = res0.by('ceil'), sharedDef = sharedShare(res0);
  cards(ch, [
    { tag: 'Symptom: a high correlation from a trivial model', title: 'Shared response inflates metrics', body: `In the toy screen ${pc(sharedDef)} of the squared delta is common to all perturbations, and the training mean reaches r = <b>${f2(res0.by('mean').pdCtrl)}</b> against control but <b>${f2(res0.by('mean').pdCent)}</b> against the centroid. <b>Fix:</b> report a metric centred on the perturbed centroid (Systema) or restricted to the top differentially expressed genes, and always list the mean baseline.` },
    { tag: 'Symptom: a metric moves when predictions are rescaled', title: 'Metric gaming', body: `Multiplying the same predictions by 3 changes PDS from ${f2(at(0.5).pds)} (at 0.5) to ${f2(at(3).pds)}, while Pearson-delta stays at ${f2(at(3).pr)}. <b>Fix:</b> report several metrics together (scale-sensitive, rank-based, correlation), and rescale predictions as a sanity check.` },
    { tag: 'Symptom: strong results on seen perturbations only', title: 'Splitting by cell instead of by perturbation', body: 'Cells from one perturbation are near-replicates. A random split across cells places them in both training and test sets, and the model is rewarded for memorisation. <b>Fix:</b> split by perturbation (and by context for transfer claims); for doubles, state whether the singles were seen.' },
    { tag: 'Symptom: a realistic pseudobulk, unrealistic cells', title: 'Averages hide heterogeneity', body: `In the toy, the mean perturbed cell has no neighbour closer than ${f2(nearest(popTwo.Y))} units, and in the partial-response population ${f2(nearest(popPart.Y))}. A model scored only on pseudobulk can look correct while predicting a phenotype no cell has. <b>Fix:</b> compare distributions (MMD, energy distance) and check the predicted variance.` },
    { tag: 'Symptom: no model reaches r = 1', title: 'Noise ceiling', body: `Observed deltas contain noise. In the toy, even the true noise-free delta correlates at <b>${f2(ceil0.pdCtrl)}</b> against control and <b>${f2(ceil0.pdCent)}</b> against the centroid. <b>Fix:</b> estimate the replicate ceiling from technical replicates or split-half deltas and report scores relative to it.` },
    { tag: 'Symptom: training fit improves, held-out fit does not', title: 'Capacity beyond the data', body: `Rank 6 reaches a training RMSE of ${f3(tr[2].rmse)} against ${f3(tr[1].rmse)} at rank 3, with held-out r of ${f2(tr[2].hist[SW])} against ${f2(tr[1].hist[SW])}. With a few dozen perturbations, flexible models fit noise. <b>Fix:</b> regularise, choose capacity by held-out perturbations, and compare with the linear baseline.` },
    { tag: 'Symptom: a headline gain from one source', title: 'Developer-reported benchmarks', body: 'Claims such as State being the "first to consistently beat linear baselines" are self-reported. Independent benchmarks (Ahlmann-Eltze et al.; PertEval-scFM; PerturBench; VCBench) found simple baselines matching or beating foundation models on most axes. <b>Fix:</b> read results from independent benchmarks, and re-run the baselines on the data of interest.' },
  ]);

  // =====================================================================================
  // 6. RECAP
  // =====================================================================================
  ch = chapter(main, 'recap', 'Recap', 'Key points.');
  recap(ch, [
    'A perturbation model predicts the <b>delta</b> (perturbed minus control) for a perturbation that was not measured; scoring must use perturbations held out of training.',
    'Most of the delta is a <b>shared response</b>. A model that predicts only the training mean reaches a high correlation against control, so metrics must be centred on the perturbed centroid or otherwise remove it.',
    'The <b>training mean, additive and linear baselines</b> are the minimum comparison. Independent benchmarks report that they match or beat large models on unseen perturbations; the quality of the perturbation embedding matters more than the architecture.',
    'When cells respond differently, a loss on single cells learns the mean cell. <b>Distribution-level losses</b> (MMD in State) and <b>flow matching</b> (CellFlow, PRiMeFlow) predict a population.',
    'Leaderboard metrics can be gamed: PDS changes with prediction scale, and MAE favours small predictions. Rescale predictions as a check and report several metrics.',
    '<b>Large perturbation models</b> pool experiments along perturbation, readout and context axes; a tensor factorisation is the simplest version. Their gains over baselines are mostly developer-reported.',
    'Estimate the <b>noise ceiling</b>, split by perturbation, and judge differences against the spread across random screens.',
    'Treat self-reported claims as hypotheses until an independent benchmark reproduces them.',
  ]);
  prose(ch, '<h3>In code</h3><p>The baselines and the centroid-centred metric take a few lines on a pseudobulk matrix of deltas (perturbations × genes). The scale check at the end is a practical test of any metric.</p>');
  code(ch, {
    Python: `import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA

# Y_train: (n_train, n_genes) deltas;  E_train, E_test: perturbation embeddings;  Y_test: held-out deltas
centroid = Y_train.mean(axis=0)                      # baseline 1: training mean
# baseline 2 (doubles): add the measured single deltas  ->  Y_single[a] + Y_single[b]
pca = PCA(n_components=8).fit(Y_train)               # baseline 3: gene axes + ridge on the embedding
ridge = Ridge(alpha=0.5).fit(E_train, pca.transform(Y_train))
pred_lin = pca.inverse_transform(ridge.predict(E_test))

def pearson_delta(pred, obs, ref):                   # ref = 0 (vs control) or centroid (Systema-style)
    return np.mean([pearsonr(p - ref, o - ref)[0] for p, o in zip(pred, obs)])

for name, pred in [("mean", np.tile(centroid, (len(Y_test), 1))), ("linear", pred_lin)]:
    print(name, pearson_delta(pred, Y_test, 0), pearson_delta(pred, Y_test, centroid),
          np.abs(pred - Y_test).mean())

# scale check: a trustworthy metric should not improve when predictions are inflated
for a in (0.5, 1, 2):
    print(a, np.abs(a * pred_lin - Y_test).mean())`,
    R: `# Y_train: n_train x n_genes deltas;  E_train, E_test: perturbation embeddings;  Y_test: held-out deltas
centroid <- colMeans(Y_train)                        # baseline 1: training mean
pc <- prcomp(Y_train, rank. = 8)                     # baseline 3: gene axes + ridge on the embedding
fit <- glmnet::glmnet(E_train, pc$x, family = "mgaussian", alpha = 0, lambda = 0.5)
scores <- predict(fit, E_test)[, , 1]
pred_lin <- sweep(scores %*% t(pc$rotation), 2, pc$center, "+")

pearson_delta <- function(pred, obs, ref = 0)        # ref = 0 (vs control) or centroid (Systema-style)
  mean(sapply(seq_len(nrow(obs)), function(i) cor(pred[i, ] - ref, obs[i, ] - ref)))
pred_mean <- matrix(centroid, nrow(Y_test), length(centroid), byrow = TRUE)
c(mean_vs_ctrl = pearson_delta(pred_mean, Y_test), lin_vs_cent = pearson_delta(pred_lin, Y_test, centroid))

# scale check: MAE should not improve when predictions are inflated
sapply(c(0.5, 1, 2), function(a) mean(abs(a * pred_lin - Y_test)))`,
  });
  seeAlso(main, [['frontier', 'foundation-models', 'Single-cell foundation models'], ['frontier', 'landscape', 'The 2024–26 landscape'], ['concepts', 'regularization', 'Regularisation'], ['concepts', 'pca', 'PCA'], ['concepts', 'neural-network', 'Neural networks'], ['concepts', 'metrics', 'Evaluating a classifier'], ['omics', 'sc-pseudobulk', 'Pseudobulk differential expression']]);
}

// ---------------------------------------------------------------------------------------------------------
/** Three-box schematic: measured perturbations → model → held-out perturbation scored. */
function schematic(res, nTrain) {
  const wrap = el('div', { class: 'figure' });
  wrap.append(el('div', { class: 'fig-title' }, 'From measured perturbations to a prediction for an unseen one (toy screen)'));
  const svg = d3.select(wrap).append('svg').attr('viewBox', '0 0 980 210').attr('width', 980).attr('role', 'img').attr('aria-label', 'training perturbations, model, held-out perturbation');
  const boxes = [
    { t: '① Measured (training)', lines: [`${nTrain} knockdowns × ${G} genes`, 'each as a delta vs control', '+ an embedding per perturbation', `+ ${ND / 2} measured doubles (step 2)`] },
    { t: '② Model', lines: ['Δ̂ = f(perturbation)', 'mean · additive · linear', 'small neural network', 'distribution-level models'] },
    { t: '③ Held-out perturbation', lines: [`${G} predicted deltas`, 'compared with observed', `mean baseline  r = ${f2(res.by('mean').pdCtrl)}`, `linear model   r = ${f2(res.by('lin').pdCtrl)}`, '(vs control; step 3)'] },
  ];
  const BW0 = 280, gap = 54;
  boxes.forEach((b, i) => {
    const x0 = 12 + i * (BW0 + gap), g = svg.append('g').attr('transform', `translate(${x0},12)`);
    g.append('rect').attr('width', BW0).attr('height', 186).attr('rx', 10).attr('fill', css('--surface-2')).attr('stroke', css('--axis'));
    g.append('text').attr('x', 14).attr('y', 26).attr('font-weight', 700).attr('font-size', 14).attr('fill', ink()).text(b.t);
    b.lines.forEach((l, j) => g.append('text').attr('x', 16).attr('y', 58 + j * 25).attr('font-family', css('--mono')).attr('font-size', 13).attr('fill', ink2()).attr('xml:space', 'preserve').text(l));
    if (i < boxes.length - 1) { const ax = x0 + BW0 + 8, ay = 12 + 93; svg.append('path').attr('d', `M${ax},${ay} L${ax + gap - 18},${ay} M${ax + gap - 24},${ay - 6} L${ax + gap - 18},${ay} L${ax + gap - 24},${ay + 6}`).attr('fill', 'none').attr('stroke', ink2()).attr('stroke-width', 2); }
  });
  return wrap;
}
