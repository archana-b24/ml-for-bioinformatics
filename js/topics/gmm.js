// Gaussian mixture models fitted by EM: ellipses for covariances, soft responsibilities as colour mixing,
// log-likelihood readout, and a side-by-side comparison with k-means on elongated clusters.
import { header, figure, slider, select, button, stats, section, steps, callout, math, seeAlso, animate, legend, tip, untip, el } from '../lib/ui.js';
import { frame, cat, css, ink, muted } from '../lib/plot.js';
import { rng, range, eig2 } from '../lib/data.js';
const d3 = window.d3;

const DATASETS = { stripes: 'Three parallel stripes', unequal: 'One wide + two tight blobs', overlap: 'Two overlapping blobs' };

export function makeData(kind) {
  const out = [];
  if (kind === 'stripes') {
    const r = rng(31), th = (35 * Math.PI) / 180, u = [Math.cos(th), Math.sin(th)], v = [-Math.sin(th), Math.cos(th)];
    [-0.42, 0, 0.42].forEach((off, g) => {
      for (let i = 0; i < 100; i++) { const a = r.normal(0, 0.36), b = r.normal(off, 0.06); out.push({ x: a * u[0] + b * v[0], y: a * u[1] + b * v[1], label: g }); }
    });
  } else if (kind === 'unequal') {
    const r = rng(12);
    for (let i = 0; i < 160; i++) out.push({ x: r.normal(-0.45, 0.28), y: r.normal(0, 0.28), label: 0 });
    for (let i = 0; i < 40; i++) out.push({ x: r.normal(0.6, 0.08), y: r.normal(0.5, 0.08), label: 1 });
    for (let i = 0; i < 40; i++) out.push({ x: r.normal(0.6, 0.08), y: r.normal(-0.5, 0.08), label: 2 });
  } else {
    const r = rng(33);
    for (let i = 0; i < 120; i++) out.push({ x: r.normal(-0.3, 0.25), y: r.normal(0, 0.25), label: 0 });
    for (let i = 0; i < 120; i++) out.push({ x: r.normal(0.3, 0.25), y: r.normal(0.05, 0.25), label: 1 });
  }
  return out;
}

// ---------- EM for a 2-D Gaussian mixture ----------
const LOG2PI = Math.log(2 * Math.PI);
export function initGMM(P, K, seed) {
  // k-means++-style spread-out starting means; round covariances; equal weights
  const r = rng(seed * 101 + K), n = P.length, mu = [];
  const f = P[r.int(n)]; mu.push([f.x, f.y]);
  const D = new Float64Array(n).fill(Infinity);
  while (mu.length < K) {
    const c = mu[mu.length - 1]; let tot = 0;
    for (let i = 0; i < n; i++) { D[i] = Math.min(D[i], (P[i].x - c[0]) ** 2 + (P[i].y - c[1]) ** 2); tot += D[i]; }
    let u = r() * tot, pick = n - 1; for (let i = 0; i < n; i++) { u -= D[i]; if (u <= 0) { pick = i; break; } }
    mu.push([P[pick].x, P[pick].y]);
  }
  return { K, pi: new Array(K).fill(1 / K), mu, S: mu.map(() => [0.05, 0, 0.05]), R: new Float64Array(n * K), ll: -Infinity };
}
/** E-step: fills responsibilities, returns total log-likelihood. */
export function eStep(P, M) {
  const { K, pi, mu, S, R } = M, n = P.length;
  const pre = S.map(([a, b, d], k) => { const det = a * d - b * b; return { ia: d / det, ib: -b / det, id: a / det, c: Math.log(pi[k]) - LOG2PI - 0.5 * Math.log(det) }; });
  let ll = 0; const lp = new Float64Array(K);
  for (let i = 0; i < n; i++) {
    let mx = -Infinity;
    for (let k = 0; k < K; k++) {
      const dx = P[i].x - mu[k][0], dy = P[i].y - mu[k][1], q = pre[k];
      lp[k] = q.c - 0.5 * (q.ia * dx * dx + 2 * q.ib * dx * dy + q.id * dy * dy);
      if (lp[k] > mx) mx = lp[k];
    }
    let s = 0; for (let k = 0; k < K; k++) s += Math.exp(lp[k] - mx);
    const lse = mx + Math.log(s); ll += lse;
    for (let k = 0; k < K; k++) R[i * K + k] = Math.exp(lp[k] - lse);
  }
  M.ll = ll; return ll;
}
/** M-step: weighted means, covariances and mixing weights from the responsibilities. */
export function mStep(P, M) {
  const { K, R } = M, n = P.length;
  for (let k = 0; k < K; k++) {
    let nk = 0, sx = 0, sy = 0;
    for (let i = 0; i < n; i++) { const w = R[i * K + k]; nk += w; sx += w * P[i].x; sy += w * P[i].y; }
    nk = Math.max(nk, 1e-9);
    const mx = sx / nk, my = sy / nk; let a = 0, b = 0, d = 0;
    for (let i = 0; i < n; i++) { const w = R[i * K + k], dx = P[i].x - mx, dy = P[i].y - my; a += w * dx * dx; b += w * dx * dy; d += w * dy * dy; }
    M.mu[k] = [mx, my]; M.S[k] = [a / nk + 1e-4, b / nk, d / nk + 1e-4]; M.pi[k] = nk / n;
  }
}
export function fitGMM(P, K, seed, maxIt = 500) {
  const M = initGMM(P, K, seed); let prev = -Infinity, it = 0;
  for (; it < maxIt; it++) { const ll = eStep(P, M); if (ll - prev < 1e-5) break; prev = ll; mStep(P, M); }
  return { M, it };
}
export function kmeansBest(P, K, nInit = 10) {
  let best = null;
  for (let s = 1; s <= nInit; s++) {
    const M = initGMM(P, K, 500 + s), C = M.mu.map((m) => m.slice()), lab = new Int32Array(P.length).fill(-1);
    for (let it = 0; it < 200; it++) {
      let ch = 0;
      P.forEach((p, i) => { let b = 0, bd = Infinity; C.forEach((c, j) => { const dd = (p.x - c[0]) ** 2 + (p.y - c[1]) ** 2; if (dd < bd) { bd = dd; b = j; } }); if (lab[i] !== b) { ch++; lab[i] = b; } });
      if (!ch) break;
      C.forEach((c, j) => { let sx = 0, sy = 0, m = 0; P.forEach((p, i) => { if (lab[i] === j) { sx += p.x; sy += p.y; m++; } }); if (m) { c[0] = sx / m; c[1] = sy / m; } });
    }
    let inert = 0; P.forEach((p, i) => { inert += (p.x - C[lab[i]][0]) ** 2 + (p.y - C[lab[i]][1]) ** 2; });
    if (!best || inert < best.inert) best = { lab, C, inert };
  }
  return best;
}
/** Fraction of points whose cluster matches the true label, under the best relabelling. */
export function matchAcc(truth, lab, K) {
  const perms = (a) => (a.length <= 1 ? [a] : a.flatMap((x, i) => perms([...a.slice(0, i), ...a.slice(i + 1)]).map((p) => [x, ...p])));
  let best = 0;
  for (const p of perms(range(K))) { let c = 0; for (let i = 0; i < truth.length; i++) if (p[lab[i]] === truth[i]) c++; best = Math.max(best, c); }
  return best / truth.length;
}
const argmaxResp = (M, n) => { const lab = new Int32Array(n); for (let i = 0; i < n; i++) { let b = 0; for (let k = 1; k < M.K; k++) if (M.R[i * M.K + k] > M.R[i * M.K + b]) b = k; lab[i] = b; } return lab; };

function ellipsePts(mu, S, c) {
  const { vals, vecs } = eig2(S[0], S[1], S[2]);
  const a = Math.sqrt(Math.max(vals[0], 0)) * c, b = Math.sqrt(Math.max(vals[1], 0)) * c;
  return d3.range(0, 2 * Math.PI + 0.01, Math.PI / 40).map((t) => ({
    x: mu[0] + a * Math.cos(t) * vecs[0][0] + b * Math.sin(t) * vecs[1][0],
    y: mu[1] + a * Math.cos(t) * vecs[0][1] + b * Math.sin(t) * vecs[1][1],
  }));
}

export default async function render(main) {
  header(main, 'Gaussian mixtures & EM',
    'A Gaussian mixture model says the data were drawn from k overlapping bell-shaped clouds, each with its own centre, <b>shape</b> and size. Expectation–maximisation (EM) fits them by alternating: work out how much each cloud is responsible for each point, then refit each cloud to the points it is responsible for. Every point gets a <b>probability</b> of belonging to each cluster, not a hard label.');

  const fig = figure(main, 'EM fitting a mixture, one iteration at a time',
    'Ellipses are each component\'s 1σ and 2σ contours; their tilt and stretch come from the covariance matrix. Each point\'s colour is a <b>mix</b> of the component colours, weighted by its responsibilities: a pure colour means "certainly this cluster", a muddy blend means "torn between two". Faded points are uncertain (largest responsibility below 0.9). Hover a point to see its probabilities.');
  let kind = 'stripes', P = makeData(kind), K = 3, seed = 2, M = null, hist = [], stop = null, conv = false;

  select(fig.controls, { label: 'Data', value: kind, options: Object.entries(DATASETS), onChange: (v) => { kind = v; P = makeData(v); K = v === 'overlap' ? 2 : 3; kS.set(K); reset(); } });
  const kS = slider(fig.controls, { label: 'k', min: 1, max: 5, value: K, onInput: (v) => { K = v; reset(); } });
  slider(fig.controls, { label: 'Init seed', min: 1, max: 10, value: 2, onInput: (v) => { seed = v; reset(); } });
  button(fig.controls, 'Run', () => run(), true);
  button(fig.controls, 'Step', () => { halt(); stepOnce(); paint(); });
  button(fig.controls, 'Reset', () => reset());
  const readout = stats(fig.root);

  const row = el('div', { class: 'fig-row' }); fig.body.append(row);
  const L = el('div', { style: 'flex: 1.6 1 420px' }), R = el('div'); row.append(L, R);
  const F = frame(L, { width: 560, height: 450, x: [-1.35, 1.35], y: [-1.1, 1.1], xLabel: 'feature 1', yLabel: 'feature 2' });
  F.layer('pts'); F.layer('ell'); F.layer('mu');
  const T = frame(R, { width: 340, height: 300, x: [0, 20], y: [0, 1], xLabel: 'EM iteration', yLabel: 'log-likelihood' });
  R.append(el('div', { class: 'fig-caption' }, 'Total log-likelihood after each iteration. EM never makes it worse.'));

  function halt() { if (stop) { stop(); stop = null; } }
  function reset() {
    halt(); M = initGMM(P, K, seed); conv = false;
    eStep(P, M); hist = [{ t: 0, ll: M.ll }];
    paint();
  }
  // one EM iteration: M-step from current responsibilities, then E-step to get new responsibilities + LL
  function stepOnce() {
    if (conv) return false;
    const prev = M.ll; mStep(P, M); eStep(P, M);
    hist.push({ t: hist.length, ll: M.ll });
    if (M.ll - prev < 1e-5 || hist.length > 500) conv = true;
    return !conv;
  }
  function run() {
    halt(); let acc = 0;
    stop = animate((dt) => { acc += dt; if (acc < 90) return true; acc = 0; const go = stepOnce(); paint(); if (!go) stop = null; return go; });
  }

  function paint() {
    const n = P.length, cols = range(K).map((k) => d3.rgb(cat(k)));
    const mix = (i) => { let r = 0, g = 0, b = 0; for (let k = 0; k < K; k++) { const w = M.R[i * K + k]; r += w * cols[k].r; g += w * cols[k].g; b += w * cols[k].b; } return `rgb(${r | 0},${g | 0},${b | 0})`; };
    const maxR = (i) => { let m = 0; for (let k = 0; k < K; k++) m = Math.max(m, M.R[i * K + k]); return m; };
    F.layer('pts').selectAll('circle').data(P).join('circle').attr('cx', (d) => F.x(d.x)).attr('cy', (d) => F.y(d.y)).attr('r', 4.2)
      .attr('fill', (d, i) => mix(i)).attr('fill-opacity', (d, i) => (maxR(i) < 0.9 ? 0.45 : 1)).attr('stroke', css('--surface')).attr('stroke-width', 1)
      .on('mousemove', (e, d) => { const i = P.indexOf(d); tip(e, range(K).map((k) => `component ${k + 1}: <b>${(M.R[i * K + k] * 100).toFixed(1)}%</b>`).join('<br>')); })
      .on('mouseleave', untip);
    const ells = []; range(K).forEach((k) => [1, 2].forEach((c) => ells.push({ k, c, pts: ellipsePts(M.mu[k], M.S[k], c) })));
    F.layer('ell').selectAll('path').data(ells).join('path').attr('fill', 'none').attr('stroke', (d) => cat(d.k)).attr('stroke-width', (d) => (d.c === 1 ? 2.5 : 1.5))
      .attr('stroke-dasharray', (d) => (d.c === 2 ? '5 4' : null)).attr('d', (d) => d3.line().x((p) => F.x(p.x)).y((p) => F.y(p.y))(d.pts));
    F.layer('mu').selectAll('circle').data(M.mu).join('circle').attr('cx', (m) => F.x(m[0])).attr('cy', (m) => F.y(m[1])).attr('r', 7)
      .attr('fill', (m, k) => cat(k)).attr('stroke', ink()).attr('stroke-width', 2)
      .on('mousemove', (e, m) => { const k = M.mu.indexOf(m); tip(e, `component ${k + 1}<br>weight π = ${M.pi[k].toFixed(2)}`); }).on('mouseleave', untip);

    T.x.domain([0, Math.max(20, hist.length)]);
    const ext = d3.extent(hist, (d) => d.ll), pad = (ext[1] - ext[0]) * 0.08 || 1;
    T.y.domain([ext[0] - pad, ext[1] + pad]);
    T.g.select('.axis.x').call(d3.axisBottom(T.x).ticks(5)); T.g.select('.axis.y').call(d3.axisLeft(T.y).ticks(5));
    T.g.select('.gridline').call(d3.axisLeft(T.y).ticks(5).tickSize(-T.w).tickFormat(''));
    T.layer('l').selectAll('path').data([hist]).join('path').attr('fill', 'none').attr('stroke', cat(0)).attr('stroke-width', 2)
      .attr('d', d3.line().x((d) => T.x(d.t)).y((d) => T.y(d.ll)));
    let unc = 0; for (let i = 0; i < n; i++) if (maxR(i) < 0.9) unc++;
    readout.set({
      'EM iterations': hist.length - 1,
      status: conv ? 'converged' : 'running',
      'log-likelihood': M.ll.toFixed(1),
      'uncertain points (max p < 0.9)': unc,
      'agreement with true groups': (matchAcc(P.map((p) => p.label), argmaxResp(M, n), Math.max(K, d3.max(P, (p) => p.label) + 1)) * 100).toFixed(0) + '%',
    });
  }
  reset();

  // ---------- comparison: k-means vs GMM on the stripes ----------
  const fig2 = figure(main, 'Same stripes, k = 3: k-means vs a Gaussian mixture',
    'k-means (best of 10 starts) can only draw straight boundaries halfway between round centres, so it chops across the stripes. The mixture (best log-likelihood of 10 starts) learns three long, thin, tilted covariances and recovers the stripes.');
  const S = makeData('stripes'), truth = S.map((p) => p.label);
  const km = kmeansBest(S, 3);
  let gm = null; for (let s = 1; s <= 10; s++) { const f = fitGMM(S, 3, s); if (!gm || f.M.ll > gm.M.ll) gm = f; }
  const gLab = argmaxResp(gm.M, S.length);
  const row2 = el('div', { class: 'fig-row' }); fig2.body.append(row2);
  [['k-means', km.lab, null], ['Gaussian mixture', gLab, gm.M]].forEach(([name, lab, MM]) => {
    const box = el('div'); row2.append(box);
    const Fr = frame(box, { width: 460, height: 380, x: [-1.35, 1.35], y: [-1.1, 1.1], margin: { top: 30 } });
    Fr.svg.append('text').attr('x', 230).attr('y', 18).attr('text-anchor', 'middle').attr('font-weight', 600).attr('font-size', 14).attr('fill', ink())
      .text(`${name}: ${(matchAcc(truth, lab, 3) * 100).toFixed(0)}% of points in their true stripe`);
    if (MM) range(3).forEach((k) => Fr.layer('e').append('path').attr('fill', 'none').attr('stroke', cat(k)).attr('stroke-width', 2)
      .attr('d', d3.line().x((p) => Fr.x(p.x)).y((p) => Fr.y(p.y))(ellipsePts(MM.mu[k], MM.S[k], 2))));
    else {
      const vor = d3.Delaunay.from(km.C, (c) => Fr.x(c[0]), (c) => Fr.y(c[1])).voronoi([0, 0, Fr.w, Fr.h]);
      range(3).forEach((k) => Fr.layer('e').append('path').attr('d', vor.renderCell(k)).attr('fill', cat(k)).attr('fill-opacity', 0.1).attr('stroke', muted()));
    }
    Fr.layer('p').selectAll('circle').data(S).join('circle').attr('cx', (d) => Fr.x(d.x)).attr('cy', (d) => Fr.y(d.y)).attr('r', 3.8)
      .attr('fill', (d, i) => cat(lab[i])).attr('stroke', css('--surface')).attr('stroke-width', 1)
      .on('mousemove', (e, d) => tip(e, `true stripe ${d.label + 1}<br>${name} cluster ${lab[S.indexOf(d)] + 1}`)).on('mouseleave', untip);
  });

  section(main, 'What to notice', `<ul>
    <li>Press <b>Run</b> on the stripes. The three round starting blobs stretch and rotate into thin ellipses as EM discovers the shape. The log-likelihood climbs steeply, then flattens; it never goes down.</li>
    <li>Early on many points are faded and muddy (uncertain). As the ellipses lock onto the stripes every point becomes a pure colour: in the converged fit (seed 2, 39 iterations) all 300 points end above 90% certainty.</li>
    <li>Below: k-means puts 70% of the stripe points in the right group; the mixture gets 100%. The difference is entirely the covariance: k-means is a Gaussian mixture whose clusters are forced to be round, equal-sized and hard-assigned.</li>
    <li><b>One wide + two tight blobs</b>: the mixture gives the wide cluster a big ellipse and keeps its edge points (100% correct, where k-means managed 95%).</li>
    <li><b>Two overlapping blobs</b>: points in the middle stay a blend no matter how long you run (83 of 240 end below 90% certainty). That is honest: the data genuinely cannot say which group they came from, and a hard label would hide it.</li>
    <li>Set <b>Init seed</b> to 1 on the stripes: EM converges to a <i>local</i> maximum, log-likelihood −132.2 instead of −14.5, with only 60% of points in their true stripe. On the wide + tight data, seeds 5, 6, 7 and 9 also get stuck. Like k-means, EM needs several starts.</li>
  </ul>`);

  steps(main, 'How it works', [
    'Choose k and start each component with a mean (here spread out like k-means++), a round covariance and equal weight π = 1/k.',
    '<b>E-step</b>: for every point, compute each component\'s density at that point times its weight, and normalise across components. These are the <b>responsibilities</b> r<sub>ik</sub>: the probability that point i came from component k.',
    '<b>M-step</b>: refit each component as a weighted average: the mean is the r-weighted mean of the points, the covariance the r-weighted spread, and the weight π<sub>k</sub> the average responsibility.',
    'Repeat. Each iteration is guaranteed not to decrease the log-likelihood. Stop when it stops improving.',
    'To choose k, compare fits with an information criterion (BIC) rather than raw likelihood, which always rises with k.',
  ]);
  math(main, 'p(x) = Σ_k π_k · N(x | μ_k, Σ_k)\nE:  r_ik = π_k N(x_i | μ_k, Σ_k) / Σ_j π_j N(x_i | μ_j, Σ_j)\nM:  N_k = Σ_i r_ik,   μ_k = Σ_i r_ik x_i / N_k,   Σ_k = Σ_i r_ik (x_i − μ_k)(x_i − μ_k)ᵀ / N_k,   π_k = N_k / n\nlog-likelihood = Σ_i log p(x_i)        k-means = the limit Σ_k = σ²I with σ → 0');

  callout(main, '', 'In biology', 'Mixtures are everywhere in omics, usually in one dimension: <b>flow / mass cytometry</b> gating (FlowSOM metaclusters and older GMM gating), calling <b>cell hashing and demultiplexing</b> barcodes (a two-component mixture of background vs signal counts per hashtag, as in <code>HTODemux</code> and <code>hashedDrops</code>), separating real cells from ambient RNA by total UMI, and <b>bimodal gene expression</b> (on vs off). Cluster-aware spatial methods such as BayesSpace and many copy-number callers are mixtures with extra structure. Responsibilities give a principled "uncertain" label instead of forcing every cell into a type.');
  callout(main, 'warn', 'Pitfalls', '<ul><li>Full covariance matrices need many parameters (d(d+1)/2 per component). In 50 PCA dimensions that is 1,275 per cluster, so constrain them (diagonal or shared) or reduce dimension first.</li><li>A component can collapse onto a single point and drive the likelihood to infinity. Real implementations add a small value to the diagonal (as here, 10⁻⁴).</li><li>Local maxima: use several starts, or start from k-means.</li><li>The clusters are only as Gaussian as your data. A banana-shaped population gets split into several ellipses.</li></ul>');
  seeAlso(main, [['concepts', 'kmeans', 'k-means'], ['concepts', 'probability-mle', 'Likelihood & Bayes'], ['concepts', 'dbscan', 'DBSCAN'], ['concepts', 'vae', 'Variational autoencoders']]);
}
