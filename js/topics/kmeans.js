// k-means: animated assign / update steps with Voronoi regions, init choice, elbow + silhouette, and failure cases.
import { header, figure, slider, select, button, stats, section, steps, callout, math, seeAlso, animate, legend, tip, untip, el } from '../lib/ui.js';
import { frame, cat, css, ink, muted } from '../lib/plot.js';
import { rng, moons, range } from '../lib/data.js';
const d3 = window.d3;

const DATASETS = {
  blobs: { name: 'Four blobs', k: 4 },
  unequal: { name: 'Unequal spread (failure case)', k: 3 },
  moons: { name: 'Two moons (failure case)', k: 2 },
};

export function makeData(kind) {
  if (kind === 'moons') return moons(240, 0.07, 2);
  const out = [];
  if (kind === 'blobs') {
    const r = rng(11), C = [[-0.6, 0.5], [0.55, 0.6], [-0.45, -0.55], [0.6, -0.4]];
    for (let i = 0; i < 240; i++) { const c = i % 4; out.push({ x: r.normal(C[c][0], 0.17), y: r.normal(C[c][1], 0.17), label: c }); }
  } else {
    const r = rng(12);
    for (let i = 0; i < 160; i++) out.push({ x: r.normal(-0.45, 0.28), y: r.normal(0, 0.28), label: 0 });
    for (let i = 0; i < 40; i++) out.push({ x: r.normal(0.6, 0.08), y: r.normal(0.5, 0.08), label: 1 });
    for (let i = 0; i < 40; i++) out.push({ x: r.normal(0.6, 0.08), y: r.normal(-0.5, 0.08), label: 2 });
  }
  return out;
}

// ---------- k-means core ----------
const d2 = (a, b) => (a.x - b.x) ** 2 + (a.y - b.y) ** 2;
export function initCentroids(P, k, method, seed) {
  const r = rng(seed * 7919 + k * 31 + (method === 'random' ? 0 : 1));
  const n = P.length;
  if (method === 'random') return r.shuffle(range(n)).slice(0, k).map((i) => ({ x: P[i].x, y: P[i].y }));
  const first = P[r.int(n)];
  const C = [{ x: first.x, y: first.y }];
  const D = new Float64Array(n).fill(Infinity);
  while (C.length < k) {
    const c = C[C.length - 1]; let tot = 0;
    for (let i = 0; i < n; i++) { D[i] = Math.min(D[i], d2(P[i], c)); tot += D[i]; }
    let u = r() * tot, pick = n - 1;
    for (let i = 0; i < n; i++) { u -= D[i]; if (u <= 0) { pick = i; break; } }
    C.push({ x: P[pick].x, y: P[pick].y });
  }
  return C;
}
function assign(P, C, lab) {
  let changed = 0;
  for (let i = 0; i < P.length; i++) {
    let best = 0, bd = Infinity;
    for (let j = 0; j < C.length; j++) { const d = d2(P[i], C[j]); if (d < bd) { bd = d; best = j; } }
    if (lab[i] !== best) { changed++; lab[i] = best; }
  }
  return changed;
}
function update(P, C, lab) {
  const sx = new Float64Array(C.length), sy = new Float64Array(C.length), cnt = new Float64Array(C.length);
  for (let i = 0; i < P.length; i++) { sx[lab[i]] += P[i].x; sy[lab[i]] += P[i].y; cnt[lab[i]]++; }
  return C.map((c, j) => (cnt[j] ? { x: sx[j] / cnt[j], y: sy[j] / cnt[j] } : { x: c.x, y: c.y }));
}
function inertia(P, C, lab) { let s = 0; for (let i = 0; i < P.length; i++) if (lab[i] >= 0) s += d2(P[i], C[lab[i]]); return s; }
export function fit(P, k, method, seed) {
  let C = initCentroids(P, k, method, seed); const lab = new Int32Array(P.length).fill(-1); let it = 0;
  for (; it < 200; it++) { if (assign(P, C, lab) === 0) break; C = update(P, C, lab); }
  return { C, lab, it, inertia: inertia(P, C, lab) };
}
export function silhouette(P, lab, k) {
  if (k < 2) return NaN;
  const n = P.length, sums = new Float64Array(k), cnt = new Float64Array(k);
  for (let i = 0; i < n; i++) cnt[lab[i]]++;
  let tot = 0;
  for (let i = 0; i < n; i++) {
    sums.fill(0);
    for (let j = 0; j < n; j++) if (j !== i) sums[lab[j]] += Math.sqrt(d2(P[i], P[j]));
    const own = lab[i];
    if (cnt[own] <= 1) continue; // singleton: s = 0
    const a = sums[own] / (cnt[own] - 1);
    let b = Infinity; for (let c = 0; c < k; c++) if (c !== own && cnt[c] > 0) b = Math.min(b, sums[c] / cnt[c]);
    tot += (b - a) / Math.max(a, b);
  }
  return tot / n;
}
export function bestOf(P, k, nInit = 10) {
  let best = null;
  for (let s = 1; s <= nInit; s++) { const f = fit(P, k, 'kpp', 100 + s); if (!best || f.inertia < best.inertia - 1e-12) best = f; }
  return best;
}

export default async function render(main) {
  header(main, 'k-means',
    'k-means splits points into k groups by alternating two simple moves: <b>assign</b> every point to its nearest centre, then <b>move</b> each centre to the mean of its points. Repeat until nothing changes.');

  const fig = figure(main, 'Watch k-means alternate between assigning and moving',
    'Shaded regions are <b>Voronoi cells</b>: every location in a cell is closer to that centre than to any other, so the cell boundaries are exactly where assignments flip. Dashed lines trace each centre\'s path. Hover a point for its distance to its centre.');
  let kind = 'blobs', P = makeData(kind), k = 4, method = 'kpp', seed = 1, st = null, stop = null;

  select(fig.controls, { label: 'Data', value: kind, options: Object.entries(DATASETS).map(([v, d]) => [v, d.name]), onChange: (v) => { kind = v; P = makeData(v); k = DATASETS[v].k; kS.set(k); drawBase(); reset(); drawElbow(); } });
  const kS = slider(fig.controls, { label: 'k', min: 1, max: 8, value: k, onInput: (v) => { k = v; reset(); highlightK(); } });
  select(fig.controls, { label: 'Init', value: method, options: [['random', 'Random points'], ['kpp', 'k-means++']], onChange: (v) => { method = v; reset(); } });
  slider(fig.controls, { label: 'Init seed', min: 1, max: 20, value: 1, onInput: (v) => { seed = v; reset(); } });
  button(fig.controls, 'Run', () => run(), true);
  button(fig.controls, 'Step', () => { if (stop) { stop(); stop = null; } stepOnce(); paint(true); });
  button(fig.controls, 'Reset', () => reset());
  const readout = stats(fig.root);

  const row1 = el('div', { class: 'fig-row' }); fig.body.append(row1);
  const plotWrap = el('div', { style: 'flex: 1.6 1 420px' }), traceWrap = el('div');
  row1.append(plotWrap, traceWrap);
  let F, T;
  function drawBase() {
    plotWrap.innerHTML = ''; traceWrap.innerHTML = '';
    F = frame(plotWrap, { width: 560, height: 450, x: [-1.4, 1.4], y: [-1.1, 1.1], xLabel: 'feature 1', yLabel: 'feature 2' });
    F.layer('cells'); F.layer('trails'); F.layer('pts'); F.layer('cent');
    T = frame(traceWrap, { width: 340, height: 300, x: [0, 10], y: [0, 1], xLabel: 'step (assign or move)', yLabel: 'inertia' });
    traceWrap.append(el('div', { class: 'fig-caption' }, 'Inertia after every half-step. It can only go down.'));
  }

  function reset() {
    if (stop) { stop(); stop = null; }
    const C = initCentroids(P, k, method, seed);
    st = { C, lab: new Int32Array(P.length).fill(-1), phase: 'assign', it: 0, changed: null, trails: C.map((c) => [c]), done: false, hist: [] };
    paint(false);
  }
  function stepOnce() {
    if (st.done) return false;
    if (st.phase === 'assign') {
      st.changed = assign(P, st.C, st.lab);
      if (st.changed === 0) st.done = true;
      st.phase = 'update';
    } else {
      st.C = update(P, st.C, st.lab);
      st.trails.forEach((t, j) => t.push(st.C[j]));
      st.it++; st.phase = 'assign';
    }
    if (!st.done) st.hist.push({ t: st.hist.length + 1, v: inertia(P, st.C, st.lab), kind: st.phase === 'update' ? 'assign' : 'move' });
    return !st.done;
  }
  function run() {
    if (stop) stop();
    let acc = 600;
    stop = animate((dt) => { acc += dt; if (acc < 550) return true; acc = 0; const go = stepOnce(); paint(true); if (!go) stop = null; return go; });
  }

  function paint(anim) {
    const { x, y } = F;
    const sx = (d) => x(d.x), sy = (d) => y(d.y);
    // Voronoi cells of the current centres
    let cells;
    if (st.C.length === 1) cells = [`M0,0H${F.w}V${F.h}H0Z`];
    else { const vor = d3.Delaunay.from(st.C, sx, sy).voronoi([0, 0, F.w, F.h]); cells = st.C.map((_, j) => vor.renderCell(j)); }
    F.layer('cells').selectAll('path').data(cells).join('path').attr('d', (d) => d)
      .attr('fill', (d, j) => cat(j)).attr('fill-opacity', 0.13).attr('stroke', muted()).attr('stroke-width', 1);
    F.layer('trails').selectAll('path').data(st.trails).join('path').attr('fill', 'none').attr('stroke', ink()).attr('stroke-opacity', 0.55)
      .attr('stroke-width', 1.5).attr('stroke-dasharray', '4 3').attr('d', d3.line().x(sx).y(sy));
    F.layer('pts').selectAll('circle').data(P).join('circle').attr('cx', sx).attr('cy', sy).attr('r', 4)
      .attr('fill', (d, i) => (st.lab[i] < 0 ? muted() : cat(st.lab[i]))).attr('stroke', css('--surface')).attr('stroke-width', 1)
      .on('mousemove', (e, d) => { const i = P.indexOf(d), j = st.lab[i]; tip(e, j < 0 ? 'not yet assigned' : `cluster ${j + 1}<br>distance to its centre: ${Math.sqrt(d2(d, st.C[j])).toFixed(3)}`); })
      .on('mouseleave', untip);
    const cs = F.layer('cent').selectAll('rect').data(st.C).join('rect').attr('width', 15).attr('height', 15)
      .attr('fill', (d, j) => cat(j)).attr('stroke', ink()).attr('stroke-width', 2);
    (anim ? cs.transition().duration(350) : cs).attr('x', (d) => sx(d) - 7.5).attr('y', (d) => sy(d) - 7.5)
      .attr('transform', (d) => `rotate(45 ${sx(d)} ${sy(d)})`);
    // inertia trace
    const H = st.hist;
    T.x.domain([0, Math.max(10, H.length + 1)]); T.y.domain([0, H.length ? d3.max(H, (d) => d.v) * 1.08 : 1]);
    T.g.select('.axis.x').call(d3.axisBottom(T.x).ticks(5)); T.g.select('.axis.y').call(d3.axisLeft(T.y).ticks(5));
    T.g.select('.gridline').call(d3.axisLeft(T.y).ticks(5).tickSize(-T.w).tickFormat(''));
    T.layer('l').selectAll('path').data([H]).join('path').attr('fill', 'none').attr('stroke', cat(0)).attr('stroke-width', 2)
      .attr('d', d3.line().x((d) => T.x(d.t)).y((d) => T.y(d.v)));
    T.layer('p').selectAll('circle').data(H).join('circle').attr('cx', (d) => T.x(d.t)).attr('cy', (d) => T.y(d.v)).attr('r', 4)
      .attr('fill', cat(0)).attr('stroke', css('--surface')).attr('stroke-width', 1.2)
      .on('mousemove', (e, d) => tip(e, `step ${d.t} (${d.kind})<br>inertia ${d.v.toFixed(2)}`)).on('mouseleave', untip);
    const assigned = st.lab[0] >= 0;
    const sil = assigned && k > 1 ? silhouette(P, st.lab, k) : NaN;
    readout.set({
      'update rounds': st.it,
      'next step': st.done ? 'converged' : st.phase === 'assign' ? 'assign points' : 'move centres',
      'points reassigned last': st.changed == null ? '–' : st.changed,
      'inertia (within-cluster SS)': assigned ? inertia(P, st.C, st.lab).toFixed(2) : '–',
      'mean silhouette': isFinite(sil) ? sil.toFixed(3) : '–',
    });
  }

  // ---------- elbow + silhouette ----------
  const fig2 = figure(main, 'Choosing k: the elbow and the silhouette',
    'For each k, the best of 10 k-means++ runs on the current data set. <b>Inertia</b> (left) always falls as k grows, so look for the "elbow" where extra clusters stop paying off. <b>Silhouette</b> (right) compares each point\'s distance to its own cluster with the nearest other cluster: 1 = cleanly separated, 0 = on a boundary. The ring marks the k chosen above.');
  const row = el('div', { class: 'fig-row' }); fig2.body.append(row);
  const eL = el('div'), eR = el('div'); row.append(eL, eR);
  let E, S, curve = [];
  function drawElbow() {
    eL.innerHTML = ''; eR.innerHTML = '';
    curve = range(8).map((i) => { const kk = i + 1, f = bestOf(P, kk); return { k: kk, inertia: f.inertia, sil: silhouette(P, f.lab, kk) }; });
    E = frame(eL, { width: 460, height: 300, x: [0.5, 8.5], y: [0, d3.max(curve, (d) => d.inertia) * 1.08], xLabel: 'k (number of clusters)', yLabel: 'inertia', xTicks: 8 });
    S = frame(eR, { width: 460, height: 300, x: [0.5, 8.5], y: [0, 1], xLabel: 'k (number of clusters)', yLabel: 'mean silhouette', xTicks: 8 });
    const draw = (Fr, key, col) => {
      const pts = curve.filter((d) => isFinite(d[key]));
      Fr.layer('l').append('path').datum(pts).attr('fill', 'none').attr('stroke', col).attr('stroke-width', 2).attr('d', d3.line().x((d) => Fr.x(d.k)).y((d) => Fr.y(d[key])));
      Fr.layer('hl');
      Fr.layer('p').selectAll('circle').data(pts).join('circle').attr('cx', (d) => Fr.x(d.k)).attr('cy', (d) => Fr.y(d[key])).attr('r', 4.5)
        .attr('fill', col).attr('stroke', css('--surface')).attr('stroke-width', 1.5)
        .on('mousemove', (e, d) => tip(e, `k = ${d.k}<br>inertia ${d.inertia.toFixed(2)}<br>silhouette ${isFinite(d.sil) ? d.sil.toFixed(3) : '–'}`)).on('mouseleave', untip);
    };
    draw(E, 'inertia', cat(0)); draw(S, 'sil', cat(2));
    highlightK();
  }
  function highlightK() {
    if (!E) return;
    [[E, 'inertia'], [S, 'sil']].forEach(([Fr, key]) => {
      const d = curve[k - 1], ok = d && isFinite(d[key]);
      Fr.layer('hl').selectAll('circle').data(ok ? [d] : []).join('circle').attr('cx', (q) => Fr.x(q.k)).attr('cy', (q) => Fr.y(q[key])).attr('r', 10)
        .attr('fill', 'none').attr('stroke', ink()).attr('stroke-width', 2);
    });
  }

  drawBase(); reset(); drawElbow();

  section(main, 'What to notice', `<ul>
    <li>Press <b>Step</b> repeatedly. The two moves alternate: colours change (assign), then the diamonds jump to the middle of their colour (move). Inertia never goes up at either move, which is why the algorithm always stops.</li>
    <li>Switch <b>Init</b> to <i>Random points</i> and set the seed to <b>5</b>, then Run. Two centres end up sharing one blob while one centre straddles two others: the run converges with inertia 43.38, against 14.71 for the right answer. k-means only finds a <b>local</b> minimum. k-means++ spreads the starting centres out and gets 14.71 on 19 of the 20 seeds, but not all (try seed 12), which is why real software (scikit-learn's <code>n_init</code>, R's <code>nstart</code>) runs several starts and keeps the lowest inertia.</li>
    <li>On the four blobs, the elbow sits at <b>k = 4</b> and the silhouette peaks there too (0.673).</li>
    <li><b>Unequal spread</b>, k = 3: the boundary between two centres is always the perpendicular bisector, halfway between them, whatever the clusters' sizes or spreads. So the two tight clusters steal the right-hand edge of the wide one: 12 of its 160 points are given to the small clusters.</li>
    <li><b>Two moons</b>, k = 2: k-means draws one straight line and cuts both moons in half (only 72% of points land with their own moon). The silhouette is still a respectable 0.489, and on the elbow chart it keeps <i>rising</i> up to k = 8, because the silhouette also rewards round, compact groups. A good score does not mean the clusters are real.</li>
  </ul>`);

  steps(main, 'How it works', [
    'Pick k starting centres. <b>Random</b>: k random data points. <b>k-means++</b>: the first at random, then each next one chosen with probability proportional to its squared distance from the nearest centre so far, so the starts spread out.',
    '<b>Assign</b>: give each point the label of its nearest centre (Euclidean distance).',
    '<b>Update</b>: move each centre to the mean of the points assigned to it.',
    'Repeat until no point changes label. Each move lowers (or keeps) the inertia, and there are finitely many labellings, so it must stop. It stops at a local minimum, not necessarily the best one.',
    'Run it several times from different starts and keep the lowest inertia.',
  ]);
  math(main, 'inertia = Σ_i ‖x_i − μ_{c(i)}‖²          (what k-means minimises)\nassign:  c(i) = argmin_j ‖x_i − μ_j‖²\nupdate:  μ_j  = mean of { x_i : c(i) = j }\nsilhouette s(i) = (b − a) / max(a, b),  a = mean dist to own cluster, b = mean dist to nearest other cluster');

  callout(main, '', 'In biology', 'k-means is the default for <b>bulk RNA-seq subtyping</b> (e.g. consensus clustering of tumours with <code>ConsensusClusterPlus</code>, which repeatedly runs k-means on resampled patients) and for grouping <b>spatial spots</b> by expression. In single-cell work it is mostly replaced by graph clustering (Leiden in scanpy, Louvain in Seurat), because cell types differ wildly in size and spread, which is exactly the "unequal spread" failure above. k-means still appears inside other tools: SC3 uses it on many PCA views and takes a consensus, and spatial methods such as BayesSpace start from k-means labels. Always run it on PCA scores of log-normalised data, never on raw counts.');
  callout(main, 'warn', 'Pitfalls', '<ul><li>k-means assumes round clusters of similar size. Elongated, nested or uneven groups get cut by straight lines.</li><li>You must choose k. The elbow is often vague; check silhouette and, more importantly, whether the clusters make biological sense (marker genes).</li><li>Features must be on comparable scales, or the largest-variance feature dominates distances. <a href="#/concepts/scaling">Scale first.</a></li><li>Results depend on the starting centres. Use k-means++ and several restarts, and set a seed.</li><li>Every point is forced into a cluster, outliers included. <a href="#/concepts/dbscan">DBSCAN</a> can label noise instead.</li></ul>');
  seeAlso(main, [['concepts', 'gmm', 'Gaussian mixtures (soft k-means)'], ['concepts', 'dbscan', 'DBSCAN'], ['concepts', 'hierarchical', 'Hierarchical clustering'], ['concepts', 'graph-clustering', 'kNN graphs & Leiden'], ['omics', 'bulk-subtypes', 'Bulk subtypes']]);
}
