// DBSCAN: eps / minPts sliders, core / border / noise points, eps discs on hover, an animated expansion,
// a k-distance plot for choosing eps, and a head-to-head with k-means on two moons.
import { header, figure, slider, select, button, checkbox, stats, section, steps, callout, math, seeAlso, animate, legend, tip, untip, el } from '../lib/ui.js';
import { frame, cat, css, ink, ink2, muted } from '../lib/plot.js';
import { rng, moons, range } from '../lib/data.js';
const d3 = window.d3;

const DATASETS = { moons: 'Two moons + scattered noise', density: 'Dense + sparse blob (hard case)' };

export function makeData(kind) {
  const r = rng(41), out = [];
  if (kind === 'moons') {
    moons(240, 0.06, 2).forEach((p) => out.push({ x: p.x, y: p.y, label: p.label }));
  } else {
    for (let i = 0; i < 120; i++) out.push({ x: r.normal(-0.6, 0.1), y: r.normal(0.3, 0.1), label: 0 });
    for (let i = 0; i < 100; i++) out.push({ x: r.normal(0.5, 0.28), y: r.normal(-0.15, 0.28), label: 1 });
  }
  for (let i = 0; i < 20; i++) out.push({ x: -1.25 + 2.5 * r(), y: -0.95 + 1.9 * r(), label: -1 }); // background noise
  return out;
}

/** DBSCAN. minPts counts the point itself (scikit-learn convention). Returns labels (-1 = noise), types, BFS order. */
export function dbscan(P, eps, minPts) {
  const n = P.length, e2 = eps * eps, nb = [];
  for (let i = 0; i < n; i++) { const a = []; for (let j = 0; j < n; j++) if ((P[i].x - P[j].x) ** 2 + (P[i].y - P[j].y) ** 2 <= e2) a.push(j); nb.push(a); }
  const core = nb.map((a) => a.length >= minPts);
  const lab = new Int32Array(n).fill(-1), order = [];
  let c = 0;
  for (let i = 0; i < n; i++) {
    if (!core[i] || lab[i] >= 0) continue;
    lab[i] = c; order.push(i);
    const queue = [i];
    while (queue.length) {
      const p = queue.shift();
      if (!core[p]) continue; // border points join but do not expand
      for (const q of nb[p]) if (lab[q] < 0) { lab[q] = c; order.push(q); queue.push(q); }
    }
    c++;
  }
  const type = range(n).map((i) => (core[i] ? 'core' : lab[i] >= 0 ? 'border' : 'noise'));
  return { lab, type, order, k: c, nb };
}
/** Distance from each point to its (minPts−1)-th nearest other point: the radius it needs to be core. */
export function kDist(P, minPts) {
  return P.map((p) => { const d = P.map((q) => Math.hypot(p.x - q.x, p.y - q.y)).sort((a, b) => a - b); return d[Math.min(minPts - 1, d.length - 1)]; });
}
export function kmeans2(P, K = 2) {
  let best = null; const r = rng(7);
  for (let s = 0; s < 10; s++) {
    const C = r.shuffle(range(P.length)).slice(0, K).map((i) => [P[i].x, P[i].y]); const lab = new Int32Array(P.length).fill(-1);
    for (let it = 0; it < 100; it++) {
      let ch = 0;
      P.forEach((p, i) => { let b = 0, bd = Infinity; C.forEach((c, j) => { const d = (p.x - c[0]) ** 2 + (p.y - c[1]) ** 2; if (d < bd) { bd = d; b = j; } }); if (lab[i] !== b) { lab[i] = b; ch++; } });
      if (!ch) break;
      C.forEach((c, j) => { let sx = 0, sy = 0, m = 0; P.forEach((p, i) => { if (lab[i] === j) { sx += p.x; sy += p.y; m++; } }); if (m) { c[0] = sx / m; c[1] = sy / m; } });
    }
    let inert = 0; P.forEach((p, i) => { inert += (p.x - C[lab[i]][0]) ** 2 + (p.y - C[lab[i]][1]) ** 2; });
    if (!best || inert < best.inert) best = { lab, C, inert };
  }
  return best;
}
/** Of the points that truly belong to a group (label ≥ 0), the fraction whose cluster's majority group is their own. */
export function purity(P, lab) {
  const by = d3.group(range(P.length).filter((i) => P[i].label >= 0 && lab[i] >= 0), (i) => lab[i]);
  let ok = 0;
  for (const [, idx] of by) { const cnt = d3.rollup(idx, (v) => v.length, (i) => P[i].label); ok += d3.max(cnt.values()); }
  return ok / P.filter((p) => p.label >= 0).length;
}

export default async function render(main) {
  header(main, 'DBSCAN',
    'DBSCAN defines a cluster as a region where points are <b>packed densely</b>: any point with at least minPts neighbours within radius ε is a "core" point, and clusters grow by chaining core points together. Points in sparse regions are labelled <b>noise</b> instead of being forced into a cluster.');

  const fig = figure(main, 'Density-based clustering: core, border and noise',
    'Large filled dots are <b>core</b> points (≥ minPts points, counting themselves, within ε). Ringed dots are <b>border</b> points: not dense themselves, but within ε of a core point. Small grey dots are <b>noise</b>. <b>Hover any point</b> to see its ε-disc and the neighbours inside it.');
  let kind = 'moons', P = makeData(kind), eps = 0.15, minPts = 5, res = null, reveal = Infinity, stop = null, showDiscs = false;

  select(fig.controls, { label: 'Data', value: kind, options: Object.entries(DATASETS), onChange: (v) => { kind = v; P = makeData(v); update(); } });
  slider(fig.controls, { label: 'ε (radius)', min: 0.04, max: 0.5, step: 0.01, value: eps, fmt: (v) => v.toFixed(2), onInput: (v) => { eps = v; update(); } });
  slider(fig.controls, { label: 'minPts', min: 2, max: 20, value: minPts, onInput: (v) => { minPts = v; update(); } });
  checkbox(fig.controls, { label: 'Show ε-discs of core points', checked: false, onChange: (c) => { showDiscs = c; paint(); } });
  button(fig.controls, 'Animate expansion', () => expand(), true);
  const readout = stats(fig.root);

  const row = el('div', { class: 'fig-row' }); fig.body.append(row);
  const L = el('div', { style: 'flex: 1.6 1 420px' }), R = el('div'); row.append(L, R);
  const F = frame(L, { width: 560, height: 450, x: [-1.4, 1.4], y: [-1.1, 1.1], xLabel: 'feature 1', yLabel: 'feature 2' });
  F.layer('discs'); F.layer('hover'); F.layer('pts');
  legend(L, [['core', cat(0)], ['border (ringed)', css('--surface-2')], ['noise', muted()]]);
  const K = frame(R, { width: 340, height: 320, x: [0, 1], y: [0, 0.5], xLabel: 'points, sorted', yLabel: 'distance to (minPts−1)-th neighbour' });
  const kCap = el('div', { class: 'fig-caption' }); R.append(kCap);
  const rx = () => F.x(eps) - F.x(0), ry = () => F.y(0) - F.y(eps);

  const colour = (i) => { const c = res.lab[i]; return c < 0 ? muted() : c < 8 ? cat(c) : ink2(); };
  function update() { if (stop) { stop(); stop = null; } reveal = Infinity; res = dbscan(P, eps, minPts); paint(); drawK(); }

  function paint() {
    const shown = new Set(res.order.slice(0, reveal === Infinity ? res.order.length : reveal));
    const vis = (i) => reveal === Infinity || shown.has(i);
    const coreIdx = range(P.length).filter((i) => res.type[i] === 'core' && vis(i));
    F.layer('discs').selectAll('ellipse').data(showDiscs ? coreIdx : []).join('ellipse')
      .attr('cx', (i) => F.x(P[i].x)).attr('cy', (i) => F.y(P[i].y)).attr('rx', rx()).attr('ry', ry())
      .attr('fill', (i) => colour(i)).attr('fill-opacity', 0.08).attr('stroke', 'none');
    F.layer('pts').selectAll('circle').data(P).join('circle').attr('cx', (d) => F.x(d.x)).attr('cy', (d) => F.y(d.y))
      .attr('r', (d, i) => (res.type[i] === 'core' ? 5 : res.type[i] === 'border' ? 4.2 : 3))
      .attr('fill', (d, i) => (!vis(i) ? css('--surface-2') : res.type[i] === 'border' ? css('--surface') : colour(i)))
      .attr('stroke', (d, i) => (!vis(i) ? muted() : res.type[i] === 'border' ? colour(i) : css('--surface')))
      .attr('stroke-width', (d, i) => (res.type[i] === 'border' && vis(i) ? 2.2 : 1))
      .attr('opacity', (d, i) => (res.type[i] === 'noise' ? 0.75 : 1))
      .on('mousemove', (e, d) => hover(e, P.indexOf(d))).on('mouseleave', () => { untip(); F.layer('hover').selectAll('*').remove(); });
    const cnt = d3.rollup(res.type, (v) => v.length, (t) => t);
    readout.set({ clusters: res.k, 'core points': cnt.get('core') || 0, 'border points': cnt.get('border') || 0, 'noise points': cnt.get('noise') || 0 });
  }
  function hover(e, i) {
    const h = F.layer('hover'); h.selectAll('*').remove();
    const p = P[i], nb = res.nb[i];
    h.selectAll('line').data(nb.filter((j) => j !== i)).join('line').attr('x1', F.x(p.x)).attr('y1', F.y(p.y))
      .attr('x2', (j) => F.x(P[j].x)).attr('y2', (j) => F.y(P[j].y)).attr('stroke', ink()).attr('stroke-opacity', 0.5).attr('stroke-width', 1.2);
    h.append('ellipse').attr('cx', F.x(p.x)).attr('cy', F.y(p.y)).attr('rx', rx()).attr('ry', ry())
      .attr('fill', ink()).attr('fill-opacity', 0.05).attr('stroke', ink()).attr('stroke-width', 1.5).attr('stroke-dasharray', '5 4');
    const t = res.type[i];
    tip(e, `<b>${t}</b> point<br>${nb.length} point${nb.length === 1 ? '' : 's'} within ε (incl. itself); core needs ≥ ${minPts}` +
      (t === 'border' ? '<br>not dense itself, but a core point is within ε' : t === 'noise' ? '<br>not dense, and no core point within ε' : `<br>cluster ${res.lab[i] + 1}`));
  }
  function expand() {
    if (stop) stop();
    reveal = 0; let acc = 0;
    const per = Math.max(1, Math.ceil(res.order.length / 120));
    stop = animate((dt) => { acc += dt; if (acc < 30) return true; acc = 0; reveal += per; if (reveal >= res.order.length) { reveal = Infinity; paint(); stop = null; return false; } paint(); return true; });
    paint();
  }
  function drawK() {
    const kd = kDist(P, minPts).map((d, i) => ({ d, i })).sort((a, b) => a.d - b.d);
    const ymax = Math.max(0.5, eps * 1.15, d3.max(kd, (q) => q.d) * 1.02);
    K.x.domain([0, kd.length]); K.y.domain([0, Math.min(ymax, 1.2)]);
    K.g.select('.axis.x').call(d3.axisBottom(K.x).ticks(4)); K.g.select('.axis.y').call(d3.axisLeft(K.y).ticks(5));
    K.g.select('.gridline').call(d3.axisLeft(K.y).ticks(5).tickSize(-K.w).tickFormat(''));
    K.layer('c').selectAll('path').data([kd]).join('path').attr('fill', 'none').attr('stroke', cat(0)).attr('stroke-width', 2)
      .attr('d', d3.line().x((q, j) => K.x(j)).y((q) => K.y(Math.min(q.d, K.y.domain()[1]))));
    const ey = K.y(eps);
    K.layer('eps').selectAll('line').data([0]).join('line').attr('x1', 0).attr('x2', K.w).attr('y1', ey).attr('y2', ey).attr('stroke', ink()).attr('stroke-width', 2).attr('stroke-dasharray', '6 4');
    K.layer('eps').selectAll('text').data([0]).join('text').attr('x', 6).attr('y', ey - 6).attr('font-size', 12).attr('fill', ink())
      .attr('paint-order', 'stroke').attr('stroke', css('--surface')).attr('stroke-width', 4).text(`ε = ${eps.toFixed(2)}`);
    const above = kd.filter((q) => q.d > eps).length;
    kCap.innerHTML = `<b>k-distance plot.</b> Each point needs a radius this big to count as core. Points above the dashed ε line (${above} now) are not core. A good ε usually sits at the "knee", where the curve turns sharply upward.`;
  }
  update();

  // ---------- head to head on moons ----------
  const fig2 = figure(main, 'Two moons: k-means (k = 2) vs DBSCAN (ε = 0.15, minPts = 5)',
    'Same 260 points. k-means must put every point in one of two round-ish groups separated by a straight line. DBSCAN follows the density, so each crescent becomes one cluster, and the scattered background points are set aside as noise (grey).');
  const M = makeData('moons'), km = kmeans2(M), db = dbscan(M, 0.15, 5);
  const row2 = el('div', { class: 'fig-row' }); fig2.body.append(row2);
  [['k-means', km.lab], ['DBSCAN', db.lab]].forEach(([name, lab]) => {
    const box = el('div'); row2.append(box);
    const Fr = frame(box, { width: 460, height: 380, x: [-1.4, 1.4], y: [-1.1, 1.1], margin: { top: 30 } });
    const noise = lab.filter((l) => l < 0).length;
    Fr.svg.append('text').attr('x', 230).attr('y', 18).attr('text-anchor', 'middle').attr('font-weight', 600).attr('font-size', 14).attr('fill', ink())
      .text(`${name}: ${(purity(M, lab) * 100).toFixed(0)}% of moon points with their own moon${noise ? `, ${noise} noise` : ''}`);
    Fr.layer('p').selectAll('circle').data(M).join('circle').attr('cx', (d) => Fr.x(d.x)).attr('cy', (d) => Fr.y(d.y)).attr('r', (d, i) => (lab[i] < 0 ? 3 : 4))
      .attr('fill', (d, i) => (lab[i] < 0 ? muted() : cat(lab[i]))).attr('stroke', css('--surface')).attr('stroke-width', 1)
      .on('mousemove', (e, d) => tip(e, `${d.label < 0 ? 'background noise point' : 'moon ' + (d.label + 1)}<br>${name}: ${lab[M.indexOf(d)] < 0 ? 'noise' : 'cluster ' + (lab[M.indexOf(d)] + 1)}`)).on('mouseleave', untip);
  });

  section(main, 'What to notice', `<ul>
    <li>With the defaults (ε = 0.15, minPts = 5) DBSCAN finds exactly the two moons and flags 21 points as noise, 18 of them from the 20 scattered background points. You never told it how many clusters to find.</li>
    <li>Hover a point at the tip of a moon: it may be a <b>border</b> point, with fewer than 5 points in its disc but a core neighbour inside it.</li>
    <li>Shrink <b>ε</b> to 0.08: the moons shatter into many small clusters and most points become noise. Grow it to 0.35: the discs bridge the gap and both moons merge into one cluster.</li>
    <li>Raise <b>minPts</b> and the k-distance curve rises, so more points fall above the ε line and become non-core. minPts is how strict you are about "dense".</li>
    <li>Switch to <b>Dense + sparse blob</b>. No single ε works: small enough to keep the dense blob tidy, and the sparse blob dissolves into noise; large enough for the sparse blob, and noise points get swept in. One global density threshold is DBSCAN's main limitation (HDBSCAN fixes this).</li>
    <li>Press <b>Animate expansion</b> to watch clusters grow outward from a seed core point, neighbour by neighbour.</li>
  </ul>`);

  steps(main, 'How it works', [
    'For each point, count the points within distance ε (including itself). If the count is at least minPts, it is a <b>core</b> point.',
    'Pick an unvisited core point and start a new cluster.',
    'Add every point within ε of it. For each newly added <b>core</b> point, add its ε-neighbours too, and so on (a breadth-first search). Non-core points that get added are <b>border</b> points; they join but do not spread the cluster.',
    'When the search runs out, move to the next unvisited core point and start another cluster.',
    'Whatever is never reached is <b>noise</b>.',
  ]);
  math(main, 'N_ε(p) = { q : ‖p − q‖ ≤ ε }\ncore(p)  ⇔  |N_ε(p)| ≥ minPts\nclusters = connected components of core points (edges between cores within ε), plus their border points');

  callout(main, '', 'In biology', 'DBSCAN and its successor <b>HDBSCAN</b> (which effectively tries every ε and keeps the most stable clusters) are used to cluster <b>UMAP embeddings</b> of single-cell data for a quick look, to find <b>rare populations</b> in cytometry, and to segment <b>spatial transcripts</b>: grouping individual mRNA molecules from MERFISH or Xenium into cells or subcellular foci, and finding dense clusters of a cell type in tissue coordinates (e.g. tertiary lymphoid structures). Treat clusters of a UMAP with care: UMAP distorts density, so dense-looking regions in 2D are partly an artefact.');
  callout(main, 'warn', 'Pitfalls', '<ul><li>One global ε cannot handle clusters of very different densities (see the hard case).</li><li>ε is in the units of your features, so scale them first. In high dimensions all distances become similar and density loses meaning; run it on a PCA or similar reduction.</li><li>Border points touching two clusters go to whichever cluster reaches them first, so results depend slightly on point order.</li><li>"Noise" is relative to your ε and minPts, not a property of the cell. Do not discard rare cells just because DBSCAN called them noise.</li></ul>');
  seeAlso(main, [['concepts', 'kmeans', 'k-means'], ['concepts', 'hierarchical', 'Hierarchical clustering'], ['concepts', 'graph-clustering', 'kNN graphs & Leiden'], ['concepts', 'curse-dim', 'Curse of dimensionality']]);
}
