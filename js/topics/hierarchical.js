// Hierarchical (agglomerative) clustering: animated merges on the scatter, a dendrogram growing alongside,
// linkage choice, and a draggable cut height that colours the clusters.
import { header, figure, slider, select, button, stats, section, steps, callout, math, seeAlso, animate, legend, tip, untip, el } from '../lib/ui.js';
import { frame, cat, css, ink, ink2, muted } from '../lib/plot.js';
import { rng, range } from '../lib/data.js';
const d3 = window.d3;

const LINKAGES = { single: 'Single (nearest pair)', complete: 'Complete (farthest pair)', average: 'Average (UPGMA)', ward: 'Ward (least variance increase)' };

/** Three groups, a thin bridge of points between two of them, and one outlier. */
export function makeData() {
  const r = rng(21), out = [];
  const G = [[-0.62, 0.45], [0.62, 0.45], [0, -0.5]];
  G.forEach((c, g) => { for (let i = 0; i < 12; i++) out.push({ x: r.normal(c[0], 0.13), y: r.normal(c[1], 0.13), group: g }); });
  [-0.3, -0.1, 0.1, 0.3].forEach((x) => out.push({ x: x + r.normal(0, 0.02), y: 0.45 + r.normal(0, 0.04), group: 3 }));
  out.push({ x: 1.05, y: -0.85, group: 4 });
  return out.map((p, i) => ({ ...p, id: i }));
}

/** Naive O(n³) agglomeration with Lance–Williams updates. Returns merges [{a, b, h, members, seg}]. */
export function agglomerate(P, method) {
  const n = P.length, D = new Float64Array(n * n);
  const e2 = (i, j) => (P[i].x - P[j].x) ** 2 + (P[i].y - P[j].y) ** 2;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) D[i * n + j] = method === 'ward' ? e2(i, j) : Math.sqrt(e2(i, j));
  const active = range(n), size = new Array(n).fill(1), id = range(n), members = range(n).map((i) => [i]);
  const merges = [];
  while (active.length > 1) {
    let bi = -1, bj = -1, bd = Infinity;
    for (let u = 0; u < active.length; u++) for (let v = u + 1; v < active.length; v++) {
      const d = D[active[u] * n + active[v]]; if (d < bd) { bd = d; bi = u; bj = v; }
    }
    const I = active[bi], J = active[bj], ni = size[I], nj = size[J];
    // representative segment drawn on the scatter: what the linkage actually measured
    const mI = members[I], mJ = members[J];
    let seg;
    if (method === 'single' || method === 'complete') {
      let best = method === 'single' ? Infinity : -1, pr = null;
      for (const a of mI) for (const b of mJ) { const d = e2(a, b); if (method === 'single' ? d < best : d > best) { best = d; pr = [a, b]; } }
      seg = [P[pr[0]], P[pr[1]]].map((p) => ({ x: p.x, y: p.y }));
    } else {
      const cen = (m) => ({ x: d3.mean(m, (i) => P[i].x), y: d3.mean(m, (i) => P[i].y) });
      seg = [cen(mI), cen(mJ)];
    }
    merges.push({ a: id[I], b: id[J], h: method === 'ward' ? Math.sqrt(bd) : bd, members: mI.concat(mJ), seg, sizes: [ni, nj] });
    // Lance–Williams: update row I to be the merged cluster, drop J
    for (const K of active) {
      if (K === I || K === J) continue;
      const dik = D[I * n + K], djk = D[J * n + K], nk = size[K];
      let d;
      if (method === 'single') d = Math.min(dik, djk);
      else if (method === 'complete') d = Math.max(dik, djk);
      else if (method === 'average') d = (ni * dik + nj * djk) / (ni + nj);
      else d = ((ni + nk) * dik + (nj + nk) * djk - nk * D[I * n + J]) / (ni + nj + nk); // Ward on squared distances
      D[I * n + K] = D[K * n + I] = d;
    }
    size[I] = ni + nj; members[I] = mI.concat(mJ); id[I] = n + merges.length - 1;
    active.splice(bj, 1);
  }
  return merges;
}

/** Cluster labels from the first `done` merges whose height is ≤ cut. Labels ranked by size (0 = largest). */
export function cutLabels(n, merges, done, cut) {
  const par = range(n);
  const find = (i) => (par[i] === i ? i : (par[i] = find(par[i])));
  for (let t = 0; t < done; t++) { const m = merges[t]; if (m.h > cut) continue; const r0 = find(m.members[0]); for (const i of m.members) par[find(i)] = r0; }
  const groups = d3.groups(range(n), find).sort((a, b) => b[1].length - a[1].length || a[1][0] - b[1][0]);
  const lab = new Int32Array(n), sizes = [];
  groups.forEach(([, mem], g) => { mem.forEach((i) => { lab[i] = g; }); sizes.push(mem.length); });
  return { lab, sizes, k: groups.length };
}
/** A cut height that yields exactly k clusters: halfway between the merge that would leave k−1 and the one before it. */
export const kCut = (merges, k) => { const m = merges.length; return k <= 1 ? merges[m - 1].h * 1.02 : (merges[m - k].h + merges[m - k + 1].h) / 2; };

export default async function render(main) {
  header(main, 'Hierarchical clustering',
    'Start with every point as its own cluster, then <b>repeatedly merge the two closest clusters</b> until only one is left. The record of merges is a tree (a <b>dendrogram</b>); cutting it at any height gives a clustering.');

  const P = makeData(), n = P.length;
  const fig = figure(main, 'Merge by merge: the scatter and the dendrogram grow together',
    'Left: each grey segment is one merge, drawn between the two points (single / complete) or the two cluster centres (average / Ward) that the linkage measured; the newest merge is thick. Right: the dendrogram. A merge\'s height is the linkage distance at which it happened. <b>Drag the dashed cut line</b> (or click the dendrogram): every branch below it is a cluster. Grey points are singletons.');
  let method = 'average', merges = agglomerate(P, method), done = n - 1, cut = 0, stop = null;

  select(fig.controls, { label: 'Linkage', value: method, options: Object.entries(LINKAGES), onChange: (v) => { method = v; merges = agglomerate(P, v); cut = threeCut(); drawDendroFrame(); paint(); } });
  const doneS = slider(fig.controls, { label: 'Merges done', min: 0, max: n - 1, value: n - 1, onInput: (v) => { halt(); done = v; paint(); } });
  button(fig.controls, 'Play', () => play(), true);
  button(fig.controls, 'Step', () => { halt(); if (done < n - 1) done++; doneS.set(done); paint(); });
  button(fig.controls, 'Reset', () => { halt(); done = 0; doneS.set(0); paint(); });
  const readout = stats(fig.root);

  const row = el('div', { class: 'fig-row' }); fig.body.append(row);
  const L = el('div'), R = el('div'); row.append(L, R);
  const S = frame(L, { width: 470, height: 420, x: [-1.15, 1.25], y: [-1.05, 0.95], xLabel: 'feature 1', yLabel: 'feature 2' });
  S.layer('hull'); S.layer('seg'); S.layer('pts');
  let Dd;
  const threeCut = () => kCut(merges, 3);
  cut = threeCut();

  // leaf order from the final tree (depth-first), so branches never cross
  function leafOrder() {
    const kids = (node) => (node < n ? null : [merges[node - n].a, merges[node - n].b]);
    const order = [], stack = [2 * n - 2];
    while (stack.length) { const v = stack.pop(); const c = kids(v); if (!c) order.push(v); else stack.push(c[1], c[0]); }
    const pos = new Array(n); order.forEach((leaf, i) => { pos[leaf] = i; }); return pos;
  }
  let pos, nodeX;
  function drawDendroFrame() {
    R.innerHTML = '';
    const hmax = merges[n - 2].h * 1.12;
    Dd = frame(R, { width: 470, height: 420, x: [-0.8, n - 0.2], y: [0, hmax], yLabel: 'merge height (linkage distance)', xLabel: 'points (leaves)', margin: { left: 54 } });
    Dd.g.select('.axis.x').call(d3.axisBottom(Dd.x).tickValues([]));
    Dd.layer('links'); Dd.layer('leaves'); Dd.layer('cut');
    pos = leafOrder();
    nodeX = new Float64Array(2 * n - 1);
    for (let i = 0; i < n; i++) nodeX[i] = pos[i];
    merges.forEach((m, t) => { nodeX[n + t] = (nodeX[m.a] + nodeX[m.b]) / 2; });
    // draggable / clickable cut line
    const setCut = (e) => { const [, my] = d3.pointer(e, Dd.g.node()); cut = Math.max(0, Math.min(hmax, Dd.y.invert(my))); paint(); };
    Dd.svg.style('cursor', 'ns-resize').on('click', setCut).call(d3.drag().on('drag', setCut));
  }
  drawDendroFrame();

  function halt() { if (stop) { stop(); stop = null; } }
  function play() {
    halt(); if (done >= n - 1) done = 0;
    let acc = 0;
    stop = animate((dt) => { acc += dt; if (acc < 260) return true; acc = 0; done++; doneS.set(done); paint(); if (done >= n - 1) { stop = null; return false; } return true; });
  }

  const colourOf = (lab, sizes) => (i) => (sizes[lab[i]] < 2 ? muted() : lab[i] < 8 ? cat(lab[i]) : ink2());

  function paint() {
    const { lab, sizes, k } = cutLabels(n, merges, done, cut);
    const col = colourOf(lab, sizes);
    // hulls of the clusters at the cut
    const hulls = d3.groups(range(n), (i) => lab[i]).filter(([g, m]) => m.length >= 3 && g < 8)
      .map(([g, m]) => ({ g, hull: d3.polygonHull(m.map((i) => [S.x(P[i].x), S.y(P[i].y)])) })).filter((d) => d.hull);
    S.layer('hull').selectAll('path').data(hulls).join('path').attr('d', (d) => 'M' + d.hull.join('L') + 'Z')
      .attr('fill', (d) => cat(d.g)).attr('fill-opacity', 0.12).attr('stroke', (d) => cat(d.g)).attr('stroke-opacity', 0.35).attr('stroke-width', 6).attr('stroke-linejoin', 'round');
    const segs = merges.slice(0, done);
    S.layer('seg').selectAll('line').data(segs).join('line')
      .attr('x1', (m) => S.x(m.seg[0].x)).attr('y1', (m) => S.y(m.seg[0].y)).attr('x2', (m) => S.x(m.seg[1].x)).attr('y2', (m) => S.y(m.seg[1].y))
      .attr('stroke', (m, t) => (t === done - 1 ? ink() : muted())).attr('stroke-width', (m, t) => (t === done - 1 ? 3.5 : 1.5));
    S.layer('pts').selectAll('circle').data(P).join('circle').attr('cx', (d) => S.x(d.x)).attr('cy', (d) => S.y(d.y)).attr('r', 5)
      .attr('fill', (d) => col(d.id)).attr('stroke', css('--surface')).attr('stroke-width', 1.5)
      .on('mousemove', (e, d) => tip(e, `point ${d.id}<br>${sizes[lab[d.id]] < 2 ? 'singleton at this cut' : `cluster ${lab[d.id] + 1} (${sizes[lab[d.id]]} points)`}`)).on('mouseleave', untip);

    // dendrogram links for merges done so far
    const linkCol = (m) => { if (m.h > cut) return muted(); const g = lab[m.members[0]]; return g < 8 ? cat(g) : ink2(); };
    const nodeH = (v) => (v < n ? 0 : merges[v - n].h);
    Dd.layer('links').selectAll('path').data(segs).join('path').attr('fill', 'none')
      .attr('d', (m) => `M${Dd.x(nodeX[m.a])},${Dd.y(nodeH(m.a))}V${Dd.y(m.h)}H${Dd.x(nodeX[m.b])}V${Dd.y(nodeH(m.b))}`)
      .attr('stroke', linkCol).attr('stroke-width', (m, t) => (t === done - 1 ? 3.5 : 2))
      .on('mousemove', (e, m) => tip(e, `merge ${merges.indexOf(m) + 1}: ${m.sizes[0]} + ${m.sizes[1]} points<br>height ${m.h.toFixed(3)}`)).on('mouseleave', untip);
    Dd.layer('leaves').selectAll('circle').data(P).join('circle').attr('cx', (d) => Dd.x(pos[d.id])).attr('cy', Dd.y(0)).attr('r', 3.2).attr('fill', (d) => col(d.id));
    const cy = Dd.y(cut);
    const cl = Dd.layer('cut').selectAll('g').data([0]).join((en) => { const g = en.append('g'); g.append('line'); g.append('text'); return g; });
    cl.select('line').attr('x1', 0).attr('x2', Dd.w).attr('y1', cy).attr('y2', cy).attr('stroke', ink()).attr('stroke-width', 2).attr('stroke-dasharray', '6 4');
    cl.select('text').attr('x', Dd.w - 4).attr('y', cy - 6).attr('text-anchor', 'end').attr('font-size', 12).attr('fill', ink()).attr('paint-order', 'stroke').attr('stroke', css('--surface')).attr('stroke-width', 5).text(`cut = ${cut.toFixed(2)} → ${k} clusters`);

    const last = done ? merges[done - 1] : null;
    readout.set({
      'merges done': `${done} / ${n - 1}`,
      'last merge height': last ? last.h.toFixed(3) : '–',
      'clusters at the cut': k,
      'of which singletons': sizes.filter((s) => s === 1).length,
      'cluster sizes': sizes.filter((s) => s > 1).join(' · ') || '–',
    });
  }
  paint();

  // ---------- small multiples: four linkages, each cut into 3 clusters ----------
  const fig2 = figure(main, 'Same points, four linkages, each cut into 3 clusters',
    'The only difference between the panels is how the distance between two <i>groups</i> is defined. Grey = a singleton cluster.');
  const grid = el('div', { class: 'fig-row' }); fig2.body.append(grid);
  Object.entries(LINKAGES).forEach(([key, name]) => {
    const box = el('div', { style: 'flex: 1 1 200px' }); grid.append(box);
    const M = agglomerate(P, key);
    const { lab, sizes } = cutLabels(n, M, n - 1, kCut(M, 3));
    const Fm = frame(box, { width: 240, height: 230, x: [-1.15, 1.25], y: [-1.05, 0.95], axes: false, grid: false, margin: { top: 26, left: 6, right: 6, bottom: 6 } });
    Fm.svg.append('text').attr('x', 120).attr('y', 17).attr('text-anchor', 'middle').attr('font-size', 13).attr('font-weight', 600).attr('fill', ink()).text(name.split(' (')[0]);
    Fm.g.append('rect').attr('width', Fm.w).attr('height', Fm.h).attr('fill', 'none').attr('stroke', css('--grid'));
    const col = colourOf(lab, sizes);
    Fm.layer('p').selectAll('circle').data(P).join('circle').attr('cx', (d) => Fm.x(d.x)).attr('cy', (d) => Fm.y(d.y)).attr('r', 4.5)
      .attr('fill', (d) => col(d.id)).attr('stroke', css('--surface')).attr('stroke-width', 1.2)
      .on('mousemove', (e, d) => tip(e, `${name}<br>${sizes[lab[d.id]] < 2 ? 'singleton' : `cluster ${lab[d.id] + 1} of size ${sizes[lab[d.id]]}`}`)).on('mouseleave', untip);
  });

  section(main, 'What to notice', `<ul>
    <li>Press <b>Play</b> with average linkage to replay the merges from scratch. The first merges join near-identical pairs inside each blob (short segments, low branches). The last few merges join whole blobs, and they happen much higher up. Big vertical gaps in the dendrogram are where natural clusters live.</li>
    <li>Switch to <b>Single</b> linkage. The four "bridge" points let it hop from the left blob to the right one, so at 3 clusters it gives {both top blobs + bridge}, {bottom blob}, and the lone outlier. This is <b>chaining</b>: single linkage only needs one close pair.</li>
    <li><b>Average</b>, <b>complete</b> and <b>Ward</b> prefer compact groups; at 3 clusters they all return the three blobs, with the outlier absorbed into the bottom one. Average and Ward split the bridge 2 + 2; complete splits it 3 + 1.</li>
    <li>Drag the cut line: one tree, many clusterings. You never re-run anything; you just choose a height. Ward's dendrogram has the most dramatic jump (the last two merges sit near 4.1 and 4.4, the one before at 1.5), so the number of clusters looks obvious.</li>
  </ul>`);

  steps(main, 'How it works', [
    'Compute the distance between every pair of points. Each point is its own cluster.',
    'Find the two clusters with the smallest <b>linkage</b> distance and merge them. Record the height of that merge.',
    'Update the distances from the new cluster to every other one. The linkage defines how: <b>single</b> = nearest pair of points, <b>complete</b> = farthest pair, <b>average</b> = mean over all pairs, <b>Ward</b> = how much the within-cluster sum of squares would grow.',
    'Repeat until one cluster remains. The merge history is the dendrogram.',
    'Cut the dendrogram at a height (or ask for k clusters) to get flat labels.',
  ]);
  math(main, 'Lance–Williams update after merging I and J, for every other cluster K:\n  single:   d(IJ,K) = min(d_IK, d_JK)\n  complete: d(IJ,K) = max(d_IK, d_JK)\n  average:  d(IJ,K) = (n_I d_IK + n_J d_JK) / (n_I + n_J)\n  Ward:     d²(IJ,K) = [(n_I+n_K) d²_IK + (n_J+n_K) d²_JK − n_K d²_IJ] / (n_I+n_J+n_K)');

  callout(main, '', 'In biology', 'The dendrograms on the edges of every expression <b>heatmap</b> (<code>pheatmap</code>, <code>ComplexHeatmap</code>, seaborn <code>clustermap</code>) come from exactly this, usually with average or complete linkage on 1 − Pearson correlation between samples or genes. In single-cell work, scanpy\'s <code>sc.tl.dendrogram</code> and Seurat\'s <code>BuildClusterTree</code> build a tree over <i>cluster</i> centroids (not cells) to order dot plots and to see which clusters are near-duplicates. Ward linkage on PCA scores is common for bulk tumour subtyping. With 50,000 cells the O(n²) distance matrix is the problem, which is one reason graph clustering took over.');
  callout(main, 'warn', 'Pitfalls', '<ul><li>The linkage choice can change the answer completely (compare the panels above). Single linkage chains through noise; Ward assumes compact, similar-size groups.</li><li>Distance choice matters just as much: Euclidean on scaled data, or correlation distance for expression profiles.</li><li>Merges are greedy and final: an early bad merge is never undone.</li><li>Leaf order within a dendrogram is partly arbitrary (any branch can be flipped). Two leaves drawn next to each other are not necessarily similar; read similarity from the merge height.</li><li>Time and memory grow as n² or worse, so it is for hundreds to a few thousand items, not whole single-cell atlases.</li></ul>');
  seeAlso(main, [['concepts', 'kmeans', 'k-means'], ['concepts', 'dbscan', 'DBSCAN'], ['concepts', 'graph-clustering', 'kNN graphs & Leiden'], ['omics', 'bulk-subtypes', 'Bulk subtypes']]);
}
