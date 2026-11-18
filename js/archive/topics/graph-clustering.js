// kNN graph + Louvain community detection (modularity with a resolution parameter), the scanpy/Seurat way.
import { header, figure, slider, select, button, checkbox, stats, section, steps, callout, math, legend, seeAlso, animate, tip, untip, el } from '../../lib/ui.js';
import { frame, cat, css, ink, ink2, muted } from '../../lib/plot.js';
import { rng } from '../../lib/data.js';
const d3 = window.d3;

// ---------- synthetic "cells" in a 2-D stand-in for PCA space ----------
const TYPES = [
  { name: 'CD4 T', n: 80, c: [-0.45, 0.32], sd: [0.13, 0.13], a: 0 },
  { name: 'CD8 T', n: 55, c: [-0.13, 0.5], sd: [0.1, 0.1], a: 0 },
  { name: 'NK', n: 30, c: [0.07, 0.76], sd: [0.07, 0.07], a: 0 },
  { name: 'B', n: 55, c: [-0.5, -0.5], sd: [0.11, 0.11], a: 0 },
  { name: 'Monocyte', n: 60, c: [0.48, -0.2], sd: [0.22, 0.07], a: 60 },
  { name: 'DC', n: 20, c: [0.72, 0.3], sd: [0.06, 0.06], a: 0 },
];
function makeCells() {
  const r = rng(31), out = [];
  TYPES.forEach((t, k) => {
    const th = t.a * Math.PI / 180;
    for (let i = 0; i < t.n; i++) {
      const u = r.normal(0, t.sd[0]), v = r.normal(0, t.sd[1]);
      out.push({ x: t.c[0] + u * Math.cos(th) - v * Math.sin(th), y: t.c[1] + u * Math.sin(th) + v * Math.cos(th), type: k });
    }
  });
  return out;
}

/** Union kNN graph (unweighted, symmetric). order[i] = other nodes sorted by distance. */
function knnGraph(order, n, k) {
  const set = new Set();
  for (let i = 0; i < n; i++) for (let q = 0; q < k; q++) { const j = order[i][q]; set.add(i < j ? i * n + j : j * n + i); }
  const edges = [...set].map((key) => [Math.floor(key / n), key % n]);
  return edges;
}

/**
 * Louvain: repeated (local moving → aggregation). Returns the partition of the original nodes after every level.
 * Quality = modularity with resolution γ: Q = (1/2m) Σ_ij [A_ij − γ k_i k_j / 2m] δ(c_i, c_j).
 */
function louvain(n, edges, gamma, seed = 1) {
  const r = rng(seed);
  let adj = Array.from({ length: n }, () => new Map());
  edges.forEach(([i, j]) => { adj[i].set(j, (adj[i].get(j) || 0) + 1); adj[j].set(i, (adj[j].get(i) || 0) + 1); });
  const m2 = 2 * edges.length;
  let member = Int32Array.from({ length: n }, (_, i) => i); // original node -> current community id
  const levels = [];
  for (let level = 0; level < 20; level++) {
    const N = adj.length, deg = new Float64Array(N), comm = Int32Array.from({ length: N }, (_, i) => i), tot = new Float64Array(N);
    for (let u = 0; u < N; u++) { for (const w of adj[u].values()) deg[u] += w; tot[u] = deg[u]; }
    let movedAny = false;
    const nodes = r.shuffle(d3.range(N));
    for (let pass = 0; pass < 50; pass++) {
      let moved = false;
      for (const u of nodes) {
        const cu = comm[u], kin = new Map();
        for (const [v, w] of adj[u]) if (v !== u) kin.set(comm[v], (kin.get(comm[v]) || 0) + w);
        tot[cu] -= deg[u];
        let best = cu, bestGain = (kin.get(cu) || 0) - gamma * tot[cu] * deg[u] / m2;
        for (const [c, w] of kin) { const g = w - gamma * tot[c] * deg[u] / m2; if (g > bestGain + 1e-12) { bestGain = g; best = c; } }
        tot[best] += deg[u];
        if (best !== cu) { comm[u] = best; moved = true; movedAny = true; }
      }
      if (!moved) break;
    }
    if (!movedAny) break;
    // renumber and aggregate
    const ids = new Map(); comm.forEach((c) => { if (!ids.has(c)) ids.set(c, ids.size); });
    const newAdj = Array.from({ length: ids.size }, () => new Map());
    for (let u = 0; u < N; u++) { const cu = ids.get(comm[u]); for (const [v, w] of adj[u]) { const cv = ids.get(comm[v]); newAdj[cu].set(cv, (newAdj[cu].get(cv) || 0) + w); } }
    member = member.map((c) => ids.get(comm[c]));
    levels.push(Int32Array.from(member));
    adj = newAdj;
  }
  if (!levels.length) levels.push(Int32Array.from(member));
  return levels;
}
function modularity(n, edges, part, gamma) {
  const m2 = 2 * edges.length, deg = new Float64Array(n);
  edges.forEach(([i, j]) => { deg[i]++; deg[j]++; });
  const K = d3.max(part) + 1, inW = new Float64Array(K), tot = new Float64Array(K);
  edges.forEach(([i, j]) => { if (part[i] === part[j]) inW[part[i]] += 2; });
  for (let i = 0; i < n; i++) tot[part[i]] += deg[i];
  let q = 0; for (let c = 0; c < K; c++) q += inW[c] / m2 - gamma * (tot[c] / m2) ** 2;
  return q;
}
function ari(a, b) {
  const n = a.length, tab = new Map(), ra = new Map(), rb = new Map(), c2 = (x) => x * (x - 1) / 2;
  for (let i = 0; i < n; i++) { const k = a[i] + ',' + b[i]; tab.set(k, (tab.get(k) || 0) + 1); ra.set(a[i], (ra.get(a[i]) || 0) + 1); rb.set(b[i], (rb.get(b[i]) || 0) + 1); }
  let s = 0; for (const v of tab.values()) s += c2(v);
  let sa = 0; for (const v of ra.values()) sa += c2(v);
  let sb = 0; for (const v of rb.values()) sb += c2(v);
  const exp = sa * sb / c2(n), mx = (sa + sb) / 2;
  return mx === exp ? 1 : (s - exp) / (mx - exp);
}
/** Number of clusters that are not connected within themselves (Louvain can produce these; Leiden cannot). */
function disconnected(n, edges, part) {
  const parent = Int32Array.from({ length: n }, (_, i) => i), find = (x) => { while (parent[x] !== x) x = parent[x] = parent[parent[x]]; return x; };
  edges.forEach(([i, j]) => { if (part[i] === part[j]) parent[find(i)] = find(j); });
  const roots = new Map(); for (let i = 0; i < n; i++) { const c = part[i]; if (!roots.has(c)) roots.set(c, new Set()); roots.get(c).add(find(i)); }
  let bad = 0; for (const s of roots.values()) if (s.size > 1) bad++;
  return bad;
}

export default async function render(main) {
  header(main, 'kNN graphs & Leiden clustering',
    'Single-cell tools do not cluster points directly. They first connect each cell to its <b>nearest neighbours</b>, then look for <b>communities</b>: groups of cells with many more links among themselves than chance would predict. The <b>resolution</b> parameter decides how many communities you get.');

  const cells = makeCells(), n = cells.length, types = cells.map((c) => c.type);
  const order = d3.range(n).map((i) => d3.range(n).filter((j) => j !== i).sort((a, b) => ((cells[a].x - cells[i].x) ** 2 + (cells[a].y - cells[i].y) ** 2) - ((cells[b].x - cells[i].x) ** 2 + (cells[b].y - cells[i].y) ** 2)));

  const fig = figure(main, 'Build the graph, then let Louvain find communities',
    `300 simulated immune cells in six types. Positions stand in for PCA space: here the graph is built from these 2-D coordinates, while real pipelines use 30–50 PCs. Lines are kNN edges. <b>Step</b> replays Louvain one level at a time: each level moves cells into the neighbouring community that most increases modularity, then merges each community into a single node. In the intermediate levels, only the largest fragment of each final cluster is coloured, and other fragments are grey.`);
  let k = 10, gamma = 1, colourBy = 'cluster', edges = [], levels = [], shown = 0, final = null, slotOf = null, stopAnim = null;
  slider(fig.controls, { label: 'k (neighbours)', min: 2, max: 30, step: 1, value: k, onInput: (v) => { k = v; rebuild(); } });
  slider(fig.controls, { label: 'Resolution γ', min: 0.05, max: 3, step: 0.05, value: gamma, fmt: (v) => v.toFixed(2), onInput: (v) => { gamma = v; recluster(); } });
  select(fig.controls, { label: 'Colour by', value: 'cluster', options: [['cluster', 'Louvain cluster'], ['type', 'true cell type']], onChange: (v) => { colourBy = v; paint(); } });
  button(fig.controls, 'Replay levels', () => replay(), true);
  button(fig.controls, 'Step', () => { if (stopAnim) { stopAnim(); stopAnim = null; } shown = Math.min(levels.length, shown + 1); paint(); });
  const readout = stats(fig.root);
  const leg = el('div'); fig.root.append(leg);
  const row = el('div', { class: 'fig-row' }); fig.body.append(row);
  const lp = el('div'), rp = el('div'); row.append(lp, rp);
  const F = frame(lp, { width: 480, height: 450, x: [-0.85, 1.0], y: [-0.85, 1.0], axes: false, grid: false, margin: { top: 10, right: 10, bottom: 10, left: 10 } });
  F.svg.append('rect').attr('x', 10).attr('y', 10).attr('width', F.w).attr('height', F.h).attr('fill', 'none').attr('stroke', css('--grid'));
  const edgePath = F.layer('edges').append('path').attr('fill', 'none').attr('stroke', muted()).attr('stroke-width', 0.7);
  const intraPath = F.layer('edges').append('path').attr('fill', 'none').attr('stroke', ink2()).attr('stroke-width', 0.9).attr('stroke-opacity', 0.55);
  const dots = F.layer('pts').selectAll('circle').data(cells).join('circle').attr('cx', (d) => F.x(d.x)).attr('cy', (d) => F.y(d.y)).attr('r', 4.2)
    .attr('stroke', css('--surface')).attr('stroke-width', 1);
  const R = frame(rp, { width: 440, height: 450, x: [0, 3], y: [0, 20], xLabel: 'resolution γ', yLabel: 'number of clusters' });

  function rebuild() { edges = knnGraph(order, n, k); sweep(); recluster(); }
  function recluster() {
    levels = louvain(n, edges, gamma); shown = levels.length; final = levels[levels.length - 1];
    // give each final cluster the colour of the true type it mostly contains (so colours are comparable across settings)
    const K = d3.max(final) + 1, counts = d3.range(K).map(() => new Array(TYPES.length).fill(0)), size = new Array(K).fill(0);
    for (let i = 0; i < n; i++) { counts[final[i]][types[i]]++; size[final[i]]++; }
    slotOf = new Array(K).fill(-1); const used = new Set();
    d3.range(K).sort((a, b) => size[b] - size[a]).forEach((c) => {
      const pref = d3.range(TYPES.length).sort((a, b) => counts[c][b] - counts[c][a]);
      const s = pref.find((t) => !used.has(t) && counts[c][t] > 0);
      if (s != null) { slotOf[c] = s; used.add(s); } else { const free = [6, 7].find((t) => !used.has(t)); if (free != null) { slotOf[c] = free; used.add(free); } }
    });
    paint(); paintCurve();
  }
  function colourOfNode(i) {
    if (colourBy === 'type') return cat(types[i]);
    if (shown === 0) return muted();
    const part = levels[shown - 1];
    if (shown === levels.length) return slotOf[part[i]] >= 0 ? cat(slotOf[part[i]]) : muted();
    // intermediate level: colour a community only if it is the biggest piece of its final cluster
    return bigPiece(part).has(part[i]) ? (slotOf[final[i]] >= 0 ? cat(slotOf[final[i]]) : muted()) : muted();
  }
  let bpCache = null;
  function bigPiece(part) {
    if (bpCache && bpCache.part === part) return bpCache.set;
    const cnt = new Map(); for (let i = 0; i < n; i++) { const key = final[i] + ',' + part[i]; cnt.set(key, (cnt.get(key) || 0) + 1); }
    const best = new Map(); for (const [key, c] of cnt) { const [f, p] = key.split(',').map(Number); if (!best.has(f) || c > best.get(f)[1]) best.set(f, [p, c]); }
    const set = new Set([...best.values()].map((v) => v[0])); bpCache = { part, set }; return set;
  }
  function paint() {
    const part = shown ? levels[shown - 1] : Int32Array.from({ length: n }, (_, i) => i);
    let dIn = '', dOut = '';
    edges.forEach(([i, j]) => {
      const s = `M${F.x(cells[i].x).toFixed(1)},${F.y(cells[i].y).toFixed(1)}L${F.x(cells[j].x).toFixed(1)},${F.y(cells[j].y).toFixed(1)}`;
      if (part[i] === part[j]) dIn += s; else dOut += s;
    });
    intraPath.attr('d', dIn); edgePath.attr('d', dOut).attr('stroke-opacity', 0.35);
    dots.attr('fill', (d, i) => colourOfNode(i))
      .on('mousemove', (e, d) => { const i = cells.indexOf(d); tip(e, `true type: <b>${TYPES[d.type].name}</b><br>${shown ? `community ${part[i] + 1} at level ${shown}` : 'own community (start)'}<br>degree ${edges.filter(([a, b]) => a === i || b === i).length}`); })
      .on('mouseleave', untip);
    const K = d3.max(part) + 1;
    readout.set({
      level: `${shown} / ${levels.length}`,
      communities: K,
      'modularity Q': modularity(n, edges, part, gamma).toFixed(3),
      'agreement with true types (ARI)': ari(part, types).toFixed(2),
      'disconnected clusters': disconnected(n, edges, part),
      'graph edges': edges.length,
    });
    leg.innerHTML = '';
    if (colourBy === 'type') legend(leg, TYPES.map((t, i) => [t.name, cat(i)]));
    else legend(leg, [['each cluster takes the colour of the true type it mostly contains', ink2()], ['grey: extra clusters / unmerged fragments', muted()]]);
  }
  let curve = [];
  function sweep() {
    curve = d3.range(0.05, 3.001, 0.05).map((g) => { const L = louvain(n, edges, g); const p = L[L.length - 1]; return { x: g, y: d3.max(p) + 1, a: ari(p, types) }; });
  }
  function paintCurve() {
    R.y.domain([0, Math.max(10, d3.max(curve, (d) => d.y) + 2)]);
    R.g.select('.axis.y').call(d3.axisLeft(R.y).ticks(6)); R.g.select('.gridline').call(d3.axisLeft(R.y).ticks(6).tickSize(-R.w).tickFormat(''));
    R.layer('ref').selectAll('line').data([6]).join('line').attr('x1', 0).attr('x2', R.w).attr('y1', (d) => R.y(d)).attr('y2', (d) => R.y(d)).attr('stroke', ink2()).attr('stroke-dasharray', '5 4');
    R.layer('ref').selectAll('text').data([6]).join('text').attr('x', R.w - 4).attr('y', (d) => R.y(d) - 5).attr('text-anchor', 'end').attr('font-size', 12).attr('fill', ink2()).text('6 true cell types');
    R.layer('c').selectAll('path').data([curve]).join('path').attr('fill', 'none').attr('stroke', cat(0)).attr('stroke-width', 2)
      .attr('d', d3.line().curve(d3.curveStepAfter).x((d) => R.x(d.x)).y((d) => R.y(d.y)));
    R.layer('pts').selectAll('circle').data(curve).join('circle').attr('cx', (d) => R.x(d.x)).attr('cy', (d) => R.y(d.y)).attr('r', 6).attr('fill', 'transparent')
      .on('mousemove', (e, d) => tip(e, `γ = ${d.x.toFixed(2)}: ${d.y} clusters<br>ARI vs true types ${d.a.toFixed(2)}`)).on('mouseleave', untip);
    const cur = { x: gamma, y: d3.max(final) + 1 };
    R.layer('cur').selectAll('circle').data([cur]).join('circle').attr('cx', (d) => R.x(d.x)).attr('cy', (d) => R.y(d.y)).attr('r', 7).attr('fill', cat(1)).attr('stroke', css('--surface')).attr('stroke-width', 2);
  }
  function replay() {
    if (stopAnim) stopAnim();
    shown = 0; paint(); let acc = 0;
    stopAnim = animate((dt) => { acc += dt; if (acc < 900) return true; acc = 0; shown++; paint(); if (shown >= levels.length) { stopAnim = null; return false; } return true; });
  }
  rebuild();

  section(main, 'What to notice', `<ul>
    <li>At the defaults (k = 10, γ = 1) Louvain finds the big populations without being told how many there are. Press <b>Replay levels</b>: level 1 already groups cells into small local communities, and each later level merges them into larger ones. Modularity rises at every level.</li>
    <li>Drag <b>resolution</b>. Low γ (≈0.1) merges CD4 T, CD8 T and NK into one "T/NK" cluster. High γ (≈3) splits real populations into pieces. The right panel shows the cluster count climbing in steps as γ grows. No setting is "correct", because resolution chooses the level of granularity you are asking about.</li>
    <li>Switch <b>Colour by</b> to true cell type and compare. The small DC population next to the monocytes and the NK cells next to CD8 T cells are the first to be merged at low resolution: the graph has many edges between touching populations.</li>
    <li>Set <b>k = 2</b>: the graph is sparse and splits into many small components, and every component becomes its own cluster. At <b>k = 30</b> each cell links far across population borders, and neighbouring types blur together.</li>
    <li>Watch <b>disconnected clusters</b>. Louvain can occasionally put cells into one community even though they are not connected inside it. Leiden's refinement step exists to rule that out.</li>
  </ul>`);

  steps(main, 'How it works', [
    'Reduce the data first (normalise → HVGs → PCA, 30–50 PCs).',
    'Build a <b>kNN graph</b>: link every cell to its k nearest cells in PCA space. scanpy weights the edges with UMAP\'s fuzzy memberships, and Seurat uses <b>shared nearest neighbours</b> (SNN), weighting each edge by how many neighbours the two cells share.',
    '<b>Modularity</b> compares the number of edges inside communities with the number expected if edges were rewired at random while keeping each cell\'s degree. Resolution γ scales that expectation: larger γ demands denser communities, so you get more and smaller ones.',
    '<b>Louvain, phase 1 (local moving):</b> visit cells in random order and move each one into the neighbouring community that most increases modularity. Repeat until nothing moves.',
    '<b>Phase 2 (aggregation):</b> collapse each community into a single node, with edge weights summed, and repeat phase 1 on this smaller graph. Stop when no merge helps.',
    '<b>Leiden</b> adds a <b>refinement</b> phase between them. Each community is re-split into well-connected sub-communities before aggregation. That guarantees every cluster is internally connected, and in practice it is also faster and finds better partitions.',
  ]);
  math(main, 'Q(γ) = (1/2m) Σ_ij [ A_ij − γ·k_i k_j / 2m ] · δ(c_i, c_j)\ngain of moving cell i into community C:   k_i,in(C) − γ · Σ_tot(C) · k_i / 2m\n(A = adjacency, k_i = degree of i, m = number of edges, Σ_tot = summed degree of C)');

  callout(main, '', 'In biology', `This is the default clustering in single-cell analysis. scanpy runs <code>sc.pp.neighbors(adata, n_neighbors=15, n_pcs=50)</code> → <code>sc.tl.leiden(adata, resolution=1.0)</code>, and Seurat runs <code>FindNeighbors(dims = 1:30)</code> → <code>FindClusters(resolution = 0.8)</code>, which uses Louvain by default and Leiden with <code>algorithm = 4</code>. Graphs suit this data: they need no assumption about cluster shape or number, they scale to millions of cells because each cell has only k edges, and they follow curved manifolds that k-means would cut through. People commonly sweep resolution (0.2–2) and check stability with <code>clustree</code>. Sub-clustering a population at higher resolution (for example, T cells into CD4, CD8, Treg and exhausted) is routine. The same graph also drives UMAP and PAGA, and spatial methods add edges between physically adjacent cells.`);
  callout(main, 'warn', 'Pitfalls', `<ul>
    <li><b>Clusters are not cell types.</b> Resolution is a dial you choose, not a discovery. Label clusters with marker genes and merge or split them on biological grounds.</li>
    <li>Results depend on k, the number of PCs, the HVG set and the random seed. Check that key clusters survive changes to all of them.</li>
    <li>Batch effects create batch-specific neighbours and therefore batch-specific clusters. Integrate first (Harmony, scVI, BBKNN).</li>
    <li>Do not run differential expression between clusters and then call the genes "discovered". The clusters were defined by those very genes (double dipping).</li>
    <li>Doublets and low-quality cells form their own small "clusters" at high resolution.</li>
  </ul>`);
  seeAlso(main, [['concepts', 'kmeans', 'k-means'], ['concepts', 'tsne-umap', 't-SNE & UMAP'], ['concepts', 'pca', 'PCA'], ['concepts', 'knn', 'k-nearest neighbours'], ['omics', 'sc-clustering', 'Graph clustering & UMAP']]);
}
