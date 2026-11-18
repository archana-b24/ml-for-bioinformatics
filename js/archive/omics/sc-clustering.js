// Tab 2 · Single-cell step 3: kNN graph, Leiden community detection, UMAP, and the resolution question.
import { header, figure, slider, select, checkbox, stats, section, steps, callout, math, legend, seeAlso, el, tip, untip } from '../../lib/ui.js';
import { frame, cat, css, muted, sequential } from '../../lib/plot.js';
import { stepCard, omicsData } from '../../omics/_shared.js';
import { canvasScatter, CT_COLOR, clusterColor, ext, f2, pct } from '../../omics/_sc_common.js';
const d3 = window.d3;

export default async function render(main) {
  header(main, 'Graph clustering & UMAP',
    'Clustering single cells is <b>community detection on a nearest-neighbour graph</b>: connect every cell to its most similar cells, then find groups that are densely connected inside and sparsely between. The resolution parameter decides how finely to cut, and the data cannot tell you the "right" value.');
  stepCard(main, {
    stage: 'Single-cell · step 3 of 7',
    question: 'Which groups of cells exist in these tumours: how many cell types and states?',
    data: 'The top 20 principal components of each cell (from step 2).',
    method: 'k-nearest-neighbour graph (k = 15) in PCA space → Leiden community detection at several resolutions; UMAP only for drawing.',
    verdict: 'ml',
    why: 'Unsupervised learning: there are no labels, and the goal is to discover structure. But clusters are a hypothesis, not a result. The number of clusters is a modelling choice that biology (marker genes) has to justify.',
  });
  const [C, I] = await Promise.all([omicsData(main, 'sc_cells.json'), omicsData(main, 'sc_integration.json')]);
  if (!C || !I) return;
  const RES = I.res;
  const n = C.n;
  const NT = C.cell_types.length;

  // ---------------- figure 1: resolution ----------------
  const fig = figure(main, 'One graph, many clusterings',
    `UMAP of ${n.toLocaleString()} of ${C.n_kept.toLocaleString()} cells, coloured by Leiden cluster. Right: what each cluster contains (rows = clusters, columns = simulated true cell type; the last column is the share of cells from the 3′ v3 chemistry). Hover a cell to see its nearest neighbours in the graph, which live in 20-dimensional PCA space, not in the 2-D picture.`);
  let emb = 'uncorrected', ri = RES.indexOf(0.1) >= 0 ? RES.indexOf(0.1) : 3, colourMode = 'cluster', showEdges = true;
  select(fig.controls, { label: 'Embedding', value: 'uncorrected', options: [['uncorrected', 'PCA as is (this step)'], ['harmony', 'after batch integration (next step)']], onChange: (v) => { emb = v; paint(); } });
  slider(fig.controls, { label: 'Leiden resolution', min: 0, max: RES.length - 1, step: 1, value: ri, fmt: (v) => String(RES[v]), onInput: (v) => { ri = v; paint(); } });
  select(fig.controls, { label: 'Colour by', value: 'cluster', options: [['cluster', 'Leiden cluster'], ['type', 'true cell type'], ['batch', 'chemistry']], onChange: (v) => { colourMode = v; paint(); } });
  checkbox(fig.controls, { label: 'show kNN edges on hover', checked: true, onChange: (c) => { showEdges = c; } });
  const rd = stats(fig.root);
  const row = el('div', { class: 'fig-row' }); fig.body.append(row);
  const pA = el('div'), pB = el('div'); row.append(pA, pB);
  const knn = C.knn ? C.knn : null;
  let U = null;
  const A = canvasScatter(pA, { width: 480, height: 440, x: [0, 1], y: [0, 1], axes: false, grid: false, margin: { top: 8, right: 8, bottom: 8, left: 8 },
    onMiss: () => A.F.layer('edges').selectAll('line').remove(),
    hover: (i) => {
      const lab = C.leiden[emb][String(RES[ri])][i];
      drawEdges(i);
      let same = '';
      if (knn) { const nb = knn[emb][i]; const sb = nb.filter((j) => C.batch[j] === C.batch[i]).length; same = `<br>${sb}/${nb.length} graph neighbours from the same chemistry`; }
      return `<b>cluster ${lab}</b> · true type ${C.cell_types[C.type[i]] || 'doublet (missed)'}<br>${C.patients[C.patient[i]]} · ${C.batches[C.batch[i]]}${same}`;
    } });
  function drawEdges(i) {
    if (!knn || !showEdges) { A.F.layer('edges').selectAll('line').remove(); return; }
    const nb = knn[emb][i];
    A.F.layer('edges').selectAll('line').data(nb).join('line')
      .attr('x1', A.F.x(U[0][i])).attr('y1', A.F.y(U[1][i])).attr('x2', (j) => A.F.x(U[0][j])).attr('y2', (j) => A.F.y(U[1][j]))
      .attr('stroke', css('--ink')).attr('stroke-width', 1.2).attr('stroke-opacity', 0.8);
  }
  const legA = el('div'); pA.append(legA);
  const heatBox = el('div'); pB.append(heatBox);

  function contingency(lab) {
    const K = d3.max(lab) + 1;
    const M = d3.range(K).map(() => new Array(NT).fill(0));
    const v3 = new Array(K).fill(0), tot = new Array(K).fill(0);
    lab.forEach((c, i) => { tot[c]++; v3[c] += C.batch[i]; if (C.type[i] >= 0) M[c][C.type[i]]++; });
    return { K, M, v3, tot };
  }
  function paint() {
    U = C.umap[emb];
    A.F.x.domain(ext(U[0])); A.F.y.domain(ext(U[1]));
    const lab = C.leiden[emb][String(RES[ri])];
    const col = colourMode === 'cluster' ? (i) => clusterColor(lab[i]) : colourMode === 'type' ? (i) => CT_COLOR(C.type[i]) : (i) => cat(C.batch[i]);
    A.draw({ idx: d3.range(n), x: U[0], y: U[1], color: col, r: 2.2, alpha: 0.85 });
    legA.innerHTML = '';
    if (colourMode === 'type') legend(legA, C.cell_types.map((t, k) => [t, CT_COLOR(k)]));
    else if (colourMode === 'batch') legend(legA, C.batches.map((b, k) => [b, cat(k)]));
    const { K, M, v3, tot } = contingency(lab);
    if (colourMode === 'cluster') legend(legA, d3.range(K).map((k) => [String(k), clusterColor(k)]));
    // heatmap: share of each cluster by true type + v3 share
    heatBox.innerHTML = '';
    const cell = Math.min(26, Math.floor(360 / Math.max(K, 8)));
    const rows = M.map((r, k) => [...r.map((v) => v / Math.max(1, tot[k])), v3[k] / Math.max(1, tot[k])]);
    const left = 96, top = 92, W = left + (NT + 1) * 34 + 10, H = top + K * cell + 6;
    const svg = d3.select(heatBox).append('svg').attr('viewBox', `0 0 ${W} ${H}`).attr('width', W);
    const sq = sequential(0, 1);
    rows.forEach((r, k) => {
      svg.append('rect').attr('x', 4).attr('y', top + k * cell + 2).attr('width', 12).attr('height', cell - 4).attr('rx', 2).attr('fill', clusterColor(k));
      svg.append('text').attr('x', left - 8).attr('y', top + k * cell + cell / 2 + 4).attr('text-anchor', 'end').attr('font-size', 11).attr('fill', css('--ink-2')).text(`cl ${k} (${tot[k]})`);
      r.forEach((v, j) => {
        svg.append('rect').attr('x', left + j * 34 + (j === NT ? 6 : 0)).attr('y', top + k * cell).attr('width', 32).attr('height', cell - 2).attr('rx', 2)
          .attr('fill', j === NT ? d3.interpolateRgb(css('--surface-2'), cat(1))(v) : sq(v))
          .on('mousemove', (e) => tip(e, j === NT ? `cluster ${k}: ${pct(v)} of cells from 3′ v3` : `cluster ${k}: ${pct(v)} ${C.cell_types[j]}`)).on('mouseleave', untip);
      });
    });
    [...C.cell_types, '% v3'].forEach((t, j) => svg.append('text').attr('transform', `translate(${left + j * 34 + 18 + (j === NT ? 6 : 0)},${top - 6}) rotate(-55)`).attr('font-size', 11).attr('fill', css('--ink-2')).text(t));
    const m = I.metrics[emb];
    const batchy = d3.range(K).filter((k) => tot[k] >= 10 && (v3[k] / tot[k] > 0.9 || v3[k] / tot[k] < 0.1)).length;
    rd.set({ resolution: RES[ri], clusters: m.n_clusters[ri], 'ARI vs true types': f2(m.ari[ri]), 'clusters >90% one chemistry': batchy });
    markRes();
  }

  // ---------------- figure 2: ARI vs resolution ----------------
  const fig2 = figure(main, 'The resolution question',
    'Adjusted Rand index (ARI) between Leiden clusters and the simulated cell types (1 = identical partitions, 0 = chance), and the number of clusters, across resolutions. In real data there is no truth column; you would judge clusters by marker genes, stability and biology.');
  const row2 = el('div', { class: 'fig-row' }); fig2.body.append(row2);
  const qA = el('div'), qB = el('div'); row2.append(qA, qB);
  const xs = d3.scaleLog();
  const R1 = frame(qA, { width: 440, height: 300, x: [RES[0], RES[RES.length - 1]], y: [0, 1.05], xScale: xs, xLabel: 'Leiden resolution (log scale)', yLabel: 'ARI vs true types' });
  const R2 = frame(qB, { width: 440, height: 300, x: [RES[0], RES[RES.length - 1]], y: [0, d3.max(I.metrics.uncorrected.n_clusters) + 2], xScale: d3.scaleLog(), xLabel: 'Leiden resolution (log scale)', yLabel: 'number of clusters' });
  [R1, R2].forEach((R) => R.g.select('.axis.x').call(d3.axisBottom(R.x).tickValues(RES.filter((r, i) => i % 2 === 0 || r === 2)).tickFormat(d3.format('~g'))));
  R2.layer('truth').append('line').attr('x1', 0).attr('x2', R2.w).attr('y1', R2.y(NT)).attr('y2', R2.y(NT)).attr('stroke', muted()).attr('stroke-dasharray', '4 4');
  R2.layer('truth').append('text').attr('x', 6).attr('y', R2.y(NT) + 15).attr('text-anchor', 'start').attr('font-size', 11).attr('fill', css('--ink-2')).text(`${NT} true types`);
  const series = [['uncorrected', 'PCA as is', cat(0)], ['harmony', 'after integration', cat(2)]];
  for (const [k, , c] of series) {
    for (const [R, key] of [[R1, 'ari'], [R2, 'n_clusters']]) {
      const pts = RES.map((r, i) => [r, I.metrics[k][key][i]]);
      R.layer('l').append('path').datum(pts).attr('fill', 'none').attr('stroke', c).attr('stroke-width', 2).attr('d', d3.line().x((d) => R.x(d[0])).y((d) => R.y(d[1])));
      R.layer('p').selectAll(null).data(pts).join('circle').attr('cx', (d) => R.x(d[0])).attr('cy', (d) => R.y(d[1])).attr('r', 4).attr('fill', c).attr('stroke', css('--surface')).attr('stroke-width', 1.5)
        .on('mousemove', (e, d) => tip(e, `${k === 'uncorrected' ? 'PCA as is' : 'after integration'} · resolution ${d[0]}: ${key === 'ari' ? 'ARI ' + f2(d[1]) : d[1] + ' clusters'}`)).on('mouseleave', untip);
    }
  }
  legend(fig2.body, series.map(([, l, c]) => [l, c]));
  function markRes() {
    for (const R of [R1, R2]) R.layer('mark').selectAll('line').data([RES[ri]]).join('line').attr('x1', (d) => R.x(d)).attr('x2', (d) => R.x(d)).attr('y1', 0).attr('y2', R.h).attr('stroke', css('--ink')).attr('stroke-width', 1.5).attr('stroke-dasharray', '4 4');
  }
  paint();

  const mu = I.metrics.uncorrected, mh = I.metrics.harmony;
  const iLow = 0, iDef = RES.indexOf(0.1), iHi = RES.length - 1;
  const batchyDef = I.clusters.uncorrected[String(RES[iDef])].filter((c) => c.n >= 10 && (c.frac_v3 > 0.9 || c.frac_v3 < 0.1)).length;
  const firstSplitH = RES.findIndex((r, i) => mh.n_clusters[i] > NT);
  section(main, 'What to notice', `<ul>
    <li>At resolution ${RES[iDef]} on the uncorrected PCA you get ${mu.n_clusters[iDef]} clusters for ${NT} true cell types (ARI ${f2(mu.ari[iDef])}). Colour by <b>chemistry</b>: ${batchyDef} of these clusters are more than 90% one chemistry, i.e. cell types split in two, one cluster per chemistry. Those are batch clusters, not biology, and no resolution fixes that: at ${RES[iLow]} you get ${mu.n_clusters[iLow]} clusters, but they are not the ${NT} types (ARI ${f2(mu.ari[iLow])}): whole cell types get lumped together while others stay split by chemistry.</li>
    <li>Hover a cell: its graph neighbours are almost all from its own chemistry. Leiden can only find what the graph contains, and here the graph encodes the batch.</li>
    <li>Switch to the <b>integrated</b> embedding: the ${NT} types are recovered exactly (ARI ${f2(mh.ari[iDef])}) over a wide range of resolutions. ${firstSplitH >= 0 ? `From ${RES[firstSplitH]} upward Leiden starts splitting real types (${mh.n_clusters[firstSplitH]} clusters at ${RES[firstSplitH]}, ${mh.n_clusters[iHi]} at ${RES[iHi]}) into sub-groups that may or may not be meaningful states.` : ''}</li>
    <li>The ARI = 1 here is a property of the toy: simulated cell types are cleanly separated. Real tumours contain continua (naive → exhausted T cells, CAF subtypes) where any cut is a choice.</li>
    <li>UMAP distances between islands mean little. Two clusters far apart on the plot may be neighbours in PCA space; edges drawn on hover show the real graph.</li>
  </ul>`);
  steps(main, 'How it works', [
    'For each cell, find its k nearest neighbours (k ≈ 10–30) by Euclidean distance on the top PCs.',
    'Turn neighbour lists into a weighted graph (Scanpy uses UMAP\'s fuzzy weights; Seurat uses shared-nearest-neighbour overlap). Here: a plain kNN graph.',
    '<b>Leiden</b> moves cells between communities to maximise <b>modularity</b> (edges inside communities minus edges expected by chance), then refines communities so each is internally connected (the fix over the older Louvain method).',
    'The <b>resolution</b> γ scales the "expected by chance" term: larger γ penalises big communities more, so you get more, smaller clusters.',
    'UMAP builds a similar fuzzy kNN graph and lays it out in 2-D for display. Cluster on the graph, never on the UMAP coordinates.',
  ]);
  math(main, `Q(γ) = Σ_c [ e_c − γ · K_c² / (4m) ]
  e_c = edges inside community c,  K_c = total degree of c,  m = total edges
ARI = (RI − E[RI]) / (max RI − E[RI])     agreement of two partitions, corrected for chance`);
  callout(main, '', 'In biology', `<b>Scanpy</b>: <code>sc.pp.neighbors</code> → <code>sc.tl.leiden(resolution=…)</code> → <code>sc.tl.umap</code>; <b>Seurat</b>: <code>FindNeighbors</code> → <code>FindClusters</code> (Louvain/Leiden on an SNN graph) → <code>RunUMAP</code>. Tools for choosing the resolution: <b>clustree</b> (how clusters split across resolutions), silhouette and bootstrap stability, and checking each split for distinct marker genes (<code>sc.tl.rank_genes_groups</code>, <code>FindAllMarkers</code>). Sub-clustering a lineage (e.g. only T cells) at a higher resolution is standard practice.`);
  callout(main, 'warn', 'Pitfalls', `<ul>
    <li><b>Treating cluster count as a finding.</b> "We found 23 clusters" depends on k, the number of PCs and the resolution.</li>
    <li><b>Clustering before checking for batch effects</b>: clusters made of one sample or one chemistry are technical until proven otherwise.</li>
    <li><b>Reading UMAP</b>: island sizes, gaps and distances are distorted; density is not preserved. Use it to look, not to measure.</li>
    <li><b>Double dipping</b>: finding clusters and then testing marker genes between the same clusters gives p-values that are far too small, because the clusters were chosen to differ.</li>
  </ul>`);
  seeAlso(main, [['omics', 'sc-integration', 'Next: batch integration'], ['concepts', 'graph-clustering', 'Graph clustering'], ['concepts', 'tsne-umap', 't-SNE & UMAP'], ['concepts', 'knn', 'k-nearest neighbours']]);
}
