// t-SNE and UMAP, lesson format.
// Arc: what → why → where → how it works (6 steps: neighbours → perplexity → t-SNE animated → UMAP → what is preserved → map lab) → pitfalls → recap.
// Data: 600 simulated single cells (20 principal components, 8 cell types, 2 batches, known geometry) from the same fictional tumour
// biopsies, plus a structure-free noise control. Embeddings are precomputed with scikit-learn and umap-learn (scripts/build_tsne_umap.py).
// Every number in the prose is computed from the data at render time.
import { header, figure, slider, select, button, stats, legend, seeAlso, animate, tip, untip, loadJSON, el, html } from '../lib/ui.js';
import { frame, cat, css, ink, ink2, muted, sequential } from '../lib/plot.js';
import { rng, mean } from '../lib/data.js';
import { loadCohort, studyCard } from '../lib/study.js';
import { lessonNav, chapter, prose, rung, worked, quiz, cards, table, code, recap } from '../lib/lesson.js';
const d3 = window.d3;

const f1 = (v) => v.toFixed(1);
const f2 = (v) => v.toFixed(2);
const sum = (a) => a.reduce((s, v) => s + v, 0);
const fmtLabel = { tsne: 't-SNE', umap: 'UMAP', pca: 'PCA' };

export default async function render(main) {
  const C = await loadCohort();
  const S = await loadJSON('data/study/tsne_umap.json');
  const TY = S.types, n = S.lab.length, lab = S.lab, X = S.X, K = 15, DIM = S.d;
  const nT = TY.length, countT = TY.map((_, t) => lab.filter((l) => l === t).length);
  const find = (ds, m, p, md = null, seed = 1, init = null) => S.embeds.find((e) => e.ds === ds && e.m === m && e.p === p && e.md === md && e.seed === seed && (init ? e.init === init : e.init !== 'pca'));
  const tsne30 = find('cells', 'tsne', 30), umap15 = find('cells', 'umap', 15, 0.5), tsne30pca = find('cells', 'tsne', 30, null, 1, 'pca');
  const pcaE = { xy: S.pca, knn: S.knn_pca, ari: S.ari_pca, m: 'pca' };
  const pcaN = { xy: S.pca_noise, m: 'pca' };

  // ---- original-space geometry (20-D) ----
  const D2 = new Float64Array(n * n);
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) { let s = 0; for (let d = 0; d < DIM; d++) { const q = X[i][d] - X[j][d]; s += q * q; } D2[i * n + j] = D2[j * n + i] = s; }
  const dist = (i, j) => Math.sqrt(D2[i * n + j]);
  const ORD = d3.range(n).map((i) => d3.range(n).filter((j) => j !== i).sort((a, b) => D2[i * n + a] - D2[i * n + b]));
  const geo = (get, dim) => { // per-type centroid and mean distance to centroid
    const cent = [], spr = [];
    for (let t = 0; t < nT; t++) {
      const idx = d3.range(n).filter((i) => lab[i] === t), c = d3.range(dim).map((d) => mean(idx.map((i) => get(i, d))));
      cent.push(c); spr.push(mean(idx.map((i) => Math.hypot(...c.map((cv, d) => get(i, d) - cv)))));
    }
    const pairs = [];
    for (let a = 0; a < nT; a++) for (let b = a + 1; b < nT; b++) pairs.push({ a, b, d: Math.hypot(...cent[a].map((v, d) => v - cent[b][d])) });
    return { cent, spr, pairs };
  };
  const rank = (a) => { const o = a.map((v, i) => [v, i]).sort((p, q) => p[0] - q[0]), r = []; o.forEach(([, i], k) => { r[i] = k; }); return r; };
  const spearman = (a, b) => { const ra = rank(a), rb = rank(b), ma = mean(ra), mb = mean(rb); let sab = 0, saa = 0, sbb = 0; ra.forEach((v, i) => { sab += (v - ma) * (rb[i] - mb); saa += (v - ma) ** 2; sbb += (rb[i] - mb) ** 2; }); return sab / Math.sqrt(saa * sbb); };
  const G0 = geo((i, d) => X[i][d], DIM);
  const geoOf = (e) => geo((i, d) => e.xy[2 * i + d], 2);
  const relD = (G) => { const m = d3.median(G.pairs, (p) => p.d); return G.pairs.map((p) => p.d / m); };
  const relS = (G) => { const m = mean(G.spr); return G.spr.map((v) => v / m); };
  const rho = (e) => spearman(G0.pairs.map((p) => p.d), geoOf(e).pairs.map((p) => p.d));
  const nnCache = new Map();
  const nnOf = (e) => { // K nearest neighbours of every cell inside a 2-D map
    if (nnCache.has(e)) return nnCache.get(e);
    const out = [];
    for (let i = 0; i < n; i++) {
      const d = new Float64Array(n); for (let j = 0; j < n; j++) d[j] = (e.xy[2 * i] - e.xy[2 * j]) ** 2 + (e.xy[2 * i + 1] - e.xy[2 * j + 1]) ** 2;
      d[i] = Infinity; out.push(Array.from(d3.range(n).sort((a, b) => d[a] - d[b]).slice(0, K)));
    }
    nnCache.set(e, out); return out;
  };
  const agree = (e1, e2) => { const a = nnOf(e1), b = nnOf(e2); return mean(a.map((r, i) => r.filter((j) => b[i].includes(j)).length / K)); };

  // ---- colouring ----
  const MODES = [['type', 'Cell type'], ['batch', 'Batch'], ['pt', 'CD8 T-cell trajectory position']];
  const ptCol = sequential(0, 1);
  const colorFor = (mode) => (mode === 'batch' ? (i) => cat(S.batch[i]) : mode === 'pt' ? (i) => (lab[i] === 1 ? ptCol(S.pt[i]) : muted()) : (i) => cat(lab[i]));
  const opacityFor = (mode) => (mode === 'pt' ? (i) => (lab[i] === 1 ? 1 : 0.22) : () => 0.85);
  const tipFor = (i, ds = 'cells') => (ds === 'noise' ? `cell ${i}<br>no cell type (random data)` : `${TY[lab[i]]} · batch ${S.batch[i] + 1}${lab[i] === 1 ? `<br>trajectory position ${f2(S.pt[i])} (0 = naive-like, 1 = exhausted-like)` : ''}`);
  const typeLegend = () => TY.map((t, i) => [`${t} (${countT[i]})`, cat(i)]);
  const legendFor = (mode) => (mode === 'batch' ? [['batch 1', cat(0)], ['batch 2', cat(1)]] : mode === 'pt' ? [['CD8 T: early', ptCol(0.05)], ['CD8 T: late', ptCol(1)], ['other cell types', muted()]] : typeLegend());

  /** Square map panel without axes (map coordinates have no units). */
  function mapPanel(parent, { width = 460, title, r = 3.4, ds = 'cells', hover = true }) {
    const F = frame(parent, { width, height: width + 18, x: [-56, 56], y: [-56, 56], axes: false, grid: false, margin: { top: 26, right: 8, bottom: 8, left: 8 } });
    F.svg.append('rect').attr('x', 8).attr('y', 26).attr('width', F.w).attr('height', F.h).attr('fill', 'none').attr('stroke', css('--grid'));
    const ttl = F.svg.append('text').attr('x', 10).attr('y', 17).attr('font-size', width < 300 ? 11.5 : 13).attr('font-weight', 600).attr('fill', ink());
    ttl.text(title || '');
    const sel = F.layer('p').selectAll('circle').data(d3.range(n)).join('circle').attr('r', r).attr('stroke', css('--surface')).attr('stroke-width', 0.6);
    if (hover) sel.on('mousemove', (e, i) => tip(e, tipFor(i, ds))).on('mouseleave', untip);
    const P = {
      F, sel, ttl, title: (t) => { ttl.text(t); return P; },
      draw(xy, color = colorFor('type'), op = () => 0.85) { sel.attr('cx', (i) => F.x(xy[2 * i])).attr('cy', (i) => F.y(xy[2 * i + 1])).attr('fill', color).attr('fill-opacity', op); return P; },
    };
    return P;
  }
  const noteTxt = (F, x, y, t, anchor = 'start') => F.layer('note').append('text').attr('x', x).attr('y', y).attr('text-anchor', anchor).attr('font-size', 11.5).attr('fill', ink2()).text(t);

  header(main, 't-SNE and UMAP',
    'Two methods that draw thousands of measurements per cell as a <b>flat map in which each cell lies near the cells most similar to it</b>. The maps show well which cells belong together. They do not reliably show how large a group is, how far apart two groups are, or whether a gap is real.');
  lessonNav(main);
  studyCard(main, C, `<b>Question for this page:</b> the biopsies from this study were profiled cell by cell. Can ${n} of those cells, each described by ${DIM} principal components, be drawn in 2-D so that their cell types become visible, and which features of the picture can be trusted?`);
  prose(main, `<p class="fig-caption">Data on this page: ${n} simulated cells from the same tumours, in ${DIM}-dimensional principal-component space (the standard input to these methods), in ${nT} cell types and 2 sequencing batches. The geometry is planted and therefore known: Tumour cells are widely spread and Treg cells tightly packed; Treg lie next to CD4 T; CD8 T cells form a continuous trajectory rather than a cluster; Endothelial cells are far from everything else. Maps were computed with scikit-learn (t-SNE) and umap-learn (UMAP).</p>`);

  // =====================================================================================
  // 1. WHAT
  // =====================================================================================
  let ch = chapter(main, 'what', 'What are t-SNE and UMAP?', 'In one line: <b>keep each cell\'s nearest neighbours close, and place everything else only approximately</b>.');
  prose(ch, `<p><b>t-SNE</b> (t-distributed stochastic neighbour embedding) and <b>UMAP</b> (uniform manifold approximation and projection) are <b>non-linear dimensionality-reduction</b> methods. Each takes a table with one row per cell and many columns, and gives every cell two coordinates, so that cells which are neighbours in the original space are neighbours on the page. Their output is an <b>embedding</b> (a map).</p>
    <p>They answer a narrower question than PCA does. PCA keeps the directions of largest variance, so it preserves large distances. t-SNE and UMAP keep <b>local neighbourhoods</b> and give up large-scale geometry to do it. The three panels below show the same ${n} cells. Hover over a cell to mark it and its ${K} nearest neighbours in the original ${DIM}-D space, in all three panels.</p>`);
  const f1g = figure(ch, 'The same cells, three ways', `Colour = cell type. Hovering a cell highlights its ${K} nearest neighbours <i>in the ${DIM}-dimensional data</i> in every panel; a method that keeps neighbourhoods keeps the highlighted cells together.`);
  const hov = stats(f1g.root);
  const row1 = el('div', { class: 'fig-row three' }); f1g.body.append(row1);
  const pan1 = [['PCA (first two components)', pcaE], ['t-SNE (perplexity 30)', tsne30], ['UMAP (n_neighbors 15, min_dist 0.5)', umap15]].map(([t, e]) => {
    const w = el('div'); row1.append(w);
    return { e, P: mapPanel(w, { width: 300, title: t, r: 2.6, hover: false }).draw(e.xy) };
  });
  legend(f1g.body, typeLegend());
  const setHov = (i) => {
    pan1.forEach(({ P }) => P.sel.attr('fill-opacity', (j) => (i == null ? 0.85 : j === i || S.nn[i].includes(j) ? 1 : 0.12)).attr('r', (j) => (i != null && (j === i || S.nn[i].includes(j)) ? 4.2 : 2.6)).attr('stroke', (j) => (i != null && j === i ? ink() : css('--surface'))).attr('stroke-width', (j) => (i != null && j === i ? 1.6 : 0.6)));
    if (i == null) { hov.set({ 'neighbours kept (hover a cell)': '–' }); return; }
    const o = {}; pan1.forEach(({ e }, k) => { o[['PCA', 't-SNE', 'UMAP'][k] + ': neighbours kept'] = S.nn[i].filter((j) => nnOf(e)[i].includes(j)).length + ' / ' + K; });
    hov.set(o);
  };
  pan1.forEach(({ P }) => P.sel.on('mousemove', (e, i) => { setHov(i); tip(e, tipFor(i)); }).on('mouseleave', () => { setHov(null); untip(); }));
  setHov(null);
  prose(ch, `<p>On average, ${Math.round(100 * pcaE.knn)}% of a cell's ${K} nearest neighbours remain among its ${K} nearest in the PCA plane, against ${Math.round(100 * tsne30.knn)}% for t-SNE and ${Math.round(100 * umap15.knn)}% for UMAP. Even the better maps keep only about ${Math.round(100 * Math.max(tsne30.knn, umap15.knn))}%, because two dimensions cannot hold every neighbourhood relation of ${DIM}-D data.</p><p>Key terms:</p>`);
  table(ch, [
    { t: 'Embedding / map', m: 'The 2-D coordinates assigned to each cell. The axes have no units and no meaning.', s: `${n} cells × 2 coordinates` },
    { t: 'Nearest neighbours', m: 'The <i>k</i> cells closest to a given cell, measured in the original space', s: `<i>k</i> = ${K} here (step 1)` },
    { t: 'Perplexity (t-SNE)', m: 'The effective number of neighbours each cell attends to; the main t-SNE setting (typically 5–50)', s: 'step 2' },
    { t: 'n_neighbors (UMAP)', m: 'The number of neighbours used to build the graph; plays the role of perplexity', s: 'step 4; default 15 in umap-learn and scanpy, 30 in Seurat' },
    { t: 'min_dist (UMAP)', m: 'How tightly UMAP is allowed to pack points together in the map', s: 'step 4; default 0.5 in scanpy, 0.1 in umap-learn, 0.3 in Seurat' },
    { t: 'Early exaggeration (t-SNE)', m: 'An initial phase in which attractions are multiplied so that clusters form before fine adjustment', s: 'step 3' },
    { t: 'KL divergence (t-SNE)', m: 'The quantity t-SNE minimises: how badly map similarities match the original similarities', s: 'step 3' },
    { t: 'Initialisation and seed', m: 'The starting layout (random, PCA or spectral) and its random number seed; the optimisation is not convex, so these change the result', s: 'step 6' },
  ], [['t', 'Term'], ['m', 'Meaning'], ['s', 'On this page']]);

  // =====================================================================================
  // 2. WHY
  // =====================================================================================
  ch = chapter(main, 'why', 'Why do we need them?', 'A table of thousands of columns cannot be inspected directly, and a linear projection often hides the structure.');
  prose(ch, `<p>Single-cell and other high-dimensional datasets raise the same practical need: to <b>see</b> the data before and after analysis. A map is used to:</p><ol>
    <li><b>Check that expected groups exist</b> (cell types, patient groups) and that unexpected groups do not.</li>
    <li><b>Detect technical structure</b>, such as batch, sequencing depth or dying cells, by colouring the map with metadata.</li>
    <li><b>Communicate results</b>: marker expression or annotations drawn on a map are the standard figure of a single-cell paper.</li>
    <li><b>Inspect continuous processes</b> such as differentiation or T-cell exhaustion, which clustering cannot describe.</li></ol>
    <p>PCA is the first thing to try and is kept as the input in every pipeline, but its first two components keep large-scale variance and mix neighbourhoods. The chart measures this with the neighbour-preservation score defined above.</p>`);
  const f2g = figure(ch, `How many of a cell's ${K} nearest neighbours survive the projection`, `For every cell, the share of its ${K} nearest neighbours in the ${DIM}-D data that are still among its ${K} nearest in the 2-D map, averaged over ${n} cells. The dashed line is what random placement would give.`);
  const rand = K / (n - 1);
  const bars = [['PCA, first two components', pcaE.knn, cat(0)], ['t-SNE, perplexity 30', tsne30.knn, cat(6)], ['UMAP, n_neighbors 15', umap15.knn, cat(6)]];
  const F2 = frame(f2g.body, { width: 820, height: 214, x: [0, 1], y: [0, 3], grid: false, axes: false, margin: { left: 190, top: 22, bottom: 40, right: 20 } });
  F2.g.append('g').attr('class', 'axis x').attr('transform', `translate(0,${F2.h})`).call(d3.axisBottom(F2.x).ticks(6).tickFormat(d3.format('.0%')));
  F2.g.append('text').attr('class', 'ax-label').attr('x', F2.w / 2).attr('y', F2.h + 36).attr('text-anchor', 'middle').text('share of the 15 nearest neighbours preserved');
  bars.forEach(([t, v, c], i) => {
    F2.layer('b').append('rect').attr('x', 0).attr('y', F2.y(3 - i) + 8).attr('width', F2.x(v)).attr('height', F2.y(0) / 3 - 16).attr('fill', c).attr('fill-opacity', 0.9)
      .on('mousemove', (e) => tip(e, `${t}<br>${f1(100 * v)}% preserved`)).on('mouseleave', untip);
    F2.layer('b').append('text').attr('x', -8).attr('y', F2.y(3 - i) + F2.y(0) / 6 + 4).attr('text-anchor', 'end').attr('font-size', 12.5).attr('fill', ink2()).text(t);
    F2.layer('b').append('text').attr('x', F2.x(v) + 6).attr('y', F2.y(3 - i) + F2.y(0) / 6 + 4).attr('font-size', 12.5).attr('fill', ink()).text(Math.round(100 * v) + '%');
  });
  F2.layer('b').append('line').attr('x1', F2.x(rand)).attr('x2', F2.x(rand)).attr('y1', 0).attr('y2', F2.h).attr('stroke', muted()).attr('stroke-dasharray', '4 4');
  F2.layer('b').append('text').attr('x', F2.x(rand) + 6).attr('y', -8).attr('font-size', 11.5).attr('fill', muted()).text(`random placement: ${f1(100 * rand)}%`);
  prose(ch, `<p>The neighbourhood methods retain roughly ${(Math.max(tsne30.knn, umap15.knn) / pcaE.knn).toFixed(1)} times as many neighbours as the PCA plane. PCA nevertheless separates the cell types by k-means about as well (adjusted Rand index ${f2(pcaE.ari)} against ${f2(S.ari_orig)} in the full ${DIM}-D data), because the planted types are far apart in the first two components; neighbourhood scores and cluster separation measure different things. The table summarises the usual choices.</p>`);
  table(ch, [
    { a: 'PCA', l: 'preserves large variance and global distances; linear', g: 'fast, deterministic, axes have meaning', w: 'neighbourhoods and curved structure are lost' },
    { a: '<b>t-SNE</b>', l: '<b>local neighbourhoods</b>; equalises local density', g: 'sharp, well-separated groups', w: 'slow for large n; sizes and gaps not interpretable; no mapping for new cells' },
    { a: '<b>UMAP</b>', l: '<b>local neighbourhoods</b>; a graph-based optimisation', g: 'fast; often a smoother layout; can map new cells', w: 'sizes and gaps equally unreliable; sensitive to its settings' },
    { a: 'Heatmap + hierarchical clustering', l: 'all features, no projection', g: 'shows the actual values', w: 'does not scale to thousands of cells' },
  ], [['a', 'Method'], ['l', 'What it keeps'], ['g', 'Strengths'], ['w', 'Limits']]);

  // =====================================================================================
  // 3. WHERE
  // =====================================================================================
  ch = chapter(main, 'where', 'Where are they used?', 'Wherever many samples have many measurements and a visual overview is needed.');
  cards(ch, [
    { tag: 'Single-cell RNA-seq', title: 'The standard cell atlas figure', body: 'scanpy (<code>sc.tl.umap</code>, <code>sc.tl.tsne</code>) and Seurat (<code>RunUMAP</code>, <code>RunTSNE</code>) draw cell maps from the top principal components. Cluster labels and marker expression are overlaid.' },
    { tag: 'Cytometry', title: 'Flow and mass cytometry', body: 'viSNE, an early application of t-SNE, reduced 30–40 marker panels to a map of immune populations. UMAP is now equally common in FlowJo and CyTOF workflows.' },
    { tag: 'Quality control', title: 'Batch, depth and doublets', body: 'Colouring the map by batch, donor, library size or mitochondrial fraction shows whether technical factors, rather than biology, organise the data. This is the most reliable use of a map.' },
    { tag: 'Trajectories', title: 'Differentiation and exhaustion', body: 'Maps of T cells or progenitors show continuous gradients. Trajectory methods (PAGA, diffusion pseudotime) are computed on the neighbour graph, and the map is used to display them.' },
    { tag: 'Spatial omics', title: 'Spots and cells in expression space', body: 'Visium spots or segmented cells are embedded by expression and coloured by tissue region, to compare expression-space structure with the physical layout.' },
    { tag: 'Foundation models', title: 'Learned cell embeddings', body: 'Embeddings from models such as scGPT or Geneformer are normally inspected with a UMAP: it shows whether the model separates cell types and removes batch.' },
    { tag: 'Patients and bulk data', title: 'Cohort overviews', body: 'Bulk RNA-seq or methylation profiles of hundreds of patients can be mapped to reveal subtypes, outliers or centre effects. With few samples, PCA is usually clearer.' },
    { tag: 'Sequences and structures', title: 'Protein and antibody embeddings', body: 'Embeddings from protein language models, TCR and antibody repertoires and metagenomic bins are routinely visualised with UMAP.' },
    { tag: 'Image-based screens', title: 'Morphological profiling', body: 'Cell Painting feature vectors for thousands of compounds are mapped to find treatments that produce similar cell states.' },
  ]);
  prose(ch, `<p><b>When not to use them:</b> to measure distances, sizes or proportions (use the original space); as the input to clustering or to a statistical test (use the principal components or the neighbour graph); to define cell types on their own; with very few samples (tens of points, where a scatter plot of PCs is simpler and keeps real distances); or when the reader needs axes that mean something (use PCA, or plot the features themselves).</p>`);

  // =====================================================================================
  // 4. HOW
  // =====================================================================================
  ch = chapter(main, 'how', 'How it works', 'Six steps: neighbours, similarities, placement of points, the UMAP variant, what survives, and a lab to tune a map.');

  // ---- Step 1: neighbours
  const medoid = (t) => { const idx = d3.range(n).filter((i) => lab[i] === t), c = d3.range(DIM).map((d) => mean(idx.map((i) => X[i][d]))); return d3.least(idx, (i) => Math.hypot(...c.map((v, d) => X[i][d] - v))); };
  const REP = d3.range(nT).map(medoid);
  const sortedDist = (i) => ORD[i].map((j) => ({ j, d: dist(i, j) }));
  let rb = rung(ch, 1, 'Start from neighbours in the original space',
    `<p>Each cell is a point in ${DIM}-dimensional space, and the distance between two cells is the ordinary straight-line (Euclidean) distance. Both methods start by finding, for each cell, which cells lie close to it. The histogram shows the distances from one cell to all others.</p>`);
  const fr1 = figure(rb, 'Distances from one cell to every other cell', `A representative cell of the chosen type (the one closest to its type's centroid). Bars: how many other cells lie at each distance. The dashed line marks the distance to its ${K}th nearest neighbour.`);
  let q1 = 3;
  select(fr1.controls, { label: 'Cell type', value: String(q1), options: TY.map((t, i) => [String(i), t]), onChange: (v) => { q1 = +v; paint1(); } });
  const s1 = stats(fr1.root);
  const F1 = frame(fr1.body, { width: 820, height: 270, x: [0, 70], y: [0, 1], xLabel: `distance to the chosen cell (${DIM}-D)`, yLabel: 'number of cells', grid: false });
  const histL = F1.layer('h'), yAx1 = F1.g.select('.axis.y');
  const lgBox1 = el('div'); fr1.body.append(lgBox1);
  function paint1() {
    const i = REP[q1], sd = sortedDist(i), bin = d3.bin().domain([0, 70]).thresholds(d3.range(1, 70, 1.5));
    const same = bin(sd.filter((o) => lab[o.j] === q1).map((o) => o.d)), oth = bin(sd.filter((o) => lab[o.j] !== q1).map((o) => o.d));
    const ymax = d3.max([...same, ...oth], (b) => b.length) * 1.1;
    F1.y.domain([0, ymax]); yAx1.call(d3.axisLeft(F1.y).ticks(5));
    histL.selectAll('*').remove();
    lgBox1.innerHTML = ''; legend(lgBox1, [[`${TY[q1]} (same type)`, cat(q1)], ['other cell types', muted()], ['15th nearest neighbour (dashed)', ink()]]);
    [[oth, muted(), 'other cell types'], [same, cat(q1), 'same cell type']].forEach(([bins, c, nm]) => histL.append('g').selectAll('rect').data(bins).join('rect').attr('x', (b) => F1.x(b.x0) + 0.5).attr('width', (b) => Math.max(0, F1.x(b.x1) - F1.x(b.x0) - 1)).attr('y', (b) => F1.y(b.length)).attr('height', (b) => F1.h - F1.y(b.length)).attr('fill', c).attr('fill-opacity', 0.8)
      .on('mousemove', (e, b) => tip(e, `${nm}<br>distance ${f1(b.x0)}–${f1(b.x1)}: ${b.length} cells`)).on('mouseleave', untip));
    const dK = sd[K - 1].d;
    histL.append('line').attr('x1', F1.x(dK)).attr('x2', F1.x(dK)).attr('y1', 0).attr('y2', F1.h).attr('stroke', ink()).attr('stroke-dasharray', '5 4').attr('stroke-width', 1.6);
    histL.append('text').attr('x', F1.x(dK) + 6).attr('y', 14).attr('font-size', 11.5).attr('fill', ink()).text(`15th neighbour: ${f1(dK)}`);
    const nbSame = sd.slice(0, K).filter((o) => lab[o.j] === q1).length;
    s1.set({ 'cell type': TY[q1], 'distance to nearest neighbour': f1(sd[0].d), [`distance to ${K}th neighbour`]: f1(dK), [`of ${K} nearest, same type`]: `${nbSame}/${K}`, 'median distance to all cells': f1(d3.median(sd, (o) => o.d)) });
  }
  paint1();
  const medD = (t) => d3.median(sortedDist(REP[t]).slice(0, K), (o) => o.d);
  prose(rb, `<p>The ${K} nearest neighbours of a representative Treg cell are at distance ${f1(medD(3))} (median), against ${f1(medD(0))} for a Tumour cell, because Tumour cells are spread more widely. Most near neighbours are of the same type, and the distance rises steeply for other types. <b>Nearby cells are informative; far-away cells are not</b>: their distances reflect which other types lie in between, and they depend on how the features were scaled and on the batch. This is why both methods rely only on local distances.</p>`);
  const e0 = REP[3], nbRows = S.nn[e0].slice(0, 5).map((j, k) => ({ k: k + 1, t: TY[lab[j]], d: f1(dist(e0, j)), b: S.batch[j] + 1 }));
  worked(rb, `The five nearest neighbours of one Treg cell (cell ${e0})`,
    `<p>${nbRows.map((r) => `#${r.k}: ${r.t}, distance ${r.d}, batch ${r.b}`).join('; ')}. ${nbRows.filter((r) => r.t === 'Treg').length} of the 5 are Treg. Treg and CD4 T are placed close together in the simulation (centroids ${f1(G0.pairs.find((p) => p.a === 2 && p.b === 3).d)} apart, against ${f1(G0.pairs.find((p) => p.a === 0 && p.b === 7).d)} for Tumour and Endothelial), so CD4 T cells would also be acceptable neighbours of a Treg cell.</p>`);

  // ---- Step 2: perplexity
  function rowP(i, perp, m = n - 1) { // Gaussian similarities of cell i to its m nearest cells, with sigma set by bisection so the entropy matches log(perplexity)
    const row = new Float64Array(n), nb = ORD[i].slice(0, m), target = Math.log(perp), mn = D2[i * n + nb[0]];
    let beta = 1, lo = -Infinity, hi = Infinity, s = 1;
    for (let it = 0; it < 50; it++) {
      s = 0; let sdp = 0;
      for (const j of nb) { const d = D2[i * n + j] - mn, p = Math.exp(-d * beta); row[j] = p; s += p; sdp += d * p; }
      const H = Math.log(s) + beta * sdp / s;
      if (Math.abs(H - target) < 1e-4) break;
      if (H > target) { lo = beta; beta = hi === Infinity ? beta * 2 : (beta + hi) / 2; } else { hi = beta; beta = lo === -Infinity ? beta / 2 : (beta + lo) / 2; }
    }
    for (const j of nb) row[j] /= s;
    return { row, sigma: 1 / Math.sqrt(2 * beta) };
  }
  rb = rung(ch, 2, 'Turn distances into neighbour weights: perplexity',
    `<p>t-SNE converts distances into <b>similarities</b>. Around each cell a Gaussian bell curve gives nearby cells a high weight and distant cells a weight close to zero. The width of the bell (σ) is chosen separately for every cell so that the <b>effective number of neighbours equals the perplexity</b>. A cell in a dense region therefore gets a narrow bell and a cell in a sparse region a wide one.</p>`);
  const fr2 = figure(rb, 'Weight carried by the nearest neighbours of four cells', 'Each curve is one representative cell of the type shown (the cell closest to its type\'s centroid). Cumulative share of the cell\'s total weight, by neighbour rank (left) and by distance in the original space (right). Large circles mark where half of the weight has been reached.');
  const T2 = [0, 2, 3, 7];
  let perp2 = 30;
  slider(fr2.controls, { label: 'perplexity', min: 2, max: 100, step: 1, value: perp2, onInput: (v) => { perp2 = v; paint2(); } });
  const s2 = stats(fr2.root);
  const row2 = el('div', { class: 'fig-row' }); fr2.body.append(row2);
  const ca = el('div'), cb = el('div'); row2.append(ca, cb);
  const FA = frame(ca, { width: 460, height: 300, x: [0, 150], y: [0, 1], xLabel: 'neighbour rank (1 = closest)', yLabel: 'cumulative share of weight' });
  const FB = frame(cb, { width: 460, height: 300, x: [0, 45], y: [0, 1], xLabel: 'distance in the original space', yLabel: 'cumulative share of weight' });
  legend(fr2.body, T2.map((t) => [TY[t], cat(t)]));
  const order2 = T2.map((t) => sortedDist(REP[t]).map((o) => o.j));
  function paint2() {
    const out = {};
    FA.layer('c').selectAll('*').remove(); FB.layer('c').selectAll('*').remove();
    T2.forEach((t, k) => {
      const { row, sigma } = rowP(REP[t], perp2), ord = order2[k];
      let c = 0; const cum = ord.map((j) => (c += row[j])), r50 = cum.findIndex((v) => v >= 0.5) + 1;
      const pa = [[0, 0], ...cum.slice(0, 150).map((v, r) => [r + 1, v])], pb = [[0, 0], ...ord.map((j, r) => [dist(REP[t], j), cum[r]])];
      FA.layer('c').append('path').attr('d', d3.line().x((q) => FA.x(q[0])).y((q) => FA.y(q[1]))(pa)).attr('fill', 'none').attr('stroke', cat(t)).attr('stroke-width', 2.2);
      FB.layer('c').append('path').attr('d', d3.line().x((q) => FB.x(Math.min(45, q[0]))).y((q) => FB.y(q[1]))(pb)).attr('fill', 'none').attr('stroke', cat(t)).attr('stroke-width', 2.2);
      FA.layer('c').append('circle').attr('cx', FA.x(Math.min(150, r50))).attr('cy', FA.y(cum[r50 - 1])).attr('r', 5).attr('fill', cat(t)).attr('stroke', css('--surface')).attr('stroke-width', 1.5).on('mousemove', (e) => tip(e, `${TY[t]}: half the weight is on the ${r50} nearest cells`)).on('mouseleave', untip);
      FB.layer('c').append('circle').attr('cx', FB.x(Math.min(45, dist(REP[t], ord[r50 - 1])))).attr('cy', FB.y(cum[r50 - 1])).attr('r', 5).attr('fill', cat(t)).attr('stroke', css('--surface')).attr('stroke-width', 1.5).on('mousemove', (e) => tip(e, `${TY[t]}: half the weight lies within distance ${f1(dist(REP[t], ord[r50 - 1]))}`)).on('mouseleave', untip);
      out[`σ, ${TY[t]}`] = f1(sigma);
    });
    out['perplexity'] = perp2;
    s2.set(out);
  }
  paint2();
  const sg = (t) => rowP(REP[t], 30).sigma;
  prose(rb, `<p>At perplexity 30, σ is ${f1(sg(0))} for the Tumour cell and ${f1(sg(3))} for the Treg cell, a ${(sg(0) / sg(3)).toFixed(1)}-fold difference, yet both cells attend to a comparable number of neighbours (left panel, similar curves) and differ in the distance covered (right panel). <b>Every cell gets the same effective neighbourhood size, whatever the density</b>. This is the source of the first limitation: in the map, loose groups are not drawn larger than tight ones. The weights are then symmetrised (averaged between cell <i>i</i> and cell <i>j</i>) and form the target similarity matrix <i>P</i>. UMAP's analogue, fixed by <code>n_neighbors</code>, appears in step 4. The Endothelial cell has σ = ${f1(sg(7))} because its type contains only ${countT[7]} cells: matching a perplexity of 30 forces the bell to reach into other cell types.</p>`);

  // ---- Step 3: placing points (live t-SNE)
  rb = rung(ch, 3, 'Place the points: attraction, repulsion and the heavy-tailed kernel',
    `<p>The 2-D map starts as a tiny random cloud. Each cell is then moved by forces: <b>neighbours pull each other together</b> (in proportion to their original similarity) and <b>all pairs push each other apart</b>. The map uses a Student-<i>t</i> kernel, whose long tails let moderately dissimilar cells sit far apart without a strong penalty, which is what separates clusters. The quantity minimised is the <b>KL divergence</b> between the original similarities and the map similarities. The first 150 iterations use <b>early exaggeration</b> (attractions ×12) so that clusters form first.</p>`);
  const fr3 = figure(rb, `Exact t-SNE, iteration by iteration, on the ${n} cells`, 'Left: the map during optimisation; the layout starts as a tiny blob and expands. Right: KL divergence against the original similarities (during exaggeration the optimised objective differs, so the curve is only meaningful afterwards). The sklearn result at the same perplexity is shown in step 6.');
  let perp3 = 30, seed3 = 1, playing = true, ts = null, P3 = null, klHist = [];
  slider(fr3.controls, { label: 'perplexity', min: 5, max: 100, step: 5, value: perp3, onInput: (v) => { perp3 = v; restart3(true); } });
  const pl = button(fr3.controls, 'Pause', () => { playing = !playing; pl.textContent = playing ? 'Pause' : 'Play'; }, true);
  button(fr3.controls, 'Restart', () => restart3(false));
  const s3 = stats(fr3.root);
  const row3 = el('div', { class: 'fig-row' }); fr3.body.append(row3);
  const ma = el('div'), mb = el('div'); row3.append(ma, mb);
  const M3 = mapPanel(ma, { width: 460, title: 't-SNE (live)', r: 3.2 });
  const KF = frame(mb, { width: 460, height: 440, x: [0, 750], y: [0, 3], xLabel: 'iteration', yLabel: 'KL divergence' });
  KF.layer('ex').append('rect').attr('x', 0).attr('y', 0).attr('width', KF.x(150)).attr('height', KF.h).attr('fill', css('--surface-2'));
  KF.layer('ex').append('text').attr('x', 6).attr('y', 16).attr('font-size', 11.5).attr('fill', ink2()).text('early exaggeration');
  legend(fr3.body, typeLegend());
  const tPath = KF.layer('kl').append('path').attr('fill', 'none').attr('stroke', cat(6)).attr('stroke-width', 2.2);
  const MAXIT = 750;
  function affin(perp) {
    const P = new Float64Array(n * n);
    for (let i = 0; i < n; i++) { const { row } = rowP(i, perp, Math.min(n - 1, 3 * perp + 1)); for (let j = 0; j < n; j++) P[i * n + j] = row[j]; }
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) { const v = Math.max((P[i * n + j] + P[j * n + i]) / (2 * n), 1e-12); P[i * n + j] = P[j * n + i] = v; }
    return P;
  }
  function makeTsne(P, seed) {
    const r = rng(seed), Y = new Float64Array(2 * n), iY = new Float64Array(2 * n), gains = new Float64Array(2 * n).fill(1), num = new Float64Array(n * n), grad = new Float64Array(2 * n);
    for (let i = 0; i < 2 * n; i++) Y[i] = r.normal(0, 1e-4);
    let iter = 0;
    const step = () => {
      const exag = iter < 150 ? 12 : 1, mom = iter < 250 ? 0.5 : 0.8; let Z = 0;
      for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) { const dx = Y[2 * i] - Y[2 * j], dy = Y[2 * i + 1] - Y[2 * j + 1], q = 1 / (1 + dx * dx + dy * dy); num[i * n + j] = q; Z += 2 * q; }
      grad.fill(0);
      for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
        const q = num[i * n + j], m = (exag * P[i * n + j] - q / Z) * q, dx = Y[2 * i] - Y[2 * j], dy = Y[2 * i + 1] - Y[2 * j + 1];
        grad[2 * i] += 4 * m * dx; grad[2 * i + 1] += 4 * m * dy; grad[2 * j] -= 4 * m * dx; grad[2 * j + 1] -= 4 * m * dy;
      }
      for (let k = 0; k < 2 * n; k++) { gains[k] = Math.sign(grad[k]) === Math.sign(iY[k]) ? Math.max(0.01, gains[k] * 0.8) : gains[k] + 0.2; iY[k] = mom * iY[k] - 50 * gains[k] * grad[k]; Y[k] += iY[k]; }
      let mx = 0, my = 0; for (let i = 0; i < n; i++) { mx += Y[2 * i]; my += Y[2 * i + 1]; }
      for (let i = 0; i < n; i++) { Y[2 * i] -= mx / n; Y[2 * i + 1] -= my / n; }
      iter++; return Z;
    };
    const kl = () => {
      let Z = 0; for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) Z += 2 / (1 + (Y[2 * i] - Y[2 * j]) ** 2 + (Y[2 * i + 1] - Y[2 * j + 1]) ** 2);
      let s = 0; for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) { const q = 1 / (1 + (Y[2 * i] - Y[2 * j]) ** 2 + (Y[2 * i + 1] - Y[2 * j + 1]) ** 2) / Z; s += 2 * P[i * n + j] * Math.log(P[i * n + j] / Math.max(q, 1e-300)); }
      return s;
    };
    return { Y, step, kl, get iter() { return iter; } };
  }
  const fitView = (Y) => { // rescale the live map to the fixed ±50 frame
    let m = 0; for (let k = 0; k < 2 * n; k++) m = Math.max(m, Math.abs(Y[k]));
    return Array.from(Y, (v) => (v / Math.max(m, 1e-9)) * 50);
  };
  function restart3(newP) {
    if (newP || !P3) P3 = affin(perp3);
    ts = makeTsne(P3, seed3); klHist = []; playing = true; pl.textContent = 'Pause'; paint3();
  }
  function paint3() {
    M3.draw(fitView(ts.Y), colorFor('type'));
    tPath.attr('d', klHist.length ? d3.line().x((q) => KF.x(q[0])).y((q) => KF.y(Math.min(3, q[1])))(klHist) : null);
    s3.set({ iteration: ts.iter, phase: ts.iter < 150 ? 'early exaggeration' : ts.iter < MAXIT ? 'refining' : 'finished', 'KL divergence': klHist.length ? f2(klHist[klHist.length - 1][1]) : '…', perplexity: perp3 });
  }
  restart3(true);
  let klFinal = null;
  animate(() => {
    if (!playing || ts.iter >= MAXIT) return true;
    for (let k = 0; k < 5 && ts.iter < MAXIT; k++) { ts.step(); if (ts.iter % 25 === 0 && ts.iter > 150) klHist.push([ts.iter, ts.kl()]); }
    paint3();
    if (ts.iter >= MAXIT) { klFinal = klHist[klHist.length - 1][1]; pl.textContent = 'Play'; playing = false; }
    return true;
  });
  prose(rb, `<p>The same optimisation is used by every implementation (scikit-learn uses a Barnes–Hut approximation and openTSNE / FIt-SNE an FFT-based one to scale it). It is <b>not convex</b>: different starting layouts reach different local minima, so two runs on the same data produce different maps (step 6), and the position of a cluster on the page is arbitrary.</p>`);

  // ---- Step 4: UMAP
  rb = rung(ch, 4, 'UMAP: the same idea with a different graph and objective',
    `<p>UMAP follows the same two stages with different mathematics. It (1) builds a <b>weighted neighbour graph</b> from the <code>n_neighbors</code> closest cells of each cell, with each cell's own distance to its nearest neighbour subtracted, so that every cell is fully connected to at least one neighbour; (2) lays this graph out in 2-D by optimising a cross-entropy that attracts connected cells and repels random pairs, starting from a <b>spectral</b> (graph-Laplacian) layout rather than random noise. Two settings matter most:</p>
     <ul><li><b>n_neighbors</b>: small values emphasise very local structure; large values include more of the global arrangement.</li>
     <li><b>min_dist</b>: the minimum spacing between points in the map. A small value gives tight, dense clumps; a larger one spreads cells out. It changes the appearance, not the neighbour graph.</li></ul>`);
  const fr4 = figure(rb, 'UMAP: n_neighbors (columns) and min_dist (rows)', 'Eight precomputed UMAP runs (umap-learn, seed 1) of the same cells. Each title gives n_neighbors, the share of 15-nearest-neighbours preserved and the number of groups that DBSCAN finds in the map; the data contain 8 cell types, one of which (CD8 T) is a continuum.');
  let mode4 = 'type';
  select(fr4.controls, { label: 'Colour by', value: mode4, options: MODES, onChange: (v) => { mode4 = v; paint4(); } });
  const g4 = el('div', { class: 'fig-row four' }); fr4.body.append(g4);
  const grid4 = [];
  [0.05, 0.5].forEach((md) => {
    g4.append(el('div', { class: 'fig-caption' }, `Row: min_dist = ${md}; n_neighbors increases from left to right`));
    [5, 15, 50, 150].forEach((nb) => { const e = find('cells', 'umap', nb, md), w = el('div'); g4.append(w); grid4.push({ e, P: mapPanel(w, { width: 215, title: `n=${nb} · ${Math.round(100 * e.knn)}% · ${e.ndb} groups`, r: 1.7 }) }); });
  });
  const lgBox4 = el('div'); fr4.body.append(lgBox4);
  function paint4() { grid4.forEach(({ e, P }) => P.draw(e.xy, colorFor(mode4), opacityFor(mode4))); lgBox4.innerHTML = ''; legend(lgBox4, legendFor(mode4)); }
  paint4();
  const u5 = find('cells', 'umap', 5, 0.05), u150 = find('cells', 'umap', 150, 0.05), u15a = find('cells', 'umap', 15, 0.05), u15b = find('cells', 'umap', 15, 0.5);
  prose(rb, `<p>With n_neighbors = 5, DBSCAN finds ${u5.ndb} groups in the map, because small neighbourhoods fragment even the CD8 trajectory into pieces; with n_neighbors = 150 it finds ${u150.ndb}. The neighbour-preservation score changes only from ${Math.round(100 * u5.knn)}% to ${Math.round(100 * u150.knn)}%, so a very different-looking picture can retain about the same local information. Changing min_dist at n_neighbors = 15 (${Math.round(100 * u15a.knn)}% against ${Math.round(100 * u15b.knn)}%) hardly alters the neighbour score, while the clumps become more diffuse.</p>`);

  // ---- Step 5: what is preserved
  rb = rung(ch, 5, 'What the map preserves, and what it does not',
    `<p>A direct check is possible here because the geometry is known. For every cell type the figure compares its <b>spread</b> (mean distance of its cells from their centroid) and the <b>distance between every pair of types</b> in the original data against the same quantities in a map. Values are rescaled to a common unit (mean spread; median pair distance) so that a faithful map would lie on the diagonal.</p>`);
  const fr5 = figure(rb, 'Cluster sizes and between-cluster distances, original data against map', 'Left: spread of each cell type, original (grey) and in the map (purple); 1 = average spread. Right: each point is a pair of cell types; distance between their centroids, original against map; dashed line = perfect agreement.');
  const OPT5 = { pca: ['PCA', pcaE], tsne30: ['t-SNE, perplexity 30', tsne30], tsne5: ['t-SNE, perplexity 5', find('cells', 'tsne', 5)], umap15: ['UMAP, n_neighbors 15', umap15], umap150: ['UMAP, n_neighbors 150', find('cells', 'umap', 150, 0.5)] };
  let o5 = 'tsne30';
  select(fr5.controls, { label: 'Map', value: o5, options: Object.entries(OPT5).map(([k, v]) => [k, v[0]]), onChange: (v) => { o5 = v; paint5(); } });
  const s5 = stats(fr5.root);
  const row5 = el('div', { class: 'fig-row' }); fr5.body.append(row5);
  const la = el('div'), lb = el('div'); row5.append(la, lb);
  const FL = frame(la, { width: 460, height: 340, x: [0, 2.6], y: [0, nT], grid: false, axes: false, margin: { left: 86, top: 8, bottom: 44, right: 14 } });
  FL.g.append('g').attr('class', 'axis x').attr('transform', `translate(0,${FL.h})`).call(d3.axisBottom(FL.x).ticks(5));
  FL.g.append('text').attr('class', 'ax-label').attr('x', FL.w / 2).attr('y', FL.h + 36).attr('text-anchor', 'middle').text('spread relative to the average type');
  const FR = frame(lb, { width: 460, height: 340, x: [0, 3], y: [0, 3], xLabel: 'centroid distance, original data (relative)', yLabel: 'centroid distance, map (relative)' });
  FR.layer('d').append('line').attr('x1', FR.x(0)).attr('y1', FR.y(0)).attr('x2', FR.x(3)).attr('y2', FR.y(3)).attr('stroke', muted()).attr('stroke-dasharray', '4 4');
  FR.layer('d').append('text').attr('x', FR.x(2.95)).attr('y', FR.y(2.7)).attr('text-anchor', 'end').attr('font-size', 11.5).attr('fill', muted()).text('perfect agreement');
  const rhoT = FR.layer('d').append('text').attr('x', FR.x(0.08)).attr('y', FR.y(2.85)).attr('font-size', 12.5).attr('fill', ink());
  legend(fr5.body, [['original data', muted()], ['map', cat(6)]]);
  const rel0S = relS(G0), rel0D = relD(G0);
  function paint5() {
    const e = OPT5[o5][1], G = geoOf(e), rs = relS(G), rd = relD(G), r = spearman(G0.pairs.map((p) => p.d), G.pairs.map((p) => p.d));
    const yb = d3.scaleBand().domain(d3.range(nT)).range([0, FL.h]).padding(0.25);
    FL.layer('b').selectAll('*').remove();
    d3.range(nT).forEach((t) => {
      [[rel0S[t], muted(), 0, 'original'], [rs[t], cat(6), 1, 'map']].forEach(([v, c, k, nm]) => FL.layer('b').append('rect').attr('x', 0).attr('y', yb(t) + k * yb.bandwidth() / 2).attr('width', FL.x(Math.min(v, 2.6))).attr('height', yb.bandwidth() / 2 - 1).attr('fill', c)
        .on('mousemove', (ev) => tip(ev, `${TY[t]} (${nm})<br>relative spread ${f2(v)}`)).on('mouseleave', untip));
      FL.layer('b').append('text').attr('x', -6).attr('y', yb(t) + yb.bandwidth() / 2 + 4).attr('text-anchor', 'end').attr('font-size', 11.5).attr('fill', ink2()).text(TY[t]);
    });
    FR.layer('p').selectAll('circle').data(G0.pairs.map((p, k) => ({ ...p, o: rel0D[k], m: rd[k] }))).join('circle').attr('cx', (d) => FR.x(Math.min(3, d.o))).attr('cy', (d) => FR.y(Math.min(3, d.m))).attr('r', 5).attr('fill', cat(6)).attr('fill-opacity', 0.85).attr('stroke', css('--surface')).attr('stroke-width', 1.2)
      .on('mousemove', (ev, d) => tip(ev, `${TY[d.a]} – ${TY[d.b]}<br>original ${f2(d.o)} · map ${f2(d.m)}`)).on('mouseleave', untip);
    rhoT.text(`rank correlation of pair distances: ${f2(r)}`);
    const ratio = (t1, t2, v) => v[t1] / v[t2];
    s5.set({ 'Tumour ÷ Treg spread, original': f1(ratio(0, 3, rel0S)), 'Tumour ÷ Treg spread, map': f1(ratio(0, 3, rs)), 'rank correlation of distances': f2(r), 'neighbours preserved': Math.round(100 * e.knn) + '%' });
  }
  paint5();
  const Gt = geoOf(tsne30), Gu = geoOf(umap15), Gp = geoOf(pcaE), rt = relS(Gt), ru = relS(Gu), rp = relS(Gp);
  prose(rb, `<p>The Tumour cells are ${f1(rel0S[0] / rel0S[3])} times as spread out as the Treg cells in the original data. In the t-SNE map the ratio is ${f1(rt[0] / rt[3])}, and in the UMAP map ${f1(ru[0] / ru[3])} (PCA: ${f1(rp[0] / rp[3])}). <b>The difference in size is partly or largely erased</b>, as step 2 predicts. Between-cluster distances are retained only roughly: the rank correlation with the true centroid distances is ${f2(rho(tsne30))} for t-SNE, ${f2(rho(umap15))} for UMAP and ${f2(rho(pcaE))} for PCA. PCA keeps large distances best, as expected, but only ${Math.round(100 * pcaE.knn)}% of neighbourhoods (step 1). Whether UMAP or t-SNE keeps global arrangement better depends on the data and the initialisation (step 6); in this simulation the numbers above are what was obtained, and a single dataset does not settle the general question.</p>`);
  quiz(rb, 'In a map, group A looks twice as wide as group B. Does A contain more variable cells?', '<p>Not necessarily. Both methods rescale each neighbourhood to a similar effective size, so apparent width mostly reflects the number of cells and the settings, not the spread in the original space. Compute spread from the principal components instead.</p>');

  // ---- Step 6: lab
  rb = rung(ch, 6, 'Lab: tune a map and check it against the known answer',
    `<p>Every control here changes a precomputed run (scikit-learn t-SNE, umap-learn UMAP) of the ${n} cells, or of a <b>control dataset of random noise with no structure at all</b>. The checks come from the original space: neighbour preservation, k-means (k = ${nT}) against the true cell types, and DBSCAN on the map, which counts the groups a reader would be tempted to report. The right-hand map repeats the settings with another random seed.</p>`);
  const fl = figure(rb, 'Map laboratory', 'Left: chosen map. Right: the same settings with a different seed (for the noise data there is one run per setting, so the right panel shows PCA). Hover a cell for details.');
  const LAB = { ds: 'cells', m: 'tsne', pi: 2, md: 0.05, seed: 1, mode: 'type' };
  const PV = { tsne: [5, 15, 30, 60, 120], umap: [5, 15, 50, 150] };
  select(fl.controls, { label: 'Data', value: LAB.ds, options: [['cells', 'Cells (8 types)'], ['noise', 'Random noise (no structure)']], onChange: (v) => { LAB.ds = v; paintL(); } });
  select(fl.controls, { label: 'Method', value: LAB.m, options: [['tsne', 't-SNE'], ['umap', 'UMAP']], onChange: (v) => { LAB.m = v; LAB.pi = Math.min(LAB.pi, PV[v].length - 1); sP.set(LAB.pi); mdSel.parentElement.style.display = v === 'umap' ? '' : 'none'; paintL(); } });
  const sP = slider(fl.controls, { label: 'perplexity / n_neighbors', min: 0, max: 4, step: 1, value: LAB.pi, fmt: (i) => String(PV[LAB.m][Math.min(i, PV[LAB.m].length - 1)]), onInput: (i) => { LAB.pi = Math.min(i, PV[LAB.m].length - 1); paintL(); } });
  const mdSel = select(fl.controls, { label: 'min_dist (UMAP)', value: '0.05', options: ['0.05', '0.5'], onChange: (v) => { LAB.md = +v; paintL(); } });
  mdSel.parentElement.style.display = 'none';
  select(fl.controls, { label: 'Seed', value: '1', options: ['1', '2', '3'], onChange: (v) => { LAB.seed = +v; paintL(); } });
  select(fl.controls, { label: 'Colour by', value: LAB.mode, options: MODES, onChange: (v) => { LAB.mode = v; paintL(); } });
  const sL = stats(fl.root);
  const rowL = el('div', { class: 'fig-row' }); fl.body.append(rowL);
  const l1 = el('div'), l2 = el('div'); rowL.append(l1, l2);
  const PL1 = mapPanel(l1, { width: 460, title: '' }), PL2 = mapPanel(l2, { width: 460, title: '' });
  const lgL = el('div'); fl.body.append(lgL);
  const oneBy = { type: 'Colour = cell type', batch: 'Colour = batch', pt: 'Colour = CD8 position' };
  function paintL() {
    const ds = LAB.ds, md = LAB.m === 'umap' ? LAB.md : null, p = PV[LAB.m][LAB.pi], mode = ds === 'noise' ? 'none' : LAB.mode;
    const e = find(ds, LAB.m, p, md, ds === 'noise' ? 1 : LAB.seed), e2 = ds === 'noise' ? pcaN : find(ds, LAB.m, p, md, LAB.seed % 3 + 1);
    const col = ds === 'noise' ? () => cat(6) : colorFor(mode), op = ds === 'noise' ? () => 0.85 : opacityFor(mode);
    const sub = `${fmtLabel[LAB.m]}, ${LAB.m === 'tsne' ? 'perplexity' : 'n_neighbors'} ${p}${md != null ? `, min_dist ${md}` : ''}`;
    PL1.title(`${sub} · seed ${e.seed}`).draw(e.xy, col, op);
    PL2.title(ds === 'noise' ? 'PCA of the same noise data' : `same settings, seed ${e2.seed}`).draw(e2.xy, col, op);
    PL1.sel.on('mousemove', (ev, i) => tip(ev, tipFor(i, ds))); PL2.sel.on('mousemove', (ev, i) => tip(ev, tipFor(i, ds)));
    lgL.innerHTML = ''; if (ds === 'cells') lgL.append(legend(document.createElement('div'), legendFor(mode)));
    const o = { 'neighbours preserved': Math.round(100 * e.knn) + '%', 'groups found by DBSCAN on the map': `${e.ndb} (true: ${ds === 'noise' ? 0 : nT})` };
    if (ds === 'cells') { o['k-means on map vs true types (ARI)'] = f2(e.ari); o['k-means on the 20-D data (ARI)'] = f2(S.ari_orig); o['neighbour agreement, seed A vs B'] = Math.round(100 * agree(e, e2)) + '%'; }
    sL.set(o);
  }
  paintL();
  const a5 = find('cells', 'tsne', 5), nz = (m, p, md) => find('noise', m, p, md), s1d = find('cells', 'tsne', 30, null, 1), s2d = find('cells', 'tsne', 30, null, 2);
  table(rb, [
    { t: 'Cells, t-SNE: perplexity 5 → 30', s: `Groups found on the map ${a5.ndb} → <b>${tsne30.ndb}</b> (true: ${nT}); neighbours preserved ${Math.round(100 * a5.knn)}% → ${Math.round(100 * tsne30.knn)}%`, m: 'Very low perplexity splits real groups into many small islands. The perplexity should be set to a value that returns plausible, stable structure, and several values should be compared.' },
    { t: 'Noise, t-SNE: perplexity 5 → 60', s: `Groups found ${nz('tsne', 5).ndb} → <b>${nz('tsne', 60).ndb}</b>, in data with no groups`, m: 'Structure-free data can produce apparent clusters, especially at low perplexity. A map alone is not evidence that groups exist.' },
    { t: 'Cells, seed 1 → 2 (t-SNE, 30)', s: `Neighbour agreement between the two maps ${Math.round(100 * agree(s1d, s2d))}%; groups found ${s1d.ndb} and ${s2d.ndb}`, m: 'Local neighbourhoods are largely reproducible, but absolute positions, rotations and the arrangement of clusters are not.' },
    { t: 'Cells, UMAP: min_dist 0.05 → 0.5', s: `Neighbours preserved ${Math.round(100 * u15a.knn)}% → ${Math.round(100 * u15b.knn)}%; groups found ${u15a.ndb} → ${u15b.ndb}`, m: 'min_dist changes how points are packed, not which cells are neighbours; do not read density from the map.' },
    { t: 'Cells, UMAP: n_neighbors 5 → 150', s: `Groups found ${u5.ndb} → ${u150.ndb}; neighbours preserved ${Math.round(100 * u5.knn)}% → ${Math.round(100 * u150.knn)}%`, m: 'Small values fragment structure; large values merge it. Both are legitimate views of the same data.' },
    { t: 'Colour by batch', s: `A batch shift is planted (${S.batch.filter((b) => b).length} of ${n} cells in batch 2). In several cell types the two batches occupy different sides of the same group`, m: 'Colouring by batch is the safest use of the map: it can reveal a technical effect, although it cannot size it.' },
  ], [['t', 'Try'], ['s', 'What you see'], ['m', 'What it means']], { cls: 'compact' });
  quiz(rb, 'K-means on the map recovers the cell types about as well as k-means on the original data. Does this justify clustering on the map?', `<p>No. The planted types here are well separated, which is why the original-space score (${f2(S.ari_orig)}) is already high and the map scores a similar value. The map changes how many groups are visible (compare the DBSCAN counts above) and, in noise, creates groups where none exist. Cluster on the principal components or the neighbour graph, and use the map only to display the result.</p>`);

  // =====================================================================================
  // 5. PITFALLS
  // =====================================================================================
  ch = chapter(main, 'pitfalls', 'Pitfalls', 'Common problems, how to recognise them, and how to address them.');
  const ndbNoise5 = nz('tsne', 5).ndb;
  cards(ch, [
    { tag: 'Symptom: dense groups look large, loose groups look small', title: 'Cluster size is not interpretable', body: `Each neighbourhood is rescaled to a common size. Tumour cells are ${f1(rel0S[0] / rel0S[3])} times as spread out as Treg cells in the data, but ${f1(rt[0] / rt[3])} times in the t-SNE map. <b>Fix:</b> compute spread or density in the principal-component space.` },
    { tag: 'Symptom: groups placed far apart or close together', title: 'Distances between clusters are unreliable', body: `The rank correlation between true and mapped centroid distances is ${f2(rho(tsne30))} (t-SNE) and ${f2(rho(umap15))} (UMAP), and it changes with initialisation and settings (t-SNE with PCA initialisation: ${f2(rho(tsne30pca))}); PCA itself gives ${f2(rho(pcaE))}. Gap width and the order of clusters should not be interpreted. <b>Fix:</b> compare cluster centroids or marker genes in the original space.` },
    { tag: 'Symptom: clusters in data that has none', title: 'Structure appears in noise', body: `The noise control has no groups, yet DBSCAN finds ${ndbNoise5} at perplexity 5 and ${nz('tsne', 60).ndb} at perplexity 60. <b>Fix:</b> test the groups in the original space (marker genes, silhouette, stability under resampling) before reporting them.` },
    { tag: 'Symptom: one population splits into islands', title: 'Fragmentation and continuous data', body: `The CD8 T cells form a continuum, but small neighbourhoods tear it apart (UMAP n_neighbors 5: ${u5.ndb} groups; 150: ${u150.ndb}). Gaps in a map do not prove gaps in the data. <b>Fix:</b> use a larger perplexity or n_neighbors, colour by a trajectory score, and use methods built for continua (diffusion pseudotime, PAGA).` },
    { tag: 'Symptom: the figure changes on re-running', title: 'Seed and initialisation', body: `Two t-SNE runs that differ only by seed share ${Math.round(100 * agree(s1d, s2d))}% of cell neighbourhoods, and the layout differs. <b>Fix:</b> fix and report the seed, prefer PCA or spectral initialisation, and compare two or three seeds before describing arrangement.` },
    { tag: 'Symptom: clustering on the map', title: 'Clustering and statistics on a map', body: `The map is a picture, not a measurement: its coordinates are not independent features, distances are distorted, and the neighbourhood size is set by perplexity. <b>Fix:</b> cluster on principal components or the neighbour graph (Leiden, Louvain), then colour the map by cluster.` },
    { tag: 'Symptom: batches form separate islands', title: 'Technical effects look like biology', body: 'A batch effect appears as a split inside a cell type, or as batch-specific groups. <b>Fix:</b> colour by batch and donor before interpreting; integrate (Harmony, scVI) or correct, then recompute, and re-check on the new map.' },
    { tag: 'Symptom: "UMAP preserves global structure"', title: 'Overstated claims about the method', body: `The claim holds partly and depends on initialisation and data. In this simulation the rank correlation is ${f2(rho(umap15))} for UMAP and ${f2(rho(tsne30))} for t-SNE; with PCA initialisation t-SNE gives ${f2(rho(tsne30pca))}. <b>Fix:</b> do not choose a method on the basis of general claims; check that the structure of interest survives several settings.` },
    { tag: 'Symptom: new samples do not fit', title: 'Mapping new data', body: 'Standard t-SNE has no function that maps a new cell; UMAP can transform new points approximately, but not exactly, and without correcting for batch. <b>Fix:</b> embed all data together, or project new cells with a reference-mapping tool (e.g. scArches, Symphony, Seurat anchors).' },
  ]);

  // =====================================================================================
  // 6. RECAP
  // =====================================================================================
  ch = chapter(main, 'recap', 'Recap', 'Key points.');
  recap(ch, [
    't-SNE and UMAP place each cell near its <b>nearest neighbours</b> and ignore most other relationships. They are tools for looking at data, not for measuring it.',
    'Both start from distances in the original (principal-component) space, convert them into neighbour weights, and optimise a 2-D layout to match them.',
    '<b>Perplexity</b> (t-SNE) and <b>n_neighbors</b> (UMAP) set how many neighbours count; small values fragment structure, large values merge it. Try several.',
    `<b>Not preserved:</b> cluster sizes (${f1(rel0S[0] / rel0S[3])}× spread difference shrinks to ${f1(rt[0] / rt[3])}×), distances between clusters, density, and the axes. <b>Preserved, approximately:</b> local neighbourhoods (${Math.round(100 * tsne30.knn)}% of neighbours kept, against ${Math.round(100 * pcaE.knn)}% in PCA).`,
    'The result depends on the <b>seed, initialisation and settings</b>. Fix and report them, and check that conclusions hold across several runs.',
    'Random data can show clusters. Confirm any group in the original space before reporting it.',
    '<b>Never cluster, compute distances, or run tests on the 2-D coordinates.</b> Use the principal components or the neighbour graph, then display the result on the map.',
    'Most dependable uses: checking batch and quality, displaying annotations and marker expression, and communicating results.',
  ]);
  prose(ch, '<h3>In code</h3><p>The embedding is computed from the principal components and the neighbour graph; clustering uses the same graph, not the map.</p>');
  code(ch, {
    Python: `import scanpy as sc

# adata: normalised, log-transformed, highly variable genes selected
sc.pp.pca(adata, n_comps=30)
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)   # the graph that clustering and UMAP both use
sc.tl.leiden(adata, resolution=1.0)                  # cluster on the graph, not on the map
sc.tl.umap(adata, min_dist=0.5, random_state=0)      # initialised from the graph (spectral)
sc.tl.tsne(adata, n_pcs=30, perplexity=30, random_state=0)

sc.pl.umap(adata, color=["leiden", "batch", "total_counts"])   # colour by batch and depth first

# or on any numeric matrix X (cells x PCs)
from sklearn.manifold import TSNE
import umap
Y1 = TSNE(2, perplexity=30, init="pca", random_state=0).fit_transform(X)
Y2 = umap.UMAP(n_neighbors=15, min_dist=0.5, random_state=0).fit_transform(X)`,
    R: `library(Seurat)

obj <- RunPCA(obj, npcs = 30)
obj <- FindNeighbors(obj, dims = 1:30)                # graph for clustering
obj <- FindClusters(obj, resolution = 0.8)            # cluster on the graph, not on the map
obj <- RunUMAP(obj, dims = 1:30, n.neighbors = 30, min.dist = 0.3, seed.use = 42)
obj <- RunTSNE(obj, dims = 1:30, perplexity = 30, seed.use = 42)

DimPlot(obj, reduction = "umap", group.by = c("seurat_clusters", "batch"))

# any matrix: Rtsne::Rtsne(pcs, perplexity = 30, pca = FALSE); uwot::umap(pcs, n_neighbors = 15)`,
  });
  seeAlso(main, [['concepts', 'pca', 'Principal component analysis'], ['concepts', 'clustering', 'Clustering'], ['omics', 'sc-clustering', 'Clustering and cell-type annotation'], ['omics', 'bulk-qc-batch', 'QC, normalisation and batch effects'], ['concepts', 'transformers', 'Embeddings, attention and transformers']]);
}
