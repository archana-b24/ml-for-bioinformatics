// Clustering and cell-type annotation, lesson format (omics, single-cell).
// Arc: what → why → where → how it works (6 steps: QC + doublets → normalise + HVGs → PCA + kNN graph + batch →
//      Leiden resolution lab → marker-based annotation → label transfer) → pitfalls → recap.
// Every number in the prose is computed from data/omics/sc_*.json at render time (simulated tumour cells with known truth).
import { header, figure, slider, select, button, checkbox, stats, legend, seeAlso, every, tip, untip, el } from '../lib/ui.js';
import { frame, cat, css, ink, ink2, muted, sequential } from '../lib/plot.js';
import { stepCard, omicsData } from './_shared.js';
import { canvasScatter, clusterColor, ext, f2, pct, fmtInt } from './_sc_common.js';
import { lessonNav, chapter, prose, rung, worked, quiz, cards, table, code, recap } from '../lib/lesson.js';
const d3 = window.d3;

const sum = (a) => a.reduce((s, v) => s + v, 0);
const f1 = (v) => v.toFixed(1);
const rk = (r) => (Number.isInteger(r) ? r.toFixed(1) : String(r));
const EMB_NAME = { uncorrected: 'No batch correction', harmony: 'Harmony (chemistry as batch)', over: 'Over-corrected (patient as batch)' };
const MARK = [['EPCAM', 'Tumour'], ['CD8A', 'CD8 T'], ['TCF7', 'CD4 T'], ['FOXP3', 'Treg'], ['MS4A1', 'B'], ['CD68', 'Macrophage'], ['COL1A1', 'Fibroblast'], ['PECAM1', 'Endothelial']];
const STATE = ['MKI67', 'GZMB', 'PDCD1'];

export default async function render(main) {
  const names = ['sc_cells', 'sc_embedding', 'sc_qc', 'sc_annotation', 'sc_integration'];
  const [C, E, Q, A, I] = await Promise.all(names.map((f) => omicsData(main, f + '.json')));
  if (!C || !E || !Q || !A || !I) return;
  const CT = C.cell_types, NT = CT.length, n = C.n, T = C.type, B = C.batch, P = C.patient;
  const all = d3.range(n), sing = all.filter((i) => T[i] >= 0);
  const typeN = CT.map((_, t) => sing.filter((i) => T[i] === t).length);
  const nDbl = n - sing.length;
  const nClus = (emb, ri) => new Set(C.leiden[emb][rk(C.res[ri])]).size;
  const RI = (r) => C.res.indexOf(r);

  // ---- adjusted Rand index between a clustering and the true cell type (singlets only)
  const c2 = (x) => x * (x - 1) / 2;
  const ariOf = (L) => {
    const m = new Map(), ca = new Map(), cb = new Map();
    sing.forEach((i) => { const k = L[i] * 16 + T[i]; m.set(k, (m.get(k) || 0) + 1); ca.set(L[i], (ca.get(L[i]) || 0) + 1); cb.set(T[i], (cb.get(T[i]) || 0) + 1); });
    let s = 0, sa = 0, sb = 0; m.forEach((v) => { s += c2(v); }); ca.forEach((v) => { sa += c2(v); }); cb.forEach((v) => { sb += c2(v); });
    const ex = sa * sb / c2(sing.length);
    return (s - ex) / ((sa + sb) / 2 - ex);
  };
  const csCache = {};
  const cs = (emb, ri) => {
    const key = emb + ri; if (csCache[key]) return csCache[key];
    const L = C.leiden[emb][rk(C.res[ri])], K = d3.max(L) + 1;
    const cnt = Array.from({ length: K }, () => new Array(NT).fill(0));
    sing.forEach((i) => { cnt[L[i]][T[i]]++; });
    const plur = new Set(cnt.filter((r) => sum(r) > 0).map((r) => r.indexOf(d3.max(r))));
    const merged = CT.filter((_, t) => !plur.has(t));
    const split = CT.filter((_, t) => cnt.filter((r) => r[t] >= 0.1 * typeN[t]).length > 1);
    return (csCache[key] = { k: new Set(L).size, ari: ariOf(L), merged, split, cnt, L });
  };
  const ARI = {}; ['uncorrected', 'harmony', 'over'].forEach((e) => { ARI[e] = C.res.map((_, ri) => cs(e, ri).ari); });
  const R02 = RI(0.2), R08 = RI(0.8), R20 = RI(2.0), R001 = RI(0.01), R05 = RI(0.5);
  const cum20 = sum(E.var_ratio.slice(0, E.npc));
  const hvRank = d3.range(E.n_genes).sort((a, b) => E.z_disp[b] - E.z_disp[a]);
  const isStruct = E.cat.map((c) => c === 'marker' || c.startsWith('module'));
  const nStruct = sum(isStruct.map(Number));
  const knnStat = (emb) => { const K = C.knn[emb]; let st = 0, sb = 0, tt = 0; all.forEach((i) => K[i].forEach((j) => { sb += B[i] === B[j] ? 1 : 0; if (T[i] >= 0) { tt++; st += T[j] === T[i] ? 1 : 0; } })); return { type: st / tt, batch: sb / (n * K[0].length) }; };
  const knnU = knnStat('uncorrected'), knnH = knnStat('harmony');
  const fracV3 = d3.mean(B);

  const nearestSame = (X, Y, k = 10) => { // share of the k nearest neighbours (in the 2-D plane) that have the same cell type
    let same = 0, tot = 0;
    sing.filter((_, q) => q % 5 === 0).forEach((i) => {
      const best = [];
      for (let j = 0; j < n; j++) {
        if (j === i) continue;
        const d = (X[i] - X[j]) ** 2 + (Y[i] - Y[j]) ** 2;
        if (best.length < k || d < best[best.length - 1][0]) { best.push([d, j]); best.sort((a, b) => a[0] - b[0]); if (best.length > k) best.pop(); }
      }
      best.forEach(([, j]) => { tot++; if (T[j] === T[i]) same++; });
    });
    return same / tot;
  };
  const pcPur = nearestSame(C.pca[0], C.pca[1]), umPur = nearestSame(C.umap.harmony[0], C.umap.harmony[1]);

  header(main, 'Clustering and cell-type annotation',
    'Single-cell clustering groups cells that have similar expression, and annotation then <b>names the groups</b>. The standard pipeline is normalise → select variable genes → PCA → nearest-neighbour graph → community detection, followed by marker-gene or reference-based labelling.');
  lessonNav(main);
  stepCard(main, {
    stage: 'Single-cell · clustering and annotation',
    question: 'Which cell types are present in these tumours, and which cell belongs to which type?',
    data: `Simulated droplet counts: ${fmtInt(Q.n)} droplets from 12 patients, 2 chemistries, ${fmtInt(E.n_genes)} genes. The cell types are known, so every step can be scored. Figures show a ${fmtInt(n)}-cell subsample of the ${fmtInt(C.n_kept)} cells that pass QC.`,
    method: 'Adaptive QC and doublet scoring → log-normalisation → highly variable genes → PCA → batch correction (Harmony) → kNN graph → Leiden clustering → UMAP for display → marker genes or logistic-regression label transfer.',
    verdict: 'ml',
    why: 'Clustering and label transfer are unsupervised and supervised learning respectively. The number of clusters is a modelling choice that marker genes have to justify, and the simulation only shows how far the answer depends on that choice.',
  });

  // =====================================================================================
  // 1. WHAT
  // =====================================================================================
  let ch = chapter(main, 'what', 'What is cell clustering and annotation?', 'In one line: <b>similar cells → graph → communities → names</b>.');
  prose(ch, `<p>A single-cell RNA-seq experiment measures thousands of genes in thousands of cells, but not which cell type each cell is. <b>Clustering</b> groups cells whose expression profiles are similar; <b>annotation</b> then assigns each group (or each cell) a biological name such as "CD8 T cell". Clustering is unsupervised: no labels are used, and the groups are hypotheses. Annotation attaches prior biological knowledge to them, either as marker genes or as a labelled reference dataset.</p>
    <p>The diagram shows the stages in this study. Numbers are taken from the data on this page.</p>`);
  ch.append(pipelineSvg([
    { t: 'Counts', l: [`${fmtInt(Q.n)} droplets`, `${fmtInt(E.n_genes)} genes`, '12 patients'] },
    { t: '1 · QC', l: [`${fmtInt(C.n_kept)} cells kept`, `${Q.summary.qc_fail} removed`, 'doublets scored'] },
    { t: '2 · Normalise', l: ['library-size scale', 'log(1 + x)', `${fmtInt(E.n_hvg)} HVGs`] },
    { t: '3 · PCA, kNN', l: [`${E.npc} PCs`, `${pct(cum20)} of variance`, 'neighbour graph'] },
    { t: '4 · Leiden', l: [`${nClus('harmony', R02)} clusters at`, 'resolution 0.2', 'UMAP to draw'] },
    { t: '5 · Annotate', l: [`${NT} cell types`, 'markers or', 'reference'] },
  ]));
  prose(ch, '<p>Key terms:</p>');
  table(ch, [
    { t: 'Count matrix', m: 'Cells × genes table of UMI counts', s: `${fmtInt(Q.n)} × ${fmtInt(E.n_genes)} here` },
    { t: 'Highly variable genes (HVGs)', m: 'Genes whose variance exceeds that expected from their mean; used to define the space in which cells are compared', s: `${fmtInt(E.n_hvg)} of ${fmtInt(E.n_genes)} selected` },
    { t: 'Principal components (PCs)', m: 'Axes of largest variance; a compact, denoised summary of each cell', s: `${E.npc} PCs keep ${pct(cum20)} of variance` },
    { t: 'kNN graph', m: 'Each cell is linked to its <i>k</i> most similar cells in PC space', s: 'k = 15 for clustering' },
    { t: 'Leiden clustering', m: 'Community detection: groups of cells densely linked to each other and sparsely to the rest', s: `${nClus('harmony', R02)}–${nClus('harmony', R20)} clusters depending on resolution` },
    { t: 'Resolution', m: 'Parameter that sets how finely the graph is cut: higher gives more, smaller clusters', s: `${C.res[0]} to ${C.res[C.res.length - 1]}` },
    { t: 'UMAP', m: 'A two-dimensional drawing of the neighbour graph; for display, not for measurement', s: 'one point per cell' },
    { t: 'Annotation', m: 'Assigning cell-type names to clusters or cells', s: `${NT} types, known here` },
    { t: 'Batch effect', m: 'Technical differences between runs, chemistries or labs that shift expression', s: `${C.batches.join(' vs ')}` },
    { t: 'Doublet', m: 'A droplet that captured two cells, giving a hybrid profile', s: `${Q.summary.n_dbl} of ${fmtInt(Q.n)} droplets` },
  ], [['t', 'Term'], ['m', 'Meaning'], ['s', 'In this study']]);

  // =====================================================================================
  // 2. WHY
  // =====================================================================================
  ch = chapter(main, 'why', 'Why do we need it?', 'A cell has thousands of measurements; a plot has two axes.');
  prose(ch, `<p>The cell types in a tumour cannot be read from any single gene, and two coordinates cannot show a ${fmtInt(E.n_genes)}-gene profile. A usable method should:</p><ol>
    <li><b>Use all informative genes together</b>, so that cells are compared on their whole profile.</li>
    <li><b>Find groups without being told how many there are</b>, because new or rare populations are often the finding.</li>
    <li><b>Scale to hundreds of thousands of cells</b>, which rules out all-pairs comparisons and slow methods.</li></ol>
    <p>Looking at the first two principal components is the obvious first attempt. The figure compares that view with a UMAP of the batch-corrected neighbour graph.</p>`);
  const fw = figure(ch, 'Two coordinates versus the neighbour graph', 'Each point is one cell, coloured by its true cell type (known in this simulation). Left view: the two largest-variance directions. Right view: UMAP of the 20-PC neighbour graph after Harmony. Hover a cell for details.');
  let wView = 'pca';
  select(fw.controls, { label: 'View', value: wView, options: [['pca', 'PC1 vs PC2'], ['umap', 'UMAP (20 PCs, Harmony)']], onChange: (v) => { wView = v; paintW(); } });
  const sW = stats(fw.root);
  const SW = umapPlot(fw.body, { width: 820, height: 380, hover: (i) => cellTip(i) });
  legend(fw.body, CT.map((t, i) => [t, cat(i)]).concat([['doublet', muted()]]));
  function paintW() {
    const isP = wView === 'pca';
    SW.show(isP ? C.pca[0] : C.umap.harmony[0], isP ? C.pca[1] : C.umap.harmony[1], (i) => typeColor(T[i]));
    SW.labels(isP ? null : C.umap.harmony);
    SW.axes(isP ? 'PC 1' : 'UMAP 1', isP ? 'PC 2' : 'UMAP 2');
    sW.set({ 'variance in PC1 + PC2': pct(E.var_ratio[0] + E.var_ratio[1], 1), 'variance in 20 PCs': pct(cum20), 'nearest neighbours with the same type (this view)': pct(isP ? pcPur : umPur, 1), 'in 20-PC space': pct(knnH.type, 1) });
  }
  paintW();
  prose(ch, `<p>PC1 and PC2 hold ${pct(E.var_ratio[0] + E.var_ratio[1], 1)} of the variance. In that plane only ${pct(pcPur, 1)} of a cell's 10 nearest neighbours share its type, so most types overlap. The neighbour graph is built from ${E.npc} PCs, where ${pct(knnH.type, 1)} of neighbours share the type, and UMAP redraws that graph in two dimensions (${pct(umPur, 1)}).</p>`);
  table(ch, [
    { a: 'Look at one marker gene', g: 'no', k: 'no', s: 'yes', n: 'Needs the answer in advance; most genes are dropout-prone (zero in many cells)' },
    { a: 'k-means on PCs', g: 'yes', k: 'must be set', s: 'yes', n: 'Assumes round clusters of similar size; k must be guessed' },
    { a: 'Hierarchical clustering', g: 'yes', k: 'cut a tree', s: 'poor (O(n²) distances)', n: 'Memory and time grow with n²' },
    { a: '<b>kNN graph + Leiden</b>', g: '<b>yes</b>', k: '<b>set by resolution</b>', s: '<b>yes</b>', n: '<b>Standard in Seurat, Scanpy; arbitrary shapes</b>' },
  ], [['a', 'Approach'], ['g', 'Uses all genes?'], ['k', 'Number of groups'], ['s', 'Scales to 10⁵+ cells?'], ['n', 'Note']]);

  // =====================================================================================
  // 3. WHERE
  // =====================================================================================
  ch = chapter(main, 'where', 'Where is it used?', 'Any single-cell or spatial dataset in which the cell populations are not known beforehand.');
  cards(ch, [
    { tag: 'Tumour microenvironment', title: 'What is in this tumour?', body: 'The example on this page: tumour, T-cell subsets, B cells, macrophages, fibroblasts and endothelium, and how their proportions differ between responders and non-responders.' },
    { tag: 'Cell atlases', title: 'Human Cell Atlas, CELLxGENE', body: 'Millions of cells are clustered and annotated with a common vocabulary (Cell Ontology), then used as references for new data.' },
    { tag: 'Reference mapping', title: 'Label a new dataset', body: 'Azimuth, CellTypist, scArches and SingleR transfer labels from an annotated reference to new cells in minutes.' },
    { tag: 'Biomarker studies', title: 'Cell-type abundance and state', body: 'Per-patient cell-type fractions and cluster-specific expression are tested for association with response (see the pseudobulk page).' },
    { tag: 'Development and differentiation', title: 'Which states exist?', body: 'Clusters define the start, intermediate and end states on which trajectory or RNA-velocity analyses are built.' },
    { tag: 'Perturbation screens', title: 'Perturb-seq', body: 'Cells carrying each guide are clustered to see which perturbations shift cells into new states.' },
    { tag: 'Cytometry', title: 'Flow and mass cytometry', body: 'PhenoGraph (kNN graph plus community detection) was introduced for mass cytometry, and the same idea underlies Seurat and Scanpy clustering; FlowSOM is a related approach.' },
    { tag: 'Spatial transcriptomics', title: 'Spots and cells in tissue', body: 'The same pipeline groups spots or segmented cells; spatial methods add neighbouring tissue to the graph to define domains.' },
    { tag: 'Quality control', title: 'Spotting artefacts', body: 'Clusters with high mitochondrial reads, ambient RNA or two sets of markers identify low-quality cells and doublets.' },
  ]);
  prose(ch, '<p><b>When not to use it:</b> the cells lie on a continuum such as differentiation, where discrete clusters are an approximation (use trajectory methods or report the continuum); there are too few cells for a stable graph (a few hundred); the question is a difference between conditions, which needs replicate-aware tests on patient-level data rather than cell-level clusters; or the reference does not contain the cell types present, so labels must come from marker genes.</p>');

  // =====================================================================================
  // 4. HOW
  // =====================================================================================
  ch = chapter(main, 'how', 'How it works', 'Six steps, from droplet QC to labelled cell types, each checked against the known truth.');

  // ---- Step 1: QC and doublets
  let rb = rung(ch, 1, 'Quality control and doublets',
    `<p>Droplets that captured damaged cells, empty droplets and doublets must be removed before comparison. Cells are flagged by <b>genes detected</b> and <b>fraction of mitochondrial reads</b>, which is high when a cell has lost cytoplasmic RNA. Two ways to set the cut-offs:</p>
     <ul><li><b>Adaptive:</b> within each batch, flag cells more than 3 median absolute deviations (MADs) from the median.</li>
     <li><b>Fixed:</b> one cut-off for all cells, commonly "< 200 genes or > 10% mitochondrial reads".</li></ul>
     <p>Doublets are scored separately: simulated doublets are generated from random pairs of cells, and each real cell receives a score for how closely it resembles them (kNN score, as in Scrublet) or from a classifier trained to separate them (as in scDblFinder).</p>`);
  const f1g = figure(rb, 'QC thresholds and doublet scores', 'Left: each droplet by genes detected and mitochondrial fraction (capped at 35%). Right: doublet score, stacked by true status; bars use a square-root scale so that the doublets stay visible. Hover a droplet for details.');
  let qRule = 'adaptive', dScore = 'dbl_clf', dThr = Q.dbl_threshold;
  select(f1g.controls, { label: 'QC rule', value: qRule, options: [['adaptive', 'Adaptive (3 MADs per batch)'], ['fixed', 'Fixed (200 genes, 10% mito)']], onChange: (v) => { qRule = v; paintQ(); } });
  select(f1g.controls, { label: 'Doublet score', value: dScore, options: [['dbl_clf', 'Classifier (scDblFinder-like)'], ['dbl_knn', 'kNN (Scrublet-like)']], onChange: (v) => { dScore = v; dThr = d3.quantile(Q[v].slice().sort(d3.ascending), 1 - Q.dbl_call_n / Q.n); thrS.set(Math.round(dThr * 100) / 100); paintD(); } });
  const thrS = slider(f1g.controls, { label: 'doublet threshold', min: 0.05, max: 0.99, step: 0.01, value: dThr, fmt: f2, onInput: (v) => { dThr = v; paintD(); } });
  const sQ = stats(f1g.root);
  let qStat = {};
  const rowQ = el('div', { class: 'fig-row' }); f1g.body.append(rowQ);
  const qa = el('div'), qb = el('div'); rowQ.append(qa, qb);
  const lgen = Q.ngenes.map((v) => Math.log10(v));
  const SQ = canvasScatter(qa, { width: 500, height: 340, x: [1.9, 3.4], y: [0, 35], xLabel: 'genes detected (log10)', yLabel: 'mitochondrial reads (%)', hover: (i) => `${Q.cell_types[Q.type[i]] || 'doublet'} · ${Q.batches[Q.batch[i]]}<br>${Q.ngenes[i]} genes · ${f1(Q.pct_mt[i])}% mito<br>truth: ${['healthy', 'low quality', 'doublet'][Q.status[i]]}` });
  const thrL = SQ.F.layer('thr');
  const QCOL = [muted(), cat(1), cat(3), cat(4)];
  legend(qa, [['kept', QCOL[0]], ['removed, low quality (correct)', QCOL[1]], ['removed, healthy cell', QCOL[2]], ['low quality, kept (missed)', QCOL[3]]]);
  const SD = frame(qb, { width: 420, height: 340, x: [0, 1], y: [0, 1], xLabel: 'doublet score', yScale: d3.scaleSqrt(), margin: { left: 64 } });
  SD.g.append('text').attr('class', 'ax-label').attr('transform', 'rotate(-90)').attr('x', -SD.h / 2).attr('y', -52).attr('text-anchor', 'middle').text('cells (square-root scale)');
  const dLayer = SD.layer('bars'), dThrL = SD.layer('thr');
  legend(qb, [['single cell', cat(0)], ['true doublet', cat(2)]]);
  const qFail = (i) => (qRule === 'adaptive' ? Q.fail[i] : Q.fixed_fail[i]);
  function paintQ() {
    const cl = (i) => { const f = qFail(i), lq = Q.status[i] === 1; return f ? (lq ? 1 : 2) : (lq ? 3 : 0); };
    const colors = QCOL;
    const order = Q.batch.map((_, i) => i).sort((a, b) => cl(a) - cl(b));
    SQ.draw({ idx: order, x: lgen, y: Q.pct_mt.map((v) => Math.min(v, 34.5)), color: (i) => colors[cl(i)], r: 2.1 });
    thrL.selectAll('*').remove();
    const lines = qRule === 'adaptive'
      ? Q.thresholds.map((t, b) => ({ gx: Math.log10(t.min_genes), my: t.max_mt, lab: t.batch, dash: b ? '2 3' : '' }))
      : [{ gx: Math.log10(200), my: 10, lab: 'fixed', dash: '' }];
    lines.forEach((l, k) => {
      thrL.append('line').attr('x1', SQ.F.x(l.gx)).attr('x2', SQ.F.x(l.gx)).attr('y1', 0).attr('y2', SQ.F.h).attr('stroke', ink()).attr('stroke-width', 1.4).attr('stroke-dasharray', l.dash);
      thrL.append('line').attr('x1', 0).attr('x2', SQ.F.w).attr('y1', SQ.F.y(l.my)).attr('y2', SQ.F.y(l.my)).attr('stroke', ink()).attr('stroke-width', 1.4).attr('stroke-dasharray', l.dash);
      thrL.append('text').attr('x', SQ.F.x(l.gx) + 4).attr('y', 12 + k * 13).attr('font-size', 11).attr('fill', ink2()).attr('stroke', css('--surface')).attr('stroke-width', 3).attr('paint-order', 'stroke').text(`${l.lab}: ${Math.round(10 ** l.gx)} genes`);
      thrL.append('text').attr('x', SQ.F.w - 4).attr('y', SQ.F.y(l.my) - 4 - (k ? -13 : 0)).attr('text-anchor', 'end').attr('font-size', 11).attr('fill', ink2()).attr('stroke', css('--surface')).attr('stroke-width', 3).attr('paint-order', 'stroke').text(`${l.lab}: ${f1(l.my)}% mito`);
    });
    const removed = sum(Q.batch.map((_, i) => qFail(i))), tp = Q.batch.filter((_, i) => qFail(i) && Q.status[i] === 1).length;
    const fpT = Q.batch.filter((_, i) => qFail(i) && Q.status[i] !== 1 && Q.type[i] === 0).length;
    qStat = { 'droplets removed': removed, 'low-quality caught': `${tp}/${Q.summary.n_lowq}`, 'healthy cells removed': removed - tp, 'of which tumour cells': fpT };
    paintD();
  }
  function paintD() {
    const bins = d3.bin().domain([0, 1]).thresholds(25);
    const bs = bins(Q[dScore].filter((_, i) => Q.status[i] !== 2)), bd = bins(Q[dScore].filter((_, i) => Q.status[i] === 2));
    SD.y.domain([0, d3.max(bs.concat(bd), (b) => b.length)]).nice();
    SD.g.select('.axis.y').call(d3.axisLeft(SD.y).tickValues([0, 10, 100, 1000, 5000].filter((v) => v <= SD.y.domain()[1])).tickFormat(d3.format('d')));
    SD.g.select('.gridline').call(d3.axisLeft(SD.y).tickValues([0, 10, 100, 1000, 5000]).tickSize(-SD.w).tickFormat(''));
    dLayer.selectAll('*').remove();
    [[bs, cat(0)], [bd, cat(2)]].forEach(([bb, c], k) => bb.forEach((b, j) => {
      const w = SD.x(b.x1) - SD.x(b.x0);
      dLayer.append('rect').attr('x', SD.x(b.x0) + (k ? w / 2 : 0)).attr('width', w / 2 - 0.5).attr('y', SD.y(b.length)).attr('height', SD.h - SD.y(b.length)).attr('fill', c)
        .on('mousemove', (e) => tip(e, `${k ? 'doublets' : 'single cells'} with score ${f2(b.x0)}–${f2(b.x1)}: ${b.length}`)).on('mouseleave', untip);
    }));
    dThrL.selectAll('*').remove();
    dThrL.append('line').attr('x1', SD.x(dThr)).attr('x2', SD.x(dThr)).attr('y1', 0).attr('y2', SD.h).attr('stroke', ink()).attr('stroke-width', 2);
    dThrL.append('text').attr('x', SD.x(dThr) - 5).attr('y', 12).attr('text-anchor', 'end').attr('font-size', 11.5).attr('fill', ink()).text('called doublet →');
    const call = Q.batch.map((_, i) => Q[dScore][i] >= dThr), td = Q.batch.filter((_, i) => call[i] && Q.status[i] === 2).length, nc = sum(call.map(Number));
    const auc = dScore === 'dbl_clf' ? Q.roc['Classifier (scDblFinder-like)'].auc : Q.roc['kNN score (Scrublet-like)'].auc;
    sQ.set({ ...qStat, 'called doublets': nc, 'true doublets caught': `${td}/${Q.summary.n_dbl}`, 'false calls': nc - td, 'score AUC': auc.toFixed(3) });
  }
  paintQ();
  const dblCaught0 = Q.batch.filter((_, i) => Q.status[i] === 2 && Q.dbl_clf[i] >= Q.dbl_threshold).length;
  const fixTum = Q.summary.fixed_fp_by_type.Tumour, nTum = Q.type.filter((t) => t === 0).length;
  prose(rb, `<p>The adaptive rule removes ${Q.summary.qc_fail} droplets and catches ${Q.summary.qc_tp} of ${Q.summary.n_lowq} low-quality cells. The fixed rule removes ${Q.summary.fixed_fail}, of which <b>${fixTum} are healthy tumour cells</b> (${pct(fixTum / nTum)} of all tumour droplets): tumour cells in this simulation have higher mitochondrial fractions, and a global 10% cut-off treats a biological difference as damage. Thresholds should be set per batch and inspected per cell type.</p>
    <p>A doublet score separates doublets well (AUC ${Q.roc['Classifier (scDblFinder-like)'].auc.toFixed(3)} for the classifier, ${Q.roc['kNN score (Scrublet-like)'].auc.toFixed(3)} for the kNN score), whereas total counts alone reach only ${Q.roc['Total counts only'].auc.toFixed(3)}. The adaptive QC rule removes ${Q.batch.filter((_, i) => Q.status[i] === 2 && Q.fail[i]).length} of the ${Q.summary.n_dbl} doublets, so a separate score is needed. At the default threshold of ${f2(Q.dbl_threshold)}, ${Q.summary.n_dbl - dblCaught0} of ${Q.summary.n_dbl} doublets remain in the data, which is why clusters expressing two cell types' markers need checking later.</p>`);

  // ---- Step 2: normalisation and HVGs
  rb = rung(ch, 2, 'Normalisation and highly variable genes',
    `<p>Sequencing depth differs between cells for technical reasons (median ${fmtInt(E.lib_median)} UMIs here, 5th–95th percentile ${fmtInt(E.lib_q[0])}–${fmtInt(E.lib_q[1])}). <b>Normalisation</b> divides each cell's counts by its total, scales to a common size and takes log(1 + x), so that cells are compared by relative expression.</p>
     <p>Most genes carry no information about cell type. <b>Highly variable gene (HVG)</b> selection keeps the genes whose variance, relative to genes of similar mean expression, is largest. These define the space used by PCA.</p>`);
  const f2g = figure(rb, 'Library size and gene selection', 'Left: PC1 against PC2 for the cells, coloured by log10 library size. Right: every gene by mean and by normalised dispersion (variance relative to genes of similar mean); filled points are the top-N genes. "Structured" genes are the markers and programme genes built into the simulation.');
  let nMode = 'raw', nHV = E.n_hvg;
  select(f2g.controls, { label: 'PCA input', value: nMode, options: [['raw', 'Raw counts, no scaling'], ['norm', 'Normalised and log-transformed']], onChange: (v) => { nMode = v; paintN(); } });
  slider(f2g.controls, { label: 'top-N variable genes', min: 50, max: E.n_genes, step: 25, value: nHV, fmt: (v) => fmtInt(v), onInput: (v) => { nHV = v; paintH(); } });
  const sN = stats(f2g.root);
  const rowN = el('div', { class: 'fig-row' }); f2g.body.append(rowN);
  const na = el('div'), nb = el('div'); rowN.append(na, nb);
  const lglib = C.lib.map((v) => Math.log10(v)), lcol = sequential(d3.quantile(lglib, 0.02), d3.quantile(lglib, 0.98));
  const SN = umapPlot(na, { width: 460, height: 340, hover: (i) => `library size ${fmtInt(C.lib[i])} UMIs<br>${CT[T[i]] || 'doublet'}` });
  const FH = frame(nb, { width: 460, height: 340, x: [0, 6.6], y: [-1.5, 10.5], xLabel: 'mean expression (log)', yLabel: 'normalised dispersion' });
  const hvL = FH.layer('g');
  legend(nb, [['structured gene, selected', cat(1)], ['noise gene, selected', cat(0)], ['not selected', muted()]]);
  const corr = (a, b) => { const ma = d3.mean(a), mb = d3.mean(b); let s = 0, x = 0, y = 0; a.forEach((v, i) => { s += (v - ma) * (b[i] - mb); x += (v - ma) ** 2; y += (b[i] - mb) ** 2; }); return s / Math.sqrt(x * y); };
  const rRaw = corr(C.pca_raw[0], lglib), rNorm = corr(C.pca[0], lglib);
  function paintN() {
    const X = nMode === 'raw' ? C.pca_raw : C.pca;
    SN.show(X[0], X[1], (i) => lcol(lglib[i]), { r: 2.4 });
    SN.axes('PC 1', 'PC 2'); SN.labels([]);
    sN.set({ 'correlation of PC1 with log library size': (nMode === 'raw' ? rRaw : rNorm).toFixed(2), 'top-N selected': fmtInt(nHV), 'structured genes selected': `${selStruct()}/${nStruct}`, 'noise genes selected': nHV - selStruct() });
  }
  const selStruct = () => hvRank.slice(0, nHV).filter((g) => isStruct[g]).length;
  const inTop = new Set();
  hvL.selectAll('circle').data(d3.range(E.n_genes)).join('circle').attr('cx', (g) => FH.x(E.log_mean[g])).attr('cy', (g) => FH.y(Math.min(10.3, E.z_disp[g]))).attr('r', 2.6)
    .on('mousemove', (e, g) => tip(e, `${E.genes[g]} (${E.cat[g].replace('module:', 'programme ')})<br>mean ${f2(E.log_mean[g])} · dispersion ${f2(E.z_disp[g])}<br>rank ${hvRank.indexOf(g) + 1}`)).on('mouseleave', untip);
  function paintH() {
    inTop.clear(); hvRank.slice(0, nHV).forEach((g) => inTop.add(g));
    hvL.selectAll('circle').attr('fill', (g) => (inTop.has(g) ? (isStruct[g] ? cat(1) : cat(0)) : muted())).attr('fill-opacity', (g) => (inTop.has(g) ? 0.85 : 0.25));
    paintN();
  }
  paintH();
  prose(rb, `<p>Without normalisation, PC1 is essentially library size: its correlation with log library size is <b>${rRaw.toFixed(2)}</b>, against <b>${rNorm.toFixed(2)}</b> after normalisation. HVG selection is a trade-off. The top 100 genes contain ${hvRank.slice(0, 100).filter((g) => isStruct[g]).length} of the ${nStruct} structured genes; the default ${fmtInt(E.n_hvg)} capture ${hvRank.slice(0, E.n_hvg).filter((g) => isStruct[g]).length}, but ${pct(1 - hvRank.slice(0, E.n_hvg).filter((g) => isStruct[g]).length / E.n_hvg)} of the selected genes are then noise, because the simulation contains ${fmtInt(E.filler_total)} such genes. Real datasets typically use 2,000–3,000 HVGs; the best number depends on how many informative genes exist.</p>`);

  // ---- Step 3: PCA, graph, batch
  rb = rung(ch, 3, 'PCA, the neighbour graph and batch effects',
    `<p><b>PCA</b> compresses the selected genes into a few components; the first ${E.npc} are kept (${pct(cum20)} of variance), and the later ones mostly carry noise. Cells are compared in this ${E.npc}-dimensional space. Each cell is then linked to its nearest neighbours, giving the <b>kNN graph</b> on which clustering runs.</p>
     <p>Chemistry ('${C.batches[0]}' and '${C.batches[1]}') shifts expression. If batches are separate in PC space, the graph links cells of the same batch to each other. <b>Harmony</b> iteratively shifts cells in PC space so that every cluster contains cells from all batches, keeping the cell-type structure.</p>`);
  const f3g = figure(rb, 'Variance per component, and who a cell\'s neighbours are', 'Left: share of the total variance explained by each of the first 30 PCs; the dashed line marks the number of PCs used. Right: UMAP of the cells; hover a cell to draw links to its 10 nearest neighbours in the 20-PC space.');
  let eMode = 'uncorrected', cMode = 'batch';
  select(f3g.controls, { label: 'Embedding', value: eMode, options: [['uncorrected', EMB_NAME.uncorrected], ['harmony', EMB_NAME.harmony]], onChange: (v) => { eMode = v; paintK(); } });
  select(f3g.controls, { label: 'Colour by', value: cMode, options: [['batch', 'Chemistry (batch)'], ['type', 'True cell type']], onChange: (v) => { cMode = v; paintK(); } });
  const sK = stats(f3g.root);
  const rowK = el('div', { class: 'fig-row' }); f3g.body.append(rowK);
  const ka = el('div'), kb = el('div'); rowK.append(ka, kb);
  const NSC = 30, FS = frame(ka, { width: 400, height: 340, x: [0.5, NSC + 0.5], y: [0, 0.04], xLabel: 'principal component', yLabel: 'fraction of variance' });
  const cumV = d3.cumsum(E.var_ratio);
  FS.layer('b').selectAll('rect').data(E.var_ratio.slice(0, NSC)).join('rect').attr('x', (d, i) => FS.x(i + 1) - 0.4 * FS.w / NSC).attr('width', 0.8 * FS.w / NSC).attr('y', (d) => FS.y(d)).attr('height', (d) => FS.h - FS.y(d))
    .attr('fill', (d, i) => (i < E.npc ? cat(0) : muted())).on('mousemove', (e, d) => tip(e, `PC ${E.var_ratio.indexOf(d) + 1}: ${pct(d, 1)} of variance<br>cumulative ${pct(cumV[E.var_ratio.indexOf(d)], 1)}`)).on('mouseleave', untip);
  FS.layer('c').append('line').attr('x1', FS.x(E.npc + 0.5)).attr('x2', FS.x(E.npc + 0.5)).attr('y1', 0).attr('y2', FS.h).attr('stroke', ink2()).attr('stroke-dasharray', '4 4');
  FS.layer('c').append('text').attr('x', FS.x(E.npc + 0.5) - 6).attr('y', FS.y(0.03)).attr('text-anchor', 'end').attr('font-size', 11.5).attr('fill', ink2()).text(`${E.npc} PCs: ${pct(cum20)} of variance`);
  legend(ka, [['PC used for the graph', cat(0)], ['PC not used', muted()]]);
  const SK = umapPlot(kb, { width: 440, height: 340, hover: (i) => { links(i); return `${CT[T[i]] || 'doublet'} · ${C.batches[B[i]]} · patient ${C.patients[P[i]]}<br>${C.knn[eMode][i].filter((j) => B[j] === B[i]).length}/10 neighbours from the same batch<br>${C.knn[eMode][i].filter((j) => T[j] === T[i]).length}/10 of the same type`; }, onMiss: () => SK.F.layer('links').selectAll('*').remove() });
  const linkL = SK.F.layer('links').attr('pointer-events', 'none');
  const links = (i) => {
    const [ux, uy] = C.umap[eMode];
    linkL.selectAll('line').data(C.knn[eMode][i]).join('line').attr('x1', SK.F.x(ux[i])).attr('y1', SK.F.y(uy[i])).attr('x2', (j) => SK.F.x(ux[j])).attr('y2', (j) => SK.F.y(uy[j])).attr('stroke', ink()).attr('stroke-width', 1.2);
  };
  const legK = el('div'); kb.append(legK);
  function paintK() {
    SK.show(C.umap[eMode][0], C.umap[eMode][1], cMode === 'batch' ? (i) => cat(B[i] ? 1 : 0) : (i) => typeColor(T[i]));
    SK.labels(cMode === 'type' ? C.umap[eMode] : null, true); SK.axes('UMAP 1', 'UMAP 2');
    legK.innerHTML = ''; legend(legK, cMode === 'batch' ? C.batches.map((b, i) => [b, cat(i)]) : CT.map((t, i) => [t, cat(i)]).concat([['doublet', muted()]]));
    const k = eMode === 'uncorrected' ? knnU : knnH;
    sK.set({ 'neighbours from the same batch': pct(k.batch, 1), [`expected if batches were mixed (${C.batches[1]} = ${pct(fracV3)})`]: pct(fracV3 ** 2 + (1 - fracV3) ** 2), 'neighbours of the same cell type': pct(k.type, 1), 'IFN-γ programme retained (R²)': I.metrics[eMode].ifn_r2.toFixed(2) });
  }
  paintK();
  prose(rb, `<p>Before correction, ${pct(knnU.batch, 1)} of neighbours come from the cell's own batch, although about ${pct(fracV3 ** 2 + (1 - fracV3) ** 2)} would be expected if batches were mixed. After Harmony the value is ${pct(knnH.batch, 1)}. Cell-type purity of the neighbours is ${pct(knnU.type, 1)} before and ${pct(knnH.type, 1)} after, so in this simulation the batch effect is smaller than the differences between cell types. It nevertheless splits clusters, as the next step shows.</p>`);

  // ---- Step 4: Leiden + resolution lab
  rb = rung(ch, 4, 'Leiden clustering and the resolution parameter (lab)',
    `<p><b>Leiden</b> finds groups of cells with more links among themselves than expected by chance. It maximises a score called <b>modularity</b>, with a <b>resolution</b> parameter that sets the size of the expected groups: low values merge, high values split. There is no value that is correct in general. Here the truth is known, so every setting can be scored with the <b>adjusted Rand index (ARI)</b> against the true cell types (1 = identical, 0 = no better than random).</p>
     <p>Choose an embedding and a resolution. The map on the left is drawn from the same UMAP regardless of clustering; the curves on the right show the ARI at every resolution. Click a point on a curve to jump to it.</p>`);
  const f4 = figure(rb, 'Lab: embedding × resolution', 'Left: UMAP coloured by the chosen property. Right: ARI against true cell type for every resolution, one curve per embedding; the ring marks the current setting. ARI is computed on the plotted singlets (doublets excluded).');
  let lEmb = 'harmony', lRi = R02, lCol = 'cluster', sweep = null;
  select(f4.controls, { label: 'Embedding', value: lEmb, options: Object.entries(EMB_NAME), onChange: (v) => { lEmb = v; paintL(); } });
  const sRes = slider(f4.controls, { label: 'resolution', min: 0, max: C.res.length - 1, step: 1, value: lRi, fmt: (v) => C.res[v], onInput: (v) => { haltSweep(); lRi = v; paintL(); } });
  select(f4.controls, { label: 'Colour by', value: lCol, options: [['cluster', 'Leiden cluster'], ['type', 'True cell type'], ['batch', 'Chemistry'], ['patient', 'Patient']], onChange: (v) => { lCol = v; paintL(); } });
  button(f4.controls, 'Sweep resolution', () => { haltSweep(); lRi = 0; let k = 0; sweep = every(700, () => { k++; if (k >= C.res.length) { haltSweep(); return; } lRi = k; sRes.set(k); paintL(); }); sRes.set(0); paintL(); }, true);
  const sL = stats(f4.root);
  const rowL = el('div', { class: 'fig-row' }); f4.body.append(rowL);
  const la = el('div'), lb = el('div'); rowL.append(la, lb);
  const SL = umapPlot(la, { width: 480, height: 380, hover: (i) => `${CT[T[i]] || 'doublet'} · ${C.batches[B[i]]} · patient ${C.patients[P[i]]}<br>cluster ${C.leiden[lEmb][rk(C.res[lRi])][i]} at resolution ${C.res[lRi]}` });
  const legL = el('div'); la.append(legL);
  const FR = frame(lb, { width: 400, height: 380, x: [0.008, 2.5], y: [0, 1.05], xScale: d3.scaleLog(), xLabel: 'Leiden resolution (log scale)', yLabel: 'ARI vs true cell type' });
  FR.g.select('.axis.x').call(d3.axisBottom(FR.x).tickValues([0.01, 0.05, 0.2, 0.5, 1, 2]).tickFormat(d3.format('~g')));
  const ECOL = { uncorrected: cat(2), harmony: cat(0), over: cat(5) };
  Object.keys(ECOL).forEach((e) => {
    FR.layer('l').append('path').datum(C.res.map((r, k) => [r, ARI[e][k]])).attr('fill', 'none').attr('stroke', ECOL[e]).attr('stroke-width', 2).attr('d', d3.line().x((d) => FR.x(d[0])).y((d) => FR.y(d[1])));
    FR.layer('p').selectAll('.pt-' + e).data(C.res.map((r, k) => ({ r, k, e }))).join('circle').attr('class', 'pt-' + e).attr('cx', (d) => FR.x(d.r)).attr('cy', (d) => FR.y(ARI[e][d.k])).attr('r', 4.5).attr('fill', ECOL[e]).attr('stroke', css('--surface')).attr('stroke-width', 1.2).style('cursor', 'pointer')
      .on('mousemove', (ev, d) => { const s = cs(d.e, d.k); tip(ev, `${EMB_NAME[d.e]}<br>resolution ${d.r}: ${s.k} clusters, ARI ${f2(s.ari)}`); }).on('mouseleave', untip)
      .on('click', (ev, d) => { haltSweep(); lEmb = d.e; lRi = d.k; selEmb.value = lEmb; sRes.set(lRi); paintL(); });
  });
  const selEmb = f4.controls.querySelector('select');
  const curL = FR.layer('cur').append('circle').attr('r', 9).attr('fill', 'none').attr('stroke', ink()).attr('stroke-width', 2);
  FR.layer('t').append('text').attr('x', FR.x(0.012)).attr('y', FR.y(0.18)).attr('font-size', 11.5).attr('fill', ink2()).text('higher resolution subdivides true types');
  legend(lb, Object.entries(EMB_NAME).map(([e, t]) => [t, ECOL[e]]));
  function haltSweep() { if (sweep) { sweep(); sweep = null; } }
  function paintL() {
    const s = cs(lEmb, lRi), L = s.L, [ux, uy] = C.umap[lEmb];
    const colour = { cluster: (i) => clusterColor(L[i]), type: (i) => typeColor(T[i]), batch: (i) => cat(B[i] ? 1 : 0), patient: (i) => clusterColor(P[i]) }[lCol];
    SL.show(ux, uy, colour);
    SL.labels(lCol === 'cluster' ? { cx: (g) => d3.mean(all.filter((i) => L[i] === g), (i) => ux[i]), cy: (g) => d3.mean(all.filter((i) => L[i] === g), (i) => uy[i]), groups: [...new Set(L)] } : lCol === 'type' ? C.umap[lEmb] : null, lCol === 'type');
    SL.axes('UMAP 1', 'UMAP 2');
    legL.innerHTML = '';
    if (lCol === 'type') legend(legL, CT.map((t, i) => [t, cat(i)]).concat([['doublet', muted()]]));
    if (lCol === 'batch') legend(legL, C.batches.map((b, i) => [b, cat(i)]));
    curL.attr('cx', FR.x(C.res[lRi])).attr('cy', FR.y(Math.max(0, s.ari)));
    sL.set({ clusters: s.k, 'ARI vs true type': s.ari.toFixed(2), 'types merged into another': s.merged.length ? s.merged.join(', ') : 'none', 'types split across clusters': s.split.length ? s.split.join(', ') : 'none', 'batch mixing (iLISI, 0–1)': I.metrics[lEmb].ilisi_norm.toFixed(2), 'IFN-γ programme kept (R²)': I.metrics[lEmb].ifn_r2.toFixed(2) });
  }
  paintL();
  const hU = cs('uncorrected', R02), hH = cs('harmony', R02), hH2 = cs('harmony', R20), hO = cs('over', R02), hH001 = cs('harmony', R001);
  prose(rb, '<p><b>Suggested comparisons</b>:</p>');
  table(rb, [
    { t: '<b>No correction</b> → <b>Harmony</b>, resolution 0.2', s: `Clusters ${hU.k} → <b>${hH.k}</b>; ARI ${f2(hU.ari)} → <b>${f2(hH.ari)}</b>`, m: `Without correction, cell types that exist in both chemistries are split into one cluster per chemistry (${hU.split.length ? hU.split.join(', ') : 'none'} split here). Correction restores one cluster per type.` },
    { t: 'Harmony, resolution <b>0.01</b> → <b>0.5</b> → <b>2.0</b>', s: `Clusters ${hH001.k} → ${cs('harmony', R05).k} → <b>${hH2.k}</b>; ARI ${f2(hH001.ari)} → ${f2(cs('harmony', R05).ari)} → <b>${f2(hH2.ari)}</b>`, m: `From resolution ${C.res[0]} to ${d3.max(C.res.filter((_, ri) => cs('harmony', ri).k === NT))} the number of clusters matches the ${NT} true types. At 2.0 the extra clusters subdivide ${hH2.split.join(', ') || 'no type'}; ARI falls although nothing is wrong with the graph, because the truth contains no such subdivision. A real dataset may well contain one.` },
    { t: '<b>Over-corrected</b> embedding, resolution 0.2', s: `Clusters <b>${hO.k}</b>; ARI ${f2(hO.ari)}; types merged: ${hO.merged.join(', ') || 'none'}; IFN-γ R² ${I.metrics.over.ifn_r2.toFixed(2)} vs ${I.metrics.harmony.ifn_r2.toFixed(2)}`, m: `Using patient as the batch removes real differences between patients (the IFN-γ programme differs between inflamed and excluded tumours) together with the technical ones. Batch-mixing metrics stay similar to Harmony (iLISI ${I.metrics.over.ilisi_norm.toFixed(2)} vs ${I.metrics.harmony.ilisi_norm.toFixed(2)}) while biology is lost.` },
    { t: 'Colour by <b>Chemistry</b> on the uncorrected embedding at 0.8', s: `${I.clusters.uncorrected['0.8'].filter((c) => c.frac_v3 < 0.05 || c.frac_v3 > 0.95).length} of ${I.clusters.uncorrected['0.8'].length} clusters (all ${fmtInt(C.n_kept)} cells) contain almost only one chemistry`, m: 'Single-batch clusters at moderate resolution indicate a technical split rather than new cell types.' },
  ], [['t', 'Try'], ['s', 'What you see'], ['m', 'What it means']], { cls: 'compact' });
  quiz(rb, 'The ARI curve for Harmony is flat near 1.0 at low resolutions. Is a flat curve evidence that the clustering is correct for real data?',
    'No. ARI needs ground truth, which is only available in a simulation. In real data, evidence for a resolution comes from marker genes that distinguish each cluster, stability across seeds and subsamples, and whether the split matches known biology.');

  // ---- Step 5: marker-based annotation
  rb = rung(ch, 5, 'Annotation with marker genes',
    '<p>Marker-based annotation reads the expression of known genes in each cluster: <i>EPCAM</i> for epithelial/tumour cells, <i>CD8A</i> for cytotoxic T cells, <i>FOXP3</i> for regulatory T cells, <i>MS4A1</i> for B cells, <i>CD68</i> for macrophages, <i>COL1A1</i> for fibroblasts and <i>PECAM1</i> for endothelium. In a <b>dot plot</b>, dot size gives the fraction of cells in which a gene is detected and colour gives its mean expression. A cluster is named after the marker detected in the largest fraction of its cells, provided that fraction is at least 50%.</p>');
  const f5 = figure(rb, 'Dot plot of marker and state genes', 'Rows: Harmony clusters at the chosen resolution. Left block: one marker per cell type. Right block: genes that vary within a type. Last two columns: the label assigned by the rule and the most frequent true type in the cluster. Hover a dot for the values.');
  let mRi = R02;
  select(f5.controls, { label: 'Resolution', value: String(mRi), options: [R02, R08, R20].map((r) => [String(r), `${C.res[r]} (${nClus('harmony', r)} clusters)`]), onChange: (v) => { mRi = +v; paintM(); } });
  const sM = stats(f5.root);
  const gIdx = (g) => C.genes.indexOf(g);
  const dotSvg = d3.select(f5.body).append('svg').attr('viewBox', '0 0 860 300').attr('width', 860).attr('role', 'img').attr('aria-label', 'dot plot of marker genes by cluster');
  function markerInfo(ri) {
    const L = C.leiden.harmony[rk(C.res[ri])], K = [...new Set(L)];
    return K.map((c) => {
      const idx = all.filter((i) => L[i] === c), frac = {}, mean = {};
      MARK.map((m) => m[0]).concat(STATE).forEach((g) => { const x = C.expr[g]; frac[g] = idx.filter((i) => x[i] > 0).length / idx.length; mean[g] = d3.mean(idx, (i) => x[i]); });
      const best = MARK.reduce((a, m) => (frac[m[0]] > frac[a[0]] ? m : a));
      const lab = frac[best[0]] >= 0.5 ? best[1] : 'unassigned';
      const tc = new Array(NT).fill(0); idx.forEach((i) => { if (T[i] >= 0) tc[T[i]]++; });
      return { c, n: idx.length, frac, mean, lab, truth: CT[tc.indexOf(d3.max(tc))], idx };
    }).sort((a, b) => CT.indexOf(a.lab) - CT.indexOf(b.lab) || b.n - a.n);
  }
  function paintM() {
    const info = markerInfo(mRi);
    const rowH = 21, top = 34, x0 = 128, dx = 46, gx = [...MARK.map((m) => m[0]), ...STATE];
    const H = top + info.length * rowH + 10;
    dotSvg.attr('viewBox', `0 0 860 ${H}`).selectAll('*').remove();
    const cx = (j) => x0 + j * dx + (j >= MARK.length ? 16 : 0) + dx / 2;
    const col = sequential(0, 4.6), rr = (f) => 1.5 + 8.5 * Math.sqrt(f);
    gx.forEach((g, j) => dotSvg.append('text').attr('x', cx(j)).attr('y', 22).attr('text-anchor', 'middle').attr('font-size', 11).attr('font-style', 'italic').attr('fill', ink2()).text(g));
    dotSvg.append('text').attr('x', 664).attr('y', 22).attr('font-size', 11).attr('fill', ink2()).text('assigned');
    dotSvg.append('text').attr('x', 760).attr('y', 22).attr('font-size', 11).attr('fill', ink2()).text('true majority');
    info.forEach((r, i) => {
      const y = top + i * rowH + rowH / 2;
      if (i % 2 === 0) dotSvg.append('rect').attr('x', 0).attr('y', y - rowH / 2).attr('width', 860).attr('height', rowH).attr('fill', css('--surface-2')).attr('fill-opacity', 0.55);
      dotSvg.append('text').attr('x', x0 - 8).attr('y', y + 4).attr('text-anchor', 'end').attr('font-size', 11.5).attr('fill', ink2()).text(`cluster ${r.c} · n = ${r.n}`);
      gx.forEach((g, j) => dotSvg.append('circle').attr('cx', cx(j)).attr('cy', y).attr('r', rr(r.frac[g])).attr('fill', col(r.mean[g])).attr('stroke', muted()).attr('stroke-width', 0.6)
        .on('mousemove', (e) => tip(e, `cluster ${r.c} · ${g}<br>detected in ${pct(r.frac[g])} of ${r.n} cells<br>mean ${f2(r.mean[g])}`)).on('mouseleave', untip));
      dotSvg.append('text').attr('x', 664).attr('y', y + 4).attr('font-size', 12).attr('font-weight', 600).attr('fill', ink()).text(r.lab);
      dotSvg.append('text').attr('x', 760).attr('y', y + 4).attr('font-size', 12).attr('fill', ink2()).text(r.truth + (r.truth === r.lab ? '  ✓' : '  ✗'));
    });
    const dup = d3.rollups(info, (v) => v.length, (r) => r.lab).filter((d) => d[1] > 1).map((d) => `${d[0]} (${d[1]})`);
    sM.set({ clusters: info.length, 'labels matching the true majority type': `${info.filter((r) => r.lab === r.truth).length}/${info.length}`, 'types given to several clusters': dup.length ? dup.join(', ') : 'none', 'cells in clusters with an unassigned label': sum(info.filter((r) => r.lab === 'unassigned').map((r) => r.n)) });
  }
  paintM();
  const at02 = (() => { const L = C.leiden.harmony[rk(C.res[R02])]; return [...new Set(L)].map((c) => { const idx = all.filter((i) => L[i] === c); return { c, n: idx.length, fr: (g) => idx.filter((i) => C.expr[g][i] > 0).length / idx.length }; }); })();
  const sm = at02.slice().sort((a, b) => a.n - b.n)[0];
  const oth = Math.max(...at02.filter((r) => r !== sm).map((r) => r.fr('FOXP3')));
  worked(rb, `Naming the smallest cluster at resolution ${C.res[R02]}`,
    `<p>Cluster ${sm.c} contains ${sm.n} cells. <i>FOXP3</i> is detected in <b>${pct(sm.fr('FOXP3'))}</b> of them, against at most ${pct(oth)} in any other cluster; the other markers are detected in at most ${pct(Math.max(...MARK.filter((m) => m[0] !== 'FOXP3').map((m) => sm.fr(m[0]))))}. The rule therefore names it <b>Treg</b>.</p>
     <p>This works because each marker is detected in all cells of its type. In real data markers are detected in a fraction of cells (dropout) and several types share genes, so a combination of 2–3 markers per type is used, and the result is reviewed by an expert.</p>`);
  prose(rb, `<p>At resolution ${rk(C.res[R20])} the ${hH2.k} clusters receive only ${new Set(markerInfo(R20).map((r) => r.lab)).size} distinct labels: the sub-clusters of one type carry the same label. Whether such sub-clusters are different states (for instance proliferating cells, <i>MKI67</i>) or an artefact of high resolution is decided with state genes and replicates, not with the identity markers.</p>`);

  // ---- Step 6: label transfer
  rb = rung(ch, 6, 'Annotation by label transfer from a reference',
    `<p>With an annotated reference, a classifier can name every cell directly. Here a <b>logistic regression</b> (L2-penalised) is trained on ${fmtInt(A.n_ref)} reference cells from 6 other patients, using the ${fmtInt(A.n_features)} HVGs, and applied to the query cells. Each prediction comes with a probability. A <b>marker-score</b> method, which averages marker-gene expression per type, is shown for comparison. Choose the method and a minimum confidence below which a cell stays <b>unassigned</b>.</p>`);
  const f6 = figure(rb, 'Label transfer and its failure modes', 'Left: UMAP coloured by the predicted label, or by whether the label agrees with the truth. Right: true type (rows) against assigned label (columns), as a share of the row; numbers are cells. "Full reference" contains all 8 types; "no Treg in reference" omits regulatory T cells.');
  const MODES = [['lt_full', 'Logistic regression, full reference'], ['lt_notreg', 'Logistic regression, no Treg in reference'], ['marker', 'Marker-score method']];
  let aM = 'lt_full', aThr = 0, aCol = 'label';
  select(f6.controls, { label: 'Method', value: aM, options: MODES, onChange: (v) => { aM = v; paintA(); } });
  slider(f6.controls, { label: 'minimum confidence', min: 0, max: 0.99, step: 0.01, value: aThr, fmt: f2, onInput: (v) => { aThr = v; paintA(); } });
  select(f6.controls, { label: 'Colour by', value: aCol, options: [['label', 'Assigned label'], ['ok', 'Agreement with truth']], onChange: (v) => { aCol = v; paintA(); } });
  const sA = stats(f6.root);
  const rowA = el('div', { class: 'fig-row' }); f6.body.append(rowA);
  const aa = el('div'), ab = el('div'); rowA.append(aa, ab);
  const SA = umapPlot(aa, { width: 470, height: 400, hover: (i) => `true: ${CT[T[i]] || 'doublet'}<br>assigned: ${CT[C.ann[aM][i]]} (confidence ${f2(C.conf[aM][i])})` });
  const legA = el('div'); aa.append(legA);
  const cm = confusionSvg(ab);
  function paintA() {
    const pred = C.ann[aM], cf = C.conf[aM], un = (i) => cf[i] < aThr;
    const okc = [muted(), cat(0), cat(1), cat(2)];
    const fill = aCol === 'label' ? (i) => (un(i) ? muted() : typeColor(pred[i])) : (i) => (un(i) ? okc[0] : T[i] < 0 ? okc[3] : pred[i] === T[i] ? okc[1] : okc[2]);
    SA.show(C.umap.harmony[0], C.umap.harmony[1], fill);
    SA.labels(null); SA.axes('UMAP 1', 'UMAP 2');
    legA.innerHTML = '';
    legend(legA, aCol === 'label' ? CT.map((t, k) => [t, cat(k)]).concat([['unassigned', muted()]]) : [['agrees', okc[1]], ['disagrees', okc[2]], ['doublet (no correct label)', okc[3]], ['unassigned', okc[0]]]);
    const M = Array.from({ length: NT + 1 }, () => new Array(NT + 1).fill(0));
    all.forEach((i) => { M[T[i] < 0 ? NT : T[i]][un(i) ? NT : pred[i]]++; });
    cm.update(M, CT);
    const asg = sing.filter((i) => !un(i)), okA = asg.filter((i) => pred[i] === T[i]).length;
    const tr = sing.filter((i) => T[i] === 3), trOk = tr.filter((i) => !un(i) && pred[i] === 3).length;
    const dbl = all.filter((i) => T[i] < 0);
    sA.set({ 'singlets assigned': pct(asg.length / sing.length, 1), 'accuracy among assigned': pct(okA / Math.max(1, asg.length), 1), 'Treg recovered': `${trOk}/${tr.length}`, 'doublets given a label': `${dbl.filter((i) => !un(i)).length}/${dbl.length}`, 'median confidence, singlets': f2(d3.median(sing, (i) => cf[i])), 'median confidence, doublets': f2(d3.median(dbl, (i) => cf[i])) });
  }
  paintA();
  const acc = (m) => d3.mean(sing, (i) => (C.ann[m][i] === T[i] ? 1 : 0));
  const medC = (m, f) => d3.median(all.filter(f), (i) => C.conf[m][i]);
  const trM = (m) => sing.filter((i) => T[i] === 3 && C.ann[m][i] === 3).length;
  const uThr = (m, thr, f) => all.filter(f).filter((i) => C.conf[m][i] < thr).length / all.filter(f).length;
  prose(rb, '<p><b>Suggested comparisons</b>:</p>');
  table(rb, [
    { t: '<b>Full reference</b>, no confidence cut-off', s: `Accuracy on singlets ${pct(acc('lt_full'), 1)}; Treg recovered ${trM('lt_full')}/${typeN[3]}`, m: 'When the reference contains every type present, label transfer is accurate on cells of those types.' },
    { t: '<b>No Treg in reference</b>', s: `Treg recovered <b>${trM('lt_notreg')}/${typeN[3]}</b>; accuracy on singlets ${pct(acc('lt_notreg'), 1)}; median Treg confidence ${f2(medC('lt_notreg', (i) => T[i] === 3))} vs ${f2(medC('lt_notreg', (i) => T[i] >= 0 && T[i] !== 3))} for other singlets`, m: 'A classifier must choose among the labels it knows. A missing type is assigned to related types, with lower confidence.' },
    { t: 'Same, minimum confidence <b>0.5</b>', s: `${pct(uThr('lt_notreg', 0.5, (i) => T[i] === 3))} of Treg become unassigned, against ${pct(uThr('lt_notreg', 0.5, (i) => T[i] >= 0 && T[i] !== 3), 1)} of other singlets`, m: 'A confidence threshold turns "wrong label" into "unknown", which flags a possible missing type.' },
    { t: 'Doublets, any method, confidence <b>0.5</b>', s: `${pct(1 - uThr('lt_full', 0.5, (i) => T[i] < 0))} of the ${nDbl} doublets still receive a label (full reference; median confidence ${f2(medC('lt_full', (i) => T[i] < 0))})`, m: 'Confidence does not identify doublets: a hybrid profile resembles one of its parents closely. Doublets need their own detection (step 1).' },
  ], [['t', 'Try'], ['s', 'What you see'], ['m', 'What it means']], { cls: 'compact' });
  prose(rb, `<p>The marker-score method needs no reference and recovers ${trM('marker')} of ${typeN[3]} Treg, but its confidence is a rescaled score rather than a calibrated probability. Reference-based methods inherit the reference's labels, batch and resolution of annotation. In practice, the two approaches are combined: transfer labels, then verify them with markers.</p>`);

  // =====================================================================================
  // 5. PITFALLS
  // =====================================================================================
  ch = chapter(main, 'pitfalls', 'Pitfalls', 'Common problems, how to recognise them, and how to address them.');
  const ovr = I.metrics.over, hmn = I.metrics.harmony;
  cards(ch, [
    { tag: 'Symptom: a "new" cluster expressing two sets of markers', title: 'Doublets', body: `Doublets resemble healthy, RNA-rich cells and pass count-based QC. After doublet filtering at the default threshold, ${Q.summary.n_dbl - dblCaught0} of ${Q.summary.n_dbl} remain, and annotation methods label them confidently (median confidence ${f2(medC('lt_full', (i) => T[i] < 0))}). <b>Fix:</b> score doublets per sample before clustering, and inspect clusters that express markers of two types.` },
    { tag: 'Symptom: sub-clusters with no distinguishing marker', title: 'Over-clustering', body: `At resolution ${rk(C.res[R20])}, ${hH2.k} clusters are found for ${NT} types, and ARI falls from ${f2(hH.ari)} to ${f2(hH2.ari)}. Clusters can always be split further. <b>Fix:</b> require each cluster to have marker genes that distinguish it from its neighbours, and check stability across seeds and subsamples.` },
    { tag: 'Symptom: clusters that contain a single batch', title: 'Uncorrected batch effects', body: `Without correction at resolution ${rk(C.res[R08])}, ${cs('uncorrected', R08).k} clusters are found for ${NT} types (${I.clusters.uncorrected['0.8'].length} using all cells), several of them one chemistry only. <b>Fix:</b> colour clusters by batch and patient; correct the embedding, or include batch in a model.` },
    { tag: 'Symptom: batch mixing looks fine, biology disappears', title: 'Over-correction', body: `With patient used as the batch, the share of the IFN-γ programme explained by the embedding drops from ${hmn.ifn_r2.toFixed(2)} to ${ovr.ifn_r2.toFixed(2)}, while batch mixing is almost unchanged (iLISI ${ovr.ilisi_norm.toFixed(2)} vs ${hmn.ilisi_norm.toFixed(2)}). <b>Fix:</b> use the technical variable (chemistry, run) as the batch, and verify that known biological differences survive.` },
    { tag: 'Symptom: conclusions drawn from distances on the map', title: 'Reading UMAP as a measurement', body: `UMAP preserves local neighbourhoods (${pct(umPur)} of 10-neighbour sets share the cell type here) but not the distances between groups, the sizes of groups or their density. <b>Fix:</b> use UMAP to display; use clusters, PCs or the graph to measure.` },
    { tag: `Symptom: ${typeN[3]} Treg cells given other labels`, title: 'A missing type in the reference', body: `Without Treg in the reference, ${trM('lt_notreg')} of ${typeN[3]} Treg are recovered and the median confidence for them is ${f2(medC('lt_notreg', (i) => T[i] === 3))}. <b>Fix:</b> choose a reference that covers the tissue, apply a confidence threshold, and verify with markers.` },
    { tag: `Symptom: ${fixTum} healthy tumour cells removed`, title: 'Fixed QC thresholds', body: `The fixed rule removed ${Q.summary.fixed_fail} droplets against ${Q.summary.qc_fail} for the adaptive rule, and tumour cells lost most. <b>Fix:</b> set thresholds per batch from the distributions, and compare removed cells by type before and after.` },
    { tag: 'Symptom: tiny p-values for cluster differences', title: 'Testing genes between clusters defined by the same genes', body: 'Clusters are chosen to differ in expression, so a test between them on the same cells has inflated significance, even with no real difference. <b>Fix:</b> use cluster markers for ranking, not as p-values; test conditions on patient-level pseudobulk (see the pseudobulk page).' },
  ]);

  // =====================================================================================
  // 6. RECAP
  // =====================================================================================
  ch = chapter(main, 'recap', 'Recap', 'Key points.');
  recap(ch, [
    '<b>Pipeline:</b> QC → normalise → highly variable genes → PCA → kNN graph → Leiden → UMAP, then annotation. Every stage is a modelling choice.',
    'QC thresholds should be set <b>per batch</b> and checked by cell type; a global cut-off can remove a real cell type.',
    'Doublets pass QC and need their own score; clusters with two sets of markers are suspect.',
    'Normalisation removes depth: without it PC1 is library size. HVG selection trades signal against noise.',
    'Correct <b>technical</b> batches only. Over-correction removes real between-patient biology while batch-mixing metrics look as good as after a correct correction.',
    '<b>Resolution has no correct value.</b> Justify each cluster with markers and stability, not with the number of clusters.',
    'UMAP is a drawing of the graph: do not read distances, sizes or gaps from it.',
    'Label transfer is limited by the reference; marker-based annotation is limited by the choice of markers. Use confidence thresholds and verify one with the other.',
  ]);
  prose(ch, '<h3>In code</h3><p>Scanpy (Python) and Seurat (R) follow the same stages. The parameters shown are common defaults, not recommendations for every dataset.</p>');
  code(ch, {
    Python: `import scanpy as sc, scanpy.external as sce

adata = sc.read_h5ad("counts.h5ad")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
import numpy as np
from scipy.stats import median_abs_deviation as mad
def outlier(x, k=3):                                        # adaptive: k MADs from the median, per batch
    return np.abs(x - np.median(x)) > k * mad(x)
for col in ["log1p_n_genes_by_counts", "log1p_total_counts", "pct_counts_mt"]:
    adata.obs["out_" + col] = adata.obs.groupby("batch")[col].transform(outlier)
adata = adata[~adata.obs.filter(like="out_").any(axis=1)].copy()   # for mito, flag the high side only
sc.pp.scrublet(adata, batch_key="batch")                    # doublet scores per sample
adata = adata[~adata.obs["predicted_doublet"]].copy()

sc.pp.normalize_total(adata, target_sum=1e4); sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000, batch_key="batch")
sc.pp.pca(adata, n_comps=30, use_highly_variable=True)
sce.pp.harmony_integrate(adata, key="batch")                # corrects X_pca -> X_pca_harmony
sc.pp.neighbors(adata, n_neighbors=15, use_rep="X_pca_harmony")
for r in (0.2, 0.5, 1.0):
    sc.tl.leiden(adata, resolution=r, key_added=f"leiden_{r}", flavor="igraph", n_iterations=2)
sc.tl.umap(adata)                                            # display only
sc.tl.rank_genes_groups(adata, "leiden_0.5", method="wilcoxon")  # markers: for ranking, not p-values

# label transfer: train on an annotated reference, apply to the query
import celltypist
pred = celltypist.annotate(adata, model="Immune_All_Low.pkl", majority_voting=True)`,
    R: `library(Seurat)

obj <- CreateSeuratObject(counts = counts, min.cells = 3)
obj[["percent.mt"]] <- PercentageFeatureSet(obj, pattern = "^MT-")
# set per-batch thresholds from the distributions; remove doublets (e.g. scDblFinder) before this point

obj <- NormalizeData(obj)
obj <- FindVariableFeatures(obj, nfeatures = 2000)
obj <- ScaleData(obj); obj <- RunPCA(obj, npcs = 30)
obj <- harmony::RunHarmony(obj, group.by.vars = "batch")    # technical variable only
obj <- FindNeighbors(obj, reduction = "harmony", dims = 1:20, k.param = 15)
obj <- FindClusters(obj, resolution = c(0.2, 0.5, 1.0), algorithm = 4)   # 4 = Leiden
obj <- RunUMAP(obj, reduction = "harmony", dims = 1:20)

markers <- FindAllMarkers(obj, only.pos = TRUE, logfc.threshold = 0.25)

# label transfer from a reference Seurat object
anchors <- FindTransferAnchors(reference = ref, query = obj, dims = 1:30)
obj <- TransferData(anchors, refdata = ref$celltype, query = obj)`,
  });
  seeAlso(main, [['concepts', 'pca', 'PCA'], ['concepts', 'clustering', 'Clustering'], ['concepts', 'tsne-umap', 't-SNE and UMAP'], ['concepts', 'logistic-regression', 'Logistic regression'], ['omics', 'bulk-qc-batch', 'QC, normalisation and batch effects'], ['omics', 'sc-pseudobulk', 'Pseudobulk differential expression'], ['omics', 'overview', 'Study overview']]);

  // ---------------------------------------------------------------------------------------------
  function typeColor(t) { return t < 0 ? muted() : cat(t); }
  function cellTip(i) { return `${CT[T[i]] || 'doublet'} · ${C.batches[B[i]]}<br>patient ${C.patients[P[i]]} (${C.resp[P[i]] === 'R' ? 'responder' : 'non-responder'})`; }
  /** UMAP-style canvas scatter that can swap its coordinates. */
  function umapPlot(parent, { width, height, hover, onMiss }) {
    const S = canvasScatter(parent, { width, height, x: [0, 1], y: [0, 1], xLabel: ' ', yLabel: ' ', axes: false, grid: false, hover, onMiss });
    const labs = S.F.g.selectAll('.ax-label').nodes();
    const labL = S.F.layer('lab').attr('pointer-events', 'none');
    S.axes = (a, b) => { labs[0].textContent = a; labs[1].textContent = b; };
    S.show = (ux, uy, color, o = {}) => { S.F.x.domain(ext(ux)); S.F.y.domain(ext(uy)); S.draw({ idx: all, x: ux, y: uy, color, r: o.r || 2.4 }); S._x = ux; S._y = uy; };
    /** labels: [ux,uy] coordinates for type names, {cx,cy,groups} for clusters, or null to clear. */
    S.labels = (spec, byType) => {
      labL.selectAll('*').remove();
      if (!spec) return;
      let items;
      if (Array.isArray(spec) && spec.length) items = CT.map((t, k) => { const ix = sing.filter((i) => T[i] === k); return { x: d3.median(ix, (i) => spec[0][i]), y: d3.median(ix, (i) => spec[1][i]), t }; });
      else if (spec.groups) items = spec.groups.map((g) => ({ x: spec.cx(g), y: spec.cy(g), t: String(g) }));
      else return;
      items.forEach((o) => labL.append('text').attr('x', S.F.x(o.x)).attr('y', S.F.y(o.y)).attr('text-anchor', 'middle').attr('font-size', 11.5).attr('font-weight', 700).attr('fill', ink()).attr('stroke', css('--surface')).attr('stroke-width', 3).attr('paint-order', 'stroke').text(o.t));
    };
    S.labelTypes = (X) => S.labels(X, true);
    return S;
  }
}

// ---------------------------------------------------------------------------------------------
/** Chain of boxes: [{t, l:[lines]}]. */
function pipelineSvg(boxes) {
  const wrap = el('div', { class: 'figure' });
  wrap.append(el('div', { class: 'fig-title' }, 'From counts to labelled cells'));
  const BW = 124, gap = 15, H = 108;
  const svg = d3.select(wrap).append('svg').attr('viewBox', `0 0 ${boxes.length * (BW + gap)} ${H + 16}`).attr('width', 820).attr('role', 'img').attr('aria-label', 'single-cell analysis stages');
  boxes.forEach((b, i) => {
    const x0 = 4 + i * (BW + gap), g = svg.append('g').attr('transform', `translate(${x0},8)`);
    g.append('rect').attr('width', BW).attr('height', H).attr('rx', 10).attr('fill', css('--surface-2')).attr('stroke', css('--axis'));
    g.append('text').attr('x', 10).attr('y', 24).attr('font-weight', 700).attr('font-size', 13).attr('fill', ink()).text(b.t);
    b.l.forEach((s, j) => g.append('text').attr('x', 10).attr('y', 50 + j * 20).attr('font-size', 12).attr('fill', ink2()).text(s));
    if (i < boxes.length - 1) svg.append('path').attr('d', `M${x0 + BW + 2},${8 + H / 2} l${gap - 8},0 m-5,-5 l5,5 l-5,5`).attr('fill', 'none').attr('stroke', ink2()).attr('stroke-width', 1.8);
  });
  return wrap;
}

/** True type (rows, 8 + doublet) against assigned label (columns, 8 + unassigned) as a row-normalised heatmap. */
function confusionSvg(parent) {
  const cell = 34, left = 96, top = 70, nR = 9, nC = 9;
  const svg = d3.select(parent).append('svg').attr('viewBox', `0 0 ${left + nC * cell + 44} ${top + nR * cell + 26}`).attr('width', left + nC * cell + 44).attr('role', 'img').attr('aria-label', 'confusion matrix');
  const g = svg.append('g').attr('transform', `translate(${left},${top})`);
  svg.append('text').attr('x', left + (nC * cell) / 2).attr('y', 12).attr('text-anchor', 'middle').attr('font-size', 11.5).attr('fill', muted()).text('assigned label');
  svg.append('text').attr('transform', `translate(12,${top + (nR * cell) / 2}) rotate(-90)`).attr('text-anchor', 'middle').attr('font-size', 11.5).attr('fill', muted()).text('true type');
  return {
    update(M, CT) {
      g.selectAll('*').remove(); svg.selectAll('.lab').remove();
      const rows = CT.concat(['doublet']), cols = CT.concat(['unassigned']);
      const col = sequential(0, 1);
      M.forEach((r, i) => { const tot = sum(r) || 1; r.forEach((v, j) => {
        g.append('rect').attr('x', j * cell).attr('y', i * cell).attr('width', cell - 1.5).attr('height', cell - 1.5).attr('rx', 3).attr('fill', col(v / tot)).attr('stroke', css('--border'))
          .on('mousemove', (e) => tip(e, `${rows[i]} → ${cols[j]}<br>${v} of ${tot} cells (${Math.round(100 * v / tot)}%)`)).on('mouseleave', untip);
        if (v) g.append('text').attr('x', j * cell + cell / 2).attr('y', i * cell + cell / 2 + 3).attr('text-anchor', 'middle').attr('font-size', 10).attr('fill', ink()).attr('pointer-events', 'none').text(v);
      }); });
      rows.forEach((t, i) => svg.append('text').attr('class', 'lab').attr('x', left - 6).attr('y', top + i * cell + cell / 2 + 4).attr('text-anchor', 'end').attr('font-size', 11).attr('fill', ink2()).text(t));
      cols.forEach((t, j) => svg.append('text').attr('class', 'lab').attr('transform', `translate(${left + j * cell + cell / 2 + 4},${top - 6}) rotate(-45)`).attr('font-size', 11).attr('fill', ink2()).text(t));
    },
  };
}
