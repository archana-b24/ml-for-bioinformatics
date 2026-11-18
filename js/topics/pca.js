// Principal component analysis, lesson format.
// Arc: what → why → where → how it works (6 steps: two features → one axis; standardise; eigenvectors by power iteration;
//      variance explained / scree; loadings and the immune axis; lab: PCs as inputs + batch effects) → pitfalls → recap.
// Every number in the prose is computed from data/study/cohort.json at render time.
import { header, figure, slider, select, button, checkbox, stats, legend, seeAlso, animate, tip, untip, el, html } from '../lib/ui.js';
import { frame, heatmap, cat, css, ink, ink2, muted, diverging } from '../lib/plot.js';
import { rng, mean, sd } from '../lib/data.js';
import { pca, fitLogistic, predictProb, auc } from '../lib/models.js';
import { loadCohort, FEATURES, studyCard } from '../lib/study.js';
import { lessonNav, chapter, prose, rung, worked, quiz, cards, table, code, recap } from '../lib/lesson.js';
const d3 = window.d3;

const f2 = (v) => (Math.abs(v) < 0.005 ? '0.00' : v.toFixed(2));
const f1 = (v) => v.toFixed(1);
const pc = (v) => Math.round(100 * v) + '%';
const sum = (a) => a.reduce((s, v) => s + v, 0);
const PF = ['ifng', 'cd8', 'pdl1', 'caf', 'log2tmb', 'prolif', 'age'];
const SH = { ifng: 'IFN-γ', cd8: 'CD8 %', pdl1: 'PD-L1', caf: 'CAF', log2tmb: 'log2 TMB', prolif: 'Proliferation', age: 'Age' };
const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
const norm = (a) => Math.sqrt(dot(a, a));
const corr = (a, b) => { const ma = mean(a), mb = mean(b); let s = 0, x = 0, z = 0; a.forEach((v, i) => { s += (v - ma) * (b[i] - mb); x += (v - ma) ** 2; z += (b[i] - mb) ** 2; }); return s / Math.sqrt(x * z); };

/** Horizontal bar chart (rows: {lab, v, ref?, tip?}); redraws into host each call. */
function hbars(host, rows, { w = 460, h = 250, dom = [0, 1], xLabel = '', fmt = f2, color = () => cat(0) } = {}) {
  const m = { l: 92, r: 42, t: 8, b: 38 }, W = w - m.l - m.r, H = h - m.t - m.b;
  const svg = d3.select(host).selectAll('svg').data([0]).join('svg').attr('viewBox', `0 0 ${w} ${h}`).attr('width', w).attr('role', 'img');
  svg.selectAll('*').remove();
  const g = svg.append('g').attr('transform', `translate(${m.l},${m.t})`);
  const x = d3.scaleLinear().domain(dom).range([0, W]), y = d3.scaleBand().domain(rows.map((r) => r.lab)).range([0, H]).padding(0.28);
  g.append('g').attr('class', 'axis x').attr('transform', `translate(0,${H})`).call(d3.axisBottom(x).ticks(5));
  g.append('text').attr('class', 'ax-label').attr('x', W / 2).attr('y', H + 34).attr('text-anchor', 'middle').text(xLabel);
  g.append('line').attr('x1', x(0)).attr('x2', x(0)).attr('y1', 0).attr('y2', H).attr('stroke', muted()).attr('stroke-dasharray', '3 3');
  rows.forEach((r) => {
    const cy = y(r.lab) + y.bandwidth() / 2, v = Math.max(dom[0], Math.min(dom[1], r.v));
    g.append('text').attr('x', -8).attr('y', cy + 4).attr('text-anchor', 'end').attr('font-size', 12).attr('fill', ink2()).text(r.lab);
    g.append('rect').attr('x', x(Math.min(0, v))).attr('y', y(r.lab)).attr('width', Math.abs(x(v) - x(0))).attr('height', y.bandwidth()).attr('fill', color(r)).attr('rx', 2)
      .on('mousemove', (e) => tip(e, r.tip || `${r.lab}: ${fmt(r.v)}`)).on('mouseleave', untip);
    g.append('text').attr('x', x(v) + (r.v >= 0 ? 4 : -4)).attr('text-anchor', r.v >= 0 ? 'start' : 'end').attr('y', cy + 4).attr('font-size', 11.5).attr('fill', ink()).text(fmt(r.v));
    if (r.ref != null) g.append('path').attr('d', d3.symbol(d3.symbolDiamond, 60)).attr('transform', `translate(${x(Math.max(dom[0], Math.min(dom[1], r.ref)))},${cy})`).attr('fill', 'none').attr('stroke', ink()).attr('stroke-width', 1.6);
  });
  return svg;
}
const arrowHead = (g, x, y, ang, color) => g.append('path').attr('d', d3.symbol(d3.symbolTriangle, 70)).attr('transform', `translate(${x},${y}) rotate(${ang + 90})`).attr('fill', color);

export default async function render(main) {
  const C = await loadCohort();
  const D = C.discovery, V = C.validation, n = D.length;
  const y = D.map((p) => p.response), yv = V.map((p) => p.response);
  const X = (pts, keys) => pts.map((p) => keys.map((k) => FEATURES[k].get(p)));
  const PA = pca(X(D, PF)), PR = pca(X(D, PF), { scale: false }), R = PA.cov;
  const col = (M, j) => M.map((r) => r[j]);
  const iTMB = PF.indexOf('log2tmb'), iF = PF.indexOf('ifng');
  const tmbPC = d3.range(PF.length).sort((a, b) => Math.abs(PA.vecs[b][iTMB]) - Math.abs(PA.vecs[a][iTMB]))[0];
  const sc = (pts, P, m) => pts.map((p) => P.project(X([p], PF)[0])[m]);
  const aucPC = (m) => { const a = auc(sc(V, PA, m), yv); return Math.max(a, 1 - a); };
  const jit = (i) => (((i * 7919) % 101) / 101 - 0.5);
  const P2 = pca(X(D, ['ifng', 'cd8'])), r2 = P2.cov[0][1];
  const block = [0, 1, 2, 3], rest = [4, 5, 6];
  const inBlock = []; block.forEach((a) => block.forEach((b) => { if (a < b) inBlock.push(Math.abs(R[a][b])); }));
  const outBlock = []; block.forEach((a) => rest.forEach((b) => outBlock.push(Math.abs(R[a][b])))); rest.forEach((a) => rest.forEach((b) => { if (a < b) outBlock.push(Math.abs(R[a][b])); }));
  const ph = (p) => p.phenotype;

  header(main, 'Principal component analysis',
    'Many measurements on the same samples are often redundant. <b>Principal component analysis (PCA) replaces them with a few new axes, each a weighted combination of the original measurements, ordered by how much of the variation in the data they capture.</b>');
  lessonNav(main);
  studyCard(main, C, '<b>Question for this page:</b> the biopsy gives several correlated measurements per patient. How many independent axes of variation do they really contain, and what does the main axis represent?');

  // ===================================================================================== 1. WHAT
  let ch = chapter(main, 'what', 'What is PCA?', 'In one line: <b>rotate the axes so that the first one points along the direction of greatest spread</b>.');
  prose(ch, `<p>PCA is an <b>unsupervised</b> method: it uses only the measurements, never the outcome. Starting from <i>p</i> numeric features, it constructs <i>p</i> new features, the <b>principal components</b> (PCs). Each PC is a weighted sum of the original features. PC1 is the direction along which patients differ most, PC2 is the direction of greatest remaining spread at right angles to PC1, and so on. Keeping the first few PCs gives a short summary of the data that loses as little variation as possible.</p>
    <p>The figure uses two features from the study, IFN-γ score and CD8 T-cell fraction, each expressed in standard deviations (SD) from its mean. They are correlated (r = <b>${f2(r2)}</b>): a tumour with high IFN-γ usually has many CD8 T cells. The left panel shows the patients with the two PC axes; the right panel shows the same patients re-plotted using the PCs as axes.</p>`);
  const fw = figure(ch, 'The same patients on original axes and on principal-component axes',
    'Left: arrows are the PC directions through the centre of the cloud, with length equal to 2 standard deviations along that direction. Right: each patient\'s coordinates along PC1 and PC2 (its <b>scores</b>). The cloud is rotated, not changed. Hover a point for its values.');
  const rw = el('div', { class: 'fig-row' }); fw.body.append(rw);
  const wa = el('div'), wb = el('div'); rw.append(wa, wb);
  const ext2 = Math.ceil(d3.max(P2.Z.flat(), Math.abs));
  const FA = frame(wa, { width: 460, height: 400, x: [-ext2, ext2], y: [-ext2, ext2], xLabel: 'IFN-γ (SD from mean)', yLabel: 'CD8 T cells (SD from mean)' });
  const FB = frame(wb, { width: 460, height: 400, x: [-ext2, ext2], y: [-ext2, ext2], xLabel: `PC1 score (${pc(P2.ratio[0])} of variance)`, yLabel: `PC2 score (${pc(P2.ratio[1])})` });
  FA.layer('pts').selectAll('circle').data(D).join('circle').attr('cx', (d, i) => FA.x(P2.Z[i][0])).attr('cy', (d, i) => FA.y(P2.Z[i][1])).attr('r', 3.8).attr('fill', cat(2)).attr('fill-opacity', 0.65).attr('stroke', css('--surface'))
    .on('mousemove', (e, d) => tip(e, `${d.id}<br>IFN-γ ${d.ifng}, CD8 ${d.cd8}%`)).on('mouseleave', untip);
  FB.layer('pts').selectAll('circle').data(D).join('circle').attr('cx', (d, i) => FB.x(P2.S[i][0])).attr('cy', (d, i) => FB.y(P2.S[i][1])).attr('r', 3.8).attr('fill', cat(2)).attr('fill-opacity', 0.65).attr('stroke', css('--surface'))
    .on('mousemove', (e, d, i) => { const k = D.indexOf(d); tip(e, `${d.id}<br>PC1 ${f2(P2.S[k][0])}, PC2 ${f2(P2.S[k][1])}`); }).on('mouseleave', untip);
  [0, 1].forEach((m) => {
    const v = P2.vecs[m], L = 2 * Math.sqrt(P2.vals[m]), x1 = FA.x(v[0] * L), y1 = FA.y(v[1] * L);
    FA.layer('ax').append('line').attr('x1', FA.x(0)).attr('y1', FA.y(0)).attr('x2', x1).attr('y2', y1).attr('stroke', ink()).attr('stroke-width', 2.4).attr('stroke-dasharray', m ? '5 4' : null);
    arrowHead(FA.layer('ax'), x1, y1, Math.atan2(y1 - FA.y(0), x1 - FA.x(0)) * 180 / Math.PI, ink());
    FA.layer('ax').append('text').attr('x', x1 + (m ? 6 : -4)).attr('y', y1 + (m ? 14 : -8)).attr('text-anchor', m ? 'start' : 'end').attr('font-size', 12.5).attr('font-weight', 700).attr('fill', ink()).attr('stroke', css('--surface')).attr('stroke-width', 3).attr('paint-order', 'stroke').text(`PC${m + 1}`);
  });
  FA.layer('ax').append('text').attr('x', 8).attr('y', 16).attr('font-size', 11.5).attr('fill', ink2()).text('PC1: both high or both low');
  FA.layer('ax').append('text').attr('x', 8).attr('y', 31).attr('font-size', 11.5).attr('fill', ink2()).text('PC2: one high, the other low');
  FB.layer('ax').append('text').attr('x', FB.w - 8).attr('y', FB.y(-2.6)).attr('text-anchor', 'end').attr('font-size', 11.5).attr('fill', ink2()).text('wide spread: most of the variance');
  FB.layer('ax').append('text').attr('x', FB.x(0) + 6).attr('y', 14).attr('font-size', 11.5).attr('fill', ink2()).text('narrow spread');
  const topLoad = [...PA.vecs[0].keys()].sort((a, b) => Math.abs(PA.vecs[0][b]) - Math.abs(PA.vecs[0][a])).slice(0, 3);
  prose(ch, '<p>Key terms:</p>');
  table(ch, [
    { t: 'Standardising', m: 'Subtracting each feature\'s mean and dividing by its SD, so every feature has mean 0 and SD 1 and unit differences no longer matter', s: `IFN-γ: mean ${f2(mean(D.map((p) => p.ifng)))}, SD ${f2(sd(D.map((p) => p.ifng)))}; PD-L1: mean ${f1(mean(D.map((p) => p.pdl1)))}%, SD ${f1(sd(D.map((p) => p.pdl1)))}%` },
    { t: 'Principal component (PC)', m: 'A new axis: a weighted sum of the standardised features. PCs are mutually uncorrelated and ordered by variance', s: `${PF.length} features give ${PF.length} PCs; PC1 uses mainly ${topLoad.map((k) => SH[PF[k]]).join(', ')}` },
    { t: 'Loading', m: 'The weight of one original feature in one PC (a unit-length vector per PC)', s: `PC1 loading of IFN-γ = ${f2(PA.vecs[0][iF])}` },
    { t: 'Score', m: 'A sample\'s coordinate on a PC: the weighted sum of its standardised features', s: `${D[0].id}: PC1 score ${f2(PA.S[0][0])}, PC2 score ${f2(PA.S[0][1])}` },
    { t: 'Eigenvalue (variance)', m: 'The variance of the scores on one PC. With standardised features the eigenvalues add up to the number of features', s: `PC1 = ${f2(PA.vals[0])}, of a total of ${f2(sum(PA.vals))}` },
    { t: 'Variance explained', m: 'A PC\'s eigenvalue divided by the total. A scree plot shows it for every PC', s: `PC1 explains ${pc(PA.ratio[0])}, PC1 + PC2 ${pc(PA.ratio[0] + PA.ratio[1])}` },
  ], [['t', 'Term'], ['m', 'Meaning'], ['s', 'In our study']]);

  // ===================================================================================== 2. WHY
  ch = chapter(main, 'why', 'Why do we need it?', 'What goes wrong when many correlated measurements are used directly, and what a summary should provide.');
  prose(ch, `<p>Biological measurements rarely vary independently. The heatmap shows the correlations between the seven numeric features in the cohort. Four of them (IFN-γ, CD8 %, PD-L1 and CAF score) form a correlated block, with |r| between ${f2(d3.min(inBlock))} and ${f2(d3.max(inBlock))} inside the block. TMB, proliferation and age are almost unrelated to the block and to each other (all other |r| at most ${f2(d3.max(outBlock))}). The block reflects one underlying biological state, the degree of immune activity in the tumour, which each assay measures imperfectly.</p>`);
  const fh = figure(ch, 'Correlation between the seven features (discovery patients)', 'Red: positive correlation; blue: negative. Hover a cell for the value. The 4 × 4 block at the top left is the immune-related group.');
  heatmap(fh.body, R, { cell: 56, color: diverging(1), rowLabels: PF.map((k) => SH[k]), colLabels: PF.map((k) => SH[k]), labelW: 100, labelH: 90, fmt: (v) => f2(v), onHover: (e, i, j, v) => tip(e, `${SH[PF[i]]} vs ${SH[PF[j]]}: r = ${f2(v)}`) });
  prose(ch, `<p>A summary of the data should therefore:</p><ol>
    <li><b>Remove redundancy.</b> Four correlated features carry roughly one feature's worth of information. Putting all four into a regression gives unstable, hard-to-interpret weights (see the <a href="#/concepts/logistic-regression">logistic regression</a> page).</li>
    <li><b>Allow the data to be drawn.</b> Seven dimensions cannot be plotted, but the first two PCs can, and here they hold ${pc(PA.ratio[0] + PA.ratio[1])} of the variance.</li>
    <li><b>Say how many independent signals there are</b>, and how strongly each feature contributes to each signal.</li></ol>
    <p>Common alternatives each meet only part of this.</p>`);
  table(ch, [
    { a: 'Keep one feature, drop the rest', r: 'yes', d: 'no (information discarded)', i: 'simple but arbitrary which one', o: 'no' },
    { a: 'Average the correlated features by hand', r: 'partly', d: 'one new axis', i: 'good if the grouping is known in advance', o: 'no' },
    { a: '<b>PCA</b>', r: '<b>yes, PCs are uncorrelated</b>', d: '<b>any number of axes, ordered</b>', i: '<b>loadings show contributions</b>', o: '<b>no (unsupervised)</b>' },
    { a: 'Supervised selection (lasso, PLS)', r: 'partly', d: 'chosen for the outcome', i: 'moderate', o: 'yes' },
    { a: 't-SNE / UMAP', r: 'not applicable', d: 'two axes for display', i: 'axes have no meaning', o: 'no' },
  ], [['a', 'Approach'], ['r', 'Removes redundancy?'], ['d', 'Compresses?'], ['i', 'Interpretable?'], ['o', 'Uses the outcome?']]);

  // ===================================================================================== 3. WHERE
  ch = chapter(main, 'where', 'Where is it used?', 'Any setting with many correlated measurements per sample, where structure, quality or a compact input is needed.');
  cards(ch, [
    { tag: 'Bulk RNA-seq QC', title: 'Do the samples group as expected?', body: 'Samples are plotted on PC1 and PC2 of log-normalised expression (e.g. DESeq2 <code>plotPCA</code> on variance-stabilised counts). Samples should separate by condition, not by sequencing batch, and an outlier is visible at once.' },
    { tag: 'Batch effects', title: 'Is a technical factor driving the data?', body: 'Colour the PC plot by batch, centre, operator or RNA quality. A PC that tracks a technical variable is evidence of a batch effect, to be modelled or corrected (<code>limma::removeBatchEffect</code>, ComBat, <code>sva</code>).' },
    { tag: 'Single-cell', title: 'The input to clustering and UMAP', body: 'Scanpy and Seurat run PCA on the highly variable genes and use the first 20–50 PCs to build the neighbour graph. This removes noise and reduces thousands of genes to a few dozen numbers per cell.' },
    { tag: 'Population genetics (GWAS)', title: 'Ancestry and population structure', body: 'PCA of the genotype matrix (EIGENSOFT <code>smartpca</code>, PLINK <code>--pca</code>) reproduces geography in the top PCs. These PCs are added as covariates so that ancestry differences are not mistaken for disease associations.' },
    { tag: 'Exploration', title: 'Proteomics, metabolomics, flow cytometry', body: 'Any samples × features table can be inspected on its first PCs: groups, gradients, outliers and replicates that do not agree. PCA is usually the first plot made on a new dataset.' },
    { tag: 'Modelling', title: 'Principal-components regression', body: 'The first few PCs replace many collinear predictors in a regression or classifier. This stabilises the fit; it works only if the outcome-relevant signal lies on high-variance PCs (see the lab in chapter 4).' },
    { tag: 'Imaging', title: 'Shape and morphology', body: 'Cell-painting profiles, nuclear shape descriptors and shape models of anatomy are summarised by a few PCs, each describing a mode of variation such as elongation or size.' },
    { tag: 'Outside biology', title: 'Faces, finance, climate', body: 'Eigenfaces for image recognition, common factors in asset returns and dominant patterns in climate fields all use the same decomposition.' },
  ]);
  prose(ch, `<p><b>When not to use it:</b> the structure is non-linear (curved manifolds, clusters separated by a threshold; consider UMAP or a neural-network embedding, usually applied after PCA); the goal is to predict an outcome and the relevant signal has low variance (use a supervised model; PCA ignores the label); the original features must stay interpretable (use feature selection); or the data are raw, skewed counts (transform first, for example log or variance-stabilising). PCA is also sensitive to outliers, because it maximises squared deviations.</p>`);

  // ===================================================================================== 4. HOW
  ch = chapter(main, 'how', 'How it works', 'Six steps, from two features and one axis to all seven features, loadings, and a model built on the PCs.');

  // ---- Step 1: two features -> one axis
  let rb = rung(ch, 1, 'Two features, one axis',
    `<p>Take IFN-γ and CD8 % (standardised). Any straight line through the centre of the cloud defines a new axis. Projecting every patient onto it gives one number per patient, and the <b>variance</b> of those numbers measures how much of the original variation that axis keeps. The first PC is the line that keeps the most. Rotate the axis to find it, or let the sweep run.</p>`);
  const f1g = figure(rb, 'Rotating a projection axis', 'Left: patients and a candidate axis; thin segments show each patient projected onto it. Right: variance along the axis for every angle; dashed line = variance of one original feature (1.00, because the data are standardised).');
  let th = 20, stop1 = null;
  const th1 = slider(f1g.controls, { label: 'axis angle (°)', min: 0, max: 180, step: 1, value: th, fmt: (v) => v + '°', onInput: (v) => { halt1(); th = v; paint1(); } });
  button(f1g.controls, 'Snap to PC1', () => { halt1(); th = thStar; th1.set(th); paint1(); }, true);
  button(f1g.controls, 'Sweep 0 → 180°', () => { halt1(); th = 0; let t = 0; stop1 = animate((dt) => { t += dt; th = Math.min(180, t / 25); th1.set(Math.round(th)); paint1(); return th < 180; }); });
  const s1 = stats(f1g.root);
  const r1 = el('div', { class: 'fig-row' }); f1g.body.append(r1);
  const g1a = el('div'), g1b = el('div'); r1.append(g1a, g1b);
  const Z2 = P2.Z, varAt = (deg) => { const a = deg * Math.PI / 180, u = [Math.cos(a), Math.sin(a)]; return sum(Z2.map((z) => dot(z, u) ** 2)) / (n - 1); };
  const thStar = ((Math.atan2(P2.vecs[0][1], P2.vecs[0][0]) * 180 / Math.PI) + 180) % 180;
  const F1a = frame(g1a, { width: 460, height: 400, x: [-ext2, ext2], y: [-ext2, ext2], xLabel: 'IFN-γ (SD from mean)', yLabel: 'CD8 T cells (SD from mean)' });
  const F1b = frame(g1b, { width: 460, height: 400, x: [0, 180], y: [0, 2.2], xLabel: 'axis angle (°)', yLabel: 'variance of projected patients', xTicks: 6 });
  F1a.layer('pts').selectAll('circle').data(D).join('circle').attr('cx', (d, i) => F1a.x(Z2[i][0])).attr('cy', (d, i) => F1a.y(Z2[i][1])).attr('r', 3.4).attr('fill', cat(2)).attr('fill-opacity', 0.6)
    .on('mousemove', (e, d) => tip(e, `${d.id}<br>IFN-γ ${d.ifng}, CD8 ${d.cd8}%`)).on('mouseleave', untip);
  const seg = F1a.layer('seg'), axl = F1a.layer('axis');
  F1b.layer('c').append('path').datum(d3.range(0, 180.1, 2)).attr('fill', 'none').attr('stroke', cat(6)).attr('stroke-width', 2.2).attr('d', d3.line().x((v) => F1b.x(v)).y((v) => F1b.y(varAt(v))));
  [[1, 'one original feature: 1.00'], [2, 'total variance: 2.00']].forEach(([v, t]) => { F1b.layer('c').append('line').attr('x1', 0).attr('x2', F1b.w).attr('y1', F1b.y(v)).attr('y2', F1b.y(v)).attr('stroke', muted()).attr('stroke-dasharray', '4 4'); F1b.layer('c').append('text').attr('x', F1b.w - 6).attr('y', F1b.y(v) - 5).attr('text-anchor', 'end').attr('font-size', 11.5).attr('fill', ink2()).text(t); });
  F1b.layer('c').append('path').attr('d', d3.symbol(d3.symbolDiamond, 110)).attr('transform', `translate(${F1b.x(thStar)},${F1b.y(varAt(thStar))})`).attr('fill', 'none').attr('stroke', ink()).attr('stroke-width', 2);
  F1b.layer('c').append('text').attr('x', F1b.x(thStar) + 12).attr('y', F1b.y(varAt(thStar)) - 8).attr('font-size', 12).attr('font-weight', 700).attr('fill', ink()).text(`PC1 (${f1(thStar)}°): ${f2(varAt(thStar))}`);
  const thMin = (thStar + 90) % 180;
  F1b.layer('c').append('text').attr('x', F1b.x(thMin) - 8).attr('y', F1b.y(varAt(thMin)) + 20).attr('text-anchor', 'end').attr('font-size', 12).attr('font-weight', 700).attr('fill', ink()).text(`PC2 (${f1(thMin)}°): ${f2(varAt(thMin))}`);
  const cur1 = F1b.layer('cur');
  function halt1() { if (stop1) { stop1(); stop1 = null; } }
  function paint1() {
    const a = th * Math.PI / 180, u = [Math.cos(a), Math.sin(a)], L = ext2 * 1.4;
    seg.selectAll('line').data(Z2).join('line').attr('x1', (z) => F1a.x(z[0])).attr('y1', (z) => F1a.y(z[1])).attr('x2', (z) => F1a.x(dot(z, u) * u[0])).attr('y2', (z) => F1a.y(dot(z, u) * u[1])).attr('stroke', muted()).attr('stroke-opacity', 0.45);
    axl.selectAll('line').data([0]).join('line').attr('x1', F1a.x(-u[0] * L)).attr('y1', F1a.y(-u[1] * L)).attr('x2', F1a.x(u[0] * L)).attr('y2', F1a.y(u[1] * L)).attr('stroke', cat(6)).attr('stroke-width', 3);
    const v = varAt(th);
    cur1.selectAll('circle').data([0]).join('circle').attr('cx', F1b.x(th)).attr('cy', F1b.y(v)).attr('r', 6).attr('fill', cat(6)).attr('stroke', css('--surface')).attr('stroke-width', 2);
    s1.set({ 'variance along the axis': f2(v), 'share of total variance': pc(v / 2), 'variance left in the perpendicular direction': f2(2 - v), 'angle of PC1': f1(thStar) + '°' });
  }
  paint1();
  prose(rb, `<p>The variance along the axis is largest at ${f1(thStar)}°, where it reaches ${f2(P2.vals[0])} (${pc(P2.ratio[0])} of the total ${f2(sum(P2.vals))}), and smallest at the perpendicular angle (${f2(P2.vals[1])}). The two variances always add up to 2.00 because the total does not change when the axes rotate. For two standardised features with correlation r, the eigenvalues are exactly 1 + r and 1 − r, here ${f2(1 + r2)} and ${f2(1 - r2)}, and PC1 is at 45° when r is positive: it gives equal weight to both features.</p>`);

  // ---- Step 2: standardise first
  rb = rung(ch, 2, 'Seven features: standardise first',
    `<p>With all seven features the same idea applies, but the features are measured in different units: PD-L1 in percent (0–100), IFN-γ in log2 expression units, age in years. PCA maximises variance, so a feature measured on a large scale dominates simply because its numbers are large. Dividing each feature by its SD (<b>standardising</b>) removes this, and then the analysis is based on the correlation matrix.</p>`);
  const f2g = figure(rb, 'Raw units vs standardised features', 'Left: the share of the total variance that each feature contributes before PCA. Right: the loadings of PC1. Switch between scalings to compare.');
  let sc2 = 'raw';
  select(f2g.controls, { label: 'Scaling', value: sc2, options: [['raw', 'Centred only (raw units)'], ['std', 'Standardised (SD = 1)']], onChange: (v) => { sc2 = v; paint2(); } });
  const s2 = stats(f2g.root);
  const r2g = el('div', { class: 'fig-row' }); f2g.body.append(r2g);
  const g2a = el('div'), g2b = el('div'); r2g.append(g2a, g2b);
  function paint2() {
    const P = sc2 === 'raw' ? PR : PA, vars = PF.map((k, j) => (sc2 === 'raw' ? P.cov[j][j] : 1)), tot = sum(vars);
    hbars(g2a, PF.map((k, j) => ({ lab: SH[k], v: vars[j] / tot, tip: `${SH[k]}: variance ${f2(vars[j])} (${pc(vars[j] / tot)} of total)` })), { dom: [0, 1], xLabel: 'share of total variance before PCA', fmt: pc });
    hbars(g2b, PF.map((k, j) => ({ lab: SH[k], v: P.vecs[0][j], tip: `${SH[k]}: PC1 loading ${f2(P.vecs[0][j])}` })), { dom: [-1, 1], xLabel: `PC1 loading (PC1 explains ${pc(P.ratio[0])})`, color: (r) => cat(r.v >= 0 ? 1 : 0) });
    s2.set({ 'PC1 variance explained': pc(P.ratio[0]), 'largest |PC1 loading|': f2(d3.max(P.vecs[0], Math.abs)), 'features with |PC1 loading| > 0.3': P.vecs[0].filter((v) => Math.abs(v) > 0.3).length + ' of ' + PF.length });
  }
  paint2();
  const rawTop = d3.range(PF.length).sort((a, b) => PR.cov[b][b] - PR.cov[a][a])[0];
  prose(rb, `<p>On the raw scale, ${SH[PF[rawTop]]} alone contributes ${pc(PR.cov[rawTop][rawTop] / sum(PR.cov.map((r, j) => r[j])))} of the total variance, and PC1 (${pc(PR.ratio[0])} of variance) is almost entirely this one feature (loading ${f2(PR.vecs[0][rawTop])}). It says nothing about immune activity. After standardising, PC1 spreads its weight over the four immune-related features. Standardise unless all features share one meaningful unit (for example log-expression of genes, where differences in variance are informative).</p>`);

  // ---- Step 3: eigenvectors by power iteration
  rb = rung(ch, 3, 'Finding the axes: eigenvectors of the correlation matrix',
    `<p>The best axis is an <b>eigenvector</b> of the correlation matrix <b>R</b>: a direction that R only stretches and does not turn. Its <b>eigenvalue</b> is the stretch factor, and equals the variance along that direction. A simple way to find the top eigenvector is <b>power iteration</b>: start from any direction, multiply by R, rescale to length 1, and repeat. Directions with large stretch factors win, so the vector turns towards PC1.</p>`);
  const sig = [P2.vals[0], P2.vals[1]];
  worked(rb, 'Two features by hand', `<p>For IFN-γ and CD8 the matrix is R = [[1, ${f2(r2)}], [${f2(r2)}, 1]].</p><ul>
    <li>The vector (1, 1) maps to (1 + ${f2(r2)}, ${f2(r2)} + 1) = ${f2(1 + r2)} × (1, 1). Same direction, stretched by <b>${f2(sig[0])}</b>: PC1.</li>
    <li>The vector (1, −1) maps to (1 − ${f2(r2)}, ${f2(r2)} − 1) = ${f2(1 - r2)} × (1, −1). Same direction, stretched by <b>${f2(sig[1])}</b>: PC2.</li>
    <li>Both vectors are rescaled to length 1: (0.71, 0.71) and (0.71, −0.71). The fitted loadings are (${f2(P2.vecs[0][0])}, ${f2(P2.vecs[0][1])}) and (${f2(P2.vecs[1][0])}, ${f2(P2.vecs[1][1])}). The sign of a whole PC is arbitrary: a vector and its negative describe the same axis, so software may report either.</li></ul>`);
  const f3 = figure(rb, 'Power iteration on the seven-feature correlation matrix', 'Left: the current vector (bars) against the true PC1 loadings (open diamonds). Right: angle between the two on a log scale. Each step multiplies by R and rescales; the angle shrinks by about λ2/λ1 per step.');
  const v0 = (() => { const g = rng(5); const v = PF.map(() => g.normal(0, 1)); const s = norm(v); const u = v.map((q) => q / s); return dot(u, PA.vecs[0]) < 0 ? u.map((q) => -q) : u; })();
  let vec3 = v0.slice(), hist3 = [], stop3 = null;
  const angTo = (v) => Math.acos(Math.min(1, Math.abs(dot(v, PA.vecs[0])))) * 180 / Math.PI;
  const resetP = () => { vec3 = v0.slice(); hist3 = [angTo(vec3)]; };
  const stepP = () => { if (hist3.length > 20) return false; const w = R.map((r) => dot(r, vec3)), s = norm(w); vec3 = w.map((q) => q / s); hist3.push(angTo(vec3)); return hist3.length <= 20; };
  const haltP = () => { if (stop3) { stop3(); stop3 = null; } };
  button(f3.controls, 'Step', () => { haltP(); stepP(); paint3(); }, true);
  button(f3.controls, 'Run', () => { haltP(); let t = 0; stop3 = animate((dt) => { t += dt; if (t > 350) { t = 0; const more = stepP(); paint3(); return more; } return true; }); });
  button(f3.controls, 'Reset', () => { haltP(); resetP(); paint3(); });
  const s3 = stats(f3.root);
  const r3 = el('div', { class: 'fig-row' }); f3.body.append(r3);
  const g3a = el('div'), g3b = el('div'); r3.append(g3a, g3b);
  const F3 = frame(g3b, { width: 460, height: 250, x: [0, 20], y: [0.01, 100], yScale: d3.scaleLog(), xLabel: 'iteration', yLabel: 'angle to PC1 (°)', margin: { left: 56 }, grid: false, axes: false });
  F3.g.append('g').attr('class', 'axis x').attr('transform', `translate(0,${F3.h})`).call(d3.axisBottom(F3.x).ticks(5));
  F3.g.append('g').attr('class', 'axis y').call(d3.axisLeft(F3.y).tickValues([0.01, 0.1, 1, 10, 100]).tickFormat(d3.format('~g')));
  F3.g.append('g').attr('class', 'gridline').call(d3.axisLeft(F3.y).tickValues([0.01, 0.1, 1, 10, 100]).tickSize(-F3.w).tickFormat(''));
  const pth3 = F3.layer('p'), tgt3 = F3.layer('t');
  F3.layer('f').append('text').attr('x', F3.w - 6).attr('y', 14).attr('text-anchor', 'end').attr('font-size', 11.5).attr('fill', ink2()).text(`expected shrink per step: λ2/λ1 = ${f2(PA.vals[1] / PA.vals[0])}`);
  function paint3() {
    hbars(g3a, PF.map((k, j) => ({ lab: SH[k], v: vec3[j], ref: PA.vecs[0][j], tip: `${SH[k]}: current ${f2(vec3[j])}, PC1 loading ${f2(PA.vecs[0][j])}` })), { dom: [-1, 1], xLabel: 'vector component (diamond = PC1 loading)', h: 250, color: (r) => cat(r.v >= 0 ? 1 : 0) });
    const pts = hist3.map((a, i) => [i, Math.max(0.01, a)]);
    pth3.selectAll('path').data([pts]).join('path').attr('fill', 'none').attr('stroke', cat(6)).attr('stroke-width', 2.2).attr('d', d3.line().x((q) => F3.x(q[0])).y((q) => F3.y(q[1])));
    tgt3.selectAll('circle').data(pts).join('circle').attr('cx', (q) => F3.x(q[0])).attr('cy', (q) => F3.y(q[1])).attr('r', 4).attr('fill', cat(6)).attr('stroke', css('--surface'))
      .on('mousemove', (e, q) => tip(e, `iteration ${q[0]}: ${q[1] <= 0.0101 ? '< 0.01' : f2(q[1])}°`)).on('mouseleave', untip);
    const Rv = R.map((r) => dot(r, vec3));
    s3.set({ iteration: hist3.length - 1, 'angle to PC1': (hist3[hist3.length - 1] < 0.01 ? '< 0.01' : f2(hist3[hist3.length - 1])) + '°', 'stretch factor vᵀRv': f2(dot(vec3, Rv)), 'true λ1': f2(PA.vals[0]) });
  }
  resetP(); paint3();
  prose(rb, `<p>Starting ${f1(hist3[0])}° away from PC1, the vector is within 1° after about ${(() => { let v = v0.slice(), k = 0; while (angTo(v) > 1 && k < 100) { const w = R.map((r) => dot(r, v)), s = norm(w); v = w.map((q) => q / s); k++; } return k; })()} steps. Convergence is slower when the first two eigenvalues are close, which is also the situation in which the PCs themselves are unstable (chapter 5). To obtain PC2, the PC1 direction is removed from the data and the procedure is repeated. Software uses a singular value decomposition (SVD) or, for matrices with thousands of columns, randomised or truncated versions of it, with the same result.</p>`);

  // ---- Step 4: scree
  rb = rung(ch, 4, 'How many components? Variance explained and the scree plot',
    `<p>All seven PCs together reproduce the data exactly; the aim is to keep only the first <i>k</i>. The <b>scree plot</b> shows the variance of each PC. Three common rules for choosing <i>k</i> are: keep the PCs before the "elbow"; keep PCs with eigenvalue above 1 (the average, <b>Kaiser rule</b>); or keep PCs whose eigenvalue exceeds what <b>shuffled data</b> would give (<b>parallel analysis</b>: each feature is permuted independently, which destroys the correlations, and PCA is repeated). The right panel shows what is lost: one patient's standardised values, and their reconstruction from the first <i>k</i> PCs.</p>`);
  const nullEv = (() => { const g = rng(11), B = 40, all = PF.map(() => []); for (let b = 0; b < B; b++) { const cols = PF.map((k, j) => { const c = col(PA.Z, j).slice(); for (let i = c.length - 1; i > 0; i--) { const q = Math.floor(g() * (i + 1)); [c[i], c[q]] = [c[q], c[i]]; } return c; }); const Pb = pca(PA.Z.map((_, i) => cols.map((c) => c[i]))); Pb.vals.forEach((v, m) => all[m].push(v)); } return all.map((a) => d3.quantile(a.sort(d3.ascending), 0.95)); })();
  const f4 = figure(rb, 'Scree plot, shuffled-data reference and reconstruction', 'Left: variance of each PC (bars; labels give the share of the total). Open diamonds: 95th percentile of the same PC from 40 shuffled copies of the data. Right: for one patient, bars = reconstruction from the first k PCs, diamonds = actual standardised value.');
  const ordP = d3.range(n).sort((a, b) => PA.S[b][0] - PA.S[a][0]);
  const pats = [[ordP[0], 'highest PC1 score'], [ordP[n - 1], 'lowest PC1 score'], [d3.range(n).sort((a, b) => Math.abs(PA.S[b][1]) - Math.abs(PA.S[a][1]))[0], 'largest |PC2 score|'], [ordP[Math.floor(n / 2)], 'median PC1 score']];
  let k4 = 2, pat4 = pats[0][0];
  slider(f4.controls, { label: 'components kept k', min: 1, max: 7, step: 1, value: k4, onInput: (v) => { k4 = v; paint4(); } });
  select(f4.controls, { label: 'Patient', value: String(pat4), options: pats.map(([i, t]) => [String(i), `${D[i].id} (${t})`]), onChange: (v) => { pat4 = +v; paint4(); } });
  const s4 = stats(f4.root);
  const r4 = el('div', { class: 'fig-row' }); f4.body.append(r4);
  const g4a = el('div'), g4b = el('div'); r4.append(g4a, g4b);
  const F4 = frame(g4a, { width: 460, height: 300, x: [0.4, 7.6], y: [0, Math.ceil(PA.vals[0] * 2) / 2 + 0.5], xLabel: 'principal component', yLabel: 'variance (eigenvalue)', xTicks: 7 });
  F4.layer('c').append('line').attr('x1', 0).attr('x2', F4.w).attr('y1', F4.y(1)).attr('y2', F4.y(1)).attr('stroke', muted()).attr('stroke-dasharray', '4 4');
  F4.layer('c').append('text').attr('x', F4.w - 4).attr('y', 14).attr('text-anchor', 'end').attr('font-size', 11.5).attr('fill', ink2()).text('dashed line: average eigenvalue = 1');
  const bars4 = F4.layer('bars');
  F4.layer('null').selectAll('path').data(nullEv).join('path').attr('d', d3.symbol(d3.symbolDiamond, 80)).attr('transform', (v, i) => `translate(${F4.x(i + 1)},${F4.y(v)})`).attr('fill', 'none').attr('stroke', ink()).attr('stroke-width', 1.8)
    .on('mousemove', (e, v, i) => tip(e, `PC${nullEv.indexOf(v) + 1} in shuffled data: ${f2(v)} (95th percentile)`)).on('mouseleave', untip);
  legend(f4.body, [['kept (first k)', cat(0)], ['not kept', css('--axis')], ['shuffled data, 95th percentile', ink()]]);
  function paint4() {
    bars4.selectAll('rect').data(PA.vals).join('rect').attr('x', (v, i) => F4.x(i + 1) - 17).attr('width', 34).attr('y', (v) => F4.y(v)).attr('height', (v) => F4.h - F4.y(v)).attr('fill', (v, i) => (i < k4 ? cat(0) : css('--axis'))).attr('rx', 2)
      .on('mousemove', (e, v) => { const i = PA.vals.indexOf(v); tip(e, `PC${i + 1}: variance ${f2(v)} (${pc(PA.ratio[i])}); cumulative ${pc(sum(PA.ratio.slice(0, i + 1)))}`); }).on('mouseleave', untip);
    bars4.selectAll('text').data(PA.ratio).join('text').attr('x', (v, i) => F4.x(i + 1)).attr('y', (v, i) => F4.y(Math.max(PA.vals[i], nullEv[i])) - 9).attr('text-anchor', 'middle').attr('font-size', 11.5).attr('fill', ink()).text((v) => pc(v));
    const z = PA.Z[pat4], sc4 = PA.S[pat4], rec = PF.map((_, j) => sum(d3.range(k4).map((m) => sc4[m] * PA.vecs[m][j])));
    hbars(g4b, PF.map((k, j) => ({ lab: SH[k], v: rec[j], ref: z[j], tip: `${SH[k]}: actual ${f2(z[j])} SD, reconstructed ${f2(rec[j])} SD` })), { dom: [-4, 4], h: 300, xLabel: 'standardised value (SD); diamond = actual', color: () => cat(0) });
    const rms = Math.sqrt(sum(PA.Z.map((zz, i) => sum(PF.map((_, j) => (zz[j] - sum(d3.range(k4).map((m) => PA.S[i][m] * PA.vecs[m][j]))) ** 2)))) / (n * PF.length));
    s4.set({ 'variance kept (first k)': pc(sum(PA.ratio.slice(0, k4))), 'RMS error per value (SD)': f2(rms), 'PCs above average (Kaiser)': PA.vals.filter((v) => v > 1).length, 'PCs above shuffled data': PA.vals.filter((v, i) => v > nullEv[i]).length });
  }
  paint4();
  const kaiser = PA.vals.filter((v) => v > 1).length, par = PA.vals.filter((v, i) => v > nullEv[i]).length;
  let k80 = 0, cum = 0; while (cum < 0.8 && k80 < PF.length) cum += PA.ratio[k80++];
  prose(rb, `<p>PC1 (${pc(PA.ratio[0])}) stands clear of the rest. The Kaiser rule keeps ${kaiser} PCs, parallel analysis keeps ${par}, and ${k80} PCs are needed to reach 80% of the variance. The rules disagree because PCs 2–7 are close to the level that uncorrelated features would give: TMB, proliferation and age are independent of the immune block and therefore each occupy a PC of their own. No rule is correct in general; the choice depends on the purpose (display: 2; input to a model: validate it).</p>`);

  // ---- Step 5: loadings and the immune axis
  rb = rung(ch, 5, 'Reading the loadings: PC1 as an immune axis',
    `<p>A PC is interpreted through its loadings. The heatmap lists the loadings of every PC (columns) on every original feature (rows); a large positive or negative value means the feature contributes strongly. The biplot shows patients on two chosen PCs, with an arrow for each feature drawn from its loadings: features whose arrows point the same way are positively correlated.</p>`);
  const f5 = figure(rb, 'Biplot and loading table', 'Left: patients on the chosen PCs; arrows show where each original feature points (scaled to fit). Right: loadings of all seven PCs, with the chosen columns outlined. The sign of a whole PC is arbitrary; the convention here makes the largest loading positive.');
  const colorOpts = { response: ['Response', (p) => p.response, [['no response', cat(0)], ['responded', cat(1)]], (p) => (p.response ? 'responded' : 'no response')],
    phenotype: ['Tumour phenotype', (p) => ['desert', 'excluded', 'inflamed'].indexOf(p.phenotype), [['desert', cat(0)], ['excluded', cat(1)], ['inflamed', cat(2)]], ph],
    centre: ['Centre', (p) => p.centreB, [['Centre A', cat(0)], ['Centre B', cat(1)]], (p) => p.centre] };
  let cb5 = 'phenotype', px = 0, py = 1;
  const pcOpts = d3.range(PF.length).map((m) => [String(m), 'PC' + (m + 1)]);
  select(f5.controls, { label: 'Colour by', value: cb5, options: Object.entries(colorOpts).map(([k, v]) => [k, v[0]]), onChange: (v) => { cb5 = v; paint5(); } });
  select(f5.controls, { label: 'x axis', value: '0', options: pcOpts, onChange: (v) => { px = +v; paint5(); } });
  select(f5.controls, { label: 'y axis', value: '1', options: pcOpts, onChange: (v) => { py = +v; paint5(); } });
  const r5 = el('div', { class: 'fig-row' }); f5.body.append(r5);
  const g5a = el('div'), g5b = el('div'); r5.append(g5a, g5b);
  const ext5 = Math.ceil(d3.max(PA.S.flat(), Math.abs));
  const F5 = frame(g5a, { width: 500, height: 420, x: [-ext5, ext5], y: [-ext5, ext5], xLabel: 'PC1', yLabel: 'PC2' });
  const hm5 = heatmap(g5b, PA.vecs[0].map((_, j) => PA.vecs.map((v) => v[j])), { cell: 44, color: diverging(1), rowLabels: PF.map((k) => SH[k]), colLabels: PA.vecs.map((_, m) => `PC${m + 1}`), labelW: 96, labelH: 84, fmt: f2, onHover: (e, i, j, v) => tip(e, `${FEATURES[PF[i]].label} on PC${j + 1}: loading ${f2(v)} (PC${j + 1} explains ${pc(PA.ratio[j])} of variance)`) });
  const outl = hm5.svg.append('g'), leg5 = el('div'); f5.body.append(leg5);
  function paint5() {
    const [, cf, lg, lab] = colorOpts[cb5], cols = lg.map((l) => l[1]);
    F5.svg.select('.layer-pts').remove();
    const pts = F5.g.insert('g', '.layer-ar').attr('class', 'layer-pts');
    F5.svg.selectAll('.ax-label').each(function (_, i) { d3.select(this).text(i === 0 ? `PC${px + 1} score (${pc(PA.ratio[px])} of variance)` : `PC${py + 1} score (${pc(PA.ratio[py])})`); });
    pts.selectAll('circle').data(D).join('circle').attr('cx', (d, i) => F5.x(PA.S[i][px])).attr('cy', (d, i) => F5.y(PA.S[i][py])).attr('r', 3.8).attr('fill', (d) => cols[cf(d)]).attr('fill-opacity', 0.7).attr('stroke', css('--surface'))
      .on('mousemove', (e, d) => { const i = D.indexOf(d); tip(e, `${d.id} · ${lab(d)}<br>PC${px + 1} ${f2(PA.S[i][px])}, PC${py + 1} ${f2(PA.S[i][py])}`); }).on('mouseleave', untip);
    const ar = F5.layer('ar'); ar.selectAll('*').remove();
    const s = ext5 * 0.62 / Math.max(0.3, d3.max(PF, (_, j) => Math.hypot(PA.vecs[px][j], PA.vecs[py][j])));
    const ends = PF.map((k, j) => ({ k, x: PA.vecs[px][j] * s, y: PA.vecs[py][j] * s })).sort((a, b) => b.y - a.y);
    let last = -1e9; ends.forEach((e) => { e.ty = Math.max(F5.y(e.y), last + 13); last = e.ty; });
    ends.forEach((e) => {
      ar.append('line').attr('x1', F5.x(0)).attr('y1', F5.y(0)).attr('x2', F5.x(e.x)).attr('y2', F5.y(e.y)).attr('stroke', ink()).attr('stroke-width', 1.8);
      arrowHead(ar, F5.x(e.x), F5.y(e.y), Math.atan2(F5.y(e.y) - F5.y(0), F5.x(e.x) - F5.x(0)) * 180 / Math.PI, ink());
      ar.append('text').attr('x', F5.x(e.x) + (e.x >= 0 ? 7 : -7)).attr('y', e.ty + 4).attr('text-anchor', e.x >= 0 ? 'start' : 'end').attr('font-size', 11.5).attr('font-weight', 600).attr('fill', ink()).attr('stroke', css('--surface')).attr('stroke-width', 3).attr('paint-order', 'stroke').text(SH[e.k]);
    });
    leg5.innerHTML = ''; legend(leg5, lg);
    outl.selectAll('rect').data([px, py]).join('rect').attr('x', (m) => 96 + m * 44 - 1).attr('y', 84 - 1).attr('width', 44).attr('height', PF.length * 44 + 1).attr('fill', 'none').attr('stroke', ink()).attr('stroke-width', 2).attr('rx', 3);
  }
  paint5();
  const mPh = (t) => mean(D.map((p, i) => (p.phenotype === t ? PA.S[i][0] : NaN)).filter((v) => !Number.isNaN(v)));
  const rResp = corr(PA.S.map((s) => s[0]), y);
  prose(rb, `<p><b>PC1 as an immune axis.</b> PC1 (${pc(PA.ratio[0])} of the variance) has loadings of ${PF.slice(0, 4).map((k, j) => `${f2(PA.vecs[0][j])} (${SH[k]})`).join(', ')}: the signs show that IFN-γ, CD8 and PD-L1 rise together while CAF score falls. This agrees with the tumour phenotypes, which were not given to PCA: mean PC1 score is ${f2(mPh('inflamed'))} in inflamed, ${f2(mPh('excluded'))} in excluded and ${f2(mPh('desert'))} in desert tumours. "Immune axis" is a name chosen by the analyst after reading the loadings and checking against known biology; PCA does not provide it.</p>
    <p><b>Variance is not relevance.</b> PC1 correlates with response (r = ${f2(rResp)}), but TMB, a true driver of response in this study, is nearly uncorrelated with the immune block and appears on PC${tmbPC + 1} (loading ${f2(PA.vecs[tmbPC][iTMB])}, ${pc(PA.ratio[tmbPC])} of variance). On the validation patients PC1 alone separates responders with an AUC of ${f2(aucPC(0))}, and PC${tmbPC + 1} with ${f2(aucPC(tmbPC))}.</p>`);

  // ---- Step 6: lab
  rb = rung(ch, 6, 'Lab: PCs as model inputs, and batch effects on the PCs',
    `<p>Fitted on the ${n} discovery patients (Centres A and B), evaluated on the ${V.length} validation patients (Centre C). A logistic regression is fitted on the first <i>k</i> PC scores, so the right panel shows what is gained as components are added. The slider adds a <b>simulated technical shift</b> to the expression-based features (IFN-γ and CAF scores) of Centre B patients, to show how a batch effect looks on the PCs; this shift is a teaching device and is not part of the cohort. The checkbox applies the simplest correction, centring each centre's features on its own mean.</p>`);
  const SD0 = Object.fromEntries(PF.map((k) => [k, sd(D.map((p) => FEATURES[k].get(p)))]));
  const labCache = new Map();
  const prep = (pts, delta, cc) => {
    const M = pts.map((p) => PF.map((k) => FEATURES[k].get(p) + (p.centreB && (k === 'ifng' || k === 'caf') ? delta * SD0[k] : 0)));
    if (cc) [...new Set(pts.map((p) => p.centre))].forEach((c) => { const ix = pts.map((p, i) => i).filter((i) => pts[i].centre === c); PF.forEach((_, j) => { const m = mean(ix.map((i) => M[i][j])); ix.forEach((i) => { M[i][j] -= m; }); }); });
    return M;
  };
  const eta = (S, m) => { const v = S.map((s) => s[m]), mu = mean(v), g = [0, 1].map((b) => v.filter((_, i) => D[i].centreB === b)); return sum(g.map((a) => a.length * (mean(a) - mu) ** 2)) / sum(v.map((q) => (q - mu) ** 2)); };
  const lab = (delta, cc) => {
    const key = delta + '|' + cc; if (labCache.has(key)) return labCache.get(key);
    const P = pca(prep(D, delta, cc)), SV = prep(V, delta, cc).map((r) => P.project(r));
    const mp = [], aucs = d3.range(1, PF.length + 1).map((k) => { const f = fitLogistic(P.S.map((s) => s.slice(0, k)), y); const pv = SV.map((s) => predictProb(s.slice(0, k), f.b0, f.w)); mp.push(mean(pv)); return auc(pv, yv); });
    const res = { P, SV, aucs, mp, eta: [eta(P.S, 0), eta(P.S, 1)] };
    labCache.set(key, res); return res;
  };
  const f6 = figure(rb, 'Principal components as inputs, with an optional batch effect', 'Left: discovery patients (Centres A and B) and validation patients (Centre C) on PC1 and PC2 of the discovery PCA. Right: validation AUC of a logistic regression on the first k PCs (lollipops; chance = 0.5; dashed = model on all seven original features, identical to k = 7).');
  let k6 = 2, d6 = 0, cc6 = false;
  slider(f6.controls, { label: 'components used k', min: 1, max: 7, step: 1, value: k6, onInput: (v) => { k6 = v; paint6(); } });
  slider(f6.controls, { label: 'technical shift in Centre B (SD)', min: 0, max: 3, step: 0.25, value: 0, fmt: (v) => v.toFixed(2), onInput: (v) => { d6 = v; paint6(); } });
  checkbox(f6.controls, { label: 'centre each centre on its own mean', checked: false, onChange: (c) => { cc6 = c; paint6(); } });
  const s6 = stats(f6.root);
  const r6 = el('div', { class: 'fig-row' }); f6.body.append(r6);
  const g6a = el('div'), g6b = el('div'); r6.append(g6a, g6b);
  const F6 = frame(g6a, { width: 500, height: 380, x: [-5, 5], y: [-4, 4], xLabel: 'PC1 score', yLabel: 'PC2 score' });
  const F6b = frame(g6b, { width: 440, height: 380, x: [0.4, 7.6], y: [0.4, 1], xLabel: 'number of PCs used (k)', yLabel: 'validation AUC', xTicks: 7 });
  F6b.layer('c').append('line').attr('x1', 0).attr('x2', F6b.w).attr('y1', F6b.y(0.5)).attr('y2', F6b.y(0.5)).attr('stroke', muted()).attr('stroke-dasharray', '4 4');
  F6b.layer('c').append('text').attr('x', 6).attr('y', F6b.y(0.5) + 15).attr('font-size', 11.5).attr('fill', ink2()).text('chance (0.5)');
  const allLine = F6b.layer('all'), lolli = F6b.layer('lol');
  legend(f6.body, [['Centre A (discovery)', cat(0)], ['Centre B (discovery)', cat(1)], ['Centre C (validation)', cat(2)]]);
  function paint6() {
    const L = lab(d6, cc6), P = L.P;
    F6.svg.selectAll('.ax-label').each(function (_, i) { d3.select(this).text(i === 0 ? `PC1 score (${pc(P.ratio[0])} of variance)` : `PC2 score (${pc(P.ratio[1])})`); });
    const dots = [...D.map((p, i) => ({ p, s: P.S[i], c: p.centreB })), ...V.map((p, i) => ({ p, s: L.SV[i], c: 2 }))];
    F6.layer('pts').selectAll('circle').data(dots).join('circle').attr('cx', (d) => F6.x(Math.max(-5, Math.min(5, d.s[0])))).attr('cy', (d) => F6.y(Math.max(-4, Math.min(4, d.s[1])))).attr('r', 3.6).attr('fill', (d) => cat(d.c)).attr('fill-opacity', 0.6).attr('stroke', css('--surface'))
      .on('mousemove', (e, d) => tip(e, `${d.p.id} (${d.p.centre})<br>PC1 ${f2(d.s[0])}, PC2 ${f2(d.s[1])}`)).on('mouseleave', untip);
    allLine.selectAll('*').remove();
    allLine.append('line').attr('x1', 0).attr('x2', F6b.w).attr('y1', F6b.y(L.aucs[6])).attr('y2', F6b.y(L.aucs[6])).attr('stroke', ink()).attr('stroke-dasharray', '5 4');
    allLine.append('text').attr('x', F6b.w - 4).attr('y', F6b.y(L.aucs[6]) - 6).attr('text-anchor', 'end').attr('font-size', 11.5).attr('fill', ink2()).text(`all 7 features: ${f2(L.aucs[6])}`);
    const ls = L.aucs.map((a, i) => ({ k: i + 1, a }));
    lolli.selectAll('line').data(ls).join('line').attr('x1', (d) => F6b.x(d.k)).attr('x2', (d) => F6b.x(d.k)).attr('y1', F6b.y(0.5)).attr('y2', (d) => F6b.y(Math.max(0.4, d.a))).attr('stroke', (d) => (d.k === k6 ? cat(6) : css('--axis'))).attr('stroke-width', 3);
    lolli.selectAll('circle').data(ls).join('circle').attr('cx', (d) => F6b.x(d.k)).attr('cy', (d) => F6b.y(Math.max(0.4, d.a))).attr('r', (d) => (d.k === k6 ? 7 : 5)).attr('fill', (d) => (d.k === k6 ? cat(6) : css('--axis'))).attr('stroke', css('--surface')).attr('stroke-width', 1.5)
      .on('mousemove', (e, d) => tip(e, `first ${d.k} PC${d.k > 1 ? 's' : ''}: validation AUC ${f2(d.a)}`)).on('mouseleave', untip);
    s6.set({ 'variance in first k PCs': pc(sum(P.ratio.slice(0, k6))), 'validation AUC (k PCs)': f2(L.aucs[k6 - 1]), [`mean predicted response, validation (observed ${pc(mean(yv))})`]: pc(L.mp[k6 - 1]), 'PC1 variance explained by centre': pc(L.eta[0]), 'PC2 variance explained by centre': pc(L.eta[1]) });
  }
  paint6();
  const L0 = lab(0, false), L3 = lab(3, false), L3c = lab(3, true), L0c = lab(0, true);
  const bestK = d3.range(PF.length).sort((a, b) => L0.aucs[b] - L0.aucs[a])[0] + 1;
  prose(rb, '<p><b>Suggested comparisons</b>:</p>');
  table(rb, [
    { t: '<b>k = 1 → 2 → 7</b>, no shift', s: `AUC ${f2(L0.aucs[0])} → ${f2(L0.aucs[1])} → <b>${f2(L0.aucs[6])}</b>; the highest AUC is at k = ${bestK}`, m: 'PCs are ordered by variance, not by relevance. The PC carrying TMB is added late, and the AUC rises when it enters. With k = 7 the model equals one on the original features, because PCA is only a rotation.' },
    { t: 'Raise the <b>shift to 3 SD</b>, k = 1 then 2', s: `Centre explains ${pc(L0.eta[0])} → <b>${pc(L3.eta[0])}</b> of PC1 and ${pc(L0.eta[1])} → <b>${pc(L3.eta[1])}</b> of PC2; validation AUC (k = 1) ${f2(L0.aucs[0])} → <b>${f2(L3.aucs[0])}</b>, (k = 2) ${f2(L0.aucs[1])} → ${f2(L3.aucs[1])}; mean predicted response (k = 2) ${pc(L0.mp[1])} → ${pc(L3.mp[1])}`, m: `A technical difference between centres becomes a leading axis of variation. Here it costs only a little AUC, because the shift ${Math.abs(L3.eta[0] - L3.eta[1]) < 0.2 ? `spreads over PC1 and PC2 (${pc(L3.eta[0])} and ${pc(L3.eta[1])})` : `falls mostly on PC${L3.eta[0] > L3.eta[1] ? 1 : 2}`} and the ranking of patients is largely preserved. It is nonetheless a distortion of the data that clustering, distance calculations and calibration all inherit.` },
    { t: 'Shift 3 SD, then <b>centre each centre</b>', s: `Centre explains ${pc(L3c.eta[0])} of PC1; validation AUC (k = 1) ${f2(L3c.aucs[0])}, (k = 2) ${f2(L3c.aucs[1])}`, m: 'Per-centre centring removes an additive shift exactly. Dedicated methods (ComBat, limma <code>removeBatchEffect</code>) do this while protecting known biological covariates.' },
    { t: 'Shift 0, <b>centring on</b>', s: `Validation AUC (k = 2) ${f2(L0.aucs[1])} → ${f2(L0c.aucs[1])}`, m: `Centre B truly enrolled more inflamed tumours, so centring also removes some real biology. When batch and biology are confounded, the PCs cannot tell them apart; the study design must.` },
  ], [['t', 'Try'], ['s', 'What you see'], ['m', 'What it means']], { cls: 'compact' });
  quiz(rb, 'A PC plot of 30 samples separates cleanly into two groups. Does this show a biological difference?', 'Not by itself. The groups may follow a technical factor such as batch, centre or RNA quality. Colour the plot by every known technical and biological variable, and check whether the separation survives in a design where batch and condition are not aligned.');

  // ===================================================================================== 5. PITFALLS
  ch = chapter(main, 'pitfalls', 'Pitfalls', 'Common problems, how to recognise them, and how to address them.');
  const boot = (() => { const g = rng(21), M = X(D, PF), out = [[], [], []]; for (let b = 0; b < 40; b++) { const idx = d3.range(n).map(() => Math.floor(g() * n)), Pb = pca(idx.map((i) => M[i])); [0, 1, 2].forEach((m) => out[m].push(Math.abs(dot(Pb.vecs[m], PA.vecs[m])))); } return out.map((a) => d3.median(a)); })();
  const ang = (c) => Math.acos(Math.min(1, c)) * 180 / Math.PI;
  const rPh = corr(PA.S.map((s) => s[0]), D.map((p) => (p.phenotype === 'inflamed' ? 1 : 0)));
  cards(ch, [
    { tag: 'Symptom: one feature dominates PC1', title: 'Features not standardised', body: `On raw units PC1 explained ${pc(PR.ratio[0])} of the variance and was essentially ${SH[PF[rawTop]]} (loading ${f2(PR.vecs[0][rawTop])}). <b>Fix:</b> standardise unless all features share one unit, and state which choice was made.` },
    { tag: 'Symptom: the PC that matters is not PC1', title: 'Variance is not relevance', body: `PCA never sees the outcome. TMB, a true driver of response here, sits on PC${tmbPC + 1} (${pc(PA.ratio[tmbPC])} of variance), while PC1 is the redundant immune block. Keeping only the first PCs as model inputs can discard the useful signal. <b>Fix:</b> use PCs for structure and display; for prediction, validate any choice of k or use a supervised or penalised model.` },
    { tag: 'Symptom: samples separate by centre or batch', title: 'Batch effects dominate a PC', body: `A simulated shift of 3 SD in two expression features made the centre label explain ${pc(L3.eta[0])} of PC1 and ${pc(L3.eta[1])} of PC2 (${pc(L0.eta[0])} and ${pc(L0.eta[1])} without it). In real data batch and biology are often confounded, as with Centre B here (${pc(D.filter((p) => p.centreB && p.phenotype === 'inflamed').length / D.filter((p) => p.centreB).length)} inflamed vs ${pc(D.filter((p) => !p.centreB && p.phenotype === 'inflamed').length / D.filter((p) => !p.centreB).length)} in Centre A). <b>Fix:</b> randomise samples across batches at design time, record batch, and correct with the design in the model.` },
    { tag: 'Symptom: PC2 or PC3 changes when a few samples are removed', title: 'Close eigenvalues give unstable PCs', body: `In 40 bootstrap resamples of the ${n} patients, the median angle between the resampled and original PC1 was ${f1(ang(boot[0]))}°, but ${f1(ang(boot[1]))}° for PC2 and ${f1(ang(boot[2]))}° for PC3 (eigenvalues ${f2(PA.vals[1])} and ${f2(PA.vals[2])} are close). When eigenvalues are close, any rotation within their plane fits the data almost equally well. <b>Fix:</b> interpret PCs whose eigenvalues are well separated, or interpret the subspace of several PCs together.` },
    { tag: 'Symptom: different rules give different k', title: 'Choosing the number of components', body: `Here the Kaiser rule keeps ${kaiser}, parallel analysis ${par} and an 80% cut-off ${k80}. <b>Fix:</b> pick k for the purpose, and for modelling treat k as a hyperparameter chosen by cross-validation, with PCA fitted inside each training fold.` },
    { tag: 'Symptom: validation worse than training', title: 'Fitting PCA on all samples', body: 'Standardising, selecting genes or fitting PCA on the training and test samples together lets test data shape the axes. <b>Fix:</b> estimate means, SDs and loadings on the training data only, then project new samples with the stored values (<code>transform</code>).' },
    { tag: 'Symptom: a few samples define PC1', title: 'Outliers, skew and non-linear structure', body: `PCA maximises squared deviations, so one extreme sample can pull a PC towards itself. TMB spans ${f1(d3.min(D, (p) => p.tmb))}–${f1(d3.max(D, (p) => p.tmb))} mut/Mb, so it was log2-transformed. Curved or clustered structure is not captured by straight axes. <b>Fix:</b> transform skewed data, inspect outliers (and run PCA with and without them), and use non-linear methods when the PCs look curved.` },
    { tag: 'Symptom: "PC1 is the immune response"', title: 'Names are interpretations', body: `PC1 correlates with the inflamed phenotype (r = ${f2(rPh)}), which supports the name. It remains a weighted sum of four imperfect assays, and also has small contributions from the rest. Also, ${n} patients support at most ${n - 1} non-trivial PCs however many features are measured; with thousands of genes, loadings are long vectors that are hard to read. <b>Fix:</b> back each interpretation with an independent check and report loadings.` },
  ]);

  // ===================================================================================== 6. RECAP
  ch = chapter(main, 'recap', 'Recap', 'Key points.');
  recap(ch, [
    'PCA finds new axes, <b>weighted sums of the original features</b>, that are uncorrelated and ordered by the variance they capture. It uses only the features, never the outcome.',
    'Each PC is an <b>eigenvector of the correlation (or covariance) matrix</b>; its eigenvalue is the variance along it. For two standardised features the eigenvalues are 1 ± r.',
    '<b>Standardise first</b> when features have different units; otherwise the feature with the largest numbers dominates PC1.',
    'The <b>scree plot</b> shows how much each PC keeps. Rules for choosing k (elbow, Kaiser, parallel analysis) can disagree; choose k for the purpose.',
    '<b>Loadings</b> show which features form a PC and a <b>score</b> places each sample on it. Naming a PC (for example an immune axis) is an interpretation that needs an independent check. The sign of a PC is arbitrary.',
    '<b>High variance is not the same as relevance to an outcome</b>: a true predictor can sit on a low-variance PC.',
    'Batch and technical factors appear as PCs. Colour PC plots by batch to find them, and note that batch and biology can be confounded.',
    'Fit scaling and PCA on training data only, project new samples with the stored parameters, and treat k as a hyperparameter when PCs feed a model.',
  ]);
  prose(ch, '<h3>In code</h3><p>Standardise, fit on the training samples, inspect the variance explained and loadings, then project new samples with the same transformation.</p>');
  code(ch, {
    Python: `import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

feats = ["ifng", "cd8", "pdl1", "caf", "log2tmb", "prolif", "age"]   # log2tmb = np.log2(tmb)
pipe = make_pipeline(StandardScaler(), PCA(n_components=None)).fit(train[feats])
pca = pipe.named_steps["pca"]

print(pca.explained_variance_ratio_.round(3))              # scree: share of variance per PC
loadings = pd.DataFrame(pca.components_.T, index=feats,
                        columns=[f"PC{i+1}" for i in range(len(feats))])
print(loadings.round(2))                                   # rows = features

scores_test = pipe.transform(test[feats])                  # project new samples (no refitting)
# single-cell / many genes: PCA(n_components=50, svd_solver="randomized")`,
    R: `feats <- c("ifng", "cd8", "pdl1", "caf", "log2tmb", "prolif", "age")   # log2tmb = log2(tmb)
fit <- prcomp(train[, feats], center = TRUE, scale. = TRUE)

summary(fit)                                  # standard deviation and proportion of variance per PC
screeplot(fit, type = "lines")
round(fit$rotation, 2)                        # loadings: rows = features, columns = PCs

scores_test <- predict(fit, newdata = test[, feats])   # project new samples with the stored centre and scale
# many genes: irlba::prcomp_irlba(x, n = 50, center = TRUE, scale. = TRUE)`,
  });
  seeAlso(main, [['concepts', 'logistic-regression', 'Logistic regression'], ['concepts', 'regularization', 'Regularisation'], ['concepts', 'clustering', 'Clustering'], ['concepts', 'tsne-umap', 't-SNE and UMAP'], ['omics', 'bulk-qc-batch', 'QC, normalisation and batch effects'], ['omics', 'sc-clustering', 'Clustering and cell-type annotation']]);
}
