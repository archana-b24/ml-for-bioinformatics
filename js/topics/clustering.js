// Clustering, lesson format.
// Arc: what → why → where → how it works (6 steps: distance → k-means → choosing k → hierarchical → kNN graph + Louvain/Leiden
//      → cluster builder with try/see/means table) → pitfalls → recap.
// The clusters are found from tumour measurements only; the known immune phenotype (inflamed / excluded / desert) is held back
// and used afterwards to judge them (adjusted Rand index). Every number in the prose is computed from the data at render time.
import { header, figure, slider, select, button, checkbox, stats, legend, seeAlso, animate, tip, untip, el, html } from '../lib/ui.js';
import { frame, cat, css, ink, ink2, muted } from '../lib/plot.js';
import { rng, mean, sd } from '../lib/data.js';
import { standardize, pca, pNormal, kmeans, kmeansInit, kmeansAssign, kmeansUpdate, inertia, silhouette, ari, agglomerate, cutTree, cutForK, knnGraph, modularity, louvain, knnTransfer } from '../lib/models.js';
import { loadCohort, FEATURES, studyCard } from '../lib/study.js';
import { lessonNav, chapter, prose, rung, worked, quiz, cards, table, code, recap } from '../lib/lesson.js';
const d3 = window.d3;

const f2 = (v) => (Math.abs(v) < 0.005 ? '0.00' : v.toFixed(2));
const f1 = (v) => v.toFixed(1);
const pc = (v) => Math.round(100 * v) + '%';
const sum = (a) => a.reduce((s, v) => s + v, 0);
const pFmt = (p) => (p < 0.001 ? '< 0.001' : p.toFixed(3));
const PHENO = ['inflamed', 'excluded', 'desert'];
const PHC = (p) => cat({ inflamed: 1, excluded: 2, desert: 6 }[p]);   // colour of the known phenotype
const CL = (c) => cat([0, 3, 4, 5, 7, 1, 2, 6][c % 8]);                 // colour of a found cluster (different slots from phenotype)
const FSETS = { two: ['ifng', 'caf'], four: ['ifng', 'caf', 'cd8', 'pdl1'], six: ['ifng', 'caf', 'cd8', 'pdl1', 'prolif', 'log2tmb'] };
const FSET_LAB = [['two', 'IFN-γ + CAF (2 features)'], ['four', 'IFN-γ, CAF, CD8, PD-L1 (4)'], ['six', 'the 4 plus proliferation and TMB (6)']];

export default async function render(main) {
  const C = await loadCohort();
  const D = C.discovery, V = C.validation, n = D.length;
  const rowsOf = (pts, keys) => pts.map((p) => keys.map((k) => FEATURES[k].get(p)));
  const phD = D.map((p) => p.phenotype), phV = V.map((p) => p.phenotype);
  const zOf = (keys) => { const st = standardize(rowsOf(D, keys)); return { Z: st.Z, Zv: rowsOf(V, keys).map(st.apply), st }; };
  const S4 = zOf(FSETS.four), Z4 = S4.Z;
  const nPh = Object.fromEntries(PHENO.map((p) => [p, phD.filter((q) => q === p).length]));
  const km = (Z, k, o = {}) => kmeans(Z, k, { nInit: 5, seed: 1, ...o });
  const compo = (lab, ph = phD) => d3.range(Math.max(...lab) + 1).map((c) => {
    const m = PHENO.map((p) => [p, ph.filter((q, i) => lab[i] === c && q === p).length]).filter((q) => q[1] > 0).sort((a, b) => b[1] - a[1]);
    return `cluster ${c + 1}: ${m.map(([p, v]) => `${v} ${p}`).join(', ')}`;
  }).join('; ');
  const km3 = km(Z4, 3), ari3 = ari(phD, km3.labels);

  header(main, 'Clustering',
    'Grouping samples, with no labels supplied, so that <b>members of a group resemble each other more than they resemble the rest</b>. Here the samples are patients described by tumour measurements, and the groups are later compared with the known immune phenotypes.');
  lessonNav(main);
  studyCard(main, C, '<b>Question for this page:</b> without being told the immune phenotype, can the tumour measurements alone recover the inflamed, excluded and desert groups, and how does the choice of method and of the number of clusters change the answer?');

  // =====================================================================================
  // 1. WHAT
  // =====================================================================================
  let ch = chapter(main, 'what', 'What is clustering?', 'In one line: <b>measure similarity → group similar samples → check whether the groups mean anything</b>.');
  prose(ch, `<p>Clustering is <b>unsupervised</b>: the algorithm receives a table of measurements and returns a group label for each row, without any outcome or class to learn from. The result is a <b>partition</b> of the samples into clusters. It depends on three choices: which features describe each sample, how similarity is measured, and the grouping rule (k-means, hierarchical, graph-based; step 2 onwards).</p>
    <p>The ${n} discovery patients carry a pathologist-assigned <b>immune phenotype</b>: <b>inflamed</b> (${nPh.inflamed} patients; T cells inside the tumour), <b>excluded</b> (${nPh.excluded}; T cells held in the stroma) and <b>desert</b> (${nPh.desert}; few T cells). The algorithm never sees this label. It is used only afterwards, as a reference for judging the clusters.</p>`);
  const f0 = figure(ch, 'The same patients, with and without the label', 'Left: what the algorithm receives, measurements only. Right: the same points coloured by the known phenotype, which is withheld from clustering. Each dot is one discovery patient; hover for details.');
  const row0 = el('div', { class: 'fig-row' }); f0.body.append(row0);
  const pa = el('div'), pb = el('div'); row0.append(pa, pb);
  const XR = [2, 10.4], YR = [0.8, 7.2];
  [pa, pb].forEach((host, side) => {
    const F = frame(host, { width: 460, height: 330, x: XR, y: YR, xLabel: 'IFN-γ score (log2 expression)', yLabel: 'CAF / TGF-β score' });
    F.layer('p').selectAll('circle').data(D).join('circle').attr('cx', (d) => F.x(d.ifng)).attr('cy', (d) => F.y(d.caf)).attr('r', 4.2)
      .attr('fill', (d) => (side ? PHC(d.phenotype) : muted())).attr('fill-opacity', 0.8).attr('stroke', css('--surface')).attr('stroke-width', 1)
      .on('mousemove', (e, d) => tip(e, `${d.id}<br>IFN-γ ${d.ifng}, CAF ${d.caf}${side ? `<br>phenotype: ${d.phenotype}` : ''}`)).on('mouseleave', untip);
    if (side) PHENO.forEach((p) => {
      const q = D.filter((d) => d.phenotype === p);
      F.layer('t').append('text').attr('x', F.x(mean(q.map((d) => d.ifng)))).attr('y', F.y(mean(q.map((d) => d.caf))) - (p === 'excluded' ? 40 : p === 'desert' ? 34 : 32)).attr('text-anchor', 'middle').attr('font-size', 13).attr('font-weight', 700).attr('fill', ink()).attr('stroke', css('--surface')).attr('stroke-width', 3.5).attr('paint-order', 'stroke').text(p);
    });
    else F.layer('t').append('text').attr('x', F.w - 6).attr('y', 14).attr('text-anchor', 'end').attr('font-size', 12).attr('fill', ink2()).text('no labels supplied');
  });
  legend(f0.body, PHENO.map((p) => [p, PHC(p)]));
  prose(ch, '<p>Key terms:</p>');
  table(ch, [
    { t: 'Feature', m: 'A measurement used to describe each sample', s: 'IFN-γ score, CAF score, CD8 %, PD-L1 TPS' },
    { t: 'Distance', m: 'A number that is small for similar samples; usually Euclidean distance after scaling', s: 'Step 1' },
    { t: 'Cluster', m: 'A group of samples the algorithm treats as similar', s: `k-means with k = 3 returns clusters of ${[...d3.rollup(km3.labels, (v) => v.length, (v) => v).entries()].sort((a, b) => a[0] - b[0]).map((q) => q[1]).join(', ')} patients` },
    { t: 'Centroid', m: 'The mean of a cluster\'s members, in feature space', s: 'k-means, step 2' },
    { t: 'k', m: 'The number of clusters; chosen by the analyst for k-means and hierarchical clustering', s: 'Step 3' },
    { t: 'Inertia', m: 'Sum of squared distances from each sample to its centroid; k-means minimises it', s: `${f1(km3.inertia)} for k = 3 (standardised units)` },
    { t: 'Silhouette', m: 'For each sample, how much closer it is to its own cluster than to the nearest other cluster (−1 to 1)', s: `mean ${f2(silhouette(Z4, km3.labels).mean)} for k = 3` },
    { t: 'Adjusted Rand index (ARI)', m: 'Agreement between two partitions of the same samples; 1 = identical, about 0 = no better than chance', s: `k = 3 clusters vs phenotype: ${f2(ari3)}` },
  ], [['t', 'Term'], ['m', 'Meaning'], ['s', 'In our study']]);

  // =====================================================================================
  // 2. WHY
  // =====================================================================================
  ch = chapter(main, 'why', 'Why do we need it?', 'When there is no label to predict, or the labels are suspected to be incomplete, the structure has to come from the measurements.');
  prose(ch, `<p>Clustering is used when:</p><ol>
    <li><b>No label exists yet.</b> New tumour subtypes, cell types and microbial communities are discovered before they are named.</li>
    <li><b>Existing labels are coarse or subjective.</b> Immune phenotype is assigned by a pathologist on a slide; measurements can show whether the groups are reproducible.</li>
    <li><b>The data are too large to inspect.</b> A single-cell experiment has tens of thousands of cells; groups make it possible to summarise, annotate and compare them.</li></ol>
    <p>The alternatives are manual thresholds on one marker, or supervised models, which need labels.</p>`);
  const tert = (key) => { const s = D.map((p, i) => [p[key], i]).sort((a, b) => a[0] - b[0]), lab = new Array(n); s.forEach(([, i], r) => { lab[i] = Math.floor(3 * r / n); }); return lab; };
  const aT = ari(phD, tert('ifng')), aP = ari(phD, tert('pdl1'));
  worked(ch, 'A single-marker rule versus clustering on several markers',
    `<p>Splitting patients into three equal groups by one marker gives an ARI against the phenotype of <b>${f2(aT)}</b> for IFN-γ and <b>${f2(aP)}</b> for PD-L1. K-means with k = 3 on four standardised measurements (IFN-γ, CAF, CD8, PD-L1) reaches <b>${f2(ari3)}</b>. The excluded and desert phenotypes both have low IFN-γ and differ mainly in stromal (CAF) signal, which a single immune marker cannot separate.</p>`);
  table(ch, [
    { a: 'Threshold on one marker', l: 'no', f: 'one', c: 'low', i: 'rule is explicit' },
    { a: '<b>Clustering</b>', l: '<b>no</b>', f: '<b>several</b>', c: '<b>k, features and method are choices</b>', i: '<b>groups need to be interpreted afterwards</b>' },
    { a: 'Supervised classifier', l: 'yes', f: 'several', c: 'learned from labels', i: 'weights or importances' },
  ], [['a', 'Approach'], ['l', 'Needs labels?'], ['f', 'Uses several features?'], ['c', 'Grouping rule'], ['i', 'Reading the result']]);

  // =====================================================================================
  // 3. WHERE
  // =====================================================================================
  ch = chapter(main, 'where', 'Where is it used?', 'Any setting where samples need to be grouped by similarity without a predefined class.');
  cards(ch, [
    { tag: 'Single-cell', title: 'Which cells belong together?', body: 'Scanpy and Seurat build a kNN graph on PCA coordinates and run Leiden or Louvain clustering. Clusters are then annotated as cell types from marker genes. This is the standard first step of a single-cell analysis.' },
    { tag: 'Tumour subtyping', title: 'Molecular subtypes', body: 'Hierarchical clustering of bulk expression defined the intrinsic breast-cancer subtypes (later assigned with the PAM50 gene set), and a consensus of several clusterings defined the consensus molecular subtypes of colorectal cancer. Subtypes were then checked against survival and drug response in independent cohorts.' },
    { tag: 'Immune phenotyping', title: 'Inflamed, excluded, desert', body: 'The example on this page: grouping tumours by immune and stromal signatures, then comparing the groups with pathology and treatment response.' },
    { tag: 'Cytometry', title: 'Gating cell populations', body: 'FlowSOM and PhenoGraph cluster millions of cells measured by flow or mass cytometry into populations, replacing manual gating on 2-D plots.' },
    { tag: 'Spatial omics', title: 'Spatial domains and niches', body: 'Spots or cells are clustered on expression plus spatial neighbours to identify tissue regions, such as tumour core, invasive edge and lymphoid aggregate.' },
    { tag: 'Microbiome', title: 'Community types', body: 'Samples are grouped by taxon abundance profiles, for example vaginal community state types or gut enterotypes, using distances designed for compositional data.' },
    { tag: 'Genomics', title: 'Gene and variant groups', body: 'Genes with similar expression across conditions are clustered into co-expression modules (hierarchical clustering with a heatmap, WGCNA). Protein sequences are clustered to remove redundancy.' },
    { tag: 'Quality control', title: 'Batches and outliers', body: 'Clustering samples on global expression often separates by batch or by failed libraries before any biology, which makes it a quick diagnostic.' },
    { tag: 'Outside biology', title: 'Customers, documents, images', body: 'Customer segmentation, topic grouping of documents and colour quantisation of images all use the same algorithms.' },
  ]);
  prose(ch, `<p><b>When not to use it:</b> labels are available and prediction is the goal (use a classifier); the groups are expected to be continuous rather than discrete, such as a differentiation trajectory (use pseudotime or a gradient score); or the data have no structure to find, because every method returns clusters whether or not groups exist (pitfalls, first card).</p>`);

  // =====================================================================================
  // 4. HOW
  // =====================================================================================
  ch = chapter(main, 'how', 'How it works', 'Six steps: distance, k-means, choosing k, hierarchical clustering, graph-based clustering, and a builder that compares them on the study.');

  // ---- Step 1: distance
  const RAWK = ['ifng', 'caf', 'cd8', 'pdl1', 'prolif', 'tmb', 'age'];
  const RLAB = { ifng: 'IFN-γ score', caf: 'CAF score', cd8: 'CD8 %', pdl1: 'PD-L1 TPS %', prolif: 'Proliferation', tmb: 'TMB (mut/Mb)', age: 'Age (years)' };
  const varOf = Object.fromEntries(RAWK.map((k) => [k, d3.variance(D, (p) => p[k])]));
  const raw4 = rowsOf(D, FSETS.four);
  const kmRaw = km(raw4, 3), ariRaw = ari(phD, kmRaw.labels);
  const pA = D.find((p) => p.phenotype === 'inflamed'), pB = D.find((p) => p.phenotype === 'desert');
  const sdK = Object.fromEntries(FSETS.four.map((k, j) => [k, S4.st.sd[j]]));
  const wrow = FSETS.four.map((k) => { const dr = FEATURES[k].get(pA) - FEATURES[k].get(pB); return { k: FEATURES[k].label, a: FEATURES[k].fmt(FEATURES[k].get(pA)), b: FEATURES[k].fmt(FEATURES[k].get(pB)), dr, ds: dr / sdK[k] }; });
  const dRaw = Math.sqrt(sum(wrow.map((r) => r.dr ** 2))), dStd = Math.sqrt(sum(wrow.map((r) => r.ds ** 2)));
  const shareRaw = wrow.map((r) => r.dr ** 2 / sum(wrow.map((q) => q.dr ** 2)));
  let rb = rung(ch, 1, 'Similarity is a distance, and scale matters',
    `<p>Each patient is a point in feature space. The <b>Euclidean distance</b> between two patients is the square root of the sum, over features, of the squared differences. Features are usually <b>standardised</b> first (subtract the mean, divide by the standard deviation) so that each contributes on a common scale. Worked example for ${pA.id} (${pA.phenotype}) and ${pB.id} (${pB.phenotype}):</p>`);
  table(rb, wrow, [['k', 'Feature'], ['a', pA.id], ['b', pB.id], ['dr', 'Difference', f1], ['ds', 'In SD units', f2], ['sq', 'Squared (SD units)', (v, r) => f2(r.ds ** 2)]], { cls: 'compact' });
  prose(rb, `<p>Distance in raw units = ${f1(dRaw)}, of which <b>${pc(Math.max(...shareRaw))}</b> of the squared total comes from ${wrow[shareRaw.indexOf(Math.max(...shareRaw))].k}, simply because it has the largest numerical range. In standard-deviation units the distance is ${f2(dStd)} and all four features count in proportion to how unusual the difference is.</p>`);
  const f1g = figure(rb, 'Which feature dominates the distance?', 'Share of the total variance (and so of the typical squared distance between two patients) contributed by each feature. With raw units a few features dominate; after standardisation each contributes equally.');
  let sc1 = 'raw';
  select(f1g.controls, { label: 'Units', value: sc1, options: [['raw', 'Raw units'], ['std', 'Standardised']], onChange: (v) => { sc1 = v; paint1(); } });
  const s1 = stats(f1g.root);
  const F1 = frame(f1g.body, { width: 820, height: 270, x: [0, 1], y: [0, 1], xLabel: 'share of total variance', axes: false, grid: false, margin: { left: 120, right: 50, top: 8, bottom: 42 } });
  F1.g.append('g').attr('class', 'axis x').attr('transform', `translate(0,${F1.h})`).call(d3.axisBottom(F1.x).ticks(5).tickFormat(d3.format('.0%')));
  const yb1 = d3.scaleBand().domain(RAWK).range([0, F1.h]).padding(0.28);
  F1.layer('l').selectAll('text').data(RAWK).join('text').attr('x', -8).attr('y', (k) => yb1(k) + yb1.bandwidth() / 2 + 4).attr('text-anchor', 'end').attr('font-size', 12).attr('fill', ink2()).text((k) => RLAB[k]);
  function paint1() {
    const tot = sum(RAWK.map((k) => varOf[k])), sh = (k) => (sc1 === 'raw' ? varOf[k] / tot : 1 / RAWK.length);
    F1.layer('b').selectAll('rect').data(RAWK).join('rect').attr('x', 0).attr('y', yb1).attr('height', yb1.bandwidth()).attr('width', (k) => F1.x(sh(k))).attr('fill', cat(0))
      .on('mousemove', (e, k) => tip(e, `${RLAB[k]}<br>variance ${varOf[k].toFixed(2)} → ${pc(sh(k))} of total`)).on('mouseleave', untip);
    F1.layer('v').selectAll('text').data(RAWK).join('text').attr('x', (k) => F1.x(sh(k)) + 5).attr('y', (k) => yb1(k) + yb1.bandwidth() / 2 + 4).attr('font-size', 11.5).attr('fill', ink()).text((k) => pc(sh(k)));
    s1.set({ 'k-means (k = 3, 4 features), ARI vs phenotype': sc1 === 'raw' ? f2(ariRaw) : f2(ari3), 'largest share': pc(d3.max(RAWK, sh)) });
  }
  paint1();
  prose(rb, `<p>The effect on the clustering is large: k-means with k = 3 on the four raw features has ARI ${f2(ariRaw)} against the phenotype, against ${f2(ari3)} after standardisation, because PD-L1 (0–100 %) outweighs the IFN-γ and CAF scores, which separate the three phenotypes better.</p>`);

  // ---- Step 2: k-means animation
  const S2 = zOf(FSETS.two), Z2 = S2.Z;
  const toRaw = (z) => [z[0] * S2.st.sd[0] + S2.st.mu[0], z[1] * S2.st.sd[1] + S2.st.mu[1]];
  rb = rung(ch, 2, 'k-means: assign, update, repeat',
    `<p>k-means starts from k <b>centroids</b> and alternates two moves: <b>assign</b> every patient to the nearest centroid, then <b>update</b> each centroid to the mean of its patients. Each move can only lower (or keep) the inertia, so the process stops when no patient changes cluster. The result depends on the starting centroids, which is why implementations run several starts and keep the lowest inertia. The figure uses two features (IFN-γ and CAF, standardised) so that every step is visible.</p>`);
  const f2g = figure(rb, 'k-means step by step', 'Left: patients coloured by their nearest centroid (large diamonds); lines join each patient to its centroid. Dashed grey diamonds mark where a centroid was before the update. Right: inertia after each move.');
  let kk = 3, initM = 'plusplus', seed2 = 1, fr = [], fi = 0, stop2 = null;
  const kSl = slider(f2g.controls, { label: 'k', min: 2, max: 6, step: 1, value: kk, fmt: String, onInput: (v) => { kk = v; reset2(); } });
  select(f2g.controls, { label: 'Initial centroids', value: initM, options: [['plusplus', 'k-means++ (spread out)'], ['random', 'random patients']], onChange: (v) => { initM = v; reset2(); } });
  button(f2g.controls, 'Step ▶', () => { halt2(); if (fi < fr.length - 1) { fi++; paint2(); } });
  button(f2g.controls, 'Run', () => run2(), true);
  button(f2g.controls, 'New start', () => { seed2++; reset2(); });
  button(f2g.controls, 'Reset', () => reset2());
  const s2 = stats(f2g.root);
  const row2 = el('div', { class: 'fig-row' }); f2g.body.append(row2);
  const c2a = el('div'), c2b = el('div'); row2.append(c2a, c2b);
  const F2 = frame(c2a, { width: 520, height: 350, x: XR, y: YR, xLabel: 'IFN-γ score', yLabel: 'CAF / TGF-β score' });
  const F2j = frame(c2b, { width: 400, height: 350, x: [0, 14], y: [0, 1], xLabel: 'move number (assign, update, assign, …)', yLabel: 'inertia (standardised units)', grid: false });
  const dots2 = F2.layer('p').selectAll('circle').data(D).join('circle').attr('r', 4).attr('stroke', css('--surface')).attr('stroke-width', 0.8)
    .attr('cx', (d) => F2.x(d.ifng)).attr('cy', (d) => F2.y(d.caf))
    .on('mousemove', (e, d) => tip(e, `${d.id} (${d.phenotype})<br>IFN-γ ${d.ifng}, CAF ${d.caf}`)).on('mouseleave', untip);
  const note2 = F2.layer('n').append('text').attr('x', 8).attr('y', 16).attr('font-size', 12.5).attr('font-weight', 700).attr('fill', ink());
  function reset2() {
    halt2();
    let Cc = kmeansInit(Z2, kk, seed2, initM), lab = null; fr = [{ C: Cc, lab: null, ph: 'init' }];
    for (let it = 0; it < 30; it++) {
      const nl = kmeansAssign(Z2, Cc), moved = lab ? nl.filter((v, i) => v !== lab[i]).length : n;
      fr.push({ C: Cc, lab: nl, ph: lab && !moved ? 'done' : 'assign', moved, J: inertia(Z2, nl, Cc) });
      if (lab && !moved) break;
      lab = nl; const Cn = kmeansUpdate(Z2, nl, Cc);
      fr.push({ C: Cn, prev: Cc, lab: nl, ph: 'update', J: inertia(Z2, nl, Cn) }); Cc = Cn;
    }
    fi = 0; paint2();
  }
  function halt2() { if (stop2) { stop2(); stop2 = null; } }
  function run2() { halt2(); if (fi >= fr.length - 1) fi = 0; let acc = 0; stop2 = animate((dt) => { acc += dt; if (acc > 650) { acc = 0; fi++; paint2(); } return fi < fr.length - 1; }); }
  function paint2() {
    const f = fr[fi];
    dots2.attr('fill', (d, i) => (f.lab ? CL(f.lab[i]) : muted())).attr('fill-opacity', 0.78);
    F2.layer('lines').selectAll('line').data(f.lab && f.ph !== 'update' ? D : []).join('line').attr('stroke', (d, i) => CL(f.lab[i])).attr('stroke-opacity', 0.35)
      .attr('x1', (d) => F2.x(d.ifng)).attr('y1', (d) => F2.y(d.caf)).attr('x2', (d, i) => F2.x(toRaw(f.C[f.lab[i]])[0])).attr('y2', (d, i) => F2.y(toRaw(f.C[f.lab[i]])[1]));
    F2.layer('prev').selectAll('path').data(f.prev || []).join('path').attr('d', d3.symbol(d3.symbolDiamond, 150)).attr('fill', 'none').attr('stroke', muted()).attr('stroke-dasharray', '3 2').attr('stroke-width', 1.6)
      .attr('transform', (c) => `translate(${F2.x(toRaw(c)[0])},${F2.y(toRaw(c)[1])})`);
    F2.layer('cen').selectAll('path').data(f.C).join('path').attr('d', d3.symbol(d3.symbolDiamond, 260)).attr('fill', (c, j) => (f.lab ? CL(j) : ink2())).attr('stroke', ink()).attr('stroke-width', 2)
      .attr('transform', (c) => `translate(${F2.x(toRaw(c)[0])},${F2.y(toRaw(c)[1])})`);
    note2.text({ init: 'start: initial centroids', assign: 'assign: nearest centroid', update: 'update: centroid = mean of its patients', done: 'converged: no patient changed cluster' }[f.ph]);
    const pts = fr.slice(0, fi + 1).filter((q) => q.J !== undefined).map((q) => [fr.indexOf(q), q.J]);
    const Jmax = Math.max(...fr.filter((q) => q.J !== undefined).map((q) => q.J)) * 1.05, Jmin = Math.min(...fr.filter((q) => q.J !== undefined).map((q) => q.J));
    F2j.y.domain([Jmin * 0.9, Jmax]); F2j.x.domain([0, Math.max(6, fr.length)]);
    F2j.g.select('.axis.y').call(d3.axisLeft(F2j.y).ticks(5)); F2j.g.select('.axis.x').call(d3.axisBottom(F2j.x).ticks(6));
    F2j.layer('j').selectAll('path').data([pts]).join('path').attr('fill', 'none').attr('stroke', ink2()).attr('stroke-width', 2).attr('d', d3.line().x((q) => F2j.x(q[0])).y((q) => F2j.y(q[1])));
    F2j.layer('jp').selectAll('circle').data(pts).join('circle').attr('r', 4.5).attr('fill', cat(0)).attr('stroke', css('--surface')).attr('stroke-width', 1.5).attr('cx', (q) => F2j.x(q[0])).attr('cy', (q) => F2j.y(q[1]))
      .on('mousemove', (e, q) => tip(e, `move ${q[0]}<br>inertia ${q[1].toFixed(2)}`)).on('mouseleave', untip);
    const last = fr[fr.length - 1];
    s2.set({ 'move': `${fi} of ${fr.length - 1}`, 'inertia': f.J !== undefined ? f.J.toFixed(2) : '—', 'patients that changed cluster': f.moved !== undefined ? f.moved : '—', 'converged inertia': fi === fr.length - 1 ? last.J.toFixed(2) : 'not yet' });
  }
  reset2();
  const finalFrames = [1, 2, 3, 4].map((sd0) => { let Cc = kmeansInit(Z2, 3, sd0, 'random'), lab = null; for (let it = 0; it < 40; it++) { const nl = kmeansAssign(Z2, Cc); if (lab && nl.every((v, i) => v === lab[i])) break; lab = nl; Cc = kmeansUpdate(Z2, nl, Cc); } return inertia(Z2, lab, Cc); });
  prose(rb, `<p>Four single random starts with k = 3 end at inertia ${finalFrames.map((v) => v.toFixed(1)).join(', ')}. ${new Set(finalFrames.map((v) => v.toFixed(1))).size > 1 ? 'Different starts settle in different local minima.' : 'On these two features all four starts reach the same solution; local minima are more frequent with more features and larger k, and "New start" shows other initialisations.'} k-means++ spreads the starting centroids, and several restarts reduce the risk.</p>`);
  quiz(rb, 'Why does k-means always find k clusters of similar size and shape, even when the real groups are not like that?', '<p>Each patient is assigned to the nearest centroid, so cluster borders are straight lines half-way between centroids, and the objective rewards compact round groups. Elongated, unequal-sized or ring-shaped groups are split or merged to fit that shape.</p>');

  // ---- Step 3: choosing k
  const KS = [2, 3, 4, 5, 6, 7, 8], kCache = {};
  const sweepK = (set) => kCache[set] || (kCache[set] = (() => { const Z = zOf(FSETS[set]).Z; return KS.map((k) => { const r = km(Z, k); return { k, J: r.inertia, sil: silhouette(Z, r.labels).mean, ari: ari(phD, r.labels), lab: r.labels }; }); })());
  rb = rung(ch, 3, 'Choosing k: elbow and silhouette',
    `<p>Inertia always falls as k increases, reaching 0 when every patient is its own cluster, so it cannot pick k by itself. The <b>elbow</b> method looks for the k beyond which the decrease flattens. The <b>silhouette</b> measures, for each patient, (b − a) / max(a, b), where a is the mean distance to the patient's own cluster and b the mean distance to the nearest other cluster; the mean over patients is high when clusters are compact and well separated. The known phenotype is not used for either criterion. The ARI is drawn only to compare with them.</p>`);
  const f3 = figure(rb, 'Elbow, silhouette and agreement with the phenotype for k = 2 to 8', 'Left: inertia (elbow plot). Right: mean silhouette and ARI against the phenotype. The larger marker is the selected k. Hover points for values.');
  let set3 = 'four', k3 = 3;
  select(f3.controls, { label: 'Features', value: set3, options: FSET_LAB, onChange: (v) => { set3 = v; paint3(); } });
  slider(f3.controls, { label: 'k', min: 2, max: 8, step: 1, value: k3, fmt: String, onInput: (v) => { k3 = v; paint3(); } });
  const s3 = stats(f3.root);
  const row3 = el('div', { class: 'fig-row' }); f3.body.append(row3);
  const c3a = el('div'), c3b = el('div'); row3.append(c3a, c3b);
  const F3a = frame(c3a, { width: 440, height: 300, x: [1.5, 8.5], y: [0, 1], xLabel: 'number of clusters k', yLabel: 'inertia (standardised units)', xTicks: 7, grid: false });
  const F3b = frame(c3b, { width: 440, height: 300, x: [1.5, 8.5], y: [0, 1], xLabel: 'number of clusters k', yLabel: 'score (0 to 1)', xTicks: 7 });
  legend(f3.body, [['inertia', ink2()], ['mean silhouette', cat(0)], ['ARI vs phenotype', cat(4)]]);
  const note3 = F3b.layer('n').append('text').attr('font-size', 11.5).attr('fill', ink2());
  function paint3() {
    const R = sweepK(set3), cur = R.find((r) => r.k === k3);
    F3a.y.domain([0, d3.max(R, (r) => r.J) * 1.05]); F3a.g.select('.axis.y').call(d3.axisLeft(F3a.y).ticks(5));
    F3a.layer('l').selectAll('path').data([R]).join('path').attr('fill', 'none').attr('stroke', ink2()).attr('stroke-width', 2).attr('d', d3.line().x((r) => F3a.x(r.k)).y((r) => F3a.y(r.J)));
    F3a.layer('p').selectAll('circle').data(R).join('circle').attr('r', (r) => (r.k === k3 ? 7 : 4.5)).attr('fill', ink2()).attr('stroke', css('--surface')).attr('stroke-width', 1.5).attr('cx', (r) => F3a.x(r.k)).attr('cy', (r) => F3a.y(r.J))
      .on('mousemove', (e, r) => tip(e, `k = ${r.k}<br>inertia ${r.J.toFixed(1)}`)).on('mouseleave', untip);
    [['sil', 0, 'silhouette'], ['ari', 4, 'ARI']].forEach(([key, c, lab]) => {
      F3b.layer('l' + key).selectAll('path').data([R]).join('path').attr('fill', 'none').attr('stroke', cat(c)).attr('stroke-width', 2).attr('stroke-dasharray', key === 'ari' ? '6 3' : null).attr('d', d3.line().x((r) => F3b.x(r.k)).y((r) => F3b.y(Math.max(0, r[key]))));
      F3b.layer('p' + key).selectAll('circle').data(R).join('circle').attr('r', (r) => (r.k === k3 ? 7 : 4.5)).attr('fill', cat(c)).attr('stroke', css('--surface')).attr('stroke-width', 1.5).attr('cx', (r) => F3b.x(r.k)).attr('cy', (r) => F3b.y(Math.max(0, r[key])))
        .on('mousemove', (e, r) => tip(e, `k = ${r.k}<br>${lab} ${f2(r[key])}`)).on('mouseleave', untip);
    });
    const bs = R.reduce((a, r) => (r.sil > a.sil ? r : a)), ba = R.reduce((a, r) => (r.ari > a.ari ? r : a));
    note3.attr('x', F3b.x(bs.k) + 8).attr('y', F3b.y(bs.sil) + 58).text(`highest silhouette: k = ${bs.k}`);
    s3.set({ 'inertia at k': cur.J.toFixed(1), 'silhouette': f2(cur.sil), 'ARI vs phenotype': f2(cur.ari), 'k with highest silhouette': bs.k, 'k with highest ARI': ba.k });
  }
  paint3();
  const R4 = sweepK('four'), R2 = sweepK('two'), bs4 = R4.reduce((a, r) => (r.sil > a.sil ? r : a)), ba4 = R4.reduce((a, r) => (r.ari > a.ari ? r : a)), bs2 = R2.reduce((a, r) => (r.sil > a.sil ? r : a));
  prose(rb, `<p>With four features the mean silhouette is highest at <b>k = ${bs4.k}</b> (${f2(bs4.sil)}) while the ARI against the phenotype is highest at <b>k = ${ba4.k}</b> (${f2(ba4.ari)}); with two features the silhouette peaks at k = ${bs2.k}. ${bs4.k !== ba4.k ? `The criteria disagree because they measure different things: the silhouette rewards the geometrically cleanest split, whereas the phenotype is a three-way biological distinction. At k = ${bs4.k} the clusters are ${compo(bs4.lab)}.` : 'Here both criteria select the same k.'} The elbow in the left panel is gradual, without a sharp bend. <b>No criterion identifies a true k; they indicate which values are defensible, and the choice is checked against biology.</b></p>`);

  // ---- Step 4: hierarchical clustering
  const MCache = {}, mergesOf = (l) => MCache[l] || (MCache[l] = agglomerate(Z4, l));
  rb = rung(ch, 4, 'Hierarchical clustering and the dendrogram',
    `<p>Agglomerative clustering starts with every patient as a cluster and repeatedly <b>merges the two closest clusters</b>. The record of merges is a tree, the <b>dendrogram</b>, where the height of a join is the distance at which it happened. <b>Cutting</b> the tree at a height gives a partition, so k does not have to be fixed in advance. The <b>linkage</b> defines the distance between two clusters: single (closest pair), complete (farthest pair), average, or Ward (smallest increase in within-cluster variance).</p>`);
  const f4 = figure(rb, 'Dendrogram of the discovery patients (four standardised features)', 'Each leaf is a patient; the coloured strip underneath is the known phenotype. Branches below the cut are coloured by cluster. The dashed line is the cut.');
  let link = 'ward', k4 = 3;
  select(f4.controls, { label: 'Linkage', value: link, options: [['ward', 'Ward'], ['average', 'Average'], ['complete', 'Complete'], ['single', 'Single']], onChange: (v) => { link = v; paint4(); } });
  slider(f4.controls, { label: 'clusters (cut position)', min: 1, max: 8, step: 1, value: k4, fmt: String, onInput: (v) => { k4 = v; paint4(); } });
  const s4 = stats(f4.root);
  const F4 = frame(f4.body, { width: 820, height: 360, x: [0, n], y: [-0.1, 1], xLabel: 'patients (ordered by the tree)', axes: false, grid: false, margin: { left: 56, right: 14, top: 10, bottom: 46 } });
  const yAx4 = F4.g.append('g').attr('class', 'axis y');
  F4.g.append('text').attr('class', 'ax-label').attr('transform', 'rotate(-90)').attr('x', -F4.h / 2).attr('y', -42).attr('text-anchor', 'middle').text('merge height (distance)');
  function paint4() {
    const M = mergesOf(link), cut = cutForK(M, k4), lab = cutTree(n, M, cut), H = M[M.length - 1].h;
    const order = (id) => (id < n ? [id] : order(M[id - n].a).concat(order(M[id - n].b)));
    const leaves = order(2 * n - 2), xs = new Array(2 * n - 1), hs = new Array(2 * n - 1).fill(0), first = new Array(2 * n - 1);
    leaves.forEach((l, i) => { xs[l] = i + 0.5; first[l] = l; });
    M.forEach((m, t) => { xs[n + t] = (xs[m.a] + xs[m.b]) / 2; hs[n + t] = m.h; first[n + t] = first[m.a]; });
    F4.y.domain([-0.09 * H, H * 1.03]);
    yAx4.call(d3.axisLeft(F4.y).tickValues(F4.y.ticks(5).filter((v) => v >= 0)));
    F4.layer('t').selectAll('path').data(M).join('path').attr('fill', 'none').attr('stroke-width', 1.5)
      .attr('stroke', (m, t) => (m.h <= cut ? CL(lab[first[n + t]]) : muted()))
      .attr('d', (m) => `M${F4.x(xs[m.a])},${F4.y(hs[m.a])}V${F4.y(m.h)}H${F4.x(xs[m.b])}V${F4.y(hs[m.b])}`)
      .on('mousemove', (e, m) => tip(e, `merge of ${m.size} patients<br>height ${m.h.toFixed(2)}`)).on('mouseleave', untip);
    F4.layer('s').selectAll('rect').data(leaves).join('rect').attr('x', (l, i) => F4.x(i) + 0.3).attr('width', F4.x(1) - F4.x(0) - 0.6).attr('y', F4.y(-0.01 * H)).attr('height', F4.y(-0.07 * H) - F4.y(-0.01 * H))
      .attr('fill', (l) => PHC(phD[l])).on('mousemove', (e, l) => tip(e, `${D[l].id}: ${phD[l]}`)).on('mouseleave', untip);
    F4.layer('c').selectAll('line').data(k4 > 1 ? [cut] : []).join('line').attr('x1', 0).attr('x2', F4.w).attr('y1', F4.y(cut)).attr('y2', F4.y(cut)).attr('stroke', ink()).attr('stroke-dasharray', '5 4').attr('stroke-width', 1.5);
    F4.layer('c').selectAll('text').data(k4 > 1 ? [cut] : []).join('text').attr('x', F4.w - 4).attr('y', F4.y(cut) - 5).attr('text-anchor', 'end').attr('font-size', 12).attr('fill', ink()).text(`cut at ${cut.toFixed(1)} → ${k4} clusters`);
    const sizes = d3.rollups(lab, (v) => v.length, (v) => v).sort((a, b) => a[0] - b[0]).map((q) => q[1]);
    s4.set({ 'clusters': k4, 'cluster sizes': sizes.join(', '), 'ARI vs phenotype': f2(ari(phD, lab)), 'tallest merge': H.toFixed(1) });
  }
  paint4();
  legend(f4.body, [...PHENO.map((p) => [p, PHC(p)]), ['cluster (below cut)', CL(0)]]);
  const aLink = Object.fromEntries(['ward', 'average', 'complete', 'single'].map((l) => { const M = mergesOf(l); return [l, ari(phD, cutTree(n, M, cutForK(M, 3)))]; }));
  const szSingle = d3.rollups(cutTree(n, mergesOf('single'), cutForK(mergesOf('single'), 3)), (v) => v.length, (v) => v).map((q) => q[1]).sort((a, b) => b - a);
  prose(rb, `<p>At three clusters the ARI against the phenotype is ${f2(aLink.ward)} for Ward, ${f2(aLink.average)} for average, ${f2(aLink.complete)} for complete and <b>${f2(aLink.single)}</b> for single linkage. Single linkage joins clusters through their closest pair, so a chain of intermediate patients links groups together; its three clusters have ${szSingle.join(', ')} patients. Hierarchical clustering needs all pairwise distances (memory grows with n²) and is practical up to a few thousand samples; for tens of thousands of cells the graph-based approach below is used.</p>`);

  // ---- Step 5: kNN graph + Louvain / Leiden
  const P4 = pca(rowsOf(D, FSETS.four)), PX = P4.S.map((s) => s[0]), PY = P4.S.map((s) => s[1]);
  const gCache = {}, edgesOf = (k) => gCache[k] || (gCache[k] = knnGraph(Z4, k));
  const GAM = [0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1, 1.5, 2, 3, 4], swCache = {};
  const sweepG = (k) => swCache[k] || (swCache[k] = GAM.map((g) => { const lab = louvain(n, edgesOf(k), g, 1); return { g, nc: Math.max(...lab) + 1, ari: ari(phD, lab) }; }));
  rb = rung(ch, 5, 'Graph-based clustering: kNN graph and Louvain / Leiden',
    `<p>Single-cell pipelines build a <b>k-nearest-neighbour (kNN) graph</b>: each sample is joined to its k most similar samples. They then search for <b>communities</b>, groups with many more edges inside than a random graph with the same degrees would give. The score is <b>modularity</b>; <b>Louvain</b> greedily moves samples between communities and merges communities until it stops improving. <b>Leiden</b> is its refinement: it guarantees that every community is internally connected and is faster. The figure uses Louvain; the <b>resolution</b> parameter plays the role of k: higher values give more, smaller communities.</p>`);
  const f5 = figure(rb, 'kNN graph, communities and the resolution parameter', 'Left: the graph drawn at each patient\'s position on the first two principal components of the four features (not a force-directed layout). Right: for the chosen number of neighbours, the ARI against the phenotype at each resolution; the number above each point is the number of communities found. Louvain with a fixed random seed.');
  let nn = 12, gam = 0.5;
  slider(f5.controls, { label: 'neighbours k', min: 4, max: 25, step: 1, value: nn, fmt: String, onInput: (v) => { nn = v; paint5(); } });
  slider(f5.controls, { label: 'resolution', min: 0, max: GAM.length - 1, step: 1, value: GAM.indexOf(0.5), fmt: (i) => String(GAM[i]), onInput: (i) => { gam = GAM[i]; paint5(); } });
  const s5 = stats(f5.root);
  const row5 = el('div', { class: 'fig-row' }); f5.body.append(row5);
  const c5a = el('div'), c5b = el('div'); row5.append(c5a, c5b);
  const F5 = frame(c5a, { width: 500, height: 340, x: d3.extent(PX).map((v, i) => v + (i ? 0.3 : -0.3)), y: d3.extent(PY).map((v, i) => v + (i ? 0.3 : -0.3)), xLabel: 'PC1', yLabel: 'PC2', grid: false });
  const F5b = frame(c5b, { width: 420, height: 340, x: [0.05, 4], y: [-0.02, 1.05], xScale: d3.scaleLog(), xLabel: 'resolution (log scale)', yLabel: 'ARI vs phenotype' });
  F5b.g.select('.axis.x').call(d3.axisBottom(F5b.x).tickValues([0.05, 0.1, 0.2, 0.5, 1, 2, 4]).tickFormat((v) => String(v)));
  function paint5() {
    const E = edgesOf(nn), lab = louvain(n, E, gam, 1), nc = Math.max(...lab) + 1;
    F5.layer('e').selectAll('line').data(E).join('line').attr('stroke', muted()).attr('stroke-opacity', 0.28).attr('x1', (e) => F5.x(PX[e[0]])).attr('y1', (e) => F5.y(PY[e[0]])).attr('x2', (e) => F5.x(PX[e[1]])).attr('y2', (e) => F5.y(PY[e[1]]));
    F5.layer('n').selectAll('circle').data(D).join('circle').attr('r', 4.4).attr('cx', (d, i) => F5.x(PX[i])).attr('cy', (d, i) => F5.y(PY[i])).attr('fill', (d, i) => CL(lab[i])).attr('stroke', css('--surface')).attr('stroke-width', 1)
      .on('mousemove', (e, d) => tip(e, `${d.id} (${d.phenotype})<br>community ${lab[D.indexOf(d)] + 1}`)).on('mouseleave', untip);
    const sw = sweepG(nn);
    F5b.layer('l').selectAll('path').data([sw]).join('path').attr('fill', 'none').attr('stroke', ink2()).attr('stroke-width', 2).attr('d', d3.line().x((r) => F5b.x(r.g)).y((r) => F5b.y(Math.max(0, r.ari))));
    F5b.layer('p').selectAll('circle').data(sw).join('circle').attr('r', (r) => (r.g === gam ? 7 : 4.5)).attr('fill', (r) => (r.g === gam ? cat(0) : ink2())).attr('stroke', css('--surface')).attr('stroke-width', 1.5).attr('cx', (r) => F5b.x(r.g)).attr('cy', (r) => F5b.y(Math.max(0, r.ari)))
      .on('mousemove', (e, r) => tip(e, `resolution ${r.g}<br>${r.nc} communities, ARI ${f2(r.ari)}`)).on('mouseleave', untip);
    F5b.layer('t').selectAll('text').data(sw).join('text').attr('x', (r) => F5b.x(r.g)).attr('y', (r) => F5b.y(Math.max(0, r.ari)) - 11).attr('text-anchor', 'middle').attr('font-size', 11.5).attr('fill', ink2()).text((r) => r.nc);
    s5.set({ 'edges': E.length, 'communities': nc, 'modularity': f2(modularity(n, E, lab, 1)), 'ARI vs phenotype': f2(ari(phD, lab)), 'community sizes': d3.rollups(lab, (v) => v.length, (v) => v).sort((a, b) => b[1] - a[1]).map((q) => q[1]).join(', ') });
  }
  paint5();
  const sw12 = sweepG(12), sw5 = sweepG(5), sw25 = sweepG(25), at = (sw, g) => sw.find((r) => r.g === g);
  prose(rb, `<p>With 12 neighbours, resolution ${at(sw12, 0.05).g} gives ${at(sw12, 0.05).nc} communities, resolution 0.5 gives ${at(sw12, 0.5).nc} and resolution 4 gives ${at(sw12, 4).nc}. The number of communities also depends on k: at resolution 1 it is ${at(sw5, 1).nc} with 5 neighbours and ${at(sw25, 1).nc} with 25. In practice the neighbour count is left near 10 to 30 and the resolution is varied, with several values compared against marker genes. Louvain is randomised: other seeds can give slightly different partitions, so results are reported with the seed fixed or stability assessed (pitfalls).</p>`);

  // ---- Step 6: builder
  rb = rung(ch, 6, 'Cluster builder: features, method and validation',
    `<p>Choose the features and a method; the clusters are found on the ${n} discovery patients (Centres A, B). Each of the ${V.length} validation patients (Centre C) is then assigned to the cluster that most of its five nearest discovery patients belong to, which tests whether the clusters generalise to a new cohort. Features are standardised.</p>`);
  const LABK = ['ifng', 'caf', 'cd8', 'pdl1', 'prolif', 'log2tmb', 'age'];
  const fl = figure(rb, 'Cluster builder', 'Left: patients on the first two principal components of the chosen features; fill = found cluster, outline = known phenotype. Right: composition of each cluster by phenotype and its response rate, in discovery and in the validation cohort. Response was never used for clustering.');
  const selK = new Set(FSETS.four), boxes = {};
  const presets = el('div', { class: 'controls' }), featRow = el('div', { class: 'controls' });
  fl.controls.before(presets); fl.controls.before(featRow);
  presets.append(el('span', { class: 'fig-caption', style: 'margin:0' }, 'Presets:'));
  [['IFN-γ + CAF', FSETS.two], ['Immune and stromal (4)', FSETS.four], ['Six features', FSETS.six], ['Age, proliferation, TMB', ['age', 'prolif', 'log2tmb']], ['All seven', LABK]].forEach(([lab, ks]) => button(presets, lab, () => { selK.clear(); ks.forEach((k) => selK.add(k)); LABK.forEach((k) => { boxes[k].checked = selK.has(k); }); paintL(); }));
  LABK.forEach((k) => { boxes[k] = checkbox(featRow, { label: FEATURES[k].label, checked: selK.has(k), onChange: (c) => { if (c) selK.add(k); else selK.delete(k); paintL(); } }); });
  let method = 'kmeans', kL = 3, nnL = 12, gamL = 0.5;
  select(fl.controls, { label: 'Method', value: method, options: [['kmeans', 'k-means'], ['ward', 'Ward hierarchical'], ['graph', 'kNN graph + Louvain']], onChange: (v) => { method = v; paintL(); } });
  slider(fl.controls, { label: 'k (k-means, Ward)', min: 2, max: 8, step: 1, value: kL, fmt: String, onInput: (v) => { kL = v; paintL(); } });
  slider(fl.controls, { label: 'neighbours (graph)', min: 4, max: 25, step: 1, value: nnL, fmt: String, onInput: (v) => { nnL = v; paintL(); } });
  slider(fl.controls, { label: 'resolution (graph)', min: 0, max: GAM.length - 1, step: 1, value: GAM.indexOf(0.5), fmt: (i) => String(GAM[i]), onInput: (i) => { gamL = GAM[i]; paintL(); } });
  const sL = stats(fl.root);
  const rowL = el('div', { class: 'fig-row' }); fl.body.append(rowL);
  const la = el('div'); rowL.append(la);
  const FL = frame(la, { width: 820, height: 340, x: [-4, 4], y: [-4, 4], xLabel: 'PC1', yLabel: 'PC2', grid: false });
  const msgL = el('div', { class: 'table-wrap' }); fl.body.append(msgL);
  legend(fl.body, [['cluster fill (cluster 1, 2, …)', CL(0)], ...PHENO.map((p) => [`outline: ${p}`, PHC(p)])]);
  const lcache = {};
  function runLab(keys, meth, kPar, nnPar, gPar) {
    const key = keys.join();
    const c = lcache[key] || (lcache[key] = (() => { const s = zOf(keys); return { ...s, pc: pca(rowsOf(D, keys)), wm: null }; })());
    let lab;
    if (meth === 'kmeans') lab = km(c.Z, kPar).labels;
    else if (meth === 'ward') { c.wm = c.wm || agglomerate(c.Z, 'ward'); lab = cutTree(n, c.wm, cutForK(c.wm, kPar)); }
    else { const k2 = 'g' + nnPar; c[k2] = c[k2] || knnGraph(c.Z, nnPar); lab = louvain(n, c[k2], gPar, 1); }
    const labV = knnTransfer(c.Z, lab, c.Zv, 5);
    return { c, lab, labV, aD: ari(phD, lab), aV: ari(phV, labV), sil: silhouette(c.Z, lab).mean };
  }
  const ranked = (lab) => d3.rollups(lab, (v) => v.length, (v) => v).sort((a, b) => b[1] - a[1] || a[0] - b[0]).map((q) => q[0]);
  function paintL() {
    const keys = LABK.filter((k) => selK.has(k));
    FL.layer('p').selectAll('*').remove();
    if (keys.length < 2) { sL.set({ note: 'select at least two features' }); msgL.innerHTML = ''; return; }
    const R = runLab(keys, method, kL, nnL, gamL), rk = ranked(R.lab), rmap = new Map(rk.map((c, i) => [c, i]));
    const S = R.c.pc.S, ex = Math.max(2.5, d3.max(S, (s) => Math.abs(s[0]))) * 1.05, ey = Math.max(2.5, d3.max(S, (s) => Math.abs(s[1]))) * 1.05;
    FL.x.domain([-ex, ex]); FL.y.domain([-ey, ey]); FL.g.select('.axis.x').call(d3.axisBottom(FL.x).ticks(5)); FL.g.select('.axis.y').call(d3.axisLeft(FL.y).ticks(5));
    FL.layer('p').selectAll('circle').data(D).join('circle').attr('cx', (d, i) => FL.x(S[i][0])).attr('cy', (d, i) => FL.y(S[i][1])).attr('r', 5).attr('fill', (d, i) => CL(rmap.get(R.lab[i])))
      .attr('fill-opacity', 0.5).attr('stroke', (d) => PHC(d.phenotype)).attr('stroke-width', 2.6)
      .on('mousemove', (e, d) => tip(e, `${d.id} (${d.phenotype})<br>cluster ${rmap.get(R.lab[D.indexOf(d)]) + 1} · ${d.response ? 'responded' : 'no response'}`)).on('mouseleave', untip);
    const rows = rk.map((c, i) => {
      const m = D.filter((d, j) => R.lab[j] === c), mv = V.filter((d, j) => R.labV[j] === c);
      return `<tr><td><i class="sw" style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${CL(i)}"></i> ${i + 1}</td><td>${m.length}</td>${PHENO.map((p) => `<td>${m.filter((d) => d.phenotype === p).length}</td>`).join('')}<td>${pc(mean(m.map((d) => d.response)))}</td><td>${mv.length}</td><td>${mv.length ? pc(mean(mv.map((d) => d.response))) : '—'}</td></tr>`;
    });
    msgL.innerHTML = `<table class="simple compact"><thead><tr><th>Cluster</th><th>n</th><th>inflamed</th><th>excluded</th><th>desert</th><th>responded (disc.)</th><th>n (valid.)</th><th>responded (valid.)</th></tr></thead><tbody>${rows.slice(0, 10).join('')}</tbody></table>${rk.length > 10 ? `<p class="fig-caption">${rk.length - 10} smaller clusters not listed.</p>` : ''}`;
    sL.set({ 'clusters': rk.length, 'silhouette (discovery)': f2(R.sil), 'ARI vs phenotype, discovery': f2(R.aD), 'ARI vs phenotype, validation': f2(R.aV), 'features': keys.length });
  }
  paintL();
  const L = (keys, m = 'kmeans', k = 3, g = 0.5, nnp = 12) => runLab(keys, m, k, nnp, g);
  const Lfour = L(FSETS.four), Ltwo = L(FSETS.two), Lsix = L(FSETS.six), Lbad = L(['age', 'prolif', 'log2tmb']), Lall = L(LABK);
  const Lk2 = L(FSETS.four, 'kmeans', 2), Lk6 = L(FSETS.four, 'kmeans', 6), Lg02 = L(FSETS.four, 'graph', 3, 0.2), Lg2 = L(FSETS.four, 'graph', 3, 2), Lw = L(FSETS.four, 'ward', 3);
  const respBy = (R) => { const rk = ranked(R.lab); return rk.map((c) => { const m = D.filter((d, j) => R.lab[j] === c); return { n: m.length, r: mean(m.map((d) => d.response)), inf: m.filter((d) => d.phenotype === 'inflamed').length / m.length }; }); };
  const rbF = respBy(Lfour), rHi = rbF.reduce((a, b) => (b.inf > a.inf ? b : a)), rLo = rbF.reduce((a, b) => (b.inf < a.inf ? b : a));
  prose(rb, '<p><b>Suggested comparisons</b>:</p>');
  table(rb, [
    { t: '<b>IFN-γ + CAF</b> → <b>four features</b> (k-means, k = 3)', s: `Validation ARI ${f2(Ltwo.aV)} → <b>${f2(Lfour.aV)}</b>; discovery ${f2(Ltwo.aD)} → ${f2(Lfour.aD)}`, m: 'CD8 % and PD-L1 are read-outs of the same immune activity that IFN-γ measures, so adding them changes the result little when IFN-γ and CAF already separate the phenotypes. Correlated features add redundancy, not new groups.' },
    { t: 'Four features → <b>six</b>, then <b>all seven</b>', s: `ARI ${f2(Lfour.aD)} → ${f2(Lsix.aD)} → <b>${f2(Lall.aD)}</b>; silhouette ${f2(Lfour.sil)} → ${f2(Lsix.sil)} → ${f2(Lall.sil)}`, m: 'Features unrelated to the phenotype add variation that is not group structure, so the groups blur. More features do not by themselves improve clustering.' },
    { t: '<b>Age, proliferation, TMB</b> only', s: `Silhouette ${f2(Lbad.sil)}, ARI vs phenotype <b>${f2(Lbad.aD)}</b>`, m: 'Clusters are still returned. They describe other structure (or noise) and do not recover the immune phenotypes, which these features do not measure.' },
    { t: 'k-means k = <b>2</b> → <b>3</b> → <b>6</b>', s: `ARI ${f2(Lk2.aD)} → <b>${f2(Lfour.aD)}</b> → ${f2(Lk6.aD)}; silhouette ${f2(Lk2.sil)} → ${f2(Lfour.sil)} → ${f2(Lk6.sil)}`, m: 'Too few clusters merge phenotypes; too many split them. ARI is only available because a reference label exists here; with new data the silhouette, stability and biology have to be used.' },
    { t: 'Graph method, resolution <b>0.2</b> → <b>2</b>', s: `${Math.max(...Lg02.lab) + 1} → <b>${Math.max(...Lg2.lab) + 1}</b> clusters; ARI ${f2(Lg02.aD)} → ${f2(Lg2.aD)}`, m: 'Resolution, not k, sets the granularity; the number of clusters follows from it.' },
    { t: 'Method: k-means vs <b>Ward</b> vs graph (4 features, k = 3)', s: `ARI ${f2(Lfour.aD)} vs ${f2(Lw.aD)} vs ${f2(L(FSETS.four, 'graph', 3, 0.5, 12).aD)}`, m: 'On compact, roughly round groups such as these the methods agree closely; they diverge when groups are elongated or touching.' },
  ], [['t', 'Try'], ['s', 'What you see'], ['m', 'What it means']], { cls: 'compact' });
  prose(rb, `<p><b>Clusters are not automatically biology.</b> With four features and k = 3, the cluster with the most inflamed tumours (${pc(rHi.inf)} of its patients) has a response rate of ${pc(rHi.r)}, and the cluster with the fewest (${pc(rLo.inf)} inflamed) has ${pc(rLo.r)}. Response tracks the clusters only because immune activity is part of the response mechanism in this simulation; response played no role in the clustering. In validation the same assignment gives ARI ${f2(Lfour.aV)}. A cluster becomes a finding once it is reproduced in an independent cohort and linked to an outcome or mechanism.</p>`);

  // =====================================================================================
  // 5. PITFALLS
  // =====================================================================================
  ch = chapter(main, 'pitfalls', 'Pitfalls', 'Common problems, how to recognise them, and how to address them.');
  const r5 = rng(5), noise = Array.from({ length: n }, () => [r5.normal(), r5.normal(), r5.normal(), r5.normal()]);
  const kNoise = km(noise, 3), silN = silhouette(noise, kNoise.labels).mean, silR = silhouette(Z4, km3.labels).mean;
  const noise2 = km(noise, 2), g0 = noise.map((r, i) => [r[0], noise2.labels[i]]);
  const a0 = g0.filter((q) => q[1] === 0).map((q) => q[0]), a1 = g0.filter((q) => q[1] === 1).map((q) => q[0]);
  const zT = (mean(a0) - mean(a1)) / Math.sqrt(d3.variance(a0) / a0.length + d3.variance(a1) / a1.length);
  const single = [...Array(20)].map((_, s) => km(Z4, 3, { nInit: 1, seed: s + 1 })), bestJ = Math.min(...single.map((r) => r.inertia));
  const nBest = single.filter((r) => r.inertia < bestJ * 1.001).length, minAri = Math.min(...single.map((r) => ari(phD, r.labels)));
  const stab = (k) => { const r = rng(11), base = km(Z4, k).labels; const v = []; for (let b = 0; b < 8; b++) { const idx = d3.range(n).map(() => r.int(n)), bs = km(idx.map((i) => Z4[i]), k, { nInit: 3, seed: b + 1 }); v.push(ari(base, kmeansAssign(Z4, bs.C))); } return mean(v); };
  const st3 = stab(3), st7 = stab(7);
  cards(ch, [
    { tag: 'Symptom: tidy clusters in featureless data', title: 'Clusters are always returned', body: `K-means with k = 3 on ${n} points of pure random noise (4 features) still gives three clusters, with a mean silhouette of ${f2(silN)}, against ${f2(silR)} for the real data. <b>Fix:</b> compare the silhouette or another quality score with the same procedure applied to shuffled or simulated data, and require the groups to reproduce.` },
    { tag: 'Symptom: tiny p-value between clusters', title: 'Testing the features used to define the clusters', body: `Splitting the noise above into two k-means clusters and testing its first feature between them gives z = ${f1(Math.abs(zT))} (p ${pFmt(pNormal(zT))}), although there is nothing but noise. The groups were chosen to differ on these features. <b>Fix:</b> test on independent data, use methods that account for the selection step, or test other variables that were not used for clustering.` },
    { tag: 'Symptom: a variable with a large range drives the groups', title: 'Unscaled features', body: `The raw four-feature distance is ${pc(Math.max(...shareRaw))} determined by one variable in the worked example, and raw k-means gave ARI ${f2(ariRaw)} against ${f2(ari3)} after standardising. <b>Fix:</b> standardise, or log-transform skewed variables first (TMB, counts), and state the choice.` },
    { tag: 'Symptom: different results on each run', title: 'Initialisation and randomness', body: `Over 20 single-start k-means runs (k = 3), ${nBest} reached the lowest inertia, and the worst had ARI ${f2(minAri)} against the phenotype. Louvain and Leiden are also randomised. <b>Fix:</b> use k-means++ with several restarts, fix and report the seed, and check how stable the clusters are across seeds.` },
    { tag: 'Symptom: silhouette says one thing, biology another', title: 'Choosing k by a single criterion', body: `The silhouette was highest at k = ${bs4.k} and the ARI at k = ${ba4.k}. <b>Fix:</b> combine the silhouette, stability and markers; report a range of k with a justification; treat the chosen k as a hypothesis.` },
    { tag: 'Symptom: clusters change when the data are resampled', title: 'Unstable clusters', body: `Re-clustering bootstrap resamples and assigning every patient to the nearest new centroid agrees with the original labels with mean ARI ${f2(st3)} at k = 3, but ${f2(st7)} at k = 7. <b>Fix:</b> measure stability by bootstrap or subsampling, and merge or drop clusters that do not reproduce.` },
    { tag: 'Symptom: one cell type split in two, or two merged', title: 'Resolution is a free parameter', body: `At 12 neighbours the graph method found ${at(sw12, 0.05).nc} to ${at(sw12, 4).nc} communities as the resolution moved from 0.05 to 4. <b>Fix:</b> inspect several resolutions against marker genes, and merge clusters that do not differ in any marker.` },
    { tag: 'Symptom: clusters follow batch, centre or sequencing depth', title: 'Technical structure', body: 'Clusters can reflect centre, batch, library size or cell-cycle phase. <b>Fix:</b> colour the clusters by every technical variable, correct or integrate batches first when appropriate, and prefer clusters that appear in several batches.' },
  ]);

  // =====================================================================================
  // 6. RECAP
  // =====================================================================================
  ch = chapter(main, 'recap', 'Recap', 'Key points.');
  recap(ch, [
    'Clustering groups samples by <b>similarity in their measurements</b>, without labels. The result depends on the features, the distance and the method.',
    '<b>Standardise</b> features so that no variable dominates the distance only because of its units.',
    '<b>k-means</b> alternates assigning samples to the nearest centroid and moving centroids to the cluster means. It finds compact, round groups and depends on its starting point.',
    '<b>Inertia, silhouette and stability</b> help to choose k. They suggest defensible values and do not identify a true number of clusters.',
    '<b>Hierarchical clustering</b> returns a tree that can be cut at any height; the linkage matters, and single linkage chains.',
    '<b>Graph-based clustering</b> (kNN graph with Louvain or Leiden) scales to many thousands of cells; the <b>resolution</b> sets the granularity and the number of clusters is a result of it.',
    'Agreement with a reference label (<b>ARI</b>) judges a clustering only when a trustworthy reference exists; otherwise rely on stability, validation in new data and biology.',
    '<b>Clusters are not automatically biology:</b> every method returns clusters, even from noise. Test them on independent data and on variables not used for clustering.',
  ]);
  prose(ch, '<h3>In code</h3><p>The Python version follows scikit-learn (patient-level data) and Scanpy (single-cell data). The R version uses base R and Seurat.</p>');
  code(ch, {
    Python: `import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score

X = StandardScaler().fit_transform(df[["ifng", "caf", "cd8", "pdl1"]])

# k-means: several k, several restarts
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
    print(k, km.inertia_, silhouette_score(X, km.labels_),
          adjusted_rand_score(df["phenotype"], km.labels_))

# hierarchical (Ward), cut into 3 clusters
lab = AgglomerativeClustering(n_clusters=3, linkage="ward").fit_predict(X)

# graph-based, as in single-cell workflows (adata has PCA computed)
import scanpy as sc
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
sc.tl.leiden(adata, resolution=0.5, random_state=0, flavor="igraph", n_iterations=2, directed=False)`,
    R: `X <- scale(df[, c("ifng", "caf", "cd8", "pdl1")])

# k-means: several restarts
km <- kmeans(X, centers = 3, nstart = 25)
km$tot.withinss
mean(cluster::silhouette(km$cluster, dist(X))[, "sil_width"])

# hierarchical (Ward), cut into 3 clusters
hc  <- hclust(dist(X), method = "ward.D2")
lab <- cutree(hc, k = 3)
mclust::adjustedRandIndex(df$phenotype, lab)

# graph-based, as in Seurat (obj has PCA computed)
obj <- FindNeighbors(obj, dims = 1:30, k.param = 20)
obj <- FindClusters(obj, resolution = 0.5, algorithm = 4)   # 4 = Leiden`,
  });
  seeAlso(main, [['concepts', 'pca', 'PCA'], ['concepts', 'tsne-umap', 't-SNE and UMAP'], ['concepts', 'multiple-testing', 'Multiple testing and FDR'], ['omics', 'sc-clustering', 'Clustering and cell-type annotation']]);
}
