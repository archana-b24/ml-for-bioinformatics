// QC, normalisation and batch effects (lesson). Merges the archived bulk-qc and bulk-batch pages.
// Arc: what → why → where → how it works (6 steps: log-CPM → outlier rules → PCA shows the centre → partial
//      confounding → naive vs protected correction → correction lab with try/see/means table) → pitfalls → recap.
// Data: data/omics/bulk_qc.json + bulk_batch.json (scripts/build_bulk.py) and bulk_qcbatch_lab.json
// (scripts/build_qc_batch_lesson.py: 150 genes × 177 patients, uncorrected log-CPM and the noise-free truth).
// Batch correction, PCA and silhouettes run live in the browser; every number in the prose is computed here.
import { header, figure, slider, select, button, checkbox, stats, legend, seeAlso, animate, tip, untip, el } from '../lib/ui.js';
import { frame, cat, css, ink, ink2, muted } from '../lib/plot.js';
import { rng, solve, mean } from '../lib/data.js';
import { auc } from '../lib/models.js';
import { lessonNav, chapter, prose, rung, worked, quiz, cards, table, code, recap } from '../lib/lesson.js';
import { stepCard, omicsData } from './_shared.js';
import { PHENOS, phenoColor, CENTRES, centreColor, respColor, median, mad } from './_bulk_common.js';
const d3 = window.d3;

const f2 = (v) => (Math.abs(v) < 0.005 ? '0.00' : v.toFixed(2));
const f3 = (v) => v.toFixed(3);
const pc = (v) => Math.round(100 * v) + '%';
const mReads = (v) => (v / 1e6).toFixed(1) + 'M';
const sum = (a) => a.reduce((s, v) => s + v, 0);
const corr = (a, b) => { const ma = mean(a), mb = mean(b); let s = 0, sa = 0, sb = 0; a.forEach((v, i) => { s += (v - ma) * (b[i] - mb); sa += (v - ma) ** 2; sb += (b[i] - mb) ** 2; }); return s / Math.sqrt(sa * sb); };

const METRICS = {
  medcor: { name: 'Median correlation to other samples', short: 'median correlation', side: 'low', tr: (v) => v, inv: (v) => v, fmt: (v) => v.toFixed(3), log: false, kind: 'statistics' },
  rdist: { name: 'Robust PCA distance (MCD Mahalanobis)', short: 'robust distance', side: 'high', tr: Math.log, inv: Math.exp, fmt: (v) => v.toFixed(1), log: true, kind: 'robust statistics' },
  iso: { name: 'Isolation-forest anomaly score', short: 'isolation forest', side: 'high', tr: (v) => v, inv: (v) => v, fmt: (v) => v.toFixed(3), log: false, kind: 'unsupervised ML' },
  lib: { name: 'Library size (total reads)', short: 'library size', side: 'low', tr: Math.log10, inv: (v) => 10 ** v, fmt: mReads, log: true, kind: 'statistics' },
};
const VNAME = { none: 'no correction', naive: 'naive (centre only)', protected: 'protected (centre, keeping phenotype)' };

export default async function render(main) {
  header(main, 'QC, normalisation and batch effects',
    'Before any biological question can be asked of an RNA-seq matrix, failed libraries have to be found, samples sequenced to different depths have to be put on one scale, and differences caused by <b>where and how</b> samples were processed have to be separated from differences caused by <b>biology</b>. The last step is the hard one when the two are entangled.');
  lessonNav(main);
  stepCard(main, {
    stage: 'Bulk RNA-seq · step 1 of 4',
    question: 'Which tumour libraries failed, how are samples of different sequencing depth made comparable, and how is the sequencing-centre effect removed without removing biology?',
    data: 'Raw counts for the 180 discovery patients (Centres A and B) and 1,933 genes, with per-sample QC metrics, the pathologist\'s immune phenotype and response.',
    method: 'log-CPM normalisation; median ± k·MAD rules on QC metrics; PCA for exploration; a per-gene linear model (ComBat-style location/scale) to remove the centre shift, with or without biological covariates protected.',
    verdict: 'stats', why: ' Every decision on this page is a summary statistic, a threshold or a per-gene linear model; nothing is predicted. The one machine-learning tool (an isolation-forest outlier score) is optional, ranks samples well, and has no natural cut-off.',
  });
  const [Q, B, L] = await Promise.all([omicsData(main, 'bulk_qc.json'), omicsData(main, 'bulk_batch.json'), omicsData(main, 'bulk_qcbatch_lab.json')]);
  if (!Q || !B || !L) return;

  // ---------------------------------------------------------------- shared numbers: QC
  const NQ = Q.ids.length, planted = Object.entries(Q.planted).map(([what, id]) => ({ what, id, i: Q.ids.indexOf(id) }));
  const plantedOf = (i) => planted.find((p) => p.i === i);
  const flags = (key, k) => {
    const M = METRICS[key], v = Q[key].map(M.tr), m = median(v), s = mad(v), thr = M.side === 'low' ? m - k * s : m + k * s;
    return { raw: M.inv(thr), f: v.map((x) => (M.side === 'low' ? x < thr : x > thr)) };
  };
  const flagged = (key, k) => flags(key, k).f.map((f, i) => (f ? Q.ids[i] : null)).filter(Boolean);
  const K0 = Q.k_default, ND = Q.norm_demo, loglib = ND.lib.map(Math.log10);
  const medCorr = (tr) => corr(ND[tr].map((q) => q[2]), loglib);
  const libMed = median(Q.lib), lowId = Q.planted['low depth'], lowLib = Q.lib[Q.ids.indexOf(lowId)];

  // ---------------------------------------------------------------- shared engine: batch correction on the lab matrix
  const N = B.ids.length, G = L.genes.length, X = L.X, T = L.T;
  const bv = B.centre.map((c) => (c === 'Centre B' ? 1 : 0)), inf = B.pheno.map((p) => p === 'inflamed'), y = B.resp;
  const nB = (c) => sum(Object.values(B.counts[c])), shareInf = (c) => B.counts[c].inflamed / nB(c);
  const colOf = (M, j) => M.map((r) => r[j]);
  const ifngOf = (M) => M.map((r) => mean(L.ifng_idx.map((j) => r[j])));
  const gapOf = (s) => mean(s.filter((v, i) => inf[i])) - mean(s.filter((v, i) => !inf[i]));
  const diffBA = (s) => mean(s.filter((v, i) => bv[i])) - mean(s.filter((v, i) => !bv[i]));
  const trueEff = d3.range(G).map((j) => gapOf(colOf(T, j)));
  const design = (prot) => d3.range(N).map((i) => {
    const r = [1, bv[i]];
    if (prot.pheno) r.push(B.pheno[i] === 'excluded' ? 1 : 0, B.pheno[i] === 'desert' ? 1 : 0);
    if (prot.resp) r.push(y[i]);
    return r;
  });
  /** Least-squares coefficients of every column of Y on design Dm. Returns {beta: per column, fitted(i, j)}. */
  function lsFit(Dm, cols) {
    const k = Dm[0].length, DtD = d3.range(k).map((a) => d3.range(k).map((b) => sum(Dm.map((r) => r[a] * r[b]))));
    const inv = d3.range(k).map((a) => solve(DtD, d3.range(k).map((b) => +(a === b))));
    const P = inv.map((row) => Dm.map((r) => sum(r.map((v, b) => v * row[b]))));          // k × N (inv symmetric)
    return cols.map((c) => P.map((p) => sum(p.map((v, i) => v * c[i]))));
  }
  /** mode: 'none' | 'loc' (subtract centre coefficient) | 'locscale' (also match residual spread, as ComBat). */
  function correct(prot, mode) {
    if (mode === 'none') return X;
    const Dm = design(prot), cols = d3.range(G).map((j) => colOf(X, j)), betas = lsFit(Dm, cols);
    const M = X.map((r) => r.slice());
    betas.forEach((be, j) => {
      const fit = Dm.map((r) => sum(r.map((v, b) => v * be[b]))), res = cols[j].map((v, i) => v - fit[i]);
      const sdOf = (g) => d3.deviation(res.filter((v, i) => bv[i] === g));
      const ratio = mode === 'locscale' ? sdOf(0) / sdOf(1) : 1;
      for (let i = 0; i < N; i++) if (bv[i]) M[i][j] = fit[i] - be[1] + res[i] * ratio;
    });
    return M;
  }
  /** Top-2 PCA by power iteration on the gene covariance. Signs fixed: PC1 rises with the IFN-γ score, PC2 with Centre B. */
  function pca2(M) {
    const mu = d3.range(G).map((j) => mean(colOf(M, j))), Z = M.map((r) => r.map((v, j) => v - mu[j]));
    const C = d3.range(G).map(() => new Float64Array(G));
    Z.forEach((r) => { for (let a = 0; a < G; a++) { const ra = r[a]; if (!ra) continue; const Ca = C[a]; for (let b = a; b < G; b++) Ca[b] += ra * r[b]; } });
    for (let a = 0; a < G; a++) for (let b = 0; b < a; b++) C[a][b] = C[b][a];
    const tot = d3.range(G).reduce((s, a) => s + C[a][a], 0), vecs = [], vals = [];
    for (let c = 0; c < 2; c++) {
      let v = d3.range(G).map((j) => 1 + 0.01 * j), lam = 0;
      for (let it = 0; it < 200; it++) {
        const w = d3.range(G).map((a) => { let s = 0; for (let b = 0; b < G; b++) s += C[a][b] * v[b]; return s; });
        vecs.forEach((u, q) => { const d = sum(u.map((x, j) => x * v[j])); w.forEach((x, a) => { w[a] -= vals[q] * d * u[a]; }); });
        const nrm = Math.hypot(...w); v = w.map((x) => x / nrm);
        if (Math.abs(nrm - lam) < 1e-9 * nrm) break; lam = nrm;
      }
      vecs.push(v); vals.push(lam);
    }
    const S = Z.map((r) => vecs.map((v) => sum(r.map((x, j) => x * v[j]))));
    const ifs = ifngOf(M);
    [corr(S.map((s) => s[0]), ifs), corr(S.map((s) => s[1]), bv)].forEach((c, q) => { if (c < 0) S.forEach((s) => { s[q] = -s[q]; }); });
    return { S, var: vals.map((l) => l / tot) };
  }
  function silhouette(S, lab) {
    const d = (a, b) => Math.hypot(S[a][0] - S[b][0], S[a][1] - S[b][1]), groups = [...new Set(lab)];
    return mean(d3.range(N).map((i) => {
      const md = Object.fromEntries(groups.map((g) => [g, []]));
      for (let k = 0; k < N; k++) if (k !== i) md[lab[k]].push(d(i, k));
      const a = mean(md[lab[i]]), b = d3.min(groups.filter((g) => g !== lab[i]), (g) => mean(md[g]));
      return (b - a) / Math.max(a, b);
    }));
  }
  const cache = new Map();
  function evaluate(prot, mode) {
    const key = `${+!!prot.pheno}${+!!prot.resp}${mode}`;
    if (cache.has(key)) return cache.get(key);
    const M = correct(prot, mode), P = pca2(M), s = ifngOf(M);
    const eff = d3.range(G).map((j) => gapOf(colOf(M, j)));
    const r = { M, S: P.S, var: P.var, ifng: s, gap: gapOf(s), auc: auc(s, y), eff,
      r2: corr(P.S.map((q) => q[0]), bv) ** 2, silB: silhouette(P.S, B.centre), silP: silhouette(P.S, B.pheno),
      rmse: Math.sqrt(mean(eff.map((e, j) => (e - trueEff[j]) ** 2))) };
    cache.set(key, r);
    return r;
  }
  const VER = { none: () => evaluate({}, 'none'), naive: () => evaluate({}, 'locscale'), protected: () => evaluate({ pheno: 1 }, 'locscale') };
  const E0 = VER.none(), EN = VER.naive(), EP = VER.protected(), truthGap = B.truth_effect;

  // =====================================================================================
  // 1. WHAT
  // =====================================================================================
  let ch = chapter(main, 'what', 'What are QC, normalisation and batch correction?', 'In one line: <b>remove broken samples → put the rest on one scale → separate technical from biological variation</b>.');
  prose(ch, `<p>These are the first three steps of almost every bulk RNA-seq analysis. <b>Quality control (QC)</b> identifies libraries whose measurements cannot be trusted. <b>Normalisation</b> removes the effect of sequencing depth, so that a gene's value reflects its share of the transcriptome rather than how many reads the sample received. <b>Batch correction</b> removes systematic differences between groups of samples processed differently (here, two sequencing centres), while keeping differences that reflect biology.</p>
    <p>The diagram follows the ${NQ} discovery tumours of the study through the pipeline used on this page.</p>`);
  ch.append(pipelineSvg(Q, B, E0, EP));
  prose(ch, '<p>Key terms:</p>');
  table(ch, [
    { t: 'Library size', m: 'Total number of reads assigned to genes in one sample', s: `median ${mReads(libMed)}; lowest ${mReads(lowLib)} (${lowId})` },
    { t: 'CPM / log-CPM', m: 'Counts per million reads; log-CPM = log2(CPM + 1)', s: 'the scale used for PCA, QC and correction here' },
    { t: 'MAD', m: 'Median absolute deviation, a spread measure that a few outliers cannot inflate', s: `samples beyond median ± ${K0}·MAD are flagged` },
    { t: 'Outlier vs failure', m: 'An outlier is unusual; a failure is technically broken. Not every outlier is a failure', s: `${planted.length} failures were planted in the simulation` },
    { t: 'PCA', m: 'Principal component analysis: the few directions that capture most variation across genes', s: `PC1 = ${pc(E0.var[0])} of variance (150 most variable genes)` },
    { t: 'Batch effect', m: 'A systematic technical difference between groups of samples (centre, run, kit, date)', s: 'Centre B shifts each gene by its own factor' },
    { t: 'Confounding', m: 'Batch and biology are correlated, so their effects are mixed in the raw data', s: `inflamed: ${pc(shareInf('Centre B'))} of Centre B vs ${pc(shareInf('Centre A'))} of Centre A` },
    { t: 'Protected covariate', m: 'A biological variable kept in the correction model so that its differences are not removed', s: 'the pathologist\'s immune phenotype' },
  ], [['t', 'Term'], ['m', 'Meaning'], ['s', 'In our study']]);

  // =====================================================================================
  // 2. WHY
  // =====================================================================================
  ch = chapter(main, 'why', 'Why do we need it?', 'Three technical effects that are larger than many biological ones.');
  const strong = planted.find((p) => p.what === 'degraded (strong)');
  prose(ch, `<ol>
    <li><b>Depth differences.</b> Library sizes range from ${mReads(d3.min(Q.lib))} to ${mReads(d3.max(Q.lib))}. On raw counts, a sample's median gene value correlates with its library size at r = <b>${f2(medCorr('raw'))}</b>: deeper samples look as if every gene were more highly expressed.</li>
    <li><b>Failed libraries.</b> A degraded sample (${strong.id}) has a median correlation of ${f3(Q.medcor[strong.i])} with the other samples, against a cohort median of ${f3(median(Q.medcor))}. Left in, it can dominate a PCA or create a false differential-expression result.</li>
    <li><b>Batch effects.</b> Before correction, the first principal component explains ${pc(E0.var[0])} of the variation and its R² with sequencing centre is <b>${f2(E0.r2)}</b>. The largest signal in the data is where the sample was sequenced.</li></ol>
    <p>Batch correction itself has a failure mode. Because Centre B enrolled more inflamed tumours, part of the difference between centres is biology. The table compares the options on the IFN-γ module score, whose true inflamed-versus-other gap in the simulation is ${f2(truthGap)} log2 units.</p>`);
  table(ch, [
    { a: 'No correction', c: `no (silhouette by centre ${f2(E0.silB)})`, b: `gap ${f2(E0.gap)}, but per-gene effects biased (error ${f2(E0.rmse)})`, u: 'not for PCA or clustering across centres' },
    { a: 'Naive correction (centre only)', c: `yes (${f2(EN.silB)})`, b: `<b>no</b>: gap falls to ${f2(EN.gap)}`, u: 'only when batch and biology are balanced' },
    { a: '<b>Protected correction</b>', c: `<b>yes (${f2(EP.silB)})</b>`, b: `<b>yes</b>: gap ${f2(EP.gap)}, per-gene error ${f2(EP.rmse)}`, u: '<b>PCA, clustering, visualisation</b>' },
    { a: 'Centre as a covariate in the downstream model', c: 'accounted for, data unchanged', b: 'yes, with correct uncertainty', u: 'differential expression, survival and other regression models' },
  ], [['a', 'Approach'], ['c', 'Removes centre?'], ['b', 'Keeps biology?'], ['u', 'Use for']]);

  // =====================================================================================
  // 3. WHERE
  // =====================================================================================
  ch = chapter(main, 'where', 'Where is it used?', 'Every high-throughput assay in which samples are processed in groups.');
  cards(ch, [
    { tag: 'Bulk RNA-seq', title: 'Multi-centre cohorts', body: 'Sequencing centre, library kit and RNA extraction batch are the usual batches. Workflow: MultiQC and RSeQC metrics, log-CPM or VST, PCA, then ComBat-seq or a batch term in the DESeq2 / edgeR / limma design.' },
    { tag: 'Public data', title: 'Combining TCGA, GEO and in-house data', body: 'Studies differ in platform, protocol and patient mix. Meta-analyses either correct (ComBat with study as batch) or analyse each study separately and combine the effect sizes.' },
    { tag: 'Microarrays', title: 'Scan date and chip lot', body: 'RMA normalisation followed by ComBat originated here. Scan date is often the strongest batch variable in older expression and methylation array data.' },
    { tag: 'DNA methylation', title: 'Plate, chip and position', body: 'Illumina EPIC arrays show plate and chip-position effects. Functional normalisation and ComBat with plate as batch are standard in epigenome-wide association studies.' },
    { tag: 'Proteomics and metabolomics', title: 'Run-order drift', body: 'Mass-spectrometry intensity drifts within and between runs. Pooled QC samples injected at intervals are used to model and remove the drift (e.g. LOESS on QC samples).' },
    { tag: 'Single-cell', title: 'Integration', body: 'Donor, chip and protocol effects are removed in a shared embedding (Harmony, scVI) rather than in the counts. The same principle applies: biology that differs between batches must not be integrated away.' },
    { tag: 'Clinical trials', title: 'Central vs local laboratories', body: 'Biomarker assays run at several sites are harmonised with bridging samples measured at every site, so that a cut-off means the same in each laboratory.' },
    { tag: 'Sample QC', title: 'Before any model', body: 'Sample swaps (sex genes vs recorded sex), degraded RNA (3′ bias), contamination and low depth are checked before analysis, and the exclusion rule is written down before outcomes are examined.' },
  ]);
  prose(ch, `<p><b>When not to correct the data:</b> for differential expression or any regression, put the batch in the model as a covariate instead (the data stay unchanged and the uncertainty is counted correctly); when batch and the variable of interest are <b>completely</b> confounded (every responder from one centre), no method can separate them; when a batch variable is unknown, estimate hidden factors (SVA, RUVseq) instead of guessing; and never fit a correction on samples that will later serve as a test set.</p>`);

  // =====================================================================================
  // 4. HOW
  // =====================================================================================
  ch = chapter(main, 'how', 'How it works', 'Six steps, from raw counts to a batch correction checked against the known truth.');

  // ---- Step 1: library size and log-CPM
  const ex50 = 50, cLow = ex50 * lowLib / 1e6, cMed = ex50 * libMed / 1e6;
  let rb = rung(ch, 1, 'Library size and log-CPM',
    `<p>A count depends on how much a gene is expressed <i>and</i> on how many reads the library received. Dividing by the library size and multiplying by 10⁶ gives <b>counts per million (CPM)</b>. A gene at 50 CPM has about ${Math.round(cLow)} reads in ${lowId} (${mReads(lowLib)} reads) and about ${Math.round(cMed)} reads in a median sample (${mReads(libMed)}). Taking <b>log2(CPM + 1)</b> then makes fold changes additive and reduces the dependence of the variance on the mean, which PCA and linear models require.</p>`);
  const fN = figure(rb, 'One sample per box, on three scales',
    `${ND.ids.length} samples ordered by sequencing depth (left = shallowest). Each box shows the distribution of gene values in one sample: whiskers at the 10th and 90th percentiles, box at the 25th–75th, white line at the median. Hover a box.`);
  let trN = 'raw';
  const TRN = { raw: ['raw counts', 'log10(count + 1)'], cpm: ['counts per million', 'log10(CPM + 1)'], logcpm: ['log-CPM', 'log2(CPM + 1)'] };
  select(fN.controls, { label: 'Scale', value: trN, options: Object.entries(TRN).map(([k, v]) => [k, v[0]]), onChange: (v) => { trN = v; drawN(); } });
  const sN = stats(fN.root);
  const wN = el('div'); fN.body.append(wN);
  legend(fN.body, [['sample', cat(0)], [`${lowId} (planted low depth)`, cat(7)]]);
  function drawN() {
    wN.innerHTML = '';
    const Qs = ND[trN], F = frame(wN, { width: 820, height: 280, x: [0, 1], y: d3.extent(Qs.flat()), yLabel: TRN[trN][1], xLabel: 'sample, ordered by library size →', xScale: d3.scaleBand().padding(0.25) });
    F.x.domain(ND.ids); F.y.nice();
    F.g.select('.axis.x').call(d3.axisBottom(F.x).tickValues([])); F.g.select('.axis.y').call(d3.axisLeft(F.y).ticks(5));
    const bw = F.x.bandwidth();
    const g = F.layer('box').selectAll('g').data(ND.ids.map((id, i) => ({ id, i, q: Qs[i] }))).join('g').attr('transform', (d) => `translate(${F.x(d.id)},0)`);
    g.append('line').attr('x1', bw / 2).attr('x2', bw / 2).attr('y1', (d) => F.y(d.q[0])).attr('y2', (d) => F.y(d.q[4])).attr('stroke', ink2()).attr('stroke-width', 1.2);
    g.append('rect').attr('width', bw).attr('y', (d) => F.y(d.q[3])).attr('height', (d) => F.y(d.q[1]) - F.y(d.q[3])).attr('rx', 2).attr('fill', (d) => (d.id === lowId ? cat(7) : cat(0))).attr('opacity', 0.85);
    g.append('line').attr('x1', 0).attr('x2', bw).attr('y1', (d) => F.y(d.q[2])).attr('y2', (d) => F.y(d.q[2])).attr('stroke', css('--surface')).attr('stroke-width', 2);
    g.append('rect').attr('width', bw).attr('height', F.h).attr('fill', 'transparent')
      .on('mousemove', (e, d) => tip(e, `<b>${d.id}</b>${d.id === lowId ? ' (planted low depth)' : ''}<br>${mReads(ND.lib[d.i])} reads<br>median ${f2(d.q[2])}`)).on('mouseleave', untip);
    const med = Qs.map((q) => q[2]);
    sN.set({ 'SD of sample medians': f3(d3.deviation(med)), 'corr(median, log library size)': f2(medCorr(trN)), 'genes passing expression filter': `${Q.n_genes_kept.toLocaleString()} of ${Q.n_genes.toLocaleString()}` });
  }
  drawN();
  const lowBox = ND.cpm[ND.ids.indexOf(lowId)], otherIQR = median(ND.cpm.filter((q, i) => ND.ids[i] !== lowId).map((q) => q[3] - q[1]));
  prose(rb, `<p>On raw counts the medians climb with depth (r = ${f2(medCorr('raw'))}); after CPM the correlation is ${f2(medCorr('cpm'))}. ${lowId} lines up with the others after CPM (interquartile range ${f2(lowBox[3] - lowBox[1])} vs a typical ${f2(otherIQR)} on the log10 scale), but its values rest on far fewer reads: dividing by depth rescales the counts but cannot recover information that was never sequenced. Real pipelines refine CPM with TMM (edgeR) or median-of-ratios (DESeq2) size factors, which are robust to a few very highly expressed genes.</p>`);

  // ---- Step 2: outlier rules
  rb = rung(ch, 2, 'Finding failed libraries with median ± k·MAD',
    `<p>For each sample, compute a QC metric, for example the median correlation of its log-CPM profile with every other sample. A failed library does not resemble the others. A sample is flagged when it lies more than <b>k</b> robust standard deviations (1.4826 × MAD) from the cohort median. The median and MAD are used instead of the mean and SD because the failures themselves would otherwise inflate the spread and hide.</p>`);
  const fO = figure(rb, 'Outlier rules on four QC metrics',
    'Left: one dot per sample on the chosen metric (vertical position is random, for visibility); the dashed line is the threshold. Right: the same samples on the first two principal components of log-CPM (1,000 most variable genes). Ringed dots are the three failures planted in the simulation. Hover a dot.');
  let keyO = 'medcor', kO = K0;
  select(fO.controls, { label: 'QC metric', value: keyO, options: Object.entries(METRICS).map(([k, m]) => [k, m.name]), onChange: (v) => { keyO = v; drawO(); } });
  slider(fO.controls, { label: 'k (MADs)', min: 1.5, max: 6, step: 0.5, value: kO, fmt: (v) => v.toFixed(1), onInput: (v) => { kO = v; drawO(); } });
  const sO = stats(fO.root);
  const rowO = el('div', { class: 'fig-row' }); fO.body.append(rowO);
  const oL = el('div'), oR = el('div'); rowO.append(oL, oR);
  legend(fO.body, [['flagged by this rule', cat(7)], ['passes', muted()], ['planted failure (ring)', ink()]]);
  const rj = rng(3), jit = Q.ids.map(() => rj());
  const pcx = Q.pc.map((p) => p[0]), pcy = Q.pc.map((p) => p[1]);
  const FR = frame(oR, { width: 470, height: 330, x: d3.extent(pcx), y: d3.extent(pcy), xLabel: `PC1 (${pc(Q.pcvar[0])} of variance)`, yLabel: `PC2 (${pc(Q.pcvar[1])})` });
  FR.x.nice(); FR.y.nice(); FR.g.select('.axis.x').call(d3.axisBottom(FR.x).ticks(5)); FR.g.select('.axis.y').call(d3.axisLeft(FR.y).ticks(5));
  const hovO = (e, i) => tip(e, `<b>${Q.ids[i]}</b> · ${Q.centre[i]}${plantedOf(i) ? ` · <b>planted: ${plantedOf(i).what}</b>` : ''}<br>library ${mReads(Q.lib[i])} reads<br>median correlation ${f3(Q.medcor[i])}<br>robust distance ${Q.rdist[i].toFixed(1)}<br>isolation score ${f3(Q.iso[i])}`);
  const ordO = d3.range(NQ).sort((a, b) => !!plantedOf(a) - !!plantedOf(b));
  const pcDots = FR.layer('pts').selectAll('circle').data(ordO).join('circle').attr('cx', (i) => FR.x(pcx[i])).attr('cy', (i) => FR.y(pcy[i])).on('mousemove', hovO).on('mouseleave', untip);
  const LOFF = { 'degraded (strong)': [0, 20, 'middle'], 'degraded (mild)': [8, -10, 'start'] };
  FR.layer('lab').selectAll('text').data(planted).join('text').attr('x', (p) => FR.x(pcx[p.i]) + (LOFF[p.what] || [9])[0]).attr('y', (p) => FR.y(pcy[p.i]) + (LOFF[p.what] || [0, 4])[1])
    .attr('text-anchor', (p) => (LOFF[p.what] || [0, 0, 'start'])[2]).attr('font-size', 11).attr('fill', ink2()).text((p) => p.id);
  function drawO() {
    const M = METRICS[keyO], F = flags(keyO, kO), v = Q[keyO];
    oL.innerHTML = '';
    const FL = frame(oL, { width: 470, height: 330, x: d3.extent([...v, F.raw]), y: [0, 1], xLabel: M.name + (M.log ? ' (log scale)' : ''), xScale: M.log ? d3.scaleLog() : d3.scaleLinear(), grid: false });
    FL.x.nice(); FL.g.select('.axis.x').call(d3.axisBottom(FL.x).ticks(5, M.log ? '~s' : '.3~f')); FL.g.select('.axis.y').remove();
    const left = FL.x(F.raw) < FL.w / 2;
    FL.layer('thr').append('line').attr('x1', FL.x(F.raw)).attr('x2', FL.x(F.raw)).attr('y1', 0).attr('y2', FL.h).attr('stroke', ink()).attr('stroke-dasharray', '5 4').attr('stroke-width', 1.5);
    FL.layer('thr').append('text').attr('x', FL.x(F.raw) + (left ? 6 : -6)).attr('y', 12).attr('text-anchor', left ? 'start' : 'end').attr('font-size', 11).attr('fill', ink2())
      .text(`median ${M.side === 'low' ? '−' : '+'} ${kO.toFixed(1)}·MAD = ${M.fmt(F.raw)}`);
    const style = (s) => s.attr('r', (i) => (plantedOf(i) ? 6.5 : 4.5)).attr('fill', (i) => (F.f[i] ? cat(7) : muted()))
      .attr('stroke', (i) => (plantedOf(i) ? ink() : css('--surface'))).attr('stroke-width', (i) => (plantedOf(i) ? 2 : 1.2));
    style(FL.layer('pts').selectAll('circle').data(ordO).join('circle').attr('cx', (i) => FL.x(v[i])).attr('cy', (i) => FL.y(0.08 + 0.78 * jit[i])).on('mousemove', hovO).on('mouseleave', untip));
    FL.layer('lab').selectAll('text').data(planted).join('text').attr('x', (p) => FL.x(v[p.i])).attr('y', (p) => FL.y(0.08 + 0.78 * jit[p.i]) - 10).attr('text-anchor', 'middle').attr('font-size', 11).attr('fill', ink2()).text((p) => p.id);
    style(pcDots);
    const nf = sum(F.f), caught = planted.filter((p) => F.f[p.i]).length;
    sO.set({ 'samples flagged': nf, 'planted failures caught': `${caught} of ${planted.length}`, 'false alarms': nf - caught, 'method family': M.kind });
  }
  drawO();
  const libF = flagged('lib', K0), isoF = flagged('iso', K0).filter((id) => !planted.some((p) => p.id === id)), cor2 = flagged('medcor', 2);
  const corF = flagged('medcor', K0), corCaught = planted.filter((p) => corF.includes(p.id)).length;
  prose(rb, `<p>At k = ${K0}, median correlation flags ${corF.length} samples and catches ${corCaught} of the ${planted.length} planted failures. At k = 2 it flags ${cor2.length}: a small k turns ordinary biological spread into "outliers". Library size at k = ${K0} flags ${libF.join(', ') || 'no samples'}; a shallow library is not necessarily a failed one, and normalisation handles depth. The isolation forest also flags ${isoF.length} healthy sample${isoF.length === 1 ? '' : 's'} (${isoF.join(', ') || 'none'}): the PCA contains two clouds (the two centres, step 3), and the edges of each cloud look anomalous to a multivariate detector. The study excludes a sample only when an expression-based rule fires, which removes ${Q.excluded.join(', ')} and leaves ${N} patients.</p>`);
  quiz(rb, `${lowId} has the lowest library size. Why is low depth on its own not a reason to exclude a sample?`,
    `<p>CPM normalisation corrects for depth, and a shallow library with a normal expression profile still measures the same biology, only with more noise for weakly expressed genes. ${lowId} is excluded here because its expression profile is abnormal (median correlation ${f3(Q.medcor[Q.ids.indexOf(lowId)])}), not because of its read count.</p>`);

  // ---- Step 3: PCA shows the centre
  rb = rung(ch, 3, 'PCA shows the sequencing centre first',
    `<p>After QC, the ${N} log-CPM profiles are summarised by PCA on the 150 most variable genes. Colouring the same plot by each known variable, technical and biological, is the standard first look for batch effects.</p>`);
  const fP = figure(rb, 'Uncorrected PCA, coloured by each known variable',
    'Each dot is a patient. Silhouette (−1 to 1) measures how well the chosen groups separate on PC1–PC2: near 0 means mixed, near 1 means fully separated. Hover a dot.');
  let colP = 'centre';
  const COLS = { centre: [(i) => centreColor(B.centre[i]), CENTRES.map((c) => [c, centreColor(c)])], pheno: [(i) => phenoColor(B.pheno[i]), PHENOS.map((p) => [p, phenoColor(p)])], resp: [(i) => respColor(y[i]), [['responder', respColor(1)], ['non-responder', respColor(0)]]] };
  select(fP.controls, { label: 'Colour by', value: colP, options: [['centre', 'sequencing centre'], ['pheno', 'immune phenotype'], ['resp', 'response']], onChange: (v) => { colP = v; paintP(); } });
  const sP = stats(fP.root);
  const FP = frame(fP.body, { width: 820, height: 340, x: d3.extent(E0.S, (s) => s[0]), y: d3.extent(E0.S, (s) => s[1]), xLabel: `PC1 (${pc(E0.var[0])} of variance)`, yLabel: `PC2 (${pc(E0.var[1])})` });
  FP.x.nice(); FP.y.nice(); FP.g.select('.axis.x').call(d3.axisBottom(FP.x)); FP.g.select('.axis.y').call(d3.axisLeft(FP.y).ticks(5));
  const hovB = (M) => (e, i) => tip(e, `<b>${B.ids[i]}</b> · ${B.centre[i]} · ${B.pheno[i]}<br>${y[i] ? 'responder' : 'non-responder'} · IFN-γ score ${f2(M.ifng[i])}`);
  const dP = FP.layer('pts').selectAll('circle').data(d3.range(N)).join('circle').attr('cx', (i) => FP.x(E0.S[i][0])).attr('cy', (i) => FP.y(E0.S[i][1])).attr('r', 4.5)
    .attr('stroke', css('--surface')).attr('stroke-width', 1.1).on('mousemove', hovB(E0)).on('mouseleave', untip);
  const legP = el('div'); fP.body.append(legP);
  const silR = silhouette(E0.S, y.map(String));
  function paintP() {
    dP.attr('fill', COLS[colP][0]); legP.innerHTML = ''; legend(legP, COLS[colP][1]);
    sP.set({ 'R² of PC1 with centre': f2(E0.r2), 'silhouette by centre': f2(E0.silB), 'silhouette by phenotype': f2(E0.silP), 'silhouette by response': f2(silR) });
  }
  paintP();
  prose(rb, `<p>Coloured by centre, PC1 splits the patients cleanly (R² ${f2(E0.r2)}). Coloured by phenotype, inflamed tumours sit mostly on the Centre B side, but that is ambiguous: Centre B also enrolled more inflamed tumours. From this plot alone it is not possible to tell how much of PC1 is technical and how much is biological.</p>`);

  // ---- Step 4: partial confounding
  rb = rung(ch, 4, 'Partial confounding: the centre difference contains biology',
    `<p>The cross-tabulation shows that the two centres enrolled different tumours:</p>`);
  table(rb, CENTRES.map((c) => ({ c, ...B.counts[c], n: nB(c), s: pc(shareInf(c)) })), [['c', 'Centre'], ['inflamed', 'Inflamed'], ['excluded', 'Excluded'], ['desert', 'Desert'], ['n', 'Total'], ['s', 'Share inflamed']], { cls: 'compact' });
  prose(rb, `<p>For any gene, the raw difference between centres (B − A) is therefore the sum of a <b>technical shift</b> and a <b>composition effect</b> (Centre B has more inflamed tumours). Fitting <code>expression ~ centre + phenotype</code> estimates the technical shift <i>within</i> each phenotype; the remainder is attributed to composition. In a simulation both parts are known, so the estimates can be checked.</p>`);
  const fC = figure(rb, 'Splitting the centre difference into technical and biological parts',
    'Bars: estimates from the uncorrected data. Open diamonds: the values known from the simulation. Naive correction subtracts the whole observed difference; protected correction subtracts only the technical estimate.');
  const geneOpts = [['score', 'IFN-γ module score (12 genes)'], ...d3.range(G).filter((j) => L.group[j] !== 'filler').map((j) => [String(j), `${L.genes[j]} (${L.group[j].replace('CAF/TGF-b', 'CAF/TGF-β')})`])];
  let gC = 'score';
  select(fC.controls, { label: 'Gene', value: gC, options: geneOpts, onChange: (v) => { gC = v; drawC(); } });
  const sC = stats(fC.root);
  const FC = frame(fC.body, { width: 820, height: 210, x: [-2, 2], y: [0, 3], xLabel: 'Centre B − Centre A (log2 expression)', grid: false, axes: false, margin: { left: 230, right: 30, top: 8 } });
  const axC = FC.g.append('g').attr('class', 'axis x').attr('transform', `translate(0,${FC.h})`);
  const zC = FC.layer('zero').append('line').attr('y1', 0).attr('y2', FC.h).attr('stroke', muted()).attr('stroke-dasharray', '4 4');
  legend(fC.body, [['observed difference', ink2()], ['technical shift (estimate)', centreColor('Centre B')], ['composition (remainder)', phenoColor('inflamed')], ['simulation truth', ink()]]);
  const Dp = design({ pheno: 1 }), techTrue = (col, tcol) => diffBA(col.map((v, i) => v - tcol[i]));
  function decomp(g) {
    const col = g === 'score' ? E0.ifng : colOf(X, +g), tcol = g === 'score' ? ifngOf(T) : colOf(T, +g);
    const obs = diffBA(col), tech = lsFit(Dp, [col])[0][1];
    return { obs, tech, bio: obs - tech, tTech: techTrue(col, tcol), tBio: diffBA(tcol) };
  }
  function drawC() {
    const d = decomp(gC), rows = [{ k: 'Observed B − A', v: d.obs, c: ink2() }, { k: 'Technical shift', v: d.tech, t: d.tTech, c: centreColor('Centre B') }, { k: 'Composition (biology)', v: d.bio, t: d.tBio, c: phenoColor('inflamed') }];
    const m = Math.max(0.5, d3.max(rows, (r) => Math.max(Math.abs(r.v), Math.abs(r.t ?? 0)))) * 1.15;
    FC.x.domain([-m, m]); axC.call(d3.axisBottom(FC.x).ticks(7)); zC.attr('x1', FC.x(0)).attr('x2', FC.x(0));
    const yb = d3.scaleBand().domain(rows.map((r) => r.k)).range([0, FC.h]).padding(0.32);
    FC.layer('bars').selectAll('rect').data(rows).join('rect').attr('y', (r) => yb(r.k)).attr('height', yb.bandwidth()).attr('x', (r) => FC.x(Math.min(0, r.v))).attr('width', (r) => Math.abs(FC.x(r.v) - FC.x(0))).attr('fill', (r) => r.c)
      .on('mousemove', (e, r) => tip(e, `${r.k}: ${f2(r.v)}${r.t != null ? `<br>simulation truth: ${f2(r.t)}` : ''}`)).on('mouseleave', untip);
    FC.layer('lab').selectAll('text').data(rows).join('text').attr('x', -10).attr('y', (r) => yb(r.k) + yb.bandwidth() / 2 + 4).attr('text-anchor', 'end').attr('font-size', 12.5).attr('fill', ink2()).text((r) => `${r.k}  ${r.v >= 0 ? '+' : '−'}${f2(Math.abs(r.v))}`);
    FC.layer('truth').selectAll('path').data(rows.filter((r) => r.t != null)).join('path').attr('d', d3.symbol(d3.symbolDiamond, 90)).attr('transform', (r) => `translate(${FC.x(r.t)},${yb(r.k) + yb.bandwidth() / 2})`)
      .attr('fill', 'none').attr('stroke', ink()).attr('stroke-width', 1.8);
    sC.set({ 'naive correction subtracts': f2(d.obs), 'protected correction subtracts': f2(d.tech), 'true technical shift': f2(d.tTech), 'biology removed by naive': f2(d.obs - d.tTech) });
  }
  drawC();
  const dS = decomp('score');
  worked(rb, 'The IFN-γ module score',
    `<div class="calc">observed  B − A                  = ${f2(dS.obs)}
technical shift (protected fit)  = ${f2(dS.tech)}     simulation truth ${f2(dS.tTech)}
composition (remainder)          = ${f2(dS.bio)}     simulation truth ${f2(dS.tBio)}</div>
     <p>For the IFN-γ score the technical shifts of its 12 genes partly cancel, so most of the observed centre difference is composition. A naive correction would subtract all ${f2(dS.obs)} from every Centre B patient, removing about ${f2(dS.obs - dS.tTech)} log2 units of real immune signal. Other genes (try CD8A or COL1A1) have large technical shifts; there both corrections subtract a similar amount.</p>`);

  // ---- Step 5: naive vs protected correction
  rb = rung(ch, 5, 'Naive versus protected correction',
    `<p>Both corrections fit one linear model per gene and subtract the centre coefficient from Centre B samples, then rescale Centre B residuals to the Centre A spread (the location/scale model behind ComBat, without its empirical-Bayes shrinkage):</p>
     <div class="calc">naive:      y_gi = α_g + β_g·centre_i + ε_gi
protected:  y_gi = α_g + β_g·centre_i + γ_g·phenotype_i + ε_gi
corrected:  y_gi − β̂_g·centre_i     (β̂ absorbs part of the biology when γ is omitted)</div>`);
  const fV = figure(rb, 'What each correction does to the IFN-γ score',
    'Each dot is a patient, grouped by phenotype and centre; black bars are group means. The dashed line in the inflamed column is where the inflamed mean would sit if the true inflamed-versus-other gap were preserved. Switch the correction to watch Centre B patients move.');
  let verV = 'none', stopV = null;
  const selV = select(fV.controls, { label: 'Correction', value: verV, options: Object.entries(VNAME), onChange: (v) => { verV = v; moveV(); } });
  button(fV.controls, 'Play all three', () => playV(), true);
  const sV = stats(fV.root);
  const allS = [...E0.ifng, ...EN.ifng, ...EP.ifng];
  const FV = frame(fV.body, { width: 820, height: 330, x: [0, 1], y: d3.extent(allS), yLabel: 'IFN-γ module score (log2)', xScale: d3.scaleBand().padding(0.1) });
  FV.x.domain(PHENOS); FV.y.nice(); FV.g.select('.axis.x').call(d3.axisBottom(FV.x)); FV.g.select('.axis.y').call(d3.axisLeft(FV.y).ticks(5));
  legend(fV.body, [...CENTRES.map((c) => [c, centreColor(c)]), ['group mean', ink()], ['inflamed mean if true gap kept', ink2()]]);
  const bwV = FV.x.bandwidth(), rv = rng(11), jv = B.ids.map(() => rv() - 0.5);
  const xV = (i) => FV.x(B.pheno[i]) + bwV * (bv[i] ? 0.72 : 0.28) + jv[i] * bwV * 0.3;
  let curV = E0.ifng.slice();
  const dV = FV.layer('pts').selectAll('circle').data(d3.range(N)).join('circle').attr('r', 4).attr('cx', xV).attr('fill', (i) => centreColor(B.centre[i])).attr('stroke', css('--surface')).attr('stroke-width', 1)
    .on('mousemove', (e, i) => tip(e, `<b>${B.ids[i]}</b> · ${B.centre[i]} · ${B.pheno[i]}<br>score ${f2(curV[i])} (uncorrected ${f2(E0.ifng[i])})`)).on('mouseleave', untip);
  const meanL = FV.layer('means');
  function paintV() {
    dV.attr('cy', (i) => FV.y(curV[i]));
    const grp = []; PHENOS.forEach((p) => [0, 1].forEach((c) => grp.push({ p, c, m: mean(curV.filter((v, i) => B.pheno[i] === p && bv[i] === c)) })));
    meanL.selectAll('line').data(grp).join('line').attr('x1', (d) => FV.x(d.p) + bwV * (d.c ? 0.56 : 0.12)).attr('x2', (d) => FV.x(d.p) + bwV * (d.c ? 0.88 : 0.44)).attr('y1', (d) => FV.y(d.m)).attr('y2', (d) => FV.y(d.m)).attr('stroke', ink()).attr('stroke-width', 2.5);
    const other = mean(curV.filter((v, i) => !inf[i]));
    meanL.selectAll('path').data([0]).join('path').attr('d', `M${FV.x('inflamed')},${FV.y(other + truthGap)}h${bwV}`).attr('stroke', ink2()).attr('stroke-dasharray', '5 4').attr('stroke-width', 1.5);
  }
  function moveV(done) {
    if (stopV) stopV();
    const from = curV.slice(), E = VER[verV](), to = E.ifng; let t = 0;
    sV.set({ 'inflamed − other gap': `${f2(E.gap)} (truth ${f2(truthGap)})`, 'share of true gap kept': pc(E.gap / truthGap), 'AUC of IFN-γ score for response': f3(E.auc) });
    stopV = animate((dt) => {
      t = Math.min(1, t + dt / 700); const e = d3.easeCubicInOut(t);
      curV = from.map((v, i) => v + (to[i] - v) * e); paintV();
      if (t >= 1) { stopV = null; if (done) done(); return false; }
      return true;
    });
  }
  function playV() {
    const seq = ['none', 'naive', 'protected']; let k = 0;
    const next = () => { if (k >= seq.length) return; verV = seq[k++]; selV.value = verV; moveV(() => { let w = 0; stopV = animate((dt) => { w += dt; if (w > 600) { stopV = null; next(); return false; } return true; }); }); };
    next();
  }
  paintV(); moveV();
  prose(rb, `<p>Naive correction treats Centre B's higher mean as technical. Because Centre B is ${pc(shareInf('Centre B'))} inflamed, its mean is high partly for biological reasons, so its inflamed tumours are pushed down: the gap falls from ${f2(E0.gap)} to <b>${f2(EN.gap)}</b> (${pc(EN.gap / truthGap)} of the truth) and the response AUC of the score from ${f3(E0.auc)} to ${f3(EN.auc)}. Protected correction estimates the shift within phenotype and keeps a gap of <b>${f2(EP.gap)}</b>. The uncorrected gap (${f2(E0.gap)}) is also close to the truth, for the reason shown in step 4; across all genes, however, the uncorrected per-gene effects are biased (step 6).</p>`);

  // ---- Step 6: the lab
  rb = rung(ch, 6, 'Correction lab: choose what to protect and check against the truth',
    `<p>The correction is recomputed live on the 150-gene matrix for the ${N} patients. Because the data are simulated, each gene's true inflamed-versus-other effect (without noise or centre shift) is known; the right panel compares the estimate from the corrected data with that truth. This is the validation that real data cannot provide, and the reason to test a pipeline on simulated or spike-in data before trusting it.</p>`);
  const fL = figure(rb, 'Batch-correction lab',
    'Left: PCA of the corrected matrix. Right: one dot per gene, true inflamed-versus-other effect against the estimate from the corrected data; dots on the diagonal are recovered exactly. Silhouettes are computed on PC1–PC2. Hover dots.');
  const prot = { pheno: false, resp: false };
  let modeL = 'locscale', colL = 'centre';
  const boxPh = checkbox(fL.controls, { label: 'protect phenotype', checked: false, onChange: (c) => { prot.pheno = c; paintL(); } });
  const boxR = checkbox(fL.controls, { label: 'protect response', checked: false, onChange: (c) => { prot.resp = c; paintL(); } });
  const selM = select(fL.controls, { label: 'Adjustment', value: modeL, options: [['none', 'none (uncorrected)'], ['loc', 'location (mean shift)'], ['locscale', 'location + scale']], onChange: (v) => { modeL = v; paintL(); } });
  select(fL.controls, { label: 'Colour by', value: colL, options: [['centre', 'centre'], ['pheno', 'phenotype'], ['resp', 'response']], onChange: (v) => { colL = v; paintL(); } });
  const presets = el('div', { class: 'controls' }); fL.controls.before(presets);
  presets.append(el('span', { class: 'fig-caption', style: 'margin:0' }, 'Presets:'));
  const setL = (ph, re, md) => { prot.pheno = ph; prot.resp = re; modeL = md; boxPh.checked = ph; boxR.checked = re; selM.value = md; paintL(); };
  [['Uncorrected', false, false, 'none'], ['Naive', false, false, 'locscale'], ['Protected', true, false, 'locscale'], ['Protect response only', false, true, 'locscale']].forEach(([lab, a, b2, md]) => button(presets, lab, () => setL(a, b2, md)));
  const sL = stats(fL.root);
  const rowL = el('div', { class: 'fig-row' }); fL.body.append(rowL);
  const lA = el('div'), lB = el('div'); rowL.append(lA, lB);
  const FA = frame(lA, { width: 480, height: 340, x: [-1, 1], y: [-1, 1], xLabel: 'PC1', yLabel: 'PC2' });
  FA.g.select('.axis.x').call(d3.axisBottom(FA.x).tickValues([])); FA.g.select('.axis.y').call(d3.axisLeft(FA.y).tickValues([]));
  let curL = null, stopL = null, ELcur = E0;
  const dA = FA.layer('pts').selectAll('circle').data(d3.range(N)).join('circle').attr('r', 4).attr('stroke', css('--surface')).attr('stroke-width', 1)
    .on('mousemove', (e, i) => hovB(ELcur)(e, i)).on('mouseleave', untip);
  const effAll = [...trueEff, ...E0.eff, ...EN.eff, ...EP.eff], eExt = d3.extent(effAll);
  const FE = frame(lB, { width: 480, height: 340, x: eExt, y: eExt, xLabel: 'true effect (simulation)', yLabel: 'estimated from corrected data' });
  FE.x.nice(); FE.y.nice(); FE.g.select('.axis.x').call(d3.axisBottom(FE.x).ticks(5)); FE.g.select('.axis.y').call(d3.axisLeft(FE.y).ticks(5));
  FE.layer('diag').append('line').attr('x1', FE.x(FE.x.domain()[0])).attr('y1', FE.y(FE.x.domain()[0])).attr('x2', FE.x(FE.x.domain()[1])).attr('y2', FE.y(FE.x.domain()[1])).attr('stroke', muted()).attr('stroke-dasharray', '4 4');
  const isIfng = new Set(L.ifng_idx);
  const dE = FE.layer('pts').selectAll('circle').data(d3.range(G).sort((a, b) => isIfng.has(a) - isIfng.has(b))).join('circle').attr('r', 3.8).attr('cx', (j) => FE.x(trueEff[j]))
    .attr('fill', (j) => (isIfng.has(j) ? cat(7) : cat(0))).attr('fill-opacity', 0.85).attr('stroke', css('--surface')).attr('stroke-width', 0.8)
    .on('mousemove', (e, j) => tip(e, `<b>${L.genes[j]}</b> (${L.group[j].replace('CAF/TGF-b', 'CAF/TGF-β')})<br>true effect ${f2(trueEff[j])}<br>estimate ${f2(ELcur.eff[j])}`)).on('mouseleave', untip);
  const legL = el('div'); fL.body.append(legL);
  function paintL() {
    const E = evaluate(prot, modeL); ELcur = E;
    const sx = d3.max(E.S, (s) => Math.max(Math.abs(s[0]), Math.abs(s[1]))) * 1.05, to = E.S.map((s) => [s[0] / sx, s[1] / sx]);
    FA.g.selectAll('.ax-label').each(function (d, i) { this.textContent = i ? `PC2 (${pc(E.var[1])})` : `PC1 (${pc(E.var[0])} of variance)`; });
    dA.attr('fill', COLS[colL][0]); dE.transition().duration(450).attr('cy', (j) => FE.y(E.eff[j]));
    legL.innerHTML = ''; legend(legL, [...COLS[colL][1], ['gene (right panel)', cat(0)], ['IFN-γ module gene', cat(7)]]);
    if (stopL) stopL();
    const from = curL || to; let t = 0;
    stopL = animate((dt) => {
      t = Math.min(1, t + dt / 600); const e = d3.easeCubicInOut(t);
      curL = from.map((p, i) => [p[0] + (to[i][0] - p[0]) * e, p[1] + (to[i][1] - p[1]) * e]);
      dA.attr('cx', (i) => FA.x(curL[i][0])).attr('cy', (i) => FA.y(curL[i][1]));
      return t < 1;
    });
    sL.set({ 'R² of PC1 with centre': f2(E.r2), 'silhouette: centre': f2(E.silB), 'silhouette: phenotype': f2(E.silP), 'IFN-γ gap (truth)': `${f2(E.gap)} (${f2(truthGap)})`, 'per-gene error vs truth (RMSE)': f3(E.rmse), 'response AUC of IFN-γ score': f3(E.auc) });
  }
  paintL();
  const EL = evaluate({}, 'loc'), ER = evaluate({ resp: 1 }, 'locscale'), EPR = evaluate({ pheno: 1, resp: 1 }, 'locscale');
  prose(rb, '<p><b>Suggested comparisons</b>:</p>');
  table(rb, [
    { t: '<b>Uncorrected</b> → <b>Naive</b>', s: `Centre silhouette ${f2(E0.silB)} → <b>${f2(EN.silB)}</b>; per-gene error ${f3(E0.rmse)} → ${f3(EN.rmse)}; IFN-γ gap ${f2(E0.gap)} → <b>${f2(EN.gap)}</b>`, m: 'Naive correction removes the centre and most of the per-gene bias, but also removes the part of the biology that differs between centres.' },
    { t: '<b>Naive</b> → <b>Protected</b>', s: `Phenotype silhouette ${f2(EN.silP)} → <b>${f2(EP.silP)}</b>; gap <b>${f2(EP.gap)}</b>; error <b>${f3(EP.rmse)}</b>; R² of PC1 with centre ${f2(EP.r2)}`, m: `Estimating the shift within phenotype recovers the true effects. PC1 still correlates with centre because a biological axis is unevenly distributed between centres; a correction that forced this to 0 would be removing biology.` },
    { t: '<b>Protect response only</b>', s: `Gap ${f2(ER.gap)}, error ${f3(ER.rmse)}`, m: 'Response is a noisy proxy for phenotype, so it protects only part of the biology. Protecting the outcome that will later be tested also makes downstream tests over-confident (pitfalls).' },
    { t: 'Add <b>response</b> to <b>Protected</b>', s: `Gap ${f2(EP.gap)} → ${f2(EPR.gap)}; error ${f3(EP.rmse)} → ${f3(EPR.rmse)}`, m: 'Once phenotype is protected, the extra covariate changes almost nothing in this simulation, but it carries the over-confidence risk.' },
    { t: 'Adjustment <b>location + scale</b> → <b>location</b> (naive)', s: `Gap ${f2(EN.gap)} → ${f2(EL.gap)}; error ${f3(EN.rmse)} → ${f3(EL.rmse)}`, m: 'The centre effect here is mainly a mean shift per gene, so the scale step matters little. It matters more when one batch is noisier.' },
  ], [['t', 'Try'], ['s', 'What you see'], ['m', 'What it means']], { cls: 'compact' });

  // =====================================================================================
  // 5. PITFALLS
  // =====================================================================================
  ch = chapter(main, 'pitfalls', 'Pitfalls', 'Common problems, how to recognise them, and how to address them.');
  cards(ch, [
    { tag: 'Symptom: the biomarker improves after QC', title: 'Exclusions chosen after seeing the result', body: 'Removing samples that weaken a result is a form of leakage and inflates the effect. <b>Fix:</b> fix the QC metrics and the threshold k before the outcome is examined, and report every exclusion with its reason.' },
    { tag: 'Symptom: a striking sample is removed', title: 'Outlier is not the same as failure', body: 'A rare histology or extreme immune infiltration can look like an outlier. <b>Fix:</b> exclude only on technical evidence (low correlation, 3′ bias, rRNA fraction), and inspect the profile before removing a sample.' },
    { tag: 'Symptom: healthy samples flagged', title: 'Batch structure misleads multivariate detectors', body: `With two centres, "distance from the centre of the data" is distance from a point between two clouds. Here the isolation forest flags ${isoF.length} healthy sample${isoF.length === 1 ? '' : 's'} at k = ${K0}. <b>Fix:</b> run outlier detection within each batch, or after inspecting the PCA.` },
    { tag: 'Symptom: known biology weaker after correction', title: 'Naive correction removes signal', body: `When biology is unbalanced across batches, a centre-only model attributes part of it to the batch. The IFN-γ gap fell from ${f2(E0.gap)} to ${f2(EN.gap)} (truth ${f2(truthGap)}). <b>Fix:</b> protect known biological covariates, or keep batch as a covariate in the downstream model.` },
    { tag: 'Symptom: very small p-values after ComBat', title: 'Protecting the variable that is tested', body: 'If the outcome (e.g. response) is protected and later tested on the corrected data, its group differences are preserved by construction and the uncertainty of the batch estimate is ignored, so tests become over-confident in unbalanced designs (Nygaard et al., <i>Biostatistics</i> 2016). <b>Fix:</b> for testing, use the uncorrected data with batch in the design (<code>~ centre + response</code>).' },
    { tag: 'Symptom: excellent cross-validated accuracy', title: 'Correction fitted on all samples', body: 'A correction estimated on the whole dataset uses information from the test samples. <b>Fix:</b> fit the correction inside each training fold, or correct new samples towards a fixed reference batch; validate on an untouched cohort.' },
    { tag: 'Symptom: batch and outcome inseparable', title: 'Complete confounding', body: `Here Centre B is ${pc(shareInf('Centre B'))} inflamed and Centre A ${pc(shareInf('Centre A'))}: partial confounding, which a model can handle. If every inflamed tumour came from one centre, no method could separate centre from biology. <b>Fix:</b> a design problem: randomise samples across sequencing runs and centres.` },
    { tag: 'Symptom: many genes shifted in one direction', title: 'CPM is not enough for composition changes', body: 'If a few genes take up a large share of the reads in some samples, every other gene appears lower after CPM. <b>Fix:</b> use TMM (edgeR) or median-of-ratios (DESeq2) size factors, which are robust to such genes.' },
  ]);

  // =====================================================================================
  // 6. RECAP
  // =====================================================================================
  ch = chapter(main, 'recap', 'Recap', 'Key points.');
  recap(ch, [
    'Raw counts depend on <b>sequencing depth</b>. Divide by library size (CPM) and take log2(CPM + 1) before PCA, QC metrics or correction.',
    'Flag failed libraries with <b>median ± k·MAD</b> rules on expression-based metrics; robust statistics keep the failures from hiding themselves.',
    'An <b>outlier is not necessarily a failure</b>, and a shallow library is not necessarily a broken one. Fix the rule before looking at outcomes.',
    'Colour the first PCA by every technical and biological variable. In multi-centre data the first component is often the <b>centre</b>.',
    'Cross-tabulate batch against biology. When they are <b>partially confounded</b>, the raw centre difference contains real biology.',
    'A <b>naive correction</b> removes that biology; a <b>protected correction</b> estimates the batch shift within levels of the biological covariates.',
    'For differential expression and other regressions, prefer <b>batch as a covariate</b> in the model to correcting the data first; do not protect a variable that will be tested.',
    'Corrections are fitted on training data only. Completely confounded designs cannot be rescued by any method.',
  ]);
  prose(ch, '<h3>In code</h3><p>Normalisation and QC in a few lines, then ComBat with the biological covariates passed as the model matrix (<code>mod</code> / <code>covar_mod</code>), which is the "protect" choice on this page.</p>');
  code(ch, {
    Python: `import numpy as np, pandas as pd
from inmoose.pycombat import pycombat_norm      # pip install inmoose

# counts: genes x samples DataFrame; meta: samples x [centre, phenotype]
lib = counts.sum(axis=0)
logcpm = np.log2(counts / lib * 1e6 + 1)

# QC: median correlation to the other samples, flagged by median - 3 MAD
c = np.corrcoef(logcpm.T.values); np.fill_diagonal(c, np.nan)
medcor = pd.Series(np.nanmedian(c, axis=1), index=logcpm.columns)
mad = 1.4826 * (medcor - medcor.median()).abs().median()
keep = medcor >= medcor.median() - 3 * mad
X, m = logcpm.loc[:, keep], meta.loc[keep]

# Batch correction that protects phenotype (for PCA / clustering only)
X_corr = pycombat_norm(X, m["centre"], covar_mod=m[["phenotype"]])`,
    R: `library(edgeR); library(sva); library(limma)

y <- DGEList(counts); y <- y[filterByExpr(y), , keep.lib.sizes = FALSE]
y <- calcNormFactors(y)                       # TMM
logcpm <- cpm(y, log = TRUE, prior.count = 1)

# QC: median correlation, median - 3 MAD
cc <- cor(logcpm); diag(cc) <- NA
medcor <- apply(cc, 2, median, na.rm = TRUE)
keep <- medcor >= median(medcor) - 3 * mad(medcor)

# Protected correction for visualisation
mod <- model.matrix(~ phenotype, data = meta[keep, ])
x_corr <- ComBat(logcpm[, keep], batch = meta$centre[keep], mod = mod)

# For testing, do not correct: put centre in the design instead
design <- model.matrix(~ centre + response, data = meta[keep, ])`,
  });
  seeAlso(main, [['omics', 'bulk-de', 'Differential expression'], ['omics', 'bulk-classifier', 'Biomarker classifier and nested CV'], ['concepts', 'pca', 'Principal component analysis'], ['concepts', 'cross-validation', 'Cross-validation and data leakage'], ['omics', 'sc-clustering', 'Single-cell clustering']]);
}

// ---------------------------------------------------------------------------------------------
/** Pipeline diagram: raw counts → QC → normalise → explore → correct → downstream, with the study's numbers. */
function pipelineSvg(Q, B, E0, EP) {
  const wrap = el('div', { class: 'figure' });
  wrap.append(el('div', { class: 'fig-title' }, 'The pipeline on this page, for the discovery cohort'));
  const svg = d3.select(wrap).append('svg').attr('viewBox', '0 0 1000 200').attr('width', 1000).attr('role', 'img').attr('aria-label', 'raw counts, QC, normalisation, PCA, batch correction, downstream analysis');
  const boxes = [
    { t: 'Raw counts', l: [`${Q.ids.length} samples`, `${Q.n_genes.toLocaleString()} genes`, '2 centres'] },
    { t: '① QC', l: ['median ± 3·MAD', `${Q.excluded.length} excluded`, `${B.ids.length} kept`] },
    { t: '② Normalise', l: ['CPM = count / lib × 10⁶', 'log2(CPM + 1)'] },
    { t: '③ Explore', l: ['PCA', `PC1 ${Math.round(100 * E0.var[0])}% of variance`, `R² with centre ${E0.r2.toFixed(2)}`] },
    { t: '④ Correct', l: ['per-gene linear fit', 'protect phenotype', `R² ${EP.r2.toFixed(2)}`] },
    { t: 'Downstream', l: ['DE (centre in model)', 'clustering, PCA', 'classifiers'] },
  ];
  const BW = 142, gap = 25, H = 160;
  boxes.forEach((b, i) => {
    const x0 = 6 + i * (BW + gap), g = svg.append('g').attr('transform', `translate(${x0},20)`);
    g.append('rect').attr('width', BW).attr('height', H).attr('rx', 10).attr('fill', css('--surface-2')).attr('stroke', css('--axis'));
    g.append('text').attr('x', 10).attr('y', 24).attr('font-weight', 700).attr('font-size', 13.5).attr('fill', ink()).text(b.t);
    b.l.forEach((s, j) => g.append('text').attr('x', 10).attr('y', 56 + j * 24).attr('font-size', 12).attr('fill', ink2()).text(s));
    if (i < boxes.length - 1) {
      const ax = x0 + BW + 3, ay = 20 + H / 2;
      svg.append('path').attr('d', `M${ax},${ay} L${ax + gap - 6},${ay} M${ax + gap - 12},${ay - 5} L${ax + gap - 6},${ay} L${ax + gap - 12},${ay + 5}`).attr('fill', 'none').attr('stroke', ink2()).attr('stroke-width', 2);
    }
  });
  return wrap;
}
