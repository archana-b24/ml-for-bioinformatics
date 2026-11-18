// Differential expression (lesson). Bulk RNA-seq step 2 of 4.
// Arc: what → why → where → how it works (6 steps: one gene by hand → linear model with the centre term →
//      empirical-Bayes variance moderation → p-value histogram and BH FDR → volcano and the planted biology →
//      DE lab with Centre C replication and try/see/means table) → pitfalls → recap.
// Data: data/omics/bulk_de.json (scripts/build_bulk.py: Welch / lm / limma-style / permuted results) and
// data/omics/bulk_de_lab.json (scripts/build_de_lesson.py: log-CPM of 1,933 genes × 177 patients, noise-free
// true effects, Centre C replication). Per-gene linear models, moderation and BH run live in the browser.
import { header, figure, slider, select, button, checkbox, stats, legend, seeAlso, tip, untip, el } from '../lib/ui.js';
import { frame, cat, css, ink, ink2, muted } from '../lib/plot.js';
import { rng, solve, mean } from '../lib/data.js';
import { pNormal } from '../lib/models.js';
import { lessonNav, chapter, prose, rung, worked, quiz, cards, table, code, recap } from '../lib/lesson.js';
import { stepCard, omicsData } from './_shared.js';
import { GROUP_ORDER, groupColor, groupLabel, centreColor } from './_bulk_common.js';
const d3 = window.d3;

const f2 = (v) => (Math.abs(v) < 0.005 ? '0.00' : v.toFixed(2));
const f1 = (v) => (isFinite(v) ? v.toFixed(1) : '∞');
const pc = (v) => Math.round(100 * v) + '%';
const pE = (p) => (p < 0.001 ? p.toExponential(1) : p.toFixed(3));
const nf = (v) => v.toLocaleString('en-GB');
const sum = (a) => a.reduce((s, v) => s + v, 0);
const nlp = (p) => -Math.log10(Math.max(p, 1e-300));
const jit = (i) => (((i * 7919) % 101) / 101 - 0.5);

export default async function render(main) {
  header(main, 'Differential expression',
    'For every gene, estimate how much its average expression differs between two groups of samples, how certain that estimate is, and how many of the genes declared different are likely to be false. The output is a <b>ranked list of genes with effect sizes and false discovery rates</b>: an inference about the population, not a prediction for a new patient.');
  lessonNav(main);
  stepCard(main, {
    stage: 'Bulk RNA-seq · step 2 of 4',
    question: 'Which genes are expressed at different levels in tumours that respond to anti-PD-1, after accounting for the sequencing centre?',
    data: 'Uncorrected log-CPM of 1,933 genes for the 177 discovery patients that passed QC, with response and centre. The centre goes into the model as a covariate instead of being removed beforehand.',
    method: 'One linear model per gene (expression ~ response + centre), limma-style empirical-Bayes variance moderation, Benjamini–Hochberg false discovery rate.',
    verdict: 'stats', why: ' The goal is inference with error control (is this gene\'s difference real, how large is it, and how many calls are false?), not a prediction for a new patient. Nothing is tuned on held-out data; success is measured by calibrated error rates and by replication, not by accuracy.',
  });
  const [J, L] = await Promise.all([omicsData(main, 'bulk_de.json'), omicsData(main, 'bulk_de_lab.json')]);
  if (!J || !L) return;

  // ------------------------------------------------------------------ data + live fits
  const G = L.genes.length, NP = L.y.length, M = J.methods, grp = L.group;
  const raw = Uint8Array.from(atob(L.x_b64), (c) => c.charCodeAt(0)), u16 = new Uint16Array(raw.buffer);
  const X = new Float32Array(u16.length); for (let k = 0; k < u16.length; k++) X[k] = u16[k] / L.x_scale;
  const gi = Object.fromEntries(L.genes.map((g, i) => [g, i]));
  const ALL = d3.range(NP), nR = sum(L.y), nB = sum(L.centreB);
  const nGrp = Object.fromEntries(GROUP_ORDER.map((g) => [g, grp.filter((x) => x === g).length]));
  const callsBy = (q, a = 0.05, extra = () => true) => Object.fromEntries(GROUP_ORDER.map((g) => [g, q.filter((v, i) => v < a && grp[i] === g && extra(i)).length]));
  const nCalls = (q, a = 0.05) => q.filter((v) => v < a).length;
  const col = (g, rows = ALL) => rows.map((i) => X[i * G + g]);
  const basePlain = baseFit(ALL, false), baseC = baseFit(ALL, true);
  const plain = moderate(basePlain, 0), lmC = moderate(baseC, 0), lim = moderate(baseC, null);
  const IF = L.genes.map((g, i) => i).filter((i) => grp[i] === 'IFNG module'), PR = L.genes.map((g, i) => i).filter((i) => grp[i] === 'PROLIF module');
  const respA = L.y.filter((v, i) => !L.centreB[i]), respB = L.y.filter((v, i) => L.centreB[i]);

  function baseFit(rows, withCentre) {
    const cB = rows.map((i) => L.centreB[i]), useC = withCentre && sum(cB) > 0 && sum(cB) < rows.length;
    const Dm = rows.map((i, r) => (useC ? [1, L.y[i], cB[r]] : [1, L.y[i]])), k = Dm[0].length, n = rows.length;
    const DtD = d3.range(k).map((a) => d3.range(k).map((b) => sum(Dm.map((r) => r[a] * r[b]))));
    const inv = d3.range(k).map((j) => solve(DtD, d3.range(k).map((a) => +(a === j))));
    const H = d3.range(k).map((a) => Dm.map((r) => sum(d3.range(k).map((b) => inv[a][b] * r[b]))));
    const d = n - k, lfc = new Float64Array(G), s2 = new Float64Array(G), beta = new Array(k);
    for (let g = 0; g < G; g++) {
      for (let a = 0; a < k; a++) { let s = 0; for (let r = 0; r < n; r++) s += H[a][r] * X[rows[r] * G + g]; beta[a] = s; }
      let rss = 0; for (let r = 0; r < n; r++) { let f = 0; for (let a = 0; a < k; a++) f += beta[a] * Dm[r][a]; rss += (X[rows[r] * G + g] - f) ** 2; }
      lfc[g] = beta[1]; s2[g] = Math.max(rss / d, 1e-8);
    }
    const e = Array.from(s2, (v) => Math.log(v) - digamma(d / 2) + Math.log(d / 2)), em = mean(e);
    const ev = sum(e.map((v) => (v - em) ** 2)) / (G - 1) - trigamma(d / 2);
    return { rows, n, d, k, useC, v11: inv[1][1], lfc, s2, em, d0est: ev > 0 ? 2 * trigammaInv(ev) : Infinity };
  }
  /** d0 = null → limma's estimate; 0 → ordinary t. Returns moderated variances, t, p and BH q. */
  function moderate(B, d0in) {
    const d0 = d0in == null ? B.d0est : d0in, fin = isFinite(d0);
    const s02 = d0 > 0 ? Math.exp(B.em + (fin ? digamma(d0 / 2) - Math.log(d0 / 2) : 0)) : NaN;
    const s2m = Array.from(B.s2, (v) => (d0 === 0 ? v : fin ? (d0 * s02 + B.d * v) / (d0 + B.d) : s02));
    const t = s2m.map((v, g) => B.lfc[g] / Math.sqrt(v * B.v11)), df = B.d + (fin ? d0 : 1e7);
    const p = t.map((v) => (df > 1e5 ? pNormal(v) : pT2(v, df)));
    return { ...B, d0, s02, s2m, t, p, q: bh(p), df };
  }
  const drawStudy = (k, seed) => {
    if (k === 'all') return ALL;
    const r = rng(seed), pick = (c) => r.shuffle(ALL.filter((i) => L.y[i] === c)).slice(0, k);
    return [...pick(1), ...pick(0)];
  };

  // =====================================================================================
  // 1. WHAT
  // =====================================================================================
  let ch = chapter(main, 'what', 'What is differential expression?', 'In one line: <b>one model per gene → effect size and p-value → false discovery rate across all genes</b>.');
  prose(ch, `<p>Differential expression (DE) analysis compares the average expression of each gene between groups of samples, here the ${nR} responders and ${NP - nR} non-responders. For each gene it reports the <b>log2 fold change</b> (how large the difference is), a <b>p-value</b> (how surprising the difference would be if there were none), and a <b>q-value</b> (the false discovery rate, FDR, at which the gene would be called significant). The tests are run gene by gene, but the analysis uses information from all genes at once in two places: variance moderation and the FDR.</p>
    <p>The diagram follows one gene, CXCL9, through the analysis used on this page.</p>`);
  ch.append(pipelineSvg(gi.CXCL9));
  prose(ch, '<p>Key terms:</p>');
  const cx = gi.CXCL9;
  table(ch, [
    { t: 'log-CPM', m: 'log2 of counts per million reads (+1). Puts samples of different sequencing depth on one scale; differences become log2 ratios', s: 'CXCL9 averages ' + f2(mean(col(cx))) },
    { t: 'log2 fold change (log2FC)', m: 'Difference in mean log expression between groups. +1 means twice as high in responders', s: `CXCL9: ${f2(lim.lfc[cx])} (×${f2(2 ** lim.lfc[cx])})` },
    { t: 'Design matrix', m: 'The table of explanatory variables given to every gene\'s model: intercept, response (0/1), centre (0/1)', s: `${NP} rows × ${baseC.k} columns` },
    { t: 'Residual variance (s²)', m: 'Spread of a gene\'s expression that the model does not explain; sets the standard error of log2FC', s: `CXCL9: s = ${f2(Math.sqrt(baseC.s2[cx]))}` },
    { t: 'Moderated t', m: 'log2FC divided by its standard error, with s² shrunk towards a value typical of all genes (empirical Bayes)', s: `CXCL9: t = ${f1(lim.t[cx])}` },
    { t: 'p-value', m: 'Probability of a difference at least this large if the gene were truly not different', s: `CXCL9: ${pE(lim.p[cx])}` },
    { t: 'False discovery rate / q-value', m: 'Expected share of false positives among the genes called. q is the smallest FDR at which this gene would be called', s: `${nCalls(lim.q)} genes at q < 0.05` },
  ], [['t', 'Term'], ['m', 'Meaning'], ['s', 'In our study']]);

  // =====================================================================================
  // 2. WHY
  // =====================================================================================
  ch = chapter(main, 'why', 'Why do we need it?', 'What a gene-by-gene comparison must provide, and why a per-gene t-test is not enough.');
  prose(ch, `<p>A gene-level comparison of responders and non-responders should:</p><ol>
    <li><b>Estimate an effect size with its uncertainty</b> for every gene, not only a yes/no call.</li>
    <li><b>Adjust for the study design.</b> Here ${pc(mean(respB))} of Centre B patients responded against ${pc(mean(respA))} in Centre A, so any gene shifted by the centre's processing also differs between responders and non-responders.</li>
    <li><b>Control false positives across ${nf(G)} tests</b>, so that the list can be trusted as a whole.</li>
    <li><b>Remain stable with few samples</b>, when each gene's own variance estimate is unreliable.</li></ol>`);
  const welchP05 = M.welch.p.filter((p) => p < 0.05).length, fW = M.welch.by_group.filler;
  worked(ch, 'A per-gene t-test at p < 0.05, without the centre',
    `<p>A Welch t-test per gene gives p &lt; 0.05 for <b>${nf(welchP05)}</b> of ${nf(G)} genes. If no gene differed at all, about 0.05 × ${nf(G)} ≈ <b>${Math.round(0.05 * G)}</b> would pass by chance. Even after a false discovery rate correction the test calls <b>${M.welch.n_hits}</b> genes, including <b>${fW[0]} of the ${nf(fW[1])} filler genes</b> that carry no planted response effect. Most of these genes differ between centres, and centre is associated with response.</p>
     <p>The t-test is computed correctly; it answers the question "do the groups differ?", which is not the same as "does response explain the difference?". Adding the centre to the model (chapter 4, step 2) reduces the list to ${M.lm.n_hits} genes.</p>`);
  prose(ch, '<p>DE is often run next to machine-learning classifiers on the same data, but the two answer different questions:</p>');
  table(ch, [
    { a: 'Question', s: 'Is gene g different in the population, and by how much?', m: 'Will this new patient respond?' },
    { a: 'Output', s: 'Effect size, p-value and q-value for every gene', m: 'A score or probability for each new sample' },
    { a: 'Model', s: 'Fixed in advance by the design (<code>~ response + centre</code>), one small model per gene', m: 'Chosen and tuned on the data, many genes combined' },
    { a: 'Judged by', s: 'Calibrated error rates; replication in an independent cohort', m: 'Performance on samples the model never saw (AUC, calibration)' },
    { a: 'Confounders', s: 'Must be adjusted for, otherwise the estimate is biased', m: 'Can be exploited silently; performance then drops at a new site' },
  ], [['a', ''], ['s', 'Differential expression (statistics)'], ['m', 'Response classifier (machine learning)']]);
  table(ch, [
    { a: 'Fold-change cut-off only', e: 'yes', c: 'no', f: 'no', n: 'no' },
    { a: 'Two-group t-test per gene', e: 'yes', c: 'no', f: 'with BH', n: 'poor (noisy variances)' },
    { a: '<b>Linear model + moderation + BH (limma, DESeq2, edgeR)</b>', e: '<b>yes</b>', c: '<b>yes</b>', f: '<b>yes</b>', n: '<b>yes</b>' },
    { a: 'Classifier feature importance', e: 'no (importance, not difference)', c: 'implicitly', f: 'no', n: 'poor' },
  ], [['a', 'Approach'], ['e', 'Effect size?'], ['c', 'Adjusts for covariates?'], ['f', 'Controls false discoveries?'], ['n', 'Works with few samples?']]);

  // =====================================================================================
  // 3. WHERE
  // =====================================================================================
  ch = chapter(main, 'where', 'Where is it used?', 'Any study that compares many measured features between conditions and needs a list with error control.');
  cards(ch, [
    { tag: 'Bulk RNA-seq', title: 'Treated vs control, responder vs non-responder', body: '<b>limma-voom</b>, <b>DESeq2</b> and <b>edgeR</b> with a design formula such as <code>~ centre + response</code>. The analysis on this page follows limma on log-CPM.' },
    { tag: 'Single-cell', title: 'Condition effects within a cell type', body: 'Counts are summed per patient and cell type (pseudobulk), then analysed with the bulk tools. Cells from one patient are not independent replicates.' },
    { tag: 'Proteomics', title: 'Protein abundance', body: 'limma on log intensities is standard for label-free and TMT data (e.g. via MSstats or DEP), with extra care for missing values.' },
    { tag: 'Methylation arrays', title: 'Differentially methylated CpGs', body: 'limma on M-values per CpG (minfi, ChAMP), followed by region-level methods such as DMRcate.' },
    { tag: 'Microarrays', title: 'Where limma started', body: 'The moderated t-statistic was developed for two-colour and Affymetrix arrays with three to five replicates per group, where per-gene variances are very noisy.' },
    { tag: 'Pathway analysis', title: 'Input to gene-set tests', body: 'The ranked list (by moderated t or signed −log10 p) feeds GSEA, fgsea or camera, which ask whether a gene set is shifted as a whole.' },
    { tag: 'CRISPR screens', title: 'Enriched or depleted guides', body: 'MAGeCK and edgeR-based workflows compare guide counts between conditions with the same count models and FDR control.' },
    { tag: 'Metabolomics, lipidomics', title: 'Feature-wise comparisons', body: 'Per-feature linear models on log abundances with covariates (batch, age, sex) and BH adjustment.' },
  ]);
  prose(ch, '<p><b>When not to use it:</b> the aim is to predict the outcome of a new patient (use a classifier with nested cross-validation and independent validation); the groups were defined from the same expression data, for example clusters (the p-values are then invalid); single cells are treated as replicates (use pseudobulk); or the outcome is time to an event (use Cox regression per gene or a survival model).</p>');

  // =====================================================================================
  // 4. HOW IT WORKS
  // =====================================================================================
  ch = chapter(main, 'how', 'How it works', 'Six steps, from one gene compared by hand to a full analysis that is checked in an independent cohort.');

  // ---- Step 1: one gene by hand
  const yR = col(cx, ALL.filter((i) => L.y[i])), yN = col(cx, ALL.filter((i) => !L.y[i]));
  const mR = mean(yR), mN = mean(yN), sp2 = (sum(yR.map((v) => (v - mR) ** 2)) + sum(yN.map((v) => (v - mN) ** 2))) / (NP - 2);
  const se1 = Math.sqrt(sp2 * (1 / yR.length + 1 / yN.length));
  let rb = rung(ch, 1, 'One gene: difference in means, standard error and t',
    `<p>Expression is analysed on the log scale, so a difference in means is a <b>log2 fold change</b>. Its precision depends on how variable the gene is and how many patients are in each group.</p>`);
  worked(rb, 'CXCL9, responders vs non-responders',
    `<div class="calc">mean responders      = ${f2(mR)}   (n = ${yR.length})\nmean non-responders  = ${f2(mN)}   (n = ${yN.length})\nlog2 fold change     = ${f2(mR)} − ${f2(mN)} = <b>${f2(mR - mN)}</b>   (×${f2(2 ** (mR - mN))})\npooled SD s          = ${f2(Math.sqrt(sp2))}\nstandard error       = s × √(1/${yR.length} + 1/${yN.length}) = ${f2(se1)}\nt = ${f2(mR - mN)} / ${f2(se1)} = <b>${f1((mR - mN) / se1)}</b>  on ${NP - 2} degrees of freedom  →  p = ${pE(plain.p[cx])}</div>`);
  const fillerTrap = d3.range(G).filter((i) => grp[i] === 'filler' && M.lm.q[i] > 0.5).sort((a, b) => M.welch.p[a] - M.welch.p[b])[0];
  const GENES1 = ['CXCL9', 'GZMB', 'CD8A', 'MKI67', 'EPCAM', 'POSTN', 'ACTB', L.genes[fillerTrap]];
  const fg1 = figure(rb, 'One gene, all patients', 'Each dot is a patient, coloured by sequencing centre; bars mark group means. Splitting by centre shows how much of a difference is shared by both centres. Hover a patient.');
  let g1 = 'CXCL9', split1 = false;
  select(fg1.controls, { label: 'Gene', value: g1, options: GENES1.map((g) => [g, `${g} (${g === L.genes[fillerTrap] ? 'filler gene' : groupLabel(grp[gi[g]])})`]), onChange: (v) => { g1 = v; paint1(); } });
  checkbox(fg1.controls, { label: 'split by centre', checked: false, onChange: (c) => { split1 = c; paint1(); } });
  const s1 = stats(fg1.root);
  const F1 = frame(fg1.body, { width: 820, height: 320, x: [0, 1], y: [0, 1], xLabel: '', yLabel: 'log-CPM', xScale: d3.scaleBand().padding(0.25) });
  legend(fg1.body, [['Centre A', centreColor('Centre A')], ['Centre B', centreColor('Centre B')], ['group mean', ink()]]);
  function paint1() {
    const g = gi[g1], v = col(g), keyOf = (i) => (split1 ? (L.centreB[i] ? 'B' : 'A') + L.y[i] : '' + L.y[i]);
    const keys = split1 ? ['A0', 'A1', 'B0', 'B1'] : ['0', '1'];
    const lab = { 0: 'non-responders', 1: 'responders', A0: 'Centre A · non-resp.', A1: 'Centre A · responders', B0: 'Centre B · non-resp.', B1: 'Centre B · responders' };
    F1.x.domain(keys); F1.y.domain([d3.min(v) - 0.3, d3.max(v) + 0.3]).nice();
    F1.g.select('.axis.x').call(d3.axisBottom(F1.x).tickFormat((k) => `${lab[k]} (n = ${ALL.filter((i) => keyOf(i) === k).length})`));
    F1.g.select('.axis.y').call(d3.axisLeft(F1.y).ticks(5)); F1.g.select('.gridline').call(d3.axisLeft(F1.y).ticks(5).tickSize(-F1.w).tickFormat(''));
    const bw = F1.x.bandwidth();
    F1.layer('pts').selectAll('circle').data(ALL).join('circle').attr('r', 3.6).attr('cx', (i) => F1.x(keyOf(i)) + bw / 2 + jit(i) * bw * 0.7).attr('cy', (i) => F1.y(v[i]))
      .attr('fill', (i) => centreColor(L.centreB[i] ? 'Centre B' : 'Centre A')).attr('fill-opacity', 0.75).attr('stroke', css('--surface')).attr('stroke-width', 0.8)
      .on('mousemove', (e, i) => tip(e, `${L.ids[i]} · ${L.centreB[i] ? 'Centre B' : 'Centre A'} · ${L.pheno[i]}<br>${L.y[i] ? 'responder' : 'non-responder'} · ${g1} ${f2(v[i])}`)).on('mouseleave', untip);
    const mns = keys.map((k) => ({ k, m: mean(ALL.filter((i) => keyOf(i) === k).map((i) => v[i])) }));
    F1.layer('mean').selectAll('line').data(mns).join('line').attr('x1', (d) => F1.x(d.k) + bw * 0.08).attr('x2', (d) => F1.x(d.k) + bw * 0.92).attr('y1', (d) => F1.y(d.m)).attr('y2', (d) => F1.y(d.m)).attr('stroke', ink()).attr('stroke-width', 3);
    F1.layer('mean').selectAll('text').data(mns).join('text').attr('x', (d) => F1.x(d.k) + bw * 0.92).attr('y', (d) => F1.y(d.m) - 6).attr('text-anchor', 'end').attr('font-size', 11.5).attr('fill', ink()).text((d) => f2(d.m));
    s1.set({ 'log2FC, two groups': f2(plain.lfc[g]), 'p, two groups': pE(plain.p[g]), 'log2FC adjusted for centre': f2(lim.lfc[g]), 'p adjusted (moderated t)': pE(lim.p[g]), 'q adjusted': pE(lim.q[g]) });
  }
  paint1();
  const ft = fillerTrap;
  const within = (g, c) => { const r = ALL.filter((i) => L.centreB[i] === c); return mean(r.filter((i) => L.y[i]).map((i) => X[i * G + g])) - mean(r.filter((i) => !L.y[i]).map((i) => X[i * G + g])); };
  prose(rb, `<p>For CXCL9 the responder–non-responder difference is present within each centre (${f2(within(cx, 0))} in Centre A, ${f2(within(cx, 1))} in Centre B). For the filler gene ${L.genes[ft]} the within-centre differences are ${f2(within(ft, 0))} and ${f2(within(ft, 1))}, yet the two-group comparison gives log2FC ${f2(plain.lfc[ft])} and p = ${pE(plain.p[ft])}: the gene is shifted between centres, and responders are more common in Centre B. The centre-adjusted p is ${pE(lim.p[ft])}.</p>`);

  // ---- Step 2: linear model + centre
  rb = rung(ch, 2, 'A linear model per gene, with the centre in the design',
    `<p>The same comparison is written as a linear regression for each gene <i>g</i>:</p>
     <div class="calc">expression_g = b0 + b1 × response + b2 × centre B + error</div>
     <p><b>b1</b> is the log2 fold change between responders and non-responders <b>within a centre</b>; b2 absorbs the centre shift. Every gene uses the same <b>design matrix</b> (one row per patient), so ${nf(G)} small regressions are fitted in one matrix operation. With only the intercept and response columns, the model reproduces the two-group t-test of step 1 exactly. The first rows:</p>`);
  table(rb, ALL.slice(0, 5).map((i) => ({ id: L.ids[i], b0: 1, r: L.y[i], c: L.centreB[i], e: f2(X[i * G + cx]) })), [['id', 'Patient'], ['b0', 'Intercept'], ['r', 'Response'], ['c', 'Centre B'], ['e', 'CXCL9 log-CPM']], { cls: 'compact' });
  const fg2 = figure(rb, 'Every gene, with and without the centre term',
    'Each dot is a gene. x: evidence from a Welch t-test without the centre; y: evidence from the linear model with the centre. Dashed lines mark the FDR 5% cut-off of each analysis. Genes far right but low have a difference explained by the centre. Square-root axes.');
  let hl2 = 'all';
  select(fg2.controls, { label: 'Highlight', value: hl2, options: [['all', 'all groups'], ...GROUP_ORDER.map((g) => [g, groupLabel(g)])], onChange: (v) => { hl2 = v; paint2(); } });
  const s2 = stats(fg2.root);
  const mx2 = d3.max([...M.welch.p, ...M.lm.p], nlp) * 1.03;
  const F2 = frame(fg2.body, { width: 820, height: 380, x: [0, mx2], y: [0, mx2], xScale: d3.scaleSqrt(), yScale: d3.scaleSqrt(), xLabel: '−log10 p, Welch t-test without centre', yLabel: '−log10 p, model with centre', xTicks: 8 });
  const cut = (m) => nlp(d3.max(M[m].p.filter((p, i) => M[m].q[i] < 0.05)));
  F2.layer('cut').append('line').attr('x1', F2.x(cut('welch'))).attr('x2', F2.x(cut('welch'))).attr('y1', 0).attr('y2', F2.h).attr('stroke', ink2()).attr('stroke-dasharray', '5 4');
  F2.layer('cut').append('line').attr('y1', F2.y(cut('lm'))).attr('y2', F2.y(cut('lm'))).attr('x1', 0).attr('x2', F2.w).attr('stroke', ink2()).attr('stroke-dasharray', '5 4');
  const ord = d3.range(G).sort((a, b) => (grp[b] === 'filler') - (grp[a] === 'filler'));
  const dots2 = F2.layer('pts').selectAll('circle').data(ord).join('circle').attr('cx', (i) => F2.x(nlp(M.welch.p[i]))).attr('cy', (i) => F2.y(nlp(M.lm.p[i])))
    .attr('stroke', css('--surface')).attr('stroke-width', 0.6)
    .on('mousemove', (e, i) => tip(e, `<b>${L.genes[i]}</b> · ${groupLabel(grp[i])}<br>without centre: p ${pE(M.welch.p[i])}, q ${pE(M.welch.q[i])}<br>with centre: p ${pE(M.lm.p[i])}, q ${pE(M.lm.q[i])}`)).on('mouseleave', untip);
  legend(fg2.body, GROUP_ORDER.map((g) => [groupLabel(g), groupColor(g)]));
  function paint2() {
    const on = (i) => hl2 === 'all' || grp[i] === hl2;
    dots2.attr('r', (i) => (grp[i] === 'filler' ? 2.6 : 4.2)).attr('fill', (i) => groupColor(grp[i])).attr('opacity', (i) => (on(i) ? (grp[i] === 'filler' ? 0.55 : 1) : 0.08));
    const h = hl2 === 'all' ? null : hl2, f = (m) => (h ? `${M[m].by_group[h][0]} of ${M[m].by_group[h][1]}` : nf(M[m].n_hits));
    s2.set({ 'called without centre (q < 0.05)': f('welch'), 'called with centre': f('lm'), 'filler genes: without → with': `${M.welch.by_group.filler[0]} → ${M.lm.by_group.filler[0]}` });
  }
  paint2();
  const topIF = d3.range(G).sort((a, b) => M.lm.p[a] - M.lm.p[b]).slice(0, IF.length).filter((i) => grp[i] === 'IFNG module').length;
  prose(rb, `<p>The centre term removes ${M.welch.by_group.filler[0] - M.lm.by_group.filler[0]} filler-gene calls and keeps all ${M.lm.by_group['IFNG module'][0]} IFN-γ module genes${topIF === IF.length ? `, which are also the ${IF.length} strongest results` : ` (${topIF} of them among the ${IF.length} strongest results)`}. The model cannot fully separate centre from response, because the two are correlated: the fold change is estimated from the contrast within each centre, which leaves less information than ${NP} patients would provide in a balanced design.</p>`);

  // ---- Step 3: empirical Bayes
  rb = rung(ch, 3, 'Borrowing strength across genes: variance moderation',
    `<p>The standard error of log2FC depends on each gene's residual variance s². With few samples, s² is itself very noisy; a gene that is stable by chance gets a tiny standard error and a large t. limma's <b>empirical Bayes</b> step fits a distribution to the s² of all genes (a prior with typical value s₀² and weight d₀, in degrees of freedom) and replaces each s² by a weighted average:</p>
     <div class="calc">s̃² = (d₀ × s₀² + d × s²) / (d₀ + d)        moderated t = log2FC / (s̃ × √v),  on d + d₀ degrees of freedom</div>
     <p>d is the residual degrees of freedom of one gene (patients − design columns). The prior acts like d₀ extra observations of a typical variance; the data decide how large d₀ is.</p>`);
  const fg3 = figure(rb, 'Shrinking variances towards a common value',
    'Each dot is a gene: its own residual SD (x) against the moderated SD (y). On the diagonal nothing changes; with stronger moderation the dots flatten towards the typical value s₀ (dotted). Smaller studies are drawn at random from the discovery cohort; the first 5 + 5 draw shown is the first in which the moderated analysis calls at least one gene. Log axes.');
  const SIZES3 = [['all', 'all 177 patients'], [15, '15 + 15 patients'], [8, '8 + 8 patients'], [5, '5 + 5 patients']];
  const cache3 = new Map(), get3 = (k, sd) => { const key = k + ':' + sd; if (!cache3.has(key)) cache3.set(key, baseFit(drawStudy(k, sd), true)); return cache3.get(key); };
  const seed3 = d3.range(1, 60).find((sd) => nCalls(moderate(get3(5, sd), null).q) > 0) || 1;
  let k3 = 5, sd3 = seed3, d03 = get3(k3, sd3).d0est;
  const reset3 = () => { d03 = get3(k3, sd3).d0est; sd0.set(Math.min(150, d03)); paint3(); };
  select(fg3.controls, { label: 'Study size', value: '5', options: SIZES3.map(([k, l]) => [String(k), l]), onChange: (v) => { k3 = v === 'all' ? 'all' : +v; reset3(); } });
  const sd0 = slider(fg3.controls, { label: 'prior weight d₀', min: 0, max: 150, step: 0.5, value: Math.min(150, d03), fmt: (v) => f1(v), onInput: (v) => { d03 = v; paint3(); } });
  button(fg3.controls, 'Use the estimated d₀', reset3);
  button(fg3.controls, 'Draw another study', () => { sd3++; reset3(); });
  const s3 = stats(fg3.root);
  const ext3 = d3.extent(SIZES3.flatMap(([k]) => Array.from(get3(k, seed3).s2, Math.sqrt))), dom3 = [ext3[0] * 0.8, ext3[1] * 1.25];
  const F3 = frame(fg3.body, { width: 820, height: 360, x: dom3, y: dom3, xScale: d3.scaleLog().clamp(true), yScale: d3.scaleLog().clamp(true), xLabel: 'residual SD of the gene (s)', yLabel: 'moderated SD (s̃)' });
  F3.g.select('.axis.x').call(d3.axisBottom(F3.x).ticks(6, '.2~f')); F3.g.select('.axis.y').call(d3.axisLeft(F3.y).ticks(6, '.2~f'));
  F3.layer('diag').append('line').attr('x1', 0).attr('y1', F3.h).attr('x2', F3.w).attr('y2', 0).attr('stroke', muted()).attr('stroke-dasharray', '4 4');
  const s0L = F3.layer('s0').append('line').attr('x1', 0).attr('x2', F3.w).attr('stroke', ink2()).attr('stroke-dasharray', '2 3').attr('stroke-width', 1.5);
  const pts3 = F3.layer('pts');
  legend(fg3.body, GROUP_ORDER.map((g) => [groupLabel(g), groupColor(g)]));
  function paint3() {
    const B = get3(k3, sd3), R = moderate(B, d03), R0 = moderate(B, 0);
    pts3.selectAll('circle').data(ord).join('circle').attr('r', (i) => (grp[i] === 'filler' ? 2.4 : 4)).attr('fill', (i) => groupColor(grp[i])).attr('opacity', (i) => (grp[i] === 'filler' ? 0.5 : 1))
      .attr('stroke', css('--surface')).attr('stroke-width', 0.5).attr('cx', (i) => F3.x(Math.sqrt(B.s2[i]))).attr('cy', (i) => F3.y(Math.sqrt(R.s2m[i])))
      .on('mousemove', (e, i) => tip(e, `<b>${L.genes[i]}</b> · ${groupLabel(grp[i])}<br>s = ${f2(Math.sqrt(B.s2[i]))} → s̃ = ${f2(Math.sqrt(R.s2m[i]))}<br>t = ${f1(R0.t[i])} → moderated ${f1(R.t[i])}`)).on('mouseleave', untip);
    const y0 = d03 > 0 ? Math.sqrt(R.s02) : null;
    s0L.attr('opacity', y0 ? 1 : 0).attr('y1', y0 ? F3.y(y0) : 0).attr('y2', y0 ? F3.y(y0) : 0);
    s3.set({ 'residual df d': B.d, 'estimated d₀': f1(B.d0est), 'prior weight d₀ / (d₀ + d)': isFinite(d03) ? pc(d03 / (d03 + B.d)) : '100%', 'calls (q < 0.05), no moderation': nCalls(R0.q), 'with this d₀': nCalls(R.q), 'IFN-γ genes called': `${callsBy(R.q)['IFNG module']} of ${IF.length}` });
  }
  paint3();
  const Ball = get3('all', seed3), agg = d3.range(101, 121).map((sd) => { const B = baseFit(drawStudy(5, sd), true), r0 = moderate(B, 0), rm = moderate(B, null); return { d: B.d, d0: B.d0est, n0: nCalls(r0.q), nm: nCalls(rm.q), i0: callsBy(r0.q)['IFNG module'], im: callsBy(rm.q)['IFNG module'] }; });
  const AG = { n: agg.length, any0: agg.filter((o) => o.n0 > 0).length, anym: agg.filter((o) => o.nm > 0).length, i0: mean(agg.map((o) => o.i0)), im: mean(agg.map((o) => o.im)), d: agg[0].d, d0: d3.median(agg, (o) => o.d0) };
  prose(rb, `<p>With all patients, d = ${Ball.d} and the estimated d₀ = ${f1(Ball.d0est)}: the prior carries ${pc(Ball.d0est / (Ball.d0est + Ball.d))} of the weight and the call count barely changes (${M.lm.n_hits} without moderation, ${M.limma.n_hits} with). With 5 + 5 patients, d = ${AG.d} and the median estimated d₀ is ${f1(AG.d0)}, so the prior carries more weight than the gene's own data. Across ${AG.n} random draws of 5 + 5 patients, the ordinary t-test called at least one gene in <b>${AG.any0}</b> draws and the moderated t in <b>${AG.anym}</b>; on average ${f1(AG.i0)} vs ${f1(AG.im)} of the ${IF.length} IFN-γ genes were called. Moderation recovers power in small experiments, which is what it was designed for, but it cannot replace samples: ${AG.n - AG.anym} of the ${AG.n} small studies still call nothing at FDR 5%.</p>`);

  // ---- Step 4: many tests, p-value histogram, BH
  rb = rung(ch, 4, 'Thousands of tests: the p-value histogram and the false discovery rate',
    `<p>At p &lt; 0.05, about 5% of genes with no true difference are called. The <b>Benjamini–Hochberg</b> (BH) procedure instead controls the expected share of false calls in the list: sort the m p-values, find the largest rank i with p<sub>(i)</sub> ≤ α × i / m, and call every gene up to that rank. The <b>q-value</b> of a gene is the smallest α at which it would be called. Bonferroni (p &lt; α / m) controls the chance of even one false call and is much stricter.</p>`);
  const fg4 = figure(rb, 'p-value histogram and the BH line',
    'Left: distribution of all p-values; the dashed line is the flat height expected if no gene differed. Right: sorted p-values against their rank (log axes) with the BH line α × i / m; points below the line up to the last crossing are called. Hover the bars.');
  const MS4 = { limma: 'moderated t, with centre', welch: 'Welch t-test, without centre', perm: 'model with centre, response labels permuted' };
  let m4 = 'limma', a4 = 0.05;
  select(fg4.controls, { label: 'Analysis', value: m4, options: Object.entries(MS4), onChange: (v) => { m4 = v; paint4(); } });
  slider(fg4.controls, { label: 'FDR α', min: 0.01, max: 0.25, step: 0.01, value: a4, fmt: (v) => v.toFixed(2), onInput: (v) => { a4 = v; paint4(); } });
  const s4 = stats(fg4.root);
  const row4 = el('div', { class: 'fig-row' }); fg4.body.append(row4);
  const c4a = el('div'), c4b = el('div'); row4.append(c4a, c4b);
  const H4 = frame(c4a, { width: 470, height: 300, x: [0, 1], y: [0, 1], xLabel: 'p-value', yLabel: 'genes' });
  const minP = d3.min(['limma', 'welch', 'perm'].flatMap((m) => M[m].p.filter((p) => p > 0)));
  const B4f = frame(c4b, { width: 470, height: 300, x: [1, G], y: [minP, 1], xScale: d3.scaleLog(), yScale: d3.scaleLog(), xLabel: 'rank i (log)', yLabel: 'p-value (log)', yTicks: 4 });
  B4f.g.select('.axis.y').call(d3.axisLeft(B4f.y).ticks(4, '.0e'));
  function paint4() {
    const p = M[m4].p, NB = 20, cnt = new Array(NB).fill(0);
    p.forEach((v) => { cnt[Math.min(NB - 1, Math.floor(v * NB))]++; });
    H4.y.domain([0, d3.max(cnt) * 1.05]); H4.g.select('.axis.y').call(d3.axisLeft(H4.y).ticks(5)); H4.g.select('.gridline').call(d3.axisLeft(H4.y).ticks(5).tickSize(-H4.w).tickFormat(''));
    H4.layer('bars').selectAll('rect').data(cnt).join('rect').attr('x', (c, i) => H4.x(i / NB) + 1).attr('width', H4.x(1 / NB) - 2).attr('y', (c) => H4.y(c)).attr('height', (c) => H4.h - H4.y(c)).attr('fill', css('--accent'))
      .on('mousemove', (e, c) => { const i = cnt.indexOf(c); tip(e, `${c} genes with p in [${(i / NB).toFixed(2)}, ${((i + 1) / NB).toFixed(2)})`); }).on('mouseleave', untip);
    H4.layer('u').selectAll('line').data([G / NB]).join('line').attr('x1', 0).attr('x2', H4.w).attr('y1', (u) => H4.y(u)).attr('y2', (u) => H4.y(u)).attr('stroke', ink()).attr('stroke-dasharray', '5 4').attr('stroke-width', 1.5);
    const sp = p.map((v) => Math.max(v, minP)).sort((a, b) => a - b);
    let kmax = 0; sp.forEach((v, i) => { if (v <= a4 * (i + 1) / G) kmax = i + 1; });
    B4f.layer('pts').selectAll('circle').data(sp).join('circle').attr('cx', (v, i) => B4f.x(i + 1)).attr('cy', (v) => B4f.y(v)).attr('r', 2.2).attr('fill', (v, i) => (i < kmax ? cat(7) : muted()));
    B4f.layer('line').selectAll('path').data([0]).join('path').attr('fill', 'none').attr('stroke', ink()).attr('stroke-width', 2)
      .attr('d', d3.line().x((i) => B4f.x(i)).y((i) => B4f.y(Math.max(minP, a4 * i / G)))(d3.range(1, G + 1, 4).concat([G])));
    s4.set({ 'p < 0.05': p.filter((v) => v < 0.05).length, 'expected if no gene differed': Math.round(0.05 * G), [`BH calls at α = ${a4.toFixed(2)}`]: kmax, 'Bonferroni calls': p.filter((v) => v < a4 / G).length, 'expected false among BH calls': f1(a4 * kmax) });
  }
  paint4();
  legend(fg4.body, [['called by BH', cat(7)], ['not called', muted()], ['BH line α·i/m', ink()]]);
  prose(rb, `<p>The p-value histogram is the first diagnostic. With the centre in the model it shows a peak near 0 that tapers into a flat floor: real differences, many of them small, on top of genes with no difference. Without the centre, ${pc(M.welch.p.filter((v) => v < 0.05).length / G)} of all genes have p &lt; 0.05. With permuted labels the histogram is flat and BH calls ${M.perm.n_hits} genes, which is the expected behaviour when no gene is associated with the (shuffled) outcome.</p>`);
  quiz(rb, `A gene has q = 0.03. Is there a 3% probability that this particular gene is a false positive?`,
    `<p>Not exactly. q = 0.03 means that if every gene with q ≤ 0.03 is called, about 3% of that list is expected to be false. Genes near the cut-off are more likely to be false than genes at the top of the list; the per-gene probability (the local FDR) is a different quantity.</p>`);

  // ---- Step 5: volcano and the planted biology
  rb = rung(ch, 5, 'The volcano plot: effect size against evidence',
    '<p>The volcano plot puts log2 fold change on the x-axis and −log10 p on the y-axis, so the genes that are both strongly and reliably different sit in the upper corners. Because the data are simulated, every gene\'s planted role is known (colour).</p>');
  const fg5 = figure(rb, 'Volcano plot, moderated t with centre',
    'Called genes are solid, others faded. Right: the share of each planted gene group that is called. Hover a gene or bar.');
  let a5 = 0.05, l5 = 0;
  slider(fg5.controls, { label: 'FDR α', min: 0.01, max: 0.25, step: 0.01, value: a5, fmt: (v) => v.toFixed(2), onInput: (v) => { a5 = v; paint5(); } });
  slider(fg5.controls, { label: 'minimum |log2FC|', min: 0, max: 1, step: 0.05, value: 0, fmt: f2, onInput: (v) => { l5 = v; paint5(); } });
  const s5 = stats(fg5.root);
  const row5 = el('div', { class: 'fig-row' }); fg5.body.append(row5);
  const c5a = el('div'), c5b = el('div'); row5.append(c5a, c5b); c5b.style.flex = '0 1 330px';
  const xm5 = d3.max(lim.lfc, Math.abs) * 1.05;
  const F5 = frame(c5a, { width: 500, height: 380, x: [-xm5, xm5], y: [0, d3.max(lim.p, nlp) * 1.05], xLabel: 'log2 fold change (responders vs non-responders)', yLabel: '−log10 p' });
  const dots5 = F5.layer('pts').selectAll('circle').data(ord).join('circle').attr('cx', (i) => F5.x(lim.lfc[i])).attr('cy', (i) => F5.y(nlp(lim.p[i])))
    .attr('r', (i) => (grp[i] === 'filler' ? 2.6 : 4.2)).attr('fill', (i) => groupColor(grp[i])).attr('stroke', css('--surface')).attr('stroke-width', 0.6)
    .on('mousemove', (e, i) => tip(e, `<b>${L.genes[i]}</b> · ${groupLabel(grp[i])}<br>log2FC ${f2(lim.lfc[i])} (true ${f2(L.truth_lfc[i])})<br>p ${pE(lim.p[i])}, q ${pE(lim.q[i])}`)).on('mouseleave', untip);
  const labs5 = F5.layer('labs');
  const FB = frame(c5b, { width: 330, height: 380, x: [0, 1], y: [0, 1], xLabel: 'share of group called', margin: { left: 118, right: 46 }, yScale: d3.scaleBand().padding(0.25), grid: false });
  FB.y.domain(GROUP_ORDER.slice().reverse()); FB.g.select('.axis.y').call(d3.axisLeft(FB.y).tickFormat(groupLabel));
  FB.g.select('.axis.x').call(d3.axisBottom(FB.x).ticks(4).tickFormat(d3.format('.0%')));
  legend(fg5.body, GROUP_ORDER.map((g) => [groupLabel(g), groupColor(g)]));
  function paint5() {
    const hit = (i) => lim.q[i] < a5 && Math.abs(lim.lfc[i]) >= l5;
    dots5.attr('opacity', (i) => (hit(i) ? 1 : 0.15));
    const top = d3.range(G).filter(hit).sort((a, b) => lim.p[a] - lim.p[b]).slice(0, 5);
    const lx = F5.x(0.55), ly = (k) => F5.y(nlp(lim.p[top[0]] || 1)) + 2 + k * 15;
    labs5.selectAll('line').data(top).join('line').attr('x1', (i) => F5.x(lim.lfc[i])).attr('y1', (i) => F5.y(nlp(lim.p[i]))).attr('x2', lx + 3).attr('y2', (i, k) => ly(k) - 4).attr('stroke', muted()).attr('stroke-width', 0.8);
    labs5.selectAll('text').data(top).join('text').attr('x', lx).attr('y', (i, k) => ly(k)).attr('text-anchor', 'end').attr('font-size', 11).attr('fill', ink()).text((i) => L.genes[i]);
    const bg = callsBy(lim.q, a5, (i) => Math.abs(lim.lfc[i]) >= l5), nh = sum(Object.values(bg));
    FB.layer('bars').selectAll('rect').data(GROUP_ORDER).join('rect').attr('y', (g) => FB.y(g)).attr('height', FB.y.bandwidth()).attr('x', 0).attr('width', (g) => FB.x(bg[g] / nGrp[g])).attr('fill', (g) => groupColor(g))
      .on('mousemove', (e, g) => tip(e, `<b>${groupLabel(g)}</b><br>${bg[g]} of ${nGrp[g]} genes called`)).on('mouseleave', untip);
    FB.layer('lab').selectAll('text').data(GROUP_ORDER).join('text').attr('y', (g) => FB.y(g) + FB.y.bandwidth() / 2 + 4).attr('x', (g) => FB.x(bg[g] / nGrp[g]) + 5).attr('font-size', 11).attr('fill', ink2()).text((g) => `${bg[g]}/${nGrp[g]}`);
    s5.set({ 'genes called': nh, 'higher / lower in responders': `${d3.range(G).filter((i) => hit(i) && lim.lfc[i] > 0).length} / ${d3.range(G).filter((i) => hit(i) && lim.lfc[i] < 0).length}`, 'IFN-γ module': `${bg['IFNG module']} / ${nGrp['IFNG module']}`, 'proliferation module': `${bg['PROLIF module']} / ${nGrp['PROLIF module']}`, 'filler genes': `${bg.filler} / ${nGrp.filler}` });
  }
  paint5();
  const tl = L.truth_lfc, tf = L.tumour_frac, callsL = d3.range(G).filter((i) => lim.q[i] < 0.05);
  const tfR = mean(ALL.filter((i) => L.y[i]).map((i) => tf[i])), tfN = mean(ALL.filter((i) => !L.y[i]).map((i) => tf[i]));
  const prCalled = PR.filter((i) => lim.q[i] < 0.05), tinyCalls = callsL.filter((i) => Math.abs(tl[i]) < 0.1), wrongSign = callsL.filter((i) => Math.sign(tl[i]) !== Math.sign(lim.lfc[i]));
  const infl = d3.median(tinyCalls.map((i) => Math.abs(lim.lfc[i]) / Math.max(0.01, Math.abs(tl[i]))));
  prose(rb, `<ul>
    <li><b>The planted IFN-γ module is recovered</b>: all ${IF.length} genes are called with log2FC between ${f2(d3.min(IF, (i) => lim.lfc[i]))} and ${f2(d3.max(IF, (i) => lim.lfc[i]))}, close to their true values (${f2(d3.min(IF, (i) => tl[i]))} to ${f2(d3.max(IF, (i) => tl[i]))}).</li>
    <li><b>Proliferation genes are prognostic, not predictive</b> in the simulation: they shorten survival but do not affect response. ${prCalled.length ? `${prCalled.length} of ${PR.length} (${prCalled.map((i) => L.genes[i]).join(', ')}) ${prCalled.length === 1 ? 'is' : 'are'} nevertheless called, lower in responders.` : `None of the ${PR.length} is called.`} The reason is composition: responding tumours contain more immune cells and fewer tumour cells (true tumour fraction ${pc(tfR)} vs ${pc(tfN)}), which dilutes every tumour-cell gene in bulk. The true bulk differences of the module are small (${f2(d3.min(PR, (i) => tl[i]))} to ${f2(d3.max(PR, (i) => tl[i]))}) against large between-patient variation, so most are not called. A DE call describes the bulk tissue; it does not show that a gene is regulated, or that it is relevant to response.</li>
    <li><b>Not every call is a planted effect.</b> ${tinyCalls.length} of ${callsL.length} calls have a true bulk difference below 0.1 in absolute value; these are small real composition effects whose estimates are larger than the truth by a median factor of ${f1(infl)}, because genes are selected for having a large estimate (the "winner's curse"). ${wrongSign.length} call${wrongSign.length === 1 ? ' has' : 's have'} the wrong sign compared with the truth. In bulk tissue almost no gene has a difference of exactly zero, so effect sizes matter more than the call itself.</li></ul>`);

  // ---- Step 6: the DE lab
  rb = rung(ch, 6, 'DE lab: design, sample size and replication',
    `<p>Draw a study from the ${NP} discovery patients, choose the analysis, and check the resulting gene list in two independent ways: against the <b>noise-free truth</b> of the simulation, and by <b>replication in the ${L.val.n} Centre C patients</b> (${L.val.n_resp} responders), analysed separately with a plain linear model. A call replicates if the Centre C log2FC has the same sign with p &lt; 0.05.</p>`);
  const fl = figure(rb, 'DE lab', 'Left: volcano plot of the drawn study (called genes solid). Right: log2FC in this study against log2FC in Centre C; called genes are solid, the rest faded. Points in the upper-right and lower-left quadrants agree in direction.');
  const labCfg = { k: 'all', centre: true, mod: true, a: 0.05, seed: 1 };
  const SZ = [['all', 'all 177 patients'], ['30', '30 + 30'], ['15', '15 + 15'], ['8', '8 + 8'], ['5', '5 + 5']];
  select(fl.controls, { label: 'Study size', value: 'all', options: SZ, onChange: (v) => { labCfg.k = v === 'all' ? 'all' : +v; paintL(); } });
  select(fl.controls, { label: 'Design', value: 'c', options: [['c', '~ response + centre'], ['r', '~ response']], onChange: (v) => { labCfg.centre = v === 'c'; paintL(); } });
  select(fl.controls, { label: 'Variance', value: 'm', options: [['m', 'moderated (limma)'], ['p', 'per gene (ordinary t)']], onChange: (v) => { labCfg.mod = v === 'm'; paintL(); } });
  slider(fl.controls, { label: 'FDR α', min: 0.01, max: 0.25, step: 0.01, value: 0.05, fmt: (v) => v.toFixed(2), onInput: (v) => { labCfg.a = v; paintL(); } });
  button(fl.controls, 'Draw another study', () => { labCfg.seed++; paintL(); }, true);
  const sL = stats(fl.root);
  const rowL = el('div', { class: 'fig-row' }); fl.body.append(rowL);
  const la = el('div'), lb = el('div'); rowL.append(la, lb);
  const FL = frame(la, { width: 470, height: 340, x: [-2.5, 2.5], y: [0, 1], xLabel: 'log2FC in this study', yLabel: '−log10 p' });
  const FR = frame(lb, { width: 470, height: 340, x: [-2.5, 2.5], y: [-2.2, 2.2], xLabel: 'log2FC in this study', yLabel: 'log2FC in Centre C' });
  [[FR.x(0), 0, FR.x(0), FR.h], [0, FR.y(0), FR.w, FR.y(0)]].forEach(([x1, y1, x2, y2]) => FR.layer('ax0').append('line').attr('x1', x1).attr('y1', y1).attr('x2', x2).attr('y2', y2).attr('stroke', muted()));
  FR.layer('ax0').append('line').attr('x1', FR.x(-2.2)).attr('y1', FR.y(-2.2)).attr('x2', FR.x(2.2)).attr('y2', FR.y(2.2)).attr('stroke', muted()).attr('stroke-dasharray', '4 4');
  const msgL = el('p', { class: 'fig-caption' }); fl.body.append(msgL);
  legend(fl.body, GROUP_ORDER.map((g) => [groupLabel(g), groupColor(g)]));
  const vl = L.val.lfc, vp = L.val.p;
  const labCache = new Map();
  const labRun = (c) => {
    const key = [c.k, c.centre, c.k === 'all' ? 0 : c.seed].join(':');
    if (!labCache.has(key)) { if (labCache.size > 40) labCache.clear(); labCache.set(key, baseFit(drawStudy(c.k, c.seed), c.centre)); }
    const B = labCache.get(key), R = moderate(B, c.mod ? null : 0);
    const calls = d3.range(G).filter((i) => R.q[i] < c.a), rep = calls.filter((i) => vp[i] < 0.05 && Math.sign(vl[i]) === Math.sign(R.lfc[i]));
    return { R, calls, rep, by: callsBy(R.q, c.a), wrong: calls.filter((i) => Math.sign(tl[i]) !== Math.sign(R.lfc[i])).length };
  };
  function paintL() {
    const o = labRun(labCfg), R = o.R, hit = new Set(o.calls), cl = (v) => Math.max(-2.5, Math.min(2.5, v));
    FL.y.domain([0, Math.max(5, d3.max(R.p, nlp) * 1.05)]); FL.g.select('.axis.y').call(d3.axisLeft(FL.y).ticks(5)); FL.g.select('.gridline').call(d3.axisLeft(FL.y).ticks(5).tickSize(-FL.w).tickFormat(''));
    const tipL = (e, i) => tip(e, `<b>${L.genes[i]}</b> · ${groupLabel(grp[i])}<br>this study: log2FC ${f2(R.lfc[i])}, q ${pE(R.q[i])}<br>Centre C: log2FC ${f2(vl[i])}, p ${pE(vp[i])}<br>true log2FC ${f2(tl[i])}`);
    const styl = (s) => s.attr('r', (i) => (grp[i] === 'filler' ? 2.4 : 4)).attr('fill', (i) => groupColor(grp[i])).attr('opacity', (i) => (hit.has(i) ? 1 : 0.12)).attr('stroke', css('--surface')).attr('stroke-width', 0.5).on('mousemove', tipL).on('mouseleave', untip);
    styl(FL.layer('pts').selectAll('circle').data(ord).join('circle').attr('cx', (i) => FL.x(cl(R.lfc[i]))).attr('cy', (i) => FL.y(nlp(R.p[i]))));
    styl(FR.layer('pts').selectAll('circle').data(ord).join('circle').attr('cx', (i) => FR.x(cl(R.lfc[i]))).attr('cy', (i) => FR.y(Math.max(-2.2, Math.min(2.2, vl[i])))));
    sL.set({ 'genes called': o.calls.length, 'IFN-γ module': `${o.by['IFNG module']} / ${IF.length}`, 'filler genes called': o.by.filler, 'replicated in Centre C': o.calls.length ? `${o.rep.length} (${pc(o.rep.length / o.calls.length)})` : '—', 'wrong sign vs truth': o.wrong, 'd₀ (prior df)': labCfg.mod ? f1(R.d0) : '0' });
    msgL.textContent = labCfg.centre && !R.useC ? 'All drawn patients come from one centre, so the centre term was dropped from this design.' : `Residual df per gene: ${R.d}${labCfg.mod ? ` + ${f1(R.d0)} from the prior` : ''}.`;
  }
  paintL();
  const T = (c) => labRun({ k: 'all', centre: true, mod: true, a: 0.05, seed: 1, ...c });
  const t1a = T({ centre: false, mod: false }), t1b = T({});
  const runs3 = d3.range(1, 6).map((s) => T({ k: 15, seed: s })), rg = (f) => `${d3.min(runs3, f)}–${d3.max(runs3, f)}`;
  const t4 = T({ a: 0.2 });
  const repTxt = (o) => (o.calls.length ? pc(o.rep.length / o.calls.length) : '—');
  prose(rb, '<p><b>Suggested comparisons</b>:</p>');
  table(rb, [
    { t: '<b>~ response</b>, ordinary t → <b>~ response + centre</b>, moderated', s: `Calls ${t1a.calls.length} → <b>${t1b.calls.length}</b>; filler genes ${t1a.by.filler} → ${t1b.by.filler}; replicated in Centre C ${repTxt(t1a)} → <b>${repTxt(t1b)}</b>`, m: 'Without the centre, most calls are centre effects that do not replicate at a third site.' },
    { t: '<b>5 + 5</b> patients: ordinary t → moderated, then draw again several times', s: `Over ${AG.n} draws (step 3): studies with any call ${AG.any0} → <b>${AG.anym}</b>; IFN-γ genes called on average ${f1(AG.i0)} → <b>${f1(AG.im)}</b>`, m: `With d = ${AG.d} residual degrees of freedom, per-gene variances are unreliable; borrowing d₀ ≈ ${f1(AG.d0)} from other genes recovers power, but most small studies remain underpowered.` },
    { t: '<b>15 + 15</b> patients, draw five studies', s: `Calls ${rg((o) => o.calls.length)}; IFN-γ genes ${rg((o) => o.by['IFNG module'])} of ${IF.length}`, m: 'Lists from small studies vary from draw to draw; a gene missing from one list is not evidence that it does not differ.' },
    { t: 'All patients, α <b>0.05</b> → <b>0.20</b>', s: `Calls ${t1b.calls.length} → <b>${t4.calls.length}</b>; replicated ${repTxt(t1b)} → ${repTxt(t4)}`, m: 'A looser FDR buys more genes at the cost of a less reliable list; α is chosen before looking at the results.' },
  ], [['t', 'Try'], ['s', 'What you see'], ['m', 'What it means']], { cls: 'compact' });
  prose(rb, `<p>Replication is never 100% even for a well-designed analysis: Centre C has its own case mix and noise, and a call with a small effect may not reach p &lt; 0.05 in ${L.val.n} patients. The two cohorts also differ in case mix, so some true differences are not identical between them.</p>`);

  // =====================================================================================
  // 5. PITFALLS
  // =====================================================================================
  ch = chapter(main, 'pitfalls', 'Pitfalls', 'Common problems, how to recognise them, and how to address them.');
  const caf = callsBy(lim.q)['CAF/TGF-b module'];
  cards(ch, [
    { tag: `Symptom: ${nf(M.welch.n_hits)} "significant" genes`, title: 'Batch or centre left out of the design', body: `Without the centre term ${M.welch.by_group.filler[0]} filler genes are called. <b>Fix:</b> put known technical factors in the design (<code>~ centre + response</code>), and check the p-value histogram and PCA before trusting a list.` },
    { tag: 'Symptom: thousands of tiny p-values in single-cell DE', title: 'Cells treated as replicates', body: 'Cells from one patient share that patient\'s biology, so a per-cell test overstates the sample size by orders of magnitude. <b>Fix:</b> sum counts per patient and cell type (pseudobulk) and use the bulk tools, or a mixed model with patient as a random effect.' },
    { tag: 'Symptom: "lower in responders" for tumour genes', title: 'Composition read as regulation', body: `Tumour-cell genes such as ${prCalled.length ? L.genes[prCalled[0]] : 'EPCAM'} appear lower in responders because responding tumours contain fewer tumour cells (${pc(tfR)} vs ${pc(tfN)}). <b>Fix:</b> interpret bulk DE as tissue-level differences; use deconvolution, single-cell or spatial data to locate the change.` },
    { tag: 'Symptom: tiny fold changes at q < 10⁻¹⁰', title: 'Ranking by p-value alone', body: 'With many samples, very small differences become significant. <b>Fix:</b> report log2FC with confidence intervals, rank by effect size among significant genes, or test against a minimum fold change (limma <code>treat</code>, DESeq2 <code>lfcThreshold</code>).' },
    { tag: 'Symptom: excellent cross-validated AUC from DE genes', title: 'DE selection outside cross-validation', body: 'Selecting genes by DE on all samples, then cross-validating a classifier on those genes, lets the test folds influence the selection. <b>Fix:</b> repeat the DE selection inside every training fold (next page).' },
    { tag: 'Symptom: "the CAF module is not involved"', title: 'Not significant read as no effect', body: `${caf} of ${nGrp['CAF/TGF-b module']} CAF/TGF-β genes are called, although this module drives the excluded phenotype. Absence of evidence at n = ${NP} is not evidence of absence, and the biology concerns where T cells are, which bulk data cannot show. <b>Fix:</b> report effect sizes and intervals; use spatial data for location questions.` },
    { tag: 'Symptom: clusters differ in their marker genes', title: 'Testing groups defined from the same data', body: 'If groups are found by clustering expression, DE between those clusters is guaranteed to find differences and the p-values are not valid. <b>Fix:</b> treat such lists as descriptive, or use held-out data or selective-inference methods.' },
    { tag: 'Symptom: list changes with every reanalysis', title: 'Too few samples, and no prior', body: `Across ${AG.n} draws of 5 + 5 patients, an ordinary t-test called any gene in ${AG.any0} and the moderated t in ${AG.anym}, and the lists differed from draw to draw. <b>Fix:</b> use moderated statistics (limma, DESeq2, edgeR), plan sample size in advance, and replicate.` },
  ]);

  // =====================================================================================
  // 6. RECAP
  // =====================================================================================
  ch = chapter(main, 'recap', 'Recap', 'Key points.');
  recap(ch, [
    'DE fits <b>one linear model per gene</b> on log expression; the response coefficient is the <b>log2 fold change</b>.',
    'It is <b>statistics, not machine learning</b>: the aim is an effect size with error control for the population, not a prediction for a new patient.',
    'Known technical factors (centre, batch) go <b>into the design</b>. Leaving them out turns batch differences into "response genes".',
    '<b>Empirical-Bayes moderation</b> shrinks each gene\'s variance towards a common value; it matters most with few samples.',
    'With thousands of tests, control the <b>false discovery rate</b> (Benjamini–Hochberg). Check the p-value histogram and a permutation first.',
    'Read the <b>volcano plot</b> for effect size and evidence together; the list recovers the planted IFN-γ module.',
    'Bulk DE reports <b>tissue-level</b> differences: composition (fewer tumour cells) can make prognostic tumour genes look response-related.',
    'Validate a list by <b>replication</b> in an independent cohort, and expect the effect sizes of weakly supported genes to be smaller there (the winner\'s curse).',
  ]);
  prose(ch, '<h3>In code</h3><p>Count-based tools (DESeq2, edgeR) and limma-voom implement the same ideas as this page with additional modelling of count noise. The centre is always in the design formula.</p>');
  code(ch, {
    Python: `# Counts: genes as columns, samples as rows; meta has "centre" and "response" (yes/no)
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

dds = DeseqDataSet(counts=counts, metadata=meta, design="~ centre + response")
dds.deseq2()                                       # size factors, dispersions (shrunk), GLM fit
res = DeseqStats(dds, contrast=["response", "yes", "no"])
res.summary()                                      # Wald test + Benjamini-Hochberg
table = res.results_df.sort_values("padj")
hits = table[(table.padj < 0.05)]
print(hits[["log2FoldChange", "pvalue", "padj"]].head(20))`,
    R: `library(edgeR); library(limma)
dge <- DGEList(counts)                             # genes x samples
keep <- filterByExpr(dge, group = meta$response); dge <- dge[keep, , keep.lib.sizes = FALSE]
dge <- calcNormFactors(dge)
design <- model.matrix(~ centre + response, data = meta)
v   <- voom(dge, design)                           # log-CPM + precision weights
fit <- eBayes(lmFit(v, design))                    # moderated t
tt  <- topTable(fit, coef = "responseyes", number = Inf, sort.by = "P")   # adj.P.Val = BH
hist(tt$P.Value, breaks = 40)                      # first diagnostic

# DESeq2 alternative
# dds <- DESeqDataSetFromMatrix(counts, meta, ~ centre + response); res <- results(DESeq(dds))`,
  });
  seeAlso(main, [['omics', 'bulk-qc-batch', 'Previous: QC, normalisation and batch effects'], ['omics', 'bulk-classifier', 'Next: biomarker classifier and nested CV'], ['concepts', 'multiple-testing', 'Multiple testing & FDR'], ['concepts', 'linear-regression', 'Linear regression'], ['omics', 'sc-pseudobulk', 'Pseudobulk differential expression'], ['omics', 'bulk-survival', 'Survival analysis']]);

  /** Pipeline diagram for one gene, with live numbers from the full limma-style fit. */
  function pipelineSvg(g) {
    const wrap = el('div', { class: 'figure' });
    wrap.append(el('div', { class: 'fig-title' }, `The analysis, for the gene ${L.genes[g]}`));
    const svg = d3.select(wrap).append('svg').attr('viewBox', '0 0 980 230').attr('width', 980).attr('role', 'img').attr('aria-label', 'expression, linear model, moderated t, false discovery rate');
    const v = col(g), m1 = mean(v.filter((x, i) => L.y[i])), m0 = mean(v.filter((x, i) => !L.y[i]));
    const boxes = [
      { t: 'One gene, all patients', l: [`log-CPM of ${L.genes[g]}`, `responders   ${f2(m1)}`, `  (n = ${nR})`, `non-resp.    ${f2(m0)}`, `  (n = ${NP - nR})`] },
      { t: '① Linear model', l: ['expr = b0 + b1·resp', '     + b2·centreB', `b1 = log2FC = ${f2(lim.lfc[g])}`, `s  = ${f2(Math.sqrt(baseC.s2[g]))}`] },
      { t: '② Moderated t', l: [`s̃  = ${f2(Math.sqrt(lim.s2m[g]))}`, `  (d₀ = ${f1(lim.d0)} prior df)`, `t̃  = ${f1(lim.t[g])}`, `p  = ${pE(lim.p[g])}`] },
      { t: '③ All genes → FDR', l: [`${nf(G)} p-values`, 'Benjamini–Hochberg', `q(${L.genes[g]}) = ${pE(lim.q[g])}`, `${nCalls(lim.q)} genes q < 0.05`] },
    ];
    const BW = 212, gap = 36, H = 200;
    boxes.forEach((b, i) => {
      const x0 = 10 + i * (BW + gap), gg = svg.append('g').attr('transform', `translate(${x0},14)`);
      gg.append('rect').attr('width', BW).attr('height', H).attr('rx', 10).attr('fill', css('--surface-2')).attr('stroke', css('--axis'));
      gg.append('text').attr('x', 12).attr('y', 24).attr('font-weight', 700).attr('font-size', 13.5).attr('fill', ink()).text(b.t);
      b.l.forEach((s, j) => gg.append('text').attr('x', 14).attr('y', 56 + j * 26).attr('font-family', css('--mono')).attr('font-size', 13).attr('fill', ink2()).attr('xml:space', 'preserve').text(s));
      if (i < boxes.length - 1) {
        const ax = x0 + BW + 6, ay = 14 + H / 2;
        svg.append('path').attr('d', `M${ax},${ay} L${ax + gap - 14},${ay} M${ax + gap - 20},${ay - 6} L${ax + gap - 14},${ay} L${ax + gap - 20},${ay + 6}`).attr('fill', 'none').attr('stroke', ink2()).attr('stroke-width', 2);
      }
    });
    return wrap;
  }
}

// ---------------------------------------------------------------------------------------------
// Special functions for the t distribution and limma's prior estimate (fitFDist), plain JS.
function bh(p) {
  const m = p.length, o = d3.range(m).sort((a, b) => p[a] - p[b]), q = new Array(m);
  let run = 1;
  for (let r = m - 1; r >= 0; r--) { run = Math.min(run, (p[o[r]] * m) / (r + 1)); q[o[r]] = run; }
  return q;
}
function lgamma(x) {
  const c = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
  let yy = x, t = x + 5.5; t -= (x + 0.5) * Math.log(t);
  let s = 1.000000000190015; for (const ci of c) s += ci / ++yy;
  return -t + Math.log(2.5066282746310005 * s / x);
}
function betacf(a, b, x) {
  const qab = a + b, qap = a + 1, qam = a - 1, tiny = 1e-300;
  let c = 1, d = 1 - qab * x / qap; if (Math.abs(d) < tiny) d = tiny; d = 1 / d; let h = d;
  for (let m = 1; m <= 300; m++) {
    const m2 = 2 * m; let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
    d = 1 + aa * d; if (Math.abs(d) < tiny) d = tiny; c = 1 + aa / c; if (Math.abs(c) < tiny) c = tiny; d = 1 / d; h *= d * c;
    aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
    d = 1 + aa * d; if (Math.abs(d) < tiny) d = tiny; c = 1 + aa / c; if (Math.abs(c) < tiny) c = tiny; d = 1 / d;
    const del = d * c; h *= del; if (Math.abs(del - 1) < 3e-12) break;
  }
  return h;
}
/** Two-sided p-value of a t statistic on df degrees of freedom: I_{df/(df+t²)}(df/2, 1/2). */
function pT2(t, df) {
  const x = df / (df + t * t), a = df / 2, b = 0.5;
  if (x <= 0) return 0; if (x >= 1) return 1;
  const lbt = lgamma(a + b) - lgamma(a) - lgamma(b) + a * Math.log(x) + b * Math.log(1 - x);
  return x < (a + 1) / (a + b + 2) ? Math.exp(lbt) * betacf(a, b, x) / a : 1 - Math.exp(lbt) * betacf(b, a, 1 - x) / b;
}
function digamma(x) {
  let r = 0; while (x < 6) { r -= 1 / x; x += 1; }
  const f = 1 / (x * x);
  return r + Math.log(x) - 0.5 / x - f * (1 / 12 - f * (1 / 120 - f * (1 / 252 - f / 240)));
}
function trigamma(x) {
  let r = 0; while (x < 6) { r += 1 / (x * x); x += 1; }
  const f = 1 / (x * x);
  return r + 1 / x + f / 2 + (1 / (x * x * x)) * (1 / 6 - f * (1 / 30 - f * (1 / 42 - f / 30)));
}
function tetragamma(x) {
  let r = 0; while (x < 6) { r -= 2 / (x * x * x); x += 1; }
  const f = 1 / (x * x);
  return r - f - 1 / (x * x * x) - f * f * (0.5 - f * (1 / 6 - f * (1 / 6 - f * 3 / 10)));
}
/** Inverse of the trigamma function by Newton's method (as in limma's trigammaInverse). */
function trigammaInv(x) {
  if (x > 1e7) return 1 / Math.sqrt(x);
  if (x < 1e-6) return 1 / x;
  let yy = 0.5 + 1 / x;
  for (let i = 0; i < 50; i++) { const tri = trigamma(yy), dif = tri * (1 - tri / x) / tetragamma(yy); yy += dif; if (Math.abs(dif / yy) < 1e-8) break; }
  return yy;
}
