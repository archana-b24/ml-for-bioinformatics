// Bulk step 4: differential expression between responders and non-responders. The case that DE is statistics.
// Driven by data/omics/bulk_de.json.
import { header, figure, slider, select, stats, section, steps, callout, math, legend, seeAlso, tip, untip, el } from '../lib/ui.js';
import { frame, css, ink, ink2, muted } from '../lib/plot.js';
import { stepCard, omicsData } from './_shared.js';
import { GROUP_ORDER, groupColor, groupLabel, f2 } from './_bulk_common.js';
const d3 = window.d3;

const MSHORT = { welch: 'Welch t-test (no centre term)', lm: 'linear model + centre', limma: 'moderated t (limma eBayes) + centre', perm: 'linear model, permuted labels' };

// Inverse standard normal CDF (Acklam's rational approximation), used to back out an approximate SE from p.
function qnorm(p) {
  const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239];
  const b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
  const c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const dd = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
  const pl = 0.02425;
  if (p < pl) { const q = Math.sqrt(-2 * Math.log(p)); return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((dd[0] * q + dd[1]) * q + dd[2]) * q + dd[3]) * q + 1); }
  if (p > 1 - pl) return -qnorm(1 - p);
  const q = p - 0.5, r = q * q;
  return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}

export default async function render(main) {
  header(main, 'Differential expression',
    'Which genes differ between responders and non-responders? This is a question about <b>the population</b>, answered gene by gene with a <b>p-value and a false discovery rate</b>. It is statistics, not machine learning, and treating it as either one alone causes the classic mistakes.');
  stepCard(main, {
    stage: 'Bulk RNA-seq · step 4 of 11',
    question: 'Which genes are expressed at different levels in tumours that respond to anti-PD-1, after accounting for the sequencing centre?',
    data: 'log-CPM of every gene for the patients who passed QC; response label; centre as a covariate. Uncorrected data: the centre goes into the model instead.',
    method: 'One linear model per gene (expression ~ response + centre), limma-style empirical-Bayes variance moderation, Benjamini–Hochberg FDR.',
    verdict: 'stats', why: ' The output is an inference with error control (is this gene\'s difference real, and how big?), not a prediction for a new patient. Nothing is tuned on held-out data, and success is measured by calibrated error rates, not accuracy.',
  });
  const D = await omicsData(main, 'bulk_de.json');
  if (!D) return;
  const G = D.genes.length, M = D.methods;
  const groupN = Object.fromEntries(GROUP_ORDER.map((g) => [g, D.group.filter((x) => x === g).length]));
  const hitsAt = (m, fdr) => M[m].q.map((q) => q < fdr);
  const byGroup = (h) => Object.fromEntries(GROUP_ORDER.map((g) => [g, h.filter((v, i) => v && D.group[i] === g).length]));
  const nHits = (m, fdr) => hitsAt(m, fdr).filter(Boolean).length;

  // ------------------------------------------------------------------ figure 1: volcano + hits by group
  const fig = figure(main, 'Volcano plot: effect size against evidence',
    `Each dot is one of ${G.toLocaleString()} genes. x = log2 fold change (responders vs non-responders), y = −log10 p-value. Genes passing the FDR threshold are filled; the rest are faded. Colour is the planted structure each gene belongs to. Hover any gene.`);
  let meth = 'limma', fdr = 0.05;
  select(fig.controls, { label: 'Method', value: meth, options: Object.entries(MSHORT), onChange: (v) => { meth = v; draw(); } });
  slider(fig.controls, { label: 'FDR threshold', min: -3, max: -0.7, step: 0.05, value: Math.log10(0.05), fmt: (v) => (10 ** v).toPrecision(2), onInput: (v) => { fdr = 10 ** v; draw(); } });
  legend(fig.controls, GROUP_ORDER.map((g) => [groupLabel(g), groupColor(g)]));
  const rd = stats(fig.root);
  const row = el('div', { class: 'fig-row' }); fig.body.append(row);
  const Vw = el('div'), Bw = el('div'); row.append(Vw, Bw);
  Bw.style.flex = '0 1 360px';
  const allLfc = Object.values(M).flatMap((m) => m.lfc), allP = Object.values(M).flatMap((m) => m.p);
  const xm = d3.max(allLfc, Math.abs);
  const FV = frame(Vw, { width: 560, height: 400, x: [-xm, xm], y: [0, -Math.log10(d3.min(allP)) * 1.05], xLabel: 'log2 fold change (R vs NR)', yLabel: '−log10 p' });
  const thrLine = FV.layer('thr').append('line').attr('x1', 0).attr('x2', FV.w).attr('stroke', ink2()).attr('stroke-dasharray', '5 4');
  const thrTxt = FV.layer('thr').append('text').attr('x', FV.w - 4).attr('text-anchor', 'end').attr('font-size', 11).attr('fill', ink2());
  const order = d3.range(G).sort((a, b) => (D.group[a] === 'filler') - (D.group[b] === 'filler')).reverse();
  const dots = FV.layer('pts').selectAll('circle').data(order).join('circle').attr('r', (i) => (D.group[i] === 'filler' ? 3 : 4.5))
    .attr('stroke', css('--surface')).attr('stroke-width', 0.8)
    .on('mousemove', (e, i) => tip(e, `<b>${D.genes[i]}</b> · ${groupLabel(D.group[i])}<br>log2FC ${M[meth].lfc[i].toFixed(2)}<br>p = ${M[meth].p[i].toExponential(2)}<br>q (FDR) = ${M[meth].q[i].toExponential(2)}`))
    .on('mouseleave', untip);
  const labs = FV.layer('labs');
  const FB = frame(Bw, { width: 360, height: 400, x: [0, 1], y: [0, 1], xLabel: 'share of group called significant', margin: { left: 118, right: 44 }, xScale: d3.scaleLinear(), yScale: d3.scaleBand().padding(0.25), grid: false });
  FB.y.domain(GROUP_ORDER.slice().reverse()); FB.g.select('.axis.y').call(d3.axisLeft(FB.y).tickFormat(groupLabel));
  FB.g.select('.axis.x').call(d3.axisBottom(FB.x).ticks(4).tickFormat(d3.format('.0%')));

  function draw() {
    const m = M[meth], h = hitsAt(meth, fdr);
    dots.attr('cx', (i) => FV.x(m.lfc[i])).attr('cy', (i) => FV.y(-Math.log10(m.p[i])))
      .attr('fill', (i) => groupColor(D.group[i])).attr('opacity', (i) => (h[i] ? 1 : 0.18));
    // p threshold equivalent to the FDR cut: the largest p among hits
    const pmax = d3.max(m.p.filter((p, i) => h[i]));
    if (pmax) { thrLine.attr('y1', FV.y(-Math.log10(pmax))).attr('y2', FV.y(-Math.log10(pmax))).attr('opacity', 1); thrTxt.attr('y', FV.y(-Math.log10(pmax)) - 5).text(`FDR ${fdr.toPrecision(2)} ⇔ p < ${pmax.toExponential(1)}`).attr('opacity', 1); }
    else { thrLine.attr('opacity', 0); thrTxt.attr('opacity', 0); }
    // label the top genes in a column beside the cloud, with leader lines, so names never overlap
    const top = d3.range(G).sort((a, b) => m.p[a] - m.p[b]).slice(0, 5).filter((i) => h[i]);
    const ly = top.map((i, k) => FV.y(-Math.log10(m.p[top[0]])) + 4 + k * 14);
    // labels sit on the inner side of the cloud (towards x = 0), right-aligned for up-genes, left-aligned for down-genes
    const pos = (i) => m.lfc[i] >= 0;
    const inner = top.length ? d3.min(top.filter(pos), (i) => m.lfc[i]) : 0, innerN = d3.max(top.filter((i) => !pos(i)), (i) => m.lfc[i]);
    const lx = (i) => (pos(i) ? FV.x(inner) - 40 : FV.x(innerN) + 40);
    labs.selectAll('line').data(top).join('line').attr('x1', (i) => FV.x(m.lfc[i])).attr('y1', (i) => FV.y(-Math.log10(m.p[i])))
      .attr('x2', (i) => lx(i) + (pos(i) ? 3 : -3)).attr('y2', (i, k) => ly[k] - 4).attr('stroke', muted()).attr('stroke-width', 0.8);
    labs.selectAll('text').data(top).join('text').attr('x', lx).attr('y', (i, k) => ly[k]).attr('text-anchor', (i) => (pos(i) ? 'end' : 'start'))
      .attr('font-size', 11).attr('fill', ink()).text((i) => D.genes[i]);
    const bg = byGroup(h), nh = h.filter(Boolean).length;
    FB.layer('bars').selectAll('rect').data(GROUP_ORDER).join('rect').attr('y', (g) => FB.y(g)).attr('height', FB.y.bandwidth()).attr('x', 0)
      .attr('width', (g) => FB.x(bg[g] / groupN[g])).attr('rx', 3).attr('fill', (g) => groupColor(g))
      .on('mousemove', (e, g) => tip(e, `<b>${groupLabel(g)}</b><br>${bg[g]} of ${groupN[g]} genes significant`)).on('mouseleave', untip);
    FB.layer('lab').selectAll('text').data(GROUP_ORDER).join('text').attr('y', (g) => FB.y(g) + FB.y.bandwidth() / 2 + 4)
      .attr('x', (g) => FB.x(bg[g] / groupN[g]) + 5).attr('font-size', 11).attr('fill', ink2()).text((g) => `${bg[g]}/${groupN[g]}`);
    rd.set({ 'genes called': nh, 'expected false among them': (fdr * nh).toFixed(1), 'IFN-γ module': `${bg['IFNG module']} / ${groupN['IFNG module']}`, 'proliferation module': `${bg['PROLIF module']} / ${groupN['PROLIF module']}`, 'filler genes': `${bg.filler} / ${groupN.filler}` });
  }
  draw();

  // ------------------------------------------------------------------ figure 2: p-value histograms
  const fig2 = figure(main, 'The p-value histogram: the first plot to make',
    'Distribution of all p-values for a method. If no gene were truly different, p-values would be uniform (dashed line). A spike near 0 on top of a flat floor is the healthy shape. Hover a bar.');
  let m2 = 'limma';
  select(fig2.controls, { label: 'Method', value: m2, options: Object.entries(MSHORT), onChange: (v) => { m2 = v; draw2(); } });
  const rd2 = stats(fig2.root);
  const bins = 20;
  const FH = frame(fig2.body, { width: 620, height: 260, x: [0, 1], y: [0, 1], xLabel: 'p-value', yLabel: 'genes' });
  function draw2() {
    const p = M[m2].p, cnt = new Array(bins).fill(0);
    p.forEach((v) => { cnt[Math.min(bins - 1, Math.floor(v * bins))]++; });
    FH.y.domain([0, d3.max(cnt) * 1.05]); FH.g.select('.axis.y').call(d3.axisLeft(FH.y).ticks(5));
    FH.layer('bars').selectAll('rect').data(cnt).join('rect').attr('x', (c, i) => FH.x(i / bins) + 1).attr('width', FH.x(1 / bins) - 2)
      .attr('y', (c) => FH.y(c)).attr('height', (c) => FH.h - FH.y(c)).attr('rx', 2).attr('fill', css('--accent'))
      .on('mousemove', (e, c) => { const i = cnt.indexOf(c); tip(e, `${c} genes with p in [${(i / bins).toFixed(2)}, ${((i + 1) / bins).toFixed(2)})`); }).on('mouseleave', untip);
    const u = G / bins;
    FH.layer('u').selectAll('line').data([u]).join('line').attr('x1', 0).attr('x2', FH.w).attr('y1', FH.y(u)).attr('y2', FH.y(u)).attr('stroke', ink()).attr('stroke-dasharray', '5 4').attr('stroke-width', 1.5);
    const pi0 = Math.min(1, p.filter((v) => v > 0.5).length / (0.5 * G));
    rd2.set({ 'genes with p < 0.05': p.filter((v) => v < 0.05).length, 'expected by chance': Math.round(0.05 * G), 'estimated share of null genes (π₀)': pi0.toFixed(2), 'called at FDR 5%': M[m2].n_hits });
  }
  draw2();

  // ------------------------------------------------------------------ figure 3: one gene across methods
  const fig3 = figure(main, 'One gene, four analyses',
    'Estimated log2 fold change with an approximate 95% confidence interval (back-calculated from the p-value), for the same gene under each method. Pick genes from different groups.');
  const pick = ['CXCL9', 'IDO1', 'GZMB', 'POSTN', 'FN1', 'MKI67', 'TOP2A', 'EPCAM', 'CD8A', 'GAPDH']
    .filter((g) => D.genes.includes(g));
  // add the filler gene the Welch test likes most but the centre-adjusted model rejects
  const fillerTrap = d3.range(G).filter((i) => D.group[i] === 'filler' && M.lm.q[i] > 0.5).sort((a, b) => M.welch.p[a] - M.welch.p[b])[0];
  if (fillerTrap != null) pick.push(D.genes[fillerTrap]);
  let gene = 'CXCL9';
  select(fig3.controls, { label: 'Gene', value: gene, options: pick.map((g) => [g, `${g} (${groupLabel(D.group[D.genes.indexOf(g)])})`]), onChange: (v) => { gene = v; draw3(); } });
  const rd3 = stats(fig3.root);
  const keys = Object.keys(MSHORT);
  const FF = frame(fig3.body, { width: 620, height: 230, x: [-1.8, 1.8], y: [0, 1], xLabel: 'log2 fold change (R vs NR) with ~95% CI', margin: { left: 230 }, yScale: d3.scaleBand().padding(0.4), grid: false });
  FF.y.domain(keys.slice().reverse()); FF.g.select('.axis.y').call(d3.axisLeft(FF.y).tickFormat((k) => MSHORT[k]));
  FF.layer('zero').append('line').attr('x1', FF.x(0)).attr('x2', FF.x(0)).attr('y1', 0).attr('y2', FF.h).attr('stroke', css('--axis'));
  function draw3() {
    const i = D.genes.indexOf(gene);
    const rows = keys.map((k) => {
      const l = M[k].lfc[i], p = Math.max(1e-300, M[k].p[i]), z = Math.abs(qnorm(p / 2));
      const se = z > 1e-3 ? Math.abs(l) / z : 1;
      return { k, l, lo: l - 1.96 * se, hi: l + 1.96 * se, p, q: M[k].q[i] };
    });
    const cy = (k) => FF.y(k) + FF.y.bandwidth() / 2;
    FF.layer('ci').selectAll('line').data(rows).join('line').attr('y1', (d) => cy(d.k)).attr('y2', (d) => cy(d.k))
      .attr('x1', (d) => FF.x(Math.max(-1.8, d.lo))).attr('x2', (d) => FF.x(Math.min(1.8, d.hi))).attr('stroke', ink2()).attr('stroke-width', 2);
    FF.layer('pt').selectAll('circle').data(rows).join('circle').attr('cy', (d) => cy(d.k)).attr('cx', (d) => FF.x(d.l)).attr('r', 6)
      .attr('fill', (d) => (d.q < 0.05 ? groupColor(D.group[i]) : css('--surface'))).attr('stroke', (d) => (d.q < 0.05 ? css('--surface') : groupColor(D.group[i]))).attr('stroke-width', 2)
      .on('mousemove', (e, d) => tip(e, `<b>${gene}</b>, ${MSHORT[d.k]}<br>log2FC ${d.l.toFixed(2)} [${d.lo.toFixed(2)}, ${d.hi.toFixed(2)}]<br>p = ${d.p.toExponential(2)}, q = ${d.q.toExponential(2)}`)).on('mouseleave', untip);
    rd3.set(Object.fromEntries(rows.map((r) => [`q, ${MSHORT[r.k]}`, r.q < 0.001 ? r.q.toExponential(1) : r.q.toFixed(3)])));
  }
  draw3();
  fig3.caption('Filled dot = significant at FDR 5%; open = not. Estimated log2 fold change with an approximate 95% confidence interval (back-calculated from the p-value), for the same gene under each method.' + (fillerTrap != null ? ` <b>${D.genes[fillerTrap]}</b> is the anonymous filler gene the Welch test ranks highest among those the centre-adjusted model finds nothing for: a pure batch artefact.` : ''));

  // ------------------------------------------------------------------ prose
  const bgL = M.limma.by_group, bgW = M.welch.by_group;
  const nRes = D.n - 3; // residual df of ~ 1 + response + centre
  section(main, 'What to notice', `<ul>
    <li><b>The Welch t-test with no centre term calls ${M.welch.n_hits} genes</b> at FDR 5%, including ${bgW.filler[0]} of ${bgW.filler[1]} anonymous filler genes. Responders are concentrated in Centre B, so every gene with a centre shift looks "associated with response". Its p-value histogram is a wall of tiny p-values. The test is not wrong; it answers the wrong question.</li>
    <li><b>Adding centre to the model</b> drops the count to ${M.lm.n_hits}. All ${bgL['IFNG module'][1]} IFN-γ module genes stay at the top, which is the planted biology.</li>
    <li><b>Permute the labels</b>: ${M.perm.n_hits} genes, and a flat p-value histogram. This is the check that the whole pipeline controls false positives.</li>
    <li><b>Moderation barely matters here</b> (${M.lm.n_hits} vs ${M.limma.n_hits} genes). The prior is worth d₀ = ${f2(D.limma_d0)} extra degrees of freedom against ${nRes} from the data, about ${(100 * D.limma_d0 / (D.limma_d0 + nRes)).toFixed(0)}% of the weight. With 3 vs 3 samples it would dominate and rescue the analysis.</li>
    <li>Pick <b>MKI67</b> in the last figure: one proliferation gene is called "lower in responders". Proliferation is not predictive in the simulation. Bulk expression is a mixture: inflamed tumours contain more immune cells and so fewer tumour cells, which dilutes every tumour-cell gene (EPCAM too). DE finds <i>bulk</i> differences, not regulation; <a href="#/omics/bulk-deconvolution">deconvolution</a> and <a href="#/omics/sc-pseudobulk">pseudobulk</a> separate the two.</li>
  </ul>`);

  section(main, 'Why this is statistics, not ML', `
    <table class="simple">
      <tr><th></th><th>Differential expression (statistics)</th><th>Response classifier (ML)</th></tr>
      <tr><td>Question</td><td>Is gene g different in the population, and by how much?</td><td>For a new patient, will they respond?</td></tr>
      <tr><td>Output</td><td>An effect size, a p-value and an FDR for every gene</td><td>A score for each new sample</td></tr>
      <tr><td>Model</td><td>Fixed in advance by the design: <code>~ response + centre</code>. One small model per gene</td><td>Chosen and tuned by data, many genes at once</td></tr>
      <tr><td>Judged by</td><td>Calibrated error rates (permutation gives 0 calls)</td><td>Accuracy on data the model never saw (AUC)</td></tr>
      <tr><td>Uses confounders</td><td>Must adjust for them, or the answer is wrong</td><td>Will happily exploit them, and look good until deployment</td></tr>
    </table>
    <p>The two are not interchangeable. A list of ${M.limma.n_hits} significant genes does not mean they predict response well (the overlap between groups can be large even when means differ), and a good predictor can rely on genes with no significant difference at all. Feature importance from a classifier is not a p-value. Use DE to learn biology, and the ML steps that follow to build and honestly test a predictor.</p>`);

  steps(main, 'How it works', [
    'Normalise counts to log-CPM (or use counts directly in a negative binomial model, as DESeq2 and edgeR do).',
    'Write the design matrix: intercept, response (0/1), centre (0/1). Each gene gets the same design.',
    'Fit ordinary least squares per gene; the response coefficient is the log2 fold change, adjusted for centre.',
    '<b>Moderate the variances</b>: estimate a prior distribution of gene variances from all genes, and shrink each gene\'s variance towards it. This stabilises genes whose variance is unluckily small, which would otherwise give huge t-statistics.',
    'Compute a moderated t-statistic and p-value per gene.',
    'Adjust for testing thousands of genes with <b>Benjamini–Hochberg</b>: sort p-values, compare the i-th smallest to (i/m)·FDR. The resulting q-value is the FDR you would accept to call that gene.',
    'Sanity checks: p-value histogram, permuted labels, and known biology at the top of the list.',
  ]);
  math(main, 'y_g = β0 + β1·response + β2·centre + ε          log2FC = β̂1\ns̃²_g = (d0·s0² + d·s²_g) / (d0 + d)          t̃_g = β̂1 / (s̃_g · √v11)\nBH: q_(i) = min_{j≥i}  p_(j) · m / j');

  callout(main, '', 'In biology', '<b>limma-voom</b>, <b>edgeR</b> (quasi-likelihood F-test) and <b>DESeq2</b> are the standard tools for bulk RNA-seq DE, all with the design formula <code>~ centre + response</code>. DESeq2 and edgeR model counts with a negative binomial and shrink dispersions; DESeq2\'s <code>lfcShrink</code> (apeglm) also shrinks fold changes for ranking. The ranked list then goes into pathway analysis (<b>fgsea</b>, GSEA, camera). For single-cell data the same tools apply only after summing cells per patient: see <a href="#/omics/sc-pseudobulk">pseudobulk DE</a>.');
  callout(main, 'warn', 'Pitfalls', `<ul>
    <li><b>Forgetting the batch</b> (above): ${M.welch.n_hits} "hits", mostly technical.</li>
    <li><b>Ranking by p-value alone.</b> With large n, tiny fold changes become significant. Report effect sizes and use a fold-change-aware test (limma <code>treat</code>) if only big changes matter.</li>
    <li><b>Selecting genes by DE on all samples, then training and cross-validating a classifier on them.</b> The selection has seen the test folds. This is the leakage trap on the <a href="#/omics/bulk-classifier">classifier page</a>.</li>
    <li><b>Composition masquerading as regulation.</b> A gene can be "up in responders" because there are more of the cells that express it, not because any cell changed its expression.</li>
    <li><b>Not significant ≠ no effect.</b> The CAF/TGF-β module separates excluded tumours, yet ${bgL['CAF/TGF-b module'][0]} of its ${bgL['CAF/TGF-b module'][1]} genes are called for response once centre is in the model. Its biology is about <i>where</i> T cells are, which needs <a href="#/omics/sp-biomarker">spatial data</a>.</li>
  </ul>`);
  seeAlso(main, [['omics', 'bulk-subtypes', 'Previous: subtypes'], ['omics', 'bulk-feature-selection', 'Next: feature selection'], ['concepts', 'multiple-testing', 'Multiple testing & FDR'], ['omics', 'sc-pseudobulk', 'Pseudobulk DE'], ['concepts', 'linear-regression', 'Linear regression']]);
}
