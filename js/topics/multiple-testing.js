// Multiple testing and FDR, lesson format.
// Arc: what → why → where → how it works (6 steps: null p-values → counts vs threshold → Bonferroni/FWER → Benjamini–Hochberg step-up
//      → π0 and q-values → spike-in lab with known truth) → pitfalls → recap.
// Data: the bulk gene-level matrix of the same study (data/omics/bulk_de_lab.json, 1,933 genes × 177 patients). Every p-value on the page is
// a Welch t-test computed in the browser; "shuffled" data permute the response labels, so every gene is truly null (while genes stay correlated).
import { header, figure, slider, select, button, checkbox, stats, legend, seeAlso, tip, untip, el, html, loadJSON } from '../lib/ui.js';
import { frame, cat, css, ink, ink2, muted } from '../lib/plot.js';
import { rng, mean } from '../lib/data.js';
import { loadCohort, studyCard } from '../lib/study.js';
import { lessonNav, chapter, prose, rung, worked, quiz, cards, table, code, recap } from '../lib/lesson.js';
const d3 = window.d3;

const f2 = (v) => v.toFixed(2);
const f1 = (v) => v.toFixed(1);
const pc = (v) => (v < 0.095 ? (100 * v).toFixed(1) : Math.round(100 * v)) + '%';
const nf = (v) => Math.round(v).toLocaleString('en-GB');
const SUP = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
const sup = (k) => String(k).split('').map((c) => SUP[c]).join('');
const sci = (p) => { if (p >= 0.001) return p.toFixed(3); const [m, e] = p.toExponential(1).split('e'); return `${m}×10${sup(+e)}`; };
const sum = (a) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i]; return s; };
const median = (a) => d3.median(Array.from(a));

// ---------------------------------------------------------------------------------------------
// Statistics core: Welch t-test per gene, Student-t tail via the regularised incomplete beta function.
function lgam(z) {
  const c = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
  let y = z, x = z, t = x + 5.5; t -= (x + 0.5) * Math.log(t);
  let s = 1.000000000190015; for (let j = 0; j < 6; j++) s += c[j] / ++y;
  return -t + Math.log(2.5066282746310005 * s / x);
}
function betacf(a, b, x) {
  const TINY = 1e-300; let c = 1, d = 1 - (a + b) * x / (a + 1); if (Math.abs(d) < TINY) d = TINY; d = 1 / d; let h = d;
  for (let m = 1; m <= 300; m++) {
    const m2 = 2 * m; let aa = m * (b - m) * x / ((a - 1 + m2) * (a + m2));
    d = 1 + aa * d; if (Math.abs(d) < TINY) d = TINY; c = 1 + aa / c; if (Math.abs(c) < TINY) c = TINY; d = 1 / d; h *= d * c;
    aa = -(a + m) * (a + b + m) * x / ((a + m2) * (a + 1 + m2));
    d = 1 + aa * d; if (Math.abs(d) < TINY) d = TINY; c = 1 + aa / c; if (Math.abs(c) < TINY) c = TINY; d = 1 / d;
    const del = d * c; h *= del; if (Math.abs(del - 1) < 1e-13) break;
  }
  return h;
}
function ibeta(x, a, b) {
  if (x <= 0) return 0; if (x >= 1) return 1;
  const bt = Math.exp(lgam(a + b) - lgam(a) - lgam(b) + a * Math.log(x) + b * Math.log(1 - x));
  return x < (a + 1) / (a + b + 2) ? bt * betacf(a, b, x) / a : 1 - bt * betacf(b, a, 1 - x) / b;
}
const pT = (t, df) => Math.min(1, Math.max(1e-300, ibeta(df / (df + t * t), df / 2, 0.5)));

/** Benjamini–Hochberg adjusted p-values: q_(k) = min over j ≥ k of m·p_(j)/j. Also returns the ordering. */
function bhQ(p, scale = 1) {
  const m = p.length, o = d3.range(m).sort((a, b) => p[a] - p[b]), q = new Float64Array(m);
  let run = 1; for (let r = m - 1; r >= 0; r--) { run = Math.min(run, scale * p[o[r]] * m / (r + 1)); q[o[r]] = run; }
  return { q, o };
}
const METHODS = [['none', 'No correction (p ≤ α)'], ['bonf', 'Bonferroni'], ['holm', 'Holm'], ['bh', 'Benjamini–Hochberg']];
function callsOf(p, method, alpha, bh) {
  const m = p.length, f = new Uint8Array(m);
  if (method === 'none') for (let i = 0; i < m; i++) f[i] = p[i] <= alpha ? 1 : 0;
  else if (method === 'bonf') for (let i = 0; i < m; i++) f[i] = p[i] <= alpha / m ? 1 : 0;
  else if (method === 'holm') { let k = 0; while (k < m && p[bh.o[k]] <= alpha / (m - k)) { f[bh.o[k]] = 1; k++; } }
  else for (let i = 0; i < m; i++) f[i] = bh.q[i] <= alpha ? 1 : 0;
  return f;
}
const thrFn = (method, alpha, m) => ({ none: () => alpha, bonf: () => alpha / m, holm: (k) => alpha / (m - k + 1), bh: (k) => alpha * k / m })[method];
const hist20 = (p) => { const c = new Array(20).fill(0); for (let i = 0; i < p.length; i++) c[Math.min(19, Math.floor(p[i] * 20))]++; return c; };

export default async function render(main) {
  const [C, LD] = await Promise.all([loadCohort(), loadJSON('data/omics/bulk_de_lab.json')]);
  const G = LD.genes.length, N = LD.y.length, Y = Uint8Array.from(LD.y), nR = sum(Y), genes = LD.genes, grp = LD.group;
  const raw = Uint8Array.from(atob(LD.x_b64), (c) => c.charCodeAt(0)), u16 = new Uint16Array(raw.buffer);
  const Xc = new Float32Array(u16.length);
  for (let g = 0; g < G; g++) { let s = 0; for (let i = 0; i < N; i++) s += u16[i * G + g]; const mu = s / N; for (let i = 0; i < N; i++) Xc[i * G + g] = u16[i * G + g] - mu; }
  const SC = 1 / LD.x_scale;

  /** Welch t-test for every gene, for a given 0/1 labelling of the N patients. diff is in log2 units. */
  function gstats(lab) {
    const n1 = sum(lab), n0 = N - n1, s1 = new Float64Array(G), q1 = new Float64Array(G), s0 = new Float64Array(G), q0 = new Float64Array(G);
    for (let i = 0; i < N; i++) { const S = lab[i] ? s1 : s0, Q = lab[i] ? q1 : q0, off = i * G; for (let g = 0; g < G; g++) { const v = Xc[off + g]; S[g] += v; Q[g] += v * v; } }
    const diff = new Float64Array(G), se = new Float64Array(G), df = new Float64Array(G), sp = new Float64Array(G), p = new Float64Array(G);
    for (let g = 0; g < G; g++) {
      const m1 = s1[g] / n1, m0 = s0[g] / n0, v1 = (q1[g] - n1 * m1 * m1) / (n1 - 1), v0 = (q0[g] - n0 * m0 * m0) / (n0 - 1), a = v1 / n1, b = v0 / n0;
      diff[g] = (m1 - m0) * SC; se[g] = Math.sqrt(a + b) * SC; df[g] = (a + b) ** 2 / (a * a / (n1 - 1) + b * b / (n0 - 1)); sp[g] = Math.sqrt(((n1 - 1) * v1 + (n0 - 1) * v0) / (N - 2)) * SC;
      p[g] = se[g] > 0 ? pT(diff[g] / se[g], df[g]) : 1;
    }
    return { diff, se, df, sp, p };
  }
  const permCache = new Map();
  const permStats = (seed) => { if (!permCache.has(seed)) permCache.set(seed, gstats(rng(seed).shuffle(Array.from(Y)))); return permCache.get(seed); };
  const REAL = gstats(Y), pR = REAL.p, bhR = bhQ(pR), NP = 20, PERMS = d3.range(NP).map((k) => permStats(100 + k).p);
  const sortedR = Array.from(pR).sort((a, b) => a - b);
  const cntBelow = (sorted, t) => d3.bisectRight(sorted, t);
  const sortedPerms = PERMS.map((p) => Array.from(p).sort((a, b) => a - b));
  const nRaw = cntBelow(sortedR, 0.05), nBonf = cntBelow(sortedR, 0.05 / G), nBH = sum(callsOf(pR, 'bh', 0.05, bhR)), expRaw = G * 0.05;
  const permRaw = sortedPerms.map((s) => cntBelow(s, 0.05));
  const ifngIdx = d3.range(G).filter((i) => grp[i] === 'IFNG module'), ifngBH = ifngIdx.filter((i) => bhR.q[i] <= 0.05).length;
  const centreStats = gstats(Uint8Array.from(LD.centreB)), nCentreBH = sum(callsOf(centreStats.p, 'bh', 0.05, bhQ(centreStats.p)));
  const respA = mean(LD.y.filter((v, i) => !LD.centreB[i])), respB = mean(LD.y.filter((v, i) => LD.centreB[i]));

  // ---- spike-in experiments: shuffled (all-null) real data + planted effects of known size ----
  const spikeOrder = (seed) => rng(seed + 777).shuffle(d3.range(G));
  function labSim(seed, frac, dEff) {
    const base = permStats(seed), p = Float64Array.from(base.p), K = Math.round(frac * G), ord = spikeOrder(seed), sp = new Uint8Array(G);
    for (let j = 0; j < K; j++) { const g = ord[j]; sp[g] = 1; const sgn = j % 2 ? 1 : -1; p[g] = pT((base.diff[g] + sgn * dEff * base.sp[g]) / base.se[g], base.df[g]); }
    return { p, sp, K, bh: bhQ(p), seed };
  }
  function outcome(sim, method, alpha) {
    const f = callsOf(sim.p, method, alpha, sim.bh); let tp = 0, fp = 0;
    for (let g = 0; g < G; g++) if (f[g]) { if (sim.sp[g]) tp++; else fp++; }
    return { f, tp, fp, n: tp + fp, fdp: tp + fp ? fp / (tp + fp) : 0 };
  }
  const SEEDS = d3.range(1, 13);
  function labMean(frac, dEff, alpha, method) {
    const r = SEEDS.map((s) => outcome(labSim(s, frac, dEff), method, alpha));
    return { tp: mean(r.map((o) => o.tp)), fp: mean(r.map((o) => o.fp)), fdp: mean(r.map((o) => o.fdp)), K: Math.round(frac * G), anyFP: mean(r.map((o) => (o.fp > 0 ? 1 : 0))), fdpMax: d3.max(r, (o) => o.fdp), fdpMin: d3.min(r, (o) => o.fdp) };
  }

  header(main, 'Multiple testing and false discovery rate',
    'Testing thousands of genes at once <b>makes false positives very likely</b> unless the threshold is adjusted. Bonferroni limits the chance of <i>any</i> false positive; Benjamini–Hochberg limits the <i>proportion</i> of false positives among the genes reported.');
  lessonNav(main);
  studyCard(main, C, '<b>Question for this page:</b> among 1,933 genes measured in the discovery patients, which are associated with response, and how many of the genes reported as significant can be expected to be false?');

  // =====================================================================================
  // 1. WHAT
  // =====================================================================================
  let ch = chapter(main, 'what', 'What is multiple testing?', 'In one line: <b>one test per gene → thousands of p-values → a threshold that accounts for how many tests were run</b>.');
  prose(ch, `<p>A <b>hypothesis test</b> asks whether a measurement differs between groups by more than chance would explain. It returns a <b>p-value</b>: the probability of a difference at least as large as the one observed if there were no real difference (the <b>null hypothesis</b>). By convention a result with p &lt; 0.05 is called significant, which means a null gene is wrongly called in about 1 test in 20.</p>
    <p>When one such test is run for each of ${nf(G)} genes, that 1-in-20 rate applies to every gene, so many false calls are expected. <b>Multiple-testing correction</b> replaces the single-test threshold by one that accounts for the number of tests. The diagram follows the ${nf(G)} genes measured in the ${N} discovery patients with bulk RNA (${nR} responders, ${N - nR} non-responders).</p>`);
  ch.append(pipelineSvg({ G, N, nR, nonR: N - nR, gi: genes.indexOf('IFNG'), REAL, pR, hist: hist20(pR), nRaw, nBonf, nBH }));
  prose(ch, '<p>Key terms:</p>');
  table(ch, [
    { t: 'Null hypothesis (<i>H</i>₀)', m: 'No true difference for this gene between responders and non-responders', s: `${nf(G)} of them, one per gene` },
    { t: 'p-value', m: 'Probability of a result at least this extreme if <i>H</i>₀ is true; uniform on 0–1 when <i>H</i>₀ is true and the test is valid', s: `IFNG: p = ${sci(pR[genes.indexOf('IFNG')])}` },
    { t: 'False positive (type I error)', m: 'A gene called significant although <i>H</i>₀ is true', s: `${nf(expRaw)} expected at p &lt; 0.05 if all ${nf(G)} were null` },
    { t: 'Family-wise error rate (FWER)', m: 'Probability of at least one false positive among all tests', s: 'Controlled by Bonferroni and Holm' },
    { t: 'False discovery proportion (FDP) and rate (FDR)', m: 'FDP is the fraction of called genes that are false in one experiment; FDR is its expected value', s: 'Controlled by Benjamini–Hochberg (BH)' },
    { t: 'Adjusted p-value / q-value', m: 'The smallest error level at which a gene would be called (FWER for Bonferroni, FDR for BH and the q-value)', s: `Calls at BH q &lt; 0.05: ${nf(nBH)}` },
    { t: 'Power', m: 'Probability that a gene with a true effect is called', s: 'Falls as the threshold becomes stricter (chapter 4, step 6)' },
  ], [['t', 'Term'], ['m', 'Meaning'], ['s', 'In our study']]);

  // =====================================================================================
  // 2. WHY
  // =====================================================================================
  ch = chapter(main, 'why', 'Why do we need it?', 'Why a per-test threshold of 0.05 stops being meaningful as the number of tests grows.');
  prose(ch, `<p>At threshold α, each null gene passes with probability α, independently of the others when the tests are independent. With <i>m</i> null tests:</p><ul>
    <li><b>Expected false positives</b> = <i>m</i> × α. For ${nf(G)} genes and α = 0.05 this is ${f1(expRaw)}.</li>
    <li><b>Probability of at least one false positive</b> = 1 − (1 − α)<sup><i>m</i></sup>. For <i>m</i> = 20 this is already ${pc(1 - 0.95 ** 20)}; for <i>m</i> = ${nf(G)} it is indistinguishable from 100%.</li></ul>
    <p>The figure draws that situation directly: every dot is a test whose null hypothesis is true.</p>`);
  const fl = figure(ch, 'Every null hypothesis is true, and some tests still pass', 'Each dot is one test with a true null hypothesis and a p-value drawn uniformly from 0–1. Highlighted dots have p &lt; α: they would be reported as discoveries. Hover a dot for its p-value.');
  const MS = [1, 5, 10, 20, 50, 100, 200, 500, 1000, 2000];
  let mi = 5, aL = 0.05, seedL = 1;
  slider(fl.controls, { label: 'number of tests m', min: 0, max: MS.length - 1, step: 1, value: mi, fmt: (i) => nf(MS[i]), onInput: (i) => { mi = i; paintLot(); } });
  slider(fl.controls, { label: 'threshold α', min: 0.01, max: 0.1, step: 0.01, value: aL, fmt: f2, onInput: (v) => { aL = v; paintLot(); } });
  button(fl.controls, 'Draw new p-values', () => { seedL++; paintLot(); }, true);
  const sLot = stats(fl.root);
  const FL = frame(fl.body, { width: 820, height: 250, x: [0, 1], y: [0, 1], axes: false, grid: false, margin: { top: 8, right: 8, bottom: 8, left: 8 } });
  const lotL = FL.layer('dots'), lotT = FL.layer('t');
  function paintLot() {
    const m = MS[mi], r = rng(5000 + seedL), ps = d3.range(m).map(() => r());
    const cols = Math.max(1, Math.round(Math.sqrt(m * FL.w / FL.h))), rows = Math.ceil(m / cols), cell = Math.min(FL.w / cols, FL.h / rows), rad = Math.max(2, Math.min(9, cell * 0.38));
    const ox = (FL.w - cols * cell) / 2, oy = (FL.h - rows * cell) / 2, hit = ps.filter((p) => p < aL).length;
    lotL.selectAll('circle').data(ps.map((p, i) => ({ p, i }))).join('circle').attr('cx', (d) => ox + (d.i % cols + 0.5) * cell).attr('cy', (d) => oy + (Math.floor(d.i / cols) + 0.5) * cell).attr('r', (d) => (d.p < aL ? rad * 1.25 : rad))
      .attr('fill', (d) => (d.p < aL ? cat(3) : cat(0))).attr('fill-opacity', (d) => (d.p < aL ? 1 : 0.35)).attr('stroke', (d) => (d.p < aL ? css('--surface') : 'none')).attr('stroke-width', 1.2)
      .on('mousemove', (e, d) => tip(e, `test ${d.i + 1}<br>p = ${d.p.toFixed(3)}<br>${d.p < aL ? 'passes p &lt; ' + aL : 'does not pass'}`)).on('mouseleave', untip);
    lotT.selectAll('text').data([0]).join('text').attr('x', FL.w - 4).attr('y', FL.h - 4).attr('text-anchor', 'end').attr('font-size', 12.5).attr('fill', ink2()).text(`${hit} of ${nf(m)} null tests pass p < ${aL}: all ${hit} are false positives`);
    sLot.set({ 'tests (all true nulls)': nf(m), 'false positives this draw': hit, 'expected m × α': f1(m * aL), 'P(at least one)': pc(1 - (1 - aL) ** m) });
  }
  paintLot();
  prose(ch, '<p>Three strategies exist, which differ in what they control:</p>');
  table(ch, [
    { a: 'No correction (p ≤ α)', c: 'Error rate per test', r: `${nf(expRaw)} expected false calls among ${nf(G)} null genes`, u: 'A single pre-specified test' },
    { a: '<b>Bonferroni / Holm</b>', c: '<b>FWER</b>: chance of any false positive', r: 'At most α, but the per-gene cut-off becomes α / m', u: 'A few important claims; GWAS; confirmatory work' },
    { a: '<b>Benjamini–Hochberg</b>', c: '<b>FDR</b>: expected share of false calls in the list', r: 'About α of the reported list is false, on average', u: 'Exploratory screens with many true effects: DE, proteomics, enrichment' },
  ], [['a', 'Approach'], ['c', 'Controls'], ['r', 'Consequence'], ['u', 'Typical use']]);

  // =====================================================================================
  // 3. WHERE
  // =====================================================================================
  ch = chapter(main, 'where', 'Where is it used?', 'Any analysis that tests many hypotheses and reports those that pass.');
  cards(ch, [
    { tag: 'Differential expression', title: 'Which genes differ between groups?', body: 'limma, DESeq2 and edgeR report a BH-adjusted p-value (<code>padj</code> / <code>adj.P.Val</code>) for every gene. A typical report lists genes at adjusted p &lt; 0.05 together with an effect-size filter.' },
    { tag: 'Genetics (GWAS)', title: 'Which variants associate with disease?', body: 'About a million independent common-variant tests per genome motivate the conventional genome-wide threshold of 5 × 10<sup>−8</sup>, which is close to Bonferroni (0.05 / 10<sup>6</sup>).' },
    { tag: 'eQTL mapping', title: 'Which variants change expression?', body: 'Millions of variant–gene pairs. Tools such as tensorQTL combine permutations within each gene with a genome-wide FDR across genes.' },
    { tag: 'Proteomics and metabolomics', title: 'Which features change?', body: 'Thousands of peptides, proteins or metabolites are tested per contrast, and BH adjustment is the usual default.' },
    { tag: 'Enrichment analysis', title: 'Which pathways are over-represented?', body: 'Each of thousands of gene sets is tested against a gene list. Adjusted p-values (FDR) are reported by GO enrichment tools and GSEA.' },
    { tag: 'Single-cell', title: 'Which genes mark a cluster?', body: 'Seurat\'s <code>FindMarkers</code> returns a Bonferroni-adjusted p-value over all genes. Clusters are derived from the same data, so these p-values are optimistic (chapter 5).' },
    { tag: 'Peak calling and screens', title: 'Which regions or guides are hits?', body: 'ChIP-seq and ATAC-seq peak callers (MACS2) attach q-values to peaks; CRISPR screen tools such as MAGeCK report an FDR per gene.' },
    { tag: 'Feature screening for ML', title: 'Which features enter the model?', body: 'Ranking thousands of features by univariate p-values is a multiple-testing problem. It must sit inside the cross-validation loop, or the performance estimate is inflated.' },
    { tag: 'Outside biology', title: 'A/B tests and imaging', body: 'Many website variants or thousands of brain voxels tested at once raise the same issue and use the same corrections.' },
  ]);
  prose(ch, '<p><b>When not to use it:</b> a single pre-specified hypothesis needs no correction; the aim is prediction rather than inference (judge the model on held-out data instead); the hypotheses are a small set of planned contrasts (correct over that set, not over the genome); or the interest is in effect sizes with intervals rather than a yes/no call (report the estimates and shrink them, for example with empirical Bayes).</p>');

  // =====================================================================================
  // 4. HOW
  // =====================================================================================
  ch = chapter(main, 'how', 'How it works', 'Six steps, from what a p-value looks like when nothing is going on to a spike-in experiment with known truth.');

  // ---- Step 1: null p-values are uniform
  let rb = rung(ch, 1, 'Under the null, p-values are uniform',
    `<p>The tests are Welch t-tests on log2 expression, responders against non-responders. Shuffling the response labels breaks any true association, so every gene becomes a true null while the correlation between genes is kept. The p-values then spread evenly across 0–1: a p-value below 0.05 occurs for 5% of null tests, below 0.5 for half, and so on.</p>`);
  const f1g = figure(rb, 'p-value histograms: real labels and shuffled labels', 'Each bar counts genes with p in a 0.05-wide bin. Left: the real response labels. Right: the labels shuffled at random (press the button for another shuffle). The dashed line marks the count expected if every p-value were uniform. The two panels use different vertical scales.');
  let sh1 = 1;
  button(f1g.controls, 'Shuffle labels again', () => { sh1++; paint1(); }, true);
  const s1 = stats(f1g.root);
  const row1 = el('div', { class: 'fig-row' }); f1g.body.append(row1);
  const c1a = el('div'), c1b = el('div'); row1.append(c1a, c1b);
  const hR = hist20(pR), uni = G / 20;
  const mkHist = (parent, ymax, title) => {
    const F = frame(parent, { width: 500, height: 300, x: [0, 1], y: [0, ymax], xLabel: 'p-value', yLabel: 'number of genes' });
    F.g.append('line').attr('x1', 0).attr('x2', F.w).attr('y1', F.y(uni)).attr('y2', F.y(uni)).attr('stroke', ink2()).attr('stroke-dasharray', '5 4').attr('stroke-width', 1.5);
    F.g.append('text').attr('x', F.w - 4).attr('y', title.startsWith('Real') ? F.y(uni) - 6 : 30).attr('text-anchor', 'end').attr('font-size', 11.5).attr('fill', ink2()).text(`dashed line: uniform, ${f1(uni)} per bin`);
    F.g.append('text').attr('x', title.startsWith('Real') ? F.x(0.075) : 8).attr('y', 14).attr('font-size', 12.5).attr('font-weight', 700).attr('fill', ink()).text(title);
    return F;
  };
  const FA = mkHist(c1a, Math.ceil(d3.max(hR) / 100) * 100, 'Real response labels');
  const FB = mkHist(c1b, 300, 'Shuffled labels (every gene is null)');
  const drawBars = (F, h, color) => F.layer('bars').selectAll('rect').data(h.map((c, i) => ({ c, i }))).join('rect').attr('x', (d) => F.x(d.i / 20) + 1).attr('width', F.w / 20 - 2).attr('y', (d) => F.y(Math.min(d.c, F.y.domain()[1]))).attr('height', (d) => F.h - F.y(Math.min(d.c, F.y.domain()[1])))
    .attr('fill', color).attr('fill-opacity', 0.85).on('mousemove', (e, d) => tip(e, `p in [${f2(d.i / 20)}, ${f2((d.i + 1) / 20)})<br>${nf(d.c)} genes`)).on('mouseleave', untip);
  drawBars(FA, hR, cat(1));
  FA.g.append('text').attr('x', FA.x(0.07)).attr('y', FA.y(700)).attr('font-size', 11.5).attr('fill', ink2()).text(`← ${nf(hR[0])} genes with p < 0.05`);
  function paint1() {
    const p = permStats(sh1).p, h = hist20(p);
    drawBars(FB, h, cat(0));
    s1.set({ 'p < 0.05, shuffled': nf(h[0]), 'expected if uniform': f1(uni), 'smallest bar': nf(d3.min(h)), 'largest bar': nf(d3.max(h)), 'p < 0.05, real labels': nf(hR[0]) });
  }
  paint1();
  prose(rb, `<p>The real histogram has a spike at 0 plus a roughly flat remainder: the spike holds the genes with a true association and the flat part the null-like ones. The shuffled histogram has no spike but is not perfectly flat either. Across ${NP} stored shuffles the number of genes with p &lt; 0.05 ranged from <b>${nf(d3.min(permRaw))}</b> to <b>${nf(d3.max(permRaw))}</b> (median ${nf(median(permRaw))}; ${f1(uni)} expected). Each gene's p-value is uniform, but genes in a module move together, so the number of false positives varies between experiments far more than for independent tests (pitfall 4).</p>`);

  // ---- Step 2: counts vs threshold
  rb = rung(ch, 2, 'Counting what passes at each threshold',
    `<p>Let <i>R</i>(<i>t</i>) be the number of genes with p ≤ <i>t</i>. In shuffled data, <i>R</i> grows as <i>m</i> × <i>t</i>, since every gene is null. In real data, the excess of <i>R</i> over <i>m</i> × <i>t</i> is the signal. The ratio <b><i>m</i> × <i>t</i> / <i>R</i>(<i>t</i>)</b> estimates the share of the genes passing <i>t</i> that are false. This ratio is the basis of the false discovery rate.</p>`);
  const f2g = figure(rb, 'Genes passing, as the threshold is relaxed', 'Number of genes with p ≤ α. Real labels (solid), 20 independent shuffles (faint lines) and the null expectation m × α (dashed). Move the slider to read the counts at one threshold.');
  let a2 = 0.05;
  const AL = d3.range(-5, Math.log10(0.5) + 0.0001, 0.05).map((v) => 10 ** v);
  slider(f2g.controls, { label: 'threshold α', min: -5, max: Math.log10(0.5), step: 0.02, value: Math.log10(a2), fmt: (v) => (10 ** v < 0.01 ? sci(10 ** v) : (10 ** v).toFixed(3)), onInput: (v) => { a2 = 10 ** v; paint2(); } });
  const s2 = stats(f2g.root);
  const F2 = frame(f2g.body, { width: 820, height: 340, x: [1e-5, 0.5], y: [0, 2000], xScale: d3.scaleLog(), axes: false, margin: { left: 62, bottom: 46, top: 14 } });
  F2.g.append('g').attr('class', 'axis y').call(d3.axisLeft(F2.y).ticks(5).tickFormat(nf)); F2.g.append('g').attr('class', 'axis x').attr('transform', `translate(0,${F2.h})`).call(d3.axisBottom(F2.x).tickValues([1e-5, 1e-4, 1e-3, 1e-2, 1e-1]).tickFormat((v) => '10' + sup(Math.round(Math.log10(v)))));
  F2.g.append('text').attr('class', 'ax-label').attr('x', F2.w / 2).attr('y', F2.h + 38).attr('text-anchor', 'middle').text('threshold α (log scale)');
  F2.g.append('text').attr('class', 'ax-label').attr('transform', 'rotate(-90)').attr('x', -F2.h / 2).attr('y', -48).attr('text-anchor', 'middle').text('genes with p ≤ α');
  const lineCt = (sorted) => d3.line().x((t) => F2.x(t)).y((t) => F2.y(cntBelow(sorted, t)))(AL);
  sortedPerms.forEach((s) => F2.layer('perm').append('path').attr('d', lineCt(s)).attr('fill', 'none').attr('stroke', cat(0)).attr('stroke-opacity', 0.3).attr('stroke-width', 1.4));
  F2.layer('lines').append('path').attr('d', d3.line().x((t) => F2.x(t)).y((t) => F2.y(G * t))(AL)).attr('fill', 'none').attr('stroke', ink2()).attr('stroke-dasharray', '6 4').attr('stroke-width', 1.8);
  F2.layer('lines').append('path').attr('d', lineCt(sortedR)).attr('fill', 'none').attr('stroke', cat(1)).attr('stroke-width', 2.6);
  F2.g.append('text').attr('x', F2.x(2e-5)).attr('y', F2.y(1000)).attr('font-size', 12).attr('fill', ink2()).text('real labels');
  F2.g.append('text').attr('x', F2.x(2e-5)).attr('y', F2.y(1750)).attr('font-size', 12).attr('fill', ink2()).text('shuffled labels (faint lines): most runs follow the dashed line m × α,');
  F2.g.append('text').attr('x', F2.x(2e-5)).attr('y', F2.y(1750) + 16).attr('font-size', 12).attr('fill', ink2()).text('but some lie well above it, because correlated genes pass together');
  const mk2 = F2.layer('mark');
  legend(f2g.body, [['real labels', cat(1)], ['shuffled labels (20 runs)', cat(0)], ['expected m × α', ink2()]]);
  function paint2() {
    const real = cntBelow(sortedR, a2), pm = permRaw.length ? sortedPerms.map((s) => cntBelow(s, a2)) : [], ex = G * a2;
    mk2.selectAll('line').data([0]).join('line').attr('x1', F2.x(a2)).attr('x2', F2.x(a2)).attr('y1', 0).attr('y2', F2.h).attr('stroke', ink()).attr('stroke-dasharray', '3 3');
    mk2.selectAll('circle').data([[real, cat(1)], [ex, ink2()]]).join('circle').attr('cx', F2.x(a2)).attr('cy', (d) => F2.y(d[0])).attr('r', 5.5).attr('fill', (d) => d[1]).attr('stroke', css('--surface')).attr('stroke-width', 1.5);
    s2.set({ 'real: genes with p ≤ α': nf(real), 'shuffled: median of 20': nf(median(pm)), 'expected m × α': f1(ex), 'estimated false share m × α / real': real ? pc(Math.min(1, ex / real)) : '—' });
  }
  paint2();
  prose(rb, `<p>At α = 0.05, ${nf(nRaw)} genes pass in the real data and about ${f1(expRaw)} would pass by chance, so roughly ${pc(expRaw / nRaw)} of that list is expected to be false. At α = 0.05 / ${nf(G)} = ${sci(0.05 / G)} (the Bonferroni cut-off), ${nf(nBonf)} genes pass and the expected chance count is 0.05.</p>
    <p>Most of the ${nf(nRaw)} passing genes carry a small planted association in this simulated study (median absolute log2 effect in the filler genes: ${f2(median(d3.range(G).filter((i) => grp[i] === 'filler').map((i) => Math.abs(LD.truth_lfc[i]))))}), so the long list reflects planted, mostly small effects, not a failure of the test.</p>`);

  // ---- Step 3: Bonferroni / FWER
  rb = rung(ch, 3, 'Bonferroni: limiting the chance of any false positive',
    `<p><b>Bonferroni</b> calls gene <i>i</i> only if p<sub><i>i</i></sub> ≤ α / <i>m</i>. For ${nf(G)} genes and α = 0.05 the cut-off is ${sci(0.05 / G)}. With <i>m</i> independent null tests, the chance that at least one passes is 1 − (1 − α/<i>m</i>)<sup><i>m</i></sup>, just below α; whatever the dependence between tests, it is at most <i>m</i> × α/<i>m</i> = α (the union bound). <b>Holm</b>'s step-down version (compare the <i>k</i>-th smallest p to α / (<i>m</i> − <i>k</i> + 1), stop at the first failure) has the same guarantee and is never less powerful.</p>`);
  const f3g = figure(rb, 'Chance of a false positive, with and without Bonferroni', 'Simulation of 300 experiments in which all m null p-values are independent and uniform, at α = 0.05. Left: the chance that an experiment contains at least one false positive, against m; markers show the simulated value at the chosen m. Right: the number of false positives per experiment.');
  let mi3 = 8, seed3 = 1;
  slider(f3g.controls, { label: 'number of tests m', min: 0, max: MS.length - 1, step: 1, value: mi3, fmt: (i) => nf(MS[i]), onInput: (i) => { mi3 = i; paint3(); } });
  button(f3g.controls, 'Run 300 new experiments', () => { seed3++; paint3(); }, true);
  const s3 = stats(f3g.root);
  const row3 = el('div', { class: 'fig-row' }); f3g.body.append(row3);
  const c3a = el('div'), c3b = el('div'); row3.append(c3a, c3b);
  const F3a = frame(c3a, { width: 500, height: 300, x: [1, 2000], y: [0, 1], xScale: d3.scaleLog(), axes: false, margin: { left: 56, bottom: 46 } });
  F3a.g.append('g').attr('class', 'axis y').call(d3.axisLeft(F3a.y).ticks(5)); F3a.g.append('g').attr('class', 'axis x').attr('transform', `translate(0,${F3a.h})`).call(d3.axisBottom(F3a.x).tickValues([1, 10, 100, 1000]).tickFormat(nf));
  F3a.g.append('text').attr('class', 'ax-label').attr('x', F3a.w / 2).attr('y', F3a.h + 38).attr('text-anchor', 'middle').text('number of tests m (log scale)');
  F3a.g.append('text').attr('class', 'ax-label').attr('transform', 'rotate(-90)').attr('x', -F3a.h / 2).attr('y', -42).attr('text-anchor', 'middle').text('P(at least one false positive)');
  const mg = d3.range(0, 3.31, 0.05).map((v) => 10 ** v);
  F3a.layer('c').append('path').attr('d', d3.line().x((m) => F3a.x(m)).y((m) => F3a.y(1 - 0.95 ** m))(mg)).attr('fill', 'none').attr('stroke', cat(3)).attr('stroke-width', 2.4);
  F3a.layer('c').append('path').attr('d', d3.line().x((m) => F3a.x(m)).y((m) => F3a.y(1 - (1 - 0.05 / m) ** m))(mg)).attr('fill', 'none').attr('stroke', cat(0)).attr('stroke-width', 2.4);
  F3a.g.append('text').attr('x', F3a.x(1.3)).attr('y', F3a.y(0.97)).attr('font-size', 11.5).attr('fill', ink2()).text('no correction: 1 − 0.95^m');
  F3a.g.append('text').attr('x', F3a.w - 4).attr('y', F3a.y(0.05) - 8).attr('text-anchor', 'end').attr('font-size', 11.5).attr('fill', ink2()).text('Bonferroni: never above α = 0.05');
  const mk3 = F3a.layer('mk');
  const F3b = frame(c3b, { width: 460, height: 300, x: [0, 1], y: [0, 1], axes: false, grid: false, margin: { left: 56, bottom: 46 } });
  const g3b = F3b.layer('h');
  legend(f3g.body, [['no correction', cat(3)], ['Bonferroni', cat(0)]]);
  function paint3() {
    const m = MS[mi3], R = 300, r = rng(9000 + seed3 * 31 + m), fpU = [], fpB = [];
    for (let e = 0; e < R; e++) { let u = 0, b = 0; for (let j = 0; j < m; j++) { const p = r(); if (p < 0.05) u++; if (p < 0.05 / m) b++; } fpU.push(u); fpB.push(b); }
    const anyU = fpU.filter((v) => v > 0).length / R, anyB = fpB.filter((v) => v > 0).length / R;
    mk3.selectAll('circle').data([[anyU, cat(3)], [anyB, cat(0)]]).join('circle').attr('cx', F3a.x(m)).attr('cy', (d) => F3a.y(d[0])).attr('r', 6).attr('fill', (d) => d[1]).attr('stroke', css('--surface')).attr('stroke-width', 1.5);
    const bwI = Math.ceil((Math.max(2, d3.max(fpU)) + 1) / 24), top = Math.ceil((Math.max(2, d3.max(fpU)) + 1) / bwI) * bwI - 1;   // integer-width bins
    const bins = d3.bin().domain([0, top + 1]).thresholds(d3.range(bwI, top + 1, bwI)); const bu = bins(fpU), bb = bins(fpB), ymax = Math.max(d3.max(bu, (b) => b.length), d3.max(bb, (b) => b.length)) * 1.12;
    F3b.svg.selectAll('.ax3').remove();
    const xs = d3.scaleLinear().domain([0, top + 1]).range([0, F3b.w]), ys = d3.scaleLinear().domain([0, ymax]).range([F3b.h, 0]);
    const ax = F3b.g.append('g').attr('class', 'ax3');
    F3b.g.insert('g', ':first-child').attr('class', 'gridline ax3').call(d3.axisLeft(ys).ticks(5).tickSize(-F3b.w).tickFormat(''));   // behind the bars
    ax.append('g').attr('class', 'axis y').call(d3.axisLeft(ys).ticks(5)); ax.append('g').attr('class', 'axis x').attr('transform', `translate(0,${F3b.h})`).call(d3.axisBottom(xs).tickValues(xs.ticks(6).filter(Number.isInteger)).tickFormat(d3.format('d')));
    ax.append('text').attr('class', 'ax-label').attr('x', F3b.w / 2).attr('y', F3b.h + 38).attr('text-anchor', 'middle').text('false positives in one experiment');
    ax.append('text').attr('class', 'ax-label').attr('transform', 'rotate(-90)').attr('x', -F3b.h / 2).attr('y', -42).attr('text-anchor', 'middle').text('number of experiments (of 300)');
    const bw = Math.max(2, xs(bu[0].x1) - xs(bu[0].x0));
    const draw = (bins2, color, shift, name) => g3b.selectAll('.' + name).data(bins2).join('rect').attr('class', name).attr('x', (b) => xs(b.x0) + shift + 1).attr('width', Math.max(1, bw / 2 - 1.5)).attr('y', (b) => ys(b.length)).attr('height', (b) => F3b.h - ys(b.length)).attr('fill', color)
      .on('mousemove', (e, b) => tip(e, `${name === 'bu' ? 'no correction' : 'Bonferroni'}<br>${b.length} experiments with ${b.x0}${b.x1 - b.x0 > 1 ? '–' + (b.x1 - 1) : ''} false positives`)).on('mouseleave', untip);
    draw(bu, cat(3), 0, 'bu'); draw(bb, cat(0), bw / 2, 'bb');
    s3.set({ 'tests per experiment': nf(m), 'experiments with ≥ 1 false positive, no correction': `${Math.round(anyU * R)} of ${R}`, 'with Bonferroni': `${Math.round(anyB * R)} of ${R}`, 'mean false positives, no correction': f1(mean(fpU)), 'theory: no correction / Bonferroni': `${pc(1 - 0.95 ** m)} / ${pc(1 - (1 - 0.05 / m) ** m)}` });
  }
  paint3();
  prose(rb, `<p>Bonferroni holds the chance of any false positive at about 5% regardless of <i>m</i>. The cost is power: the cut-off tightens with <i>m</i> and only strong effects survive. On the real data it keeps <b>${nf(nBonf)}</b> genes, against ${nf(nRaw)} with no correction.</p>`);
  quiz(rb, `A study tests 20,000 genes. What per-gene cut-off does Bonferroni apply at α = 0.05, and what would a p-value of 0.0001 mean there?`, `<p>0.05 / 20,000 = ${sci(0.05 / 20000)}. A p-value of 0.0001 would not pass: it is 40 times above the cut-off. By itself it would count as extreme for one test, but among 20,000 tests about ${f1(20000 * 0.0001)} such values are expected under the null.</p>`);

  // ---- Step 4: BH step-up
  rb = rung(ch, 4, 'Benjamini–Hochberg: the step-up procedure',
    `<p>Controlling the share of false calls rather than the chance of any is less strict when many genes carry real effects. <b>Benjamini–Hochberg (BH)</b> sorts the <i>m</i> p-values, p<sub>(1)</sub> ≤ … ≤ p<sub>(<i>m</i>)</sub>, and finds the largest rank <i>k</i>* for which <b>p<sub>(<i>k</i>)</sub> ≤ α · <i>k</i> / <i>m</i></b>. It then calls all genes up to rank <i>k</i>*. The threshold rises with rank, so each additional discovery makes the next one easier to call. Under independence (and positive dependence), the expected share of false calls is at most α.</p>`);
  const f4g = figure(rb, 'The step-up plot, real data', 'Each point is a gene: rank of its p-value (1 = smallest) against the p-value, both on log scales. The slanted line is the BH threshold α·k/m. Genes up to the last rank under the line (k*) are called. Diamonds are the genes of the IFN-γ module.');
  let a4 = 0.05, showB = true, showU = false;
  slider(f4g.controls, { label: 'FDR level α', min: 0.005, max: 0.25, step: 0.005, value: a4, fmt: f2, onInput: (v) => { a4 = v; paint4(); } });
  checkbox(f4g.controls, { label: 'Bonferroni cut-off', checked: showB, onChange: (c) => { showB = c; paint4(); } });
  checkbox(f4g.controls, { label: 'no correction (α)', checked: showU, onChange: (c) => { showU = c; paint4(); } });
  const s4 = stats(f4g.root);
  const ptsR = bhR.o.map((g, r) => ({ g, k: r + 1, p: pR[g], q: bhR.q[g] }));
  const P4 = stepPlot(f4g.body, { width: 820, height: 380, m: G, yLo: Math.max(1e-60, 10 ** Math.floor(Math.log10(sortedR[0]))) });
  legend(f4g.body, [['called by BH', cat(1)], ['not called', cat(0)], ['BH threshold α·k/m', cat(3)], ['Bonferroni α/m', cat(4)]]);
  function paint4() {
    const k = ptsR.filter((d) => d.q <= a4).length, kb = cntBelow(sortedR, a4 / G), ku = cntBelow(sortedR, a4);
    P4.draw({ pts: ptsR, called: (d) => d.k <= k, mark: (d) => grp[d.g] === 'IFNG module', alpha: a4, kstar: k, bonf: showB, raw: showU, label: `k* = ${nf(k)}`, tipFn: (d) => `${genes[d.g]} (${grp[d.g]})<br>rank ${d.k} · p = ${sci(d.p)}<br>BH threshold at this rank: ${sci(a4 * d.k / G)}<br>BH-adjusted p = ${sci(d.q)}` });
    s4.set({ 'BH calls (k*)': nf(k), 'Bonferroni calls': nf(kb), 'uncorrected calls': nf(ku), 'expected false calls ≤ α × k*': f1(a4 * k), 'IFN-γ module genes called': `${ifngIdx.filter((g) => bhR.q[g] <= a4).length} of ${ifngIdx.length}` });
  }
  paint4();
  prose(rb, `<p>At α = 0.05, BH calls <b>${nf(nBH)}</b> genes, against ${nf(nBonf)} for Bonferroni; ${ifngBH} of the ${ifngIdx.length} IFN-γ module genes (the strongest planted effect) are among them. The step-up rule calls every gene up to <i>k</i>*, including any whose own p-value lies above the line. At most about α × <i>k</i>* = ${f1(0.05 * nBH)} of the ${nf(nBH)} calls are expected to be false, if the tests behave like the theory assumes.</p>`);

  // ---- Step 5: pi0 and q-values
  const piOf = (l) => Math.min(1, sum(Array.from(pR).map((p) => (p > l ? 1 : 0))) / (G * (1 - l)));
  rb = rung(ch, 5, 'Adjusted p-values, π₀ and q-values',
    `<p>The <b>BH-adjusted p-value</b> of a gene is the smallest FDR level α at which BH would call it: q<sub>(<i>k</i>)</sub> = min<sub><i>j</i> ≥ <i>k</i></sub> <i>m</i> · p<sub>(<i>j</i>)</sub> / <i>j</i>. It lets a reader apply any threshold without rerunning the procedure. BH is conservative by the factor <b>π₀</b>, the proportion of genes that are truly null: its guarantee is FDR ≤ π₀ · α. <b>Storey's q-value</b> estimates π₀ from the right-hand side of the p-value histogram, where nearly all p-values come from null genes, and multiplies by it: π̂₀(λ) = #{p &gt; λ} / (<i>m</i> (1 − λ)).</p>`);
  const f5g = figure(rb, 'Estimating π₀ from the histogram', 'Real p-values in 20 bins. The estimate assumes that every p-value above λ (shaded) comes from a null gene. The dashed line is the height of a flat histogram with that estimated null proportion.');
  let lam5 = 0.5;
  slider(f5g.controls, { label: 'λ', min: 0.1, max: 0.9, step: 0.05, value: lam5, fmt: f2, onInput: (v) => { lam5 = v; paint5(); } });
  const s5 = stats(f5g.root);
  const F5 = frame(f5g.body, { width: 820, height: 300, x: [0, 1], y: [0, Math.ceil(d3.max(hR) / 100) * 100], xLabel: 'p-value (real labels)', yLabel: 'number of genes' });
  F5.layer('shade').append('rect').attr('y', 0).attr('height', F5.h).attr('fill', css('--surface-2'));
  F5.layer('bars').selectAll('rect').data(hR.map((c, i) => ({ c, i }))).join('rect').attr('x', (d) => F5.x(d.i / 20) + 1).attr('width', F5.w / 20 - 2).attr('y', (d) => F5.y(d.c)).attr('height', (d) => F5.h - F5.y(d.c)).attr('fill', cat(1)).attr('fill-opacity', 0.85)
    .on('mousemove', (e, d) => tip(e, `p in [${f2(d.i / 20)}, ${f2((d.i + 1) / 20)})<br>${nf(d.c)} genes`)).on('mouseleave', untip);
  const ln5 = F5.layer('ln'), tx5 = F5.layer('tx');
  const storeyQ = (pi0) => bhQ(pR, pi0).q;
  function paint5() {
    const pi0 = piOf(lam5), h = pi0 * G / 20;
    F5.layer('shade').select('rect').attr('x', F5.x(lam5)).attr('width', F5.w - F5.x(lam5));
    ln5.selectAll('line').data([0]).join('line').attr('x1', 0).attr('x2', F5.w).attr('y1', F5.y(h)).attr('y2', F5.y(h)).attr('stroke', ink()).attr('stroke-dasharray', '6 4').attr('stroke-width', 1.8);
    tx5.selectAll('text').data([0, 1]).join('text').attr('font-size', 12).attr('fill', ink2()).attr('text-anchor', (d) => (d ? 'start' : 'end')).attr('x', (d) => (d ? F5.x(lam5) + 6 : F5.w - 6)).attr('y', (d) => (d ? 16 : F5.y(h) - 8))
      .text((d) => (d ? `p > ${f2(lam5)}: ${nf(Math.round(pi0 * G * (1 - lam5)))} genes → π̂₀ = ${f2(pi0)}` : `flat level for π̂₀ = ${f2(pi0)}: ${f1(h)} per bin`));
    s5.set({ 'λ': f2(lam5), 'estimated π₀': f2(pi0), 'BH calls at 0.05': nf(nBH), 'q-value calls at 0.05': nf(sum(Array.from(storeyQ(pi0)).map((q) => (q <= 0.05 ? 1 : 0)))) });
  }
  paint5();
  const k5 = Math.round(nBH * 0.5), g5 = bhR.o[k5 - 1], pi05 = piOf(0.5);
  worked(rb, `One gene, three numbers (rank ${k5})`,
    `<p>The gene <b>${genes[g5]}</b> has the ${k5}th smallest p-value out of ${nf(G)}: p = ${sci(pR[g5])}.</p>
     <ul><li>Naive scaling: <i>m</i> · p / <i>k</i> = ${nf(G)} × ${sci(pR[g5])} / ${k5} = ${sci(G * pR[g5] / k5)}.</li>
     <li>BH-adjusted p-value (the smallest such value at this rank or beyond, so that adjusted values never decrease with rank): <b>${sci(bhR.q[g5])}</b>.</li>
     <li>Storey q-value at λ = 0.5 (π̂₀ = ${f2(pi05)}): ${sci(storeyQ(pi05)[g5])}.</li></ul>
     <p>Calling this gene is acceptable at any FDR level above its adjusted p-value. The q-value is lower because it credits the share of genes that are not null.</p>`);
  prose(rb, `<p>On this dataset π̂₀ = ${f2(pi05)} at λ = 0.5, so the q-value method calls ${nf(sum(Array.from(storeyQ(pi05)).map((q) => (q <= 0.05 ? 1 : 0))))} genes against ${nf(nBH)} for BH. π̂₀ depends on λ and on the assumption that p-values above λ are null; it is least reliable when the test is mis-specified or effects are small and numerous, as here.</p>`);
  quiz(rb, 'A gene has q = 0.03. Is the probability that this gene is a false positive 3%?', '<p>No. The q-value is the FDR of the list of all genes with q at or below 0.03: on average 3% of that list is false. The probability that this particular gene is null (its <i>local</i> false discovery rate) is usually higher, because the genes at the end of the list are the least certain (pitfall 8).</p>');

  // ---- Step 6: lab
  rb = rung(ch, 6, 'Lab: a spike-in experiment with known truth',
    `<p>Real data have no answer key. This experiment starts from the real expression matrix with the response labels shuffled, so every gene is null, then <b>plants</b> a true difference between the two groups in a chosen share of genes. Because the planted genes are known, true and false positives can be counted exactly. The effect is set in standard deviations of each gene; n = ${nR} versus ${N - nR} patients.</p>`);
  const flb = figure(rb, 'Spike-in lab', 'Left: the step-up plot for one experiment. Each point is a gene, coloured by its true status and whether the selected procedure called it (see legend). The line is the cut-off of the selected procedure. Right: the outcome of all four procedures on the same experiment. Bars: true positives plus false positives (labels give both counts). Click a bar to select it.');
  const lab = { seed: 1, frac: 0.1, d: 0.6, alpha: 0.05, method: 'bh' };
  const sFr = slider(flb.controls, { label: 'share of genes with a real effect', min: 0, max: 0.4, step: 0.01, value: lab.frac, fmt: (v) => Math.round(v * 100) + '%', onInput: (v) => { lab.frac = v; paintL(); } });
  slider(flb.controls, { label: 'effect size (SD)', min: 0.2, max: 1.2, step: 0.05, value: lab.d, fmt: f2, onInput: (v) => { lab.d = v; paintL(); } });
  slider(flb.controls, { label: 'α / FDR level', min: 0.01, max: 0.2, step: 0.01, value: lab.alpha, fmt: f2, onInput: (v) => { lab.alpha = v; paintL(); } });
  const selM = select(flb.controls, { label: 'Procedure', value: lab.method, options: METHODS, onChange: (v) => { lab.method = v; paintL(); } });
  button(flb.controls, 'New experiment', () => { lab.seed++; paintL(); }, true);
  const sL = stats(flb.root);
  const rowL = el('div', { class: 'fig-row' }); flb.body.append(rowL);
  const lA = el('div'), lB = el('div'); rowL.append(lA, lB);
  const PL = stepPlot(lA, { width: 500, height: 360, m: G, yLo: 1e-14 });
  const FB6 = frame(lB, { width: 460, height: 360, x: [0, 1], y: [0, 4], axes: false, grid: false, margin: { left: 104, right: 14, bottom: 46 } });
  const bars6 = FB6.layer('bars'), ax6 = FB6.layer('ax');
  const lgL = legend(flb.body, [['true effect, called', cat(2)], ['false positive, called', cat(7)], ['true effect, missed (open circle)', cat(2)], ['null, not called', cat(0)]]);
  lgL.children[2].querySelector('i').style.cssText = `background:transparent;border:2px solid ${cat(2)};box-sizing:border-box`;   // ring, as in the plot
  function paintL() {
    const sim = labSim(lab.seed, lab.frac, lab.d), res = Object.fromEntries(METHODS.map(([k]) => [k, outcome(sim, k, lab.alpha)])), cur = res[lab.method];
    const ord = sim.bh.o.map((g, r) => ({ g, k: r + 1, p: sim.p[g], sp: sim.sp[g], f: cur.f[g] }));
    PL.draw({ pts: ord, fillFn: (d) => (d.f ? (d.sp ? cat(2) : cat(7)) : d.sp ? 'none' : cat(0)), strokeFn: (d) => (!d.f && d.sp ? cat(2) : css('--surface')), opacity: (d) => (!d.f && !d.sp ? 0.4 : 1), thr: thrFn(lab.method, lab.alpha, G), thrLabel: 'cut-off: ' + { none: 'α', bonf: 'α/m', holm: 'α/(m−k+1)', bh: 'α·k/m' }[lab.method],
      tipFn: (d) => `rank ${d.k} · p = ${sci(d.p)}<br>${d.sp ? 'planted effect' : 'true null'} · ${d.f ? 'called' : 'not called'}` });
    const xmax = Math.max(sim.K, d3.max(Object.values(res), (o) => o.n), 10) * 1.4, xs = d3.scaleLinear().domain([0, xmax]).range([0, FB6.w]), ysb = d3.scaleBand().domain(METHODS.map((q) => q[0])).range([0, FB6.h]).padding(0.28);
    ax6.selectAll('*').remove();
    ax6.append('g').attr('class', 'axis x').attr('transform', `translate(0,${FB6.h})`).call(d3.axisBottom(xs).ticks(5));
    ax6.append('text').attr('class', 'ax-label').attr('x', FB6.w / 2).attr('y', FB6.h + 38).attr('text-anchor', 'middle').text('genes called');
    METHODS.forEach(([k, name]) => ax6.append('text').attr('x', -8).attr('y', ysb(k) + ysb.bandwidth() / 2 + 4).attr('text-anchor', 'end').attr('font-size', 11.5).attr('font-weight', k === lab.method ? 700 : 400).attr('fill', ink2()).text({ none: 'No correction', bonf: 'Bonferroni', holm: 'Holm', bh: 'BH' }[k]));
    ax6.append('line').attr('x1', xs(sim.K)).attr('x2', xs(sim.K)).attr('y1', 0).attr('y2', FB6.h).attr('stroke', ink()).attr('stroke-dasharray', '4 4');
    ax6.append('text').attr('x', xs(sim.K) + 4).attr('y', 11).attr('font-size', 11.5).attr('fill', ink2()).text(`${sim.K} real effects`);
    const rowsB = METHODS.map(([k]) => ({ k, ...res[k] }));
    bars6.selectAll('g').remove();
    const gb = bars6.selectAll('g').data(rowsB).join('g').style('cursor', 'pointer').on('click', (e, d) => { lab.method = d.k; selM.value = d.k; paintL(); })
      .on('mousemove', (e, d) => tip(e, `${d.n} called: ${d.tp} true, ${d.fp} false<br>false discovery proportion ${pc(d.fdp)}<br>power ${sim.K ? pc(d.tp / sim.K) : '—'}`)).on('mouseleave', untip);
    gb.append('rect').attr('x', 0).attr('y', (d) => ysb(d.k) + 3).attr('width', (d) => xs(d.tp)).attr('height', ysb.bandwidth() - 6).attr('fill', cat(2));
    gb.append('rect').attr('x', (d) => xs(d.tp)).attr('y', (d) => ysb(d.k) + 3).attr('width', (d) => xs(d.fp)).attr('height', ysb.bandwidth() - 6).attr('fill', cat(7));
    gb.append('rect').attr('x', 0).attr('y', (d) => ysb(d.k)).attr('width', (d) => xs(d.n)).attr('height', ysb.bandwidth()).attr('fill', 'none').attr('stroke', (d) => (d.k === lab.method ? ink() : 'none')).attr('stroke-width', 2);
    gb.append('text').attr('x', (d) => xs(d.n) + 5).attr('y', (d) => ysb(d.k) + ysb.bandwidth() / 2 + 4).attr('font-size', 11.5).attr('fill', ink2()).text((d) => `${d.tp} + ${d.fp}`);
    sL.set({ 'real effects planted': sim.K, 'genes called': cur.n, 'true positives': cur.tp, 'false positives': cur.fp, 'false discovery proportion': cur.n ? pc(cur.fdp) : '—', 'power': sim.K ? pc(cur.tp / sim.K) : '—' });
  }
  paintL();

  prose(rb, '<p><b>Suggested comparisons</b> (each row averages 12 experiments, seeds 1–12; α = 0.05):</p>');
  const r1 = labMean(0.1, 0.6, 0.05, 'none'), r1b = labMean(0.1, 0.6, 0.05, 'bonf'), r1h = labMean(0.1, 0.6, 0.05, 'bh');
  const r2 = labMean(0.1, 1.0, 0.05, 'bonf'), r2h = labMean(0.1, 1.0, 0.05, 'bh');
  const r3 = labMean(0, 0.6, 0.05, 'none'), r3b = labMean(0, 0.6, 0.05, 'bonf'), r3h = labMean(0, 0.6, 0.05, 'bh');
  const r4 = labMean(0.3, 0.6, 0.05, 'bh'), r4b = labMean(0.3, 0.6, 0.05, 'bonf');
  table(rb, [
    { t: `<b>No correction</b> → <b>Bonferroni</b> → <b>BH</b> (10% real, 0.6 SD)`, s: `True calls ${f1(r1.tp)} → ${f1(r1b.tp)} → ${f1(r1h.tp)} of ${r1.K}; false calls ${f1(r1.fp)} → ${f1(r1b.fp)} → ${f1(r1h.fp)}`, m: `No correction finds the most real genes, but ${pc(r1.fdp)} of its list is false. Bonferroni removes nearly all false calls and misses ${pc(1 - r1b.tp / r1.K)} of the real genes. BH finds ${pc(r1h.tp / r1.K)} of them, with a false share of ${pc(r1h.fdp)} (target ≤ 5%).` },
    { t: 'Raise the effect size <b>0.6 → 1.0</b> SD', s: `Power: Bonferroni ${pc(r2.tp / r2.K)}, BH ${pc(r2h.tp / r2h.K)}; BH false share ${pc(r2h.fdp)}`, m: '<b>Larger effects shrink the penalty for correcting.</b> Detection becomes easy, and the choice of procedure matters less.' },
    { t: 'Set the real share to <b>0%</b>', s: `Experiments with ≥ 1 false call: none ${pc(r3.anyFP)}, Bonferroni ${pc(r3b.anyFP)}, BH ${pc(r3h.anyFP)}`, m: `<b>With no real effects, FDR control and FWER control coincide</b>: every call is a false call, so BH must stay at or below α in the share of experiments with any call (here ${Math.round(r3h.anyFP * 12)} of 12). Without correction there are ${nf(r3.fp)} false calls on average.` },
    { t: 'Set the real share to <b>30%</b>', s: `BH true calls ${f1(r4.tp)} of ${r4.K} vs Bonferroni ${f1(r4b.tp)}; BH false share ${pc(r4.fdp)}`, m: `<b>The more real effects, the more BH gains over Bonferroni</b>, and its realised false share falls below α, in line with FDR ≤ π₀ · α (π₀ = ${f2(1 - r4.K / G)} here).` },
    { t: 'Press <b>New experiment</b> repeatedly (BH, 10%, 0.6 SD)', s: `False share ranges from ${pc(r1h.fdpMin)} to ${pc(r1h.fdpMax)} over 12 experiments`, m: '<b>FDR is an average.</b> A single experiment can be above or below α; genes are correlated, so the spread is wider than for independent tests.' },
  ], [['t', 'Try'], ['s', 'What you see'], ['m', 'What it means']], { cls: 'compact' });

  // =====================================================================================
  // 5. PITFALLS
  // =====================================================================================
  ch = chapter(main, 'pitfalls', 'Pitfalls', 'Common problems, how to recognise them, and how to address them.');
  const simP = labSim(1, 0.1, 0.6), oP = outcome(simP, 'bh', 0.05), calledOrd = simP.bh.o.filter((g) => oP.f[g]);
  const seg = Math.min(100, Math.floor(calledOrd.length / 3)), fpFirst = calledOrd.slice(0, seg).filter((g) => !simP.sp[g]).length, fpLast = calledOrd.slice(-seg).filter((g) => !simP.sp[g]).length;
  const bhLowEff = bhR.o.filter((g) => bhR.q[g] <= 0.05 && Math.abs(REAL.diff[g]) < 0.1).length;
  const Hm = sum(d3.range(1, G + 1).map((i) => 1 / i));
  cards(ch, [
    { tag: `Symptom: "${nf(nRaw)} significant genes"`, title: 'Reporting uncorrected p-values', body: `At p &lt; 0.05 with ${nf(G)} tests, about ${f1(expRaw)} genes pass in any data set, including shuffled ones (range ${nf(d3.min(permRaw))}–${nf(d3.max(permRaw))} over ${NP} shuffles). <b>Fix:</b> report an adjusted p-value (FDR or FWER) and state which one.` },
    { tag: 'Symptom: results change with the analysis scope', title: 'The wrong family of tests', body: 'The correction must cover every hypothesis tested in the search: all genes, and all contrasts, subgroups and models tried. Correcting over only the genes that looked interesting, or over each contrast separately while reporting the union, understates the error. <b>Fix:</b> define the family before looking, and count every test.' },
    { tag: 'Symptom: thousands of hits that follow a technical variable', title: 'Confounding inflates the number of hits', body: `The response rate is ${pc(respA)} in Centre A and ${pc(respB)} in Centre B, and ${nf(nCentreBH)} of ${nf(G)} genes differ between centres at BH q &lt; 0.05. Gene-level differences between responders may partly reflect the centre. A valid correction cannot repair a biased p-value. <b>Fix:</b> model the batch (as a covariate), check that the p-value histogram is flat on the right, and use shuffles within centre.` },
    { tag: `Symptom: shuffled runs give ${nf(d3.min(permRaw))} to ${nf(d3.max(permRaw))} hits`, title: 'Correlated genes', body: `Genes in a module move together, so false positives arrive in clusters. BH is proven under independence or positive dependence, and its realised false share varies between experiments (step 6). For arbitrary dependence, the Benjamini–Yekutieli variant divides the level by Σ1/<i>i</i> = ${f1(Hm)} for ${nf(G)} tests, at a large cost in power. <b>Fix:</b> permutation checks, and awareness that the FDR is an average.` },
    { tag: 'Symptom: nothing survives Bonferroni', title: 'Bonferroni with many real effects', body: `Bonferroni keeps ${nf(nBonf)} genes where BH keeps ${nf(nBH)}. When many true effects are expected, controlling FWER discards most of them. <b>Fix:</b> use FDR for exploration; keep FWER for a small number of confirmatory claims.` },
    { tag: 'Symptom: tiny differences listed as hits', title: 'Significance is not importance', body: `${nf(bhLowEff)} of the ${nf(nBH)} BH calls have an estimated difference below 0.1 log2 units. With enough patients, trivial effects reach very small p-values. <b>Fix:</b> report effect sizes with intervals, and add a minimum effect-size criterion chosen in advance.` },
    { tag: 'Symptom: no gene can pass', title: 'Permutation p-values have a floor', body: `With <i>B</i> = 999 permutations the smallest possible p-value is 1/(<i>B</i> + 1) = 0.001, above the Bonferroni cut-off ${sci(0.05 / G)}, so no gene can pass it. Reaching it needs more than ${nf(G / 0.05 - 1)} permutations. <b>Fix:</b> more permutations, a parametric tail approximation, or an FDR procedure.` },
    { tag: 'Symptom: the last genes in the list are wrong more often', title: 'FDR describes the list, not each gene', body: `In one spike-in run (10% real, 0.6 SD, BH at 0.05) the list has ${oP.n} genes and a false share of ${pc(oP.fdp)}. Among its ${seg} smallest p-values ${fpFirst} are false; among its ${seg} largest, ${fpLast}. <b>Fix:</b> rank follow-up by p-value, and give priority to genes well inside the cut-off.` },
  ]);

  // =====================================================================================
  // 6. RECAP
  // =====================================================================================
  ch = chapter(main, 'recap', 'Recap', 'Key points.');
  recap(ch, [
    'A p-value is uniform on 0–1 when the null hypothesis is true. Among <i>m</i> null tests, about <i>m</i> × α pass the threshold α by chance.',
    `With ${nf(G)} genes, ${f1(expRaw)} false positives are expected at p &lt; 0.05. The number of hits in the real data (${nf(nRaw)}) must be read against that background.`,
    '<b>Bonferroni and Holm</b> control the chance of any false positive (FWER). The per-gene cut-off is α / <i>m</i>, which is strict and costs power.',
    '<b>Benjamini–Hochberg</b> sorts the p-values and calls every gene up to the largest rank <i>k</i> with p<sub>(<i>k</i>)</sub> ≤ α <i>k</i> / <i>m</i>. It controls the expected share of false calls (FDR).',
    'An <b>adjusted p-value or q-value</b> is the smallest error level at which a gene would be called. A q-value of 0.05 describes the list up to that gene, not the gene.',
    'Storey\'s π̂₀ estimates the share of true nulls from the flat right-hand side of the p-value histogram; it must be checked against the data.',
    'FDR is an average. With correlated genes, one experiment can fall well above or below α.',
    'Correction cannot repair biased p-values: model batch effects, check the histogram, define the family beforehand, and report effect sizes alongside adjusted p-values.',
  ]);
  prose(ch, '<h3>In code</h3><p>Adjusted p-values are one function call in both languages. For genome-scale expression, use the p-values and adjustment built into the differential-expression package rather than a separate t-test loop.</p>');
  code(ch, {
    Python: `import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

# expr: genes x patients array; resp: boolean vector of responders
t, p = stats.ttest_ind(expr[:, resp], expr[:, ~resp], axis=1, equal_var=False)   # Welch t-test per gene

bonf = multipletests(p, alpha=0.05, method="bonferroni")[0]   # FWER
holm = multipletests(p, alpha=0.05, method="holm")[0]         # FWER, never less powerful than Bonferroni
reject, q, _, _ = multipletests(p, alpha=0.05, method="fdr_bh")   # BH; q = adjusted p-values
print(p.size, (p < 0.05).sum(), bonf.sum(), reject.sum())

# SciPy >= 1.11 equivalent: stats.false_discovery_control(p, method="bh")
# Benjamini–Yekutieli (any dependence): method="fdr_by"

# Null check: shuffle the labels and confirm the p-value histogram is flat
perm = np.random.default_rng(1).permutation(resp)
_, p0 = stats.ttest_ind(expr[:, perm], expr[:, ~perm], axis=1, equal_var=False)`,
    R: `# expr: genes x patients matrix; resp: logical vector of responders
p <- apply(expr, 1, function(x) t.test(x[resp], x[!resp])$p.value)   # Welch t-test per gene

bonf <- p.adjust(p, method = "bonferroni")   # FWER
holm <- p.adjust(p, method = "holm")
bh   <- p.adjust(p, method = "BH")           # adjusted p-values for FDR control
c(raw = sum(p < 0.05), bonferroni = sum(bonf < 0.05), BH = sum(bh < 0.05))

# Storey q-values and pi0 (Bioconductor)
# qv <- qvalue::qvalue(p); qv$pi0; sum(qv$qvalues < 0.05)

# With limma or DESeq2 the adjustment is already in the table:
# topTable(fit, number = Inf)$adj.P.Val      # BH; results(dds)$padj`,
  });
  seeAlso(main, [['concepts', 'logistic-regression', 'Logistic regression'], ['concepts', 'cross-validation', 'Cross-validation and leakage'], ['concepts', 'regularization', 'Regularisation'], ['concepts', 'metrics', 'Evaluating a classifier'], ['omics', 'bulk-de', 'Differential expression']]);
}

// ---------------------------------------------------------------------------------------------
/** Log-log rank/p-value plot with threshold lines (the BH "step-up" picture). draw() can be called repeatedly. */
function stepPlot(parent, { width, height, m, yLo }) {
  const F = frame(parent, { width, height, x: [1, m], y: [yLo, 1], xScale: d3.scaleLog(), yScale: d3.scaleLog(), axes: false, grid: false, margin: { left: 68, bottom: 46, top: 12 } });
  const lo = Math.round(Math.log10(yLo)), stepY = Math.max(1, Math.ceil(-lo / 6)), tv = d3.range(0, lo - 1, -stepY).map((k) => 10 ** k);
  F.g.append('g').attr('class', 'gridline').selectAll('line').data(tv).join('line').attr('x1', 0).attr('x2', F.w).attr('y1', (v) => F.y(v)).attr('y2', (v) => F.y(v)).attr('stroke', css('--axis')).attr('stroke-opacity', 0.35);
  F.g.append('g').attr('class', 'axis y').call(d3.axisLeft(F.y).tickValues(tv).tickFormat((v) => '10' + sup(Math.round(Math.log10(v)))));
  F.g.append('g').attr('class', 'axis x').attr('transform', `translate(0,${F.h})`).call(d3.axisBottom(F.x).tickValues([1, 10, 100, 1000].filter((v) => v <= m)).tickFormat(nf));
  F.g.append('text').attr('class', 'ax-label').attr('x', F.w / 2).attr('y', F.h + 38).attr('text-anchor', 'middle').text('rank k of the p-value (1 = smallest), log scale');
  F.g.append('text').attr('class', 'ax-label').attr('transform', 'rotate(-90)').attr('x', -F.h / 2).attr('y', -54).attr('text-anchor', 'middle').text('p-value (log scale)');
  const Lr = F.layer('lines'), Lp = F.layer('pts'), Lt = F.layer('txt');
  const clip = (p) => Math.max(yLo * 1.001, p);
  return {
    F,
    draw(o) {
      const pts = o.pts, thr = o.thr || ((k) => o.alpha * k / m), alpha = o.alpha;
      const ks = d3.range(1, m + 1, Math.max(1, Math.floor(m / 400))).concat([m]);
      const lines = [];
      if (o.thr) lines.push({ f: thr, c: ink(), name: o.thrLabel });
      else {
        lines.push({ f: thr, c: cat(3), name: 'BH: α·k/m', x: 0.55 });
        if (o.bonf) lines.push({ f: () => alpha / m, c: cat(4), name: 'Bonferroni: α/m', x: 0.0 });
        if (o.raw) lines.push({ f: () => alpha, c: muted(), name: 'no correction: α', x: 0.0, dash: '4 4' });
      }
      Lr.selectAll('path').data(lines).join('path').attr('fill', 'none').attr('stroke', (d) => d.c).attr('stroke-width', 2.2).attr('stroke-dasharray', (d) => d.dash || null)
        .attr('d', (d) => d3.line().x((k) => F.x(k)).y((k) => F.y(clip(d.f(k))))(ks));
      Lt.selectAll('text.ln').data(lines).join('text').attr('class', 'ln').attr('font-size', 11.5).attr('fill', ink2())
        .attr('x', (d) => (o.thr ? 6 : d.x ? F.x(m) * d.x : 6)).attr('text-anchor', (d) => (!o.thr && d.x ? 'end' : 'start'))
        .attr('y', (d) => F.y(clip(d.f(o.thr ? 3 : d.x ? m * d.x : 3))) + (o.thr ? -8 : d.x ? 16 : -6)).text((d) => d.name);
      const order = [...pts.filter((d) => (o.fillFn ? o.fillFn(d) === cat(0) : !o.called(d))), ...pts.filter((d) => (o.fillFn ? o.fillFn(d) !== cat(0) : o.called(d)))];
      Lp.selectAll('circle').data(order, (d) => d.g).join('circle').attr('cx', (d) => F.x(d.k)).attr('cy', (d) => F.y(clip(d.p))).attr('r', (d) => (o.mark && o.mark(d) ? 0 : 2.8))
        .attr('fill', (d) => (o.fillFn ? o.fillFn(d) : o.called(d) ? cat(1) : cat(0))).attr('fill-opacity', (d) => (o.opacity ? o.opacity(d) : o.called(d) ? 0.9 : 0.45))
        .attr('stroke', (d) => (o.strokeFn ? o.strokeFn(d) : css('--surface'))).attr('stroke-width', (d) => (o.strokeFn && o.strokeFn(d) !== css('--surface') ? 1.4 : 0.5))
        .on('mousemove', (e, d) => tip(e, o.tipFn(d))).on('mouseleave', untip);
      Lp.selectAll('path.mk').data(o.mark ? pts.filter(o.mark) : []).join('path').attr('class', 'mk').attr('d', d3.symbol(d3.symbolDiamond, 55)).attr('transform', (d) => `translate(${F.x(d.k)},${F.y(clip(d.p))})`)
        .attr('fill', (d) => (o.called(d) ? cat(1) : cat(0))).attr('stroke', ink()).attr('stroke-width', 1.2).on('mousemove', (e, d) => tip(e, o.tipFn(d))).on('mouseleave', untip);
      const ks2 = o.kstar;
      Lt.selectAll('line.k').data(ks2 ? [ks2] : []).join('line').attr('class', 'k').attr('x1', (k) => F.x(k)).attr('x2', (k) => F.x(k)).attr('y1', 0).attr('y2', F.h).attr('stroke', ink()).attr('stroke-dasharray', '3 3');
      Lt.selectAll('text.k').data(ks2 ? [ks2] : []).join('text').attr('class', 'k').attr('x', (k) => F.x(k) - 5).attr('text-anchor', 'end').attr('y', 14).attr('font-size', 12).attr('fill', ink()).text(o.label);
    },
  };
}

/** The diagram for chapter 1: data → one test per gene → p-values → correction. */
function pipelineSvg(I) {
  const wrap = el('div', { class: 'figure' });
  wrap.append(el('div', { class: 'fig-title' }, 'From a data matrix to a list of significant genes'));
  const svg = d3.select(wrap).append('svg').attr('viewBox', '0 0 980 250').attr('width', 980).attr('role', 'img').attr('aria-label', 'expression matrix, one test per gene, p-value histogram, corrected gene lists');
  const gi = I.gi, t = I.REAL.diff[gi] / I.REAL.se[gi];
  const boxes = [
    { t: 'Data', lines: [`${nf(I.G)} genes`, `${I.N} patients`, `${I.nR} responders`, `${I.nonR} non-responders`, 'log2 expression'] },
    { t: '① One test per gene', lines: ['Welch t-test, e.g. IFNG:', `difference ${f2(I.REAL.diff[gi])} log2`, `t = ${f1(t)}`, `p = ${sci(I.pR[gi])}`, `…repeated ${nf(I.G)} times`] },
    { t: `② ${nf(I.G)} p-values`, lines: [], hist: true },
    { t: '③ Adjust and call', lines: [`p < 0.05:  ${nf(I.nRaw)}`, `Bonferroni:  ${nf(I.nBonf)}`, `BH (FDR 5%):  ${nf(I.nBH)}`, '', 'Which list to report', 'depends on the error', 'to be controlled.'] },
  ];
  const BW = 212, gap = 36, H = 220;
  boxes.forEach((b, i) => {
    const x0 = 10 + i * (BW + gap), g = svg.append('g').attr('transform', `translate(${x0},14)`);
    g.append('rect').attr('width', BW).attr('height', H).attr('rx', 10).attr('fill', css('--surface-2')).attr('stroke', css('--axis'));
    g.append('text').attr('x', 12).attr('y', 24).attr('font-weight', 700).attr('font-size', 13.5).attr('fill', ink()).text(b.t);
    b.lines.forEach((l, j) => g.append('text').attr('x', 14).attr('y', 54 + j * 24).attr('font-family', css('--mono')).attr('font-size', 12.5).attr('fill', ink2()).text(l));
    if (b.hist) {
      const hx = d3.scaleLinear().domain([0, 20]).range([16, BW - 16]), hy = d3.scaleLinear().domain([0, d3.max(I.hist)]).range([H - 40, 44]);
      g.selectAll('rect.b').data(I.hist).join('rect').attr('x', (d, k) => hx(k) + 0.5).attr('width', (BW - 32) / 20 - 1).attr('y', (d) => hy(d)).attr('height', (d) => H - 40 - hy(d)).attr('fill', cat(1)).attr('fill-opacity', 0.85);
      g.append('line').attr('x1', 16).attr('x2', BW - 16).attr('y1', H - 40).attr('y2', H - 40).attr('stroke', muted());
      g.append('text').attr('x', 16).attr('y', H - 24).attr('font-size', 11).attr('fill', muted()).text('0');
      g.append('text').attr('x', BW - 16).attr('y', H - 24).attr('text-anchor', 'end').attr('font-size', 11).attr('fill', muted()).text('1');
      g.append('text').attr('x', BW / 2).attr('y', H - 8).attr('text-anchor', 'middle').attr('font-size', 11.5).attr('fill', ink2()).text('histogram of p-values');
    }
    if (i < boxes.length - 1) {
      const ax = x0 + BW + 6, ay = 14 + H / 2;
      svg.append('path').attr('d', `M${ax},${ay} L${ax + gap - 14},${ay} M${ax + gap - 20},${ay - 6} L${ax + gap - 14},${ay} L${ax + gap - 20},${ay + 6}`).attr('fill', 'none').attr('stroke', ink2()).attr('stroke-width', 2);
    }
  });
  return wrap;
}
