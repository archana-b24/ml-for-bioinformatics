// Multiple testing & FDR: a simulated differential-expression experiment with 2,000 genes.
import { header, figure, slider, button, stats, section, steps, callout, math, legend, seeAlso, tip, untip, el } from '../lib/ui.js';
import { frame, cat, css, ink, muted } from '../lib/plot.js';
import { rng } from '../lib/data.js';
const d3 = window.d3;

const M = 2000;

// Complementary error function (Numerical Recipes erfcc, fractional error < 1.2e-7 everywhere).
function erfc(x) {
  const z = Math.abs(x), t = 1 / (1 + 0.5 * z);
  const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 +
    t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
  return x >= 0 ? r : 2 - r;
}
const twoSidedP = (z) => Math.max(1e-300, erfc(Math.abs(z) / Math.SQRT2));

/** One simulated DE experiment: each gene has a true log2FC (0 for nulls), a standard error, an estimate and a p-value. */
function simulate(frac, effect, seed) {
  const r = rng(seed);
  const nDE = Math.round(M * frac);
  const isDE = r.shuffle(d3.range(M).map((i) => i < nDE));
  const genes = isDE.map((de, i) => {
    const se = 0.15 + 0.35 * r();
    const lfc = de ? (r() < 0.5 ? -1 : 1) * effect * (0.5 + r()) : 0;
    const est = lfc + r.normal(0, se);
    const z = est / se;
    return { id: 'Gene' + String(i + 1).padStart(4, '0'), de, lfc, est, se, z, p: twoSidedP(z) };
  });
  // BH-adjusted p-values (q-values in the BH sense): q_(i) = min_{j >= i} m p_(j) / j
  const order = genes.slice().sort((a, b) => a.p - b.p);
  let run = 1;
  for (let i = M - 1; i >= 0; i--) { run = Math.min(run, (order[i].p * M) / (i + 1)); order[i].q = run; order[i].rank = i + 1; }
  return { genes, order, nDE };
}

function calls(sim, alpha) {
  const { genes, order } = sim;
  const count = (pred) => { const c = genes.filter(pred); return { n: c.length, fp: c.filter((g) => !g.de).length }; };
  let k = 0;
  order.forEach((g, i) => { if (g.p <= (alpha * (i + 1)) / M) k = i + 1; });
  return {
    raw: count((g) => g.p < alpha),
    bonf: count((g) => g.p < alpha / M),
    bh: count((g) => g.q <= alpha),
    k,
  };
}

export default async function render(main) {
  header(main, 'Multiple testing & FDR',
    'Test 2,000 genes at p &lt; 0.05 and about 100 null genes will pass <b>by chance alone</b>. Multiple-testing corrections decide what you are willing to be wrong about: Bonferroni controls the chance of <i>any</i> false hit, Benjamini–Hochberg controls the <b>fraction</b> of false hits among your discoveries.');

  let frac = 0.1, effect = 1, alpha = 0.05, seed = 7, sim;

  // ---------------- Figure 1: histogram + volcano ----------------
  const fig = figure(main, 'A differential-expression experiment: 2,000 genes, treated vs control', '');
  slider(fig.controls, { label: '% truly DE', min: 0, max: 40, step: 1, value: 10, fmt: (v) => v + '%', onInput: (v) => { frac = v / 100; resim(); } });
  slider(fig.controls, { label: 'Effect size', min: 0.4, max: 2, step: 0.1, value: 1, fmt: (v) => v.toFixed(1) + '×', onInput: (v) => { effect = v; resim(); } });
  slider(fig.controls, { label: 'α / FDR level', min: 0.01, max: 0.2, step: 0.01, value: 0.05, fmt: (v) => v.toFixed(2), onInput: (v) => { alpha = v; paint(); } });
  button(fig.controls, 'New experiment', () => { seed++; resim(); });
  const readout = stats(fig.root);
  legend(fig.root, [['null gene (no true change)', cat(0)], ['truly DE gene', cat(1)], ['not significant after BH', muted()]]);
  const row = el('div', { class: 'fig-row' }); fig.body.append(row);
  const hDiv = el('div'), vDiv = el('div'); row.append(hDiv, vDiv);
  const H = frame(hDiv, { width: 480, height: 340, x: [0, 1], y: [0, 400], xLabel: 'p-value', yLabel: 'number of genes' });
  const V = frame(vDiv, { width: 480, height: 340, x: [-4, 4], y: [0, 16], xLabel: 'estimated log₂ fold change', yLabel: '−log₁₀ p' });

  // ---------------- Figure 2: sorted p-values with BH line ----------------
  const fig2 = figure(main, 'Sorted p-values: where each correction draws its line',
    'Genes sorted from smallest p-value (rank 1) to largest, on log–log axes. Benjamini–Hochberg finds the <b>last</b> gene that sits under the sloped line α·rank/m and calls everything up to it significant. Hover a point for its p- and q-value.');
  const legend2 = legend(fig2.root, [['null gene', cat(0)], ['truly DE gene', cat(1)], ['BH line α·i/m', ink()], ['Bonferroni α/m', cat(7)], ['raw α', muted()]]);
  legend2.remove(); fig2.controls.append(legend2);
  const S = frame(fig2.body, { width: 980, height: 360, x: [1, M], y: [1e-12, 1], xScale: d3.scaleLog(), yScale: d3.scaleLog(), xLabel: 'rank of gene (sorted by p)', yLabel: 'p-value (log)', yTicks: 6 });
  S.g.select('.axis.y').call(d3.axisLeft(S.y).ticks(6, '.0e'));
  S.g.select('.axis.x').call(d3.axisBottom(S.x).ticks(6, '~s'));
  const S2 = stats(fig2.root);

  function resim() { sim = simulate(frac, effect, seed); paint(); }

  function paint() {
    const c = calls(sim, alpha);
    const { genes, order, nDE } = sim;
    const sigBH = (g) => g.q <= alpha;

    // histogram, stacked null / DE
    const bins = d3.range(20).map((b) => ({ b, nul: 0, de: 0 }));
    genes.forEach((g) => { const b = Math.min(19, Math.floor(g.p * 20)); g.de ? bins[b].de++ : bins[b].nul++; });
    const ymax = Math.max(200, d3.max(bins, (d) => d.nul + d.de) * 1.08);
    H.y.domain([0, ymax]); H.g.select('.axis.y').call(d3.axisLeft(H.y).ticks(5));
    H.g.select('.gridline').call(d3.axisLeft(H.y).ticks(5).tickSize(-H.w).tickFormat(''));
    const bw = H.x(0.05) - H.x(0) - 1.5;
    const rects = [];
    bins.forEach((d) => { rects.push({ ...d, k: 'nul', y0: 0, y1: d.nul }); rects.push({ ...d, k: 'de', y0: d.nul, y1: d.nul + d.de }); });
    H.layer('bars').selectAll('rect').data(rects).join('rect')
      .attr('x', (d) => H.x(d.b * 0.05) + 0.75).attr('width', bw).attr('y', (d) => H.y(d.y1)).attr('height', (d) => H.y(d.y0) - H.y(d.y1))
      .attr('fill', (d) => (d.k === 'nul' ? cat(0) : cat(1))).attr('rx', 1.5)
      .on('mousemove', (e, d) => tip(e, `p in [${(d.b * 0.05).toFixed(2)}, ${(d.b * 0.05 + 0.05).toFixed(2)})<br>${d.nul} null genes · ${d.de} truly DE`)).on('mouseleave', untip);
    const flat = (M - nDE) / 20;
    H.layer('ref').selectAll('*').remove();
    H.layer('ref').append('line').attr('x1', 0).attr('x2', H.w).attr('y1', H.y(flat)).attr('y2', H.y(flat))
      .attr('stroke', ink()).attr('stroke-width', 1.5).attr('stroke-dasharray', '5 4');
    H.layer('ref').append('text').attr('x', H.w - 4).attr('y', H.y(flat) - 6).attr('text-anchor', 'end').attr('font-size', 11.5).attr('fill', ink())
      .text(`expected nulls per bin = ${Math.round(flat)}`);

    // volcano
    const ylim = 16, clampY = (p) => Math.min(ylim, -Math.log10(p));
    V.layer('thr').selectAll('*').remove();
    const pBH = c.k ? order[c.k - 1].p : null;
    const hline = (p, color, label, dash) => {
      if (p == null) return; const yy = V.y(clampY(p));
      V.layer('thr').append('line').attr('x1', 0).attr('x2', V.w).attr('y1', yy).attr('y2', yy).attr('stroke', color).attr('stroke-width', 1.5).attr('stroke-dasharray', dash);
      V.layer('thr').append('text').attr('x', 4).attr('y', yy - 4).attr('font-size', 11).attr('fill', ink()).text(label);
    };
    hline(alpha, muted(), `raw p = ${alpha}`, '3 3');
    hline(alpha / M, cat(7), 'Bonferroni', '6 3');
    hline(pBH, ink(), 'BH cut-off', '');
    const vdata = genes.slice().sort((a, b) => (sigBH(a) - sigBH(b)));
    V.layer('pts').selectAll('circle').data(vdata).join('circle')
      .attr('cx', (d) => V.x(Math.max(-4, Math.min(4, d.est)))).attr('cy', (d) => V.y(clampY(d.p)))
      .attr('r', (d) => (sigBH(d) ? 4 : 2.6)).attr('fill', (d) => (sigBH(d) ? (d.de ? cat(1) : cat(0)) : muted()))
      .attr('fill-opacity', (d) => (sigBH(d) ? 1 : 0.45)).attr('stroke', (d) => (sigBH(d) ? css('--surface') : 'none')).attr('stroke-width', 1)
      .on('mousemove', (e, d) => tip(e, `<b>${d.id}</b> (${d.de ? 'truly DE' : 'null'})<br>log₂FC est ${d.est.toFixed(2)} (true ${d.lfc.toFixed(2)})<br>p = ${d.p.toExponential(2)} · q = ${d.q.toExponential(2)}`))
      .on('mouseleave', untip);

    // sorted p-values
    const lo = 1e-12, cy = (p) => S.y(Math.max(lo, p));
    S.layer('lines').selectAll('*').remove();
    const ln = (pts, color, dash) => S.layer('lines').append('path').datum(pts).attr('fill', 'none').attr('stroke', color).attr('stroke-width', 2).attr('stroke-dasharray', dash)
      .attr('d', d3.line().x((d) => S.x(d[0])).y((d) => cy(d[1])));
    ln([[1, alpha], [M, alpha]], muted(), '3 3');
    ln([[1, alpha / M], [M, alpha / M]], cat(7), '6 3');
    ln(d3.range(0, 101).map((t) => { const i = Math.pow(M, t / 100); return [i, (alpha * i) / M]; }), ink(), '');
    if (c.k) S.layer('lines').append('line').attr('x1', S.x(c.k)).attr('x2', S.x(c.k)).attr('y1', 0).attr('y2', S.h).attr('stroke', ink()).attr('stroke-width', 1).attr('stroke-dasharray', '2 3');
    if (c.k) S.layer('lines').append('text').attr('x', S.x(c.k) + 5).attr('y', 14).attr('font-size', 11.5).attr('fill', ink()).text(`BH: top ${c.k} genes called`);
    S.layer('pts').selectAll('circle').data(order).join('circle')
      .attr('cx', (d) => S.x(d.rank)).attr('cy', (d) => cy(d.p)).attr('r', 3)
      .attr('fill', (d) => (d.de ? cat(1) : cat(0))).attr('stroke', css('--surface')).attr('stroke-width', 0.6)
      .on('mousemove', (e, d) => tip(e, `<b>${d.id}</b> (${d.de ? 'truly DE' : 'null'})<br>rank ${d.rank} · p = ${d.p.toExponential(2)}<br>BH line at this rank: ${(alpha * d.rank / M).toExponential(2)}<br>q-value = ${d.q.toExponential(2)}`))
      .on('mouseleave', untip);

    const pct = (o) => (o.n ? Math.round((100 * o.fp) / o.n) + '%' : '–');
    const tp = (o) => o.n - o.fp;
    readout.set({
      'truly DE genes': nDE,
      [`raw p < ${alpha}: called (false)`]: `${c.raw.n} (${c.raw.fp})`,
      'Bonferroni: called (false)': `${c.bonf.n} (${c.bonf.fp})`,
      [`BH FDR ${alpha}: called (false)`]: `${c.bh.n} (${c.bh.fp})`,
      'false discovery proportion: raw / BH': `${pct(c.raw)} / ${pct(c.bh)}`,
    });
    S2.set({
      'power (true DE found): raw': nDE ? Math.round((100 * tp(c.raw)) / nDE) + '%' : '–',
      'power: Bonferroni': nDE ? Math.round((100 * tp(c.bonf)) / nDE) + '%' : '–',
      'power: BH': nDE ? Math.round((100 * tp(c.bh)) / nDE) + '%' : '–',
      'BH p-value cut-off': pBH ? pBH.toExponential(2) : 'none',
    });
    fig.caption(`Left: the p-value histogram. Null genes give a <b>flat</b> floor (about ${Math.round(flat)} per bin); truly DE genes pile into the <b>spike near 0</b>. Right: a volcano plot; only genes passing BH at FDR ${alpha} are coloured. Here raw p &lt; ${alpha} calls ${c.raw.n} genes of which <b>${c.raw.fp} are nulls</b>; BH calls ${c.bh.n}, with ${c.bh.fp} nulls among them.`);
  }
  resim();

  section(main, 'What to notice', `<ul>
    <li>Set <b>% truly DE to 0</b>. The histogram is flat, yet raw p &lt; 0.05 still "finds" about 100 genes (5% of 2,000). Every one is false. Bonferroni and BH both call (almost) nothing, which is correct.</li>
    <li>At the default (10% DE, effect 1×), raw p &lt; 0.05 has the highest power but a large share of its hits are nulls. <b>Bonferroni</b> is safe but misses many real genes. <b>BH</b> sits in between: about α of its calls are false, as promised.</li>
    <li>Raise <b>% truly DE</b>. BH calls more genes, because its threshold <i>adapts</i>: the more signal there is, the further along the line the last crossing point moves. Bonferroni's threshold never moves.</li>
    <li>In the sorted plot, the BH line is a straight diagonal on log–log axes. The cut-off is the <b>last</b> crossing, so a gene above the line can still be called if a later gene dips under it (the "step-up" rule).</li>
    <li>Press <b>New experiment</b> a few times. The false-discovery proportion of BH bounces around α: FDR control is a promise about the average, not about each experiment.</li>
  </ul>`);

  steps(main, 'How it works', [
    'Run one test per gene (here a Wald z-test on the log fold change: estimate ÷ standard error) and get m p-values.',
    'Under the null, a p-value is <b>uniform on [0, 1]</b>. That is why the null genes form the flat floor of the histogram, and why α·m of them fall below α.',
    '<b>Bonferroni:</b> call gene i if p<sub>i</sub> &lt; α/m. This bounds the family-wise error rate (the probability of even one false call) at α. With m = 2,000 and α = 0.05 the threshold is 2.5×10⁻⁵.',
    '<b>Benjamini–Hochberg:</b> sort the p-values p<sub>(1)</sub> ≤ … ≤ p<sub>(m)</sub>. Find the largest k with p<sub>(k)</sub> ≤ α·k/m. Call genes 1…k. This bounds the expected <b>false discovery rate</b> (false calls ÷ all calls) at α·m₀/m ≤ α, where m₀ is the number of nulls.',
    'The <b>q-value</b> of a gene (BH-adjusted p, <code>p.adjust(method="BH")</code>, the <code>padj</code> column in DESeq2) is the smallest FDR level at which that gene would be called. "q = 0.03" means: if you call this gene and everything ranked above it, you expect about 3% of that list to be false.',
  ]);
  math(main, 'BH adjusted p:  q_(i) = min over j ≥ i of  m · p_(j) / j\nexpected false calls at raw α ≈ α · m₀        (α = 0.05, m₀ = 1,800  →  90)\nStorey π₀ estimate  = #{p > λ} / ((1 − λ) · m)  (height of the flat floor ÷ m/bins)');

  callout(main, '', 'In biology', `<ul>
    <li><b>Bulk RNA-seq DE</b> (DESeq2, edgeR, limma-voom) reports <code>padj</code> = BH q-values. The standard cut-off "padj &lt; 0.05" is exactly the BH line above.</li>
    <li><b>Look at the p-value histogram</b> before trusting any DE result. A spike at 0 on a flat floor is healthy. A floor that rises toward 1, or a U-shape, signals a mis-specified model (wrong dispersion, ignored batch, pseudoreplication across cells in scRNA-seq).</li>
    <li>Storey's <b>q-value</b> package (and <code>qvalue</code> in Bioconductor) estimates π₀, the fraction of nulls, from the flat floor and is slightly more powerful than BH when many genes are DE.</li>
    <li>The same logic applies to GWAS (where the conventional 5×10⁻⁸ threshold is a Bonferroni correction for ~1 million independent tests), eQTL mapping, peak calling in ChIP-/ATAC-seq, and gene-set enrichment.</li>
  </ul>`);
  callout(main, 'warn', 'Pitfalls', `<ul>
    <li><b>FDR is about the list, not a gene.</b> q = 0.04 does not mean "this gene has a 4% chance of being null". Genes near the cut-off are much more likely to be false than the ones at the top (the local FDR).</li>
    <li><b>Filtering after looking</b> (e.g. testing only genes that looked different in a pilot) breaks the uniform-null assumption. Independent filtering on mean expression, as DESeq2 does, is fine because it ignores the condition labels.</li>
    <li><b>Many tiny effects:</b> with huge n, almost everything becomes significant. Pair q-values with an effect-size threshold (|log₂FC| &gt; 1) or use <code>lfcShrink</code>.</li>
    <li>Treating cells as replicates in scRNA-seq makes p-values far too small, so even BH cannot rescue it. Aggregate to pseudobulk first.</li>
  </ul>`);
  seeAlso(main, [['omics', 'bulk-de', 'Differential expression'], ['omics', 'sc-pseudobulk', 'Pseudobulk DE'], ['concepts', 'metrics', 'Classification metrics'], ['concepts', 'probability-mle', 'Likelihood & Bayes']]);
}
