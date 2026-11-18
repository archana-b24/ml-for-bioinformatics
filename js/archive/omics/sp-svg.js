// Spatial step 2: spatially variable genes with Moran's I and a permutation null, computed live for map genes
// and precomputed for the whole panel (raw, and after removing cell-type means). Data: sp_data.json + sp_svg.json.
import { header, figure, select, button, stats, section, steps, callout, math, legend, seeAlso, el, tip, untip, animate } from '../../lib/ui.js';
import { frame, css, cat, ink2, muted } from '../../lib/plot.js';
import { rng } from '../../lib/data.js';
import { stepCard, omicsData } from '../../omics/_shared.js';
import { tissueMap, exprScale, delaunayEdges, neighbours, lognorm, moran, fmt, secShort } from '../../omics/_spatial.js';
const d3 = window.d3;

const CATS = [
  ['marker', 'cell-type marker'], ['module:IFNG', 'IFN-γ module'], ['module:TGFB_CAF', 'CAF/TGF-β module'], ['module:TLS', 'TLS module'],
  ['module:PROLIF', 'proliferation module'], ['module:EXHAUSTION', 'exhaustion module'], ['filler (cell-type)', 'anonymous, cell-type specific'], ['filler (flat)', 'anonymous, flat'],
];
const catColor = (c) => { const k = CATS.findIndex((q) => q[0] === c); return k < 0 ? muted() : (k === 7 ? muted() : cat(k)); };
const catLabel = (c) => (CATS.find((q) => q[0] === c) || [c, c])[1];
const fq = (q) => (q < 0.001 ? '< 0.001' : q.toFixed(3));

export default async function render(main) {
  header(main, 'Spatially variable genes',
    'A gene is <b>spatially variable</b> if cells near each other have more similar expression than random pairs of cells. Moran\'s I measures exactly that, and a <b>permutation test</b> (shuffle the positions, recompute) says whether the pattern could be chance.');
  const card = stepCard(main, {
    stage: 'Spatial · step 2 of 6',
    question: 'Which genes are organised in space at all, and which of those patterns are more than "this is where the tumour cells are"?',
    data: '…', method: 'Moran\'s I on the Delaunay neighbour graph, with a 199-permutation null and Benjamini–Hochberg FDR, before and after removing cell-type means.',
    verdict: 'stats', why: ' This is a hypothesis test per gene with a resampling null. No model is trained and nothing is predicted.',
  });
  const D = await omicsData(main, 'sp_data.json');
  const SV = await omicsData(main, 'sp_svg.json');
  if (!D || !SV) return;
  const NS = D.sections.length, NG = SV.genes.length, gi = Object.fromEntries(SV.genes.map((g, i) => [g, i]));
  card.querySelectorAll('.step-row')[1].querySelector('span').innerHTML = `The ${NG}-gene panel measured in ${NS} sections (${d3.sum(D.cells, (c) => c.n).toLocaleString()} cells). Per-cell maps are shown for ${D.genes.length} of the genes.`;

  const G = D.cells.map((s) => { const e = delaunayEdges(s.x, s.y, 35); return { e, nb: neighbours(s.n, e) }; });
  const vals = {};
  const getV = (si, g) => (vals[si + g] ||= lognorm(D.cells[si].expr[g], D.cells[si].lib));

  // ------------------------------------------------------------------ figure 1: Moran's I, live
  const fig1 = figure(main, "Moran's I: expression against the average of the neighbours",
    'Left: one gene\'s counts in one section. Right: each dot is a cell; x is its expression (centred), y is the mean expression of its graph neighbours. The slope of the best-fit line <b>is</b> Moran\'s I. Press "Shuffle positions" to see what no spatial pattern looks like. (The live value is recomputed in your browser from coordinates rounded to 1 µm, so it can differ from the build script\'s in the third decimal.)');
  let sec = 0, gene = 'CXCL9', shuffled = false, shufSeed = 1;
  select(fig1.controls, { label: 'Section', value: '0', options: D.sections.map((s, i) => [String(i), s]), onChange: (v) => { sec = +v; shuffled = false; paint1(); resetPerm(); } });
  const gsel = select(fig1.controls, { label: 'Gene', value: gene, options: D.genes.map((g) => [g, `${g} (${catLabel(SV.cat[gi[g]])})`]), onChange: (v) => { gene = v; shuffled = false; paint1(); resetPerm(); } });
  const shufBtn = button(fig1.controls, 'Shuffle positions', () => { shuffled = !shuffled; shufSeed++; paint1(); });
  const rd1 = stats(fig1.root);
  const row1 = el('div', { class: 'fig-row' }); fig1.body.append(row1);
  const mW = el('div'), sW = el('div'); row1.append(mW, sW);
  const M1 = tissueMap(mW, { size: D.size, px: 440 });
  const F1 = frame(sW, { width: 460, height: 420, x: [-1, 3], y: [-1, 2], xLabel: 'expression of the cell (centred log-norm)', yLabel: 'mean of its neighbours' });

  function permuted(v, seed) { const r = rng(seed), p = v.slice(); for (let i = p.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [p[i], p[j]] = [p[j], p[i]]; } return p; }

  function paint1() {
    const s = D.cells[sec], v0 = getV(sec, gene), v = shuffled ? permuted(v0, shufSeed) : v0;
    const counts = shuffled ? permuted(s.expr[gene], shufSeed) : s.expr[gene];
    const sc = exprScale(s.expr[gene]);
    M1.draw(s.x, s.y, (i) => sc.color(counts[i]) || css('--border'), () => 5.5);
    M1.hover((x, y, near) => { const i = near(x, y); return i < 0 ? null : `${gene}: ${counts[i]} counts<br>log-norm ${v[i].toFixed(2)}`; });
    const { I, z, lag } = moran(v, G[sec].nb);
    const xe = d3.extent(z), ye = d3.extent(lag);
    F1.x.domain([xe[0] - 0.1, xe[1] + 0.1]); F1.y.domain([Math.min(ye[0], -0.3) - 0.05, Math.max(ye[1], 0.3) + 0.05]);
    F1.g.select('.axis.x').call(d3.axisBottom(F1.x).ticks(5)); F1.g.select('.axis.y').call(d3.axisLeft(F1.y).ticks(5));
    F1.g.select('.gridline').call(d3.axisLeft(F1.y).ticks(5).tickSize(-F1.w).tickFormat(''));
    const pts = d3.range(z.length).filter((i) => i % 2 === 0);
    F1.layer('pts').selectAll('circle').data(pts).join('circle').attr('cx', (i) => F1.x(z[i])).attr('cy', (i) => F1.y(lag[i])).attr('r', 2.6)
      .attr('fill', cat(0)).attr('fill-opacity', 0.35);
    const xs = F1.x.domain();
    F1.layer('fit').selectAll('line').data([0]).join('line').attr('x1', F1.x(xs[0])).attr('x2', F1.x(xs[1])).attr('y1', F1.y(I * xs[0])).attr('y2', F1.y(I * xs[1]))
      .attr('stroke', css('--ink')).attr('stroke-width', 2);
    F1.layer('lab').selectAll('text').data([0]).join('text').attr('x', F1.w - 4).attr('y', 14).attr('text-anchor', 'end').attr('font-size', 13).attr('fill', ink2())
      .text(`slope = Moran's I = ${I.toFixed(3)}`);
    const r = SV.res[sec], k = gi[gene];
    shufBtn.textContent = shuffled ? 'Restore positions' : 'Shuffle positions';
    rd1.set({ "Moran's I (live)": I.toFixed(3), 'build-script I': r.I[k].toFixed(3), 'z vs permutation null': r.z[k].toFixed(1), 'FDR q': fq(r.q[k]), 'I after removing cell types': `${r.I_res[k].toFixed(3)} (q ${fq(r.q_res[k])})` });
  }
  paint1();

  // ------------------------------------------------------------------ figure 2: the permutation null
  const fig2 = figure(main, 'Is it more than chance? A permutation null',
    `Each shuffle keeps the same expression values and the same graph but assigns values to cells at random, then recomputes I. Repeat ${SV.nperm} times; the empirical p-value is the share of shuffles at least as large as the real I (counting the real one). It can never go below 1/(${SV.nperm}+1), so for ranking ${NG} genes the build also turns the distance from the null into a z-score.`);
  button(fig2.controls, `Run ${SV.nperm} shuffles`, () => runPerm(), true);
  button(fig2.controls, 'Reset', () => resetPerm());
  const rd2 = stats(fig2.root);
  const F2 = frame(fig2.body, { width: 760, height: 260, x: [-0.1, 1], y: [0, 60], xLabel: "Moran's I", yLabel: 'shuffles' });
  let nulls = [], stopP = null;
  function resetPerm() { if (stopP) stopP(); stopP = null; nulls = []; paint2(); }
  function runPerm() {
    resetPerm();
    const v = getV(sec, gene), nb = G[sec].nb, r = rng(777 + sec);
    stopP = animate(() => {
      for (let q = 0; q < 6 && nulls.length < SV.nperm; q++) {
        const p = v.slice(); for (let i = p.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [p[i], p[j]] = [p[j], p[i]]; }
        nulls.push(moran(p, nb).I);
      }
      paint2();
      return nulls.length < SV.nperm;
    });
  }
  function paint2() {
    const I = moran(getV(sec, gene), G[sec].nb).I;
    const lo = Math.min(-0.05, d3.min(nulls) ?? 0), hi = Math.max(I + 0.05, 0.1);
    F2.x.domain([lo, hi]);
    const bins = d3.bin().domain([lo, hi]).thresholds(d3.range(lo, hi, (hi - lo) / 80))(nulls);
    F2.y.domain([0, Math.max(10, d3.max(bins, (b) => b.length) || 0)]);
    F2.g.select('.axis.x').call(d3.axisBottom(F2.x).ticks(8)); F2.g.select('.axis.y').call(d3.axisLeft(F2.y).ticks(4));
    F2.g.select('.gridline').call(d3.axisLeft(F2.y).ticks(4).tickSize(-F2.w).tickFormat(''));
    F2.layer('bars').selectAll('rect').data(bins).join('rect').attr('x', (b) => F2.x(b.x0)).attr('width', (b) => Math.max(0, F2.x(b.x1) - F2.x(b.x0) - 1))
      .attr('y', (b) => F2.y(b.length)).attr('height', (b) => F2.h - F2.y(b.length)).attr('fill', muted());
    F2.layer('obs').selectAll('line').data([I]).join('line').attr('x1', F2.x(I)).attr('x2', F2.x(I)).attr('y1', 0).attr('y2', F2.h).attr('stroke', cat(1)).attr('stroke-width', 2.5);
    F2.layer('obsl').selectAll('text').data([I]).join('text').attr('x', F2.x(I) - 6).attr('y', 14).attr('text-anchor', 'end').attr('font-size', 12).attr('fill', ink2()).text(`real I = ${I.toFixed(3)}`);
    const ge = nulls.filter((u) => u >= I).length;
    rd2.set({ gene: `${gene} in ${secShort(D.sections[sec])}`, shuffles: nulls.length, 'null mean ± sd': nulls.length ? `${d3.mean(nulls).toFixed(4)} ± ${(d3.deviation(nulls) || 0).toFixed(4)}` : '–', 'shuffles ≥ real I': ge, 'empirical p = (≥ + 1)/(n + 1)': nulls.length ? ((ge + 1) / (nulls.length + 1)).toFixed(3) : '–', 'z = (I − mean) / sd': nulls.length > 2 ? ((I - d3.mean(nulls)) / d3.deviation(nulls)).toFixed(1) : '–' });
  }
  paint2();
  runPerm();

  // ------------------------------------------------------------------ figure 3: whole panel, raw vs residual
  const fig3 = figure(main, 'Every panel gene: patterned, or just where the cell types are?',
    `Each dot is one of the ${NG} genes. x: Moran's I on log-normalised expression. y: Moran's I after subtracting each cell type's mean, so only within-cell-type spatial structure is left. Hover for details; click a gene that has a map to load it above.`);
  let sec3 = 0;
  select(fig3.controls, { label: 'Section', value: '0', options: D.sections.map((s, i) => [String(i), s]), onChange: (v) => { sec3 = +v; paint3(); } });
  legend(fig3.controls, CATS.map(([c, l]) => [l, catColor(c)]));
  const rd3 = stats(fig3.root);
  const F3 = frame(fig3.body, { width: 760, height: 440, x: [-0.05, 1], y: [-0.05, 0.6], xLabel: "Moran's I (raw)", yLabel: "Moran's I after removing cell-type means" });
  function paint3() {
    const r = SV.res[sec3];
    F3.x.domain([Math.min(-0.05, d3.min(r.I)), d3.max(r.I) + 0.03]); F3.y.domain([Math.min(-0.05, d3.min(r.I_res)), d3.max(r.I_res) + 0.03]);
    F3.g.select('.axis.x').call(d3.axisBottom(F3.x).ticks(6)); F3.g.select('.axis.y').call(d3.axisLeft(F3.y).ticks(5));
    F3.g.select('.gridline').call(d3.axisLeft(F3.y).ticks(5).tickSize(-F3.w).tickFormat(''));
    const order = d3.range(NG).sort((a, b) => (SV.cat[a] === 'filler (flat)') - (SV.cat[b] === 'filler (flat)')).reverse();
    F3.layer('pts').selectAll('circle').data(order, (k) => k).join('circle').attr('cx', (k) => F3.x(r.I[k])).attr('cy', (k) => F3.y(r.I_res[k]))
      .attr('r', (k) => (D.genes.includes(SV.genes[k]) ? 6 : 4.5)).attr('fill', (k) => catColor(SV.cat[k])).attr('stroke', (k) => (D.genes.includes(SV.genes[k]) ? css('--ink') : css('--surface'))).attr('stroke-width', 1.2)
      .style('cursor', (k) => (D.genes.includes(SV.genes[k]) ? 'pointer' : 'default'))
      .on('mousemove', (e, k) => tip(e, `<b>${SV.genes[k]}</b> · ${catLabel(SV.cat[k])}<br>I = ${r.I[k].toFixed(3)} (z ${r.z[k].toFixed(1)}, q ${fq(r.q[k])})<br>after removing cell types: ${r.I_res[k].toFixed(3)} (z ${r.z_res[k].toFixed(1)}, q ${fq(r.q_res[k])})${D.genes.includes(SV.genes[k]) ? '<br><i>click to map</i>' : ''}`))
      .on('mouseleave', untip)
      .on('click', (e, k) => { const g = SV.genes[k]; if (!D.genes.includes(g)) return; gene = g; gsel.value = g; sec = sec3; fig1.root.querySelector('select').value = String(sec3); shuffled = false; paint1(); resetPerm(); fig1.root.scrollIntoView({ behavior: 'smooth' }); });
    const nRaw = r.q.filter((q) => q < 0.05).length, nRes = r.q_res.filter((q) => q < 0.05).length;
    const med = (c, key) => d3.median(d3.range(NG).filter((k) => SV.cat[k] === c), (k) => r[key][k]).toFixed(3);
    rd3.set({ 'SVGs at q < 0.05 (raw)': `${nRaw} of ${NG}`, 'after removing cell-type means': `${nRes} of ${NG}`, 'median I, cell-type markers': `${med('marker', 'I')} → ${med('marker', 'I_res')}`, 'median I, IFN-γ module': `${med('module:IFNG', 'I')} → ${med('module:IFNG', 'I_res')}`, 'median I, flat genes': `${med('filler (flat)', 'I')} → ${med('filler (flat)', 'I_res')}` });
  }
  paint3();

  // ------------------------------------------------------------------ prose (numbers from the JSON)
  const r0 = SV.res[0], ie = D.phen.indexOf('excluded'), rE = SV.res[ie];
  const cnt = (r, key) => r[key].filter((q) => q < 0.05).length;
  const byCat = (r, c) => SV.genes.map((g, k) => k).filter((k) => SV.cat[k] === c);
  const nFlat = SV.cat.filter((c) => c === 'filler (flat)').length;
  const flatSig = SV.res.map((r) => byCat(r, 'filler (flat)').filter((k) => r.q[k] < 0.05).length);
  const flatK = byCat(null, 'filler (flat)');
  const flatI = SV.res.flatMap((r) => flatK.map((k) => r.I[k]));
  const flatRes = SV.res.map((r) => flatK.filter((k) => r.q_res[k] < 0.05).length);
  const k9 = gi.CXCL9, kE = gi.EPCAM, kT = gi.POSTN;
  section(main, 'What to notice', `<ul>
    <li><b>EPCAM</b> in ${secShort(D.sections[0])} has I = ${r0.I[kE].toFixed(3)}: the tumour nests are big contiguous blobs, so tumour-cell markers are strongly autocorrelated. After removing cell-type means it drops to ${r0.I_res[kE].toFixed(3)}. Its pattern <i>is</i> the cell-type map, nothing more.</li>
    <li><b>CXCL9</b> keeps structure beyond cell type: I = ${r0.I[k9].toFixed(3)} raw and ${r0.I_res[k9].toFixed(3)} after removing cell-type means in ${secShort(D.sections[0])}. Tumour cells and macrophages express it only where CD8 T cells (and their IFN-γ) are nearby, so the same cell type is high in some places and low in others. That is a spatially variable <i>programme</i>, not just composition.</li>
    <li><b>POSTN</b> (CAF/TGF-β module) in ${secShort(D.sections[ie])}: I = ${rE.I[kT].toFixed(3)} raw, ${rE.I_res[kT].toFixed(3)} after removing cell types, because fibroblasts at the tumour margin are TGF-β-activated and fibroblasts deeper in stroma are less so.</li>
    <li>In the panel scatter, ${cnt(r0, 'q')} of ${NG} genes are "spatially variable" at q < 0.05 in ${secShort(D.sections[0])}, but only ${cnt(r0, 'q_res')} after removing cell-type means, and those are almost all IFN-γ and TLS programme genes. In ${secShort(D.sections[ie])} the survivors include <b>POSTN</b>; in ${secShort(D.sections[D.phen.indexOf('desert')])}, ${cnt(SV.res[D.phen.indexOf('desert')], 'q_res')}.</li>
    <li>A surprise worth understanding: the grey <b>"flat"</b> genes differ less than 1.6-fold between cell types in the reference, yet ${flatSig.join(', ')} of the ${nFlat} pass q < 0.05 raw in the ${NS} sections (their I is mostly small: median ${d3.median(flatI).toFixed(3)}, though up to ${d3.max(flatI).toFixed(3)}). Two things leak cell-type structure into them: those small fold differences, and <b>closure</b> (counts are normalised to the cell's panel total, which is dominated by cell-type markers, so a "flat" gene's share still shifts with cell type). With ${D.cells[0].n.toLocaleString()} cells, a little is significant. After removing cell-type means, ${flatRes.join(', ')} remain.</li>
    <li>Press <b>Shuffle positions</b>: the slope collapses to about zero. The permutation null in the middle figure is centred near 0 (exactly −1/(n−1) in expectation) with a tiny spread, because there are thousands of cells.</li>
  </ul>`);

  steps(main, 'How it works', [
    'Build the neighbour graph (<a href="#/archive/omics/sp-data">previous step</a>) and row-normalise it into <code>W</code>.',
    'For each gene, centre the normalised expression: <code>z = x − mean(x)</code>.',
    'Compute each cell\'s neighbour average <code>Wz</code>. Moran\'s I is the regression slope of <code>Wz</code> on <code>z</code>: near +1 means neighbours agree, 0 means no spatial structure, negative means neighbours disagree (a checkerboard).',
    `Shuffle expression across cells (keeping the graph) ${SV.nperm} times to get a null distribution; p = (number of shuffles ≥ real I + 1) / (${SV.nperm} + 1). Because that empirical p floors at 1/(${SV.nperm}+1), the build converts the distance from the null into a z-score and a normal p-value (as Squidpy's <code>pval_z_sim</code>), then controls the false discovery rate over the ${NG} genes with Benjamini–Hochberg.`,
    'Ask the follow-up question: subtract cell-type means (or regress out cell-type proportions on spot data) and test again, to separate "patterned programme" from "patterned composition".',
  ]);
  math(main, `I = (Σ_i z_i · (Wz)_i) / (Σ_i z_i²)        W row-normalised, so I is the slope of Wz on z\nE[I] under no pattern = −1/(n−1)\np = (#{I_perm ≥ I_obs} + 1) / (${SV.nperm} + 1)   → smallest possible p = ${(1 / (SV.nperm + 1)).toFixed(3)}`);

  callout(main, '', 'In biology', `<ul>
    <li><b>Squidpy</b> (<code>sq.gr.spatial_autocorr</code>) computes Moran's I and Geary's C with permutation or analytic p-values; Seurat has <code>FindSpatiallyVariableFeatures</code>.</li>
    <li><b>SpatialDE</b> fits a Gaussian-process model per gene and tests a spatial kernel against noise; <b>SPARK</b> and <b>SPARK-X</b> are faster, better-calibrated alternatives (SPARK-X scales to millions of cells); <b>nnSVG</b> uses nearest-neighbour Gaussian processes.</li>
    <li>On Visium, genes like collagens, immunoglobulins and tumour markers top every SVG list because tissue regions differ in composition; methods like <b>C-SIDE</b> (spacexr) test spatial patterns within a cell type after deconvolution.</li>
  </ul>`);
  callout(main, 'warn', 'Pitfalls', `<ul>
    <li><b>SVG ≠ interesting.</b> Most top hits are cell-type markers tracing tissue architecture. Here ${cnt(r0, 'q')} hits shrink to ${cnt(r0, 'q_res')} once composition is removed.</li>
    <li><b>Huge n makes tiny I significant.</b> With thousands of cells, I = 0.02 can reach the minimum p-value. Rank by effect size (I), not by p.</li>
    <li><b>The permutation p-value has a floor</b> of 1/(${SV.nperm}+1) = ${(1 / (SV.nperm + 1)).toFixed(3)}; many genes tie there, so do more permutations or use the analytic approximation when you need to rank.</li>
    <li>The answer depends on the graph and the normalisation. Library size is itself spatially patterned (big tumour cells, small lymphocytes), so unnormalised counts make almost every gene "spatially variable".</li>
    <li>Each section is tested separately; a gene patterned in one section and not another is a statement about two patients, not a population.</li>
  </ul>`);
  seeAlso(main, [['omics', 'sp-data', 'Spatial data & neighbourhoods'], ['omics', 'sp-domains', 'Spatial domains'], ['concepts', 'multiple-testing', 'Multiple testing'], ['omics', 'bulk-de', 'Differential expression']]);
}
