// Spatial step 4: unmixing Visium-like spots with the single-cell reference. NNLS vs a Poisson model with a gene-wise
// platform factor (cell2location/RCTD-like, simplified), with and without converting RNA share to cell share.
// Precomputed in scripts/build_sc_spatial.py -> data/omics/sp_deconvolution.json.
import { header, figure, select, checkbox, stats, section, steps, callout, math, legend, seeAlso, el, tip, untip } from '../lib/ui.js';
import { frame, css, cat, ink2, muted, diverging } from '../lib/plot.js';
import { stepCard, omicsData } from './_shared.js';
import { tissueMap, ctColor, secShort, pct } from './_spatial.js';
const d3 = window.d3;

const METHODS = [['nnls', 'NNLS on marker genes'], ['poisson', 'Poisson model + platform factor']];

export default async function render(main) {
  header(main, 'Spot deconvolution',
    'A 55 µm spot holds several cells, so its expression is a <b>mixture</b>. If we know what each cell type looks like (from single-cell data), we can solve for the mixing proportions: the same idea as <a href="#/omics/bulk-deconvolution">bulk deconvolution</a>, run once per spot.');
  const card = stepCard(main, {
    stage: 'Spatial · step 4 of 6',
    question: 'On whole-transcriptome spot data, how many CD8 T cells, tumour cells and fibroblasts are in each spot?',
    data: '…', method: 'Reference profiles per cell type from the annotated single-cell data. Per spot, fit expression ≈ Σ proportion × profile by non-negative least squares, or by a Poisson likelihood that also learns a per-gene platform factor.',
    verdict: 'both', why: ' The model is a constrained regression (statistics). Fitting it for hundreds of spots and thousands of genes, with learned nuisance factors, is done with ML-style optimisation, and the reference itself comes from ML label transfer.',
  });
  const V = await omicsData(main, 'sp_deconvolution.json');
  if (!V) return;
  const CT = V.cell_types, NT = CT.length, NS = V.sections.length;
  card.querySelectorAll('.step-row')[1].querySelector('span').innerHTML = `${V.n_spots} simulated Visium-like spots (${V.spot_diam} µm, ${V.pitch} µm pitch) over the ${NS} sections, ${V.cells_per_spot.min}–${V.cells_per_spot.max} cells each, with gene-wise capture differences versus the single-cell platform. The fit uses ${V.n_genes} cell-type marker genes (cell-state programme genes such as the IFN-γ module are excluded because they differ between reference and tissue). Truth = the cells actually under each spot.`;

  let meth = 'poisson', size = true, ct = 1, sec = 0;
  const key = () => `${meth}_${size ? 1 : 0}`;

  // ------------------------------------------------------------------ figure 1: maps
  const fig1 = figure(main, 'Estimated versus true composition, spot by spot',
    'Each disc is a spot, shaded by the fraction of the chosen cell type. Left: the estimate. Right: the truth (counted cells). Hover a spot for its full composition.');
  select(fig1.controls, { label: 'Method', value: meth, options: METHODS, onChange: (v) => { meth = v; paintAll(); } });
  checkbox(fig1.controls, { label: 'convert RNA share → cell share', checked: size, onChange: (v) => { size = v; paintAll(); } });
  select(fig1.controls, { label: 'Cell type', value: String(ct), options: CT.map((c, i) => [String(i), c]), onChange: (v) => { ct = +v; paintAll(); } });
  select(fig1.controls, { label: 'Section', value: '0', options: V.sections.map((s, i) => [String(i), s]), onChange: (v) => { sec = +v; paintAll(); } });
  const leg1 = el('div'); fig1.controls.append(leg1);
  const rd1 = stats(fig1.root);
  const row1 = el('div', { class: 'fig-row' }); fig1.body.append(row1);
  const aW = el('div'), bW = el('div'); row1.append(aW, bW);
  const MA = tissueMap(aW, { size: 1000, px: 420, label: 'Estimated fraction' });
  const MB = tissueMap(bW, { size: 1000, px: 420, label: 'True fraction (cells under the spot)' });

  // ------------------------------------------------------------------ figure 2: scatter + bias
  const fig2 = figure(main, 'Accuracy across all spots, and systematic bias',
    `Left: every spot in all ${NS} sections, estimated against true fraction for the chosen cell type (diagonal = perfect). Right: mean error (estimate − truth) per cell type for the current method; bars right of zero mean the type is over-called.`);
  const rd2 = stats(fig2.root);
  const row2 = el('div', { class: 'fig-row' }); fig2.body.append(row2);
  const sW = el('div'), biasW = el('div'); row2.append(sW, biasW);
  const FS = frame(sW, { width: 420, height: 380, x: [0, 1], y: [0, 1], xLabel: 'true fraction', yLabel: 'estimated fraction' });
  const FB = frame(biasW, { width: 460, height: 380, x: [-0.1, 0.1], y: [0, 1], xLabel: 'mean bias (estimate − truth)', margin: { left: 92 }, grid: false, yScale: d3.scaleBand().padding(0.25) });
  FB.y.domain(CT);
  FB.g.select('.axis.y').call(d3.axisLeft(FB.y));

  function spotColor(f, mx) { return d3.interpolateRgb(css('--surface-2'), ctColor(ct))(Math.min(1, f / mx)); }
  function paintMaps() {
    const E = V.est[key()], ids = d3.range(V.n_spots).filter((j) => V.sec[j] === sec);
    const mx = Math.max(0.05, d3.max(ids, (j) => Math.max(V.true[j][ct], E[j][ct])));
    leg1.innerHTML = '';
    legend(leg1, [[`${CT[ct]} 0%`, spotColor(0, mx)], [pct(mx / 2), spotColor(mx / 2, mx)], [pct(mx), spotColor(mx, mx)]]);
    const ov = (src) => (ctx, k) => ids.forEach((j) => {
      ctx.fillStyle = spotColor(src[j][ct], mx); ctx.beginPath(); ctx.arc(V.x[j] * k, V.y[j] * k, (V.spot_diam / 2) * k, 0, 6.2832); ctx.fill();
      ctx.strokeStyle = css('--border'); ctx.lineWidth = 1; ctx.stroke();
    });
    MA.overlay(ov(E)); MB.overlay(ov(V.true));
    MA.draw([], [], () => null); MB.draw([], [], () => null);
    const hv = (x, y) => {
      const j = ids.find((q) => (V.x[q] - x) ** 2 + (V.y[q] - y) ** 2 < (V.spot_diam / 2) ** 2);
      if (j == null) return null;
      const rows = d3.range(NT).filter((t) => V.true[j][t] > 0 || E[j][t] >= 0.02).map((t) => `<tr><td>${CT[t]}</td><td style="text-align:right;padding-left:10px">${pct(V.true[j][t])}</td><td style="text-align:right;padding-left:10px">${pct(E[j][t])}</td></tr>`).join('');
      return `<b>Spot (${V.x[j]}, ${V.y[j]})</b>, ${V.n_cells[j]} cells<table><tr><td></td><td style="padding-left:10px"><i>true</i></td><td style="padding-left:10px"><i>est.</i></td></tr>${rows}</table>`;
    };
    MA.hover(hv); MB.hover(hv);
    const err = ids.map((j) => E[j][ct] - V.true[j][ct]);
    rd1.set({ [`spots in ${secShort(V.sections[sec])}`]: ids.length, [`${CT[ct]}: mean true`]: pct(d3.mean(ids, (j) => V.true[j][ct]), 1), [`${CT[ct]}: mean estimated`]: pct(d3.mean(ids, (j) => E[j][ct]), 1), 'mean abs. error in this section': pct(d3.mean(err, Math.abs), 1) });
  }
  function paintAcc() {
    const E = V.est[key()], m = V.metrics[key()];
    const mx = Math.max(0.1, d3.max(V.true, (r) => r[ct]), d3.max(E, (r) => r[ct])) * 1.05;
    FS.x.domain([0, mx]); FS.y.domain([0, mx]);
    FS.g.select('.axis.x').call(d3.axisBottom(FS.x).ticks(5).tickFormat(d3.format('.0%'))); FS.g.select('.axis.y').call(d3.axisLeft(FS.y).ticks(5).tickFormat(d3.format('.0%')));
    FS.g.select('.gridline').call(d3.axisLeft(FS.y).ticks(5).tickSize(-FS.w).tickFormat(''));
    FS.layer('diag').selectAll('line').data([0]).join('line').attr('x1', FS.x(0)).attr('y1', FS.y(0)).attr('x2', FS.x(mx)).attr('y2', FS.y(mx)).attr('stroke', muted()).attr('stroke-dasharray', '4 3');
    FS.layer('pts').selectAll('circle').data(d3.range(V.n_spots)).join('circle').attr('cx', (j) => FS.x(V.true[j][ct])).attr('cy', (j) => FS.y(E[j][ct])).attr('r', 4)
      .attr('fill', ctColor(ct)).attr('fill-opacity', 0.6).attr('stroke', css('--surface')).attr('stroke-width', 1)
      .on('mousemove', (e, j) => tip(e, `${V.sections[V.sec[j]]} · spot (${V.x[j]}, ${V.y[j]})<br>${CT[ct]}: true ${pct(V.true[j][ct])}, est. ${pct(E[j][ct])}`)).on('mouseleave', untip);
    const bm = Math.max(0.04, d3.max(Object.values(V.metrics), (q) => d3.max(q.bias, Math.abs))) * 1.1;
    FB.x.domain([-bm, bm]); FB.g.select('.axis.x').call(d3.axisBottom(FB.x).ticks(5).tickFormat(d3.format('+.0%')));
    const dv = diverging(bm);
    FB.layer('zero').selectAll('line').data([0]).join('line').attr('x1', FB.x(0)).attr('x2', FB.x(0)).attr('y1', 0).attr('y2', FB.h).attr('stroke', css('--axis'));
    FB.layer('bars').selectAll('rect').data(m.bias).join('rect').attr('y', (d, t) => FB.y(CT[t])).attr('height', FB.y.bandwidth())
      .attr('x', (d) => FB.x(Math.min(0, d))).attr('width', (d) => Math.abs(FB.x(d) - FB.x(0))).attr('fill', (d) => dv(d)).attr('stroke', (d, t) => (t === ct ? css('--ink') : 'none')).attr('stroke-width', 1.5)
      .on('mousemove', (e, d) => tip(e, `${CT[m.bias.indexOf(d)]}: bias ${(d * 100).toFixed(1)} points, r = ${m.r[m.bias.indexOf(d)].toFixed(2)}`)).on('mouseleave', untip);
    FB.layer('lab').selectAll('text').data(m.bias).join('text').attr('y', (d, t) => FB.y(CT[t]) + FB.y.bandwidth() / 2 + 4)
      .attr('x', (d) => FB.x(d) + (d >= 0 ? 4 : -4)).attr('text-anchor', (d) => (d >= 0 ? 'start' : 'end')).attr('font-size', 11).attr('fill', ink2()).text((d) => { const r = (d * 100).toFixed(1); return +r === 0 ? '0.0' : (d > 0 ? '+' : '−') + Math.abs(+r).toFixed(1); });
    rd2.set({ method: `${METHODS.find((q) => q[0] === meth)[1].split(' ')[0]}${size ? ', cell share' : ', RNA share'}`, 'RMSE (all types, all spots)': m.rmse.toFixed(4), [`r, ${CT[ct]}`]: m.r[ct].toFixed(3), [`bias, ${CT[ct]}`]: `${(m.bias[ct] * 100).toFixed(1).replace('-', '−')} points` });
  }
  function paintAll() { paintMaps(); paintAcc(); }
  paintAll();

  // ------------------------------------------------------------------ prose (numbers from the JSON)
  const Mx = V.metrics, rl = V.ref_lib;
  const sgn = (v) => { const r = (v * 100).toFixed(1); return +r === 0 ? '0.0' : (v > 0 ? '+' : '−') + Math.abs(+r).toFixed(1); };
  const bpt = (k, t) => sgn(Mx[k].bias[t]);
  const regMean = (si, r) => { const ids = d3.range(V.n_spots).filter((j) => V.sec[j] === si && V.region[j] === r); return ids.length ? d3.mean(ids, (j) => V.est.poisson_1[j][1]) : 0; };
  const iE = V.sections.findIndex((q) => q.includes('excluded'));
  const order = Object.keys(Mx).sort((a, b) => Mx[a].rmse - Mx[b].rmse);
  const lab = (k) => `${k.startsWith('nnls') ? 'NNLS' : 'Poisson'}, ${k.endsWith('1') ? 'cell share' : 'RNA share'}`;
  const fibI = CT.indexOf('Fibroblast'), endI = CT.indexOf('Endothelial');
  section(main, 'What to notice', `<ul>
    <li><b>RNA share is not cell share.</b> With the Poisson model and the box unticked, the mean error is ${bpt('poisson_0', 0)} points for tumour cells (over-called) and ${bpt('poisson_0', 1)} for CD8 T cells (under-called). In the reference, a tumour cell carries ${rl[0].toFixed(2)}× the average RNA and a CD8 T cell ${rl[1].toFixed(2)}×, and a spot's signal is weighted by RNA. Tick the box (divide by RNA per cell and renormalise): the biases shrink to ${bpt('poisson_1', 0)} and ${bpt('poisson_1', 1)}, and RMSE falls from ${Mx.poisson_0.rmse.toFixed(4)} to ${Mx.poisson_1.rmse.toFixed(4)}.</li>
    <li><b>NNLS ignores platform effects.</b> Spot data capture some genes better than single-cell data. NNLS fits the reference as is, so its errors are structured (fibroblasts ${bpt('nnls_0', fibI)}, endothelial ${bpt('nnls_0', endI)} points). The Poisson model learns a per-gene factor and absorbs most of it.</li>
    <li>Ticking the box <i>hurts</i> NNLS (RMSE ${Mx.nnls_0.rmse.toFixed(4)} → ${Mx.nnls_1.rmse.toFixed(4)}, tumour bias ${bpt('nnls_1', 0)}): a correction that is right in principle cannot rescue a model that is biased for other reasons. Ranking all four by RMSE: ${order.map((k) => `${lab(k)} ${Mx[k].rmse.toFixed(4)}`).join(' < ')}.</li>
    <li>Pick <b>CD8 T</b> and compare ${secShort(V.sections[0])} with ${secShort(V.sections[iE])}. With the best method, spots centred on tumour nests are on average ${pct(regMean(0, 0))} CD8 T in ${secShort(V.sections[0])} but ${pct(regMean(iE, 0))} in ${secShort(V.sections[iE])}, while margin spots are ${pct(regMean(0, 1))} vs ${pct(regMean(iE, 1))}. Deconvolution recovers the exclusion pattern, but only at spot resolution: it cannot say whether a CD8 T cell inside a "50% tumour" spot is touching a tumour cell.</li>
    <li>Correlations per type are high even for rare cells (CD8 T r = ${Mx[order[0]].r[1].toFixed(2)} with the best method), but rare types with small fractions have the largest relative errors. Check the scatter near zero.</li>
  </ul>`);

  steps(main, 'How it works', [
    'Build a <b>reference</b>: the average expression profile of each annotated cell type in the single-cell data (from the <a href="#/omics/sc-annotation">annotation step</a>), normalised to sum to 1. Note the RNA content per cell of each type.',
    'Choose informative genes: markers that are specific to one type. Leave out genes driven by cell <i>state</i> (IFN-γ response, proliferation), which differ between the reference patients and the tissue.',
    '<b>NNLS:</b> for each spot, find non-negative weights w with spot ≈ Σ<sub>t</sub> w<sub>t</sub> · profile<sub>t</sub> by least squares.',
    '<b>Poisson model:</b> treat counts as Poisson with mean Σ<sub>t</sub> w<sub>t</sub> · profile<sub>tg</sub> · m<sub>g</sub>, where m<sub>g</sub> is a learned per-gene platform factor shared by all spots; fit w and m by multiplicative (EM-style) updates.',
    'Weights estimate each type\'s share of the spot\'s <b>RNA</b>. Divide by RNA per cell and renormalise to get the share of <b>cells</b>.',
  ]);
  math(main, 'y_sg ~ Poisson( Σ_t w_st · r_tg · m_g )        r_tg = reference profile, m_g = platform factor\nRNA share_st = w_st / Σ_t w_st\ncell share_st ∝ RNA share_st / (RNA per cell)_t');

  callout(main, '', 'In biology', `<ul>
    <li><b>cell2location</b> (a Bayesian negative-binomial model with gene-wise technical effects, fitted by variational inference), <b>RCTD</b> (spacexr; Poisson-lognormal with platform effects), <b>Stereoscope</b>, <b>CARD</b> (adds a spatial smoothness prior), <b>SPOTlight</b> (NMF) and <b>Tangram</b> (maps single cells onto spots by optimising a similarity objective).</li>
    <li>Benchmarks (e.g. Li et al. 2022, Nature Methods) find cell2location and RCTD among the most accurate, and all methods degrade when the reference misses a cell type present in the tissue.</li>
    <li>Visium HD and imaging platforms reduce the need for deconvolution, but 55 µm Visium remains the most common whole-transcriptome spatial data in published tumour cohorts.</li>
  </ul>`);
  callout(main, 'warn', 'Pitfalls', `<ul>
    <li><b>Missing reference types</b> get absorbed into the closest available type; a tumour subclone absent from the single-cell data becomes "fibroblast" or "macrophage" with no warning.</li>
    <li><b>Proportions of RNA vs cells:</b> report which one you mean. Large cells (tumour, macrophages) dominate RNA share.</li>
    <li><b>Platform and patient effects</b> between reference and spots bias simple least squares; use a method that models them.</li>
    <li>The truth here is known because the data are simulated. In real data, validate against matched imaging or histology, never against another deconvolution method.</li>
  </ul>`);
  seeAlso(main, [['omics', 'bulk-deconvolution', 'Bulk cell-type deconvolution'], ['omics', 'sc-annotation', 'Cell-type annotation'], ['omics', 'sp-data', 'Spots vs single cells'], ['concepts', 'linear-regression', 'Linear regression']]);
}
