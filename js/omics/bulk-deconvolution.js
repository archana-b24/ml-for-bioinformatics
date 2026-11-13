// Bulk step 10: cell-type deconvolution by non-negative least squares against reference profiles,
// with the ways it fails (wrong scale, missing cell type, non-specific genes).
import { header, figure, select, stats, section, steps, callout, math, legend, seeAlso, tip, untip, el } from '../lib/ui.js';
import { frame, cat, css, ink2, muted } from '../lib/plot.js';
import { stepCard, omicsData } from './_shared.js';
const d3 = window.d3;
const f2 = (v) => v.toFixed(2), f3 = (v) => v.toFixed(3), pc = (v) => (100 * v).toFixed(1) + '%';

export default async function render(main) {
  header(main, 'Cell-type deconvolution',
    'A bulk sample is a smoothie: its expression is roughly the sum of each cell type\'s profile, weighted by how much of that cell type is in the tumour. Given reference profiles, <b>non-negative least squares</b> unmixes the smoothie back into cell-type fractions.');
  const D = await omicsData(main, 'bulk_deconv.json');
  if (!D) return;
  const CT = D.cell_types, n = D.true.length, K = CT.length, SC = D.scen;
  const cd8 = CT.indexOf('CD8 T'), fib = CT.indexOf('Fibroblast');

  stepCard(main, {
    stage: 'Bulk RNA-seq · step 10 of 11',
    question: 'Is the interferon signal in responders explained by more CD8 T cells in the sample, and how much of each tumour is stroma?',
    data: `${n} discovery bulk profiles (linear CPM) and reference expression profiles for ${K} cell types (in real life: sorted cells or a single-cell atlas).`,
    method: 'Non-negative least squares (NNLS) per sample on ' + D.n_markers + ' marker genes, fractions renormalised to sum to 1; compared with the simulated true fractions.',
    verdict: 'stats',
    why: 'This is a constrained linear regression with a known mixing model, not learning from labels. The ML-flavoured tools (CIBERSORT\'s ν-SVR) are robust regressions on the same model.',
  });

  // ---------------- Figure 1
  const f1g = figure(main, 'True vs estimated fractions',
    'Left: each dot is a patient, simulated true fraction (x) vs NNLS estimate (y) for one cell type; dashed = perfect agreement. Right: correlation between true and estimated fraction for every cell type in the chosen scenario (click a bar to show that cell type).');
  let sk = 'markers', ci = cd8, colBy = 'centre';
  select(f1g.controls, { label: 'Scenario', value: sk, options: Object.entries(SC).map(([k, v]) => [k, v.label]), onChange: (v) => { sk = v; draw2(); } });
  const ctSel = select(f1g.controls, { label: 'Cell type', value: String(ci), options: CT.map((c, i) => [String(i), c]), onChange: (v) => { ci = +v; draw1(); } });
  select(f1g.controls, { label: 'Colour by', value: colBy, options: [['centre', 'sequencing centre'], ['pheno', 'immune phenotype']], onChange: (v) => { colBy = v; draw1(); } });
  const leg = el('div'); f1g.root.append(leg);
  const s1 = stats(f1g.root);
  const row = el('div', { class: 'fig-row' }); f1g.body.append(row);
  const L = el('div'), R = el('div'); row.append(L, R);
  const S = frame(L, { width: 440, height: 400, x: [0, 1], y: [0, 1], xLabel: 'true fraction', yLabel: 'estimated fraction' });
  const B = frame(R, { width: 440, height: 400, x: [-0.2, 1.18], y: [0, K], xLabel: 'Pearson r (true vs estimated)', grid: false, margin: { left: 86 } });
  B.g.select('.axis.y').remove();
  const PH = ['inflamed', 'excluded', 'desert'];
  let onScen = () => {};
  function draw1() {
    const E = SC[sk].est;
    const xs = D.true.map((r) => r[ci]), ys = E.map((r) => r[ci]);
    const m = Math.min(1, Math.max(0.05, d3.max([...xs, ...ys]) * 1.1));
    S.x.domain([0, m]); S.y.domain([0, m]);
    S.g.select('.axis.x').call(d3.axisBottom(S.x).ticks(5, '.0%')); S.g.select('.axis.y').call(d3.axisLeft(S.y).ticks(5, '.0%'));
    S.g.select('.gridline').call(d3.axisLeft(S.y).ticks(5).tickSize(-S.w).tickFormat(''));
    S.layer('diag').selectAll('line').data([0]).join('line').attr('x1', S.x(0)).attr('y1', S.y(0)).attr('x2', S.x(m)).attr('y2', S.y(m)).attr('stroke', muted()).attr('stroke-dasharray', '4 4');
    const colour = (i) => (colBy === 'centre' ? cat(D.centre[i] === 'Centre A' ? 0 : 1) : cat(2 + PH.indexOf(D.pheno[i])));
    S.layer('p').selectAll('circle').data(d3.range(n)).join('circle').attr('cx', (i) => S.x(xs[i])).attr('cy', (i) => S.y(ys[i])).attr('r', 4.5)
      .attr('fill', colour).attr('stroke', css('--surface')).attr('stroke-width', 1)
      .on('mousemove', (e, i) => tip(e, `${D.ids[i]} · ${D.centre[i]} · ${D.pheno[i]}<br>${CT[ci]}: true ${pc(xs[i])}, estimated ${pc(ys[i])}`)).on('mouseleave', untip);
    leg.innerHTML = '';
    legend(leg, colBy === 'centre' ? [['Centre A', cat(0)], ['Centre B', cat(1)]] : PH.map((p, j) => [p, cat(2 + j)]));
    const r = SC[sk].r[ci], bA = SC[sk].bias['Centre A'][ci], bB = SC[sk].bias['Centre B'][ci];
    s1.set({ 'cell type': CT[ci], 'Pearson r': r == null ? '– (fixed at 0)' : f3(r), 'RMSE': pc(SC[sk].rmse[ci]),
      'mean bias, Centre A': (bA >= 0 ? '+' : '') + pc(bA), 'mean bias, Centre B': (bB >= 0 ? '+' : '') + pc(bB) });
    B.layer('t').selectAll('text').attr('font-weight', (d, k) => (k === ci ? 700 : 400));
  }
  function draw2() {
    const r = SC[sk].r;
    const bh = B.h / K;
    B.layer('z').selectAll('line').data([0]).join('line').attr('x1', B.x(0)).attr('x2', B.x(0)).attr('y1', 0).attr('y2', B.h).attr('stroke', muted());
    B.layer('b').selectAll('rect').data(CT).join('rect').attr('x', (c, k) => B.x(Math.min(0, r[k] ?? 0))).attr('y', (c, k) => k * bh + 4).attr('height', bh - 8).attr('rx', 2)
      .attr('width', (c, k) => Math.abs(B.x(r[k] ?? 0) - B.x(0))).attr('fill', (c, k) => cat(k)).style('cursor', 'pointer')
      .on('click', (e, c) => { ci = CT.indexOf(c); ctSel.value = String(ci); draw1(); })
      .on('mousemove', (e, c) => { const k = CT.indexOf(c); tip(e, `<b>${c}</b><br>r = ${r[k] == null ? '–' : f3(r[k])}<br>RMSE ${pc(SC[sk].rmse[k])}`); }).on('mouseleave', untip);
    B.layer('v').selectAll('text').data(CT).join('text').attr('x', (c, k) => B.x(Math.max(0, r[k] ?? 0)) + 4).attr('y', (c, k) => k * bh + bh / 2 + 4).attr('font-size', 11).attr('fill', ink2())
      .text((c, k) => (r[k] == null ? 'absent from reference' : f2(r[k])));
    B.layer('t').selectAll('text').data(CT).join('text').attr('x', -6).attr('y', (c, k) => k * bh + bh / 2 + 4).attr('text-anchor', 'end').attr('font-size', 12).attr('fill', ink2()).text((c) => c);
    draw1();
    onScen();
  }
  draw2();

  // ---------------- Figure 2: stacked compositions
  const f2g = figure(main, 'Patient compositions: truth above, estimate below',
    'Twelve patients from each immune phenotype. Each column is one tumour; the upper bar is the simulated truth, the lower bar the estimate from the scenario chosen above. Hover for numbers.');
  legend(f2g.root, CT.map((c, k) => [c, cat(k)]));
  const pick = PH.flatMap((p) => d3.range(n).filter((i) => D.pheno[i] === p).slice(0, 12));
  const box2 = el('div'); f2g.body.append(box2);
  const W = 900, Hh = 350;
  const svg = d3.select(box2).append('svg').attr('viewBox', `0 0 ${W} ${Hh}`).attr('width', W);
  const x0 = 70, cw = (W - x0 - 10) / pick.length, bH = 120, yT = 24, yE = yT + bH + 40;
  svg.append('text').attr('x', x0 - 8).attr('y', yT + bH / 2 + 4).attr('text-anchor', 'end').attr('font-size', 12).attr('fill', ink2()).text('true');
  svg.append('text').attr('x', x0 - 8).attr('y', yE + bH / 2 + 4).attr('text-anchor', 'end').attr('font-size', 12).attr('fill', ink2()).text('estimated');
  PH.forEach((p, j) => svg.append('text').attr('x', x0 + (j * 12 + 6) * cw).attr('y', 14).attr('text-anchor', 'middle').attr('font-size', 12).attr('fill', ink2()).text(p));
  const gT = svg.append('g'), gE = svg.append('g');
  function stack(g, M, y0) {
    const rects = pick.flatMap((i, c) => { let acc = 0; return CT.map((ct, k) => { const r = { i, c, k, v: M[i][k], a: acc }; acc += M[i][k]; return r; }); });
    g.selectAll('rect').data(rects).join('rect').attr('x', (d) => x0 + d.c * cw + 1).attr('width', cw - 2).attr('y', (d) => y0 + d.a * bH).attr('height', (d) => Math.max(0, d.v * bH))
      .attr('fill', (d) => cat(d.k)).on('mousemove', (e, d) => tip(e, `${D.ids[d.i]} (${D.pheno[d.i]}, ${D.resp[d.i] ? 'responder' : 'non-responder'})<br>${CT.map((ct, k) => `${ct}: true ${pc(D.true[d.i][k])} · est ${pc(SC[sk].est[d.i][k])}`).join('<br>')}`))
      .on('mouseleave', untip);
  }
  stack(gT, D.true, yT);
  onScen = () => stack(gE, SC[sk].est, yE);
  onScen();
  svg.append('text').attr('x', x0).attr('y', Hh - 4).attr('font-size', 11).attr('fill', muted()).text('● under a column = responder');
  svg.selectAll('circle.resp').data(pick.filter((i) => D.resp[i])).join('circle').attr('cx', (i) => x0 + pick.indexOf(i) * cw + cw / 2).attr('cy', yE + bH + 12).attr('r', 4).attr('fill', ink2());

  // ---------------- Figure 3: bulk CD8 cannot see location
  const f3g = figure(main, 'More CD8 T cells is not the same as T cells inside the tumour',
    'Estimated CD8 T-cell fraction per patient by immune phenotype (marker-gene NNLS); filled dots are responders. Excluded tumours contain T cells, but trapped in stroma: bulk deconvolution cannot see where the cells are.');
  const box3 = el('div'); f3g.body.append(box3);
  const est = SC.markers.est.map((r) => r[cd8]);
  const Fz = frame(box3, { width: 900, height: 300, x: [-0.5, 2.5], y: [0, d3.max(est) * 1.1], yLabel: 'estimated CD8 T fraction', xTicks: 3 });
  Fz.g.select('.axis.x').call(d3.axisBottom(Fz.x).tickValues([0, 1, 2]).tickFormat((i) => PH[i]));
  Fz.g.select('.axis.y').call(d3.axisLeft(Fz.y).ticks(5, '.0%'));
  const jr = d3.range(n).map((i) => ((i * 0.6180339887) % 1) - 0.5);
  Fz.layer('p').selectAll('circle').data(d3.range(n)).join('circle').attr('cx', (i) => Fz.x(PH.indexOf(D.pheno[i]) + jr[i] * 0.5)).attr('cy', (i) => Fz.y(est[i])).attr('r', 4.5)
    .attr('fill', (i) => (D.resp[i] ? cat(2 + PH.indexOf(D.pheno[i])) : css('--surface'))).attr('stroke', (i) => cat(2 + PH.indexOf(D.pheno[i]))).attr('stroke-width', 1.6)
    .on('mousemove', (e, i) => tip(e, `${D.ids[i]} · ${D.pheno[i]} · ${D.resp[i] ? 'responder' : 'non-responder'}<br>estimated CD8 ${pc(est[i])} (true ${pc(D.true[i][cd8])})`)).on('mouseleave', untip);
  const s3 = stats(f3g.root);
  s3.set(Object.fromEntries(PH.map((p) => [`${p}: mean est. CD8 · response rate`, `${pc(D.cd8_by_pheno[p])} · ${pc(D.resp_by_pheno[p])}`]).concat([['AUC for response: est. CD8 / IFN-γ score', `${f3(D.auc_resp.est_cd8)} / ${f3(D.auc_resp.ifng)}`]])));

  // ---------------- prose
  const m = SC.markers, a = SC.all, lg = SC.log, mi = SC.missing;
  const mr = (s) => d3.mean(s.r.filter((v) => v != null));
  const tum = CT.indexOf('Tumour');
  section(main, 'What to notice', `<ul>
    <li>With <b>marker genes in linear space</b> the mean correlation across cell types is ${f3(mr(m))}; for CD8 T cells r = ${f3(m.r[cd8])}. The reference was built from the same simulated cells, which is the best case; real references come from other patients and platforms.</li>
    <li>Colour by <b>sequencing centre</b>: Centre B estimates are shifted (mean CD8 bias ${pc(m.bias['Centre B'][cd8])} vs ${pc(m.bias['Centre A'][cd8])} at Centre A) because Centre B's chemistry changes each gene by a different factor, and the reference is not platform-matched.</li>
    <li><b>All 2,000 genes</b> (mean r ${f3(mr(a))}): least squares is dominated by the most highly expressed genes (housekeeping genes, which say nothing about cell type), by genes whose expression depends on programs rather than identity (IFN-γ genes rise in T cells, macrophages and tumour), and by Centre B's gene-wise shifts. Signature matrices are deliberately small and cell-type specific.</li>
    <li><b>Log scale</b>: the correlations still look respectable (mean r ${f3(mr(lg))}), but the fractions themselves are wrong: tumour RMSE ${pc(lg.rmse[tum])} versus ${pc(m.rmse[tum])} on the linear scale. Mixing is additive in counts, not in logs (log(a + b) ≠ log a + log b), so NNLS on log data fits the wrong model. Correlation hides bias; check RMSE too.</li>
    <li><b>Missing cell type</b>: drop fibroblasts from the reference and their RNA has to go somewhere: the tumour fraction RMSE becomes ${pc(mi.rmse[tum])} (versus ${pc(m.rmse[tum])}). NNLS never says "I don't know this cell type".</li>
    <li>In the last figure, excluded tumours still have an estimated ${pc(D.cd8_by_pheno.excluded)} CD8 T cells on average but only ${pc(D.resp_by_pheno.excluded)} respond. The estimated CD8 fraction predicts response with AUC ${f3(D.auc_resp.est_cd8)}, below the IFN-γ score's ${f3(D.auc_resp.ifng)}: it is location and activation state, not just count, that matters. That is the motivation for single-cell and spatial data.</li>
  </ul>`);
  steps(main, 'How it works', [
    'Build a <b>signature matrix</b> S (genes × cell types): average expression of each cell type, in linear units (CPM or TPM), on genes that are specific to one cell type.',
    'For each bulk sample b, solve  min ‖S·f − b‖²  subject to f ≥ 0. This is NNLS: ordinary least squares with a non-negativity constraint, solved by an active-set algorithm (Lawson–Hanson).',
    'Normalise f to sum to 1 to get <b>relative fractions of RNA</b>. To get fractions of <i>cells</i>, divide by each cell type\'s RNA content (tumour cells carry more mRNA than T cells); the simulation assumes equal content.',
    'Check against ground truth where you can: mixtures of sorted cells, or matched flow cytometry / histology / single-cell data.',
  ]);
  math(main, 'bulk_g ≈ Σ_t f_t · S_{g,t}        (linear expression units, not logs)\nf̂ = argmin_{f ≥ 0} ‖ S f − b ‖²      then  f̂ ← f̂ / Σ_t f̂_t');
  callout(main, '', 'In biology', '<b>CIBERSORTx</b> (ν-support-vector regression on a signature matrix, with batch correction between the signature and bulk platforms), <b>EPIC</b> (constrained least squares, models an uncharacterised tumour fraction), <b>quanTIseq</b> (constrained least squares for immune cells), <b>MuSiC</b> and <b>BayesPrism</b> (single-cell references, Bayesian), and <b>MCP-counter</b> / <b>xCell</b> (marker-based scores, not true fractions). For spatial spots: <b>cell2location</b>, <b>RCTD</b>, <b>Tangram</b>. The <code>immunedeconv</code> R package runs many of them side by side.');
  callout(main, 'warn', 'Pitfalls', `<ul>
    <li><b>Log-transformed input</b>: deconvolve linear-scale data (TPM/CPM).</li>
    <li><b>Missing or mismatched cell types</b> in the reference silently redistribute their signal to the closest types.</li>
    <li><b>Platform and batch differences</b> between reference and bulk bias estimates (CIBERSORTx's "B-mode"/"S-mode" correction exists for this).</li>
    <li><b>Fractions of RNA, not cells</b>, unless corrected for cell size / mRNA content.</li>
    <li><b>Collinear cell types</b> (CD4 vs Treg, subsets of macrophages) are poorly separable; trust broad lineages more than fine states.</li>
    <li>Deconvolution estimates <b>abundance, not location or activity</b>: excluded and inflamed tumours can have the same CD8 fraction.</li>
  </ul>`);
  seeAlso(main, [['omics', 'sp-deconvolution', 'Spot deconvolution (spatial)'], ['omics', 'sc-annotation', 'Cell-type annotation'], ['concepts', 'linear-regression', 'Linear regression'], ['omics', 'bulk-validation', 'Next: locking & validation']]);
}
