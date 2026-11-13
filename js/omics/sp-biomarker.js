// Spatial step 6: close the loop. A spatial feature (CD8 T density inside the tumour compartment) vs the bulk IFN-γ score,
// on the 4 main sections + a 36-section simulated spatial cohort. Data: sp_biomarker.json, sp_data.json, bulk_classifier.json.
import { header, figure, select, checkbox, stats, section, steps, callout, math, legend, seeAlso, el, tip, untip } from '../lib/ui.js';
import { frame, css, cat, ink2, muted } from '../lib/plot.js';
import { rng } from '../lib/data.js';
import { stepCard, omicsData } from './_shared.js';
import { tissueMap, ctColor } from './_spatial.js';
import { PHENOS, phenoColor, auc } from './_bulk_common.js';
const d3 = window.d3;

const FEAT = {
  ifn_bulk: { label: 'bulk IFN-γ (CXCL9) score', kind: 'bulk', unit: 'log2' },
  cd8_frac: { label: 'CD8 T fraction of all cells', kind: 'bulk-like', unit: '' },
  d_tumour: { label: 'CD8 T per 100 cells in the tumour compartment', kind: 'spatial', unit: '' },
  infil: { label: 'infiltration ratio log2(tumour / stroma CD8 density)', kind: 'spatial', unit: 'log2' },
  frac_inside: { label: 'share of CD8 T cells inside the tumour compartment', kind: 'spatial', unit: '' },
};
const KIND_COL = { bulk: () => cat(7), 'bulk-like': () => cat(3), spatial: () => cat(2) };

export default async function render(main) {
  header(main, 'Back to the biomarker',
    'Bulk RNA-seq gave us an IFN-γ biomarker for anti-PD-1 response, but it could not tell an <b>inflamed</b> tumour (T cells inside) from an <b>excluded</b> one (T cells stuck in the stroma). A spatial feature that measures <b>where</b> the CD8 T cells are separates them, and that is the part of the biology bulk was blind to.');
  const card = stepCard(main, {
    stage: 'Spatial · step 6 of 6',
    question: 'Does a spatial feature separate inflamed from excluded tumours better than the bulk IFN-γ score? Does it predict response better?',
    data: '…',
    method: 'One number per section, computed from cell positions and types: CD8 T density in a data-derived tumour compartment (cells whose 10 nearest neighbours are at least half tumour). Compared with bulk-like features by ROC AUC with bootstrap intervals.',
    verdict: 'both', why: ' The feature is hand-designed from biology and evaluated with statistics (AUC, confidence intervals). If it went into a response classifier, every honesty rule of the <a href="#/omics/bulk-classifier">bulk classifier</a> page would apply again.',
  });
  const B = await omicsData(main, 'sp_biomarker.json');
  const D = await omicsData(main, 'sp_data.json');
  const BC = await omicsData(main, 'bulk_classifier.json');
  if (!B || !D) return;
  const C = B.cohort, n = B.n;
  const nPh = Object.fromEntries(PHENOS.map((p) => [p, C.filter((c) => c.ph === p).length]));
  card.querySelectorAll('.step-row')[1].querySelector('span').innerHTML = `The ${C.filter((c) => c.main).length} main sections plus a simulated cohort of ${n - C.filter((c) => c.main).length} smaller sections (${n} in total: ${PHENOS.map((p) => `${nPh[p]} ${p}`).join(', ')}), each with a response label drawn from its phenotype's response rate (${B.n_resp} responders). The bulk IFN-γ score is computed from the same tissue, with assay noise added for cohort sections.`;

  // ------------------------------------------------------------------ figure 1: two sections side by side
  const fig1 = figure(main, 'Same T cells, different places',
    'Tumour cells (grey) and CD8 T cells (coloured) in an inflamed and an excluded section; other cells hidden. Hover a CD8 T cell for its distance to the nearest tumour cell.');
  const iI = D.phen.indexOf('inflamed'), iE = D.phen.indexOf('excluded');
  let pair = [iI, iE];
  select(fig1.controls, { label: 'Left', value: String(iI), options: D.sections.map((s, i) => [String(i), s]), onChange: (v) => { pair[0] = +v; paint1(); } });
  select(fig1.controls, { label: 'Right', value: String(iE), options: D.sections.map((s, i) => [String(i), s]), onChange: (v) => { pair[1] = +v; paint1(); } });
  legend(fig1.controls, [['tumour cell', muted()], ['CD8 T cell', ctColor(1)]]);
  const row1 = el('div', { class: 'fig-row' }); fig1.body.append(row1);
  const panes = [0, 1].map(() => { const w = el('div'); row1.append(w); const rd = el('div'); w.append(rd); return { w, rd }; });
  panes.forEach((p) => { p.map = tissueMap(p.w, { size: D.size, px: 420 }); p.stats = stats(p.w); });
  function paint1() {
    pair.forEach((si, k) => {
      const s = D.cells[si], m = C.find((c) => c.name === D.sections[si]), P = panes[k];
      P.rd.innerHTML = `<div style="font-size:13px;font-weight:600;color:var(--ink-2);margin:0 0 4px 2px">${D.sections[si]}</div>`;
      const tum = d3.range(s.n).filter((i) => s.type[i] === 0);
      const del = d3.Delaunay.from(tum, (i) => s.x[i], (i) => s.y[i]);
      const dist = (i) => { const j = tum[del.find(s.x[i], s.y[i])]; return Math.hypot(s.x[i] - s.x[j], s.y[i] - s.y[j]); };
      P.map.draw(s.x, s.y, (i) => (s.type[i] === 0 ? muted() : s.type[i] === 1 ? ctColor(1) : null), (i) => (s.type[i] === 1 ? 7 : 5), (i) => (s.type[i] === 0 ? 0.5 : 1));
      P.map.hover((x, y, near) => { const i = near(x, y); return i < 0 || s.type[i] !== 1 ? null : `CD8 T cell<br>nearest tumour cell: ${dist(i).toFixed(0)} µm`; });
      P.stats.set({ 'CD8 T fraction': (100 * m.cd8_frac).toFixed(1) + '%', 'bulk IFN-γ score': m.ifn_bulk.toFixed(2), 'CD8 T / 100 tumour-compartment cells': m.d_tumour.toFixed(1), 'CD8 T inside tumour compartment': (100 * m.frac_inside).toFixed(0) + '%' });
    });
  }
  paint1();

  // ------------------------------------------------------------------ figure 2: cohort strip plot + ROC
  const fig2 = figure(main, 'Across the cohort: which feature separates what?',
    `Left: every section's value for the chosen feature, by immune phenotype (outlined dots = responders; larger dots = the ${C.filter((c) => c.main).length} main sections). Right: ROC curves for all five features on the chosen task. Hover for details.`);
  let feat = 'd_tumour', task = 'phen';
  select(fig2.controls, { label: 'Feature (left)', value: feat, options: Object.entries(FEAT).map(([k, v]) => [k, `${v.label} (${v.kind})`]), onChange: (v) => { feat = v; paint2(); } });
  select(fig2.controls, { label: 'Task', value: task, options: [['phen', 'inflamed vs excluded'], ['resp', 'responders vs non-responders (all sections)']], onChange: (v) => { task = v; paint2(); } });
  const rd2 = stats(fig2.root);
  const row2 = el('div', { class: 'fig-row' }); fig2.body.append(row2);
  const sW = el('div'), rW = el('div'); row2.append(sW, rW);
  const FS = frame(sW, { width: 440, height: 380, x: [0, 1], y: [0, 1], yLabel: '', xScale: d3.scaleBand().padding(0.3), margin: { left: 54 } });
  FS.x.domain(PHENOS); FS.g.select('.axis.x').call(d3.axisBottom(FS.x));
  const yLab = FS.g.append('text').attr('class', 'ax-label').attr('transform', 'rotate(-90)').attr('x', -FS.h / 2).attr('y', -40).attr('text-anchor', 'middle');
  const FR = frame(rW, { width: 440, height: 380, x: [0, 1], y: [0, 1], xLabel: 'false-positive rate', yLabel: 'true-positive rate' });
  FR.layer('d').append('line').attr('x1', FR.x(0)).attr('y1', FR.y(0)).attr('x2', FR.x(1)).attr('y2', FR.y(1)).attr('stroke', muted()).attr('stroke-dasharray', '4 3');
  const featCol = (k) => cat({ ifn_bulk: 7, cd8_frac: 3, d_tumour: 2, infil: 0, frac_inside: 4 }[k]);
  legend(rW, Object.keys(FEAT).map((k) => [FEAT[k].label.split(' (')[0].replace('infiltration ratio log2(tumour / stroma CD8 density)', 'infiltration ratio'), featCol(k)]));
  const jit = rng(11);
  const jx = C.map(() => (jit() - 0.5) * 0.6);

  function taskData() {
    if (task === 'phen') { const ix = C.map((c, i) => i).filter((i) => C[i].ph !== 'desert'); return { ix, y: ix.map((i) => +(C[i].ph === 'inflamed')) }; }
    const ix = C.map((c, i) => i); return { ix, y: ix.map((i) => C[i].resp) };
  }
  function roc(y, s) {
    const o = d3.range(s.length).sort((a, b) => s[b] - s[a]), P = d3.sum(y), Nn = y.length - P;
    const pts = [{ x: 0, y: 0 }]; let tp = 0, fp = 0;
    o.forEach((i, k) => { if (y[i]) tp++; else fp++; if (k === o.length - 1 || s[o[k + 1]] !== s[i]) pts.push({ x: fp / Nn, y: tp / P }); });
    return pts;
  }
  function bootCI(y, s, seed) {
    const r = rng(seed), out = [];
    for (let b = 0; b < 500; b++) {
      const ii = y.map(() => Math.floor(r() * y.length)), yy = ii.map((i) => y[i]);
      if (d3.sum(yy) === 0 || d3.sum(yy) === yy.length) continue;
      out.push(auc(yy, ii.map((i) => s[i])));
    }
    out.sort(d3.ascending);
    return [d3.quantile(out, 0.025), d3.quantile(out, 0.975)];
  }
  const CI = {};
  const getCI = (t, k) => { const key = t + k; if (!CI[key]) { const prev = task; task = t; const { ix, y } = taskData(); task = prev; CI[key] = bootCI(y, ix.map((i) => C[i][k]), 99); } return CI[key]; };
  const aucOf = (t, k) => { const prev = task; task = t; const { ix, y } = taskData(); task = prev; return auc(y, ix.map((i) => C[i][k])); };

  function paint2() {
    const v = C.map((c) => c[feat]);
    FS.y.domain(d3.extent(v)).nice();
    FS.g.select('.axis.y').call(d3.axisLeft(FS.y).ticks(6)); FS.g.select('.gridline').call(d3.axisLeft(FS.y).ticks(6).tickSize(-FS.w).tickFormat(''));
    yLab.text(FEAT[feat].label.length > 44 ? FEAT[feat].label.split(' (')[0].replace(' log2(tumour / stroma CD8 density)', ' (log2)') : FEAT[feat].label);
    FS.layer('p').selectAll('circle').data(C).join('circle')
      .attr('cx', (c, i) => FS.x(c.ph) + FS.x.bandwidth() / 2 + jx[i] * FS.x.bandwidth()).attr('cy', (c) => FS.y(c[feat])).attr('r', (c) => (c.main ? 7.5 : 5))
      .attr('fill', (c) => phenoColor(c.ph)).attr('fill-opacity', (c) => (c.resp ? 1 : 0.45)).attr('stroke', (c) => (c.resp ? css('--ink') : css('--surface'))).attr('stroke-width', (c) => (c.resp ? 1.8 : 1.2))
      .on('mousemove', (e, c) => tip(e, `<b>${c.name}</b> · ${c.ph} · ${c.resp ? 'responder' : 'non-responder'}<br>${Object.keys(FEAT).map((k) => `${FEAT[k].label.split(' (')[0]}: ${c[k].toFixed(2)}`).join('<br>')}`)).on('mouseleave', untip);
    const { ix, y } = taskData();
    const curves = Object.keys(FEAT).map((k) => ({ k, pts: roc(y, ix.map((i) => C[i][k])) }));
    FR.layer('c').selectAll('path').data(curves, (d) => d.k).join('path').attr('fill', 'none').attr('stroke', (d) => featCol(d.k)).attr('stroke-width', (d) => (d.k === feat ? 3.2 : 1.8)).attr('opacity', (d) => (d.k === feat ? 1 : 0.7))
      .attr('d', (d) => d3.line().curve(d3.curveStepAfter).x((p) => FR.x(p.x)).y((p) => FR.y(p.y))(d.pts));
    const a = aucOf(task, feat), ci = getCI(task, feat), aB = aucOf(task, 'ifn_bulk'), ciB = getCI(task, 'ifn_bulk');
    rd2.set({ task: task === 'phen' ? `inflamed vs excluded (${ix.length} sections)` : `response (${d3.sum(y)} of ${y.length})`, [`AUC, ${FEAT[feat].kind} feature`]: `${a.toFixed(3)} [${ci[0].toFixed(2)}, ${ci[1].toFixed(2)}]`, 'AUC, bulk IFN-γ': `${aB.toFixed(3)} [${ciB[0].toFixed(2)}, ${ciB[1].toFixed(2)}]`, 'build-script AUC': (task === 'phen' ? B.auc[feat].inf_vs_exc : B.auc[feat].resp).toFixed(3) });
  }
  paint2();

  // ------------------------------------------------------------------ figure 3: bulk vs spatial plane
  const fig3 = figure(main, 'Closing the loop: what bulk sees versus what spatial sees',
    'Each section placed by its bulk IFN-γ score (x) and its CD8 T density inside the tumour compartment (y). Bulk alone has to cut along x; the spatial feature cuts along y.');
  let hideDesert = false;
  checkbox(fig3.controls, { label: 'hide desert tumours', checked: false, onChange: (v) => { hideDesert = v; paint3(); } });
  legend(fig3.controls, [...PHENOS.map((p) => [p, phenoColor(p)]), ['responder (outlined)', css('--ink')]]);
  const F3 = frame(fig3.body, { width: 760, height: 400, x: d3.extent(C, (c) => c.ifn_bulk), y: [0, d3.max(C, (c) => c.d_tumour) * 1.05], xLabel: 'bulk IFN-γ (CXCL9) score, log2', yLabel: 'CD8 T per 100 tumour-compartment cells' });
  function paint3() {
    const data = C.filter((c) => !hideDesert || c.ph !== 'desert');
    F3.layer('p').selectAll('circle').data(data, (c) => c.name).join('circle').attr('cx', (c) => F3.x(c.ifn_bulk)).attr('cy', (c) => F3.y(c.d_tumour)).attr('r', (c) => (c.main ? 8 : 5.5))
      .attr('fill', (c) => phenoColor(c.ph)).attr('fill-opacity', (c) => (c.resp ? 1 : 0.5)).attr('stroke', (c) => (c.resp ? css('--ink') : css('--surface'))).attr('stroke-width', (c) => (c.resp ? 1.8 : 1.2))
      .on('mousemove', (e, c) => tip(e, `<b>${c.name}</b> · ${c.ph} · ${c.resp ? 'responder' : 'non-responder'}<br>bulk IFN-γ ${c.ifn_bulk.toFixed(2)}<br>CD8 T / 100 tumour-compartment cells ${c.d_tumour.toFixed(1)}`)).on('mouseleave', untip);
    F3.layer('l').selectAll('text').data(data.filter((c) => c.main), (c) => c.name).join('text').attr('x', (c) => F3.x(c.ifn_bulk) + 11).attr('y', (c) => F3.y(c.d_tumour) + 4).attr('font-size', 11.5).attr('fill', ink2()).text((c) => c.name.split(' ')[0]);
  }
  paint3();

  // ------------------------------------------------------------------ prose (numbers from the JSON)
  const ex = C.filter((c) => c.ph === 'excluded'), inf = C.filter((c) => c.ph === 'inflamed');
  const rng_ = (arr, k) => `${d3.min(arr, (c) => c[k]).toFixed(1)}–${d3.max(arr, (c) => c[k]).toFixed(1)}`;
  const overlapIFN = ex.filter((c) => c.ifn_bulk > d3.min(inf, (q) => q.ifn_bulk)).length;
  const respRate = (p) => { const g = C.filter((c) => c.ph === p); return `${g.filter((c) => c.resp).length}/${g.length}`; };
  // ceiling: response depends only on phenotype in the simulation, so phenotype itself is the best possible score
  const phScore = C.map((c) => ({ inflamed: 2, excluded: 1, desert: 0 }[c.ph]));
  const oracle = auc(C.map((c) => c.resp), phScore);
  const m1 = C.find((c) => c.name === D.sections[iI]), m3 = C.find((c) => c.name === D.sections[iE]);
  let bestBulk = null;
  if (BC && BC.res) {
    const ks = BC.ks.map((k) => [k, BC.res[`real|nested|${k}`]]).filter((q) => q[1]);
    if (ks.length) { const b = ks.reduce((a, q) => (q[1].mean > a[1].mean ? q : a)); bestBulk = { k: b[0], m: b[1].mean, sd: b[1].sd }; }
  }
  section(main, 'What to notice', `<ul>
    <li>In the main sections, ${m1.name.split(' ')[0]} and ${m3.name.split(' ')[0]} have CD8 T fractions of ${(100 * m1.cd8_frac).toFixed(1)}% and ${(100 * m3.cd8_frac).toFixed(1)}%: both have T cells. Inside the tumour compartment the densities are ${m1.d_tumour.toFixed(1)} versus ${m3.d_tumour.toFixed(1)} per 100 cells.</li>
    <li><b>Inflamed vs excluded:</b> the bulk IFN-γ score reaches AUC ${aucOf('phen', 'ifn_bulk').toFixed(3)}, the CD8 T fraction ${aucOf('phen', 'cd8_frac').toFixed(3)}, the spatial density ${aucOf('phen', 'd_tumour').toFixed(3)}. ${overlapIFN} of the ${ex.length} excluded sections score higher on bulk IFN-γ than the lowest inflamed section, because their T cells (and the IFN-γ they make) are still in the sample, just in the wrong place.</li>
    <li>Be suspicious of a perfect AUC. Here "excluded" was <i>simulated</i> by placing CD8 T cells outside the nests, so a feature that measures exactly that is close to circular. The robust lesson is the direction (spatial context adds what bulk cannot see), not the 1.000.</li>
    <li><b>Response</b> is a different, harder question. In the simulation response depends only on phenotype (${PHENOS.map((p) => `${p} ${respRate(p)}`).join(', ')}), so even knowing the true phenotype gives AUC ${oracle.toFixed(3)}. The spatial density reaches ${aucOf('resp', 'd_tumour').toFixed(3)} and bulk IFN-γ ${aucOf('resp', 'ifn_bulk').toFixed(3)}, and the plain CD8 fraction ${aucOf('resp', 'cd8_frac').toFixed(3)}. Switch the task to response and read the intervals: with ${B.n_resp} responders they overlap heavily, so this cohort cannot say which feature predicts response best. A better phenotype separator does not automatically make a better response predictor.</li>
    <li>Tick <b>hide desert tumours</b> in the last figure: inflamed and excluded sections overlap along the x axis (bulk) and separate along the y axis (spatial).</li>
  </ul>`);

  steps(main, 'How it works', [
    'Define a tumour compartment from the data, not from annotations: a cell belongs to it if at least half of its 10 nearest neighbours are tumour cells.',
    'Count CD8 T cells per 100 cells inside and outside the compartment; derive the density inside, the share of CD8 T cells inside, and the log ratio of the two densities.',
    'Compare features on a cohort with ROC AUC and bootstrap confidence intervals, separately for the question the feature was designed for (phenotype) and the clinical question (response).',
    'Before using it clinically: lock the definition, then validate on an independent cohort exactly as in the <a href="#/omics/bulk-validation">bulk validation</a> step.',
  ]);
  math(main, 'tumour compartment = { cells with ≥ 5 of 10 nearest neighbours tumour }\ndensity_in = 100 · #CD8 in compartment / #cells in compartment\ninfiltration = log2( (density_in + 0.2) / (density_out + 0.2) )');

  callout(main, '', 'In biology', `<ul>
    <li>The <b>Immunoscore</b> (CD3/CD8 density in tumour core and invasive margin, from immunohistochemistry) is a validated prognostic test in colorectal cancer, and the same logic underlies inflamed / excluded / desert classification in NSCLC and melanoma.</li>
    <li>Multiplex immunofluorescence (Akoya Vectra/PhenoCycler), imaging mass cytometry and <b>Xenium/CosMx</b> make these features measurable on clinical biopsies; <b>Squidpy</b>, <b>scimap</b> and HALO/QuPath compute densities and distances.</li>
    <li>Meta-analyses (e.g. Lu et al. 2019, JAMA Oncology) found multiplex IHC/IF spatial features predicted anti-PD-(L)1 response better than PD-L1 IHC, tumour mutational burden or gene-expression profiling alone, though on heterogeneous, mostly retrospective cohorts.</li>
  </ul>`);
  callout(main, 'warn', 'Pitfalls', `<ul>
    <li><b>Small n.</b> ${n} sections with ${B.n_resp} responders cannot rank features whose AUCs differ by 0.05; report intervals.</li>
    <li><b>Sampling.</b> A biopsy is a few square millimetres of a heterogeneous tumour; the margin may not be in it at all.</li>
    <li><b>Circularity.</b> A feature that re-measures how a phenotype was defined will look perfect against that phenotype. Validate against outcome.</li>
    <li><b>Same rules as bulk.</b> If you tune thresholds or combine spatial features in a model, use nested cross-validation and an external cohort${bestBulk ? `; the bulk classifier reached nested AUC ${bestBulk.m.toFixed(3)} ± ${bestBulk.sd.toFixed(3)} (k = ${bestBulk.k}) on its ${BC.n} patients, and a spatial feature has to beat that under the same protocol` : ''}.</li>
  </ul>`);
  section(main, 'The study, end to end', `<p>Bulk RNA-seq found an IFN-γ signature that predicts response (<a href="#/omics/bulk-classifier">response classifier</a>). Single-cell data showed which cells make it. Spatial data showed why a T-cell / IFN-γ signal cannot be the whole story: excluded tumours carry T-cell and IFN-γ transcripts too, but their CD8 T cells sit in TGF-β-rich fibroblast stroma instead of touching tumour cells. The natural next step is a model that combines a bulk score with a spatial feature, tested with the same honesty. See the <a href="#/omics/overview">study map</a> for every step and whether it was ML or statistics.</p>`);
  seeAlso(main, [['omics', 'overview', 'The mock study & ML map'], ['omics', 'bulk-classifier', 'Response classifier'], ['omics', 'bulk-validation', 'Validating the signature'], ['omics', 'sp-niches', 'Niches & communication'], ['concepts', 'metrics', 'Classification metrics']]);
}
