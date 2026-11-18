// Pseudobulk differential expression, lesson format.
// Arc: what → why → where → how it works (6 steps: within-patient correlation → aggregation → fit and test →
//      shuffled-patient calibration → random 6-vs-6 subsets (lab) → power) → pitfalls → recap.
// Prose numbers come from data/omics/sc_pseudobulk.json and sc_pseudobulk_lab.json (scripts/build_sc_pseudobulk_lab.py).
import { header, figure, slider, select, button, checkbox, stats, legend, seeAlso, animate, tip, untip, el } from '../lib/ui.js';
import { frame, cat, css, ink, ink2, muted } from '../lib/plot.js';
import { rng, mean, sd } from '../lib/data.js';
import { pNormal } from '../lib/models.js';
import { lessonNav, chapter, prose, rung, worked, quiz, cards, table, code, recap } from '../lib/lesson.js';
import { stepCard, omicsData } from './_shared.js';
import { canvasScatter, f2, pct, fmtInt } from './_sc_common.js';
const d3 = window.d3;

const sum = (a) => a.reduce((s, v) => s + v, 0);
const med = (a) => d3.median(a);
const pS = (nlp) => (nlp < 3 ? (10 ** -nlp).toPrecision(2) : `1e−${Math.round(nlp)}`);   // plain text (used in stats / tooltips)
const pH = (nlp) => (nlp < 3 ? (10 ** -nlp).toPrecision(2) : `10<sup>−${Math.round(nlp)}</sup>`);
const Phi = (x) => (x >= 0 ? 1 - pNormal(x) / 2 : pNormal(x) / 2);
const zOf = (alpha) => { let lo = 0, hi = 12; for (let i = 0; i < 40; i++) { const m = (lo + hi) / 2; if (pNormal(m) > alpha) lo = m; else hi = m; } return (lo + hi) / 2; };

/** Redraw the grid lines of a frame() after its y domain changed. */
const regrid = (F) => F.g.select('.gridline').call(d3.axisLeft(F.y).ticks(5).tickSize(-F.w).tickFormat(''));

/** Intraclass correlation (one-way ANOVA estimator) of cell values grouped by patient, and the effective number of cells. */
function iccOf(g) {
  const by = new Map();
  g.cell_pat.forEach((p, i) => { if (!by.has(p)) by.set(p, []); by.get(p).push(g.cell_val[i]); });
  const k = by.size, N = g.cell_val.length, gm = mean(g.cell_val);
  let ssb = 0, ssw = 0, sn2 = 0;
  by.forEach((v) => { const m = mean(v); ssb += v.length * (m - gm) ** 2; ssw += sum(v.map((x) => (x - m) ** 2)); sn2 += v.length ** 2; });
  const n0 = (N - sn2 / N) / (k - 1), msb = ssb / (k - 1), msw = ssw / (N - k);
  const icc = Math.max(0, (msb - msw) / (msb + (n0 - 1) * msw));
  const m = N / k;
  return { icc, N, k, m, deff: 1 + (m - 1) * icc, neff: N / (1 + (m - 1) * icc) };
}

/** Null simulation: 6 vs 6 patients, m cells each, patient effect SD tau, cell noise SD 1, no group difference. Returns false-positive rates. */
function simFP(m, tau, nSim = 200, seed = 5) {
  const r = rng(seed + m);
  let fc = 0, fp = 0;
  for (let s = 0; s < nSim; s++) {
    const cs = [0, 0], cq = [0, 0], cn = [0, 0], pm = [[], []];
    for (let g = 0; g < 2; g++) for (let p = 0; p < 6; p++) {
      const u = r.normal(0, tau); let sm = 0;
      for (let c = 0; c < m; c++) { const x = u + r.normal(); cs[g] += x; cq[g] += x * x; sm += x; }
      cn[g] += m; pm[g].push(sm / m);
    }
    const mu = [cs[0] / cn[0], cs[1] / cn[1]], vr = [cq[0] / cn[0] - mu[0] ** 2, cq[1] / cn[1] - mu[1] ** 2];
    if (Math.abs(mu[1] - mu[0]) / Math.sqrt(vr[0] / cn[0] + vr[1] / cn[1]) > 1.96) fc++;
    const v = (sd(pm[0]) ** 2 + sd(pm[1]) ** 2) / 2;
    if (Math.abs(mean(pm[1]) - mean(pm[0])) / Math.sqrt(2 * v / 6) > 2.228) fp++;
  }
  return { cell: fc / nSim, pat: fp / nSim };
}

export default async function render(main) {
  const D = await omicsData(main, 'sc_pseudobulk.json');
  const L = await omicsData(main, 'sc_pseudobulk_lab.json');
  if (!D || !L) return;
  const TYPES = D.types, NP = D.patients.length, isR = D.resp.map((r) => r === 'R');
  const ORD = [...d3.range(NP).filter((i) => isR[i]), ...d3.range(NP).filter((i) => !isR[i])];     // responders first
  const nR = sum(isR.map(Number)), nNR = NP - nR;
  const hitsNv = (t) => D.de[t].tp_nv + D.de[t].fp_nv, hitsPb = (t) => D.de[t].tp_pb + D.de[t].fp_pb;
  const totCells = sum(TYPES.map((t) => D.de[t].n_cells)), totNv = sum(TYPES.map(hitsNv)), totPb = sum(TYPES.map(hitsPb));
  const T = D.de.Tumour;
  const permMean = (t, k) => mean(D.de[t].perm.map((p) => p[k]));
  const gT = L.genes.Tumour, G0 = gT[0], G1 = gT[1];
  const iccT = iccOf(G1), iccS = iccOf(G0);
  const confounded = (s, k) => { const b = L.batch, R = s[3].slice(0, k), N = s[3].slice(k); return new Set(R.map((i) => b[i])).size === 1 && new Set(N.map((i) => b[i])).size === 1 && b[R[0]] !== b[N[0]]; };

  header(main, 'Pseudobulk differential expression',
    'Response to treatment is a property of a <b>patient</b>, not of a cell. Aggregating the cells of each patient into one profile before testing makes the number of patients, rather than the number of cells, the sample size, which is what the comparison is about.');
  lessonNav(main);
  stepCard(main, {
    stage: 'Single-cell · pseudobulk differential expression',
    question: 'Within each cell type, which genes differ between responders and non-responders?',
    data: `${fmtInt(totCells)} annotated cells of ${TYPES.length} types from ${NP} patients (${nR} responders, ${nNR} non-responders), raw counts.`,
    method: 'Sum counts per patient × cell type, then fit a bulk-style linear model with empirical-Bayes variance moderation (limma-style; edgeR and DESeq2 are equivalent in role). Compared with a per-cell Wilcoxon test.',
    verdict: 'stats',
    why: 'This is a hypothesis test, and the open question is what counts as an independent observation. A predictive model would not change the answer: the replicates are patients.',
  });

  // =====================================================================================
  // 1. WHAT
  // =====================================================================================
  let ch = chapter(main, 'what', 'What is pseudobulk differential expression?', 'In one line: <b>sum cells per patient → one profile per patient → bulk differential expression</b>.');
  prose(ch, `<p>Single-cell RNA-seq measures thousands of cells per patient. To ask whether a gene differs between responders and non-responders <i>within one cell type</i>, the counts of all cells of that type in a patient are <b>summed</b> into one <b>pseudobulk</b> profile. Each cell type then has one profile per patient, and a standard bulk RNA-seq differential-expression model (limma, edgeR or DESeq2) is fitted on those profiles. The diagram shows the three stages for tumour cells in this study.</p>`);
  ch.append(flowSvg(D, L));
  prose(ch, '<p>Key terms:</p>');
  table(ch, [
    { t: 'Unit of replication', m: 'The smallest thing that can be re-drawn independently if the study were repeated. For a patient-level question it is the patient.', s: `${NP} patients, not ${fmtInt(totCells)} cells` },
    { t: 'Pseudoreplication', m: 'Treating measurements that share a source (cells from one patient) as independent replicates.', s: 'cell-level tests on patient-level questions' },
    { t: 'Pseudobulk sample', m: 'The summed counts of all cells of one type in one patient, converted to log counts per million (log-CPM).', s: `${D.de.Tumour.n_patients} tumour samples, ${T.genes.length.toLocaleString()} genes` },
    { t: 'Moderated <i>t</i>-statistic', m: 'A <i>t</i>-test whose per-gene variance is partly borrowed from all genes (empirical Bayes). It stabilises estimates when there are few samples.', s: 'limma; edgeR and DESeq2 do the equivalent' },
    { t: 'Intraclass correlation (ICC)', m: 'The share of a gene\'s variance that lies between patients rather than between cells of one patient.', s: `${f2(iccT.icc)} for ${G1.gene} in tumour cells` },
    { t: 'False discovery rate (FDR)', m: 'The expected share of reported genes that are false positives; controlled here at 5% (Benjamini–Hochberg).', s: 'applied across all genes tested' },
  ], [['t', 'Term'], ['m', 'Meaning'], ['s', 'In this study']], { cls: 'compact' });

  // =====================================================================================
  // 2. WHY
  // =====================================================================================
  ch = chapter(main, 'why', 'Why do we need it?', 'Cells from one patient resemble each other more than they resemble cells from another patient.');
  prose(ch, `<p>Cells of one type from one patient share genotype, tumour microenvironment, sample handling and sequencing batch. A test that counts every cell as an independent observation therefore treats <b>patient-to-patient differences</b> as if they were <b>group differences measured many times</b>. The standard error shrinks as cells are added, while the real uncertainty, which is determined by the number of patients, does not. The figure shows one gene across the patients of this study.</p>`);
  const fs = figure(ch, 'One gene, cell by cell and patient by patient',
    'Each dot is a cell (log-normalised expression). Each column is a patient, responders left and non-responders right; diamonds are patient means. Tick "collapse" to replace the cells by their patient means, which is the data the pseudobulk test sees. Hover for details.');
  let sct = 'Tumour', sgi = 0, collapsed = false, tw = 0, stopS = null, jit = [];
  select(fs.controls, { label: 'Cell type', value: sct, options: TYPES, onChange: (v) => { sct = v; sgi = 0; refill(); paintS(); } });
  const sgSel = select(fs.controls, { label: 'Gene', value: '0', options: [], onChange: (v) => { sgi = +v; paintS(); } });
  const refill = () => { sgSel.innerHTML = ''; L.genes[sct].forEach((g, i) => sgSel.append(el('option', { value: i, selected: i === sgi }, `${g.gene} (${g.label})`))); };
  refill();
  checkbox(fs.controls, { label: 'Collapse cells to patient means', checked: false, onChange: (v) => { collapsed = v; stopS && stopS(); const t0 = tw, t1 = v ? 1 : 0; let el0 = 0; stopS = animate((dt) => { el0 += dt; const u = Math.min(1, el0 / 600); tw = t0 + (t1 - t0) * (1 - (1 - u) ** 3); paintS(); return u < 1; }); } });
  const sSt = stats(fs.root);
  const FS = frame(fs.body, { width: 820, height: 380, x: [0, 12.6], y: [0, 5], yLabel: 'log-normalised expression', margin: { bottom: 44, top: 26 } });
  FS.g.select('.axis.x').remove();
  const xPos = (i) => i + 0.5 + (i >= nR ? 0.6 : 0);
  [[0, nR, 1, 'Responders'], [nR, NP, 0, 'Non-responders']].forEach(([a, b, c, t]) => {
    FS.layer('band').append('rect').attr('x', FS.x(xPos(a) - 0.5)).attr('width', FS.x(xPos(b - 1) + 0.5) - FS.x(xPos(a) - 0.5)).attr('y', 0).attr('height', FS.h).attr('fill', cat(c)).attr('fill-opacity', 0.07);
    FS.layer('band').append('text').attr('x', FS.x((xPos(a) + xPos(b - 1)) / 2)).attr('y', -9).attr('text-anchor', 'middle').attr('font-size', 12.5).attr('font-weight', 600).attr('fill', ink()).text(`${t} (${b - a})`);
  });
  ORD.forEach((p, i) => FS.layer('lab').append('text').attr('x', FS.x(xPos(i))).attr('y', FS.h + 16).attr('text-anchor', 'middle').attr('font-size', 11.5).attr('fill', ink2()).text(D.patients[p]));
  FS.layer('lab').append('text').attr('x', FS.w / 2).attr('y', FS.h + 36).attr('text-anchor', 'middle').attr('class', 'ax-label').text('patient (one column each)');
  legend(fs.body, [['cell of a responder', cat(1)], ['cell of a non-responder', cat(0)], ['patient mean', ink()]]);
  function paintS() {
    const g = L.genes[sct][sgi], N = g.cell_val.length;
    if (jit.length !== N || jit.g !== sct + sgi) { const r = rng(9 + sgi); jit = g.cell_val.map(() => (r() - 0.5) * 0.78); jit.g = sct + sgi; }
    const ypad = d3.extent(g.cell_val); FS.y.domain([Math.max(0, ypad[0] - 0.1), ypad[1] * 1.06 + 0.1]);
    FS.g.select('.axis.y').call(d3.axisLeft(FS.y).ticks(5)); regrid(FS);
    const col = (p) => cat(isR[p] ? 1 : 0), slot = (p) => ORD.indexOf(p);
    const pm = new Map(); g.cell_pat.forEach((p, i) => { if (!pm.has(p)) pm.set(p, []); pm.get(p).push(g.cell_val[i]); });
    const pmean = (p) => mean(pm.get(p));
    FS.layer('cells').selectAll('circle').data(g.cell_val.map((v, i) => i)).join('circle').attr('r', 2.6)
      .attr('cx', (i) => FS.x(xPos(slot(g.cell_pat[i])) + jit[i] * (1 - tw)))
      .attr('cy', (i) => FS.y(g.cell_val[i] * (1 - tw) + pmean(g.cell_pat[i]) * tw)).attr('fill', (i) => col(g.cell_pat[i])).attr('fill-opacity', 0.55 - 0.3 * tw)
      .on('mousemove', (e, i) => tip(e, `<b>${D.patients[g.cell_pat[i]]}</b> · ${isR[g.cell_pat[i]] ? 'responder' : 'non-responder'}<br>${g.gene}: ${g.cell_val[i]}`)).on('mouseleave', untip);
    const pts = [...pm.keys()].map((p) => ({ p, m: pmean(p), n: pm.get(p).length, pb: g.pb_val[g.pb_pat.indexOf(p)] }));
    FS.layer('means').selectAll('path').data(pts).join('path').attr('d', d3.symbol(d3.symbolDiamond, 120)).attr('transform', (d) => `translate(${FS.x(xPos(slot(d.p)))},${FS.y(d.m)})`)
      .attr('fill', ink()).attr('stroke', css('--surface')).attr('stroke-width', 1.3)
      .on('mousemove', (e, d) => tip(e, `<b>${D.patients[d.p]}</b> · ${d.n} cells<br>mean ${f2(d.m)} · pseudobulk log₂CPM ${d.pb}`)).on('mouseleave', untip);
    const gm = [1, 0].map((r) => mean(pts.filter((d) => isR[d.p] === !!r).map((d) => d.m)));
    FS.layer('gm').selectAll('g').data([1, 0]).join('g').each(function (r, k) {
      const a = r ? 0 : nR, b = r ? nR : NP, y = FS.y(gm[k]);
      const s = d3.select(this); s.selectAll('*').remove();
      s.append('line').attr('x1', FS.x(xPos(a) - 0.5)).attr('x2', FS.x(xPos(b - 1) + 0.5)).attr('y1', y).attr('y2', y).attr('stroke', ink2()).attr('stroke-dasharray', '5 4').attr('stroke-width', 1.5);
      s.append('text').attr('x', FS.x(xPos(a) - 0.4)).attr('y', 12).attr('font-size', 11.5).attr('fill', ink2()).attr('stroke', css('--surface')).attr('stroke-width', 3).attr('paint-order', 'stroke').text(`dashed line: mean of patient means ${f2(gm[k])}`);
    });
    const icc = iccOf(g);
    sSt.set({ 'units compared': collapsed ? `${pts.length} patients` : `${fmtInt(N)} cells`, 'p-value for the view shown': pS(collapsed ? g.nlp_pb : g.nlp_nv), 'cell-level p (Wilcoxon)': pS(g.nlp_nv), 'pseudobulk p (limma)': pS(g.nlp_pb),
      'between-patient share of variance (ICC)': f2(icc.icc), 'effective number of cells': `${fmtInt(icc.neff)} of ${fmtInt(icc.N)}` });
  }
  paintS();
  prose(ch, `<p>In tumour cells, <b>${G1.gene}</b> has no planted difference between responders and non-responders, yet the cell-level test gives p = ${pH(G1.nlp_nv)}; pseudobulk gives ${pH(G1.nlp_pb)}. Its ICC is ${f2(iccT.icc)}, so with ${Math.round(iccT.m)} cells per patient on average the design effect is ${f2(iccT.deff)}: the ${fmtInt(iccT.N)} cells carry about as much information as <b>${fmtInt(iccT.neff)}</b> independent cells. For ${G0.gene}, a planted interferon-γ gene, the ICC is ${f2(iccS.icc)}, and the cell-level p-value (${pH(G0.nlp_nv)}) still overstates the evidence compared with the patient-level value (${pH(G0.nlp_pb)}).</p>`);
  table(ch, [
    { a: 'Cell-level test (Wilcoxon, <i>t</i>-test)', u: 'cells', c: 'no', n: 'Suitable for describing clusters and marker genes; not for comparing patient groups.' },
    { a: 'Mixed model with a patient random effect (e.g. NEBULA, MAST with random effects)', u: 'cells, with patient structure', c: 'yes', n: 'Keeps cell-level information; slower and less stable with few patients.' },
    { a: '<b>Pseudobulk + limma / edgeR / DESeq2</b>', u: '<b>patients</b>', c: '<b>yes</b>', n: '<b>Fast, well calibrated, uses mature bulk tooling; discards within-patient detail.</b>' },
  ], [['a', 'Approach'], ['u', 'Unit tested'], ['c', 'Accounts for patient correlation?'], ['n', 'Notes']], { cls: 'compact' });

  // =====================================================================================
  // 3. WHERE
  // =====================================================================================
  ch = chapter(main, 'where', 'Where is it used?', 'Any single-cell or spatial study that compares conditions across donors, patients or animals.');
  cards(ch, [
    { tag: 'Clinical cohorts', title: 'Responders vs non-responders', body: 'The question of this page: genes that differ by outcome in tumour, T cells or macrophages, tested across patients.' },
    { tag: 'Disease atlases', title: 'Disease vs healthy donors', body: 'Cell-type-specific differential expression in lung, kidney or gut atlases, with donor as the unit (muscat, decoupler, Seurat <code>AggregateExpression</code>).' },
    { tag: 'Treatment time courses', title: 'Before vs after therapy', body: 'Paired biopsies: pseudobulk per patient and time point, with patient as a blocking factor in the design.' },
    { tag: 'Perturbation screens', title: 'Guide or drug vs control', body: 'Cells are aggregated per replicate well, donor or pool, so that biological replicates, not cells, are tested.' },
    { tag: 'Spatial omics', title: 'Region or niche per section', body: 'Spots or cells from one tissue section are aggregated per section and region, then compared across sections or donors.' },
    { tag: 'Model organisms', title: 'Knock-out vs wild type', body: 'Aggregation per animal. Typical designs have 3–6 animals per group, so variance moderation matters.' },
    { tag: 'Pathway analysis', title: 'Signatures per cell type', body: 'Pseudobulk profiles feed activity inference (decoupler, GSEA) at patient level, which gives one score per patient and cell type.' },
    { tag: 'Biomarker discovery', title: 'Candidate genes for validation', body: 'Genes ranked on patient-level evidence are carried into a bulk or targeted validation cohort (see the bulk differential-expression page).' },
  ]);
  prose(ch, '<p><b>When not to use it:</b> fewer than about three patients per group (no variance can be estimated); differences in cell-type <i>abundance</i> (use compositional methods such as scCODA or propeller); comparisons <i>within</i> one sample, such as marker genes between clusters (cells are the only replicates available, and the result is descriptive); continuous trajectories, where per-cell models with a patient random effect fit better; and questions about cell-to-cell variability, which aggregation removes.</p>');

  // =====================================================================================
  // 4. HOW
  // =====================================================================================
  ch = chapter(main, 'how', 'How it works', 'Six steps: why cells are correlated, how to aggregate, fit and test, then what a 6-versus-6 comparison can and cannot show.');

  // ---- Step 1: simulation
  let rb = rung(ch, 1, 'Within-patient correlation inflates false positives',
    `<p>Let each cell have expression = (patient effect) + (cell noise). The patient effect has standard deviation τ and the cell noise 1, so the ICC is τ² / (τ² + 1). The simulation below has <b>no</b> difference between the groups. A test is calibrated if it reports p &lt; 0.05 in 5% of such experiments. The design effect, 1 + (<i>m</i> − 1) × ICC for <i>m</i> cells per patient, gives the factor by which the effective number of cells falls short of the nominal one.</p>`);
  const f1 = figure(rb, 'False-positive rate under the null (6 vs 6 patients, teaching simulation)',
    'Each point is the share of 200 simulated experiments with p < 0.05. Blue: every cell is a replicate (z-test on cells). Orange: patient means are the replicates (<i>t</i>-test, 10 degrees of freedom). This is a simulation, separate from the study data; the study values appear in the text below.');
  let tau = 0.6;
  slider(f1.controls, { label: 'patient-to-patient SD τ', min: 0, max: 1.2, step: 0.05, value: tau, fmt: (v) => v.toFixed(2), onInput: (v) => { tau = v; paint1(); } });
  const s1 = stats(f1.root);
  const F1 = frame(f1.body, { width: 820, height: 290, x: [0, 100], y: [0, 0.6], xLabel: 'cells per patient (m)', yLabel: 'false-positive rate', margin: { top: 16 } });
  F1.layer('ref').append('line').attr('x1', 0).attr('x2', F1.w).attr('y1', F1.y(0.05)).attr('y2', F1.y(0.05)).attr('stroke', ink()).attr('stroke-dasharray', '5 4').attr('stroke-width', 1.5);
  F1.layer('ref').append('text').attr('x', F1.w - 4).attr('y', F1.y(0.05) - 6).attr('text-anchor', 'end').attr('font-size', 11.5).attr('fill', ink2()).text('nominal 5%');
  const note1 = F1.layer('ref').append('text').attr('x', 10).attr('y', 14).attr('font-size', 12).attr('fill', ink());
  legend(f1.body, [['cells as replicates', cat(0)], ['patients as replicates', cat(1)], ['nominal 5%', ink()]]);
  const MS = [5, 10, 20, 40, 70, 100];
  function paint1() {
    const R = MS.map((m) => ({ m, ...simFP(m, tau) }));
    F1.y.domain([0, Math.max(0.12, d3.max(R, (r) => Math.max(r.cell, r.pat)) * 1.15)]); F1.g.select('.axis.y').call(d3.axisLeft(F1.y).ticks(5, '.0%')); regrid(F1);
    F1.layer('ref').selectAll('line').attr('y1', F1.y(0.05)).attr('y2', F1.y(0.05)); F1.layer('ref').select('text').attr('y', F1.y(0.05) - 6);
    [['cell', cat(0)], ['pat', cat(1)]].forEach(([k, c]) => {
      F1.layer(k).selectAll('path').data([R]).join('path').attr('fill', 'none').attr('stroke', c).attr('stroke-width', 2.2).attr('d', d3.line().x((r) => F1.x(r.m)).y((r) => F1.y(r[k])));
      F1.layer(k + 'p').selectAll('circle').data(R).join('circle').attr('cx', (r) => F1.x(r.m)).attr('cy', (r) => F1.y(r[k])).attr('r', 5).attr('fill', c).attr('stroke', css('--surface')).attr('stroke-width', 1.4)
        .on('mousemove', (e, r) => tip(e, `m = ${r.m} cells per patient<br>${k === 'cell' ? 'cells as replicates' : 'patients as replicates'}: ${pct(r[k], 1)} false positives`)).on('mouseleave', untip);
    });
    const icc = tau ** 2 / (1 + tau ** 2), r40 = R[3];
    note1.text(tau === 0 ? 'τ = 0: no patient effect, both tests are calibrated' : `ICC ${f2(icc)}: at 40 cells per patient the cell-level test rejects ${pct(r40.cell)} of null experiments`);
    s1.set({ ICC: f2(icc), 'design effect at m = 40': f2(1 + 39 * icc), 'effective cells (of 480)': fmtInt(480 / (1 + 39 * icc)), 'cell-level, m = 40': pct(r40.cell, 1), 'patient-level, m = 40': pct(r40.pat, 1) });
  }
  paint1();
  prose(rb, `<p>The cell-level error rate rises with both the ICC and the number of cells per patient, whereas the patient-level test stays near 5%. In the study, ${G1.gene} in tumour cells has an ICC of ${f2(iccT.icc)}, which falls within the range where the simulation shows substantial inflation.</p>`);

  // ---- Step 2: aggregation
  rb = rung(ch, 2, 'Aggregate: one profile per patient and cell type',
    `<p>For each cell type, raw counts are summed over the cells of each patient. Samples with fewer than 10 cells of that type are dropped, because a profile built from a handful of cells is dominated by sampling noise. The summed counts are scaled to counts per million and log-transformed (log₂(CPM + 1)); edgeR and DESeq2 use their own normalisation factors instead. Always sum <b>raw counts</b>: averaging log-normalised values changes the mean–variance relationship the bulk models assume.</p>`);
  const f2g = figure(rb, 'Cells per patient and cell type', 'Numbers are cells. Dashed outlines mark patient × cell-type combinations below the 10-cell cut-off, which are excluded. Colour scale is square-root of the count. Hover a cell for details.');
  heatSvg(f2g.body, D, L, ORD, isR);
  const lowN = sum(TYPES.map((t) => L.ncells[t].filter((n) => n < 10).length)), tN = L.ncells.Tumour;
  prose(rb, `<p>Tumour cells are well covered (${d3.min(tN)}–${d3.max(tN)} per patient), whereas ${TYPES.filter((t) => D.de[t].n_patients < NP).map((t) => `${t} cells reach the cut-off in only ${D.de[t].n_patients} patients (${D.de[t].n_R} responders, ${D.de[t].n_NR} non-responders)`).join(' and ')}. The sample size for each cell type is therefore smaller than the number of patients in the study, and it is not the same for every cell type. Tumour cell counts alone vary ${(d3.max(tN) / d3.min(tN)).toFixed(1)}-fold between patients, so some pseudobulk profiles are far more precise than others; weighting (limma-voom, edgeR quasi-likelihood) accounts for this.</p>`);

  // ---- Step 3: fit and test
  rb = rung(ch, 3, 'Fit a linear model per gene and test the group coefficient',
    `<p>For gene <i>g</i> and patient <i>p</i>, the pseudobulk value <i>y</i> is modelled as a group mean plus patient-level noise. Covariates such as sequencing batch or sex can be added to the design. With only a few patients, each gene's variance estimate is unstable, so limma shrinks it towards a value shared across genes (<b>empirical Bayes</b>), which adds effective degrees of freedom.</p>
     <div class="calc">y_pg = β0g + β1g × responder_p + ε_pg          ε ~ N(0, σg²)
moderated σ̃g² = (d0 s0² + d sg²) / (d0 + d)        t = β̂1g / (σ̃g √(1/n₁ + 1/n₀))
then Benjamini–Hochberg across the genes tested</div>`);
  const f3 = figure(rb, 'The same cells, two tests', 'Each dot is a gene; x is log₂ fold change (responders / non-responders), y is −log₁₀ p. Large dots pass 5% FDR. Red: the simulation planted a difference for this gene (interferon-γ, exhaustion and related programmes). Grey: no planted difference. Hover for gene details.');
  let ct = 'Tumour';
  const sel3 = select(f3.controls, { label: 'Cell type', value: ct, options: TYPES, onChange: (v) => { ct = v; sel4.value = v; paint3(); paint4(); } });
  const st3 = stats(f3.root);
  const row3 = el('div', { class: 'fig-row' }); f3.body.append(row3);
  const pA = el('div'), pB = el('div'); row3.append(pA, pB);
  const hov = (i) => { const e = D.de[ct]; return `<b>${e.genes[i]}</b> · ${e.planted[i] ? 'planted difference' : 'no planted difference'}<br>cells as replicates: log₂FC ${e.lfc_nv[i]}, p = ${pS(e.nlp_nv[i])}${e.sig_nv[i] ? ' (FDR < 5%)' : ''}<br>pseudobulk: log₂FC ${e.lfc_pb[i]}, p = ${pS(e.nlp_pb[i])}${e.sig_pb[i] ? ' (FDR < 5%)' : ''}`; };
  const vA = canvasScatter(pA, { width: 460, height: 360, x: [-4, 4], y: [0, 10], xLabel: 'log₂ fold change', yLabel: '−log₁₀ p (cells as replicates)', hover: hov });
  const vB = canvasScatter(pB, { width: 460, height: 360, x: [-4, 4], y: [0, 10], xLabel: 'log₂ fold change', yLabel: '−log₁₀ p (patients as replicates)', hover: hov });
  const cap3 = [el('div', { class: 'fig-caption' }), el('div', { class: 'fig-caption' })]; pA.prepend(cap3[0]); pB.prepend(cap3[1]);
  legend(f3.body, [['planted difference', cat(7)], ['no planted difference', muted()], ['large dot: FDR < 5%', ink()]]);
  function paint3() {
    const e = D.de[ct], idx = d3.range(e.genes.length).sort((a, b) => e.planted[a] - e.planted[b]);
    [[vA, e.lfc_nv, e.nlp_nv, e.sig_nv], [vB, e.lfc_pb, e.nlp_pb, e.sig_pb]].forEach(([P, lfc, nlp, sig], k) => {
      const xm = Math.max(1, d3.quantile(lfc.map(Math.abs), 0.999)) * 1.1;
      P.F.x.domain([-xm, xm]); P.F.y.domain([0, Math.max(3, d3.max(nlp)) * 1.12]);
      P.F.g.select('.axis.x').call(d3.axisBottom(P.F.x).ticks(5)); P.F.g.select('.axis.y').call(d3.axisLeft(P.F.y).ticks(5)); regrid(P.F);
      P.draw({ idx, x: lfc, y: nlp, color: (i) => (e.planted[i] ? cat(7) : muted()), r: (i) => (sig[i] ? 3.6 : 1.8), alpha: 0.8 });
      const L2 = P.F.layer('ann'); L2.selectAll('*').remove();
      d3.range(e.genes.length).sort((a, b) => nlp[b] - nlp[a]).slice(0, 3).forEach((i, r) => L2.append('text').attr('x', P.F.x(lfc[i]) + (lfc[i] > 0 ? -8 : 8)).attr('y', P.F.y(nlp[i]) - 6 + r * 13).attr('text-anchor', lfc[i] > 0 ? 'end' : 'start').attr('font-size', 11).attr('fill', ink()).text(e.genes[i]));
      const hits = sig.filter(Boolean).length;
      if (!hits) L2.append('text').attr('x', P.F.w / 2).attr('y', 14).attr('text-anchor', 'middle').attr('font-size', 12.5).attr('font-weight', 600).attr('fill', ink()).attr('stroke', css('--surface')).attr('stroke-width', 3).attr('paint-order', 'stroke').text('no gene reaches FDR 5%');
      else { const cut = d3.min(nlp.filter((v, i) => sig[i])); L2.append('line').attr('x1', 0).attr('x2', P.F.w).attr('y1', P.F.y(cut)).attr('y2', P.F.y(cut)).attr('stroke', ink2()).attr('stroke-dasharray', '5 4'); L2.append('text').attr('x', 4).attr('y', P.F.y(cut) - 5).attr('text-anchor', 'start').attr('font-size', 11).attr('fill', ink2()).text('smallest p at FDR 5%'); }
    });
    cap3[0].innerHTML = `<b>Cells as replicates</b>: ${fmtInt(e.n_cells_R)} vs ${fmtInt(e.n_cells_NR)} cells`;
    cap3[1].innerHTML = `<b>Pseudobulk</b>: ${e.n_R} vs ${e.n_NR} patients`;
    st3.set({ 'genes tested': fmtInt(e.genes.length), 'planted genes': e.n_planted, 'cell-level hits (planted / other)': `${hitsNv(ct)} (${e.tp_nv} / ${e.fp_nv})`, 'pseudobulk hits (planted / other)': `${hitsPb(ct)} (${e.tp_pb} / ${e.fp_pb})` });
  }
  paint3();
  const rankPb = d3.range(T.genes.length).sort((a, b) => T.nlp_pb[b] - T.nlp_pb[a]);
  const top20 = rankPb.slice(0, 20).filter((i) => T.planted[i]).length, top20exp = 20 * T.n_planted / T.genes.length;
  prose(rb, `<p><b>The two tests answer differently.</b> In tumour cells the cell-level test reports ${hitsNv('Tumour')} genes at 5% FDR: all ${T.tp_nv} planted genes and ${T.fp_nv} others, so only ${pct(T.tp_nv / hitsNv('Tumour'))} of its hits correspond to a planted difference. Pseudobulk reports <b>${hitsPb('Tumour')}</b>. Across all ${TYPES.length} cell types it reports ${totPb} gene${totPb === 1 ? '' : 's'} at 5% FDR, against ${totNv} for the cell-level test. This is the expected direction (pseudobulk is the more conservative test), but the size of the gap and the empty result need explaining; steps 4 and 6 do so.</p>
    <p>The empty result is not uninformative. ${top20} of the 20 most significant pseudobulk genes in tumour cells are planted, against about ${top20exp.toFixed(1)} expected by chance, and the strongest, ${G0.gene}, has an uncorrected p of ${pH(G0.nlp_pb)}. The ranking carries information; the evidence for any single gene does not survive correction for ${T.genes.length.toLocaleString()} tests.</p>`);

  // ---- Step 4: calibration by shuffling
  rb = rung(ch, 4, 'Check calibration by shuffling patients',
    `<p>Real labels cannot show whether a test is calibrated, because the truth is unknown. Shuffling does: assigning patients to two fake groups, each holding ${nR / 2} true responders and ${nNR / 2} true non-responders, cancels the response effect, so any gene reported is a false positive. A calibrated test finds about 5% of genes at p &lt; 0.05 and a flat p-value histogram; with FDR control it should report almost nothing.</p>`);
  const f4 = figure(rb, `${D.n_perm} shuffles of the patient labels`, 'Left: genes passing 5% FDR in each shuffle. Right: p-value histogram for the first shuffle; the dashed line is the flat histogram expected under the null. Hover bars for values.');
  const sel4 = select(f4.controls, { label: 'Cell type', value: ct, options: TYPES, onChange: (v) => { ct = v; sel3.value = v; paint3(); paint4(); } });
  const st4 = stats(f4.root);
  const row4 = el('div', { class: 'fig-row' }); f4.body.append(row4);
  const qA = el('div'), qB = el('div'); row4.append(qA, qB);
  const N4 = frame(qA, { width: 460, height: 300, x: [0.5, D.n_perm + 0.5], y: [0, 10], xLabel: 'shuffle', yLabel: 'genes with FDR < 5%', margin: { left: 64 } });
  N4.g.selectAll('.ax-label').filter(function () { return !!this.getAttribute('transform'); }).attr('y', -52);   // clear the 4-digit tick labels
  const H4 = frame(qB,{ width: 440, height: 300, x: [0, 1], y: [0, 1], xLabel: 'p-value', yLabel: 'share of genes' });
  legend(f4.body, [['cells as replicates', cat(0)], ['pseudobulk', cat(1)], ['expected if calibrated', ink()]]);
  const noteN = N4.layer('n').append('text').attr('x', 8).attr('y', 14).attr('font-size', 12).attr('fill', ink());
  function paint4() {
    const e = D.de[ct], mx = d3.max(e.perm, (p) => Math.max(p.naive, p.pb));
    N4.y.domain([0, Math.max(5, mx * 1.12)]); N4.g.select('.axis.y').call(d3.axisLeft(N4.y).ticks(5)); regrid(N4); N4.g.select('.axis.x').call(d3.axisBottom(N4.x).ticks(5));
    const bw = N4.x(1) - N4.x(0), flat = e.perm.flatMap((p, k) => [{ k, v: p.naive, s: 0 }, { k, v: p.pb, s: 1 }]);
    N4.layer('b').selectAll('rect').data(flat).join('rect').attr('x', (d) => N4.x(d.k + 1) - bw * 0.42 + d.s * bw * 0.42).attr('width', bw * 0.4).attr('y', (d) => N4.y(d.v)).attr('height', (d) => N4.h - N4.y(d.v)).attr('fill', (d) => cat(d.s))
      .on('mousemove', (ev, d) => tip(ev, `shuffle ${d.k + 1}: ${d.v} genes (${d.s ? 'pseudobulk' : 'cells as replicates'})`)).on('mouseleave', untip);
    noteN.text(`mean per shuffle: ${permMean(ct, 'naive').toFixed(1)} (cells) vs ${permMean(ct, 'pb').toFixed(1)} (pseudobulk)`);
    const tot = (h) => sum(h) || 1, hs = [[e.perm_hist_nv, 0], [e.perm_hist_pb, 1]].map(([h, s]) => ({ s, v: h.map((x) => x / tot(h)) }));
    H4.y.domain([0, d3.max(hs.flatMap((h) => h.v)) * 1.1]); H4.g.select('.axis.y').call(d3.axisLeft(H4.y).ticks(5, '.0%')); regrid(H4);
    const w = H4.w / 20;
    H4.layer('b').selectAll('g').data(hs).join('g').attr('fill', (d) => cat(d.s)).selectAll('rect').data((d) => d.v.map((v, b) => ({ v, b, s: d.s }))).join('rect')
      .attr('x', (d) => H4.x(d.b / 20) + d.s * w / 2).attr('width', w / 2 - 0.5).attr('y', (d) => H4.y(d.v)).attr('height', (d) => H4.h - H4.y(d.v))
      .on('mousemove', (ev, d) => tip(ev, `p in ${(d.b / 20).toFixed(2)}–${((d.b + 1) / 20).toFixed(2)}: ${pct(d.v, 1)} of genes (${d.s ? 'pseudobulk' : 'cells as replicates'})`)).on('mouseleave', untip);
    H4.layer('x').selectAll('line').data([0.05]).join('line').attr('x1', 0).attr('x2', H4.w).attr('y1', (d) => H4.y(d)).attr('y2', (d) => H4.y(d)).attr('stroke', ink()).attr('stroke-width', 1.8).attr('stroke-dasharray', '5 4');
    st4.set({ 'p < 0.05, cells as replicates': pct(e.perm_hist_nv[0] / tot(e.perm_hist_nv)), 'p < 0.05, pseudobulk': pct(e.perm_hist_pb[0] / tot(e.perm_hist_pb)), 'shuffles with ≥ 1 false hit (cells)': `${e.perm.filter((p) => p.naive > 0).length} of ${D.n_perm}`, 'shuffles with ≥ 1 false hit (pseudobulk)': `${e.perm.filter((p) => p.pb > 0).length} of ${D.n_perm}` });
  }
  paint4();
  const nvAll = sum(TYPES.map((t) => permMean(t, 'naive'))), pbAll = sum(TYPES.map((t) => permMean(t, 'pb')));
  prose(rb, `<p>In tumour cells the cell-level test reports on average <b>${permMean('Tumour', 'naive').toFixed(0)}</b> genes per shuffle where none exist; pseudobulk reports ${permMean('Tumour', 'pb').toFixed(1)}. The inflation grows with the number of cells (${fmtInt(D.de.Tumour.n_cells)} tumour cells against ${fmtInt(D.de.B.n_cells)} B cells; select a cell type to compare). Summed over all cell types, the means are ${nvAll.toFixed(0)} against ${pbAll.toFixed(1)}. The pseudobulk p-value histogram is not flat either: with ${nR} vs ${nNR} patients it leans towards large p-values, so the test is, if anything, <b>conservative</b>. This is one reason the real comparison returns no hits.</p>`);

  // ---- Step 5: lab, random subsets
  rb = rung(ch, 5, 'Lab: how much does the choice of patients matter?',
    `<p>The study has ${nR} responders and ${nNR} non-responders. This lab repeats the pseudobulk analysis on random subsets of <i>k</i> patients per group (up to 40 draws per setting, precomputed with the same model) and reports, for each draw, the number of genes passing 5% FDR and how many of the 20 top-ranked genes are planted ones. At <i>k</i> = ${nR} the single available draw is the real analysis.</p>`);
  const f5 = figure(rb, 'Random k-versus-k subsets of patients', 'Left: genes passing 5% FDR per draw (square-root axis). Yellow dots are draws where all responders come from one chemistry batch and all non-responders from the other, so response is confounded with batch. Right: planted genes among the 20 most significant genes; the dashed line is the number expected by chance.');
  let c5 = 'Tumour', k5 = 3;
  select(f5.controls, { label: 'Cell type', value: c5, options: TYPES, onChange: (v) => { c5 = v; const km = Math.max(...Object.keys(L.subsets[c5]).map(Number)); sk.input.max = km; if (k5 > km) { k5 = km; sk.set(km); } paint5(); } });
  const sk = slider(f5.controls, { label: 'patients per group (k)', min: 2, max: 6, step: 1, value: k5, fmt: (v) => v, onInput: (v) => { k5 = v; paint5(); } });
  const st5 = stats(f5.root);
  const row5 = el('div', { class: 'fig-row' }); f5.body.append(row5);
  const uA = el('div'), uB = el('div'); row5.append(uA, uB);
  const A5 = frame(uA, { width: 500, height: 320, xScale: d3.scaleSqrt(), x: [0, 100], y: [0, 1], xLabel: 'genes at FDR 5% (square-root scale)', axes: true, grid: false, margin: { left: 20 } });
  A5.g.select('.axis.y').remove();
  const B5 = frame(uB, { width: 420, height: 320, x: [-0.5, 20.5], y: [0, 10], xLabel: 'planted genes among the top 20', yLabel: 'number of draws' });
  legend(f5.body, [['batch-balanced draw', cat(0)], ['response confounded with batch', cat(3)], ['expected by chance', ink()]]);
  const lab5 = A5.layer('t').append('text').attr('x', 6).attr('y', 14).attr('font-size', 12).attr('fill', ink());
  function paint5() {
    const S = L.subsets[c5][String(k5)], P = L.power[c5];
    const dr = S.map((s) => ({ hits: s[0], top: s[1], conf: confounded(s, k5), pats: s[3] }));
    const mx = Math.max(10, d3.max(dr, (d) => d.hits) * 1.05); A5.x.domain([0, mx]);
    A5.g.select('.axis.x').call(d3.axisBottom(A5.x).tickValues([0, 1, 5, 25, 100, 400, 800].filter((v) => v <= mx)));
    const r = rng(4), ys = dr.map(() => 0.1 + 0.8 * r());
    A5.layer('d').selectAll('circle').data(dr).join('circle').attr('cx', (d) => A5.x(d.hits)).attr('cy', (d, i) => A5.y(ys[i])).attr('r', 5).attr('fill', (d) => (d.conf ? cat(3) : cat(0))).attr('fill-opacity', 0.75).attr('stroke', css('--surface')).attr('stroke-width', 1)
      .on('mousemove', (e, d) => tip(e, `${d.hits} genes at FDR 5%<br>${d.top} planted in top 20<br>responders ${d.pats.slice(0, k5).map((i) => D.patients[i]).join(', ')}<br>non-responders ${d.pats.slice(k5).map((i) => D.patients[i]).join(', ')}${d.conf ? '<br><b>response confounded with batch</b>' : ''}`)).on('mouseleave', untip);
    const A5a = A5.layer('a'); A5a.selectAll('text').remove();
    const c0 = dr.findIndex((d) => d.conf);
    if (c0 >= 0) A5a.append('text').attr('x', A5.x(dr[c0].hits) - 8).attr('y', A5.y(ys[c0]) + 4).attr('text-anchor', 'end').attr('font-size', 11.5).attr('fill', ink()).text('confounded with batch');
    const cnt = d3.range(21).map((v) => ({ v, n: dr.filter((d) => d.top === v).length })); B5.y.domain([0, Math.max(4, d3.max(cnt, (c) => c.n) + 1)]); B5.g.select('.axis.y').call(d3.axisLeft(B5.y).ticks(5)); regrid(B5);
    B5.layer('b').selectAll('rect').data(cnt).join('rect').attr('x', (c) => B5.x(c.v - 0.4)).attr('width', B5.x(0.8) - B5.x(0)).attr('y', (c) => B5.y(c.n)).attr('height', (c) => B5.h - B5.y(c.n)).attr('fill', cat(0))
      .on('mousemove', (e, c) => tip(e, `${c.n} draws with ${c.v} planted genes in the top 20`)).on('mouseleave', untip);
    const exp = 20 * P.n_planted / P.n_genes;
    B5.layer('e').selectAll('line').data([exp]).join('line').attr('x1', (v) => B5.x(v)).attr('x2', (v) => B5.x(v)).attr('y1', 0).attr('y2', B5.h).attr('stroke', ink()).attr('stroke-dasharray', '5 4').attr('stroke-width', 1.6);
    const cf = dr.filter((d) => d.conf), bl = dr.filter((d) => !d.conf);
    lab5.text(dr.length === 1 ? 'a single draw: the real analysis' : `${dr.length} random draws`);
    st5.set({ 'patients tested': `${k5} + ${k5}`, 'draws with ≥ 1 hit': `${bl.filter((d) => d.hits > 0).length} of ${bl.length} balanced`, 'median hits (balanced draws)': med(bl.map((d) => d.hits)) ?? '–', 'median planted in top 20': `${med(dr.map((d) => d.top))} (chance ${exp.toFixed(1)})`, 'confounded draws': cf.length ? `${cf.length}: median ${med(cf.map((d) => d.hits))} hits` : 'none' });
  }
  paint5();
  const cfAll = []; const balAll = [];
  Object.entries(L.subsets.Tumour).forEach(([k, S]) => S.forEach((s) => (confounded(s, +k) ? cfAll : balAll).push(s[0])));
  const k3 = L.subsets.Tumour['3'].map((s) => s[1]), k5t = L.subsets.Tumour['5'].map((s) => s[1]);
  prose(rb, `<p><b>Reading the lab.</b> In tumour cells, ${cfAll.length} of ${cfAll.length + balAll.length} random draws were confounded with batch (all responders in one chemistry, all non-responders in the other); they reported a median of <b>${med(cfAll)}</b> genes, against ${med(balAll)} for the balanced draws. Pseudobulk removes pseudoreplication, but it cannot separate response from a batch effect that is perfectly aligned with it. In the actual study each chemistry holds ${nR / 2} responders and ${nNR / 2} non-responders, so the design is balanced and batch can be modelled as a covariate. Ranking quality also depends on <i>k</i>: the median number of planted genes in the top 20 is ${med(k3)} at k = 3 and ${med(k5t)} at k = 5, against ${(20 * T.n_planted / T.genes.length).toFixed(1)} by chance.</p>`);

  // ---- Step 6: power
  rb = rung(ch, 6, 'What 6 versus 6 can detect',
    `<p>A result of no genes at 5% FDR is a statement about <b>power</b>, not about absence of an effect. Power is the probability that a test detects an effect of a given size. For a gene with standardised effect <i>d</i> (log₂ fold change divided by the residual SD across patients), a comparison of <i>n</i> patients per group has power ≈ Φ(<i>d</i>√(<i>n</i>/2) − <i>z</i>), where <i>z</i> is the critical value for the significance level α. Because ${T.genes.length.toLocaleString()} genes are tested, α must be much smaller than 0.05; the slider sets it.</p>`);
  const f6 = figure(rb, 'Power for the planted genes as the number of patients grows', 'Effect sizes d are estimated from the 6 v 6 data for the genes the simulation planted. Blue: average power over those genes. Orange: share of them with power ≥ 80%. The vertical line marks the study size. Normal approximation; estimates of |d| from 12 patients are noisy and biased upwards, so the curves are optimistic.');
  let c6 = 'Tumour', la = 4;
  select(f6.controls, { label: 'Cell type', value: c6, options: TYPES, onChange: (v) => { c6 = v; paint6(); } });
  slider(f6.controls, { label: 'significance level α', min: 2, max: 6, step: 0.25, value: la, fmt: (v) => `1e−${v}`, onInput: (v) => { la = v; paint6(); } });
  const st6 = stats(f6.root);
  const F6 = frame(f6.body, { width: 820, height: 320, x: [2, 40], y: [0, 1], xLabel: 'patients per group (n)', yLabel: 'power', margin: { top: 16 } });
  const NS = d3.range(2, 41);
  legend(f6.body, [['average power', cat(0)], ['share of genes with power ≥ 80%', cat(1)], ['study size (n = 6)', ink()]]);
  const powerAt = (ds, n, z) => mean(ds.map((d) => Phi(d * Math.sqrt(n / 2) - z)));
  const powerCurve = (c, alpha) => { const ds = L.power[c].d_planted, z = zOf(alpha); return { ds, z, mean: NS.map((n) => powerAt(ds, n, z)), share: NS.map((n) => ds.filter((d) => Phi(d * Math.sqrt(n / 2) - z) >= 0.8).length / ds.length) }; };
  function paint6() {
    const pc = powerCurve(c6, 10 ** -la), idx80 = pc.mean.findIndex((v) => v >= 0.8), n80 = idx80 < 0 ? null : NS[idx80];
    [['mean', cat(0)], ['share', cat(1)]].forEach(([k, c]) => {
      const pts = NS.map((n, i) => ({ n, v: pc[k][i] }));
      F6.layer(k).selectAll('path').data([pts]).join('path').attr('fill', 'none').attr('stroke', c).attr('stroke-width', 2.4).attr('d', d3.line().x((p) => F6.x(p.n)).y((p) => F6.y(p.v)));
      F6.layer(k + 'p').selectAll('circle').data(pts.filter((p) => p.n % 2 === 0)).join('circle').attr('cx', (p) => F6.x(p.n)).attr('cy', (p) => F6.y(p.v)).attr('r', 4).attr('fill', c).attr('stroke', css('--surface')).attr('stroke-width', 1.2)
        .on('mousemove', (e, p) => tip(e, `n = ${p.n} per group<br>${k === 'mean' ? 'average power' : 'share with power ≥ 80%'}: ${pct(p.v)}`)).on('mouseleave', untip);
    });
    const R = F6.layer('ref'); R.selectAll('*').remove();
    R.append('line').attr('x1', F6.x(6)).attr('x2', F6.x(6)).attr('y1', 0).attr('y2', F6.h).attr('stroke', ink()).attr('stroke-dasharray', '5 4').attr('stroke-width', 1.5);
    R.append('text').attr('x', F6.x(6) + 6).attr('y', 14).attr('font-size', 11.5).attr('fill', ink()).text(`study: ${pct(pc.mean[4])} average power`);
    R.append('line').attr('x1', 0).attr('x2', F6.w).attr('y1', F6.y(0.8)).attr('y2', F6.y(0.8)).attr('stroke', muted()).attr('stroke-dasharray', '2 4');
    R.append('text').attr('x', F6.w - 4).attr('y', F6.y(0.8) - 5).attr('text-anchor', 'end').attr('font-size', 11.5).attr('fill', ink2()).text('80% power');
    st6.set({ 'planted genes': pc.ds.length, 'median |d|': f2(med(pc.ds)), 'average power at n = 6': pct(pc.mean[4]), 'genes with power ≥ 80% at n = 6': pct(pc.share[4]), 'n per group for 80% average power': n80 ?? '> 40' });
  }
  paint6();
  const pw = powerCurve('Tumour', 1e-4), i80 = pw.mean.findIndex((v) => v >= 0.8);
  prose(rb, `<p>For the planted genes in tumour cells the estimated median |d| is ${f2(med(L.power.Tumour.d_planted))}, and at α = 10<sup>−4</sup> the average power at 6 patients per group is <b>${pct(pw.mean[4])}</b>. About ${i80 < 0 ? 'more than 40' : NS[i80]} patients per group would be needed for an average power of 80%, and the share of planted genes individually reaching 80% power at 6 per group is ${pct(pw.share[4])}. A cohort of 6 versus 6 can therefore detect only very large effects after correction for thousands of genes. The result of this study is accordingly stated as: <b>the pseudobulk analysis ranks the planted genes well above chance, no gene is individually significant at 5% FDR, and a larger cohort is required</b>. A cell-level test would have reported ${totNv} significant genes, most of them without a planted difference.</p>`);
  quiz(rb, 'A collaborator sequences twice as many cells per patient and finds the cell-level p-values become smaller. Does the pseudobulk result change?',
    'Only slightly. More cells per patient make each patient\'s profile more precise, which removes technical noise, but the between-patient variance and the number of patients are unchanged, so power improves little. Adding patients is what increases power. The cell-level p-values shrink only because the test counts the extra cells as replicates.');
  table(rb, [
    { t: 'Simulation: move τ from 0 to 0.8', s: `Cell-level false-positive rate at m = 40: ${pct(simFP(40, 0).cell, 1)} → <b>${pct(simFP(40, 0.8).cell, 1)}</b>; patient-level ${pct(simFP(40, 0).pat, 1)} → ${pct(simFP(40, 0.8).pat, 1)}`, m: 'Correlation inside patients, not the number of cells, causes the inflation. With no patient effect, cells could be treated as replicates.' },
    { t: 'Shuffle labels, tumour cells', s: `Mean false hits per shuffle: ${permMean('Tumour', 'naive').toFixed(0)} (cells) → <b>${permMean('Tumour', 'pb').toFixed(1)}</b> (pseudobulk)`, m: 'The cell-level test is miscalibrated; pseudobulk is calibrated or conservative.' },
    { t: 'Subsets: k = 3, tumour cells', s: `Median hits in balanced draws ${med(L.subsets.Tumour['3'].filter((s) => !confounded(s, 3)).map((s) => s[0]))}; in confounded draws (${L.subsets.Tumour['3'].filter((s) => confounded(s, 3)).length} of ${L.subsets.Tumour['3'].length}): <b>${med(L.subsets.Tumour['3'].filter((s) => confounded(s, 3)).map((s) => s[0])) ?? '–'}</b>`, m: 'Perfect confounding of response and batch produces hundreds of hits that no test can attribute to response. Balanced designs, or a batch covariate, prevent it.' },
    { t: 'Cell type B', s: `Only ${D.de.B.n_patients} patients reach 10 cells (${D.de.B.n_R} vs ${D.de.B.n_NR}); the subset slider stops at k = ${Math.max(...Object.keys(L.subsets.B).map(Number))}`, m: 'Rare cell types have smaller effective sample sizes. Report the number of patients per cell type with every result.' },
    { t: 'Power: α from 10<sup>−2</sup> to 10<sup>−6</sup>', s: `Average power at n = 6, tumour: ${pct(powerCurve('Tumour', 1e-2).mean[4])} → <b>${pct(powerCurve('Tumour', 1e-6).mean[4])}</b>`, m: 'The multiple-testing burden, not only the sample size, sets the detectable effect.' },
  ], [['t', 'Try'], ['s', 'What you see'], ['m', 'What it means']], { cls: 'compact' });

  // =====================================================================================
  // 5. PITFALLS
  // =====================================================================================
  ch = chapter(main, 'pitfalls', 'Pitfalls', 'Common problems, how to recognise them, and how to address them.');
  cards(ch, [
    { tag: 'Symptom: hundreds of significant genes from a small cohort', title: 'Pseudoreplication', body: `Cells were used as replicates. Tumour cells give ${hitsNv('Tumour')} hits here against ${hitsPb('Tumour')} from pseudobulk, and ${permMean('Tumour', 'naive').toFixed(0)} false hits per shuffled comparison. <b>Fix:</b> aggregate per patient, or use a mixed model with a patient random effect.` },
    { tag: 'Symptom: top genes track the sequencing batch', title: 'Response confounded with batch', body: `A random draw that places all responders in one chemistry gave a median of ${med(cfAll)} hits in tumour cells. <b>Fix:</b> balance response across batches at design time; include batch as a covariate; never correct the data with integrated values and then test.` },
    { tag: 'Symptom: noisy profiles for rare cell types', title: 'Too few cells per sample', body: `Profiles from a handful of cells are dominated by sampling noise, and some patients drop out (${D.de.B.n_patients} of ${NP} for B cells). <b>Fix:</b> set a minimum cell count, use precision weights, and report the sample size per cell type.` },
    { tag: 'Symptom: "nothing differs between groups"', title: 'Reading an empty result as no effect', body: `At 6 vs 6 the average power is ${pct(pw.mean[4])} for the planted genes. <b>Fix:</b> report effect sizes with confidence intervals and the ranking, state the power, and plan a validation cohort.` },
    { tag: 'Symptom: marker genes appear "differential"', title: 'Double dipping', body: 'Clusters were defined from the expression of some genes, and the same genes are then tested between clusters or conditions. The p-values are optimistic by construction. <b>Fix:</b> use labels that do not depend on the tested genes (independent annotation, sample-level labels), or data splitting.' },
    { tag: 'Symptom: odd fold changes from log-averaged data', title: 'Testing normalised or integrated values', body: 'Batch-corrected embeddings and averaged log values are not counts, and the negative-binomial models assume counts. <b>Fix:</b> sum raw counts per sample; model batch as a covariate.' },
    { tag: 'Symptom: apparent differences in abundance', title: 'Composition changes mistaken for expression', body: 'If responders carry more T cells, whole-tissue profiles differ even when each cell type is unchanged. <b>Fix:</b> test within annotated cell types, and analyse proportions with compositional methods.' },
    { tag: 'Symptom: top genes fail to replicate', title: 'Winner\'s curse', body: `Effect sizes of genes selected from 12 patients are inflated. <b>Fix:</b> validate in independent patients, and interpret fold changes of top-ranked genes cautiously.` },
  ]);

  // =====================================================================================
  // 6. RECAP
  // =====================================================================================
  ch = chapter(main, 'recap', 'Recap', 'Key points.');
  recap(ch, [
    'For a question about patients, the <b>patient</b> is the unit of replication. Cells from one patient are correlated, so treating them as independent replicates is <b>pseudoreplication</b>.',
    `The inflation grows with the intraclass correlation and with the number of cells per patient; in tumour cells here, the cell-level test reported on average ${permMean('Tumour', 'naive').toFixed(0)} significant genes per shuffled comparison.`,
    '<b>Pseudobulk</b>: sum raw counts per patient and cell type, drop samples with few cells, normalise, and fit a bulk model (limma, edgeR or DESeq2) with the batch as a covariate.',
    'Empirical-Bayes moderation borrows variance information across genes and is essential when there are only a handful of samples per group.',
    'Check calibration by shuffling patient labels in a way that cancels the true effect: the number of false hits should be near zero.',
    `With ${nR} vs ${nNR} patients, power is low after multiple-testing correction. <b>No significant genes is not evidence of no effect</b>; report the ranking, effect sizes and the number of patients per cell type.`,
    'Pseudobulk cannot repair confounding: if response is aligned with batch, no test can separate them. Balance the design, and validate in independent patients.',
  ]);
  prose(ch, '<h3>In code</h3><p>The Python route aggregates with decoupler and tests with PyDESeq2; the R route uses muscat with edgeR. Both assume raw counts, one row per patient × cell type, and a sample-level design table.</p>');
  code(ch, {
    Python: `import decoupler as dc
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

# adata.layers["counts"]: raw counts; obs has "patient", "cell_type", "response", "batch"
pb = dc.get_pseudobulk(adata, sample_col="patient", groups_col="cell_type",
                       layer="counts", mode="sum", min_cells=10, min_counts=1000)
# (decoupler >= 2: dc.pp.pseudobulk(...) with the same arguments)

for ct in pb.obs["cell_type"].unique():
    sub = pb[pb.obs["cell_type"] == ct].copy()
    dc.filter_by_expr(sub, group="response", min_count=10, min_total_count=15)
    dds = DeseqDataSet(adata=sub, design="~ batch + response")
    dds.deseq2()
    res = DeseqStats(dds, contrast=["response", "R", "NR"])
    res.summary()                       # log2FC, Wald p-value, BH-adjusted p
    print(ct, (res.results_df["padj"] < 0.05).sum(), "genes at 5% FDR")`,
    R: `library(muscat); library(edgeR)

# sce: SingleCellExperiment with colData: cluster_id (cell type), sample_id (patient),
#      group_id (response), batch
sce <- prepSCE(sce, kid = "cluster_id", sid = "sample_id", gid = "group_id", drop = TRUE)
pb  <- aggregateData(sce, assay = "counts", by = c("cluster_id", "sample_id"), fun = "sum")

design <- model.matrix(~ batch + group_id, data = as.data.frame(colData(pb)))
res <- pbDS(pb, method = "edgeR", design = design, coef = ncol(design), min_cells = 10)
tbl <- resDS(sce, res)                  # one table per cell type, FDR in p_adj.loc

# cells as replicates (for comparison only): Wilcoxon / Seurat FindMarkers on cells`,
  });
  seeAlso(main, [['omics', 'bulk-de', 'Bulk differential expression'], ['omics', 'bulk-qc-batch', 'QC, normalisation and batch effects'], ['omics', 'sc-clustering', 'Clustering and cell-type annotation'], ['omics', 'sp-biomarker', 'Spatial biomarker']]);
}

// ---------------------------------------------------------------------------------------------
/** Heatmap of cells per patient (columns, responders first) × cell type (rows). */
function heatSvg(parent, D, L, ORD, isR) {
  const types = D.types, cw = 52, ch = 34, left = 104, top = 46, W = left + ORD.length * cw + 8, H = top + types.length * ch + 24;
  const svg = d3.select(parent).append('svg').attr('viewBox', `0 0 ${W} ${H}`).attr('width', W).attr('role', 'img').attr('aria-label', 'cells per patient and cell type');
  const mx = d3.max(types, (t) => d3.max(L.ncells[t])), col = d3.scaleLinear().domain([0, Math.sqrt(mx)]).range([css('--surface-2'), css('--accent')]).clamp(true);
  ORD.forEach((p, j) => {
    svg.append('text').attr('x', left + j * cw + cw / 2).attr('y', top - 8).attr('text-anchor', 'middle').attr('font-size', 11.5).attr('fill', ink2()).text(D.patients[p]);
    svg.append('rect').attr('x', left + j * cw + 2).attr('y', top - 36).attr('width', cw - 4).attr('height', 6).attr('rx', 3).attr('fill', cat(isR[p] ? 1 : 0));
  });
  svg.append('text').attr('x', left - 8).attr('y', top - 29).attr('text-anchor', 'end').attr('font-size', 11).attr('fill', ink2()).text('response');
  types.forEach((t, i) => {
    svg.append('text').attr('x', left - 8).attr('y', top + i * ch + ch / 2 + 4).attr('text-anchor', 'end').attr('font-size', 12).attr('fill', ink2()).text(t);
    ORD.forEach((p, j) => {
      const n = L.ncells[t][p], low = n < 10, g = svg.append('g').attr('transform', `translate(${left + j * cw},${top + i * ch})`);
      g.append('rect').attr('x', 1).attr('y', 1).attr('width', cw - 2).attr('height', ch - 2).attr('rx', 3).attr('fill', col(Math.sqrt(n))).attr('stroke', low ? ink2() : 'none').attr('stroke-dasharray', low ? '3 2' : null)
        .on('mousemove', (e) => tip(e, `<b>${D.patients[p]}</b> · ${t}<br>${n} cells${low ? ' (below 10: excluded)' : ''}`)).on('mouseleave', untip);
      g.append('text').attr('x', cw / 2).attr('y', ch / 2 + 4).attr('text-anchor', 'middle').attr('font-size', 11.5).attr('fill', Math.sqrt(n) > Math.sqrt(mx) * 0.6 ? css('--surface') : ink()).attr('pointer-events', 'none').text(n);
    });
  });
  svg.append('text').attr('x', left).attr('y', H - 6).attr('font-size', 11.5).attr('fill', ink2()).text('Each patient × cell-type entry with at least 10 cells becomes one pseudobulk sample.');
}

/** The three stages (cells → pseudobulk samples → patient-level test) for tumour cells. */
function flowSvg(D, L) {
  const wrap = el('div', { class: 'figure' });
  wrap.append(el('div', { class: 'fig-title' }, 'The three stages, for tumour cells'));
  const svg = d3.select(wrap).append('svg').attr('viewBox', '0 0 980 250').attr('width', 980).attr('role', 'img').attr('aria-label', 'cells, pseudobulk samples, patient-level test');
  const isR = D.resp.map((r) => r === 'R'), nC = L.ncells.Tumour, T = D.de.Tumour, NP = D.patients.length;
  const BW = 280, gap = 55, H = 220, r = rng(2);
  const boxes = [
    { t: '① Cells', l: [`${fmtInt(T.n_cells)} tumour cells`, `from ${NP} patients`], f: (g) => {
      D.patients.forEach((_, p) => { const cx = 36 + (p % 6) * 44, cy = 112 + Math.floor(p / 6) * 56, n = Math.min(16, Math.round(nC[p] / 14)); for (let i = 0; i < n; i++) g.append('circle').attr('cx', cx + (r() - 0.5) * 30).attr('cy', cy + (r() - 0.5) * 30).attr('r', 3).attr('fill', cat(isR[p] ? 1 : 0)).attr('fill-opacity', 0.75); });
    } },
    { t: '② Sum counts per patient', l: [`${T.n_patients} pseudobulk samples`, `× ${T.genes.length.toLocaleString()} genes`], f: (g) => {
      D.patients.forEach((_, p) => { const x0 = 18 + (p % 6) * 44, y0 = 88 + Math.floor(p / 6) * 56; g.append('rect').attr('x', x0).attr('y', y0).attr('width', 36).attr('height', 44).attr('rx', 4).attr('fill', cat(isR[p] ? 1 : 0)).attr('fill-opacity', 0.35).attr('stroke', cat(isR[p] ? 1 : 0)); g.append('text').attr('x', x0 + 18).attr('y', y0 + 27).attr('text-anchor', 'middle').attr('font-size', 11).attr('fill', ink()).text(nC[p]); });
    } },
    { t: '③ Test between patients', l: [`${T.n_R} responders vs ${T.n_NR} non-responders`, 'moderated t, FDR'], f: (g) => {
      [1, 0].forEach((rr) => { const y = rr ? 120 : 184; for (let i = 0; i < 6; i++) g.append('circle').attr('cx', 40 + i * 34 + (r() - 0.5) * 8).attr('cy', y + (r() - 0.5) * 18).attr('r', 5.5).attr('fill', cat(rr)).attr('stroke', css('--surface')).attr('stroke-width', 1.3); g.append('text').attr('x', 0).attr('y', y - 22).attr('font-size', 11.5).attr('fill', ink2()).text(rr ? 'responders' : 'non-responders'); });
    } },
  ];
  boxes.forEach((b, i) => {
    const x0 = 10 + i * (BW + gap), g = svg.append('g').attr('transform', `translate(${x0},14)`);
    g.append('rect').attr('width', BW).attr('height', H).attr('rx', 10).attr('fill', css('--surface-2')).attr('stroke', css('--axis'));
    g.append('text').attr('x', 14).attr('y', 26).attr('font-weight', 700).attr('font-size', 13.5).attr('fill', ink()).text(b.t);
    b.l.forEach((s, j) => g.append('text').attr('x', 14).attr('y', 48 + j * 18).attr('font-size', 12.5).attr('fill', ink2()).text(s));
    b.f(g.append('g').attr('transform', 'translate(14,0)'));
    if (i < boxes.length - 1) { const ax = x0 + BW + 6, ay = 14 + H / 2; svg.append('path').attr('d', `M${ax},${ay} L${ax + gap - 14},${ay} M${ax + gap - 20},${ay - 6} L${ax + gap - 14},${ay} L${ax + gap - 20},${ay + 6}`).attr('fill', 'none').attr('stroke', ink2()).attr('stroke-width', 2); }
  });
  return wrap;
}
