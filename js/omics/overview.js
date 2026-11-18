// Tab 2 overview, lesson format: the study design, the planted biology, the analysis map with each page's
// ML / statistics verdict, and how this gene-level draw relates to the patient-level cohort of Tab 1.
// Arc: what → why → where → how it works (6 steps) → pitfalls → recap. Every number in prose comes from data/omics/*.json
// and data/study/cohort.json at render time. The VERDICT table must match the stepCard of each omics page.
import { header, figure, slider, select, stats, legend, seeAlso, tip, untip, el } from '../lib/ui.js';
import { frame, cat, css, ink, ink2, muted } from '../lib/plot.js';
import { mean } from '../lib/data.js';
import { auc, roc } from '../lib/models.js';
import { loadCohort } from '../lib/study.js';
import { lessonNav, chapter, prose, rung, worked, quiz, cards, table, code, recap } from '../lib/lesson.js';
import { stepCard, omicsData } from './_shared.js';
import { PHENOS, phenoColor, CENTRES, centreColor, respColor, GROUP_ORDER, groupColor, groupLabel } from './_bulk_common.js';
const d3 = window.d3;

const f2 = (v) => (+v).toFixed(2);
const f3 = (v) => (+v).toFixed(3);
const pc = (v) => Math.round(100 * v) + '%';
const nf = (v) => Math.round(v).toLocaleString('en-GB');
const sum = (a) => a.reduce((s, v) => s + v, 0);
const VLABEL = { ml: 'Machine learning', stats: 'Statistics', both: 'ML + statistics' };
const vcol = (v) => css({ ml: '--good', stats: '--accent', both: '--warn' }[v]);
const pill = (v) => `<span class="tag" style="border-color:${vcol(v)}">${VLABEL[v]}</span>`;

export default async function render(main) {
  const [C, B, Q, DE, CL, SV, VA, SCQ, SCI, SCA, PB, SPB, SPS] = await Promise.all([loadCohort(), ...['bulk_batch', 'bulk_qc', 'bulk_de', 'bulk_classifier', 'bulk_survival', 'bulk_validation', 'sc_qc', 'sc_integration', 'sc_annotation', 'sc_pseudobulk', 'sp_biomarker', 'sp_svg'].map((n) => omicsData(main, n + '.json'))]);
  if ([B, Q, DE, CL, SV, VA, SCQ, SCI, SCA, PB, SPB, SPS].some((d) => !d)) return;
  const D1 = C.discovery, V1 = C.validation;
  const nSeq = Q.ids.length, nKept = B.ids.length, nExcl = Q.excluded.length;
  const respB = B.resp, nRespB = sum(respB);
  const M = DE.methods, lim = M.limma, welch = M.welch;
  const GT = Object.fromEntries(GROUP_ORDER.map((g) => [g, welch.by_group[g][1]]));
  const share = (c, p) => B.counts[c][p] / sum(Object.values(B.counts[c]));
  const rate = (ix) => ix.length ? sum(ix.map((i) => respB[i])) / ix.length : 0;
  const idx = B.ids.map((_, i) => i);
  const byPh = Object.fromEntries(PHENOS.map((p) => { const ix = idx.filter((i) => B.pheno[i] === p); return [p, { n: ix.length, r: sum(ix.map((i) => respB[i])) }]; }));
  const byC = Object.fromEntries(CENTRES.map((c) => { const ix = idx.filter((i) => B.centre[i] === c); return [c, { n: ix.length, r: sum(ix.map((i) => respB[i])) }]; }));
  const nValR = sum(VA.yv);
  const cl = (m, k) => CL.res[`real|${m}|${k}`].mean, pm = (m, k) => CL.res[`perm|${m}|${k}`].mean;
  const hr = SV.inter, ci = SV.cindex;
  const AN = SCA.methods, ap = SPB.auc;
  const PBT = Object.values(PB.de);
  const fpNv = sum(PBT.map((t) => t.fp_nv)), fpPb = sum(PBT.map((t) => t.fp_pb));
  const rawSil = B.versions.raw.sil_batch, proSil = B.versions.protected.sil_batch;
  const welchP05 = welch.p.filter((p) => p < 0.05).length;

  // One row per omics page: the stage and verdict mirror that page's stepCard.
  const PAGES = [
    { id: 'bulk-qc-batch', lane: 0, row: 0, title: 'QC, normalisation and batch effects', stage: 'Bulk RNA-seq · step 1 of 4', verdict: 'stats',
      q: 'Which libraries failed, and how is the centre effect removed without removing biology?', why: 'Summary statistics, thresholds and per-gene linear models; nothing is predicted.',
      head: `centre silhouette ${f2(rawSil)} → ${f2(proSil)}`, long: `${nExcl} of ${nSeq} libraries excluded; silhouette by centre falls from ${f2(rawSil)} to ${f2(proSil)} with a response-protecting correction.` },
    { id: 'bulk-de', lane: 0, row: 1, title: 'Differential expression', stage: 'Bulk RNA-seq · step 2 of 4', verdict: 'stats',
      q: 'Which genes differ between responders and non-responders, after accounting for centre?', why: 'Inference with error control, not prediction for a new patient.',
      head: `${lim.n_hits} genes, q < 0.05 (no centre: ${welch.n_hits})`, long: `${lim.n_hits} genes called with a centre term, ${welch.n_hits} without; ${lim.by_group['IFNG module'][0]} of ${GT['IFNG module']} IFN-γ module genes recovered.` },
    { id: 'bulk-classifier', lane: 0, row: 2, title: 'Biomarker classifier and nested CV', stage: 'Bulk RNA-seq · step 3 of 4', verdict: 'both',
      q: 'Can expression predict response in patients from another hospital, and how accurate is it honestly?', why: 'Building the classifier is machine learning; estimating and validating its accuracy is statistics.',
      head: `nested AUC ${f2(cl('nested', 200))} vs leaky ${f2(cl('leaky', 200))}`, long: `With 200 selected genes the leaky estimate is ${f2(cl('leaky', 200))} and the nested estimate ${f2(cl('nested', 200))}; a locked 2-gene model scores ${f2(VA.strategies.frozen.auc)} in Centre C.` },
    { id: 'bulk-survival', lane: 0, row: 3, title: 'Survival analysis and risk scores', stage: 'Bulk RNA-seq · step 4 of 4', verdict: 'both',
      q: 'Which tumours progress early, and does the score reflect drug benefit or general tumour biology?', why: 'Cox and Kaplan–Meier are statistics; the penalised risk score is machine learning.',
      head: `proliferation HR ${f2(hr.prolif.hr_io)} (docetaxel ${f2(hr.prolif.hr_ctrl)})`, long: `Proliferation hazard ratio ${f2(hr.prolif.hr_io)} on anti-PD-1 and ${f2(hr.prolif.hr_ctrl)} on docetaxel; IFN-γ ${f2(hr.ifng.hr_io)} and ${f2(hr.ifng.hr_ctrl)}. Cross-validated C-index ${f3(ci.cv)}.` },
    { id: 'sc-clustering', lane: 1, row: 0, title: 'Clustering and cell-type annotation', stage: 'Single-cell · clustering and annotation', verdict: 'ml',
      q: 'Which cell types are present, and which cell belongs to which type?', why: 'Clustering and label transfer are unsupervised and supervised learning.',
      head: `Treg recall ${pc(AN.lt_full.recall.Treg)} vs ${pc(AN.lt_notreg.recall.Treg)} without reference`, long: `${SCI.n_types} cell types in ${nf(SCQ.n)} droplets from ${SCI.n_patients} patients; a regulatory T-cell (Treg) type absent from the reference is recalled ${pc(AN.lt_notreg.recall.Treg)} of the time.` },
    { id: 'sc-pseudobulk', lane: 1, row: 1, title: 'Pseudobulk differential expression', stage: 'Single-cell · pseudobulk differential expression', verdict: 'stats',
      q: 'Within each cell type, which genes differ between responders and non-responders?', why: 'A hypothesis test where the replicates are patients, not cells.',
      head: `false calls: ${fpNv} (cells) vs ${fpPb} (patients)`, long: `Across ${PBT.length} cell types, treating cells as replicates gives ${fpNv} false calls at FDR 5%; pseudobulk gives ${fpPb}.` },
    { id: 'sp-biomarker', lane: 2, row: 2, title: 'Spatial biomarker', stage: 'Spatial · one lesson, six earlier steps merged', verdict: 'both',
      q: 'Are CD8 T cells inside the tumour nests or held outside, and does a spatial feature beat the bulk score?', why: 'Graph and permutation tests are statistics; domains and deconvolution are fitted models.',
      head: `response AUC ${f2(ap.cd8_frac.resp)} (CD8) vs ${f2(ap.ifn_bulk.resp)} (bulk)`, long: `Inflamed vs excluded AUC: ${f2(ap.infil.inf_vs_exc)} for the spatial infiltration ratio, ${f2(ap.ifn_bulk.inf_vs_exc)} for bulk IFN-γ. For response the spatial inside-share scores ${f2(ap.frac_inside.resp)}.` },
  ];
  const PG = Object.fromEntries(PAGES.map((p) => [p.id, p]));
  const cnt = (v) => PAGES.filter((p) => p.verdict === v).length;

  header(main, 'Study overview and analysis map',
    'One fictional anti-PD-1 lung cancer study, seen at three scales: <b>bulk RNA-seq, single-cell and spatial</b>. This page describes the design, the biology planted in the simulation, and a map of the seven analysis pages that follow, each with its verdict on whether the step is machine learning, statistics or both.');
  lessonNav(main);
  stepCard(main, {
    stage: 'The study · overview',
    question: 'Which patients with non-small-cell lung cancer (NSCLC) respond to anti-PD-1 immunotherapy, and which analysis step needs which kind of tool?',
    data: `Bulk discovery cohort (${nSeq} sequenced, ${nKept} after QC) and a validation cohort (${VA.n_v} patients, Centre C), plus a simulated docetaxel control arm (${SV.n_ctrl} patients); single-cell data from ${SCI.n_patients} patients; spatial data from ${SPS.sections.length} tissue sections and ${SPB.n} simulated section summaries (one per patient). All come from one simulated ground truth.`,
    method: 'A map of the seven analysis pages, with the evidence behind each verdict, plus a comparison with the patient-level cohort of Tab 1.',
    verdict: 'both',
    why: ` The study as a whole needs both: statistics for "is this difference real?" and for honest error estimates, machine learning for "predict a new patient" and "find structure without labels". Of the seven pages, ${cnt('stats')} are statistics, ${cnt('ml')} machine learning and ${cnt('both')} both.`,
  });

  // =====================================================================================
  // 1. WHAT
  // =====================================================================================
  let ch = chapter(main, 'what', 'What is the study?', 'In one line: <b>one cohort, three data layers, one planted answer</b>.');
  prose(ch, `<p>The study is a simulated anti-PD-1 immunotherapy trial in NSCLC. Anti-PD-1 drugs release the brake that tumours place on T cells, so response depends on whether T cells are present, active, and able to reach tumour cells. The simulation encodes that biology in gene expression, then measures it three ways. Because the truth is planted, every method on the following pages can be scored against the correct answer, which real data never allows.</p>
    <p>The diagram gives the size of each layer, computed from the data files.</p>`);
  ch.append(designSvg({ nSeq, nKept, nRespB, nGenes: Q.n_genes, nVal: VA.n_v, nValR, nDroplets: SCQ.n, nPat: SCI.n_patients, nTypes: SCI.n_types, nSec: SPS.sections.length, nPanel: SPS.genes.length, nSpC: SPB.n, nSpR: SPB.n_resp, nCtrl: SV.n_ctrl }));
  prose(ch, '<p>Key terms used throughout the tab:</p>');
  table(ch, [
    { t: 'Immune phenotype', m: 'How T cells are distributed in a tumour: <b>inflamed</b> (inside the tumour), <b>excluded</b> (present but held in the stroma), <b>desert</b> (few T cells)', s: `${PHENOS.map((p) => `${byPh[p].n} ${p}`).join(', ')} among ${nKept} patients` },
    { t: 'IFN-γ module', m: 'Genes induced by interferon-γ signalling (CXCL9, CXCL10, GZMB, IDO1 …). The planted <b>predictive</b> signal: it raises the chance of response', s: `${GT['IFNG module']} genes; a module score is their mean` },
    { t: 'CAF / TGF-β module', m: 'Cancer-associated fibroblast (CAF) and TGF-β genes (POSTN, FN1, ACTA2 …). Marks excluded tumours and lowers response', s: `${GT['CAF/TGF-b module']} genes` },
    { t: 'Proliferation module', m: 'Cell-cycle genes (MKI67, TOP2A …). <b>Prognostic</b> only: shortens survival, no direct effect on response', s: `${GT['PROLIF module']} genes` },
    { t: 'Response', m: 'Binary outcome used as the label for prediction and for differential expression', s: `${nRespB} of ${nKept} (${pc(nRespB / nKept)}) responded` },
    { t: 'Centre', m: 'Sequencing hospital. A technical batch that is also partly confounded with biology', s: `Centre B: ${pc(share('Centre B', 'inflamed'))} inflamed tumours; Centre A: ${pc(share('Centre A', 'inflamed'))}` },
  ], [['t', 'Term'], ['m', 'Meaning'], ['s', 'In this study']]);

  // =====================================================================================
  // 2. WHY
  // =====================================================================================
  ch = chapter(main, 'why', 'Why a map of analysis steps?', 'Machine learning is one tool among several, and each step of an omics analysis asks a different kind of question.');
  prose(ch, `<p>A genomic dataset has far more measurements than patients (${nf(Q.n_genes)} genes for ${nKept} patients here, of whom ${nRespB} responded). Under those conditions the choice of method determines whether a result is reliable. Three kinds of question recur:</p>`);
  table(ch, [
    { k: '<b>Inference</b>', q: 'Is this difference real, and how large is it?', t: 'Statistics: linear models, permutation tests, FDR', s: 'Calibrated error rates; replication', e: 'differential expression, pseudobulk' },
    { k: '<b>Prediction</b>', q: 'Can a new patient be classified or ranked?', t: 'Machine learning: penalised regression, trees, neural networks', s: 'Accuracy on data the model never saw', e: 'response classifier, risk score' },
    { k: '<b>Structure</b>', q: 'What groups or gradients exist without labels?', t: 'Unsupervised learning: clustering, embeddings', s: 'Stability and agreement with known biology', e: 'cell types, spatial domains' },
  ], [['k', 'Question type'], ['q', 'Question'], ['t', 'Typical tool'], ['s', 'Measure of success'], ['e', 'In this tab']]);
  worked(ch, 'Why the verdict matters: 1,933 genes, one t-test each',
    `<p>A Welch t-test on every gene, with no centre term, gives p &lt; 0.05 for <b>${nf(welchP05)}</b> of ${nf(Q.n_genes)} genes, although 5% (about ${Math.round(0.05 * Q.n_genes)}) would pass by chance alone. After false-discovery-rate (FDR) control, ${welch.n_hits} genes are still called, ${welch.by_group.filler[0]} of them among the ${nf(GT.filler)} filler genes that carry no planted response effect. Some filler genes are raised in one cell type, so differences in cell composition between groups make part of these calls real but indirect. A linear model with a centre term calls ${lim.n_hits}. This is an inference problem, so the remedy is a better model of the data, not a more flexible predictor.</p>
     <p>The next problem, building a classifier from the same genes, is a prediction problem with a different remedy: an accuracy estimate that is independent of gene selection (step 4 below).</p>`);

  // =====================================================================================
  // 3. WHERE
  // =====================================================================================
  ch = chapter(main, 'where', 'Where does this design appear?', 'The three-layer structure is common in immunotherapy biomarker research.');
  cards(ch, [
    { tag: 'Trial biomarkers', title: 'Bulk RNA-seq on pre-treatment biopsies', body: 'An IFN-γ-related T-cell-inflamed expression profile associated with response to anti-PD-1 in melanoma and other tumours (Ayers et al., <i>J Clin Invest</i> 2017). The 12-gene module here follows the same logic.' },
    { tag: 'Immune phenotypes', title: 'Inflamed, excluded, desert', body: 'A widely used classification of tumour–immune relationships (Chen &amp; Mellman, <i>Nature</i> 2017). Bulk expression reports T-cell abundance but not location.' },
    { tag: 'Stromal exclusion', title: 'TGF-β and fibroblasts', body: 'In the IMvigor210 urothelial cohort, a TGF-β stromal signature was associated with T cells held outside the tumour and with non-response (Mariathasan et al., <i>Nature</i> 2018).' },
    { tag: 'Multi-centre cohorts', title: 'Batch and case-mix differences', body: 'Samples from different hospitals differ technically and in patient mix. Models need a centre term, and claims need an external cohort.' },
    { tag: 'Single-cell atlases', title: 'Which cells carry the signal?', body: 'Clustering and annotation of tumour dissociates identify T cells, macrophages and fibroblasts; pseudobulk tests compare patients within a cell type.' },
    { tag: 'Spatial transcriptomics', title: 'Where are the T cells?', body: 'Tissue sections give each cell a position, so infiltration into tumour nests can be measured directly instead of inferred from bulk expression.' },
    { tag: 'Prognostic scores', title: 'Risk groups from expression', body: 'Penalised Cox models give risk scores. A control arm is needed to tell a predictive marker from one that is prognostic under any treatment.' },
    { tag: 'Locked signatures', title: 'Validation in a new centre', body: 'A frozen model with a pre-specified test is the standard evidence before a signature is considered for clinical use.' },
  ]);
  prose(ch, '<p><b>When not to use this page as a template:</b> the data here are simulated with a known answer, so accuracy figures describe the simulation, not any real cohort; the cohort is small (a few hundred patients, 12 patients for single-cell), so confidence intervals are wide; and the page describes workflow and tool choice, not a benchmark of methods.</p>');

  // =====================================================================================
  // 4. HOW
  // =====================================================================================
  ch = chapter(main, 'how', 'How it works', 'Six steps: the cohort, the planted biology, the map, the evidence behind the verdicts, the link to Tab 1, and a lab.');

  // ---- Step 1: the cohort as dots
  let rb = rung(ch, 1, 'The discovery cohort',
    `<p>${nSeq} patients were sequenced at two centres; ${nExcl} libraries failed QC (<a href="#/omics/bulk-qc-batch">QC and batch effects</a>), leaving ${nKept}. Each dot below is one patient. Two features matter for every later step: response is uncommon (${pc(nRespB / nKept)}), and centre is entangled with phenotype.</p>`);
  const fig1 = figure(rb, 'The cohort, one dot per patient', 'Dots are grouped by centre and sorted by phenotype, then by response. Bars: observed response rate per phenotype and per centre. Hover for details.');
  let colorBy = 'pheno';
  select(fig1.controls, { label: 'Colour by', value: colorBy, options: [['pheno', 'immune phenotype'], ['resp', 'response to anti-PD-1'], ['centre', 'sequencing centre']], onChange: (v) => { colorBy = v; paint1(); } });
  const leg1 = el('div'); fig1.controls.append(leg1);
  const row1 = el('div', { class: 'fig-row' }); fig1.body.append(row1);
  const wafW = el('div'), barW = el('div'); row1.append(wafW, barW);
  const cols = 14, cell = 17;
  const svg1 = d3.select(wafW).append('svg').attr('viewBox', '0 0 520 186').attr('width', 520).attr('role', 'img').attr('aria-label', 'one dot per patient, grouped by centre');
  const pts = [];
  CENTRES.forEach((c, k) => {
    const ix = idx.filter((i) => B.centre[i] === c).sort((a, b) => PHENOS.indexOf(B.pheno[a]) - PHENOS.indexOf(B.pheno[b]) || respB[b] - respB[a]);
    const x0 = 10 + k * 262;
    svg1.append('text').attr('x', x0).attr('y', 16).attr('fill', ink()).attr('font-size', 13).attr('font-weight', 600).text(`${c}: ${ix.length} patients`);
    ix.forEach((i, j) => pts.push({ i, cx: x0 + 7 + (j % cols) * cell, cy: 36 + Math.floor(j / cols) * cell }));
  });
  const dots = svg1.selectAll('circle').data(pts).join('circle').attr('cx', (d) => d.cx).attr('cy', (d) => d.cy).attr('r', 7).attr('stroke', css('--surface')).attr('stroke-width', 1.2)
    .on('mousemove', (e, d) => tip(e, `<b>${B.ids[d.i]}</b><br>${B.centre[d.i]} · ${B.pheno[d.i]}<br>${respB[d.i] ? 'responder' : 'non-responder'}<br>IFN-γ module score ${B.versions.protected.ifng[d.i].toFixed(2)}`)).on('mouseleave', untip);
  const groups = [...PHENOS.map((p) => ({ k: p, ...byPh[p], col: phenoColor(p) })), ...CENTRES.map((c) => ({ k: c, ...byC[c], col: centreColor(c) }))];
  const F1 = frame(barW, { width: 400, height: 280, x: [0, 1], y: [0, 1], yLabel: 'response rate', xScale: d3.scaleBand().padding(0.3), margin: { bottom: 40 } });
  F1.x.domain(groups.map((g) => g.k));
  F1.g.select('.axis.x').call(d3.axisBottom(F1.x));
  F1.g.select('.axis.y').call(d3.axisLeft(F1.y).tickFormat(d3.format('.0%')));
  F1.layer('bars').selectAll('rect').data(groups).join('rect').attr('x', (d) => F1.x(d.k)).attr('width', F1.x.bandwidth()).attr('y', (d) => F1.y(d.r / d.n)).attr('height', (d) => F1.h - F1.y(d.r / d.n)).attr('rx', 3).attr('fill', (d) => d.col)
    .on('mousemove', (e, d) => tip(e, `<b>${d.k}</b><br>${d.r} of ${d.n} responded (${pc(d.r / d.n)})`)).on('mouseleave', untip);
  F1.layer('lab').selectAll('text').data(groups).join('text').attr('x', (d) => F1.x(d.k) + F1.x.bandwidth() / 2).attr('y', (d) => F1.y(d.r / d.n) - 5).attr('text-anchor', 'middle').attr('font-size', 11).attr('fill', ink2()).text((d) => pc(d.r / d.n));
  F1.layer('sep').append('line').attr('x1', F1.x('Centre A') - 10).attr('x2', F1.x('Centre A') - 10).attr('y1', 0).attr('y2', F1.h).attr('stroke', css('--axis')).attr('stroke-dasharray', '3 3');
  function paint1() {
    const f = { pheno: (i) => phenoColor(B.pheno[i]), resp: (i) => respColor(respB[i]), centre: (i) => centreColor(B.centre[i]) }[colorBy];
    dots.attr('fill', (d) => f(d.i));
    leg1.innerHTML = '';
    legend(leg1, { pheno: PHENOS.map((p) => [p, phenoColor(p)]), resp: [['responder', respColor(1)], ['non-responder', respColor(0)]], centre: CENTRES.map((c) => [c, centreColor(c)]) }[colorBy]);
  }
  paint1();
  prose(rb, `<p>Centre B enrolled more inflamed tumours (${pc(share('Centre B', 'inflamed'))} against ${pc(share('Centre A', 'inflamed'))} in Centre A), so its response rate is higher (${pc(byC['Centre B'].r / byC['Centre B'].n)} against ${pc(byC['Centre A'].r / byC['Centre A'].n)}). This is <b>confounding</b>: any gene that differs between centres can look associated with response. Response is also noisy within phenotypes: ${pc(byPh.inflamed.r / byPh.inflamed.n)} of inflamed tumours respond and ${pc(byPh.desert.r / byPh.desert.n)} of desert tumours do.</p>`);

  // ---- Step 2: the planted biology, seen by differential expression
  rb = rung(ch, 2, 'The planted biology: what differential expression recovers',
    `<p>Three expression modules are planted: IFN-γ (<b>predictive</b>), CAF / TGF-β (exclusion) and proliferation (<b>prognostic</b>). The figure counts how many genes of each module a per-gene test calls significant (q &lt; 0.05), and how that changes when the centre is modelled.</p>`);
  const f2g = figure(rb, 'Genes called significant, by module', 'Each bar is the share of a module\'s genes called significant at q < 0.05 for response. The label gives the count. Choose a method: the last one permutes the response labels, so any call is a false positive.');
  let meth = 'welch';
  select(f2g.controls, { label: 'Method', value: meth, options: Object.entries(M).map(([k, m]) => [k, m.label]), onChange: (v) => { meth = v; paint2(); } });
  const s2 = stats(f2g.root);
  const F2 = frame(f2g.body, { width: 820, height: 290, x: [0, 1], y: GROUP_ORDER.map(groupLabel).reverse(), xLabel: 'share of the module\'s genes called significant', yScale: d3.scaleBand().padding(0.28), grid: false, margin: { left: 150, right: 60 } });
  F2.g.select('.axis.x').call(d3.axisBottom(F2.x).ticks(5).tickFormat(d3.format('.0%')));
  F2.layer('track').selectAll('rect').data(GROUP_ORDER).join('rect').attr('x', 0).attr('width', F2.w).attr('y', (g) => F2.y(groupLabel(g))).attr('height', F2.y.bandwidth()).attr('rx', 3).attr('fill', css('--surface-2'));
  const bars2 = F2.layer('bars').selectAll('rect').data(GROUP_ORDER).join('rect').attr('x', 0).attr('y', (g) => F2.y(groupLabel(g))).attr('height', F2.y.bandwidth()).attr('rx', 3).attr('fill', groupColor)
    .on('mousemove', (e, g) => { const h = M[meth].by_group[g]; tip(e, `<b>${groupLabel(g)}</b><br>${h[0]} of ${h[1]} genes called<br>${M[meth].label}`); }).on('mouseleave', untip);
  const lab2 = F2.layer('lab').selectAll('text').data(GROUP_ORDER).join('text').attr('y', (g) => F2.y(groupLabel(g)) + F2.y.bandwidth() / 2 + 4).attr('font-size', 11.5).attr('fill', ink2());
  F2.layer('note').append('text').attr('x', F2.w - 4).attr('y', F2.y('filler') + F2.y.bandwidth() / 2 + 4).attr('text-anchor', 'end').attr('font-size', 11.5).attr('fill', muted()).text('filler genes: no planted response effect');
  function paint2() {
    const m = M[meth];
    bars2.attr('width', (g) => F2.x(m.by_group[g][0] / m.by_group[g][1]));
    lab2.attr('x', (g) => F2.x(m.by_group[g][0] / m.by_group[g][1]) + 6).text((g) => `${m.by_group[g][0]} / ${nf(m.by_group[g][1])}`);
    s2.set({ 'genes called': m.n_hits, 'in the filler genes': m.by_group.filler[0], 'IFN-γ module genes': `${m.by_group['IFNG module'][0]} / ${GT['IFNG module']}`, 'CAF / TGF-β module genes': `${m.by_group['CAF/TGF-b module'][0]} / ${GT['CAF/TGF-b module']}` });
  }
  paint2();
  prose(rb, `<p>The IFN-γ module is recovered completely by every real method (${lim.by_group['IFNG module'][0]} of ${GT['IFNG module']}) and is the only module whose result does not depend on the centre term. The other modules do depend on it. Without a centre term ${welch.by_group['CAF/TGF-b module'][0]} of ${GT['CAF/TGF-b module']} CAF / TGF-β genes and ${welch.by_group['PROLIF module'][0]} of ${GT['PROLIF module']} proliferation genes pass; with it, ${lim.by_group['CAF/TGF-b module'][0]} and ${lim.by_group['PROLIF module'][0]}. The additional calls largely reflect the centre difference.</p>
    <p><b>The simulation departs from the simple story here.</b> The CAF / TGF-β module is planted to lower the chance of response, yet bulk differential expression with ${nRespB} responders does not detect it once centre is modelled. Its main signature is spatial: it marks excluded tumours, in which T cells are present but sit outside the tumour (<a href="#/omics/sp-biomarker">spatial page</a>). Proliferation has no effect on response by design and appears instead in survival (<a href="#/omics/bulk-survival">survival analysis</a>).</p>`);

  // ---- Step 3: the analysis map
  rb = rung(ch, 3, 'The analysis map and the verdict for each step',
    `<p>The seven analysis pages follow one question each. The map colours each by verdict. Dashed arrows show where the output of one layer feeds another. Click a box to open the page; hover for the reason and the headline result from that page's data.</p>`);
  const f3g = figure(rb, 'The analysis map', 'Colour is the verdict: machine learning, statistics, or both. The line under each title is a headline number from that page. Dashed arrows link layers.');
  let hl = 'all';
  select(f3g.controls, { label: 'Highlight', value: hl, options: [['all', 'all steps'], ['ml', 'machine learning'], ['stats', 'statistics'], ['both', 'ML + statistics']], onChange: (v) => { hl = v; paint3(); } });
  legend(f3g.controls, Object.entries(VLABEL).map(([k, t]) => [t, vcol(k)]));
  const s3 = stats(f3g.root);
  const MW = 820, colW = 250, gap = 35, bh = 62, gy = 14, top = 30, MH = top + 4 * (bh + gy);
  const svg3 = d3.select(f3g.body).append('svg').attr('viewBox', `0 0 ${MW} ${MH}`).attr('width', MW).attr('role', 'img').attr('aria-label', 'map of the seven analysis pages');
  svg3.append('defs').append('marker').attr('id', 'ov-arrow').attr('viewBox', '0 0 10 10').attr('refX', 9).attr('refY', 5).attr('markerWidth', 7).attr('markerHeight', 7).attr('orient', 'auto').append('path').attr('d', 'M0,0 L10,5 L0,10 z').attr('fill', muted());
  ['Bulk RNA-seq', 'Single-cell RNA-seq', 'Spatial omics'].forEach((t, k) => svg3.append('text').attr('x', k * (colW + gap)).attr('y', 18).attr('font-weight', 700).attr('font-size', 14).attr('fill', ink()).text(t));
  PAGES.forEach((p) => { p.x = p.lane * (colW + gap); p.y = top + p.row * (bh + gy); });
  [['bulk-qc-batch', 'bulk-de'], ['bulk-de', 'bulk-classifier'], ['bulk-classifier', 'bulk-survival'], ['sc-clustering', 'sc-pseudobulk']].forEach(([a, b]) => svg3.append('line').attr('x1', PG[a].x + 20).attr('x2', PG[a].x + 20).attr('y1', PG[a].y + bh).attr('y2', PG[b].y).attr('stroke', css('--axis')).attr('stroke-width', 1.5));
  const LINKS = [['bulk-de', 'sc-pseudobulk', 'the same question (response) asked within each cell type'], ['bulk-classifier', 'sp-biomarker', 'the bulk signature is re-examined in space'], ['sc-clustering', 'sp-biomarker', 'single-cell types give the reference for spatial deconvolution']];
  LINKS.forEach(([a, b, why]) => {
    const A = PG[a], Bn = PG[b], ax = A.x + colW, bx = Bn.x, ay = A.y + bh / 2, by = Bn.y + bh / 2, mx = (ax + bx) / 2;
    svg3.append('path').attr('d', `M${ax},${ay} C${mx},${ay} ${mx},${by} ${bx},${by}`).attr('fill', 'none').attr('stroke', muted()).attr('stroke-width', 1.5).attr('stroke-dasharray', '5 4').attr('marker-end', 'url(#ov-arrow)').append('title').text(why);
  });
  svg3.append('text').attr('x', 2 * (colW + gap) - gap - 150).attr('y', PG['bulk-classifier'].y + bh / 2 - 8).attr('text-anchor', 'middle').attr('font-size', 11.5).attr('fill', muted()).text('bulk signature re-examined in space');
  const g3 = svg3.append('g').selectAll('a').data(PAGES).join('a').attr('href', (d) => `#/omics/${d.id}`);
  g3.append('rect').attr('class', 'box').attr('x', (d) => d.x).attr('y', (d) => d.y).attr('width', colW).attr('height', bh).attr('rx', 8).attr('stroke-width', 2).attr('fill', css('--surface-2'));
  g3.append('rect').attr('class', 'bar').attr('x', (d) => d.x).attr('y', (d) => d.y).attr('width', 6).attr('height', bh).attr('rx', 3);
  g3.append('text').attr('x', (d) => d.x + 16).attr('y', (d) => d.y + 21).attr('font-size', 11).attr('font-weight', 600).attr('fill', ink()).text((d) => d.title);
  g3.append('text').attr('x', (d) => d.x + 16).attr('y', (d) => d.y + 40).attr('font-size', 10.5).attr('fill', ink2()).text((d) => d.head);
  g3.append('text').attr('x', (d) => d.x + 16).attr('y', (d) => d.y + 55).attr('font-size', 10.5).attr('fill', muted()).text((d) => VLABEL[d.verdict]);
  g3.on('mousemove', (e, d) => tip(e, `<b>${d.title}</b><br>${d.q}<br>${pill(d.verdict)} ${d.why}<br><i>${d.long}</i>`)).on('mouseleave', untip);
  function paint3() {
    g3.attr('opacity', (d) => (hl === 'all' || d.verdict === hl ? 1 : 0.25));
    g3.select('rect.box').attr('stroke', (d) => vcol(d.verdict));
    g3.select('rect.bar').attr('fill', (d) => vcol(d.verdict));
    s3.set({ pages: PAGES.length, 'machine learning': cnt('ml'), statistics: cnt('stats'), 'ML + statistics': cnt('both') });
  }
  paint3();
  table(rb, PAGES.map((p) => ({ ...p, link: `<a href="#/omics/${p.id}">${p.title}</a>`, v: pill(p.verdict) })), [['link', 'Page'], ['q', 'Question'], ['v', 'Verdict'], ['why', 'Why']], { cls: 'compact' });
  prose(rb, `<p>The verdict follows from the output of the step. A <b>probability, a ranking or a label for a new sample</b> is a prediction (ML). A <b>p-value, an effect size with an interval or an error rate</b> is inference (statistics). A <b>grouping without labels</b> is structure (unsupervised ML). Pages marked "ML + statistics" contain both, usually a fitted model followed by a statistical estimate of how well it works.</p>`);
  quiz(rb, 'Differential expression and the response classifier use the same 1,933 genes and the same patients. Why are their verdicts different?',
    `<p>Differential expression asks whether a gene's difference is real and reports an error rate (${lim.n_hits} genes at q &lt; 0.05); nothing is predicted for a new patient. The classifier outputs a probability for a new patient and is judged by accuracy on unseen data (a locked 2-gene model scores AUC ${f2(VA.strategies.frozen.auc)} in Centre C), which makes it machine learning with a statistical accuracy estimate.</p>`);

  // ---- Step 4: evidence for the ML verdicts (leaky vs nested)
  rb = rung(ch, 4, 'What an ML verdict adds: an honest accuracy estimate',
    `<p>Where a step is machine learning, the main risk is an over-optimistic accuracy estimate. Choosing the best <i>k</i> genes using all ${CL.n} patients and then cross-validating the classifier lets the held-out folds influence the selection (<b>leakage</b>). <b>Nested</b> cross-validation repeats the selection inside each training fold. Permuting the response labels removes all signal, so an accuracy consistently well above 0.5 on permuted labels is leakage.</p>`);
  const f4 = figure(rb, 'Cross-validated AUC against the number of selected genes', `Mean over ${CL.reps} repeats; the band is ± 1 SD. Choose real or permuted labels. The dashed line is chance (AUC 0.5).`);
  let lab4 = 'real';
  select(f4.controls, { label: 'Labels', value: lab4, options: [['real', 'real response labels'], ['perm', 'permuted labels (no signal)']], onChange: (v) => { lab4 = v; paint4(); } });
  const s4 = stats(f4.root);
  const ks = CL.ks;
  const F4 = frame(f4.body, { width: 820, height: 300, x: [5, 200], y: [0.4, 1], xLabel: 'genes selected (k)', yLabel: 'cross-validated AUC', xScale: d3.scaleLog(), margin: { right: 110 } });
  F4.g.select('.axis.x').call(d3.axisBottom(F4.x).tickValues(ks).tickFormat(d3.format('d')));
  F4.layer('ref').append('line').attr('x1', 0).attr('x2', F4.w).attr('y1', F4.y(0.5)).attr('y2', F4.y(0.5)).attr('stroke', muted()).attr('stroke-dasharray', '4 4');
  const SER4 = [['leaky', cat(3), 'selection on all data (leaky)'], ['nested', cat(2), 'selection inside each fold (nested)']];
  const bandL = F4.layer('band'), lineL = F4.layer('line'), dotL = F4.layer('dots'), labL = F4.layer('lab');
  function paint4() {
    const data = SER4.map(([m, c, t]) => ({ m, c, t, pts: ks.map((k) => ({ k, v: CL.res[`${lab4}|${m}|${k}`].mean, s: CL.res[`${lab4}|${m}|${k}`].sd })) }));
    bandL.selectAll('path').data(data).join('path').attr('fill', (d) => d.c).attr('fill-opacity', 0.16).attr('d', (d) => d3.area().x((q) => F4.x(q.k)).y0((q) => F4.y(q.v - q.s)).y1((q) => F4.y(q.v + q.s))(d.pts));
    lineL.selectAll('path').data(data).join('path').attr('fill', 'none').attr('stroke', (d) => d.c).attr('stroke-width', 2.4).attr('d', (d) => d3.line().x((q) => F4.x(q.k)).y((q) => F4.y(q.v))(d.pts));
    dotL.selectAll('circle').data(data.flatMap((d) => d.pts.map((q) => ({ ...q, c: d.c, t: d.t })))).join('circle').attr('cx', (q) => F4.x(q.k)).attr('cy', (q) => F4.y(q.v)).attr('r', 4.2).attr('fill', (q) => q.c).attr('stroke', css('--surface')).attr('stroke-width', 1.2)
      .on('mousemove', (e, q) => tip(e, `${q.t}<br>k = ${q.k}: AUC ${f3(q.v)} ± ${f3(q.s)}`)).on('mouseleave', untip);
    labL.selectAll('text').data(data).join('text').attr('x', F4.w + 6).attr('y', (d) => F4.y(d.pts[d.pts.length - 1].v) + 4).attr('font-size', 11.5).attr('fill', ink2()).text((d) => d.m === 'leaky' ? 'leaky' : 'nested');
    s4.set({ 'leaky, k = 5': f3(CL.res[`${lab4}|leaky|5`].mean), 'leaky, k = 200': f3(CL.res[`${lab4}|leaky|200`].mean), 'nested, k = 5': f3(CL.res[`${lab4}|nested|5`].mean), 'nested, k = 200': f3(CL.res[`${lab4}|nested|200`].mean) });
  }
  paint4();
  prose(rb, `<p>With real labels the two estimates agree at small <i>k</i> (${f2(cl('leaky', 5))} and ${f2(cl('nested', 5))} at k = 5) and separate as more genes are chosen: ${f2(cl('leaky', 200))} leaky against ${f2(cl('nested', 200))} nested at k = 200. With permuted labels the nested estimate stays near chance (${f2(pm('nested', 20))} at k = 20), whereas the leaky one reaches ${f2(pm('leaky', 20))}. Leakage is therefore small when few, strong genes are used and large when many genes are screened, and it is invisible without a permutation check.</p>`);

  // ---- Step 5: two draws of the same study
  const hasPh = (a, p) => a.filter((q) => q.phenotype === p).length / a.length;
  const infl2 = (mx) => mx.inflamed / sum(Object.values(mx));
  rb = rung(ch, 5, 'Two draws of the same fictional study',
    `<p>Tab 1 and Tab 2 describe the <b>same fictional study</b> at two levels, but they are <b>separate simulated draws</b>, not the same patients. Tab 1 generates a small patient-level table (a few summary scores per patient) so that the concepts can be taught on one dataset. Tab 2 generates gene-level data (counts for ${nf(Q.n_genes)} genes), single-cell and spatial layers. Identifiers differ (${D1[0].id} … in Tab 1, ${B.ids[0]} … here) and rows must not be matched.</p>`);
  table(rb, [
    { s: 'Tab 1 discovery', n: D1.length, c: `A ${D1.filter((p) => p.centre === 'Centre A').length} · B ${D1.filter((p) => p.centre === 'Centre B').length}`, r: `${sum(D1.map((p) => p.response))} (${pc(mean(D1.map((p) => p.response)))})`, i: pc(hasPh(D1, 'inflamed')), d: 'summary scores per patient' },
    { s: 'Tab 1 validation', n: V1.length, c: 'C 120', r: `${sum(V1.map((p) => p.response))} (${pc(mean(V1.map((p) => p.response)))})`, i: pc(hasPh(V1, 'inflamed')), d: 'summary scores per patient' },
    { s: 'Tab 2 discovery', n: `${nKept} (${nSeq} sequenced)`, c: `A ${byC['Centre A'].n} · B ${byC['Centre B'].n}`, r: `${nRespB} (${pc(nRespB / nKept)})`, i: pc(infl2(VA.mix_d)), d: `${nf(Q.n_genes)} genes` },
    { s: 'Tab 2 validation', n: VA.n_v, c: 'C 120', r: `${nValR} (${pc(nValR / VA.n_v)})`, i: pc(infl2(VA.mix_v)), d: `${nf(Q.n_genes)} genes` },
  ], [['s', 'Data set'], ['n', 'Patients'], ['c', 'By centre'], ['r', 'Responders'], ['i', 'Inflamed share'], ['d', 'Measurements']], { cls: 'compact' });
  prose(rb, `<p>The two tabs are separate simulations with the same main signals: in both, IFN-γ activity is the main predictor of response (+) and CAF / TGF-β stromal exclusion works against it (−), proliferation is prognostic only, and centre is confounded with phenotype. Only Tab 1 adds tumour mutational burden (+); Tab 2 drives response through immune phenotype, adjusted by IFN-γ. The cohorts differ in size (${D1.length} against ${nKept}), in response rate (${pc(mean(D1.map((p) => p.response)))} against ${pc(nRespB / nKept)}) and in case mix. Qualitative findings should carry across, while numbers such as AUCs and counts of significant genes should be compared only with that sampling variation in mind. Tab 1 has no single-cell or spatial layer and no per-gene data; Tab 2 has no TMB or PD-L1 column.</p>`);

  // ---- Step 6: lab
  rb = rung(ch, 6, 'Lab: does a marker behave the same in both draws?',
    `<p>A one-marker classifier is the simplest test of the shared truth. Patients are ranked by one score and the top fraction is called "responder". The curve (ROC) traces the share of responders caught (true-positive rate, TPR) against the share of non-responders wrongly called (false-positive rate, FPR) over all possible fractions. There is no fitting: Tab 1 validation patients are scored with the same marker, not a refitted model.</p>`);
  const SCORES = {
    ifng: { lab: 'IFN-γ score', t1: (p) => p.ifng, t2: SV.io.ifng },
    prolif: { lab: 'Proliferation score', t1: (p) => p.prolif, t2: SV.io.prolif },
    caf: { lab: 'CAF / TGF-β score', t1: (p) => p.caf, t2: null },
  };
  const SETS = [['Tab 1 discovery', 6, D1.map((p) => p.response), (k) => D1.map(SCORES[k].t1)], ['Tab 1 validation', 3, V1.map((p) => p.response), (k) => V1.map(SCORES[k].t1)], ['Tab 2 discovery', 7, SV.io.resp, (k) => SCORES[k].t2]]
    .map(([name, slot, y, get]) => ({ name, slot, y, get }));
  const evalSet = (set, k, f) => {
    const s = set.get(k); if (!s) return null;
    const o = s.map((v, i) => i).sort((a, b) => s[b] - s[a]), m = Math.max(1, Math.round(f * s.length)), top = new Set(o.slice(0, m));
    const P = sum(set.y), N = set.y.length - P, tp = [...top].filter((i) => set.y[i]).length;
    return { auc: auc(s, set.y), tpr: tp / P, fpr: (top.size - tp) / N, roc: roc(s, set.y), tp, P, fp: top.size - tp, N };
  };
  const fl = figure(rb, 'One marker, three data sets', 'Each curve is one data set. The dot marks the chosen fraction of patients called responders. The diagonal is chance. Hover a dot for counts.');
  let sk = 'ifng', frac = 0.25;
  select(fl.controls, { label: 'Marker', value: sk, options: Object.entries(SCORES).map(([k, v]) => [k, v.lab]), onChange: (v) => { sk = v; paintL(); } });
  slider(fl.controls, { label: 'fraction called responder', min: 0.05, max: 0.8, step: 0.01, value: frac, fmt: (v) => pc(v), onInput: (v) => { frac = v; paintL(); } });
  const sL = stats(fl.root);
  const FL = frame(fl.body, { width: 560, height: 380, x: [0, 1], y: [0, 1], xLabel: 'false-positive rate (non-responders called)', yLabel: 'true-positive rate (responders caught)' });
  FL.layer('diag').append('line').attr('x1', FL.x(0)).attr('y1', FL.y(0)).attr('x2', FL.x(1)).attr('y2', FL.y(1)).attr('stroke', muted()).attr('stroke-dasharray', '4 4');
  const curveL = FL.layer('curves'), dotLL = FL.layer('dots'), noteL = FL.layer('note');
  legend(fl.body, SETS.map((s) => [s.name, cat(s.slot)]));
  function paintL() {
    const ev = SETS.map((s) => ({ s, e: evalSet(s, sk, frac) })).filter((d) => d.e);
    curveL.selectAll('path').data(ev).join('path').attr('fill', 'none').attr('stroke', (d) => cat(d.s.slot)).attr('stroke-width', 2.3).attr('d', (d) => d3.line().x((q) => FL.x(q.fpr)).y((q) => FL.y(q.tpr))(d.e.roc));
    dotLL.selectAll('circle').data(ev).join('circle').attr('cx', (d) => FL.x(d.e.fpr)).attr('cy', (d) => FL.y(d.e.tpr)).attr('r', 6).attr('fill', (d) => cat(d.s.slot)).attr('stroke', css('--surface')).attr('stroke-width', 1.6)
      .on('mousemove', (e, d) => tip(e, `<b>${d.s.name}</b><br>${d.e.tp} of ${d.e.P} responders caught<br>${d.e.fp} of ${d.e.N} non-responders called`)).on('mouseleave', untip);
    noteL.selectAll('text').data(SCORES[sk].t2 ? [] : [0]).join('text').attr('x', FL.x(0.98)).attr('y', FL.y(0.08)).attr('text-anchor', 'end').attr('font-size', 11.5).attr('fill', muted()).text('Tab 2: no per-patient CAF score in this data');
    sL.set(Object.fromEntries(ev.map((d) => [d.s.name, `AUC ${f2(d.e.auc)} · TPR ${pc(d.e.tpr)} · FPR ${pc(d.e.fpr)}`])));
  }
  paintL();
  const E = (k, f = 0.25) => Object.fromEntries(SETS.map((s) => [s.name, evalSet(s, k, f)]));
  const eI = E('ifng'), eP = E('prolif'), eC = E('caf');
  const rng3 = (e) => { const a = Object.values(e).filter(Boolean).map((v) => v.auc); return `${f2(d3.min(a))}–${f2(d3.max(a))}`; };
  table(rb, [
    { t: 'IFN-γ, fraction 25%', s: `Responders caught ${Object.values(eI).map((v) => pc(v.tpr)).join(' / ')}; non-responders called ${Object.values(eI).map((v) => pc(v.fpr)).join(' / ')} (order: Tab 1 discovery, Tab 1 validation, Tab 2 discovery). AUC ${rng3(eI)}`, m: 'The predictive marker separates the groups in all three data sets, with similar but not identical values. The curves differ by sampling variation and by how each score is built.' },
    { t: 'Proliferation', s: `AUC ${Object.values(eP).map((v) => f2(v.auc)).join(' / ')}`, m: `Not a response marker by design. Values below 0.5 mean higher scores accompany non-response, and it is not a planted effect on response. In Tab 2 the association comes through tumour purity: proliferation genes are expressed only in tumour cells, and desert tumours have the most tumour cells. Its role is prognostic (<a href="#/omics/bulk-survival">survival page</a>).` },
    { t: 'CAF / TGF-β', s: `AUC ${f2(eC['Tab 1 discovery'].auc)} / ${f2(eC['Tab 1 validation'].auc)} (Tab 1 only)`, m: 'Lower scores accompany response, as planted. Tab 2 expresses the same programme as a gene module, tested in differential expression (step 2), not as a patient score.' },
    { t: 'Fraction 5% → 80%', s: `Dots move up and right along each curve`, m: 'A larger fraction catches more responders and wrongly calls more non-responders. The fraction, like a probability threshold, is a decision rather than a property of the marker.' },
  ], [['t', 'Try'], ['s', 'What you see'], ['m', 'What it means']], { cls: 'compact' });

  // =====================================================================================
  // 5. PITFALLS
  // =====================================================================================
  ch = chapter(main, 'pitfalls', 'Pitfalls', 'Common problems when moving through a multi-layer study, how to recognise them, and how to address them.');
  cards(ch, [
    { tag: 'Symptom: rows from Tab 1 and Tab 2 do not line up', title: 'Treating the two cohorts as one', body: `Tab 1 has ${D1.length + V1.length} patients and Tab 2 ${nKept + VA.n_v}; they are separate draws with different response rates (${pc(mean(D1.map((p) => p.response)))} and ${pc(nRespB / nKept)} in discovery). <b>Fix:</b> compare findings, not rows.` },
    { tag: 'Symptom: "we used ML" for every step', title: 'Choosing the tool before the question', body: `Differential expression, QC and pseudobulk tests are inference; a flexible predictor would not give an error rate. <b>Fix:</b> state the output (p-value, probability, grouping) first. ${cnt('stats')} of the ${PAGES.length} pages are plain statistics.` },
    { tag: 'Symptom: hundreds of significant genes', title: 'Confounding by centre', body: `With no centre term, ${welch.n_hits} genes are called against ${lim.n_hits} with it, including ${welch.by_group.filler[0]} unplanted filler genes. <b>Fix:</b> model the batch, and check that the correction does not remove biology (step 2, QC page).` },
    { tag: 'Symptom: accuracy drops in a new centre', title: 'Leakage and selection', body: `Selecting genes on all patients gives AUC ${f2(cl('leaky', 200))} at k = 200 against ${f2(cl('nested', 200))} nested, and ${f2(pm('leaky', 20))} with permuted labels at k = 20. <b>Fix:</b> every data-driven step goes inside the resampling loop; run a permutation check.` },
    { tag: 'Symptom: tiny p-values from thousands of cells', title: 'Cells as replicates', body: `Cells from one patient are correlated. Treating them as independent gives ${fpNv} false calls across ${PBT.length} cell types, against ${fpPb} for pseudobulk. <b>Fix:</b> aggregate to patients and test at that level.` },
    { tag: 'Symptom: a risk score that "predicts" benefit', title: 'Prognostic is not predictive', body: `Proliferation raises the hazard on anti-PD-1 (HR ${f2(hr.prolif.hr_io)}) and in the docetaxel arm (${f2(hr.prolif.hr_ctrl)}), so it does not indicate benefit from the drug. IFN-γ lowers it only on treatment (${f2(hr.ifng.hr_io)} against ${f2(hr.ifng.hr_ctrl)}). <b>Fix:</b> test the marker-by-treatment interaction.` },
    { tag: 'Symptom: bulk score misses exclusion', title: 'Expecting bulk data to show location', body: `Separating inflamed from excluded tumours gives AUC ${f2(ap.ifn_bulk.inf_vs_exc)} for bulk IFN-γ and ${f2(ap.infil.inf_vs_exc)} for the spatial infiltration ratio (near-circular by construction, since the simulation defines phenotype by infiltration). For <i>response</i>, however, the spatial inside-share scores only ${f2(ap.frac_inside.resp)}, below the simple CD8 fraction (${f2(ap.cd8_frac.resp)}). <b>Fix:</b> match the feature to the question, and test it rather than assume spatial is better.` },
    { tag: 'Symptom: every method works', title: 'Mistaking a simulation for evidence', body: 'A planted truth makes methods look reliable. Real data has no answer key, unmeasured confounders and messier batches. <b>Fix:</b> keep the permutation, nested and external checks even when they appear unnecessary.' },
  ]);

  // =====================================================================================
  // 6. RECAP
  // =====================================================================================
  ch = chapter(main, 'recap', 'Recap', 'Key points.');
  recap(ch, [
    `One fictional study at three scales: <b>bulk</b> (${nKept} patients after QC, ${nf(Q.n_genes)} genes), <b>single-cell</b> (${SCI.n_patients} patients) and <b>spatial</b> (${SPS.sections.length} tissue sections, ${SPB.n} section summaries).`,
    'The planted biology: <b>IFN-γ</b> is predictive, <b>CAF / TGF-β</b> marks T-cell exclusion, <b>proliferation</b> is prognostic only, and <b>centre</b> is a batch partly confounded with phenotype.',
    `The map has ${cnt('stats')} statistics pages, ${cnt('ml')} machine-learning page and ${cnt('both')} that need both. The verdict follows from the output: p-value (statistics), prediction (ML), grouping (unsupervised ML).`,
    `Modelling the centre cuts significant genes from ${welch.n_hits} to ${lim.n_hits}; the IFN-γ module is recovered either way.`,
    'Machine-learning steps need an honest accuracy estimate: nested resampling, a permutation check and an external cohort.',
    `Tab 1 (${D1.length} + ${V1.length} patients) and Tab 2 (${nKept} + ${VA.n_v}) share the main signals but are separate simulations; compare findings, not rows or exact values.`,
    'Where the simulation disagrees with the simple story (CAF not detected in bulk; spatial inside-share weak for response), the disagreement is reported rather than tuned away.',
  ]);
  prose(ch, '<h3>In code</h3><p>The same checks apply to any cohort with several data layers: compare an agreed score across cohorts, and keep the centre as a covariate when testing genes. The snippet uses pandas and a per-gene linear model.</p>');
  code(ch, {
    Python: `import pandas as pd, statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score

# 1) The same marker in every cohort draw: compare findings, not rows
for name, df in {"discovery": disc, "validation": val}.items():
    print(name, len(df), df.response.mean().round(2), roc_auc_score(df.response, df.ifng).round(3))

# 2) Inference per gene, with the centre as a covariate (statistics)
res = []
for g in expr.columns:
    d = meta.assign(y=expr[g])
    fit = smf.ols("y ~ response + C(centre)", data=d).fit()
    res.append((g, fit.params["response"], fit.pvalues["response"]))
tab = pd.DataFrame(res, columns=["gene", "lfc", "p"])
# Benjamini-Hochberg: statsmodels.stats.multitest.multipletests(tab.p, method="fdr_bh")`,
    R: `# 1) The same marker in every cohort draw
sapply(list(discovery = disc, validation = val),
       function(d) pROC::auc(d$response, d$ifng))

# 2) Inference per gene, with the centre as a covariate (statistics)
library(limma)
design <- model.matrix(~ response + centre, data = meta)
fit <- eBayes(lmFit(expr_matrix, design))      # genes x samples
topTable(fit, coef = "response", number = Inf, adjust.method = "BH")`,
  });
  seeAlso(main, [['omics', 'bulk-qc-batch', 'QC and batch effects'], ['omics', 'bulk-de', 'Differential expression'], ['omics', 'bulk-classifier', 'Classifier and nested CV'], ['omics', 'bulk-survival', 'Survival and risk scores'], ['omics', 'sc-clustering', 'Single-cell clustering'], ['omics', 'sc-pseudobulk', 'Pseudobulk DE'], ['omics', 'sp-biomarker', 'Spatial biomarker'], ['topics', 'logistic-regression', 'Logistic regression (Tab 1)']]);
}

// ---------------------------------------------------------------------------------------------
/** The three-layer design diagram. Sizes are passed in from the data files. */
function designSvg(n) {
  const wrap = el('div', { class: 'figure' });
  wrap.append(el('div', { class: 'fig-title' }, 'The study design: three data layers'));
  const svg = d3.select(wrap).append('svg').attr('viewBox', '0 0 820 250').attr('width', 820).attr('role', 'img').attr('aria-label', 'bulk, single-cell and spatial data layers');
  const boxes = [
    { t: 'Bulk RNA-seq', k: 'which genes, can they predict?', lines: [`${n.nSeq} biopsies sequenced`, `${n.nKept} after QC (centres A, B)`, `${n.nRespB} responders`, `${nf(n.nGenes)} genes`, `Validation: ${n.nVal} patients,`, `Centre C (${n.nValR} responders)`, `Docetaxel control arm: ${n.nCtrl}`] },
    { t: 'Single-cell RNA-seq', k: 'which cells carry the signal?', lines: [`${nf(n.nDroplets)} droplets`, `${n.nPat} patients, 2 chemistries`, `${n.nTypes} cell types`, 'Replicate unit: the patient', '', ''] },
    { t: 'Spatial transcriptomics', k: 'where are the T cells?', lines: [`${n.nSec} tissue sections`, `${n.nPanel}-gene panel`, 'Inflamed, excluded and desert', `${n.nSpC} section summaries`, `(${n.nSpR} from responders)`, ''] },
  ];
  const BW = 250, gap = 35;
  boxes.forEach((b, i) => {
    const x0 = i * (BW + gap), g = svg.append('g').attr('transform', `translate(${x0},8)`);
    g.append('rect').attr('width', BW).attr('height', 232).attr('rx', 10).attr('fill', css('--surface-2')).attr('stroke', css('--axis'));
    g.append('rect').attr('width', 6).attr('height', 232).attr('rx', 3).attr('fill', cat(i === 0 ? 6 : i === 1 ? 2 : 3));
    g.append('text').attr('x', 18).attr('y', 28).attr('font-weight', 700).attr('font-size', 14).attr('fill', ink()).text(b.t);
    g.append('text').attr('x', 18).attr('y', 48).attr('font-size', 12).attr('font-style', 'italic').attr('fill', muted()).text(b.k);
    b.lines.forEach((l, j) => g.append('text').attr('x', 18).attr('y', 78 + j * 24).attr('font-size', 12.5).attr('fill', ink2()).text(l));
    if (i < boxes.length - 1) {
      const ax = x0 + BW + 5, ay = 8 + 116;
      svg.append('path').attr('d', `M${ax},${ay} L${ax + gap - 12},${ay} M${ax + gap - 18},${ay - 6} L${ax + gap - 12},${ay} L${ax + gap - 18},${ay + 6}`).attr('fill', 'none').attr('stroke', ink2()).attr('stroke-width', 2);
    }
  });
  return wrap;
}
