// Tab 2 overview: the mock anti-PD-1 NSCLC study, its three data layers, and a clickable map of every step
// coloured by "ML, statistics or both". Numbers come from data/omics/bulk_qc.json and bulk_batch.json.
import { header, figure, select, stats, section, steps, callout, legend, seeAlso, tip, untip, el } from '../lib/ui.js';
import { frame, css, ink, ink2, muted } from '../lib/plot.js';
import { rng } from '../lib/data.js';
import { TABS } from '../registry.js';
import { stepCard, omicsData } from './_shared.js';
import { PHENOS, phenoColor, CENTRES, centreColor, respColor, pct, f3 } from './_bulk_common.js';
const d3 = window.d3;

// Study design constants, mirrored from DESIGN in scripts/mock_biology.py (the single-cell and spatial layers
// are described here by design; their own pages load their own data).
const DESIGN = { validation_n: 120, validation_centre: 'Centre C', sc_patients: 12, sc_cells: 6000, sp_sections: 4 };

// Verdict for each step: is this a machine-learning problem, a statistics problem, or both? One-line why.
const VERDICT = {
  overview: ['both', 'The whole study: statistics for inference, ML for prediction and structure-finding.'],
  'bulk-qc': ['both', 'MAD thresholds are statistics; multivariate outlier scores (robust distance, isolation forest) are unsupervised ML.'],
  'bulk-batch': ['stats', 'PCA to look, then a linear model (ComBat-style) to remove the centre effect.'],
  'bulk-subtypes': ['ml', 'Clustering is unsupervised learning; consensus and silhouette check it is stable.'],
  'bulk-de': ['stats', 'Per-gene inference with error control. The goal is a p-value, not a prediction.'],
  'bulk-feature-selection': ['ml', 'Penalised regression (lasso / elastic net) chooses a sparse predictor.'],
  'bulk-classifier': ['ml', 'Supervised prediction; the hard part is estimating accuracy honestly.'],
  'bulk-model-compare': ['both', 'The candidates are ML models; choosing between them (paired folds, uncertainty, calibration) is statistics.'],
  'bulk-explain': ['ml', 'Attributing a model prediction to its inputs.'],
  'bulk-survival': ['both', 'Cox model and Kaplan–Meier are statistics; the lasso penalty is ML.'],
  'bulk-deconvolution': ['stats', 'Constrained regression (NNLS) against a reference profile.'],
  'bulk-validation': ['stats', 'A frozen model, a pre-specified test and a confidence interval.'],
  'sc-qc': ['both', 'MAD thresholds for QC; a classifier for doublets.'],
  'sc-embedding': ['both', 'Count normalisation and HVG selection are statistics; PCA is unsupervised.'],
  'sc-clustering': ['ml', 'Graph community detection and UMAP are unsupervised ML.'],
  'sc-integration': ['ml', 'Harmony / scVI-style models learn a batch-free space.'],
  'sc-annotation': ['ml', 'Label transfer is classification; marker scoring is the simple baseline.'],
  'sc-trajectory': ['both', 'The ordering is unsupervised ML; claims about genes or groups along it are statistics, replicated by patient.'],
  'sc-pseudobulk': ['stats', 'Back to patient-level linear models: cells are not replicates.'],
  'sp-data': ['stats', 'Building a spatial neighbour graph is geometry, not learning.'],
  'sp-svg': ['stats', "Moran's I with a permutation null."],
  'sp-domains': ['ml', 'Clustering with spatial smoothing.'],
  'sp-deconvolution': ['both', 'A probabilistic regression model, fitted with ML-style optimisation.'],
  'sp-niches': ['stats', 'Permutation tests for neighbourhood enrichment and ligand–receptor scores.'],
  'sp-biomarker': ['both', 'A spatial feature tested as a biomarker, with the same honesty rules.'],
};
const VLABEL = { ml: 'Machine learning', stats: 'Statistics', both: 'ML + statistics' };
const vcol = (v) => css({ ml: '--good', stats: '--accent', both: '--warn' }[v]);

export default async function render(main) {
  header(main, 'The mock study & ML map',
    'This tab follows <b>one fictional anti-PD-1 lung cancer study</b> from raw counts to a biomarker. Every step asks a biological question first, then picks the tool. Often that tool is <b>plain statistics, not machine learning</b>, and knowing which is which is most of the skill.');
  stepCard(main, {
    stage: 'The study',
    question: 'Which patients with non-small-cell lung cancer (NSCLC) respond to anti-PD-1 immunotherapy, and why?',
    data: 'A bulk RNA-seq discovery cohort, an external validation cohort, single-cell RNA-seq on a subset, and spatial transcriptomics on four tumour sections. All simulated from one shared ground truth.',
    method: 'The whole pipeline: QC → exploration → subtypes → differential expression → biomarker → validation, then single-cell and spatial follow-up.',
    verdict: 'both', why: ' Statistics answers "is this difference real?"; ML answers "can I predict a new patient?" and "what structure is in here?".',
  });

  const qc = await omicsData(main, 'bulk_qc.json');
  const B = await omicsData(main, 'bulk_batch.json');
  if (!qc || !B) return;

  const nDisc = qc.ids.length, nKept = B.ids.length, nExcl = qc.excluded.length;
  const disc = { A: qc.centre.filter((c) => c === 'Centre A').length, B: qc.centre.filter((c) => c === 'Centre B').length };
  const rr = (filt) => { const ix = B.resp.map((r, i) => i).filter(filt); return { n: ix.length, r: ix.filter((i) => B.resp[i]).length }; };
  const byPh = Object.fromEntries(PHENOS.map((p) => [p, rr((i) => B.pheno[i] === p)]));
  const byC = Object.fromEntries(CENTRES.map((c) => [c, rr((i) => B.centre[i] === c)]));
  const allR = rr(() => true);
  const share = (c, p) => B.counts[c][p] / Object.values(B.counts[c]).reduce((a, b) => a + b, 0);

  // ------------------------------------------------------------------ figure 1: the cohort as dots
  const fig1 = figure(main, 'The discovery cohort, one dot per patient',
    `${nDisc} patients were sequenced at two centres; ${nExcl} failed QC (<a href="#/omics/bulk-qc">next page</a>), leaving ${nKept}. Each dot is a patient that passed QC. Hover for details.`);
  let colorBy = 'pheno';
  select(fig1.controls, { label: 'Colour by', value: 'pheno', options: [['pheno', 'immune phenotype'], ['resp', 'response to anti-PD-1'], ['centre', 'sequencing centre']], onChange: (v) => { colorBy = v; paint1(); } });
  const leg1 = el('div'); fig1.controls.append(leg1);
  const rd1 = stats(fig1.root);
  const row1 = el('div', { class: 'fig-row' }); fig1.body.append(row1);
  const wafW = el('div'), barW = el('div'); row1.append(wafW, barW);

  // waffle: one block per centre, sorted by phenotype then response so the composition is visible
  const W = 520, cols = 14, cell = 17, H = 186;
  const svg1 = d3.select(wafW).append('svg').attr('viewBox', `0 0 ${W} ${H}`).attr('width', W);
  const pts = [];
  CENTRES.forEach((c, ci) => {
    const ix = B.ids.map((_, i) => i).filter((i) => B.centre[i] === c)
      .sort((a, b) => PHENOS.indexOf(B.pheno[a]) - PHENOS.indexOf(B.pheno[b]) || B.resp[b] - B.resp[a]);
    const x0 = 10 + ci * 262;
    svg1.append('text').attr('x', x0).attr('y', 16).attr('fill', ink()).attr('font-size', 13).attr('font-weight', 600)
      .text(`${c}: ${ix.length} kept`);
    ix.forEach((i, k) => pts.push({ i, cx: x0 + 7 + (k % cols) * cell, cy: 36 + Math.floor(k / cols) * cell }));
  });
  const dots = svg1.selectAll('circle').data(pts).join('circle').attr('cx', (d) => d.cx).attr('cy', (d) => d.cy).attr('r', 7)
    .attr('stroke', css('--surface')).attr('stroke-width', 1.2)
    .on('mousemove', (e, d) => tip(e, `<b>${B.ids[d.i]}</b><br>${B.centre[d.i]} · ${B.pheno[d.i]}<br>${B.resp[d.i] ? 'responder' : 'non-responder'}<br>IFN-γ module score ${B.versions.protected.ifng[d.i].toFixed(2)}`))
    .on('mouseleave', untip);

  // bars: response rate per phenotype and per centre
  const groups = [...PHENOS.map((p) => ({ k: p, ...byPh[p], col: phenoColor(p) })), ...CENTRES.map((c) => ({ k: c, ...byC[c], col: centreColor(c) }))];
  const F1 = frame(barW, { width: 400, height: 300, x: [0, 1], y: [0, 1], yLabel: 'response rate', xScale: d3.scaleBand().padding(0.3), margin: { bottom: 40 } });
  F1.x.domain(groups.map((g) => g.k));
  F1.g.select('.axis.x').call(d3.axisBottom(F1.x));
  F1.g.select('.axis.y').call(d3.axisLeft(F1.y).tickFormat(d3.format('.0%')));
  F1.layer('bars').selectAll('rect').data(groups).join('rect').attr('x', (d) => F1.x(d.k)).attr('width', F1.x.bandwidth())
    .attr('y', (d) => F1.y(d.r / d.n)).attr('height', (d) => F1.h - F1.y(d.r / d.n)).attr('rx', 3).attr('fill', (d) => d.col)
    .on('mousemove', (e, d) => tip(e, `<b>${d.k}</b><br>${d.r} of ${d.n} responded (${pct(d.r / d.n)})`)).on('mouseleave', untip);
  F1.layer('lab').selectAll('text').data(groups).join('text').attr('x', (d) => F1.x(d.k) + F1.x.bandwidth() / 2).attr('y', (d) => F1.y(d.r / d.n) - 5)
    .attr('text-anchor', 'middle').attr('font-size', 11).attr('fill', ink2()).text((d) => pct(d.r / d.n));
  F1.layer('sep').append('line').attr('x1', F1.x('Centre A') - 10).attr('x2', F1.x('Centre A') - 10).attr('y1', 0).attr('y2', F1.h).attr('stroke', css('--axis')).attr('stroke-dasharray', '3 3');

  function paint1() {
    const f = { pheno: (i) => phenoColor(B.pheno[i]), resp: (i) => respColor(B.resp[i]), centre: (i) => centreColor(B.centre[i]) }[colorBy];
    dots.attr('fill', (d) => f(d.i));
    leg1.innerHTML = '';
    legend(leg1, { pheno: PHENOS.map((p) => [p, phenoColor(p)]), resp: [['responder', respColor(1)], ['non-responder', respColor(0)]], centre: CENTRES.map((c) => [c, centreColor(c)]) }[colorBy]);
  }
  paint1();
  rd1.set({
    'discovery patients': `${nDisc} (A ${disc.A} · B ${disc.B})`, 'after QC': nKept,
    responders: `${allR.r} (${pct(allR.r / allR.n)})`,
    'inflamed share, A vs B': `${pct(share('Centre A', 'inflamed'))} vs ${pct(share('Centre B', 'inflamed'))}`,
    'validation cohort': `${DESIGN.validation_n} (${DESIGN.validation_centre})`,
  });

  // ------------------------------------------------------------------ figure 2: the ML map
  const omics = TABS.find((t) => t.id === 'omics');
  const lanes = omics.sections.filter((s) => s.name !== 'The study');
  const fig2 = figure(main, 'The ML map: every step of the study',
    'Each box is a page. Colour is the verdict: is the step a machine-learning problem, a statistics problem, or both? Click a box to open it; hover for the one-line reason. Dashed arrows show where one data layer feeds another.');
  let hl = 'all';
  select(fig2.controls, { label: 'Highlight', value: 'all', options: [['all', 'all steps'], ['ml', 'machine learning'], ['stats', 'statistics'], ['both', 'ML + statistics']], onChange: (v) => { hl = v; paint2(); } });
  legend(fig2.controls, Object.entries(VLABEL).map(([k, t]) => [t, vcol(k)]));
  const rd2 = stats(fig2.root);

  const MW = 960, colW = 284, gapX = 48, bh = 34, gapY = 10, top = 34;
  const MH = top + 11 * (bh + gapY) + 10;
  const svg2 = d3.select(fig2.body).append('svg').attr('viewBox', `0 0 ${MW} ${MH}`).attr('width', MW);
  const defs = svg2.append('defs');
  defs.append('marker').attr('id', 'ov-arrow').attr('viewBox', '0 0 10 10').attr('refX', 9).attr('refY', 5).attr('markerWidth', 7).attr('markerHeight', 7).attr('orient', 'auto')
    .append('path').attr('d', 'M0,0 L10,5 L0,10 z').attr('fill', muted());
  const pos = {};
  let num = 1; // the overview itself is page 01
  lanes.forEach((lane, li) => {
    const x = 10 + li * (colW + gapX);
    svg2.append('text').attr('x', x).attr('y', 18).attr('fill', ink()).attr('font-weight', 700).attr('font-size', 14).text(lane.name);
    lane.items.forEach(([id, title, blurb], k) => { num += 1; pos[id] = { x, y: top + k * (bh + gapY), id, title, blurb, num, lane: li }; });
  });
  const nodes = Object.values(pos);
  // within-lane flow arrows
  const flow = svg2.append('g');
  nodes.forEach((n) => {
    const next = nodes.find((m) => m.lane === n.lane && m.num === n.num + 1);
    if (next) flow.append('line').attr('x1', n.x + 20).attr('x2', n.x + 20).attr('y1', n.y + bh).attr('y2', next.y).attr('stroke', css('--axis')).attr('stroke-width', 1.5);
  });
  // cross-layer links: where one layer's output is another's input
  const LINKS = [
    ['sc-annotation', 'bulk-deconvolution', 'annotated cells become the deconvolution reference'],
    ['sc-annotation', 'sp-deconvolution', 'the same reference unmixes spatial spots'],
    ['bulk-validation', 'sp-biomarker', 'the bulk signature is re-examined in space'],
  ];
  const cross = svg2.append('g');
  LINKS.forEach(([a, b, why]) => {
    const A = pos[a], Bn = pos[b];
    const ax = A.lane < Bn.lane ? A.x + colW : A.x, bx = A.lane < Bn.lane ? Bn.x : Bn.x + colW;
    const ay = A.y + bh / 2, by = Bn.y + bh / 2, mx = (ax + bx) / 2;
    cross.append('path').attr('d', `M${ax},${ay} C${mx},${ay} ${mx},${by} ${bx},${by}`).attr('fill', 'none')
      .attr('stroke', muted()).attr('stroke-width', 1.5).attr('stroke-dasharray', '5 4').attr('marker-end', 'url(#ov-arrow)')
      .append('title').text(why);
  });
  const g2 = svg2.append('g').selectAll('a').data(nodes).join('a').attr('href', (d) => `#/omics/${d.id}`);
  g2.append('rect').attr('x', (d) => d.x).attr('y', (d) => d.y).attr('width', colW).attr('height', bh).attr('rx', 7)
    .attr('stroke-width', 2);
  g2.append('rect').attr('class', 'bar').attr('x', (d) => d.x).attr('y', (d) => d.y).attr('width', 6).attr('height', bh).attr('rx', 3);
  g2.append('text').attr('x', (d) => d.x + 14).attr('y', (d) => d.y + bh / 2 + 4.5).attr('font-size', 13).attr('fill', ink())
    .text((d) => `${String(d.num).padStart(2, '0')}  ${d.title}`);
  g2.on('mousemove', (e, d) => tip(e, `<b>${d.title}</b><br>${d.blurb}<br><span class="tag">${VLABEL[VERDICT[d.id][0]]}</span>${VERDICT[d.id][1]}`))
    .on('mouseleave', untip);

  function paint2() {
    g2.attr('opacity', (d) => (hl === 'all' || VERDICT[d.id][0] === hl ? 1 : 0.25));
    g2.select('rect:not(.bar)').attr('fill', css('--surface-2')).attr('stroke', (d) => vcol(VERDICT[d.id][0]));
    g2.select('rect.bar').attr('fill', (d) => vcol(VERDICT[d.id][0]));
    const cnt = (v) => nodes.filter((n) => VERDICT[n.id][0] === v).length;
    rd2.set({ steps: nodes.length, 'machine learning': cnt('ml'), statistics: cnt('stats'), 'ML + statistics': cnt('both') });
  }
  paint2();

  // ------------------------------------------------------------------ figure 3: the planted signals
  const fig3 = figure(main, 'The planted biology: what a perfect analysis should find',
    'Each dot is a patient (after QC). The y-axis is the mean of the 12 interferon-γ (IFN-γ) module genes, after batch correction. Response is drawn from the phenotype, nudged up or down by this score. Hover for details.');
  const r3 = rng(7);
  const jit = B.ids.map(() => (r3() - 0.5) * 0.55);
  const ifng = B.versions.protected.ifng;
  const F3 = frame(fig3.body, { width: 620, height: 320, x: [0, 1], y: d3.extent(ifng), yLabel: 'IFN-γ module score (log-CPM)', xScale: d3.scaleBand().padding(0.15) });
  F3.x.domain(PHENOS);
  F3.y.nice();
  F3.g.select('.axis.x').call(d3.axisBottom(F3.x));
  F3.g.select('.axis.y').call(d3.axisLeft(F3.y));
  const sub = d3.scaleLinear().domain([-0.5, 0.5]).range([0, F3.x.bandwidth()]);
  F3.layer('pts').selectAll('circle').data(B.ids.map((_, i) => i)).join('circle')
    .attr('cx', (i) => F3.x(B.pheno[i]) + sub(jit[i] * 0.5 + (B.resp[i] ? 0.22 : -0.22))).attr('cy', (i) => F3.y(ifng[i])).attr('r', 4.5)
    .attr('fill', (i) => respColor(B.resp[i])).attr('stroke', css('--surface')).attr('stroke-width', 1.2)
    .on('mousemove', (e, i) => tip(e, `<b>${B.ids[i]}</b> · ${B.pheno[i]}<br>${B.resp[i] ? 'responder' : 'non-responder'}<br>IFN-γ score ${ifng[i].toFixed(2)}`)).on('mouseleave', untip);
  legend(fig3.root, [['responder', respColor(1)], ['non-responder', respColor(0)]]);
  stats(fig3.root).set({
    'AUC of IFN-γ score for response': f3(B.versions.protected.auc_ifng),
    ...Object.fromEntries(PHENOS.map((p) => [`${p}: response`, `${byPh[p].r}/${byPh[p].n} (${pct(byPh[p].r / byPh[p].n)})`])),
  });

  section(main, 'What to notice', `<ul>
    <li><b>Batch and biology are tangled.</b> Colour the cohort by centre, then by phenotype: Centre B enrolled far more inflamed tumours (${pct(share('Centre B', 'inflamed'))} vs ${pct(share('Centre A', 'inflamed'))}), so its response rate is higher (${pct(byC['Centre B'].r / byC['Centre B'].n)} vs ${pct(byC['Centre A'].r / byC['Centre A'].n)}). Any analysis that ignores the centre will confuse a technical difference with a biological one.</li>
    <li><b>Response is noisy.</b> Even among inflamed tumours only ${pct(byPh.inflamed.r / byPh.inflamed.n)} respond, and some desert tumours respond. No biomarker will reach an AUC of 1, and a reported AUC near 1 on data like this should make you suspicious.</li>
    <li>On the map, use <b>Highlight</b>: the statistics steps sit where the question is "is this difference real?" (differential expression, pseudobulk, validation). The ML steps sit where the question is "predict a new sample" or "find structure without labels".</li>
  </ul>`);

  steps(main, 'The study design', [
    `<b>Discovery cohort (bulk RNA-seq):</b> ${nDisc} pre-treatment tumour biopsies, ${disc.A} from Centre A and ${disc.B} from Centre B. Labels: response (yes/no) and progression-free survival. A pathologist has scored each tumour's immune phenotype from CD8 staining.`,
    `<b>External validation cohort:</b> ${DESIGN.validation_n} patients from ${DESIGN.validation_centre}, sequenced later, with a different case mix. It is touched once, at the very end, by a frozen model.`,
    `<b>Single-cell RNA-seq:</b> ${DESIGN.sc_patients} patients, about ${DESIGN.sc_cells.toLocaleString()} cells, two chemistry versions. Used to see <i>which cells</i> carry the bulk signal and to build a reference.`,
    `<b>Spatial transcriptomics:</b> ${DESIGN.sp_sections} tumour sections (two inflamed responders, one excluded and one desert non-responder). Used to see <i>where</i> the T cells are.`,
  ]);

  section(main, 'The planted story (and its traps)', `<ul>
    <li><b>Three immune phenotypes.</b> <i>Inflamed</i>: CD8 T cells inside the tumour. <i>Excluded</i>: T cells present but trapped in the stroma. <i>Desert</i>: few T cells at all.</li>
    <li><b>The predictive biomarker</b> is an IFN-γ / cytotoxic module (CXCL9, CXCL10, GZMB, IDO1 …). It raises the chance of response.</li>
    <li><b>The exclusion module</b> (TGF-β / cancer-associated fibroblasts: POSTN, FN1, ACTA2 …) marks excluded tumours. Bulk RNA-seq sees T cells in an excluded tumour; only spatial data shows they never reach the tumour.</li>
    <li><b>The prognostic-not-predictive trap.</b> A proliferation module (MKI67, TOP2A …) shortens survival in everyone, treated or not, but has no effect on response. A survival analysis will find it; it is not a response biomarker.</li>
    <li><b>Technical traps:</b> a confounded batch, three failed libraries, ${qc.n_genes.toLocaleString()} genes for ${nKept} patients (so leakage inflates accuracy easily), and cells that are not independent replicates.</li>
  </ul>`);

  callout(main, '', 'In biology', 'The design is modelled on real anti-PD-1 biomarker work in NSCLC and melanoma: the 18-gene T-cell-inflamed gene expression profile built around IFN-γ signalling (Ayers et al., <i>J Clin Invest</i> 2017), the inflamed / excluded / desert classification (Chen &amp; Mellman, <i>Nature</i> 2017), and TGF-β-driven exclusion in urothelial cancer (Mariathasan et al., <i>Nature</i> 2018, the IMvigor210 cohort). The genes are real symbols; the patients and every number are simulated.');
  callout(main, 'warn', 'Pitfalls', `<ul>
    <li>"We used ML" is not a method. At each step ask: is the output a <b>prediction for a new sample</b>, a <b>structure</b>, or an <b>inference</b> about a population? Only the first two are ML problems.</li>
    <li>Mock data has a known answer, so you can check every method against the truth. Real data does not, which is why the honesty checks (permuted labels, nested CV, external validation) matter more there, not less.</li>
  </ul>`);
  seeAlso(main, [['omics', 'bulk-qc', 'Step 1: QC'], ['omics', 'bulk-classifier', 'The leakage trap'], ['omics', 'sp-biomarker', 'Back to the biomarker'], ['concepts', 'what-is-ml', 'What "learning" means']]);
}
