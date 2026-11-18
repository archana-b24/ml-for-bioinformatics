// The 2024–26 landscape, lesson format.
// Arc: what → why → where → how it works (6 steps: timeline map → who builds what → data scale → claims vs independent
//      evidence → lab: a challenger vs a baseline on the mock study → checklist) → pitfalls → recap.
// Facts and URLs come only from docs/frontier_research.md. The lab is a teaching toy on the mock patient cohort.
import { header, figure, slider, select, button, checkbox, stats, legend, seeAlso, tip, untip, el, html } from '../lib/ui.js';
import { frame, cat, css, ink, ink2, muted, sequential } from '../lib/plot.js';
import { rng, mean } from '../lib/data.js';
import { fitLogistic, predictProb, auc, stratifiedSplit } from '../lib/models.js';
import { fitForest, forestProb } from '../lib/models.js';
import { loadCohort, FEATURES } from '../lib/study.js';
import { lessonNav, chapter, prose, rung, worked, quiz, cards, table, code, recap } from '../lib/lesson.js';
import { sourceCard, cite, REFS } from './_shared.js';
const d3 = window.d3;

const f2 = (v) => v.toFixed(2);
const f3 = (v) => v.toFixed(3);
const sgn2 = (v) => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(2);
const sum = (a) => a.reduce((s, v) => s + v, 0);
const fmtM = (v) => (v >= 1e6 ? d3.format('.3~s')(v).replace('M', ' million') : d3.format(',')(v));

const CATS = ['Cell foundation models', 'Perturbation & virtual cells', 'Independent benchmarks', 'Spatial & pathology', 'Bulk & clinical', 'Agents, LLMs & sequence'];
const CATS2 = [['Cell foundation', 'models'], ['Perturbation &', 'virtual cells'], ['Independent', 'benchmarks'], ['Spatial &', 'pathology'], ['Bulk &', 'clinical'], ['Agents, LLMs', '& sequence']];
const KINDS = ['Academic labs', 'Non-profit institutes', 'Tech companies', 'Biotech & diagnostics', 'Pharma (inferred)', 'Independent evaluators', 'Not stated in brief'];

// t = decimal year. A month is used only where the cited identifier (DOI, arXiv number, URL date) or the brief gives it.
// cells = training or data-set size in cells, only where the brief states it. k = organisation type (index into KINDS).
const M = [
  { n: 'UCE', c: 0, k: 0, t: 2023.8, when: '2023 (bioRxiv)', org: 'Stanford', ref: 'uce', what: 'Genes as ESM2 protein embeddings, so one model spans species', approx: true },
  { n: 'scGPT', c: 0, k: 0, t: 2024.3, when: '2024 (Nat Methods)', org: 'Bo Wang lab', ref: 'scgpt', cells: 33e6, what: 'Gene token + binned value token; masked generative pretraining', approx: true },
  { n: 'scFoundation', c: 0, k: 3, t: 2024.45, when: '2024 (Nat Methods)', org: 'BioMap / Tsinghua', ref: 'scfoundation', what: 'Asymmetric encoder-decoder; read-depth-aware masked regression', approx: true },
  { n: 'Geneformer V2', c: 0, k: 0, t: 2024.95, when: 'Dec 2024', org: 'Theodoris lab (Gladstone / Broad)', ref: 'geneformer', cells: 104e6, what: 'Rank-value encoding + BERT masked language model' },
  { n: 'C2S-Scale', c: 0, k: 2, t: 2025.28, when: 'Apr 2025 (bioRxiv)', org: 'Google Research / DeepMind + Yale', ref: 'c2s', what: 'Cells written as gene-name text; Gemma-2 language model (2B / 27B)' },
  { n: 'TranscriptFormer', c: 0, k: 1, t: 2025.32, when: 'Apr 2025 (bioRxiv)', org: 'CZI / Biohub', ref: 'transcriptformer', cells: 112e6, what: 'Autoregressive next gene + count; 12 species' },
  { n: 'rbio1', c: 0, k: 1, t: 2025.63, when: 'Aug 2025 (bioRxiv)', org: 'CZI', ref: 'rbio1', what: 'Language model post-trained with RL, a virtual-cell model as soft verifier' },
  { n: 'Stack', c: 0, k: 1, t: 2026.04, when: 'Jan 2026', org: 'Arc Institute', ref: 'stackNews', what: 'Attends across genes and across cells; in-context learning from prompt cells' },
  { n: 'Tahoe-100M', c: 1, k: 3, t: 2025.14, when: 'Feb 2025 (bioRxiv)', org: 'Tahoe', ref: 'tahoe', cells: 100e6, what: 'Perturbation atlas (data resource)' },
  { n: 'LPM', c: 1, k: 4, t: 2025.22, when: 'Mar 2025 (arXiv)', org: 'Genentech (inferred from the GitHub organisation)', ref: 'lpm', what: 'Perturbation × readout × context, disentangled' },
  { n: 'CellFlow', c: 1, k: 0, t: 2025.27, when: 'Apr 2025 (bioRxiv)', org: 'Theis lab', ref: 'cellflow', what: 'Conditional flow matching from control to perturbed distribution' },
  { n: 'X-Atlas/Orion', c: 1, k: 3, t: 2025.44, when: 'Jun 2025 (bioRxiv)', org: 'Xaira', ref: 'xaira', cells: 8e6, what: '8-million-cell Perturb-seq (data resource)' },
  { n: 'State', c: 1, k: 1, t: 2025.5, when: '2025', org: 'Arc Institute', ref: 'stateNews', what: 'Set transformer maps a control set of cells to a perturbed set', approx: true },
  { n: 'TxPert', c: 1, k: 3, t: 2025.55, when: '2025', org: 'Valence Labs / Recursion', ref: 'txpert', what: 'Graph neural network over knowledge graphs for unseen perturbations', approx: true },
  { n: 'VCC 2025', c: 1, k: 1, t: 2025.85, when: '2025', org: 'Arc Institute', ref: 'vcc2025', what: 'Open challenge: H1 hESC, 300 CRISPRi perturbations', approx: true },
  { n: 'PRiMeFlow', c: 1, k: 3, t: 2026.28, when: 'Apr 2026 (arXiv)', org: 'Altos (VCC generalist prize)', ref: 'primeflow', what: 'Flow matching for perturbation response' },
  { n: 'PerturBench', c: 2, k: 5, t: 2024.6, when: 'Aug 2024 (arXiv)', org: 'benchmark authors', ref: 'perturbench', what: 'Perturbation-model benchmark' },
  { n: 'Kedzierska et al.', c: 2, k: 5, t: 2025.4, when: '2025 (Genome Biol)', org: 'independent', ref: 'kedzierska', what: 'Zero-shot embeddings vs HVG+PCA, scVI and Harmony', approx: true },
  { n: 'Ahlmann-Eltze et al.', c: 2, k: 5, t: 2025.5, when: '2025 (Nat Methods)', org: 'Ahlmann-Eltze, Huber & Anders', ref: 'ahlmann', what: 'An additive baseline beats 5 foundation models on double perturbations', approx: true },
  { n: 'PertEval-scFM', c: 2, k: 5, t: 2025.53, when: '2025 (ICML)', org: 'benchmark authors', ref: 'perteval', what: 'Benchmark of single-cell foundation-model embeddings for perturbations', approx: true },
  { n: 'VCBench', c: 2, k: 5, t: 2026.46, when: 'Jun 2026 (bioRxiv)', org: 'benchmark authors', ref: 'vcbench', what: 'Simple baselines match or beat foundation models on 4 of 5 axes' },
  { n: 'CellPLM', c: 3, k: 6, t: 2024.3, when: '2024 (ICLR)', org: '—', what: 'Cells are tokens, a tissue is the sentence', approx: true },
  { n: 'Prov-GigaPath', c: 3, k: 2, t: 2024.4, when: '2024 (Nature)', org: 'Microsoft / Providence', ref: 'gigapath', what: 'Whole-slide pathology foundation model', approx: true },
  { n: 'HEST-1k', c: 3, k: 5, t: 2024.45, when: 'Jun 2024 (arXiv)', org: 'benchmark authors', ref: 'hest', what: 'H&E → expression benchmark; best Pearson r about 0.41 on 50 HVGs' },
  { n: 'OCTO-vc', c: 3, k: 3, t: 2024.96, when: 'Dec 2024 (press release)', org: 'Noetik', ref: 'noetik', what: 'Industry multimodal tumour model (announcement)' },
  { n: 'Nicheformer', c: 3, k: 0, t: 2025.5, when: '2025 (Nat Methods)', org: 'Theis lab', ref: 'nicheformer', cells: 110e6, what: 'Pretrained on SpatialCorpus-110M; spatial labels onto dissociated cells', approx: true },
  { n: 'Novae', c: 3, k: 6, t: 2025.6, when: '2025 (Nat Methods)', org: '—', ref: 'novae', what: 'Graph attention + SwAV self-supervision; zero-shot spatial domains', approx: true },
  { n: 'M-Optimus-1', c: 3, k: 3, t: 2026.3, when: '2026', org: 'Bioptimus', ref: 'moptimus', what: 'Multimodal pathology model (vendor numbers)', approx: true },
  { n: 'Tempus IPS', c: 4, k: 3, t: 2024.85, when: 'Nov 2024 (medRxiv)', org: 'Tempus', ref: 'tempus', what: 'ML on TMB + 8 RNA genes; development n = 1707, validation n = 1600' },
  { n: 'BulkFormer', c: 4, k: 6, t: 2025.44, when: 'Jun 2025 (bioRxiv)', org: '—', ref: 'bulkformer', what: 'Masked-expression foundation model for bulk RNA' },
  { n: 'GPTCelltype', c: 5, k: 6, t: 2024.25, when: '2024 (Nat Methods)', org: '—', ref: 'gptcelltype', what: 'A language model labels clusters from marker genes', approx: true },
  { n: 'CellAgent', c: 5, k: 6, t: 2024.52, when: 'Jul 2024 (arXiv)', org: '—', ref: 'cellagent', what: 'Language-model agent for single-cell analysis' },
  { n: 'AI co-scientist', c: 5, k: 2, t: 2025.13, when: '2025', org: 'Google', ref: 'coscientist', what: 'Gemini multi-agent system; Elo tournament ranking of hypotheses', approx: true },
  { n: 'Biomni', c: 5, k: 0, t: 2025.42, when: '2025', org: 'Stanford', ref: 'biomni', what: 'ReAct agent with about 150 tools', approx: true },
  { n: 'AlphaGenome', c: 5, k: 2, t: 2025.46, when: 'Jun 2025', org: 'Google DeepMind', what: 'DNA → RNA tracks from up to 1 Mb of sequence' },
];
const TODAY = 2026.86; // the brief was compiled 2026-11-12

// Evidence grid. State per source: claim (a claim, not evidence) | supports | mixed | contradicts | none.
const EV = [
  { m: 'Zero-shot cell embeddings', who: 'Geneformer, scGPT', s: ['claim', 'none', 'contradicts'], v: 'Contradicted', cls: 'bad',
    claim: 'General-purpose embeddings that transfer without fine-tuning (developers).', ev: `${cite('kedzierska', 'Kedzierska et al. 2025')}: zero-shot embeddings are worse than HVG+PCA, scVI and Harmony.` },
  { m: 'Foundation models for perturbation', who: 'various developers', s: ['claim', 'contradicts', 'contradicts'], v: 'Contradicted', cls: 'bad',
    claim: 'Models learn transferable perturbation responses (various developers).', ev: `${cite('ahlmann', 'Ahlmann-Eltze et al. 2025')}: an additive baseline beats 5 foundation models and 2 other deep models on double perturbations; nothing beats the training mean on unseen single perturbations. ${cite('vcbench', 'VCBench')}: simple baselines match or beat foundation models on 4 of 5 axes. ${cite('vcc2025', 'VCC 2025 wrap-up')}: almost nobody beat the mean baseline on MAE.` },
  { m: 'State', who: 'Arc Institute', s: ['claim', 'none', 'none'], v: 'Unconfirmed', cls: 'warn',
    claim: '"First to consistently beat linear baselines" (Arc, self-reported).', ev: 'No independent replication in the brief. The loss is described as MMD only by secondary sources (unverified).' },
  { m: 'Stack', who: 'Arc Institute', s: ['claim', 'none', 'none'], v: 'Unconfirmed', cls: 'warn',
    claim: 'In-context prediction for new conditions from a prompt set of cells (Arc, self-reported).', ev: 'No independent evaluation in the brief.' },
  { m: 'C2S-Scale', who: 'Google / Yale', s: ['claim', 'none', 'none'], v: 'Early', cls: 'warn',
    claim: 'A language model on cell sentences; a silmitasertib hypothesis (preprint).', ev: 'The discovery rests on a single in-vitro validation.' },
  { m: 'VCC 2025 metrics', who: 'Arc Institute', s: ['claim', 'contradicts', 'contradicts'], v: 'Metric caveat', cls: 'warn',
    claim: 'PDS, DES and MAE rank virtual-cell models.', ev: `${cite('vcc2025', "Arc's own wrap-up")}: L1-based PDS rewards inflating prediction magnitude. ${cite('systema', 'Systema')}: systematic perturbed-vs-control differences inflate metrics, so it evaluates against the perturbed centroid.` },
  { m: 'H&E → expression', who: 'pathology models', s: ['claim', 'none', 'mixed'], v: 'Modest', cls: 'warn',
    claim: 'Pathology foundation models predict expression from histology; M-Optimus numbers are vendor claims.', ev: `${cite('hest', 'HEST-1k')}: frozen embeddings → PCA → ridge reach Pearson r of about 0.41 at best, on 50 HVGs.` },
  { m: 'Tempus Immune Profile Score', who: 'Tempus', s: ['claim', 'none', 'none'], v: 'Needs external', cls: 'warn',
    claim: 'TMB + 8 RNA genes predicts immunotherapy benefit (development n = 1707, validation n = 1600).', ev: `Company-run validation in a ${cite('tempus', 'medRxiv preprint')}; no independent cohort in the brief.` },
  { m: 'GPTCelltype', who: 'authors', s: ['claim', 'none', 'none'], v: 'Plausible', cls: 'good',
    claim: 'More than 75% concordance with manual labels in most tissues (authors).', ev: 'Concordance with manual labels is not ground truth. No independent benchmark in the brief.' },
  { m: 'Agents', who: 'Biomni, AI co-scientist', s: ['claim', 'none', 'none'], v: 'Unconfirmed', cls: 'warn',
    claim: 'General agents and ranked hypotheses (developers).', ev: 'No independent benchmark in the brief.' },
];
const COLS = ['Developer report', 'Challenge or organiser report', 'Independent evaluation'];

export default async function render(main) {
  const C = await loadCohort();
  const D = C.discovery, V = C.validation;
  const y = D.map((p) => p.response), yv = V.map((p) => p.response);

  const models = M.filter((d) => d.c !== 2), bench = M.filter((d) => d.c === 2);
  const withCells = M.filter((d) => d.cells);
  const noMonth = M.filter((d) => d.approx).length;
  const medM = d3.median(models, (d) => d.t), medB = d3.median(bench, (d) => d.t);
  const nInd = EV.filter((e) => e.s[2] !== 'none').length, nContra = EV.filter((e) => e.s[2] === 'contradicts').length;
  const nNone = EV.filter((e) => e.s[1] === 'none' && e.s[2] === 'none').length;

  header(main, 'The 2024–26 landscape',
    'Between 2024 and 2026 single-cell machine learning moved from models trained on tens of millions of cells to corpora of about 100 million, language models that read cells as text, and agents. <b>Independent benchmarks have repeatedly found that simple baselines are competitive.</b> This page maps who builds what and at what scale, and compares each claim with the evidence behind it.');
  lessonNav(main);
  sourceCard(main, {
    org: `Literature brief covering ${M.length} methods, data sets and benchmarks (one organisation type per entry, where the brief names one)`,
    when: `Brief compiled 2026-11-12; ${noMonth} of ${M.length} entries have only a year, so their position within the year is approximate`,
    links: [['docs/frontier_research.md', 'docs/frontier_research.md']],
    claim: '<b>Question for this page:</b> who is building what, at what scale, and which headline claims have been checked by someone other than the developers? Figures 1–4 use only facts from the brief. The lab in step 5 is a teaching toy on the mock patient cohort, not a reproduction of any paper.',
    selfReported: false,
  });

  // =====================================================================================
  // 1. WHAT
  // =====================================================================================
  let ch = chapter(main, 'what', 'What is the 2024–26 landscape?', 'In one line: <b>tokenise biology → pretrain on a huge corpus → reuse for downstream tasks</b>, followed by a round of independent tests.');
  prose(ch, `<p>Most of the methods in this tab share one recipe. A measurement (a gene, a cell, a patch of tissue, a stretch of DNA) is converted into <b>tokens</b>; a large model is <b>pretrained</b> on millions of cells with a self-supervised task that needs no labels; the result is <b>reused</b> for annotation, perturbation prediction, spatial analysis or biomarkers. The diagram shows the recipe with an example for each stage.</p>`);
  ch.append(recipeSvg());
  prose(ch, '<p>Terms used on this page:</p>');
  table(ch, [
    { t: 'Foundation model', m: 'A large model pretrained once on a broad corpus and reused for many tasks', s: `Geneformer V2 (${fmtM(M.find((d) => d.n === 'Geneformer V2').cells)} cells)` },
    { t: 'Zero-shot', m: 'Using the pretrained model as it is, with no task-specific training', s: 'Embedding cells with Geneformer and clustering them directly' },
    { t: 'Fine-tuning', m: 'Further training on a labelled data set for one task', s: 'Training a cell-type head on top of the embedding' },
    { t: 'Baseline', m: 'A simple, well-tuned reference method that a new model must beat', s: 'HVG+PCA, the training mean, an additive model' },
    { t: 'Benchmark', m: 'A fixed task, split and metric used to compare methods', s: 'PerturBench, VCBench, HEST-1k' },
    { t: 'Virtual cell', m: 'A model that predicts how a cell’s expression changes under a perturbation', s: 'State, CellFlow, LPM' },
    { t: 'Agent', m: 'A language model that calls tools (databases, code) in a loop', s: 'Biomni, AI co-scientist, CellAgent' },
  ], [['t', 'Term'], ['m', 'Meaning'], ['s', 'Example from the brief']], { cls: 'compact' });

  // =====================================================================================
  // 2. WHY
  // =====================================================================================
  ch = chapter(main, 'why', 'Why do we need a map?', 'Announcements arrive faster than independent evaluations, so claims and evidence need to be read side by side.');
  prose(ch, `<p>A bioinformatics group choosing a method needs three things that individual papers do not provide:</p><ol>
    <li><b>Orientation.</b> Who builds what, for which task, and on how much data.</li>
    <li><b>A comparable yardstick.</b> Model size, patient count and perturbation count are different units, so entries cannot be ranked on one scale.</li>
    <li><b>Evidence from outside the developing group.</b> A result reported by its developers is a claim; a result reproduced by others is evidence.</li></ol>
    <p>The third point matters because evaluations arrive after the models they test. Of the ${M.length} entries in the brief, ${bench.length} are independent benchmarks. The median benchmark in the brief dates from ${medB.toFixed(1)}, against ${medM.toFixed(1)} for the median model or resource, a lag of about ${Math.round((medB - medM) * 12)} months.</p>`);
  const fy = figure(ch, 'Models and independent evaluations, by year', 'Bars count the entries in the brief by year. The first group is models, data resources and tools; the second is independent benchmarks. 2026 is a partial year (the brief was compiled in November). Hover a bar for the entries.');
  const years = [2023, 2024, 2025, 2026];
  const yrData = years.map((yr) => ({ yr, a: models.filter((d) => Math.floor(d.t) === yr), b: bench.filter((d) => Math.floor(d.t) === yr) }));
  const FY = frame(fy.body, { width: 820, height: 280, x: [0, 4], y: [0, Math.max(...yrData.map((d) => d.a.length)) + 2], xLabel: 'publication year', yLabel: 'entries in the brief', axes: false });
  FY.g.append('g').attr('class', 'axis y').call(d3.axisLeft(FY.y).ticks(5));
  const xb = d3.scaleBand().domain(years).range([0, FY.w]).paddingInner(0.35).paddingOuter(0.3), xin = d3.scaleBand().domain([0, 1]).range([0, xb.bandwidth()]).padding(0.08);
  FY.g.append('g').attr('class', 'axis x').attr('transform', `translate(0,${FY.h})`).call(d3.axisBottom(xb));
  const bars = yrData.flatMap((d) => [{ yr: d.yr, s: 0, list: d.a }, { yr: d.yr, s: 1, list: d.b }]);
  FY.layer('bars').selectAll('rect').data(bars).join('rect').attr('x', (d) => xb(d.yr) + xin(d.s)).attr('width', xin.bandwidth()).attr('y', (d) => FY.y(d.list.length)).attr('height', (d) => FY.h - FY.y(d.list.length))
    .attr('fill', (d) => cat(d.s ? 2 : 0)).on('mousemove', (e, d) => tip(e, `<b>${d.yr}: ${d.list.length} ${d.s ? 'independent benchmarks' : 'models, resources and tools'}</b><br>${d.list.map((q) => q.n).join(', ') || 'none'}`)).on('mouseleave', untip);
  FY.layer('lab').selectAll('text').data(bars).join('text').attr('x', (d) => xb(d.yr) + xin(d.s) + xin.bandwidth() / 2).attr('y', (d) => FY.y(d.list.length) - 5).attr('text-anchor', 'middle').attr('font-size', 12).attr('fill', ink()).text((d) => d.list.length);
  FY.layer('note').append('text').attr('x', FY.w - 6).attr('y', 14).attr('text-anchor', 'end').attr('font-size', 12).attr('fill', ink2()).text(`benchmarks: ${bench.filter((d) => d.t < 2025).length} in 2023–24, ${bench.filter((d) => d.t >= 2025).length} in 2025–26`);
  legend(fy.body, [['models, resources and tools', cat(0)], ['independent benchmarks', cat(2)]]);

  // =====================================================================================
  // 3. WHERE
  // =====================================================================================
  ch = chapter(main, 'where', 'Where do these methods apply?', 'Seven task families, with the methods in the brief that target each and the question to ask before adopting one.');
  const catList = (i) => M.filter((d) => d.c === i && d.c !== 2).map((d) => d.n).slice(0, 5).join(', ');
  cards(ch, [
    { tag: 'Cell annotation', title: 'What is this cell?', body: `Embeddings from cell foundation models (${catList(0)}) and language models (GPTCelltype labels clusters from marker genes). Check against HVG+PCA with the same classifier.` },
    { tag: 'Perturbation screens', title: 'What would this knockout do?', body: `Virtual-cell models (${catList(1)}) predict expression after a perturbation. Compare with the training mean and an additive model first.` },
    { tag: 'Large atlases', title: 'Which data to pretrain or test on?', body: `Perturbation atlases (Tahoe-100M, ${fmtM(100e6)} cells; X-Atlas/Orion, ${fmtM(8e6)} cells) are data resources that any group can use as test sets.` },
    { tag: 'Spatial', title: 'Which niche is this?', body: 'Nicheformer and Novae assign spatial context or domains; CellPLM treats cells as tokens and a tissue as the sentence, imputing masked cells from neighbours.' },
    { tag: 'Histology', title: 'Expression from the slide', body: 'Pathology foundation models (Prov-GigaPath, M-Optimus-1) feed H&E embeddings into a simple regressor; HEST-1k is the reference benchmark.' },
    { tag: 'Biomarkers', title: 'A score for treatment benefit', body: 'Tempus IPS combines TMB with 8 RNA genes; BulkFormer is a masked-expression model for bulk RNA. The mock cohort on this site is the same kind of problem.' },
    { tag: 'Agents and sequence', title: 'Draft the analysis; read the genome', body: 'Biomni, AI co-scientist and CellAgent orchestrate tools; AlphaGenome predicts RNA tracks from up to 1 Mb of DNA.' },
  ]);
  prose(ch, `<p><b>When not to use a foundation model:</b> when the task has a few thousand labelled cells and a standard pipeline (HVG+PCA, scVI) already meets the need; when the test conditions (a new cell line, perturbation or donor) are far from the pretraining corpus and no benchmark covers them; or when a claim exists only as an announcement. In the brief, ${nNone} of the ${EV.length} headline claims in the scoreboard have neither an organiser nor an independent evaluation.</p>`);

  // =====================================================================================
  // 4. HOW IT WORKS
  // =====================================================================================
  ch = chapter(main, 'how', 'How to read the landscape', 'Six steps: when, who, how much, how well supported, a controlled comparison on mock data, and a checklist for new papers.');

  // ---- Step 1: timeline
  let rb = rung(ch, 1, 'When: the timeline',
    `<p>Each dot is one entry in the brief, placed by publication date and grouped into six lanes. Hollow dots have only a year in the brief and sit mid-year; filled dots have a month taken from the cited identifier. Colour shows either the lane or the type of organisation. Clicking a dot opens its source.</p>`);
  const f1 = figure(rb, 'Figure 1. Who built what, and when', 'Hover for the source and a one-line summary; click to open it. The dashed line marks the date the brief was compiled.');
  let colourBy = 'cat'; const on = CATS.map(() => true);
  select(f1.controls, { label: 'Colour by', value: colourBy, options: [['cat', 'Lane (task)'], ['kind', 'Organisation type']], onChange: (v) => { colourBy = v; paint1(); } });
  CATS.forEach((c, i) => checkbox(f1.controls, { label: ['Foundation', 'Perturbation', 'Benchmarks', 'Spatial', 'Bulk', 'Agents'][i], checked: true, onChange: (v) => { on[i] = v; paint1(); } }));
  const leg1 = el('div'); f1.body.append(leg1);
  const hold1 = el('div'); f1.body.append(hold1);
  const note1 = html('', 'div', 'fig-caption'); f1.body.append(note1);
  const col1 = (d) => cat(colourBy === 'cat' ? d.c : d.k);
  function paint1() {
    hold1.innerHTML = ''; leg1.innerHTML = '';
    legend(leg1, (colourBy === 'cat' ? CATS : KINDS).map((c, i) => [c, cat(i)]));
    const data = M.filter((d) => on[d.c]);
    const F = frame(hold1, { width: 820, height: 560, x: [2023.6, 2027.0], y: [0, 1], xLabel: 'publication date', grid: false, margin: { left: 12, right: 14 } });
    F.g.select('.axis.x').call(d3.axisBottom(F.x).tickValues([2024, 2024.5, 2025, 2025.5, 2026, 2026.5]).tickFormat((v) => (v % 1 ? 'mid ' + Math.floor(v) : String(v))));
    F.g.select('.axis.y').remove();
    const laneH = F.h / CATS.length;
    CATS.forEach((c, i) => {
      F.layer('lanes').append('rect').attr('x', 0).attr('width', F.w).attr('y', i * laneH).attr('height', laneH).attr('fill', i % 2 ? css('--surface-2') : 'none').attr('opacity', 0.6);
      F.layer('lanes').append('text').attr('x', 4).attr('y', i * laneH + 13).attr('font-size', 11).attr('fill', muted()).text(c + (on[i] ? '' : ' (hidden)'));
    });
    const pos = new Map();
    d3.group(data, (d) => d.c).forEach((arr) => {
      const ends = [];
      arr.slice().sort((a, b) => a.t - b.t).forEach((d) => {
        const x0 = F.x(d.t); let row = ends.findIndex((e) => e < x0 - 6); if (row < 0) { row = ends.length; ends.push(0); }
        ends[row] = x0 + 16 + d.n.length * 6.4; d._row = row;
      });
      arr.forEach((d) => pos.set(d, [F.x(d.t), d.c * laneH + 30 + d._row * 17]));
    });
    F.layer('now').append('line').attr('x1', F.x(TODAY)).attr('x2', F.x(TODAY)).attr('y1', 0).attr('y2', F.h).attr('stroke', muted()).attr('stroke-dasharray', '3 3');
    F.layer('now').append('text').attr('x', F.x(TODAY) - 4).attr('y', F.h - 6).attr('text-anchor', 'end').attr('font-size', 11).attr('fill', muted()).text('brief compiled');
    const g = F.layer('dots').selectAll('g').data(data).join('g').attr('transform', (d) => `translate(${pos.get(d)})`).style('cursor', (d) => (d.ref ? 'pointer' : 'default'))
      .on('click', (e, d) => { if (d.ref) { untip(); window.open(REFS[d.ref][1], '_blank', 'noopener'); } })
      .on('mousemove', (e, d) => tip(e, `<b>${d.n}</b> · ${d.when}<br>${d.org} · ${KINDS[d.k]}<br>${d.what}${d.cells ? `<br>${fmtM(d.cells)} cells` : ''}${d.ref ? `<br><i>${REFS[d.ref][0]}</i> (click to open)` : ''}`))
      .on('mouseleave', untip);
    g.append('circle').attr('r', 6).attr('fill', (d) => (d.approx ? css('--surface') : col1(d))).attr('stroke', (d) => (d.approx ? col1(d) : css('--surface'))).attr('stroke-width', (d) => (d.approx ? 2.5 : 1.5));
    g.append('text').attr('x', 10).attr('y', 4).attr('font-size', 11.5).attr('fill', ink()).text((d) => d.n);
    note1.innerHTML = `${data.length} of ${M.length} entries shown; ${data.filter((d) => d.approx).length} hollow (year only).`;
  }
  paint1();
  const perY = (yr) => M.filter((d) => Math.floor(d.t) === yr).length;
  prose(rb, `<p>The busiest lane is <b>${CATS[d3.greatest(CATS.map((c, i) => [i, M.filter((d) => d.c === i).length]), (a) => a[1])[0]]}</b>. The brief has ${perY(2024)} entries dated 2024, ${perY(2025)} dated 2025 and ${perY(2026)} dated 2026 so far. Switching to <i>Organisation type</i> shows that the independent benchmarks (one colour) are a separate group from the model developers.</p>`);

  // ---- Step 2: who builds what
  const cnt = (k, c) => M.filter((d) => d.k === k && d.c === c);
  const kTot = KINDS.map((_, k) => M.filter((d) => d.k === k).length);
  const devKinds = [0, 1, 2, 3, 4];
  const topDev = d3.greatest(devKinds.map((k) => [k, kTot[k]]), (a) => a[1]);
  rb = rung(ch, 2, 'Who: organisation type by lane',
    `<p>Each entry is assigned one type from what the brief states: academic lab, non-profit institute, technology company, biotech or diagnostics company, or pharma. Pharma is deliberately a separate row because it is nearly absent from the brief's model literature.</p>`);
  const f2g = figure(rb, 'Figure 2. Entries by organisation type and lane', 'Cell shade and number: entries in the brief. Click a cell to list them. The last column gives the row total.');
  let selCell = [topDev[0], 0];
  const SVG2W = 820, LW = 178, CW = 96, RH = 40, TH = 54;
  const svg2 = d3.select(f2g.body).append('svg').attr('viewBox', `0 0 ${SVG2W} ${TH + KINDS.length * RH + 6}`).attr('width', SVG2W).attr('role', 'img').attr('aria-label', 'organisation type by lane');
  const mx = d3.max(KINDS.flatMap((_, k) => CATS.map((_, c) => cnt(k, c).length))), col2 = sequential(0, mx + 1);
  CATS2.forEach((t, c) => t.forEach((line, j) => svg2.append('text').attr('x', LW + c * CW + CW / 2).attr('y', 18 + j * 15).attr('text-anchor', 'middle').attr('font-size', 11.5).attr('fill', ink2()).text(line)));
  svg2.append('text').attr('x', LW + CATS.length * CW + 30).attr('y', 32).attr('text-anchor', 'middle').attr('font-size', 11.5).attr('fill', ink2()).text('total');
  KINDS.forEach((kn, k) => svg2.append('text').attr('x', LW - 10).attr('y', TH + k * RH + RH / 2 + 4).attr('text-anchor', 'end').attr('font-size', 12.5).attr('fill', ink()).text(kn));
  KINDS.forEach((_, k) => svg2.append('text').attr('x', LW + CATS.length * CW + 30).attr('y', TH + k * RH + RH / 2 + 4).attr('text-anchor', 'middle').attr('font-size', 12.5).attr('fill', ink2()).text(kTot[k]));
  const cells2 = KINDS.flatMap((_, k) => CATS.map((_, c) => ({ k, c, n: cnt(k, c).length })));
  const rects2 = svg2.append('g').selectAll('g').data(cells2).join('g').attr('transform', (d) => `translate(${LW + d.c * CW},${TH + d.k * RH})`).style('cursor', 'pointer')
    .on('click', (e, d) => { selCell = [d.k, d.c]; paint2(); })
    .on('mousemove', (e, d) => tip(e, `<b>${KINDS[d.k]} × ${CATS[d.c]}</b><br>${d.n ? cnt(d.k, d.c).map((q) => q.n).join(', ') : 'no entries in the brief'}`)).on('mouseleave', untip);
  rects2.append('rect').attr('width', CW - 4).attr('height', RH - 4).attr('rx', 5).attr('fill', (d) => (d.n ? col2(d.n) : css('--surface-2')));
  rects2.append('text').attr('x', (CW - 4) / 2).attr('y', (RH - 4) / 2 + 5).attr('text-anchor', 'middle').attr('font-size', 13).attr('font-weight', 600).attr('fill', ink()).text((d) => (d.n ? d.n : ''));
  const sel2 = svg2.append('rect').attr('rx', 5).attr('fill', 'none').attr('stroke', ink()).attr('stroke-width', 2.2).attr('width', CW - 4).attr('height', RH - 4).style('pointer-events', 'none');
  const list2 = html('', 'p', 'fig-caption'); f2g.body.append(list2);
  function paint2() {
    sel2.attr('x', LW + selCell[1] * CW).attr('y', TH + selCell[0] * RH);
    const L = cnt(...selCell);
    list2.innerHTML = `<b>${KINDS[selCell[0]]} × ${CATS[selCell[1]]}</b>: ` + (L.length ? L.map((d) => `${d.n} (${d.org})`).join('; ') : 'no entries in the brief.');
  }
  paint2();
  prose(rb, `<p>${KINDS[topDev[0]]} account for the most entries among developers (${topDev[1]} of ${M.length}). Pharma has ${kTot[4]} (${M.filter((d) => d.k === 4).map((d) => d.n).join(', ')}, and its affiliation is inferred from a GitHub organisation, so it is unverified). The brief states that no 2024–26 transcriptomics foundation-model papers were found from Novartis, Sanofi, J&amp;J, Bayer, Insitro, Illumina, Isomorphic or Meta. This reflects what the search returned and does not show that such work does not exist. Companies appear mainly as data generators (Tahoe, Xaira, ${cite('mosaic', 'Owkin MOSAIC')}, ${cite('noetik', 'Noetik')}) and as partners; the AstraZeneca × SOPHiA multimodal stratification work is at announcement level only.</p>`);

  // ---- Step 3: scale
  rb = rung(ch, 3, 'How much: data scale',
    `<p>Only ${withCells.length} entries state a cell count in the brief, so only those can be compared on this axis. Model size, patient number and perturbation number are different units and are left out. A cell count also says little about the number of independent biological samples, which is the quantity that limits generalisation; the slider converts cells to samples under an assumed cell count per sample.</p>`);
  const f3g = figure(rb, 'Figure 3. Corpus size in cells (logarithmic axis)', 'Bars: cells in the model’s pretraining corpus or in the data resource. The right-hand label converts cells to an equivalent number of samples under the assumed cells per sample; the assumption is illustrative, not from the brief.');
  let sortBy = 'size', perSample = 5;
  select(f3g.controls, { label: 'Order', value: sortBy, options: [['size', 'By size'], ['date', 'By date']], onChange: (v) => { sortBy = v; paint3(); } });
  slider(f3g.controls, { label: 'assumed cells per sample (thousands)', min: 1, max: 20, step: 1, value: perSample, fmt: (v) => v + 'k', onInput: (v) => { perSample = v; paint3(); } });
  const s3 = stats(f3g.root);
  const F3 = frame(f3g.body, { width: 820, height: 290, x: [1e6, 3e8], xScale: d3.scaleLog(), y: [0, 1], grid: false, axes: false, margin: { left: 130, right: 150, top: 14 }, xLabel: 'cells (log scale)' });
  F3.g.append('g').attr('class', 'axis x').attr('transform', `translate(0,${F3.h})`).call(d3.axisBottom(F3.x).tickValues([1e6, 1e7, 1e8]).tickFormat((v) => d3.format('.0s')(v).replace('M', ' M')));
  F3.layer('ref').append('line').attr('x1', F3.x(1e8)).attr('x2', F3.x(1e8)).attr('y1', 0).attr('y2', F3.h).attr('stroke', muted()).attr('stroke-dasharray', '4 4');
  F3.layer('ref').append('text').attr('x', F3.x(1e8) - 5).attr('y', 11).attr('text-anchor', 'end').attr('font-size', 11.5).attr('fill', ink2()).text('100 million cells');
  function paint3() {
    const rows = withCells.slice().sort((a, b) => (sortBy === 'size' ? b.cells - a.cells : a.t - b.t));
    const yb = d3.scaleBand().domain(rows.map((d) => d.n)).range([20, F3.h]).padding(0.28);
    const L = F3.layer('bars');
    L.selectAll('*').remove();
    rows.forEach((d) => {
      const g = L.append('g').on('mousemove', (e) => tip(e, `<b>${d.n}</b> · ${d.when}<br>${d.org}<br>${fmtM(d.cells)} cells<br>≈ ${d3.format(',.0f')(d.cells / (perSample * 1000))} samples at ${perSample}k cells each`)).on('mouseleave', untip);
      g.append('rect').attr('x', 0).attr('y', yb(d.n)).attr('height', yb.bandwidth()).attr('width', F3.x(d.cells)).attr('fill', cat(d.c)).attr('rx', 3);
      g.append('text').attr('x', -8).attr('y', yb(d.n) + yb.bandwidth() / 2 + 4).attr('text-anchor', 'end').attr('font-size', 12).attr('fill', ink()).text(d.n);
      g.append('text').attr('x', F3.x(d.cells) + 6).attr('y', yb(d.n) + yb.bandwidth() / 2 + 4).attr('font-size', 11.5).attr('fill', ink2()).text(`${fmtM(d.cells)} · ≈ ${d3.format(',.0f')(d.cells / (perSample * 1000))} samples`);
    });
    const big = withCells.filter((d) => d.cells >= 1e8).length;
    s3.set({ 'entries ≥ 100 million cells': big, 'largest ÷ scGPT': (d3.max(withCells, (d) => d.cells) / 33e6).toFixed(1) + '×', 'largest ÷ smallest': (d3.max(withCells, (d) => d.cells) / d3.min(withCells, (d) => d.cells)).toFixed(1) + '×', 'samples in this study': `${C.all.length} patients` });
  }
  paint3();
  const maxC = d3.max(withCells, (d) => d.cells);
  worked(rb, 'What 100 million cells is, in samples',
    `<p>At an assumed 5,000 cells per sample, ${fmtM(maxC)} cells correspond to about <b>${d3.format(',.0f')(maxC / 5000)} samples</b>, or about ${d3.format(',.0f')(maxC / 5000 / C.all.length)} studies the size of this site's mock cohort (${C.all.length} patients). The corpora are very large in cells, but the number of independent donors, tissues and perturbation conditions in them is much smaller than the cell count suggests. These counts are cells; they say nothing about accuracy, which is what the next step addresses.</p>`);

  // ---- Step 4: scoreboard
  rb = rung(ch, 4, 'How well supported: claims vs independent evidence',
    `<p>For each headline claim in the brief, the grid shows three possible sources: the developers' own report, a challenge or organiser report, and an independent evaluation. A filled dot is a claim (not evidence); the other marks show what a source found. An empty circle means that no such source appears in the brief, which is not the same as the claim being wrong.</p>`);
  const f4 = figure(rb, 'Figure 4. The scoreboard', 'Hover a mark for its meaning; click a row to read the claim, the evidence and the sources below the grid.');
  const MARK = { claim: [d3.symbolCircle, () => ink(), 'a claim by the developers or organisers'], supports: [d3.symbolCircle, () => cat(1), 'supports the claim'], mixed: [d3.symbolDiamond, () => cat(2), 'partly supports, partly limits'], contradicts: [d3.symbolTimes, () => cat(3), 'contradicts the claim'], none: [d3.symbolCircle, () => 'none', 'nothing in the brief'] };
  legend(f4.body, [['claim (not evidence)', ink()], ['mixed evidence', cat(2)], ['contradicts the claim', cat(3)], ['not in the brief (empty circle)', muted()]]);
  const GW = 820, GL = 250, GC = [356, 478, 600], GH = 34, GT = 46;
  const svg4 = d3.select(f4.body).append('svg').attr('viewBox', `0 0 ${GW} ${GT + EV.length * GH + 4}`).attr('width', GW).attr('role', 'img').attr('aria-label', 'claims by source of evidence');
  COLS.forEach((c, j) => c.split(' ').reduce((acc, w) => { const l = acc[acc.length - 1]; if (l && (l + ' ' + w).length < 15) acc[acc.length - 1] = l + ' ' + w; else acc.push(w); return acc; }, []).forEach((line, i) => svg4.append('text').attr('x', GC[j]).attr('y', 14 + i * 14).attr('text-anchor', 'middle').attr('font-size', 11.5).attr('fill', ink2()).text(line)));
  svg4.append('text').attr('x', 700).attr('y', 32).attr('font-size', 11.5).attr('fill', ink2()).text('verdict (this page)');
  let selRow = 0;
  const rows4 = svg4.selectAll('g.row').data(EV).join('g').attr('class', 'row').attr('transform', (d, i) => `translate(0,${GT + i * GH})`).style('cursor', 'pointer').on('click', (e, d) => { selRow = EV.indexOf(d); paint4(); });
  rows4.append('rect').attr('class', 'bg').attr('x', 0).attr('y', 1).attr('width', GW).attr('height', GH - 2).attr('rx', 5).attr('fill', 'none');
  rows4.append('text').attr('x', 8).attr('y', GH / 2 - 1).attr('font-size', 12.5).attr('fill', ink()).text((d) => d.m);
  rows4.append('text').attr('x', 8).attr('y', GH / 2 + 12).attr('font-size', 10.5).attr('fill', muted()).text((d) => d.who);
  rows4.each(function (d) {
    d.s.forEach((st, j) => {
      const [sym, colf, desc] = MARK[st], g = d3.select(this).append('g').attr('transform', `translate(${GC[j]},${GH / 2})`);
      const c = colf();
      g.append('path').attr('d', d3.symbol(sym, st === 'none' ? 90 : 190)).attr('fill', st === 'none' || st === 'contradicts' ? 'none' : c).attr('stroke', st === 'none' ? muted() : st === 'contradicts' ? c : css('--surface')).attr('stroke-width', st === 'contradicts' ? 3 : st === 'none' ? 1.5 : 1.2);
      g.on('mousemove', (e) => tip(e, `<b>${d.m}</b><br>${COLS[j]}: ${desc}`)).on('mouseleave', untip);
    });
  });
  rows4.append('text').attr('x', 700).attr('y', GH / 2 + 4).attr('font-size', 12.5).attr('font-weight', 600).attr('fill', ink()).text((d) => d.v);
  const det4 = el('div', { class: 'worked', style: 'margin-top:10px' }); f4.body.append(det4);
  function paint4() {
    rows4.select('rect.bg').attr('fill', (d, i) => (i === selRow ? css('--accent-soft') : 'none'));
    const d = EV[selRow];
    det4.innerHTML = `<div class="w-k">Selected claim</div><div class="w-t">${d.m} · <span class="tag" style="border-left:3px solid var(--${d.cls});border-radius:4px">${d.v}</span></div><p><b>Claim:</b> ${d.claim}</p><p><b>Evidence:</b> ${d.ev}</p>`;
  }
  paint4();
  prose(rb, `<p>Of the ${EV.length} claims, ${nInd} have an independent evaluation in the brief and ${nContra} of those contradict the claim as stated; ${EV.length - nInd} have none. The verdicts apply to <i>the claim as stated</i>, using only the evidence in the brief. <b>Unconfirmed</b> means that no independent test is known, not that the claim is wrong. The pattern is consistent: ${cite('kedzierska', 'zero-shot embeddings')} trail HVG+PCA, ${cite('ahlmann', 'perturbation foundation models')} trail additive and mean baselines, and in ${cite('vcc2025', 'VCC 2025')} hybrids of statistics and deep learning won. Scale has bought coverage of species, tissues and modalities; the brief contains no independent result showing that it has bought accuracy on unseen conditions. The ${cite('vcc2026', 'VCC 2026')} round (zero-shot, 6 new cell lines) is the next test, and its winners had not been announced when the brief was compiled.</p>`);

  // ---- Step 5: lab
  const LAB10 = ['ifng', 'log2tmb', 'caf', 'pdl1', 'cd8', 'prolif', 'age', 'smoker', 'male', 'centreB'], F3K = ['ifng', 'log2tmb', 'caf'];
  const BASES = { pdl1: ['PD-L1 only (single marker)', ['pdl1']], ifng: ['IFN-γ only', ['ifng']], three: ['IFN-γ + TMB + CAF (logistic)', F3K] };
  const rows = (pts, keys) => pts.map((p) => keys.map((k) => FEATURES[k].get(p)));
  const split = stratifiedSplit(y, 0.3, 11), trI = split.train, teI = split.test;
  const cache = new Map();
  function evalCfg(bk, nTrees, depth) {
    const key = [bk, nTrees, depth].join('|'); if (cache.has(key)) return cache.get(key);
    const bkeys = BASES[bk][1];
    const fitB = (idx) => fitLogistic(rows(idx.map((i) => D[i]), bkeys), idx.map((i) => y[i]));
    const fitC = (idx) => fitForest(rows(idx.map((i) => D[i]), LAB10), idx.map((i) => y[i]), { nTrees, maxDepth: depth, minLeaf: 1, mtry: 3, seed: 3 });
    const all = D.map((_, i) => i), bFull = fitB(all), cFull = fitC(all), bSub = fitB(trI), cSub = fitC(trI);
    const pB = (f, pts) => pts.map((p) => predictProb(bkeys.map((k) => FEATURES[k].get(p)), f.b0, f.w));
    const pC = (f, pts) => pts.map((p) => forestProb(f, LAB10.map((k) => FEATURES[k].get(p))));
    const teP = teI.map((i) => D[i]), yte = teI.map((i) => y[i]);
    const sets = [
      { id: 'train', name: 'Training patients (discovery)', n: D.length, y, pb: pB(bFull, D), pc: pC(cFull, D) },
      { id: 'hold', name: 'Held-out patients, same centres', n: teI.length, y: yte, pb: pB(bSub, teP), pc: pC(cSub, teP) },
      { id: 'newc', name: 'New hospital (Centre C)', n: V.length, y: yv, pb: pB(bFull, V), pc: pC(cFull, V) },
    ];
    sets.forEach((s) => { s.ab = auc(s.pb, s.y); s.ac = auc(s.pc, s.y); s.d = s.ac - s.ab; s.ci = s.id === 'train' ? null : bootDiff(s); });
    cache.set(key, sets); return sets;
  }
  function bootDiff(s) {
    const r = rng(5), n = s.y.length, ds = [];
    for (let b = 0; b < 300; b++) { const idx = Array.from({ length: n }, () => r.int(n)), yy = idx.map((i) => s.y[i]); if (!sum(yy) || sum(yy) === n) continue; ds.push(auc(idx.map((i) => s.pc[i]), yy) - auc(idx.map((i) => s.pb[i]), yy)); }
    ds.sort((a, b) => a - b); return [ds[Math.floor(0.025 * ds.length)], ds[Math.floor(0.975 * ds.length)]];
  }
  rb = rung(ch, 5, 'Lab: challenger vs baseline on the mock study',
    `<p>The same comparison that the benchmarks in step 4 make, on the mock cohort. The <b>baseline</b> is a logistic regression. The <b>challenger</b> is a random forest on all ten measurements, with an adjustable size. Both are fitted on the ${D.length} discovery patients and evaluated three ways: on the training patients, on ${teI.length} held-out patients from the same two centres, and on the ${V.length} patients of the new hospital. <b>This is a teaching toy</b>: a random forest is not a foundation model, and the numbers say nothing about any published method. It illustrates how the verdict depends on the baseline and on the evaluation split.</p>`);
  const fl = figure(rb, 'Figure 5. Lab: does the larger model beat the baseline?', 'Bars: AUC (area under the ROC curve; 0.5 is chance, marked by the dashed line; 1 is perfect). Brackets: 95% bootstrap interval of the challenger-minus-baseline difference on the evaluation sets with patients the models have not seen.');
  let bk = 'three', nT = 100, dep = 6, rep = 'train';
  select(fl.controls, { label: 'Baseline', value: bk, options: Object.entries(BASES).map(([k, v]) => [k, v[0]]), onChange: (v) => { bk = v; paintL(); } });
  slider(fl.controls, { label: 'forest: trees', min: 0, max: 3, step: 1, value: 3, fmt: (i) => [1, 5, 25, 100][i], onInput: (i) => { nT = [1, 5, 25, 100][i]; paintL(); } });
  slider(fl.controls, { label: 'forest: depth', min: 1, max: 10, step: 1, value: dep, onInput: (v) => { dep = v; paintL(); } });
  select(fl.controls, { label: 'Result reported', value: rep, options: [['train', 'Training patients'], ['hold', 'Held-out, same centres'], ['newc', 'New hospital']], onChange: (v) => { rep = v; paintL(); } });
  button(fl.controls, 'Reset', () => { location.hash = location.hash; }, false);
  const sL = stats(fl.root);
  const FL = frame(fl.body, { width: 820, height: 340, x: [0, 3], y: [0.4, 1.02], yLabel: 'AUC', axes: false, margin: { top: 44, bottom: 56 } });
  FL.g.append('g').attr('class', 'axis y').call(d3.axisLeft(FL.y).ticks(6));
  const xg = d3.scaleBand().domain([0, 1, 2]).range([0, FL.w]).paddingInner(0.3).paddingOuter(0.15), xi = d3.scaleBand().domain([0, 1]).range([0, xg.bandwidth()]).padding(0.1);
  FL.layer('chance').append('line').attr('x1', 0).attr('x2', FL.w).attr('y1', FL.y(0.5)).attr('y2', FL.y(0.5)).attr('stroke', muted()).attr('stroke-dasharray', '4 4');
  legend(fl.body, [['baseline (logistic)', cat(6)], ['challenger (random forest)', cat(2)]]);
  const msgL = html('', 'p', 'fig-caption'); fl.body.append(msgL);
  function paintL() {
    const sets = evalCfg(bk, nT, dep), L = FL.layer('bars'); L.selectAll('*').remove();
    sets.forEach((s, i) => {
      [[s.ab, 6, 'baseline'], [s.ac, 2, 'challenger']].forEach(([v, c, nm], j) => {
        const x0 = xg(i) + xi(j);
        L.append('rect').attr('x', x0).attr('width', xi.bandwidth()).attr('y', FL.y(v)).attr('height', FL.y(0.4) - FL.y(v)).attr('fill', cat(c)).attr('rx', 3)
          .on('mousemove', (e) => tip(e, `<b>${nm}</b> · ${s.name}<br>AUC ${f3(v)} on ${s.n} patients`)).on('mouseleave', untip);
        L.append('text').attr('x', x0 + xi.bandwidth() / 2).attr('y', FL.y(v) - 5).attr('text-anchor', 'middle').attr('font-size', 12).attr('fill', ink()).text(f3(v));
      });
      L.append('text').attr('x', xg(i) + xg.bandwidth() / 2).attr('y', FL.h + 18).attr('text-anchor', 'middle').attr('font-size', 12.5).attr('fill', ink()).text(s.name);
      L.append('text').attr('x', xg(i) + xg.bandwidth() / 2).attr('y', FL.h + 34).attr('text-anchor', 'middle').attr('font-size', 11.5).attr('fill', ink2())
        .text(s.ci ? `difference ${sgn2(s.d)} (${sgn2(s.ci[0])} to ${sgn2(s.ci[1])})` : `difference ${sgn2(s.d)} (seen in training)`);
      if (s.id === rep) L.append('rect').attr('x', xg(i) - 6).attr('y', -40).attr('width', xg.bandwidth() + 12).attr('height', FL.h + 40 + 44).attr('fill', 'none').attr('stroke', ink()).attr('stroke-width', 1.4).attr('stroke-dasharray', '5 4').attr('rx', 6).style('pointer-events', 'none');
    });
    FL.layer('bars').append('text').attr('x', xg(sets.findIndex((s) => s.id === rep)) + xg.bandwidth() / 2).attr('y', -26).attr('text-anchor', 'middle').attr('font-size', 11.5).attr('fill', ink2()).text('reported as the headline');
    const r = sets.find((s) => s.id === rep), nc = sets[2];
    const verdict = (s) => (s.ci && s.ci[0] > 0 ? 'better' : s.ci && s.ci[1] < 0 ? 'worse' : s.ci ? 'not distinguishable' : s.d > 0 ? 'higher' : 'lower');
    sL.set({ 'headline AUC: challenger vs baseline': `${f3(r.ac)} vs ${f3(r.ab)}`, 'headline difference': sgn2(r.d), 'new hospital: challenger vs baseline': `${f3(nc.ac)} vs ${f3(nc.ab)}`, 'new hospital verdict': verdict(nc) });
    msgL.innerHTML = `<b>Headline.</b> Reporting only “${r.name.toLowerCase()}”, the challenger’s AUC is ${f3(r.ac)} against ${f3(r.ab)} for the baseline (${sgn2(r.d)}). On the new hospital the challenger is <b>${verdict(nc)}</b> ${verdict(nc) === 'not distinguishable' ? 'from' : 'than'} the baseline (${sgn2(nc.d)}; 95% interval ${sgn2(nc.ci[0])} to ${sgn2(nc.ci[1])}). In this simulation the true response rule is additive on the log-odds scale, which favours a logistic model; a data set with strong interactions could give a different result.`;
  }
  paintL();
  const sA = evalCfg('three', 100, 6), sB = evalCfg('pdl1', 100, 6), sC = evalCfg('three', 100, 2), sD = evalCfg('three', 100, 10), sE = evalCfg('three', 1, 6);
  const word = (d) => (d > 0 ? 'ahead by' : 'behind by');
  prose(rb, '<p><b>Suggested comparisons</b> (values below are computed for the default settings: 100 trees, depth 6 unless stated):</p>');
  table(rb, [
    { t: '<b>Baseline</b>: PD-L1 only → three-feature logistic', s: `Challenger is ${word(sB[2].d)} ${f2(Math.abs(sB[2].d))} AUC on the new hospital against PD-L1 only, and ${word(sA[2].d)} ${f2(Math.abs(sA[2].d))} against the three-feature model`, m: '<b>The verdict depends on the baseline.</b> A weak baseline makes any model look good; benchmark papers that include the strong simple baseline report different conclusions.' },
    { t: '<b>Result reported</b>: training → new hospital', s: `Challenger AUC ${f3(sA[0].ac)} on the training patients, ${f3(sA[1].ac)} held out, ${f3(sA[2].ac)} at the new hospital; baseline ${f3(sA[0].ab)}, ${f3(sA[1].ab)}, ${f3(sA[2].ab)}`, m: '<b>The evaluation split fixes the headline.</b> The flexible model fits the training data much more closely than it predicts new patients.' },
    { t: '<b>Depth</b> 2 → 10 (100 trees)', s: `Training AUC ${f3(sC[0].ac)} → ${f3(sD[0].ac)}; new hospital ${f3(sC[2].ac)} → ${f3(sD[2].ac)}`, m: 'More capacity raises the training score; the effect on unseen patients is smaller or absent. A larger model needs a harder test, not only a larger score.' },
    { t: '<b>Trees</b> 1 → 100', s: `New-hospital AUC ${f3(sE[2].ac)} → ${f3(sA[2].ac)}`, m: 'Averaging many trees stabilises the predictions; a single tree is noisy.' },
    { t: 'Read the <b>interval</b>', s: `Held-out difference ${sgn2(sA[1].d)} with 95% interval ${sgn2(sA[1].ci[0])} to ${sgn2(sA[1].ci[1])} on ${teI.length} patients`, m: 'With a small test set, an AUC gap of a few hundredths is often within noise. Single-split leaderboards should be read with the sampling error in mind.' },
  ], [['t', 'Try'], ['s', 'What you see'], ['m', 'What it means']], { cls: 'compact' });

  // ---- Step 6: checklist
  rb = rung(ch, 6, 'A checklist for any new method',
    `<p>The six questions below apply to a model released next month as much as to the entries above. The last column gives an instance from the scoreboard.</p>`);
  table(rb, [
    { q: '<b>Tokenisation</b>: what is a token?', l: 'A gene, a ranked gene, a binned value, a cell, an image patch, a DNA base.', e: 'Geneformer ranks genes; Cell2Sentence writes them as text; scGPT bins values.' },
    { q: '<b>Pretraining task</b>', l: 'Masked, next-token, contrastive or distribution matching. It determines what an embedding is good for.', e: 'Geneformer: masked language model; TranscriptFormer: autoregressive; State: set-to-set.' },
    { q: '<b>Evaluation split</b>', l: 'Whether test cells, perturbations, donors or cell lines are unseen.', e: `Challenge 2026 is zero-shot on 6 new cell lines; the lab above shows the same model under three splits.` },
    { q: '<b>Baselines</b>', l: 'HVG+PCA, scVI, the training mean, an additive model and a linear model in the table.', e: `${nContra} of ${EV.length} scoreboard claims are contradicted once such baselines are included.` },
    { q: '<b>Metric</b>', l: 'Whether a trivial predictor scores well (shared perturbation response, scale inflation).', e: 'VCC 2025: L1-based PDS rewards inflating prediction magnitude; Systema uses the perturbed centroid.' },
    { q: '<b>Who ran it</b>', l: 'Developer, challenge organiser or independent group.', e: `${EV.length - nInd} of ${EV.length} scoreboard claims have no independent evaluation in the brief.` },
  ], [['q', 'Question'], ['l', 'What to look for'], ['e', 'Instance']], { cls: 'compact' });
  quiz(rb, 'A model reports a higher AUC than every baseline on a random hold-out from its own training cohorts. What additional evidence would change the assessment?',
    `<p>An evaluation on a source the model has not seen (a new centre, donor or cell line), including the strongest simple baseline, with an interval. In the lab, the training-set difference was ${sgn2(sA[0].d)} but the new-hospital difference ${sgn2(sA[2].d)}.</p>`);

  // =====================================================================================
  // 5. PITFALLS
  // =====================================================================================
  ch = chapter(main, 'pitfalls', 'Pitfalls', 'Common ways of misreading a landscape, how to recognise them, and how to address them.');
  cards(ch, [
    { tag: 'Symptom: “largest model, so best”', title: 'Scale read as accuracy', body: `The largest corpus in the brief is ${fmtM(maxC)} cells, but ${nContra} of the ${EV.length} scoreboard claims are contradicted by independent evaluations. <b>Fix:</b> compare on a held-out task against a tuned simple baseline.` },
    { tag: 'Symptom: one number in an abstract', title: 'Developer-reported results', body: `${EV.filter((e) => e.s[2] === 'none').length} of ${EV.length} claims have no independent evaluation in the brief. Self-reported is not wrong, but it is untested. <b>Fix:</b> note who ran the evaluation (step 6) and wait for replication where the decision is costly.` },
    { tag: 'Symptom: “no evidence against it”', title: 'Absence of evaluation read as endorsement', body: '“Unconfirmed” describes the search, not the method. A claim with no independent test may be right or wrong. <b>Fix:</b> treat it as a hypothesis, and run the comparison on local data.' },
    { tag: 'Symptom: a leaderboard rank', title: 'Metric inflation', body: 'In VCC 2025 an L1-based score rewarded inflating the predicted magnitude, and shared perturbed-vs-control differences inflate other metrics. <b>Fix:</b> ask whether a trivial predictor (training mean, perturbed centroid) scores well on the metric.' },
    { tag: 'Symptom: adjacent numbers compared', title: 'Different units on one axis', body: 'Cells, patients, perturbations and parameters are not comparable. Figure 3 includes only the entries with a cell count. <b>Fix:</b> compare like with like, and count independent samples as well as cells.' },
    { tag: 'Symptom: pharma looks absent', title: 'Absence in the brief read as absence in the field', body: `The brief found ${kTot[4]} pharma entry and states that its search returned nothing from several large companies. Internal models are not published. <b>Fix:</b> report the search limits with the conclusion.` },
    { tag: 'Symptom: preprint cited as fact', title: 'Preprints and announcements', body: `${M.filter((d) => /bioRxiv|medRxiv|arXiv|press/.test(d.when)).length} of ${M.length} entries are preprints or announcements. Peer review does not guarantee replication, but the claims are unreviewed. <b>Fix:</b> label the publication status next to the claim.` },
    { tag: 'Symptom: dates look precise', title: 'Approximate timelines', body: `${noMonth} of ${M.length} entries have only a year in the brief, and one affiliation (LPM) is inferred. <b>Fix:</b> use the timeline for ordering by year and re-check dates at the source.` },
  ]);

  // =====================================================================================
  // 6. RECAP
  // =====================================================================================
  ch = chapter(main, 'recap', 'Recap', 'Key points.');
  recap(ch, [
    'Most 2024–26 methods share one recipe: <b>tokenise → pretrain on a large unlabelled corpus → reuse</b> for annotation, perturbation, spatial or biomarker tasks.',
    `The brief lists ${M.length} entries across six lanes; the most entries come from ${[0,1,2,3].sort((a, b) => kTot[b] - kTot[a]).slice(0, 3).map((k) => KINDS[k].toLowerCase()).join(', ').replace(/, ([^,]*)$/, ' and $1')}. Pharma is nearly absent from the published model literature the brief found.`,
    `Corpora reached about 100 million cells (${withCells.filter((d) => d.cells >= 1e8).length} entries). Cells are not independent samples, and counts of cells, patients and perturbations are not comparable.`,
    'A developer report is a <b>claim</b>; an independent evaluation is <b>evidence</b>. Both belong in the same table.',
    `Of ${EV.length} headline claims, ${nInd} have an independent evaluation in the brief and ${nContra} are contradicted; the rest are unconfirmed, which is not the same as wrong.`,
    'In independent benchmarks, zero-shot embeddings trail HVG+PCA and perturbation foundation models trail additive and mean baselines; scale has added coverage, not demonstrated accuracy on unseen conditions.',
    'The verdict depends on the <b>baseline</b>, the <b>split</b> and the <b>metric</b>: the same model can be reported as better or worse by changing one of them.',
    'Ask the six questions of any new paper: tokens, pretraining task, split, baselines, metric, who ran it.',
  ]);
  prose(ch, '<h3>In code</h3><p>A minimal local check of any pretrained embedding: compare it with HVG+PCA using the same classifier and donor-grouped cross-validation.</p>');
  code(ch, {
    Python: `import numpy as np, scanpy as sc
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import f1_score

# adata.obsm["X_fm"]: embedding from the foundation model (zero-shot); adata.obs: "label", "donor"
sc.pp.normalize_total(adata); sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
sc.pp.pca(adata, n_comps=50, use_highly_variable=True)          # baseline: HVG + PCA

def score(X):
    pred = cross_val_predict(LogisticRegression(max_iter=2000), X, adata.obs["label"],
                             groups=adata.obs["donor"], cv=GroupKFold(5))
    return f1_score(adata.obs["label"], pred, average="macro")

print("HVG+PCA  ", score(adata.obsm["X_pca"]))
print("foundation", score(adata.obsm["X_fm"]))                   # same classifier, same donor splits`,
    R: `library(glmnet)
# emb_fm: cells x d foundation-model embedding; expr: log-normalised cells x HVGs; meta: label, donor
pcs <- prcomp(expr, rank. = 50)$x                              # baseline: HVG + PCA

score <- function(X) {
  folds <- as.integer(factor(meta$donor)) %% 5 + 1               # donor-grouped folds
  pred <- character(nrow(X))
  for (k in 1:5) {
    fit <- glmnet(X[folds != k, ], meta$label[folds != k], family = "multinomial", lambda = 0.01)
    pred[folds == k] <- as.character(predict(fit, X[folds == k, ], type = "class"))
  }
  mean(pred == meta$label)                                       # accuracy; use macro-F1 for imbalance
}
c(pca = score(pcs), foundation = score(emb_fm))`,
  });
  seeAlso(main, [['frontier', 'foundation-models', 'Single-cell foundation models'], ['frontier', 'virtual-cells', 'Virtual cells and perturbation prediction'], ['frontier', 'he-to-expression', 'H&E to gene expression'], ['concepts', 'cross-validation', 'Cross-validation and data leakage'], ['concepts', 'train-test', 'Train/test split and overfitting']]);
}

// ---------------------------------------------------------------------------------------------
/** The "tokenise → pretrain → reuse" recipe, with one example per stage. */
function recipeSvg() {
  const wrap = el('div', { class: 'figure' });
  wrap.append(el('div', { class: 'fig-title' }, 'The common recipe, with examples from the brief'));
  const svg = d3.select(wrap).append('svg').attr('viewBox', '0 0 1020 250').attr('width', '100%').style('max-width', '1020px').attr('role', 'img').attr('aria-label', 'tokenise, pretrain, reuse');
  const boxes = [
    { t: '① Tokenise', l: ['gene rank (Geneformer)', 'gene + binned value (scGPT)', 'gene names as text (C2S)', 'image patch (pathology)', 'DNA base (AlphaGenome)'] },
    { t: '② Pretrain', l: ['Unlabelled cells, 10⁷–10⁸', 'Hide part of the input,', 'predict it: masked or', 'next-token loss, or', 'distribution matching'] },
    { t: '③ Reuse', l: ['Embeddings for annotation', 'Fine-tuned classifier heads', 'Perturbation prediction', 'Spatial domains, niches', 'In-context (Stack)'] },
    { t: '④ Test vs a baseline', l: ['HVG+PCA, scVI, Harmony', 'training mean, additive', 'unseen donors, cell lines', 'independent benchmarks', '(Kedzierska, VCBench, …)'], hl: true },
  ];
  const BW = 230, gap = 26;
  boxes.forEach((b, i) => {
    const x0 = 10 + i * (BW + gap), g = svg.append('g').attr('transform', `translate(${x0},14)`);
    g.append('rect').attr('width', BW).attr('height', 220).attr('rx', 10).attr('fill', b.hl ? css('--accent-soft') : css('--surface-2')).attr('stroke', css('--axis'));
    g.append('text').attr('x', 12).attr('y', 28).attr('font-weight', 700).attr('font-size', 16).attr('fill', ink()).text(b.t);
    b.l.forEach((l, j) => g.append('text').attr('x', 14).attr('y', 64 + j * 30).attr('font-size', 14.5).attr('fill', ink2()).text(l));
    if (i < boxes.length - 1) svg.append('path').attr('d', `M${x0 + BW + 6},124 L${x0 + BW + gap - 8},124 M${x0 + BW + gap - 14},118 L${x0 + BW + gap - 8},124 L${x0 + BW + gap - 14},130`).attr('fill', 'none').attr('stroke', ink2()).attr('stroke-width', 2);
  });
  return wrap;
}
