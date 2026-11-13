// The 2024–26 landscape: timeline / scale scatter of frontier methods + an honest scoreboard (claim vs independent evidence).
import { header, figure, select, checkbox, section, steps, seeAlso, legend, tip, untip, el, html } from '../lib/ui.js';
import { frame, cat, css, ink, ink2, muted } from '../lib/plot.js';
import { sourceCard, coreIdea, realityCheck, useIt, cite, REFS } from './_shared.js';
const d3 = window.d3;

const CATS = ['Cell foundation models', 'Perturbation & virtual cells', 'Independent benchmarks', 'Spatial & pathology', 'Bulk & clinical', 'Agents, LLMs & sequence'];

// t = decimal year. Month is used only where the brief (or a fetched page) gives it; otherwise mid-year and `approx`.
// cells = training/data scale in cells, only where the brief states it.
const M = [
  { n: 'UCE', c: 0, t: 2023.8, when: '2023 (bioRxiv)', org: 'Stanford', ref: 'uce', what: 'Genes as ESM2 protein embeddings, so one model spans species', approx: true },
  { n: 'scGPT', c: 0, t: 2024.3, when: '2024 (Nat Methods)', org: 'Bo Wang lab', ref: 'scgpt', cells: 33e6, what: 'Gene token + binned value; masked generative pretraining', approx: true, page: 'zero-shot-check' },
  { n: 'scFoundation', c: 0, t: 2024.45, when: '2024 (Nat Methods)', org: 'BioMap / Tsinghua', ref: 'scfoundation', what: 'Asymmetric encoder-decoder, read-depth-aware masked regression', approx: true },
  { n: 'Geneformer V2', c: 0, t: 2024.95, when: 'Dec 2024', org: 'Theodoris lab (Gladstone/Broad)', ref: 'geneformer', cells: 104e6, what: 'Rank-value encoding + BERT masked LM', page: 'geneformer' },
  { n: 'C2S-Scale', c: 0, t: 2025.28, when: 'Apr 2025 (bioRxiv)', org: 'Google Research/DeepMind + Yale', ref: 'c2s', what: 'Cells written as gene-name text; Gemma-2 LM (2B/27B)', page: 'cell2sentence' },
  { n: 'TranscriptFormer', c: 0, t: 2025.32, when: 'Apr 2025 (bioRxiv)', org: 'CZI / Biohub', ref: 'transcriptformer', cells: 112e6, what: 'Autoregressive next gene + count; 12 species' },
  { n: 'rbio1', c: 0, t: 2025.63, when: 'Aug 2025 (bioRxiv)', org: 'CZI', ref: 'rbio1', what: 'LLM post-trained with RL, a virtual-cell model as soft verifier' },
  { n: 'Stack', c: 0, t: 2026.04, when: 'Jan 2026', org: 'Arc Institute', ref: 'stackNews', what: 'Attends across genes and cells; in-context learning from prompt cells', page: 'in-context' },
  { n: 'Tahoe-100M', c: 1, t: 2025.14, when: 'Feb 2025 (bioRxiv)', org: 'Tahoe', ref: 'tahoe', cells: 100e6, what: 'Perturbation atlas (data resource)' },
  { n: 'LPM', c: 1, t: 2025.22, when: 'Mar 2025 (arXiv)', org: 'Genentech (inferred from GitHub org)', ref: 'lpm', what: 'Perturbation × readout × context, disentangled', page: 'perturbation-tensor' },
  { n: 'CellFlow', c: 1, t: 2025.27, when: 'Apr 2025 (bioRxiv)', org: 'Theis lab', ref: 'cellflow', what: 'Conditional flow matching, control → perturbed', page: 'flow-matching' },
  { n: 'X-Atlas/Orion', c: 1, t: 2025.44, when: 'Jun 2025 (bioRxiv)', org: 'Xaira', ref: 'xaira', cells: 8e6, what: '8M-cell Perturb-seq (data resource)' },
  { n: 'State', c: 1, t: 2025.5, when: '2025', org: 'Arc Institute', ref: 'stateNews', what: 'Set transformer maps a control set to a perturbed set', approx: true, page: 'state-sets' },
  { n: 'TxPert', c: 1, t: 2025.55, when: '2025', org: 'Valence Labs / Recursion', ref: 'txpert', what: 'GNN over knowledge graphs for unseen perturbations', approx: true },
  { n: 'VCC 2025', c: 1, t: 2025.85, when: '2025', org: 'Arc Institute', ref: 'vcc2025', what: 'Open challenge: H1 hESC, 300 CRISPRi perturbations', approx: true, page: 'metric-gaming' },
  { n: 'PRiMeFlow', c: 1, t: 2026.28, when: 'Apr 2026 (arXiv)', org: 'Altos (VCC generalist prize)', ref: 'primeflow', what: 'Flow matching for perturbation response', page: 'flow-matching' },
  { n: 'PerturBench', c: 2, t: 2024.6, when: 'Aug 2024 (arXiv)', org: 'independent', ref: 'perturbench', what: 'Perturbation-model benchmark', page: 'perturbation-arena' },
  { n: 'Ahlmann-Eltze', c: 2, t: 2025.5, when: '2025 (Nat Methods)', org: 'EMBL', ref: 'ahlmann', what: 'Simple baselines beat foundation models on perturbation prediction', approx: true, page: 'perturbation-arena' },
  { n: 'PertEval-scFM', c: 2, t: 2025.53, when: '2025 (ICML)', org: 'independent', ref: 'perteval', what: 'Benchmark of scFM embeddings for perturbations', approx: true, page: 'perturbation-arena' },
  { n: 'Kedzierska', c: 2, t: 2025.4, when: '2025 (Genome Biol)', org: 'independent', ref: 'kedzierska', what: 'Zero-shot scFM embeddings vs HVG+PCA / scVI / Harmony', approx: true, page: 'zero-shot-check' },
  { n: 'VCBench', c: 2, t: 2026.46, when: 'Jun 2026 (bioRxiv)', org: 'independent', ref: 'vcbench', what: 'Simple baselines match/beat FMs on 4 of 5 axes', page: 'perturbation-arena' },
  { n: 'CellPLM', c: 3, t: 2024.3, when: '2024 (ICLR)', org: '—', what: 'Cells are tokens, tissue is the sentence', approx: true, page: 'masked-cells' },
  { n: 'Prov-GigaPath', c: 3, t: 2024.4, when: '2024 (Nature)', org: 'Microsoft / Providence', ref: 'gigapath', what: 'Whole-slide pathology foundation model', approx: true, page: 'he-to-expression' },
  { n: 'HEST-1k', c: 3, t: 2024.45, when: 'Jun 2024 (arXiv)', org: 'independent', ref: 'hest', what: 'H&E → expression benchmark; best Pearson ≈ 0.41', page: 'he-to-expression' },
  { n: 'OCTO-vc', c: 3, t: 2024.96, when: 'Dec 2024 (press release)', org: 'Noetik', ref: 'noetik', what: 'Industry multimodal tumour model (announcement)' },
  { n: 'Nicheformer', c: 3, t: 2025.5, when: '2025 (Nat Methods)', org: 'Theis lab', ref: 'nicheformer', cells: 110e6, what: 'Pretrained on SpatialCorpus-110M; spatial labels onto dissociated cells', approx: true, page: 'spatial-niches' },
  { n: 'Novae', c: 3, t: 2025.6, when: '2025 (Nat Methods)', org: '—', ref: 'novae', what: 'Graph attention + SwAV, zero-shot spatial domains', approx: true, page: 'spatial-niches' },
  { n: 'M-Optimus-1', c: 3, t: 2026.3, when: '2026', org: 'Bioptimus', ref: 'moptimus', what: 'Multimodal pathology model (vendor numbers)', approx: true, page: 'he-to-expression' },
  { n: 'Tempus IPS', c: 4, t: 2024.85, when: 'Nov 2024 (medRxiv)', org: 'Tempus', ref: 'tempus', what: 'ML on TMB + 8 RNA genes; dev n=1707, validation n=1600', page: 'clinical-signature' },
  { n: 'BulkFormer', c: 4, t: 2025.44, when: 'Jun 2025 (bioRxiv)', org: '—', ref: 'bulkformer', what: 'Masked-expression foundation model for bulk RNA' },
  { n: 'GPTCelltype', c: 5, t: 2024.25, when: '2024 (Nat Methods)', org: '—', ref: 'gptcelltype', what: 'LLM labels clusters from marker genes', approx: true, page: 'llm-annotation' },
  { n: 'CellAgent', c: 5, t: 2024.52, when: 'Jul 2024 (arXiv)', org: '—', ref: 'cellagent', what: 'LLM agent for scRNA analysis', page: 'agents' },
  { n: 'AI co-scientist', c: 5, t: 2025.13, when: '19 Feb 2025 (blog)', org: 'Google', ref: 'coscientist', what: 'Gemini multi-agent system, Elo tournament of hypotheses', page: 'agents' },
  { n: 'Biomni', c: 5, t: 2025.42, when: '2 Jun 2025 (bioRxiv)', org: 'Stanford', ref: 'biomni', what: 'Agent with 150 tools, 105 software packages, 59 databases', page: 'agents' },
  { n: 'AlphaGenome', c: 5, t: 2025.46, when: 'Jun 2025', org: 'Google DeepMind', what: 'DNA → RNA tracks from up to 1 Mb of sequence', page: 'sequence-to-expression' },
];

const SCORE = [
  ['Zero-shot cell embeddings (Geneformer, scGPT)', 'General-purpose embeddings that transfer without fine-tuning (developers)', `${cite('kedzierska', 'Kedzierska et al. 2025')}: zero-shot is worse than HVG+PCA, scVI and Harmony`, 'Contradicted', 'zero-shot-check'],
  ['Foundation models for perturbation prediction', 'Learn transferable perturbation responses (various developers)', `${cite('ahlmann', 'Ahlmann-Eltze et al. 2025')}: additive beats 5 FMs on doubles, nothing beats the mean on unseen singles; ${cite('vcbench', 'VCBench')}: simple baselines match or beat FMs on 4 of 5 axes; ${cite('perteval', 'PertEval-scFM')}`, 'Contradicted', 'perturbation-arena'],
  ['State', '"First to consistently beat linear baselines" (Arc, self-reported)', 'No independent replication in our brief. Loss reported as MMD by secondary sources only.', 'Unconfirmed', 'state-sets'],
  ['Stack', 'In-context prediction of new conditions (Arc, self-reported)', 'No independent evaluation in our brief', 'Unconfirmed', 'in-context'],
  ['C2S-Scale', 'LLM on cell sentences; silmitasertib hypothesis (Google/Yale preprint)', 'The discovery rests on a single in-vitro validation', 'Early', 'cell2sentence'],
  ['VCC 2025 metrics', 'PDS, DES and MAE rank virtual-cell models', `${cite('vcc2025', "Arc's own wrap-up")}: almost nobody beat the mean on MAE; L1-based PDS rewards inflating prediction magnitude`, 'Metric caveat', 'metric-gaming'],
  ['H&E → expression', 'Pathology FMs predict expression from histology', `${cite('hest', 'HEST-1k')}: the best model reaches Pearson ≈ 0.41 on 50 HVGs. M-Optimus numbers are vendor claims`, 'Modest', 'he-to-expression'],
  ['Tempus Immune Profile Score', 'TMB + 8 RNA genes predicts immunotherapy benefit (dev n=1707, val n=1600)', 'Company-run validation in a medRxiv preprint; no independent cohort in our brief', 'Needs external', 'clinical-signature'],
  ['GPTCelltype', '>75% concordance with manual labels in most tissues (authors)', 'Concordance with manual labels, not ground truth; novel or contaminated clusters are not covered', 'Plausible', 'llm-annotation'],
  ['Biomni / AI co-scientist', 'General agents and ranked hypotheses (developers)', 'No independent benchmark in our brief', 'Unconfirmed', 'agents'],
];
const VERDICT = { Contradicted: 'bad', Unconfirmed: 'warn', Early: 'warn', 'Metric caveat': 'warn', Modest: 'warn', 'Needs external': 'warn', Plausible: 'good' };

export default async function render(main) {
  header(main, 'The 2024–26 landscape',
    'In three years single-cell ML went from 30-million-cell models to 100-million-cell models, LLMs that read cells as text, and AI agents. <b>Independent benchmarks have repeatedly found simple baselines just as good.</b> This page maps who is building what, and keeps score.');

  sourceCard(main, {
    org: 'Compiled from our literature brief (35 methods, datasets and benchmarks)',
    when: 'Brief compiled 2026-11-12',
    links: [['Literature brief (docs/frontier_research.md)', 'docs/frontier_research.md']],
    claim: 'Each dot links to its source and, where one exists, to the page in this tab that rebuilds it on mock data.',
    selfReported: false,
  });

  coreIdea(main, `<p>Almost every method here has the same recipe: <b>tokenise</b> biology (genes, cells, patches of tissue, DNA bases), <b>pretrain</b> a transformer with a self-supervised task on a huge corpus, then <b>reuse</b> the model for annotation, perturbation prediction or biomarkers. The three ML ingredients to understand are masked or next-token prediction, distribution-level generative models (set losses, flow matching), and LLM agents that call tools.</p>
  <p>The useful question for each one is not "how big?" but "<b>does it beat a well-tuned simple baseline on data it has not seen?</b>"</p>`);

  // ------------------------------------------------------------ timeline figure
  const fig = figure(main, 'Who built what, when, and on how much data',
    'Hover a dot for the source; click it to open the page that rebuilds it. Hollow dots: our brief gives only the year, so they sit mid-year.');
  let view = 'lanes';
  const on = CATS.map(() => true);
  select(fig.controls, { label: 'Vertical axis', value: view, options: [['lanes', 'Category lanes'], ['scale', 'Data scale (cells, log)']], onChange: (v) => { view = v; paint(); } });
  CATS.forEach((c, i) => checkbox(fig.controls, { label: c.split(' ')[0] === 'Cell' ? 'Foundation' : c.split(' ')[0].replace(',', ''), checked: true, onChange: (v) => { on[i] = v; paint(); } }));
  legend(fig.body, CATS.map((c, i) => [c, cat(i)]));
  const holder = el('div'); fig.body.append(holder);
  const note = html('', 'div', 'fig-caption'); fig.body.append(note);

  function paint() {
    holder.innerHTML = '';
    const lanes = view === 'lanes';
    const data = M.filter((d) => on[d.c] && (lanes || d.cells));
    const F = frame(holder, {
      width: 1020, height: lanes ? 560 : 420, x: [2023.6, 2026.95], y: lanes ? [0, 1] : [5e6, 2e8],
      yScale: lanes ? d3.scaleLinear() : d3.scaleLog(), xLabel: 'publication date', yLabel: lanes ? '' : 'cells in training / data set',
      grid: !lanes, margin: { left: lanes ? 12 : 60, right: 20 },
    });
    F.g.select('.axis.x').call(d3.axisBottom(F.x).tickValues([2024, 2024.5, 2025, 2025.5, 2026, 2026.5]).tickFormat((v) => (v % 1 ? 'mid ' + Math.floor(v) : String(v))));
    if (lanes) F.g.select('.axis.y').remove();
    else F.g.select('.axis.y').call(d3.axisLeft(F.y).tickValues([5e6, 1e7, 2e7, 5e7, 1e8, 2e8]).tickFormat(d3.format('.0s')));
    const laneH = F.h / CATS.length;
    if (lanes) {
      CATS.forEach((c, i) => {
        F.layer('lanes').append('rect').attr('x', 0).attr('width', F.w).attr('y', i * laneH).attr('height', laneH).attr('fill', i % 2 ? css('--surface-2') : 'none').attr('opacity', 0.6);
        F.layer('lanes').append('text').attr('x', 4).attr('y', i * laneH + 13).attr('font-size', 11).attr('fill', muted()).text(c);
      });
    }
    // place dots; in lanes view, stagger within a lane to avoid label collisions
    const byLane = d3.group(data, (d) => d.c);
    const pos = new Map();
    byLane.forEach((arr) => {
      const ends = []; // greedy row packing so labels never overlap within a lane
      arr.slice().sort((a, b) => a.t - b.t).forEach((d) => {
        const x0 = F.x(d.t);
        let row = ends.findIndex((e) => e < x0 - 8); if (row < 0) { row = ends.length; ends.push(0); }
        ends[row] = x0 + 14 + d.n.length * 6.6;
        d._row = row;
      });
      arr.forEach((d) => pos.set(d, [F.x(d.t), lanes ? d.c * laneH + 30 + d._row * 17 : F.y(d.cells)]));
    });
    F.layer('now').append('line').attr('x1', F.x(2026.87)).attr('x2', F.x(2026.87)).attr('y1', 0).attr('y2', F.h).attr('stroke', muted()).attr('stroke-dasharray', '3 3');
    F.layer('now').append('text').attr('x', F.x(2026.87) - 4).attr('y', F.h - 4).attr('text-anchor', 'end').attr('font-size', 11).attr('fill', muted()).text('today');
    const g = F.layer('dots').selectAll('g').data(data).join('g').attr('transform', (d) => `translate(${pos.get(d)})`)
      .style('cursor', (d) => (d.page ? 'pointer' : 'default'))
      .on('click', (e, d) => { if (d.page) { untip(); location.hash = `#/frontier/${d.page}`; } })
      .on('mousemove', (e, d) => tip(e, `<b>${d.n}</b> · ${d.when}<br>${d.org}<br>${d.what}${d.cells ? `<br>${d3.format('.3~s')(d.cells).replace('M', ' million')} cells` : ''}${d.ref ? `<br><i>${REFS[d.ref][0]}</i>` : ''}${d.page ? '<br><b>Click to open the rebuild ▸</b>' : ''}`))
      .on('mouseleave', untip);
    g.append('circle').attr('r', (d) => (d.page ? 7 : 5)).attr('fill', (d) => (d.approx ? css('--surface') : cat(d.c)))
      .attr('stroke', (d) => (d.approx ? cat(d.c) : css('--surface'))).attr('stroke-width', (d) => (d.approx ? 2.5 : 1.5));
    g.append('text').attr('x', 9).attr('y', 4).attr('font-size', 11.5).attr('fill', ink()).text((d) => d.n);
    note.innerHTML = lanes ? `${data.length} entries shown. Larger dots have a rebuild page in this tab.`
      : `Only the ${data.length} entries whose cell count appears in our brief are shown. Model sizes, patient counts and perturbation counts are not comparable units, so they are left off this axis.`;
  }
  paint();

  // ------------------------------------------------------------ scoreboard
  section(main, 'Scoreboard: claim vs independent evidence', '<p>Verdicts are about the <i>claim as stated</i>, using only the evidence in our brief. "Unconfirmed" means no independent test has been published that we know of; it does not mean wrong.</p>');
  const tbl = el('table', { class: 'simple' });
  tbl.innerHTML = '<thead><tr><th>Method</th><th>Claim (by whom)</th><th>Independent evidence</th><th>Verdict</th></tr></thead>';
  const tb = el('tbody'); tbl.append(tb);
  SCORE.forEach(([m, claim, ev, v, page]) => {
    const tr = el('tr');
    tr.innerHTML = `<td><a href="#/frontier/${page}">${m}</a></td><td>${claim}</td><td>${ev}</td><td><span class="tag" style="border-left:3px solid var(--${VERDICT[v]});border-radius:4px">${v}</span></td>`;
    tb.append(tr);
  });
  const wrap = el('div', { style: 'overflow-x:auto;max-width:1060px' }, tbl); main.append(wrap);

  section(main, 'What to notice', `<ul>
    <li>Switch to <b>Data scale</b>: the corpora reached about 100 million cells in 2024–25 (Geneformer V2, TranscriptFormer, Tahoe-100M, SpatialCorpus-110M), roughly three times scGPT's 33 million.</li>
    <li>Turn off everything except <b>Independent</b>: the benchmarks cluster in 2025–26, after the models they test, and almost all of them say "check against a simple baseline first".</li>
    <li>The <b>Perturbation</b> lane is the busiest in 2025. That is the "virtual cell" push, and it is where the evaluation fights (Systema, VCC metrics) are happening.</li>
    <li>No 2024–26 transcriptomics foundation-model papers turned up from Novartis, Sanofi, J&amp;J, Bayer, Insitro, Illumina, Isomorphic or Meta. In our brief, industry shows up mostly as data generators and partners (Tahoe, Xaira, Owkin ${cite('mosaic', 'MOSAIC')}, Noetik).</li>
  </ul>`);

  steps(main, 'How to read any frontier paper', [
    '<b>Tokenisation:</b> what is a token (a gene, a ranked gene, a binned value, a cell, a histology patch, a DNA base)?',
    '<b>Pretraining task:</b> masked prediction, next-token prediction, contrastive, or distribution matching? That decides what the embedding is good at.',
    '<b>Evaluation split:</b> are test cells, perturbations, donors or cell lines truly unseen?',
    '<b>Baselines:</b> are HVG+PCA, scVI, the training mean, the additive model and a linear model in the table?',
    '<b>Metric:</b> could a trivial predictor score well (shared perturbation response, scale inflation)?',
    '<b>Who ran it:</b> developer-reported, challenge-reported, or independent?',
  ]);

  realityCheck(main, `The pattern across 2024–26 is consistent: ${cite('kedzierska', 'zero-shot embeddings')} trail HVG+PCA; ${cite('ahlmann', 'perturbation foundation models')} trail additive and mean baselines; in ${cite('vcc2025', 'VCC 2025')} hybrids of statistics and deep learning won and almost nobody beat the mean on MAE. Scale has clearly bought <i>coverage</i> (species, tissues, modalities). It has not yet clearly bought <i>accuracy on unseen conditions</i>. The ${cite('vcc2026', 'VCC 2026')} zero-shot round on 6 new cell lines is the next real test; its winners had not been announced as of 2026-11-12.`);

  useIt(main, `<ul>
    <li>Use foundation models as <b>feature extractors you benchmark</b>, not as oracles: run your HVG+PCA / scVI pipeline in parallel and keep whichever wins on <i>your</i> held-out donors.</li>
    <li>For perturbation screens, start from the arena's baselines (mean, additive, linear) before any "virtual cell".</li>
    <li>For spatial data, niche models (Novae, Nicheformer) are the most immediately useful, because they give you domains without labels.</li>
    <li>Agents are useful for drafting analysis code. Treat their output like a junior analyst's: rerun it and check every number.</li>
  </ul>`);

  seeAlso(main, [['frontier', 'perturbation-arena', 'Perturbation baselines arena'], ['frontier', 'zero-shot-check', 'Do foundation embeddings beat PCA?'],
    ['concepts', 'foundation-models', 'Self-supervision & foundation models'], ['concepts', 'transformer', 'Transformers'], ['concepts', 'attention', 'Attention'], ['concepts', 'embeddings', 'Embeddings'], ['omics', 'overview', 'The mock study & ML map']]);
}
