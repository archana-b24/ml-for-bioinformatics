// LLM cell annotation (GPTCelltype-style): marker list in, label out. A mock "knowledge base" stands in for the LLM.
import { header, figure, slider, select, checkbox, stats, section, steps, math, legend, seeAlso, tip, untip, el } from '../lib/ui.js';
import { frame, cat, css, ink, muted } from '../lib/plot.js';
import { sourceCard, toyBadge, coreIdea, realityCheck, useIt, cite } from './_shared.js';
const d3 = window.d3;

// What the mock "LLM" knows: canonical markers per type (textbook knowledge), plus how common the label is in its training text.
const KB = {
  'T cell': { m: ['CD3E', 'CD3D', 'CD2', 'IL7R', 'CD3G'], prior: 1.0 },
  'CD8 T cell': { m: ['CD8A', 'CD8B', 'GZMK', 'CD3E'], prior: 0.6 },
  'B cell': { m: ['MS4A1', 'CD79A', 'CD19', 'CD79B'], prior: 0.9 },
  'Plasma cell': { m: ['JCHAIN', 'MZB1', 'IGHG1', 'XBP1'], prior: 0.5 },
  'Monocyte': { m: ['LYZ', 'CD14', 'FCN1', 'S100A8'], prior: 0.8 },
  'Macrophage': { m: ['CD68', 'C1QA', 'APOE', 'MRC1'], prior: 0.8 },
  'NK cell': { m: ['NKG7', 'GNLY', 'KLRD1', 'PRF1'], prior: 0.6 },
  'Fibroblast': { m: ['COL1A1', 'DCN', 'LUM', 'PDGFRA'], prior: 0.8 },
  'Endothelial cell': { m: ['PECAM1', 'VWF', 'CLDN5', 'CDH5'], prior: 0.7 },
  'Epithelial cell': { m: ['EPCAM', 'KRT8', 'KRT18', 'KRT19'], prior: 1.0 },
  'Mast cell': { m: ['TPSAB1', 'CPA3', 'KIT', 'MS4A2'], prior: 0.4 },
  'Erythrocyte': { m: ['HBB', 'HBA1', 'HBA2', 'ALAS2'], prior: 0.4 },
};
const LABELS = Object.keys(KB);
// Mock clusters: top DE genes (ordered by log-fold-change) and the truth a careful expert would assign.
const CLUSTERS = [
  { name: 'C0', truth: 'CD8 T cell', kind: 'clean', genes: ['CD8A', 'GZMK', 'CD8B', 'CCL5', 'CD3E', 'NKG7', 'CD3D', 'GZMA', 'CD2', 'IL7R'] },
  { name: 'C1', truth: 'B cell', kind: 'clean', genes: ['MS4A1', 'CD79A', 'BANK1', 'CD19', 'CD79B', 'HLA-DQA1', 'TCL1A', 'CD74', 'FCER2', 'IGHM'] },
  { name: 'C2', truth: 'Macrophage', kind: 'clean', genes: ['C1QA', 'APOE', 'CD68', 'C1QB', 'MRC1', 'SPP1', 'CD163', 'LYZ', 'FCGR3A', 'TREM2'] },
  { name: 'C3', truth: 'Fibroblast', kind: 'clean', genes: ['DCN', 'COL1A1', 'LUM', 'COL1A2', 'PDGFRA', 'FBLN1', 'SFRP2', 'COL3A1', 'MFAP5', 'IGFBP7'] },
  { name: 'C4', truth: 'Mast cell', kind: 'clean', genes: ['TPSAB1', 'CPA3', 'MS4A2', 'KIT', 'HPGDS', 'TPSB2', 'GATA2', 'IL1RL1', 'CD9', 'SLC18A2'] },
  { name: 'C5', truth: 'Ionocyte (not in KB)', kind: 'novel', genes: ['FOXI1', 'CFTR', 'ASCL3', 'ATP6V1B1', 'BSND', 'KRT8', 'CLCNKB', 'KRT18', 'STAP1', 'TMEM61'] },
  { name: 'C6', truth: 'T–B doublets', kind: 'doublet', genes: ['MS4A1', 'CD3E', 'CD79A', 'IL7R', 'CD3D', 'CD19', 'BANK1', 'CD2', 'LTB', 'CD79B'] },
  { name: 'C7', truth: 'Low-quality / stressed', kind: 'lowq', genes: ['MT-CO1', 'MT-ND1', 'HSPA1A', 'JUN', 'FOS', 'MALAT1', 'NEAT1', 'DUSP1', 'MT-ATP6', 'IER2'] },
];
const AMBIENT = ['HBB', 'HBA1', 'LYZ']; // soup genes that leak into every cluster's DE list (e.g. lysed red cells)

function annotate(genes, { n, beta = 4 }) {
  const list = genes.slice(0, n);
  const raw = LABELS.map((lab) => {
    let s = 0; const hits = [];
    list.forEach((g, i) => { if (KB[lab].m.includes(g)) { s += 1 / (1 + 0.25 * i); hits.push(g); } });
    return { lab, s, hits, score: s + 0.35 * KB[lab].prior }; // the prior: LLMs lean towards common labels
  });
  const mx = d3.max(raw, (d) => d.score), Z = d3.sum(raw, (d) => Math.exp(beta * (d.score - mx)));
  raw.forEach((d) => { d.p = Math.exp(beta * (d.score - mx)) / Z; });
  return raw.sort((a, b) => b.p - a.p);
}
const correct = (cl, lab) => {
  if (cl.kind === 'clean') return lab === cl.truth || (cl.truth === 'CD8 T cell' && lab === 'T cell');
  return lab === 'unknown'; // for novel / doublet / low-quality the honest answer is "I can't tell"
};

export default async function render(main) {
  header(main, 'LLM cell annotation',
    'Give a large language model the top marker genes of each cluster and ask "which cell type is this?". It works well when the cluster is clean and textbook, because the model has read the textbooks. <b>It always answers, so it is confidently wrong</b> on doublets, novel types and contaminated marker lists.');
  sourceCard(main, {
    org: 'GPTCelltype, Nature Methods 2024 (academic); related agent work in CellAgent',
    when: '2024',
    links: ['gptcelltype', 'cellagent'],
    claim: 'LLM annotations from marker gene lists reached >75% concordance with manual annotations in most tissues.',
  });
  coreIdea(main, `<p>Annotation by an expert is a lookup: "<i>CD79A</i>, <i>MS4A1</i>… that's a B cell". An LLM does the same lookup from what it absorbed in training text. The input is a short list of DE genes per cluster, written as a prompt; the output is a label. It never sees the expression matrix.</p>
  <p><b>Key ML ingredient:</b> a pretrained LLM used <b>zero-shot</b> through a prompt, with no single-cell training at all. Below, a mock "knowledge base" of textbook markers plays the LLM: it scores each label by marker overlap (earlier genes count more), adds a small prior for common labels, and turns scores into a softmax "confidence".</p>`);

  let ci = 0, n = 8, amb = 0, allowUnknown = false;
  const fig = figure(main, 'Marker list → label, and where it breaks',
    'Left: the marker list sent to the "LLM" (genes that match its knowledge base are highlighted; ambient genes are marked ⚠). Right: its softmax confidence over labels. Below: every cluster at once, scored against what a careful expert would say. For doublets, novel types and low-quality clusters the right answer is "unknown".');
  toyBadge(fig, 'No language model is called; a marker-overlap lookup imitates one.');
  select(fig.controls, { label: 'Cluster', value: '0', options: CLUSTERS.map((c, i) => [String(i), `${c.name} · truth: ${c.truth}`]), onChange: (v) => { ci = +v; draw(); } });
  slider(fig.controls, { label: 'Markers sent (top n)', min: 2, max: 10, step: 1, value: n, onInput: (v) => { n = v; draw(); } });
  slider(fig.controls, { label: 'Ambient genes in list', min: 0, max: 3, step: 1, value: amb, onInput: (v) => { amb = v; draw(); } });
  checkbox(fig.controls, { label: ' allow "unknown" (abstain if overlap is weak)', checked: false, onChange: (v) => { allowUnknown = v; draw(); } });
  const st = stats(fig.root);
  const row = el('div', { class: 'fig-row' }); fig.body.append(row);
  const listDiv = el('div'), barDiv = el('div'); row.append(listDiv, barDiv);
  const tableDiv = el('div', { style: 'margin-top:10px;overflow-x:auto' }); fig.body.append(tableDiv);
  legend(fig.root, [['top label', cat(0)], ['other labels', muted()]]);

  const genesFor = (cl) => [...AMBIENT.slice(0, amb), ...cl.genes];
  const decide = (res) => (allowUnknown && res[0].s < 1.2 ? 'unknown' : res[0].lab);

  function draw() {
    const cl = CLUSTERS[ci], genes = genesFor(cl), res = annotate(genes, { n }), label = decide(res);
    listDiv.innerHTML = '';
    listDiv.append(el('div', { style: 'font-size:13px;color:var(--ink-2);margin-bottom:6px' }, `Prompt (our wording): "Identify the cell type of a cluster with these marker genes:"`));
    const chips = el('div', { style: 'display:flex;flex-wrap:wrap;gap:6px' });
    genes.slice(0, n).forEach((g, i) => {
      const inKB = LABELS.some((l) => KB[l].m.includes(g)), isAmb = i < amb;
      chips.append(el('span', { class: 'tag', style: `font-family:var(--mono);font-size:13px;padding:3px 9px;${inKB ? 'background:var(--accent-soft);color:var(--ink);font-weight:600' : ''}` }, (isAmb ? '⚠ ' : '') + g));
    });
    listDiv.append(chips);
    const ans = el('div', { class: 'card', style: 'margin-top:12px' });
    ans.innerHTML = `<div style="font-size:13px;color:var(--ink-2)">Toy "LLM" answer</div><div style="font-size:20px;font-weight:600">${label}</div>
      <div style="font-size:13px;color:var(--ink-2)">${label === 'unknown' ? 'best overlap too weak, so it abstains' : `because of ${res[0].hits.join(', ') || 'no matching markers (prior only)'}`} · truth: ${cl.truth}</div>
      <div style="font-size:13px;margin-top:4px">${correct(cl, label) ? '✓ acceptable' : '✗ wrong'}</div>`;
    listDiv.append(ans);
    barDiv.innerHTML = '';
    const top = res.slice(0, 6);
    const F = frame(barDiv, { width: 460, height: 260, x: [0, 1], y: [0, 1], yScale: d3.scaleBand().padding(0.2), margin: { left: 120 }, xLabel: 'softmax "confidence"', grid: false, axes: false });
    F.y.domain(top.map((d) => d.lab));
    F.g.append('g').attr('class', 'axis y').call(d3.axisLeft(F.y));
    F.g.append('g').attr('class', 'axis x').attr('transform', `translate(0,${F.h})`).call(d3.axisBottom(F.x).ticks(5, '.0%'));
    F.layer('b').selectAll('rect').data(top).join('rect').attr('x', 0).attr('y', (d) => F.y(d.lab)).attr('height', F.y.bandwidth()).attr('rx', 3)
      .attr('width', (d) => F.x(d.p)).attr('fill', (d, i) => (i === 0 ? cat(0) : muted()))
      .on('mousemove', (e, d) => tip(e, `<b>${d.lab}</b><br>confidence ${(100 * d.p).toFixed(0)}%<br>matched: ${d.hits.join(', ') || 'none'}`)).on('mouseleave', untip);

    // all clusters
    let ok = 0, wrongConfident = 0;
    const rows = CLUSTERS.map((c) => {
      const r = annotate(genesFor(c), { n }), lab = decide(r), good = correct(c, lab);
      if (good) ok++; else if (lab !== 'unknown' && r[0].p > 0.5) wrongConfident++;
      return `<tr${c === cl ? ' style="background:var(--surface-2)"' : ''}><td>${c.name}</td><td>${c.truth}</td><td><b>${lab}</b></td><td>${lab === 'unknown' ? '–' : (100 * r[0].p).toFixed(0) + '%'}</td><td>${good ? '✓' : '✗'}</td></tr>`;
    }).join('');
    tableDiv.innerHTML = `<table class="simple"><thead><tr><th>Cluster</th><th>Truth (expert)</th><th>Toy LLM label</th><th>Confidence</th><th>OK?</th></tr></thead><tbody>${rows}</tbody></table>`;
    st.set({ 'clusters acceptable': `${ok} / ${CLUSTERS.length}`, 'confidently wrong (>50%)': wrongConfident, 'top-label confidence (this cluster)': `${(100 * res[0].p).toFixed(0)}%` });
  }
  draw();

  section(main, 'What to notice', `<ul>
    <li>The five <b>clean</b> clusters (C0–C4) are labelled correctly: textbook markers in, textbook label out.</li>
    <li><b>C5, the ionocyte</b>, is not in the knowledge base. Two generic keratins (<i>KRT8</i>, <i>KRT18</i>) are enough for a confident "Epithelial cell": true but useless, and the distinctive <i>FOXI1</i>/<i>CFTR</i> signal is ignored. A real LLM may know ionocytes; it will fail the same way on whatever it does not know.</li>
    <li><b>C6, the doublets</b>, get a single confident label. Nothing in a marker list says "two cell types glued together"; check doublet scores before annotating.</li>
    <li><b>C7, low quality</b>: with no marker matches the answer comes from the prior alone. Tick <i>allow "unknown"</i>: C5 and C7 are flagged instead, at no cost to the clean ones. The doublet is <i>not</i> flagged, because it matches two types strongly; abstention catches weak evidence, not contradictory evidence.</li>
    <li>Add <b>ambient genes</b>: at 3, <i>HBB</i>/<i>HBA1</i>/<i>LYZ</i> lead every list. The clean clusters survive with lower confidence, but C5–C7 become "Erythrocyte" at about 97%, and the "unknown" option no longer helps because the overlap with the erythrocyte markers is strong. Contaminated DE lists are a garbage-in problem the model cannot see.</li>
    <li>Send only <b>3 markers</b>: the clean clusters still pass, but the doublet flips from "T cell" to "B cell" (whichever markers happen to rank first) and the ionocyte loses its keratins and falls back to the prior.</li>
  </ul>`);
  steps(main, 'How it works', [
    'Cluster the cells and compute DE genes per cluster (e.g. Wilcoxon, one-vs-rest).',
    'Write the top genes of each cluster into a prompt, often with tissue and species context.',
    'The LLM returns a label per cluster. GPTCelltype uses the model zero-shot: no fine-tuning, no access to the counts.',
    'Compare with manual annotation (the papers report concordance per tissue), then review disagreements by hand.',
  ]);
  math(main, 'toy score(label) = Σ_{i ∈ top-n markers} [gene_i ∈ KB(label)] / (1 + 0.25·i)  +  0.35·prior(label)\nconfidence = softmax(4 · score);  abstain if best overlap < 1.2 (when allowed)');
  realityCheck(main, `<ul>
    <li>The >75% concordance figure is the authors' own evaluation (${cite('gptcelltype')}) against manual labels, which are themselves imperfect. "Most tissues" means some tissues were lower.</li>
    <li>The model's "confidence" is not calibrated: an LLM states a label with the same fluency whether it is right or not. Low-quality clusters, doublets and rare or novel types are where errors concentrate.</li>
    <li>Results depend on the prompt, the model version and the marker list. Closed models change over time, which hurts reproducibility; record the model version and prompt in your methods.</li>
  </ul>`);
  useIt(main, `<ul>
    <li>Use an LLM as a <b>fast first draft</b> or a second opinion next to reference-based label transfer (see <a href="#/omics/sc-annotation">cell-type annotation</a>), never as the final word.</li>
    <li>Clean up first: remove doublets and low-quality cells, correct ambient RNA, and send DE genes from pseudobulk or a robust test.</li>
    <li>Ask for "unknown" explicitly in the prompt and ask the model to list the genes behind its answer; check those genes in your own dot plot.</li>
    <li>For spatial panels with a few hundred genes, the textbook markers may not be on the panel at all; the lookup then rests on whatever weak genes remain.</li>
  </ul>`);
  seeAlso(main, [['concepts', 'transformer', 'Transformers'], ['concepts', 'calibration', 'Calibration'], ['concepts', 'foundation-models', 'Foundation models'],
    ['omics', 'sc-annotation', 'Cell-type annotation'], ['omics', 'sc-qc', 'QC & doublet detection'], ['frontier', 'cell2sentence', 'Cell2Sentence'], ['frontier', 'agents', 'Agents & AI co-scientists']]);
}
