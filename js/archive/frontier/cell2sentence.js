// Cell2Sentence / C2S-Scale: write each cell as gene names ordered by expression, then learn from the text.
import { header, figure, slider, select, stats, section, steps, math, legend, seeAlso, tip, untip, el } from '../../lib/ui.js';
import { frame, heatmap, cat, css, ink, muted, sequential } from '../../lib/plot.js';
import { rng } from '../../lib/data.js';
import { sourceCard, toyBadge, coreIdea, realityCheck, useIt, cite } from '../../frontier/_shared.js';
const d3 = window.d3;

const HK = { MALAT1: 40, TMSB4X: 30, ACTB: 26, EEF1A1: 22, RPS27: 20, RPL13: 18, B2M: 16, GAPDH: 12, FTL: 10, 'MT-CO1': 14 };
const TYPES = ['CD4 T', 'CD8 T', 'B cell', 'Monocyte', 'NK cell', 'Dendritic'];
const MARK = [
  { CD3E: 5, IL7R: 1.6, LTB: 2.2, CD4: 0.6, CCR7: 1.1, TCF7: 0.8 },
  { CD3E: 5, LTB: 1.2, IL7R: 0.8, CD8A: 1, CD8B: 0.7, GZMK: 1, CCL5: 1.6 },
  { MS4A1: 4, CD79A: 5, BANK1: 2, 'HLA-DRA': 4, CD74: 6 },
  { LYZ: 8, CD14: 3, S100A8: 6, S100A9: 6, FCN1: 3, CD74: 4 },
  { NKG7: 6, GNLY: 6, KLRD1: 3, PRF1: 2, CCL5: 3 },
  { FCER1A: 2.5, CLEC10A: 2, CD1C: 2, 'HLA-DRA': 5, CD74: 7, LYZ: 2 },
];
const NOISE = { JUN: 1.2, FOS: 1.2, HSPA1A: 0.8, IFI6: 0.6, ISG15: 0.6, XIST: 1, DUSP1: 1, TPT1: 1.5 };
const GENES = [...new Set([...Object.keys(HK), ...MARK.flatMap((m) => Object.keys(m)), ...Object.keys(NOISE)])];
const SHORT = ['CD4 T', 'CD8 T', 'B', 'Mono', 'NK', 'DC'];
const G = GENES.length, T = TYPES.length, PER = 60, KMAX = 40;

function makeCells(depth, seed) {
  const r = rng(seed), cells = [];
  for (let t = 0; t < T; t++) for (let i = 0; i < PER; i++) {
    const size = depth * Math.exp(r.normal(0, 0.3));
    const counts = GENES.map((g) => r.poisson(size * (HK[g] || MARK[t][g] || NOISE[g] || 0.15)));
    const jitter = GENES.map(() => r()); // random tie-break among equal counts
    const order = d3.range(G).filter((j) => counts[j] > 0).sort((a, b) => counts[b] - counts[a] || jitter[a] - jitter[b]);
    cells.push({ type: t, idx: i, counts, order, split: i % 2 ? 'test' : 'train' });
  }
  return cells;
}
// Bernoulli naive Bayes on "word appears among the first k words".
function nbEval(cells, k) {
  const tr = cells.filter((c) => c.split === 'train'), te = cells.filter((c) => c.split === 'test');
  const cnt = Array.from({ length: T }, () => new Float64Array(G)), n = new Float64Array(T);
  tr.forEach((c) => { n[c.type]++; c.order.slice(0, k).forEach((j) => { cnt[c.type][j]++; }); });
  const lp = cnt.map((row, t) => Array.from(row, (v) => Math.log((v + 1) / (n[t] + 2))));
  const lq = cnt.map((row, t) => Array.from(row, (v) => Math.log(1 - (v + 1) / (n[t] + 2))));
  const base = lq.map((row) => d3.sum(row));
  const conf = Array.from({ length: T }, () => new Array(T).fill(0));
  let ok = 0;
  te.forEach((c) => {
    const w = c.order.slice(0, k);
    const sc = d3.range(T).map((t) => base[t] + d3.sum(w, (j) => lp[t][j] - lq[t][j]));
    const pred = d3.maxIndex(sc); conf[c.type][pred]++; if (pred === c.type) ok++;
  });
  return { acc: ok / te.length, conf: conf.map((r) => r.map((v) => v / d3.sum(r))) };
}

export default async function render(main) {
  header(main, 'Cell2Sentence / C2S-Scale',
    'Write each cell as <b>plain text: its gene names, ordered from most to least expressed</b>. Once cells are sentences, any language model can read, classify or even generate them, and you can mix cell sentences with ordinary English in one prompt.');
  sourceCard(main, {
    org: 'Google Research / Google DeepMind with Yale',
    when: 'bioRxiv 2025 (preprint)',
    links: ['c2s'],
    claim: 'Gemma-2 causal language models (2B and 27B parameters) trained on cells written as text can be used for single-cell tasks; the paper highlights a drug hypothesis (silmitasertib) backed by an in-vitro experiment.',
    status: 'preprint',
  });
  coreIdea(main, `<p>Most single-cell foundation models invent special gene tokens and a custom architecture. Cell2Sentence keeps the model unchanged and changes the data instead: <code>MALAT1 TMSB4X ACTB … LYZ S100A8 CD14 …</code>. Ranking by expression is lossy (the counts are gone), but rank order keeps most of what distinguishes cell types, and the text is something an off-the-shelf LLM understands.</p>
  <p><b>Key ML ingredient:</b> a standard causal (next-token) <a href="#/archive/topics/transformer">language model</a>, here Gemma-2, fine-tuned on cell sentences together with natural-language metadata. Below, the LLM is replaced by the simplest possible text classifier, a <a href="#/archive/topics/naive-bayes">naive Bayes</a> bag of words, to show how much information the first <i>k</i> words carry.</p>`);

  const DATA = { deep: makeCells(1, 21), shallow: makeCells(0.25, 21) };
  const curves = Object.fromEntries(Object.entries(DATA).map(([key, cells]) => [key, d3.range(1, KMAX + 1).map((k) => ({ k, acc: nbEval(cells, k).acc }))]));
  let k = 14, depth = 'deep';

  const fig = figure(main, 'Cells as sentences → a bag-of-top-k-words classifier',
    `${T} immune cell types × ${PER} cells, ${G} genes. Odd-numbered cells are held out for testing. The classifier only sees <i>which</i> gene names appear among the first <i>k</i> words, not their counts. "Shallow" divides every expected count by 4, mimicking low-UMI cells where ties and dropouts scramble the order.`);
  toyBadge(fig, 'Naive Bayes stands in for the Gemma-2 language model.');
  slider(fig.controls, { label: 'Words kept (k)', min: 1, max: KMAX, step: 1, value: k, onInput: (v) => { k = v; draw(); } });
  select(fig.controls, { label: 'Sequencing depth', value: depth, options: [['deep', 'deep'], ['shallow', 'shallow (¼ UMIs)']], onChange: (v) => { depth = v; draw(); } });
  const st = stats(fig.root);
  const sentBox = el('div', { style: 'font-family:var(--mono);font-size:12.5px;line-height:1.7;background:var(--code-bg);border-radius:8px;padding:8px 12px;margin-bottom:12px' });
  fig.body.append(sentBox);
  const row = el('div', { class: 'fig-row' }); fig.body.append(row);
  const curveDiv = el('div'), confDiv = el('div'); row.append(curveDiv, confDiv);
  legend(fig.root, [['deep sequencing', cat(0)], ['shallow (¼ UMIs)', cat(1)], ['chance (1/6)', muted()]]);

  const C = frame(curveDiv, { width: 500, height: 330, x: [1, KMAX], y: [0, 1], xLabel: 'k = number of leading gene names kept', yLabel: 'held-out accuracy' });
  C.layer('chance').append('line').attr('x1', 0).attr('x2', C.w).attr('y1', C.y(1 / T)).attr('y2', C.y(1 / T)).attr('stroke', muted()).attr('stroke-dasharray', '4 4').attr('stroke-width', 1.5);
  Object.keys(curves).forEach((key, i) => {
    C.layer('lines').append('path').datum(curves[key]).attr('fill', 'none').attr('stroke', cat(i)).attr('stroke-width', 2)
      .attr('d', d3.line().x((d) => C.x(d.k)).y((d) => C.y(d.acc)));
    C.layer('pts').selectAll(null).data(curves[key]).join('circle').attr('cx', (d) => C.x(d.k)).attr('cy', (d) => C.y(d.acc)).attr('r', 4)
      .attr('fill', cat(i)).attr('stroke', css('--surface')).attr('stroke-width', 1.2).attr('opacity', 0.0001)
      .on('mousemove', (e, d) => tip(e, `${key}: k=${d.k}<br>accuracy ${(100 * d.acc).toFixed(1)}%`)).on('mouseleave', untip);
  });
  const kLine = C.layer('k').append('line').attr('y1', 0).attr('y2', C.h).attr('stroke', ink()).attr('stroke-width', 1).attr('stroke-dasharray', '2 3');

  function draw() {
    const cells = DATA[depth], res = nbEval(cells, k);
    kLine.attr('x1', C.x(k)).attr('x2', C.x(k));
    sentBox.innerHTML = '';
    [0, 1, 3, 4].forEach((t) => {
      const c = cells.find((x) => x.type === t && x.split === 'test');
      const words = c.order.slice(0, Math.min(c.order.length, 26)).map((j, i) => `<span style="${i < k ? 'color:var(--ink);font-weight:600' : 'color:var(--muted)'}">${GENES[j]}</span>`).join(' ');
      sentBox.append(el('div', { html: `<span style="color:var(--ink-2)">${TYPES[t]}:</span> ${words}${c.order.length > 26 ? ' <span style="color:var(--muted)">…</span>' : ''}` }));
    });
    confDiv.innerHTML = '';
    confDiv.append(el('div', { style: 'font-size:13px;color:var(--ink-2);margin:0 0 4px' }, `Confusion at k = ${k} (rows: true type, columns: predicted; fraction of row)`));
    heatmap(confDiv, res.conf, { cell: 44, rowLabels: TYPES, colLabels: SHORT, labelW: 84, labelH: 70, color: sequential(0, 1), fmt: (v) => (v ? v.toFixed(2) : ''),
      onHover: (e, i, j, v) => tip(e, `true ${TYPES[i]} → predicted ${TYPES[j]}: ${(100 * v).toFixed(0)}%`) });
    const meanLen = d3.mean(cells, (c) => c.order.length);
    st.set({ 'held-out accuracy': `${(100 * res.acc).toFixed(1)}%`, 'mean sentence length (genes detected)': meanLen.toFixed(1), 'CD4 T cells called CD8 T': `${(100 * res.conf[0][1]).toFixed(0)}%` });
  }
  draw();

  section(main, 'What to notice', `<ul>
    <li>At <b>k = 1</b> every sentence starts with the same housekeeping word, so the classifier is at chance. The first several words of every cell are housekeeping genes; C2S orders by raw expression, without Geneformer's per-gene normalisation.</li>
    <li>Accuracy climbs once k passes the housekeeping block and the markers (<i>LYZ</i>, <i>GNLY</i>, <i>CD79A</i>) enter the window.</li>
    <li>The confusion matrix shows that <b>CD4 vs CD8 T cells</b> are the last pair to separate: they share <i>CD3E</i>, and their distinguishing genes are lowly expressed, so they sit late in the sentence.</li>
    <li>Switch to <b>shallow</b>: sentences get shorter, ties are broken at random and low-expressed markers drop out, so the plateau sits lower. Rank-based text inherits all the sparsity of the counts.</li>
  </ul>`);
  steps(main, 'How it works', [
    'For each cell, sort genes by expression and write their names as a space-separated string. Keep the first k words (long sentences are truncated).',
    'Wrap the sentence in a natural-language prompt, for example (our wording) "Predict the cell type of: …".',
    'Fine-tune a pretrained causal LM (Gemma-2 in C2S-Scale) on millions of such prompts, so that one model handles several cell-level tasks through text.',
    'Use the model\'s text outputs (labels, generated sentences, hypotheses) as leads, and test them in the lab.',
  ]);
  math(main, 'sentence(cell) = [gene names sorted by expression, descending][:k]\ntoy classifier (Bernoulli naive Bayes), with p_tg = P(gene g in first k | type t):\n  score(t) = log P(t) + Σ_g [ x_g·log p_tg + (1 − x_g)·log(1 − p_tg) ]');
  realityCheck(main, `<ul>
    <li>C2S-Scale is a <b>preprint</b> (${cite('c2s')}). Its headline biology result, the silmitasertib hypothesis, rests on a <b>single in-vitro validation</b> so far.</li>
    <li>Converting counts to ranks discards magnitude. Independent zero-shot evaluations of other single-cell foundation models (${cite('kedzierska', 'Kedzierska et al. 2025')}) found them behind HVG + PCA baselines; check any C2S embedding against such a baseline on your own data.</li>
    <li>A big LLM is expensive to run per cell. For plain annotation, the toy above shows that a bag of marker words already does most of the work.</li>
  </ul>`);
  useIt(main, `<ul>
    <li>The "cell as text" trick is useful even without C2S: writing the top marker genes of a cluster as text is exactly how <a href="#/archive/frontier/llm-annotation">LLM-based annotation</a> works.</li>
    <li>Watch sequencing depth: rank sentences from shallow 10x or targeted spatial panels (a few hundred genes) are much shorter and noisier than the training data.</li>
    <li>Treat generated cells or LLM-proposed drug hypotheses as leads to prioritise experiments, never as results.</li>
  </ul>`);
  seeAlso(main, [['concepts', 'transformer', 'Transformers'], ['concepts', 'naive-bayes', 'Naive Bayes'], ['concepts', 'embeddings', 'Embeddings'], ['concepts', 'foundation-models', 'Foundation models'],
    ['omics', 'sc-annotation', 'Cell-type annotation'], ['frontier', 'geneformer', 'Geneformer'], ['frontier', 'llm-annotation', 'LLM cell annotation']]);
}
