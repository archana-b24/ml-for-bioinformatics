// Single-cell foundation models, lesson format.
// Arc: what → why → where → how it works (6 steps: rank-value encoding → masked-gene pretraining → cell as text (C2S)
//      → LLM annotation → zero-shot lab against HVG + PCA → in-context learning) → pitfalls → recap.
// Every figure is a teaching toy on simulated tumour-microenvironment cells (js/frontier/_fm_toy.js), not a reproduction.
import { header, figure, slider, select, button, checkbox, stats, legend, callout, seeAlso, tip, untip, el, html } from '../lib/ui.js';
import { frame, cat, css, ink, ink2, muted } from '../lib/plot.js';
import { rng } from '../lib/data.js';
import { lessonNav, chapter, prose, rung, quiz, cards, table, code, recap } from '../lib/lesson.js';
import { sourceCard, toyBadge, cite, pca } from './_shared.js';
import { TYPES, MARKERS, GENES, G, gclass, L, makeCells, makeCorpus, pretrain, encode, embedCell, hvgPca, knnMetrics, centroidAcc } from './_fm_toy.js';
const d3 = window.d3;

const f2 = (v) => v.toFixed(2);
const f1 = (v) => v.toFixed(1);
const pc = (v) => Math.round(100 * v) + '%';
const sum = (a) => a.reduce((s, v) => s + v, 0);
const typeMarkers = TYPES.map((_, t) => new Set(Object.keys(MARKERS[t])));

export default async function render(main) {
  const PER = 30, OV0 = 0.7, S0 = 1;
  const cells = makeCells(S0, PER);
  const model0 = pretrain(makeCorpus(OV0));
  cells.forEach((c, i) => { c.i = i; c.enc = encode(c.counts, model0.med); });
  const b0 = cells.filter((c) => c.b === 0), b1 = cells.filter((c) => c.b === 1);
  const cellAt = (t, k) => cells[t * PER + k];
  const nTok = (c, ord, k) => ord.slice(0, k).filter((j) => gclass(GENES[j]) === 'hk').length;

  header(main, 'Single-cell foundation models',
    'Models pretrained on millions of cells without labels, by writing each cell as an <b>ordered list of genes</b> and learning to predict hidden entries. The result is an embedding (a vector per cell) that can be reused for annotation, integration and prediction. Whether that embedding beats a plain PCA is an empirical question, and independent benchmarks have found that often it does not.');
  lessonNav(main);
  sourceCard(main, {
    org: 'Geneformer: Theodoris lab (Gladstone / Broad). Cell2Sentence / C2S-Scale: Google Research / DeepMind and Yale. Zero-shot evaluation: Kedzierska et al. (independent). Annotation by LLM: GPTCelltype.',
    when: 'Geneformer: Nature 2023, V2 trained on about 104M cells (Dec 2024). C2S-Scale: bioRxiv 2025. Kedzierska et al.: Genome Biology 2025. GPTCelltype: Nature Methods 2024.',
    links: ['geneformer', 'kedzierska', 'c2s', 'gptcelltype', 'scgpt'],
    claim: 'Authors: pretraining gives context-aware predictions with limited data (Geneformer); marker-gene annotation by a language model reaches more than 75% concordance with manual labels in most tissues (GPTCelltype). Independent test: used zero-shot, Geneformer and scGPT embeddings were worse than HVG + PCA, scVI and Harmony (Kedzierska et al.).',
  });
  callout(main, 'warn', 'Teaching toys, not reproductions',
    `Every figure on this page uses ${cells.length} simulated cells from ${TYPES.length} tumour-microenvironment types (${TYPES.join(', ')}) over ${G} genes, in two simulated sequencing batches. A co-occurrence counter stands in for the transformer and a keyword lookup stands in for the language model. The figures show the mechanism and the evaluation logic. They do not reproduce any published model or number.`);
  prose(main, '<p><b>Question for this page:</b> what does a single-cell foundation model learn from unlabelled cells, and does its zero-shot embedding separate cell types better than the standard highly-variable-gene (HVG) + PCA pipeline?</p>');

  // =====================================================================================
  // 1. WHAT
  // =====================================================================================
  let ch = chapter(main, 'what', 'What is a single-cell foundation model?', 'In one line: <b>cells written as ordered gene lists → self-supervised pretraining → a reusable embedding</b>.');
  prose(ch, `<p>A <b>foundation model</b> is trained once on a very large unlabelled corpus with a self-supervised task, then reused for many downstream tasks. In single-cell biology the corpus is millions of expression profiles and the self-supervised task is to <b>predict genes that have been hidden</b> from a cell. No cell-type labels are used in pretraining.</p>
    <p>A transformer reads sequences of discrete tokens, but a cell is a vector of counts with no natural order. The models differ mainly in how a cell is turned into a sequence. Three variants appear in this page:</p>
    <ul><li><b>Geneformer</b>: genes are ordered by <i>rank value</i> (expression relative to the gene's typical level in the corpus). Each gene is a token. A BERT-style model learns to predict masked tokens.</li>
    <li><b>Cell2Sentence (C2S)</b>: genes are ordered by expression and written as <i>plain text</i> (gene names separated by spaces), so that an ordinary language model can read and write cells, together with text such as the tissue or the cell type.</li>
    <li><b>LLM annotation</b>: a general language model receives a list of marker genes and returns a cell-type name. No single-cell training is involved.</li></ul>
    <p>The diagram follows one simulated cell through the Geneformer-style pipeline.</p>`);
  ch.append(pipelineSvg(cellAt(4, 4), model0));
  prose(ch, '<p>Key terms:</p>');
  table(ch, [
    { t: 'Token', m: 'One discrete unit read by the model', s: 'a gene (Geneformer) or a word of text (C2S)' },
    { t: 'Rank-value encoding', m: 'Sort genes by expression divided by the gene\'s corpus-wide non-zero median; keep the order', s: `${G} genes → a sentence of the top ${L} tokens` },
    { t: 'Masked-gene pretraining', m: 'Hide some tokens and train the model to predict them from the rest', s: 'chapter 4, step 2' },
    { t: 'Embedding', m: 'A short vector per cell taken from the model; similar cells have similar vectors', s: '8 numbers per cell in the toy' },
    { t: 'Zero-shot', m: 'Use the pretrained embedding as it is, with no training on the new data', s: 'chapter 4, step 5' },
    { t: 'Fine-tuning', m: 'Continue training with a small labelled set for one task', s: 'a labelled-cells slider in step 5' },
    { t: 'In-context learning', m: 'Condition the model on example cells in its input, without changing its weights', s: 'chapter 4, step 6' },
  ], [['t', 'Term'], ['m', 'Meaning'], ['s', 'In the toy']]);

  // =====================================================================================
  // 2. WHY
  // =====================================================================================
  ch = chapter(main, 'why', 'Why do we need it?', 'What the standard pipeline leaves unsolved, and what a pretrained model is meant to supply.');
  prose(ch, `<p>The standard single-cell pipeline (normalise, select HVGs, PCA, cluster) is fitted separately to each dataset. Three problems recur:</p><ol>
    <li><b>Technical variation.</b> Batches, chemistries and sequencing depth shift expression, and cells can group by batch instead of by type.</li>
    <li><b>Few labels.</b> Annotating clusters by hand is slow, and each study repeats the work.</li>
    <li><b>No transfer.</b> Nothing learned from one atlas carries to the next study.</li></ol>
    <p>A foundation model is meant to address all three: an encoder that is robust to technical variation, and that has seen enough cells to transfer without new training. The figure shows the first problem on the toy data, using the standard pipeline only.</p>`);
  const fw = figure(ch, 'The baseline: HVG + PCA on two batches',
    'Each point is a simulated cell, projected on the first two PCs of the 30 most variable genes. Circles: batch 1; triangles: batch 2 (a different chemistry with higher depth and shifted genes). Increase the batch effect and compare the two colourings.');
  let wMode = 'type', wS = S0, wCells = cells;
  select(fw.controls, { label: 'Colour by', value: wMode, options: [['type', 'Cell type'], ['batch', 'Batch']], onChange: (v) => { wMode = v; paintW(); } });
  slider(fw.controls, { label: 'Batch effect', min: 0, max: 2, step: 0.25, value: wS, fmt: f2, onInput: (v) => { wS = v; wCells = v === S0 ? cells : makeCells(v, PER); paintW(); } });
  const sW = stats(fw.root);
  const leg2 = el('div'); fw.root.append(leg2);
  const FW = frame(fw.body, { width: 820, height: 350, x: [-1, 1], y: [-1, 1], xLabel: 'PC 1', yLabel: 'PC 2', margin: { top: 26 } });
  FW.layer('note').append('text').attr('x', FW.w).attr('y', -10).attr('text-anchor', 'end').attr('font-size', 12).attr('fill', muted()).text('standard pipeline, no batch correction');
  function paintW() {
    const E = hvgPca(wCells), m = knnMetrics(E, wCells), P = pca(E, 2, 40).scores;
    scatterTo(FW, P, wCells, wMode);
    leg2.innerHTML = ''; legend(leg2, wMode === 'type' ? TYPES.map((t, i) => [t, cat(i)]) : [['batch 1', cat(6)], ['batch 2', cat(7)]]);
    sW.set({ 'neighbours with the right type': pc(m.acc), 'batch mixing (1 = fully mixed)': f2(m.mix) });
  }
  paintW();
  const mW = knnMetrics(hvgPca(cells), cells), mW0 = knnMetrics(hvgPca(makeCells(0, PER)), makeCells(0, PER)), mW2 = knnMetrics(hvgPca(makeCells(2, PER)), makeCells(2, PER));
  prose(ch, `<p>At the default batch effect (${f2(S0)}) the nearest-neighbour type accuracy of the baseline is ${pc(mW.acc)} and batch mixing is ${f2(mW.mix)}. With no batch effect they are ${pc(mW0.acc)} and ${f2(mW0.mix)}; at a batch effect of 2 they are ${pc(mW2.acc)} and ${f2(mW2.mix)}, so strong batch effects mainly cost mixing, and type accuracy more slowly. This is the situation in which an encoder trained on a large, varied corpus is expected to help. The comparison in step 5 tests that expectation.</p>`);
  table(ch, [
    { a: 'Marker lookup by hand', l: 'none', b: 'no', c: 'slow; depends on the annotator' },
    { a: 'HVG + PCA (+ clustering)', l: 'none', b: 'only with a separate correction step', c: 'fitted per dataset; the standard baseline' },
    { a: 'scVI, Harmony and similar integration', l: 'none to few', b: 'yes, by design', c: 'fitted per project; strong baselines in published benchmarks' },
    { a: '<b>Foundation-model embedding</b>', l: 'none (zero-shot) or few (fine-tuned)', b: 'depends on pretraining and on the new data', c: 'large pretraining cost; the embedding is reused' },
    { a: 'LLM on a marker list', l: 'none', b: 'not applicable (works on gene names)', c: 'fast; always returns an answer' },
  ], [['a', 'Approach'], ['l', 'Labels needed'], ['b', 'Handles batch effects?'], ['c', 'Notes']]);

  // =====================================================================================
  // 3. WHERE
  // =====================================================================================
  ch = chapter(main, 'where', 'Where is it used?', 'Model families and the tasks they are applied to. Each entry gives the authors\' framing; independent evaluations are cited where they exist.');
  cards(ch, [
    { tag: 'Geneformer', title: 'Rank-value tokens, masked-gene pretraining', body: `Pretrained on about 104M cells (V2). Used for cell embeddings, fine-tuned classifiers and in-silico deletion of a gene token to nominate candidate genes. Model card: ${cite('geneformer', 'Hugging Face')}.` },
    { tag: 'scGPT, scFoundation, UCE', title: 'Other pretrained encoders', body: `${cite('scgpt', 'scGPT')} pairs a gene token with a binned expression value (33M cells). ${cite('scfoundation', 'scFoundation')} and ${cite('uce', 'UCE')} follow the same pattern of large-scale pretraining and a reusable cell embedding.` },
    { tag: 'Cell2Sentence / C2S-Scale', title: 'Cells as text for a language model', body: `Gene names ordered by expression feed a Gemma-2 language model (2B and 27B). The text interface allows prompts that mix cells and natural language. The headline drug-candidate result rests on one in-vitro validation in a preprint (${cite('c2s', 'bioRxiv 2025')}).` },
    { tag: 'LLM annotation', title: 'Cell type from a marker list', body: `${cite('gptcelltype', 'GPTCelltype')} reports more than 75% concordance with manual annotation in most tissues. Agent systems such as ${cite('cellagent', 'CellAgent')} wrap the same idea in a tool-using workflow.` },
    { tag: 'In-context models', title: 'Prompt with example cells', body: `${cite('stackNews', 'Stack')} (Arc Institute) attends across genes and across cells, and takes a prompt set of cells as context. Gains are self-reported.` },
    { tag: 'Bulk RNA', title: 'Masked expression on bulk samples', body: `${cite('bulkformer', 'BulkFormer')} applies masked-expression pretraining to bulk profiles, such as the cohort used elsewhere on this site.` },
    { tag: 'Spatial', title: 'Niches and domains', body: `${cite('nicheformer', 'Nicheformer')} and ${cite('novae', 'Novae')} extend the idea to spatial transcriptomics, with zero-shot spatial domains as a stated aim.` },
    { tag: 'Benchmarks', title: 'Independent evaluations', body: `${cite('kedzierska', 'Kedzierska et al. 2025')} tested Geneformer and scGPT zero-shot against simple baselines. For perturbation prediction, ${cite('ahlmann', 'Ahlmann-Eltze et al. 2025')} reported that additive and linear baselines matched or beat the foundation models tested. The first is the subject of chapter 4, step 5; the second is covered on the virtual-cells page.` },
  ]);
  prose(ch, '<p><b>When not to use it:</b> a small, well-annotated dataset from a single batch (HVG + PCA and clustering are quicker and easier to audit); a tissue or species that the pretraining corpus barely covers; panels with few genes (spatial panels give very short input lists); or any analysis where the embedding is the claim and no baseline comparison has been run on the same cells.</p>');

  // =====================================================================================
  // 4. HOW
  // =====================================================================================
  ch = chapter(main, 'how', 'How it works', 'Six steps: encoding, pretraining, text, annotation, the zero-shot test and in-context use.');

  // ---- Step 1: rank-value encoding
  let rb = rung(ch, 1, 'Rank-value encoding: counts to an ordered sentence',
    `<p>Each cell is first scaled to 10,000 counts in total. Each gene's value is then divided by that gene's <b>non-zero median</b> across the pretraining corpus, which gives the <b>rank value</b>. Genes are sorted from highest to lowest rank value, and the first ${L} (in the toy) form the sentence. Sorting by raw count would place housekeeping genes first in every cell. Dividing by the gene's typical level pushes them down and brings cell-specific genes forward.</p>`);
  const fig1 = figure(rb, 'One cell, two orderings',
    'Left: genes sorted by raw count. Right: genes sorted by rank value. Bars show the quantity being sorted, scaled to the top gene. Hover a bar for the numbers.');
  toyBadge(fig1, `${cells.length} simulated cells, ${G} genes; medians from a simulated pretraining corpus.`);
  let cur = cellAt(4, 4);
  const tSel = select(fig1.controls, { label: 'Cell type', value: '4', options: TYPES.map((t, i) => [String(i), t]), onChange: (v) => { cur = cellAt(+v, cSl.get()); drawF1(); } });
  const cSl = slider(fig1.controls, { label: 'Cell #', min: 0, max: PER - 1, step: 1, value: 4, onInput: (v) => { cur = cellAt(cur.t, v); drawF1(); } });
  const s1 = stats(fig1.root);
  const r1 = el('div', { class: 'fig-row' }); fig1.body.append(r1);
  const l1 = el('div'), l2 = el('div'); r1.append(l1, l2);
  legend(fig1.body, [['housekeeping gene', cat(6)], ['marker of this cell\'s type', cat(0)], ['other gene', muted()]]);
  const colOf = (j) => (gclass(GENES[j]) === 'hk' ? cat(6) : typeMarkers[cur.t].has(GENES[j]) ? cat(0) : muted());
  function rankList(parent, order, val, fmtV, title) {
    parent.innerHTML = '';
    const N = 12, rowH = 21, W = 420, H = 34 + N * rowH;
    const svg = d3.select(parent).append('svg').attr('viewBox', `0 0 ${W} ${H}`).attr('width', W).attr('role', 'img');
    svg.append('text').attr('x', 0).attr('y', 16).attr('fill', ink()).attr('font-weight', 600).attr('font-size', 13).text(title);
    const top = order.slice(0, N), vmax = val(top[0]) || 1;
    top.forEach((j, i) => {
      const y = 28 + i * rowH;
      svg.append('text').attr('x', 18).attr('y', y + 14).attr('text-anchor', 'end').attr('font-size', 11).attr('fill', muted()).text(i + 1);
      svg.append('text').attr('x', 26).attr('y', y + 14).attr('font-size', 12).attr('fill', ink()).attr('font-family', css('--mono')).text(GENES[j]);
      svg.append('rect').attr('x', 100).attr('y', y + 3).attr('height', rowH - 7).attr('rx', 3).attr('width', Math.max(2, (W - 170) * val(j) / vmax)).attr('fill', colOf(j))
        .on('mousemove', (e) => tip(e, `<b>${GENES[j]}</b><br>raw count ${cur.counts[j]}<br>per 10k: ${f1(cur.enc.norm[j])}<br>corpus non-zero median: ${f1(model0.med[j])}<br>rank value: ${f2(cur.enc.rv[j])}`)).on('mouseleave', untip);
      svg.append('text').attr('x', W - 4).attr('y', y + 14).attr('text-anchor', 'end').attr('font-size', 11).attr('fill', ink2()).text(fmtV(j));
    });
  }
  function drawF1() {
    rankList(l1, cur.enc.raw, (j) => cur.counts[j], (j) => cur.counts[j], 'Sorted by raw count');
    rankList(l2, cur.enc.order, (j) => cur.enc.rv[j], (j) => f2(cur.enc.rv[j]), 'Sorted by rank value');
    const own = (o) => o.slice(0, 10).filter((j) => typeMarkers[cur.t].has(GENES[j])).length;
    s1.set({ 'housekeeping in top 10, raw': nTok(cur, cur.enc.raw, 10), 'housekeeping in top 10, rank value': nTok(cur, cur.enc.order, 10), 'own markers in top 10, raw': own(cur.enc.raw), 'own markers in top 10, rank value': own(cur.enc.order) });
  }
  drawF1();
  const hkRaw = d3.mean(cells, (c) => nTok(c, c.enc.raw, 10)), hkRv = d3.mean(cells, (c) => nTok(c, c.enc.order, 10));
  const mkRaw = d3.mean(cells, (c) => c.enc.raw.slice(0, 10).filter((j) => typeMarkers[c.t].has(GENES[j])).length), mkRv = d3.mean(cells, (c) => c.enc.order.slice(0, 10).filter((j) => typeMarkers[c.t].has(GENES[j])).length);
  prose(rb, `<p>Across all ${cells.length} cells, ${f1(hkRaw)} of the top 10 genes by raw count are housekeeping genes, against ${f1(hkRv)} by rank value; cell-specific markers rise from ${f1(mkRaw)} to ${f1(mkRv)} of the top 10. Only the <b>order</b> is kept. A second consequence is that multiplying every count in a cell by a constant (a deeper library) leaves the sentence unchanged, which gives some robustness to sequencing depth. Absolute magnitudes are discarded.</p>`);

  // ---- Step 2: masked-gene pretraining
  const alpha = 0.5;
  const predict = (ctx) => {
    const inCtx = new Set(ctx);
    return d3.range(G).filter((g) => !inCtx.has(g)).map((g) => {
      let s = Math.log((model0.freq[g] + alpha) / (model0.n + alpha * G));
      ctx.forEach((c) => { s += Math.log((model0.co[g][c] + alpha) / (model0.freq[g] + alpha * G)); });
      return { g, s };
    }).sort((a, b) => b.s - a.s);
  };
  const freqOrder = d3.range(G).sort((a, b) => model0.freq[b] - model0.freq[a]);
  let hit = 0, base = 0, tot = 0;
  b0.forEach((c) => { const s = c.enc.order.slice(0, L); s.forEach((m, k) => { const ctx = s.filter((_, i) => i !== k); tot++; if (predict(ctx)[0].g === m) hit++; if (freqOrder.find((g) => !ctx.includes(g)) === m) base++; }); });
  rb = rung(ch, 2, 'Masked-gene pretraining',
    `<p>Pretraining hides some tokens and trains the model to recover them from the visible ones. Recovering a hidden gene requires knowing which genes tend to appear together: <i>CD79A</i> and <i>MS4A1</i>, or <i>COL1A1</i> and <i>DCN</i>. In Geneformer a transformer learns this and, as a by-product, an internal vector for every gene and for every cell. In the toy, a <b>co-occurrence counter</b> replaces the transformer: it scores every unseen gene by its overall frequency and by how often it appeared in the same sentence as each visible gene (a naive Bayes rule).</p>`);
  const f2f = figure(rb, 'Mask one token and predict it',
    `The first ${L} tokens of the selected cell's sentence. The masked position is predicted from the other ${L - 1} tokens. Bars: relative score of the six best candidates (softmax). Click a token to mask it.`);
  toyBadge(f2f, 'A co-occurrence counter stands in for the transformer.');
  const mSl = slider(f2f.controls, { label: 'Masked position', min: 1, max: L, step: 1, value: 3, onInput: () => drawF2() });
  const rM = rng(5);
  button(f2f.controls, 'Mask a random position', () => { mSl.set(1 + rM.int(L)); drawF2(); });
  const chips = el('div', { style: 'display:flex;flex-wrap:wrap;gap:6px;margin:6px 0 10px' }); f2f.body.append(chips);
  const predBox = el('div'); f2f.body.append(predBox);
  const s2 = stats(f2f.root);
  function drawF2() {
    const s = cur.enc.order.slice(0, L), k = mSl.get() - 1;
    chips.innerHTML = '';
    s.forEach((j, i) => {
      const c = el('button', { class: 'btn', style: `font-family:var(--mono);font-size:12.5px;${i === k ? 'border-color:var(--accent);background:var(--accent-soft)' : ''}`, title: 'mask this token' }, i === k ? '[MASK]' : GENES[j]);
      c.onclick = () => { mSl.set(i + 1); drawF2(); };
      chips.append(c);
    });
    const ctx = s.filter((_, i) => i !== k), all = predict(ctx), preds = all.slice(0, 6), truth = s[k];
    const mx = preds[0].s, p = preds.map((d) => Math.exp(d.s - mx)), Z = sum(all.map((d) => Math.exp(d.s - mx)));
    predBox.innerHTML = '';
    const F = frame(predBox, { width: 560, height: 210, x: [0, 1], y: [0, 6], yScale: d3.scaleBand().padding(0.2), margin: { left: 90 }, xLabel: 'probability assigned by the toy model', grid: false, axes: false });
    F.y.domain(preds.map((d) => GENES[d.g])).range([0, F.h]);
    F.g.append('g').attr('class', 'axis y').call(d3.axisLeft(F.y));
    F.g.append('g').attr('class', 'axis x').attr('transform', `translate(0,${F.h})`).call(d3.axisBottom(F.x).ticks(5));
    F.layer('b').selectAll('rect').data(preds).join('rect').attr('x', 0).attr('y', (d) => F.y(GENES[d.g])).attr('height', F.y.bandwidth()).attr('rx', 3)
      .attr('width', (d, i) => F.x(p[i] / Z)).attr('fill', (d) => (d.g === truth ? cat(1) : muted()))
      .on('mousemove', (e, d) => tip(e, `<b>${GENES[d.g]}</b>${d.g === truth ? ' (the hidden gene)' : ''}<br>log score ${f2(d.s)}`)).on('mouseleave', untip);
    const rank = all.findIndex((d) => d.g === truth) + 1;
    F.layer('n').append('text').attr('x', F.w - 4).attr('y', F.h - 8).attr('text-anchor', 'end').attr('font-size', 12).attr('fill', ink2()).text(`hidden gene ${GENES[truth]}: rank ${rank} of ${all.length} (highlighted if in the top 6)`);
    s2.set({ 'hidden gene': GENES[truth], 'its rank among candidates': rank, 'toy top-1 accuracy, all masks': pc(hit / tot), 'always guess the most frequent gene': pc(base / tot) });
  }
  legend(f2f.body, [['hidden (true) gene', cat(1)], ['other candidates', muted()]]);
  tSel.addEventListener('change', drawF2); cSl.input.addEventListener('input', drawF2);
  drawF2();
  prose(rb, `<p>Each of the ${tot} masked tokens in the batch-1 cells was predicted by the toy model (top-1 accuracy ${pc(hit / tot)}) and by a baseline that always names the most frequent unseen gene (${pc(base / tot)}). Masked-gene prediction has no labels, yet doing it well requires knowledge of cell-type structure. Cell-type markers are easiest to recover, because the other genes in the sentence identify the type. Taking the model's vector for each gene, and averaging the vectors of the sentence (earlier tokens weighted more), gives the <b>cell embedding</b> used in step 5. Geneformer also supports <i>in-silico deletion</i>: remove a token, re-embed, and measure how far the cell moves.</p>`);
  quiz(rb, 'Why is a masked cell-type marker easier to recover than a gene with a similar level in every cell type?',
    `<p>A marker co-occurs with other genes of the same type (<i>NKG7</i> with <i>CCL5</i> and <i>CD8A</i>), so the visible tokens narrow the candidates. A gene that is expressed similarly everywhere is not tied to any particular context, so the visible tokens say little about it. What a masked-gene model learns is limited to what the context makes predictable.</p>`);

  // ---- Step 3: Cell2Sentence
  const KS = d3.range(2, 41, 2);
  const tie = (g, i) => ((g * 7919 + i * 104729) % 1009);
  cells.forEach((c) => { c.sent = c.counts.map((v, j) => j).filter((j) => c.counts[j] > 0).sort((a, b) => c.counts[b] - c.counts[a] || tie(a, c.i) - tie(b, c.i)); c.lg = c.enc.norm.map((v) => Math.log1p(v)); });
  const curve = d3.range(40).map((r) => d3.mean(b0, (c) => (c.sent[r] != null ? c.lg[c.sent[r]] : 0)));
  const bow = (c, K) => { const v = new Array(G).fill(0); c.sent.slice(0, K).forEach((j) => { v[j] = 1; }); return v; };
  const recon = (c, K) => { const v = new Array(G).fill(0); c.sent.slice(0, K).forEach((j, r) => { v[j] = curve[r]; }); return v; };
  const pear = (a, b) => { const ma = d3.mean(a), mb = d3.mean(b); let s = 0, sa = 0, sb = 0; a.forEach((x, i) => { s += (x - ma) * (b[i] - mb); sa += (x - ma) ** 2; sb += (b[i] - mb) ** 2; }); return sa && sb ? s / Math.sqrt(sa * sb) : 0; };
  const kRes = KS.map((K) => ({ K, acc: centroidAcc(b0.map((c) => bow(c, K)), b0.map((c) => c.t), b1.map((c) => bow(c, K)), b1.map((c) => c.t)), r: d3.mean(b1, (c) => pear(recon(c, K), c.lg)) }));
  const kBest = d3.max(kRes, (d) => d.acc), k95 = kRes.find((d) => d.acc >= 0.95 * kBest);
  rb = rung(ch, 3, 'Cell2Sentence: the same idea as plain text',
    `<p>C2S skips the special gene tokens. A cell becomes <b>a sentence of gene names, most expressed first</b>, and a standard language model is trained on it, with natural-language text such as tissue or cell type in the same prompt. The sentence has to be <b>truncated</b> (a real cell expresses thousands of genes), so the choice of length matters. The figure measures what a sentence of length <i>K</i> still contains.</p>`);
  const f3 = figure(rb, 'What does a sentence of K genes keep?',
    'Top: the selected cell as text, with its own marker genes highlighted. Bottom: for every K, the accuracy of a type classifier that sees only which genes appear in the sentence (centroids from batch 1, tested on batch 2) and the correlation between expression rebuilt from the sentence and the true expression.');
  toyBadge(f3, 'Sentence = genes by raw count, ties broken at random.');
  const kSl = slider(f3.controls, { label: 'Sentence length K', min: 2, max: 40, step: 2, value: 12, onInput: () => drawF3() });
  const sent = el('div', { class: 'calc', style: 'margin:4px 0 8px' }); f3.body.append(sent);
  const s3 = stats(f3.root);
  const F3 = frame(f3.body, { width: 820, height: 270, x: [0, 41], y: [0, 1.02], xLabel: 'sentence length K (genes kept)', yLabel: 'score' });
  const kLn = F3.layer('k');
  [['acc', cat(0)], ['r', cat(2)]].forEach(([k, col]) => {
    F3.layer('c').append('path').datum(kRes).attr('fill', 'none').attr('stroke', col).attr('stroke-width', 2.2).attr('d', d3.line().x((d) => F3.x(d.K)).y((d) => F3.y(d[k])));
    F3.layer('c').selectAll('circle.' + k).data(kRes).join('circle').attr('class', k).attr('cx', (d) => F3.x(d.K)).attr('cy', (d) => F3.y(d[k])).attr('r', 3.6).attr('fill', col).attr('stroke', css('--surface')).attr('stroke-width', 1)
      .on('mousemove', (e, d) => tip(e, `K = ${d.K}<br>${k === 'acc' ? 'type accuracy' : 'reconstruction r'} ${f2(d[k])}`)).on('mouseleave', untip);
  });
  F3.layer('t').append('text').attr('x', F3.x(k95.K) + 8).attr('y', F3.y(0.3)).attr('font-size', 12).attr('fill', ink2()).text(`≥ 95% of the best type accuracy from K = ${k95.K}`);
  legend(f3.body, [['type accuracy from gene membership', cat(0)], ['expression rebuilt from the order', cat(2)], ['current K', ink()]]);
  function drawF3() {
    const K = kSl.get(), r = pear(recon(cur, K), cur.lg), kr = kRes.find((d) => d.K === K);
    sent.innerHTML = '';
    cur.sent.slice(0, K).forEach((j, i) => sent.append(el('span', { style: `margin-right:6px;${typeMarkers[cur.t].has(GENES[j]) ? 'background:var(--accent-soft);border-radius:4px;padding:0 3px' : ''}` }, GENES[j] + (i < K - 1 ? '' : ''))));
    kLn.selectAll('*').remove();
    kLn.append('line').attr('x1', F3.x(K)).attr('x2', F3.x(K)).attr('y1', 0).attr('y2', F3.h).attr('stroke', ink()).attr('stroke-width', 1.5).attr('stroke-dasharray', '4 3');
    s3.set({ 'genes in the sentence': `${K} of ${cur.sent.length} expressed`, 'share of this cell\'s UMIs': pc(sum(cur.sent.slice(0, K).map((j) => cur.counts[j])) / sum(cur.counts)), 'type accuracy at K (all cells)': pc(kr.acc), 'reconstruction r, this cell': f2(r) });
  }
  tSel.addEventListener('change', drawF3); cSl.input.addEventListener('input', drawF3);
  drawF3();
  const kPeak = kRes.reduce((a, b) => (b.acc > a.acc ? b : a)), acc10 = kRes.find((d) => d.K === 10).acc, last = kRes[kRes.length - 1];
  prose(rb, `<p>Housekeeping genes fill the top of a raw-count ordering, so short sentences carry almost no information about type (accuracy ${pc(acc10)} at K = 10). Accuracy rises sharply once the first markers enter, reaches ${pc(kBest)} at K = ${kPeak.K}, and then ${last.acc < kBest - 0.02 ? `declines to ${pc(last.acc)} at K = ${last.K}` : 'stays flat'} because longer sentences add genes that do not distinguish types. This is the problem that Geneformer's division by gene medians addresses. Reconstruction of the whole profile keeps improving with K but stays below 1: order carries the shape of the profile, not the magnitudes. Real cells have thousands of expressed genes, so the cut-off removes much more. The text format also costs more tokens per cell than a gene-token model. In return, a single language model can read cells and natural-language prompts together, which is the approach taken by C2S-Scale (results self-reported).</p>`);

  // ---- Step 4: LLM annotation
  const lgOf = (c) => c.enc.norm.map((v) => Math.log1p(v));
  const topDE = (group, rest) => {
    const gm = GENES.map((_, j) => d3.mean(group, (c) => lgOf(c)[j])), rm = GENES.map((_, j) => d3.mean(rest, (c) => lgOf(c)[j]));
    return d3.range(G).filter((j) => gm[j] > 0.3).sort((a, b) => (gm[b] - rm[b]) - (gm[a] - rm[a])).slice(0, 10).map((j) => GENES[j]);
  };
  const rA = rng(7), mk = (counts) => ({ counts, enc: encode(counts, model0.med) });
  const dbl = d3.range(24).map((i) => mk(cellAt(1, i).counts.map((v, j) => v + cellAt(3, i).counts[j])));
  const stress = d3.range(24).map((i) => { const c = cellAt(i % TYPES.length, 10 + (i >> 1)); return mk(c.counts.map((v, j) => (['JUN', 'FOS', 'HSPA1A', 'MT-CO1'].includes(GENES[j]) ? v * 8 + 5 : Math.round(v * 0.5)))); });
  const KBN = { 'Tumour': 'Epithelial cell', 'CD8 T': 'CD8 T cell', 'CD4 T': 'CD4 T cell', 'B cell': 'B cell', 'Macrophage': 'Macrophage', 'CAF': 'Fibroblast' };
  const CL = [...TYPES.map((t, i) => ({ name: 'C' + i, truth: KBN[t], kind: 'clean', genes: topDE(b0.filter((c) => c.t === i), b0.filter((c) => c.t !== i)) })),
    { name: 'C6', truth: 'CD8 T + B cell doublets', kind: 'doublet', genes: topDE(dbl, b0) }, { name: 'C7', truth: 'stressed / low-quality cells', kind: 'lowq', genes: topDE(stress, b0) }];
  const KB = {
    'Epithelial cell': [['EPCAM', 'KRT8', 'KRT18', 'KRT19', 'KRT5'], 1], 'CD8 T cell': [['CD8A', 'CD8B', 'GZMK', 'CCL5'], 0.6], 'CD4 T cell': [['CD4', 'IL7R', 'LTB', 'CCR7', 'TCF7'], 0.6],
    'T cell': [['CD3E', 'CD3D', 'IL7R', 'LTB'], 1], 'B cell': [['MS4A1', 'CD79A', 'CD79B', 'BANK1'], 0.9], 'Macrophage': [['CD68', 'C1QA', 'C1QB', 'APOE', 'CD14'], 0.8],
    'Monocyte': [['LYZ', 'CD14', 'FCN1', 'S100A8'], 0.8], 'Fibroblast': [['COL1A1', 'COL1A2', 'DCN', 'LUM', 'FAP', 'PDGFRA'], 0.8], 'NK cell': [['NKG7', 'GNLY', 'KLRD1', 'PRF1'], 0.6],
    'Erythrocyte': [['HBB', 'HBA1', 'HBA2'], 0.4], 'Plasma cell': [['JCHAIN', 'MZB1', 'IGHG1'], 0.5],
  };
  const AMB = ['HBB', 'HBA1', 'LYZ'];
  let ci = 0, nM = 8, nAmb = 0, abst = false;
  function annotate(cl, ab = abst) {
    const list = cl.genes.slice();
    AMB.slice(0, nAmb).forEach((g, i) => list.splice(1 + 2 * i, 0, g));
    const use = list.slice(0, nM);
    const raw = Object.entries(KB).map(([lab, [m, prior]]) => { let s = 0; const hits = []; use.forEach((g, i) => { if (m.includes(g)) { s += 1 / (1 + 0.25 * i); hits.push(g); } }); return { lab, hits, score: s + 0.35 * prior }; });
    const mx = d3.max(raw, (d) => d.score), Z = sum(raw.map((d) => Math.exp(4 * (d.score - mx))));
    raw.forEach((d) => { d.p = Math.exp(4 * (d.score - mx)) / Z; });
    raw.sort((a, b) => b.p - a.p);
    const say = ab && raw[0].p < 0.5 ? 'unknown' : raw[0].lab;
    const ok = cl.kind === 'clean' ? (say === cl.truth ? 'yes' : (say === 'T cell' && cl.truth.endsWith('T cell') ? 'coarse' : say === 'unknown' ? 'missed' : 'no')) : (say === 'unknown' ? 'yes' : 'no');
    return { use, raw, say, ok };
  }
  rb = rung(ch, 4, 'LLM annotation: a label from a marker list',
    `<p>A language model can also be used with no single-cell training. For each cluster, the top differentially expressed genes are written into a prompt ("which cell type expresses ${CL[1].genes.slice(0, 4).join(', ')}?") and the model answers from what it has read about these genes. The model never sees the expression matrix. In the toy, a keyword table of textbook markers stands in for the language model: it scores each label by marker overlap (earlier genes count more), adds a small prior for common labels and converts the scores to probabilities.</p>`);
  const f4 = figure(rb, 'Marker list → label, and where it fails',
    'Left: the marker list sent to the toy "LLM" (✓ marks genes that are in its knowledge base; ⚠ marks injected ambient-RNA genes). Right: probability over labels. Table: every cluster at the current settings. For doublets and stressed cells, "unknown" is the correct answer.');
  toyBadge(f4, 'A keyword lookup imitates a language model; no model is called.');
  select(f4.controls, { label: 'Cluster', value: '0', options: CL.map((c, i) => [String(i), `${c.name} · truth: ${c.truth}`]), onChange: (v) => { ci = +v; drawF4(); } });
  slider(f4.controls, { label: 'Markers sent', min: 2, max: 10, step: 1, value: nM, onInput: (v) => { nM = v; drawF4(); } });
  slider(f4.controls, { label: 'Ambient genes added', min: 0, max: 3, step: 1, value: nAmb, onInput: (v) => { nAmb = v; drawF4(); } });
  checkbox(f4.controls, { label: 'abstain below 50% confidence', checked: abst, onChange: (c) => { abst = c; drawF4(); } });
  const s4 = stats(f4.root);
  const row4 = el('div', { class: 'fig-row' }); f4.body.append(row4);
  const gl = el('div'), gb = el('div'); row4.append(gl, gb);
  const tbl4 = el('div'); f4.body.append(tbl4);
  function drawF4() {
    const cl = CL[ci], r = annotate(cl);
    gl.innerHTML = '';
    const inKB = new Set(Object.values(KB).flatMap(([m]) => m));
    gl.append(html(`<div class="calc">${r.use.map((g, i) => `${String(i + 1).padStart(2)}  ${g}${AMB.includes(g) && !cl.genes.includes(g) ? '  ⚠ ambient' : ''}${inKB.has(g) ? '  ✓' : ''}`).join('\n')}</div>`));
    gb.innerHTML = '';
    const top = r.raw.slice(0, 6);
    const F = frame(gb, { width: 440, height: 250, x: [0, 1], y: [0, 6], yScale: d3.scaleBand().padding(0.2), margin: { left: 110 }, xLabel: 'probability', grid: false, axes: false });
    F.y.domain(top.map((d) => d.lab)).range([0, F.h]);
    F.g.append('g').attr('class', 'axis y').call(d3.axisLeft(F.y));
    F.g.append('g').attr('class', 'axis x').attr('transform', `translate(0,${F.h})`).call(d3.axisBottom(F.x).ticks(5));
    F.layer('b').selectAll('rect').data(top).join('rect').attr('y', (d) => F.y(d.lab)).attr('height', F.y.bandwidth()).attr('rx', 3).attr('width', (d) => F.x(d.p))
      .attr('fill', (d) => (cl.kind === 'clean' && d.lab === cl.truth ? cat(1) : muted()))
      .on('mousemove', (e, d) => tip(e, `<b>${d.lab}</b><br>p = ${f2(d.p)}<br>matched: ${d.hits.join(', ') || 'none'}`)).on('mouseleave', untip);
    F.layer('n').append('text').attr('x', F.w - 4).attr('y', F.h - 6).attr('text-anchor', 'end').attr('font-size', 12).attr('fill', ink2()).text(r.say === 'unknown' ? 'answer: unknown (abstained)' : `answer: ${r.say}`);
    const all = CL.map((c) => ({ c, r: annotate(c) })), clean = all.filter((d) => d.c.kind === 'clean');
    s4.set({ 'answer for this cluster': r.say, 'top-label probability': f2(r.raw[0].p), 'clean clusters correct': `${clean.filter((d) => d.r.ok === 'yes').length}/${clean.length}`, 'doublet / stressed handled': `${all.filter((d) => d.c.kind !== 'clean' && d.r.ok === 'yes').length}/2` });
    tbl4.innerHTML = '';
    table(tbl4, all.map(({ c, r: q }) => ({ n: c.name, t: c.truth, a: q.say, p: f2(q.raw[0].p), v: `<span class="pill ${q.ok === 'yes' ? 'yes' : 'no'}">${q.ok === 'yes' ? 'correct' : q.ok === 'coarse' ? 'coarse only' : q.ok === 'missed' ? 'abstained, should answer' : 'wrong'}</span>` })),
      [['n', 'Cluster'], ['t', 'Truth'], ['a', 'Toy answer'], ['p', 'Confidence'], ['v', 'Verdict']], { cls: 'compact' });
  }
  drawF4();
  const base4 = CL.map((c) => annotate(c, false)), withAb = CL.map((c) => annotate(c, true)), cleanOk = base4.filter((q, i) => CL[i].kind === 'clean' && q.ok === 'yes').length;
  const wrongNoAbst = base4.filter((q, i) => CL[i].kind !== 'clean' && q.ok === 'no').length;
  prose(rb, `<p>With eight markers and no abstention, the toy labels ${cleanOk} of ${TYPES.length} clean clusters correctly and gives a wrong answer for ${wrongNoAbst} of the 2 doublet and stressed clusters: the lookup always names some cell type. Abstaining below 50% confidence handles ${withAb.filter((q, i) => CL[i].kind !== 'clean' && q.ok === 'yes').length} of the 2 and abstains on ${withAb.filter((q, i) => CL[i].kind === 'clean' && q.ok === 'missed').length} clean clusters; a real LLM's stated confidence is not a calibrated probability, so this threshold is an assumption of the toy. Adding ambient genes shifts probability towards unrelated labels. The practical use is a first-pass label that an annotator checks, reported with its marker evidence. The published concordance (above 75% in most tissues, authors' own evaluation) is measured against manual annotation, which is itself imperfect.</p>`);

  // ---- Step 5: zero-shot lab
  const cache = new Map();
  const cellsFor = (s) => { const k = 'c' + s; if (!cache.has(k)) cache.set(k, s === S0 ? cells : makeCells(s, PER)); return cache.get(k); };
  const modelFor = (ov) => { const k = 'm' + ov; if (!cache.has(k)) cache.set(k, ov === OV0 ? model0 : pretrain(makeCorpus(ov))); return cache.get(k); };
  const hvgFor = (s) => { const k = 'h' + s; if (!cache.has(k)) { const cs = cellsFor(s), E = hvgPca(cs); cache.set(k, { E, m: knnMetrics(E, cs) }); } return cache.get(k); };
  const fmFor = (ov, s) => { const k = `f${ov}|${s}`; if (!cache.has(k)) { const cs = cellsFor(s), mo = modelFor(ov), E = cs.map((c) => embedCell(mo, c.counts)); cache.set(k, { E, m: knnMetrics(E, cs) }); } return cache.get(k); };
  const refAcc = (E, cs) => { const i0 = cs.map((c, i) => i).filter((i) => cs[i].b === 0), i1 = cs.map((c, i) => i).filter((i) => cs[i].b === 1); return centroidAcc(i0.map((i) => E[i]), i0.map((i) => cs[i].t), i1.map((i) => E[i]), i1.map((i) => cs[i].t)); };
  const probe = (E, cs, m, poolB, testB, draws = 10) => {
    const r = rng(3); let tot = 0;
    for (let d = 0; d < draws; d++) {
      const pick = new Set();
      TYPES.forEach((_, t) => r.shuffle(cs.map((c, i) => i).filter((i) => cs[i].t === t && (poolB == null || cs[i].b === poolB))).slice(0, m).forEach((i) => pick.add(i)));
      const P = [...pick], T = cs.map((c, i) => i).filter((i) => !pick.has(i) && (testB == null || cs[i].b === testB));
      tot += centroidAcc(P.map((i) => E[i]), P.map((i) => cs[i].t), T.map((i) => E[i]), T.map((i) => cs[i].t));
    }
    return tot / draws;
  };
  rb = rung(ch, 5, 'The zero-shot test against HVG + PCA',
    `<p>The test that matters for a user is whether the pretrained embedding, <b>used as it is</b>, separates cell types and mixes batches better than the standard pipeline on the same cells. Two metrics: <b>type accuracy</b> (do a cell's 10 nearest neighbours vote for its true type?) and <b>batch mixing</b> (are neighbours drawn from both batches in equal proportion? 1 = fully mixed). Two things are varied. The <b>pretraining coverage</b> is the share of the corpus that comes from the same tissue; the rest is from other tissues. The <b>batch effect</b> is the size of the difference between the two chemistries. The toy model never sees these cells during pretraining, while the HVG selection is fitted to them.</p>`);
  const fl = figure(rb, 'Lab: frozen toy encoder vs HVG + PCA',
    'Both panels show the first two PCs of the respective embedding, so the 2D picture is a projection: the neighbour metrics are computed in the full 8 dimensions. Move the sliders and compare the numbers below the panels. The probe uses a few labelled cells per type (drawn from both batches) to build one centroid per type, and is tested on the remaining cells: a stand-in for fine-tuning with limited labels.');
  toyBadge(fl, 'Frozen = embedding with no training on these cells.');
  let lOv = OV0, lS = S0, lM = 'type', lMl = 5;
  select(fl.controls, { label: 'Colour by', value: lM, options: [['type', 'Cell type'], ['batch', 'Batch']], onChange: (v) => { lM = v; paintL(); } });
  slider(fl.controls, { label: 'Pretraining coverage of this tissue', min: 0, max: 1, step: 0.1, value: lOv, fmt: pc, onInput: (v) => { lOv = Math.round(v * 10) / 10; paintL(); } });
  slider(fl.controls, { label: 'Batch effect', min: 0, max: 2, step: 0.25, value: lS, fmt: f2, onInput: (v) => { lS = v; paintL(); } });
  slider(fl.controls, { label: 'Labelled cells per type (probe)', min: 1, max: 20, step: 1, value: lMl, onInput: (v) => { lMl = v; paintL(); } });
  button(fl.controls, 'Reset', () => { lOv = OV0; lS = S0; lMl = 5; fl.controls.querySelectorAll('input[type=range]').forEach((r, i) => { const v = [OV0, S0, 5][i]; r.value = v; r.dispatchEvent(new Event('input')); }); });
  const sL = stats(fl.root);
  const legL = el('div'); fl.root.append(legL);
  const rowL = el('div', { class: 'fig-row' }); fl.body.append(rowL);
  const pa = el('div'), pb = el('div'); rowL.append(pa, pb);
  const FA = frame(pa, { width: 480, height: 345, x: [-1, 1], y: [-1, 1], xLabel: 'PC 1', yLabel: 'PC 2', margin: { top: 26 } }), FB = frame(pb, { width: 480, height: 345, x: [-1, 1], y: [-1, 1], xLabel: 'PC 1', yLabel: 'PC 2', margin: { top: 26 } });
  FA.layer('t').append('text').attr('x', 0).attr('y', -10).attr('font-size', 12.5).attr('font-weight', 600).attr('fill', ink()).text('Frozen toy encoder (rank-value tokens)');
  FB.layer('t').append('text').attr('x', 0).attr('y', -10).attr('font-size', 12.5).attr('font-weight', 600).attr('fill', ink()).text('HVG + PCA (fitted to these cells)');
  const msgL = html('', 'p', 'fig-caption'); fl.body.append(msgL);
  function paintL() {
    const cs = cellsFor(lS), fm = fmFor(lOv, lS), hv = hvgFor(lS);
    scatterTo(FA, pca(fm.E, 2, 40).scores, cs, lM); scatterTo(FB, pca(hv.E, 2, 40).scores, cs, lM);
    legL.innerHTML = ''; legend(legL, lM === 'type' ? TYPES.map((t, i) => [t, cat(i)]) : [['batch 1', cat(6)], ['batch 2', cat(7)]]);
    const pf = probe(fm.E, cs, lMl, null, null), ph = probe(hv.E, cs, lMl, null, null);
    sL.set({ 'toy encoder: type accuracy': pc(fm.m.acc), 'HVG + PCA: type accuracy': pc(hv.m.acc), 'toy encoder: batch mixing': f2(fm.m.mix), 'HVG + PCA: batch mixing': f2(hv.m.mix), [`toy encoder: probe, ${lMl} labels/type`]: pc(pf), [`HVG + PCA: probe, ${lMl} labels/type`]: pc(ph) });
    msgL.innerHTML = `At coverage ${pc(lOv)} and batch effect ${f2(lS)}, the toy encoder is ${fm.m.acc > hv.m.acc + 0.005 ? 'ahead of' : fm.m.acc < hv.m.acc - 0.005 ? 'behind' : 'level with'} HVG + PCA on type accuracy.`;
  }
  paintL();
  const E1 = (ov, s) => fmFor(ov, s).m, H1 = (s) => hvgFor(s).m;
  const rows5 = [
    { t: 'Coverage <b>70% → 0%</b> (batch effect 1)', s: `Toy encoder type accuracy ${pc(E1(0.7, 1).acc)} → <b>${pc(E1(0, 1).acc)}</b>; HVG + PCA stays at ${pc(H1(1).acc)}`, m: 'A frozen encoder is only as good as its pretraining coverage. HVG selection adapts to the new data, so it does not depend on coverage.' },
    { t: 'Coverage <b>70% → 100%</b>', s: `${pc(E1(0.7, 1).acc)} → <b>${pc(E1(1, 1).acc)}</b>`, m: 'Beyond adequate coverage the gain is small in the toy.' },
    { t: 'Batch effect <b>0 → 2</b> (coverage 70%)', s: `Batch mixing, toy encoder ${f2(E1(0.7, 0).mix)} → <b>${f2(E1(0.7, 2).mix)}</b>; HVG + PCA ${f2(H1(0).mix)} → <b>${f2(H1(2).mix)}</b>`, m: 'Neither pipeline removes a strong batch effect without an explicit correction step. Rank-value encoding gives some robustness to depth only.' },
    { t: 'Labelled cells <b>1 → 20</b> per type (probe, coverage 70%, batch effect 1)', s: `Toy encoder ${pc(probe(fmFor(0.7, 1).E, cells, 1, null, null))} → <b>${pc(probe(fmFor(0.7, 1).E, cells, 20, null, null))}</b>; HVG + PCA ${pc(probe(hvgFor(1).E, cells, 1, null, null))} → <b>${pc(probe(hvgFor(1).E, cells, 20, null, null))}</b>`, m: `Accuracy rises with the number of labelled cells for both embeddings. ${probe(fmFor(0.7, 1).E, cells, 1, null, null) > probe(hvgFor(1).E, cells, 1, null, null) ? 'The embedding that separates types better needs fewer labels, which is the sense in which fine-tuning with limited labels is expected to help.' : 'The embeddings need similar numbers of labels here.'}` },
  ];
  table(rb, rows5, [['t', 'Try'], ['s', 'What you see'], ['m', 'What it means']], { cls: 'compact' });
  const fmWins = E1(0.7, 1).acc > H1(1).acc;
  prose(rb, `<p><b>Reading the result honestly.</b> In this toy, with adequate coverage, the frozen encoder ${fmWins ? 'is ahead of' : 'does not beat'} HVG + PCA on type accuracy (${pc(E1(0.7, 1).acc)} against ${pc(H1(1).acc)}), and with no coverage it ${E1(0, 1).acc > H1(1).acc ? 'is still ahead of' : 'falls behind'} it (${pc(E1(0, 1).acc)}). <b>This does not reproduce the published finding</b> that zero-shot Geneformer and scGPT embeddings were worse than HVG + PCA, scVI and Harmony (${cite('kedzierska', 'Kedzierska et al. 2025')}). The toy is easy in ways real data are not: the cell types are separated by a handful of genes, the batch effect is a simple depth and gene shift, and the encoder was pretrained on the same gene set. Two further features favour the encoder here: rank-value encoding is insensitive to the depth difference between batches, and the HVG + PCA baseline has no batch-correction step, whereas the published comparison included scVI and Harmony. What the toy shows is the structure of a fair test. The baseline must be run on the same cells, and the result depends on how well the pretraining data covers the target tissue. Whether a given real model wins has to be measured on the user's own labelled data.</p>`);
  quiz(rb, 'The HVG baseline uses labels nowhere, but it selects genes on the data being analysed. Is the comparison with a frozen encoder fair?',
    `<p>It is the comparison a user faces: the standard pipeline is always fitted to the new data, while a zero-shot encoder is not. It does favour the baseline on adaptation, which is why published benchmarks also report fine-tuned and integrated variants. A foundation model that is only competitive after fine-tuning is a different claim from one that works zero-shot.</p>`);

  // ---- Step 6: in-context learning
  rb = rung(ch, 6, 'In-context learning: adapting with a prompt',
    `<p>Some recent models (for example ${cite('stackNews', 'Stack')}) take a <b>prompt set of labelled cells</b> as part of the input and adjust their output to that context <b>without updating weights</b>. For a new dataset, a few example cells with known labels are supplied and the model labels the rest. The toy uses the simplest possible version: one centroid per type computed from the prompt cells (batch 2, the new context), compared with a <b>frozen reference</b> built from batch 1 (what the model already knows). It is a nearest-centroid rule, not attention, so it shows the setting rather than the mechanism.</p>`);
  const f6 = figure(rb, 'Accuracy on the new context: frozen reference vs prompt',
    'Test cells are the batch 2 cells not in the prompt. Solid lines: centroids from k prompt cells per type (batch 2), averaged over 10 random prompts. Dashed lines: centroids from all batch 1 cells (frozen reference, independent of k). Both embeddings are those of step 5 at 70% coverage.');
  toyBadge(f6);
  let kP = 3, sP = 1.5;
  slider(f6.controls, { label: 'Prompt cells per type (k)', min: 1, max: 20, step: 1, value: kP, onInput: (v) => { kP = v; paintP(); } });
  slider(f6.controls, { label: 'Shift of the new context', min: 0.25, max: 2, step: 0.25, value: sP, fmt: f2, onInput: (v) => { sP = v; paintP(); } });
  const s6 = stats(f6.root);
  const KP = [1, 2, 3, 5, 8, 12, 20];
  const F6 = frame(f6.body, { width: 820, height: 300, x: [0, 21], y: [0, 1.02], xLabel: 'prompt cells per type (k)', yLabel: 'accuracy on the new context' });
  const ly6 = F6.layer('l'), kL6 = F6.layer('k');
  legend(f6.body, [['toy encoder, prompt', cat(0)], ['toy encoder, frozen reference', cat(0)], ['HVG + PCA, prompt', cat(2)], ['HVG + PCA, frozen reference', cat(2)]]);
  const noteP = html('', 'p', 'fig-caption'); f6.body.append(noteP);
  function paintP() {
    const cs = cellsFor(sP), series = [['toy encoder', fmFor(OV0, sP).E, cat(0)], ['HVG + PCA', hvgFor(sP).E, cat(2)]];
    ly6.selectAll('*').remove(); kL6.selectAll('*').remove();
    const out = {};
    series.forEach(([nm, E, col]) => {
      const ref = refAcc(E, cs);
      const pts = KP.map((k) => ({ k, a: probe(E, cs, k, 1, 1) }));
      ly6.append('line').attr('x1', 0).attr('x2', F6.w).attr('y1', F6.y(ref)).attr('y2', F6.y(ref)).attr('stroke', col).attr('stroke-width', 2).attr('stroke-dasharray', '6 4');
      ly6.append('path').datum(pts).attr('fill', 'none').attr('stroke', col).attr('stroke-width', 2.4).attr('d', d3.line().x((d) => F6.x(d.k)).y((d) => F6.y(d.a)));
      ly6.selectAll('circle.' + nm.replace(/\W/g, '')).data(pts).join('circle').attr('class', nm.replace(/\W/g, '')).attr('cx', (d) => F6.x(d.k)).attr('cy', (d) => F6.y(d.a)).attr('r', 3.8).attr('fill', col).attr('stroke', css('--surface')).attr('stroke-width', 1)
        .on('mousemove', (e, d) => tip(e, `${nm}<br>k = ${d.k}: accuracy ${f2(d.a)}<br>frozen reference ${f2(ref)}`)).on('mouseleave', untip);
      out[nm] = { ref, cur: probe(E, cs, kP, 1, 1) };
    });
    kL6.append('line').attr('x1', F6.x(kP)).attr('x2', F6.x(kP)).attr('y1', 0).attr('y2', F6.h).attr('stroke', ink()).attr('stroke-width', 1.5).attr('stroke-dasharray', '4 3');
    s6.set({ 'toy encoder: frozen reference': pc(out['toy encoder'].ref), [`toy encoder: prompt, k = ${kP}`]: pc(out['toy encoder'].cur), 'HVG + PCA: frozen reference': pc(out['HVG + PCA'].ref), [`HVG + PCA: prompt, k = ${kP}`]: pc(out['HVG + PCA'].cur) });
    noteP.innerHTML = `With a context shift of ${f2(sP)}, ${kP} prompt cells per type ${out['toy encoder'].cur > out['toy encoder'].ref ? 'raise' : 'do not raise'} the toy encoder's accuracy relative to its frozen reference (${pc(out['toy encoder'].ref)} → ${pc(out['toy encoder'].cur)}).`;
  }
  paintP();
  const cK = cellsFor(1.5), nRef = cK.filter((c) => c.b === 0).length;
  const inf = (E) => { const ref = refAcc(E, cK); return { ref, k3: probe(E, cK, 3, 1, 1), k20: probe(E, cK, 20, 1, 1), kM: KP.find((k) => probe(E, cK, k, 1, 1) >= ref - 0.01) }; };
  const iF = inf(fmFor(OV0, 1.5).E), iH = inf(hvgFor(1.5).E);
  prose(rb, `<p>At the default context shift (1.5), the frozen reference reaches ${pc(iF.ref)} for the toy encoder and ${pc(iH.ref)} for HVG + PCA. With 3 prompt cells per type the values are ${pc(iF.k3)} and ${pc(iH.k3)}, and with 20 they are ${pc(iF.k20)} and ${pc(iH.k20)}. The toy encoder's prompt ${iF.kM ? `matches its reference from k = ${iF.kM}` : 'does not reach its reference within k = 20'}; for HVG + PCA the prompt ${iH.kM ? `matches the reference from k = ${iH.kM}` : 'does not reach the reference within k = 20'}. The comparison is lopsided in labelled cells: the reference uses all ${nRef} labelled cells of the old context, the prompt only ${6 * 3}–${6 * 20}. A prompt is useful because it needs <b>no</b> labelled cells from the old context, and the toy shows how many it needs to replace them. These numbers describe the toy setup only. The in-context gains reported for Stack are the authors' own and should be tested on one's own data.</p>`);

  // =====================================================================================
  // 5. PITFALLS
  // =====================================================================================
  ch = chapter(main, 'pitfalls', 'Pitfalls', 'Common problems, how to recognise them, and how to address them.');
  const noOwn = cells.filter((c) => c.enc.order.slice(0, L).filter((j) => typeMarkers[c.t].has(GENES[j])).length < 3).length;
  cards(ch, [
    { tag: 'Symptom: a clean-looking UMAP, no accuracy number', title: 'No baseline on the same cells', body: `A tidy embedding plot is not evidence. In the toy lab, HVG + PCA reaches ${pc(H1(1).acc)} type accuracy at the default settings. <b>Fix:</b> compute type accuracy and batch mixing for HVG + PCA, scVI or Harmony and the foundation embedding on the same cells and report both.` },
    { tag: 'Symptom: good on atlases, poor on a rare tissue', title: 'Pretraining coverage mismatch', body: `In the toy lab, type accuracy of the frozen encoder falls from ${pc(E1(0.7, 1).acc)} to ${pc(E1(0, 1).acc)} when the corpus contains none of the target tissue. <b>Fix:</b> check what tissues and species the model was trained on, and fine-tune if coverage is poor.` },
    { tag: 'Symptom: short or empty sentences', title: 'Vocabulary and truncation', body: `Tokens are genes from a fixed vocabulary, and the sentence is cut to a fixed length. In the toy, a cell has on average ${f1(d3.mean(cells, (c) => c.enc.order.slice(0, L).filter((j) => typeMarkers[c.t].has(GENES[j])).length))} of its own type's markers among the first ${L} tokens, and ${pc(noOwn / cells.length)} of cells have fewer than three. Targeted panels give very short input lists. <b>Fix:</b> map gene identifiers to the model's vocabulary and check how many genes survive.` },
    { tag: 'Symptom: depth or abundance differences ignored', title: 'Magnitudes are discarded', body: 'A rank-based encoding keeps order only. Two cells with the same gene order but very different expression levels give the same sentence. This helps with depth and hides real quantitative differences. <b>Fix:</b> use quantitative methods (pseudobulk differential expression) for effect sizes.' },
    { tag: 'Symptom: a label for a mixed cluster', title: 'LLM annotation always answers', body: `In the toy, ${wrongNoAbst} of the 2 doublet and stressed clusters receive a wrong label. A language model's stated confidence is not a calibrated probability. <b>Fix:</b> send the marker list together with doublet and QC scores, require the model to cite marker genes, and have an annotator check clusters that matter.` },
    { tag: 'Symptom: strong annotation scores on public data', title: 'Memorisation of published markers', body: 'A language model may have seen the paper and marker table behind a public dataset, so benchmarks on well-known tissues overstate performance on new data. <b>Fix:</b> evaluate on unpublished or deliberately obscure clusters.' },
    { tag: 'Symptom: candidate genes without validation', title: 'In-silico deletion hits treated as findings', body: 'Embedding shift after deleting a token measures what the model learned from correlations in the corpus, not a causal effect. The strongest claims for such nominations come from the developers. <b>Fix:</b> treat hits as hypotheses to compare with differential expression, pathways and experiments.' },
    { tag: 'Symptom: labels generated by the model are scored by the model', title: 'Circular evaluation', body: 'Using the foundation model\'s own predicted labels as ground truth, or tuning the prompt on the same clusters used to report accuracy, inflates results. <b>Fix:</b> keep an independently annotated set that is never used for prompt or model selection.' },
  ]);

  // =====================================================================================
  // 6. RECAP
  // =====================================================================================
  ch = chapter(main, 'recap', 'Recap', 'Key points.');
  recap(ch, [
    'A single-cell foundation model is pretrained <b>without labels</b> by predicting hidden genes, then reused through its cell embedding.',
    '<b>Rank-value encoding</b> (Geneformer) divides each gene by its corpus-wide non-zero median and sorts, so that cell-specific genes rise above housekeeping genes. Only order is kept.',
    '<b>Cell2Sentence</b> writes the same ordering as text for a language model. Sentence length trades information against cost, and order alone does not preserve magnitudes.',
    '<b>LLM annotation</b> maps a marker list to a label from the model\'s training text. It always answers, so doublets, novel types and contaminated lists need explicit checks.',
    '<b>Zero-shot</b> use means no training on the new data. It must be compared with HVG + PCA (and scVI or Harmony) on the same cells, measured by type accuracy and batch mixing.',
    'Published independent evaluation found that zero-shot Geneformer and scGPT embeddings were worse than these baselines. In the toy, the outcome depended on pretraining coverage, and the toy does not reproduce that finding.',
    '<b>In-context learning</b> conditions a model on labelled example cells without changing weights. The gains reported are the authors\' own.',
    'Treat in-silico deletions and generated labels as hypotheses, and keep an independent labelled set for evaluation.',
  ]);
  prose(ch, '<h3>In code</h3><p>The recommended workflow is a baseline first, then the foundation embedding, with both scored in the same way. The embedding itself comes from the model\'s own tooling (the Geneformer repository documents tokenisation and embedding extraction), saved per cell and loaded here.</p>');
  code(ch, {
    Python: `import numpy as np, scanpy as sc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

# 1) Baseline: HVG + PCA on the same cells
sc.pp.normalize_total(adata, target_sum=1e4); sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
sc.pp.pca(adata, n_comps=30, use_highly_variable=True)
X_pca = adata.obsm["X_pca"]

# 2) Foundation-model embedding for the same cells, saved by the model's own tooling
#    (for Geneformer: tokenise, then extract cell embeddings; see the repository documentation)
X_fm = np.load("geneformer_cell_embeddings.npy")      # shape (n_cells, d), same cell order

# 3) Score both identically: k-NN type accuracy on held-out annotated cells
knn = KNeighborsClassifier(n_neighbors=10)
for name, X in [("HVG+PCA", X_pca), ("foundation", X_fm)]:
    acc = cross_val_score(knn, X, adata.obs["cell_type"], cv=5).mean()
    print(name, round(acc, 3))

# Batch mixing: compare neighbour batch labels (scib-metrics provides ready-made scores)`,
    R: `library(Seurat); library(class)

# 1) Baseline: HVG + PCA on the same cells
obj <- NormalizeData(obj); obj <- FindVariableFeatures(obj, nfeatures = 2000)
obj <- ScaleData(obj); obj <- RunPCA(obj, npcs = 30)
X_pca <- Embeddings(obj, "pca")

# 2) Foundation-model embedding for the same cells (exported from the model's tooling)
X_fm <- as.matrix(read.csv("geneformer_cell_embeddings.csv", row.names = 1))
X_fm <- X_fm[colnames(obj), ]                       # same cell order

# 3) Score both identically: leave-one-out k-NN accuracy
for (nm in c("pca", "fm")) {
  X <- if (nm == "pca") X_pca else X_fm
  pred <- knn.cv(X, cl = obj$cell_type, k = 10)
  cat(nm, mean(pred == obj$cell_type), "\\n")
}`,
  });
  seeAlso(main, [['concepts', 'transformers', 'Embeddings, attention and transformers'], ['concepts', 'pca', 'Principal component analysis'], ['omics', 'sc-clustering', 'Clustering and cell-type annotation'], ['omics', 'bulk-qc-batch', 'QC, normalisation and batch effects'], ['frontier', 'virtual-cells', 'Virtual cells and perturbation prediction'], ['frontier', 'landscape', 'The 2024–26 landscape']]);
}

// ---------------------------------------------------------------------------------------------
/** Scatter of 2D scores into a frame, coloured by type (circle = batch 1, triangle = batch 2) or by batch. Rescales the axes to the data. */
function scatterTo(F, P, cs, mode) {
  const ex = d3.extent(P, (p) => p[0]), ey = d3.extent(P, (p) => p[1]), px = (ex[1] - ex[0]) * 0.06 || 1, py = (ey[1] - ey[0]) * 0.06 || 1;
  F.x.domain([ex[0] - px, ex[1] + px]); F.y.domain([ey[0] - py, ey[1] + py]);
  F.g.select('.axis.x').call(d3.axisBottom(F.x).ticks(5)); F.g.select('.axis.y').call(d3.axisLeft(F.y).ticks(5));
  F.g.select('.gridline').call(d3.axisLeft(F.y).ticks(5).tickSize(-F.w).tickFormat(''));
  const sym = (b) => d3.symbol(b ? d3.symbolTriangle : d3.symbolCircle, 46)();
  F.layer('pts').selectAll('path').data(cs).join('path').attr('d', (c) => sym(c.b)).attr('transform', (c, i) => `translate(${F.x(P[i][0])},${F.y(P[i][1])})`)
    .attr('fill', (c) => (mode === 'type' ? cat(c.t) : cat(6 + c.b))).attr('fill-opacity', 0.78).attr('stroke', css('--surface')).attr('stroke-width', 0.8)
    .on('mousemove', (e, c) => tip(e, `${TYPES[c.t]} · batch ${c.b + 1}`)).on('mouseleave', untip);
}

/** The pipeline diagram for one cell: counts → rank-value sentence → pretrained model → uses. */
function pipelineSvg(P, model) {
  const wrap = el('div', { class: 'figure' });
  wrap.append(el('div', { class: 'fig-title' }, `One ${TYPES[P.t]} cell through a Geneformer-style pipeline (toy)`));
  const svg = d3.select(wrap).append('svg').attr('viewBox', '0 0 980 250').attr('width', 980).attr('role', 'img').attr('aria-label', 'counts, rank-value sentence, pretrained model, uses');
  const top = P.enc.order.slice(0, 6), raw = P.enc.raw.slice(0, 6);
  const E = embedCell(model, P.counts);
  const boxes = [
    { t: 'Counts', lines: ['top genes by count:', ...raw.map((j) => `${GENES[j].padEnd(8)} ${String(P.counts[j]).padStart(3)}`)] },
    { t: '① Rank-value sentence', lines: ['÷ corpus median, sort:', ...top.map((j, i) => `${i + 1}. ${GENES[j]}`)] },
    { t: '② Pretrained model', lines: ['mask some tokens,', 'predict them from', 'the rest (no labels)', '', 'learns which genes', 'appear together', 'in which cells'] },
    { t: '③ Reuse', lines: ['cell embedding:', `[${E.slice(0, 3).map(f2).join(', ')}, …]`, '', 'then: cluster, label,', 'fine-tune a classifier,', 'delete a token', '(in-silico perturbation)'] },
  ];
  const BW = 212, gap = 36, H = 220;
  boxes.forEach((b, i) => {
    const x0 = 10 + i * (BW + gap), g = svg.append('g').attr('transform', `translate(${x0},14)`);
    g.append('rect').attr('width', BW).attr('height', H).attr('rx', 10).attr('fill', css('--surface-2')).attr('stroke', css('--axis'));
    g.append('text').attr('x', 12).attr('y', 24).attr('font-weight', 700).attr('font-size', 13.5).attr('fill', ink()).text(b.t);
    b.lines.forEach((l, j) => g.append('text').attr('x', 14).attr('y', 52 + j * 21).attr('font-family', css('--mono')).attr('font-size', 12.5).attr('fill', j === 0 && i < 2 ? ink2() : ink()).attr('xml:space', 'preserve').text(l));
    if (i < boxes.length - 1) {
      const ax = x0 + BW + 6, ay = 14 + H / 2;
      svg.append('path').attr('d', `M${ax},${ay} L${ax + gap - 14},${ay} M${ax + gap - 20},${ay - 6} L${ax + gap - 14},${ay} L${ax + gap - 20},${ay + 6}`).attr('fill', 'none').attr('stroke', ink2()).attr('stroke-width', 2);
    }
  });
  return wrap;
}
