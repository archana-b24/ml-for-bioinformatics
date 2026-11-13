// Geneformer: rank-value encoding, masked-gene prediction, in-silico deletion, rebuilt on a 40-gene toy corpus.
import { header, figure, slider, select, button, stats, section, steps, math, legend, seeAlso, tip, untip, el } from '../lib/ui.js';
import { frame, cat, css, ink, ink2, muted } from '../lib/plot.js';
import { rng } from '../lib/data.js';
import { sourceCard, toyBadge, coreIdea, realityCheck, useIt, cite, pca } from './_shared.js';
const d3 = window.d3;

// ---- toy corpus: 4 immune cell types, 40 genes ----
const HK = { ACTB: 60, GAPDH: 35, MALAT1: 90, TMSB4X: 70, EEF1A1: 55, RPL13: 45, RPS27: 50, B2M: 40, FTL: 30, 'MT-CO1': 35 };
const MARK = [
  { CD3E: 6, CD3D: 5, IL7R: 4, CCR7: 3, CD2: 4, LTB: 7, TCF7: 2.5 },
  { MS4A1: 6, CD79A: 7, CD79B: 4, BANK1: 3, CD19: 2.5 },
  { LYZ: 9, CD14: 4, S100A8: 8, S100A9: 8, FCN1: 4, CST3: 5 },
  { NKG7: 8, GNLY: 9, KLRD1: 4, PRF1: 3, GZMB: 4 },
];
const OTHER = { JUN: 1.5, FOS: 1.5, HSPA1A: 1, IFI6: 0.8, ISG15: 0.8, XIST: 1.2, 'HLA-DRA': 0 };
const TYPES = ['T cell', 'B cell', 'Monocyte', 'NK cell'];
const GENES = [...Object.keys(HK), ...MARK.flatMap((m) => Object.keys(m)), ...Object.keys(OTHER)];
const G = GENES.length, PER = 60, L = 12;
const gclass = (g, t) => (g in HK ? 'hk' : (MARK[t] && g in MARK[t]) ? 'own' : 'other');

function lam(g, t) {
  if (g in HK) return HK[g] * (g === 'FTL' && t === 2 ? 2.5 : 1);
  if (g === 'HLA-DRA') return t === 1 || t === 2 ? 6 : 0.2;
  if (g in OTHER) return OTHER[g];
  return g in MARK[t] ? MARK[t][g] : 0.6;
}

function buildCorpus() {
  const r = rng(11), cells = [];
  for (let t = 0; t < 4; t++) for (let i = 0; i < PER; i++) {
    const size = Math.exp(r.normal(0, 0.35));
    cells.push({ type: t, idx: i, counts: GENES.map((g) => r.poisson(lam(g, t) * size)) });
  }
  // normalise to 10k per cell, then compute each gene's non-zero median across the corpus
  cells.forEach((c) => { const tot = c.counts.reduce((a, b) => a + b, 0); c.norm = c.counts.map((v) => (v / tot) * 1e4); });
  const med = GENES.map((_, j) => d3.median(cells.map((c) => c.norm[j]).filter((v) => v > 0)) || 1);
  cells.forEach((c) => {
    c.rv = c.norm.map((v, j) => v / med[j]);
    c.rawOrder = d3.range(G).filter((j) => c.counts[j] > 0).sort((a, b) => c.counts[b] - c.counts[a]);
    c.tokens = d3.range(G).filter((j) => c.counts[j] > 0).sort((a, b) => c.rv[b] - c.rv[a]);
  });
  return { cells, med };
}

export default async function render(main) {
  header(main, 'Geneformer: rank-value encoding',
    'Geneformer writes each cell as a <b>sentence of gene tokens, ordered by how unusually high each gene is in that cell</b>, then trains a BERT-style model to fill in hidden genes. Cell embeddings and "in-silico deletions" fall out of that one pretraining task.');
  sourceCard(main, {
    org: 'Theodoris lab (Gladstone / Broad); NVIDIA BioNeMo hosts a training recipe',
    when: 'Nature 2023 (V1, ~30M cells, Genecorpus-30M); V2 on ~104M cells (Dec 2024)',
    links: ['geneformer'],
    claim: 'Pretraining enables "context-aware predictions in settings with limited data in network biology"; fine-tuning with limited data "consistently boosted predictive accuracy", and in-silico perturbation can nominate disease-driving genes (model card wording).',
  });
  coreIdea(main, `<p>Language models need tokens in an order. A cell has no word order, so Geneformer makes one: <b>normalise each gene by its typical (non-zero median) level across the whole corpus, then sort the genes from highest to lowest</b>. Housekeeping genes such as <i>ACTB</i> are high everywhere, so after this division they sink. Genes that are high <i>for this cell compared with other cells</i> (markers, transcription factors) rise to the front.</p>
  <p><b>Key ML ingredient:</b> a BERT-style <a href="#/concepts/transformer">transformer</a> trained by <b>masked language modelling</b>: hide about 15% of tokens and predict them from the rest. Below, a co-occurrence counter stands in for the transformer so you can see the training objective, not the architecture.</p>`);

  const { cells, med } = buildCorpus();
  const train = cells.filter((c) => c.idx % 2 === 0), test = cells.filter((c) => c.idx % 2 === 1);
  let cur = cells[2 * PER + 4];

  // ---------- Figure 1: raw order vs rank-value encoding ----------
  const f1 = figure(main, '1 · One cell, two orderings',
    'Left: genes sorted by raw UMI count. Right: Geneformer-style rank-value encoding (count ÷ cell total, then ÷ the gene\'s non-zero median in the corpus). Bars show the value being sorted, scaled to the top gene. Hover a bar for numbers.');
  toyBadge(f1, `${cells.length} simulated cells, ${G} genes, ${TYPES.length} types.`);
  select(f1.controls, { label: 'Cell type', value: '2', options: TYPES.map((t, i) => [String(i), t]), onChange: (v) => { cur = cells[+v * PER + cellSl.get()]; updateAll(); } });
  const cellSl = slider(f1.controls, { label: 'Cell #', min: 0, max: PER - 1, step: 1, value: 4, onInput: (v) => { cur = cells[cur.type * PER + v]; updateAll(); } });
  legend(f1.root, [['housekeeping', cat(1)], ['marker of this cell\'s type', cat(0)], ['other gene', muted()]]);
  const r1 = el('div', { class: 'fig-row' }); f1.body.append(r1);
  const left = el('div'), right = el('div'); r1.append(left, right);
  const st1 = stats(f1.root);
  const colOf = (c) => (c === 'hk' ? cat(1) : c === 'own' ? cat(0) : muted());

  function rankList(parent, order, val, fmtV, title) {
    parent.innerHTML = '';
    const N = 15, rowH = 21, W = 420, H = 34 + N * rowH;
    const svg = d3.select(parent).append('svg').attr('viewBox', `0 0 ${W} ${H}`).attr('width', W);
    svg.append('text').attr('x', 0).attr('y', 16).attr('fill', ink()).attr('font-weight', 600).attr('font-size', 13).text(title);
    const top = order.slice(0, N), vmax = val(top[0]) || 1;
    top.forEach((j, i) => {
      const y = 28 + i * rowH, g = GENES[j], c = gclass(g, cur.type);
      svg.append('text').attr('x', 18).attr('y', y + 14).attr('text-anchor', 'end').attr('font-size', 11).attr('fill', muted()).text(i + 1);
      svg.append('text').attr('x', 26).attr('y', y + 14).attr('font-size', 12).attr('fill', ink()).attr('font-family', css('--mono')).text(g);
      svg.append('rect').attr('x', 100).attr('y', y + 3).attr('height', rowH - 7).attr('rx', 3)
        .attr('width', Math.max(2, (W - 170) * val(j) / vmax)).attr('fill', colOf(c))
        .on('mousemove', (e) => tip(e, `<b>${g}</b><br>raw count ${cur.counts[j]}<br>per 10k: ${cur.norm[j].toFixed(0)}<br>corpus non-zero median: ${med[j].toFixed(0)}<br>rank value: ${cur.rv[j].toFixed(2)}`)).on('mouseleave', untip);
      svg.append('text').attr('x', W - 4).attr('y', y + 14).attr('text-anchor', 'end').attr('font-size', 11).attr('fill', ink2()).text(fmtV(j));
    });
  }
  function drawF1() {
    rankList(left, cur.rawOrder, (j) => cur.counts[j], (j) => cur.counts[j], 'Raw count order');
    rankList(right, cur.tokens, (j) => cur.rv[j], (j) => cur.rv[j].toFixed(2), 'Rank-value encoding (Geneformer)');
    const hkRaw = cur.rawOrder.slice(0, 10).filter((j) => gclass(GENES[j], cur.type) === 'hk').length;
    const hkRv = cur.tokens.slice(0, 10).filter((j) => gclass(GENES[j], cur.type) === 'hk').length;
    const mk = (o) => o.slice(0, 10).filter((j) => gclass(GENES[j], cur.type) === 'own').length;
    st1.set({ 'housekeeping in top 10 (raw)': hkRaw, 'housekeeping in top 10 (rank-value)': hkRv, 'own markers in top 10 (raw)': mk(cur.rawOrder), 'own markers in top 10 (rank-value)': mk(cur.tokens) });
  }

  // ---------- toy "model": co-occurrence counts among the first L tokens of training cells ----------
  const co = Array.from({ length: G }, () => new Float64Array(G)), freq = new Float64Array(G);
  train.forEach((c) => { const s = c.tokens.slice(0, L); s.forEach((a) => { freq[a]++; s.forEach((b) => { if (a !== b) co[a][b]++; }); }); });
  const alpha = 0.5;
  function predict(context) { // naive-Bayes style score: sum over context genes of log P(g | c)
    const inCtx = new Set(context);
    return d3.range(G).filter((g) => !inCtx.has(g)).map((g) => {
      let s = Math.log((freq[g] + alpha) / (train.length + alpha * G));
      context.forEach((c) => { s += Math.log((co[g][c] + alpha) / (freq[c] + alpha * G)); });
      return { g, s };
    }).sort((a, b) => b.s - a.s);
  }
  // held-out accuracy (top-1), vs always guessing the most frequent missing gene
  let hit = 0, base = 0, tot = 0;
  const freqOrder = d3.range(G).sort((a, b) => freq[b] - freq[a]);
  test.forEach((c) => { const s = c.tokens.slice(0, L); s.forEach((m, k) => {
    const ctx = s.filter((_, i) => i !== k); tot++;
    if (predict(ctx)[0].g === m) hit++;
    if (freqOrder.find((g) => !ctx.includes(g)) === m) base++;
  }); });

  // ---------- Figure 2: mask and predict ----------
  const f2 = figure(main, '2 · Mask a token, predict it from the others',
    `The first ${L} tokens of the selected cell's sentence. One position is masked; the toy model scores every gene not already in the sentence by how often it co-occurred with the visible genes in the training cells (even-numbered cells). Click a chip to mask it.`);
  toyBadge(f2, 'A co-occurrence counter replaces the transformer.');
  const maskSl = slider(f2.controls, { label: 'Masked position', min: 1, max: L, step: 1, value: 3, onInput: () => drawF2() });
  button(f2.controls, 'Mask random', () => { maskSl.set(1 + Math.floor(rngM() * L)); drawF2(); });
  const rngM = rng(5);
  const chips = el('div', { style: 'display:flex;flex-wrap:wrap;gap:6px;margin:6px 0 10px' }); f2.body.append(chips);
  const predBox = el('div'); f2.body.append(predBox);
  const st2 = stats(f2.root);
  function drawF2() {
    const s = cur.tokens.slice(0, L), k = maskSl.get() - 1;
    chips.innerHTML = '';
    s.forEach((j, i) => {
      const c = el('button', { class: 'btn', style: `font-family:var(--mono);font-size:12.5px;${i === k ? 'border-color:var(--accent);background:var(--accent-soft)' : ''}`, title: 'mask this token' }, i === k ? '[MASK]' : GENES[j]);
      c.onclick = () => { maskSl.set(i + 1); drawF2(); };
      chips.append(c);
    });
    const ctx = s.filter((_, i) => i !== k), preds = predict(ctx).slice(0, 6), truth = s[k];
    const mx = preds[0].s; const p = preds.map((d) => Math.exp(d.s - mx)); const Z = d3.sum(p);
    predBox.innerHTML = '';
    const F = frame(predBox, { width: 560, height: 220, x: [0, 1], y: [0, 6], yScale: d3.scaleBand().padding(0.2), margin: { left: 90 }, xLabel: 'relative score among the top 6 (softmax)', grid: false, axes: false });
    F.y.domain(preds.map((d) => GENES[d.g]));
    F.g.append('g').attr('class', 'axis y').call(d3.axisLeft(F.y));
    F.g.append('g').attr('class', 'axis x').attr('transform', `translate(0,${F.h})`).call(d3.axisBottom(F.x).ticks(5));
    F.layer('b').selectAll('rect').data(preds).join('rect').attr('x', 0).attr('y', (d) => F.y(GENES[d.g])).attr('height', F.y.bandwidth())
      .attr('width', (d, i) => F.x(p[i] / Z)).attr('rx', 3).attr('fill', (d) => (d.g === truth ? cat(2) : muted()))
      .on('mousemove', (e, d) => tip(e, `<b>${GENES[d.g]}</b>${d.g === truth ? ' (the hidden gene)' : ''}<br>log score ${d.s.toFixed(2)}`)).on('mouseleave', untip);
    const rank = predict(ctx).findIndex((d) => d.g === truth) + 1;
    st2.set({ 'hidden gene': GENES[truth], 'its rank among predictions': rank, 'toy model top-1 on held-out cells': `${(100 * hit / tot).toFixed(0)}%`, 'guess-most-frequent baseline': `${(100 * base / tot).toFixed(0)}%` });
  }
  legend(f2.root, [['the hidden (true) gene', cat(2)], ['other candidates', muted()]]);

  // ---------- Figure 3: in-silico deletion ----------
  // gene embeddings: PCA of row-normalised co-occurrence profiles; cell embedding: position-weighted mean of its first L tokens
  const prof = d3.range(G).map((a) => { const s = d3.sum(co[a]) + 1e-9; return d3.range(G).map((b) => Math.sqrt(co[a][b] / s)); });
  const gEmb = pca(prof, 6, 40).scores;
  const D = gEmb[0].length;
  function embed(tokens) {
    const v = new Array(D).fill(0); let ws = 0;
    tokens.slice(0, L).forEach((j, i) => { const w = 1 / (1 + 0.15 * i); ws += w; for (let d = 0; d < D; d++) v[d] += w * gEmb[j][d]; });
    return v.map((x) => x / ws);
  }
  const cos = (a, b) => d3.sum(a, (x, i) => x * b[i]) / Math.sqrt(d3.sum(a, (x) => x * x) * d3.sum(b, (x) => x * x));
  cells.forEach((c) => { c.emb = embed(c.tokens); });
  const P = pca(cells.map((c) => c.emb), 2, 50);
  const proj = (v) => P.axes.map((a) => d3.sum(a, (x, d) => x * (v[d] - P.mean[d])));
  cells.forEach((c, i) => { c.xy = P.scores[i]; });
  const cent = TYPES.map((_, t) => d3.range(D).map((d) => d3.mean(cells.filter((c) => c.type === t), (c) => c.emb[d])));

  const f3 = figure(main, '3 · In-silico deletion: drop a gene token, watch the cell move',
    'Left: every cell\'s toy embedding (position-weighted mean of its first 12 gene vectors), projected to 2D. The ring is the selected cell; the arrow shows where it moves when the chosen gene is deleted from its sentence (later genes shift up to fill the gap). Right: embedding shift (1 − cosine similarity) for deleting each of the cell\'s first 12 tokens. Click a bar to delete that gene.');
  toyBadge(f3);
  const r3 = el('div', { class: 'fig-row' }); f3.body.append(r3);
  const mapDiv = el('div'), barDiv = el('div'); r3.append(mapDiv, barDiv);
  legend(f3.root, TYPES.map((t, i) => [t, cat(i)]));
  const st3 = stats(f3.root);
  const ext = (k) => d3.extent(cells, (c) => c.xy[k]);
  const [x0, x1] = ext(0), [y0, y1] = ext(1), px = (x1 - x0) * 0.08, py = (y1 - y0) * 0.08;
  const M = frame(mapDiv, { width: 480, height: 400, x: [x0 - px, x1 + px], y: [y0 - py, y1 + py], xLabel: 'embedding PC1', yLabel: 'embedding PC2' });
  M.layer('pts').selectAll('circle').data(cells).join('circle').attr('cx', (c) => M.x(c.xy[0])).attr('cy', (c) => M.y(c.xy[1])).attr('r', 4)
    .attr('fill', (c) => cat(c.type)).attr('fill-opacity', 0.55).attr('stroke', css('--surface')).attr('stroke-width', 1)
    .on('mousemove', (e, c) => tip(e, `${TYPES[c.type]} #${c.idx}<br>first tokens: ${c.tokens.slice(0, 5).map((j) => GENES[j]).join(' ')}`)).on('mouseleave', untip)
    .on('click', (e, c) => { cur = c; cellSl.set(c.idx); f1.controls.querySelector('select').value = String(c.type); updateAll(); });
  M.svg.append('defs').append('marker').attr('id', 'gf-arrow').attr('viewBox', '0 0 10 10').attr('refX', 9).attr('refY', 5).attr('markerWidth', 7).attr('markerHeight', 7).attr('orient', 'auto')
    .append('path').attr('d', 'M0,0 L10,5 L0,10 z').attr('fill', ink());
  let delGene = null;
  function shiftFor(j) { const t2 = cur.tokens.filter((g) => g !== j); const e2 = embed(t2); return { e2, shift: 1 - cos(cur.emb, e2) }; }
  function drawF3() {
    const toks = cur.tokens.slice(0, L);
    const shifts = toks.map((j) => ({ j, ...shiftFor(j) }));
    if (delGene == null || !toks.includes(delGene)) delGene = shifts.reduce((a, b) => (b.shift > a.shift ? b : a)).j;
    const sel = shifts.find((s) => s.j === delGene), xy2 = proj(sel.e2);
    M.layer('sel').selectAll('*').remove();
    const ly = M.layer('sel');
    ly.append('line').attr('x1', M.x(cur.xy[0])).attr('y1', M.y(cur.xy[1])).attr('x2', M.x(xy2[0])).attr('y2', M.y(xy2[1]))
      .attr('stroke', ink()).attr('stroke-width', 2).attr('marker-end', 'url(#gf-arrow)');
    ly.append('circle').attr('cx', M.x(cur.xy[0])).attr('cy', M.y(cur.xy[1])).attr('r', 8).attr('fill', 'none').attr('stroke', ink()).attr('stroke-width', 2);
    barDiv.innerHTML = '';
    const B = frame(barDiv, { width: 420, height: 400, x: [0, d3.max(shifts, (s) => s.shift) * 1.1 || 0.01], y: [0, 1], yScale: d3.scaleBand().padding(0.18), margin: { left: 84 }, xLabel: 'embedding shift (1 − cosine)', grid: false, axes: false });
    B.y.domain(toks.map((j) => GENES[j]));
    B.g.append('g').attr('class', 'axis y').call(d3.axisLeft(B.y));
    B.g.append('g').attr('class', 'axis x').attr('transform', `translate(0,${B.h})`).call(d3.axisBottom(B.x).ticks(4, '.3f'));
    B.layer('b').selectAll('rect').data(shifts).join('rect').attr('x', 0).attr('y', (s) => B.y(GENES[s.j])).attr('height', B.y.bandwidth()).attr('rx', 3)
      .attr('width', (s) => B.x(s.shift)).attr('fill', (s) => (s.j === delGene ? cat(cur.type) : muted())).style('cursor', 'pointer')
      .on('click', (e, s) => { delGene = s.j; drawF3(); })
      .on('mousemove', (e, s) => tip(e, `delete <b>${GENES[s.j]}</b><br>shift ${s.shift.toFixed(4)}`)).on('mouseleave', untip);
    const near = (v) => d3.minIndex(cent, (c) => -cos(v, c));
    st3.set({ 'deleted gene': GENES[delGene], 'embedding shift': sel.shift.toFixed(4), 'nearest type centroid before': TYPES[near(cur.emb)], 'after deletion': TYPES[near(sel.e2)] });
  }

  function updateAll() { delGene = null; drawF1(); drawF2(); drawF3(); }
  updateAll();

  section(main, 'What to notice', `<ul>
    <li>In figure 1, the raw-count list of almost any cell starts with housekeeping genes (<i>MALAT1</i>, <i>TMSB4X</i>, <i>ACTB</i>). In the rank-value list they fall away and the cell's own markers move to the front. The readouts count this for the selected cell.</li>
    <li>The encoding throws away absolute magnitude: only the order survives. In the default monocyte, <i>CD14</i> has 6 UMIs and ranks 3rd, while <i>ACTB</i> has 57 UMIs and does not make the top 15.</li>
    <li>In figure 2, mask a marker such as <i>LYZ</i> in a monocyte: the other monocyte tokens make it easy to guess. Masking a noise gene such as <i>FOS</i> is much harder. Compare the toy model's held-out accuracy with the "guess the most frequent gene" baseline.</li>
    <li>In figure 3, the shift depends on two things: how different the deleted gene's vector is from the rest of the sentence (deleting <i>LYZ</i> in the default monocyte moves it most), and its position, because early tokens carry more weight and deleting them shifts every later gene up. A large shift is a statement about the model, not proof of a biological dependency.</li>
  </ul>`);
  steps(main, 'How it works', [
    'Build a large corpus of single-cell profiles (Geneformer V2: 104M cells, per the model card).',
    'For each gene, compute its <b>non-zero median</b> expression across the corpus after per-cell depth normalisation.',
    'For each cell, divide by those medians and sort genes descending. The ordered gene IDs are the tokens (input length is capped, so low-ranked genes are dropped: 2048 tokens for V1, 4096 for V2).',
    'Pretrain a BERT-style transformer: mask 15% of the genes in each cell and predict their identity from the unmasked context (model card).',
    'Extract a cell embedding from the model (our toy simply averages gene vectors, weighted by position). Fine-tune with a small labelled set for classification tasks.',
    '<b>In-silico deletion:</b> remove a gene\'s token, re-embed, and measure the cosine shift of the cell (or of other genes). Rank genes by shift to nominate candidates.',
  ]);
  math(main, 'rank value(g, cell) = (count_g / total_counts × 10⁴) / median_{cells with count_g>0}(same)\ntokens = genes sorted by rank value, descending\nloss = − Σ_{masked positions} log P(token | unmasked tokens)');
  realityCheck(main, `<ul>
    <li>${cite('kedzierska', 'Kedzierska et al. 2025 (Genome Biology)')} evaluated Geneformer and scGPT <b>zero-shot</b> (embeddings without fine-tuning) and found them worse than simple baselines such as HVG + PCA, scVI and Harmony for cell-type clustering and batch integration. See <a href="#/frontier/zero-shot-check">the zero-shot check</a>.</li>
    <li>For perturbation prediction, ${cite('ahlmann', 'Ahlmann-Eltze et al. 2025')} found that simple additive and linear baselines matched or beat foundation models (five foundation models and two other deep-learning models on double perturbations). See <a href="#/frontier/perturbation-arena">the baselines arena</a>.</li>
    <li>The strongest claims (gains with limited data, in-silico perturbation nominating disease genes) are the developers' own; treat in-silico deletion hits as hypotheses that need wet-lab validation.</li>
  </ul>`);
  useIt(main, `<ul>
    <li>Run a <b>simple baseline first</b> (HVG + PCA, scVI) and only keep the Geneformer embedding if it beats it on your own labelled cells.</li>
    <li>Fine-tuning a classifier head on your annotated cells is where the published gains are; zero-shot embeddings are the weak point.</li>
    <li>Use in-silico deletion to <b>rank</b> candidate genes for a CRISPR screen, then check them against your own DE results and pathway knowledge.</li>
    <li>Gene ID mapping matters: tokens are Ensembl IDs from the pretraining vocabulary; genes missing from the vocabulary (and spatial panels with only a few hundred genes) give the model very short sentences.</li>
  </ul>`);
  seeAlso(main, [['concepts', 'transformer', 'Transformers'], ['concepts', 'attention', 'Attention'], ['concepts', 'embeddings', 'Embeddings'], ['concepts', 'foundation-models', 'Self-supervision & foundation models'],
    ['omics', 'sc-embedding', 'Normalise, HVGs, PCA'], ['omics', 'sc-annotation', 'Cell-type annotation'], ['frontier', 'cell2sentence', 'Cell2Sentence'], ['frontier', 'zero-shot-check', 'Zero-shot reality check']]);
}
