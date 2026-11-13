// Transformers: clickable block diagram with a live parameter count, sinusoidal positional encodings,
// and causal vs bidirectional masks applied to a real (seeded) attention head on a gene "sentence".
import { header, figure, slider, select, button, stats, section, steps, callout, math, seeAlso, legend, tip, untip, every, el } from '../lib/ui.js';
import { frame, cat, css, ink, ink2, muted, diverging, sequential } from '../lib/plot.js';
import { rng } from '../lib/data.js';
const d3 = window.d3;

const BLOCKS = [
  { id: 'tok', label: 'Tokens', sub: 'gene IDs', x: 0, what: 'The input is a sequence of n discrete tokens. For Geneformer a token is a gene ID, and a cell is its expressed genes listed from most to least (relatively) expressed. For an English model a token is a word piece.', shape: 'n integers' },
  { id: 'emb', label: 'Embedding', sub: '+ position', x: 1, what: 'Each token ID looks up a learned vector of length d (a row of a V × d table). A positional encoding is added so the model knows where each token sits; without it, attention sees an unordered bag.', shape: 'n × d' },
  { id: 'mha', label: 'Multi-head', sub: 'attention', x: 2, block: true, what: 'Every token gathers information from every other token (or only earlier ones, under a causal mask). h heads run in parallel, each with its own queries, keys and values of width d/h, and their outputs are concatenated and mixed. This is the only place where tokens exchange information.', shape: 'n × d' },
  { id: 'an1', label: 'Add & norm', sub: 'residual', x: 3, block: true, what: 'The attention output is added to its own input (a residual or skip connection), then layer-normalised (each token vector rescaled to mean 0, variance 1, with learned gain and bias). The residual means each block only has to learn a correction, and gradients have a direct path back to the input.', shape: 'n × d' },
  { id: 'mlp', label: 'MLP', sub: 'per token', x: 4, block: true, what: 'A two-layer network (d → 4d → d, usually with a GELU activation) applied to each token separately, with the same weights at every position. About two-thirds of each block\'s parameters live here. It transforms what attention gathered.', shape: 'n × d (4d inside)' },
  { id: 'an2', label: 'Add & norm', sub: 'residual', x: 5, block: true, what: 'A second residual connection and layer norm. The output has exactly the same shape as the block\'s input, which is why blocks can be stacked N times.', shape: 'n × d' },
  { id: 'head', label: 'Head', sub: 'task output', x: 6.6, what: 'The final token vectors feed a task head. In pretraining it is a linear layer + softmax over the vocabulary (predict the masked or the next token). For downstream use you take the token vectors, or a pooled cell vector, as an embedding, or fine-tune a small classifier on top (e.g. cell type).', shape: 'n × V  (or 1 × classes)' },
];

const VOCABS = { genes: ['Gene vocabulary (25,000 genes)', 25000], bert: ['English word pieces (30,522, as BERT)', 30522] };
const GENE_SENT = ['CD3E', 'CCL5', 'NKG7', 'GZMA', 'CD8A', 'IL7R', 'GZMK', 'CST7'];

export default async function render(main) {
  header(main, 'Transformers',
    'A transformer is a stack of identical blocks. Each block lets every token <b>gather information from the others (attention)</b>, then <b>processes each token on its own (an MLP)</b>, with a shortcut around each step. Add position information at the input, choose which tokens may see which (the mask), and the same architecture becomes BERT, GPT, Geneformer or scGPT.');

  // ================= Figure 1: block diagram =================
  const f1 = figure(main, 'The architecture, block by block',
    'Click any block to see what it does and the shape of the data leaving it (n tokens, each a vector of width d). The shaded group is one transformer block; it is repeated N times with different weights. Curved arrows are the residual connections. The parameter count is computed from the settings.');
  let N = 6, dm = 256, vocab = 'genes', selB = 'mha', pulse = -1, stopPulse = null;
  slider(f1.controls, { label: 'Layers N', min: 1, max: 24, value: 6, onInput: (v) => { N = v; draw1(); } });
  select(f1.controls, { label: 'Width d', value: '256', options: ['128', '256', '512', '768', '1024'].map((v) => [v, v]), onChange: (v) => { dm = +v; draw1(); } });
  select(f1.controls, { label: 'Vocabulary', value: 'genes', options: Object.entries(VOCABS).map(([k, v]) => [k, v[0]]), onChange: (v) => { vocab = v; draw1(); } });
  button(f1.controls, 'Send a token through', () => {
    if (stopPulse) stopPulse(); pulse = 0; draw1();
    stopPulse = every(420, () => { pulse++; if (pulse >= BLOCKS.length) { pulse = -1; stopPulse(); stopPulse = null; } draw1(); });
  }, true);
  const ro1 = stats(f1.root);
  const diagHost = el('div'); f1.body.append(diagHost);
  const info = el('div', { class: 'callout', style: 'margin-top:8px' }); f1.body.append(info);

  function params() {
    const V = VOCABS[vocab][1], d = dm;
    const attn = 4 * d * d + 4 * d, mlp = 8 * d * d + 5 * d, ln = 4 * d;
    const block = attn + mlp + ln, emb = V * d;
    return { attn, mlp, ln, block, emb, total: emb + N * block + 2 * d };
  }
  const fmtN = (v) => (v >= 1e9 ? (v / 1e9).toFixed(2) + ' B' : v >= 1e6 ? (v / 1e6).toFixed(1) + ' M' : (v / 1e3).toFixed(0) + ' k');

  function draw1() {
    diagHost.innerHTML = '';
    const Wd = 1060, Hd = 250, bw = 112, bh = 74, gap = 140, y0 = 92;
    const svg = d3.select(diagHost).append('svg').attr('viewBox', `0 0 ${Wd} ${Hd}`).attr('width', Wd);
    svg.append('defs').append('marker').attr('id', 'tf-arr').attr('viewBox', '0 0 10 10').attr('refX', 9).attr('refY', 5).attr('markerWidth', 6).attr('markerHeight', 6).attr('orient', 'auto')
      .append('path').attr('d', 'M0,0 L10,5 L0,10 z').attr('fill', muted());
    const X = (b) => 14 + b.x * gap;
    // repeated-block background
    const bx0 = X(BLOCKS[2]) - 16, bx1 = X(BLOCKS[5]) + bw + 16;
    for (let k = Math.min(N, 3) - 1; k >= 1; k--) svg.append('rect').attr('x', bx0 + 6 * k).attr('y', y0 - 44 - 6 * k).attr('width', bx1 - bx0).attr('height', bh + 88).attr('rx', 12).attr('fill', css('--surface-2')).attr('stroke', css('--grid'));
    svg.append('rect').attr('x', bx0).attr('y', y0 - 44).attr('width', bx1 - bx0).attr('height', bh + 88).attr('rx', 12).attr('fill', css('--surface-2')).attr('stroke', ink2()).attr('stroke-dasharray', '5 4');
    svg.append('text').attr('x', bx1 - 10).attr('y', y0 + bh + 36).attr('text-anchor', 'end').attr('font-size', 14).attr('font-weight', 700).attr('fill', ink()).text(`× N = ${N}`);
    svg.append('text').attr('x', bx0 + 10).attr('y', y0 + bh + 36).attr('font-size', 12).attr('fill', ink2()).text('one transformer block');
    // residual arcs: input of MHA -> add1, input of MLP -> add2
    [[2, 3], [4, 5]].forEach(([a, b]) => {
      const xa = X(BLOCKS[a]) - 22, xb = X(BLOCKS[b]) + bw / 2;
      svg.append('path').attr('d', `M${xa},${y0 + bh / 2} L${xa},${y0 - 22} L${xb},${y0 - 22} L${xb},${y0 - 2}`).attr('fill', 'none').attr('stroke', cat(1)).attr('stroke-width', 2).attr('marker-end', 'url(#tf-arr)');
    });
    svg.append('text').attr('x', X(BLOCKS[2]) + 40).attr('y', y0 - 28).attr('font-size', 11).attr('fill', ink2()).text('residual: x + Attention(x)');
    svg.append('text').attr('x', X(BLOCKS[4]) + 40).attr('y', y0 - 28).attr('font-size', 11).attr('fill', ink2()).text('residual: x + MLP(x)');
    // arrows between blocks
    BLOCKS.slice(0, -1).forEach((b, i) => {
      const nb = BLOCKS[i + 1];
      svg.append('line').attr('x1', X(b) + bw).attr('x2', X(nb) - 3).attr('y1', y0 + bh / 2).attr('y2', y0 + bh / 2).attr('stroke', muted()).attr('stroke-width', 1.5).attr('marker-end', 'url(#tf-arr)');
    });
    // blocks
    BLOCKS.forEach((b, i) => {
      const g = svg.append('g').style('cursor', 'pointer').on('click', () => { selB = b.id; draw1(); });
      const on = b.id === selB, hot = i === pulse;
      g.append('rect').attr('x', X(b)).attr('y', y0).attr('width', bw).attr('height', bh).attr('rx', 9)
        .attr('fill', hot ? css('--accent') : on ? css('--accent-soft') : css('--surface')).attr('stroke', on ? ink() : ink2()).attr('stroke-width', on ? 2.2 : 1.2);
      g.append('text').attr('x', X(b) + bw / 2).attr('y', y0 + 32).attr('text-anchor', 'middle').attr('font-size', 14).attr('font-weight', 600).attr('fill', hot ? css('--surface') : ink()).text(b.label);
      g.append('text').attr('x', X(b) + bw / 2).attr('y', y0 + 52).attr('text-anchor', 'middle').attr('font-size', 12).attr('fill', hot ? css('--surface') : ink2()).text(b.sub);
      g.append('text').attr('x', X(b) + bw / 2).attr('y', y0 + bh + 18).attr('text-anchor', 'middle').attr('font-size', 11).attr('font-family', css('--mono')).attr('fill', muted())
        .text(b.shape.replace('d', dm).replace('V', VOCABS[vocab][1].toLocaleString()).replace(/n /, 'n ').replace('4' + dm, String(4 * dm)));
    });
    const b = BLOCKS.find((q) => q.id === selB), P = params();
    info.innerHTML = `<span class="lbl">${b.label} ${b.sub}</span><div>${b.what}</div>`;
    ro1.set({ 'parameters per block': fmtN(P.block), 'of which MLP': (100 * P.mlp / P.block).toFixed(0) + '%', 'embedding table V×d': fmtN(P.emb), 'total (head tied to embedding)': fmtN(P.total), heads: `${Math.max(1, dm / 64)} × width ${Math.min(64, dm)}` });
  }
  draw1();

  // ================= Figure 2: positional encoding =================
  const f2 = figure(main, 'Positional encoding: a barcode for each position',
    'Left: the sinusoidal encoding from the original transformer paper. Row p is the vector added to the token at position p; each pair of columns is a sine and cosine wave, fast on the left, slow on the right. Right: the dot product of position p\'s code with every other position\'s. Hover the map for values.');
  let dPE = 64, pSel = 20;
  select(f2.controls, { label: 'Encoding width', value: '64', options: ['16', '32', '64'].map((v) => [v, v]), onChange: (v) => { dPE = +v; draw2(); } });
  slider(f2.controls, { label: 'Position p', min: 0, max: 63, value: 20, onInput: (v) => { pSel = v; draw2(); } });
  const ro2 = stats(f2.root);
  const row2 = el('div', { class: 'fig-row' }); f2.body.append(row2);
  const peHost = el('div'), simHost = el('div'); row2.append(peHost, simHost);
  const NP = 64;
  const pe = (p, i, d) => { const k = Math.floor(i / 2), w = 1 / 10000 ** (2 * k / d); return i % 2 === 0 ? Math.sin(p * w) : Math.cos(p * w); };

  function draw2() {
    peHost.innerHTML = ''; simHost.innerHTML = '';
    const cw = 440 / dPE, ch = 5.4, left = 46, top = 10;
    const svg = d3.select(peHost).append('svg').attr('viewBox', `0 0 ${left + 450} ${top + NP * ch + 46}`).attr('width', left + 450);
    const col = diverging(1);
    for (let p = 0; p < NP; p++) for (let i = 0; i < dPE; i++) {
      svg.append('rect').attr('x', left + i * cw).attr('y', top + p * ch).attr('width', cw + 0.3).attr('height', ch + 0.3).attr('fill', col(pe(p, i, dPE)))
        .on('mousemove', (e) => tip(e, `position ${p}, dim ${i} (${i % 2 ? 'cos' : 'sin'}, wavelength ${(2 * Math.PI * 10000 ** (2 * Math.floor(i / 2) / dPE)).toFixed(1)})<br>value <b>${pe(p, i, dPE).toFixed(3)}</b>`)).on('mouseleave', untip);
    }
    svg.append('rect').attr('x', left - 1).attr('y', top + pSel * ch - 1).attr('width', dPE * cw + 2).attr('height', ch + 2).attr('fill', 'none').attr('stroke', ink()).attr('stroke-width', 2);
    [0, 16, 32, 48, 63].forEach((p) => svg.append('text').attr('x', left - 6).attr('y', top + p * ch + ch).attr('text-anchor', 'end').attr('font-size', 11).attr('fill', muted()).text(p));
    svg.append('text').attr('transform', `translate(12,${top + NP * ch / 2}) rotate(-90)`).attr('text-anchor', 'middle').attr('font-size', 12).attr('fill', ink2()).text('position p');
    svg.append('text').attr('x', left + 220).attr('y', top + NP * ch + 18).attr('text-anchor', 'middle').attr('font-size', 12).attr('fill', ink2()).text(`encoding dimension (0 … ${dPE - 1}): fast waves → slow waves`);
    // similarity
    const sims = d3.range(NP).map((q) => d3.range(dPE).reduce((s, i) => s + pe(pSel, i, dPE) * pe(q, i, dPE), 0));
    const F = frame(simHost, { width: 520, height: 380, x: [0, NP - 1], y: [d3.min(sims) - 1, dPE / 2 + 1], xLabel: 'other position q', yLabel: `PE(${pSel}) · PE(q)` });
    F.layer('l').append('path').datum(sims).attr('d', d3.line().x((v, q) => F.x(q)).y((v) => F.y(v))).attr('fill', 'none').attr('stroke', cat(0)).attr('stroke-width', 2);
    F.layer('l').append('line').attr('x1', F.x(pSel)).attr('x2', F.x(pSel)).attr('y1', 0).attr('y2', F.h).attr('stroke', muted()).attr('stroke-dasharray', '4 4');
    F.layer('p').selectAll('circle').data(sims).join('circle').attr('cx', (v, q) => F.x(q)).attr('cy', (v) => F.y(v)).attr('r', (v, q) => (q === pSel ? 6 : 3)).attr('fill', cat(0)).attr('stroke', css('--surface')).attr('stroke-width', 1)
      .on('mousemove', (e, v) => tip(e, `q = ${sims.indexOf(v)}: <b>${v.toFixed(2)}</b>`)).on('mouseleave', untip);
    const d1 = sims[Math.min(NP - 1, pSel + 1)], d10 = sims[Math.min(NP - 1, pSel + 10)];
    ro2.set({ 'PE(p)·PE(p) = d/2': sims[pSel].toFixed(1), 'with p+1': d1.toFixed(1), 'with p+10': d10.toFixed(1), 'same at p=5 vs 6 (shift-invariant)': d3.range(dPE).reduce((s, i) => s + pe(5, i, dPE) * pe(6, i, dPE), 0).toFixed(1) });
  }
  draw2();

  // ================= Figure 3: masks =================
  const f3 = figure(main, 'Bidirectional vs causal: the mask decides what each token may see',
    'A cell written as a gene "sentence" (genes ordered from highest to lowest relative expression, as Geneformer does). Left: the mask, row = query, column = key; blocked pairs are set to −∞ before the softmax. Right: the attention weights of one seeded random head after masking. Click a row to highlight it. The training objective changes with the mask.');
  let mode = 'bert', rowSel = 4;
  select(f3.controls, { label: 'Model style', value: 'bert', options: [['bert', 'Bidirectional encoder (BERT, Geneformer): masked-token prediction'], ['gpt', 'Causal decoder (GPT, Cell2Sentence): next-token prediction']], onChange: (v) => { mode = v; draw3(); } });
  const ro3 = stats(f3.root);
  const taskHost = el('div'); f3.body.append(taskHost);
  const row3 = el('div', { class: 'fig-row' }); f3.body.append(row3);
  const maskHost = el('div'), wHost = el('div'); row3.append(maskHost, wHost);
  legend(f3.root, [['allowed', css('--accent')], ['blocked (−∞)', css('--surface-2')], ['masked / target token', cat(1)]]);
  // a seeded random head over seeded random embeddings
  const r3 = rng(5), n3 = GENE_SENT.length, d3k = 8;
  const S0 = d3.range(n3).map(() => d3.range(n3).map(() => r3.normal(0, 1.2)));
  const MASKED = [2, 5]; // positions replaced by [MASK] in the BERT view (~25% here; 15% in BERT)

  function draw3() {
    taskHost.innerHTML = ''; maskHost.innerHTML = ''; wHost.innerHTML = '';
    const allowed = (i, j) => (mode === 'bert' ? true : j <= i);
    const W = S0.map((r, i) => { const v = r.map((s, j) => (allowed(i, j) ? s : -Infinity)); const m = Math.max(...v); const e = v.map((x) => (x === -Infinity ? 0 : Math.exp(x - m))); const z = d3.sum(e); return e.map((x) => x / z); });
    // the training task strip
    const svg = d3.select(taskHost).append('svg').attr('viewBox', '0 0 1060 132').attr('width', 1060);
    const tx = (j) => 150 + j * 108;
    svg.append('text').attr('x', 8).attr('y', 40).attr('font-size', 12).attr('fill', ink2()).text('input');
    svg.append('text').attr('x', 8).attr('y', 112).attr('font-size', 12).attr('fill', ink2()).text('predict');
    GENE_SENT.forEach((g, j) => {
      const isMask = mode === 'bert' && MASKED.includes(j);
      svg.append('rect').attr('x', tx(j) - 46).attr('y', 22).attr('width', 92).attr('height', 28).attr('rx', 6).attr('fill', isMask ? cat(1) : css('--surface-2')).attr('stroke', j === rowSel ? ink() : 'none').attr('stroke-width', 2);
      svg.append('text').attr('x', tx(j)).attr('y', 41).attr('text-anchor', 'middle').attr('font-size', 12).attr('font-weight', 600).attr('fill', isMask ? css('--surface') : ink()).text(isMask ? '[MASK]' : g);
      svg.append('text').attr('x', tx(j)).attr('y', 14).attr('text-anchor', 'middle').attr('font-size', 10).attr('fill', muted()).text(`rank ${j + 1}`);
      let target = null;
      if (mode === 'bert' && isMask) target = g;
      if (mode === 'gpt') target = j < n3 - 1 ? GENE_SENT[j + 1] : '(end)';
      if (target) {
        svg.append('line').attr('x1', tx(j)).attr('x2', tx(j)).attr('y1', 52).attr('y2', 90).attr('stroke', cat(1)).attr('stroke-width', 2);
        svg.append('rect').attr('x', tx(j) - 46).attr('y', 94).attr('width', 92).attr('height', 26).attr('rx', 6).attr('fill', 'none').attr('stroke', cat(1)).attr('stroke-width', 1.5);
        svg.append('text').attr('x', tx(j)).attr('y', 112).attr('text-anchor', 'middle').attr('font-size', 12).attr('fill', ink()).text(target);
      }
    });
    // matrices
    const mat = (host, title, M, colorFn, fmtFn) => {
      const cell = 44, left = 64, top = 66;
      const s = d3.select(host).append('svg').attr('viewBox', `0 0 ${left + n3 * cell + 10} ${top + n3 * cell + 8}`).attr('width', left + n3 * cell + 10);
      s.append('text').attr('x', left).attr('y', 14).attr('font-size', 13).attr('font-weight', 600).attr('fill', ink()).text(title);
      const lab = (j) => (mode === 'bert' && MASKED.includes(j) ? '[MASK]' : GENE_SENT[j]);
      GENE_SENT.forEach((g, j) => {
        s.append('text').attr('transform', `translate(${left + j * cell + cell / 2 + 3},${top - 6}) rotate(-45)`).attr('font-size', 10.5).attr('fill', ink2()).text(lab(j));
        s.append('text').attr('x', left - 6).attr('y', top + j * cell + cell / 2 + 4).attr('text-anchor', 'end').attr('font-size', 10.5).attr('font-weight', j === rowSel ? 700 : 400).attr('fill', j === rowSel ? ink() : ink2()).text(lab(j));
      });
      M.forEach((r, i) => r.forEach((v, j) => {
        s.append('rect').attr('x', left + j * cell).attr('y', top + i * cell).attr('width', cell - 2).attr('height', cell - 2).attr('rx', 3).attr('fill', colorFn(v, i, j))
          .attr('opacity', i === rowSel ? 1 : 0.6).style('cursor', 'pointer').on('click', () => { rowSel = i; draw3(); })
          .on('mousemove', (e) => tip(e, `query ${lab(i)} → key ${lab(j)}: ${allowed(i, j) ? 'allowed' : 'blocked'}${allowed(i, j) ? `<br>weight <b>${W[i][j].toFixed(3)}</b>` : ''}`)).on('mouseleave', untip);
        const t = fmtFn(v, i, j);
        if (t) s.append('text').attr('x', left + j * cell + cell / 2 - 1).attr('y', top + i * cell + cell / 2 + 4).attr('text-anchor', 'middle').attr('font-size', 10.5).attr('pointer-events', 'none').attr('fill', ink()).text(t);
      }));
      s.append('rect').attr('x', left - 2).attr('y', top + rowSel * cell - 2).attr('width', n3 * cell + 2).attr('height', cell + 2).attr('fill', 'none').attr('stroke', ink()).attr('stroke-width', 2).attr('rx', 4);
    };
    const A = GENE_SENT.map((_, i) => GENE_SENT.map((__, j) => (allowed(i, j) ? 1 : 0)));
    mat(maskHost, mode === 'bert' ? 'Mask: full (every token sees all)' : 'Mask: causal (lower triangle)', A, (v) => (v ? css('--accent-soft') : css('--surface-2')), (v) => (v ? '' : '×'));
    const sc = sequential(0, 0.8);
    mat(wHost, 'Attention weights after masking', W, (v, i, j) => (allowed(i, j) ? sc(v) : css('--surface-2')), (v, i, j) => (allowed(i, j) ? v.toFixed(2).replace(/^0/, '') : ''));
    const blocked = A.flat().filter((v) => !v).length;
    ro3.set({ 'pairs blocked': `${blocked} of ${n3 * n3}`, [`keys visible to ${GENE_SENT[rowSel]}`]: A[rowSel].filter(Boolean).length, 'training targets per sentence': mode === 'bert' ? `${MASKED.length} (masked positions)` : `${n3} (every next token)`, 'can generate text left→right': mode === 'bert' ? 'no' : 'yes' });
  }
  draw3();

  section(main, 'What to notice', `<ul>
    <li>The parameter readout: set <b>d = 768, N = 12</b> with the BERT vocabulary and the total is about 109 M, close to BERT-base's published ~110 M (the rest is position and segment embeddings). Parameters grow with <b>d²</b>: doubling the width quadruples each block, while doubling N only doubles the block total.</li>
    <li>"Of which MLP" stays at about two-thirds whatever the width: the d → 4d → d layer holds 8d² weights against attention\'s 4d².</li>
    <li>In the positional encoding, the left columns flicker quickly down the positions and the right columns barely change. Together they give each position a unique code, like the hands of a clock.</li>
    <li>On the right, the dot product peaks at q = p and falls off with distance. It depends only on the gap: compare the "p+1" readout with "5 vs 6". So a model can learn "attend one step back" once and apply it anywhere, which is exactly what head 2 on the <a href="#/concepts/attention">attention page</a> does.</li>
    <li>Switch to the <b>causal</b> mask: 28 of the 64 pairs are blocked, and the first gene can only attend to itself. Each row's weights still sum to 1 over the keys it can see.</li>
    <li>Count the training targets: the masked LM learns from 2 of 8 positions per sentence (15% in BERT and Geneformer), the causal LM from all 8. That is one reason decoders are cheaper to pretrain per token; the price is that each token only sees its left context.</li>
  </ul>`);

  steps(main, 'How it works', [
    '<b>Tokenise</b> the input and look up an embedding for each token; add a positional encoding (fixed sinusoids as here, or learned vectors, or rotary encodings that rotate queries and keys by position).',
    'Pass the n × d matrix through a <b>block</b>: multi-head self-attention (tokens exchange information), add the input back and layer-normalise, then an MLP on each token, add and normalise again. Modern models usually normalise <i>before</i> each sub-layer ("pre-norm"), which trains more stably.',
    'Stack N blocks. Early layers tend to capture local, surface patterns; later layers more abstract, contextual ones.',
    'Choose the <b>mask and objective</b>. Encoder (BERT-style): no mask, hide ~15% of tokens and predict them from both sides; good for embeddings and classification. Decoder (GPT-style): causal mask, predict every next token; can generate.',
    'Pretrain on a large unlabelled corpus, then use the token or pooled embeddings directly, or fine-tune with a small task head on labelled data.',
  ]);
  math(main, 'block(x):  a = LayerNorm(x + MultiHeadAttention(x, mask))\n           y = LayerNorm(a + W₂ · GELU(W₁ a))          W₁: d→4d, W₂: 4d→d\nPE(p, 2k) = sin(p / 10000^(2k/d))    PE(p, 2k+1) = cos(p / 10000^(2k/d))\nparams/block ≈ 12 d²   (attention 4d², MLP 8d²)\ncausal mask:  score(i, j) = −∞ for j > i');

  callout(main, '', 'In biology', `<ul>
    <li><b>Geneformer</b> (Theodoris et al., Nature 2023) is a BERT-style encoder. A cell becomes a sequence of up to 2,048 gene tokens ranked by expression relative to each gene\'s median across the corpus (rank-value encoding), and the model learns by predicting masked genes. Because position = rank, the positional encoding carries the expression information. See <a href="#/frontier/geneformer">Geneformer: rank-value encoding</a>.</li>
    <li><b>scGPT</b> (Cui et al., Nat Methods 2024) embeds each gene token together with its binned expression value and is pretrained generatively, with a special attention mask that lets genes being predicted see only the "known" genes. It is a hybrid between the two masks above.</li>
    <li><b>Cell2Sentence / C2S-Scale</b> writes a cell as plain text, gene names in descending expression, and trains an ordinary GPT-style causal language model on it, so the same model can read papers and "write" cells. See <a href="#/frontier/cell2sentence">Cell2Sentence</a>.</li>
    <li>Beyond cells: Enformer and AlphaGenome (DNA), ESM (proteins) and pathology foundation models (vision transformers over H&amp;E patches) all use this block.</li>
    <li>A reality check worth knowing: zero-shot embeddings from these models often do not beat HVG + PCA or scVI for clustering and batch integration (<a href="#/frontier/zero-shot-check">the zero-shot check</a>).</li>
  </ul>`);
  callout(main, 'warn', 'Pitfalls', `<ul>
    <li>Attention cost grows as n². Single-cell transformers cap the input (e.g. the top 2,048 genes), so genes ranked below the cut are invisible to the model.</li>
    <li>Transformers are data-hungry. On a few thousand cells a linear model or scVI is usually better; the architecture pays off only with large pretraining corpora.</li>
    <li>Encoders are not generators, and decoders see only the left context. Pick the style that matches the task.</li>
    <li>Pretraining corpora overrepresent some tissues and technologies (e.g. 10x blood), so embeddings for rare tissues can be poor.</li>
    <li>Training is fragile: warm-up, learning-rate schedules and normalisation placement matter (see <a href="#/concepts/training-tricks">training tricks</a>).</li>
  </ul>`);
  seeAlso(main, [['concepts', 'attention', 'Attention'], ['concepts', 'embeddings', 'Embeddings'], ['concepts', 'foundation-models', 'Self-supervision & foundation models'], ['frontier', 'geneformer', 'Geneformer'], ['frontier', 'cell2sentence', 'Cell2Sentence']]);
}
