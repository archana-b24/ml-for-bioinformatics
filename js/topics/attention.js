// Attention: real (seeded) Q, K, V matrices for a short gene list; three heads with different
// hand-built behaviours; click a token to see its softmax row and the weighted sum of values.
import { header, figure, slider, select, button, stats, section, steps, callout, math, seeAlso, legend, tip, untip, el } from '../lib/ui.js';
import { cat, css, ink, ink2, muted, diverging, heatmap, sequential } from '../lib/plot.js';
import { rng } from '../lib/data.js';
const d3 = window.d3;

// ---- the toy vocabulary: 8 marker genes from 3 programmes ----
const GENES = [
  ['CD3E', 0], ['LYZ', 2], ['MS4A1', 1], ['CD8A', 0], ['CD14', 2], ['GZMB', 0], ['CD79A', 1], ['S100A8', 2],
];
const PROG = ['T / cytotoxic', 'B cell', 'myeloid'];
const D = 8, DK = 4, W1 = 0.9, W2 = 0.35; // model width, head width, positional frequencies

const matmul = (A, B) => A.map((r) => B[0].map((_, j) => r.reduce((s, a, k) => s + a * B[k][j], 0)));
const T = (A) => A[0].map((_, j) => A.map((r) => r[j]));
const softmaxRow = (r) => { const m = Math.max(...r); const e = r.map((v) => Math.exp(v - m)); const s = e.reduce((a, b) => a + b, 0); return e.map((v) => v / s); };
// heatmap() sizes its viewBox to the cells, so rotated column labels on the last columns get clipped; add room on the right.
const widen = ({ svg }) => { const [x, y, w, h] = svg.attr('viewBox').split(' ').map(Number); svg.attr('viewBox', `${x} ${y} ${w + 34} ${h}`).attr('width', w + 34); };
const zeros = (r, c) => Array.from({ length: r }, () => new Array(c).fill(0));

// Embedding: dims 0-3 = gene content (its programme + gene-specific noise), dims 4-7 = sinusoidal position.
function embed(order, seed = 3) {
  const r = rng(seed); const content = {};
  GENES.forEach(([g, p]) => { content[g] = [0, 1, 2, 3].map((k) => (k === p ? 1 : 0) + r.normal(0, 0.2)); });
  return order.map(([g], i) => [...content[g], Math.sin(W1 * i), Math.cos(W1 * i), Math.sin(W2 * i), Math.cos(W2 * i)]);
}

function buildHeads() {
  const r = rng(17);
  const noise = (M, s) => M.map((row) => row.map((v) => v + r.normal(0, s)));
  // Head A: "same programme" — project onto the content dims, so q·k is large when programmes match.
  const A = zeros(D, DK); [0, 1, 2].forEach((k) => { A[k][k] = 3.0; });
  const WQa = noise(A, 0.1), WKa = noise(A, 0.1);
  // Head B: "previous gene" — the query rotates the positional code back by one step, the key reads it as is.
  const Bq = zeros(D, DK), Bk = zeros(D, DK);
  [[4, 5, W1, 0], [6, 7, W2, 2]].forEach(([s, c, w, o]) => {
    Bq[s][o] = 3 * Math.cos(w); Bq[c][o] = -3 * Math.sin(w); Bq[s][o + 1] = 3 * Math.sin(w); Bq[c][o + 1] = 3 * Math.cos(w);
    Bk[s][o] = 3; Bk[c][o + 1] = 3;
  });
  const WQb = noise(Bq, 0.05), WKb = noise(Bk, 0.05);
  // Head C: random, untrained.
  const rand = () => zeros(D, DK).map((row) => row.map(() => r.normal(0, 0.6)));
  const WQc = rand(), WKc = rand();
  const WV = [0, 1, 2].map(() => rand());
  return [
    { name: 'Head 1: same programme', WQ: WQa, WK: WKa, WV: WV[0] },
    { name: 'Head 2: previous gene', WQ: WQb, WK: WKb, WV: WV[1] },
    { name: 'Head 3: random (untrained)', WQ: WQc, WK: WKc, WV: WV[2] },
  ];
}

export default async function render(main) {
  header(main, 'Attention',
    'Attention lets every token in a sequence <b>look at every other token and decide how much to borrow from each</b>. Each token asks a question (its query), every token advertises what it contains (its key), and the answer is a weighted average of what they carry (their values). The weights come from query–key similarity.');

  const HEADS = buildHeads();
  let order = GENES.slice(), head = 0, tau = 1, qi = 0, shuffles = 0;
  const shufR = rng(99);

  function compute(h = head, tauV = tau) {
    const X = embed(order), H = HEADS[h];
    const Q = matmul(X, H.WQ), K = matmul(X, H.WK), V = matmul(X, H.WV);
    const S = matmul(Q, T(K)).map((r) => r.map((v) => v / Math.sqrt(DK)));
    const Aw = S.map((r) => softmaxRow(r.map((v) => v / tauV)));
    const O = matmul(Aw, V);
    return { X, Q, K, V, S, A: Aw, O };
  }

  // ================= Figure 1 =================
  const f1 = figure(main, 'Click a gene to see what it attends to',
    'Eight marker genes as tokens, each embedded as an 8-number vector: its programme (with gene-specific noise) plus a code for its position. Left: the attention matrix softmax(QKᵀ/√d / τ), one row per query; the arcs show the selected row. Right: the selected token\'s output, built line by line as a weighted sum of value vectors. All numbers are computed live from seeded weight matrices.');
  select(f1.controls, { label: 'Head', value: '0', options: HEADS.map((h, i) => [String(i), h.name]), onChange: (v) => { head = +v; drawAll(); } });
  slider(f1.controls, { label: 'Temperature τ', min: -1, max: 0.7, step: 0.05, value: 0, fmt: (v) => (10 ** v).toFixed(2), onInput: (v) => { tau = 10 ** v; drawAll(); } });
  button(f1.controls, 'Shuffle gene order', () => { order = shufR.shuffle(order); shuffles++; qi = 0; drawAll(); });
  button(f1.controls, 'Original order', () => { order = GENES.slice(); qi = 0; drawAll(); });
  const ro1 = stats(f1.root);
  legend(f1.root, PROG.map((p, i) => [p + ' programme', cat(i)]));
  const row1 = el('div', { class: 'fig-row' }); f1.body.append(row1);
  const leftHost = el('div'), rightHost = el('div'); row1.append(leftHost, rightHost);

  function drawLeft(C) {
    leftHost.innerHTML = '';
    const n = order.length, Wd = 540, cellW = 52, x0 = 92, chipY = 128;
    const svg = d3.select(leftHost).append('svg').attr('viewBox', `0 0 ${Wd} 600`).attr('width', Wd);
    const cx = (j) => x0 + j * cellW + cellW / 2;
    // arcs from the query to each key
    const w = C.A[qi];
    order.forEach((_, j) => {
      if (j === qi) {
        svg.append('circle').attr('cx', cx(j)).attr('cy', chipY - 27).attr('r', 10).attr('fill', 'none').attr('stroke', ink()).attr('stroke-width', 0.5 + 6 * w[j]).attr('opacity', 0.15 + 0.85 * w[j]);
      } else {
        const x1 = cx(qi), x2 = cx(j), hgt = 18 + Math.abs(x2 - x1) * 0.3;
        svg.append('path').attr('d', `M${x1},${chipY - 16} C${x1},${chipY - 16 - hgt} ${x2},${chipY - 16 - hgt} ${x2},${chipY - 16}`)
          .attr('fill', 'none').attr('stroke', ink()).attr('stroke-width', 0.5 + 12 * w[j]).attr('opacity', 0.1 + 0.9 * w[j]).attr('stroke-linecap', 'round');
      }
    });
    // token chips
    order.forEach(([g, p], j) => {
      const gg = svg.append('g').style('cursor', 'pointer').on('click', () => { qi = j; drawAll(); })
        .on('mousemove', (e) => tip(e, `<b>${g}</b> (${PROG[p]})<br>weight from ${order[qi][0]}: <b>${w[j].toFixed(3)}</b>`)).on('mouseleave', untip);
      gg.append('rect').attr('x', cx(j) - 25).attr('y', chipY - 14).attr('width', 50).attr('height', 26).attr('rx', 6)
        .attr('fill', css('--surface-2')).attr('stroke', j === qi ? ink() : cat(p)).attr('stroke-width', j === qi ? 2.5 : 1.5);
      gg.append('rect').attr('x', cx(j) - 25).attr('y', chipY + 9).attr('width', 50).attr('height', 3).attr('fill', cat(p));
      gg.append('text').attr('x', cx(j)).attr('y', chipY + 4).attr('text-anchor', 'middle').attr('font-size', 10).attr('font-weight', 600).attr('fill', ink()).text(g);
    });
    svg.append('text').attr('x', 8).attr('y', 18).attr('font-size', 12).attr('fill', ink2()).text(`Query: ${order[qi][0]}  (arc thickness = attention weight)`);
    // heatmap
    const top = 200, cell = cellW, sc = sequential(0, 1);
    svg.append('text').attr('x', x0 + n * cell / 2).attr('y', top - 12).attr('text-anchor', 'middle').attr('font-size', 12).attr('fill', ink2()).text('keys (attended to) →');
    svg.append('text').attr('transform', `translate(14,${top + n * cell * 0.2}) rotate(-90)`).attr('text-anchor', 'end').attr('font-size', 12).attr('fill', ink2()).text('← queries');
    C.A.forEach((r, i) => r.forEach((v, j) => {
      svg.append('rect').attr('x', x0 + j * cell).attr('y', top + i * (cell * 0.72)).attr('width', cell - 2).attr('height', cell * 0.72 - 2).attr('rx', 2).attr('fill', sc(v))
        .style('cursor', 'pointer').on('click', () => { qi = i; drawAll(); })
        .on('mousemove', (e) => tip(e, `${order[i][0]} → ${order[j][0]}<br>score q·k/√d = ${C.S[i][j].toFixed(2)}<br>weight = <b>${v.toFixed(3)}</b>`)).on('mouseleave', untip);
      svg.append('text').attr('x', x0 + j * cell + cell / 2 - 1).attr('y', top + i * cell * 0.72 + cell * 0.36 + 3).attr('text-anchor', 'middle').attr('font-size', 10).attr('pointer-events', 'none')
        .attr('fill', v > 0.55 ? css('--surface') : ink2()).text(v >= 0.005 ? v.toFixed(2).replace(/^0/, '') : '');
    }));
    svg.append('rect').attr('x', x0 - 2).attr('y', top + qi * cell * 0.72 - 2).attr('width', n * cell + 2).attr('height', cell * 0.72 + 2).attr('fill', 'none').attr('stroke', ink()).attr('stroke-width', 2).attr('rx', 3);
    order.forEach(([g, p], i) => {
      svg.append('text').attr('x', x0 - 6).attr('y', top + i * cell * 0.72 + cell * 0.36 + 4).attr('text-anchor', 'end').attr('font-size', 11).attr('fill', i === qi ? ink() : ink2()).attr('font-weight', i === qi ? 700 : 400).text(g);
      svg.append('text').attr('x', x0 + i * cell + cell / 2).attr('y', top + n * cell * 0.72 + 14).attr('text-anchor', 'middle').attr('font-size', 10).attr('fill', ink2()).text(g);
    });
    svg.attr('viewBox', `0 0 ${Wd} ${top + n * cell * 0.72 + 24}`);
  }

  function drawRight(C) {
    rightHost.innerHTML = '';
    const n = order.length, Wd = 540, rowH = 34, top = 56, cv = 22;
    const svg = d3.select(rightHost).append('svg').attr('viewBox', `0 0 ${Wd} ${top + (n + 2) * rowH + 20}`).attr('width', Wd);
    const col = diverging(1.5);
    const xs = { tok: 64, score: 112, bar: 128, v: 262, wv: 400 };
    svg.append('text').attr('x', 8).attr('y', 16).attr('font-size', 12).attr('fill', ink2()).text(`output(${order[qi][0]}) = Σⱼ weightⱼ · valueⱼ`);
    [['key', xs.tok, 'end'], ['score', xs.score - 4, 'end'], ['weight', xs.bar, 'start'], ['value vⱼ', xs.v, 'start'], ['weight · vⱼ', xs.wv, 'start']]
      .forEach(([t, x, a]) => svg.append('text').attr('x', x).attr('y', top - 12).attr('text-anchor', a).attr('font-size', 11).attr('fill', muted()).text(t));
    const w = C.A[qi];
    const cells = (y, vec, x, alpha = 1, label) => vec.forEach((v, k) => svg.append('rect').attr('x', x + k * (cv + 2)).attr('y', y).attr('width', cv).attr('height', rowH - 10).attr('rx', 2)
      .attr('fill', col(v)).attr('opacity', alpha).on('mousemove', (e) => tip(e, `${label} [${k + 1}] = <b>${v.toFixed(3)}</b>`)).on('mouseleave', untip));
    order.forEach(([g, p], j) => {
      const y = top + j * rowH;
      svg.append('text').attr('x', xs.tok).attr('y', y + 16).attr('text-anchor', 'end').attr('font-size', 11).attr('font-weight', j === qi ? 700 : 400).attr('fill', ink()).text(g);
      svg.append('text').attr('x', xs.score - 4).attr('y', y + 16).attr('text-anchor', 'end').attr('font-size', 11).attr('font-family', css('--mono')).attr('fill', ink2()).text(C.S[qi][j].toFixed(1));
      svg.append('rect').attr('x', xs.bar).attr('y', y + 4).attr('width', 120).attr('height', rowH - 16).attr('fill', css('--surface-2')).attr('rx', 2);
      svg.append('rect').attr('x', xs.bar).attr('y', y + 4).attr('width', 120 * w[j]).attr('height', rowH - 16).attr('fill', cat(p)).attr('rx', 2);
      svg.append('text').attr('x', w[j] > 0.8 ? xs.bar + 120 * w[j] - 4 : xs.bar + 120 * w[j] + 4).attr('y', y + 16).attr('text-anchor', w[j] > 0.8 ? 'end' : 'start').attr('font-size', 10).attr('fill', w[j] > 0.8 ? css('--surface') : ink2()).text(w[j].toFixed(2));
      cells(y, C.V[j], xs.v, 1, `v(${g})`);
      cells(y, C.V[j].map((v) => v * w[j]), xs.wv, 1, `${w[j].toFixed(2)} · v(${g})`);
    });
    const ySum = top + n * rowH + 10;
    svg.append('line').attr('x1', xs.wv).attr('x2', xs.wv + 4 * (cv + 2)).attr('y1', ySum - 6).attr('y2', ySum - 6).attr('stroke', ink()).attr('stroke-width', 1.5);
    svg.append('text').attr('x', xs.wv - 8).attr('y', ySum + 16).attr('text-anchor', 'end').attr('font-size', 12).attr('font-weight', 700).attr('fill', ink()).text('sum = output →');
    cells(ySum, C.O[qi], xs.wv, 1, `output(${order[qi][0]})`);
    svg.append('text').attr('x', xs.v).attr('y', ySum + rowH + 10).attr('font-size', 10).attr('fill', muted()).text('blue = negative, red = positive');
  }

  // ================= Figure 2: the matrices =================
  const f2 = figure(main, 'Inside the head: Q, K, V and the score matrix',
    'Every token\'s embedding x (a row of X) is multiplied by three learned matrices: Q = X·W<sub>Q</sub>, K = X·W<sub>K</sub>, V = X·W<sub>V</sub>. Each is 8 tokens × 4 numbers here. The scores are QKᵀ/√4: row i, column j is how well query i matches key j. Softmax along each row turns scores into the weights above. Hover any cell.');
  const matHost = el('div', { style: 'display:flex;flex-wrap:wrap;gap:6px 18px;align-items:flex-end' }); f2.body.append(matHost);

  function drawMats(C) {
    matHost.innerHTML = '';
    const names = order.map(([g]) => g);
    const block = (title, M, opts) => {
      const d = el('div'); matHost.append(d);
      d.append(el('div', { style: `font-size:12px;color:${ink2()};margin:0 0 2px 4px` }, title));
      const m = d3.max(M.flat(), Math.abs) || 1;
      (opts.colLabels ? widen : () => {})(heatmap(d, M, Object.assign({ cell: 17, color: diverging(m), onHover: (e, i, j, v) => tip(e, `${title}<br>${names[i]}, ${opts.colLabels ? opts.colLabels[j] : 'dim ' + (j + 1)}: <b>${v.toFixed(3)}</b>`) }, opts)));
    };
    block('X: embeddings (8 tokens × 8 dims)', C.X, { rowLabels: names, labelW: 52, colLabels: ['prog T', 'prog B', 'prog M', 'gene', 'sin .9p', 'cos .9p', 'sin .35p', 'cos .35p'], labelH: 52 });
    block('Q = X·W_Q', C.Q, { rowLabels: names, labelW: 52 });
    block('K = X·W_K', C.K, { rowLabels: names, labelW: 52 });
    block('V = X·W_V', C.V, { rowLabels: names, labelW: 52 });
    block('scores QKᵀ/√d', C.S, { rowLabels: names, labelW: 52, colLabels: names, labelH: 52, cell: 20 });
  }

  // ================= Figure 3: multi-head =================
  const f3 = figure(main, 'Three heads, three different questions',
    'A transformer layer runs several heads in parallel on the same tokens, each with its own W<sub>Q</sub>, W<sub>K</sub>, W<sub>V</sub>. Here head 1 has learned "which genes share my programme", head 2 "which gene came just before me", and head 3 is still random. Their outputs (4 numbers each) are concatenated into a 12-number vector and mixed by one more matrix W<sub>O</sub>. Click a map to make that head the selected one above.');
  const ro3 = stats(f3.root);
  const mhHost = el('div', { style: 'display:flex;flex-wrap:wrap;gap:14px 22px;align-items:flex-start' }); f3.body.append(mhHost);

  function drawHeads() {
    mhHost.innerHTML = '';
    const names = order.map(([g]) => g); const ent = [];
    HEADS.forEach((H, h) => {
      const C = compute(h);
      const d = el('div', { style: 'cursor:pointer' + (h === head ? `;outline:2px solid ${ink()};outline-offset:3px;border-radius:6px` : '') });
      d.addEventListener('click', () => { head = h; f1.controls.querySelector('select').value = String(h); drawAll(); });
      mhHost.append(d);
      d.append(el('div', { style: `font-size:13px;font-weight:600;margin-bottom:2px` }, H.name));
      widen(heatmap(d, C.A, { cell: 26, color: sequential(0, 1), rowLabels: names, colLabels: names, labelW: 54, labelH: 46, onHover: (e, i, j, v) => tip(e, `${H.name}<br>${names[i]} → ${names[j]}: <b>${v.toFixed(3)}</b>`) }));
      const e = d3.mean(C.A, (r) => -r.reduce((s, p) => s + (p > 0 ? p * Math.log2(p) : 0), 0)); ent.push(e);
    });
    ro3.set(Object.fromEntries(HEADS.map((H, h) => [`${H.name.split(':')[0]}: mean tokens attended (2^entropy)`, (2 ** ent[h]).toFixed(1)])));
  }

  function drawAll() {
    const C = compute();
    drawLeft(C); drawRight(C); drawMats(C); drawHeads();
    const w = C.A[qi], ent = -w.reduce((s, p) => s + (p > 0 ? p * Math.log2(p) : 0), 0);
    const sameProg = w.reduce((s, p, j) => s + (order[j][1] === order[qi][1] ? p : 0), 0);
    const prevW = qi > 0 ? w[qi - 1] : null;
    ro1.set({ query: order[qi][0], 'top key': order[w.indexOf(Math.max(...w))][0], 'max weight': Math.max(...w).toFixed(2), 'weight on same-programme genes': sameProg.toFixed(2), 'weight on previous gene': prevW == null ? 'n/a (first)' : prevW.toFixed(2), 'effective # tokens (2^entropy)': (2 ** ent).toFixed(1) });
  }
  drawAll();

  section(main, 'What to notice', `<ul>
    <li>With <b>head 1</b>, click CD3E: almost all its weight goes to CD3E, CD8A and GZMB, the three T-cell genes, even though they sit at positions 1, 4 and 6. Attention has no notion of distance unless the model learns one: any token can reach any other in a single step.</li>
    <li>Switch to <b>head 2</b> and click any gene after the first: the weight lands on the gene immediately to its left. This head ignores content and reads only the positional code. The first gene has no predecessor, so it falls back mostly on itself.</li>
    <li>Press <b>Shuffle gene order</b>. Head 1's pattern moves with the genes (same programmes still find each other); head 2's pattern stays a stripe just below the diagonal. One layer can hold a content head and a position head side by side.</li>
    <li><b>Temperature</b>: at τ = 0.1 each row becomes nearly one-hot (hard lookup); as τ grows towards 5 every row flattens towards uniform weights (1/8 each) and the output drifts towards the plain average of all values. The √d in the formula is a fixed temperature that stops dot products in wide heads from saturating the softmax.</li>
    <li><b>Head 3</b> (random weights) spreads its attention more evenly than the trained-by-hand heads: compare the "mean tokens attended" readouts. Training is what turns a diffuse head into a sharp, meaningful one.</li>
    <li>On the right, the output is literally the weighted rows of V added up. If the weights are one-hot, the output is a copy of one value vector; otherwise it is a blend.</li>
  </ul>`);

  steps(main, 'How it works', [
    'Embed each token as a vector x (see <a href="#/concepts/embeddings">embeddings</a>), and add a positional code so the model can know order.',
    'Project every token three ways: query <code>q = x·W_Q</code> ("what am I looking for?"), key <code>k = x·W_K</code> ("what do I offer?") and value <code>v = x·W_V</code> ("what do I pass on if chosen?").',
    'Score every query against every key with a dot product, divided by √d<sub>k</sub> to keep the scores in a range where softmax still has a useful gradient.',
    'Softmax each row of scores into weights that are positive and sum to 1.',
    'Each token\'s output is the weighted sum of all value vectors. Run several heads in parallel, concatenate their outputs, and mix them with W<sub>O</sub>.',
    'All of W<sub>Q</sub>, W<sub>K</sub>, W<sub>V</sub> and W<sub>O</sub> are learned by gradient descent. The patterns here were set by hand so you can see what learned heads often end up doing.',
  ]);
  math(main, 'Attention(Q, K, V) = softmax( Q Kᵀ / √d_k ) V          Q = X W_Q,  K = X W_K,  V = X W_V\nMultiHead(X) = concat(head₁, …, head_h) W_O\ncost: n² scores per head  (n = sequence length)');

  callout(main, '', 'In biology', `<ul>
    <li><b>Gene tokens:</b> Geneformer and scGPT treat each cell as a sequence of gene tokens and use attention so every gene can condition on every other expressed gene. People inspect their attention maps for regulatory hints (e.g. a TF attending to its targets), but attention weights are not proven causal links.</li>
    <li><b>DNA:</b> Enformer and AlphaGenome put attention on top of convolutions so that a promoter can attend to an enhancer up to ~100 kb away, the long-range interaction the RNN page shows is hard for recurrent models.</li>
    <li><b>Spatial and pathology:</b> Novae uses graph attention over a cell's spatial neighbours; pathology foundation models (vision transformers such as UNI, Virchow or Prov-GigaPath) attend between image patches, and attention-based multiple-instance learning (ABMIL) weights tiles of a whole-slide image to predict a slide-level label.</li>
    <li><b>Proteins:</b> in AlphaFold2 and protein language models (ESM), attention between residue pairs tracks residue contacts in the 3-D structure.</li>
  </ul>`);
  callout(main, 'warn', 'Pitfalls', `<ul>
    <li><b>Attention is not explanation.</b> A large weight means a value vector was mixed in, not that it caused the prediction. Different heads and layers can tell different stories, and the values themselves may be near zero.</li>
    <li>Without positional encodings, attention is <b>permutation-invariant</b>: it sees a bag of tokens. For gene sets that can be what you want; for DNA it is not.</li>
    <li>Cost grows as n². A cell with 2,000 expressed genes means 4 million scores per head per layer, which is why single-cell models cap the input length or use efficient attention variants.</li>
    <li>Softmax saturates if scores get large (big d<sub>k</sub> without the √d<sub>k</sub>, or a very low temperature): the gradient through near one-hot weights is close to zero.</li>
  </ul>`);
  seeAlso(main, [['concepts', 'transformer', 'Transformers'], ['concepts', 'embeddings', 'Embeddings'], ['concepts', 'rnn', 'Recurrent networks'], ['concepts', 'gnn', 'Graph neural networks (GAT)'], ['frontier', 'geneformer', 'Geneformer']]);
}
