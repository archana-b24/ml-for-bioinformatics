// Zero-shot reality check: a frozen "foundation" encoder vs plain HVG + PCA on the same cells (illustrating Kedzierska et al. 2025).
import { header, figure, slider, select, stats, section, steps, math, legend, seeAlso, tip, untip, el } from '../lib/ui.js';
import { frame, cat, css } from '../lib/plot.js';
import { rng } from '../lib/data.js';
import { sourceCard, toyBadge, coreIdea, realityCheck, useIt, cite, pca } from './_shared.js';
const d3 = window.d3;

const G = 60, NT = 5, NB = 3, PER = 24, K = 15;
const TYPES = ['T cell', 'B cell', 'Monocyte', 'NK cell', 'Dendritic'];

// Gene programs. Genes 0-29 carry the immune types of "your" dataset; genes 30-49 carry other tissues' types
// (epithelial, stromal...) that dominate the pretraining corpus; 50-59 are housekeeping.
function programs() {
  const r = rng(3), base = d3.range(G).map((g) => (g >= 50 ? 3.2 : 0.4 + 0.6 * r()));
  const prog = (genes) => { const v = new Array(G).fill(0); genes.forEach((g) => { v[g] = 1.4 + 0.6 * r(); }); return v; };
  const ours = d3.range(NT).map((t) => prog(d3.range(6).map((i) => t * 6 + i)));
  const other = d3.range(4).map((t) => prog(d3.range(5).map((i) => 30 + t * 5 + i)));
  return { base, ours, other };
}
const P = programs();
function sampleCell(r, prog, batchShift, depth) {
  const size = depth * Math.exp(r.normal(0, 0.45));
  return d3.range(G).map((g) => r.poisson(size * Math.exp(P.base[g] + prog[g] + batchShift[g])));
}
function batchShifts(strength) {
  const r = rng(17);
  return d3.range(NB).map(() => d3.range(G).map((g) => (g < 50 && r() < 0.35 ? strength * r.normal(0, 0.8) : 0)));
}
function yourData(strength) {
  const r = rng(29), sh = batchShifts(strength), cells = [];
  const depth = [1, 2.2, 0.6]; // batches also differ in sequencing depth (scaled by the batch-strength slider)
  for (let b = 0; b < NB; b++) for (let t = 0; t < NT; t++) for (let i = 0; i < PER; i++) cells.push({ b, t, x: sampleCell(r, P.ours[t], sh[b], Math.pow(depth[b], strength)) });
  return cells;
}
function pretrainCorpus(overlap) {
  const r = rng(41), zero = new Array(G).fill(0), cells = [];
  for (let i = 0; i < 240; i++) {
    const mine = r() < overlap;
    const prog = mine ? P.ours[r.int(NT)] : P.other[r.int(4)];
    cells.push(sampleCell(r, prog, zero, 0.4 + 2.5 * r())); // a corpus with wildly varying depth
  }
  return cells;
}
const log1p = (x) => x.map((v) => Math.log1p(v));

// --- the two pipelines ---
function hvgPca(cells) {
  const X = cells.map((c) => { const tot = d3.sum(c.x); return c.x.map((v) => Math.log1p((v / tot) * 1e4)); });
  const mu = d3.range(G).map((g) => d3.mean(X, (r) => r[g])), sdv = d3.range(G).map((g) => d3.deviation(X, (r) => r[g]) || 1);
  const hvg = d3.range(G).sort((a, b) => sdv[b] / (mu[b] + 0.5) - sdv[a] / (mu[a] + 0.5)).slice(0, 30);
  const Z = X.map((r) => hvg.map((g) => (r[g] - mu[g]) / sdv[g]));
  return pca(Z, 8, 30).scores;
}
function frozenEncoder(overlap) {
  // "pretraining": PCA on raw log1p counts of a corpus that mostly contains other cell types; no depth normalisation
  const corpus = pretrainCorpus(overlap).map(log1p);
  const enc = pca(corpus, 8, 30);
  const sc = d3.range(8).map((k) => d3.deviation(enc.scores, (s) => s[k]) || 1);
  return (x) => { const v = log1p(x); return enc.axes.map((a, k) => Math.tanh(d3.sum(a, (w, g) => w * (v[g] - enc.mean[g])) / sc[k])); };
}
function knnMetrics(E, cells) {
  const n = E.length, D = E[0].length; let acc = 0, mix = 0;
  for (let i = 0; i < n; i++) {
    const d = new Float64Array(n);
    for (let j = 0; j < n; j++) { let s = 0; for (let k = 0; k < D; k++) { const u = E[i][k] - E[j][k]; s += u * u; } d[j] = j === i ? Infinity : s; }
    const nn = d3.range(n).sort((a, b) => d[a] - d[b]).slice(0, K);
    const votes = new Array(NT).fill(0); nn.forEach((j) => { votes[cells[j].t]++; });
    if (d3.maxIndex(votes) === cells[i].t) acc++;
    mix += nn.filter((j) => cells[j].b !== cells[i].b).length / K;
  }
  return { acc: acc / n, mix: (mix / n) / ((NB - 1) / NB) }; // mixing: 1 = neighbours drawn from batches at random
}

export default async function render(main) {
  header(main, 'Do foundation embeddings beat PCA?',
    'Single-cell foundation models promise embeddings you can use <b>zero-shot</b>, with no training on your data. An independent benchmark found that, used this way, they were <b>worse than highly-variable genes + PCA</b>. This page shows the kind of mechanism that produces that result.');
  sourceCard(main, {
    org: 'Kedzierska et al. (independent evaluation of Geneformer and scGPT)',
    when: 'Genome Biology 2025',
    links: ['kedzierska', 'geneformer', 'scgpt'],
    claim: 'Zero-shot Geneformer and scGPT embeddings underperform simple baselines (HVG + PCA, scVI, Harmony) at separating cell types and at batch integration.',
    selfReported: false,
    status: 'independent benchmark',
  });
  coreIdea(main, `<p>"Zero-shot" means you take the pretrained model, push your cells through it and use the output vectors directly: no fine-tuning. That only works if the directions the model learned during pretraining are the directions that matter in <i>your</i> tissue, and if the model has learned to ignore technical variation (depth, batch). Neither is guaranteed. A 30-second HVG + PCA is fitted to your data, so it automatically finds the axes your cells vary along.</p>
  <p><b>Key ML ingredient:</b> <a href="#/concepts/foundation-models">transfer of a frozen representation</a>, evaluated with a kNN classifier (does the embedding keep cell types apart?) and a batch-mixing score (does it remove batch?).</p>`);

  let overlap = 0.15, strength = 1, colorBy = 'type';
  const fig = figure(main, 'Same cells, two embeddings',
    `${NT} immune types × ${NB} batches × ${PER} cells, ${G} genes, simulated as Poisson counts with batch-specific gene shifts and sequencing depths. <b>Left:</b> depth-normalise, log, top 30 HVGs, scale, PCA (8 PCs). <b>Right:</b> a frozen "foundation" encoder: a nonlinear projection learned on a different pretraining corpus (raw log counts, mostly non-immune cell types), applied to your cells unchanged. Both are shown on their first two components; metrics use all 8 dimensions.`);
  toyBadge(fig, 'The frozen encoder is constructed to illustrate the published finding, not to prove it.');
  slider(fig.controls, { label: 'Pretraining corpus overlap with your cell types', min: 0, max: 1, step: 0.05, value: overlap, fmt: (v) => `${Math.round(v * 100)}%`, onInput: (v) => { overlap = v; draw(); } });
  slider(fig.controls, { label: 'Batch effect strength', min: 0, max: 2, step: 0.1, value: strength, onInput: (v) => { strength = v; draw(); } });
  select(fig.controls, { label: 'Colour by', value: colorBy, options: [['type', 'cell type'], ['batch', 'batch']], onChange: (v) => { colorBy = v; draw(); } });
  const row = el('div', { class: 'fig-row' }); fig.body.append(row);
  const L = el('div'), R = el('div'); row.append(L, R);
  const legDiv = el('div'); fig.root.append(legDiv);
  const table = el('div', { style: 'margin-top:8px' }); fig.root.append(table);

  function panel(div, E, cells, title) {
    div.innerHTML = '';
    div.append(el('div', { style: 'font-size:13px;font-weight:600;margin-bottom:2px' }, title));
    const xs = E.map((e) => e[0]), ys = E.map((e) => e[1]);
    const pad = (a) => { const [lo, hi] = d3.extent(a), p = (hi - lo) * 0.06 || 1; return [lo - p, hi + p]; };
    const F = frame(div, { width: 460, height: 360, x: pad(xs), y: pad(ys), xLabel: 'component 1', yLabel: 'component 2' });
    F.layer('p').selectAll('circle').data(cells).join('circle').attr('cx', (c, i) => F.x(xs[i])).attr('cy', (c, i) => F.y(ys[i])).attr('r', 4)
      .attr('fill', (c) => cat(colorBy === 'type' ? c.t : c.b)).attr('fill-opacity', 0.8).attr('stroke', css('--surface')).attr('stroke-width', 1)
      .on('mousemove', (e, c) => tip(e, `${TYPES[c.t]} · batch ${'ABC'[c.b]}<br>total UMIs ${d3.sum(c.x)}`)).on('mouseleave', untip);
  }
  let cache = {};
  function draw() {
    const kS = strength.toFixed(1), kO = overlap.toFixed(2);
    if (!cache.cells || cache.kS !== kS) { cache.cells = yourData(strength); cache.hvg = hvgPca(cache.cells); cache.mH = knnMetrics(cache.hvg, cache.cells); cache.kS = kS; cache.kO = null; }
    if (cache.kO !== kO) { const enc = frozenEncoder(overlap); cache.fm = cache.cells.map((c) => enc(c.x)); cache.mF = knnMetrics(cache.fm, cache.cells); cache.kO = kO; }
    const { cells, hvg, fm, mH, mF } = cache;
    panel(L, hvg, cells, 'HVG + PCA (fitted on your data)');
    panel(R, fm, cells, 'Frozen "foundation" encoder (zero-shot)');
    legDiv.innerHTML = '';
    legend(legDiv, colorBy === 'type' ? TYPES.map((t, i) => [t, cat(i)]) : ['batch A', 'batch B (deeper)', 'batch C (shallower)'].map((t, i) => [t, cat(i)]));
    const pc = (v) => `${(100 * v).toFixed(0)}%`;
    table.innerHTML = `<table class="simple" style="max-width:640px"><thead><tr><th></th><th>kNN cell-type accuracy (k=${K}) ↑</th><th>Batch mixing (1 = random) ↑</th></tr></thead><tbody>
      <tr><td>HVG + PCA</td><td><b>${pc(mH.acc)}</b></td><td><b>${mH.mix.toFixed(2)}</b></td></tr>
      <tr><td>Frozen encoder, zero-shot</td><td><b>${pc(mF.acc)}</b></td><td><b>${mF.mix.toFixed(2)}</b></td></tr></tbody></table>`;
  }
  draw();

  section(main, 'What to notice', `<ul>
    <li>With low <b>overlap</b> (the default 15%), the frozen encoder's axes were learned on mostly other cell types. Your immune types blur together on the right, and its kNN accuracy is below HVG + PCA's.</li>
    <li>Colour by <b>batch</b>: the encoder never learned to divide by sequencing depth, so the deep batch B and shallow batch C pull apart. HVG + PCA starts with depth normalisation, so that axis is gone before PCA sees the data.</li>
    <li>Raise overlap towards 100%: the encoder improves, because the pretraining now covered your biology. This is the honest argument for foundation models: <b>big, diverse pretraining corpora</b>. It is also why results vary by tissue.</li>
    <li>Set batch strength to 0: both look better, and the gap narrows. The difference shows up most where real data is hardest.</li>
    <li>Neither embedding is "batch-corrected". For integration, both baselines in the paper (scVI, Harmony) are explicitly trained to remove batch.</li>
  </ul>`);
  steps(main, 'How it works (the benchmark protocol)', [
    'Take datasets with trusted cell-type labels and known batches; never fine-tune the foundation model on them (zero-shot).',
    'Embed the cells with each method: foundation model outputs, HVG + PCA, scVI, Harmony.',
    'Score cell-type separation (e.g. kNN accuracy, silhouette, clustering agreement with labels).',
    'Score batch integration (how well batches mix within each cell type).',
    'Compare. Kedzierska et al. found the zero-shot foundation embeddings behind the simple baselines on these tasks.',
  ]);
  math(main, 'kNN accuracy = mean_i [ majority label of i\'s k nearest neighbours == label_i ]\nbatch mixing = mean_i (fraction of neighbours from other batches) / ((B − 1) / B)');
  realityCheck(main, `<ul>
    <li>The finding is about <b>zero-shot</b> use (${cite('kedzierska', 'Kedzierska et al. 2025')}). Fine-tuned foundation models can do better on specific tasks; the developers report such gains, and those are self-reported.</li>
    <li>Similar "simple baselines are hard to beat" results appear for perturbation prediction (${cite('ahlmann', 'Ahlmann-Eltze et al. 2025')}, ${cite('perteval', 'PertEval-scFM')}, ${cite('vcbench', 'VCBench 2026')}).</li>
    <li>Our toy builds in two failure modes on purpose (domain mismatch and no depth normalisation). Real models normalise in their own ways (Geneformer's rank-value encoding, scGPT's value binning), so their failure modes differ in detail.</li>
  </ul>`);
  useIt(main, `<ul>
    <li>Before adopting any foundation embedding, run this exact comparison on <b>your</b> data: same kNN / silhouette / batch metrics, against HVG + PCA and scVI.</li>
    <li>If you need integration, use a method trained for it (scVI, Harmony) or fine-tune; don't expect a frozen embedding to remove your batches.</li>
    <li>Check overlap: is your tissue, disease or technology (e.g. spatial panels, nuclei) well represented in the pretraining corpus?</li>
  </ul>`);
  seeAlso(main, [['concepts', 'pca', 'PCA'], ['concepts', 'knn', 'k-nearest neighbours'], ['concepts', 'foundation-models', 'Foundation models'], ['concepts', 'embeddings', 'Embeddings'],
    ['omics', 'sc-embedding', 'Normalise, HVGs, PCA'], ['omics', 'sc-integration', 'Batch integration'], ['frontier', 'geneformer', 'Geneformer'], ['frontier', 'perturbation-arena', 'Perturbation baselines arena']]);
}
