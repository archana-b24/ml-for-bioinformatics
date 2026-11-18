// Novae / Nicheformer: niches = message passing over the tissue graph, then self-supervised prototypes.
import { header, figure, slider, button, stats, section, steps, math, legend, seeAlso, every, tip, untip, el } from '../../lib/ui.js';
import { frame, cat, css, ink2 } from '../../lib/plot.js';
import { rng } from '../../lib/data.js';
import { sourceCard, toyBadge, coreIdea, realityCheck, useIt, cite } from '../../frontier/_shared.js';
const d3 = window.d3;

const TYPES = ['Tumour', 'CAF', 'T cell', 'B cell', 'Macrophage'];
const NICHES = ['Tumour core', 'Invasive margin', 'Lymphoid aggregate', 'Stroma'];
// cell-type composition of each niche (rows = niches, cols = TYPES)
const COMP = [[0.8, 0.05, 0.05, 0, 0.1], [0.3, 0.2, 0.25, 0, 0.25], [0, 0.05, 0.45, 0.45, 0.05], [0.02, 0.7, 0.1, 0.03, 0.15]];
const GENES = [...TYPES.map((t) => t + ' marker'), 'hypoxia', 'IFN response', 'ECM remodelling'];
const CORE = { x: 0.35, y: 0.6, r: 0.22 }, MARGIN_R = 0.32, TLS = { x: 0.78, y: 0.25, r: 0.13 };
const KNN = 6, MAXHOPS = 4;

function makeTissue(seed = 11) {
  const r = rng(seed), G = 18, cells = [];
  for (let i = 0; i < G; i++) for (let j = 0; j < G; j++) {
    const x = (i + 0.5 + (r() - 0.5) * 0.7) / G, y = (j + 0.5 + (r() - 0.5) * 0.7) / G;
    const d1 = Math.hypot(x - CORE.x, y - CORE.y), dT = Math.hypot(x - TLS.x, y - TLS.y);
    const niche = d1 < CORE.r ? 0 : d1 < MARGIN_R ? 1 : dT < TLS.r ? 2 : 3;
    let u = r(), type = 0; for (; type < 4; type++) { u -= COMP[niche][type]; if (u < 0) break; }
    const e = TYPES.map((_, k) => (k === type ? 1.2 : 0) + r.normal(0, 0.3));
    e.push((niche === 0 ? 0.9 : 0) + r.normal(0, 0.3));
    e.push((niche === 2 ? 0.8 : niche === 1 ? 0.6 : 0) + r.normal(0, 0.3));
    e.push((niche === 1 ? 0.7 : niche === 3 ? 0.25 : 0) + r.normal(0, 0.3));
    cells.push({ x, y, niche, type, e });
  }
  // k-nearest-neighbour spatial graph
  cells.forEach((c, i) => {
    c.nb = cells.map((d, j) => [j, (c.x - d.x) ** 2 + (c.y - d.y) ** 2]).filter(([j]) => j !== i).sort((a, b) => a[1] - b[1]).slice(0, KNN).map(([j]) => j);
  });
  // message passing: h(l+1)_i = mean of h(l) over {i} ∪ neighbours(i)
  const H = [cells.map((c) => c.e)];
  for (let l = 1; l <= MAXHOPS; l++) {
    const prev = H[l - 1];
    H.push(cells.map((c, i) => { const s = prev[i].slice(); c.nb.forEach((j) => prev[j].forEach((v, g) => { s[g] += v; })); return s.map((v) => v / (KNN + 1)); }));
  }
  return { cells, H };
}

const d2 = (a, b) => { let s = 0; for (let g = 0; g < a.length; g++) s += (a[g] - b[g]) ** 2; return s; };
function kmeans(X, K, seed = 3, restarts = 4) {
  let best = null;
  for (let rs = 0; rs < restarts; rs++) {
    const r = rng(seed + rs * 101);
    const C = [X[r.int(X.length)].slice()];
    while (C.length < K) { // k-means++ seeding
      const w = X.map((x) => Math.min(...C.map((c) => d2(x, c)))); const tot = w.reduce((a, b) => a + b, 0);
      let u = r() * tot, i = 0; for (; i < X.length - 1; i++) { u -= w[i]; if (u <= 0) break; }
      C.push(X[i].slice());
    }
    let lab = new Array(X.length).fill(0);
    for (let it = 0; it < 30; it++) {
      lab = X.map((x) => { let bk = 0, bd = Infinity; C.forEach((c, k) => { const d = d2(x, c); if (d < bd) { bd = d; bk = k; } }); return bk; });
      C.forEach((c, k) => { const m = X.filter((_, i) => lab[i] === k); if (m.length) for (let g = 0; g < c.length; g++) c[g] = m.reduce((s, v) => s + v[g], 0) / m.length; });
    }
    const inertia = X.reduce((s, x, i) => s + d2(x, C[lab[i]]), 0);
    if (!best || inertia < best.inertia) best = { lab, C, inertia };
  }
  return best;
}
function ari(a, b) {
  const n = a.length, ka = Math.max(...a) + 1, kb = Math.max(...b) + 1;
  const T = Array.from({ length: ka }, () => new Array(kb).fill(0));
  a.forEach((v, i) => { T[v][b[i]] += 1; });
  const c2 = (x) => x * (x - 1) / 2;
  const sij = T.flat().reduce((s, v) => s + c2(v), 0);
  const sa = T.reduce((s, row) => s + c2(row.reduce((p, q) => p + q, 0)), 0);
  const sb = d3.range(kb).reduce((s, j) => s + c2(T.reduce((p, row) => p + row[j], 0)), 0);
  const exp = sa * sb / c2(n), mx = (sa + sb) / 2;
  return mx === exp ? 0 : (sij - exp) / (mx - exp);
}
/** Relabel clusters so cluster colours line up with the closest true niche (greedy on the contingency table). */
function align(lab, truth, K) {
  const T = Array.from({ length: K }, () => new Array(4).fill(0));
  lab.forEach((k, i) => { T[k][truth[i]] += 1; });
  const map = new Array(K).fill(-1), used = new Set();
  for (let it = 0; it < Math.min(K, 4); it++) {
    let bk = -1, bn = -1, bv = -1;
    T.forEach((row, k) => { if (map[k] >= 0) return; row.forEach((v, nn) => { if (!used.has(nn) && v > bv) { bv = v; bk = k; bn = nn; } }); });
    map[bk] = bn; used.add(bn);
  }
  let extra = 4; map.forEach((m, k) => { if (m < 0) map[k] = extra++; });
  return lab.map((k) => map[k]);
}

export default async function render(main) {
  header(main, 'Novae & Nicheformer: niches',
    'A cell type says what a cell <i>is</i>; a niche says what <i>neighbourhood</i> it lives in. Spatial foundation models find niches by <b>letting each cell absorb its neighbours\' expression over the tissue graph</b>, then grouping cells by those neighbourhood summaries, without labels.');
  sourceCard(main, {
    org: 'Novae: Nature Methods 2025 (Python package <code>novae</code>). Nicheformer: Theis lab (Helmholtz Munich), Nature Methods 2025.',
    when: '2025',
    links: ['novae', 'novaePypi', 'nicheformer'],
    claim: 'Novae: graph attention over cell neighbourhoods plus SwAV-style self-supervision gives zero-shot spatial domains across tissues and panels. Nicheformer: a transformer pretrained on SpatialCorpus-110M learns spatial context and transfers spatial labels onto dissociated scRNA-seq cells.',
  });
  coreIdea(main, `<p>Take a spatial dataset (Xenium, MERFISH, CosMx…). Build a <b>graph</b>: each cell is a node, linked to its k nearest neighbours in space. A <a href="#/archive/topics/gnn">graph neural network</a> then does <b>message passing</b>: each cell's representation is replaced by a (weighted) average of itself and its neighbours. After one hop a cell "knows" its immediate neighbours; after two hops, their neighbours too. A T cell inside a tumour and a T cell inside a lymphoid aggregate start identical, but after a few hops they carry very different neighbourhood signatures.</p>
  <p>The second ingredient is <b>self-supervised clustering</b>. Novae uses SwAV: learn a small set of <b>prototypes</b> (cluster centres) and softly assign every cell to them, training so that two augmented views of the same neighbourhood land on the same prototype. The prototypes become the niches, and because they are learned across many slides they can be reused zero-shot. Nicheformer goes the other way: it learns spatial context from spatial data, then predicts that context for dissociated cells that have no coordinates.</p>
  <p><b>Key ML ingredient:</b> graph message passing + prototype (cluster-centre) self-supervision.</p>`);

  const { cells, H } = makeTissue();
  const truth = cells.map((c) => c.niche);
  const types = cells.map((c) => c.type);

  const fig = figure(main, 'Mini-replication: message passing turns cell types into niches',
    `${cells.length} synthetic cells in four planted niches (dashed outlines). Each cell expresses its type marker plus three niche-dependent state genes, all noisy. Message passing averages each cell with its ${KNN} spatial neighbours per hop; then k-means finds K prototypes on the smoothed features. Point opacity = the cell's soft-assignment confidence to its prototype (SwAV-style softmax). Hover a cell for details.`);
  toyBadge(fig, 'Mean aggregation stands in for Novae\'s learned graph attention; k-means stands in for SwAV training.');
  let hops = 0, K = 4, playing = null;
  const sH = slider(fig.controls, { label: 'Message-passing hops', min: 0, max: MAXHOPS, step: 1, value: 0, onInput: (v) => { hops = v; update(); } });
  slider(fig.controls, { label: 'Prototypes K', min: 2, max: 6, step: 1, value: 4, onInput: (v) => { K = v; update(); } });
  const playBtn = button(fig.controls, 'Play hops 0→4', () => {
    if (playing) { playing(); playing = null; }
    hops = 0; sH.set(0); update();
    playing = every(900, () => { if (hops >= MAXHOPS) { playing(); playing = null; return; } hops += 1; sH.set(hops); update(); });
  }, true);
  playBtn.setAttribute('aria-label', 'Animate message passing from 0 to 4 hops');
  const readout = stats(fig.root);

  const row = el('div', { class: 'fig-row' }); fig.body.append(row);
  const leftBox = el('div'), rightBox = el('div'); row.append(leftBox, rightBox);
  leftBox.append(el('div', { class: 'fig-caption', style: 'margin:0 0 4px' }, 'What each cell is (cell type)'));
  rightBox.append(el('div', { class: 'fig-caption', style: 'margin:0 0 4px' }, 'Where each cell lives (learned niche)'));
  const mk = (box) => {
    const F = frame(box, { width: 440, height: 440, x: [0, 1], y: [0, 1], axes: false, grid: false, margin: { top: 6, right: 6, bottom: 6, left: 6 } });
    const px = (r) => F.x(r) - F.x(0);
    const ol = F.layer('outline');
    [[CORE, CORE.r], [CORE, MARGIN_R], [TLS, TLS.r]].forEach(([c, rr]) => ol.append('circle').attr('cx', F.x(c.x)).attr('cy', F.y(c.y)).attr('r', px(rr))
      .attr('fill', 'none').attr('stroke', ink2()).attr('stroke-dasharray', '4 4').attr('stroke-width', 1));
    return F;
  };
  const FL = mk(leftBox), FR = mk(rightBox);
  legend(leftBox, TYPES.map((t, i) => [t, cat(i)]));
  const rightLegend = el('div'); rightBox.append(rightLegend);

  const hover = (e, i, lab, conf) => {
    const c = cells[i];
    tip(e, `<b>${TYPES[c.type]}</b><br>true niche: ${NICHES[c.niche]}<br>learned: ${lab != null ? (lab < 4 ? 'prototype ≈ ' + NICHES[lab] : 'extra prototype ' + (lab + 1)) : ''}<br>confidence: ${conf != null ? conf.toFixed(2) : ''}`);
  };
  FL.layer('pts').selectAll('circle').data(cells).join('circle').attr('cx', (d) => FL.x(d.x)).attr('cy', (d) => FL.y(d.y)).attr('r', 5)
    .attr('fill', (d) => cat(d.type)).attr('stroke', css('--surface')).attr('stroke-width', 1)
    .on('mousemove', (e, d) => hover(e, cells.indexOf(d))).on('mouseleave', untip);
  const ptsR = FR.layer('pts').selectAll('circle').data(cells).join('circle').attr('cx', (d) => FR.x(d.x)).attr('cy', (d) => FR.y(d.y)).attr('r', 5)
    .attr('stroke', css('--surface')).attr('stroke-width', 1);

  function update() {
    const X = H[hops];
    const km = kmeans(X, K);
    const lab = align(km.lab, truth, K);
    // SwAV-style soft assignment: softmax over prototypes of -d²/τ, τ = mean within-cluster squared distance
    const tau = km.inertia / X.length || 1;
    const conf = X.map((x) => { const s = km.C.map((c) => Math.exp(-d2(x, c) / tau)); const z = s.reduce((a, b) => a + b, 0); return Math.max(...s) / z; });
    ptsR.attr('fill', (d, i) => cat(lab[i])).attr('fill-opacity', (d, i) => 0.25 + 0.75 * conf[i])
      .on('mousemove', (e, d) => { const i = cells.indexOf(d); hover(e, i, lab[i], conf[i]); }).on('mouseleave', untip);
    rightLegend.innerHTML = '';
    legend(rightLegend, [...new Set(lab)].sort((a, b) => a - b).map((m) => [m < 4 ? '≈ ' + NICHES[m] : 'extra prototype', cat(m)]));
    // Nicheformer-style transfer: even cells = spatial reference (labels = learned niches); odd cells = "dissociated" (own expression only)
    const ref = cells.map((_, i) => i).filter((i) => i % 2 === 0), qry = cells.map((_, i) => i).filter((i) => i % 2 === 1);
    const labs = [...new Set(lab)];
    const cents = labs.map((l) => { const m = ref.filter((i) => lab[i] === l); return m.length ? GENES.map((_, g) => m.reduce((s, i) => s + cells[i].e[g], 0) / m.length) : null; });
    let correct = 0; qry.forEach((i) => { let bl = -1, bd = Infinity; cents.forEach((c, k) => { if (!c) return; const d = d2(cells[i].e, c); if (d < bd) { bd = d; bl = labs[k]; } }); if (bl === lab[i]) correct++; });
    const counts = labs.map((l) => qry.filter((i) => lab[i] === l).length);
    readout.set({
      hops,
      'ARI vs true niches': ari(lab, truth).toFixed(2),
      'ARI vs cell types': ari(lab, types).toFixed(2),
      'mean confidence': d3.mean(conf).toFixed(2),
      'transfer to "dissociated" cells (own expression only)': `${(100 * correct / qry.length).toFixed(0)}% (majority class ${(100 * Math.max(...counts) / qry.length).toFixed(0)}%)`,
    });
  }
  update();

  section(main, 'What to notice', `<ul>
    <li>At <b>0 hops</b> the clusters follow <b>cell types</b>, not niches: compare the two ARI readouts (ARI = adjusted Rand index; 1 = identical partitions, 0 = chance). The T cells of the margin and of the lymphoid aggregate are lumped together.</li>
    <li>Press <b>Play</b> (K = 4). After one hop, ARI vs cell types collapses from 0.93 to 0.30 and ARI vs niches jumps from 0.41 to 0.69; the right panel snaps to the dashed outlines. Two hops (each cell has now heard from its neighbours' neighbours) give the best match, 0.71.</li>
    <li>Keep going: ARI vs niches <b>falls</b> to 0.62 at 3 hops and 0.56 at 4. Boundaries blur and the thin invasive margin bleeds into its neighbours; too much smoothing erases small niches. Hop count is the "resolution" knob of a niche model.</li>
    <li>Set <b>K = 6</b>: extra prototypes split real niches, ARI vs niches drops to 0.40 at 2 hops, and mean confidence falls (look at the faded cells at boundaries). Prototype count is a choice, not a discovery.</li>
    <li>The <b>transfer</b> readout is the Nicheformer idea: labels learned with spatial context are predicted for cells from their own expression only. At 2 hops it reaches 70% against a 55% majority-class baseline. (At 0 hops it is trivially 100%, because the labels were computed from the same features.) It works at all because the toy's state genes (hypoxia, IFN response, ECM remodelling) depend on the niche. If a niche left no trace in a cell's own transcriptome, no model could transfer it.</li>
  </ul>`);
  steps(main, 'How it works', [
    'Build a spatial graph: connect each cell to its k nearest neighbours (or all cells within a radius).',
    'Start from each cell\'s own features (normalised expression, or an embedding from a single-cell model).',
    '<b>Message passing</b>: replace each cell\'s vector with an aggregate of itself and its neighbours. Novae learns <a href="#/archive/topics/attention">attention</a> weights for this aggregate; the toy uses a plain mean. Repeat for L hops.',
    '<b>Prototype learning</b>: keep K learnable prototype vectors. SwAV assigns each neighbourhood softly to prototypes and trains so that two augmented views (e.g. a perturbed copy of the neighbourhood) agree on the assignment. The toy uses k-means plus a softmax for the soft assignment.',
    'Read niches off as the most likely prototype per cell. Because prototypes are shared across slides, a new slide can be labelled without retraining ("zero-shot").',
    'Nicheformer-style transfer: train a model on spatial data to predict a cell\'s spatial context, then apply it to dissociated scRNA-seq cells.',
  ]);
  math(main, 'h_i(l+1) = Σ_j∈{i}∪N(i)  a_ij · h_j(l)        (toy: a_ij = 1/(k+1);  Novae: a_ij = learned attention)\nq_ik = softmax_k( −‖h_i − c_k‖² / τ )                (soft assignment to prototype c_k)');
  realityCheck(main, `<p>Both papers report strong results on their own benchmarks; those numbers are <b>self-reported</b>. Our brief did not find an independent, head-to-head benchmark of spatial foundation models comparable to the critiques of single-cell models (${cite('kedzierska', 'Kedzierska et al. 2025')} showed zero-shot scRNA foundation-model embeddings losing to HVG+PCA, scVI and Harmony). Until that exists, treat "zero-shot spatial domains" as a hypothesis.</p>
  <p>The toy shows why a strong baseline is cheap: averaging expression over the spatial graph and clustering already recovers planted niches. A foundation model earns its keep only if it beats that on your tissue, or if its prototypes transfer consistently across slides, panels and patients.</p>`);
  useIt(main, `<ul>
    <li><b>Baseline first:</b> on a Xenium/CosMx/MERFISH object, compute neighbourhood-averaged expression (or cell-type composition) over a kNN graph and cluster it. Compare any foundation-model niches against this with ARI and by eye.</li>
    <li><b>Sweep the resolution:</b> number of neighbours × hops sets niche size. Report results across a few settings; thin structures such as invasive margins vanish with heavy smoothing.</li>
    <li><b>Novae</b> (<code>pip install novae</code>) is worth trying when you have many slides from different panels and want one shared set of domains across them.</li>
    <li><b>Transfer to scRNA-seq</b> (Nicheformer-style): only trust predicted niche labels on dissociated cells if a held-out spatial slide shows the label is predictable from expression alone.</li>
  </ul>`);
  seeAlso(main, [['concepts', 'gnn', 'Graph neural networks'], ['concepts', 'attention', 'Attention'], ['concepts', 'contrastive', 'Contrastive learning'], ['concepts', 'kmeans', 'k-means'], ['concepts', 'foundation-models', 'Foundation models'],
    ['omics', 'sp-data', 'Spatial data & neighbourhoods'], ['omics', 'sp-domains', 'Spatial domains'], ['omics', 'sp-niches', 'Niches & communication'], ['frontier', 'masked-cells', 'CellPLM: impute from neighbours']]);
}
