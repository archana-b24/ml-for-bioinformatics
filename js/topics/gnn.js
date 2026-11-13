// Graph neural networks: message passing on a spatial cell graph, over-smoothing, GCN vs GAT weights.
import { header, figure, slider, select, button, stats, section, steps, callout, math, seeAlso, animate, legend, tip, untip, el } from '../lib/ui.js';
import { frame, cat, css, ink, ink2, muted, diverging } from '../lib/plot.js';
import { rng, mean, sd } from '../lib/data.js';
const d3 = window.d3;

const MAXL = 150;
const DOMAINS = ['tumour', 'stroma', 'immune'];
const DOMAIN_MEAN = [1, 0, -1];
const SYMBOLS = [d3.symbolCircle, d3.symbolSquare, d3.symbolTriangle];

// ---------- a seeded "tissue": jittered grid of cells, 3 spatial domains, Delaunay neighbour graph ----------
function makeTissue() {
  const r = rng(31), cells = [];
  const nx = 9, ny = 6, sp = 1 / (nx - 1);
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const x = i * sp + (r() - 0.5) * 0.55 * sp + (j % 2 ? sp / 2 : 0) - sp / 4;
    const y = j * sp * 0.95 + (r() - 0.5) * 0.55 * sp;
    const b = x + 0.08 * Math.sin(y * 7);
    const dom = b < 0.36 ? 0 : b < 0.64 ? 1 : 2;
    cells.push({ id: cells.length, x, y, dom, h0: DOMAIN_MEAN[dom] + r.normal(0, 0.6) });
  }
  const del = d3.Delaunay.from(cells, (d) => d.x, (d) => d.y);
  const nbrs = cells.map(() => new Set());
  const edges = [];
  for (let t = 0; t < del.triangles.length; t += 3) {
    const tri = [del.triangles[t], del.triangles[t + 1], del.triangles[t + 2]];
    for (let k = 0; k < 3; k++) {
      const a = tri[k], b = tri[(k + 1) % 3];
      if (nbrs[a].has(b)) continue;
      if (Math.hypot(cells[a].x - cells[b].x, cells[a].y - cells[b].y) > 1.6 * sp) continue; // drop long hull edges
      nbrs[a].add(b); nbrs[b].add(a); edges.push([a, b]);
    }
  }
  // fix duplicated edges where (b,a) seen first
  const seen = new Set(), uniq = [];
  edges.forEach(([a, b]) => { const k = Math.min(a, b) + ':' + Math.max(a, b); if (!seen.has(k)) { seen.add(k); uniq.push([Math.min(a, b), Math.max(a, b)]); } });
  return { cells, nbrs: nbrs.map((s) => [...s]), edges: uniq };
}

// Mean aggregation with a self-loop: h_i <- (h_i + sum_j h_j) / (deg_i + 1). Returns H[0..MAXL].
function propagate(cells, nbrs) {
  const H = [Float64Array.from(cells, (c) => c.h0)];
  for (let L = 1; L <= MAXL; L++) {
    const prev = H[L - 1], next = new Float64Array(cells.length);
    for (let i = 0; i < cells.length; i++) {
      let s = prev[i];
      for (const j of nbrs[i]) s += prev[j];
      next[i] = s / (nbrs[i].length + 1);
    }
    H.push(next);
  }
  return H;
}

function summary(cells, h) {
  const byDom = [0, 1, 2].map((d) => cells.filter((c) => c.dom === d).map((c) => h[c.id]));
  const gap = mean(byDom[0]) - mean(byDom[2]);
  // pooled within-domain sd
  let ss = 0, n = 0;
  byDom.forEach((v) => { const m = mean(v); v.forEach((x) => { ss += (x - m) ** 2; n++; }); });
  const within = Math.sqrt(ss / (n - 3));
  // nearest-domain-mean assignment (how separable the three domains still are)
  const mus = byDom.map(mean);
  let ok = 0;
  cells.forEach((c) => { const v = h[c.id]; let best = 0; mus.forEach((m, k) => { if (Math.abs(v - m) < Math.abs(v - mus[best])) best = k; }); if (best === c.dom) ok++; });
  return { gap, within, acc: ok / cells.length, spread: d3.max(h) - d3.min(h) };
}

function hops(nbrs, src, L) {
  const dist = new Array(nbrs.length).fill(Infinity); dist[src] = 0;
  let frontier = [src];
  for (let k = 1; k <= L && frontier.length; k++) {
    const nf = [];
    frontier.forEach((i) => nbrs[i].forEach((j) => { if (dist[j] === Infinity) { dist[j] = k; nf.push(j); } }));
    frontier = nf;
  }
  return dist;
}

export default async function render(main) {
  header(main, 'Graph neural networks',
    'Many biological objects are <b>graphs</b>: cells next to cells in tissue, genes linked in pathways, drugs linked to targets. A GNN updates each node by <b>passing messages</b> from its neighbours, so after L layers every node has "heard" from everything within L hops.');

  // =================== Figure 1: message passing on a tissue graph ===================
  const { cells, nbrs, edges } = makeTissue();
  const H = propagate(cells, nbrs);
  const S = H.map((h) => summary(cells, h));
  const degs = nbrs.map((n) => n.length);

  const fig = figure(main, 'Message passing on a spatial cell graph',
    '');
  let L = 0, sel = null, playing = null, pulse = null;
  const layerS = slider(fig.controls, { label: 'Layers (rounds) L', min: 0, max: MAXL, step: 1, value: 0, onInput: (v) => { stopPlay(); L = v; paint(); } });
  button(fig.controls, 'Pass one round', () => { stopPlay(); stepOnce(); }, true);
  button(fig.controls, 'Play', () => play());
  button(fig.controls, 'Reset', () => { stopPlay(); L = 0; layerS.set(0); paint(); });
  const readout = stats(fig.root);
  fig.root.insertBefore(readout.root, fig.body);

  const row = el('div', { class: 'fig-row' }); fig.body.append(row);
  const left = el('div'), right = el('div'); row.append(left, right);

  const pad = 0.06;
  const G = frame(left, { width: 520, height: 400, x: [-pad, 1 + pad], y: [-pad, 5 / 8 * 0.95 + pad], axes: false, grid: false, margin: { top: 8, right: 8, bottom: 8, left: 8 } });
  const color = diverging(1.6);
  const eL = G.layer('edges'), mL = G.layer('msgs'), nL = G.layer('nodes');
  eL.selectAll('line').data(edges).join('line')
    .attr('x1', (d) => G.x(cells[d[0]].x)).attr('y1', (d) => G.y(cells[d[0]].y)).attr('x2', (d) => G.x(cells[d[1]].x)).attr('y2', (d) => G.y(cells[d[1]].y))
    .attr('stroke', css('--axis')).attr('stroke-width', 1.2);
  const nodes = nL.selectAll('path').data(cells).join('path')
    .attr('transform', (d) => `translate(${G.x(d.x)},${G.y(d.y)})`)
    .attr('d', (d) => d3.symbol(SYMBOLS[d.dom], d.dom === 2 ? 190 : 150)())
    .attr('stroke', css('--surface')).attr('stroke-width', 1.5).style('cursor', 'pointer')
    .on('mousemove', (e, d) => tip(e, `<b>cell ${d.id}</b> · ${DOMAINS[d.dom]}<br>score at layer ${L}: <b>${H[L][d.id].toFixed(3)}</b><br>start value: ${d.h0.toFixed(2)} · neighbours: ${degs[d.id]}`))
    .on('mouseleave', untip)
    .on('click', (e, d) => { sel = sel === d.id ? null : d.id; paint(); });

  // Right: over-smoothing curves
  const C = frame(right, { width: 460, height: 400, x: [0, MAXL], xScale: d3.scaleSymlog().constant(2), y: [0.002, 3], yScale: d3.scaleLog(), xLabel: 'layers L (compressed scale)', yLabel: 'value (log scale)' });
  C.g.select('.axis.x').call(d3.axisBottom(C.x).tickValues([0, 1, 2, 5, 10, 20, 50, 100, 150]));
  C.g.select('.axis.y').call(d3.axisLeft(C.y).ticks(4, '~g'));
  const lineGen = (k) => d3.line().x((d, i) => C.x(i)).y((d) => C.y(Math.max(0.002, d[k])));
  C.layer('lines').selectAll('path').data(['gap', 'within']).join('path')
    .attr('fill', 'none').attr('stroke', (k, i) => cat(i)).attr('stroke-width', 2).attr('d', (k) => lineGen(k)(S));
  const marker = C.layer('marker').append('line').attr('y1', 0).attr('y2', C.h).attr('stroke', ink2()).attr('stroke-dasharray', '3 3');
  const dots = C.layer('dots').selectAll('circle').data(['gap', 'within']).join('circle').attr('r', 5)
    .attr('fill', (k, i) => cat(i)).attr('stroke', css('--surface')).attr('stroke-width', 1.5);
  legend(right, [['between-domain gap: mean(tumour) − mean(immune)  (signal)', cat(0)], ['within-domain spread, pooled sd  (noise)', cat(1)]]);

  // caption with numbers computed from the seeded graph
  const first = (pred) => S.findIndex(pred);
  const L10 = first((s) => s.gap < 0.1 * S[0].gap);
  const accPeak = d3.maxIndex(S, (s) => s.acc);
  fig.caption(`${cells.length} cells, ${edges.length} edges (neighbours = touching cells in a Delaunay triangulation). Shape = true domain (● tumour, ■ stroma, ▲ immune). Colour = one feature, a noisy marker score (red high, blue low). ` +
    `Each round, every cell replaces its score with the <b>average of itself and its neighbours</b>. Two rounds cut the noise sd from ${S[0].within.toFixed(2)} to ${S[2].within.toFixed(2)} while the gap only drops from ${S[0].gap.toFixed(2)} to ${S[2].gap.toFixed(2)}; ` +
    `by L = ${L10} the gap itself is below 10% of its start, and by L = ${MAXL} every cell sits within ${S[MAXL].spread.toFixed(3)} of every other. Click a cell to see its receptive field.`);

  function paint() {
    const h = H[L];
    const dist = sel == null ? null : hops(nbrs, sel, L);
    nodes.attr('fill', (d) => color(h[d.id]))
      .attr('opacity', (d) => (dist && dist[d.id] === Infinity ? 0.25 : 1))
      .attr('stroke', (d) => (d.id === sel ? ink() : css('--surface'))).attr('stroke-width', (d) => (d.id === sel ? 3 : 1.5));
    eL.selectAll('line').attr('opacity', (d) => (dist && (dist[d[0]] === Infinity || dist[d[1]] === Infinity) ? 0.2 : 1));
    marker.attr('x1', C.x(L)).attr('x2', C.x(L));
    dots.attr('cx', C.x(L)).attr('cy', (k) => C.y(Math.max(0.002, S[L][k])));
    const s = S[L];
    const o = { 'layer L': L, 'gap (signal)': s.gap.toFixed(3), 'within sd (noise)': s.within.toFixed(3), 'cells nearest own domain mean': (100 * s.acc).toFixed(0) + '%' };
    if (dist) o[`receptive field of cell ${sel}`] = dist.filter((v) => v !== Infinity).length + ' cells';
    readout.set(o);
  }

  // one round, with messages visibly travelling along every edge (both directions)
  function stepOnce(done) {
    if (L >= MAXL) { done && done(false); return; }
    if (pulse) pulse();
    const h = H[L], msgs = [];
    edges.forEach(([a, b]) => { msgs.push([a, b]); msgs.push([b, a]); });
    const m = mL.selectAll('circle').data(msgs).join('circle').attr('r', 3).attr('fill', (d) => color(h[d[0]])).attr('stroke', ink2()).attr('stroke-width', 0.5);
    let t = 0;
    pulse = animate((dt) => {
      t += dt / 420;
      const u = Math.min(1, t), e = u * u * (3 - 2 * u);
      m.attr('cx', (d) => G.x(cells[d[0]].x + e * (cells[d[1]].x - cells[d[0]].x))).attr('cy', (d) => G.y(cells[d[0]].y + e * (cells[d[1]].y - cells[d[0]].y)));
      if (u >= 1) { mL.selectAll('circle').remove(); pulse = null; L++; layerS.set(L); paint(); done && done(true); return false; }
      return true;
    });
  }
  function play() {
    stopPlay();
    let alive = true; playing = () => { alive = false; };
    const go = () => { if (alive) stepOnce((more) => { if (more && alive && L < MAXL) go(); }); };
    go();
  }
  function stopPlay() { if (playing) playing(); playing = null; if (pulse) { pulse(); pulse = null; mL.selectAll('circle').remove(); } }
  paint();

  // =================== Figure 2: GCN vs GAT neighbour weights ===================
  const fig2 = figure(main, 'Who gets listened to? GCN weights vs GAT attention',
    'A T cell (centre) aggregates its six neighbours plus itself. Each cell has a 2-number feature vector [immune marker, tumour marker]. <b>GCN</b> fixes the weights from the graph alone: 1/√(dᵢ·dⱼ), so a neighbour with many neighbours of its own (a hub) counts less. <b>GAT</b> computes the weights from the features with a small learned scoring function, then a softmax. Here the GAT scoring vector is set by the slider instead of being learned, so you can see what changing it does.');
  const TYPES = ['tumour', 'fibroblast', 'macrophage', 'T cell'];
  const BASE = [[0.1, 1.0], [0.25, 0.3], [0.8, 0.15], [1.0, 0.05]];
  const r2 = rng(7);
  const ego = [
    { type: 3, deg: 6 }, // centre (self)
    { type: 0, deg: 7 }, { type: 0, deg: 5 }, { type: 1, deg: 3 }, { type: 1, deg: 8 }, { type: 2, deg: 4 }, { type: 3, deg: 2 },
  ].map((c, i) => ({ ...c, i, f: BASE[c.type].map((v) => +(v + r2.normal(0, 0.05)).toFixed(2)), ang: i ? (i - 1) / 6 * 2 * Math.PI - Math.PI / 2 : 0 }));
  let theta = 0, sharp = 4, show = 'gat';
  const thS = slider(fig2.controls, { label: 'GAT looks for', min: 0, max: 90, step: 1, value: 0, fmt: (v) => (v < 30 ? 'immune' : v > 60 ? 'tumour' : 'mixed') + ` (${v}°)`, onInput: (v) => { theta = v; draw2(); } });
  slider(fig2.controls, { label: 'Attention sharpness', min: 0, max: 10, step: 0.5, value: 4, onInput: (v) => { sharp = v; draw2(); } });
  select(fig2.controls, { label: 'Edge widths show', value: 'gat', options: [['gcn', 'GCN weights'], ['gat', 'GAT attention']], onChange: (v) => { show = v; draw2(); } });
  const readout2 = stats(fig2.root); fig2.root.insertBefore(readout2.root, fig2.body);
  const row2 = el('div', { class: 'fig-row' }); fig2.body.append(row2);
  const l2 = el('div'), r2d = el('div'); row2.append(l2, r2d);

  const E = frame(l2, { width: 460, height: 380, x: [-1.35, 1.35], y: [-1.2, 1.2], axes: false, grid: false, margin: { top: 6, right: 6, bottom: 6, left: 6 } });
  const pos = (c) => (c.i ? [Math.cos(c.ang), Math.sin(-c.ang)] : [0, 0]);
  const B = frame(r2d, { width: 460, height: 380, x: [0, 0.7], y: [0, 1], xLabel: 'weight in the centre cell\'s average', margin: { left: 118 }, grid: false });
  B.g.select('.axis.y').remove();
  legend(r2d, [['GCN  1/√(dᵢdⱼ), normalised', cat(4)], ['GAT  softmax(score)', cat(6)]]);

  function weights() {
    // GCN symmetric normalisation with self-loops (degrees include the self-loop), then renormalised to sum to 1 for display
    const gcnRaw = ego.map((c) => 1 / Math.sqrt((ego[0].deg + 1) * (c.deg + 1)));
    const gs = d3.sum(gcnRaw), gcn = gcnRaw.map((v) => v / gs);
    // GAT: e_ij = LeakyReLU(a_src·h_i + a_dst·h_j), W = identity; alpha = softmax_j(e_ij) over neighbours + self
    const t = theta * Math.PI / 180, aDst = [sharp * Math.cos(t), sharp * Math.sin(t)], aSrc = [0.3, 0.3];
    const src = aSrc[0] * ego[0].f[0] + aSrc[1] * ego[0].f[1];
    const e = ego.map((c) => { const z = src + aDst[0] * c.f[0] + aDst[1] * c.f[1]; return z > 0 ? z : 0.2 * z; });
    const mx = d3.max(e), ex = e.map((v) => Math.exp(v - mx)), es = d3.sum(ex);
    return { gcn, gat: ex.map((v) => v / es), gcnRaw, gs };
  }
  function draw2() {
    const W = weights(), w = W[show];
    E.layer('e').selectAll('line').data(ego.slice(1)).join('line')
      .attr('x1', E.x(0)).attr('y1', E.y(0)).attr('x2', (c) => E.x(pos(c)[0])).attr('y2', (c) => E.y(pos(c)[1]))
      .attr('stroke', show === 'gcn' ? cat(4) : cat(6)).attr('stroke-opacity', 0.75).attr('stroke-width', (c) => 1 + 30 * w[c.i]);
    E.layer('w').selectAll('text').data(ego.slice(1)).join('text')
      .attr('x', (c) => E.x(0.5 * pos(c)[0]) + 8).attr('y', (c) => E.y(0.5 * pos(c)[1]) - 6)
      .attr('font-size', 12).attr('fill', ink()).attr('paint-order', 'stroke').attr('stroke', css('--surface')).attr('stroke-width', 3).text((c) => w[c.i].toFixed(2));
    const g = E.layer('n').selectAll('g').data(ego).join((en) => { const gg = en.append('g'); gg.append('circle'); gg.append('text').attr('class', 'nm'); gg.append('text').attr('class', 'ft'); return gg; })
      .attr('transform', (c) => `translate(${E.x(pos(c)[0])},${E.y(pos(c)[1])})`)
      .on('mousemove', (ev, c) => tip(ev, `<b>${c.i ? TYPES[c.type] : 'centre T cell (self)'}</b><br>features [immune, tumour] = [${c.f.join(', ')}]<br>degree ${c.deg}<br>GCN weight ${W.gcn[c.i].toFixed(3)} · GAT weight ${W.gat[c.i].toFixed(3)}`))
      .on('mouseleave', untip);
    g.select('circle').attr('r', (c) => (c.i ? 20 : 24)).attr('fill', (c) => cat(c.type)).attr('stroke', (c) => (c.i ? css('--surface') : ink())).attr('stroke-width', 2);
    g.select('.nm').attr('text-anchor', (c) => (c.i ? 'middle' : 'end')).attr('x', (c) => (c.i ? 0 : -30)).attr('y', (c) => (!c.i ? 4 : pos(c)[1] < -0.3 ? 36 : -28)).attr('font-size', 12).attr('fill', ink())
      .attr('paint-order', 'stroke').attr('stroke', css('--surface')).attr('stroke-width', 3)
      .text((c) => (c.i ? `${TYPES[c.type]} (d=${c.deg})` : `self (d=${c.deg}) · ${w[0].toFixed(2)}`));
    g.select('.ft').attr('text-anchor', 'middle').attr('y', 4).attr('font-size', 10).attr('fill', css('--surface')).attr('font-weight', 700).text((c) => (c.i ? c.i : 'i'));

    // right: grouped bars per neighbour
    const labels = ego.map((c) => (c.i ? `${c.i}: ${TYPES[c.type]}` : 'i: self'));
    const yb = d3.scaleBand().domain(labels).range([0, B.h]).padding(0.25);
    B.x.domain([0, Math.max(0.4, d3.max([...W.gcn, ...W.gat]) * 1.1)]);
    B.g.select('.axis.x').call(d3.axisBottom(B.x).ticks(5));
    const data = ego.flatMap((c) => [{ c, k: 'gcn', v: W.gcn[c.i] }, { c, k: 'gat', v: W.gat[c.i] }]);
    B.layer('bars').selectAll('rect').data(data).join('rect')
      .attr('y', (d) => yb(labels[d.c.i]) + (d.k === 'gat' ? yb.bandwidth() / 2 : 0)).attr('height', yb.bandwidth() / 2 - 1)
      .attr('x', 0).attr('width', (d) => B.x(d.v)).attr('fill', (d) => (d.k === 'gcn' ? cat(4) : cat(6))).attr('rx', 2);
    B.layer('lab').selectAll('text').data(labels).join('text').attr('x', -8).attr('y', (t) => yb(t) + yb.bandwidth() / 2 + 4)
      .attr('text-anchor', 'end').attr('font-size', 12).attr('fill', ink2()).text((t) => t);
    const agg = (wt) => [0, 1].map((k) => d3.sum(ego, (c) => wt[c.i] * c.f[k]));
    const og = agg(W.gcn), oa = agg(W.gat);
    const tumourShare = (wt) => d3.sum(ego.filter((c) => c.type === 0), (c) => wt[c.i]);
    readout2.set({
      'GCN output [immune, tumour]': `[${og.map((v) => v.toFixed(2)).join(', ')}]`,
      'GAT output [immune, tumour]': `[${oa.map((v) => v.toFixed(2)).join(', ')}]`,
      'weight on tumour neighbours: GCN / GAT': `${tumourShare(W.gcn).toFixed(2)} / ${tumourShare(W.gat).toFixed(2)}`,
    });
  }
  draw2();

  section(main, 'What to notice', `<ul>
    <li><b>The first two or three rounds are good.</b> Press <i>Pass one round</i> a couple of times: isolated speckle (a pale red cell inside the blue immune zone) is averaged away, while the three domains stay distinct. This is the same neighbour-smoothing idea that spatial-domain methods use.</li>
    <li><b>Many rounds are bad.</b> Drag L to ${MAXL}: the whole tissue fades to one colour. Both curves on the right head towards zero, so no downstream layer can tell the domains apart. This is <b>over-smoothing</b>, and it is why most GNNs use only 2–4 layers.</li>
    <li>The noise curve falls much faster than the signal curve at first. Averaging cancels independent noise between neighbours, but the domain difference only erodes at the borders.</li>
    <li>Click a cell, then move the slider: the bright region is its <b>receptive field</b>, everything within L hops. The cell's value at layer L depends only on these cells.</li>
    <li>In the second figure, with <i>sharpness</i> 0 every GAT weight is equal (1/7). Increase it and turn the dial from <i>immune</i> to <i>tumour</i>: the same graph gives completely different weights, because <b>GAT decides from the features</b>, while GCN's weights never change.</li>
    <li>GCN gives the fibroblast with degree 3 a bigger weight than the fibroblast with degree 8. Hub nodes are down-weighted purely because of the graph structure.</li>
  </ul>`);

  steps(main, 'How it works', [
    'Build the graph. For spatial data, connect each cell to its physical neighbours (k nearest, a radius, or a Delaunay triangulation). For knowledge graphs, edges come from databases: gene–gene interactions, pathways, drug–target links.',
    'Give every node a feature vector: the cell\'s expression profile (or its PCA / scVI embedding), or a gene\'s identity embedding.',
    '<b>Message:</b> every neighbour j sends a vector to node i, usually a linear map W·hⱼ.',
    '<b>Aggregate:</b> node i combines incoming messages with a permutation-invariant operation: a mean, a sum, a degree-normalised sum (GCN), or an attention-weighted sum (GAT). Order does not matter because neighbours have no natural order.',
    '<b>Update:</b> pass the aggregate (often combined with the node\'s own vector) through a nonlinearity, e.g. ReLU. That is one layer. Stack L layers to reach L hops.',
    'Read out: use node vectors directly (cell-type or domain labels), combine pairs (link prediction), or pool all nodes (a whole-tissue or whole-molecule prediction). Train end to end with gradient descent.',
  ]);
  math(main, 'GCN:  hᵢ\' = σ( Σ_{j ∈ N(i) ∪ {i}}  1/√(d̃ᵢ d̃ⱼ) · W hⱼ )        d̃ = degree + 1 (self-loop)\nGAT:  eᵢⱼ = LeakyReLU( aᵀ [W hᵢ ‖ W hⱼ] ),   αᵢⱼ = softmaxⱼ(eᵢⱼ),   hᵢ\' = σ( Σⱼ αᵢⱼ W hⱼ )\nFigure 1 drops W and σ and uses a plain mean:  hᵢ\' = (hᵢ + Σⱼ hⱼ) / (dᵢ + 1)\nRepeating it L times converges to one value per connected graph: over-smoothing.');

  callout(main, '', 'In biology', `<ul>
    <li><b>Spatial omics.</b> Cells or Visium spots are nodes and physical proximity defines edges. <b>Novae</b> (Nat Methods 2025) runs a graph-attention encoder over each cell's neighbourhood and trains it self-supervised, which gives spatial domains that transfer across slides without labels. Other GNN methods for spatial domains include SpaGCN, STAGATE (graph attention autoencoder) and GraphST, and CellCharter and BANKSY use explicit neighbour aggregation. Figure 1 is the core move all of them share.</li>
    <li><b>Knowledge graphs for perturbations.</b> <b>TxPert</b> (Valence Labs / Recursion 2025) runs a GNN over gene–gene knowledge graphs (interactions, pathways, Perturb-seq similarity), so a gene never perturbed in training can borrow information from its graph neighbours. GEARS used the same idea earlier.</li>
    <li><b>Molecules</b> are graphs too: atoms are nodes and bonds are edges. Most property-prediction and docking models in drug discovery are message-passing networks.</li>
    <li>Cell–cell communication and cell-graph methods on H&amp;E images build a graph after segmenting nuclei (e.g. with Cellpose or HoVer-Net) and classify the tissue from it.</li>
  </ul>`);
  callout(main, 'warn', 'Pitfalls', `<ul>
    <li><b>Over-smoothing:</b> deep GNNs make every node alike. Use few layers, residual (skip) connections, or jumping-knowledge readouts that keep each layer's output.</li>
    <li><b>The graph is a modelling choice.</b> k, the radius, and whether you cut edges across tissue boundaries all change the result. A Visium spot graph and a single-cell graph are not interchangeable.</li>
    <li><b>Leakage across nodes:</b> neighbouring cells share information, so a random cell-level train/test split leaks. Split by slide, region or patient.</li>
    <li><b>Heterophily:</b> averaging assumes neighbours are similar. At a tumour–immune interface, the interesting cells are the ones whose neighbours are <i>different</i>, and a plain mean blurs exactly that signal. Attention or concatenating self and neighbour features (as in GraphSAGE) helps.</li>
    <li>Knowledge graphs are biased towards well-studied genes, so GNN gains can mostly reflect how many edges a gene has in the database.</li>
  </ul>`);
  seeAlso(main, [['concepts', 'attention', 'Attention'], ['concepts', 'graph-clustering', 'kNN graphs & Leiden'], ['omics', 'sp-data', 'Spatial neighbourhoods'], ['omics', 'sp-domains', 'Spatial domains'], ['frontier', 'spatial-niches', 'Novae & Nicheformer'], ['frontier', 'perturbation-arena', 'Perturbation baselines']]);
}
