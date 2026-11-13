// Embeddings: one-hot vs dense, a toy skip-gram (word2vec) trained live on synthetic gene "sentences",
// cosine similarity on hover, and vector arithmetic.
import { header, figure, slider, select, button, stats, section, steps, callout, math, legend, seeAlso, animate, tip, untip, el, html } from '../lib/ui.js';
import { frame, heatmap, cat, css, ink, ink2, muted, diverging } from '../lib/plot.js';
import { rng } from '../lib/data.js';
const d3 = window.d3;

const PROGRAMS = [
  { name: 'T cell', genes: ['CD3D', 'CD3E', 'CD2', 'IL7R'] },
  { name: 'B cell', genes: ['MS4A1', 'CD79A', 'CD79B', 'CD19'] },
  { name: 'Myeloid', genes: ['LYZ', 'CD14', 'S100A8', 'FCN1'] },
  { name: 'Interferon', genes: ['ISG15', 'IFI6', 'MX1', 'IFIT3'] },
  { name: 'Cell cycle', genes: ['MKI67', 'TOP2A', 'CDK1', 'CCNB1'] },
];
const GENES = PROGRAMS.flatMap((p, k) => p.genes.map((g) => ({ name: g, prog: k })));
const G = GENES.length, D = 3, EPOCHS = 60, N_CELLS = 300;
const idx = (name) => GENES.findIndex((g) => g.name === name);

/** Synthetic "cells": each is a bag of 3 genes from one program, sometimes plus 2 from a second program.
 *  With probability `noise` each gene is swapped for a random gene (dropout / ambient RNA, loosely). */
function makeCorpus(noise, seed = 7) {
  const r = rng(seed), cells = [];
  for (let s = 0; s < N_CELLS; s++) {
    const k = r.int(PROGRAMS.length);
    const cell = r.shuffle([0, 1, 2, 3]).slice(0, 3).map((j) => k * 4 + j);
    if (r() < 0.25) { const k2 = r.int(PROGRAMS.length); if (k2 !== k) r.shuffle([0, 1, 2, 3]).slice(0, 2).forEach((j) => cell.push(k2 * 4 + j)); }
    cells.push(cell.map((g) => (r() < noise ? r.int(G) : g)));
  }
  const pairs = [];
  cells.forEach((c) => c.forEach((a, i) => c.forEach((b, j) => { if (i !== j && a !== b) pairs.push([a, b]); })));
  const counts = new Float64Array(G); cells.flat().forEach((g) => counts[g]++);
  const w = Array.from(counts, (c) => Math.pow(c + 1, 0.75)), tot = d3.sum(w);
  let acc = 0; const cdf = w.map((v) => (acc += v / tot));
  return { cells, pairs, cdf };
}

const sig = (z) => 1 / (1 + Math.exp(-z));
function cosine(W, a, b) {
  let d = 0, na = 0, nb = 0;
  for (let k = 0; k < D; k++) { d += W[a * D + k] * W[b * D + k]; na += W[a * D + k] ** 2; nb += W[b * D + k] ** 2; }
  return d / (Math.sqrt(na * nb) + 1e-12);
}
const cosVec = (u, v) => { let d = 0, a = 0, b = 0; for (let k = 0; k < u.length; k++) { d += u[k] * v[k]; a += u[k] * u[k]; b += v[k] * v[k]; } return d / (Math.sqrt(a * b) + 1e-12); };

/** Eigen-decomposition of a small symmetric matrix (Jacobi rotations). Returns {vals, vecs} sorted by value, vecs as columns-list. */
function symEig(A) {
  const n = A.length, M = A.map((r) => r.slice()), V = A.map((r, i) => r.map((_, j) => (i === j ? 1 : 0)));
  for (let sweep = 0; sweep < 50; sweep++) {
    let off = 0; for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += M[i][j] ** 2;
    if (off < 1e-18) break;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
      if (Math.abs(M[p][q]) < 1e-15) continue;
      const th = 0.5 * Math.atan2(2 * M[p][q], M[q][q] - M[p][p]), c = Math.cos(th), s = Math.sin(th);
      for (let k = 0; k < n; k++) { const a = M[k][p], b = M[k][q]; M[k][p] = c * a - s * b; M[k][q] = s * a + c * b; }
      for (let k = 0; k < n; k++) { const a = M[p][k], b = M[q][k]; M[p][k] = c * a - s * b; M[q][k] = s * a + c * b; }
      for (let k = 0; k < n; k++) { const a = V[k][p], b = V[k][q]; V[k][p] = c * a - s * b; V[k][q] = s * a + c * b; }
    }
  }
  const order = d3.range(n).sort((i, j) => M[j][j] - M[i][i]);
  return { vals: order.map((i) => M[i][i]), vecs: order.map((i) => V.map((r) => r[i])) };
}

export default async function render(main) {
  header(main, 'Embeddings',
    'An embedding turns a discrete thing (a word, a gene, a cell) into a short vector of numbers, so that <b>things used in similar contexts end up close together</b>. Nobody tells the model which genes belong together: it learns that from which genes appear together.');

  // ---------------- Figure 1: skip-gram training ----------------
  const fig = figure(main, 'Skip-gram (word2vec) trained live on gene "sentences"',
    `Training data: ${N_CELLS} synthetic cells. Each cell is a bag of genes: 3 genes from one of five programs, and in a quarter of cells 2 extra genes from a second program. The model learns one 3-D vector per gene so that genes that share cells get a high dot product, and random gene pairs get a low one. Skip-gram vectors all share one common direction, so we subtract the mean vector and plot the plane that holds most of what is left (its share of variance is in the readout). Hover a gene to see its cosine similarity to all the others.`);
  let noise = 0.1, K = 4, lr0 = 0.05;
  let data, W, C, epoch, hist, trng, stop = null, hovered = -1, disp = 1;
  // View: skip-gram vectors share a common offset direction, so we centre them (subtract the mean gene vector).
  // Cosines use the centred 3-D vectors; the plot shows the top-2 principal plane, rotated to match the previous frame.
  let E = new Float64Array(G * D), XY = [], basis = null, R = [[1, 0], [0, 1]], varShown = 1;
  function view(keepOrientation) {
    const mu = d3.range(D).map((k) => d3.mean(d3.range(G), (i) => W[i * D + k]));
    for (let i = 0; i < G; i++) for (let k = 0; k < D; k++) E[i * D + k] = W[i * D + k] - mu[k];
    const cov = d3.range(D).map((a) => d3.range(D).map((b) => d3.sum(d3.range(G), (i) => E[i * D + a] * E[i * D + b]) / G));
    const { vals, vecs } = symEig(cov);
    varShown = (vals[0] + vals[1]) / Math.max(1e-12, d3.sum(vals));
    basis = [vecs[0], vecs[1]];
    const raw = d3.range(G).map((i) => proj(d3.range(D).map((k) => E[i * D + k]), [[1, 0], [0, 1]]));
    if (keepOrientation && XY.length) {           // 2-D Procrustes: best rotation (or reflection) onto the last frame
      let best = null;
      for (const flip of [1, -1]) {
        let sxx = 0, sxy = 0;
        raw.forEach((p, i) => { const q = XY[i], px = p[0], py = flip * p[1]; sxx += px * q[0] + py * q[1]; sxy += px * q[1] - py * q[0]; });
        const th = Math.atan2(sxy, sxx), score = Math.hypot(sxx, sxy);
        if (!best || score > best.score) best = { score, R: [[Math.cos(th), Math.sin(th)], [-Math.sin(th) * flip, Math.cos(th) * flip]] };
      }
      R = best.R;
    } else R = [[1, 0], [0, 1]];
    XY = raw.map((p) => [p[0] * R[0][0] + p[1] * R[1][0], p[0] * R[0][1] + p[1] * R[1][1]]);
  }
  function proj(v, Rm = R) {
    const p = [0, 1].map((c) => d3.sum(d3.range(D), (k) => v[k] * basis[c][k]));
    return [p[0] * Rm[0][0] + p[1] * Rm[1][0], p[0] * Rm[0][1] + p[1] * Rm[1][1]];
  }

  slider(fig.controls, { label: 'Noise genes', min: 0, max: 0.6, step: 0.05, value: noise, fmt: (v) => Math.round(v * 100) + '%', onInput: (v) => { noise = v; reset(true); } });
  slider(fig.controls, { label: 'Negatives per pair', min: 1, max: 8, step: 1, value: K, onInput: (v) => { K = v; reset(true); } });
  const runBtn = button(fig.controls, 'Train', () => (stop ? pause() : run()), true);
  button(fig.controls, 'Reset', () => reset(false));
  const readout = stats(fig.root);

  const row = el('div', { class: 'fig-row' }); fig.body.append(row);
  const left = el('div'), right = el('div'); row.append(left, right);
  const F = frame(left, { width: 500, height: 440, x: [-1, 1], y: [-1, 1], xLabel: 'embedding dimension 1', yLabel: 'embedding dimension 2', grid: false });
  legend(left, PROGRAMS.map((p, k) => [p.name + ' program', cat(k)]));
  const Lc = frame(right, { width: 460, height: 440, x: [0, EPOCHS], y: [-1, 1], xLabel: 'epoch (one pass over all gene pairs)', yLabel: 'mean cosine similarity' });
  legend(right, [['gene pairs in the same program', ink()], ['gene pairs in different programs', muted()]]);
  Lc.layer('zero').append('line').attr('x1', 0).attr('x2', Lc.w).attr('y1', Lc.y(0)).attr('y2', Lc.y(0)).attr('stroke', css('--axis')).attr('stroke-dasharray', '3 3');

  function reset(autorun) {
    if (stop) { stop(); stop = null; }
    data = makeCorpus(noise);
    const r0 = rng(11);
    W = new Float64Array(G * D).map(() => (r0() - 0.5) * 0.2);
    C = new Float64Array(G * D);           // word2vec initialises the context vectors at zero
    epoch = 0; XY = []; hist = [{ e: 0, ...summary() }]; trng = rng(23);
    paintAll();
    if (autorun) run(); else runBtn.textContent = 'Train';
  }

  function sampleNeg() { const u = trng(); let i = 0; while (i < G - 1 && data.cdf[i] < u) i++; return i; }
  function trainEpoch() {
    const lr = lr0 * Math.max(0.1, 1 - epoch / EPOCHS);
    const order = trng.shuffle(data.pairs);
    let loss = 0;
    const gw = new Float64Array(D);
    for (const [a, b] of order) {
      gw.fill(0);
      for (let t = 0; t <= K; t++) {
        const c = t === 0 ? b : sampleNeg();
        if (t > 0 && c === b) continue;
        let dot = 0; for (let k = 0; k < D; k++) dot += W[a * D + k] * C[c * D + k];
        const s = sig(dot), label = t === 0 ? 1 : 0, g = s - label;
        loss += t === 0 ? -Math.log(s + 1e-9) : -Math.log(1 - s + 1e-9);
        for (let k = 0; k < D; k++) { gw[k] += g * C[c * D + k]; C[c * D + k] -= lr * g * W[a * D + k]; }
      }
      for (let k = 0; k < D; k++) W[a * D + k] -= lr * gw[k];
    }
    epoch++;
    hist.push({ e: epoch, loss: loss / order.length, ...summary() });
  }
  function summary() {
    view(true);
    let ws = 0, wn = 0, bs = 0, bn = 0;
    for (let a = 0; a < G; a++) for (let b = a + 1; b < G; b++) {
      const c = cosine(E, a, b);
      if (GENES[a].prog === GENES[b].prog) { ws += c; wn++; } else { bs += c; bn++; }
    }
    return { within: ws / wn, between: bs / bn };
  }

  function run() {
    if (epoch >= EPOCHS) reset(false);
    runBtn.textContent = 'Pause';
    let acc = 0;
    stop = animate((dt) => {
      acc += dt; if (acc < 45) return true; acc = 0;
      trainEpoch(); paintAll();
      if (epoch >= EPOCHS) { stop = null; runBtn.textContent = 'Train again'; return false; }
      return true;
    });
  }
  function pause() { if (stop) stop(); stop = null; runBtn.textContent = 'Resume'; }

  const P = (i) => XY[i];
  const vecE = (i) => d3.range(D).map((k) => E[i * D + k]);
  function paint1() {
    let m = 0.05; XY.forEach((p) => { m = Math.max(m, Math.abs(p[0]), Math.abs(p[1])); });
    disp = epoch === 0 ? m * 1.25 : disp + 0.35 * (m * 1.25 - disp);
    F.x.domain([-disp, disp]); F.y.domain([-disp, disp]);
    F.g.select('.axis.x').call(d3.axisBottom(F.x).ticks(5)); F.g.select('.axis.y').call(d3.axisLeft(F.y).ticks(5));
    const links = hovered < 0 ? [] : GENES.map((g, j) => ({ j, c: cosine(E, hovered, j) })).filter((d) => d.j !== hovered && d.c > 0);
    F.layer('links').selectAll('line').data(links).join('line')
      .attr('x1', hovered < 0 ? 0 : F.x(P(hovered)[0])).attr('y1', hovered < 0 ? 0 : F.y(P(hovered)[1]))
      .attr('x2', (d) => F.x(P(d.j)[0])).attr('y2', (d) => F.y(P(d.j)[1]))
      .attr('stroke', ink2()).attr('stroke-width', (d) => 0.5 + 2.5 * d.c).attr('opacity', (d) => 0.15 + 0.6 * d.c);
    const gs = F.layer('genes').selectAll('g.gene').data(GENES).join((en) => {
      const g = en.append('g').attr('class', 'gene').style('cursor', 'pointer');
      g.append('circle').attr('r', 6).attr('stroke', css('--surface')).attr('stroke-width', 1.5);
      g.append('text').attr('font-size', 11).attr('dx', 8).attr('dy', 4).attr('fill', ink2());
      return g;
    });
    gs.attr('transform', (d, i) => `translate(${F.x(P(i)[0])},${F.y(P(i)[1])})`)
      .on('mousemove', (e, d) => { hovered = GENES.indexOf(d); paint1(); showTip(e, hovered); })
      .on('mouseleave', () => { hovered = -1; paint1(); untip(); });
    gs.select('circle').attr('fill', (d) => cat(d.prog)).attr('r', (d, i) => (i === hovered ? 8 : 6));
    gs.select('text').text((d) => d.name).attr('font-weight', (d, i) => (i === hovered ? 700 : 400));
  }
  function showTip(e, a) {
    const sims = GENES.map((g, j) => ({ g, c: cosine(E, a, j) })).filter((d) => d.g !== GENES[a]).sort((x, y) => y.c - x.c);
    tip(e, `<b>${GENES[a].name}</b> (${PROGRAMS[GENES[a].prog].name})<br>vector = [${vecE(a).map((v) => v.toFixed(2)).join(', ')}]<br>most similar (cosine):<br>` +
      sims.slice(0, 4).map((d) => `${d.g.name} ${d.c.toFixed(2)}`).join('<br>') + `<br>least similar: ${sims[sims.length - 1].g.name} ${sims[sims.length - 1].c.toFixed(2)}`);
  }
  function paintCurves() {
    const ln = (key) => d3.line().x((d) => Lc.x(d.e)).y((d) => Lc.y(d[key]));
    Lc.layer('c').selectAll('path').data([['within', ink(), null], ['between', muted(), '6 4']]).join('path')
      .attr('fill', 'none').attr('stroke', (d) => d[1]).attr('stroke-width', 2).attr('stroke-dasharray', (d) => d[2])
      .attr('d', (d) => ln(d[0])(hist));
    const last = hist[hist.length - 1];
    readout.set({ epoch: `${epoch} / ${EPOCHS}`, 'variance in this 2-D view': Math.round(varShown * 100) + '%', 'skip-gram loss': last.loss == null ? '–' : last.loss.toFixed(3), 'same-program cosine': last.within.toFixed(2), 'cross-program cosine': last.between.toFixed(2) });
  }
  function paintAll() { paint1(); paintCurves(); if (epoch % 4 === 0 || epoch >= EPOCHS) paint2(); paint3(); }

  // ---------------- Figure 2: one-hot vs dense ----------------
  const fig2 = figure(main, 'One-hot vs learned embedding: what the representation "knows"',
    'Each cell of a heatmap is the cosine similarity between two genes (red = similar, blue = opposite). Genes are ordered by program. Hover a cell for its value. The right heatmap updates as the model in the figure above trains.');
  const row2 = el('div', { class: 'fig-row' }); fig2.body.append(row2);
  const h1 = el('div'), h2 = el('div'); row2.append(h1, h2);
  const oneHotTxt = html('', 'p', 'fig-caption'), denseTxt = html('', 'p', 'fig-caption');
  const h1Map = el('div'), h2Map = el('div');
  h1.append(html('<b>One-hot</b> (20 numbers per gene)', 'div'), oneHotTxt, h1Map);
  h2.append(html('<b>Learned dense embedding</b> (3 numbers per gene)', 'div'), denseTxt, h2Map);
  const e3 = idx('CD3E');
  oneHotTxt.innerHTML = `CD3E = [${GENES.map((g, i) => (i === e3 ? 1 : 0)).join(',')}]<br>Every pair of different genes has cosine 0: CD3E is as far from CD3D as from MKI67.`;
  const names = GENES.map((g) => g.name);
  const hm = (parent, M) => {
    parent.innerHTML = '';
    heatmap(parent, M, { cell: 15, color: diverging(1), rowLabels: names, colLabels: names, labelW: 56, labelH: 52,
      onHover: (e, i, j, v) => tip(e, `${names[i]} · ${names[j]}<br>cosine = <b>${v.toFixed(2)}</b>`) });
  };
  hm(h1Map, GENES.map((a, i) => GENES.map((b, j) => (i === j ? 1 : 0))));
  function paint2() {
    hm(h2Map, GENES.map((a, i) => GENES.map((b, j) => cosine(E, i, j))));
    denseTxt.innerHTML = `CD3E = [${vecE(e3).map((v) => v.toFixed(2)).join(', ')}] after ${epoch} epochs.<br>Blocks on the diagonal are the five programs, found from co-occurrence alone.`;
  }

  // ---------------- Figure 3: vector arithmetic ----------------
  const fig3 = figure(main, 'Vector arithmetic: a − b + c',
    'The offset b → a (dashed arrow) is a direction in embedding space. Adding the same offset to c (solid arrow) asks "b is to a as c is to ?". The answer is the gene closest by cosine to the star, excluding a, b and c themselves.');
  let A = 'CD3E', B = 'MS4A1', Cg = 'CD79A';
  const opts = GENES.map((g) => [g.name, `${g.name} (${PROGRAMS[g.prog].name})`]);
  select(fig3.controls, { label: 'a', value: A, options: opts, onChange: (v) => { A = v; paint3(); } });
  select(fig3.controls, { label: '− b', value: B, options: opts, onChange: (v) => { B = v; paint3(); } });
  select(fig3.controls, { label: '+ c', value: Cg, options: opts, onChange: (v) => { Cg = v; paint3(); } });
  const readout3 = stats(fig3.root);
  const F3 = frame(fig3.body, { width: 560, height: 400, x: [-1, 1], y: [-1, 1], xLabel: 'embedding dimension 1', yLabel: 'embedding dimension 2', grid: false });
  const defs = F3.svg.append('defs');
  defs.append('marker').attr('id', 'emb-arrow').attr('viewBox', '0 0 10 10').attr('refX', 9).attr('refY', 5).attr('markerWidth', 7).attr('markerHeight', 7).attr('orient', 'auto')
    .append('path').attr('d', 'M0,0L10,5L0,10z').attr('fill', ink());
  function paint3() {
    if (!F3) return;
    F3.x.domain([-disp, disp]); F3.y.domain([-disp, disp]);
    F3.g.select('.axis.x').call(d3.axisBottom(F3.x).ticks(5)); F3.g.select('.axis.y').call(d3.axisLeft(F3.y).ticks(5));
    const a = idx(A), b = idx(B), c = idx(Cg);
    const pa = P(a), pb = P(b), pc = P(c);
    const r3 = d3.range(D).map((k) => E[a * D + k] - E[b * D + k] + E[c * D + k]), r = proj(r3);
    const ranked = GENES.map((g, j) => ({ g, j, c: cosVec(r3, vecE(j)) })).filter((d) => ![a, b, c].includes(d.j)).sort((x, y) => y.c - x.c);
    F3.layer('pts').selectAll('circle').data(GENES).join('circle').attr('cx', (d, i) => F3.x(P(i)[0])).attr('cy', (d, i) => F3.y(P(i)[1]))
      .attr('r', (d, i) => ([a, b, c, ranked[0].j].includes(i) ? 7 : 4.5)).attr('fill', (d) => cat(d.prog))
      .attr('opacity', (d, i) => ([a, b, c, ranked[0].j].includes(i) ? 1 : 0.35)).attr('stroke', css('--surface')).attr('stroke-width', 1.5);
    F3.layer('lbl').selectAll('text').data([[a, 'a'], [b, 'b'], [c, 'c'], [ranked[0].j, 'answer']]).join('text')
      .attr('x', (d) => F3.x(P(d[0])[0]) + 9).attr('y', (d) => F3.y(P(d[0])[1]) - 7).attr('font-size', 12).attr('fill', ink())
      .text((d) => `${d[1]}: ${GENES[d[0]].name}`);
    const arrows = [[pb, pa, '5 4'], [pc, r, null]];
    F3.layer('arr').selectAll('line').data(arrows).join('line')
      .attr('x1', (d) => F3.x(d[0][0])).attr('y1', (d) => F3.y(d[0][1])).attr('x2', (d) => F3.x(d[1][0])).attr('y2', (d) => F3.y(d[1][1]))
      .attr('stroke', ink()).attr('stroke-width', 2).attr('stroke-dasharray', (d) => d[2]).attr('marker-end', 'url(#emb-arrow)');
    F3.layer('star').selectAll('path').data([r]).join('path').attr('d', d3.symbol(d3.symbolStar, 180)())
      .attr('transform', `translate(${F3.x(r[0])},${F3.y(r[1])})`).attr('fill', ink()).attr('stroke', css('--surface'));
    readout3.set({ 'nearest gene to a − b + c': `${ranked[0].g.name} (${PROGRAMS[ranked[0].g.prog].name})`, cosine: ranked[0].c.toFixed(3), 'runner-up': `${ranked[1].g.name} ${ranked[1].c.toFixed(2)}` });
  }

  reset(true);

  section(main, 'What to notice', `<ul>
    <li><b>Programs emerge from co-occurrence alone.</b> At epoch 0 the 20 genes sit in a random blob. Within a few epochs the five programs split into separate arms, and the same-program cosine (solid line) climbs towards 1 while the cross-program cosine (dashed) falls below 0.</li>
    <li>Hover a gene, e.g. <b>ISG15</b>. The thick links go to the other interferon genes. The model was never told that ISG15 and MX1 are both interferon-stimulated genes.</li>
    <li>Push <b>noise genes</b> to 60%: most "cells" are now mostly random genes, the arms blur, and the gap between the two cosine curves shrinks. Embeddings are only as good as the co-occurrence signal they are trained on.</li>
    <li>In the one-hot heatmap every off-diagonal cell is 0. One-hot vectors can say "same gene or not", nothing more. The learned heatmap has five red blocks: similarity has become geometry.</li>
    <li>In the arithmetic panel, "MS4A1 is to CD3E as CD79A is to ?" returns a T-cell gene: the B→T offset is a consistent direction. Try a − b + c with a and b from the same program: the offset is small, so the answer is just a neighbour of c.</li>
  </ul>`);

  steps(main, 'How it works', [
    '<b>One-hot</b>: give each of V tokens (words, genes) its own axis. Gene 2 of 20 is <code>[0,1,0,…,0]</code>. All tokens are equally far apart, and the vector length grows with the vocabulary.',
    '<b>Embedding table</b>: a learnable matrix E of size V × d, with d ≪ V. Looking up a token\'s row is the same as multiplying its one-hot vector by E. So an embedding layer is just a linear layer applied to one-hot inputs.',
    '<b>Skip-gram objective</b>: for each pair (gene, context gene) that appears together in a cell, increase σ(e<sub>gene</sub> · c<sub>context</sub>). For K randomly drawn "negative" genes, decrease it. Each gene has two vectors: e (plotted) and c (used as context).',
    'Gradient descent on that loss makes genes that share contexts get similar e vectors, because they must all score high against the same c vectors.',
    '<b>Use</b>: compare tokens by <b>cosine similarity</b> (the angle between vectors, ignoring length), cluster them, add them up (a cell ≈ the average of its gene vectors), or feed them into a bigger model as its first layer.',
  ]);
  math(main, 'loss(a, b) = −log σ(e_a · c_b) − Σ_{k=1..K} log σ(−e_a · c_nk),   n_k ~ P(gene)^0.75\ncos(u, v) = u·v / (‖u‖ ‖v‖)            embedding lookup:  e_a = onehot(a)ᵀ E');

  callout(main, '', 'In biology', `<ul>
    <li><b>Gene2vec</b> (Du et al. 2019) trained exactly this skip-gram on co-expression in ~1,000 GEO datasets. Genes in the same pathway end up close.</li>
    <li><b>Geneformer and scGPT</b> start with a learned gene-token embedding table (one row per gene, a few hundred dimensions), then refine it with a transformer. Their "cell embedding" is often the average of the output gene vectors: the same "cell = sum of genes" idea as above.</li>
    <li><b>Cells are embedded too</b>: PCA scores, scVI latent means and foundation-model cell embeddings are all dense vectors where cosine or Euclidean distance means "similar state". The kNN graph that Leiden and UMAP use is built on them.</li>
    <li>DNA k-mers (DNA2vec), protein sequences (ESM) and spatial niches (Novae) are embedded the same way. Some cross-species methods (SATURN, UCE) align genes across species through protein-language-model embeddings, because one-hot gene IDs cannot be matched across genomes.</li>
  </ul>`);
  callout(main, 'warn', 'Pitfalls', `<ul>
    <li><b>Axes mean nothing on their own.</b> Rotate or reflect the embedding and the loss is unchanged. Only distances and angles are meaningful, so never interpret "dimension 1" as a biological axis without checking.</li>
    <li><b>Co-occurrence is not causation or function.</b> Genes driven by a shared batch artefact (e.g. mitochondrial or ribosomal genes from stressed cells) will also embed together.</li>
    <li>Rare tokens get few updates, so their vectors are noisy. Lowly expressed genes are rare tokens.</li>
    <li>Analogy results are flattering: the raw nearest neighbour of a − b + c is usually a or c, which is why they are excluded by convention.</li>
    <li>Real embeddings use 50–1,000 dimensions. Plotting them needs PCA or UMAP, which adds its own distortion.</li>
  </ul>`);
  seeAlso(main, [['concepts', 'attention', 'Attention'], ['concepts', 'transformer', 'Transformers'], ['concepts', 'pca', 'PCA'], ['concepts', 'tsne-umap', 't-SNE & UMAP'], ['frontier', 'geneformer', 'Geneformer'], ['frontier', 'zero-shot-check', 'Do foundation embeddings beat PCA?']]);
}
