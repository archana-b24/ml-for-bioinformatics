// CellPLM: treat a tissue as a sentence of cells; mask cells and reconstruct them from spatial neighbours.
import { header, figure, slider, select, button, stats, section, steps, math, legend, seeAlso, tip, untip, el } from '../../lib/ui.js';
import { frame, cat, css, ink, ink2, sequential } from '../../lib/plot.js';
import { rng, linspace } from '../../lib/data.js';
import { sourceCard, toyBadge, coreIdea, realityCheck, useIt, cite, pearson } from '../../frontier/_shared.js';
const d3 = window.d3;

const NG = 10, NB_KEEP = 160, KATT = 8;
const GENES = d3.range(NG).map((g) => `gene ${g + 1}`);
const METHODS = ['Global mean of visible cells', 'Nearest visible neighbour', `Attention over ${KATT} nearest visible cells`];

function makeTissue(seed = 5) {
  const r = rng(seed), G = 20, cells = [];
  // each gene = smooth spatial field (two bumps + a gradient) + per-cell noise
  const fields = GENES.map(() => {
    const bumps = [0, 1].map(() => ({ x: r(), y: r(), w: 0.12 + 0.15 * r(), a: (r() < 0.5 ? -1 : 1) * (1 + r()) }));
    const gx = r.normal(0, 0.8), gy = r.normal(0, 0.8);
    return (x, y) => 1.5 + gx * (x - 0.5) + gy * (y - 0.5) + bumps.reduce((s, b) => s + b.a * Math.exp(-((x - b.x) ** 2 + (y - b.y) ** 2) / (2 * b.w * b.w)), 0);
  });
  for (let i = 0; i < G; i++) for (let j = 0; j < G; j++) {
    const x = (i + 0.5 + (r() - 0.5) * 0.7) / G, y = (j + 0.5 + (r() - 0.5) * 0.7) / G;
    cells.push({ x, y, e: fields.map((f) => f(x, y) + r.normal(0, 0.35)) });
  }
  cells.forEach((c, i) => {
    c.nb = cells.map((d, j) => [j, (c.x - d.x) ** 2 + (c.y - d.y) ** 2]).filter(([j]) => j !== i).sort((a, b) => a[1] - b[1]).slice(0, NB_KEEP);
  });
  return cells;
}

/** Reconstruct masked cells. mask[i] = true if hidden. Returns {pred: {method: rows}, att: weights per masked cell, mse: [..]} */
function reconstruct(cells, mask, sigma) {
  const vis = cells.filter((_, i) => !mask[i]);
  const gmean = GENES.map((_, g) => d3.mean(vis, (c) => c.e[g]));
  const preds = [[], [], []], att = new Map(), sse = [0, 0, 0];
  let n = 0;
  cells.forEach((c, i) => {
    if (!mask[i]) return;
    const nb = []; for (const [j, dd] of c.nb) { if (!mask[j]) { nb.push([j, dd]); if (nb.length === KATT) break; } }
    if (!nb.length) nb.push([-1, 0]);
    const w = nb.map(([j, dd]) => (j < 0 ? 1 : Math.exp(-dd / (2 * sigma * sigma))));
    let z = w.reduce((a, b) => a + b, 0); if (z < 1e-300) { w.fill(1); z = w.length; }
    const wn = w.map((v) => v / z);
    att.set(i, nb.map(([j], k) => [j, wn[k]]));
    const p0 = gmean, p1 = nb[0][0] < 0 ? gmean : cells[nb[0][0]].e;
    const p2 = GENES.map((_, g) => nb.reduce((s, [j], k) => s + wn[k] * (j < 0 ? gmean[g] : cells[j].e[g]), 0));
    [p0, p1, p2].forEach((p, m) => { preds[m][i] = p; p.forEach((v, g) => { sse[m] += (v - c.e[g]) ** 2; }); });
    n += NG;
  });
  return { preds, att, mse: sse.map((s) => s / Math.max(1, n)) };
}

export default async function render(main) {
  header(main, 'CellPLM: impute cells from neighbours',
    'Language models learn by hiding words and predicting them from context. CellPLM applies that to tissue: <b>cells are the tokens, the tissue is the sentence</b>, and the model learns to reconstruct a hidden cell from the cells around it.');
  sourceCard(main, {
    org: 'CellPLM authors (academic); published at ICLR 2024',
    when: '2024',
    links: [],
    claim: 'A transformer whose tokens are whole cells, pretrained by masking cells and reconstructing their expression from other cells (including spatial neighbours), gives a cell-level pretrained model that transfers to downstream tasks.',
  });
  coreIdea(main, `<p>Single-cell foundation models such as Geneformer and scGPT treat <b>genes</b> as tokens and one cell as a sentence. CellPLM moves one level up: each <b>cell</b> is a token (its expression vector, embedded), and a set of cells from the same tissue section is the sentence. Spatial coordinates are added as a positional encoding, the way word position is in a text transformer.</p>
  <p>Training is <b>masked reconstruction</b>: hide some cells' expression and ask the model to rebuild it from the visible cells. To do that well, the model must learn which neighbours are informative, which is exactly what <a href="#/archive/topics/attention">attention</a> weights encode. A model that can do this has learned that tissue is spatially organised, and the same machinery can denoise sparse spatial panels or impute genes.</p>
  <p><b>Key ML ingredient:</b> masked-token pretraining with cells (not genes) as tokens and space as position.</p>`);

  const cells = makeTissue();
  const perm = (seed) => rng(seed).shuffle(cells.map((_, i) => i));
  let order = perm(17), maskSeed = 17;
  const rankOf = () => { const r = new Array(cells.length); order.forEach((i, k) => { r[i] = k; }); return r; };
  let ranks = rankOf();
  let frac = 0.3, sigma = 0.05, gene = 0;
  const maskFor = (f) => ranks.map((k) => k < Math.round(f * cells.length));

  const fig = figure(main, 'Mini-replication: hide cells, rebuild them from the tissue',
    `${cells.length} synthetic cells; each of ${NG} genes is a smooth spatial pattern plus per-cell noise. Left: what the model sees (hidden cells are empty rings). Right: the reconstruction by attention over the ${KATT} nearest visible cells, with weight ∝ exp(−distance²/2σ²). Hover a hidden cell to see its attention weights as lines.`);
  toyBadge(fig, 'Fixed distance-based attention stands in for CellPLM\'s learned transformer.');
  select(fig.controls, { label: 'Gene', value: '0', options: GENES.map((g, i) => [String(i), g]), onChange: (v) => { gene = +v; update(); } });
  slider(fig.controls, { label: 'Fraction masked', min: 0.05, max: 0.9, step: 0.05, value: frac, fmt: (v) => Math.round(v * 100) + '%', onInput: (v) => { frac = v; update(); } });
  slider(fig.controls, { label: 'Attention width σ', min: 0.01, max: 0.3, step: 0.01, value: sigma, fmt: (v) => v.toFixed(2), onInput: (v) => { sigma = v; update(true); } });
  button(fig.controls, 'New random mask', () => { maskSeed += 1; order = perm(maskSeed); ranks = rankOf(); update(true); });
  const readout = stats(fig.root);

  const row = el('div', { class: 'fig-row' }); fig.body.append(row);
  const lb = el('div'), rb = el('div'); row.append(lb, rb);
  lb.append(el('div', { class: 'fig-caption', style: 'margin:0 0 4px' }, 'Input: visible cells (rings = masked)'));
  rb.append(el('div', { class: 'fig-caption', style: 'margin:0 0 4px' }, 'Output: masked cells filled in by attention'));
  const opt = { width: 420, height: 420, x: [0, 1], y: [0, 1], axes: false, grid: false, margin: { top: 6, right: 6, bottom: 6, left: 6 } };
  const FL = frame(lb, opt), FR = frame(rb, opt);
  const scaleNote = el('div', { class: 'fig-caption' }); fig.body.append(scaleNote);

  // error-vs-mask-fraction curve
  const fig2 = figure(main, 'Reconstruction error as more of the tissue is hidden',
    'Mean squared error over all hidden cells and genes, for each masking fraction (masks are nested: every 40% mask contains the 20% mask). The vertical line marks the slider setting.');
  const curveBox = el('div'); fig2.body.append(curveBox);
  const FC = frame(curveBox, { width: 620, height: 300, x: [0, 0.92], y: [0, 1], xLabel: 'fraction of cells masked', yLabel: 'MSE on hidden cells' });
  FC.g.select('.axis.x').call(d3.axisBottom(FC.x).ticks(6, '.0%'));
  legend(fig2.body, METHODS.map((m, i) => [m, cat(i)]));

  let rec;
  function update(curveToo = false) {
    const mask = maskFor(frac);
    rec = reconstruct(cells, mask, sigma);
    const vals = cells.map((c) => c.e[gene]);
    const col = sequential(d3.quantile(vals, 0.02), d3.quantile(vals, 0.98));
    const sur = css('--surface');
    FL.layer('pts').selectAll('circle').data(cells).join('circle').attr('cx', (d) => FL.x(d.x)).attr('cy', (d) => FL.y(d.y)).attr('r', 5.5)
      .attr('fill', (d, i) => (mask[i] ? 'none' : col(d.e[gene]))).attr('stroke', (d, i) => (mask[i] ? ink2() : sur)).attr('stroke-width', (d, i) => (mask[i] ? 1.3 : 1))
      .on('mousemove', (e, d) => { const i = cells.indexOf(d); tip(e, mask[i] ? 'hidden from the model' : `${GENES[gene]} = ${d.e[gene].toFixed(2)}`); }).on('mouseleave', untip);
    const att = FR.layer('att');
    FR.layer('pts').selectAll('circle').data(cells).join('circle').attr('cx', (d) => FR.x(d.x)).attr('cy', (d) => FR.y(d.y)).attr('r', (d, i) => (mask[i] ? 6 : 5.5))
      .attr('fill', (d, i) => col(mask[i] ? rec.preds[2][i][gene] : d.e[gene])).attr('fill-opacity', (d, i) => (mask[i] ? 1 : 0.35))
      .attr('stroke', (d, i) => (mask[i] ? ink() : sur)).attr('stroke-width', (d, i) => (mask[i] ? 1.5 : 1))
      .on('mousemove', (e, d) => {
        const i = cells.indexOf(d);
        if (!mask[i]) { tip(e, `visible: ${d.e[gene].toFixed(2)}`); return; }
        const w = rec.att.get(i).filter(([j]) => j >= 0);
        att.selectAll('line').data(w).join('line').attr('x1', FR.x(d.x)).attr('y1', FR.y(d.y)).attr('x2', ([j]) => FR.x(cells[j].x)).attr('y2', ([j]) => FR.y(cells[j].y))
          .attr('stroke', ink()).attr('stroke-width', ([, a]) => 0.5 + 5 * a).attr('stroke-opacity', 0.75);
        tip(e, `<b>hidden cell</b><br>true ${GENES[gene]}: ${d.e[gene].toFixed(2)}<br>attention: ${rec.preds[2][i][gene].toFixed(2)}<br>nearest neighbour: ${rec.preds[1][i][gene].toFixed(2)}<br>global mean: ${rec.preds[0][i][gene].toFixed(2)}<br>top weight: ${d3.max(w, ([, a]) => a).toFixed(2)}`);
      })
      .on('mouseleave', () => { att.selectAll('line').remove(); untip(); });
    const hid = cells.map((_, i) => i).filter((i) => mask[i]);
    const r2 = pearson(hid.map((i) => rec.preds[2][i][gene]), hid.map((i) => cells[i].e[gene]));
    readout.set({
      'cells hidden': `${hid.length} / ${cells.length}`,
      'MSE: global mean': rec.mse[0].toFixed(3),
      'MSE: nearest neighbour': rec.mse[1].toFixed(3),
      'MSE: attention': rec.mse[2].toFixed(3),
      [`Pearson r (attention, ${GENES[gene]})`]: r2.toFixed(2),
    });
    scaleNote.innerHTML = `Colour: ${GENES[gene]} expression, light = low, dark = high (the noise floor is a variance of 0.35² ≈ 0.12, so no method can get MSE below about 0.12).`;
    if (curveToo || !curve) drawCurve();
    FC.layer('now').selectAll('line').data([frac]).join('line').attr('x1', (d) => FC.x(d)).attr('x2', (d) => FC.x(d)).attr('y1', 0).attr('y2', FC.h)
      .attr('stroke', ink2()).attr('stroke-dasharray', '4 3');
  }
  let curve = null;
  function drawCurve() {
    const fs = linspace(0.05, 0.9, 18);
    curve = fs.map((f) => ({ f, mse: reconstruct(cells, maskFor(f), sigma).mse }));
    const ymax = d3.max(curve, (d) => d3.max(d.mse)) * 1.08;
    FC.y.domain([0, ymax]); FC.g.select('.axis.y').call(d3.axisLeft(FC.y).ticks(5));
    FC.g.select('.gridline').call(d3.axisLeft(FC.y).ticks(5).tickSize(-FC.w).tickFormat(''));
    const lay = FC.layer('lines');
    lay.selectAll('path').data([0, 1, 2]).join('path').attr('fill', 'none').attr('stroke', (m) => cat(m)).attr('stroke-width', 2)
      .attr('d', (m) => d3.line().x((d) => FC.x(d.f)).y((d) => FC.y(d.mse[m]))(curve));
    const pts = curve.flatMap((d) => [0, 1, 2].map((m) => ({ f: d.f, m, v: d.mse[m] })));
    FC.layer('dots').selectAll('circle').data(pts).join('circle').attr('cx', (d) => FC.x(d.f)).attr('cy', (d) => FC.y(d.v)).attr('r', 4)
      .attr('fill', (d) => cat(d.m)).attr('stroke', css('--surface')).attr('stroke-width', 1.2)
      .on('mousemove', (e, d) => tip(e, `${METHODS[d.m]}<br>${Math.round(d.f * 100)}% masked: MSE ${d.v.toFixed(3)}`)).on('mouseleave', untip);
  }
  update(true);

  section(main, 'What to notice', `<ul>
    <li><b>Attention beats the global mean</b> at every masking level in the bottom chart, because the genes are spatially patterned. If expression had no spatial structure the two lines would meet.</li>
    <li><b>The nearest single neighbour</b> is worse than attention over several neighbours: one neighbour copies its noise; a weighted average cancels much of it. That is why the model attends to many cells, not one.</li>
    <li>Push <b>fraction masked</b> towards 90%. Visible cells become far apart and the attention error climbs from 0.15 (10% masked) to 0.23 (90%), while the global mean barely moves. Less context means worse reconstruction; pretraining has to leave enough of the tissue visible.</li>
    <li>Set <b>σ</b> very small (0.01): almost all weight goes to the closest visible cell, so attention collapses onto nearest neighbour (MSE 0.24 vs 0.26 at 30% masked). Set it to 0.30: weights become nearly uniform over the ${KATT} cells. In this toy's smooth patterns that is slightly better (0.145 vs 0.149 at the default), because averaging cancels noise; in tissue with sharp boundaries, uniform weights would blur them. A learned model tunes this per cell and per gene.</li>
    <li>Hover a hidden cell at a sharp edge in the pattern: its neighbours disagree, and the reconstruction is a compromise. Imputation smooths boundaries, and rare cells surrounded by different cells get overwritten by their neighbourhood.</li>
  </ul>`);
  steps(main, 'How it works', [
    'Embed each cell\'s expression vector into a token. Add a positional encoding built from its (x, y) coordinates.',
    'Randomly choose a fraction of cells and replace their tokens with a learned [MASK] token (their expression is hidden; their position is kept).',
    'Run a <a href="#/archive/topics/transformer">transformer</a> over all cell tokens of the section. Each hidden cell\'s output is an attention-weighted mix of the other cells.',
    'Decode each hidden cell\'s output back to expression and compute the reconstruction loss against its true expression. Backpropagate.',
    'After pretraining, reuse the encoder for downstream tasks (cell typing, denoising, gene imputation, perturbation prediction) or fine-tune it.',
  ]);
  math(main, 'x̂_i = Σ_j α_ij · x_j      α_ij = softmax_j( q_i·k_j / √d )         (CellPLM: learned q, k)\ntoy:  α_ij = softmax_j( −‖p_i − p_j‖² / 2σ² ) over the 8 nearest visible cells\nloss = mean over hidden cells of ‖x̂_i − x_i‖²');
  realityCheck(main, `<p>CellPLM's downstream results are <b>self-reported</b> (ICLR 2024; our brief has no link or independent benchmark for it). For single-cell foundation models generally, independent work such as ${cite('kedzierska', 'Kedzierska et al. 2025')} found zero-shot embeddings losing to HVG+PCA, scVI and Harmony, so cell-level pretraining has to prove itself on your task, not by its loss.</p>
  <p>The toy also shows a built-in bias: spatial imputation pulls every cell towards its neighbourhood. That is good for denoising smooth tissue and bad for rare or infiltrating cells (a single T cell inside tumour looks like tumour after imputation). Masked-reconstruction loss on held-out cells is the honest metric; compare it with the neighbour-averaging baseline, which costs nothing to compute.</p>`);
  useIt(main, `<ul>
    <li><b>Denoising a sparse imaging panel:</b> before trusting any model, compute the toy's baseline (kernel-weighted average of the k nearest cells) and its held-out reconstruction error on a random 10–20% of cells. A pretrained model should beat it.</li>
    <li><b>Never run DE on imputed values:</b> neighbour borrowing creates spatial autocorrelation and inflates significance. Keep raw counts for statistics.</li>
    <li><b>Watch rare cells:</b> compare cell-type calls before and after imputation for small populations at boundaries.</li>
    <li><b>Masking as QC:</b> the same mask-and-reconstruct test tells you how spatially predictable each gene is, which is a quick screen for spatially patterned genes.</li>
  </ul>`);
  seeAlso(main, [['concepts', 'attention', 'Attention'], ['concepts', 'transformer', 'Transformers'], ['concepts', 'foundation-models', 'Foundation models'], ['concepts', 'knn', 'k-nearest neighbours'],
    ['omics', 'sp-data', 'Spatial data & neighbourhoods'], ['omics', 'sp-svg', 'Spatially variable genes'], ['omics', 'sp-deconvolution', 'Spot deconvolution'], ['frontier', 'spatial-niches', 'Novae & Nicheformer'], ['frontier', 'geneformer', 'Geneformer: masked genes']]);
}
