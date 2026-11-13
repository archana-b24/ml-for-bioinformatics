// Large Perturbation Model (LPM): disentangle perturbation × readout × context.
// Mini-replication: a CP (multilinear) factorisation of a perturbation × gene × cell-line tensor, fitted by masked ALS, predicting unmeasured combos.
import { header, figure, slider, button, stats, section, steps, math, legend, seeAlso, every, tip, untip, el } from '../lib/ui.js';
import { frame, cat, css, ink2, muted, diverging } from '../lib/plot.js';
import { rng } from '../lib/data.js';
import { sourceCard, toyBadge, coreIdea, realityCheck, useIt, cite, pearson, cholSolve } from './_shared.js';
const d3 = window.d3;

const P = 30, G = 50, C = 6, R_TRUE = 3, SWEEPS = 30, LAMBDA = 1;
const CTX = ['K562', 'HepG2', 'A549', 'iPSC', 'T cell', 'MCF7'];

function simulate(noise) {
  const r = rng(77);
  const A = d3.range(P).map(() => d3.range(R_TRUE).map(() => r.normal(0, 1)));
  const B = d3.range(G).map(() => d3.range(R_TRUE).map(() => r.normal(0, 1)));
  const D = d3.range(C).map(() => d3.range(R_TRUE).map(() => 0.2 + 1.4 * r()));
  const Y = d3.range(P).map((p) => d3.range(C).map((c) => d3.range(G).map((g) => {
    let s = 0; for (let k = 0; k < R_TRUE; k++) s += A[p][k] * B[g][k] * D[c][k];
    return s + r.normal(0, noise);
  })));
  return Y; // Y[p][c][g]
}
function makeMask(frac) {
  const r = rng(91);
  const M = d3.range(P).map(() => d3.range(C).map(() => r() < frac));
  M.forEach((row, p) => { if (!row.some(Boolean)) row[p % C] = true; }); // every perturbation measured somewhere
  d3.range(C).forEach((c) => { if (!M.some((row) => row[c])) M[c][c] = true; });
  // make sure each perturbation has at least one held-out context, when possible
  return M;
}

export const __test = { simulate, makeMask };
export default async function render(main) {
  header(main, 'Large Perturbation Model: perturbation × gene × context',
    'Every perturbation experiment has three ingredients: <b>what you did</b> (the perturbation), <b>what you measured</b> (the readout, e.g. genes) and <b>where you did it</b> (the context: cell line, donor, assay). Learn a separate representation for each and you can predict combinations <b>nobody has run yet</b>.');

  sourceCard(main, {
    org: 'LPM authors (arXiv 2503.23535; Nature Computational Science 2025). Genentech affiliation is inferred from the GitHub organisation and <b>unverified</b>.',
    when: '2025',
    links: ['lpm'],
    claim: 'LPM disentangles perturbation, readout and context, so that one model can be trained across heterogeneous perturbation experiments and queried for unseen combinations.',
  });

  coreIdea(main, `<p>Picture all your perturbation data as a big 3-D block: perturbations along one side, genes along another, cell lines along the third. Most of the block is empty, because nobody has run every perturbation in every cell line. If the response really decomposes into "perturbation programme × gene loading × how strongly this context expresses the programme", then the few measured slices pin down the factors, and the factors fill in the empty slices.</p>
  <p><b>Key ML ingredient:</b> <b>disentangled, shared representations</b>, learned by <a href="#/concepts/embeddings">embedding</a> each axis separately and combining them. This toy uses the simplest combination, a multilinear CP product fitted by alternating least squares. That is <b>not LPM's architecture</b>; it isolates the disentangling idea so you can watch it work.</p>`);

  const fig = figure(main, 'Fill in the unmeasured perturbation × cell-line combinations',
    `${P} perturbations × ${G} genes × ${C} cell lines, generated from ${R_TRUE} hidden programmes whose strength differs by cell line. Left: each column is a perturbation, each row a cell line. Grey cells were measured (training); coloured cells were never measured and are predicted, coloured by the Pearson r between predicted and true delta. Click a coloured cell to inspect it. Bottom left: held-out r during fitting, against two baselines (dashed). Bottom right: predicted vs true genes for the selected combination.`);
  toyBadge(fig, 'CP factorisation + masked ALS, not LPM\'s network');
  let rank = 3, frac = 0.4, noise = 0.5, Y, M, fit, hist, stop = null, sel = null, base;
  slider(fig.controls, { label: 'Model rank', min: 1, max: 6, step: 1, value: rank, onInput: (v) => { rank = v; refit(); } });
  slider(fig.controls, { label: 'Fraction measured', min: 0.2, max: 0.8, step: 0.05, value: frac, fmt: (v) => Math.round(v * 100) + '%', onInput: (v) => { frac = v; rebuild(); } });
  slider(fig.controls, { label: 'Noise', min: 0.1, max: 1.5, step: 0.1, value: noise, fmt: (v) => v.toFixed(1), onInput: (v) => { noise = v; rebuild(); } });
  button(fig.controls, 'Refit', () => refit(), true);
  legend(fig.root, [['factorised model', cat(0)], ['perturbation-mean baseline', cat(1)], ['cell-line-mean baseline', cat(2)]]);
  const readout = stats(fig.root);

  const a = el('div', { style: 'margin-bottom:10px' }); fig.body.append(a);
  const row = el('div', { class: 'fig-row' }); fig.body.append(row);
  const b = el('div'), c = el('div'); row.append(b, c);
  const cell = 28, hmW = 46 + P * cell + 6, hmH = 30 + C * 20 + 30;
  const hm = d3.select(a).append('svg').attr('viewBox', `0 0 ${hmW} ${hmH}`).attr('width', hmW);
  const hmG = hm.append('g').attr('transform', 'translate(46,24)');
  hm.append('text').attr('x', 46).attr('y', 14).attr('font-size', 11).attr('fill', ink2()).text('perturbation →');
  CTX.forEach((t, i) => hm.append('text').attr('x', 42).attr('y', 24 + i * 20 + 14).attr('text-anchor', 'end').attr('font-size', 11).attr('fill', ink2()).text(t));
  const colorR = diverging(1);
  const keyG = hm.append('g').attr('transform', `translate(46,${24 + C * 20 + 10})`);
  [-1, -0.5, 0, 0.5, 1].forEach((v, i) => { keyG.append('rect').attr('x', i * 34).attr('width', 32).attr('height', 9).attr('fill', colorR(v)); keyG.append('text').attr('x', i * 34 + 16).attr('y', 20).attr('text-anchor', 'middle').attr('font-size', 10).attr('fill', ink2()).text(v); });
  keyG.append('text').attr('x', 5 * 34 + 6).attr('y', 9).attr('font-size', 10).attr('fill', ink2()).text('held-out Pearson r');
  const F2 = frame(b, { width: 460, height: 340, x: [0, SWEEPS], y: [-0.2, 1], xLabel: 'ALS sweep', yLabel: 'mean held-out r' });
  const F3 = frame(c, { width: 460, height: 340, x: [-6, 6], y: [-6, 6], xLabel: 'true delta (gene)', yLabel: 'predicted delta' });

  function baselines() {
    const pm = [], cm = [];
    for (let p = 0; p < P; p++) for (let cc = 0; cc < C; cc++) {
      if (M[p][cc]) continue;
      const others = d3.range(C).filter((k) => M[p][k]);
      const pPred = d3.range(G).map((g) => d3.mean(others, (k) => Y[p][k][g]));
      const peers = d3.range(P).filter((q) => M[q][cc]);
      const cPred = d3.range(G).map((g) => d3.mean(peers, (q) => Y[q][cc][g]));
      pm.push(pearson(pPred, Y[p][cc])); cm.push(peers.length ? pearson(cPred, Y[p][cc]) : 0);
    }
    return { pm: d3.mean(pm), cm: d3.mean(cm) };
  }
  function initFit() {
    const r = rng(5);
    fit = {
      A: d3.range(P).map(() => d3.range(rank).map(() => r.normal(0, 0.5))),
      B: d3.range(G).map(() => d3.range(rank).map(() => r.normal(0, 0.5))),
      D: d3.range(C).map(() => d3.range(rank).map(() => 1 + r.normal(0, 0.1))),
    };
    hist = [];
  }
  const predict = (p, cc) => d3.range(G).map((g) => { let s = 0; for (let k = 0; k < rank; k++) s += fit.A[p][k] * fit.B[g][k] * fit.D[cc][k]; return s; });
  // one masked ALS update of factor matrix F, where entries(i) yields [feature vector, target] pairs
  function solveRows(F, entries) {
    F.forEach((row, i) => {
      const AtA = d3.range(rank).map(() => new Array(rank).fill(0)), Atb = d3.range(rank).map(() => [0]);
      entries(i, (z, y) => { for (let u = 0; u < rank; u++) { Atb[u][0] += z[u] * y; for (let w = 0; w < rank; w++) AtA[u][w] += z[u] * z[w]; } });
      for (let u = 0; u < rank; u++) AtA[u][u] += LAMBDA;
      const x = cholSolve(AtA, Atb); for (let u = 0; u < rank; u++) row[u] = x[u][0];
    });
  }
  const had = (u, v) => u.map((x, k) => x * v[k]);
  function sweep() {
    const { A, B, D } = fit;
    solveRows(A, (p, add) => { for (let cc = 0; cc < C; cc++) if (M[p][cc]) for (let g = 0; g < G; g++) add(had(B[g], D[cc]), Y[p][cc][g]); });
    solveRows(B, (g, add) => { for (let p = 0; p < P; p++) for (let cc = 0; cc < C; cc++) if (M[p][cc]) add(had(A[p], D[cc]), Y[p][cc][g]); });
    solveRows(D, (cc, add) => { for (let p = 0; p < P; p++) if (M[p][cc]) for (let g = 0; g < G; g++) add(had(A[p], B[g]), Y[p][cc][g]); });
  }
  function evalFit() {
    const rs = [];
    let se = 0, n = 0;
    for (let p = 0; p < P; p++) for (let cc = 0; cc < C; cc++) {
      const pr = predict(p, cc);
      if (M[p][cc]) { pr.forEach((v, g) => { se += (v - Y[p][cc][g]) ** 2; n++; }); } else rs.push({ p, c: cc, r: pearson(pr, Y[p][cc]) });
    }
    return { rs, rmse: Math.sqrt(se / n), mean: rs.length ? d3.mean(rs, (d) => d.r) : NaN };
  }

  function rebuild() { Y = simulate(noise); M = makeMask(frac); base = baselines(); sel = null; refit(); }
  function refit() {
    if (stop) stop();
    initFit(); hist.push({ x: 0, y: evalFit().mean }); paint();
    let s = 0;
    stop = every(60, () => { sweep(); s++; hist.push({ x: s, y: evalFit().mean }); paint(); if (s >= SWEEPS) { stop(); stop = null; } });
  }
  function paint() {
    const ev = evalFit(), rmap = new Map(ev.rs.map((d) => [d.p + ',' + d.c, d.r]));
    if (!sel && ev.rs.length) sel = { p: ev.rs[0].p, c: ev.rs[0].c };
    const cells = []; for (let p = 0; p < P; p++) for (let cc = 0; cc < C; cc++) cells.push({ p, c: cc, obs: M[p][cc], r: rmap.get(p + ',' + cc) });
    hmG.selectAll('rect').data(cells).join('rect').attr('x', (d) => d.p * cell).attr('y', (d) => d.c * 20).attr('width', cell - 1.5).attr('height', 18.5).attr('rx', 2)
      .attr('fill', (d) => (d.obs ? css('--surface-2') : colorR(d.r)))
      .attr('stroke', (d) => (sel && d.p === sel.p && d.c === sel.c ? css('--ink') : 'none')).attr('stroke-width', 2)
      .style('cursor', (d) => (d.obs ? 'default' : 'pointer'))
      .on('click', (e, d) => { if (!d.obs) { sel = { p: d.p, c: d.c }; paint(); } })
      .on('mousemove', (e, d) => tip(e, `perturbation ${d.p + 1} in ${CTX[d.c]}<br>${d.obs ? 'measured (training)' : `held out · r = <b>${d.r.toFixed(2)}</b>`}`)).on('mouseleave', untip);
    hmG.selectAll('circle.obs').data(cells.filter((d) => d.obs)).join('circle').attr('class', 'obs').attr('cx', (d) => d.p * cell + (cell - 1.5) / 2).attr('cy', (d) => d.c * 20 + 9.25).attr('r', 1.6).attr('fill', muted()).attr('pointer-events', 'none');
    // curves
    F2.layer('base').selectAll('path').data([[base.pm, cat(1)], [base.cm, cat(2)]]).join('path').attr('fill', 'none').attr('stroke', (d) => d[1]).attr('stroke-width', 2).attr('stroke-dasharray', '5 4')
      .attr('d', (d) => d3.line()([[F2.x(0), F2.y(d[0])], [F2.x(SWEEPS), F2.y(d[0])]]));
    F2.layer('curve').selectAll('path').data([hist]).join('path').attr('fill', 'none').attr('stroke', cat(0)).attr('stroke-width', 2)
      .attr('d', d3.line().x((d) => F2.x(d.x)).y((d) => F2.y(Math.max(-0.2, d.y))));
    // selected scatter
    if (sel) {
      const pr = predict(sel.p, sel.c), tr = Y[sel.p][sel.c];
      const others = d3.range(C).filter((k) => M[sel.p][k]);
      const pm = d3.range(G).map((g) => d3.mean(others, (k) => Y[sel.p][k][g]));
      const cl = (v) => Math.max(-6, Math.min(6, v));
      F3.layer('diag').selectAll('line').data([0]).join('line').attr('x1', F3.x(-6)).attr('y1', F3.y(-6)).attr('x2', F3.x(6)).attr('y2', F3.y(6)).attr('stroke', muted()).attr('stroke-dasharray', '4 4');
      const pts = [...tr.map((t, g) => ({ t, v: pm[g], k: 1, g })), ...tr.map((t, g) => ({ t, v: pr[g], k: 0, g }))];
      F3.layer('pts').selectAll('circle').data(pts).join('circle').attr('cx', (d) => F3.x(cl(d.t))).attr('cy', (d) => F3.y(cl(d.v))).attr('r', 4)
        .attr('fill', (d) => cat(d.k)).attr('opacity', (d) => (d.k ? 0.6 : 0.9)).attr('stroke', css('--surface')).attr('stroke-width', 1)
        .on('mousemove', (e, d) => tip(e, `gene ${d.g + 1}<br>true ${d.t.toFixed(2)}<br>${d.k ? 'perturbation-mean' : 'factorised'} ${d.v.toFixed(2)}`)).on('mouseleave', untip);
      F3.layer('title').selectAll('text').data([sel]).join('text').attr('x', 6).attr('y', 12).attr('font-size', 12).attr('fill', ink2())
        .text((d) => `perturbation ${d.p + 1} in ${CTX[d.c]} · r = ${pearson(pr, tr).toFixed(2)} vs ${pearson(pm, tr).toFixed(2)}`);
    }
    readout.set({
      'held-out combos': ev.rs.length,
      'factorised: mean held-out r': ev.mean.toFixed(3),
      'perturbation-mean baseline': base.pm.toFixed(3),
      'cell-line-mean baseline': base.cm.toFixed(3),
      'training RMSE': ev.rmse.toFixed(3),
      'ALS sweep': `${hist.length - 1} / ${SWEEPS}`,
    });
  }
  rebuild();

  section(main, 'What to notice', `<ul>
    <li>The <b>cell-line-mean</b> baseline (average response of other perturbations in that cell line) is poor: perturbations are different from each other, and it ignores which one you ran.</li>
    <li>The <b>perturbation-mean</b> baseline (same perturbation, averaged over the cell lines where it was measured) is strong, because most of a perturbation's effect is shared across contexts. Beating it is the real test. The factorised model can, because it also learns how strongly each cell line expresses each programme.</li>
    <li>Set <b>rank = 1</b>: the model can only scale one programme per context, and held-out r drops. Set the rank above the true value of 3 with little data: training RMSE keeps falling while held-out r stops improving or drops, which is <a href="#/concepts/bias-variance">overfitting</a>.</li>
    <li>Lower <b>fraction measured</b> towards 20%: some perturbations are seen in only one cell line, and the context factors become poorly determined. Data coverage, not model cleverness, sets the ceiling.</li>
  </ul>`);

  steps(main, 'How it works', [
    'Arrange every measured experiment as entries (perturbation p, readout g, context c) → value. Unmeasured combinations are simply missing.',
    'Give each perturbation, each readout and each context its own vector of R numbers (an embedding).',
    'Model a value as the combination of the three vectors. Here: the sum over k of A[p,k]·B[g,k]·D[c,k] (a CP decomposition).',
    'Fit only on the observed entries. Alternating least squares fixes two factor matrices and solves a small ridge regression for the third, then rotates.',
    'Predict a missing (p, c) slice from the learned factors, and evaluate on combinations that were held out.',
  ]);
  math(main, 'Ŷ[p, g, c] = Σ_k  A[p,k] · B[g,k] · D[c,k]\nmin  Σ_(p,g,c observed) (Y − Ŷ)²  +  λ(‖A‖² + ‖B‖² + ‖D‖²)   (alternate: solve A | B,D; then B | A,D; then D | A,B)');

  realityCheck(main, `<ul>
    <li>LPM's reported results are the <b>authors' own</b> (${cite('lpm', 'arXiv 2503.23535')}). Our brief has not found an independent head-to-head of LPM against the simple baselines on this page.</li>
    <li>The broader evidence favours exactly this kind of structured, low-capacity model: ${cite('ahlmann', 'Ahlmann-Eltze et al. 2025')} found that a linear model using embeddings pretrained on <i>perturbation data</i> did best, and that nothing beat the training mean on unseen single perturbations.</li>
    <li>Transferring to a <b>new context</b> is the hard part, and the ${cite('vcc2026', 'Virtual Cell Challenge 2026')} tests it directly (zero-shot on 6 new cell lines). Winners had not been announced as of 2026-11-12.</li>
    <li>A factorisation can only fill in a context whose factor it has estimated. A cell line with no measurements at all needs another source of information (expression of the line at baseline, a cell embedding), which is where the claims of large models are hardest to check.</li>
  </ul>`);

  useIt(main, `<ul>
    <li>If you have several Perturb-seq or drug-screen datasets that share some perturbations but differ in cell type, pool them as a (perturbation × gene × context) tensor and try a low-rank factorisation (e.g. tensor completion, or simply a linear model with perturbation and context embeddings) before a large model.</li>
    <li>Always compare against the <b>perturbation-mean</b> baseline: "the effect seen in other cell lines". In practice it is the number a biologist would guess, and it is often hard to beat.</li>
    <li>The learned context factors are interpretable: which cell lines over- or under-express each programme. That is a hypothesis about context dependence (e.g. a pathway only active in T cells) you can test.</li>
  </ul>`);

  seeAlso(main, [['frontier', 'perturbation-arena', 'Perturbation baselines arena'], ['frontier', 'in-context', 'In-context learning on cells'], ['frontier', 'metric-gaming', 'VCC metrics'],
    ['concepts', 'embeddings', 'Embeddings'], ['concepts', 'pca', 'PCA'], ['concepts', 'regularization', 'Regularization'], ['concepts', 'bias-variance', 'Under- vs overfitting'], ['omics', 'bulk-deconvolution', 'Cell-type deconvolution']]);
}
