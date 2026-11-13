// Sequence -> expression (AlphaGenome-style idea, DeepMind 2025): a tiny 1-D CNN learns DNA motifs from mock sequences,
// trained live in the browser; filters shown as a logo/heatmap; in-silico mutagenesis (ISM) per position.
import { header, figure, slider, select, button, stats, section, steps, math, legend, seeAlso, animate, tip, untip, el, html } from '../lib/ui.js';
import { frame, cat, css, ink, ink2, muted, diverging } from '../lib/plot.js';
import { rng, range } from '../lib/data.js';
import { sourceCard, toyBadge, coreIdea, realityCheck, useIt, cite, pearson } from './_shared.js';
const d3 = window.d3;

const BASES = 'ACGT', L = 50, NTR = 800, NTE = 200, WID = 8, MAXEP = 150;
const ACT = 'TGACTC', REP = 'CCGCGG', A_EFF = 1.5, R_EFF = -1.0, NOISE = 0.2;
/** Binding strength of the best site: 1 for a perfect match, 2/3 with one mismatch, 1/3 with two, else 0. */
export function site(s, mot) {
  let best = 0;
  for (let p = 0; p + mot.length <= s.length; p++) { let m = 0; for (let k = 0; k < mot.length; k++) if (s[p + k] === mot[k]) m++; if (m > best) best = m; }
  return Math.max(0, best - 3) / 3;
}
const truthFn = (s) => A_EFF * site(s, ACT) + R_EFF * site(s, REP);

export function makeData() {
  const r = rng(31);
  return range(NTR + NTE).map(() => {
    const b = range(L).map(() => BASES[r.int(4)]);
    const hasA = r() < 0.5, hasR = r() < 0.45;
    let pa = -99;
    if (hasA) { pa = r.int(L - ACT.length + 1); for (let k = 0; k < ACT.length; k++) b[pa + k] = ACT[k]; }
    if (hasR) {
      let pr; do { pr = r.int(L - REP.length + 1); } while (Math.abs(pr - pa) < 7);
      for (let k = 0; k < REP.length; k++) b[pr + k] = REP[k];
    }
    const s = b.join('');
    return { s, idx: Array.from(s, (c) => BASES.indexOf(c)), y: truthFn(s) + r.normal(0, NOISE) };
  });
}

/** Conv(F filters, width 8) -> soft global max pool -> linear. Parameters in flat arrays; Adam. */
export function makeModel(F, seed, opt = {}) {
  const r = rng(seed);
  const P = { W: Float64Array.from({ length: F * WID * 4 }, () => r.normal(0, opt.wsd ?? 0.3)), c: new Float64Array(F).fill(opt.c0 ?? 0.1), v: Float64Array.from({ length: F }, () => r.normal(0, 0.5)), b: new Float64Array(1) };
  const keys = Object.keys(P), M = {}, V = {}, Gr = {};
  keys.forEach((k) => { M[k] = new Float64Array(P[k].length); V[k] = new Float64Array(P[k].length); Gr[k] = new Float64Array(P[k].length); });
  let t = 0;
  const NPOS = L - WID + 1;
  const beta = opt.beta ?? 8;
  /** Soft max-pool (log-mean-exp, temperature beta): every window gets gradient in proportion to how well it matches. */
  function fwd(idx) {
    const h = new Float64Array(F), att = [];
    for (let f = 0; f < F; f++) {
      const off = f * WID * 4, sc = new Float64Array(NPOS); let mx = -Infinity;
      for (let p = 0; p < NPOS; p++) { let s = P.c[f]; for (let k = 0; k < WID; k++) s += P.W[off + k * 4 + idx[p + k]]; sc[p] = s; if (s > mx) mx = s; }
      let Z = 0; for (let p = 0; p < NPOS; p++) { sc[p] = Math.exp(beta * (sc[p] - mx)); Z += sc[p]; }
      for (let p = 0; p < NPOS; p++) sc[p] /= Z;
      h[f] = mx + Math.log(Z / NPOS) / beta; att.push(sc);
    }
    let y = P.b[0]; for (let f = 0; f < F; f++) y += P.v[f] * h[f];
    return { y, h, att };
  }
  function trainEpoch(data, rr, lr = 0.03, B = 30) {
    const order = rr.shuffle(range(data.length));
    for (let s0 = 0; s0 < order.length; s0 += B) {
      keys.forEach((k) => Gr[k].fill(0));
      const batch = order.slice(s0, s0 + B);
      for (const i of batch) {
        const d = data[i], o = fwd(d.idx), dy = (2 * (o.y - d.y)) / batch.length;
        Gr.b[0] += dy;
        for (let f = 0; f < F; f++) {
          Gr.v[f] += dy * o.h[f];
          const g = dy * P.v[f], off = f * WID * 4, a = o.att[f]; Gr.c[f] += g;
          for (let p = 0; p < NPOS; p++) { const ga = g * a[p]; if (ga === 0) continue; for (let k = 0; k < WID; k++) Gr.W[off + k * 4 + d.idx[p + k]] += ga; }
        }
      }
      t++; const c1 = 1 - 0.9 ** t, c2 = 1 - 0.999 ** t;
      keys.forEach((k) => { const p = P[k], m = M[k], v = V[k], g = Gr[k]; for (let q = 0; q < p.length; q++) { m[q] = 0.9 * m[q] + 0.1 * g[q]; v[q] = 0.999 * v[q] + 0.001 * g[q] * g[q]; p[q] -= lr * (m[q] / c1) / (Math.sqrt(v[q] / c2) + 1e-8); } });
    }
  }
  const predict = (idx) => fwd(idx).y;
  const filter = (f) => range(4).map((bse) => range(WID).map((k) => P.W[f * WID * 4 + k * 4 + bse]));
  return { F, P, fwd, predict, trainEpoch, filter };
}

function evalSet(model, data) {
  const pr = data.map((d) => model.predict(d.idx)), ys = data.map((d) => d.y);
  return { mse: d3.mean(pr, (p, i) => (p - ys[i]) ** 2), r: pearson(pr, ys) };
}

/** Draw one filter: heatmap (rows A,C,G,T) and a logo of the positive, column-centred weights above it. */
function drawFilter(parent, Wf, title, vSign) {
  const cell = 26, left = 18, top = 66, width = left + WID * cell + 6, height = top + 4 * cell + 22;
  const svg = d3.select(parent).append('svg').attr('viewBox', `0 0 ${width} ${height}`).attr('width', width);
  svg.append('text').attr('x', 0).attr('y', 12).attr('font-size', 12).attr('font-weight', 600).attr('fill', ink()).text(title);
  svg.append('text').attr('x', width - 4).attr('y', 12).attr('text-anchor', 'end').attr('font-size', 11).attr('fill', ink2()).text(vSign);
  const m = Math.max(0.3, d3.max(Wf.flat(), (v) => Math.abs(v)));
  const col = diverging(m);
  const g = svg.append('g').attr('transform', `translate(${left},${top})`);
  Wf.forEach((row, b) => {
    svg.append('text').attr('x', left - 5).attr('y', top + b * cell + cell / 2 + 4).attr('text-anchor', 'end').attr('font-size', 11).attr('fill', ink2()).text(BASES[b]);
    row.forEach((v, k) => g.append('rect').attr('x', k * cell).attr('y', b * cell).attr('width', cell - 1).attr('height', cell - 1).attr('rx', 2).attr('fill', col(v))
      .on('mousemove', (e) => tip(e, `${title}, position ${k + 1}, base ${BASES[b]}<br>weight ${v.toFixed(2)}`)).on('mouseleave', untip));
  });
  // logo: heights = weight minus column mean, positive part only, stacked small to large
  const lg = svg.append('g').attr('transform', `translate(${left},${top - 4})`);
  const maxH = 46;
  const colH = range(WID).map((k) => { const mu = d3.mean(Wf, (r) => r[k]); return Wf.map((r, b) => ({ b, h: Math.max(0, r[k] - mu) })); });
  const scaleH = maxH / Math.max(0.5, d3.max(colH, (c) => d3.sum(c, (d) => d.h)));
  colH.forEach((c, k) => {
    let y = 0;
    c.slice().sort((a, b) => a.h - b.h).forEach((d) => {
      const hh = d.h * scaleH; if (hh < 4) return;
      lg.append('text').attr('font-family', 'var(--mono)').attr('font-weight', 700).attr('font-size', 20).attr('fill', ink()).attr('text-anchor', 'middle')
        .attr('transform', `translate(${k * cell + cell / 2},${-y}) scale(1.15,${hh / 14.5})`).text(BASES[d.b]);
      y += hh;
    });
  });
}

export default async function render(main) {
  header(main, 'Sequence → expression (AlphaGenome)',
    'A convolutional network can read DNA directly: each filter slides along the sequence and fires where it matches a pattern. Train one on sequences whose expression depends on hidden <b>motifs</b>, and the filters <b>rediscover the motifs</b>; mutate each base in silico and the model tells you <b>which positions matter</b>.');

  sourceCard(main, {
    org: 'Google DeepMind (AlphaGenome)',
    when: 'June 2025',
    links: [],
    claim: 'AlphaGenome predicts RNA and other genomic tracks directly from DNA sequence, with an input window of up to 1 Mb. Our brief records no URL for it, and its Nature publication is listed as unverified, so we cite no link here.',
    status: 'no link in our brief',
  });

  coreIdea(main, `<p>Gene expression is partly written in the DNA around a gene: transcription factors bind short <b>motifs</b> (6–20 bases), and the combination of motifs present helps set how much RNA is made. A <b>1-D convolutional neural network</b> (CNN) is a natural reader: each filter is a small weight matrix (4 bases × w positions) that slides along the one-hot encoded sequence, and a high score means "this window looks like my motif". This is the same idea as a position weight matrix (PWM), but learned from expression data rather than from binding assays.</p>
  <p><b>In-silico mutagenesis (ISM)</b> interprets the trained model: change one base at a time, re-predict, and record the change. Positions where mutations swing the prediction are the ones the model relies on.</p>
  <p><b>Key ML ingredient:</b> convolution over sequence (weight sharing across positions) plus perturbation-based attribution. Models such as AlphaGenome scale this to very long inputs (up to 1 Mb, per DeepMind) and many output tracks; this page shrinks it to a couple of filters on 50-base mock sequences.</p>`);

  // ---------------------------------------------------------------- figure 1: train
  const DATA = makeData(), TR = DATA.slice(0, NTR), TE = DATA.slice(NTR);
  const fig = figure(main, 'Train a tiny CNN on mock promoters',
    `${NTR} training and ${NTE} test sequences, ${L} bases each (motifs planted in about half). Hidden rule: expression = ${A_EFF} × (activator <code>${ACT}</code> site strength) − ${Math.abs(R_EFF)} × (repressor <code>${REP}</code> site strength) + noise (sd ${NOISE}). Site strength is 1 for the best perfect match, 2/3 with one mismatch, 1/3 with two. Model: filters of width ${WID} → soft maximum over positions → weighted sum, trained with Adam in your browser.`);
  toyBadge(fig, `Not AlphaGenome: a few filters and ${L} bases instead of a deep network and 1 Mb.`);
  const S = { F: 2, seed: 6 };
  let model, ep, hist, rTrain, stop = null;
  select(fig.controls, { label: 'Filters', value: '2', options: [['1', '1 filter'], ['2', '2 filters'], ['4', '4 filters']], onChange: (v) => { S.F = +v; reset(); run(); } });
  slider(fig.controls, { label: 'Init seed', min: 1, max: 9, value: S.seed, onInput: (v) => { S.seed = v; reset(); run(); } });
  const runBtn = button(fig.controls, 'Pause', () => { if (stop) { halt(); } else run(); }, true);
  button(fig.controls, 'Reset', () => { reset(); paintAll(); });
  const readout = stats(fig.root);
  const row = el('div', { class: 'fig-row' }); fig.body.append(row);
  const left = el('div'), right = el('div', { style: 'flex:1 1 380px;min-width:0' }); row.append(left, right);
  legend(left, [['training MSE', cat(0)], ['test MSE', cat(1)]]);
  const LC = frame(left, { width: 460, height: 320, x: [0, MAXEP], y: [0, 1.0], xLabel: 'epoch', yLabel: 'mean squared error' });
  LC.g.append('line').attr('x1', 0).attr('x2', LC.w).attr('y1', LC.y(NOISE * NOISE)).attr('y2', LC.y(NOISE * NOISE)).attr('stroke', muted()).attr('stroke-dasharray', '4 3');
  LC.g.append('text').attr('x', LC.w - 4).attr('y', LC.y(NOISE * NOISE) + 14).attr('text-anchor', 'end').attr('font-size', 11).attr('fill', ink2()).text('noise floor');
  right.append(html(`<div style="font-size:13px;color:var(--ink-2);margin:0 0 4px">Learned filters (logo = weights above the column mean). Planted motifs, for comparison: activator <code>${ACT}</code>, repressor <code>${REP}</code>.</div>`));
  const filtBox = el('div', { style: 'display:flex;flex-wrap:wrap;gap:10px 18px' }); right.append(filtBox);

  function reset() { halt(); model = makeModel(S.F, S.seed); ep = 0; hist = []; rTrain = rng(100 + S.seed); record(); }
  function record() { const a = evalSet(model, TR), b = evalSet(model, TE); hist.push({ ep, tr: a.mse, te: b.mse, r: b.r }); }
  function halt() { if (stop) { stop(); stop = null; } runBtn.textContent = 'Train'; }
  function paintCurve() {
    [['tr', cat(0)], ['te', cat(1)]].forEach(([k, c]) => {
      LC.layer('l' + k).selectAll('path').data([hist]).join('path').attr('fill', 'none').attr('stroke', c).attr('stroke-width', 2)
        .attr('d', d3.line().x((d) => LC.x(d.ep)).y((d) => LC.y(Math.min(1.0, d[k]))));
    });
    const h = hist[hist.length - 1];
    readout.set({ epoch: `${ep} / ${MAXEP}`, 'train MSE': h.tr.toFixed(3), 'test MSE': h.te.toFixed(3), 'test Pearson r': h.r.toFixed(3) });
  }
  function paintFilters() {
    filtBox.innerHTML = '';
    range(model.F).forEach((f) => {
      const d = el('div', { style: 'width:230px;max-width:100%' }); filtBox.append(d);
      const v = model.P.v[f];
      drawFilter(d, model.filter(f), `filter ${f + 1}`, `output weight ${v >= 0 ? '+' : ''}${v.toFixed(2)} (${v >= 0 ? 'activating' : 'repressing'})`);
    });
  }
  function paintAll() { paintCurve(); paintFilters(); paintISM(); }
  function run() {
    halt(); if (ep >= MAXEP) reset();
    runBtn.textContent = 'Pause';
    let fr = 0;
    stop = animate(() => {
      for (let k = 0; k < 2 && ep < MAXEP; k++) { model.trainEpoch(TR, rTrain); ep++; record(); }
      paintCurve();
      if (++fr % 6 === 0 || ep >= MAXEP) { paintFilters(); paintISM(); }
      if (ep >= MAXEP) { halt(); return false; }
      return true;
    });
  }

  // ---------------------------------------------------------------- figure 2: ISM
  const fig2 = figure(main, 'In-silico mutagenesis: which bases does the model use?',
    'Each cell: predicted expression after changing that position to that base, minus the prediction for the current sequence (red = higher, blue = lower). The bar track above shows the largest absolute change per position. Brackets under the sequence mark planted motifs. <b>Click a cell</b> to make that mutation and watch the prediction and the truth update.');
  const withBoth = TE.map((d, i) => i).filter((i) => TE[i].s.includes(ACT) && TE[i].s.includes(REP));
  let seqI = withBoth[0], cur = TE[seqI].s;
  const sSl = slider(fig2.controls, { label: 'Test sequence', min: 0, max: NTE - 1, value: seqI, fmt: (v) => `#${v + 1}`, onInput: (v) => { seqI = v; cur = TE[v].s; paintISM(); } });
  button(fig2.controls, 'Next with both motifs', () => { const nx = withBoth.find((i) => i > seqI) ?? withBoth[0]; seqI = nx; sSl.set(nx); cur = TE[nx].s; paintISM(); });
  button(fig2.controls, 'Undo mutations', () => { cur = TE[seqI].s; paintISM(); });
  const readout2 = stats(fig2.root);
  const cellW = 14, cellH = 16, lab = 22, trackH = 46, topY = trackH + 8;
  const ismW = lab + L * cellW + 4, ismH = topY + 4 * cellH + 50;
  const ismSvg = d3.select(fig2.body).append('svg').attr('viewBox', `0 0 ${ismW} ${ismH}`).attr('width', ismW).style('max-width', '100%');
  legend(fig2.body, [['planted activator', cat(2)], ['planted repressor', cat(3)]]);

  function paintISM() {
    if (!model) return;
    const idx = Array.from(cur, (c) => BASES.indexOf(c));
    const base = model.predict(idx);
    const M = range(4).map(() => new Array(L).fill(0));
    for (let p = 0; p < L; p++) for (let b = 0; b < 4; b++) {
      if (b === idx[p]) continue;
      const old = idx[p]; idx[p] = b; M[b][p] = model.predict(idx) - base; idx[p] = old;
    }
    const imp = range(L).map((p) => d3.max(M, (r) => Math.abs(r[p])));
    const mAbs = Math.max(0.3, d3.max(imp));
    const col = diverging(mAbs);
    ismSvg.selectAll('*').remove();
    const hy = d3.scaleLinear().domain([0, mAbs]).range([0, trackH]);
    const g = ismSvg.append('g').attr('transform', `translate(${lab},0)`);
    g.selectAll('rect.imp').data(imp).join('rect').attr('class', 'imp').attr('x', (_, p) => p * cellW + 1).attr('width', cellW - 2)
      .attr('y', (v) => trackH - hy(v)).attr('height', (v) => hy(v)).attr('fill', cat(0)).attr('rx', 1)
      .on('mousemove', (e, v) => tip(e, `largest |Δ| at this position: ${v.toFixed(2)}`)).on('mouseleave', untip);
    ismSvg.append('text').attr('x', 0).attr('y', 12).attr('font-size', 10).attr('fill', ink2()).text('|Δ|');
    range(4).forEach((b) => {
      ismSvg.append('text').attr('x', lab - 5).attr('y', topY + b * cellH + cellH / 2 + 4).attr('text-anchor', 'end').attr('font-size', 11).attr('fill', ink2()).text(BASES[b]);
      for (let p = 0; p < L; p++) {
        const isCur = b === idx[p];
        g.append('rect').attr('x', p * cellW).attr('y', topY + b * cellH).attr('width', cellW - 1).attr('height', cellH - 1).attr('rx', 1.5)
          .attr('fill', isCur ? css('--surface-2') : col(M[b][p])).attr('stroke', isCur ? muted() : 'none').attr('stroke-width', isCur ? 0.8 : 0)
          .style('cursor', isCur ? 'default' : 'pointer')
          .on('mousemove', (e) => tip(e, isCur ? `position ${p + 1}: current base ${BASES[b]}` : `position ${p + 1}: ${BASES[idx[p]]}→${BASES[b]}<br>Δ prediction ${M[b][p] >= 0 ? '+' : ''}${M[b][p].toFixed(2)}<br>click to apply`))
          .on('mouseleave', untip)
          .on('click', () => { if (isCur) return; untip(); cur = cur.slice(0, p) + BASES[b] + cur.slice(p + 1); paintISM(); });
      }
    });
    const sy = topY + 4 * cellH + 14;
    for (let p = 0; p < L; p++) g.append('text').attr('x', p * cellW + cellW / 2).attr('y', sy).attr('text-anchor', 'middle').attr('font-size', 11).attr('font-family', 'var(--mono)')
      .attr('fill', cur[p] === TE[seqI].s[p] ? ink() : css('--bad')).attr('font-weight', cur[p] === TE[seqI].s[p] ? 400 : 700).text(cur[p]);
    // brackets for planted motifs in the original sequence
    const orig = TE[seqI].s;
    [[ACT, cat(2)], [REP, cat(3)]].forEach(([mot, c]) => {
      let at = orig.indexOf(mot);
      while (at >= 0) {
        g.append('rect').attr('x', at * cellW + 1).attr('y', sy + 6).attr('width', mot.length * cellW - 2).attr('height', 5).attr('rx', 2).attr('fill', c);
        at = orig.indexOf(mot, at + 1);
      }
    });
    const nMut = range(L).filter((p) => cur[p] !== orig[p]).length;
    readout2.set({ 'model prediction': base.toFixed(2), 'true rule (no noise)': truthFn(cur).toFixed(2), 'mutations applied': nMut,
      'motifs present now': `${cur.includes(ACT) ? 'activator' : ''}${cur.includes(ACT) && cur.includes(REP) ? ' + ' : ''}${cur.includes(REP) ? 'repressor' : ''}` || 'none' });
  }

  reset(); paintAll(); run();

  section(main, 'What to notice', `<ul>
    <li>Within about ten epochs the test error falls from about 0.9 to about 0.1, most of the way to the dashed noise floor (${NOISE}² = ${(NOISE * NOISE).toFixed(2)}); test Pearson r ends near 0.89. One filter's logo spells the activator <code>${ACT}</code> and gets a positive output weight; the other spells the repressor <code>${REP}</code> and gets a negative one. Nobody told the model the motifs; they are the simplest explanation of the expression values.</li>
    <li>Filters are 8 wide but the motifs are 6 long, so one or two columns of each logo stay flat: the model only learns what the data constrain. The motif can also sit shifted by a column inside the filter (look at the repressor filter).</li>
    <li>With <b>1 filter</b>, the model can learn only one motif (with seed 6, the repressor), and the test error stalls near 0.37 with r below 0.5. With <b>4 filters</b>, one extra filter learns a shifted copy of the activator and another learns weak noise, while the training error drops below the test error: extra capacity is not extra biology.</li>
    <li>Move the <b>Init seed</b> slider: some seeds get stuck with both filters on parts of the same motif. Gradient descent on motif-finding has local minima, which is why real models use many more filters than motifs.</li>
    <li>In the ISM map, large changes cluster under the coloured brackets; mutations elsewhere barely matter. Breaking one base of the activator turns a perfect site into a one-mismatch site (true rule: strength 1 → 2/3, so expression falls by 0.5). Click a cell inside the activator and compare the model's prediction with the true rule.</li>
    <li>Try to <i>create</i> a site by clicking bases elsewhere: the model responds whenever a window matches its filter well enough, which is how real models predict the effect of variants that create new binding sites.</li>
  </ul>`);

  steps(main, 'How it works', [
    'One-hot encode the DNA: each base becomes a 4-vector (A, C, G, T), so a sequence of length L is a 4 × L matrix.',
    'Each convolution filter is a 4 × w weight matrix. Slide it along the sequence and, at every position, sum the weights of the bases present: a motif match gives a high score.',
    'Pooling over positions asks "how well does this motif occur anywhere?". We use a soft maximum (log-mean-exp) so every window gets some gradient early in training; a hard max only updates the single best window and gets stuck more often. Real models use many layers and dilated convolutions or attention so they can combine motifs over long distances.',
    'A final linear layer combines filter outputs into predicted expression; train all weights end to end by gradient descent on squared error.',
    'Interpret by in-silico mutagenesis: for every position and alternative base, re-predict and record the change. Average or take the maximum over bases to get a per-position importance track.',
  ]);
  math(main, 'score_f(p) = c_f + Σ_k W_f[k, base(p+k)]          (convolution)\nh_f = (1/β) log mean_p exp(β score_f(p)),  β = 8   (soft max pool)\nŷ = b + Σ_f v_f h_f\nISM(p, b) = ŷ(sequence with base b at p) − ŷ(sequence)');

  realityCheck(main, `<ul>
    <li>Everything we say about AlphaGenome's scope (DNA → RNA tracks, up to 1 Mb input) comes from DeepMind's own description as summarised in our brief. We have <b>no URL</b> for it in our brief, its Nature publication is listed as <b>unverified</b>, and we have no independent benchmark to quote.</li>
    <li>Real regulatory grammar is far messier than two motifs that add up: motif spacing, orientation, chromatin, 3-D contacts and cell type all matter. A model can fit training genes well and still mispredict the effect of a single variant, which is the use case people care about most.</li>
    <li>ISM shows what the <i>model</i> relies on, not what the cell does. If a confounding sequence feature (for example GC content) correlates with expression, ISM will happily highlight it.</li>
    <li>Like other foundation models on this site (for example ${cite('kedzierska', 'the zero-shot critique of single-cell models')}), sequence models deserve comparison with simple baselines such as k-mer counts plus a linear model before you trust their extra complexity.</li>
  </ul>`);

  useIt(main, `<ul>
    <li>For a variant of interest (eQTL, GWAS hit), compare predicted output for reference and alternative alleles; treat the difference as a hypothesis for a reporter assay or CRISPR edit, not a result.</li>
    <li>Run ISM around your gene's promoter and check whether the high-importance positions overlap known motifs (match the filters or ISM peaks against a motif database).</li>
    <li>Always ask whether your cell type and assay are represented in the model's training tracks; predictions for unseen cell types are extrapolations.</li>
  </ul>`);

  seeAlso(main, [['concepts', 'cnn', 'Convolutions (CNNs)'], ['concepts', 'interpretability', 'Feature importance & SHAP'], ['concepts', 'gradient-descent', 'Gradient descent'],
    ['frontier', 'geneformer', 'Geneformer: in-silico deletion'], ['frontier', 'landscape', 'The 2024–26 landscape']]);
}
