// Contrastive learning: a SimCLR-style encoder trained live with the InfoNCE (NT-Xent) loss.
// Figure 1: input space with augmented pairs (left) and embeddings on the unit circle (right).
// Figure 2: one anchor's view of the loss: a softmax over candidates, temperature and number of negatives.
import { header, figure, slider, button, stats, section, steps, callout, math, legend, seeAlso, animate, tip, untip, el } from '../lib/ui.js';
import { frame, cat, css, ink, ink2, muted } from '../lib/plot.js';
import { rng } from '../lib/data.js';
import { mlp } from './autoencoder.js';
const d3 = window.d3;

const K = 6, NPER = 40;
function makeData() { // six blobs on a 3 × 2 grid
  const r = rng(17), out = [];
  for (let k = 0; k < K; k++) {
    const cx = -1 + (k % 3), cy = k < 3 ? 0.55 : -0.55;
    for (let i = 0; i < NPER; i++) out.push({ k, x: [cx + r.normal(0, 0.13), cy + r.normal(0, 0.13)] });
  }
  return out;
}

export default async function render(main) {
  header(main, 'Contrastive learning',
    'No labels, just one rule: <b>two augmented views of the same example should get similar embeddings, and views of different examples should not</b>. Pull positive pairs together, push everything else apart, and an encoder learns which differences matter and which (the augmentations) do not.');

  // =================== Figure 1 ===================
  const data = makeData();
  const fig = figure(main, 'Train a SimCLR-style encoder on the circle',
    `${data.length} points in six hidden groups (colours are for you; the model never sees them). Each step takes a batch of points, makes <b>two augmented views</b> of each by adding random jitter, and passes all views through the encoder (2 → 16 → 16 tanh → 2, then scaled to length 1, so every embedding lives on the unit circle). Left: the input space; hollow circles joined by a line are the two views of each point in the current batch. Right: where the encoder puts every (un-jittered) point; the chords join the current batch's positive pairs; the coloured bars outside the circle are a histogram of embedding angles per group.`);
  let tau = 0.2, B = 32, aug = 0.15, net, stepN = 0, stop = null, batch = [], hist = [], br;
  slider(fig.controls, { label: 'temperature τ', min: -1.3, max: 0, step: 0.05, value: Math.log10(0.2), fmt: (v) => (10 ** v).toFixed(2), onInput: (v) => { tau = 10 ** v; } });
  slider(fig.controls, { label: 'batch size', min: 2, max: 7, step: 1, value: 5, fmt: (v) => 2 ** v, onInput: (v) => { B = 2 ** v; } });
  slider(fig.controls, { label: 'augmentation jitter', min: 0.02, max: 0.8, step: 0.02, value: 0.15, fmt: (v) => v.toFixed(2), onInput: (v) => { aug = v; if (!stop) { makeBatch(); paint(); } } });
  const runBtn = button(fig.controls, 'Train', () => (stop ? pause() : run()), true);
  button(fig.controls, 'Reset', () => reset());
  const ro = stats(fig.root);
  const row = el('div', { class: 'fig-row' }); fig.body.append(row);
  const left = el('div'), right = el('div'); row.append(left, right);
  legend(left, d3.range(K).map((k) => [`hidden group ${k + 1}`, cat(k)]).concat([['augmented pair (batch)', ink()]]));
  const FI = frame(left, { width: 460, height: 440, x: [-1.8, 1.8], y: [-1.8, 1.8], xLabel: 'input x₁', yLabel: 'input x₂' });
  ['pts', 'pairs', 'views'].forEach((n) => FI.layer(n));
  legend(right, [['positive pair in current batch (chord)', ink2()]]);
  const FE = frame(right, { width: 460, height: 440, x: [-1.45, 1.45], y: [-1.45, 1.45], axes: false, grid: false, margin: { top: 10, right: 10, bottom: 10, left: 30 } });
  ['circle', 'hist', 'chords', 'pts'].forEach((n) => FE.layer(n));
  const LC = frame(fig.body, { width: 940, height: 150, x: [0, 600], y: [0, 5], xLabel: 'training step', yLabel: 'InfoNCE loss', margin: { left: 56 } });

  const nrm = (h) => { const n = Math.hypot(h[0], h[1]) || 1e-9; return [h[0] / n, h[1] / n, n]; };
  const embed = (x) => nrm(net.forward(x));
  function reset() { pause(); net = mlp([2, 16, 16, 2], ['tanh', 'tanh', 'linear'], 9); br = rng(23); stepN = 0; hist = []; makeBatch(); paint(); }
  function makeBatch() {
    const idx = br.shuffle(d3.range(data.length)).slice(0, B);
    batch = idx.map((i) => ({ i, k: data[i].k, v: [0, 1].map(() => [data[i].x[0] + br.normal(0, aug), data[i].x[1] + br.normal(0, aug)]) }));
  }
  /** One NT-Xent step on the current batch. Returns the mean loss. */
  function trainStep() {
    const views = []; batch.forEach((b) => b.v.forEach((x) => views.push(x)));
    const n = views.length, E = views.map(embed), pos = (a) => a ^ 1;
    const S = E.map((e) => E.map((f) => (e[0] * f[0] + e[1] * f[1]) / tau));
    const Wt = S.map((row, a) => { let mx = -Infinity; row.forEach((s, j) => { if (j !== a && s > mx) mx = s; }); let Z = 0; const w = row.map((s, j) => (j === a ? 0 : Math.exp(s - mx))); w.forEach((v) => { Z += v; }); return w.map((v) => v / Z); });
    let loss = 0; for (let a = 0; a < n; a++) loss -= Math.log(Wt[a][pos(a)] + 1e-12);
    loss /= n;
    // gradient wrt each embedding e_i: (1/(n τ)) Σ_j [(w_ij − δ_j,p(i)) + (w_ji − δ_i,p(j))] e_j
    for (let i = 0; i < n; i++) {
      const g = [0, 0];
      for (let j = 0; j < n; j++) { if (j === i) continue; const c = Wt[i][j] - (j === pos(i) ? 1 : 0) + Wt[j][i] - (i === pos(j) ? 1 : 0); g[0] += c * E[j][0]; g[1] += c * E[j][1]; }
      g[0] /= n * tau; g[1] /= n * tau;
      const e = embed(views[i]), dot = e[0] * g[0] + e[1] * g[1]; // re-forward so the net caches this view, then back through the normalisation
      net.backward([(g[0] - e[0] * dot) / e[2], (g[1] - e[1] * dot) / e[2]]);
    }
    net.step(0.01, 1); stepN++;
    return loss;
  }
  function run() {
    runBtn.textContent = 'Pause';
    stop = animate(() => {
      const t0 = performance.now(); let l = 0, c = 0;
      while (performance.now() - t0 < 12 && c < 6) { makeBatch(); l += trainStep(); c++; }
      hist.push([stepN, l / c]); if (hist.length > 4000) hist.shift();
      paint(); return true;
    });
  }
  function pause() { if (stop) { stop(); stop = null; } runBtn.textContent = 'Train'; }

  function paint() {
    const E = data.map((d) => embed(d.x));
    // input space
    FI.layer('pts').selectAll('circle').data(data).join('circle').attr('cx', (d) => FI.x(d.x[0])).attr('cy', (d) => FI.y(d.x[1])).attr('r', 4)
      .attr('fill', (d) => cat(d.k)).attr('fill-opacity', 0.55).attr('stroke', css('--surface')).attr('stroke-width', 1);
    const cl = (v) => Math.max(-1.8, Math.min(1.8, v));
    FI.layer('pairs').selectAll('line').data(batch).join('line').attr('x1', (b) => FI.x(cl(b.v[0][0]))).attr('y1', (b) => FI.y(cl(b.v[0][1]))).attr('x2', (b) => FI.x(cl(b.v[1][0]))).attr('y2', (b) => FI.y(cl(b.v[1][1])))
      .attr('stroke', ink()).attr('stroke-width', 1.5);
    FI.layer('views').selectAll('circle').data(batch.flatMap((b) => b.v.map((v) => ({ v, k: b.k })))).join('circle').attr('cx', (d) => FI.x(cl(d.v[0]))).attr('cy', (d) => FI.y(cl(d.v[1]))).attr('r', 4.5)
      .attr('fill', css('--surface')).attr('stroke', (d) => cat(d.k)).attr('stroke-width', 2);
    // embedding circle
    FE.layer('circle').selectAll('circle').data([1]).join('circle').attr('cx', FE.x(0)).attr('cy', FE.y(0)).attr('r', FE.x(1) - FE.x(0)).attr('fill', 'none').attr('stroke', muted()).attr('stroke-width', 1.2);
    const NB = 48, counts = d3.range(K).map(() => new Array(NB).fill(0));
    E.forEach((e, i) => { const b = Math.floor(((Math.atan2(e[1], e[0]) + Math.PI) / (2 * Math.PI)) * NB) % NB; counts[data[i].k][b]++; });
    const bars = [];
    for (let b = 0; b < NB; b++) { let acc = 0; for (let k = 0; k < K; k++) { if (counts[k][b]) { bars.push({ b, k, r0: acc, r1: acc + counts[k][b] }); acc += counts[k][b]; } } }
    const rs = d3.scaleLinear().domain([0, NPER]).range([1.07, 1.42]).clamp(true);
    const arc = d3.arc().innerRadius((d) => FE.x(rs(d.r0)) - FE.x(0)).outerRadius((d) => FE.x(rs(d.r1)) - FE.x(0))
      .startAngle((d) => Math.PI / 2 - (-Math.PI + (d.b + 1) * 2 * Math.PI / NB)).endAngle((d) => Math.PI / 2 - (-Math.PI + d.b * 2 * Math.PI / NB));
    FE.layer('hist').attr('transform', `translate(${FE.x(0)},${FE.y(0)})`).selectAll('path').data(bars).join('path').attr('d', arc).attr('fill', (d) => cat(d.k)).attr('stroke', css('--surface')).attr('stroke-width', 0.5)
      .on('mousemove', (e, d) => tip(e, `group ${d.k + 1}: ${d.r1 - d.r0} points in this ${360 / NB}° bin`)).on('mouseleave', untip);
    const be = batch.map((b) => ({ k: b.k, a: embed(b.v[0]), c: embed(b.v[1]) }));
    FE.layer('chords').selectAll('line').data(be).join('line').attr('x1', (d) => FE.x(d.a[0])).attr('y1', (d) => FE.y(d.a[1])).attr('x2', (d) => FE.x(d.c[0])).attr('y2', (d) => FE.y(d.c[1]))
      .attr('stroke', ink2()).attr('stroke-opacity', 0.55).attr('stroke-width', 1.2);
    FE.layer('pts').selectAll('circle').data(E).join('circle').attr('cx', (e) => FE.x(e[0])).attr('cy', (e) => FE.y(e[1])).attr('r', 4.2)
      .attr('fill', (e, i) => cat(data[i].k)).attr('stroke', css('--surface')).attr('stroke-width', 1)
      .on('mousemove', (ev, e) => tip(ev, `embedding angle ${(Math.atan2(e[1], e[0]) * 180 / Math.PI).toFixed(0)}°`)).on('mouseleave', untip);
    // loss curve
    const xmax = Math.max(600, stepN), lmax = Math.max(1, d3.max(hist, (h) => h[1]) || 1);
    LC.x.domain([0, xmax]); LC.y.domain([0, lmax * 1.05]);
    LC.g.select('.axis.x').call(d3.axisBottom(LC.x).ticks(8)); LC.g.select('.axis.y').call(d3.axisLeft(LC.y).ticks(3));
    LC.layer('c').selectAll('path').data([hist]).join('path').attr('fill', 'none').attr('stroke', cat(0)).attr('stroke-width', 2).attr('d', d3.line().x((h) => LC.x(h[0])).y((h) => LC.y(h[1])));
    // metrics: alignment, uniformity, kNN accuracy of hidden groups (labels used only here, for evaluation)
    const align = d3.mean(be, (d) => (d.a[0] - d.c[0]) ** 2 + (d.a[1] - d.c[1]) ** 2);
    let uni = 0, cnt = 0;
    for (let i = 0; i < E.length; i++) for (let j = i + 1; j < E.length; j++) { uni += Math.exp(-2 * ((E[i][0] - E[j][0]) ** 2 + (E[i][1] - E[j][1]) ** 2)); cnt++; }
    // closest pair of group centre directions (hidden groups used only for grading)
    const ang = d3.range(K).map((k) => { const m = [0, 0]; E.forEach((e, i) => { if (data[i].k === k) { m[0] += e[0]; m[1] += e[1]; } }); return Math.atan2(m[1], m[0]); });
    let minGap = 360;
    for (let a = 0; a < K; a++) for (let b2 = a + 1; b2 < K; b2++) { let d = Math.abs(ang[a] - ang[b2]) * 180 / Math.PI; d = Math.min(d, 360 - d); minGap = Math.min(minGap, d); }
    ro.set({ step: stepN, 'loss (last step)': hist.length ? hist[hist.length - 1][1].toFixed(3) : '–', 'chance loss log(2B−1)': Math.log(2 * B - 1).toFixed(3), 'alignment ‖e₁−e₂‖² (batch)': align.toFixed(3), 'uniformity (lower = more spread)': Math.log(uni / cnt).toFixed(2), 'closest two group centres (ideal 60°)': minGap.toFixed(0) + '°' });
  }
  reset();

  // =================== Figure 2 ===================
  const fig2 = figure(main, "One anchor's view: InfoNCE is a softmax classification",
    'Freeze everything and look at a single anchor (the large dark point). Its loss asks: among the positive and all the negatives in the batch, can the anchor pick out its positive? Each candidate gets probability ∝ exp(similarity / τ). <b>Drag</b> the positive (green ring) or any negative around the circle. Lines from the anchor have width ∝ the candidate\'s probability: that is also how hard the gradient pushes the anchor away from it.');
  let tau2 = 0.2, nNeg = 15, posAng = 20, negAng = [];
  const s1 = slider(fig2.controls, { label: 'temperature τ', min: -1.3, max: 0.3, step: 0.05, value: Math.log10(0.2), fmt: (v) => (10 ** v).toFixed(2), onInput: (v) => { tau2 = 10 ** v; paint2(); } });
  slider(fig2.controls, { label: 'number of negatives', min: 1, max: 63, step: 1, value: nNeg, onInput: (v) => { nNeg = v; makeNeg(); paint2(); } });
  const ro2 = stats(fig2.root);
  const row2 = el('div', { class: 'fig-row' }); fig2.body.append(row2);
  const l2 = el('div'), r2 = el('div'); row2.append(l2, r2);
  legend(l2, [['anchor', ink()], ['positive', cat(2)], ['negative', cat(1)]]);
  const FC = frame(l2, { width: 440, height: 420, x: [-1.35, 1.35], y: [-1.35, 1.35], axes: false, grid: false, margin: { top: 10, right: 20, bottom: 10, left: 20 } });
  ['circle', 'lines', 'pts'].forEach((n) => FC.layer(n));
  legend(r2, [['p(positive)', cat(2)], ['p(negative)', cat(1)]]);
  const PB = frame(r2, { width: 460, height: 420, x: [0, 1], y: [0, 1], xLabel: 'candidates, sorted by similarity to the anchor', yLabel: 'softmax probability', margin: { bottom: 42 } });
  PB.g.select('.axis.x').remove();
  function makeNeg() { const r = rng(5); negAng = d3.range(63).map(() => -180 + 360 * r()).slice(0, nNeg); }
  makeNeg();
  const dragAng = (e) => { const [mx, my] = d3.pointer(e, FC.g.node()); return Math.atan2(FC.y.invert(my), FC.x.invert(mx)) * 180 / Math.PI; };
  function paint2() {
    const rad = (a) => a * Math.PI / 180, pt = (a) => [Math.cos(rad(a)), Math.sin(rad(a))];
    const cands = [{ pos: true, a: posAng, idx: -1 }].concat(negAng.map((a, i) => ({ pos: false, a, idx: i })));
    cands.forEach((c) => { c.s = Math.cos(rad(c.a)); });
    const mx = d3.max(cands, (c) => c.s / tau2), Z = d3.sum(cands, (c) => Math.exp(c.s / tau2 - mx));
    cands.forEach((c) => { c.p = Math.exp(c.s / tau2 - mx) / Z; });
    FC.layer('circle').selectAll('circle').data([1]).join('circle').attr('cx', FC.x(0)).attr('cy', FC.y(0)).attr('r', FC.x(1) - FC.x(0)).attr('fill', 'none').attr('stroke', muted()).attr('stroke-width', 1.2);
    FC.layer('lines').selectAll('line').data(cands).join('line').attr('x1', FC.x(1)).attr('y1', FC.y(0)).attr('x2', (c) => FC.x(pt(c.a)[0])).attr('y2', (c) => FC.y(pt(c.a)[1]))
      .attr('stroke', (c) => (c.pos ? cat(2) : cat(1))).attr('stroke-opacity', 0.75).attr('stroke-width', (c) => 0.4 + 14 * c.p);
    const drag = d3.drag().on('drag', (e, c) => { const a = dragAng(e.sourceEvent); if (c.pos) posAng = a; else negAng[c.idx] = a; paint2(); });
    FC.layer('pts').selectAll('circle.c').data(cands, (c) => c.idx).join('circle').attr('class', 'c').attr('cx', (c) => FC.x(pt(c.a)[0])).attr('cy', (c) => FC.y(pt(c.a)[1]))
      .attr('r', (c) => (c.pos ? 9 : 6)).attr('fill', (c) => (c.pos ? css('--surface') : cat(1))).attr('stroke', (c) => (c.pos ? cat(2) : css('--surface'))).attr('stroke-width', (c) => (c.pos ? 3.5 : 1.5))
      .style('cursor', 'grab').call(drag)
      .on('mousemove', (e, c) => tip(e, `${c.pos ? 'positive' : 'negative'}<br>cosine similarity ${c.s.toFixed(2)}<br>probability ${c.p.toFixed(3)}`)).on('mouseleave', untip);
    FC.layer('pts').selectAll('circle.anchor').data([0]).join('circle').attr('class', 'anchor').attr('cx', FC.x(1)).attr('cy', FC.y(0)).attr('r', 10).attr('fill', ink()).attr('stroke', css('--surface')).attr('stroke-width', 2);
    // probability bars
    const sorted = cands.slice().sort((a, b) => b.s - a.s), bw = 1 / sorted.length;
    const pmax = Math.max(0.05, d3.max(sorted, (c) => c.p));
    PB.y.domain([0, pmax * 1.1]); PB.g.select('.axis.y').call(d3.axisLeft(PB.y).ticks(5));
    PB.layer('b').selectAll('rect').data(sorted).join('rect').attr('x', (c, i) => PB.x(i * bw) + 1).attr('width', Math.max(1, PB.x(bw) - PB.x(0) - 2))
      .attr('y', (c) => PB.y(c.p)).attr('height', (c) => PB.y(0) - PB.y(c.p)).attr('fill', (c) => (c.pos ? cat(2) : cat(1))).attr('rx', 1.5)
      .on('mousemove', (e, c) => tip(e, `${c.pos ? 'positive' : 'negative'}: similarity ${c.s.toFixed(2)}, probability ${c.p.toFixed(3)}`)).on('mouseleave', untip);
    const negs = cands.filter((c) => !c.pos), wn = d3.sum(negs, (c) => c.p);
    const effNeg = wn > 0 ? (wn * wn) / d3.sum(negs, (c) => c.p * c.p) : 0;
    const pp = cands[0].p;
    ro2.set({ 'p(positive)': pp.toFixed(3), 'loss −log p(positive)': (-Math.log(pp)).toFixed(3), 'loss if all equal: log(N+1)': Math.log(nNeg + 1).toFixed(3), 'negatives that matter (effective count)': effNeg.toFixed(1) + ' of ' + nNeg });
  }
  paint2();

  section(main, 'What to notice', `<ul>
    <li><b>Press Train.</b> At first the encoder maps the plane onto the circle arbitrarily and neighbouring colours overlap. Within a few hundred steps the groups slide apart and each claims its own arc: the histogram bars become six separate coloured clumps. No labels were used. The readout <i>closest two group centres</i> uses the hidden groups only to grade the result: six evenly spread groups would be 60° apart.</li>
    <li><b>Why groups form.</b> The loss only says "your jittered twin is closer than everyone else". Points in the same blob are jittered into each other's neighbourhood, so a smooth encoder cannot tell them apart and they end up together; separate blobs can be pushed apart freely.</li>
    <li><b>Augmentation defines "same".</b> Raise the jitter towards 0.8, larger than the gap between blobs, and press Reset, then Train. Neighbouring blobs now produce overlapping views, the encoder is told they are interchangeable, and the clumps merge. Go the other way, to 0.02: each point now only has to beat a tiny wobble of itself, so the loss is happy to spread every point evenly round the circle. The groups still occupy separate arcs, but each smears into a wide band instead of a tight clump.</li>
    <li><b>Temperature (second figure).</b> At τ = 0.05 nearly all the probability sits on the most similar candidate, and the effective count of negatives that matter drops towards 1 or 2: the loss focuses on the <i>hardest</i> negatives. Raise τ to 2 and every candidate gets a similar share; the effective count approaches all of them and the loss barely cares where the positive is.</li>
    <li><b>Batch size = number of negatives.</b> In the second figure, add negatives: the loss for the same positive goes up, because there are more candidates to beat, and its ceiling log(N+1) rises. With 2 views each of B points, every anchor has 2B − 2 negatives; SimCLR needed batches of thousands for this reason, and MoCo keeps a queue of old embeddings to get many negatives cheaply.</li>
  </ul>`);

  steps(main, 'How it works', [
    'Take a batch of B unlabelled examples. Make two random <b>augmentations</b> of each (crops and colour jitter for images; here, Gaussian jitter). The two views of one example are a <b>positive pair</b>; all other views in the batch are <b>negatives</b>.',
    'Encode every view and normalise to unit length, so similarity is the cosine (the dot product). Usually a small "projection head" sits on top; the representation used later is taken from before it.',
    'For each anchor view, compute a softmax over its 2B − 1 candidates using similarity / τ, and take −log of the probability given to its positive. This is the InfoNCE (NT-Xent) loss: a (2B − 1)-way classification problem whose "label" comes free from the augmentation.',
    'Backpropagate. The gradient pulls the anchor towards its positive and pushes it away from each negative in proportion to that negative\'s probability, so hard negatives get the biggest push.',
    'After training, throw away the head and use the encoder\'s output as features for clustering, kNN or a small classifier trained on few labels (a linear probe).',
  ]);
  math(main, 'L(anchor i) = −log [ exp(sim(zᵢ, zⱼ₊)/τ) / Σ_{k≠i} exp(sim(zᵢ, z_k)/τ) ]\n∂L/∂zᵢ = (1/τ) [ Σ_k p_k z_k − zⱼ₊ ]        (p_k = softmax weight of candidate k)\nalignment: E‖z − z⁺‖²     uniformity: log E exp(−2‖zᵢ − zⱼ‖²)        (Wang & Isola 2020)');

  callout(main, '', 'In biology', 'In single-cell, <b>CLEAR</b> and <b>contrastive-sc</b> treat dropout and noise as the augmentation, so cells learn embeddings robust to sampling noise; <b>scGPT</b> and <b>CellPLM</b> add contrastive terms to masked pretraining. In spatial omics, <b>Novae</b> and <b>GraphST</b> use contrastive objectives over tissue neighbourhoods (a niche and its perturbed copy are a positive pair). Image models for pathology (e.g. <b>UNI</b>, <b>CTransPath</b>) were pretrained with contrastive or related self-distillation objectives on millions of H&E tiles, and <b>CLIP</b>-style training pairs two modalities, such as an H&E tile and the expression measured at that spot.');
  callout(main, 'warn', 'Pitfalls', `<ul>
    <li><b>The augmentations are the inductive bias.</b> Whatever they change, the model learns to ignore. If an augmentation alters biology (heavy gene dropout that erases a rare marker), the model learns to ignore the biology.</li>
    <li><b>False negatives.</b> Two different cells of the same type are treated as negatives and pushed apart. With few types and big batches this happens constantly; low temperatures make it worse because they push hardest on the most similar "negatives".</li>
    <li><b>Collapse.</b> Without negatives (or a substitute such as BYOL's predictor or VICReg's variance term), mapping everything to one point gives perfect alignment. The uniformity readout above is the guard against this.</li>
    <li><b>Batch effects as shortcuts.</b> If both views always share the batch and negatives often come from other batches, "which batch am I from" becomes an easy way to win, and the embedding encodes batch.</li>
  </ul>`);
  seeAlso(main, [['concepts', 'embeddings', 'Embeddings'], ['concepts', 'foundation-models', 'Self-supervision & foundation models'], ['concepts', 'autoencoder', 'Autoencoders'], ['concepts', 'tsne-umap', 't-SNE & UMAP'], ['frontier', 'spatial-niches', 'Novae & Nicheformer']]);
}
