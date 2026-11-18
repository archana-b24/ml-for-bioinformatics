// Self-supervision & foundation models: pretrain an encoder on unlabelled cells by masked-gene prediction,
// then fit a linear probe on a few labels and compare with training from scratch and with PCA.
import { header, figure, slider, select, button, stats, section, steps, callout, math, legend, seeAlso, animate, tip, untip, el } from '../../lib/ui.js';
import { frame, cat, css, ink, ink2, muted } from '../../lib/plot.js';
import { rng, solve } from '../../lib/data.js';
import { mlp, pcaFit } from './autoencoder.js';
const d3 = window.d3;

const NS = 40, NNOISE = 10, GNOISE = 1.0;            // structured genes, extra noisy genes
const N_UNLAB = 1500, N_POOL = 600, N_TEST = 600;
const NS_LIST = [4, 8, 16, 32, 64, 128, 256], REPEATS = 10, MASK = 0.3;

/** Cells driven by two hidden programmes u = (u1, u2). Label = "u1 > 0" (say, responder vs non-responder state).
 *  Each structured gene responds nonlinearly to its own mix of u1 and u2, plus noise. Optional extra genes are pure,
 *  high-variance noise (a technical artefact that has nothing to do with the cells' state). */
function makeWorld(noisy) {
  const r = rng(61);
  const genes = d3.range(NS).map((g) => { const a = (g / NS) * Math.PI + r.normal(0, 0.15); return { w: [Math.cos(a), Math.sin(a)], c: r.normal(0, 0.4), amp: 0.8 + 0.6 * r() }; });
  const D = NS + (noisy ? NNOISE : 0);
  const cell = (rr) => {
    const u = [rr.normal(), rr.normal()];
    const x = genes.map((gn) => gn.amp * Math.tanh(1.5 * (gn.w[0] * u[0] + gn.w[1] * u[1]) + gn.c) + rr.normal(0, GNOISE));
    if (noisy) for (let k = 0; k < NNOISE; k++) x.push(rr.normal(0, 5));
    return { u, x, y: u[0] > 0 ? 1 : 0 };
  };
  const ru = rng(62), rp = rng(63), rt = rng(64);
  const unlab = d3.range(N_UNLAB).map(() => cell(ru)), pool = d3.range(N_POOL).map(() => cell(rp)), test = d3.range(N_TEST).map(() => cell(rt));
  const mu = d3.range(D).map((j) => d3.mean(unlab, (c) => c.x[j]));
  [unlab, pool, test].forEach((set) => set.forEach((c) => { c.xc = c.x.map((v, j) => v - mu[j]); }));
  return { D, unlab, pool, test };
}

/** Ridge-penalised logistic regression by Newton's method. X rows (already standardised), y in {0,1}. */
function fitLogit(X, y, lam = 1) {
  const n = X.length, p = X[0].length + 1, w = new Float64Array(p);
  const Xi = X.map((x) => { const r = new Float64Array(p); r[0] = 1; for (let j = 1; j < p; j++) r[j] = x[j - 1]; return r; });
  const g = new Float64Array(p), Hf = new Float64Array(p * p);
  for (let it = 0; it < 8; it++) {
    g.fill(0); Hf.fill(0);
    for (let i = 0; i < n; i++) {
      const xi = Xi[i]; let z = 0; for (let j = 0; j < p; j++) z += w[j] * xi[j];
      const pr = 1 / (1 + Math.exp(-z)), s = Math.max(1e-6, pr * (1 - pr)), e = pr - y[i];
      for (let j = 0; j < p; j++) { const a = xi[j]; g[j] += e * a; const sa = s * a, off = j * p; for (let k = j; k < p; k++) Hf[off + k] += sa * xi[k]; }
    }
    const H = Array.from({ length: p }, (_, j) => Array.from({ length: p }, (_, k) => (k >= j ? Hf[j * p + k] : Hf[k * p + j])));
    for (let j = 1; j < p; j++) { g[j] += lam * w[j]; H[j][j] += lam; }
    H[0][0] += 1e-6;
    const step = solve(H, Array.from(g)); let mx = 0;
    for (let j = 0; j < p; j++) { w[j] -= step[j]; mx = Math.max(mx, Math.abs(step[j])); }
    if (mx < 1e-5) break;
  }
  return w;
}
const predict = (w, x) => { let z = w[0]; for (let j = 0; j < x.length; j++) z += w[j + 1] * x[j]; return z > 0 ? 1 : 0; };

export default async function render(main) {
  header(main, 'Self-supervision & foundation models',
    'Labels are expensive; unlabelled data is cheap. <b>Pretrain</b> a model on lots of unlabelled data with a task that needs no labels (hide part of each input and predict it), then <b>reuse</b> what it learned for your task with only a few labels. Scale that up to millions of cells and you have a foundation model. Whether it beats a simple baseline is an empirical question, and in single-cell biology the answer is often no.');

  const fig = figure(main, 'Pretrain on 1,500 unlabelled cells, then learn from a handful of labels',
    `Toy world: every cell is driven by two hidden programmes; the label we care about is whether the first one is on. ${NS} genes each respond (nonlinearly, noisily) to a different mix of the two. <b>Pretraining</b> (no labels): hide a random 30% of each cell's genes and train an encoder (genes → 32 tanh → 2 numbers) plus decoder to predict the hidden genes from the visible ones. <b>Downstream</b>: draw n labelled cells and fit a logistic regression on three kinds of features, then score it on ${N_TEST} held-out cells. Each point on the curve averages ${REPEATS} random draws of the labelled cells (bands: worst to best draw). *The pretraining curve scores masked predictions on the ${NS} informative genes of held-out cells.`);
  let noisy = false, W, net, epoch = 0, lossHist = [], stop = null, curves = null, nIdx = 2, pca;
  const MAXEP = 60;
  select(fig.controls, { label: 'Data', value: 'clean', options: [['clean', `clean: ${NS} informative genes`], ['noisy', `+ ${NNOISE} high-variance noise genes`]], onChange: (v) => { noisy = v === 'noisy'; setup(); } });
  const preBtn = button(fig.controls, 'Pretrain', () => { if (!FS) return; if (stop) pause(); else run(); }, true);
  button(fig.controls, 'Reset encoder', () => { pause(); newNet(); order = null; preBtn.textContent = 'Pretrain'; refresh(true); });
  slider(fig.controls, { label: 'labelled cells n', min: 0, max: NS_LIST.length - 1, step: 1, value: nIdx, fmt: (v) => NS_LIST[v], onInput: (v) => { nIdx = v; paint(); } });
  const ro = stats(fig.root);
  const row = el('div', { class: 'fig-row' }); fig.body.append(row);
  const left = el('div'), right = el('div'); row.append(left, right);
  legend(left, [['label 0', cat(0)], ['label 1', cat(1)], ['the n labelled cells (ringed)', ink()], ['linear probe boundary', ink2()]]);
  const FE = frame(left, { width: 460, height: 380, x: [-3, 3], y: [-3, 3], xLabel: 'encoder output 1', yLabel: 'encoder output 2' });
  ['test', 'bound', 'lab'].forEach((n) => FE.layer(n));
  const LP = frame(left, { width: 460, height: 150, x: [0, MAXEP], y: [0, 1], xLabel: 'pretraining epoch', yLabel: 'masked MSE*' });
  legend(right, [['pretrained encoder + linear probe', cat(2)], ['PCA (2 PCs) + linear probe', cat(3)], ['from scratch: logistic on all genes', cat(7)]]);
  const LC = frame(right, { width: 470, height: 520, x: [4, 256], y: [0.45, 1], xScale: d3.scaleLog().base(2), xLabel: 'number of labelled cells (log scale)', yLabel: 'held-out accuracy' });
  LC.g.select('.axis.x').call(d3.axisBottom(LC.x).tickValues(NS_LIST).tickFormat(d3.format('d')));
  LC.g.select('.axis.y').call(d3.axisLeft(LC.y).ticks(6, '.0%'));

  // ---- features ----
  const standardiser = (F) => { const D = F[0].length, m = d3.range(D).map((j) => d3.mean(F, (f) => f[j])), s = d3.range(D).map((j) => d3.deviation(F, (f) => f[j]) || 1); return (f) => f.map((v, j) => (v - m[j]) / s[j]); };
  const encOut = () => net.layerOut(2);
  function featureSets(all) {
    // embeddings come from the encoder half only: run the full net and read the bottleneck layer
    const emb = (set) => set.map((c) => { net.forward(c.xc); const z = encOut(); return [z[0], z[1]]; });
    const pc = (set) => set.map((c) => { const v = pca.vecs; return [0, 1].map((k) => c.xc.reduce((s, x, j) => s + x * v[k][j], 0)); });
    const raw = (set) => set.map((c) => c.xc);
    const out = all ? {} : FS;
    (all ? [['pre', emb], ['pca', pc], ['raw', raw]] : [['pre', emb]]).forEach(([k, f]) => { const U = f(W.unlab), st = standardiser(U); out[k] = { pool: f(W.pool).map(st), test: f(W.test).map(st) }; });
    return out;
  }
  function drawLabels(n, rep) {
    const r = rng(1000 + 37 * n + rep), pos = [], neg = [];
    r.shuffle(d3.range(N_POOL)).forEach((i) => { (W.pool[i].y ? pos : neg).push(i); });
    return pos.slice(0, n / 2).concat(neg.slice(0, n / 2));
  }
  let FS = null;
  function curvePoint(k, ni) {
    const n = NS_LIST[ni];
    const accs = d3.range(REPEATS).map((rep) => {
      const idx = drawLabels(n, rep), w = fitLogit(idx.map((i) => FS[k].pool[i]), idx.map((i) => W.pool[i].y));
      return d3.mean(W.test, (c, t) => (predict(w, FS[k].test[t]) === c.y ? 1 : 0));
    });
    return { n, m: d3.mean(accs), lo: d3.min(accs), hi: d3.max(accs) };
  }
  // The from-scratch curve (logistic on all genes) is the expensive one and never changes during pretraining:
  // compute it once per dataset, one point per animation frame, so no interaction blocks for long.
  let queue = [], pump = null;
  function refresh(all) {
    FS = featureSets(all);
    if (all) {
      curves = { raw: new Array(NS_LIST.length).fill(null) };
      curves.pca = NS_LIST.map((n, ni) => curvePoint('pca', ni));
      queue = NS_LIST.map((n, ni) => ni);
      if (pump) pump();
      pump = animate(() => { const ni = queue.shift(); curves.raw[ni] = curvePoint('raw', ni); paint(); if (!queue.length) { pump = null; return false; } return true; });
    }
    curves.pre = NS_LIST.map((n, ni) => curvePoint('pre', ni));
    paint();
  }

  // ---- pretraining ----
  function newNet() { net = mlp([W.D, 32, 2, 32, W.D], ['tanh', 'linear', 'tanh', 'linear'], 71); epoch = 0; lossHist = [[0, maskedLoss()]]; mr = rng(5); }
  let mr = rng(5);
  const masked = new Float64Array(64), dy = new Float64Array(64);
  function maskedLoss() {
    const r = rng(99); let s = 0, c = 0;
    W.test.slice(0, 200).forEach((cl) => {
      const m = cl.xc.map(() => r() < MASK), inp = cl.xc.map((v, j) => (m[j] ? 0 : v)), o = net.forward(inp);
      m.forEach((h, j) => { if (h && j < NS) { s += (o[j] - cl.xc[j]) ** 2; c++; } }); // score the informative genes only
    });
    return s / Math.max(1, c);
  }
  let cursor = 0, order = null;
  function trainChunk(budget) {
    const t0 = performance.now(), D = W.D, B = 50;
    while (performance.now() - t0 < budget) {
      if (!order || cursor >= order.length) { const done = !!order; order = mr.shuffle(d3.range(N_UNLAB)); cursor = 0; if (done) { epoch++; lossHist.push([epoch, maskedLoss()]); return true; } }
      for (let b = 0; b < B && cursor < order.length; b++, cursor++) {
        const xc = W.unlab[order[cursor]].xc;
        const inp = new Array(D); let any = false;
        for (let j = 0; j < D; j++) { masked[j] = mr() < MASK ? 1 : 0; any = any || masked[j]; inp[j] = masked[j] ? 0 : xc[j]; }
        if (!any) continue;
        const o = net.forward(inp);
        for (let j = 0; j < D; j++) dy[j] = masked[j] ? o[j] - xc[j] : 0;
        net.backward(dy.subarray(0, D));
      }
      net.step(0.005, B);
    }
    return false;
  }
  function run() {
    if (epoch >= MAXEP) { newNet(); order = null; }
    preBtn.textContent = 'Pause';
    stop = animate(() => {
      const ended = trainChunk(12);
      if (ended && (epoch % 5 === 0 || epoch >= MAXEP)) refresh(); else paint();
      if (epoch >= MAXEP) { pause(); return false; }
      return true;
    });
  }
  function pause() { if (stop) { stop(); stop = null; } preBtn.textContent = epoch >= MAXEP ? 'Pretrain again' : 'Pretrain'; }

  function paint() {
    if (!FS) return;
    const n = NS_LIST[nIdx];
    // embedding panel (standardised encoder outputs)
    const E = FS.pre.test, P = FS.pre.pool;
    FE.layer('test').selectAll('circle').data(W.test).join('circle').attr('cx', (c, i) => FE.x(Math.max(-3, Math.min(3, E[i][0])))).attr('cy', (c, i) => FE.y(Math.max(-3, Math.min(3, E[i][1])))).attr('r', 3)
      .attr('fill', (c) => cat(c.y)).attr('fill-opacity', 0.35);
    const idx = drawLabels(n, 0), w = fitLogit(idx.map((i) => P[i]), idx.map((i) => W.pool[i].y));
    FE.layer('lab').selectAll('circle').data(idx).join('circle').attr('cx', (i) => FE.x(Math.max(-3, Math.min(3, P[i][0])))).attr('cy', (i) => FE.y(Math.max(-3, Math.min(3, P[i][1])))).attr('r', 5.5)
      .attr('fill', (i) => cat(W.pool[i].y)).attr('stroke', ink()).attr('stroke-width', 1.8)
      .on('mousemove', (e, i) => tip(e, `labelled cell, label ${W.pool[i].y}<br>hidden programme 1 = ${W.pool[i].u[0].toFixed(2)}`)).on('mouseleave', untip);
    // boundary w0 + w1 x + w2 y = 0, clipped to the box
    const seg = [];
    if (Math.abs(w[2]) > 1e-9) [-3, 3].forEach((x) => seg.push([x, -(w[0] + w[1] * x) / w[2]]));
    else if (Math.abs(w[1]) > 1e-9) [-3, 3].forEach((y) => seg.push([-(w[0] + w[2] * y) / w[1], y]));
    FE.layer('bound').selectAll('line').data(seg.length ? [seg] : []).join('line').attr('x1', (s) => FE.x(s[0][0])).attr('y1', (s) => FE.y(s[0][1])).attr('x2', (s) => FE.x(s[1][0])).attr('y2', (s) => FE.y(s[1][1]))
      .attr('stroke', ink2()).attr('stroke-width', 2).attr('stroke-dasharray', '6 4');
    FE.svg.select('clipPath').empty() && FE.svg.append('clipPath').attr('id', 'fm-clip').append('rect').attr('width', FE.w).attr('height', FE.h);
    FE.layer('bound').attr('clip-path', 'url(#fm-clip)');
    // pretraining loss
    LP.y.domain([0, d3.max(lossHist, (h) => h[1]) * 1.1]); LP.g.select('.axis.y').call(d3.axisLeft(LP.y).ticks(3));
    LP.layer('c').selectAll('path').data([lossHist]).join('path').attr('fill', 'none').attr('stroke', cat(2)).attr('stroke-width', 2).attr('d', d3.line().x((h) => LP.x(h[0])).y((h) => LP.y(h[1])));
    LP.layer('d').selectAll('circle').data([lossHist[lossHist.length - 1]]).join('circle').attr('cx', (h) => LP.x(h[0])).attr('cy', (h) => LP.y(h[1])).attr('r', 4.5).attr('fill', cat(2)).attr('stroke', css('--surface')).attr('stroke-width', 1.5);
    // learning curves
    const keys = [['raw', cat(7), 'from scratch'], ['pca', cat(3), 'PCA + probe'], ['pre', cat(2), 'pretrained + probe']];
    LC.layer('band').selectAll('path').data(keys).join('path').attr('fill', (k) => k[1]).attr('fill-opacity', 0.12)
      .attr('d', (k) => d3.area().x((d) => LC.x(d.n)).y0((d) => LC.y(Math.max(0.45, d.lo))).y1((d) => LC.y(d.hi))(curves[k[0]].filter(Boolean)));
    LC.layer('line').selectAll('path').data(keys).join('path').attr('fill', 'none').attr('stroke', (k) => k[1]).attr('stroke-width', 2)
      .attr('d', (k) => d3.line().x((d) => LC.x(d.n)).y((d) => LC.y(Math.max(0.45, d.m)))(curves[k[0]].filter(Boolean)));
    LC.layer('pts').selectAll('g').data(keys).join('g').each(function (k) {
      d3.select(this).selectAll('circle').data(curves[k[0]].filter(Boolean)).join('circle').attr('cx', (d) => LC.x(d.n)).attr('cy', (d) => LC.y(Math.max(0.45, d.m))).attr('r', 4.5)
        .attr('fill', k[1]).attr('stroke', css('--surface')).attr('stroke-width', 1.5)
        .on('mousemove', (e, d) => tip(e, `${k[2]}, n = ${d.n}<br>mean accuracy ${(100 * d.m).toFixed(1)}%<br>range over ${REPEATS} draws ${(100 * d.lo).toFixed(0)}–${(100 * d.hi).toFixed(0)}%`)).on('mouseleave', untip);
    });
    LC.layer('mk').selectAll('line').data([n]).join('line').attr('x1', (d) => LC.x(d)).attr('x2', (d) => LC.x(d)).attr('y1', 0).attr('y2', LC.h).attr('stroke', muted()).attr('stroke-dasharray', '3 3');
    const at = (k) => (curves[k][nIdx] ? (100 * curves[k][nIdx].m).toFixed(1) + '%' : '…');
    ro.set({ 'pretraining epoch': `${epoch} / ${MAXEP}`, 'masked MSE* (informative genes, held-out)': lossHist[lossHist.length - 1][1].toFixed(3), [`pretrained + probe @ n=${n}`]: at('pre'), [`PCA + probe @ n=${n}`]: at('pca'), [`from scratch @ n=${n}`]: at('raw') });
  }

  function setup() { // split across two frames so neither blocks for long
    pause(); FS = null; W = makeWorld(noisy); pca = pcaFit(W.unlab.map((c) => c.xc));
    animate(() => { newNet(); order = null; preBtn.textContent = 'Pretrain'; refresh(true); return false; });
  }
  setup();

  section(main, 'What to notice', `<ul>
    <li><b>Before pretraining</b>, the green curve uses a randomly initialised encoder: two random nonlinear mixtures of the genes. In the left panel the two labels overlap. Press <b>Pretrain</b> and watch the masked-gene error fall; every 5 epochs the probe curves are recomputed, and the green curve climbs as the embedding sorts itself out. No label was used for that.</li>
    <li><b>Few labels.</b> After pretraining, look at n = 4–8: the probe on 2 pretrained features beats logistic regression on all ${NS} raw genes (red curve), which has to estimate ${NS} weights from a handful of cells. As n grows the gap narrows to a point or two. Pretraining helps most when labels are scarce, and even then, in this toy, by a few points rather than a revolution: a regularised linear model on correlated genes is already a decent baseline.</li>
    <li><b>The honest baseline.</b> On the clean data, PCA, which also uses only unlabelled cells and takes milliseconds, does about as well as the pretrained encoder. Here the signal is the biggest source of variation, so the top 2 PCs already contain it. This is the single-cell story in miniature: careful benchmarks keep finding that PCA on highly variable genes, or scVI, matches large pretrained models.</li>
    <li><b>Switch to "+ ${NNOISE} high-variance noise genes"</b> and pretrain again. Now PCA's top components chase the noise genes (big variance, no structure) and the PCA curve sinks, while masked prediction ignores them because noise is unpredictable from the other genes. Pretraining learns what is <i>shared and predictable</i>, not merely what is <i>variable</i>. (Real pipelines would filter or scale such genes first, which would rescue PCA.)</li>
    <li><b>Drag the n slider</b> to see which cells were labelled (ringed) and the probe's boundary in the embedding.</li>
  </ul>`);

  steps(main, 'How it works', [
    '<b>Pretext task.</b> Invent a task whose answer is already in the data. Here: mask 30% of the genes and predict them from the rest (like BERT\'s masked words; Geneformer and scGPT mask genes). Contrastive objectives (two views of one cell should match) are the other big family.',
    '<b>Pretrain</b> an encoder on as much unlabelled data as you have. To predict hidden genes it must learn which genes move together, i.e. the cell\'s underlying state.',
    '<b>Transfer.</b> Freeze the encoder and train only a small classifier on its output (a <b>linear probe</b>), or <b>fine-tune</b>: start from the pretrained weights and keep training everything on the labelled task with a small learning rate. Fine-tuning usually wins with more labels; probes are cheaper and less prone to overfitting with few. (This toy only shows the probe.)',
    '<b>Zero-shot</b> use skips the labelled step entirely: embed cells and cluster or map them directly.',
    '<b>Benchmark against simple baselines</b>, fitted with the same labels and the same unlabelled data: PCA, highly variable genes, scVI, a linear model. A foundation model is only worth its cost if it beats them on your task.',
  ]);
  math(main, 'pretrain:  min  E_x E_mask ‖ x_masked − decoder(encoder(x_visible))_masked ‖²      (no labels)\nprobe:     min  Σ_{i ≤ n} logloss( yᵢ , w·encoder(xᵢ) + b ) + λ‖w‖²                  (n labels, encoder frozen)');

  callout(main, '', 'In biology', 'Single-cell foundation models: <b>Geneformer</b> (rank-value encoded genes, masked prediction, ~30M then ~100M cells), <b>scGPT</b> (gene + binned expression tokens, 33M cells), <b>scFoundation</b>, <b>UCE</b>, <b>CellPLM</b> and <b>Nicheformer</b>; for pathology, <b>UNI</b>, <b>Virchow</b> and <b>CONCH</b>; for sequence, <b>Enformer</b>, <b>AlphaGenome</b> and protein language models such as <b>ESM</b>. The independent picture is sobering: Kedzierska et al. (Genome Biology 2025) found zero-shot Geneformer and scGPT embeddings often worse than highly-variable-gene selection + PCA, scVI or Harmony for clustering and batch integration, and perturbation benchmarks find simple additive or mean baselines hard to beat. See the <a href="#/archive/frontier/landscape">landscape</a>, <a href="#/archive/frontier/geneformer">Geneformer</a> and <a href="#/archive/frontier/zero-shot-check">zero-shot reality check</a> pages. Protein language models and pathology models have clearer wins.');
  callout(main, 'warn', 'Pitfalls', `<ul>
    <li><b>No baseline, no claim.</b> Always report PCA / HVG / scVI / linear baselines fitted on the same data. In this toy, PCA ties on the clean data.</li>
    <li><b>Pretraining data ≠ your data.</b> If your tissue, technology or disease is rare in the pretraining corpus, the learned features may not transfer, and the gain can be negative.</li>
    <li><b>Leakage through pretraining.</b> If your test cells (or the same donors) were in the pretraining corpus, downstream scores are inflated. Check what the model saw.</li>
    <li><b>Big models overfit small labelled sets when fine-tuned.</b> With a few dozen labels, a frozen probe, strong regularisation and nested cross-validation are safer than full fine-tuning.</li>
    <li><b>Benchmarks choose winners.</b> Results depend on the metric and the split; a model can lead one benchmark and trail another. Look for independent evaluations, not only the authors'.</li>
  </ul>`);
  seeAlso(main, [['concepts', 'autoencoder', 'Autoencoders'], ['concepts', 'contrastive', 'Contrastive learning'], ['concepts', 'pca', 'PCA'], ['concepts', 'transformer', 'Transformers'], ['frontier', 'landscape', 'The 2024–26 landscape'], ['frontier', 'geneformer', 'Geneformer'], ['frontier', 'zero-shot-check', 'Do foundation embeddings beat PCA?']]);
}
