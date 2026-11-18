// Neural networks and backpropagation, lesson format.
// Arc: what → why → where → how it works (6 steps: one neuron → hidden layer → activations → backprop worked example →
//      live training lab → overfitting, dropout, early stopping) → pitfalls → recap.
// Every number in the prose is computed from data/study/cohort.json (and from networks trained at render time with fixed seeds).
import { header, figure, slider, select, button, checkbox, stats, legend, seeAlso, animate, tip, untip, el, html } from '../lib/ui.js';
import { frame, decisionCanvas, cat, css, ink, ink2, muted } from '../lib/plot.js';
import { sigmoid, mean, rng } from '../lib/data.js';
import { fitLogistic, logLoss, auc, standardize, makeMLP } from '../lib/models.js';
import { loadCohort, FEATURES, studyCard } from '../lib/study.js';
import { lessonNav, chapter, prose, rung, worked, quiz, cards, table, code, recap } from '../lib/lesson.js';
const d3 = window.d3;

const f2 = (v) => (Math.abs(v) < 0.005 ? '0.00' : v.toFixed(2));
const f3 = (v) => (Math.abs(v) < 0.0005 ? '0.000' : v.toFixed(3));
const f1 = (v) => v.toFixed(1);
const pc = (v) => Math.round(100 * v) + '%';
const sum = (a) => a.reduce((s, v) => s + v, 0);
const acc = (p, y) => p.filter((q, i) => (q >= 0.5) === (y[i] === 1)).length / y.length;
const sci = (v) => v.toExponential(1).replace('e-', '×10<sup>−') + '</sup>';
const relu = (z) => (z > 0 ? z : 0);

/** Train a network for a fixed number of epochs (full pass over X per epoch). */
function fitNet(X, y, hidden, { epochs = 500, lr = 0.02, batch = 32, seed = 3, act = 'relu', l2 = 0, dropout = 0 } = {}) {
  const net = makeMLP([X[0].length, ...hidden, 1], { act, seed, l2, dropout });
  for (let e = 0; e < epochs; e++) net.epoch(X, y, lr, batch);
  return net;
}

export default async function render(main) {
  const C = await loadCohort();
  const D = C.discovery, V = C.validation, n = D.length;
  const y = D.map((p) => p.response), yv = V.map((p) => p.response);
  const nR = sum(y), base = nR / n;

  // ---- two 2-feature tasks used by the figures -------------------------------------------------------------
  // 'toy': is the tumour immune-excluded (phenotype label) from IFN-γ and log2 CD8? The excluded tumours form a band between
  //        desert (low, low) and inflamed (high, high), a shape no single straight boundary can isolate.
  // 'resp': response from IFN-γ and CAF score (the true rule is linear in these, so a straight boundary suffices).
  function mkTask(key, featFn, labFn, labels) {
    const st = standardize(D.map(featFn));
    const X = st.Z, yy = D.map(labFn), Xv = V.map((p) => st.apply(featFn(p))), yyv = V.map(labFn);
    const lr = fitLogistic(X, yy), lp = (z) => sigmoid(lr.b0 + lr.w[0] * z[0] + lr.w[1] * z[1]);
    const pvl = Xv.map(lp), ptl = X.map(lp);
    const raw = (a, b) => [(a - st.mu[0]) / st.sd[0], (b - st.mu[1]) / st.sd[1]];
    const ext = (j) => { const v = D.concat(V).map((p) => featFn(p)[j]); const [a, b] = d3.extent(v), pad = (b - a) * 0.04; return [a - pad, b + pad]; };
    return { key, st, X, y: yy, Xv, yv: yyv, raw, lr, featFn, xr: ext(0), yr: ext(1), ...labels,
      refLoss: logLoss(pvl, yyv), refAcc: acc(pvl, yyv), refAuc: auc(pvl, yyv), refTrAcc: acc(ptl, yy), lp };
  }
  const T = {
    toy: mkTask('toy', (p) => [p.ifng, Math.log2(p.cd8)], (p) => +(p.phenotype === 'excluded'),
      { xl: 'IFN-γ score (log2 expression)', yl: 'log2 CD8 T cells (% of cells)', c0: 'other tumour', c1: 'immune-excluded', name: 'immune-excluded vs other' }),
    resp: mkTask('resp', (p) => [p.ifng, p.caf], (p) => p.response,
      { xl: 'IFN-γ score (log2 expression)', yl: 'CAF / TGF-β score (log2 expression)', c0: 'no response', c1: 'responded', name: 'response' }),
  };
  const evalNet = (net, t) => {
    const pt = net.predict(t.X), pv = net.predict(t.Xv);
    return { trL: logLoss(pt, t.y), vaL: logLoss(pv, t.yv), trAcc: acc(pt, t.y), vaAcc: acc(pv, t.yv), vaAuc: auc(pv, t.yv) };
  };
  // fixed networks for the toy task: 1, 2, 3, 4 and 8 hidden ReLU units
  const HS = [1, 2, 3, 4, 8];
  const toyNets = HS.map((H) => fitNet(T.toy.X, T.toy.y, [H], { epochs: 500, lr: 0.02, seed: 3 }));
  const toyEv = toyNets.map((net) => evalNet(net, T.toy));
  const toyOf = (H) => toyEv[HS.indexOf(H)];

  // ---- the three-feature network used by the diagram ------------------------------------------------------------
  const F3 = ['ifng', 'log2tmb', 'caf'];
  const rows3 = (pts) => pts.map((p) => F3.map((k) => FEATURES[k].get(p)));
  const st3 = standardize(rows3(D));
  const Xv3 = rows3(V).map(st3.apply);
  let net1 = null, bestL = Infinity;
  for (let s = 1; s <= 6; s++) { const m = fitNet(st3.Z, y, [3], { epochs: 400, lr: 0.02, seed: s }); const l = logLoss(m.predict(st3.Z), y); if (l < bestL) { bestL = l; net1 = m; } }
  const p1 = (p) => net1.forward(st3.apply(F3.map((k) => FEATURES[k].get(p))));
  const auc1 = auc(net1.predict(Xv3), yv);
  let deadUnits = 0;
  const deadShare = (() => {
    let off = 0; const offU = [0, 0, 0]; D.forEach((p) => { const z = st3.apply(F3.map((k) => FEATURES[k].get(p))); for (let i = 0; i < 3; i++) if (net1.W[0][i * 3] * z[0] + net1.W[0][i * 3 + 1] * z[1] + net1.W[0][i * 3 + 2] * z[2] + net1.b[0][i] <= 0) { off++; offU[i]++; } });
    deadUnits = offU.filter((c) => c === n).length;
    return off / (3 * n);
  })();

  // ---- comparison with logistic regression on tabular data (several seeds) --------------------------------------
  const F7 = ['ifng', 'log2tmb', 'caf', 'pdl1', 'cd8', 'prolif', 'age'];
  const cmpSets = { f3: F3, f7: F7 };
  const SEEDS = [1, 2, 3, 4, 5];
  const cmp = {};
  Object.entries(cmpSets).forEach(([k, keys]) => {
    const st = standardize(D.map((p) => keys.map((q) => FEATURES[q].get(p)))), Xv = V.map((p) => st.apply(keys.map((q) => FEATURES[q].get(p))));
    const lr = fitLogistic(st.Z, y);
    const lrAuc = auc(Xv.map((r) => sigmoid(lr.b0 + sum(r.map((v, j) => v * lr.w[j])))), yv);
    const mk = (H) => SEEDS.map((s) => auc(fitNet(st.Z, y, [H], { epochs: 200, lr: 0.01, seed: s }).predict(Xv), yv));
    cmp[k] = { lr: lrAuc, h4: mk(4), h16: mk(16) };
  });
  const rng3 = (a) => `${mean(a).toFixed(3)} (${d3.min(a).toFixed(3)}–${d3.max(a).toFixed(3)})`;

  header(main, 'Neural networks and backpropagation',
    'A neural network is a <b>stack of logistic regressions</b>: each layer computes weighted sums of the previous layer\'s outputs and bends them with a non-linear function. Training adjusts every weight by <b>backpropagation</b>, an application of the chain rule that reports how the loss changes with each weight.');
  lessonNav(main);
  studyCard(main, C, '<b>Question for this page:</b> what does a hidden layer add to logistic regression, how are its weights learned, and does it improve prediction of response on tabular patient data? Two tasks are used: a non-linear toy task (is a tumour immune-excluded, from IFN-γ and CD8?) and the real response task.');

  // =====================================================================================
  // 1. WHAT
  // =====================================================================================
  let ch = chapter(main, 'what', 'What is a neural network?', 'In one line: <b>weighted sums → non-linear function → repeat</b>, then a sigmoid at the end.');
  prose(ch, `<p>A <b>neural network</b> (multilayer perceptron, MLP) is a function from measurements to a prediction, built from simple units called <b>neurons</b>. Each neuron computes a weighted sum of its inputs plus a bias, then applies an <b>activation function</b>. A neuron with a sigmoid activation is exactly a logistic regression. Neurons are arranged in <b>layers</b>: the input layer holds the measurements, one or more <b>hidden layers</b> compute intermediate features, and the output layer returns the probability.</p>
    <p>The diagram shows a small network (3 inputs, 3 hidden ReLU neurons, 1 output) trained on the ${n} discovery patients, with IFN-γ score, log2 TMB and CAF score as inputs. Its AUC on the ${V.length} validation patients (Centre C) is ${f3(auc1)}. Select a patient to follow the calculation.</p>`);
  const pos = D.filter((p) => p.response);
  const by = (arr, f) => arr.slice().sort((a, b) => f(a) - f(b))[0];
  const picks = [
    { key: 'hot', name: 'Highest predicted probability', p: by(pos, (p) => -p1(p)) },
    { key: 'border', name: 'Borderline', p: by(D, (p) => Math.abs(p1(p) - 0.5)) },
    { key: 'desert', name: 'Immune desert, non-responder', p: by(D.filter((p) => p.phenotype === 'desert' && !p.response), (p) => p1(p)) },
  ];
  const fd = figure(ch, 'One patient through a trained network',
    'Left to right: standardised inputs → hidden layer (z = weighted sum, a = ReLU(z)) → output probability. Line width is the size of a weight; <span class="muted">blue</span> lines are negative weights and <span class="muted">orange</span> lines positive. Hover a line for the weight.');
  let pk = 'hot';
  select(fd.controls, { label: 'Patient', value: pk, options: picks.map((v) => [v.key, `${v.name} (${v.p.id})`]), onChange: (v) => { pk = v; paintDiag(); } });
  const sd1 = stats(fd.root);
  const FD = d3.select(fd.body).append('svg').attr('viewBox', '0 0 820 335').attr('width', 820).attr('role', 'img').attr('aria-label', 'network diagram for one patient');
  legend(fd.body, [['negative weight', cat(0)], ['positive weight', cat(1)]]);
  const maxW = d3.max([...net1.W[0], ...net1.W[1]], Math.abs);
  function paintDiag() {
    const P = picks.find((v) => v.key === pk).p;
    const raw = F3.map((k) => FEATURES[k].get(P)), x = st3.apply(raw);
    const zH = [0, 1, 2].map((i) => net1.W[0][i * 3] * x[0] + net1.W[0][i * 3 + 1] * x[1] + net1.W[0][i * 3 + 2] * x[2] + net1.b[0][i]), aH = zH.map(relu);
    const zO = net1.b[1][0] + sum(aH.map((a, i) => a * net1.W[1][i])), pO = sigmoid(zO);
    FD.selectAll('*').remove();
    const X0 = 200, X1 = 440, X2 = 660, ys = [80, 170, 260];
    const wl = (w) => 0.8 + 4 * Math.min(1, Math.abs(w) / maxW);
    const edge = (x0, y0, x1, y1, w, lab) => FD.append('line').attr('x1', x0).attr('y1', y0).attr('x2', x1).attr('y2', y1).attr('stroke', w >= 0 ? cat(1) : cat(0)).attr('stroke-width', wl(w)).attr('stroke-opacity', 0.75)
      .on('mousemove', (e) => tip(e, `${lab}<br>weight ${f2(w)}`)).on('mouseleave', untip);
    ys.forEach((yy, i) => ys.forEach((y2, j) => edge(X0 + 22, yy, X1 - 22, y2, net1.W[0][j * 3 + i], `${FEATURES[F3[i]].label} → hidden ${j + 1}`)));
    ys.forEach((yy, j) => edge(X1 + 22, yy, X2 - 24, 170, net1.W[1][j], `hidden ${j + 1} → output`));
    [['Inputs (standardised)', X0], ['Hidden layer (ReLU)', X1], ['Output (sigmoid)', X2]].forEach(([t, x0]) => FD.append('text').attr('x', x0).attr('y', 20).attr('text-anchor', 'middle').attr('font-size', 12.5).attr('font-weight', 700).attr('fill', ink()).text(t));
    const node = (cx, cy, r, fill, label, sub) => { FD.append('circle').attr('cx', cx).attr('cy', cy).attr('r', r).attr('fill', fill).attr('stroke', css('--axis')).attr('stroke-width', 1.5); FD.append('text').attr('x', cx).attr('y', cy + 4).attr('text-anchor', 'middle').attr('font-size', 12.5).attr('font-weight', 700).attr('fill', ink()).text(label); };
    ys.forEach((yy, i) => {
      node(X0, yy, 22, css('--surface-2'), (x[i] >= 0 ? '+' : '−') + f2(Math.abs(x[i])));
      FD.append('text').attr('x', X0 - 32).attr('y', yy - 2).attr('text-anchor', 'end').attr('font-size', 12.5).attr('fill', ink2()).text(FEATURES[F3[i]].label);
      FD.append('text').attr('x', X0 - 32).attr('y', yy + 14).attr('text-anchor', 'end').attr('font-size', 11.5).attr('fill', muted()).text(`raw ${i === 1 ? f1(P.tmb) + ' mut/Mb' : f2(raw[i])}`);
      node(X1, yy, 22, aH[i] > 0 ? css('--accent-soft') : css('--surface-2'), f2(aH[i]));
      FD.append('text').attr('x', X1).attr('y', yy + 38).attr('text-anchor', 'middle').attr('font-size', 11.5).attr('fill', muted()).text(`z = ${f2(zH[i])}${aH[i] > 0 ? '' : ' → off'}`);
    });
    node(X2, 170, 26, css('--surface-2'), f2(pO));
    FD.append('text').attr('x', X2 + 36).attr('y', 166).attr('font-size', 12.5).attr('fill', ink2()).text(`z = ${f2(zO)}`);
    FD.append('text').attr('x', X2 + 36).attr('y', 183).attr('font-size', 12.5).attr('fill', ink2()).text(`p = σ(z) = ${f2(pO)}`);
    FD.append('text').attr('x', 6).attr('y', 328).attr('font-size', 11.5).attr('fill', muted()).text('A hidden neuron is "off" when z ≤ 0: ReLU outputs 0 and the neuron passes nothing on.');
    sd1.set({ patient: `${P.id} (${P.phenotype})`, 'predicted probability': f2(pO), outcome: P.response ? 'responded' : 'no response', 'hidden neurons active': `${aH.filter((a) => a > 0).length} of 3` });
  }
  paintDiag();
  prose(ch, '<p>Key terms:</p>');
  table(ch, [
    { t: 'Neuron (unit)', m: 'Weighted sum of inputs plus bias, passed through an activation function', s: `3 hidden neurons here; ${net1.nParams} parameters in total` },
    { t: 'Weight and bias', m: 'The numbers the network learns; each weight scales one connection', s: `largest |weight| = ${f2(maxW)}` },
    { t: 'Layer', m: 'A group of neurons that read the same inputs: input, hidden, output', s: 'input 3 → hidden 3 → output 1' },
    { t: 'Activation function', m: 'The non-linear function applied to each weighted sum (ReLU, sigmoid, tanh)', s: 'ReLU(z) = max(0, z) in the hidden layer; sigmoid at the output' },
    { t: 'Forward pass', m: 'Computing the prediction from inputs, layer by layer', s: 'left to right in the diagram' },
    { t: 'Loss', m: 'A number measuring how far predictions are from outcomes; here binary cross-entropy (log-loss)', s: `log-loss on the discovery set: ${f3(logLoss(net1.predict(st3.Z), y))}` },
    { t: 'Backpropagation', m: 'Computing the gradient of the loss with respect to every weight by the chain rule', s: 'chapter 4, step 4' },
    { t: 'Epoch, learning rate', m: 'One pass through the training data; the size of each weight update', s: 'chapter 4, step 5' },
  ], [['t', 'Term'], ['m', 'Meaning'], ['s', 'In this example']]);

  // =====================================================================================
  // 2. WHY
  // =====================================================================================
  ch = chapter(main, 'why', 'Why do we need it?', 'Logistic regression draws one straight boundary. Some structure needs a curved or multi-part boundary, or features that must themselves be learned.');
  prose(ch, `<p>A logistic regression assigns each feature a constant effect on the log-odds, so its decision boundary is a straight line (a flat plane in more dimensions). Three situations call for more:</p><ol>
    <li><b>Non-monotonic or interacting effects.</b> A class that occupies a band or island in feature space cannot be separated by one line.</li>
    <li><b>Learned features.</b> When the inputs are raw pixels, sequence letters or thousands of correlated genes, the useful features are not known in advance. Hidden layers learn them.</li>
    <li><b>A single framework across data types.</b> The same training procedure applies to images, sequences and tables, which is why networks underlie most current biological foundation models.</li></ol>
    <p>The figure shows the first case on the study data. Immune-excluded tumours have intermediate IFN-γ and CD8: lower than inflamed tumours, higher than deserts. They form a band between the other two groups.</p>`);
  const fw = figure(ch, 'A band-shaped class: logistic regression vs a network',
    'Each dot is a discovery patient, coloured by whether the tumour is immune-excluded (phenotype label from the pathology-style annotation). Shading: the model\'s predicted probability of "excluded". Accuracy is reported on the discovery patients and on the Centre C validation patients.');
  let wi = 0;
  select(fw.controls, { label: 'Model', value: '0', options: [['0', 'Logistic regression (one neuron)'], ['1', 'Network, 2 hidden neurons'], ['2', 'Network, 8 hidden neurons']], onChange: (v) => { wi = +v; paintW(); } });
  const sW = stats(fw.root);
  const FW = frame(fw.body, { width: 820, height: 360, x: T.toy.xr, y: T.toy.yr, xLabel: T.toy.xl, yLabel: T.toy.yl, grid: false });
  const toyProb = (m) => (a, b) => { const z = T.toy.raw(a, b); return m === 0 ? T.toy.lp(z) : toyNets[m === 1 ? 1 : 4].forward(z); };
  const surfW = decisionCanvas(FW, toyProb(0), { res: 4, alpha: 0.4 });
  FW.layer('pts').selectAll('circle').data(D).join('circle').attr('cx', (d) => FW.x(d.ifng)).attr('cy', (d) => FW.y(Math.log2(d.cd8))).attr('r', 3.8)
    .attr('fill', (d, i) => cat(T.toy.y[i])).attr('fill-opacity', 0.85).attr('stroke', css('--surface')).attr('stroke-width', 0.8)
    .on('mousemove', (e, d) => tip(e, `${d.id} · ${d.phenotype}<br>IFN-γ ${d.ifng}, CD8 ${d.cd8}%`)).on('mouseleave', untip);
  const cen = (ph) => ({ ph, x: mean(D.filter((p) => p.phenotype === ph).map((p) => p.ifng)), y: mean(D.filter((p) => p.phenotype === ph).map((p) => Math.log2(p.cd8))) });
  FW.layer('lab').selectAll('text').data(['desert', 'excluded', 'inflamed'].map(cen)).join('text').attr('x', (d) => FW.x(d.x)).attr('y', (d) => FW.y(d.y) - 14).attr('text-anchor', 'middle')
    .attr('font-size', 12.5).attr('font-weight', 700).attr('fill', ink()).attr('stroke', css('--surface')).attr('stroke-width', 3).attr('paint-order', 'stroke').text((d) => `${d.ph} (mean)`);
  FW.layer('lab').selectAll('circle').data(['desert', 'excluded', 'inflamed'].map(cen)).join('circle').attr('cx', (d) => FW.x(d.x)).attr('cy', (d) => FW.y(d.y)).attr('r', 5).attr('fill', 'none').attr('stroke', ink()).attr('stroke-width', 2);
  legend(fw.body, [[T.toy.c0, cat(0)], [T.toy.c1, cat(1)]]);
  const wAcc = [T.toy.refAcc, toyOf(2).vaAcc, toyOf(8).vaAcc], wTr = [T.toy.refTrAcc, toyOf(2).trAcc, toyOf(8).trAcc];
  function paintW() {
    surfW.draw(toyProb(wi));
    sW.set({ 'discovery accuracy': pc(wTr[wi]), 'Centre C accuracy': pc(wAcc[wi]), 'always predict "other"': pc(1 - mean(T.toy.yv)) });
  }
  paintW();
  prose(ch, `<p>Logistic regression reaches ${pc(T.toy.refAcc)} accuracy on Centre C, ${T.toy.refAcc <= 1 - mean(T.toy.yv) ? 'not above' : 'little above'} the ${pc(1 - mean(T.toy.yv))} obtained by always predicting "other": a single line cannot cut out a band. A network with two hidden neurons reaches ${pc(toyOf(2).vaAcc)} and with eight, ${pc(toyOf(8).vaAcc)}.</p>`);
  table(ch, [
    { a: 'Logistic regression', b: 'straight line / plane', d: 'small', t: 'strong baseline', i: 'odds ratios with CIs' },
    { a: 'Trees, random forest, boosting', b: 'axis-aligned boxes, many of them', d: 'moderate', t: 'often the best on tables', i: 'moderate (importances, SHAP)' },
    { a: '<b>Neural network</b>', b: 'smooth, arbitrarily curved', d: 'large', t: 'rarely better than boosting on small tables', i: 'low' },
    { a: 'Convolutional / transformer networks', b: 'learned features of images or sequences', d: 'very large', t: 'the standard for images and sequences', i: 'low' },
  ], [['a', 'Approach'], ['b', 'Boundary shape'], ['d', 'Data needed'], ['t', 'On tabular omics'], ['i', 'Interpretability']]);

  // =====================================================================================
  // 3. WHERE
  // =====================================================================================
  ch = chapter(main, 'where', 'Where is it used?', 'Neural networks are the standard when the inputs are images, sequences or very large collections of cells; they are one option among several on small tables.');
  cards(ch, [
    { tag: 'Digital pathology', title: 'Reading H&E slides', body: 'Convolutional networks classify tumour regions, grade cancers and predict molecular status from stained slides. The input is pixels, so there are no hand-made features to start from.' },
    { tag: 'Variant calling', title: 'DeepVariant', body: 'A convolutional network classifies images of aligned reads as reference, heterozygous or homozygous at each candidate site.' },
    { tag: 'Sequence to function', title: 'Splicing and regulation', body: 'SpliceAI, Enformer and related models predict splice sites or expression-related signals directly from DNA sequence, learning motifs in the first layers.' },
    { tag: 'Protein structure', title: 'AlphaFold', body: 'Deep networks (attention-based) predict three-dimensional structure from sequence and alignments.' },
    { tag: 'Single-cell', title: 'Autoencoders and scVI', body: 'A network compresses each cell\'s expression into a few latent dimensions and reconstructs it. scVI uses this for batch correction, denoising and clustering.' },
    { tag: 'Cell annotation', title: 'Label transfer', body: 'Classifiers trained on annotated atlases assign cell types to new data, including network-based tools such as scANVI.' },
    { tag: 'Foundation models', title: 'Pretrained on millions of cells', body: 'Transformer networks pretrained on large cell collections are fine-tuned for annotation or perturbation prediction. Benchmarks of their benefit over simple baselines are mixed.' },
    { tag: 'Drug discovery', title: 'Molecular property prediction', body: 'Graph networks over atoms and bonds predict activity and toxicity of compounds.' },
    { tag: 'Inside this page', title: 'Response from a biomarker table', body: 'A small network on IFN-γ, TMB, CAF and four further measurements. See the comparison below: this is the case where a network is rarely worth its cost.' },
  ]);
  prose(ch, `<p><b>Comparison on this cohort.</b> The true response rule is linear in three measurements, so a flexible model has nothing extra to find. Networks with one hidden layer were trained on the ${n} discovery patients (200 epochs, 5 random initialisations each) and evaluated on Centre C. AUC is shown as mean (range over initialisations).</p>`);
  table(ch, [
    { m: 'Logistic regression', a: f3(cmp.f3.lr), b: f3(cmp.f7.lr) },
    { m: 'Network, 4 hidden neurons', a: rng3(cmp.f3.h4), b: rng3(cmp.f7.h4) },
    { m: 'Network, 16 hidden neurons', a: rng3(cmp.f3.h16), b: rng3(cmp.f7.h16) },
  ], [['m', 'Model'], ['a', 'AUC, 3 true features'], ['b', 'AUC, 7 features (adds PD-L1, CD8, proliferation, age)']]);
  const allAuc = [...cmp.f3.h4, ...cmp.f3.h16, ...cmp.f7.h4, ...cmp.f7.h16], allMin = d3.min(allAuc), allMax = d3.max(allAuc);
  const lrBest = Math.max(cmp.f3.lr, cmp.f7.lr), nnBest = Math.max(mean(cmp.f3.h4), mean(cmp.f3.h16), mean(cmp.f7.h4), mean(cmp.f7.h16));
  prose(ch, `<p><b>When not to use one:</b> ${nnBest > lrBest + 0.005 ? 'here the best network mean exceeds the logistic regression, but only by ' + f3(nnBest - lrBest) + ' AUC, within the spread across initialisations' : 'on this cohort the best network mean (' + f3(nnBest) + ') does not exceed the logistic regression (' + f3(lrBest) + ')'}, and the networks vary from one initialisation to the next (single runs range from ${f3(allMin)} to ${f3(allMax)}, so one run can land above or below the logistic regression). For a table with a few hundred patients and tens to thousands of measurements, a penalised logistic regression or boosted trees are usually as accurate, faster to tune, and easier to interpret and validate. A network earns its cost when there are many thousands of samples or cells, raw images or sequences, or a task such as reconstruction or embedding that other tools do not cover.</p>`);

  // =====================================================================================
  // 4. HOW
  // =====================================================================================
  ch = chapter(main, 'how', 'How it works', 'Six steps: one neuron, a hidden layer, activation functions, backpropagation on a worked example, training live, and overfitting.');

  const mu = T.toy.st.mu, sdv = T.toy.st.sd;
  const boundary = (w1, w2, b, t, xr, yr) => {
    const [m0, m1] = t.st.mu, [s0, s1] = t.st.sd;
    if (Math.abs(w2) < 1e-6) { const x = m0 + s0 * (-b / (w1 || 1e-6)); return [[x, yr[0]], [x, yr[1]]]; }
    const yAt = (x) => m1 + s1 * (-(b + w1 * (x - m0) / s0) / w2);
    return [[xr[0], yAt(xr[0])], [xr[1], yAt(xr[1])]];
  };

  // ---- Step 1: one neuron
  let rb = rung(ch, 1, 'One neuron is a logistic regression',
    `<p>A neuron computes <b>z = w₁x₁ + w₂x₂ + b</b> and outputs <b>σ(z) = 1 / (1 + e<sup>−z</sup>)</b>. With inputs IFN-γ and log2 CD8 (each standardised), the weights w₁, w₂ and the bias b position a boundary. Its p = 0.5 line is where z = 0, always straight. Dragging the sliders changes that line; the fitted logistic regression is the best single neuron.</p>`);
  const f1g = figure(rb, 'A single neuron on the excluded-tumour task', 'Shading: predicted probability of "immune-excluded". Solid line: p = 0.5. The "Fit" button sets the weights to the logistic-regression solution.');
  const N1 = { w1: 1.0, w2: 0.5, b: -0.3 };
  const sl1 = [['w₁ (IFN-γ)', 'w1', -4, 4], ['w₂ (log2 CD8)', 'w2', -4, 4], ['bias b', 'b', -3, 3]].map(([lab, k, lo, hi]) => slider(f1g.controls, { label: lab, min: lo, max: hi, step: 0.05, value: N1[k], fmt: f2, onInput: (v) => { N1[k] = v; paint1(); } }));
  button(f1g.controls, 'Fit (logistic regression)', () => { N1.w1 = T.toy.lr.w[0]; N1.w2 = T.toy.lr.w[1]; N1.b = T.toy.lr.b0; sl1.forEach((s, i) => s.set([N1.w1, N1.w2, N1.b][i])); paint1(); }, true);
  const s1 = stats(f1g.root);
  const F1 = frame(f1g.body, { width: 820, height: 340, x: T.toy.xr, y: T.toy.yr, xLabel: T.toy.xl, yLabel: T.toy.yl, grid: false });
  const prob1 = (a, b2) => { const z = T.toy.raw(a, b2); return sigmoid(N1.w1 * z[0] + N1.w2 * z[1] + N1.b); };
  const surf1 = decisionCanvas(F1, prob1, { res: 4, alpha: 0.4 });
  F1.g.append('clipPath').attr('id', 'nn1-clip').append('rect').attr('width', F1.w).attr('height', F1.h);
  const bl1 = F1.layer('b').attr('clip-path', 'url(#nn1-clip)');
  F1.layer('pts').selectAll('circle').data(D).join('circle').attr('cx', (d) => F1.x(d.ifng)).attr('cy', (d) => F1.y(Math.log2(d.cd8))).attr('r', 3.8)
    .attr('fill', (d, i) => cat(T.toy.y[i])).attr('fill-opacity', 0.85).attr('stroke', css('--surface')).attr('stroke-width', 0.8)
    .on('mousemove', (e, d) => tip(e, `${d.id} · ${d.phenotype}<br>IFN-γ ${d.ifng}, CD8 ${d.cd8}%`)).on('mouseleave', untip);
  F1.layer('t').append('text').attr('x', F1.w - 6).attr('y', 16).attr('text-anchor', 'end').attr('font-size', 12).attr('fill', ink2()).attr('stroke', css('--surface')).attr('stroke-width', 3).attr('paint-order', 'stroke').text('one neuron: one straight boundary, p = 0.5');
  legend(f1g.body, [[T.toy.c0, cat(0)], [T.toy.c1, cat(1)], ['p = 0.5 boundary', ink()]]);
  function paint1() {
    surf1.draw(prob1);
    const [a, b2] = boundary(N1.w1, N1.w2, N1.b, T.toy, T.toy.xr, T.toy.yr);
    bl1.selectAll('line').data([0]).join('line').attr('x1', F1.x(a[0])).attr('y1', F1.y(a[1])).attr('x2', F1.x(b2[0])).attr('y2', F1.y(b2[1])).attr('stroke', ink()).attr('stroke-width', 2.4);
    const pt = T.toy.X.map((z) => sigmoid(N1.w1 * z[0] + N1.w2 * z[1] + N1.b));
    s1.set({ 'discovery accuracy': pc(acc(pt, T.toy.y)), 'log-loss': f3(logLoss(pt, T.toy.y)), 'best single neuron: accuracy': pc(T.toy.refTrAcc), 'best single neuron: log-loss': f3(logLoss(T.toy.X.map(T.toy.lp), T.toy.y)) });
  }
  paint1();
  prose(rb, `<p>The fitted single neuron (logistic regression) reaches about ${pc(T.toy.refTrAcc)} accuracy on the discovery patients, and other weights for a straight boundary do no better in practice, while ${pc(1 - mean(T.toy.y))} of tumours are "other". The limit comes from the straight boundary, not from poor fitting.</p>`);

  // ---- Step 2: hidden layer
  rb = rung(ch, 2, 'A hidden layer combines several boundaries',
    `<p>A <b>hidden layer</b> contains several neurons that each see the same inputs. Each has its own boundary (z = 0, a dashed line below). The output neuron takes a weighted sum of the hidden outputs and applies a sigmoid, so it can place probability where several hidden neurons agree. Two lines can bound a band; more lines give smoother shapes. The network below was trained for 500 epochs with fixed seed.</p>`);
  const f2g = figure(rb, 'Number of hidden neurons', 'Dashed lines: the z = 0 boundary of each hidden neuron. Shading: output probability of the whole network. The slider selects among networks trained with 1, 2, 3, 4 and 8 hidden ReLU neurons.');
  let hi = 1;
  slider(f2g.controls, { label: 'hidden neurons', min: 0, max: HS.length - 1, step: 1, value: hi, fmt: (i) => String(HS[i]), onInput: (v) => { hi = v; paint2(); } });
  const s2 = stats(f2g.root);
  const F2 = frame(f2g.body, { width: 820, height: 340, x: T.toy.xr, y: T.toy.yr, xLabel: T.toy.xl, yLabel: T.toy.yl, grid: false });
  const surf2 = decisionCanvas(F2, (a, b2) => toyNets[hi].forward(T.toy.raw(a, b2)), { res: 4, alpha: 0.4 });
  F2.g.append('clipPath').attr('id', 'nn2-clip').append('rect').attr('width', F2.w).attr('height', F2.h);
  const bl2 = F2.layer('b').attr('clip-path', 'url(#nn2-clip)');
  F2.layer('pts').selectAll('circle').data(D).join('circle').attr('cx', (d) => F2.x(d.ifng)).attr('cy', (d) => F2.y(Math.log2(d.cd8))).attr('r', 3.6)
    .attr('fill', (d, i) => cat(T.toy.y[i])).attr('fill-opacity', 0.85).attr('stroke', css('--surface')).attr('stroke-width', 0.8)
    .on('mousemove', (e, d) => tip(e, `${d.id} · ${d.phenotype}`)).on('mouseleave', untip);
  F2.layer('t').append('text').attr('x', F2.w - 6).attr('y', 16).attr('text-anchor', 'end').attr('font-size', 12).attr('fill', ink2()).attr('stroke', css('--surface')).attr('stroke-width', 3).attr('paint-order', 'stroke').text('each dashed line = one hidden neuron');
  legend(f2g.body, [[T.toy.c0, cat(0)], [T.toy.c1, cat(1)], ['hidden-neuron boundary (z = 0)', ink2()]]);
  function paint2() {
    surf2.draw((a, b2) => toyNets[hi].forward(T.toy.raw(a, b2)));
    const net = toyNets[hi], H = HS[hi];
    const segs = d3.range(H).map((i) => boundary(net.W[0][i * 2], net.W[0][i * 2 + 1], net.b[0][i], T.toy, T.toy.xr, T.toy.yr));
    bl2.selectAll('line').data(segs).join('line').attr('x1', (s) => F2.x(s[0][0])).attr('y1', (s) => F2.y(s[0][1])).attr('x2', (s) => F2.x(s[1][0])).attr('y2', (s) => F2.y(s[1][1]))
      .attr('stroke', ink2()).attr('stroke-width', 1.6).attr('stroke-dasharray', '6 4');
    const e = toyEv[hi];
    s2.set({ 'hidden neurons': H, parameters: net.nParams, 'discovery accuracy': pc(e.trAcc), 'Centre C accuracy': pc(e.vaAcc), 'Centre C log-loss': f3(e.vaL) });
  }
  paint2();
  prose(rb, `<p>With one hidden neuron the network is still a monotonic function of one weighted sum, so it has a single boundary: Centre C accuracy ${pc(toyOf(1).vaAcc)}, ${toyOf(1).vaAcc < 1 - mean(T.toy.yv) ? 'below' : 'near'} the ${pc(1 - mean(T.toy.yv))} obtained by always predicting "other". Two neurons give a band: ${pc(toyOf(2).vaAcc)}. Further neurons add little here (${pc(toyOf(8).vaAcc)} with eight): the band is already captured, and extra capacity only matters for more complicated shapes.</p>`);

  // ---- Step 3: activation functions
  const WK = { W: [[0.9, -0.5], [0.3, 0.7]], b: [0.1, 0.2], v: [1.1, -0.7], c: 0.05 };
  rb = rung(ch, 3, 'Activation functions supply the bend',
    `<p>Without an activation function, every layer is a weighted sum, and a weighted sum of weighted sums is again a weighted sum: the whole network collapses to one neuron. A non-linear activation prevents this. Common choices:</p>
     <ul><li><b>ReLU</b>, max(0, z): zero for negative z, identity for positive z. Cheap, and its slope is 1 wherever it is active. The default for hidden layers.</li>
     <li><b>Sigmoid</b>, 1 / (1 + e<sup>−z</sup>): maps to 0–1; used at the output of a binary classifier. Flat for large |z|.</li>
     <li><b>tanh</b>: a sigmoid rescaled to −1…1 and centred at 0.</li></ul>`);
  const f3g = figure(rb, 'Three activation functions and their slopes', 'Solid: activation a(z). Dashed: its slope a′(z), the quantity backpropagation multiplies by. Drag z to read values.');
  let zA = 1.5, showD = true;
  slider(f3g.controls, { label: 'z', min: -6, max: 6, step: 0.05, value: zA, fmt: f2, onInput: (v) => { zA = v; paint3(); } });
  checkbox(f3g.controls, { label: 'show slopes (dashed)', checked: true, onChange: (c) => { showD = c; paint3(); } });
  const s3 = stats(f3g.root);
  const AC = [
    { k: 'relu', n: 'ReLU', c: cat(2), f: relu, d: (z) => (z > 0 ? 1 : 0) },
    { k: 'sig', n: 'sigmoid', c: cat(6), f: sigmoid, d: (z) => sigmoid(z) * (1 - sigmoid(z)) },
    { k: 'tanh', n: 'tanh', c: cat(4), f: Math.tanh, d: (z) => 1 - Math.tanh(z) ** 2 },
  ];
  const F3a = frame(f3g.body, { width: 820, height: 300, x: [-6, 6], y: [-1.3, 2.2], xLabel: 'z (weighted sum)', yLabel: 'a(z), a′(z)' });
  const gz = d3.range(-6, 6.01, 0.05), ln3 = d3.line();
  F3a.g.append('clipPath').attr('id', 'nn3-clip').append('rect').attr('width', F3a.w).attr('height', F3a.h);
  const L3 = F3a.layer('c').attr('clip-path', 'url(#nn3-clip)');
  AC.forEach((a) => {
    L3.append('path').attr('fill', 'none').attr('stroke', a.c).attr('stroke-width', 2.4).attr('d', ln3.x((z) => F3a.x(z)).y((z) => F3a.y(a.f(z)))(gz));
    L3.append('path').attr('class', 'sl-' + a.k).attr('fill', 'none').attr('stroke', a.c).attr('stroke-width', 1.6).attr('stroke-dasharray', '5 4').attr('d', d3.line().x((z) => F3a.x(z)).y((z) => F3a.y(a.d(z)))(gz));
  });
  const note3 = (x, yv2, t, anchor = 'start') => F3a.layer('t').append('text').attr('x', F3a.x(x)).attr('y', F3a.y(yv2)).attr('text-anchor', anchor).attr('font-size', 11.5).attr('fill', ink2()).attr('stroke', css('--surface')).attr('stroke-width', 3).attr('paint-order', 'stroke').text(t);
  note3(-5.9, 1.95, 'ReLU: slope 0 for z < 0 (unit off), 1 for z > 0');
  note3(-5.9, 1.7, `sigmoid: slope never exceeds ${f2(0.25)}, and is near 0 for |z| > 4`);
  note3(5.9, -1.15, 'tanh: slope up to 1, also flat in the tails', 'end');
  const mk3 = F3a.layer('m');
  legend(f3g.body, [['ReLU', AC[0].c], ['sigmoid', AC[1].c], ['tanh', AC[2].c]]);
  function paint3() {
    L3.selectAll('path[class^=sl-]').attr('display', showD ? null : 'none');
    mk3.selectAll('line').data([0]).join('line').attr('x1', F3a.x(zA)).attr('x2', F3a.x(zA)).attr('y1', 0).attr('y2', F3a.h).attr('stroke', muted()).attr('stroke-dasharray', '3 3');
    mk3.selectAll('circle').data(AC).join('circle').attr('cx', F3a.x(zA)).attr('cy', (a) => F3a.y(a.f(zA))).attr('r', 5.5).attr('fill', (a) => a.c).attr('stroke', css('--surface')).attr('stroke-width', 1.5);
    const o = {}; AC.forEach((a) => { o[`${a.n}: a(z) / slope`] = `${f2(a.f(zA))} / ${f2(a.d(zA))}`; }); s3.set(o);
  }
  paint3();
  // collapse of two linear layers, using the worked-example weights
  const lin = [0, 1].map((k) => WK.v[0] * WK.W[0][k] + WK.v[1] * WK.W[1][k]), linB = WK.v[0] * WK.b[0] + WK.v[1] * WK.b[1] + WK.c;
  prose(rb, `<p><b>Why stacking needs a non-linearity.</b> Take the two-neuron network of step 4 with identity activations. Its output is v₁(w₁₁x₁ + w₁₂x₂ + b₁) + v₂(w₂₁x₁ + w₂₂x₂ + b₂) + c = <b>${f2(lin[0])}x₁ ${lin[1] < 0 ? '−' : '+'} ${f2(Math.abs(lin[1]))}x₂ ${linB < 0 ? '−' : '+'} ${f2(Math.abs(linB))}</b>: one neuron, with a straight boundary.</p>`);
  quiz(rb, 'A network has 10 sigmoid layers. By how much can the gradient shrink at most, from the slope factors alone?',
    `Each layer multiplies the gradient by a sigmoid slope of at most ${f2(0.25)}. After 10 layers the factor is at most 0.25<sup>10</sup> = <b>${sci(0.25 ** 10)}</b>, so early layers receive almost no learning signal (<b>vanishing gradients</b>). A ReLU unit that is active has slope 1 and does not shrink the signal, one reason ReLU is the usual hidden activation.`);

  // ---- Step 4: backpropagation worked example
  const stF = standardize(D.map((p) => [p.ifng, p.caf]));
  const bp = (() => {
    const cand = pos.map((p) => ({ p, x: stF.apply([p.ifng, p.caf]) })), tgt = [0.8, -0.3];
    const pick = cand.slice().sort((a, b) => Math.hypot(a.x[0] - tgt[0], a.x[1] - tgt[1]) - Math.hypot(b.x[0] - tgt[0], b.x[1] - tgt[1]))[0];
    const x = pick.x, yT = 1;
    const z = [0, 1].map((j) => WK.W[j][0] * x[0] + WK.W[j][1] * x[1] + WK.b[j]), h = z.map(relu);
    const zo = WK.v[0] * h[0] + WK.v[1] * h[1] + WK.c, p = sigmoid(zo), L = -Math.log(p);
    const dz = p - yT, dv = h.map((q) => dz * q), dc = dz, dh = WK.v.map((v) => dz * v), dzh = dh.map((g, j) => g * (z[j] > 0 ? 1 : 0));
    const dW = [0, 1].map((j) => x.map((xi) => dzh[j] * xi)), db = dzh.slice();
    const eta = 0.5;
    const nW = WK.W.map((r, j) => r.map((w, k) => w - eta * dW[j][k])), nb = WK.b.map((b2, j) => b2 - eta * db[j]), nv = WK.v.map((v, j) => v - eta * dv[j]), nc = WK.c - eta * dc;
    const fwd = (W, b2, v, c) => { const hh = [0, 1].map((j) => relu(W[j][0] * x[0] + W[j][1] * x[1] + b2[j])); const q = sigmoid(v[0] * hh[0] + v[1] * hh[1] + c); return -Math.log(q); };
    const e = 1e-6, Wp = WK.W.map((r) => r.slice()), Wm = WK.W.map((r) => r.slice()); Wp[0][0] += e; Wm[0][0] -= e;
    const fd = (fwd(Wp, WK.b, WK.v, WK.c) - fwd(Wm, WK.b, WK.v, WK.c)) / (2 * e);
    return { P: pick.p, x, y: yT, z, h, zo, p, L, dz, dv, dc, dh, dzh, dW, db, nW, nb, nv, nc, eta, L2: fwd(nW, nb, nv, nc), fd };
  })();
  rb = rung(ch, 4, 'Backpropagation: the chain rule, worked on one patient',
    `<p>Training needs the <b>gradient</b>: how much the loss L changes if each weight changes slightly. The loss depends on a weight through a chain of steps (weight → z → activation → next z → … → p → L), so the derivative is the <b>product of the derivatives along the chain</b> (the chain rule). Backpropagation computes these products from the output backwards, reusing each partial result, so the cost is about that of one forward pass.</p>
     <p>Example: the network below has 2 inputs (standardised IFN-γ and CAF score), 2 ReLU hidden neurons and a sigmoid output. The weights are illustrative round numbers, not fitted. Patient <b>${bp.P.id}</b> (a responder, y = 1) has x = (${f2(bp.x[0])}, ${f2(bp.x[1])}). Move the slider through the steps.</p>`);
  const f4g = figure(rb, 'Forward pass, backward pass and one update', 'Step through the calculation. Numbers on nodes: values; numbers on lines: weights (forward steps) or gradients ∂L/∂weight (backward steps).');
  const STEPS = ['inputs', 'forward: hidden layer', 'forward: output and loss', 'backward 1: error at the output', 'backward 2: output weights and hidden neurons', 'backward 3: first-layer weights', `update: w ← w − η·∂L/∂w, η = ${bp.eta}`];
  let st4 = 0;
  const sl4 = slider(f4g.controls, { label: 'step', min: 0, max: STEPS.length - 1, step: 1, value: 0, fmt: (i) => `${i + 1}/${STEPS.length}`, onInput: (v) => { st4 = v; paint4(); } });
  button(f4g.controls, 'Next step', () => { st4 = (st4 + 1) % STEPS.length; sl4.set(st4); paint4(); }, true);
  const s4 = stats(f4g.root);
  const F4 = d3.select(f4g.body).append('svg').attr('viewBox', '0 0 820 340').attr('width', 820).attr('role', 'img').attr('aria-label', 'computation graph for backpropagation');
  const fdiff = f3;
  function paint4() {
    F4.selectAll('*').remove();
    const fwdOn = st4 >= 1, outOn = st4 >= 2, bwd = st4 >= 3;
    const NX = { x: 150, h: 390, o: 610, l: 765 }, NY = { x1: 100, x2: 250, h1: 100, h2: 250, o: 175 };
    const line = (x0, y0, x1, y1, col, w = 2, dash = null) => F4.append('line').attr('x1', x0).attr('y1', y0).attr('x2', x1).attr('y2', y1).attr('stroke', col).attr('stroke-width', w).attr('stroke-dasharray', dash);
    const lab = (x, yy, t, col = ink(), anchor = 'middle', bold = false) => F4.append('text').attr('x', x).attr('y', yy).attr('text-anchor', anchor).attr('font-size', 12).attr('font-weight', bold ? 700 : 400).attr('fill', col).attr('stroke', css('--surface')).attr('stroke-width', 3.5).attr('paint-order', 'stroke').text(t);
    const edges = [];
    [0, 1].forEach((j) => [0, 1].forEach((k) => edges.push({ x0: NX.x + 22, y0: k ? NY.x2 : NY.x1, x1: NX.h - 22, y1: j ? NY.h2 : NY.h1, t: j === k ? 0.5 : 0.15, side: j === k ? 0 : (k ? -1 : 1), kind: 'W', j, k })));
    [0, 1].forEach((j) => edges.push({ x0: NX.h + 22, y0: j ? NY.h2 : NY.h1, x1: NX.o - 24, y1: NY.o, t: 0.55, kind: 'v', j }));
    edges.forEach((e) => {   // all edges first, so no edge is drawn over another edge's label
      const gradCol = bwd && ((e.kind === 'v' && st4 >= 4) || (e.kind === 'W' && st4 >= 5));
      line(e.x0, e.y0, e.x1, e.y1, gradCol ? cat(5) : muted(), gradCol ? 2.4 : 1.6);
    });
    edges.forEach((e) => {
      const gradCol = bwd && ((e.kind === 'v' && st4 >= 4) || (e.kind === 'W' && st4 >= 5));
      const lx = e.x0 + (e.x1 - e.x0) * e.t, ly = e.y0 + (e.y1 - e.y0) * e.t - 5 + (e.kind === 'v' ? (e.j ? 24 : -4) : 0);
      let txt = null;
      if (e.kind === 'W') { const w = WK.W[e.j][e.k]; txt = st4 === 6 ? `${f2(w)} → ${f2(bp.nW[e.j][e.k])}` : gradCol ? `∂L/∂w = ${fdiff(bp.dW[e.j][e.k])}` : fwdOn || st4 === 0 ? `w = ${f2(w)}` : null; }
      else { const v = WK.v[e.j]; txt = st4 === 6 ? `${f2(v)} → ${f2(bp.nv[e.j])}` : gradCol ? `∂L/∂v = ${fdiff(bp.dv[e.j])}` : outOn || fwdOn ? `v = ${f2(v)}` : null; }
      if (txt) lab(lx + 5 * (e.side || 0), ly, txt, gradCol ? ink() : ink2(), e.side ? (e.side < 0 ? 'end' : 'start') : 'middle');   // crossing edges: label beside the line
    });
    const node = (cx, cy, r, fill, l1, l2) => { F4.append('circle').attr('cx', cx).attr('cy', cy).attr('r', r).attr('fill', fill).attr('stroke', css('--axis')).attr('stroke-width', 1.5); F4.append('text').attr('x', cx).attr('y', cy + 4).attr('text-anchor', 'middle').attr('font-size', 12.5).attr('font-weight', 700).attr('fill', ink()).text(l1); if (l2) lab(cx, cy + r + 15, l2, muted()); };
    node(NX.x, NY.x1, 22, css('--surface-2'), f2(bp.x[0]), null); lab(NX.x - 30, NY.x1 + 4, 'x₁ IFN-γ (z-score)', ink2(), 'end');
    node(NX.x, NY.x2, 22, css('--surface-2'), f2(bp.x[1]), null); lab(NX.x - 30, NY.x2 + 4, 'x₂ CAF score (z-score)', ink2(), 'end');
    [0, 1].forEach((j) => {
      const cy = j ? NY.h2 : NY.h1;
      node(NX.h, cy, 24, css('--surface-2'), fwdOn ? f2(bp.h[j]) : `h${j + 1}`, fwdOn ? `z = ${f2(bp.z[j])}, a = ReLU(z)` : null);
      if (st4 >= 4) lab(NX.h, cy + 55, `∂L/∂z = ${fdiff(bp.dzh[j])}`, ink(), 'middle', true);
      lab(NX.h, cy - 31, `h${'₁₂'[j]}`, ink2());
    });
    node(NX.o, NY.o, 26, css('--surface-2'), outOn ? f2(bp.p) : 'p', outOn ? `z = ${f2(bp.zo)}, p = σ(z)` : null);
    lab(NX.o, NY.o - 36, 'output', ink2());
    if (st4 >= 3) lab(NX.o + 20, NY.o + 74, `∂L/∂z = p − y = ${fdiff(bp.dz)}`, ink(), 'middle', true);
    F4.append('rect').attr('x', NX.l - 38).attr('y', NY.o - 24).attr('width', 76).attr('height', 48).attr('rx', 8).attr('fill', css('--surface-2')).attr('stroke', css('--axis'));
    F4.append('text').attr('x', NX.l).attr('y', NY.o - 4).attr('text-anchor', 'middle').attr('font-size', 11.5).attr('fill', ink2()).text(`y = ${bp.y}`);
    F4.append('text').attr('x', NX.l).attr('y', NY.o + 13).attr('text-anchor', 'middle').attr('font-size', 12.5).attr('font-weight', 700).attr('fill', ink()).text(outOn ? `L = ${f3(st4 === 6 ? bp.L2 : bp.L)}` : 'loss L');
    line(NX.o + 26, NY.o, NX.l - 38, NY.o, muted(), 1.6);
    lab(10, 330, `Step ${st4 + 1}: ${STEPS[st4]}`, ink(), 'start', true);
    if (bwd) lab(NX.l + 40, 332, '← gradients flow backwards', ink2(), 'end');
    s4.set({ step: `${st4 + 1} of ${STEPS.length}`, 'loss before / after one update': outOn ? `${f3(bp.L)} / ${f3(bp.L2)}` : '—', 'backprop ∂L/∂w₁₁ vs finite difference': st4 >= 5 ? `${fdiff(bp.dW[0][0])} vs ${fdiff(bp.fd)}` : '—' });
  }
  paint4();
  const cw = (j) => `z${'₁₂'[j]} = ${f2(WK.W[j][0])}×${f2(bp.x[0])} ${WK.W[j][1] < 0 ? '−' : '+'} ${f2(Math.abs(WK.W[j][1]))}×${f2(bp.x[1])} + ${f2(WK.b[j])} = ${f2(bp.z[j])}  → h${'₁₂'[j]} = ${f2(bp.h[j])}`;
  worked(rb, 'The numbers behind the figure',
    `<div class="calc">Forward
${cw(0)}
${cw(1)}
zₒ = ${f2(WK.v[0])}×${f2(bp.h[0])} ${WK.v[1] < 0 ? '−' : '+'} ${f2(Math.abs(WK.v[1]))}×${f2(bp.h[1])} + ${f2(WK.c)} = ${f2(bp.zo)}   p = σ(zₒ) = ${f3(bp.p)}   L = −ln p = ${f3(bp.L)}

Backward (chain rule, one factor per arrow)
∂L/∂zₒ   = p − y = ${f3(bp.dz)}                 (sigmoid + cross-entropy)
∂L/∂v₁   = ∂L/∂zₒ · h₁ = ${f3(bp.dz)} × ${f2(bp.h[0])} = ${f3(bp.dv[0])}
∂L/∂h₁   = ∂L/∂zₒ · v₁ = ${f3(bp.dz)} × ${f2(WK.v[0])} = ${f3(bp.dh[0])}
∂L/∂z₁   = ∂L/∂h₁ · ReLU′(z₁) = ${f3(bp.dh[0])} × ${bp.z[0] > 0 ? 1 : 0} = ${f3(bp.dzh[0])}
∂L/∂w₁₁  = ∂L/∂z₁ · x₁ = ${f3(bp.dzh[0])} × ${f2(bp.x[0])} = ${f3(bp.dW[0][0])}</div>
    <p>A weight is moved against its gradient: w₁₁ becomes ${f2(WK.W[0][0])} − ${bp.eta} × ${f3(bp.dW[0][0])} = ${f3(bp.nW[0][0])}. After updating all weights, the loss for this patient falls from ${f3(bp.L)} to ${f3(bp.L2)}. A finite-difference check (change w₁₁ by ±10<sup>−6</sup>, recompute L) gives ${f3(bp.fd)}, matching ${f3(bp.dW[0][0])}. ${bp.z[0] > 0 && bp.z[1] > 0 ? 'Both hidden neurons are active for this patient. ' : 'At least one hidden neuron is off for this patient. '}A ReLU that is off has ReLU′ = 0, so every weight feeding it receives zero gradient from this patient.</p>`);

  // ---- Step 5: live training lab
  rb = rung(ch, 5, 'Training live: the lab',
    `<p>Training repeats one cycle: forward pass, loss, backpropagation, update of all weights. The weights are updated on small random batches of patients (mini-batch gradient descent, here with the Adam update rule), and one pass through all ${n} discovery patients is an <b>epoch</b>. Start with the non-linear toy task, then switch to the real response task. Loss curves show the discovery patients and the ${V.length} Centre C patients; the dashed line is the logistic regression on the same two inputs.</p>`);
  const fl = figure(rb, 'Lab: train a network', 'Left: network output over the input plane (filled dots: discovery patients used for training; rings: Centre C). Right: log-loss per epoch. Press Play; change any setting to rebuild the network from a fresh initialisation.');
  const lab = { task: 'toy', H: 4, layers: 1, act: 'relu', lrLog: -1.7, seed: 1, epoch: 0, hist: [], playing: false, showV: true };
  const MAXE = 800;
  let netL = null, stopL = null;
  const ctl = (fn) => (v) => { fn(v); buildL(); };
  select(fl.controls, { label: 'Task', value: lab.task, options: [['toy', 'Toy: immune-excluded vs other (non-linear)'], ['resp', 'Response from IFN-γ and CAF (linear truth)']], onChange: (v) => { lab.task = v; rebuildTask(); buildL(); } });
  slider(fl.controls, { label: 'neurons per hidden layer', min: 1, max: 16, step: 1, value: lab.H, onInput: ctl((v) => { lab.H = v; }) });
  select(fl.controls, { label: 'hidden layers', value: '1', options: [['1', '1'], ['2', '2']], onChange: ctl((v) => { lab.layers = +v; }) });
  select(fl.controls, { label: 'activation', value: lab.act, options: [['relu', 'ReLU'], ['tanh', 'tanh'], ['sigmoid', 'sigmoid']], onChange: ctl((v) => { lab.act = v; }) });
  slider(fl.controls, { label: 'learning rate', min: -3, max: 0, step: 0.1, value: lab.lrLog, fmt: (v) => String(+(10 ** v).toPrecision(2)), onInput: (v) => { lab.lrLog = v; } });
  const playL = button(fl.controls, 'Play', () => (lab.playing ? pauseL() : playLoop()), true);
  button(fl.controls, 'Step (10 epochs)', () => { pauseL(); trainL(10); paintL(); });
  button(fl.controls, 'Reset', () => buildL());
  button(fl.controls, 'New initialisation', () => { lab.seed += 1; buildL(); });
  checkbox(fl.controls, { label: 'show Centre C points', checked: true, onChange: (c) => { lab.showV = c; drawPointsL(); } });
  const sL = stats(fl.root);
  const rowL = el('div', { class: 'fig-row' }); fl.body.append(rowL);
  const colA = el('div'), colB = el('div'); rowL.append(colA, colB);
  let tk = T.toy, FL = null, FC = null, surfL = null;
  const mkSurface = () => {
    colA.innerHTML = '';
    FL = frame(colA, { width: 480, height: 360, x: tk.xr, y: tk.yr, xLabel: tk.xl, yLabel: tk.yl, grid: false });
    surfL = decisionCanvas(FL, (a, b2) => (netL ? netL.forward(tk.raw(a, b2)) : 0.5), { res: 5, alpha: 0.42 });
    legend(colA, [[tk.c0, cat(0)], [tk.c1, cat(1)]]);
  };
  const drawPointsL = () => {
    FL.layer('tr').selectAll('circle').data(D).join('circle').attr('cx', (d) => FL.x(tk.featFn(d)[0])).attr('cy', (d) => FL.y(tk.featFn(d)[1])).attr('r', 3.6)
      .attr('fill', (d, i) => cat(tk.y[i])).attr('fill-opacity', 0.85).attr('stroke', css('--surface')).attr('stroke-width', 0.8)
      .on('mousemove', (e, d) => tip(e, `${d.id} (discovery) · ${d.phenotype}<br>${d.response ? 'responded' : 'no response'}`)).on('mouseleave', untip);
    FL.layer('va').selectAll('circle').data(lab.showV ? V : []).join('circle').attr('cx', (d) => FL.x(tk.featFn(d)[0])).attr('cy', (d) => FL.y(tk.featFn(d)[1])).attr('r', 3.8)
      .attr('fill', 'none').attr('stroke', (d, i) => cat(tk.yv[V.indexOf(d)])).attr('stroke-width', 1.6)
      .on('mousemove', (e, d) => tip(e, `${d.id} (Centre C) · ${d.phenotype}<br>${d.response ? 'responded' : 'no response'}`)).on('mouseleave', untip);
  };
  const colorC = { tr: cat(2), va: cat(3) };
  const mkLoss = () => {
    colB.innerHTML = '';
    colB.append(el('div', { class: 'fig-caption', style: 'margin:0 0 4px' }, 'Log-loss per epoch (lower is better)'));
    FC = frame(colB, { width: 480, height: 300, x: [0, MAXE], y: [0, 1], xLabel: 'epoch', yLabel: 'log-loss' });
    FC.layer('ref').append('line').attr('x1', 0).attr('x2', FC.w).attr('y1', FC.y(Math.min(1, tk.refLoss))).attr('y2', FC.y(Math.min(1, tk.refLoss))).attr('stroke', ink2()).attr('stroke-dasharray', '6 4').attr('stroke-width', 1.6);
    FC.layer('ref').append('text').attr('x', FC.w - 4).attr('y', FC.y(Math.min(1, tk.refLoss)) - 5).attr('text-anchor', 'end').attr('font-size', 11.5).attr('fill', ink2()).text(`logistic regression, Centre C: ${f3(tk.refLoss)}`);
    legend(colB, [['network, discovery (training)', colorC.tr], ['network, Centre C', colorC.va], ['logistic regression, Centre C', ink2()]]);
  };
  function rebuildTask() { tk = T[lab.task]; mkSurface(); mkLoss(); drawPointsL(); }
  function buildL() {
    pauseL();
    netL = makeMLP([2, ...Array(lab.layers).fill(lab.H), 1], { act: lab.act, seed: lab.seed });
    lab.epoch = 0; lab.hist = [];
    mkLoss(); paintL();
  }
  function trainL(k) {
    for (let e = 0; e < k && lab.epoch < MAXE; e++) {
      netL.epoch(tk.X, tk.y, 10 ** lab.lrLog, 32); lab.epoch++;
      lab.hist.push([lab.epoch, logLoss(netL.predict(tk.X), tk.y), logLoss(netL.predict(tk.Xv), tk.yv)]);
    }
  }
  function playLoop() {
    lab.playing = true; playL.textContent = 'Pause';
    stopL = animate(() => { if (!lab.playing) return false; trainL(3); paintL(); if (lab.epoch >= MAXE) { pauseL(); return false; } return true; });
  }
  function pauseL() { lab.playing = false; playL.textContent = 'Play'; if (stopL) { stopL(); stopL = null; } }
  function paintL() {
    surfL.draw((a, b2) => netL.forward(tk.raw(a, b2)));
    const mkp = (col, j) => FC.layer('c' + j).selectAll('path').data([lab.hist]).join('path').attr('fill', 'none').attr('stroke', col).attr('stroke-width', 2)
      .attr('d', d3.line().x((h) => FC.x(h[0])).y((h) => FC.y(Math.min(1, h[j])))(lab.hist));
    mkp(colorC.tr, 1); mkp(colorC.va, 2);
    const pv = netL.predict(tk.Xv), pt = netL.predict(tk.X);
    sL.set({ epoch: lab.epoch, parameters: netL.nParams, 'discovery log-loss': f3(logLoss(pt, tk.y)), 'Centre C log-loss': f3(logLoss(pv, tk.yv)), 'Centre C accuracy': pc(acc(pv, tk.yv)), 'Centre C AUC': f3(auc(pv, tk.yv)), 'logistic regression: accuracy / AUC': `${pc(tk.refAcc)} / ${f3(tk.refAuc)}` });
  }
  rebuildTask(); buildL();

  // reference runs behind the Try / What you see / What it means table
  const run = (task, o) => {
    const t = T[task], net = makeMLP([2, ...Array(o.layers || 1).fill(o.H || 4), 1], { act: o.act || 'relu', seed: o.seed || 1 });
    let prev = Infinity, rises = 0;
    for (let e = 0; e < (o.epochs || 150); e++) { net.epoch(t.X, t.y, o.lr || 0.02, 32); const l = logLoss(net.predict(t.X), t.y); if (l > prev) rises++; prev = l; }
    return { ...evalNet(net, t), rises };
  };
  const lrRuns = [0.002, 0.02, 1].map((lr) => run('toy', { lr, epochs: 150 }));
  const actRuns = ['relu', 'tanh', 'sigmoid'].map((act) => run('toy', { act, lr: 0.02, epochs: 150 }));
  const seedRuns = [1, 2, 3, 4, 5].map((seed) => run('toy', { H: 4, seed, epochs: 400 }));
  const respNet = run('resp', { H: 8, epochs: 400 });
  prose(rb, '<p><b>Suggested comparisons</b>:</p>');
  table(rb, [
    { t: 'Toy task: hidden neurons <b>1</b> → <b>2</b>', s: `Centre C accuracy ${pc(toyOf(1).vaAcc)} → <b>${pc(toyOf(2).vaAcc)}</b>`, m: 'One hidden neuron is still one boundary. A second allows a band: <b>capacity must match the shape of the problem</b>.' },
    { t: 'Switch to <b>Response</b> and train 8 neurons', s: `Centre C AUC ${f3(respNet.vaAuc)} vs logistic regression ${f3(T.resp.refAuc)}; the network curve ${respNet.vaL < T.resp.refLoss ? 'ends below' : 'does not go below'} the dashed line`, m: 'When the truth is linear, a flexible model has nothing extra to learn; at best it matches the straight boundary.' },
    { t: 'Learning rate <b>0.002</b> → <b>0.02</b> → <b>1</b> (toy, 150 epochs)', s: `Discovery log-loss ${f3(lrRuns[0].trL)} → ${f3(lrRuns[1].trL)} → <b>${f3(lrRuns[2].trL)}</b>; epochs in which it rose: ${lrRuns[0].rises} → ${lrRuns[1].rises} → <b>${lrRuns[2].rises}</b> of 150`, m: '<b>Step size controls training.</b> A small step is slow; a large step gives no further gain and can make the loss rise from one epoch to the next.' },
    { t: 'Activation <b>ReLU</b> → <b>tanh</b> → <b>sigmoid</b> (toy, 150 epochs)', s: `Discovery log-loss ${f3(actRuns[0].trL)} → ${f3(actRuns[1].trL)} → ${f3(actRuns[2].trL)}`, m: 'On a network this shallow the activations behave alike. Differences show mainly in deep networks, where sigmoid and tanh shrink gradients (step 3).' },
    { t: '<b>New initialisation</b> (toy, 4 neurons, 5 seeds)', s: `Centre C accuracy ${pc(d3.min(seedRuns, (r) => r.vaAcc))}–${pc(d3.max(seedRuns, (r) => r.vaAcc))}`, m: '<b>Random starting weights change the result.</b> Report the spread over several initialisations, not one run.' },
  ], [['t', 'Try'], ['s', 'What you see'], ['m', 'What it means']], { cls: 'compact' });
  prose(rb, '<p class="fig-caption">Centre C is displayed here only to show generalisation. Adjusting settings while watching it makes any final Centre C result optimistic; step 6 uses a separate monitoring set to decide when to stop.</p>');

  // ---- Step 6: overfitting, dropout, early stopping
  const ord = rng(21).shuffle(D.map((_, i) => i)), trI = ord.slice(0, 50), moI = ord.slice(50);
  const stO = standardize(trI.map((i) => F7.map((k) => FEATURES[k].get(D[i]))));
  const toZ = (p) => stO.apply(F7.map((k) => FEATURES[k].get(p)));
  const XO = trI.map((i) => toZ(D[i])), yO = trI.map((i) => y[i]), XM = moI.map((i) => toZ(D[i])), yM = moI.map((i) => y[i]), XC = V.map(toZ);
  const OVER = { none: ['No regularisation', {}], d3: ['Dropout 0.3', { dropout: 0.3 }], d5: ['Dropout 0.5', { dropout: 0.5 }], l2: ['L2 weight decay 0.01', { l2: 0.01 }] };
  const overCache = {};
  const overRun = (k) => {
    if (overCache[k]) return overCache[k];
    const net = makeMLP([7, 32, 32, 1], { seed: 5, ...OVER[k][1] }), rec = [];
    for (let e = 1; e <= 300; e++) {
      net.epoch(XO, yO, 0.001, 16);
      if (e === 1 || e % 3 === 0) { const pc2 = net.predict(XC); rec.push({ e, tr: logLoss(net.predict(XO), yO), mo: logLoss(net.predict(XM), yM), cl: logLoss(pc2, yv), ca: auc(pc2, yv) }); }
    }
    const best = rec.reduce((b, r) => (r.mo < b.mo ? r : b), rec[0]);
    return (overCache[k] = { rec, best, last: rec[rec.length - 1] });
  };
  const O0 = overRun('none');
  rb = rung(ch, 6, 'Overfitting, dropout and early stopping',
    `<p>A network has many weights (${makeMLP([7, 32, 32, 1]).nParams} in the one below), so with few patients it can fit noise. Three standard controls:</p>
     <ul><li><b>Early stopping.</b> Track the loss on a monitoring set that is not used for weight updates and keep the weights from its minimum.</li>
     <li><b>Dropout.</b> During training, each hidden neuron is switched off at random with probability <i>q</i> on every update, so no neuron can depend on specific others. All neurons are used at prediction time.</li>
     <li><b>Weight decay (L2).</b> A penalty on the size of the weights, as in ridge regression.</li></ul>
     <p>Set-up: 7 measurements (IFN-γ, log2 TMB, CAF, PD-L1, CD8, proliferation, age), a network with two hidden layers of 32 neurons, trained on only ${trI.length} discovery patients. The remaining ${moI.length} discovery patients form the monitoring set. Centre C is used once, for the final read-out.</p>`);
  const fo = figure(rb, 'Training and monitoring loss', 'Solid lines: log-loss on the training patients and on the monitoring patients over 300 epochs. Vertical line: the epoch with the lowest monitoring loss, where early stopping would halt.');
  let ok = 'none';
  select(fo.controls, { label: 'Regularisation', value: ok, options: Object.entries(OVER).map(([k, v]) => [k, v[0]]), onChange: (v) => { ok = v; paintO(); } });
  const sO = stats(fo.root);
  const FO = frame(fo.body, { width: 820, height: 320, x: [0, 300], y: [0, 2.6], xLabel: 'epoch', yLabel: 'log-loss' });
  const LO = FO.layer('c'), MO = FO.layer('m');
  legend(fo.body, [['training patients', cat(2)], ['monitoring patients', cat(3)], ['lowest monitoring loss', ink()]]);
  function paintO() {
    const r = overRun(ok), ln = d3.line().x((q) => FO.x(q.e));
    [['tr', cat(2)], ['mo', cat(3)]].forEach(([k, c], i) => LO.selectAll('path.p' + i).data([r.rec]).join('path').attr('class', 'p' + i).attr('fill', 'none').attr('stroke', c).attr('stroke-width', 2).attr('d', ln.y((q) => FO.y(Math.min(2.6, q[k])))(r.rec)));
    MO.selectAll('*').remove();
    MO.append('line').attr('x1', FO.x(r.best.e)).attr('x2', FO.x(r.best.e)).attr('y1', 0).attr('y2', FO.h).attr('stroke', ink()).attr('stroke-width', 1.6).attr('stroke-dasharray', '4 3');
    MO.append('text').attr('x', FO.x(r.best.e) + 6).attr('y', 16).attr('font-size', 11.5).attr('fill', ink()).text(`early stop: epoch ${r.best.e}`);
    MO.append('text').attr('x', FO.w - 6).attr('y', FO.y(Math.min(2.55, r.last.mo)) - 6).attr('text-anchor', 'end').attr('font-size', 11.5).attr('fill', ink2()).text(`monitoring loss at epoch 300: ${f2(r.last.mo)}`);
    MO.append('text').attr('x', FO.w - 6).attr('y', FO.y(0.28)).attr('text-anchor', 'end').attr('font-size', 11.5).attr('fill', ink2()).text(`training loss at epoch 300: ${f2(r.last.tr)}`);
    sO.set({ 'epoch of lowest monitoring loss': r.best.e, 'monitoring loss: best / final': `${f3(r.best.mo)} / ${f3(r.last.mo)}`, 'Centre C AUC: stopped at best / final': `${f3(r.best.ca)} / ${f3(r.last.ca)}`, 'Centre C log-loss: best / final': `${f3(r.best.cl)} / ${f3(r.last.cl)}` });
  }
  paintO();
  table(rb, Object.keys(OVER).map((k) => { const r = overRun(k); return { k: OVER[k][0], b: r.best.e, m: `${f3(r.best.mo)} / ${f3(r.last.mo)}`, c: `${f3(r.best.ca)} / ${f3(r.last.ca)}` }; }),
    [['k', 'Setting'], ['b', 'Best epoch'], ['m', 'Monitoring loss: best / final'], ['c', 'Centre C AUC: stopped at best / final']], { cls: 'compact' });
  const rising = O0.last.mo > O0.best.mo * 1.1;
  prose(rb, `<p>${rising ? `Without regularisation the training loss keeps falling (${f2(O0.last.tr)} at epoch 300) while the monitoring loss reaches its minimum at epoch ${O0.best.e} (${f2(O0.best.mo)}) and then rises to ${f2(O0.last.mo)}: the network has started to fit noise in the ${trI.length} patients.` : `In this run the monitoring loss did not rise markedly after its minimum (epoch ${O0.best.e}: ${f2(O0.best.mo)}; epoch 300: ${f2(O0.last.mo)}).`} In the table, dropout and weight decay reduce the rise in monitoring loss (final values ${f2(overRun('d3').last.mo)}, ${f2(overRun('d5').last.mo)} and ${f2(overRun('l2').last.mo)} against ${f2(O0.last.mo)}), and the Centre C AUC at epoch 300 is ${f3(O0.last.ca)} without regularisation and ${f3(d3.min(['d3', 'd5', 'l2'], (k) => overRun(k).last.ca))}–${f3(d3.max(['d3', 'd5', 'l2'], (k) => overRun(k).last.ca))} with it. Stopped at the best monitoring epoch, all four settings give ${f3(d3.min(Object.keys(OVER), (k) => overRun(k).best.ca))}–${f3(d3.max(Object.keys(OVER), (k) => overRun(k).best.ca))}. These come from one seed and one small sample, so differences at the second decimal or below should not be read as a ranking. Early stopping needs a monitoring set separate from the final test data; stopping on Centre C would make its AUC optimistic.</p>`);

  // =====================================================================================
  // 5. PITFALLS
  // =====================================================================================
  ch = chapter(main, 'pitfalls', 'Pitfalls', 'Common problems, how to recognise them, and how to address them.');
  const rawX = D.map((p) => F7.map((k) => (k === 'log2tmb' ? p.tmb : FEATURES[k].get(p))));
  const netRaw = fitNet(rawX, y, [16], { epochs: 150, lr: 0.01, seed: 1 }), netStd = fitNet(standardize(D.map((p) => F7.map((k) => FEATURES[k].get(p)))).Z, y, [16], { epochs: 150, lr: 0.01, seed: 1 });
  const lRaw = logLoss(netRaw.predict(rawX), y), lStd = logLoss(netStd.predict(standardize(D.map((p) => F7.map((k) => FEATURES[k].get(p)))).Z), y);
  cards(ch, [
    { tag: 'Symptom: loss barely falls', title: 'Unscaled inputs', body: `PD-L1 runs 0–100 while IFN-γ spans a few units, so some weights need far larger steps than others. Training the same 7-feature network for 150 epochs gives a discovery log-loss of <b>${f3(lRaw)}</b> on raw inputs and <b>${f3(lStd)}</b> on standardised ones. <b>Fix:</b> standardise every input using training-set means and standard deviations.` },
    { tag: 'Symptom: results change between runs', title: 'Random initialisation', body: `Starting weights are random. On the 7-feature task the validation AUC of a 16-neuron network ranged from <b>${f3(d3.min(cmp.f7.h16))}</b> to <b>${f3(d3.max(cmp.f7.h16))}</b> across 5 seeds. <b>Fix:</b> fix and report seeds, repeat training several times, and report the spread or average several networks.` },
    { tag: 'Symptom: training loss falls, monitoring loss rises', title: 'Overfitting', body: `With ${trI.length} training patients, the unregularised network reached a training loss of ${f2(O0.last.tr)} and a monitoring loss of ${f2(O0.last.mo)} at epoch 300 (best: ${f2(O0.best.mo)}). <b>Fix:</b> early stopping, dropout, weight decay, a smaller network, more data, or a simpler model.` },
    { tag: 'Symptom: early layers do not learn', title: 'Vanishing gradients and dead ReLU units', body: `Sigmoid or tanh layers shrink the gradient at every layer (at most ${sci(0.25 ** 10)} over 10 sigmoid layers). A ReLU unit outputs 0 and passes no gradient for any patient whose input to it is negative; a <b>dead</b> unit is one for which this holds for every patient, so it never learns again. Being off for some patients is normal: in the diagram network ${pc(deadShare)} of (patient, hidden neuron) pairs are off, and ${deadUnits} of the 3 hidden neurons ${deadUnits === 1 ? 'is' : 'are'} dead. <b>Fix:</b> ReLU-type activations, sensible initialisation, normalisation layers and residual connections in deep networks.` },
    { tag: 'Symptom: optimistic test result', title: 'Tuning on the test data', body: 'Choosing the number of epochs, layers or learning rate by watching the final test set makes the reported performance optimistic, as with any model selection. <b>Fix:</b> tune on a monitoring split or by nested cross-validation, then evaluate once on untouched patients (such as a new centre).' },
    { tag: 'Symptom: confident but wrong probabilities', title: 'Poor calibration', body: `Overfitted networks produce extreme probabilities. At epoch 300 the unregularised network has Centre C log-loss <b>${f3(O0.last.cl)}</b> versus <b>${f3(O0.best.cl)}</b> at its best monitoring epoch (AUC ${f3(O0.last.ca)} vs ${f3(O0.best.ca)}): the log-loss degrades far more than the AUC. <b>Fix:</b> check calibration curves and recalibrate if the probabilities will be used.` },
    { tag: 'Symptom: weights cannot be read', title: 'No odds ratios', body: 'A hidden-layer weight is not a log odds ratio: its effect depends on the other neurons and on the activation. <b>Fix:</b> use post-hoc tools (permutation importance, SHAP) with their own caveats, or choose a model whose coefficients can be reported if interpretation is the goal.' },
  ]);

  // =====================================================================================
  // 6. RECAP
  // =====================================================================================
  ch = chapter(main, 'recap', 'Recap', 'Key points.');
  recap(ch, [
    'A neuron is a <b>weighted sum followed by an activation</b>; with a sigmoid it is a logistic regression. A network stacks layers of such neurons.',
    'Hidden layers let the model draw <b>several boundaries and combine them</b>. One hidden neuron still gives a single boundary; two can bound a band.',
    'A <b>non-linear activation</b> (ReLU, tanh, sigmoid) is essential: without it every stack of layers collapses to one weighted sum.',
    '<b>Backpropagation</b> applies the chain rule from the output backwards: each weight\'s gradient is the product of the local derivatives along its path to the loss.',
    'Training repeats forward pass, loss, backpropagation and a small step against the gradient, over many epochs and mini-batches. Learning rate, initialisation and scaling matter.',
    '<b>Overfitting</b> is controlled by early stopping on a separate monitoring set, dropout, weight decay and smaller networks.',
    `On <b>tabular omics</b> the gain over penalised logistic regression or boosted trees is usually small: on this cohort the best mean network AUC (over random initialisations) was ${f3(nnBest)} against ${f3(lrBest)} for logistic regression. Networks pay off with images, sequences and very large samples.`,
    'Report results over several random initialisations, tune on data separate from the final test set, and check calibration.',
  ]);
  prose(ch, '<h3>In code</h3><p>scikit-learn\'s <code>MLPClassifier</code> is sufficient for small tabular networks; PyTorch or Keras are used for images, sequences and custom architectures. Gradients are computed automatically (backpropagation is built in).</p>');
  code(ch, {
    Python: `import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

feats = ["ifng", "log2tmb", "caf", "pdl1", "cd8", "prolif", "age"]
train["log2tmb"] = np.log2(train["tmb"]);  test["log2tmb"] = np.log2(test["tmb"])

# Network: standardise, hidden layers, L2 penalty (alpha), early stopping on an internal 20% split
aucs = []
for seed in range(5):
    net = make_pipeline(StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(16,), alpha=1e-2, early_stopping=True,
                      validation_fraction=0.2, n_iter_no_change=20, max_iter=1000, random_state=seed))
    net.fit(train[feats], train["response"])
    aucs.append(roc_auc_score(test["response"], net.predict_proba(test[feats])[:, 1]))
print("network AUC: mean %.3f, range %.3f-%.3f" % (np.mean(aucs), min(aucs), max(aucs)))

# Baseline to beat
lr = make_pipeline(StandardScaler(), LogisticRegression(C=1.0)).fit(train[feats], train["response"])
print("logistic AUC:", roc_auc_score(test["response"], lr.predict_proba(test[feats])[:, 1]))

# The same network in PyTorch, showing the training loop (backpropagation is loss.backward())
import torch, torch.nn as nn
Xt = torch.tensor(StandardScaler().fit_transform(train[feats]), dtype=torch.float32)
yt = torch.tensor(train["response"].values, dtype=torch.float32)
model = nn.Sequential(nn.Linear(len(feats), 16), nn.ReLU(), nn.Dropout(0.3), nn.Linear(16, 1))
opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-2)
for epoch in range(200):
    model.train(); opt.zero_grad()
    loss = nn.functional.binary_cross_entropy_with_logits(model(Xt).squeeze(1), yt)
    loss.backward(); opt.step()`,
    R: `train$log2tmb <- log2(train$tmb); test$log2tmb <- log2(test$tmb)
feats <- c("ifng", "log2tmb", "caf", "pdl1", "cd8", "prolif", "age")

# Standardise with training-set means and SDs
mu <- colMeans(train[feats]); s <- apply(train[feats], 2, sd)
Xtr <- scale(train[feats], mu, s); Xte <- scale(test[feats], mu, s)

# One hidden layer, weight decay; repeat over seeds and report the spread
library(nnet); library(pROC)
aucs <- sapply(1:5, function(seed) {
  set.seed(seed)
  fit <- nnet(Xtr, train$response, size = 16, decay = 1e-2, maxit = 500, entropy = TRUE, trace = FALSE)
  as.numeric(auc(test$response, as.vector(predict(fit, Xte))))
})
c(mean = mean(aucs), min = min(aucs), max = max(aucs))

# Baseline to beat
lr <- glm(train$response ~ Xtr, family = binomial)
auc(test$response, as.vector(plogis(cbind(1, Xte) %*% coef(lr))))

# Larger or deeper networks: the keras / torch packages follow the same fit-and-validate pattern`,
  });
  seeAlso(main, [['concepts', 'logistic-regression', 'Logistic regression'], ['concepts', 'gradient-descent', 'Gradient descent'], ['concepts', 'regularization', 'Regularisation'], ['concepts', 'cross-validation', 'Cross-validation and leakage'], ['concepts', 'tree-ensembles', 'Random forests and boosting'], ['concepts', 'transformers', 'Embeddings, attention and transformers']]);
}
