// Gradient descent, lesson format.
// Arc: what → why → where → how it works (6 steps: loss as "how wrong" → gradient as slope → learning rate on the
//      real loss surface (raw vs centred IFN-γ) → stochastic and mini-batch → momentum and Adam → training lab for
//      the response model with try/see/means table) → pitfalls → recap.
// Model on the cohort: linear regression of CD8 % on IFN-γ (MSE), then logistic regression of response (log-loss).
// Every number in the prose is computed from data/study/cohort.json at render time.
import { header, figure, slider, select, button, checkbox, stats, legend, seeAlso, animate, tip, untip, el, html } from '../lib/ui.js';
import { frame, cat, css, ink, ink2, muted } from '../lib/plot.js';
import { rng, mean, sigmoid, eig2 } from '../lib/data.js';
import { fitLogistic, predictProb, logLoss, auc, standardize } from '../lib/models.js';
import { loadCohort, FEATURES, studyCard } from '../lib/study.js';
import { lessonNav, chapter, prose, rung, worked, quiz, cards, table, code, recap } from '../lib/lesson.js';
const d3 = window.d3;

const um = (t) => t.replace(/^-/, '−');
const f1 = (v) => um(v.toFixed(1));
const f2 = (v) => (Math.abs(v) < 0.005 ? '0.00' : um(v.toFixed(2)));
const f3 = (v) => um(v.toFixed(3));
const pc = (v) => Math.round(100 * v) + '%';
const sgn = (v) => (v < 0 ? '− ' + f2(-v) : '+ ' + f2(v));
const fLr = (v) => (v >= 0.1 ? v.toFixed(2) : v >= 0.01 ? v.toFixed(3) : v.toFixed(4));
const expo = (v) => (v >= 0.01 ? f2(v) : v.toExponential(1));
let clipN = 0;

/** Sufficient statistics of the MSE surface for prediction a + b·x̃ (x̃ = IFN-γ raw or centred). */
function surface(xs, ys, shift) {
  const xt = xs.map((x) => x - shift), m = (f) => mean(xt.map(f));
  const Sx = mean(xt), Sxx = m((x) => x * x), Sy = mean(ys), Syy = mean(ys.map((v) => v * v)), Sxy = mean(xt.map((x, i) => x * ys[i]));
  const L = (a, b) => Syy - 2 * a * Sy - 2 * b * Sxy + a * a + 2 * a * b * Sx + b * b * Sxx;
  const g = (a, b) => [2 * (a + b * Sx - Sy), 2 * (a * Sx + b * Sxx - Sxy)];
  const bB = (Sxy - Sx * Sy) / (Sxx - Sx * Sx), aB = Sy - bB * Sx;
  const lam = eig2(2, 2 * Sx, 2 * Sxx).vals;
  return { L, g, best: [aB, bB], Lmin: L(aB, bB), lam, xt, shift };
}

/** Contour map of a loss surface into `parent`. Returns {F, clip, trail(layerName)}. */
function lossMap(parent, S, { A, B, width = 460, height = 400, xLabel, yLabel, onClick }) {
  const F = frame(parent, { width, height, x: A, y: B, xLabel, yLabel, grid: false });
  const id = 'gd-clip-' + (++clipN);
  F.svg.append('defs').append('clipPath').attr('id', id).append('rect').attr('width', F.w).attr('height', F.h);
  const n = 80, vals = new Array(n * n);
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    const a = A[0] + (A[1] - A[0]) * (i + 0.5) / n, b = B[1] - (B[1] - B[0]) * (j + 0.5) / n;
    vals[j * n + i] = Math.log(S.L(a, b) - S.Lmin + 1);
  }
  const mx = d3.max(vals), levels = d3.range(15).map((k) => mx * k / 15);
  const shade = d3.scaleLinear().domain([0, levels.length - 1]).range([css('--accent'), css('--surface-2')]);
  const proj = d3.geoTransform({ point(px, py) { this.stream.point(F.x(A[0] + (A[1] - A[0]) * px / n), F.y(B[1] - (B[1] - B[0]) * py / n)); } });
  F.layer('contours').attr('clip-path', `url(#${id})`).selectAll('path').data(d3.contours().size([n, n]).thresholds(levels)(vals)).join('path')
    .attr('d', d3.geoPath(proj)).attr('fill', (d, i) => shade(i)).attr('fill-opacity', 0.85).attr('stroke', css('--surface')).attr('stroke-width', 0.6);
  F.layer('min').append('path').attr('d', d3.symbol(d3.symbolCross, 110)).attr('transform', `translate(${F.x(S.best[0])},${F.y(S.best[1])})`).attr('fill', ink()).attr('stroke', css('--surface'));
  F.g.append('rect').attr('width', F.w).attr('height', F.h).attr('fill', 'transparent').style('cursor', onClick ? 'crosshair' : 'default')
    .on('mousemove', (e) => { const [px, py] = d3.pointer(e); const a = F.x.invert(px), b = F.y.invert(py); tip(e, `a = ${f2(a)}, b = ${f2(b)}<br>MSE = ${f1(S.L(a, b))}`); })
    .on('mouseleave', untip).on('click', (e) => { if (onClick) { const [px, py] = d3.pointer(e); onClick([F.x.invert(px), F.y.invert(py)]); } });
  const trail = (name) => F.layer(name).attr('clip-path', `url(#${id})`).style('pointer-events', 'none');
  return { F, trail };
}

/** Train logistic regression by (mini-batch) gradient descent. Pure; deterministic given cfg. */
function trainLogistic(Xtr, ytr, Xva, yva, { opt = 'gd', lr = 0.3, batch = 180, epochs = 100, seed = 5 }) {
  const n = Xtr.length, p = Xtr[0].length, r = rng(seed);
  let w = new Array(p + 1).fill(0);
  const m = new Array(p + 1).fill(0), v = new Array(p + 1).fill(0);
  const prob = (row) => sigmoid(w[0] + row.reduce((s, x, j) => s + x * w[j + 1], 0));
  const tr = [logLoss(Xtr.map(prob), ytr)], va = [logLoss(Xva.map(prob), yva)];
  let t = 0, diverged = false;
  for (let e = 0; e < epochs && !diverged; e++) {
    const order = batch >= n ? d3.range(n) : r.shuffle(d3.range(n));
    for (let s = 0; s < n; s += batch) {
      const idx = order.slice(s, s + batch), g = new Array(p + 1).fill(0);
      idx.forEach((i) => { const err = prob(Xtr[i]) - ytr[i]; g[0] += err; for (let j = 0; j < p; j++) g[j + 1] += err * Xtr[i][j]; });
      t++;
      for (let j = 0; j <= p; j++) {
        const gj = g[j] / idx.length;
        if (opt === 'gd') w[j] -= lr * gj;
        else if (opt === 'mom') { m[j] = 0.9 * m[j] + gj; w[j] -= lr * m[j]; }
        else { m[j] = 0.9 * m[j] + 0.1 * gj; v[j] = 0.999 * v[j] + 0.001 * gj * gj; w[j] -= lr * (m[j] / (1 - 0.9 ** t)) / (Math.sqrt(v[j] / (1 - 0.999 ** t)) + 1e-8); }
      }
      if (!w.every((q) => isFinite(q) && Math.abs(q) < 1e6)) { diverged = true; break; }
    }
    tr.push(diverged ? NaN : logLoss(Xtr.map(prob), ytr)); va.push(diverged ? NaN : logLoss(Xva.map(prob), yva));
  }
  const pv = diverged ? null : Xva.map(prob);
  return { w, tr, va, diverged, updates: t, pv, aucV: pv ? auc(pv, yva) : NaN };
}

export default async function render(main) {
  const C = await loadCohort();
  const D = C.discovery, V = C.validation, n = D.length;
  const xs = D.map((p) => p.ifng), ys = D.map((p) => p.cd8), xbar = mean(xs);
  const RAW = surface(xs, ys, 0), CEN = surface(xs, ys, xbar);
  const y = D.map((p) => p.response), yv = V.map((p) => p.response);

  header(main, 'Gradient descent',
    'The procedure that fits most machine-learning models. A <b>loss function</b> scores how wrong the current model is; the <b>gradient</b> gives the direction in which the loss rises fastest; gradient descent takes a small step in the opposite direction and repeats until the loss stops falling.');
  lessonNav(main);
  studyCard(main, C, '<b>Question for this page:</b> how does a model find its weights? The running model predicts CD8 T-cell % from the IFN-γ score (a straight line), and later the probability of response from three biopsy features.');

  // =====================================================================================
  // 1. WHAT
  // =====================================================================================
  let ch = chapter(main, 'what', 'What is gradient descent?', 'In one line: <b>measure the error → find the downhill direction → take a small step → repeat</b>.');
  prose(ch, `<p>A model has <b>parameters</b> (weights), and each setting of the parameters makes different predictions. Fitting means choosing the setting with the lowest <b>loss</b>, a single number that summarises how far the predictions are from the observed values. Gradient descent is an <b>iterative</b> method for this: it starts from an initial guess and improves it one small step at a time, using only local information about the slope of the loss.</p>
    <p>The diagram follows the first step on the study data. The model is a straight line, CD8 % ≈ a + b × (IFN-γ − ${f2(xbar)}), where ${f2(xbar)} is the mean IFN-γ score, so <i>a</i> is the predicted CD8 % for a patient with average IFN-γ and <i>b</i> is the change per unit of IFN-γ.</p>`);
  const eta0 = 0.1, g0 = CEN.g(0, 0), a1 = -eta0 * g0[0], b1 = -eta0 * g0[1];
  ch.append(loopSvg(CEN, eta0));
  prose(ch, '<p>Key terms:</p>');
  table(ch, [
    { t: 'Parameters (θ)', m: 'The numbers the model learns', s: 'intercept <i>a</i> and slope <i>b</i> of the line' },
    { t: 'Loss <i>L</i>(θ)', m: 'One number measuring how wrong the predictions are, averaged over patients', s: `mean squared error (MSE) = ${f1(CEN.L(0, 0))} at a = b = 0` },
    { t: 'Loss surface', m: 'The loss as a function of every possible parameter setting', s: 'a bowl over the (a, b) plane (step 3)' },
    { t: 'Gradient ∇<i>L</i>', m: 'The vector of slopes of the loss, one per parameter; it points uphill', s: `(${f1(g0[0])}, ${f1(g0[1])}) at the start` },
    { t: 'Learning rate (η)', m: 'The step-size multiplier applied to the gradient', s: `η = ${eta0}` },
    { t: 'Iteration / update', m: 'One application of θ ← θ − η∇L', s: `(0, 0) → (${f2(a1)}, ${f2(b1)})` },
    { t: 'Batch, epoch', m: 'The patients used for one gradient; one full pass through the training data', s: `full batch = all ${n} patients (step 4)` },
    { t: 'Convergence', m: 'The point at which further steps no longer reduce the loss appreciably', s: `minimum MSE = ${f1(CEN.Lmin)}` },
  ], [['t', 'Term'], ['m', 'Meaning'], ['s', 'In our study']]);

  // =====================================================================================
  // 2. WHY
  // =====================================================================================
  ch = chapter(main, 'why', 'Why do we need it?', 'Most models have no formula for their best parameters, and searching all settings is infeasible.');
  prose(ch, `<p>For a straight line fitted by least squares there is an exact formula (the normal equations), and it gives a = ${f2(CEN.best[0])}, b = ${f2(CEN.best[1])} for the study data. Most models are not so convenient:</p><ol>
    <li><b>No closed form.</b> Logistic regression, Cox regression, penalised models with a lasso term and every neural network must be fitted iteratively.</li>
    <li><b>Too many parameters for exact second-order methods.</b> Newton's method (IRLS) needs a p × p matrix of second derivatives. That is cheap for 4 parameters and impossible for the millions in a single-cell foundation model.</li>
    <li><b>Too much data for one pass.</b> Atlases with millions of cells do not fit in memory at once; gradients can be estimated from small random batches instead.</li></ol>`);
  let it01 = 0; { let a = 0, b = 0; while (CEN.L(a, b) - CEN.Lmin > 0.01 * CEN.Lmin && it01 < 1000) { const g = CEN.g(a, b); a -= eta0 * g[0]; b -= eta0 * g[1]; it01++; } }
  worked(ch, 'Grid search versus following the slope',
    `<p>A brute-force alternative is to evaluate the loss on a grid and keep the best point. With 100 candidate values per parameter:</p>
     <ul><li>the 2-parameter line needs 100<sup>2</sup> = <b>10,000</b> loss evaluations;</li>
     <li>a logistic model with three features and an intercept needs 100<sup>4</sup> = <b>100 million</b>;</li>
     <li>a network with 10,000 weights needs 10<sup>20,000</sup>, which is not computable.</li></ul>
     <p>Gradient descent with η = ${eta0} reaches within 1% of the minimum MSE in <b>${it01} steps</b>, each costing about one pass over the ${n} patients. Its cost grows with the number of steps, not exponentially with the number of parameters.</p>`);
  table(ch, [
    { a: 'Exact formula (normal equations)', u: 'Linear regression, ridge', c: 'one solve of a p × p system', l: 'exists only for a few models' },
    { a: 'Newton / IRLS', u: 'GLMs (glm, statsmodels), DESeq2, Cox', c: 'p × p second-derivative matrix per step', l: 'too costly beyond a few thousand parameters' },
    { a: '<b>Gradient descent</b> (full batch)', u: '<b>any differentiable loss</b>', c: '<b>one gradient per step</b>', l: 'needs a learning rate; slow on stretched surfaces' },
    { a: 'Stochastic / mini-batch GD', u: 'large data, neural networks', c: 'a gradient on a small batch', l: 'noisy; needs a decaying learning rate' },
    { a: 'Momentum, Adam', u: 'deep learning default', c: 'as SGD, plus running averages', l: 'extra settings (β₁, β₂)' },
  ], [['a', 'Approach'], ['u', 'Typical use'], ['c', 'Cost per step'], ['l', 'Limitation']]);

  // =====================================================================================
  // 3. WHERE
  // =====================================================================================
  ch = chapter(main, 'where', 'Where is it used?', 'Wherever a model is fitted by minimising a differentiable loss.');
  cards(ch, [
    { tag: 'Deep learning', title: 'Every neural network', body: 'Weights in scVI, Geneformer, scGPT and image models of H&amp;E slides are trained with mini-batch gradient descent, usually Adam or AdamW. Backpropagation is the method for computing the gradient.' },
    { tag: 'Single-cell', title: 'Latent-variable models', body: 'scVI and totalVI minimise a variational loss on mini-batches of cells; cell2location and Stereoscope fit spatial deconvolution models the same way.' },
    { tag: 'Embeddings', title: 't-SNE and UMAP', body: 't-SNE minimises a KL divergence by gradient descent with momentum; UMAP optimises its layout with stochastic gradient descent on a cross-entropy loss. The iteration count and learning rate change the picture.' },
    { tag: 'Spatial mapping', title: 'Tangram', body: 'Tangram maps single cells onto spatial spots by optimising a mapping matrix with gradient descent so that projected expression matches the measured spots.' },
    { tag: 'Penalised regression', title: 'Large lasso and elastic-net models', body: 'glmnet uses coordinate descent, a close relative that updates one weight at a time. scikit-learn\'s <code>SGDClassifier</code> fits logistic and linear models by SGD when the data are too large for exact solvers.' },
    { tag: 'Sequence models', title: 'Genomics deep learning', body: 'Models that predict chromatin accessibility or expression from DNA sequence (Enformer-style CNNs and transformers) are trained by Adam on millions of sequence windows.' },
    { tag: 'Matrix factorisation', title: 'Imputation and recommendation', body: 'Low-rank factorisations of cell × gene or drug × cell-line matrices are often fitted by stochastic gradient descent over observed entries.' },
    { tag: 'Outside biology', title: 'Language and vision models', body: 'Large language models and image classifiers are trained by the same algorithm, at much larger scale, with learning-rate warm-up and decay schedules.' },
  ]);
  prose(ch, `<p><b>When not to use it:</b> a closed-form solution exists (ordinary least squares); the model is small and a Newton/IRLS solver is faster and more exact (standard <code>glm</code> fits, Cox models); the objective is not differentiable or the parameters are discrete (decision-tree splits, cluster assignments in k-means, feature subsets), which need greedy or combinatorial methods instead.</p>`);

  // =====================================================================================
  // 4. HOW IT WORKS
  // =====================================================================================
  ch = chapter(main, 'how', 'How it works', 'Six steps, from measuring error to training the response model and checking it on Centre C.');

  // ---- Step 1: loss
  let rb = rung(ch, 1, 'A loss function measures how wrong the model is',
    `<p>For a prediction ŷ of a measured value y, the <b>residual</b> is y − ŷ. The <b>mean squared error</b> (MSE) averages the squared residuals over patients: MSE = (1/n) Σ (yᵢ − ŷᵢ)². Squaring stops positive and negative errors cancelling, makes the loss smooth, and penalises large errors more than small ones. Adjust the line and observe the MSE.</p>`);
  const fA = figure(rb, 'How wrong is this line?', 'Each dot is a discovery patient. Vertical segments are residuals (measured − predicted CD8 %). Hover a patient for its contribution to the loss.');
  let aA = 4, bA = 0.5;
  const sAa = slider(fA.controls, { label: 'a (CD8 % at mean IFN-γ)', min: 0, max: 20, step: 0.1, value: aA, fmt: f1, onInput: (v) => { aA = v; paintA(); } });
  const sAb = slider(fA.controls, { label: 'b (slope)', min: -2, max: 8, step: 0.05, value: bA, fmt: f2, onInput: (v) => { bA = v; paintA(); } });
  button(fA.controls, 'Least-squares line', () => { aA = CEN.best[0]; bA = CEN.best[1]; sAa.set(aA); sAb.set(bA); paintA(); });
  const sA = stats(fA.root);
  const FA = frame(fA.body, { width: 820, height: 320, x: [2.2, 10.2], y: [-5, 45], xLabel: 'IFN-γ score (log2 expression)', yLabel: 'CD8 T cells (% of cells)' });
  const resL = FA.layer('res'), lineA = FA.layer('line');
  const ptsA = FA.layer('pts').selectAll('circle').data(D).join('circle').attr('cx', (d) => FA.x(d.ifng)).attr('cy', (d) => FA.y(d.cd8)).attr('r', 4)
    .attr('fill', cat(0)).attr('stroke', css('--surface')).attr('stroke-width', 1);
  legend(fA.body, [['patient', cat(0)], ['current line', cat(2)], ['residual', ink2()]]);
  const predA = (x) => aA + bA * (x - xbar);
  function paintA() {
    resL.selectAll('line').data(D).join('line').attr('x1', (d) => FA.x(d.ifng)).attr('x2', (d) => FA.x(d.ifng)).attr('y1', (d) => FA.y(d.cd8)).attr('y2', (d) => FA.y(Math.max(-5, Math.min(45, predA(d.ifng)))))
      .attr('stroke', ink2()).attr('stroke-opacity', 0.45).attr('stroke-width', 1.2);
    lineA.selectAll('line').data([0]).join('line').attr('x1', FA.x(2.2)).attr('x2', FA.x(10.2)).attr('y1', FA.y(predA(2.2))).attr('y2', FA.y(predA(10.2))).attr('stroke', cat(2)).attr('stroke-width', 2.5);
    const r = D.map((d) => d.cd8 - predA(d.ifng)), sq = r.map((v) => v * v), mse = mean(sq);
    ptsA.on('mousemove', (e, d) => { const i = D.indexOf(d); tip(e, `${d.id} · IFN-γ ${d.ifng}, CD8 ${d.cd8}%<br>predicted ${f1(predA(d.ifng))}% · residual ${f1(r[i])}<br>squared error ${f1(sq[i])} (${pc(sq[i] / (n * mse))} of the total)`); }).on('mouseleave', untip);
    const top = sq.slice().sort((p, q) => q - p).slice(0, Math.round(n / 20));
    sA.set({ MSE: f1(mse), 'RMSE (√MSE, in %)': f1(Math.sqrt(mse)), 'mean |residual|': f1(mean(r.map(Math.abs))), [`share of MSE from the worst ${top.length} patients`]: pc(top.reduce((s, v) => s + v, 0) / (n * mse)) });
  }
  paintA();
  const rBest = D.map((d) => d.cd8 - CEN.best[0] - CEN.best[1] * (d.ifng - xbar)), sqB = rBest.map((v) => v * v).sort((p, q) => q - p);
  const topShare = sqB.slice(0, Math.round(n / 20)).reduce((s, v) => s + v, 0) / sqB.reduce((s, v) => s + v, 0);
  prose(rb, `<p>The least-squares line reaches MSE ${f1(CEN.Lmin)} (a typical error of √${f1(CEN.Lmin)} = ${f1(Math.sqrt(CEN.Lmin))} percentage points). The worst ${Math.round(n / 20)} patients (5%) contribute ${pc(topShare)} of that MSE: CD8 % is right-skewed and its spread grows with IFN-γ, and squaring gives those large residuals most of the weight.</p>`);
  const fit1 = fitLogistic(D.map((p) => [p.ifng]), y), pr1 = D.map((p) => sigmoid(fit1.b0 + fit1.w[0] * p.ifng));
  const iHi = d3.maxIndex(D, (p, i) => (p.response ? pr1[i] : -1)), iLo = d3.minIndex(D, (p, i) => (p.response ? pr1[i] : 2)), iFp = d3.maxIndex(D, (p, i) => (p.response ? -1 : pr1[i]));
  const ex = [iHi, iLo, iFp].map((i) => ({ p: D[i], q: pr1[i], l: -Math.log(D[i].response ? pr1[i] : 1 - pr1[i]) }));
  const base = mean(y);
  worked(rb, 'Log-loss for a yes/no outcome',
    `<p>For response (1/0) the model outputs a probability p, and the standard loss is the <b>log-loss</b> (binary cross-entropy): −ln p for a responder and −ln(1 − p) for a non-responder. It is small when the model gave high probability to what happened and large when it was confidently wrong. With a logistic regression on IFN-γ:</p>
     <div class="calc">${ex.map((e) => `${e.p.id}  ${e.p.response ? 'responded   ' : 'no response '}  p = ${f3(e.q)}   loss = −ln(${e.p.response ? f3(e.q) : f3(1 - e.q)}) = ${f2(e.l)}`).join('\n')}</div>
     <p>Mean log-loss over all ${n} patients: <b>${f3(logLoss(pr1, y))}</b>. Predicting p = 0.5 for everyone gives ln 2 = ${f3(Math.LN2)}; predicting the response rate (${f2(base)}) for everyone gives ${f3(logLoss(y.map(() => base), y))}. Minimising log-loss is the same as maximising the likelihood.</p>`);
  quiz(rb, 'Why not use the plain sum of residuals as the loss?', 'Positive and negative residuals cancel, so a poor line through the middle of the data could score 0. Squaring (or taking absolute values) prevents this; squaring also gives a smooth bowl whose slope is defined everywhere, which gradient descent needs.');

  // ---- Step 2: gradient = slope
  const Lb = (b) => CEN.L(CEN.best[0], b), dLb = (b) => CEN.g(CEN.best[0], b)[1];
  rb = rung(ch, 2, 'The gradient is the slope of the loss',
    `<p>Hold <i>a</i> at its best value and vary only the slope <i>b</i>. The MSE traces a parabola. Its <b>derivative</b> dL/db is the slope of that curve at the current point: positive means the loss rises as b increases, so b should decrease, and vice versa. The update <b>b ← b − η · dL/db</b> therefore always moves downhill, and moves further when the slope is steep.</p>`);
  const fB = figure(rb, 'One parameter: slope of the loss and the step it implies', 'The green segment is the tangent at the current b. The arrow shows the next update with learning rate η = 0.1. Hover the curve to read the loss.');
  let bB = -1;
  const sBb = slider(fB.controls, { label: 'b (slope of the line)', min: -2, max: 8, step: 0.05, value: bB, fmt: f2, onInput: (v) => { bB = v; paintB(); } });
  button(fB.controls, 'Take one step', () => { bB = Math.max(-2, Math.min(8, bB - 0.1 * dLb(bB))); sBb.set(bB); paintB(); }, true);
  const sB = stats(fB.root);
  const FB = frame(fB.body, { width: 820, height: 280, x: [-2, 8], y: [0, Lb(-2) * 1.05], xLabel: 'b (change in CD8 % per unit IFN-γ)', yLabel: 'MSE' });
  const gb = d3.range(-2, 8.001, 0.05);
  FB.layer('c').append('path').datum(gb).attr('fill', 'none').attr('stroke', cat(0)).attr('stroke-width', 2.2).attr('d', d3.line().x((b) => FB.x(b)).y((b) => FB.y(Lb(b))));
  FB.layer('c').append('rect').attr('width', FB.w).attr('height', FB.h).attr('fill', 'transparent')
    .on('mousemove', (e) => { const b = FB.x.invert(d3.pointer(e)[0]); tip(e, `b = ${f2(b)}<br>MSE = ${f1(Lb(b))}<br>slope dL/db = ${f1(dLb(b))}`); }).on('mouseleave', untip);
  FB.layer('min').append('path').attr('d', d3.symbol(d3.symbolCross, 90)).attr('transform', `translate(${FB.x(CEN.best[1])},${FB.y(CEN.Lmin)})`).attr('fill', ink());
  const tanL = FB.layer('tan').style('pointer-events', 'none');
  legend(fB.body, [['MSE as a function of b', cat(0)], ['tangent (slope)', cat(2)], ['next step', ink()]]);
  function paintB() {
    const L0 = Lb(bB), s = dLb(bB), hw = 0.9, b2 = Math.max(-2, Math.min(8, bB - 0.1 * s));
    tanL.selectAll('line.t').data([0]).join('line').attr('class', 't').attr('x1', FB.x(bB - hw)).attr('x2', FB.x(bB + hw)).attr('y1', FB.y(L0 - s * hw)).attr('y2', FB.y(L0 + s * hw)).attr('stroke', cat(2)).attr('stroke-width', 2.5);
    tanL.selectAll('path.ar').data([0]).join('path').attr('class', 'ar').attr('fill', 'none').attr('stroke', ink()).attr('stroke-width', 1.8)
      .attr('d', () => { const x0 = FB.x(bB), x1 = FB.x(b2), yy = FB.y(L0) - 14, d = Math.sign(x1 - x0) * 7; return Math.abs(x1 - x0) < 3 ? '' : `M${x0},${yy} L${x1},${yy} M${x1 - d},${yy - 5} L${x1},${yy} L${x1 - d},${yy + 5}`; });
    tanL.selectAll('circle').data([0]).join('circle').attr('cx', FB.x(bB)).attr('cy', FB.y(L0)).attr('r', 6).attr('fill', cat(2)).attr('stroke', css('--surface')).attr('stroke-width', 1.5);
    sB.set({ b: f2(bB), MSE: f1(L0), 'slope dL/db': f1(s), 'next b = b − 0.1 × slope': f2(b2), 'distance to best b': f2(Math.abs(bB - CEN.best[1])) });
  }
  paintB();
  const bW = 1, h = 1e-4, gW = dLb(bW), gFD = (Lb(bW + h) - Lb(bW - h)) / (2 * h);
  worked(rb, 'Computing the gradient',
    `<p>For MSE the derivative has a simple form: dL/db = −(2/n) Σ (yᵢ − ŷᵢ) · x̃ᵢ, where x̃ = IFN-γ − ${f2(xbar)}. Each patient pulls on the slope in proportion to its residual times its distance from the mean IFN-γ. At b = ${bW}:</p>
     <div class="calc">analytic   dL/db = ${f3(gW)}
numerical  [L(b + 0.0001) − L(b − 0.0001)] / 0.0002 = ${f3(gFD)}</div>
     <p>With several parameters the <b>gradient</b> is the vector of these partial derivatives, one per parameter (∂L/∂a, ∂L/∂b). Software computes it automatically (automatic differentiation; backpropagation in neural networks); comparing with the numerical version is the standard check of a hand-written gradient.</p>`);

  // ---- Step 3: learning rate on the real 2-D surface
  const lamR = RAW.lam, lamC = CEN.lam, limR = 2 / lamR[0], limC = 2 / lamC[0];
  rb = rung(ch, 3, 'The learning rate: too small, too large, and the shape of the bowl',
    `<p>With both parameters free, the loss is a bowl over the (a, b) plane. Each update moves perpendicular to the contour lines. The learning rate η sets how far: too small and progress is slow; too large and each step overshoots the valley floor, so the iterates oscillate and then diverge.</p>
     <p>How large η can be depends on the curvature of the bowl. For MSE the bound is <b>η &lt; 2 / λ<sub>max</sub></b>, where λ<sub>max</sub> is the largest eigenvalue of the matrix of second derivatives (the steepest curvature). The select below changes only how the same line is written: with IFN-γ as measured, <i>a</i> is the CD8 % at IFN-γ = 0, far outside the data, and the intercept and slope become strongly coupled.</p>`);
  const fC = figure(rb, 'Gradient descent on the CD8 ~ IFN-γ loss surface',
    'Left: contours of MSE over (a, b); darker is lower, the cross marks the minimum. Each dot is one update. Click the map to start from another point. Right: distance above the minimum MSE, on a log scale.');
  let par = 'cen', S = CEN, lrC = 0.1, pathC = [], stopC = null;
  const DOM = { cen: { A: [-2, 20], B: [-2, 8], x: 'a (CD8 % at mean IFN-γ)', start: [0, 0], lr: 0.1 }, raw: { A: [-32, 10], B: [-1, 8], x: 'a (CD8 % at IFN-γ = 0)', start: [0, 0], lr: 0.02 } };
  select(fC.controls, { label: 'IFN-γ', value: par, options: [['cen', 'centred at its mean'], ['raw', 'as measured (uncentred)']], onChange: (v) => { par = v; S = v === 'raw' ? RAW : CEN; lrC = DOM[v].lr; sLr.set(Math.log10(lrC)); buildC(); resetC(DOM[par].start); } });
  const sLr = slider(fC.controls, { label: 'learning rate η', min: -3.5, max: 0, step: 0.05, value: Math.log10(lrC), fmt: (v) => fLr(10 ** v), onInput: (v) => { lrC = 10 ** v; resetC(pathC[0]); } });
  button(fC.controls, 'Run', () => runC(), true);
  button(fC.controls, 'Step', () => { haltC(); stepC(); paintC(); });
  button(fC.controls, 'Reset', () => { haltC(); resetC(DOM[par].start); });
  const sC = stats(fC.root);
  const rowC = el('div', { class: 'fig-row' }); fC.body.append(rowC);
  const cL = el('div'), cR = el('div'); rowC.append(cL, cR);
  let MC, LC;
  legend(fC.body, [['path of updates', ink()], ['current point', cat(2)], ['minimum', ink2()], ['loss above minimum', cat(0)]]);
  function buildC() {
    cL.innerHTML = ''; cR.innerHTML = '';
    MC = lossMap(cL, S, { A: DOM[par].A, B: DOM[par].B, xLabel: DOM[par].x, yLabel: 'b (slope)', onClick: (p) => { haltC(); resetC(p); } });
    LC = frame(cR, { width: 420, height: 400, x: [0, 50], y: [1e-3, 1e3], xLabel: 'update', yLabel: 'MSE − minimum (log scale)', yScale: d3.scaleLog() });
  }
  function resetC(p) { haltC(); pathC = [p.slice()]; paintC(); }
  function haltC() { if (stopC) { stopC(); stopC = null; } }
  function stepC() {
    const w = pathC[pathC.length - 1], g = S.g(...w), nw = [w[0] - lrC * g[0], w[1] - lrC * g[1]];
    if (!isFinite(S.L(...nw)) || S.L(...nw) > 1e7 || pathC.length > 1500) return false;
    pathC.push(nw); return Math.hypot(...g) > 1e-3;
  }
  function runC() { haltC(); stopC = animate(() => { let go = true; for (let k = 0; k < 4 && go; k++) go = stepC(); paintC(); return go; }); }
  function paintC() {
    const { F } = MC, pts = pathC;
    MC.trail('path').selectAll('path').data([pts]).join('path').attr('fill', 'none').attr('stroke', ink()).attr('stroke-width', 1.4).attr('d', d3.line().x((q) => F.x(q[0])).y((q) => F.y(q[1])));
    MC.trail('dots').selectAll('circle').data(pts).join('circle').attr('cx', (q) => F.x(q[0])).attr('cy', (q) => F.y(q[1])).attr('r', (q, i) => (i === pts.length - 1 ? 6.5 : 2.6))
      .attr('fill', (q, i) => (i === pts.length - 1 ? cat(2) : ink())).attr('stroke', css('--surface')).attr('stroke-width', 1);
    const gap = pts.map((q, i) => [i, Math.max(1e-3, S.L(...q) - S.Lmin)]);
    LC.x.domain([0, Math.max(50, pts.length - 1)]); LC.y.domain([1e-3, Math.max(1e3, d3.max(gap, (q) => q[1]))]);
    LC.g.select('.axis.x').call(d3.axisBottom(LC.x).ticks(5)); LC.g.select('.axis.y').call(d3.axisLeft(LC.y).ticks(5, '~g'));
    LC.layer('c').selectAll('path').data([gap]).join('path').attr('fill', 'none').attr('stroke', cat(0)).attr('stroke-width', 2).attr('d', d3.line().x((q) => LC.x(q[0])).y((q) => LC.y(q[1])));
    const last = pts[pts.length - 1], g = S.g(...last), L = S.L(...last);
    const lim = 2 / S.lam[0];
    sC.set({ update: pts.length - 1, MSE: L > 1e5 ? L.toExponential(1) : f1(L), '|gradient|': Math.hypot(...g) > 1e4 ? Math.hypot(...g).toExponential(1) : f3(Math.hypot(...g)), 'stability limit 2/λmax': f3(lim), status: lrC >= lim ? 'η above limit: diverges' : Math.hypot(...g) < 1e-3 ? 'converged' : pts.length > 1500 ? 'stopped at 1,500 updates' : 'moving' });
  }
  buildC(); resetC(DOM.cen.start);
  for (let k = 0; k < 300 && stepC(); k++);
  paintC();
  const runPlain = (Sx, eta, k, w = [0, 0]) => { for (let i = 0; i < k; i++) { const g = Sx.g(...w); w = [w[0] - eta * g[0], w[1] - eta * g[1]]; } return w; };
  const wR300 = runPlain(RAW, 0.02, 300), wC30 = runPlain(CEN, 0.1, 30);
  prose(rb, `<p>Centred: λ<sub>max</sub> = ${f2(lamC[0])}, so η must stay below ${f3(limC)}; the curvatures in the two directions differ by a factor of ${f1(lamC[0] / lamC[1])} and 30 updates at η = 0.1 bring the MSE to within ${expo(CEN.L(...wC30) - CEN.Lmin)} of the minimum. Uncentred: λ<sub>max</sub> = ${f1(lamR[0])}, the limit drops to ${f3(limR)}, and the curvatures differ by a factor of <b>${Math.round(lamR[0] / lamR[1])}</b>. The bowl becomes a long, narrow valley; after 300 updates at η = 0.02 the slope is still ${f2(wR300[1])} instead of ${f2(RAW.best[1])}. The data and the best line are identical in both cases; only the conditioning of the problem has changed. This is why features are centred or standardised before gradient-based fitting.</p>`);
  quiz(rb, `With IFN-γ uncentred, why does η = 0.03 diverge although η = 0.02 converges?`, `The steepest direction has curvature λ<sub>max</sub> = ${f1(lamR[0])}. Along that direction each update multiplies the error by (1 − η·λ<sub>max</sub>): ${f2(1 - 0.02 * lamR[0])} for η = 0.02 (shrinks, with alternating sign) and ${f2(1 - 0.03 * lamR[0])} for η = 0.03 (grows in size at every step).`);

  // ---- Step 4: stochastic and mini-batch
  rb = rung(ch, 4, 'Stochastic and mini-batch gradient descent',
    `<p>The full gradient averages over every patient. <b>Stochastic gradient descent</b> (SGD) estimates it from one randomly chosen patient, and <b>mini-batch</b> GD from a small random batch (in deep learning, typically 32–512 samples). One <b>epoch</b> is one pass through all patients, so smaller batches mean more, cheaper, noisier updates per epoch. Each run below uses the same compute budget per epoch: ${n} per-patient gradients.</p>`);
  const BS = [1, 10, 30, n];
  const fD = figure(rb, 'Same data, different batch sizes', 'Left: the path of every update on the centred loss surface. Right: MSE above the minimum at the end of each epoch; the most recent run for each batch size is kept for comparison. Batches are drawn with a fixed random seed.');
  let bsD = 10, lrD = 0.03, decayD = false, runD = null, stopD = null;
  const histD = new Map();
  select(fD.controls, { label: 'batch size', value: String(bsD), options: BS.map((b) => [String(b), b === n ? `${n} (full batch)` : String(b)]), onChange: (v) => { bsD = +v; startD(); } });
  slider(fD.controls, { label: 'learning rate η', min: -2.5, max: -0.6, step: 0.05, value: Math.log10(lrD), fmt: (v) => fLr(10 ** v), onInput: (v) => { lrD = 10 ** v; } });
  checkbox(fD.controls, { label: 'shrink η each epoch (η / (1 + epoch/2))', checked: false, onChange: (c) => { decayD = c; } });
  button(fD.controls, 'Run 20 epochs', () => startD(), true);
  button(fD.controls, 'Clear', () => { haltD(); histD.clear(); runD = null; paintD(); });
  const sD = stats(fD.root);
  const rowD = el('div', { class: 'fig-row' }); fD.body.append(rowD);
  const dL = el('div'), dR = el('div'); rowD.append(dL, dR);
  const MD = lossMap(dL, CEN, { A: [-2, 20], B: [-2, 8], xLabel: 'a (CD8 % at mean IFN-γ)', yLabel: 'b (slope)' });
  const LD = frame(dR, { width: 420, height: 400, x: [0, 20], y: [1e-3, 1e3], xLabel: 'epoch', yLabel: 'MSE − minimum (log scale)', yScale: d3.scaleLog() });
  legend(fD.body, BS.map((b, i) => [b === n ? 'full batch' : `batch ${b}`, cat(i + 3)]));
  const colD = (b) => cat(BS.indexOf(b) + 3);
  function haltD() { if (stopD) { stopD(); stopD = null; } }
  function sgdRun(bs, eta, decay, epochs = 20) {
    const r = rng(17), xt = CEN.xt; let w = [0, 0]; const path = [w.slice()], ep = [Math.max(1e-3, CEN.L(...w) - CEN.Lmin)];
    for (let e = 0; e < epochs; e++) {
      const order = bs >= n ? d3.range(n) : r.shuffle(d3.range(n)), et = decay ? eta / (1 + e / 2) : eta;
      for (let s = 0; s < n; s += bs) {
        let ga = 0, gbb = 0; const idx = order.slice(s, s + bs);
        idx.forEach((i) => { const err = w[0] + w[1] * xt[i] - ys[i]; ga += 2 * err; gbb += 2 * err * xt[i]; });
        w = [w[0] - et * ga / idx.length, w[1] - et * gbb / idx.length];
        if (!isFinite(w[0]) || Math.abs(w[0]) > 1e6) return { path, ep, diverged: true, bs };
        path.push(w.slice());
      }
      ep.push(Math.max(1e-3, CEN.L(...w) - CEN.Lmin));
    }
    return { path, ep, diverged: false, bs };
  }
  function startD() {
    haltD();
    const full = sgdRun(bsD, lrD, decayD), perEp = Math.ceil(n / bsD);
    let k = 0, acc = 0;
    stopD = animate((dt) => {
      acc += dt; if (acc < 70) return true; acc = 0; k++;
      runD = { ...full, path: full.path.slice(0, 1 + Math.min(full.path.length - 1, k * perEp)), ep: full.ep.slice(0, k + 1) };
      histD.set(bsD, runD); paintD();
      return k < full.ep.length - 1;
    });
  }
  function paintD() {
    const { F } = MD, pts = runD ? runD.path : [];
    MD.trail('path').selectAll('path').data([pts]).join('path').attr('fill', 'none').attr('stroke', runD ? colD(runD.bs) : 'none').attr('stroke-width', 1.1).attr('stroke-opacity', 0.85).attr('d', d3.line().x((q) => F.x(q[0])).y((q) => F.y(q[1])));
    MD.trail('dot').selectAll('circle').data(pts.length ? [pts[pts.length - 1]] : []).join('circle').attr('cx', (q) => F.x(q[0])).attr('cy', (q) => F.y(q[1])).attr('r', 6.5).attr('fill', runD ? colD(runD.bs) : 'none').attr('stroke', css('--surface')).attr('stroke-width', 1.5);
    const runs = [...histD.values()];
    LD.y.domain([1e-3, Math.max(1e3, d3.max(runs, (r) => d3.max(r.ep)) || 1)]); LD.g.select('.axis.y').call(d3.axisLeft(LD.y).ticks(5, '~g'));
    LD.layer('c').selectAll('path').data(runs).join('path').attr('fill', 'none').attr('stroke', (r) => colD(r.bs)).attr('stroke-width', 2).attr('d', (r) => d3.line().x((q, i) => LD.x(i)).y((q) => LD.y(q))(r.ep));
    if (!runD) { sD.set({ status: 'press "Run 20 epochs"' }); return; }
    const ep = runD.ep.length - 1, w = runD.path[runD.path.length - 1];
    sD.set({ epoch: ep, updates: runD.path.length - 1, 'per-patient gradients': ep * n, MSE: runD.diverged ? 'diverged' : f2(CEN.L(...w)), 'above minimum': runD.diverged ? '—' : expo(CEN.L(...w) - CEN.Lmin) });
  }
  BS.forEach((b) => histD.set(b, sgdRun(b, lrD, decayD)));
  runD = histD.get(bsD);
  paintD();
  const fin = (r) => r.ep[r.ep.length - 1];
  const sg1 = sgdRun(1, 0.03, false), sg1d = sgdRun(1, 0.03, true), sgFull = sgdRun(n, 0.03, false), sg10 = sgdRun(10, 0.03, false);
  const ep1 = (r) => r.ep[1];
  prose(rb, `<p>At η = 0.03, after <b>one epoch</b> the MSE is ${expo(ep1(sgFull))} above the minimum with the full batch (1 update), ${expo(ep1(sg10))} with batches of 10 (${Math.ceil(n / 10)} updates) and ${expo(ep1(sg1))} with single patients (${n} updates). After 20 epochs the order changes: full batch ${expo(fin(sgFull))}, batch 10 ${expo(fin(sg10))}, batch 1 ${sg1.diverged ? 'diverged' : expo(fin(sg1))}. Small batches make fast early progress but never settle, because each update follows one noisy estimate of the gradient. Shrinking η over epochs removes most of this noise floor (batch 1 with shrinking η: ${expo(fin(sg1d))}). On a GPU, mini-batches of 32–512 balance the two effects.</p>`);

  // ---- Step 5: momentum and Adam
  rb = rung(ch, 5, 'Momentum and Adam',
    `<p>Plain gradient descent uses only the current slope, so on a narrow valley (the uncentred surface of step 3) it zigzags across the steep direction and advances slowly along the shallow one. Two refinements are standard:</p>
     <ul><li><b>Momentum</b> keeps a running sum of past gradients (v ← β·v + ∇L, θ ← θ − η·v, β ≈ 0.9). Consistent directions accumulate speed; oscillating ones cancel.</li>
     <li><b>Adam</b> also keeps a running average of squared gradients and divides each parameter's step by its square root, so every parameter moves at a similar rate regardless of the scale of its gradient. Adam (and AdamW, with weight decay) is the default optimiser for neural networks.</li></ul>`);
  const OPT = [{ k: 'gd', lab: 'gradient descent (η 0.02)', c: cat(0) }, { k: 'mom', lab: 'momentum (η 0.02, β 0.9)', c: cat(1) }, { k: 'adam', lab: 'Adam (η 0.5)', c: cat(2) }];
  const NE = 300;
  const trajE = OPT.map((o) => {
    let w = [0, 0]; const v = [0, 0], s2 = [0, 0], out = [w.slice()];
    for (let t = 1; t <= NE; t++) {
      const g = RAW.g(...w);
      for (let j = 0; j < 2; j++) {
        if (o.k === 'gd') w[j] -= 0.02 * g[j];
        else if (o.k === 'mom') { v[j] = 0.9 * v[j] + g[j]; w[j] -= 0.02 * v[j]; }
        else { v[j] = 0.9 * v[j] + 0.1 * g[j]; s2[j] = 0.999 * s2[j] + 0.001 * g[j] * g[j]; w[j] -= 0.5 * (v[j] / (1 - 0.9 ** t)) / (Math.sqrt(s2[j] / (1 - 0.999 ** t)) + 1e-8); }
      }
      out.push(w.slice());
    }
    return { ...o, out };
  });
  const fE = figure(rb, 'Three optimisers on the uncentred surface', 'All three start at a = b = 0 on the narrow valley from step 3. Drag the slider or press Play to advance the updates.');
  let tE = 40, stopE = null;
  const sTE = slider(fE.controls, { label: 'updates', min: 0, max: NE, step: 1, value: tE, fmt: String, onInput: (v) => { if (stopE) { stopE(); stopE = null; } tE = v; paintE(); } });
  button(fE.controls, 'Play', () => { if (stopE) stopE(); tE = 0; let acc = 0; stopE = animate((dt) => { acc += dt; if (acc < 30) return true; acc = 0; tE = Math.min(NE, tE + 2); sTE.set(tE); paintE(); return tE < NE; }); }, true);
  const sE = stats(fE.root);
  const rowE = el('div', { class: 'fig-row' }); fE.body.append(rowE);
  const eL = el('div'), eR = el('div'); rowE.append(eL, eR);
  const ME = lossMap(eL, RAW, { A: DOM.raw.A, B: DOM.raw.B, xLabel: 'a (CD8 % at IFN-γ = 0)', yLabel: 'b (slope)' });
  const LE = frame(eR, { width: 420, height: 400, x: [0, NE], y: [1e-3, 2e3], xLabel: 'update', yLabel: 'MSE − minimum (log scale)', yScale: d3.scaleLog() });
  LE.g.select('.axis.y').call(d3.axisLeft(LE.y).ticks(5, '~g'));
  const curE = LE.layer('cur');
  legend(fE.body, OPT.map((o) => [o.lab, o.c]));
  function paintE() {
    const { F } = ME;
    ME.trail('p').selectAll('path').data(trajE).join('path').attr('fill', 'none').attr('stroke', (o) => o.c).attr('stroke-width', 1.8).attr('d', (o) => d3.line().x((q) => F.x(q[0])).y((q) => F.y(q[1]))(o.out.slice(0, tE + 1)));
    ME.trail('d').selectAll('circle').data(trajE).join('circle').attr('cx', (o) => F.x(o.out[tE][0])).attr('cy', (o) => F.y(o.out[tE][1])).attr('r', 6).attr('fill', (o) => o.c).attr('stroke', css('--surface')).attr('stroke-width', 1.5);
    LE.layer('c').selectAll('path').data(trajE).join('path').attr('fill', 'none').attr('stroke', (o) => o.c).attr('stroke-width', 2).attr('d', (o) => d3.line().x((q, i) => LE.x(i)).y((q) => LE.y(Math.max(1e-3, Math.min(2e3, RAW.L(...q) - RAW.Lmin))))(o.out.slice(0, tE + 1)));
    curE.selectAll('line').data([tE]).join('line').attr('x1', LE.x(tE)).attr('x2', LE.x(tE)).attr('y1', 0).attr('y2', LE.h).attr('stroke', muted()).attr('stroke-dasharray', '3 3');
    sE.set(Object.fromEntries(trajE.map((o) => [`${o.k === 'gd' ? 'GD' : o.k === 'mom' ? 'momentum' : 'Adam'}: MSE above min`, expo(RAW.L(...o.out[tE]) - RAW.Lmin)])));
  }
  paintE();
  const endE = trajE.map((o) => RAW.L(...o.out[NE]) - RAW.Lmin);
  const hitE = trajE.map((o) => o.out.findIndex((q) => RAW.L(...q) - RAW.Lmin < 0.01));
  prose(rb, `<p>After ${NE} updates the MSE above the minimum is ${expo(endE[0])} for plain gradient descent, ${expo(endE[1])} for momentum and ${expo(endE[2])} for Adam. Momentum first comes within 0.01 of the minimum at update ${hitE[1]}, Adam at update ${hitE[2]}${hitE[0] < 0 ? ', while plain gradient descent does not get there within the run' : `, plain gradient descent at update ${hitE[0]}`}. Both methods overshoot and curve back before settling; that is expected behaviour, not an error. Centring the feature (step 3) removes most of the problem for plain gradient descent as well.</p>`);

  // ---- Step 6: lab
  rb = rung(ch, 6, 'Training lab: the response model by gradient descent',
    `<p>The same algorithm fits the logistic regression for response, with log-loss as the loss. Weights start at 0 and are trained on the ${n} discovery patients; after every epoch the log-loss is also computed on the ${V.length} validation patients from Centre C, who are never used for updates. The dashed lines show the exact minimum found by Newton's method (IRLS) on the same features.</p>`);
  const LAB = ['ifng', 'log2tmb', 'caf', 'pdl1', 'cd8', 'prolif', 'age', 'smoker', 'male', 'centreB'];
  const FSET = { three: ['ifng', 'log2tmb', 'caf'], all: LAB };
  const LRS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3];
  const prep = (fs, std) => {
    const X = D.map((p) => FSET[fs].map((k) => FEATURES[k].get(p))), Xv = V.map((p) => FSET[fs].map((k) => FEATURES[k].get(p)));
    if (!std) return { X, Xv };
    const st = standardize(X); return { X: st.Z, Xv: Xv.map(st.apply) };
  };
  const exactCache = {};
  const exact = (fs, std) => {
    const k = fs + std;
    if (!exactCache[k]) { const { X, Xv } = prep(fs, std), f = fitLogistic(X, y); const pv = Xv.map((r) => predictProb(r, f.b0, f.w)); exactCache[k] = { tr: logLoss(X.map((r) => predictProb(r, f.b0, f.w)), y), va: logLoss(pv, yv), auc: auc(pv, yv) }; }
    return exactCache[k];
  };
  const lab = (c) => { const { X, Xv } = prep(c.fs, c.std); return trainLogistic(X, y, Xv, yv, { opt: c.opt, lr: LRS[c.lri], batch: c.bs, epochs: c.ep }); };
  const cfg = { fs: 'three', std: true, opt: 'gd', lri: 5, bs: n, ep: 100 };
  const fL = figure(rb, 'Train the response model', 'Left: log-loss after each epoch on training (discovery) and validation (Centre C) patients; dashed lines mark the exact fit. Right: predicted probability for each validation patient after the last epoch. Hover for details.');
  const rowCtl = el('div', { class: 'controls' }); fL.controls.before(rowCtl);
  select(rowCtl, { label: 'features', value: cfg.fs, options: [['three', 'IFN-γ, log2 TMB, CAF'], ['all', 'all ten features']], onChange: (v) => { cfg.fs = v; paintL(); } });
  checkbox(rowCtl, { label: 'standardise features', checked: true, onChange: (c) => { cfg.std = c; paintL(); } });
  select(rowCtl, { label: 'optimiser', value: cfg.opt, options: [['gd', 'gradient descent'], ['mom', 'momentum (β 0.9)'], ['adam', 'Adam']], onChange: (v) => { cfg.opt = v; paintL(); } });
  select(rowCtl, { label: 'batch size', value: String(cfg.bs), options: [[String(n), `${n} (full)`], ['32', '32'], ['8', '8'], ['1', '1']], onChange: (v) => { cfg.bs = +v; paintL(); } });
  slider(fL.controls, { label: 'learning rate η', min: 0, max: LRS.length - 1, step: 1, value: cfg.lri, fmt: (i) => String(LRS[i]), onInput: (i) => { cfg.lri = i; paintL(); } });
  slider(fL.controls, { label: 'epochs', min: 1, max: 300, step: 1, value: cfg.ep, fmt: String, onInput: (v) => { cfg.ep = v; paintL(); } });
  const sL = stats(fL.root);
  const rowL = el('div', { class: 'fig-row' }); fL.body.append(rowL);
  const lA = el('div'), lB = el('div'); rowL.append(lA, lB);
  const FL = frame(lA, { width: 480, height: 320, x: [0, 100], y: [0.3, 0.8], xLabel: 'epoch', yLabel: 'log-loss' });
  const FS = frame(lB, { width: 480, height: 320, x: [0, 1], y: [-0.6, 1.6], xLabel: 'predicted P(response), validation patients', grid: false, axes: false });
  FS.g.append('g').attr('class', 'axis x').attr('transform', `translate(0,${FS.h})`).call(d3.axisBottom(FS.x).ticks(5));
  FS.layer('lab').selectAll('text').data([[0, 'no response'], [1, 'responded']]).join('text').attr('x', 4).attr('y', (d) => FS.y(d[0] + 0.42)).attr('font-size', 12).attr('fill', ink2()).text((d) => d[1]);
  const msgL = html('', 'p', 'fig-caption'); fL.body.append(msgL);
  legend(fL.body, [['training log-loss', cat(0)], ['validation log-loss', cat(1)], ['exact fit (dashed)', ink2()]]);
  const jit = (i) => (((i * 7919) % 101) / 101 - 0.5) * 0.7;
  function paintL() {
    const R = lab(cfg), E = exact(cfg.fs, cfg.std);
    const curves = [{ v: R.tr, c: cat(0), k: 'training' }, { v: R.va, c: cat(1), k: 'validation' }];
    const ok = (q) => isFinite(q);
    const hiY = Math.min(1.5, Math.max(0.75, d3.max([...R.tr, ...R.va].filter(ok)) || 0.75));
    FL.x.domain([0, cfg.ep]); FL.y.domain([Math.min(0.3, E.tr - 0.03), hiY]);
    FL.g.select('.axis.x').call(d3.axisBottom(FL.x).ticks(5)); FL.g.select('.axis.y').call(d3.axisLeft(FL.y).ticks(5));
    FL.g.select('.gridline').call(d3.axisLeft(FL.y).ticks(5).tickSize(-FL.w).tickFormat(''));
    FL.layer('ex').selectAll('line').data([[E.tr, cat(0)], [E.va, cat(1)]]).join('line').attr('x1', 0).attr('x2', FL.w).attr('y1', (d) => FL.y(d[0])).attr('y2', (d) => FL.y(d[0])).attr('stroke', (d) => d[1]).attr('stroke-dasharray', '5 4').attr('stroke-width', 1.4);
    FL.layer('c').selectAll('path').data(curves).join('path').attr('fill', 'none').attr('stroke', (d) => d.c).attr('stroke-width', 2.2)
      .attr('d', (d) => d3.line().defined((q) => ok(q)).x((q, i) => FL.x(i)).y((q) => FL.y(Math.min(hiY, q)))(d.v));
    FL.layer('hov').selectAll('rect').data([0]).join('rect').attr('width', FL.w).attr('height', FL.h).attr('fill', 'transparent')
      .on('mousemove', (e) => { const i = Math.max(0, Math.min(cfg.ep, Math.round(FL.x.invert(d3.pointer(e)[0])))); tip(e, `epoch ${i}<br>training ${ok(R.tr[i]) ? f3(R.tr[i]) : 'diverged'}<br>validation ${ok(R.va[i]) ? f3(R.va[i]) : 'diverged'}`); }).on('mouseleave', untip);
    const st = FS.layer('strip');
    st.selectAll('circle').data(R.pv ? V : []).join('circle').attr('cx', (d, i) => FS.x(R.pv[i])).attr('cy', (d, i) => FS.y(d.response + jit(i))).attr('r', 4)
      .attr('fill', (d) => cat(d.response)).attr('fill-opacity', 0.8).attr('stroke', css('--surface')).attr('stroke-width', 0.8)
      .on('mousemove', (e, d) => { const i = V.indexOf(d); tip(e, `${d.id} (${d.phenotype})<br>predicted ${f2(R.pv[i])} · ${d.response ? 'responded' : 'no response'}`); }).on('mouseleave', untip);
    const lastT = R.tr[R.tr.length - 1];
    const status = R.diverged ? 'diverged' : lastT - E.tr < 0.002 ? 'at the exact minimum' : 'not yet converged';
    sL.set({ updates: R.updates, 'training log-loss': R.diverged ? '—' : f3(lastT), 'exact minimum': f3(E.tr), 'validation log-loss': R.diverged ? '—' : f3(R.va[R.va.length - 1]), 'validation AUC': R.diverged ? '—' : f3(R.aucV), status });
    msgL.innerHTML = R.diverged ? 'The weights grew without bound (diverged). Reduce the learning rate or standardise the features.' : '';
  }
  paintL();
  // Try / see / means: each row is a fixed run of the same function as the lab.
  const run = (o) => { const R = lab({ ...cfg, ...o }); return { ...R, last: R.tr[R.tr.length - 1], lastV: R.va[R.va.length - 1] }; };
  const E3 = exact('three', true), Eall = exact('all', true);
  const rFast = run({ lri: 5 }), rSlow = run({ lri: 2 }), rBig = run({ lri: 7 });
  const rRawGD = run({ fs: 'all', std: false, lri: 3 }), rRawGD2 = run({ fs: 'all', std: false, lri: 0 }), rRawAd = run({ fs: 'all', std: false, opt: 'adam', lri: 3 });
  const rB8 = run({ bs: 8, lri: 3, ep: 5 }), rFull5 = run({ lri: 3, ep: 5 });
  const rAll = run({ fs: 'all', ep: 300 }), vMin = d3.minIndex(rAll.va);
  const st = (R) => (R.diverged ? 'diverged' : `${f3(R.last)}`);
  table(rb, [
    { t: 'η <b>0.3</b> → <b>0.01</b> → <b>3</b> (100 epochs, standardised)', s: `Training log-loss ${st(rFast)} → ${st(rSlow)} → ${st(rBig)}; exact minimum ${f3(E3.tr)}`, m: 'Too small a step stops short of the minimum within the budget; on this convex loss a large step still converges once the features are standardised.' },
    { t: 'All ten features, <b>unstandardised</b>: GD η 0.03 → η 0.001 → <b>Adam</b> η 0.03', s: `${st(rRawGD)} → ${st(rRawGD2)} → ${st(rRawAd)}; exact minimum ${f3(exact('all', false).tr)}`, m: `Raw features (PD-L1 up to 100%, age in years) make the surface badly conditioned. Plain GD needs a tiny η and then converges slowly; Adam rescales each weight's step${rRawAd.last - exact('all', false).tr > 0.01 ? ', which helps but does not fully solve it in 100 epochs' : ''}.` },
    { t: 'Batch <b>8</b> vs <b>full</b>, η 0.03, only 5 epochs', s: `Training log-loss ${f3(rB8.last)} (${rB8.updates} updates) vs ${f3(rFull5.last)} (${rFull5.updates} updates)`, m: 'For the same number of passes through the data, mini-batches make many more updates and get further.' },
    { t: '<b>All ten</b> features, 300 epochs', s: `Training ${f3(rAll.last)} vs ${f3(rFast.last)} with three features; validation ${f3(rAll.lastV)} vs ${f3(rFast.lastV)}; lowest validation loss at epoch ${vMin} (${f3(rAll.va[vMin])})`, m: `A lower training loss does not imply a lower validation loss. Stopping when validation loss stops improving (<b>early stopping</b>) acts as a form of regularisation${vMin >= 290 ? '; here the validation curve is still falling at the end, so it would not stop early' : ''}.` },
    { t: 'Compare with the exact fit', s: `Validation AUC ${f3(rFast.aucV)} (GD) vs ${f3(E3.auc)} (IRLS)`, m: 'Once converged, gradient descent and Newton\'s method give the same model; the optimiser affects speed and stability, not the answer, for a convex loss.' },
  ], [['t', 'Try'], ['s', 'What you see'], ['m', 'What it means']], { cls: 'compact' });

  // =====================================================================================
  // 5. PITFALLS
  // =====================================================================================
  ch = chapter(main, 'pitfalls', 'Pitfalls', 'Common problems, how to recognise them, and how to address them.');
  const wR1k = runPlain(RAW, 0.02, 1000);
  cards(ch, [
    { tag: 'Symptom: loss becomes NaN or explodes', title: 'Learning rate too large', body: `Each step overshoots and the error grows. For the uncentred CD8 model any η above ${f3(limR)} diverges. <b>Fix:</b> lower η (by factors of 3–10), standardise features, use gradient clipping, or a learning-rate warm-up for large networks.` },
    { tag: 'Symptom: slow, zigzagging progress', title: 'Unscaled or uncentred features', body: `Uncentred IFN-γ makes the curvature ratio ${Math.round(lamR[0] / lamR[1])} instead of ${f1(lamC[0] / lamC[1])}. Features on very different scales (PD-L1 %, age, log expression) do the same. <b>Fix:</b> centre and standardise inputs; Adam reduces but does not remove the problem.` },
    { tag: 'Symptom: weights differ from glm/statsmodels', title: 'Stopped before convergence', body: `After 300 updates at η = 0.02 on the uncentred surface the slope is ${f2(wR300[1])}; after 1,000 it is ${f2(wR1k[1])}; the exact value is ${f2(RAW.best[1])}. The loss curve had flattened visually long before. <b>Fix:</b> monitor the gradient norm and the change in loss, and compare with an exact solver when one exists.` },
    { tag: 'Symptom: loss jumps around and never settles', title: 'Noise of small batches', body: `With batch size 1 and a constant η, the MSE after 20 epochs is ${sg1.diverged ? 'divergent' : expo(fin(sg1)) + ' above the minimum'}, against ${expo(fin(sg1d))} with a shrinking η. <b>Fix:</b> decay the learning rate, enlarge the batch, or average the last iterates.` },
    { tag: 'Symptom: training loss keeps falling, validation does not', title: 'Optimising the wrong target', body: 'Gradient descent minimises the training loss and nothing else. A lower training loss with more features or more epochs can leave validation performance unchanged or worse. <b>Fix:</b> track a validation loss; use early stopping and regularisation.' },
    { tag: 'Symptom: different runs, different answers', title: 'Non-convex losses', body: 'Linear and logistic regression have a single bowl-shaped minimum, so every start gives the same answer. Neural networks, t-SNE and UMAP have many minima, and the result depends on the random seed and initialisation. <b>Fix:</b> fix and report seeds, repeat runs, and report the variation across them.' },
    { tag: 'Symptom: a few samples dominate', title: 'Loss choice and outliers', body: `Squared error gives large residuals most of the influence: the worst 5% of patients supply ${pc(topShare)} of the CD8 MSE. <b>Fix:</b> transform skewed outcomes (e.g. log CD8 %), or use an absolute or Huber loss that grows linearly for large errors.` },
    { tag: 'Symptom: excellent loss, poor decisions', title: 'Loss is not the evaluation metric', body: 'Log-loss and MSE are chosen because they are smooth and have useful gradients. The quantity of interest (AUC, sensitivity at a threshold, calibration) must be checked separately on held-out data.' },
  ]);

  // =====================================================================================
  // 6. RECAP
  // =====================================================================================
  ch = chapter(main, 'recap', 'Recap', 'Key points.');
  recap(ch, [
    'A <b>loss function</b> turns "how wrong" into one number: MSE for measurements, log-loss for yes/no outcomes. Fitting means minimising it.',
    'The <b>gradient</b> is the vector of slopes of the loss with respect to each parameter. It points uphill, so gradient descent steps the other way: <b>θ ← θ − η∇L</b>.',
    'The <b>learning rate</b> η trades speed against stability. Too small is slow; above 2/λ<sub>max</sub> (on a quadratic loss) the iterates diverge.',
    'The <b>shape</b> of the loss surface sets the speed. Centring and standardising features turns a narrow valley into a round bowl.',
    '<b>Stochastic and mini-batch</b> gradient descent use a random subset of samples per step: cheaper and faster early, noisier near the end, so the learning rate is usually decayed.',
    '<b>Momentum</b> accumulates consistent directions; <b>Adam</b> also rescales each parameter\'s step. Adam/AdamW is the default for neural networks.',
    'For convex models (linear, logistic) every correct optimiser reaches the same answer; for neural networks the result depends on initialisation and seed.',
    'The optimiser only minimises training loss. Generalisation is checked on held-out data, and early stopping uses that check to decide when to stop.',
  ]);
  prose(ch, '<h3>In code</h3><p>For linear and logistic regression, use the exact solvers (<code>lm</code>, <code>glm</code>, statsmodels). Write a gradient-descent loop to understand the mechanism, and use SGD-based fitters or a deep-learning framework when the data or model are large.</p>');
  code(ch, {
    Python: `import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score, log_loss

# 1) Gradient descent by hand: CD8 % on centred IFN-gamma, MSE loss
x = train["ifng"].to_numpy();  y = train["cd8"].to_numpy()
xc = x - x.mean();  a, b, eta = 0.0, 0.0, 0.1
for step in range(200):
    resid = y - (a + b * xc)
    grad_a, grad_b = -2 * resid.mean(), -2 * (resid * xc).mean()
    a, b = a - eta * grad_a, b - eta * grad_b
print(a, b, np.polyfit(xc, y, 1))          # compare with the exact fit

# 2) Logistic regression by mini-batch SGD (log-loss), standardised features
feats = ["ifng", "log2tmb", "caf"]
clf = make_pipeline(StandardScaler(),
                    SGDClassifier(loss="log_loss", penalty=None, learning_rate="invscaling",
                                  eta0=0.1, max_iter=1000, tol=1e-6, random_state=0))
clf.fit(train[feats], train["response"])
p = clf.predict_proba(test[feats])[:, 1]
print("validation log-loss", log_loss(test["response"], p), "AUC", roc_auc_score(test["response"], p))

# 3) Neural networks (PyTorch): torch.optim.Adam(model.parameters(), lr=1e-3)`,
    R: `# 1) Gradient descent by hand: CD8 % on centred IFN-gamma, MSE loss
xc <- train$ifng - mean(train$ifng); y <- train$cd8
a <- 0; b <- 0; eta <- 0.1
for (step in 1:200) {
  resid <- y - (a + b * xc)
  a <- a - eta * (-2 * mean(resid))
  b <- b - eta * (-2 * mean(resid * xc))
}
c(a, b); coef(lm(y ~ xc))                   # compare with the exact fit

# 2) Log-loss minimised by a general-purpose optimiser (BFGS), versus glm (IRLS)
X <- cbind(1, scale(train[, c("ifng", "log2tmb", "caf")])); yr <- train$response
nll  <- function(w) { p <- plogis(X %*% w); -mean(yr * log(p) + (1 - yr) * log(1 - p)) }
grad <- function(w) { p <- plogis(X %*% w); drop(t(X) %*% (p - yr)) / nrow(X) }
optim(rep(0, ncol(X)), nll, grad, method = "BFGS")$par
coef(glm(yr ~ X - 1, family = binomial))`,
  });
  seeAlso(main, [['concepts', 'linear-regression', 'Linear regression'], ['concepts', 'logistic-regression', 'Logistic regression'], ['concepts', 'regularization', 'Regularisation'], ['concepts', 'train-test', 'Train/test split and overfitting'], ['concepts', 'neural-network', 'Neural networks and backpropagation']]);
}

// ---------------------------------------------------------------------------------------------
/** "One iteration" loop diagram with real numbers from the first step on the centred CD8 model. */
function loopSvg(S, eta) {
  const wrap = el('div', { class: 'figure' });
  wrap.append(el('div', { class: 'fig-title' }, 'One iteration of gradient descent, on the study data'));
  const svg = d3.select(wrap).append('svg').attr('viewBox', '0 0 980 250').attr('width', 980).attr('role', 'img').attr('aria-label', 'guess, loss, gradient, update loop');
  const g0 = S.g(0, 0), a1 = -eta * g0[0], b1 = -eta * g0[1];
  const boxes = [
    { t: '① Current guess', lines: ['a = 0.00', 'b = 0.00', '', 'predicts CD8 = 0%', 'for every patient'] },
    { t: '② Loss: how wrong?', lines: ['MSE = mean of', '  (measured − predicted)²', `  = ${f1(S.L(0, 0))}`] },
    { t: '③ Gradient: which way?', lines: [`∂L/∂a = ${f2(g0[0])}`, `∂L/∂b = ${f2(g0[1])}`, '', 'negative: raising a and b', 'lowers the loss'] },
    { t: '④ Step downhill', lines: [`a ← 0 − ${eta} × (${f2(g0[0])})`, `  = ${f2(a1)}`, `b ← 0 − ${eta} × (${f2(g0[1])})`, `  = ${f2(b1)}`, `new MSE = ${f1(S.L(a1, b1))}`] },
  ];
  const BW = 212, gap = 36, H = 180;
  boxes.forEach((b, i) => {
    const x0 = 10 + i * (BW + gap), g = svg.append('g').attr('transform', `translate(${x0},10)`);
    g.append('rect').attr('width', BW).attr('height', H).attr('rx', 10).attr('fill', css('--surface-2')).attr('stroke', css('--axis'));
    g.append('text').attr('x', 12).attr('y', 24).attr('font-weight', 700).attr('font-size', 13.5).attr('fill', ink()).text(b.t);
    b.lines.forEach((l, j) => g.append('text').attr('x', 14).attr('y', 52 + j * 23).attr('font-family', css('--mono')).attr('font-size', 12.5).attr('fill', ink2()).attr('xml:space', 'preserve').text(l));
    if (i < boxes.length - 1) {
      const ax = x0 + BW + 6, ay = 10 + H / 2;
      svg.append('path').attr('d', `M${ax},${ay} L${ax + gap - 14},${ay} M${ax + gap - 20},${ay - 6} L${ax + gap - 14},${ay} L${ax + gap - 20},${ay + 6}`).attr('fill', 'none').attr('stroke', ink2()).attr('stroke-width', 2);
    }
  });
  const xr = 10 + 3 * (BW + gap) + BW / 2, xl = 10 + BW / 2;
  svg.append('path').attr('d', `M${xr},${10 + H} L${xr},${10 + H + 30} L${xl},${10 + H + 30} L${xl},${10 + H + 6} M${xl - 6},${10 + H + 12} L${xl},${10 + H + 6} L${xl + 6},${10 + H + 12}`).attr('fill', 'none').attr('stroke', ink2()).attr('stroke-width', 2);
  svg.append('text').attr('x', (xl + xr) / 2).attr('y', 10 + H + 50).attr('text-anchor', 'middle').attr('font-size', 12.5).attr('fill', ink2()).text('repeat with the new guess until the loss stops falling');
  return wrap;
}
