// Gradient boosting on 1-D regression: each stage fits a small tree to the residuals y − F(x), then adds
// a shrunken copy (learning rate ν) to the model. Animated stage by stage, with train/test error curves.
import { header, figure, slider, select, button, stats, section, steps, callout, math, legend, seeAlso, animate, tip, untip, el } from '../lib/ui.js';
import { frame, cat, css, ink, ink2, muted } from '../lib/plot.js';
import { regression1d, linspace } from '../lib/data.js';
const d3 = window.d3;

const M = 200;                       // stages computed per setting
const NOISE = 0.3;
const TRUTH = (x) => Math.sin(3 * x) + 0.4 * x;
const GRID = linspace(-1, 1, 241);

// Regression tree on 1-D x (sorted ascending) by exhaustive SSE splits. Returns {segs:[[lo,hi,value]], f(x)}.
function fitTree(xs, r, depth, minLeaf = 3) {
  const segs = [];
  (function grow(lo, hi, d, xlo, xhi) {          // indices [lo, hi)
    const n = hi - lo; let s = 0; for (let i = lo; i < hi; i++) s += r[i];
    const mean = s / n;
    if (d === 0 || n < 2 * minLeaf) { segs.push([xlo, xhi, mean]); return; }
    let best = null, sl = 0, sq = 0, tot2 = 0;
    for (let i = lo; i < hi; i++) tot2 += r[i] * r[i];
    for (let i = lo; i < hi - 1; i++) {
      sl += r[i]; sq += r[i] * r[i];
      const nl = i - lo + 1, nr = n - nl;
      if (nl < minLeaf || nr < minLeaf || xs[i] === xs[i + 1]) continue;
      const sr = s - sl, sse = (sq - sl * sl / nl) + ((tot2 - sq) - sr * sr / nr);
      if (!best || sse < best.sse) best = { sse, i };
    }
    if (!best) { segs.push([xlo, xhi, mean]); return; }
    const thr = (xs[best.i] + xs[best.i + 1]) / 2;
    grow(lo, best.i + 1, d - 1, xlo, thr); grow(best.i + 1, hi, d - 1, thr, xhi);
  })(0, xs.length, depth, -Infinity, Infinity);
  const f = (x) => { for (const [a, b, v] of segs) if (x <= b && x > a) return v; return segs[segs.length - 1][2]; };
  return { segs, f };
}

// Run boosting for M stages. stage m (0..M): model F_m; tree[m] (1..M) is the tree added at stage m.
function boost(train, test, nu, depth) {
  const xs = train.map((p) => p.x), ys = train.map((p) => p.y);
  const f0 = d3.mean(ys);
  let Ftr = ys.map(() => f0), Fte = test.map(() => f0), Fg = GRID.map(() => f0);
  const mse = (F, data) => d3.mean(data, (p, i) => (p.y - F[i]) ** 2);
  const stages = [{ Ftr, Fg, trainMSE: mse(Ftr, train), testMSE: mse(Fte, test) }];
  for (let m = 1; m <= M; m++) {
    const r = ys.map((y, i) => y - Ftr[i]);
    const tree = fitTree(xs, r, depth);
    Ftr = Ftr.map((v, i) => v + nu * tree.f(xs[i]));
    Fte = Fte.map((v, i) => v + nu * tree.f(test[i].x));
    Fg = Fg.map((v, i) => v + nu * tree.f(GRID[i]));
    stages[m - 1].next = { r, tree };          // residuals of F_{m-1} and the tree fitted to them
    stages.push({ Ftr, Fg, trainMSE: mse(Ftr, train), testMSE: mse(Fte, test) });
  }
  const best = stages.reduce((b, s, m) => (s.testMSE < stages[b].testMSE ? m : b), 0);
  return { stages, best, f0 };
}

export default async function render(main) {
  header(main, 'Gradient boosting',
    'Start with a crude guess, look at what it gets wrong, and fit a small tree to <b>just those errors (the residuals)</b>. Add a shrunken copy of that tree to the model, and repeat. Each tree is weak on its own; hundreds of them, each correcting the last, add up to a flexible and very accurate model.');

  const train = regression1d(50, TRUTH, NOISE, 11), test = regression1d(400, TRUTH, NOISE, 12);
  let nu = 0.3, depth = 1, m = 0, run = boost(train, test, nu, depth), stopAnim = null, stopTween = null, tween = 1;

  const fig = figure(main, 'One stage at a time: residuals → tree → updated fit',
    'Left: 50 training points and the current model F (thick line); grey stems are the <b>residuals</b>, the part of each y the model has not explained yet. Right: the same residuals plotted on their own, the tree fitted to them (dashed), and the shrunken copy ν × tree that actually gets added (solid). Press <b>Step</b> to watch the add happen; the stems shrink where the tree helped. Bottom: training and test error after each stage.');
  slider(fig.controls, { label: 'learning rate ν', min: 0.05, max: 1, step: 0.05, value: nu, fmt: (v) => v.toFixed(2), onInput: (v) => { nu = v; recompute(); } });
  select(fig.controls, { label: 'tree depth', value: '1', options: [['1', '1 (stump: 2 leaves)'], ['2', '2 (4 leaves)'], ['3', '3 (8 leaves)']], onChange: (v) => { depth = +v; recompute(); } });
  const mS = slider(fig.controls, { label: 'stage m (trees added)', min: 0, max: M, step: 1, value: 0, onInput: (v) => { halt(); m = v; tween = 1; paint(); } });
  button(fig.controls, 'Step', () => { halt(); step(false); }, true);
  const playB = button(fig.controls, 'Play', () => { if (stopAnim) halt(); else play(); });
  button(fig.controls, 'Reset', () => { halt(); m = 0; mS.set(0); tween = 1; paint(); });
  const readout = stats(fig.root);

  const row = el('div', { class: 'fig-row' }); fig.body.append(row);
  const left = el('div'), right = el('div'); row.append(left, right);
  const YD = [-2.1, 2.1];
  const A = frame(left, { width: 520, height: 360, x: [-1, 1], y: YD, xLabel: 'x', yLabel: 'y' });
  const B = frame(right, { width: 520, height: 360, x: [-1, 1], y: [-1.6, 1.6], xLabel: 'x', yLabel: 'residual  y − F(x)' });
  legend(fig.body, [['training points', cat(0)], ['true function', muted()], ['current model F', ink()], ['tree fitted to residuals', cat(1)], ['training error', cat(2)], ['test error (400 new points)', cat(3)]]);
  const errBox = el('div'); fig.body.append(errBox);
  const E = frame(errBox, { width: 1060, height: 250, x: [0, M], y: [0.01, 1], yScale: d3.scaleLog(), xScale: d3.scaleSymlog().constant(10), xLabel: 'stage m (number of trees, stretched near 0)', yLabel: 'mean squared error (log scale)', margin: { left: 56 } });

  E.g.select('.axis.x').call(d3.axisBottom(E.x).tickValues([0, 1, 2, 5, 10, 20, 50, 100, 200]).tickFormat(d3.format('d')));
  const aTitle = A.g.append('text').attr('x', 6).attr('y', 14).attr('font-size', 13).attr('fill', ink2());
  const bTitle = B.g.append('text').attr('x', 6).attr('y', 14).attr('font-size', 13).attr('fill', ink2());
  line0(A, TRUTH);
  const aStem = A.layer('stems'), aFit = A.layer('fit'), aPts = A.layer('pts');
  B.layer('zero').append('line').attr('x1', 0).attr('x2', B.w).attr('y1', B.y(0)).attr('y2', B.y(0)).attr('stroke', muted()).attr('stroke-width', 1);
  const bStem = B.layer('stems'), bPts = B.layer('pts'), bTree = B.layer('tree');
  const eL = E.layer('c'), eM = E.layer('m');
  E.layer('floor').append('line').attr('x1', 0).attr('x2', E.w).attr('stroke', muted()).attr('stroke-dasharray', '5 4').attr('stroke-width', 1.5).attr('class', 'floor');
  const floorT = E.layer('floor').append('text').attr('font-size', 12).attr('fill', ink2()).attr('text-anchor', 'end').attr('x', E.w - 4).text(`noise floor σ² = ${(NOISE ** 2).toFixed(2)} (no model can beat this on average)`);

  function line0(F, fn) {
    F.layer('truth').append('path').datum(GRID).attr('fill', 'none').attr('stroke', muted()).attr('stroke-width', 1.5).attr('stroke-dasharray', '4 4')
      .attr('d', d3.line().x((x) => F.x(x)).y((x) => F.y(fn(x))));
  }
  const stepPath = (segs, scale, F) => {
    const pts = [];
    segs.forEach(([a, b, v]) => { const lo = Math.max(-1, a), hi = Math.min(1, b); pts.push([F.x(lo), F.y(scale * v)], [F.x(hi), F.y(scale * v)]); });
    return d3.line()(pts);
  };

  function recompute() { halt(); run = boost(train, test, nu, depth); tween = 1; paint(); }

  function paint() {
    const S = run.stages, cur = S[m], nxt = cur.next;           // nxt undefined at m = M
    const t = tween;                                             // 0..1 blend from F_{m-1} to F_m while animating
    const prev = m > 0 ? S[m - 1] : cur;
    const blend = (a, b) => a.map((v, i) => v + (b[i] - v) * t);
    const Fg = t < 1 ? blend(prev.Fg, cur.Fg) : cur.Fg, Ftr = t < 1 ? blend(prev.Ftr, cur.Ftr) : cur.Ftr;
    // left: model + residual stems
    aStem.selectAll('line').data(train).join('line').attr('x1', (d) => A.x(d.x)).attr('x2', (d) => A.x(d.x))
      .attr('y1', (d) => A.y(d.y)).attr('y2', (d, i) => A.y(Ftr[i])).attr('stroke', muted()).attr('stroke-width', 1.3);
    aFit.selectAll('path').data([Fg]).join('path').attr('fill', 'none').attr('stroke', ink()).attr('stroke-width', 2.5)
      .attr('d', d3.line().x((v, i) => A.x(GRID[i])).y((v) => A.y(v)));
    aPts.selectAll('circle').data(train).join('circle').attr('cx', (d) => A.x(d.x)).attr('cy', (d) => A.y(d.y)).attr('r', 4.5)
      .attr('fill', cat(0)).attr('stroke', css('--surface')).attr('stroke-width', 1.2)
      .on('mousemove', (e, d) => { const i = train.indexOf(d); tip(e, `x = ${d.x.toFixed(2)}, y = ${d.y.toFixed(2)}<br>F<sub>${m}</sub>(x) = ${cur.Ftr[i].toFixed(2)}<br>residual = ${(d.y - cur.Ftr[i]).toFixed(2)}`); })
      .on('mouseleave', untip);
    aTitle.text(m === 0 ? 'Stage 0: F₀ = mean of y (a flat line)' : `Stage ${m}: F${sub(m)} = F${sub(m - 1)} + ${nu.toFixed(2)} × tree ${m}`);
    // right: residuals of the *displayed* model and the next tree; while tweening, show the tree just added
    const showStage = t < 1 ? m - 1 : m, info = S[showStage].next;
    if (info) {
      bStem.selectAll('line').data(train).join('line').attr('x1', (d) => B.x(d.x)).attr('x2', (d) => B.x(d.x)).attr('y1', B.y(0)).attr('y2', (d, i) => B.y(info.r[i])).attr('stroke', muted()).attr('stroke-width', 1);
      bPts.selectAll('circle').data(train).join('circle').attr('cx', (d) => B.x(d.x)).attr('cy', (d, i) => B.y(Math.max(-1.6, Math.min(1.6, info.r[i])))).attr('r', 4.2)
        .attr('fill', cat(0)).attr('stroke', css('--surface')).attr('stroke-width', 1.2);
      bTree.selectAll('path.full').data([info.tree]).join('path').attr('class', 'full').attr('fill', 'none').attr('stroke', cat(1)).attr('stroke-width', 2).attr('stroke-dasharray', '6 4').attr('d', (d) => stepPath(d.segs, 1, B));
      bTree.selectAll('path.shrunk').data([info.tree]).join('path').attr('class', 'shrunk').attr('fill', 'none').attr('stroke', cat(1)).attr('stroke-width', 3).attr('d', (d) => stepPath(d.segs, nu, B));
      bTitle.text(`Residuals of F${sub(showStage)}, and tree ${showStage + 1} fitted to them (${info.tree.segs.length} leaves)`);
    } else {   // last stage: residuals only, no further tree
      const rr = train.map((p, i) => p.y - cur.Ftr[i]);
      bStem.selectAll('line').data(train).join('line').attr('x1', (d) => B.x(d.x)).attr('x2', (d) => B.x(d.x)).attr('y1', B.y(0)).attr('y2', (d, i) => B.y(rr[i])).attr('stroke', muted()).attr('stroke-width', 1);
      bPts.selectAll('circle').data(train).join('circle').attr('cx', (d) => B.x(d.x)).attr('cy', (d, i) => B.y(Math.max(-1.6, Math.min(1.6, rr[i])))).attr('r', 4.2)
        .attr('fill', cat(0)).attr('stroke', css('--surface')).attr('stroke-width', 1.2);
      bTree.selectAll('path').remove();
      bTitle.text(`Residuals of F${sub(M)}: all ${M} trees added`);
    }
    // error curves
    const ymax = Math.max(S[0].testMSE, S[0].trainMSE) * 1.15, ymin = Math.min(0.06, d3.min(S, (q) => q.trainMSE) * 0.8);
    const yt = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2].filter((v) => v >= ymin && v <= ymax);
    E.y.domain([ymin, ymax]); E.g.select('.axis.y').call(d3.axisLeft(E.y).tickValues(yt).tickFormat(d3.format('~g')));
    E.g.select('.gridline').call(d3.axisLeft(E.y).tickValues(yt).tickSize(-E.w).tickFormat(''));
    E.layer('floor').select('line').attr('y1', E.y(NOISE ** 2)).attr('y2', E.y(NOISE ** 2)); floorT.attr('y', E.y(NOISE ** 2) - 6);
    const ln = (key) => d3.line().x((d, i) => E.x(i)).y((d) => E.y(d[key]));
    eL.selectAll('path.tr').data([S]).join('path').attr('class', 'tr').attr('fill', 'none').attr('stroke', cat(2)).attr('stroke-width', 2).attr('d', ln('trainMSE'));
    eL.selectAll('path.te').data([S]).join('path').attr('class', 'te').attr('fill', 'none').attr('stroke', cat(3)).attr('stroke-width', 2).attr('d', ln('testMSE'));
    eL.selectAll('path.best').data([run.best]).join('path').attr('class', 'best').attr('d', d3.symbol().type(d3.symbolTriangle).size(70)())
      .attr('transform', (b) => `translate(${E.x(b)},${E.y(S[b].testMSE) + 10})`).attr('fill', ink());
    eL.selectAll('text.best').data([run.best]).join('text').attr('class', 'best').attr('font-size', 12).attr('fill', ink2())
      .attr('x', (b) => E.x(b) + (b > M * 0.75 ? -8 : 8)).attr('text-anchor', (b) => (b > M * 0.75 ? 'end' : 'start')).attr('y', E.h - 8).text((b) => `lowest test error, stage ${b}`);
    eM.selectAll('line').data([m]).join('line').attr('x1', (d) => E.x(d)).attr('x2', (d) => E.x(d)).attr('y1', 0).attr('y2', E.h).attr('stroke', ink()).attr('stroke-dasharray', '3 3');
    eM.selectAll('circle').data([['trainMSE', cat(2)], ['testMSE', cat(3)]]).join('circle').attr('cx', E.x(m)).attr('cy', ([k]) => E.y(cur[k])).attr('r', 5)
      .attr('fill', ([, c]) => c).attr('stroke', css('--surface')).attr('stroke-width', 1.5)
      .on('mousemove', (e, [k]) => tip(e, `stage ${m}<br>${k === 'trainMSE' ? 'training' : 'test'} MSE = ${cur[k].toFixed(3)}`)).on('mouseleave', untip);
    readout.set({ 'stage m': m, 'training MSE': cur.trainMSE.toFixed(3), 'test MSE': cur.testMSE.toFixed(3), 'best test MSE (stage)': `${S[run.best].testMSE.toFixed(3)} (${run.best})`, 'test MSE at stage 200': S[M].testMSE.toFixed(3) });
  }
  const sub = (k) => String(k).split('').map((c) => '₀₁₂₃₄₅₆₇₈₉'[+c]).join('');

  // animate one stage: tween F_{m} -> F_{m+1}
  function step(fast) {
    if (m >= M) return false;
    m += 1; mS.set(m); tween = 0;
    const dur = fast ? (m < 10 ? 450 : m < 30 ? 160 : 0) : 700;
    if (!dur) { tween = 1; paint(); return true; }
    let el2 = 0;
    stopTween = animate((dt) => { el2 += dt; tween = Math.min(1, el2 / dur); paint(); return tween < 1; });
    if (!fast) stopAnim = () => { stopAnim = null; tween = 1; paint(); };
    return true;
  }
  function play() {
    halt(); if (m >= M) { m = 0; }
    playB.textContent = 'Pause';
    let acc = 1e9;
    const stop = animate((dt) => {
      acc += dt;
      const wait = m < 10 ? 750 : m < 30 ? 240 : 40;
      if (tween < 1 || acc < wait) return true;
      acc = 0;
      if (!step(true)) { halt(); return false; }
      return true;
    });
    stopAnim = () => { stop(); stopAnim = null; playB.textContent = 'Play'; tween = 1; paint(); };
  }
  function halt() { if (stopTween) { stopTween(); stopTween = null; } if (stopAnim) stopAnim(); }
  paint();

  // ---- numbers for the prose (fixed settings, computed from the seeded data) ----
  const R = (n, d) => boost(train, test, n, d);
  const d1 = R(0.3, 1), first = d1.stages[0].next.tree.segs, firstThr = first[0][1];
  const fast = R(1, 3), slow = R(0.1, 1), mid = R(0.3, 3);
  const f3 = (v) => v.toFixed(3);
  section(main, 'What to notice', `<ul>
    <li>At stage 0 the model is a flat line at the mean of y (${d1.f0.toFixed(2)}). The first stump splits at x = ${firstThr.toFixed(2)}: it is the single cut that best separates positive residuals from negative ones. With ν = 0.3, only 30% of that step is added, so after one stage the line has barely moved.</li>
    <li>Step through the first 10 stages and watch the right panel: each new tree targets whatever pattern is left in the residuals, so successive stumps cut in different places. A stump can only make one step, yet the sum of many stumps traces the sine curve.</li>
    <li>With the defaults (ν = 0.3, stumps), test error is lowest at stage ${d1.best} (MSE ${f3(d1.stages[d1.best].testMSE)}) and is ${f3(d1.stages[M].testMSE)} at stage ${M}. The noise floor is ${(NOISE ** 2).toFixed(2)}; the remaining gap is the price of building a smooth curve out of a limited number of steps from only 50 noisy points.</li>
    <li>Set <b>ν = 1 and depth 3</b>: training error drops to ${f3(fast.stages[M].trainMSE)} by stage ${M}, far below the noise floor, so the model is fitting noise. Test error is lowest after ${fast.best === 1 ? 'just one tree' : `${fast.best} trees`} (${f3(fast.stages[fast.best].testMSE)}) and then climbs to ${f3(fast.stages[M].testMSE)}: every later tree mostly fits noise in the residuals. That is overfitting, and why boosting needs early stopping.</li>
    <li>Set <b>ν = 0.1 with stumps</b>: progress is slow and steady, and test error is still ${f3(slow.stages[M].testMSE)} at stage ${M} (lowest: ${f3(slow.stages[slow.best].testMSE)} at stage ${slow.best}). Smaller steps need more trees but are more forgiving about exactly when you stop. Depth 3 with ν = 0.3 reaches its best (${f3(mid.stages[mid.best].testMSE)}) at stage ${mid.best}, then drifts up.</li>
  </ul>`);

  steps(main, 'How it works', [
    'Start with a constant prediction F₀(x) = mean(y).',
    'Compute the <b>residuals</b> r<sub>i</sub> = y<sub>i</sub> − F(x<sub>i</sub>): what the model still gets wrong on each training point.',
    'Fit a small regression tree (often depth 1–6) to the residuals, using x as input and r as the target.',
    'Add a <b>shrunken</b> copy: F ← F + ν·tree. The learning rate ν (0.01–0.3 in practice) stops any one tree from dominating.',
    'Repeat for many stages; pick the number of stages with a validation set or cross-validation (early stopping).',
    'Why "gradient": for squared error, the residual is exactly the negative gradient of the loss with respect to F(x). For other losses (logistic, Poisson, Cox) each tree fits the negative gradient of <i>that</i> loss instead, so the same recipe handles classification and survival.',
  ]);
  math(main, 'F₀(x) = mean(y)\nr_i = y_i − F_{m−1}(x_i)  =  −∂/∂F [ ½ (y_i − F)² ]   at F = F_{m−1}(x_i)\nh_m = small tree fitted to (x_i, r_i)\nF_m(x) = F_{m−1}(x) + ν · h_m(x)');

  callout(main, '', 'In biology', 'Gradient-boosted trees (<code>XGBoost</code>, <code>LightGBM</code>, <code>CatBoost</code>) win most competitions on tabular data, and omics tables are tabular: response prediction from expression or clinical variables, variant pathogenicity scores built on many annotations, and protein or peptide property predictors. They are often the strongest baseline to beat before trying deep learning, and pair naturally with TreeSHAP for per-sample explanations.');
  callout(main, 'warn', 'Pitfalls', `<ul>
    <li><b>Boosting overfits if you let it run.</b> Unlike a random forest, adding trees can make test error worse. Always early-stop on held-out data.</li>
    <li>Learning rate and number of trees trade off: halve ν and you need roughly twice as many trees. Tune them together, not separately.</li>
    <li>With few samples (typical in omics: tens to a few hundred patients) boosting can memorise quickly; use shallow trees, row and feature subsampling, and strong regularisation.</li>
    <li>Early stopping on the test set is leakage: the stage count is a hyperparameter and must be chosen on validation data inside the CV loop.</li>
  </ul>`);
  seeAlso(main, [['concepts', 'decision-tree', 'Decision trees'], ['concepts', 'random-forest', 'Random forests & bagging'], ['concepts', 'gradient-descent', 'Gradient descent'], ['concepts', 'bias-variance', 'Under- vs overfitting'], ['concepts', 'training-tricks', 'Early stopping']]);
}
