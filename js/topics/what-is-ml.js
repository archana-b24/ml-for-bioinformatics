// What "learning" means: (1) fit a function from examples, watched step by step; (2) a clickable map of the kinds of ML.
import { header, figure, slider, button, checkbox, stats, section, steps, callout, math, seeAlso, legend, every, animate, tip, untip, el } from '../lib/ui.js';
import { frame, cat, css, ink, ink2, muted } from '../lib/plot.js';
import { rng, ridge, linspace, mean, range } from '../lib/data.js';
const d3 = window.d3;

// ---------- figure 1 model: Gaussian bumps (RBF features) + intercept, fitted by ridge ----------
const TRUTH = (x) => Math.sin(2.5 * x) + 0.3 * x;
const CENTERS = linspace(-1.1, 1.1, 12), WIDTH = 0.22, LAMBDA = 0.05;
const feats = (x) => [1, ...CENTERS.map((c) => Math.exp(-((x - c) ** 2) / (2 * WIDTH * WIDTH)))];
const predict = (w, x) => feats(x).reduce((s, f, i) => s + f * w[i], 0);
const GRID = linspace(-1, 1, 161);

function powerMax(A) { // largest eigenvalue of a symmetric PSD matrix
  let v = A.map(() => 1), lam = 0;
  for (let it = 0; it < 60; it++) {
    const u = A.map((r) => r.reduce((s, a, j) => s + a * v[j], 0));
    lam = Math.hypot(...u); v = u.map((t) => t / (lam || 1));
  }
  return lam;
}

export default async function render(main) {
  header(main, 'What "learning" means',
    'A model is a function with adjustable knobs (parameters). <b>Learning means turning those knobs until the function\'s outputs match the examples</b>, in the hope that it will also be right on examples it has never seen.');

  // ================= Figure 1: points -> learned curve =================
  const fig = figure(main, 'Learning a curve from examples',
    'Dots are training examples (x, y). The solid line is the learned function: 13 parameters (an intercept plus the heights of 12 smooth bumps) chosen to make the line pass close to the dots. The dashed line is the true rule that generated the data, which the model never sees. <b>Click on the plot to add your own example.</b>');
  const r = rng(7);
  const pool = range(60).map(() => { const x = r() * 2 - 1; return { x, y: TRUTH(x) + r.normal(0, 0.25) }; });
  let n = 8, extra = [], w = null, gdStop = null, gdIter = 0;
  const data = () => pool.slice(0, n).concat(extra);

  slider(fig.controls, { label: 'Training examples', min: 2, max: 60, step: 1, value: n, onInput: (v) => { n = v; stopGD(); fit(); } });
  let showTruth = true;
  checkbox(fig.controls, { label: 'Show true function', checked: true, onChange: (c) => { showTruth = c; paint(); } });
  button(fig.controls, 'Watch it learn (gradient descent)', () => runGD(), true);
  button(fig.controls, 'Clear my points', () => { extra = []; stopGD(); fit(); });
  const readout = stats(fig.root);
  const leg = el('div'); fig.body.append(leg);
  legend(leg, [['training examples', cat(0)], ['learned function', cat(1)], ['true rule (hidden from the model)', muted()]]);

  const F = frame(fig.body, { width: 900, height: 360, x: [-1.05, 1.05], y: [-2, 2], xLabel: 'input x (e.g. gene expression)', yLabel: 'output y' });
  F.layer('truth'); F.layer('fit'); F.layer('pts');
  F.svg.style('cursor', 'crosshair').on('click', (e) => {
    const [mx, my] = d3.pointer(e, F.g.node());
    const x = F.x.invert(mx), y = F.y.invert(my);
    if (x < -1.05 || x > 1.05 || y < -2 || y > 2) return;
    extra.push({ x, y, mine: true }); stopGD(); fit();
  });

  const design = (pts) => pts.map((p) => feats(p.x));
  function fit() {
    const pts = data(), X = design(pts);
    w = ridge(X, pts.map((p) => p.y), LAMBDA); gdIter = null; paint();
  }
  function stopGD() { if (gdStop) { gdStop(); gdStop = null; } }
  function runGD() {
    stopGD();
    const pts = data(), X = design(pts), y = pts.map((p) => p.y), p = X[0].length;
    const XtX = range(p).map((i) => range(p).map((j) => X.reduce((s, row) => s + row[i] * row[j], 0)));
    const lr = 1 / (2 * powerMax(XtX) + 2 * LAMBDA);
    w = new Array(p).fill(0); gdIter = 0;
    gdStop = animate(() => {
      for (let k = 0; k < 3; k++) {
        const res = X.map((row, i) => row.reduce((s, f, j) => s + f * w[j], 0) - y[i]);
        const g = w.map((wj, j) => 2 * X.reduce((s, row, i) => s + row[j] * res[i], 0) + 2 * LAMBDA * wj);
        w = w.map((wj, j) => wj - lr * g[j]); gdIter++;
      }
      paint();
      return gdIter < 1500;
    });
  }

  function paint() {
    const pts = data();
    F.layer('truth').selectAll('path').data(showTruth ? [GRID] : []).join('path').attr('fill', 'none').attr('stroke', muted())
      .attr('stroke-width', 2).attr('stroke-dasharray', '5 4').attr('d', d3.line().x((x) => F.x(x)).y((x) => F.y(TRUTH(x))));
    const curve = GRID.map((x) => ({ x, y: Math.max(-2.2, Math.min(2.2, predict(w, x))) }));
    F.layer('fit').selectAll('path').data([curve]).join('path').attr('fill', 'none').attr('stroke', cat(1)).attr('stroke-width', 2.5)
      .attr('d', d3.line().x((d) => F.x(d.x)).y((d) => F.y(d.y)));
    F.layer('pts').selectAll('circle').data(pts).join('circle').attr('cx', (d) => F.x(d.x)).attr('cy', (d) => F.y(d.y))
      .attr('r', (d) => (d.mine ? 6 : 4.5)).attr('fill', cat(0)).attr('stroke', (d) => (d.mine ? ink() : css('--surface'))).attr('stroke-width', 1.5)
      .on('mousemove', (e, d) => tip(e, `x = ${d.x.toFixed(2)}<br>y = ${d.y.toFixed(2)}<br>prediction = ${predict(w, d.x).toFixed(2)}${d.mine ? '<br><i>your point</i>' : ''}`))
      .on('mouseleave', untip);
    const trainErr = mean(pts.map((p) => (predict(w, p.x) - p.y) ** 2));
    const trueErr = mean(GRID.map((x) => (predict(w, x) - TRUTH(x)) ** 2));
    readout.set({
      'training examples': pts.length,
      'training error (MSE)': trainErr.toFixed(3),
      'error vs true rule (MSE)': trueErr.toFixed(3),
      'fitted by': gdIter == null ? 'closed form' : `GD step ${gdIter}`,
    });
  }
  fit();

  // ================= Figure 2: the map of ML =================
  const fig2 = figure(main, 'A map of machine learning',
    'Click a box on the map. What separates the families is <b>what feedback the learner gets</b>: correct answers (supervised), none at all (unsupervised), answers hidden inside the data itself (self-supervised), or a reward after acting (reinforcement).');
  const row = el('div', { class: 'fig-row' }); fig2.body.append(row);
  const mapDiv = el('div'), toyDiv = el('div'); row.append(mapDiv, toyDiv);
  const toyTitle = el('div', { class: 'fig-title', style: 'margin:2px 0 4px' });
  const toySvgDiv = el('div');
  const toyText = el('div', { class: 'prose', style: 'font-size:14px' });
  toyDiv.append(toyTitle, toySvgDiv, toyText);

  const FAM = [
    { id: 'sup', label: 'Supervised', sub: 'labels given', y: 62 },
    { id: 'unsup', label: 'Unsupervised', sub: 'no labels', y: 172 },
    { id: 'self', label: 'Self-supervised', sub: 'labels from the data', y: 262 },
    { id: 'rl', label: 'Reinforcement', sub: 'reward after acting', y: 332 },
  ];
  const LEAVES = [
    { id: 'reg', fam: 'sup', label: 'Regression', y: 35 }, { id: 'cls', fam: 'sup', label: 'Classification', y: 89 },
    { id: 'clu', fam: 'unsup', label: 'Clustering', y: 145 }, { id: 'dim', fam: 'unsup', label: 'Dim. reduction', y: 199 },
    { id: 'mask', fam: 'self', label: 'Masked prediction', y: 262 }, { id: 'pol', fam: 'rl', label: 'Policy learning', y: 332 },
  ];
  const ms = d3.select(mapDiv).append('svg').attr('viewBox', '0 0 470 370').attr('width', 470).attr('role', 'img').attr('aria-label', 'Map of machine learning');
  const RX = 8, root = { x: 10, y: 197, w: 92, h: 44 }, famX = 140, famW = 150, leafX = 330, leafW = 132, bh = 38;
  const link = (x1, y1, x2, y2) => `M${x1},${y1} C${(x1 + x2) / 2},${y1} ${(x1 + x2) / 2},${y2} ${x2},${y2}`;
  const linkG = ms.append('g');
  FAM.forEach((f) => linkG.append('path').attr('d', link(root.x + root.w, root.y, famX, f.y)).attr('fill', 'none').attr('stroke', css('--axis')).attr('stroke-width', 1.5));
  LEAVES.forEach((l) => { const f = FAM.find((q) => q.id === l.fam); linkG.append('path').attr('d', link(famX + famW, f.y, leafX, l.y)).attr('fill', 'none').attr('stroke', css('--axis')).attr('stroke-width', 1.5); });
  const rg = ms.append('g');
  rg.append('rect').attr('x', root.x).attr('y', root.y - root.h / 2).attr('width', root.w).attr('height', root.h).attr('rx', RX).attr('fill', css('--surface-2')).attr('stroke', css('--axis'));
  rg.append('text').attr('x', root.x + root.w / 2).attr('y', root.y - 3).attr('text-anchor', 'middle').attr('font-size', 13).attr('font-weight', 600).attr('fill', ink()).text('Machine');
  rg.append('text').attr('x', root.x + root.w / 2).attr('y', root.y + 13).attr('text-anchor', 'middle').attr('font-size', 13).attr('font-weight', 600).attr('fill', ink()).text('learning');
  const famG = ms.append('g').selectAll('g').data(FAM).join('g').style('cursor', 'pointer')
    .on('click', (e, f) => choose(LEAVES.find((l) => l.fam === f.id).id));
  famG.append('rect').attr('x', famX).attr('y', (f) => f.y - 22).attr('width', famW).attr('height', 44).attr('rx', RX);
  famG.append('text').attr('x', famX + 10).attr('y', (f) => f.y - 3).attr('font-size', 13.5).attr('font-weight', 600).attr('fill', ink()).text((f) => f.label);
  famG.append('text').attr('x', famX + 10).attr('y', (f) => f.y + 13).attr('font-size', 11.5).attr('fill', ink2()).text((f) => f.sub);
  const leafG = ms.append('g').selectAll('g').data(LEAVES).join('g').style('cursor', 'pointer').on('click', (e, l) => choose(l.id));
  leafG.append('rect').attr('x', leafX).attr('y', (l) => l.y - bh / 2 + 3).attr('width', leafW).attr('height', bh - 6).attr('rx', RX);
  leafG.append('text').attr('x', leafX + leafW / 2).attr('y', (l) => l.y + 4.5).attr('text-anchor', 'middle').attr('font-size', 13).attr('fill', ink()).text((l) => l.label);

  let stopToy = null;
  function choose(id) {
    const leaf = LEAVES.find((l) => l.id === id);
    leafG.select('rect').attr('fill', (l) => (l.id === id ? css('--accent-soft') : css('--surface'))).attr('stroke', (l) => (l.id === id ? css('--accent') : css('--axis'))).attr('stroke-width', (l) => (l.id === id ? 2 : 1));
    famG.select('rect').attr('fill', (f) => (f.id === leaf.fam ? css('--accent-soft') : css('--surface'))).attr('stroke', (f) => (f.id === leaf.fam ? css('--accent') : css('--axis'))).attr('stroke-width', (f) => (f.id === leaf.fam ? 2 : 1));
    if (stopToy) { stopToy(); stopToy = null; }
    toySvgDiv.innerHTML = '';
    const T = TOYS[id];
    toyTitle.textContent = T.title;
    stopToy = T.draw(toySvgDiv) || null;
    toyText.innerHTML = `<p style="margin:4px 0"><b>Given:</b> ${T.given}<br><b>Learns:</b> ${T.learns}</p><p style="margin:4px 0"><b>In biology:</b> ${T.bio}</p>`;
  }
  choose('reg');

  section(main, 'What to notice', `<ul>
    <li>With <b>2 examples</b> the training error is 0.000 but the error against the true rule is 0.49: the line passes exactly through both dots and is wrong almost everywhere else. Away from data the model has no evidence, so it falls back on its default (flat, near zero). <b>Perfect training error is not learning.</b></li>
    <li>Slide to <b>20</b> and then <b>60 examples</b>: the error against the true rule drops to 0.038 and then 0.015. The training error <i>rises</i> to 0.060, which is the noise variance (0.25² ≈ 0.06): the model has stopped chasing noise it cannot predict.</li>
    <li>Press <b>Watch it learn</b>: the line starts flat (all parameters zero) and gradient descent nudges it toward the dots. The training error falls step by step until it matches the closed-form answer. That loop is all "training" means.</li>
    <li>Click to add a wild point far from the dashed line. The model bends to chase it, because it only knows the examples and cannot tell noise from signal.</li>
    <li>In the map, the toy pictures differ mainly in <b>what feedback is available</b>, not in the maths. The same gradient-descent machinery trains models in all four families.</li>
  </ul>`);

  steps(main, 'How it works', [
    'Choose a <b>model family</b>: a function with parameters, e.g. <code>ŷ = w₀ + w₁·bump₁(x) + … + w₁₂·bump₁₂(x)</code>. Choosing the family encodes your assumptions (here: the answer is a smooth curve).',
    'Choose a <b>loss</b>: a number that says how wrong the predictions are on the training examples, e.g. the mean squared error.',
    'Choose an <b>optimiser</b> that adjusts the parameters to lower the loss. Here the problem is small enough to solve in one shot (the closed form), and gradient descent reaches the same answer step by step.',
    '<b>Evaluate on data the model never saw.</b> Low training error is easy (a model can memorise); the real target is low error on new examples, which is what the "error vs true rule" readout measures.',
  ]);
  math(main, 'learning  =  find θ minimising  (1/n) Σᵢ loss( f_θ(xᵢ), yᵢ )  +  λ·penalty(θ)\ngoal       =  small error on NEW (x, y) drawn from the same source, not just on the n you have');

  callout(main, '', 'In biology', '<ul style="margin:4px 0"><li><b>Supervised:</b> Horvath\'s epigenetic clock is a penalised linear regression from CpG methylation to age; CellTypist is a logistic-regression classifier trained on labelled scRNA-seq atlases.</li><li><b>Unsupervised:</b> PCA and Leiden clustering in every Scanpy/Seurat workflow; the cluster names come later from a human reading marker genes.</li><li><b>Self-supervised:</b> Geneformer and scGPT hide genes in each cell and learn to predict them from the rest, across tens of millions of cells, with no labels at all.</li><li><b>Reinforcement:</b> rare in omics; used mainly in generative chemistry (e.g. REINVENT steers a molecule generator towards a property score).</li></ul>');
  callout(main, 'warn', 'Pitfalls', '<ul><li>"The model fits the training data" is not evidence that it learned anything. Always ask how it does on held-out samples (<a href="#/concepts/train-test">train/test</a>).</li><li>A model can only learn patterns present in its examples. Train on one tissue, sequencing centre or ancestry and it may silently fail on another.</li><li>Clusters from unsupervised learning are hypotheses, not cell types. Different settings give different clusters, all of them "valid".</li></ul>');
  seeAlso(main, [['concepts', 'train-test', 'Train, validation, test'], ['concepts', 'bias-variance', 'Under- vs overfitting'], ['concepts', 'gradient-descent', 'Gradient descent'], ['concepts', 'foundation-models', 'Self-supervision & foundation models']]);
}

// ---------- toy pictures for the map (each returns an optional stop() for its animation) ----------
const TW = 470, TH = 270;
function toyFrame(div) {
  return frame(div, { width: TW, height: TH, x: [-1.3, 1.3], y: [-1.05, 1.05], grid: false, axes: false, margin: { top: 6, right: 6, bottom: 6, left: 6 } });
}
const dots = (F, pts, color) => F.layer('pts').selectAll('circle').data(pts).join('circle').attr('cx', (d) => F.x(d.x)).attr('cy', (d) => F.y(d.y))
  .attr('r', 4.5).attr('fill', color).attr('stroke', css('--surface')).attr('stroke-width', 1.2);
const note = (F, x, y, t, anchor = 'start') => F.g.append('text').attr('x', F.x(x)).attr('y', F.y(y)).attr('text-anchor', anchor).attr('font-size', 12).attr('fill', ink2()).text(t);

const TOYS = {
  reg: {
    title: 'Supervised · Regression: predict a number',
    given: 'inputs <i>x</i> paired with a numeric answer <i>y</i>.',
    learns: 'a function <i>f(x)</i> that predicts <i>y</i> for a new <i>x</i>.',
    bio: 'predicting age from DNA methylation (epigenetic clocks), drug IC50 from a cell line\'s expression, or protein abundance from mRNA.',
    draw(div) {
      const F = toyFrame(div), r = rng(11);
      const pts = range(26).map(() => { const x = r() * 2.2 - 1.1; return { x, y: 0.6 * x - 0.1 + r.normal(0, 0.15) }; });
      const xs = pts.map((p) => p.x), ys = pts.map((p) => p.y), mx = mean(xs), my = mean(ys);
      const b = d3.sum(pts, (p) => (p.x - mx) * (p.y - my)) / d3.sum(pts, (p) => (p.x - mx) ** 2), a = my - b * mx;
      F.layer('line').append('path').attr('d', `M${F.x(-1.2)},${F.y(a - 1.2 * b)}L${F.x(1.2)},${F.y(a + 1.2 * b)}`).attr('stroke', cat(1)).attr('stroke-width', 2.5);
      F.layer('res').selectAll('line').data(pts).join('line').attr('x1', (d) => F.x(d.x)).attr('x2', (d) => F.x(d.x)).attr('y1', (d) => F.y(d.y)).attr('y2', (d) => F.y(a + b * d.x)).attr('stroke', muted()).attr('stroke-width', 1);
      dots(F, pts, cat(0));
      note(F, -1.25, 0.9, 'x = methylation at a CpG, y = age'); note(F, 1.25, -0.95, 'grey ticks = errors the fit minimises', 'end');
    },
  },
  cls: {
    title: 'Supervised · Classification: predict a category',
    given: 'inputs <i>x</i> each tagged with a class (e.g. responder / non-responder).',
    learns: 'a decision boundary that assigns a class (or a probability) to a new <i>x</i>.',
    bio: 'calling cell types in scRNA-seq from a labelled reference (CellTypist), or responder vs non-responder from a baseline tumour RNA-seq profile.',
    draw(div) {
      const F = toyFrame(div), r = rng(12);
      const pts = range(50).map((i) => { const c = i % 2; return { x: r.normal(c ? 0.45 : -0.45, 0.28), y: r.normal(c ? 0.25 : -0.25, 0.28), label: c }; });
      const m0 = [mean(pts.filter((p) => !p.label).map((p) => p.x)), mean(pts.filter((p) => !p.label).map((p) => p.y))];
      const m1 = [mean(pts.filter((p) => p.label).map((p) => p.x)), mean(pts.filter((p) => p.label).map((p) => p.y))];
      // nearest-centroid boundary: perpendicular bisector of the two class means
      const mid = [(m0[0] + m1[0]) / 2, (m0[1] + m1[1]) / 2], dir = [-(m1[1] - m0[1]), m1[0] - m0[0]], L = 3 / Math.hypot(...dir);
      F.layer('line').append('path').attr('d', `M${F.x(mid[0] - dir[0] * L)},${F.y(mid[1] - dir[1] * L)}L${F.x(mid[0] + dir[0] * L)},${F.y(mid[1] + dir[1] * L)}`)
        .attr('stroke', ink()).attr('stroke-width', 2).attr('stroke-dasharray', '6 4');
      F.svg.append('defs').append('clipPath').attr('id', 'toyclip').append('rect').attr('width', F.w).attr('height', F.h);
      F.layer('line').attr('clip-path', 'url(#toyclip)');
      dots(F, pts, (d) => cat(d.label));
      note(F, -1.25, 0.9, 'dashed line = learned decision boundary');
      const lg = el('div'); div.prepend(lg); legend(lg, [['class 0 (e.g. non-responder)', cat(0)], ['class 1 (e.g. responder)', cat(1)]]);
    },
  },
  clu: {
    title: 'Unsupervised · Clustering: find groups',
    given: 'only inputs <i>x</i>, with no answers attached.',
    learns: 'a grouping of similar points (here k-means with k = 3, animated from grey to coloured).',
    bio: 'Leiden clustering of scRNA-seq cells on a kNN graph; the clusters become cell-type <i>hypotheses</i> that are named afterwards from marker genes.',
    draw(div) {
      const F = toyFrame(div), r = rng(13);
      const C = [[-0.7, 0.4], [0.6, 0.5], [0, -0.5]];
      const pts = range(75).map((i) => ({ x: r.normal(C[i % 3][0], 0.2), y: r.normal(C[i % 3][1], 0.2) }));
      let cents = [[-1.1, -0.9], [0.0, 0.9], [1.1, -0.9]], it = 0, assign = null;
      const step = () => {
        assign = pts.map((p) => d3.minIndex(cents, (c) => (p.x - c[0]) ** 2 + (p.y - c[1]) ** 2));
        dots(F, pts, (d, i) => cat(assign[i]));
        F.layer('cent').selectAll('path').data(cents).join('path').attr('d', d3.symbol(d3.symbolCross, 160)).attr('transform', (c) => `translate(${F.x(c[0])},${F.y(c[1])})`)
          .attr('fill', ink()).attr('stroke', css('--surface')).attr('stroke-width', 1.5);
        cents = cents.map((c, k) => { const m = pts.filter((_, i) => assign[i] === k); return m.length ? [mean(m.map((p) => p.x)), mean(m.map((p) => p.y))] : c; });
      };
      dots(F, pts, muted());
      note(F, -1.25, 0.95, '+ = cluster centres; no labels were used');
      return every(700, () => { if (it++ < 6) step(); });
    },
  },
  dim: {
    title: 'Unsupervised · Dimension reduction: find the main axes',
    given: 'only inputs <i>x</i> with many correlated features (here 2, in practice ~2,000 genes).',
    learns: 'a few new axes that keep most of the variation (here PCA\'s first component).',
    bio: 'PCA of highly variable genes to ~50 components before neighbour graphs in scRNA-seq; scVI learns a non-linear version of the same idea.',
    draw(div) {
      const F = toyFrame(div), r = rng(14);
      const pts = range(60).map(() => { const t = r.normal(0, 0.55), s = r.normal(0, 0.12); return { x: 0.85 * t - 0.5 * s, y: 0.5 * t + 0.85 * s }; });
      const mx = mean(pts.map((p) => p.x)), my = mean(pts.map((p) => p.y));
      const a = mean(pts.map((p) => (p.x - mx) ** 2)), b = mean(pts.map((p) => (p.x - mx) * (p.y - my))), d = mean(pts.map((p) => (p.y - my) ** 2));
      const th = 0.5 * Math.atan2(2 * b, a - d), u = [Math.cos(th), Math.sin(th)];
      const proj = pts.map((p) => { const t = (p.x - mx) * u[0] + (p.y - my) * u[1]; return { x: mx + t * u[0], y: my + t * u[1] }; });
      F.layer('pc').append('path').attr('d', `M${F.x(mx - 1.4 * u[0])},${F.y(my - 1.4 * u[1])}L${F.x(mx + 1.4 * u[0])},${F.y(my + 1.4 * u[1])}`).attr('stroke', cat(1)).attr('stroke-width', 2.5);
      F.layer('proj').selectAll('line').data(pts).join('line').attr('x1', (p) => F.x(p.x)).attr('y1', (p) => F.y(p.y)).attr('x2', (p, i) => F.x(proj[i].x)).attr('y2', (p, i) => F.y(proj[i].y)).attr('stroke', muted()).attr('stroke-width', 1);
      F.layer('pp').selectAll('circle').data(proj).join('circle').attr('cx', (p) => F.x(p.x)).attr('cy', (p) => F.y(p.y)).attr('r', 2.5).attr('fill', cat(1));
      dots(F, pts, cat(0));
      note(F, -1.25, 0.95, 'line = PC1; each point is replaced by its position along it');
    },
  },
  mask: {
    title: 'Self-supervised · Masked prediction: hide part, predict it',
    given: 'unlabelled data; the "label" is a piece of the input that is hidden on purpose.',
    learns: 'representations good enough to fill in the blank, which turn out to be useful for other tasks.',
    bio: 'Geneformer orders each cell\'s genes by expression, masks some, and predicts them (pretrained on ~30 million cells); scGPT and other single-cell foundation models use similar objectives.',
    draw(div) {
      const svg = d3.select(div).append('svg').attr('viewBox', `0 0 ${TW} ${TH}`).attr('width', TW);
      const genes = ['CD3E', 'CD8A', '[MASK]', 'GZMB', 'NKG7', 'CCL5'];
      const bw = 66, gap = 8, x0 = (TW - genes.length * (bw + gap) + gap) / 2;
      svg.append('text').attr('x', x0).attr('y', 22).attr('font-size', 12).attr('fill', ink2()).text('one cell written as its genes, highest expression first:');
      genes.forEach((g, i) => {
        const masked = g === '[MASK]';
        svg.append('rect').attr('x', x0 + i * (bw + gap)).attr('y', 34).attr('width', bw).attr('height', 34).attr('rx', 6)
          .attr('fill', masked ? css('--accent-soft') : css('--surface-2')).attr('stroke', masked ? css('--accent') : css('--axis')).attr('stroke-width', masked ? 2 : 1);
        svg.append('text').attr('x', x0 + i * (bw + gap) + bw / 2).attr('y', 56).attr('text-anchor', 'middle').attr('font-size', 12.5).attr('font-family', css('--mono')).attr('fill', ink()).text(g);
      });
      const mxp = x0 + 2 * (bw + gap) + bw / 2;
      svg.append('path').attr('d', `M${mxp},72 L${mxp},98`).attr('stroke', ink2()).attr('stroke-width', 1.5).attr('marker-end', 'url(#arrM)');
      svg.append('defs').append('marker').attr('id', 'arrM').attr('viewBox', '0 0 10 10').attr('refX', 8).attr('refY', 5).attr('markerWidth', 7).attr('markerHeight', 7).attr('orient', 'auto')
        .append('path').attr('d', 'M0,0L10,5L0,10z').attr('fill', ink2());
      svg.append('text').attr('x', mxp + 8).attr('y', 90).attr('font-size', 12).attr('fill', ink2()).text('model\'s guess (illustrative numbers):');
      const guess = [['CD8B', 0.58], ['CD2', 0.17], ['CD4', 0.07], ['MS4A1', 0.01]];
      const bx = 150, by = 110, bwMax = 160;
      guess.forEach(([g, p], i) => {
        svg.append('text').attr('x', bx - 8).attr('y', by + i * 32 + 16).attr('text-anchor', 'end').attr('font-size', 12.5).attr('font-family', css('--mono')).attr('fill', ink()).text(g);
        svg.append('rect').attr('x', bx).attr('y', by + i * 32 + 3).attr('height', 20).attr('rx', 3).attr('width', Math.max(2, p * bwMax / 0.6)).attr('fill', i === 0 ? cat(1) : cat(0));
        svg.append('text').attr('x', bx + Math.max(2, p * bwMax / 0.6) + 6).attr('y', by + i * 32 + 17).attr('font-size', 12).attr('fill', ink2()).text(p.toFixed(2) + (i === 0 ? '  ← the hidden gene: low loss' : ''));
      });
    },
  },
  pol: {
    title: 'Reinforcement · Policy learning: act, get reward, improve',
    given: 'an environment to act in and a reward signal (here: −1 per move, +10 at the goal; dark cells are walls).',
    learns: 'a policy: which action to take in each state. Arrows show the policy found by value iteration; the dot follows it.',
    bio: 'uncommon in omics. The main use is generative chemistry (e.g. REINVENT rewards a molecule generator for good predicted properties) and adaptive design of experiments.',
    draw(div) {
      const N = 6, cell = 40, ox = (TW - N * cell) / 2, oy = 10;
      const walls = new Set(['1,1', '1,2', '1,3', '3,2', '3,3', '3,4', '4,4']), goal = [5, 0], start = [0, 5];
      const key = (i, j) => `${i},${j}`, ok = (i, j) => i >= 0 && j >= 0 && i < N && j < N && !walls.has(key(i, j));
      const A = [[0, -1, '↑'], [1, 0, '→'], [0, 1, '↓'], [-1, 0, '←']];
      const V = {}; for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) V[key(i, j)] = 0;
      for (let it = 0; it < 60; it++) for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
        if (!ok(i, j) || (i === goal[0] && j === goal[1])) continue;
        V[key(i, j)] = d3.max(A, ([di, dj]) => { const [a, b] = ok(i + di, j + dj) ? [i + di, j + dj] : [i, j]; return (a === goal[0] && b === goal[1] ? 10 : -1) + 0.95 * V[key(a, b)]; });
      }
      const pol = (i, j) => d3.maxIndex(A, ([di, dj]) => { const [a, b] = ok(i + di, j + dj) ? [i + di, j + dj] : [i, j]; return (a === goal[0] && b === goal[1] ? 10 : -1) + 0.95 * V[key(a, b)]; });
      const svg = d3.select(div).append('svg').attr('viewBox', `0 0 ${TW} ${TH}`).attr('width', TW);
      for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
        const w = walls.has(key(i, j)), g = i === goal[0] && j === goal[1];
        svg.append('rect').attr('x', ox + i * cell).attr('y', oy + j * cell).attr('width', cell - 2).attr('height', cell - 2).attr('rx', 4)
          .attr('fill', w ? ink2() : g ? css('--accent-soft') : css('--surface-2')).attr('stroke', g ? css('--accent') : 'none');
        if (g) svg.append('text').attr('x', ox + i * cell + cell / 2 - 1).attr('y', oy + j * cell + cell / 2 + 4).attr('text-anchor', 'middle').attr('font-size', 11).attr('font-weight', 600).attr('fill', ink()).text('goal');
        else if (!w) svg.append('text').attr('x', ox + i * cell + cell / 2 - 1).attr('y', oy + j * cell + cell / 2 + 5).attr('text-anchor', 'middle').attr('font-size', 15).attr('fill', muted()).text(A[pol(i, j)][2]);
      }
      const agent = svg.append('circle').attr('r', 9).attr('fill', cat(1)).attr('stroke', css('--surface')).attr('stroke-width', 2);
      let pos = start.slice();
      const place = () => agent.attr('cx', ox + pos[0] * cell + cell / 2 - 1).attr('cy', oy + pos[1] * cell + cell / 2 - 1);
      place();
      return every(450, () => {
        if (pos[0] === goal[0] && pos[1] === goal[1]) pos = start.slice();
        else { const [di, dj] = A[pol(...pos)]; pos = [pos[0] + di, pos[1] + dj]; }
        place();
      });
    },
  },
};
