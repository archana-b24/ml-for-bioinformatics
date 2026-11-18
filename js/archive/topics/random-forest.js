// Random forests & bagging: bootstrap samples -> fully grown (jagged, high-variance) trees -> averaged votes.
// Shows in-bag vs out-of-bag points, individual tree boundaries vs the smooth forest average, and OOB vs test error.
import { header, figure, slider, select, button, checkbox, stats, section, steps, callout, math, legend, seeAlso, animate, tip, untip, el } from '../../lib/ui.js';
import { frame, decisionCanvas, cat, css, ink, ink2, muted } from '../../lib/plot.js';
import { rng, moons } from '../../lib/data.js';
const d3 = window.d3;

const N_TREES = 200;
const XD = [-1.75, 1.8], YD = [-1.35, 1.4];

// ---------- CART classification tree on 2 features, fully grown (min leaf 1) ----------
function growTree(data, idx, mtry, r, depth = 0) {
  let c1 = 0; idx.forEach((i) => { c1 += data[i].label; });
  const n = idx.length, p = c1 / n;
  if (c1 === 0 || c1 === n || depth >= 24) return { leaf: true, p };
  const feats = mtry === 1 ? r.shuffle([0, 1]) : [0, 1];
  let best = null;
  for (let k = 0; k < feats.length; k++) {
    const f = feats[k], key = f === 0 ? 'x' : 'y';
    const s = idx.slice().sort((a, b) => data[a][key] - data[b][key]);
    let l1 = 0;
    for (let t = 0; t < n - 1; t++) {
      l1 += data[s[t]].label;
      const va = data[s[t]][key], vb = data[s[t + 1]][key];
      if (va === vb) continue;
      const nl = t + 1, nr = n - nl, r1 = c1 - l1;
      const gl = 1 - (l1 / nl) ** 2 - (1 - l1 / nl) ** 2, gr = 1 - (r1 / nr) ** 2 - (1 - r1 / nr) ** 2;
      const w = (nl * gl + nr * gr) / n;
      if (!best || w < best.w - 1e-12) best = { w, f, key, thr: (va + vb) / 2 };
    }
    // random forest: consider only the one randomly chosen feature, unless it cannot split at all
    if (mtry === 1 && best) break;
  }
  if (!best) return { leaf: true, p };
  const L = idx.filter((i) => data[i][best.key] <= best.thr), R = idx.filter((i) => data[i][best.key] > best.thr);
  return { leaf: false, f: best.f, thr: best.thr, l: growTree(data, L, mtry, r, depth + 1), r: growTree(data, R, mtry, r, depth + 1) };
}
function predictTree(t, x, y) { while (!t.leaf) t = (t.f === 0 ? x : y) <= t.thr ? t.l : t.r; return t.p; }
function countLeaves(t) { return t.leaf ? 1 : countLeaves(t.l) + countLeaves(t.r); }

function buildForest(train, mtry, seed) {
  const r = rng(seed), n = train.length, trees = [];
  for (let k = 0; k < N_TREES; k++) {
    const counts = new Uint8Array(n), idx = [];
    for (let i = 0; i < n; i++) { const j = r.int(n); counts[j]++; idx.push(j); }
    const tree = growTree(train, idx, mtry, r);
    trees.push({ tree, counts, leaves: countLeaves(tree) });
  }
  return trees;
}

export default async function render(main) {
  header(main, 'Random forests & bagging',
    'A single deep decision tree is jagged and unstable: change a few samples and it changes a lot. A random forest trains <b>hundreds of trees, each on a different random resample of the data</b>, and averages their votes. The trees\' individual mistakes point in different directions, so they cancel, and the average boundary comes out smooth.');

  const train = moons(160, 0.22, 31), test = moons(600, 0.22, 32);
  const forests = { rf: buildForest(train, 1, 101), bag: buildForest(train, 2, 101) };

  // per-forest summary curves: test error and OOB error after k trees, plus single-tree test accuracy
  function curves(trees) {
    const n = train.length, sumTe = new Float64Array(test.length), sumOob = new Float64Array(n), cntOob = new Uint16Array(n);
    const teP = trees.map((t) => test.map((p) => predictTree(t.tree, p.x, p.y)));
    const trP = trees.map((t) => train.map((p) => predictTree(t.tree, p.x, p.y)));
    const errOf = (p, lab) => (p === 0.5 ? 0.5 : (p > 0.5 ? 1 : 0) !== lab ? 1 : 0);
    const out = [];
    trees.forEach((t, k) => {
      for (let i = 0; i < test.length; i++) sumTe[i] += teP[k][i];
      for (let i = 0; i < n; i++) if (!t.counts[i]) { sumOob[i] += trP[k][i]; cntOob[i]++; }
      let e = 0; for (let i = 0; i < test.length; i++) e += errOf(sumTe[i] / (k + 1), test[i].label);
      let eo = 0, no = 0; for (let i = 0; i < n; i++) if (cntOob[i]) { eo += errOf(sumOob[i] / cntOob[i], train[i].label); no++; }
      out.push({ k: k + 1, test: e / test.length, oob: no ? eo / no : NaN, covered: no });
    });
    const single = teP.map((pr) => d3.mean(pr, (p, i) => errOf(p, test[i].label)));
    const oobFrac = d3.mean(trees, (t) => t.counts.reduce((s, c) => s + (c === 0), 0) / n);
    return { out, single, meanSingle: d3.mean(single), oobFrac, trP };
  }
  const C = { rf: curves(forests.rf), bag: curves(forests.bag) };
  const pct = (v) => (100 * v).toFixed(1) + '%';

  // ================= Figure 1: bootstrap samples and their trees =================
  const fig1 = figure(main, 'Each tree sees a different bootstrap sample',
    'Each panel is one tree of the forest. Its training set is a <b>bootstrap sample</b>: 160 draws <i>with replacement</i> from the 160 training points. Bigger dots were drawn more than once; <b>hollow rings were never drawn</b> (they are "out-of-bag" for that tree). The shading is that tree\'s own prediction. Each tree is grown until every leaf is pure, so it bends around its particular sample, noise included.');
  let offset = 0;
  button(fig1.controls, 'Show next 6 trees', () => { offset = (offset + 6) % N_TREES; paint1(); }, true);
  button(fig1.controls, 'Back to trees 1–6', () => { offset = 0; paint1(); });
  const grid1 = el('div', { style: 'display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:12px' });
  fig1.body.append(grid1);
  legend(fig1.body, [['class 0', cat(0)], ['class 1', cat(1)]]);
  const readout1 = stats(fig1.root);

  function paint1() {
    grid1.innerHTML = '';
    const trees = forests.rf;
    for (let k = offset; k < offset + 6; k++) {
      const t = trees[k], cell = el('div');
      const nOob = t.counts.reduce((s, c) => s + (c === 0), 0);
      let ok = 0; train.forEach((p, i) => { if (!t.counts[i] && ((predictTree(t.tree, p.x, p.y) > 0.5 ? 1 : 0) === p.label)) ok++; });
      cell.append(el('div', { style: 'font-size:13px;color:var(--ink-2);margin-bottom:2px' }, `Tree ${k + 1}: ${t.leaves} leaves · ${nOob} out-of-bag points, ${pct(ok / nOob)} of them right`));
      grid1.append(cell);
      const F = frame(cell, { width: 300, height: 230, x: XD, y: YD, grid: false, axes: false, margin: { top: 4, right: 4, bottom: 4, left: 4 } });
      F.svg.style('background', css('--surface')).style('border', '1px solid var(--border)').style('border-radius', '6px');
      decisionCanvas(F, (x, y) => (predictTree(t.tree, x, y) > 0.5 ? 1 : 0), { res: 3, classes: true, alpha: 0.3 });
      const pts = train.map((p, i) => ({ ...p, c: t.counts[i] }));
      F.layer('pts').selectAll('circle').data(pts.filter((d) => d.c > 0)).join('circle')
        .attr('cx', (d) => F.x(d.x)).attr('cy', (d) => F.y(d.y)).attr('r', (d) => 2.4 + 1.6 * d.c)
        .attr('fill', (d) => cat(d.label)).attr('stroke', css('--surface')).attr('stroke-width', 1)
        .on('mousemove', (e, d) => tip(e, `class ${d.label}<br>drawn ${d.c}× into this tree's sample`)).on('mouseleave', untip);
      F.layer('oob').selectAll('circle').data(pts.filter((d) => d.c === 0)).join('circle')
        .attr('cx', (d) => F.x(d.x)).attr('cy', (d) => F.y(d.y)).attr('r', 3.6)
        .attr('fill', 'none').attr('stroke', (d) => cat(d.label)).attr('stroke-width', 1.6)
        .on('mousemove', (e, d) => tip(e, `class ${d.label}<br>out-of-bag: this tree never saw it`)).on('mouseleave', untip);
    }
    const sub = forests.rf.slice(offset, offset + 6);
    readout1.set({
      'trees shown': `${offset + 1}–${offset + 6} of ${N_TREES}`,
      'mean out-of-bag share (all 200 trees)': pct(C.rf.oobFrac),
      'mean leaves per tree': d3.mean(forests.rf, (t) => t.leaves).toFixed(1),
      'test accuracy of these single trees': sub.map((t, j) => (100 * (1 - C.rf.single[offset + j])).toFixed(0) + '%').join(' · '),
    });
  }
  paint1();

  // ================= Figure 2: average the votes =================
  const fig2 = figure(main, 'Add trees and watch the boundary smooth out',
    'Left: the forest\'s prediction is the <b>average vote</b> of its first <i>k</i> trees; colour strength shows how lopsided the vote is (pale = close to 50/50). The thick line is where the vote is exactly 50%. Thin grey lines are the boundaries of the individual trees (up to 15 shown). Right: error as trees are added. <b>Out-of-bag (OOB) error</b> scores each training point using only the trees that never saw it, so it estimates test error for free, with no held-out set. Hover a point for its votes.');
  let mode = 'rf', nT = 1, showTrees = true, stopAnim = null;
  select(fig2.controls, { label: 'Trees use', value: 'rf', options: [['rf', 'random forest (1 random feature per split)'], ['bag', 'bagging only (both features per split)']], onChange: (v) => { mode = v; prepGrid(); paint2(); } });
  const nS = slider(fig2.controls, { label: 'number of trees k', min: 1, max: N_TREES, step: 1, value: nT, onInput: (v) => { halt(); nT = v; paint2(); } });
  button(fig2.controls, 'Play: grow the forest', () => play(), true);
  checkbox(fig2.controls, { label: 'show individual tree boundaries', checked: true, onChange: (v) => { showTrees = v; paint2(); } });
  const readout2 = stats(fig2.root);
  const row = el('div', { class: 'fig-row' }); fig2.body.append(row);
  const left = el('div'), right = el('div'); row.append(left, right);
  legend(fig2.body, [['class 0', cat(0)], ['class 1', cat(1)], ['test error (600 new points)', cat(2)], ['out-of-bag error', cat(3)], ['average single tree, test error', muted()]]);

  const F = frame(left, { width: 520, height: 440, x: XD, y: YD, xLabel: 'x₁', yLabel: 'x₂', grid: false });
  const res = 4, gw = Math.ceil(F.w / res), gh = Math.ceil(F.h / res);
  let treeGrids = null, sumGrid = new Float32Array(gw * gh);
  // tree value at every canvas cell centre; computed once per mode
  const gridCache = {};
  function prepGrid() {
    if (!gridCache[mode]) {
      gridCache[mode] = forests[mode].map((t) => {
        const g = new Float32Array(gw * gh);
        for (let j = 0; j < gh; j++) { const py = F.y.invert(j * res + res / 2); for (let i = 0; i < gw; i++) g[j * gw + i] = predictTree(t.tree, F.x.invert(i * res + res / 2), py); }
        return g;
      });
    }
    treeGrids = gridCache[mode];
  }
  prepGrid();
  const canvas = decisionCanvas(F, () => 0.5, { res, alpha: 0.5 });
  const treeL = F.layer('trees'), meanL = F.layer('mean'), ptL = F.layer('pts');
  // contours on a grid padded by one replicated cell on each side, clipped to the plot, so no lines run along the border
  const proj = d3.geoTransform({ point(px, py) { this.stream.point((px - 1) * res, (py - 1) * res); } });
  const contourRaw = d3.contours().size([gw + 2, gh + 2]).thresholds([0.5]);
  const padded = new Float32Array((gw + 2) * (gh + 2));
  const contour = (g) => {
    for (let j = 0; j < gh + 2; j++) { const jj = Math.min(gh - 1, Math.max(0, j - 1)); for (let i = 0; i < gw + 2; i++) padded[j * (gw + 2) + i] = g[jj * gw + Math.min(gw - 1, Math.max(0, i - 1))]; }
    return contourRaw(padded);
  };
  F.svg.append('defs').append('clipPath').attr('id', 'rf-clip').append('rect').attr('width', F.w).attr('height', F.h);
  treeL.attr('clip-path', 'url(#rf-clip)'); meanL.attr('clip-path', 'url(#rf-clip)');

  const E = frame(right, { width: 500, height: 440, x: [1, N_TREES], y: [0, 0.3], xScale: d3.scaleLog(), xLabel: 'number of trees k (log scale)', yLabel: 'error rate' });
  E.g.select('.axis.x').call(d3.axisBottom(E.x).tickValues([1, 2, 5, 10, 20, 50, 100, 200]).tickFormat(d3.format('d')));
  E.g.select('.axis.y').call(d3.axisLeft(E.y).ticks(6).tickFormat(d3.format('.0%')));
  const eL = E.layer('curves'), eM = E.layer('marker');

  function paint2() {
    const grids = treeGrids, k = nT, cc = C[mode];
    sumGrid.fill(0);
    for (let t = 0; t < k; t++) { const g = grids[t]; for (let q = 0; q < g.length; q++) sumGrid[q] += g[q]; }
    for (let q = 0; q < sumGrid.length; q++) sumGrid[q] /= k;
    canvas.draw((x, y) => { const i = Math.min(gw - 1, Math.max(0, Math.floor(F.x(x) / res))), j = Math.min(gh - 1, Math.max(0, Math.floor(F.y(y) / res))); return 0.5 + 0.55 * (sumGrid[j * gw + i] - 0.5); }); // compressed so the colour fades gradually with vote strength
    const shown = showTrees ? Math.min(k, 15) : 0;
    treeL.selectAll('path').data(d3.range(shown).map((t) => contour(grids[t])[0])).join('path').attr('d', d3.geoPath(proj))
      .attr('fill', 'none').attr('stroke', muted()).attr('stroke-width', 1).attr('stroke-opacity', 0.75);
    meanL.selectAll('path').data([contour(sumGrid)[0]]).join('path').attr('d', d3.geoPath(proj))
      .attr('fill', 'none').attr('stroke', ink()).attr('stroke-width', 2.5);
    // per-point votes
    const pts = train.map((p, i) => {
      let s = 0, so = 0, no = 0;
      for (let t = 0; t < k; t++) { const v = cc.trP[t][i]; s += v; if (!forests[mode][t].counts[i]) { so += v; no++; } }
      return { ...p, vote: s / k, oob: no ? so / no : null, no };
    });
    ptL.selectAll('circle').data(pts).join('circle').attr('cx', (d) => F.x(d.x)).attr('cy', (d) => F.y(d.y)).attr('r', 4.2)
      .attr('fill', (d) => cat(d.label)).attr('stroke', css('--surface')).attr('stroke-width', 1.2)
      .on('mousemove', (e, d) => tip(e, `class ${d.label}<br>votes for class 1: ${pct(d.vote)} of ${k} tree${k > 1 ? 's' : ''}<br>${d.no ? `OOB vote (${d.no} trees that never saw it): ${pct(d.oob)}` : 'no OOB trees yet'}`))
      .on('mouseleave', untip);
    // error curves
    const ln = (key) => d3.line().defined((d) => isFinite(d[key])).x((d) => E.x(d.k)).y((d) => E.y(d[key]));
    eL.selectAll('path.te').data([cc.out]).join('path').attr('class', 'te').attr('fill', 'none').attr('stroke', cat(2)).attr('stroke-width', 2).attr('d', ln('test'));
    eL.selectAll('path.oob').data([cc.out]).join('path').attr('class', 'oob').attr('fill', 'none').attr('stroke', cat(3)).attr('stroke-width', 2).attr('d', ln('oob'));
    eL.selectAll('line.single').data([cc.meanSingle]).join('line').attr('class', 'single').attr('x1', 0).attr('x2', E.w).attr('y1', (d) => E.y(d)).attr('y2', (d) => E.y(d))
      .attr('stroke', muted()).attr('stroke-width', 1.5).attr('stroke-dasharray', '5 4');
    eL.selectAll('text.single').data([cc.meanSingle]).join('text').attr('class', 'single').attr('x', E.w - 4).attr('y', (d) => E.y(d) - 6).attr('text-anchor', 'end').attr('font-size', 12).attr('fill', ink2())
      .text((d) => `a typical single tree: ${pct(d)}`);
    const cur = cc.out[k - 1];
    eM.selectAll('line').data([k]).join('line').attr('x1', (d) => E.x(d)).attr('x2', (d) => E.x(d)).attr('y1', 0).attr('y2', E.h).attr('stroke', ink()).attr('stroke-width', 1.2).attr('stroke-dasharray', '3 3');
    eM.selectAll('circle').data([['test', cat(2)], ['oob', cat(3)]].filter(([key]) => isFinite(cur[key]))).join('circle')
      .attr('cx', E.x(k)).attr('cy', ([key]) => E.y(cur[key])).attr('r', 5).attr('fill', ([, c]) => c).attr('stroke', css('--surface')).attr('stroke-width', 1.5);
    readout2.set({
      'trees k': k,
      'forest test accuracy': pct(1 - cur.test),
      'OOB accuracy estimate': isFinite(cur.oob) ? pct(1 - cur.oob) : '—',
      'points with an OOB vote': `${cur.covered} / ${train.length}`,
      'average single-tree test accuracy': pct(1 - cc.meanSingle),
    });
  }
  function halt() { if (stopAnim) { stopAnim(); stopAnim = null; } }
  function play() {
    halt(); nT = 1; let acc = 0;
    stopAnim = animate((dt) => {
      acc += dt; const wait = nT < 15 ? 260 : nT < 40 ? 90 : 30;
      if (acc < wait) return true; acc = 0;
      nT = Math.min(N_TREES, nT + (nT < 40 ? 1 : 4)); nS.set(nT); paint2();
      return nT < N_TREES;
    });
  }
  paint2();

  // ---- numbers for the prose, computed from the seeded figure ----
  const rf = C.rf, bag = C.bag, last = (c) => c.out[N_TREES - 1];
  const at = (c, k) => c.out[k - 1];
  const singleRange = d3.extent(rf.single);
  section(main, 'What to notice', `<ul>
    <li>In the top figure, every tree is different, even though all were trained the same way on the same 160 points. On average ${pct(rf.oobFrac)} of the points are left out of each bootstrap sample (the theory says 1 − 1/e ≈ 36.8%). A single tree's test accuracy ranges from ${pct(1 - singleRange[1])} to ${pct(1 - singleRange[0])} across the 200 trees.</li>
    <li>Press <b>Play</b>. With 1 tree the boundary is a jagged staircase with islands around noisy points. By 10 trees the islands fade into pale, uncertain regions, and by 200 the 50% line is a much smoother curve through the gap between the two moons, with pale bands where the trees disagree. The small steps that remain are there because every tree can only cut horizontally or vertically. No single tree has this shape: it only exists in the average.</li>
    <li>Test error falls from ${pct(at(rf, 1).test)} with 1 tree to ${pct(at(rf, 10).test)} with 10 and ${pct(last(rf).test)} with 200, below the ${pct(rf.meanSingle)} of a typical single tree. By 20 trees it is already ${pct(at(rf, 20).test)}, so most of the gain comes early; adding trees never makes it overfit, it just stops helping.</li>
    <li>The OOB curve tracks the test curve: at 200 trees OOB accuracy is ${pct(1 - last(rf).oob)} against a true test accuracy of ${pct(1 - last(rf).test)}. With only a few trees, the OOB estimate is noisy because many points have few or no OOB votes.</li>
    <li>Switch to <b>bagging only</b>. Its trees always pick the best split from both features, so they are more alike (correlated) and their errors cancel less. Here that gives ${pct(last(bag).test)} test error at 200 trees, against ${pct(last(rf).test)} for the random forest. With only two features the effect is small; with 20,000 genes, feature subsampling matters much more.</li>
  </ul>`);

  steps(main, 'How it works', [
    '<b>Bootstrap</b>: draw n samples from the n training points with replacement. About 63% of points appear (some several times); the other ~37% are out-of-bag for this tree.',
    'Grow a deep decision tree on that sample. At each split, consider only a <b>random subset of features</b> (often √p for classification). This makes the trees less alike.',
    'Repeat for hundreds of trees. Each one is a low-bias, high-variance model: it fits its own sample, noise included.',
    'Predict by <b>averaging</b>: the mean of the trees\' class votes (or of their predicted values, for regression).',
    'For a free error estimate, predict each training point using only the trees for which it was out-of-bag, and score those predictions.',
  ]);
  math(main, 'forest(x) = (1/B) Σ_b tree_b(x)\nVar(average of B trees) = ρσ² + (1 − ρ)σ²/B      σ² = variance of one tree, ρ = correlation between trees\nmore trees shrinks the second term; feature subsampling lowers ρ, which shrinks the first');

  callout(main, '', 'In biology', 'Random forests are a default first model for omics classification: tumour subtype or response from expression, microbiome case/control status from taxa abundances, and cell-type calls in single-cell tools such as <i>SingleCellNet</i>. They need no feature scaling, handle many more genes than samples, and give OOB error and feature importances out of the box (scikit-learn <code>RandomForestClassifier</code>, R <code>ranger</code>). Random survival forests extend the idea to censored outcomes.');
  callout(main, 'warn', 'Pitfalls', `<ul>
    <li><b>OOB error is only honest if nothing upstream peeked at the labels.</b> If you selected the top 100 genes by a t-test on all samples first, the OOB (and CV) error will be optimistic. Put feature selection inside the resampling loop.</li>
    <li>Averaging reduces variance, not bias. If every tree misses the same pattern, so does the forest.</li>
    <li>Impurity-based feature importance is biased towards continuous and high-cardinality features, and correlated genes split the credit between them. Prefer permutation importance, and read it with care.</li>
    <li>Samples that are not independent (several biopsies per patient, cells from one donor) leak into the OOB estimate: a patient's other samples are in-bag. Use grouped CV instead.</li>
    <li>Like single trees, forests cannot extrapolate beyond the range of the training data.</li>
  </ul>`);
  seeAlso(main, [['concepts', 'decision-tree', 'Decision trees'], ['concepts', 'gradient-boosting', 'Gradient boosting'], ['concepts', 'bias-variance', 'Under- vs overfitting'], ['concepts', 'interpretability', 'Feature importance & SHAP'], ['omics', 'bulk-model-compare', 'Model comparison in the mock study']]);
}
