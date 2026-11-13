// Decision trees: greedy axis-aligned splits chosen to reduce Gini impurity, grown depth by depth.
import { header, figure, slider, select, stats, section, steps, callout, math, legend, seeAlso, tip, untip, el } from '../lib/ui.js';
import { frame, cat, css, ink, ink2, muted } from '../lib/plot.js';
import { rng } from '../lib/data.js';
const d3 = window.d3;

const FEAT = ['x₁', 'x₂'];
const RULE = (x, y) => ((x > 0.55 && y > 0.4) || (x < 0.3 && y < 0.35) ? 1 : 0);
function makeData(n, seed) {
  const r = rng(seed);
  return d3.range(n).map(() => { const x = r(), y = r(); let label = RULE(x, y); if (r() < 0.08) label = 1 - label; return { x, y, label, v: [x, y] }; });
}
const gini = (c0, c1) => { const n = c0 + c1; return n ? 1 - (c0 / n) ** 2 - (c1 / n) ** 2 : 0; };
const counts = (pts) => { const c = [0, 0]; pts.forEach((p) => c[p.label]++); return c; };

// For one feature, weighted child impurity at every candidate threshold (midpoints between sorted values).
function scan(pts, f) {
  const s = pts.slice().sort((a, b) => a.v[f] - b.v[f]), tot = counts(s), L = [0, 0], out = [];
  for (let i = 0; i < s.length - 1; i++) {
    L[s[i].label]++;
    if (s[i].v[f] === s[i + 1].v[f]) continue;
    const nl = i + 1, nr = s.length - nl, R = [tot[0] - L[0], tot[1] - L[1]];
    out.push({ thr: (s[i].v[f] + s[i + 1].v[f]) / 2, w: (nl * gini(L[0], L[1]) + nr * gini(R[0], R[1])) / s.length, gl: gini(L[0], L[1]), gr: gini(R[0], R[1]), nl, nr });
  }
  return out;
}
let NID = 0;
function grow(pts, depth, maxDepth, box) {
  const c = counts(pts), node = { id: NID++, pts, depth, c, g: gini(c[0], c[1]), box, pred: c[1] > c[0] ? 1 : 0 };
  if (depth >= maxDepth || node.g === 0 || pts.length < 4) return node;
  let best = null;
  [0, 1].forEach((f) => scan(pts, f).forEach((s) => { if (!best || s.w < best.w - 1e-12) best = { ...s, f }; }));
  if (!best || best.w >= node.g - 1e-9) return node;
  node.f = best.f; node.thr = best.thr; node.gain = node.g - best.w;
  const lb = box.slice(), rb = box.slice();
  if (best.f === 0) { lb[1] = best.thr; rb[0] = best.thr; } else { lb[3] = best.thr; rb[2] = best.thr; }
  node.left = grow(pts.filter((p) => p.v[best.f] <= best.thr), depth + 1, maxDepth, lb);
  node.right = grow(pts.filter((p) => p.v[best.f] > best.thr), depth + 1, maxDepth, rb);
  return node;
}
const isLeaf = (n, d) => !n.left || n.depth >= d;
function leaves(n, d, out = []) { if (isLeaf(n, d)) out.push(n); else { leaves(n.left, d, out); leaves(n.right, d, out); } return out; }
function predict(n, p, d) { while (!isLeaf(n, d)) n = p.v[n.f] <= n.thr ? n.left : n.right; return n.pred; }

export default async function render(main) {
  header(main, 'Decision trees',
    'Ask a sequence of yes/no questions of the form "is gene A above 0.55?". Each question is chosen <b>greedily</b> to split the samples into groups that are as pure as possible. The result is a flowchart, and on the data it carves space into axis-aligned boxes.');

  const train = makeData(180, 41), test = makeData(600, 42);
  NID = 0;
  const root = grow(train, 0, 8, [0, 1, 0, 1]);
  const depthOf = (n) => (n.left ? 1 + Math.max(depthOf(n.left), depthOf(n.right)) : 0);
  const fullDepth = depthOf(root);
  const accAt = (d, data) => d3.mean(data, (p) => predict(root, p, d) === p.label);

  // ---------------- Figure 1: grow the tree ----------------
  const fig = figure(main, 'Grow the tree one level at a time',
    `Left: 180 training samples; the true pattern is two orange corners, plus 8% of labels flipped at random (noise). Shaded boxes are the leaves, coloured by their majority class and faded when impure. Right: the same tree as a flowchart, read top-down; left branch = "yes". Each node shows its split, its Gini impurity and its sample count. <b>Hover</b> a box or <b>click</b> a node to highlight it; the clicked node is used in the split-search figure below.`);
  let depth = 2, focus = root;
  slider(fig.controls, { label: 'max depth', min: 0, max: fullDepth, step: 1, value: depth, onInput: (v) => { depth = v; if (focus.depth >= depth) focus = root; paint(); paint2(); } });
  const readout = stats(fig.root);
  const row = el('div', { class: 'fig-row' }); fig.body.append(row);
  const left = el('div', { style: 'flex: 0 1 430px' }), right = el('div'); row.append(left, right);
  legend(fig.body, [['class 0', cat(0)], ['class 1', cat(1)]]);

  const F = frame(left, { width: 430, height: 430, x: [0, 1], y: [0, 1], xLabel: 'x₁ (gene A)', yLabel: 'x₂ (gene B)', grid: false });
  const boxL = F.layer('boxes'), splitL = F.layer('splits'), ptL = F.layer('pts'), hiL = F.layer('hi');
  ptL.selectAll('circle').data(train).join('circle').attr('cx', (d) => F.x(d.x)).attr('cy', (d) => F.y(d.y)).attr('r', 4).attr('fill', (d) => cat(d.label))
    .attr('stroke', css('--surface')).attr('stroke-width', 1).style('pointer-events', 'none');
  const T = d3.select(right).append('svg').attr('viewBox', '0 0 600 430').attr('width', 600);
  const tLinks = T.append('g'), tNodes = T.append('g');

  const boxRect = (sel) => sel.attr('x', (d) => F.x(d.box[0])).attr('y', (d) => F.y(d.box[3])).attr('width', (d) => F.x(d.box[1]) - F.x(d.box[0])).attr('height', (d) => F.y(d.box[2]) - F.y(d.box[3]));
  function paint() {
    const lv = leaves(root, depth);
    boxL.selectAll('rect').data(lv, (d) => d.id).join('rect').call(boxRect).attr('fill', (d) => cat(d.pred)).attr('fill-opacity', (d) => 0.08 + 0.42 * (1 - 2 * d.g))
      .attr('stroke', css('--surface')).attr('stroke-width', 1)
      .on('mousemove', (e, d) => { hl(d); tip(e, `leaf: ${d.c[0]} blue, ${d.c[1]} orange<br>Gini = ${d.g.toFixed(3)} → predicts class ${d.pred}`); })
      .on('mouseleave', () => { hl(null); untip(); });
    const internal = []; (function walk(n) { if (!isLeaf(n, depth)) { internal.push(n); walk(n.left); walk(n.right); } })(root);
    splitL.selectAll('line').data(internal, (d) => d.id).join('line').attr('stroke', ink()).attr('stroke-width', (d) => Math.max(1.2, 3 - d.depth * 0.5))
      .attr('x1', (d) => F.x(d.f === 0 ? d.thr : d.box[0])).attr('x2', (d) => F.x(d.f === 0 ? d.thr : d.box[1]))
      .attr('y1', (d) => F.y(d.f === 1 ? d.thr : d.box[2])).attr('y2', (d) => F.y(d.f === 1 ? d.thr : d.box[3]));
    // tree diagram
    const H = d3.hierarchy(root, (n) => (isLeaf(n, depth) ? null : [n.left, n.right]));
    const nLeaves = H.leaves().length, W = 600, top = 30, Ht = 430 - top - 34, small = nLeaves > 5;
    d3.tree().size([W - (small ? 40 : 120), Math.max(1, Ht)]).separation(() => 1)(H);
    const dy = depth ? Ht / depth : 0;
    H.each((d) => { d.px = d.x + (small ? 20 : 60); d.py = top + d.depth * dy; });
    tLinks.selectAll('path').data(H.links()).join('path').attr('fill', 'none').attr('stroke', muted()).attr('stroke-width', 1.4)
      .attr('d', (l) => `M${l.source.px},${l.source.py} C${l.source.px},${(l.source.py + l.target.py) / 2} ${l.target.px},${(l.source.py + l.target.py) / 2} ${l.target.px},${l.target.py}`);
    const g = tNodes.selectAll('g.node').data(H.descendants(), (d) => d.data.id).join((en) => {
      const gg = en.append('g').attr('class', 'node').style('cursor', 'pointer');
      gg.append('rect'); gg.append('text').attr('class', 't1'); gg.append('text').attr('class', 't2'); gg.append('rect').attr('class', 'b0'); gg.append('rect').attr('class', 'b1');
      return gg;
    });
    const bw = small ? 34 : 112, bh = small ? 20 : 46;
    g.attr('transform', (d) => `translate(${d.px},${d.py})`)
      .on('click', (e, d) => { focus = d.data; paint(); paint2(); })
      .on('mousemove', (e, d) => { hl(d.data); tip(e, `${d.data.left && !isLeaf(d.data, depth) ? `split: ${FEAT[d.data.f]} ≤ ${d.data.thr.toFixed(2)}<br>` : 'leaf<br>'}n = ${d.data.pts.length} (${d.data.c[0]} blue, ${d.data.c[1]} orange)<br>Gini = ${d.data.g.toFixed(3)}`); })
      .on('mouseleave', () => { hl(null); untip(); });
    g.select('rect').attr('x', -bw / 2).attr('y', -bh / 2).attr('width', bw).attr('height', bh).attr('rx', 5)
      .attr('fill', css('--surface')).attr('stroke', (d) => (d.data === focus ? ink() : css('--axis'))).attr('stroke-width', (d) => (d.data === focus ? 2.2 : 1));
    g.select('.t1').attr('text-anchor', 'middle').attr('y', small ? 4 : -7).attr('font-size', small ? 11 : 13.5).attr('fill', ink()).attr('font-weight', 600)
      .text((d) => (small ? (isLeaf(d.data, depth) ? '' : `${FEAT[d.data.f]}`) : isLeaf(d.data, depth) ? `leaf → ${d.data.pred ? 'orange' : 'blue'}` : `${FEAT[d.data.f]} ≤ ${d.data.thr.toFixed(2)}`));
    g.select('.t2').attr('text-anchor', 'middle').attr('y', 9).attr('font-size', 12).attr('fill', ink2()).text((d) => (small ? '' : `G=${d.data.g.toFixed(2)} n=${d.data.pts.length}`));
    // class-mix bar under each node
    g.select('.b0').attr('x', -bw / 2 + 4).attr('y', bh / 2 - (small ? 5 : 9)).attr('height', small ? 3 : 5).attr('width', (d) => (bw - 8) * d.data.c[0] / d.data.pts.length).attr('fill', cat(0));
    g.select('.b1').attr('x', (d) => -bw / 2 + 4 + (bw - 8) * d.data.c[0] / d.data.pts.length).attr('y', bh / 2 - (small ? 5 : 9)).attr('height', small ? 3 : 5).attr('width', (d) => (bw - 8) * d.data.c[1] / d.data.pts.length).attr('fill', cat(1));
    hl(null);
    readout.set({ depth, leaves: lv.length, 'train accuracy': (100 * accAt(depth, train)).toFixed(1) + '%', 'test accuracy (600 new samples)': (100 * accAt(depth, test)).toFixed(1) + '%', 'mean leaf Gini (weighted)': d3.sum(lv, (n) => n.g * n.pts.length / train.length).toFixed(3) });
  }
  function hl(n) {
    const show = n || (focus !== root ? focus : null);
    hiL.selectAll('rect').data(show ? [show] : []).join('rect').call(boxRect).attr('fill', 'none').attr('stroke', ink()).attr('stroke-width', 2.5).attr('stroke-dasharray', '6 3').style('pointer-events', 'none');
  }

  // ---------------- Figure 2: best split search ----------------
  const fig2 = figure(main, 'How one split is chosen: scan every threshold',
    'For the node selected above (the root by default), try every possible threshold on each feature. For each, compute the Gini impurity of the two child groups, weighted by their sizes. The tree takes the lowest point over both features (the dotted line shows the best the other feature can do). Drag the threshold slider to see the child groups.');
  let f2 = 0, thrPos = 0.5;
  select(fig2.controls, { label: 'feature to split on', value: '0', options: [['0', 'x₁ (vertical cut)'], ['1', 'x₂ (horizontal cut)']], onChange: (v) => { f2 = +v; paint2(); } });
  const thrS = slider(fig2.controls, { label: 'threshold', min: 0, max: 1, step: 0.005, value: thrPos, fmt: (v) => v.toFixed(3), onInput: (v) => { thrPos = v; paint2(); } });
  button2(fig2.controls, 'jump to best', () => { const b = bestOf(focus); f2 = b.f; fig2.controls.querySelector('select').value = String(b.f); thrPos = b.thr; thrS.set(b.thr); paint2(); });
  const r2 = stats(fig2.root);
  const row2 = el('div', { class: 'fig-row' }); fig2.body.append(row2);
  const l2 = el('div', { style: 'flex: 0 1 400px' }), rr2 = el('div'); row2.append(l2, rr2);
  legend(fig2.body, [['weighted child Gini vs threshold (chosen feature)', cat(2)], ['best achievable on the other feature', cat(3)], ['parent Gini (no split)', muted()]]);
  const P = frame(l2, { width: 400, height: 380, x: [0, 1], y: [0, 1], xLabel: 'x₁', yLabel: 'x₂', grid: false });
  const pBox = P.layer('box'), pPts = P.layer('pts'), pLine = P.layer('line');
  const C = frame(rr2, { width: 520, height: 380, x: [0, 1], y: [0, 0.55], xLabel: 'threshold', yLabel: 'weighted child Gini impurity' });
  const cL = C.layer('c'), cM = C.layer('m');

  function bestOf(n) { let b = null; [0, 1].forEach((f) => scan(n.pts, f).forEach((s) => { if (!b || s.w < b.w - 1e-12) b = { ...s, f }; })); return b; }
  function button2(parent, label, fn) { const b = el('button', { class: 'btn' }, label); b.addEventListener('click', fn); parent.append(b); }
  function paint2() {
    const n = focus, b = bestOf(n);
    const sc = [0, 1].map((f) => scan(n.pts, f));
    C.y.domain([0, Math.max(0.1, n.g * 1.15)]); C.g.select('.axis.y').call(d3.axisLeft(C.y).ticks(5));
    C.x.domain(f2 === 0 ? [n.box[0], n.box[1]] : [n.box[2], n.box[3]]); C.g.select('.axis.x').call(d3.axisBottom(C.x).ticks(5));
    const curves = sc.map((s, f) => ({ f, s })).filter((d) => d.f === f2);
    const other = sc[1 - f2];
    cL.selectAll('path.curve').data(curves).join('path').attr('class', 'curve').attr('fill', 'none').attr('stroke', cat(2)).attr('stroke-width', 2)
      .attr('d', (d) => d3.line().curve(d3.curveStepAfter).x((q) => C.x(q.thr)).y((q) => C.y(q.w))(d.s));
    cL.selectAll('line.parent').data([n.g]).join('line').attr('class', 'parent').attr('x1', 0).attr('x2', C.w).attr('y1', (d) => C.y(d)).attr('y2', (d) => C.y(d)).attr('stroke', muted()).attr('stroke-dasharray', '5 4').attr('stroke-width', 1.5);
    const otherBest = other.length ? d3.min(other, (q) => q.w) : n.g;
    cL.selectAll('line.other').data([otherBest]).join('line').attr('class', 'other').attr('x1', 0).attr('x2', C.w).attr('y1', (d) => C.y(d)).attr('y2', (d) => C.y(d)).attr('stroke', cat(3)).attr('stroke-dasharray', '2 3').attr('stroke-width', 1.5);
    cL.selectAll('text.other').data([otherBest]).join('text').attr('class', 'other').attr('x', 6).attr('y', (d) => C.y(d) + 15).attr('text-anchor', 'start').attr('font-size', 11).attr('fill', ink2())
      .text(`best possible on ${FEAT[1 - f2]}: ${otherBest.toFixed(3)}`);
    const mine = curves[0].s.length ? curves[0].s.reduce((a, q) => (q.w < a.w ? q : a)) : null;
    cM.selectAll('path').data(mine ? [mine] : []).join('path').attr('d', d3.symbol().type(d3.symbolTriangle).size(90)()).attr('transform', (d) => `translate(${C.x(d.thr)},${C.y(d.w) + 9})`).attr('fill', ink());
    cM.selectAll('line').data([thrPos]).join('line').attr('x1', (d) => C.x(d)).attr('x2', (d) => C.x(d)).attr('y1', 0).attr('y2', C.h).attr('stroke', ink()).attr('stroke-width', 1.4);
    // node panel
    P.x.domain([0, 1]); P.y.domain([0, 1]);
    pBox.selectAll('rect').data([n]).join('rect').attr('x', P.x(n.box[0])).attr('y', P.y(n.box[3])).attr('width', P.x(n.box[1]) - P.x(n.box[0])).attr('height', P.y(n.box[2]) - P.y(n.box[3]))
      .attr('fill', css('--surface-2')).attr('stroke', ink()).attr('stroke-dasharray', '6 3');
    const side = (p) => p.v[f2] <= thrPos;
    pPts.selectAll('circle').data(train).join('circle').attr('cx', (d) => P.x(d.x)).attr('cy', (d) => P.y(d.y))
      .attr('r', (d) => (n.pts.includes(d) ? 4.2 : 2.5)).attr('fill', (d) => cat(d.label)).attr('fill-opacity', (d) => (n.pts.includes(d) ? 1 : 0.15)).attr('stroke', css('--surface')).attr('stroke-width', 1);
    const lo = f2 === 0 ? [n.box[2], n.box[3]] : [n.box[0], n.box[1]];
    pLine.selectAll('line').data([0]).join('line').attr('stroke', ink()).attr('stroke-width', 2.5)
      .attr('x1', f2 === 0 ? P.x(thrPos) : P.x(lo[0])).attr('x2', f2 === 0 ? P.x(thrPos) : P.x(lo[1]))
      .attr('y1', f2 === 1 ? P.y(thrPos) : P.y(lo[0])).attr('y2', f2 === 1 ? P.y(thrPos) : P.y(lo[1]));
    const L = counts(n.pts.filter(side)), R = counts(n.pts.filter((p) => !side(p))), nl = L[0] + L[1], nr = R[0] + R[1];
    const w = (nl * gini(...L) + nr * gini(...R)) / n.pts.length;
    r2.set({ 'parent Gini': n.g.toFixed(3), [`left child (${FEAT[f2]} ≤ t)`]: `${L[0]} blue / ${L[1]} orange · G=${gini(...L).toFixed(3)}`, 'right child': `${R[0]} blue / ${R[1]} orange · G=${gini(...R).toFixed(3)}`,
      'weighted child Gini': w.toFixed(3), 'impurity decrease': (n.g - w).toFixed(3), 'tree\'s choice': b ? `${FEAT[b.f]} ≤ ${b.thr.toFixed(3)}` : 'pure: no split' });
  }
  paint(); paint2();

  const bR = bestOf(root);
  const accs = d3.range(0, fullDepth + 1).map((d) => ({ d, tr: accAt(d, train), te: accAt(d, test) }));
  const bestD = accs.reduce((a, c) => (c.te > a.te ? c : a));
  const pct = (v) => (100 * v).toFixed(1) + '%';
  section(main, 'What to notice', `<ul>
    <li><b>Depth 0</b> is a single leaf that predicts the majority class for everyone (${pct(accs[0].te)} test accuracy). <b>Depth 1</b> is one cut: the root split ${FEAT[bR.f]} ≤ ${bR.thr.toFixed(2)} drops the Gini impurity from ${root.g.toFixed(3)} to ${bR.w.toFixed(3)}.</li>
    <li>Test accuracy is best at <b>depth ${bestD.d}</b> (${pct(bestD.te)}). Keep going to depth ${fullDepth} and training accuracy reaches ${pct(accs[fullDepth].tr)}, but test accuracy falls to ${pct(accs[fullDepth].te)}: the deep levels carve tiny boxes around individual noisy points. That is overfitting.</li>
    <li>Every boundary is a horizontal or vertical line. A diagonal pattern would need a long staircase of cuts.</li>
    <li>In the split search, the curve is jagged with several dips. The tree takes the single lowest one, with no look-ahead: a split that looks mediocre now but enables a great split later will never be chosen. That is what "greedy" means.</li>
    <li>Click a deep node: its points are few, so its impurity curve is coarse and the "best" threshold is decided by one or two samples.</li>
  </ul>`);

  steps(main, 'How it works', [
    'Start with all samples in one node. Its <b>Gini impurity</b> is 1 − Σ p<sub>k</sub>², the chance that two randomly drawn samples from the node have different labels (0 = pure).',
    'For every feature and every threshold, split the node in two and compute the size-weighted impurity of the children.',
    'Keep the split with the largest impurity decrease (the minimum of the curves above).',
    'Recurse into each child. Stop when a node is pure, too small, or at the maximum depth.',
    'To predict, drop a sample down the tree and report the majority class of the leaf it lands in (or the mean, for regression trees).',
  ]);
  math(main, 'Gini(node) = 1 − Σₖ pₖ²          (two classes: 2p(1−p), max 0.5)\nsplit score = (n_L/n)·Gini(L) + (n_R/n)·Gini(R)      choose the minimum\nalternative: entropy −Σ pₖ log₂ pₖ  (information gain)');

  callout(main, '', 'In biology', 'Single trees are the classic interpretable clinical model: a gating-style flowchart such as "CD8A high and PD-L1 positive → likely responder" can be printed on one page. Flow cytometry gating is itself a hand-built decision tree on two markers at a time, and automated gating tools learn similar trees. In practice, omics prediction uses ensembles of trees (<a href="#/concepts/random-forest">random forests</a>, <a href="#/concepts/gradient-boosting">gradient boosting</a>), because single deep trees are unstable.');
  callout(main, 'warn', 'Pitfalls', `<ul>
    <li><b>High variance.</b> Change a few samples and the root split can flip, changing the whole tree. Never over-read a single tree's structure.</li>
    <li>Unlimited depth memorises noise. Limit depth, require a minimum leaf size, or prune with cross-validation.</li>
    <li>Impurity-based feature importance favours features with many possible thresholds (continuous or high-cardinality features).</li>
    <li>Trees are invariant to monotone feature transforms (no scaling needed), but they cannot extrapolate: outside the training range they predict a constant.</li>
  </ul>`);
  seeAlso(main, [['concepts', 'random-forest', 'Random forests & bagging'], ['concepts', 'gradient-boosting', 'Gradient boosting'], ['concepts', 'bias-variance', 'Under- vs overfitting'], ['concepts', 'interpretability', 'Feature importance & SHAP']]);
}
