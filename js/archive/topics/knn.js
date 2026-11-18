// k-nearest neighbours: classify by majority vote of the k closest training points; k sets the smoothness.
import { header, figure, slider, button, stats, section, steps, callout, math, legend, seeAlso, tip, untip, el } from '../../lib/ui.js';
import { frame, decisionCanvas, cat, css, ink, ink2, muted } from '../../lib/plot.js';
import { moons } from '../../lib/data.js';
const d3 = window.d3;

// Indices of the k nearest points to (qx, qy). Simple partial insertion sort: fast enough for n = 200.
function nearest(train, qx, qy, k, skip = -1) {
  const best = []; // [d2, idx] sorted ascending, length ≤ k
  for (let i = 0; i < train.length; i++) {
    if (i === skip) continue;
    const p = train[i], d = (p.x - qx) ** 2 + (p.y - qy) ** 2;
    if (best.length < k || d < best[best.length - 1][0]) {
      let j = Math.min(best.length, k - 1);
      if (best.length < k) best.push(null);
      while (j > 0 && best[j - 1][0] > d) { best[j] = best[j - 1]; j--; }
      best[j] = [d, i];
    }
  }
  return best;
}
const vote = (train, nb) => nb.reduce((s, [, i]) => s + train[i].label, 0) / nb.length;

export default async function render(main) {
  header(main, 'k-nearest neighbours',
    'To classify a new point, find the <b>k training points closest to it</b> and take a majority vote. There is no training step at all: the data <i>is</i> the model. The single knob, k, decides how smooth the boundary is.');

  const train = moons(200, 0.22, 2), test = moons(400, 0.22, 77);
  const ks = d3.range(1, 100, 2);
  const curve = ks.map((k) => ({
    k,
    train: d3.mean(train, (p) => (vote(train, nearest(train, p.x, p.y, k)) > 0.5 ? 1 : 0) === p.label),
    test: d3.mean(test, (p) => (vote(train, nearest(train, p.x, p.y, k)) > 0.5 ? 1 : 0) === p.label),
  }));
  const best = curve.reduce((a, c) => (c.test > a.test ? c : a));
  const at = (k) => curve.find((c) => c.k === k);

  // ---------------- Figure ----------------
  const fig = figure(main, 'Drag k and watch the boundary smooth out',
    'Left: 200 training points from two interleaved "moons", shaded by the vote share of class 1 (orange) among the k nearest neighbours. <b>Click anywhere</b> to drop a query point and see its k neighbours and their vote. Right: accuracy on the training points and on 400 fresh test points from the same distribution, for every odd k.');
  let k = 15, q = { x: 0.15, y: 0.05 };
  slider(fig.controls, { label: 'k (neighbours)', min: 1, max: 99, step: 2, value: k, onInput: (v) => { k = v; paint(); } });
  button(fig.controls, 'k = 1', () => ksl.set(1));
  button(fig.controls, `best test k (${best.k})`, () => ksl.set(best.k));
  button(fig.controls, 'k = 99', () => ksl.set(99));
  const ksl = { set: (v) => { const s = fig.controls.querySelector('input[type=range]'); s.value = v; s.dispatchEvent(new Event('input')); } };
  const readout = stats(fig.root);

  const row = el('div', { class: 'fig-row' }); fig.body.append(row);
  const left = el('div'), right = el('div'); row.append(left, right);
  legend(fig.body, [['class 0', cat(0)], ['class 1', cat(1)], ['training accuracy', cat(2)], ['test accuracy', cat(6)]]);

  const F = frame(left, { width: 480, height: 440, x: [-1.75, 1.75], y: [-1.3, 1.3], xLabel: 'feature 1', yLabel: 'feature 2', grid: false });
  const surf = decisionCanvas(F, (x, y) => vote(train, nearest(train, x, y, k)), { res: 5, alpha: 0.4 });
  const linkL = F.layer('links'), ptL = F.layer('pts'), qL = F.layer('q');
  ptL.selectAll('circle').data(train).join('circle').attr('cx', (d) => F.x(d.x)).attr('cy', (d) => F.y(d.y)).attr('r', 4.2)
    .attr('fill', (d) => cat(d.label)).attr('stroke', css('--surface')).attr('stroke-width', 1.1);
  F.svg.style('cursor', 'crosshair').on('click', (e) => {
    const [mx, my] = d3.pointer(e, F.g.node());
    if (mx < 0 || my < 0 || mx > F.w || my > F.h) return;
    q = { x: F.x.invert(mx), y: F.y.invert(my) }; paintQuery();
  });

  const A = frame(right, { width: 480, height: 440, x: [0, 100], y: [0.7, 1.005], xLabel: 'k (left = flexible, right = smooth)', yLabel: 'accuracy' });
  const yfmt = d3.format('.0%');
  A.g.select('.axis.y').call(d3.axisLeft(A.y).ticks(5).tickFormat(yfmt));
  [['train', cat(2)], ['test', cat(6)]].forEach(([key, c]) => {
    A.layer('c').append('path').datum(curve).attr('fill', 'none').attr('stroke', c).attr('stroke-width', 2).attr('d', d3.line().x((d) => A.x(d.k)).y((d) => A.y(d[key])));
  });
  A.layer('dots').selectAll('circle').data(curve.flatMap((d) => [{ ...d, key: 'train' }, { ...d, key: 'test' }])).join('circle')
    .attr('cx', (d) => A.x(d.k)).attr('cy', (d) => A.y(d[d.key])).attr('r', 6).attr('fill', 'transparent')
    .on('mousemove', (e, d) => tip(e, `k = ${d.k}<br>train ${(100 * d.train).toFixed(1)}% · test ${(100 * d.test).toFixed(1)}%`)).on('mouseleave', untip);
  A.layer('lab').append('text').attr('x', A.x(best.k) + 6).attr('y', A.y(0.72)).attr('font-size', 11).attr('fill', ink2()).text(`best test accuracy at k = ${best.k}`);
  A.layer('lab').append('line').attr('x1', A.x(best.k)).attr('x2', A.x(best.k)).attr('y1', A.y(0.7)).attr('y2', A.y(best.test)).attr('stroke', muted()).attr('stroke-dasharray', '3 3');
  const cur = A.layer('cur');

  function paintQuery() {
    const nb = nearest(train, q.x, q.y, k), v = vote(train, nb), r = Math.sqrt(nb[nb.length - 1][0]);
    linkL.selectAll('line').data(nb).join('line').attr('x1', F.x(q.x)).attr('y1', F.y(q.y)).attr('x2', ([, i]) => F.x(train[i].x)).attr('y2', ([, i]) => F.y(train[i].y))
      .attr('stroke', ([, i]) => cat(train[i].label)).attr('stroke-width', 1.4).attr('stroke-opacity', 0.9);
    qL.selectAll('ellipse').data([0]).join('ellipse').attr('cx', F.x(q.x)).attr('cy', F.y(q.y)).attr('rx', F.x(r) - F.x(0)).attr('ry', F.y(0) - F.y(r))
      .attr('fill', 'none').attr('stroke', ink()).attr('stroke-width', 1.2).attr('stroke-dasharray', '4 3');
    ptL.selectAll('circle').attr('r', (d, i) => (nb.some(([, j]) => j === i) ? 6 : 4.2)).attr('stroke', (d, i) => (nb.some(([, j]) => j === i) ? ink() : css('--surface')));
    qL.selectAll('path').data([0]).join('path').attr('d', d3.symbol().type(d3.symbolStar).size(220)()).attr('transform', `translate(${F.x(q.x)},${F.y(q.y)})`)
      .attr('fill', v > 0.5 ? cat(1) : cat(0)).attr('stroke', ink()).attr('stroke-width', 1.5);
    const n1 = Math.round(v * nb.length);
    readout.set({ k, 'query vote (class 0 : class 1)': `${nb.length - n1} : ${n1}`, 'predicted class': v > 0.5 ? '1 (orange)' : '0 (blue)', 'train accuracy': (100 * at(k).train).toFixed(1) + '%', 'test accuracy': (100 * at(k).test).toFixed(1) + '%' });
  }
  function paint() {
    surf.draw();
    const c = at(k);
    cur.selectAll('line').data([0]).join('line').attr('x1', A.x(k)).attr('x2', A.x(k)).attr('y1', 0).attr('y2', A.h).attr('stroke', ink()).attr('stroke-width', 1.2);
    cur.selectAll('circle').data([['train', cat(2)], ['test', cat(6)]]).join('circle').attr('cx', A.x(k)).attr('cy', ([key]) => A.y(c[key])).attr('r', 5.5)
      .attr('fill', ([, col]) => col).attr('stroke', css('--surface')).attr('stroke-width', 1.5);
    paintQuery();
  }
  paint();

  const pct = (v) => (100 * v).toFixed(1) + '%';
  section(main, 'What to notice', `<ul>
    <li><b>k = 1</b>: the boundary wraps around every single point, with little islands of the wrong colour. Training accuracy is ${pct(at(1).train)} (each point is its own nearest neighbour), but test accuracy is only ${pct(at(1).test)}. That gap is <a href="#/archive/topics/bias-variance">overfitting</a>.</li>
    <li>As k grows the islands vanish and the boundary follows the two moons. Test accuracy peaks at ${pct(best.test)} around <b>k = ${best.k}</b>.</li>
    <li><b>k = 99</b> (half the training set votes): the boundary flattens towards a straight-ish line, the moons' tips get misclassified, and test accuracy falls to ${pct(at(99).test)}. That is underfitting.</li>
    <li>Read the right panel from left to right. At k = 1 the gap between training and test accuracy is ${(100 * (at(1).train - at(1).test)).toFixed(1)} points; by k ≈ ${best.k} it has closed, and for large k both curves fall together. A big train–test gap means variance (overfitting); both curves low means bias (underfitting).</li>
    <li>Click near the tip of a moon with a large k: the circle has to reach into the other moon to collect k points, and the vote flips.</li>
  </ul>`);

  steps(main, 'How it works', [
    '<b>"Training"</b> is just storing the labelled points.',
    'For a new point, compute its distance to every stored point (usually Euclidean).',
    'Take the k closest: its <b>neighbours</b>.',
    'Predict the majority class among them (or, for regression, their mean). The vote share is a crude probability, which is the shading here.',
    'Choose k by <a href="#/archive/topics/cross-validation">cross-validation</a>, never by training accuracy, which always favours k = 1.',
  ]);
  math(main, 'ŷ(x) = majority{ yᵢ : i ∈ N_k(x) }        P̂(y=1 | x) = (1/k) Σ_{i∈N_k(x)} yᵢ\nN_k(x) = the k training points with smallest ‖x − xᵢ‖');

  callout(main, '', 'In biology', 'Neighbour graphs are everywhere in single-cell analysis. Scanpy and Seurat build a <b>kNN graph</b> on PCA coordinates (<code>sc.pp.neighbors</code>, <code>FindNeighbors</code>) before Leiden clustering and UMAP. Label transfer from a reference atlas (Seurat <code>TransferData</code>, scArches + kNN, Symphony) is literally kNN classification of query cells in a shared embedding. <b>scrublet</b> and <b>DoubletFinder</b> flag doublets by how many artificial doublets appear among a cell\'s neighbours, and MAGIC and kNN-smoothing impute expression by averaging neighbours.');
  callout(main, 'warn', 'Pitfalls', `<ul>
    <li><b>Scale matters.</b> Distance mixes all features, so an unscaled feature with a large range dominates. <a href="#/archive/topics/scaling">Standardise first.</a></li>
    <li>In high dimensions all distances look alike (<a href="#/archive/topics/curse-dim">curse of dimensionality</a>). Use kNN on a reduced space (top PCs), not on 20,000 raw genes.</li>
    <li>Imbalanced classes: with a large k the majority class wins most votes. Distance-weighted votes help.</li>
    <li>Prediction is slow for big training sets (every query scans the data); real tools use approximate search (Annoy, HNSW, pynndescent).</li>
  </ul>`);
  seeAlso(main, [['concepts', 'bias-variance', 'Under- vs overfitting'], ['concepts', 'scaling', 'Feature scaling'], ['concepts', 'curse-dim', 'Curse of dimensionality'], ['concepts', 'graph-clustering', 'kNN graphs & Leiden'], ['omics', 'sc-annotation', 'Cell-type annotation']]);
}
