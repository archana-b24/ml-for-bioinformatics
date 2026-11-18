// Feature scaling: the same two-feature dataset, raw vs standardised, seen through kNN distances and gradient descent.
import { header, figure, slider, button, checkbox, stats, section, steps, callout, math, seeAlso, animate, legend, tip, untip } from '../../lib/ui.js';
import { frame, decisionCanvas, cat, css, ink, ink2 } from '../../lib/plot.js';
import { rng, mean, sd, solve } from '../../lib/data.js';
const d3 = window.d3;

// ---- data: gene A (raw counts, large spread, uninformative) and gene B (log expression, small spread, informative) ----
function makeData(n = 160, seed = 7) {
  const r = rng(seed), out = [];
  for (let i = 0; i < n; i++) {
    const label = i % 2;
    out.push({ a: Math.max(2, r.normal(40, 12)), b: r.normal(label ? 3 : 1, 0.6), label });
  }
  return out;
}

/** k nearest indices to (qa, qb) under a feature-wise scale (sa, sb). Skips index `skip`. */
function knn(data, qa, qb, k, sa, sb, skip = -1) {
  const best = []; // [dist2, idx] sorted ascending, length <= k
  for (let i = 0; i < data.length; i++) {
    if (i === skip) continue;
    const da = (data[i].a - qa) / sa, db = (data[i].b - qb) / sb, d2 = da * da + db * db;
    if (best.length < k || d2 < best[best.length - 1][0]) {
      let j = Math.min(best.length, k - 1);
      best[j] = [d2, i];
      while (j > 0 && best[j - 1][0] > best[j][0]) { [best[j - 1], best[j]] = [best[j], best[j - 1]]; j--; }
    }
  }
  return best;
}

export default async function render(main) {
  header(main, 'Feature scaling',
    'Distances and gradients treat every feature\'s units as if they were comparable. When one feature is measured in thousands and another in fractions, <b>the big-number feature silently dominates</b>. Standardising (subtract the mean, divide by the standard deviation) puts them on equal footing.');

  const data = makeData();
  const A = data.map((d) => d.a), B = data.map((d) => d.b);
  const mA = mean(A), mB = mean(B), sA = sd(A), sB = sd(B);
  const yMean = mean(data.map((d) => d.label));

  const fig = figure(main, 'The same cells, raw vs standardised',
    'Left: k-nearest-neighbour classification. The shading is the kNN vote; the ring is the query point (click to move it), lines go to its k neighbours, and the outlined region is every point at least as close as the k-th neighbour. Right: least-squares fit of the label on both features. Contours are equal loss in weight space (equal axes); the trail is gradient descent at 90% of its largest stable learning rate.');
  let scaled = false, k = 7, query = { a: 30, b: 2.6 }, gdStop = null;

  checkbox(fig.controls, { label: 'Standardise features (z-score)', checked: false, onChange: (c) => { scaled = c; drawAll(); } });
  slider(fig.controls, { label: 'k neighbours', min: 1, max: 25, step: 1, value: k, onInput: (v) => { k = v; drawKnn(); } });
  button(fig.controls, 'Replay descent', () => runGD(), true);
  legend(fig.root, [['resting cell (class 0)', cat(0)], ['activated cell (class 1)', cat(1)], ['gradient-descent path', ink()]]);
  const readout = stats(fig.root);

  const row = d3.select(fig.body).append('div').attr('class', 'fig-row').node();
  const left = document.createElement('div'), right = document.createElement('div');
  row.append(left, right);

  // current scale factors for distances / features
  const fs = () => (scaled ? [sA, sB] : [1, 1]);
  const toView = (d) => (scaled ? [(d.a - mA) / sA, (d.b - mB) / sB] : [d.a, d.b]);
  const fromView = (u, v) => (scaled ? { a: u * sA + mA, b: v * sB + mB } : { a: u, b: v });

  function looAcc(sa, sb, kk) {
    let ok = 0;
    data.forEach((d, i) => {
      const nb = knn(data, d.a, d.b, kk, sa, sb, i);
      const v = nb.reduce((s, [, j]) => s + data[j].label, 0) / nb.length;
      const pred = v > 0.5 ? 1 : v < 0.5 ? 0 : data[nb[0][1]].label; // even-k ties: nearest neighbour decides
      if (pred === d.label) ok++;
    });
    return ok / data.length;
  }

  // ---------- kNN panel ----------
  let K;
  function drawKnn() {
    left.innerHTML = '';
    const xr = scaled ? [-3.2, 3.2] : [0, 90], yr = scaled ? [-3.2, 3.2] : [-1, 5.4];
    K = frame(left, { width: 460, height: 450, x: xr, y: yr,
      xLabel: scaled ? 'gene A (z-score)' : 'gene A (raw counts)', yLabel: scaled ? 'gene B (z-score)' : 'gene B (log expression)' });
    const [sa, sb] = fs();
    decisionCanvas(K, (u, v) => { const p = fromView(u, v); const nb = knn(data, p.a, p.b, k, sa, sb); return nb.reduce((s, [, j]) => s + data[j].label, 0) / nb.length; }, { res: 4, alpha: 0.3 });
    const clipId = 'scaling-clip-' + Math.random().toString(36).slice(2, 8);
    K.svg.append('defs').append('clipPath').attr('id', clipId).append('rect').attr('width', K.w).attr('height', K.h);
    const nb = knn(data, query.a, query.b, k, sa, sb);
    const rK = Math.sqrt(nb[nb.length - 1][0]); // radius in (scaled) distance units
    const [qu, qv] = toView(query);
    // neighbourhood ball: radius rK in distance units -> rK*sa in feature-A units, rK*sb in feature-B units
    const ru = scaled ? rK : rK * sa, rv = scaled ? rK : rK * sb;
    const ball = K.g.append('g').attr('clip-path', `url(#${clipId})`);
    ball.append('ellipse').attr('cx', K.x(qu)).attr('cy', K.y(qv))
      .attr('rx', Math.abs(K.x(qu + ru) - K.x(qu))).attr('ry', Math.abs(K.y(qv + rv) - K.y(qv)))
      .attr('fill', css('--accent-soft')).attr('fill-opacity', 0.35).attr('stroke', ink()).attr('stroke-width', 1.5).attr('stroke-dasharray', '5 4');
    K.layer('links').selectAll('line').data(nb).join('line')
      .attr('x1', K.x(qu)).attr('y1', K.y(qv)).attr('x2', ([, j]) => K.x(toView(data[j])[0])).attr('y2', ([, j]) => K.y(toView(data[j])[1]))
      .attr('stroke', ink()).attr('stroke-width', 1.2).attr('opacity', 0.7);
    const nbSet = new Set(nb.map(([, j]) => j));
    K.layer('pts').selectAll('circle').data(data).join('circle')
      .attr('cx', (d) => K.x(toView(d)[0])).attr('cy', (d) => K.y(toView(d)[1]))
      .attr('r', (d, i) => (nbSet.has(i) ? 6 : 4)).attr('fill', (d) => cat(d.label))
      .attr('stroke', (d, i) => (nbSet.has(i) ? ink() : css('--surface'))).attr('stroke-width', (d, i) => (nbSet.has(i) ? 1.8 : 1.2))
      .on('mousemove', (e, d) => tip(e, `<b>class ${d.label}</b><br>gene A ${d.a.toFixed(1)} counts<br>gene B ${d.b.toFixed(2)}`))
      .on('mouseleave', untip);
    K.layer('q').append('circle').attr('cx', K.x(qu)).attr('cy', K.y(qv)).attr('r', 8)
      .attr('fill', 'none').attr('stroke', ink()).attr('stroke-width', 2.5);
    K.svg.style('cursor', 'crosshair').on('click', (e) => {
      const [mx, my] = d3.pointer(e, K.g.node());
      if (mx < 0 || my < 0 || mx > K.w || my > K.h) return;
      query = fromView(K.x.invert(mx), K.y.invert(my));
      drawKnn();
    });
    const votes = nb.reduce((s, [, j]) => s + data[j].label, 0);
    accCache[k] = accCache[k] || [looAcc(1, 1, k), looAcc(sA, sB, k)];
    setStats({ 'query: neighbours that are activated': `${votes} / ${nb.length}`,
      'kNN accuracy, raw (leave-one-out)': (100 * accCache[k][0]).toFixed(0) + '%',
      'kNN accuracy, standardised': (100 * accCache[k][1]).toFixed(0) + '%' });
  }
  const accCache = {};

  // ---------- gradient-descent panel ----------
  // Least squares of centred label on centred features (raw) or z-scored features. Loss = mean (y - Xw)^2.
  function quad(isScaled) {
    const X = data.map((d) => isScaled ? [(d.a - mA) / sA, (d.b - mB) / sB] : [d.a - mA, d.b - mB]);
    const y = data.map((d) => d.label - yMean), n = X.length;
    const H = [[0, 0], [0, 0]], g = [0, 0];
    X.forEach((x, i) => { for (let p = 0; p < 2; p++) { g[p] += x[p] * y[i] / n; for (let q = 0; q < 2; q++) H[p][q] += x[p] * x[q] / n; } });
    const c = y.reduce((s, v) => s + v * v, 0) / n;
    const f = (w) => c - 2 * (g[0] * w[0] + g[1] * w[1]) + (H[0][0] * w[0] * w[0] + 2 * H[0][1] * w[0] * w[1] + H[1][1] * w[1] * w[1]);
    const grad = (w) => [2 * (H[0][0] * w[0] + H[0][1] * w[1] - g[0]), 2 * (H[1][0] * w[0] + H[1][1] * w[1] - g[1])];
    const tr = H[0][0] + H[1][1], det = H[0][0] * H[1][1] - H[0][1] ** 2, disc = Math.sqrt(tr * tr / 4 - det);
    const lmax = 2 * (tr / 2 + disc), lmin = 2 * (tr / 2 - disc); // Hessian of the loss = 2H
    const wStar = solve(H, g);
    return { f, grad, lmax, lmin, wStar, fStar: f(wStar), lr: 0.9 * 2 / lmax };
  }
  const QR = quad(false), QS = quad(true);
  const START = [0.5, -0.5], DOM = [-0.75, 0.95], MAXIT = 400;
  function descend(Q) {
    const path = [START.slice()];
    let w = START.slice(), conv = null;
    for (let t = 1; t <= MAXIT; t++) {
      const g = Q.grad(w); w = [w[0] - Q.lr * g[0], w[1] - Q.lr * g[1]]; path.push(w);
      if (conv === null && Math.hypot(w[0] - Q.wStar[0], w[1] - Q.wStar[1]) < 0.01) conv = t;
    }
    return { path, conv };
  }
  const runR = descend(QR), runS = descend(QS);

  let G;
  function drawGD() {
    right.innerHTML = '';
    const Q = scaled ? QS : QR;
    G = frame(right, { width: 460, height: 450, x: DOM, y: DOM, grid: false,
      xLabel: scaled ? 'weight on gene A (per SD)' : 'weight on gene A (per count)', yLabel: scaled ? 'weight on gene B (per SD)' : 'weight on gene B (per log unit)' });
    const [lo, hi] = DOM, n = 90, vals = new Array(n * n);
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) vals[j * n + i] = Q.f([lo + (hi - lo) * i / (n - 1), hi - (hi - lo) * j / (n - 1)]) - Q.fStar;
    // levels evenly spaced in sqrt(loss), i.e. in distance from the optimum: same picture density for both versions
    const top = Q.f([0, 0]) - Q.fStar;
    const levels = d3.range(1, 15).map((m) => top * (m / 5) ** 2);
    const cs = d3.contours().size([n, n]).thresholds(levels)(vals);
    const shade = d3.scaleLinear().domain([0, levels.length]).range([css('--accent'), css('--surface-2')]);
    const proj = d3.geoTransform({ point(px, py) { this.stream.point(G.x(lo + (hi - lo) * px / (n - 1)), G.y(hi - (hi - lo) * py / (n - 1))); } });
    G.layer('c').selectAll('path').data(cs).join('path').attr('d', d3.geoPath(proj))
      .attr('fill', (d, i) => shade(i)).attr('stroke', css('--surface')).attr('stroke-width', 0.6);
    G.layer('star').append('text').attr('x', G.x(Q.wStar[0])).attr('y', G.y(Q.wStar[1]) + 5).attr('text-anchor', 'middle')
      .attr('font-size', 16).attr('font-weight', 700).attr('fill', ink()).text('×');
    G.layer('path'); G.layer('dots');
    G.g.append('text').attr('x', 6).attr('y', 14).attr('font-size', 12).attr('fill', ink2())
      .text(`η = ${Q.lr.toPrecision(2)}   ·   bowl stretch (condition number) = ${(Q.lmax / Q.lmin).toFixed(0)}`);
    runGD();
  }
  function runGD() {
    if (gdStop) gdStop();
    const run = scaled ? runS : runR;
    let t = 0, acc = 0;
    const paintGD = () => {
      const pts = run.path.slice(0, t + 1);
      G.layer('path').selectAll('path').data([pts]).join('path').attr('fill', 'none').attr('stroke', ink()).attr('stroke-width', 1.5)
        .attr('d', d3.line().x((p) => G.x(p[0])).y((p) => G.y(p[1])));
      G.layer('dots').selectAll('circle').data(pts).join('circle').attr('cx', (p) => G.x(p[0])).attr('cy', (p) => G.y(p[1]))
        .attr('r', (p, i) => (i === pts.length - 1 ? 6 : 2.5)).attr('fill', (p, i) => (i === pts.length - 1 ? cat(3) : ink()))
        .attr('stroke', css('--surface')).attr('stroke-width', 1);
      setStats({ 'GD step': t, 'GD steps to converge, raw': runR.conv ?? `>${MAXIT}`, 'GD steps to converge, standardised': runS.conv ?? `>${MAXIT}` });
    };
    paintGD();
    gdStop = animate((dt) => { acc += dt; if (acc < 12) return true; acc = 0; t++; paintGD(); return t < MAXIT && !(run.conv !== null && t > run.conv + 3); });
  }

  let cur = {};
  function setStats(o) { cur = Object.assign(cur, o); readout.set(cur); }
  function drawAll() { drawKnn(); drawGD(); }
  drawAll();

  const accK = accCache[7];
  section(main, 'What to notice', `<ul>
    <li><b>Raw</b>: the neighbourhood is a tall, thin sliver. Gene A spans tens of counts while gene B spans a few log units, so "nearest" just means "most similar gene A count". The kNN shading forms vertical stripes, even though gene A carries no information about the class. Leave-one-out accuracy at k = 7 is only ${(100 * accK[0]).toFixed(0)}%.</li>
    <li><b>Tick "Standardise"</b>: the neighbourhood becomes a disc, the boundary turns horizontal (gene B is what matters), and accuracy jumps to ${(100 * accK[1]).toFixed(0)}%. Nothing about the data changed except the units.</li>
    <li><b>Gradient descent</b>: raw, the loss bowl is stretched by a factor of about ${(QR.lmax / QR.lmin).toFixed(0)}. The steep direction forces a tiny learning rate, so the trail zig-zags and needs ${runR.conv ?? '>' + MAXIT} steps. Standardised, the bowl is nearly round and the same rule converges in ${runS.conv} steps.</li>
    <li>Move the query point around in raw mode: its neighbours come from both classes whenever gene A is similar, wherever gene B is.</li>
  </ul>`);

  steps(main, 'How it works', [
    'For each feature j, compute its mean μⱼ and standard deviation σⱼ <b>on the training data only</b>.',
    'Transform every value: <code>z = (x − μⱼ) / σⱼ</code>. Each feature now has mean 0 and SD 1, so a one-unit difference means "one typical spread" for every feature.',
    'Apply the same μ and σ (from training) to validation, test and future data.',
    'Methods that care: anything using distances (kNN, k-means, SVM with RBF kernel, UMAP, PCA), and anything trained by gradient descent or penalised by size (lasso, ridge, neural networks). Methods that do not: decision trees, random forests and boosting split one feature at a time, so a monotone rescaling changes nothing.',
  ]);
  math(main, 'z = (x − μ) / σ\nloss Hessian for least squares = (2/n) XᵀX ;  GD needs η < 2 / λ_max, and its speed is set by λ_min / λ_max');

  callout(main, '', 'In biology', 'In scRNA-seq the standard pipeline (Scanpy / Seurat) log-normalises counts and then <b>scales each highly variable gene to unit variance</b> before PCA. Without it, a few very highly expressed genes (ribosomal, mitochondrial, haemoglobin) dominate the principal components and every downstream kNN graph. Scaling is usually clipped (e.g. <code>sc.pp.scale(max_value=10)</code>) so that rare, extreme genes do not take over in the other direction. In bulk biomarker work, combining a lab value in mg/L with log-CPM gene expression in one lasso or kNN model without standardising lets the lab value decide everything.');
  callout(main, 'warn', 'Pitfalls', '<ul><li><b>Leakage</b>: computing μ and σ on the whole dataset, test set included, before splitting. Fit the scaler inside each cross-validation fold.</li><li>Standardising a feature that is pure noise <b>promotes</b> it to equal weight. Unit variance is a choice: it assumes every feature matters equally a priori. Filtering uninformative features (e.g. selecting HVGs) still matters.</li><li>Outliers inflate σ. Robust scaling (median and IQR) or a log transform may be better for count-like data.</li><li>Sparse count matrices become dense once you subtract the mean. Scanpy keeps them sparse by scaling without centring when asked (<code>zero_center=False</code>).</li></ul>');
  seeAlso(main, [['concepts', 'knn', 'k-nearest neighbours'], ['concepts', 'gradient-descent', 'Gradient descent'], ['concepts', 'pca', 'PCA'], ['concepts', 'curse-dim', 'Curse of dimensionality']]);
}
