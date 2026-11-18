// Support vector machines: soft-margin SVM solved in the dual with SMO (maximal-violating-pair working set,
// as in LIBSVM's WSS1). Margin + support vectors, C (soft margin), linear vs RBF kernel, KKT view, and a 1-D lift.
import { header, figure, slider, select, button, stats, section, steps, callout, math, legend, seeAlso, animate, tip, untip, el } from '../../lib/ui.js';
import { frame, decisionCanvas, cat, css, ink, ink2, muted } from '../../lib/plot.js';
import { rng, blobs, moons, circles, sigmoid } from '../../lib/data.js';
const d3 = window.d3;

// ---------- kernels ----------
const kLinear = (a, b) => a[0] * b[0] + a[1] * b[1];
const kRBF = (g) => (a, b) => Math.exp(-g * ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2));

/**
 * Soft-margin SVM dual:  min ½ αᵀQα − Σα   s.t. 0 ≤ α ≤ C, Σ yα = 0,  Q_ij = y_i y_j K_ij.
 * SMO: each iteration picks the maximal violating pair (i, j), solves the 2-variable problem in closed form,
 * and updates the gradient G = Qα − 1 in O(n). Stops when the KKT gap m − M < eps.
 */
function smo(X, y, kern, C, { eps = 1e-3, maxIter = 40000 } = {}) {
  const n = X.length, K = new Float64Array(n * n);
  for (let i = 0; i < n; i++) for (let j = i; j < n; j++) { const v = kern(X[i], X[j]); K[i * n + j] = v; K[j * n + i] = v; }
  const a = new Float64Array(n), G = new Float64Array(n).fill(-1);
  let it = 0, mUp = 0, mLow = 0;
  for (; it < maxIter; it++) {
    let i = -1, j = -1, gmax = -Infinity, gmin = Infinity;
    for (let t = 0; t < n; t++) {
      const v = -y[t] * G[t];
      const up = (y[t] > 0 && a[t] < C) || (y[t] < 0 && a[t] > 0);
      const low = (y[t] > 0 && a[t] > 0) || (y[t] < 0 && a[t] < C);
      if (up && v > gmax) { gmax = v; i = t; }
      if (low && v < gmin) { gmin = v; j = t; }
    }
    mUp = gmax; mLow = gmin;
    if (i < 0 || j < 0 || gmax - gmin < eps) break;
    // move α_i += y_i·s, α_j −= y_j·s  (keeps Σ yα fixed); objective is quadratic in s
    const eta = Math.max(K[i * n + i] + K[j * n + j] - 2 * K[i * n + j], 1e-12);
    let s = (gmax - gmin) / eta;
    s = Math.min(s, y[i] > 0 ? C - a[i] : a[i], y[j] > 0 ? a[j] : C - a[j]);
    a[i] += y[i] * s; a[j] -= y[j] * s;
    for (let t = 0; t < n; t++) G[t] += y[t] * s * (K[t * n + i] - K[t * n + j]);
  }
  // bias: average over free support vectors (0 < α < C); otherwise the midpoint of the feasible range
  let bs = 0, nf = 0;
  for (let t = 0; t < n; t++) if (a[t] > 1e-8 && a[t] < C - 1e-8) { bs += -y[t] * G[t]; nf++; }
  const b = nf ? bs / nf : (mUp + mLow) / 2;
  const sv = []; for (let t = 0; t < n; t++) if (a[t] > 1e-8) sv.push(t);
  const f = (p) => { let s = b; for (const t of sv) s += a[t] * y[t] * kern(X[t], p); return s; };
  const fTrain = Array.from({ length: n }, (_, t) => { let s = b; for (const q of sv) s += a[q] * y[q] * K[q * n + t]; return s; });
  let w2 = 0; for (const p of sv) for (const q of sv) w2 += a[p] * a[q] * y[p] * y[q] * K[p * n + q];
  return { a, b, f, fTrain, sv, it, capped: it >= maxIter, margin: 2 / Math.sqrt(Math.max(w2, 1e-12)), C };
}

const DATA = {
  blobs: { name: 'two overlapping blobs', make: () => blobs(120, [[-0.42, -0.32], [0.42, 0.32]], 0.27, 7) },
  moons: { name: 'two moons', make: () => moons(120, 0.13, 8) },
  circles: { name: 'ring around a core', make: () => circles(120, 0.1, 9) },
};
const toXY = (pts) => ({ X: pts.map((p) => [p.x, p.y]), y: pts.map((p) => (p.label ? 1 : -1)) });
const D = [-1.3, 1.3];

export default async function render(main) {
  header(main, 'Support vector machines',
    'Many lines separate two classes; an SVM picks the one with the <b>widest empty street</b> between them. Only the points on the edge of the street (the <b>support vectors</b>) decide where it goes. Swap the dot product for a <b>kernel</b> and the same algorithm draws curved boundaries, as if the data had been lifted into a higher-dimensional space.');

  // ================= Figure 1: margin, support vectors, C, kernel =================
  const fig = figure(main, 'Maximum margin, soft margin, and kernels',
    'Left: the solid line is the decision boundary f(x) = 0; dashed lines are the edges of the margin, f(x) = ±1. Ringed points are <b>support vectors</b> (α &gt; 0): a solid ring sits exactly on the margin edge, a dashed ring is inside the margin or on the wrong side and is paying a penalty. All other points have α = 0 and could be deleted without moving the boundary. <b>Click</b> to add an orange point, <b>shift-click</b> for a blue one. Right: every training point placed by its signed margin y·f(x) and its weight α. Hover any point for its numbers.');
  let dsKey = 'blobs', kernel = 'linear', logC = 0, logG = Math.log10(3), pts = DATA.blobs.make(), model = null, ms = 0;
  select(fig.controls, { label: 'data', value: dsKey, options: Object.entries(DATA).map(([k, v]) => [k, v.name]), onChange: (k) => { dsKey = k; pts = DATA[k].make(); solve(); } });
  select(fig.controls, { label: 'kernel', value: kernel, options: [['linear', 'linear  K = x·x′'], ['rbf', 'RBF  K = exp(−γ‖x − x′‖²)']], onChange: (k) => { kernel = k; gS.input.disabled = k !== 'rbf'; solve(); } });
  slider(fig.controls, { label: 'C (penalty for violations)', min: -2, max: 3, step: 0.1, value: logC, fmt: (v) => fmtNum(10 ** v), onInput: (v) => { logC = v; solve(); } });
  const gS = slider(fig.controls, { label: 'RBF γ (kernel width⁻¹)', min: -1, max: 2, step: 0.05, value: logG, fmt: (v) => fmtNum(10 ** v), onInput: (v) => { logG = v; solve(); } });
  gS.input.disabled = true;
  button(fig.controls, 'Reset points', () => { pts = DATA[dsKey].make(); solve(); });
  const readout = stats(fig.root);
  const row = el('div', { class: 'fig-row' }); fig.body.append(row);
  const left = el('div'), right = el('div'); row.append(left, right);
  legend(fig.body, [['class −1', cat(0)], ['class +1', cat(1)], ['support vector on the margin (0 < α < C)', ink()], ['support vector inside margin or misclassified (α = C)', muted()]]);

  const F = frame(left, { width: 500, height: 470, x: D, y: D, xLabel: 'x₁', yLabel: 'x₂', grid: false });
  const res = 4, gw = Math.ceil(F.w / res), gh = Math.ceil(F.h / res), fGrid = new Float32Array(gw * gh);
  const canvas = decisionCanvas(F, () => 0.5, { res, alpha: 0.38 });
  F.svg.append('defs').append('clipPath').attr('id', 'svm-clip').append('rect').attr('width', F.w).attr('height', F.h);
  const cL = F.layer('contours').attr('clip-path', 'url(#svm-clip)'), ringL = F.layer('rings'), ptL = F.layer('pts');
  const proj = d3.geoTransform({ point(px, py) { this.stream.point((px - 1) * res, (py - 1) * res); } });
  const padded = new Float32Array((gw + 2) * (gh + 2));
  F.svg.style('cursor', 'crosshair').on('click', (e) => {
    const [mx, my] = d3.pointer(e, F.g.node());
    if (mx < 0 || my < 0 || mx > F.w || my > F.h) return;
    pts = pts.concat([{ x: F.x.invert(mx), y: F.y.invert(my), label: e.shiftKey ? 0 : 1 }]); solve();
  });

  const K = frame(right, { width: 500, height: 470, x: [-1.5, 3], y: [-0.05, 1.08], xLabel: 'signed margin  y·f(x)', yLabel: 'α / C  (weight in the solution)' });
  const kBand = K.layer('band'), kPts = K.layer('pts');
  [[0, 'boundary'], [1, 'margin edge']].forEach(([v, t]) => {
    kBand.append('line').attr('x1', K.x(v)).attr('x2', K.x(v)).attr('y1', 0).attr('y2', K.h).attr('stroke', ink2()).attr('stroke-width', 1.2).attr('stroke-dasharray', v ? '5 4' : null);
    kBand.append('text').attr('x', K.x(v) + 4).attr('y', K.h - 6).attr('font-size', 11.5).attr('fill', ink2()).text(t);
  });
  kBand.append('text').attr('x', K.w - 4).attr('text-anchor', 'end').attr('y', K.y(0.1)).attr('font-size', 12).attr('fill', ink2()).text('α = 0: beyond the margin, ignored');
  kBand.append('text').attr('x', K.x(-1.45)).attr('y', K.y(0.9)).attr('font-size', 12).attr('fill', ink2()).text('α = C: violators, weight capped');

  function solve() {
    const { X, y } = toXY(pts), C = 10 ** logC, kern = kernel === 'rbf' ? kRBF(10 ** logG) : kLinear;
    const t0 = performance.now();
    model = smo(X, y, kern, C);
    ms = performance.now() - t0;
    paint();
  }

  function paint() {
    const m = model, { y } = toXY(pts);
    // decision values on the canvas grid (reused for the contours)
    for (let j = 0; j < gh; j++) { const py = F.y.invert(j * res + res / 2); for (let i = 0; i < gw; i++) fGrid[j * gw + i] = m.f([F.x.invert(i * res + res / 2), py]); }
    canvas.draw((px, py) => { const i = Math.min(gw - 1, Math.max(0, Math.floor(F.x(px) / res))), j = Math.min(gh - 1, Math.max(0, Math.floor(F.y(py) / res))); return sigmoid(2.2 * fGrid[j * gw + i]); });
    for (let j = 0; j < gh + 2; j++) { const jj = Math.min(gh - 1, Math.max(0, j - 1)); for (let i = 0; i < gw + 2; i++) padded[j * (gw + 2) + i] = fGrid[jj * gw + Math.min(gw - 1, Math.max(0, i - 1))]; }
    const cs = d3.contours().size([gw + 2, gh + 2]).thresholds([-1, 0, 1])(padded);
    cL.selectAll('path').data(cs).join('path').attr('d', d3.geoPath(proj)).attr('fill', 'none').attr('stroke', ink())
      .attr('stroke-width', (d) => (d.value === 0 ? 2.5 : 1.5)).attr('stroke-dasharray', (d) => (d.value === 0 ? null : '6 4'));
    const C = m.C, info = pts.map((p, i) => ({ ...p, i, a: m.a[i], yf: y[i] * m.fTrain[i], bound: m.a[i] > C - 1e-8 * Math.max(1, C) }));
    const tipH = (d) => `class ${d.label ? '+1' : '−1'}<br>α = ${fmtNum(d.a)}${d.a > 1e-8 ? (d.bound ? ' (= C)' : ' (on margin)') : ' (not a support vector)'}<br>y·f(x) = ${d.yf.toFixed(2)}`;
    ringL.selectAll('circle').data(info.filter((d) => d.a > 1e-8)).join('circle').attr('cx', (d) => F.x(d.x)).attr('cy', (d) => F.y(d.y)).attr('r', 8)
      .attr('fill', 'none').attr('stroke', (d) => (d.bound ? muted() : ink())).attr('stroke-width', 1.8).attr('stroke-dasharray', (d) => (d.bound ? '3 2' : null));
    ptL.selectAll('circle').data(info).join('circle').attr('cx', (d) => F.x(d.x)).attr('cy', (d) => F.y(d.y)).attr('r', 4.5)
      .attr('fill', (d) => cat(d.label)).attr('stroke', css('--surface')).attr('stroke-width', 1.2)
      .on('mousemove', (e, d) => tip(e, tipH(d))).on('mouseleave', untip);
    // KKT panel: jitter α=0 points vertically a little so they don't stack
    const jit = rng(5); const jv = info.map(() => jit() * 0.04);
    kPts.selectAll('circle').data(info).join('circle').attr('cx', (d) => K.x(Math.max(-1.5, Math.min(3, d.yf)))).attr('cy', (d, i) => K.y(d.a / C + (d.a > 1e-8 ? 0 : jv[i])))
      .attr('r', 4.5).attr('fill', (d) => cat(d.label)).attr('stroke', (d) => (d.a > 1e-8 ? ink() : css('--surface'))).attr('stroke-width', (d) => (d.a > 1e-8 ? 1.4 : 1.2))
      .on('mousemove', (e, d) => tip(e, tipH(d))).on('mouseleave', untip);
    const acc = d3.mean(info, (d) => (d.yf > 0 ? 1 : 0)), nB = info.filter((d) => d.a > 1e-8 && d.bound).length;
    readout.set({
      'support vectors': `${m.sv.length} of ${pts.length}`,
      'on margin / violating': `${m.sv.length - nB} / ${nB}`,
      [kernel === 'linear' ? 'margin width 2/‖w‖' : 'margin width (feature space)']: m.margin.toFixed(3),
      'training accuracy': (100 * acc).toFixed(1) + '%',
      'SMO iterations · time': `${m.it}${m.capped ? ' (cap)' : ''} · ${ms.toFixed(1)} ms`,
    });
  }
  solve();

  // ================= Figure 2: the kernel trick in 1-D =================
  const fig2 = figure(main, 'The kernel trick: lift, then cut with a straight line',
    'The 1-D points on the bottom row cannot be split by one threshold: orange sits in the middle, blue on both sides. Add a second coordinate z = t·x² and lift them. Once lifted, a straight line separates the classes (a linear SVM, C = 10, is refitted at every t). Where that line crosses the parabola, drop back down: one straight cut up here is <b>two thresholds</b> on the original line. An RBF kernel does this kind of lift into infinitely many dimensions without ever computing the coordinates, because the SVM only needs dot products K(x, x′).');
  const r1 = rng(23), pts1 = [];
  while (pts1.length < 44) { const x = r1() * 2 - 1; if (Math.abs(Math.abs(x) - 0.45) < 0.07) continue; pts1.push({ x, label: Math.abs(x) < 0.45 ? 1 : 0 }); }
  let lift = 0, stopL = null;
  const lS = slider(fig2.controls, { label: 'lift t', min: 0, max: 1, step: 0.01, value: 0, fmt: (v) => v.toFixed(2), onInput: (v) => { if (stopL) { stopL(); stopL = null; } lift = v; paint2(); } });
  button(fig2.controls, 'Animate the lift', () => {
    if (stopL) stopL(); lift = 0; let tt = 0;
    stopL = animate((dt) => { tt += dt; lift = Math.min(1, tt / 2200); lS.set(lift); paint2(); return lift < 1; });
  }, true);
  const readout2 = stats(fig2.root);
  const L = frame(fig2.body, { width: 760, height: 360, x: [-1.1, 1.1], y: [-0.28, 1.1], xLabel: 'x (the original single feature)', yLabel: 'z = t·x²  (lifted coordinate)' });
  L.svg.style('max-width', '760px').style('display', 'block').style('margin', '0 auto');
  L.svg.append('defs').append('clipPath').attr('id', 'svm-clip2').append('rect').attr('width', L.w).attr('height', L.h);
  const base = -0.18;
  L.layer('base').append('line').attr('x1', 0).attr('x2', L.w).attr('y1', L.y(base)).attr('y2', L.y(base)).attr('stroke', muted()).attr('stroke-width', 1);
  L.layer('base').append('text').attr('x', 4).attr('y', L.y(base) + 16).attr('font-size', 11.5).attr('fill', ink2()).text('original 1-D data');
  const lPar = L.layer('par'), lLine = L.layer('line').attr('clip-path', 'url(#svm-clip2)'), lDrop = L.layer('drop'), lBase = L.layer('basepts'), lPts = L.layer('pts');

  function paint2() {
    const X = pts1.map((p) => [p.x, lift * p.x * p.x]), y = pts1.map((p) => (p.label ? 1 : -1));
    const m = smo(X, y, kLinear, 10);
    let w0 = 0, w1 = 0; m.sv.forEach((t) => { w0 += m.a[t] * y[t] * X[t][0]; w1 += m.a[t] * y[t] * X[t][1]; });
    const xs = d3.range(-1.1, 1.1001, 0.01);
    lPar.selectAll('path').data([xs]).join('path').attr('fill', 'none').attr('stroke', muted()).attr('stroke-width', 1.2).attr('stroke-dasharray', '3 3')
      .attr('d', d3.line().x((x) => L.x(x)).y((x) => L.y(lift * x * x)));
    // boundary w0·x + w1·z + b = 0 (and margins ±1), drawn as lines across the plot
    const lineAt = (v) => {
      if (Math.abs(w1) > 1e-9) return [[-1.1, (v - m.b - w0 * -1.1) / w1], [1.1, (v - m.b - w0 * 1.1) / w1]];
      const x0 = (v - m.b) / (w0 || 1e-9); return [[x0, -0.3], [x0, 1.1]];
    };
    const flat = Math.hypot(w0, w1) < 1e-3;          // no useful cut: the SVM just predicts one class everywhere
    lLine.selectAll('line').data(flat ? [] : [-1, 0, 1]).join('line').each(function (v) {
      const [[x1, z1], [x2, z2]] = lineAt(v);
      d3.select(this).attr('x1', L.x(x1)).attr('y1', L.y(z1)).attr('x2', L.x(x2)).attr('y2', L.y(z2));
    }).attr('stroke', ink()).attr('stroke-width', (v) => (v ? 1.3 : 2.5)).attr('stroke-dasharray', (v) => (v ? '6 4' : null));
    // crossings of the boundary with the parabola: w1·t·x² + w0·x + b = 0
    let cross = [];
    const A = w1 * lift, Bq = w0, Cq = m.b;
    if (flat) cross = [];
    else if (Math.abs(A) > 1e-9) { const disc = Bq * Bq - 4 * A * Cq; if (disc >= 0) cross = [(-Bq - Math.sqrt(disc)) / (2 * A), (-Bq + Math.sqrt(disc)) / (2 * A)]; }
    else if (Math.abs(Bq) > 1e-9) cross = [-Cq / Bq];
    cross = cross.filter((x) => x > -1.1 && x < 1.1);
    lDrop.selectAll('line').data(cross).join('line').attr('x1', (x) => L.x(x)).attr('x2', (x) => L.x(x)).attr('y1', (x) => L.y(lift * x * x)).attr('y2', L.y(base))
      .attr('stroke', ink()).attr('stroke-width', 1.5).attr('stroke-dasharray', '2 3');
    lDrop.selectAll('path').data(cross).join('path').attr('d', d3.symbol().type(d3.symbolTriangle).size(60)()).attr('transform', (x) => `translate(${L.x(x)},${L.y(base) - 9}) rotate(180)`).attr('fill', ink());
    lBase.selectAll('circle').data(pts1).join('circle').attr('cx', (d) => L.x(d.x)).attr('cy', L.y(base)).attr('r', 4).attr('fill', (d) => cat(d.label)).attr('fill-opacity', 0.55).attr('stroke', css('--surface')).attr('stroke-width', 1);
    lPts.selectAll('circle').data(pts1).join('circle').attr('cx', (d) => L.x(d.x)).attr('cy', (d) => L.y(lift * d.x * d.x)).attr('r', 4.5)
      .attr('fill', (d) => cat(d.label)).attr('stroke', (d, i) => (m.a[i] > 1e-8 ? ink() : css('--surface'))).attr('stroke-width', (d, i) => (m.a[i] > 1e-8 ? 1.6 : 1.2))
      .on('mousemove', (e, d) => tip(e, `x = ${d.x.toFixed(2)}, z = ${(lift * d.x * d.x).toFixed(2)}<br>class ${d.label ? '+1' : '−1'}`)).on('mouseleave', untip);
    const acc = d3.mean(m.fTrain, (f, i) => (f * y[i] > 0 ? 1 : 0));
    readout2.set({ 'lift t': lift.toFixed(2), 'training accuracy of the straight cut': (100 * acc).toFixed(1) + '%', 'support vectors': m.sv.length, 'margin width': flat ? '— (no cut)' : m.margin.toFixed(3), 'thresholds back on the line': flat ? 'none: predicts the majority class' : cross.length ? cross.map((x) => x.toFixed(2)).join(', ') : 'none' });
  }
  paint2();

  // ---- numbers for the prose (computed from the seeded datasets) ----
  const run = (key, kern, C) => { const { X, y } = toXY(DATA[key].make()); const m = smo(X, y, kern, C); return { m, acc: d3.mean(m.fTrain, (f, i) => (f * y[i] > 0 ? 1 : 0)) }; };
  const bLo = run('blobs', kLinear, 0.01), bMid = run('blobs', kLinear, 1), bHi = run('blobs', kLinear, 100);
  const cLin = run('circles', kLinear, 1), cRbf = run('circles', kRBF(3), 1), mHi = run('moons', kRBF(100), 1000), mOk = run('moons', kRBF(3), 1);
  const pc = (v) => (100 * v).toFixed(1) + '%';
  const lift0 = (() => { const X = pts1.map((p) => [p.x, 0]), y = pts1.map((p) => (p.label ? 1 : -1)); const m = smo(X, y, kLinear, 10); return d3.mean(m.fTrain, (f, i) => (f * y[i] > 0 ? 1 : 0)); })();
  section(main, 'What to notice', `<ul>
    <li>On the blobs with C = 1, ${bMid.m.sv.length} of the 120 points are support vectors; the boundary depends on them alone. Hover a point far from the line: α = 0. Click to add an orange point deep in orange territory, beyond the margin: it gets α = 0 too, and the boundary does not move at all. Add one inside the street and it does.</li>
    <li><b>C sets how much a violation costs.</b> At C = 0.01 violations are cheap, so the street is wide (margin ${bLo.m.margin.toFixed(2)}) and ${bLo.m.sv.length} points become support vectors. At C = 100 it narrows to ${bHi.m.margin.toFixed(2)} with ${bHi.m.sv.length} support vectors, as the SVM works harder to classify every point. Small C is more regularised.</li>
    <li>The right panel is the whole theory in one picture (the <b>KKT conditions</b>): points beyond the margin (y·f &gt; 1) have α = 0; points exactly on it (y·f = 1) have 0 &lt; α &lt; C; points inside it or misclassified (y·f &lt; 1) have α = C. Nothing ever lands elsewhere.</li>
    <li>Choose the <b>ring</b> data: a linear SVM manages only ${pc(cLin.acc)} training accuracy, because no straight line can put the core on one side and the whole ring on the other. Switch to <b>RBF</b> with γ = 3 and it reaches ${pc(cRbf.acc)} with a closed curve.</li>
    <li>γ is the inverse width of each support vector's influence. On the moons, γ = 3, C = 1 gives a smooth curve (${pc(mOk.acc)} training accuracy). Push γ to 100 and C to 1000: training accuracy hits ${pc(mHi.acc)}, but the boundary now wraps tightly around individual points, the shading fades to nothing between them, and ${mHi.m.sv.length} of the 120 points are support vectors. The model is memorising points rather than describing the shape. That is overfitting.</li>
    <li>In the lift figure, at t = 0 no single threshold helps, so the SVM gives up and predicts blue everywhere: ${pc(lift0)} right, which is just the share of blue points (${pts1.filter((p) => !p.label).length} of ${pts1.length}). As t grows, the middle points stay low while the outer ones rise, and the classes become separable.</li>
  </ul>`);

  steps(main, 'How it works', [
    'For a linear SVM the boundary is f(x) = w·x + b = 0. Scale w so that the closest points have |f(x)| = 1; the street between f = −1 and f = +1 then has width 2/‖w‖.',
    'Maximising the width means minimising ‖w‖². A <b>soft margin</b> lets point i violate by a slack ξ<sub>i</sub> at a cost of C·ξ<sub>i</sub>, so the SVM trades width against violations. This is the same as minimising the <b>hinge loss</b> max(0, 1 − y·f(x)) plus an L2 penalty on w.',
    'The optimal w turns out to be a weighted sum of training points, w = Σ α<sub>i</sub> y<sub>i</sub> x<sub>i</sub>, and most α<sub>i</sub> are zero. The points with α<sub>i</sub> &gt; 0 are the support vectors.',
    'The solver (here SMO, as in LIBSVM) works on the α\'s directly: repeatedly pick the pair of points that most violate the optimality conditions, and re-optimise those two in closed form, until no pair violates by more than a tolerance.',
    'Training and prediction only ever use dot products x<sub>i</sub>·x<sub>j</sub>. Replace them with a <b>kernel</b> K(x<sub>i</sub>, x<sub>j</sub>), a dot product in some lifted feature space, and you get a nonlinear boundary for free. That swap is the <b>kernel trick</b>.',
  ]);
  math(main, 'primal:  min ½‖w‖² + C Σ ξ_i    s.t.  y_i (w·x_i + b) ≥ 1 − ξ_i,  ξ_i ≥ 0\ndual:    max Σ α_i − ½ Σ_ij α_i α_j y_i y_j K(x_i, x_j)    s.t.  0 ≤ α_i ≤ C,  Σ α_i y_i = 0\npredict: f(x) = Σ_i α_i y_i K(x_i, x) + b          RBF: K(x, x′) = exp(−γ ‖x − x′‖²)');

  callout(main, '', 'In biology', 'SVMs were the workhorse of early expression-based cancer classification (for example the leukaemia and multi-tumour microarray studies around 2000), because they cope well with many more genes than samples. <i>CIBERSORT</i> uses ν-support vector regression to deconvolve bulk RNA-seq into immune cell fractions. String and spectrum kernels let SVMs classify DNA and protein sequences (splice sites, binding motifs) directly, and SVM-RFE is a classic gene-selection method.');
  callout(main, 'warn', 'Pitfalls', `<ul>
    <li><b>Scale your features.</b> The margin and the RBF distance are both measured in raw units, so one large-valued gene dominates. <a href="#/archive/topics/scaling">Standardise first.</a></li>
    <li>C and γ interact strongly; tune them together on a log grid with cross-validation, not one at a time.</li>
    <li>SVM outputs are distances, not probabilities. If you need calibrated probabilities, add Platt scaling or isotonic calibration, fitted on held-out data.</li>
    <li>Kernel SVMs scale poorly with the number of samples (the kernel matrix is n × n), which matters for single-cell data with 10⁵–10⁶ cells. Use a linear SVM or a different model there.</li>
  </ul>`);
  seeAlso(main, [['concepts', 'logistic-regression', 'Logistic regression'], ['concepts', 'loss-functions', 'Loss functions (hinge)'], ['concepts', 'regularization', 'Regularization'], ['concepts', 'scaling', 'Feature scaling'], ['omics', 'bulk-deconvolution', 'Cell-type deconvolution']]);
}

function fmtNum(v) { return v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v >= 1 ? v.toFixed(2) : v >= 0.01 ? v.toFixed(3) : v.toExponential(1); }
