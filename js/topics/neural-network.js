// Neural network playground: a small multilayer perceptron trained live in the browser with hand-written backprop.
// Decision surface (left), network diagram with one activation map per neuron (right), loss curves (below).
import { header, figure, slider, select, button, checkbox, stats, section, steps, callout, math, legend, seeAlso, animate, tip, untip, el } from '../lib/ui.js';
import { frame, decisionCanvas, cat, css, ink2, muted } from '../lib/plot.js';
import { rng, moons, circles, spiral, xor } from '../lib/data.js';
const d3 = window.d3;

const DOM = 1.35; // plotted input range [-DOM, DOM]
const DATA = {
  circles: ['Circles', () => circles(200, 0.08, 3)],
  xor: ['XOR (four quadrants)', () => xor(200, 0.12, 5)],
  moons: ['Two moons', () => moons(200, 0.1, 2)],
  spiral: ['Spiral (hard)', () => spiral(300, 0.05, 4)],
};
const ACTS = {
  tanh: { name: 'tanh', f: Math.tanh, d: (a) => 1 - a * a, he: false },
  relu: { name: 'ReLU', f: (z) => (z > 0 ? z : 0), d: (a) => (a > 0 ? 1 : 0), he: true },
  sigmoid: { name: 'sigmoid', f: (z) => 1 / (1 + Math.exp(-z)), d: (a) => a * (1 - a), he: false },
  linear: { name: 'linear (none)', f: (z) => z, d: () => 1, he: false },
};
const FEATS = {
  xy: { names: ['x₁', 'x₂'], f: (x, y, o) => { o[0] = x; o[1] = y; } },
  poly: { names: ['x₁', 'x₂', 'x₁²', 'x₂²', 'x₁x₂'], f: (x, y, o) => { o[0] = x; o[1] = y; o[2] = x * x; o[3] = y * y; o[4] = x * y; } },
};

/** Fully connected net. sizes = [nIn, h1, ..., 1]. Hidden layers use `act`; the output unit is a sigmoid (probability of class 1). */
export function makeNet(sizes, actName, seed) {
  const r = rng(seed), act = ACTS[actName], L = sizes.length - 1;
  const W = [], b = [], gW = [], gb = [], A = [], D = [];
  for (let l = 0; l < L; l++) {
    const nin = sizes[l], nout = sizes[l + 1];
    const sd = act.he && l < L - 1 ? Math.sqrt(2 / nin) : Math.sqrt(2 / (nin + nout)); // He for ReLU, Glorot otherwise
    W.push(Float64Array.from({ length: nin * nout }, () => r.normal(0, sd)));
    b.push(new Float64Array(nout)); gW.push(new Float64Array(nin * nout)); gb.push(new Float64Array(nout));
  }
  for (let l = 0; l <= L; l++) { A.push(new Float64Array(sizes[l])); D.push(new Float64Array(sizes[l])); }
  const shuf = rng(seed + 977);

  function forward(inp) {
    A[0].set(inp);
    for (let l = 0; l < L; l++) {
      const nin = sizes[l], nout = sizes[l + 1], w = W[l], a = A[l], out = A[l + 1], last = l === L - 1;
      for (let i = 0; i < nout; i++) {
        let z = b[l][i];
        for (let j = 0; j < nin; j++) z += w[i * nin + j] * a[j];
        out[i] = last ? 1 / (1 + Math.exp(-z)) : act.f(z);
      }
    }
    return A[L][0];
  }
  /** One epoch of mini-batch SGD on binary cross-entropy. */
  function epoch(X, Y, idx, lr, batch = 10) {
    for (let i = idx.length - 1; i > 0; i--) { const j = Math.floor(shuf() * (i + 1)); const t = idx[i]; idx[i] = idx[j]; idx[j] = t; }
    for (let s = 0; s < idx.length; s += batch) {
      const m = Math.min(batch, idx.length - s);
      for (let l = 0; l < L; l++) { gW[l].fill(0); gb[l].fill(0); }
      for (let k = s; k < s + m; k++) {
        const n = idx[k];
        forward(X[n]);
        D[L][0] = A[L][0] - Y[n]; // dLoss/dz for sigmoid + cross-entropy
        for (let l = L - 1; l >= 0; l--) {
          const nin = sizes[l], nout = sizes[l + 1], w = W[l], a = A[l], dn = D[l + 1], g = gW[l];
          for (let i = 0; i < nout; i++) { const d = dn[i]; gb[l][i] += d; for (let j = 0; j < nin; j++) g[i * nin + j] += d * a[j]; }
          if (l > 0) for (let j = 0; j < nin; j++) { let sum = 0; for (let i = 0; i < nout; i++) sum += w[i * nin + j] * dn[i]; D[l][j] = sum * act.d(a[j]); }
        }
      }
      for (let l = 0; l < L; l++) {
        const w = W[l], g = gW[l], bb = b[l], gbb = gb[l];
        for (let k = 0; k < w.length; k++) w[k] -= lr * g[k] / m;
        for (let k = 0; k < bb.length; k++) bb[k] -= lr * gbb[k] / m;
      }
    }
  }
  function evaluate(X, Y, idx) {
    let loss = 0, ok = 0;
    for (const n of idx) { const p = Math.min(1 - 1e-7, Math.max(1e-7, forward(X[n]))); loss -= Y[n] ? Math.log(p) : Math.log(1 - p); ok += (p >= 0.5) === (Y[n] === 1); }
    return { loss: loss / idx.length, acc: ok / idx.length };
  }
  const nParams = W.reduce((s, w) => s + w.length, 0) + b.reduce((s, v) => s + v.length, 0);
  return { sizes, L, W, b, A, act, forward, epoch, evaluate, nParams };
}

function makeData(key, featKey) {
  const pts = DATA[key][1]();
  const nf = FEATS[featKey].names.length;
  const X = pts.map((p) => { const o = new Float64Array(nf); FEATS[featKey].f(p.x, p.y, o); return o; });
  const Y = pts.map((p) => p.label);
  const order = rng(11).shuffle(pts.map((_, i) => i));
  const nTr = Math.round(0.7 * pts.length);
  return { pts, X, Y, train: order.slice(0, nTr), test: order.slice(nTr), nf };
}

/** Headless run used to check the numbers quoted in the prose. */
export function experiment({ data = 'circles', layers = 2, units = 4, act = 'tanh', feats = 'xy', lr = 0.03, epochs = 300, seed = 1, every: ev = 50 }) {
  const D = makeData(data, feats);
  const net = makeNet([D.nf, ...Array(layers).fill(units), 1], act, seed);
  const out = [];
  for (let e = 1; e <= epochs; e++) {
    net.epoch(D.X, D.Y, D.train.slice(), lr);
    if (e % ev === 0) { const tr = net.evaluate(D.X, D.Y, D.train), te = net.evaluate(D.X, D.Y, D.test); out.push([e, +tr.loss.toFixed(3), +te.loss.toFixed(3), +te.acc.toFixed(3)]); }
  }
  return out;
}

export default async function render(main) {
  header(main, 'Neural network playground',
    'A neural network is a stack of simple units. Each one draws a line, then bends it with a non-linear <b>activation</b>. Later layers combine those bent pieces into curved shapes. Watch a small network <b>learn to bend space</b> until the two classes fall on different sides.');

  const fig = figure(main, 'Train a multilayer perceptron live',
    'Left: the network\'s predicted class at every point of the plane (stronger colour = more confident). Filled dots are training points; hollow dots are held-out test points. Right: every neuron drawn as a tiny map of its own output over the same plane, with weights as lines (orange = positive, blue = negative, thicker = larger). <b>Hover over a neuron</b> to show its map full size on the left.');

  const st = { data: 'circles', feats: 'xy', layers: 2, units: 4, act: 'tanh', lr: 0.03, seed: 1, playing: false, epoch: 0, hover: null, hist: [] };
  let D, net, stopLoop = null;

  select(fig.controls, { label: 'Data', value: st.data, options: Object.entries(DATA).map(([k, v]) => [k, v[0]]), onChange: (v) => { st.data = v; build(true); } });
  select(fig.controls, { label: 'Inputs', value: st.feats, options: [['xy', 'x₁, x₂'], ['poly', 'x₁, x₂, x₁², x₂², x₁x₂']], onChange: (v) => { st.feats = v; build(true); } });
  slider(fig.controls, { label: 'Hidden layers', min: 0, max: 4, value: st.layers, onInput: (v) => { st.layers = v; build(); } });
  slider(fig.controls, { label: 'Units / layer', min: 1, max: 8, value: st.units, onInput: (v) => { st.units = v; build(); } });
  select(fig.controls, { label: 'Activation', value: st.act, options: Object.entries(ACTS).map(([k, v]) => [k, v.name]), onChange: (v) => { st.act = v; build(); } });
  slider(fig.controls, { label: 'Learning rate', min: -3, max: 1, step: 0.1, value: Math.log10(st.lr), fmt: (v) => (+(10 ** v).toPrecision(2)).toString(), onInput: (v) => { st.lr = 10 ** v; } });
  const playBtn = button(fig.controls, 'Play', () => (st.playing ? pause() : play()), true);
  button(fig.controls, 'Step', () => { pause(); train(1); paint(); });
  button(fig.controls, 'Reset', () => build());
  button(fig.controls, 'New random init', () => { st.seed += 1; build(); });
  let showTest = true;
  checkbox(fig.controls, { label: 'Show test points', checked: true, onChange: (v) => { showTest = v; drawPoints(); } });
  const readout = stats(fig.root);

  const row = el('div', { class: 'fig-row' });
  const left = el('div'), right = el('div');
  row.append(left, right); fig.body.append(row);

  // --- left: decision surface + loss curve
  const F = frame(left, { width: 460, height: 400, x: [-DOM, DOM], y: [-DOM, DOM], xLabel: 'x₁', yLabel: 'x₂', grid: false });
  const surf = decisionCanvas(F, () => 0.5, { res: 5, alpha: 0.42 });
  const surfLabel = F.g.append('text').attr('x', 6).attr('y', 14).attr('font-size', 12).attr('fill', ink2());
  const lossTitle = el('div', { class: 'fig-caption', style: 'margin:6px 0 0' }, 'Loss (binary cross-entropy) per epoch');
  left.append(lossTitle);
  legend(left, [['training loss', cat(2)], ['test loss', cat(3)]]);
  const LC = frame(left, { width: 460, height: 180, x: [0, 100], y: [0, 0.8], xLabel: 'epoch', yLabel: 'loss' });

  // --- right: network diagram on a canvas
  const NW = 480, NH = 420, T = 34, G = 24; // logical canvas size, tile size, tile resolution
  const dpr = 2;
  const netCanvas = el('canvas', { width: NW * dpr, height: NH * dpr, style: `width:100%;max-width:${NW}px;height:auto;cursor:default`, 'aria-label': 'Network diagram with per-neuron activation maps' });
  right.append(netCanvas);
  const nctx = netCanvas.getContext('2d'); nctx.scale(dpr, dpr);
  const tileCanvas = el('canvas', { width: G, height: G });
  const tctx = tileCanvas.getContext('2d');
  const tileImg = tctx.createImageData(G, G);
  legend(right, [['class 0 / negative', cat(0)], ['class 1 / positive', cat(1)]]);
  right.append(el('div', { class: 'fig-caption', style: 'margin-top:4px' }, 'Each tile shows that neuron\'s output across the input plane (x₁ left→right, x₂ bottom→top). The first column is the raw inputs; the last is the output probability.'));

  let colScale, layout = [], maps = [];
  const gridIn = [];

  function build(newData = false) {
    if (newData || !D) D = makeData(st.data, st.feats);
    net = makeNet([D.nf, ...Array(st.layers).fill(st.units), 1], st.act, st.seed);
    st.epoch = 0; st.hist = []; st.hover = null;
    colScale = d3.scaleLinear().domain([-1, 0, 1]).range([cat(0), css('--div-mid'), cat(1)]).clamp(true);
    // tile grid inputs
    gridIn.length = 0;
    for (let gy = 0; gy < G; gy++) for (let gx = 0; gx < G; gx++) {
      const o = new Float64Array(D.nf); FEATS[st.feats].f(-DOM + 2 * DOM * (gx + 0.5) / G, DOM - 2 * DOM * (gy + 0.5) / G, o); gridIn.push(o);
    }
    maps = net.sizes.map((n) => Array.from({ length: n }, () => new Float32Array(G * G)));
    // node positions
    const nCols = net.sizes.length, x0 = 34, x1 = NW - T - 14;
    layout = net.sizes.map((n, l) => {
      const x = nCols === 1 ? x0 : x0 + (x1 - x0) * l / (nCols - 1);
      const gap = Math.min(46, (NH - 50) / n), top = 30 + (NH - 40 - gap * n) / 2;
      return Array.from({ length: n }, (_, i) => ({ x, y: top + i * gap + (gap - T) / 2 }));
    });
    drawPoints();
    paint();
  }

  function train(nEpochs) {
    for (let e = 0; e < nEpochs; e++) {
      net.epoch(D.X, D.Y, D.train.slice(), st.lr);
      st.epoch++;
      const tr = net.evaluate(D.X, D.Y, D.train), te = net.evaluate(D.X, D.Y, D.test);
      st.hist.push([st.epoch, tr.loss, te.loss]);
      if (!isFinite(tr.loss)) { pause(); break; }
    }
  }

  function play() {
    st.playing = true; playBtn.textContent = 'Pause';
    stopLoop = animate(() => {
      if (!st.playing) return false;
      train(2); paint();
      if (st.epoch >= 3000) { pause(); return false; }
      return true;
    });
  }
  function pause() { st.playing = false; playBtn.textContent = 'Play'; if (stopLoop) { stopLoop(); stopLoop = null; } }

  const tmp = new Float64Array(5);
  function predictAt(px, py) { FEATS[st.feats].f(px, py, tmp); return net.forward(tmp.subarray(0, D.nf)); }
  /** Map a neuron's activation to [-1, 1] for colouring. */
  function norm(l, v, maxAbs) {
    if (l === net.L) return 2 * v - 1;
    if (l === 0) return v / DOM;
    if (st.act === 'sigmoid') return 2 * v - 1;
    if (st.act === 'tanh') return v;
    return v / (maxAbs || 1);
  }

  function drawPoints() {
    const pts = D.pts.map((p, i) => ({ ...p, i, test: D.test.includes(i) })).filter((p) => showTest || !p.test);
    F.layer('pts').selectAll('circle').data(pts).join('circle')
      .attr('cx', (d) => F.x(d.x)).attr('cy', (d) => F.y(d.y)).attr('r', 4.2)
      .attr('fill', (d) => (d.test ? css('--surface') : cat(d.label))).attr('stroke', (d) => (d.test ? cat(d.label) : css('--surface')))
      .attr('stroke-width', (d) => (d.test ? 2 : 1.2))
      .on('mousemove', (e, d) => tip(e, `${d.test ? 'test' : 'training'} point · class <b>${d.label}</b><br>P(class 1) = ${predictAt(d.x, d.y).toFixed(3)}`))
      .on('mouseleave', untip);
  }

  function computeMaps() {
    for (let k = 0; k < gridIn.length; k++) {
      net.forward(gridIn[k]);
      for (let l = 0; l < net.A.length; l++) { const a = net.A[l], m = maps[l]; for (let i = 0; i < a.length; i++) m[i][k] = a[i]; }
    }
  }

  function drawNet() {
    nctx.clearRect(0, 0, NW, NH);
    const font = getComputedStyle(document.body).fontFamily;
    nctx.font = `12px ${font}`; nctx.fillStyle = ink2(); nctx.textAlign = 'center';
    layout.forEach((col, l) => {
      const label = l === 0 ? 'Input' : l === net.L ? 'Output' : `Hidden ${l}`;
      nctx.fillText(label, col[0].x + T / 2, 14);
    });
    // weights
    let wmax = 1e-9; net.W.forEach((w) => w.forEach((v) => { wmax = Math.max(wmax, Math.abs(v)); }));
    for (let l = 0; l < net.L; l++) {
      const nin = net.sizes[l], nout = net.sizes[l + 1], w = net.W[l];
      for (let i = 0; i < nout; i++) for (let j = 0; j < nin; j++) {
        const v = w[i * nin + j], a = layout[l][j], bnode = layout[l + 1][i];
        const xa = a.x + T, ya = a.y + T / 2, xb = bnode.x, yb = bnode.y + T / 2, mx = (xa + xb) / 2;
        const hl = st.hover && ((st.hover.l === l + 1 && st.hover.i === i) || (st.hover.l === l && st.hover.i === j));
        nctx.strokeStyle = colScale(Math.sign(v) * Math.max(0.25, Math.abs(v) / wmax));
        nctx.globalAlpha = st.hover ? (hl ? 1 : 0.15) : 0.9;
        nctx.lineWidth = 0.6 + 4 * Math.abs(v) / wmax;
        nctx.beginPath(); nctx.moveTo(xa, ya); nctx.bezierCurveTo(mx, ya, mx, yb, xb, yb); nctx.stroke();
      }
    }
    nctx.globalAlpha = 1;
    // tiles
    nctx.imageSmoothingEnabled = true;
    const border = css('--axis'), accent = css('--ink');
    layout.forEach((col, l) => col.forEach((p, i) => {
      const m = maps[l][i]; let mx = 0; for (let k = 0; k < m.length; k++) mx = Math.max(mx, Math.abs(m[k]));
      for (let k = 0; k < m.length; k++) {
        const c = d3.color(colScale(norm(l, m[k], mx)));
        tileImg.data[4 * k] = c.r; tileImg.data[4 * k + 1] = c.g; tileImg.data[4 * k + 2] = c.b; tileImg.data[4 * k + 3] = 255;
      }
      tctx.putImageData(tileImg, 0, 0);
      nctx.drawImage(tileCanvas, p.x, p.y, T, T);
      const on = st.hover && st.hover.l === l && st.hover.i === i;
      nctx.strokeStyle = on ? accent : border; nctx.lineWidth = on ? 2 : 1;
      nctx.strokeRect(p.x + 0.5, p.y + 0.5, T - 1, T - 1);
      if (l === 0) { nctx.fillStyle = ink2(); nctx.textAlign = 'right'; nctx.fillText(FEATS[st.feats].names[i], p.x - 5, p.y + T / 2 + 4); }
    }));
  }

  netCanvas.addEventListener('mousemove', (e) => {
    const r = netCanvas.getBoundingClientRect(), s = NW / r.width;
    const mx = (e.clientX - r.left) * s, my = (e.clientY - r.top) * s;
    let hit = null;
    layout.forEach((col, l) => col.forEach((p, i) => { if (mx >= p.x && mx <= p.x + T && my >= p.y && my <= p.y + T) hit = { l, i }; }));
    const changed = JSON.stringify(hit) !== JSON.stringify(st.hover);
    st.hover = hit;
    if (hit) {
      const { l, i } = hit;
      const name = l === 0 ? `input ${FEATS[st.feats].names[i]}` : l === net.L ? 'output unit (sigmoid → P(class 1))' : `hidden layer ${l}, unit ${i + 1} (${ACTS[st.act].name})`;
      const extra = l > 0 ? `<br>bias = ${net.b[l - 1][i].toFixed(2)}<br>incoming weights: ${Array.from(net.W[l - 1].subarray(i * net.sizes[l - 1], (i + 1) * net.sizes[l - 1])).map((v) => v.toFixed(2)).join(', ')}` : '';
      tip(e, `<b>${name}</b>${extra}`);
    } else untip();
    if (changed) { paintSurface(); drawNet(); }
  });
  netCanvas.addEventListener('mouseleave', () => { untip(); if (st.hover) { st.hover = null; paintSurface(); drawNet(); } });

  function paintSurface() {
    if (st.hover) {
      const { l, i } = st.hover; const m = maps[l][i]; let mx = 0; for (let k = 0; k < m.length; k++) mx = Math.max(mx, Math.abs(m[k]));
      surf.draw((px, py) => { predictAt(px, py); return 0.5 + 0.5 * norm(l, net.A[l][i], mx); });
      surfLabel.text(l === net.L ? 'output P(class 1)' : `showing: ${l === 0 ? 'input ' + FEATS[st.feats].names[i] : `layer ${l}, unit ${i + 1}`}`);
    } else { surf.draw(predictAt); surfLabel.text(''); }
  }

  function paint() {
    computeMaps();
    paintSurface();
    drawNet();
    // loss curve
    const h = st.hist;
    const xmax = Math.max(100, st.epoch);
    const ymax = Math.max(0.8, Math.min(3, d3.max(h, (d) => Math.max(d[1], d[2])) || 0));
    LC.x.domain([0, xmax]); LC.y.domain([0, ymax]);
    LC.g.select('.axis.x').call(d3.axisBottom(LC.x).ticks(5));
    LC.g.select('.axis.y').call(d3.axisLeft(LC.y).ticks(4));
    LC.g.select('.gridline').call(d3.axisLeft(LC.y).ticks(4).tickSize(-LC.w).tickFormat(''));
    const ln = (k) => d3.line().defined((d) => isFinite(d[k])).x((d) => LC.x(d[0])).y((d) => LC.y(Math.min(ymax, d[k])));
    const thin = h.length > 600 ? h.filter((_, i) => i % Math.ceil(h.length / 600) === 0 || i === h.length - 1) : h;
    LC.layer('curves').selectAll('path').data([[1, cat(2)], [2, cat(3)]]).join('path')
      .attr('fill', 'none').attr('stroke', (d) => d[1]).attr('stroke-width', 2).attr('d', (d) => ln(d[0])(thin));
    const tr = net.evaluate(D.X, D.Y, D.train), te = net.evaluate(D.X, D.Y, D.test);
    readout.set({
      epoch: st.epoch,
      'training loss': isFinite(tr.loss) ? tr.loss.toFixed(3) : 'diverged',
      'test loss': isFinite(te.loss) ? te.loss.toFixed(3) : 'diverged',
      'test accuracy': (100 * te.acc).toFixed(0) + '%',
      parameters: net.nParams,
      architecture: net.sizes.join('–'),
    });
  }

  build(true);

  section(main, 'What to notice', `<ul>
    <li><b>Zero hidden layers</b> is plain <a href="#/concepts/logistic-regression">logistic regression</a>: one straight boundary. On Circles it stalls near 50% test accuracy however long you train.</li>
    <li>Keep zero hidden layers but switch <b>Inputs</b> to include x₁² and x₂². Now a circle is a straight line in the new feature space, and the same linear model solves it. Hidden layers <b>learn such features for you</b> instead of you having to guess them.</li>
    <li>With the default 2 layers of 4 tanh units, Circles is solved in about 200 epochs (test accuracy reaches 100%). Hover over hidden-layer-1 neurons: each is just a soft line. Layer-2 neurons combine those lines into curved regions.</li>
    <li>Switch the activation to <b>linear</b>. Stacking any number of linear layers is still one linear function, so the boundary stays straight. The non-linearity is what makes depth useful.</li>
    <li>With <b>ReLU</b> the boundary is made of straight segments (piecewise linear), while tanh gives smooth curves.</li>
    <li><b>Spiral</b> is hard. 1 layer of 4 units stays near 52% test accuracy even after 1,000 epochs. With 3 layers of 8 tanh units and learning rate 0.1, it sits on a plateau for about 400 epochs, then drops sharply and passes 90% test accuracy by epoch 500. Spikes in the loss after that are SGD overshooting.</li>
    <li><b>Learning rate.</b> At 0.001 almost nothing happens in 400 epochs. At 3 the loss curve is jagged, jumping up and down between epochs. At 10 it never settles. This is the same trade-off as the <a href="#/concepts/gradient-descent">gradient-descent</a> ball.</li>
    <li>On XOR with 1 layer of 2 units, press <b>New random init</b> a few times. After 500 epochs the first six initialisations end at test accuracies between 63% and 88%, and none of them is perfect, although two units are enough in principle. Different starts end in different valleys of a non-convex loss. Give it 2 layers of 4 units instead and it reaches 98%.</li>
  </ul>`);

  steps(main, 'How it works', [
    'Each neuron computes a weighted sum of its inputs plus a bias, <code>z = w·a + b</code>. That is a line (in 2D) or a hyperplane, which is why first-layer tiles look like soft half-planes.',
    'It then applies an <a href="#/concepts/activations">activation</a> <code>a = f(z)</code>, such as tanh or ReLU. Without this bend, layers would collapse into one linear map.',
    'The output neuron uses a sigmoid, so its value is a probability P(class 1). The loss is the binary cross-entropy, <code>−[y log p + (1−y) log(1−p)]</code>.',
    '<b>Backpropagation</b> applies the chain rule from the output back to the first layer, giving the gradient of the loss with respect to every weight (see <a href="#/concepts/backprop">Backpropagation</a>).',
    'Mini-batch SGD: shuffle the training points, then for each batch of 10, step every weight a little downhill, <code>w ← w − η·∂L/∂w</code>. One pass through the data is one <b>epoch</b>.',
    'Test points are never used for updates. The gap between the two loss curves is how far the model has overfit.',
  ]);
  math(main, 'a⁽ˡ⁾ = f( W⁽ˡ⁾ a⁽ˡ⁻¹⁾ + b⁽ˡ⁾ )          p = σ( w⁽ᴸ⁾·a⁽ᴸ⁻¹⁾ + b⁽ᴸ⁾ )\nδ⁽ᴸ⁾ = p − y      δ⁽ˡ⁾ = (W⁽ˡ⁺¹⁾ᵀ δ⁽ˡ⁺¹⁾) ⊙ f′(z⁽ˡ⁾)      ∂L/∂W⁽ˡ⁾ = δ⁽ˡ⁾ a⁽ˡ⁻¹⁾ᵀ');

  callout(main, '', 'In biology', 'The same building block (linear layer + non-linearity) is inside nearly every deep model in omics. <b>scVI</b> uses small MLPs as its encoder and decoder to map 2,000 highly variable genes to a 10–30-dimensional latent space. Cell-type classifiers such as <b>CellTypist</b> (logistic regression) and its MLP variants use the zero-hidden-layer versus hidden-layer trade-off shown here. Drug-response models on bulk RNA-seq often use MLPs too. With a few hundred patients rather than millions of cells, the overfitting gap between the two loss curves is the main risk.');
  callout(main, 'warn', 'Pitfalls', `<ul>
    <li>More layers and units are not automatically better. With few samples (typical for patient-level omics), a big network memorises the training set. Watch the test curve, not the training curve.</li>
    <li>Unscaled inputs (raw counts next to log-fold-changes) make training slow and unstable. <a href="#/concepts/scaling">Standardise</a> or log-transform first.</li>
    <li>ReLU units can "die": if a unit's input is negative for every training point, its gradient is zero and it never recovers. Dead units show up here as flat single-colour tiles.</li>
    <li>Results depend on the random initialisation. Report variation over several seeds, not one lucky run.</li>
  </ul>`);
  seeAlso(main, [['concepts', 'activations', 'Activation functions'], ['concepts', 'backprop', 'Backpropagation'], ['concepts', 'gradient-descent', 'Gradient descent'], ['concepts', 'logistic-regression', 'Logistic regression'], ['concepts', 'training-tricks', 'Dropout, batch norm, early stopping']]);
}
