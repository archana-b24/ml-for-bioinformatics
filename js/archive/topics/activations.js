// Activation functions: shapes + derivatives, then a deep-stack demo of vanishing gradients (sigmoid vs ReLU).
import { header, figure, slider, select, button, stats, section, steps, callout, math, legend, seeAlso, tip, untip, el } from '../../lib/ui.js';
import { frame, cat, css, ink, muted } from '../../lib/plot.js';
import { rng, linspace } from '../../lib/data.js';
const d3 = window.d3;

const K = Math.sqrt(2 / Math.PI);
const sig = (x) => 1 / (1 + Math.exp(-x));
// Each activation: value f(x) and derivative d(x). Slot = categorical colour index, fixed.
export const ACTS = {
  relu: { name: 'ReLU', slot: 0, f: (x) => (x > 0 ? x : 0), d: (x) => (x > 0 ? 1 : 0) },
  leaky: { name: 'Leaky ReLU (0.1)', slot: 1, f: (x) => (x > 0 ? x : 0.1 * x), d: (x) => (x > 0 ? 1 : 0.1) },
  sigmoid: { name: 'Sigmoid', slot: 2, f: sig, d: (x) => sig(x) * (1 - sig(x)) },
  tanh: { name: 'tanh', slot: 3, f: Math.tanh, d: (x) => 1 - Math.tanh(x) ** 2 },
  gelu: { name: 'GELU', slot: 4, f: (x) => 0.5 * x * (1 + Math.tanh(K * (x + 0.044715 * x ** 3))),
    d: (x) => { const t = Math.tanh(K * (x + 0.044715 * x ** 3)); return 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * K * (1 + 3 * 0.044715 * x * x); } },
  silu: { name: 'SiLU / Swish', slot: 5, f: (x) => x * sig(x), d: (x) => { const s = sig(x); return s * (1 + x * (1 - s)); } },
};

/**
 * Gradient norms per layer of a deep MLP (width W, `depth` hidden layers, linear scalar output, squared-error loss)
 * for one random batch. Returns [{layer, g}] with g = ‖∂L/∂Wₗ‖ (Frobenius), layer 1 = closest to the input.
 */
export function layerGrads(actKey, depth, init, seed, W = 32, B = 64) {
  const act = ACTS[actKey], r = rng(seed);
  const sizes = [W, ...Array(depth).fill(W), 1];
  const L = sizes.length - 1;
  const Ws = [];
  for (let l = 0; l < L; l++) {
    const nin = sizes[l], nout = sizes[l + 1];
    const sd = init === 'he' ? Math.sqrt(2 / nin) : init === 'small' ? 0.05 : Math.sqrt(2 / (nin + nout));
    Ws.push(Float64Array.from({ length: nin * nout }, () => r.normal(0, sd)));
  }
  const G = Ws.map((w) => new Float64Array(w.length));
  const Z = sizes.map((n) => new Float64Array(n)), A = sizes.map((n) => new Float64Array(n)), Dl = sizes.map((n) => new Float64Array(n));
  for (let s = 0; s < B; s++) {
    for (let j = 0; j < W; j++) A[0][j] = r.normal();
    const target = r.normal();
    for (let l = 0; l < L; l++) {
      const nin = sizes[l], nout = sizes[l + 1], w = Ws[l];
      for (let i = 0; i < nout; i++) {
        let z = 0; for (let j = 0; j < nin; j++) z += w[i * nin + j] * A[l][j];
        Z[l + 1][i] = z; A[l + 1][i] = l === L - 1 ? z : act.f(z);
      }
    }
    Dl[L][0] = (A[L][0] - target) / B;
    for (let l = L - 1; l >= 0; l--) {
      const nin = sizes[l], nout = sizes[l + 1], w = Ws[l], g = G[l];
      for (let i = 0; i < nout; i++) { const d = Dl[l + 1][i]; for (let j = 0; j < nin; j++) g[i * nin + j] += d * A[l][j]; }
      if (l > 0) for (let j = 0; j < nin; j++) { let sum = 0; for (let i = 0; i < nout; i++) sum += w[i * nin + j] * Dl[l + 1][i]; Dl[l][j] = sum * act.d(Z[l][j]); }
    }
  }
  return G.map((g, l) => ({ layer: l + 1, g: Math.sqrt(g.reduce((s, v) => s + v * v, 0)), out: l === L - 1 }));
}

export default async function render(main) {
  header(main, 'Activation functions',
    'An activation is the small non-linear function each neuron applies to its weighted sum. Without it, a deep network collapses into a single linear map. Its <b>shape</b> decides what the network can represent, and its <b>slope</b> decides how well gradients survive the trip back through many layers.');

  // ---------------- Figure 1: shapes and derivatives ----------------
  const fig = figure(main, 'Six activations and their derivatives',
    'Left: f(x). Right: the derivative f′(x), the factor by which this unit scales a gradient passing backwards through it. Move the mouse over either panel to read the values at that x.');
  let focus = 'all', hx = 2;
  select(fig.controls, { label: 'Highlight', value: 'all', options: [['all', 'All six'], ...Object.entries(ACTS).map(([k, a]) => [k, a.name])], onChange: (v) => { focus = v; drawCurves(); } });
  const readout = stats(fig.root);
  legend(fig.body, Object.values(ACTS).map((a) => [a.name, cat(a.slot)]));
  const row = el('div', { class: 'fig-row' });
  const left = el('div'), right = el('div');
  row.append(left, right); fig.body.append(row);
  const xs = linspace(-4, 4, 321);
  const Ff = frame(left, { width: 470, height: 360, x: [-4, 4], y: [-1.5, 4], xLabel: 'input z', yLabel: 'f(z)' });
  const Fd = frame(right, { width: 470, height: 360, x: [-4, 4], y: [-0.2, 1.2], xLabel: 'input z', yLabel: "f′(z)  (gradient factor)" });
  for (const F of [Ff, Fd]) {
    F.layer('zero').append('line').attr('x1', F.x(0)).attr('x2', F.x(0)).attr('y1', 0).attr('y2', F.h).attr('stroke', css('--axis'));
    F.layer('zero').append('line').attr('x1', 0).attr('x2', F.w).attr('y1', F.y(0)).attr('y2', F.y(0)).attr('stroke', css('--axis'));
  }
  // reference: sigmoid's maximum slope 0.25
  Fd.layer('ref').append('line').attr('x1', 0).attr('x2', Fd.w).attr('y1', Fd.y(0.25)).attr('y2', Fd.y(0.25)).attr('stroke', muted()).attr('stroke-dasharray', '4 4');
  Fd.layer('ref').append('text').attr('x', 6).attr('y', Fd.y(0.25) - 5).attr('font-size', 11).attr('fill', muted()).text('0.25 = sigmoid\'s best case');

  function drawCurves() {
    for (const [F, key] of [[Ff, 'f'], [Fd, 'd']]) {
      const data = Object.entries(ACTS).map(([k, a]) => ({ k, a, pts: xs.map((x) => [x, a[key](x)]) }));
      F.layer('curves').selectAll('path').data(data).join('path')
        .attr('fill', 'none').attr('stroke', (d) => cat(d.a.slot))
        .attr('stroke-width', (d) => (focus === d.k ? 3 : 2))
        .attr('opacity', (d) => (focus === 'all' || focus === d.k ? 1 : 0.15))
        .attr('d', (d) => d3.line().x((p) => F.x(p[0])).y((p) => F.y(Math.max(-1.6, Math.min(F.y.domain()[1] + 0.2, p[1]))))(d.pts));
    }
    drawCursor();
  }
  function drawCursor() {
    for (const [F, key] of [[Ff, 'f'], [Fd, 'd']]) {
      const cur = F.layer('cursor');
      cur.selectAll('line').data([0]).join('line').attr('x1', F.x(hx)).attr('x2', F.x(hx)).attr('y1', 0).attr('y2', F.h).attr('stroke', ink()).attr('stroke-opacity', 0.35);
      const shown = Object.entries(ACTS).filter(([k]) => focus === 'all' || focus === k);
      cur.selectAll('circle').data(shown).join('circle').attr('cx', F.x(hx)).attr('cy', ([, a]) => F.y(a[key](hx))).attr('r', 4.5)
        .attr('fill', ([, a]) => cat(a.slot)).attr('stroke', css('--surface')).attr('stroke-width', 1.5);
    }
    const shown = Object.entries(ACTS).filter(([k]) => focus === 'all' || focus === k);
    const o = { z: hx.toFixed(2) };
    shown.forEach(([, a]) => { o[`${a.name}: f, f′`] = `${a.f(hx).toFixed(2)}, ${a.d(hx).toFixed(2)}`; });
    readout.set(o);
  }
  for (const F of [Ff, Fd]) {
    F.svg.on('mousemove', (e) => {
      const [mx] = d3.pointer(e, F.g.node());
      hx = Math.max(-4, Math.min(4, F.x.invert(mx)));
      drawCursor();
    });
  }
  drawCurves();

  // ---------------- Figure 2: vanishing gradients ----------------
  const fig2 = figure(main, 'Stack layers and watch the gradient shrink',
    'A fully connected network with 32 units per hidden layer is evaluated on one random batch of 64 inputs. Each bar is the size of the loss gradient for that layer\'s weights, ‖∂L/∂Wₗ‖, on a <b>log scale</b>. Layer 1 is closest to the input, and the rightmost bar is the output layer, where the gradient starts. A bar 10× shorter means that layer learns 10× more slowly.');
  let depth = 5, init = 'glorot', seed = 1, actA = 'sigmoid', actB = 'relu';
  select(fig2.controls, { label: 'Left', value: actA, options: Object.entries(ACTS).map(([k, a]) => [k, a.name]), onChange: (v) => { actA = v; drawBars(); } });
  select(fig2.controls, { label: 'Right', value: actB, options: Object.entries(ACTS).map(([k, a]) => [k, a.name]), onChange: (v) => { actB = v; drawBars(); } });
  slider(fig2.controls, { label: 'Hidden layers', min: 2, max: 15, value: depth, onInput: (v) => { depth = v; drawBars(); } });
  select(fig2.controls, { label: 'Weight init', value: init, options: [['glorot', 'Glorot / Xavier'], ['he', 'He (for ReLU)'], ['small', 'Small, N(0, 0.05²)']], onChange: (v) => { init = v; drawBars(); } });
  button(fig2.controls, 'New random weights', () => { seed += 1; drawBars(); });
  const readout2 = stats(fig2.root);
  const row2 = el('div', { class: 'fig-row' });
  const bl = el('div'), br = el('div');
  row2.append(bl, br); fig2.body.append(row2);

  function barPanel(parent, actKey, res, yDom) {
    parent.innerHTML = '';
    const a = ACTS[actKey];
    parent.append(el('div', { class: 'fig-caption', style: 'margin:0 0 4px;font-weight:600;color:var(--ink)' }, a.name));
    const labels = res.map((d) => (d.out ? 'out' : String(d.layer)));
    const F = frame(parent, { width: 470, height: 300, xScale: d3.scaleBand().padding(0.18), x: labels, yScale: d3.scaleLog(), y: yDom, xLabel: 'layer (input → output)', yLabel: '‖∂L/∂W‖ (log)' });
    const e0 = Math.round(Math.log10(yDom[0])), e1 = Math.round(Math.log10(yDom[1])), stepE = Math.ceil((e1 - e0) / 6);
    const decades = d3.range(e0, e1 + 1, stepE).map((e) => 10 ** e);
    const SUP = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
    const pow = (v) => '10' + String(Math.round(Math.log10(v))).split('').map((c) => SUP[c]).join('');
    F.g.select('.axis.y').call(d3.axisLeft(F.y).tickValues(decades).tickFormat(pow));
    F.g.select('.gridline').call(d3.axisLeft(F.y).tickValues(decades).tickSize(-F.w).tickFormat(''));
    const last = res[res.length - 1].g;
    F.layer('bars').selectAll('rect').data(res).join('rect')
      .attr('x', (d, i) => F.x(labels[i])).attr('width', F.x.bandwidth())
      .attr('y', (d) => F.y(Math.max(yDom[0], d.g))).attr('height', (d) => F.h - F.y(Math.max(yDom[0], d.g)))
      .attr('rx', 2).attr('fill', cat(a.slot))
      .on('mousemove', (e, d) => tip(e, `<b>${d.out ? 'output layer' : 'hidden layer ' + d.layer}</b><br>‖∂L/∂W‖ = ${d.g.toExponential(2)}<br>${(d.g / last).toPrecision(2)}× the output layer`))
      .on('mouseleave', untip);
  }
  function drawBars() {
    const ra = layerGrads(actA, depth, init, seed), rb = layerGrads(actB, depth, init, seed);
    const all = [...ra, ...rb].map((d) => d.g).filter((g) => g > 0);
    const hi = 10 ** Math.ceil(Math.log10(d3.max(all)) + 0.01);
    const lo = Math.max(1e-14, 10 ** Math.floor(Math.log10(d3.min(all)) - 0.01));
    barPanel(bl, actA, ra, [lo, hi]); barPanel(br, actB, rb, [lo, hi]);
    const ratio = (r) => r[0].g / r[r.length - 1].g;
    const fmt = (v) => (v >= 0.01 ? v.toPrecision(2) : v.toExponential(1));
    readout2.set({
      [`${ACTS[actA].name}: layer 1 ÷ output`]: fmt(ratio(ra)),
      [`${ACTS[actB].name}: layer 1 ÷ output`]: fmt(ratio(rb)),
      'hidden layers': depth,
    });
  }
  drawBars();

  section(main, 'What to notice', `<ul>
    <li>The sigmoid's slope is never more than <b>0.25</b> (at z = 0), and it falls towards 0 once |z| > 4. Each sigmoid layer multiplies the backward signal by at most 0.25. tanh peaks at 1, which is why it trains better than the sigmoid, but it still flattens out.</li>
    <li>In the second figure, with 5 hidden layers and Glorot initialisation, the sigmoid bars fall steadily from right to left. The first sigmoid layer receives a gradient about <b>190× smaller</b> than the output layer (ratio 0.0053). The ReLU bars stay roughly level, and its first layer even gets 4× more than the output layer. Drag to 15 layers: the sigmoid ratio collapses to about 1.2 × 10⁻¹⁰, while ReLU's stays near 2. The early layers of a deep sigmoid network barely learn.</li>
    <li>ReLU's slope is exactly 1 wherever the unit is active, so it does not shrink the gradient itself. Its <i>level</i> still depends on the weights. At 15 layers with Glorot init the ReLU bars are flat but small (about 0.002), because ReLU zeroes half its inputs and Glorot does not compensate, so the signal’s variance halves at every layer. Switch to <b>He</b> initialisation, which is designed for ReLU: the bars rise to about 0.3–1.4.</li>
    <li>Choose <b>Small</b> initialisation and every bar drops: with ReLU at 5 layers they are about 1,000× lower than with Glorot. Tiny weights shrink the signal on the way forward and on the way back, which is why initialisation schemes exist.</li>
    <li>Press <b>New random weights</b>: the exact ratios change from draw to draw, but sigmoid always loses orders of magnitude while ReLU stays within a small factor.</li>
    <li>ReLU's derivative is 0 for every negative input. A unit that only ever sees negative inputs gets no gradient and stops learning (a "dead" ReLU). Leaky ReLU, GELU and SiLU all have some slope there.</li>
    <li>GELU and SiLU are smooth versions of ReLU that dip slightly below zero (minimum ≈ −0.17 for GELU, ≈ −0.28 for SiLU). Transformers (GELU) and many modern CNNs (SiLU) use them.</li>
  </ul>`);

  steps(main, 'How it works', [
    'A neuron computes <code>z = w·x + b</code> and outputs <code>a = f(z)</code>. Composing linear maps gives another linear map, so without f, depth adds nothing.',
    'During <a href="#/archive/topics/backprop">backpropagation</a>, the gradient arriving at a layer is multiplied by <code>f′(z)</code> and by the weights before it is passed to the layer below.',
    'Over L layers these factors multiply. If each is typically below 1, the gradient shrinks geometrically (it <b>vanishes</b>). If each is above 1, it grows (it <b>explodes</b>).',
    'Sigmoid and tanh <b>saturate</b>: for large |z| they are flat, and f′ ≈ 0. ReLU does not saturate for positive z, which made training deep networks practical around 2010–2012.',
    'Modern networks also use good initialisation (Glorot for tanh, He for ReLU), normalisation layers and residual connections. All three keep these products close to 1.',
  ]);
  math(main, 'ReLU(z) = max(0, z)          sigmoid σ(z) = 1/(1+e^−z),  σ′ = σ(1−σ) ≤ 0.25\ntanh′(z) = 1 − tanh²(z) ≤ 1    GELU(z) = z·Φ(z)    SiLU(z) = z·σ(z)\n∂L/∂a⁽¹⁾ = ∏ₗ ( W⁽ˡ⁺¹⁾ᵀ · diag f′(z⁽ˡ⁾) ) · ∂L/∂a⁽ᴸ⁾   →  sigmoid: shrinks by ≲ 0.25·‖W‖ per layer');

  callout(main, '', 'In biology', 'The activation is a visible design choice in omics models. <b>scVI</b> and most single-cell VAEs use ReLU (or Leaky ReLU) hidden layers with batch norm. Their final decoder layer uses <b>softmax</b> (a multi-class sigmoid) to turn outputs into gene-expression proportions, which are then scaled by library size. Transformer foundation models such as <b>Geneformer</b> and <b>scGPT</b> use GELU in their feed-forward blocks. Outputs that must be positive, such as the mean or dispersion of a negative binomial for counts, are passed through <code>exp</code> or <b>softplus</b>, a smooth ReLU.');
  callout(main, 'warn', 'Pitfalls', `<ul>
    <li>Sigmoid or tanh in many stacked hidden layers gives vanishing gradients. Use ReLU-family activations inside the network. Keep the sigmoid for an output that must be a probability.</li>
    <li>A large learning rate can push many ReLU units into the dead zone at once. Training loss then plateaus and never recovers. Leaky ReLU or a lower learning rate helps.</li>
    <li>The output activation must match the target: sigmoid for binary, softmax for one-of-K classes, identity for regression, softplus/exp for positive values such as counts.</li>
  </ul>`);
  seeAlso(main, [['concepts', 'neural-network', 'Neural network playground'], ['concepts', 'backprop', 'Backpropagation'], ['concepts', 'training-tricks', 'Dropout, batch norm, early stopping'], ['concepts', 'transformer', 'Transformers']]);
}
