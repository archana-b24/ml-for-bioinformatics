// Convolutions: a 3×3 kernel slides over a paintable 12×12 image -> 10×10 feature map -> ReLU -> 2×2 max pool (5×5).
// Second figure: receptive-field growth and parameter sharing on a synthetic H&E-like patch.
import { header, figure, slider, select, button, checkbox, stats, section, steps, callout, math, seeAlso, animate, legend, tip, untip, el } from '../../lib/ui.js';
import { cat, css, ink, ink2, muted, diverging, sequential } from '../../lib/plot.js';
import { rng } from '../../lib/data.js';
const d3 = window.d3;

const N = 12, K = 3, M = N - K + 1, P = M / 2; // input 12, kernel 3, feature map 10, pooled 5
const PRESETS = {
  seven: ['............', '.#########..', '.#########..', '........##..', '.......##...', '......##....', '.....##.....', '....##......', '....##......', '...##.......', '...##.......', '............'],
  square: ['............', '............', '..########..', '..#......#..', '..#......#..', '..#......#..', '..#......#..', '..#......#..', '..#......#..', '..########..', '............', '............'],
  edge: Array.from({ length: N }, () => '......######'),
  diagonal: Array.from({ length: N }, (_, i) => Array.from({ length: N }, (_, j) => (j === i || j === i + 1 ? '#' : '.')).join('')),
  blank: Array.from({ length: N }, () => '............'),
};
const KERNELS = {
  vedge: { name: 'Vertical edge (Sobel)', k: [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]] },
  hedge: { name: 'Horizontal edge (Sobel)', k: [[-1, -2, -1], [0, 0, 0], [1, 2, 1]] },
  blur: { name: 'Blur (box average)', k: [[1, 1, 1], [1, 1, 1], [1, 1, 1]].map((r) => r.map((v) => v / 9)) },
  sharpen: { name: 'Sharpen', k: [[0, -1, 0], [-1, 5, -1], [0, -1, 0]] },
  identity: { name: 'Identity', k: [[0, 0, 0], [0, 1, 0], [0, 0, 0]] },
  custom: { name: 'Custom (click cells to edit)', k: [[0, 1, 0], [1, -4, 1], [0, 1, 0]] },
};
const fmt = (v) => { const r = Math.round(v * 100) / 100; const s = Number.isInteger(r) ? String(r) : r.toFixed(Math.abs(r) < 1 ? 2 : 1); return s.replace('-', '−').replace(/^0\./, '.').replace(/^−0\./, '−.'); };

export default async function render(main) {
  header(main, 'Convolutional neural networks',
    'A convolution slides one small grid of weights (a <b>kernel</b>) across the whole image and computes a weighted sum at every position. The same few weights look for the same pattern <b>everywhere</b>, so a CNN learns "edge" or "nucleus" once instead of separately for every pixel location.');

  /* ======================= Figure 1: slide, ReLU, pool ======================= */
  const fig = figure(main, 'Slide a 3×3 kernel over the image',
    'Click or drag on the input to paint pixels. The highlighted 3×3 window is multiplied element-by-element with the kernel and summed to give one feature-map cell. The 12×12 input gives a 10×10 feature map (no padding, stride 1). <b>ReLU</b> then sets negatives to 0 and <b>2×2 max pooling</b> keeps the largest value in each 2×2 block. Hover any output cell to see which input pixels it depends on (its <b>receptive field</b>).');
  let img = toGrid(PRESETS.seven), kname = 'vedge', ker = KERNELS.vedge.k.map((r) => r.slice());
  let pos = M * M; // how many feature-map cells are computed (animation cursor)
  let stop = null, hover = null;

  select(fig.controls, { label: 'Image', value: 'seven', options: [['seven', 'Digit 7'], ['square', 'Square outline'], ['edge', 'Vertical edge'], ['diagonal', 'Diagonal line'], ['blank', 'Blank (paint your own)']], onChange: (v) => { img = toGrid(PRESETS[v]); finish(); } });
  const kSel = select(fig.controls, { label: 'Kernel', value: 'vedge', options: Object.entries(KERNELS).map(([k, v]) => [k, v.name]), onChange: (v) => { kname = v; ker = KERNELS[v].k.map((r) => r.slice()); finish(); } });
  button(fig.controls, 'Play slide', () => play(), true);
  button(fig.controls, 'Step', () => { halt(); if (pos >= M * M) pos = 0; pos++; draw(); });
  button(fig.controls, 'Show all', () => finish());
  button(fig.controls, 'Shift image →', () => { img = img.map((r) => [r[N - 1], ...r.slice(0, N - 1)]); finish(); });
  const readout = stats(fig.root);

  const row1 = el('div', { class: 'fig-row', style: 'align-items:center' }), row2 = el('div', { class: 'fig-row', style: 'margin-top:10px' });
  fig.body.append(row1, row2);
  const pIn = el('div'), pK = el('div', { style: 'flex: 0.8 1 230px' }), pF = el('div'), pR = el('div'), pP = el('div');
  row1.append(pIn, pK, pF); row2.append(pR, pP, el('div', { style: 'flex: 0.8 1 230px' }));
  const CELL = 24;
  const mk = (parent, n, title, cell = CELL) => {
    const top = 22, s = n * cell;
    const svg = d3.select(parent).append('svg').attr('viewBox', `0 0 ${s + 4} ${s + top + 4}`).attr('width', s + 4).style('max-width', '100%').style('touch-action', 'none');
    svg.append('text').attr('x', 2).attr('y', 14).attr('font-size', 13).attr('font-weight', 600).attr('fill', ink()).text(title);
    const g = svg.append('g').attr('transform', `translate(2,${top})`);
    return { svg, g, cell, n };
  };
  const GI = mk(pIn, N, 'Input image (12×12)');
  const GF = mk(pF, M, 'Feature map (10×10)', 28.8);
  const GR = mk(pR, M, 'After ReLU', 28.8);
  const GP = mk(pP, P, 'After 2×2 max pool (5×5)', 57.6);
  const kSvg = d3.select(pK).append('svg').attr('viewBox', '0 0 230 150').attr('width', 230).style('max-width', '100%');
  const calc = el('div', { class: 'mono', style: 'font-size:12.5px; line-height:1.5; color:var(--ink-2); margin-top:4px' });
  pK.append(calc);
  legend(row2.lastChild, [['negative', diverging(1)(-1)], ['zero', diverging(1)(0)], ['positive', diverging(1)(1)]]);
  row2.lastChild.append(el('p', { class: 'fig-caption', html: 'Feature-map colours are on a symmetric scale; ReLU and pooled maps only have values ≥ 0. Note that, like PyTorch, this computes the kernel <i>without</i> flipping it (strictly, a cross-correlation).' }));

  // painting
  let painting = null;
  const cellAt = (evt) => { const [mx, my] = d3.pointer(evt, GI.g.node()); const j = Math.floor(mx / CELL), i = Math.floor(my / CELL); return i >= 0 && j >= 0 && i < N && j < N ? [i, j] : null; };
  GI.svg.style('cursor', 'crosshair')
    .on('pointerdown', (evt) => { const c = cellAt(evt); if (!c) return; painting = img[c[0]][c[1]] ? 0 : 1; img[c[0]][c[1]] = painting; finish(); })
    .on('pointermove', (evt) => { if (painting == null) return; const c = cellAt(evt); if (c && img[c[0]][c[1]] !== painting) { img[c[0]][c[1]] = painting; finish(); } });
  d3.select(window).on('pointerup.cnn', () => { painting = null; });

  const conv = () => { const F = []; for (let i = 0; i < M; i++) { F.push([]); for (let j = 0; j < M; j++) { let s = 0; for (let a = 0; a < K; a++) for (let b = 0; b < K; b++) s += img[i + a][j + b] * ker[a][b]; F[i].push(s); } } return F; };
  const pool = (R) => Array.from({ length: P }, (_, i) => Array.from({ length: P }, (_, j) => Math.max(R[2 * i][2 * j], R[2 * i][2 * j + 1], R[2 * i + 1][2 * j], R[2 * i + 1][2 * j + 1])));

  function halt() { if (stop) { stop(); stop = null; } }
  function finish() { halt(); pos = M * M; draw(); }
  function play() {
    halt(); pos = 0; let acc = 0;
    stop = animate((dt) => { acc += dt; if (acc < 70) return true; acc = 0; pos++; draw(); return pos < M * M; });
  }

  function grid(G, data, color, { done = () => true, labels = true, fs = 10.5 } = {}) {
    const c = G.cell, cells = [];
    data.forEach((r, i) => r.forEach((v, j) => cells.push({ i, j, v, on: done(i, j) })));
    G.g.selectAll('rect.cell').data(cells).join('rect').attr('class', 'cell').attr('x', (d) => d.j * c).attr('y', (d) => d.i * c)
      .attr('width', c - 1.5).attr('height', c - 1.5).attr('rx', 3)
      .attr('fill', (d) => (d.on ? color(d.v) : css('--surface'))).attr('stroke', css('--grid')).attr('stroke-width', 0.8);
    G.g.selectAll('text.cv').data(labels ? cells.filter((d) => d.on) : []).join('text').attr('class', 'cv')
      .attr('x', (d) => d.j * c + c / 2).attr('y', (d) => d.i * c + c / 2 + fs * 0.36).attr('text-anchor', 'middle').attr('font-size', fs)
      .attr('fill', ink()).attr('pointer-events', 'none').text((d) => fmt(d.v));
  }
  function box(G, i, j, h, w, layerName, color, sw = 2.5) {
    const c = G.cell;
    G.g.selectAll('rect.' + layerName).data(i == null ? [] : [0]).join('rect').attr('class', layerName).attr('pointer-events', 'none')
      .attr('x', j * c - 1).attr('y', i * c - 1).attr('width', w * c + 0.5).attr('height', h * c + 0.5).attr('rx', 4)
      .attr('fill', 'none').attr('stroke', color).attr('stroke-width', sw);
  }

  function draw() {
    const F = conv(), R = F.map((r) => r.map((v) => Math.max(0, v))), Pm = pool(R);
    const fmax = Math.max(1e-6, d3.max(F.flat(), Math.abs)), rmax = Math.max(1e-6, d3.max(R.flat()));
    const doneF = (i, j) => i * M + j < pos;
    const doneP = (i, j) => doneF(2 * i + 1, 2 * j + 1);
    grid(GI, img, sequential(0, 1), { labels: false });
    grid(GF, F, diverging(fmax), { done: doneF });
    grid(GR, R, sequential(0, rmax), { done: doneF });
    grid(GP, Pm, sequential(0, rmax), { done: doneP, fs: 14 });
    // current window or hovered receptive field
    let cur = null;
    if (hover) cur = hover;
    else if (pos > 0 && pos < M * M) cur = { kind: 'f', i: Math.floor((pos - 1) / M), j: (pos - 1) % M };
    else if (pos === M * M) cur = null;
    [GI, GF, GR, GP].forEach((G) => { box(G, null, null, 0, 0, 'win'); box(G, null, null, 0, 0, 'win2'); });
    const hl = cat(1);
    if (cur && cur.kind === 'f') { box(GI, cur.i, cur.j, 3, 3, 'win', hl); box(GF, cur.i, cur.j, 1, 1, 'win', hl); box(GR, cur.i, cur.j, 1, 1, 'win', hl); }
    if (cur && cur.kind === 'p') { box(GI, 2 * cur.i, 2 * cur.j, 4, 4, 'win', hl); box(GR, 2 * cur.i, 2 * cur.j, 2, 2, 'win2', hl); box(GF, 2 * cur.i, 2 * cur.j, 2, 2, 'win2', hl); box(GP, cur.i, cur.j, 1, 1, 'win', hl); }
    drawKernel(cur && cur.kind === 'f' ? cur : null, F);
    // hover handlers on outputs
    const onF = (kind) => (evt, d) => { hover = { kind, i: d.i, j: d.j }; draw(); tip(evt, kind === 'p' ? `pooled cell (${d.i},${d.j}) = <b>${fmt(d.v)}</b><br>max of a 2×2 block of the ReLU map<br>receptive field: 4×4 input pixels` : `feature (${d.i},${d.j}) = <b>${fmt(d.v)}</b><br>receptive field: 3×3 input pixels`); };
    const off = () => { hover = null; untip(); draw(); };
    GF.g.selectAll('rect.cell').on('mouseenter', onF('f')).on('mousemove', (evt, d) => tip(evt, `feature (${d.i},${d.j}) = <b>${fmt(d.v)}</b><br>receptive field: 3×3 input pixels`)).on('mouseleave', off);
    GR.g.selectAll('rect.cell').on('mouseenter', onF('f')).on('mouseleave', off);
    GP.g.selectAll('rect.cell').on('mouseenter', onF('p')).on('mouseleave', off);
    const nPos = F.flat().filter((v) => v > 1e-9).length;
    readout.set({ 'window position': `${Math.min(pos, M * M)} / ${M * M}`, 'max |feature|': fmt(fmax), 'cells > 0 after ReLU': `${nPos} / ${M * M}`, 'pooled max': fmt(d3.max(Pm.flat())), 'kernel weights': '9 (shared by all 100 positions)' });
  }

  function drawKernel(cur, F) {
    kSvg.selectAll('*').remove();
    kSvg.append('text').attr('x', 0).attr('y', 14).attr('font-size', 13).attr('font-weight', 600).attr('fill', ink()).text(kname === 'custom' ? 'Kernel (click: +1, shift-click: −1)' : 'Kernel (3×3)');
    const c = 38, kmax = Math.max(1e-6, d3.max(ker.flat(), Math.abs)), sc = diverging(kmax);
    ker.forEach((r, a) => r.forEach((v, b) => {
      const g = kSvg.append('g').attr('transform', `translate(${b * c},${22 + a * c})`).style('cursor', 'pointer');
      g.append('rect').attr('width', c - 2).attr('height', c - 2).attr('rx', 4).attr('fill', sc(v)).attr('stroke', css('--grid'));
      g.append('text').attr('x', c / 2 - 1).attr('y', c / 2 + 4).attr('text-anchor', 'middle').attr('font-size', 13).attr('fill', ink()).text(fmt(v));
      g.on('click', (evt) => {
        if (kname !== 'custom') { kname = 'custom'; kSel.value = 'custom'; }
        ker[a][b] = Math.max(-5, Math.min(5, Math.round(ker[a][b]) + (evt.shiftKey ? -1 : 1))); KERNELS.custom.k = ker.map((x) => x.slice()); finish();
      });
    }));
    const tx = 3 * c + 12;
    kSvg.append('text').attr('x', tx).attr('y', 50).attr('font-size', 12).attr('fill', ink2()).text('× window');
    kSvg.append('text').attr('x', tx).attr('y', 68).attr('font-size', 12).attr('fill', ink2()).text('then sum');
    kSvg.append('text').attr('x', tx).attr('y', 96).attr('font-size', 12).attr('fill', muted()).text('= 1 output');
    if (cur) {
      const terms = [];
      for (let a = 0; a < K; a++) for (let b = 0; b < K; b++) if (ker[a][b] !== 0 && img[cur.i + a][cur.j + b]) terms.push(`${fmt(ker[a][b])}·1`);
      calc.innerHTML = `cell (${cur.i},${cur.j}): ${terms.length ? terms.join(' + ') : 'no lit pixel meets a non-zero weight'}<br>= <b style="color:var(--ink)">${fmt(F[cur.i][cur.j])}</b>`;
    } else calc.innerHTML = 'Press <b>Play slide</b> or <b>Step</b>, or hover an output cell.';
  }
  draw();

  /* ======================= Figure 2: receptive field + parameter sharing ======================= */
  const fig2 = figure(main, 'Stacking layers: receptive fields grow, weights stay few',
    'A synthetic H&E-like patch, 128×128 pixels at 0.5 µm per pixel (roughly 20× magnification), so 64 µm across. The nested squares are the <b>receptive field</b> of one unit in the centre after each layer: the region of input pixels that can influence it. With plain 3×3 convolutions it grows by only 2 pixels per layer; pooling doubles the step size of every later layer, so the field grows much faster and deep units can "see" whole groups of cells.');
  let nL = 4, usePool = true, nF = 32;
  slider(fig2.controls, { label: '3×3 conv layers', min: 1, max: 8, step: 1, value: nL, onInput: (v) => { nL = v; draw2(); } });
  checkbox(fig2.controls, { label: '2×2 max pool after every 2nd conv', checked: true, onChange: (v) => { usePool = v; draw2(); } });
  slider(fig2.controls, { label: 'filters per layer', min: 8, max: 64, step: 8, value: nF, onInput: (v) => { nF = v; draw2(); } });
  const read2 = stats(fig2.root);
  const row3 = el('div', { class: 'fig-row' }); fig2.body.append(row3);
  const tWrap = el('div'), rfWrap = el('div');
  row3.append(tWrap, rfWrap);
  const S = 128, Z = 3.4;
  const svgT = d3.select(tWrap).append('svg').attr('viewBox', `0 0 ${S * Z} ${S * Z}`).attr('width', S * Z).style('max-width', '100%').attr('role', 'img').attr('aria-label', 'Synthetic tissue patch with receptive fields');
  // synthetic tissue: eosin background + nuclei (hematoxylin)
  const R0 = rng(11), nuclei = [];
  for (let t = 0; t < 400 && nuclei.length < 46; t++) {
    const n = { x: R0() * S, y: R0() * S, rx: 5 + R0() * 4, ry: 4 + R0() * 3, a: R0() * 180 };
    if (nuclei.every((m) => Math.hypot(m.x - n.x, m.y - n.y) > m.rx + n.rx + 2)) nuclei.push(n);
  }
  svgT.append('rect').attr('width', S * Z).attr('height', S * Z).attr('fill', cat(4)).attr('opacity', 0.28);
  const fib = rng(12);
  for (let k = 0; k < 14; k++) { const y0 = fib() * S; svgT.append('path').attr('d', `M0,${y0 * Z} C${S * Z * 0.3},${(y0 + fib() * 30 - 15) * Z} ${S * Z * 0.6},${(y0 + fib() * 30 - 15) * Z} ${S * Z},${(y0 + fib() * 20 - 10) * Z}`).attr('stroke', cat(4)).attr('stroke-width', 3).attr('fill', 'none').attr('opacity', 0.35); }
  svgT.selectAll('ellipse').data(nuclei).join('ellipse').attr('cx', (d) => d.x * Z).attr('cy', (d) => d.y * Z).attr('rx', (d) => d.rx * Z).attr('ry', (d) => d.ry * Z)
    .attr('transform', (d) => `rotate(${d.a},${d.x * Z},${d.y * Z})`).attr('fill', cat(6)).attr('opacity', 0.75);
  const rfL = svgT.append('g');
  const tbl = el('div');
  rfWrap.append(tbl);

  function layersSpec() { const L = []; for (let i = 0; i < nL; i++) { L.push({ t: 'conv', name: `conv ${i + 1}` }); if (usePool && i % 2 === 1 && i < nL - 1) L.push({ t: 'pool', name: 'pool' }); } return L; }
  function draw2() {
    let r = 1, j = 1, params = 0, cin = 3;
    const rows = [];
    layersSpec().forEach((l) => {
      if (l.t === 'conv') { r += 2 * j; params += 9 * cin * nF + nF; cin = nF; } else { r += j; j *= 2; }
      rows.push({ ...l, r, j });
    });
    rfL.selectAll('*').remove();
    const convRows = rows.filter((d) => d.t === 'conv');
    convRows.forEach((d, k) => {
      const s = Math.min(d.r, S), o = (S - s) / 2;
      rfL.append('rect').attr('x', o * Z).attr('y', o * Z).attr('width', s * Z).attr('height', s * Z).attr('fill', 'none')
        .attr('stroke', k === convRows.length - 1 ? ink() : ink2()).attr('stroke-width', k === convRows.length - 1 ? 2.5 : 1).attr('stroke-dasharray', k === convRows.length - 1 ? null : '3 3');
    });
    rfL.append('circle').attr('cx', S * Z / 2).attr('cy', S * Z / 2).attr('r', 4).attr('fill', cat(1)).attr('stroke', css('--surface')).attr('stroke-width', 1.5);
    tbl.innerHTML = `<table class="simple"><tr><th>layer</th><th>receptive field</th><th>in µm</th><th>stride so far</th></tr>${rows.map((d) => `<tr><td>${d.name}</td><td>${d.r}×${d.r} px</td><td>${(d.r * 0.5).toFixed(1)}</td><td>${d.j}</td></tr>`).join('')}</table>`;
    const last = rows[rows.length - 1];
    const dense = (S * S * 3) * (S * S * nF);
    read2.set({ 'final receptive field': `${last.r} px = ${(last.r * 0.5).toFixed(1)} µm`, 'conv weights (all layers)': params.toLocaleString('en-US'), ['one dense layer, 128×128×3 → 128×128×' + nF]: dense.toExponential(1).replace('e+', '×10^') });
  }
  draw2();

  section(main, 'What to notice', `<ul>
    <li>With the <b>vertical-edge</b> kernel on the digit 7, the strong responses sit along the left and right sides of the slanted stroke: positive where, reading left to right, the window passes from background (0) onto the stroke (1), and negative where it passes off the stroke. Along the middle of the horizontal top bar the response is 0; only its two ends light up. Switch to <b>horizontal edge</b> and the roles swap.</li>
    <li>A perfectly straight vertical edge gives exactly ±4 with the Sobel kernel (1 + 2 + 1 lit pixels on one side). Try the <b>Vertical edge</b> image: two columns of +4 (every window that straddles the edge) and 0 everywhere else, because inside a flat region the −1 −2 −1 and +1 +2 +1 cancel.</li>
    <li>Press <b>Shift image →</b>: the feature map shifts by one cell too, with identical values. That is <b>translation equivariance</b>, and it comes free from reusing the same 9 weights at every position.</li>
    <li><b>ReLU</b> throws away one sign. After it, the vertical-edge map only reports the left-hand edges of strokes (0 → 1 reading left to right); a real CNN learns a second kernel (the negated one) if it needs the other side.</li>
    <li><b>Max pooling</b> halves the resolution but keeps "was the pattern anywhere in this 2×2 block?". Hover a pooled cell: it depends on a 4×4 patch of the input, larger than the 3×3 of a single convolution.</li>
    <li>In the second figure, 8 conv layers without pooling see only 17×17 px (8.5 µm, about one nucleus). Turn pooling on and the last unit sees 68×68 px (34 µm), a neighbourhood of several cells.</li>
  </ul>`);

  steps(main, 'How it works', [
    'A <b>kernel</b> is a small grid of weights (3×3 here; in a colour image it is 3×3×channels). It is learned by <a href="#/archive/topics/backprop">backpropagation</a>, not designed by hand: the edge and blur kernels above are what early layers of trained CNNs tend to end up looking like.',
    'Slide it over the input. At each position, multiply overlapping values and sum (plus a bias). The grid of results is a <b>feature map</b>: "how strongly does this pattern appear here?"',
    'A layer has many kernels (e.g. 32 or 64), giving one feature map each. They are stacked as channels and become the input to the next layer.',
    'Apply a non-linearity (ReLU). Without it, stacked convolutions would collapse into one bigger linear filter.',
    '<b>Pool</b> (or use stride 2) to downsample. Deeper layers then work at coarser resolution with larger receptive fields, combining edges into textures, textures into nuclei and glands, and those into tissue architecture.',
    'At the end, average-pool each feature map to one number and feed the vector to a classifier or regression head.',
  ]);
  math(main, 'feature[i, j] = Σₐ Σ_b  input[i + a, j + b] · kernel[a, b]  + bias\noutput size  = (N − K + 2·padding) / stride + 1      (12 − 3 + 0)/1 + 1 = 10\nparameters per conv layer = K·K·C_in·C_out + C_out    (3·3·3·32 + 32 = 896)');

  callout(main, '', 'In biology', `CNNs are the workhorse of <b>computational pathology</b>. Whole-slide H&E images (often 100,000 × 100,000 px) are tiled into 224×224 or 256×256 patches at 20× (≈0.5 µm/px); a CNN or vision transformer embeds each patch, and the embeddings are aggregated per slide with attention-based multiple-instance learning to predict subtype, MSI status or survival. In spatial transcriptomics, tools such as <b>ST-Net</b> and its successors regress <b>Visium spot expression from the H&E patch under the spot</b>, and benchmarks such as HEST compare pathology foundation models (UNI, CONCH, Virchow) on that task. CNNs also do <b>nucleus segmentation</b> (StarDist, Cellpose, HoVer-Net) for imaging-based spatial data, and 1D convolutions over DNA sequence underlie models like Basenji and Enformer.`);
  callout(main, 'warn', 'Pitfalls', `<ul>
    <li><b>Stain and scanner batch effects:</b> a CNN can learn the hospital (colour, JPEG artefacts, pen marks) instead of the biology. Use stain normalisation or augmentation, and test on a different site.</li>
    <li><b>Patch-level leakage:</b> patches from the same slide or patient must never be split across train and test.</li>
    <li>The <b>receptive field</b> bounds what a unit can know. If the signal is tissue architecture over hundreds of µm, a model working on small high-magnification patches cannot see it.</li>
    <li>Edge effects: without padding, border pixels feed fewer outputs (the 12×12 input shrinks to 10×10). "Same" padding keeps the size but invents values at the border.</li>
  </ul>`);
  seeAlso(main, [['concepts', 'backprop', 'Backpropagation'], ['concepts', 'neural-network', 'Neural network playground'], ['concepts', 'activations', 'Activation functions'], ['frontier', 'he-to-expression', 'H&E → gene expression']]);
}

function toGrid(rows) { return rows.map((r) => [...r].map((ch) => (ch === '#' ? 1 : 0))); }
