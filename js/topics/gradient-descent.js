// EXEMPLAR PAGE — the template other topic pages follow:
// header -> interactive figure(s) with controls + live readouts -> "What to notice" -> "How it works" steps
// -> optional math -> biology connection -> pitfalls -> see also.
import { header, figure, slider, select, button, stats, section, steps, callout, math, seeAlso, animate, legend } from '../lib/ui.js';
import { frame, cat, css, ink } from '../lib/plot.js';
const d3 = window.d3;

// Loss landscapes f(w1, w2) with analytic gradients.
const SURFACES = {
  bowl: { name: 'Stretched bowl (convex)', f: (a, b) => 0.5 * a * a + 4 * b * b, g: (a, b) => [a, 8 * b], start: [-2.6, 1.3], dom: [-3, 3] },
  valley: { name: 'Rosenbrock banana valley', f: (a, b) => (1 - a) ** 2 + 5 * (b - a * a) ** 2, g: (a, b) => [-2 * (1 - a) - 20 * a * (b - a * a), 10 * (b - a * a)], start: [-1.6, 2.2], dom: [-2.2, 2.4] },
  bumpy: { name: 'Two minima (non-convex)', f: (a, b) => 0.15 * (a * a + b * b) ** 2 - (a * a + b * b) * 0.6 + 0.4 * a + 0.3 * b * b, g: (a, b) => { const r = a * a + b * b; return [0.6 * r * a - 1.2 * a + 0.4, 0.6 * r * b - 1.2 * b + 0.6 * b]; }, start: [-0.2, 2.2], dom: [-2.8, 2.8] },
};

export default async function render(main) {
  header(main, 'Gradient descent',
    'Training a model means finding the parameters with the lowest loss. Gradient descent does this by repeatedly taking a small step <b>downhill</b>: the gradient says which direction is uphill, so you step the opposite way.');

  const fig = figure(main, 'A ball rolling down a loss landscape',
    'Contours are lines of equal loss (darker = lower). Each dot is one update. Click anywhere on the map to start from there.');
  let surf = SURFACES.bowl, lr = 0.1, path = [], stop = null;

  select(fig.controls, { label: 'Landscape', value: 'bowl', options: Object.entries(SURFACES).map(([k, v]) => [k, v.name]), onChange: (k) => { surf = SURFACES[k]; draw(); reset(surf.start); } });
  slider(fig.controls, { label: 'Learning rate η', min: -3, max: 0, step: 0.05, value: -1, fmt: (v) => (10 ** v).toFixed(3), onInput: (v) => { lr = 10 ** v; reset(path[0]); } });
  button(fig.controls, 'Run', () => run(), true);
  button(fig.controls, 'Step', () => { stepOnce(); paint(); });
  button(fig.controls, 'Reset', () => reset(path[0]));
  const readout = stats(fig.root);

  const row = d3.select(fig.body).append('div').attr('class', 'fig-row').node();
  const left = document.createElement('div'), right = document.createElement('div');
  row.append(left, right);

  let F, L;
  function draw() {
    left.innerHTML = ''; right.innerHTML = '';
    const [lo, hi] = surf.dom;
    F = frame(left, { width: 460, height: 420, x: [lo, hi], y: [lo, hi], xLabel: 'parameter w₁', yLabel: 'parameter w₂', grid: false });
    // contours
    const n = 90, vals = new Array(n * n);
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) vals[j * n + i] = surf.f(lo + (hi - lo) * i / (n - 1), hi - (hi - lo) * j / (n - 1));
    const ext = d3.extent(vals);
    const levels = d3.range(16).map((k) => ext[0] + (ext[1] - ext[0]) * (Math.pow(k / 16, 2.2)));
    const cs = d3.contours().size([n, n]).thresholds(levels)(vals);
    const shade = d3.scaleLinear().domain([0, levels.length]).range([css('--accent'), css('--surface-2')]);
    const proj = d3.geoTransform({ point(px, py) { this.stream.point(F.x(lo + (hi - lo) * px / (n - 1)), F.y(hi - (hi - lo) * py / (n - 1))); } });
    F.layer('contours').selectAll('path').data(cs).join('path').attr('d', d3.geoPath(proj))
      .attr('fill', (d, i) => shade(i)).attr('stroke', css('--surface')).attr('stroke-width', 0.6);
    F.layer('path'); F.layer('dots');
    F.svg.style('cursor', 'crosshair').on('click', (e) => {
      const [mx, my] = d3.pointer(e, F.g.node());
      reset([F.x.invert(mx), F.y.invert(my)]);
    });
    // loss curve panel
    L = frame(right, { width: 420, height: 420, x: [0, 60], y: [0, 1], xLabel: 'step', yLabel: 'loss (log scale)', yScale: d3.scaleLog() });
  }

  function reset(p) {
    if (stop) { stop(); stop = null; }
    path = [{ w: p.slice(), loss: surf.f(...p) }];
    paint();
  }
  function stepOnce() {
    const last = path[path.length - 1].w;
    if (path.length > 400) return false;
    const g = surf.g(...last);
    const w = [last[0] - lr * g[0], last[1] - lr * g[1]];
    const bad = !isFinite(w[0]) || Math.abs(w[0]) > 1e4;
    path.push({ w: bad ? last : w, loss: surf.f(...(bad ? last : w)), diverged: bad });
    return !bad && Math.hypot(...g) > 1e-4;
  }
  function run() {
    if (stop) stop();
    let acc = 0;
    stop = animate((dt) => { acc += dt; if (acc < 45) return true; acc = 0; const go = stepOnce(); paint(); return go && path.length < 400; });
  }
  function paint() {
    const [lo, hi] = surf.dom, clampv = (v) => Math.max(lo - 1, Math.min(hi + 1, v));
    const pts = path.map((p) => ({ x: clampv(p.w[0]), y: clampv(p.w[1]) }));
    F.layer('path').selectAll('path').data([pts]).join('path').attr('fill', 'none').attr('stroke', ink()).attr('stroke-width', 1.5)
      .attr('d', d3.line().x((d) => F.x(d.x)).y((d) => F.y(d.y)));
    F.layer('dots').selectAll('circle').data(pts).join('circle').attr('cx', (d) => F.x(d.x)).attr('cy', (d) => F.y(d.y))
      .attr('r', (d, i) => (i === pts.length - 1 ? 7 : 3.2)).attr('fill', (d, i) => (i === pts.length - 1 ? cat(1) : ink()))
      .attr('stroke', css('--surface')).attr('stroke-width', 1.5);
    // loss curve
    const losses = path.map((p, i) => ({ x: i, y: Math.max(1e-6, p.loss - minLoss() + 1e-3) }));
    L.x.domain([0, Math.max(20, path.length)]); L.y.domain([1e-3, Math.max(1, d3.max(losses, (d) => d.y))]);
    L.g.select('.axis.x').call(d3.axisBottom(L.x).ticks(5)); L.g.select('.axis.y').call(d3.axisLeft(L.y).ticks(4, '~g'));
    L.layer('c').selectAll('path').data([losses]).join('path').attr('fill', 'none').attr('stroke', cat(0)).attr('stroke-width', 2)
      .attr('d', d3.line().x((d) => L.x(d.x)).y((d) => L.y(d.y)));
    const last = path[path.length - 1];
    const g = surf.g(...last.w);
    readout.set({ step: path.length - 1, loss: last.loss.toFixed(4), '|gradient|': Math.hypot(...g).toExponential(2), status: last.diverged || last.loss > 1e3 ? 'diverged' : Math.hypot(...g) < 1e-3 ? 'converged' : 'moving' });
  }
  const minLoss = () => ({ bowl: 0, valley: 0, bumpy: -1.15 }[Object.keys(SURFACES).find((k) => SURFACES[k] === surf)]);

  draw(); reset(surf.start);

  section(main, 'What to notice', `<ul>
    <li><b>Too small η</b> (≈0.005): the ball crawls; thousands of steps to converge.</li>
    <li><b>Too large η</b> on the stretched bowl (≈0.25+): it zig-zags across the narrow direction and then <b>diverges</b>. The steep axis sets the speed limit for the whole problem.</li>
    <li>On the <b>banana valley</b> it quickly reaches the valley floor, then crawls along it. That's why plain GD is slow, and why <a href="#/concepts/optimizers">momentum and Adam</a> exist.</li>
    <li>On the <b>two-minima</b> surface, where you click decides which minimum you end up in. Neural-net losses are non-convex like this, so initialisation matters.</li>
  </ul>`);

  steps(main, 'How it works', [
    'Start with some parameters <code>w</code> (random, or zeros).',
    'Compute the loss on the training data, e.g. mean squared error.',
    'Compute the <b>gradient</b> ∇L(w): the vector of partial derivatives, pointing uphill. For neural nets this is what <a href="#/concepts/backprop">backpropagation</a> computes.',
    'Update: <code>w ← w − η·∇L(w)</code>. η (the learning rate) is the step size.',
    'Repeat until the loss stops improving. <b>Stochastic</b> GD uses a random mini-batch for each gradient: noisier, much cheaper, and the noise even helps escape bad regions.',
  ]);
  math(main, 'w(t+1) = w(t) − η ∇L(w(t))\nconverges on a convex quadratic iff  η < 2 / λ_max(Hessian)   (bowl: λ_max = 8 ⇒ η < 0.25)');

  callout(main, '', 'In biology', 'Every model you fit in omics (logistic regression on expression, the lasso for biomarker panels, scVI and Geneformer) is trained by some variant of gradient descent. When an scVI run gives a NaN loss, it is usually the "learning rate too large" picture above.');
  callout(main, 'warn', 'Pitfalls', '<ul><li>Unscaled features make the bowl stretched, so GD is slow and unstable. <a href="#/concepts/scaling">Standardise first.</a></li><li>A falling training loss says nothing about generalisation. Watch validation loss too.</li><li>Learning-rate schedules (warm-up, cosine decay) are standard in deep learning for exactly the trade-off shown here.</li></ul>');
  seeAlso(main, [['concepts', 'optimizers', 'Optimizers race'], ['concepts', 'loss-functions', 'Loss functions'], ['concepts', 'backprop', 'Backpropagation']]);
}
