// Linear regression: least squares as literally minimising the total area of residual squares.
import { header, figure, slider, select, button, checkbox, stats, section, steps, callout, math, legend, seeAlso, tip, untip, el } from '../../lib/ui.js';
import { frame, cat, css, ink, ink2, muted } from '../../lib/plot.js';
import { rng } from '../../lib/data.js';
const d3 = window.d3;

// Ordinary least squares for one predictor: slope = cov(x,y)/var(x), intercept = ȳ − slope·x̄.
function ols(pts) {
  const n = pts.length, mx = d3.mean(pts, (p) => p.x), my = d3.mean(pts, (p) => p.y);
  let sxy = 0, sxx = 0;
  pts.forEach((p) => { sxy += (p.x - mx) * (p.y - my); sxx += (p.x - mx) ** 2; });
  const b = sxx > 1e-12 ? sxy / sxx : 0;
  return { b, a: my - b * mx, n };
}
const sse = (pts, a, b) => pts.reduce((s, p) => s + (p.y - (a + b * p.x)) ** 2, 0);

function makePoints() {
  const r = rng(11);
  return d3.range(10).map((i) => { const x = 0.6 + i * 0.95 + r.normal(0, 0.25); return { x, y: 1.4 + 0.55 * x + r.normal(0, 0.9) }; });
}

export default async function render(main) {
  header(main, 'Linear regression',
    'Fit a straight line by making the residuals (the vertical gaps between points and line) small. "Least squares" is literal: draw each residual as a square, and the best line is the one with the <b>smallest total square area</b>.');

  // ---------------- Figure 1: squares + SSE surface ----------------
  const fig = figure(main, 'Least squares = smallest total area of the squares',
    'Left: drag any point. Each residual is drawn as a square whose side is the residual, so its area is the squared error. Right: total squared error (SSE) for every possible (slope, intercept). The cross is the least-squares (OLS) minimum and the ring is the line drawn on the left. In manual mode, drag the two handles on the line, use the sliders, or click the right panel.');

  let pts = makePoints(), mode = 'ols', showOls = false;
  let man = { a: 4.5, b: 0.05 };
  select(fig.controls, { label: 'Line', value: 'ols', options: [['ols', 'Least squares (OLS)'], ['manual', 'Manual: beat OLS yourself']], onChange: (v) => { mode = v; syncCtl(); paint(); } });
  const sB = slider(fig.controls, { label: 'slope', min: -0.5, max: 1.7, step: 0.01, value: man.b, fmt: (v) => v.toFixed(2), onInput: (v) => { man.b = v; mode = 'manual'; modeSel(); syncCtl(); paint(); } });
  const sA = slider(fig.controls, { label: 'intercept', min: -3, max: 6, step: 0.05, value: man.a, fmt: (v) => v.toFixed(2), onInput: (v) => { man.a = v; mode = 'manual'; modeSel(); syncCtl(); paint(); } });
  const showC = checkbox(fig.controls, { label: 'reveal OLS line', checked: false, onChange: (c) => { showOls = c; paint(); } });
  button(fig.controls, 'Add an outlier', () => { pts.push({ x: 8.6, y: 0.6 }); paint(); });
  button(fig.controls, 'Reset points', () => { pts = makePoints(); paint(); });
  const readout = stats(fig.root);
  const modeSelEl = fig.controls.querySelector('select');
  const modeSel = () => { modeSelEl.value = mode; };
  const syncCtl = () => { showC.parentNode.style.display = mode === 'manual' ? '' : 'none'; };

  const row = el('div', { class: 'fig-row' }); fig.body.append(row);
  const left = el('div'), right = el('div'); row.append(left, right);
  legend(fig.body, [['data point', cat(0)], ['residual square (area = squared error)', cat(1)], ['least-squares line', ink()], ['your manual line', cat(2)]]);

  const XD = [0, 10], YD = [-1, 10];
  const F = frame(left, { width: 480, height: 430, x: XD, y: YD, xLabel: 'x (e.g. gene A expression)', yLabel: 'y (e.g. gene B expression)' });
  F.g.append('clipPath').attr('id', 'lr-clip').append('rect').attr('width', F.w).attr('height', F.h);
  const sqL = F.layer('squares').attr('clip-path', 'url(#lr-clip)');
  const lineL = F.layer('line').attr('clip-path', 'url(#lr-clip)');
  const ptL = F.layer('pts'), hL = F.layer('handles');

  const BD = [-0.5, 1.7], AD = [-3, 6];
  const S = frame(right, { width: 480, height: 430, x: BD, y: AD, margin: { bottom: 46 }, xLabel: 'slope b', yLabel: 'intercept a', grid: false });
  const contL = S.layer('cont'), markL = S.layer('mark');
  S.svg.style('cursor', 'crosshair').on('click', (e) => {
    const [mx, my] = d3.pointer(e, S.g.node());
    if (mx < 0 || my < 0 || mx > S.w || my > S.h) return;
    man = { b: S.x.invert(mx), a: S.y.invert(my) }; mode = 'manual'; modeSel(); syncCtl(); paint();
  });

  // handles at fixed x positions define the manual line by two y-values
  const HX = [1, 9];
  const drag = d3.drag().on('drag', function (e, d) {
    d.x = Math.max(XD[0], Math.min(XD[1], F.x.invert(e.x)));
    d.y = Math.max(YD[0], Math.min(YD[1], F.y.invert(e.y)));
    paint();
  });
  const hdrag = d3.drag().on('drag', function (e, i) {
    const yv = F.y.invert(e.y);
    const y0 = man.a + man.b * HX[0], y1 = man.a + man.b * HX[1];
    const ys = i === 0 ? [yv, y1] : [y0, yv];
    man.b = (ys[1] - ys[0]) / (HX[1] - HX[0]); man.a = ys[0] - man.b * HX[0];
    mode = 'manual'; modeSel(); syncCtl(); paint();
  });

  let surfKey = '';
  function drawSurface(fit) {
    const n = 70, vals = new Array(n * n);
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      const b = BD[0] + (BD[1] - BD[0]) * i / (n - 1), a = AD[1] - (AD[1] - AD[0]) * j / (n - 1);
      vals[j * n + i] = Math.log(sse(pts, a, b));
    }
    const lo = Math.log(sse(pts, fit.a, fit.b)), hi = d3.max(vals);
    const levels = d3.range(18).map((k) => lo + (hi - lo) * Math.pow(k / 18, 1.6) + 1e-6);
    const cs = d3.contours().size([n, n]).thresholds(levels)(vals);
    const shade = d3.scaleLinear().domain([0, levels.length]).range([css('--accent'), css('--surface-2')]);
    const proj = d3.geoTransform({ point(px, py) { this.stream.point(S.x(BD[0] + (BD[1] - BD[0]) * px / (n - 1)), S.y(AD[1] - (AD[1] - AD[0]) * py / (n - 1))); } });
    contL.selectAll('path').data(cs).join('path').attr('d', d3.geoPath(proj))
      .attr('fill', (d, i) => shade(i)).attr('stroke', css('--surface')).attr('stroke-width', 0.6);
  }

  function paint() {
    const fit = ols(pts);
    const cur = mode === 'ols' ? fit : man;
    if (mode === 'manual') { sB.set(Math.max(-0.5, Math.min(1.7, man.b))); sA.set(Math.max(-3, Math.min(6, man.a))); }
    // residual squares: side = |residual| in y-pixels, extending away from the line on the right of the point
    const sq = pts.map((p) => { const yh = cur.a + cur.b * p.x, s = Math.abs(F.y(p.y) - F.y(yh)); return { p, yh, s, r: p.y - yh }; });
    sqL.selectAll('rect').data(sq).join('rect')
      .attr('x', (d) => F.x(d.p.x)).attr('y', (d) => Math.min(F.y(d.p.y), F.y(d.yh)))
      .attr('width', (d) => d.s).attr('height', (d) => d.s)
      .attr('fill', cat(1)).attr('fill-opacity', 0.18).attr('stroke', cat(1)).attr('stroke-width', 1.2)
      .on('mousemove', (e, d) => tip(e, `residual = ${d.r.toFixed(2)}<br>squared error = ${(d.r * d.r).toFixed(2)}`)).on('mouseleave', untip);
    const ln = (a, b) => [{ x: XD[0], y: a + b * XD[0] }, { x: XD[1], y: a + b * XD[1] }];
    const lines = [];
    if (mode === 'ols' || showOls) lines.push({ pts: ln(fit.a, fit.b), c: ink(), dash: mode === 'manual' ? '6 4' : null });
    if (mode === 'manual') lines.push({ pts: ln(man.a, man.b), c: cat(2), dash: null });
    lineL.selectAll('path').data(lines).join('path').attr('fill', 'none').attr('stroke', (d) => d.c).attr('stroke-width', 2)
      .attr('stroke-dasharray', (d) => d.dash).attr('d', (d) => d3.line().x((q) => F.x(q.x)).y((q) => F.y(q.y))(d.pts));
    ptL.selectAll('circle').data(pts).join('circle').attr('r', 6).attr('fill', cat(0)).attr('stroke', css('--surface')).attr('stroke-width', 1.5)
      .style('cursor', 'grab').attr('cx', (d) => F.x(d.x)).attr('cy', (d) => F.y(d.y)).call(drag)
      .on('mousemove', (e, d) => tip(e, `x = ${d.x.toFixed(2)}, y = ${d.y.toFixed(2)}`)).on('mouseleave', untip);
    hL.selectAll('rect').data(mode === 'manual' ? HX.map((_, i) => i) : []).join('rect')
      .attr('width', 12).attr('height', 12).attr('rx', 2).attr('fill', css('--surface')).attr('stroke', cat(2)).attr('stroke-width', 2.5)
      .attr('x', (i) => F.x(HX[i]) - 6).attr('y', (i) => F.y(man.a + man.b * HX[i]) - 6).style('cursor', 'ns-resize').call(hdrag);

    // surface (redraw only when points change)
    const key = pts.map((p) => p.x.toFixed(3) + ',' + p.y.toFixed(3)).join(';');
    if (key !== surfKey) { surfKey = key; drawSurface(fit); }
    const cl = (v, d) => Math.max(d[0], Math.min(d[1], v));
    const cross = d3.symbol().type(d3.symbolCross).size(140)();
    markL.selectAll('path.min').data([fit]).join('path').attr('class', 'min').attr('d', cross)
      .attr('transform', (d) => `translate(${S.x(cl(d.b, BD))},${S.y(cl(d.a, AD))}) rotate(45)`).attr('fill', ink()).attr('stroke', css('--surface')).attr('stroke-width', 1);
    markL.selectAll('circle').data(mode === 'manual' ? [man] : [fit]).join('circle').attr('r', 8).attr('fill', 'none')
      .attr('stroke', mode === 'manual' ? cat(2) : ink()).attr('stroke-width', 2.5)
      .attr('cx', (d) => S.x(cl(d.b, BD))).attr('cy', (d) => S.y(cl(d.a, AD)));

    const eOls = sse(pts, fit.a, fit.b), eCur = sse(pts, cur.a, cur.b);
    const my = d3.mean(pts, (p) => p.y), sst = d3.sum(pts, (p) => (p.y - my) ** 2);
    const o = { 'SSE (total square area)': eCur.toFixed(2) };
    if (mode === 'manual') { o['OLS SSE (the best possible)'] = eOls.toFixed(2); o['you vs OLS'] = eCur - eOls < 0.005 ? 'tied: you found it!' : `+${(100 * (eCur / eOls - 1)).toFixed(1)}% worse`; }
    o['slope'] = cur.b.toFixed(3); o['intercept'] = cur.a.toFixed(3);
    o['R² of this line'] = (1 - eCur / sst).toFixed(3);
    readout.set(o);
  }
  syncCtl(); paint();

  section(main, 'What to notice', `<ul>
    <li>Switch to <b>Manual</b> and try to beat OLS. You can tie it but never win: OLS is <i>defined</i> as the line with the smallest SSE. The readout shows how far behind you are.</li>
    <li>The bowl on the right is a long, tilted ellipse. Slope and intercept trade off: if you raise the slope you must lower the intercept to stay near the data, because all the x values are positive (far from x = 0). Centring x first would make the bowl round.</li>
    <li>Press <b>Add an outlier</b>. Its square is huge, so the line tilts towards it to shrink that one square, even though that enlarges all the others. Squaring makes large errors dominate. This is why least squares is <b>sensitive to outliers</b>.</li>
    <li>Drag a point at the far left or right up and down: the line swings a lot. Drag a point in the middle: the line mostly shifts up and down. Points at extreme x have high <b>leverage</b>.</li>
    <li>The SSE surface is a perfect bowl with exactly one bottom. That is why linear regression has a closed-form answer and needs no gradient descent.</li>
  </ul>`);

  steps(main, 'How it works', [
    'Choose a model: <code>ŷ = a + b·x</code> (intercept a, slope b). With many features it is <code>ŷ = Xβ</code>.',
    'For each point compute the <b>residual</b> <code>y − ŷ</code>: what the line gets wrong.',
    'Add up the squared residuals to get the <b>sum of squared errors (SSE)</b>, the total area of the squares.',
    'Pick (a, b) to minimise SSE. Because SSE is a quadratic bowl, setting its derivatives to zero gives the answer directly: <code>b = cov(x, y) / var(x)</code> and <code>a = ȳ − b·x̄</code>.',
    'Judge the fit with <b>R²</b> = 1 − SSE / SST, the fraction of the variance of y that the line explains (SST is the SSE of a flat line at ȳ).',
  ]);
  math(main, 'minimise  SSE(β) = Σᵢ (yᵢ − xᵢᵀβ)²  = ‖y − Xβ‖²\nnormal equations:  XᵀX β = Xᵀy   ⇒   β̂ = (XᵀX)⁻¹ Xᵀy\nSquared error is the maximum-likelihood choice if the noise is Gaussian.');

  callout(main, '', 'In biology', 'Linear models are the workhorse of omics statistics. <b>limma</b> fits one linear regression per gene (expression ~ condition + batch + covariates), and <b>DESeq2</b> and <b>edgeR</b> fit the count-data generalisation (a negative-binomial GLM). eQTL mapping (for example <b>Matrix eQTL</b> or <b>tensorQTL</b>) regresses each gene\'s expression on genotype dosage (0/1/2). Regressing out covariates such as library size or percent mitochondrial reads (<code>sc.pp.regress_out</code> in Scanpy) is the same least-squares fit, keeping the residuals.');
  callout(main, 'warn', 'Pitfalls', `<ul>
    <li><b>Outliers and high-leverage points</b> pull the line hard (try the outlier button). Look at residual plots; consider robust losses such as Huber.</li>
    <li>With more features than samples (p ≫ n, the usual case with 20,000 genes), XᵀX cannot be inverted and OLS interpolates the noise. Use <a href="#/archive/topics/regularization">ridge or lasso</a>.</li>
    <li>Correlated predictors (collinearity) make individual coefficients unstable, even when predictions are fine.</li>
    <li>A good fit is not causation, and a straight line can hide a curve. Always plot the data.</li>
  </ul>`);
  seeAlso(main, [['concepts', 'loss-functions', 'Loss functions'], ['concepts', 'regularization', 'Regularization (L1 / L2)'], ['concepts', 'gradient-descent', 'Gradient descent'], ['concepts', 'logistic-regression', 'Logistic regression']]);
}
