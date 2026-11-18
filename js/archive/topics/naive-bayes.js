// Naive Bayes: model each feature separately per class, multiply the evidence. Fails when features are correlated.
import { header, figure, select, checkbox, stats, section, steps, callout, math, legend, seeAlso, tip, untip, el } from '../../lib/ui.js';
import { frame, decisionCanvas, cat, css, ink, ink2, muted } from '../../lib/plot.js';
import { rng, sigmoid } from '../../lib/data.js';
const d3 = window.d3;

// Draw n points from a 2D Gaussian with means m, sds s, correlation rho.
function gauss2(r, n, m, s, rho, label) {
  return d3.range(n).map(() => {
    const z1 = r.normal(), z2 = r.normal();
    return { x: m[0] + s[0] * z1, y: m[1] + s[1] * (rho * z1 + Math.sqrt(1 - rho * rho) * z2), label };
  });
}
const DATA = {
  indep: {
    name: 'Independent features (assumption holds)',
    make: () => { const r = rng(31); return [...gauss2(r, 110, [-0.45, -0.3], [0.26, 0.42], 0, 0), ...gauss2(r, 110, [0.4, 0.35], [0.42, 0.22], 0, 1)]; },
  },
  corr: {
    name: 'Correlated features (assumption fails)',
    make: () => { const r = rng(32); return [...gauss2(r, 110, [-0.25, 0], [0.45, 0.45], 0.92, 0), ...gauss2(r, 110, [0.25, 0], [0.45, 0.45], 0.92, 1)]; },
  },
};
const npdf = (x, m, s) => Math.exp(-0.5 * ((x - m) / s) ** 2) / (s * Math.sqrt(2 * Math.PI));
const lnpdf = (x, m, s) => -0.5 * ((x - m) / s) ** 2 - Math.log(s) - 0.5 * Math.log(2 * Math.PI);

function fitNB(pts) {
  return [0, 1].map((c) => {
    const P = pts.filter((p) => p.label === c), mx = d3.mean(P, (p) => p.x), my = d3.mean(P, (p) => p.y);
    return { prior: P.length / pts.length, mx, my, sx: d3.deviation(P, (p) => p.x), sy: d3.deviation(P, (p) => p.y),
      // full covariance, for the comparison model (QDA)
      cxx: d3.mean(P, (p) => (p.x - mx) ** 2), cyy: d3.mean(P, (p) => (p.y - my) ** 2), cxy: d3.mean(P, (p) => (p.x - mx) * (p.y - my)) };
  });
}
// Per-feature log-likelihood ratios: the "evidence" each feature contributes towards class 1.
function evidence(M, x, y) {
  const [a, b] = M;
  return { prior: Math.log(b.prior / a.prior), f1: lnpdf(x, b.mx, b.sx) - lnpdf(x, a.mx, a.sx), f2: lnpdf(y, b.my, b.sy) - lnpdf(y, a.my, a.sy) };
}
const nbProb = (M, x, y) => { const e = evidence(M, x, y); return sigmoid(e.prior + e.f1 + e.f2); };
function qdaLog(c, x, y) {
  const det = c.cxx * c.cyy - c.cxy ** 2, dx = x - c.mx, dy = y - c.my;
  return Math.log(c.prior) - 0.5 * Math.log(det) - 0.5 * (c.cyy * dx * dx - 2 * c.cxy * dx * dy + c.cxx * dy * dy) / det;
}
const qdaProb = (M, x, y) => sigmoid(qdaLog(M[1], x, y) - qdaLog(M[0], x, y));

export default async function render(main) {
  header(main, 'Naive Bayes',
    'Learn, for each class, how each feature is distributed <b>on its own</b>. To classify, multiply the per-feature evidence together (add it, on the log scale) with the class prior. The "naive" part is pretending the features are independent within each class.');

  const fig = figure(main, 'One bell curve per class per feature, multiplied together',
    'Centre: the data, shaded by the naive Bayes probability of class 1 (orange). Dashed ellipses are the fitted per-class Gaussians (1 and 2 standard deviations); they are always axis-aligned, because naive Bayes has no correlation term. Top and right: the fitted 1D densities for each feature, which are all the model knows. <b>Click a point</b> to see how its evidence adds up on the right.');
  let key = 'indep', pts = DATA[key].make(), M = fitNB(pts), sel = null, showQ = false;
  select(fig.controls, { label: 'Data', value: key, options: Object.entries(DATA).map(([k, v]) => [k, v.name]), onChange: (k) => { key = k; pts = DATA[k].make(); M = fitNB(pts); sel = null; paint(); } });
  checkbox(fig.controls, { label: 'show full-covariance Gaussian boundary (QDA)', checked: false, onChange: (c) => { showQ = c; paint(); } });
  const readout = stats(fig.root);

  const row = el('div', { class: 'fig-row' }); fig.body.append(row);
  const left = el('div'), right = el('div', { style: 'flex: 0 1 380px' }); row.append(left, right);
  legend(fig.body, [['class 0', cat(0)], ['class 1', cat(1)], ['naive Bayes boundary (p = 0.5)', ink()], ['QDA boundary (uses correlation)', cat(2)]]);

  const D = [-1.5, 1.5];
  const F = frame(left, { width: 560, height: 540, x: D, y: D, margin: { top: 90, right: 96, left: 50, bottom: 42 }, xLabel: 'feature 1 (gene A, log-expr.)', yLabel: 'feature 2 (gene B, log-expr.)', grid: false });
  const surf = decisionCanvas(F, (x, y) => nbProb(M, x, y), { res: 4, alpha: 0.32 });
  F.g.append('clipPath').attr('id', 'nb-clip').append('rect').attr('width', F.w).attr('height', F.h);
  const ellL = F.layer('ell').attr('clip-path', 'url(#nb-clip)'), bndL = F.layer('bnd').attr('clip-path', 'url(#nb-clip)'), ptL = F.layer('pts');
  const topL = F.g.append('g').attr('transform', 'translate(0,-8)'), rightL = F.g.append('g').attr('transform', `translate(${F.w + 8},0)`);
  F.svg.on('click', (e) => { if (e.target.tagName !== 'circle') { sel = null; paintSel(); } });

  const E = frame(right, { width: 380, height: 320, x: [-6, 6], y: [0, 4], margin: { left: 92, bottom: 42 }, xLabel: '← evidence for class 0 · evidence for class 1 →', axes: false, grid: false });
  E.g.append('g').attr('class', 'axis x').attr('transform', `translate(0,${E.h})`);
  const eL = E.layer('bars');
  const evNote = el('div', { class: 'fig-caption' }); right.append(evNote);

  function boundaryPaths(fn) {
    const n = 120, vals = new Array(n * n);
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) vals[j * n + i] = fn(D[0] + (D[1] - D[0]) * i / (n - 1), D[1] - (D[1] - D[0]) * j / (n - 1));
    const proj = d3.geoTransform({ point(px, py) { this.stream.point(F.x(D[0] + (D[1] - D[0]) * px / (n - 1)), F.y(D[1] - (D[1] - D[0]) * py / (n - 1))); } });
    return d3.geoPath(proj)(d3.contours().size([n, n]).thresholds([0.5])(vals)[0]);
  }
  function paint() {
    surf.draw((x, y) => nbProb(M, x, y));
    const ells = M.flatMap((c, i) => [1, 2].map((k) => ({ c, i, k })));
    ellL.selectAll('ellipse').data(ells).join('ellipse').attr('cx', (d) => F.x(d.c.mx)).attr('cy', (d) => F.y(d.c.my))
      .attr('rx', (d) => F.x(d.k * d.c.sx) - F.x(0)).attr('ry', (d) => F.y(0) - F.y(d.k * d.c.sy))
      .attr('fill', 'none').attr('stroke', (d) => cat(d.i)).attr('stroke-width', 1.6).attr('stroke-dasharray', '5 4');
    const bs = [{ d: boundaryPaths((x, y) => nbProb(M, x, y)), c: ink() }];
    if (showQ) bs.push({ d: boundaryPaths((x, y) => qdaProb(M, x, y)), c: cat(2) });
    bndL.selectAll('path').data(bs).join('path').attr('d', (d) => d.d).attr('fill', 'none').attr('stroke', (d) => d.c).attr('stroke-width', 2.2);
    ptL.selectAll('circle').data(pts).join('circle').attr('cx', (d) => F.x(d.x)).attr('cy', (d) => F.y(d.y)).attr('r', 4.3)
      .attr('fill', (d) => cat(d.label)).attr('stroke', css('--surface')).attr('stroke-width', 1.1).style('cursor', 'pointer')
      .on('click', (e, d) => { sel = d; paintSel(); })
      .on('mousemove', (e, d) => tip(e, `class ${d.label}<br>P(class 1) naive Bayes = ${nbProb(M, d.x, d.y).toFixed(3)}<br>P(class 1) QDA = ${qdaProb(M, d.x, d.y).toFixed(3)}`)).on('mouseleave', untip);
    // marginal densities
    const xs = d3.range(D[0], D[1] + 1e-9, 0.02);
    const top = M.map((c, i) => ({ i, pts: xs.map((v) => [v, npdf(v, c.mx, c.sx)]) })), side = M.map((c, i) => ({ i, pts: xs.map((v) => [v, npdf(v, c.my, c.sy)]) }));
    const ymax = d3.max([...top, ...side], (d) => d3.max(d.pts, (p) => p[1]));
    const hy = d3.scaleLinear().domain([0, ymax]).range([0, 72]);
    topL.selectAll('path').data(top).join('path').attr('fill', (d) => cat(d.i)).attr('fill-opacity', 0.18).attr('stroke', (d) => cat(d.i)).attr('stroke-width', 2)
      .attr('d', (d) => d3.area().x((p) => F.x(p[0])).y0(0).y1((p) => -hy(p[1]))(d.pts));
    rightL.selectAll('path').data(side).join('path').attr('fill', (d) => cat(d.i)).attr('fill-opacity', 0.18).attr('stroke', (d) => cat(d.i)).attr('stroke-width', 2)
      .attr('d', (d) => d3.area().y((p) => F.y(p[0])).x0(0).x1((p) => hy(p[1]))(d.pts));
    topL.selectAll('text').data([0]).join('text').attr('x', 4).attr('y', -58).attr('font-size', 11.5).attr('fill', ink2()).text('P(feature 1 | class)');
    rightL.selectAll('text').data([0]).join('text').attr('transform', `translate(78,${F.h - 4}) rotate(-90)`).attr('font-size', 11.5).attr('fill', ink2()).text('P(feature 2 | class)');
    const acc = (fn) => d3.mean(pts, (p) => (fn(M, p.x, p.y) >= 0.5) === (p.label === 1));
    readout.set({ 'naive Bayes accuracy': (100 * acc(nbProb)).toFixed(1) + '%', 'QDA accuracy (full covariance)': (100 * acc(qdaProb)).toFixed(1) + '%', 'within-class correlation (class 0, class 1)': `${(M[0].cxy / Math.sqrt(M[0].cxx * M[0].cyy)).toFixed(2)}, ${(M[1].cxy / Math.sqrt(M[1].cxx * M[1].cyy)).toFixed(2)}` });
    // default selection: in the correlated case, a class-1 point that naive Bayes gets wrong but QDA gets right
    if (!sel) sel = key === 'corr' ? (pts.find((p) => p.label === 1 && nbProb(M, p.x, p.y) < 0.4 && qdaProb(M, p.x, p.y) > 0.9) || pts[150])
      : pts.find((p) => p.label === 1 && Math.abs(nbProb(M, p.x, p.y) - 0.8) < 0.1) || pts[150];
    paintSel();
  }
  function paintSel() {
    ptL.selectAll('circle').attr('r', (d) => (d === sel ? 7 : 4.3)).attr('stroke', (d) => (d === sel ? ink() : css('--surface'))).attr('stroke-width', (d) => (d === sel ? 2 : 1.1));
    if (!sel) { eL.selectAll('*').remove(); evNote.innerHTML = 'Click a point to see its evidence.'; return; }
    const e = evidence(M, sel.x, sel.y), tot = e.prior + e.f1 + e.f2;
    const rows = [['prior', e.prior], ['feature 1', e.f1], ['feature 2', e.f2], ['total', tot]];
    const m = Math.max(3, d3.max(rows, (r) => Math.abs(r[1])) * 1.1);
    E.x.domain([-m, m]); E.g.select('.axis.x').call(d3.axisBottom(E.x).ticks(5));
    const by = d3.scaleBand().domain(rows.map((r) => r[0])).range([0, E.h]).padding(0.3);
    eL.selectAll('line.zero').data([0]).join('line').attr('class', 'zero').attr('x1', E.x(0)).attr('x2', E.x(0)).attr('y1', 0).attr('y2', E.h).attr('stroke', muted());
    eL.selectAll('rect').data(rows).join('rect').attr('x', (r) => E.x(Math.min(0, r[1]))).attr('width', (r) => Math.abs(E.x(r[1]) - E.x(0)))
      .attr('y', (r) => by(r[0])).attr('height', by.bandwidth()).attr('rx', 3).attr('fill', (r) => (r[1] >= 0 ? cat(1) : cat(0))).attr('fill-opacity', (r) => (r[0] === 'total' ? 1 : 0.65));
    eL.selectAll('text.lab').data(rows).join('text').attr('class', 'lab').attr('x', -8).attr('y', (r) => by(r[0]) + by.bandwidth() / 2 + 4).attr('text-anchor', 'end').attr('font-size', 12).attr('fill', ink2())
      .attr('font-weight', (r) => (r[0] === 'total' ? 700 : 400)).text((r) => r[0]);
    eL.selectAll('text.val').data(rows).join('text').attr('class', 'val').attr('font-size', 11.5).attr('fill', ink())
      .attr('x', (r) => E.x(r[1]) + (r[1] >= 0 ? 4 : -4)).attr('text-anchor', (r) => (r[1] >= 0 ? 'start' : 'end')).attr('y', (r) => by(r[0]) + by.bandwidth() / 2 + 4).text((r) => r[1].toFixed(2));
    evNote.innerHTML = `Log-likelihood ratios for the selected point (true class ${sel.label}). Bars add up: total log-odds ${tot.toFixed(2)} → P(class 1) = ${sigmoid(tot).toFixed(3)}. With QDA it would be ${qdaProb(M, sel.x, sel.y).toFixed(3)}.`;
  }
  paint();

  // numbers for the prose, computed from the same seeded data
  const acc = (k, fn) => { const P = DATA[k].make(), MM = fitNB(P); return (100 * d3.mean(P, (p) => (fn(MM, p.x, p.y) >= 0.5) === (p.label === 1))).toFixed(1) + '%'; };
  section(main, 'What to notice', `<ul>
    <li>On <b>independent features</b> the ellipses fit the clouds well and the boundary curves (because the two classes have different spreads). Naive Bayes gets ${acc('indep', nbProb)}, essentially the same as QDA (${acc('indep', qdaProb)}), which also models correlation. When the assumption holds, nothing is lost.</li>
    <li>Switch to <b>correlated features</b>. Each class is a thin diagonal cigar, but naive Bayes can only draw axis-aligned ellipses, so it sees two big overlapping blobs. Accuracy drops to ${acc('corr', nbProb)}.</li>
    <li>Look at the right-hand marginals in that case: feature 2 has the <i>same</i> distribution in both classes, so on its own it is useless and naive Bayes gives it almost zero evidence (click a few points). Yet jointly with feature 1 it is what separates the classes. Tick the QDA box: the full-covariance boundary runs along the diagonal and gets ${acc('corr', qdaProb)}.</li>
    <li>The evidence bars are additive. That is the whole model: a prior plus one independent vote per feature.</li>
  </ul>`);

  steps(main, 'How it works', [
    'For each class c, estimate the <b>prior</b> P(c): the fraction of training samples in that class.',
    'For each class and each feature separately, fit a simple distribution: here a Gaussian (mean and standard deviation). Other variants use counts (multinomial NB for word or k-mer counts) or presence/absence (Bernoulli NB).',
    'For a new sample, apply Bayes\' rule with the naive assumption: <code>P(c | x) ∝ P(c) · Π<sub>j</sub> P(x<sub>j</sub> | c)</code>.',
    'In practice add logs instead of multiplying (to avoid underflow), and pick the class with the larger total.',
    'Training is one pass over the data: no iterations, no tuning. It works with tiny samples and thousands of features.',
  ]);
  math(main, 'log P(c=1|x) − log P(c=0|x) = log[P(1)/P(0)] + Σⱼ log[ P(xⱼ|1) / P(xⱼ|0) ]\n                                   prior                one vote per feature');

  callout(main, '', 'In biology', 'The <b>RDP classifier</b>, the standard way to assign taxonomy to 16S rRNA amplicon reads (also in QIIME 2 as <code>classify-sklearn</code>, and DADA2\'s <code>assignTaxonomy</code>), is a multinomial naive Bayes over 8-mer counts. Naive Bayes on k-mers is also a quick baseline for classifying metagenomic reads and promoter sequences. It is a sensible baseline for few-sample, many-gene classification, as long as you remember that co-expressed genes violate independence (see Pitfalls).');
  callout(main, 'warn', 'Pitfalls', `<ul>
    <li><b>Correlated features are double-counted.</b> Twenty genes from the same co-expression module (say, an interferon signature) each cast an independent vote, so the same signal is counted twenty times and the probabilities become wildly overconfident, even if the class ranking is still reasonable.</li>
    <li>Predicted probabilities are usually badly <a href="#/archive/topics/calibration">calibrated</a> (pushed towards 0 and 1). Use the ranking, or recalibrate.</li>
    <li>A category never seen with a class gives probability zero and vetoes everything. Use smoothing (a pseudocount, "Laplace smoothing").</li>
    <li>Gaussian NB assumes bell-shaped features. Raw counts are not; log-transform them or use a count-based variant.</li>
  </ul>`);
  seeAlso(main, [['concepts', 'probability-mle', 'Likelihood & Bayes'], ['concepts', 'logistic-regression', 'Logistic regression'], ['concepts', 'gmm', 'Gaussian mixtures & EM'], ['concepts', 'calibration', 'Calibration']]);
}
