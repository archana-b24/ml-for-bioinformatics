// Calibration: reliability diagrams, Platt scaling, isotonic regression, Brier score.
import { header, figure, slider, select, stats, section, steps, callout, math, legend, seeAlso, tip, untip, el } from '../lib/ui.js';
import { frame, cat, css, ink, muted } from '../lib/plot.js';
import { rng, sigmoid } from '../lib/data.js';
const d3 = window.d3;

const N = 1500, BINS = 10;
const logit = (p) => Math.log(p / (1 - p));
const clip = (p) => Math.min(1 - 1e-6, Math.max(1e-6, p));

/** Patients with a true risk; the label is a coin flip with that risk. */
function makeSet(seed) {
  const r = rng(seed);
  return d3.range(N).map(() => { const s = r.normal(-0.3, 1.5); const y = r() < sigmoid(s) ? 1 : 0; return { s, y }; });
}

/** Platt scaling: fit y ~ sigmoid(a * logit(p) + b) by Newton's method. */
function fitPlatt(ps, ys) {
  let a = 1, b = 0;
  for (let it = 0; it < 30; it++) {
    let g0 = 0, g1 = 0, h00 = 1e-9, h01 = 0, h11 = 1e-9;
    ps.forEach((p, i) => { const z = logit(p), q = sigmoid(a * z + b), e = q - ys[i], w = q * (1 - q); g0 += e * z; g1 += e; h00 += w * z * z; h01 += w * z; h11 += w; });
    const det = h00 * h11 - h01 * h01;
    const da = (h11 * g0 - h01 * g1) / det, db = (h00 * g1 - h01 * g0) / det;
    a -= da; b -= db;
    if (Math.abs(da) + Math.abs(db) < 1e-9) break;
  }
  return { a, b, f: (p) => sigmoid(a * logit(p) + b) };
}

/** Isotonic regression by pool-adjacent-violators; returns a monotone step function. */
function fitIsotonic(ps, ys) {
  const idx = d3.range(ps.length).sort((i, j) => ps[i] - ps[j]);
  const blocks = [];
  idx.forEach((i) => {
    blocks.push({ sum: ys[i], n: 1, lo: ps[i], hi: ps[i] });
    while (blocks.length > 1 && blocks[blocks.length - 2].sum / blocks[blocks.length - 2].n >= blocks[blocks.length - 1].sum / blocks[blocks.length - 1].n) {
      const b2 = blocks.pop(), b1 = blocks[blocks.length - 1];
      b1.sum += b2.sum; b1.n += b2.n; b1.hi = b2.hi;
    }
  });
  const steps = blocks.map((b) => ({ lo: b.lo, hi: b.hi, v: b.sum / b.n }));
  const f = (p) => {
    let lo = 0, hi = steps.length - 1;
    if (p <= steps[0].hi) return clip(steps[0].v);
    if (p >= steps[hi].lo) return clip(steps[hi].v);
    while (lo < hi) { const m = (lo + hi + 1) >> 1; if (steps[m].lo <= p) lo = m; else hi = m - 1; }
    // between blocks: interpolate linearly (as sklearn does)
    const s = steps[lo], t = steps[Math.min(lo + 1, steps.length - 1)];
    if (p <= s.hi || t === s) return clip(s.v);
    return clip(s.v + ((t.v - s.v) * (p - s.hi)) / (t.lo - s.hi));
  };
  return { steps, f };
}

function metrics(ps, ys) {
  const n = ps.length;
  const brier = d3.sum(ps, (p, i) => (p - ys[i]) ** 2) / n;
  const ll = -d3.sum(ps, (p, i) => (ys[i] ? Math.log(clip(p)) : Math.log(1 - clip(p)))) / n;
  const bins = d3.range(BINS).map(() => ({ n: 0, sp: 0, sy: 0 }));
  ps.forEach((p, i) => { const b = bins[Math.min(BINS - 1, Math.floor(p * BINS))]; b.n++; b.sp += p; b.sy += ys[i]; });
  bins.forEach((b, k) => { b.k = k; b.mp = b.n ? b.sp / b.n : null; b.obs = b.n ? b.sy / b.n : null; });
  const ece = d3.sum(bins, (b) => (b.n ? (b.n / n) * Math.abs(b.mp - b.obs) : 0));
  // AUC via rank-sum with ties averaged
  const order = d3.range(n).sort((i, j) => ps[i] - ps[j]);
  const rank = new Array(n);
  for (let i = 0; i < n;) { let j = i; while (j + 1 < n && ps[order[j + 1]] === ps[order[i]]) j++; for (let k = i; k <= j; k++) rank[order[k]] = (i + j) / 2 + 1; i = j + 1; }
  const npos = d3.sum(ys), nneg = n - npos;
  const auc = (d3.sum(ys.map((y, i) => (y ? rank[i] : 0))) - (npos * (npos + 1)) / 2) / (npos * nneg);
  return { brier, ll, ece, bins, auc };
}

export default async function render(main) {
  header(main, 'Calibration',
    'A classifier is <b>calibrated</b> if, among all the cases where it says 80%, about 80% really are positive. Ranking well (high AUC) and being calibrated are different things: you can fix calibration afterwards with a <b>monotone remapping</b> that leaves the ranking alone.');

  const cal = makeSet(11), test = makeSet(12);
  let T = 2.2, method = 'none';

  const fig = figure(main, 'Reliability diagram on held-out patients', '');
  slider(fig.controls, { label: 'Model overconfidence (temperature)', min: 0.4, max: 3, step: 0.05, value: T, fmt: (v) => v.toFixed(2) + '×', onInput: (v) => { T = v; update(); } });
  select(fig.controls, { label: 'Correction', value: 'none', options: [['none', 'None (raw model)'], ['platt', 'Platt scaling'], ['iso', 'Isotonic regression']], onChange: (v) => { method = v; update(); } });
  const readout = stats(fig.root);
  legend(fig.root, [['raw model', cat(0)], ['after correction', cat(1)], ['perfect calibration', muted()]]);
  const row = el('div', { class: 'fig-row' }); fig.body.append(row);
  const lDiv = el('div'), rDiv = el('div'); row.append(lDiv, rDiv);
  const R = frame(lDiv, { width: 480, height: 400, x: [0, 1], y: [0, 1], xLabel: 'predicted probability (bin mean)', yLabel: 'observed fraction positive', margin: { bottom: 90 } });
  // prediction histogram under the reliability diagram
  const histH = 40, histY = R.h + 48;
  const HG = R.g.append('g').attr('transform', `translate(0,${histY})`);
  HG.append('text').attr('x', 0).attr('y', -4).attr('font-size', 11).attr('fill', css('--ink-2')).text('how many test patients get each predicted probability');
  const M = frame(rDiv, { width: 480, height: 400, x: [0, 1], y: [0, 1], xLabel: 'raw model probability', yLabel: 'corrected probability', margin: { bottom: 90 } });

  // fixed diagonals
  [R, M].forEach((F) => F.layer('diag').append('line').attr('x1', F.x(0)).attr('y1', F.y(0)).attr('x2', F.x(1)).attr('y2', F.y(1))
    .attr('stroke', muted()).attr('stroke-width', 1.5).attr('stroke-dasharray', '5 4'));

  function update() {
    const predict = (d) => clip(sigmoid(T * d.s));
    const pc = cal.map(predict), yc = cal.map((d) => d.y);
    const pt = test.map(predict), yt = test.map((d) => d.y);
    const platt = fitPlatt(pc, yc), iso = fitIsotonic(pc, yc);
    const corr = method === 'platt' ? platt.f : method === 'iso' ? iso.f : null;
    const raw = metrics(pt, yt);
    const fixed = corr ? metrics(pt.map(corr), yt) : null;

    // reliability curves
    const curve = (m, key, color) => {
      const pts = m.bins.filter((b) => b.n >= 5);
      R.layer('c-' + key).selectAll('path').data([pts]).join('path').attr('fill', 'none').attr('stroke', color).attr('stroke-width', 2)
        .attr('d', d3.line().x((b) => R.x(b.mp)).y((b) => R.y(b.obs)));
      R.layer('p-' + key).selectAll('circle').data(pts).join('circle').attr('cx', (b) => R.x(b.mp)).attr('cy', (b) => R.y(b.obs))
        .attr('r', (b) => 3 + Math.sqrt(b.n) / 4).attr('fill', color).attr('stroke', css('--surface')).attr('stroke-width', 1.5)
        .on('mousemove', (e, b) => tip(e, `bin ${(b.k / BINS).toFixed(1)}–${((b.k + 1) / BINS).toFixed(1)} · ${b.n} patients<br>mean predicted ${b.mp.toFixed(2)}<br>observed positive ${b.obs.toFixed(2)}`))
        .on('mouseleave', untip);
    };
    curve(raw, 'raw', cat(0));
    if (fixed) curve(fixed, 'fix', cat(1));
    else { R.layer('c-fix').selectAll('*').remove(); R.layer('p-fix').selectAll('*').remove(); }

    // gap bars for the raw model (what ECE averages)
    R.layer('gap').selectAll('line').data((fixed || raw).bins.filter((b) => b.n >= 5)).join('line')
      .attr('x1', (b) => R.x(b.mp)).attr('x2', (b) => R.x(b.mp)).attr('y1', (b) => R.y(b.mp)).attr('y2', (b) => R.y(b.obs))
      .attr('stroke', css('--bad')).attr('stroke-width', 1.5).attr('stroke-opacity', 0.6);

    // histogram of predictions
    const show = fixed ? test.map((d) => corr(predict(d))) : pt;
    const counts = d3.range(20).map((k) => show.filter((p) => Math.min(19, Math.floor(p * 20)) === k).length);
    const hy = d3.scaleLinear().domain([0, d3.max(counts)]).range([0, histH]);
    HG.selectAll('rect').data(counts).join('rect').attr('x', (c, k) => R.x(k / 20) + 0.5).attr('width', R.x(0.05) - R.x(0) - 1)
      .attr('y', (c) => histH - hy(c)).attr('height', (c) => hy(c)).attr('fill', fixed ? cat(1) : cat(0)).attr('rx', 1);

    // mapping panel
    const grid = d3.range(0.002, 0.999, 0.002);
    M.layer('platt').selectAll('path').data([grid]).join('path').attr('fill', 'none').attr('stroke', cat(1))
      .attr('stroke-width', method === 'platt' ? 2.5 : 1.2).attr('stroke-opacity', method === 'platt' ? 1 : 0.35)
      .attr('d', d3.line().x((p) => M.x(p)).y((p) => M.y(platt.f(p))));
    M.layer('iso').selectAll('path').data([grid]).join('path').attr('fill', 'none').attr('stroke', cat(2))
      .attr('stroke-width', method === 'iso' ? 2.5 : 1.2).attr('stroke-opacity', method === 'iso' ? 1 : 0.35)
      .attr('d', d3.line().curve(d3.curveStepAfter).x((p) => M.x(p)).y((p) => M.y(iso.f(p))));
    M.layer('lbl').selectAll('*').remove();
    M.layer('lbl').append('text').attr('x', 6).attr('y', 14).attr('font-size', 12).attr('fill', ink())
      .text(`Platt: sigmoid(${platt.a.toFixed(2)}·logit p ${platt.b < 0 ? '−' : '+'} ${Math.abs(platt.b).toFixed(2)})`);
    M.layer('lbl').append('text').attr('x', 6).attr('y', 30).attr('font-size', 12).attr('fill', ink()).text(`Isotonic: ${iso.steps.length} steps (fit on ${N} calibration patients)`);

    const f3 = (v) => v.toFixed(3);
    const arrow = (k) => (fixed ? `${f3(raw[k])} → ${f3(fixed[k])}` : f3(raw[k]));
    readout.set({ 'Brier score (lower is better)': arrow('brier'), 'ECE (mean |gap|)': arrow('ece'), 'log-loss': arrow('ll'), 'AUC (ranking)': arrow('auc') });
    fig.caption(`Left: ${N} test patients binned by predicted probability; each dot is a bin (size = count), red ticks are the gaps that ECE averages. ${!raw.bins[8].n ? '' : T > 1.05 ? `Points <b>below</b> the diagonal at high p mean the model is <b>overconfident</b> (in the 0.8–0.9 bin it says ${raw.bins[8].mp.toFixed(2)}, but only ${raw.bins[8].obs.toFixed(2)} are positive).` : T < 0.95 ? `Points <b>above</b> the diagonal at high p mean the model is <b>underconfident</b> (in the 0.8–0.9 bin it says ${raw.bins[8].mp.toFixed(2)}, yet ${raw.bins[8].obs.toFixed(2)} are positive).` : 'At temperature 1× the model output is the true risk, so it is already calibrated up to sampling noise.'} Right: the two correction maps, each fit on a separate set of ${N} calibration patients.`);
  }
  update();

  section(main, 'What to notice', `<ul>
    <li>At the default (2.2×) the raw curve is an S that crosses the diagonal: predictions near 0 and 1 are too extreme. This is the signature of an <b>overconfident</b> model, typical of boosted trees and deep nets trained to near-zero training loss.</li>
    <li>Slide the temperature to <b>1×</b>: the curve hugs the diagonal and ECE drops to its noise floor. Below 1× the S flips (underconfident, predictions bunched toward the middle), which is typical of random forests and naive regularised models.</li>
    <li>Switch on <b>Platt scaling</b>. Here the distortion is exactly a stretch of the log-odds, so Platt's two-parameter logistic undoes it almost perfectly: its slope comes out close to 1/temperature.</li>
    <li><b>Isotonic regression</b> is a free-form staircase (16 steps here), so it can fix any monotone distortion, not just a log-odds stretch. It reaches the lowest ECE (0.024 at the default), but its Brier score and log-loss are a touch worse than Platt's because the steps carry sampling noise from the calibration set. With a few hundred calibration patients that noise grows quickly.</li>
    <li>Watch the <b>AUC</b> readout: Platt leaves it exactly unchanged because it never reorders patients. Isotonic can only create ties, so AUC barely moves. Calibration and discrimination are separate properties.</li>
  </ul>`);

  steps(main, 'How it works', [
    '<b>Reliability diagram:</b> bin held-out predictions (here 10 bins of width 0.1). For each bin, plot the mean predicted probability against the fraction that were actually positive. Calibrated means on the diagonal.',
    '<b>ECE</b> (expected calibration error) is the count-weighted average vertical gap. <b>Brier score</b> is the mean squared error of the probability against the 0/1 outcome; it rewards both calibration and sharpness.',
    'To correct, hold out a <b>calibration set</b> that the model never trained on.',
    '<b>Platt scaling:</b> fit a 1-feature logistic regression of the label on logit(p). Two parameters, so it needs little data; it assumes the distortion is sigmoid-shaped.',
    '<b>Isotonic regression:</b> fit the best non-decreasing step function from p to the label (pool-adjacent-violators). More flexible, needs more data (rule of thumb: ≥1,000 calibration cases).',
    'Apply the fitted map to new predictions. It is monotone, so the ranking (and so AUC) is preserved.',
  ]);
  math(main, 'Brier = (1/n) Σ (pᵢ − yᵢ)²          ECE = Σ_bins (n_b / n) · |mean p_b − mean y_b|\nPlatt:  p′ = σ(a · logit p + b)       temperature scaling is the special case b = 0');

  callout(main, '', 'In biology', `<ul>
    <li><b>Clinical risk scores</b> from expression signatures (e.g. a response-to-immunotherapy classifier on bulk RNA-seq) are read as probabilities by clinicians, so calibration matters as much as AUC. TRIPOD reporting guidelines require a calibration plot.</li>
    <li><b>Transferring a model to a new cohort or platform</b> (microarray to RNA-seq, a new hospital) usually keeps the ranking roughly intact but shifts calibration. Recalibrating the intercept and slope on local data is the standard fix.</li>
    <li><b>Cell-type annotation</b> tools (CellTypist, scANVI) output per-cell probabilities. These are often overconfident on cell states absent from the reference, so a "0.95" label on a novel population deserves suspicion.</li>
  </ul>`);
  callout(main, 'warn', 'Pitfalls', `<ul>
    <li>Fitting the calibrator on the <b>training</b> data: the model is overconfident there by construction, so you will over-correct. Use a separate split or cross-validation (<code>CalibratedClassifierCV</code>).</li>
    <li>Binned ECE depends on the number of bins and gets noisy in sparse bins. Look at the plot, not only the number.</li>
    <li>Calibration is a property of a model <b>on a population</b>. A model calibrated on a 30%-prevalence cohort is miscalibrated where prevalence is 5%.</li>
    <li>Class rebalancing (oversampling, class weights) deliberately miscalibrates probabilities. Recalibrate afterwards if you need real risks.</li>
  </ul>`);
  seeAlso(main, [['concepts', 'metrics', 'Classification metrics'], ['concepts', 'logistic-regression', 'Logistic regression'], ['concepts', 'imbalance', 'Class imbalance'], ['omics', 'bulk-model-compare', 'Model comparison & calibration']]);
}
