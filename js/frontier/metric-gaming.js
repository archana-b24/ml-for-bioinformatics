// Virtual Cell Challenge 2025 metrics: how one scale factor on the predicted delta moves PDS, MAE and Pearson-delta differently.
import { header, figure, slider, stats, section, steps, math, legend, seeAlso, tip, untip, el } from '../lib/ui.js';
import { frame, cat, css, ink2, muted } from '../lib/plot.js';
import { rng } from '../lib/data.js';
import { sourceCard, toyBadge, coreIdea, realityCheck, useIt, cite, pearson } from './_shared.js';
const d3 = window.d3;

const G = 150, K = 40;
const ALPHAS = d3.range(0, 5.001, 0.1);
const SHRINK = 0.4;

// truth: shared stress response + sparse perturbation-specific programme, for test and training perturbations
function simulate(stress, noise) {
  const r = rng(101);
  const s = d3.range(G).map(() => r.normal(0, 0.6));
  const one = () => {
    const mag = Math.exp(r.normal(0, 0.5)), v = new Array(G).fill(0);
    for (let k = 0; k < 15; k++) v[r.int(G)] += r.normal(0, mag);
    return v.map((x, g) => stress * s[g] + x + r.normal(0, 0.05));
  };
  const test = d3.range(K).map(one), train = d3.range(K).map(one);
  const rp = rng(202);
  const pred = test.map((t) => t.map((x) => SHRINK * x + rp.normal(0, noise)));
  const meanBase = d3.range(G).map((g) => d3.mean(train, (t) => t[g]));
  return { test, pred, meanBase };
}
const l1 = (a, b) => { let s = 0; for (let g = 0; g < G; g++) s += Math.abs(a[g] - b[g]); return s / G; };
// simplified PDS: for each perturbation, fraction of other true profiles that are farther (L1) from its prediction than its own truth (ties = 1/2)
function pds(P, T) {
  let tot = 0;
  P.forEach((p, k) => {
    const dk = l1(p, T[k]); let sc = 0;
    T.forEach((t, j) => { if (j === k) return; const dj = l1(p, t); sc += dj > dk + 1e-12 ? 1 : Math.abs(dj - dk) <= 1e-12 ? 0.5 : 0; });
    tot += sc / (T.length - 1);
  });
  return tot / P.length;
}
const mae = (P, T) => d3.mean(P, (p, k) => l1(p, T[k]));
const pdelta = (P, T) => d3.mean(P, (p, k) => pearson(p, T[k]));
function metrics(sim, a) {
  const P = sim.pred.map((p) => p.map((x) => a * x));
  return { pds: pds(P, sim.test), mae: mae(P, sim.test), pr: a === 0 ? 0 : pdelta(P, sim.test), P };
}

export const __test = { simulate, metrics, ALPHAS };
export default async function render(main) {
  header(main, 'Virtual Cell Challenge metrics: gaming with one number',
    'A perturbation model predicts a <b>delta</b>: how each gene moves relative to control. Multiply that whole prediction by a constant α and nothing about the biology changes. Yet one metric ignores α completely, one prefers to keep predictions small, and one <b>rewards blowing them up</b>.');

  sourceCard(main, {
    org: 'Arc Institute (organiser); entrants including BioMap, XLearning, Outlier and Altos',
    when: '2025 challenge; 2026 round under way',
    links: ['vcc2025', 'vcc2026', 'systema'],
    claim: 'Arc\'s wrap-up of the 2025 challenge: almost nobody beat the mean baseline on MAE, and the L1-based perturbation discrimination score (PDS) rewards inflating prediction magnitude.',
    selfReported: false,
    status: 'organiser\'s retrospective',
  });

  coreIdea(main, `<p>The 2025 challenge (H1 hESC cells, 300 CRISPRi perturbations) scored models on three metrics: <b>PDS</b> (can you tell which perturbation a prediction belongs to?), <b>DES</b> (do you recover the differentially expressed genes?) and <b>MAE</b> (how far are the predicted expression values?). This page rebuilds simplified versions of PDS and MAE, plus the widely used <b>Pearson-delta</b>, and scales a fixed set of predictions by α.</p>
  <p><b>Key ML ingredient:</b> the evaluation metric <i>is</i> the objective. Whatever a leaderboard rewards, teams will optimise, so a metric that is not invariant to a meaningless transformation (here, scale) can be gamed without improving the biology.</p>`);

  const fig = figure(main, 'One set of predictions, rescaled',
    '40 test perturbations × 150 genes. The "model" predicts 0.4 × the true delta plus noise, a deliberately simple stand-in for a shrunk, noisy but genuinely informative model. Dashed lines: the mean baseline (the average delta of 40 separate training perturbations, predicted for every test perturbation). The vertical marker is the current α.');
  toyBadge(fig, 'simplified PDS, not the official scoring code');
  let alpha = 1, stress = 0.5, noise = 0.4, sim, curves, base;
  slider(fig.controls, { label: 'Scale factor α', min: 0, max: 5, step: 0.1, value: alpha, fmt: (v) => v.toFixed(1), onInput: (v) => { alpha = v; paint(); } });
  slider(fig.controls, { label: 'Generic stress', min: 0, max: 1.5, step: 0.05, value: stress, fmt: (v) => v.toFixed(2), onInput: (v) => { stress = v; recompute(); } });
  slider(fig.controls, { label: 'Prediction noise', min: 0.05, max: 0.6, step: 0.05, value: noise, fmt: (v) => v.toFixed(2), onInput: (v) => { noise = v; recompute(); } });
  legend(fig.root, [['PDS (simplified)', cat(0)], ['Pearson-delta', cat(1)], ['MAE', cat(2)], ['mean baseline (dashed)', muted()]]);
  const readout = stats(fig.root);

  const row = el('div', { class: 'fig-row' }); fig.body.append(row);
  const [a, b, c] = [0, 1, 2].map(() => el('div', { style: 'flex:1 1 260px' })); row.append(a, b, c);
  const F1 = frame(a, { width: 380, height: 330, x: [0, 5], y: [0, 1], xLabel: 'scale factor α', yLabel: 'score (higher = better)' });
  const F2 = frame(b, { width: 380, height: 330, x: [0, 5], y: [0, 1], xLabel: 'scale factor α', yLabel: 'MAE (lower = better)' });
  const F3 = frame(c, { width: 380, height: 330, x: [-3, 3], y: [-3, 3], xLabel: 'true delta (perturbation 1)', yLabel: 'α × predicted delta' });

  function recompute() {
    sim = simulate(stress, noise);
    curves = ALPHAS.map((al) => { const m = metrics(sim, al); return { a: al, pds: m.pds, mae: m.mae, pr: m.pr }; });
    const MB = sim.test.map(() => sim.meanBase);
    base = { pds: pds(MB, sim.test), mae: mae(MB, sim.test), pr: pdelta(MB, sim.test) };
    F2.y.domain([0, Math.max(base.mae, d3.max(curves, (d) => d.mae)) * 1.05]); F2.g.select('.axis.y').call(d3.axisLeft(F2.y).ticks(5));
    paint();
  }
  const ln = (F, data, key, color, dash) => F.layer(key + (dash ? 'b' : '')).selectAll('path').data([data]).join('path').attr('fill', 'none').attr('stroke', color).attr('stroke-width', 2)
    .attr('stroke-dasharray', dash ? '5 4' : null).attr('d', d3.line().x((d) => F.x(d.a)).y((d) => F.y(d.v)));
  const hline = (F, v, color, key) => ln(F, [{ a: 0, v }, { a: 5, v }], key, color, true);
  function paint() {
    ln(F1, curves.map((d) => ({ a: d.a, v: d.pds })), 'pds', cat(0));
    ln(F1, curves.filter((d) => d.a > 0).map((d) => ({ a: d.a, v: Math.max(0, d.pr) })), 'pr', cat(1));
    hline(F1, base.pds, cat(0), 'pds'); hline(F1, Math.max(0, base.pr), cat(1), 'pr');
    ln(F2, curves.map((d) => ({ a: d.a, v: d.mae })), 'mae', cat(2)); hline(F2, base.mae, cat(2), 'mae');
    const m = metrics(sim, alpha);
    [F1, F2].forEach((F) => F.layer('mark').selectAll('line').data([alpha]).join('line').attr('x1', F.x(alpha)).attr('x2', F.x(alpha)).attr('y1', 0).attr('y2', F.h).attr('stroke', ink2()).attr('stroke-width', 1.5));
    const dots = (F, pts) => F.layer('dots').selectAll('circle').data(pts).join('circle').attr('cx', (d) => F.x(alpha)).attr('cy', (d) => F.y(d.v)).attr('r', 5).attr('fill', (d) => d.c).attr('stroke', css('--surface')).attr('stroke-width', 1.5)
      .on('mousemove', (e, d) => tip(e, `${d.n} at α=${alpha.toFixed(1)}: <b>${d.v.toFixed(3)}</b>`)).on('mouseleave', untip);
    dots(F1, [{ v: m.pds, c: cat(0), n: 'PDS' }, ...(alpha > 0 ? [{ v: Math.max(0, m.pr), c: cat(1), n: 'Pearson-delta' }] : [])]);
    dots(F2, [{ v: m.mae, c: cat(2), n: 'MAE' }]);
    // scatter for perturbation 1
    const pts = sim.test[0].map((t, g) => ({ t, p: m.P[0][g], g }));
    F3.layer('diag').selectAll('line').data([0]).join('line').attr('x1', F3.x(-3)).attr('y1', F3.y(-3)).attr('x2', F3.x(3)).attr('y2', F3.y(3)).attr('stroke', muted()).attr('stroke-dasharray', '4 4');
    F3.layer('pts').selectAll('circle').data(pts).join('circle').attr('cx', (d) => F3.x(Math.max(-3, Math.min(3, d.t)))).attr('cy', (d) => F3.y(Math.max(-3, Math.min(3, d.p))))
      .attr('r', 4).attr('fill', cat(0)).attr('opacity', 0.75).attr('stroke', css('--surface')).attr('stroke-width', 1)
      .on('mousemove', (e, d) => tip(e, `gene ${d.g + 1}<br>true ${d.t.toFixed(2)}, predicted ${d.p.toFixed(2)}`)).on('mouseleave', untip);
    const best = curves.reduce((x, y) => (y.mae < x.mae ? y : x));
    readout.set({
      'PDS at α': m.pds.toFixed(3), 'MAE at α': m.mae.toFixed(3), 'Pearson-delta at α': alpha > 0 ? m.pr.toFixed(3) : 'undefined',
      'α minimising MAE': best.a.toFixed(1), 'mean baseline: PDS / MAE / Pearson-δ': `${base.pds.toFixed(2)} / ${base.mae.toFixed(3)} / ${base.pr.toFixed(2)}`,
    });
  }
  recompute();

  section(main, 'What to notice', `<ul>
    <li><b>Pearson-delta is a flat line.</b> Correlation ignores scale, so any α &gt; 0 gives the same value. It cannot tell a well-calibrated model from one that is 5× too large.</li>
    <li><b>PDS keeps rising long after MAE has started to get worse.</b> The model's predictions are 0.4× too small, so the "right" size is α = 2.5, but MAE is already lowest near α = 0.5 because of the noise. PDS climbs from there to near 1 and then plateaus. With L1 distance, a large prediction is far from every true profile, and what decides the ranking is whether its <i>signs</i> agree with the right perturbation. Inflating costs almost nothing.</li>
    <li><b>MAE punishes inflation</b> and rewards timidity. Push the prediction-noise slider up and watch the MAE-optimal α in the readout shrink. In the default setting, the model's MAE never gets below the dashed mean baseline at any α. Set generic stress to 0 and it can.</li>
    <li>Raise <b>generic stress</b>: the mean baseline's Pearson-delta climbs, because every perturbation shares the stress response and the mean predicts it perfectly. This is the inflation that ${cite('systema', 'Systema')} removes by measuring deltas against the perturbed centroid.</li>
    <li>The mean baseline's PDS is exactly 0.5 (chance): it predicts the same profile for every perturbation, so it can never discriminate them.</li>
  </ul>`);

  steps(main, 'How it works', [
    'For each test perturbation k, the model predicts a delta vector p_k over genes (perturbed minus control).',
    '<b>MAE</b>: average absolute difference between p_k and the true delta t_k, over genes and perturbations.',
    '<b>Pearson-delta</b>: correlation between p_k and t_k across genes, averaged over perturbations. Scale- and shift-invariant.',
    '<b>PDS (simplified)</b>: compute the L1 distance from p_k to every true delta t_j. Score = the fraction of other perturbations whose truth is farther from p_k than t_k is (ties count ½). 1 = perfect, 0.5 = chance. The official score\'s normalisation differs; the L1-based ranking is the part that matters here.',
    'Scale every prediction by α and recompute all three. A metric that changes with α is rewarding magnitude, not just direction.',
  ]);
  math(main, 'MAE(α)      = mean_k,g | α·p_kg − t_kg |\nPearson(α)  = mean_k corr(α·p_k, t_k) = mean_k corr(p_k, t_k)   for every α > 0\nPDS(α)      = mean_k  #{ j≠k : ‖α·p_k − t_j‖₁ > ‖α·p_k − t_k‖₁ } / (K−1)');

  realityCheck(main, `<ul>
    <li>This is not a hypothetical: Arc's own ${cite('vcc2025', '2025 wrap-up')} lists "L1-based PDS rewards inflating prediction magnitude" and "almost nobody beat the mean baseline on MAE" among the lessons. Winners (BioMap, XLearning, Outlier) combined statistical and deep-learning components.</li>
    <li>Independent work points the same way: ${cite('systema', 'Systema')} shows that systematic perturbed-vs-control differences inflate correlation metrics; ${cite('ahlmann', 'Ahlmann-Eltze et al. 2025')} found nothing beat the training mean on unseen single perturbations.</li>
    <li>Our PDS is a simplified re-implementation for teaching. Do not read the curves' absolute values as challenge scores.</li>
    <li>The ${cite('vcc2026', '2026 round')} (zero-shot on 6 new cell lines) has not announced winners as of 2026-11-12, so we do not know how its metrics behaved.</li>
  </ul>`);

  useIt(main, `<ul>
    <li>When you benchmark perturbation models on your own Perturb-seq or drug screen, <b>report several metrics together</b> (a scale-sensitive one like MAE, a rank or discrimination one, and a correlation computed against the perturbed centroid) and always include the mean and additive baselines.</li>
    <li>Do a 30-second sanity check: multiply your model's predictions by 0.5 and 2. If your headline metric improves, the metric, not the model, is talking.</li>
    <li>The same trap applies outside perturbations: correlation-based "accuracy" for imputation or spatial deconvolution ignores calibration. Check scale explicitly.</li>
  </ul>`);

  seeAlso(main, [['frontier', 'perturbation-arena', 'Perturbation baselines arena'], ['frontier', 'state-sets', 'State: predicting distributions'], ['frontier', 'landscape', 'The 2024–26 landscape'],
    ['concepts', 'metrics', 'Classification metrics'], ['concepts', 'loss-functions', 'Loss functions'], ['concepts', 'calibration', 'Calibration'], ['omics', 'bulk-model-compare', 'Model comparison & calibration']]);
}
