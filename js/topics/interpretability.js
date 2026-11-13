// Feature importance & SHAP: permutation importance and exact Shapley values on a small synthetic expression model.
import { header, figure, slider, checkbox, stats, section, steps, callout, math, legend, seeAlso, tip, untip, el } from '../lib/ui.js';
import { frame, cat, css, ink, ink2, muted } from '../lib/plot.js';
import { rng, sigmoid, solve } from '../lib/data.js';
const d3 = window.d3;

const GENES = ['CD8A', 'GZMB', 'PDCD1', 'MKI67', 'COL1A1', 'ACTB'];
const ROLE = {
  CD8A: 'one noisy readout of the cytotoxic T-cell program that drives response',
  GZMB: 'a second noisy readout of the same T-cell program (r ≈ 0.8 with CD8A)',
  PDCD1: 'PD-1; helps, and amplifies the T-cell program (interaction)',
  MKI67: 'proliferation; small positive effect',
  COL1A1: 'stromal / fibroblast; blocks response',
  ACTB: 'housekeeping; no effect',
};
// true log-odds of response, driven by a hidden T-cell program activity T (the model never sees T)
const truth = (g) => -0.2 + 1.6 * g.T + 0.6 * g.PDCD1 + 0.7 * g.T * g.PDCD1 + 0.4 * g.MKI67 - 1.1 * g.COL1A1;

function makeData(n, seed) {
  const r = rng(seed);
  return d3.range(n).map(() => {
    const g = {}; g.T = r.normal(); g.CD8A = 0.9 * g.T + Math.sqrt(1 - 0.81) * r.normal(); g.GZMB = 0.9 * g.T + Math.sqrt(1 - 0.81) * r.normal();
    g.PDCD1 = r.normal(); g.MKI67 = r.normal(); g.COL1A1 = r.normal(); g.ACTB = r.normal();
    g.y = r() < sigmoid(truth(g)) ? 1 : 0;
    return g;
  });
}

/** Logistic regression on the given genes plus a CD8A×PDCD1 interaction term, fit by Newton's method with a small ridge. */
function fitModel(train, feats) {
  const phi = (g) => [1, ...feats.map((f) => g[f]), g.CD8A * g.PDCD1];
  const X = train.map(phi), y = train.map((g) => g.y), p = X[0].length;
  let w = new Array(p).fill(0);
  for (let it = 0; it < 25; it++) {
    const H = d3.range(p).map((i) => new Array(p).fill(0)), gr = new Array(p).fill(0);
    X.forEach((x, n) => { const q = sigmoid(x.reduce((s, v, j) => s + v * w[j], 0)), e = q - y[n], ww = q * (1 - q);
      for (let i = 0; i < p; i++) { gr[i] += e * x[i]; for (let j = 0; j < p; j++) H[i][j] += ww * x[i] * x[j]; } });
    for (let i = 1; i < p; i++) { H[i][i] += 1; gr[i] += w[i]; } // ridge λ = 1 (not on intercept)
    const step = solve(H, gr);
    w = w.map((v, i) => v - step[i]);
    if (d3.sum(step, Math.abs) < 1e-8) break;
  }
  const logit = (g) => phi(g).reduce((s, v, j) => s + v * w[j], 0);
  return { feats, w, logit, prob: (g) => sigmoid(logit(g)) };
}

function auc(scores, ys) {
  const idx = d3.range(scores.length).sort((a, b) => scores[a] - scores[b]);
  let rsum = 0, npos = 0; idx.forEach((i, r) => { if (ys[i]) { rsum += r + 1; npos++; } });
  const nneg = ys.length - npos; return (rsum - (npos * (npos + 1)) / 2) / (npos * nneg);
}

/** Permutation importance on the test set: drop in AUC when one gene's column is shuffled (10 repeats). */
function permImportance(model, test) {
  const ys = test.map((g) => g.y), base = auc(test.map(model.logit), ys);
  return model.feats.map((f, k) => {
    const r = rng(100 + k), drops = d3.range(10).map(() => {
      const perm = r.shuffle(test.map((g) => g[f]));
      return base - auc(test.map((g, i) => model.logit({ ...g, [f]: perm[i] })), ys);
    });
    return { f, mean: d3.mean(drops), lo: d3.min(drops), hi: d3.max(drops) };
  });
}

/** Exact interventional Shapley values of the log-odds by enumerating all 2^d subsets against a background sample. */
function shapley(model, x, background) {
  const F = model.feats, d = F.length, v = new Array(1 << d);
  for (let S = 0; S < 1 << d; S++) {
    let s = 0;
    background.forEach((b) => { const z = { ...b }; for (let i = 0; i < d; i++) if (S & (1 << i)) z[F[i]] = x[F[i]]; s += model.logit(z); });
    v[S] = s / background.length;
  }
  const fact = [1]; for (let i = 1; i <= d; i++) fact[i] = fact[i - 1] * i;
  const pop = (S) => { let c = 0; while (S) { c += S & 1; S >>= 1; } return c; };
  const phi = F.map((f, i) => {
    let s = 0;
    for (let S = 0; S < 1 << d; S++) if (!(S & (1 << i))) { const k = pop(S); s += (fact[k] * fact[d - k - 1] / fact[d]) * (v[S | (1 << i)] - v[S]); }
    return s;
  });
  return { base: v[0], phi, fx: v[(1 << d) - 1] };
}

export default async function render(main) {
  header(main, 'Feature importance & SHAP',
    'Two different questions: <b>which genes does the model rely on overall</b> (permutation importance), and <b>why did it give this patient this score</b> (SHAP values). SHAP splits one prediction into additive per-gene contributions using Shapley values from game theory, so they sum exactly to the prediction.');

  const train = makeData(400, 21), test = makeData(200, 22);
  const bg = train.slice(0, 100);
  const models = { with: fitModel(train, GENES), without: fitModel(train, GENES.filter((g) => g !== 'GZMB')) };
  const imps = { with: permImportance(models.with, test), without: permImportance(models.without, test) };
  const aucs = { with: auc(test.map(models.with.logit), test.map((g) => g.y)), without: auc(test.map(models.without.logit), test.map((g) => g.y)) };
  let useGZMB = true, pid = 7;

  const fig = figure(main, 'A response classifier on 6 genes: global importance and one patient\'s explanation', '');
  checkbox(fig.controls, { label: 'Include GZMB (co-expressed with CD8A)', checked: true, onChange: (v) => { useGZMB = v; paint(); } });
  slider(fig.controls, { label: 'Test patient #', min: 0, max: 199, step: 1, value: pid, onInput: (v) => { pid = v; paint(); } });
  const readout = stats(fig.root);
  legend(fig.root, [['pushes toward response', css('--div-pos')], ['pushes toward non-response', css('--div-neg')]]);
  const row = el('div', { class: 'fig-row' }); fig.body.append(row);
  const lDiv = el('div'), rDiv = el('div'); row.append(lDiv, rDiv);
  lDiv.append(el('div', { class: 'fig-caption', style: 'margin:0 0 4px' }, 'Permutation importance (test set, 10 shuffles; whisker = range)'));
  rDiv.append(el('div', { class: 'fig-caption', style: 'margin:0 0 4px' }, 'SHAP waterfall for one patient (log-odds of response)'));
  const P = frame(lDiv, { width: 470, height: 300, x: [0, 0.2], y: [0, 1], xLabel: 'drop in test AUC when the gene is shuffled', margin: { left: 70 }, grid: false });
  P.g.select('.axis.y').remove();
  const W = frame(rDiv, { width: 490, height: 340, x: [-3, 3], y: [0, 1], xLabel: 'model log-odds of response', margin: { left: 118 }, grid: false });
  W.g.select('.axis.y').remove();

  function paint() {
    const key = useGZMB ? 'with' : 'without', model = models[key], imp = imps[key];
    // ----- permutation bars -----
    const by = d3.scaleBand().domain(GENES).range([0, P.h]).padding(0.28);
    P.x.domain([Math.min(0, d3.min(imp, (d) => d.lo)), Math.max(0.05, d3.max(imps.with.concat(imps.without), (d) => d.hi)) * 1.05]);
    P.g.select('.axis.x').call(d3.axisBottom(P.x).ticks(5));
    P.layer('lab').selectAll('text').data(GENES).join('text').attr('x', -8).attr('y', (g) => by(g) + by.bandwidth() / 2 + 4).attr('text-anchor', 'end')
      .attr('font-size', 12).attr('fill', (g) => (model.feats.includes(g) ? ink() : muted())).text((g) => g);
    P.layer('bars').selectAll('rect').data(imp, (d) => d.f).join('rect').attr('x', (d) => P.x(Math.min(0, d.mean))).attr('y', (d) => by(d.f))
      .attr('width', (d) => Math.abs(P.x(d.mean) - P.x(0))).attr('height', by.bandwidth()).attr('fill', cat(0)).attr('rx', 2)
      .on('mousemove', (e, d) => tip(e, `<b>${d.f}</b>: ${ROLE[d.f]}<br>mean AUC drop ${d.mean.toFixed(3)} (range ${d.lo.toFixed(3)} to ${d.hi.toFixed(3)})`)).on('mouseleave', untip);
    P.layer('wh').selectAll('line').data(imp, (d) => d.f).join('line').attr('x1', (d) => P.x(d.lo)).attr('x2', (d) => P.x(d.hi))
      .attr('y1', (d) => by(d.f) + by.bandwidth() / 2).attr('y2', (d) => by(d.f) + by.bandwidth() / 2).attr('stroke', ink()).attr('stroke-width', 1.5);
    P.layer('zero').selectAll('line').data([0]).join('line').attr('x1', P.x(0)).attr('x2', P.x(0)).attr('y1', 0).attr('y2', P.h).attr('stroke', css('--axis'));
    P.layer('na').selectAll('text').data(useGZMB ? [] : ['GZMB']).join('text').attr('x', 6).attr('y', by('GZMB') + by.bandwidth() / 2 + 4)
      .attr('font-size', 11.5).attr('fill', muted()).text('not in this model');

    // ----- SHAP waterfall -----
    const x = test[pid];
    const sh = shapley(model, x, bg);
    const items = model.feats.map((f, i) => ({ f, phi: sh.phi[i], val: x[f] })).sort((a, b) => Math.abs(b.phi) - Math.abs(a.phi));
    let acc = sh.base; items.forEach((d) => { d.start = acc; acc += d.phi; d.end = acc; });
    const rows = ['E[f(x)]', ...items.map((d) => d.f), 'f(x)'];
    const wy = d3.scaleBand().domain(rows).range([0, W.h]).padding(0.25);
    const ext = d3.extent([sh.base, sh.fx, ...items.flatMap((d) => [d.start, d.end])]);
    const pad = 0.15 * (ext[1] - ext[0] || 1);
    W.x.domain([ext[0] - pad, ext[1] + pad]);
    W.g.select('.axis.x').call(d3.axisBottom(W.x).ticks(6));
    W.layer('lab').selectAll('text').data(rows).join('text').attr('x', -8).attr('y', (r) => wy(r) + wy.bandwidth() / 2 + 4).attr('text-anchor', 'end').attr('font-size', 12)
      .attr('fill', ink()).text((r) => { const it = items.find((d) => d.f === r); return it ? `${r} = ${it.val >= 0 ? '+' : ''}${it.val.toFixed(2)}` : r === 'f(x)' ? 'prediction f(x)' : 'baseline E[f]'; });
    W.layer('ref').selectAll('line').data([sh.base, sh.fx]).join('line').attr('x1', (v) => W.x(v)).attr('x2', (v) => W.x(v)).attr('y1', 0).attr('y2', W.h)
      .attr('stroke', muted()).attr('stroke-dasharray', '3 3');
    W.layer('bars').selectAll('rect').data(items, (d) => d.f).join('rect').attr('x', (d) => W.x(Math.min(d.start, d.end))).attr('y', (d) => wy(d.f))
      .attr('width', (d) => Math.max(1, Math.abs(W.x(d.end) - W.x(d.start)))).attr('height', wy.bandwidth()).attr('rx', 2)
      .attr('fill', (d) => (d.phi >= 0 ? css('--div-pos') : css('--div-neg')))
      .on('mousemove', (e, d) => tip(e, `<b>${d.f}</b> = ${d.val.toFixed(2)} SD<br>SHAP value ${d.phi >= 0 ? '+' : ''}${d.phi.toFixed(3)} log-odds<br>${ROLE[d.f]}`)).on('mouseleave', untip);
    W.layer('conn').selectAll('line').data(items.slice(0, -1), (d) => d.f).join('line').attr('x1', (d) => W.x(d.end)).attr('x2', (d) => W.x(d.end))
      .attr('y1', (d) => wy(d.f) + wy.bandwidth()).attr('y2', (d, i) => wy(items[i + 1].f)).attr('stroke', ink2()).attr('stroke-width', 1);
    W.layer('val').selectAll('text').data(items, (d) => d.f).join('text')
      .attr('x', (d) => W.x(Math.max(d.start, d.end)) + 4).attr('y', (d) => wy(d.f) + wy.bandwidth() / 2 + 4).attr('font-size', 11).attr('fill', ink())
      .text((d) => (d.phi >= 0 ? '+' : '') + d.phi.toFixed(2));
    W.layer('ends').selectAll('circle').data([['E[f(x)]', sh.base], ['f(x)', sh.fx]]).join('circle').attr('cx', (d) => W.x(d[1])).attr('cy', (d) => wy(d[0]) + wy.bandwidth() / 2)
      .attr('r', 5).attr('fill', ink()).attr('stroke', css('--surface')).attr('stroke-width', 1.5);
    W.layer('endt').selectAll('text').data([['E[f(x)]', sh.base], ['f(x)', sh.fx]]).join('text').attr('x', (d) => W.x(d[1]) + 9).attr('y', (d) => wy(d[0]) + wy.bandwidth() / 2 + 4)
      .attr('font-size', 11.5).attr('fill', ink()).text((d) => `${d[1].toFixed(2)}  (p = ${sigmoid(d[1]).toFixed(2)})`);

    const cd = imp.find((d) => d.f === 'CD8A');
    readout.set({
      'test AUC': aucs[key].toFixed(3),
      'CD8A importance': cd.mean.toFixed(3),
      'GZMB importance': useGZMB ? imp.find((d) => d.f === 'GZMB').mean.toFixed(3) : '—',
      [`patient ${pid}: responded?`]: x.y ? 'yes' : 'no',
      'baseline + Σ SHAP = f(x)': `${sh.base.toFixed(2)} ${d3.sum(sh.phi) >= 0 ? '+' : '−'} ${Math.abs(d3.sum(sh.phi)).toFixed(2)} = ${sh.fx.toFixed(2)}`,
    });
  }
  paint();
  const gz = imps.with.find((d) => d.f === 'GZMB').mean, cdW = imps.with.find((d) => d.f === 'CD8A').mean, cdWo = imps.without.find((d) => d.f === 'CD8A').mean;
  const actb = imps.with.find((d) => d.f === 'ACTB').mean;
  fig.caption(`Synthetic cohort: 400 training and 200 test patients, 6 standardised genes. The true response depends on a hidden cytotoxic T-cell program (which <b>both CD8A and GZMB</b> measure, with independent noise), PDCD1 (plus an interaction with the program), MKI67 and COL1A1; ACTB is noise. The model is a ridge-penalised logistic regression with the interaction term. Hover a bar for each gene's true role. SHAP values are computed exactly by enumerating all ${2 ** 6} gene subsets against 100 background patients.`);

  section(main, 'What to notice', `<ul>
    <li><b>The correlation pitfall.</b> CD8A and GZMB carry the same biological signal, so the model splits the credit: CD8A's permutation importance is ${cdW.toFixed(3)} and GZMB's is ${gz.toFixed(3)}. Neither looks as important as the T-cell program really is, because shuffling one leaves the other to cover for it. Untick <i>Include GZMB</i> and CD8A's importance jumps to ${cdWo.toFixed(3)}, while test AUC hardly moves (${aucs.with.toFixed(3)} → ${aucs.without.toFixed(3)}). Same biology, different ranking: importance describes the <b>model</b>, not the biology.</li>
    <li>ACTB, pure noise, gets an importance of ${actb.toFixed(3)}. With 200 test patients, values this close to 0 are within shuffle-to-shuffle noise (see the whiskers).</li>
    <li>Move through patients. The waterfall always starts at the cohort-average log-odds E[f] and ends exactly at this patient's prediction: SHAP values are <b>additive</b> (check the last readout).</li>
    <li>A gene's SHAP value depends on the patient's other genes. Because of the CD8A×PDCD1 interaction, two patients with the same CD8A level get different CD8A contributions if their PDCD1 differs. Shapley values split the interaction's effect evenly between the two genes.</li>
    <li>A negative SHAP value for COL1A1 does not mean "COL1A1 blocks response in this patient". It means: compared with the average patient, this patient's COL1A1 level moves the <b>model's</b> score down.</li>
  </ul>`);

  steps(main, 'How it works', [
    '<b>Permutation importance:</b> score the trained model on test data. Shuffle one feature\'s column (breaking its link to the outcome while keeping its distribution), score again, and record the drop. Repeat a few times and per feature.',
    '<b>Shapley values</b> come from cooperative game theory: the "players" are features and the "payout" is the prediction. A feature\'s value is its average marginal contribution over every order in which features could be added.',
    'For a set S of "known" features, define v(S) as the model\'s average output when the features in S are fixed to this patient\'s values and the rest are drawn from background patients.',
    'Enumerate all 2<sup>d</sup> subsets (64 for 6 genes) and combine the differences v(S ∪ {i}) − v(S) with the Shapley weights. This is exact; the <code>shap</code> library uses fast approximations (TreeSHAP for tree models, KernelSHAP sampling for anything else) because 2<sup>d</sup> explodes for thousands of genes.',
    'The values add up: E[f] + Σ φᵢ = f(x). Plot them as a waterfall for one sample, or as a beeswarm across samples for a global view.',
  ]);
  math(main, 'φᵢ = Σ over S ⊆ N∖{i}   |S|! (d − |S| − 1)! / d!  ·  [ v(S ∪ {i}) − v(S) ]\nv(S) = mean over background b of  f(x_S, b_rest)          E[f] + Σᵢ φᵢ = f(x)');

  callout(main, '', 'In biology', `<ul>
    <li>Explaining a <b>bulk RNA-seq response classifier</b> per patient: SHAP waterfalls show which genes pushed one tumour's score up (e.g. a T-cell inflamed signature) or down (a stromal TGF-β program).</li>
    <li>Gene expression is full of <b>co-expressed modules</b> (CD8A, CD8B, GZMB, PRF1, NKG7 in cytotoxic cells). Importance gets spread across, or arbitrarily concentrated in, members of a module. Interpret importance at the level of modules or pathways, or cluster correlated genes first.</li>
    <li>In single-cell and spatial models, SHAP is used to rank genes behind a cell-type classifier, and attribution methods (integrated gradients, in-silico perturbation in Geneformer) play the same role for deep models.</li>
  </ul>`);
  callout(main, 'warn', 'Pitfalls', `<ul>
    <li><b>Importance is not causation.</b> A gene can be important because it is a proxy (a batch marker, a sex gene, a tumour-purity marker) rather than a driver.</li>
    <li><b>Correlated features:</b> permuting one of two correlated genes creates unrealistic patients (high CD8A, low GZMB) and the model is evaluated off its data. Interventional SHAP has the same issue. Conditional variants exist but answer a different question.</li>
    <li>Importance computed on the <b>training</b> set rewards overfitting. Use held-out data.</li>
    <li>SHAP values depend on the background set. A different reference cohort changes E[f] and every φ.</li>
  </ul>`);
  seeAlso(main, [['omics', 'bulk-explain', 'Explaining predictions'], ['concepts', 'random-forest', 'Random forests'], ['concepts', 'regularization', 'Regularization'], ['concepts', 'logistic-regression', 'Logistic regression']]);
}
