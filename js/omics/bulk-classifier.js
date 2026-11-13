// Bulk step 6: response classifier, and the feature-selection leakage trap (leaky vs nested CV, real vs permuted labels).
// Driven by data/omics/bulk_classifier.json.
import { header, figure, slider, select, button, stats, section, steps, callout, math, legend, seeAlso, every, tip, untip, el } from '../lib/ui.js';
import { frame, cat, css, ink, ink2, muted } from '../lib/plot.js';
import { rng } from '../lib/data.js';
import { stepCard, omicsData } from './_shared.js';
import { f3 } from './_bulk_common.js';
const d3 = window.d3;

const SCHEME = { leaky: 'leaky: genes chosen on all patients', nested: 'nested: genes chosen inside each training fold' };
const scol = (s) => (s === 'leaky' ? cat(7) : cat(0));

export default async function render(main) {
  header(main, 'Response classifier & nested CV',
    'Train a model to predict response and estimate how well it will do on new patients. With far more genes than patients, the estimate is easy to fool: if you pick the genes using all patients and <b>then</b> cross-validate, the test folds have already been seen, and even <b>random labels</b> look predictable.');
  stepCard(main, {
    stage: 'Bulk RNA-seq · step 6 of 11',
    question: 'Can pre-treatment expression predict who will respond, and how accurate would that prediction be for the next patient?',
    data: 'Batch-corrected log-CPM of all genes for the patients who passed QC, and the response label.',
    method: 'Pipeline = pick the top-k genes by t-statistic → standardise → L2-penalised logistic regression (penalty tuned by inner 3-fold CV). Performance by 5-fold CV AUC, repeated 8 times, with selection outside (leaky) or inside (nested) the folds.',
    verdict: 'ml', why: ' Supervised prediction. The statistics comes in at evaluation: an honest estimate of generalisation, and a null distribution from permuted labels.',
  });
  const C = await omicsData(main, 'bulk_classifier.json');
  if (!C) return;
  const KS = C.ks, R = C.res, key = (lab, s, k) => R[`${lab}|${s}|${k}`];

  // ------------------------------------------------------------------ figure 1: what each scheme lets the selection step see
  const fig0 = figure(main, 'Where the leak happens',
    `Each row is one round of 5-fold cross-validation over ${C.n} patients. Blue cells are the test fold. The bar above each row shows which patients the <b>gene-selection</b> step looked at. In the leaky scheme it saw the test fold too.`);
  let fold = 0, playing = null;
  button(fig0.controls, 'Play folds', () => { if (playing) { playing(); playing = null; return; } playing = every(900, () => { fold = (fold + 1) % 5; paint0(); }); }, true);
  legend(fig0.controls, [['training patients', muted()], ['test fold', cat(0)], ['seen by gene selection', ink()], ['test patients seen (leak)', cat(7)]]);
  const row0 = el('div', { class: 'fig-row' }); fig0.body.append(row0);
  const panels0 = ['leaky', 'nested'].map((s) => {
    const w = el('div'); row0.append(w);
    const W = 440, H = 260, x0 = 54, cw = (W - x0 - 10) / C.n;
    const svg = d3.select(w).append('svg').attr('viewBox', `0 0 ${W} ${H}`).attr('width', W);
    svg.append('text').attr('x', 0).attr('y', 14).attr('font-weight', 600).attr('font-size', 13).attr('fill', ink()).text(s === 'leaky' ? 'Leaky CV' : 'Nested (honest) CV');
    const rows = d3.range(5).map((f) => {
      const y = 34 + f * 44;
      svg.append('text').attr('x', 0).attr('y', y + 26).attr('font-size', 11).attr('fill', ink2()).text(`fold ${f + 1}`);
      const lo = Math.round(f * C.n / 5), hi = Math.round((f + 1) * C.n / 5);
      const cells = svg.append('g').selectAll('rect').data(d3.range(C.n)).join('rect').attr('x', (i) => x0 + i * cw).attr('y', y + 14).attr('width', Math.max(1, cw - 0.4)).attr('height', 18)
        .attr('fill', (i) => (i >= lo && i < hi ? cat(0) : muted())).attr('opacity', 0.85);
      const seen = svg.append('g');
      if (s === 'leaky') seen.append('rect').attr('x', x0).attr('y', y + 4).attr('width', C.n * cw).attr('height', 6).attr('rx', 3).attr('fill', ink());
      else {
        if (lo > 0) seen.append('rect').attr('x', x0).attr('y', y + 4).attr('width', lo * cw).attr('height', 6).attr('rx', 3).attr('fill', ink());
        if (hi < C.n) seen.append('rect').attr('x', x0 + hi * cw).attr('y', y + 4).attr('width', (C.n - hi) * cw).attr('height', 6).attr('rx', 3).attr('fill', ink());
      }
      if (s === 'leaky') seen.append('rect').attr('x', x0 + lo * cw).attr('y', y + 4).attr('width', (hi - lo) * cw).attr('height', 6).attr('rx', 3).attr('fill', cat(7));
      return { g: [cells, seen] };
    });
    return { rows };
  });
  function paint0() { panels0.forEach((p) => p.rows.forEach((r, f) => r.g.forEach((g) => g.attr('opacity', f === fold ? 1 : 0.25)))); }
  paint0();

  // ------------------------------------------------------------------ figure 2: ROC + AUC vs k
  const fig = figure(main, 'The leakage trap, measured',
    'Left: out-of-fold ROC curves (first of 8 repeats) for the chosen number of genes k. Right: mean CV AUC against k for both schemes, with ±1 SD band and the 8 individual repeats as small dots; solid = real labels, dashed = permuted labels. Hover anything.');
  let lab = 'perm', k = 200;
  select(fig.controls, { label: 'Labels', value: lab, options: [['real', 'real response labels'], ['perm', 'permuted (random) labels']], onChange: (v) => { lab = v; draw(); } });
  const kSl = slider(fig.controls, { label: 'genes selected k', min: 0, max: KS.length - 1, step: 1, value: KS.indexOf(k), fmt: (v) => KS[v], onInput: (v) => { k = KS[v]; draw(); } });
  legend(fig.controls, [[SCHEME.leaky, scol('leaky')], [SCHEME.nested, scol('nested')]]);
  const rd = stats(fig.root);
  const row = el('div', { class: 'fig-row' }); fig.body.append(row);
  const Lw = el('div'), Rw = el('div'); row.append(Lw, Rw);
  Lw.style.flex = '0 1 380px';
  const FR = frame(Lw, { width: 380, height: 380, x: [0, 1], y: [0, 1], xLabel: 'false positive rate', yLabel: 'true positive rate' });
  FR.layer('diag').append('line').attr('x1', FR.x(0)).attr('y1', FR.y(0)).attr('x2', FR.x(1)).attr('y2', FR.y(1)).attr('stroke', muted()).attr('stroke-dasharray', '4 4');
  const FK = frame(Rw, { width: 540, height: 380, x: [0, KS.length - 1], y: [0.25, 1], xLabel: 'genes selected k (log spacing)', yLabel: 'cross-validated AUC' });
  FK.g.select('.axis.x').call(d3.axisBottom(FK.x).tickValues(d3.range(KS.length)).tickFormat((i) => KS[i]));
  FK.layer('chance').append('line').attr('x1', 0).attr('x2', FK.w).attr('y1', FK.y(0.5)).attr('y2', FK.y(0.5)).attr('stroke', muted()).attr('stroke-dasharray', '3 3');
  FK.layer('chance').append('text').attr('x', 4).attr('y', FK.y(0.5) + 14).attr('text-anchor', 'start').attr('font-size', 11).attr('fill', ink2()).text('chance');
  const jr = rng(5), jit = d3.range(C.reps).map(() => (jr() - 0.5) * 0.18);
  ['real', 'perm'].forEach((lb) => ['leaky', 'nested'].forEach((s) => {
    const pts = KS.map((kk, i) => ({ i, kk, m: key(lb, s, kk).mean, sd: key(lb, s, kk).sd, a: key(lb, s, kk).auc }));
    FK.layer('band').append('path').datum(pts).attr('fill', scol(s)).attr('opacity', 0.12).attr('d', d3.area().x((d) => FK.x(d.i)).y0((d) => FK.y(d.m - d.sd)).y1((d) => FK.y(d.m + d.sd)));
    FK.layer('reps').selectAll(null).data(pts.flatMap((d) => d.a.map((a, r) => ({ i: d.i, a, r })))).join('circle')
      .attr('cx', (d) => FK.x(d.i + jit[d.r])).attr('cy', (d) => FK.y(d.a)).attr('r', 2).attr('fill', scol(s)).attr('opacity', 0.45);
    FK.layer('lines').append('path').datum(pts).attr('fill', 'none').attr('stroke', scol(s)).attr('stroke-width', 2).attr('stroke-dasharray', lb === 'perm' ? '6 4' : null)
      .attr('d', d3.line().x((d) => FK.x(d.i)).y((d) => FK.y(d.m)));
    FK.layer('pts').selectAll(null).data(pts).join('circle').attr('cx', (d) => FK.x(d.i)).attr('cy', (d) => FK.y(d.m)).attr('r', 4.5).attr('fill', scol(s)).attr('stroke', css('--surface')).attr('stroke-width', 1.2)
      .on('mousemove', (e, d) => tip(e, `<b>${lb === 'real' ? 'real' : 'permuted'} labels, ${s}</b><br>k = ${d.kk}: AUC ${d.m.toFixed(3)} ± ${d.sd.toFixed(3)}<br>repeats: ${d.a.map((a) => a.toFixed(2)).join(', ')}`)).on('mouseleave', untip);
    const end = pts[pts.length - 1];
    FK.layer('lab').append('text').attr('x', FK.x(end.i) - 6).attr('y', FK.y(end.m) + (s === 'leaky' ? -9 : 15)).attr('text-anchor', 'end').attr('font-size', 11).attr('fill', ink2())
      .text(`${lb === 'real' ? 'real' : 'permuted'}, ${s}`);
  }));
  const kLine = FK.layer('k').append('line').attr('y1', 0).attr('y2', FK.h).attr('stroke', ink()).attr('stroke-dasharray', '2 3');

  function draw() {
    const rows = ['leaky', 'nested'].map((s) => ({ s, r: key(lab, s, k) }));
    FR.layer('roc').selectAll('path').data(rows).join('path').attr('fill', 'none').attr('stroke', (d) => scol(d.s)).attr('stroke-width', 2.5)
      .attr('stroke-dasharray', lab === 'perm' ? '6 4' : null)
      .attr('d', (d) => d3.line().curve(d3.curveStepAfter).x((p) => FR.x(p[0])).y((p) => FR.y(p[1]))([...d.r.roc, [1, 1]]))
      .on('mousemove', (e, d) => tip(e, `<b>${SCHEME[d.s]}</b><br>AUC of this repeat ${d.r.auc[0].toFixed(3)}`)).on('mouseleave', untip);
    kLine.attr('x1', FK.x(KS.indexOf(k))).attr('x2', FK.x(KS.indexOf(k)));
    const lk = key(lab, 'leaky', k), ns = key(lab, 'nested', k);
    rd.set({ labels: lab === 'real' ? 'real' : 'permuted', k, 'leaky AUC': `${f3(lk.mean)} ± ${f3(lk.sd)}`, 'nested AUC': `${f3(ns.mean)} ± ${f3(ns.sd)}`, 'optimism (leaky − nested)': (lk.mean - ns.mean >= 0 ? '+' : '') + (lk.mean - ns.mean).toFixed(3) });
  }
  draw();

  // numbers for prose
  const kMax = KS[KS.length - 1];
  const bestNested = KS.reduce((a, b) => (key('real', 'nested', b).mean > key('real', 'nested', a).mean ? b : a));
  const pl = (k_) => key('perm', 'leaky', k_).mean, pn = (k_) => key('perm', 'nested', k_).mean;
  const permNestedRange = d3.extent(KS, pn);
  const spread = (lb, only) => d3.max(['leaky', 'nested'].filter((s) => !only || s === only).flatMap((s) => KS.map((k_) => d3.max(key(lb, s, k_).auc) - d3.min(key(lb, s, k_).auc))));
  section(main, 'What to notice', `<ul>
    <li><b>Permuted labels, k = ${kMax}:</b> the response labels are shuffled, so no gene can truly predict them. Yet leaky CV reports AUC ${f3(pl(kMax))}. Nested CV reports ${f3(pn(kMax))}, i.e. chance. Across all k, nested CV on permuted labels stays between ${f3(permNestedRange[0])} and ${f3(permNestedRange[1])}.</li>
    <li>Slide k down on permuted labels: the leaky AUC falls (${f3(pl(KS[0]))} at k = ${KS[0]}) but never reaches chance. Selecting the ${KS[0]} genes most correlated with random labels <i>across all patients</i> already bakes the answer into the test folds.</li>
    <li><b>Real labels:</b> at k = ${KS[0]} the two schemes nearly agree (${f3(key('real', 'leaky', KS[0]).mean)} vs ${f3(key('real', 'nested', KS[0]).mean)}): the top genes are the IFN-γ module, which any training fold finds too. At k = ${kMax} they split: leaky ${f3(key('real', 'leaky', kMax).mean)}, nested ${f3(key('real', 'nested', kMax).mean)}. The extra genes are noise that only "help" when they were chosen while looking at the test patients.</li>
    <li>The honest curve peaks at k = ${bestNested} (AUC ${f3(key('real', 'nested', bestNested).mean)}) and falls as more genes are added. A small biomarker is not just cheaper; here it is better. (Choosing k by looking at this curve is itself a small optimism: in a full analysis k is tuned in an inner loop too.)</li>
    <li>The 8 repeats (small dots) use different fold splits. With real labels they span up to ${spread('real').toFixed(2)} AUC at one k; with permuted labels in nested CV, up to ${spread('perm', 'nested').toFixed(2)}. A single CV run is itself a noisy estimate; report the spread.</li>
  </ul>`);
  steps(main, 'How it works', [
    '<b>Define the whole pipeline</b> as one function of training data: select top-k genes by |t| → compute their means and SDs → standardise → tune the L2 penalty by inner CV → fit logistic regression.',
    '<b>Outer loop:</b> split patients into 5 stratified folds. For each fold, run the <i>entire</i> pipeline on the other 4 folds only, then score the held-out fold.',
    'Pool the out-of-fold scores and compute one AUC. Repeat with different fold splits (here 8 times) and report mean ± SD.',
    '<b>Leaky version</b> (what not to do): run gene selection once on all patients, then cross-validate only the model fit. The selection step has seen every test patient\'s label.',
    '<b>Null check:</b> shuffle the response labels and run the identical procedure. An honest pipeline must give AUC ≈ 0.5. If it does not, something is leaking.',
  ]);
  math(main, 'nested CV, for each outer fold f:\n    pipeline_f = fit(select genes + scale + tune + model,  all data except f)\n    scores_f   = pipeline_f(f)\nAUC = P(score of a random responder > score of a random non-responder)        chance = 0.5');

  callout(main, '', 'In biology', 'This exact error was documented in the microarray era: Ambroise &amp; McLachlan (<i>PNAS</i> 2002) showed published cancer classifiers reaching near-zero error after "selection bias", and Simon et al. (<i>JNCI</i> 2003) showed the same with random labels. Tools that make the honest version the default: <b>scikit-learn</b> <code>Pipeline</code> inside <code>cross_val_score</code> / <code>GridSearchCV</code>, <b>caret</b> and <b>tidymodels</b> recipes in R, <b>mlr3</b> with nested resampling. Batch correction, normalisation factors, imputation and SMOTE oversampling all belong inside the pipeline for the same reason. Reporting guidelines for clinical prediction models (TRIPOD+AI) ask for exactly this.');
  callout(main, 'warn', 'Pitfalls', `<ul>
    <li><b>Any step that looks at labels must be inside the fold:</b> gene filtering by DE, choosing k, tuning penalties, picking the "best" model.</li>
    <li><b>Unsupervised steps can leak too</b> (batch correction or PCA on all samples), usually less dramatically. For a deployable model, freeze them on the training set.</li>
    <li><b>Patients, not samples:</b> if one patient has several biopsies, keep them in the same fold.</li>
    <li><b>Nested CV estimates the pipeline, not a specific gene list.</b> Each fold may pick different genes. The final model is refitted on all data; its performance is what nested CV estimated, to be confirmed on an <a href="#/omics/bulk-validation">external cohort</a>.</li>
    <li>With ${C.n_resp} responders out of ${C.n}, even honest AUCs have wide confidence intervals; differences of a few hundredths between models are rarely meaningful (<a href="#/omics/bulk-model-compare">model comparison</a>).</li>
  </ul>`);
  seeAlso(main, [['omics', 'bulk-feature-selection', 'Previous: feature selection'], ['omics', 'bulk-model-compare', 'Next: model comparison'], ['concepts', 'cross-validation', 'Cross-validation'], ['concepts', 'train-test', 'Train, validation, test'], ['concepts', 'metrics', 'ROC & AUC'], ['frontier', 'clinical-signature', 'Small signature + clinical score']]);
}
