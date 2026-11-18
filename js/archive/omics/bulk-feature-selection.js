// Bulk step 5: biomarker feature selection with lasso / elastic-net paths and stability selection.
// Driven by data/omics/bulk_fs.json.
import { header, figure, slider, select, button, stats, section, steps, callout, math, legend, seeAlso, animate, tip, untip, el } from '../../lib/ui.js';
import { frame, css, ink, ink2, muted } from '../../lib/plot.js';
import { stepCard, omicsData } from '../../omics/_shared.js';
import { GROUP_ORDER, groupColor, groupLabel, f3 } from '../../omics/_bulk_common.js';
const d3 = window.d3;

const PNAME = { lasso: 'lasso (L1)', enet: 'elastic net (50% L1, 50% L2)' };

export default async function render(main) {
  header(main, 'Biomarker feature selection',
    'From about two thousand genes, pick a handful that predict response. The <b>lasso</b> does this by penalising the size of coefficients so hard that most become exactly zero. The catch: with <b>correlated genes</b> it picks one almost at random, so the specific genes it picks are less trustworthy than the biology they point to.');
  stepCard(main, {
    stage: 'Bulk RNA-seq · step 5 of 11',
    question: 'Which small set of genes carries the information about response, so that a cheap clinical assay could measure it?',
    data: 'Standardised, batch-corrected log-CPM (patients × genes) and the response label.',
    method: 'L1-penalised (lasso) and elastic-net logistic regression along a path of penalty strengths λ; cross-validated AUC with the one-standard-error rule; stability selection on 100 half-samples.',
    verdict: 'ml', why: ' A penalised predictive model, tuned by cross-validation for out-of-sample performance. The penalty trades a little bias for much less variance: classic statistical learning.',
  });
  const S = await omicsData(main, 'bulk_fs.json');
  if (!S) return;
  const L = S.lambda, nL = L.length;

  // ------------------------------------------------------------------ figure 1: regularisation path
  const fig = figure(main, 'The regularisation path',
    'Top: each line is one gene\'s coefficient as the penalty λ weakens from left to right (the 70 genes that enter first are drawn). Bottom: cross-validated AUC (3 × 5-fold) at each λ, with ±1 SD band. Drag the slider or press Sweep. Hover a line.');
  let pname = 'lasso', li = S.paths.lasso.one_se, stop = null;
  select(fig.controls, { label: 'Penalty', value: pname, options: Object.entries(PNAME), onChange: (v) => { pname = v; li = S.paths[v].one_se; sl.set(li); draw(); } });
  const sl = slider(fig.controls, { label: 'λ index (weaker →)', min: 0, max: nL - 1, step: 1, value: li, fmt: (v) => `λ = ${L[v].toPrecision(2)}`, onInput: (v) => { li = v; update(); } });
  button(fig.controls, 'Sweep', () => {
    if (stop) stop();
    let acc = 0; li = 0; sl.set(0); update();
    stop = animate((dt) => { acc += dt; if (acc < 160) return true; acc = 0; li += 1; sl.set(li); update(); return li < nL - 1; });
  }, true);
  button(fig.controls, 'Jump to 1-SE choice', () => { li = S.paths[pname].one_se; sl.set(li); update(); });
  legend(fig.controls, GROUP_ORDER.map((g) => [groupLabel(g), groupColor(g)]));
  const rd = stats(fig.root);
  const chips = el('div', { class: 'prose', style: 'max-width:none' }); fig.root.append(chips);
  const host = el('div'); fig.body.append(host);
  const xS = () => d3.scaleLog().domain([L[0], L[nL - 1]]);
  let FP, FA, vlineP, vlineA, lines;

  function draw() {
    host.innerHTML = '';
    const P = S.paths[pname];
    const ym = d3.max(P.coef.flat(), Math.abs) * 1.05;
    FP = frame(host, { width: 900, height: 300, x: [L[0], L[nL - 1]], y: [-ym, ym], yLabel: 'coefficient (per SD)', xScale: xS() });
    FP.g.select('.axis.x').call(d3.axisBottom(FP.x).ticks(6, '.2~g'));
    FP.layer('zero').append('line').attr('x1', 0).attr('x2', FP.w).attr('y1', FP.y(0)).attr('y2', FP.y(0)).attr('stroke', css('--axis'));
    const ord = d3.range(P.genes.length).sort((a, b) => (P.group[b] === 'filler') - (P.group[a] === 'filler'));
    lines = FP.layer('paths').selectAll('path').data(ord).join('path').attr('fill', 'none')
      .attr('stroke', (j) => groupColor(P.group[j])).attr('stroke-width', (j) => (P.group[j] === 'filler' ? 1.2 : 2))
      .attr('d', (j) => d3.line().x((c, k) => FP.x(L[k])).y((c) => FP.y(c))(P.coef[j]))
      .on('mousemove', (e, j) => tip(e, `<b>${P.genes[j]}</b> · ${groupLabel(P.group[j])}<br>coef at current λ: ${P.coef[j][li].toFixed(3)}<br>enters at λ = ${L[P.coef[j].findIndex((c) => c !== 0)].toPrecision(2)}`))
      .on('mouseleave', untip);
    vlineP = FP.layer('v').append('line').attr('y1', 0).attr('y2', FP.h).attr('stroke', ink()).attr('stroke-width', 1.5);

    FA = frame(host, { width: 900, height: 220, x: [L[0], L[nL - 1]], y: [0.4, 0.9], xLabel: 'penalty λ (log scale; left = strong, right = weak)', yLabel: 'CV AUC', xScale: xS() });
    FA.g.select('.axis.x').call(d3.axisBottom(FA.x).ticks(6, '.2~g'));
    const pts = L.map((l, k) => ({ l, m: P.auc_mean[k], s: P.auc_sd[k] || 0, n: P.nnz[k] }));
    FA.layer('band').append('path').datum(pts).attr('fill', css('--accent-soft')).attr('d', d3.area().x((d) => FA.x(d.l)).y0((d) => FA.y(d.m - d.s)).y1((d) => FA.y(d.m + d.s)));
    FA.layer('line').append('path').datum(pts).attr('fill', 'none').attr('stroke', css('--accent')).attr('stroke-width', 2).attr('d', d3.line().x((d) => FA.x(d.l)).y((d) => FA.y(d.m)));
    FA.layer('pts').selectAll('circle').data(pts).join('circle').attr('cx', (d) => FA.x(d.l)).attr('cy', (d) => FA.y(d.m)).attr('r', 4).attr('fill', css('--accent')).attr('stroke', css('--surface')).attr('stroke-width', 1.2)
      .on('mousemove', (e, d) => tip(e, `λ = ${d.l.toPrecision(2)}<br>CV AUC ${d.m.toFixed(3)} ± ${d.s.toFixed(3)}<br>${d.n} genes non-zero`)).on('mouseleave', untip);
    FA.layer('chance').append('line').attr('x1', 0).attr('x2', FA.w).attr('y1', FA.y(0.5)).attr('y2', FA.y(0.5)).attr('stroke', muted()).attr('stroke-dasharray', '3 3');
    [['best', P.best], ['1-SE', P.one_se]].forEach(([t, k], z) => {
      FA.layer('mk').append('text').attr('x', FA.x(L[k])).attr('y', FA.y(P.auc_mean[k]) - 12 - z * 13).attr('text-anchor', 'middle').attr('font-size', 11).attr('fill', ink2()).text(t);
    });
    vlineA = FA.layer('v').append('line').attr('y1', 0).attr('y2', FA.h).attr('stroke', ink()).attr('stroke-width', 1.5);
    update();
  }
  function update() {
    const P = S.paths[pname];
    vlineP.attr('x1', FP.x(L[li])).attr('x2', FP.x(L[li]));
    vlineA.attr('x1', FA.x(L[li])).attr('x2', FA.x(L[li]));
    lines.attr('opacity', (j) => (P.coef[j][li] !== 0 ? 1 : 0.35));
    const on = d3.range(P.genes.length).filter((j) => P.coef[j][li] !== 0).sort((a, b) => Math.abs(P.coef[b][li]) - Math.abs(P.coef[a][li]));
    const ifng = on.filter((j) => P.group[j] === 'IFNG module').length;
    rd.set({ λ: L[li].toPrecision(2), 'non-zero genes': P.nnz[li], 'CV AUC': `${P.auc_mean[li].toFixed(3)} ± ${(P.auc_sd[li] || 0).toFixed(3)}`, 'IFN-γ module genes in': `${ifng} of 12`, 'filler genes in': on.filter((j) => P.group[j] === 'filler').length + (P.nnz[li] > P.genes.length ? '+' : '') });
    chips.innerHTML = `<p class="fig-caption" style="margin:4px 0">Selected at this λ (largest |coefficient| first${P.nnz[li] > on.length ? `; ${on.length} of ${P.nnz[li]} tracked` : ''}):</p>` +
      (on.length ? on.slice(0, 30).map((j) => `<span class="tag" style="border-left:6px solid ${groupColor(P.group[j])}">${P.genes[j]} ${P.coef[j][li] > 0 ? '+' : '−'}${Math.abs(P.coef[j][li]).toFixed(2)}</span>`).join('') + (on.length > 30 ? ` <span class="tag">+${on.length - 30} more</span>` : '') : '<span class="tag">none: the penalty is too strong, every coefficient is zero</span>');
  }
  draw();

  // ------------------------------------------------------------------ figure 2: stability selection
  const fig2 = figure(main, 'Stability selection: would we pick the same genes again?',
    'The lasso was refitted on 100 random halves of the patients at a fixed λ. Bars show how often each gene was selected (every gene selected at least once, up to 40). Genes above the threshold are "stable". Hover a bar.');
  let strength = 'medium', thr = 0.6;
  const SNAME = { strong: 'strong penalty', medium: 'medium penalty', weak: 'weak penalty' };
  select(fig2.controls, { label: 'Penalty', value: strength, options: Object.entries(SNAME).map(([k, v]) => [k, `${v} (λ = ${(1 / (S.stability[k].C * S.n)).toPrecision(2)})`]), onChange: (v) => { strength = v; draw2(); } });
  slider(fig2.controls, { label: 'Stability threshold π', min: 0.3, max: 0.95, step: 0.05, value: thr, fmt: (v) => v.toFixed(2), onInput: (v) => { thr = v; draw2(); } });
  const rd2 = stats(fig2.root);
  const FS = frame(fig2.body, { width: 900, height: 280, x: [0, 1], y: [0, 1], yLabel: 'selection frequency', xScale: d3.scaleBand().padding(0.2), margin: { bottom: 70 } });
  const thrL = FS.layer('thr').append('line').attr('x1', 0).attr('x2', FS.w).attr('stroke', ink()).attr('stroke-dasharray', '5 4').attr('stroke-width', 1.5);
  function draw2() {
    const st = S.stability[strength];
    let idx = d3.range(st.genes.length).filter((i) => st.freq[i] > 0);
    if (!idx.length) idx = [0]; // nothing ever selected: keep one empty slot so the axes still draw
    FS.x.domain(idx);
    FS.g.select('.axis.x').call(d3.axisBottom(FS.x).tickFormat((i) => (st.freq[i] > 0 ? st.genes[i] : 'no gene selected'))).selectAll('text').attr('transform', 'rotate(-55)').attr('text-anchor', 'end').attr('dx', '-0.4em').attr('dy', '0.2em');
    FS.layer('bars').selectAll('rect').data(idx).join('rect').attr('x', (i) => FS.x(i)).attr('width', FS.x.bandwidth()).attr('rx', 2)
      .attr('y', (i) => FS.y(st.freq[i])).attr('height', (i) => FS.h - FS.y(st.freq[i])).attr('fill', (i) => groupColor(st.group[i])).attr('opacity', (i) => (st.freq[i] >= thr ? 1 : 0.45))
      .on('mousemove', (e, i) => tip(e, `<b>${st.genes[i]}</b> · ${groupLabel(st.group[i])}<br>selected in ${Math.round(100 * st.freq[i])} of 100 half-samples`)).on('mouseleave', untip);
    thrL.attr('y1', FS.y(thr)).attr('y2', FS.y(thr));
    const stable = idx.filter((i) => st.freq[i] >= thr).length;
    const ifngSum = d3.sum(idx.filter((i) => st.group[i] === 'IFNG module'), (i) => st.freq[i]);
    const bound = thr > 0.5 ? (st.q ** 2 / ((2 * thr - 1) * S.p)) : null;
    rd2.set({ 'genes selected per half-sample (q)': st.q.toFixed(2), 'stable genes at π': stable, 'bound on expected false selections': bound == null ? 'needs π > 0.5' : bound.toFixed(3), 'IFN-γ genes selected per half-sample': ifngSum.toFixed(2) });
  }
  draw2();

  // numbers for prose
  const PL = S.paths.lasso, PE = S.paths.enet;
  const ifngIn = (P, k) => d3.range(P.genes.length).filter((j) => P.group[j] === 'IFNG module' && P.coef[j][k] !== 0).length;
  const med = S.stability.medium, wk = S.stability.weak;
  const medIfng = d3.sum(med.genes.map((g, i) => (med.group[i] === 'IFNG module' ? med.freq[i] : 0)));
  const fillerHi = wk.genes.map((g, i) => ({ g, f: wk.freq[i], grp: wk.group[i] })).filter((d) => d.grp === 'filler').slice(0, 2);
  section(main, 'What to notice', `<ul>
    <li><b>Sweep the lasso path.</b> The first genes to enter are ${PL.genes.slice(0, 3).join(', ')}, all from the planted IFN-γ module. CV AUC jumps from 0.5 to ${f3(PL.auc_mean[PL.best])} by λ = ${L[PL.best].toPrecision(2)} (${PL.nnz[PL.best]} genes), then slowly declines as filler genes flood in: by the weakest penalty ${PL.nnz[nL - 1]} genes are in and AUC is ${f3(PL.auc_mean[nL - 1])}. That decline is overfitting.</li>
    <li><b>The one-standard-error rule</b> picks the most penalised model whose AUC is within one SE of the best: λ = ${L[PL.one_se].toPrecision(2)}, with ${PL.nnz[PL.one_se]} genes (${S.final.genes.join(', ')}). Simpler, almost as good, and more likely to replicate.</li>
    <li>The module has 12 genes, all genuinely predictive and highly correlated with each other. At the 1-SE choice the lasso keeps ${ifngIn(PL, PL.one_se)} of them. <b>Switch to the elastic net</b>: at its own 1-SE choice ${ifngIn(PE, PE.one_se)} module genes are in, with smaller, shared coefficients. The L2 part makes correlated genes enter together.</li>
    <li><b>Stability, medium penalty:</b> no gene reaches the usual threshold of 0.6 (the best is ${med.genes[0]} at ${med.freq[0].toFixed(2)}). But the IFN-γ genes together are selected ${medIfng.toFixed(2)} times per half-sample out of q = ${med.q.toFixed(2)}. The <i>module</i> is rock-stable; <i>which member</i> carries it is a coin toss. Report the module, not the gene.</li>
    <li><b>Weak penalty:</b> q rises to ${wk.q.toFixed(1)} and filler genes such as ${fillerHi.map((d) => `${d.g} (${d.f.toFixed(2)})`).join(' and ')} appear among the top. Lower the threshold to 0.3 and they count as "stable"; the error bound then no longer applies. <b>Strong penalty:</b> nothing is ever selected on half the data.</li>
  </ul>`);
  steps(main, 'How it works', [
    'Standardise every gene (mean 0, SD 1) so the penalty treats them equally.',
    'Fit logistic regression while adding λ·Σ|β| (lasso) or λ·(α·Σ|β| + (1−α)/2·Σβ²) (elastic net) to the loss. The L1 term has a corner at zero, so small coefficients become exactly zero (see <a href="#/archive/topics/regularization">regularisation</a>).',
    'Repeat for a grid of λ from "everything zero" to "almost unpenalised", warm-starting each fit from the last. This is the path.',
    'Choose λ by cross-validated AUC, standardising inside each fold. Prefer the 1-SE choice over the maximum.',
    '<b>Stability selection:</b> draw many half-samples, fit the lasso at a fixed λ on each, and count how often each gene is selected. Keep genes above a threshold π. Meinshausen and Bühlmann give a bound on the expected number of false selections.',
    'Treat the selected genes as a hypothesis about biology. The honest estimate of predictive performance needs the whole selection inside cross-validation: the <a href="#/archive/omics/bulk-classifier">next step</a>.',
  ]);
  math(main, 'lasso:        min_β  −loglik(β) + λ Σ_j |β_j|\nelastic net:  min_β  −loglik(β) + λ [ α Σ|β_j| + (1−α)/2 Σ β_j² ]\nstability:    E[false selections] ≤ q² / ((2π − 1) · p)        (q = mean genes selected, p = genes)');

  callout(main, '', 'In biology', '<b>glmnet</b> (R; <code>cv.glmnet</code> with <code>lambda.1se</code>) and <b>scikit-learn</b> (<code>LogisticRegressionCV</code> with <code>penalty="l1"</code> or <code>"elasticnet"</code>) are the workhorses; <b>stabs</b> in R implements stability selection. Clinical expression signatures were built this way, e.g. Oncotype-style recurrence scores and immunotherapy response signatures. Group-aware penalties (<b>group lasso</b>, <code>grpreg</code>) or selecting pre-defined gene sets (pathway scores, GSVA) tackle the correlated-gene problem head on, and are why many immuno-oncology "signatures" are module averages rather than lasso picks.');
  callout(main, 'warn', 'Pitfalls', `<ul>
    <li><b>Interpreting the chosen genes as "the" biomarkers.</b> Rerun on slightly different patients and the lasso picks different members of the same module. Biological claims need the module, the stability, or experiments.</li>
    <li><b>Selecting genes on all patients, then cross-validating a model on them.</b> The CV AUC is then optimistic. Selection is part of training.</li>
    <li><b>Lasso coefficients are biased towards zero</b> and have no simple p-values. Use them to predict, not to estimate effect sizes.</li>
    <li>Here batch correction was fitted on all patients before selection. That leaks a little information too; a locked model needs a locked correction.</li>
  </ul>`);
  seeAlso(main, [['omics', 'bulk-de', 'Previous: differential expression'], ['omics', 'bulk-classifier', 'Next: classifier & nested CV'], ['concepts', 'regularization', 'Regularization (L1 / L2)'], ['concepts', 'cross-validation', 'Cross-validation'], ['omics', 'bulk-survival', 'Cox-lasso survival']]);
}
