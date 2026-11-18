// Random forests and gradient boosting, lesson format.
// Arc: what → why → where → how it works (6 steps: bagging → random feature subsets + out-of-bag error → boosting as residual
//      fitting → boosting hyperparameters and early stopping → feature importance + SHAP → head-to-head lab vs logistic regression)
//      → pitfalls → recap. Every number in the prose is computed from data/study/cohort.json at render time.
import { header, figure, slider, select, button, checkbox, stats, legend, seeAlso, tip, untip, el, html } from '../lib/ui.js';
import { frame, decisionCanvas, cat, css, ink, ink2, muted } from '../lib/plot.js';
import { sigmoid, mean, rng } from '../lib/data.js';
import { fitLogistic, predictProb, logLoss, auc, roc, standardize, stratifiedSplit, fitForest, forestProb, forestStaged, aucDefined, fitBoost, boostRaw, boostStaged, treeValue, permImportance, exactShapley, fitTree } from '../lib/models.js';
import { loadCohort, FEATURES, studyCard } from '../lib/study.js';
import { lessonNav, chapter, prose, rung, worked, quiz, cards, table, code, recap } from '../lib/lesson.js';
const d3 = window.d3;

const f2 = (v) => (Math.abs(v) < 0.005 ? '0.00' : v.toFixed(2));
const f3 = (v) => v.toFixed(3);
const f3m = (v) => v.toFixed(3).replace('-', '−');
const pc = (v) => Math.round(100 * v) + '%';
const sg = (v) => (v < 0 ? '−' : '+') + f2(Math.abs(v));
const sum = (a) => a.reduce((s, v) => s + v, 0);
const jit = (i) => (((i * 7919) % 101) / 101 - 0.5) * 0.09;
const corr = (a, b) => { const ma = mean(a), mb = mean(b); let s = 0, x = 0, z = 0; a.forEach((v, i) => { s += (v - ma) * (b[i] - mb); x += (v - ma) ** 2; z += (b[i] - mb) ** 2; }); return s / Math.sqrt(x * z || 1); };
const K10 = ['ifng', 'log2tmb', 'caf', 'pdl1', 'cd8', 'prolif', 'age', 'smoker', 'male', 'centreB'];
const K3 = K10.slice(0, 3), K2 = ['ifng', 'log2tmb'];
const ROLE = { ifng: 'true driver', log2tmb: 'true driver', caf: 'true driver', pdl1: 'proxy of IFN-γ', cd8: 'proxy of IFN-γ', prolif: 'no effect on response', age: 'no effect', smoker: 'acts only via TMB', male: 'no effect', centreB: 'confounded', noise: 'random control' };

export default async function render(main) {
  const C = await loadCohort();
  const D = C.discovery, V = C.validation;
  const y = D.map((p) => p.response), yv = V.map((p) => p.response), n = D.length, nR = sum(y);
  // a random-number column with no relation to anything: a control for the importance figure
  const rn = rng(77), NOISE = new Map(); [...D, ...V].forEach((p) => NOISE.set(p, rn()));
  const FE = { ...FEATURES, noise: { label: 'Random noise (control)', get: (p) => NOISE.get(p) } };
  const K11 = [...K10, 'noise'];
  const rowsOf = (pts, keys) => pts.map((p) => keys.map((k) => FE[k].get(p)));
  const memo = new Map(), once = (key, fn) => { if (!memo.has(key)) memo.set(key, fn()); return memo.get(key); };
  const RFo = (m, t = 100, d = 6) => ({ nTrees: t, maxDepth: d, minLeaf: 3, mtry: m, seed: 5 });
  const GBo = (t = 100, lr = 0.1, d = 2) => ({ nTrees: t, lr, maxDepth: d, minLeaf: 3, seed: 1 });
  const getRF = (keys, o) => once('rf' + keys + JSON.stringify(o), () => fitForest(rowsOf(D, keys), y, o));
  const getGB = (keys, o) => once('gb' + keys + JSON.stringify(o), () => fitBoost(rowsOf(D, keys), y, o));
  const getLR = (keys) => once('lr' + keys, () => { const st = standardize(rowsOf(D, keys)), f = fitLogistic(st.Z, y); return (M) => M.map((r) => predictProb(st.apply(r), f.b0, f.w)); });
  const rfScore = (F) => (M) => M.map((r) => forestProb(F, r));
  const gbScore = (B, k) => (M) => M.map((r) => sigmoid(boostRaw(B, r, k)));
  const X10 = rowsOf(D, K10), X10v = rowsOf(V, K10);
  // one evaluation of the three model families on the same features (used by the lab, the try-table and the pitfalls)
  const evalCfg = ({ keys, rfTrees = 100, rfDepth = 6, rfMtry = 3, gbTrees = 100, gbLr = 0.1, gbDepth = 2 }) => {
    const Xt = rowsOf(D, keys), Xv = rowsOf(V, keys), p = keys.length;
    const lr = getLR(keys), rf = rfScore(getRF(keys, RFo(Math.min(rfMtry, p), rfTrees, rfDepth))), gb = gbScore(getGB(keys, GBo(gbTrees, gbLr, gbDepth)), gbTrees);
    const out = { Xt, Xv };
    [['lr', lr], ['rf', rf], ['gb', gb]].forEach(([k, f]) => { const st = f(Xt), sv = f(Xv); out[k] = { st, sv, tr: auc(st, y), va: auc(sv, yv), ll: logLoss(sv, yv), mp: mean(sv) }; });
    return out;
  };
  const base = nR / n;

  header(main, 'Random forests and gradient boosting',
    'Two ways of combining many decision trees into one predictor. A <b>random forest</b> averages trees grown independently on resampled data; <b>gradient boosting</b> adds trees one at a time, each fitted to the errors the previous trees left. Both capture thresholds and interactions without being told about them. Whether that helps depends on the data, and on this study a plain logistic regression remains a strong competitor.');
  lessonNav(main);
  studyCard(main, C, '<b>Question for this page:</b> does combining many trees predict response in the validation centre better than logistic regression, and which measurements does the ensemble rely on?');

  // =====================================================================================
  // 1. WHAT
  // =====================================================================================
  let ch = chapter(main, 'what', 'What are tree ensembles?', 'In one line: <b>many weak, different trees → one stronger prediction</b>, by averaging (forest) or by adding corrections (boosting).');
  const F10 = getRF(K10, RFo(3)), B10 = getGB(K10, GBo());
  const exRow = (P) => K10.map((k) => FE[k].get(P));
  const stumpThr = fitTree(rowsOf(D, ['ifng']), y, { maxDepth: 1, minLeaf: 5 }).thr;
  const exP = D.filter((p) => p.response).sort((a, b) => Math.abs(forestProb(F10, exRow(a)) - 0.7) - Math.abs(forestProb(F10, exRow(b)) - 0.7))[0];
  prose(ch, `<p>A <b>decision tree</b> predicts by asking a sequence of yes/no questions about the measurements ("IFN-γ below ${stumpThr.toFixed(2)}?") and returning the response rate of the patients who ended up in the same leaf. A single tree is easy to read but unstable (chapter 2). A <b>tree ensemble</b> trains many trees and combines their outputs. The two standard recipes differ in how the trees are made and how they are combined:</p>
    <ul><li><b>Random forest</b> (bagging): each tree is trained on a <b>bootstrap resample</b> of the patients, and considers only a random subset of the features at each split. The trees are independent; their predictions are <b>averaged</b>.</li>
    <li><b>Gradient boosting</b> (XGBoost, LightGBM, scikit-learn): small trees are trained <b>sequentially</b>. Each new tree is fitted to the <b>residual errors</b> of the sum of all previous trees, and its output is added after being scaled down by a <b>learning rate</b>.</li></ul>
    <p>The diagram follows patient ${exP.id}, who responded, through a forest and a boosted model fitted to ten measurements on the ${n} discovery patients.</p>`);
  ch.append(ensembleSvg(exP, exRow(exP), F10, B10));
  prose(ch, '<p>Key terms:</p>');
  const nIn = mean(F10.trees.map((t) => t.inbag.filter((v) => v > 0).length));
  table(ch, [
    { t: 'Bootstrap resample', m: 'A training set of the same size drawn <i>with replacement</i>; some patients appear several times, others not at all', s: `${Math.round(nIn)} of ${n} distinct patients on average (${pc(nIn / n)})` },
    { t: 'Bagging', m: 'Bootstrap aggregating: train one model per resample and average them', s: `${F10.trees.length} trees averaged in the forest above` },
    { t: 'mtry', m: 'Number of features a split is allowed to consider; drawn afresh at every node', s: `3 of ${K10.length} features per split (about √p)` },
    { t: 'Out-of-bag (OOB)', m: 'The patients a given tree did not see; they provide a built-in held-out test', s: `${n - Math.round(nIn)} patients per tree, about ${pc(1 - nIn / n)} of the cohort` },
    { t: 'Boosting', m: 'Adding trees one at a time, each fitted to the residuals of the current model', s: `${B10.trees.length} trees of depth 2, starting from the base rate (${pc(base)})` },
    { t: 'Learning rate (shrinkage)', m: 'Factor that scales each new tree before it is added; small values need more trees', s: 'η = 0.1 above' },
    { t: 'Feature importance', m: 'A summary of how much the fitted ensemble relies on each measurement (impurity, permutation, SHAP)', s: 'chapter 4, step 5' },
  ], [['t', 'Term'], ['m', 'Meaning'], ['s', 'In our study']]);

  // =====================================================================================
  // 2. WHY
  // =====================================================================================
  ch = chapter(main, 'why', 'Why do we need them?', 'What a single tree gets wrong, and what combining many trees changes.');
  prose(ch, `<p>A fully grown tree fits its training patients almost perfectly, so small changes in the training set change which questions it asks and where it splits. Its predictions on new patients therefore vary widely from one training sample to the next (<b>high variance</b>). Averaging many such trees keeps their flexibility, so thresholds and interactions can be captured, while the variance falls. Boosting addresses the opposite problem: many very shallow trees have high bias individually, and adding them one by one reduces it.</p>`);
  const fy = figure(ch, 'One tree is an unstable predictor', 'Each dot is a separate decision tree grown on a different bootstrap resample of the discovery patients (all ten measurements) and scored by AUC on the validation centre. Change the maximum depth to see how the spread of single trees compares with the average of all 40 and with logistic regression on the same measurements.');
  let yd = 6;
  select(fy.controls, { label: 'Tree depth', value: yd, options: [[2, '2'], [4, '4'], [6, '6'], [10, '10 (almost unrestricted)']], onChange: (v) => { yd = +v; paintY(); } });
  const sY = stats(fy.root);
  const FY = frame(fy.body, { width: 820, height: 250, x: [0.5, 0.95], y: [0, 1], xLabel: 'validation AUC (Centre C)', grid: false, axes: false, margin: { left: 20, right: 20, top: 50, bottom: 44 } });
  FY.g.append('g').attr('class', 'axis x').attr('transform', `translate(0,${FY.h})`).call(d3.axisBottom(FY.x).ticks(9));
  const lrAll = auc(getLR(K10)(X10v), yv), lr3 = auc(getLR(K3)(rowsOf(V, K3)), yv);
  const lyr = FY.layer('lines'), dyr = FY.layer('dots');
  function paintY() {
    const F = getRF(K10, { nTrees: 40, maxDepth: yd, minLeaf: 1, mtry: 10, seed: 21 });
    const a = F.trees.map((t) => auc(X10v.map((r) => treeValue(t.root, r)), yv)), avg = auc(X10v.map((r) => forestProb(F, r)), yv);
    dyr.selectAll('circle').data(a).join('circle').attr('cx', (v) => FY.x(v)).attr('cy', (v, i) => FY.h * (0.2 + 0.72 * (((i * 37) % 40) / 39))).attr('r', 5).attr('fill', ink2()).attr('fill-opacity', 0.6).attr('stroke', css('--surface'))
      .on('mousemove', (e, v) => tip(e, `single tree · validation AUC ${f3(v)}`)).on('mouseleave', untip);
    const L = [{ x: avg, c: cat(4), t: `average of the 40 trees ${f3(avg)}`, up: true }, { x: lrAll, c: cat(2), t: `logistic regression ${f3(lrAll)}`, up: false }];
    lyr.selectAll('line').data(L).join('line').attr('x1', (d) => FY.x(d.x)).attr('x2', (d) => FY.x(d.x)).attr('y1', 0).attr('y2', FY.h).attr('stroke', (d) => d.c).attr('stroke-width', 2.5);
    lyr.selectAll('text').data(L).join('text').attr('x', (d) => FY.x(d.x) - 6).attr('y', (d) => (d.up ? -8 : -28)).attr('text-anchor', 'end').attr('font-size', 12).attr('fill', ink()).text((d) => d.t);
    sY.set({ 'mean single tree': f3(mean(a)), 'worst / best single tree': `${f3(d3.min(a))} / ${f3(d3.max(a))}`, 'average of 40 trees': f3(avg), 'logistic regression': f3(lrAll) });
    paintY.res = { a, avg };
  }
  paintY();
  legend(fy.body, [['single tree', ink2()], ['average of 40 trees', cat(4)], ['logistic regression', cat(2)]]);
  const yr = paintY.res;
  prose(ch, `<p>At depth ${yd}, single trees score ${f3(d3.min(yr.a))} to ${f3(d3.max(yr.a))} on the validation centre (mean ${f3(mean(yr.a))}); their average scores ${f3(yr.avg)}. The ensemble is a better predictor than any typical member, which is the reason for the method. The logistic regression line (${f3(lrAll)}) is a reminder that the comparison that matters is with a simple model, not only with a single tree.</p>`);
  table(ch, [
    { a: 'Single decision tree', i: 'easy to read, unstable', x: 'thresholds, interactions', c: 'low', d: 'small' },
    { a: 'Logistic regression', i: 'weights, odds ratios', x: 'only if added by hand', c: 'low', d: 'small to moderate' },
    { a: '<b>Random forest</b>', i: 'importance scores only', x: 'automatic', c: 'low (few settings to tune)', d: 'moderate' },
    { a: '<b>Gradient boosting</b>', i: 'importance scores, SHAP', x: 'automatic, depth sets the order', c: 'moderate (rate, depth, trees)', d: 'moderate to large' },
  ], [['a', 'Approach'], ['i', 'Interpretability'], ['x', 'Non-linearity and interactions'], ['c', 'Tuning effort'], ['d', 'Data needed']]);

  // =====================================================================================
  // 3. WHERE
  // =====================================================================================
  ch = chapter(main, 'where', 'Where are they used?', 'Tabular data with mixed scales, thresholds or interactions, and moderate numbers of samples.');
  cards(ch, [
    { tag: 'Clinical prediction', title: 'Response, risk and outcome models', body: 'Gradient-boosted trees are a standard strong baseline for tabular clinical and biomarker data (lab values, mutation counts, scores). Random forests are the usual first non-linear comparison.' },
    { tag: 'Variant filtering', title: 'Separating real variants from artefacts', body: 'Variant callers and pathogenicity predictors combine many annotations (depth, strand bias, conservation scores) with tree ensembles, because the annotations differ in scale and interact.' },
    { tag: 'Gene regulatory networks', title: 'GENIE3 and GRNBoost2', body: 'Each gene\'s expression is predicted from all transcription factors with a forest (GENIE3) or boosted trees (GRNBoost2, used in SCENIC). The feature importances become candidate regulatory links.' },
    { tag: 'Microbiome', title: 'Taxon tables to phenotype', body: 'Random forests are widely used on 16S and metagenomic abundance tables: counts are skewed, features outnumber samples, and OOB error gives a quick estimate without a separate split.' },
    { tag: 'Biomarker discovery', title: 'Ranking genes, CpGs or proteins', body: 'Importance scores from a forest or from SHAP on boosted trees shortlist candidate markers, to be confirmed in an independent cohort (see the pitfalls on importance).' },
    { tag: 'Survival', title: 'Random survival forests', body: 'The same bagging idea applied to time-to-event outcomes: leaves hold survival curves, and the forest averages them.' },
    { tag: 'Missing values', title: 'Imputation with missForest', body: 'Each variable with gaps is predicted from the others by a forest, iterating until the imputed values stabilise.' },
    { tag: 'Quality control', title: 'Flagging bad samples or cells', body: 'A handful of QC metrics with non-linear thresholds (library size, mitochondrial fraction, duplication rate) feed a classifier that outputs a pass probability.' },
    { tag: 'Outside biology', title: 'Credit, fraud and ranking', body: 'LightGBM and XGBoost dominate many tabular-data competitions and production ranking systems.' },
  ]);
  prose(ch, '<p><b>When not to use them:</b> the signal is close to linear and the sample is small (a regularised logistic regression is usually as accurate and also gives effect sizes, see the head-to-head in chapter 4); inference about an effect is needed (a tree ensemble gives no odds ratio or confidence interval); the model must extrapolate beyond the measured range (trees predict a constant outside it); or the inputs are raw images or sequences (neural networks learn features first).</p>');

  // =====================================================================================
  // 4. HOW
  // =====================================================================================
  ch = chapter(main, 'how', 'How it works', 'Six steps: bagging, random feature subsets with out-of-bag error, boosting, its hyperparameters, feature importance, and a head-to-head with logistic regression.');

  // ---- Step 1: bagging
  const X2 = rowsOf(D, K2), X2v = rowsOf(V, K2);
  const F2 = fitForest(X2, y, { nTrees: 100, maxDepth: 8, minLeaf: 2, seed: 11 });
  const tA = F2.trees.map((t) => auc(X2v.map((r) => treeValue(t.root, r)), yv)), eA = forestStaged(F2, X2v).map((s) => auc(s, yv));
  const lr2 = auc(getLR(K2)(X2v), yv);
  let rb = rung(ch, 1, 'Bagging: average trees grown on bootstrap resamples',
    `<p>Training patients are drawn <b>with replacement</b> to form a resample of the same size, a tree is grown on it, and the process is repeated. Each tree sees a different version of the data and so draws slightly different boundaries; the <b>average</b> of their outputs is the forest's probability. The figure uses two measurements (IFN-γ and log2 TMB) so the boundaries can be drawn. Patients drawn as open rings are absent from the displayed tree's resample.</p>`);
  const f1 = figure(rb, 'One tree vs the average of many', 'Shading: predicted class (blue = no response, orange = response; stronger = more confident). Left: a single deep tree. Right: the average of the first k trees. Bottom: validation AUC of every individual tree (dots) and of the running average (line).');
  let ti = 0, kk = 30;
  slider(f1.controls, { label: 'single tree shown (#)', min: 1, max: 100, step: 1, value: 1, onInput: (v) => { ti = v - 1; paint1(); } });
  slider(f1.controls, { label: 'trees averaged (k)', min: 1, max: 100, step: 1, value: kk, onInput: (v) => { kk = v; paint1(); } });
  button(f1.controls, 'Next resample', () => { ti = (ti + 1) % 100; s1t.set(ti + 1); paint1(); });
  const s1t = { set: (v) => { f1.controls.querySelector('input').value = v; f1.controls.querySelector('.val').textContent = v; } };
  const s1 = stats(f1.root);
  const row1 = el('div', { class: 'fig-row' }); f1.body.append(row1);
  const c1a = el('div'), c1b = el('div'); row1.append(c1a, c1b);
  const mk = (par, title) => { const Fz = frame(par, { width: 470, height: 320, x: [2.2, 10.2], y: [0.6, 5.5], xLabel: 'IFN-γ score', yLabel: 'log2 TMB (mut/Mb)', grid: false }); Fz.g.append('text').attr('x', 8).attr('y', 16).attr('font-weight', 700).attr('font-size', 12.5).attr('fill', ink()).text(title); return Fz; };
  const FA = mk(c1a, 'single tree'), FB = mk(c1b, 'average of k trees');
  const sa = decisionCanvas(FA, () => 0.5, { res: 5, alpha: 0.4 }), sb = decisionCanvas(FB, () => 0.5, { res: 5, alpha: 0.4 });
  const ptsA = FA.layer('pts'), ptsB = FB.layer('pts');
  ptsB.selectAll('circle').data(D).join('circle').attr('cx', (d) => FB.x(d.ifng)).attr('cy', (d) => FB.y(d.log2tmb)).attr('r', 3.6).attr('fill', (d) => cat(d.response)).attr('stroke', css('--surface')).attr('stroke-width', 1)
    .on('mousemove', (e, d) => tip(e, `${d.id}<br>IFN-γ ${d.ifng}, TMB ${d.tmb}<br>forest P = ${f2(forestProb(F2, [d.ifng, d.log2tmb], kk))} · ${d.response ? 'responded' : 'no response'}`)).on('mouseleave', untip);
  const FC = frame(f1.body, { width: 820, height: 230, x: [0, 101], y: [0.5, 0.95], xLabel: 'number of trees', yLabel: 'validation AUC' });
  FC.layer('dots').selectAll('circle').data(tA).join('circle').attr('cx', (v, i) => FC.x(i + 1)).attr('cy', (v) => FC.y(Math.max(0.5, v))).attr('r', 3.4).attr('fill', ink2()).attr('fill-opacity', 0.55)
    .on('mousemove', (e, v) => { tip(e, `single tree · validation AUC ${f3(v)}`); }).on('mouseleave', untip);
  FC.layer('ens').append('path').datum(eA).attr('fill', 'none').attr('stroke', cat(4)).attr('stroke-width', 2.4).attr('d', d3.line().x((v, i) => FC.x(i + 1)).y((v) => FC.y(v)));
  FC.layer('lr').append('line').attr('x1', 0).attr('x2', FC.w).attr('y1', FC.y(lr2)).attr('y2', FC.y(lr2)).attr('stroke', cat(2)).attr('stroke-width', 2).attr('stroke-dasharray', '6 4');
  FC.layer('lr').append('text').attr('x', FC.w - 4).attr('y', FC.y(lr2) - 6).attr('text-anchor', 'end').attr('font-size', 12).attr('fill', ink()).text('logistic regression, same two features');
  FC.layer('lab').append('text').attr('x', 70).attr('y', FC.y(0.6)).attr('font-size', 12).attr('fill', ink()).text('individual trees');
  FC.layer('lab').append('text').attr('x', FC.w - 4).attr('text-anchor', 'end').attr('y', FC.y(eA[99]) - 8).attr('font-size', 12).attr('fill', ink()).text('average of the first k trees');
  const cur1 = FC.layer('cur');
  legend(f1.body, [['single tree', ink2()], ['running average', cat(4)], ['logistic regression', cat(2)]]);
  function paint1() {
    const T = F2.trees[ti];
    sa.draw((a, b) => treeValue(T.root, [a, b])); sb.draw((a, b) => forestProb(F2, [a, b], kk));
    ptsA.selectAll('circle').data(D).join('circle').attr('cx', (d) => FA.x(d.ifng)).attr('cy', (d) => FA.y(d.log2tmb)).attr('r', 3.6).attr('stroke-width', 1.4)
      .attr('fill', (d, i) => (T.inbag[i] ? cat(d.response) : 'none')).attr('stroke', (d, i) => (T.inbag[i] ? css('--surface') : cat(d.response)))
      .on('mousemove', (e, d) => { const i = D.indexOf(d); tip(e, `${d.id}<br>${T.inbag[i] ? `in this resample (${T.inbag[i]}×)` : 'not in this resample (out-of-bag)'}<br>${d.response ? 'responded' : 'no response'}`); }).on('mouseleave', untip);
    cur1.selectAll('*').remove();
    cur1.append('line').attr('x1', FC.x(kk)).attr('x2', FC.x(kk)).attr('y1', 0).attr('y2', FC.h).attr('stroke', muted()).attr('stroke-dasharray', '3 3');
    cur1.append('circle').attr('cx', FC.x(ti + 1)).attr('cy', FC.y(Math.max(0.5, tA[ti]))).attr('r', 6).attr('fill', 'none').attr('stroke', ink()).attr('stroke-width', 2);
    s1.set({ 'tree shown: validation AUC': f3(tA[ti]), [`average of first ${kk} trees`]: f3(eA[kk - 1]), 'mean single tree': f3(mean(tA)), 'logistic regression': f3(lr2) });
  }
  paint1();
  const nOut = F2.trees[0].inbag.filter((v) => !v).length;
  prose(rb, `<p>On these two measurements single trees score ${f3(d3.min(tA))} to ${f3(d3.max(tA))} on the validation centre (mean ${f3(mean(tA))}); the average of 100 scores <b>${f3(eA[99])}</b>, against ${f3(lr2)} for logistic regression on the same two measurements. The averaged surface is smooth where single trees draw jagged rectangles. Tree 1 leaves out ${nOut} of ${n} patients (${pc(nOut / n)}); the fraction approaches 1/e ≈ 37% as n grows, because each patient is missed by one draw with probability 1 − 1/n, and by all n draws with probability (1 − 1/n)<sup>n</sup>.</p>
    <p>Averaging reduces variance only to the extent that the trees' errors differ. Trees trained on resamples of the same patients are correlated, so the gain levels off (the line above flattens after a few dozen trees). Step 2 makes the trees less alike.</p>`);

  // ---- Step 2: random feature subsets + OOB
  const MT = d3.range(1, 11);
  const treeCorr = (F, X) => { const m = F.trees.slice(0, 30).map((t) => X.map((r) => treeValue(t.root, r))); let s = 0, c = 0; for (let i = 0; i < m.length; i++) for (let j = i + 1; j < m.length; j++) { s += corr(m[i], m[j]); c++; } return s / c; };
  const MR = {}; MT.forEach((m) => {
    const F = getRF(K10, RFo(m)), oobS = forestStaged(F, X10, true), valS = forestStaged(F, X10v);
    const last = oobS[99], ok = last.map((v, i) => i).filter((i) => !Number.isNaN(last[i]));
    MR[m] = { F, oobA: oobS.map((s) => aucDefined(s, y)), valA: valS.map((s) => auc(s, yv)), corr: treeCorr(F, X10v), oobErr: ok.filter((i) => (last[i] >= 0.5 ? 1 : 0) !== y[i]).length / ok.length };
  });
  rb = rung(ch, 2, 'Random feature subsets and out-of-bag error',
    `<p>If one measurement dominates, every tree starts with the same split and the trees stay alike. A random forest therefore lets each split consider only <b>mtry</b> randomly chosen features of the ${K10.length} available (drawn afresh at every node). Weaker features then get used, the trees decorrelate, and the average benefits. A very small mtry makes each tree individually weaker, so intermediate values usually work best.</p>
     <p>The <b>out-of-bag (OOB)</b> estimate gives a held-out score without a separate test set. Each patient is absent from about 37% of the trees; averaging only those trees gives a prediction for that patient that the patient did not influence. The OOB curve below uses discovery patients only, so it can be compared with the validation curve from the independent centre.</p>`);
  const f2g = figure(rb, 'mtry, number of trees and OOB error', 'Left: AUC against the number of trees for the chosen mtry. OOB: discovery patients scored only by trees that did not see them. Validation: Centre C. Right: mean correlation between the predictions of two trees on the validation patients (30 trees, all pairs); click a point to choose its mtry.');
  let mm = 3;
  const sM = slider(f2g.controls, { label: 'mtry (features per split)', min: 1, max: 10, step: 1, value: mm, onInput: (v) => { mm = v; paint2(); } });
  const s2 = stats(f2g.root);
  const row2 = el('div', { class: 'fig-row' }); f2g.body.append(row2);
  const c2a = el('div'), c2b = el('div'); row2.append(c2a, c2b);
  const F2a = frame(c2a, { width: 500, height: 310, x: [1, 100], y: [0.6, 0.92], xLabel: 'number of trees', yLabel: 'AUC' });
  const F2b = frame(c2b, { width: 380, height: 310, x: [1, 10], y: [0, 1], xLabel: 'mtry', yLabel: 'mean correlation between trees', xTicks: 9 });
  F2a.layer('ref').append('line').attr('x1', 0).attr('x2', F2a.w).attr('y1', F2a.y(lrAll)).attr('y2', F2a.y(lrAll)).attr('stroke', cat(2)).attr('stroke-dasharray', '6 4').attr('stroke-width', 2);
  F2a.layer('ref').append('text').attr('x', F2a.w - 4).attr('y', F2a.y(lrAll) - 6).attr('text-anchor', 'end').attr('font-size', 12).attr('fill', ink()).text('logistic regression, validation');
  const curA = F2a.layer('curves'), curB = F2b.layer('pts');
  F2b.layer('line').append('path').datum(MT).attr('fill', 'none').attr('stroke', muted()).attr('stroke-width', 1.6).attr('d', d3.line().x((m) => F2b.x(m)).y((m) => F2b.y(MR[m].corr)));
  F2b.layer('lab').append('text').attr('x', F2b.w - 4).attr('y', 14).attr('text-anchor', 'end').attr('font-size', 12).attr('fill', ink()).text('higher mtry: trees more alike');
  F2b.layer('lab').append('text').attr('x', 6).attr('y', F2b.h - 8).attr('font-size', 12).attr('fill', ink()).text('lower mtry: more diverse trees');
  legend(f2g.body, [['OOB AUC (discovery)', cat(3)], ['validation AUC (Centre C)', cat(6)], ['logistic regression, validation', cat(2)]]);
  function paint2() {
    const R = MR[mm], mkLine = (arr) => arr.map((v, i) => [i + 1, v]).filter((q) => !Number.isNaN(q[1]) && q[0] >= 3);
    curA.selectAll('path').data([{ a: R.oobA, c: cat(3) }, { a: R.valA, c: cat(6) }]).join('path').attr('fill', 'none').attr('stroke', (d) => d.c).attr('stroke-width', 2.2).attr('d', (d) => d3.line().x((q) => F2a.x(q[0])).y((q) => F2a.y(Math.max(0.6, q[1])))(mkLine(d.a)));
    curA.selectAll('text').data([[R.oobA[99], 'OOB'], [R.valA[99], 'validation']]).join('text').attr('x', F2a.w - 4).attr('y', (d) => F2a.y(d[0]) + (d[1] === 'OOB' ? 16 : -6)).attr('text-anchor', 'end').attr('font-size', 12).attr('fill', ink()).text((d) => `${d[1]} ${f3(d[0])}`);
    curB.selectAll('circle').data(MT).join('circle').attr('cx', (m) => F2b.x(m)).attr('cy', (m) => F2b.y(MR[m].corr)).attr('r', (m) => (m === mm ? 8 : 5.5)).attr('fill', (m) => (m === mm ? cat(6) : css('--surface'))).attr('stroke', cat(6)).attr('stroke-width', 2).style('cursor', 'pointer')
      .on('mousemove', (e, m) => tip(e, `mtry ${m}<br>tree-to-tree correlation ${f2(MR[m].corr)}<br>OOB AUC ${f3(MR[m].oobA[99])} · validation AUC ${f3(MR[m].valA[99])}`)).on('mouseleave', untip)
      .on('click', (e, m) => { mm = m; sM.set(m); paint2(); });
    s2.set({ 'OOB AUC (100 trees)': f3(R.oobA[99]), 'OOB error rate': pc(R.oobErr), 'validation AUC': f3(R.valA[99]), 'tree-to-tree correlation': f2(R.corr) });
  }
  paint2();
  const bestOob = MT.reduce((a, m) => (MR[m].oobA[99] > MR[a].oobA[99] ? m : a), 1), bestVal = MT.reduce((a, m) => (MR[m].valA[99] > MR[a].valA[99] ? m : a), 1);
  prose(rb, `<p>The correlation between two trees falls from ${f2(MR[10].corr)} at mtry 10 (every split sees every feature) to ${f2(MR[1].corr)} at mtry 1. In this cohort the OOB AUC is highest at mtry ${bestOob} (${f3(MR[bestOob].oobA[99])}) and the validation AUC at mtry ${bestVal} (${f3(MR[bestVal].valA[99])}); the validation AUC ranges only from ${f3(d3.min(MT, (m) => MR[m].valA[99]))} to ${f3(d3.max(MT, (m) => MR[m].valA[99]))} across mtry, a spread comparable to the sampling noise of ${V.length} validation patients, so the exact value matters little here. More trees do not cause overfitting: the curves flatten and stay flat.</p>`);
  const Rw = MR[3], oobPat = D.map((p, i) => ({ p, i, c: Rw.F.trees.filter((t) => !t.inbag[i]).length })).filter((q) => q.p.response && q.c > 30).sort((a, b) => a.p.id.localeCompare(b.p.id))[0];
  const oobTrees = Rw.F.trees.filter((t) => !t.inbag[oobPat.i]), rowO = exRow(oobPat.p);
  worked(rb, `OOB prediction for patient ${oobPat.p.id}`,
    `<p>${oobPat.p.id} was left out of <b>${oobTrees.length} of 100</b> trees (mtry 3). Averaging only those trees gives <b>p = ${f2(mean(oobTrees.map((t) => treeValue(t.root, rowO))))}</b>. Averaging all 100 trees, including ${100 - oobTrees.length} that were trained on ${oobPat.p.id}, gives p = ${f2(forestProb(Rw.F, rowO))}. The patient ${oobPat.p.response ? 'responded' : 'did not respond'}. Repeating this for every patient and scoring the predictions gives the OOB AUC (${f3(Rw.oobA[99])}) and OOB error rate (${pc(Rw.oobErr)}).</p>`);
  quiz(rb, 'The OOB AUC (' + f3(Rw.oobA[99]) + ') is lower than the validation AUC (' + f3(Rw.valA[99]) + '). Does the forest generalise better than OOB suggests?',
    '<p>Not necessarily. OOB is an estimate for patients like the discovery patients (Centres A and B); validation is a different population (Centre C) with its own mix of tumour types. The two numbers answer different questions. Logistic regression shows the same direction (validation above discovery), which points to the cohorts rather than the forest.</p>');

  // ---- Step 3: boosting as residual fitting
  const Xg = rowsOf(D, ['ifng']), Xgv = rowsOf(V, ['ifng']);
  let gk = 2, glr = 0.5, stop3 = null;
  const gModel = () => once('g1' + glr, () => fitBoost(Xg, y, { nTrees: 40, lr: glr, maxDepth: 2, minLeaf: 8, lambda: 1, seed: 1 }));
  const pk = (M, k, x) => sigmoid(boostRaw(M, [x], k));
  rb = rung(ch, 3, 'Boosting: fit the residuals, add a small correction, repeat',
    `<p>Boosting starts from a constant prediction (the base rate, ${pc(base)}). At each round:</p>
     <div class="calc">1. residual of each patient = outcome − current predicted probability   (y − p)
2. fit a small tree (depth 2 here) to those residuals
3. add the tree to the model after multiplying it by the learning rate η
4. recompute p and repeat</div>
     <p>For log-loss, each leaf's value is the sum of the residuals in the leaf divided by the sum of p(1 − p) over its patients (plus a regulariser λ); this is a Newton step, which is what XGBoost and LightGBM use. Patients that are still badly predicted keep large residuals, so later trees concentrate on them. The figure uses IFN-γ alone so the whole model can be drawn as a curve.</p>`);
  const f3g = figure(rb, 'Adding trees one at a time', 'Left: patients (jittered vertically) and the current predicted probability. Right: each patient\'s residual under the current model, and the correction that the next tree would add (in probability units). Press "Add next tree" to accept it.');
  select(f3g.controls, { label: 'learning rate η', value: glr, options: [0.1, 0.3, 0.5, 1].map((v) => [v, String(v)]), onChange: (v) => { glr = +v; gk = Math.min(gk, 39); paint3(); } });
  button(f3g.controls, 'Add next tree', () => { gk = Math.min(39, gk + 1); paint3(); }, true);
  button(f3g.controls, 'Add 5', () => { gk = Math.min(39, gk + 5); paint3(); });
  button(f3g.controls, 'Reset', () => { gk = 0; paint3(); });
  const s3 = stats(f3g.root);
  const row3 = el('div', { class: 'fig-row' }); f3g.body.append(row3);
  const c3a = el('div'), c3b = el('div'); row3.append(c3a, c3b);
  const G3a = frame(c3a, { width: 490, height: 320, x: [2.2, 10.2], y: [-0.08, 1.08], xLabel: 'IFN-γ score', yLabel: 'P(response)' });
  const G3b = frame(c3b, { width: 490, height: 320, x: [2.2, 10.2], y: [-1, 1], xLabel: 'IFN-γ score', yLabel: 'residual (y − p) and next correction' });
  G3a.layer('pts').selectAll('circle').data(D).join('circle').attr('cx', (d) => G3a.x(d.ifng)).attr('cy', (d, i) => G3a.y(d.response + jit(i) * 0.7)).attr('r', 3.4).attr('fill', (d) => cat(d.response)).attr('fill-opacity', 0.65).attr('stroke', css('--surface')).attr('stroke-width', 0.8);
  G3b.layer('z').append('line').attr('x1', 0).attr('x2', G3b.w).attr('y1', G3b.y(0)).attr('y2', G3b.y(0)).attr('stroke', muted()).attr('stroke-dasharray', '4 4');
  G3b.layer('t').append('text').attr('x', 6).attr('y', 14).attr('font-size', 11.5).attr('fill', ink()).text('above 0: model under-predicts');
  G3b.layer('t').append('text').attr('x', 6).attr('y', G3b.h - 6).attr('font-size', 11.5).attr('fill', ink()).text('below 0: model over-predicts');
  const g3gx = d3.range(2.2, 10.21, 0.1), cL = G3a.layer('curve'), rL = G3b.layer('res');
  legend(f3g.body, [['no response', cat(0)], ['responded', cat(1)], ['current model', cat(6)], ['previous model', muted()], ['next tree\'s correction', cat(4)]]);
  function paint3() {
    const M = gModel(), line = (f) => d3.line().x((q) => G3a.x(q)).y((q) => G3a.y(f(q)))(g3gx);
    cL.selectAll('path').data([{ f: (x) => pk(M, Math.max(0, gk - 1), x), c: muted(), d: '5 4', on: gk > 0 }, { f: (x) => pk(M, gk, x), c: cat(6), d: null, on: true }].filter((q) => q.on)).join('path')
      .attr('fill', 'none').attr('stroke', (d) => d.c).attr('stroke-width', 2.6).attr('stroke-dasharray', (d) => d.d).attr('d', (d) => line(d.f));
    rL.selectAll('circle').data(D).join('circle').attr('cx', (d) => G3b.x(d.ifng)).attr('cy', (d) => G3b.y(d.response - pk(M, gk, d.ifng))).attr('r', 3.4).attr('fill', (d) => cat(d.response)).attr('fill-opacity', 0.6).attr('stroke', css('--surface')).attr('stroke-width', 0.8)
      .on('mousemove', (e, d) => tip(e, `${d.id} · IFN-γ ${d.ifng}<br>outcome ${d.response}, current p ${f2(pk(M, gk, d.ifng))}<br>residual ${sg(d.response - pk(M, gk, d.ifng))}`)).on('mouseleave', untip);
    rL.selectAll('path').data(gk < 40 ? [0] : []).join('path').attr('fill', 'none').attr('stroke', cat(4)).attr('stroke-width', 2.8).attr('d', d3.line().x((q) => G3b.x(q)).y((q) => G3b.y(pk(M, gk + 1, q) - pk(M, gk, q)))(g3gx));
    const st = boostStaged(M, Xg), sv = boostStaged(M, Xgv);
    s3.set({ 'trees added': gk, 'training log-loss': f3(logLoss(st[gk].map(sigmoid), y)), 'validation log-loss': f3(logLoss(sv[gk].map(sigmoid), yv)), 'validation AUC': f3(auc(sv[gk], yv)) });
  }
  paint3();
  { const M = gModel(), st = boostStaged(M, Xg), sv = boostStaged(M, Xgv), ll = (S, k) => logLoss(S[k].map(sigmoid), yv), bestK = d3.range(0, 41).reduce((b, k) => (ll(sv, k) < ll(sv, b) ? k : b), 0);
    const nz = M.trees[0], leaves = []; (function w(nd) { if (nd.left) { w(nd.left); w(nd.right); } else leaves.push(nd.v); })(nz);
    prose(rb, `<p>With η = ${glr}, the validation log-loss is ${f3(ll(sv, 0))} for the constant base-rate model, ${f3(ll(sv, 1))} after one tree, ${f3(ll(sv, 10))} after 10 and ${f3(ll(sv, 40))} after 40. The first tree's leaf values (on the log-odds scale, before scaling by η) are ${leaves.map(sg).join(', ')}. Early trees make large corrections. With η = ${glr} the validation log-loss is lowest after ${bestK} trees (${f3(ll(sv, bestK))}) and ${bestK < 40 ? 'then rises: once the main trend is captured, further trees fit noise in the training patients' : 'is still falling at 40 trees'}. A larger η takes bigger steps and reaches its best fit sooner, but overshoots more easily; small rates need many more trees and are usually more accurate (step 4).</p>`); }

  // ---- Step 4: boosting hyperparameters and early stopping
  const sp = stratifiedSplit(y, 0.3, 7), Xtr = sp.train.map((i) => X10[i]), ytr = sp.train.map((i) => y[i]), Xho = sp.test.map((i) => X10[i]), yho = sp.test.map((i) => y[i]);
  const gb4 = (lr, d) => once(`g4${lr}${d}`, () => {
    const B = fitBoost(Xtr, ytr, { nTrees: 200, lr, maxDepth: d, minLeaf: 3, seed: 1 }), L = (X, yy) => boostStaged(B, X).map((s) => ({ ll: logLoss(s.map(sigmoid), yy), a: auc(s, yy) }));
    const tr = L(Xtr, ytr), ho = L(Xho, yho), va = L(X10v, yv), best = ho.reduce((b, v, i) => (v.ll < ho[b].ll ? i : b), 0), bestV = va.reduce((b, v, i) => (v.ll < va[b].ll ? i : b), 0);
    return { tr, ho, va, best, bestV };
  });
  rb = rung(ch, 4, 'Boosting hyperparameters and early stopping',
    `<p>Unlike a forest, a boosted model <b>can overfit as trees are added</b>: training loss keeps falling while the loss on new patients eventually rises. The three settings that matter most interact: the <b>learning rate η</b> (step size), the <b>number of trees</b>, and the <b>tree depth</b> (how complex each correction can be; depth d allows interactions among up to d features). <b>Early stopping</b> picks the number of trees from a held-out set: here 30% of the discovery patients (${Xho.length}), so the validation centre stays untouched.</p>`);
  const f4g = figure(rb, 'Boosting on all ten measurements: training, hold-out and validation loss', 'Log-loss against the number of trees (lower is better). The model is fitted on 70% of the discovery patients. The open diamond marks the hold-out minimum, which is the early-stopping point; the dashed vertical line is the number of trees chosen with the slider below.');
  let lr4 = 0.1, d4 = 2, k4 = 200;
  select(f4g.controls, { label: 'learning rate η', value: lr4, options: [0.02, 0.05, 0.1, 0.2, 0.5].map((v) => [v, String(v)]), onChange: (v) => { lr4 = +v; paint4(); } });
  slider(f4g.controls, { label: 'tree depth', min: 1, max: 3, step: 1, value: d4, onInput: (v) => { d4 = v; paint4(); } });
  slider(f4g.controls, { label: 'trees used', min: 0, max: 200, step: 1, value: k4, onInput: (v) => { k4 = v; paint4(); } });
  const s4 = stats(f4g.root);
  const F4 = frame(f4g.body, { width: 820, height: 330, x: [0, 200], y: [0, 0.9], xLabel: 'number of trees', yLabel: 'log-loss' });
  const c4 = F4.layer('c');
  legend(f4g.body, [['training (70% of discovery)', cat(2)], ['hold-out (30% of discovery)', cat(3)], ['validation (Centre C)', cat(6)]]);
  function paint4() {
    const R = gb4(lr4, d4), ser = [['tr', cat(2), 'training'], ['ho', cat(3), 'hold-out'], ['va', cat(6), 'validation']];
    c4.selectAll('*').remove();
    ser.forEach(([k, c]) => c4.append('path').datum(R[k]).attr('fill', 'none').attr('stroke', c).attr('stroke-width', 2.2).attr('d', d3.line().x((v, i) => F4.x(i)).y((v) => F4.y(Math.min(0.9, v.ll)))));
    ser.forEach(([k, c, t]) => c4.append('text').attr('x', F4.w - 4).attr('y', Math.max(12, Math.min(F4.h - 4, F4.y(Math.min(0.9, R[k][200].ll)) + (k === 'tr' ? 14 : k === 'ho' ? -8 : 14)))).attr('text-anchor', 'end').attr('font-size', 12).attr('fill', ink()).text(t));
    c4.append('path').attr('d', d3.symbol(d3.symbolDiamond, 110)).attr('transform', `translate(${F4.x(R.best)},${F4.y(R.ho[R.best].ll)})`).attr('fill', 'none').attr('stroke', ink()).attr('stroke-width', 2);
    c4.append('text').attr('x', F4.x(R.best) + (R.best > 150 ? -12 : 12)).attr('y', F4.y(R.ho[R.best].ll) - 14).attr('text-anchor', R.best > 150 ? 'end' : 'start').attr('font-size', 12).attr('fill', ink()).text(`early stop: ${R.best} trees`);
    c4.append('line').attr('x1', F4.x(k4)).attr('x2', F4.x(k4)).attr('y1', 0).attr('y2', F4.h).attr('stroke', muted()).attr('stroke-dasharray', '4 4');
    s4.set({ 'hold-out minimum at': `${R.best} trees`, 'validation AUC there': f3(R.va[R.best].a), 'validation AUC at 200 trees': f3(R.va[200].a), [`validation AUC at ${k4} trees`]: f3(R.va[k4].a), 'training loss at 200 trees': f3(R.tr[200].ll) });
  }
  paint4();
  const A4 = (lr, d) => gb4(lr, d);
  prose(rb, `<p>With η = 0.1 and depth 2, the hold-out loss is lowest at ${A4(0.1, 2).best} trees. With η = 0.5 and depth 3 it is lowest at ${A4(0.5, 3).best} trees, and by 200 trees the validation loss has risen from ${f3(A4(0.5, 3).va[A4(0.5, 3).best].ll)} to ${f3(A4(0.5, 3).va[200].ll)}, while the training loss has fallen to ${f3(A4(0.5, 3).tr[200].ll)}: the model is memorising the training patients. With η = 0.02 and depth 2 the hold-out minimum comes much later, at ${A4(0.02, 2).best} trees, against ${A4(0.1, 2).best} for η = 0.1: a smaller rate needs more trees for the same fit.</p>
    <p>Settings that were not varied here: <b>minimum samples per leaf</b> and <b>L2 penalty λ</b> (both shrink leaf values), <b>row subsampling</b> and <b>column subsampling</b> (stochastic boosting: each tree sees a random part of the data, which adds a forest-like variance reduction), and <b>minimum gain</b> to split. The hold-out minimum is itself noisy on ${Xho.length} patients; in practice it is chosen by cross-validation inside the training data.</p>`);

  // ---- Step 5: feature importance
  const impFor = (model, drop) => once(`imp${model}${drop}`, () => {
    const keys = K11.filter((k) => !(drop && k === 'ifng')), Xt = rowsOf(D, keys), Xv = rowsOf(V, keys);
    let imp, score;
    if (model === 'rf') { const F = getRF(keys, { nTrees: 100, maxDepth: 6, minLeaf: 3, mtry: 3, seed: 9 }); imp = F.importance; score = rfScore(F); }
    else { const B = getGB(keys, { nTrees: 100, lr: 0.1, maxDepth: 2, minLeaf: 3, seed: 9 }); const g = new Array(keys.length).fill(0); B.trees.forEach((r) => (function w(nd) { if (nd.left) { g[nd.f] += nd.gain; w(nd.left); w(nd.right); } })(r)); const t = sum(g) || 1; imp = g.map((v) => v / t); score = (M) => M.map((r) => boostRaw(B, r)); }
    return { keys, imp, pt: permImportance(score, Xt, y, { reps: 4 }).map((v) => v.drop), pv: permImportance(score, Xv, yv, { reps: 4 }).map((v) => v.drop), aucV: auc(score(Xv), yv), aucT: auc(score(Xt), y) };
  });
  rb = rung(ch, 5, 'Feature importance: impurity, permutation and SHAP',
    `<p>A tree ensemble has no coefficients, so importance is summarised in other ways:</p>
     <ul><li><b>Impurity importance</b> (forests: mean decrease in impurity; boosting: total gain): the improvement in the split criterion contributed by each feature, summed over all splits and trees. It is free to compute but is calculated on the training data, and it favours features with many possible split points.</li>
     <li><b>Permutation importance</b>: shuffle one feature's values and record how much the AUC falls. It measures what the fitted model uses for prediction, and can be computed on held-out data. Correlated features share the credit.</li>
     <li><b>SHAP</b> values: each feature's share of a single patient's prediction, from Shapley's rule for dividing a total fairly among contributors. Averaged over patients they give a global ranking.</li></ul>
     <p>The figure adds a column of random numbers to the ten measurements as a control: a feature with no relation to anything should receive no importance.</p>`);
  const f5g = figure(rb, 'Three importance measures on the same model', 'Each bar is a share of that measure\'s total (negative permutation drops are shown as zero; hover for the raw value). Impurity importance and permutation on the training patients use the data the model was fitted on; permutation on validation uses Centre C. The right-hand label gives the planted truth.');
  let m5 = 'rf', drop5 = false;
  select(f5g.controls, { label: 'Model', value: m5, options: [['rf', 'Random forest'], ['gb', 'Gradient boosting']], onChange: (v) => { m5 = v; paint5(); } });
  checkbox(f5g.controls, { label: 'Leave out IFN-γ', checked: false, onChange: (c) => { drop5 = c; paint5(); } });
  const s5 = stats(f5g.root);
  const F5 = frame(f5g.body, { width: 820, height: 470, x: [0, 1], y: [0, 1], xLabel: 'share of total importance', grid: false, axes: false, margin: { left: 150, right: 150, top: 8, bottom: 44 } });
  const ax5 = F5.g.append('g').attr('class', 'axis x').attr('transform', `translate(0,${F5.h})`), bars5 = F5.layer('bars');
  const MS = [['imp', 'impurity (training)', cat(3)], ['pt', 'permutation, training patients', cat(5)], ['pv', 'permutation, validation patients', cat(7)]];
  legend(f5g.body, MS.map((m) => [m[1], m[2]]));
  function paint5() {
    const R = impFor(m5, drop5), keys = R.keys, share = (arr) => { const t = sum(arr.map((v) => Math.max(0, v))) || 1; return arr.map((v) => Math.max(0, v) / t); };
    const S = { imp: share(R.imp), pt: share(R.pt), pv: share(R.pv) }, raw = { imp: R.imp, pt: R.pt, pv: R.pv };
    const xs = d3.scaleLinear().domain([0, Math.max(0.3, d3.max(MS, (m) => d3.max(S[m[0]])) * 1.08)]).range([0, F5.w]);
    ax5.call(d3.axisBottom(xs).ticks(6).tickFormat(d3.format('.0%')));
    const yb = d3.scaleBand().domain(keys).range([0, F5.h]).padding(0.22), ys = d3.scaleBand().domain(MS.map((m) => m[0])).range([0, yb.bandwidth()]).padding(0.12);
    bars5.selectAll('*').remove();
    keys.forEach((k, j) => {
      bars5.append('text').attr('x', -8).attr('y', yb(k) + yb.bandwidth() / 2 + 4).attr('text-anchor', 'end').attr('font-size', 12).attr('fill', ink2()).text(FE[k].label);
      bars5.append('text').attr('x', F5.w + 8).attr('y', yb(k) + yb.bandwidth() / 2 + 4).attr('font-size', 11).attr('fill', muted()).text(ROLE[k]);
      MS.forEach(([m, lab, c]) => bars5.append('rect').attr('x', 0).attr('y', yb(k) + ys(m)).attr('height', ys.bandwidth()).attr('width', Math.max(1, xs(S[m][j]))).attr('fill', c)
        .on('mousemove', (e) => tip(e, `${FE[k].label}<br>${lab}<br>share ${pc(S[m][j])} · raw ${m === 'imp' ? f3(raw[m][j]) : 'AUC drop ' + f3(raw[m][j])}`)).on('mouseleave', untip));
    });
    const top = (arr) => keys.map((k, j) => [k, arr[j]]).sort((a, b) => b[1] - a[1]).slice(0, 3).map((q) => FE[q[0]].label).join(', ');
    const jn = keys.indexOf('noise'), js = keys.indexOf('smoker');
    s5.set({ 'validation AUC': f3(R.aucV), 'top 3 by impurity': top(R.imp), 'top 3 by permutation (validation)': top(R.pv), 'noise column: impurity / perm. (train)': `${pc(S.imp[jn])} / ${pc(S.pt[jn])}`, 'ever-smoker: impurity': pc(S.imp[js]) });
  }
  paint5();
  const R0 = impFor('rf', false), R1 = impFor('rf', true), i0 = K11.indexOf('noise'), sh0 = (R, k) => { const j = R.keys.indexOf(k); const t = sum(R.pv.map((v) => Math.max(0, v))); return Math.max(0, R.pv[j]) / t; };
  const impSh = (R, k) => R.imp[R.keys.indexOf(k)];
  prose(rb, `<p>Reading the figure for the forest: IFN-γ leads on every measure. The control column receives ${pc(impSh(R0, 'noise'))} of the impurity importance, more than sex (${pc(impSh(R0, 'male'))}) or smoking status (${pc(impSh(R0, 'smoker'))}), because a continuous variable offers many more candidate splits than a yes/no variable and some of them fit the training patients by chance. Its permutation importance on the validation patients is ${f3(R0.pv[i0]).replace('-', '−')} AUC, i.e. the model does not rely on it. Permutation importance on the <i>training</i> patients is inflated for features the forest used to memorise individual patients (noise: ${f3(R0.pt[i0]).replace('-', '−')}), so held-out data are preferred.</p>
    <p>Importance describes the fitted model and not biology. Tick "Leave out IFN-γ": PD-L1 and CD8, which are driven by the same immune activity and have no direct effect, take over its role (PD-L1 permutation share ${pc(sh0(R0, 'pdl1'))} → ${pc(sh0(R1, 'pdl1'))}, CD8 ${pc(sh0(R0, 'cd8'))} → ${pc(sh0(R1, 'cd8'))}), and the validation AUC moves from ${f3(R0.aucV)} to ${f3(R1.aucV)}.</p>`);
  // SHAP worked example on three features (exact Shapley values: 2^3 coalitions)
  const F3m = getRF(K3, { nTrees: 100, maxDepth: 5, minLeaf: 3, mtry: 3, seed: 4 }), bg = D.filter((p, i) => i % 3 === 0).map((p) => K3.map((k) => FE[k].get(p)));
  const shP = D.filter((p) => p.response).sort((a, b) => Math.abs(forestProb(F3m, K3.map((k) => FE[k].get(a))) - 0.75) - Math.abs(forestProb(F3m, K3.map((k) => FE[k].get(b))) - 0.75))[0];
  const xs3 = K3.map((k) => FE[k].get(shP)), sh = exactShapley((r) => forestProb(F3m, r), xs3, bg);
  worked(rb, `SHAP values for patient ${shP.id} (forest on the three true drivers)`,
    `<p>The forest's average prediction over ${bg.length} background patients is <b>${f2(sh.base)}</b>. For ${shP.id} the forest predicts <b>${f2(sh.full)}</b>. Shapley values share the difference of ${sg(sh.full - sh.base)} among the three measurements, by averaging each one's added effect over every order in which they could be revealed:</p>
     <div class="calc">${K3.map((k, j) => `${FE[k].label.padEnd(20)} value ${String(f2(xs3[j])).padStart(5)}   φ = ${sg(sh.phi[j])}`).join('\n')}\n${'sum of φ'.padEnd(20)}                  = ${sg(sum(sh.phi))}   (base ${f2(sh.base)} + sum = ${f2(sh.base + sum(sh.phi))})</div>
     <p>The contributions add up exactly to the prediction. For tree models, TreeSHAP computes these values efficiently for hundreds of features; this example enumerates all 2³ coalitions. As with other importance measures, SHAP explains the model, not the mechanism.</p>`);

  // ---- Step 6: head-to-head lab
  rb = rung(ch, 6, 'Lab: forest, boosting and logistic regression on the same patients',
    `<p>All three models are fitted on the ${n} discovery patients and scored on the ${V.length} Centre C patients, using the same features. The response in this simulation is generated from a logistic model of three measurements, which favours logistic regression; real data may differ, and the purpose is to show how to check. The right panel gives validation AUC with a 95% bootstrap interval from resampling the validation patients, and training AUC (open diamond) for the same model.</p>`);
  const f6 = figure(rb, 'Model comparison lab', 'Left: ROC curves on the validation patients (the diagonal is chance). Right: validation AUC with 95% interval (filled) and training AUC (open diamond). Presets set several controls at once. Forest and boosting each refit only when their own settings change.');
  const cfg = { keys: 'ten', rfTrees: 100, rfDepth: 6, rfMtry: 3, gbTrees: 100, gbLr: 0.1, gbDepth: 2 }, ctl = {};
  const pre6 = el('div', { class: 'controls' }); f6.controls.before(pre6);
  pre6.append(el('span', { class: 'fig-caption', style: 'margin:0' }, 'Presets:'));
  const PRE = [['Defaults', {}], ['Three true features', { keys: 'three' }], ['Forest: deep trees', { rfDepth: 12 }], ['Forest: stumps', { rfDepth: 1 }], ['Boosting: aggressive', { gbLr: 0.5, gbDepth: 3, gbTrees: 200 }]];
  const setCfg = (o) => { Object.assign(cfg, { keys: 'ten', rfTrees: 100, rfDepth: 6, rfMtry: 3, gbTrees: 100, gbLr: 0.1, gbDepth: 2 }, o); sync6(); paint6(); };
  PRE.forEach(([lab, o]) => button(pre6, lab, () => setCfg(o)));
  select(f6.controls, { label: 'Features', value: cfg.keys, options: [['ten', 'all ten measurements'], ['three', 'the three true drivers']], onChange: (v) => { cfg.keys = v; paint6(); } });
  const mkS = (key, label, min, max, step, fmt) => { ctl[key] = slider(f6.controls, { label, min, max, step, value: cfg[key], fmt, onInput: (v) => { cfg[key] = v; paint6(); } }); };
  mkS('rfTrees', 'forest: trees', 10, 150, 10); mkS('rfDepth', 'forest: depth', 1, 12, 1); mkS('rfMtry', 'forest: mtry', 1, 10, 1);
  mkS('gbTrees', 'boosting: trees', 10, 200, 10); mkS('gbLr', 'boosting: η', 0.02, 0.5, 0.02, (v) => v.toFixed(2)); mkS('gbDepth', 'boosting: depth', 1, 3, 1);
  const sync6 = () => { Object.entries(ctl).forEach(([k, s]) => s.set(cfg[k])); f6.controls.querySelector('select').value = cfg.keys; };
  const s6 = stats(f6.root);
  const row6 = el('div', { class: 'fig-row' }); f6.body.append(row6);
  const c6a = el('div'), c6b = el('div'); row6.append(c6a, c6b);
  const R6 = frame(c6a, { width: 430, height: 340, x: [0, 1], y: [0, 1], xLabel: 'false-positive rate (1 − specificity)', yLabel: 'true-positive rate (sensitivity)' });
  R6.layer('diag').append('line').attr('x1', 0).attr('y1', R6.h).attr('x2', R6.w).attr('y2', 0).attr('stroke', muted()).attr('stroke-dasharray', '4 4');
  const A6 = frame(c6b, { width: 450, height: 340, x: [0.5, 1], y: [0, 1], xLabel: 'AUC', grid: false, axes: false, margin: { left: 150, right: 20, top: 8, bottom: 44 } });
  A6.g.append('g').attr('class', 'axis x').attr('transform', `translate(0,${A6.h})`).call(d3.axisBottom(A6.x).ticks(6));
  const MOD = [['lr', 'Logistic regression', cat(2)], ['rf', 'Random forest', cat(4)], ['gb', 'Gradient boosting', cat(6)]];
  legend(f6.body, MOD.map((m) => [m[1], m[2]]));
  const bootCI = (E, B = 200) => { const r = rng(5), m = yv.length, res = { lr: [], rf: [], gb: [], drf: [], dgb: [] }; for (let b = 0; b < B; b++) { const idx = Array.from({ length: m }, () => r.int(m)), yy = idx.map((i) => yv[i]); if (!yy.some(Boolean) || yy.every(Boolean)) continue; const a = {}; ['lr', 'rf', 'gb'].forEach((k) => { a[k] = auc(idx.map((i) => E[k].sv[i]), yy); res[k].push(a[k]); }); res.drf.push(a.rf - a.lr); res.dgb.push(a.gb - a.lr); } const q = (arr) => [d3.quantile(arr.slice().sort(d3.ascending), 0.025), d3.quantile(arr.slice().sort(d3.ascending), 0.975)]; return Object.fromEntries(Object.entries(res).map(([k, v]) => [k, q(v)])); };
  function paint6() {
    const keys = cfg.keys === 'three' ? K3 : K10;
    const E = evalCfg({ ...cfg, keys }), CI = bootCI(E);
    R6.layer('roc').selectAll('*').remove(); A6.layer('a').selectAll('*').remove();
    MOD.forEach(([k, lab, c]) => R6.layer('roc').append('path').datum(roc(E[k].sv, yv)).attr('fill', 'none').attr('stroke', c).attr('stroke-width', 2.2).attr('d', d3.line().x((q) => R6.x(q.fpr)).y((q) => R6.y(q.tpr))));
    const yb = d3.scaleBand().domain(MOD.map((m) => m[0])).range([0, A6.h]).padding(0.45), g = A6.layer('a');
    MOD.forEach(([k, lab, c]) => {
      const cy = yb(k) + yb.bandwidth() / 2;
      g.append('text').attr('x', -8).attr('y', cy + 4).attr('text-anchor', 'end').attr('font-size', 12).attr('fill', ink2()).text(lab);
      g.append('line').attr('x1', A6.x(Math.max(0.5, CI[k][0]))).attr('x2', A6.x(Math.min(1, CI[k][1]))).attr('y1', cy).attr('y2', cy).attr('stroke', ink2()).attr('stroke-width', 2);
      g.append('circle').attr('cx', A6.x(E[k].va)).attr('cy', cy).attr('r', 6).attr('fill', c).attr('stroke', css('--surface')).attr('stroke-width', 1.5)
        .on('mousemove', (e) => tip(e, `${lab}<br>validation AUC ${f3(E[k].va)} (95% ${f3(CI[k][0])}–${f3(CI[k][1])})<br>training AUC ${f3(E[k].tr)}`)).on('mouseleave', untip);
      g.append('path').attr('d', d3.symbol(d3.symbolDiamond, 80)).attr('transform', `translate(${A6.x(Math.max(0.5, E[k].tr))},${cy + 0})`).attr('fill', 'none').attr('stroke', c).attr('stroke-width', 2);
      g.append('text').attr('x', A6.x(E[k].va)).attr('y', cy - 12).attr('text-anchor', 'middle').attr('font-size', 11.5).attr('fill', ink()).text(f3(E[k].va));
    });
    const dd = (d, ci) => `${d >= 0 ? '+' : '−'}${f3(Math.abs(d))} (${ci[0] >= 0 ? '+' : '−'}${f3(Math.abs(ci[0]))} to ${ci[1] >= 0 ? '+' : '−'}${f3(Math.abs(ci[1]))})`;
    s6.set({ 'validation AUC: logistic': f3(E.lr.va), 'forest': f3(E.rf.va), 'boosting': f3(E.gb.va), 'forest − logistic (95% CI)': dd(E.rf.va - E.lr.va, CI.drf), 'boosting − logistic (95% CI)': dd(E.gb.va - E.lr.va, CI.dgb), 'training AUC: forest / boosting': `${f3(E.rf.tr)} / ${f3(E.gb.tr)}` });
  }
  paint6();
  const T = (o) => evalCfg({ keys: K10, ...o }), T3 = evalCfg({ keys: K3 }), T0 = T({}), Tdeep = T({ rfDepth: 12 }), Tstump = T({ rfDepth: 1 }), Tagg = T({ gbLr: 0.5, gbDepth: 3, gbTrees: 200 });
  prose(rb, '<p><b>Suggested comparisons</b>:</p>');
  table(rb, [
    { t: '<b>Defaults</b>, all ten measurements', s: `Validation AUC: logistic ${f3(T0.lr.va)}, forest ${f3(T0.rf.va)}, boosting ${f3(T0.gb.va)}`, m: `On this cohort the simple model is ${T0.lr.va >= Math.max(T0.rf.va, T0.gb.va) ? 'at least as good as' : 'close to'} both ensembles. Check the interval on the difference before claiming a winner.` },
    { t: '<b>Three true drivers</b> only', s: `Logistic ${f3(T3.lr.va)}, forest ${f3(T3.rf.va)}, boosting ${f3(T3.gb.va)}`, m: `Forest ${f3(T0.rf.va)} → ${f3(T3.rf.va)} and boosting ${f3(T0.gb.va)} → ${f3(T3.gb.va)} when the proxies and noise are removed. The true relationship is a smooth sum on the log-odds scale, which logistic regression fits by construction, so ${T3.lr.va >= Math.max(T3.rf.va, T3.gb.va) ? 'it stays at least as high' : 'it is not clearly ahead'} here.` },
    { t: 'Forest depth <b>12</b> vs <b>1</b>', s: `Training AUC ${f3(Tdeep.rf.tr)} vs ${f3(Tstump.rf.tr)}; validation ${f3(Tdeep.rf.va)} vs ${f3(Tstump.rf.va)}`, m: 'Deep trees fit the training patients almost perfectly, so training AUC is not a performance estimate for a forest. Stumps cannot capture the threshold structure.' },
    { t: 'Boosting <b>aggressive</b> (η 0.5, depth 3, 200 trees)', s: `Validation AUC ${f3(T0.gb.va)} → ${f3(Tagg.gb.va)}; validation log-loss ${f3(T0.gb.ll)} → ${f3(Tagg.gb.ll)}; training AUC ${f3(Tagg.gb.tr)}`, m: 'Ranking (AUC) can survive overfitting better than probabilities (log-loss) do; the predicted probabilities become over-confident first.' },
    { t: 'Calibration in Centre C', s: `Mean predicted probability in Centre C: logistic ${pc(T0.lr.mp)}, forest ${pc(T0.rf.mp)}, boosting ${pc(T0.gb.mp)}; observed ${pc(mean(yv))}`, m: 'Discrimination (AUC) and calibration are separate properties. Here all three under-predict by a similar amount, which reflects the different case mix of Centre C more than the model type; probabilities should be checked, and recalibrated if needed, before use as risks.' },
  ], [['t', 'Try'], ['s', 'What you see'], ['m', 'What it means']], { cls: 'compact' });

  // =====================================================================================
  // 5. PITFALLS
  // =====================================================================================
  ch = chapter(main, 'pitfalls', 'Pitfalls', 'Common problems, how to recognise them, and how to address them.');
  const fx = getRF(['ifng'], { nTrees: 50, maxDepth: 6, minLeaf: 3, mtry: 1, seed: 2 }), lx = fitLogistic(rowsOf(D, ['ifng']), y);
  const pR = (x) => forestProb(fx, [x]), pL = (x) => sigmoid(lx.b0 + lx.w[0] * x);
  const maxD = d3.max(D, (p) => p.ifng);
  const A5 = Tagg, CI0 = bootCI(T0);
  cards(ch, [
    { tag: 'Symptom: training AUC near 1', title: 'Scoring the forest on its own training data', body: `The forest reaches a training AUC of <b>${f3(T0.rf.tr)}</b> at depth 6 and ${f3(Tdeep.rf.tr)} at depth 12, against ${f3(T0.rf.va)} and ${f3(Tdeep.rf.va)} on the validation centre. The trees have seen those patients. <b>Fix:</b> report OOB, cross-validated or independent-cohort performance.` },
    { tag: 'Symptom: reported test performance is optimistic', title: 'Tuning or early stopping on the test set', body: `Number of trees chosen on the hold-out inside discovery vs on the validation centre itself: ${[[0.1, 2], [0.5, 3], [0.02, 2]].map(([l, d]) => `η = ${l}, depth ${d}: ${A4(l, d).best} vs ${A4(l, d).bestV} trees (validation loss ${f3(A4(l, d).va[A4(l, d).best].ll)} vs ${f3(A4(l, d).va[A4(l, d).bestV].ll)})`).join('; ')}. The second figure in each pair is optimistic, because the test patients chose the setting; the bias grows with the number of settings tried. <b>Fix:</b> tune with an inner split or nested cross-validation, and use the final test set once.` },
    { tag: 'Symptom: a random column receives importance', title: 'Impurity importance favours continuous features', body: `A column of random numbers received ${pc(impSh(R0, 'noise'))} of the forest's impurity importance, more than sex (${pc(impSh(R0, 'male'))}) or smoking status (${pc(impSh(R0, 'smoker'))}), although it has no relation to the outcome. <b>Fix:</b> use permutation importance on held-out data or SHAP, and always include a random-feature control when ranking candidate markers.` },
    { tag: 'Symptom: important features are not causal', title: 'Importance describes the model, not the biology', body: `PD-L1 and CD8 carry no effect of their own, yet when IFN-γ is removed they take over (permutation share ${pc(sh0(R0, 'pdl1'))} → ${pc(sh0(R1, 'pdl1'))} and ${pc(sh0(R0, 'cd8'))} → ${pc(sh0(R1, 'cd8'))}). Correlated features split or exchange credit. <b>Fix:</b> interpret importance as predictive use, group correlated features, and do not read it as a causal ranking.` },
    { tag: 'Symptom: validation loss rises while training loss falls', title: 'Boosting without early stopping', body: `Aggressive boosting (η 0.5, depth 3, 200 trees) has a training AUC of ${f3(A5.gb.tr)} and a validation log-loss of ${f3(A5.gb.ll)}, compared with ${f3(T0.gb.ll)} for the default settings. <b>Fix:</b> lower η, use early stopping, add subsampling or an L2 penalty.` },
    { tag: 'Symptom: flat predictions outside the training range', title: 'Trees cannot extrapolate', body: `A forest fitted on IFN-γ alone (observed up to ${f2(maxD)}) predicts ${f2(pR(maxD))} at the maximum and ${f2(pR(14))} at IFN-γ 14, the same value, whereas logistic regression continues from ${f2(pL(maxD))} to ${f2(pL(14))}. A new cohort or platform with a shifted scale gives the forest no signal beyond the training range. <b>Fix:</b> check that new data fall inside the training range, and normalise across platforms.` },
    { tag: 'Symptom: probabilities that look reasonable but are off', title: 'Uncalibrated probabilities', body: `In Centre C the mean predicted response is ${pc(T0.lr.mp)} for logistic regression, ${pc(T0.rf.mp)} for the forest and ${pc(T0.gb.mp)} for boosting; the observed rate is ${pc(mean(yv))}; the overfitted boosting setting gives ${pc(A5.gb.mp)} with a validation log-loss of ${f3(A5.gb.ll)}. Forest outputs are averages of leaf proportions and are not guaranteed to be calibrated, and boosted probabilities become over-confident as the model overfits. <b>Fix:</b> inspect a calibration plot, and recalibrate (Platt or isotonic) on data not used for fitting.` },
    { tag: 'Symptom: complex model, no gain', title: 'An ensemble chosen without a baseline comparison', body: `With the default settings the validation AUC is ${f3(T0.lr.va)} for logistic regression, ${f3(T0.rf.va)} for the forest and ${f3(T0.gb.va)} for boosting. The 95% bootstrap interval for forest − logistic is ${f3m(CI0.drf[0])} to ${f3m(CI0.drf[1])}, and for boosting − logistic ${f3m(CI0.dgb[0])} to ${f3m(CI0.dgb[1])}; with ${V.length} validation patients, differences of a few hundredths in AUC cannot be distinguished from noise. The simulation is logistic by construction, so the simple model has an advantage that real data may not give. <b>Fix:</b> always fit the regularised logistic baseline first, and keep the ensemble only if it wins by more than the uncertainty.` },
  ]);

  // =====================================================================================
  // 6. RECAP
  // =====================================================================================
  ch = chapter(main, 'recap', 'Recap', 'Key points.');
  recap(ch, [
    'A single deep tree has <b>high variance</b>: its predictions change with the training sample. Ensembles combine many trees to reduce error.',
    'A <b>random forest</b> averages independent trees grown on bootstrap resamples, with a random subset of <b>mtry</b> features at each split. Adding trees does not cause overfitting; it only flattens the curve.',
    'The <b>out-of-bag</b> predictions (each patient scored only by trees that did not see it) give a built-in held-out estimate. It describes the training population, not a new centre.',
    '<b>Gradient boosting</b> adds small trees one at a time, each fitted to the residuals of the model so far and scaled by a learning rate. It <b>can overfit</b>, so choose the number of trees by early stopping on an inner hold-out or by cross-validation.',
    'The main boosting settings are <b>learning rate, number of trees, depth</b>, and row/column subsampling and L2 penalties; the main forest settings are <b>mtry</b> and leaf size.',
    '<b>Impurity importance</b> is biased towards continuous features; <b>permutation importance</b> on held-out data and <b>SHAP</b> describe what the model uses. None of them is a causal ranking, and correlated features share credit.',
    'Trees do not extrapolate and give no effect sizes. Their probabilities may need <b>calibration</b>.',
    'Compare against a <b>regularised logistic regression</b> with an uncertainty interval on the difference. On this cohort the simple model was as good as or better than both ensembles.',
  ]);
  prose(ch, '<h3>In code</h3><p>Both families are available in scikit-learn; XGBoost and LightGBM offer faster, more heavily tuned boosting. Use OOB or cross-validated scores to compare, and permutation importance on held-out data.</p>');
  code(ch, {
    Python: `import numpy as np
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

feats = ["ifng", "log2tmb", "caf", "pdl1", "cd8", "prolif", "age", "smoker", "male", "centreB"]
Xtr, ytr, Xte, yte = train[feats], train["response"], test[feats], test["response"]

rf = RandomForestClassifier(n_estimators=500, max_features="sqrt", min_samples_leaf=3,
                            oob_score=True, random_state=0).fit(Xtr, ytr)
print("OOB accuracy", rf.oob_score_)                       # built-in held-out estimate

gb = HistGradientBoostingClassifier(learning_rate=0.05, max_depth=3, max_iter=1000,
                                    early_stopping=True, validation_fraction=0.2,
                                    n_iter_no_change=20, random_state=0).fit(Xtr, ytr)
lr = make_pipeline(StandardScaler(), LogisticRegression(C=1.0)).fit(Xtr, ytr)

for name, m in [("forest", rf), ("boosting", gb), ("logistic", lr)]:
    print(name, roc_auc_score(yte, m.predict_proba(Xte)[:, 1]))

# Permutation importance on held-out data (AUC drop), 20 repeats
imp = permutation_importance(rf, Xte, yte, scoring="roc_auc", n_repeats=20, random_state=0)
print(sorted(zip(imp.importances_mean.round(3), feats), reverse=True))
# SHAP for boosted trees: import shap; shap.TreeExplainer(xgb_model).shap_values(Xte)`,
    R: `library(ranger); library(xgboost); library(pROC)
feats <- c("ifng","log2tmb","caf","pdl1","cd8","prolif","age","smoker","male","centreB")

rf <- ranger(response ~ ., data = train[, c(feats, "response")], num.trees = 500,
             probability = TRUE, importance = "permutation")   # OOB-based permutation importance
p_rf <- predict(rf, test)$predictions[, "1"]                   # response coded as a factor
rf$prediction.error                                            # OOB Brier score
sort(rf$variable.importance, decreasing = TRUE)

dtrain <- xgb.DMatrix(as.matrix(train[, feats]), label = train$response)
cv <- xgb.cv(params = list(objective = "binary:logistic", eta = 0.05, max_depth = 3,
                           subsample = 0.8, colsample_bytree = 0.8, eval_metric = "auc"),
             data = dtrain, nrounds = 1000, nfold = 5, early_stopping_rounds = 20, verbose = 0)
gb <- xgb.train(params = list(objective = "binary:logistic", eta = 0.05, max_depth = 3),
                data = dtrain, nrounds = cv$best_iteration)
p_gb <- predict(gb, as.matrix(test[, feats]))

lr <- glm(response ~ ., family = binomial, data = train[, c(feats, "response")])
sapply(list(rf = p_rf, gb = p_gb, lr = predict(lr, test, type = "response")),
       function(p) auc(test$response, p))`,
  });
  seeAlso(main, [['concepts', 'decision-tree', 'Decision trees'], ['concepts', 'logistic-regression', 'Logistic regression'], ['concepts', 'cross-validation', 'Cross-validation and data leakage'], ['concepts', 'metrics', 'Evaluating a classifier'], ['concepts', 'regularization', 'Regularisation'], ['omics', 'bulk-classifier', 'Biomarker classifier and nested CV']]);
}

// ---------------------------------------------------------------------------------------------
/** The "two recipes" diagram for one patient: forest (average of independent trees) and boosting (sum of corrections). */
function ensembleSvg(P, row, F, B) {
  const wrap = el('div', { class: 'figure' });
  wrap.append(el('div', { class: 'fig-title' }, `Two ways to combine trees, for patient ${P.id}`));
  const svg = d3.select(wrap).append('svg').attr('viewBox', '0 0 980 372').attr('width', 980).attr('role', 'img').attr('aria-label', 'a random forest averages independent trees; boosting sums sequential corrections');
  const pF = forestProb(F, row), zB = boostRaw(B, row), nVote = F.trees.filter((t) => treeValue(t.root, row) > 0.5).length;
  const rowsDef = [
    { y: 6, title: 'Random forest: independent trees on bootstrap resamples, then an average', items: [...F.trees.slice(0, 5).map((t, i) => ({ w: 124, t: `Tree ${i + 1}`, l: [`resample ${i + 1}`, `${t.inbag.filter((v) => v > 0).length} distinct`, `p = ${f2(treeValue(t.root, row))}`] })), { w: 30, dots: true }, { w: 210, t: 'Average of 100 trees', strong: true, l: [`p = ${f2(pF)}`, `${nVote}/100 trees: p > 0.5`] }] },
    { y: 180, title: 'Gradient boosting: trees added one at a time, each correcting the sum so far', items: [{ w: 124, t: 'Start', l: ['base rate', `log-odds ${sg(B.f0)}`, `p = ${f2(sigmoid(B.f0))}`] }, ...B.trees.slice(0, 4).map((t, i) => ({ w: 124, t: `+ Tree ${i + 1}`, l: [`leaf ${sg(treeValue(t, row))}`, `× η ${B.lr}`, `= ${sg(B.lr * treeValue(t, row))}`], plus: true })), { w: 30, dots: true }, { w: 210, t: 'Sum of 100 trees', strong: true, l: [`log-odds ${sg(zB)}`, `p = σ(${f2(zB)}) = ${f2(sigmoid(zB))}`] }] },
  ];
  rowsDef.forEach((R) => {
    svg.append('text').attr('x', 10).attr('y', R.y + 14).attr('font-weight', 700).attr('font-size', 13.5).attr('fill', ink()).text(R.title);
    let x = 10;
    R.items.forEach((it, i) => {
      const g = svg.append('g').attr('transform', `translate(${x},${R.y + 28})`);
      if (it.dots) g.append('text').attr('x', it.w / 2).attr('y', 62).attr('text-anchor', 'middle').attr('font-size', 22).attr('fill', ink2()).text('…');
      else {
        g.append('rect').attr('width', it.w).attr('height', 112).attr('rx', 10).attr('fill', it.strong ? css('--accent-soft') : css('--surface-2')).attr('stroke', css('--axis'));
        g.append('text').attr('x', 12).attr('y', 24).attr('font-weight', 700).attr('font-size', 13).attr('fill', ink()).text(it.t);
        it.l.forEach((s, j) => g.append('text').attr('x', 12).attr('y', 50 + j * 24).attr('font-family', css('--mono')).attr('font-size', 12).attr('fill', ink2()).text(s));
      }
      if (i < R.items.length - 1) svg.append('text').attr('x', x + it.w + 8).attr('y', R.y + 28 + 62).attr('text-anchor', 'middle').attr('font-size', 18).attr('fill', ink2()).text(R.items[i + 1].plus ? '+' : i === R.items.length - 2 ? '→' : '');
      x += it.w + 16;
    });
  });
  svg.append('text').attr('x', 10).attr('y', 362).attr('font-size', 12.5).attr('fill', muted()).text(`Observed outcome: ${P.response ? 'responded' : 'no response'}. Forest: depth 6, mtry 3. Boosting: depth 2, η ${B.lr}. Both use ten measurements.`);
  return wrap;
}
