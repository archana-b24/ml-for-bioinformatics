// Biomarker classifier and nested CV (lesson). Bulk RNA-seq step 3 of 4.
// Arc: what → why → where → how it works (6 steps: the pipeline and CV folds → lasso / elastic-net paths and
//      stability → the leakage trap (real vs permuted labels) → model comparison (prespecified IFN-γ score vs lasso
//      vs forest vs boosting) → locking and external validation in Centre C → classifier lab, run live, with
//      try/see/means table) → pitfalls → recap.
// Data: bulk_fs / bulk_classifier / bulk_compare / bulk_explain / bulk_validation.json (scripts/build_bulk.py and
// build_bulk_extra.py, sklearn), bulk_de_lab.json (discovery log-CPM, 177 × 1,933) and bulk_clf_lab.json
// (Centre C log-CPM, 120 × 1,933; scripts/build_classifier_lesson.py). The lab runs gene selection, scaling and
// ridge-logistic fits inside 5-fold CV in the browser. Every number in the prose is computed at render time.
import { figure, header, slider, select, button, stats, legend, seeAlso, every, animate, tip, untip, el, html } from '../lib/ui.js';
import { frame, cat, css, ink, ink2, muted } from '../lib/plot.js';
import { rng, solve, sigmoid, mean } from '../lib/data.js';
import { auc, roc } from '../lib/models.js';
import { lessonNav, chapter, prose, rung, worked, quiz, cards, table, code, recap } from '../lib/lesson.js';
import { stepCard, omicsData } from './_shared.js';
import { GROUP_ORDER, groupColor, groupLabel } from './_bulk_common.js';
const d3 = window.d3;

const f2 = (v) => (Math.abs(v) < 0.005 ? '0.00' : v.toFixed(2));
const f3 = (v) => v.toFixed(3);
const pc = (v) => Math.round(100 * v) + '%';
const sg = (v) => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(2);
const nf = (v) => v.toLocaleString('en-GB');
const sum = (a) => a.reduce((s, v) => s + v, 0);
const SCOL = { leaky: () => cat(7), nested: () => cat(0) };
const MORD = ['ifng', 'lasso', 'rf', 'gb'], MSHORT = { ifng: 'IFN-γ score', lasso: 'Lasso', rf: 'Random forest', gb: 'Boosting' };

export default async function render(main) {
  header(main, 'Biomarker classifier and nested CV',
    'Select a small set of genes from thousands and combine them into a model that predicts response for a new patient. The difficult part is not fitting the model but <b>estimating its accuracy honestly</b>: every step that looks at the outcome, including gene selection, must be repeated inside each cross-validation fold, and the final model must be <b>locked</b> and tested on an independent cohort.');
  lessonNav(main);
  stepCard(main, {
    stage: 'Bulk RNA-seq · step 3 of 4',
    question: 'Can pre-treatment tumour expression predict who responds to anti-PD-1, which genes carry the signal, and how accurate would the prediction be for patients from another hospital?',
    data: '1,933 genes (log-CPM) for the 177 discovery patients from Centres A and B that passed QC, with response; 120 patients from Centre C kept aside for external validation.',
    method: 'Lasso and elastic-net logistic regression; gene selection, scaling and tuning inside nested cross-validation; comparison with a prespecified IFN-γ score, a random forest and gradient boosting; a locked model applied unchanged to Centre C.',
    verdict: 'both', why: ' Building the classifier is machine learning (a penalised predictive model tuned for out-of-sample accuracy). Estimating and validating its accuracy is statistics: nested resampling, permutation nulls, confidence intervals, calibration and a prespecified external test.',
  });
  const [FS, CL, CP, EX, VA, LD, LC] = await Promise.all(['bulk_fs.json', 'bulk_classifier.json', 'bulk_compare.json', 'bulk_explain.json', 'bulk_validation.json', 'bulk_de_lab.json', 'bulk_clf_lab.json'].map((f) => omicsData(main, f)));
  if (![FS, CL, CP, EX, VA, LD, LC].every(Boolean)) return;

  // ------------------------------------------------------------------ shared numbers
  const n = FS.n, nR = FS.n_resp, p = FS.p, L = FS.lambda, nL = L.length, PL = FS.paths.lasso, PE = FS.paths.enet;
  const KS = CL.ks, ck = (lab, s, k) => CL.res[`${lab}|${s}|${k}`], kMax = KS[KS.length - 1], kMin = KS[0];
  const S = VA.strategies, LK = VA.lock, M = CP.models;
  const mAUC = (k) => d3.mean(M[k].auc), sdAUC = (k) => d3.deviation(M[k].auc);
  const ifngIn = (P, i) => P.genes.filter((g, j) => P.group[j] === 'IFNG module' && P.coef[j][i] !== 0).length;
  const nIF = LD.group.filter((g) => g === 'IFNG module').length;

  // ------------------------------------------------------------------ live pipeline (lab + worked examples)
  const G = LD.genes.length, XD = unpack(LD.x_b64, LD.x_scale), XC = unpack(LC.x_b64, LC.x_scale);
  const yD = LD.y, yC = LC.y, nD = yD.length, nC = yC.length, ALL = d3.range(nD);
  const IFIDX = LD.genes.map((g, j) => j).filter((j) => LD.group[j] === 'IFNG module');
  const permOf = (seed) => rng(seed).shuffle(yD);
  const tstat = (rows, yy) => {
    const s = [new Float64Array(G), new Float64Array(G)], ss = [new Float64Array(G), new Float64Array(G)], c = [0, 0];
    rows.forEach((i) => { const k = yy[i], o = i * G; c[k]++; for (let g = 0; g < G; g++) { const v = XD[o + g]; s[k][g] += v; ss[k][g] += v * v; } });
    const t = new Float64Array(G);
    for (let g = 0; g < G; g++) {
      const m1 = s[1][g] / c[1], m0 = s[0][g] / c[0], v = (ss[1][g] - c[1] * m1 * m1 + ss[0][g] - c[0] * m0 * m0) / (c[0] + c[1] - 2);
      t[g] = (m1 - m0) / Math.sqrt(Math.max(v, 1e-9) * (1 / c[0] + 1 / c[1]));
    }
    return t;
  };
  const topK = (t, k) => d3.range(G).sort((a, b) => Math.abs(t[b]) - Math.abs(t[a])).slice(0, k);
  const featOf = (X, i, genes) => (genes ? genes.map((g) => X[i * G + g]) : [mean(IFIDX.map((g) => X[i * G + g]))]);
  /** Whole pipeline on training rows: [select genes] → standardise → ridge-logistic. */
  function fitPipe(rows, yy, cfg, fixed) {
    const genes = cfg.mode === 'ifng' ? null : fixed || topK(tstat(rows, yy), cfg.k);
    const R = rows.map((i) => featOf(XD, i, genes)), q = R[0].length;
    const mu = d3.range(q).map((j) => mean(R.map((r) => r[j])));
    const sdv = d3.range(q).map((j) => Math.sqrt(mean(R.map((r) => (r[j] - mu[j]) ** 2))) || 1);
    const b = ridgeLogistic(R.map((r) => r.map((v, j) => (v - mu[j]) / sdv[j])), rows.map((i) => yy[i]), cfg.mode === 'ifng' ? 0 : cfg.lam);
    return { genes, predict: (X, i) => sigmoid(b[0] + featOf(X, i, genes).reduce((s, v, j) => s + b[j + 1] * (v - mu[j]) / sdv[j], 0)) };
  }
  function runLab(cfg) {
    const yy = cfg.lab === 'perm' ? permOf(cfg.seed || 1) : yD;
    const folds = stratFolds(yy, 5, 3), oof = new Array(nD);
    const fixed = cfg.mode !== 'ifng' && cfg.place === 'leaky' ? topK(tstat(ALL, yy), cfg.k) : null;
    folds.forEach((te) => { const tr = ALL.filter((i) => !te.includes(i)); const m = fitPipe(tr, yy, cfg, fixed); te.forEach((i) => { oof[i] = m.predict(XD, i); }); });
    const fin = fitPipe(ALL, yy, cfg), pC = d3.range(nC).map((i) => fin.predict(XC, i));
    const pTr = ALL.map((i) => fin.predict(XD, i));
    return { cfg, yy, oof, pC, cv: auc(oof, yy), ext: auc(pC, yC), app: auc(pTr, yy), genes: fin.genes, nIf: fin.genes ? fin.genes.filter((g) => LD.group[g] === 'IFNG module').length : nIF };
  }

  // =====================================================================================
  // 1. WHAT
  // =====================================================================================
  let ch = chapter(main, 'what', 'What is a biomarker classifier?', 'In one line: <b>many genes → a few selected genes → a fitted model → an honest accuracy estimate → a locked rule tested on new patients</b>.');
  prose(ch, `<p>A <b>biomarker classifier</b> is a fixed rule that turns a sample's molecular measurements into a predicted class or probability, here the probability that a patient responds to anti-PD-1. In bulk RNA-seq it is usually built in two parts: <b>feature selection</b> (choosing a small set of genes out of ${nf(p)}) and a <b>model</b> that combines them, typically a penalised logistic regression. Because the same data serve to choose the genes, fit the weights and estimate accuracy, the order of these steps decides whether the reported accuracy can be trusted.</p>
    <p>The diagram follows the analysis on this page, with the numbers from the simulated study.</p>`);
  ch.append(pipelineSvg([
    { t: 'Discovery data', l: [`${n} patients`, `${nR} responders`, `${nf(p)} genes`] },
    { t: 'Select genes', l: ['lasso / top-k by t', 'inside each', 'training fold'] },
    { t: 'Fit the model', l: ['penalised logistic', 'regression on the', 'selected genes'] },
    { t: 'Nested CV estimate', l: [`internal AUC ${f2(S.internal.auc)}`, '(each patient scored', 'by a model blind to it)'] },
    { t: 'Lock', l: [`${LK.genes.join(' + ')}`, 'weights, scaling,', `cut-off p ≥ ${f2(VA.thr)}`] },
    { t: 'External test', l: [`Centre C, ${VA.n_v} patients`, `AUC ${f2(S.frozen.auc)}`, 'model unchanged'] },
  ]));
  prose(ch, '<p>Key terms:</p>');
  table(ch, [
    { t: 'Feature selection', m: 'Choosing which genes enter the model', s: `${PL.nnz[PL.one_se]} of ${nf(p)} genes kept by the lasso` },
    { t: 'Lasso (L1 penalty)', m: 'Logistic regression with a penalty λ·Σ|β| that sets most coefficients exactly to zero', s: `selects ${FS.final.genes.join(' and ')}` },
    { t: 'Elastic net', m: 'Lasso plus a ridge (L2) term, so correlated genes enter together', s: `keeps ${ifngIn(PE, PE.one_se)} IFN-γ module genes at its chosen λ` },
    { t: 'Cross-validation (CV)', m: 'Split patients into k folds; train on k−1, score the held-out fold, rotate', s: '5 folds of about 35 patients' },
    { t: 'Nested CV', m: 'An inner CV loop for selection and tuning runs inside each outer training fold', s: 'the outer folds are used only for scoring' },
    { t: 'Leakage', m: 'Information from the test patients reaches a training step', s: `random labels reach AUC ${f2(ck('perm', 'leaky', kMax).mean)}` },
    { t: 'Locked model', m: 'Genes, preprocessing, weights and cut-off frozen before new data are seen', s: 'the model card in step 5' },
    { t: 'External validation', m: 'The locked model applied unchanged to an independent cohort', s: `Centre C, ${VA.n_v} patients` },
  ], [['t', 'Term'], ['m', 'Meaning'], ['s', 'In this study']]);

  // =====================================================================================
  // 2. WHY
  // =====================================================================================
  ch = chapter(main, 'why', 'Why do we need it?', `With ${nf(p)} genes and ${n} patients, a flexible enough model can fit any labels, including random ones.`);
  prose(ch, `<p>Differential expression (the previous step) ranks genes one at a time and controls false discoveries. It does not produce a rule for scoring a new patient, and it does not say how accurate such a rule would be. A classifier must provide:</p><ol>
    <li><b>A prediction for each new patient</b>, from a fixed list of genes and weights.</li>
    <li><b>A small panel</b>, so it can be measured with a cheap targeted assay.</li>
    <li><b>An accuracy estimate that holds for new patients</b>, which is the hardest part when genes far outnumber patients.</li></ol>`);
  const wPerm = runLab({ mode: 'topk', k: 50, place: 'leaky', lab: 'perm', lam: 1 });
  worked(ch, 'Fitting random labels',
    `<p>The response labels of the ${nD} patients are shuffled, so no gene can carry real information about them. The 50 genes with the largest |t| against these random labels are selected, and a ridge-logistic model is fitted on them:</p>
     <ul><li>AUC on the same patients (apparent): <b>${f2(wPerm.app)}</b>.</li>
     <li>5-fold CV, with the genes chosen once on all patients: <b>${f2(wPerm.cv)}</b>.</li>
     <li>The same model applied to the ${nC} Centre C patients and their real outcomes: <b>${f2(wPerm.ext)}</b>.</li></ul>
     <p>With enough genes to choose from, some correlate with any labelling by chance. Only an estimate in which the test patients play no part in any training step, including selection, reveals that the model has learned nothing.</p>`);
  table(ch, [
    { a: 'One gene or a DE list with a cut-off', s: 'no (or by p-value only)', h: 'not estimated', n: 'tiny' },
    { a: '<b>Prespecified score</b> (e.g. the mean of a published IFN-γ gene set)', s: 'none from these data', h: '<b>CV or a new cohort; no selection to leak</b>', n: 'small' },
    { a: 'Select genes on all patients, then cross-validate', s: 'yes', h: 'no: optimistic, often severely', n: 'any' },
    { a: '<b>Penalised model with selection inside nested CV</b>', s: '<b>yes</b>', h: '<b>yes</b>', n: '<b>moderate</b>' },
    { a: 'The same, then a locked model on an external cohort', s: 'yes', h: 'yes, plus transportability', n: 'two cohorts' },
  ], [['a', 'Approach'], ['s', 'Data-driven gene selection?'], ['h', 'Honest accuracy estimate?'], ['n', 'Data needed']]);

  // =====================================================================================
  // 3. WHERE
  // =====================================================================================
  ch = chapter(main, 'where', 'Where is it used?', 'Any setting where an omics profile is turned into a decision rule for individual samples.');
  cards(ch, [
    { tag: 'Immuno-oncology', title: 'Response to checkpoint inhibitors', body: 'IFN-γ and T-cell-inflamed expression profiles (for example an 18-gene GEP) predict anti-PD-1 response. Many such signatures are prespecified gene-set scores rather than lasso selections.' },
    { tag: 'Breast cancer', title: 'Recurrence scores and subtypes', body: 'Oncotype DX (21 genes) and PAM50/Prosigna lock their genes, normalisers and algorithm, and use reference samples on every run.' },
    { tag: 'Diagnostics', title: 'Tissue of origin, cancer detection', body: 'Penalised classifiers on expression or DNA-methylation panels assign tumour type or detect cancer from cell-free DNA.' },
    { tag: 'Infectious disease', title: 'Host-response signatures', body: 'Small blood-transcript panels distinguish bacterial from viral infection, or active from latent tuberculosis.' },
    { tag: 'Toxicology', title: 'Predicting drug toxicity', body: 'Expression signatures from treated cell lines or animal tissue classify compounds as likely hepatotoxic or not.' },
    { tag: 'Microbiome and proteomics', title: 'Case–control classifiers', body: 'Random forests or lasso on taxa or protein abundances; the same leakage and validation rules apply to any high-dimensional assay.' },
  ]);
  prose(ch, `<p><b>When not to use it:</b> the question is which genes differ between groups (use differential expression with FDR control); the outcome is time to an event (use Cox regression, next step); there are too few events to both select and validate (with ${nR} responders, a prespecified score is often the better choice); or the cohort is treated only with one drug, so a predictor of response cannot be distinguished from a marker of prognosis without a control arm.</p>`);

  // =====================================================================================
  // 4. HOW IT WORKS
  // =====================================================================================
  ch = chapter(main, 'how', 'How it works', 'Six steps, from cross-validation of a whole pipeline to a locked model tested in a new centre.');

  // ---- Step 1: pipeline and CV folds
  let rb = rung(ch, 1, 'Cross-validate the whole pipeline, not only the final fit',
    `<p>The <b>pipeline</b> is everything that turns training data into a prediction rule: gene selection, the means and SDs used for standardisation, the penalty λ, and the model weights. In <b>k-fold CV</b>, the patients are split into k folds; each fold is held out once while the <i>entire</i> pipeline is rerun on the other k − 1 folds. Each patient is then scored by a rule that never saw them, and one AUC (area under the ROC curve: the probability that a random responder scores above a random non-responder; 0.5 = chance) is computed from these out-of-fold scores. When λ or the number of genes is also tuned by CV, that inner loop runs inside each outer training set: <b>nested CV</b>.</p>`);
  const f0 = figure(rb, 'Which patients the gene-selection step sees',
    `Each row is one round of 5-fold CV over ${n} patients; the highlighted row is the current round. Coloured cells are its test fold. The bar above each row shows the patients that gene selection used. In the leaky scheme it includes the test fold.`);
  let fold = 0, stop0 = null;
  button(f0.controls, 'Play folds', () => { if (stop0) { stop0(); stop0 = null; return; } stop0 = every(900, () => { fold = (fold + 1) % 5; paint0(); }); }, true);
  const s0 = stats(f0.root);
  const row0 = el('div', { class: 'fig-row' }); f0.body.append(row0);
  const panels0 = ['leaky', 'nested'].map((s) => foldPanel(row0, s, n));
  legend(f0.body, [['training patients', muted()], ['test fold, leaky scheme', cat(7)], ['test fold, nested scheme', cat(0)], ['used by gene selection', ink()]]);
  function paint0() {
    panels0.forEach((pp) => pp.forEach((g, f) => g.attr('opacity', f === fold ? 1 : 0.22)));
    s0.set({ round: `${fold + 1} of 5`, 'test patients': Math.round((fold + 1) * n / 5) - Math.round(fold * n / 5), 'seen by selection, leaky': Math.round((fold + 1) * n / 5) - Math.round(fold * n / 5), 'seen by selection, nested': 0 });
  }
  paint0();

  // ---- Step 2: lasso / elastic net
  rb = rung(ch, 2, 'Select genes with a penalty: lasso and elastic net',
    `<p>The lasso fits a logistic regression while penalising the sum of absolute coefficients, λ·Σ|β| (genes standardised first, so the penalty treats them equally). A strong penalty sets every coefficient to zero; as λ weakens, genes enter one by one. The <b>path</b> is the set of fits over a grid of λ, and λ is chosen by cross-validated AUC. The <b>one-standard-error (1-SE) rule</b> picks the strongest penalty whose AUC is within one SE of the best, which favours smaller panels. The <b>elastic net</b> adds a ridge term, (1 − α)/2·Σβ², so correlated genes share the weight instead of one being chosen.</p>`);
  const f2g = figure(rb, 'The regularisation path',
    `Top: each line is one gene's coefficient (per SD) as λ weakens from left to right; the ${PL.genes.length} genes that enter first are drawn, coloured by the planted group. Bottom: CV AUC at each λ with ±1 SD band. Drag the slider or press Sweep; hover a line or point.`);
  let pname = 'lasso', li = PL.one_se, stop2 = null;
  select(f2g.controls, { label: 'Penalty', value: pname, options: [['lasso', 'lasso (L1)'], ['enet', 'elastic net (α = 0.5)']], onChange: (v) => { pname = v; li = FS.paths[v].one_se; sl2.set(li); draw2(); } });
  const sl2 = slider(f2g.controls, { label: 'λ (weaker →)', min: 0, max: nL - 1, step: 1, value: li, fmt: (v) => L[v].toPrecision(2), onInput: (v) => { halt2(); li = v; upd2(); } });
  button(f2g.controls, 'Sweep', () => { halt2(); let acc = 0; li = 0; sl2.set(0); upd2(); stop2 = animate((dt) => { acc += dt; if (acc < 150) return true; acc = 0; li++; sl2.set(li); upd2(); return li < nL - 1; }); }, true);
  button(f2g.controls, '1-SE choice', () => { halt2(); li = FS.paths[pname].one_se; sl2.set(li); upd2(); });
  const s2 = stats(f2g.root);
  const host2 = el('div'); f2g.body.append(host2);
  const grpUsed = GROUP_ORDER.filter((g) => PL.group.includes(g) || PE.group.includes(g));
  legend(f2g.body, grpUsed.map((g) => [groupLabel(g), groupColor(g)]));
  const chips = el('div', { class: 'fig-caption' }); f2g.body.append(chips);
  let FP, FA, vP, vA, lines2;
  function halt2() { if (stop2) { stop2(); stop2 = null; } }
  function draw2() {
    host2.innerHTML = '';
    const P = FS.paths[pname], ym = d3.max(P.coef.flat(), Math.abs) * 1.05;
    FP = frame(host2, { width: 820, height: 250, x: [L[0], L[nL - 1]], y: [-ym, ym], yLabel: 'coefficient (per SD)', xScale: d3.scaleLog() });
    FP.g.select('.axis.x').call(d3.axisBottom(FP.x).ticks(6, '.2~g'));
    const ord = d3.range(P.genes.length).sort((a, b) => (P.group[b] === 'filler') - (P.group[a] === 'filler'));
    lines2 = FP.layer('paths').selectAll('path').data(ord).join('path').attr('fill', 'none').attr('stroke', (j) => groupColor(P.group[j])).attr('stroke-width', (j) => (P.group[j] === 'filler' ? 1.1 : 2))
      .attr('d', (j) => d3.line().x((c, k) => FP.x(L[k])).y((c) => FP.y(c))(P.coef[j]))
      .on('mousemove', (e, j) => tip(e, `<b>${P.genes[j]}</b> · ${groupLabel(P.group[j])}<br>coefficient at this λ: ${P.coef[j][li].toFixed(3)}`)).on('mouseleave', untip);
    vP = FP.layer('v').append('line').attr('y1', 0).attr('y2', FP.h).attr('stroke', ink()).attr('stroke-width', 1.5);
    FA = frame(host2, { width: 820, height: 190, x: [L[0], L[nL - 1]], y: [0.45, 0.9], xLabel: 'penalty λ (log scale; left = strong, right = weak)', yLabel: 'CV AUC', xScale: d3.scaleLog() });
    FA.g.select('.axis.x').call(d3.axisBottom(FA.x).ticks(6, '.2~g'));
    const pts = L.map((l, k) => ({ l, m: P.auc_mean[k], s: P.auc_sd[k] || 0, nz: P.nnz[k] }));
    FA.layer('band').append('path').datum(pts).attr('fill', css('--accent-soft')).attr('d', d3.area().x((d) => FA.x(d.l)).y0((d) => FA.y(d.m - d.s)).y1((d) => FA.y(d.m + d.s)));
    FA.layer('line').append('path').datum(pts).attr('fill', 'none').attr('stroke', css('--accent')).attr('stroke-width', 2).attr('d', d3.line().x((d) => FA.x(d.l)).y((d) => FA.y(d.m)));
    FA.layer('pts').selectAll('circle').data(pts).join('circle').attr('cx', (d) => FA.x(d.l)).attr('cy', (d) => FA.y(d.m)).attr('r', 4).attr('fill', css('--accent')).attr('stroke', css('--surface'))
      .on('mousemove', (e, d) => tip(e, `λ = ${d.l.toPrecision(2)}<br>CV AUC ${f3(d.m)} ± ${f3(d.s)}<br>${d.nz} genes non-zero`)).on('mouseleave', untip);
    [['best', P.best, -12, 'middle', 0], ['1-SE', P.one_se, 20, 'end', -6]].forEach(([t, k, dy, an, dx]) => FA.layer('mk').append('text').attr('x', FA.x(L[k]) + dx).attr('y', FA.y(P.auc_mean[k]) + dy).attr('text-anchor', an).attr('font-size', 11).attr('fill', ink2()).text(t));
    vA = FA.layer('v').append('line').attr('y1', 0).attr('y2', FA.h).attr('stroke', ink()).attr('stroke-width', 1.5);
    upd2();
  }
  function upd2() {
    const P = FS.paths[pname];
    [[vP, FP], [vA, FA]].forEach(([v, F]) => v.attr('x1', F.x(L[li])).attr('x2', F.x(L[li])));
    lines2.attr('opacity', (j) => (P.coef[j][li] !== 0 ? 1 : 0.3));
    const on = d3.range(P.genes.length).filter((j) => P.coef[j][li] !== 0).sort((a, b) => Math.abs(P.coef[b][li]) - Math.abs(P.coef[a][li]));
    s2.set({ λ: L[li].toPrecision(2), 'non-zero genes': P.nnz[li], 'CV AUC': `${f3(P.auc_mean[li])} ± ${f3(P.auc_sd[li] || 0)}`, 'IFN-γ module genes in': `${ifngIn(P, li)} of ${nIF}` });
    chips.innerHTML = 'Selected (largest |coefficient| first): ' + (on.length ? on.slice(0, 16).map((j) => `<span class="tag" style="border-left:5px solid ${groupColor(P.group[j])}">${P.genes[j]}</span>`).join('') + (P.nnz[li] > 16 ? ` <span class="tag">+${P.nnz[li] - 16} more</span>` : '') : '<span class="tag">none: every coefficient is zero</span>');
  }
  draw2();
  const st = FS.stability.medium, stIf = sum(st.genes.map((g, i) => (st.group[i] === 'IFNG module' ? st.freq[i] : 0)));
  prose(rb, `<p>The lasso's AUC rises from 0.5 to ${f3(PL.auc_mean[PL.best])} at ${PL.nnz[PL.best]} genes, then declines as filler genes enter (${f3(PL.auc_mean[nL - 1])} with ${PL.nnz[nL - 1]} genes): that decline is overfitting. The 1-SE rule keeps ${PL.nnz[PL.one_se]} genes, <b>${FS.final.genes.join(' and ')}</b>, both from the planted IFN-γ module of ${nIF} correlated genes. Which members are chosen is unstable: refitted on 100 random halves of the patients, the most frequent gene (${st.genes[0]}) is selected in only ${pc(st.freq[0])} of fits, whereas on average ${f2(stIf)} of the ${f2(st.q)} genes selected per fit belong to the IFN-γ module. The <b>module</b> is stable; the choice of member is not.</p>`);

  // ---- Step 3: leakage trap
  rb = rung(ch, 3, 'The leakage trap: selection outside the folds',
    `<p>Pipeline: choose the top-k genes by |t|, standardise, fit an L2-penalised logistic regression (λ tuned by inner CV); 5-fold CV repeated ${CL.reps} times. In the <b>leaky</b> scheme, the genes are chosen once on all ${CL.n} patients and only the model fit is cross-validated. In the <b>nested</b> scheme, selection runs inside each training fold. As a null check, the same procedure is run on <b>permuted labels</b>, where the true AUC is 0.5.</p>`);
  const f3g = figure(rb, 'The leakage trap, measured',
    `Left: out-of-fold ROC curves (first repeat) at the chosen k. Right: mean CV AUC against k for both schemes, ±1 SD band, small dots = individual repeats; solid = real labels, dashed = permuted labels.`);
  let lab3 = 'perm', k3 = kMax;
  select(f3g.controls, { label: 'Labels', value: lab3, options: [['real', 'real response labels'], ['perm', 'permuted (random) labels']], onChange: (v) => { lab3 = v; draw3(); } });
  slider(f3g.controls, { label: 'genes selected k', min: 0, max: KS.length - 1, step: 1, value: KS.length - 1, fmt: (v) => KS[v], onInput: (v) => { k3 = KS[v]; draw3(); } });
  const s3 = stats(f3g.root);
  const row3 = el('div', { class: 'fig-row' }); f3g.body.append(row3);
  const a3 = el('div'), b3 = el('div'); row3.append(a3, b3);
  const FR = frame(a3, { width: 400, height: 360, x: [0, 1], y: [0, 1], xLabel: 'false-positive rate', yLabel: 'true-positive rate' });
  diag(FR);
  const FK = frame(b3, { width: 500, height: 360, x: [-0.3, KS.length - 0.7], y: [0.3, 1], xLabel: 'genes selected k', yLabel: 'cross-validated AUC' });
  FK.g.select('.axis.x').call(d3.axisBottom(FK.x).tickValues(d3.range(KS.length)).tickFormat((i) => KS[i]));
  FK.layer('ch').append('line').attr('x1', 0).attr('x2', FK.w).attr('y1', FK.y(0.5)).attr('y2', FK.y(0.5)).attr('stroke', muted()).attr('stroke-dasharray', '3 3');
  const jr = rng(5), jit3 = d3.range(CL.reps).map(() => (jr() - 0.5) * 0.2);
  ['real', 'perm'].forEach((lb) => ['leaky', 'nested'].forEach((s) => {
    const pts = KS.map((kk, i) => ({ i, kk, ...ck(lb, s, kk) }));
    FK.layer('band').append('path').datum(pts).attr('fill', SCOL[s]()).attr('opacity', 0.1).attr('d', d3.area().x((d) => FK.x(d.i)).y0((d) => FK.y(d.mean - d.sd)).y1((d) => FK.y(d.mean + d.sd)));
    FK.layer('reps').selectAll(null).data(pts.flatMap((d) => d.auc.map((a, r) => ({ i: d.i, a, r })))).join('circle').attr('cx', (d) => FK.x(d.i + jit3[d.r])).attr('cy', (d) => FK.y(d.a)).attr('r', 2).attr('fill', SCOL[s]()).attr('opacity', 0.4);
    FK.layer('lines').append('path').datum(pts).attr('fill', 'none').attr('stroke', SCOL[s]()).attr('stroke-width', 2).attr('stroke-dasharray', lb === 'perm' ? '6 4' : null).attr('d', d3.line().x((d) => FK.x(d.i)).y((d) => FK.y(d.mean)));
    FK.layer('pts').selectAll(null).data(pts).join('circle').attr('cx', (d) => FK.x(d.i)).attr('cy', (d) => FK.y(d.mean)).attr('r', 4.5).attr('fill', SCOL[s]()).attr('stroke', css('--surface')).attr('stroke-width', 1.2)
      .on('mousemove', (e, d) => tip(e, `<b>${lb === 'real' ? 'real' : 'permuted'} labels, ${s}</b><br>k = ${d.kk}: AUC ${f3(d.mean)} ± ${f3(d.sd)}`)).on('mouseleave', untip);
  }));
  const kLine = FK.layer('k').append('line').attr('y1', 0).attr('y2', FK.h).attr('stroke', ink()).attr('stroke-dasharray', '2 3');
  legend(f3g.body, [['leaky: genes chosen on all patients', SCOL.leaky()], ['nested: genes chosen inside each training fold', SCOL.nested()], ['dashed line = permuted labels', muted()]]);
  function draw3() {
    const rows = ['leaky', 'nested'].map((s) => ({ s, r: ck(lab3, s, k3) }));
    FR.layer('roc').selectAll('path').data(rows).join('path').attr('fill', 'none').attr('stroke', (d) => SCOL[d.s]()).attr('stroke-width', 2.5).attr('stroke-dasharray', lab3 === 'perm' ? '6 4' : null)
      .attr('d', (d) => d3.line().curve(d3.curveStepAfter).x((q) => FR.x(q[0])).y((q) => FR.y(q[1]))([...d.r.roc, [1, 1]]));
    kLine.attr('x1', FK.x(KS.indexOf(k3))).attr('x2', FK.x(KS.indexOf(k3)));
    const lk = ck(lab3, 'leaky', k3), ns = ck(lab3, 'nested', k3);
    s3.set({ k: k3, 'leaky AUC': `${f3(lk.mean)} ± ${f3(lk.sd)}`, 'nested AUC': `${f3(ns.mean)} ± ${f3(ns.sd)}`, 'optimism (leaky − nested)': sg(lk.mean - ns.mean) });
  }
  draw3();
  const bestK = KS.reduce((a, b) => (ck('real', 'nested', b).mean > ck('real', 'nested', a).mean ? b : a));
  const pnR = d3.extent(KS, (k) => ck('perm', 'nested', k).mean);
  prose(rb, `<p>With permuted labels and k = ${kMax}, leaky CV reports AUC ${f2(ck('perm', 'leaky', kMax).mean)}; nested CV stays between ${f2(pnR[0])} and ${f2(pnR[1])} for every k. With real labels the schemes agree at k = ${kMin} (${f2(ck('real', 'leaky', kMin).mean)} vs ${f2(ck('real', 'nested', kMin).mean)}), because any training fold finds the same IFN-γ genes, and diverge at k = ${kMax} (${f2(ck('real', 'leaky', kMax).mean)} vs ${f2(ck('real', 'nested', kMax).mean)}). The honest curve is highest at k = ${bestK}: here a small panel is also the more accurate one.</p>`);
  quiz(rb, 'A study standardises all genes, removes batch effects with ComBat and filters to the 500 most variable genes, all on the full cohort, before nested CV. Which of these steps leak?',
    'Only steps that use the outcome create the large optimism shown above. Variance filtering and standardisation do not use labels, so their leakage is small. ComBat with response as a protected covariate does use the labels and can inflate accuracy. For a model that will score new patients one at a time, every step must be re-estimable from the training data alone, so all three belong inside the pipeline.');

  // ---- Step 4: model comparison
  rb = rung(ch, 4, 'Compare models on the same folds, including a prespecified score',
    `<p>Four candidates were evaluated with ${CP.reps} repeats of the same nested 5-fold CV (inner 3-fold CV for tuning): a <b>prespecified IFN-γ score</b> (the mean of the ${nIF} module genes, chosen before seeing these data, in a one-feature logistic regression), the <b>lasso</b>, a <b>random forest</b> and <b>gradient boosting</b>. Sharing the folds makes the comparison <b>paired</b>, which removes fold-to-fold noise. Calibration is assessed with the <b>calibration slope</b> (1 is ideal; below 1, predictions are too extreme; above 1, too timid).</p>`);
  const f4 = figure(rb, 'Discrimination and calibration of four candidates',
    'Left: out-of-fold AUC, one dot per repeat of the whole nested CV; the bar is the mean. Right: reliability diagram for the selected model (patients in equal-sized groups by predicted probability; on the diagonal = calibrated). Hover for values.');
  let m4 = 'rf';
  select(f4.controls, { label: 'Model', value: m4, options: MORD.map((k) => [k, M[k].label]), onChange: (v) => { m4 = v; draw4(); } });
  const s4 = stats(f4.root);
  const row4 = el('div', { class: 'fig-row' }); f4.body.append(row4);
  const a4 = el('div'), b4 = el('div'); row4.append(a4, b4);
  const aLo = Math.floor((d3.min(MORD, (k) => d3.min(M[k].auc)) - 0.02) * 50) / 50, aHi = Math.ceil((d3.max(MORD, (k) => d3.max(M[k].auc)) + 0.02) * 50) / 50;
  const FD = frame(a4, { width: 460, height: 340, x: [-0.5, 3.5], y: [aLo, aHi], yLabel: 'out-of-fold AUC (per repeat)' });
  FD.g.select('.axis.x').call(d3.axisBottom(FD.x).tickValues([0, 1, 2, 3]).tickFormat((i) => MSHORT[MORD[i]]));
  const FC = frame(b4, { width: 460, height: 340, x: [0, 1], y: [0, 1], xLabel: 'mean predicted probability (group)', yLabel: 'observed response rate (group)' });
  diag(FC);
  legend(f4.body, MORD.map((k, i) => [M[k].label, cat(i + 1)]));
  function draw4() {
    const op = (k) => (k === m4 ? 1 : 0.5);
    FD.layer('bars').selectAll('line').data(MORD).join('line').attr('x1', (k, i) => FD.x(i - 0.3)).attr('x2', (k, i) => FD.x(i + 0.3)).attr('y1', (k) => FD.y(mAUC(k))).attr('y2', (k) => FD.y(mAUC(k)))
      .attr('stroke', (k, i) => cat(i + 1)).attr('stroke-width', 3).attr('opacity', op);
    FD.layer('dots').selectAll('circle').data(MORD.flatMap((k, i) => M[k].auc.map((a, r) => ({ k, i, r, a })))).join('circle').attr('cx', (d) => FD.x(d.i + (d.r - (CP.reps - 1) / 2) * 0.09)).attr('cy', (d) => FD.y(d.a)).attr('r', 5)
      .attr('fill', (d) => cat(d.i + 1)).attr('stroke', css('--surface')).attr('stroke-width', 1.2).attr('opacity', (d) => op(d.k))
      .on('mousemove', (e, d) => tip(e, `<b>${M[d.k].label}</b><br>repeat ${d.r + 1}: AUC ${f3(d.a)}`)).on('mouseleave', untip);
    const ci = MORD.indexOf(m4) + 1, bins = qBins(CP.y, M[m4].p, 6);
    FC.layer('l').selectAll('path').data([bins]).join('path').attr('fill', 'none').attr('stroke', cat(ci)).attr('stroke-width', 2).attr('d', d3.line().x((b) => FC.x(b.mp)).y((b) => FC.y(b.obs)));
    FC.layer('p').selectAll('circle').data(bins).join('circle').attr('cx', (b) => FC.x(b.mp)).attr('cy', (b) => FC.y(b.obs)).attr('r', 6).attr('fill', cat(ci)).attr('stroke', css('--surface')).attr('stroke-width', 1.5)
      .on('mousemove', (e, b) => tip(e, `${b.n} patients<br>mean predicted ${f2(b.mp)}<br>observed ${f2(b.obs)}`)).on('mouseleave', untip);
    const o = { 'mean AUC ± SD': `${f3(mAUC(m4))} ± ${f3(sdAUC(m4))}`, 'calibration slope (repeat 1)': f2(M[m4].rep0.slope), 'Brier score': f3(M[m4].rep0.brier) };
    if (m4 !== 'lasso') { const dd = CP.diff_vs_lasso[m4]; o['paired Δ AUC vs lasso'] = `${sg(d3.mean(dd))} (better in ${d3.sum(dd, (v) => v > 0)} of ${dd.length} repeats)`; }
    s4.set(o);
  }
  draw4();
  const best4 = MORD.slice().sort((a, b) => mAUC(b) - mAUC(a))[0], spread4 = d3.max(MORD, mAUC) - d3.min(MORD, mAUC);
  const optim = ['lasso', 'rf', 'gb'].map((k) => `${MSHORT[k].toLowerCase()} ${f2(d3.max(M[k].plain_mean))} vs ${f2(mAUC(k))}`);
  prose(rb, `<p>The mean AUCs span ${f3(spread4)}, from ${f3(d3.min(MORD, mAUC))} to ${f3(d3.max(MORD, mAUC))}, and the highest belongs to the <b>${best4 === 'ifng' ? 'prespecified IFN-γ score, which involves no gene selection at all' : M[best4].label.toLowerCase()}</b>. The signal in this simulation is additive and carried by one correlated module, so flexible learners have nothing extra to find, and data-driven selection adds variance. Calibration differs more than AUC: slopes ${MORD.map((k) => `${MSHORT[k]} ${f2(M[k].rep0.slope)}`).join(', ')}. Reporting the best-looking tuning configuration instead of the nested estimate is a further, smaller optimism (best grid point vs nested: ${optim.join('; ')}).</p>`);

  // ---- Step 5: lock and validate
  rb = rung(ch, 5, 'Lock the model and validate it externally',
    `<p>Nested CV estimates the performance of the <i>procedure</i>; the genes it picks may differ between folds. The deliverable is one model refitted on all discovery patients and then <b>locked</b>: every element that turns raw data into a decision is written down before the validation outcomes are seen.</p>`);
  table(rb, [
    { k: 'Genes', v: `${LK.genes.join(', ')} (lasso, 1-SE rule)` },
    { k: 'Weights', v: `unpenalised refit on those genes: ${LK.genes.map((g, j) => `${g} ${f2(LK.coef[j])}`).join(', ')} per SD (the lasso's own coefficients, ${LK.lasso_coef.map(f2).join(' and ')}, are shrunk towards zero)` },
    { k: 'Preprocessing', v: `log2 CPM; z-score with discovery means ${LK.mu.map(f2).join(', ')} and SDs ${LK.sd.map(f2).join(', ')}` },
    { k: 'Linear predictor', v: `${LK.intercept < 0 ? '−' : ''}${f2(Math.abs(LK.intercept))} ${LK.genes.map((g, j) => `${LK.coef[j] >= 0 ? '+' : '−'} ${f2(Math.abs(LK.coef[j]))}·z(${g})`).join(' ')}` },
    { k: 'Cut-off', v: `P(response) ≥ ${f2(VA.thr)} (Youden index on internal CV predictions)` },
  ], [['k', 'Locked element'], ['v', 'Value']], { cls: 'compact' });
  const f5 = figure(rb, 'Internal vs external performance of the locked model',
    'Choose the cohort and how Centre C data are fed to the model. Left: ROC curve (grey = internal CV reference). Right: calibration in five equal-sized groups; the dashed vertical line is the frozen cut-off.');
  let k5 = 'frozen';
  select(f5.controls, { label: 'Cohort / strategy', value: k5, options: Object.entries(S).map(([k, v]) => [k, v.label]), onChange: (v) => { k5 = v; draw5(); } });
  const s5 = stats(f5.root);
  const row5 = el('div', { class: 'fig-row' }); f5.body.append(row5);
  const a5 = el('div'), b5 = el('div'); row5.append(a5, b5);
  const R5 = frame(a5, { width: 460, height: 340, x: [0, 1], y: [0, 1], xLabel: 'false-positive rate', yLabel: 'true-positive rate' });
  diag(R5);
  R5.layer('ref').append('path').datum(rocXY(S.internal.p, S.internal.y)).attr('fill', 'none').attr('stroke', muted()).attr('stroke-width', 2).attr('d', d3.line().curve(d3.curveStepAfter).x((d) => R5.x(d[0])).y((d) => R5.y(d[1])));
  const C5 = frame(b5, { width: 460, height: 340, x: [0, 1], y: [0, 1], xLabel: 'mean predicted probability (group)', yLabel: 'observed response rate (group)' });
  diag(C5);
  C5.layer('thr').append('line').attr('x1', C5.x(VA.thr)).attr('x2', C5.x(VA.thr)).attr('y1', 0).attr('y2', C5.h).attr('stroke', ink2()).attr('stroke-dasharray', '5 4');
  legend(f5.body, [['selected cohort / strategy', cat(1)], ['internal CV reference', muted()]]);
  function draw5() {
    const s = S[k5];
    R5.layer('roc').selectAll('path').data([rocXY(s.p, s.y)]).join('path').attr('fill', 'none').attr('stroke', cat(1)).attr('stroke-width', 2.5).attr('d', d3.line().curve(d3.curveStepAfter).x((d) => R5.x(d[0])).y((d) => R5.y(d[1])));
    const b = qBins(s.y, s.p, 5), rbn = qBins(S.internal.y, S.internal.p, 5);
    C5.layer('ref').selectAll('path').data([rbn]).join('path').attr('fill', 'none').attr('stroke', muted()).attr('stroke-width', 2).attr('d', d3.line().x((d) => C5.x(d.mp)).y((d) => C5.y(d.obs)));
    C5.layer('l').selectAll('path').data([b]).join('path').attr('fill', 'none').attr('stroke', cat(1)).attr('stroke-width', 2).attr('d', d3.line().x((d) => C5.x(d.mp)).y((d) => C5.y(d.obs)));
    C5.layer('p').selectAll('circle').data(b).join('circle').attr('cx', (d) => C5.x(d.mp)).attr('cy', (d) => C5.y(d.obs)).attr('r', 6).attr('fill', cat(1)).attr('stroke', css('--surface')).attr('stroke-width', 1.5)
      .on('mousemove', (e, d) => tip(e, `${d.n} patients<br>mean predicted ${f2(d.mp)}<br>observed ${f2(d.obs)}`)).on('mouseleave', untip);
    s5.set({ 'AUC [95% CI]': `${f3(s.auc)} [${f2(s.ci[0])}, ${f2(s.ci[1])}]`, 'mean predicted vs observed': `${f2(s.mean_p)} vs ${f2(s.obs)}`, 'calibration-in-the-large': sg(s.citl), 'calibration slope': f2(s.slope), 'sensitivity / specificity at cut-off': `${f2(s.sens)} / ${f2(s.spec)}` });
  }
  draw5();
  const fr = S.frozen, it = S.internal, ap = S.apparent;
  prose(rb, `<p>The AUC ladder: apparent ${f2(ap.auc)}, internal CV ${f2(it.auc)}, Centre C ${f2(fr.auc)} (95% CI ${f2(fr.ci[0])}–${f2(fr.ci[1])}). ${fr.auc > it.auc ? `The external AUC is <i>higher</i> than the internal one. This does not indicate a better model: Centre C enrolled more desert tumours (${VA.mix_v.desert} of ${VA.n_v}, against ${VA.mix_d.desert} of ${VA.n_d} in discovery), which are easy to classify as non-responders, so discrimination depends on case mix. ` : ''}A per-gene technical offset at a new centre shifts every patient's score by the same amount, which leaves the AUC unchanged but moves the probabilities. For the two locked genes the Centre C shift is small, so the drift is modest: frozen, the mean prediction is ${f2(fr.mean_p)} against ${f2(fr.obs)} observed (calibration-in-the-large ${sg(fr.citl)}); with the offset estimated from bridging samples (the same RNA sequenced at both centres) it is ${f2(S.bridge.mean_p)}. Models that sum many genes accumulate larger offsets (step 6). Re-standardising within Centre C (${f2(S.rez.mean_p)}) makes each patient's score depend on the rest of the cohort, so it is not a locked model. Recalibrating on Centre C outcomes is legitimate model updating, but the updated model then needs a new validation cohort.</p>`);

  // ---- Step 6: lab
  rb = rung(ch, 6, 'Classifier lab: build, cross-validate and validate',
    `<p>The lab runs the pipeline live on the ${nD} discovery patients (uncorrected log-CPM of all ${nf(G)} genes): select the top-k genes by |t| or use the prespecified IFN-γ score, standardise with training-fold statistics, and fit a ridge-penalised logistic regression with penalty λ. The 5-fold CV estimate is compared with the AUC of the final model, refitted on all discovery patients, in the ${nC} Centre C patients with their real outcomes. λ is set by hand here; in a full analysis it would be tuned in an inner loop.</p>`);
  const KOPT = [1, 2, 5, 10, 20, 50], LOPT = [0.1, 1, 10, 100];
  const cfg = { mode: 'topk', k: 20, place: 'leaky', lab: 'real', lam: 1, seed: 1 };
  const f6 = figure(rb, 'Classifier lab',
    'Left: ROC curves of the current run, out-of-fold in discovery (dashed) and in Centre C (solid). Right: every configuration tried so far, the CV estimate (x) against the Centre C result (y). Points below the diagonal were optimistic. Hover a point for its settings.');
  const sMode = select(f6.controls, { label: 'Genes', value: cfg.mode, options: [['topk', 'top-k by |t|'], ['ifng', `prespecified IFN-γ score (${nIF} genes)`]], onChange: (v) => { cfg.mode = v; run6(); } });
  const sK = slider(f6.controls, { label: 'k', min: 0, max: KOPT.length - 1, step: 1, value: KOPT.indexOf(cfg.k), fmt: (i) => KOPT[i], onInput: (i) => { cfg.k = KOPT[i]; run6(); } });
  const sPl = select(f6.controls, { label: 'Selection', value: cfg.place, options: [['leaky', 'once, on all patients (leaky)'], ['nested', 'inside each training fold (nested)']], onChange: (v) => { cfg.place = v; run6(); } });
  const sLab = select(f6.controls, { label: 'Labels', value: cfg.lab, options: [['real', 'real'], ['perm', 'permuted']], onChange: (v) => { cfg.lab = v; run6(); } });
  slider(f6.controls, { label: 'ridge λ', min: 0, max: LOPT.length - 1, step: 1, value: 1, fmt: (i) => LOPT[i], onInput: (i) => { cfg.lam = LOPT[i]; run6(); } });
  button(f6.controls, 'New permutation', () => { cfg.seed++; if (cfg.lab !== 'perm') { cfg.lab = 'perm'; sLab.value = 'perm'; } run6(); });
  button(f6.controls, 'Clear history', () => { hist6.length = 0; run6(); });
  const s6 = stats(f6.root);
  const row6 = el('div', { class: 'fig-row' }); f6.body.append(row6);
  const a6 = el('div'), b6 = el('div'); row6.append(a6, b6);
  const R6 = frame(a6, { width: 460, height: 360, x: [0, 1], y: [0, 1], xLabel: 'false-positive rate', yLabel: 'true-positive rate' });
  diag(R6);
  const H6 = frame(b6, { width: 460, height: 360, x: [0.3, 1], y: [0.3, 1], xLabel: 'CV AUC in discovery (the estimate)', yLabel: 'AUC in Centre C (the result)' });
  diag(H6, [0.3, 1]);
  H6.layer('lab').append('text').attr('x', H6.x(0.98)).attr('y', H6.y(0.33)).attr('text-anchor', 'end').attr('font-size', 11.5).attr('fill', ink2()).text('below diagonal: estimate too optimistic');
  legend(f6.body, [['selection inside folds (nested)', SCOL.nested()], ['selection on all patients (leaky)', SCOL.leaky()], ['prespecified score', cat(2)], ['open marker = permuted labels', muted()]]);
  const genes6 = el('div', { class: 'fig-caption' }); f6.body.append(genes6);
  const hist6 = [];
  const keyOf = (c) => `${c.mode}|${c.mode === 'ifng' ? '' : c.k + '|' + c.place}|${c.lab}${c.lab === 'perm' ? c.seed : ''}|${c.mode === 'ifng' ? '' : c.lam}`;
  const colOf = (c) => (c.mode === 'ifng' ? cat(2) : SCOL[c.place]());
  const descr = (c) => (c.mode === 'ifng' ? 'IFN-γ score' : `top ${c.k}, ${c.place}, λ ${c.lam}`) + (c.lab === 'perm' ? `, permuted #${c.seed}` : ', real labels');
  function run6() {
    const r = runLab({ ...cfg }), kk = keyOf(cfg);
    if (!hist6.some((h) => h.key === kk)) { hist6.push({ key: kk, cfg: { ...cfg }, cv: r.cv, ext: r.ext }); if (hist6.length > 24) hist6.shift(); }
    sK.input.disabled = sPl.disabled = cfg.mode === 'ifng';
    R6.layer('roc').selectAll('path').data([[r.oof, r.yy, '6 4'], [r.pC, yC, null]]).join('path').attr('fill', 'none').attr('stroke', colOf(cfg)).attr('stroke-width', 2.4).attr('stroke-dasharray', (d) => d[2])
      .attr('d', (d) => d3.line().curve(d3.curveStepAfter).x((q) => R6.x(q[0])).y((q) => R6.y(q[1]))(rocXY(d[0], d[1])));
    H6.layer('pts').selectAll('circle').data(hist6, (h) => h.key).join('circle').attr('cx', (h) => H6.x(Math.max(0.3, h.cv))).attr('cy', (h) => H6.y(Math.max(0.3, h.ext))).attr('r', (h) => (h.key === kk ? 7.5 : 5.5))
      .attr('fill', (h) => (h.cfg.lab === 'perm' ? css('--surface') : colOf(h.cfg))).attr('stroke', (h) => (h.cfg.lab === 'perm' ? colOf(h.cfg) : css('--surface'))).attr('stroke-width', (h) => (h.cfg.lab === 'perm' ? 2.2 : 1.4))
      .on('mousemove', (e, h) => tip(e, `<b>${descr(h.cfg)}</b><br>CV AUC ${f3(h.cv)} · Centre C AUC ${f3(h.ext)}`)).on('mouseleave', untip);
    s6.set({ 'CV AUC (discovery)': f3(r.cv), 'Centre C AUC': f3(r.ext), 'optimism (CV − Centre C)': sg(r.cv - r.ext), 'apparent AUC (training data)': f3(r.app), 'IFN-γ module genes in final model': r.genes ? `${r.nIf} of ${r.genes.length}` : `${nIF} (fixed score)`, 'Centre C mean predicted vs observed': `${f2(mean(r.pC))} vs ${f2(mean(yC))}` });
    genes6.innerHTML = r.genes ? 'Final model genes (largest |t| first): ' + r.genes.slice(0, 20).map((g) => `<span class="tag" style="border-left:5px solid ${groupColor(LD.group[g])}">${LD.genes[g]}</span>`).join('') + (r.genes.length > 20 ? ` <span class="tag">+${r.genes.length - 20} more</span>` : '') : `Score = mean log-CPM of ${LD.genes.filter((g, j) => LD.group[j] === 'IFNG module').join(', ')}.`;
  }
  // presets for the try/see/means table (each run is a full 5-fold CV + final fit + Centre C)
  const T = {
    ifng: runLab({ mode: 'ifng', lab: 'real', lam: 1 }),
    n20: runLab({ mode: 'topk', k: 20, place: 'nested', lab: 'real', lam: 1 }),
    l20: runLab({ mode: 'topk', k: 20, place: 'leaky', lab: 'real', lam: 1 }),
    lp: runLab({ mode: 'topk', k: 50, place: 'leaky', lab: 'perm', lam: 1, seed: 1 }),
    np: runLab({ mode: 'topk', k: 50, place: 'nested', lab: 'perm', lam: 1, seed: 1 }),
    lo: runLab({ mode: 'topk', k: 50, place: 'nested', lab: 'real', lam: 0.1 }),
    hi: runLab({ mode: 'topk', k: 50, place: 'nested', lab: 'real', lam: 100 }),
  };
  const PR = Object.fromEntries(['leaky', 'nested'].map((pl) => [pl, [1, 2, 3, 4].map((sd) => (sd === 1 ? T[pl === 'leaky' ? 'lp' : 'np'] : runLab({ mode: 'topk', k: 50, place: pl, lab: 'perm', lam: 1, seed: sd })))]));
  const rngTxt = (rs, key) => { const e = d3.extent(rs, (r) => r[key]); return `${f2(e[0])}–${f2(e[1])}`; };
  [T.ifng, T.n20, T.np, T.lp].forEach((r) => hist6.push({ key: keyOf({ seed: 1, ...r.cfg }), cfg: { seed: 1, ...r.cfg }, cv: r.cv, ext: r.ext }));
  run6();
  prose(rb, '<p><b>Suggested comparisons</b>:</p>');
  const vs = (r) => `CV ${f2(r.cv)}, Centre C ${f2(r.ext)}`;
  table(rb, [
    { t: '<b>Prespecified IFN-γ score</b>', s: vs(T.ifng), m: `No data-driven selection, so nothing can leak.${T.ifng.ext > T.ifng.cv ? ' Centre C scores higher than the CV estimate because of its case mix (step 5), not because the estimate was pessimistic by design.' : ''}` },
    { t: 'Top 20 genes, <b>nested</b> → <b>leaky</b>', s: `${vs(T.n20)} → ${vs(T.l20)}`, m: `Selecting on all patients ${T.l20.cv > T.n20.cv ? 'raises the CV estimate without improving the model' : 'changes the estimate little here, because the top genes are the same strong IFN-γ module in every fold'}. The final model, and therefore the Centre C result, is identical in both schemes: only the estimate changes.` },
    { t: 'Top 50, <b>permuted labels</b>, leaky → nested', s: `${vs(T.lp)} → ${vs(T.np)}`, m: 'Leaky CV reports a predictive model built on random labels; nested CV and Centre C return values near 0.5 (see the next row for the spread across permutations). The permutation null is the simplest check for leakage.' },
    { t: 'Top 50, nested, λ <b>0.1</b> → <b>100</b>', s: `${vs(T.lo)} → ${vs(T.hi)}`, m: `${T.hi.ext > T.lo.ext ? 'A stronger penalty helps when many genes are noise: shrinkage reduces the variance of the weights.' : 'The penalty matters less than the choice of genes here.'} Ridge shrinks but keeps every selected gene.` },
    { t: '<b>New permutation</b> several times (top 50)', s: `Over permutations 1–4: leaky CV ${rngTxt(PR.leaky, 'cv')}, nested CV ${rngTxt(PR.nested, 'cv')}, Centre C ${rngTxt([...PR.leaky, ...PR.nested], 'ext')}`, m: `Each permutation is one draw from the null.${d3.min(PR.leaky, (r) => r.cv) > 0.5 ? ' The leaky estimate is above 0.5 in every draw.' : ''} Nested CV values somewhat below 0.5 are expected on null data: with labels fixed in total, each training fold is slightly enriched for the class its test fold lacks, which pushes predictions the wrong way.` },
    { t: 'Compare <b>mean predicted vs observed</b> in Centre C', s: `IFN-γ score ${f2(mean(T.ifng.pC))}; top 50 with λ 0.1: ${f2(mean(T.lo.pC))}; observed ${f2(mean(yC))}`, m: 'The lab uses uncorrected data. A weakly penalised model on many genes adds up many small per-gene technical offsets between centres, so its probabilities shift although its ranking (AUC) is unaffected; an averaged score is far less exposed (step 5).' },
  ], [['t', 'Try'], ['s', 'What you see'], ['m', 'What it means']], { cls: 'compact' });

  // =====================================================================================
  // 5. PITFALLS
  // =====================================================================================
  ch = chapter(main, 'pitfalls', 'Pitfalls', 'Common problems, how to recognise them, and how to address them.');
  const ci10 = EX.genes.indexOf('CXCL10');
  const reCoef = EX.genes.map((g, j) => ({ g, a: EX.lin.coef[j], b: EX.lin.coef_drop_cxcl10[j], r: EX.corr[ci10][j] })).filter((d) => d.b != null && d.r > 0.8);
  cards(ch, [
    { tag: 'Symptom: excellent CV AUC, poor external AUC', title: 'Selection outside the folds', body: `Genes chosen on all patients before CV. In step 3, random labels reach AUC ${f2(ck('perm', 'leaky', kMax).mean)} this way. <b>Fix:</b> put every outcome-dependent step inside the pipeline (scikit-learn <code>Pipeline</code>, tidymodels, mlr3) and run a permutation null.` },
    { tag: 'Symptom: the reported model is the best of many', title: 'Tuning on the reporting folds', body: `Reporting the best grid point or the best of several learners from the same CV is optimistic (step 4: ${optim[0]}). <b>Fix:</b> nested CV for the whole selection-and-tuning procedure; prespecify the primary model.` },
    { tag: 'Symptom: a different gene list on every rerun', title: 'Correlated genes', body: `The lasso picks one member of a correlated module almost at random (the most stable gene is selected in ${pc(st.freq[0])} of half-samples). <b>Fix:</b> report the module or a stability frequency, use the elastic net, or use a prespecified gene-set score.` },
    { tag: 'Symptom: "gene X is the key driver"', title: 'Reading coefficients as importance', body: `In an 8-gene logistic model, removing CXCL10 (weight ${f2(EX.lin.coef[ci10])}) moves its weight to correlated genes: ${reCoef.map((d) => `${d.g} ${f2(d.a)} → ${f2(d.b)}`).join(', ')}. <b>Fix:</b> treat weights as properties of the model, not of the biology; causal claims need experiments.` },
    { tag: 'Symptom: probabilities off in a new centre', title: 'Calibration drift', body: `In the lab, a weakly penalised 50-gene model ranks Centre C patients with AUC ${f2(T.lo.ext)}, but its mean predicted probability there is ${f2(mean(T.lo.pC))} against ${f2(mean(yC))} observed. <b>Fix:</b> report calibration alongside AUC, use bridging or reference samples, and treat recalibration as model updating that requires new validation.` },
    { tag: 'Symptom: the validation set is normalised as a batch', title: 'Cohort-wise preprocessing', body: 'Z-scoring, ComBat or quantile normalisation across the validation cohort uses the other validation patients, and cannot be applied to a single new patient. <b>Fix:</b> freeze the preprocessing parameters with the model.' },
    { tag: 'Symptom: wide confidence intervals', title: 'Too few events', body: `${nR} responders in discovery and ${sum(yC)} in Centre C. The external AUC interval spans ${f2(fr.ci[0])}–${f2(fr.ci[1])}, too wide to separate a good biomarker from a mediocre one. <b>Fix:</b> plan the validation sample size (100 or more events is a common target) and report intervals.` },
    { tag: 'Symptom: "predicts benefit from anti-PD-1"', title: 'Prognostic vs predictive', body: 'All patients here received the drug. A marker that separates responders from non-responders in a single arm may only indicate better prognosis. <b>Fix:</b> a predictive claim needs a treatment-by-marker interaction in a randomised or controlled comparison (next step).' },
  ]);

  // =====================================================================================
  // 6. RECAP
  // =====================================================================================
  ch = chapter(main, 'recap', 'Recap', 'Key points.');
  recap(ch, [
    'A biomarker classifier is a <b>whole pipeline</b>: gene selection, preprocessing, tuning and the model. Cross-validation must repeat all of it in every fold.',
    'Selection outside the folds is the classic leak: with thousands of genes, <b>random labels</b> reach a high CV AUC. A permutation null detects it.',
    'The <b>lasso</b> selects a sparse panel; the <b>1-SE rule</b> favours smaller panels; the <b>elastic net</b> keeps correlated genes together.',
    'Selected genes from a correlated module are <b>interchangeable</b>: report the module or its stability, not a single gene.',
    '<b>Nested CV</b> estimates the procedure. Tune and compare models inside it, on shared folds, and report the spread across repeats.',
    'A <b>prespecified score</b> is a strong baseline; here it matches or beats data-driven selection.',
    '<b>Lock</b> genes, preprocessing, weights and cut-off before validation. External AUC measures ranking; calibration shows whether the probabilities transfer.',
    'Discrimination depends on case mix; predictive (treatment-specific) claims need a control arm.',
  ]);
  prose(ch, '<h3>In code</h3><p>Every step that uses the data goes into one pipeline object, which the outer CV loop refits on each training fold.</p>');
  code(ch, {
    Python: `from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (GridSearchCV, StratifiedKFold, cross_val_predict,
                                     permutation_test_score)
from sklearn.metrics import roc_auc_score

# X: patients x genes (log-CPM), y: response (0/1)
pipe = make_pipeline(StandardScaler(),
                     LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5, max_iter=5000))
inner = GridSearchCV(pipe, {"logisticregression__C": [0.01, 0.03, 0.1, 0.3, 1]},
                     cv=StratifiedKFold(3, shuffle=True, random_state=0), scoring="roc_auc")
outer = StratifiedKFold(5, shuffle=True, random_state=1)
p_oof = cross_val_predict(inner, X, y, cv=outer, method="predict_proba")[:, 1]
print("nested CV AUC", roc_auc_score(y, p_oof))

# permutation null for the whole procedure (slow: refits everything per permutation)
score, perm_scores, pval = permutation_test_score(inner, X, y, cv=outer,
                                                  scoring="roc_auc", n_permutations=100)

# lock: refit once on all discovery patients, save, and apply unchanged to the validation cohort
final = inner.fit(X, y).best_estimator_
p_ext = final.predict_proba(X_ext)[:, 1]
print("external AUC", roc_auc_score(y_ext, p_ext))`,
    R: `library(glmnet); library(pROC)
# x: patients x genes matrix (log-CPM), y: response (0/1)
outer <- caret::createFolds(y, k = 5)
p_oof <- numeric(length(y))
for (te in outer) {
  # glmnet standardises internally; cv.glmnet's inner CV picks lambda on the training fold only
  fit <- cv.glmnet(x[-te, ], y[-te], family = "binomial", alpha = 0.5, type.measure = "auc", nfolds = 3)
  p_oof[te] <- predict(fit, x[te, ], s = "lambda.1se", type = "response")
}
auc(y, p_oof)                                     # nested CV estimate

final <- cv.glmnet(x, y, family = "binomial", alpha = 0.5, type.measure = "auc")
coef(final, s = "lambda.1se")                     # locked genes and weights
auc(y_ext, predict(final, x_ext, s = "lambda.1se", type = "response"))

# a full nested pipeline with tuning: tidymodels (rsample::nested_cv) or mlr3 (AutoTuner + resample)`,
  });
  seeAlso(main, [['omics', 'bulk-de', 'Previous: differential expression'], ['omics', 'bulk-survival', 'Next: survival analysis and risk scores'], ['concepts', 'cross-validation', 'Cross-validation and data leakage'], ['concepts', 'regularization', 'Regularisation (ridge and lasso)'], ['concepts', 'logistic-regression', 'Logistic regression'], ['concepts', 'metrics', 'Evaluating a classifier'], ['concepts', 'tree-ensembles', 'Random forests and gradient boosting']]);
}

// ---------------------------------------------------------------------------------------------
function unpack(b64, scale) {
  const raw = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0)), u16 = new Uint16Array(raw.buffer);
  const X = new Float32Array(u16.length); for (let k = 0; k < u16.length; k++) X[k] = u16[k] / scale;
  return X;
}
/** Stratified k folds of indices (deterministic). */
function stratFolds(y, k, seed) {
  const r = rng(seed), folds = Array.from({ length: k }, () => []);
  [0, 1].forEach((c) => r.shuffle(y.map((v, i) => i).filter((i) => y[i] === c)).forEach((i, j) => folds[j % k].push(i)));
  return folds;
}
/** Ridge-penalised logistic regression by Newton steps; intercept unpenalised. Returns [b0, ...w]. */
function ridgeLogistic(Z, y, lam) {
  const n = Z.length, k = Z[0].length + 1, b = new Array(k).fill(0);
  for (let it = 0; it < 30; it++) {
    const g = new Array(k).fill(0), H = Array.from({ length: k }, () => new Array(k).fill(0));
    for (let i = 0; i < n; i++) {
      const a = [1, ...Z[i]]; let z = 0; for (let j = 0; j < k; j++) z += a[j] * b[j];
      const q = sigmoid(z), w = Math.max(q * (1 - q), 1e-9), e = y[i] - q;
      for (let u = 0; u < k; u++) { g[u] += a[u] * e; const wa = w * a[u]; for (let v = u; v < k; v++) H[u][v] += wa * a[v]; }
    }
    for (let u = 0; u < k; u++) for (let v = 0; v < u; v++) H[u][v] = H[v][u];
    for (let u = 1; u < k; u++) { H[u][u] += lam + 1e-6; g[u] -= lam * b[u]; }
    const s = solve(H, g); let mx = 0;
    for (let u = 0; u < k; u++) { const d = Math.max(-3, Math.min(3, s[u])); b[u] += d; mx = Math.max(mx, Math.abs(d)); }
    if (mx < 1e-6) break;
  }
  return b;
}
const rocXY = (score, y) => roc(score, y).map((q) => [q.fpr, q.tpr]);
/** Equal-count groups by predicted probability. */
function qBins(y, p, k) {
  const o = d3.range(y.length).sort((a, b) => p[a] - p[b]);
  return d3.range(k).map((j) => { const ii = o.slice(Math.round(j * o.length / k), Math.round((j + 1) * o.length / k)); return { n: ii.length, mp: d3.mean(ii, (i) => p[i]), obs: d3.mean(ii, (i) => y[i]) }; });
}
function diag(F, d = [0, 1]) {
  F.layer('diag').append('line').attr('x1', F.x(d[0])).attr('y1', F.y(d[0])).attr('x2', F.x(d[1])).attr('y2', F.y(d[1])).attr('stroke', muted()).attr('stroke-dasharray', '4 4');
}
/** One panel of the fold diagram; returns one d3 group per CV round. */
function foldPanel(parent, scheme, n) {
  const w = el('div'); parent.append(w);
  const W = 460, H = 262, x0 = 56, cw = (W - x0 - 8) / n;
  const svg = d3.select(w).append('svg').attr('viewBox', `0 0 ${W} ${H}`).attr('width', W).attr('role', 'img');
  svg.append('text').attr('x', 0).attr('y', 14).attr('font-weight', 600).attr('font-size', 13).attr('fill', ink()).text(scheme === 'leaky' ? 'Leaky: selection before CV' : 'Nested: selection inside each fold');
  return d3.range(5).map((f) => {
    const y = 30 + f * 46, lo = Math.round(f * n / 5), hi = Math.round((f + 1) * n / 5), g = svg.append('g');
    g.append('text').attr('x', 0).attr('y', y + 28).attr('font-size', 11).attr('fill', ink2()).text(`round ${f + 1}`);
    g.selectAll('rect.c').data(d3.range(n)).join('rect').attr('class', 'c').attr('x', (i) => x0 + i * cw).attr('y', y + 16).attr('width', Math.max(1, cw - 0.4)).attr('height', 18)
      .attr('fill', (i) => (i >= lo && i < hi ? SCOL[scheme]() : muted())).attr('opacity', 0.9);
    const seg = (a, b) => g.append('rect').attr('x', x0 + a * cw).attr('y', y + 6).attr('width', (b - a) * cw).attr('height', 6).attr('rx', 3).attr('fill', ink());
    if (scheme === 'leaky') seg(0, n); else { if (lo > 0) seg(0, lo); if (hi < n) seg(hi, n); }
    return g;
  });
}
/** Row of boxes with arrows: the analysis on this page. */
function pipelineSvg(boxes) {
  const wrap = el('div', { class: 'figure' });
  wrap.append(el('div', { class: 'fig-title' }, 'The analysis on this page'));
  const BW = 156, gap = 20, H = 112, W = boxes.length * BW + (boxes.length - 1) * gap + 4;
  const svg = d3.select(wrap).append('svg').attr('viewBox', `0 0 ${W} ${H + 8}`).attr('width', W).attr('role', 'img');
  boxes.forEach((b, i) => {
    const x0 = 2 + i * (BW + gap), g = svg.append('g').attr('transform', `translate(${x0},4)`);
    g.append('rect').attr('width', BW).attr('height', H).attr('rx', 9).attr('fill', css('--surface-2')).attr('stroke', i === 3 || i === 5 ? css('--accent') : css('--axis'));
    g.append('text').attr('x', 10).attr('y', 22).attr('font-weight', 700).attr('font-size', 12.5).attr('fill', ink()).text(b.t);
    b.l.forEach((t, j) => g.append('text').attr('x', 10).attr('y', 46 + j * 20).attr('font-size', 11.5).attr('fill', ink2()).text(t));
    if (i < boxes.length - 1) {
      const ax = x0 + BW + 3, ay = 4 + H / 2;
      svg.append('path').attr('d', `M${ax},${ay} L${ax + gap - 6},${ay} M${ax + gap - 11},${ay - 5} L${ax + gap - 6},${ay} L${ax + gap - 11},${ay + 5}`).attr('fill', 'none').attr('stroke', ink2()).attr('stroke-width', 1.8);
    }
  });
  return wrap;
}
