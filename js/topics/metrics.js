// Evaluating a classifier, lesson format.
// Arc: what → why → where → how it works (6 steps: confusion matrix → ROC and AUC → precision–recall → prevalence
//      → calibration → model comparison lab with try/see/means table) → pitfalls → recap.
// All predictions are those of logistic models fitted on the discovery patients and applied to Centre C (validation).
// Every number in the prose is computed from data/study/cohort.json at render time.
import { header, figure, slider, select, button, stats, legend, seeAlso, animate, tip, untip, el, html } from '../lib/ui.js';
import { frame, cat, css, ink, ink2, muted } from '../lib/plot.js';
import { sigmoid, mean, rng } from '../lib/data.js';
import { fitLogistic, predictProb, logLoss, auc, roc, confusion, standardize } from '../lib/models.js';
import { loadCohort, FEATURES, studyCard } from '../lib/study.js';
import { lessonNav, chapter, prose, rung, worked, quiz, cards, table, code, recap } from '../lib/lesson.js';
const d3 = window.d3;

const um = (t) => t.replace(/^-/, '−');
const f2 = (v) => (Number.isFinite(v) ? um(v.toFixed(2)) : '—');
const f3 = (v) => (Number.isFinite(v) ? um(v.toFixed(3)) : '—');
const pc = (v) => (Number.isFinite(v) ? Math.round(100 * v) + '%' : '—');
const pc1 = (v) => (100 * v).toFixed(1) + '%';
const sum = (a) => a.reduce((s, v) => s + v, 0);
const logit = (p) => Math.log(p / (1 - p));
const clip = (p) => Math.min(1 - 1e-9, Math.max(1e-9, p));
const brier = (p, y) => mean(p.map((q, i) => (q - y[i]) ** 2));

export default async function render(main) {
  const C = await loadCohort();
  const D = C.discovery, V = C.validation;
  const y = D.map((p) => p.response), yv = V.map((p) => p.response);
  const rowsOf = (pts, keys) => pts.map((p) => keys.map((k) => FEATURES[k].get(p)));
  const F3 = ['ifng', 'log2tmb', 'caf'];
  const LAB = ['ifng', 'log2tmb', 'caf', 'pdl1', 'cd8', 'prolif', 'age', 'smoker', 'male', 'centreB'];

  // ---- models fitted on discovery patients, scored on discovery and on Centre C
  const MODELS = {
    three: { label: 'IFN-γ + TMB + CAF (the true three)', keys: F3 },
    ifng: { label: 'IFN-γ only', keys: ['ifng'] },
    pdl1: { label: 'PD-L1 only', keys: ['pdl1'] },
    centre: { label: 'Centre only', keys: ['centreB'] },
    all: { label: 'All 10 features', keys: LAB },
    over: { label: 'All 10 features, fitted on only 30 patients', keys: LAB, n: 30 },
  };
  Object.values(MODELS).forEach((m) => {
    const tr = D.slice(0, m.n || D.length), ytr = tr.map((p) => p.response);
    const st = standardize(rowsOf(tr, m.keys)), fit = fitLogistic(st.Z, ytr);
    m.pt = st.Z.map((r) => predictProb(r, fit.b0, fit.w));
    m.pv = V.map((p) => predictProb(st.apply(m.keys.map((k) => FEATURES[k].get(p))), fit.b0, fit.w));
    m.aucT = auc(m.pt, ytr); m.aucV = auc(m.pv, yv); m.roc = roc(m.pv, yv); m.brier = brier(m.pv, yv); m.ll = logLoss(m.pv, yv);
  });
  const M3 = MODELS.three, pv = M3.pv, rocV = M3.roc;
  const nV = V.length, nPos = sum(yv), nNeg = nV - nPos, piV = nPos / nV;
  const piD = mean(y);
  const cm = (thr, s = pv) => confusion(s, yv, thr);
  const c50 = cm(0.5);
  const ppvAt = (sens, spec, pi) => (sens * pi) / Math.max(1e-12, sens * pi + (1 - spec) * (1 - pi));
  const npvAt = (sens, spec, pi) => (spec * (1 - pi)) / Math.max(1e-12, spec * (1 - pi) + (1 - sens) * pi);
  const prAt = (pts, pi) => pts.slice(1).filter((q) => q.tpr > 0).map((q) => ({ r: q.tpr, p: (q.tpr * pi) / (q.tpr * pi + q.fpr * (1 - pi)), thr: q.thr, fpr: q.fpr }));
  const apOf = (pr) => { let a = 0, r0 = 0; pr.forEach((q) => { a += (q.r - r0) * q.p; r0 = q.r; }); return a; };
  const hm = (A, P, N) => { const q1 = A / (2 - A), q2 = 2 * A * A / (1 + A); return Math.sqrt((A * (1 - A) + (P - 1) * (q1 - A * A) + (N - 1) * (q2 - A * A)) / (P * N)); };
  const aucV = M3.aucV, aucSE = hm(aucV, nPos, nNeg);

  header(main, 'Evaluating a classifier',
    'A classifier returns a score for each sample. Evaluation asks <b>how well the scores separate the two classes</b>, <b>what a chosen cut-off costs and gains</b>, and <b>whether the probabilities can be taken at face value</b>. Each question has its own measure: confusion matrix, ROC and precision–recall curves, calibration.');
  lessonNav(main);
  studyCard(main, C, '<b>Question for this page:</b> a logistic model fitted on Centres A and B predicts anti-PD-1 response. How should its predictions on the independent Centre C patients be judged, and which measures change when the proportion of responders changes?');

  // =====================================================================================
  // 1. WHAT
  // =====================================================================================
  let ch = chapter(main, 'what', 'What is classifier evaluation?', 'In one line: <b>compare predictions with known outcomes, on patients the model has not seen</b>.');
  prose(ch, `<p>A binary classifier produces, for each sample, a <b>score</b> (here, the predicted probability of response). A <b>threshold</b> turns the score into a call, positive or negative. Comparing calls with the true outcomes gives four counts: <b>true positives</b> (TP), <b>false positives</b> (FP), <b>true negatives</b> (TN) and <b>false negatives</b> (FN), arranged in a 2 × 2 <b>confusion matrix</b>. Every threshold-based measure is a ratio of these counts.</p>
    <p>The page uses the logistic model from the <i>Logistic regression</i> lesson (IFN-γ score, log2 TMB, CAF score), fitted on the ${D.length} discovery patients and scored on the ${nV} Centre C patients, of whom ${nPos} (${pc(piV)}) responded. Three families of measure answer three different questions:</p>`);
  table(ch, [
    { f: 'Threshold-based', q: 'What happens at one chosen cut-off?', m: 'Sensitivity, specificity, PPV, NPV, accuracy, F1', s: `at 0.5: sensitivity ${pc(c50.sens)}, specificity ${pc(c50.spec)}, PPV ${pc(c50.ppv)}` },
    { f: 'Ranking (threshold-free)', q: 'Do responders receive higher scores than non-responders?', m: 'ROC curve and AUC; precision–recall curve and average precision', s: `AUC ${f3(aucV)}; average precision ${f3(apOf(prAt(rocV, piV)))}` },
    { f: 'Probability quality', q: 'Can a predicted 30% be read as a 30% chance?', m: 'Calibration plot, Brier score, log-loss', s: `Brier ${f3(M3.brier)}; mean prediction ${pc(mean(pv))} vs observed ${pc(piV)}` },
  ], [['f', 'Family'], ['q', 'Question'], ['m', 'Main measures'], ['s', 'Value on Centre C']]);
  prose(ch, '<p>Key terms, with the values at threshold 0.5:</p>');
  table(ch, [
    { t: 'Sensitivity (recall, TPR)', d: 'TP / (TP + FN): the share of true responders called positive', v: `${c50.tp} / ${c50.tp + c50.fn} = ${pc(c50.sens)}` },
    { t: 'Specificity (TNR)', d: 'TN / (TN + FP): the share of true non-responders called negative', v: `${c50.tn} / ${c50.tn + c50.fp} = ${pc(c50.spec)}` },
    { t: 'PPV (precision)', d: 'TP / (TP + FP): the share of positive calls that are correct', v: `${c50.tp} / ${c50.tp + c50.fp} = ${pc(c50.ppv)}` },
    { t: 'NPV', d: 'TN / (TN + FN): the share of negative calls that are correct', v: `${c50.tn} / ${c50.tn + c50.fn} = ${pc(c50.tn / (c50.tn + c50.fn))}` },
    { t: 'Accuracy', d: '(TP + TN) / all: the share of all calls that are correct', v: `${c50.tp + c50.tn} / ${nV} = ${pc(c50.acc)}` },
    { t: 'Prevalence', d: 'The share of positives in the population being tested', v: `${nPos} / ${nV} = ${pc(piV)} (discovery: ${pc(piD)})` },
    { t: 'ROC curve / AUC', d: 'Sensitivity against 1 − specificity over every threshold; AUC is the area under it', v: `AUC = ${f3(aucV)}` },
    { t: 'Calibration', d: 'Agreement between predicted probabilities and observed frequencies', v: `Brier score = ${f3(M3.brier)}` },
  ], [['t', 'Term'], ['d', 'Definition'], ['v', 'On Centre C']], { cls: 'compact' });

  // =====================================================================================
  // 2. WHY
  // =====================================================================================
  ch = chapter(main, 'why', 'Why do we need these measures?', 'A single number such as accuracy hides which kind of error a model makes, and it depends on how common the positive class is.');
  prose(ch, `<ol><li><b>Errors have different costs.</b> Treating a non-responder exposes the patient to toxicity; missing a responder withholds benefit. The two error types need separate rates (sensitivity, specificity).</li>
    <li><b>The cut-off is a choice, not a property of the model.</b> Ranking measures (AUC) describe the model across all cut-offs; confusion-matrix measures describe one.</li>
    <li><b>Prevalence changes what a positive call means.</b> Sensitivity and specificity belong to the test; PPV and NPV also depend on how common the condition is in the tested group.</li>
    <li><b>Probabilities are used directly</b> (expected benefit, risk communication), so their numerical accuracy needs checking, which neither AUC nor accuracy does.</li></ol>
    <p>The figure compares three simple classifiers on Centre C. Accuracy alone does not separate them as prevalence falls.</p>`);
  {
    const f = figure(ch, 'Accuracy under different prevalence of responders', 'Three classifiers evaluated on Centre C: "nobody responds" (always negative), "everybody responds" (always positive) and the model at threshold 0.5. Sensitivity and specificity are fixed properties of each classifier. For prevalence other than the observed value, accuracy and PPV are recomputed from them (as in step 4). Hover a bar for the value.');
    const PV = [['obs', `Observed (${pc(piV)})`], ['0.1', '10%'], ['0.02', '2%']];
    let pk = 'obs';
    select(f.controls, { label: 'Prevalence of responders', value: pk, options: PV, onChange: (v) => { pk = v; paintW(); } });
    const W = frame(f.body, { width: 820, height: 320, x: [0, 4], y: [0, 1.05], xLabel: '', yLabel: 'value', xScale: d3.scaleLinear(), grid: true });
    W.g.select('.axis.x').remove();
    const CL = [{ n: 'nobody responds', sens: 0, spec: 1 }, { n: 'everybody responds', sens: 1, spec: 0 }, { n: 'model at 0.5', sens: c50.sens, spec: c50.spec }];
    const MT = ['accuracy', 'sensitivity', 'specificity', 'PPV'];
    const lay = W.layer('b'), x0 = d3.scaleBand().domain(MT).range([0, W.w]).paddingInner(0.18), x1 = d3.scaleBand().domain(CL.map((c) => c.n)).range([0, x0.bandwidth()]).padding(0.08);
    MT.forEach((m) => W.g.append('text').attr('x', x0(m) + x0.bandwidth() / 2).attr('y', W.h + 20).attr('text-anchor', 'middle').attr('font-size', 13).attr('fill', ink2()).text(m));
    legend(f.body, CL.map((c, i) => [c.n, cat(i === 2 ? 6 : i === 0 ? 0 : 1)]));
    const note = html('', 'p', 'fig-caption'); f.body.append(note);
    function paintW() {
      const pi = pk === 'obs' ? piV : +pk, bars = [];
      CL.forEach((c, i) => {
        const v = { accuracy: c.sens * pi + c.spec * (1 - pi), sensitivity: c.sens, specificity: c.spec, PPV: c.sens * pi + (1 - c.spec) * (1 - pi) > 0 && c.sens > 0 ? ppvAt(c.sens, c.spec, pi) : NaN };
        MT.forEach((m) => bars.push({ c, i, m, v: v[m] }));
      });
      lay.selectAll('g').remove();
      const g = lay.selectAll('g').data(bars).join('g');
      g.append('rect').attr('x', (d) => x0(d.m) + x1(d.c.n)).attr('width', x1.bandwidth()).attr('y', (d) => W.y(Number.isFinite(d.v) ? d.v : 0)).attr('height', (d) => W.h - W.y(Number.isFinite(d.v) ? d.v : 0))
        .attr('fill', (d) => cat(d.i === 2 ? 6 : d.i === 0 ? 0 : 1)).attr('fill-opacity', 0.85)
        .on('mousemove', (e, d) => tip(e, `${d.c.n}<br>${d.m}: ${Number.isFinite(d.v) ? pc1(d.v) : 'undefined (no positive calls)'}`)).on('mouseleave', untip);
      g.append('text').attr('x', (d) => x0(d.m) + x1(d.c.n) + x1.bandwidth() / 2).attr('y', (d) => W.y(Number.isFinite(d.v) ? d.v : 0) - 4).attr('text-anchor', 'middle').attr('font-size', 11.5).attr('fill', ink())
        .text((d) => (Number.isFinite(d.v) ? Math.round(100 * d.v) : 'n/a'));
      const am = CL[2].sens * pi + CL[2].spec * (1 - pi);
      note.innerHTML = `At prevalence ${pc1(pi)}, "nobody responds" has accuracy ${pc1(1 - pi)}, against ${pc1(am)} for the model; the model's PPV is ${pc1(ppvAt(CL[2].sens, CL[2].spec, pi))}.`;
    }
    paintW();
  }
  prose(ch, `<p>On Centre C the responder rate is ${pc(piV)}, so "nobody responds" reaches ${pc(1 - piV)} accuracy, below the model's ${pc(c50.acc)}. When responders are rare, the trivial classifier reaches the higher accuracy while having zero sensitivity. This is the <b>accuracy paradox</b>, and it is why sensitivity, specificity and PPV are reported separately.</p>`);

  // =====================================================================================
  // 3. WHERE
  // =====================================================================================
  ch = chapter(main, 'where', 'Where is it used?', 'Any setting in which a model assigns samples to classes, from clinical tests to genomic pipelines.');
  cards(ch, [
    { tag: 'Diagnostics', title: 'Screening and detection tests', body: 'Blood-based cancer detection, newborn screening and pathogen assays report sensitivity and specificity, and PPV at the intended prevalence. Screening populations have low prevalence, which dominates PPV.' },
    { tag: 'Variant calling', title: 'Precision and recall of calls', body: 'Callers are benchmarked against truth sets (e.g. GIAB) with precision, recall and F1 per variant class. True negatives are effectively unbounded, so specificity and ROC are rarely used.' },
    { tag: 'Single-cell', title: 'Cell-type annotation and doublets', body: 'Annotation tools and doublet detectors are compared by per-class precision and recall; rare cell types need precision–recall rather than overall accuracy.' },
    { tag: 'Genomics', title: 'Regulatory element and variant-effect prediction', body: 'Enhancer, splice-site and pathogenicity predictors are tested on data where positives are a small minority; auPRC is the usual summary alongside auROC.' },
    { tag: 'Clinical prediction', title: 'Biomarker and risk models', body: 'Response to therapy, readmission or progression models are reported with AUC, calibration plots and Brier score; guidelines such as TRIPOD ask for both.' },
    { tag: 'Risk scores', title: 'Polygenic risk scores', body: 'Evaluated by AUC and by absolute-risk calibration across ancestry groups, where a score can rank well and still be miscalibrated in a new population.' },
    { tag: 'Quality control', title: 'Pass/fail filters', body: 'Sample, cell or variant filters are tuned on the trade-off between discarding good data (false positives) and keeping bad data (false negatives).' },
    { tag: 'Machine-learning benchmarks', title: 'Comparing methods', body: 'Papers comparing classifiers report AUC and auPRC on held-out data; the confidence interval and the dataset prevalence determine whether the difference is meaningful.' },
  ]);
  prose(ch, '<p><b>When not to use these measures:</b> the output is a number (use RMSE, R²) or a time-to-event (use the concordance index and time-dependent calibration); there are three or more classes (use per-class precision/recall with macro or micro averaging, or a multi-class confusion matrix); the task is ranking a long list where only the top matters (use precision at k or enrichment); or the labels themselves are noisy (the measures then estimate agreement with the labels, not with the biology).</p>');

  // =====================================================================================
  // 4. HOW IT WORKS
  // =====================================================================================
  ch = chapter(main, 'how', 'How it works', 'Six steps: the confusion matrix, ROC and AUC, precision–recall, prevalence, calibration, and a comparison of models on Centre C.');

  // ---- Step 1: confusion matrix and threshold
  let rb = rung(ch, 1, 'Counting: the confusion matrix and the threshold',
    `<p>The model gives each of the ${nV} Centre C patients a probability. Choosing a threshold (here 0.5) splits them into "predicted responder" and "predicted non-responder". Cross-tabulating with what happened gives four counts. Moving the threshold moves patients between columns, and trades one error type for the other.</p>`);
  {
    const f = figure(rb, 'Threshold and confusion matrix', 'Left: predicted probability of every Centre C patient, split by actual outcome; patients to the right of the line are called positive. Right: the resulting confusion matrix (the solid-coloured cells are correct calls). Drag the threshold.');
    let thr = 0.5;
    slider(f.controls, { label: 'threshold', min: 0.02, max: 0.98, step: 0.01, value: thr, fmt: (v) => v.toFixed(2), onInput: (v) => { thr = v; paint(); } });
    const st = stats(f.root);
    const row = el('div', { class: 'fig-row' }); f.body.append(row);
    const a = el('div'), b = el('div'); row.append(a, b);
    const S = frame(a, { width: 500, height: 330, x: [0, 1], y: [-0.6, 1.6], xLabel: 'predicted P(response)', grid: false, axes: false });
    S.g.append('g').attr('class', 'axis x').attr('transform', `translate(0,${S.h})`).call(d3.axisBottom(S.x).ticks(5));
    S.layer('lab').selectAll('text').data([[0, 'no response'], [1, 'responded']]).join('text').attr('x', 4).attr('y', (d) => S.y(d[0] + 0.45)).attr('font-size', 12).attr('fill', ink2()).text((d) => d[1]);
    const jit = (i) => (((i * 7919) % 101) / 101 - 0.5) * 0.34;
    S.layer('pts').selectAll('circle').data(V).join('circle').attr('cx', (d, i) => S.x(pv[i])).attr('cy', (d, i) => S.y(d.response + jit(i))).attr('r', 4.2)
      .attr('fill', (d) => cat(d.response)).attr('fill-opacity', 0.8).attr('stroke', css('--surface')).attr('stroke-width', 0.8)
      .on('mousemove', (e, d) => tip(e, `${d.id} (${d.phenotype})<br>predicted ${f2(pv[V.indexOf(d)])} · ${d.response ? 'responded' : 'no response'}`)).on('mouseleave', untip);
    const tl = S.layer('thr');
    const Mx = d3.select(b).append('svg').attr('viewBox', '0 0 430 330').attr('width', 430).attr('role', 'img').attr('aria-label', 'confusion matrix');
    function paint() {
      const c = cm(thr);
      tl.selectAll('*').remove();
      tl.append('line').attr('x1', S.x(thr)).attr('x2', S.x(thr)).attr('y1', 0).attr('y2', S.h).attr('stroke', ink()).attr('stroke-width', 2);
      tl.append('text').attr('x', S.x(thr) + 6).attr('y', 12).attr('font-size', 11.5).attr('fill', ink()).text('called positive →');
      tl.append('text').attr('x', S.x(thr) - 6).attr('y', 12).attr('text-anchor', 'end').attr('font-size', 11.5).attr('fill', ink()).text('← called negative');
      Mx.selectAll('*').remove();
      const cells = [[0, 0, 'TP', c.tp, 1, true, 'true positive: responder called positive'], [0, 1, 'FN', c.fn, 1, false, 'false negative: responder missed'], [1, 0, 'FP', c.fp, 0, false, 'false positive: non-responder called positive'], [1, 1, 'TN', c.tn, 0, true, 'true negative: non-responder called negative']];
      const x0 = 110, y0 = 58, cw = 150, chh = 100;
      Mx.append('text').attr('x', x0 + cw).attr('y', 18).attr('text-anchor', 'middle').attr('font-size', 12.5).attr('font-weight', 700).attr('fill', ink2()).text('Predicted');
      ['positive', 'negative'].forEach((t, j) => Mx.append('text').attr('x', x0 + cw * j + cw / 2).attr('y', 44).attr('text-anchor', 'middle').attr('font-size', 12).attr('fill', ink2()).text(t));
      Mx.append('text').attr('transform', `translate(16,${y0 + chh}) rotate(-90)`).attr('text-anchor', 'middle').attr('font-size', 12.5).attr('font-weight', 700).attr('fill', ink2()).text('Actual');
      ['responded', 'no response'].forEach((t, i) => Mx.append('text').attr('x', x0 - 8).attr('y', y0 + chh * i + chh / 2 + 4).attr('text-anchor', 'end').attr('font-size', 12).attr('fill', ink2()).text(t));
      cells.forEach(([i, j, nm, v, cls, ok, desc]) => {
        const g = Mx.append('g').attr('transform', `translate(${x0 + cw * j},${y0 + chh * i})`);
        g.append('rect').attr('width', cw - 4).attr('height', chh - 4).attr('rx', 8).attr('fill', cat(cls)).attr('fill-opacity', ok ? 0.55 : 0.2).attr('stroke', cat(cls)).attr('stroke-width', ok ? 2 : 1)
          .on('mousemove', (e) => tip(e, `${nm}: ${desc}<br>${v} patients`)).on('mouseleave', untip);
        g.append('text').attr('x', (cw - 4) / 2).attr('y', 48).attr('text-anchor', 'middle').attr('font-size', 30).attr('font-weight', 700).attr('fill', ink()).text(v);
        g.append('text').attr('x', (cw - 4) / 2).attr('y', 74).attr('text-anchor', 'middle').attr('font-size', 12.5).attr('fill', ink2()).text(nm);
      });
      Mx.append('text').attr('x', x0).attr('y', y0 + 2 * chh + 24).attr('font-size', 12).attr('fill', muted()).text(`called positive: ${c.tp + c.fp}   ·   called negative: ${c.tn + c.fn}`);
      st.set({ 'sensitivity TP/(TP+FN)': pc(c.sens), 'specificity TN/(TN+FP)': pc(c.spec), 'PPV TP/(TP+FP)': c.tp + c.fp ? pc(c.ppv) : 'n/a', 'NPV TN/(TN+FN)': c.tn + c.fn ? pc(c.tn / (c.tn + c.fn)) : 'n/a', accuracy: pc(c.acc) });
    }
    paint();
  }
  const c30 = cm(0.3), c70 = cm(0.7);
  worked(rb, `Reading the matrix at threshold 0.5 (Centre C, n = ${nV})`,
    `<p>${c50.tp + c50.fp} patients are called positive; ${c50.tp} of them responded. ${c50.tp + c50.fn} patients actually responded; the model found ${c50.tp} of them.</p>
     <ul><li><b>Sensitivity</b> = ${c50.tp} / (${c50.tp} + ${c50.fn}) = <b>${pc(c50.sens)}</b>: of the responders, how many were found.</li>
     <li><b>Specificity</b> = ${c50.tn} / (${c50.tn} + ${c50.fp}) = <b>${pc(c50.spec)}</b>: of the non-responders, how many were correctly left out.</li>
     <li><b>PPV</b> = ${c50.tp} / (${c50.tp} + ${c50.fp}) = <b>${pc(c50.ppv)}</b>: of those called positive, how many responded.</li>
     <li><b>NPV</b> = ${c50.tn} / (${c50.tn} + ${c50.fn}) = <b>${pc(c50.tn / (c50.tn + c50.fn))}</b>: of those called negative, how many did not respond.</li></ul>
     <p>Lowering the threshold to 0.3 gives sensitivity ${pc(c30.sens)} and specificity ${pc(c30.spec)}; raising it to 0.7 gives ${pc(c70.sens)} and ${pc(c70.spec)}. Sensitivity and specificity always move in opposite directions as the threshold moves.</p>`);
  quiz(rb, `Why can sensitivity and specificity be reported without knowing the proportion of responders, while PPV cannot?`,
    `<p>Sensitivity uses only the responders (the first row of the matrix) and specificity only the non-responders (the second row), so each is unaffected by how many patients are in the other row. PPV and NPV mix the two rows within a column, so they change when the row totals change (step 4).</p>`);

  // ---- Step 2: ROC and AUC
  const ap = apOf(prAt(rocV, piV));
  rb = rung(ch, 2, 'Sweeping the threshold: the ROC curve and AUC',
    `<p>Step 1 gave one (sensitivity, specificity) pair per threshold. The <b>receiver operating characteristic (ROC) curve</b> plots all of them: sensitivity (true-positive rate) on the vertical axis against 1 − specificity (false-positive rate) on the horizontal axis. A random score follows the diagonal; a perfect score passes through the top-left corner.</p>
     <p>The <b>area under the curve (AUC)</b> has a direct meaning: it is the probability that a randomly chosen responder has a higher score than a randomly chosen non-responder (ties count one half). On Centre C there are ${nPos} × ${nNeg} = ${nPos * nNeg} such pairs, and AUC = ${f3(aucV)}.</p>`);
  {
    const f = figure(rb, 'ROC curve of the model on Centre C', 'Each step along the curve is one patient: up for a responder, right for a non-responder, in descending order of score. The marker shows the operating point of the chosen threshold. The button draws random responder/non-responder pairs and tracks how often the responder has the higher score.');
    let thr = 0.5, stop = null;
    slider(f.controls, { label: 'threshold', min: 0.02, max: 0.98, step: 0.01, value: thr, fmt: (v) => v.toFixed(2), onInput: (v) => { thr = v; paint(); } });
    button(f.controls, 'Draw random pairs', () => run(), true);
    const st = stats(f.root), pairTxt = html('', 'p', 'fig-caption');
    const R = frame(f.body, { width: 520, height: 420, x: [0, 1], y: [0, 1], xLabel: 'false-positive rate = 1 − specificity', yLabel: 'sensitivity (TPR)' });
    drawROC(R, rocV, aucV);
    const opL = R.layer('op');
    f.body.append(pairTxt);
    legend(f.body, [['model', cat(6)], ['random score (AUC 0.5)', muted()], ['operating point at the threshold', ink()]]);
    function paint() {
      const c = cm(thr);
      opL.selectAll('*').remove();
      opL.append('circle').attr('cx', R.x(1 - c.spec)).attr('cy', R.y(c.sens)).attr('r', 7).attr('fill', ink()).attr('stroke', css('--surface')).attr('stroke-width', 2);
      opL.append('text').attr('x', R.x(1 - c.spec) + 12).attr('y', R.y(c.sens) + 18).attr('font-size', 12).attr('fill', ink()).text(`threshold ${thr.toFixed(2)}`);
      const rs = pairState; st.set({ 'sensitivity': pc(c.sens), 'specificity': pc(c.spec), 'AUC': f3(aucV), 'pairs drawn': rs.n, 'responder scored higher': rs.n ? pc(rs.win / rs.n) : '—' });
    }
    const pairState = { n: 0, win: 0 };
    const pos = V.map((p, i) => i).filter((i) => yv[i]), neg = V.map((p, i) => i).filter((i) => !yv[i]), rr = rng(11);
    function run() {
      if (stop) stop(); pairState.n = 0; pairState.win = 0;
      stop = animate(() => {
        for (let k = 0; k < 4; k++) {
          const i = pos[Math.floor(rr() * pos.length)], j = neg[Math.floor(rr() * neg.length)];
          pairState.n++; pairState.win += pv[i] > pv[j] ? 1 : pv[i] === pv[j] ? 0.5 : 0;
          pairTxt.innerHTML = `Last pair: ${V[i].id} (responded, ${f2(pv[i])}) vs ${V[j].id} (no response, ${f2(pv[j])}) → ${pv[i] > pv[j] ? 'ranked correctly' : 'ranked incorrectly'}`;
        }
        paint();
        return pairState.n < 800;
      });
    }
    paint();
  }
  prose(rb, `<p>With ${nV} patients the AUC is imprecise: the Hanley–McNeil standard error is ${f3(aucSE)}, giving an approximate 95% interval of ${f2(aucV - 1.96 * aucSE)}–${f2(Math.min(1, aucV + 1.96 * aucSE))}. The drawn pairs fluctuate around the AUC and approach it as their number grows. AUC does not depend on prevalence, and it is unchanged by any monotone transformation of the scores, including the recalibration in step 5.</p>`);

  // ---- Step 3: PR
  rb = rung(ch, 3, 'Precision–recall curve',
    `<p>The ROC curve uses false-positive <i>rate</i>, which divides by all non-responders (${nNeg}). When negatives vastly outnumber positives, many false positives barely move that rate. The <b>precision–recall (PR) curve</b> instead plots precision (PPV) against recall (sensitivity), so every false positive directly lowers precision. A random classifier gives a horizontal line at the prevalence (${pc(piV)}), not at 0.5. The summary is <b>average precision (AP)</b>, the area under the PR curve; AP = ${f3(ap)} here.</p>`);
  {
    const f = figure(rb, 'Precision–recall curve of the model on Centre C', 'Precision falls as the threshold is lowered and recall rises. The dashed line is the precision of a random classifier (the prevalence). The marker shows the operating point of the chosen threshold; hover the curve for thresholds.');
    let thr = 0.5;
    slider(f.controls, { label: 'threshold', min: 0.02, max: 0.98, step: 0.01, value: thr, fmt: (v) => v.toFixed(2), onInput: (v) => { thr = v; paint(); } });
    const st = stats(f.root);
    const P = frame(f.body, { width: 560, height: 400, x: [0, 1], y: [0, 1], xLabel: 'recall = sensitivity', yLabel: 'precision = PPV' });
    const pr = prAt(rocV, piV);
    drawPR(P, pr, piV, ap);
    const opL = P.layer('op');
    legend(f.body, [['model', cat(6)], ['random classifier (prevalence)', muted()], ['operating point at the threshold', ink()]]);
    function paint() {
      const c = cm(thr), f1 = 2 * c.ppv * c.sens / Math.max(1e-12, c.ppv + c.sens);
      opL.selectAll('*').remove();
      if (c.tp + c.fp) {
        opL.append('circle').attr('cx', P.x(c.sens)).attr('cy', P.y(c.ppv)).attr('r', 7).attr('fill', ink()).attr('stroke', css('--surface')).attr('stroke-width', 2);
        opL.append('text').attr('x', P.x(c.sens) + 12).attr('y', P.y(c.ppv) - 10).attr('font-size', 12).attr('fill', ink()).text(`threshold ${thr.toFixed(2)}`);
      }
      st.set({ precision: c.tp + c.fp ? pc(c.ppv) : 'n/a', recall: pc(c.sens), 'F1 = 2PR/(P+R)': c.tp + c.fp ? f2(f1) : 'n/a', 'average precision': f3(ap), 'baseline precision': pc(piV) });
    }
    paint();
  }
  prose(rb, `<p>The curve starts at the left with the few highest-scoring patients. Its first points can be noisy: a single false positive among the top-ranked patients lowers precision sharply when few patients have been called. <b>F1</b>, the harmonic mean of precision and recall, summarises one operating point; AP summarises the whole curve. Neither uses true negatives.</p>`);

  // ---- Step 4: prevalence
  const secs = c50.sens, specs = c50.spec;
  const scr = 10000 * 0.01, scrTP = scr * secs, scrFP = (10000 - scr) * (1 - specs);
  rb = rung(ch, 4, 'Imbalance and prevalence',
    `<p>Sensitivity and specificity describe how the model treats responders and non-responders separately, so they carry over to a population with a different mix. PPV does not. By Bayes' rule, for prevalence π:</p>
     <div class="calc">PPV = sens × π / ( sens × π + (1 − spec) × (1 − π) )</div>
     <p>The model's operating point at 0.5 has sensitivity ${pc(secs)} and specificity ${pc(specs)}. Applied to a population of 10,000 with 1% responders, it finds ${Math.round(scrTP)} of ${scr} responders and also flags ${Math.round(scrFP)} non-responders, so PPV is ${pc1(scrTP / (scrTP + scrFP))}: about ${(scrFP / scrTP).toFixed(1)} false positives for every true positive, with the same ROC curve.</p>`);
  {
    const f = figure(rb, 'Same model, different prevalence', 'The Centre C scores are reweighted so that responders make up the chosen share of the population; sensitivity and specificity at each threshold are unchanged. Left: the ROC curve does not change. Right: the PR curve and its baseline move with prevalence.');
    const PREV = [...new Set([0.005, 0.01, 0.02, 0.05, 0.1, 0.2, piV, 0.5, 0.7])].sort((a, b) => a - b);
    let pi = piV, thr = 0.5;
    slider(f.controls, { label: 'prevalence', min: 0, max: PREV.length - 1, step: 1, value: PREV.indexOf(piV), fmt: (i) => pc1(PREV[i]) + (PREV[i] === piV ? ' (observed)' : ''), onInput: (i) => { pi = PREV[i]; paint(); } });
    slider(f.controls, { label: 'threshold', min: 0.02, max: 0.98, step: 0.01, value: thr, fmt: (v) => v.toFixed(2), onInput: (v) => { thr = v; paint(); } });
    const st = stats(f.root);
    const row = el('div', { class: 'fig-row' }); f.body.append(row);
    const a = el('div'), b = el('div'); row.append(a, b);
    const R = frame(a, { width: 460, height: 400, x: [0, 1], y: [0, 1], xLabel: '1 − specificity', yLabel: 'sensitivity' });
    drawROC(R, rocV, aucV);
    const P = frame(b, { width: 460, height: 400, x: [0, 1], y: [0, 1], xLabel: 'recall = sensitivity', yLabel: 'precision = PPV' });
    const ro = R.layer('op'), pl = P.layer('pr');
    legend(f.body, [['model', cat(6)], ['random classifier', muted()], ['operating point at the threshold', ink()]]);
    function paint() {
      const c = cm(thr), pr = prAt(rocV, pi), a2 = apOf(pr), ppv = ppvAt(c.sens, c.spec, pi);
      ro.selectAll('*').remove(); pl.selectAll('*').remove();
      ro.append('circle').attr('cx', R.x(1 - c.spec)).attr('cy', R.y(c.sens)).attr('r', 6.5).attr('fill', ink()).attr('stroke', css('--surface')).attr('stroke-width', 2);
      pl.append('line').attr('x1', 0).attr('x2', P.w).attr('y1', P.y(pi)).attr('y2', P.y(pi)).attr('stroke', muted()).attr('stroke-dasharray', '5 4').attr('stroke-width', 1.5);
      pl.append('text').attr('x', P.w - 4).attr('y', P.y(pi) - 6).attr('text-anchor', 'end').attr('font-size', 11.5).attr('fill', muted()).text(`random: precision = prevalence = ${pc1(pi)}`);
      pl.append('path').datum(pr).attr('fill', 'none').attr('stroke', cat(6)).attr('stroke-width', 2.4).attr('d', d3.line().curve(d3.curveStepBefore).x((q) => P.x(q.r)).y((q) => P.y(q.p)));
      if (c.tp + c.fp) pl.append('circle').attr('cx', P.x(c.sens)).attr('cy', P.y(ppv)).attr('r', 6.5).attr('fill', ink()).attr('stroke', css('--surface')).attr('stroke-width', 2);
      pl.append('text').attr('x', P.w - 4).attr('y', 14).attr('text-anchor', 'end').attr('font-size', 12).attr('fill', ink2()).text(`AP = ${f3(a2)}`);
      const acc = c.sens * pi + c.spec * (1 - pi);
      st.set({ 'AUC (unchanged)': f3(aucV), 'average precision': f3(a2), PPV: c.sens > 0 ? pc1(ppv) : 'n/a', NPV: pc1(npvAt(c.sens, c.spec, pi)), accuracy: pc1(acc), '"nobody responds" accuracy': pc1(1 - pi), 'false positives per true positive': c.sens > 0 ? (c.spec < 1 ? ((1 - c.spec) * (1 - pi) / (c.sens * pi)).toFixed(1) : '0') : 'n/a' });
    }
    paint();
  }
  prose(rb, `<p>As prevalence falls, AUC is constant while PPV and AP fall, so a model with a high AUC can still have a low PPV in a screening population. Conversely, a PPV or AP measured at one prevalence cannot be transferred to another without reweighting as above. "Imbalance" is therefore not a flaw of the model: it changes which measure answers the question. Centre C is mildly imbalanced (${pc(piV)} positive), so the effects in this figure appear mainly at the low-prevalence settings.</p>`);
  quiz(rb, 'A test has 90% sensitivity and 90% specificity. What is its PPV at 1% prevalence?',
    `<p>Per 1,000 people: 10 have the condition, 9 are found; 990 do not, and 10% of them (99) are flagged. PPV = 9 / (9 + 99) = ${pc1(9 / 108)}. Even a test with 90% sensitivity and 90% specificity yields mostly false positives at low prevalence.</p>`);

  // ---- Step 5: calibration
  rb = rung(ch, 5, 'Calibration: do the probabilities mean what they say?',
    `<p>A model is <b>calibrated</b> if, among patients given a predicted probability of 0.3, about 30% respond. A <b>reliability diagram</b> checks this: patients are grouped by predicted probability, and each group's mean prediction (horizontal) is plotted against its observed response rate (vertical). Points on the diagonal indicate calibration; points below it mean the predictions are too high. The <b>Brier score</b>, the mean squared difference between prediction and outcome (0 or 1), summarises both discrimination and calibration; lower is better, and always predicting the prevalence gives ${f3(piV * (1 - piV))} on Centre C.</p>`);
  {
    const f = figure(rb, 'Reliability diagram on Centre C', 'Patients are sorted by predicted probability and cut into equal-sized groups. Marker area is proportional to group size; bars are 95% Wilson intervals for the observed rate; ticks along the bottom are the individual predictions. The sliders distort the model: sharpness multiplies the log-odds (values above 1 make predictions more extreme), shift adds a constant to them. Ranking, and hence AUC, is not changed by either.');
    let sh = 1, shift = 0, nb = 5;
    const sS = slider(f.controls, { label: 'sharpness', min: 0.2, max: 3, step: 0.05, value: sh, fmt: f2, onInput: (v) => { sh = v; paint(); } });
    const sT = slider(f.controls, { label: 'shift (log-odds)', min: -2, max: 2, step: 0.05, value: shift, fmt: f2, onInput: (v) => { shift = v; paint(); } });
    select(f.controls, { label: 'groups', value: String(nb), options: [['3', '3'], ['5', '5'], ['8', '8']], onChange: (v) => { nb = +v; paint(); } });
    button(f.controls, 'Reset', () => { sh = 1; shift = 0; sS.set(1); sT.set(0); paint(); });
    const st = stats(f.root);
    const row = el('div', { class: 'fig-row' }); f.body.append(row);
    const a = el('div'), b = el('div'); row.append(a, b);
    const Rl = frame(a, { width: 500, height: 400, x: [0, 1], y: [0, 1], xLabel: 'mean predicted probability', yLabel: 'observed response rate' });
    const Hf = frame(b, { width: 420, height: 400, x: [0, 1], y: [0, 1], xLabel: 'predicted probability', yLabel: 'share of patients', grid: false });
    const zV = pv.map((p) => logit(clip(p)));
    legend(f.body, [['observed rate in a group', cat(6)], ['perfect calibration', muted()], ['responders', cat(1)], ['non-responders', cat(0)]]);
    function paint() {
      const p2 = zV.map((z) => sigmoid(sh * z + shift));
      drawRel(Rl, p2, yv, nb);
      const hb = d3.bin().domain([0, 1]).thresholds(10);
      const h0 = hb(p2.filter((_, i) => !yv[i])), h1 = hb(p2.filter((_, i) => yv[i]));
      const mx = Math.max(d3.max(h0, (q) => q.length), d3.max(h1, (q) => q.length)) / nV;
      Hf.y.domain([0, mx * 1.1]); Hf.g.select('.axis.y').call(d3.axisLeft(Hf.y).ticks(5).tickFormat((v) => Math.round(100 * v) + '%'));
      const hl = Hf.layer('h'); hl.selectAll('*').remove();
      [h0, h1].forEach((h, k) => hl.selectAll('.b' + k).data(h).join('rect').attr('x', (q) => Hf.x(q.x0) + k * (Hf.x(0.1) - Hf.x(0)) / 2).attr('width', (Hf.x(0.1) - Hf.x(0)) / 2 - 1).attr('y', (q) => Hf.y(q.length / nV)).attr('height', (q) => Hf.h - Hf.y(q.length / nV))
        .attr('fill', cat(k)).attr('fill-opacity', 0.85).on('mousemove', (e, q) => tip(e, `${k ? 'responders' : 'non-responders'}, predicted ${f2(q.x0)}–${f2(q.x1)}<br>${q.length} patients`)).on('mouseleave', untip));
      const bs = brier(p2, yv);
      st.set({ 'AUC (unchanged)': f3(auc(p2, yv)), 'Brier score': f3(bs), 'Brier of prevalence-only': f3(piV * (1 - piV)), 'log-loss': f3(logLoss(p2, yv)), 'mean prediction': pc1(mean(p2)), 'observed rate': pc1(piV) });
    }
    paint();
  }
  const zs = pv.map((p) => logit(clip(p)));
  const sharp2 = zs.map((z) => sigmoid(2 * z)), up1 = zs.map((z) => sigmoid(z + 1));
  prose(rb, `<p>On Centre C the model's mean prediction is ${pc1(mean(pv))} against an observed rate of ${pc1(piV)}, with Brier score ${f3(M3.brier)}. Doubling the log-odds leaves the AUC at ${f3(auc(sharp2, yv))} but changes the Brier score to ${f3(brier(sharp2, yv))}; adding 1 to the log-odds changes it to ${f3(brier(up1, yv))}. <b>AUC and calibration are separate properties</b>: a model can rank perfectly and still give probabilities that are too extreme or too high. Calibration can be repaired without retraining (Platt scaling fits exactly the slope and intercept varied here; isotonic regression fits a step function), but only on data not used to judge it.</p>
    <p>With ${nV} patients, each of the five groups holds about ${Math.round(nV / 5)} patients and its interval is wide, so small departures from the diagonal are not evidence of miscalibration.</p>`);

  // ---- Step 6: lab
  rb = rung(ch, 6, 'Lab: compare models with every measure',
    `<p>Each model is fitted on discovery patients (Centres A and B) and evaluated on the ${nV} Centre C patients. The three panels show the ranking (ROC), the precision–recall trade-off and the calibration of the selected model, with the threshold marked on the first two. Features are standardised before fitting.</p>`);
  {
    const f = figure(rb, 'Model lab', 'Choose a model and drag the threshold. The last option fits all ten features on only the first 30 discovery patients, to show overfitting.');
    let mk = 'three', thr = 0.5;
    select(f.controls, { label: 'Model', value: mk, options: Object.entries(MODELS).map(([k, m]) => [k, m.label]), onChange: (v) => { mk = v; paint(); } });
    slider(f.controls, { label: 'threshold', min: 0.02, max: 0.98, step: 0.01, value: thr, fmt: (v) => v.toFixed(2), onInput: (v) => { thr = v; paint(); } });
    const st = stats(f.root);
    const row = el('div', { class: 'fig-row' }); f.body.append(row);
    const mk3 = () => { const d = el('div', { style: 'flex:1 1 260px' }); row.append(d); return d; };
    const R = frame(mk3(), { width: 300, height: 300, x: [0, 1], y: [0, 1], xLabel: '1 − specificity', yLabel: 'sensitivity', margin: { left: 46 } });
    const P = frame(mk3(), { width: 300, height: 300, x: [0, 1], y: [0, 1], xLabel: 'recall', yLabel: 'precision', margin: { left: 46 } });
    const L = frame(mk3(), { width: 300, height: 300, x: [0, 1], y: [0, 1], xLabel: 'mean predicted', yLabel: 'observed rate', margin: { left: 46 } });
    legend(f.body, [['model', cat(6)], ['random / perfect calibration', muted()], ['operating point', ink()]]);
    const msg = html('', 'p', 'fig-caption'); f.body.append(msg);
    function paint() {
      const m = MODELS[mk], c = cm(thr, m.pv), pr = prAt(m.roc, piV), a = m.aucV, se = hm(a, nPos, nNeg);
      drawROC(R, m.roc, a, { compact: true });
      drawPR(P, pr, piV, apOf(pr), { compact: true });
      drawRel(L, m.pv, yv, 5, { compact: true });
      R.layer('op').append('circle').attr('cx', R.x(1 - c.spec)).attr('cy', R.y(c.sens)).attr('r', 6).attr('fill', ink()).attr('stroke', css('--surface')).attr('stroke-width', 2);
      if (c.tp + c.fp) P.layer('op').append('circle').attr('cx', P.x(c.sens)).attr('cy', P.y(c.ppv)).attr('r', 6).attr('fill', ink()).attr('stroke', css('--surface')).attr('stroke-width', 2);
      st.set({ 'AUC train': f3(m.aucT), 'AUC Centre C': f3(a), '95% CI (approx.)': `${f2(a - 1.96 * se)}–${f2(Math.min(1, a + 1.96 * se))}`, AP: f3(apOf(pr)), 'Brier': f3(m.brier), 'log-loss': f3(m.ll), sensitivity: pc(c.sens), specificity: pc(c.spec), PPV: c.tp + c.fp ? pc(c.ppv) : 'n/a' });
      msg.innerHTML = mk === 'centre' ? 'Every Centre C patient has the same value of the only feature, so every patient receives the same score and the ROC curve is the diagonal.' : '';
    }
    paint();
  }
  const A = MODELS;
  const c30b = cm(0.3);
  table(rb, [
    { t: '<b>IFN-γ only</b> → <b>true three</b>', s: `AUC ${f3(A.ifng.aucV)} → <b>${f3(A.three.aucV)}</b>; Brier ${f3(A.ifng.brier)} → <b>${f3(A.three.brier)}</b>`, m: `Informative features raise the ranking measures. Intervals of about ±${f2(1.96 * aucSE)} on ${nV} patients mean that small AUC differences between models are not reliable.` },
    { t: '<b>PD-L1 only</b>', s: `AUC ${f3(A.pdl1.aucV)}; average precision ${f3(apOf(prAt(A.pdl1.roc, piV)))}`, m: 'A proxy marker can rank patients reasonably because it is induced by the causal feature, but it ranks less well than the model that contains the cause.' },
    { t: '<b>Centre only</b>', s: `AUC ${f3(A.centre.aucV)}; Brier ${f3(A.centre.brier)}`, m: `Centre is only weakly related to response in Centres A and B (training AUC ${f3(A.centre.aucT)}), and Centre C patients all share one value, so the model cannot rank them. Predictions at the prevalence level give Brier ${f3(A.centre.brier)}, compared with ${f3(piV * (1 - piV))} for the exact prevalence.` },
    { t: '<b>All 10 features, 30 patients</b>', s: `Train AUC ${f3(A.over.aucT)} → Centre C ${f3(A.over.aucV)}; Brier ${f3(A.over.brier)}; log-loss ${f3(A.over.ll)}`, m: `Few patients and many features overfit: the training AUC is the most favourable number. On Centre C the log-loss is ${f3(A.over.ll)} against ${f3(A.three.ll)} for the true three, and the Brier score ${f3(A.over.brier)} against ${f3(A.three.brier)}: ${A.over.ll > A.three.ll ? 'the extra features degrade the probabilities more than the ranking' : 'the probabilities remain comparable'}.` },
    { t: 'Threshold <b>0.5</b> → <b>0.3</b> (true three)', s: `Sensitivity ${pc(c50.sens)} → <b>${pc(c30b.sens)}</b>; specificity ${pc(c50.spec)} → <b>${pc(c30b.spec)}</b>; PPV ${pc(c50.ppv)} → <b>${pc(c30b.ppv)}</b>`, m: 'The AUC and the curves do not change, only the operating point. The threshold should follow from the relative costs of the two errors.' },
  ], [['t', 'Try'], ['s', 'What you see'], ['m', 'What it means']], { cls: 'compact' });

  // =====================================================================================
  // 5. PITFALLS
  // =====================================================================================
  ch = chapter(main, 'pitfalls', 'Pitfalls', 'Common problems, how to recognise them, and how to address them.');
  const youden = (pts) => pts.slice(1).reduce((b, q) => (q.tpr - q.fpr > b.j ? { j: q.tpr - q.fpr, thr: q.thr } : b), { j: -1, thr: 0.5 });
  const dRoc = roc(M3.pt, y), tD = youden(dRoc).thr, tV = youden(rocV).thr;
  const cD = cm(tD), cVo = cm(tV);
  const byPh = ['inflamed', 'excluded', 'desert'].map((k) => { const ix = V.map((p, i) => i).filter((i) => V[i].phenotype === k); return { k, n: ix.length, a: auc(ix.map((i) => pv[i]), ix.map((i) => yv[i])), pos: sum(ix.map((i) => yv[i])) }; }).filter((q) => q.pos > 0 && q.pos < q.n);
  const aucTxt = byPh.length ? byPh.map((q) => `${q.k} tumours (n = ${q.n}): ${f3(q.a)}`).join('; ') : 'not computable';
  cards(ch, [
    { tag: 'Symptom: accuracy looks high, yet few positives are found', title: 'Accuracy under imbalance', body: `At 2% prevalence, "nobody responds" has ${pc(0.98)} accuracy and zero sensitivity. <b>Cause:</b> accuracy is dominated by the majority class. <b>Fix:</b> report sensitivity, specificity and PPV, and use precision–recall measures when positives are rare.` },
    { tag: 'Symptom: excellent AUC, but most positive calls are wrong', title: 'AUC read as PPV', body: `AUC ${f3(aucV)} here, yet at 1% prevalence the PPV at threshold 0.5 would be ${pc1(ppvAt(c50.sens, c50.spec, 0.01))}. <b>Cause:</b> AUC ignores prevalence. <b>Fix:</b> state the intended prevalence and report PPV and AP for it.` },
    { tag: 'Symptom: sensitivity and specificity look better than they are', title: 'Choosing the threshold on the test data', body: `The threshold maximising sensitivity + specificity − 1 is ${f2(tD)} on discovery and ${f2(tV)} on Centre C. Chosen on Centre C itself it gives sensitivity ${pc(cVo.sens)} and specificity ${pc(cVo.spec)}; carried over from discovery, ${pc(cD.sens)} and ${pc(cD.spec)}. <b>Cause:</b> a threshold tuned on the evaluation data is part of the model. <b>Fix:</b> fix the threshold before touching the test set.` },
    { tag: 'Symptom: a "best" threshold that suits nobody', title: 'The Youden point ignores costs', body: 'The point farthest from the diagonal assumes false positives and false negatives cost the same and that prevalence is that of the sample. <b>Fix:</b> pick the threshold from the clinical costs and the target prevalence, or from a required sensitivity.' },
    { tag: 'Symptom: two models with AUC 0.80 and 0.82 are declared different', title: 'No uncertainty on the metric', body: `On ${nV} patients the AUC of the true-three model is ${f3(aucV)} ± ${f3(1.96 * aucSE)}. <b>Cause:</b> test sets are small and the metrics are noisy. <b>Fix:</b> report intervals (bootstrap or Hanley–McNeil), and compare models on the same patients with a paired test such as DeLong's.` },
    { tag: 'Symptom: good AUC, poor probabilities', title: 'Discrimination without calibration', body: `Doubling the log-odds leaves the AUC at ${f3(auc(sharp2, yv))} while the Brier score moves from ${f3(M3.brier)} to ${f3(brier(sharp2, yv))}. <b>Fix:</b> draw a reliability diagram, report Brier or log-loss, and recalibrate on separate data before using the probabilities in decisions.` },
    { tag: 'Symptom: one pooled number hides weak groups', title: 'Averaging over subgroups', body: `Pooled AUC ${f3(aucV)}; within phenotype — ${aucTxt}. Within-group AUCs also use fewer patients. <b>Cause:</b> part of the pooled AUC comes from differences between groups. <b>Fix:</b> report the measures within clinically relevant subgroups, with their sample sizes.` },
    { tag: 'Symptom: performance drops in a new centre', title: 'Evaluating on the training distribution', body: `Training AUC of the model fitted on 30 patients is ${f3(A.over.aucT)} against ${f3(A.over.aucV)} on Centre C. Prevalence also differs between sites (${pc(piD)} in discovery, ${pc(piV)} in Centre C). <b>Fix:</b> evaluate on independent cohorts, nested cross-validation inside the training data, and report calibration per site.` },
  ]);

  // =====================================================================================
  // 6. RECAP
  // =====================================================================================
  ch = chapter(main, 'recap', 'Recap', 'Key points.');
  recap(ch, [
    'A threshold turns a score into a call; comparing calls with outcomes gives the <b>confusion matrix</b> (TP, FP, TN, FN), and every threshold-based measure is a ratio of these counts.',
    '<b>Sensitivity</b> and <b>specificity</b> describe the test and are unaffected by prevalence; <b>PPV</b> and <b>NPV</b> also depend on prevalence (Bayes\' rule).',
    'The <b>ROC curve</b> plots sensitivity against 1 − specificity over all thresholds; <b>AUC</b> is the probability that a random responder outranks a random non-responder, and does not depend on prevalence.',
    'The <b>precision–recall curve</b> is preferable when positives are rare because false positives directly reduce precision; its random baseline is the prevalence, and it changes with prevalence.',
    '<b>Accuracy</b> can be high for a classifier that finds no positives; report both error rates and PPV.',
    '<b>Calibration</b> (reliability diagram, Brier score, log-loss) checks whether probabilities can be taken literally. It is independent of AUC and can be repaired by recalibration on separate data.',
    'Metrics from small test sets are imprecise; report <b>confidence intervals</b> and fix the threshold before testing.',
    'Evaluate on independent patients (here Centre C), report prevalence, and check performance in subgroups and new sites.',
  ]);
  prose(ch, '<h3>In code</h3><p>All of these measures are available in scikit-learn and in R; the threshold-free ones take the scores, the others take thresholded calls.</p>');
  code(ch, {
    Python: `import numpy as np
from sklearn.metrics import (confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve,
                             average_precision_score, brier_score_loss, log_loss)
from sklearn.calibration import calibration_curve

# p: predicted probabilities on held-out patients, y: 0/1 outcomes
tn, fp, fn, tp = confusion_matrix(y, p >= 0.5).ravel()
sens, spec, ppv = tp / (tp + fn), tn / (tn + fp), tp / (tp + fp)

auc = roc_auc_score(y, p);  fpr, tpr, thr = roc_curve(y, p)
prec, rec, thr = precision_recall_curve(y, p);  ap = average_precision_score(y, p)

brier = brier_score_loss(y, p);  ll = log_loss(y, p)
obs, pred = calibration_curve(y, p, n_bins=5, strategy="quantile")   # reliability diagram

# PPV at another prevalence (Bayes' rule)
pi = 0.01;  ppv_pi = sens * pi / (sens * pi + (1 - spec) * (1 - pi))

# Uncertainty: bootstrap the AUC
rng = np.random.default_rng(0)
boot = [roc_auc_score(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(1000))
        if 0 < y[i].sum() < len(i)]
print(auc, np.percentile(boot, [2.5, 97.5]))`,
    R: `library(pROC); library(PRROC)
# p: predicted probabilities on held-out patients, y: 0/1 outcomes
pred <- as.integer(p >= 0.5)
tab <- table(factor(pred, 0:1), factor(y, 0:1))      # rows: predicted, columns: actual
tp <- tab[2, 2]; fp <- tab[2, 1]; fn <- tab[1, 2]; tn <- tab[1, 1]
c(sens = tp / (tp + fn), spec = tn / (tn + fp), ppv = tp / (tp + fp))

r <- roc(y, p);  auc(r);  ci.auc(r)                   # DeLong interval
pr <- pr.curve(scores.class0 = p[y == 1], scores.class1 = p[y == 0], curve = TRUE)
pr$auc.integral                                       # average precision (area under PR)

mean((p - y)^2)                                       # Brier score
# Reliability diagram: observed rate in 5 groups of equal size
g <- cut(p, quantile(p, 0:5 / 5), include.lowest = TRUE)
cbind(predicted = tapply(p, g, mean), observed = tapply(y, g, mean))`,
  });
  seeAlso(main, [['concepts', 'logistic-regression', 'Logistic regression'], ['omics', 'bulk-classifier', 'Biomarker classifier and nested CV']]);
}

// ---------------------------------------------------------------------------------------------
// Shared drawing helpers. Each clears and redraws the 'dyn' layer of its frame.
function drawROC(F, pts, a, opt = {}) {
  const L = F.layer('dyn'); L.selectAll('*').remove();
  L.append('line').attr('x1', F.x(0)).attr('y1', F.y(0)).attr('x2', F.x(1)).attr('y2', F.y(1)).attr('stroke', muted()).attr('stroke-dasharray', '5 4').attr('stroke-width', 1.5);
  L.append('path').datum(pts).attr('fill', cat(6)).attr('fill-opacity', 0.12).attr('d', d3.area().x((q) => F.x(q.fpr)).y0(F.y(0)).y1((q) => F.y(q.tpr)));
  L.append('path').datum(pts).attr('fill', 'none').attr('stroke', cat(6)).attr('stroke-width', 2.4).attr('d', d3.line().x((q) => F.x(q.fpr)).y((q) => F.y(q.tpr)));
  L.append('text').attr('x', F.x(0.98)).attr('y', F.y(0.08)).attr('text-anchor', 'end').attr('font-size', 12.5).attr('font-weight', 700).attr('fill', ink()).text(`AUC = ${f3(a)}`);
  if (!opt.compact) {
    L.append('text').attr('x', F.x(0.04)).attr('y', F.y(0.97)).attr('font-size', 11.5).attr('fill', muted()).text('↖ better');
    L.append('text').attr('transform', `translate(${F.x(0.52)},${F.y(0.47)}) rotate(${-Math.atan((F.y(0) - F.y(1)) / (F.x(1) - F.x(0))) * 180 / Math.PI})`).attr('font-size', 11.5).attr('fill', muted()).attr('text-anchor', 'middle').attr('dy', 14).text('random score');
  }
  L.selectAll('circle.h').data(pts.slice(1)).join('circle').attr('class', 'h').attr('cx', (q) => F.x(q.fpr)).attr('cy', (q) => F.y(q.tpr)).attr('r', 5).attr('fill', 'transparent')
    .on('mousemove', (e, q) => tip(e, `threshold ${f2(q.thr)}<br>sensitivity ${pc(q.tpr)}, specificity ${pc(1 - q.fpr)}`)).on('mouseleave', untip);
  F.layer('op').selectAll('*').remove();
}
function drawPR(F, pr, pi, ap, opt = {}) {
  const L = F.layer('dyn'); L.selectAll('*').remove();
  L.append('line').attr('x1', 0).attr('x2', F.w).attr('y1', F.y(pi)).attr('y2', F.y(pi)).attr('stroke', muted()).attr('stroke-dasharray', '5 4').attr('stroke-width', 1.5);
  L.append('text').attr('x', F.w - 4).attr('y', F.y(pi) - 6).attr('text-anchor', 'end').attr('font-size', 11.5).attr('fill', muted()).text(opt.compact ? 'random' : `random classifier: precision = prevalence = ${pc1(pi)}`);
  L.append('path').datum(pr).attr('fill', 'none').attr('stroke', cat(6)).attr('stroke-width', 2.4).attr('d', d3.line().curve(d3.curveStepBefore).x((q) => F.x(q.r)).y((q) => F.y(q.p)));
  L.append('text').attr('x', F.w - 4).attr('y', F.y(0.1)).attr('text-anchor', 'end').attr('font-size', 12.5).attr('font-weight', 700).attr('fill', ink()).text(`AP = ${f3(ap)}`);
  L.selectAll('circle.h').data(pr).join('circle').attr('class', 'h').attr('cx', (q) => F.x(q.r)).attr('cy', (q) => F.y(q.p)).attr('r', 5).attr('fill', 'transparent')
    .on('mousemove', (e, q) => tip(e, `threshold ${f2(q.thr)}<br>recall ${pc(q.r)}, precision ${pc(q.p)}`)).on('mouseleave', untip);
  F.layer('op').selectAll('*').remove();
}
/** Reliability diagram: equal-count groups, Wilson intervals, marker area ~ group size. */
function drawRel(F, p, y, k, opt = {}) {
  const L = F.layer('dyn'); L.selectAll('*').remove();
  const ord = p.map((v, i) => i).sort((a, b) => p[a] - p[b]), bins = [];
  for (let i = 0; i < k; i++) {
    const ix = ord.slice(Math.round(i * ord.length / k), Math.round((i + 1) * ord.length / k)), n = ix.length, o = sum(ix.map((j) => y[j])) / n, z = 1.96, d = 1 + z * z / n;
    const ctr = (o + z * z / (2 * n)) / d, hw = z * Math.sqrt(o * (1 - o) / n + z * z / (4 * n * n)) / d;
    bins.push({ x: mean(ix.map((j) => p[j])), o, n, lo: ctr - hw, hi: ctr + hw });
  }
  L.append('line').attr('x1', F.x(0)).attr('y1', F.y(0)).attr('x2', F.x(1)).attr('y2', F.y(1)).attr('stroke', muted()).attr('stroke-dasharray', '5 4').attr('stroke-width', 1.5);
  L.append('text').attr('x', F.x(0.6)).attr('y', F.y(0.6) + 16).attr('font-size', 11.5).attr('fill', muted()).attr('transform', `rotate(-45,${F.x(0.6)},${F.y(0.6) + 16})`).text(opt.compact ? '' : 'perfect calibration');
  if (!opt.compact) {
    L.append('text').attr('x', F.x(0.98)).attr('y', F.y(0.04)).attr('text-anchor', 'end').attr('font-size', 11.5).attr('fill', muted()).text('below the diagonal: predictions too high');
    L.append('text').attr('x', F.x(0.02)).attr('y', F.y(0.97)).attr('font-size', 11.5).attr('fill', muted()).text('above: predictions too low');
  }
  L.selectAll('line.e').data(bins).join('line').attr('class', 'e').attr('x1', (b) => F.x(b.x)).attr('x2', (b) => F.x(b.x)).attr('y1', (b) => F.y(b.lo)).attr('y2', (b) => F.y(b.hi)).attr('stroke', ink2()).attr('stroke-width', 1.5);
  L.append('path').datum(bins).attr('fill', 'none').attr('stroke', cat(6)).attr('stroke-width', 2).attr('d', d3.line().x((b) => F.x(b.x)).y((b) => F.y(b.o)));
  L.selectAll('circle.m').data(bins).join('circle').attr('class', 'm').attr('cx', (b) => F.x(b.x)).attr('cy', (b) => F.y(b.o)).attr('r', (b) => 3 + 1.1 * Math.sqrt(b.n)).attr('fill', cat(6)).attr('stroke', css('--surface')).attr('stroke-width', 1.5)
    .on('mousemove', (e, b) => tip(e, `${b.n} patients<br>mean predicted ${pc(b.x)}<br>observed ${pc(b.o)} (95% interval ${pc(b.lo)}–${pc(b.hi)})`)).on('mouseleave', untip);
  L.selectAll('line.r').data(p).join('line').attr('class', 'r').attr('x1', (v) => F.x(v)).attr('x2', (v) => F.x(v)).attr('y1', F.h).attr('y2', F.h - 7).attr('stroke', ink2()).attr('stroke-opacity', 0.45);
}
