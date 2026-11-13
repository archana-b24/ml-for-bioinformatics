// Bulk step 11: lock the signature (genes, coefficients, preprocessing, threshold), apply it unchanged to an
// external cohort from another centre, and separate discrimination from calibration drift.
import { header, figure, select, slider, stats, section, steps, callout, math, legend, seeAlso, tip, untip, el, html } from '../lib/ui.js';
import { frame, cat, css, ink2, muted } from '../lib/plot.js';
import { stepCard, omicsData } from './_shared.js';
const d3 = window.d3;
const f2 = (v) => v.toFixed(2), f3 = (v) => v.toFixed(3), sg = (v) => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(2);

function rocOf(y, p) {
  const idx = d3.range(y.length).sort((a, b) => p[b] - p[a]);
  const P = d3.sum(y), N = y.length - P; let tp = 0, fp = 0; const pts = [{ x: 0, y: 0 }];
  idx.forEach((i, k) => { if (y[i]) tp++; else fp++; if (k === idx.length - 1 || p[idx[k + 1]] !== p[i]) pts.push({ x: fp / N, y: tp / P }); });
  return pts;
}
function qBins(y, p, k) { // equal-count bins
  const o = d3.range(y.length).sort((a, b) => p[a] - p[b]);
  return d3.range(k).map((j) => { const ii = o.slice(Math.round(j * o.length / k), Math.round((j + 1) * o.length / k)); return { n: ii.length, mp: d3.mean(ii, (i) => p[i]), obs: d3.mean(ii, (i) => y[i]) }; });
}

export default async function render(main) {
  header(main, 'Locking & validating the signature',
    'A biomarker is only a biomarker once it is <b>frozen</b>: the genes, weights, preprocessing and cut-off are written down before anyone looks at new patients. Then you apply it unchanged to an <b>independent cohort</b> and ask two separate questions: does it still rank patients (discrimination), and are its probabilities still right (calibration)?');
  const D = await omicsData(main, 'bulk_validation.json');
  if (!D) return;
  const S = D.strategies, L = D.lock;

  stepCard(main, {
    stage: 'Bulk RNA-seq · step 11 of 11',
    question: 'Does the response signature found in Centres A and B work in a new centre, on new patients, without re-tuning?',
    data: `Locked model from ${D.n_d} discovery patients; external cohort of ${D.n_v} anti-PD-1 patients from Centre C (different sequencing chemistry and depth, more desert tumours); 12 bridging RNA samples sequenced at both Centre A and Centre C.`,
    method: 'Apply the frozen lasso logistic model; AUC with bootstrap CI, calibration-in-the-large, calibration slope, Brier score, sensitivity/specificity at the frozen cut-off.',
    verdict: 'stats',
    why: 'Validation is a pre-specified statistical evaluation of a fixed model. Nothing is learned from the validation cohort, except in the explicitly separate "model updating" step.',
  });

  // ---------------- model card
  main.append(html(`<table class="simple" style="max-width:780px;margin:10px 0 4px"><tr><th colspan="2">Locked model card</th></tr>
    <tr><td>Genes</td><td>${L.genes.map((g) => `<code>${g}</code>`).join(', ')} (selected by the 1-SE lasso in step 5)</td></tr>
    <tr><td>Weights</td><td>unpenalised logistic refit on those genes ("relaxed lasso"): the lasso's own coefficients (${L.lasso_coef.map(f3).join(', ')}) are shrunk toward zero and would give badly under-confident probabilities</td></tr>
    <tr><td>Preprocessing</td><td>log2 CPM; z-score with the discovery means (${L.mu.map(f2).join(', ')}) and SDs (${L.sd.map(f2).join(', ')})</td></tr>
    <tr><td>Linear predictor</td><td>${f3(L.intercept)} ${L.genes.map((g, j) => `${L.coef[j] >= 0 ? '+' : '−'} ${Math.abs(L.coef[j]).toFixed(3)}·z(${g})`).join(' ')}</td></tr>
    <tr><td>Output</td><td>P(response) = 1 / (1 + e<sup>−linear predictor</sup>)</td></tr>
    <tr><td>Decision cut-off</td><td>P ≥ ${f3(D.thr)} (Youden index on internal cross-validated predictions), frozen</td></tr></table>`));

  // ---------------- Figure 1: ROC + calibration by strategy
  const f1 = figure(main, 'Internal vs external performance',
    'Choose how the Centre C data are fed to the locked model. Left: ROC curve (grey = internal cross-validated reference). Right: calibration with equal-sized groups of patients; dot size ∝ patients. The vertical line on the right is the frozen cut-off.');
  let sk = 'frozen', nb = 5;
  const opts = Object.entries(S).map(([k, v]) => [k, v.label]);
  select(f1.controls, { label: 'Cohort / strategy', value: sk, options: opts, onChange: (v) => { sk = v; draw(); } });
  slider(f1.controls, { label: 'Calibration groups', min: 3, max: 10, step: 1, value: nb, onInput: (v) => { nb = v; draw(); } });
  legend(f1.root, [['selected', cat(0)], ['internal CV reference', muted()]]);
  const s1 = stats(f1.root);
  const row = el('div', { class: 'fig-row' }); f1.body.append(row);
  const Lb = el('div'), Rb = el('div'); row.append(Lb, Rb);
  const Rc = frame(Lb, { width: 440, height: 400, x: [0, 1], y: [0, 1], xLabel: 'false-positive rate', yLabel: 'true-positive rate' });
  Rc.layer('diag').append('line').attr('x1', Rc.x(0)).attr('y1', Rc.y(0)).attr('x2', Rc.x(1)).attr('y2', Rc.y(1)).attr('stroke', muted()).attr('stroke-dasharray', '4 4');
  const refRoc = rocOf(S.internal.y, S.internal.p);
  Rc.layer('ref').append('path').datum(refRoc).attr('fill', 'none').attr('stroke', muted()).attr('stroke-width', 2).attr('d', d3.line().curve(d3.curveStepAfter).x((d) => Rc.x(d.x)).y((d) => Rc.y(d.y)));
  const Cc = frame(Rb, { width: 440, height: 400, x: [0, 1], y: [0, 1], xLabel: 'mean predicted probability (group)', yLabel: 'observed response rate (group)' });
  Cc.layer('diag').append('line').attr('x1', Cc.x(0)).attr('y1', Cc.y(0)).attr('x2', Cc.x(1)).attr('y2', Cc.y(1)).attr('stroke', muted()).attr('stroke-dasharray', '4 4');
  Cc.layer('thr').append('line').attr('x1', Cc.x(D.thr)).attr('x2', Cc.x(D.thr)).attr('y1', 0).attr('y2', Cc.h).attr('stroke', ink2()).attr('stroke-dasharray', '5 4');
  function draw() {
    const s = S[sk];
    Rc.layer('roc').selectAll('path').data([rocOf(s.y, s.p)]).join('path').attr('fill', 'none').attr('stroke', cat(0)).attr('stroke-width', 2.5)
      .attr('d', d3.line().curve(d3.curveStepAfter).x((d) => Rc.x(d.x)).y((d) => Rc.y(d.y)));
    const b = qBins(s.y, s.p, nb), rb = qBins(S.internal.y, S.internal.p, nb);
    Cc.layer('ref').selectAll('path').data([rb]).join('path').attr('fill', 'none').attr('stroke', muted()).attr('stroke-width', 2).attr('d', d3.line().x((d) => Cc.x(d.mp)).y((d) => Cc.y(d.obs)));
    Cc.layer('line').selectAll('path').data([b]).join('path').attr('fill', 'none').attr('stroke', cat(0)).attr('stroke-width', 2).attr('d', d3.line().x((d) => Cc.x(d.mp)).y((d) => Cc.y(d.obs)));
    Cc.layer('pts').selectAll('circle').data(b).join('circle').attr('cx', (d) => Cc.x(d.mp)).attr('cy', (d) => Cc.y(d.obs)).attr('r', (d) => 3 + Math.sqrt(d.n))
      .attr('fill', cat(0)).attr('stroke', css('--surface')).attr('stroke-width', 1.5)
      .on('mousemove', (e, d) => tip(e, `${d.n} patients<br>mean predicted ${f2(d.mp)}<br>observed ${f2(d.obs)}`)).on('mouseleave', untip);
    s1.set({ 'AUC [95% CI]': `${f3(s.auc)} [${f2(s.ci[0])}, ${f2(s.ci[1])}]`, 'mean predicted vs observed': `${f2(s.mean_p)} vs ${f2(s.obs)}`,
      'calibration-in-the-large': sg(s.citl), 'calibration slope': f2(s.slope), 'Brier': f3(s.brier),
      'sensitivity / specificity at frozen cut-off': `${f2(s.sens)} / ${f2(s.spec)}` });
  }
  draw();

  // ---------------- Figure 2: gene-wise technical shift
  const G = D.genes_shift;
  const f2g = figure(main, 'Why the probabilities drift: a gene-wise technical shift',
    'Each dot is one of the 600 most expressed genes: mean log2 CPM at Centre A (x) vs Centre C (y). Off-diagonal scatter is mostly technical (each gene changes by its own factor at a new centre). The locked genes are highlighted. Raw differences in means mix that technical shift with case mix; the bridging samples (same RNA, both centres) isolate the technical part.');
  const s2 = stats(f2g.root);
  const box = el('div'); f2g.body.append(box);
  const ext = d3.extent([...G.meanA, ...G.meanC]);
  const Sc = frame(box, { width: 900, height: 380, x: [ext[0] - 0.3, ext[1] + 0.3], y: [ext[0] - 0.3, ext[1] + 0.3], xLabel: 'mean log2 CPM, Centre A (discovery)', yLabel: 'mean log2 CPM, Centre C' });
  Sc.layer('d').append('line').attr('x1', Sc.x(ext[0])).attr('y1', Sc.y(ext[0])).attr('x2', Sc.x(ext[1])).attr('y2', Sc.y(ext[1])).attr('stroke', muted()).attr('stroke-dasharray', '4 4');
  Sc.layer('p').selectAll('circle').data(G.meanA.map((a, i) => ({ a, c: G.meanC[i], t: G.true[i] }))).join('circle')
    .attr('cx', (d) => Sc.x(d.a)).attr('cy', (d) => Sc.y(d.c)).attr('r', 3).attr('fill', muted()).attr('opacity', 0.55)
    .on('mousemove', (e, d) => tip(e, `A ${f2(d.a)} · C ${f2(d.c)}<br>simulated technical shift ${sg(d.t)} log2`)).on('mouseleave', untip);
  Sc.layer('lk').selectAll('circle').data(L.genes.map((g, j) => ({ g, a: G.lock_meanA[j], c: G.lock_meanC[j], j }))).join('circle')
    .attr('cx', (d) => Sc.x(d.a)).attr('cy', (d) => Sc.y(d.c)).attr('r', 7).attr('fill', (d) => cat(1 + d.j)).attr('stroke', css('--surface')).attr('stroke-width', 2)
    .on('mousemove', (e, d) => tip(e, `<b>${d.g}</b><br>raw difference C − A ${sg(d.c - d.a)}<br>true technical shift ${sg(G.lock_true[d.j])}<br>bridging estimate ${sg(G.lock_offset_est[d.j])}`)).on('mouseleave', untip);
  Sc.layer('lt').selectAll('text').data(L.genes.map((g, j) => ({ g, a: G.lock_meanA[j], c: G.lock_meanC[j] }))).join('text')
    .attr('x', (d) => Sc.x(d.a) + 10).attr('y', (d) => Sc.y(d.c) + 4).attr('font-size', 12).attr('fill', ink2()).text((d) => d.g);
  const o2 = {};
  L.genes.forEach((g, j) => { o2[`${g}: raw C−A · true shift · bridging est.`] = `${sg(G.lock_meanC[j] - G.lock_meanA[j])} · ${sg(G.lock_true[j])} · ${sg(G.lock_offset_est[j])}`; });
  o2['response rate discovery / Centre C'] = `${f2(D.resp_rate.d)} / ${f2(D.resp_rate.v)}`;
  s2.set(o2);


  // ---------------- Figure 3: stress test (live): add a technical shift, recompute everything
  const f3g = figure(main, 'Stress test: what if the new assay reads a gene higher or lower?',
    'Add an extra technical shift (log2 units) to one locked gene in every Centre C sample, as a new kit, lab or platform might, and the locked model is re-applied live. Right: predicted probabilities, responders above the axis, non-responders below; dashed line = frozen cut-off.');
  let dg = L.genes.length - 1, delta = 0, useBridge = false;
  select(f3g.controls, { label: 'Shifted gene', value: String(dg), options: L.genes.map((g, j) => [String(j), g]), onChange: (v) => { dg = +v; draw3(); } });
  slider(f3g.controls, { label: 'Extra shift (log2)', min: -1.5, max: 1.5, step: 0.1, value: 0, fmt: (v) => (v >= 0 ? '+' : '') + v.toFixed(1), onInput: (v) => { delta = v; draw3(); } });
  const cb = el('input', { type: 'checkbox' }); cb.addEventListener('change', () => { useBridge = cb.checked; draw3(); });
  f3g.controls.append(el('label', { class: 'control' }, cb, 'correct with bridging samples (they see the shift too)'));
  const s3 = stats(f3g.root);
  const box3 = el('div'); f3g.body.append(box3);
  const Hs = frame(box3, { width: 900, height: 300, x: [0, 1], y: [-1, 1], xLabel: 'predicted probability of response (locked model)', grid: false, margin: { left: 20 } });
  Hs.g.select('.axis.y').remove();
  Hs.g.append('line').attr('x1', 0).attr('x2', Hs.w).attr('y1', Hs.y(0)).attr('y2', Hs.y(0)).attr('stroke', css('--axis'));
  Hs.g.append('text').attr('x', 4).attr('y', 14).attr('fill', ink2()).attr('font-size', 12).text('responders ↑');
  Hs.g.append('text').attr('x', 4).attr('y', Hs.h - 4).attr('fill', ink2()).attr('font-size', 12).text('non-responders ↓');
  Hs.layer('t').append('line').attr('x1', Hs.x(D.thr)).attr('x2', Hs.x(D.thr)).attr('y1', 0).attr('y2', Hs.h).attr('stroke', ink2()).attr('stroke-dasharray', '5 4');
  const yv = D.yv;
  const aucOf = (yy, p) => { let c = 0, t = 0; yy.forEach((a, i) => { if (!a) return; yy.forEach((b, j) => { if (b) return; t++; c += p[i] > p[j] ? 1 : p[i] === p[j] ? 0.5 : 0; }); }); return c / t; };
  const citlOf = (yy, p) => { // Newton on a in logit P = a + logit(p)
    const lp = p.map((v) => Math.log(v / (1 - v))); let a = 0;
    for (let k = 0; k < 30; k++) { let g = 0, h = 0; lp.forEach((l, i) => { const q = 1 / (1 + Math.exp(-(a + l))); g += yy[i] - q; h += q * (1 - q); }); a += g / h; }
    return a;
  };
  function draw3() {
    const p = D.xv.map((x) => {
      let lp = L.intercept;
      x.forEach((v, j) => { let xx = v + (j === dg ? delta : 0); if (useBridge) xx -= D.offset[j] + (j === dg ? delta : 0); lp += L.coef[j] * (xx - L.mu[j]) / L.sd[j]; });
      return 1 / (1 + Math.exp(-lp));
    });
    const w = 0.02, stU = {}, stD = {};
    const pts = p.map((v, i) => { const b = Math.min(49, Math.floor(v / w)), st = yv[i] ? stU : stD; st[b] = (st[b] || 0) + 1; return { v, y: yv[i], b, s: st[b] }; });
    const maxS = Math.max(...Object.values(stU), ...Object.values(stD), 8);
    const dy = Math.min(9, (Hs.h / 2 - 20) / maxS);
    Hs.layer('d').selectAll('circle').data(pts).join('circle').attr('cx', (d) => Hs.x((d.b + 0.5) * w)).attr('cy', (d) => Hs.y(0) + (d.y ? -1 : 1) * (6 + (d.s - 0.5) * dy))
      .attr('r', Math.max(2.2, dy / 2 - 0.3)).attr('fill', (d) => cat(d.y)).attr('stroke', css('--surface')).attr('stroke-width', 0.8)
      .on('mousemove', (e, d) => tip(e, `${d.y ? 'responder' : 'non-responder'}<br>predicted ${f3(d.v)}`)).on('mouseleave', untip);
    const P1 = d3.sum(yv), N0 = yv.length - P1;
    const sens = d3.sum(pts, (d) => d.y && d.v >= D.thr) / P1, spec = d3.sum(pts, (d) => !d.y && d.v < D.thr) / N0;
    s3.set({ AUC: f3(aucOf(yv, p)), 'mean predicted vs observed': `${f2(d3.mean(p))} vs ${f2(P1 / yv.length)}`, 'calibration-in-the-large': sg(citlOf(yv, p)),
      'sensitivity / specificity at frozen cut-off': `${f2(sens)} / ${f2(spec)}`, 'patients flagged positive': `${d3.sum(p, (v) => v >= D.thr)} of ${yv.length}` });
  }
  draw3();
  legend(f3g.root, [['responder', cat(1)], ['non-responder', cat(0)]]);

  // ---------------- prose
  const fr = S.frozen, br = S.bridge, rz = S.rez, rc = S.recal, it = S.internal, ap = S.apparent;
  const mixTxt = (m) => `${m.inflamed} inflamed / ${m.excluded} excluded / ${m.desert} desert`;
  const lpShift = L.genes.map((g, j) => L.coef[j] * G.lock_true[j] / L.sd[j]);
  const netShift = d3.sum(lpShift);
  section(main, 'What to notice', `<ul>
    <li>The ladder: in-sample AUC ${f3(ap.auc)}, internal cross-validated ${f3(it.auc)}, external frozen ${f3(fr.auc)} (95% CI ${f2(fr.ci[0])}–${f2(fr.ci[1])}). ${fr.auc > it.auc ? `The external AUC is <i>higher</i> than internal. That is not proof of a better model: Centre C enrolled more desert tumours (${mixTxt(D.mix_v)}), which are easy to call as non-responders, and with ${D.n_v} patients the CI is wide. Discrimination depends on case mix (spectrum).` : 'The CI is wide: ' + D.n_v + ' patients is a small validation set.'}</li>
    <li><b>Discrimination barely depends on the preprocessing choice</b> (frozen ${f3(fr.auc)}, bridged ${f3(br.auc)}, re-standardised ${f3(rz.auc)}). A per-gene offset moves every patient's score by the same amount, which leaves the ranking, and hence the AUC, unchanged.</li>
    <li>In this cohort the real Centre C shift is mild for the locked genes (${L.genes.map((g, j) => `${g} ${sg(G.lock_true[j])}`).join(', ')} log2), and the effects partly cancel in the linear predictor (net ${sg(netShift)} log-odds). Frozen as-is, mean predicted ${f2(fr.mean_p)} vs observed ${f2(fr.obs)}; with the bridging offset ${f2(br.mean_p)} vs ${f2(br.obs)}. Note that 12 bridging samples estimate the offset only roughly (${L.genes.map((g, j) => `${g}: estimated ${sg(G.lock_offset_est[j])}, true ${sg(G.lock_true[j])}`).join('; ')}); more bridging samples, or commercial reference RNA run in every batch, tighten it. A two-gene model was lucky here; you cannot count on it.</li>
    <li>Use the <b>stress test</b>: shift ${L.genes[L.genes.length - 1]} by +1 log2 (a twofold change in reads, ordinary between kits or labs). AUC does not move, but the mean prediction and the number of patients above the frozen cut-off jump. Tick the bridging box: the 12 bridging samples see the same shift and remove it, without touching the model or the outcomes.</li>
    <li><b>Re-standardising within Centre C</b> looks attractive but is not a locked model: z-scores now depend on who else is in the cohort, so the case mix leaks into each patient's score (mean predicted ${f2(rz.mean_p)} vs ${f2(rz.obs)} observed), and a single new patient cannot be scored at all.</li>
    <li><b>Recalibration</b> (intercept and slope re-estimated on Centre C outcomes, out-of-fold) gives calibration-in-the-large ${sg(rc.citl)}. That is legitimate <i>model updating</i>, but then Centre C is no longer an untouched validation cohort for the updated model.</li>
  </ul>`);
  steps(main, 'How it works', [
    '<b>Lock</b> everything that turns raw data into a decision: gene list, normalisation, scaling parameters, coefficients, cut-off, and how missing genes or failed QC are handled. Write it down (and ideally deposit code and parameters) before unblinding the validation outcomes.',
    '<b>Process the new samples</b> exactly as specified. If the platform or centre differs, decide in advance how to handle it (bridging samples, reference standards) without using the validation outcomes.',
    '<b>Discrimination</b>: AUC (or C-index) with a confidence interval. <b>Calibration</b>: calibration-in-the-large (are predictions too high or low on average?), calibration slope (too extreme or too timid?), and a calibration plot.',
    '<b>Clinical utility</b>: at the frozen cut-off, sensitivity, specificity and predictive values; decision-curve analysis asks whether acting on the model beats "treat all" or "treat none".',
    'If performance holds, that is evidence for <b>this</b> model in <b>this</b> kind of population. Changes to the model (recalibration, refitting) create a new model that needs its own validation.',
  ]);
  math(main, 'calibration-in-the-large:  logit P(y=1) = a + logit(p̂)        ideal a = 0\ncalibration slope:         logit P(y=1) = a + b · logit(p̂)    ideal b = 1\nbridging offset (gene g):  δ_g = mean_bridge[ x_g(Centre C) − x_g(Centre A) ],  x′ = x − δ');
  main.append(el('h2', {}, 'What "validated" means'));
  main.append(html(`<table class="simple prose"><tr><th>Level</th><th>Evidence</th><th>This study</th></tr>
    <tr><td>Apparent</td><td>performance on the training data</td><td>AUC ${f3(ap.auc)}: not evidence</td></tr>
    <tr><td>Internal</td><td>cross-validation / bootstrap within the discovery cohort (TRIPOD type 1b)</td><td>AUC ${f3(it.auc)}</td></tr>
    <tr><td>Temporal / geographic external</td><td>new patients from a different time or centre, model frozen (TRIPOD types 2b–3)</td><td>Centre C: AUC ${f3(fr.auc)}, calibration-in-the-large ${sg(fr.citl)} frozen, ${sg(br.citl)} bridged</td></tr>
    <tr><td>Clinical validity &amp; utility</td><td>prospective use, ideally a randomised biomarker-stratified trial; analytical validity of the assay</td><td>not done; a single-arm cohort cannot show the marker predicts <i>benefit</i> (step 9)</td></tr></table>`));
  callout(main, '', 'In biology', 'Reporting follows <b>TRIPOD+AI</b> (prediction models) and <b>REMARK</b> (tumour markers); risk of bias is judged with <b>PROBAST</b>. Calibration and validation statistics: <code>rms</code> (<code>val.prob</code>), <code>pmcalibration</code>, <code>CalibrationCurves</code> in R; <code>scikit-learn</code> <code>CalibrationDisplay</code> in Python. Real examples of locking: commercial expression assays (Oncotype DX, Prosigna/PAM50) freeze genes, normalisers and algorithm and use reference samples on every run; the T-cell-inflamed GEP for pembrolizumab was locked on NanoString before being tested in independent trials. Cross-platform transfer (RNA-seq to NanoString or qPCR) is its own validation step.');
  callout(main, 'warn', 'Pitfalls', `<ul>
    <li><b>"Validated" after re-tuning</b>: re-selecting genes, re-fitting or re-choosing the cut-off on the validation cohort makes it a second training set.</li>
    <li><b>Reporting only AUC</b>: AUC is blind to a constant shift in the score; the stress test above breaks calibration and the cut-off while AUC stays put.</li>
    <li><b>Cohort-wise normalisation</b> (z-scoring, ComBat, quantile normalisation across the validation set) uses the other validation patients and is not available for one new patient.</li>
    <li><b>Small validation sets</b>: a CI of ${f2(fr.ci[0])}–${f2(fr.ci[1])} cannot distinguish a good biomarker from a mediocre one. Plan the validation sample size (Riley et al. recommend at least 100–200 events).</li>
    <li><b>Spectrum / case-mix differences</b> change PPV and NPV even when the model is perfect; report the validation cohort's composition.</li>
    <li><b>Predictive claims</b> need randomised evidence of a treatment interaction, not just good discrimination in treated patients.</li>
  </ul>`);
  seeAlso(main, [['omics', 'bulk-batch', 'Batch effects'], ['omics', 'bulk-model-compare', 'Calibration in model comparison'], ['concepts', 'calibration', 'Calibration'], ['omics', 'bulk-survival', 'Prognostic vs predictive'], ['omics', 'sc-qc', 'Next: single-cell QC']]);
}
