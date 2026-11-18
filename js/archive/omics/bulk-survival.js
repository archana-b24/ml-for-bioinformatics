// Bulk step 9: Cox-lasso risk score, Kaplan-Meier curves, and the prognostic-vs-predictive trap
// (needs a control arm: a marker that separates survival equally in both arms is prognostic, not predictive).
import { header, figure, slider, select, stats, section, steps, callout, math, legend, seeAlso, tip, untip, el } from '../../lib/ui.js';
import { frame, cat, css, ink2, muted } from '../../lib/plot.js';
import { stepCard, omicsData } from '../../omics/_shared.js';
const d3 = window.d3;
const f2 = (v) => v.toFixed(2), f3 = (v) => v.toFixed(3), f1 = (v) => v.toFixed(1);
const pfmt = (p) => (p < 0.001 ? p.toExponential(1) : p.toFixed(3));

// ---- survival maths (small, exact) ----
function km(t, e) {
  const o = d3.range(t.length).sort((a, b) => t[a] - t[b]);
  let atRisk = t.length, s = 1; const pts = [{ t: 0, s: 1, n: atRisk }]; const cens = [];
  let k = 0;
  while (k < o.length) {
    const tt = t[o[k]]; let d = 0, c = 0;
    while (k < o.length && t[o[k]] === tt) { if (e[o[k]]) d++; else c++; k++; }
    if (d) { s *= 1 - d / atRisk; pts.push({ t: tt, s, n: atRisk }); }
    if (c) cens.push({ t: tt, s });
    atRisk -= d + c;
  }
  return { pts, cens };
}
function medianOf(curve) { const p = curve.pts.find((q) => q.s <= 0.5); return p ? p.t : null; }
function erfc(x) { // Abramowitz-Stegun 7.1.26
  const z = Math.abs(x), t = 1 / (1 + 0.5 * z);
  const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
  return x >= 0 ? r : 2 - r;
}
function logrank(t, e, g) { // g: 0/1 groups
  const times = [...new Set(t.filter((_, i) => e[i]))].sort((a, b) => a - b);
  let O1 = 0, E1 = 0, Vv = 0, O0 = 0;
  times.forEach((tt) => {
    let n1 = 0, n0 = 0, d1 = 0, d0 = 0;
    for (let i = 0; i < t.length; i++) if (t[i] >= tt) { if (g[i]) { n1++; if (t[i] === tt && e[i]) d1++; } else { n0++; if (t[i] === tt && e[i]) d0++; } }
    const N = n1 + n0, d = d1 + d0; if (N < 2) return;
    E1 += d * n1 / N; O1 += d1; O0 += d0; Vv += d * (n1 / N) * (n0 / N) * (N - d) / (N - 1);
  });
  const chi = (O1 - E1) ** 2 / Vv, E0 = O1 + O0 - E1;
  return { chi, p: erfc(Math.sqrt(chi / 2)), hr: (O1 / E1) / (O0 / E0) };
}

export default async function render(main) {
  header(main, 'Survival & risk scores',
    'Responders live longer, so a gene score that predicts survival is easy to find. The hard question is whether it tells you <b>who benefits from the drug</b> (predictive) or only <b>who does well anyway</b> (prognostic). With a single treated cohort you cannot tell them apart.');
  const D = await omicsData(main, 'bulk_survival.json');
  if (!D) return;
  const IO = D.io, CT = D.ctrl, I = D.inter;

  stepCard(main, {
    stage: 'Bulk RNA-seq · step 9 of 11',
    question: 'Which tumours progress early on anti-PD-1, and is that a property of the drug response or of the tumour\'s biology regardless of treatment?',
    data: `Progression-free survival (PFS) for ${D.n_io} anti-PD-1-treated discovery patients and, for the treatment-interaction question, a randomised docetaxel control arm (n = ${D.n_ctrl}, simulated with the same biology but no immunotherapy benefit).`,
    method: `Cox proportional-hazards model with an elastic-net penalty (α = ${D.path.alpha}${D.path.alpha === 1 ? ', the pure lasso' : ', a lasso–ridge mix'}; λ at the minimum cross-validated partial-likelihood deviance) to build a risk score;`+' Kaplan–Meier curves and log-rank tests; Cox models with a treatment × biomarker interaction.',
    verdict: 'both',
    why: 'Penalised Cox is ML-style variable selection on thousands of genes; whether a marker is predictive is a classical statistical question that only a randomised comparison with an interaction test can answer.',
  });

  // ---------------- Figure 1: Cox-lasso path
  const P = D.path, md = D.model;
  const f1g = figure(main, 'Building the risk score: the penalised Cox path',
    'Left: cross-validated partial-likelihood deviance (lower = better) along the penalty path, with ±1 SE bars; the number of genes in the model is on top. The large orange marker is the chosen λ (minimum deviance). Weaker penalties than shown (more genes) overfit badly and are cut off. Right: the genes it keeps and their coefficients (positive = higher hazard = earlier progression).');
  const row1 = el('div', { class: 'fig-row' }); f1g.body.append(row1);
  const L1 = el('div'), R1 = el('div'); row1.append(L1, R1);
  // weak penalties overfit so badly that their deviance would flatten the interesting region: show the path up to
  // the point where CV deviance exceeds 1.3x the null-model deviance
  const keepN = Math.max(P.jsel + 3, P.dev.findIndex((d) => d > P.dev[0] * 1.3) === -1 ? P.dev.length : P.dev.findIndex((d) => d > P.dev[0] * 1.3));
  const ll = P.lam.slice(0, keepN).map(Math.log10);
  const dl = d3.extent(P.dev.slice(0, keepN).flatMap((d, i) => [d - P.se[i], d + P.se[i]]));
  const A = frame(L1, { width: 440, height: 360, x: [d3.max(ll) + 0.05, d3.min(ll) - 0.05], y: [dl[0] - 2, dl[1] + 2], xLabel: 'log10 λ  (← stronger penalty · weaker →)', yLabel: 'CV deviance', margin: { top: 26 } });
  const pts1 = ll.map((x, i) => ({ x, d: P.dev[i], s: P.se[i], k: P.nnz[i], i }));
  A.layer('e').selectAll('line').data(pts1).join('line').attr('x1', (d) => A.x(d.x)).attr('x2', (d) => A.x(d.x)).attr('y1', (d) => A.y(d.d - d.s)).attr('y2', (d) => A.y(d.d + d.s)).attr('stroke', muted());
  A.layer('p').selectAll('circle').data(pts1).join('circle').attr('cx', (d) => A.x(d.x)).attr('cy', (d) => A.y(d.d)).attr('r', (d) => (d.i === P.jsel ? 7 : 4.5))
    .attr('fill', (d) => (d.i === P.jsel ? cat(1) : cat(0))).attr('stroke', css('--surface')).attr('stroke-width', 1.5)
    .on('mousemove', (e, d) => tip(e, `log10 λ = ${f2(d.x)}<br>CV deviance ${f1(d.d)} ± ${f1(d.s)}<br>${d.k} genes${d.i === P.jmin ? '<br>(minimum)' : ''}${d.i === P.j1se ? '<br>(1-SE rule would pick this)' : ''}${d.i === P.jsel ? '<br>(chosen)' : ''}`)).on('mouseleave', untip);
  A.layer('k').selectAll('text').data(pts1.filter((d, i) => i % 3 === 0)).join('text').attr('x', (d) => A.x(d.x)).attr('y', -10).attr('text-anchor', 'middle').attr('font-size', 10).attr('fill', muted()).text((d) => d.k);
  const ng = md.genes.length;
  const cmax = d3.max(md.coef.map(Math.abs)) * 1.2 || 1;
  const order = d3.range(ng).sort((a, b) => md.coef[b] - md.coef[a]);
  const Bc = frame(R1, { width: 440, height: 360, x: [-cmax, cmax], y: [0, Math.max(ng, 6)], xLabel: 'Cox coefficient (log hazard ratio per SD)', grid: false, margin: { left: 80 } });
  Bc.g.select('.axis.y').remove();
  const bh = Bc.h / Math.max(ng, 6);
  const grpCol = (g) => (g === 'PROLIF module' ? cat(1) : g === 'IFNG module' ? cat(0) : g === 'CAF/TGF-b module' ? cat(2) : muted());
  Bc.layer('z').append('line').attr('x1', Bc.x(0)).attr('x2', Bc.x(0)).attr('y1', 0).attr('y2', Bc.h).attr('stroke', muted());
  order.forEach((j, r) => {
    Bc.layer('b').append('rect').attr('x', Bc.x(Math.min(0, md.coef[j]))).attr('width', Math.abs(Bc.x(md.coef[j]) - Bc.x(0))).attr('y', r * bh + 3).attr('height', bh - 6).attr('rx', 2)
      .attr('fill', grpCol(md.group[j])).on('mousemove', (e) => tip(e, `<b>${md.genes[j]}</b> (${md.group[j]})<br>coef ${f3(md.coef[j])} · HR per SD ${f2(Math.exp(md.coef[j]))}`)).on('mouseleave', untip);
    Bc.layer('t').append('text').attr('x', -6).attr('y', r * bh + bh / 2 + 4).attr('text-anchor', 'end').attr('font-size', 12).attr('fill', ink2()).text(md.genes[j]);
  });
  legend(f1g.root, [['IFN-γ module', cat(0)], ['proliferation module', cat(1)], ['CAF/TGF-β module', cat(2)], ['other', muted()]]);

  // ---------------- Figure 2: KM explorer
  const f2g = figure(main, 'Kaplan–Meier curves: split patients by a score',
    'Patients above the chosen quantile of the score form the high group. Ticks are censored patients (still progression-free at last follow-up). Compare the in-sample risk score with the cross-validated one, and the two arms.');
  const SC = {
    risk_in: ['Penalised Cox risk score, in-sample (optimistic)', 'risk_in'],
    risk_cv: ['Penalised Cox risk score, cross-validated', 'risk_cv'],
    ifng: ['IFN-γ module score', 'ifng'],
    prolif: ['Proliferation module score', 'prolif'],
  };
  let sk = 'risk_cv', ak = 'io', q = 0.5;
  select(f2g.controls, { label: 'Score', value: sk, options: Object.entries(SC).map(([k, v]) => [k, v[0]]), onChange: (v) => { sk = v; draw2(); } });
  select(f2g.controls, { label: 'Arm', value: ak, options: [['io', `anti-PD-1 (n=${D.n_io})`], ['ctrl', `docetaxel control (n=${D.n_ctrl})`]], onChange: (v) => { ak = v; draw2(); } });
  slider(f2g.controls, { label: 'Split at quantile', min: 0.2, max: 0.8, step: 0.05, value: q, fmt: f2, onInput: (v) => { q = v; draw2(); } });
  legend(f2g.root, [['low score', cat(0)], ['high score', cat(1)]]);
  const s2 = stats(f2g.root);
  const tmax = Math.ceil(d3.max([...IO.t, ...CT.t]) / 6) * 6;
  const kmBox = el('div'); f2g.body.append(kmBox);
  const KM = frame(kmBox, { width: 900, height: 360, x: [0, tmax], y: [0, 1], xLabel: 'months since start of treatment', yLabel: 'progression-free' });
  function scoreOf(arm, key) {
    if (arm === 'io') return IO[SC[key][1]];
    return key.startsWith('risk') ? CT.risk : CT[key];
  }
  function drawKM(Fr, t, e, g) {
    const curves = [0, 1].map((gg) => { const idx = d3.range(t.length).filter((i) => g[i] === gg); return km(idx.map((i) => t[i]), idx.map((i) => e[i])); });
    Fr.layer('c').selectAll('path').data(curves).join('path').attr('fill', 'none').attr('stroke', (c, i) => cat(i)).attr('stroke-width', 2)
      .attr('d', (c) => d3.line().curve(d3.curveStepAfter).x((d) => Fr.x(d.t)).y((d) => Fr.y(d.s))([...c.pts, { t: d3.max(t), s: c.pts[c.pts.length - 1].s }]));
    const ticks = curves.flatMap((c, i) => c.cens.map((d) => ({ ...d, i })));
    Fr.layer('cens').selectAll('line').data(ticks).join('line').attr('x1', (d) => Fr.x(d.t)).attr('x2', (d) => Fr.x(d.t)).attr('y1', (d) => Fr.y(d.s) - 4).attr('y2', (d) => Fr.y(d.s) + 4)
      .attr('stroke', (d) => cat(d.i)).attr('stroke-width', 1.5);
    Fr.layer('hit').selectAll('path').data(curves).join('path').attr('fill', 'none').attr('stroke', 'transparent').attr('stroke-width', 12)
      .attr('d', (c) => d3.line().curve(d3.curveStepAfter).x((d) => Fr.x(d.t)).y((d) => Fr.y(d.s))(c.pts))
      .on('mousemove', (ev, c) => { const [mx] = d3.pointer(ev, Fr.g.node()); const tt = Fr.x.invert(mx); const p = c.pts.filter((d) => d.t <= tt).pop(); tip(ev, `${curves.indexOf(c) ? 'high' : 'low'} score<br>${f1(tt)} months: ${(100 * p.s).toFixed(0)}% progression-free`); })
      .on('mouseleave', untip);
    return curves;
  }
  function split(s, qq) { const thr = d3.quantile(s.slice().sort(d3.ascending), qq); return s.map((v) => (v > thr ? 1 : 0)); }
  function draw2() {
    const arm = ak === 'io' ? IO : CT, s = scoreOf(ak, sk);
    const g = split(s, q);
    const curves = drawKM(KM, arm.t, arm.e, g);
    const lr = logrank(arm.t, arm.e, g);
    const med = curves.map(medianOf);
    const o = {
      'median PFS low / high': med.map((m) => (m == null ? 'not reached' : f1(m) + ' mo')).join(' / '),
      'log-rank χ²': f1(lr.chi), 'p': pfmt(lr.p), 'hazard ratio high vs low (O/E)': f2(lr.hr),
      'n low / high': `${d3.sum(g, (v) => 1 - v)} / ${d3.sum(g)}`,
    };
    if (sk.startsWith('risk')) o['C-index'] = f3(ak === 'ctrl' ? D.cindex.ctrl : D.cindex[sk === 'risk_in' ? 'in' : 'cv']);
    s2.set(o);
  }
  draw2();

  // ---------------- Figure 3: prognostic vs predictive (2 x 2)
  const f3g = figure(main, 'Prognostic or predictive? The same split in both arms',
    'Each panel splits one arm at its median score. A <b>prognostic</b> marker separates the curves by a similar amount in both arms. A <b>predictive</b> marker separates them only in the arm whose drug it is linked to. The interaction p-value tests exactly that difference.');
  const grid = el('div', { class: 'fig-row' }); f3g.body.append(grid);
  const cell = (title) => { const d = el('div'); d.append(el('div', { class: 'fig-title', style: 'font-size:13.5px;font-weight:600' }, title)); grid.append(d); return d; };
  const panels = [];
  [['ifng', 'IFN-γ score'], ['prolif', 'Proliferation score']].forEach(([key, lab]) => {
    [['io', 'anti-PD-1'], ['ctrl', 'docetaxel']].forEach(([arm, al]) => {
      const box = cell(`${lab} · ${al}`);
      const Fr = frame(box, { width: 440, height: 250, x: [0, tmax], y: [0, 1], xLabel: 'months', yLabel: 'progression-free' });
      const A2 = arm === 'io' ? IO : CT, g = split(A2[key], 0.5);
      drawKM(Fr, A2.t, A2.e, g);
      const lr = logrank(A2.t, A2.e, g);
      Fr.g.append('text').attr('x', Fr.w - 4).attr('y', 14).attr('text-anchor', 'end').attr('font-size', 12).attr('fill', ink2()).text(`HR high vs low ${f2(lr.hr)}, log-rank p = ${pfmt(lr.p)}`);
      panels.push({ key, arm, lr });
    });
  });
  legend(f3g.root, [['below median', cat(0)], ['above median', cat(1)]]);
  const s3 = stats(f3g.root);
  s3.set({
    'IFN-γ: HR per SD, anti-PD-1 / docetaxel': `${f2(I.ifng.hr_io)} / ${f2(I.ifng.hr_ctrl)}`, 'IFN-γ × arm interaction p': pfmt(I.ifng.p[2]),
    'Proliferation: HR per SD, anti-PD-1 / docetaxel': `${f2(I.prolif.hr_io)} / ${f2(I.prolif.hr_ctrl)}`, 'Proliferation × arm interaction p': pfmt(I.prolif.p[2]),
  });

  // ---------------- prose
  const sgnOf = (grp) => { const c = md.coef.filter((v, j) => md.group[j] === grp); return c.length ? (c.every((v) => v > 0) ? 'all positive' : c.every((v) => v < 0) ? 'all negative' : 'mixed') : null; };
  const signTxt = [sgnOf('PROLIF module') && `Proliferation genes: coefficients ${sgnOf('PROLIF module')} (positive = more proliferation, earlier progression).`, sgnOf('IFNG module') && `Interferon genes: ${sgnOf('IFNG module')} (negative = they mark responders, who progress later).`].filter(Boolean).join(' ');
  const nProl = md.group.filter((g) => g === 'PROLIF module').length, nIfn = md.group.filter((g) => g === 'IFNG module').length, nFill = md.group.filter((g) => g === 'filler').length;
  const prolInAll = D.outer_genes.filter((gs) => gs.some((g) => ['MKI67', 'TOP2A', 'CCNB1', 'CDK1', 'BIRC5', 'UBE2C', 'TPX2', 'MCM2'].includes(g))).length;
  section(main, 'What to notice', `<ul>
    <li>At the chosen λ the penalised Cox model keeps <b>${ng} genes</b>: ${nProl} from the proliferation module and ${nIfn} from the IFN-γ module${ng - nProl - nIfn ? `, plus ${ng - nProl - nIfn} others, ${nFill} of them anonymous filler genes with no planted effect (false positives)` : ''}. ${signTxt} Proliferation genes were selected in ${prolInAll} of the ${D.outer_genes.length} outer CV folds.</li>
    <li>Switch the score between <b>in-sample</b> and <b>cross-validated</b>: the C-index drops from ${f3(D.cindex.in)} to ${f3(D.cindex.cv)}. A KM plot of a risk score evaluated on the patients it was trained on is optimistic; the honest version uses out-of-fold scores.</li>
    <li>In the anti-PD-1 arm, the <b>IFN-γ score</b> gives HR ${f2(I.ifng.hr_io)} per SD, but in the docetaxel arm ${f2(I.ifng.hr_ctrl)}: interaction p = ${pfmt(I.ifng.p[2])}. It is <b>predictive</b> of anti-PD-1 benefit.</li>
    <li>The <b>proliferation score</b> gives HR ${f2(I.prolif.hr_io)} on anti-PD-1 and ${f2(I.prolif.hr_ctrl)} on docetaxel; interaction p = ${pfmt(I.prolif.p[2])}. Same direction in both arms: <b>prognostic</b>, not predictive. (Its bulk score is somewhat lower in responders, AUC ${f3(I.prolif.auc_resp)} for response, but only because proliferation genes come from tumour cells and so also measure tumour purity; the planted program itself is independent of response. The interaction test is what settles it.)</li>
    <li>Applied to the control arm, the locked risk score still has C-index ${f3(D.cindex.ctrl)}: it carries the proliferation (prognostic) part. A trial that enrolled "high-risk" patients by this score would select poor-prognosis tumours, not anti-PD-1 responders.</li>
    <li>Move the <b>split quantile</b>: the p-value changes with the cut-off. Picking the cut-off that gives the smallest p is a classic, inflating mistake (the "optimal cut-point" problem).</li>
  </ul>`);
  steps(main, 'How it works', [
    '<b>Censoring</b>: patients still progression-free at last contact contribute "at least this long" (time, event = 0). Survival methods use that partial information instead of dropping them.',
    '<b>Kaplan–Meier</b>: at each event time multiply the running survival by (1 − events / patients still at risk). The <b>log-rank test</b> compares observed and expected events in each group at every event time.',
    '<b>Cox model</b>: hazard(t | x) = baseline(t) · exp(β·x). The partial likelihood compares each patient who progresses with everyone still at risk at that moment, so the baseline cancels.',
    '<b>Penalised Cox</b>: maximise partial likelihood − λ·[α·Σ|β| + (1−α)/2·Σβ²] over all genes (α = 1 is the lasso; α = 0.5 keeps groups of correlated genes together). Fitted here by proximal gradient; glmnet uses coordinate descent. Choose λ by cross-validated partial likelihood, inside each outer CV fold.',
    '<b>Evaluate honestly</b>: nested CV gives each patient a risk score from a model that did not see them; report the C-index (the probability that, of two comparable patients, the one with higher risk progresses first).',
    '<b>Predictive?</b> Fit hazard ~ arm + marker + arm × marker in a randomised trial. Only a significant (and pre-specified) interaction supports "this marker predicts benefit from this drug".',
  ]);
  math(main, 'h(t | x) = h0(t) · exp(β·x)                         HR per unit = exp(β)\npartial log-lik: ℓ(β) = Σ_events [ β·x_i − log Σ_{j: t_j ≥ t_i} exp(β·x_j) ]\nelastic net:     maximise ℓ(β) − λ [ α Σ|β_k| + (1−α)/2 Σ β_k² ]\npredictive test: h = h0 · exp(β1·arm + β2·marker + β3·arm·marker),  H0: β3 = 0');
  callout(main, '', 'In biology', '<code>glmnet</code> (R, <code>family="cox"</code>) and <code>scikit-survival</code> (<code>CoxnetSurvivalAnalysis</code>, random survival forests) build gene-expression risk scores; <code>survival</code>/<code>survminer</code> in R and <code>lifelines</code> in Python for Kaplan–Meier, log-rank and Cox. The prognostic-vs-predictive distinction is central to immunotherapy biomarkers: tumour mutational burden and IFN-γ/T-cell-inflamed signatures are judged by treatment interaction in randomised trials (e.g. atezolizumab vs docetaxel in OAK/POPLAR), whereas proliferation signatures in many cancers are prognostic whatever the treatment. Reporting follows REMARK (tumour-marker prognostic studies).');
  callout(main, 'warn', 'Pitfalls', `<ul>
    <li><b>Single-arm cohort</b>: in treated patients alone, "survival differs by marker" is compatible with prognostic and predictive. You need a comparator arm and an interaction test.</li>
    <li><b>Response and survival are entangled</b>: responders live longer, so anything that predicts response also looks prognostic in the treated arm.</li>
    <li><b>Optimised cut-points</b>: scanning thresholds for the smallest log-rank p inflates false positives; pre-specify (e.g. median) or use the continuous score.</li>
    <li><b>Resubstitution KM curves</b>: training a risk score and plotting it on the same patients (C-index ${f3(D.cindex.in)} here) overstates separation.</li>
    <li><b>Proportional hazards</b> is an assumption: immunotherapy curves often cross or plateau (a cured tail), which a single HR summarises poorly.</li>
    <li><b>Informative censoring</b> (patients lost because they got sicker) biases KM upward.</li>
  </ul>`);
  seeAlso(main, [['omics', 'bulk-feature-selection', 'Lasso feature selection'], ['concepts', 'regularization', 'L1 / L2 regularization'], ['concepts', 'cross-validation', 'Cross-validation'], ['omics', 'bulk-deconvolution', 'Next: cell-type deconvolution']]);
}
