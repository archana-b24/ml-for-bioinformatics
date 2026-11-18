// Linear regression, lesson format.
// Arc: what → why → where → how it works (6 steps: residuals and squares → the fitted line, R² and uncertainty → residual checks →
//      several predictors → confounding by centre → model builder with validation) → pitfalls → recap.
// Outcome: CD8 T-cell % of the tumour. Every number in the prose is computed from data/study/cohort.json at render time.
import { header, figure, slider, select, button, checkbox, stats, legend, seeAlso, tip, untip, el, html } from '../lib/ui.js';
import { frame, cat, css, ink, ink2, muted, surface, sequential } from '../lib/plot.js';
import { rng, mean, sd, range } from '../lib/data.js';
import { fitLinear, predictLinear, t975, qNorm, standardize, pNormal } from '../lib/models.js';
import { loadCohort, FEATURES, studyCard } from '../lib/study.js';
import { lessonNav, chapter, prose, rung, worked, quiz, cards, table, code, recap } from '../lib/lesson.js';
const d3 = window.d3;

const um = (t) => t.replace(/^-/, '−');
const f1 = (v) => um(v.toFixed(1));
const f2 = (v) => (Math.abs(v) < 0.005 ? '0.00' : um(v.toFixed(2)));
const f3 = (v) => um(v.toFixed(3));
const pc = (v) => Math.round(100 * v) + '%';
const sgn = (v) => (v < 0 ? '− ' + f2(-v) : '+ ' + f2(v));
const pFmt = (p) => (p < 0.001 ? '< 0.001' : p.toFixed(3));
const sum = (a) => a.reduce((s, v) => s + v, 0);
const L2 = Math.log2;

export default async function render(main) {
  const C = await loadCohort();
  const D = C.discovery, V = C.validation, n = D.length;
  const xs = D.map((p) => p.ifng), ys = D.map((p) => p.cd8);
  const Y = (p, ln) => (ln ? L2(p.cd8) : p.cd8);
  const fit1 = fitLinear(xs.map((x) => [x]), ys), b1 = fit1.w[0], a1 = fit1.b0;
  const mx = mean(xs), my = mean(ys), sxx = sum(xs.map((x) => (x - mx) ** 2)), sxy = sum(xs.map((x, i) => (x - mx) * (ys[i] - my)));
  const YM = Math.ceil(d3.max(ys) / 5) * 5, XR = [2.2, 10.2];
  const r1 = corr(xs, ys);
  const yl = D.map((p) => L2(p.cd8));
  const fitL = fitLinear(xs.map((x) => [x]), yl);

  header(main, 'Linear regression',
    'A model that predicts a <b>number</b>, such as the percentage of CD8 T cells in a tumour, from one or more measurements. It draws the straight line (or flat plane) that leaves the <b>smallest total squared error</b>, and the slope of that line is the estimated change in the outcome per unit of the measurement.');
  lessonNav(main);
  studyCard(main, C, '<b>Question for this page:</b> how does the CD8 T-cell fraction of a tumour change with its IFN-γ score, how well can it be predicted, and which other measurements carry additional information?');

  // =====================================================================================
  // 1. WHAT
  // =====================================================================================
  let ch = chapter(main, 'what', 'What is linear regression?', 'In one line: <b>outcome ≈ intercept + slope × measurement</b>, with the line chosen to minimise squared errors.');
  prose(ch, `<p>Linear regression describes how the average of a continuous <b>outcome</b> changes with one or more <b>predictors</b> (also called features or covariates). For one predictor the model is a straight line, <i>ŷ</i> = <i>a</i> + <i>b</i>·<i>x</i>. In the study, the outcome is the percentage of cells in the biopsy that are CD8 T cells, and the predictor is the IFN-γ score. The fitted line for the ${n} discovery patients is <b>CD8 ≈ ${f1(a1)} ${sgn(b1)} × IFN-γ</b>. The annotated plot shows its parts.</p>`);
  const exPt = D.filter((p) => p.ifng > 6.8 && p.ifng < 8.2).sort((p, q) => (q.cd8 - a1 - b1 * q.ifng) - (p.cd8 - a1 - b1 * p.ifng))[0];
  ch.append(anatomy());
  prose(ch, '<p>Key terms:</p>');
  table(ch, [
    { t: 'Outcome (response variable, <i>y</i>)', m: 'The number to be predicted', s: 'CD8 T cells, % of cells' },
    { t: 'Predictor (feature, <i>x</i>)', m: 'A measurement used to predict the outcome', s: 'IFN-γ score (log2 expression)' },
    { t: 'Slope (coefficient, <i>b</i>)', m: 'Average change in the outcome per one-unit increase of the predictor', s: `+${f2(b1)} CD8 percentage points per IFN-γ unit` },
    { t: 'Intercept (<i>a</i>)', m: 'Fitted value when every predictor is 0', s: `${f1(a1)}: an extrapolation, since no patient has IFN-γ near 0` },
    { t: 'Fitted value (<i>ŷ</i>)', m: 'The point on the line for a given <i>x</i>', s: `${f1(a1 + b1 * exPt.ifng)}% for ${exPt.id} (IFN-γ ${f2(exPt.ifng)})` },
    { t: 'Residual (<i>e</i>)', m: 'Observed minus fitted: what the line fails to explain', s: `${exPt.id} observed ${f1(exPt.cd8)}%, residual ${sgn(exPt.cd8 - a1 - b1 * exPt.ifng)}` },
    { t: 'R²', m: 'Fraction of the variance of <i>y</i> explained by the model', s: `${f2(fit1.r2)}: ${pc(fit1.r2)} of the variation in CD8 %` },
    { t: 'Standard error (SE)', m: 'Typical sampling error of an estimated coefficient', s: `slope ${f2(b1)} ± ${f2(fit1.se[1])}` },
  ], [['t', 'Term'], ['m', 'Meaning'], ['s', 'In the study']]);

  // =====================================================================================
  // 2. WHY
  // =====================================================================================
  ch = chapter(main, 'why', 'Why do we need it?', 'What a model for a continuous outcome should provide, and where simpler summaries fall short.');
  prose(ch, `<p>For a continuous outcome, a useful analysis should:</p><ol>
    <li><b>Quantify the effect in the units of the outcome</b> ("CD8 rises by this many percentage points per IFN-γ unit"), not only whether an effect exists.</li>
    <li><b>Adjust for other measurements</b>, so that the effect of one variable is separated from those of others.</li>
    <li><b>Report uncertainty</b> for the effect and for a prediction for a new patient.</li></ol>
    <p>Two common simpler summaries each fail at least one of these.</p>`);
  const med = d3.median(xs), hi = D.filter((p) => p.ifng >= med), lo = D.filter((p) => p.ifng < med);
  const mHi = mean(hi.map((p) => p.cd8)), mLo = mean(lo.map((p) => p.cd8));
  const fHi = fitLinear(hi.map((p) => [p.ifng]), hi.map((p) => p.cd8)), fLo = fitLinear(lo.map((p) => [p.ifng]), lo.map((p) => p.cd8));
  const bins6 = qBins(D, 6);
  const fw = figure(ch, 'Approach 1: compare two groups (median split)',
    `Each dot is a patient. The median split puts ${hi.length} patients in a high-IFN-γ group and ${lo.length} in a low-IFN-γ group and compares group means. Diamonds are mean CD8 % in six equal-sized groups of patients, ordered by IFN-γ; they follow a rising line, with no step at the median.`);
  let wMode = 'split';
  select(fw.controls, { label: 'Show', value: wMode, options: [['split', 'Median split (two group means)'], ['line', 'Regression line'], ['both', 'Both']], onChange: (v) => { wMode = v; paintW(); } });
  const wStats = stats(fw.root);
  const FW = frame(fw.body, { width: 820, height: 330, x: XR, y: [0, YM], xLabel: 'IFN-γ score (log2 expression)', yLabel: 'CD8 T cells (% of cells)' });
  FW.layer('pts').selectAll('circle').data(D).join('circle').attr('cx', (d) => FW.x(d.ifng)).attr('cy', (d) => FW.y(d.cd8)).attr('r', 3.8)
    .attr('fill', cat(0)).attr('fill-opacity', 0.55).attr('stroke', surface()).attr('stroke-width', 0.8)
    .on('mousemove', (e, d) => tip(e, `${d.id} · IFN-γ ${d.ifng}<br>CD8 ${d.cd8}%`)).on('mouseleave', untip);
  FW.layer('bins').selectAll('path').data(bins6).join('path').attr('d', d3.symbol(d3.symbolDiamond, 90)).attr('transform', (b) => `translate(${FW.x(b.x)},${FW.y(b.y)})`)
    .attr('fill', ink()).attr('stroke', surface())
    .on('mousemove', (e, b) => tip(e, `IFN-γ ${f1(b.lo)}–${f1(b.hi)} (${b.n} patients)<br>mean CD8 ${f1(b.y)}%`)).on('mouseleave', untip);
  const fwL = FW.layer('fit');
  legend(fw.body, [['patient', cat(0)], ['mean of a sixth of patients', ink()], ['group means (median split)', cat(1)], ['regression line', cat(2)]]);
  function paintW() {
    fwL.selectAll('*').remove();
    if (wMode !== 'line') {
      [[XR[0], med, mLo], [med, XR[1], mHi]].forEach(([x0, x1, m]) => fwL.append('line').attr('x1', FW.x(x0)).attr('x2', FW.x(x1)).attr('y1', FW.y(m)).attr('y2', FW.y(m)).attr('stroke', cat(1)).attr('stroke-width', 3));
      fwL.append('line').attr('x1', FW.x(med)).attr('x2', FW.x(med)).attr('y1', 0).attr('y2', FW.h).attr('stroke', muted()).attr('stroke-dasharray', '4 4');
      note(fwL, FW.x(med) + 6, 14, `median IFN-γ ${f2(med)}`, { fill: muted(), size: 11.5 });
      note(fwL, FW.x(med) - 8, FW.y(mLo) - 8, `low group mean ${f1(mLo)}%`, { fill: ink2(), anchor: 'end' });
      note(fwL, FW.x(med) + 8, FW.y(mHi) - 8, `high group mean ${f1(mHi)}%`, { fill: ink2() });
    }
    if (wMode !== 'split') fwL.append('path').datum(XR).attr('fill', 'none').attr('stroke', cat(2)).attr('stroke-width', 2.5).attr('d', d3.line().x((x) => FW.x(x)).y((x) => FW.y(a1 + b1 * x)));
    wStats.set({ 'high − low group means': `${f1(mHi - mLo)} points`, 'line: change per IFN-γ unit': `${f2(b1)} points`, 'slope inside high group': f2(fHi.w[0]), 'slope inside low group': f2(fLo.w[0]) });
  }
  paintW();
  prose(ch, `<p>The split gives a single number (${f1(mHi - mLo)} points) and discards information: patients just either side of the median are assigned to different groups, and within each group CD8 % still rises with IFN-γ (slopes ${f2(fLo.w[0])} and ${f2(fHi.w[0])}). The regression line uses every patient and reports the effect <i>per unit</i>.</p>`);
  worked(ch, 'Approach 2: the correlation coefficient',
    `<p>The correlation between IFN-γ and CD8 % is <b>r = ${f2(r1)}</b>, and r² = ${f2(r1 * r1)} equals the R² of the line. The correlation has no units, so it does not say how many CD8 points a unit of IFN-γ is worth. The slope does: <b>b = r × SD(y) / SD(x) = ${f2(r1)} × ${f2(sd(ys))} / ${f2(sd(xs))} = ${f2(r1 * sd(ys) / sd(xs))}</b> points per unit. Correlation also cannot adjust for other variables or give prediction intervals.</p>`);
  table(ch, [
    { a: 'Compare two groups (t-test)', e: 'one difference, in outcome units', k: 'no: needs a cut-off', j: 'partly', p: 'no' },
    { a: 'Correlation coefficient', e: 'no (unit-free strength only)', k: 'yes', j: 'no', p: 'no' },
    { a: '<b>Linear regression</b>', e: '<b>yes, per unit, with SE and CI</b>', k: '<b>yes</b>', j: '<b>yes (multiple regression)</b>', p: '<b>yes (confidence and prediction intervals)</b>' },
    { a: 'Random forest / boosting / neural net', e: 'not directly', k: 'yes', j: 'yes, incl. interactions', p: 'needs extra methods' },
  ], [['a', 'Approach'], ['e', 'Effect size in outcome units?'], ['k', 'Uses continuous predictor without cut-off?'], ['j', 'Adjusts for other variables?'], ['p', 'Uncertainty for a new prediction?']]);

  // =====================================================================================
  // 3. WHERE
  // =====================================================================================
  ch = chapter(main, 'where', 'Where is it used?', 'Any setting with a continuous outcome and measured predictors: for prediction, for an adjusted effect size, or both.');
  cards(ch, [
    { tag: 'Clinical biomarkers', title: 'How much does a marker move the outcome?', body: 'CD8 infiltration from an IFN-γ score, as on this page; blood pressure, tumour size or a lab value from age, dose and genotype. Output: an adjusted effect per unit with a confidence interval.' },
    { tag: 'Differential expression', title: 'limma and voom', body: '<b>limma</b> fits one linear model per gene (expression ~ condition + batch + covariates) and moderates the variances across genes. <b>voom</b> converts counts to log-CPM with precision weights so the same machinery applies to RNA-seq. See the <a href="#/omics/bulk-de">differential expression</a> page.' },
    { tag: 'Genetics (eQTL)', title: 'Is expression linked to genotype?', body: 'Matrix eQTL and tensorQTL regress each gene\'s expression on genotype dosage (0, 1 or 2 alleles) with covariates such as ancestry components. The slope is the per-allele effect on expression.' },
    { tag: 'Genetics (GWAS)', title: 'Quantitative traits', body: 'For a continuous trait (height, LDL cholesterol) each variant is tested with a linear model (PLINK <code>--glm</code>, REGENIE) with ancestry covariates. Polygenic scores are weighted sums of such per-variant effects.' },
    { tag: 'Deconvolution', title: 'Cell fractions from bulk expression', body: 'A bulk profile is modelled as a signature matrix times unknown cell fractions. Several methods solve this by (non-negative, weighted) least squares, for example MuSiC; CIBERSORT uses support-vector regression instead of squared error.' },
    { tag: 'Covariate adjustment', title: 'Removing technical effects', body: 'Regressing out library size or percent mitochondrial reads (<code>sc.pp.regress_out</code> in Scanpy) or batch (<code>removeBatchEffect</code> in limma) fits a linear model and keeps the residuals.' },
    { tag: 'Assay calibration', title: 'Standard curves', body: 'ELISA, qPCR and mass-spectrometry assays fit a line of signal against known concentration, then invert it to read concentrations from unknown samples.' },
    { tag: 'Epidemiology', title: 'Adjusted associations', body: 'Cohort studies report the association of an exposure with a continuous outcome after adjusting for age, sex and other confounders, as in step 5 of this page.' },
    { tag: 'Inside other models', title: 'Building block', body: 'The last layer of a regression network is a linear model on learned features. Ridge and lasso (<a href="#/concepts/regularization">regularisation</a>) are penalised versions, and logistic regression is the same weighted sum with a different outcome.' },
  ]);
  prose(ch, '<p><b>When not to use it:</b> the outcome is yes/no (use <a href="#/concepts/logistic-regression">logistic regression</a>), a count (use a Poisson or negative-binomial model, as in DESeq2 and edgeR) or a time to an event (use Cox regression); observations are not independent, for example repeated measures per patient (use a mixed model); there are more predictors than samples (use penalised regression); or the relationship is strongly non-linear and cannot be linearised by a transformation (consider splines or tree ensembles).</p>');

  // =====================================================================================
  // 4. HOW IT WORKS
  // =====================================================================================
  ch = chapter(main, 'how', 'How it works', 'Six steps: from squared errors to a multivariable model checked on an independent cohort.');

  // ---- Step 1: residuals and squares
  const u = xs.map((x) => x - 6), Su = sum(u), Suu = sum(u.map((v) => v * v)), Sy = sum(ys), Syy = sum(ys.map((v) => v * v)), Suy = sum(u.map((v, i) => v * ys[i]));
  const sse = (a, b) => Syy - 2 * a * Sy - 2 * b * Suy + n * a * a + 2 * a * b * Su + b * b * Suu;
  const aB = a1 + 6 * b1, sseB = sse(aB, b1), sst = fit1.tss;
  let rb = rung(ch, 1, 'Residuals and the sum of squared errors',
    `<p>Any line gives a fitted value for each patient. The <b>residual</b> is observed minus fitted. Large residuals of either sign are errors, so they are squared before being added: the <b>sum of squared errors (SSE)</b>. A flat line at the mean CD8 % (${f1(my)}) has SSE = ${Math.round(sst)}, called the total sum of squares (SST). The least-squares line has SSE = ${Math.round(sseB)}, and <b>R² = 1 − SSE / SST = ${f2(1 - sseB / sst)}</b>.</p>
     <p>The line is written as CD8 = <i>a</i> + <i>b</i> × (IFN-γ − 6), so that <i>a</i> is the fitted CD8 % at the typical IFN-γ of 6 and the two parameters do not trade off against each other.</p>`);
  const f4 = figure(rb, 'Choosing a line by hand', 'Left: patients, the current line, and residual squares for 18 of the patients (side = residual, so area ∝ squared error; all residuals are used in the SSE). Right: SSE for every (a, b); darker is lower, the cross marks the least-squares minimum. Drag the sliders or click the right panel.');
  let A1 = my, B1 = 0;
  const sA = slider(f4.controls, { label: 'a (CD8 % at IFN-γ 6)', min: -2, max: 24, step: 0.1, value: A1, fmt: f1, onInput: (v) => { A1 = v; paint1(); } });
  const sB = slider(f4.controls, { label: 'b (slope)', min: -2, max: 9, step: 0.05, value: B1, fmt: f2, onInput: (v) => { B1 = v; paint1(); } });
  button(f4.controls, 'Flat line at the mean', () => { A1 = my; B1 = 0; sync1(); });
  button(f4.controls, 'Snap to least squares', () => { A1 = aB; B1 = b1; sync1(); }, true);
  const s4 = stats(f4.root);
  const row4 = el('div', { class: 'fig-row' }); f4.body.append(row4);
  const c4a = el('div'), c4b = el('div'); row4.append(c4a, c4b);
  const F4 = frame(c4a, { width: 500, height: 340, x: XR, y: [0, YM], xLabel: 'IFN-γ score', yLabel: 'CD8 T cells (% of cells)' });
  F4.g.append('clipPath').attr('id', 'lr4-clip').append('rect').attr('width', F4.w).attr('height', F4.h);
  const sqL = F4.layer('sq').attr('clip-path', 'url(#lr4-clip)'), lnL = F4.layer('ln').attr('clip-path', 'url(#lr4-clip)');
  F4.layer('pts').selectAll('circle').data(D).join('circle').attr('cx', (d) => F4.x(d.ifng)).attr('cy', (d) => F4.y(d.cd8)).attr('r', 3.4)
    .attr('fill', cat(0)).attr('fill-opacity', 0.6).attr('stroke', surface()).attr('stroke-width', 0.7);
  const sub = D.slice().sort((p, q) => p.ifng - q.ifng).filter((_, i) => i % 10 === 5);
  const AG = [-2, 24], BG = [-2, 9], NG = 40, cells = [];
  for (let i = 0; i < NG; i++) for (let j = 0; j < NG; j++) { const a = AG[0] + (i + 0.5) * (AG[1] - AG[0]) / NG, b = BG[0] + (j + 0.5) * (BG[1] - BG[0]) / NG; cells.push({ a, b, L: sse(a, b) }); }
  const col4 = sequential(Math.log(sst * 2.5), Math.log(sseB));
  const H4 = frame(c4b, { width: 440, height: 340, x: AG, y: BG, xLabel: 'a (CD8 % at IFN-γ 6)', yLabel: 'b (slope)', grid: false });
  const cw = H4.w / NG, chh = H4.h / NG;
  H4.layer('heat').selectAll('rect').data(cells).join('rect').attr('x', (d) => H4.x(d.a) - cw / 2).attr('y', (d) => H4.y(d.b) - chh / 2).attr('width', cw + 0.5).attr('height', chh + 0.5)
    .attr('fill', (d) => col4(Math.log(Math.min(sst * 2.5, d.L))))
    .on('mousemove', (e, d) => tip(e, `a = ${f1(d.a)}, b = ${f2(d.b)}<br>SSE = ${Math.round(d.L)}`)).on('mouseleave', untip)
    .on('click', (e, d) => { A1 = d.a; B1 = d.b; sync1(); });
  H4.layer('min').append('path').attr('d', d3.symbol(d3.symbolCross, 110)).attr('transform', `translate(${H4.x(aB)},${H4.y(b1)}) rotate(45)`).attr('fill', ink()).attr('stroke', surface());
  note(H4.layer('min'), H4.x(aB) + 10, H4.y(b1) - 9, 'least squares', { size: 11.5 });
  const cur4 = H4.layer('cur');
  function sync1() { sA.set(A1); sB.set(B1); paint1(); }
  function paint1() {
    const sq = sub.map((p) => { const yh = A1 + B1 * (p.ifng - 6); return { p, yh, s: Math.abs(F4.y(p.cd8) - F4.y(yh)), r: p.cd8 - yh }; });
    sqL.selectAll('rect').data(sq).join('rect').attr('x', (d) => F4.x(d.p.ifng)).attr('y', (d) => Math.min(F4.y(d.p.cd8), F4.y(d.yh))).attr('width', (d) => d.s).attr('height', (d) => d.s)
      .attr('fill', cat(1)).attr('fill-opacity', 0.12).attr('stroke', cat(1)).attr('stroke-width', 1)
      .on('mousemove', (e, d) => tip(e, `${d.p.id}: residual ${sgn(d.r)}<br>squared error ${f1(d.r * d.r)}`)).on('mouseleave', untip);
    lnL.selectAll('path').data([XR]).join('path').attr('fill', 'none').attr('stroke', cat(2)).attr('stroke-width', 2.8).attr('d', d3.line().x((x) => F4.x(x)).y((x) => F4.y(A1 + B1 * (x - 6))));
    cur4.selectAll('circle').data([0]).join('circle').attr('cx', H4.x(Math.max(AG[0], Math.min(AG[1], A1)))).attr('cy', H4.y(Math.max(BG[0], Math.min(BG[1], B1)))).attr('r', 6).attr('fill', cat(2)).attr('stroke', surface()).attr('stroke-width', 2);
    const e = sse(A1, B1);
    s4.set({ 'SSE of this line': Math.round(e), 'SSE least squares': Math.round(sseB), 'SSE flat line (SST)': Math.round(sst), 'R² of this line': f2(1 - e / sst), 'typical error (RMSE)': f1(Math.sqrt(e / n)) + ' points' });
  }
  paint1();
  legend(f4.body, [['patient', cat(0)], ['current line', cat(2)], ['squared residual', cat(1)]]);
  prose(rb, `<p>The right panel has one basin and a single lowest point, which is why least squares has a closed-form answer. Squaring makes the penalty grow faster than the error: a patient ${f1(Math.abs(d3.max(fit1.resid, Math.abs)))} points off the line contributes ${Math.round(d3.max(fit1.resid, Math.abs) ** 2)} to the SSE, as much as ${Math.round(d3.max(fit1.resid, Math.abs) ** 2 / (fit1.sigma ** 2))} typical patients (typical squared residual ${f1(fit1.sigma ** 2)}). Least squares is therefore sensitive to outliers.</p>`);

  // ---- Step 2: closed form, interpretation, uncertainty
  rb = rung(ch, 2, 'The least-squares line: slope, intercept, R² and uncertainty',
    `<p>Setting the derivatives of SSE to zero gives the solution directly: <b>slope = Σ(x − x̄)(y − ȳ) / Σ(x − x̄)²</b> and <b>intercept = ȳ − slope × x̄</b>.</p>`);
  worked(rb, 'Slope and intercept by hand',
    `<div class="calc">mean IFN-γ x̄ = ${f2(mx)}     mean CD8 ȳ = ${f2(my)} %
Σ(x − x̄)(y − ȳ) = ${f1(sxy)}      Σ(x − x̄)² = ${f1(sxx)}
slope b = ${f1(sxy)} / ${f1(sxx)} = <b>${f2(b1)}</b> CD8 points per IFN-γ unit
intercept a = ${f2(my)} − ${f2(b1)} × ${f2(mx)} = <b>${f2(a1)}</b>
residual SD σ̂ = √(SSE / (n − 2)) = ${f2(fit1.sigma)}      SE(b) = σ̂ / √Σ(x − x̄)² = ${f2(fit1.sigma)} / ${f2(Math.sqrt(sxx))} = <b>${f2(fit1.se[1])}</b>
95% CI for b = ${f2(b1)} ± ${f2(t975(fit1.df))} × ${f2(fit1.se[1])} = <b>${f2(b1 - t975(fit1.df) * fit1.se[1])} to ${f2(b1 + t975(fit1.df) * fit1.se[1])}</b></div>
     <ul><li><b>Slope:</b> patients whose IFN-γ differs by one log2 unit differ by ${f2(b1)} CD8 points on average. This describes a difference between patients, not what happens to one patient if their score is changed.</li>
     <li><b>Intercept:</b> ${f1(a1)}% at IFN-γ = 0. This is negative and lies far outside the observed range (${f1(d3.min(xs))}–${f1(d3.max(xs))}), so it has no biological meaning and serves only to position the line.</li></ul>`);
  const fx = figure(rb, 'Fitted line, confidence band and prediction band', 'Narrow orange band: 95% confidence interval for the <i>mean</i> CD8 % at each IFN-γ. Wide blue band: 95% prediction interval for a <i>new individual</i> patient. Move the slider to read both. "Draw bootstrap fits" refits the line to patients resampled with replacement; the spread of these lines is the sampling variability that the standard error describes.');
  const tq = t975(fit1.df), gx = d3.range(XR[0], XR[1] + 0.01, 0.2);
  const bandOf = (x) => { const yh = a1 + b1 * x, cm = tq * Math.sqrt(fit1.varMean([x])), pm = tq * Math.sqrt(fit1.varNew([x])); return { x, yh, cl: yh - cm, ch: yh + cm, pl: yh - pm, ph: yh + pm }; };
  const bands = gx.map(bandOf);
  const yLo = Math.min(0, Math.floor(d3.min(bands, (b) => b.pl) / 5) * 5), yHi = Math.max(YM, Math.ceil(d3.max(bands, (b) => b.ph) / 5) * 5);
  let x0 = 6; const boots = [], rB = rng(5);
  slider(fx.controls, { label: 'IFN-γ score', min: 2.5, max: 10, step: 0.05, value: x0, fmt: f2, onInput: (v) => { x0 = v; paint2(); } });
  button(fx.controls, 'Draw 10 bootstrap fits', () => { for (let k = 0; k < 10; k++) { const idx = range(n).map(() => rB.int(n)); const f = fitLinear(idx.map((i) => [xs[i]]), idx.map((i) => ys[i])); boots.push({ a: f.b0, b: f.w[0] }); } paint2(); });
  button(fx.controls, 'Clear', () => { boots.length = 0; paint2(); });
  const s2 = stats(fx.root);
  const F2 = frame(fx.body, { width: 820, height: 360, x: XR, y: [yLo, yHi], xLabel: 'IFN-γ score (log2 expression)', yLabel: 'CD8 T cells (% of cells)' });
  F2.g.append('clipPath').attr('id', 'lr2-clip').append('rect').attr('width', F2.w).attr('height', F2.h);
  const bL = F2.layer('b').attr('clip-path', 'url(#lr2-clip)');
  const area = (lo, hi) => d3.area().x((b) => F2.x(b.x)).y0((b) => F2.y(b[lo])).y1((b) => F2.y(b[hi]));
  bL.append('path').datum(bands).attr('d', area('pl', 'ph')).attr('fill', cat(0)).attr('fill-opacity', 0.14);
  bL.append('path').datum(bands).attr('d', area('cl', 'ch')).attr('fill', cat(1)).attr('fill-opacity', 0.35);
  if (yLo < 0) { bL.append('line').attr('x1', 0).attr('x2', F2.w).attr('y1', F2.y(0)).attr('y2', F2.y(0)).attr('stroke', muted()).attr('stroke-dasharray', '4 4'); note(bL, F2.w - 8, F2.y(0) - 6, 'CD8 = 0%: values below this are impossible', { fill: muted(), size: 11.5, anchor: 'end' }); }
  const bootL = F2.layer('boot').attr('clip-path', 'url(#lr2-clip)');
  F2.layer('pts').selectAll('circle').data(D).join('circle').attr('cx', (d) => F2.x(d.ifng)).attr('cy', (d) => F2.y(d.cd8)).attr('r', 3.4).attr('fill', cat(0)).attr('fill-opacity', 0.6).attr('stroke', surface()).attr('stroke-width', 0.7);
  F2.layer('l').append('path').datum(XR).attr('fill', 'none').attr('stroke', ink()).attr('stroke-width', 2.5).attr('d', d3.line().x((x) => F2.x(x)).y((x) => F2.y(a1 + b1 * x)));
  const mkL = F2.layer('mk');
  legend(fx.body, [['patient', cat(0)], ['fitted line', ink()], ['95% CI for the mean', cat(1)], ['95% prediction interval, new patient', cat(0)], ['bootstrap fit', muted()]]);
  function paint2() {
    bootL.selectAll('path').data(boots).join('path').attr('fill', 'none').attr('stroke', muted()).attr('stroke-opacity', 0.35).attr('stroke-width', 1).attr('d', (d) => d3.line().x((x) => F2.x(x)).y((x) => F2.y(d.a + d.b * x))(XR));
    const bd = bandOf(x0);
    mkL.selectAll('*').remove();
    mkL.append('line').attr('x1', F2.x(x0)).attr('x2', F2.x(x0)).attr('y1', F2.y(bd.pl)).attr('y2', F2.y(bd.ph)).attr('stroke', ink()).attr('stroke-width', 1.5);
    mkL.append('circle').attr('cx', F2.x(x0)).attr('cy', F2.y(bd.yh)).attr('r', 6).attr('fill', ink()).attr('stroke', surface()).attr('stroke-width', 2);
    note(mkL, F2.x(x0) + (x0 > 8 ? -10 : 10), F2.y(bd.ph) + (x0 > 8 ? 4 : 12), `new patient: ${f1(bd.pl)} to ${f1(bd.ph)}%`, { size: 11.5, anchor: x0 > 8 ? 'end' : 'start' });
    const bs = boots.map((q) => q.b);
    s2.set({ 'IFN-γ': f2(x0), 'fitted CD8 %': f1(bd.yh), '95% CI for the mean': `${f1(bd.cl)} to ${f1(bd.ch)}`, '95% interval, new patient': `${f1(bd.pl)} to ${f1(bd.ph)}`, 'SE of slope (formula)': f2(fit1.se[1]), 'SD of bootstrap slopes': bs.length > 2 ? `${f2(sd(bs))} (${bs.length} fits)` : '—' });
  }
  paint2();
  const lowB = bandOf(3);
  prose(rb, `<p>The confidence band is narrowest at the mean IFN-γ (${f2(mx)}) and widens towards the edges, where the line is anchored by fewer patients. The prediction band is much wider because an individual patient also carries their own residual (σ̂ = ${f2(fit1.sigma)}). At IFN-γ 3 the lower prediction bound is ${f1(lowB.pl)}%, a negative percentage, a first sign that a constant-spread straight-line model does not suit these data at the extremes (step 3). After a few dozen bootstrap fits, the SD of their slopes is shown beside the formula SE; the two are expected to be close, because the standard error measures exactly this sampling variability.</p>`);

  // ---- Step 3: residual checks
  rb = rung(ch, 3, 'Checking the assumptions with residual plots',
    `<p>The standard errors and intervals above assume that (i) the relationship is linear, (ii) residuals have constant spread and (iii) are roughly normal, and (iv) patients are independent. Plotting residuals against fitted values checks (i) and (ii), and a normal Q-Q plot (residuals against the quantiles expected under normality) checks (iii). Panels are redrawn for the outcome on its original scale or after taking log2.</p>`);
  const rs = (ln) => {
    const yy = D.map((p) => Y(p, ln)), f = fitLinear(xs.map((x) => [x]), yy), ord = range(n).sort((a, b) => f.fitted[a] - f.fitted[b]), t = Math.floor(n / 3);
    const sdOf = (ix) => sd(ix.map((i) => f.resid[i])), z = f.resid.map((e) => e / f.sigma);
    return { f, ord, ratio: sdOf(ord.slice(n - t)) / sdOf(ord.slice(0, t)), skew: mean(z.map((v) => v ** 3)), z, zmax: d3.max(z, Math.abs) };
  };
  const RS = { raw: rs(false), log: rs(true) };
  const f3g = figure(rb, 'Residual plots', 'Left: residual against fitted value; the line is the average residual in six groups, dashed lines mark ±2 residual SD. A flat line and an even band indicate a suitable model. Right: standardised residuals against normal quantiles; points on the dashed diagonal indicate normal residuals. Hover for patients.');
  let scale3 = 'raw';
  select(f3g.controls, { label: 'Outcome', value: scale3, options: [['raw', 'CD8 (%)'], ['log', 'log2 CD8']], onChange: (v) => { scale3 = v; paint3(); } });
  const s3 = stats(f3g.root);
  const row3 = el('div', { class: 'fig-row' }); f3g.body.append(row3);
  const c3a = el('div'), c3b = el('div'); row3.append(c3a, c3b);
  legend(f3g.body, [['patient', cat(0)], ['group average of residuals', cat(2)], ['largest |standardised residual|', cat(1)]]);
  function paint3() {
    const R = RS[scale3], ln = scale3 === 'log', f = R.f;
    c3a.innerHTML = ''; c3b.innerHTML = '';
    const emax = Math.ceil(d3.max(f.resid, Math.abs) * 1.05);
    const FA = frame(c3a, { width: 500, height: 330, x: d3.extent(f.fitted).map((v, i) => v + (i ? 0.4 : -0.4)), y: [-emax, emax], xLabel: ln ? 'fitted log2 CD8' : 'fitted CD8 (%)', yLabel: 'residual' });
    const worst = R.z.reduce((b, v, i) => (Math.abs(v) > Math.abs(R.z[b]) ? i : b), 0);
    FA.layer('p').selectAll('circle').data(D).join('circle').attr('cx', (d, i) => FA.x(f.fitted[i])).attr('cy', (d, i) => FA.y(f.resid[i])).attr('r', (d, i) => (i === worst ? 5.5 : 3.6))
      .attr('fill', (d, i) => (i === worst ? cat(1) : cat(0))).attr('fill-opacity', 0.65).attr('stroke', surface()).attr('stroke-width', 0.7)
      .on('mousemove', (e, d) => { const i = D.indexOf(d); tip(e, `${d.id}<br>fitted ${f2(f.fitted[i])}, observed ${f2(Y(d, ln))}<br>residual ${f2(f.resid[i])}`); }).on('mouseleave', untip);
    [0, 2, -2].forEach((k) => FA.layer('r').append('line').attr('x1', 0).attr('x2', FA.w).attr('y1', FA.y(k * f.sigma)).attr('y2', FA.y(k * f.sigma)).attr('stroke', k ? ink2() : muted()).attr('stroke-dasharray', k ? '5 4' : null));
    const grp = d3.range(6).map((g) => { const ix = R.ord.slice(Math.round(g * n / 6), Math.round((g + 1) * n / 6)); return { x: mean(ix.map((i) => f.fitted[i])), y: mean(ix.map((i) => f.resid[i])) }; });
    FA.layer('m').append('path').datum(grp).attr('fill', 'none').attr('stroke', cat(2)).attr('stroke-width', 2.5).attr('d', d3.line().x((q) => FA.x(q.x)).y((q) => FA.y(q.y)));
    note(FA.layer('m'), 8, 14, R.ratio > 1.4 ? `spread grows with the fitted value (×${f1(R.ratio)})` : R.ratio < 0.7 ? `spread larger at low fitted values (×${f1(R.ratio)})` : `spread about constant (×${f1(R.ratio)})`, { size: 12 });
    const FQ = frame(c3b, { width: 440, height: 330, x: [-3.2, 3.2], y: [-3.2, 3.2], xLabel: 'normal quantile', yLabel: 'standardised residual' });
    FQ.layer('d').append('line').attr('x1', FQ.x(-3.2)).attr('y1', FQ.y(-3.2)).attr('x2', FQ.x(3.2)).attr('y2', FQ.y(3.2)).attr('stroke', ink2()).attr('stroke-dasharray', '5 4');
    const zs = R.z.map((v, i) => ({ v, i })).sort((p, q) => p.v - q.v);
    FQ.layer('p').selectAll('circle').data(zs).join('circle').attr('cx', (d, k) => FQ.x(qNorm((k + 0.5) / n))).attr('cy', (d) => FQ.y(Math.max(-3.2, Math.min(3.2, d.v)))).attr('r', 3.6)
      .attr('fill', (d) => (d.i === worst ? cat(1) : cat(0))).attr('fill-opacity', 0.65).attr('stroke', surface()).attr('stroke-width', 0.7)
      .on('mousemove', (e, d) => tip(e, `${D[d.i].id}<br>standardised residual ${f2(d.v)}`)).on('mouseleave', untip);
    note(FQ.layer('d'), 8, 14, Math.abs(R.skew) > 0.5 ? `long ${R.skew > 0 ? 'upper' : 'lower'} tail (skewness ${f2(R.skew)})` : `close to the line (skewness ${f2(R.skew)})`, { size: 12 });
    s3.set({ 'R²': f2(f.r2), 'residual SD': f2(f.sigma) + (ln ? ' log2 units' : ' points'), 'spread, top ÷ bottom third': '× ' + f1(R.ratio), 'skewness of residuals': f2(R.skew), 'largest |std. residual|': f1(R.zmax) });
  }
  paint3();
  const fL = RS.log.f, fR = RS.raw.f;
  prose(rb, `<p>On the original scale the residual spread is ${f1(RS.raw.ratio)} times larger among the highest fitted values than the lowest and the residuals are right-skewed (skewness ${f2(RS.raw.skew)}): CD8 % behaves multiplicatively, so a given relative change is a larger absolute change at higher values. After taking log2, the skewness is ${f2(RS.log.skew)} and the ratio is ${f1(RS.log.ratio)}: the fan has reversed mildly, with the largest spread at the low end (the smallest CD8 values are the most variable in relative terms), so the log scale over-corrects slightly but is much closer to even. R² rises from ${f2(fR.r2)} to ${f2(fL.r2)}. The slope then means a <b>fold change</b>: ${f2(fL.w[0])} log2 units per IFN-γ unit, i.e. CD8 is multiplied by 2<sup>${f2(fL.w[0])}</sup> = <b>${f2(2 ** fL.w[0])}</b> (+${pc(2 ** fL.w[0] - 1)}) per unit. Steps 4–6 use log2 CD8 as the outcome.</p>`);
  quiz(rb, 'The slope on log2 CD8 is 1.0. What does that mean for CD8 %?', 'Each +1 unit of the predictor doubles the expected CD8 %, since 2<sup>1</sup> = 2. A slope of −1 would halve it. Log-scale slopes are multiplicative on the original scale.');

  // ---- Step 4: several predictors
  const M = ['ifng', 'caf', 'pdl1', 'prolif', 'age', 'centreB'];
  const fitM = fitLinear(D.map((p) => M.map((k) => FEATURES[k].get(p))), yl), tM = t975(fitM.df);
  const NOTE = { ifng: 'drives CD8 (through immune activity)', caf: 'none directly; marks excluded tumours', pdl1: 'none directly; second readout of activity', prolif: 'none', age: 'none', centreB: 'none; centres differ only in phenotype mix' };
  const sdM = Object.fromEntries(M.map((k) => [k, sd(D.map((p) => FEATURES[k].get(p)))]));
  const rowsM = M.map((k, j) => ({ k, lab: FEATURES[k].label + (FEATURES[k].unit === '%' ? ' (%)' : ''), w: fitM.w[j], se: fitM.se[j + 1], lo: fitM.w[j] - tM * fitM.se[j + 1], hi: fitM.w[j] + tM * fitM.se[j + 1], p: pNormal(fitM.w[j] / fitM.se[j + 1]), fc: 2 ** (fitM.w[j] * sdM[k]), t: NOTE[k] }));
  rb = rung(ch, 4, 'Several predictors: "holding the others fixed"',
    `<p>Multiple regression adds terms: log2 CD8 = <i>b</i>₀ + <i>b</i>₁·IFN-γ + <i>b</i>₂·CAF + … Each coefficient is the average change in the outcome per unit of that predictor <b>among patients who have the same values of all the other predictors</b>. Because the data are simulated, the last column states what was built in.</p>`);
  table(rb, rowsM, [['lab', 'Predictor'], ['w', 'Estimate (log2 CD8 per unit)', f3], ['se', 'SE', f3], ['lo', '95% CI', (v, r) => `${f3(r.lo)} to ${f3(r.hi)}`], ['p', 'p-value', pFmt], ['fc', 'Fold change per 1 SD', f2], ['t', 'Built into the simulation']], { cls: 'compact' });
  const mk = Object.fromEntries(rowsM.map((r) => [r.k, r]));
  const fIfOnly = fitL;
  const resOf = (keys, yv) => { const f = fitLinear(D.map((p) => keys.map((k) => FEATURES[k].get(p))), yv); return f.resid; };
  const AV = Object.fromEntries(M.map((k, j) => { const others = M.filter((q) => q !== k); const ry = resOf(others, yl), rx = resOf(others, D.map((p) => FEATURES[k].get(p))); return [k, { ry, rx, alone: fitLinear(D.map((p) => [FEATURES[k].get(p)]), yl).w[0] }]; }));
  const fm = figure(rb, 'Coefficients and the added-variable plot', 'Left: each predictor\'s effect per 1 SD (comparable across predictors) with 95% interval; open diamonds mark "no direct effect" where that is built into the simulation. Right: for the selected predictor, the part of the outcome and of the predictor not explained by the others; the slope of this cloud is exactly the coefficient in the table.');
  let selM = 'ifng';
  select(fm.controls, { label: 'Predictor', value: selM, options: M.map((k) => [k, FEATURES[k].label]), onChange: (v) => { selM = v; paint4(); } });
  const s4m = stats(fm.root);
  const row5 = el('div', { class: 'fig-row' }); fm.body.append(row5);
  const c5a = el('div'), c5b = el('div'); row5.append(c5a, c5b);
  legend(fm.body, [['positive effect', cat(1)], ['negative effect', cat(0)], ['no direct effect built in', ink()]]);
  function paint4() {
    forest(c5a, rowsM.map((r) => ({ k: r.k, lab: FEATURES[r.k].label, w: r.w * sdM[r.k], lo: r.lo * sdM[r.k], hi: r.hi * sdM[r.k], zero: ['prolif', 'age', 'centreB'].includes(r.k), tip: `${FEATURES[r.k].label}<br>${f3(r.w * sdM[r.k])} log2 CD8 per SD (${f2(sdM[r.k])} units)<br>95% CI ${f3(r.lo * sdM[r.k])} to ${f3(r.hi * sdM[r.k])}` })), { width: 470, label: 'log2 CD8 per 1 SD of the predictor', hl: selM });
    c5b.innerHTML = '';
    const a = AV[selM], k = selM, w = mk[k].w;
    const FV = frame(c5b, { width: 450, height: 300, x: d3.extent(a.rx).map((v, i) => v + (i ? 1 : -1) * 0.05 * (d3.max(a.rx) - d3.min(a.rx))), y: d3.extent(a.ry).map((v, i) => v + (i ? 0.2 : -0.2)), xLabel: `${FEATURES[k].label}, adjusted for the other five`, yLabel: 'log2 CD8, adjusted', xTicks: 5 });
    FV.layer('p').selectAll('circle').data(D).join('circle').attr('cx', (d, i) => FV.x(a.rx[i])).attr('cy', (d, i) => FV.y(a.ry[i])).attr('r', 3.4).attr('fill', cat(0)).attr('fill-opacity', 0.6).attr('stroke', surface()).attr('stroke-width', 0.7)
      .on('mousemove', (e, d) => tip(e, d.id)).on('mouseleave', untip);
    const [x0v, x1v] = FV.x.domain();
    FV.layer('l').append('line').attr('x1', FV.x(x0v)).attr('x2', FV.x(x1v)).attr('y1', FV.y(w * x0v)).attr('y2', FV.y(w * x1v)).attr('stroke', ink()).attr('stroke-width', 2.5);
    note(FV.layer('l'), 8, 14, `slope = ${f3(w)} (SE ${f3(mk[k].se)})`, { size: 12 });
    s4m.set({ 'alone: slope (no others in model)': f3(a.alone), 'adjusted: slope (others held fixed)': f3(w), 'adjusted p-value': pFmt(mk[k].p), 'R² of full model': f2(fitM.r2) });
  }
  paint4();
  prose(rb, `<p>With IFN-γ alone, log2 CD8 rises by ${f3(fIfOnly.w[0])} per unit; with the five other predictors in the model it is ${f3(mk.ifng.w)}. The adjusted effect is the one among patients matched on the others. Proliferation, age and centre have intervals that ${[mk.prolif, mk.age, mk.centreB].every((r) => r.lo < 0 && r.hi > 0) ? 'all include 0, consistent with the absence of a direct effect.' : 'mostly include 0 (' + [mk.prolif, mk.age, mk.centreB].filter((r) => r.lo < 0 && r.hi > 0).length + ' of 3); with 95% intervals about 1 in 20 true zeros is expected to exclude 0 by chance.'} CAF score (p ${pFmt(mk.caf.p)}) and PD-L1 (p ${pFmt(mk.pdl1.p)}) have no direct effect in the simulation, yet ${mk.caf.p < 0.05 || mk.pdl1.p < 0.05 ? 'at least one has a coefficient distinguishable from 0' : 'their coefficients are small'}. IFN-γ is a noisy measurement of immune activity, and these variables carry additional information about the same underlying state (CAF distinguishes excluded from desert tumours; PD-L1 is a second readout of activity). A non-zero coefficient shows association given the others, not a direct effect. R² is ${f2(fitM.r2)} against ${f2(fL.r2)} for IFN-γ alone: R² never decreases when a predictor is added, so the adjusted R² (${f2(fitM.r2adj)}) penalises extra terms.</p>`);
  quiz(rb, 'Why is the IFN-γ coefficient different in the one-predictor and six-predictor models?', 'The coefficient answers a different question. Alone, it compares patients who differ in IFN-γ and in everything correlated with it. With the others included, it compares patients with the same CAF, PD-L1, proliferation, age and centre. The two agree only when the other variables are unrelated to IFN-γ.');

  // ---- Step 5: confounding by centre
  const isB = (p) => p.centreB === 1;
  const fCe0 = fitLinear(D.map((p) => [p.centreB]), yl), fCe1 = fitLinear(D.map((p) => [p.ifng, p.centreB]), yl), t1 = t975(fCe1.df);
  const DA = D.filter((p) => !isB(p)), DB = D.filter(isB);
  const fA = fitLinear(DA.map((p) => [p.ifng]), DA.map((p) => L2(p.cd8))), fB = fitLinear(DB.map((p) => [p.ifng]), DB.map((p) => L2(p.cd8)));
  const inflA = DA.filter((p) => p.phenotype === 'inflamed').length / DA.length, inflB = DB.filter((p) => p.phenotype === 'inflamed').length / DB.length;
  rb = rung(ch, 5, 'Confounding by centre',
    `<p>A <b>confounder</b> is a variable associated with both the predictor and the outcome. Centre B enrolled more inflamed tumours (${pc(inflB)} against ${pc(inflA)} at Centre A), so its patients have higher IFN-γ <i>and</i> higher CD8. Centre has no direct effect in the simulation. A regression on centre alone nevertheless finds one:</p>`);
  const fc = figure(rb, 'Centre and IFN-γ', 'Each dot is a patient (log2 CD8 against IFN-γ). Select the view: one pooled line; a separate line per centre; or the two centre means (diamonds), which lie on the pooled line.');
  let mode5 = 'sep';
  select(fc.controls, { label: 'View', value: mode5, options: [['pooled', 'One pooled line'], ['sep', 'Separate line per centre'], ['means', 'Centre means']], onChange: (v) => { mode5 = v; paint5(); } });
  const s5 = stats(fc.root);
  const F5 = frame(fc.body, { width: 820, height: 340, x: XR, y: [Math.floor(d3.min(yl)), Math.ceil(d3.max(yl))], xLabel: 'IFN-γ score (log2 expression)', yLabel: 'log2 CD8 (% of cells)' });
  F5.layer('p').selectAll('circle').data(D).join('circle').attr('cx', (d) => F5.x(d.ifng)).attr('cy', (d) => F5.y(L2(d.cd8))).attr('r', 3.8).attr('fill', (d) => cat(d.centreB)).attr('fill-opacity', 0.6).attr('stroke', surface()).attr('stroke-width', 0.7)
    .on('mousemove', (e, d) => tip(e, `${d.id} · ${d.centre} · ${d.phenotype}<br>IFN-γ ${d.ifng}, CD8 ${d.cd8}%`)).on('mouseleave', untip);
  const L5 = F5.layer('l');
  legend(fc.body, [['Centre A', cat(0)], ['Centre B', cat(1)], ['pooled fit', ink()]]);
  function paint5() {
    L5.selectAll('*').remove();
    const ln = (a, b, c, w = 2.5) => L5.append('path').datum(XR).attr('fill', 'none').attr('stroke', c).attr('stroke-width', w).attr('d', d3.line().x((x) => F5.x(x)).y((x) => F5.y(a + b * x)));
    if (mode5 !== 'sep') ln(fitL.b0, fitL.w[0], ink());
    if (mode5 === 'sep') { ln(fA.b0, fA.w[0], cat(0)); ln(fB.b0, fB.w[0], cat(1)); }
    if (mode5 === 'means') [[DA, 0], [DB, 1]].forEach(([g, c]) => {
      L5.append('path').attr('d', d3.symbol(d3.symbolDiamond, 260)).attr('transform', `translate(${F5.x(mean(g.map((p) => p.ifng)))},${F5.y(mean(g.map((p) => L2(p.cd8))))})`).attr('fill', cat(c)).attr('stroke', ink()).attr('stroke-width', 2);
      note(L5, F5.x(mean(g.map((p) => p.ifng))) + (c ? 12 : -12), F5.y(mean(g.map((p) => L2(p.cd8)))) + (c ? -12 : 20), `Centre ${c ? 'B' : 'A'} mean`, { anchor: c ? 'start' : 'end' });
    });
    s5.set({ 'mean IFN-γ, A / B': `${f2(mean(DA.map((p) => p.ifng)))} / ${f2(mean(DB.map((p) => p.ifng)))}`, 'slope, A / B': `${f3(fA.w[0])} / ${f3(fB.w[0])}`, 'pooled slope': f3(fitL.w[0]), 'B − A, centre only': f3(fCe0.w[0]), 'B − A, adjusted for IFN-γ': f3(fCe1.w[1]) });
  }
  paint5();
  prose(rb, `<p>On centre alone, patients from Centre B have ${f2(fCe0.w[0])} more log2 CD8 (95% CI ${f2(fCe0.w[0] - t975(fCe0.df) * fCe0.se[1])} to ${f2(fCe0.w[0] + t975(fCe0.df) * fCe0.se[1])}), a ${f2(2 ** fCe0.w[0])}-fold difference. After adjusting for IFN-γ the difference is ${f2(fCe1.w[1])} (95% CI ${f2(fCe1.w[1] - t1 * fCe1.se[2])} to ${f2(fCe1.w[1] + t1 * fCe1.se[2])}). Both centres follow nearly the same line (slopes ${f3(fA.w[0])} and ${f3(fB.w[0])}); Centre B sits further along it. Adjustment works only for confounders that were measured, and a noisy measurement adjusts imperfectly. In omics practice, batch and centre are included as covariates in the design matrix for this reason.</p>`);

  // ---- Step 6: model builder
  const LAB = ['ifng', 'caf', 'pdl1', 'prolif', 'age', 'centreB', 'male', 'smoker', 'log2tmb'], ZERO = ['prolif', 'age', 'centreB', 'male', 'smoker', 'log2tmb'];
  const order = rng(21).shuffle(range(n));
  const evalM = (keys, ln, nT) => {
    const trn = order.slice(0, nT).map((i) => D[i]), st = standardize(trn.map((p) => keys.map((k) => FEATURES[k].get(p))));
    const f = fitLinear(st.Z, trn.map((p) => Y(p, ln))), yV = V.map((p) => Y(p, ln)), pv = V.map((p) => predictLinear(st.apply(keys.map((k) => FEATURES[k].get(p))), f.b0, f.w));
    const rssV = sum(pv.map((v, i) => (yV[i] - v) ** 2)), mV = mean(yV), tssV = sum(yV.map((v) => (v - mV) ** 2));
    return { f, st, pv, yV, keys, trainR2: f.r2, valR2: 1 - rssV / tssV, rmse: Math.sqrt(rssV / V.length) };
  };
  rb = rung(ch, 6, 'Model builder: predictors, training size and validation',
    `<p>Models are fitted on discovery patients (Centres A, B) and evaluated on the ${V.length} independent patients of Centre C. Predictors are standardised, so each coefficient is the effect of one SD. The validation R² compares predictions with observations in Centre C and can be negative if the model is worse than predicting the Centre C mean.</p>`);
  const fl = figure(rb, 'Model builder', 'Left: coefficients with 95% intervals; open diamonds mark predictors with no direct effect in the simulation. Right: predicted against observed outcome for every validation patient; points on the diagonal are perfect predictions. Reduce the training size to see overfitting.');
  const selL = new Set(['ifng']), boxes = {};
  let logL = true, nTr = n;
  const presets = el('div', { class: 'controls' }), featRow = el('div', { class: 'controls' });
  fl.controls.before(presets); fl.controls.before(featRow);
  presets.append(el('span', { class: 'fig-caption', style: 'margin:0' }, 'Presets:'));
  [['IFN-γ only', ['ifng']], ['IFN-γ + CAF + PD-L1', ['ifng', 'caf', 'pdl1']], ['Centre only', ['centreB']], ['Centre + IFN-γ', ['ifng', 'centreB']], ['Everything', LAB]].forEach(([lab, ks]) => button(presets, lab, () => { selL.clear(); ks.forEach((k) => selL.add(k)); LAB.forEach((k) => { boxes[k].checked = selL.has(k); }); paintL(); }));
  LAB.forEach((k) => { boxes[k] = checkbox(featRow, { label: FEATURES[k].label, checked: selL.has(k), onChange: (c) => { if (c) selL.add(k); else selL.delete(k); paintL(); } }); });
  select(fl.controls, { label: 'Outcome', value: 'log', options: [['raw', 'CD8 (%)'], ['log', 'log2 CD8']], onChange: (v) => { logL = v === 'log'; paintL(); } });
  slider(fl.controls, { label: 'training patients', min: 15, max: n, step: 5, value: n, fmt: (v) => String(v), onInput: (v) => { nTr = v; paintL(); } });
  const sL = stats(fl.root);
  const rowL = el('div', { class: 'fig-row' }); fl.body.append(rowL);
  const la = el('div'), lb = el('div'); rowL.append(la, lb);
  const PH = ['inflamed', 'excluded', 'desert'];
  legend(fl.body, [['inflamed tumour', cat(0)], ['excluded', cat(1)], ['desert', cat(2)], ['no direct effect in the simulation', ink()]]);
  function paintL() {
    const keys = LAB.filter((k) => selL.has(k));
    lb.innerHTML = '';
    if (!keys.length) { la.innerHTML = ''; sL.set({ note: 'select at least one predictor' }); return; }
    const E = evalM(keys, logL, nTr), t = t975(E.f.df);
    forest(la, keys.map((k, j) => ({ k, lab: FEATURES[k].label, w: E.f.w[j], lo: E.f.w[j] - t * E.f.se[j + 1], hi: E.f.w[j] + t * E.f.se[j + 1], zero: ZERO.includes(k), tip: `${FEATURES[k].label}<br>${f3(E.f.w[j])} per SD<br>95% CI ${f3(E.f.w[j] - t * E.f.se[j + 1])} to ${f3(E.f.w[j] + t * E.f.se[j + 1])}` })), { width: 470, label: logL ? 'log2 CD8 per 1 SD of the predictor' : 'CD8 (% points) per 1 SD of the predictor' });
    const ext = d3.extent([...E.pv, ...E.yV]), pad = (ext[1] - ext[0]) * 0.05, dom = [ext[0] - pad, ext[1] + pad];
    const FS = frame(lb, { width: 450, height: 330, x: dom, y: dom, xLabel: logL ? 'predicted log2 CD8' : 'predicted CD8 (%)', yLabel: logL ? 'observed log2 CD8' : 'observed CD8 (%)' });
    FS.layer('d').append('line').attr('x1', FS.x(dom[0])).attr('y1', FS.y(dom[0])).attr('x2', FS.x(dom[1])).attr('y2', FS.y(dom[1])).attr('stroke', ink2()).attr('stroke-dasharray', '5 4');
    FS.layer('p').selectAll('circle').data(V).join('circle').attr('cx', (d, i) => FS.x(E.pv[i])).attr('cy', (d, i) => FS.y(E.yV[i])).attr('r', 3.8)
      .attr('fill', (d) => cat(PH.indexOf(d.phenotype))).attr('fill-opacity', 0.7).attr('stroke', surface()).attr('stroke-width', 0.7)
      .on('mousemove', (e, d) => { const i = V.indexOf(d); tip(e, `${d.id} (${d.phenotype})<br>predicted ${f2(E.pv[i])}, observed ${f2(E.yV[i])}`); }).on('mouseleave', untip);
    note(FS.layer('d'), 8, 14, `validation R² = ${f2(E.valR2)}`, { size: 12.5 });
    sL.set({ 'training R²': f2(E.trainR2), 'training adjusted R²': f2(E.f.r2adj), 'validation R²': f2(E.valR2), 'validation RMSE': f2(E.rmse) + (logL ? ' log2' : ' points'), 'patients per predictor': String(+(nTr / keys.length).toFixed(1)) });
  }
  paintL();
  const e1 = evalM(['ifng'], true, n), e3 = evalM(['ifng', 'caf', 'pdl1'], true, n), eAll = evalM(LAB, true, n), eSm = evalM(LAB, true, 20), eRaw = evalM(['ifng'], false, n);
  const cZ0 = fitLinear(D.map((p) => [p.centreB]), yl), cZ1 = fitLinear(D.map((p) => [p.ifng, p.centreB]), yl);
  const eP = evalM(['ifng', 'prolif'], true, n), tP = t975(eP.f.df), pr = eP.f.w[1], prSE = eP.f.se[2];
  const eIP = evalM(['ifng', 'pdl1'], true, n);
  prose(rb, '<p><b>Suggested comparisons</b>:</p>');
  table(rb, [
    { t: '<b>IFN-γ only</b> → <b>IFN-γ + CAF + PD-L1</b>', s: `Validation R² ${f2(e1.valR2)} → <b>${f2(e3.valR2)}</b>`, m: 'Predictors that carry additional information improve prediction in a new centre; the gain here is modest.' },
    { t: '<b>Centre only</b> → <b>Centre + IFN-γ</b>', s: `Centre B coefficient ${f2(cZ0.w[0])} (SE ${f2(cZ0.se[1])}) → <b>${f2(cZ1.w[1])}</b> (SE ${f2(cZ1.se[2])})`, m: '<b>Confounding.</b> The centre effect is explained by the case mix once IFN-γ is in the model.' },
    { t: '<b>Everything</b>, training size ' + n + ' → <b>20</b>', s: `Training R² ${f2(eAll.trainR2)} → <b>${f2(eSm.trainR2)}</b>, validation R² ${f2(eAll.valR2)} → <b>${f2(eSm.valR2)}</b>`, m: '<b>Overfitting.</b> With few patients per predictor, training R² overstates performance on new patients.' },
    { t: 'Outcome <b>CD8 %</b> → <b>log2 CD8</b> (IFN-γ only)', s: `Validation R² ${f2(eRaw.valR2)} → <b>${f2(e1.valR2)}</b>`, m: 'R² is measured on different scales, so it is not a like-for-like comparison, but the log model also has valid intervals (step 3).' },
    { t: 'Add <b>proliferation</b> to IFN-γ', s: `Coefficient ${f3(pr)} per SD, 95% CI ${f3(pr - tP * prSE)} to ${f3(pr + tP * prSE)}`, m: `No direct effect is built in. ${Math.abs(pr) < tP * prSE ? 'The interval includes 0.' : 'The interval excludes 0: a false positive that occurs about 1 time in 20.'}` },
    { t: 'Add <b>PD-L1</b> to IFN-γ', s: `SE of the IFN-γ coefficient ${f3(e1.f.se[1])} → <b>${f3(eIP.f.se[1])}</b> (r between the two = ${f2(corr(xs, D.map((p) => p.pdl1)))})`, m: 'Correlated predictors inflate each other\'s standard errors (collinearity), even when predictions improve.' },
  ], [['t', 'Try'], ['s', 'What you see'], ['m', 'What it means']], { cls: 'compact' });

  // =====================================================================================
  // 5. PITFALLS
  // =====================================================================================
  ch = chapter(main, 'pitfalls', 'Pitfalls', 'Common problems, how to recognise them, and how to address them.');
  const hiLev = fit1.lev.reduce((b, v, i) => (v > fit1.lev[b] ? i : b), 0);
  const cook = fit1.resid.map((e, i) => (e * e / (2 * fit1.sigma ** 2)) * fit1.lev[i] / (1 - fit1.lev[i]) ** 2), iC = cook.reduce((b, v, i) => (v > cook[b] ? i : b), 0);
  const fDrop = fitLinear(xs.filter((_, i) => i !== iC).map((x) => [x]), ys.filter((_, i) => i !== iC));
  const nar = D.filter((p) => p.ifng >= 5.5 && p.ifng <= 7), fNar = fitLinear(nar.map((p) => [p.ifng]), nar.map((p) => p.cd8));
  const rN = rng(77); let hits = 0; const NN = 1000;
  for (let i = 0; i < NN; i++) { const z = D.map(() => rN.normal()), f = fitLinear(z.map((v) => [v]), yl); if (pNormal(f.w[0] / f.se[1]) < 0.05) hits++; }
  cards(ch, [
    { tag: 'Symptom: a prediction below zero', title: 'Extrapolation', body: `The fitted intercept is ${f1(a1)}% at IFN-γ = 0, and the lower prediction bound at IFN-γ 3 is ${f1(lowB.pl)}%. The line is a local summary of the observed range (${f1(d3.min(xs))}–${f1(d3.max(xs))}). <b>Fix:</b> predict only within the range of the data, centre predictors so the intercept is meaningful, and model a positive outcome on the log scale.` },
    { tag: 'Symptom: fan-shaped residual plot', title: 'Non-constant variance and skew', body: `On the original scale the residual spread is ${f1(RS.raw.ratio)}× larger at high fitted values; after log2 it is ${f1(RS.log.ratio)}× (a mild reversal). Standard errors and intervals assume equal spread. <b>Fix:</b> transform the outcome (log for concentrations and fractions), use weighted least squares (voom) or robust standard errors.` },
    { tag: 'Symptom: one patient moves the line', title: 'Outliers and leverage', body: `The most influential patient (${D[iC].id}, IFN-γ ${f2(D[iC].ifng)}, CD8 ${f1(D[iC].cd8)}%) has a residual of ${f1(fit1.resid[iC])}. Dropping this patient changes the slope from ${f2(b1)} to ${f2(fDrop.w[0])}. Leverage is highest at the edges of the predictor range (largest here ${f3(fit1.lev[hiLev])}, against an average of ${f3(2 / n)}). <b>Fix:</b> plot residuals and Cook's distance, check the measurement, report with and without, consider robust regression.` },
    { tag: 'Symptom: a coefficient depends on the model', title: 'Confounding and collinearity', body: `A coefficient depends on what else is in the model (step 5: centre goes from ${f2(cZ0.w[0])} to ${f2(cZ1.w[1])}). IFN-γ and PD-L1 correlate at r = ${f2(corr(xs, D.map((p) => p.pdl1)))}, so the variance of each coefficient is inflated by 1/(1 − r²) = ${f2(1 / (1 - corr(xs, D.map((p) => p.pdl1)) ** 2))}. <b>Fix:</b> decide covariates from the causal question, include measured confounders, and combine or drop near-duplicate predictors.` },
    { tag: 'Symptom: high training R², poor R² on new patients', title: 'Overfitting', body: `Nine predictors fitted on only 20 patients (step 6, "Everything" at training size 20) give training R² = ${f2(eSm.trainR2)} but R² = ${f2(eSm.valR2)} on the validation patients. With p close to n, least squares fits noise, and with p > n it cannot be solved uniquely. <b>Fix:</b> keep several observations per predictor, validate on independent data, use ridge or lasso (<a href="#/concepts/regularization">regularisation</a>).` },
    { tag: 'Symptom: "R² is low, so the effect is not real"', title: 'Misreading R²', body: `R² depends on the range of the predictor as well as on the slope. Restricted to IFN-γ between 5.5 and 7 (${nar.length} patients), the slope is ${f2(fNar.w[0])} but R² falls from ${f2(fit1.r2)} to ${f2(fNar.r2)}. A small R² can accompany a real slope, and a high R² does not show that the model is correct. <b>Fix:</b> report the slope with its interval, and check residual plots.` },
    { tag: 'Symptom: many "significant" predictors', title: 'Many tests', body: `${NN} columns of pure noise were regressed on log2 CD8, and ${hits} (${pc(hits / NN)}) had p < 0.05, against 5% expected by chance. Fitting 20,000 gene-wise models gives about 1,000 such false positives. <b>Fix:</b> control the false-discovery rate (Benjamini–Hochberg), as limma, DESeq2 and edgeR do.` },
    { tag: 'Symptom: a causal claim from a slope', title: 'Association is not causation', body: 'A coefficient describes how outcomes differ between patients. Without randomisation, hidden common causes and measurement error can produce or hide a slope, and the simulation shows how centre produces a spurious one. <b>Fix:</b> state results as associations, adjust for confounders on a stated causal model, and test causal hypotheses experimentally.' },
  ]);

  // =====================================================================================
  // 6. RECAP
  // =====================================================================================
  ch = chapter(main, 'recap', 'Recap', 'Key points.');
  recap(ch, [
    'Linear regression predicts a <b>number</b> as intercept + weighted sum of predictors, and is fitted by <b>minimising the sum of squared residuals</b>.',
    'The slope is the average change in the outcome per unit of the predictor, <b>holding the other predictors fixed</b>. It describes differences between patients, and the intercept is often an extrapolation.',
    'The least-squares solution is unique and has a closed form. It reports a <b>standard error and confidence interval</b> for each coefficient, and a wider <b>prediction interval</b> for a new individual.',
    '<b>R²</b> is the fraction of the variance explained. It never falls when predictors are added, and it depends on the range of the data.',
    'Check <b>residual plots</b>: curvature, a fan shape or skew mean the intervals are unreliable. Taking log of a positive, skewed outcome turns slopes into fold changes and often fixes both.',
    '<b>Confounding</b> (here, centre) creates effects that vanish after adjustment. Include measured confounders, and note that unmeasured ones remain.',
    'With few observations per predictor, <b>validate</b> on independent patients; with many predictors, regularise. Many tests need FDR control.',
    'The same model underlies limma, eQTL and GWAS tests, covariate regression and many deconvolution methods.',
  ]);
  prose(ch, '<h3>In code</h3><p>Use a statistics package for coefficients, intervals and p-values (inference), and a machine-learning package for predictions on new patients, with a held-out evaluation.</p>');
  code(ch, {
    Python: `import numpy as np, statsmodels.formula.api as smf
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

train["log2cd8"] = np.log2(train["cd8"]);  test["log2cd8"] = np.log2(test["cd8"])

# 1) Inference: coefficients, SE, 95% CI, p-values, R2
m = smf.ols("log2cd8 ~ ifng + caf + pdl1 + prolif + age + C(centre)", data=train).fit()
print(m.summary())              # m.conf_int() for intervals
pred = m.get_prediction(test).summary_frame(alpha=0.05)   # mean CI and obs_ci (prediction interval)

# 2) Prediction: standardised predictors, judged on held-out patients
feats = ["ifng", "caf", "pdl1"]
reg = make_pipeline(StandardScaler(), LinearRegression()).fit(train[feats], train["log2cd8"])
print("validation R2", r2_score(test["log2cd8"], reg.predict(test[feats])))`,
    R: `train$log2cd8 <- log2(train$cd8); test$log2cd8 <- log2(test$cd8)

# 1) Inference: coefficients, SE, 95% CI, p-values, R2
m <- lm(log2cd8 ~ ifng + caf + pdl1 + prolif + age + centre, data = train)
summary(m); confint(m)
plot(m)                                           # residual, Q-Q and leverage plots
predict(m, newdata = test, interval = "prediction")

# 2) Held-out R2
p <- predict(m, newdata = test)
1 - sum((test$log2cd8 - p)^2) / sum((test$log2cd8 - mean(test$log2cd8))^2)

# One model per gene (differential expression): limma
# fit <- limma::lmFit(log_cpm, model.matrix(~ condition + batch)); fit <- limma::eBayes(fit)`,
  });
  seeAlso(main, [['concepts', 'logistic-regression', 'Logistic regression'], ['concepts', 'gradient-descent', 'Gradient descent'], ['concepts', 'regularization', 'Regularisation'], ['concepts', 'train-test', 'Train/test split and overfitting'], ['omics', 'bulk-de', 'Differential expression']]);

  // -------------------------------------------------------------------------------------------
  /** The annotated anatomy of one regression line (chapter 1). */
  function anatomy() {
    const fg = figure(document.createElement('div'), 'Anatomy of a regression line', `Each dot is a discovery patient. The line is the least-squares fit; the annotations mark the slope triangle, a fitted value and a residual.`);
    const F = frame(fg.body, { width: 820, height: 340, x: XR, y: [0, YM], xLabel: 'IFN-γ score (log2 expression)', yLabel: 'CD8 T cells (% of cells)' });
    F.layer('p').selectAll('circle').data(D).join('circle').attr('cx', (d) => F.x(d.ifng)).attr('cy', (d) => F.y(d.cd8)).attr('r', 3.8).attr('fill', cat(0)).attr('fill-opacity', 0.55).attr('stroke', surface()).attr('stroke-width', 0.8)
      .on('mousemove', (e, d) => tip(e, `${d.id} · IFN-γ ${d.ifng}<br>CD8 ${d.cd8}%<br>fitted ${f1(a1 + b1 * d.ifng)}%, residual ${sgn(d.cd8 - a1 - b1 * d.ifng)}`)).on('mouseleave', untip);
    const g = F.layer('a'), yh = (x) => a1 + b1 * x, P = exPt, py = yh(P.ifng);
    g.append('path').datum(XR).attr('fill', 'none').attr('stroke', ink()).attr('stroke-width', 2.5).attr('d', d3.line().x((x) => F.x(x)).y((x) => F.y(yh(x))));
    const seg = (x1, y1, x2, y2, o = {}) => { const l = g.append('line').attr('x1', F.x(x1)).attr('y1', F.y(y1)).attr('x2', F.x(x2)).attr('y2', F.y(y2)).attr('stroke', o.c || ink2()).attr('stroke-width', o.w || 1.2); if (o.d) l.attr('stroke-dasharray', o.d); };
    const T = (x, y, txt, o = {}) => note(g, F.x(x), F.y(y), txt, o);
    // slope triangle above the line (sparse region), labelled at the left with a leader line
    seg(4.8, yh(4.8), 4.8, yh(5.8), { c: cat(1), w: 3 }); seg(4.8, yh(5.8), 5.8, yh(5.8), { c: cat(1), w: 3 });
    T(5.3, yh(5.8) + 1.6, 'run: +1 unit', { anchor: 'middle', size: 11.5 });
    seg(4.7, 18, 4.8, yh(5.3), { d: '3 3' });
    T(2.4, 19.3, `slope: rise of +${f2(b1)} CD8 points per +1 IFN-γ unit`, { size: 11.5 });
    // residual of one patient
    seg(P.ifng, py, P.ifng, P.cd8, { c: cat(1), w: 3 });
    g.append('circle').attr('cx', F.x(P.ifng)).attr('cy', F.y(py)).attr('r', 5).attr('fill', ink()).attr('stroke', surface()).attr('stroke-width', 1.5);
    g.append('circle').attr('cx', F.x(P.ifng)).attr('cy', F.y(P.cd8)).attr('r', 6.5).attr('fill', 'none').attr('stroke', ink()).attr('stroke-width', 1.8);
    T(P.ifng, P.cd8 + 3.4, `observed y = ${f1(P.cd8)}% (${P.id})`, { anchor: 'middle', size: 11.5 });
    seg(5.9, 33, P.ifng - 0.06, (py + P.cd8) / 2, { d: '3 3' });
    T(5.85, 33.3, `residual ${sgn(P.cd8 - py)} = observed − fitted`, { anchor: 'end', size: 11.5 });
    seg(P.ifng + 0.05, py - 0.3, 8.0, 4.6, { d: '3 3' });
    T(10.15, 2.4, `fitted value ŷ = ${f1(py)}% (the point on the line)`, { anchor: 'end', size: 11.5 });
    seg(9.55, yh(9.55) + 0.4, 9.85, 31, { d: '3 3' });
    T(10.15, 32, `fitted line: ŷ = ${f1(a1)} ${sgn(b1)} × x`, { anchor: 'end', size: 11.5 });
    T(2.4, YM - 1.4, `intercept ${f1(a1)} = the line extended to IFN-γ = 0, outside the plot`, { size: 11.5, fill: muted() });
    return fg.root;
  }
}

// ---------------------------------------------------------------------------------------------
/** Text with a surface-coloured halo so it stays legible over points and lines. */
function note(layer, x, y, txt, o = {}) {
  return layer.append('text').attr('x', x).attr('y', y).attr('font-size', o.size || 12.5).attr('fill', o.fill || ink()).attr('text-anchor', o.anchor || 'start')
    .attr('stroke', surface()).attr('stroke-width', 3.5).attr('paint-order', 'stroke').attr('stroke-linejoin', 'round').text(txt);
}
function corr(a, b) {
  const ma = mean(a), mb = mean(b);
  let sab = 0, saa = 0, sbb = 0;
  a.forEach((v, i) => { sab += (v - ma) * (b[i] - mb); saa += (v - ma) ** 2; sbb += (b[i] - mb) ** 2; });
  return sab / Math.sqrt(saa * sbb);
}
/** Mean CD8 % in k equal-sized groups ordered by IFN-γ. */
function qBins(pts, k) {
  const s = pts.slice().sort((a, b) => a.ifng - b.ifng);
  return range(k).map((i) => { const part = s.slice(Math.round(i * s.length / k), Math.round((i + 1) * s.length / k)); return { x: mean(part.map((p) => p.ifng)), y: mean(part.map((p) => p.cd8)), lo: part[0].ifng, hi: part[part.length - 1].ifng, n: part.length }; });
}
/** Forest (coefficient) plot into `box`. rows: [{k, lab, w, lo, hi, zero, tip}]; zero = draw an open diamond at 0 (no direct effect built in). */
function forest(box, rows, { width = 480, label = '', hl = null } = {}) {
  box.innerHTML = '';
  const m = { l: 120, r: 16, t: 8, b: 46 }, rowH = 32, H = rows.length * rowH, W = width - m.l - m.r;
  const svg = d3.select(box).append('svg').attr('viewBox', `0 0 ${width} ${H + m.t + m.b}`).attr('width', width).attr('role', 'img');
  const g = svg.append('g').attr('transform', `translate(${m.l},${m.t})`);
  const ext = Math.max(1e-9, d3.max(rows, (r) => Math.max(Math.abs(r.lo), Math.abs(r.hi), Math.abs(r.w)))) * 1.1;
  const x = d3.scaleLinear().domain([-ext, ext]).range([0, W]);
  g.append('g').attr('class', 'axis x').attr('transform', `translate(0,${H})`).call(d3.axisBottom(x).ticks(5));
  g.append('text').attr('class', 'ax-label').attr('x', W / 2).attr('y', H + 38).attr('text-anchor', 'middle').text(label);
  g.append('line').attr('x1', x(0)).attr('x2', x(0)).attr('y1', 0).attr('y2', H).attr('stroke', muted()).attr('stroke-dasharray', '4 4');
  rows.forEach((r, i) => {
    const cy = i * rowH + rowH / 2, on = r.k === hl;
    if (on) g.append('rect').attr('x', -m.l).attr('width', width).attr('y', cy - rowH / 2).attr('height', rowH).attr('fill', css('--accent-soft')).attr('opacity', 0.6);
    g.append('text').attr('x', -8).attr('y', cy + 4).attr('text-anchor', 'end').attr('font-size', 12).attr('fill', ink2()).attr('font-weight', on ? 700 : 400).text(r.lab);
    g.append('line').attr('x1', x(r.lo)).attr('x2', x(r.hi)).attr('y1', cy).attr('y2', cy).attr('stroke', ink2()).attr('stroke-width', 2);
    if (r.zero) g.append('path').attr('d', d3.symbol(d3.symbolDiamond, 90)).attr('transform', `translate(${x(0)},${cy})`).attr('fill', 'none').attr('stroke', ink()).attr('stroke-width', 1.5);
    g.append('circle').attr('cx', x(r.w)).attr('cy', cy).attr('r', 5.5).attr('fill', cat(r.w >= 0 ? 1 : 0)).attr('stroke', surface()).attr('stroke-width', 1.5)
      .on('mousemove', (e) => tip(e, r.tip)).on('mouseleave', untip);
  });
}
