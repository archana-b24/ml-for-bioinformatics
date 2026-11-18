// Logistic regression, lesson format (template page for the rewrite).
// Arc: what → why → where → how it works (7 steps: odds → odds ratio → log-odds line → fit the curve → 2 features
//      → coefficients + patient calculator → model builder with try/see/means table) → pitfalls → recap.
// Every number in the prose is computed from data/study/cohort.json at render time.
import { header, figure, slider, select, button, checkbox, stats, legend, seeAlso, animate, tip, untip, el, html } from '../lib/ui.js';
import { frame, decisionCanvas, cat, css, ink, ink2, muted, sequential } from '../lib/plot.js';
import { sigmoid, mean, sd } from '../lib/data.js';
import { fitLogistic, predictProb, logLoss, auc, confusion, standardize, pNormal } from '../lib/models.js';
import { loadCohort, FEATURES, studyCard } from '../lib/study.js';
import { lessonNav, chapter, prose, rung, worked, quiz, cards, table, code, recap } from '../lib/lesson.js';
const d3 = window.d3;

const um = (t) => t.replace(/^-/, '−');
const f2 = (v) => (Math.abs(v) < 0.005 ? '0.00' : um(v.toFixed(2)));
const f1 = (v) => um(v.toFixed(1));
const pc = (v) => Math.round(100 * v) + '%';
const sgn = (v) => (v < 0 ? '− ' + f2(-v) : '+ ' + f2(v));
const logit = (p) => Math.log(p / (1 - p));
const pFmt = (p) => (p < 0.001 ? '< 0.001' : p.toFixed(3));
const yes = (r) => `<span class="pill ${r ? 'yes' : 'no'}">${r ? 'responded' : 'no response'}</span>`;
const sum = (a) => a.reduce((s, v) => s + v, 0);

export default async function render(main) {
  const C = await loadCohort();
  const D = C.discovery, V = C.validation;
  const y = D.map((p) => p.response), yv = V.map((p) => p.response);
  const rowsOf = (pts, keys) => pts.map((p) => keys.map((k) => FEATURES[k].get(p)));
  const F3 = ['ifng', 'log2tmb', 'caf'];
  const fit1 = fitLogistic(rowsOf(D, ['ifng']), y);
  const fit2 = fitLogistic(rowsOf(D, ['ifng', 'log2tmb']), y);
  const fit3 = fitLogistic(rowsOf(D, F3), y);
  const prob3 = (p) => predictProb(F3.map((k) => FEATURES[k].get(p)), fit3.b0, fit3.w);
  const n = D.length, nR = sum(y), base = nR / n;
  const TR = C.truth.response_logit, TRUE_NAT = { ifng: TR.ifng, log2tmb: TR.log2_tmb, caf: TR.caf };

  header(main, 'Logistic regression',
    'A model that turns a handful of measurements into the <b>probability of a yes/no outcome</b>, such as "will this patient respond to treatment?". It adds the measurements up with weights, exactly like a straight-line fit, then bends the total through an S-shaped curve so the answer always lands between 0 and 1.');
  lessonNav(main);
  studyCard(main, C, '<b>Question for this page:</b> given a tumour biopsy, what is the probability that a patient responds to anti-PD-1, and which measurements contribute?');

  // =====================================================================================
  // 1. WHAT
  // =====================================================================================
  let ch = chapter(main, 'what', 'What is logistic regression?', 'In one line: <b>weighted sum → S-curve → probability</b>.');
  prose(ch, `<p>Logistic regression is a <b>classifier</b>. It takes measurements on a sample (here, a patient) and returns the <b>probability</b> that the sample belongs to the positive class (here, responds to treatment). It is the standard first model for a binary outcome and the usual baseline against which more complex methods are compared.</p>
    <p>The model works in three steps. The diagram follows one patient from the study through a model fitted to IFN-γ score, TMB and CAF score (chapter 4 explains how the weights are estimated).</p>`);
  const exP = D.filter((p) => p.response).sort((a, b) => Math.abs(prob3(a) - 0.72) - Math.abs(prob3(b) - 0.72))[0];
  ch.append(pipelineSvg(exP, fit3));
  prose(ch, '<p>Key terms:</p>');
  const zEx = logit(prob3(exP));
  table(ch, [
    { t: 'Outcome (label, <i>y</i>)', m: 'The yes/no thing we want to predict, coded 1 / 0', s: 'responded = 1, did not = 0' },
    { t: 'Feature (<i>x</i>)', m: 'A measurement used as input', s: 'IFN-γ score, TMB, CAF score' },
    { t: 'Weight / coefficient (<i>w</i>)', m: 'How far one unit of a feature moves the score', s: `w for IFN-γ = ${f2(fit3.w[0])}` },
    { t: 'Intercept (<i>b</i>₀)', m: 'The score when every feature is 0; it sets the baseline', s: `b₀ = ${f2(fit3.b0)}` },
    { t: 'Score / log-odds (<i>z</i>)', m: 'The weighted sum. Any number from −∞ to +∞', s: `z = ${f2(zEx)} for ${exP.id}` },
    { t: 'Sigmoid σ(z)', m: 'The S-curve that maps any z into the range 0–1', s: `σ(${f2(zEx)}) = ${f2(prob3(exP))}` },
    { t: 'Probability (<i>p</i>)', m: 'The model output: P(outcome = 1 | these features)', s: `${pc(prob3(exP))} chance ${exP.id} responds` },
    { t: 'Threshold', m: 'The cut-off that turns p into a yes/no call', s: '0.5 by default; the choice depends on the application (chapter 4, step 6)' },
  ], [['t', 'Term'], ['m', 'Meaning'], ['s', 'In our study']]);

  // =====================================================================================
  // 2. WHY
  // =====================================================================================
  ch = chapter(main, 'why', 'Why do we need it?', 'What a model for a binary outcome should provide, and where simpler approaches fall short.');
  prose(ch, `<p>For each patient, a useful model should:</p><ol>
    <li><b>Return a probability, not only a class label.</b> A predicted 70% and a predicted 51% chance of response support different decisions.</li>
    <li><b>Combine several measurements</b>, each with an estimated weight.</li>
    <li><b>Quantify the contribution of each measurement</b>, with a confidence interval, so the result can be reported and reproduced.</li></ol>
    <p>Two common simpler approaches each fail at least one of these.</p>`);

  // ---- Figure: straight line vs S-curve
  const fw = figure(ch, 'Approach 1: linear regression on a 0/1 outcome',
    'Each dot is a patient: responders at the top (1), non-responders at the bottom (0), spread vertically a little so they do not overlap. Diamonds are the actual response rate in six equal-sized groups of patients, from lowest to highest IFN-γ. Grey bands mark values outside 0–1, which are not valid probabilities.');
  let wMode = 'both';
  select(fw.controls, { label: 'Show', value: wMode, options: [['line', 'Straight line (linear regression)'], ['curve', 'S-curve (logistic regression)'], ['both', 'Both']], onChange: (v) => { wMode = v; paintW(); } });
  const wStats = stats(fw.root);
  const FW = frame(fw.body, { width: 820, height: 330, x: [2.2, 10.2], y: [-0.4, 1.4], xLabel: 'IFN-γ score (log2 expression)', yLabel: 'P(response)' });
  [[-0.4, 0], [1, 1.4]].forEach(([a, b]) => FW.layer('bad').append('rect').attr('x', 0).attr('width', FW.w).attr('y', FW.y(b)).attr('height', FW.y(a) - FW.y(b)).attr('fill', css('--surface-2')));
  FW.layer('bad').append('text').attr('x', 8).attr('y', FW.y(1.32)).attr('fill', muted()).attr('font-size', 11.5).text('invalid: probability above 1');
  FW.layer('bad').append('text').attr('x', 8).attr('y', FW.y(-0.3)).attr('fill', muted()).attr('font-size', 11.5).text('invalid: probability below 0');
  const jit = (i) => (((i * 7919) % 101) / 101 - 0.5) * 0.09;
  FW.layer('pts').selectAll('circle').data(D).join('circle').attr('cx', (d) => FW.x(d.ifng)).attr('cy', (d, i) => FW.y(d.response + jit(i)))
    .attr('r', 4).attr('fill', (d) => cat(d.response)).attr('fill-opacity', 0.75).attr('stroke', css('--surface')).attr('stroke-width', 1)
    .on('mousemove', (e, d) => tip(e, `${d.id} · IFN-γ ${d.ifng}<br>${d.response ? 'responded' : 'no response'}`)).on('mouseleave', untip);
  const bins6 = qBins(D, (p) => p.ifng, 6);
  const xs = D.map((p) => p.ifng), mx = mean(xs), my = base;
  const olsB = sum(D.map((p) => (p.ifng - mx) * (p.response - my))) / sum(D.map((p) => (p.ifng - mx) ** 2)), olsA = my - olsB * mx;
  const lineL = FW.layer('fit');
  FW.layer('bins').selectAll('path').data(bins6).join('path').attr('d', d3.symbol(d3.symbolDiamond, 90)).attr('transform', (b) => `translate(${FW.x(b.x)},${FW.y(b.r / b.n)})`)
    .attr('fill', ink()).attr('stroke', css('--surface'))
    .on('mousemove', (e, b) => tip(e, `IFN-γ ${f1(b.lo)}–${f1(b.hi)}<br>${b.r} of ${b.n} responded (${pc(b.r / b.n)})`)).on('mouseleave', untip);
  legend(fw.body, [['no response (0)', cat(0)], ['responded (1)', cat(1)], ['observed rate in a group', ink()], ['straight line', cat(2)], ['S-curve', cat(6)]]);
  function paintW() {
    const grid = d3.range(2.2, 10.21, 0.05);
    const ls = [];
    if (wMode !== 'curve') ls.push({ c: cat(2), pts: grid.map((x) => [x, olsA + olsB * x]) });
    if (wMode !== 'line') ls.push({ c: cat(6), pts: grid.map((x) => [x, sigmoid(fit1.b0 + fit1.w[0] * x)]) });
    lineL.selectAll('path').data(ls).join('path').attr('fill', 'none').attr('stroke', (d) => d.c).attr('stroke-width', 2.5)
      .attr('d', (d) => d3.line().x((q) => FW.x(q[0])).y((q) => FW.y(Math.max(-0.4, Math.min(1.4, q[1]))))(d.pts));
    const below = D.filter((p) => olsA + olsB * p.ifng < 0).length, above = D.filter((p) => olsA + olsB * p.ifng > 1).length;
    wStats.set({ 'straight line: patients given p < 0': below, 'given p > 1': above, 'line at IFN-γ 10': f2(olsA + olsB * 10), 'S-curve at IFN-γ 10': f2(sigmoid(fit1.b0 + fit1.w[0] * 10)) });
  }
  paintW();
  const below0 = D.filter((p) => olsA + olsB * p.ifng < 0).length;
  prose(ch, `<p>The straight line assigns a <b>negative probability to ${below0} patients</b>. It also cannot follow the observed relationship: the response rate (diamonds) stays near zero at low IFN-γ, then rises, and must level off below 1. This <b>S-shaped</b> pattern is what the logistic curve describes, and its output stays within 0–1 at every value.</p>`);

  const thr6 = D.map((p) => (p.ifng > 6 ? 1 : 0));
  const cRule = confusion(thr6, y, 0.5);
  worked(ch, 'Approach 2: a single cut-off, "IFN-γ > 6"',
    `<p>This rule classifies ${sum(thr6)} patients as responders, with sensitivity <b>${pc(cRule.sens)}</b> and specificity <b>${pc(cRule.spec)}</b>. However:</p>
     <ul><li>Patients at IFN-γ 6.1 and 9.5 receive the same call, although their fitted probabilities differ (${pc(sigmoid(fit1.b0 + fit1.w[0] * 6.1))} vs ${pc(sigmoid(fit1.b0 + fit1.w[0] * 9.5))}).</li>
     <li>Additional markers such as TMB or CAF score cannot be incorporated without further ad hoc rules.</li>
     <li>It provides no effect size or confidence interval.</li></ul>
     <p>Logistic regression addresses all three, and a cut-off can still be applied to its output.</p>`);
  table(ch, [
    { a: 'Cut-off rule on one marker', p: 'no', c: 'no', i: 'partly', d: 'tiny' },
    { a: 'Linear regression on 0/1', p: 'no (goes below 0, above 1)', c: 'yes', i: 'yes', d: 'small' },
    { a: '<b>Logistic regression</b>', p: '<b>yes</b>', c: '<b>yes</b>', i: '<b>yes (odds ratios + CIs)</b>', d: '<b>small to moderate</b>' },
    { a: 'Random forest / boosting / neural net', p: 'yes (often needs calibrating)', c: 'yes, incl. interactions', i: 'harder', d: 'large' },
  ], [['a', 'Approach'], ['p', 'Gives a valid probability?'], ['c', 'Combines features?'], ['i', 'Interpretable effect sizes?'], ['d', 'Data needed']]);

  // =====================================================================================
  // 3. WHERE
  // =====================================================================================
  ch = chapter(main, 'where', 'Where is it used?', 'Any setting with a binary outcome that requires a predicted probability or an interpretable effect size.');
  cards(ch, [
    { tag: 'Clinical prediction', title: 'Will this patient respond?', body: 'Response to immunotherapy from an IFN-γ signature, TMB and PD-L1, as in the example on this page. Output: a probability per patient and an odds ratio per biomarker.' },
    { tag: 'Diagnostics', title: 'Tumour or normal? Cancer or not?', body: 'Penalised logistic regression on DNA-methylation CpGs or a gene panel. Many blood-based cancer-detection and tissue-of-origin classifiers have a logistic model at their core.' },
    { tag: 'Genetics (GWAS)', title: 'Is this variant linked to disease?', body: 'Each SNP is tested by regressing case/control status on genotype plus ancestry PCs (PLINK <code>--glm</code>, REGENIE). The reported effect is the per-allele odds ratio.' },
    { tag: 'Single-cell', title: 'Which cell type is this?', body: 'CellTypist trains one regularised logistic regression per cell type on gene expression and labels large datasets quickly. The coefficients can be read as marker-gene weights.' },
    { tag: 'Epidemiology', title: 'Which risk factors matter?', body: 'Case–control studies (e.g. smoking and lung cancer) report <i>adjusted</i> odds ratios from a logistic model that includes age, sex and other confounders.' },
    { tag: 'Real-world evidence', title: 'Propensity scores', body: 'The probability that a patient received treatment, given their covariates, from a logistic model. Used to match or weight treated vs untreated patients in observational data.' },
    { tag: 'Quality control', title: 'Pass or fail?', body: 'Flagging failed samples, low-quality cells or artefactual variant calls from a handful of QC metrics, with a probability that can be thresholded.' },
    { tag: 'Inside deep learning', title: 'The last layer of a classifier', body: 'A neural network for a yes/no task ends in a sigmoid on a weighted sum: a logistic regression on features the network learned itself.' },
    { tag: 'Outside biology', title: 'Spam, credit, clicks', body: 'Spam filters, loan default risk and ad click-through prediction have long relied on logistic regression because it is fast, stable and interpretable.' },
  ]);
  prose(ch, `<p><b>When not to use it:</b> the outcome is not yes/no (use multinomial or ordinal logistic regression for 3+ classes, Cox regression for time-to-event, linear regression for a number); the signal depends on strong interactions or thresholds you cannot write down as features (consider tree ensembles or boosting); or the inputs are raw images or sequences (neural networks learn suitable features first).</p>`);

  // =====================================================================================
  // 4. BUILD IT UP
  // =====================================================================================
  ch = chapter(main, 'how', 'How it works', 'Six steps, from probability and odds to a multivariable model evaluated on an independent cohort.');

  // ---- Step 1: probability, odds, log-odds
  let rb = rung(ch, 1, 'Probability, odds and log-odds',
    `<p>${nR} of ${n} patients responded. This can be expressed in three equivalent ways:</p>
     <ul><li><b>Probability</b> = responders / everyone = ${nR}/${n} = <b>${f2(base)}</b>. Always between 0 and 1.</li>
     <li><b>Odds</b> = responders / non-responders = ${nR}/${n - nR} = <b>${f2(nR / (n - nR))}</b>. Between 0 and ∞; odds of 1 correspond to 50:50.</li>
     <li><b>Log-odds</b> (also called the <b>logit</b>) = ln(odds) = ln(${f2(nR / (n - nR))}) = <b>${f2(Math.log(nR / (n - nR)))}</b>. Any value from −∞ to +∞; 0 corresponds to 50:50.</li></ul>
     <p>A weighted sum of features can take any value, so it cannot be used directly as a probability, but it can represent log-odds. Logistic regression models the <b>log-odds as a weighted sum of the features</b> and converts the result back to a probability.</p>`);
  const f1g = figure(rb, 'Converting between scales', 'Log-odds map the 0–1 probability range onto the whole number line, symmetrically around 0.');
  let pp = base;
  slider(f1g.controls, { label: 'probability p', min: 0.01, max: 0.99, step: 0.01, value: Math.round(base * 100) / 100, fmt: (v) => v.toFixed(2), onInput: (v) => { pp = v; paint1(); } });
  const s1 = stats(f1g.root);
  const F1 = frame(f1g.body, { width: 820, height: 250, x: [0, 1], y: [-5, 5], xLabel: 'probability p', yLabel: 'log-odds = ln(p / (1 − p))' });
  F1.layer('c').append('path').datum(d3.range(0.005, 0.996, 0.005)).attr('fill', 'none').attr('stroke', cat(6)).attr('stroke-width', 2.2).attr('d', d3.line().x((p) => F1.x(p)).y((p) => F1.y(logit(p))));
  F1.layer('c').append('line').attr('x1', 0).attr('x2', F1.w).attr('y1', F1.y(0)).attr('y2', F1.y(0)).attr('stroke', muted()).attr('stroke-dasharray', '4 4');
  const m1 = F1.layer('m');
  function paint1() {
    m1.selectAll('line').data([0, 1]).join('line').attr('stroke', muted()).attr('stroke-dasharray', '3 3')
      .attr('x1', (d) => (d ? 0 : F1.x(pp))).attr('x2', F1.x(pp)).attr('y1', (d) => (d ? F1.y(logit(pp)) : F1.h)).attr('y2', F1.y(logit(pp)));
    m1.selectAll('circle').data([0]).join('circle').attr('cx', F1.x(pp)).attr('cy', F1.y(logit(pp))).attr('r', 6).attr('fill', cat(6)).attr('stroke', css('--surface')).attr('stroke-width', 1.5);
    const o = pp / (1 - pp);
    s1.set({ probability: pp.toFixed(2), odds: o >= 1 ? `${f2(o)} to 1` : `1 to ${f2(1 / o)}`, 'log-odds': f2(logit(pp)), 'out of 100 such patients': `${Math.round(pp * 100)} respond` });
  }
  paint1();

  // ---- Step 2: odds ratio from a 2x2 table
  const med = d3.median(D, (p) => p.ifng);
  const hi = D.filter((p) => p.ifng >= med), lo = D.filter((p) => p.ifng < med);
  const hR = sum(hi.map((p) => p.response)), lR = sum(lo.map((p) => p.response));
  const oddsH = hR / (hi.length - hR), oddsL = lR / (lo.length - lR), OR = oddsH / oddsL;
  const fitBin = fitLogistic(D.map((p) => [p.ifng >= med ? 1 : 0]), y);
  rb = rung(ch, 2, 'The odds ratio',
    `<p>Patients split at the median IFN-γ score (${f2(med)}):</p>`);
  table(rb, [
    { g: `High IFN-γ (≥ ${f2(med)})`, r: hR, nr: hi.length - hR, o: `${hR} / ${hi.length - hR} = <b>${f2(oddsH)}</b>`, p: pc(hR / hi.length) },
    { g: `Low IFN-γ (< ${f2(med)})`, r: lR, nr: lo.length - lR, o: `${lR} / ${lo.length - lR} = <b>${oddsL.toFixed(3)}</b>`, p: pc(lR / lo.length) },
  ], [['g', 'Group'], ['r', 'Responded'], ['nr', 'Did not'], ['p', 'Response rate'], ['o', 'Odds of response']]);
  prose(rb, `<p><b>Odds ratio (OR)</b> = ${f2(oddsH)} / ${oddsL.toFixed(3)} = <b>${f1(OR)}</b> (OR = 1: no difference). A logistic regression with "high IFN-γ" (1/0) as its only feature estimates a weight of ${f2(fitBin.w[0])}, and e<sup>${f2(fitBin.w[0])}</sup> = <b>${f1(Math.exp(fitBin.w[0]))}</b>, the same value. <b>A logistic regression coefficient is a log odds ratio</b>, for binary and continuous features alike.</p>`);
  const rr = (hR / hi.length) / (lR / lo.length);
  prose(rb, `<p>An odds ratio is not a risk ratio: the response rates are ${pc(hR / hi.length)} vs ${pc(lR / lo.length)}, a risk ratio of ${f1(rr)}. The two agree only when the outcome is rare (below about 10%).</p>`);

  // ---- Step 4+5: sigmoid by hand + likelihood + gradient descent
  const bestA = fit1.b0 + fit1.w[0] * 6, bestB = fit1.w[0];
  rb = rung(ch, 3, 'Fitting the curve: likelihood and gradient descent',
    `<p>With one feature, centred at 6 for readability:</p>
     <div class="calc">z = a + b × (IFN-γ − 6)        p = σ(z) = 1 / (1 + e^(−z))</div>
     <p><b>a</b> is the log-odds at IFN-γ = 6 (shifts the curve); <b>b</b> is the change in log-odds per unit of IFN-γ (sets the steepness). The model assumes this change is constant: log-odds are linear in each feature.</p>
     <p>Each patient contributes the probability the curve assigns to their observed outcome (p for responders, 1 − p otherwise). The product of these is the <b>likelihood</b>; the mean of their −ln values is the <b>log-loss</b>. Fitting minimises log-loss, which for logistic regression is convex: a single minimum (right panel).</p>`);
  const f4 = figure(rb, 'Manual fit and gradient descent',
    'Left: patients and the current curve (dashed = maximum-likelihood fit). Right: log-loss for each (a, b); darker is lower, the cross marks the minimum. Click the right panel to set (a, b), or run gradient descent from the current point.');
  let A = -2.5, B = 0.2, path4 = [], stop4 = null;
  const sA = slider(f4.controls, { label: 'a (log-odds at IFN-γ 6)', min: -4, max: 3, step: 0.05, value: A, fmt: f2, onInput: (v) => { halt4(); A = v; path4 = []; paint4(); } });
  const sB = slider(f4.controls, { label: 'b (slope)', min: -1, max: 3.5, step: 0.05, value: B, fmt: f2, onInput: (v) => { halt4(); B = v; path4 = []; paint4(); } });
  button(f4.controls, 'Fit by gradient descent', () => runGD(), true);
  button(f4.controls, 'Snap to best', () => { halt4(); A = bestA; B = bestB; path4 = []; paint4(); });
  const s4 = stats(f4.root);
  const row4 = el('div', { class: 'fig-row' }); f4.body.append(row4);
  const c4a = el('div'), c4b = el('div'); row4.append(c4a, c4b);
  const F4 = frame(c4a, { width: 500, height: 330, x: [2.2, 10.2], y: [-0.08, 1.08], xLabel: 'IFN-γ score', yLabel: 'P(response)' });
  F4.layer('pts').selectAll('circle').data(D).join('circle').attr('cx', (d) => F4.x(d.ifng)).attr('cy', (d, i) => F4.y(d.response + jit(i) * 0.7))
    .attr('r', 3.6).attr('fill', (d) => cat(d.response)).attr('fill-opacity', 0.7).attr('stroke', css('--surface')).attr('stroke-width', 0.8);
  const gx = d3.range(2.2, 10.21, 0.05);
  F4.layer('best').append('path').datum(gx).attr('fill', 'none').attr('stroke', ink2()).attr('stroke-width', 1.5).attr('stroke-dasharray', '5 4')
    .attr('d', d3.line().x((x) => F4.x(x)).y((x) => F4.y(sigmoid(bestA + bestB * (x - 6)))));
  const cur4 = F4.layer('cur');
  const lossAt = (a, b) => logLoss(D.map((p) => sigmoid(a + b * (p.ifng - 6))), y);
  const AG = [-4, 3], BG = [-1, 3.5], NG = 44;
  const cells = [];
  for (let i = 0; i < NG; i++) for (let j = 0; j < NG; j++) { const a = AG[0] + (i + 0.5) * (AG[1] - AG[0]) / NG, b = BG[0] + (j + 0.5) * (BG[1] - BG[0]) / NG; cells.push({ a, b, L: lossAt(a, b) }); }
  const bestL = lossAt(bestA, bestB);
  const col4 = sequential(Math.log(2.5), Math.log(bestL));
  const H4 = frame(c4b, { width: 420, height: 330, x: AG, y: BG, xLabel: 'a (log-odds at IFN-γ 6)', yLabel: 'b (slope)', grid: false });
  const cw = H4.w / NG, chh = H4.h / NG;
  H4.layer('heat').selectAll('rect').data(cells).join('rect').attr('x', (d) => H4.x(d.a) - cw / 2).attr('y', (d) => H4.y(d.b) - chh / 2).attr('width', cw + 0.5).attr('height', chh + 0.5)
    .attr('fill', (d) => col4(Math.log(Math.min(2.5, d.L))))
    .on('mousemove', (e, d) => tip(e, `a = ${f2(d.a)}, b = ${f2(d.b)}<br>log-loss = ${d.L.toFixed(3)}`)).on('mouseleave', untip)
    .on('click', (e, d) => { halt4(); A = d.a; B = d.b; path4 = []; paint4(); });
  H4.layer('min').append('path').attr('d', d3.symbol(d3.symbolCross, 90)).attr('transform', `translate(${H4.x(bestA)},${H4.y(bestB)})`).attr('fill', ink()).attr('stroke', css('--surface'));
  const trail = H4.layer('trail');
  function paint4() {
    cur4.selectAll('path').data([gx]).join('path').attr('fill', 'none').attr('stroke', cat(6)).attr('stroke-width', 2.8)
      .attr('d', d3.line().x((x) => F4.x(x)).y((x) => F4.y(sigmoid(A + B * (x - 6)))));
    trail.selectAll('path').data([path4]).join('path').attr('fill', 'none').attr('stroke', ink()).attr('stroke-width', 1.5).attr('d', d3.line().x((q) => H4.x(q[0])).y((q) => H4.y(q[1])));
    trail.selectAll('circle').data([0]).join('circle').attr('cx', H4.x(A)).attr('cy', H4.y(B)).attr('r', 6).attr('fill', cat(6)).attr('stroke', css('--surface')).attr('stroke-width', 2);
    const L = lossAt(A, B), mid = B !== 0 ? 6 - A / B : NaN;
    s4.set({ 'current log-loss': L.toFixed(3), 'best possible': bestL.toFixed(3), 'curve crosses 50% at IFN-γ': isFinite(mid) && mid > 0 && mid < 14 ? f2(mid) : '—', 'P at IFN-γ 4 / 6 / 8': [4, 6, 8].map((x) => f2(sigmoid(A + B * (x - 6)))).join(' / ') });
  }
  function halt4() { if (stop4) { stop4(); stop4 = null; } }
  function runGD() {
    halt4(); path4 = [[A, B]]; let k = 0;
    stop4 = animate(() => {
      for (let s = 0; s < 3; s++) {
        let ga = 0, gb = 0;
        D.forEach((p) => { const e = sigmoid(A + B * (p.ifng - 6)) - p.response; ga += e; gb += e * (p.ifng - 6); });
        A -= 1.2 * ga / n; B -= 1.2 * gb / n; k++;
      }
      path4.push([A, B]); sA.set(A); sB.set(B); paint4();
      return k < 450 && Math.hypot(A - bestA, B - bestB) > 0.002;
    });
  }
  paint4();
  prose(rb, `<p>Gradient descent repeatedly moves (a, b) a small step downhill. The gradient is the mean of <b>(prediction − outcome) × feature value</b>, so poorly predicted patients have the most influence. Standard software uses faster solvers (Newton/IRLS, L-BFGS) and reaches the same estimates: a = ${f2(bestA)}, b = ${f2(bestB)}.</p>`);

  // ---- Step 6: two features
  rb = rung(ch, 4, 'Two features: a linear decision boundary',
    `<p>z = b₀ + w₁·IFN-γ + w₂·log2(TMB). The set of points where p = 0.5 (z = 0) is the <b>decision boundary</b>, and it is always a straight line (a plane in higher dimensions). TMB is log2-transformed because it is strongly skewed (${f1(d3.min(D, (p) => p.tmb))}–${f1(d3.max(D, (p) => p.tmb))} mut/Mb); +1 on this scale means TMB doubled.</p>`);
  const f5 = figure(rb, 'One feature vs two features', 'Shading: predicted probability of response. Solid line: p = 0.5; dashed: p = 0.25 and 0.75. Hover a patient for details.');
  let mode5 = 'two';
  select(f5.controls, { label: 'Model', value: mode5, options: [['one', 'IFN-γ only'], ['two', 'IFN-γ + TMB']], onChange: (v) => { mode5 = v; paint5(); } });
  const s5 = stats(f5.root);
  const F5 = frame(f5.body, { width: 820, height: 380, x: [2.2, 10.2], y: [0.6, 5.5], xLabel: 'IFN-γ score', yLabel: 'log2 TMB (mut/Mb)', grid: false });
  const fitOf = () => (mode5 === 'one' ? { b0: fit1.b0, w: [fit1.w[0], 0] } : { b0: fit2.b0, w: fit2.w });
  const surf5 = decisionCanvas(F5, (a, b) => { const f = fitOf(); return sigmoid(f.b0 + f.w[0] * a + f.w[1] * b); }, { res: 4, alpha: 0.35 });
  F5.g.append('clipPath').attr('id', 'lr5-clip').append('rect').attr('width', F5.w).attr('height', F5.h);
  const ct5 = F5.layer('ct').attr('clip-path', 'url(#lr5-clip)');
  F5.layer('pts').selectAll('circle').data(D).join('circle').attr('cx', (d) => F5.x(d.ifng)).attr('cy', (d) => F5.y(d.log2tmb)).attr('r', 4.2)
    .attr('fill', (d) => cat(d.response)).attr('stroke', css('--surface')).attr('stroke-width', 1.1)
    .on('mousemove', (e, d) => { const f = fitOf(); tip(e, `${d.id} (${d.phenotype})<br>IFN-γ ${d.ifng}, TMB ${d.tmb} mut/Mb<br>model P = ${f2(sigmoid(f.b0 + f.w[0] * d.ifng + f.w[1] * d.log2tmb))} · ${d.response ? 'responded' : 'no response'}`); }).on('mouseleave', untip);
  legend(f5.body, [['no response', cat(0)], ['responded', cat(1)], ['p = 0.5 boundary', ink()]]);
  function paint5() {
    const f = fitOf();
    surf5.draw((a, b) => sigmoid(f.b0 + f.w[0] * a + f.w[1] * b));
    const segs = [0.25, 0.5, 0.75].map((p) => {
      const L = logit(p) - f.b0;
      if (Math.abs(f.w[1]) < 1e-9) { const x = L / f.w[0]; return { p, a: [x, 0.6], c: [x, 5.5] }; }
      return { p, a: [2.2, (L - f.w[0] * 2.2) / f.w[1]], c: [10.2, (L - f.w[0] * 10.2) / f.w[1]] };
    });
    ct5.selectAll('line').data(segs).join('line').attr('x1', (d) => F5.x(d.a[0])).attr('y1', (d) => F5.y(d.a[1])).attr('x2', (d) => F5.x(d.c[0])).attr('y2', (d) => F5.y(d.c[1]))
      .attr('stroke', (d) => (d.p === 0.5 ? ink() : ink2())).attr('stroke-width', (d) => (d.p === 0.5 ? 2.4 : 1.2)).attr('stroke-dasharray', (d) => (d.p === 0.5 ? null : '5 4'));
    const pr = D.map((d) => sigmoid(f.b0 + f.w[0] * d.ifng + f.w[1] * d.log2tmb));
    s5.set({ 'AUC (discovery)': auc(pr, y).toFixed(3), 'log-loss': logLoss(pr, y).toFixed(3), 'weights': mode5 === 'one' ? `IFN-γ ${f2(f.w[0])}` : `IFN-γ ${f2(f.w[0])}, log2 TMB ${f2(f.w[1])}` });
  }
  paint5();
  prose(rb, `<p>With IFN-γ alone the boundary is vertical. Adding TMB tilts it, so a patient with moderate IFN-γ and high TMB can be classified as a likely responder. The slope gives the trade-off: one unit of IFN-γ is equivalent to ${f2(fit2.w[0] / fit2.w[1])} doublings of TMB.</p>`);

  // ---- Step 7: reading coefficients
  rb = rung(ch, 5, 'Interpreting the coefficients',
    `<p>Three-feature model. Each weight is the log odds ratio <b>per one-unit change in that feature, with the other features held fixed</b>. Because the data are simulated, the last column gives the true value.</p>`);
  const coefRows = F3.map((k, j) => {
    const w = fit3.w[j], se = fit3.se[j + 1];
    return { k, lab: FEATURES[k].label, w, se, lo: Math.exp(w - 1.96 * se), hi: Math.exp(w + 1.96 * se), or: Math.exp(w), p: pNormal(w / se), t: TRUE_NAT[k] };
  });
  table(rb, coefRows, [['lab', 'Feature'], ['w', 'Weight (log-OR)', f2], ['se', 'SE', f2], ['or', 'Odds ratio', f2], ['lo', '95% CI for OR', (v, r) => `${f2(r.lo)} – ${f2(r.hi)}`], ['p', 'p-value', pFmt], ['t', 'True weight', f2]], { cls: 'compact' });
  const cr = Object.fromEntries(coefRows.map((r) => [r.k, r]));
  const w1 = cr.ifng.w, wt = cr.log2tmb.w, orT = Math.exp(wt);
  const newP = (p0, or) => { const o = (p0 / (1 - p0)) * or; return o / (1 + o); };
  prose(rb, `<ul>
    <li><b>IFN-γ:</b> each +1 log2 unit multiplies the odds by <b>${f2(cr.ifng.or)}</b>; from 5 to 7, by e<sup>2 × ${f2(w1)}</sup> = ${f2(Math.exp(2 * w1))}. Effects add on the log-odds scale and multiply on the odds scale.</li>
    <li><b>TMB:</b> each doubling multiplies the odds by <b>${f2(orT)}</b>.</li>
    <li><b>CAF score:</b> OR <b>${f2(cr.caf.or)}</b>${cr.caf.hi > 1 ? `, but the 95% CI crosses 1 although the true effect is non-zero (${f2(TRUE_NAT.caf)}): a real effect can be missed at n = ${n}` : ''}.</li>
    <li>The same OR changes probability most near 50%: doubling TMB moves ${pc(0.1)} to ${pc(newP(0.1, orT))}, but ${pc(0.5)} to ${pc(newP(0.5, orT))}.</li></ul>`);

  const pos = D.filter((p) => p.response), negs = D.filter((p) => !p.response);
  const by = (arr, f) => arr.slice().sort((a, b) => f(a) - f(b))[0];
  const vign = [
    { key: 'hot', name: 'Highest predicted probability', p: by(pos, (p) => -prob3(p)), note: 'High IFN-γ, high TMB and a low CAF score all increase the predicted probability.' },
    { key: 'desert', name: 'Immune desert', p: by(negs.filter((p) => p.phenotype === 'desert'), (p) => prob3(p)), note: 'Little immune activity; very low predicted probability.' },
    { key: 'border', name: 'Borderline', p: by(D, (p) => Math.abs(prob3(p) - 0.5)), note: 'Predicted probability close to 0.5.' },
    { key: 'tmb', name: 'Excluded, high TMB', p: by(D.filter((p) => p.phenotype === 'excluded'), (p) => -p.tmb), note: 'High TMB, but T cells are confined to the stroma.' },
    { key: 'wrong', name: 'False positive', p: by(negs, (p) => -prob3(p)), note: 'High predicted probability, but no response: three features do not capture all relevant biology.' },
    { key: 'surprise', name: 'False negative', p: by(pos, (p) => prob3(p)), note: 'Responded despite a low predicted probability.' },
  ];
  const fx = figure(rb, 'Patient calculator', 'Select a patient or enter custom values. Bars show each feature\'s contribution to the log-odds relative to a patient with average values.');
  let vk = 'hot', custom = { ifng: 6, tmb: 8, caf: 4 };
  const custWrap = el('div', { class: 'controls' });
  select(fx.controls, { label: 'Patient', value: vk, options: [...vign.map((v) => [v.key, `${v.name} (${v.p.id})`]), ['custom', 'Custom values']], onChange: (v) => { vk = v; custWrap.style.display = v === 'custom' ? '' : 'none'; paintX(); } });
  fx.controls.after(custWrap); custWrap.style.display = 'none';
  slider(custWrap, { label: 'IFN-γ', min: 2.5, max: 10, step: 0.05, value: 6, fmt: f2, onInput: (v) => { custom.ifng = v; paintX(); } });
  slider(custWrap, { label: 'TMB (mut/Mb)', min: 1, max: 40, step: 0.5, value: 8, fmt: f1, onInput: (v) => { custom.tmb = v; paintX(); } });
  slider(custWrap, { label: 'CAF score', min: 1.5, max: 7, step: 0.05, value: 4, fmt: f2, onInput: (v) => { custom.caf = v; paintX(); } });
  const rowX = el('div', { class: 'fig-row' }); fx.body.append(rowX);
  const xa = el('div'), xb = el('div'); rowX.append(xa, xb);
  const means3 = F3.map((k) => mean(D.map((p) => FEATURES[k].get(p))));
  const zBar = fit3.b0 + sum(fit3.w.map((w, j) => w * means3[j]));
  const FX = frame(xb, { width: 460, height: 230, x: [-5, 5], y: [0, 4], xLabel: 'push on the log-odds vs the average patient', grid: false, axes: false, margin: { left: 120, top: 8, right: 40 } });
  FX.g.append('g').attr('class', 'axis x').attr('transform', `translate(0,${FX.h})`).call(d3.axisBottom(FX.x).ticks(7));
  FX.layer('z').append('line').attr('x1', FX.x(0)).attr('x2', FX.x(0)).attr('y1', 0).attr('y2', FX.h).attr('stroke', muted());
  const barL = FX.layer('bars');
  function paintX() {
    const v = vign.find((q) => q.key === vk);
    const P = v ? v.p : { id: 'Custom', ifng: custom.ifng, tmb: custom.tmb, log2tmb: Math.log2(custom.tmb), caf: custom.caf };
    const xsP = F3.map((k) => FEATURES[k].get(P));
    const z = fit3.b0 + sum(fit3.w.map((w, j) => w * xsP[j])), odds = Math.exp(z), p = odds / (1 + odds);
    const terms = F3.map((k, j) => `${sgn(fit3.w[j])} × ${f2(xsP[j])}   (${FEATURES[k].label}${k === 'log2tmb' ? ': TMB ' + f1(P.tmb) + ' mut/Mb' : ''})`);
    xa.innerHTML = '';
    if (v) xa.append(html(`<p><b>${P.id}</b> · ${P.phenotype} tumour · ${P.centre} · age ${P.age} · PD-L1 ${P.pdl1}% · ${yes(P.response)}<br><span class="fig-caption">${v.note}</span></p>`));
    xa.append(html(`<div class="calc">z = ${f2(fit3.b0)}  (intercept)\n${terms.map((t) => '    ' + t).join('\n')}\n  = <b>${f2(z)}</b>   (log-odds)\n\nodds = e^${f2(z)} = <b>${f2(odds)}</b>\np = odds / (1 + odds) = <b>${f2(p)}</b>  →  ${pc(p)} chance of response${v ? `\n\ntrue probability in the simulation: ${f2(P.p_true)}` : ''}</div>`));
    const contrib = F3.map((k, j) => ({ lab: FEATURES[k].label, c: fit3.w[j] * (xsP[j] - means3[j]) }));
    contrib.push({ lab: 'Total vs average', c: z - zBar, tot: true });
    const yb = d3.scaleBand().domain(contrib.map((c) => c.lab)).range([0, FX.h]).padding(0.3);
    barL.selectAll('rect').data(contrib).join('rect').attr('y', (d) => yb(d.lab)).attr('height', yb.bandwidth())
      .attr('x', (d) => FX.x(Math.min(0, Math.max(-5, d.c)))).attr('width', (d) => Math.abs(FX.x(Math.max(-5, Math.min(5, d.c))) - FX.x(0)))
      .attr('fill', (d) => (d.tot ? ink2() : d.c >= 0 ? cat(1) : cat(0)));
    barL.selectAll('text.lab').data(contrib).join('text').attr('class', 'lab').attr('x', -8).attr('y', (d) => yb(d.lab) + yb.bandwidth() / 2 + 4).attr('text-anchor', 'end').attr('font-size', 12).attr('fill', ink2()).text((d) => d.lab);
    barL.selectAll('text.val').data(contrib).join('text').attr('class', 'val').attr('font-size', 11.5).attr('fill', ink())
      .attr('x', (d) => FX.x(Math.max(-5, Math.min(5, d.c))) + (d.c >= 0 ? 4 : -4)).attr('text-anchor', (d) => (d.c >= 0 ? 'start' : 'end'))
      .attr('y', (d) => yb(d.lab) + yb.bandwidth() / 2 + 4).text((d) => (d.c >= 0 ? '+' : '') + f2(d.c));
  }
  paintX();
  prose(rb, `<p class="fig-caption">Reference: average feature values give log-odds ${f2(zBar)} (p = ${f2(sigmoid(zBar))}; below the ${pc(base)} response rate because the sigmoid is non-linear).</p>`);

  rb = rung(ch, 6, 'Model builder: feature selection, confounding and thresholds',
    `<p>Fitted on the ${n} discovery patients (Centres A, B); evaluated on the ${V.length} independent validation patients (Centre C). Features are standardised, so weights are per standard deviation.</p>`);
  const LAB = ['ifng', 'log2tmb', 'caf', 'pdl1', 'cd8', 'prolif', 'age', 'smoker', 'male', 'centreB'];
  const sdD = Object.fromEntries(LAB.map((k) => [k, sd(D.map((p) => FEATURES[k].get(p)))]));
  const fl = figure(rb, 'Model builder', 'Left: each feature\'s weight per standard deviation (so features are comparable), with 95% interval; the open diamond is the true value used in the simulation. Right: the predicted probability for every validation patient, split by what actually happened. Drag the threshold to see the trade-off.');
  const sel = new Set(['ifng']);
  const boxes = {};
  const presets = el('div', { class: 'controls' });
  const featRow = el('div', { class: 'controls' });
  fl.controls.before(presets); fl.controls.before(featRow);
  presets.append(el('span', { class: 'fig-caption', style: 'margin:0' }, 'Presets:'));
  const PRE = [['IFN-γ only', ['ifng']], ['True three', ['ifng', 'log2tmb', 'caf']], ['PD-L1 only', ['pdl1']], ['IFN-γ + PD-L1', ['ifng', 'pdl1']], ['Centre only', ['centreB']], ['Centre + IFN-γ', ['ifng', 'centreB']], ['Everything', LAB]];
  PRE.forEach(([lab, ks]) => button(presets, lab, () => { sel.clear(); ks.forEach((k) => sel.add(k)); LAB.forEach((k) => { boxes[k].checked = sel.has(k); }); paintL(); }));
  LAB.forEach((k) => { boxes[k] = checkbox(featRow, { label: FEATURES[k].label, checked: sel.has(k), onChange: (c) => { if (c) sel.add(k); else sel.delete(k); paintL(); } }); });
  const L2S = [0, 0.3, 1, 3, 10, 30, 100];
  let l2i = 0, thrL = 0.5;
  slider(fl.controls, { label: 'L2 penalty λ', min: 0, max: L2S.length - 1, step: 1, value: 0, fmt: (i) => String(L2S[i]), onInput: (i) => { l2i = i; paintL(); } });
  slider(fl.controls, { label: 'threshold', min: 0.05, max: 0.95, step: 0.01, value: 0.5, fmt: (v) => v.toFixed(2), onInput: (v) => { thrL = v; paintL(); } });
  const sL = stats(fl.root);
  const rowL = el('div', { class: 'fig-row' }); fl.body.append(rowL);
  const la = el('div'), lb = el('div'); rowL.append(la, lb);
  const coefSvg = d3.select(la).append('svg').attr('width', 480).attr('role', 'img');
  const FS = frame(lb, { width: 480, height: 300, x: [0, 1], y: [-0.6, 1.6], xLabel: 'predicted P(response), validation patients', grid: false, axes: false });
  FS.g.append('g').attr('class', 'axis x').attr('transform', `translate(0,${FS.h})`).call(d3.axisBottom(FS.x).ticks(5));
  FS.layer('lab').selectAll('text').data([[0, 'no response'], [1, 'responded']]).join('text').attr('x', 4).attr('y', (d) => FS.y(d[0] + 0.42)).attr('font-size', 12).attr('fill', ink2()).text((d) => d[1]);
  const stripL = FS.layer('strip'), thrLine = FS.layer('thr');
  const msgL = html('', 'p', 'fig-caption'); fl.body.append(msgL);
  legend(fl.body, [['no response', cat(0)], ['responded', cat(1)], ['true effect', ink()]]);
  function paintL() {
    const keys = LAB.filter((k) => sel.has(k));
    coefSvg.selectAll('*').remove(); stripL.selectAll('*').remove(); thrLine.selectAll('*').remove();
    if (!keys.length) { sL.set({ note: 'select at least one feature' }); msgL.innerHTML = ''; return; }
    const X = rowsOf(D, keys), st = standardize(X);
    const fit = fitLogistic(st.Z, y, { l2: L2S[l2i] });
    const pt = st.Z.map((r) => predictProb(r, fit.b0, fit.w));
    const pv = V.map((p) => predictProb(st.apply(keys.map((k) => FEATURES[k].get(p))), fit.b0, fit.w));
    // coefficient plot
    const rows = keys.map((k, j) => ({ k, w: fit.w[j], se: fit.se[j + 1], t: (TRUE_NAT[k] || 0) * sdD[k] }));
    const ext = Math.max(1, d3.max(rows, (r) => Math.max(Math.abs(r.w) + 1.96 * r.se, Math.abs(r.t)))) * 1.08;
    const m = { l: 130, r: 14, t: 10, b: 40 }, W = 480 - m.l - m.r, Hh = 34 * keys.length;
    coefSvg.attr('viewBox', `0 0 480 ${Hh + m.t + m.b}`);
    const g = coefSvg.append('g').attr('transform', `translate(${m.l},${m.t})`);
    const xS = d3.scaleLinear().domain([-Math.min(ext, 4), Math.min(ext, 4)]).range([0, W]).clamp(true);
    const yS = d3.scaleBand().domain(keys).range([0, Math.min(Hh, 34 * keys.length)]).padding(0.4);
    const hh = Math.min(Hh, 34 * keys.length);
    g.append('g').attr('class', 'axis x').attr('transform', `translate(0,${hh})`).call(d3.axisBottom(xS).ticks(5));
    g.append('text').attr('class', 'ax-label').attr('x', W / 2).attr('y', hh + 34).attr('text-anchor', 'middle').text('weight per 1 SD (log-odds)');
    g.append('line').attr('x1', xS(0)).attr('x2', xS(0)).attr('y1', 0).attr('y2', hh).attr('stroke', muted()).attr('stroke-dasharray', '4 4');
    rows.forEach((r) => {
      const cy = yS(r.k) + yS.bandwidth() / 2;
      g.append('text').attr('x', -8).attr('y', cy + 4).attr('text-anchor', 'end').attr('font-size', 12).attr('fill', ink2()).text(FEATURES[r.k].label);
      g.append('line').attr('x1', xS(r.w - 1.96 * r.se)).attr('x2', xS(r.w + 1.96 * r.se)).attr('y1', cy).attr('y2', cy).attr('stroke', ink2()).attr('stroke-width', 2);
      g.append('circle').attr('cx', xS(r.w)).attr('cy', cy).attr('r', 5.5).attr('fill', cat(r.w >= 0 ? 1 : 0)).attr('stroke', css('--surface')).attr('stroke-width', 1.5)
        .on('mousemove', (e) => tip(e, `${FEATURES[r.k].label}<br>weight per SD ${f2(r.w)} ± ${f2(1.96 * r.se)}<br>odds ratio per SD ${f2(Math.exp(r.w))}<br>true: ${f2(r.t)}`)).on('mouseleave', untip);
      g.append('path').attr('d', d3.symbol(d3.symbolDiamond, 70)).attr('transform', `translate(${xS(r.t)},${cy})`).attr('fill', 'none').attr('stroke', ink()).attr('stroke-width', 1.5);
    });
    // strip
    stripL.selectAll('circle').data(V).join('circle').attr('cx', (d, i) => FS.x(pv[i])).attr('cy', (d, i) => FS.y(d.response + jit(i) * 3.4)).attr('r', 4)
      .attr('fill', (d) => cat(d.response)).attr('fill-opacity', 0.8).attr('stroke', css('--surface')).attr('stroke-width', 0.8)
      .on('mousemove', (e, d) => { const i = V.indexOf(d); tip(e, `${d.id} (${d.phenotype})<br>predicted ${f2(pv[i])} · ${d.response ? 'responded' : 'no response'}`); }).on('mouseleave', untip);
    thrLine.append('line').attr('x1', FS.x(thrL)).attr('x2', FS.x(thrL)).attr('y1', 0).attr('y2', FS.h).attr('stroke', ink()).attr('stroke-width', 2);
    thrLine.append('text').attr('x', FS.x(thrL) + 5).attr('y', 12).attr('font-size', 11.5).attr('fill', ink()).text('treat →');
    const cm = confusion(pv, yv, thrL);
    sL.set({ 'train AUC': auc(pt, y).toFixed(3), 'validation AUC': auc(pv, yv).toFixed(3), 'validation log-loss': logLoss(pv, yv).toFixed(3), 'responders caught': `${cm.tp}/${cm.tp + cm.fn}`, 'non-responders treated': `${cm.fp}/${cm.fp + cm.tn}`, converged: fit.converged ? 'yes' : 'NO' });
    msgL.innerHTML = L2S[l2i] > 0 ? 'With a penalty, the error bars come from the penalised fit and are only approximate; they illustrate how the weights shrink towards 0.' : '';
  }
  paintL();
  const valAUC = (keys, l2 = 0) => { const st = standardize(rowsOf(D, keys)); const f = fitLogistic(st.Z, y, { l2 }); return auc(V.map((p) => predictProb(st.apply(keys.map((k) => FEATURES[k].get(p))), f.b0, f.w)), yv); };
  const fP = fitLogistic(D.map((p) => [p.pdl1]), y), fPI = fitLogistic(D.map((p) => [p.ifng, p.pdl1]), y);
  const fC = fitLogistic(D.map((p) => [p.centreB]), y), fCI = fitLogistic(D.map((p) => [p.ifng, p.centreB]), y);
  const fPr = fitLogistic(D.map((p) => [p.prolif]), y), pPr = pNormal(fPr.w[0] / fPr.se[1]);
  const ciTxt = (w, se) => `${f2(Math.exp(w))} (${f2(Math.exp(w - 1.96 * se))}–${f2(Math.exp(w + 1.96 * se))})`;
  const pv3 = V.map(prob3), c50 = confusion(pv3, yv, 0.5), c25 = confusion(pv3, yv, 0.25);
  const inflB = D.filter((p) => p.centreB && p.phenotype === 'inflamed').length / D.filter((p) => p.centreB).length;
  const inflA = D.filter((p) => !p.centreB && p.phenotype === 'inflamed').length / D.filter((p) => !p.centreB).length;
  prose(rb, '<p><b>Suggested comparisons</b>:</p>');
  table(rb, [
    { t: '<b>IFN-γ only</b> → <b>True three</b>', s: `Validation AUC ${valAUC(['ifng']).toFixed(3)} → <b>${valAUC(F3).toFixed(3)}</b>; weights land near the diamonds`, m: 'Informative features improve performance on new patients; a correctly specified model recovers the true effects.' },
    { t: '<b>PD-L1 only</b> → <b>IFN-γ + PD-L1</b>', s: `Odds ratio per +10% TPS: ${f2(Math.exp(10 * fP.w[0]))} (p ${pFmt(pNormal(fP.w[0] / fP.se[1]))}) → <b>${f2(Math.exp(10 * fPI.w[1]))}</b> (p ${pFmt(pNormal(fPI.w[1] / fPI.se[2]))})`, m: `<b>A weight depends on what else is in the model.</b> PD-L1 is induced by IFN-γ (r = ${f2(corr(D.map((p) => p.ifng), D.map((p) => p.pdl1)))}): it predicts response, but adds nothing beyond IFN-γ.` },
    { t: '<b>Centre only</b> → <b>Centre + IFN-γ</b>', s: `Centre B odds ratio (95% CI): ${ciTxt(fC.w[0], fC.se[1])} → <b>${ciTxt(fCI.w[1], fCI.se[2])}</b>`, m: `<b>Confounding.</b> Centre B enrolled more inflamed tumours (${pc(inflB)} vs ${pc(inflA)}). The model reports associations, not causes.` },
    { t: '<b>Proliferation</b> alone', s: `Odds ratio ${ciTxt(fPr.w[0], fPr.se[1])}, p = ${pFmt(pPr)}`, m: `<b>Noise makes non-zero weights.</b> It has no true effect on response (it only shortens survival). ${pPr < 0.05 ? 'Here it is a false positive.' : 'The interval includes 1.'} Test 20,000 genes and ~1,000 pass p &lt; 0.05 by chance.` },
    { t: '<b>Everything</b>, then raise λ', s: `Validation AUC ${valAUC(LAB).toFixed(3)} (λ = 0) vs ${valAUC(LAB, 10).toFixed(3)} (λ = 10); useless weights shrink first`, m: '<b>Regularisation</b> trades a little bias for stability. Uninformative features add little here; with thousands of genes, an unpenalised model would overfit.' },
    { t: 'Threshold <b>0.50</b> → <b>0.25</b> (True three)', s: `Responders caught ${c50.tp}/${c50.tp + c50.fn} → <b>${c25.tp}/${c25.tp + c25.fn}</b>; non-responders treated ${c50.fp} → <b>${c25.fp}</b>`, m: '<b>The threshold is a decision</b> that depends on the relative cost of false positives and false negatives.' },
  ], [['t', 'Try'], ['s', 'What you see'], ['m', 'What it means']], { cls: 'compact' });
  prose(rb, `<p><b>Calibration.</b> In Centre C the model predicted ${pc(mean(pv3))} response on average, against ${pc(mean(yv))} observed. Discrimination (AUC) and calibration are separate properties; predicted probabilities should be checked, and recalibrated if necessary, in each new setting.</p>`);

  // =====================================================================================
  // 8. PITFALLS
  // =====================================================================================
  ch = chapter(main, 'pitfalls', 'Pitfalls', 'Common problems, how to recognise them, and how to address them.');
  const small = D.slice(0, 20), ySmall = small.map((p) => p.response);
  const stS = standardize(rowsOf(small, LAB));
  const fS0 = fitLogistic(stS.Z, ySmall), fS1 = fitLogistic(stS.Z, ySmall, { l2: 1 });
  cards(ch, [
    { tag: 'Symptom: huge weights, "did not converge"', title: 'Perfect separation', body: `When some combination of features splits the classes perfectly, the maximum-likelihood fit is an infinitely steep curve and the weights diverge. Common when features ≫ patients. Fitting all ${LAB.length} features on just ${small.length} patients: largest |weight| = <b>${f1(d3.max(fS0.w, Math.abs))}</b> per SD (converged: ${fS0.converged ? 'yes' : 'no'}). With a small L2 penalty (λ = 1): <b>${f2(d3.max(fS1.w, Math.abs))}</b>. <b>Fix:</b> regularise (ridge / lasso / elastic net), or use Firth's penalised likelihood.` },
    { tag: 'Symptom: unstable weights', title: 'Too many features for the events', body: `The information is in the <i>smaller</i> class: here ${nR} responders, not ${n} patients. A common rule of thumb is ≥ 10–20 events per feature, so this cohort supports roughly ${Math.floor(nR / 20)}–${Math.floor(nR / 10)} features unpenalised. <b>Fix:</b> pre-specify a few features, or penalise and validate.` },
    { tag: 'Symptom: high training AUC, lower AUC on new data', title: 'Leakage', body: 'Selecting genes, tuning λ or standardising on all patients before splitting lets the test set leak into training. <b>Fix:</b> every data-driven step goes inside the cross-validation loop, and the final check is on patients never touched (like Centre C).' },
    { tag: 'Symptom: weights you cannot compare', title: 'Unscaled features', body: 'A weight is per unit: per % PD-L1, per year of age, per log2 unit. Raw weights are therefore not comparable across features, and penalties hit small-unit features harder. <b>Fix:</b> standardise (per SD) when comparing or penalising; report natural units for interpretation.' },
    { tag: 'Symptom: residual pattern in binned plots', title: 'Non-linear effects', body: 'The model assumes each feature adds a constant amount to the log-odds. Skewed variables (TMB, counts) or U-shaped effects break this. <b>Fix:</b> log-transform skewed features, inspect binned log-odds against each feature, add splines or interaction terms if needed.' },
    { tag: `Symptom: "${Math.round(OR)}× more likely!"`, title: 'Reading an odds ratio as a risk ratio', body: `High vs low IFN-γ: odds ratio ${f1(OR)}, risk ratio ${f1(rr)}. When the outcome is common, the odds ratio is much bigger than the risk ratio. <b>Fix:</b> report predicted probabilities or risk differences alongside odds ratios.` },
    { tag: 'Symptom: correlated features swap signs', title: 'Collinearity', body: `IFN-γ, CD8 % and PD-L1 are strongly correlated (r = ${f2(corr(D.map((p) => p.ifng), D.map((p) => p.cd8)))} for IFN-γ vs CD8). Included together, their weights become unstable and hard to interpret, although predictions are largely unaffected. <b>Fix:</b> pick one, combine them into a score, or use ridge.` },
    { tag: 'Symptom: accuracy looks great', title: 'Imbalance and the default threshold', body: `Only ${pc(base)} respond, so "predict nobody responds" is already ${pc(1 - base)} accurate. <b>Fix:</b> judge with AUC, log-loss and calibration, and choose the threshold on purpose (chapter 4, step 6).` },
  ]);

  // =====================================================================================
  // 9. RECAP
  // =====================================================================================
  ch = chapter(main, 'recap', 'Recap', 'Key points.');
  recap(ch, [
    'Logistic regression predicts the <b>probability of a yes/no outcome</b>: weighted sum → log-odds → sigmoid → probability.',
    'It assumes each feature adds a constant amount to the <b>log-odds</b>. Check it by plotting binned log-odds; log-transform skewed features.',
    'A weight is a <b>log odds ratio</b>: e<sup>w</sup> is how much the odds multiply per unit of the feature, <i>holding the other features fixed</i>.',
    'It is fitted by minimising <b>log-loss</b> (maximising likelihood). The loss is a single smooth bowl, so the answer is unique and fast to find.',
    'The decision boundary is a <b>straight line</b> (a flat plane in more dimensions). Curved boundaries need new features or a different model.',
    'Odds ratios are not risk ratios; weights depend on what else is in the model; association is not causation.',
    'With many features or few events, <b>regularise</b> and validate on patients the model has never seen. Judge it with AUC, log-loss and calibration, not accuracy.',
    'The threshold that turns a probability into a decision is a <b>clinical choice</b>.',
  ]);
  prose(ch, '<h3>In code</h3><p>Use a stats package when you want odds ratios and p-values (inference), and a machine-learning package when you want the best predictions on new patients (with standardisation, a penalty and held-out evaluation).</p>');
  code(ch, {
    Python: `import numpy as np, statsmodels.formula.api as smf
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

train["log2tmb"] = np.log2(train["tmb"]);  test["log2tmb"] = np.log2(test["tmb"])

# 1) Inference: weights, 95% CIs, p-values, odds ratios
m = smf.logit("response ~ ifng + log2tmb + caf", data=train).fit()
print(m.summary())
print(np.exp(m.conf_int().assign(OR=m.params)))

# 2) Prediction: standardise + L2 penalty, judged on held-out patients
feats = ["ifng", "log2tmb", "caf"]
clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0))   # C = 1/lambda for a penalty on the summed log-loss (step 6)
clf.fit(train[feats], train["response"])
p = clf.predict_proba(test[feats])[:, 1]
print("validation AUC", roc_auc_score(test["response"], p))`,
    R: `train$log2tmb <- log2(train$tmb); test$log2tmb <- log2(test$tmb)

# 1) Inference: weights, 95% CIs, p-values, odds ratios
m <- glm(response ~ ifng + log2tmb + caf, family = binomial, data = train)
summary(m)
exp(cbind(OR = coef(m), confint(m)))

# 2) Prediction on held-out patients
p <- predict(m, newdata = test, type = "response")
pROC::auc(test$response, p)

# Many features (e.g. genes): penalised, lambda chosen by cross-validation
# fit <- glmnet::cv.glmnet(x, y, family = "binomial", alpha = 0)   # alpha = 1 for lasso`,
  });
  seeAlso(main, [['concepts', 'linear-regression', 'Linear regression'], ['concepts', 'gradient-descent', 'Gradient descent'], ['concepts', 'regularization', 'Regularisation'], ['concepts', 'metrics', 'Evaluating a classifier'], ['omics', 'bulk-classifier', 'Biomarker classifier and nested CV']]);
}

// ---------------------------------------------------------------------------------------------
function qBins(pts, key, k) {
  const s = pts.slice().sort((a, b) => key(a) - key(b)), out = [];
  for (let i = 0; i < k; i++) {
    const part = s.slice(Math.round(i * s.length / k), Math.round((i + 1) * s.length / k));
    out.push({ x: mean(part.map(key)), lo: key(part[0]), hi: key(part[part.length - 1]), n: part.length, r: part.filter((p) => p.response).length });
  }
  return out;
}
function corr(a, b) {
  const ma = mean(a), mb = mean(b);
  let sab = 0, saa = 0, sbb = 0;
  a.forEach((v, i) => { sab += (v - ma) * (b[i] - mb); saa += (v - ma) ** 2; sbb += (b[i] - mb) ** 2; });
  return sab / Math.sqrt(saa * sbb);
}

/** The "three moves" diagram for one patient. */
function pipelineSvg(P, fit) {
  const wrap = el('div', { class: 'figure' });
  wrap.append(el('div', { class: 'fig-title' }, `The three moves, for patient ${P.id}`));
  const svg = d3.select(wrap).append('svg').attr('viewBox', '0 0 980 230').attr('width', 980).attr('role', 'img').attr('aria-label', 'measurements, weighted sum, sigmoid, probability');
  const { b0, w } = fit;
  const xsP = [P.ifng, P.log2tmb, P.caf];
  const z = b0 + w[0] * xsP[0] + w[1] * xsP[1] + w[2] * xsP[2], p = sigmoid(z);
  const boxes = [
    { t: 'Measurements', lines: [`IFN-γ score  ${f2(P.ifng)}`, `TMB  ${f1(P.tmb)} mut/Mb`, `  → log2 TMB  ${f2(P.log2tmb)}`, `CAF score  ${f2(P.caf)}`] },
    { t: '① Weighted sum → z', lines: [`z = ${f2(b0)}`, `  ${sgn(w[0])} × ${f2(xsP[0])}`, `  ${sgn(w[1])} × ${f2(xsP[1])}`, `  ${sgn(w[2])} × ${f2(xsP[2])}`, `  = ${f2(z)}`] },
    { t: '② Sigmoid squashes z', lines: [], sig: true },
    { t: '③ Probability → decision', lines: [`p = ${f2(p)}  (${pc(p)})`, `p ≥ 0.5 → predict`, `"${p >= 0.5 ? 'responds' : 'does not respond'}"`, '', `Actually: ${P.response ? 'responded ✓' : 'no response'}`] },
  ];
  const BW = 212, gap = 36, H = 200;
  boxes.forEach((b, i) => {
    const x0 = 10 + i * (BW + gap), g = svg.append('g').attr('transform', `translate(${x0},14)`);
    g.append('rect').attr('width', BW).attr('height', H).attr('rx', 10).attr('fill', css('--surface-2')).attr('stroke', css('--axis'));
    g.append('text').attr('x', 12).attr('y', 24).attr('font-weight', 700).attr('font-size', 13.5).attr('fill', ink()).text(b.t);
    b.lines.forEach((l, j) => g.append('text').attr('x', 14).attr('y', 54 + j * 24).attr('font-family', css('--mono')).attr('font-size', 13).attr('fill', ink2()).attr('xml:space', 'preserve').text(l));
    if (b.sig) {
      const sx = d3.scaleLinear().domain([-6, 6]).range([16, BW - 16]), sy = d3.scaleLinear().domain([0, 1]).range([H - 30, 46]);
      g.append('line').attr('x1', sx(-6)).attr('x2', sx(6)).attr('y1', sy(0.5)).attr('y2', sy(0.5)).attr('stroke', muted()).attr('stroke-dasharray', '3 3');
      g.append('path').datum(d3.range(-6, 6.01, 0.1)).attr('fill', 'none').attr('stroke', cat(6)).attr('stroke-width', 2.5).attr('d', d3.line().x((v) => sx(v)).y((v) => sy(sigmoid(v))));
      const zc = Math.max(-6, Math.min(6, z));
      g.append('line').attr('x1', sx(zc)).attr('x2', sx(zc)).attr('y1', sy(0)).attr('y2', sy(p)).attr('stroke', ink2()).attr('stroke-dasharray', '3 3');
      g.append('circle').attr('cx', sx(zc)).attr('cy', sy(p)).attr('r', 6).attr('fill', cat(6)).attr('stroke', css('--surface')).attr('stroke-width', 1.5);
      g.append('text').attr('x', 14).attr('y', H - 10).attr('font-family', css('--mono')).attr('font-size', 12.5).attr('fill', ink2()).text(`σ(${f2(z)}) = ${f2(p)}`);
      g.append('text').attr('x', 6).attr('y', sy(1) + 4).attr('text-anchor', 'start').attr('font-size', 11).attr('fill', muted()).text('1');
      g.append('text').attr('x', 6).attr('y', sy(0) + 4).attr('text-anchor', 'start').attr('font-size', 11).attr('fill', muted()).text('0');
    }
    if (i < boxes.length - 1) {
      const ax = x0 + BW + 6, ay = 14 + H / 2;
      svg.append('path').attr('d', `M${ax},${ay} L${ax + gap - 14},${ay} M${ax + gap - 20},${ay - 6} L${ax + gap - 14},${ay} L${ax + gap - 20},${ay + 6}`).attr('fill', 'none').attr('stroke', ink2()).attr('stroke-width', 2);
    }
  });
  return wrap;
}
