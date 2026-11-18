// Survival analysis and risk scores (lesson). Bulk RNA-seq step 4 of 4.
// Arc: what → why → where → how it works (6 steps: Kaplan–Meier by hand → log-rank and the cut-point problem →
//      Cox model and hazard ratios → C-index → Cox elastic-net risk-score lab with Centre C validation and
//      try/see/means table → prognostic vs predictive with the control arm) → pitfalls → recap.
// Data: data/omics/bulk_survival.json (scripts/build_bulk_extra.py: PFS of 177 anti-PD-1 discovery patients and a
// simulated docetaxel control arm, module scores, CV path) and data/omics/bulk_survival_lab.json
// (scripts/build_survival_lesson.py: elastic-net Cox path with per-λ genes, C-indices and Centre C risk scores).
// Kaplan–Meier, log-rank, Cox (Newton, Breslow ties) and the C-index run live in the browser.
import { header, figure, slider, select, button, stats, legend, seeAlso, animate, tip, untip, el } from '../lib/ui.js';
import { frame, cat, css, ink, ink2, muted } from '../lib/plot.js';
import { rng, solve, mean } from '../lib/data.js';
import { pNormal } from '../lib/models.js';
import { lessonNav, chapter, prose, rung, worked, quiz, cards, table, code, recap } from '../lib/lesson.js';
import { stepCard, omicsData } from './_shared.js';
import { groupColor, groupLabel } from './_bulk_common.js';
const d3 = window.d3;

const f2 = (v) => (Math.abs(v) < 0.005 ? '0.00' : v.toFixed(2));
const f1 = (v) => (v == null || !isFinite(v) ? 'not reached' : v.toFixed(1));
const f3 = (v) => v.toFixed(3);
const pc = (v) => Math.round(100 * v) + '%';
const pE = (p) => (p < 0.001 ? p.toExponential(1) : p.toFixed(3));
const sum = (a) => a.reduce((s, v) => s + v, 0);
const zs = (a, ref = a) => { const m = mean(ref), s = Math.sqrt(mean(ref.map((v) => (v - m) ** 2))); return a.map((v) => (v - m) / s); };

export default async function render(main) {
  header(main, 'Survival analysis and risk scores',
    'Methods for outcomes measured as <b>time until an event</b>, such as progression or death, when some patients have not yet had the event at the end of follow-up. They estimate survival curves, compare groups, quantify how a biomarker changes the event rate, and combine many genes into a <b>risk score</b>. A separate question, which only a control arm can answer, is whether a marker identifies patients who <b>benefit from the treatment</b> or only patients who do well regardless.');
  lessonNav(main);
  stepCard(main, {
    stage: 'Bulk RNA-seq · step 4 of 4',
    question: 'Which tumours progress early on anti-PD-1, can a gene-expression risk score identify them in a new centre, and does that score reflect benefit from the drug or tumour biology that matters under any treatment?',
    data: 'Progression-free survival (PFS) for the 177 anti-PD-1 discovery patients that passed QC (Centres A and B, batch-corrected log-CPM), the 120 Centre C validation patients, and a simulated randomised docetaxel control arm (n = 160) with the same tumour biology and no immunotherapy benefit.',
    method: 'Kaplan–Meier curves and log-rank tests; Cox proportional-hazards regression; a Cox elastic-net risk score with the penalty chosen by cross-validated partial likelihood; Harrell\'s C-index; a treatment × marker interaction test.',
    verdict: 'both',
    why: ' Building a risk score from 1,933 genes is a penalised, cross-validated prediction task (machine learning). Estimating survival under censoring, hazard ratios with confidence intervals, and whether a marker is predictive of treatment benefit are classical statistical questions.',
  });
  const [S, L] = await Promise.all([omicsData(main, 'bulk_survival.json'), omicsData(main, 'bulk_survival_lab.json')]);
  if (!S || !L) return;

  // ------------------------------------------------------------------ data
  const IO = S.io, CT = S.ctrl, VA = L.val;
  const nIO = IO.t.length, nEv = sum(IO.e), nCens = nIO - nEv;
  const kmIO = km(IO.t, IO.e), medIO = kmMedian(kmIO), s12 = kmAt(kmIO, 12);
  const tMax = Math.ceil(d3.max([...IO.t, ...CT.t, ...VA.t]) / 6) * 6;
  const zIf = zs(IO.ifng), zPr = zs(IO.prolif);
  const coxIf = coxFit(zIf.map((v) => [v]), IO.t, IO.e), coxPr = coxFit(zPr.map((v) => [v]), IO.t, IO.e);
  const coxBoth = coxFit(zIf.map((v, i) => [v, zPr[i]]), IO.t, IO.e);
  const cIf = cindex(IO.t, IO.e, IO.ifng.map((v) => -v)), cPr = cindex(IO.t, IO.e, IO.prolif);
  const jSel = S.path.jsel, stSel = L.steps[jSel];

  // =====================================================================================
  // 1. WHAT
  // =====================================================================================
  let ch = chapter(main, 'what', 'What is survival analysis?', 'In one line: <b>time to an event, using every patient, including those whose event has not yet happened</b>.');
  prose(ch, `<p>Survival analysis (also called time-to-event analysis) models how long it takes until an event occurs. In oncology the event is usually death (overall survival, OS) or disease progression or death (progression-free survival, PFS). The defining feature of such data is <b>censoring</b>: when the analysis is run, some patients have not yet had the event. For them the exact time is unknown; it is only known to be <i>longer</i> than their follow-up. Of the ${nIO} anti-PD-1 discovery patients, <b>${nEv} progressed</b> during follow-up and <b>${nCens} were censored</b> (${pc(nCens / nIO)}).</p>
    <p>The figure shows a random sample of patients. Each bar is one patient's follow-up from the start of treatment. Lowering the follow-up cut-off truncates every patient's follow-up at that time, as in an earlier analysis of the same study (for simplicity, all patients are treated as starting on the same day): more patients become censored, and fewer events are available.</p>`);

  // ---- Figure: swimmer plot with analysis cut-off
  const r0 = rng(11), sw = r0.shuffle(d3.range(nIO)).slice(0, 22).sort((a, b) => IO.t[b] - IO.t[a]);
  const fs = figure(ch, 'Follow-up of 22 patients',
    'Filled circle: progression observed at that time (event). Open circle: censored, meaning the patient was progression-free at last contact or at the cut-off. Hover a bar for details.');
  let cut = tMax;
  slider(fs.controls, { label: 'follow-up cut-off', min: 6, max: tMax, step: 1, value: tMax, fmt: (v) => v + ' mo', onInput: (v) => { cut = v; paintSw(); } });
  const sSw = stats(fs.root);
  const FSw = frame(fs.body, { width: 820, height: 330, x: [0, tMax], y: [0, sw.length], xLabel: 'months since start of treatment', grid: false, axes: false, margin: { left: 60 } });
  FSw.g.append('g').attr('class', 'axis x').attr('transform', `translate(0,${FSw.h})`).call(d3.axisBottom(FSw.x).ticks(8));
  const bhS = FSw.h / sw.length, swL = FSw.layer('sw'), cutL = FSw.layer('cut');
  legend(fs.body, [['progressed (event)', cat(1)], ['censored', cat(0)]]);
  function paintSw() {
    const rows = sw.map((i, r) => { const t = Math.min(IO.t[i], cut), ev = IO.e[i] && IO.t[i] <= cut; return { i, r, t, ev }; });
    swL.selectAll('rect').data(rows).join('rect').attr('x', 0).attr('y', (d) => d.r * bhS + bhS * 0.3).attr('height', bhS * 0.4).attr('width', (d) => FSw.x(d.t))
      .attr('fill', (d) => (d.ev ? cat(1) : cat(0))).attr('fill-opacity', 0.45)
      .on('mousemove', (e, d) => tip(e, `patient ${d.i + 1}<br>followed ${f1(d.t)} months<br>${d.ev ? 'progressed' : 'censored (progression-free when last seen)'}${IO.resp[d.i] ? '<br>responder' : ''}`)).on('mouseleave', untip);
    swL.selectAll('circle').data(rows).join('circle').attr('cx', (d) => FSw.x(d.t)).attr('cy', (d) => d.r * bhS + bhS / 2).attr('r', 5)
      .attr('fill', (d) => (d.ev ? cat(1) : css('--surface'))).attr('stroke', (d) => (d.ev ? css('--surface') : cat(0))).attr('stroke-width', 2);
    swL.selectAll('text').data(rows).join('text').attr('x', -8).attr('y', (d) => d.r * bhS + bhS / 2 + 4).attr('text-anchor', 'end').attr('font-size', 11).attr('fill', muted()).text((d) => '#' + (d.i + 1));
    cutL.selectAll('line').data([cut]).join('line').attr('x1', FSw.x(cut)).attr('x2', FSw.x(cut)).attr('y1', 0).attr('y2', FSw.h).attr('stroke', ink2()).attr('stroke-dasharray', '4 4');
    const evAll = IO.t.filter((t, i) => IO.e[i] && t <= cut).length;
    sSw.set({ 'events in sample': rows.filter((d) => d.ev).length, 'censored in sample': rows.filter((d) => !d.ev).length, 'all patients: events / censored': `${evAll} / ${nIO - evAll}`, 'Kaplan–Meier median PFS': f1(kmMedian(km(IO.t.map((t) => Math.min(t, cut)), IO.e.map((e, i) => +(e && IO.t[i] <= cut))))) + (kmMedian(km(IO.t.map((t) => Math.min(t, cut)), IO.e.map((e, i) => +(e && IO.t[i] <= cut)))) == null ? '' : ' mo') });
  }
  paintSw();
  prose(ch, '<p>Key terms:</p>');
  const hr1 = Math.exp(coxPr.beta[0]);
  table(ch, [
    { t: 'Time (<i>t</i>) and event indicator (<i>e</i>)', m: 'Each patient contributes a time and a 0/1 flag: 1 = event observed at <i>t</i>, 0 = censored at <i>t</i>', s: `${nEv} events, ${nCens} censored` },
    { t: 'Censoring', m: 'The event time is only known to exceed the follow-up time (right-censoring)', s: `${pc(nCens / nIO)} of patients` },
    { t: 'Survival function <i>S</i>(<i>t</i>)', m: 'Probability of remaining event-free beyond time <i>t</i>', s: `S(12 months) = ${f2(s12)}` },
    { t: 'Median survival', m: 'The time at which <i>S</i>(<i>t</i>) first drops to 0.5 or below', s: `${f1(medIO)} months` },
    { t: 'Hazard <i>h</i>(<i>t</i>)', m: 'The instantaneous event rate at time <i>t</i> among patients still event-free', s: 'events per patient-month' },
    { t: 'Hazard ratio (HR)', m: 'Ratio of the hazards in two groups, or per unit of a marker. HR &gt; 1: events occur sooner', s: `${f2(hr1)} per SD of proliferation score` },
    { t: 'Risk score', m: 'A weighted sum of features from a Cox model; higher = earlier expected event', s: 'elastic-net Cox on 1,933 genes (step 5)' },
    { t: 'C-index', m: 'Probability that, of two comparable patients, the one with the higher risk score has the event first. 0.5 = random', s: `proliferation score alone: ${f3(cPr)}` },
  ], [['t', 'Term'], ['m', 'Meaning'], ['s', 'In this study']]);

  // =====================================================================================
  // 2. WHY
  // =====================================================================================
  ch = chapter(main, 'why', 'Why do we need it?', 'Standard summaries of the times either discard information or are biased when follow-up is incomplete.');
  prose(ch, `<p>For a time-to-event outcome, a useful method should:</p><ol>
    <li><b>Use censored patients.</b> A patient followed for 30 months without progression carries information: progression takes longer than 30 months.</li>
    <li><b>Keep the timing.</b> Progression at 2 months and at 20 months are different outcomes.</li>
    <li><b>Give unbiased estimates when follow-up is incomplete</b>, under the assumption that censoring is unrelated to prognosis.</li></ol>
    <p>The figure compares the Kaplan–Meier estimate with two simple alternatives on the same patients.</p>`);
  const fn = figure(ch, 'Three ways to estimate the PFS curve',
    'Kaplan–Meier (step curve) uses censored patients until they leave follow-up. "Censored as events" treats every censored time as a progression. "Drop censored" analyses only patients with an observed progression. Dashed line: 50%.');
  let naive = 'both';
  select(fn.controls, { label: 'Show', value: naive, options: [['km', 'Kaplan–Meier only'], ['ign', 'Kaplan–Meier + censored as events'], ['drop', 'Kaplan–Meier + drop censored'], ['both', 'All three']], onChange: (v) => { naive = v; paintN(); } });
  const sN = stats(fn.root);
  const FN = frame(fn.body, { width: 820, height: 300, x: [0, tMax], y: [0, 1], xLabel: 'months since start of treatment', yLabel: 'progression-free' });
  FN.layer('h').append('line').attr('x1', 0).attr('x2', FN.w).attr('y1', FN.y(0.5)).attr('y2', FN.y(0.5)).attr('stroke', muted()).attr('stroke-dasharray', '4 4');
  const kmIgn = km(IO.t, IO.t.map(() => 1)), dropI = d3.range(nIO).filter((i) => IO.e[i]);
  const kmDrop = km(dropI.map((i) => IO.t[i]), dropI.map(() => 1));
  const NC = [{ k: 'km', c: kmIO, col: ink(), lab: 'Kaplan–Meier' }, { k: 'ign', c: kmIgn, col: cat(2), lab: 'censored as events' }, { k: 'drop', c: kmDrop, col: cat(4), lab: 'drop censored' }];
  legend(fn.body, NC.map((d) => [d.lab, d.col]));
  function paintN() {
    const show = NC.filter((d) => d.k === 'km' || naive === 'both' || naive === d.k);
    FN.layer('c').selectAll('path').data(show).join('path').attr('fill', 'none').attr('stroke', (d) => d.col).attr('stroke-width', (d) => (d.k === 'km' ? 2.6 : 2))
      .attr('d', (d) => stepPath(FN, d.c, tMax));
    const o = {}; show.forEach((d) => { o[`median, ${d.lab}`] = f1(kmMedian(d.c)) + ' mo'; o[`12-month PFS, ${d.lab}`] = pc(kmAt(d.c, 12)); });
    sN.set(o);
  }
  paintN();
  const prog12 = d3.range(nIO).filter((i) => IO.t[i] >= 12 || IO.e[i]), lost12 = nIO - prog12.length;
  prose(ch, `<p>Treating censored times as progressions gives a median of ${f1(kmMedian(kmIgn))} months; dropping censored patients gives ${f1(kmMedian(kmDrop))} months. Both are shorter than the Kaplan–Meier median of <b>${f1(medIO)} months</b>, because both ignore that censored patients were still progression-free when last seen. Collapsing the outcome to "progressed by 12 months" and fitting a logistic regression is also lossy: ${lost12} patients were censored before 12 months and have no 0/1 label, and a progression at 1 month counts the same as one at 11 months.</p>`);
  table(ch, [
    { a: 'Mean or median of observed times (t-test, Wilcoxon)', c: 'no (biased downwards)', t: 'yes', x: 'yes', h: 'difference in mean time' },
    { a: 'Logistic regression on "event by time τ"', c: 'no (patients censored before τ are excluded)', t: 'no', x: 'yes', h: 'odds ratio at one time point' },
    { a: '<b>Kaplan–Meier + log-rank</b>', c: '<b>yes</b>', t: '<b>yes</b>', x: 'groups only', h: '<b>survival curve, median, test</b>' },
    { a: '<b>Cox regression</b> (penalised for many genes)', c: '<b>yes</b>', t: '<b>yes (ranks of times)</b>', x: '<b>yes, continuous and many</b>', h: '<b>hazard ratio per unit, risk score</b>' },
  ], [['a', 'Approach'], ['c', 'Uses censored patients?'], ['t', 'Uses timing?'], ['x', 'Several covariates?'], ['h', 'Output']]);

  // =====================================================================================
  // 3. WHERE
  // =====================================================================================
  ch = chapter(main, 'where', 'Where is it used?', 'Any outcome measured as a time, when follow-up ends before every event has occurred.');
  cards(ch, [
    { tag: 'Clinical trials', title: 'Overall and progression-free survival', body: 'Primary endpoints in most oncology trials: Kaplan–Meier curves, a stratified log-rank test and a Cox hazard ratio with 95% CI.' },
    { tag: 'Biomarkers', title: 'Prognostic gene signatures', body: 'Proliferation-based signatures in breast cancer (e.g. Oncotype DX, MammaPrint) and gene-expression risk scores fitted with penalised Cox regression on TCGA-style cohorts.' },
    { tag: 'Immuno-oncology', title: 'Predictive markers of benefit', body: 'TMB, PD-L1 and T-cell-inflamed (IFN-γ) signatures are assessed by their interaction with treatment arm in randomised trials of checkpoint inhibitors versus chemotherapy.' },
    { tag: 'Genetics', title: 'Age at onset', body: 'Time to disease onset by genotype (e.g. penetrance curves for BRCA1/2 or HTT repeat length), where unaffected carriers are censored at their current age.' },
    { tag: 'Real-world data', title: 'Registry and EHR cohorts', body: 'Time to next treatment or death in observational data, usually with adjustment or weighting for confounders, and careful definition of time zero.' },
    { tag: 'Machine learning', title: 'Survival models beyond Cox', body: 'Random survival forests, gradient-boosted Cox models and neural network extensions (DeepSurv) for non-linear effects; evaluated with the C-index and time-dependent Brier scores.' },
    { tag: 'Microbiology and lab', title: 'Time to growth or death', body: 'Bacterial killing curves, time to colony appearance, and organism lifespan studies (e.g. C. elegans, Drosophila), where escaped or lost animals are censored.' },
    { tag: 'Engineering', title: 'Reliability', body: 'The same methods, under the name reliability analysis, model time to failure of components and instruments.' },
  ]);
  prose(ch, '<p><b>When not to use it:</b> every patient was followed until the event (then ordinary regression on the times is possible, though survival methods still work); the outcome is defined at one fixed time and all patients reach that time (logistic regression); several event types compete and the question concerns one of them (death from other causes prevents progression: use competing-risks methods such as cumulative incidence and Fine–Gray models); or censoring is strongly related to prognosis (the standard methods are then biased).</p>');

  // =====================================================================================
  // 4. HOW
  // =====================================================================================
  ch = chapter(main, 'how', 'How it works', 'Six steps, from a survival curve computed by hand to a gene-expression risk score validated in a new centre and tested against a control arm.');

  // ---- Step 1: Kaplan–Meier by hand
  const ev6 = kmIO.pts.filter((p) => p.d > 0).slice(0, 5);
  let rb = rung(ch, 1, 'The Kaplan–Meier estimate',
    `<p>At each time an event occurs, the fraction of patients <i>still at risk</i> (not yet progressed and not yet censored) who remain event-free is (1 − events / at risk). <i>S</i>(<i>t</i>) is the product of these fractions up to <i>t</i>. A censored patient stays in the denominator until their censoring time and then leaves without counting as an event. The first event times in the anti-PD-1 arm:</p>`);
  let sRun = 1;
  table(rb, ev6.map((p) => { sRun *= 1 - p.d / p.n; return { t: f2(p.t), n: p.n, d: p.d, f: `1 − ${p.d}/${p.n} = ${f3(1 - p.d / p.n)}`, s: f3(sRun) }; }),
    [['t', 'Time (months)'], ['n', 'At risk'], ['d', 'Events'], ['f', 'Fraction event-free'], ['s', '<i>S</i>(<i>t</i>)']], { cls: 'compact' });
  const fk = figure(rb, 'Building the Kaplan–Meier curve',
    'The curve drops only at event times; tick marks are censored patients. The shaded band is a pointwise 95% confidence interval (Greenwood\'s formula), which widens as fewer patients remain at risk. Numbers below the axis: patients still at risk.');
  let kT = tMax, stopK = null;
  const sK = slider(fk.controls, { label: 'show up to month', min: 0, max: tMax, step: 0.5, value: tMax, fmt: (v) => v.toFixed(1), onInput: (v) => { haltK(); kT = v; paintK(); } });
  button(fk.controls, 'Animate', () => { haltK(); kT = 0; stopK = animate((dt) => { kT = Math.min(tMax, kT + dt * 0.012); sK.set(kT); paintK(); return kT < tMax; }); }, true);
  const sKs = stats(fk.root);
  const FK = frame(fk.body, { width: 820, height: 320, x: [0, tMax], y: [0, 1], xLabel: 'months since start of treatment', yLabel: 'progression-free', margin: { bottom: 62 } });
  FK.layer('h').append('line').attr('x1', 0).attr('x2', FK.w).attr('y1', FK.y(0.5)).attr('y2', FK.y(0.5)).attr('stroke', muted()).attr('stroke-dasharray', '4 4');
  FK.g.select('.ax-label').attr('y', FK.h + 54);
  FK.layer('risk').selectAll('text').data(d3.range(0, tMax + 1, 6)).join('text').attr('x', (t) => FK.x(t)).attr('y', FK.h + 36).attr('text-anchor', 'middle').attr('font-size', 10.5).attr('fill', muted())
    .text((t) => IO.t.filter((v) => v >= t).length);
  FK.layer('risk').append('text').attr('x', -16).attr('y', FK.h + 36).attr('text-anchor', 'end').attr('font-size', 10.5).attr('fill', muted()).text('at risk');
  function paintK() {
    const pts = kmIO.pts.filter((p) => p.t <= kT), last = pts[pts.length - 1];
    const band = [...pts.flatMap((p, i) => [[p.t, p.lo, p.hi], [i + 1 < pts.length ? pts[i + 1].t : kT, p.lo, p.hi]])];
    FK.layer('band').selectAll('path').data([band]).join('path').attr('fill', cat(0)).attr('fill-opacity', 0.15)
      .attr('d', d3.area().x((q) => FK.x(q[0])).y0((q) => FK.y(q[1])).y1((q) => FK.y(q[2])));
    FK.layer('c').selectAll('path').data([0]).join('path').attr('fill', 'none').attr('stroke', cat(0)).attr('stroke-width', 2.4).attr('d', stepPath(FK, { pts }, kT));
    FK.layer('cens').selectAll('line').data(kmIO.cens.filter((c) => c.t <= kT)).join('line').attr('x1', (d) => FK.x(d.t)).attr('x2', (d) => FK.x(d.t)).attr('y1', (d) => FK.y(d.s) - 4).attr('y2', (d) => FK.y(d.s) + 4).attr('stroke', ink2()).attr('stroke-width', 1.2);
    sKs.set({ month: kT.toFixed(1), 'S(t)': f3(last.s), '95% CI': `${f2(last.lo)}–${f2(last.hi)}`, 'still at risk': IO.t.filter((v) => v > kT).length, 'events so far': IO.t.filter((v, i) => IO.e[i] && v <= kT).length, 'censored so far': IO.t.filter((v, i) => !IO.e[i] && v <= kT).length });
  }
  function haltK() { if (stopK) { stopK(); stopK = null; } }
  paintK();

  // ---- Step 2: log-rank and cut-points
  rb = rung(ch, 2, 'Comparing groups: the log-rank test and the cut-point problem',
    `<p>To compare two groups, the <b>log-rank test</b> goes through every event time and asks how many of the <i>d</i> events would be expected in group 1 if both groups had the same hazard: <i>E</i>₁ = <i>d</i> × (at risk in group 1 / at risk overall). Summing observed minus expected over all event times, divided by its variance, gives a χ² statistic with 1 degree of freedom. The ratio (<i>O</i>₁/<i>E</i>₁) / (<i>O</i>₀/<i>E</i>₀) approximates the hazard ratio.</p>
     <p>A continuous score must first be split into groups. The lower panel shows how the p-value depends on where the split is placed.</p>`);
  const fl = figure(rb, 'Kaplan–Meier curves split by a score',
    'Top: patients above the chosen quantile of the score form the "high" group. Bottom: log-rank p-value (−log10 scale) for every split from the 20th to the 80th percentile; the dot is the current split.');
  const SC = { prolif: ['Proliferation module score', IO.prolif, CT.prolif], ifng: ['IFN-γ module score', IO.ifng, CT.ifng], risk_cv: ['Cox elastic-net risk score (out-of-fold)', IO.risk_cv, null], noise: ['Random noise (no association)', (() => { const r = rng(5); return IO.t.map(() => r.normal()); })(), null] };
  let sk = 'prolif', q2 = 0.5, arm2 = 'io';
  select(fl.controls, { label: 'Score', value: sk, options: Object.entries(SC).map(([k, v]) => [k, v[0]]), onChange: (v) => { sk = v; paintL2(); } });
  select(fl.controls, { label: 'Arm', value: arm2, options: [['io', `anti-PD-1 (n = ${nIO})`], ['ctrl', `docetaxel control (n = ${CT.t.length})`]], onChange: (v) => { arm2 = v; paintL2(); } });
  slider(fl.controls, { label: 'split at quantile', min: 0.2, max: 0.8, step: 0.05, value: 0.5, fmt: f2, onInput: (v) => { q2 = v; paintL2(); } });
  const sL2 = stats(fl.root);
  const FL = frame(fl.body, { width: 820, height: 290, x: [0, tMax], y: [0, 1], xLabel: 'months since start of treatment', yLabel: 'progression-free' });
  const FP = frame(fl.body, { width: 820, height: 170, x: [0.2, 0.8], y: [0, 6], xLabel: 'split quantile', yLabel: '−log10 p' });
  FP.layer('a').append('line').attr('x1', 0).attr('x2', FP.w).attr('y1', FP.y(-Math.log10(0.05))).attr('y2', FP.y(-Math.log10(0.05))).attr('stroke', muted()).attr('stroke-dasharray', '4 4');
  FP.layer('a').append('text').attr('x', FP.w - 4).attr('y', FP.y(-Math.log10(0.05)) - 4).attr('text-anchor', 'end').attr('font-size', 11).attr('fill', muted()).text('p = 0.05');
  legend(fl.body, [['low score', cat(0)], ['high score', cat(1)]]);
  const QS = d3.range(0.2, 0.801, 0.05);
  function armData(a, key) { const s = a === 'io' ? SC[key][1] : SC[key][2]; const A = a === 'io' ? IO : CT; return s ? { t: A.t, e: A.e, s } : null; }
  function paintL2() {
    const dd = armData(arm2, sk);
    if (!dd) { sL2.set({ note: 'this score exists only for the anti-PD-1 arm; showing anti-PD-1' }); arm2 = 'io'; fl.controls.querySelectorAll('select')[1].value = 'io'; return paintL2(); }
    const g = splitAt(dd.s, q2), lr = logrank(dd.t, dd.e, g);
    const curves = [0, 1].map((k) => { const ix = d3.range(g.length).filter((i) => g[i] === k); return km(ix.map((i) => dd.t[i]), ix.map((i) => dd.e[i])); });
    drawKM(FL, curves, tMax);
    const scan = QS.map((qq) => ({ q: qq, p: logrank(dd.t, dd.e, splitAt(dd.s, qq)).p }));
    FP.y.domain([0, Math.max(3, d3.max(scan, (d) => -Math.log10(d.p)) * 1.1)]);
    FP.g.select('.axis.y').call(d3.axisLeft(FP.y).ticks(4));
    FP.layer('a').selectAll('line').attr('y1', FP.y(-Math.log10(0.05))).attr('y2', FP.y(-Math.log10(0.05)));
    FP.layer('a').select('text').attr('y', FP.y(-Math.log10(0.05)) - 4);
    FP.layer('l').selectAll('path').data([scan]).join('path').attr('fill', 'none').attr('stroke', ink2()).attr('stroke-width', 2).attr('d', d3.line().x((d) => FP.x(d.q)).y((d) => FP.y(-Math.log10(d.p))));
    FP.layer('d').selectAll('circle').data([lr]).join('circle').attr('cx', FP.x(q2)).attr('cy', FP.y(-Math.log10(lr.p))).attr('r', 6).attr('fill', cat(1)).attr('stroke', css('--surface')).attr('stroke-width', 1.5);
    const minP = d3.min(scan, (d) => d.p);
    sL2.set({ 'median PFS low / high': curves.map((c) => f1(kmMedian(c))).join(' / ') + ' mo', 'log-rank χ²': f1(lr.chi), p: pE(lr.p), 'HR high vs low (O/E)': f2(lr.hr), 'smallest p over all splits': pE(minP) });
  }
  paintL2();
  // pitfall numbers: how often does a pure-noise score reach p < 0.05 at its best split?
  const rN = rng(2024), NSIM = 200;
  let hitBest = 0, hitMed = 0;
  for (let k = 0; k < NSIM; k++) {
    const s = IO.t.map(() => rN.normal());
    const ps = QS.map((qq) => logrank(IO.t, IO.e, splitAt(s, qq)).p);
    if (d3.min(ps) < 0.05) hitBest++;
    if (logrank(IO.t, IO.e, splitAt(s, 0.5)).p < 0.05) hitMed++;
  }
  prose(rb, `<p>Across ${NSIM} simulated random-noise scores, a median split reached p &lt; 0.05 in ${pc(hitMed / NSIM)} of cases (close to the nominal 5%), but the best of the ${QS.length} candidate splits did so in <b>${pc(hitBest / NSIM)}</b>. Choosing the cut-point that minimises the p-value inflates false positives; pre-specify the cut-point or keep the score continuous (step 3).</p>`);

  // ---- Step 3: Cox
  const beta0 = coxPr.beta[0];
  rb = rung(ch, 3, 'The Cox proportional-hazards model',
    `<p>The Cox model relates the hazard to covariates without specifying how the baseline hazard changes over time:</p>
     <div class="calc">h(t | x) = h₀(t) × exp(β₁x₁ + β₂x₂ + …)        HR per unit of x₁ = exp(β₁)</div>
     <p>Each covariate multiplies the hazard by a constant factor at every time point: the <b>proportional-hazards assumption</b>. β is estimated by maximising the <b>partial likelihood</b>: at each event time, the probability that the patient who progressed is the one who did, out of everyone still at risk, is exp(βx<sub>i</sub>) / Σ exp(βx<sub>j</sub>). The unknown <i>h</i>₀(<i>t</i>) cancels, and only the ordering of event times matters. In the figure, the proliferation score is standardised (per SD).</p>`);
  const fc = figure(rb, 'Fitting a hazard ratio for the proliferation score',
    'Left: log partial likelihood as a function of β (the maximum is the Cox estimate). Right: observed Kaplan–Meier curves for thirds of the proliferation score (solid) and the curves implied by the model at the mean score of each third (dashed, with a Breslow baseline). Move β to see how the fitted curves spread apart.');
  let bC = 0;
  const sBC = slider(fc.controls, { label: 'β (log hazard ratio per SD)', min: -0.6, max: 1.2, step: 0.01, value: 0, fmt: f2, onInput: (v) => { bC = v; paintC(); } });
  button(fc.controls, 'Fit (maximum partial likelihood)', () => { bC = beta0; sBC.set(beta0); paintC(); }, true);
  const sC = stats(fc.root);
  const rowC = el('div', { class: 'fig-row' }); fc.body.append(rowC);
  const ca = el('div'), cb = el('div'); rowC.append(ca, cb);
  const bGrid = d3.range(-0.6, 1.2001, 0.02), plG = bGrid.map((b) => ({ b, l: coxLL([b], zPr.map((v) => [v]), IO.t, IO.e) }));
  const FC1 = frame(ca, { width: 460, height: 300, x: [-0.6, 1.2], y: d3.extent(plG, (d) => d.l), xLabel: 'β per SD of proliferation', yLabel: 'log partial likelihood', margin: { left: 60 } });
  FC1.g.select('.ax-label[transform]').attr('y', -48);
  FC1.layer('c').append('path').datum(plG).attr('fill', 'none').attr('stroke', cat(6)).attr('stroke-width', 2.2).attr('d', d3.line().x((d) => FC1.x(d.b)).y((d) => FC1.y(d.l)));
  FC1.layer('m').append('line').attr('x1', FC1.x(beta0)).attr('x2', FC1.x(beta0)).attr('y1', 0).attr('y2', FC1.h).attr('stroke', muted()).attr('stroke-dasharray', '4 4');
  const FC2 = frame(cb, { width: 460, height: 300, x: [0, tMax], y: [0, 1], xLabel: 'months', yLabel: 'progression-free' });
  const terc = d3.range(nIO).sort((a, b) => zPr[a] - zPr[b]), thirds = [0, 1, 2].map((k) => terc.slice(Math.round(k * nIO / 3), Math.round((k + 1) * nIO / 3)));
  const kmT = thirds.map((ix) => km(ix.map((i) => IO.t[i]), ix.map((i) => IO.e[i]))), mT = thirds.map((ix) => mean(ix.map((i) => zPr[i])));
  FC2.layer('km').selectAll('path').data(kmT).join('path').attr('fill', 'none').attr('stroke', (d, i) => cat([0, 5, 1][i])).attr('stroke-width', 1.8).attr('stroke-opacity', 0.75).attr('d', (c) => stepPath(FC2, c, tMax));
  legend(fc.body, [['lowest third', cat(0)], ['middle third', cat(5)], ['highest third', cat(1)], ['partial likelihood', cat(6)]]);
  function paintC() {
    const H0 = breslow([bC], zPr.map((v) => [v]), IO.t, IO.e);
    FC2.layer('fit').selectAll('path').data(mT).join('path').attr('fill', 'none').attr('stroke', (d, i) => cat([0, 5, 1][i])).attr('stroke-width', 2.4).attr('stroke-dasharray', '6 4')
      .attr('d', (m) => d3.line().curve(d3.curveStepAfter).x((p) => FC2.x(p[0])).y((p) => FC2.y(Math.exp(-p[1] * Math.exp(bC * m))))([[0, 0], ...H0, [tMax, H0[H0.length - 1][1]]]));
    const l = coxLL([bC], zPr.map((v) => [v]), IO.t, IO.e);
    FC1.layer('p').selectAll('circle').data([0]).join('circle').attr('cx', FC1.x(bC)).attr('cy', FC1.y(l)).attr('r', 6).attr('fill', cat(6)).attr('stroke', css('--surface')).attr('stroke-width', 1.5);
    sC.set({ β: f2(bC), 'HR per SD = exp(β)': f2(Math.exp(bC)), 'log partial likelihood': f2(l), 'maximum at β': `${f2(beta0)} (HR ${f2(Math.exp(beta0))})` });
  }
  paintC();
  const ciHR = (b, se) => `${f2(Math.exp(b))} (${f2(Math.exp(b - 1.96 * se))}–${f2(Math.exp(b + 1.96 * se))})`;
  table(rb, [
    { m: 'IFN-γ only', a: ciHR(coxIf.beta[0], coxIf.se[0]), p1: pE(pNormal(coxIf.beta[0] / coxIf.se[0])), b: '—', p2: '—' },
    { m: 'Proliferation only', a: '—', p1: '—', b: ciHR(coxPr.beta[0], coxPr.se[0]), p2: pE(pNormal(coxPr.beta[0] / coxPr.se[0])) },
    { m: 'Both together', a: ciHR(coxBoth.beta[0], coxBoth.se[0]), p1: pE(pNormal(coxBoth.beta[0] / coxBoth.se[0])), b: ciHR(coxBoth.beta[1], coxBoth.se[1]), p2: pE(pNormal(coxBoth.beta[1] / coxBoth.se[1])) },
  ], [['m', 'Cox model (anti-PD-1 arm)'], ['a', 'HR per SD of IFN-γ (95% CI)'], ['p1', 'p'], ['b', 'HR per SD of proliferation (95% CI)'], ['p2', 'p']], { cls: 'compact' });
  prose(rb, `<p>Higher IFN-γ lowers the hazard (HR &lt; 1, later progression) and higher proliferation raises it. As in logistic regression, each HR is adjusted for the other covariates in the model; the coefficient is a log hazard ratio, and the standard error comes from the curvature of the partial likelihood at its maximum.</p>`);

  // ---- Step 4: C-index
  const pairs = comparablePairs(IO.t, IO.e);
  rb = rung(ch, 4, 'Judging a risk score: the concordance index',
    `<p>A risk score is judged by <b>discrimination</b>: do patients with higher scores progress sooner? A pair of patients is <i>comparable</i> if the one with the shorter time had an observed event (otherwise the order is unknown). Harrell's <b>C-index</b> is the fraction of comparable pairs in which the patient who progressed first had the higher score; ties count one half. It is the survival analogue of the AUC: 0.5 is random ordering, 1 is perfect.</p>`);
  worked(rb, 'C-index of the module scores in the anti-PD-1 arm',
    `<p>Of the ${nIO} × ${nIO - 1} / 2 = ${(nIO * (nIO - 1) / 2).toLocaleString('en-GB')} pairs, <b>${pairs.toLocaleString('en-GB')}</b> are comparable. Using the proliferation score as the risk score gives C = <b>${f3(cPr)}</b>. Using the IFN-γ score with its sign reversed (high IFN-γ = low risk) gives C = <b>${f3(cIf)}</b>. Values in the range 0.6–0.7 are typical for single biomarkers; the C-index depends on the follow-up and case mix, so it is only comparable within the same cohort.</p>`);

  // ---- Step 5: Cox elastic-net lab
  rb = rung(ch, 5, 'Risk-score lab: a Cox elastic net on 1,933 genes',
    `<p>With ${nEv} events and 1,933 genes, an unpenalised Cox model cannot be fitted. The <b>Cox elastic net</b> maximises the partial likelihood minus a penalty λ × [α Σ|β| + (1 − α)/2 Σβ²] (α = ${L.alpha}: half lasso, which sets most coefficients to exactly zero, half ridge, which keeps correlated genes together). Each λ gives a model; cross-validated partial likelihood selected the model with ${S.path.nnz[jSel]} genes. Every model is applied unchanged to Centre C (genes standardised within the centre to remove its technical shift) and to the docetaxel control arm.</p>`);
  const fx = figure(rb, 'Risk-score lab',
    'Top: C-index along the penalty path, in the training data (in-sample), out-of-fold in 5-fold cross-validation (λ fixed in advance, so not nested), in Centre C, and in the docetaxel arm. Bottom left: genes kept at the current λ, coloured by their planted module. Bottom right: Centre C patients split at the discovery median of the locked score.');
  let jL = jSel;
  const sJ = slider(fx.controls, { label: 'penalty λ (strong → weak)', min: 0, max: L.steps.length - 1, step: 1, value: jSel, fmt: (j) => `${L.steps[j].g.length} genes`, onInput: (j) => { jL = j; paintX(); } });
  button(fx.controls, 'CV choice', () => { jL = jSel; sJ.set(jSel); paintX(); });
  const sX = stats(fx.root);
  const CL = [['c_in', 'in-sample', cat(0)], ['c_cv', 'out-of-fold (CV)', cat(1)], ['c_val', 'Centre C', cat(2)], ['c_ctrl', 'docetaxel arm', cat(4)]];
  const FX0 = frame(fx.body, { width: 820, height: 230, x: [0, L.steps.length - 1], y: [0.45, 1], xLabel: 'genes in the model (penalty decreasing →)', yLabel: 'C-index', axes: false });
  FX0.g.append('g').attr('class', 'axis x').attr('transform', `translate(0,${FX0.h})`).call(d3.axisBottom(FX0.x).tickValues(d3.range(0, L.steps.length, 2)).tickFormat((j) => L.steps[j].g.length));
  FX0.g.append('g').attr('class', 'axis y').call(d3.axisLeft(FX0.y).ticks(5));
  FX0.layer('h').append('line').attr('x1', 0).attr('x2', FX0.w).attr('y1', FX0.y(0.5)).attr('y2', FX0.y(0.5)).attr('stroke', muted()).attr('stroke-dasharray', '4 4');
  CL.forEach(([k, , c]) => {
    FX0.layer('l').append('path').datum(L.steps).attr('fill', 'none').attr('stroke', c).attr('stroke-width', 2).attr('d', d3.line().x((d, j) => FX0.x(j)).y((d) => FX0.y(d[k])));
    FX0.layer('p').selectAll(null).data(L.steps).join('circle').attr('cx', (d, j) => FX0.x(j)).attr('cy', (d) => FX0.y(d[k])).attr('r', 3.2).attr('fill', c)
      .on('mousemove', (e, d) => tip(e, `${d.g.length} genes<br>C-index in-sample ${f3(d.c_in)} · CV ${f3(d.c_cv)}<br>Centre C ${f3(d.c_val)} · docetaxel ${f3(d.c_ctrl)}`)).on('mouseleave', untip);
  });
  FX0.layer('sel').append('text').attr('x', FX0.x(jSel)).attr('y', -2).attr('text-anchor', 'middle').attr('font-size', 10.5).attr('fill', muted()).text('CV choice');
  legend(fx.body, CL.map(([, l, c]) => [l, c]));
  const rowX = el('div', { class: 'fig-row' }); fx.body.append(rowX);
  const xa = el('div'), xb = el('div'); rowX.append(xa, xb);
  const coefSvg = d3.select(xa).insert('svg', ':first-child').attr('width', 460).attr('role', 'img');
  const FXK = frame(xb, { width: 460, height: 300, x: [0, tMax], y: [0, 1], xLabel: 'months (Centre C)', yLabel: 'progression-free' });
  const GRPS = [...new Set(L.group)];
  legend(xa, GRPS.map((g) => [groupLabel(g), groupColor(g)]));
  legend(xb, [['Centre C: low risk', cat(0)], ['Centre C: high risk', cat(1)]]);
  function paintX() {
    const st = L.steps[jL];
    FX0.layer('cur').selectAll('line').data([jL]).join('line').attr('x1', FX0.x(jL)).attr('x2', FX0.x(jL)).attr('y1', 0).attr('y2', FX0.h).attr('stroke', ink()).attr('stroke-width', 1.5);
    // coefficient bars (top 14 by |coef|)
    coefSvg.selectAll('*').remove();
    const rows = st.g.map((g, k) => ({ g: L.genes[g], grp: L.group[g], c: st.c[k] })).sort((a, b) => Math.abs(b.c) - Math.abs(a.c));
    const show = rows.slice(0, 14), m = { l: 70, r: 16, t: 8, b: 44 }, W = 460 - m.l - m.r, Hh = Math.max(4, show.length) * 18;
    coefSvg.attr('viewBox', `0 0 460 ${Hh + m.t + m.b}`);
    const g = coefSvg.append('g').attr('transform', `translate(${m.l},${m.t})`);
    const ext = Math.max(0.05, d3.max(show, (r) => Math.abs(r.c)) || 0.05) * 1.1, xS = d3.scaleLinear().domain([-ext, ext]).range([0, W]);
    g.append('g').attr('class', 'axis x').attr('transform', `translate(0,${Hh})`).call(d3.axisBottom(xS).ticks(5));
    g.append('text').attr('class', 'ax-label').attr('x', W / 2).attr('y', Hh + 36).attr('text-anchor', 'middle').text('log HR per SD (+ = earlier progression)');
    g.append('line').attr('x1', xS(0)).attr('x2', xS(0)).attr('y1', 0).attr('y2', Hh).attr('stroke', muted());
    if (!show.length) g.append('text').attr('x', W / 2).attr('y', Hh / 2).attr('text-anchor', 'middle').attr('font-size', 12).attr('fill', muted()).text('no genes: the penalty sets every coefficient to 0');
    show.forEach((r, i) => {
      g.append('rect').attr('x', xS(Math.min(0, r.c))).attr('width', Math.abs(xS(r.c) - xS(0))).attr('y', i * 18 + 3).attr('height', 12).attr('rx', 2).attr('fill', groupColor(r.grp))
        .on('mousemove', (e) => tip(e, `<b>${r.g}</b> (${groupLabel(r.grp)})<br>coefficient ${f3(r.c)} · HR per SD ${f2(Math.exp(r.c))}`)).on('mouseleave', untip);
      g.append('text').attr('x', -6).attr('y', i * 18 + 13).attr('text-anchor', 'end').attr('font-size', 11).attr('fill', ink2()).text(r.g);
    });
    if (rows.length > show.length) g.append('text').attr('x', W).attr('y', Hh - 4).attr('text-anchor', 'end').attr('font-size', 11).attr('fill', muted()).text(`+ ${rows.length - show.length} more genes`);
    // Centre C KM
    const grp = st.rv.map((v) => +(v > st.thr));
    const curves = [0, 1].map((k) => { const ix = d3.range(grp.length).filter((i) => grp[i] === k); return ix.length ? km(ix.map((i) => VA.t[i]), ix.map((i) => VA.e[i])) : null; });
    drawKM(FXK, curves, tMax);
    const lr = st.g.length && sum(grp) > 0 && sum(grp) < grp.length ? logrank(VA.t, VA.e, grp) : null;
    sX.set({ genes: st.g.length, 'C-index in-sample': f3(st.c_in), 'out-of-fold': f3(st.c_cv), 'Centre C': f3(st.c_val), 'docetaxel arm': f3(st.c_ctrl), 'Centre C HR high vs low': lr ? `${f2(lr.hr)} (p ${pE(lr.p)})` : '—' });
  }
  paintX();
  const big = L.steps.length - 1, best = d3.maxIndex(L.steps, (d) => d.c_val);
  const nGrp = (st, grp) => st.g.filter((g) => L.group[g] === grp).length;
  table(rb, [
    { t: '<b>CV choice</b>', s: `${stSel.g.length} genes (${nGrp(stSel, 'PROLIF module')} proliferation, ${nGrp(stSel, 'IFNG module')} IFN-γ module, ${nGrp(stSel, 'filler')} filler genes with no planted effect); C-index in-sample ${f3(stSel.c_in)}, out-of-fold ${f3(stSel.c_cv)}, Centre C <b>${f3(stSel.c_val)}</b>`, m: 'Proliferation genes enter with positive coefficients and immune genes with negative ones, matching the planted biology; the filler genes have no planted effect on outcome. They enter by chance or because some filler genes are raised in one cell type and so track tumour composition, which is linked to response; some carry sizeable coefficients. The score transfers to a new centre.' },
    { t: 'Slide to the <b>weakest penalty</b>', s: `${L.steps[big].g.length} genes; in-sample ${f3(L.steps[big].c_in)}, out-of-fold <b>${f3(L.steps[big].c_cv)}</b>, Centre C ${f3(L.steps[big].c_val)}`, m: '<b>Overfitting.</b> The in-sample C-index approaches 1 while performance on unseen patients falls. Only out-of-fold or external estimates are informative.' },
    { t: 'Find the best <b>Centre C</b> value', s: `Highest at ${L.steps[best].g.length} genes (${f3(L.steps[best].c_val)}), close to the CV choice`, m: 'Cross-validation inside the discovery cohort selected a penalty near the one that generalises best. Choosing λ by looking at Centre C would make it no longer an independent test.' },
    { t: 'Compare <b>Centre C</b> with the <b>docetaxel arm</b>', s: `At the CV choice: ${f3(stSel.c_val)} vs ${f3(stSel.c_ctrl)}`, m: 'The score still ranks patients in the arm that received no immunotherapy. A large part of it is <b>prognostic</b> (step 6).' },
  ], [['t', 'Try'], ['s', 'What you see'], ['m', 'What it means']], { cls: 'compact' });

  // ---- Step 6: prognostic vs predictive
  rb = rung(ch, 6, 'Prognostic or predictive? The control arm',
    `<p>A <b>prognostic</b> marker is associated with outcome whatever the treatment. A <b>predictive</b> marker identifies patients who benefit more from a specific treatment than from the alternative. In a single treated cohort the two look the same. A randomised comparison is needed: fit h = h₀ × exp(β₁·arm + β₂·marker + β₃·arm × marker) on both arms together. β₃ is the <b>interaction</b>; a predictive marker has β₃ ≠ 0.</p>`);
  const fp = figure(rb, 'The same marker in both arms',
    'Each panel splits one arm at the median of the marker (pooled across arms). Hazard ratios are per SD of the marker standardised across both arms, so they differ slightly from the step 3 table. A prognostic marker separates the curves similarly in both arms; a predictive marker separates them mainly in the anti-PD-1 arm.');
  let mk = 'ifng';
  select(fp.controls, { label: 'Marker', value: mk, options: [['ifng', 'IFN-γ module score'], ['prolif', 'Proliferation module score']], onChange: (v) => { mk = v; paintP(); } });
  const sP = stats(fp.root);
  const rowP = el('div', { class: 'fig-row' }); fp.body.append(rowP);
  const pa = el('div'), pb = el('div'); rowP.append(pa, pb);
  const ttl = (d, s) => d.append(el('div', { class: 'fig-caption', style: 'margin:0 0 2px 50px;font-weight:600' }, s));
  ttl(pa, `anti-PD-1 (n = ${nIO})`); ttl(pb, `docetaxel (n = ${CT.t.length})`);
  const FPa = frame(pa, { width: 460, height: 280, x: [0, tMax], y: [0, 1], xLabel: 'months', yLabel: 'progression-free' });
  const FPb = frame(pb, { width: 460, height: 280, x: [0, tMax], y: [0, 1], xLabel: 'months', yLabel: 'progression-free' });
  legend(fp.body, [['below pooled median', cat(0)], ['above pooled median', cat(1)]]);
  const arm = [...IO.t.map(() => 1), ...CT.t.map(() => 0)], tAll = [...IO.t, ...CT.t], eAll = [...IO.e, ...CT.e];
  const inter = {};
  ['ifng', 'prolif'].forEach((k) => {
    const z = zs([...IO[k], ...CT[k]]);
    const f = coxFit(z.map((v, i) => [arm[i], v, arm[i] * v]), tAll, eAll);
    const fi = coxFit(z.slice(0, nIO).map((v) => [v]), IO.t, IO.e), fc2 = coxFit(z.slice(nIO).map((v) => [v]), CT.t, CT.e);
    inter[k] = { z, f, hrIO: Math.exp(fi.beta[0]), hrC: Math.exp(fc2.beta[0]), pIO: pNormal(fi.beta[0] / fi.se[0]), pC: pNormal(fc2.beta[0] / fc2.se[0]), pInt: pNormal(f.beta[2] / f.se[2]) };
  });
  function paintP() {
    const I = inter[mk], med = d3.median(I.z);
    [[FPa, 0, nIO, IO], [FPb, nIO, nIO + CT.t.length, CT]].forEach(([F, a, b, A]) => {
      const g = I.z.slice(a, b).map((v) => +(v > med));
      drawKM(F, [0, 1].map((k) => { const ix = d3.range(g.length).filter((i) => g[i] === k); return km(ix.map((i) => A.t[i]), ix.map((i) => A.e[i])); }), tMax);
    });
    sP.set({ 'HR per SD, anti-PD-1': `${f2(I.hrIO)} (p ${pE(I.pIO)})`, 'HR per SD, docetaxel': `${f2(I.hrC)} (p ${pE(I.pC)})`, 'interaction p (arm × marker)': pE(I.pInt), verdict: I.pInt < 0.05 ? 'predictive' : I.pC < 0.05 ? 'prognostic' : 'neither' });
  }
  paintP();
  const Ii = inter.ifng, Ip = inter.prolif;
  prose(rb, `<p><b>IFN-γ</b>: HR per SD ${f2(Ii.hrIO)} on anti-PD-1 and ${f2(Ii.hrC)} on docetaxel, interaction p = ${pE(Ii.pInt)}: <b>predictive</b>, as planted (IFN-γ drives response to anti-PD-1 only). <b>Proliferation</b>: ${f2(Ip.hrIO)} and ${f2(Ip.hrC)}, interaction p = ${pE(Ip.pInt)}: <b>prognostic</b>, with a similar effect under either drug. The step-5 risk score is dominated by proliferation genes, which is why it still has C-index ${f3(stSel.c_ctrl)} in the docetaxel arm. Selecting "high-risk" patients by this score would enrich for poor-prognosis tumours, not for patients who benefit from anti-PD-1.</p>
    <p class="fig-caption">A result that differs from the planted rule: the simulation gives proliferation no effect on response, yet its bulk score has AUC ${f3(S.inter.prolif.auc_resp)} for response (lower in responders). Proliferation genes are expressed by tumour cells, so the bulk score also tracks tumour purity, and inflamed tumours that respond contain more immune cells. The interaction test, not the association with response, settles whether a marker is predictive.</p>`);
  quiz(rb, 'In the anti-PD-1 arm alone, both scores are strongly associated with PFS. Which single analysis distinguishes the prognostic marker from the predictive one?',
    `<p>A Cox model on both randomised arms with an arm × marker interaction term. The marker effect within the treated arm alone cannot separate the two: here IFN-γ has p ${pE(Ii.pIO)} and proliferation p ${pE(Ip.pIO)} on anti-PD-1, yet only IFN-γ has a significant interaction (p ${pE(Ii.pInt)} vs ${pE(Ip.pInt)}).</p>`);

  // =====================================================================================
  // 5. PITFALLS
  // =====================================================================================
  ch = chapter(main, 'pitfalls', 'Pitfalls', 'Common problems, how to recognise them, and how to address them.');
  const respHR = logrank(IO.t, IO.e, IO.resp.map((r) => 1 - r));
  cards(ch, [
    { tag: 'Symptom: "the marker predicts benefit" from one treated cohort', title: 'Prognostic mistaken for predictive', body: `In the anti-PD-1 arm, proliferation has HR ${f2(Ip.hrIO)} per SD, and a similar effect under docetaxel (${f2(Ip.hrC)}). <b>Fix:</b> claim predictive value only from a randomised comparison with a pre-specified treatment × marker interaction test.` },
    { tag: 'Symptom: a striking p-value at an unusual cut-off', title: 'Optimised cut-points', body: `Scanning cut-offs for the smallest log-rank p gave p &lt; 0.05 for ${pc(hitBest / NSIM)} of random-noise scores (step 2). <b>Fix:</b> pre-specify the cut-off (e.g. median, or one from a previous study), use the continuous score in a Cox model, or correct for the scan.` },
    { tag: 'Symptom: near-perfect separation of training curves', title: 'Resubstitution', body: `A risk score plotted on the patients it was fitted to looks better than it is: C-index ${f3(stSel.c_in)} in-sample vs ${f3(stSel.c_cv)} out-of-fold at the CV choice, and ${f3(L.steps[big].c_in)} vs ${f3(L.steps[big].c_cv)} with the weakest penalty. <b>Fix:</b> nested cross-validation for internal estimates, and a locked score tested in an independent cohort.` },
    { tag: 'Symptom: crossing curves or a long plateau', title: 'Non-proportional hazards', body: 'Immunotherapy curves often cross chemotherapy curves or level off (a durable-response tail), so a single HR averages an effect that changes over time. <b>Fix:</b> plot the curves, test Schoenfeld residuals, report restricted mean survival time or time-varying effects.' },
    { tag: 'Symptom: responders live much longer', title: 'Immortal-time (guarantee-time) bias', body: `Comparing survival by response status is biased in real data: a patient must survive long enough to be assessed as a responder. In this simulation response is fixed at time zero, so the comparison (HR ${f2(respHR.hr)} non-responders vs responders) is not biased here. <b>Fix:</b> landmark analysis from the assessment time, or response as a time-dependent covariate.` },
    { tag: 'Symptom: curve stays high where few patients remain', title: 'Informative censoring and sparse tails', body: 'If patients leave follow-up because they are deteriorating, censoring is related to prognosis and Kaplan–Meier is biased upwards. The tail of the curve also rests on few patients. <b>Fix:</b> check reasons for censoring, show numbers at risk and confidence bands, and truncate the plot where few remain.' },
    { tag: 'Symptom: score fails in a new hospital', title: 'Unscaled transfer across centres', body: 'Gene-level technical shifts between centres change the absolute score, so a fixed cut-off can assign almost all new patients to one group. <b>Fix:</b> standardise within cohort (as done for Centre C here), use bridging samples, or use rank-based scores; recalibrate before absolute risks are reported.' },
    { tag: 'Symptom: unstable hazard ratios', title: 'Too few events', body: `Information in survival data comes from events, not patients: ${nEv} events here. A common rule of thumb is at least 10–20 events per covariate in an unpenalised Cox model. <b>Fix:</b> pre-specify a few covariates, or penalise and validate.` },
  ]);

  // =====================================================================================
  // 6. RECAP
  // =====================================================================================
  ch = chapter(main, 'recap', 'Recap', 'Key points.');
  recap(ch, [
    'Time-to-event data are pairs (time, event). <b>Censored</b> patients contribute "event-free at least this long" and must not be dropped or treated as events.',
    '<b>Kaplan–Meier</b> multiplies the fraction event-free at each event time among patients still at risk; report numbers at risk and confidence bands.',
    'The <b>log-rank test</b> compares observed and expected events between groups. Pre-specify cut-offs: scanning them inflates false positives.',
    'The <b>Cox model</b> gives a hazard ratio per unit of each covariate, assuming the ratio is constant over time. exp(β) is the HR, adjusted for the other covariates.',
    'Many genes require a <b>penalised Cox model</b> (lasso / elastic net) with λ chosen by cross-validated partial likelihood.',
    'Judge risk scores by the <b>C-index</b> on patients the model has not seen: out-of-fold and in an independent cohort, not in-sample.',
    'A marker associated with survival in one treated arm may be <b>prognostic</b>. Only a treatment × marker <b>interaction</b> in a randomised comparison shows it is <b>predictive</b>.',
  ]);
  prose(ch, '<h3>In code</h3><p>Kaplan–Meier, log-rank and Cox models come from <code>lifelines</code> (Python) or <code>survival</code> (R); penalised Cox models from <code>scikit-survival</code> or <code>glmnet</code>.</p>');
  code(ch, {
    Python: `from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored

# Kaplan-Meier and log-rank (df: one row per patient with pfs_months, event, prolif, ifng, arm)
hi = df["prolif"] > df["prolif"].median()
km = KaplanMeierFitter().fit(df.pfs_months[hi], df.event[hi], label="high")
print(km.median_survival_time_)
print(logrank_test(df.pfs_months[hi], df.pfs_months[~hi], df.event[hi], df.event[~hi]).p_value)

# Cox model with a treatment x marker interaction (both randomised arms)
df["arm_x_ifng"] = df["arm"] * df["ifng"]
cph = CoxPHFitter().fit(df[["pfs_months", "event", "arm", "ifng", "arm_x_ifng"]], "pfs_months", "event")
cph.print_summary()                    # exp(coef) = hazard ratios; check_assumptions() for PH

# Elastic-net Cox risk score on genes (choose alpha by CV in a real analysis)
y = Surv.from_arrays(event=train.event.astype(bool), time=train.pfs_months)
net = CoxnetSurvivalAnalysis(l1_ratio=0.5, alphas=[0.05]).fit(Xtrain_scaled, y)
risk = net.predict(Xtest_scaled)
print(concordance_index_censored(test.event.astype(bool), test.pfs_months, risk)[0])`,
    R: `library(survival); library(glmnet)

# Kaplan-Meier and log-rank
fit <- survfit(Surv(pfs_months, event) ~ prolif > median(prolif), data = df)
print(fit); plot(fit)                  # survminer::ggsurvplot(fit, risk.table = TRUE)
survdiff(Surv(pfs_months, event) ~ prolif > median(prolif), data = df)

# Cox model with a treatment x marker interaction (both randomised arms)
m <- coxph(Surv(pfs_months, event) ~ arm * ifng, data = df)
summary(m)                             # exp(coef) = hazard ratios
cox.zph(m)                             # proportional-hazards check

# Elastic-net Cox risk score, lambda by cross-validated partial likelihood
cv <- cv.glmnet(x_train, Surv(train$pfs_months, train$event), family = "cox", alpha = 0.5)
risk <- predict(cv, newx = x_test, s = "lambda.min")
concordance(Surv(test$pfs_months, test$event) ~ risk, reverse = TRUE)`,
  });
  seeAlso(main, [['omics', 'bulk-classifier', 'Previous: biomarker classifier and nested CV'], ['omics', 'sc-clustering', 'Next: single-cell clustering and annotation'], ['concepts', 'logistic-regression', 'Logistic regression'], ['concepts', 'regularization', 'Regularisation (ridge and lasso)'], ['concepts', 'cross-validation', 'Cross-validation and data leakage'], ['concepts', 'multiple-testing', 'Multiple testing and FDR']]);
}

// ---------------------------------------------------------------------------------------------
// Survival maths (small, exact; n ≤ 340)
/** Kaplan–Meier: {pts:[{t,s,n,d,lo,hi}], cens:[{t,s}]}; Greenwood 95% band. */
function km(t, e) {
  const o = d3.range(t.length).sort((a, b) => t[a] - t[b]);
  let atRisk = t.length, s = 1, gw = 0, k = 0;
  const pts = [{ t: 0, s: 1, n: atRisk, d: 0, lo: 1, hi: 1 }], cens = [];
  while (k < o.length) {
    const tt = t[o[k]]; let d = 0, c = 0;
    while (k < o.length && t[o[k]] === tt) { if (e[o[k]]) d++; else c++; k++; }
    if (d) {
      s *= 1 - d / atRisk; if (atRisk > d) gw += d / (atRisk * (atRisk - d));
      const h = 1.96 * s * Math.sqrt(gw);
      pts.push({ t: tt, s, n: atRisk, d, lo: Math.max(0, s - h), hi: Math.min(1, s + h) });
    }
    if (c) cens.push({ t: tt, s });
    atRisk -= d + c;
  }
  return { pts, cens };
}
function kmMedian(c) { const p = c.pts.find((q) => q.s <= 0.5); return p ? p.t : null; }
function kmAt(c, t) { let s = 1; c.pts.forEach((p) => { if (p.t <= t) s = p.s; }); return s; }
function stepPath(F, c, tEnd) {
  const p = c.pts;
  return d3.line().curve(d3.curveStepAfter).x((d) => F.x(d.t)).y((d) => F.y(d.s))([...p, { t: Math.max(tEnd, p[p.length - 1].t), s: p[p.length - 1].s }]);
}
function drawKM(F, curves, tEnd) {
  const cs = curves.map((c, i) => (c ? { c, i } : null)).filter(Boolean);
  F.layer('c').selectAll('path').data(cs).join('path').attr('fill', 'none').attr('stroke', (d) => cat(d.i)).attr('stroke-width', 2.2).attr('d', (d) => stepPath(F, d.c, tEnd));
  F.layer('cens').selectAll('line').data(cs.flatMap((d) => d.c.cens.map((q) => ({ ...q, i: d.i })))).join('line')
    .attr('x1', (d) => F.x(d.t)).attr('x2', (d) => F.x(d.t)).attr('y1', (d) => F.y(d.s) - 4).attr('y2', (d) => F.y(d.s) + 4).attr('stroke', (d) => cat(d.i)).attr('stroke-width', 1.3);
  F.layer('hit').selectAll('path').data(cs).join('path').attr('fill', 'none').attr('stroke', 'transparent').attr('stroke-width', 12).attr('d', (d) => stepPath(F, d.c, tEnd))
    .on('mousemove', (ev, d) => { const tt = F.x.invert(d3.pointer(ev, F.g.node())[0]); tip(ev, `${d.i ? 'high' : 'low'} group<br>${tt.toFixed(1)} months: ${pc(kmAt(d.c, tt))} progression-free<br>n = ${d.c.pts[0].n}`); })
    .on('mouseleave', untip);
}
function splitAt(s, q) { const thr = d3.quantile(s.slice().sort(d3.ascending), q); return s.map((v) => +(v > thr)); }
/** Log-rank test, O(n log n). g = 0/1. Returns {chi, p, hr}. */
function logrank(t, e, g) {
  const o = d3.range(t.length).sort((a, b) => t[a] - t[b]);
  let n1 = sum(g), n0 = t.length - n1, O1 = 0, E1 = 0, Vv = 0, O0 = 0, k = 0;
  while (k < o.length) {
    const tt = t[o[k]]; let d1 = 0, d0 = 0, r1 = 0, r0 = 0;
    while (k < o.length && t[o[k]] === tt) { const i = o[k]; if (g[i]) { r1++; if (e[i]) d1++; } else { r0++; if (e[i]) d0++; } k++; }
    const N = n1 + n0, d = d1 + d0;
    if (d && N > 1) { E1 += d * n1 / N; O1 += d1; O0 += d0; Vv += d * (n1 / N) * (n0 / N) * (N - d) / (N - 1); }
    n1 -= r1; n0 -= r0;
  }
  const chi = Vv > 0 ? (O1 - E1) ** 2 / Vv : 0, E0 = O1 + O0 - E1;
  return { chi, p: pNormal(Math.sqrt(chi)), hr: (O1 / E1) / (O0 / E0) };
}
/** Risk-set sums for the Cox partial likelihood (Breslow ties). Calls fn(eventRows, S0, S1, S2) per distinct event time. */
function riskSets(beta, X, t, e, fn) {
  const p = beta.length, o = d3.range(t.length).sort((a, b) => t[b] - t[a]);
  let S0 = 0; const S1 = new Array(p).fill(0), S2 = Array.from({ length: p }, () => new Array(p).fill(0));
  let k = 0;
  while (k < o.length) {
    const tt = t[o[k]], grp = [];
    while (k < o.length && t[o[k]] === tt) grp.push(o[k++]);
    grp.forEach((i) => { const w = Math.exp(X[i].reduce((s, v, j) => s + v * beta[j], 0)); S0 += w; for (let a = 0; a < p; a++) { S1[a] += w * X[i][a]; for (let b = 0; b < p; b++) S2[a][b] += w * X[i][a] * X[i][b]; } });
    const ev = grp.filter((i) => e[i]);
    if (ev.length) fn(tt, ev, S0, S1, S2);
  }
}
function coxLL(beta, X, t, e) {
  let ll = 0;
  riskSets(beta, X, t, e, (tt, ev, S0) => { ev.forEach((i) => { ll += X[i].reduce((s, v, j) => s + v * beta[j], 0) - Math.log(S0); }); });
  return ll;
}
/** Cox regression by Newton–Raphson. Returns {beta, se, ll}. */
function coxFit(X, t, e) {
  const p = X[0].length; let beta = new Array(p).fill(0), H = null;
  for (let it = 0; it < 30; it++) {
    const g = new Array(p).fill(0); H = Array.from({ length: p }, () => new Array(p).fill(0));
    riskSets(beta, X, t, e, (tt, ev, S0, S1, S2) => {
      const d = ev.length;
      ev.forEach((i) => { for (let a = 0; a < p; a++) g[a] += X[i][a]; });
      for (let a = 0; a < p; a++) { g[a] -= d * S1[a] / S0; for (let b = 0; b < p; b++) H[a][b] += d * (S2[a][b] / S0 - (S1[a] / S0) * (S1[b] / S0)); }
    });
    const step = solve(H, g);
    beta = beta.map((b, a) => b + step[a]);
    if (Math.max(...step.map(Math.abs)) < 1e-9) break;
  }
  const se = d3.range(p).map((a) => Math.sqrt(solve(H, d3.range(p).map((b) => +(a === b)))[a]));
  return { beta, se, ll: coxLL(beta, X, t, e) };
}
/** Breslow cumulative baseline hazard at x = 0: [[t, H0(t)], ...]. */
function breslow(beta, X, t, e) {
  const out = []; let H = 0;
  riskSets(beta, X, t, e, (tt, ev, S0) => { out.push([tt, ev.length / S0]); });
  return out.reverse().map(([tt, dh]) => { H += dh; return [tt, H]; });
}
/** Harrell's C-index (higher risk = earlier event). */
function cindex(t, e, r) {
  let num = 0, den = 0;
  for (let i = 0; i < t.length; i++) if (e[i]) for (let j = 0; j < t.length; j++) if (t[j] > t[i]) { den++; num += r[i] > r[j] ? 1 : r[i] === r[j] ? 0.5 : 0; }
  return num / den;
}
function comparablePairs(t, e) { let n = 0; for (let i = 0; i < t.length; i++) if (e[i]) for (let j = 0; j < t.length; j++) if (t[j] > t[i]) n++; return n; }
