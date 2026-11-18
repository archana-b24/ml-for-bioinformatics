// What is machine learning? Entry lesson for the whole site.
// Arc: what → why (rules are unknown; ML vs classical statistics) → where → how it works (6 steps: the cohort → a rule
//      learned by search → checking it on new patients → learning-curve lab with try/see/means table → unsupervised
//      clustering → self-supervised masking) → pitfalls → recap.
// Every number in the prose is computed from data/study/cohort.json at render time.
import { figure, header, slider, select, button, checkbox, stats, legend, seeAlso, animate, tip, untip, el, html } from '../lib/ui.js';
import { frame, cat, css, ink, ink2, muted } from '../lib/plot.js';
import { rng, mean, sd, ridge } from '../lib/data.js';
import { fitLogistic, predictProb, auc, confusion, standardize, pNormal } from '../lib/models.js';
import { loadCohort, FEATURES, studyCard } from '../lib/study.js';
import { lessonNav, chapter, prose, rung, worked, quiz, cards, table, code, recap } from '../lib/lesson.js';
const d3 = window.d3;

const um = (t) => t.replace(/^-/, '−');
const f2 = (v) => (Math.abs(v) < 0.005 ? '0.00' : um(v.toFixed(2)));
const f1 = (v) => um(v.toFixed(1));
const pc = (v) => Math.round(100 * v) + '%';
const sum = (a) => a.reduce((s, v) => s + v, 0);
const pFmt = (p) => (p < 0.001 ? '< 0.001' : p.toFixed(3));
const jit = (i) => (((i * 7919) % 101) / 101 - 0.5);
const CENTRES = ['Centre A', 'Centre B', 'Centre C'];
const PHEN = ['inflamed', 'excluded', 'desert'];
const LAB10 = ['ifng', 'log2tmb', 'caf', 'pdl1', 'cd8', 'prolif', 'age', 'smoker', 'male', 'centreB'];
const CUTF = { ifng: 'IFN-γ score', log2tmb: 'TMB (log2)', caf: 'CAF / TGF-β score', pdl1: 'PD-L1 TPS (%)', cd8: 'CD8 T cells (%)', prolif: 'Proliferation score', age: 'Age (years)' };

export default async function render(main) {
  const C = await loadCohort();
  const D = C.discovery, V = C.validation, ALL = C.all;
  const y = D.map((p) => p.response), yv = V.map((p) => p.response);
  const n = D.length, nV = V.length, nR = sum(y), base = nR / n, baseV = mean(yv);
  const get = (k) => (p) => FEATURES[k].get(p);

  // Learned single cut-off on IFN-γ (used in chapters 1 and 2).
  const cutI = bestCut(D.map(get('ifng')), y);
  const cutPred = (p) => (cutI.dir > 0 ? p.ifng > cutI.t : p.ifng < cutI.t);
  // Logistic regression on the three true measurements.
  const F3 = ['ifng', 'log2tmb', 'caf'];
  const fit3 = fitLogistic(D.map((p) => F3.map((k) => get(k)(p))), y);
  const prob3 = (p) => predictProb(F3.map((k) => get(k)(p)), fit3.b0, fit3.w);

  header(main, 'What is machine learning?',
    'Machine learning is the practice of <b>learning a rule from examples</b> instead of writing it by hand. An algorithm is shown cases whose answer is known, searches for a rule (a function from measurements to answer) that fits them, and is then judged by how well that rule works on <b>new cases it has not seen</b>.');
  lessonNav(main);
  studyCard(main, C, '<b>Question for this page:</b> what does it mean for an algorithm to "learn" which patients respond to anti-PD-1, how is that different from a classical statistical analysis of the same data, and how much data does learning need?');

  // =====================================================================================
  // 1. WHAT
  // =====================================================================================
  let ch = chapter(main, 'what', 'What is machine learning?', 'In one line: <b>examples in → rule out → rule judged on new examples</b>.');
  prose(ch, `<p>A <b>rule</b> (also called a <b>model</b> or <b>function</b>) maps the measurements on a sample to an answer: for example, from a tumour biopsy to "responds" or "does not respond". In classical programming a person writes the rule. In machine learning (ML) a person chooses a <b>family</b> of possible rules, and an algorithm chooses the member of that family that best fits a set of <b>examples</b> with known answers. The test of success is not the fit to those examples but the accuracy on <b>new</b> samples.</p>
    <p>The diagram shows the simplest possible case on the study data: the rule family is "respond if IFN-γ is above a cut-off", and learning means choosing the cut-off.</p>`);
  const exV = V.filter((p) => p.response && cutPred(p)).sort((a, b) => Math.abs(a.ifng - cutI.t - 1) - Math.abs(b.ifng - cutI.t - 1))[0];
  ch.append(pipelineSvg(n, nR, cutI, exV));
  prose(ch, '<p>Key terms:</p>');
  table(ch, [
    { t: 'Example / sample', m: 'One unit with measurements (and possibly a known answer)', s: `one patient, e.g. ${D[0].id}` },
    { t: 'Feature (<i>x</i>)', m: 'A measurement used as input', s: 'IFN-γ score, TMB, PD-L1, age, …' },
    { t: 'Label / outcome (<i>y</i>)', m: 'The answer to be predicted', s: 'responded (1) or not (0)' },
    { t: 'Model / rule (<i>f</i>)', m: 'A function from features to a prediction, <i>ŷ</i> = <i>f</i>(<i>x</i>)', s: `"respond if IFN-γ > ${f2(cutI.t)}"` },
    { t: 'Parameter', m: 'A number inside the rule that learning adjusts', s: `the cut-off ${f2(cutI.t)}` },
    { t: 'Training', m: 'Choosing parameter values that fit the training examples', s: `${n} discovery patients (Centres A, B)` },
    { t: 'Generalisation', m: 'How well the rule works on new samples', s: `${nV} validation patients (Centre C)` },
    { t: 'Supervised / unsupervised / self-supervised', m: 'Learning with labels / without labels / with labels made from the data itself', s: 'predict response / find patient groups / predict a hidden measurement' },
  ], [['t', 'Term'], ['m', 'Meaning'], ['s', 'In our study']]);
  prose(ch, '<p>Three broad kinds of learning differ in <b>what the examples contain</b>, not in the mathematics. Chapter 4 runs each of them on the study.</p>');
  table(ch, [
    { k: '<b>Supervised</b>', g: 'features + a known label for each example', l: 'a rule that predicts the label', e: 'predict response to anti-PD-1 from the biopsy (step 4)' },
    { k: '<b>Unsupervised</b>', g: 'features only', l: 'structure: groups, axes of variation', e: 'find patient subgroups (immune phenotypes) or cell types (step 5)' },
    { k: '<b>Self-supervised</b>', g: 'features only; part of each example is hidden and used as the label', l: 'a representation that can fill in the hidden part', e: 'predict a masked measurement from the others; the principle behind foundation models (step 6)' },
  ], [['k', 'Kind'], ['g', 'Each example contains'], ['l', 'What is learned'], ['e', 'In our study']]);

  // =====================================================================================
  // 2. WHY
  // =====================================================================================
  ch = chapter(main, 'why', 'Why do we need it?', 'When the rule is unknown, and when it is the right tool instead of classical statistics.');
  prose(ch, `<p>The biology that decides response is not known in advance, and it involves several measurements at once. A hand-written rule encodes a guess; a learned rule is fitted to the data and can be checked on new patients. The table compares rules on the ${nV} validation patients. Because the data are simulated, the last row shows the <b>true rule</b> (predict response when the simulated probability is at least 0.5), which no learned rule can be expected to beat.</p>`);
  const pdlPred = V.map((p) => (p.pdl1 >= 50 ? 1 : 0));
  const rules = [
    { r: 'PD-L1 TPS ≥ 50% (fixed in advance; a common clinical cut-off)', pv: pdlPred, sc: V.map((p) => p.pdl1), how: 'written by hand' },
    { r: `IFN-γ ${cutI.dir > 0 ? '>' : '<'} ${f2(cutI.t)}`, pv: V.map((p) => (cutPred(p) ? 1 : 0)), sc: V.map((p) => cutI.dir * p.ifng), how: 'cut-off learned on discovery' },
    { r: 'Logistic regression on IFN-γ, TMB, CAF', pv: V.map((p) => (prob3(p) >= 0.5 ? 1 : 0)), sc: V.map(prob3), how: 'weights learned on discovery' },
    { r: 'True rule (simulation only)', pv: V.map((p) => (p.p_true >= 0.5 ? 1 : 0)), sc: V.map((p) => p.p_true), how: 'known only because the data are simulated' },
  ].map((o) => { const cm = confusion(o.pv, yv, 0.5); return { ...o, acc: cm.acc, sens: cm.sens, spec: cm.spec, auc: auc(o.sc, yv) }; });
  table(ch, rules, [['r', 'Rule'], ['how', 'Origin'], ['acc', 'Accuracy', pc], ['sens', 'Responders found', pc], ['spec', 'Non-responders correctly excluded', pc], ['auc', 'AUC', (v) => v.toFixed(3)]], { cls: 'compact', rowClass: (r) => (r.how.startsWith('known') ? 'hl' : '') });
  const [rP, rC, rL, rT] = rules;
  prose(ch, `<p>The hand-written PD-L1 rule finds ${pc(rP.sens)} of responders. The learned rules come close to the true rule: accuracy ${pc(rC.acc)} and ${pc(rL.acc)} against ${pc(rT.acc)}. ${rC.acc > rT.acc ? `The single learned cut-off even scores slightly higher than the true rule on these ${nV} patients. This is chance: with ${nV} patients one patient is ${f1(100 / nV)} percentage points, and the true rule is the best possible rule only on average over many patients.` : ''} The true rule itself misclassifies ${pc(1 - rT.acc)} of patients, because response is random given the measurements: two patients with identical biopsies can have different outcomes. This <b>irreducible error</b> sets a ceiling for every model. AUC (the probability that a random responder is scored above a random non-responder; 0.5 = chance, 1 = perfect ranking) separates the models more clearly than accuracy: ${rL.auc.toFixed(3)} for the three-measurement model vs ${rC.auc.toFixed(3)} for IFN-γ alone.</p>`);

  prose(ch, '<h3>Machine learning or classical statistics?</h3><p>Both fit models to data, often the <i>same</i> model. They differ in the question asked.</p>');
  table(ch, [
    { a: 'Goal', s: '<b>Inference</b>: estimate an effect and its uncertainty (is TMB associated with response, and how strongly?)', m: '<b>Prediction</b>: give the right answer for new samples (which Centre C patients will respond?)' },
    { a: 'Output', s: 'effect sizes, confidence intervals, p-values', m: 'predictions, probabilities, a rule that can be deployed' },
    { a: 'Judged by', s: 'validity of assumptions, correct error control', m: 'performance on held-out data' },
    { a: 'Typical models', s: 'few, pre-specified, interpretable (linear models, Cox, mixed models)', m: 'any model that predicts well, including many features, trees, neural networks' },
    { a: 'Typical omics step', s: 'differential expression, survival association, GWAS', m: 'response classifier, cell-type annotation, imputation' },
  ], [['a', ''], ['s', 'Classical statistics'], ['m', 'Machine learning']], { cls: 'compact' });
  const jT = F3.indexOf('log2tmb'), wT = fit3.w[jT], seT = fit3.se[jT + 1];
  worked(ch, 'One model, two questions',
    `<p>The logistic regression on IFN-γ, TMB and CAF, fitted to the ${n} discovery patients, answers both kinds of question:</p>
     <ul><li><b>Inference.</b> Each doubling of TMB multiplies the odds of response by <b>${f2(Math.exp(wT))}</b> (95% CI ${f2(Math.exp(wT - 1.96 * seT))}–${f2(Math.exp(wT + 1.96 * seT))}, p = ${pFmt(pNormal(wT / seT))}), holding IFN-γ and CAF fixed. The true value in the simulation is ${f2(Math.exp(C.truth.response_logit.log2_tmb))}. This statement concerns a population effect and depends on the model assumptions.</li>
     <li><b>Prediction.</b> On the ${nV} Centre C patients the model ranks a responder above a non-responder ${pc(rL.auc)} of the time (AUC ${rL.auc.toFixed(3)}). This statement concerns individual patients and is checked directly on new data.</li></ul>
     <p>Use statistics when the question is "is there an effect, how large, and how sure are we?", particularly with few samples or a designed experiment. Use ML when the question is "what will happen for this new sample?" and enough labelled examples exist to fit and test a rule. Many analyses need both. The <a href="#/omics/overview">study overview in tab 2</a> marks each step of an omics analysis as ML, statistics or both.</p>`);

  // =====================================================================================
  // 3. WHERE
  // =====================================================================================
  ch = chapter(main, 'where', 'Where is it used?', 'Examples of each kind of learning in bioinformatics.');
  cards(ch, [
    { tag: 'Supervised · clinical', title: 'Treatment response and risk', body: 'Predicting response to immunotherapy from tumour RNA-seq, TMB and PD-L1, as on this page; risk scores from clinical and molecular data.' },
    { tag: 'Supervised · cells', title: 'Cell-type annotation', body: 'Classifiers trained on labelled reference atlases (for example CellTypist, a regularised logistic regression) label cells in new scRNA-seq datasets.' },
    { tag: 'Supervised · regression', title: 'Epigenetic clocks', body: 'Penalised linear regression from DNA-methylation CpGs to chronological age (Horvath clock); the residual is read as accelerated ageing.' },
    { tag: 'Supervised · variants', title: 'Variant effect prediction', body: 'Scores such as CADD rank genetic variants by likely deleteriousness, trained on features of sequence, conservation and annotation.' },
    { tag: 'Unsupervised · patients', title: 'Molecular subtypes', body: 'Clustering tumour expression profiles into subtypes (for example the intrinsic breast-cancer subtypes) that are later tested for clinical relevance.' },
    { tag: 'Unsupervised · cells', title: 'Clusters and cell types', body: 'PCA, a nearest-neighbour graph and Leiden clustering in every Scanpy or Seurat workflow. Clusters are named afterwards from marker genes.' },
    { tag: 'Unsupervised · QC', title: 'Batch and outlier detection', body: 'PCA of all samples, coloured by batch or centre, is the standard first check that technical variation does not dominate biology.' },
    { tag: 'Self-supervised · cells', title: 'Single-cell foundation models', body: 'Models such as Geneformer and scGPT are pretrained by hiding part of each cell\'s expression profile and predicting it, then adapted to tasks with few labels.' },
    { tag: 'Self-supervised · sequence', title: 'Protein and DNA language models', body: 'Models trained to predict masked amino acids or bases (for example the ESM protein models) learn representations used for structure and variant-effect prediction.' },
  ]);
  prose(ch, `<p><b>When not to use it:</b> the question is about an effect, not a prediction (estimate it with a pre-specified statistical model and report its uncertainty); there are too few labelled examples to both fit and test a rule (tens of samples, thousands of features); the answer is already known from mechanism or a validated assay; or the decision requires a causal claim (what would happen under a different treatment), which a model fitted to observational data does not provide on its own.</p>`);

  // =====================================================================================
  // 4. HOW
  // =====================================================================================
  ch = chapter(main, 'how', 'How it works', 'Six steps: the examples, a rule learned by search, checking it on new patients, a learning lab, then learning without labels.');

  // ---- Step 1: the cohort in depth
  let rb = rung(ch, 1, 'The examples: the study cohort',
    `<p>Every lesson on this site uses one simulated cohort of ${ALL.length} patients with non-small-cell lung cancer treated with an anti-PD-1 antibody. The columns fall into four roles:</p>`);
  const roleRows = [
    { c: 'IFN-γ score, TMB, CAF / TGF-β score', r: 'feature (causal)', d: 'The three measurements that drive response in the simulation. IFN-γ and TMB raise the probability of response; CAF lowers it.' },
    { c: 'PD-L1 TPS, CD8 T cells', r: 'feature (proxy)', d: 'Driven by the same immune activity as IFN-γ. Correlated with response, but add almost nothing once IFN-γ is known.' },
    { c: 'Proliferation score', r: 'feature (prognostic only)', d: 'Shortens survival but has no effect on response.' },
    { c: 'Age, sex, smoking, centre', r: 'clinical / design', d: 'Smoking acts only through TMB. Centre B enrolled more inflamed tumours, so centre is confounded with response.' },
    { c: 'Immune phenotype', r: 'hidden structure', d: 'Inflamed, excluded or desert, from pathology review. Not used as a model input; unsupervised learning tries to rediscover it (step 5).' },
    { c: 'Responded; overall survival', r: 'label / outcome', d: 'Response (yes/no) is the label for supervised learning. Survival (months, with censoring) is used on the survival pages.' },
    { c: 'True response probability', r: 'truth (simulation only)', d: 'The probability the simulation used for each patient. Never available in real data; used here to check every model.' },
  ];
  table(rb, roleRows, [['c', 'Columns'], ['r', 'Role'], ['d', 'What it is']], { cls: 'compact' });
  const byC = CENTRES.map((c) => { const ps = ALL.filter((p) => p.centre === c); return { c, n: ps.length, r: mean(ps.map((p) => p.response)), infl: mean(ps.map((p) => (p.phenotype === 'inflamed' ? 1 : 0))) }; });
  prose(rb, `<p><b>The split.</b> The ${n} <b>discovery</b> patients (Centre A: ${byC[0].n}, Centre B: ${byC[1].n}) are used to learn. The ${nV} <b>validation</b> patients come from a third hospital (Centre C) and are used only to check the learned rules. Splitting by hospital rather than at random is the stricter test, because a new hospital differs in case mix: ${pc(byC[0].infl)}, ${pc(byC[1].infl)} and ${pc(byC[2].infl)} of tumours are inflamed in Centres A, B and C, and the response rates are ${byC.map((b) => pc(b.r)).join(', ')}.</p>`);
  const fe = figure(rb, 'Explore a column', 'Each dot is a patient, in one row per centre, coloured by outcome. Rows A and B are the discovery cohort; row C is the validation cohort. Diamonds mark the mean of responders and non-responders in each centre. Hover a patient for details.');
  let col = 'ifng';
  select(fe.controls, { label: 'Column', value: col, options: Object.entries(CUTF), onChange: (v) => { col = v; paintE(); } });
  const sE = stats(fe.root);
  const FE = frame(fe.body, { width: 820, height: 260, x: [0, 1], y: [0, 3], xLabel: '', grid: false, axes: false, margin: { left: 124, bottom: 42 } });
  const axE = FE.g.append('g').attr('class', 'axis x').attr('transform', `translate(0,${FE.h})`);
  const axLab = FE.g.append('text').attr('class', 'ax-label').attr('x', FE.w / 2).attr('y', FE.h + 36).attr('text-anchor', 'middle');
  const rowY = (c) => FE.y(2.5 - CENTRES.indexOf(c));
  CENTRES.forEach((c, i) => {
    FE.layer('rows').append('text').attr('x', -10).attr('y', rowY(c) + 4).attr('text-anchor', 'end').attr('font-size', 12.5).attr('fill', ink2()).text(c + (i < 2 ? ' (disc.)' : ' (valid.)'));
    if (i) FE.layer('rows').append('line').attr('x1', 0).attr('x2', FE.w).attr('y1', FE.y(3 - i)).attr('y2', FE.y(3 - i)).attr('stroke', css('--border'));
  });
  legend(fe.body, [['no response', cat(0)], ['responded', cat(1)], ['group mean', ink()]]);
  function paintE() {
    const g = get(col), xs = ALL.map(g), lo = d3.min(xs), hi = d3.max(xs), pad = (hi - lo) * 0.03;
    FE.x.domain([lo - pad, hi + pad]);
    axE.call(d3.axisBottom(FE.x).ticks(8)); axLab.text(CUTF[col]);
    FE.layer('pts').selectAll('circle').data(ALL).join('circle').attr('cx', (d) => FE.x(g(d))).attr('cy', (d, i) => rowY(d.centre) + jit(i) * 52)
      .attr('r', 3.6).attr('fill', (d) => cat(d.response)).attr('fill-opacity', 0.75).attr('stroke', css('--surface')).attr('stroke-width', 0.8)
      .on('mousemove', (e, d) => tip(e, `${d.id} · ${d.centre} · ${d.phenotype}<br>${CUTF[col]}: ${FEATURES[col].fmt(g(d))}<br>${d.response ? 'responded' : 'no response'}`)).on('mouseleave', untip);
    const ms = [];
    CENTRES.forEach((c) => [0, 1].forEach((r) => { const ps = ALL.filter((p) => p.centre === c && p.response === r); ms.push({ c, r, m: mean(ps.map(g)), n: ps.length }); }));
    FE.layer('m').selectAll('path').data(ms).join('path').attr('d', d3.symbol(d3.symbolDiamond, 110))
      .attr('transform', (d) => `translate(${FE.x(d.m)},${rowY(d.c) + (d.r ? -1 : 1) * 30})`).attr('fill', ink()).attr('stroke', (d) => cat(d.r)).attr('stroke-width', 2)
      .on('mousemove', (e, d) => tip(e, `${d.c}, ${d.r ? 'responders' : 'non-responders'} (n = ${d.n})<br>mean ${FEATURES[col].fmt(d.m)}`)).on('mouseleave', untip);
    const r1 = D.filter((p) => p.response).map(g), r0 = D.filter((p) => !p.response).map(g);
    sE.set({ 'discovery mean, responders': FEATURES[col].fmt(mean(r1)), 'non-responders': FEATURES[col].fmt(mean(r0)), 'difference in SDs': f2((mean(r1) - mean(r0)) / sd(D.map(g))), 'correlation with IFN-γ': f2(corr(ALL.map(g), ALL.map((p) => p.ifng))) });
  }
  paintE();
  quiz(rb, 'PD-L1 differs clearly between responders and non-responders. Does that make it one of the causes of response in this simulation?',
    `<p>No. PD-L1 is induced by the same immune activity that raises IFN-γ (correlation ${f2(corr(ALL.map((p) => p.pdl1), ALL.map((p) => p.ifng)))}), so it tracks response without driving it. A supervised model can use it for prediction, but its weight says nothing about mechanism.</p>`);

  // ---- Step 2: learning a cut-off
  rb = rung(ch, 2, 'Learning as search: the best cut-off',
    `<p>The simplest rule family is "predict response when one measurement is above (or below) a cut-off". Every such rule has one parameter, the cut-off. <b>Learning</b> means trying candidate values and keeping the one that classifies the most discovery patients correctly. Larger models follow the same pattern with more parameters and a cleverer search.</p>`);
  const fc = figure(rb, 'Learning a cut-off from the discovery patients', 'Top: the 180 discovery patients along the chosen measurement; the shaded side of the line is predicted to respond. Bottom: discovery accuracy for every cut-off (the dashed line is the accuracy of predicting "no response" for everyone). Drag the cut-off, or let the algorithm sweep all values and keep the best.');
  let cf = 'ifng', cDir = 1, cT = 6, stopC = null, sweep = [];
  const cfSel = select(fc.controls, { label: 'Measurement', value: cf, options: Object.entries(CUTF), onChange: (v) => { halt(); cf = v; resetC(); } });
  const cS = slider(fc.controls, { label: 'cut-off', min: 0, max: 1, step: 0.01, value: 0.5, fmt: (v) => f2(+v), onInput: (v) => { halt(); cT = v; paintC(); } });
  const cBox = checkbox(fc.controls, { label: 'predict response below the cut-off', checked: false, onChange: (c) => { halt(); cDir = c ? -1 : 1; paintC(); } });
  button(fc.controls, 'Learn the best cut-off', () => runSweep(), true);
  const sC = stats(fc.root);
  const FC1 = frame(fc.body, { width: 820, height: 170, x: [0, 1], y: [-0.5, 1.5], grid: false, axes: false, margin: { left: 96, bottom: 24 } });
  const ax1 = FC1.g.append('g').attr('class', 'axis x').attr('transform', `translate(0,${FC1.h})`);
  FC1.layer('lab').selectAll('text').data([[1, 'responded'], [0, 'no response']]).join('text').attr('x', -10).attr('y', (d) => FC1.y(d[0]) + 4).attr('text-anchor', 'end').attr('font-size', 12.5).attr('fill', ink2()).text((d) => d[1]);
  const shadeC = FC1.layer('shade').append('rect').attr('y', 0).attr('height', FC1.h).attr('fill', cat(1)).attr('fill-opacity', 0.1);
  const ptsC = FC1.layer('pts'), lineC = FC1.layer('thr');
  const FC2 = frame(fc.body, { width: 820, height: 200, x: [0, 1], y: [0.3, 0.9], xLabel: '', yLabel: 'accuracy', margin: { left: 96, bottom: 44 } });
  const ax2Lab = FC2.g.append('text').attr('class', 'ax-label').attr('x', FC2.w / 2).attr('y', FC2.h + 36).attr('text-anchor', 'middle');
  FC2.layer('base').append('line').attr('x1', 0).attr('x2', FC2.w).attr('y1', FC2.y(1 - base)).attr('y2', FC2.y(1 - base)).attr('stroke', muted()).attr('stroke-dasharray', '5 4');
  FC2.layer('base').append('text').attr('x', FC2.w - 4).attr('y', FC2.y(1 - base) + 14).attr('text-anchor', 'end').attr('font-size', 11.5).attr('fill', muted()).text(`everyone "no response": ${pc(1 - base)}`);
  const curveC = FC2.layer('curve'), markC = FC2.layer('mark');
  legend(fc.body, [['no response', cat(0)], ['responded', cat(1)], ['accuracy, response above cut-off', cat(6)], ['accuracy, response below cut-off', cat(3)]]);
  const accAt = (xs, t, dir) => mean(xs.map((v, i) => ((dir > 0 ? v > t : v < t) ? 1 : 0) === y[i] ? 1 : 0));
  function resetC() {
    const xs = D.map(get(cf)), lo = d3.min(xs), hi = d3.max(xs), pad = (hi - lo) * 0.03;
    [FC1, FC2].forEach((F) => F.x.domain([lo - pad, hi + pad]));
    ax1.call(d3.axisBottom(FC1.x).ticks(8)); FC2.g.select('.axis.x').call(d3.axisBottom(FC2.x).ticks(8)); ax2Lab.text(CUTF[cf] + ' (cut-off)');
    const step = (hi - lo) / 200;
    cS.input.min = lo; cS.input.max = hi; cS.input.step = step;
    cT = d3.median(xs); cS.set(cT); sweep = [];
    curveC.selectAll('*').remove();
    ptsC.selectAll('circle').data(D).join('circle').attr('cx', (d) => FC1.x(get(cf)(d))).attr('cy', (d, i) => FC1.y(d.response) + jit(i) * 34).attr('r', 3.6)
      .attr('fill', (d) => cat(d.response)).attr('fill-opacity', 0.75).attr('stroke', css('--surface')).attr('stroke-width', 0.8)
      .on('mousemove', (e, d) => tip(e, `${d.id}<br>${CUTF[cf]}: ${FEATURES[cf].fmt(get(cf)(d))}<br>${d.response ? 'responded' : 'no response'}`)).on('mouseleave', untip);
    paintC();
  }
  function paintC() {
    const xs = D.map(get(cf)), acc = accAt(xs, cT, cDir);
    const x0 = FC1.x(cT);
    shadeC.attr('x', cDir > 0 ? x0 : 0).attr('width', cDir > 0 ? FC1.w - x0 : x0);
    lineC.selectAll('line').data([0]).join('line').attr('x1', x0).attr('x2', x0).attr('y1', 0).attr('y2', FC1.h).attr('stroke', ink()).attr('stroke-width', 2);
    curveC.selectAll('path').data([1, -1].map((d) => sweep.filter((s) => s.dir === d))).join('path').attr('fill', 'none').attr('stroke', (d, i) => cat(i ? 3 : 6)).attr('stroke-width', 2)
      .attr('d', (d) => d3.line().x((s) => FC2.x(s.t)).y((s) => FC2.y(Math.max(0.3, s.acc)))(d));
    markC.selectAll('circle').data([0]).join('circle').attr('cx', x0).attr('cy', FC2.y(Math.max(0.3, acc))).attr('r', 6).attr('fill', cat(cDir > 0 ? 6 : 3)).attr('stroke', css('--surface')).attr('stroke-width', 1.5);
    const cm = confusion(xs.map((v) => ((cDir > 0 ? v > cT : v < cT) ? 1 : 0)), y);
    const vv = V.map(get(cf)), accV = mean(vv.map((v, i) => ((cDir > 0 ? v > cT : v < cT) ? 1 : 0) === yv[i] ? 1 : 0));
    sC.set({ rule: `respond if ${cDir > 0 ? '>' : '<'} ${FEATURES[cf].fmt(cT)}`, 'discovery: correct': `${cm.tp + cm.tn} / ${n}`, 'discovery accuracy': pc(acc), 'responders found': `${cm.tp} / ${cm.tp + cm.fn}`, 'validation accuracy (Centre C)': pc(accV) });
  }
  function halt() { if (stopC) { stopC(); stopC = null; } }
  function runSweep() {
    halt();
    const xs = D.map(get(cf)), cands = candCuts(xs);
    sweep = []; let i = 0, best = { acc: -1 };
    stopC = animate(() => {
      for (let s = 0; s < 3 && i < 2 * cands.length; s++, i++) {
        const dir = i < cands.length ? 1 : -1, t = cands[i % cands.length], acc = accAt(xs, t, dir);
        sweep.push({ t, dir, acc }); if (acc > best.acc) best = { t, dir, acc };
        cT = t; cDir = dir;
      }
      if (i >= 2 * cands.length) { cT = best.t; cDir = best.dir; }
      cS.set(cT); cBox.checked = cDir < 0; paintC();
      return i < 2 * cands.length;
    });
  }
  resetC();
  { const xs0 = D.map(get(cf)); sweep = [1, -1].flatMap((dir) => candCuts(xs0).map((t) => ({ t, dir, acc: accAt(xs0, t, dir) }))); paintC(); }
  prose(rb, `<p>For IFN-γ, the search keeps <b>IFN-γ ${cutI.dir > 0 ? '>' : '<'} ${f2(cutI.t)}</b>, which classifies ${pc(cutI.acc)} of discovery patients correctly. The three ingredients of every learning method are visible here: a <b>model family</b> (one cut-off on one measurement), a <b>loss</b> that scores a candidate on the examples (here the error rate, 1 − accuracy), and an <b>optimiser</b> that searches for the lowest loss (here, trying every value). Logistic regression replaces them with a weighted sum, log-loss and gradient-based search; a neural network with stacked layers, the same loss and the same kind of search.</p>`);

  // ---- Step 3: generalisation
  const cutRows = Object.keys(CUTF).map((k) => {
    const b = bestCut(D.map(get(k)), y), vv = V.map(get(k));
    const accV = mean(vv.map((v, i) => ((b.dir > 0 ? v > b.t : v < b.t) ? 1 : 0) === yv[i] ? 1 : 0));
    return { k, lab: CUTF[k], rule: `${b.dir > 0 ? '>' : '<'} ${FEATURES[k].fmt(b.t)}`, a: b.acc, v: accV, d: accV - b.acc };
  });
  const best = cutRows.slice().sort((a, b) => b.a - a.a)[0];
  const cP = cutRows.find((r) => r.k === 'prolif'), cA = cutRows.find((r) => r.k === 'age');
  rb = rung(ch, 3, 'Checking the rule on new patients',
    `<p>A rule fitted to examples always looks at least somewhat better on those examples than on new ones, because the search also adapts to their noise. The appropriate measure is performance on patients that played no part in learning. The table repeats the search for each measurement and then applies the learned rule, unchanged, to Centre C.</p>`);
  table(rb, cutRows, [['lab', 'Measurement'], ['rule', 'Learned rule (respond if …)'], ['a', 'Discovery accuracy', pc], ['v', 'Validation accuracy', pc], ['d', 'Change', (v) => { const r = Math.round(100 * v); return (r > 0 ? '+' : r < 0 ? '−' : '±') + Math.abs(r) + ' pts'; }]], { cls: 'compact', rowClass: (r) => (r.k === 'prolif' || r.k === 'age' ? 'hl' : '') });
  prose(rb, `<p>Two baselines frame these numbers. Predicting "no response" for every patient scores ${pc(1 - base)} in discovery and ${pc(1 - baseV)} in validation, so an accuracy near that value means the rule has learned nothing useful. Proliferation and age have no effect on response, yet their best cut-offs reach ${pc(cP.a)} and ${pc(cA.a)} on discovery; on Centre C they score ${pc(cP.v)} and ${pc(cA.v)}. ${best.k === 'ifng' ? 'The rule chosen as best on discovery, IFN-γ, is also best on validation here.' : `The rule chosen as best on discovery is ${best.lab}.`} Choosing the best of several candidates on the same examples makes the winner look better than it is; this is why a final check uses data that were not used for any choice.</p>`);
  quiz(rb, `The proliferation cut-off scores ${pc(cP.a)} on discovery, above the ${pc(1 - base)} baseline. Has it found a real signal?`,
    `<p>No. With ${n} patients and a free choice of cut-off and direction, some cut-off will separate the classes a little better than the baseline by chance. On validation it scores ${pc(cP.v)}, ${cP.v <= 1 - baseV ? 'no better than' : 'close to'} the ${pc(1 - baseV)} baseline. In the simulation proliferation affects survival, not response.</p>`);

  // ---- Step 4: the lab
  rb = rung(ch, 4, 'Lab: how much data does learning need?',
    `<p>Supervised learning improves with more labelled examples, up to the limit set by the irreducible error. In the lab, a model is trained on <b>k</b> patients drawn from the discovery cohort and tested on all ${nV} Centre C patients. Each k is repeated on 25 random draws (each draw contains at least one responder and one non-responder, and each larger sample contains the smaller ones).</p>`);
  const MODELS = {
    cut: { lab: 'Cut-off on IFN-γ (1 parameter)', feats: ['ifng'] },
    lr2: { lab: 'Logistic: IFN-γ + TMB (3 parameters)', feats: ['ifng', 'log2tmb'] },
    lr3: { lab: 'Logistic: IFN-γ + TMB + CAF (4 parameters)', feats: F3 },
    lr10: { lab: 'Logistic: all 10 measurements (11 parameters)', feats: LAB10 },
    nn1: { lab: '1-nearest neighbour, all 10 (memorises)', feats: LAB10 },
  };
  const KS = [4, 6, 8, 10, 15, 20, 30, 45, 60, 90, 120, 150, 180], NDRAW = 25;
  const orders = Array.from({ length: NDRAW }, (_, d) => drawOrder(y, 100 + d));
  const cache = new Map();
  const evalAt = (mk, k) => {
    const key = mk + '|' + k;
    if (!cache.has(key)) cache.set(key, orders.map((o) => trainEval(mk, o.slice(0, k))));
    return cache.get(key);
  };
  function trainEval(mk, idx) {
    const M = MODELS[mk], tr = idx.map((i) => D[i]), ytr = tr.map((p) => p.response);
    const rows = (ps) => ps.map((p) => M.feats.map((k) => get(k)(p)));
    let ptr, pv, sv, rule = null;
    if (mk === 'cut') {
      const b = bestCut(tr.map((p) => p.ifng), ytr), f = (p) => ((b.dir > 0 ? p.ifng > b.t : p.ifng < b.t) ? 1 : 0);
      ptr = tr.map(f); pv = V.map(f); sv = V.map((p) => b.dir * p.ifng); rule = b;
    } else if (mk === 'nn1') {
      const st = standardize(rows(tr)), Z = st.Z, Zv = rows(V).map(st.apply);
      ptr = ytr.slice();
      pv = Zv.map((z) => ytr[d3.minIndex(Z, (q) => q.reduce((s, v, j) => s + (v - z[j]) ** 2, 0))]); sv = pv;
    } else {
      const st = standardize(rows(tr)), fit = fitLogistic(st.Z, ytr, { l2: 1 });
      ptr = st.Z.map((z) => (predictProb(z, fit.b0, fit.w) >= 0.5 ? 1 : 0));
      sv = rows(V).map((r) => predictProb(st.apply(r), fit.b0, fit.w)); pv = sv.map((p) => (p >= 0.5 ? 1 : 0));
      rule = { st, fit };
    }
    return { idx, rule, accT: mean(ptr.map((v, i) => (v === ytr[i] ? 1 : 0))), accV: mean(pv.map((v, i) => (v === yv[i] ? 1 : 0))), aucV: auc(sv, yv), pv };
  }
  const trueAcc = rT.acc, trueAuc = rT.auc;
  const fl = figure(rb, 'Learning curve on the validation cohort',
    'Left: validation performance against the number of training patients k (log scale). The line is the mean over 25 draws, the band spans the middle 80% of draws, and the dashed line is training performance. Right: one draw at the selected k, on two of the measurements. Large ringed dots are the training patients; small dots are Centre C patients, crossed when the model misclassifies them. The line is the decision boundary when it can be drawn in these two dimensions.');
  let mk = 'lr3', ki = 3, metric = 'acc', stopL = null, grown = KS.length;
  select(fl.controls, { label: 'Model', value: mk, options: Object.entries(MODELS).map(([k, m]) => [k, m.lab]), onChange: (v) => { mk = v; grow(); } });
  select(fl.controls, { label: 'Metric', value: metric, options: [['acc', 'accuracy (threshold 0.5)'], ['auc', 'AUC']], onChange: (v) => { metric = v; paintL(); } });
  const kS = slider(fl.controls, { label: 'training patients k', min: 0, max: KS.length - 1, step: 1, value: ki, fmt: (i) => String(KS[i]), onInput: (i) => { ki = i; paintL(); } });
  button(fl.controls, 'Replay: add examples', () => grow(), true);
  const sL = stats(fl.root);
  const rowL = el('div', { class: 'fig-row' }); fl.body.append(rowL);
  const la = el('div'), lb = el('div'); rowL.append(la, lb);
  const FL = frame(la, { width: 480, height: 330, xScale: d3.scaleLog(), x: [4, 180], y: [0.4, 1.0], xLabel: 'training patients k', yLabel: 'validation performance' });
  FL.g.select('.axis.x').call(d3.axisBottom(FL.x).tickValues([4, 10, 20, 50, 100, 180]).tickFormat(d3.format('d')));
  const refL = FL.layer('ref'), bandL = FL.layer('band'), lineL = FL.layer('line'), dotL = FL.layer('dots');
  const FR = frame(lb, { width: 460, height: 330, x: [2.2, 10.2], y: [0.6, 5.5], xLabel: 'IFN-γ score', yLabel: 'log2 TMB' });
  FR.g.append('clipPath').attr('id', 'wml-clip').append('rect').attr('width', FR.w).attr('height', FR.h);
  const bndR = FR.layer('bnd').attr('clip-path', 'url(#wml-clip)'), valR = FR.layer('val'), xR = FR.layer('x'), trR = FR.layer('tr');
  legend(fl.body, [['no response', cat(0)], ['responded', cat(1)], ['validation mean ± middle 80%', cat(6)], ['training (dashed)', ink2()], ['true rule (ceiling)', ink()], ['everyone "no response"', muted()]]);
  function grow() {
    if (stopL) stopL();
    grown = 0; let last = 0;
    stopL = animate((dt, t) => {
      if (t - last < 90) return true;
      last = t; evalAt(mk, KS[grown]); grown++; ki = grown - 1; kS.set(ki); paintL();
      return grown < KS.length;
    });
  }
  function paintL() {
    const val = (r) => (metric === 'acc' ? r.accV : r.aucV);
    FL.y.domain(metric === 'acc' ? [0.4, 0.9] : [0.4, 1.0]);
    FL.g.select('.axis.y').call(d3.axisLeft(FL.y).ticks(6));
    FL.g.select('.gridline').call(d3.axisLeft(FL.y).ticks(6).tickSize(-FL.w).tickFormat(''));
    const refs = [{ v: metric === 'acc' ? trueAcc : trueAuc, c: ink(), t: 'true rule' }, { v: metric === 'acc' ? 1 - baseV : 0.5, c: muted(), t: metric === 'acc' ? 'everyone "no response"' : 'chance' }];
    refL.selectAll('line').data(refs).join('line').attr('x1', 0).attr('x2', FL.w).attr('y1', (d) => FL.y(d.v)).attr('y2', (d) => FL.y(d.v)).attr('stroke', (d) => d.c).attr('stroke-width', 1.4).attr('stroke-dasharray', '2 4');
    refL.selectAll('text').data(refs).join('text').attr('x', FL.w - 4).attr('y', (d) => FL.y(d.v) - 5).attr('text-anchor', 'end').attr('font-size', 11).attr('fill', ink2()).text((d) => `${d.t} ${metric === 'acc' ? pc(d.v) : d.v.toFixed(2)}`);
    const pts = KS.slice(0, grown).map((k) => {
      const rs = evalAt(mk, k), vs = rs.map(val).sort((a, b) => a - b);
      return { k, m: mean(vs), lo: d3.quantile(vs, 0.1), hi: d3.quantile(vs, 0.9), tr: mean(rs.map((r) => (metric === 'acc' ? r.accT : 1))) };
    });
    bandL.selectAll('path').data([pts]).join('path').attr('fill', cat(6)).attr('fill-opacity', 0.16)
      .attr('d', d3.area().x((d) => FL.x(d.k)).y0((d) => FL.y(d.lo)).y1((d) => FL.y(d.hi)));
    const ls = [{ p: pts.map((d) => [d.k, d.m]), c: cat(6), dash: null }];
    if (metric === 'acc') ls.push({ p: pts.map((d) => [d.k, d.tr]), c: ink2(), dash: '6 4' });
    lineL.selectAll('path').data(ls).join('path').attr('fill', 'none').attr('stroke', (d) => d.c).attr('stroke-width', 2).attr('stroke-dasharray', (d) => d.dash)
      .attr('d', (d) => d3.line().x((q) => FL.x(q[0])).y((q) => FL.y(Math.max(FL.y.domain()[0], q[1])))(d.p));
    dotL.selectAll('circle').data(pts).join('circle').attr('cx', (d) => FL.x(d.k)).attr('cy', (d) => FL.y(Math.max(FL.y.domain()[0], d.m)))
      .attr('r', (d) => (d.k === KS[ki] ? 7 : 4.5)).attr('fill', cat(6)).attr('stroke', (d) => (d.k === KS[ki] ? ink() : css('--surface'))).attr('stroke-width', 1.5)
      .on('mousemove', (e, d) => tip(e, `k = ${d.k} training patients<br>validation ${metric === 'acc' ? 'accuracy' : 'AUC'}: mean ${f2(d.m)}<br>middle 80% of draws: ${f2(d.lo)}–${f2(d.hi)}${metric === 'acc' ? `<br>training accuracy: ${f2(d.tr)}` : ''}`)).on('mouseleave', untip);
    // right panel: draw 0 at the selected k
    const k = KS[ki], r0 = evalAt(mk, k)[0], inTr = new Set(r0.idx);
    valR.selectAll('circle').data(V).join('circle').attr('cx', (d) => FR.x(d.ifng)).attr('cy', (d) => FR.y(d.log2tmb)).attr('r', 2.8).attr('fill', (d) => cat(d.response)).attr('fill-opacity', 0.55)
      .on('mousemove', (e, d) => { const i = V.indexOf(d); tip(e, `${d.id} (Centre C)<br>predicted ${r0.pv[i] ? 'response' : 'no response'} · ${d.response ? 'responded' : 'no response'}`); }).on('mouseleave', untip);
    const wrong = V.filter((d, i) => r0.pv[i] !== d.response);
    xR.selectAll('path').data(wrong).join('path').attr('d', d3.symbol(d3.symbolCross, 26)).attr('transform', (d) => `translate(${FR.x(d.ifng)},${FR.y(d.log2tmb)}) rotate(45)`).attr('fill', ink2()).attr('pointer-events', 'none');
    trR.selectAll('circle').data(D.filter((p, i) => inTr.has(i))).join('circle').attr('cx', (d) => FR.x(d.ifng)).attr('cy', (d) => FR.y(d.log2tmb)).attr('r', 5.5)
      .attr('fill', (d) => cat(d.response)).attr('stroke', ink()).attr('stroke-width', 1.4)
      .on('mousemove', (e, d) => tip(e, `${d.id} (training)<br>IFN-γ ${d.ifng}, TMB ${d.tmb}<br>${d.response ? 'responded' : 'no response'}`)).on('mouseleave', untip);
    let seg = [];
    if (mk === 'cut') seg = [[r0.rule.t, 0.6], [r0.rule.t, 5.5]];
    if (mk === 'lr2') {
      const { st, fit } = r0.rule, a = fit.w[0] / st.sd[0], b = fit.w[1] / st.sd[1], c0 = fit.b0 - a * st.mu[0] - b * st.mu[1];
      seg = Math.abs(b) < 1e-9 ? [[-c0 / a, 0.6], [-c0 / a, 5.5]] : [[2.2, -(c0 + a * 2.2) / b], [10.2, -(c0 + a * 10.2) / b]];
    }
    bndR.selectAll('line').data(seg.length ? [seg] : []).join('line').attr('x1', (d) => FR.x(d[0][0])).attr('y1', (d) => FR.y(d[0][1])).attr('x2', (d) => FR.x(d[1][0])).attr('y2', (d) => FR.y(d[1][1])).attr('stroke', ink()).attr('stroke-width', 2.2);
    const cur = pts.find((d) => d.k === k);
    sL.set({ k, 'validation (mean of 25 draws)': cur ? (metric === 'acc' ? pc(cur.m) : f2(cur.m)) : '…', 'range, middle 80%': cur ? (metric === 'acc' ? `${pc(cur.lo)}–${pc(cur.hi)}` : `${f2(cur.lo)}–${f2(cur.hi)}`) : '…', 'this draw: training acc.': pc(r0.accT), 'this draw: validation acc.': pc(r0.accV), boundary: seg.length ? 'drawn' : 'not drawable in 2 dimensions' });
  }
  paintL();
  // numbers for the try / see / means table
  const mAt = (m, k, f) => mean(evalAt(m, k).map(f)), qAt = (m, k, f, q) => d3.quantile(evalAt(m, k).map(f).sort((a, b) => a - b), q);
  const A = (r) => r.accV, U = (r) => r.aucV;
  const cut4 = mAt('cut', 4, A), cut180 = mAt('cut', 180, A), l3k10 = mAt('lr3', 10, U), l3k180 = mAt('lr3', 180, U);
  const l3k20 = mAt('lr3', 20, U), l10k20 = mAt('lr10', 20, U), l10k180 = mAt('lr10', 180, U);
  const nnT = mAt('nn1', 180, (r) => r.accT), nnV = mAt('nn1', 180, A), l3A180 = mAt('lr3', 180, A);
  const lo10 = qAt('lr3', 10, A, 0.1), hi10 = qAt('lr3', 10, A, 0.9);
  table(rb, [
    { t: '<b>Cut-off on IFN-γ</b>, slide k from 4 to 180', s: `Mean validation accuracy ${pc(cut4)} → <b>${pc(cut180)}</b>`, m: 'More examples pin down the parameter; performance rises and then levels off.' },
    { t: '<b>Logistic, three measurements</b>, metric AUC, k = 10 vs 180', s: `AUC ${f2(l3k10)} → <b>${f2(l3k180)}</b> (true rule ${f2(trueAuc)})`, m: 'With all discovery patients the model ranks Centre C patients almost as well as the true rule.' },
    { t: 'Same model, accuracy, k = 10; watch the band', s: `Middle 80% of draws: ${pc(lo10)} to ${pc(hi10)}`, m: 'With few examples, the result depends heavily on which patients were drawn. A single small study can over- or underestimate performance by chance.' },
    { t: '<b>All 10 measurements</b> vs <b>three</b>, AUC at k = 20 and 180', s: `k = 20: ${f2(l10k20)} vs ${f2(l3k20)}; k = 180: ${f2(l10k180)} vs ${f2(l3k180)}`, m: 'Extra parameters need extra examples. Uninformative features cost most when data are scarce.' },
    { t: '<b>1-nearest neighbour</b>, k = 180', s: `Training accuracy ${pc(nnT)}; validation <b>${pc(nnV)}</b> (logistic: ${pc(l3A180)})`, m: 'A rule that memorises the examples is perfect on them and says little about new patients. Training performance is not evidence of learning.' },
    { t: 'Compare any curve with the <b>true rule</b> line', s: `No model exceeds ${pc(trueAcc)} accuracy on average`, m: 'The remaining error is irreducible: the measurements do not fully determine the outcome.' },
  ], [['t', 'Try'], ['s', 'What you see'], ['m', 'What it means']], { cls: 'compact' });
  if (l10k180 > l3k180 || cut180 > l3A180) prose(rb, `<p class="fig-caption">${cut180 > l3A180 ? `At k = 180 the single IFN-γ cut-off scores a higher validation accuracy (${pc(cut180)}) than the three-measurement logistic model (${pc(l3A180)}). Accuracy at one threshold is coarse: one patient in Centre C is ${f1(100 / nV)} points, and the logistic model has the higher AUC. ` : ''}${l10k180 > l3k180 ? 'With all 180 patients the ten-measurement model performs as well as the three-measurement one, because the small penalty keeps the uninformative weights near zero.' : ''}</p>`);

  // ---- Step 5: unsupervised
  rb = rung(ch, 5, 'Without labels: finding patient subgroups',
    `<p>Unsupervised learning receives the measurements but no outcome. A common task is <b>clustering</b>: dividing samples into groups that are similar within and different between. <b>k-means</b> is the simplest method: place k centres, assign each patient to the nearest centre, move each centre to the mean of its patients, and repeat until nothing changes. Here it sees only the IFN-γ and CAF scores of all ${ALL.length} patients (standardised to mean 0, SD 1); neither response nor the pathology phenotype is given to it.</p>`);
  const fk = figure(rb, 'k-means on IFN-γ and CAF scores', 'Each dot is a patient, coloured by cluster; crosses are the cluster centres. Use "Step" to run one assign-and-move iteration, or "Run" to iterate until convergence. The table compares the clusters with the hidden pathology phenotype, which the algorithm never sees.');
  let K = 3, showPh = false, stopK = null;
  const Zk = standardize(ALL.map((p) => [p.ifng, p.caf])).Z;
  let cents = [], asg = null, iter = 0;
  slider(fk.controls, { label: 'clusters k', min: 2, max: 5, step: 1, value: K, onInput: (v) => { K = v; initK(); } });
  button(fk.controls, 'Step', () => { haltK(); stepK(); paintK(); });
  button(fk.controls, 'Run', () => runK(), true);
  button(fk.controls, 'Reset', () => initK());
  checkbox(fk.controls, { label: 'colour by hidden phenotype', checked: false, onChange: (c) => { showPh = c; paintK(); } });
  const sK = stats(fk.root);
  const rowK = el('div', { class: 'fig-row' }); fk.body.append(rowK);
  const ka = el('div'), kb = el('div', { style: 'min-width:300px' }); rowK.append(ka, kb);
  const FK = frame(ka, { width: 500, height: 340, x: [2.2, 10.2], y: [1.4, 7.4], xLabel: 'IFN-γ score', yLabel: 'CAF / TGF-β score' });
  const legK = el('div'); fk.body.append(legK);
  const muK = [mean(ALL.map((p) => p.ifng)), mean(ALL.map((p) => p.caf))], sdK = [sd(ALL.map((p) => p.ifng)), sd(ALL.map((p) => p.caf))];
  function initK() {
    haltK();
    const r = rng(11 + K);
    cents = [Zk[r.int(Zk.length)]];
    while (cents.length < K) { // k-means++ seeding
      const d2 = Zk.map((z) => d3.min(cents, (c) => (z[0] - c[0]) ** 2 + (z[1] - c[1]) ** 2)), tot = sum(d2);
      let u = r() * tot, i = 0; while (u > d2[i] && i < d2.length - 1) { u -= d2[i]; i++; }
      cents.push(Zk[i]);
    }
    cents = cents.map((c) => c.slice()); asg = null; iter = 0; paintK();
  }
  function stepK() {
    const na = Zk.map((z) => d3.minIndex(cents, (c) => (z[0] - c[0]) ** 2 + (z[1] - c[1]) ** 2));
    const changed = !asg || na.some((a, i) => a !== asg[i]);
    asg = na; iter++;
    cents = cents.map((c, j) => { const m = Zk.filter((_, i) => asg[i] === j); return m.length ? [mean(m.map((z) => z[0])), mean(m.map((z) => z[1]))] : c; });
    return changed;
  }
  function haltK() { if (stopK) { stopK(); stopK = null; } }
  function runK() { haltK(); let last = 0; stopK = animate((dt, t) => { if (t - last < 350) return true; last = t; const ch2 = stepK(); paintK(); return ch2 && iter < 50; }); }
  function paintK() {
    const colOf = (d, i) => (showPh ? cat(2 + PHEN.indexOf(d.phenotype)) : asg ? cat(asg[i] + 2) : muted());
    FK.layer('pts').selectAll('circle').data(ALL).join('circle').attr('cx', (d) => FK.x(d.ifng)).attr('cy', (d) => FK.y(d.caf)).attr('r', 3.8)
      .attr('fill', colOf).attr('stroke', css('--surface')).attr('stroke-width', 0.8)
      .on('mousemove', (e, d) => { const i = ALL.indexOf(d); tip(e, `${d.id} · ${d.phenotype}${asg ? `<br>cluster ${asg[i] + 1}` : ''}<br>IFN-γ ${d.ifng}, CAF ${d.caf}`); }).on('mouseleave', untip);
    FK.layer('c').selectAll('path').data(cents).join('path').attr('d', d3.symbol(d3.symbolCross, 220))
      .attr('transform', (c) => `translate(${FK.x(muK[0] + c[0] * sdK[0])},${FK.y(muK[1] + c[1] * sdK[1])})`).attr('fill', ink()).attr('stroke', css('--surface')).attr('stroke-width', 1.5);
    legK.innerHTML = '';
    legend(legK, showPh ? PHEN.map((p, i) => [p, cat(2 + i)]) : cents.map((c, j) => [`cluster ${j + 1}`, cat(j + 2)]).concat([['centre', ink()]]));
    kb.innerHTML = '';
    if (!asg) { kb.append(html('<p class="fig-caption">Press Step or Run to assign patients to clusters.</p>')); sK.set({ iterations: 0 }); return; }
    const tab = cents.map((c, j) => { const r = { c: `cluster ${j + 1}` }; PHEN.forEach((p) => { r[p] = ALL.filter((q, i) => asg[i] === j && q.phenotype === p).length; }); return r; });
    table(kb, tab, [['c', ''], ...PHEN.map((p) => [p, p])], { cls: 'compact' });
    const pur = sum(tab.map((r) => Math.max(...PHEN.map((p) => r[p])))) / ALL.length;
    const wss = sum(Zk.map((z, i) => (z[0] - cents[asg[i]][0]) ** 2 + (z[1] - cents[asg[i]][1]) ** 2));
    sK.set({ iterations: iter, 'within-cluster spread (sum of squares)': f1(wss), 'purity vs phenotype': pc(pur) });
  }
  initK();
  const km = kmeansFull(Zk, 3, rng(14)), km4 = kmeansFull(Zk, 4, rng(15));
  const pur3 = purity(km, ALL), pur4 = purity(km4, ALL);
  prose(rb, `<p>With k = 3 the clusters agree with the hidden phenotypes for <b>${pc(pur3)}</b> of patients (<b>purity</b>: the share of patients in their cluster's most common phenotype). The algorithm was not told that three phenotypes exist; k is a choice. With k = 4 the purity is ${pc(pur4)}, but one phenotype is split in two, and k-means returns k groups whether or not k real groups exist. The agreement here is high because the simulation plants three well-separated phenotypes on exactly these two scores; real tumours are less tidy, and clusters are <b>hypotheses</b> to be named and tested, for example by checking whether they differ in response. The same procedure, on thousands of genes, groups single cells into putative cell types.</p>`);

  // ---- Step 6: self-supervised
  rb = rung(ch, 6, 'Labels from the data itself: predicting a masked measurement',
    `<p>Self-supervised learning creates its own labels: hide part of each example and train a model to predict it from the rest. No outcome is needed, so any amount of unlabelled data can be used. Here one measurement is hidden and predicted by a linear model from the other five, trained on the discovery patients and checked on Centre C. Response is never used.</p>`);
  const MASK = { ifng: 'IFN-γ score', cd8: 'CD8 T cells', pdl1: 'PD-L1 TPS', caf: 'CAF / TGF-β score', log2tmb: 'TMB (log2)', prolif: 'Proliferation score' };
  const fitMask = (target) => {
    const others = Object.keys(MASK).filter((k) => k !== target);
    const st = standardize(D.map((p) => others.map((k) => get(k)(p)))), ty = D.map(get(target)), my = mean(ty);
    const w = ridge(st.Z, ty.map((v) => v - my), 0.5);
    const pred = (p) => my + st.apply(others.map((k) => get(k)(p))).reduce((s, v, j) => s + v * w[j], 0);
    const pv = V.map(pred), tv = V.map(get(target)), mv = mean(tv);
    const r2 = 1 - sum(pv.map((q, i) => (q - tv[i]) ** 2)) / sum(tv.map((t) => (t - mv) ** 2));
    return { others, w, pv, tv, r2 };
  };
  const R2 = Object.fromEntries(Object.keys(MASK).map((k) => [k, fitMask(k).r2]));
  const fm = figure(rb, 'Fill in the hidden measurement', 'Each dot is a Centre C patient: the measured value (x) against the value predicted from the other five measurements (y), coloured by the hidden pathology phenotype. Points on the diagonal are predicted exactly. Bars show the weight of each visible measurement (per SD).');
  let mt = 'ifng';
  select(fm.controls, { label: 'Hidden measurement', value: mt, options: Object.entries(MASK), onChange: (v) => { mt = v; paintM(); } });
  const sM = stats(fm.root);
  const rowM = el('div', { class: 'fig-row' }); fm.body.append(rowM);
  const ma = el('div'), mb = el('div'); rowM.append(ma, mb);
  const FM = frame(ma, { width: 480, height: 330, x: [0, 1], y: [0, 1], xLabel: 'measured', yLabel: 'predicted from the other five' });
  const FW = frame(mb, { width: 440, height: 330, x: [-1, 1], y: [0, 5], xLabel: 'weight per SD of the visible measurement', grid: false, axes: false, margin: { left: 130, bottom: 42 } });
  const axW = FW.g.append('g').attr('class', 'axis x').attr('transform', `translate(0,${FW.h})`);
  legend(fm.body, PHEN.map((p, i) => [p, cat(2 + i)]));
  function paintM() {
    const r = fitMask(mt), all = r.tv.concat(r.pv), lo = d3.min(all), hi = d3.max(all), pad = (hi - lo) * 0.05;
    FM.x.domain([lo - pad, hi + pad]); FM.y.domain([lo - pad, hi + pad]);
    FM.g.select('.axis.x').call(d3.axisBottom(FM.x).ticks(6)); FM.g.select('.axis.y').call(d3.axisLeft(FM.y).ticks(6));
    FM.g.select('.gridline').call(d3.axisLeft(FM.y).ticks(6).tickSize(-FM.w).tickFormat(''));
    FM.layer('diag').selectAll('line').data([0]).join('line').attr('x1', FM.x(lo - pad)).attr('y1', FM.y(lo - pad)).attr('x2', FM.x(hi + pad)).attr('y2', FM.y(hi + pad)).attr('stroke', muted()).attr('stroke-dasharray', '5 4');
    FM.layer('pts').selectAll('circle').data(V).join('circle').attr('cx', (d, i) => FM.x(r.tv[i])).attr('cy', (d, i) => FM.y(r.pv[i])).attr('r', 3.8)
      .attr('fill', (d) => cat(2 + PHEN.indexOf(d.phenotype))).attr('fill-opacity', 0.8).attr('stroke', css('--surface')).attr('stroke-width', 0.8)
      .on('mousemove', (e, d) => { const i = V.indexOf(d); tip(e, `${d.id} · ${d.phenotype}<br>${MASK[mt]}: measured ${f2(r.tv[i])}, predicted ${f2(r.pv[i])}`); }).on('mouseleave', untip);
    const ext = Math.max(0.5, d3.max(r.w, Math.abs)) * 1.15;
    FW.x.domain([-ext, ext]); axW.call(d3.axisBottom(FW.x).ticks(5));
    const yb = d3.scaleBand().domain(r.others).range([0, FW.h]).padding(0.3);
    FW.layer('z').selectAll('line').data([0]).join('line').attr('x1', FW.x(0)).attr('x2', FW.x(0)).attr('y1', 0).attr('y2', FW.h).attr('stroke', muted());
    FW.layer('b').selectAll('rect').data(r.w).join('rect').attr('y', (d, j) => yb(r.others[j])).attr('height', yb.bandwidth()).attr('x', (d) => FW.x(Math.min(0, d))).attr('width', (d) => Math.abs(FW.x(d) - FW.x(0))).attr('fill', (d) => (d >= 0 ? cat(1) : cat(0)))
      .on('mousemove', (e, d) => tip(e, `weight ${f2(d)} per SD`)).on('mouseleave', untip);
    FW.layer('l').selectAll('text').data(r.others).join('text').attr('x', -8).attr('y', (k) => yb(k) + yb.bandwidth() / 2 + 4).attr('text-anchor', 'end').attr('font-size', 12).attr('fill', ink2()).text((k) => MASK[k]);
    sM.set({ 'R² on Centre C': f2(r2c(r.r2)), 'labels used': 'none', 'patients used to learn': n });
  }
  paintM();
  const immune = ['ifng', 'cd8', 'pdl1'].map((k) => f2(r2c(R2[k]))).join(', ');
  prose(rb, `<p><b>R²</b> is the share of the variation in the hidden value that the prediction explains (1 = perfect, 0 = no better than the average). The immune measurements are largely recoverable from each other (R² ${immune} for IFN-γ, CD8 and PD-L1), because they share one underlying cause. TMB and proliferation are not (R² ${f2(r2c(R2.log2tmb))} and ${f2(r2c(R2.prolif))}): nothing else in the biopsy carries their information. The pretext task therefore uncovers the structure in the data, here the immune axis, without any outcome. <b>Foundation models</b> apply the same principle at scale: a large network learns to fill in masked genes across millions of cells, and its internal representation is then reused for a labelled task where only a few examples exist.</p>`);
  quiz(rb, 'Why can a self-supervised model use far more data than a supervised one?',
    '<p>It needs no labels. Measurements (expression profiles, sequences, images) are abundant; outcomes such as treatment response are expensive and few. A model pretrained on unlabelled data can then be adapted with a small labelled set, although whether this beats a simple baseline must be tested case by case.</p>');

  // =====================================================================================
  // 5. PITFALLS
  // =====================================================================================
  ch = chapter(main, 'pitfalls', 'Pitfalls', 'Common problems, how to recognise them, and how to address them.');
  const shift = byC.map((b) => `${b.c.slice(-1)}: ${pc(b.r)}`).join(', ');
  const pdlIfn = fitLogistic(D.map((p) => [p.ifng, p.pdl1]), y), pdlOnly = fitLogistic(D.map((p) => [p.pdl1]), y);
  cards(ch, [
    { tag: 'Symptom: near-perfect performance', title: 'Judging on the training data', body: `1-nearest neighbour on all discovery patients scores ${pc(nnT)} on its own examples and ${pc(nnV)} on Centre C. <b>Fix:</b> report performance only on data that played no part in fitting or in any choice made along the way.` },
    { tag: `Symptom: "${pc(1 - baseV)} accurate"`, title: 'No baseline', body: `Predicting "no response" for everyone is ${pc(1 - baseV)} accurate in Centre C. <b>Fix:</b> always report the trivial baseline, and use measures that are insensitive to class balance (AUC, sensitivity and specificity).` },
    { tag: 'Symptom: the best candidate disappoints', title: 'Selecting on the same data', body: `Among ${cutRows.length} single-measurement rules, cut-offs on unrelated measurements reach ${pc(cP.a)} (proliferation) and ${pc(cA.a)} (age) on discovery. <b>Fix:</b> choose models, features and thresholds inside cross-validation, and keep a final test set untouched.` },
    { tag: 'Symptom: a predictive feature with no role', title: 'Prediction is not mechanism', body: `PD-L1 alone predicts response (odds ratio per +10% TPS ${f2(Math.exp(10 * pdlOnly.w[0]))}), but next to IFN-γ its odds ratio is ${f2(Math.exp(10 * pdlIfn.w[1]))}. <b>Fix:</b> do not read a predictive model as a causal one; test mechanisms with designed experiments or causal methods.` },
    { tag: 'Symptom: worse in a new hospital', title: 'Distribution shift', body: `Response rates by centre (${shift}) and the mix of tumour phenotypes differ between hospitals. A rule tuned to one case mix can be miscalibrated in another. <b>Fix:</b> validate externally, by centre or batch, and recalibrate if necessary.` },
    { tag: 'Symptom: tidy clusters', title: 'Clusters taken as ground truth', body: `k-means returns k groups for any k: with k = 4 it splits a phenotype in two (purity ${pc(pur4)}). <b>Fix:</b> treat clusters as hypotheses; check their stability and test them against independent biology.` },
    { tag: 'Symptom: unstable results', title: 'Too many parameters for the data', body: `At k = 20 training patients the ten-measurement model reaches AUC ${f2(l10k20)}, against ${f2(l3k20)} for the three-measurement model. In omics, with thousands of features, the gap is much larger. <b>Fix:</b> fewer features chosen in advance, penalisation, and more samples.` },
    { tag: 'Symptom: excellent test score, fails in use', title: 'Leakage', body: 'Information from the test patients enters training: normalising or selecting genes on all samples before splitting, or the same patient appearing in both sets. <b>Fix:</b> split first, then fit every data-driven step on the training part only.' },
  ]);

  // =====================================================================================
  // 6. RECAP
  // =====================================================================================
  ch = chapter(main, 'recap', 'Recap', 'Key points.');
  recap(ch, [
    'Machine learning <b>learns a rule from examples</b>: choose a model family, a loss and an optimiser, and let the optimiser pick the parameters.',
    'A rule is judged by its performance on <b>new samples</b>. Training performance is optimistic, and a memorising rule can be perfect on its examples.',
    '<b>Supervised</b> learning uses labels (response); <b>unsupervised</b> learning finds structure without labels (patient subgroups, cell types); <b>self-supervised</b> learning hides part of the data and predicts it (foundation models).',
    '<b>Statistics</b> answers "is there an effect, and how sure are we?"; <b>ML</b> answers "what will happen for this new sample?". The same model can serve both.',
    'Performance rises with the number of examples and levels off at the <b>irreducible error</b> set by how much the measurements determine the outcome.',
    'More parameters need more examples. With few samples and many features, simple models or penalties perform better.',
    'Always compare with a trivial baseline, and validate on data from a new source (here, a new hospital).',
    'Predictive features are not necessarily causal, and clusters are hypotheses, not facts.',
  ]);
  prose(ch, '<h3>In code</h3><p>The core loop is the same in every library: fit on the training set, evaluate on a held-out set. The learning curve of step 4 and the clustering of step 5 are shown as well.</p>');
  code(ch, {
    Python: `import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, roc_auc_score

feats = ["ifng", "log2tmb", "caf"]
train = df[df.cohort == "discovery"];  test = df[df.cohort == "validation"]

# Supervised: learn on discovery, judge on the new centre
clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0))
clf.fit(train[feats], train["response"])
p = clf.predict_proba(test[feats])[:, 1]
print("accuracy", accuracy_score(test["response"], p >= 0.5), "AUC", roc_auc_score(test["response"], p))
print("baseline", 1 - test["response"].mean())

# Learning curve: k training patients, repeated draws
rng = np.random.default_rng(0)
for k in [10, 30, 90, 180]:
    aucs = []
    for _ in range(25):
        s = train.sample(k, random_state=rng.integers(1e9))
        if s["response"].nunique() < 2: continue
        aucs.append(roc_auc_score(test["response"], clf.fit(s[feats], s["response"]).predict_proba(test[feats])[:, 1]))
    print(k, np.mean(aucs))

# Unsupervised: k-means on standardised scores (no labels)
Z = StandardScaler().fit_transform(df[["ifng", "caf"]])
df["cluster"] = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(Z)`,
    R: `train <- subset(df, cohort == "discovery"); test <- subset(df, cohort == "validation")

# Supervised: learn on discovery, judge on the new centre
m <- glm(response ~ ifng + log2tmb + caf, family = binomial, data = train)
p <- predict(m, newdata = test, type = "response")
mean((p >= 0.5) == test$response)          # accuracy
pROC::auc(test$response, p)                 # AUC
1 - mean(test$response)                     # baseline

# Unsupervised: k-means on standardised scores (no labels)
set.seed(1)
km <- kmeans(scale(df[, c("ifng", "caf")]), centers = 3, nstart = 10)
table(km$cluster, df$phenotype)`,
  });
  seeAlso(main, [['concepts', 'train-test', 'Train/test split and overfitting'], ['concepts', 'logistic-regression', 'Logistic regression'], ['concepts', 'clustering', 'Clustering'], ['concepts', 'pca', 'Principal component analysis'], ['frontier', 'foundation-models', 'Single-cell foundation models'], ['omics', 'overview', 'Study overview and analysis map']]);
}

// ---------------------------------------------------------------------------------------------
const r2c = (v) => Math.max(0, v);
/** Candidate cut-offs: midpoints between consecutive distinct values. */
function candCuts(xs) {
  const u = [...new Set(xs)].sort((a, b) => a - b), out = [];
  for (let i = 1; i < u.length; i++) out.push((u[i - 1] + u[i]) / 2);
  return out;
}
/** Best single cut-off and direction by training accuracy. */
function bestCut(xs, y) {
  let best = { t: xs[0], dir: 1, acc: -1 };
  const cands = candCuts(xs);
  [1, -1].forEach((dir) => cands.forEach((t) => {
    let c = 0; xs.forEach((v, i) => { if (((dir > 0 ? v > t : v < t) ? 1 : 0) === y[i]) c++; });
    if (c / xs.length > best.acc + 1e-12) best = { t, dir, acc: c / xs.length };
  }));
  return best;
}
/** A seeded random order of indices whose first 4 contain both classes. */
function drawOrder(y, seed) {
  const o = rng(seed).shuffle(y.map((_, i) => i));
  for (let k = 1; k < 4; k++) if (new Set(o.slice(0, k + 1).map((i) => y[i])).size === 1) {
    const j = o.findIndex((i, m) => m > k && y[i] !== y[o[0]]);
    [o[k], o[j]] = [o[j], o[k]]; break;
  }
  return o;
}
function kmeansFull(Z, k, r) {
  let cents = [Z[r.int(Z.length)]];
  while (cents.length < k) {
    const d2 = Z.map((z) => d3.min(cents, (c) => (z[0] - c[0]) ** 2 + (z[1] - c[1]) ** 2)), tot = d2.reduce((s, v) => s + v, 0);
    let u = r() * tot, i = 0; while (u > d2[i] && i < d2.length - 1) { u -= d2[i]; i++; }
    cents.push(Z[i]);
  }
  let asg = null;
  for (let it = 0; it < 100; it++) {
    const na = Z.map((z) => d3.minIndex(cents, (c) => (z[0] - c[0]) ** 2 + (z[1] - c[1]) ** 2));
    if (asg && na.every((a, i) => a === asg[i])) break;
    asg = na;
    cents = cents.map((c, j) => { const m = Z.filter((_, i) => asg[i] === j); return m.length ? [mean(m.map((z) => z[0])), mean(m.map((z) => z[1]))] : c; });
  }
  return asg;
}
function purity(asg, pts) {
  const k = Math.max(...asg) + 1;
  let s = 0;
  for (let j = 0; j < k; j++) s += Math.max(...PHEN.map((p) => pts.filter((q, i) => asg[i] === j && q.phenotype === p).length));
  return s / pts.length;
}
function corr(a, b) {
  const ma = mean(a), mb = mean(b);
  let sab = 0, saa = 0, sbb = 0;
  a.forEach((v, i) => { sab += (v - ma) * (b[i] - mb); saa += (v - ma) ** 2; sbb += (b[i] - mb) ** 2; });
  return sab / Math.sqrt(saa * sbb);
}

/** Examples → learner → rule → prediction, for the learned IFN-γ cut-off and one Centre C patient. */
function pipelineSvg(n, nR, cut, P) {
  const wrap = el('div', { class: 'figure' });
  wrap.append(el('div', { class: 'fig-title' }, 'Learning a rule, then applying it to a new patient'));
  const svg = d3.select(wrap).append('svg').attr('viewBox', '0 0 980 210').attr('width', 980).attr('role', 'img').attr('aria-label', 'examples, learning algorithm, learned rule, prediction for a new patient');
  const op = cut.dir > 0 ? '>' : '<';
  const boxes = [
    { t: 'Examples (training)', lines: [`${n} discovery patients`, 'each with: IFN-γ score', `and outcome (${nR} responded)`, '', 'Centres A and B'] },
    { t: '① Learning algorithm', lines: ['try every cut-off t', `count patients with`, `"IFN-γ ${op} t" = responded`, 'keep the t with', 'the most correct'] },
    { t: '② Learned rule', lines: [`respond if`, `IFN-γ ${op} ${f2(cut.t)}`, '', `correct on ${pc(cut.acc)}`, 'of training patients'] },
    { t: '③ New patient', lines: [`${P.id}, Centre C`, `IFN-γ = ${f2(P.ifng)}`, `→ predict "${(cut.dir > 0 ? P.ifng > cut.t : P.ifng < cut.t) ? 'responds' : 'no response'}"`, '', `actually: ${P.response ? 'responded' : 'no response'}`] },
  ];
  const BW = 212, gap = 36, H = 180;
  boxes.forEach((b, i) => {
    const x0 = 10 + i * (BW + gap), g = svg.append('g').attr('transform', `translate(${x0},14)`);
    g.append('rect').attr('width', BW).attr('height', H).attr('rx', 10).attr('fill', css('--surface-2')).attr('stroke', css('--axis'));
    g.append('text').attr('x', 12).attr('y', 24).attr('font-weight', 700).attr('font-size', 13.5).attr('fill', ink()).text(b.t);
    b.lines.forEach((l, j) => g.append('text').attr('x', 14).attr('y', 54 + j * 24).attr('font-family', css('--mono')).attr('font-size', 13).attr('fill', ink2()).attr('xml:space', 'preserve').text(l));
    if (i < boxes.length - 1) {
      const ax = x0 + BW + 6, ay = 14 + H / 2;
      svg.append('path').attr('d', `M${ax},${ay} L${ax + gap - 14},${ay} M${ax + gap - 20},${ay - 6} L${ax + gap - 14},${ay} L${ax + gap - 20},${ay + 6}`).attr('fill', 'none').attr('stroke', ink2()).attr('stroke-width', 2);
    }
  });
  return wrap;
}
