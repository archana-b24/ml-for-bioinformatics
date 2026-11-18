// Regularisation (ridge, lasso, elastic net), lesson format.
// Arc: what → why → where → how it works (6 steps: standardise → geometry → coefficient paths → choose λ by CV →
//      lasso selection and stability → lab with try/see/means table) → pitfalls → recap.
// Data: the study cohort (10 measurements) plus 120 simulated noise genes (data/study/regularization.json, scripts/build_regularization.py).
// Every number in the prose is computed at render time; penalised fits use fitPenalised() from js/lib/models.js.
import { loadJSON, header, figure, slider, select, button, checkbox, stats, legend, seeAlso, tip, untip, el, html } from '../lib/ui.js';
import { frame, cat, css, ink, ink2, muted } from '../lib/plot.js';
import { rng, mean, sd } from '../lib/data.js';
import { standardize, fitPenalised, lipschitz, lambdaMax, auc, logLoss, predictProb, fitLogistic } from '../lib/models.js';
import { loadCohort, FEATURES, studyCard } from '../lib/study.js';
import { lessonNav, chapter, prose, rung, worked, quiz, cards, table, code, recap } from '../lib/lesson.js';
const d3 = window.d3;

const um = (t) => t.replace(/^-/, '−');
const f2 = (v) => (Math.abs(v) < 0.005 ? '0.00' : um(v.toFixed(2)));
const f3 = (v) => um(v.toFixed(3));
const pc = (v) => Math.round(100 * v) + '%';
const sum = (a) => a.reduce((s, v) => s + v, 0);
const lg = (v) => Math.log10(v);
const lamTxt = (v) => (v >= 0.1 ? v.toFixed(2) : v >= 0.01 ? v.toFixed(3) : v.toPrecision(2));
const NZ = 1e-8;

const LAB = ['ifng', 'log2tmb', 'caf', 'pdl1', 'cd8', 'prolif', 'age', 'smoker', 'male', 'centreB'];
const PEN = { ridge: { a: 0, label: 'Ridge (L2)' }, lasso: { a: 1, label: 'Lasso (L1)' }, enet: { a: 0.5, label: 'Elastic net (α = 0.5)' } };
const K = 28, GRID = d3.range(K).map((k) => 10 ** (-3 + 2.8 * k / (K - 1)));   // ascending λ
const NN = 40;                                                                  // noise genes in the main analysis

export default async function render(main) {
  const C = await loadCohort();
  const R = await loadJSON('data/study/regularization.json');
  const D = C.discovery, V = C.validation, n = D.length;
  const y = D.map((p) => p.response), yv = V.map((p) => p.response), nR = sum(y);
  const row = Object.fromEntries(R.ids.map((id, i) => [id, i]));
  const nameOf = (j) => (j < LAB.length ? FEATURES[LAB[j]].label : R.genes[j - LAB.length]);
  const kindOf = (j) => (j < 3 ? 'true' : j < LAB.length ? 'other' : 'noise');
  const colOf = (j) => (j < 3 ? cat(1 + j) : j < LAB.length ? cat(0) : muted());
  const rowsOf = (P, N) => P.map((p) => [...LAB.map((k) => FEATURES[k].get(p)), ...R.X[row[p.id]].slice(0, N)]);

  // ---------- fitting helpers ----------
  const cacheM = {};
  const mk = (N) => cacheM[N] || (cacheM[N] = (() => { const st = standardize(rowsOf(D, N)); return { st, Z: st.Z, Zv: rowsOf(V, N).map(st.apply), L: lipschitz(st.Z) }; })());
  const probs = (Z, f) => Z.map((r) => predictProb(r, f.b0, f.w));
  const evalF = (f, Z, yy) => { const p = probs(Z, f); return { auc: auc(p, yy), ll: logLoss(p, yy) }; };
  const nzOf = (f) => f.w.filter((w) => Math.abs(w) > NZ).length;
  function runPath(Z, yy, alpha, L, iters = 200) {
    const out = new Array(K); let init = null;
    for (let k = K - 1; k >= 0; k--) { init = fitPenalised(Z, yy, { lambda: GRID[k], alpha, init, iters, L }); out[k] = init; }
    return out;
  }
  const cachePath = {};
  const pathOf = (pen, N) => { const key = pen + N; if (!cachePath[key]) { const m = mk(N); const pth = runPath(m.Z, y, PEN[pen].a, m.L); cachePath[key] = pth.map((f) => ({ ...f, tr: evalF(f, m.Z, y), va: evalF(f, m.Zv, yv), nz: nzOf(f) })); } return cachePath[key]; };
  function cvRun(N, alpha, folds = 5) {
    const r = rng(11), fold = new Array(n), raw = rowsOf(D, N);
    [0, 1].forEach((c) => r.shuffle(d3.range(n).filter((i) => y[i] === c)).forEach((i, k) => { fold[i] = k % folds; }));
    const ll = [], au = [];
    for (let f = 0; f < folds; f++) {
      const tr = d3.range(n).filter((i) => fold[i] !== f), te = d3.range(n).filter((i) => fold[i] === f);
      const st = standardize(tr.map((i) => raw[i])), Zt = te.map((i) => st.apply(raw[i])), yt = te.map((i) => y[i]), yT = tr.map((i) => y[i]);
      const pth = runPath(st.Z, yT, alpha, lipschitz(st.Z), 100);
      ll.push(pth.map((q) => logLoss(probs(Zt, q), yt))); au.push(pth.map((q) => auc(probs(Zt, q), yt)));
    }
    const stat = (M) => GRID.map((_, k) => { const v = M.map((m) => m[k]); return { m: mean(v), se: sd(v) / Math.sqrt(folds) }; });
    const L = stat(ll), A = stat(au);
    const iMin = d3.minIndex(L, (q) => q.m); let i1 = iMin;
    for (let k = iMin; k < K; k++) if (L[k].m <= L[iMin].m + L[iMin].se) i1 = k;
    return { ll: L, au: A, iMin, i1se: i1 };
  }
  const P40 = { ridge: pathOf('ridge', NN), lasso: pathOf('lasso', NN), enet: pathOf('enet', NN) };
  const CV = { ridge: cvRun(NN, 0), lasso: cvRun(NN, 1), enet: cvRun(NN, 0.5) };
  const lamL = GRID[CV.lasso.i1se], lamR = GRID[CV.ridge.iMin];
  const fL = P40.lasso[CV.lasso.i1se], fR = P40.ridge[CV.ridge.iMin];
  const selL = fL.w.map((w, j) => (Math.abs(w) > NZ ? j : -1)).filter((j) => j >= 0);
  const selNoise = selL.filter((j) => kindOf(j) === 'noise');
  const m40 = mk(NN);
  const stT = standardize(rowsOf(D, 0).map((r) => r.slice(0, 3)));
  const fOr = fitPenalised(stT.Z, y, { lambda: 1e-6, alpha: 0 });
  const oracleV = evalF(fOr, V.map((p) => stT.apply([p.ifng, p.log2tmb, p.caf])), yv);
  const base = nR / n;

  header(main, 'Regularisation (ridge and lasso)',
    'When there are many candidate measurements and few patients, an ordinary fit chases noise. Regularisation adds a <b>penalty on the size of the coefficients</b>, which shrinks them towards zero (ridge), sets some exactly to zero (lasso), or does both (elastic net). The penalty strength is chosen by cross-validation.');
  lessonNav(main);
  studyCard(main, C, `<b>Question for this page:</b> if the ${LAB.length} study measurements are joined by tens of simulated genes that have <i>no</i> relation to response, can a model still predict response in the ${V.length} validation patients, and which measurements does it keep?`);

  // =====================================================================================
  // 1. WHAT
  // =====================================================================================
  let ch = chapter(main, 'what', 'What is regularisation?', 'In one line: <b>fit the data, but pay a price for large coefficients</b>.');
  prose(ch, `<p>A regression model chooses coefficients to minimise a <b>loss</b> (for response, the log-loss of logistic regression). <b>Regularisation</b> (also called penalisation or shrinkage) minimises the loss <i>plus a penalty</i> that grows with the size of the coefficients. The weight on the penalty is a number λ ≥ 0 that is chosen, not estimated by the fit itself.</p>
    <div class="calc">minimise   mean log-loss  +  λ × [ α · Σ|w<sub>j</sub>|   +   (1 − α)/2 · Σ w<sub>j</sub>² ]
                                       lasso part      ridge part</div>
    <p>Three standard choices follow from the same formula (this is the glmnet convention; scikit-learn parameterises it differently, see the code in chapter 6):</p>
    <ul><li><b>Ridge</b> (α = 0, L2 penalty): all coefficients shrink smoothly towards 0 and are almost never exactly 0. It copes well with correlated measurements.</li>
    <li><b>Lasso</b> (α = 1, L1 penalty): coefficients shrink and many become <b>exactly 0</b>, so the model also performs feature selection.</li>
    <li><b>Elastic net</b> (0 &lt; α &lt; 1): a mixture; it can select features, but tends to keep correlated ones together.</li></ul>`);
  const fw = figure(ch, 'What the two penalties do to one coefficient',
    'Left: the price paid for a coefficient of a given size (λ × penalty). Right: the coefficient the penalised fit returns, against what the unpenalised fit would return. The right panel is exact for a simple case (one feature, squared-error loss, standardised); it shows the two behaviours in their purest form.');
  let lamW = 0.8;
  slider(fw.controls, { label: 'penalty strength λ', min: 0, max: 2, step: 0.05, value: lamW, fmt: f2, onInput: (v) => { lamW = v; paintW(); } });
  const sw = stats(fw.root);
  const rowW = el('div', { class: 'fig-row' }); fw.body.append(rowW);
  const wa = el('div'), wb = el('div'); rowW.append(wa, wb);
  const WA = frame(wa, { width: 440, height: 300, x: [-3, 3], y: [0, 6], xLabel: 'coefficient w', yLabel: 'penalty paid' });
  const WB = frame(wb, { width: 440, height: 300, x: [-3, 3], y: [-3, 3], xLabel: 'unpenalised estimate', yLabel: 'penalised estimate' });
  const clipId = 'reg-wa-clip';
  WB.layer('dead');
  WA.g.append('clipPath').attr('id', clipId).append('rect').attr('width', WA.w).attr('height', WA.h);
  const WAl = WA.layer('c').attr('clip-path', `url(#${clipId})`), WBl = WB.layer('c');
  WB.layer('id').append('line').attr('x1', WB.x(-3)).attr('y1', WB.y(-3)).attr('x2', WB.x(3)).attr('y2', WB.y(3)).attr('stroke', muted()).attr('stroke-dasharray', '4 4');
  WB.layer('id').append('text').attr('x', WB.x(1.5)).attr('y', WB.y(2.15)).attr('font-size', 11.5).attr('fill', muted()).attr('transform', `rotate(-45 ${WB.x(1.5)} ${WB.y(2.15)})`).text('no penalty (y = x)');
  const WPEN = [['Ridge', cat(2), (w, l) => l * w * w / 2, (w, l) => w / (1 + l)], ['Lasso', cat(6), (w, l) => l * Math.abs(w), (w, l) => Math.sign(w) * Math.max(0, Math.abs(w) - l)], ['Elastic net (α = 0.5)', cat(3), (w, l) => l * (0.5 * Math.abs(w) + 0.25 * w * w), (w, l) => Math.sign(w) * Math.max(0, Math.abs(w) - 0.5 * l) / (1 + 0.5 * l)]];
  const gx = d3.range(-3, 3.001, 0.05);
  const dead = WB.layer('dead').append('rect').attr('y', 0).attr('height', WB.h).attr('fill', css('--surface-2')).attr('opacity', 0.9);
  const deadT = WB.layer('dead').append('text').attr('font-size', 11.5).attr('fill', ink2()).attr('text-anchor', 'middle').attr('y', 16).text('lasso: exactly 0');
  function paintW() {
    WAl.selectAll('path').data(WPEN).join('path').attr('fill', 'none').attr('stroke', (d) => d[1]).attr('stroke-width', 2.2).attr('d', (d) => d3.line().x((w) => WA.x(w)).y((w) => WA.y(d[2](w, lamW)))(gx));
    WBl.selectAll('path').data(WPEN).join('path').attr('fill', 'none').attr('stroke', (d) => d[1]).attr('stroke-width', 2.2).attr('d', (d) => d3.line().x((w) => WB.x(w)).y((w) => WB.y(d[3](w, lamW)))(gx));
    dead.attr('x', WB.x(-lamW)).attr('width', Math.max(0, WB.x(lamW) - WB.x(-lamW)));
    deadT.attr('x', WB.x(0)).attr('opacity', lamW > 0.35 ? 1 : 0);
    sw.set({ 'ridge: estimate 2.0 becomes': f2(WPEN[0][3](2, lamW)), 'lasso: 2.0 becomes': f2(WPEN[1][3](2, lamW)), 'elastic net: 2.0 becomes': f2(WPEN[2][3](2, lamW)), 'lasso: 0.5 becomes': f2(WPEN[1][3](0.5, lamW)) });
  }
  paintW();
  legend(fw.body, [['ridge', cat(2)], ['lasso', cat(6)], ['elastic net', cat(3)]]);
  prose(ch, `<p>Ridge divides every estimate by the same factor (1 + λ), so large and small coefficients shrink proportionally and none reaches zero. Lasso subtracts the same amount λ from every estimate and sets it to 0 if that would change its sign, which creates the grey zone in the right panel. Key terms:</p>`);
  table(ch, [
    { t: 'Penalty strength (λ)', m: 'Weight on the penalty. λ = 0 reproduces the ordinary fit; a very large λ gives all-zero coefficients', s: `smallest λ at which the lasso is all-zero here: ${f3(lambdaMax(m40.Z, y))}` },
    { t: 'Mixing parameter (α)', m: 'Share of the L1 (lasso) part: 0 = ridge, 1 = lasso', s: '0, 0.5 and 1 are used on this page' },
    { t: 'Shrinkage', m: 'Pulling coefficients towards 0; it adds a little bias and removes variance', s: `at the cross-validated λ the IFN-γ weight is ${f2(P40.lasso[CV.lasso.i1se].w[0])} per SD (lasso)` },
    { t: 'Sparse model', m: 'A model in which most coefficients are exactly 0', s: `${selL.length} of ${m40.Z[0].length} features non-zero at the 1-SE lasso λ` },
    { t: 'Standardisation', m: 'Rescaling each feature to mean 0 and SD 1 before fitting, so the penalty treats all features alike', s: 'required for the penalty to be meaningful (step 1)' },
    { t: 'Cross-validation (CV)', m: 'Repeatedly holding out part of the patients to estimate how well each λ predicts new patients', s: 'step 4' },
  ], [['t', 'Term'], ['m', 'Meaning'], ['s', 'In this study']]);

  // =====================================================================================
  // 2. WHY
  // =====================================================================================
  ch = chapter(main, 'why', 'Why do we need it?', 'An ordinary fit uses every feature it is given, including those that carry no information.');
  prose(ch, `<p>The study has ${n} discovery patients, of whom ${nR} responded. Response is driven by three measurements (IFN-γ, TMB and CAF score); the other ${LAB.length - 3} measurements are proxies or unrelated. Real expression studies add thousands of genes. To mimic this, <b>${R.genes.length} simulated genes</b> are available on this page. They are generated at random, in blocks of ten mildly correlated genes, and have no relation to response, so any weight a model gives them is noise.</p>
    <p>The ordinary (maximum-likelihood) fit has two problems when features are numerous relative to patients:</p>
    <ol><li><b>Overfitting.</b> With enough features the model can fit the training patients almost perfectly by using noise, and then predicts new patients poorly.</li>
    <li><b>Instability or failure.</b> Correlated features give coefficients that swing between fits, and once the number of features approaches the number of patients the fit may have no unique solution.</li></ol>
    <p>The figure adds noise genes one block at a time and evaluates each model on the ${V.length} patients of Centre C, which were not used for fitting.</p>`);
  const NS = [0, 10, 20, 40, 80, 120];
  const WH = NS.map((N) => {
    const m = mk(N), fU = fitPenalised(m.Z, y, { lambda: 1e-4, alpha: 0 }), fRi = fitPenalised(m.Z, y, { lambda: lamR, alpha: 0 }), fLa = fitPenalised(m.Z, y, { lambda: lamL, alpha: 1, iters: 250, L: m.L });
    return { N, U: [evalF(fU, m.Z, y), evalF(fU, m.Zv, yv)], R: evalF(fRi, m.Zv, yv), L: evalF(fLa, m.Zv, yv), Lk: fLa.w.filter((w, j) => j >= LAB.length && Math.abs(w) > NZ).length, Lt: nzOf(fLa) };
  });
  const fy = figure(ch, 'Validation performance as noise genes are added',
    `Each point refits the model on the ${n} discovery patients with the ${LAB.length} study measurements plus the number of noise genes shown, then evaluates on Centre C. The penalties use the λ chosen by cross-validation in step 4 (with ${NN} noise genes) and are not re-tuned for each point. "No penalty" is maximum likelihood with a negligible λ of 0.0001, which only keeps the fit finite.`);
  let metric = 'auc', showTr = true;
  select(fy.controls, { label: 'Metric', value: metric, options: [['auc', 'AUC (higher is better)'], ['ll', 'Log-loss (lower is better)']], onChange: (v) => { metric = v; paintY(); } });
  checkbox(fy.controls, { label: 'show training curve', checked: true, onChange: (c) => { showTr = c; paintY(); } });
  const sy = stats(fy.root);
  const FY = frame(fy.body, { width: 820, height: 340, x: [0, 120], y: [0.4, 1], xLabel: 'noise genes added to the 10 study measurements', yLabel: 'AUC', xTicks: 6 });
  const LY = FY.layer('c');
  legend(fy.body, [['no penalty, validation', cat(0)], ['ridge, validation', cat(2)], ['lasso, validation', cat(6)], ['no penalty, training', ink2()], ['3 planted features, no penalty', muted()]]);
  function paintY() {
    const key = metric, get = (o) => o[key];
    const ser = [{ nm: 'no penalty, validation', c: cat(0), v: WH.map((q) => get(q.U[1])) }, { nm: 'ridge, validation', c: cat(2), v: WH.map((q) => get(q.R)) }, { nm: 'lasso, validation', c: cat(6), v: WH.map((q) => get(q.L)) }];
    if (showTr) ser.push({ nm: 'no penalty, training', c: ink2(), dash: '5 4', v: WH.map((q) => get(q.U[0])) });
    const all = ser.flatMap((s) => s.v).concat(get(oracleV));
    const lo = metric === 'auc' ? Math.min(0.5, d3.min(all) - 0.02) : 0, hi = metric === 'auc' ? 1.0 : Math.min(3, Math.max(0.8, d3.max(all) * 1.05));
    FY.y.domain([lo, hi]);
    FY.g.select('.axis.y').call(d3.axisLeft(FY.y).ticks(5)); FY.g.select('.gridline').call(d3.axisLeft(FY.y).ticks(5).tickSize(-FY.w).tickFormat(''));
    FY.g.select('.ax-label[transform]').text(metric === 'auc' ? 'AUC' : 'log-loss');
    LY.selectAll('*').remove();
    LY.append('line').attr('x1', 0).attr('x2', FY.w).attr('y1', FY.y(get(oracleV))).attr('y2', FY.y(get(oracleV))).attr('stroke', muted()).attr('stroke-dasharray', '2 4').attr('stroke-width', 2);
    LY.append('text').attr('x', FY.w - 4).attr('y', FY.y(get(oracleV)) + (metric === 'auc' ? -6 : 14)).attr('text-anchor', 'end').attr('font-size', 11.5).attr('fill', muted()).text('reference: only the 3 planted features');
    ser.forEach((s) => {
      LY.append('path').datum(s.v).attr('fill', 'none').attr('stroke', s.c).attr('stroke-width', 2.2).attr('stroke-dasharray', s.dash || null).attr('d', d3.line().x((v, i) => FY.x(NS[i])).y((v) => FY.y(Math.min(hi, Math.max(lo, v)))));
      LY.selectAll(null).data(s.v).join('circle').attr('cx', (v, i) => FY.x(NS[i])).attr('cy', (v) => FY.y(Math.min(hi, Math.max(lo, v)))).attr('r', 4.5).attr('fill', s.c).attr('stroke', css('--surface')).attr('stroke-width', 1.5)
        .on('mousemove', (e, v) => tip(e, `${s.nm}<br>${NS[s.v.indexOf(v)]} noise genes<br>${metric === 'auc' ? 'AUC' : 'log-loss'} ${f3(v)}`)).on('mouseleave', untip);
    });
    const last = WH[WH.length - 1];
    sy.set({ 'features / patients (max)': `${LAB.length + last.N} / ${n}`, 'no penalty: training → validation': `${f3(get(last.U[0]))} → ${f3(get(last.U[1]))}`, 'ridge, validation': f3(get(last.R)), 'lasso, validation': f3(get(last.L)) });
  }
  paintY();
  const w0 = WH[0], wl = WH[WH.length - 1];
  prose(ch, `<p>With no noise genes the unpenalised model reaches a validation AUC of <b>${f3(w0.U[1].auc)}</b>. With ${wl.N} noise genes its training AUC rises to <b>${f3(wl.U[0].auc)}</b> while the validation AUC falls to <b>${f3(wl.U[1].auc)}</b>, and its validation log-loss (${f2(wl.U[1].ll)}) is worse than predicting the base rate for everyone (${f2(logLoss(yv.map(() => base), yv))}). Under the penalties the validation AUC at ${wl.N} noise genes is ${f3(wl.R.auc)} (ridge) and ${f3(wl.L.auc)} (lasso). The reference line is a model that was told which three features matter (${f3(oracleV.auc)}). ${WH.every((q) => q.Lk === 0) ? `The lasso line is flat because at this λ it keeps ${nzOf(fL)} features and no noise gene enters; its AUC` : `The lasso keeps up to ${d3.max(WH, (q) => q.Lk)} noise genes at this λ; its AUC`} is below the unpenalised value with no noise genes (${f3(w0.L.auc)} against ${f3(w0.U[1].auc)}), so the penalty costs some signal when there is nothing to remove. Regularisation does not remove the cost of noise features, but it limits it.</p>`);
  const sdIdx = (j) => sd(m40.st.Z.map((r) => r[j]));
  worked(ch, 'Why more features than patients is a problem', `<p>With ${n} patients, a model with ${n - 1} or more free coefficients can reproduce any pattern of outcomes in the training set, including pure noise, because there is a combination of weights for every possible labelling. Fitting then has no unique answer. The penalty removes the freedom: among all coefficient sets that fit the data about equally well, it prefers the one with the smallest total size. Even when the number of features is below ${n}, the rule of thumb of 10–20 responders per feature (here ${nR} responders, so ${Math.floor(nR / 20)}–${Math.floor(nR / 10)} features) shows that the study is already near the limit with ${LAB.length} measurements.</p>`);
  table(ch, [
    { a: 'Select features by hand (domain knowledge)', s: 'yes', p: 'depends', i: 'high', d: 'Best when the biology is known; does not scale to thousands of genes' },
    { a: 'Univariate filter (top genes by p-value), then an ordinary fit', s: 'yes', p: 'no', i: 'medium', d: 'Selection must be repeated inside every cross-validation fold or the performance is inflated' },
    { a: '<b>Ridge</b>', s: '<b>no (all kept)</b>', p: '<b>yes</b>', i: '<b>medium</b>', d: 'Good for many weak, correlated features' },
    { a: '<b>Lasso</b>', s: '<b>yes</b>', p: '<b>yes</b>', i: '<b>high</b>', d: 'Selection can be unstable (step 5)' },
    { a: '<b>Elastic net</b>', s: '<b>yes</b>', p: '<b>yes</b>', i: '<b>medium–high</b>', d: 'Usual default for correlated omics features' },
  ], [['a', 'Approach'], ['s', 'Selects features?'], ['p', 'Handles features > patients?'], ['i', 'Interpretability'], ['d', 'Comment']]);

  // =====================================================================================
  // 3. WHERE
  // =====================================================================================
  ch = chapter(main, 'where', 'Where is it used?', 'Whenever there are many candidate predictors relative to the number of samples, which is the usual situation in omics.');
  cards(ch, [
    { tag: 'Expression biomarkers', title: 'Which genes predict response?', body: 'Penalised logistic or Cox regression (glmnet, scikit-learn) on thousands of genes yields a compact signature and a score for each patient, as in the lab of step 6.' },
    { tag: 'Epigenetics', title: 'DNA-methylation clocks', body: 'Horvath\'s epigenetic age clock is an elastic-net regression of age on several hundred of the tens of thousands of CpG sites measured on array platforms.' },
    { tag: 'Genetics', title: 'Polygenic risk scores', body: 'Millions of variants are available for thousands of people, so effect sizes are shrunk. Lasso-based (lassosum) and Bayesian shrinkage methods (LDpred, PRS-CS) belong to the same family of ideas.' },
    { tag: 'Transcriptome imputation', title: 'Predicting expression from genotype', body: 'PrediXcan fits an elastic net per gene on nearby variants and uses it to impute expression in genotyped cohorts.' },
    { tag: 'Survival', title: 'Penalised Cox regression', body: 'Prognostic signatures from many genes use the same penalties inside the Cox model (glmnet <code>family = "cox"</code>).' },
    { tag: 'Single-cell', title: 'Cell-type classifiers', body: 'Logistic-regression classifiers on expression (for example CellTypist) are trained with a penalty so that thousands of genes can be used with a limited number of labelled cells.' },
    { tag: 'Related shrinkage', title: 'Differential expression', body: 'DESeq2 can shrink log fold changes, and limma/edgeR moderate gene variances. These are not ridge or lasso, but they apply the same principle: borrow strength to stabilise noisy estimates.' },
    { tag: 'Deep learning', title: 'Weight decay', body: 'The "weight decay" setting of neural-network optimisers is an L2 penalty on the weights, that is, ridge regularisation.' },
    { tag: 'Proteomics, metabolomics', title: 'Panels from many analytes', body: 'Small diagnostic panels are often derived with lasso or elastic net from hundreds of measured proteins or metabolites in a few dozen to a few hundred samples.' },
  ]);
  prose(ch, `<p><b>When not to use it:</b> when estimates and p-values are the goal, because penalised coefficients are biased towards zero and standard confidence intervals do not apply; when the features are few and the sample is large (an ordinary fit is adequate); when relationships are strongly non-linear (tree ensembles or neural networks are more suitable, possibly with their own regularisation); when features come in natural groups that should be kept or dropped together (group lasso); and when a stable, reproducible list of "the" biomarkers is required, because lasso selection among correlated features is unstable (step 5).</p>`);

  // =====================================================================================
  // 4. HOW
  // =====================================================================================
  ch = chapter(main, 'how', 'How it works', 'Six steps: scaling, the geometry of the penalties, coefficient paths, choosing λ, selection stability, and a lab on the study.');

  // ---- Step 1: standardisation
  const raw10 = rowsOf(D, 0), sd10 = LAB.map((_, j) => sd(raw10.map((r) => r[j]))), st10 = standardize(raw10);
  let rb = rung(ch, 1, 'Standardise first: the penalty depends on units',
    `<p>The penalty adds up the sizes of the coefficients, but a coefficient is expressed <i>per unit of its feature</i>: per percentage point of PD-L1 (SD ${sd10[3].toFixed(0)}), per year of age (SD ${sd10[6].toFixed(0)}), per log2 unit of TMB (SD ${f2(sd10[1])}). A feature in large units needs only a small coefficient, so an unscaled penalty favours it. Features are therefore <b>standardised</b> (mean 0, SD 1) before fitting. The figure fits ridge to the ${LAB.length} study measurements both ways at the same λ, with weights shown per SD.</p>`);
  const f1 = figure(rb, 'The same λ, with and without standardising', 'Dots: ridge weights per SD of each feature. Diamonds: the unpenalised fit. Raw-unit coefficients were multiplied by the feature SD to put them on the same scale.');
  const unp10 = fitPenalised(st10.Z, y, { lambda: 1e-6, alpha: 0 });
  let s1 = 12;
  slider(f1.controls, { label: 'penalty strength λ', min: 0, max: K - 1, step: 1, value: s1, fmt: (i) => lamTxt(GRID[i]), onInput: (v) => { s1 = v; paint1(); } });
  const ss1 = stats(f1.root);
  const F1 = frame(f1.body, { width: 820, height: 330, x: [-0.6, 1.5], y: [0, LAB.length], xLabel: 'weight per SD of the feature (log-odds)', grid: false, axes: false, margin: { left: 190, top: 8, right: 20 } });
  F1.g.append('g').attr('class', 'axis x').attr('transform', `translate(0,${F1.h})`).call(d3.axisBottom(F1.x).ticks(8));
  F1.layer('z').append('line').attr('x1', F1.x(0)).attr('x2', F1.x(0)).attr('y1', 0).attr('y2', F1.h).attr('stroke', muted()).attr('stroke-dasharray', '4 4');
  const yb1 = (j) => (j + 0.5) * F1.h / LAB.length;
  LAB.forEach((k, j) => {
    F1.layer('lab').append('line').attr('x1', 0).attr('x2', F1.w).attr('y1', yb1(j)).attr('y2', yb1(j)).attr('stroke', css('--grid') || muted()).attr('stroke-opacity', 0.35);
    F1.layer('lab').append('text').attr('x', -8).attr('y', yb1(j) + 4).attr('text-anchor', 'end').attr('font-size', 12).attr('fill', ink2()).text(`${FEATURES[k].label} (SD ${sd10[j] >= 10 ? sd10[j].toFixed(0) : f2(sd10[j])})`);
  });
  const L1 = F1.layer('d');
  legend(f1.body, [['standardised, ridge', cat(2)], ['raw units, ridge', cat(6)], ['unpenalised', ink()]]);
  const fitRaw = (lam) => fitPenalised(raw10, y, { lambda: lam, alpha: 0 });
  const rawUnp = fitRaw(1e-6);
  function paint1() {
    const lam = GRID[s1], fs = fitPenalised(st10.Z, y, { lambda: lam, alpha: 0 }), fr = fitRaw(lam);
    const pts = [];
    LAB.forEach((k, j) => { pts.push({ j, s: 'u', v: unp10.w[j], t: 'unpenalised' }, { j, s: 's', v: fs.w[j], t: 'standardised, ridge' }, { j, s: 'r', v: fr.w[j] * sd10[j], t: 'raw units, ridge' }); });
    const sym = { u: d3.symbolDiamond, s: d3.symbolCircle, r: d3.symbolSquare }, col = { u: ink(), s: cat(2), r: cat(6) };
    L1.selectAll('path').data(pts).join('path').attr('d', (d) => d3.symbol(sym[d.s], d.s === 'u' ? 80 : 62)()).attr('transform', (d) => `translate(${F1.x(Math.max(-0.6, Math.min(1.5, d.v)))},${yb1(d.j) + (d.s === 's' ? -5 : d.s === 'r' ? 5 : 0)})`)
      .attr('fill', (d) => (d.s === 'u' ? 'none' : col[d.s])).attr('stroke', (d) => (d.s === 'u' ? ink() : css('--surface'))).attr('stroke-width', (d) => (d.s === 'u' ? 1.6 : 1.2))
      .on('mousemove', (e, d) => tip(e, `${FEATURES[LAB[d.j]].label}<br>${d.t}: ${f2(d.v)} per SD`)).on('mouseleave', untip);
    const kp = (w, j) => pc(w / unp10.w[j]);
    ss1.set({ λ: lamTxt(lam), 'IFN-γ weight kept (standardised)': kp(fs.w[0], 0), 'IFN-γ weight kept (raw units)': kp(fr.w[0] * sd10[0], 0), 'Ever smoker weight kept (standardised)': kp(fs.w[7], 7), 'Ever smoker weight kept (raw units)': kp(fr.w[7] * sd10[7], 7) });
  }
  paint1();
  { const lam = GRID[12], fs = fitPenalised(st10.Z, y, { lambda: lam, alpha: 0 }), fr = fitRaw(lam), r = (w, j) => Math.abs(w) / Math.abs(unp10.w[j]);
    prose(rb, `<p>At λ = ${lamTxt(lam)} the standardised fit keeps ${pc(r(fs.w[0], 0))} of the unpenalised IFN-γ weight, whereas the raw-unit fit keeps ${pc(r(fr.w[0] * sd10[0], 0))}: the penalty depends on the unit. For the binary feature "ever smoker" (SD ${f2(sd10[7])}) the order is reversed: ${pc(r(fs.w[7], 7))} is kept after standardising and ${pc(r(fr.w[7] * sd10[7], 7))} in raw units. Ridge also spreads weight across correlated features: the PD-L1 weight is ${f2(fs.w[3])} per SD against ${f2(unp10.w[3])} unpenalised. After standardising, the <b>penalty reflects only the strength of the signal</b>. The scaling is learned on the training patients and applied to new ones (chapter 6 code); glmnet standardises internally by default and reports coefficients in the original units.</p>`); }

  // ---- Step 2: geometry
  rb = rung(ch, 2, 'Geometry: why lasso gives exact zeros and ridge does not',
    `<p>Consider just two features: IFN-γ (a true predictor) and PD-L1 (a proxy: it rises with the same immune activity but carries little extra information). A penalised fit is the point that makes the log-loss as small as possible <b>inside a region</b> whose size is set by λ: a diamond |w₁| + |w₂| ≤ t for the lasso, a circle w₁² + w₂² ≤ t for ridge, a rounded diamond for elastic net. The contours of the log-loss are shown in grey; the penalised solution is where the region first touches the lowest contour it can reach. The corners of the diamond lie on the axes, so the touching point is often a corner, where one weight is exactly zero.</p>`);
  const f2c = figure(rb, 'Loss contours, penalty region and solution path', 'Left: weights of IFN-γ and PD-L1 (per SD). The shaded region is the set of weights whose penalty does not exceed that of the current solution. The cross marks the unpenalised fit. The line is the path of solutions as λ grows. Right: each weight against λ for the chosen penalty. Hover the left panel for the log-loss at a point.');
  const X2 = D.map((p) => [st10.Z[D.indexOf(p)][0], st10.Z[D.indexOf(p)][3]]), L2 = lipschitz(X2);
  const unp2 = fitPenalised(X2, y, { lambda: 1e-6, alpha: 0 });
  const lim = 1.15 * Math.max(Math.abs(unp2.w[0]), Math.abs(unp2.w[1]));
  let pen2 = 'lasso', s2 = 12;
  select(f2c.controls, { label: 'Penalty', value: pen2, options: Object.entries(PEN).map(([k, v]) => [k, v.label]), onChange: (v) => { pen2 = v; trace2(); paint2(); } });
  slider(f2c.controls, { label: 'penalty strength λ', min: 0, max: K - 1, step: 1, value: s2, fmt: (i) => lamTxt(GRID[i]), onInput: (v) => { s2 = v; paint2(); } });
  const ss2 = stats(f2c.root);
  const row2 = el('div', { class: 'fig-row' }); f2c.body.append(row2);
  const ga = el('div'), gb = el('div'); row2.append(ga, gb);
  const G = frame(ga, { width: 500, height: 480, x: [-lim, lim], y: [-lim, lim], xLabel: 'IFN-γ weight (per SD)', yLabel: 'PD-L1 weight (per SD)', grid: false });
  const NG = 36, lossCell = (a, b) => { let b0 = Math.log(base / (1 - base)); for (let t = 0; t < 5; t++) { let g = 0, h = 0; X2.forEach((r, i) => { const q = 1 / (1 + Math.exp(-(b0 + a * r[0] + b * r[1]))); g += q - y[i]; h += q * (1 - q); }); b0 -= g / Math.max(h, 1e-6); } return logLoss(X2.map((r) => 1 / (1 + Math.exp(-(b0 + a * r[0] + b * r[1])))), y); };
  const cells = [], gridV = [];
  for (let j = 0; j < NG; j++) for (let i = 0; i < NG; i++) { const a = -lim + 2 * lim * (i + 0.5) / NG, b = lim - 2 * lim * (j + 0.5) / NG; gridV.push(lossCell(a, b)); }
  const lmin = d3.min(gridV), levels = [0.004, 0.015, 0.035, 0.065, 0.11, 0.17].map((d) => lmin + d);
  const gp = d3.geoPath(d3.geoTransform({ point(px, py) { this.stream.point(px * G.w / NG, py * G.h / NG); } }));
  G.layer('loss').selectAll('path').data(d3.contours().size([NG, NG]).thresholds(levels)(gridV)).join('path').attr('d', gp).attr('fill', 'none').attr('stroke', muted()).attr('stroke-width', 1.3);
  G.g.append('clipPath').attr('id', 'reg-g-clip').append('rect').attr('x', 7).attr('y', 7).attr('width', G.w - 14).attr('height', G.h - 14);
  G.layer('loss').attr('clip-path', 'url(#reg-g-clip)'); G.layer('reg').attr('clip-path', 'url(#reg-g-clip)');
  G.layer('ax').append('line').attr('x1', G.x(0)).attr('x2', G.x(0)).attr('y1', 0).attr('y2', G.h).attr('stroke', muted()).attr('stroke-dasharray', '3 4');
  G.layer('ax').append('line').attr('x1', 0).attr('x2', G.w).attr('y1', G.y(0)).attr('y2', G.y(0)).attr('stroke', muted()).attr('stroke-dasharray', '3 4');
  const NP = 70, penV = (a, b, al) => al * (Math.abs(a) + Math.abs(b)) + (1 - al) * (a * a + b * b) / 2;
  const regionL = G.layer('reg'), pathL = G.layer('path');
  G.layer('hit').append('rect').attr('width', G.w).attr('height', G.h).attr('fill', 'none').attr('pointer-events', 'all')
    .on('mousemove', (e) => { const r = G.svg.node().getBoundingClientRect(), sc = r.width / 500, a = G.x.invert((e.clientX - r.left) / sc - G.margin.left), b = G.y.invert((e.clientY - r.top) / sc - G.margin.top); tip(e, `IFN-γ ${f2(a)}, PD-L1 ${f2(b)}<br>log-loss ${f3(lossCell(a, b))}`); }).on('mouseleave', untip);
  G.layer('cross').append('path').attr('d', d3.symbol(d3.symbolCross, 90)).attr('transform', `translate(${G.x(unp2.w[0])},${G.y(unp2.w[1])}) rotate(45)`).attr('fill', ink()).attr('stroke', css('--surface'));
  G.layer('cross').append('text').attr('x', G.x(unp2.w[0]) + 8).attr('y', G.y(unp2.w[1]) + 24).attr('text-anchor', 'end').attr('font-size', 11.5).attr('fill', ink2()).text('unpenalised fit');
  const cur2 = G.layer('cur'), note2 = G.layer('note').append('text').attr('font-size', 12).attr('fill', ink()).attr('text-anchor', 'middle');
  const H2 = frame(gb, { width: 400, height: 480, x: [-3, -0.2], y: [-0.2, unp2.w[0] * 1.15], xLabel: 'log10 λ', yLabel: 'weight per SD' });
  let tr2 = [];
  const solve2 = () => { const out = []; let init = null; for (let k = K - 1; k >= 0; k--) { init = fitPenalised(X2, y, { lambda: GRID[k], alpha: PEN[pen2].a, init, iters: 300, L: L2 }); out[k] = init; } return out; };
  function trace2() { tr2 = solve2(); }
  trace2();
  H2.layer('z').append('line').attr('x1', 0).attr('x2', H2.w).attr('y1', H2.y(0)).attr('y2', H2.y(0)).attr('stroke', muted()).attr('stroke-dasharray', '4 4');
  const hl = H2.layer('l'), hm = H2.layer('m');
  const hLab = H2.layer('t');
  hLab.append('text').attr('font-size', 11.5).attr('fill', ink2()).attr('id', 'reg-h-a');
  function paint2() {
    const f = tr2[s2], al = PEN[pen2].a === 0 ? 0 : PEN[pen2].a, thr = penV(f.w[0], f.w[1], al === 0 ? 0 : al);
    const reg = d3.contours().size([NP, NP]).thresholds([-thr - 1e-9])(d3.range(NP * NP).map((q) => { const i = q % NP, j = Math.floor(q / NP); return -penV(-lim + 2 * lim * (i + 0.5) / NP, lim - 2 * lim * (j + 0.5) / NP, al); }));
    const gp2 = d3.geoPath(d3.geoTransform({ point(px, py) { this.stream.point(px * G.w / NP, py * G.h / NP); } }));
    regionL.selectAll('path').data(reg).join('path').attr('d', gp2).attr('fill', css('--accent-soft')).attr('fill-opacity', 0.75).attr('stroke', cat(PEN[pen2].a === 1 ? 6 : PEN[pen2].a === 0 ? 2 : 3)).attr('stroke-width', 2);
    pathL.selectAll('path').data([tr2]).join('path').attr('fill', 'none').attr('stroke', ink2()).attr('stroke-width', 1.6).attr('d', d3.line().x((q) => G.x(q.w[0])).y((q) => G.y(q.w[1])));
    cur2.selectAll('circle').data([f]).join('circle').attr('cx', G.x(f.w[0])).attr('cy', G.y(f.w[1])).attr('r', 6.5).attr('fill', cat(PEN[pen2].a === 1 ? 6 : PEN[pen2].a === 0 ? 2 : 3)).attr('stroke', css('--surface')).attr('stroke-width', 2);
    note2.attr('x', G.w / 2).attr('y', G.h - 8).text(Math.abs(f.w[1]) < NZ ? 'PD-L1 weight is exactly 0 (solution on an axis)' : '');
    [0, 1].forEach((j) => hl.selectAll('path.l' + j).data([0]).join('path').attr('class', 'l' + j).attr('fill', 'none').attr('stroke', cat(j === 0 ? 1 : 4)).attr('stroke-width', 2.2).attr('d', d3.line().x((q, i) => H2.x(lg(GRID[i]))).y((q) => H2.y(q.w[j]))(tr2)));
    hm.selectAll('line').data([lg(GRID[s2])]).join('line').attr('x1', (v) => H2.x(v)).attr('x2', (v) => H2.x(v)).attr('y1', 0).attr('y2', H2.h).attr('stroke', ink2()).attr('stroke-dasharray', '3 3');
    hLab.select('text').attr('x', 8).attr('y', 16).text('lines: IFN-γ (blue), PD-L1 (purple)').attr('opacity', 0);
    ss2.set({ 'IFN-γ weight': f2(f.w[0]), 'PD-L1 weight': Math.abs(f.w[1]) < NZ ? '0 (exact)' : f2(f.w[1]), 'sum of |weights|': f2(Math.abs(f.w[0]) + Math.abs(f.w[1])) });
  }
  legend(f2c.body, [['IFN-γ weight', cat(1)], ['PD-L1 weight', cat(4)], ['penalty region', css('--accent-soft')], ['log-loss contours', muted()]]);
  paint2();
  const zeroIdx = (pen) => { const keep = pen2; pen2 = pen; const t = solve2(); pen2 = keep; return t.map((f, i) => (Math.abs(f.w[1]) < NZ ? i : -1)).filter((i) => i >= 0); };
  const zL = zeroIdx('lasso'), zR = zeroIdx('ridge');
  prose(rb, `<p>In this fit ${zL.length ? `the lasso sets the PD-L1 weight to exactly 0 for every λ from ${lamTxt(GRID[d3.min(zL)])} upwards (${zL.length} of the ${K} grid values)` : 'the lasso does not set the PD-L1 weight to 0 on this grid'}, while ridge ${zR.length ? `reaches 0 at ${zR.length} grid values` : 'never reaches 0 on the grid'}: it only gets closer. The cause is the shape of the penalty. Near an axis the diamond's boundary is a sharp corner, so a small tilt of the loss contours leaves the solution on the corner, whereas the circle has no corner. Penalised and constrained forms are equivalent: each λ corresponds to a region size t, and the shaded region always passes through the solution.</p>`);

  // ---- Step 3: coefficient paths
  rb = rung(ch, 3, 'Coefficient paths: what survives as λ grows',
    `<p>With ${LAB.length} study measurements plus ${NN} noise genes (${LAB.length + NN} features for ${n} patients) the same fit is repeated over a grid of ${K} values of λ, starting from the largest and using each solution to start the next ("warm starts"). Plotting each coefficient against log λ gives the <b>coefficient path</b>. Standard practice is to read it from right to left: at the right every coefficient is 0, and features enter the model one by one as λ decreases.</p>`);
  const f3f = figure(rb, 'Coefficient paths on the study measurements and noise genes', 'Each line is one feature (standardised). Orange, green and amber: the three planted predictors; blue: other study measurements; grey: noise genes. Dotted vertical lines mark the cross-validated choices of step 4; the solid line marks the current λ. Hover a line for its name.');
  let pen3 = 'lasso', s3 = CV.lasso.i1se;
  select(f3f.controls, { label: 'Penalty', value: pen3, options: Object.entries(PEN).map(([k, v]) => [k, v.label]), onChange: (v) => { pen3 = v; s3 = CV[v].i1se; sl3.set(s3); paint3(true); } });
  const sl3 = slider(f3f.controls, { label: 'penalty strength λ', min: 0, max: K - 1, step: 1, value: s3, fmt: (i) => lamTxt(GRID[i]), onInput: (v) => { s3 = v; paint3(false); } });
  const ss3 = stats(f3f.root);
  const F3 = frame(f3f.body, { width: 820, height: 380, x: [-3, -0.2], y: [-0.7, 1.0], xLabel: 'log10 λ (penalty strength →)', yLabel: 'standardised weight (log-odds per SD)' });
  F3.layer('z').append('line').attr('x1', 0).attr('x2', F3.w).attr('y1', F3.y(0)).attr('y2', F3.y(0)).attr('stroke', muted());
  const pl3 = F3.layer('p'), mk3 = F3.layer('m'), lb3 = F3.layer('lab');
  function paint3(redraw) {
    const pth = P40[pen3], J = d3.range(pth[0].w.length);
    const ymax = Math.max(0.3, d3.max(pth, (f) => d3.max(f.w))) * 1.1, ymin = Math.min(-0.1, d3.min(pth, (f) => d3.min(f.w))) * 1.1;
    if (redraw) {
      F3.y.domain([ymin, ymax]); F3.g.select('.axis.y').call(d3.axisLeft(F3.y).ticks(5)); F3.g.select('.gridline').call(d3.axisLeft(F3.y).ticks(5).tickSize(-F3.w).tickFormat(''));
      F3.layer('z').select('line').attr('y1', F3.y(0)).attr('y2', F3.y(0));
      const order = J.slice().sort((a, b) => (kindOf(a) === 'noise' ? 0 : 1) - (kindOf(b) === 'noise' ? 0 : 1));
      pl3.selectAll('*').remove();
      order.forEach((j) => {
        const line = d3.line().x((f, i) => F3.x(lg(GRID[i]))).y((f) => F3.y(f.w[j]));
        pl3.append('path').datum(pth).attr('fill', 'none').attr('stroke', colOf(j)).attr('stroke-width', kindOf(j) === 'true' ? 2.6 : kindOf(j) === 'other' ? 1.6 : 1).attr('stroke-opacity', kindOf(j) === 'noise' ? 0.55 : 1).attr('d', line);
        pl3.append('path').datum(pth).attr('fill', 'none').attr('stroke', 'transparent').attr('stroke-width', 8).attr('d', line).attr('pointer-events', 'stroke')
          .on('mousemove', (e) => tip(e, `${nameOf(j)}<br>${kindOf(j) === 'true' ? 'planted predictor' : kindOf(j) === 'other' ? 'other study measurement' : 'noise gene'}<br>weight at current λ: ${f2(pth[s3].w[j])}`)).on('mouseleave', untip);
      });
      lb3.selectAll('*').remove();
      [['CV min', CV[pen3].iMin, 14], ['CV 1-SE', CV[pen3].i1se, 28]].forEach(([t, i, yy]) => {
        lb3.append('line').attr('x1', F3.x(lg(GRID[i]))).attr('x2', F3.x(lg(GRID[i]))).attr('y1', 0).attr('y2', F3.h).attr('stroke', ink2()).attr('stroke-dasharray', '2 4');
        lb3.append('text').attr('x', F3.x(lg(GRID[i])) + 4).attr('y', yy).attr('font-size', 11.5).attr('fill', ink2()).text(t);
      });
    }
    mk3.selectAll('line').data([lg(GRID[s3])]).join('line').attr('x1', (v) => F3.x(v)).attr('x2', (v) => F3.x(v)).attr('y1', 0).attr('y2', F3.h).attr('stroke', ink()).attr('stroke-width', 2);
    const f = pth[s3], nzj = J.filter((j) => Math.abs(f.w[j]) > NZ);
    ss3.set({ λ: lamTxt(GRID[s3]), 'non-zero features': nzj.length, 'planted predictors kept': `${nzj.filter((j) => j < 3).length} of 3`, 'noise genes kept': `${nzj.filter((j) => kindOf(j) === 'noise').length} of ${NN}`, 'validation AUC': f3(f.va.auc) });
  }
  legend(f3f.body, [['IFN-γ', cat(1)], ['TMB (log2)', cat(2)], ['CAF score', cat(3)], ['other study measurements', cat(0)], ['noise genes', muted()]]);
  paint3(true);
  const pl = P40.lasso, pr = P40.ridge;
  const firstIn = (pth, j) => { for (let k = K - 1; k >= 0; k--) if (Math.abs(pth[k].w[j]) > NZ) return k; return -1; };
  const firstNoise = d3.min(d3.range(LAB.length, LAB.length + NN), (j) => -firstIn(pl, j)) * -1;
  prose(rb, `<p>In the lasso path the first feature to enter, at λ = ${lamTxt(GRID[firstIn(pl, 0)])}, is ${nameOf(d3.maxIndex(pl[Math.max(0, firstIn(pl, 0))].w.map(Math.abs)))}; the first noise gene enters at λ = ${lamTxt(GRID[firstNoise])}, and by the smallest λ ${pl[0].nz} of ${pl[0].w.length} features are in the model. Ridge keeps all ${pr[K - 1].nz} features at every λ: the noise genes receive small weights that fade as λ grows but never vanish. Along the lasso path the validation AUC rises from ${f3(pl[K - 1].va.auc)} (nothing selected, a constant prediction) to ${f3(d3.max(pl, (f) => f.va.auc))} at its best, then falls to ${f3(pl[0].va.auc)} as noise genes enter. Choosing the point on this curve is the task of the next step.</p>`);

  // ---- Step 4: cross-validation
  rb = rung(ch, 4, 'Choosing λ by cross-validation',
    `<p>λ cannot be chosen by minimising the training error: that would always select the smallest λ. Instead the ${n} patients are split into 5 folds. For each λ, the model is fitted on four folds and its log-loss is measured on the fifth; this is repeated for each fold and averaged. Standardisation and fitting are repeated inside each fold. The λ with the lowest average is <b>λ<sub>min</sub></b>; the <b>1-SE rule</b> instead takes the largest λ (simplest model) whose average is within one standard error of the minimum, which usually gives a sparser model with similar accuracy.</p>`);
  const f4 = figure(rb, 'Five-fold cross-validation curve', 'Dots with bars: mean ± 1 standard error across the five folds. Dashed line: the same metric measured on Centre C with the model fitted on all discovery patients, which was not used to choose λ. Numbers along the top: features with non-zero weight at that λ (fit on all discovery patients).');
  let pen4 = 'lasso', met4 = 'll';
  select(f4.controls, { label: 'Penalty', value: pen4, options: Object.entries(PEN).map(([k, v]) => [k, v.label]), onChange: (v) => { pen4 = v; paint4(); } });
  select(f4.controls, { label: 'Metric', value: met4, options: [['ll', 'Log-loss (lower is better)'], ['au', 'AUC (higher is better)']], onChange: (v) => { met4 = v; paint4(); } });
  const ss4 = stats(f4.root);
  const F4 = frame(f4.body, { width: 820, height: 350, x: [-3.1, -0.1], y: [0.4, 0.9], xLabel: 'log10 λ', yLabel: 'cross-validated log-loss', margin: { top: 28 } });
  const l4 = F4.layer('c');
  legend(f4.body, [['5-fold CV mean ± SE', cat(2)], ['Centre C (validation)', ink2()], ['λ min', cat(6)], ['λ 1-SE', cat(3)]]);
  function paint4() {
    const cv = CV[pen4], S = met4 === 'll' ? cv.ll : cv.au, pth = P40[pen4], vv = pth.map((f) => (met4 === 'll' ? f.va.ll : f.va.auc));
    const lo = d3.min(S, (q) => q.m - q.se), hi = d3.max(S, (q) => q.m + q.se);
    const a = Math.min(lo, d3.min(vv)), b = Math.max(hi, d3.max(vv)), padd = (b - a) * 0.08;
    F4.y.domain([a - padd, b + padd]); F4.g.select('.axis.y').call(d3.axisLeft(F4.y).ticks(5)); F4.g.select('.gridline').call(d3.axisLeft(F4.y).ticks(5).tickSize(-F4.w).tickFormat(''));
    F4.g.select('.ax-label[transform]').text(met4 === 'll' ? 'cross-validated log-loss' : 'cross-validated AUC');
    l4.selectAll('*').remove();
    l4.append('path').datum(vv).attr('fill', 'none').attr('stroke', ink2()).attr('stroke-width', 2).attr('stroke-dasharray', '5 4').attr('d', d3.line().x((v, i) => F4.x(lg(GRID[i]))).y((v) => F4.y(v)));
    l4.selectAll('line.eb').data(S).join('line').attr('class', 'eb').attr('x1', (q, i) => F4.x(lg(GRID[i]))).attr('x2', (q, i) => F4.x(lg(GRID[i]))).attr('y1', (q) => F4.y(q.m - q.se)).attr('y2', (q) => F4.y(q.m + q.se)).attr('stroke', cat(2)).attr('stroke-width', 1.4);
    l4.append('path').datum(S).attr('fill', 'none').attr('stroke', cat(2)).attr('stroke-width', 2.2).attr('d', d3.line().x((q, i) => F4.x(lg(GRID[i]))).y((q) => F4.y(q.m)));
    l4.selectAll('circle').data(S).join('circle').attr('cx', (q, i) => F4.x(lg(GRID[i]))).attr('cy', (q) => F4.y(q.m)).attr('r', (q, i) => (i === cv.iMin || i === cv.i1se ? 7 : 3.8)).attr('fill', (q, i) => (i === cv.iMin ? cat(6) : i === cv.i1se ? cat(3) : cat(2))).attr('stroke', css('--surface')).attr('stroke-width', 1.5)
      .on('mousemove', (e, q) => { const i = S.indexOf(q); tip(e, `λ = ${lamTxt(GRID[i])}<br>CV ${met4 === 'll' ? 'log-loss' : 'AUC'} ${f3(q.m)} ± ${f3(q.se)}<br>Centre C ${f3(vv[i])}<br>${pth[i].nz} features non-zero`); }).on('mouseleave', untip);
    d3.range(1, K, 3).forEach((i) => l4.append('text').attr('x', F4.x(lg(GRID[i]))).attr('y', -10).attr('text-anchor', 'middle').attr('font-size', 11.5).attr('fill', ink2()).text(pth[i].nz));
    l4.append('text').attr('x', 0).attr('y', -10).attr('text-anchor', 'start').attr('font-size', 11.5).attr('fill', muted()).text('');
    const fm = pth[cv.iMin], f1s = pth[cv.i1se];
    ss4.set({ 'λ min': `${lamTxt(GRID[cv.iMin])} (${fm.nz} features)`, 'λ 1-SE': `${lamTxt(GRID[cv.i1se])} (${f1s.nz} features)`, 'CV log-loss at min': f3(cv.ll[cv.iMin].m), 'Centre C AUC at 1-SE': f3(f1s.va.auc) });
  }
  paint4();
  const cl = CV.lasso, cr = CV.ridge, ce = CV.enet;
  prose(rb, `<p>For the lasso, λ<sub>min</sub> = ${lamTxt(GRID[cl.iMin])} keeps ${P40.lasso[cl.iMin].nz} features and the 1-SE choice λ = ${lamTxt(GRID[cl.i1se])} keeps ${P40.lasso[cl.i1se].nz}; their Centre C AUCs are ${f3(P40.lasso[cl.iMin].va.auc)} and ${f3(P40.lasso[cl.i1se].va.auc)}. Ridge reaches a cross-validated log-loss of ${f3(cr.ll[cr.iMin].m)} at λ = ${lamTxt(GRID[cr.iMin])} (Centre C AUC ${f3(P40.ridge[cr.iMin].va.auc)}), and the elastic net ${f3(ce.ll[ce.iMin].m)} (${f3(P40.enet[ce.iMin].va.auc)}). ${(d3.max([cl, cr, ce].map((c) => c.ll[c.iMin].m)) - d3.min([cl, cr, ce].map((c) => c.ll[c.iMin].m))) < cl.ll[cl.iMin].se ? `The three minima differ by less than one standard error (${f3(cl.ll[cl.iMin].se)} for the lasso), so the cross-validation does not separate the penalties; this is common when a few features dominate and the rest are noise.` : 'The three penalties differ by more than one standard error at their minima.'} The cross-validation curve estimates performance only up to its noise: with ${n} patients and ${nR} responders the bars are wide, and the 1-SE rule is a response to this uncertainty. glmnet's <code>cv.glmnet</code> reports both choices as <code>lambda.min</code> and <code>lambda.1se</code>.</p>`);
  quiz(rb, 'Why is the cross-validation estimate at λ<sub>min</sub> slightly optimistic as an estimate of performance on new patients?', `<p>λ<sub>min</sub> was chosen because it had the best cross-validation score out of ${K} values, so that score has been selected to be favourable. Here the CV log-loss at the minimum is ${f3(cl.ll[cl.iMin].m)} for the lasso and the Centre C log-loss is ${f3(P40.lasso[cl.iMin].va.ll)}${P40.lasso[cl.iMin].va.ll <= cl.ll[cl.iMin].m ? ' (lower, so no optimism is visible)' : ''}: the optimism from tuning one λ is smaller than the sampling differences between the cohorts, but it grows with the number of tuned choices (gene filters, α, model type). An unbiased estimate needs data not used for choosing λ (nested cross-validation, or an independent cohort such as Centre C).</p>`);

  // ---- Step 5: lasso selection and stability
  const ST = R.stab, NF = LAB.length + ST.n_noise;
  const selM = (li) => ST.sel[li].map((s) => [...s].map((c) => +c));
  const freqAt = (li) => d3.range(NF).map((j) => mean(selM(li).map((r) => r[j])));
  const li0 = d3.minIndex(ST.lambdas, (l) => Math.abs(lg(l) - lg(lamL)));
  rb = rung(ch, 5, 'Lasso as feature selection, and how stable it is',
    `<p>The lasso returns a list of features, which is often read as "the biomarkers". To test how reproducible that list is, the lasso was refitted on <b>${ST.B} random half-samples</b> of the discovery patients (${ST.m} each, computed in Python with scikit-learn; script <code>scripts/build_regularization.py</code>). <b>Stability selection</b> (Meinshausen and Bühlmann) keeps the features that are selected in at least a chosen fraction π of the subsamples. In the figure each row is a feature and each column a subsample; a filled cell means that the feature had a non-zero weight in that subsample.</p>`);
  const f5 = figure(rb, 'Which features does the lasso pick in each half-sample?', 'Rows: the features most often selected at the chosen λ (at most 18 shown). Left: one column per half-sample. Right: selection frequency; the dashed line is the threshold π. Orange, green and amber rows are the planted predictors, blue rows other study measurements, grey rows noise genes.');
  let li = li0, thr5 = 0.6;
  slider(f5.controls, { label: 'penalty strength λ', min: 0, max: ST.lambdas.length - 1, step: 1, value: li, fmt: (i) => lamTxt(ST.lambdas[i]), onInput: (v) => { li = v; paint5(); } });
  slider(f5.controls, { label: 'threshold π', min: 0.3, max: 0.95, step: 0.05, value: thr5, fmt: (v) => v.toFixed(2), onInput: (v) => { thr5 = v; paint5(); } });
  const ss5 = stats(f5.root);
  const W5 = 820, RH = 17, NR = 18, M5 = { l: 128, t: 34 }, HX = 7.4, BX = M5.l + ST.B * HX + 28, BW = W5 - BX - 16;
  const svg5 = d3.select(f5.body).append('svg').attr('viewBox', `0 0 ${W5} ${M5.t + NR * RH + 44}`).attr('width', W5).attr('role', 'img');
  const g5 = svg5.append('g'), bx5 = d3.scaleLinear().domain([0, 1]).range([0, BW - 36]);
  svg5.append('text').attr('x', M5.l).attr('y', 16).attr('font-size', 12).attr('fill', ink2()).text(`half-samples 1–${ST.B} →`);
  svg5.append('text').attr('x', BX).attr('y', 16).attr('font-size', 12).attr('fill', ink2()).text('selection frequency');
  svg5.append('g').attr('class', 'axis x').attr('transform', `translate(${BX},${M5.t + NR * RH + 4})`).call(d3.axisBottom(bx5).ticks(5));
  const thrLine = svg5.append('line').attr('stroke', ink()).attr('stroke-dasharray', '4 3').attr('stroke-width', 1.6), thrTxt = svg5.append('text').attr('font-size', 11.5).attr('fill', ink()).attr('text-anchor', 'middle');
  legend(f5.body, [['planted predictors', cat(1)], ['other study measurements', cat(0)], ['noise genes', muted()]]);
  const stabNames = (j) => nameOf(j);
  function paint5() {
    const M = selM(li), fr = d3.range(NF).map((j) => mean(M.map((r) => r[j])));
    const top = d3.range(NF).filter((j) => fr[j] > 0).sort((a, b) => fr[b] - fr[a] || a - b).slice(0, NR);
    g5.selectAll('*').remove();
    top.forEach((j, r) => {
      const yy = M5.t + r * RH;
      g5.append('text').attr('x', M5.l - 6).attr('y', yy + RH * 0.72).attr('text-anchor', 'end').attr('font-size', 11.5).attr('fill', ink2()).text(stabNames(j));
      g5.selectAll(null).data(M).join('rect').attr('x', (m, b) => M5.l + b * HX).attr('y', yy + 1).attr('width', HX - 1).attr('height', RH - 2).attr('rx', 1.5).attr('fill', (m) => (m[j] ? colOf(j) : css('--surface-2'))).attr('fill-opacity', (m) => (m[j] && kindOf(j) === 'noise' ? 0.75 : 1));
      g5.append('rect').attr('x', BX).attr('y', yy + 2).attr('width', bx5(fr[j])).attr('height', RH - 4).attr('fill', colOf(j)).attr('fill-opacity', fr[j] >= thr5 ? 1 : 0.45)
        .on('mousemove', (e) => tip(e, `${stabNames(j)}<br>selected in ${Math.round(fr[j] * ST.B)} of ${ST.B} half-samples (${pc(fr[j])})`)).on('mouseleave', untip);
      g5.append('text').attr('x', BX + bx5(fr[j]) + 4).attr('y', yy + RH * 0.72).attr('font-size', 11).attr('fill', ink2()).text(Math.round(fr[j] * 100) + '%');
    });
    thrLine.attr('x1', BX + bx5(thr5)).attr('x2', BX + bx5(thr5)).attr('y1', M5.t - 4).attr('y2', M5.t + NR * RH);
    thrTxt.attr('x', BX + bx5(thr5)).attr('y', M5.t + NR * RH + 38).text('');
    const stable = d3.range(NF).filter((j) => fr[j] >= thr5), everNoise = d3.range(LAB.length, NF).filter((j) => fr[j] > 0).length;
    const sizes = M.map((r) => sum(r));
    ss5.set({ λ: lamTxt(ST.lambdas[li]), 'features per half-sample': `${d3.min(sizes)}–${d3.max(sizes)} (mean ${mean(sizes).toFixed(1)})`, 'stable at π': `${stable.length} (${stable.filter((j) => j < 3).length} planted, ${stable.filter((j) => kindOf(j) === 'noise').length} noise)`, 'noise genes selected at least once': `${everNoise} of ${ST.n_noise}` });
  }
  paint5();
  { const M = selM(li0), fr = freqAt(li0), sizes = M.map((r) => sum(r)), pcs = (j) => Math.round(fr[j] * ST.B);
    const sets = new Set(M.map((r) => r.join(''))).size, lo5 = Math.max(0, li0 - 3), hi5 = Math.min(ST.lambdas.length - 1, li0 + 3), frLo = freqAt(lo5), frHi = freqAt(hi5);
    prose(rb, `<p>At λ = ${lamTxt(ST.lambdas[li0])} (nearest to the 1-SE choice above) the half-sample fits keep between ${d3.min(sizes)} and ${d3.max(sizes)} features, and ${sets} different feature sets occur in ${ST.B} fits. IFN-γ is selected in ${pcs(0)}, TMB in ${pcs(1)} and CAF score in ${pcs(2)} of the ${ST.B} fits. ${everNoise0(fr)} of the ${ST.n_noise} noise genes were selected at least once, though most only rarely. PD-L1 and CD8, which carry the same signal as IFN-γ, are selected in ${pcs(3)} and ${pcs(4)}: when features are correlated, the lasso picks one of them and the choice changes with the sample. Three grid steps lower (λ = ${lamTxt(ST.lambdas[lo5])}), ${everNoise0(frLo)} noise genes are selected at least once and fits keep up to ${d3.max(selM(lo5).map((r) => sum(r)))} features; three steps higher (λ = ${lamTxt(ST.lambdas[hi5])}), CAF score is selected in ${Math.round(frHi[2] * ST.B)} of ${ST.B} fits, against ${pcs(2)} here, and noise genes appear in ${everNoise0(frHi)} of ${ST.n_noise} cases.</p>
      <p>The lasso is best regarded as a <b>predictive model that happens to be sparse</b>. A feature list should be reported with its selection frequencies, as here, and not as a single set. The elastic net (α &lt; 1) distributes weight across correlated features, which usually makes the lists more reproducible; this page does not test that claim, but the lab lets the penalty be switched to the elastic net.</p>`); }
  function everNoise0(fr) { return d3.range(LAB.length, NF).filter((j) => fr[j] > 0).length; }

  // ---- Step 6: lab
  rb = rung(ch, 6, 'Lab: build a penalised model and test it on Centre C',
    `<p>Choose a penalty, the number of noise genes added to the ${LAB.length} study measurements, and λ. Features are standardised using the ${n} discovery patients. Training results use the discovery patients; validation results use the ${V.length} Centre C patients, which are never used for fitting or for λ selection in this lab.</p>`);
  const fl = figure(rb, 'Penalised model builder', 'Left: the largest standardised weights at the chosen λ (up to 12 shown). Right: training and validation AUC for every λ with the current penalty and number of noise genes; the vertical line marks the chosen λ. Paths are computed with a fixed number of solver iterations and are approximate at the smallest λ.');
  let penL = 'lasso', nL = NN, sL = CV.lasso.i1se;
  const NOISE_STEPS = [0, 10, 20, 40, 60, 80];
  select(fl.controls, { label: 'Penalty', value: penL, options: Object.entries(PEN).map(([k, v]) => [k, v.label]), onChange: (v) => { penL = v; paintL(true); } });
  slider(fl.controls, { label: 'noise genes added', min: 0, max: NOISE_STEPS.length - 1, step: 1, value: NOISE_STEPS.indexOf(nL), fmt: (i) => String(NOISE_STEPS[i]), onInput: (v) => { nL = NOISE_STEPS[v]; paintL(true); } });
  const slL = slider(fl.controls, { label: 'penalty strength λ', min: 0, max: K - 1, step: 1, value: sL, fmt: (i) => lamTxt(GRID[i]), onInput: (v) => { sL = v; paintL(false); } });
  button(fl.controls, 'Set λ to CV 1-SE (40 genes)', () => { sL = CV[penL].i1se; slL.set(sL); paintL(false); });
  const ssL = stats(fl.root);
  const rowL = el('div', { class: 'fig-row' }); fl.body.append(rowL);
  const la = el('div'), lb = el('div'); rowL.append(la, lb);
  const BARS = 12, bsvg = d3.select(la).append('svg').attr('viewBox', `0 0 440 ${BARS * 24 + 64}`).attr('width', 440).attr('role', 'img');
  const FL = frame(lb, { width: 440, height: BARS * 24 + 64, x: [-3, -0.2], y: [0.4, 1], xLabel: 'log10 λ', yLabel: 'AUC' });
  const pl6 = FL.layer('p'), mk6 = FL.layer('m');
  legend(fl.body, [['planted predictor', cat(1)], ['other study measurement', cat(0)], ['noise gene', muted()], ['validation AUC', cat(2)], ['training AUC', ink2()]]);
  function paintL(redraw) {
    const pth = pathOf(penL, nL), f = pth[sL], m = mk(nL);
    if (redraw) {
      const all = pth.flatMap((q) => [q.tr.auc, q.va.auc]); FL.y.domain([Math.min(0.5, d3.min(all) - 0.02), 1]);
      FL.g.select('.axis.y').call(d3.axisLeft(FL.y).ticks(5)); FL.g.select('.gridline').call(d3.axisLeft(FL.y).ticks(5).tickSize(-FL.w).tickFormat(''));
      pl6.selectAll('*').remove();
      pl6.append('line').attr('x1', 0).attr('x2', FL.w).attr('y1', FL.y(oracleV.auc)).attr('y2', FL.y(oracleV.auc)).attr('stroke', muted()).attr('stroke-dasharray', '2 4').attr('stroke-width', 2);
      pl6.append('text').attr('x', FL.w - 4).attr('y', FL.y(oracleV.auc) - 6).attr('text-anchor', 'end').attr('font-size', 11).attr('fill', muted()).text('3 planted features, no penalty');
      [['tr', ink2(), '5 4'], ['va', cat(2), null]].forEach(([k, c, d]) => pl6.append('path').datum(pth).attr('fill', 'none').attr('stroke', c).attr('stroke-width', 2.2).attr('stroke-dasharray', d).attr('d', d3.line().x((q, i) => FL.x(lg(GRID[i]))).y((q) => FL.y(q[k].auc))));
    }
    mk6.selectAll('line').data([lg(GRID[sL])]).join('line').attr('x1', (v) => FL.x(v)).attr('x2', (v) => FL.x(v)).attr('y1', 0).attr('y2', FL.h).attr('stroke', ink()).attr('stroke-width', 2);
    const J = f.w.map((w, j) => j).filter((j) => Math.abs(f.w[j]) > NZ).sort((a, b) => Math.abs(f.w[b]) - Math.abs(f.w[a])).slice(0, BARS);
    const ext = Math.max(0.3, d3.max(J, (j) => Math.abs(f.w[j])) || 0.3) * 1.15, xs = d3.scaleLinear().domain([-ext, ext]).range([130, 424]);
    bsvg.selectAll('*').remove();
    bsvg.append('line').attr('x1', xs(0)).attr('x2', xs(0)).attr('y1', 6).attr('y2', BARS * 24 + 6).attr('stroke', muted());
    bsvg.append('g').attr('class', 'axis x').attr('transform', `translate(0,${BARS * 24 + 8})`).call(d3.axisBottom(xs).ticks(5));
    bsvg.append('text').attr('class', 'ax-label').attr('x', 277).attr('y', BARS * 24 + 46).attr('text-anchor', 'middle').text('standardised weight (log-odds per SD)');
    J.forEach((j, r) => {
      const idx = j, yy = 8 + r * 24;
      bsvg.append('text').attr('x', 124).attr('y', yy + 15).attr('text-anchor', 'end').attr('font-size', 11.5).attr('fill', ink2()).text(nameOf(idx));
      bsvg.append('rect').attr('x', Math.min(xs(0), xs(f.w[j]))).attr('width', Math.abs(xs(f.w[j]) - xs(0))).attr('y', yy + 3).attr('height', 16).attr('fill', colOf(idx))
        .on('mousemove', (e) => tip(e, `${nameOf(idx)}<br>weight ${f2(f.w[j])} per SD`)).on('mouseleave', untip);
      bsvg.append('text').attr('x', xs(f.w[j]) + (f.w[j] >= 0 ? 4 : -4)).attr('y', yy + 15).attr('text-anchor', f.w[j] >= 0 ? 'start' : 'end').attr('font-size', 11).attr('fill', ink2()).text(f2(f.w[j]));
    });
    if (!J.length) bsvg.append('text').attr('x', 220).attr('y', 60).attr('text-anchor', 'middle').attr('font-size', 12.5).attr('fill', ink2()).text('all weights are exactly 0 (constant prediction)');
    const nzAll = f.w.filter((w) => Math.abs(w) > NZ).length, nzNoise = f.w.filter((w, j) => j >= LAB.length && Math.abs(w) > NZ).length;
    ssL.set({ 'features / patients': `${LAB.length + nL} / ${n}`, 'non-zero': `${nzAll} (${nzNoise} noise)`, 'training AUC': f3(f.tr.auc), 'validation AUC': f3(f.va.auc), 'validation log-loss': f3(f.va.ll) });
  }
  paintL(true);
  const noiseKept = (f) => f.w.filter((w, j) => j >= LAB.length && Math.abs(w) > NZ).length;
  const at = (pen, N, i) => pathOf(pen, N)[i], s1se = CV.lasso.i1se, nz = (q) => q.nz;
  prose(rb, '<p><b>Suggested comparisons</b>:</p>');
  table(rb, [
    { t: '<b>Lasso</b>, 40 genes, CV 1-SE λ', s: `${nz(at('lasso', 40, s1se))} features kept; validation AUC <b>${f3(at('lasso', 40, s1se).va.auc)}</b> (training ${f3(at('lasso', 40, s1se).tr.auc)})`, m: `A sparse model whose training and validation performance are close; the reference with only the 3 planted predictors reaches ${f3(oracleV.auc)}.` },
    { t: '<b>Ridge</b> at its CV λ', s: `All ${nz(at('ridge', 40, cr.iMin))} features non-zero; validation AUC ${f3(at('ridge', 40, cr.iMin).va.auc)}`, m: 'Ridge cannot drop features, so noise genes keep small weights. It is competitive in prediction but gives no selection.' },
    { t: '<b>Elastic net</b> at its CV 1-SE λ', s: `${nz(at('enet', 40, ce.i1se))} features kept; validation AUC ${f3(at('enet', 40, ce.i1se).va.auc)}`, m: 'Between the two: sparse, but correlated features tend to enter together.' },
    { t: 'Slide λ to the <b>smallest</b> value (lasso, 40 genes)', s: `Training AUC ${f3(at('lasso', 40, 0).tr.auc)}, validation AUC <b>${f3(at('lasso', 40, 0).va.auc)}</b>, ${nz(at('lasso', 40, 0))} features`, m: 'With almost no penalty the model fits noise: training rises, validation falls.' },
    { t: `Lasso at <b>λ min</b> (${lamTxt(GRID[cl.iMin])}), genes <b>0 → 80</b>`, s: `Validation AUC ${f3(at('lasso', 0, cl.iMin).va.auc)} → <b>${f3(at('lasso', 80, cl.iMin).va.auc)}</b>; noise genes kept: ${noiseKept(at('lasso', 0, cl.iMin))} → ${noiseKept(at('lasso', 80, cl.iMin))}`, m: 'How many noise genes enter, and what they cost, depends on λ and on how many there are, so λ must be re-tuned for each dataset. At the stronger 1-SE λ no noise gene enters here, and the result does not change.' },
    { t: 'Set λ to the <b>largest</b> value', s: `${nz(at('lasso', 40, K - 1))} features; validation AUC ${f3(at('lasso', 40, K - 1).va.auc)}`, m: 'Everything is shrunk to zero and the model predicts the base rate for every patient (AUC 0.5): too much penalty underfits.' },
  ], [['t', 'Try'], ['s', 'What you see'], ['m', 'What it means']], { cls: 'compact' });

  // =====================================================================================
  // 5. PITFALLS
  // =====================================================================================
  ch = chapter(main, 'pitfalls', 'Pitfalls', 'Common problems, how to recognise them, and how to address them.');
  const nCB = D.filter((p) => p.centreB).length;
  cards(ch, [
    { tag: 'Symptom: penalty seems to ignore some features', title: 'Features not standardised', body: `Without standardising, the penalty favours features measured in large units (PD-L1 in % has SD ${sd10[3].toFixed(0)}, TMB on the log2 scale ${f2(sd10[1])}), as step 1 shows. <b>Fix:</b> standardise inside every training fold and apply the same scaling to new data. Check what the software does by default: glmnet standardises internally, scikit-learn does not.` },
    { tag: 'Symptom: CV looks better than the held-out cohort', title: 'Tuning and selection on the same data', body: `The best CV score is selected to be favourable. For one tuned λ the effect is small: here the lasso's CV log-loss at λ<sub>min</sub> is ${f3(cl.ll[cl.iMin].m)} and its Centre C log-loss ${f3(P40.lasso[cl.iMin].va.ll)}${P40.lasso[cl.iMin].va.ll <= cl.ll[cl.iMin].m ? ', so no optimism is visible' : ''}. The effect grows with every extra choice, and filtering genes on all patients before CV can inflate the estimate severely. <b>Fix:</b> put every data-driven step (filtering, scaling, choosing λ) inside the folds; use nested CV or an independent cohort for the final estimate.` },
    { tag: 'Symptom: different runs give different gene lists', title: 'Lasso selection is unstable', body: `Among correlated features the lasso keeps one more or less arbitrarily. In the ${ST.B} half-samples, ${everNoise0(freqAt(li0))} of ${ST.n_noise} noise genes were selected at least once at the default λ, and PD-L1 and CD8 each appear in some fits only. <b>Fix:</b> report selection frequencies (stability selection), use the elastic net, and validate the signature in new patients.` },
    { tag: 'Symptom: "the model says gene X matters"', title: 'Shrunken weights are not effect sizes', body: `Penalised coefficients are biased towards zero (at the CV λ the IFN-γ weight is ${f2(P40.lasso[CV.lasso.i1se].w[0])} per SD, against ${f2(unp10.w[0])} in an unpenalised fit of the 10 study measurements). Usual p-values and confidence intervals do not account for the penalty or the selection. <b>Fix:</b> use penalised models for prediction; for inference use an unpenalised model on pre-specified features, or methods designed for post-selection inference.` },
    { tag: 'Symptom: optimistic CV in multi-centre data', title: 'Folds that ignore structure', body: `Random folds put patients from the same centre, batch or family on both sides of the split. Centre B has ${nCB} of the ${n} discovery patients and a different phenotype mix, so a model can learn centre rather than biology. <b>Fix:</b> group the folds by centre, batch or patient, and test on a cohort from a different source, as Centre C does here.` },
    { tag: 'Symptom: λ from a tutorial does not transfer', title: 'λ is dataset-specific', body: `The best λ changed when the number of noise genes changed (lab, step 6). scikit-learn's default <code>C = 1</code> corresponds to λ = 1/(C·n) = ${lamTxt(1 / n)} in this page's units. <b>Fix:</b> always tune λ (<code>cv.glmnet</code>, <code>LogisticRegressionCV</code>) for the dataset and metric at hand.` },
    { tag: 'Symptom: a clinical covariate disappears', title: 'Penalising what should stay', body: 'Age, sex, centre or batch are often covariates that should always be adjusted for. A lasso may drop them. <b>Fix:</b> give them a penalty factor of 0 (glmnet <code>penalty.factor</code>), or fit them in a first stage and penalise only the genes.' },
    { tag: 'Symptom: ridge keeps thousands of features', title: 'Using the wrong penalty for the aim', body: `Ridge never selects (all ${pr[K - 1].nz} features stay non-zero), and a lasso cannot select more features than there are patients. <b>Fix:</b> ridge for the best prediction from many weak features; lasso or elastic net when a short list is needed; elastic net when features are correlated.` },
  ]);

  // =====================================================================================
  // 6. RECAP
  // =====================================================================================
  ch = chapter(main, 'recap', 'Recap', 'Key points.');
  recap(ch, [
    'Regularisation minimises <b>loss + λ × penalty</b>. It adds a little bias to remove variance, which pays off when features are many relative to patients.',
    '<b>Ridge</b> (L2) shrinks all coefficients smoothly and never reaches exactly 0; <b>lasso</b> (L1) sets some exactly to 0, so it selects features; the <b>elastic net</b> mixes the two (glmnet α = 0, 1 and between).',
    'Lasso gives exact zeros because the L1 region has corners on the axes; the L2 region is round and has none.',
    '<b>Standardise</b> features first, inside each training fold. Otherwise the penalty depends on units.',
    'The coefficient path shows features entering as λ decreases. λ is chosen by <b>cross-validation</b> (<code>lambda.min</code>, or the sparser <code>lambda.1se</code>), never by training error.',
    'Lasso feature lists are <b>unstable</b> among correlated features. Report selection frequencies or use the elastic net, and validate in new patients.',
    'Penalised coefficients are shrunken and selection-affected: use them for prediction, not as unbiased effect sizes or with naive p-values.',
    'Noise features still cost performance. Penalisation limits the damage but does not replace independent validation or good features.',
  ]);
  prose(ch, '<h3>In code</h3><p>glmnet (R) and scikit-learn (Python) fit the same models. glmnet\'s <code>lambda</code> is on the scale used in this page; scikit-learn\'s <code>C</code> is an inverse penalty strength, C = 1 / (λ × n) for the lasso and ridge.</p>');
  code(ch, {
    Python: `import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import roc_auc_score

# X_train: patients x features (study measurements + genes); y_train: 0/1 response
cv = LogisticRegressionCV(
    penalty="elasticnet", solver="saga", l1_ratios=[0.0, 0.5, 1.0],  # 0 = ridge, 0.5 = elastic net, 1 = lasso
    Cs=np.logspace(-2, 3, 30), cv=5, scoring="neg_log_loss", max_iter=5000)
model = make_pipeline(StandardScaler(), cv)       # scaling uses all training patients; use cross_val_score(model, ...) for a strictly nested estimate
model.fit(X_train, y_train)
w = model[-1].coef_.ravel()
print("non-zero features:", np.flatnonzero(w))
print("validation AUC", roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]))

# Stability selection by hand: lasso on repeated half-samples, count how often each feature is selected
rng = np.random.default_rng(0); freq = np.zeros(X_train.shape[1])
for _ in range(50):
    idx = rng.choice(len(y_train), len(y_train) // 2, replace=False)
    m = make_pipeline(StandardScaler(), LogisticRegression(penalty="l1", solver="saga", C=0.1, max_iter=5000))
    freq += (m.fit(X_train[idx], y_train[idx])[-1].coef_.ravel() != 0)
freq /= 50`,
    R: `library(glmnet)
# x: numeric matrix (patients x features); y: 0/1 response. glmnet standardises internally (standardize = TRUE)
set.seed(1)
cvfit <- cv.glmnet(x, y, family = "binomial", alpha = 1,        # alpha = 0 ridge, 1 lasso, 0.5 elastic net
                   nfolds = 5, type.measure = "deviance")
plot(cvfit)                                                      # the cross-validation curve
coef(cvfit, s = "lambda.1se")                                    # sparse coefficients (original units); also "lambda.min"
plot(cvfit$glmnet.fit, xvar = "lambda")                          # coefficient paths
p <- predict(cvfit, newx = x_val, s = "lambda.1se", type = "response")
pROC::auc(y_val, as.numeric(p))

# Keep clinical covariates unpenalised (penalty.factor = 0), penalise only the genes
# fit <- cv.glmnet(cbind(clin, genes), y, family = "binomial", penalty.factor = c(rep(0, ncol(clin)), rep(1, ncol(genes))))
# Stability selection: stabs::stabsel(x, y, fitfun = glmnet.lasso, cutoff = 0.75, PFER = 1)`,
  });
  seeAlso(main, [['concepts', 'logistic-regression', 'Logistic regression'], ['concepts', 'linear-regression', 'Linear regression'], ['concepts', 'cross-validation', 'Cross-validation and data leakage'], ['concepts', 'train-test', 'Train/test split and overfitting'], ['concepts', 'multiple-testing', 'Multiple testing and FDR'], ['omics', 'bulk-classifier', 'Biomarker classifier and nested CV']]);
}
