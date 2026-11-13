// Bulk step 8: per-patient explanations. Linear model: coefficient x (value - background mean), which IS the
// exact Shapley value for a linear model. Boosting: exact interventional Shapley values (all 2^8 coalitions).
import { header, figure, slider, select, stats, section, steps, callout, math, legend, seeAlso, tip, untip, el } from '../lib/ui.js';
import { frame, heatmap, cat, css, ink, ink2, muted, diverging } from '../lib/plot.js';
import { sigmoid } from '../lib/data.js';
import { stepCard, omicsData } from './_shared.js';
const d3 = window.d3;
const f2 = (v) => v.toFixed(2), f3 = (v) => v.toFixed(3), sgn = (v) => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(2);

export default async function render(main) {
  header(main, 'Explaining predictions',
    'A response model gives each patient a probability. An explanation splits that number into <b>additive contributions, one per gene</b>, relative to an average patient. That tells you what the <i>model</i> used for this patient. It does not tell you what <i>causes</i> response.');
  const D = await omicsData(main, 'bulk_explain.json');
  if (!D) return;
  const G = D.genes, n = D.y.length, P = G.length;
  const MOD = { lin: { label: 'Logistic regression (L2)', ...D.lin }, gb: { label: 'Gradient boosting (depth 2)', ...D.gb } };

  stepCard(main, {
    stage: 'Bulk RNA-seq · step 8 of 11',
    question: 'For this patient, which genes pushed the predicted response probability up or down, and which genes does the model rely on overall?',
    data: `${n} discovery patients; an ${P}-gene candidate panel (${G.join(', ')}), standardised.`,
    method: 'Linear contributions β·(x − x̄) for logistic regression; exact interventional Shapley values for gradient boosting, against a background of ' + D.n_bg + ' patients.',
    verdict: 'ml',
    why: 'Attribution is a tool for inspecting a fitted ML model (debugging, plausibility, per-patient reports). It is not an inferential test: no p-values, no causal claims.',
  });

  // ---------------- Figure 1: per-patient waterfall
  const order = d3.range(n).sort((a, b) => MOD.gb.f[a] - MOD.gb.f[b]);
  const pick = (cond, by) => d3.range(n).filter(cond).sort(by)[0];
  const ex = {
    tp: pick((i) => D.y[i] === 1, (a, b) => MOD.gb.f[b] - MOD.gb.f[a]),
    fp: pick((i) => D.y[i] === 0, (a, b) => MOD.gb.f[b] - MOD.gb.f[a]),
    fn: pick((i) => D.y[i] === 1, (a, b) => MOD.gb.f[a] - MOD.gb.f[b]),
    tn: pick((i) => D.y[i] === 0, (a, b) => MOD.gb.f[a] - MOD.gb.f[b]),
  };
  const f1 = figure(main, 'One patient, one prediction, split into gene contributions',
    'The bars walk from the model\'s average output (the base value, in log-odds) to this patient\'s output. Red bars push toward response, blue bars away. The label on each bar is the patient\'s standardised expression (z-score) for that gene. Patients are ordered by the boosting model\'s prediction.');
  let mk = 'gb', pi = ex.tp;
  select(f1.controls, { label: 'Model', value: mk, options: [['lin', MOD.lin.label], ['gb', MOD.gb.label]], onChange: (v) => { mk = v; draw1(); } });
  const sl = slider(f1.controls, { label: 'Patient (by prediction)', min: 0, max: n - 1, step: 1, value: order.indexOf(pi), fmt: (v) => `#${v + 1}`, onInput: (v) => { pi = order[v]; draw1(); } });
  select(f1.controls, { label: 'Jump to', value: 'tp', options: [['tp', 'most confident responder'], ['fp', 'most confident false positive'], ['fn', 'most confident false negative'], ['tn', 'most confident non-responder']], onChange: (v) => { pi = ex[v]; sl.set(order.indexOf(pi)); draw1(); } });
  legend(f1.root, [['pushes toward response', css('--div-pos')], ['pushes away from response', css('--div-neg')]]);
  const s1 = stats(f1.root);
  const W = el('div'); f1.body.append(W);
  const allF = [...MOD.lin.f, ...MOD.gb.f, MOD.lin.base, MOD.gb.base];
  const wlo = Math.floor(d3.min(allF)) - 0.5, whi = Math.ceil(d3.max(allF)) + 0.5;
  const F = frame(W, { width: 900, height: 360, x: [wlo, whi], y: [0, P + 2], xLabel: 'model output (log-odds of response)', axes: true, grid: false, margin: { left: 120, right: 110 } });
  F.g.select('.axis.y').remove();
  F.g.append('g').attr('class', 'gridline').call(d3.axisBottom(F.x).ticks(8).tickSize(F.h).tickFormat('')).lower();
  const pTicks = [0.05, 0.2, 0.5, 0.8, 0.95].filter((p) => Math.log(p / (1 - p)) > wlo && Math.log(p / (1 - p)) < whi);
  F.g.selectAll('text.pt').data(pTicks).join('text').attr('class', 'pt').attr('x', (p) => F.x(Math.log(p / (1 - p)))).attr('y', -1).attr('text-anchor', 'middle').attr('font-size', 10).attr('fill', muted()).text((p) => `p=${p}`);
  const rowY = (k) => F.y(P + 1 - k) - 10;

  function draw1() {
    const m = MOD[mk], phi = m.phi[pi], fx = m.f[pi];
    const ord = d3.range(P).sort((a, b) => Math.abs(phi[b]) - Math.abs(phi[a]));
    let acc = m.base; const bars = [];
    ord.forEach((j, k) => { bars.push({ j, k: k + 1, a: acc, b: acc + phi[j], v: phi[j] }); acc += phi[j]; });
    const lab = [{ k: 0, t: 'base value', x: m.base }, ...bars.map((b) => ({ k: b.k, t: G[b.j] })), { k: P + 1, t: 'prediction', x: fx }];
    F.layer('lab').selectAll('text').data(lab).join('text').attr('x', -8).attr('y', (d) => rowY(d.k) + 14).attr('text-anchor', 'end').attr('font-size', 12)
      .attr('fill', (d) => (d.x != null ? ink() : ink2())).attr('font-weight', (d) => (d.x != null ? 600 : 400)).text((d) => d.t);
    F.layer('bars').selectAll('rect').data(bars).join('rect').attr('x', (d) => F.x(Math.min(d.a, d.b))).attr('width', (d) => Math.max(1, Math.abs(F.x(d.b) - F.x(d.a))))
      .attr('y', (d) => rowY(d.k)).attr('height', 20).attr('rx', 3).attr('fill', (d) => css(d.v >= 0 ? '--div-pos' : '--div-neg'))
      .on('mousemove', (e, d) => tip(e, `<b>${G[d.j]}</b> (${D.group[d.j]})<br>z = ${f2(D.x[pi][d.j])}<br>contribution ${sgn(d.v)} log-odds`)).on('mouseleave', untip);
    F.layer('val').selectAll('text').data(bars).join('text').attr('x', (d) => F.x(Math.max(d.a, d.b)) + 5).attr('y', (d) => rowY(d.k) + 14).attr('font-size', 11).attr('fill', ink2())
      .text((d) => `${sgn(d.v)}  (z ${f2(D.x[pi][d.j])})`);
    F.layer('conn').selectAll('line').data(bars.slice(0, -1)).join('line').attr('x1', (d) => F.x(d.b)).attr('x2', (d) => F.x(d.b)).attr('y1', (d) => rowY(d.k) + 20).attr('y2', (d) => rowY(d.k + 1))
      .attr('stroke', muted()).attr('stroke-dasharray', '2 2');
    F.layer('ends').selectAll('circle').data([{ k: 0, x: m.base }, { k: P + 1, x: fx }]).join('circle').attr('cx', (d) => F.x(d.x)).attr('cy', (d) => rowY(d.k) + 10).attr('r', 6)
      .attr('fill', ink()).attr('stroke', css('--surface')).attr('stroke-width', 1.5);
    s1.set({ patient: D.ids[pi], phenotype: D.pheno[pi], 'actual outcome': D.y[pi] ? 'responder' : 'non-responder',
      'base probability': f2(sigmoid(m.base)), 'predicted probability': f2(sigmoid(fx)), 'sum of contributions': sgn(d3.sum(phi)) + ' log-odds' });
  }
  draw1();

  // ---------------- Figure 2: global importance vs local dependence
  const f2g = figure(main, 'Global importance vs local effects',
    'Left: global importance = mean |contribution| over all patients, for both models. Right: every patient\'s contribution for one gene against its expression (a dependence plot). A straight line means the model uses the gene linearly; steps and plateaus are the trees.');
  let gj = 1, mk2 = 'gb';
  select(f2g.controls, { label: 'Gene', value: String(gj), options: G.map((g, j) => [String(j), g]), onChange: (v) => { gj = +v; draw2(); } });
  select(f2g.controls, { label: 'Model', value: mk2, options: [['lin', MOD.lin.label], ['gb', MOD.gb.label]], onChange: (v) => { mk2 = v; draw2(); } });
  legend(f2g.root, [['logistic regression (left)', cat(0)], ['gradient boosting (left)', cat(1)]]);
  const s2 = stats(f2g.root);
  const row2 = el('div', { class: 'fig-row' }); f2g.body.append(row2);
  const L2 = el('div'), R2 = el('div'); row2.append(L2, R2);
  const imp = { lin: G.map((g, j) => d3.mean(MOD.lin.phi, (r) => Math.abs(r[j]))), gb: G.map((g, j) => d3.mean(MOD.gb.phi, (r) => Math.abs(r[j]))) };
  const gord = d3.range(P).sort((a, b) => imp.lin[b] + imp.gb[b] - imp.lin[a] - imp.gb[a]);
  const I = frame(L2, { width: 440, height: 380, x: [0, d3.max([...imp.lin, ...imp.gb]) * 1.1], y: [0, P], xLabel: 'mean |contribution| (log-odds)', grid: false, margin: { left: 70 } });
  I.g.select('.axis.y').remove();
  const bh = I.h / P;
  gord.forEach((j, r) => {
    ['lin', 'gb'].forEach((k, s) => {
      I.layer('b').append('rect').attr('x', 0).attr('y', r * bh + 4 + s * (bh - 8) / 2).attr('width', I.x(imp[k][j])).attr('height', (bh - 8) / 2 - 1).attr('rx', 2).attr('fill', cat(s))
        .on('mousemove', (e) => tip(e, `<b>${G[j]}</b> · ${MOD[k].label}<br>mean |contribution| ${f3(imp[k][j])}`)).on('mouseleave', untip)
        .on('click', () => { gj = j; f2g.controls.querySelector('select').value = String(j); draw2(); }).style('cursor', 'pointer');
    });
    I.layer('t').append('text').attr('class', 'gl').attr('data-j', j).attr('x', -6).attr('y', r * bh + bh / 2 + 4).attr('text-anchor', 'end').attr('font-size', 12).attr('fill', ink2()).text(G[j]);
  });
  const xr = d3.extent(D.x.flat()), allphi = [...MOD.lin.phi.flat(), ...MOD.gb.phi.flat()];
  const prng = Math.max(...allphi.map(Math.abs)) * 1.05;
  const S = frame(R2, { width: 440, height: 380, x: [Math.floor(xr[0]), Math.ceil(xr[1])], y: [-prng, prng], xLabel: 'expression (z-score)', yLabel: 'contribution (log-odds)' });
  S.layer('zero').append('line').attr('x1', 0).attr('x2', S.w).attr('y1', S.y(0)).attr('y2', S.y(0)).attr('stroke', muted());
  function draw2() {
    I.layer('t').selectAll('text.gl').attr('font-weight', function () { return +this.getAttribute('data-j') === gj ? 700 : 400; }).attr('fill', function () { return +this.getAttribute('data-j') === gj ? ink() : ink2(); });
    const pts = d3.range(n).map((i) => ({ i, x: D.x[i][gj], v: MOD[mk2].phi[i][gj], y: D.y[i] }));
    S.layer('p').selectAll('circle').data(pts).join('circle').attr('cx', (d) => S.x(d.x)).attr('cy', (d) => S.y(d.v)).attr('r', 4)
      .attr('fill', (d) => cat(d.y)).attr('stroke', css('--surface')).attr('stroke-width', 1)
      .on('mousemove', (e, d) => tip(e, `${D.ids[d.i]} (${D.pheno[d.i]}, ${d.y ? 'responder' : 'non-responder'})<br>${G[gj]} z = ${f2(d.x)}<br>contribution ${sgn(d.v)}`)).on('mouseleave', untip);
    const r = d3.range(n).map((i) => MOD[mk2].phi[i][gj]);
    s2.set({ gene: `${G[gj]} (${D.group[gj]})`, 'mean |contribution|': f3(imp[mk2][gj]), 'range across patients': `${sgn(d3.min(r))} to ${sgn(d3.max(r))}`,
      'patients pushed up / down': `${d3.sum(r, (v) => v > 0.01)} / ${d3.sum(r, (v) => v < -0.01)}` });
  }
  draw2();
  R2.append(el('div', { class: 'legend' }, el('span', {}, el('i', { style: `background:${cat(1)}` }), 'responder'), el('span', {}, el('i', { style: `background:${cat(0)}` }), 'non-responder')));

  // ---------------- Figure 3: correlated genes share credit
  const f3g = figure(main, 'Correlated genes share (and swap) credit',
    'Left: correlation between the panel genes across patients. Right: logistic-regression coefficients with all genes (filled) and after removing CXCL10 (outlined). The model needs the interferon signal, not any particular gene; which gene gets the credit is partly arbitrary.');
  const row3 = el('div', { class: 'fig-row' }); f3g.body.append(row3);
  const L3 = el('div'), R3 = el('div'); row3.append(L3, R3);
  heatmap(L3, D.corr, { cell: 40, color: diverging(1), rowLabels: G, colLabels: G, fmt: (v) => v.toFixed(1), labelW: 60, labelH: 60,
    onHover: (e, i, j, v) => tip(e, `${G[i]} vs ${G[j]}: r = ${f2(v)}`) });
  const cf = D.lin.coef, cd = D.lin.coef_drop_cxcl10;
  const cm = d3.max([...cf, ...cd.filter((v) => v != null)].map(Math.abs)) * 1.15;
  const Cf = frame(R3, { width: 440, height: 380, x: [-cm, cm], y: [0, P], xLabel: 'coefficient (log-odds per SD)', grid: false, margin: { left: 70 } });
  Cf.g.select('.axis.y').remove();
  const ch = Cf.h / P;
  Cf.layer('z').append('line').attr('x1', Cf.x(0)).attr('x2', Cf.x(0)).attr('y1', 0).attr('y2', Cf.h).attr('stroke', muted());
  G.forEach((g, j) => {
    Cf.layer('t').append('text').attr('x', -6).attr('y', j * ch + ch / 2 + 4).attr('text-anchor', 'end').attr('font-size', 12).attr('fill', ink2()).text(g);
    Cf.layer('b').append('rect').attr('x', Cf.x(Math.min(0, cf[j]))).attr('width', Math.abs(Cf.x(cf[j]) - Cf.x(0))).attr('y', j * ch + 5).attr('height', ch / 2 - 5).attr('fill', cat(0))
      .on('mousemove', (e) => tip(e, `<b>${g}</b><br>all genes: ${sgn(cf[j])}<br>without CXCL10: ${cd[j] == null ? '(removed)' : sgn(cd[j])}`)).on('mouseleave', untip);
    if (cd[j] != null) Cf.layer('b').append('rect').attr('x', Cf.x(Math.min(0, cd[j]))).attr('width', Math.abs(Cf.x(cd[j]) - Cf.x(0))).attr('y', j * ch + ch / 2).attr('height', ch / 2 - 5)
      .attr('fill', 'none').attr('stroke', cat(0)).attr('stroke-width', 2)
      .on('mousemove', (e) => tip(e, `<b>${g}</b><br>all genes: ${sgn(cf[j])}<br>without CXCL10: ${sgn(cd[j])}`)).on('mouseleave', untip);
  });

  // ---------------- prose
  const j9 = G.indexOf('CXCL9'), j10 = G.indexOf('CXCL10'), jk = G.indexOf('MKI67'), jf = G.indexOf('FAP');
  const topLin = gord[0];
  const r910 = D.corr[j9][j10];
  section(main, 'What to notice', `<ul>
    <li>Both panel models are only moderately accurate out of fold (AUC ${f3(D.lin.auc_oof)} logistic, ${f3(D.gb.auc_oof)} boosting). An explanation describes the model, including its mistakes: <b>Jump to → most confident false positive</b> and you see the interferon genes pushing a non-responder up, exactly as they do for a true responder.</li>
    <li>For the linear model each contribution is just coefficient × (z − background mean), so the dependence plot is a straight line. For boosting it is a staircase; trees also cap the effect at the extremes.</li>
    <li><b>MKI67</b> (proliferation) is a trap. The proliferation program is planted <i>independent</i> of response, yet MKI67 gets mean |contribution| ${f3(imp.lin[jk])} (logistic, coefficient ${sgn(cf[jk])}) and ${f3(imp.gb[jk])} (boosting). Why? MKI67 is made only by tumour cells, so its bulk level also tracks tumour purity, and immune-desert tumours are the purest. The model is right to use it as a proxy for "few immune cells"; reading the explanation as "proliferation blocks response" would be wrong.</li>
    <li>CXCL9 and CXCL10 correlate at r = ${f2(r910)}. In the last figure, dropping CXCL10 changes the CXCL9 coefficient from ${sgn(cf[j9])} to ${sgn(cd[j9])}. A large attribution for one gene in a correlated module is not evidence that this gene, rather than its partners, matters.</li>
    <li>FAP (CAF / exclusion marker) contributes ${cf[jf] < 0 ? 'negatively' : 'positively'} in the linear model (coefficient ${sgn(cf[jf])}), consistent with T-cell exclusion lowering response, but the model cannot tell exclusion from a lower tumour fraction.</li>
  </ul>`);
  steps(main, 'How it works', [
    'Pick a <b>background</b> (reference) population; here ' + D.n_bg + ' random discovery patients. The base value is the mean model output over the background, in log-odds.',
    'For a patient x and a subset S of genes, define v(S) = the average model output when genes in S take the patient\'s values and the other genes take each background patient\'s values.',
    'The <b>Shapley value</b> of gene j is its marginal effect v(S ∪ {j}) − v(S), averaged over all subsets S with the classic Shapley weights. With ' + P + ' genes that is 2<sup>' + P + '</sup> = ' + (2 ** P) + ' subsets, small enough to compute exactly; SHAP libraries approximate this for larger models (KernelSHAP) or compute it fast for trees (TreeSHAP).',
    'The values add up exactly: base + Σ contributions = this patient\'s output (the "sum of contributions" readout).',
    'For a linear model with this background the Shapley value reduces to β<sub>j</sub>·(x<sub>j</sub> − x̄<sub>j</sub>): no sampling needed.',
    'Global importance is a summary of local values (mean |contribution|). It says how much the model uses a gene, not in which direction or why.',
  ]);
  math(main, 'φ_j(x) = Σ_{S ⊆ F∖{j}}  |S|! (|F|−|S|−1)! / |F|!  ·  [ v(S ∪ {j}) − v(S) ]\nv(S)   = mean_b f(x_S, background_b,−S)        f = log-odds\nf(x)   = base + Σ_j φ_j(x)                       linear model:  φ_j = β_j (x_j − x̄_j)');
  callout(main, '', 'In biology', 'The <code>shap</code> Python package (TreeExplainer for XGBoost/LightGBM/random forests, KernelExplainer for anything) is the standard; in R, <code>fastshap</code>, <code>kernelshap</code> and <code>DALEX</code>. In omics it is used to show which genes drive a classifier for one tumour (per-patient reports in molecular tumour boards), to sanity-check that a response model uses immune genes rather than a batch gene, and to rank features in single-cell classifiers. Integrated gradients plays the same role for deep models (e.g. on gene-expression foundation models).');
  callout(main, 'warn', 'Pitfalls', `<ul>
    <li><b>Explanation ≠ causation.</b> A positive contribution from CXCL9 means the model's output rises with CXCL9 in this data, not that raising CXCL9 would make a tumour respond. MKI67 shows the reverse trap: a large attribution driven by tumour purity, not by proliferation.</li>
    <li><b>Correlated features</b> split or swap credit arbitrarily (r = ${f2(r910)} for CXCL9/CXCL10 here). Interpret modules or gene sets, not single genes.</li>
    <li><b>Background choice changes the numbers.</b> Contributions are relative to "an average patient" from the chosen background; a different cohort or centre gives a different base and different contributions.</li>
    <li><b>Interventional vs conditional Shapley</b>: replacing one gene with a background value creates gene combinations that never occur biologically (CXCL9 high, CXCL10 low). That is a known limitation of all SHAP-style methods.</li>
    <li><b>In-sample explanations</b> of an overfit model will "explain" noise convincingly. Check out-of-fold performance first.</li>
  </ul>`);
  seeAlso(main, [['concepts', 'interpretability', 'Feature importance & SHAP'], ['omics', 'bulk-model-compare', 'Model comparison'], ['omics', 'bulk-feature-selection', 'Feature selection'], ['omics', 'bulk-survival', 'Next: survival & risk scores']]);
}
