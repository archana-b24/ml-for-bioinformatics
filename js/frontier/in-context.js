// In-context learning on cells (Stack, Arc Institute 2026): condition on a prompt of example cells from a new
// context and predict that context's response. Teaching toy: Bayesian-style attention over a library of known
// responses, blended with the plain prompt average, compared with a zero-shot mean shift.
import { header, figure, slider, select, button, stats, section, steps, math, legend, seeAlso, tip, untip, el } from '../lib/ui.js';
import { frame, cat, css, ink, muted } from '../lib/plot.js';
import { rng, range } from '../lib/data.js';
import { sourceCard, toyBadge, coreIdea, realityCheck, useIt, cite, colMeans } from './_shared.js';
const d3 = window.d3;

const G = 20, C = 24, SIG = 1.0, TZ = 0.35;
const KS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64];
const FAM = [[2.2, 0.5], [-1.4, 1.9], [-0.7, -2.2]];
const FAM_NAMES = ['family A', 'family B', 'family C'];
const NOVEL = [2.3, -2.4], OFF = 3.0;   // novel responses also move genes outside the training plane
const T_TEST = 50, R_DRAW = 10;

const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
const sq = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2; return s; };
const rmse = (a, b) => Math.sqrt(sq(a, b) / a.length);

/** The fixed "pretraining" world: two gene-loading axes and a library of 24 training contexts in 3 response families. */
function makeWorld() {
  const r = rng(17);
  const a = range(G).map(() => r.normal()), b = range(G).map(() => r.normal());
  const na = Math.hypot(...a); a.forEach((v, i) => { a[i] = v / na; });
  const ab = dot(a, b); b.forEach((v, i) => { b[i] = v - ab * a[i]; });
  const nb = Math.hypot(...b); b.forEach((v, i) => { b[i] = v / nb; });
  const toGene = (z) => a.map((v, g) => z[0] * v + z[1] * b[g]);
  const lib = range(C).map((i) => { const f = FAM[i % 3]; const z = [r.normal(f[0], TZ), r.normal(f[1], TZ)]; return { z, fam: i % 3, d: toGene(z) }; });
  const mbar = colMeans(lib.map((l) => l.d));
  // a gene direction outside the plane of all training responses: only "novel" contexts use it
  const o = range(G).map(() => r.normal());
  const oa = dot(o, a), ob = dot(o, b); o.forEach((v, i) => { o[i] = v - oa * a[i] - ob * b[i]; });
  const no = Math.hypot(...o); o.forEach((v, i) => { o[i] = v / no; });
  return { a, b, o, toGene, lib, mbar, proj: (v) => [dot(a, v), dot(b, v)] };
}

function testDelta(W, r, scen) {
  const f = scen === 'novel' ? NOVEL : FAM[r.int(3)];
  const d = W.toGene([r.normal(f[0], TZ), r.normal(f[1], TZ)]);
  return scen === 'novel' ? d.map((v, g) => v + OFF * W.o[g]) : d;
}

/** In-context predictor: attention over library responses keyed by the prompt mean, then a data-driven blend with the prompt mean. */
function icl(W, m, k) {
  const tg2 = (TZ * TZ * 2) / G;
  const s2 = (SIG * SIG) / k + tg2;
  const logits = W.lib.map((l) => -sq(m, l.d) / (2 * s2));
  const mx = Math.max(...logits);
  const e = logits.map((v) => Math.exp(v - mx)); const Z = e.reduce((s, v) => s + v, 0);
  const w = e.map((v) => v / Z);
  const ret = new Array(G).fill(0);
  W.lib.forEach((l, c) => { for (let g = 0; g < G; g++) ret[g] += w[c] * l.d[g]; });
  const resid = Math.max(tg2, sq(m, ret) / G - (SIG * SIG) / k);   // how much of the prompt the library fails to explain
  const k0 = (SIG * SIG) / resid;                                   // pseudo-count of trust in the library
  const pred = ret.map((v, g) => (k0 * v + k * m[g]) / (k0 + k));
  return { pred, w, ret, k0 };
}

/** Average RMSE per gene over many test contexts and prompt draws, for every k. */
function curves(W, scen) {
  const r = rng(scen === 'novel' ? 202 : 101);
  const out = { zero: [], prompt: [], icl: [] };
  const deltas = range(T_TEST).map(() => testDelta(W, r, scen));
  KS.forEach((k) => {
    let z = 0, p = 0, q = 0;
    deltas.forEach((D) => {
      for (let t = 0; t < R_DRAW; t++) {
        const s = SIG / Math.sqrt(k);
        const m = D.map((v) => v + r.normal(0, s));                 // the mean of k cells' shifts is exactly N(Δ, σ²/k)
        z += rmse(W.mbar, D); p += rmse(m, D); q += rmse(icl(W, m, k).pred, D);
      }
    });
    const n = T_TEST * R_DRAW;
    out.zero.push(z / n); out.prompt.push(p / n); out.icl.push(q / n);
  });
  return out;
}

export default async function render(main) {
  header(main, 'In-context learning on cells (Stack)',
    'Instead of retraining a model for every new cell line or donor, show it a <b>prompt: a handful of example cells from the new context</b>, and let it predict the rest. A model that has seen many contexts can read a noisy prompt the way you read a few words of a familiar song: it <b>recognises which known pattern the prompt resembles</b>.');

  sourceCard(main, {
    org: 'Arc Institute',
    when: 'January 2026',
    links: ['stackNews', 'stackGit'],
    claim: 'Stack is a tabular transformer that attends both across genes and across cells, and does in-context learning from a prompt set of cells: it predicts cells for a new condition without fine-tuning. Reported wins over other approaches are Arc\'s own.',
    status: 'no independent benchmark in our brief',
  });

  coreIdea(main, `<p>Classical models learn a fixed function: you train on contexts A, B, C and then apply the same weights to context D. <b>In-context learning</b> (ICL) adds a second input, the <b>prompt</b>: a small set of cells measured in context D under the new condition. The model's weights stay frozen; what changes from task to task is only the prompt it conditions on. Large language models made this idea famous: they "learn" a task from a few examples in the prompt.</p>
  <p>For cells, the prompt is a <i>set</i>, so the model must be indifferent to cell order. Stack does this with attention across cells as well as across genes. Why should a prompt beat simply averaging the prompt cells? Because single cells are noisy. With 3 example cells the plain average is mostly noise, but a model that has seen many contexts during pretraining can match the noisy prompt against the <b>library of responses it has absorbed</b> and fill in the rest.</p>
  <p><b>Key ML ingredient:</b> <b>conditioning on a set of examples at inference time</b> (attention over a prompt), learned by pretraining across many contexts. Our toy keeps only that logic: attention weights over 24 known context responses, keyed by the prompt.</p>`);

  // ---------------------------------------------------------------- figure
  const W = makeWorld();
  const fig = figure(main, 'Predict a new context\'s response from k prompt cells',
    `Mock data: each context has a response Δ, a shift in 20 genes. The 24 training contexts fall into three response families. A prompt cell's shift is the true Δ plus cell noise (σ = ${SIG} per gene). Left: the 2-D plane that holds every training family (all 20 genes are used in the maths; a novel response also moves genes off this plane). Right: error averaged over ${T_TEST} new contexts × ${R_DRAW} prompt draws.`);
  toyBadge(fig, 'Attention over a fixed library stands in for a pretrained transformer.');
  const S = { kIdx: 2, scen: 'seen', seed: 1 };
  const kSl = slider(fig.controls, { label: 'Prompt cells k', min: 0, max: KS.length - 1, value: S.kIdx, fmt: (i) => KS[i], onInput: (v) => { S.kIdx = v; paint(); } });
  select(fig.controls, { label: 'New context', value: S.scen, options: [['seen', 'resembles a training family'], ['novel', 'genuinely novel response']], onChange: (v) => { S.scen = v; newContext(); CUR = curves(W, S.scen); paint(); } });
  button(fig.controls, 'Draw another context', () => { S.seed += 1; newContext(); paint(); }, true);
  const readout = stats(fig.root);

  const row = el('div', { class: 'fig-row' }); fig.body.append(row);
  const left = el('div'), right = el('div'); row.append(left, right);
  legend(left, [['training contexts (size = attention)', cat(0)], ['prompt cells', cat(1)], ['zero-shot mean shift', cat(3)], ['prompt average', cat(4)], ['in-context prediction', cat(2)], ['true response', ink()]]);
  const F = frame(left, { width: 480, height: 420, x: [-4.2, 4.2], y: [-4.2, 4.2], xLabel: 'response axis 1', yLabel: 'response axis 2' });
  legend(right, [['zero-shot mean shift', cat(3)], ['prompt average', cat(4)], ['in-context', cat(2)]]);
  const P = frame(right, { width: 480, height: 420, x: [1, 64], y: [0, 1.15], xScale: d3.scaleLog(), xLabel: 'prompt cells k (log scale)', yLabel: 'RMSE per gene (lower is better)' });
  P.svg.selectAll('.axis.x .tick text').text((d) => ([1, 2, 4, 8, 16, 32, 64].includes(d) ? d : ''));

  let D, cells, CUR = curves(W, S.scen);
  function newContext() {
    const r = rng(1000 * S.seed + (S.scen === 'novel' ? 7 : 3));
    D = testDelta(W, r, S.scen);
    cells = range(64).map(() => D.map((v) => v + r.normal(0, SIG)));   // first k of these form the prompt
  }
  newContext();

  function paint() {
    const k = KS[S.kIdx];
    const m = colMeans(cells.slice(0, k));
    const res = icl(W, m, k);
    const P2 = (v) => W.proj(v);
    const wMax = Math.max(...res.w);
    F.layer('lib').selectAll('circle').data(W.lib).join('circle')
      .attr('cx', (d) => F.x(d.z[0])).attr('cy', (d) => F.y(d.z[1]))
      .attr('r', (d, i) => 4 + 9 * Math.sqrt(res.w[i] / wMax)).attr('fill', cat(0)).attr('fill-opacity', (d, i) => 0.25 + 0.75 * res.w[i] / wMax)
      .attr('stroke', css('--surface')).attr('stroke-width', 1.2)
      .on('mousemove', (e, d) => tip(e, `training context, ${FAM_NAMES[d.fam]}<br>attention weight ${(res.w[W.lib.indexOf(d)] * 100).toFixed(1)}%`)).on('mouseleave', untip);
    F.layer('cells').selectAll('circle').data(cells.slice(0, k).map(P2)).join('circle')
      .attr('cx', (d) => F.x(d[0])).attr('cy', (d) => F.y(d[1])).attr('r', 4).attr('fill', cat(1)).attr('opacity', 0.55)
      .attr('stroke', css('--surface')).attr('stroke-width', 1);
    const marks = [
      { name: 'zero-shot mean shift', v: W.mbar, c: cat(3), shape: d3.symbolSquare },
      { name: 'prompt average', v: m, c: cat(4), shape: d3.symbolCross },
      { name: 'in-context prediction', v: res.pred, c: cat(2), shape: d3.symbolDiamond },
    ].map((o) => ({ ...o, p: P2(o.v), err: rmse(o.v, D) }));
    const tp = P2(D);
    F.layer('truth').selectAll('circle').data([tp]).join('circle').attr('cx', (d) => F.x(d[0])).attr('cy', (d) => F.y(d[1])).attr('r', 9)
      .attr('fill', 'none').attr('stroke', ink()).attr('stroke-width', 2);
    F.layer('marks').selectAll('path').data(marks).join('path')
      .attr('d', (d) => d3.symbol(d.shape, 150)()).attr('transform', (d) => `translate(${F.x(d.p[0])},${F.y(d.p[1])})`)
      .attr('fill', (d) => d.c).attr('stroke', css('--surface')).attr('stroke-width', 1.5)
      .on('mousemove', (e, d) => tip(e, `${d.name}<br>RMSE per gene vs truth: ${d.err.toFixed(3)}`)).on('mouseleave', untip);

    // curves
    const series = [['zero', cat(3)], ['prompt', cat(4)], ['icl', cat(2)]];
    P.layer('lines').selectAll('path').data(series).join('path').attr('fill', 'none').attr('stroke-width', 2).attr('stroke', (d) => d[1])
      .attr('d', (d) => d3.line().x((_, i) => P.x(KS[i])).y((v) => P.y(Math.min(v, 1.15)))(CUR[d[0]]));
    P.layer('pts').selectAll('circle').data(series.flatMap(([key, c]) => CUR[key].map((v, i) => ({ key, c, v, k: KS[i] })))).join('circle')
      .attr('cx', (d) => P.x(d.k)).attr('cy', (d) => P.y(Math.min(d.v, 1.15))).attr('r', 4).attr('fill', (d) => d.c).attr('stroke', css('--surface')).attr('stroke-width', 1.2)
      .on('mousemove', (e, d) => tip(e, `${{ zero: 'zero-shot mean shift', prompt: 'prompt average', icl: 'in-context' }[d.key]}, k = ${d.k}<br>mean RMSE ${d.v.toFixed(3)}`)).on('mouseleave', untip);
    P.layer('cursor').selectAll('line').data([k]).join('line').attr('x1', (d) => P.x(d)).attr('x2', (d) => P.x(d)).attr('y1', 0).attr('y2', P.h)
      .attr('stroke', muted()).attr('stroke-dasharray', '4 3').attr('stroke-width', 1.5);

    const top = res.w.indexOf(wMax);
    readout.set({
      'this context: zero-shot / prompt avg / in-context': marks.map((o) => o.err.toFixed(2)).join(' / '),
      [`average over ${T_TEST * R_DRAW} draws at k = ${k}`]: [CUR.zero[S.kIdx], CUR.prompt[S.kIdx], CUR.icl[S.kIdx]].map((v) => v.toFixed(2)).join(' / '),
      'top attention': `${(wMax * 100).toFixed(0)}% on a ${FAM_NAMES[W.lib[top].fam]} context`,
      'trust in library (pseudo-cells k₀)': res.k0 > 999 ? '>999' : res.k0.toFixed(0),
    });
  }
  paint();
  kSl.set(S.kIdx);

  section(main, 'What to notice', `<ul>
    <li>With <b>k = 1–4</b> prompt cells and a context that resembles a training family, the plain prompt average (cross) is far from the truth (ring): single-cell noise dominates. The in-context prediction (diamond) jumps to the right family because the attention puts most of its weight on that family's contexts. On the right, the green curve sits below both baselines across the small-k range.</li>
    <li>The zero-shot mean shift (square) ignores the prompt, so its curve is flat: it predicts the average of all training contexts, which is a response no family actually has.</li>
    <li>As k grows, the prompt average improves as 1/√k and the gap narrows; the in-context advantage is largest in the <b>few-shot regime</b>. Part of it is a toy artefact you should know about: every training response lives in one 2-D gene-program plane, so the attention-based prediction ignores noise in the other 18 directions. Real responses are not this low-dimensional.</li>
    <li>Switch to <b>genuinely novel response</b>: the new context moves genes outside the plane the training contexts span (invisible on the left, but counted in the error). The attention still lands on the nearest family, but the residual check sees that the library cannot explain the prompt, the "trust in library" readout drops to a few pseudo-cells or fewer, and the prediction falls back on the prompt average. The green curve now stays only modestly below the pink one at k = 1–2 and merges with it by k ≈ 8: in-context learning can only borrow from what pretraining covered, so on new biology it is roughly as good as the noisy prompt and no better.</li>
    <li>Press <b>Draw another context</b> a few times: single draws scatter around the averaged curves, which is why one impressive example proves little.</li>
  </ul>`);

  steps(main, 'How it works', [
    'Pretraining (here: given) exposes the model to many contexts, so it effectively stores a library of how responses look. In our toy that library is 24 explicit response vectors; in Stack it is implicit in the transformer weights.',
    'At inference, take k prompt cells from the new context under the new condition, and compute each cell\'s shift from that context\'s control cells.',
    'Summarise the prompt (here: its mean; in a transformer: attention across the prompt cells, which is permutation-invariant).',
    'Attend over the library: weight each known response by how well it explains the prompt, with a temperature that shrinks as k grows (more cells, less noise, sharper attention).',
    'Check the residual: if the library cannot explain the prompt, lower the trust k₀ and lean on the prompt itself. Output the blend.',
  ]);
  math(main, 'm = (1/k) Σ shifts of prompt cells                        (prompt average)\nw_c ∝ exp( −‖m − Δ_c‖² / 2(σ²/k + τ²) )                   (attention over library)\nr = Σ w_c Δ_c ,   k₀ = σ² / max(τ², ‖m − r‖²/G − σ²/k)\nprediction = (k₀ r + k m) / (k₀ + k)');

  realityCheck(main, `<ul>
    <li>The claims that Stack predicts unseen conditions well come from ${cite('stackNews', 'Arc\'s announcement')} and ${cite('stackGit', 'repository')}. Our brief lists <b>no independent benchmark of Stack</b> yet.</li>
    <li>Independent perturbation benchmarks keep finding that simple baselines are hard to beat: in ${cite('ahlmann', 'Ahlmann-Eltze et al. 2025')} nothing beat the training mean on unseen single perturbations, and ${cite('vcbench', 'VCBench 2026')} reports simple baselines matching or beating foundation models on 4 of 5 axes. Any ICL claim should be checked against the zero-shot mean and the plain prompt average shown here.</li>
    <li>"Novel context" is the hard case, and it is the one that matters for discovery. The ${cite('vcc2026', 'Virtual Cell Challenge 2026')} is zero-shot on 6 new cell lines; winners were not announced as of our brief (2026-11-12).</li>
    <li>Leakage is easy: if prompt cells and evaluation cells share a well, donor or batch, the prompt carries batch signal that inflates the score. Split by context, not by cell.</li>
  </ul>`);

  useIt(main, `<ul>
    <li>If you have a few cells from a new donor or line under a condition, report three numbers: zero-shot (training mean shift), prompt average, and the ICL model. The model has earned its place only where it beats both.</li>
    <li>Plot error against prompt size k, as on the right. If the ICL advantage vanishes at the k you can actually afford, use the simpler estimate.</li>
    <li>Ask whether your new context resembles the pretraining corpus (tissue, species, assay). If not, expect the "novel" behaviour above.</li>
  </ul>`);

  seeAlso(main, [['frontier', 'state-sets', 'State: predicting distributions'], ['frontier', 'perturbation-arena', 'Perturbation baselines arena'], ['frontier', 'zero-shot-check', 'Do foundation embeddings beat PCA?'],
    ['concepts', 'attention', 'Attention'], ['concepts', 'transformer', 'Transformers'], ['concepts', 'probability-mle', 'Likelihood & Bayes'], ['concepts', 'foundation-models', 'Foundation models']]);
}
