// Tab 2 · Single-cell step 5: cell-type annotation. Marker scores vs supervised label transfer from a reference; the missing-type trap.
import { header, figure, slider, select, stats, section, steps, callout, math, legend, seeAlso, el, tip, untip } from '../lib/ui.js';
import { frame, cat, css, muted, sequential } from '../lib/plot.js';
import { stepCard, omicsData } from './_shared.js';
import { canvasScatter, CT_COLOR, ext, f2, pct } from './_sc_common.js';
const d3 = window.d3;

const METHODS = [
  ['marker', 'Marker-gene scores (no training)'],
  ['lt_full', 'Label transfer, full reference'],
  ['lt_notreg', 'Label transfer, reference without Tregs'],
];

export default async function render(main) {
  header(main, 'Cell-type annotation',
    'Naming clusters used to be done by eye from marker genes. Now it is often a <b>supervised classifier trained on a labelled reference atlas</b>. That is faster and more consistent, but a classifier can only choose among the labels it was trained on, and it will do so <b>confidently</b> even for cells it has never seen.');
  stepCard(main, {
    stage: 'Single-cell · step 5 of 7',
    question: 'What is each cell, in the vocabulary other studies use (CD8 T, Treg, CAF…)?',
    data: 'Normalised expression of our query cells, plus an independently generated, labelled reference dataset (6 other patients, another chemistry).',
    method: 'Two approaches: (1) per-cell marker scores (mean z-scored expression of known markers; pick the highest); (2) multinomial logistic regression trained on the reference and applied to every query cell (CellTypist-style label transfer).',
    verdict: 'ml',
    why: 'Label transfer is textbook supervised classification, with the textbook failure modes: distribution shift between reference and query, and classes that are missing from training. Marker scores need no training but depend entirely on a good marker list.',
  });
  const [A, C] = await Promise.all([omicsData(main, 'sc_annotation.json'), omicsData(main, 'sc_cells.json')]);
  if (!A || !C) return;
  const CT = A.cell_types, NT = CT.length, TREG = CT.indexOf('Treg');
  const n = C.n;

  // ---------------- figure 1 ----------------
  const fig = figure(main, 'Predicted labels, confidence and the confusion matrix',
    `Left: integrated UMAP of ${n.toLocaleString()} cells coloured by predicted label; cells below the confidence threshold are drawn as hollow grey rings ("unassigned"). Right: confusion matrix over all ${A.methods.marker.confusion.flat().reduce((s, v) => s + v, 0).toLocaleString()} kept cells: rows are the simulated truth (last row: doublets that survived QC), columns are predicted labels, shading is the share of the row.`);
  let meth = 'lt_full', thr = 0.5;
  select(fig.controls, { label: 'Method', value: meth, options: METHODS, onChange: (v) => { meth = v; paint(); } });
  slider(fig.controls, { label: 'Confidence threshold', min: 0, max: 0.95, step: 0.05, value: thr, fmt: (v) => v.toFixed(2), onInput: (v) => { thr = v; paint(); } });
  const rd = stats(fig.root);
  const row = el('div', { class: 'fig-row' }); fig.body.append(row);
  const pA = el('div'), pB = el('div'); row.append(pA, pB);
  const U = C.umap.harmony;
  const S = canvasScatter(pA, { width: 460, height: 420, x: ext(U[0]), y: ext(U[1]), axes: false, grid: false, margin: { top: 6, right: 6, bottom: 6, left: 6 },
    hover: (i) => `true: <b>${CT[C.type[i]] || 'doublet (missed)'}</b><br>predicted: <b>${CT[C.ann[meth][i]]}</b> · confidence ${C.conf[meth][i].toFixed(2)}` });
  legend(pA, [...CT.map((t, k) => [t, CT_COLOR(k)]), ['below threshold', muted()]]);
  const box = el('div'); pB.append(box);

  function paint() {
    const pred = C.ann[meth], conf = C.conf[meth];
    S.draw({ idx: d3.range(n).sort((a, b) => conf[b] - conf[a]), x: U[0], y: U[1], color: (i) => (conf[i] < thr ? muted() : CT_COLOR(pred[i])), r: (i) => (conf[i] < thr ? 3.2 : 2.2), alpha: 0.85 });
    // confusion
    box.innerHTML = '';
    const Mx = A.methods[meth].confusion, rows = A.rows;
    const cell = 34, left = 120, top = 86;
    const svg = d3.select(box).append('svg').attr('viewBox', `0 0 ${left + NT * cell + 40} ${top + rows.length * cell + 8}`).attr("width", left + NT * cell + 40);
    const sq = sequential(0, 1);
    Mx.forEach((r, i) => {
      const tot = d3.sum(r) || 1;
      svg.append('text').attr('x', left - 8).attr('y', top + i * cell + cell / 2 + 4).attr('text-anchor', 'end').attr('font-size', 11).attr('fill', css('--ink-2')).text(rows[i]);
      r.forEach((v, j) => {
        const f = v / tot;
        svg.append('rect').attr('x', left + j * cell).attr('y', top + i * cell).attr('width', cell - 2).attr('height', cell - 2).attr('rx', 2).attr('fill', sq(f))
          .attr('stroke', i === j ? css('--ink') : 'none').attr('stroke-width', i === j ? 1 : 0)
          .on('mousemove', (e) => tip(e, `true <b>${rows[i]}</b> → predicted <b>${CT[j]}</b>: ${v} cells (${pct(f)} of row)`)).on('mouseleave', untip);
        if (v > 0) svg.append('text').attr('x', left + j * cell + cell / 2 - 1).attr('y', top + i * cell + cell / 2 + 3).attr('text-anchor', 'middle').attr('font-size', 9).attr('pointer-events', 'none')
          .attr('fill', f > 0.55 ? css('--surface') : css('--ink')).text(v);
      });
    });
    CT.forEach((t, j) => svg.append('text').attr('transform', `translate(${left + j * cell + cell / 2 + 3},${top - 6}) rotate(-55)`).attr('font-size', 11).attr('fill', css('--ink-2')).text(t));
    const m = A.methods[meth];
    let low = 0, lowT = 0, lowD = 0, nT = 0, nD = 0;
    for (let i = 0; i < n; i++) {
      if (C.type[i] === TREG) nT++; if (C.type[i] < 0) nD++;
      if (conf[i] < thr) { low++; if (C.type[i] === TREG) lowT++; if (C.type[i] < 0) lowD++; }
    }
    rd.set({ 'accuracy (true singlets)': pct(m.acc, 1), 'Treg recall': pct(m.recall.Treg), 'unassigned (shown cells)': `${low} / ${n}`, 'of Tregs shown': `${lowT} / ${nT} unassigned`, 'of missed doublets shown': `${lowD} / ${nD} unassigned` });
    paintHist();
  }

  // ---------------- figure 2: confidence distributions ----------------
  const fig2 = figure(main, 'Does the classifier know when it does not know?',
    'Distribution of the top-class confidence for the selected method (shared controls above), split into ordinary singlets, true Tregs, and doublets that survived QC. Each group is scaled to its own total.');
  const H = frame(fig2.body, { width: 760, height: 260, x: [0, 1], y: [0, 1], xLabel: 'confidence of the predicted label', yLabel: 'share of group' });
  const groups = [['singlets (not Treg)', 'hist_singlet', muted()], ['true Tregs', 'hist_treg', cat(TREG)], ['doublets (missed by QC)', 'hist_doublet', cat(7)]];
  legend(fig2.body, [...groups.map(([l, , c]) => [l, c]), ['confidence threshold', css('--ink')]]);
  function paintHist() {
    const m = A.methods[meth], bins = A.conf_bins;
    const data = groups.map(([l, k, c]) => { const h = m[k], tot = d3.sum(h) || 1; return { l, c, v: h.map((x) => x / tot) }; });
    H.y.domain([0, d3.max(data.flatMap((d) => d.v)) * 1.05]);
    H.g.select('.axis.y').call(d3.axisLeft(H.y).ticks(4, '.0%'));
    const bw = (H.x(bins[1]) - H.x(bins[0])) / 3;
    H.layer('bars').selectAll('g').data(data).join('g').attr('fill', (d) => d.c)
      .selectAll('rect').data((d, gi) => d.v.map((v, b) => ({ v, b, gi, l: d.l }))).join('rect')
      .attr('x', (d) => H.x(bins[d.b]) + d.gi * bw).attr('width', bw - 0.5).attr('y', (d) => H.y(d.v)).attr('height', (d) => H.h - H.y(d.v))
      .on('mousemove', (e, d) => tip(e, `${d.l}: ${pct(d.v, 1)} with confidence ${bins[d.b].toFixed(2)}–${bins[d.b + 1].toFixed(2)}`)).on('mouseleave', untip);
    H.layer('thr').selectAll('line').data([thr]).join('line').attr('x1', (d) => H.x(d)).attr('x2', (d) => H.x(d)).attr('y1', 0).attr('y2', H.h).attr('stroke', css('--ink')).attr('stroke-width', 2).attr('stroke-dasharray', '5 4');
  }
  paint();

  const mk = A.methods.marker, lf = A.methods.lt_full, ln = A.methods.lt_notreg;
  const tregRow = ln.confusion[TREG], tregTot = d3.sum(tregRow);
  const tregTop = d3.range(NT).sort((a, b) => tregRow[b] - tregRow[a]).slice(0, 3).map((j) => `${CT[j]} (${tregRow[j]})`).join(', ');
  section(main, 'What to notice', `<ul>
    <li>With a <b>full reference</b>, label transfer gets ${pct(lf.acc, 1)} of true singlets right, and so do plain marker scores (${pct(mk.acc, 1)}). That is a property of this toy: the simulated types have strong, clean markers. Real benchmarks show more errors between close types (CD4 vs CD8 T, macrophage subsets).</li>
    <li>Switch to the <b>reference without Tregs</b>. The classifier has no "Treg" option, so all ${tregTot} Tregs get some other label: ${tregTop}. Accuracy on all singlets barely moves (${pct(ln.acc, 1)}) because Tregs are rare, which is exactly why the error is easy to miss.</li>
    <li>The saving grace is confidence: Tregs now have median confidence ${f2(ln.conf_treg)} (vs ${f2(lf.conf_treg)} with the full reference), so a threshold around 0.5 sends most of them to "unassigned". Move the slider and watch the Treg bars in the second figure.</li>
    <li>Doublets that slipped through QC are classified <b>confidently</b> (median ${f2(lf.conf_doublet)} for label transfer, ${f2(mk.conf_doublet)} for marker scores). A softmax over known classes is not an out-of-distribution detector; confidence thresholds catch novel types only when they happen to fall between classes.</li>
  </ul>`);
  steps(main, 'How it works', [
    '<b>Marker scores</b>: z-score every gene across cells, average the z-scores of each cell type\'s marker list, assign the type with the highest score. (The "confidence" shown for markers is a softmax of those scores, an ad-hoc number.)',
    `<b>Label transfer</b>: take a labelled reference, restrict both datasets to shared informative genes (our ${A.n_features} HVGs), scale each dataset separately to reduce batch differences.`,
    `Train a regularised multinomial logistic regression on the ${A.n_ref.toLocaleString()} reference cells (in the "without Tregs" version the ${A.n_ref_treg} reference Tregs are removed).`,
    'Predict class probabilities for each query cell; the label is the argmax and the confidence is its probability.',
    'Leave low-confidence cells unassigned, then check the result against marker genes and clusters. Annotation should be reviewed, not just accepted.',
  ]);
  math(main, `P(type = c | x) = exp(w_cᵀx + b_c) / Σ_c' exp(w_c'ᵀx + b_c')      (softmax over the reference's classes only)
marker score_c(x) = mean_{g ∈ markers(c)} z_g(x)`);
  callout(main, '', 'In biology', `<b>CellTypist</b> (logistic-regression models trained on immune atlases, essentially this page's method at scale), <b>Azimuth</b> / Seurat reference mapping, <b>scANVI</b> and <b>SingleR</b> (correlation to reference profiles) are standard. Foundation models (<b>scGPT</b>, <b>Geneformer</b>) can be fine-tuned for annotation, but independent zero-shot evaluations have found their embeddings often do not beat simple HVG or scVI baselines; see <a href="#/frontier/zero-shot-check">the zero-shot check</a> and <a href="#/frontier/llm-annotation">LLM-assisted annotation</a> on Tab 3. We did not run a foundation model on this toy.`);
  callout(main, 'warn', 'Pitfalls', `<ul>
    <li><b>Missing classes</b>: tumour cells, rare or disease-specific states absent from a healthy reference are forced into the nearest known label.</li>
    <li><b>Distribution shift</b>: reference from blood, query from tumour; different chemistries or tissues lower accuracy and calibration.</li>
    <li><b>Overall accuracy hides rare-type failure</b>: report per-class recall (here, Treg recall) rather than one number.</li>
    <li><b>Annotating doublets</b>: a confident label does not make a cell real; check doublet scores and co-expression of incompatible markers (e.g. EPCAM with PTPRC).</li>
  </ul>`);
  seeAlso(main, [['omics', 'sc-trajectory', 'Next: trajectories & pseudotime'], ['concepts', 'logistic-regression', 'Logistic regression'], ['concepts', 'calibration', 'Calibration'], ['frontier', 'geneformer', 'Geneformer'], ['concepts', 'imbalance', 'Class imbalance']]);
}
