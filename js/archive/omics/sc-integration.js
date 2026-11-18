// Tab 2 · Single-cell step 4: batch integration (Harmony), and how to tell correction from over-correction.
import { header, figure, select, stats, section, steps, callout, math, legend, seeAlso, el, tip, untip } from '../../lib/ui.js';
import { cat, css, muted, sequential } from '../../lib/plot.js';
import { stepCard, omicsData } from '../../omics/_shared.js';
import { canvasScatter, CT_COLOR, ext, f2, pct } from '../../omics/_sc_common.js';
const d3 = window.d3;

const METHODS = [
  ['uncorrected', 'No correction'],
  ['harmony', 'Harmony (batch = chemistry)'],
  ['over', 'Over-corrected (batch = patient, strong)'],
];

export default async function render(main) {
  header(main, 'Batch integration',
    'Cells from the two chemistries sit apart even when they are the same cell type. Integration moves them together, but the goal is not "maximum mixing": it is <b>mixing technical batches while keeping real biological differences</b>, and the metrics that reward the first do not see the second.');
  stepCard(main, {
    stage: 'Single-cell · step 4 of 7',
    question: 'Can we compare CD8 T cells from a 3′ v2 patient with CD8 T cells from a 3′ v3 patient, without the chemistry dominating?',
    data: 'The 20-PC embedding from step 2, plus each cell\'s batch label (chemistry) and patient.',
    method: 'Harmony: soft k-means clustering with a penalty for batch-pure clusters, then a per-cluster linear correction, iterated. Scored with kNN mixing (iLISI, kBET) and biology-conservation metrics (ASW, ARI, IFN-γ signal).',
    verdict: 'ml',
    why: 'Unsupervised ML: it learns a corrected embedding with no cell-type labels. Because it can also remove real biology, you need explicit evaluation on both axes, and a choice of batch key that is truly technical.',
  });
  const [C, I] = await Promise.all([omicsData(main, 'sc_cells.json'), omicsData(main, 'sc_integration.json')]);
  if (!C || !I) return;
  const n = C.n, M = I.metrics;
  const cx = C.expr.CXCL9;
  const cxSc = sequential(0, d3.quantile(cx.filter((v) => v > 0), 0.95) || 1);

  // ---------------- figure 1: UMAPs ----------------
  const fig = figure(main, 'Same cells, three embeddings',
    `UMAP of ${n.toLocaleString()} cells. Left: coloured by chemistry (should mix). Right: coloured by what should stay apart. Hover any cell.`);
  let meth = 'uncorrected', right = 'type';
  select(fig.controls, { label: 'Embedding', value: meth, options: METHODS, onChange: (v) => { meth = v; paint(); } });
  select(fig.controls, { label: 'Right panel colour', value: right, options: [['type', 'true cell type'], ['patient', 'patient'], ['cxcl9', 'CXCL9 (IFN-γ gene) expression']], onChange: (v) => { right = v; paint(); } });
  const rd = stats(fig.root);
  const row = el('div', { class: 'fig-row' }); fig.body.append(row);
  const pA = el('div'), pB = el('div'); row.append(pA, pB);
  const hov = (i) => `<b>${C.cell_types[C.type[i]] || 'doublet (missed)'}</b> · ${C.patients[C.patient[i]]} (${C.resp[C.patient[i]]}, ${C.phen[C.patient[i]]}) · ${C.batches[C.batch[i]]}<br>CXCL9 log-expression ${cx[i]}`;
  const opt = { width: 460, height: 400, x: [0, 1], y: [0, 1], axes: false, grid: false, margin: { top: 6, right: 6, bottom: 6, left: 6 }, hover: hov };
  const A = canvasScatter(pA, opt), B = canvasScatter(pB, opt);
  legend(pA, C.batches.map((b, k) => [b, cat(k)]));
  const legB = el('div'); pB.append(legB);
  const order = d3.range(n);
  function paint() {
    const U = C.umap[meth];
    for (const P of [A, B]) { P.F.x.domain(ext(U[0])); P.F.y.domain(ext(U[1])); }
    A.draw({ idx: order, x: U[0], y: U[1], color: (i) => cat(C.batch[i]), r: 2, alpha: 0.7 });
    const col = right === 'type' ? (i) => CT_COLOR(C.type[i]) : right === 'patient' ? (i) => d3.color(cat(C.patient[i] % 6)).darker(C.patient[i] >= 6 ? 0.9 : 0).formatHex() : (i) => cxSc(cx[i]);
    const ord = right === 'cxcl9' ? order.slice().sort((a, b) => cx[a] - cx[b]) : order;
    B.draw({ idx: ord, x: U[0], y: U[1], color: col, r: 2, alpha: 0.8 });
    legB.innerHTML = '';
    if (right === 'type') legend(legB, C.cell_types.map((t, k) => [t, CT_COLOR(k)]));
    else if (right === 'patient') legend(legB, C.patients.map((p, k) => [`${p} (${C.batches[Math.floor(k / 6)]})`, d3.color(cat(k % 6)).darker(k >= 6 ? 0.9 : 0).formatHex()]));
    else legend(legB, [['CXCL9 not detected', cxSc(0)], ['high CXCL9', cxSc(cxSc.domain()[1])]]);
    const m = M[meth];
    rd.set({ 'batch mixing (iLISI, 0–1)': f2(m.ilisi_norm), 'kBET acceptance': pct(m.kbet), 'cell-type ARI (best res.)': f2(m.ari_best), 'IFN-γ signal kept (R²)': f2(m.ifn_r2) });
  }
  paint();

  // ---------------- figure 2: scorecard ----------------
  const fig2 = figure(main, 'The scorecard: batch removal vs biology conservation',
    'Each metric scaled so that higher is better. Batch-removal metrics are computed on each cell\'s 30 nearest neighbours; biology metrics use the simulated truth. The last row asks whether a real, patient-level biological signal (each cell\'s IFN-γ programme score, which differs between inflamed and non-inflamed tumours) can still be read off the embedding.');
  const rows = [
    ['Batch removal', null],
    ['iLISI (normalised)', 'ilisi_norm', 'effective number of chemistries among neighbours, rescaled: 0 = all one chemistry, 1 = perfectly mixed'],
    ['kBET acceptance', 'kbet', 'share of cells whose neighbourhood batch mix is not significantly different from the global mix (χ² test, p > 0.05)'],
    ['1 − |ASW batch|', 'asw_batch', 'silhouette of chemistry labels; 1 = chemistries not separable by silhouette. Note it is already high without correction: a weak metric'],
    ['Biology conservation', null],
    ['ASW cell type', 'asw_type', 'silhouette of true cell types, rescaled to 0–1'],
    ['cLISI (normalised)', 'clisi_norm', '1 = neighbours are all the same cell type'],
    ['ARI at best resolution', 'ari_best', 'agreement of Leiden clusters with true types'],
    ['IFN-γ signal kept (R²)', 'ifn_r2', 'variance of the per-cell IFN-γ programme score explained by a linear map from the embedding (CD8 T, macrophage, tumour cells)'],
  ];
  const W = 900, rowH = 30, left = 200, colW = (W - left - 10) / 3;
  const svg = d3.select(fig2.body).append('svg').attr('viewBox', `0 0 ${W} ${rows.length * rowH + 40}`).attr('width', W);
  const SHORT = ['No correction', 'Harmony', 'Over-corrected'];
  METHODS.forEach(([k], j) => svg.append('text').attr('x', left + j * colW + 4).attr('y', 16).attr('font-size', 12).attr('font-weight', 600).attr('fill', css('--ink')).text(SHORT[j]));
  const mCol = [muted(), cat(2), cat(7)];
  rows.forEach(([lab, key, desc], r) => {
    const y = 30 + r * rowH;
    if (!key) { svg.append('text').attr('x', 0).attr('y', y + 19).attr('font-size', 12).attr('font-weight', 700).attr('fill', css('--ink-2')).text(lab.toUpperCase()); return; }
    svg.append('text').attr('x', left - 10).attr('y', y + 19).attr('text-anchor', 'end').attr('font-size', 12).attr('fill', css('--ink-2')).text(lab)
      .on('mousemove', (e) => tip(e, desc)).on('mouseleave', untip);
    METHODS.forEach(([k], j) => {
      const v = M[k][key], x0 = left + j * colW;
      svg.append('rect').attr('x', x0).attr('y', y + 6).attr('width', colW - 70).attr('height', rowH - 12).attr('rx', 3).attr('fill', css('--surface-2'));
      svg.append('rect').attr('x', x0).attr('y', y + 6).attr('width', Math.max(0, Math.min(1, v)) * (colW - 70)).attr('height', rowH - 12).attr('rx', 3).attr('fill', mCol[j])
        .on('mousemove', (e) => tip(e, `<b>${lab}</b> · ${METHODS[j][1]}: ${f2(v)}<br>${desc}`)).on('mouseleave', untip);
      svg.append('text').attr('x', x0 + colW - 64).attr('y', y + 19).attr('font-size', 12).attr('fill', css('--ink')).text(f2(v));
    });
  });

  const u = M.uncorrected, h = M.harmony, o = M.over;
  section(main, 'What to notice', `<ul>
    <li><b>No correction</b>: colour by chemistry and every cell type is two islands. iLISI is ${f2(u.ilisi_norm)} and kBET accepts ${pct(u.kbet)} of neighbourhoods: almost every cell's neighbours come from its own chemistry.</li>
    <li><b>Harmony</b> with chemistry as the batch key: the islands merge (iLISI ${f2(h.ilisi_norm)}, kBET ${pct(h.kbet)}), the ${I.n_types} cell types stay separate (ARI ${f2(h.ari_best)}), and the IFN-γ signal is intact (R² ${f2(h.ifn_r2)} vs ${f2(u.ifn_r2)} before). Colour the right panel by <b>CXCL9</b>: IFN-γ-high cells are still distinguishable within the tumour, macrophage and CD8 T islands. Then do the same for the over-corrected embedding.</li>
    <li><b>Over-corrected</b> (patient as the batch key, strong penalty): the batch-removal scores are just as good (iLISI ${f2(o.ilisi_norm)}, kBET ${pct(o.kbet)}), patients look even better mixed (on average ${o.plisi.toFixed(1)} patients among each cell's neighbours vs ${h.plisi.toFixed(1)} with Harmony), but the IFN-γ R² collapses to <b>${f2(o.ifn_r2)}</b>. Patients differ in IFN-γ <i>because they differ in biology</i> (inflamed vs desert), so "removing patient effects" removed the biomarker. Cell types also start to merge (${o.n_clusters[0]} clusters at the lowest resolution vs ${I.n_types} true types).</li>
    <li>kBET acceptance stays modest even after Harmony: with 30 neighbours and a χ² test it is a strict metric, and our two chemistries are not exactly 50:50 within every cell type. Look at all metrics, not one.</li>
  </ul>`);
  steps(main, 'How it works (Harmony)', [
    'Start from PCA coordinates, L2-normalised per cell.',
    'Soft-cluster cells into K groups (soft k-means). Add a <b>diversity penalty</b> θ: assignments to a cluster are down-weighted when that cluster already has more than its share of the cell\'s batch. This pushes clusters to contain all batches.',
    'Within each cluster, fit a ridge regression of the coordinates on batch (a "mixture of experts"), and subtract each cell\'s batch effect, weighted by its soft membership.',
    'Re-cluster on the corrected coordinates and repeat until the clusters stop changing.',
    'Evaluate on two axes: <b>batch removal</b> (are neighbourhoods batch-mixed?) and <b>biology conservation</b> (are cell types and known biological signals still there?).',
  ]);
  math(main, `R_ik ∝ exp(−‖z_i − y_k‖² / σ) · Π_b ( E_kb / O_kb )^(θ·φ_ib)     soft assignment with diversity penalty
ẑ_i = z_i − Σ_k R_ik · W_kᵀ φ_i      (W_k = ridge regression of z on batch within cluster k)
iLISI_i = 1 / Σ_b p_ib²   (p_ib = share of batch b among i's neighbours; 1 = one batch, B = perfectly mixed)`);
  callout(main, '', 'In biology', `<b>Harmony</b> (R and <code>harmonypy</code>, <code>sc.external.pp.harmony_integrate</code>) is fast and a strong default on PCA. <b>scVI</b> / <b>scANVI</b> (scvi-tools) learn a batch-aware deep generative model of the counts and scale to millions of cells; <b>Seurat</b> anchors (CCA/RPCA), <b>BBKNN</b> and <b>Scanorama</b> are other common choices. <b>scIB</b> (Luecken et al., 2022) is the standard benchmark and metric suite (iLISI, cLISI, kBET, ASW, ARI/NMI, trajectory and cell-cycle conservation); its conclusion was that the best method depends on the task: scANVI, Scanorama and scVI did well on complex atlas tasks, Harmony on simpler batch structures like this one.`);
  callout(main, 'warn', 'Pitfalls', `<ul>
    <li><b>Using patient or condition as the batch key</b> when it carries biology. If responders were all run on one chemistry, correcting for chemistry would remove the response signal too: the design has to be balanced first.</li>
    <li><b>Optimising for mixing alone</b>: the most mixed embedding can be the most damaged. Always check a known biological signal.</li>
    <li><b>Using corrected values for differential expression</b>: most methods correct an embedding, not counts. Do DE on raw counts with batch in the model (see <a href="#/archive/omics/sc-pseudobulk">pseudobulk DE</a>).</li>
    <li>This toy uses a compact re-implementation of Harmony on simulated data; real datasets have batch effects that differ by cell type and are rarely this clean.</li>
  </ul>`);
  seeAlso(main, [['omics', 'sc-annotation', 'Next: cell-type annotation'], ['omics', 'bulk-batch', 'Batch effects in bulk'], ['concepts', 'kmeans', 'k-means'], ['concepts', 'regularization', 'Ridge regularisation'], ['concepts', 'autoencoder', 'Autoencoders (scVI)']]);
}
