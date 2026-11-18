// Tab 2 · Single-cell step 2: normalisation, highly variable genes, PCA. Mostly statistics; PCA is the first (unsupervised) ML.
import { header, figure, slider, select, stats, section, steps, callout, math, legend, seeAlso, el, tip, untip } from '../../lib/ui.js';
import { frame, cat, css, muted, sequential } from '../../lib/plot.js';
import { stepCard, omicsData } from '../../omics/_shared.js';
import { canvasScatter, CT_COLOR, ext, fmtInt, pct } from '../../omics/_sc_common.js';
const d3 = window.d3;

const GROUPS = [
  ['cell-type marker', (c) => c === 'marker', () => cat(2)],
  ['IFN-γ module', (c) => c === 'module:IFNG', () => cat(7)],
  ['CAF/TGF-β module', (c) => c === 'module:TGFB_CAF', () => cat(3)],
  ['proliferation module', (c) => c === 'module:PROLIF', () => cat(6)],
  ['exhaustion / TLS module, receptors', (c) => c === 'module:EXHAUSTION' || c === 'module:TLS' || c === 'receptor', () => cat(4)],
  ['housekeeping', (c) => c === 'housekeeping', () => cat(0)],
  ['filler (no planted signal)', () => true, () => muted()],
];
const groupOf = (c) => GROUPS.findIndex((g) => g[1](c));

export default async function render(main) {
  header(main, 'Normalise, HVGs, PCA',
    'Raw counts mostly measure <b>how many molecules the droplet happened to capture</b>. Normalising removes that, choosing highly variable genes keeps the informative ones, and PCA compresses ~1,000 noisy genes into a few dozen axes where cells can be compared.');
  stepCard(main, {
    stage: 'Single-cell · step 2 of 7',
    question: 'How do we turn sparse, noisy counts into a space where "similar cells are close"?',
    data: 'QC-passed, doublet-filtered UMI counts (cells × ~2,000 genes in this toy; ~30,000 in real data).',
    method: 'Library-size normalisation + log1p, highly variable genes (mean–dispersion, binned z-score), scaling, PCA.',
    verdict: 'both',
    why: 'Normalisation and HVG selection are statistical modelling of count noise. PCA is unsupervised ML (a linear embedding) and its only job is denoising for the next steps. No labels are involved.',
  });
  const [E, C] = await Promise.all([omicsData(main, 'sc_embedding.json'), omicsData(main, 'sc_cells.json')]);
  if (!E || !C) return;
  const G = E.genes.length;
  const grp = E.cat.map(groupOf);
  const planted = E.cat.map((c) => c === 'marker' || c.startsWith('module'));
  const nPlanted = planted.filter(Boolean).length;
  const rank = d3.range(G).filter((i) => E.log_mean[i] > Math.log1p(1e-3)).sort((a, b) => E.z_disp[b] - E.z_disp[a]);
  const hvgSet = (n) => { const s = new Uint8Array(G); rank.slice(0, n).forEach((i) => { s[i] = 1; }); return s; };

  // ---------------- figure 1: HVGs ----------------
  const fig = figure(main, 'Which genes vary more than their mean predicts?',
    'Each dot is a gene. Left: mean expression vs dispersion (variance / mean, log scale) after normalisation. Dispersion falls with mean for pure counting noise, so genes are ranked by a <b>z-score within bins of similar mean</b>. Right: the rank of each gene on that z-score. Highlighted genes are the selected HVGs; colour is what the simulation planted.');
  let nHvg = E.n_hvg;
  slider(fig.controls, { label: 'Number of HVGs', min: 100, max: 2000, step: 50, value: nHvg, onInput: (v) => { nHvg = v; paint(); } });
  const rd = stats(fig.root);
  const row = el('div', { class: 'fig-row' }); fig.body.append(row);
  const pA = el('div'), pB = el('div'); row.append(pA, pB);
  const hov = (i) => `<b>${E.genes[i]}</b> · ${GROUPS[grp[i]][0]}<br>log mean ${E.log_mean[i].toFixed(2)} · log dispersion ${E.log_disp[i].toFixed(2)}<br>binned z ${E.z_disp[i].toFixed(2)} · rank ${rank.indexOf(i) + 1}`;
  const A = canvasScatter(pA, { width: 470, height: 360, x: ext(E.log_mean), y: ext(E.log_disp), xLabel: 'log(1 + mean CP10k)', yLabel: 'log dispersion', hover: hov });
  const rx = rank.map((_, k) => k + 1), zr = rank.map((i) => E.z_disp[i]);
  const rIdx = d3.range(rank.length), rankX = [], rankY = [];
  rank.forEach((g, k) => { rankX[k] = k + 1; rankY[k] = E.z_disp[g]; });
  const B = canvasScatter(pB, { width: 470, height: 360, x: [0, rank.length + 20], y: ext(zr), xLabel: 'gene rank by binned dispersion z-score', yLabel: 'dispersion z-score', hover: (k) => hov(rank[k]) });
  legend(fig.body, GROUPS.map((g) => [g[0], g[2]()]));
  const cut = B.F.layer('cut').append('line').attr('y1', 0).attr('y2', B.F.h).attr('stroke', css('--ink')).attr('stroke-width', 2).attr('stroke-dasharray', '5 4');
  const order = d3.range(G).sort((a, b) => grp[b] - grp[a]);
  function paint() {
    const s = hvgSet(nHvg);
    A.draw({ idx: order, x: E.log_mean, y: E.log_disp, color: (i) => GROUPS[grp[i]][2](), r: (i) => (s[i] ? 3.4 : 1.6), alpha: 0.8 });
    B.draw({ idx: rIdx.slice().sort((a, b) => grp[rank[b]] - grp[rank[a]]), x: rankX, y: rankY, color: (k) => GROUPS[grp[rank[k]]][2](), r: (k) => (k < nHvg ? 3 : 1.6), alpha: 0.8 });
    cut.attr('x1', B.F.x(nHvg)).attr('x2', B.F.x(nHvg));
    let pl = 0, fi = 0;
    for (let i = 0; i < G; i++) if (s[i]) { if (planted[i]) pl++; if (E.cat[i] === 'filler') fi++; }
    rd.set({ 'HVGs': fmtInt(nHvg), 'planted genes captured': `${pl} / ${nPlanted}`, 'HVGs that are filler': pct(fi / nHvg) });
  }
  paint();

  // ---------------- figure 2: PCA ----------------
  const fig2 = figure(main, 'PCA before and after normalisation',
    `${fmtInt(C.n)} of ${fmtInt(C.n_kept)} cells shown. Left: the first two principal components. Right: share of variance explained by each of the first 50 PCs of the normalised data; the dashed line marks the ${E.npc} PCs passed on to clustering. Hover a cell to see its type and library size.`);
  let norm = 'norm', colorBy = 'lib';
  select(fig2.controls, { label: 'Input to PCA', value: 'norm', options: [['raw', 'log1p of raw counts (no normalisation)'], ['norm', 'log CP10k, HVGs, scaled']], onChange: (v) => { norm = v; paint2(); } });
  select(fig2.controls, { label: 'Colour by', value: 'lib', options: [['lib', 'library size (total UMIs)'], ['type', 'true cell type'], ['batch', 'chemistry']], onChange: (v) => { colorBy = v; paint2(); } });
  const rd2 = stats(fig2.root);
  const row2 = el('div', { class: 'fig-row' }); fig2.body.append(row2);
  const qA = el('div'), qB = el('div'); row2.append(qA, qB);
  const loglib = C.lib.map((v) => Math.log10(v));
  const libSc = sequential(d3.quantile(loglib, 0.02), d3.quantile(loglib, 0.98));
  const P = canvasScatter(qA, { width: 470, height: 380, x: [0, 1], y: [0, 1], xLabel: 'PC1', yLabel: 'PC2',
    hover: (i) => `<b>${C.cell_types[C.type[i]] || 'doublet (missed)'}</b> · ${C.patients[C.patient[i]]} · ${C.batches[C.batch[i]]}<br>total UMIs ${fmtInt(C.lib[i])}` });
  const legP = el('div'); qA.append(legP);
  const nv = 50, vr = E.var_ratio.slice(0, nv);
  const S = frame(qB, { width: 440, height: 380, x: [0.5, nv + 0.5], y: [0, d3.max(vr) * 1.1], xLabel: 'principal component', yLabel: 'share of variance' });
  S.g.select('.axis.y').call(d3.axisLeft(S.y).ticks(5, '.0%'));
  S.layer('bars').selectAll('rect').data(vr).join('rect').attr('x', (d, i) => S.x(i + 0.6)).attr('width', S.x(1.4) - S.x(0.6)).attr('y', (d) => S.y(d)).attr('height', (d) => S.h - S.y(d))
    .attr('fill', (d, i) => (i < E.npc ? css('--accent') : muted())).attr('rx', 1)
    .on('mousemove', (e, d) => { const i = vr.indexOf(d); tip(e, `PC${i + 1}: ${pct(d, 2)} · cumulative ${pct(d3.sum(vr.slice(0, i + 1)), 1)}`); }).on('mouseleave', untip);
  S.layer('cut').append('line').attr('x1', S.x(E.npc + 0.5)).attr('x2', S.x(E.npc + 0.5)).attr('y1', 0).attr('y2', S.h).attr('stroke', css('--ink')).attr('stroke-width', 2).attr('stroke-dasharray', '5 4');
  S.layer('cut').append('text').attr('x', S.x(E.npc + 0.5) + 6).attr('y', 14).attr('font-size', 12).attr('fill', css('--ink-2')).text(`${E.npc} PCs kept: ${pct(d3.sum(vr.slice(0, E.npc)), 0)} of all variance`);

  function paint2() {
    const xy = norm === 'raw' ? C.pca_raw : C.pca;
    P.F.x.domain(ext(xy[0])); P.F.y.domain(ext(xy[1]));
    P.F.g.select('.axis.x').call(d3.axisBottom(P.F.x).ticks(5)); P.F.g.select('.axis.y').call(d3.axisLeft(P.F.y).ticks(5));
    const col = colorBy === 'lib' ? (i) => libSc(loglib[i]) : colorBy === 'type' ? (i) => CT_COLOR(C.type[i]) : (i) => cat(C.batch[i]);
    P.draw({ idx: d3.range(C.n), x: xy[0], y: xy[1], color: col, r: 2.2, alpha: 0.8 });
    legP.innerHTML = '';
    if (colorBy === 'lib') legend(legP, [[`low library (${fmtInt(10 ** libSc.domain()[0])} UMIs)`, libSc(libSc.domain()[0])], [`high library (${fmtInt(10 ** libSc.domain()[1])} UMIs)`, libSc(libSc.domain()[1])]]);
    else if (colorBy === 'type') legend(legP, C.cell_types.map((t, k) => [t, CT_COLOR(k)]));
    else legend(legP, C.batches.map((b, k) => [b, cat(k)]));
    rd2.set(norm === 'raw'
      ? { '|corr(PC1, log library size)|': E.r_lib_raw.toFixed(2), 'what PC1 mostly measures': 'capture depth' }
      : { 'max |corr(PC1 or PC2, log library size)|': E.r_lib_norm.toFixed(2), 'what PC1/PC2 measure': 'cell identity' });
  }
  paint2();

  const nPl = (n) => { const s = hvgSet(n); let c = 0; for (let i = 0; i < G; i++) if (s[i] && planted[i]) c++; return c; };
  const hs0 = hvgSet(E.n_hvg), hkTot = E.cat.filter((c) => c === 'housekeeping').length, hkSel = E.cat.filter((c, i) => c === 'housekeeping' && hs0[i]).length;
  section(main, 'What to notice', `<ul>
    <li>With the default ${E.n_hvg} HVGs, <b>${E.planted_hvg} of ${E.planted_total}</b> planted marker and module genes are kept. But look at the housekeeping genes (blue, far right): they carry no cell-type signal, yet ${hkSel} of ${hkTot} are selected. They are alone in the highest-mean bin, so the binned z-score only compares them with each other and half of them always "win". This is a known quirk of the binned (Seurat v1 / Scanpy <code>seurat</code>) method and one reason count-model methods (<code>seurat_v3</code>, Pearson residuals) exist.</li>
    <li>Slide down to 200 HVGs: ${nPl(200)} of ${nPlanted} planted genes survive, and weaker programmes start to vanish. Slide up to 2,000: everything is "selected", and the filler genes that carry only noise dilute the PCA.</li>
    <li>Switch PCA input to <b>raw counts</b> and colour by library size: PC1 is a depth gradient (|r| = ${E.r_lib_raw.toFixed(2)}). After normalisation the correlation drops to ${E.r_lib_norm.toFixed(2)} and the PCs separate cell types instead. Some correlation remains because cell types genuinely differ in RNA content (tumour cells are larger), which is biology, not artefact.</li>
    <li>Colour the normalised PCA by <b>chemistry</b>: the two batches already sit apart. Normalisation fixes depth, not chemistry. That is the job of <a href="#/archive/omics/sc-integration">integration</a>.</li>
    <li>The scree plot has no sharp elbow after the first handful of PCs. Choosing ${E.npc} is a judgement call. Note how little of the total variance even the kept PCs explain: the rest is mostly per-cell counting noise, which is exactly what we want to throw away.</li>
  </ul>`);
  steps(main, 'How it works', [
    '<b>Normalise</b>: divide each cell\'s counts by its total and multiply by 10,000 (CP10k), so every cell has the same depth.',
    '<b>Log-transform</b>: log(1 + x) stabilises the variance (a gene going 1→2 counts matters as much as 100→200) and makes effects multiplicative.',
    '<b>Highly variable genes</b>: for each gene compute mean and dispersion, bin genes by mean, z-score the dispersion within each bin, keep the top ~1,000–3,000. Genes that vary more than counting noise predicts are likely to differ between cell states.',
    '<b>Scale</b> each HVG to mean 0, variance 1 (clipped), so a few highly expressed genes do not dominate.',
    '<b>PCA</b>: find the orthogonal directions of largest variance and keep the top 20–50. Later steps (kNN graph, UMAP, clustering, integration) all work in this space because distances in raw gene space are dominated by noise.',
  ]);
  math(main, `x̃ᵢg = log(1 + 10⁴ · xᵢg / Σg xᵢg)
z-dispersion_g = (log(σ²_g/μ_g) − mean_bin) / sd_bin     (bins of similar mean)
Z = X_scaled · V₁..ₖ     (V = top-k eigenvectors of the gene covariance)`);
  callout(main, '', 'In biology', `<b>Scanpy</b>: <code>sc.pp.normalize_total</code>, <code>sc.pp.log1p</code>, <code>sc.pp.highly_variable_genes(flavor="seurat")</code> (this page) or <code>"seurat_v3"</code> on raw counts, <code>sc.pp.scale</code>, <code>sc.tl.pca</code>. <b>Seurat</b>: <code>NormalizeData</code> / <code>FindVariableFeatures</code> / <code>ScaleData</code> / <code>RunPCA</code>, or <b>sctransform</b> (regularised negative-binomial residuals instead of log-normalisation). Analytic Pearson residuals and <b>scran</b> pooled size factors are well-benchmarked alternatives. Deep generative models such as <b>scVI</b> learn the embedding and the count noise model jointly instead of PCA.`);
  callout(main, 'warn', 'Pitfalls', `<ul>
    <li><b>Skipping normalisation</b>: PC1 becomes sequencing depth and clusters split by library size.</li>
    <li><b>Selecting HVGs across batches without care</b>: genes that differ by chemistry look "highly variable". Scanpy's <code>batch_key</code> selects genes variable <i>within</i> batches.</li>
    <li><b>Regressing out too much</b> (cell cycle, % mito, library size) can delete real biology, e.g. proliferating tumour cells.</li>
    <li><b>Reading PCA distances literally</b>: PCs are linear, and continuous processes like differentiation are curved in them.</li>
  </ul>`);
  seeAlso(main, [['omics', 'sc-clustering', 'Next: graph clustering & UMAP'], ['concepts', 'pca', 'PCA'], ['concepts', 'scaling', 'Feature scaling'], ['concepts', 'curse-dim', 'Curse of dimensionality'], ['omics', 'bulk-qc', 'Bulk normalisation']]);
}
