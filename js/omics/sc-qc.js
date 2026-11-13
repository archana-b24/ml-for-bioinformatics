// Tab 2 · Single-cell step 1: QC thresholds (robust statistics) and doublet detection (a classifier on simulated doublets).
import { header, figure, slider, select, checkbox, stats, section, steps, callout, math, legend, seeAlso, el, tip, untip } from '../lib/ui.js';
import { frame, cat, css, muted } from '../lib/plot.js';
import { stepCard, omicsData } from './_shared.js';
import { canvasScatter, fmtInt, pct, f3 } from './_sc_common.js';
const d3 = window.d3;

const median = (a) => { const s = Float64Array.from(a).sort(), m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };
const mad = (a) => { const m = median(a); return [m, 1.4826 * median(a.map((v) => Math.abs(v - m)))]; };
const STATUS = ['intact cell', 'low quality (damaged)', 'doublet'];
const statusColor = (s) => (s === 0 ? muted() : s === 1 ? cat(7) : cat(6));

export default async function render(main) {
  header(main, 'QC & doublet detection',
    'Before any biology, decide which barcodes are <b>real, intact single cells</b>. Damaged cells are outliers on simple per-cell metrics, so a robust statistical cut-off works. Doublets look like plausible cells, so we <b>train a classifier on doublets we simulate ourselves</b>.');
  stepCard(main, {
    stage: 'Single-cell · step 1 of 7',
    question: 'Which of the 12 patients\' barcodes are intact single cells, and which are damaged cells or two cells captured in one droplet?',
    data: 'Raw UMI counts for every barcode (cells × genes), two 10x chemistries (3′ v2 and 3′ v3).',
    method: 'Per-batch MAD thresholds on library size, genes detected and % mitochondrial reads; a gradient-boosted classifier trained to tell real barcodes from artificial doublets (scDblFinder-style).',
    verdict: 'both',
    why: 'Damaged cells are simple outliers: robust statistics are enough and easier to defend. Doublets need ML because no single metric separates them, but the labels can be manufactured by summing random pairs of real cells.',
  });
  const D = await omicsData(main, 'sc_qc.json');
  if (!D) return;
  const N = D.n;
  const lt = D.total.map((v) => Math.log10(Math.max(1, v))), lg = D.ngenes.map((v) => Math.log10(Math.max(1, v)));
  const isLowq = (i) => D.status[i] === 1;
  const nLowq = D.status.filter((s) => s === 1).length, nDbl = D.status.filter((s) => s === 2).length;

  // ---------------- figure 1: MAD thresholds ----------------
  function qcRule(nmad) {
    const thr = [0, 1].map((b) => {
      const ids = d3.range(N).filter((i) => D.batch[i] === b);
      const [mc, sc] = mad(ids.map((i) => lt[i])), [mg, sg] = mad(ids.map((i) => lg[i])), [mm, sm] = mad(ids.map((i) => D.pct_mt[i]));
      return { minC: mc - nmad * sc, minG: mg - nmad * sg, maxMt: mm + nmad * sm };
    });
    const fail = d3.range(N).map((i) => { const t = thr[D.batch[i]]; return lt[i] < t.minC || lg[i] < t.minG || D.pct_mt[i] > t.maxMt; });
    return { thr, fail };
  }
  const fixedFail = d3.range(N).map((i) => D.ngenes[i] < 200 || D.pct_mt[i] > 10);
  function score(fail) {
    let caught = 0, lostOk = 0, lostTum = 0, removed = 0;
    for (let i = 0; i < N; i++) if (fail[i]) { removed++; if (isLowq(i)) caught++; else if (D.status[i] === 0) { lostOk++; if (D.type[i] === 0) lostTum++; } }
    return { caught, lostOk, lostTum, removed };
  }
  const base = score(qcRule(3).fail), fixedS = score(fixedFail), thr3 = qcRule(3).thr;

  const fig = figure(main, 'Per-batch MAD thresholds vs fixed cut-offs',
    'Each dot is a barcode. Colour is the <b>simulated truth</b>, which you never have in real data. Dashed lines are the thresholds for the selected chemistry; shaded points fail QC. Hover a cell for its metrics.');
  let nmad = 3, bsel = 0, useFixed = false;
  slider(fig.controls, { label: 'Threshold width (MADs)', min: 1.5, max: 6, step: 0.5, value: 3, fmt: (v) => v.toFixed(1), onInput: (v) => { nmad = v; paint(); } });
  select(fig.controls, { label: 'Chemistry', value: '0', options: [['0', D.batches[0]], ['1', D.batches[1]]], onChange: (v) => { bsel = +v; paint(); } });
  checkbox(fig.controls, { label: 'Use fixed cut-offs instead (≥200 genes, ≤10% mito)', checked: false, onChange: (c) => { useFixed = c; paint(); } });
  const rd = stats(fig.root);
  const row = el('div', { class: 'fig-row' }); fig.body.append(row);
  const pA = el('div'), pB = el('div'); row.append(pA, pB);
  const hov = (i) => `<b>${STATUS[D.status[i]]}</b> · ${D.cell_types[D.type[i]]}${D.status[i] === 2 ? ' + another type' : ''}<br>${D.batches[D.batch[i]]}<br>UMIs ${fmtInt(D.total[i])} · genes ${fmtInt(D.ngenes[i])} · mito ${D.pct_mt[i]}%`;
  const A = canvasScatter(pA, { width: 470, height: 380, x: [1.8, 4.9], y: [0, 65], xLabel: 'log₁₀ total UMIs', yLabel: '% mitochondrial reads', hover: hov });
  const B = canvasScatter(pB, { width: 470, height: 380, x: [1.8, 4.9], y: [1.5, 3.9], xLabel: 'log₁₀ total UMIs', yLabel: 'log₁₀ genes detected', hover: hov });
  legend(fig.body, [[STATUS[0], muted()], [STATUS[1], cat(7)], [STATUS[2], cat(6)]]);
  const order = d3.range(N).sort((a, b) => D.status[a] - D.status[b]);   // truth classes drawn on top

  function paint() {
    const { thr, fail } = useFixed ? { thr: null, fail: fixedFail } : qcRule(nmad);
    const ids = order.filter((i) => D.batch[i] === bsel);
    const col = (i) => statusColor(D.status[i]);
    const r = (i) => (fail[i] ? 1.6 : 2.4);
    A.draw({ idx: ids, x: lt, y: D.pct_mt, color: col, r, alpha: 0.75 });
    B.draw({ idx: ids, x: lt, y: lg, color: col, r, alpha: 0.75 });
    const lines = useFixed
      ? { A: [{ y: 10 }], B: [{ y: Math.log10(200) }] }
      : { A: [{ x: thr[bsel].minC }, { y: thr[bsel].maxMt }], B: [{ x: thr[bsel].minC }, { y: thr[bsel].minG }] };
    for (const [P, L] of [[A, lines.A], [B, lines.B]]) {
      P.F.layer('thr').selectAll('line').data(L).join('line')
        .attr('x1', (d) => (d.x != null ? P.F.x(d.x) : 0)).attr('x2', (d) => (d.x != null ? P.F.x(d.x) : P.F.w))
        .attr('y1', (d) => (d.y != null ? P.F.y(d.y) : 0)).attr('y2', (d) => (d.y != null ? P.F.y(d.y) : P.F.h))
        .attr('stroke', css('--ink')).attr('stroke-width', 2).attr('stroke-dasharray', '5 4');
    }
    const s = score(fail);
    const o = {
      'barcodes removed': fmtInt(s.removed),
      'damaged cells caught': `${s.caught} / ${nLowq}`,
      'intact cells lost': fmtInt(s.lostOk),
      '…of which tumour': fmtInt(s.lostTum),
    };
    if (!useFixed) Object.assign(o, { [`min UMIs (${D.batches[bsel]})`]: fmtInt(10 ** thr[bsel].minC), [`max mito (${D.batches[bsel]})`]: thr[bsel].maxMt.toFixed(1) + '%' });
    rd.set(o);
  }
  paint();

  // ---------------- figure 2: doublets ----------------
  const passed = d3.range(N).filter((i) => D.dbl_clf[i] >= 0);
  const fig2 = figure(main, 'Doublet scores: what a model trained on fake doublets can see',
    `We summed ${fmtInt(D.n_sim)} random pairs of QC-passed barcodes to make artificial doublets, then scored every real barcode two ways: the fraction of its ${D.k_knn} nearest neighbours (in PCA space) that are artificial (Scrublet-like), and the out-of-fold probability from a gradient-boosted classifier trained on real vs artificial (scDblFinder-like). Left: ROC curves against the true doublets. Right: score distributions.`);
  let scoreKey = 'dbl_clf', rate = Math.round(1000 * D.dbl_call_n / D.n_passed) / 10;
  select(fig2.controls, { label: 'Score', value: 'dbl_clf', options: [['dbl_clf', 'Classifier (scDblFinder-like)'], ['dbl_knn', 'kNN fraction (Scrublet-like)']], onChange: (v) => { scoreKey = v; paint2(); } });
  slider(fig2.controls, { label: 'Expected doublet rate', min: 1, max: 15, step: 0.5, value: rate, fmt: (v) => v.toFixed(1) + '%', onInput: (v) => { rate = v; paint2(); } });
  const rd2 = stats(fig2.root);
  const row2 = el('div', { class: 'fig-row' }); fig2.body.append(row2);
  const qA = el('div'), qB = el('div'); row2.append(qA, qB);
  const R = frame(qA, { width: 420, height: 360, x: [0, 1], y: [0, 1], xLabel: 'false-positive rate (singlets called doublet)', yLabel: 'true-positive rate' });
  R.layer('diag').append('line').attr('x1', 0).attr('y1', R.h).attr('x2', R.w).attr('y2', 0).attr('stroke', muted()).attr('stroke-dasharray', '3 4');
  const rocNames = Object.keys(D.roc);
  const rocCol = [muted(), cat(0), cat(6)];
  rocNames.forEach((k, j) => {
    const r = D.roc[k];
    R.layer('roc').append('path').datum(r.fpr.map((f, i) => [f, r.tpr[i]])).attr('fill', 'none').attr('stroke', rocCol[j]).attr('stroke-width', 2)
      .attr('d', d3.line().x((d) => R.x(d[0])).y((d) => R.y(d[1])));
  });
  const rocDot = R.layer('dot').append('circle').attr('r', 5).attr('fill', css('--ink')).attr('stroke', css('--surface')).attr('stroke-width', 1.5);
  legend(qA, rocNames.map((k, j) => [`${k} · AUC ${D.roc[k].auc.toFixed(3)}`, rocCol[j]]));
  const H = frame(qB, { width: 440, height: 360, x: [0, 1], y: [0, 1], xLabel: 'doublet score', yLabel: 'share of group (per bin)' });
  legend(qB, [['true singlets (intact or damaged)', muted()], ['true doublets', cat(6)], ['call threshold', css('--ink')]]);

  const nDp = passed.filter((i) => D.status[i] === 2).length, nSp = passed.length - nDp;
  function callStats(key, rt) {
    const sc = passed.map((i) => D[key][i]);
    const nCall = Math.round(rt / 100 * passed.length);
    const thr = [...sc].sort((a, b) => b - a)[Math.max(0, nCall - 1)];
    let tp = 0, fp = 0;
    passed.forEach((i, k) => { if (sc[k] >= thr) { if (D.status[i] === 2) tp++; else fp++; } });
    return { sc, thr, tp, fp };
  }
  const call0 = callStats('dbl_clf', rate);
  function paint2() {
    const { sc, thr, tp, fp } = callStats(scoreKey, rate);
    rocDot.attr('cx', R.x(fp / nSp)).attr('cy', R.y(tp / nDp));
    const mx = d3.max(sc) || 1;
    const bins = d3.bin().domain([0, mx]).thresholds(d3.range(0, mx, mx / 40));
    const groups = [[passed.filter((i) => D.status[i] !== 2), muted()], [passed.filter((i) => D.status[i] === 2), cat(6)]];
    const hs = groups.map(([ids, c]) => { const b = bins(ids.map((i) => D[scoreKey][i])); return { c, b: b.map((x) => ({ x0: x.x0, x1: x.x1, f: x.length / ids.length })) }; });
    H.x.domain([0, mx]); H.y.domain([0, d3.max(hs.flatMap((h) => h.b.map((x) => x.f))) * 1.05]);
    H.g.select('.axis.x').call(d3.axisBottom(H.x).ticks(5)); H.g.select('.axis.y').call(d3.axisLeft(H.y).ticks(5, '.0%'));
    H.layer('bars').selectAll('g').data(hs).join('g').attr('fill', (d) => d.c).attr('fill-opacity', 0.65)
      .selectAll('rect').data((d) => d.b).join('rect')
      .attr('x', (d) => H.x(d.x0) + 0.5).attr('width', (d) => Math.max(0, H.x(d.x1) - H.x(d.x0) - 1))
      .attr('y', (d) => H.y(d.f)).attr('height', (d) => H.h - H.y(d.f))
      .on('mousemove', (e, d) => tip(e, `score ${d.x0.toFixed(2)}–${d.x1.toFixed(2)}: ${pct(d.f, 1)} of group`)).on('mouseleave', untip);
    H.layer('thr').selectAll('line').data([thr]).join('line').attr('x1', (d) => H.x(d)).attr('x2', (d) => H.x(d)).attr('y1', 0).attr('y2', H.h)
      .attr('stroke', css('--ink')).attr('stroke-width', 2).attr('stroke-dasharray', '5 4');
    rd2.set({ 'called doublets': fmtInt(tp + fp), 'true doublets caught': `${tp} / ${nDp}`, 'singlets wrongly removed': fmtInt(fp), 'doublets left in data': fmtInt(nDp - tp), precision: pct(tp / Math.max(1, tp + fp)) });
  }
  paint2();

  const S = D.summary;
  const dblKnn = D.roc['kNN score (Scrublet-like)'].auc, dblClf = D.roc['Classifier (scDblFinder-like)'].auc, dblTot = D.roc['Total counts only'].auc;
  section(main, 'What to notice', `<ul>
    <li>At 3 MADs the per-batch rule removes ${fmtInt(base.removed)} barcodes and catches <b>${base.caught} of ${nLowq}</b> damaged cells, losing only ${fmtInt(base.lostOk)} intact cells. The thresholds differ by chemistry (minimum UMIs ${fmtInt(10 ** thr3[0].minC)} for ${D.batches[0]} vs ${fmtInt(10 ** thr3[1].minC)} for ${D.batches[1]}) because v3 captures more molecules per cell. A single global cut-off would be wrong for one of them.</li>
    <li>Tick <b>fixed cut-offs</b>. The copy-pasted "≤10% mito" rule removes ${fmtInt(fixedS.removed)} barcodes, including <b>${fmtInt(fixedS.lostTum)} intact tumour cells</b>: in this study tumour cells are metabolically active and naturally carry more mitochondrial RNA. Fixed thresholds silently delete a biological population.</li>
    <li>Drag the MAD width down to 1.5: you start cutting into the healthy cloud. Up to 6: damaged cells slip through. There is no "correct" width, only a documented one.</li>
    <li>In the doublet panel, total UMIs alone gives AUC ${dblTot.toFixed(3)} (doublets have about twice the RNA), but the kNN score (${dblKnn.toFixed(3)}) and the classifier (${dblClf.toFixed(3)}) do far better because they use the <i>expression profile</i>: a doublet sits between two cell types.</li>
    <li>With the classifier, calling the expected ${rate.toFixed(1)}% of QC-passed barcodes catches ${call0.tp} of ${nDp} doublets, removes ${call0.fp} genuine singlets and leaves ${nDp - call0.tp} doublets in the data. Doublets of two similar cell types (two T-cell subsets, say) sit inside a real cluster and are nearly invisible to every method; the ones left resurface later as low-confidence or "hybrid" cells.</li>
  </ul>`);
  steps(main, 'How it works', [
    'Compute three numbers per barcode: total UMIs, genes detected, % reads from mitochondrial genes (a broken membrane leaks cytoplasmic RNA but keeps mitochondria).',
    'Within each batch, compute the <b>median</b> and the <b>median absolute deviation</b> (MAD) of each metric (log scale for counts). Flag cells more than k MADs on the bad side. Median and MAD ignore the outliers they are hunting, unlike mean and SD.',
    'For doublets, simulate artificial doublets by adding the counts of random pairs of real barcodes.',
    'Embed real and artificial barcodes together (normalise, highly variable genes, PCA).',
    'Train a classifier (here gradient boosting on PCs, library size, genes detected and the kNN fraction) to separate real from artificial, with cross-validation so every real cell gets an out-of-fold score.',
    'Real barcodes that look like artificial doublets are called doublets; the number called is set by the expected rate (about 0.8% per 1,000 cells loaded on 10x).',
  ]);
  math(main, `fail if  log10(UMIs) < median − k·MAD   or   %mito > median + k·MAD     (per batch)
MAD = 1.4826 · median(|x − median(x)|)        (1.4826 makes MAD = SD for normal data)
doublet score = P(artificial | PCs, UMIs, genes, kNN fraction)`);
  callout(main, '', 'In biology', `<b>Scanpy</b> (<code>sc.pp.calculate_qc_metrics</code>) and <b>Seurat</b> compute the metrics; <b>scater</b>/<code>isOutlier</code> implements the MAD rule per batch. For doublets, <b>scDblFinder</b> (R/Bioconductor, gradient-boosted trees on simulated doublets) and <b>Scrublet</b> (Python, kNN on simulated doublets) are the standard tools; in the independent benchmark of Xi &amp; Li (2021) DoubletFinder was most accurate, and scDblFinder's authors later reported it on top on the same datasets (Germain et al. 2021). Ambient RNA removal (<b>SoupX</b>, <b>CellBender</b>) is a separate step not shown here.`);
  callout(main, 'warn', 'Pitfalls', `<ul>
    <li><b>Fixed thresholds from a tutorial</b> (200 genes, 10% mito) are tuned for PBMCs. Tumour, hepatocytes and cardiomyocytes have high mito by biology.</li>
    <li><b>Thresholds across batches</b>: compute them per sample or chemistry, or a deeper batch loses its smallest cells.</li>
    <li><b>Running doublet detection on merged samples</b>: doublets only form within one channel, so simulate per sample.</li>
    <li><b>Over-filtering</b>: a cluster that only exists before QC might be a real, fragile cell type (neutrophils, for example, have low RNA content).</li>
    <li>In this toy, truth labels let us score the methods. In real data you never see the colours, so check what you removed (which cell types, which patients).</li>
  </ul>`);
  seeAlso(main, [['omics', 'sc-embedding', 'Next: normalise, HVGs, PCA'], ['omics', 'bulk-qc', 'Bulk QC'], ['concepts', 'knn', 'k-nearest neighbours'], ['concepts', 'imbalance', 'Class imbalance']]);
}
