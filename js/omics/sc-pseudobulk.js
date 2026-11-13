// Tab 2 · Single-cell step 7: pseudobulk differential expression and the pseudoreplication trap.
import { header, figure, select, slider, stats, section, steps, callout, math, legend, seeAlso, el, tip, untip } from '../lib/ui.js';
import { frame, cat, css, muted } from '../lib/plot.js';
import { stepCard, omicsData } from './_shared.js';
import { canvasScatter, f2, pct, fmtInt } from './_sc_common.js';
const d3 = window.d3;

export default async function render(main) {
  header(main, 'Pseudobulk DE',
    'Responders vs non-responders is a comparison of <b>patients</b>. Thousands of cells from one patient are not thousands of independent replicates; they share that patient\'s genetics, tumour and sample handling. Tests that treat cells as replicates produce <b>tiny p-values for differences that are not there</b>. Summing cells per patient first (pseudobulk) gives honest answers.');
  stepCard(main, {
    stage: 'Single-cell · step 7 of 7',
    question: 'Within each cell type, which genes differ between responders and non-responders?',
    data: 'Raw counts of annotated cells from 12 patients (6 responders, 6 non-responders), per cell type.',
    method: 'Pseudobulk: sum counts per patient × cell type, then a patient-level linear model with empirical-Bayes variance shrinkage (limma-style; edgeR/DESeq2 in practice). Compared with a per-cell Wilcoxon test.',
    verdict: 'stats',
    why: 'This is a hypothesis test with a well-defined unit of replication. No ML model is needed, and none would fix the problem: the issue is what counts as an independent observation.',
  });
  const D = await omicsData(main, 'sc_pseudobulk.json');
  if (!D) return;
  const TYPES = D.types;

  // ---------------- figure 1: volcanoes ----------------
  const fig = figure(main, 'Real labels: two volcano plots of the same cells',
    'Each dot is a gene; x = log₂ fold change (responders / non-responders), y = −log₁₀ p. Large dots pass 5% FDR (Benjamini–Hochberg). Colour marks genes the simulation actually made different through the planted programmes (IFN-γ, exhaustion, TLS…), which differ between responder and non-responder patients on average, but not in every patient.');
  let ct = 'Tumour';
  select(fig.controls, { label: 'Cell type', value: ct, options: TYPES.map((t) => [t, t]), onChange: (v) => { ct = v; paint(); paint2(); } });
  const rd = stats(fig.root);
  const row = el('div', { class: 'fig-row' }); fig.body.append(row);
  const pA = el('div'), pB = el('div'); row.append(pA, pB);
  const hov = (key) => (i) => { const e = D.de[ct]; return `<b>${e.genes[i]}</b> · ${e.planted[i] ? 'planted difference' : 'no planted difference'}<br>cell-level: log₂FC ${e.lfc_nv[i]}, p = 10^−${e.nlp_nv[i]}${e.sig_nv[i] ? ' (FDR < 5%)' : ''}<br>pseudobulk: log₂FC ${e.lfc_pb[i]}, p = 10^−${e.nlp_pb[i]}${e.sig_pb[i] ? ' (FDR < 5%)' : ''}`; };
  const A = canvasScatter(pA, { width: 460, height: 380, x: [-4, 4], y: [0, 10], xLabel: 'log₂ fold change', yLabel: '−log₁₀ p (cells as replicates)', hover: hov('nv') });
  const B = canvasScatter(pB, { width: 460, height: 380, x: [-4, 4], y: [0, 10], xLabel: 'log₂ fold change', yLabel: '−log₁₀ p (patients as replicates)', hover: hov('pb') });
  legend(fig.body, [['planted difference', cat(7)], ['no planted difference', muted()], ['large dot = FDR < 5%', css('--ink')]]);
  const titleA = el('div', { class: 'fig-caption' }), titleB = el('div', { class: 'fig-caption' });
  pA.prepend(titleA); pB.prepend(titleB);

  function paint() {
    const e = D.de[ct];
    const idx = d3.range(e.genes.length).sort((a, b) => e.planted[a] - e.planted[b]);
    for (const [P, lfc, nlp, sig] of [[A, e.lfc_nv, e.nlp_nv, e.sig_nv], [B, e.lfc_pb, e.nlp_pb, e.sig_pb]]) {
      const xm = Math.max(1, d3.quantile(lfc.map(Math.abs), 0.999)) * 1.1;
      P.F.x.domain([-xm, xm]); P.F.y.domain([0, Math.max(3, d3.max(nlp)) * 1.05]);
      P.F.g.select('.axis.x').call(d3.axisBottom(P.F.x).ticks(5)); P.F.g.select('.axis.y').call(d3.axisLeft(P.F.y).ticks(5));
      P.draw({ idx, x: lfc, y: nlp, color: (i) => (e.planted[i] ? cat(7) : muted()), r: (i) => (sig[i] ? 3.6 : 1.8), alpha: 0.8 });
    }
    titleA.innerHTML = `<b>Per-cell Wilcoxon</b>: ${fmtInt(e.n_cells_R)} vs ${fmtInt(e.n_cells_NR)} cells`;
    titleB.innerHTML = `<b>Pseudobulk</b>: ${e.n_R} vs ${e.n_NR} patients (≥ 10 cells each)`;
    rd.set({ 'cell-level hits (planted / other)': `${e.tp_nv + e.fp_nv} (${e.tp_nv} / ${e.fp_nv})`, 'pseudobulk hits (planted / other)': `${e.tp_pb + e.fp_pb} (${e.tp_pb} / ${e.fp_pb})`, 'genes tested': fmtInt(e.genes.length), 'top cell-level −log₁₀ p': f2(d3.max(e.nlp_nv)) });
  }

  // ---------------- figure 2: null ----------------
  const fig2 = figure(main, 'Shuffled patients: a comparison where nothing should be found',
    `Each fake comparison splits the 12 patients into two groups that each hold 3 real responders and 3 real non-responders, so the true response signal cancels. Left: number of genes passing 5% FDR in each of ${D.n_perm} such shuffles. Right: the p-value histogram for one shuffle. Under the null it should be flat.`);
  const rd2 = stats(fig2.root);
  const row2 = el('div', { class: 'fig-row' }); fig2.body.append(row2);
  const qA = el('div'), qB = el('div'); row2.append(qA, qB);
  const N = frame(qA, { width: 460, height: 300, x: [0.5, D.n_perm + 0.5], y: [0, 10], xLabel: 'shuffle', yLabel: 'genes with FDR < 5%' });
  legend(qA, [['cells as replicates', cat(1)], ['pseudobulk', cat(0)]]);
  const H = frame(qB, { width: 440, height: 300, x: [0, 1], y: [0, 1], xLabel: 'p-value', yLabel: 'share of genes' });
  legend(qB, [['cells as replicates', cat(1)], ['pseudobulk', cat(0)], ['expected if calibrated', css('--ink')]]);

  function paint2() {
    const e = D.de[ct];
    const mx = d3.max(e.perm, (p) => Math.max(p.naive, p.pb));
    N.y.domain([0, Math.max(5, mx * 1.1)]); N.g.select('.axis.y').call(d3.axisLeft(N.y).ticks(5));
    N.g.select('.axis.x').call(d3.axisBottom(N.x).ticks(5));
    const bw = N.x(1) - N.x(0);
    const flat = e.perm.flatMap((p, k) => [{ k, v: p.naive, s: 0 }, { k, v: p.pb, s: 1 }]);
    N.layer('b').selectAll('rect').data(flat).join('rect')
      .attr('x', (d) => N.x(d.k + 1) - bw * 0.42 + d.s * bw * 0.42).attr('width', bw * 0.4).attr('y', (d) => N.y(d.v)).attr('height', (d) => N.h - N.y(d.v))
      .attr('fill', (d) => (d.s ? cat(0) : cat(1)))
      .on('mousemove', (ev, d) => tip(ev, `shuffle ${d.k + 1}: ${d.v} "significant" genes (${d.s ? 'pseudobulk' : 'cells as replicates'})`)).on('mouseleave', untip);
    const tot = (h) => d3.sum(h) || 1;
    const hs = [[e.perm_hist_nv, cat(1), 0], [e.perm_hist_pb, cat(0), 1]].map(([h, c, s]) => ({ c, s, v: h.map((x) => x / tot(h)) }));
    H.y.domain([0, d3.max(hs.flatMap((h) => h.v)) * 1.08]); H.g.select('.axis.y').call(d3.axisLeft(H.y).ticks(5, '.0%'));
    const w = H.w / 20;
    H.layer('b').selectAll('g').data(hs).join('g').attr('fill', (d) => d.c)
      .selectAll('rect').data((d) => d.v.map((v, b) => ({ v, b, s: d.s }))).join('rect')
      .attr('x', (d) => H.x(d.b / 20) + d.s * w / 2).attr('width', w / 2 - 0.5).attr('y', (d) => H.y(d.v)).attr('height', (d) => H.h - H.y(d.v))
      .on('mousemove', (ev, d) => tip(ev, `p in ${(d.b / 20).toFixed(2)}–${((d.b + 1) / 20).toFixed(2)}: ${pct(d.v, 1)} of genes (${d.s ? 'pseudobulk' : 'cells as replicates'})`)).on('mouseleave', untip);
    H.layer('exp').selectAll('line').data([0.05]).join('line').attr('x1', 0).attr('x2', H.w).attr('y1', (d) => H.y(d)).attr('y2', (d) => H.y(d)).attr('stroke', css('--ink')).attr('stroke-width', 2).attr('stroke-dasharray', '5 4');
    rd2.set({ 'mean false hits, cells as replicates': (d3.mean(e.perm, (p) => p.naive)).toFixed(1), 'mean false hits, pseudobulk': (d3.mean(e.perm, (p) => p.pb)).toFixed(1), 'p < 0.05, one shuffle (cells)': pct(e.perm_hist_nv[0] / tot(e.perm_hist_nv)), 'p < 0.05, one shuffle (pseudobulk)': pct(e.perm_hist_pb[0] / tot(e.perm_hist_pb)) });
  }
  paint(); paint2();

  // prose numbers, all from the JSON
  const T = D.de.Tumour;
  const meanNv = (t) => d3.mean(D.de[t].perm, (p) => p.naive), meanPb = (t) => d3.mean(D.de[t].perm, (p) => p.pb);
  const rankPb = d3.range(T.genes.length).sort((a, b) => T.nlp_pb[b] - T.nlp_pb[a]);
  const top20 = rankPb.slice(0, 20).filter((i) => T.planted[i]).length;
  const nPl = T.planted.filter(Boolean).length;
  const top20exp = (20 * nPl / T.genes.length).toFixed(1);
  const cx = T.genes.indexOf('CXCL9');
  const pexp = (nlp) => (nlp < 3 ? (10 ** -nlp).toFixed(3) : `10<sup>−${Math.round(nlp)}</sup>`);
  const totPbHits = d3.sum(TYPES, (t) => D.de[t].tp_pb + D.de[t].fp_pb);
  section(main, 'What to notice', `<ul>
    <li><b>Shuffled patients, tumour cells</b>: treating cells as replicates gives on average <b>${meanNv('Tumour').toFixed(0)}</b> "significant" genes per shuffle where none exist; pseudobulk gives ${meanPb('Tumour').toFixed(1)}. Try other cell types: the false hits scale with the number of cells (${fmtInt(D.de.Tumour.n_cells)} tumour vs ${fmtInt(D.de.B.n_cells)} B cells), because more cells make the per-cell test more confident about a patient-to-patient difference it mistakes for a group effect.</li>
    <li>The cell-level p-value histogram piles up near 0 under the null; the pseudobulk histogram does not (if anything it leans towards large p-values, i.e. slightly conservative with so few patients). A flat histogram under the null is what "calibrated" means.</li>
    <li><b>Real labels, tumour cells</b>: CXCL9 gets p = ${pexp(T.nlp_nv[cx])} cell-level and p = ${pexp(T.nlp_pb[cx])} pseudobulk. The planted IFN-γ genes are genuinely higher in responders on average, but ${T.n_R} vs ${T.n_NR} patients with heterogeneous tumours (one inflamed non-responder, one excluded responder) is a small study: across all ${TYPES.length} cell types, pseudobulk passes ${totPbHits} gene${totPbHits === 1 ? '' : 's'} at 5% FDR.</li>
    <li>Pseudobulk is not blind: ${top20} of its top 20 tumour genes are among the ${nPl} planted ones (about ${top20exp} expected by chance). It ranks the right genes; it just refuses to claim certainty that 12 patients cannot provide. The honest conclusion is "promising, needs more patients", and the cell-level analysis would have reported p-values like ${pexp(d3.max(T.nlp_nv))}.</li>
  </ul>`);
  steps(main, 'How it works', [
    'For each cell type, take the cells of each patient (drop patients with fewer than ~10 cells of that type).',
    '<b>Sum the raw counts</b> per patient: each patient becomes one "bulk" sample per cell type.',
    'Filter lowly expressed genes, normalise library sizes (log-CPM here; TMM or DESeq2 size factors in practice).',
    'Fit a linear model per gene across patients (response + covariates such as batch or sex), with <b>empirical-Bayes shrinkage</b> of the gene variances, which borrows strength across genes when there are few samples.',
    'Test the response coefficient, adjust for multiple testing (Benjamini–Hochberg FDR).',
    'The cell-level alternative (Wilcoxon on every cell, Scanpy\'s default for <i>marker</i> finding) is fine for describing clusters, wrong for comparing conditions.',
  ]);
  math(main, `pseudobulk  y_pg = log₂ CPM( Σ_{cells c of patient p} counts_cg )
model       y_pg = β₀g + β₁g · responder_p + ε_pg,      ε ~ N(0, σ_g²)
shrinkage   s̃_g² = (d₀ s₀² + d s_g²) / (d₀ + d)           (limma moderated t, d = n_patients − 2)
the trap:   cells within a patient are correlated, so the effective n is close to #patients, not #cells`);
  callout(main, '', 'In biology', `<b>edgeR</b> (quasi-likelihood), <b>DESeq2</b> and <b>limma-voom</b> on pseudobulk counts are the recommended approach; Squair et al. (2021, <i>Nat. Commun.</i>) showed cell-level methods produce hundreds of false discoveries, and Crowell et al. (2020, <i>muscat</i>) benchmarked pseudobulk vs mixed models. Helpers: <code>decoupler.get_pseudobulk</code> / <b>pyDESeq2</b> in Python, <code>AggregateExpression</code> in Seurat, <b>muscat</b> in R. Mixed models with a patient random effect (e.g. NEBULA, MAST with random effects) are an alternative when cell-level structure matters. For differences in cell-type <i>abundance</i>, use compositional methods (scCODA, propeller, Milo).`);
  callout(main, 'warn', 'Pitfalls', `<ul>
    <li><b>Pseudoreplication</b>: using cells as replicates for a between-patient question. The single most common statistical error in single-cell papers.</li>
    <li><b>Confounding</b>: if all responders were sequenced on one chemistry, no model can separate response from batch. Balance the design before sequencing.</li>
    <li><b>Few cells per patient</b> make noisy pseudobulk samples; weight or drop them rather than pretend.</li>
    <li><b>Double dipping</b>: testing genes between clusters that were defined by those same genes.</li>
    <li><b>Using integrated (batch-corrected) values for DE</b>: corrected embeddings are not counts. Model batch as a covariate instead.</li>
  </ul>`);
  seeAlso(main, [['omics', 'bulk-de', 'Bulk differential expression'], ['concepts', 'multiple-testing', 'Multiple testing & FDR'], ['omics', 'sc-trajectory', 'Trajectories & pseudotime'], ['omics', 'sp-data', 'Next section: spatial omics']]);
}
