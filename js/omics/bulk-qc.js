// Bulk step 1: QC, normalisation and outlier detection. Driven by data/omics/bulk_qc.json.
import { header, figure, slider, select, stats, section, steps, callout, math, legend, seeAlso, tip, untip, el } from '../lib/ui.js';
import { frame, cat, css, ink, ink2, muted } from '../lib/plot.js';
import { rng } from '../lib/data.js';
import { stepCard, omicsData } from './_shared.js';
import { median, mad } from './_bulk_common.js';
const d3 = window.d3;

// The four QC metrics, each with the scale on which a MAD rule is applied and the "bad" side.
const METRICS = {
  medcor: { name: 'Median correlation to other samples', short: 'median correlation', side: 'low', tr: (v) => v, inv: (v) => v, fmt: (v) => v.toFixed(3), log: false, kind: 'statistics' },
  rdist: { name: 'Robust PCA distance (MCD Mahalanobis)', short: 'robust distance', side: 'high', tr: Math.log, inv: Math.exp, fmt: (v) => v.toFixed(1), log: true, kind: 'robust statistics' },
  iso: { name: 'Isolation-forest anomaly score', short: 'isolation forest', side: 'high', tr: (v) => v, inv: (v) => v, fmt: (v) => v.toFixed(3), log: false, kind: 'unsupervised ML' },
  lib: { name: 'Library size (total reads)', short: 'library size', side: 'low', tr: Math.log10, inv: (v) => 10 ** v, fmt: (v) => (v / 1e6).toFixed(1) + 'M', log: true, kind: 'statistics' },
};

export default async function render(main) {
  header(main, 'QC, normalisation & outliers',
    'Before asking any biological question, find the libraries that failed and put every sample on the same scale. The tools are mostly <b>robust statistics</b>: a sample is an outlier if it sits many <b>median absolute deviations</b> from the rest.');
  stepCard(main, {
    stage: 'Bulk RNA-seq · step 1 of 11',
    question: 'Which of the tumour libraries are technically broken, and how do we make samples sequenced to different depths comparable?',
    data: 'A raw count matrix: patients × genes, plus per-sample metrics (library size, correlation to other samples, position in PCA space).',
    method: 'log-CPM normalisation; MAD thresholds on QC metrics; robust Mahalanobis distance and an isolation forest on the top principal components.',
    verdict: 'both', why: ' Thresholds on a few metrics are statistics and usually enough. Multivariate outlier detectors (isolation forest) are unsupervised ML: useful to rank suspicious samples, but they have no natural cut-off.',
  });
  const Q = await omicsData(main, 'bulk_qc.json');
  if (!Q) return;
  const N = Q.ids.length;
  const planted = Object.entries(Q.planted).map(([what, id]) => ({ what, id, i: Q.ids.indexOf(id) }));
  const plantedOf = (i) => planted.find((p) => p.i === i);

  function flags(key, k) {
    const M = METRICS[key], v = Q[key].map(M.tr), m = median(v), s = mad(v);
    const thr = M.side === 'low' ? m - k * s : m + k * s;
    return { thr, raw: M.inv(thr), f: v.map((x) => (M.side === 'low' ? x < thr : x > thr)) };
  }

  // ------------------------------------------------------------------ figure 1: outlier detection
  const fig = figure(main, 'Find the failed libraries',
    'Left: one dot per sample on the chosen QC metric; the dashed line is median ± k·MAD. Right: the same samples on the first two principal components of log-CPM expression (top 1,000 variable genes). Ringed dots are the three failures planted in the simulation. Hover any dot.');
  let key = 'medcor', k = Q.k_default;
  select(fig.controls, { label: 'QC metric', value: key, options: Object.entries(METRICS).map(([kk, m]) => [kk, m.name]), onChange: (v) => { key = v; draw(); } });
  slider(fig.controls, { label: 'Threshold k (MADs)', min: 1.5, max: 6, step: 0.5, value: k, fmt: (v) => v.toFixed(1), onInput: (v) => { k = v; draw(); } });
  legend(fig.controls, [['flagged by this rule', cat(7)], ['passes', muted()]]);
  const rd = stats(fig.root);
  const row = el('div', { class: 'fig-row' }); fig.body.append(row);
  const L = el('div'), R = el('div'); row.append(L, R);
  const tbl = el('div', { class: 'prose', style: 'max-width:none;margin-top:6px' }); fig.root.append(tbl);

  const r = rng(3), jit = Q.ids.map(() => r());
  const pcx = Q.pc.map((p) => p[0]), pcy = Q.pc.map((p) => p[1]);
  const FR = frame(R, { width: 460, height: 340, x: d3.extent(pcx), y: d3.extent(pcy), xLabel: `PC1 (${(100 * Q.pcvar[0]).toFixed(0)}% of variance)`, yLabel: `PC2 (${(100 * Q.pcvar[1]).toFixed(0)}%)` });
  FR.x.nice(); FR.y.nice(); FR.g.select('.axis.x').call(d3.axisBottom(FR.x).ticks(5)); FR.g.select('.axis.y').call(d3.axisLeft(FR.y).ticks(5));
  const hov = (e, i) => tip(e, `<b>${Q.ids[i]}</b> · ${Q.centre[i]}${plantedOf(i) ? ` · <b>planted: ${plantedOf(i).what}</b>` : ''}<br>library ${(Q.lib[i] / 1e6).toFixed(2)}M reads<br>median correlation ${Q.medcor[i].toFixed(3)}<br>robust distance ${Q.rdist[i].toFixed(1)}<br>isolation score ${Q.iso[i].toFixed(3)}`);
  const ord = d3.range(N).sort((a, b) => !!plantedOf(a) - !!plantedOf(b));
  const pcDots = FR.layer('pts').selectAll('circle').data(ord).join('circle').attr('cx', (i) => FR.x(pcx[i])).attr('cy', (i) => FR.y(pcy[i]))
    .on('mousemove', hov).on('mouseleave', untip);
  const LOFF = { 'degraded (strong)': [0, 20, 'middle'], 'degraded (mild)': [6, -10, 'start'] };
  const lo = (p) => LOFF[p.what] || [9, 4, 'start'];
  FR.layer('lab').selectAll('text').data(planted).join('text').attr('x', (p) => FR.x(pcx[p.i]) + lo(p)[0]).attr('y', (p) => FR.y(pcy[p.i]) + lo(p)[1])
    .attr('text-anchor', (p) => lo(p)[2]).attr('font-size', 11).attr('fill', ink2()).text((p) => `${p.id} (${p.what})`);

  let FL;
  function draw() {
    const M = METRICS[key], F = flags(key, k);
    L.innerHTML = '';
    const v = Q[key];
    const ext = d3.extent([...v, F.raw]);
    FL = frame(L, { width: 460, height: 340, x: ext, y: [0, 1], xLabel: M.name + (M.log ? ' (log scale)' : ''), xScale: M.log ? d3.scaleLog() : d3.scaleLinear(), axes: true, grid: false });
    FL.x.nice();
    FL.g.select('.axis.x').call(d3.axisBottom(FL.x).ticks(5, M.log ? '~s' : '.3~f'));
    FL.g.select('.axis.y').remove();
    FL.layer('thr').append('line').attr('x1', FL.x(F.raw)).attr('x2', FL.x(F.raw)).attr('y1', 0).attr('y2', FL.h).attr('stroke', ink()).attr('stroke-dasharray', '5 4').attr('stroke-width', 1.5);
    const leftHalf = FL.x(F.raw) < FL.w / 2;
    FL.layer('thr').append('text').attr('x', FL.x(F.raw) + (leftHalf ? 6 : -6)).attr('y', 12).attr('text-anchor', leftHalf ? 'start' : 'end')
      .attr('font-size', 11).attr('fill', ink2()).text(`median ${M.side === 'low' ? '−' : '+'} ${k.toFixed(1)}·MAD = ${M.fmt(F.raw)}`);
    FL.layer('pts').selectAll('circle').data(ord).join('circle').attr('cx', (i) => FL.x(v[i])).attr('cy', (i) => FL.y(0.08 + 0.78 * jit[i]))
      .attr('r', (i) => (plantedOf(i) ? 6.5 : 4.5)).attr('fill', (i) => (F.f[i] ? cat(7) : muted()))
      .attr('stroke', (i) => (plantedOf(i) ? ink() : css('--surface'))).attr('stroke-width', (i) => (plantedOf(i) ? 2 : 1.2))
      .on('mousemove', hov).on('mouseleave', untip);
    FL.layer('lab').selectAll('text').data(planted).join('text').attr('x', (p) => FL.x(v[p.i])).attr('y', (p) => FL.y(0.08 + 0.78 * jit[p.i]) - 9)
      .attr('text-anchor', 'middle').attr('font-size', 11).attr('fill', ink2()).text((p) => p.id);
    pcDots.attr('r', (i) => (plantedOf(i) ? 6.5 : 4.5)).attr('fill', (i) => (F.f[i] ? cat(7) : muted()))
      .attr('stroke', (i) => (plantedOf(i) ? ink() : css('--surface'))).attr('stroke-width', (i) => (plantedOf(i) ? 2 : 1.2));

    const caught = planted.filter((p) => F.f[p.i]);
    const nflag = F.f.filter(Boolean).length;
    rd.set({ 'flagged by this rule': nflag, 'planted failures caught': `${caught.length} of ${planted.length}`, 'false alarms': nflag - caught.length, 'method family': M.kind });

    // table: every sample flagged by any rule at this k
    const all = Object.fromEntries(Object.keys(METRICS).map((kk) => [kk, flags(kk, k).f]));
    const excl = d3.range(N).filter((i) => all.medcor[i] || all.rdist[i]);
    const anyIdx = d3.range(N).filter((i) => Object.values(all).some((f) => f[i]));
    const show = anyIdx.slice(0, 14);
    tbl.innerHTML = `<table class="simple"><tr><th>sample at k = ${k.toFixed(1)}</th>${Object.values(METRICS).map((m) => `<th>${m.short}</th>`).join('')}<th>excluded?</th></tr>` +
      show.map((i) => `<tr><td><b>${Q.ids[i]}</b>${plantedOf(i) ? ` <span class="tag">planted: ${plantedOf(i).what}</span>` : ''}</td>${Object.keys(METRICS).map((kk) => `<td>${all[kk][i] ? '●' : ''}</td>`).join('')}<td>${excl.includes(i) ? 'yes' : ''}</td></tr>`).join('') +
      `</table><p class="fig-caption">${anyIdx.length} samples are flagged by at least one rule${anyIdx.length > show.length ? ` (first ${show.length} shown)` : ''}. The pipeline excludes a sample only if an <i>expression-based</i> rule fires (median correlation or robust distance): here that is ${excl.length} sample${excl.length === 1 ? '' : 's'}. A shallow library alone is not a failure.</p>`;
  }
  draw();

  // ------------------------------------------------------------------ figure 2: normalisation
  const ND = Q.norm_demo;
  const fig2 = figure(main, 'Why normalise: the same samples on three scales',
    `${ND.ids.length} samples, ordered by sequencing depth (left = shallowest). Each box is the distribution of genes in one sample: whiskers at the 10th and 90th percentile, box at the 25th–75th, line at the median. Hover a box.`);
  let tr = 'raw';
  const TR = { raw: ['raw counts', 'log10(count + 1)'], cpm: ['counts per million', 'log10(CPM + 1)'], logcpm: ['log-CPM (edgeR / limma-voom input)', 'log2(CPM + 1)'] };
  select(fig2.controls, { label: 'Scale', value: tr, options: Object.entries(TR).map(([kk, v]) => [kk, v[0]]), onChange: (v) => { tr = v; draw2(); } });
  const rd2 = stats(fig2.root);
  const NW = el('div'); fig2.body.append(NW);
  const loglib = ND.lib.map(Math.log10);
  function draw2() {
    NW.innerHTML = '';
    const Qs = ND[tr], all = Qs.flat();
    const F = frame(NW, { width: 900, height: 300, x: [0, 1], y: d3.extent(all), yLabel: TR[tr][1], xLabel: 'sample, ordered by library size →', xScale: d3.scaleBand().padding(0.25) });
    F.x.domain(ND.ids); F.y.nice();
    F.g.select('.axis.x').call(d3.axisBottom(F.x).tickValues([]));
    F.g.select('.axis.y').call(d3.axisLeft(F.y).ticks(5));
    const bw = F.x.bandwidth();
    const g = F.layer('box').selectAll('g').data(ND.ids.map((id, i) => ({ id, i, q: Qs[i] }))).join('g').attr('transform', (d) => `translate(${F.x(d.id)},0)`);
    g.append('line').attr('x1', bw / 2).attr('x2', bw / 2).attr('y1', (d) => F.y(d.q[0])).attr('y2', (d) => F.y(d.q[4])).attr('stroke', ink2()).attr('stroke-width', 1.2);
    g.append('rect').attr('width', bw).attr('y', (d) => F.y(d.q[3])).attr('height', (d) => F.y(d.q[1]) - F.y(d.q[3])).attr('rx', 2)
      .attr('fill', (d) => (d.id === Q.planted['low depth'] ? cat(7) : cat(0))).attr('opacity', 0.85);
    g.append('line').attr('x1', 0).attr('x2', bw).attr('y1', (d) => F.y(d.q[2])).attr('y2', (d) => F.y(d.q[2])).attr('stroke', css('--surface')).attr('stroke-width', 2);
    g.append('rect').attr('width', bw).attr('y', 0).attr('height', F.h).attr('fill', 'transparent')
      .on('mousemove', (e, d) => tip(e, `<b>${d.id}</b>${d.id === Q.planted['low depth'] ? ' (planted low depth)' : ''}<br>${(ND.lib[d.i] / 1e6).toFixed(2)}M reads<br>median ${d.q[2].toFixed(2)}`)).on('mouseleave', untip);
    const med = Qs.map((q) => q[2]);
    rd2.set({ 'spread of sample medians (SD)': d3.deviation(med).toFixed(3), 'corr(median, log library size)': corr(med, loglib).toFixed(2), 'genes kept by expression filter': `${Q.n_genes_kept.toLocaleString()} of ${Q.n_genes.toLocaleString()}` });
  }
  const corr = (a, b) => { const ma = d3.mean(a), mb = d3.mean(b); let s = 0, sa = 0, sb = 0; a.forEach((v, i) => { s += (v - ma) * (b[i] - mb); sa += (v - ma) ** 2; sb += (b[i] - mb) ** 2; }); return s / Math.sqrt(sa * sb); };
  draw2();
  legend(fig2.root, [['sample', cat(0)], [`${Q.planted['low depth']} (planted low depth)`, cat(7)]]);

  // numbers for prose
  const at = (kk, kv) => flags(kk, kv).f.map((f, i) => (f ? Q.ids[i] : null)).filter(Boolean);
  const isoFalse = at('iso', Q.k_default).filter((id) => !Object.values(Q.planted).includes(id));
  const libFlag = at('lib', Q.k_default);
  const med2 = at('medcor', 2);
  const corrRaw = corr(ND.raw.map((q) => q[2]), loglib), corrCpm = corr(ND.cpm.map((q) => q[2]), loglib);

  section(main, 'What to notice', `<ul>
    <li>At the default k = ${Q.k_default}, <b>median correlation</b> catches all three planted failures (${planted.map((p) => p.id).join(', ')}) with no false alarms. A broken library simply does not look like the others.</li>
    <li>Drop k to 2 on the same metric: ${med2.length} samples are flagged. MAD rules assume most samples are fine; a small k turns normal biological spread into "outliers".</li>
    <li>Switch to <b>library size</b>: at k = ${Q.k_default} it flags ${libFlag.join(', ')}. Only ${Q.planted['low depth']} is a real failure; the others are just shallower libraries with normal expression profiles, and normalisation handles them.</li>
    <li>Switch to the <b>isolation forest</b>: at k = ${Q.k_default} it also flags ${isoFalse.join(', ')}, which are healthy. The PCA has two big clouds (the two centres, see the <a href="#/omics/bulk-batch">next step</a>), and a tree-based anomaly score finds the edges of each cloud "unusual". It ranks samples well; the cut-off is still your call.</li>
    <li>In the normalisation figure, raw medians rise with depth (correlation ${corrRaw.toFixed(2)}); after CPM it drops to ${corrCpm.toFixed(2)}. The shallow library ${Q.planted['low depth']} keeps a wider, lumpier box even after CPM: dividing by depth cannot create information that was never sequenced.</li>
  </ul>`);

  steps(main, 'How it works', [
    '<b>Library size:</b> total reads per sample. Divide every count by it (×10⁶) to get counts per million (CPM). This removes the depth difference, the biggest technical effect in any RNA-seq data.',
    '<b>Log-transform:</b> log2(CPM + 1). Counts are right-skewed and their variance grows with the mean; the log makes fold changes additive and lets linear models and PCA behave.',
    `<b>Filter genes:</b> keep genes with CPM ≥ 1 in at least 10% of samples (an edgeR <code>filterByExpr</code>-style rule). In this mock every one of the ${Q.n_genes.toLocaleString()} genes passes; in real data the rule usually removes a large share of genes that are noise.`,
    '<b>QC metrics:</b> for each sample, the median Pearson correlation of its log-CPM profile with every other sample, and its position in PCA space.',
    '<b>Robust distance:</b> fit a minimum covariance determinant (MCD) estimate of centre and spread on the top 5 PCs, which ignores the most extreme samples, then compute each sample\'s Mahalanobis distance from it.',
    '<b>Isolation forest:</b> random trees split the PC space at random; anomalies are isolated in few splits. The score is the average path length, inverted.',
    '<b>MAD rule:</b> flag samples beyond median ± k × 1.4826 × MAD (the 1.4826 makes MAD equal the SD for normal data). Exclude only on expression-based evidence, and write down which samples and why before looking at outcomes.',
  ]);
  math(main, 'CPM_gi = count_gi / libsize_i × 10⁶          logCPM = log2(CPM + 1)\nMAD = median_i |x_i − median(x)|          flag if  x_i < median − k·1.4826·MAD   (or > for "high is bad")');

  callout(main, '', 'In biology', '<b>edgeR</b> uses TMM normalisation (a trimmed mean of log ratios) on top of library size, and <b>DESeq2</b> uses median-of-ratios size factors; both protect against a few very highly expressed genes eating the library. <b>limma-voom</b> works on log-CPM with precision weights. Sample QC is usually <b>MultiQC</b> plus RSeQC/Qualimap metrics (5′–3′ coverage bias for degradation, rRNA and mitochondrial fraction), followed by a PCA and a sample correlation heatmap. Isolation forests and autoencoders appear in large-cohort QC pipelines, but the thresholds people trust are still simple ones.');
  callout(main, 'warn', 'Pitfalls', `<ul>
    <li><b>Deciding exclusions after seeing the result.</b> Dropping the samples that "hurt" your biomarker is a form of leakage. Fix the QC rule first.</li>
    <li><b>Outlier ≠ failure.</b> A genuinely unusual tumour (rare histology, extreme immune infiltration) can look like an outlier. Check the reason before you remove it.</li>
    <li><b>Batch structure fools multivariate detectors.</b> With two centres in the data, "distance from the centre" is distance from a point between two clouds. Detect outliers within batch, or after looking at the PCA.</li>
    <li>CPM is not enough for differential expression when a few genes dominate the library: use TMM or median-of-ratios factors.</li>
  </ul>`);
  seeAlso(main, [['omics', 'bulk-batch', 'Next: batch effects'], ['omics', 'sc-qc', 'Single-cell QC & doublets'], ['concepts', 'scaling', 'Feature scaling'], ['concepts', 'pca', 'PCA'], ['omics', 'overview', 'The study map']]);
}
