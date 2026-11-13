// Bulk step 2: exploration with PCA, and batch correction that does (or does not) protect biology.
// Driven by data/omics/bulk_batch.json.
import { header, figure, select, button, stats, section, steps, callout, math, legend, seeAlso, animate, tip, untip, el } from '../lib/ui.js';
import { frame, css, ink, ink2 } from '../lib/plot.js';
import { rng } from '../lib/data.js';
import { stepCard, omicsData } from './_shared.js';
import { PHENOS, phenoColor, CENTRES, centreColor, respColor, f2, f3 } from './_bulk_common.js';
const d3 = window.d3;

const VNAME = { raw: 'no correction', naive: 'naive correction (centre only)', protected: 'protected correction (centre + phenotype kept)' };

export default async function render(main) {
  header(main, 'Exploration & batch effects',
    'The first PCA of almost any multi-centre study shows <b>the centres, not the biology</b>. Removing that technical shift is easy; removing it <b>without also removing real biology that happens to differ between centres</b> is the actual problem.');
  stepCard(main, {
    stage: 'Bulk RNA-seq · step 2 of 11',
    question: 'What is the biggest source of variation in the data, and is it technical or biological?',
    data: 'log-CPM expression of the patients who passed QC, with sequencing centre, pathologist immune phenotype and response.',
    method: 'PCA for exploration; a per-gene linear model of location and scale (ComBat-style) to remove the centre effect, with or without biological covariates protected.',
    verdict: 'stats', why: ' PCA is a look, not a model. The correction itself is a linear model per gene; ComBat adds empirical-Bayes shrinkage. Nothing is being predicted.',
  });
  const B = await omicsData(main, 'bulk_batch.json');
  if (!B) return;
  const N = B.ids.length, V = B.versions;
  const nB = (c) => Object.values(B.counts[c]).reduce((a, b) => a + b, 0);

  // ------------------------------------------------------------------ figure 1: PCA morph
  const fig = figure(main, 'PCA before and after correction',
    'Each dot is a patient on the first two principal components (top 1,000 most variable genes). Switch the correction and watch the dots move. Colour by centre, then by phenotype, in each version.');
  let ver = 'raw', colorBy = 'centre';
  select(fig.controls, { label: 'Data', value: ver, options: Object.entries(VNAME), onChange: (v) => { ver = v; morph(); } });
  select(fig.controls, { label: 'Colour by', value: colorBy, options: [['centre', 'sequencing centre'], ['pheno', 'immune phenotype'], ['resp', 'response']], onChange: (v) => { colorBy = v; paint(); } });
  button(fig.controls, 'Play all three', () => playAll());
  const leg = el('div'); fig.controls.append(leg);
  const rd = stats(fig.root);
  const F = frame(fig.body, { width: 620, height: 400, x: [-1, 1], y: [-1, 1], xLabel: 'PC1', yLabel: 'PC2' });
  // normalise each version's PCs to a common box so the morph is about shape, not axis units
  const norm = {};
  for (const k of Object.keys(V)) {
    const xs = V[k].pc.map((p) => p[0]), ys = V[k].pc.map((p) => p[1]);
    const s = Math.max(d3.max(xs.map(Math.abs)), d3.max(ys.map(Math.abs)));
    norm[k] = V[k].pc.map((p) => [p[0] / s, p[1] / s]);
  }
  F.g.select('.axis.x').call(d3.axisBottom(F.x).tickValues([])); F.g.select('.axis.y').call(d3.axisLeft(F.y).tickValues([]));
  let cur = norm.raw.map((p) => p.slice());
  const dots = F.layer('pts').selectAll('circle').data(d3.range(N)).join('circle').attr('r', 4.5).attr('stroke', css('--surface')).attr('stroke-width', 1.2)
    .on('mousemove', (e, i) => tip(e, `<b>${B.ids[i]}</b><br>${B.centre[i]} · ${B.pheno[i]} · ${B.resp[i] ? 'responder' : 'non-responder'}<br>IFN-γ score ${V[ver].ifng[i].toFixed(2)}`))
    .on('mouseleave', untip);
  const place = () => dots.attr('cx', (i) => F.x(cur[i][0])).attr('cy', (i) => F.y(cur[i][1]));
  let stop = null;
  function morph(done) {
    if (stop) stop();
    const from = cur.map((p) => p.slice()), to = norm[ver];
    let t = 0;
    readout();
    stop = animate((dt) => {
      t = Math.min(1, t + dt / 700);
      const e = d3.easeCubicInOut(t);
      cur = from.map((p, i) => [p[0] + (to[i][0] - p[0]) * e, p[1] + (to[i][1] - p[1]) * e]);
      place();
      if (t >= 1) { stop = null; if (done) done(); return false; }
      return true;
    });
  }
  function playAll() {
    const seq = ['raw', 'naive', 'protected'];
    let k = 0;
    const next = () => { if (k >= seq.length) return; ver = seq[k++]; fig.controls.querySelector('select').value = ver; morph(() => { const s = every0(next); }); };
    next();
  }
  // pause ~0.6 s between morphs using animate() so navigation cancels it
  function every0(fn) { let w = 0; return animate((dt) => { w += dt; if (w > 600) { fn(); return false; } return true; }); }
  function paint() {
    const f = { centre: (i) => centreColor(B.centre[i]), pheno: (i) => phenoColor(B.pheno[i]), resp: (i) => respColor(B.resp[i]) }[colorBy];
    dots.attr('fill', (i) => f(i));
    leg.innerHTML = '';
    legend(leg, { centre: CENTRES.map((c) => [c, centreColor(c)]), pheno: PHENOS.map((p) => [p, phenoColor(p)]), resp: [['responder', respColor(1)], ['non-responder', respColor(0)]] }[colorBy]);
  }
  function readout() {
    const v = V[ver];
    F.g.selectAll('.ax-label').filter((d, i) => i === 0).text(`PC1 (${(100 * v.var[0]).toFixed(0)}% of variance)`);
    F.g.selectAll('.ax-label').filter((d, i) => i === 1).text(`PC2 (${(100 * v.var[1]).toFixed(0)}%)`);
    rd.set({ 'R² of PC1 with centre': f3(v.r2_batch[0]), 'silhouette by centre': f3(v.sil_batch), 'silhouette by phenotype': f3(v.sil_pheno), ['IFN-γ gap (truth ' + f2(B.truth_effect) + ')']: f2(v.effect) });
  }
  place(); paint(); readout();

  // ------------------------------------------------------------------ figure 2: what correction does to biology
  const fig2 = figure(main, 'Why naive correction erases biology',
    'Left: how many patients of each phenotype each centre enrolled. Right: the IFN-γ module score per patient, split by phenotype and centre. Solid bars are group means. The dashed line in the inflamed column marks where the inflamed mean would sit if the true (noise-free) inflamed-versus-other gap were preserved. Hover dots.');
  let ver2 = 'naive';
  select(fig2.controls, { label: 'Data', value: ver2, options: Object.entries(VNAME), onChange: (v) => { ver2 = v; draw2(); } });
  legend(fig2.controls, CENTRES.map((c) => [c, centreColor(c)]));
  const rd2 = stats(fig2.root);
  const row = el('div', { class: 'fig-row' }); fig2.body.append(row);
  const Lw = el('div'), Rw = el('div'); row.append(Lw, Rw);
  Lw.style.flex = '0 1 330px';
  // stacked counts
  const FL = frame(Lw, { width: 330, height: 340, x: [0, 1], y: [0, d3.max(CENTRES, nB)], yLabel: 'patients', xScale: d3.scaleBand().padding(0.35) });
  FL.x.domain(CENTRES); FL.g.select('.axis.x').call(d3.axisBottom(FL.x));
  CENTRES.forEach((c) => {
    let acc = 0;
    PHENOS.forEach((p) => {
      const n = B.counts[c][p];
      FL.layer('bars').append('rect').attr('x', FL.x(c)).attr('width', FL.x.bandwidth()).attr('y', FL.y(acc + n)).attr('height', FL.y(acc) - FL.y(acc + n))
        .attr('fill', phenoColor(p)).attr('stroke', css('--surface')).attr('stroke-width', 1)
        .on('mousemove', (e) => tip(e, `<b>${c}</b>: ${n} ${p} (${(100 * n / nB(c)).toFixed(0)}%)`)).on('mouseleave', untip);
      if (n >= 8) FL.layer('lab').append('text').attr('x', FL.x(c) + FL.x.bandwidth() / 2).attr('y', FL.y(acc + n / 2) + 4).attr('text-anchor', 'middle').attr('font-size', 11).attr('fill', css('--surface')).text(`${p} ${n}`);
      acc += n;
    });
  });
  const r = rng(11), jit = B.ids.map(() => r() - 0.5);
  const allIfng = Object.values(V).flatMap((v) => v.ifng);
  const FRr = frame(Rw, { width: 560, height: 340, x: [0, 1], y: d3.extent(allIfng), yLabel: 'IFN-γ module score', xScale: d3.scaleBand().padding(0.12) });
  FRr.x.domain(PHENOS); FRr.y.nice(); FRr.g.select('.axis.x').call(d3.axisBottom(FRr.x)); FRr.g.select('.axis.y').call(d3.axisLeft(FRr.y));
  const bw = FRr.x.bandwidth();
  const xpos = (i) => FRr.x(B.pheno[i]) + bw * (B.centre[i] === 'Centre A' ? 0.28 : 0.72) + jit[i] * bw * 0.3;
  const pts2 = FRr.layer('pts').selectAll('circle').data(d3.range(N)).join('circle').attr('r', 4).attr('cx', xpos)
    .attr('fill', (i) => centreColor(B.centre[i])).attr('stroke', css('--surface')).attr('stroke-width', 1.1)
    .on('mousemove', (e, i) => tip(e, `<b>${B.ids[i]}</b> · ${B.centre[i]} · ${B.pheno[i]}<br>score ${V[ver2].ifng[i].toFixed(2)} (raw ${V.raw.ifng[i].toFixed(2)})`)).on('mouseleave', untip);
  const meanLayer = FRr.layer('means');
  function draw2() {
    const v = V[ver2];
    pts2.attr('cy', (i) => FRr.y(v.ifng[i]));
    const groups = [];
    PHENOS.forEach((p) => CENTRES.forEach((c) => {
      const ix = d3.range(N).filter((i) => B.pheno[i] === p && B.centre[i] === c);
      groups.push({ p, c, m: d3.mean(ix, (i) => v.ifng[i]) });
    }));
    meanLayer.selectAll('line').data(groups).join('line').attr('x1', (d) => FRr.x(d.p) + bw * (d.c === 'Centre A' ? 0.12 : 0.56)).attr('x2', (d) => FRr.x(d.p) + bw * (d.c === 'Centre A' ? 0.44 : 0.88))
      .attr('y1', (d) => FRr.y(d.m)).attr('y2', (d) => FRr.y(d.m)).attr('stroke', ink()).attr('stroke-width', 2.5);
    const nonInf = d3.mean(d3.range(N).filter((i) => B.pheno[i] !== 'inflamed'), (i) => v.ifng[i]);
    meanLayer.selectAll('path').data([0]).join('path').attr('d', `M${FRr.x('inflamed')},${FRr.y(nonInf + B.truth_effect)}h${bw}`)
      .attr('stroke', ink2()).attr('stroke-dasharray', '5 4').attr('stroke-width', 1.5);
    const gap = (c) => { const a = groups.find((g) => g.c === c && g.p === 'inflamed').m; const b = d3.mean(d3.range(N).filter((i) => B.centre[i] === c && B.pheno[i] !== 'inflamed'), (i) => v.ifng[i]); return a - b; };
    rd2.set({ 'inflamed − other gap': `${f2(v.effect)} (truth ${f2(B.truth_effect)})`, 'gap kept': `${(100 * v.effect / B.truth_effect).toFixed(0)}%`, 'AUC of IFN-γ score for response': f3(v.auc_ifng), 'gap within Centre B': f2(gap('Centre B')) });
  }
  draw2();

  const shareInf = (c) => B.counts[c].inflamed / nB(c);
  section(main, 'What to notice', `<ul>
    <li><b>No correction:</b> PC1 explains ${(100 * V.raw.var[0]).toFixed(0)}% of the variance and its R² with centre is ${f3(V.raw.r2_batch[0])}. The biggest thing in this data set is which building the sample was sequenced in.</li>
    <li><b>Naive correction</b> (subtract each centre's mean per gene) makes centre invisible (R² ${V.naive.r2_batch[0].toExponential(0)}), but the IFN-γ gap between inflamed and other tumours falls from ${f2(V.raw.effect)} to ${f2(V.naive.effect)}, and the AUC of the IFN-γ score for response from ${f3(V.raw.auc_ifng)} to ${f3(V.naive.auc_ifng)}. Why: Centre B is ${(100 * shareInf('Centre B')).toFixed(0)}% inflamed versus ${(100 * shareInf('Centre A')).toFixed(0)}% for Centre A, so part of "Centre B's mean" is biology. Subtracting it pushes Centre B's inflamed tumours down (look at the right-hand dots in the inflamed column).</li>
    <li><b>Protected correction</b> estimates the centre shift <i>while holding phenotype fixed</i>. Centre is mostly gone (silhouette ${f3(V.protected.sil_batch)}), phenotypes separate best (silhouette ${f3(V.protected.sil_pheno)}), and the gap is ${f2(V.protected.effect)} against a truth of ${f2(B.truth_effect)}.</li>
    <li>PC1 of the protected data still has R² ${f3(V.protected.r2_batch[0])} with centre. That is correct: centre and phenotype are correlated, so a biological axis is partly predictable from centre. A correction that drives this to zero has removed biology.</li>
  </ul>`);
  steps(main, 'How it works', [
    'Run PCA on the log-CPM matrix and colour by every known technical and biological variable. Here PC1 lines up with centre.',
    'Cross-tabulate centre against biology. If they are unbalanced (here: phenotype), the batch effect and the biology are <b>confounded</b>, and any correction has to be told which differences to keep.',
    'For each gene fit <code>expression ~ centre + covariates to protect</code>. The centre coefficient is the batch shift estimated <i>within</i> levels of the protected covariates.',
    'Subtract the centre coefficient, and rescale residuals so each centre has the same spread (location/scale). ComBat additionally shrinks the per-gene shifts towards their average across genes (empirical Bayes), which helps with small batches.',
    'Check: PCA again, the centre-silhouette should fall, the biology-silhouette should not, and a known biological effect (here the IFN-γ gap) should survive.',
    'For differential expression and classifiers, it is usually better <b>not</b> to correct the data at all and instead put centre in the model as a covariate (<a href="#/omics/bulk-de">next steps</a>).',
  ]);
  math(main, 'y_gi = α_g + β_g·centre_i + γ_g·phenotype_i + ε_gi\ncorrected y_gi = y_gi − β̂_g·centre_i      (naive: γ omitted, so β̂ absorbs part of the biology)');

  callout(main, '', 'In biology', '<b>ComBat</b> (sva package; <code>pycombat</code> in Python) and its count version <b>ComBat-seq</b> are the standard tools; the <code>mod=</code> argument is exactly the "protect these covariates" choice shown here. <b>limma::removeBatchEffect</b> does the same linear-model removal for plotting. For DE in <b>DESeq2</b>, <b>edgeR</b> or <b>limma</b>, add the batch to the design formula (<code>~ centre + response</code>) instead of correcting first. <b>RUVseq</b> and <b>SVA</b> estimate hidden batch factors when the batch is unknown. Single-cell data needs different tools (Harmony, scVI), see <a href="#/omics/sc-integration">batch integration</a>.');
  callout(main, 'warn', 'Pitfalls', `<ul>
    <li><b>Correcting away the signal.</b> If biology is unbalanced across batches and you do not protect it, the correction removes it, as above.</li>
    <li><b>Protecting the outcome, then testing it.</b> If you protect the variable you will later test or predict (response), ComBat preserves its group differences by construction, and downstream p-values become too small (Nygaard et al., <i>Biostatistics</i> 2016). Here we protected the pathologist's phenotype, which is a legitimate known covariate, but be aware the <a href="#/omics/bulk-subtypes">subtype clustering</a> on this data is then partly helped by that choice.</li>
    <li><b>Correcting within cross-validation.</b> A correction fitted on all samples uses information from test samples. For a deployable classifier, the correction (or reference batch) must be fixed in advance; see <a href="#/omics/bulk-validation">validation</a>.</li>
    <li><b>Complete confounding cannot be fixed.</b> If every responder were from Centre B, no method could separate centre from response. That is a design problem: randomise samples across sequencing runs.</li>
  </ul>`);
  seeAlso(main, [['omics', 'bulk-qc', 'Previous: QC'], ['omics', 'bulk-subtypes', 'Next: subtypes'], ['omics', 'sc-integration', 'Single-cell integration'], ['concepts', 'pca', 'PCA'], ['concepts', 'train-test', 'Leakage']]);
}
