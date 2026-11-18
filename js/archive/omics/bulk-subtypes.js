// Bulk step 3: unsupervised subtypes via consensus k-means, checked with silhouette, PAC and ARI.
// Driven by data/omics/bulk_subtypes.json.
import { header, figure, slider, select, stats, section, steps, callout, math, legend, seeAlso, tip, untip, el } from '../../lib/ui.js';
import { frame, cat, css, ink, ink2, sequential } from '../../lib/plot.js';
import { stepCard, omicsData } from '../../omics/_shared.js';
import { PHENOS, phenoColor, CENTRES, centreColor, respColor, f2, f3 } from '../../omics/_bulk_common.js';
const d3 = window.d3;

const VNAME = { protected: 'batch-corrected (protected)', raw: 'uncorrected log-CPM' };

/** Map arbitrary cluster ids to colour slots: greedily match clusters to phenotypes by overlap. */
function relabel(cl, pheno) {
  const K = d3.max(cl) + 1, pairs = [];
  for (let c = 0; c < K; c++) PHENOS.forEach((p, j) => pairs.push({ c, j, n: cl.filter((v, i) => v === c && pheno[i] === p).length }));
  pairs.sort((a, b) => b.n - a.n);
  const map = {}, used = new Set();
  pairs.forEach(({ c, j }) => { if (!(c in map) && !used.has(j)) { map[c] = j; used.add(j); } });
  let next = 3;
  for (let c = 0; c < K; c++) if (!(c in map)) map[c] = next++;
  return map;
}

export default async function render(main) {
  header(main, 'Unsupervised subtypes',
    'Do the tumours fall into natural groups? Clustering will <b>always</b> return k groups, whatever k you ask for. The work is checking whether the groups are <b>stable</b>, <b>well separated</b> and <b>not just the batch</b>.');
  stepCard(main, {
    stage: 'Bulk RNA-seq · step 3 of 11',
    question: 'Are there molecular subtypes of these tumours, and do they match the immune phenotypes a pathologist sees?',
    data: 'Top 10 principal components of the 1,000 most variable genes, for each patient that passed QC. No labels are used.',
    method: 'Consensus clustering: k-means on 100 random 80% subsamples, then hierarchical clustering of how often each pair of patients lands together. Checked with silhouette, PAC and agreement (ARI) with known labels.',
    verdict: 'ml', why: ' Clustering is unsupervised learning. The consensus resampling and the external checks are what make it trustworthy.',
  });
  const S = await omicsData(main, 'bulk_subtypes.json');
  if (!S) return;
  const N = S.ids.length, KS = S.ks;

  // ------------------------------------------------------------------ figure 1: clusters on PCA + consensus matrix
  const fig = figure(main, 'Pick k and look at the clusters',
    'Left: patients on PC1–PC2, coloured by cluster (or by a known label to compare). Cluster colours are matched to the phenotype they overlap most. Right: the consensus matrix for 90 patients: cell (i, j) is the fraction of subsamples in which patients i and j were clustered together. The strips show each patient\'s cluster and pathologist phenotype.');
  let ver = 'protected', k = 3, colorBy = 'cluster';
  const kSl = slider(fig.controls, { label: 'k (clusters)', min: KS[0], max: KS[KS.length - 1], step: 1, value: k, onInput: (v) => { k = v; draw(); } });
  select(fig.controls, { label: 'Data', value: ver, options: Object.entries(VNAME), onChange: (v) => { ver = v; draw(); } });
  select(fig.controls, { label: 'Colour by', value: colorBy, options: [['cluster', 'cluster'], ['pheno', 'pathologist phenotype'], ['centre', 'centre'], ['resp', 'response']], onChange: (v) => { colorBy = v; draw(); } });
  const leg = el('div'); fig.controls.append(leg);
  const rd = stats(fig.root);
  const row = el('div', { class: 'fig-row' }); fig.body.append(row);
  const Lw = el('div'), Rw = el('div'); row.append(Lw, Rw);

  function draw() {
    const D = S.data[ver], R = D.k[k], map = relabel(R.cl, S.pheno);
    const ccol = (i) => cat(map[R.cl[i]]);
    const f = { cluster: ccol, pheno: (i) => phenoColor(S.pheno[i]), centre: (i) => centreColor(S.centre[i]), resp: (i) => respColor(S.resp[i]) }[colorBy];
    Lw.innerHTML = ''; Rw.innerHTML = '';
    const xs = D.pc.map((p) => p[0]), ys = D.pc.map((p) => p[1]);
    const F = frame(Lw, { width: 460, height: 400, x: d3.extent(xs), y: d3.extent(ys), xLabel: 'PC1', yLabel: 'PC2' });
    F.x.nice(); F.y.nice(); F.g.select('.axis.x').call(d3.axisBottom(F.x).ticks(5)); F.g.select('.axis.y').call(d3.axisLeft(F.y).ticks(5));
    F.layer('pts').selectAll('circle').data(d3.range(N)).join('circle').attr('cx', (i) => F.x(xs[i])).attr('cy', (i) => F.y(ys[i])).attr('r', 4.5)
      .attr('fill', f).attr('stroke', css('--surface')).attr('stroke-width', 1.2)
      .on('mousemove', (e, i) => tip(e, `<b>${S.ids[i]}</b><br>cluster ${map[R.cl[i]] + 1}<br>${S.pheno[i]} · ${S.centre[i]} · ${S.resp[i] ? 'responder' : 'non-responder'}`)).on('mouseleave', untip);
    const used = [...new Set(R.cl.map((c) => map[c]))].sort();
    leg.innerHTML = '';
    legend(leg, { cluster: used.map((s) => [`cluster ${s + 1}`, cat(s)]), pheno: PHENOS.map((p) => [p, phenoColor(p)]), centre: CENTRES.map((c) => [c, centreColor(c)]), resp: [['responder', respColor(1)], ['non-responder', respColor(0)]] }[colorBy]);

    // consensus matrix (stored for the protected data only)
    const P = S.data.protected.k[k], pmap = relabel(P.cl, S.pheno), m = P.cm.length, cs = 4, off = 22;
    const svg = d3.select(Rw).append('svg').attr('viewBox', `0 0 ${off + m * cs + 10} ${off + m * cs + 22}`).attr('width', off + m * cs + 10);
    const sc = sequential(0, 99);
    const g = svg.append('g').attr('transform', `translate(${off},${off})`);
    P.cm.forEach((rowv, a) => rowv.forEach((v, b) => g.append('rect').attr('x', b * cs).attr('y', a * cs).attr('width', cs).attr('height', cs).attr('fill', sc(v))));
    g.append('rect').attr('width', m * cs).attr('height', m * cs).attr('fill', 'transparent')
      .on('mousemove', (e) => {
        const [mx, my] = d3.pointer(e); const a = Math.min(m - 1, Math.floor(my / cs)), b = Math.min(m - 1, Math.floor(mx / cs));
        const ia = P.cm_idx[a], ib = P.cm_idx[b];
        tip(e, `<b>${S.ids[ia]}</b> (${S.pheno[ia]}) × <b>${S.ids[ib]}</b> (${S.pheno[ib]})<br>together in ${P.cm[a][b]}% of subsamples`);
      }).on('mouseleave', untip);
    P.cm_idx.forEach((i, a) => {
      svg.append('rect').attr('x', 2).attr('y', off + a * cs).attr('width', 8).attr('height', cs).attr('fill', cat(pmap[P.cl[i]]));
      svg.append('rect').attr('x', 11).attr('y', off + a * cs).attr('width', 8).attr('height', cs).attr('fill', phenoColor(S.pheno[i]));
      svg.append('rect').attr('x', off + a * cs).attr('y', 2).attr('width', cs).attr('height', 8).attr('fill', cat(pmap[P.cl[i]]));
      svg.append('rect').attr('x', off + a * cs).attr('y', 11).attr('width', cs).attr('height', 8).attr('fill', phenoColor(S.pheno[i]));
    });
    svg.append('text').attr('x', off).attr('y', off + m * cs + 16).attr('font-size', 11).attr('fill', ink2())
      .text(`consensus at k = ${k} (corrected data; strips: cluster, phenotype)`);
    rd.set({ silhouette: f3(R.sil), 'PAC (lower = stabler)': f3(R.pac), 'ARI with phenotype': f3(R.ari_pheno), 'ARI with centre': f3(R.ari_batch) });
  }
  draw();
  legend(fig.root, [['consensus 0%', sequential(0, 99)(0)], ['consensus 100%', sequential(0, 99)(99)]]);

  // ------------------------------------------------------------------ figure 2: metrics across k
  const fig2 = figure(main, 'Choosing k: internal scores versus external truth',
    'Left: scores computed from the data alone (what you have in a real study). Right: agreement with labels the clustering never saw (adjusted Rand index, ARI: 1 = identical partition, 0 = chance). Solid = corrected data, dashed = uncorrected. Hover the points.');
  const row2 = el('div', { class: 'fig-row' }); fig2.body.append(row2);
  const A = el('div'), Bw = el('div'); row2.append(A, Bw);
  const panel = (host, keys, title) => {
    const F = frame(host, { width: 440, height: 280, x: [KS[0] - 0.2, KS[KS.length - 1] + 0.2], y: [0, 1], xLabel: 'k', yLabel: title, xTicks: KS.length });
    F.g.select('.axis.x').call(d3.axisBottom(F.x).tickValues(KS).tickFormat(d3.format('d')));
    keys.forEach(([key, lab, slot]) => ['protected', 'raw'].forEach((v) => {
      const pts = KS.map((kk) => ({ x: kk, y: S.data[v].k[kk][key] }));
      F.layer('l').append('path').datum(pts).attr('fill', 'none').attr('stroke', cat(slot)).attr('stroke-width', 2).attr('stroke-dasharray', v === 'raw' ? '5 4' : null)
        .attr('d', d3.line().x((d) => F.x(d.x)).y((d) => F.y(d.y)));
      F.layer('p').selectAll(null).data(pts).join('circle').attr('cx', (d) => F.x(d.x)).attr('cy', (d) => F.y(d.y)).attr('r', 4.5).attr('fill', cat(slot)).attr('stroke', css('--surface')).attr('stroke-width', 1.2)
        .on('mousemove', (e, d) => tip(e, `<b>${lab}</b>, ${VNAME[v]}<br>k = ${d.x}: ${f3(d.y)}`)).on('mouseleave', untip);
    }));
    const kl = F.layer('k').append('line').attr('y1', 0).attr('y2', F.h).attr('stroke', ink()).attr('stroke-width', 1).attr('stroke-dasharray', '2 3');
    return (kk) => kl.attr('x1', F.x(kk)).attr('x2', F.x(kk));
  };
  const mk1 = panel(A, [['sil', 'silhouette', 0], ['pac', 'PAC', 3]], 'internal score');
  const mk2 = panel(Bw, [['ari_pheno', 'ARI with phenotype', 2], ['ari_batch', 'ARI with centre', 6]], 'ARI');
  legend(fig2.root, [['silhouette', cat(0)], ['PAC', cat(3)], ['ARI with phenotype', cat(2)], ['ARI with centre', cat(6)]]);
  const syncK = () => { mk1(k); mk2(k); };
  syncK();
  kSl.input.addEventListener('input', syncK);

  // numbers for prose
  const best = (v, key, dir = 1) => KS.reduce((a, b) => (dir * S.data[v].k[b][key] > dir * S.data[v].k[a][key] ? b : a));
  const P3 = S.data.protected.k[3], R2 = S.data.raw.k[2];
  const bestRawSil = best('raw', 'sil'), bestProtSil = best('protected', 'sil');
  section(main, 'What to notice', `<ul>
    <li>On the <b>corrected</b> data, silhouette peaks at k = ${bestProtSil} (${f3(S.data.protected.k[bestProtSil].sil)}), PAC is ${f3(P3.pac)} (every pair is always or never together), and the three clusters match the pathologist's phenotypes with ARI ${f3(P3.ari_pheno)}. The consensus matrix at k = 3 is three clean blocks.</li>
    <li>Switch to <b>uncorrected</b> data: silhouette now peaks at k = ${bestRawSil} (${f3(S.data.raw.k[bestRawSil].sil)}, higher than anything on the corrected data!) and those two clusters agree with the <b>centres</b> perfectly (ARI ${f3(R2.ari_batch)}) and with phenotype hardly at all (ARI ${f3(R2.ari_pheno)}). A beautifully separated clustering can be a batch effect.</li>
    <li>Push k to 5 or 6 on corrected data: PAC rises to ${f3(S.data.protected.k[5].pac)} and ${f3(S.data.protected.k[6].pac)}, and the consensus blocks turn grey and blurry. Patients are being split differently from one subsample to the next: the extra "subtypes" are not stable.</li>
    <li>Colour by <b>response</b> at k = 3: the inflamed cluster holds most responders, but not all, and has many non-responders. Subtypes describe tumours; they are not a response predictor.</li>
  </ul>`);
  steps(main, 'How it works', [
    'Reduce to the top principal components (here 10) so distances reflect the main biological axes rather than thousands of noisy genes.',
    'Repeat 100 times: draw 80% of patients at random and run k-means with the chosen k.',
    'For every pair of patients, record the fraction of the subsamples (that contained both) in which they landed in the same cluster. That is the <b>consensus matrix</b>.',
    'Cluster the consensus matrix itself (average-linkage hierarchical clustering on 1 − consensus) to get the final k groups.',
    '<b>PAC</b> (proportion of ambiguous clustering) is the share of pairs with consensus between 0.1 and 0.9. Near 0 = stable. <b>Silhouette</b> compares each patient\'s distance to its own cluster with the nearest other cluster (from −1 to 1).',
    'Then check against what you know: ARI with phenotype (good) and with centre (bad), and survival or response differences between clusters, tested with ordinary statistics.',
  ]);
  math(main, 'consensus(i, j) = #(subsamples with i, j in the same cluster) / #(subsamples containing both)\nPAC = fraction of pairs with 0.1 < consensus < 0.9         silhouette(i) = (b_i − a_i) / max(a_i, b_i)');

  callout(main, '', 'In biology', 'This is how the TCGA molecular subtypes were defined: <b>ConsensusClusterPlus</b> (R/Bioconductor) on mRNA, often combined across data types with <b>iCluster</b> or <b>MOFA</b>. Immune subtypes of solid tumours (Thorsson et al., <i>Immunity</i> 2018) and the CMS colorectal subtypes followed the same recipe, with networks across several clustering methods to agree on robust groups. In single-cell work the same question becomes graph clustering, see <a href="#/archive/omics/sc-clustering">Leiden & UMAP</a>.');
  callout(main, 'warn', 'Pitfalls', `<ul>
    <li><b>Clusters of batch.</b> Always cross-tabulate clusters against centre, sequencing run and purity before naming a subtype.</li>
    <li><b>Circularity.</b> The correction here protected the pathologist's phenotype, so the near-perfect agreement on corrected data is partly helped by that choice (see <a href="#/archive/omics/bulk-batch">batch effects</a>). The honest evidence is that k = 3 is also the most stable and best-separated solution.</li>
    <li><b>"Significant" subtypes by construction.</b> Clustering on genes, then running DE between the clusters on the same genes, gives tiny p-values for any partition. The test is circular.</li>
    <li><b>Consensus can look stable for the wrong k</b> when groups are very unequal in size; look at several scores and at the biology, not one magic number.</li>
  </ul>`);
  seeAlso(main, [['omics', 'bulk-batch', 'Previous: batch effects'], ['omics', 'bulk-de', 'Next: differential expression'], ['concepts', 'kmeans', 'k-means'], ['concepts', 'hierarchical', 'Hierarchical clustering'], ['omics', 'sp-domains', 'Spatial domains']]);
}
