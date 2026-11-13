// Spatial step 3: spatial domains = clustering on [expression PCs, neighbour-averaged PCs] with weight lambda.
// Precomputed k-means for 6 lambdas in scripts/build_sc_spatial.py -> data/omics/sp_domains.json (+ sp_data.json for positions).
import { header, figure, select, slider, button, stats, section, steps, callout, math, legend, seeAlso, el, tip, untip, every } from '../lib/ui.js';
import { frame, css, cat, ink2, sequential } from '../lib/plot.js';
import { stepCard, omicsData } from './_shared.js';
import { tissueMap, regionColor, ctColor, secShort, pct } from './_spatial.js';
const d3 = window.d3;

export default async function render(main) {
  header(main, 'Spatial domains',
    'A tissue region (tumour nest, invasive margin, stroma, lymphoid aggregate) is not one cell type: it is a <b>neighbourhood with a typical mix</b>. Clustering cells on their own expression finds cell types; clustering on expression <b>plus the average of their neighbours</b> finds regions.');
  const card = stepCard(main, {
    stage: 'Spatial · step 3 of 6',
    question: 'Can we find the tumour nests, the margin around them and the stroma from the data alone, without a pathologist drawing them?',
    data: '…', method: 'PCA on the panel, then k-means on [√(1−λ)·own PCs, √λ·neighbour-averaged PCs] for λ from 0 to 0.95. Scored against the simulated regions with the adjusted Rand index (ARI).',
    verdict: 'ml', why: ' Domain finding is unsupervised clustering. The spatial part is a smoothing prior: neighbours probably share a domain.',
  });
  const D = await omicsData(main, 'sp_data.json');
  const M = await omicsData(main, 'sp_domains.json');
  if (!D || !M) return;
  const L = M.lambdas, NS = D.sections.length, R = M.regions, K = M.k;
  card.querySelectorAll('.step-row')[1].querySelector('span').innerHTML = `All ${NS} sections pooled (${d3.sum(D.cells, (c) => c.n).toLocaleString()} cells, ${D.n_panel} genes, 20 PCs). Each cell's neighbour average is a Gaussian-weighted mean (σ = 30 µm) over its 15 nearest cells. k = ${K} clusters, matched to the true regions for colouring.`;
  const KEYS = Object.keys(M.labels), lk = (i) => KEYS.find((k) => +k === L[i]);
  const pureN = (i) => M.composition[lk(i)].filter((row) => d3.max(row) > 0.9).length;
  const tlsRec = (i) => { let a = 0, b = 0; D.cells.forEach((c, si) => c.region.forEach((r, j) => { if (r === 3) { b++; if (M.labels[lk(i)][si][j] === 3) a++; } })); return b ? a / b : 0; };
  const best = d3.maxIndex(M.ari);

  // ------------------------------------------------------------------ figure 1: domains vs truth
  const fig1 = figure(main, 'Turn up the neighbour weight λ',
    'Left: k-means clusters at the chosen λ. Right: the simulated regions (truth). Clusters are coloured by the region they overlap most. Hover a cell to compare its cluster with the truth.');
  let li = 0, sec = 0, stopPlay = null;
  const sl = slider(fig1.controls, { label: 'Neighbour weight λ', min: 0, max: L.length - 1, step: 1, value: 0, fmt: (v) => L[v].toFixed(2), onInput: (v) => { li = v; paint(); } });
  select(fig1.controls, { label: 'Section', value: '0', options: D.sections.map((s, i) => [String(i), s]), onChange: (v) => { sec = +v; paint(); } });
  const playB = button(fig1.controls, 'Play sweep', () => {
    if (stopPlay) { stopPlay(); stopPlay = null; playB.textContent = 'Play sweep'; return; }
    playB.textContent = 'Stop'; li = 0; sl.set(0); paint();
    stopPlay = every(900, () => { li = (li + 1) % L.length; sl.set(li); paint(); if (li === L.length - 1) { stopPlay(); stopPlay = null; playB.textContent = 'Play sweep'; } });
  });
  legend(fig1.controls, R.map((r, k) => [r, regionColor(k)]));
  const rd1 = stats(fig1.root);
  const row1 = el('div', { class: 'fig-row' }); fig1.body.append(row1);
  const aW = el('div'), bW = el('div'); row1.append(aW, bW);
  const MA = tissueMap(aW, { size: D.size, px: 440, label: 'Clusters (data only)' });
  const MB = tissueMap(bW, { size: D.size, px: 440, label: 'Simulated regions (truth)' });

  // ------------------------------------------------------------------ figure 2: ARI curve + composition
  const fig2 = figure(main, 'How good are the domains, and what are they made of?',
    'Left: ARI of the clusters against the true regions and against the cell-type labels, for each λ (pooled over sections). Right: cell-type make-up of each cluster at the current λ. Hover for values.');
  const row2 = el('div', { class: 'fig-row' }); fig2.body.append(row2);
  const cW = el('div'), hW = el('div'); row2.append(cW, hW);
  const F = frame(cW, { width: 460, height: 320, x: [0, 1], y: [0, 1], xLabel: 'neighbour weight λ', yLabel: 'adjusted Rand index' });
  legend(cW, [['vs true regions', cat(0)], ['vs cell types', cat(1)]]);
  [['ari', 0], ['ari_type', 1]].forEach(([key, c]) => {
    const pts = L.map((l, i) => ({ x: l, y: M[key][i], i }));
    F.layer('l' + key).append('path').datum(pts).attr('fill', 'none').attr('stroke', cat(c)).attr('stroke-width', 2).attr('d', d3.line().x((d) => F.x(d.x)).y((d) => F.y(d.y)));
    F.layer('p' + key).selectAll('circle').data(pts).join('circle').attr('cx', (d) => F.x(d.x)).attr('cy', (d) => F.y(d.y)).attr('r', 4.5).attr('fill', cat(c)).attr('stroke', css('--surface')).attr('stroke-width', 1.5)
      .on('mousemove', (e, d) => tip(e, `λ = ${d.x}<br>ARI ${key === 'ari' ? 'vs regions' : 'vs cell types'}: ${d.y.toFixed(3)}`)).on('mouseleave', untip);
  });
  const marker = F.layer('m').append('line').attr('y1', 0).attr('y2', F.h).attr('stroke', css('--ink')).attr('stroke-dasharray', '4 3');
  const hWrap = el('div'); hW.append(hWrap);

  function paint() {
    const s = D.cells[sec], lab = M.labels[lk(li)][sec];
    MA.draw(s.x, s.y, (i) => regionColor(lab[i]), () => 6);
    MB.draw(s.x, s.y, (i) => regionColor(s.region[i]), () => 6);
    const hov = (x, y, near) => { const i = near(x, y); return i < 0 ? null : `<b>${D.cell_types[s.type[i]]}</b><br>cluster: ${R[lab[i]]}-like<br>truth: ${R[s.region[i]]}`; };
    MA.hover(hov); MB.hover(hov);
    rd1.set({ λ: L[li].toFixed(2), 'ARI vs regions (all sections)': M.ari[li].toFixed(3), 'ARI vs cell types': M.ari_type[li].toFixed(3), [`ARI vs regions (${secShort(D.sections[sec])})`]: M.ari_section[lk(li)][sec].toFixed(3), 'clusters >90% one cell type': `${pureN(li)} of ${K}`, 'TLS cells in the TLS-like cluster': pct(tlsRec(li)) });
    marker.attr('x1', F.x(L[li])).attr('x2', F.x(L[li]));
    hWrap.innerHTML = '';
    const comp = M.composition[lk(li)];
    heat(hWrap, comp);
  }
  function heat(parent, comp) {
    const cell = 34, left = 120, top = 70, w = left + M.cell_types.length * cell + 50, h = top + K * cell + 6;
    const svg = d3.select(parent).append('svg').attr('viewBox', `0 0 ${w} ${h}`).attr('width', w);
    const sc = sequential(0, 1);
    comp.forEach((row, k) => row.forEach((v, t) => {
      svg.append('rect').attr('x', left + t * cell).attr('y', top + k * cell).attr('width', cell - 2).attr('height', cell - 2).attr('rx', 3).attr('fill', sc(v))
        .on('mousemove', (e) => tip(e, `cluster "${R[k]}": ${pct(v, 1)} ${M.cell_types[t]}`)).on('mouseleave', untip);
      if (v >= 0.1) svg.append('text').attr('x', left + t * cell + cell / 2 - 1).attr('y', top + k * cell + cell / 2 + 4).attr('text-anchor', 'middle').attr('font-size', 10.5).attr('fill', v > 0.55 ? css('--surface') : css('--ink')).attr('pointer-events', 'none').text(Math.round(v * 100));
    }));
    R.forEach((r, k) => {
      svg.append('rect').attr('x', 4).attr('y', top + k * cell + cell / 2 - 6).attr('width', 10).attr('height', 10).attr('rx', 2).attr('fill', regionColor(k));
      svg.append('text').attr('x', left - 6).attr('y', top + k * cell + cell / 2 + 4).attr('text-anchor', 'end').attr('font-size', 11.5).attr('fill', ink2()).text(r + '-like');
    });
    M.cell_types.forEach((c, t) => svg.append('text').attr('transform', `translate(${left + t * cell + cell / 2 + 3},${top - 6}) rotate(-50)`).attr('font-size', 11).attr('fill', ink2()).text(c));
    svg.append('text').attr('x', left).attr('y', 12).attr('font-size', 12).attr('fill', ink2()).text('% of each cluster\'s cells by type');
  }
  paint();

  // ------------------------------------------------------------------ prose (numbers from the JSON)
  const last = L.length - 1;
  const nm = (i) => M.composition[lk(i)].map((row, k) => d3.max(row) > 0.9 ? `"${R[k]}-like" = ${pct(d3.max(row))} ${M.cell_types[d3.maxIndex(row)]}` : null).filter(Boolean);
  const fibNest = (i) => { let a = 0, b = 0; D.cells.forEach((c, si) => c.type.forEach((t, j) => { if (t === 6 && c.region[j] === 0) { b++; if (M.labels[lk(i)][si][j] === 0) a++; } })); return b ? a / b : 0; };
  const tlsTop = M.composition[lk(last)][3];
  section(main, 'What to notice', `<ul>
    <li>At <b>λ = 0</b> (no neighbours) the clusters are cell types: ARI against the cell-type labels is ${M.ari_type[0].toFixed(3)}, against the regions only ${M.ari[0].toFixed(3)}. ${pureN(0)} of ${K} clusters are more than 90% one cell type, and ${pct(fibNest(0))} of the fibroblasts that sit inside tumour nests are put in the nest cluster.</li>
    <li>Push λ up: the region ARI peaks at <b>${M.ari[best].toFixed(3)} at λ = ${L[best]}</b>, the nests become contiguous and ${pct(fibNest(best))} of in-nest fibroblasts now join the nest cluster. But look at the heatmap: ${pureN(best)} of the ${K} clusters are still essentially one cell type (${nm(best).join('; ')}). The best ARI is a hybrid of "regions" and "cell types", and one number hides that.</li>
    <li>Low λ values change almost nothing (${L.slice(0, 3).map((l, i) => `λ = ${l}: ${M.ari[i].toFixed(3)}`).join(', ')}): a cell's own expression dominates until the neighbour term carries most of the weight.</li>
    <li>At <b>λ = ${L[last]}</b> the region ARI falls to ${M.ari[last].toFixed(3)}, yet this is the only setting that finds the <b>TLS</b>: ${pct(tlsRec(last))} of TLS cells land in a cluster that is ${pct(tlsTop[4])} B and ${pct(tlsTop[2])} CD4 T cells. It pays for that by splitting the tumour into two nest-like clusters. Different λ answer different questions; there is no single "right" smoothing.</li>
    <li>Here λ can be judged against the truth. In real data there is none, so λ and k are chosen by stability, by agreement with histology, or by eye. That is the honest weak point of every domain method.</li>
  </ul>`);

  steps(main, 'How it works', [
    'Normalise, scale and reduce the panel to principal components (PCs), as in <a href="#/omics/sc-embedding">single-cell embedding</a>.',
    'For each cell, compute the weighted average of its neighbours\' PCs (Gaussian weights on distance). This is a feature describing the cell\'s <b>surroundings</b>.',
    'Concatenate own and neighbour features with weights √(1−λ) and √λ, so the squared distance k-means uses is (1−λ)·own distance + λ·neighbourhood distance.',
    `Cluster (here k-means, k = ${K}). Real tools use Gaussian mixtures with a Potts / Markov random field prior (BayesSpace, HMRF), graph neural networks (SpaGCN, GraphST) or the same augmentation trick (BANKSY).`,
    'Name domains by their composition and markers, then check them against histology.',
  ]);
  math(main, 'feature_i = [ √(1−λ) · PC_i ,  √λ · Σ_j w_ij PC_j ]      w_ij ∝ exp(−d_ij² / 2σ²), j ∈ 15 nearest neighbours\n‖feature_i − feature_k‖² = (1−λ)‖PC_i − PC_k‖² + λ‖nbr_i − nbr_k‖²\nARI = 1 for perfect agreement with the truth, ≈ 0 for random labels');

  callout(main, '', 'In biology', `<ul>
    <li><b>BANKSY</b> augments each cell with neighbourhood-averaged expression and a λ parameter exactly like this figure; it works for both cell typing (small λ) and domain finding (large λ).</li>
    <li><b>BayesSpace</b> (Visium) and HMRF-based methods (Giotto) put a spatial prior on cluster labels; <b>SpaGCN</b>, <b>STAGATE</b> and <b>GraphST</b> learn embeddings with graph neural networks; <b>CellCharter</b> clusters aggregated neighbourhoods.</li>
    <li>In tumours, domains map to nest / invasive margin / stroma / tertiary lymphoid structures (TLS), and features like "fraction of CD8 T cells in the margin domain" become candidate biomarkers.</li>
  </ul>`);
  callout(main, 'warn', 'Pitfalls', `<ul>
    <li><b>λ and k are hyperparameters with no ground truth.</b> Tuning them until the picture "looks right" bakes your expectation into the result.</li>
    <li><b>Oversmoothing</b> erases thin structures such as the invasive margin and single vessels, and blurs boundaries: here λ = ${L[last]} drops the region ARI to ${M.ari[last].toFixed(3)}.</li>
    <li><b>Pooling sections</b> lets batch or patient differences masquerade as domains; integrate first or cluster per section and match.</li>
    <li>ARI against a pathologist's annotation is the usual benchmark, but annotations are themselves coarse and disagree between pathologists.</li>
  </ul>`);
  seeAlso(main, [['concepts', 'kmeans', 'k-means'], ['omics', 'sc-clustering', 'Single-cell clustering'], ['omics', 'sp-niches', 'Niches & communication'], ['concepts', 'gnn', 'Graph neural networks']]);
}
