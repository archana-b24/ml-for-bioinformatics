// Tab 2 · Single-cell step 6: diffusion pseudotime along the CD8 T-cell naive -> effector -> exhausted continuum.
import { header, figure, select, stats, section, steps, callout, math, legend, seeAlso, el, tip, untip } from '../../lib/ui.js';
import { frame, cat, css, muted, sequential } from '../../lib/plot.js';
import { stepCard, omicsData } from '../../omics/_shared.js';
import { canvasScatter, ext, f2, pct } from '../../omics/_sc_common.js';
const d3 = window.d3;

const SETTINGS = [
  ['good', 'Batch-centred, root = most naive-like cell'],
  ['bad', 'Batch-centred, root = most effector-like cell (wrong root)'],
  ['nobatch', 'No batch correction, root = most naive-like cell'],
];
const TREND_GENES = [['TCF7', 0], ['IL7R', 0], ['GZMB', 3], ['PRF1', 3], ['PDCD1', 7], ['HAVCR2', 7], ['TOX', 6]];

export default async function render(main) {
  header(main, 'Trajectories & pseudotime',
    'Some cell states are not separate types but points on a <b>continuum</b>. Pseudotime orders cells along it, here CD8 T cells from naive/memory-like to effector to exhausted, so you can ask which genes switch on along the way. The ordering is inferred, not observed: <b>you choose the start</b>.');
  stepCard(main, {
    stage: 'Single-cell · step 6 of 7',
    question: 'Where along the naive → effector → exhausted path do responders\' and non-responders\' CD8 T cells sit, and which genes mark the transition?',
    data: 'The cells labelled CD8 T in step 5, re-embedded on their own highly variable genes.',
    method: 'Diffusion map (a random walk on the kNN graph) → diffusion pseudotime (DPT): distance from a chosen root cell in diffusion space.',
    verdict: 'both',
    why: 'The ordering is unsupervised ML (manifold learning on a graph). Claims about genes changing along pseudotime, or groups differing, are statistics, and the unit of replication is still the patient.',
  });
  const [T, Cl] = await Promise.all([omicsData(main, 'sc_trajectory.json'), omicsData(main, 'sc_cells.json')]);
  if (!T) return;
  const n = T.n;
  const coords = { good: [T.dc1, T.dc2], bad: [T.dc1, T.dc2], nobatch: [T.dc1_raw, T.dc2_raw] };
  const ptOf = { good: T.pt, bad: T.pt_bad, nobatch: T.pt_nobatch };
  const rhoOf = { good: T.rho_good, bad: T.rho_bad, nobatch: T.rho_nobatch };
  const rootOf = { good: T.root, bad: T.root_bad, nobatch: T.root };
  const ptSc = sequential(0, 1);
  const trueIdx = d3.range(n).filter((i) => T.true_cd8[i] === 1);
  const batches = Cl ? Cl.batches : ["3' v2", "3' v3"];

  // ---------------- figure 1 ----------------
  const fig = figure(main, 'Diffusion map of CD8 T cells, and how good the ordering is',
    `Left: ${n} CD8 T cells on the first two diffusion components. The ring marks the root cell. Right: inferred pseudotime vs the simulated "true" position on the naive → exhausted path (known only because this is a simulation), for the ${trueIdx.length} cells that really are CD8 T.`);
  let set = 'good', colourBy = 'pt';
  select(fig.controls, { label: 'Setting', value: set, options: SETTINGS, onChange: (v) => { set = v; paint(); } });
  select(fig.controls, { label: 'Colour by', value: colourBy, options: [['pt', 'inferred pseudotime'], ['true', 'simulated true position'], ['batch', 'chemistry'], ...T.genes.map((g) => [g, `${g} expression`])], onChange: (v) => { colourBy = v; paint(); } });
  const rd = stats(fig.root);
  const row = el('div', { class: 'fig-row' }); fig.body.append(row);
  const pA = el('div'), pB = el('div'); row.append(pA, pB);
  const hov = (i) => `pseudotime ${ptOf[set][i].toFixed(2)}${T.true_cd8[i] ? ` · true position ${T.true_t[i].toFixed(2)}` : ' · not a true CD8 T cell'}<br>${T.patients[T.patient[i]]} (${T.resp[i] ? 'R' : 'NR'}) · ${batches[T.batch[i]]}<br>TCF7 ${T.expr.TCF7[i]} · GZMB ${T.expr.GZMB[i]} · PDCD1 ${T.expr.PDCD1[i]} · HAVCR2 ${T.expr.HAVCR2[i]}`;
  const A = canvasScatter(pA, { width: 460, height: 400, x: [0, 1], y: [0, 1], xLabel: 'diffusion component 1', yLabel: 'diffusion component 2', hover: hov });
  const legA = el('div'); pA.append(legA);
  const ring = A.F.layer('root').append('circle').attr('r', 9).attr('fill', 'none').attr('stroke', css('--ink')).attr('stroke-width', 2.5);
  const B = canvasScatter(pB, { width: 440, height: 400, x: [-0.02, 1.02], y: [-0.02, 1.02], xLabel: 'simulated true position (0 = naive, 1 = exhausted)', yLabel: 'inferred pseudotime', hover: hov });
  function paint() {
    const [x, y] = coords[set], pt = ptOf[set];
    A.F.x.domain(ext(x)); A.F.y.domain(ext(y));
    A.F.g.select('.axis.x').call(d3.axisBottom(A.F.x).ticks(4)); A.F.g.select('.axis.y').call(d3.axisLeft(A.F.y).ticks(4));
    let col, leg;
    if (colourBy === 'pt') { col = (i) => ptSc(pt[i]); leg = [['pseudotime 0', ptSc(0)], ['pseudotime 1', ptSc(1)]]; }
    else if (colourBy === 'true') { col = (i) => (T.true_cd8[i] ? ptSc(T.true_t[i]) : muted()); leg = [['naive (0)', ptSc(0)], ['exhausted (1)', ptSc(1)], ['not a true CD8 T cell', muted()]]; }
    else if (colourBy === 'batch') { col = (i) => cat(T.batch[i]); leg = batches.map((b, k) => [b, cat(k)]); }
    else { const v = T.expr[colourBy], mx = d3.quantile(v.filter((a) => a > 0), 0.95) || 1, sc = sequential(0, mx); col = (i) => sc(v[i]); leg = [[`${colourBy} not detected`, sc(0)], [`${colourBy} high`, sc(mx)]]; }
    const ord = d3.range(n);
    A.draw({ idx: ord, x, y, color: col, r: 2.8, alpha: 0.9 });
    legA.innerHTML = ''; legend(legA, leg);
    const r = rootOf[set];
    ring.attr('cx', A.F.x(x[r])).attr('cy', A.F.y(y[r]));
    B.draw({ idx: trueIdx, x: T.true_t, y: pt, color: (i) => cat(T.batch[i]), r: 2.6, alpha: 0.7 });
    rd.set({ 'Spearman ρ (pseudotime vs truth)': f2(rhoOf[set]), 'root cell TCF7 / GZMB': `${T.expr.TCF7[r]} / ${T.expr.GZMB[r]}`, cells: n });
  }
  legend(pB, batches.map((b, k) => [b, cat(k)]));
  paint();

  // ---------------- figure 2: gene trends ----------------
  const fig2 = figure(main, 'Genes along pseudotime',
    'Mean log-expression in 20 equal-sized pseudotime bins, each gene scaled to its own maximum. Uses the batch-centred, naive-root pseudotime. Hover a line.');
  const tr = T.trend_good;
  const G = frame(fig2.body, { width: 760, height: 300, x: [0, 1], y: [0, 1.05], xLabel: 'pseudotime (bin centre)', yLabel: 'expression / gene max' });
  TREND_GENES.forEach(([g, c], k) => {
    const ys = tr.y[g], mx = d3.max(ys) || 1;
    const pts = tr.x.map((x, b) => [x, ys[b] / mx]);
    const col = d3.color(cat(c)).darker(k % 2 ? 0.8 : 0).formatHex();
    G.layer('l').append('path').datum(pts).attr('fill', 'none').attr('stroke', col).attr('stroke-width', 2).attr('stroke-dasharray', k % 2 ? '6 3' : null)
      .attr('d', d3.line().curve(d3.curveMonotoneX).x((d) => G.x(d[0])).y((d) => G.y(d[1])))
      .on('mousemove', (e) => tip(e, `<b>${g}</b> · Spearman ρ with pseudotime ${f2(T.gene_rho[g])}`)).on('mouseleave', untip);
  });
  legend(fig2.body, TREND_GENES.map(([g, c], k) => [g, d3.color(cat(c)).darker(k % 2 ? 0.8 : 0).formatHex()]));

  // ---------------- figure 3: responders vs non-responders ----------------
  const fig3 = figure(main, 'Who has more exhausted CD8 T cells? Count patients, not cells',
    'Left: pseudotime distribution of all CD8 T cells, by response group (each group scaled to its total). Right: share of each patient\'s CD8 T cells with pseudotime > 0.6, one dot per patient (patients with ≥ 10 CD8 T cells).');
  const row3 = el('div', { class: 'fig-row' }); fig3.body.append(row3);
  const hA = el('div'), hB = el('div'); row3.append(hA, hB);
  const H = frame(hA, { width: 440, height: 280, x: [0, 1], y: [0, 1], xLabel: 'pseudotime', yLabel: 'share of cells' });
  const hR = T.pt_by_resp.R, hN = T.pt_by_resp.NR, sR = d3.sum(hR), sN = d3.sum(hN);
  const hd = [[hR.map((v) => v / sR), cat(2), 'responders'], [hN.map((v) => v / sN), cat(7), 'non-responders']];
  H.y.domain([0, d3.max(hd.flatMap((d) => d[0])) * 1.1]); H.g.select('.axis.y').call(d3.axisLeft(H.y).ticks(4, '.0%'));
  hd.forEach(([v, c, l], gi) => H.layer('b' + gi).selectAll('rect').data(v).join('rect')
    .attr('x', (d, b) => H.x(b / 20) + gi * H.w / 40).attr('width', H.w / 40 - 0.5).attr('y', (d) => H.y(d)).attr('height', (d) => H.h - H.y(d)).attr('fill', c)
    .on('mousemove', (e, d) => tip(e, `${l}: ${pct(d, 1)} of cells`)).on('mouseleave', untip));
  legend(hA, [[`responders (${T.n_by_resp.R} cells)`, cat(2)], [`non-responders (${T.n_by_resp.NR} cells)`, cat(7)]]);
  const per = T.patients.map((p, k) => {
    const ids = d3.range(n).filter((i) => T.patient[i] === k);
    return { p, k, n: ids.length, f: ids.length ? ids.filter((i) => T.pt[i] > 0.6).length / ids.length : 0, r: ids.length ? T.resp[ids[0]] : null, ph: T.phen[k] };
  }).filter((d) => d.n >= 10);
  const P = frame(hB, { width: 380, height: 280, x: [-0.5, 1.5], y: [0, Math.max(0.1, d3.max(per, (d) => d.f) * 1.15)], xLabel: '', yLabel: 'share of CD8 T cells with pseudotime > 0.6', xTicks: 2 });
  P.g.select('.axis.x').call(d3.axisBottom(P.x).tickValues([0, 1]).tickFormat((d) => (d === 1 ? 'responders' : 'non-responders')));
  P.g.select('.axis.y').call(d3.axisLeft(P.y).ticks(5, '.0%'));
  const jit = (k) => ((k * 37) % 11 - 5) * 0.035;
  P.layer('d').selectAll('circle').data(per).join('circle').attr('cx', (d) => P.x(d.r + jit(d.k))).attr('cy', (d) => P.y(d.f)).attr('r', 6)
    .attr('fill', (d) => (d.r ? cat(2) : cat(7))).attr('stroke', css('--surface')).attr('stroke-width', 1.5)
    .on('mousemove', (e, d) => tip(e, `<b>${d.p}</b> (${d.r ? 'R' : 'NR'}, ${d.ph}): ${pct(d.f)} of ${d.n} CD8 T cells`)).on('mouseleave', untip);
  const mR = d3.median(per.filter((d) => d.r === 1), (d) => d.f), mN = d3.median(per.filter((d) => d.r === 0), (d) => d.f);
  [[1, mR], [0, mN]].forEach(([g, m]) => P.layer('m').append('line').attr('x1', P.x(g - 0.3)).attr('x2', P.x(g + 0.3)).attr('y1', P.y(m)).attr('y2', P.y(m)).attr('stroke', css('--ink')).attr('stroke-width', 2));
  const nPR = per.filter((d) => d.r === 1).length, nPN = per.filter((d) => d.r === 0).length;

  section(main, 'What to notice', `<ul>
    <li>With batch-centring and a naive-like root, pseudotime tracks the simulated truth with Spearman ρ = <b>${f2(T.rho_good)}</b>. Colour by TCF7, GZMB, then HAVCR2: memory genes fade, effector genes peak mid-way, exhaustion genes rise at the end.</li>
    <li>Pick the <b>wrong root</b> (the most effector-like cell): the same diffusion map now gives ρ = ${f2(T.rho_bad)}. Pseudotime is distance from wherever you start; the algorithm cannot tell direction. Biology (here, naive markers) has to choose the root.</li>
    <li>Pick <b>no batch correction</b>: the first diffusion component is the chemistry (|ρ| with batch ${f2(T.dc1_batch)}; colour by chemistry), and pseudotime is mostly a batch axis (ρ with truth ${f2(T.rho_nobatch)}). Sub-setting cells and re-embedding them brings the batch effect back, so it has to be corrected again.</li>
    <li>A straight line is not much worse here: |ρ| between PC1 and the truth is ${f2(T.rho_pc1)}. On a single, unbranched path, diffusion pseudotime mainly buys robustness to curvature and noise; its real advantage shows on curved or branching trajectories.</li>
    <li>Responders' CD8 T cells sit <i>further</i> along the path: ${pct(T.frac_exh.R)} of responder vs ${pct(T.frac_exh.NR)} of non-responder cells are beyond 0.6. That is not a paradox. Responders here are mostly inflamed tumours where T cells meet their antigen continuously, and "exhausted" (PD-1⁺, TIM-3⁺, CXCL13⁺) marks that ongoing recognition. Real studies of checkpoint therapy report the same: tumour-reactive CD8 T cells look exhausted, and the TCF7⁺ progenitors that feed them matter for response.</li>
    <li>The cell-level histogram has ${T.n_by_resp.R} vs ${T.n_by_resp.NR} cells, but the right panel shows what you can actually test: ${nPR} responders vs ${nPN} non-responders (patients with ≥ 10 CD8 T cells), medians ${pct(mR)} vs ${pct(mN)}, with overlap. Patients are the replicates; that is the subject of the <a href="#/archive/omics/sc-pseudobulk">next page</a>.</li>
  </ul>`);
  steps(main, 'How it works', [
    'Subset the CD8 T cells, choose their own HVGs, centre each chemistry, scale, and run PCA (10 PCs).',
    'Build a kNN graph (k = 30) with Gaussian edge weights whose width adapts to local density.',
    'Normalise the graph into a random-walk transition matrix. Its leading eigenvectors are the <b>diffusion components</b>: coordinates in which distance reflects how easily a random walk travels between cells, summing over all paths, which is robust to noise.',
    'Choose a <b>root</b> cell from biology (highest naive-marker score).',
    '<b>Diffusion pseudotime</b> = distance from the root in diffusion space, rescaled to 0–1.',
    'Relate genes to pseudotime (smoothed trends, or a regression per gene), and compare groups at the patient level.',
  ]);
  math(main, `W_ij = exp(−‖x_i − x_j‖² / (σ_i σ_j))  on kNN pairs;   T = D⁻¹ W   (random walk)
DC_k = ψ_k · λ_k / (1 − λ_k)      (eigenvectors of T, weighted to sum over all walk lengths)
DPT(i) = ‖DC(i) − DC(root)‖`);
  callout(main, '', 'In biology', `<b>Scanpy</b>: <code>sc.tl.diffmap</code> + <code>sc.tl.dpt</code> (this page), <b>PAGA</b> (<code>sc.tl.paga</code>) for a coarse graph of which clusters connect. <b>Monocle 3</b> (principal graphs), <b>Slingshot</b> (curves through clusters), <b>Palantir</b> and <b>CellRank</b> (fate probabilities, optionally using <b>RNA velocity</b> from scVelo to set direction). Saelens et al. (2019) benchmarked 45 trajectory methods and found that no single method is best for every topology. In tumour T cells, exhaustion trajectories (TCF7⁺ progenitor → PD-1⁺TIM-3⁺ terminally exhausted) are a major focus of checkpoint-response studies.`);
  callout(main, 'warn', 'Pitfalls', `<ul>
    <li><b>Pseudotime is not time</b>: it orders states, it does not measure hours, and a snapshot cannot prove cells actually move along the path.</li>
    <li><b>Every algorithm returns a trajectory</b>, even for discrete clusters. Check that a continuum exists before ordering it.</li>
    <li><b>Root and topology are assumptions</b>: wrong root or forcing a linear path onto a branching process gives confident nonsense.</li>
    <li><b>Batch and patient effects</b> can masquerade as a trajectory, as in the "no batch correction" setting.</li>
    <li><b>Testing genes vs pseudotime with cells as replicates</b> gives tiny p-values; use patient-aware models (e.g. tradeSeq with care, or pseudobulk along pseudotime bins).</li>
  </ul>`);
  seeAlso(main, [['omics', 'sc-pseudobulk', 'Next: pseudobulk DE'], ['concepts', 'graph-clustering', 'Graph methods'], ['concepts', 'pca', 'PCA'], ['concepts', 'tsne-umap', 't-SNE & UMAP'], ['omics', 'sc-integration', 'Batch integration']]);
}
