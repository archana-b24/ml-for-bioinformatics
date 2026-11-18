// Curse of dimensionality: distance concentration, where the volume of a cube lives, and why scRNA-seq runs PCA before kNN.
import { header, figure, slider, stats, section, steps, callout, math, seeAlso, legend, tip, untip } from '../../lib/ui.js';
import { frame, cat, css, ink, ink2 } from '../../lib/plot.js';
import { rng } from '../../lib/data.js';
const d3 = window.d3;

const DIMS = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000];

/** log Γ(x) by the Lanczos approximation (g = 7). */
function lgamma(x) {
  const c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61503916999185,
    12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
  if (x < 0.5) return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * x))) - lgamma(1 - x);
  x -= 1; let a = c[0]; const t = x + 7.5;
  for (let i = 1; i < 9; i++) a += c[i] / (x + i);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}
/** Fraction of the cube [-1,1]^d that lies inside the inscribed unit ball. */
const ballFrac = (d) => Math.exp((d / 2) * Math.log(Math.PI) - lgamma(d / 2 + 1) - d * Math.log(2));
/** Fraction of the cube within distance eps (of half-width 1) of its surface. */
const shellFrac = (d, eps) => 1 - Math.pow(1 - eps, d);

export default async function render(main) {
  header(main, 'Curse of dimensionality',
    'Our intuition for "near" and "far" comes from two or three dimensions. With hundreds of features, random points become <b>almost equally far from each other</b>, and almost all of the space sits in the corners and near the edges. Nearest-neighbour methods then have little to go on.');

  // ---------------- Figure 1: distance concentration ----------------
  const N = 120, r = rng(21);
  const X = Array.from({ length: N }, () => Float64Array.from({ length: DIMS[DIMS.length - 1] }, () => r()));
  // cumulative squared distances over coordinates; snapshot at each d in DIMS
  const nP = N * (N - 1) / 2, acc = new Float64Array(nP), snaps = {};
  let di = 0;
  for (let c = 0; c < DIMS[DIMS.length - 1]; c++) {
    let p = 0;
    for (let i = 0; i < N; i++) { const xi = X[i][c]; for (let j = i + 1; j < N; j++) { const t = xi - X[j][c]; acc[p++] += t * t; } }
    if (c + 1 === DIMS[di]) { snaps[DIMS[di]] = Float64Array.from(acc, Math.sqrt); di++; }
  }
  const pairIndex = (i, j) => (i < j ? i * N - (i * (i + 1)) / 2 + (j - i - 1) : pairIndex(j, i));
  const summary = {};
  DIMS.forEach((d) => {
    const D = snaps[d];
    let rc = 0;
    for (let i = 0; i < N; i++) {
      let lo = Infinity, hi = 0;
      for (let j = 0; j < N; j++) if (j !== i) { const v = D[pairIndex(i, j)]; if (v < lo) lo = v; if (v > hi) hi = v; }
      rc += (hi - lo) / lo;
    }
    const m = d3.mean(D), s = d3.deviation(D);
    summary[d] = { contrast: rc / N, mean: m, cv: s / m, min: d3.min(D), max: d3.max(D) };
  });

  const f1 = figure(main, 'Pairwise distances concentrate',
    `${N} points drawn uniformly in the unit cube [0,1]<sup>d</sup>. Left: histogram of all ${nP.toLocaleString()} pairwise distances, divided by their mean so every d shares one axis. Right: for each point, (farthest − nearest) / nearest distance to the other points, averaged over points. The marked dot is the current d.`);
  let dIdx = 4;
  slider(f1.controls, { label: 'Dimensions d', min: 0, max: DIMS.length - 1, step: 1, value: dIdx, fmt: (v) => DIMS[v], onInput: (v) => { dIdx = v; paint1(); } });
  const st1 = stats(f1.root);
  const row1 = d3.select(f1.body).append('div').attr('class', 'fig-row').node();
  const l1 = document.createElement('div'), r1 = document.createElement('div');
  row1.append(l1, r1);
  const H = frame(l1, { width: 460, height: 360, x: [0, 2.2], y: [0, 0.5], xLabel: 'distance ÷ mean distance', yLabel: 'fraction of pairs' });
  const C = frame(r1, { width: 460, height: 360, x: [1, 1000], y: [0.01, 1000], xScale: d3.scaleLog(), yScale: d3.scaleLog(), xLabel: 'dimensions d (log scale)', yLabel: '(max − min) / min' });
  C.g.select('.axis.x').call(d3.axisBottom(C.x).ticks(4, '~g'));
  C.g.select('.axis.y').call(d3.axisLeft(C.y).ticks(5, '~g'));
  const cpts = DIMS.map((d) => ({ x: d, y: summary[d].contrast }));
  C.layer('line').append('path').datum(cpts).attr('fill', 'none').attr('stroke', cat(0)).attr('stroke-width', 2)
    .attr('d', d3.line().x((p) => C.x(p.x)).y((p) => C.y(p.y)));
  C.layer('pts').selectAll('circle').data(cpts).join('circle').attr('cx', (p) => C.x(p.x)).attr('cy', (p) => C.y(p.y)).attr('r', 4)
    .attr('fill', cat(0)).attr('stroke', css('--surface')).attr('stroke-width', 1.2)
    .on('mousemove', (e, p) => tip(e, `d = ${p.x}<br>relative contrast ${p.y.toFixed(3)}`)).on('mouseleave', untip);
  C.layer('cur');

  function paint1() {
    const d = DIMS[dIdx], D = snaps[d], s = summary[d];
    const bins = d3.bin().domain([0, 2.2]).thresholds(d3.range(0, 2.2, 0.05))(Array.from(D, (v) => Math.min(2.199, v / s.mean)));
    const ymax = Math.max(0.1, d3.max(bins, (b) => b.length / nP) * 1.1);
    H.y.domain([0, ymax]);
    H.g.select('.axis.y').call(d3.axisLeft(H.y).ticks(5));
    H.g.select('.gridline').call(d3.axisLeft(H.y).ticks(5).tickSize(-H.w).tickFormat(''));
    H.layer('bars').selectAll('rect').data(bins).join('rect')
      .attr('x', (b) => H.x(b.x0) + 0.5).attr('width', (b) => Math.max(0, H.x(b.x1) - H.x(b.x0) - 1))
      .attr('y', (b) => H.y(b.length / nP)).attr('height', (b) => H.h - H.y(b.length / nP)).attr('fill', cat(0))
      .on('mousemove', (e, b) => tip(e, `${b.x0.toFixed(2)}–${b.x1.toFixed(2)} × mean<br>${(100 * b.length / nP).toFixed(1)}% of pairs`)).on('mouseleave', untip);
    C.layer('cur').selectAll('circle').data([s]).join('circle').attr('cx', C.x(d)).attr('cy', C.y(s.contrast)).attr('r', 7.5)
      .attr('fill', cat(1)).attr('stroke', css('--surface')).attr('stroke-width', 2);
    st1.set({ d, 'mean distance': s.mean.toFixed(2), 'closest pair': s.min.toFixed(2), 'farthest pair': s.max.toFixed(2),
      'spread (SD ÷ mean)': s.cv.toFixed(3), '(max − min) / min, per point': s.contrast.toFixed(3) });
  }
  paint1();

  // ---------------- Figure 2: where the volume is ----------------
  const f2 = figure(main, 'Almost all of a high-dimensional cube is corners and skin',
    'Take the cube [−1, 1]<sup>d</sup>. The blue curve is the fraction of its volume inside the ball that touches every face (the "middle"); the rest is in the corners. The orange curve is the fraction within ε of the surface. The table checks both formulas by throwing 4,000 random points into the cube.');
  let d2 = 10, eps = 0.05;
  slider(f2.controls, { label: 'Dimensions d', min: 1, max: 200, step: 1, value: d2, onInput: (v) => { d2 = v; paint2(); } });
  slider(f2.controls, { label: 'Shell thickness ε', min: 0.01, max: 0.2, step: 0.01, value: eps, fmt: (v) => v.toFixed(2), onInput: (v) => { eps = v; paint2(); } });
  legend(f2.root, [['inside the inscribed ball', cat(0)], ['within ε of the surface', cat(1)]]);
  const st2 = stats(f2.root);
  const V = frame(f2.body, { width: 760, height: 320, x: [1, 200], y: [0, 1], xScale: d3.scaleLog(), xLabel: 'dimensions d (log scale)', yLabel: 'fraction of cube volume' });
  V.g.select('.axis.x').call(d3.axisBottom(V.x).tickValues([1, 2, 5, 10, 20, 50, 100, 200]).tickFormat(d3.format('~g')));
  const dd = d3.range(1, 201);
  V.layer('ball').append('path').datum(dd).attr('fill', 'none').attr('stroke', cat(0)).attr('stroke-width', 2)
    .attr('d', d3.line().x((d) => V.x(d)).y((d) => V.y(ballFrac(d))));
  V.layer('shell'); V.layer('marks'); V.layer('rule');
  function paint2() {
    V.layer('shell').selectAll('path').data([dd]).join('path').attr('fill', 'none').attr('stroke', cat(1)).attr('stroke-width', 2)
      .attr('d', d3.line().x((d) => V.x(d)).y((d) => V.y(shellFrac(d, eps))));
    V.layer('rule').selectAll('line').data([d2]).join('line').attr('x1', V.x(d2)).attr('x2', V.x(d2)).attr('y1', 0).attr('y2', V.h)
      .attr('stroke', css('--muted')).attr('stroke-dasharray', '3 3');
    const b = ballFrac(d2), sh = shellFrac(d2, eps);
    V.layer('marks').selectAll('circle').data([[b, 0], [sh, 1]]).join('circle').attr('cx', V.x(d2)).attr('cy', (p) => V.y(p[0])).attr('r', 6)
      .attr('fill', (p) => cat(p[1])).attr('stroke', css('--surface')).attr('stroke-width', 1.5)
      .on('mousemove', (e, p) => tip(e, `d = ${d2}<br>${p[1] ? 'outer shell' : 'inscribed ball'}: ${(100 * p[0]).toPrecision(3)}%`)).on('mouseleave', untip);
    // Monte Carlo check (seeded so the numbers are stable for a given d and ε)
    const M = 4000, r2 = rng(1000 * d2 + Math.round(eps * 100)); let inBall = 0, inShell = 0;
    for (let k = 0; k < M; k++) {
      let s2 = 0, mx = 0;
      for (let c = 0; c < d2; c++) { const u = r2() * 2 - 1; s2 += u * u; const a = Math.abs(u); if (a > mx) mx = a; }
      if (s2 <= 1) inBall++;
      if (mx >= 1 - eps) inShell++;
    }
    const pct = (v) => (v >= 0.001 ? (100 * v).toFixed(1) + '%' : v === 0 ? '0%' : (100 * v).toExponential(1) + '%');
    st2.set({ d: d2, 'in ball (formula)': pct(b), 'in ball (4,000 random points)': pct(inBall / M), 'in ε-shell (formula)': pct(sh), 'in ε-shell (random points)': pct(inShell / M) });
  }
  paint2();

  // ---------------- Figure 3: PCA before kNN ----------------
  const NC = 150, NSIG = 30, SHIFT = 1.0, NOISE = [0, 20, 100, 300, 1000, 2000], KNN = 10, NPC = 10;
  const r3 = rng(33), type = Array.from({ length: NC }, (_, i) => i % 2);
  const G = NSIG + NOISE[NOISE.length - 1];
  const E = Array.from({ length: NC }, (_, i) => Float64Array.from({ length: G }, (_, g) => r3.normal(g < NSIG && type[i] ? SHIFT : 0, 1)));
  // cumulative Gram matrix over genes; snapshot after the signal genes + each noise level
  const K = Array.from({ length: NC }, () => new Float64Array(NC)), res3 = {};
  let ni = 0;
  for (let g = 0; g < G; g++) {
    for (let i = 0; i < NC; i++) { const v = E[i][g]; if (v === 0) continue; const Ki = K[i]; for (let j = i; j < NC; j++) Ki[j] += v * E[j][g]; }
    if (g + 1 === NSIG + NOISE[ni]) { res3[NOISE[ni]] = analyse(K); ni++; }
  }
  function purity(dist2) {
    let tot = 0;
    for (let i = 0; i < NC; i++) {
      const order = d3.range(NC).filter((j) => j !== i).sort((a, b) => dist2(i, a) - dist2(i, b)).slice(0, KNN);
      tot += order.filter((j) => type[j] === type[i]).length / KNN;
    }
    return tot / NC;
  }
  function analyse(Kup) {
    const Kf = (i, j) => (i <= j ? Kup[i][j] : Kup[j][i]);
    const raw = purity((i, j) => Kf(i, i) + Kf(j, j) - 2 * Kf(i, j));
    // double-centred Gram -> PCA scores via orthogonal (subspace) iteration
    const rowM = d3.range(NC).map((i) => d3.mean(d3.range(NC), (j) => Kf(i, j))), allM = d3.mean(rowM);
    const Kc = d3.range(NC).map((i) => Float64Array.from({ length: NC }, (_, j) => Kf(i, j) - rowM[i] - rowM[j] + allM));
    const rq = rng(4);
    let Q = d3.range(NPC).map(() => Float64Array.from({ length: NC }, () => rq.normal()));
    let lam = new Array(NPC).fill(0);
    for (let it = 0; it < 60; it++) {
      const Z = Q.map((q) => Float64Array.from({ length: NC }, (_, i) => { let s = 0; const row = Kc[i]; for (let j = 0; j < NC; j++) s += row[j] * q[j]; return s; }));
      for (let a = 0; a < NPC; a++) { // Gram–Schmidt
        for (let b = 0; b < a; b++) { let dp = 0; for (let i = 0; i < NC; i++) dp += Z[a][i] * Z[b][i]; for (let i = 0; i < NC; i++) Z[a][i] -= dp * Z[b][i]; }
        let nrm = 0; for (let i = 0; i < NC; i++) nrm += Z[a][i] ** 2; nrm = Math.sqrt(nrm);
        lam[a] = nrm; for (let i = 0; i < NC; i++) Z[a][i] /= nrm;
      }
      Q = Z;
    }
    const scores = d3.range(NC).map((i) => Q.map((q, a) => q[i] * Math.sqrt(lam[a])));
    const pc = purity((i, j) => { let s = 0; for (let a = 0; a < NPC; a++) s += (scores[i][a] - scores[j][a]) ** 2; return s; });
    return { raw, pc, scores, varShare: lam[0] / Kc.reduce((s, row, i) => s + row[i], 0) };
  }

  const f3 = figure(main, 'Why single-cell pipelines run PCA before building the kNN graph',
    `A toy: ${NC} cells of two types that differ by ${SHIFT} SD in ${NSIG} marker genes, plus a growing number of pure-noise genes (every gene has SD 1). Left: the cells on PC1 vs PC2. Right: kNN purity, the share of each cell's ${KNN} nearest neighbours that are its own type (0.5 = chance), using all genes or the top ${NPC} principal components.`);
  let nIdx = 4;
  slider(f3.controls, { label: 'Noise genes', min: 0, max: NOISE.length - 1, step: 1, value: nIdx, fmt: (v) => NOISE[v], onInput: (v) => { nIdx = v; paint3(); } });
  legend(f3.root, [['cell type A', cat(0)], ['cell type B', cat(1)], ['kNN on all genes', cat(2)], [`kNN on top ${NPC} PCs`, cat(6)]]);
  const st3 = stats(f3.root);
  const row3 = d3.select(f3.body).append('div').attr('class', 'fig-row').node();
  const l3 = document.createElement('div'), rr3 = document.createElement('div');
  row3.append(l3, rr3);
  let P3;
  const Pu = frame(rr3, { width: 460, height: 380, x: [0, 2000], y: [0.45, 1.02], xScale: d3.scaleSymlog().constant(20), xLabel: 'number of noise genes', yLabel: `kNN purity (k = ${KNN})` });
  Pu.g.select('.axis.x').call(d3.axisBottom(Pu.x).tickValues(NOISE).tickFormat(d3.format('~g')));
  Pu.g.append('line').attr('x1', 0).attr('x2', Pu.w).attr('y1', Pu.y(0.5)).attr('y2', Pu.y(0.5)).attr('stroke', css('--muted')).attr('stroke-dasharray', '4 4');
  Pu.g.append('text').attr('x', Pu.w - 4).attr('y', Pu.y(0.5) - 5).attr('text-anchor', 'end').attr('font-size', 11).attr('fill', ink2()).text('chance');
  [['raw', 2], ['pc', 6]].forEach(([key, c]) => {
    const pts = NOISE.map((g) => ({ x: g, y: res3[g][key] }));
    Pu.layer('l' + key).append('path').datum(pts).attr('fill', 'none').attr('stroke', cat(c)).attr('stroke-width', 2)
      .attr('d', d3.line().x((p) => Pu.x(p.x)).y((p) => Pu.y(p.y)));
    Pu.layer('p' + key).selectAll('circle').data(pts).join('circle').attr('cx', (p) => Pu.x(p.x)).attr('cy', (p) => Pu.y(p.y)).attr('r', 4)
      .attr('fill', cat(c)).attr('stroke', css('--surface')).attr('stroke-width', 1.2)
      .on('mousemove', (e, p) => tip(e, `${p.x} noise genes<br>${key === 'raw' ? 'all genes' : 'top ' + NPC + ' PCs'}: purity ${p.y.toFixed(2)}`)).on('mouseleave', untip);
  });
  Pu.layer('cur');

  function paint3() {
    const g = NOISE[nIdx], R = res3[g];
    l3.innerHTML = '';
    const ext = d3.max(R.scores, (s) => Math.max(Math.abs(s[0]), Math.abs(s[1]))) * 1.08;
    P3 = frame(l3, { width: 460, height: 380, x: [-ext, ext], y: [-ext, ext], xLabel: 'PC1', yLabel: 'PC2' });
    P3.layer('pts').selectAll('circle').data(R.scores).join('circle').attr('cx', (s) => P3.x(s[0])).attr('cy', (s) => P3.y(s[1])).attr('r', 4.5)
      .attr('fill', (s, i) => cat(type[i])).attr('stroke', css('--surface')).attr('stroke-width', 1.2)
      .on('mousemove', (e, s) => tip(e, `cell type ${type[R.scores.indexOf(s)] ? 'B' : 'A'}<br>PC1 ${s[0].toFixed(2)}, PC2 ${s[1].toFixed(2)}`)).on('mouseleave', untip);
    Pu.layer('cur').selectAll('circle').data([R.raw, R.pc]).join('circle').attr('cx', Pu.x(g)).attr('cy', (v) => Pu.y(v)).attr('r', 8)
      .attr('fill', 'none').attr('stroke', ink()).attr('stroke-width', 2);
    st3.set({ 'genes in total': NSIG + g, 'purity, all genes': R.raw.toFixed(2), [`purity, top ${NPC} PCs`]: R.pc.toFixed(2), 'variance on PC1': (100 * R.varShare).toFixed(1) + '%' });
  }
  paint3();

  const s = summary;
  section(main, 'What to notice', `<ul>
    <li>At <b>d = 2</b> distances spread from almost 0 to about twice the mean, and the farthest point from a typical point is on average ${(s[2].contrast + 1).toFixed(0)}× farther away than its nearest neighbour. At <b>d = 1000</b> the histogram is a spike (SD ÷ mean = ${s[1000].cv.toFixed(3)}) and the farthest point is only ${(100 * s[1000].contrast).toFixed(0)}% farther than the nearest. "Nearest" has nearly lost its meaning.</li>
    <li>The inscribed ball holds 78.5% of the square at d = 2, ${(100 * ballFrac(10)).toFixed(2)}% of the cube at d = 10, and only ${(100 * ballFrac(20)).toExponential(1)}% at d = 20. Random points almost never land in the middle: the Monte Carlo column finds none at all by d = 20.</li>
    <li>A skin only 5% thick holds ${(100 * shellFrac(10, 0.05)).toFixed(0)}% of the volume at d = 10 and ${(100 * shellFrac(100, 0.05)).toFixed(1)}% at d = 100. In high dimensions every point is an "edge case".</li>
    <li><b>PCA rescue</b>: with ${NOISE[4]} noise genes, kNN on all genes finds only ${res3[1000].raw.toFixed(2)} same-type neighbours (chance is 0.5), while kNN on the top ${NPC} PCs finds ${res3[1000].pc.toFixed(2)}. The noise genes add a large random amount to every distance, and PCA discards most of them because they share no structure.</li>
  </ul>`);

  steps(main, 'How it works', [
    'A squared distance is a <b>sum over dimensions</b>. With d independent coordinates, the sum grows like d, but its random fluctuation only grows like √d. So the spread relative to the mean shrinks like 1/√d: every distance converges to the same value.',
    'Volume grows with the <b>product</b> of side lengths. Shrinking each side by a factor (1 − ε) shrinks the volume by (1 − ε)<sup>d</sup>, which goes to 0. So almost all volume is near the boundary.',
    'Methods built on "nearby points are similar" (kNN, kernel methods, density estimation, clustering) need neighbours to be meaningfully closer than non-neighbours. That needs either lots more data or fewer, more informative dimensions.',
    'Real data rescue us because they are rarely uniform: they lie near a <b>low-dimensional structure</b> (cells vary along a few programmes, not 20,000 independent axes). PCA and other embeddings find that structure and drop the noise directions.',
  ]);
  math(main, 'for i.i.d. coordinates:  E‖x − y‖² = d · 2σ²,   SD(‖x − y‖²) ∝ √d   ⇒  spread / mean ∝ 1/√d\nball volume in [−1,1]^d :  π^(d/2) / Γ(d/2 + 1) / 2^d        shell of thickness ε :  1 − (1 − ε)^d');

  callout(main, '', 'In biology', 'A single cell is a point in ~20,000-dimensional gene space, most of it technical noise from sparse counts. Scanpy and Seurat therefore select 2,000 highly variable genes, run PCA, and build the kNN graph (for Leiden clustering and UMAP) on the top 30–50 PCs, never on raw genes. scVI does the same job with a learned 10–30-dimensional latent space. In bulk biomarker studies the curse shows up as p ≫ n: 20,000 genes and 100 patients means some gene set will separate responders by chance, which is why <a href="#/archive/topics/regularization">regularisation</a> and honest <a href="#/archive/topics/cross-validation">cross-validation</a> are essential.');
  callout(main, 'warn', 'Pitfalls', '<ul><li>Adding features is not free. Uninformative features dilute distances (see the purity curve), so feature selection can improve a kNN or clustering result.</li><li>The concentration result is for independent, comparable coordinates. Real features are correlated, so the <i>effective</i> dimension is lower. That is why kNN still works on PCs.</li><li>Choosing the number of PCs matters: too few loses rare cell types, too many lets the noise back in. Look at the variance-explained "elbow" and check that conclusions are stable across choices.</li><li>Euclidean distance is not the only option. Cosine or correlation distance removes some of the library-size effect in expression data.</li></ul>');
  seeAlso(main, [['concepts', 'pca', 'PCA'], ['concepts', 'knn', 'k-nearest neighbours'], ['concepts', 'graph-clustering', 'kNN graphs & Leiden'], ['concepts', 'scaling', 'Feature scaling']]);
}
