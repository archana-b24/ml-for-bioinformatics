// Spatial step 5: neighbourhood enrichment (who touches whom, vs shuffled labels) and ligand-receptor proximity scores.
// Data: sp_data.json (cells) + sp_niches.json (precomputed z-scores over 300 label permutations; per-cell receiver scores).
import { header, figure, select, button, stats, section, steps, callout, math, legend, seeAlso, el, tip, untip, animate } from '../../lib/ui.js';
import { frame, css, cat, ink2, muted, diverging } from '../../lib/plot.js';
import { rng } from '../../lib/data.js';
import { stepCard, omicsData } from '../../omics/_shared.js';
import { tissueMap, ctColor, delaunayEdges, secShort, pct } from '../../omics/_spatial.js';
const d3 = window.d3;

export default async function render(main) {
  header(main, 'Niches & cell–cell communication',
    'Two cell types are <b>enriched as neighbours</b> if they touch more often than they would if the same cells were scattered at random. Ask that of CD8 T cells and tumour cells, and the inflamed and excluded tumours that bulk could not separate <b>look opposite</b>.');
  const card = stepCard(main, {
    stage: 'Spatial · step 5 of 6',
    question: 'Are CD8 T cells in contact with tumour cells, or held in the fibroblast stroma? And are they receiving the signals (CXCL9/10 to attract them, TGF-β to exclude them) that the bulk modules hinted at?',
    data: '…', method: 'Count graph edges between each pair of cell types, compare with label permutations (z-score). For ligand–receptor pairs, score ligand in one cell × receptor in its neighbour, again against permutations.',
    verdict: 'stats', why: ' Both are permutation tests on a fixed graph. The biology lives in the null you choose and in the gene pairs you test, not in a learned model.',
  });
  const D = await omicsData(main, 'sp_data.json');
  const N = await omicsData(main, 'sp_niches.json');
  if (!D || !N) return;
  const CT = N.cell_types, NT = CT.length, NS = N.sections.length, TUM = 0, CD8 = 1, FIB = 6;
  const iI = D.phen.indexOf('inflamed'), iE = D.phen.indexOf('excluded'), iD = D.phen.indexOf('desert');
  card.querySelectorAll('.step-row')[1].querySelector('span').innerHTML = `The ${NS} single-cell-resolution sections with annotated cell types and their Delaunay graphs (≤ 35 µm edges). ${N.nperm} label permutations per section. Ligand–receptor pairs: ${N.pairs.join(', ')}.`;
  const E = D.cells.map((s) => delaunayEdges(s.x, s.y, 35));

  // ------------------------------------------------------------------ figure 1: enrichment heatmap + CD8 rows
  const fig1 = figure(main, 'Neighbourhood enrichment: who touches whom more than chance?',
    `Left: z-score for every pair of cell types in one section (red = neighbours more often than with shuffled labels, blue = less often). Right: the CD8 T row for all ${NS} sections. Hover for values.`);
  let sec = iI;
  select(fig1.controls, { label: 'Section', value: String(sec), options: D.sections.map((s, i) => [String(i), s]), onChange: (v) => { sec = +v; paint1(); } });
  const rd1 = stats(fig1.root);
  const row1 = el('div', { class: 'fig-row' }); fig1.body.append(row1);
  const hW = el('div'), bW = el('div'); row1.append(hW, bW);
  const zmax = 20;
  const dv = diverging(zmax);
  const lgW = el('div'); fig1.controls.append(lgW);
  legend(lgW, [[`z ≤ −${zmax}`, dv(-zmax)], ['0', dv(0)], [`z ≥ +${zmax}`, dv(zmax)]]);

  // grouped bars: CD8 vs each type, per section
  const FB = frame(bW, { width: 480, height: 330, x: [0, 1], y: [-20, 15], yLabel: 'z (CD8 T next to …)', xScale: d3.scaleBand().padding(0.2), margin: { bottom: 56 } });
  const showT = [TUM, FIB, CD8, 5];
  FB.x.domain(showT.map((t) => CT[t]));
  const allz = N.enrichment.flatMap((e) => showT.map((t) => e.z[CD8][t]));
  FB.y.domain([Math.min(-3, d3.min(allz)) - 1, Math.max(3, d3.max(allz)) + 1]);
  FB.g.select('.axis.x').call(d3.axisBottom(FB.x)); FB.g.select('.axis.y').call(d3.axisLeft(FB.y).ticks(6));
  FB.g.select('.gridline').call(d3.axisLeft(FB.y).ticks(6).tickSize(-FB.w).tickFormat(''));
  const sub = d3.scaleBand().domain(d3.range(NS)).range([0, FB.x.bandwidth()]).padding(0.08);
  FB.layer('z').append('line').attr('x1', 0).attr('x2', FB.w).attr('y1', FB.y(0)).attr('y2', FB.y(0)).attr('stroke', css('--axis'));
  const phenCol = (i) => cat(['inflamed', 'excluded', 'desert'].indexOf(D.phen[i]));
  const barsB = FB.layer('bars').selectAll('rect').data(showT.flatMap((t) => d3.range(NS).map((i) => ({ t, i, z: N.enrichment[i].z[CD8][t] })))).join('rect')
    .attr('x', (d) => FB.x(CT[d.t]) + sub(d.i)).attr('width', sub.bandwidth()).attr('y', (d) => FB.y(Math.max(0, d.z))).attr('height', (d) => Math.abs(FB.y(d.z) - FB.y(0)))
    .attr('fill', (d) => phenCol(d.i)).attr('rx', 2)
    .on('mousemove', (e, d) => tip(e, `<b>${D.sections[d.i]}</b><br>CD8 T – ${CT[d.t]}: z = ${d.z.toFixed(2)}`)).on('mouseleave', untip);
  legend(bW, [['inflamed (S1, S2)', phenCol(iI)], ['excluded', phenCol(iE)], ['desert', phenCol(iD)]]);

  function paint1() {
    hW.innerHTML = '';
    const z = N.enrichment[sec].z, cell = 36, left = 92, top = 74, w = left + NT * cell + 40, h = top + NT * cell + 6;
    const svg = d3.select(hW).append('svg').attr('viewBox', `0 0 ${w} ${h}`).attr('width', w);
    z.forEach((row, a) => row.forEach((v, b) => {
      svg.append('rect').attr('x', left + b * cell).attr('y', top + a * cell).attr('width', cell - 2).attr('height', cell - 2).attr('rx', 3).attr('fill', dv(v))
        .attr('stroke', (a === CD8 && (b === TUM || b === FIB)) ? css('--ink') : 'none').attr('stroke-width', 2)
        .on('mousemove', (e) => tip(e, `${CT[a]} – ${CT[b]}: z = ${v.toFixed(2)}<br>(${N.enrichment[sec].count[a]} and ${N.enrichment[sec].count[b]} cells)`)).on('mouseleave', untip);
      svg.append('text').attr('x', left + b * cell + cell / 2 - 1).attr('y', top + a * cell + cell / 2 + 4).attr('text-anchor', 'middle').attr('font-size', 10).attr('pointer-events', 'none')
        .attr('fill', Math.abs(v) > zmax * 0.6 ? css('--surface') : css('--ink')).text(Math.abs(v) >= 99.5 ? Math.sign(v) * 99 : Math.round(v));
    }));
    CT.forEach((c, t) => {
      svg.append('text').attr('x', left - 6).attr('y', top + t * cell + cell / 2 + 4).attr('text-anchor', 'end').attr('font-size', 11).attr('fill', ink2()).attr('font-weight', t === CD8 ? 700 : 400).text(c);
      svg.append('text').attr('transform', `translate(${left + t * cell + cell / 2 + 3},${top - 6}) rotate(-50)`).attr('font-size', 11).attr('fill', ink2()).text(c);
    });
    barsB.attr('opacity', (d) => (d.i === sec ? 1 : 0.4)).attr('stroke', (d) => (d.i === sec ? css('--ink') : 'none'));
    const zz = N.enrichment[sec].z;
    rd1.set({ section: secShort(D.sections[sec]), 'CD8 T cells': N.enrichment[sec].count[CD8], 'z: CD8 T – Tumour': zz[CD8][TUM].toFixed(2), 'z: CD8 T – Fibroblast': zz[CD8][FIB].toFixed(2), 'z: Tumour – Fibroblast': zz[TUM][FIB].toFixed(2) });
  }
  paint1();

  // ------------------------------------------------------------------ figure 2: the permutation, live
  const fig2 = figure(main, 'Where the z-score comes from: shuffle the labels',
    'Keep every cell where it is and keep the graph; shuffle only the cell-type labels, and recount edges between the two chosen types. The map shows the current shuffle (or the real labels); the histogram collects the shuffled counts.');
  let sec2 = iE, ta = CD8, tb = TUM, nulls = [], stopP = null, curLab = null;
  select(fig2.controls, { label: 'Section', value: String(sec2), options: D.sections.map((s, i) => [String(i), s]), onChange: (v) => { sec2 = +v; reset2(); } });
  select(fig2.controls, { label: 'Type A', value: String(ta), options: CT.map((c, i) => [String(i), c]), onChange: (v) => { ta = +v; reset2(); } });
  select(fig2.controls, { label: 'Type B', value: String(tb), options: CT.map((c, i) => [String(i), c]), onChange: (v) => { tb = +v; reset2(); } });
  button(fig2.controls, 'Shuffle 200×', () => run2(), true);
  button(fig2.controls, 'Reset', () => reset2());
  const lg2 = el('div'); fig2.controls.append(lg2);
  const rd2 = stats(fig2.root);
  const row2 = el('div', { class: 'fig-row' }); fig2.body.append(row2);
  const m2 = el('div'), h2 = el('div'); row2.append(m2, h2);
  const M2 = tissueMap(m2, { size: D.size, px: 400 });
  const F2 = frame(h2, { width: 480, height: 330, x: [0, 100], y: [0, 10], xLabel: 'A–B edges', yLabel: 'shuffles' });
  const countAB = (lab) => E[sec2].filter(([i, j]) => (lab[i] === ta && lab[j] === tb) || (lab[i] === tb && lab[j] === ta)).length;
  function reset2() { if (stopP) stopP(); stopP = null; nulls = []; curLab = null; paint2(); }
  function run2() {
    reset2();
    const r = rng(4242 + sec2 * 10 + ta * 100 + tb), lab0 = D.cells[sec2].type;
    stopP = animate(() => {
      for (let q = 0; q < 4 && nulls.length < 200; q++) {
        const p = lab0.slice(); for (let i = p.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [p[i], p[j]] = [p[j], p[i]]; }
        nulls.push(countAB(p)); curLab = p;
      }
      paint2();
      if (nulls.length >= 200) { curLab = null; paint2(); return false; }
      return true;
    });
  }
  function paint2() {
    const s = D.cells[sec2], lab = curLab || s.type;
    lg2.innerHTML = ''; legend(lg2, [[`A: ${CT[ta]}`, ctColor(ta)], [`B: ${CT[tb]}`, ctColor(tb)], ['other', css('--border')]]);
    M2.draw(s.x, s.y, (i) => (lab[i] === ta ? ctColor(ta) : lab[i] === tb ? ctColor(tb) : css('--border')), (i) => (lab[i] === ta || lab[i] === tb ? 6 : 4.5));
    M2.hover((x, y, near) => { const i = near(x, y); return i < 0 ? null : `${curLab ? 'shuffled label: ' : ''}${CT[lab[i]]}`; });
    const obs = countAB(s.type);
    const lo = Math.min(obs, d3.min(nulls) ?? obs), hi = Math.max(obs, d3.max(nulls) ?? obs);
    const pad = Math.max(5, (hi - lo) * 0.15);
    F2.x.domain([Math.max(0, lo - pad), hi + pad]);
    const bins = d3.bin().domain(F2.x.domain()).thresholds(40)(nulls);
    F2.y.domain([0, Math.max(8, d3.max(bins, (b) => b.length) || 0)]);
    F2.g.select('.axis.x').call(d3.axisBottom(F2.x).ticks(6)); F2.g.select('.axis.y').call(d3.axisLeft(F2.y).ticks(4));
    F2.g.select('.gridline').call(d3.axisLeft(F2.y).ticks(4).tickSize(-F2.w).tickFormat(''));
    F2.layer('b').selectAll('rect').data(bins).join('rect').attr('x', (b) => F2.x(b.x0)).attr('width', (b) => Math.max(0, F2.x(b.x1) - F2.x(b.x0) - 1)).attr('y', (b) => F2.y(b.length)).attr('height', (b) => F2.h - F2.y(b.length)).attr('fill', muted());
    F2.layer('o').selectAll('line').data([obs]).join('line').attr('x1', F2.x(obs)).attr('x2', F2.x(obs)).attr('y1', 0).attr('y2', F2.h).attr('stroke', css('--ink')).attr('stroke-width', 2.5);
    F2.layer('ol').selectAll('text').data([obs]).join('text').attr('x', F2.x(obs) + 5).attr('y', 14).attr('font-size', 12).attr('fill', ink2()).text(`real: ${obs}`);
    const mu = d3.mean(nulls), sdv = d3.deviation(nulls);
    rd2.set({ 'real A–B edges': obs, shuffles: nulls.length, 'expected under shuffling': nulls.length ? mu.toFixed(1) : '–', 'live z': nulls.length > 2 && sdv > 0 ? ((obs - mu) / sdv).toFixed(2) : '–', [`build z (${N.nperm} shuffles)`]: N.enrichment[sec2].z[ta][tb].toFixed(2) });
  }
  paint2();
  run2();

  // ------------------------------------------------------------------ figure 3: ligand-receptor
  const fig3 = figure(main, 'Ligand–receptor proximity: who is receiving which signal?',
    'Each cell\'s receiver score = its receptor expression × the mean ligand expression of its neighbours. The map shows that score for every CD8 T cell (outlined discs) on top of the tumour and fibroblast layout; the bars show its mean over CD8 T cells in each section, the cells whose fate the question is about.');
  let pair = N.pairs[0], sec3 = iI;
  select(fig3.controls, { label: 'Pair (ligand → receptor)', value: pair, options: N.pairs, onChange: (v) => { pair = v; paint3(); } });
  select(fig3.controls, { label: 'Section', value: String(sec3), options: D.sections.map((s, i) => [String(i), s]), onChange: (v) => { sec3 = +v; paint3(); } });
  const lg3 = el('div'); fig3.controls.append(lg3);
  const rd3 = stats(fig3.root);
  const row3 = el('div', { class: 'fig-row' }); fig3.body.append(row3);
  const m3 = el('div'), b3 = el('div'); row3.append(m3, b3);
  const M3 = tissueMap(m3, { size: D.size, px: 420 });
  const F3 = frame(b3, { width: 460, height: 320, x: [0, 1], y: [0, 0.1], yLabel: 'mean receiver score on CD8 T cells', xScale: d3.scaleBand().padding(0.3), margin: { left: 58, bottom: 34 } });
  F3.x.domain(D.sections.map(secShort));
  F3.g.select('.axis.x').call(d3.axisBottom(F3.x));
  function paint3() {
    const s = D.cells[sec3], L = N.lr[sec3][pair], rv = L.recv;
    const r8 = d3.range(s.n).filter((i) => s.type[i] === CD8).map((i) => rv[i]).sort(d3.ascending);
    const hi = Math.max(0.02, d3.quantile(r8, 0.95) || 0);
    const sc = d3.interpolateRgb(css('--surface'), css('--accent'));
    M3.overlay((ctx, k) => {
      s.type.forEach((t, i) => {
        if (t !== CD8) return;
        ctx.fillStyle = sc(Math.min(1, rv[i] / hi)); ctx.strokeStyle = css('--ink'); ctx.lineWidth = 1.3;
        ctx.beginPath(); ctx.arc(s.x[i] * k, s.y[i] * k, 8 * k, 0, 6.2832); ctx.fill(); ctx.stroke();
      });
    });
    M3.draw(s.x, s.y, (i) => (s.type[i] === TUM ? css('--muted') : s.type[i] === FIB ? ctColor(FIB) : s.type[i] === CD8 ? null : css('--border')), () => 4.5, (i) => (s.type[i] === TUM ? 0.45 : s.type[i] === FIB ? 0.35 : 1));
    lg3.innerHTML = ''; legend(lg3, [['tumour cell', css('--muted')], ['fibroblast', ctColor(FIB)], ['CD8 T, score 0', sc(0)], [`CD8 T, score ≥ ${hi.toFixed(2)}`, sc(1)]]);
    M3.hover((x, y, near) => { const i = near(x, y); return i < 0 ? null : `<b>${CT[s.type[i]]}</b> · ${D.regions[s.region[i]]}<br>receiver score ${rv[i].toFixed(2)}`; });
    const vals = N.lr.map((l) => l[pair].recv_by_type[CD8]);
    F3.y.domain([0, d3.max(vals) * 1.2 || 0.01]);
    F3.g.select('.axis.y').call(d3.axisLeft(F3.y).ticks(5));
    F3.g.select('.gridline').call(d3.axisLeft(F3.y).ticks(5).tickSize(-F3.w).tickFormat(''));
    F3.layer('b').selectAll('rect').data(vals).join('rect').attr('x', (d, i) => F3.x(secShort(D.sections[i]))).attr('width', F3.x.bandwidth())
      .attr('y', (d) => F3.y(d)).attr('height', (d) => F3.h - F3.y(d)).attr('rx', 3).attr('fill', (d, i) => phenCol(i)).attr('stroke', (d, i) => (i === sec3 ? css('--ink') : 'none')).attr('stroke-width', 1.5)
      .on('mousemove', (e, d) => { const i = vals.indexOf(d); tip(e, `<b>${D.sections[i]}</b><br>${pair} on CD8 T cells: ${d.toFixed(4)}<br>section-wide proximity z = ${N.lr[i][pair].z.toFixed(2)}`); }).on('mouseleave', untip);
    F3.layer('t').selectAll('text').data(vals).join('text').attr('x', (d, i) => F3.x(secShort(D.sections[i])) + F3.x.bandwidth() / 2).attr('y', (d) => F3.y(d) - 5).attr('text-anchor', 'middle').attr('font-size', 11).attr('fill', ink2()).text((d) => d.toFixed(3));
    rd3.set({ pair, 'section-wide proximity z': L.z.toFixed(2), 'mean on CD8 T cells': L.recv_by_type[CD8].toFixed(4), 'mean on fibroblasts': L.recv_by_type[FIB].toFixed(4), 'mean on tumour cells': L.recv_by_type[TUM].toFixed(4) });
  }
  legend(b3, [['inflamed', phenCol(iI)], ['excluded', phenCol(iE)], ['desert', phenCol(iD)]]);
  paint3();

  // ------------------------------------------------------------------ prose (numbers from the JSON)
  const zI = (i, a, b) => N.enrichment[i].z[a][b].toFixed(2);
  const infl = D.phen.map((p, i) => (p === 'inflamed' ? i : -1)).filter((i) => i >= 0);
  const rb = (p, i, t = CD8) => N.lr[i][p].recv_by_type[t].toFixed(3);
  const P9 = N.pairs.find((p) => p.startsWith('CXCL9')), PT = N.pairs.find((p) => p.startsWith('TGFB1')), P13 = N.pairs.find((p) => p.startsWith('CXCL13'));
  const ratio = (p) => (N.lr[iE][p].recv_by_type[CD8] / d3.mean(infl, (i) => N.lr[i][p].recv_by_type[CD8])).toFixed(1);
  section(main, 'What to notice', `<ul>
    <li><b>The climax of the study.</b> CD8 T – Tumour: z = ${infl.map((i) => zI(i, CD8, TUM)).join(' and ')} in the inflamed sections, <b>${zI(iE, CD8, TUM)}</b> in the excluded one. CD8 T – Fibroblast: ${infl.map((i) => zI(i, CD8, FIB)).join(' and ')} versus <b>${N.enrichment[iE].z[CD8][FIB] > 0 ? '+' : ''}${zI(iE, CD8, FIB)}</b>. The excluded tumour has CD8 T cells (${N.enrichment[iE].count[CD8]} of them, against ${infl.map((i) => N.enrichment[i].count[CD8]).join(' and ')}), but they sit with the fibroblasts, not the tumour.</li>
    <li>Tumour – Fibroblast is strongly negative everywhere (${D.sections.map((s, i) => zI(i, TUM, FIB)).join(', ')}): nests and stroma are separate compartments. Large |z| for abundant types is easy; it is the CD8 rows that carry the biology.</li>
    <li>In the shuffle figure, the excluded section has ${countAB0(iE)} real CD8 T – tumour edges; shuffling labels gives many more. The size of that gap relative to the spread of the shuffles is the z-score. Try ${secShort(D.sections[iI])}: the real count sits to the right of the null.</li>
    <li><b>Signals.</b> CD8 T cells receive ${P9} at ${infl.map((i) => rb(P9, i)).join(' / ')} in the inflamed sections versus ${rb(P9, iE)} in the excluded one, and ${PT} at ${infl.map((i) => rb(PT, i)).join(' / ')} versus ${rb(PT, iE)} (${ratio(PT)}× the inflamed mean). That fits the planted story: attracted less, inhibited more.</li>
    <li>But check ${secShort(D.sections[iD])}: its few CD8 T cells (${N.enrichment[iD].count[CD8]}) also get ${rb(PT, iD)} of TGF-β signal, about the same as the excluded section. The score measures "receptor-bearing cell sits next to ligand", and in any fibroblast-rich stroma that is true. It is not proof of signalling.</li>
    <li>${P13} has a section-wide z of ${infl.map((i) => N.lr[i][P13].z.toFixed(1)).join(' and ')} in the inflamed sections (the TLS, where B cells carry CXCR5) and ${N.lr[iE][P13].z.toFixed(1)} in the excluded one, which has no TLS.</li>
  </ul>`);
  function countAB0(i) { const s = D.cells[i]; return E[i].filter(([a, b]) => (s.type[a] === CD8 && s.type[b] === TUM) || (s.type[b] === CD8 && s.type[a] === TUM)).length; }

  steps(main, 'How it works', [
    'Start from the neighbour graph and the cell-type labels.',
    'For every pair of types (A, B), count graph edges joining an A cell and a B cell.',
    `Shuffle the labels across cells (${N.nperm} times), keeping positions and graph, and recount. This null keeps each type's abundance but destroys any spatial preference.`,
    'z = (observed − mean of shuffles) / sd of shuffles. Positive: A and B are neighbours more than chance (co-localised); negative: they avoid each other.',
    'For a ligand → receptor pair, score each edge by ligand(i) × receptor(j) and compare with shuffled expression; for interpretation, average the receiver score over the cell type you care about (here CD8 T cells).',
  ]);
  math(main, 'n_AB = #{edges (i,j): type_i = A, type_j = B}\nz_AB = (n_AB − mean(n_AB | shuffled labels)) / sd(n_AB | shuffled labels)\nreceiver_j(L→R) = R_j · mean_{i ∈ nbrs(j)} L_i');

  callout(main, '', 'In biology', `<ul>
    <li><b>Squidpy</b> <code>sq.gr.nhood_enrichment</code> (this exact test), <code>co_occurrence</code> and <code>ligrec</code> (CellPhoneDB-style permutation test). histoCAT and the imaging-mass-cytometry community introduced the neighbourhood permutation test.</li>
    <li><b>CellChat</b> (spatial mode), <b>LIANA</b> / LIANA+ (a consensus over several ligand–receptor methods and resources), <b>COMMOT</b> (optimal transport), <b>NicheNet</b> (links ligands to downstream target genes) and <b>MISTy</b> (explains a cell's expression from its neighbourhood).</li>
    <li>In NSCLC and melanoma, the distance of CD8 T cells to tumour cells, and their enrichment in tumour versus stroma, are among the most reproducible spatial correlates of response to immune checkpoint blockade (e.g. multiplex IF and CODEX studies).</li>
  </ul>`);
  callout(main, 'warn', 'Pitfalls', `<ul>
    <li><b>The null decides the answer.</b> Shuffling labels everywhere ignores tissue structure: tumour–fibroblast avoidance is "significant" because nests exist. Restricting shuffles within domains asks a sharper question.</li>
    <li><b>Co-location is not communication.</b> Ligand × receptor proximity is a hypothesis generator; real signalling needs protein, the right receptor complex and downstream response (NicheNet-style target genes help).</li>
    <li><b>Segmentation spill-over</b> makes neighbouring cells share transcripts, which inflates ligand × receptor products across contacts.</li>
    <li><b>z-scores scale with cell numbers.</b> Compare effect sizes (observed / expected) across sections of different size, and remember n = ${NS} patients here.</li>
  </ul>`);
  seeAlso(main, [['omics', 'sp-domains', 'Spatial domains'], ['omics', 'sp-biomarker', 'Back to the biomarker'], ['concepts', 'multiple-testing', 'Multiple testing'], ['omics', 'bulk-de', 'Differential expression']]);
}
