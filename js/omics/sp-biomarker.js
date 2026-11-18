// Spatial transcriptomics → biomarker, lesson format.
// Arc: what → why (bulk cannot see location) → where → how it works (7 steps: spots vs cells → neighbour graph → Moran's I →
//      domains → deconvolution → neighbourhood enrichment → CD8-in-tumour vs CD8-in-stroma lab) → pitfalls → recap.
// Data: data/omics/sp_*.json (simulated sections; see scripts/mock_biology.py). Every number in the prose is computed at render time.
import { header, figure, slider, select, button, checkbox, stats, legend, seeAlso, animate, every, tip, untip, el, html } from '../lib/ui.js';
import { frame, css, cat, ink, ink2, muted, diverging } from '../lib/plot.js';
import { rng } from '../lib/data.js';
import { lessonNav, chapter, prose, rung, worked, quiz, cards, table, code, recap } from '../lib/lesson.js';
import { stepCard, omicsData } from './_shared.js';
import { tissueMap, ctColor, regionColor, exprScale, delaunayEdges, neighbours, lognorm, moran, secShort, pct } from './_spatial.js';
import { PHENOS, phenoColor, auc } from './_bulk_common.js';
const d3 = window.d3;

const f1 = (v) => v.toFixed(1), f2 = (v) => v.toFixed(2), f3 = (v) => v.toFixed(3);
const TUM = 0, CD8 = 1, FIB = 6;
const FEAT = {
  ifn_bulk: { label: 'bulk IFN-γ score', short: 'bulk IFN-γ', kind: 'bulk' },
  cd8_frac: { label: 'CD8 T fraction of all cells', short: 'CD8 fraction', kind: 'bulk-like' },
  d_tumour: { label: 'CD8 T per 100 cells inside the tumour compartment', short: 'CD8 in tumour', kind: 'spatial' },
  infil: { label: 'infiltration ratio, log2(tumour / stroma CD8 density)', short: 'infiltration ratio', kind: 'spatial' },
  frac_inside: { label: 'share of CD8 T cells inside the tumour compartment', short: 'share inside', kind: 'spatial' },
};
const featCol = (k) => cat({ ifn_bulk: 7, cd8_frac: 3, d_tumour: 2, infil: 0, frac_inside: 4 }[k]);

export default async function render(main) {
  header(main, 'Spatial biomarker',
    'Bulk and single-cell RNA-seq discard where cells were. Spatial transcriptomics keeps the <b>position of every measurement</b>, which allows one question bulk data cannot answer: are the CD8 T cells <b>inside the tumour</b> (an inflamed tumour) or <b>held in the stroma outside it</b> (an excluded tumour)?');
  lessonNav(main);
  const card = stepCard(main, {
    stage: 'Spatial · biomarker',
    question: 'Are the CD8 T cells inside the tumour nests or held outside them, and does a spatial feature capture that better than the bulk IFN-γ score?',
    data: '…',
    method: 'Neighbour graph on cell or spot coordinates, then Moran\'s I, spatial clustering, deconvolution against the single-cell reference, neighbourhood enrichment, and finally one hand-built feature per section evaluated by ROC AUC.',
    verdict: 'both',
    why: 'The neighbour graph is a modelling choice; Moran\'s I and neighbourhood enrichment are statistics (permutation nulls). Domain finding and deconvolution are fitted models (clustering, constrained regression). The final feature is hand-designed, evaluated with statistics, and would become a classifier input.',
  });
  const [S, V, SV, DM, N, B] = await Promise.all(['sp_data', 'sp_deconvolution', 'sp_svg', 'sp_domains', 'sp_niches', 'sp_biomarker'].map((n) => omicsData(main, n + '.json')));
  if (![S, V, SV, DM, N, B].every(Boolean)) return;
  const NS = S.sections.length, CT = S.cell_types, NC = d3.sum(S.cells, (c) => c.n);
  card.querySelectorAll('.step-row')[1].querySelector('span').innerHTML = `${NS} simulated tissue sections (${NC.toLocaleString()} cells, ${S.n_panel}-gene panel, ${V.n_spots} Visium-like spots), plus ${B.n} simulated section summaries, one per patient (the ${B.cohort.filter((c) => c.main).length} main sections and ${B.n - B.cohort.filter((c) => c.main).length} smaller ones; ${B.n_resp} responders).`;
  const iI = S.phen.indexOf('inflamed'), iE = S.phen.indexOf('excluded'), iD = S.phen.indexOf('desert');
  // neighbour graphs: all Delaunay edges up to 100 µm with their lengths; the 35 µm graph is the analysis graph
  const EDG = S.cells.map((s) => delaunayEdges(s.x, s.y, 100).map(([a, b]) => [a, b, Math.hypot(s.x[a] - s.x[b], s.y[a] - s.y[b])]));
  const NB = EDG.map((e, i) => neighbours(S.cells[i].n, e.filter((q) => q[2] <= 35)));
  const ctN = (si, t) => S.cells[si].type.filter((u) => u === t).length;
  const cohort = B.cohort, nMain = cohort.filter((c) => c.main).length;

  // =====================================================================================
  // 1. WHAT
  // =====================================================================================
  let ch = chapter(main, 'what', 'What is a spatial biomarker?', 'In one line: <b>measurements with coordinates → who is next to whom → a per-sample number that describes tissue organisation</b>.');
  prose(ch, `<p><b>Spatial transcriptomics</b> measures gene expression while recording where in the tissue each measurement was made. Two resolutions are common. <b>Single-cell-resolution</b> platforms (Xenium, CosMx, MERFISH) report a gene panel per segmented cell. <b>Spot-based</b> platforms (Visium) report whole-transcriptome counts per ${V.spot_diam} µm spot on a ${V.pitch} µm grid, and each spot contains a mixture of several cells.</p>
    <p>A <b>spatial biomarker</b> is a quantity computed from those coordinates and labels that carries clinical information, such as the density of CD8 T cells inside the tumour compartment. This page builds one in seven steps on simulated sections of a lung tumour, using the same fictional study as the bulk and single-cell pages.</p>`);
  ch.append(pipelineSvg(S, V, NC));
  table(ch, [
    { t: 'Spot', d: `A capture area of ${V.spot_diam} µm that sums the RNA of every cell above it (here ${V.cells_per_spot.min}–${V.cells_per_spot.max} cells).` },
    { t: 'Neighbour graph', d: 'Nodes are cells or spots; an edge joins two that are close. Every spatial statistic is defined on this graph.' },
    { t: "Moran's I", d: 'Correlation between a value and the average of its neighbours\' values. Near 0: no spatial pattern.' },
    { t: 'Domain', d: 'A contiguous tissue region with a consistent make-up (tumour nest, margin, stroma, tertiary lymphoid structure).' },
    { t: 'Deconvolution', d: 'Estimating the cell-type proportions inside each spot from its expression and a single-cell reference.' },
    { t: 'Inflamed / excluded / desert', d: 'Immune phenotypes: T cells inside the tumour, T cells present but outside it, or few T cells anywhere.' },
  ], [['t', 'Term'], ['d', 'Meaning']]);

  // =====================================================================================
  // 2. WHY
  // =====================================================================================
  ch = chapter(main, 'why', 'Why do we need it?', 'Bulk expression reports how many T cells are present, not where they are.');
  prose(ch, `<p>Anti-PD-1 therapy works only if T cells can reach tumour cells. An <b>inflamed</b> tumour has CD8 T cells among tumour cells; an <b>excluded</b> tumour has them, but they remain in fibroblast-rich stroma; a <b>desert</b> has few. A bulk RNA-seq sample averages over the whole piece of tissue, so an inflamed and an excluded tumour can give similar T-cell signal. The two sections below contain comparable numbers of CD8 T cells in different places.</p>`);
  const fw = figure(ch, 'Same T cells, different places', 'Tumour cells (grey), fibroblasts (faint) and CD8 T cells (coloured); other cell types are hidden. Hover a CD8 T cell for its distance to the nearest tumour cell. The numbers underneath each map are what bulk or a bulk-like count would report.');
  legend(fw.controls, [['tumour cell', muted()], ['fibroblast', ctColor(FIB)], ['CD8 T cell', ctColor(CD8)]]);
  const rowW = el('div', { class: 'fig-row' }); fw.body.append(rowW);
  [iI, iE].forEach((si) => {
    const w = el('div'); rowW.append(w);
    w.append(el('div', { style: 'font-size:13px;font-weight:600;color:var(--ink-2);margin:0 0 4px 2px' }, S.sections[si]));
    const s = S.cells[si], M = tissueMap(w, { size: S.size, px: 430 }), m = cohort.find((c) => c.name === S.sections[si]);
    const tum = d3.range(s.n).filter((i) => s.type[i] === TUM), del = d3.Delaunay.from(tum, (i) => s.x[i], (i) => s.y[i]);
    const dist = (i) => { const j = tum[del.find(s.x[i], s.y[i])]; return Math.hypot(s.x[i] - s.x[j], s.y[i] - s.y[j]); };
    M.draw(s.x, s.y, (i) => (s.type[i] === TUM ? muted() : s.type[i] === CD8 ? ctColor(CD8) : s.type[i] === FIB ? ctColor(FIB) : null), (i) => (s.type[i] === CD8 ? 7 : 5), (i) => (s.type[i] === TUM ? 0.45 : s.type[i] === FIB ? 0.3 : 1));
    M.hover((x, y, near) => { const i = near(x, y); return i < 0 || s.type[i] !== CD8 ? null : `CD8 T cell<br>nearest tumour cell: ${dist(i).toFixed(0)} µm`; });
    stats(w).set({ 'CD8 fraction': pct(m.cd8_frac, 1), 'bulk IFN-γ score': f2(m.ifn_bulk), 'CD8 inside tumour compartment': pct(m.frac_inside) });
  });
  const mI = cohort.find((c) => c.name === S.sections[iI]), mE = cohort.find((c) => c.name === S.sections[iE]);
  prose(ch, `<p>The CD8 fractions are ${pct(mI.cd8_frac, 1)} (inflamed) and ${pct(mE.cd8_frac, 1)} (excluded), and the bulk IFN-γ scores are ${f2(mI.ifn_bulk)} and ${f2(mE.ifn_bulk)}. These values differ much less than the tissue organisation does: ${pct(mI.frac_inside)} of CD8 T cells lie inside the tumour compartment in the first section and ${pct(mE.frac_inside)} in the second. Spatial data adds the part of the biology that an average cannot represent.</p>
    <p>The simulation has a limit that matters for the rest of the page: the excluded sections were <b>generated</b> by placing CD8 T cells outside the tumour nests. A feature that measures that placement therefore recovers the generating rule, which is close to circular. Step 7 returns to this.</p>`);

  // =====================================================================================
  // 3. WHERE
  // =====================================================================================
  ch = chapter(main, 'where', 'Where is it used?', 'Any setting where the arrangement of cells carries information that their counts do not.');
  cards(ch, [
    { tag: 'Immuno-oncology', title: 'Inflamed, excluded or desert?', body: 'CD8 density in the tumour core versus the invasive margin, from multiplex immunofluorescence, imaging mass cytometry or spatial RNA. The <b>Immunoscore</b> (CD3/CD8 density by immunohistochemistry) is a validated prognostic test in colorectal cancer.' },
    { tag: 'Tumour microenvironment', title: 'Which cells touch which?', body: 'Neighbourhood enrichment (<code>squidpy.gr.nhood_enrichment</code>) and ligand–receptor scoring on the spatial graph test whether T cells contact tumour cells, macrophages or fibroblasts more often than chance.' },
    { tag: 'Tissue architecture', title: 'Domains and layers', body: 'Spatially aware clustering (BayesSpace, SpaGCN, STAGATE) labels cortical layers, tumour nests, margins and tertiary lymphoid structures without manual annotation.' },
    { tag: 'Gene discovery', title: 'Spatially variable genes', body: 'Moran\'s I (Squidpy, Seurat), SpatialDE, SPARK-X and nnSVG rank genes whose expression is organised in space.' },
    { tag: 'Spot data', title: 'Deconvolution', body: 'cell2location, RCTD (spacexr) and Stereoscope estimate the cell-type mix per Visium spot from a single-cell reference, the spatial counterpart of bulk deconvolution.' },
    { tag: 'Clinical translation', title: 'Predicting response', body: 'A published meta-analysis of multiplex IHC/IF (Lu et al., JAMA Oncology 2019) reported that spatial features predicted anti-PD-(L)1 response better than PD-L1 IHC, TMB or expression profiling alone, on heterogeneous, mostly retrospective cohorts.' },
  ]);
  prose(ch, '<p><b>When not to use it:</b> when the question concerns composition only (single-cell or bulk data are cheaper and larger); when only one or two sections per condition are available (spatial statistics describe sections, not populations, and replication is the limiting factor); and when the sampled area is small relative to the heterogeneity of the tumour.</p>');

  // =====================================================================================
  // 4. HOW
  // =====================================================================================
  ch = chapter(main, 'how', 'How it works', 'Seven steps, from the unit of measurement to a per-section feature evaluated against response.');

  // ---- Step 1: spots versus cells
  let rb = rung(ch, 1, 'Spots versus cells',
    `<p>A spot-based platform does not see cells. It sums the RNA under a circular capture area, so one measurement is a <b>mixture</b>. The figure overlays spots (${V.spot_diam} µm on a ${V.pitch} µm grid) on the true cell map. Changing the spot diameter shows how mixing and coverage trade off.</p>`);
  const f1g = figure(rb, 'One tissue, measured as spots', 'Dots: cells, coloured by type. Discs: spots, coloured by majority cell type or by CD8 T fraction. Hover a spot for its composition.');
  let sec1 = iI, diam = V.spot_diam, view1 = 'maj';
  select(f1g.controls, { label: 'Section', value: String(sec1), options: S.sections.map((s, i) => [String(i), s]), onChange: (v) => { sec1 = +v; paint1(); } });
  slider(f1g.controls, { label: 'spot diameter', min: 20, max: 100, step: 5, value: diam, fmt: (v) => v + ' µm', onInput: (v) => { diam = v; paint1(); } });
  select(f1g.controls, { label: 'Colour spots by', value: view1, options: [['maj', 'majority cell type'], ['cd8', 'CD8 T fraction']], onChange: (v) => { view1 = v; paint1(); } });
  const leg1 = el('div'); f1g.controls.append(leg1);
  const rd1 = stats(f1g.root);
  const row1 = el('div', { class: 'fig-row' }); f1g.body.append(row1);
  const m1w = el('div'); row1.append(m1w);
  const M1 = tissueMap(m1w, { size: S.size, px: 470 });
  const note1 = el('div', { class: 'prose', style: 'font-size:14px;align-self:center' }); row1.append(note1);
  let spots1 = [];
  function paint1() {
    const s = S.cells[sec1], R = diam / 2;
    const ids = d3.range(V.n_spots).filter((j) => V.sec[j] === sec1);
    spots1 = ids.map((j) => {
      const comp = new Array(8).fill(0); let n = 0;
      for (let i = 0; i < s.n; i++) if ((s.x[i] - V.x[j]) ** 2 + (s.y[i] - V.y[j]) ** 2 <= R * R) { comp[s.type[i]]++; n++; }
      return { x: V.x[j], y: V.y[j], n, comp: comp.map((c) => (n ? c / n : 0)), cnt: comp };
    });
    const live = spots1.filter((p) => p.n > 0), f8max = d3.max(live, (p) => p.comp[CD8]) || 1;
    const seq = d3.interpolateRgb(css('--surface-2'), ctColor(CD8));
    leg1.innerHTML = '';
    if (view1 === 'cd8') legend(leg1, [['no CD8 T', seq(0)], [`CD8 T ${pct(f8max / 2)}`, seq(0.5)], [`CD8 T ${pct(f8max)}`, seq(1)]]);
    else legend(leg1, CT.map((c, t) => [c, ctColor(t)]));
    M1.overlay((ctx, k) => live.forEach((p) => {
      ctx.globalAlpha = 0.85; ctx.fillStyle = view1 === 'cd8' ? seq(p.comp[CD8] / f8max) : ctColor(d3.maxIndex(p.comp));
      ctx.beginPath(); ctx.arc(p.x * k, p.y * k, R * k, 0, 6.2832); ctx.fill(); ctx.globalAlpha = 1; ctx.strokeStyle = css('--surface'); ctx.lineWidth = 1.5; ctx.stroke();
    }));
    M1.draw(s.x, s.y, (i) => ctColor(s.type[i]), () => 4.5, () => 0.3);
    M1.hover((x, y) => {
      const p = live.find((q) => (q.x - x) ** 2 + (q.y - y) ** 2 < R * R); if (!p) return null;
      return `<b>Spot (${p.x}, ${p.y})</b>: ${p.n} cells<br>` + p.comp.map((f, t) => [f, t]).filter((q) => q[0] > 0).sort((a, b) => b[0] - a[0]).map(([f, t]) => `${CT[t]} ${pct(f)}`).join('<br>');
    });
    const pure = live.filter((p) => d3.max(p.comp) >= 0.9).length, nCD8 = d3.sum(live, (p) => p.cnt[CD8]), maj8 = live.filter((p) => d3.maxIndex(p.comp) === CD8).length, with8 = live.filter((p) => p.cnt[CD8] > 0).length;
    rd1.set({ 'cells per spot (median)': d3.median(live, (p) => p.n) ?? 0, 'spots dominated by one type (≥ 90%)': live.length ? pct(pure / live.length) : '–', 'cells covered by spots': pct(d3.sum(live, (p) => p.n) / s.n), 'spots containing CD8 T': with8, 'of those, CD8 T is the majority': maj8 });
    note1.innerHTML = `<p>At ${diam} µm the grid covers ${pct(d3.sum(live, (p) => p.n) / s.n)} of the cells and ${nCD8} of the ${ctN(sec1, CD8)} CD8 T cells. A CD8 T cell is rarely alone in a spot: in ${S.sections[sec1]}, CD8 T cells are the majority in ${maj8} of the ${with8} spots that contain one.</p>`;
  }
  paint1();
  prose(rb, `<p>Widening the spot captures more cells per measurement and mixes more types together. Narrowing it gives purer spots and leaves most of the tissue unmeasured. Single-cell-resolution platforms remove the mixing; spot platforms need <b>deconvolution</b> (step 5) to recover the cell types.</p>`);

  // ---- Step 2: neighbour graph
  rb = rung(ch, 2, 'The neighbour graph',
    `<p>Every spatial statistic needs a definition of "near". The usual choice is a graph. Two common constructions are the <b>Delaunay triangulation</b> (connect natural neighbours, then drop edges longer than a maximum length) and <b>k nearest neighbours</b> (connect each cell to its k closest). The analysis graph on this page is Delaunay with edges up to 35 µm, which gives a mean of ${f2(S.cells[0].mean_deg)} neighbours per cell in ${S.sections[0]}.</p>`);
  const f2g = figure(rb, 'Build the graph in a 200 µm window', 'Click the left map to move the window. Right: cells and edges inside it. Edges between a CD8 T cell and a tumour cell are drawn thick.');
  let sec2 = iI, gt = 'delaunay', par = { delaunay: 35, knn: 6 }, win = { x: 380, y: 380 };
  const PARAM = { delaunay: { label: 'maximum edge length', min: 15, max: 100, step: 5, fmt: (v) => v + ' µm' }, knn: { label: 'k', min: 2, max: 15, step: 1, fmt: (v) => 'k = ' + v } };
  select(f2g.controls, { label: 'Section', value: String(sec2), options: S.sections.map((s, i) => [String(i), s]), onChange: (v) => { sec2 = +v; paint2(); } });
  select(f2g.controls, { label: 'Graph', value: gt, options: [['delaunay', 'Delaunay, length-capped'], ['knn', 'k nearest neighbours']], onChange: (v) => { gt = v; setPar(); paint2(); } });
  const slotP = el('span'); f2g.controls.append(slotP);
  let parSl;
  function setPar() { slotP.innerHTML = ''; const P = PARAM[gt]; parSl = slider(slotP, { ...P, value: par[gt], onInput: (v) => { par[gt] = v; paint2(); } }); }
  setPar();
  legend(f2g.controls, CT.map((c, t) => [c, ctColor(t)]));
  const rd2 = stats(f2g.root);
  const row2 = el('div', { class: 'fig-row' }); f2g.body.append(row2);
  const m2w = el('div'), g2w = el('div'); row2.append(m2w, g2w);
  const M2 = tissueMap(m2w, { size: S.size, px: 430 }), WIN = 200;
  M2.canvas.addEventListener('click', (e) => { const r = M2.canvas.getBoundingClientRect(); win = { x: Math.max(0, Math.min(S.size - WIN, (e.clientX - r.left) / r.width * S.size - WIN / 2)), y: Math.max(0, Math.min(S.size - WIN, (e.clientY - r.top) / r.height * S.size - WIN / 2)) }; paint2(); });
  const F2 = frame(g2w, { width: 470, height: 470, x: [0, WIN], y: [WIN, 0], axes: false, grid: false, margin: { top: 4, right: 4, bottom: 4, left: 4 } });
  F2.g.insert('rect', ':first-child').attr('width', F2.w).attr('height', F2.h).attr('rx', 8).attr('fill', css('--surface-2'));
  function paint2() {
    const s = S.cells[sec2], inW = d3.range(s.n).filter((i) => s.x[i] >= win.x && s.x[i] < win.x + WIN && s.y[i] >= win.y && s.y[i] < win.y + WIN), setW = new Set(inW);
    let edges = [];
    if (gt === 'delaunay') edges = EDG[sec2].filter(([a, b, L]) => L <= par.delaunay && (setW.has(a) || setW.has(b)));
    else {
      const cand = d3.range(s.n).filter((i) => Math.abs(s.x[i] - win.x - WIN / 2) < WIN / 2 + 90 && Math.abs(s.y[i] - win.y - WIN / 2) < WIN / 2 + 90), seen = new Set();
      inW.forEach((i) => cand.filter((j) => j !== i).map((j) => [j, (s.x[i] - s.x[j]) ** 2 + (s.y[i] - s.y[j]) ** 2]).sort((a, b) => a[1] - b[1]).slice(0, par.knn).forEach(([j, d]) => {
        const key = Math.min(i, j) * s.n + Math.max(i, j); if (!seen.has(key)) { seen.add(key); edges.push([i, j, Math.sqrt(d)]); }
      }));
    }
    const deg = new Map(); edges.forEach(([a, b]) => { deg.set(a, (deg.get(a) || 0) + 1); deg.set(b, (deg.get(b) || 0) + 1); });
    const ct8 = ([a, b]) => (s.type[a] === CD8 && s.type[b] === TUM) || (s.type[a] === TUM && s.type[b] === CD8);
    const X = (v) => F2.x(v - win.x), Y = (v) => F2.y(v - win.y), cl = (v) => v;
    F2.layer('e').selectAll('line').data(edges.filter(([a, b]) => setW.has(a) && setW.has(b)), (d) => d[0] + '-' + d[1]).join('line')
      .attr('x1', (d) => X(s.x[d[0]])).attr('y1', (d) => Y(s.y[d[0]])).attr('x2', (d) => X(s.x[d[1]])).attr('y2', (d) => Y(s.y[d[1]]))
      .attr('stroke', (d) => (ct8(d) ? ctColor(CD8) : muted())).attr('stroke-width', (d) => (ct8(d) ? 3 : 1)).attr('opacity', (d) => (ct8(d) ? 1 : 0.7))
      .on('mousemove', (e, d) => tip(e, `${CT[s.type[d[0]]]} – ${CT[s.type[d[1]]]}<br>length ${d[2].toFixed(0)} µm`)).on('mouseleave', untip);
    F2.layer('n').selectAll('circle').data(inW, (i) => i).join('circle').attr('cx', (i) => X(s.x[i])).attr('cy', (i) => Y(s.y[i])).attr('r', 6).attr('fill', (i) => ctColor(s.type[i])).attr('stroke', css('--surface')).attr('stroke-width', 1.5)
      .on('mousemove', (e, i) => tip(e, `${CT[s.type[i]]}<br>${deg.get(i) || 0} neighbours in this graph`)).on('mouseleave', untip);
    M2.overlay((ctx, k) => { ctx.strokeStyle = css('--ink'); ctx.lineWidth = 2.5; ctx.strokeRect(win.x * k, win.y * k, WIN * k, WIN * k); });
    M2.draw(s.x, s.y, (i) => ctColor(s.type[i]), () => 4.5, () => 0.55);
    const edgesIn = edges.filter(([a, b]) => setW.has(a) && setW.has(b)), iso = inW.filter((i) => !deg.get(i)).length;
    rd2.set({ 'cells in window': inW.length, 'edges in window': edgesIn.length, 'neighbours per cell (mean)': gt === 'knn' ? '≥ ' + par.knn : f2(d3.mean(inW, (i) => deg.get(i) || 0)), 'cells with no neighbour': iso, 'CD8 T – tumour edges': edgesIn.filter(ct8).length });
  }
  paint2();
  prose(rb, `<p>The graph is a modelling choice. A short length cap leaves isolated cells at the edge of gaps; a long cap links cells across empty space. k nearest neighbours gives every cell the same number of neighbours regardless of local density, which suits unevenly packed tissue but connects sparse cells to distant ones. Results in later steps depend on this choice, so it should be reported.</p>`);

  // ---- Step 3: Moran's I
  const nGenes = SV.genes.length, gi = Object.fromEntries(SV.genes.map((g, i) => [g, i]));
  const sigRaw = SV.res.map((r) => r.q.filter((q) => q < 0.05).length), sigRes = SV.res.map((r) => r.q_res.filter((q) => q < 0.05).length);
  rb = rung(ch, 3, "Moran's I: is a gene organised in space?",
    `<p><b>Moran's I</b> is the slope of a regression of each cell's value on the <b>average of its neighbours' values</b> (both centred). It is near 0 when positions carry no information, positive when neighbours resemble each other, and negative for a checkerboard. Significance comes from a <b>permutation test</b>: shuffle the values across cells, keep the graph, recompute I, and compare. The test below uses ${SV.nperm} shuffles per gene.</p>`);
  const f3g = figure(rb, "Expression against neighbour average", 'Left: one gene\'s counts in one section. Right: each dot is a cell; the slope of the line is Moran\'s I. "Shuffle positions" reassigns the values at random, which removes the spatial pattern.');
  let sec3 = iI, gene = 'CXCL9', shuf = false, seedS = 1;
  select(f3g.controls, { label: 'Section', value: String(sec3), options: S.sections.map((s, i) => [String(i), s]), onChange: (v) => { sec3 = +v; shuf = false; paint3(); } });
  select(f3g.controls, { label: 'Gene', value: gene, options: S.genes.map((g) => [g, g]), onChange: (v) => { gene = v; shuf = false; paint3(); } });
  const shufB = button(f3g.controls, 'Shuffle positions', () => { shuf = !shuf; seedS++; paint3(); });
  const rd3 = stats(f3g.root);
  const row3 = el('div', { class: 'fig-row' }); f3g.body.append(row3);
  const m3w = el('div'), s3w = el('div'); row3.append(m3w, s3w);
  const M3 = tissueMap(m3w, { size: S.size, px: 450 });
  const F3 = frame(s3w, { width: 470, height: 430, x: [-1, 3], y: [-1, 2], xLabel: 'expression of the cell (centred, log-normalised)', yLabel: 'mean of its neighbours' });
  const vcache = {};
  const normV = (si, g) => (vcache[si + g] ||= lognorm(S.cells[si].expr[g], S.cells[si].lib));
  const permute = (v, seed) => { const r = rng(seed), p = v.slice(); for (let i = p.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [p[i], p[j]] = [p[j], p[i]]; } return p; };
  function paint3() {
    const s = S.cells[sec3], v0 = normV(sec3, gene), v = shuf ? permute(v0, seedS) : v0, counts = shuf ? permute(s.expr[gene], seedS) : s.expr[gene], sc = exprScale(s.expr[gene]);
    M3.draw(s.x, s.y, (i) => sc.color(counts[i]) || css('--border'), () => 5.5);
    M3.hover((x, y, near) => { const i = near(x, y); return i < 0 ? null : `${gene}: ${counts[i]} counts<br>${CT[s.type[i]]}`; });
    const { I, z, lag } = moran(v, NB[sec3]), xe = d3.extent(z), ye = d3.extent(lag);
    F3.x.domain([xe[0] - 0.1, xe[1] + 0.1]); F3.y.domain([Math.min(ye[0], -0.3) - 0.05, Math.max(ye[1], 0.3) + 0.05]);
    F3.g.select('.axis.x').call(d3.axisBottom(F3.x).ticks(5)); F3.g.select('.axis.y').call(d3.axisLeft(F3.y).ticks(5));
    F3.g.select('.gridline').call(d3.axisLeft(F3.y).ticks(5).tickSize(-F3.w).tickFormat(''));
    F3.layer('p').selectAll('circle').data(d3.range(z.length).filter((i) => i % 2 === 0)).join('circle').attr('cx', (i) => F3.x(z[i])).attr('cy', (i) => F3.y(lag[i])).attr('r', 2.6).attr('fill', cat(0)).attr('fill-opacity', 0.35);
    const xs = F3.x.domain();
    F3.layer('f').selectAll('line').data([0]).join('line').attr('x1', F3.x(xs[0])).attr('x2', F3.x(xs[1])).attr('y1', F3.y(I * xs[0])).attr('y2', F3.y(I * xs[1])).attr('stroke', ink()).attr('stroke-width', 2);
    F3.layer('t').selectAll('text').data([0]).join('text').attr('x', F3.w - 4).attr('y', 14).attr('text-anchor', 'end').attr('font-size', 13).attr('fill', ink2()).text(`slope = Moran's I = ${f3(I)}`);
    shufB.textContent = shuf ? 'Restore positions' : 'Shuffle positions';
    const r = SV.res[sec3], k = gi[gene], fq = (q) => (q < 0.001 ? '< 0.001' : f3(q));
    rd3.set({ "Moran's I (live)": f3(I), 'z against permutation null': f1(r.z[k]), 'FDR q': fq(r.q[k]), 'I after removing cell-type means': `${f3(r.I_res[k])} (q ${fq(r.q_res[k])})` });
  }
  paint3();
  const r0 = SV.res[iI], rE = SV.res[iE];
  prose(rb, `<p>EPCAM in ${S.sections[iI]} has I = ${f3(r0.I[gi.EPCAM])} because tumour nests are large contiguous blocks, but after subtracting each cell type's mean it falls to ${f3(r0.I_res[gi.EPCAM])}: the pattern is the position of tumour cells, not a regulated gradient. CXCL9 keeps a smaller I = ${f3(r0.I_res[gi.CXCL9])} (q ${f3(r0.q_res[gi.CXCL9])}) after that adjustment, because it is expressed where CD8 T cells (and their IFN-γ) are nearby. Across the ${nGenes}-gene panel, ${sigRaw[iI]} genes reach q < 0.05 in ${secShort(S.sections[iI])}, but only ${sigRes[iI]} after the cell-type adjustment (${sigRes[iD]} in the desert section).</p>`);

  // ---- Step 4: domains
  const L = DM.lambdas, KEYS = Object.keys(DM.labels), lk = (i) => KEYS.find((k) => +k === L[i]);
  rb = rung(ch, 4, 'Spatial domains',
    `<p>A <b>domain</b> is found by clustering cells on their expression after mixing in their neighbours'. Principal components (PCs) of each cell are combined with the neighbour-averaged PCs, weighted by <b>λ</b> (0 = own expression only, close to 1 = mostly neighbours), and k-means (k = ${DM.k}) groups the result. The simulated regions (tumour nest, margin, stroma, TLS) are the reference for scoring with the <b>adjusted Rand index</b> (ARI; 1 = identical partitions, 0 = chance).</p>`);
  const f4g = figure(rb, 'Increase the neighbour weight λ', 'Left: clusters (data only). Right: simulated regions (truth). Hover a cell to compare. "Play sweep" steps through every λ.');
  let li = 0, sec4 = iI, stopPlay = null;
  const sl4 = slider(f4g.controls, { label: 'neighbour weight λ', min: 0, max: L.length - 1, step: 1, value: 0, fmt: (v) => L[v].toFixed(2), onInput: (v) => { li = v; paint4(); } });
  select(f4g.controls, { label: 'Section', value: String(sec4), options: S.sections.map((s, i) => [String(i), s]), onChange: (v) => { sec4 = +v; paint4(); } });
  const playB = button(f4g.controls, 'Play sweep', () => {
    if (stopPlay) { stopPlay(); stopPlay = null; playB.textContent = 'Play sweep'; return; }
    playB.textContent = 'Stop'; li = 0; sl4.set(0); paint4();
    stopPlay = every(900, () => { li = (li + 1) % L.length; sl4.set(li); paint4(); if (li === L.length - 1) { stopPlay(); stopPlay = null; playB.textContent = 'Play sweep'; } });
  });
  legend(f4g.controls, S.regions.map((r, k) => [r, regionColor(k)]));
  const rd4 = stats(f4g.root);
  const row4 = el('div', { class: 'fig-row' }); f4g.body.append(row4);
  const a4 = el('div'), b4 = el('div'); row4.append(a4, b4);
  const MA = tissueMap(a4, { size: S.size, px: 440, label: 'Clusters (data only)' }), MB = tissueMap(b4, { size: S.size, px: 440, label: 'Simulated regions (truth)' });
  function paint4() {
    const s = S.cells[sec4], lab = DM.labels[lk(li)][sec4];
    MA.draw(s.x, s.y, (i) => regionColor(lab[i]), () => 6); MB.draw(s.x, s.y, (i) => regionColor(s.region[i]), () => 6);
    const hov = (x, y, near) => { const i = near(x, y); return i < 0 ? null : `<b>${CT[s.type[i]]}</b><br>cluster: ${S.regions[lab[i]]}-like<br>truth: ${S.regions[s.region[i]]}`; };
    MA.hover(hov); MB.hover(hov);
    rd4.set({ λ: L[li].toFixed(2), 'ARI vs regions (all sections)': f3(DM.ari[li]), 'ARI vs cell types': f3(DM.ari_type[li]), [`ARI vs regions (${secShort(S.sections[sec4])})`]: f3(DM.ari_section[lk(li)][sec4]) });
  }
  paint4();
  const bestL = d3.maxIndex(DM.ari), bestT = d3.maxIndex(DM.ari_type);
  prose(rb, `<p>Agreement with the simulated regions is modest at every λ: ARI ranges from ${f3(d3.min(DM.ari))} to ${f3(d3.max(DM.ari))}, with the maximum at λ = ${L[bestL].toFixed(2)}. Agreement with the <i>cell-type</i> labels is higher (up to ${f3(d3.max(DM.ari_type))} at λ = ${L[bestT].toFixed(2)}) and collapses at λ = ${L[L.length - 1].toFixed(2)} (${f3(DM.ari_type[L.length - 1])}), where smoothing merges small structures into large ones. The clusters therefore follow the cell types more closely than the region boundaries. This differs from the usual expectation that neighbour smoothing recovers anatomical domains. It is a property of the simulation, not a general statement about real tissue: regions here differ mainly in cell-type proportions, not in distinct expression programmes, so expression alone only partly identifies them.</p>`);

  // ---- Step 5: deconvolution
  const METHODS = [['nnls', 'NNLS on marker genes'], ['poisson', 'Poisson model + platform factor']];
  rb = rung(ch, 5, 'Deconvolution with the single-cell reference',
    `<p>A spot's expression is approximately a weighted sum of cell-type profiles. With profiles from the annotated single-cell data (the reference), the weights are the proportions, found by <b>non-negative least squares</b> (NNLS) or a Poisson model that also learns a per-gene platform factor (the approach of cell2location and RCTD, simplified). Cell types contain different amounts of RNA, so the fitted weights are RNA shares and must be converted to <b>cell shares</b> by dividing by each type's RNA content.</p>`);
  const f5g = figure(rb, 'Estimated against true composition', `Left: each dot is a spot (all ${NS} sections); the diagonal is perfect agreement. Right: mean error per cell type for the current method (right of zero = over-estimated).`);
  let meth = 'poisson', size5 = true, ct5 = CD8;
  const key5 = () => `${meth}_${size5 ? 1 : 0}`;
  select(f5g.controls, { label: 'Method', value: meth, options: METHODS, onChange: (v) => { meth = v; paint5(); } });
  checkbox(f5g.controls, { label: 'convert RNA share to cell share', checked: size5, onChange: (v) => { size5 = v; paint5(); } });
  select(f5g.controls, { label: 'Cell type', value: String(ct5), options: CT.map((c, i) => [String(i), c]), onChange: (v) => { ct5 = +v; paint5(); } });
  const rd5 = stats(f5g.root);
  const row5 = el('div', { class: 'fig-row' }); f5g.body.append(row5);
  const s5w = el('div'), b5w = el('div'); row5.append(s5w, b5w);
  const F5 = frame(s5w, { width: 440, height: 390, x: [0, 1], y: [0, 1], xLabel: 'true fraction', yLabel: 'estimated fraction' });
  const FB = frame(b5w, { width: 460, height: 390, x: [-0.1, 0.1], y: [0, 1], xLabel: 'mean bias (estimate − truth), percentage points', margin: { left: 92 }, grid: false, yScale: d3.scaleBand().padding(0.25) });
  FB.y.domain(CT); FB.g.select('.axis.y').call(d3.axisLeft(FB.y));
  const bmax = Math.max(0.04, d3.max(Object.values(V.metrics), (q) => d3.max(q.bias, Math.abs))) * 1.1;
  FB.x.domain([-bmax, bmax]); FB.g.select('.axis.x').call(d3.axisBottom(FB.x).ticks(5).tickFormat((d) => (d3.format('+.0f')(d * 100))));
  FB.layer('z').append('line').attr('x1', FB.x(0)).attr('x2', FB.x(0)).attr('y1', 0).attr('y2', FB.h).attr('stroke', css('--axis'));
  function paint5() {
    const E = V.est[key5()], m = V.metrics[key5()], mx = Math.max(0.1, d3.max(V.true, (r) => r[ct5]), d3.max(E, (r) => r[ct5])) * 1.05;
    F5.x.domain([0, mx]); F5.y.domain([0, mx]);
    F5.g.select('.axis.x').call(d3.axisBottom(F5.x).ticks(5).tickFormat(d3.format('.0%'))); F5.g.select('.axis.y').call(d3.axisLeft(F5.y).ticks(5).tickFormat(d3.format('.0%')));
    F5.g.select('.gridline').call(d3.axisLeft(F5.y).ticks(5).tickSize(-F5.w).tickFormat(''));
    F5.layer('d').selectAll('line').data([0]).join('line').attr('x1', F5.x(0)).attr('y1', F5.y(0)).attr('x2', F5.x(mx)).attr('y2', F5.y(mx)).attr('stroke', muted()).attr('stroke-dasharray', '4 3');
    F5.layer('p').selectAll('circle').data(d3.range(V.n_spots)).join('circle').attr('cx', (j) => F5.x(V.true[j][ct5])).attr('cy', (j) => F5.y(E[j][ct5])).attr('r', 3.8).attr('fill', ctColor(ct5)).attr('fill-opacity', 0.55).attr('stroke', css('--surface')).attr('stroke-width', 0.8)
      .on('mousemove', (e, j) => tip(e, `${V.sections[V.sec[j]]} · spot (${V.x[j]}, ${V.y[j]})<br>${CT[ct5]}: true ${pct(V.true[j][ct5])}, estimated ${pct(E[j][ct5])}`)).on('mouseleave', untip);
    FB.layer('b').selectAll('rect').data(m.bias).join('rect').attr('y', (d, t) => FB.y(CT[t])).attr('height', FB.y.bandwidth()).attr('x', (d) => FB.x(Math.min(0, d))).attr('width', (d) => Math.abs(FB.x(d) - FB.x(0))).attr('fill', (d, t) => ctColor(t))
      .attr('stroke', (d, t) => (t === ct5 ? css('--ink') : 'none')).attr('stroke-width', 1.5)
      .on('mousemove', (e, d) => { const t = m.bias.indexOf(d); tip(e, `${CT[t]}: bias ${(d * 100).toFixed(1)} points, r = ${f2(m.r[t])}`); }).on('mouseleave', untip);
    rd5.set({ 'RMSE (all types, all spots)': f3(m.rmse), [`r, ${CT[ct5]}`]: f3(m.r[ct5]), [`bias, ${CT[ct5]}`]: `${(m.bias[ct5] * 100).toFixed(1)} points` });
  }
  paint5();
  const mk = (k) => V.metrics[k], worstK = Object.keys(V.metrics).reduce((a, k) => (mk(k).rmse > mk(a).rmse ? k : a)), bestK = Object.keys(V.metrics).reduce((a, k) => (mk(k).rmse < mk(a).rmse ? k : a));
  const lab5 = (k) => `${k.startsWith('poisson') ? 'Poisson' : 'NNLS'}, ${k.endsWith('1') ? 'cell share' : 'RNA share'}`;
  prose(rb, `<p>Across the four combinations the RMSE ranges from ${f3(mk(bestK).rmse)} (${lab5(bestK)}) to ${f3(mk(worstK).rmse)} (${lab5(worstK)}). The conversion from RNA share to cell share helps the Poisson model (${f3(mk('poisson_0').rmse)} → ${f3(mk('poisson_1').rmse)}) but <b>raises</b> the NNLS error (${f3(mk('nnls_0').rmse)} → ${f3(mk('nnls_1').rmse)}), mainly through a tumour bias of ${(mk('nnls_1').bias[TUM] * 100).toFixed(1)} points. The textbook correction is therefore not automatically beneficial: it is only as good as the RNA-content estimates it divides by. CD8 T proportions correlate with the truth at r = ${f2(mk(bestK).r[CD8])} in the best configuration. The reference here is the same simulation that generated the spots, so these accuracies are optimistic for real data.</p>`);

  // ---- Step 6: neighbourhood enrichment
  const NT = CT.length, dvz = diverging(20);
  rb = rung(ch, 6, 'Neighbourhood enrichment: who touches whom',
    `<p>Count the graph edges between each pair of cell types, then compare with the same count after <b>shuffling the cell-type labels</b> (${N.nperm} permutations, positions and graph fixed). The z-score is (observed − mean of shuffles) / sd of shuffles. Positive means the two types are neighbours more often than chance; negative means less often. For the biomarker the relevant entries are CD8 T – tumour and CD8 T – fibroblast.</p>`);
  const f6g = figure(rb, 'Neighbourhood enrichment z-scores', `Left: z for every pair of cell types in one section (warm = more contact than chance, cool = less). Right: CD8 T cells against four cell types in all ${NS} sections. Hover for values.`);
  let sec6 = iE;
  select(f6g.controls, { label: 'Section', value: String(sec6), options: S.sections.map((s, i) => [String(i), s]), onChange: (v) => { sec6 = +v; paint6(); } });
  legend(f6g.controls, [['z ≤ −20', dvz(-20)], ['0', dvz(0)], ['z ≥ +20', dvz(20)]]);
  const rd6 = stats(f6g.root);
  const row6 = el('div', { class: 'fig-row' }); f6g.body.append(row6);
  const h6 = el('div'), b6 = el('div'); row6.append(h6, b6);
  const showT = [TUM, FIB, CD8, 5], allz = N.enrichment.flatMap((e) => showT.map((t) => e.z[CD8][t]));
  const FZ = frame(b6, { width: 480, height: 380, x: [0, 1], y: [Math.min(-3, d3.min(allz)) - 1, Math.max(3, d3.max(allz)) + 1], yLabel: 'z (CD8 T next to …)', xScale: d3.scaleBand().padding(0.2), margin: { bottom: 34 } });
  FZ.x.domain(showT.map((t) => CT[t])); FZ.g.select('.axis.x').call(d3.axisBottom(FZ.x));
  const subB = d3.scaleBand().domain(d3.range(NS)).range([0, FZ.x.bandwidth()]).padding(0.08);
  FZ.layer('z').append('line').attr('x1', 0).attr('x2', FZ.w).attr('y1', FZ.y(0)).attr('y2', FZ.y(0)).attr('stroke', css('--axis'));
  const bars6 = FZ.layer('b').selectAll('rect').data(showT.flatMap((t) => d3.range(NS).map((i) => ({ t, i, z: N.enrichment[i].z[CD8][t] })))).join('rect')
    .attr('x', (d) => FZ.x(CT[d.t]) + subB(d.i)).attr('width', subB.bandwidth()).attr('y', (d) => FZ.y(Math.max(0, d.z))).attr('height', (d) => Math.abs(FZ.y(d.z) - FZ.y(0))).attr('fill', (d) => phenoColor(S.phen[d.i])).attr('rx', 2)
    .on('mousemove', (e, d) => tip(e, `<b>${S.sections[d.i]}</b><br>CD8 T – ${CT[d.t]}: z = ${f1(d.z)}`)).on('mouseleave', untip);
  legend(b6, PHENOS.map((p) => [`${p} (${S.phen.map((q, i) => (q === p ? secShort(S.sections[i]) : null)).filter(Boolean).join(', ')})`, phenoColor(p)]));
  function paint6() {
    h6.innerHTML = '';
    const z = N.enrichment[sec6].z, cell = 33, left = 92, top = 78, w = left + NT * cell + 44, h = top + NT * cell + 6;
    const svg = d3.select(h6).append('svg').attr('viewBox', `0 0 ${w} ${h}`).attr('width', w);
    z.forEach((row, a) => row.forEach((v, b) => {
      svg.append('rect').attr('x', left + b * cell).attr('y', top + a * cell).attr('width', cell - 2).attr('height', cell - 2).attr('rx', 3).attr('fill', dvz(v)).attr('stroke', a === CD8 && (b === TUM || b === FIB) ? css('--ink') : 'none').attr('stroke-width', 2)
        .on('mousemove', (e) => tip(e, `${CT[a]} – ${CT[b]}: z = ${f1(v)}`)).on('mouseleave', untip);
      svg.append('text').attr('x', left + b * cell + cell / 2 - 1).attr('y', top + a * cell + cell / 2 + 4).attr('text-anchor', 'middle').attr('font-size', 10).attr('pointer-events', 'none').attr('fill', Math.abs(v) > 12 ? css('--surface') : css('--ink')).text(Math.round(Math.max(-99, Math.min(99, v))));
    }));
    CT.forEach((c, t) => {
      svg.append('text').attr('x', left - 6).attr('y', top + t * cell + cell / 2 + 4).attr('text-anchor', 'end').attr('font-size', 11).attr('fill', ink2()).attr('font-weight', t === CD8 ? 700 : 400).text(c);
      svg.append('text').attr('transform', `translate(${left + t * cell + cell / 2 + 3},${top - 6}) rotate(-50)`).attr('font-size', 11).attr('fill', ink2()).text(c);
    });
    bars6.attr('opacity', (d) => (d.i === sec6 ? 1 : 0.4)).attr('stroke', (d) => (d.i === sec6 ? css('--ink') : 'none'));
    rd6.set({ section: secShort(S.sections[sec6]), 'CD8 T cells': N.enrichment[sec6].count[CD8], 'z: CD8 T – tumour': f1(z[CD8][TUM]), 'z: CD8 T – fibroblast': f1(z[CD8][FIB]) });
  }
  paint6();
  const zz = (i, t) => N.enrichment[i].z[CD8][t];
  prose(rb, `<p>In the inflamed sections CD8 T cells touch tumour cells more than chance (z = ${S.phen.map((q, i) => (q === 'inflamed' ? f1(zz(i, TUM)) : null)).filter(Boolean).join(' and ')}). In the excluded section they touch fibroblasts (z = ${f1(zz(iE, FIB))}) and avoid tumour cells (z = ${f1(zz(iE, TUM))}). The desert section also shows a negative CD8 T – tumour score (${f1(zz(iD, TUM))}), based on only ${N.enrichment[iD].count[CD8]} CD8 T cells (against ${N.enrichment[iE].count[CD8]} in the excluded section), which shows that a negative z does not by itself distinguish exclusion from absence. Per-section z-scores also grow with the number of cells, so they are not comparable between sections of different size.</p>`);

  // ---- Step 7: the biomarker lab
  const phScore = cohort.map((c) => ({ inflamed: 2, excluded: 1, desert: 0 }[c.ph])), oracle = auc(cohort.map((c) => c.resp), phScore);
  const taskData = (task) => { const ix = cohort.map((c, i) => i).filter((i) => task === 'resp' || cohort[i].ph !== 'desert'); return { ix, y: ix.map((i) => (task === 'resp' ? cohort[i].resp : +(cohort[i].ph === 'inflamed'))) }; };
  const aucOf = (task, k) => { const { ix, y } = taskData(task); return auc(y, ix.map((i) => cohort[i][k])); };
  const bootCI = (task, k) => { const { ix, y } = taskData(task), s = ix.map((i) => cohort[i][k]), r = rng(99), out = []; for (let b = 0; b < 400; b++) { const ii = y.map(() => Math.floor(r() * y.length)), yy = ii.map((i) => y[i]); if (d3.sum(yy) && d3.sum(yy) < yy.length) out.push(auc(yy, ii.map((i) => s[i]))); } out.sort(d3.ascending); return [d3.quantile(out, 0.025), d3.quantile(out, 0.975)]; };
  const rocPts = (y, s) => { const o = d3.range(s.length).sort((a, b) => s[b] - s[a]), P = d3.sum(y), Nn = y.length - P, pts = [{ x: 0, y: 0 }]; let tp = 0, fp = 0; o.forEach((i, k) => { if (y[i]) tp++; else fp++; if (k === o.length - 1 || s[o[k + 1]] !== s[i]) pts.push({ x: fp / Nn, y: tp / P }); }); return pts; };
  rb = rung(ch, 7, 'The biomarker: CD8 inside the tumour or in the stroma',
    `<p>A per-section feature is computed from the cell map. The <b>tumour compartment</b> is defined from the data: a cell belongs to it if at least half of its 10 nearest neighbours are tumour cells. Three quantities follow: <b>CD8 T per 100 cells inside the compartment</b>, the <b>share of CD8 T cells that lie inside</b>, and the <b>infiltration ratio</b> log2((density inside + 0.2) / (density outside + 0.2)). Each is compared with bulk-like features on a cohort of ${B.n} simulated sections (${nMain} main sections plus ${B.n - nMain} smaller ones; ${PHENOS.map((p) => `${cohort.filter((c) => c.ph === p).length} ${p}`).join(', ')}).</p>`);
  const f7g = figure(rb, 'Lab: choose a feature, a task and a cut-off', 'Left: value of the feature per section (outlined dot = responder, larger dot = one of the main sections); the dashed line is the cut-off. Right: ROC curve of the chosen feature with the cut-off marked, and the bulk IFN-γ score for reference. Hover a dot for details.');
  let feat = 'infil', task = 'phen', cut = null;
  select(f7g.controls, { label: 'Feature', value: feat, options: Object.entries(FEAT).map(([k, v]) => [k, `${v.label} (${v.kind})`]), onChange: (v) => { feat = v; cut = null; paint7(true); } });
  select(f7g.controls, { label: 'Task', value: task, options: [['phen', 'inflamed vs excluded (desert omitted)'], ['resp', 'responders vs non-responders (all sections)']], onChange: (v) => { task = v; paint7(false); } });
  const cutSl = slider(f7g.controls, { label: 'cut-off', min: 0, max: 1, step: 0.01, value: 0.5, fmt: (v) => v.toFixed(2), onInput: (v) => { cut = v; paint7(false); } });
  const rd7 = stats(f7g.root);
  const row7 = el('div', { class: 'fig-row' }); f7g.body.append(row7);
  const sw = el('div'), rw = el('div'); row7.append(sw, rw);
  const FS = frame(sw, { width: 450, height: 400, x: [0, 1], y: [0, 1], xScale: d3.scaleBand().padding(0.3), margin: { left: 56 } });
  FS.x.domain(PHENOS); FS.g.select('.axis.x').call(d3.axisBottom(FS.x));
  const yLab = FS.g.append('text').attr('class', 'ax-label').attr('transform', 'rotate(-90)').attr('x', -FS.h / 2).attr('y', -42).attr('text-anchor', 'middle');
  const FR = frame(rw, { width: 450, height: 400, x: [0, 1], y: [0, 1], xLabel: 'false-positive rate (1 − specificity)', yLabel: 'true-positive rate (sensitivity)' });
  FR.layer('d').append('line').attr('x1', FR.x(0)).attr('y1', FR.y(0)).attr('x2', FR.x(1)).attr('y2', FR.y(1)).attr('stroke', muted()).attr('stroke-dasharray', '4 3');
  const leg7 = el('div'); rw.append(leg7);
  const jr = rng(11), jx = cohort.map(() => (jr() - 0.5) * 0.6);
  function paint7(resetCut) {
    const v = cohort.map((c) => c[feat]), ex = d3.extent(v);
    if (resetCut || cut === null) { cut = d3.median(v); cutSl.input.min = ex[0]; cutSl.input.max = ex[1]; cutSl.input.step = (ex[1] - ex[0]) / 100; cutSl.set(cut); }
    FS.y.domain(ex).nice(); FS.g.select('.axis.y').call(d3.axisLeft(FS.y).ticks(6)); FS.g.select('.gridline').call(d3.axisLeft(FS.y).ticks(6).tickSize(-FS.w).tickFormat(''));
    yLab.text(FEAT[feat].short);
    leg7.innerHTML = ''; legend(leg7, [[`chosen feature (${FEAT[feat].short})`, featCol(feat)], ['bulk IFN-γ', featCol('ifn_bulk')], ['chance', muted()]]);
    FS.layer('p').selectAll('circle').data(cohort).join('circle').attr('cx', (c, i) => FS.x(c.ph) + FS.x.bandwidth() / 2 + jx[i] * FS.x.bandwidth()).attr('cy', (c) => FS.y(c[feat])).attr('r', (c) => (c.main ? 7.5 : 5))
      .attr('fill', (c) => phenoColor(c.ph)).attr('fill-opacity', (c) => (c.resp ? 1 : 0.45)).attr('stroke', (c) => (c.resp ? css('--ink') : css('--surface'))).attr('stroke-width', (c) => (c.resp ? 1.8 : 1.2))
      .on('mousemove', (e, c) => tip(e, `<b>${c.name}</b> · ${c.ph} · ${c.resp ? 'responder' : 'non-responder'}<br>${Object.keys(FEAT).map((k) => `${FEAT[k].short}: ${f2(c[k])}`).join('<br>')}`)).on('mouseleave', untip);
    FS.layer('c').selectAll('line').data([cut]).join('line').attr('x1', 0).attr('x2', FS.w).attr('y1', FS.y(cut)).attr('y2', FS.y(cut)).attr('stroke', ink()).attr('stroke-width', 1.5).attr('stroke-dasharray', '5 4');
    FS.layer('ct').selectAll('text').data([cut]).join('text').attr('x', FS.w - 4).attr('y', FS.y(cut) - 5).attr('text-anchor', 'end').attr('font-size', 11.5).attr('fill', ink2()).text('cut-off');
    const { ix, y } = taskData(task), sc = ix.map((i) => cohort[i][feat]), P = d3.sum(y), Nn = y.length - P;
    const tp = d3.sum(y.map((yy, k) => (yy && sc[k] >= cut ? 1 : 0))), fp = d3.sum(y.map((yy, k) => (!yy && sc[k] >= cut ? 1 : 0)));
    const curves = [{ k: 'ifn_bulk', pts: rocPts(y, ix.map((i) => cohort[i].ifn_bulk)) }, { k: feat, pts: rocPts(y, sc) }];
    FR.layer('c').selectAll('path').data(curves).join('path').attr('fill', 'none').attr('stroke', (d) => featCol(d.k)).attr('stroke-width', (d, i) => (i ? 3 : 1.8)).attr('opacity', (d, i) => (i ? 1 : 0.8))
      .attr('d', (d) => d3.line().curve(d3.curveStepAfter).x((p) => FR.x(p.x)).y((p) => FR.y(p.y))(d.pts));
    FR.layer('m').selectAll('circle').data([0]).join('circle').attr('cx', FR.x(fp / Nn)).attr('cy', FR.y(tp / P)).attr('r', 6.5).attr('fill', ink()).attr('stroke', css('--surface')).attr('stroke-width', 2);
    FR.layer('t').selectAll('text').data([0]).join('text').attr('x', FR.x(fp / Nn) + 10).attr('y', FR.y(tp / P) + 16).attr('font-size', 11.5).attr('fill', ink2()).text('current cut-off');
    const a = aucOf(task, feat), ci = bootCI(task, feat), aB = aucOf(task, 'ifn_bulk'), cB = bootCI(task, 'ifn_bulk');
    rd7.set({ task: task === 'phen' ? `${y.length} sections: ${P} inflamed, ${Nn} excluded` : `${y.length} sections: ${P} responders`, [`AUC, ${FEAT[feat].short}`]: `${f3(a)} [${f2(ci[0])}, ${f2(ci[1])}]`, 'AUC, bulk IFN-γ': `${f3(aB)} [${f2(cB[0])}, ${f2(cB[1])}]`, 'sensitivity at cut-off': pct(tp / P), 'specificity at cut-off': pct(1 - fp / Nn) });
  }
  paint7(true);
  const gr = (p) => cohort.filter((c) => c.ph === p), nR = (p) => gr(p).filter((c) => c.resp).length;
  const bestR = Object.keys(FEAT).reduce((a, k) => (aucOf('resp', k) > aucOf('resp', a) ? k : a));
  const oppo = cohort.filter((c) => c.ph === 'excluded').filter((c) => c.ifn_bulk > d3.min(gr('inflamed'), (q) => q.ifn_bulk)).length;
  table(rb, [
    { t: 'Feature <b>infiltration ratio</b>, task inflamed vs excluded', s: `AUC ${f3(aucOf('phen', 'infil'))}; bulk IFN-γ ${f3(aucOf('phen', 'ifn_bulk'))}; ${oppo} of ${gr('excluded').length} excluded sections score above the lowest inflamed section on bulk IFN-γ.`, m: 'The spatial feature separates the two phenotypes completely, but excluded tumours were simulated by moving CD8 T cells out of the nests. The feature measures the generating rule, so an AUC of 1 is expected and is not independent evidence.' },
    { t: 'Switch the task to <b>responders</b>', s: `AUC: CD8 fraction ${f3(aucOf('resp', 'cd8_frac'))}, CD8 in tumour ${f3(aucOf('resp', 'd_tumour'))}, bulk IFN-γ ${f3(aucOf('resp', 'ifn_bulk'))}, infiltration ratio ${f3(aucOf('resp', 'infil'))}, share inside ${f3(aucOf('resp', 'frac_inside'))}.`, m: `The ranking differs from the textbook expectation: the ${FEAT[bestR].short} (${FEAT[bestR].kind}) scores highest. Response was assigned from phenotype alone (inflamed ${nR('inflamed')}/${gr('inflamed').length}, excluded ${nR('excluded')}/${gr('excluded').length}, desert ${nR('desert')}/${gr('desert').length}), so even the true phenotype gives AUC ${f3(oracle)}: the phenotype is the ceiling for any feature in this simulation, and the CD8 fraction reaches ${f3(aucOf('resp', 'cd8_frac'))} without any spatial information.` },
    { t: 'Read the bootstrap intervals in the readout', s: `For responders, bulk IFN-γ is ${f3(aucOf('resp', 'ifn_bulk'))} [${f2(bootCI('resp', 'ifn_bulk')[0])}, ${f2(bootCI('resp', 'ifn_bulk')[1])}] and CD8 in tumour ${f3(aucOf('resp', 'd_tumour'))} [${f2(bootCI('resp', 'd_tumour')[0])}, ${f2(bootCI('resp', 'd_tumour')[1])}].`, m: `With ${B.n_resp} responders among ${B.n} sections the intervals overlap widely, so the ordering of the features is not reliable.` },
    { t: 'Move the <b>cut-off</b>', s: 'The dot moves along the ROC curve; sensitivity falls as specificity rises.', m: 'A cut-off turns a ranking into a decision. The AUC summarises every cut-off; the clinical choice depends on the cost of each error.' },
  ], [['t', 'Try'], ['s', 'What you see'], ['m', 'What it means']]);
  quiz(rb, `The infiltration ratio gives AUC ${f3(aucOf('phen', 'infil'))} for inflamed vs excluded. Does that show it will predict response in real patients?`,
    `No. Here the phenotype labels were produced by the same rule the feature measures, and response was assigned from the phenotype with probabilities below 1 (${PHENOS.map((p) => `${p} ${nR(p)}/${gr(p).length}`).join(', ')}). Evidence for clinical use requires an independent cohort with outcome data and a definition fixed beforehand.`);

  // =====================================================================================
  // 5. PITFALLS
  // =====================================================================================
  ch = chapter(main, 'pitfalls', 'Pitfalls', 'Common problems, how to recognise them, and how to address them.');
  cards(ch, [
    { tag: 'Symptom: "this spot expresses CD8A, so T cells are here"', title: 'Spots are mixtures', body: `Every spot contains several cell types (${V.cells_per_spot.min}–${V.cells_per_spot.max} cells here), and CD8 T cells are rarely the majority. <b>Fix:</b> deconvolve first, or use single-cell-resolution data; state what a spot value represents.` },
    { tag: 'Symptom: results change with the graph', title: 'Graph choice', body: `Edge cap, k and radius change neighbour counts (${f2(S.cells[0].mean_deg)} per cell at 35 µm here) and therefore every statistic built on them. <b>Fix:</b> fix the graph before looking at results, report it, and check that conclusions survive a reasonable change.` },
    { tag: 'Symptom: hundreds of spatially variable genes', title: 'Composition mistaken for regulation', body: `${sigRaw[iI]} of ${nGenes} genes pass q < 0.05 in ${secShort(S.sections[iI])}, ${sigRes[iI]} after removing cell-type means. Most hits track where cell types are. <b>Fix:</b> test within cell types, and rank by I rather than by p, which is near its floor for many genes.` },
    { tag: 'Symptom: clusters that do not match anatomy', title: 'Domains follow cell types', body: `Maximum ARI against the regions is ${f3(d3.max(DM.ari))} here, below the agreement with cell types. <b>Fix:</b> validate domains against independent annotation, and report the smoothing weight and k.` },
    { tag: 'Symptom: proportions that do not sum correctly or are biased', title: 'Deconvolution reference mismatch', body: `Bias reached ${(d3.max(Object.values(V.metrics), (q) => d3.max(q.bias, Math.abs)) * 100).toFixed(1)} points for a single type even with a matched reference. <b>Fix:</b> use a reference from the same tissue and platform, and check per-type bias.` },
    { tag: 'Symptom: AUC of 1.0', title: 'Circularity', body: 'A feature that re-measures how a phenotype was defined, or was simulated, will separate it perfectly. <b>Fix:</b> evaluate against outcome in an independent cohort, with the definition fixed in advance.' },
    { tag: 'Symptom: very wide intervals', title: 'Few sections, few events', body: `${B.n} sections and ${B.n_resp} responders cannot rank features whose AUCs differ by about 0.05. Spatial statistics describe a section, not a population. <b>Fix:</b> report bootstrap intervals and increase the number of patients, not of cells.` },
    { tag: 'Symptom: result depends on the field of view', title: 'Sampling', body: 'A biopsy covers a few square millimetres of a heterogeneous tumour, and the invasive margin may be absent from it. <b>Fix:</b> record the sampled area and compare features across multiple regions where available.' },
  ]);

  // =====================================================================================
  // 6. RECAP
  // =====================================================================================
  ch = chapter(main, 'recap', 'Recap', 'Key points.');
  recap(ch, [
    'Spatial data adds <b>coordinates</b> to every measurement. Spots are mixtures of cells; single-cell-resolution platforms are not.',
    'Every spatial statistic is defined on a <b>neighbour graph</b>. Its construction (edge cap, k) is a modelling choice and should be reported.',
    "<b>Moran's I</b> measures similarity between neighbours, and a permutation test gives its significance. Most hits follow cell-type layout, so test within cell types.",
    'Spatial <b>domains</b> from neighbour-smoothed clustering follow cell types more closely than anatomical regions in this simulation.',
    '<b>Deconvolution</b> recovers spot composition from a single-cell reference; the RNA-to-cell share correction can help or hurt depending on the method.',
    '<b>Neighbourhood enrichment</b> compares observed edges between cell types with shuffled labels: CD8 T – tumour contact distinguishes inflamed from excluded sections.',
    'A <b>per-section feature</b> (CD8 inside versus outside the tumour compartment) separates inflamed from excluded tumours, but here that is near-circular because the phenotypes were simulated that way.',
    `For response, the ${FEAT[bestR].short} ranked highest (AUC ${f3(aucOf('resp', bestR))}) and the intervals overlap. Claims need an independent cohort.`,
  ]);
  prose(ch, '<h3>In code</h3><p>Squidpy (Python) and Seurat/spacexr (R) cover the graph, autocorrelation, enrichment and deconvolution steps used above.</p>');
  code(ch, {
    Python: `import squidpy as sq

# 1) neighbour graph (Delaunay; use coord_type="grid" for Visium)
sq.gr.spatial_neighbors(adata, coord_type="generic", delaunay=True, radius=(0, 35))

# 2) Moran's I with a permutation null, per gene
sq.gr.spatial_autocorr(adata, mode="moran", n_perms=199, n_jobs=4)
print(adata.uns["moranI"].sort_values("I", ascending=False).head())

# 3) neighbourhood enrichment between annotated cell types
sq.gr.nhood_enrichment(adata, cluster_key="cell_type")
sq.pl.nhood_enrichment(adata, cluster_key="cell_type")

# 4) per-section feature: CD8 T per 100 cells in the tumour compartment
W = adata.obsp["spatial_connectivities"]
is_tum = (adata.obs.cell_type == "Tumour").to_numpy(float)
nbr_tumour = W @ is_tum
deg = W.sum(1).A1
in_tumour = (nbr_tumour / deg) >= 0.5
density_in = 100 * ((adata.obs.cell_type == "CD8 T") & in_tumour).sum() / in_tumour.sum()`,
    R: `library(Seurat); library(spacexr)

# Moran's I per gene on a spatial Seurat object
obj <- FindSpatiallyVariableFeatures(obj, selection.method = "moransi",
                                     features = rownames(obj), x.cuts = 100, y.cuts = 100)
head(SpatiallyVariableFeatures(obj, selection.method = "moransi"))

# Deconvolution of spots with an annotated single-cell reference (RCTD)
reference <- Reference(counts_sc, cell_types, nUMI_sc)
puck <- SpatialRNA(coords, counts_spots, nUMI_spots)
rctd <- create.RCTD(puck, reference, max_cores = 4)
rctd <- run.RCTD(rctd, doublet_mode = "full")
weights <- normalize_weights(rctd@results$weights)   # proportions per spot`,
  });
  seeAlso(main, [['omics', 'sc-clustering', 'Single-cell clustering'], ['omics', 'bulk-classifier', 'Response classifier'], ['omics', 'bulk-survival', 'Survival'], ['concepts', 'logistic-regression', 'Logistic regression']]);
}

// ---------------------------------------------------------------------------------------------
/** Pipeline diagram for the page: measurements → graph → statistics → feature, with numbers from the data. */
function pipelineSvg(S, V, NC) {
  const wrap = el('div', { class: 'figure' });
  wrap.append(el('div', { class: 'fig-title' }, 'From measurements to a spatial biomarker'));
  const svg = d3.select(wrap).append('svg').attr('viewBox', '0 0 980 210').attr('width', 980).attr('role', 'img').attr('aria-label', 'measurements, neighbour graph, statistics, per-section feature');
  const boxes = [
    { t: '1 · Measurements', l: [`${NC.toLocaleString()} cells (${S.sections.length} sections)`, `${S.n_panel} panel genes`, `${V.n_spots} spots, ${V.spot_diam} µm`, 'each with x, y'] },
    { t: '2 · Neighbour graph', l: ['Delaunay, edges ≤ 35 µm', `${(S.cells[0].mean_deg).toFixed(2)} neighbours per cell`, 'defines "near" for', 'every later step'] },
    { t: '3 · Spatial statistics', l: ["Moran's I + permutations", 'domains (clustering)', 'deconvolution of spots', 'neighbourhood enrichment'] },
    { t: '4 · One number per section', l: ['CD8 T inside the tumour', 'versus in the stroma', '→ inflamed / excluded', '→ association with response'] },
  ];
  const BW = 212, gap = 36, H = 180;
  boxes.forEach((b, i) => {
    const x0 = 10 + i * (BW + gap), g = svg.append('g').attr('transform', `translate(${x0},14)`);
    g.append('rect').attr('width', BW).attr('height', H).attr('rx', 10).attr('fill', css('--surface-2')).attr('stroke', css('--axis'));
    g.append('text').attr('x', 12).attr('y', 26).attr('font-weight', 700).attr('font-size', 13.5).attr('fill', ink()).text(b.t);
    b.l.forEach((l, j) => g.append('text').attr('x', 14).attr('y', 62 + j * 26).attr('font-size', 13).attr('fill', ink2()).text(l));
    if (i < boxes.length - 1) { const ax = x0 + BW + 6, ay = 14 + H / 2; svg.append('path').attr('d', `M${ax},${ay} L${ax + gap - 14},${ay} M${ax + gap - 20},${ay - 6} L${ax + gap - 14},${ay} L${ax + gap - 20},${ay + 6}`).attr('fill', 'none').attr('stroke', ink2()).attr('stroke-width', 2); }
  });
  return wrap;
}
