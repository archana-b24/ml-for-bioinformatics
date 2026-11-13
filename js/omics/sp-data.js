// Spatial step 1: what spatial data looks like (single cells vs Visium-like spots) and how a neighbour graph is built.
// Data: data/omics/sp_data.json (cells, types, regions, 19 genes) + sp_deconvolution.json (spot centres, true composition).
import { header, figure, select, slider, stats, section, steps, callout, math, legend, seeAlso, el, tip, untip } from '../lib/ui.js';
import { css, ink2 } from '../lib/plot.js';
import { stepCard, omicsData } from './_shared.js';
import { tissueMap, ctColor, regionColor, exprScale, pct, fmt } from './_spatial.js';
const d3 = window.d3;

export default async function render(main) {
  header(main, 'Spatial data & neighbourhoods',
    'Bulk RNA-seq averages a tumour into one number per gene; single-cell RNA-seq keeps the cells but throws away where they were. Spatial transcriptomics keeps the <b>x, y position of every measurement</b>, and the first thing every spatial method does is turn positions into a <b>neighbour graph</b>: who touches whom.');
  const card = stepCard(main, {
    stage: 'Spatial · step 1 of 6',
    question: 'Bulk found T-cell genes in both inflamed and excluded tumours. <b>Where</b> are those CD8 T cells: inside the tumour nests, or held outside them?',
    data: '…',
    method: 'Represent each section as cells with coordinates, then connect neighbours (Delaunay triangulation, k nearest neighbours or a fixed radius).',
    verdict: 'stats', why: ' Building the graph is geometry, not learning. But every later step (Moran\'s I, domain smoothing, niche tests) is defined on this graph, so its choices matter.',
  });

  const D = await omicsData(main, 'sp_data.json');
  const V = await omicsData(main, 'sp_deconvolution.json');
  if (!D || !V) return;
  const NS = D.sections.length;
  const S = D.cells;
  const nTot = d3.sum(S, (s) => s.n);
  card.querySelectorAll('.step-row')[1].querySelector('span').innerHTML =
    `${NS} simulated tumour sections (${D.size}×${D.size} µm each, ${d3.min(S, (s) => s.n) === d3.max(S, (s) => s.n) ? d3.min(S, (s) => s.n).toLocaleString() + ' cells each' : d3.min(S, (s) => s.n).toLocaleString() + '–' + d3.max(S, (s) => s.n).toLocaleString() + ' cells'}, ${nTot.toLocaleString()} in total), imaged at single-cell resolution with a ${D.n_panel}-gene panel (Xenium/CosMx-like), plus Visium-like 55 µm spots made from the same cells.`;

  const CD8 = 1;
  const secStats = S.map((s) => {
    const n8 = s.type.filter((t) => t === CD8).length;
    const in8 = s.type.filter((t, i) => t === CD8 && s.region[i] === 0).length;
    const comp = d3.range(8).map((t) => s.type.filter((u) => u === t).length / s.n);
    return { n8, in8, f8: n8 / s.n, inside: n8 ? in8 / n8 : 0, comp, medLib: d3.median(s.lib) };
  });

  // ------------------------------------------------------------------ figure 1: the sections
  const fig1 = figure(main, 'Four tumour sections, one dot per cell',
    'Each dot is a segmented cell, coloured by its annotated type (or by a gene\'s counts). Regions are the simulation\'s ground truth, which a real experiment never gives you. Hover a cell for details.');
  let sec = 0, colorBy = 'type';
  select(fig1.controls, { label: 'Section', value: '0', options: D.sections.map((s, i) => [String(i), s]), onChange: (v) => { sec = +v; paint1(); } });
  select(fig1.controls, { label: 'Colour by', value: 'type', options: [['type', 'cell type'], ['region', 'region (simulated truth)'], ...D.genes.map((g) => [g, `gene: ${g}`])], onChange: (v) => { colorBy = v; paint1(); } });
  const leg1 = el('div'); fig1.controls.append(leg1);
  const rd1 = stats(fig1.root);
  const row1 = el('div', { class: 'fig-row' }); fig1.body.append(row1);
  const mapW = el('div'), barW = el('div'); row1.append(mapW, barW);
  const M1 = tissueMap(mapW, { size: D.size, px: 460 });

  // composition bars: what a bulk sample of each section would "see"
  const bw = 440, bh = 300, bm = { l: 120, r: 12, t: 26, b: 34 };
  const svgB = d3.select(barW).append('svg').attr('viewBox', `0 0 ${bw} ${bh}`).attr('width', bw);
  svgB.append('text').attr('x', bm.l).attr('y', 14).attr('font-size', 12.5).attr('font-weight', 600).attr('fill', ink2()).text('Cell-type mix per section (what bulk sees)');
  const xb = d3.scaleLinear().domain([0, 1]).range([bm.l, bw - bm.r]);
  const yb = d3.scaleBand().domain(d3.range(NS)).range([bm.t, bh - bm.b]).padding(0.28);
  svgB.append('g').attr('class', 'axis x').attr('transform', `translate(0,${bh - bm.b})`).call(d3.axisBottom(xb).ticks(5).tickFormat(d3.format('.0%')));
  const rowsB = svgB.selectAll('g.r').data(secStats).join('g').attr('class', 'r');
  rowsB.each(function (d, i) {
    let acc = 0;
    const g = d3.select(this);
    d.comp.forEach((f, t) => {
      g.append('rect').attr('x', xb(acc)).attr('width', Math.max(0, xb(acc + f) - xb(acc))).attr('y', yb(i)).attr('height', yb.bandwidth()).attr('fill', ctColor(t))
        .on('mousemove', (e) => tip(e, `<b>${D.sections[i]}</b><br>${D.cell_types[t]}: ${pct(f, 1)} of cells`)).on('mouseleave', untip);
      acc += f;
    });
  });
  const labB = svgB.selectAll('text.sl').data(D.sections).join('text').attr('class', 'sl').attr('x', bm.l - 6).attr('y', (d, i) => yb(i) + yb.bandwidth() / 2 + 4)
    .attr('text-anchor', 'end').attr('font-size', 12).text((d) => d);

  function paint1() {
    const s = S[sec];
    let colorFn;
    leg1.innerHTML = '';
    if (colorBy === 'type') { colorFn = (i) => ctColor(s.type[i]); legend(leg1, D.cell_types.map((c, t) => [c, ctColor(t)])); }
    else if (colorBy === 'region') { colorFn = (i) => regionColor(s.region[i]); legend(leg1, D.regions.map((r, k) => [r, regionColor(k)])); }
    else {
      const v = s.expr[colorBy], sc = exprScale(v);
      colorFn = (i) => sc.color(v[i]) || css('--border');
      legend(leg1, [[`${colorBy} = 0 counts`, css('--border')], ['low', sc.color(1)], [`high (log1p ≥ ${sc.hi.toFixed(1)})`, sc.color(1e6)]]);
    }
    M1.draw(s.x, s.y, colorFn, (i) => (s.type[i] === 0 ? 6.5 : 5.5));
    M1.hover((x, y, near) => {
      const i = near(x, y); if (i < 0) return null;
      const g = colorBy !== 'type' && colorBy !== 'region' ? `<br>${colorBy}: ${s.expr[colorBy][i]} counts` : '';
      return `<b>${D.cell_types[s.type[i]]}</b> · ${D.regions[s.region[i]]}<br>x ${s.x[i]}, y ${s.y[i]} µm<br>${s.lib[i]} transcripts in panel${g}`;
    });
    labB.attr('font-weight', (d, i) => (i === sec ? 700 : 400)).attr('fill', (d, i) => (i === sec ? css('--ink') : ink2()));
    rowsB.attr('opacity', (d, i) => (i === sec ? 1 : 0.45));
    const st = secStats[sec];
    rd1.set({ cells: s.n.toLocaleString(), 'panel genes': D.n_panel, 'median transcripts / cell': st.medLib, 'CD8 T cells': `${st.n8} (${pct(st.f8, 1)})`, 'CD8 T inside tumour nests': pct(st.inside) });
  }
  paint1();

  // ------------------------------------------------------------------ figure 2: spots vs cells
  const fig2 = figure(main, 'Spots versus single cells',
    `The same tissue as a Visium-like array: ${V.spot_diam} µm spots on a ${V.pitch} µm grid. Each spot sums the RNA of every cell under it, so one measurement is a <b>mixture</b>. Hover a spot to see what is inside.`);
  let sec2 = 0, view2 = 'spots';
  select(fig2.controls, { label: 'Section', value: '0', options: D.sections.map((s, i) => [String(i), s]), onChange: (v) => { sec2 = +v; paint2(); } });
  select(fig2.controls, { label: 'Show', value: 'spots', options: [['spots', 'spots, coloured by majority type'], ['cd8', 'spots, coloured by CD8 T fraction'], ['cells', 'cells only']], onChange: (v) => { view2 = v; paint2(); } });
  const leg2 = el('div'); fig2.controls.append(leg2);
  const rd2 = stats(fig2.root);
  const row2 = el('div', { class: 'fig-row' }); fig2.body.append(row2);
  const m2W = el('div'); row2.append(m2W);
  const M2 = tissueMap(m2W, { size: D.size, px: 460 });
  const note2 = el('div', { class: 'prose', style: 'font-size:14px;' }); row2.append(note2);
  const spIdx = (s) => d3.range(V.n_spots).filter((j) => V.sec[j] === s);

  function paint2() {
    const s = S[sec2], ids = spIdx(sec2);
    const seq = d3.interpolateRgb(css('--surface-2'), ctColor(CD8));
    const f8max = d3.max(ids, (j) => V.true[j][CD8]) || 1;
    leg2.innerHTML = '';
    if (view2 === 'cd8') legend(leg2, [['no CD8 T', seq(0)], [`CD8 T ${pct(f8max / 2)}`, seq(0.5)], [`CD8 T ${pct(f8max)}`, seq(1)]]);
    else legend(leg2, D.cell_types.map((c, t) => [c, ctColor(t)]));
    M2.overlay(view2 === 'cells' ? null : (ctx, k) => {
      ids.forEach((j) => {
        const fr = V.true[j];
        const top = d3.maxIndex(fr);
        ctx.globalAlpha = 0.88;
        ctx.fillStyle = view2 === 'cd8' ? seq(fr[CD8] / f8max) : ctColor(top);
        ctx.beginPath(); ctx.arc(V.x[j] * k, V.y[j] * k, (V.spot_diam / 2) * k, 0, 6.2832); ctx.fill();
        ctx.globalAlpha = 1; ctx.strokeStyle = css('--surface'); ctx.lineWidth = 1.5; ctx.stroke();
      });
    });
    M2.draw(s.x, s.y, (i) => ctColor(s.type[i]), () => 5, () => (view2 === 'cells' ? 1 : 0.28));
    M2.hover((x, y) => {
      if (view2 === 'cells') return null;
      const j = ids.find((q) => (V.x[q] - x) ** 2 + (V.y[q] - y) ** 2 < (V.spot_diam / 2) ** 2);
      if (j == null) return null;
      const parts = V.true[j].map((f, t) => [f, t]).filter((p) => p[0] > 0).sort((a, b) => b[0] - a[0]);
      return `<b>Spot (${V.x[j]}, ${V.y[j]})</b>: ${V.n_cells[j]} cells<br>` + parts.map(([f, t]) => `${D.cell_types[t]} ${pct(f)}`).join('<br>');
    });
    const ncell = ids.map((j) => V.n_cells[j]);
    const pure = ids.filter((j) => d3.max(V.true[j]) >= 0.999).length;
    const mixed = ids.filter((j) => d3.max(V.true[j]) < 0.5).length;
    const with8 = ids.filter((j) => V.true[j][CD8] > 0);
    const cd8Pure = with8.filter((j) => V.true[j][CD8] >= 0.5).length;
    rd2.set({ spots: ids.length, 'cells per spot (median)': d3.median(ncell), 'single-type spots': pct(pure / ids.length), 'spots with no majority type': pct(mixed / ids.length), 'spots containing a CD8 T cell where it is the majority': `${cd8Pure} of ${with8.length}` });
    note2.innerHTML = `<p>Across all ${NS} sections there are <b>${V.n_spots.toLocaleString()} spots</b>, holding ${V.cells_per_spot.min}–${V.cells_per_spot.max} cells (median ${V.cells_per_spot.median}). Spots also miss the cells in the gaps between them: they cover only the circles you see.</p>
      <p>A CD8 T cell is never alone in a spot: in ${D.sections[sec2]}, ${cd8Pure ? 'only ' + cd8Pure : 'none'} of the ${with8.length} spots that contain one ${cd8Pure === 1 ? 'has' : 'have'} CD8 T cells as the majority. So "this spot expresses <i>CD8A</i>" never directly means "a T cell is here touching tumour". Recovering the mixture is the job of <a href="#/omics/sp-deconvolution">spot deconvolution</a>; single-cell platforms avoid it but measure fewer genes (here ${D.n_panel}, versus the whole transcriptome on Visium).</p>`;
  }
  paint2();

  // ------------------------------------------------------------------ figure 3: neighbour graph
  const fig3 = figure(main, 'Building the neighbour graph',
    'Click the left map to move the 220 µm window. The right panel draws every edge of the chosen graph inside the window. Edges between a CD8 T cell and a tumour cell are drawn thick.');
  let sec3 = 0, gtype = 'delaunay', par = { delaunay: 35, knn: 6, radius: 30 }, win = null;
  const PARAM = { delaunay: { label: 'max edge length (µm)', min: 20, max: 80, step: 5 }, knn: { label: 'k neighbours', min: 2, max: 20, step: 1 }, radius: { label: 'radius (µm)', min: 15, max: 60, step: 5 } };
  select(fig3.controls, { label: 'Section', value: '0', options: D.sections.map((s, i) => [String(i), s]), onChange: (v) => { sec3 = +v; win = null; build(); } });
  select(fig3.controls, { label: 'Graph', value: 'delaunay', options: [['delaunay', 'Delaunay triangulation'], ['knn', 'k nearest neighbours'], ['radius', 'fixed radius']], onChange: (v) => { gtype = v; resetSlider(); build(); } });
  const slotS = el('span'); fig3.controls.append(slotS);
  let sl;
  function resetSlider() {
    slotS.innerHTML = '';
    const p = PARAM[gtype];
    sl = slider(slotS, { ...p, value: par[gtype], onInput: (v) => { par[gtype] = v; build(); } });
  }
  resetSlider();
  const rd3 = stats(fig3.root);
  const row3 = el('div', { class: 'fig-row' }); fig3.body.append(row3);
  const a3 = el('div'), b3 = el('div'); row3.append(a3, b3);
  const M3a = tissueMap(a3, { size: D.size, px: 400, label: 'Whole section (click to move window)' });
  const WIN = 220;
  const M3b = tissueMap(b3, { size: WIN, px: 460, label: 'Zoomed window with graph edges', scaleBar: 50 });
  M3a.canvas.addEventListener('click', (e) => {
    const r = M3a.canvas.getBoundingClientRect();
    win = [Math.max(0, Math.min(D.size - WIN, (e.clientX - r.left) / r.width * D.size - WIN / 2)), Math.max(0, Math.min(D.size - WIN, (e.clientY - r.top) / r.height * D.size - WIN / 2))];
    paint3();
  });

  let E = [];
  function grid(s, cellSz) {
    const m = new Map();
    s.x.forEach((x, i) => { const key = Math.floor(x / cellSz) + ',' + Math.floor(s.y[i] / cellSz); (m.get(key) || m.set(key, []).get(key)).push(i); });
    return m;
  }
  function edgesFor(s, gtype, par) {
    const n = s.n, e = [];
    if (gtype === 'delaunay') {
      const del = d3.Delaunay.from(d3.range(n), (i) => s.x[i], (i) => s.y[i]);
      const t = del.triangles, seen = new Set();
      for (let q = 0; q < t.length; q += 3) for (const [a, b] of [[t[q], t[q + 1]], [t[q + 1], t[q + 2]], [t[q], t[q + 2]]]) {
        const lo = Math.min(a, b), hi = Math.max(a, b), key = lo * n + hi;
        if (seen.has(key)) continue; seen.add(key);
        if (Math.hypot(s.x[a] - s.x[b], s.y[a] - s.y[b]) <= par.delaunay) e.push([lo, hi]);
      }
    } else {
      const R = gtype === 'radius' ? par.radius : 60, gm = grid(s, R), seen = new Set();
      for (let i = 0; i < n; i++) {
        const gx = Math.floor(s.x[i] / R), gy = Math.floor(s.y[i] / R), cand = [];
        for (let a = -1; a <= 1; a++) for (let b = -1; b <= 1; b++) for (const j of gm.get((gx + a) + ',' + (gy + b)) || []) {
          if (j === i) continue; const d = Math.hypot(s.x[i] - s.x[j], s.y[i] - s.y[j]); if (gtype === 'knn' || d <= R) cand.push([d, j]);
        }
        const use = gtype === 'knn' ? cand.sort((p, q) => p[0] - q[0]).slice(0, par.knn) : cand;
        for (const [, j] of use) { const lo = Math.min(i, j), hi = Math.max(i, j), key = lo * n + hi; if (!seen.has(key)) { seen.add(key); e.push([lo, hi]); } }
      }
    }
    return e;
  }
  function build() {
    const e = edgesFor(S[sec3], gtype, par);
    E = e;
    if (!win) {
      // default window: centred on the CD8 T cell whose 100 µm surroundings are the most even tumour / non-tumour mix
      const s = S[sec3];
      let best = [D.size / 2, D.size / 2], bs = -1;
      s.type.forEach((t, i) => {
        if (t !== CD8) return;
        let a = 0, b = 0;
        for (let j = 0; j < s.n; j++) if ((s.x[j] - s.x[i]) ** 2 + (s.y[j] - s.y[i]) ** 2 < 1e4) { if (s.type[j] === 0) a++; else b++; }
        const sc = Math.min(a, b) - (Math.abs(s.x[i] - D.size / 2) + Math.abs(s.y[i] - D.size / 2)) / 50;
        if (sc > bs) { bs = sc; best = [s.x[i], s.y[i]]; }
      });
      win = [Math.max(0, Math.min(D.size - WIN, best[0] - WIN / 2)), Math.max(0, Math.min(D.size - WIN, best[1] - WIN / 2))];
    }
    paint3();
  }
  function paint3() {
    const s = S[sec3], n = s.n;
    const deg = new Array(n).fill(0); E.forEach(([a, b]) => { deg[a]++; deg[b]++; });
    const cross = E.filter(([a, b]) => s.type[a] !== s.type[b]).length;
    const t8 = E.filter(([a, b]) => (s.type[a] === 0 && s.type[b] === CD8) || (s.type[b] === 0 && s.type[a] === CD8)).length;
    const iso = deg.filter((d) => d === 0).length;
    M3a.overlay((ctx, k) => { ctx.strokeStyle = css('--ink'); ctx.lineWidth = 2; ctx.strokeRect(win[0] * k, win[1] * k, WIN * k, WIN * k); });
    M3a.draw(s.x, s.y, (i) => ctColor(s.type[i]), () => 6);
    const inW = (i) => s.x[i] >= win[0] - 20 && s.x[i] <= win[0] + WIN + 20 && s.y[i] >= win[1] - 20 && s.y[i] <= win[1] + WIN + 20;
    const ids = d3.range(n).filter(inW);
    const lx = ids.map((i) => s.x[i] - win[0]), ly = ids.map((i) => s.y[i] - win[1]);
    const loc = new Map(ids.map((i, q) => [i, q]));
    M3b.overlay((ctx, k) => {
      ctx.lineCap = 'round';
      E.forEach(([a, b]) => {
        if (!loc.has(a) || !loc.has(b)) return;
        const hot = (s.type[a] === 0 && s.type[b] === CD8) || (s.type[b] === 0 && s.type[a] === CD8);
        ctx.strokeStyle = hot ? css('--ink') : css('--muted'); ctx.lineWidth = (hot ? 3 : 1) * k * 0.9; ctx.globalAlpha = hot ? 1 : 0.7;
        ctx.beginPath(); ctx.moveTo(lx[loc.get(a)] * k, ly[loc.get(a)] * k); ctx.lineTo(lx[loc.get(b)] * k, ly[loc.get(b)] * k); ctx.stroke();
      });
      ctx.globalAlpha = 1;
      ids.forEach((i, q) => { ctx.fillStyle = ctColor(s.type[i]); ctx.strokeStyle = css('--surface'); ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(lx[q] * k, ly[q] * k, 5 * k, 0, 6.2832); ctx.fill(); ctx.stroke(); });
    });
    M3b.draw([], [], () => null);
    M3b.hover((x, y) => {
      let best = -1, bd = 100;
      ids.forEach((i, q) => { const d = (lx[q] - x) ** 2 + (ly[q] - y) ** 2; if (d < bd) { bd = d; best = q; } });
      if (best < 0) return null;
      const i = ids[best];
      const nb = E.filter(([a, b]) => a === i || b === i).map(([a, b]) => (a === i ? b : a));
      const tally = d3.rollups(nb, (v) => v.length, (j) => D.cell_types[s.type[j]]).map(([c, m]) => `${m} ${c}`).join(', ');
      return `<b>${D.cell_types[s.type[i]]}</b> · degree ${deg[i]}<br>neighbours: ${tally || 'none'}`;
    });
    rd3.set({ edges: E.length.toLocaleString(), 'mean degree': fmt(2 * E.length / n, 2), 'isolated cells': iso, 'edges joining different types': pct(cross / Math.max(1, E.length)), 'CD8 T–tumour edges': t8 });
  }
  build();

  const mdeg = (gt, pv) => fmt(2 * edgesFor(S[0], gt, { [gt]: pv }).length / S[0].n, 2);
  const isoAt = (pv) => { const e = edgesFor(S[0], 'delaunay', { delaunay: pv }), d = new Array(S[0].n).fill(0); e.forEach(([a, b]) => { d[a]++; d[b]++; }); return d.filter((x) => !x).length; };
  const nTypes = d3.median(d3.range(V.n_spots), (j) => V.true[j].filter((f) => f > 0).length);
  const s1 = secStats[0], s3i = D.phen.indexOf('excluded'), s3 = secStats[s3i], s4i = D.phen.indexOf('desert');
  section(main, 'What to notice', `<ul>
    <li>In the bar chart, ${D.sections[0]} and ${D.sections[s3i]} both contain CD8 T cells (${pct(s1.f8, 1)} vs ${pct(s3.f8, 1)} of cells), which is why bulk T-cell genes did not cleanly separate them. Now colour the map by <b>cell type</b> and switch between them: in ${D.sections[0]}, ${pct(s1.inside)} of CD8 T cells sit inside tumour nests; in ${D.sections[s3i]}, only ${pct(s3.inside)}. That difference is invisible to any method that has no coordinates.</li>
    <li>Colour by <b>CXCL9</b> or <b>CD8A</b> in ${D.sections[s4i]}: almost nothing. A desert tumour has few T cells anywhere.</li>
    <li>In the spot view, the median spot holds ${nTypes} different cell types. Switch to <b>CD8 T fraction</b> and compare ${D.sections[0]} with ${D.sections[s3i]}: the T-cell signal is spread over 55 µm discs, so whether a T cell is inside a nest or at its edge is blurred.</li>
    <li>In the graph figure (${D.sections[0]}), Delaunay with the 35 µm cut gives mean degree ${mdeg('delaunay', 35)}; raising the cut to 80 µm only adds long edges along the tissue border (${mdeg('delaunay', 80)}), while cutting at 20 µm drops real neighbours (${mdeg('delaunay', 20)}, with ${isoAt(20)} isolated cells). With <b>kNN</b>, mean degree is set by you, not the tissue (k = 6 gives ${mdeg('knn', 6)}, since k-neighbour relations are not symmetric).</li>
    <li>Click the window onto a tumour-nest border in ${D.sections[0]} and then in ${D.sections[s3i]}: thick CD8 T–tumour edges are common in one and rare in the other. <a href="#/omics/sp-niches">Niche analysis</a> turns that count into a test.</li>
  </ul>`);

  steps(main, 'How it works', [
    '<b>Segment</b> cells on the image (nuclear stain plus membrane markers) and assign each decoded transcript to a cell. Each cell gets a count vector over the panel genes and an x, y centroid. On Visium you skip this: each spot is the unit.',
    '<b>Annotate</b> cell types, usually by transferring labels from a single-cell reference (the <a href="#/omics/sc-annotation">annotation step</a>).',
    '<b>Connect neighbours.</b> Delaunay triangulation links each cell to the cells whose territories touch it; trim edges longer than a cell-to-cell distance. Alternatives are k nearest neighbours or every cell within a radius r. On Visium, the hexagonal grid gives each spot its 6 neighbours.',
    'Store the result as a sparse adjacency matrix <code>A</code> (A<sub>ij</sub> = 1 if i and j are neighbours). Row-normalising it gives <code>W</code>, the "average of my neighbours" operator that Moran\'s I, smoothing and niche tests all use.',
  ]);
  math(main, `W = D⁻¹ A        (D = diag(degree))\n(W x)_i = mean of x over the neighbours of cell i\nthis build: Delaunay, edges ≤ 35 µm → mean degree ${S.map((s) => fmt(s.mean_deg, 2)).join(', ')} in ${D.sections.map((s) => s.split(' ')[0]).join(', ')}`);

  callout(main, '', 'In biology', `<ul>
    <li><b>Platforms.</b> Imaging-based, single-cell resolution: 10x <b>Xenium</b>, NanoString/Bruker <b>CosMx</b>, Vizgen <b>MERSCOPE</b> (hundreds to a few thousand genes). Sequencing-based: 10x <b>Visium</b> (55 µm spots, whole transcriptome), Visium HD (2 µm bins), Slide-seq, Stereo-seq. Protein imaging (CODEX/PhenoCycler, IMC) gives the same cell-plus-coordinate data structure.</li>
    <li><b>Graphs in practice.</b> <code>squidpy.gr.spatial_neighbors</code> builds Delaunay, kNN, radius or grid graphs on an AnnData object; Giotto and Seurat have equivalents. Most downstream tools (Squidpy, SpatialDE, BANKSY, CellChat spatial mode) take this graph as input.</li>
  </ul>`);
  callout(main, 'warn', 'Pitfalls', `<ul>
    <li><b>Segmentation errors leak expression.</b> A T cell squeezed against a tumour cell picks up tumour transcripts; this looks like "T cells expressing EPCAM" and can fake contact signals. Check with marker co-expression and use segmentation-aware tools.</li>
    <li><b>Section edges and holes.</b> Cells at the border have fewer neighbours; untrimmed Delaunay connects across tears and air spaces.</li>
    <li><b>One section is one sample.</b> ${D.n_panel} genes × ${nTot.toLocaleString()} cells feels big, but here there are ${NS} patients. Comparisons between phenotypes are n = ${NS}, not n = ${nTot.toLocaleString()}; the <a href="#/omics/sp-biomarker">biomarker page</a> needs a cohort.</li>
    <li>Gene panels are chosen in advance. A targeted panel cannot find a gene it does not contain.</li>
  </ul>`);
  seeAlso(main, [['omics', 'sp-svg', 'Spatially variable genes'], ['omics', 'sp-deconvolution', 'Spot deconvolution'], ['omics', 'sc-annotation', 'Cell-type annotation'], ['concepts', 'graph-clustering', 'Graph clustering'], ['concepts', 'knn', 'k-nearest neighbours']]);
}
