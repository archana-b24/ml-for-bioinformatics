// Variational autoencoders: a plain autoencoder and a VAE trained side by side, live, on the same toy cells.
// Figure 1: the two 2-D latent spaces (gappy vs smooth, near N(0,1)), β slider, hover to decode.
// Figure 2: sample z ~ N(0, I) and decode with each model; interpolate between two cell types.
import { header, figure, slider, select, button, stats, section, steps, callout, math, legend, seeAlso, animate, tip, untip, el } from '../lib/ui.js';
import { frame, heatmap, cat, css, ink, ink2, muted, sequential } from '../lib/plot.js';
import { rng } from '../lib/data.js';
import { mlp } from './autoencoder.js';
const d3 = window.d3;

const G = 20, BLOCKS = [[0, 7], [7, 14], [14, 20]], TYPES = ['type A', 'type B', 'type C'];
const SIG2 = 0.03; // assumed decoder noise variance: recon term = ‖x − x̂‖² / (2·SIG2)
const geneType = (g) => BLOCKS.findIndex(([a, b]) => g >= a && g < b);

/** Toy cells: 3 types; each type switches on its own 6–7 marker genes one after another as it matures (a ∈ [0,1]). */
function cells(n, seed) {
  const r = rng(seed);
  return d3.range(n).map((i) => {
    const k = i % 3, a = r(), [lo, hi] = BLOCKS[k], nb = hi - lo;
    const x = d3.range(G).map((g) => {
      const base = g >= lo && g < hi ? 0.2 + 0.8 / (1 + Math.exp(-10 * (a - (g - lo) / (nb - 1)))) : 0.05;
      return base + r.normal(0, 0.05);
    });
    return { k, a, x };
  });
}
const rmsd = (u, v) => { let s = 0; for (let j = 0; j < G; j++) s += (u[j] - v[j]) ** 2; return Math.sqrt(s / G); };

export default async function render(main) {
  header(main, 'Variational autoencoders',
    'A plain <a href="#/concepts/autoencoder">autoencoder</a> gives every input one exact code, and leaves the space between codes undefined. A VAE encodes each input as a <b>small cloud</b> of codes and pulls all the clouds towards one standard bell curve, N(0, 1). The result is a <b>smooth, filled-in latent space you can sample from</b>: pick any point near the centre, decode it, and get something that looks like data.');

  const train = cells(300, 31), test = cells(150, 32);
  let beta = 1, stepN = 0, goal = 3000, stop = null, hover = [0, 0], samples = [], sampleDraw = 0;
  const TARGET = 3000;
  let ae, vae, br, er;
  function build() {
    ae = { enc: mlp([G, 16, 2], ['tanh', 'linear'], 41), dec: mlp([2, 16, G], ['tanh', 'linear'], 42) };
    vae = { enc: mlp([G, 16, 4], ['tanh', 'linear'], 41), dec: mlp([2, 16, G], ['tanh', 'linear'], 42) };
    br = rng(3); er = rng(4); stepN = 0;
  }

  // =================== Figure 1 ===================
  const fig = figure(main, 'Two latent spaces, trained side by side',
    `Both models see the same ${train.length} toy cells: 20 genes, three cell types, and within each type a maturation gradient that switches its 6–7 marker genes on one after another. Both squeeze each cell to 2 numbers with the same network sizes (20 → 16 → 2 → 16 → 20). The only difference is the VAE's noise and its KL penalty. Points are coloured by cell type, which neither model sees. Faint rings mark 1 and 2 standard deviations of N(0, 1). On the right, ellipses show the VAE's uncertainty cloud (±1 sd) for every tenth cell. Blue shading marks the codes whose decoded profile is close to some real cell (refreshed every 20 frames). <b>Hover anywhere</b> in either panel to decode that point.`);
  const betaSl = slider(fig.controls, { label: 'KL weight β', min: -2, max: 1, step: 0.05, value: 0, fmt: (v) => (10 ** v).toFixed(v < -1 ? 3 : 2), onInput: (v) => { beta = 10 ** v; goal = Math.max(goal, stepN + 1500); if (!stop) run(); } });
  const runBtn = button(fig.controls, 'Pause', () => { if (stop) pause(); else { if (stepN >= goal) goal = stepN + 1500; run(); } }, true);
  button(fig.controls, 'Restart from scratch', () => { build(); goal = TARGET; run(); });
  const ro = stats(fig.root);
  legend(fig.body, TYPES.map((t, i) => [t, cat(i)]).concat([['N(0,1) rings', muted()], ['shading: this z decodes to a realistic cell', css('--accent-soft')]]));
  const row = el('div', { class: 'fig-row' }); fig.body.append(row);
  const left = el('div'), right = el('div'); row.append(left, right);
  const title = (p, t) => p.append(el('div', { class: 'fig-caption', style: 'margin:0 0 4px;font-weight:600' }, t));
  title(left, 'Plain autoencoder: code = one point');
  title(right, 'VAE: code = a cloud, pulled towards N(0, 1)');
  const FA = frame(left, { width: 460, height: 420, x: [-3.5, 3.5], y: [-3.5, 3.5], xLabel: 'z₁', yLabel: 'z₂' });
  const FV = frame(right, { width: 460, height: 420, x: [-3.5, 3.5], y: [-3.5, 3.5], xLabel: 'z₁', yLabel: 'z₂' });
  [FA, FV].forEach((F) => ['bg', 'rings', 'ell', 'pts', 'interp', 'samp', 'hov'].forEach((n) => F.layer(n)));
  const decRow = el('div', { class: 'fig-row' }); fig.body.append(decRow);
  const decL = el('div'), decR = el('div'); decRow.append(decL, decR);
  const DA = frame(decL, { width: 460, height: 150, x: [-0.5, G - 0.5], y: [-0.1, 1.15], yLabel: 'decoded', margin: { top: 22 } });
  const DV = frame(decR, { width: 460, height: 150, x: [-0.5, G - 0.5], y: [-0.1, 1.15], yLabel: 'decoded', margin: { top: 22 } });
  [DA, DV].forEach((D) => { D.g.select('.axis.x').call(d3.axisBottom(D.x).tickValues([0, 7, 14, 19]).tickFormat((g) => 'gene ' + (g + 1))); D.layer('bars'); D.t = D.g.append('text').attr('x', 4).attr('y', -8).attr('font-size', 12).attr('fill', ink2()); });

  // ---- training ----
  const dr = new Float64Array(G), dh = new Float64Array(4);
  function stepAE(x) {
    const z = Array.from(ae.enc.forward(x)), o = ae.dec.forward(z);
    for (let j = 0; j < G; j++) dr[j] = (o[j] - x[j]) / SIG2;
    const dz = ae.dec.backward(dr); ae.enc.backward(dz);
  }
  function stepVAE(x) {
    const h = vae.enc.forward(x), mu = [h[0], h[1]], lv = [Math.max(-10, Math.min(4, h[2])), Math.max(-10, Math.min(4, h[3]))];
    const eps = [er.normal(), er.normal()], sd = lv.map((v) => Math.exp(0.5 * v));
    const z = [mu[0] + sd[0] * eps[0], mu[1] + sd[1] * eps[1]], o = vae.dec.forward(z);
    for (let j = 0; j < G; j++) dr[j] = (o[j] - x[j]) / SIG2;
    const dz = vae.dec.backward(dr);
    for (let c = 0; c < 2; c++) { dh[c] = dz[c] + beta * mu[c]; dh[2 + c] = dz[c] * eps[c] * 0.5 * sd[c] + beta * 0.5 * (sd[c] * sd[c] - 1); }
    vae.enc.backward(dh);
  }
  function trainStep() {
    const B = 32;
    for (let i = 0; i < B; i++) { const d = train[br.int(train.length)]; stepAE(d.x); stepVAE(d.x); }
    [ae.enc, ae.dec, vae.enc, vae.dec].forEach((m) => m.step(0.004, B));
    stepN++;
  }
  function run() {
    if (stop) stop();
    runBtn.textContent = 'Pause';
    let frameN = 0;
    stop = animate(() => {
      const t0 = performance.now();
      while (performance.now() - t0 < 14 && stepN < goal) trainStep();
      paint(); if (++frameN % 20 === 0 || stepN >= goal) paint2();
      if (stepN >= goal) { pause(); return false; }
      return true;
    });
  }
  function pause() { if (stop) { stop(); stop = null; } runBtn.textContent = stepN >= goal ? 'Train 1,500 more' : 'Train'; }

  // ---- model queries ----
  const encA = (x) => { const z = ae.enc.forward(x); return [z[0], z[1]]; };
  const encV = (x) => { const h = vae.enc.forward(x); return { mu: [h[0], h[1]], sd: [Math.exp(0.5 * Math.min(4, h[2])), Math.exp(0.5 * Math.min(4, h[3]))] }; };
  const decA = (z) => Array.from(ae.dec.forward(z));
  const decV = (z) => Array.from(vae.dec.forward(z));
  function metrics() {
    let ra = 0, rv = 0, kl = 0;
    test.forEach((d) => {
      ra += rmsd(decA(encA(d.x)), d.x) ** 2;
      const q = encV(d.x); rv += rmsd(decV(q.mu), d.x) ** 2;
      for (let c = 0; c < 2; c++) kl += 0.5 * (q.mu[c] ** 2 + q.sd[c] ** 2 - 2 * Math.log(q.sd[c]) - 1);
    });
    return { ra: ra / test.length, rv: rv / test.length, kl: kl / test.length };
  }

  let domA = 3.5;
  function paint() {
    const pa = train.map((d) => ({ d, z: encA(d.x) }));
    const pv = train.map((d) => ({ d, ...encV(d.x) }));
    const ext = d3.max(pa, (p) => Math.max(Math.abs(p.z[0]), Math.abs(p.z[1])));
    const want = Math.max(3.5, Math.ceil(ext * 1.15));
    if (want > domA || want < domA * 0.6) domA = want;
    FA.x.domain([-domA, domA]); FA.y.domain([-domA, domA]);
    FA.g.select('.axis.x').call(d3.axisBottom(FA.x).ticks(5)); FA.g.select('.axis.y').call(d3.axisLeft(FA.y).ticks(5));
    FA.g.select('.gridline').call(d3.axisLeft(FA.y).ticks(5).tickSize(-FA.w).tickFormat(''));
    [FA, FV].forEach((F) => F.layer('rings').selectAll('circle').data([1, 2]).join('circle').attr('cx', F.x(0)).attr('cy', F.y(0))
      .attr('r', (r) => F.x(r) - F.x(0)).attr('fill', 'none').attr('stroke', muted()).attr('stroke-width', 1.2).attr('stroke-dasharray', '4 3'));
    const ptTip = (name) => (e, p) => tip(e, `${TYPES[p.d.k]}, maturation ${p.d.a.toFixed(2)}<br>${name} code (${(p.z || p.mu)[0].toFixed(2)}, ${(p.z || p.mu)[1].toFixed(2)})${p.sd ? `<br>cloud sd (${p.sd[0].toFixed(2)}, ${p.sd[1].toFixed(2)})` : ''}`);
    FA.layer('pts').selectAll('circle').data(pa).join('circle').attr('cx', (p) => FA.x(p.z[0])).attr('cy', (p) => FA.y(p.z[1])).attr('r', 4)
      .attr('fill', (p) => cat(p.d.k)).attr('stroke', css('--surface')).attr('stroke-width', 1).on('mousemove', ptTip('AE')).on('mouseleave', untip);
    FV.layer('pts').selectAll('circle').data(pv).join('circle').attr('cx', (p) => FV.x(p.mu[0])).attr('cy', (p) => FV.y(p.mu[1])).attr('r', 4)
      .attr('fill', (p) => cat(p.d.k)).attr('stroke', css('--surface')).attr('stroke-width', 1).on('mousemove', ptTip('VAE mean')).on('mouseleave', untip);
    const cl = (v) => Math.min(v, 3.5);
    FV.layer('ell').selectAll('ellipse').data(pv.filter((p, i) => i % 10 === 0)).join('ellipse')
      .attr('cx', (p) => FV.x(p.mu[0])).attr('cy', (p) => FV.y(p.mu[1])).attr('rx', (p) => FV.x(cl(p.sd[0])) - FV.x(0)).attr('ry', (p) => FV.x(cl(p.sd[1])) - FV.x(0))
      .attr('fill', (p) => cat(p.d.k)).attr('fill-opacity', 0.08).attr('stroke', (p) => cat(p.d.k)).attr('stroke-opacity', 0.5).attr('stroke-width', 1);
    paintSamples(); paintHover();
    const m = metrics();
    ro.set({ 'training step': stepN, β: beta.toFixed(beta < 0.1 ? 3 : 2), 'AE reconstruction RMSE': m.ra ** 0.5 > 0 ? Math.sqrt(m.ra).toFixed(3) : '–', 'VAE reconstruction RMSE': Math.sqrt(m.rv).toFixed(3), 'VAE KL per cell (nats)': m.kl.toFixed(2) });
  }
  function paintSamples() {
    const mark = (F, pts) => F.layer('samp').selectAll('path').data(pts).join('path')
      .attr('transform', (z) => `translate(${F.x(Math.max(F.x.domain()[0], Math.min(F.x.domain()[1], z[0])))},${F.y(Math.max(F.y.domain()[0], Math.min(F.y.domain()[1], z[1])))})`)
      .attr('d', d3.symbol(d3.symbolCross, 70)).attr('fill', ink()).attr('stroke', css('--surface')).attr('stroke-width', 1);
    mark(FA, samples); mark(FV, samples);
  }
  function showDecoded(D, prof, label) {
    D.t.text(label);
    D.layer('bars').selectAll('rect').data(prof || []).join('rect').attr('x', (v, g) => D.x(g - 0.4)).attr('width', D.x(0.8) - D.x(0))
      .attr('y', (v) => D.y(Math.max(0, Math.min(1.15, v)))).attr('height', (v) => D.y(0) - D.y(Math.max(0, Math.min(1.15, v)))).attr('fill', (v, g) => cat(geneType(g))).attr('rx', 2);
  }
  function paintHover() {
    if (!hover) {
      showDecoded(DA, null, 'AE decoder: hover over the latent panels'); showDecoded(DV, null, 'VAE decoder: hover over the latent panels');
      [FA, FV].forEach((F) => F.layer('hov').selectAll('*').remove()); return;
    }
    const z = hover;
    showDecoded(DA, decA(z), `AE decodes z = (${z[0].toFixed(2)}, ${z[1].toFixed(2)})`);
    showDecoded(DV, decV(z), `VAE decodes z = (${z[0].toFixed(2)}, ${z[1].toFixed(2)})`);
    [FA, FV].forEach((F) => F.layer('hov').selectAll('circle').data([z]).join('circle').attr('cx', F.x(z[0])).attr('cy', F.y(z[1])).attr('r', 7)
      .attr('fill', 'none').attr('stroke', ink()).attr('stroke-width', 2).attr('pointer-events', 'none'));
  }
  [FA, FV].forEach((F) => F.svg.on('mousemove.dec', (e) => { const [mx, my] = d3.pointer(e, F.g.node()); hover = [F.x.invert(mx), F.y.invert(my)]; paintHover(); }));

  // =================== Figure 2 ===================
  const fig2 = figure(main, 'Sample from N(0, 1) and decode; walk between cell types',
    'Generation means: draw z from the standard normal, then run only the decoder. Top: 12 real cells for reference, then 12 decodes of the <i>same</i> 12 random z from each model (they are the black crosses in the panels above). Each row is one cell, each column one gene; darker = higher. A realistic decode lights up one block, in a left-to-right staircase. Bottom: 9 evenly spaced steps along the straight line between the average codes of two cell types.');
  let pair = '0-1';
  button(fig2.controls, 'Draw 12 new samples', () => { sampleDraw++; drawSamples(); paint(); paint2(); }, true);
  select(fig2.controls, { label: 'Interpolate', value: pair, options: [['0-1', 'type A → type B'], ['1-2', 'type B → type C'], ['0-2', 'type A → type C']], onChange: (v) => { pair = v; paint2(); } });
  const ro2 = stats(fig2.root);
  const row2 = el('div', { class: 'fig-row' }); fig2.body.append(row2);
  const fb = { style: 'flex:1 1 240px' }, cReal = el('div', fb), cA = el('div', fb), cV = el('div', fb); row2.append(cReal, cA, cV);
  const row3 = el('div', { class: 'fig-row' }); fig2.body.append(row3);
  const iA = el('div'), iV = el('div'); row3.append(iA, iV);
  const realShow = d3.range(12).map((i) => test[(i * 37) % test.length]);
  function drawSamples() { const r = rng(100 + sampleDraw); samples = d3.range(12).map(() => [r.normal(), r.normal()]); }
  drawSamples();
  const nnDist = (prof) => d3.min(train, (d) => rmsd(prof, d.x));
  function strip(parent, name, M, rowLabels) {
    parent.innerHTML = '';
    parent.append(el('div', { class: 'fig-caption', style: 'margin:0 0 4px;font-weight:600' }, name));
    heatmap(parent, M.map((r) => r.map((v) => Math.max(0, Math.min(1, v)))), { cell: 11, color: sequential(0, 1), rowLabels, labelW: 46,
      onHover: (e, i, j, v) => tip(e, `${rowLabels[i]}, gene ${j + 1} (${TYPES[geneType(j)]} marker)<br>value ${M[i][j].toFixed(2)}`) });
  }
  function centroid(enc, k) { const zs = train.filter((d) => d.k === k).map((d) => enc(d.x)); return [d3.mean(zs, (z) => z[0]), d3.mean(zs, (z) => z[1])]; }
  function paint2() {
    const sa = samples.map((z) => decA(z)), sv = samples.map((z) => decV(z));
    const labs = d3.range(12).map((i) => `#${i + 1}`);
    strip(cReal, 'Real cells', realShow.map((d) => d.x), realShow.map((d) => TYPES[d.k].replace('type ', '')));
    strip(cA, 'AE decodes of z ~ N(0,1)', sa, labs);
    strip(cV, 'VAE decodes of z ~ N(0,1)', sv, labs);
    const [k1, k2] = pair.split('-').map(Number);
    const ts = d3.range(9).map((i) => i / 8);
    const path = (enc, dec) => { const a = centroid(enc, k1), b = centroid(enc, k2); return { a, b, M: ts.map((t) => dec([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])])) }; };
    const pA = path(encA, decA), pV = path((x) => encV(x).mu, decV);
    const tl = ts.map((t) => `t=${t.toFixed(2).replace(/0$/, '')}`);
    strip(iA, `AE: ${TYPES[k1]} → ${TYPES[k2]}`, pA.M, tl);
    strip(iV, `VAE: ${TYPES[k1]} → ${TYPES[k2]}`, pV.M, tl);
    [[FA, pA], [FV, pV]].forEach(([F, p]) => F.layer('interp').selectAll('line').data([p]).join('line')
      .attr('x1', F.x(p.a[0])).attr('y1', F.y(p.a[1])).attr('x2', F.x(p.b[0])).attr('y2', F.y(p.b[1])).attr('stroke', ink2()).attr('stroke-width', 2).attr('stroke-dasharray', '1 4').attr('stroke-linecap', 'round'));
    const ref = d3.median(test, (d) => nnDist(d.x));
    // realism map: decode a grid of z and shade by distance of the decode to its nearest real cell
    const sub = train.filter((d, i) => i % 4 === 0), N = 26;
    [[FA, decA], [FV, decV]].forEach(([F, dec]) => {
      const [lo, hi] = F.x.domain(), cs = (hi - lo) / N, cells = [];
      for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
        const z = [lo + (i + 0.5) * cs, lo + (j + 0.5) * cs], pr = dec(z);
        const dist = d3.min(sub, (d) => rmsd(pr, d.x));
        cells.push({ x0: lo + i * cs, y1: lo + (j + 1) * cs, cs, real: Math.max(0, Math.min(1, 1 - (dist - ref) / 0.12)) });
      }
      F.layer('bg').selectAll('rect').data(cells).join('rect').attr('x', (c) => F.x(c.x0)).attr('y', (c) => F.y(c.y1))
        .attr('width', (c) => F.x(c.cs) - F.x(0)).attr('height', (c) => F.x(c.cs) - F.x(0)).attr('shape-rendering', 'crispEdges')
        .attr('fill', css('--accent')).attr('fill-opacity', (c) => 0.28 * c.real);
    });
    ro2.set({ 'held-out real cell → nearest training cell (RMS)': ref.toFixed(3), 'AE sample → nearest real cell (median)': d3.median(sa, nnDist).toFixed(3), 'VAE sample → nearest real cell (median)': d3.median(sv, nnDist).toFixed(3) });
  }

  build(); paint(); paint2(); run();

  section(main, 'What to notice', `<ul>
    <li><b>Shape of the two spaces.</b> Both models lay the three types out as strands (each strand is one type's maturation gradient). Nothing ties the autoencoder's codes to N(0, 1): they sit wherever training happened to leave them, here bunched inside the inner ring. The VAE's strands stretch out to fill the 2-sd disc, and its uncertainty ellipses overlap their neighbours. The blue shading shows where a decoded z lands close to a real cell: the VAE's shaded region covers much more of the N(0, 1) disc than the AE's, where the wedges between strands decode to blends.</li>
    <li><b>Hover into a gap.</b> In the AE panel, move the pointer into unshaded space between strands: the decoded bars become a blend of two or three types that no real cell has. The VAE also decodes blends in its own unshaded gaps (the default hover point at the origin is one), but those gaps cover far less of the disc that N(0, 1) samples from.</li>
    <li><b>Sampling.</b> In the second figure, compare the AE and VAE rows with the real cells. The readout gives the median distance from each decoded sample to its nearest real cell, next to the distance you would expect from a genuinely new cell (held-out cell to nearest training cell). Press <i>Draw 12 new samples</i> a few times: the AE gap is not about one unlucky draw.</li>
    <li><b>β slider.</b> Training continues from the current weights when you move it. Push β up to 10: the KL term wins, the clouds swell, codes collapse towards the origin, and decodes drift towards the average cell (reconstruction error rises, KL falls). Push β down to 0.01: the VAE drifts towards plain-autoencoder behaviour (ellipses shrink, KL grows). The drift is gradual; press <i>Train 1,500 more</i> to keep going.</li>
    <li><b>Interpolation.</b> Walking from one cell type to another passes through blends in both models: there is no cell "halfway between" two types, and a VAE does not invent one. The VAE's path is shorter and stays in densely trained territory, so each step changes smoothly.</li>
  </ul>`);

  steps(main, 'How it works', [
    'The encoder outputs two numbers per latent dimension: a mean μ and a log-variance log σ². Together they describe a small Gaussian cloud q(z | x) of plausible codes for this input.',
    '<b>Reparameterisation trick:</b> draw ε ~ N(0, 1) and set z = μ + σ·ε. The randomness sits in ε, so gradients can flow back through μ and σ.',
    'The decoder rebuilds x̂ from the noisy z. Because the code jitters, nearby codes must decode to similar outputs: this is what makes the space smooth.',
    'Loss = reconstruction error + β · KL(q(z | x) ‖ N(0, I)). The KL term pulls every cloud towards the origin and towards unit width, so the codes of all inputs together fill the region where N(0, I) puts its mass.',
    'To generate: sample z ~ N(0, I) and decode. To embed: use μ. With β = 1 this loss is the negative evidence lower bound (ELBO) on log p(x); other β values trade sample quality against reconstruction (β-VAE).',
  ]);
  math(main, 'loss(x) = ‖x − decoder(z)‖² / (2σ²_dec)  +  β · KL( N(μ, σ²) ‖ N(0, 1) ),     z = μ + σ·ε,  ε ~ N(0, 1)\nKL per latent dim = ½ (μ² + σ² − log σ² − 1)          (here σ²_dec = 0.01, 2 latent dims)');
  callout(main, '', 'Honest simplifications', 'This is a teaching toy: 20 made-up genes, a Gaussian decoder (squared error), 2 latent dimensions so you can see them, one noise sample per cell per step, and 3,000 Adam steps of 32 cells. Real single-cell VAEs use 10–30 latent dimensions, count likelihoods and far more data. The exact numbers in the readouts depend on these choices; the qualitative contrast (gappy vs filled-in latent space) is the lesson.');
  callout(main, '', 'In biology', '<b>scVI</b> (scvi-tools) is the workhorse: a VAE whose decoder outputs a negative-binomial or zero-inflated distribution over raw counts, with the batch fed to the decoder so the latent space does not need to encode it. That is how scVI integrates batches. <b>scANVI</b> adds cell-type labels for semi-supervised annotation, <b>totalVI</b> models RNA plus surface protein (CITE-seq), <b>MultiVI</b> joins RNA and ATAC, and <b>scGen</b> and <b>CPA</b> use latent arithmetic to predict perturbation responses. The smooth latent space is also why scVI posteriors give usable differential-expression estimates.');
  callout(main, 'warn', 'Pitfalls', `<ul>
    <li><b>Posterior collapse.</b> With a strong decoder or a large β, the model ignores z (KL → 0) and decodes the average. Watch the KL term: near zero means the latent space carries no information.</li>
    <li><b>Blurry generations.</b> A Gaussian decoder averages over uncertainty, so samples are smoothed. Diffusion models and GANs give sharper samples; VAEs give better latent spaces and likelihoods.</li>
    <li><b>The latent is not "biology".</b> Latent dimensions are not guaranteed to be independent factors. Batch, library size and cell type can all mix unless the model structure (as in scVI's batch covariate) separates them.</li>
    <li><b>Interpolating between cell types</b> produces blends, not real intermediates. Do not read a decoded path as a differentiation trajectory without independent evidence.</li>
  </ul>`);
  seeAlso(main, [['concepts', 'autoencoder', 'Autoencoders'], ['concepts', 'diffusion', 'Diffusion models'], ['concepts', 'gan', 'GANs'], ['concepts', 'probability-mle', 'Likelihood & Bayes'], ['omics', 'sc-integration', 'Batch integration']]);
}
