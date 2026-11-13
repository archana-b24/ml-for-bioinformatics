// H&E -> expression: frozen pathology encoder -> PCA -> ridge -> per-gene Pearson r (HEST-1k protocol), on synthetic patches.
import { header, figure, slider, select, checkbox, stats, section, steps, math, legend, seeAlso, tip, untip, el } from '../lib/ui.js';
import { frame, cat, css, ink2 } from '../lib/plot.js';
import { rng, clamp } from '../lib/data.js';
import { sourceCard, toyBadge, coreIdea, realityCheck, useIt, cite, pearson, ridgeMulti, pca } from './_shared.js';
const d3 = window.d3;

const S = 16, P = 8, PER = 40;
const PROGS = ['Immune genes', 'Tumour genes', 'Stroma genes', 'Invisible genes'];
const GENES = PROGS.flatMap((p, k) => d3.range(6).map((i) => `${['imm', 'tum', 'str', 'hid'][k]}${i + 1}`));
const ENCODERS = {
  bank: 'Multi-scale filter bank (strong, pathology-FM stand-in)',
  rand: 'Random 3×3 conv filters (weak)',
  hist: 'Intensity histogram only (weakest)',
};

function makeData(seed = 21) {
  const r = rng(seed);
  const stain = d3.range(P).map(() => ({ b: r.normal(0, 0.15), c: 0.7 + 0.6 * r() }));
  const gw = GENES.map(() => 0.8 + 0.6 * r()), gsd = GENES.map(() => 0.35 + 0.3 * r());
  const off = d3.range(P).map(() => GENES.map(() => r.normal(0, 0.8))); // patient-level expression offsets (biology + batch)
  const spots = [];
  for (let p = 0; p < P; p++) for (let s = 0; s < PER; s++) {
    const z = [r(), r(), r()], hid = r();
    const img = new Float64Array(S * S).fill(0.12);
    const th = r() * Math.PI, ph = r() * 6.28;
    for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) img[y * S + x] += z[2] * 0.35 * (0.5 + 0.5 * Math.sin(2 * Math.PI * (x * Math.cos(th) + y * Math.sin(th)) / 4.5 + ph));
    const blob = (sig, amp) => { const cx = r() * S, cy = r() * S; for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) { const dd = (x - cx) ** 2 + (y - cy) ** 2; if (dd < 9 * sig * sig) img[y * S + x] += amp * Math.exp(-dd / (2 * sig * sig)); } };
    for (let k = 0; k < Math.round(z[0] * 14); k++) blob(0.7, 0.8);   // lymphocytes: many small dark nuclei
    for (let k = 0; k < Math.round(z[1] * 5); k++) blob(1.8, 0.5);    // tumour: few large nuclei
    for (let i = 0; i < S * S; i++) img[i] = clamp(stain[p].c * (img[i] + r.normal(0, 0.04)) + stain[p].b, 0, 1.3);
    const lat = [z[0], z[1], z[2], hid];
    const y = GENES.map((_, g) => gw[g] * lat[Math.floor(g / 6)] * 2 + off[p][g] + r.normal(0, gsd[g]));
    spots.push({ p, z, img, y });
  }
  return spots;
}

// ---- frozen encoders (no training: fixed filters, like using a pretrained model as-is) ----
function blur(img, sig) {
  const rad = Math.ceil(3 * sig), k = d3.range(-rad, rad + 1).map((d) => Math.exp(-d * d / (2 * sig * sig))); const ks = d3.sum(k);
  const tmp = new Float64Array(S * S), out = new Float64Array(S * S);
  for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) { let s = 0; for (let d = -rad; d <= rad; d++) s += k[d + rad] * img[y * S + clamp(x + d, 0, S - 1)]; tmp[y * S + x] = s / ks; }
  for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) { let s = 0; for (let d = -rad; d <= rad; d++) s += k[d + rad] * tmp[clamp(y + d, 0, S - 1) * S + x]; out[y * S + x] = s / ks; }
  return out;
}
const mean = (a) => { let s = 0; for (const v of a) s += v; return s / a.length; };
const sdv = (a) => { const m = mean(a); let s = 0; for (const v of a) s += (v - m) ** 2; return Math.sqrt(s / a.length); };
function encBank(img) {
  const b1 = blur(img, 0.8), b2 = blur(img, 1.6), b4 = blur(img, 3.6);
  const dog1 = b1.map((v, i) => v - b2[i]), dog2 = b2.map((v, i) => v - b4[i]);
  const relu = (a) => mean(a.map((v) => Math.max(0, v)));
  const E = [0, 1, 2, 3].map(() => 0);
  for (let y = 1; y < S - 1; y++) for (let x = 1; x < S - 1; x++) {
    const gx = b1[y * S + x + 1] - b1[y * S + x - 1], gy = b1[(y + 1) * S + x] - b1[(y - 1) * S + x];
    [0, 1, 2, 3].forEach((k) => { const t = k * Math.PI / 4; E[k] += (gx * Math.cos(t) + gy * Math.sin(t)) ** 2; });
  }
  E.sort((a, b) => b - a); // rotation-invariant: strongest orientation first
  return [relu(dog1), relu(dog2), sdv(dog1), sdv(dog2), ...E.map((e) => e / (S * S)), mean(img), sdv(img)];
}
const RF = (() => { const r = rng(99); return d3.range(8).map(() => d3.range(9).map(() => r.normal(0, 1))); })();
function encRand(img) {
  return RF.map((f) => { let s = 0; for (let y = 1; y < S - 1; y++) for (let x = 1; x < S - 1; x++) { let v = 0; for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) v += f[(dy + 1) * 3 + dx + 1] * img[(y + dy) * S + x + dx]; s += Math.max(0, v); } return s / ((S - 2) ** 2); });
}
function encHist(img) { const h = new Array(6).fill(0); for (const v of img) h[clamp(Math.floor(v / 1.3 * 6), 0, 5)] += 1; return h.map((c) => c / img.length); }

/** HEST-style evaluation for one split: standardise -> PCA(train) -> ridge -> per-gene Pearson r on test. */
function evalSplit(F, Y, tr, te, nPC, lambda) {
  const p = F[0].length, mu = d3.range(p).map((j) => d3.mean(tr, (i) => F[i][j])), sg = d3.range(p).map((j) => d3.deviation(tr, (i) => F[i][j]) || 1);
  const Z = (i) => F[i].map((v, j) => (v - mu[j]) / sg[j]);
  const k = Math.min(nPC, p);
  const pc = pca(tr.map(Z), k, 25);
  const proj = (z) => pc.axes.map((a) => a.reduce((s, w, j) => s + w * (z[j] - pc.mean[j]), 0));
  const Xtr = tr.map((i) => proj(Z(i))), Xte = te.map((i) => proj(Z(i)));
  const ym = GENES.map((_, g) => d3.mean(tr, (i) => Y[i][g]));
  const W = ridgeMulti(Xtr, tr.map((i) => Y[i].map((v, g) => v - ym[g])), lambda);
  const pred = Xte.map((x) => GENES.map((_, g) => ym[g] + x.reduce((s, v, c) => s + v * W[c][g], 0)));
  return GENES.map((_, g) => pearson(pred.map((r) => r[g]), te.map((i) => Y[i][g])));
}

export default async function render(main) {
  header(main, 'H&E → gene expression',
    'Every tumour has an H&E slide; few have spatial transcriptomics. Pathology foundation models turn image patches into feature vectors, and a <b>simple regression on those frozen features</b> predicts spot-level expression. The HEST-1k benchmark measures how well that works.');
  sourceCard(main, {
    org: 'Benchmark: HEST-1k (academic). Encoders: UNI, Prov-GigaPath (Microsoft/Providence, Nature 2024), Virchow (Paige), H-optimus (Bioptimus); M-Optimus-1 (Bioptimus, multimodal, 2026).',
    when: '2024–2026',
    links: ['hest', 'gigapath', 'moptimus'],
    claim: 'On HEST-1k (frozen embeddings → PCA → ridge, Pearson r on 50 highly variable genes) the best pathology encoder reaches a mean r of about 0.41. Bioptimus reports further gains for its multimodal M-Optimus-1 (vendor claim).',
    selfReported: false,
    status: 'benchmark; M-Optimus numbers are vendor claims',
  });
  coreIdea(main, `<p>A pathology foundation model is a large vision transformer pretrained by self-supervision on millions of H&E tiles. It is then <b>frozen</b>: you never update it, you only use it to turn each 224-pixel patch under a Visium spot into a feature vector. HEST-1k pairs H&E with spatial transcriptomics across many tissues and asks a narrow, fair question: <i>with a fixed, deliberately simple head, how much expression can each encoder's features predict?</i></p>
  <p>The protocol: embed patches → reduce with <a href="#/concepts/pca">PCA</a> → fit <a href="#/concepts/regularization">ridge regression</a> per gene → report Pearson r between predicted and measured expression on <b>held-out patients</b>. Keeping the head simple means differences come from the encoder, not from clever tuning.</p>
  <p><b>Key ML ingredient:</b> transfer learning by <b>linear probing</b> of frozen self-supervised image features.</p>`);

  const spots = makeData();
  const Y = spots.map((s) => s.y);

  // ---- figure 1: patch gallery ----
  const fig1 = figure(main, 'Synthetic "H&E" patches',
    `${spots.length} patches of 16×16 pixels from ${P} patients (${PER} spots each). Darker = more haematoxylin. Each patch mixes three hidden tissue states: <b>immune</b> (many small dark nuclei), <b>tumour</b> (a few large nuclei) and <b>stroma</b> (oriented fibres). Each patient also has its own stain intensity and contrast, as real slides do.`);
  toyBadge(fig1);
  const gal = el('div', { style: 'display:flex;flex-wrap:wrap;gap:8px' }); fig1.body.append(gal);
  const lo = css('--surface'), hi = cat(6), interp = d3.interpolateRgb(lo, hi);
  const pick = [];
  [0, 1, 2].forEach((k) => { const srt = spots.map((s, i) => [i, s.z[k] - (s.z[0] + s.z[1] + s.z[2] - s.z[k]) / 2]).sort((a, b) => b[1] - a[1]); pick.push(...srt.slice(0, 4).map(([i]) => [i, k])); });
  pick.forEach(([i, k]) => {
    const s = spots[i];
    const cv = el('canvas', { width: S, height: S, style: `width:60px;height:60px;image-rendering:pixelated;border-radius:4px;border:1px solid var(--border)` });
    const ctx = cv.getContext('2d'), im = ctx.createImageData(S, S);
    s.img.forEach((v, j) => { const c = d3.color(interp(clamp(v / 1.1, 0, 1))); im.data[4 * j] = c.r; im.data[4 * j + 1] = c.g; im.data[4 * j + 2] = c.b; im.data[4 * j + 3] = 255; });
    ctx.putImageData(im, 0, 0);
    gal.append(el('div', { style: 'text-align:center;font-size:12px;color:var(--ink-2)' }, cv, el('div', {}, ['immune-rich', 'tumour-rich', 'stroma-rich'][k]), el('div', {}, `patient ${s.p + 1}`)));
  });

  // ---- figure 2: HEST protocol ----
  const fig = figure(main, 'Mini-replication of the HEST protocol: frozen features → PCA → ridge → per-gene Pearson r',
    `${GENES.length} genes: 6 each driven by the immune, tumour and stroma states (visible in the image) and 6 driven by a state the image does not show ("invisible"). Every gene also has a patient-level offset. Results are averaged over 4 folds. Hover bars for values.`);
  toyBadge(fig, 'Hand-made filters stand in for a pathology foundation model; the r values are not HEST numbers.');
  let enc = 'bank', nPC = 6, lambda = 1, leaky = false;
  select(fig.controls, { label: 'Frozen encoder', value: enc, options: Object.entries(ENCODERS), onChange: (v) => { enc = v; update(); } });
  slider(fig.controls, { label: 'PCA components', min: 1, max: 10, step: 1, value: nPC, onInput: (v) => { nPC = v; update(); } });
  slider(fig.controls, { label: 'Ridge λ', min: -2, max: 3, step: 0.25, value: 0, fmt: (v) => (10 ** v).toPrecision(2), onInput: (v) => { lambda = 10 ** v; update(); } });
  checkbox(fig.controls, { label: 'Leaky split (random spots, not patients)', checked: false, onChange: (v) => { leaky = v; update(); } });
  const readout = stats(fig.root);
  const row = el('div', { class: 'fig-row' }); fig.body.append(row);
  const lb = el('div'), rb = el('div', { style: 'flex:0 1 360px' }); row.append(lb, rb);
  lb.append(el('div', { class: 'fig-caption', style: 'margin:0 0 4px' }, 'Per-gene Pearson r on held-out data (selected encoder)'));
  rb.append(el('div', { class: 'fig-caption', style: 'margin:0 0 4px' }, 'Mean r by gene group, all three encoders'));
  const FB = frame(lb, { width: 560, height: 300, x: [0, 1], y: [-0.3, 1], yLabel: 'Pearson r', margin: { bottom: 34 } });
  const bx = d3.scaleBand().domain(GENES).range([0, FB.w]).padding(0.18);
  FB.g.select('.axis.x').call(d3.axisBottom(bx).tickValues(GENES.filter((_, i) => i % 6 === 0)).tickFormat((g) => PROGS[GENES.indexOf(g) / 6].split(' ')[0]));
  FB.g.select('.axis.x').selectAll('text').attr('text-anchor', 'start').attr('dx', -6);
  FB.layer('zero').append('line').attr('x1', 0).attr('x2', FB.w).attr('y1', FB.y(0)).attr('y2', FB.y(0)).attr('stroke', ink2());
  legend(lb, PROGS.map((p, k) => [p, cat(k)]));
  const encKeys = Object.keys(ENCODERS);
  const FS = frame(rb, { width: 360, height: 300, x: [-0.1, 1], y: [0, 1], xLabel: 'mean Pearson r', yScale: d3.scaleBand().padding(0.3), margin: { left: 92 } });
  FS.y.domain(encKeys.map((k) => ['Filter bank', 'Random conv', 'Histogram'][encKeys.indexOf(k)]));
  FS.g.select('.axis.y').call(d3.axisLeft(FS.y)); FS.g.select('.gridline').remove();
  legend(rb, [['visible genes', cat(0)], ['invisible genes', cat(3)]]);

  const feats = {};
  const getF = (k) => feats[k] || (feats[k] = spots.map((s) => ({ bank: encBank, rand: encRand, hist: encHist }[k])(s.img)));
  const folds = () => {
    if (!leaky) return d3.range(4).map((f) => spots.map((s, i) => i).filter((i) => Math.floor(spots[i].p / 2) === f));
    const sh = rng(5).shuffle(spots.map((_, i) => i));
    return d3.range(4).map((f) => sh.filter((_, k) => k % 4 === f));
  };
  const cvR = (k) => {
    const F = getF(k), fs = folds();
    const rs = fs.map((te) => { const set = new Set(te); return evalSplit(F, Y, spots.map((_, i) => i).filter((i) => !set.has(i)), te, nPC, lambda); });
    return GENES.map((_, g) => d3.mean(rs, (r) => r[g]));
  };

  function update() {
    const all = Object.fromEntries(encKeys.map((k) => [k, cvR(k)]));
    const r = all[enc];
    FB.layer('bars').selectAll('rect').data(GENES).join('rect').attr('x', (g) => bx(g)).attr('width', bx.bandwidth())
      .attr('y', (g, i) => FB.y(Math.max(0, r[i]))).attr('height', (g, i) => Math.abs(FB.y(r[i]) - FB.y(0))).attr('fill', (g, i) => cat(Math.floor(i / 6))).attr('rx', 1.5)
      .on('mousemove', (e, g) => { const i = GENES.indexOf(g); tip(e, `<b>${g}</b> (${PROGS[Math.floor(i / 6)]})<br>Pearson r = ${r[i].toFixed(2)}`); }).on('mouseleave', untip);
    const vis = (rr) => d3.mean(rr.slice(0, 18)), inv = (rr) => d3.mean(rr.slice(18));
    const pts = encKeys.flatMap((k, j) => [{ k, j, v: vis(all[k]), c: 0 }, { k, j, v: inv(all[k]), c: 3 }]);
    const yName = (j) => FS.y.domain()[j];
    FS.layer('seg').selectAll('line').data(encKeys).join('line').attr('x1', (k) => FS.x(inv(all[k]))).attr('x2', (k) => FS.x(vis(all[k])))
      .attr('y1', (k, j) => FS.y(yName(j)) + FS.y.bandwidth() / 2).attr('y2', (k, j) => FS.y(yName(j)) + FS.y.bandwidth() / 2).attr('stroke', ink2()).attr('stroke-width', 1.5);
    FS.layer('dots').selectAll('circle').data(pts).join('circle').attr('cx', (d) => FS.x(d.v)).attr('cy', (d) => FS.y(yName(d.j)) + FS.y.bandwidth() / 2).attr('r', (d) => (d.k === enc ? 7 : 5))
      .attr('fill', (d) => cat(d.c)).attr('stroke', css('--surface')).attr('stroke-width', 1.5)
      .on('mousemove', (e, d) => tip(e, `${ENCODERS[d.k]}<br>${d.c ? 'invisible' : 'visible'} genes: mean r = ${d.v.toFixed(2)}`)).on('mouseleave', untip);
    readout.set({
      split: leaky ? 'random spots (leaky)' : 'by patient (HEST-style)',
      'features → PCs': `${getF(enc)[0].length} → ${Math.min(nPC, getF(enc)[0].length)}`,
      'mean r, all genes': d3.mean(r).toFixed(2),
      'mean r, visible genes': vis(r).toFixed(2),
      'mean r, invisible genes': inv(r).toFixed(2),
    });
  }
  update();

  section(main, 'What to notice', `<ul>
    <li><b>The encoder is the whole game.</b> With PCA and ridge held fixed (6 PCs, λ = 1), mean r over all genes is 0.35 for the filter bank (which "knows" what nuclei and fibres look like), 0.27 for random filters and 0.21 for the histogram. That is the comparison HEST makes between real pathology models.</li>
    <li><b>Invisible genes stay near zero</b> (mean r 0.05–0.06) with a patient split, whatever the encoder. Morphology cannot predict a state that leaves no trace in the image (think interferon signalling, or a metabolic programme). Some genes are simply not predictable from H&E.</li>
    <li>Tick <b>Leaky split</b>. Random spots put the same patients in train and test, so the model can learn "this stain intensity = this patient = this expression offset". With the filter bank, mean r rises from 0.35 to 0.41 and invisible genes from 0.06 to 0.15. With random filters, which mostly encode stain, invisible genes jump from 0.05 to 0.24, and the gap to the good encoder nearly closes. No extra biology was learned. HEST splits by patient for this reason.</li>
    <li>Drop <b>PCA components</b> to 1 or 2: mean r falls from 0.35 to 0.28, because two axes cannot carry three tissue states. Large λ shrinks predictions towards the mean; Pearson r ignores scale, so it falls only when the ranking of spots degrades.</li>
  </ul>`);
  steps(main, 'How it works', [
    'Cut the H&E image into a patch centred on each spatial-transcriptomics spot (HEST: 224×224 pixels at a fixed magnification).',
    'Run each patch through a <b>frozen</b> pretrained encoder (UNI, Prov-GigaPath, Virchow, H-optimus, …) to get one feature vector per spot.',
    'Standardise the features and reduce them with PCA, fitted on training patients only.',
    'Fit ridge regression from PCs to the log-expression of each target gene (HEST: 50 highly variable genes).',
    'Predict on held-out <b>patients</b> and report the Pearson r per gene, averaged over genes and folds.',
  ]);
  math(main, 'z = PCA_k( standardise( f_frozen(patch) ) )\nŷ_g = w_gᵀ z + b_g,   w_g = argmin ‖y_g − Zw‖² + λ‖w‖²\nscore = mean_g Pearson( ŷ_g, y_g )  on held-out patients');
  realityCheck(main, `<p>On ${cite('hest', 'HEST-1k')}, the best encoder's mean Pearson r on 50 HVGs is about <b>0.41</b>. That is useful for broad compartments (immune-rich vs tumour vs stroma) and far from a substitute for measuring expression gene by gene. Encoder papers such as ${cite('gigapath', 'Prov-GigaPath')} report their own wins; Bioptimus's ${cite('moptimus', 'M-Optimus-1')} numbers are <b>vendor claims</b>.</p>
  <p>Two caveats the toy makes concrete: (1) any split that lets a patient's spots appear in both train and test inflates r, and (2) a high average hides genes at r ≈ 0. Look at per-gene scores, not only the mean. Industry datasets that pair H&E with omics at scale (Owkin ${cite('mosaic', 'MOSAIC')}, Noetik ${cite('noetik', 'OCTO-vc')}) exist, but their model performance claims are announcement-level.</p>`);
  useIt(main, `<ul>
    <li><b>Cheap virtual spatial on archival H&E:</b> if you have a handful of Visium sections with matched H&E, linear-probe a public pathology encoder on them and predict immune or stromal programme scores across a larger H&E-only cohort. Predict programme scores, not single genes.</li>
    <li><b>Evaluate honestly:</b> leave-one-patient-out, per-gene r with a null (shuffle spots within patient), and report the genes that fail.</li>
    <li><b>Sanity-check the stain:</b> check whether predictions track slide or site. If a classifier can tell the slide from the features, the regression can exploit it.</li>
    <li><b>Use it as a feature, not a readout:</b> morphology embeddings are strong covariates for spatial domain calling or deconvolution priors, even when per-gene prediction is weak.</li>
  </ul>`);
  seeAlso(main, [['concepts', 'foundation-models', 'Foundation models'], ['concepts', 'cnn', 'Convolutions (CNNs)'], ['concepts', 'pca', 'PCA'], ['concepts', 'regularization', 'Regularization'], ['concepts', 'cross-validation', 'Cross-validation'], ['concepts', 'train-test', 'Train / test & leakage'],
    ['omics', 'sp-data', 'Spatial data'], ['omics', 'sp-deconvolution', 'Spot deconvolution'], ['omics', 'sp-biomarker', 'Spatial biomarker']]);
}
