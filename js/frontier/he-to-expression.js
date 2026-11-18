// H&E to gene expression, lesson format (frontier tab).
// Arc: what → why → where → how it works (6 steps: patch→embedding, PCA+ridge probe, patient-level splits,
//      what is predictable, lab with stain/centre shift, spatial context and niche models) → pitfalls → recap.
// Everything on the page is a teaching toy: simulated 16×16 "H&E" patches and spot-level expression for patients of the mock study.
// All numbers in prose are computed at render time. Facts and URLs come from docs/frontier_research.md.
import { header, figure, slider, select, checkbox, stats, legend, seeAlso, tip, untip, el, html } from '../lib/ui.js';
import { frame, cat, css, ink, ink2, muted, sequential } from '../lib/plot.js';
import { rng, clamp, mean } from '../lib/data.js';
import { auc } from '../lib/models.js';
import { loadCohort } from '../lib/study.js';
import { lessonNav, chapter, prose, rung, worked, quiz, cards, table, code, recap } from '../lib/lesson.js';
import { sourceCard, toyBadge, cite, pearson, ridgeMulti, pca } from './_shared.js';
const d3 = window.d3;

const um = (t) => t.replace(/^-/, '−');
const f2 = (v) => (Math.abs(v) < 0.005 ? '0.00' : um(v.toFixed(2)));
const pc = (v) => Math.round(100 * v) + '%';
const S = 16, GRID = 6, PER = GRID * GRID, NAB = 12, MAXHOP = 6;
const GROUPS = ['Immune genes', 'Tumour genes', 'Stroma genes', 'Signalling genes'];
const GSHORT = ['immune', 'tumour', 'stroma', 'signalling'];
const GENES = GROUPS.flatMap((_, k) => d3.range(6).map((i) => `${['imm', 'tum', 'str', 'sig'][k]}${i + 1}`));
const ENC = { bank: 'Multi-scale filter bank', rand: 'Random 3×3 filters', hist: 'Intensity histogram' };
const ENCK = Object.keys(ENC);
const FNAMES = {
  bank: ['blob response, fine', 'blob response, coarse', 'texture SD, fine', 'texture SD, coarse', 'edge energy 1 (strongest)', 'edge energy 2', 'edge energy 3', 'edge energy 4', 'mean intensity', 'intensity SD'],
  rand: d3.range(8).map((i) => `random filter ${i + 1}`),
  hist: d3.range(6).map((i) => `intensity bin ${i + 1}`),
};
const STAIN = { 'Centre A': { c: 0.8, b: -0.02 }, 'Centre B': { c: 1.2, b: 0.12 }, 'Centre C': { c: 1.55, b: 0.25 } };
const BASE = { inflamed: [0.62, 0.45, 0.28], excluded: [0.28, 0.55, 0.55], desert: [0.1, 0.45, 0.4] };

// ---------------------------------------------------------------------------------------------
// Simulation: patients from the study, a 6×6 grid of spots each, smooth tissue fields, 16×16 patches, spot expression.
function makeData(C) {
  const D = C.discovery, V = C.validation, r = rng(21);
  const take = (arr, centre, ph, n) => arr.filter((p) => p.centre === centre && p.phenotype === ph).slice(0, n);
  const pats = [...take(D, 'Centre A', 'inflamed', 2), ...take(D, 'Centre A', 'excluded', 2), ...take(D, 'Centre A', 'desert', 2),
    ...take(D, 'Centre B', 'inflamed', 4), ...take(D, 'Centre B', 'excluded', 1), ...take(D, 'Centre B', 'desert', 1),
    ...take(V, 'Centre C', 'inflamed', 1), ...take(V, 'Centre C', 'excluded', 2), ...take(V, 'Centre C', 'desert', 1)];
  const mi = mean(C.all.map((p) => p.ifng)), si = Math.sqrt(mean(C.all.map((p) => (p.ifng - mi) ** 2)));
  const gw = GENES.map(() => 0.8 + 0.6 * r()), gsd = GENES.map(() => 0.35 + 0.3 * r());
  const spots = [];
  pats.forEach((pt, p) => {
    const st = { c: STAIN[pt.centre].c + r.normal(0, 0.06), b: STAIN[pt.centre].b + r.normal(0, 0.03) };
    const off = GENES.map(() => r.normal(0, 0.2)), ifz = (pt.ifng - mi) / si;
    const bumps = d3.range(4).map(() => d3.range(2).map(() => ({ x: r(), y: r(), a: r.normal(0, 0.25) })));
    const field = (k, x, y) => bumps[k].reduce((s, q) => s + q.a * Math.exp(-((x - q.x) ** 2 + (y - q.y) ** 2) / (2 * 0.25 ** 2)), 0);
    for (let gi = 0; gi < GRID; gi++) for (let gj = 0; gj < GRID; gj++) {
      const x = (gj + 0.5) / GRID, y = (gi + 0.5) / GRID, b = BASE[pt.phenotype];
      const z = [0, 1, 2].map((k) => clamp(b[k] + field(k, x, y) + r.normal(0, 0.07), 0.02, 0.98));
      const h = clamp(0.5 + 0.18 * ifz + field(3, x, y) + r.normal(0, 0.08), 0.02, 0.98);
      const img = new Float64Array(S * S).fill(0.12), th = r() * Math.PI, ph = r() * 6.28;
      for (let yy = 0; yy < S; yy++) for (let xx = 0; xx < S; xx++) img[yy * S + xx] += z[2] * 0.5 * (0.5 + 0.5 * Math.sin(2 * Math.PI * (xx * Math.cos(th) + yy * Math.sin(th)) / 4.5 + ph));
      const blob = (sig, amp) => { const cx = r() * S, cy = r() * S; for (let yy = 0; yy < S; yy++) for (let xx = 0; xx < S; xx++) { const dd = (xx - cx) ** 2 + (yy - cy) ** 2; if (dd < 9 * sig * sig) img[yy * S + xx] += amp * Math.exp(-dd / (2 * sig * sig)); } };
      for (let k = 0; k < Math.round(z[0] * 14); k++) blob(0.7, 0.8);
      for (let k = 0; k < Math.round(z[1] * 9); k++) blob(1.8, 0.7);
      for (let i = 0; i < S * S; i++) img[i] = clamp(st.c * (img[i] + r.normal(0, 0.04)) + st.b, 0, 1.3);
      const lat = [z[0], z[1], z[2], h];
      spots.push({ p, s: gi * GRID + gj, gi, gj, z, img, pt, y: GENES.map((_, g) => gw[g] * lat[Math.floor(g / 6)] * 3 + off[g] + r.normal(0, gsd[g])) });
    }
  });
  // stain-normalised copy: each patient's pixels re-centred and re-scaled to the pooled mean and SD
  const all = spots.flatMap((s) => Array.from(s.img)), mu = mean(all), sg = Math.sqrt(mean(all.map((v) => (v - mu) ** 2)));
  pats.forEach((_, p) => {
    const mine = spots.filter((s) => s.p === p), px = mine.flatMap((s) => Array.from(s.img)), m = mean(px), q = Math.sqrt(mean(px.map((v) => (v - m) ** 2)));
    mine.forEach((s) => { s.imgN = s.img.map((v) => (v - m) / q * sg + mu); });
  });
  return { pats, spots };
}

// ---- frozen encoders: fixed functions of the patch, never trained (stand-ins for a pathology foundation model) ----
function blur(img, sig) {
  const rad = Math.ceil(3 * sig), k = d3.range(-rad, rad + 1).map((d) => Math.exp(-d * d / (2 * sig * sig))), ks = d3.sum(k);
  const tmp = new Float64Array(S * S), out = new Float64Array(S * S);
  for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) { let s = 0; for (let d = -rad; d <= rad; d++) s += k[d + rad] * img[y * S + clamp(x + d, 0, S - 1)]; tmp[y * S + x] = s / ks; }
  for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) { let s = 0; for (let d = -rad; d <= rad; d++) s += k[d + rad] * tmp[clamp(y + d, 0, S - 1) * S + x]; out[y * S + x] = s / ks; }
  return out;
}
const mn = (a) => { let s = 0; for (const v of a) s += v; return s / a.length; };
const sdv = (a) => { const m = mn(a); let s = 0; for (const v of a) s += (v - m) ** 2; return Math.sqrt(s / a.length); };
function encBank(img) {
  const b1 = blur(img, 0.8), b2 = blur(img, 1.6), b4 = blur(img, 3.6), d1 = b1.map((v, i) => v - b2[i]), d2 = b2.map((v, i) => v - b4[i]);
  const relu = (a) => mn(a.map((v) => Math.max(0, v))), E = [0, 0, 0, 0];
  for (let y = 1; y < S - 1; y++) for (let x = 1; x < S - 1; x++) {
    const gx = b1[y * S + x + 1] - b1[y * S + x - 1], gy = b1[(y + 1) * S + x] - b1[(y - 1) * S + x];
    E.forEach((_, k) => { const t = k * Math.PI / 4; E[k] += (gx * Math.cos(t) + gy * Math.sin(t)) ** 2; });
  }
  E.sort((a, b) => b - a);
  return [relu(d1), relu(d2), sdv(d1), sdv(d2), ...E.map((e) => e / (S * S)), mn(img), sdv(img)];
}
const RF = (() => { const r = rng(99); return d3.range(8).map(() => d3.range(9).map(() => r.normal(0, 1))); })();
const encRand = (img) => RF.map((f) => { let s = 0; for (let y = 1; y < S - 1; y++) for (let x = 1; x < S - 1; x++) { let v = 0; for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) v += f[(dy + 1) * 3 + dx + 1] * img[(y + dy) * S + x + dx]; s += Math.max(0, v); } return s / ((S - 2) ** 2); });
const encHist = (img) => { const h = new Array(6).fill(0); for (const v of img) h[clamp(Math.floor(v / 1.3 * 6), 0, 5)] += 1; return h.map((c) => c / img.length); };
const ENCF = { bank: encBank, rand: encRand, hist: encHist };

export default async function render(main) {
  const C = await loadCohort();
  const { pats, spots } = makeData(C);
  const N = spots.length, NA = NAB * PER, Y = spots.map((s) => s.y);
  const featCache = {};
  const feats = (enc, norm) => featCache[enc + norm] || (featCache[enc + norm] = spots.map((s) => ENCF[enc](norm ? s.imgN : s.img)));

  // ---- HEST-style probe: standardise → PCA → ridge, all fitted on the training spots only ----
  function fitPredict(F, tr, te, nPC, lam) {
    const p = F[0].length, mu = d3.range(p).map((j) => d3.mean(tr, (i) => F[i][j])), sg = d3.range(p).map((j) => d3.deviation(tr, (i) => F[i][j]) || 1);
    const Z = (i) => F[i].map((v, j) => (v - mu[j]) / sg[j]), pcs = pca(tr.map(Z), Math.min(nPC, p), 25);
    const proj = (z) => pcs.axes.map((a) => a.reduce((s, w, j) => s + w * (z[j] - pcs.mean[j]), 0));
    const ym = GENES.map((_, g) => d3.mean(tr, (i) => Y[i][g])), W = ridgeMulti(tr.map((i) => proj(Z(i))), tr.map((i) => Y[i].map((v, g) => v - ym[g])), lam);
    return te.map((i) => { const x = proj(Z(i)); return GENES.map((_, g) => ym[g] + x.reduce((s, v, c) => s + v * W[c][g], 0)); });
  }
  const shuffled = rng(5).shuffle(d3.range(NA));
  const resCache = {};
  /** Held-out predictions and per-gene Pearson r (pooled and within-patient) for one configuration. */
  function res(o = {}) {
    const k = Object.assign({ enc: 'bank', norm: false, nPC: 6, lam: 1, mode: 'patient' }, o), key = JSON.stringify(k);
    if (resCache[key]) return resCache[key];
    const F = feats(k.enc, k.norm), pred = new Array(N), ev = [];
    let folds;
    if (k.mode === 'patient') folds = d3.range(4).map((f) => [d3.range(NA).filter((i) => spots[i].p % 4 === f), d3.range(NA).filter((i) => spots[i].p % 4 !== f)]);
    else if (k.mode === 'spot') folds = d3.range(4).map((f) => [shuffled.filter((_, j) => j % 4 === f), shuffled.filter((_, j) => j % 4 !== f)]);
    else folds = [[d3.range(NA, N), d3.range(NA)]];
    folds.forEach(([te, tr]) => { fitPredict(F, tr, te, k.nPC, k.lam).forEach((v, j) => { pred[te[j]] = v; ev.push(te[j]); }); });
    const r = GENES.map((_, g) => rGene(pred, ev, g, false)), rw = GENES.map((_, g) => rGene(pred, ev, g, true));
    const grp = (a) => d3.range(4).map((q) => mean(a.slice(6 * q, 6 * q + 6)));
    const bias = mean(GENES.map((_, g) => Math.abs(mean(ev.map((i) => pred[i][g] - Y[i][g])))));
    return (resCache[key] = { pred, ev, r, rw, g: grp(r), gw: grp(rw), mean: mean(r), meanW: mean(rw), bias });
  }
  function rGene(pred, ev, g, within, Yo = Y, idx = null) {
    const a = ev.map((i) => pred[i][g]), b = ev.map((i, j) => Yo[idx ? idx[j] : i][g]);
    if (within) {
      const sa = {}, sb = {}, cnt = {};
      ev.forEach((i, j) => { const p = spots[i].p; sa[p] = (sa[p] || 0) + a[j]; sb[p] = (sb[p] || 0) + b[j]; cnt[p] = (cnt[p] || 0) + 1; });
      ev.forEach((i, j) => { const p = spots[i].p; a[j] -= sa[p] / cnt[p]; b[j] -= sb[p] / cnt[p]; });
    }
    return pearson(a, b);
  }
  const centreAUC = (enc, norm) => { const F = feats(enc, norm), ab = d3.range(NA), lab = ab.map((i) => (spots[i].pt.centre === 'Centre B' ? 1 : 0)); return d3.max(F[0].map((_, j) => { const a = auc(ab.map((i) => F[i][j]), lab); return Math.max(a, 1 - a); })); };

  const R0 = res(), Rs = res({ mode: 'spot' }), Rc = res({ mode: 'centre' });
  const nSig = (q) => q.g[3];

  // permutation bands for the default configuration (95th percentile of |r| when expression is unrelated to the image)
  const rp = rng(7), evA = d3.range(NA);
  const bandPooled = (() => { const v = []; for (let t = 0; t < 120; t++) { const perm = rp.shuffle(d3.range(NAB)), idx = evA.map((i) => perm[spots[i].p] * PER + (i % PER)); GENES.forEach((_, g) => v.push(Math.abs(rGene(R0.pred, evA, g, false, Y, idx)))); } return d3.quantile(v.sort(d3.ascending), 0.95); })();
  const bandWithin = (() => { const v = []; for (let t = 0; t < 120; t++) { const idx = evA.map((i) => spots[i].p * PER + rp.int(PER)); const perm = []; d3.range(NAB).forEach((p) => { const s = rp.shuffle(d3.range(PER)); d3.range(PER).forEach((q) => { perm[p * PER + q] = p * PER + s[q]; }); }); const ii = evA.map((i) => perm[i]); GENES.forEach((_, g) => v.push(Math.abs(rGene(R0.pred, evA, g, true, Y, ii)))); void idx; } return d3.quantile(v.sort(d3.ascending), 0.95); })();

  // ---- helpers for figures ----
  const patchColour = d3.interpolateRgb(css('--surface'), cat(6));
  function drawPatch(cv, img, scale = 1) {
    const ctx = cv.getContext('2d'), im = ctx.createImageData(S, S);
    img.forEach((v, j) => { const c = d3.color(patchColour(clamp(v / 1.1, 0, 1))); im.data[4 * j] = c.r; im.data[4 * j + 1] = c.g; im.data[4 * j + 2] = c.b; im.data[4 * j + 3] = 255; });
    ctx.putImageData(im, 0, 0); void scale;
  }
  const PHC = { inflamed: cat(0), excluded: cat(1), desert: cat(2) };
  const phLegend = (p) => legend(p, [['inflamed tumour', PHC.inflamed], ['excluded', PHC.excluded], ['desert', PHC.desert]]);
  function geneBars(parent, { width, height, names }) {
    const F = frame(parent, { width, height, x: [0, 1], y: [-0.4, 1], yLabel: 'Pearson r', axes: false, margin: { bottom: names ? 54 : 40 } });
    F.g.append('g').attr('class', 'axis y').call(d3.axisLeft(F.y).ticks(7));
    const bx = d3.scaleBand().domain(d3.range(24)).range([0, F.w]).padding(0.16);
    F.layer('bg'); F.layer('zero').append('line').attr('x1', 0).attr('x2', F.w).attr('y1', F.y(0)).attr('y2', F.y(0)).attr('stroke', ink2());
    GSHORT.forEach((t, k) => F.g.append('text').attr('x', bx(6 * k) + 3 * bx.step() - bx.step() / 2 + bx.bandwidth() / 2).attr('y', F.h + (names ? 40 : 18)).attr('text-anchor', 'middle').attr('font-size', 12).attr('fill', ink2()).text(t));
    if (names) GENES.forEach((n, g) => F.g.append('text').attr('x', bx(g) + bx.bandwidth() / 2).attr('y', F.h + 14).attr('text-anchor', 'middle').attr('font-size', 9.5).attr('fill', muted()).text(n));
    F.bx = bx;
    F.draw = (r, extra = {}) => {
      F.layer('band').selectAll('*').remove();
      if (extra.band != null) {
        F.layer('band').append('rect').attr('x', 0).attr('width', F.w).attr('y', F.y(extra.band)).attr('height', F.y(-extra.band) - F.y(extra.band)).attr('fill', css('--surface-2')).attr('stroke', muted()).attr('stroke-dasharray', '4 3');
        F.layer('band').append('text').attr('x', F.w - 4).attr('y', F.y(extra.band) - 5).attr('text-anchor', 'end').attr('font-size', 11).attr('fill', ink2()).text(extra.bandLabel || '');
      }
      F.layer('bars').selectAll('rect').data(r).join('rect').attr('x', (_, i) => bx(i)).attr('width', bx.bandwidth()).attr('rx', 1.5).attr('fill', (_, i) => cat(Math.floor(i / 6)))
        .attr('y', (v) => F.y(Math.max(0, v))).attr('height', (v) => Math.abs(F.y(v) - F.y(0)))
        .on('mousemove', (e, v) => { const i = r.indexOf(v); tip(e, `<b>${GENES[i]}</b> (${GSHORT[Math.floor(i / 6)]})<br>Pearson r = ${f2(v)}`); }).on('mouseleave', untip);
    };
    return F;
  }
  const groupLegend = (p) => legend(p, GROUPS.map((g, k) => [g, cat(k)]));

  // =====================================================================================
  header(main, 'H&E to gene expression',
    'Spatial transcriptomics measures expression at thousands of positions on a tissue section but is rarely available, whereas an H&E slide exists for nearly every tumour. A <b>pathology foundation model</b> turns the image patch at each position into a feature vector, and a simple regression on those frozen features <b>predicts the expression measured there</b>. How much is recoverable, and under which evaluation, is the substance of this page.');
  lessonNav(main);
  sourceCard(main, {
    org: 'Benchmark: HEST-1k (academic). Encoders: UNI, Prov-GigaPath (Microsoft/Providence, Nature 2024), Virchow (Paige), H-optimus (Bioptimus); M-Optimus-1 (Bioptimus, multimodal, 2026). Spatial context models: Novae, Nicheformer, CellPLM.',
    when: '2024–2026', links: ['hest', 'gigapath', 'moptimus', 'novae', 'nicheformer'], selfReported: false,
    claim: 'On HEST-1k (frozen embeddings → PCA → ridge, Pearson r on 50 highly variable genes) the best pathology encoder reaches a mean r of about 0.41. The M-Optimus-1 gains are vendor claims.',
    status: 'benchmark; vendor numbers unverified',
  });
  main.append(html(`<div class="study-card"><div class="sc-k">The running example · teaching toy</div>
    <div class="sc-t">${pats.length} patients of the mock study, each with a ${GRID}×${GRID} grid of simulated spots</div>
    <div class="sc-d">Each spot has a 16×16-pixel simulated H&E patch and a measured expression vector for ${GENES.length} genes. ${NAB} patients come from Centres A and B (with their real phenotype in the study: inflamed, excluded or desert); ${pats.length - NAB} from Centre C are kept for a new-centre test. Centres differ in stain intensity. <b>The patches, the filters standing in for a foundation model and the resulting r values are a teaching toy; they are not a reproduction of any published model or number.</b></div>
    <div class="sc-q"><b>Question for this page:</b> how much of spot-level expression can be predicted from morphology alone, and how should that be evaluated?</div></div>`));

  // =====================================================================================
  // 1. WHAT
  // =====================================================================================
  let ch = chapter(main, 'what', 'What is H&E-to-expression prediction?', 'In one line: <b>image patch → frozen encoder → embedding → small regression head → expression of each gene</b>.');
  prose(ch, `<p>Haematoxylin and eosin (H&E) staining is the routine histology stain. In a spatial transcriptomics experiment (for example 10x Visium) the section is divided into <b>spots</b>, each with a measured expression vector and a matching H&E patch. A model trained on such pairs can then be applied to H&E images that have no expression data.</p>
    <p>The modern recipe separates the problem into two parts. A <b>pathology foundation model</b> (UNI, Prov-GigaPath, Virchow, H-optimus) was pretrained elsewhere, by self-supervision on very many H&E tiles, and is used <b>frozen</b>: it only converts a patch into an <b>embedding</b>. A deliberately small <b>head</b> (PCA followed by ridge regression, the protocol of the ${cite('hest', 'HEST-1k')} benchmark) maps the embedding to expression. The diagram follows one simulated spot through a toy version of that pipeline.</p>`);
  // example spot: immune-rich spot of an inflamed tumour with median error among such spots
  const cand = d3.range(NA).filter((i) => spots[i].z[0] > 0.6 && spots[i].pt.phenotype === 'inflamed');
  const errOf = (i) => [0, 6, 12, 18].reduce((s, g) => s + Math.abs(R0.pred[i][g] - Y[i][g]), 0);
  const exI = cand.sort((a, b) => errOf(a) - errOf(b))[Math.floor(cand.length * 0.5)];
  ch.append(pipelineSvg(spots[exI], R0.pred[exI], feats('bank', false)[exI]));
  prose(ch, '<p>Key terms:</p>');
  table(ch, [
    { t: 'Spot', m: 'One measured position on the section, with an expression vector (here a 6×6 grid per patient)', s: `${N} spots, ${pats.length} patients` },
    { t: 'Patch', m: 'The H&E image cropped around a spot (HEST: 224×224 pixels at a fixed magnification)', s: `${S}×${S} simulated pixels` },
    { t: 'Encoder / foundation model', m: 'A network pretrained on many images, here used without further training ("frozen")', s: `${ENCK.length} fixed toy encoders` },
    { t: 'Embedding', m: 'The encoder output for one patch: a vector of numbers describing its appearance', s: `${FNAMES.bank.length} numbers per patch (filter bank)` },
    { t: 'Head / linear probe', m: 'The small trainable model on top of the frozen embedding', s: 'standardise → PCA → ridge' },
    { t: 'Pearson r (per gene)', m: 'Correlation across held-out spots between predicted and measured expression of one gene', s: `mean r ${f2(R0.mean)} over ${GENES.length} genes` },
    { t: 'Held-out patient', m: 'A patient none of whose spots were used for training; the unit of a fair split', s: 'step 3' },
  ], [['t', 'Term'], ['m', 'Meaning'], ['s', 'In this toy']]);

  // =====================================================================================
  // 2. WHY
  // =====================================================================================
  ch = chapter(main, 'why', 'Why do we need it?', 'Expression is informative, but measuring it spatially is not routine, whereas histology is.');
  prose(ch, `<p>Three practical constraints motivate the approach:</p><ol>
    <li><b>Availability.</b> Spatial transcriptomics is run on selected samples; H&E slides exist for nearly every biopsy and resection, including archives with long follow-up.</li>
    <li><b>Resolution.</b> Bulk RNA-seq averages over the whole sample and discards where in the tissue a signal arises (tumour nest, stroma, lymphoid aggregate).</li>
    <li><b>Triage and hypothesis generation.</b> A predicted map can indicate which sections justify a full spatial assay, or which cohort members to examine first.</li></ol>`);
  table(ch, [
    { a: 'Bulk RNA-seq', s: 'no', n: 'yes (tissue lysed)', q: 'gene level, averaged over the sample' },
    { a: 'Spatial transcriptomics', s: 'yes', n: 'yes (specialised assay)', q: 'measured; limited by the technology (capture, dropout)' },
    { a: '<b>H&E → predicted expression</b>', s: '<b>yes, at spot level</b>', n: '<b>no (existing slide)</b>', q: '<b>partial; depends on how visible the signal is</b>' },
  ], [['a', 'Approach'], ['s', 'Spatial?'], ['n', 'New assay needed?'], ['q', 'Gene-level accuracy']]);
  const pm = (g0, i) => d3.mean(d3.range(PER), (s) => R0.pred[i * PER + s].slice(g0, g0 + 6).reduce((a, b) => a + b, 0) / 6);
  const patImm = d3.range(NAB).map((i) => pm(0, i)), patCD8 = pats.slice(0, NAB).map((p) => p.cd8);
  worked(ch, 'A patient-level sanity check against an independent measurement',
    `<p>The study recorded CD8 T-cell abundance per patient. Averaging the held-out predictions of the six immune genes over each patient's spots gives a predicted immune score. Across the ${NAB} Centre A/B patients the correlation with CD8 % is <b>r = ${f2(pearson(patImm, patCD8))}</b>, and the correlation of the measured immune genes with CD8 % is ${f2(pearson(d3.range(NAB).map((i) => d3.mean(d3.range(PER), (s) => Y[i * PER + s].slice(0, 6).reduce((a, b) => a + b, 0) / 6)), patCD8))}. With n = ${NAB} this is a plausibility check, not evidence of accuracy; its role is to show that a predicted programme score can be compared with a separate assay.</p>`);

  // =====================================================================================
  // 3. WHERE
  // =====================================================================================
  ch = chapter(main, 'where', 'Where is it used?', 'Settings with abundant H&E, scarce spatial data and a question about broad tissue programmes.');
  cards(ch, [
    { tag: 'Cohort studies', title: 'Programme scores on archival slides', body: 'Train a head on a few matched Visium sections, then predict immune, stromal or proliferation programme scores for a large H&E-only cohort with outcome data. Scores of programmes, not single genes, are the realistic target.' },
    { tag: 'Triage', title: 'Which sections to sequence', body: 'Predicted maps can rank sections by predicted immune infiltration or tumour fraction before committing to a spatial assay.' },
    { tag: 'Benchmarking', title: 'Comparing pathology encoders', body: 'HEST-1k fixes the head and the split so that differences in Pearson r reflect the encoder (UNI, Prov-GigaPath, Virchow, H-optimus).' },
    { tag: 'Spatial domains', title: 'Morphology as an extra view', body: 'Image embeddings can be used as covariates when calling spatial domains or when deconvolving spots, even when single-gene prediction is weak.' },
    { tag: 'Quality control', title: 'Consistency checks', body: 'Where predicted and measured programme scores disagree strongly within a section, the discrepancy can flag a technical problem or a signal that morphology does not carry.' },
    { tag: 'Training data', title: 'Paired image–expression corpora', body: `Large paired collections (HEST-1k; industry sets such as Owkin ${cite('mosaic', 'MOSAIC')} and Noetik ${cite('noetik', 'OCTO-vc')}) are what multimodal models, such as M-Optimus-1, are trained on. Performance statements for those are announcement-level.` },
  ]);
  prose(ch, '<p><b>When not to use it:</b> the target is a single gene that must be quantified accurately (measure it); the signal is not expressed in morphology (step 4); the slides come from a scanner or stain not represented in training and no external check is possible (step 5); or a clinical decision would rest on the prediction without prospective validation.</p>');

  // =====================================================================================
  // 4. HOW
  // =====================================================================================
  ch = chapter(main, 'how', 'How it works', 'Six steps: from a patch to an embedding, through a linear probe and a fair split, to what is predictable and how context from neighbouring spots helps.');

  // ---- Step 1
  let rb = rung(ch, 1, 'A frozen encoder turns a patch into numbers',
    `<p>The toy encoders are fixed functions of the ${S}×${S} patch (no weights are learned from expression). Each spot mixes three hidden tissue states: <b>immune</b> (many small dark nuclei), <b>tumour</b> (few large nuclei) and <b>stroma</b> (oriented fibres). The fourth gene group, signalling, depends on a state that leaves no mark in the image. Select a patch to see its embedding as z-scores relative to all spots.</p>`);
  const f1 = figure(rb, 'Patch and embedding', 'Left: nine patches, three most immune-rich, tumour-rich and stroma-rich (darker = more haematoxylin). Right: the selected patch\'s embedding, in standard deviations from the mean of all spots. The "true mix" is known only because the data are simulated.');
  toyBadge(f1, 'Hand-made filters stand in for a pathology foundation model.');
  let enc1 = 'bank', sel1 = 0;
  select(f1.controls, { label: 'Encoder', value: enc1, options: Object.entries(ENC), onChange: (v) => { enc1 = v; paint1(); } });
  const row1 = el('div', { class: 'fig-row' }); f1.body.append(row1);
  const l1 = el('div', { style: 'flex:0 1 420px' }), r1 = el('div'); row1.append(l1, r1);
  const pick = [0, 1, 2].flatMap((k) => d3.range(NA).map((i) => [i, spots[i].z[k] - (spots[i].z[0] + spots[i].z[1] + spots[i].z[2] - spots[i].z[k]) / 2]).sort((a, b) => b[1] - a[1]).slice(0, 3).map(([i]) => [i, k]));
  const gal = el('div', { style: 'display:grid;grid-template-columns:repeat(3,64px);gap:8px;justify-content:start' }), big = el('div'), info = el('div', { class: 'fig-caption', style: 'margin:6px 0 0' });
  const cvs = pick.map(([i, k], j) => { const cv = el('canvas', { width: S, height: S, style: 'width:64px;height:64px;image-rendering:pixelated;border-radius:4px;cursor:pointer;border:2px solid transparent', title: `${['immune', 'tumour', 'stroma'][k]}-rich` }); drawPatch(cv, spots[i].img); cv.onclick = () => { sel1 = j; paint1(); }; gal.append(cv); return cv; });
  const bigCv = el('canvas', { width: S, height: S, style: 'width:128px;height:128px;image-rendering:pixelated;border-radius:6px;border:1px solid var(--border)' });
  big.append(bigCv, info); l1.append(gal, el('div', { style: 'height:8px' }), big);
  const F1 = frame(r1, { width: 460, height: 300, x: [-3, 3], y: [0, 1], xLabel: 'embedding value (SD from the mean of all spots)', yScale: d3.scaleBand().padding(0.25), margin: { left: 160 }, grid: false });
  function paint1() {
    const i = pick[sel1][0], Fm = feats(enc1, false), p = Fm[0].length, mu = d3.range(p).map((j) => d3.mean(Fm, (r) => r[j])), sg = d3.range(p).map((j) => d3.deviation(Fm, (r) => r[j]) || 1);
    const zs = d3.range(p).map((j) => (Fm[i][j] - mu[j]) / sg[j]);
    cvs.forEach((cv, j) => { cv.style.borderColor = j === sel1 ? ink() : 'transparent'; });
    drawPatch(bigCv, spots[i].img);
    info.innerHTML = `${spots[i].pt.id} · Centre ${spots[i].pt.centre.slice(-1)} · ${spots[i].pt.phenotype}<br>true mix: immune ${pc(spots[i].z[0])}, tumour ${pc(spots[i].z[1])}, stroma ${pc(spots[i].z[2])}`;
    F1.y.domain(FNAMES[enc1]).range([0, F1.h]);
    F1.g.select('.axis.y').remove(); F1.g.append('g').attr('class', 'axis y').call(d3.axisLeft(F1.y).tickSize(0)).selectAll('text').attr('font-size', 11);
    F1.layer('z').selectAll('line').data([0]).join('line').attr('x1', F1.x(0)).attr('x2', F1.x(0)).attr('y1', 0).attr('y2', F1.h).attr('stroke', muted());
    F1.layer('b').selectAll('rect').data(zs).join('rect').attr('y', (_, j) => F1.y(FNAMES[enc1][j])).attr('height', F1.y.bandwidth()).attr('x', (v) => F1.x(Math.min(0, clamp(v, -3, 3)))).attr('width', (v) => Math.abs(F1.x(clamp(v, -3, 3)) - F1.x(0))).attr('fill', cat(6))
      .on('mousemove', (e, v) => tip(e, `z-score ${f2(v)}`)).on('mouseleave', untip);
  }
  paint1();
  const imRich = pick.slice(0, 3).map(([i]) => spots[i].z[0]);
  prose(rb, `<p>The immune-rich patches (z<sub>immune</sub> = ${imRich.map((v) => pc(v)).join(', ')}) show many small nuclei, which the filter bank registers as strong fine-scale blob responses. A good encoder produces embeddings in which the tissue states are linearly separable; it does not know which genes will be requested later.</p>`);

  // ---- Step 2
  rb = rung(ch, 2, 'The probe: PCA, then ridge regression per gene',
    `<p>The embedding is standardised and reduced to a few principal components (PCA), fitted on training spots only. For each gene, ridge regression then fits expression as a weighted sum of the components, with a penalty λ on the weights:</p>
     <div class="calc">ŷ<sub>g</sub> = b<sub>g</sub> + w<sub>g</sub> · z        w<sub>g</sub> = argmin ‖y<sub>g</sub> − Z w‖² + λ‖w‖²</div>
     <p>The head stays small so that differences between encoders reflect the encoder. Below, every point is a held-out spot (its patient was not used for training); settings are 6 components, λ = 1, filter bank.</p>`);
  const f2g = figure(rb, 'Predicted vs measured expression, held-out patients', 'Left: one point per spot. Right: one point per patient (mean over its spots). Colour: the patient\'s tumour phenotype in the study. r is the Pearson correlation.');
  toyBadge(f2g);
  let gSel = 0;
  select(f2g.controls, { label: 'Gene', value: '0', options: GENES.map((g, i) => [String(i), `${g} (${GSHORT[Math.floor(i / 6)]})`]), onChange: (v) => { gSel = +v; paint2(); } });
  const s2 = stats(f2g.root);
  const row2 = el('div', { class: 'fig-row' }); f2g.body.append(row2);
  const a2 = el('div'), b2 = el('div'); row2.append(a2, b2);
  const FA = frame(a2, { width: 480, height: 330, x: [0, 1], y: [0, 1], xLabel: 'measured expression', yLabel: 'predicted expression' });
  const FB = frame(b2, { width: 480, height: 330, x: [0, 1], y: [0, 1], xLabel: 'measured, patient mean', yLabel: 'predicted, patient mean' });
  const txtA = FA.g.append('text').attr('x', 8).attr('y', 16).attr('font-weight', 700).attr('font-size', 13).attr('fill', ink()), txtB = FB.g.append('text').attr('x', 8).attr('y', 16).attr('font-weight', 700).attr('font-size', 13).attr('fill', ink());
  function paint2() {
    const g = gSel, ev = R0.ev, mx = ev.map((i) => Y[i][g]), py = ev.map((i) => R0.pred[i][g]);
    const pm2 = d3.range(NAB).map((p) => ({ p, m: d3.mean(d3.range(PER), (s) => Y[p * PER + s][g]), q: d3.mean(d3.range(PER), (s) => R0.pred[p * PER + s][g]) }));
    const ext = (a, b) => { const lo = Math.min(d3.min(a), d3.min(b)), hi = Math.max(d3.max(a), d3.max(b)), pad = (hi - lo) * 0.06; return [lo - pad, hi + pad]; };
    FA.x.domain(d3.extent(mx)); FA.y.domain(d3.extent(py)); FB.x.domain(ext(pm2.map((d) => d.m), pm2.map((d) => d.m))); FB.y.domain(ext(pm2.map((d) => d.q), pm2.map((d) => d.q)));
    [[FA, 'm'], [FB, 'm']].forEach(([F]) => { F.g.select('.axis.x').call(d3.axisBottom(F.x).ticks(5)); F.g.select('.axis.y').call(d3.axisLeft(F.y).ticks(5)); });
    FA.layer('p').selectAll('circle').data(ev).join('circle').attr('cx', (i) => FA.x(Y[i][g])).attr('cy', (i) => FA.y(R0.pred[i][g])).attr('r', 3.6).attr('fill', (i) => PHC[spots[i].pt.phenotype]).attr('fill-opacity', 0.7).attr('stroke', css('--surface')).attr('stroke-width', 0.7)
      .on('mousemove', (e, i) => tip(e, `${spots[i].pt.id} · spot ${spots[i].s + 1}<br>measured ${f2(Y[i][g])}, predicted ${f2(R0.pred[i][g])}`)).on('mouseleave', untip);
    FB.layer('p').selectAll('circle').data(pm2).join('circle').attr('cx', (d) => FB.x(d.m)).attr('cy', (d) => FB.y(d.q)).attr('r', 6).attr('fill', (d) => PHC[pats[d.p].phenotype]).attr('stroke', css('--surface')).attr('stroke-width', 1.5)
      .on('mousemove', (e, d) => tip(e, `${pats[d.p].id} (${pats[d.p].centre}, ${pats[d.p].phenotype})<br>measured ${f2(d.m)}, predicted ${f2(d.q)}`)).on('mouseleave', untip);
    const rS = R0.r[g], rP = pearson(pm2.map((d) => d.m), pm2.map((d) => d.q));
    txtA.text(`spots: r = ${f2(rS)}`); txtB.text(`patients (n = ${NAB}): r = ${f2(rP)}`);
    s2.set({ gene: `${GENES[g]} (${GSHORT[Math.floor(g / 6)]})`, 'r over spots': f2(rS), 'r within patients': f2(R0.rw[g]), 'r over patient means': f2(rP) });
  }
  paint2(); phLegend(f2g.body);
  prose(rb, `<p>Across the ${GENES.length} genes the mean held-out r is <b>${f2(R0.mean)}</b>: ${GROUPS.map((gname, k) => `${f2(R0.g[k])} for ${gname.toLowerCase()}`).join(', ')}. For a gene in a visible group, most of the agreement comes from between-patient differences (the clusters in the right panel); the within-patient r, which removes each patient's mean, is lower (mean ${f2(R0.meanW)}). Step 4 returns to this.</p>`);

  // ---- Step 3
  rb = rung(ch, 3, 'Splitting by patient, not by spot',
    `<p>Spots from one patient share a stain, a genotype and a local environment, and neighbouring spots overlap. A random split of spots places each patient on both sides of the split, so a model can score well by recognising the <i>patient</i> (for instance through stain intensity) and recalling that patient's expression offsets. HEST therefore splits by patient. The diagram marks the test spots of fold 1; the bars compare the two protocols on the same data.</p>`);
  const f3 = figure(rb, 'Two ways to split the same spots', 'Left: rows are patients, columns are the 36 spots of each; blue cells are the test spots of fold 1 and grey cells are used for training. Right: mean held-out r per gene group for the two protocols. Hover for values.');
  toyBadge(f3);
  let enc3 = 'bank', mode3 = 'patient';
  select(f3.controls, { label: 'Encoder', value: enc3, options: Object.entries(ENC), onChange: (v) => { enc3 = v; paint3(); } });
  select(f3.controls, { label: 'Fold diagram', value: mode3, options: [['patient', 'split by patient (HEST-style)'], ['spot', 'random split of spots']], onChange: (v) => { mode3 = v; paint3(); } });
  const s3 = stats(f3.root);
  const row3 = el('div', { class: 'fig-row' }); f3.body.append(row3);
  const a3 = el('div'), b3 = el('div'); row3.append(a3, b3);
  const FD = frame(a3, { width: 480, height: 300, x: [0, PER], y: [0, NAB], xLabel: 'spots (6×6 grid per patient, left to right)', axes: false, grid: false, margin: { left: 84, bottom: 36 } });
  const cw3 = FD.w / PER, ch3 = FD.h / NAB;
  pats.slice(0, NAB).forEach((p, i) => FD.g.append('text').attr('x', -6).attr('y', i * ch3 + ch3 / 2 + 4).attr('text-anchor', 'end').attr('font-size', 10.5).attr('fill', ink2()).text(`${p.id} · ${p.centre.slice(-1)}`));
  const FG = frame(b3, { width: 480, height: 300, x: [0, 1], y: [0, 1], yLabel: 'mean Pearson r', axes: false, margin: { bottom: 40 } });
  const gx = d3.scaleBand().domain(d3.range(4)).range([0, FG.w]).padding(0.25), gx2 = d3.scaleBand().domain([0, 1]).range([0, gx.bandwidth()]).padding(0.08);
  const SPL = [['patient', 'split by patient', cat(5)], ['spot', 'random split of spots', cat(6)]];
  function paint3() {
    const fold = (i) => (mode3 === 'patient' ? spots[i].p % 4 : shuffled.indexOf(i) % 4);
    FD.layer('cells').selectAll('rect').data(d3.range(NA)).join('rect').attr('x', (i) => (i % PER) * cw3).attr('y', (i) => Math.floor(i / PER) * ch3).attr('width', cw3 - 0.6).attr('height', ch3 - 1.4)
      .attr('fill', (i) => (fold(i) === 0 ? cat(0) : css('--surface-2'))).attr('stroke', (i) => (fold(i) === 0 ? 'none' : css('--border')))
      .on('mousemove', (e, i) => tip(e, `${spots[i].pt.id}, spot ${spots[i].s + 1}<br>${fold(i) === 0 ? 'test (fold 1)' : 'training'}`)).on('mouseleave', untip);
    const rr = SPL.map(([m]) => res({ enc: enc3, mode: m }));
    const maxr = Math.max(0.5, d3.max(rr, (q) => d3.max(q.g)) * 1.15), ys = d3.scaleLinear().domain([Math.min(0, d3.min(rr, (q) => d3.min(q.g))) - 0.02, maxr]).range([FG.h, 0]);
    FG.g.selectAll('.ax3').remove();
    FG.g.append('g').attr('class', 'axis y ax3').call(d3.axisLeft(ys).ticks(6));
    FG.g.select('.gridline').remove();
    FG.layer('x').selectAll('text').data(GSHORT).join('text').attr('x', (_, k) => gx(k) + gx.bandwidth() / 2).attr('y', FG.h + 18).attr('text-anchor', 'middle').attr('font-size', 12).attr('fill', ink2()).text((t) => t);
    FG.layer('zero').selectAll('line').data([0]).join('line').attr('x1', 0).attr('x2', FG.w).attr('y1', ys(0)).attr('y2', ys(0)).attr('stroke', ink2());
    const data = rr.flatMap((q, m) => q.g.map((v, k) => ({ v, k, m })));
    FG.layer('bars').selectAll('rect').data(data).join('rect').attr('x', (d) => gx(d.k) + gx2(d.m)).attr('width', gx2.bandwidth()).attr('fill', (d) => SPL[d.m][2]).attr('y', (d) => ys(Math.max(0, d.v))).attr('height', (d) => Math.abs(ys(d.v) - ys(0)))
      .on('mousemove', (e, d) => tip(e, `${SPL[d.m][1]}<br>${GROUPS[d.k]}: mean r = ${f2(d.v)}`)).on('mouseleave', untip);
    FG.layer('val').selectAll('text').data(data).join('text').attr('x', (d) => gx(d.k) + gx2(d.m) + gx2.bandwidth() / 2).attr('y', (d) => ys(Math.max(0, d.v)) - 4).attr('text-anchor', 'middle').attr('font-size', 10.5).attr('fill', ink2()).text((d) => f2(d.v));
    s3.set({ 'mean r, by patient': f2(rr[0].mean), 'mean r, random spots': f2(rr[1].mean), 'signalling genes, by patient': f2(rr[0].g[3]), 'signalling genes, random spots': f2(rr[1].g[3]) });
  }
  paint3(); legend(f3.body, [...SPL.map(([, t, c]) => [t, c]), ['test spots, fold 1', cat(0)]]);
  const rsR = res({ enc: 'rand', mode: 'spot' }), rpR = res({ enc: 'rand' });
  prose(rb, `<p>With the filter bank the random split raises the mean r from ${f2(R0.mean)} to ${f2(Rs.mean)}, and the signalling genes, which have no image signal, from ${f2(R0.g[3])} to ${f2(Rs.g[3])}. With the random-filter encoder the signalling genes go from ${f2(rpR.g[3])} to ${f2(rsR.g[3])}. No additional biology was learned: the extra correlation is patient identity (through stain and per-patient offsets) being recalled.</p>`);

  // ---- Step 4
  const brHi = bandPooled;
  rb = rung(ch, 4, 'What is and is not predictable',
    `<p>Per-gene results (patient split, filter bank, 6 components, λ = 1) separate into genes whose variation is expressed in tissue appearance and genes whose variation is not. The shaded band is an empirical null: the 95th percentile of |r| after shuffling whole patients' expression profiles (pooled) or spots within each patient (within-patient), repeated 120 times. With only ${NAB} patients the pooled band is wide.</p>`);
  const f4 = figure(rb, 'Per-gene held-out Pearson r', 'Bars: held-out r for each gene. Shaded band: values reached by chance. "Pooled" correlates all held-out spots together; "within patient" first subtracts each patient\'s mean from prediction and measurement.');
  toyBadge(f4);
  let mode4 = 'pooled';
  select(f4.controls, { label: 'Correlation', value: mode4, options: [['pooled', 'pooled over all spots'], ['within', 'within patient (mean removed)']], onChange: (v) => { mode4 = v; paint4(); } });
  const s4 = stats(f4.root);
  const FP = geneBars(f4.body, { width: 820, height: 320, names: true });
  const noteP = FP.g.append('text').attr('x', 10).attr('y', 16).attr('font-size', 12).attr('fill', ink());
  function paint4() {
    const w = mode4 === 'within', r = w ? R0.rw : R0.r, band = w ? bandWithin : brHi;
    FP.draw(r, { band, bandLabel: `chance band (95%): ±${f2(band)}` });
    const above = (k) => r.slice(6 * k, 6 * k + 6).filter((v) => v > band).length;
    noteP.text(w ? 'Within patient: variation between spots of one tumour only' : 'Pooled: includes differences between patients');
    s4.set({ 'immune genes above band': `${above(0)}/6`, 'tumour genes above band': `${above(1)}/6`, 'stroma genes above band': `${above(2)}/6`, 'signalling genes above band': `${above(3)}/6`, 'mean r': f2(mean(r)) });
  }
  paint4(); groupLegend(f4.body);
  prose(rb, `<p>Immune, tumour and stroma genes follow what the patch shows and reach pooled r up to ${f2(d3.max(R0.r.slice(0, 18)))}; the signalling genes, whose state is not visible, reach ${f2(d3.max(R0.r.slice(18)))} at most. The pooled r of the signalling genes is not exactly zero (mean ${f2(R0.g[3])}) because in the simulation patients with high IFN-γ have both more immune cells and higher signalling, so the image predicts the signalling genes <i>between</i> patients; the within-patient mean falls to ${f2(R0.gw[3])}. The ${cite('hest', 'HEST-1k')} ceiling of about 0.41 for the best encoder on 50 highly variable genes is a mean over genes of this kind, which is why per-gene results should be read, not only the mean.</p>`);
  quiz(rb, 'A signalling gene shows pooled r = ' + f2(R0.r[18]) + ' but within-patient r = ' + f2(R0.rw[18]) + '. What can be concluded?', 'The image carries information about which <i>patients</i> have higher expression of the gene (through a correlated feature such as immune infiltration), but little about which <i>spots within a tumour</i> do. For spatial questions the within-patient number is the relevant one.');
  quiz(rb, 'Why does the chance band in the "within patient" view differ from the pooled one?', 'Pooled correlation can be created or destroyed by 12 patient-level values, so the null (shuffling whole patients) is wide. Within-patient correlation uses all 36 spots per patient, so its null is narrower.');

  // ---- Step 5: lab
  rb = rung(ch, 5, 'Model builder: encoder, head, split and stain',
    `<p>Choose an encoder and head, a split protocol and whether the patches are stain-normalised (each patient's pixels re-centred and re-scaled to the pooled mean and SD). Centres differ in stain intensity (A lighter, B darker, and C darker still with higher contrast, as from a different scanner). The <i>new-centre</i> split trains on the ${NAB} Centre A/B patients and tests on the ${pats.length - NAB} Centre C patients.</p>`);
  const fl = figure(rb, 'Model builder', 'Left: held-out r for every gene under the chosen settings. Right: mean r by gene group for the three encoders under the same settings (the selected one is full colour).');
  toyBadge(fl, 'The r values are not HEST numbers.');
  const st = { enc: 'bank', nPC: 6, lam: 1, mode: 'patient', norm: false };
  select(fl.controls, { label: 'Encoder', value: st.enc, options: Object.entries(ENC), onChange: (v) => { st.enc = v; paintL(); } });
  slider(fl.controls, { label: 'PCA components', min: 1, max: 8, step: 1, value: st.nPC, onInput: (v) => { st.nPC = v; paintL(); } });
  slider(fl.controls, { label: 'Ridge λ', min: -2, max: 3, step: 0.25, value: 0, fmt: (v) => (10 ** v).toPrecision(2), onInput: (v) => { st.lam = 10 ** v; paintL(); } });
  select(fl.controls, { label: 'Split', value: st.mode, options: [['patient', 'by patient (4 folds)'], ['spot', 'random spots (leaky)'], ['centre', 'new centre (train A+B, test C)']], onChange: (v) => { st.mode = v; paintL(); } });
  checkbox(fl.controls, { label: 'Stain-normalise patches', checked: false, onChange: (v) => { st.norm = v; paintL(); } });
  const sL = stats(fl.root);
  const rowL = el('div', { class: 'fig-row' }); fl.body.append(rowL);
  const aL = el('div'), bL = el('div'); rowL.append(aL, bL);
  const FL = geneBars(aL, { width: 500, height: 300, names: false });
  const FE = frame(bL, { width: 440, height: 300, x: [0, 1], y: [0, 1], yLabel: 'mean Pearson r', axes: false, margin: { bottom: 40 } });
  const ex = d3.scaleBand().domain(d3.range(4)).range([0, FE.w]).padding(0.2), ex2 = d3.scaleBand().domain(d3.range(3)).range([0, ex.bandwidth()]).padding(0.06), EC = [cat(5), cat(6), cat(7)];
  function paintL() {
    const o = { norm: st.norm, nPC: st.nPC, lam: st.lam, mode: st.mode }, cur = res({ enc: st.enc, ...o }), all = ENCK.map((e) => res({ enc: e, ...o }));
    FL.draw(cur.r);
    const ys = d3.scaleLinear().domain([Math.min(-0.1, d3.min(all, (q) => d3.min(q.g)) - 0.02), 1]).range([FE.h, 0]);
    FE.g.selectAll('.axl').remove(); FE.g.append('g').attr('class', 'axis y axl').call(d3.axisLeft(ys).ticks(6)); FE.g.select('.gridline').remove();
    FE.layer('x').selectAll('text').data(GSHORT).join('text').attr('x', (_, k) => ex(k) + ex.bandwidth() / 2).attr('y', FE.h + 18).attr('text-anchor', 'middle').attr('font-size', 12).attr('fill', ink2()).text((t) => t);
    FE.layer('zero').selectAll('line').data([0]).join('line').attr('x1', 0).attr('x2', FE.w).attr('y1', ys(0)).attr('y2', ys(0)).attr('stroke', ink2());
    const data = all.flatMap((q, m) => q.g.map((v, k) => ({ v, k, m })));
    FE.layer('bars').selectAll('rect').data(data).join('rect').attr('x', (d) => ex(d.k) + ex2(d.m)).attr('width', ex2.bandwidth()).attr('fill', (d) => EC[d.m]).attr('fill-opacity', (d) => (ENCK[d.m] === st.enc ? 1 : 0.5))
      .attr('y', (d) => ys(Math.max(0, d.v))).attr('height', (d) => Math.abs(ys(d.v) - ys(0)))
      .on('mousemove', (e, d) => tip(e, `${ENC[ENCK[d.m]]}<br>${GROUPS[d.k]}: mean r = ${f2(d.v)}`)).on('mouseleave', untip);
    sL.set({ 'mean r': f2(cur.mean), 'immune / tumour / stroma': cur.g.slice(0, 3).map(f2).join(' / '), 'signalling genes': f2(cur.g[3]), 'within-patient mean r': f2(cur.meanW), 'centre readable from features (AUC)': centreAUC(st.enc, st.norm).toFixed(2) });
  }
  paintL(); legend(fl.body, [...ENCK.map((e, i) => [ENC[e], EC[i]]), ['gene colours: as in step 4', ink2()]]);
  const nb = res({ norm: true }), nrR = res({ enc: 'rand', norm: true }), cNorm = res({ mode: 'centre', norm: true }), cHist = res({ enc: 'hist' }), p1 = res({ nPC: 1 }), p8 = res({ nPC: 8 }), l3 = res({ lam: 1000 });
  prose(rb, '<p><b>Suggested comparisons</b>:</p>');
  table(rb, [
    { t: '<b>Encoder</b>: filter bank → histogram only', s: `Mean r ${f2(R0.mean)} → <b>${f2(cHist.mean)}</b>; filter bank → random filters: ${f2(rpR.mean)}`, m: 'With the head fixed, the encoder determines how much of the image signal reaches the regression. This is the comparison HEST makes between pathology models.' },
    { t: '<b>Split</b>: by patient → random spots', s: `Mean r ${f2(R0.mean)} → <b>${f2(Rs.mean)}</b>; signalling ${f2(R0.g[3])} → <b>${f2(Rs.g[3])}</b>`, m: 'The leaky protocol rewards recognising the patient. The increase is largest where there is no image signal.' },
    { t: '<b>Stain-normalise</b> (patient split, random filters)', s: `Centre readable from features: AUC ${centreAUC('rand', false).toFixed(2)} → <b>${centreAUC('rand', true).toFixed(2)}</b>; mean r ${f2(rpR.mean)} → <b>${f2(nrR.mean)}</b>`, m: `Normalisation removes much of the stain signal. It cannot go to 0.5, because Centre B also has more inflamed tumours in the study, so some centre information is real tissue content.` },
    { t: '<b>New centre</b> (filter bank)', s: `Mean r ${f2(R0.mean)} (patient split) → <b>${f2(Rc.mean)}</b> (Centre C); average per-gene bias ${f2(R0.bias)} → <b>${f2(Rc.bias)}</b>, and ${f2(cNorm.bias)} with stain normalisation`, m: `Pearson r ignores a shift that moves every prediction at the new site by the same amount, so r can hold up while the predicted levels are wrong. With only ${pats.length - NAB} test patients the estimate is also noisy. Check bias or calibration on an unseen centre, not only r.` },
    { t: '<b>PCA components</b>: 1 → 8 (filter bank)', s: `Mean r ${f2(p1.mean)} → ${f2(p8.mean)} (6 components: ${f2(R0.mean)})`, m: 'Three tissue states need at least three directions; more components add little once those are captured, and eventually add noise.' },
    { t: '<b>Ridge λ</b>: 1 → 1000', s: `Mean r ${f2(R0.mean)} → ${f2(l3.mean)}`, m: 'Strong shrinkage flattens predictions towards the mean. Pearson r ignores scale, so it falls only when the ranking of spots degrades; check calibration if absolute values matter.' },
  ], [['t', 'Try'], ['s', 'What you see'], ['m', 'What it means']], { cls: 'compact' });

  // ---- Step 6: spatial context
  rb = rung(ch, 6, 'Spatial context: neighbours and niche models',
    `<p>A spot's expression is not independent of its neighbours: tissue states form regions. Models that use this context average or attend over the spatial neighbour graph. The figure applies the simplest version to the step-2 predictions: each spot's prediction is replaced by the mean over itself and its 8 grid neighbours, repeated for the chosen number of hops.</p>`);
  const nbrs = spots.map((s) => spots.filter((t) => t.p === s.p && t !== s && Math.abs(t.gi - s.gi) <= 1 && Math.abs(t.gj - s.gj) <= 1).map((t) => spots.indexOf(t)));
  const smoothAt = [R0.pred];
  for (let h = 1; h <= MAXHOP; h++) smoothAt.push(smoothAt[h - 1].map((v, i) => (i < NA ? v.map((x, g) => (x + nbrs[i].reduce((s, j) => s + smoothAt[h - 1][j][g], 0)) / (1 + nbrs[i].length)) : v)));
  const rHop = smoothAt.map((P) => { const rr = GENES.map((_, g) => rGene(P, R0.ev, g, false)); return d3.range(4).map((q) => mean(rr.slice(6 * q, 6 * q + 6))); });
  const f6 = figure(rb, 'Smoothing predictions over the spot graph', 'Left: measured expression and smoothed prediction for one patient and gene (each map scaled to its own range). Right: mean held-out r by gene group against the number of hops; 0 hops is the step-2 result.');
  toyBadge(f6, 'Fixed mean aggregation; real niche models learn attention weights.');
  let hop = 0, pSel = 3, gSel6 = 0;
  select(f6.controls, { label: 'Patient', value: '3', options: pats.slice(0, NAB).map((p, i) => [String(i), `${p.id} (${p.centre.slice(-1)}, ${p.phenotype})`]), onChange: (v) => { pSel = +v; paint6(); } });
  select(f6.controls, { label: 'Gene', value: '0', options: GENES.map((g, i) => [String(i), `${g} (${GSHORT[Math.floor(i / 6)]})`]), onChange: (v) => { gSel6 = +v; paint6(); } });
  slider(f6.controls, { label: 'Hops', min: 0, max: MAXHOP, step: 1, value: 0, onInput: (v) => { hop = v; paint6(); } });
  const s6 = stats(f6.root);
  const row6 = el('div', { class: 'fig-row' }); f6.body.append(row6);
  const a6 = el('div'), b6 = el('div'); row6.append(a6, b6);
  const CS = 36, svg6 = d3.select(a6).append('svg').attr('viewBox', '0 0 480 300').attr('width', 480).attr('role', 'img').attr('aria-label', 'measured and smoothed predicted maps');
  const FH = frame(b6, { width: 480, height: 300, x: [0, MAXHOP], y: [0, 1], xLabel: 'hops of neighbour averaging', yLabel: 'mean Pearson r', xTicks: MAXHOP });
  FH.layer('l'); GROUPS.forEach((_, k) => FH.layer('l').append('path').datum(rHop.map((v, h) => [h, v[k]])).attr('fill', 'none').attr('stroke', cat(k)).attr('stroke-width', 2.2).attr('d', d3.line().x((d) => FH.x(d[0])).y((d) => FH.y(d[1]))));
  FH.layer('dots').selectAll('circle').data(GROUPS.flatMap((_, k) => rHop.map((v, h) => ({ h, k, v: v[k] })))).join('circle').attr('cx', (d) => FH.x(d.h)).attr('cy', (d) => FH.y(d.v)).attr('r', 3.6).attr('fill', (d) => cat(d.k)).attr('stroke', css('--surface'))
    .on('mousemove', (e, d) => tip(e, `${GROUPS[d.k]}, ${d.h} hops<br>mean r = ${f2(d.v)}`)).on('mouseleave', untip);
  const hopL = FH.layer('hop');
  function paint6() {
    const base = pSel * PER, meas = d3.range(PER).map((s) => Y[base + s][gSel6]), prd = d3.range(PER).map((s) => smoothAt[hop][base + s][gSel6]);
    svg6.selectAll('*').remove();
    [[meas, 'measured', 10], [prd, `predicted, ${hop} hop${hop === 1 ? '' : 's'}`, 250]].forEach(([v, lab, x0]) => {
      const sc = sequential(d3.min(v), d3.max(v)), g = svg6.append('g').attr('transform', `translate(${x0},40)`);
      g.append('text').attr('x', 0).attr('y', -14).attr('font-size', 13).attr('font-weight', 700).attr('fill', ink()).text(lab);
      g.selectAll('rect').data(v).join('rect').attr('x', (_, s) => (s % GRID) * CS).attr('y', (_, s) => Math.floor(s / GRID) * CS).attr('width', CS - 2).attr('height', CS - 2).attr('rx', 3).attr('fill', (d) => sc(d)).attr('stroke', css('--border'))
        .on('mousemove', (e, d) => tip(e, `${lab}: ${f2(d)}`)).on('mouseleave', untip);
    });
    svg6.append('text').attr('x', 10).attr('y', 290).attr('font-size', 11.5).attr('fill', muted()).text('each map scaled to its own range; spots on a 6×6 grid');
    hopL.selectAll('line').data([hop]).join('line').attr('x1', (h) => FH.x(h)).attr('x2', (h) => FH.x(h)).attr('y1', 0).attr('y2', FH.h).attr('stroke', ink2()).attr('stroke-dasharray', '4 3');
    s6.set({ 'this gene, spots of this patient': f2(pearson(meas, prd)), 'gene, all patients (0 hops → now)': `${f2(R0.r[gSel6])} → ${f2(rGene(smoothAt[hop], R0.ev, gSel6, false))}`, 'immune genes, mean r': f2(rHop[hop][0]), 'signalling genes, mean r': f2(rHop[hop][3]) });
  }
  paint6(); groupLegend(f6.body);
  const best = rHop.map((v) => v[0]).indexOf(d3.max(rHop, (v) => v[0]));
  prose(rb, `<p>In the toy, one hop raises the immune genes' mean r from ${f2(rHop[0][0])} to ${f2(rHop[1][0])}, and the best value is ${f2(rHop[best][0])} at ${best} hop${best === 1 ? '' : 's'}; after that the curve ${rHop[MAXHOP][0] < rHop[best][0] ? 'declines' : 'does not decline'} as the averaging blurs the real regional structure. The simulated fields are smooth by construction, so the gain is built in; in tissue with sharp boundaries over-smoothing costs resolution earlier. The signalling genes, whose signal is not in the image, gain only to the extent that their between-patient component survives.</p>
    <p>Spatial foundation models learn this context rather than fixing it. <b>Novae</b> (${cite('novae', 'Nat Methods 2025')}) applies graph attention over cell neighbourhoods with self-supervised prototypes, giving spatial domains without labels. <b>Nicheformer</b> (${cite('nicheformer', 'Nat Methods 2025')}) is pretrained on SpatialCorpus-110M and transfers spatial context labels onto dissociated cells. <b>CellPLM</b> (ICLR 2024) treats cells as tokens and a tissue as a sentence, and reconstructs masked cells from their spatial neighbours. They operate on expression data and complement, rather than replace, image-based prediction.</p>`);

  // =====================================================================================
  // 5. PITFALLS
  // =====================================================================================
  ch = chapter(main, 'pitfalls', 'Pitfalls', 'Common problems, how to recognise them, and how to address them.');
  const nBelow = R0.r.filter((v) => v <= bandPooled).length;
  cards(ch, [
    { tag: 'Symptom: high r that does not hold on a new patient', title: 'Spots from the same patient in train and test', body: `A random split of spots gave a mean r of ${f2(Rs.mean)} against ${f2(R0.mean)} for a split by patient, and ${f2(Rs.g[3])} against ${f2(R0.g[3])} for the signalling genes. <b>Fix:</b> split by patient (or slide), and tune hyper-parameters inside the training patients only.` },
    { tag: 'Symptom: good r at a new site, wrong levels', title: 'Stain and scanner shift', body: `Trained on Centres A and B, the filter bank gives a mean r of ${f2(Rc.mean)} on Centre C (${f2(R0.mean)} with a patient split), but the predicted levels are off by ${f2(Rc.bias)} per gene on average, against ${f2(R0.bias)}. Pearson r does not see a uniform shift. Stain intensity is readable from the features (AUC up to ${centreAUC('rand', false).toFixed(2)} for the random filters). <b>Fix:</b> stain normalisation or augmentation, and an external-site test before use.` },
    { tag: 'Symptom: a good mean hides failing genes', title: 'The mean over genes', body: `${nBelow} of ${GENES.length} genes are inside the chance band in step 4, including the signalling genes. <b>Fix:</b> report per-gene (or per-programme) r with a null, and list the genes that fail.` },
    { tag: 'Symptom: maps look right but vary little within a tumour', title: 'Between-patient versus within-patient signal', body: `Mean pooled r is ${f2(R0.mean)}, within-patient r is ${f2(R0.meanW)}. A model can rank patients correctly and still not localise a signal inside a section. <b>Fix:</b> report both when the question is spatial.` },
    { tag: 'Symptom: over-optimistic confidence', title: 'Few patients, many spots', body: `${NA} spots from ${NAB} patients are not ${NA} independent observations; the pooled chance band is ±${f2(bandPooled)} here. <b>Fix:</b> count patients as the sample size and use patient-level resampling for intervals.` },
    { tag: 'Symptom: preprocessing fitted on everything', title: 'Leakage through standardisation and PCA', body: 'Standardising or running PCA on all spots before splitting lets test patients shape the features. <b>Fix:</b> fit every step on training patients inside each fold, as in the code below.' },
    { tag: 'Symptom: blurred, over-confident maps', title: 'Over-smoothing across neighbours', body: `Context helps up to a point (immune genes: ${f2(rHop[0][0])} → ${f2(rHop[best][0])} → ${f2(rHop[MAXHOP][0])} at ${MAXHOP} hops). With neighbours in both train and test, spatial autocorrelation leaks in the same way as a spot-level split. <b>Fix:</b> keep whole sections on one side of the split and choose the radius by validation.` },
    { tag: 'Symptom: a headline number from a press release', title: 'Vendor and self-reported results', body: `The HEST-1k figure of about 0.41 is from an independent benchmark with a fixed protocol; ${cite('moptimus', 'M-Optimus-1')} results are vendor claims and industry data-set announcements describe scale rather than accuracy. <b>Fix:</b> compare encoders under one protocol (same head, same patient split, same genes).` },
  ]);

  // =====================================================================================
  // 6. RECAP
  // =====================================================================================
  ch = chapter(main, 'recap', 'Recap', 'Key points.');
  recap(ch, [
    'H&E-to-expression prediction applies a <b>frozen pathology encoder</b> to the patch at each spot and fits a <b>small head</b> (PCA + ridge) from the embedding to expression.',
    'The HEST-1k benchmark fixes the head, the genes and the <b>patient-level split</b>, so that differences reflect the encoder. The best mean r reported there is about 0.41 on 50 highly variable genes.',
    `Genes whose variation is expressed in morphology (immune, tumour, stroma programmes) are predictable; a state that leaves no trace in the image is not (toy: signalling genes ${f2(R0.g[3])} against ${f2(mean(R0.g.slice(0, 3)))} for the visible groups).`,
    'Split by <b>patient or slide</b>. A random split of spots rewards recognising the patient and inflates r, most of all for genes with no image signal.',
    'Check <b>per-gene</b> results against a null, and report <b>within-patient</b> r when the question is where in a tumour a signal arises.',
    'Stain, scanner and site are confounders: normalise or augment, and test on an unseen centre.',
    'Neighbour context improves spot predictions up to a point. Novae, Nicheformer and CellPLM learn spatial context from expression data; they complement image-based prediction.',
    'Vendor and self-reported performance numbers need an independent benchmark before they are relied on.',
  ]);
  prose(ch, '<h3>In code</h3><p>The encoder is any frozen model that returns one vector per patch; the head below is the HEST-style probe with every step fitted inside the training patients.</p>');
  code(ch, {
    Python: `import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# X: (n_spots, d) frozen embeddings of the patch under each spot (pathology encoder, not trained here)
# Y: (n_spots, n_genes) log-normalised expression; patient: (n_spots,) patient or slide id
pred = np.zeros_like(Y)
for tr, te in GroupKFold(n_splits=4).split(X, Y, groups=patient):       # split by patient
    head = make_pipeline(StandardScaler(), PCA(n_components=32), Ridge(alpha=100.0))
    head.fit(X[tr], Y[tr])                                              # all steps fitted on training patients only
    pred[te] = head.predict(X[te])

r = np.array([np.corrcoef(pred[:, g], Y[:, g])[0, 1] for g in range(Y.shape[1])])
print("mean r", r.mean(), "genes with r > 0.1:", (r > 0.1).sum())

# within-patient r: remove each patient's mean from prediction and measurement
def demean(A): return A - np.vstack([A[patient == p].mean(0) for p in patient])
r_within = [np.corrcoef(demean(pred)[:, g], demean(Y)[:, g])[0, 1] for g in range(Y.shape[1])]`,
    R: `library(glmnet)

# X: spots x d frozen embeddings; Y: spots x genes (log-normalised); patient: factor of patient / slide ids
folds <- as.integer(factor(patient)) %% 4 + 1               # whole patients per fold
pred  <- matrix(NA_real_, nrow(Y), ncol(Y))
for (f in 1:4) {
  tr <- folds != f; te <- !tr
  pc <- prcomp(X[tr, ], center = TRUE, scale. = TRUE, rank. = 32)     # PCA on training patients only
  Ztr <- pc$x; Zte <- predict(pc, X[te, ])
  for (g in seq_len(ncol(Y))) {
    fit <- glmnet(Ztr, Y[tr, g], alpha = 0, lambda = 1)               # ridge
    pred[te, g] <- predict(fit, Zte)
  }
}
r <- sapply(seq_len(ncol(Y)), function(g) cor(pred[, g], Y[, g]))
summary(r)`,
  });
  seeAlso(main, [['frontier', 'foundation-models', 'Single-cell foundation models'], ['frontier', 'virtual-cells', 'Virtual cells'], ['frontier', 'landscape', 'The 2024–26 landscape'], ['concepts', 'pca', 'PCA'], ['concepts', 'regularization', 'Regularisation'], ['concepts', 'cross-validation', 'Cross-validation and leakage'], ['omics', 'sp-biomarker', 'Spatial biomarker']]);
}

// ---------------------------------------------------------------------------------------------
/** Diagram: patch → frozen encoder → embedding → head → predicted vs measured, for one simulated spot. */
function pipelineSvg(sp, pred, emb) {
  const wrap = el('div', { class: 'figure' });
  wrap.append(el('div', { class: 'fig-title' }, `One simulated spot (${sp.pt.id}, ${sp.pt.centre}, ${sp.pt.phenotype} tumour)`));
  wrap.append(html('<span class="tag" style="background:var(--accent-soft);color:var(--ink)">Teaching toy, not the paper\'s model or numbers</span>', 'div'));
  const svg = d3.select(wrap).append('svg').attr('viewBox', '0 0 980 250').attr('width', 980).attr('role', 'img').attr('aria-label', 'patch, frozen encoder, embedding, regression head, predicted versus measured expression');
  const BW = [170, 180, 190, 250], gap = 40, H = 220;
  const titles = ['① H&E patch', '② Frozen encoder', '③ Head: PCA → ridge', '④ Predicted vs measured'];
  let x0 = 10;
  BW.forEach((bw, i) => {
    const g = svg.append('g').attr('transform', `translate(${x0},14)`);
    g.append('rect').attr('width', bw).attr('height', H).attr('rx', 10).attr('fill', css('--surface-2')).attr('stroke', css('--axis'));
    g.append('text').attr('x', 12).attr('y', 24).attr('font-weight', 700).attr('font-size', 13).attr('fill', ink()).text(titles[i]);
    if (i === 0) {
      const sc = 7, interp = d3.interpolateRgb(css('--surface'), cat(6));
      sp.img.forEach((v, j) => g.append('rect').attr('x', 29 + (j % S) * sc).attr('y', 44 + Math.floor(j / S) * sc).attr('width', sc + 0.3).attr('height', sc + 0.3).attr('fill', interp(clamp(v / 1.1, 0, 1))));
      g.append('text').attr('x', 12).attr('y', 186).attr('font-size', 12).attr('fill', ink2()).text(`${S}×${S} pixels under one spot`);
      g.append('text').attr('x', 12).attr('y', 204).attr('font-size', 12).attr('fill', ink2()).text(`true mix: ${Math.round(sp.z[0] * 100)}% immune`);
    } else if (i === 1) {
      const sx = d3.scaleLinear().domain([-0.1, 1]).range([0, bw - 28]), mx = d3.max(emb), ys = d3.scaleBand().domain(d3.range(emb.length)).range([40, 168]).padding(0.25);
      g.selectAll('rect.e').data(emb).join('rect').attr('class', 'e').attr('x', 14).attr('y', (_, j) => ys(j)).attr('height', ys.bandwidth()).attr('width', (v) => Math.max(1, sx(v / mx))).attr('fill', cat(6));
      g.append('text').attr('x', 12).attr('y', 190).attr('font-size', 12).attr('fill', ink2()).text(`${emb.length} numbers, weights fixed`);
      g.append('text').attr('x', 12).attr('y', 207).attr('font-size', 12).attr('fill', ink2()).text('(bars scaled to the largest)');
    } else if (i === 2) {
      ['standardise (training mean, SD)', '↓', 'PCA: 6 components', '↓', 'ridge: one weight vector', 'per gene (24 genes)', '', 'fitted on training', 'patients only'].forEach((t, j) => g.append('text').attr('x', 14).attr('y', 54 + j * 18).attr('font-size', 12.5).attr('fill', j > 6 ? muted() : ink2()).text(t));
    } else {
      const reps = [0, 6, 12, 18], names = ['imm1', 'tum1', 'str1', 'sig1'];
      const vals = reps.flatMap((gi) => [pred[gi], sp.y[gi]]), lo = Math.min(0, d3.min(vals)), hi = d3.max(vals), sx = d3.scaleLinear().domain([lo, hi]).range([46, bw - 14]);
      reps.forEach((gi, j) => {
        const yy = 40 + j * 42;
        g.append('text').attr('x', 12).attr('y', yy + 15).attr('font-size', 12).attr('fill', ink2()).text(names[j]);
        g.append('rect').attr('x', sx(Math.min(0, pred[gi]))).attr('y', yy).attr('height', 14).attr('width', Math.abs(sx(pred[gi]) - sx(0))).attr('fill', cat(j));
        g.append('path').attr('d', d3.symbol(d3.symbolDiamond, 90)).attr('transform', `translate(${sx(sp.y[gi])},${yy + 24})`).attr('fill', ink()).attr('stroke', css('--surface'));
      });
      g.append('text').attr('x', 12).attr('y', 212).attr('font-size', 11.5).attr('fill', ink2()).text('bar = predicted, diamond = measured');
    }
    if (i < 3) svg.append('path').attr('d', `M${x0 + bw + 6},124 L${x0 + bw + gap - 6},124 M${x0 + bw + gap - 12},118 L${x0 + bw + gap - 6},124 L${x0 + bw + gap - 12},130`).attr('fill', 'none').attr('stroke', ink2()).attr('stroke-width', 2);
    x0 += bw + gap;
  });
  return wrap;
}
