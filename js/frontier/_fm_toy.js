// Toy engine for the "Single-cell foundation models" lesson: simulated tumour-microenvironment cells, a rank-value
// tokeniser, a co-occurrence "pretraining" step and baselines. Teaching toy only; no transformer, no real data.
// d3-free on purpose, so the numbers can be checked outside the browser.
import { rng } from '../lib/data.js';
import { pca } from './_shared.js';

export const TYPES = ['Tumour', 'CD8 T', 'CD4 T', 'B cell', 'Macrophage', 'CAF'];
export const HK = { ACTB: 30, MALAT1: 60, TMSB4X: 45, EEF1A1: 40, RPL13: 30, RPS27: 32, B2M: 28, GAPDH: 20, FTL: 18, 'MT-CO1': 24 };
const MARK = [
  { EPCAM: 5, KRT8: 8, KRT18: 7, KRT19: 6, KRT5: 4, SFN: 3, CLDN4: 3 },
  { CD3E: 5, CD3D: 4, CD8A: 5, CD8B: 3, GZMK: 3.5, CCL5: 7, NKG7: 4, PTPRC: 4 },
  { CD3E: 5, CD3D: 4, CD4: 2, IL7R: 4, LTB: 5, CCR7: 2, TCF7: 2.5, PTPRC: 4 },
  { MS4A1: 6, CD79A: 7, CD79B: 4, BANK1: 3, CD74: 9, 'HLA-DRA': 7, PTPRC: 3 },
  { CD68: 4, C1QA: 6, C1QB: 5, APOE: 6, LYZ: 8, CD14: 3, 'HLA-DRA': 6, CD74: 8, PTPRC: 2 },
  { COL1A1: 8, COL1A2: 7, DCN: 8, LUM: 5, FAP: 2.5, PDGFRA: 2.5, IGFBP7: 6 },
];
export const MARKERS = MARK;
// Programs of other tissues: present in the pretraining corpus, almost absent from the tumour biopsies.
const OTHER = [
  { ALB: 9, APOA1: 7, TTR: 6, SERPINA1: 6 }, { GFAP: 7, SNAP25: 6, SYT1: 5, MBP: 6 },
  { MYH7: 8, ACTA1: 9, TNNT2: 6, MB: 7 }, { INS: 9, GCG: 6, PRSS1: 7, CELA3A: 6 },
];
const NOISE = { JUN: 1.5, FOS: 1.5, HSPA1A: 1, IFI6: 0.8, ISG15: 0.8, XIST: 1 };
export const GENES = [...new Set([...Object.keys(HK), ...MARK.flatMap((m) => Object.keys(m)), ...Object.keys(NOISE), ...OTHER.flatMap((m) => Object.keys(m))])];
export const G = GENES.length;
export const GI = Object.fromEntries(GENES.map((g, i) => [g, i]));
export const gclass = (g) => (g in HK ? 'hk' : g in NOISE ? 'noise' : OTHER.some((o) => g in o) ? 'other' : 'marker');
const MIX = 0.5; // each cell takes up to this share of its profile from another type (ambient RNA, doublets, states)
const rate = (g, prog) => {
  if (prog[g] == null) return g in HK ? HK[g] : g in NOISE ? NOISE[g] : OTHER.some((o) => g in o) ? 0.04 : 0.25;
  return prog[g];
};

const meanOf = (a) => a.reduce((s, v) => s + v, 0) / a.length;
const median = (a) => { const s = a.slice().sort((x, y) => x - y); return s[Math.floor(s.length / 2)]; };

/** Batch shifts: two "chemistries" that differ in depth and in a third of the genes. strength 0 = no batch effect. */
function batchShifts(strength) {
  const r = rng(17);
  return [0, 1].map((b) => GENES.map((g) => (b && !(g in HK) && r() < 0.35 ? strength * r.normal(0, 0.8) : (r(), 0))));
}

/** The biopsy dataset: `per` cells of each type in each of two batches. */
export function makeCells(strength, per = 30, seed = 29) {
  const r = rng(seed), sh = batchShifts(strength), cells = [];
  for (let b = 0; b < 2; b++) for (let t = 0; t < TYPES.length; t++) for (let i = 0; i < per; i++) {
    const size = (b ? 1 + 1.5 * strength : 1) * Math.exp(r.normal(0, 0.35));
    const other = MARK[(t + 1 + r.int(TYPES.length - 1)) % TYPES.length], lam = MIX * r() ** 1.5;
    cells.push({ t, b, counts: GENES.map((g, j) => r.poisson(size * ((1 - lam) * rate(g, MARK[t]) + (g in HK ? lam * rate(g, MARK[t]) : lam * rate(g, other))) * Math.exp(sh[b][j]))) });
  }
  return cells;
}

/** A pretraining corpus: `overlap` is the share of cells from the same tissue (the rest come from other tissues). */
export function makeCorpus(overlap, n = 300, seed = 41) {
  const r = rng(seed), cells = [];
  for (let i = 0; i < n; i++) {
    const mine = r() < overlap;
    const prog = mine ? MARK[r.int(TYPES.length)] : OTHER[r.int(OTHER.length)];
    const size = 0.4 + 2.5 * r();
    cells.push({ t: -1, b: -1, counts: GENES.map((g) => r.poisson(size * rate(g, prog))) });
  }
  return cells;
}

export const norm1e4 = (counts) => { const tot = counts.reduce((a, b) => a + b, 0) || 1; return counts.map((v) => (v / tot) * 1e4); };

/** Non-zero median of each gene (per-10k scale) across a corpus. */
export function geneMedians(corpus) {
  const nz = GENES.map(() => []);
  corpus.forEach((c) => norm1e4(c.counts).forEach((v, j) => { if (v > 0) nz[j].push(v); }));
  const all = nz.map((a) => (a.length ? median(a) : null)), fb = median(all.filter((v) => v != null));
  return all.map((v) => (v == null ? fb : v));
}

/** Rank-value encoding of one cell: returns { norm, rv, order (expressed genes, high to low rank value), raw (high to low count) }. */
export function encode(counts, med) {
  const norm = norm1e4(counts), rv = norm.map((v, j) => v / med[j]);
  const expressed = counts.map((c, j) => j).filter((j) => counts[j] > 0);
  return {
    norm, rv,
    order: expressed.slice().sort((a, b) => rv[b] - rv[a]),
    raw: expressed.slice().sort((a, b) => counts[b] - counts[a]),
  };
}

export const L = 12; // sentence length (tokens kept)

/** "Pretraining": rank-value tokenise the corpus, count co-occurrence among the first L tokens, derive gene vectors. */
export function pretrain(corpus, dim = 8) {
  const med = geneMedians(corpus), co = Array.from({ length: G }, () => new Float64Array(G)), freq = new Float64Array(G);
  corpus.forEach((c) => { c.tok = encode(c.counts, med).order.slice(0, L); c.tok.forEach((a) => { freq[a]++; c.tok.forEach((b) => { if (a !== b) co[a][b]++; }); }); });
  const prof = co.map((row, a) => { const s = row.reduce((x, y) => x + y, 0) + 1e-9; return Array.from(row, (v) => Math.sqrt(v / s)); });
  const sc = pca(prof, dim, 40).scores;
  const gvec = sc.map((v, a) => (freq[a] ? v : new Array(dim).fill(0)));
  return { med, co, freq, gvec, n: corpus.length };
}

/** Frozen cell embedding: position-weighted mean of the gene vectors of the first L tokens, scaled to unit length. */
export function embedTokens(model, tokens) {
  const d = model.gvec[0].length, v = new Array(d).fill(0); let ws = 0;
  tokens.slice(0, L).forEach((j, i) => { const w = 1 / (1 + 0.15 * i); ws += w; for (let k = 0; k < d; k++) v[k] += w * model.gvec[j][k]; });
  const nrm = Math.hypot(...v.map((x) => x / ws)) || 1;
  return v.map((x) => x / ws / nrm);
}
export const embedCell = (model, counts) => embedTokens(model, encode(counts, model.med).order);

/** Baseline: log1p(CP10k) → 30 highly variable genes (chosen on the data being analysed) → z-score → 8 PCs. */
export function hvgPca(cells, nHvg = 30, dim = 8) {
  const X = cells.map((c) => norm1e4(c.counts).map((v) => Math.log1p(v)));
  const mu = GENES.map((_, g) => meanOf(X.map((r) => r[g]))), sdv = GENES.map((_, g) => Math.sqrt(meanOf(X.map((r) => (r[g] - mu[g]) ** 2))) || 1);
  const hvg = GENES.map((_, g) => g).sort((a, b) => sdv[b] / (mu[b] + 0.5) - sdv[a] / (mu[a] + 0.5)).slice(0, nHvg);
  return pca(X.map((r) => hvg.map((g) => (r[g] - mu[g]) / sdv[g])), dim, 30).scores;
}

/** k-NN type accuracy (leave-one-out) and batch mixing (1 = neighbours drawn from the two batches at random). */
export function knnMetrics(E, cells, K = 10) {
  const n = E.length, D = E[0].length; let acc = 0, mix = 0;
  const d = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) { let s = 0; for (let k = 0; k < D; k++) { const u = E[i][k] - E[j][k]; s += u * u; } d[j] = j === i ? Infinity : s; }
    const nn = Array.from({ length: n }, (_, j) => j).sort((a, b) => d[a] - d[b]).slice(0, K);
    const votes = new Array(TYPES.length).fill(0); nn.forEach((j) => { votes[cells[j].t]++; });
    if (votes.indexOf(Math.max(...votes)) === cells[i].t) acc++;
    mix += nn.filter((j) => cells[j].b !== cells[i].b).length / K;
  }
  return { acc: acc / n, mix: Math.min(1, (mix / n) / 0.5) };
}

/** Nearest-centroid classifier built from labelled `prompt` embeddings; returns accuracy on `test` embeddings. */
export function centroidAcc(promptE, promptT, testE, testT) {
  const D = testE[0].length, cen = TYPES.map(() => ({ s: new Array(D).fill(0), n: 0 }));
  promptE.forEach((e, i) => { const c = cen[promptT[i]]; c.n++; e.forEach((v, k) => { c.s[k] += v; }); });
  const C = cen.map((c) => c.s.map((v) => v / Math.max(1, c.n)));
  let hit = 0;
  testE.forEach((e, i) => {
    let best = -1, bd = Infinity;
    C.forEach((c, t) => { if (!cen[t].n) return; let s = 0; for (let k = 0; k < D; k++) s += (e[k] - c[k]) ** 2; if (s < bd) { bd = s; best = t; } });
    if (best === testT[i]) hit++;
  });
  return hit / testE.length;
}
