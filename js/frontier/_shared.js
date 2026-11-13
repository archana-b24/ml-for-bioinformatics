// Shared helpers for Tab 3 (frontier) pages: source cards, honesty labels, citations, and small numeric tools.
// Not a page (no registry id); imported by js/frontier/<id>.js.
import { el, html, callout, section } from '../lib/ui.js';

// Every URL here comes from docs/frontier_research.md (or a DOI listed there).
export const REFS = {
  geneformer: ['Geneformer (Hugging Face)', 'https://huggingface.co/ctheodoris/Geneformer'],
  kedzierska: ['Kedzierska et al. 2025, Genome Biology', 'https://link.springer.com/article/10.1186/s13059-025-03574-x'],
  scgpt: ['scGPT, Nature Methods 2024', 'https://www.nature.com/articles/s41592-024-02201-0.pdf'],
  scfoundation: ['scFoundation (GitHub)', 'https://github.com/biomap-research/scFoundation'],
  uce: ['UCE (GitHub)', 'https://www.github.com/snap-stanford/UCE'],
  transcriptformer: ['TranscriptFormer, bioRxiv 2025', 'https://www.biorxiv.org/content/10.1101/2025.04.25.650731.full.pdf'],
  rbio1: ['rbio1, bioRxiv 2025', 'https://www.biorxiv.org/content/10.1101/2025.08.18.670981.full.pdf'],
  c2s: ['C2S-Scale, bioRxiv 2025', 'https://doi.org/10.1101/2025.04.14.648850'],
  stackNews: ['Arc Institute: Stack (news)', 'https://arcinstitute.org/news/foundation-model-stack'],
  stackGit: ['Stack (GitHub)', 'https://github.com/arcinstitute/stack'],
  stateNews: ['Arc Institute: State (news)', 'https://arcinstitute.org/news/virtual-cell-model-state'],
  stateGit: ['State (GitHub)', 'https://github.com/arcinstitute/state'],
  ahlmann: ['Ahlmann-Eltze, Huber & Anders, Nat Methods 2025 (code + paper)', 'https://github.com/const-ae/linear_perturbation_prediction-Paper'],
  systema: ['Systema (Brbić lab, EPFL)', 'https://brbiclab.epfl.ch/projects/systema'],
  perteval: ['PertEval-scFM, ICML 2025', 'https://proceedings.mlr.press/v267/wenteler25a.html'],
  perturbench: ['PerturBench (arXiv)', 'https://arxiv.org/html/2408.10609v4'],
  vcbench: ['VCBench 2026 (bioRxiv)', 'https://www.biorxiv.org/content/10.64898/2026.06.18.733146v1.full.pdf'],
  vcc2025: ['Virtual Cell Challenge 2025 wrap-up', 'https://arcinstitute.org/news/virtual-cell-challenge-2025-wrap-up'],
  vcc2026: ['Virtual Cell Challenge 2026', 'https://arcinstitute.org/news/virtual-cell-challenge-2026'],
  primeflow: ['PRiMeFlow (arXiv)', 'https://arxiv.org/html/2604.13986v1'],
  cellflow: ['CellFlow, bioRxiv 2025', 'https://www.biorxiv.org/content/10.1101/2025.04.11.648220.full.pdf'],
  lpm: ['Large Perturbation Model, arXiv 2503.23535', 'https://arxiv.org/abs/2503.23535'],
  txpert: ['TxPert (Valence Labs / Recursion)', 'https://www.valencelabs.com/txpert-predicting-cellular-responses-to-unseen-genetic-perturbations/'],
  tahoe: ['Tahoe-100M, bioRxiv 2025', 'https://www.biorxiv.org/content/10.1101/2025.02.20.639398.full.pdf'],
  xaira: ['Xaira X-Atlas/Orion, bioRxiv 2025', 'https://www.biorxiv.org/content/10.1101/2025.06.11.659105.full.pdf'],
  nicheformer: ['Nicheformer, Nat Methods 2025 (DOI)', 'https://doi.org/10.1038/s41592-025-02814-z'],
  novae: ['Novae, Nat Methods 2025 (DOI)', 'https://doi.org/10.1038/s41592-025-02899-6'],
  novaePypi: ['novae (PyPI)', 'https://pypi.org/project/novae/'],
  gigapath: ['Prov-GigaPath, Nature 2024 (PubMed)', 'https://pubmed.ncbi.nlm.nih.gov/38778098/'],
  hest: ['HEST-1k benchmark (arXiv)', 'https://arxiv.org/html/2406.16192v2'],
  moptimus: ['Bioptimus M-Optimus docs', 'https://docs.bioptimus.com/documentation/models/m-optimus'],
  mosaic: ['Owkin MOSAIC data overview', 'https://docs.owkin.com/user-guide/what-data-is-available-in-k/overview-of-mosaic-data'],
  noetik: ['Noetik OCTO-vc (press release)', 'https://www.businesswire.com/news/home/20241218973612/en/'],
  bulkformer: ['BulkFormer, bioRxiv 2025', 'https://www.biorxiv.org/content/10.1101/2025.06.11.659222.full.pdf'],
  tempus: ['Tempus Immune Profile Score, medRxiv', 'https://medrxiv.org/content/10.1101/2024.11.05.24316583v2.full.pdf'],
  biomni: ['Biomni (PMC)', 'https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12157518/'],
  coscientist: ['Google Research: AI co-scientist', 'https://research.google/blog/accelerating-scientific-breakthroughs-with-an-ai-co-scientist/'],
  cellagent: ['CellAgent (arXiv 2407.09811)', 'https://arxiv.org/abs/2407.09811v1'],
  gptcelltype: ['GPTCelltype, Nat Methods 2024 (DOI)', 'https://doi.org/10.1038/s41592-024-02235-4'],
};

/** Inline citation link: cite('kedzierska') or cite('kedzierska', 'custom text'). */
export function cite(key, text) {
  const [label, url] = REFS[key];
  return `<a href="${url}" target="_blank" rel="noopener">${text || label}</a>`;
}

/**
 * Source card under the header.
 * { org, when, links: [refKey | [label,url]], claim, selfReported=true, status }
 */
export function sourceCard(main, { org, when, links = [], claim, selfReported = true, status }) {
  const linkHtml = links.map((l) => (typeof l === 'string' ? cite(l) : `<a href="${l[1]}" target="_blank" rel="noopener">${l[0]}</a>`)).join(' · ');
  const card = el('div', { class: 'card', style: 'margin:6px 0 16px;max-width:860px' });
  card.append(html(`
    <div style="display:grid;grid-template-columns:110px minmax(0,1fr);gap:4px 12px;font-size:14px">
      <span style="color:var(--muted)">Who</span><span>${org}</span>
      <span style="color:var(--muted)">When</span><span>${when}</span>
      <span style="color:var(--muted)">Source</span><span>${linkHtml || '<i>no public link in our brief</i>'}</span>
      <span style="color:var(--muted)">Claim</span><span>${claim} ${selfReported ? '<span class="tag">self-reported</span>' : ''}${status ? `<span class="tag">${status}</span>` : ''}</span>
    </div>`));
  main.append(card);
  return card;
}

/** "Teaching toy" label to put at the top of a mini-replication figure. */
export function toyBadge(fig, extra = '') {
  const b = html(`<span class="tag" style="background:var(--accent-soft);color:var(--ink)">Teaching toy, not the paper's model or numbers</span>${extra ? ' <span style="font-size:13px;color:var(--ink-2)">' + extra + '</span>' : ''}`, 'div');
  b.style.margin = '2px 0 6px';
  fig.root.insertBefore(b, fig.controls);
  return b;
}

export const coreIdea = (main, bodyHtml) => section(main, 'The core idea', bodyHtml);
export const realityCheck = (main, bodyHtml) => callout(main, 'warn', 'Reality check', bodyHtml);
export const useIt = (main, bodyHtml) => callout(main, 'good', "How you'd use it in your own work", bodyHtml);

// ---------- small numeric helpers (plain arrays, fast loops) ----------
export function pearson(a, b) {
  const n = a.length; let ma = 0, mb = 0;
  for (let i = 0; i < n; i++) { ma += a[i]; mb += b[i]; }
  ma /= n; mb /= n;
  let sab = 0, saa = 0, sbb = 0;
  for (let i = 0; i < n; i++) { const da = a[i] - ma, db = b[i] - mb; sab += da * db; saa += da * da; sbb += db * db; }
  return saa < 1e-15 || sbb < 1e-15 ? 0 : sab / Math.sqrt(saa * sbb);
}
export const vsub = (a, b) => a.map((v, i) => v - b[i]);
export const vadd = (a, b) => a.map((v, i) => v + b[i]);
export const vscale = (a, s) => a.map((v) => v * s);
export const colMeans = (X) => { const m = new Array(X[0].length).fill(0); X.forEach((r) => r.forEach((v, j) => { m[j] += v; })); return m.map((v) => v / X.length); };

/** Solve (A) X = B for symmetric positive-definite A (n×n) and B (n×m) by Cholesky. */
export function cholSolve(A, B) {
  const n = A.length, L = A.map(() => new Float64Array(n));
  for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) {
    let s = A[i][j]; for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
    L[i][j] = i === j ? Math.sqrt(Math.max(s, 1e-12)) : s / L[j][j];
  }
  const m = B[0].length, X = B.map((r) => Float64Array.from(r));
  for (let c = 0; c < m; c++) {
    for (let i = 0; i < n; i++) { let s = X[i][c]; for (let k = 0; k < i; k++) s -= L[i][k] * X[k][c]; X[i][c] = s / L[i][i]; }
    for (let i = n - 1; i >= 0; i--) { let s = X[i][c]; for (let k = i + 1; k < n; k++) s -= L[k][i] * X[k][c]; X[i][c] = s / L[i][i]; }
  }
  return X.map((r) => Array.from(r));
}
/** Multi-output ridge: X (n×p), Y (n×m) -> W (p×m) minimising ||XW-Y||² + λ||W||². */
export function ridgeMulti(X, Y, lambda = 1) {
  const p = X[0].length, m = Y[0].length;
  const A = Array.from({ length: p }, () => new Array(p).fill(0)), B = Array.from({ length: p }, () => new Array(m).fill(0));
  for (const [r, x] of X.entries()) {
    for (let i = 0; i < p; i++) { const xi = x[i]; if (!xi) continue; for (let j = 0; j < p; j++) A[i][j] += xi * x[j]; for (let c = 0; c < m; c++) B[i][c] += xi * Y[r][c]; }
  }
  for (let i = 0; i < p; i++) A[i][i] += lambda;
  return cholSolve(A, B);
}
/** Top-k principal axes of centred X (n×p) by power iteration with deflation. Returns {axes (k×p), scores (n×k), mean}. */
export function pca(X, k = 2, iters = 60, seed = 7) {
  const n = X.length, p = X[0].length, mu = colMeans(X);
  const C = X.map((r) => r.map((v, j) => v - mu[j]));
  const axes = [];
  let s = seed;
  for (let c = 0; c < k; c++) {
    let v = Array.from({ length: p }, (_, j) => Math.sin(++s * 12.9898 + j) );
    for (let it = 0; it < iters; it++) {
      const t = C.map((r) => r.reduce((acc, x, j) => acc + x * v[j], 0));
      const w = new Array(p).fill(0); C.forEach((r, i) => r.forEach((x, j) => { w[j] += x * t[i]; }));
      axes.forEach((a) => { const d = a.reduce((acc, x, j) => acc + x * w[j], 0); a.forEach((x, j) => { w[j] -= d * x; }); });
      const nrm = Math.hypot(...w) || 1; v = w.map((x) => x / nrm);
    }
    axes.push(v);
  }
  const scores = C.map((r) => axes.map((a) => r.reduce((acc, x, j) => acc + x * a[j], 0)));
  return { axes, scores, mean: mu };
}
