// Seeded randomness + toy datasets + tiny linear algebra. Everything deterministic given a seed.

export function rng(seed = 42) {
  // mulberry32
  let a = seed >>> 0;
  const r = () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  r.normal = (mu = 0, sd = 1) => { let u = 0, v = 0; while (!u) u = r(); while (!v) v = r(); return mu + sd * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
  r.int = (n) => Math.floor(r() * n);
  r.pick = (arr) => arr[Math.floor(r() * arr.length)];
  r.shuffle = (arr) => { const a2 = arr.slice(); for (let i = a2.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [a2[i], a2[j]] = [a2[j], a2[i]]; } return a2; };
  r.poisson = (lam) => { // Knuth for small, normal approx for large
    if (lam > 50) return Math.max(0, Math.round(r.normal(lam, Math.sqrt(lam))));
    const L = Math.exp(-lam); let k = 0, p = 1; do { k++; p *= r(); } while (p > L); return k - 1; };
  return r;
}

// ---- 2D toy datasets: return [{x, y, label}] in roughly [-1, 1] ----
export function blobs(n = 200, centers = [[-0.5, -0.4], [0.5, 0.4]], sd = 0.2, seed = 1) {
  const r = rng(seed); const out = [];
  for (let i = 0; i < n; i++) { const c = i % centers.length; out.push({ x: r.normal(centers[c][0], sd), y: r.normal(centers[c][1], sd), label: c }); }
  return out;
}
export function moons(n = 200, noise = 0.1, seed = 2) {
  const r = rng(seed); const out = [];
  for (let i = 0; i < n; i++) {
    const c = i % 2, t = Math.PI * r();
    const x = c ? 1 - Math.cos(t) : Math.cos(t), y = c ? 0.5 - Math.sin(t) : Math.sin(t);
    out.push({ x: (x - 0.5) * 0.8 + r.normal(0, noise), y: (y - 0.25) * 0.8 + r.normal(0, noise), label: c });
  }
  return out;
}
export function circles(n = 200, noise = 0.08, seed = 3) {
  const r = rng(seed); const out = [];
  for (let i = 0; i < n; i++) {
    const c = i % 2, t = 2 * Math.PI * r(), rad = c ? 0.35 : 0.8;
    out.push({ x: rad * Math.cos(t) + r.normal(0, noise), y: rad * Math.sin(t) + r.normal(0, noise), label: c });
  }
  return out;
}
export function spiral(n = 300, noise = 0.05, seed = 4) {
  const r = rng(seed); const out = [];
  for (let i = 0; i < n; i++) {
    const c = i % 2, t = (i / n) * 3 * Math.PI + 0.5, rad = t / (3.5 * Math.PI);
    const s = c ? Math.PI : 0;
    out.push({ x: rad * Math.cos(t + s) + r.normal(0, noise), y: rad * Math.sin(t + s) + r.normal(0, noise), label: c });
  }
  return out;
}
export function xor(n = 200, noise = 0.12, seed = 5) {
  const r = rng(seed); const out = [];
  for (let i = 0; i < n; i++) {
    const x = r() * 2 - 1, y = r() * 2 - 1;
    out.push({ x: x + r.normal(0, noise * 0.3), y: y + r.normal(0, noise * 0.3), label: (x * y > 0) ? 1 : 0 });
  }
  return out;
}
/** 1-D regression: y = f(x) + noise, x in [-1,1]. */
export function regression1d(n = 30, f = (x) => Math.sin(2.5 * x), noise = 0.2, seed = 6) {
  const r = rng(seed); const out = [];
  for (let i = 0; i < n; i++) { const x = r() * 2 - 1; out.push({ x, y: f(x) + r.normal(0, noise) }); }
  return out.sort((a, b) => a.x - b.x);
}

// ---- tiny linear algebra on plain arrays ----
export const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
export const transpose = (A) => A[0].map((_, j) => A.map((r) => r[j]));
export const matmul = (A, B) => A.map((r) => B[0].map((_, j) => r.reduce((s, v, k) => s + v * B[k][j], 0)));
export function solve(A, b) { // Gaussian elimination with partial pivoting; A square
  const n = A.length; const M = A.map((r, i) => [...r, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    const d = M[c][c] || 1e-12;
    for (let r = 0; r < n; r++) if (r !== c) { const f = M[r][c] / d; for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k]; }
  }
  return M.map((r, i) => r[n] / (M[i][i] || 1e-12));
}
/** Ridge / OLS: X rows of features (no intercept column added), returns weights. */
export function ridge(X, y, lambda = 0) {
  const Xt = transpose(X); const XtX = matmul(Xt, X);
  for (let i = 0; i < XtX.length; i++) XtX[i][i] += lambda;
  const Xty = Xt.map((r) => dot(r, y));
  return solve(XtX, Xty);
}
export const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
export const sd = (a) => { const m = mean(a); return Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / Math.max(1, a.length - 1)); };
export const sigmoid = (z) => 1 / (1 + Math.exp(-z));
export const dist2 = (a, b) => (a.x - b.x) ** 2 + (a.y - b.y) ** 2;
export const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
export const range = (n) => Array.from({ length: n }, (_, i) => i);
export const linspace = (a, b, n) => range(n).map((i) => a + (b - a) * i / (n - 1));
/** 2x2 symmetric eigen-decomposition -> {vals:[l1,l2], vecs:[[x,y],[x,y]]} sorted desc. */
export function eig2(a, b, d) {
  const tr = a + d, det = a * d - b * b, disc = Math.sqrt(Math.max(0, tr * tr / 4 - det));
  const l1 = tr / 2 + disc, l2 = tr / 2 - disc;
  const v = (l) => { const x = b, y = l - a; const n = Math.hypot(x, y); return n < 1e-12 ? [1, 0] : [x / n, y / n]; };
  const v1 = Math.abs(b) < 1e-12 ? (a >= d ? [1, 0] : [0, 1]) : v(l1);
  return { vals: [l1, l2], vecs: [v1, [-v1[1], v1[0]]] };
}
