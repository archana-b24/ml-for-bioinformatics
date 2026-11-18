// Small, exact model fitters + evaluation helpers shared by the lesson pages. Plain arrays, no deps.
import { solve, sigmoid, rng } from './data.js';

/** Column-standardise rows X (array of arrays). Returns {Z, mu, sd, apply(row)}. */
export function standardize(X) {
  const p = X[0].length, n = X.length;
  const mu = Array.from({ length: p }, (_, j) => X.reduce((s, r) => s + r[j], 0) / n);
  const sd = mu.map((m, j) => Math.sqrt(X.reduce((s, r) => s + (r[j] - m) ** 2, 0) / Math.max(1, n - 1)) || 1);
  const apply = (r) => r.map((v, j) => (v - mu[j]) / sd[j]);
  return { Z: X.map(apply), mu, sd, apply };
}

/**
 * Logistic regression by Newton / IRLS with optional L2 penalty (intercept unpenalised).
 * X: rows of features (no intercept column). Returns {b0, w, se:[se_b0, ...se_w], loglik, converged, iters}.
 * se are from the inverse of the (penalised) Hessian: textbook Wald standard errors when l2 = 0.
 */
export function fitLogistic(X, y, { l2 = 0, maxIter = 60, tol = 1e-9 } = {}) {
  const n = X.length, p = X[0] ? X[0].length : 0, k = p + 1;
  const A = X.map((r) => [1, ...r]);
  let beta = new Array(k).fill(0), converged = false, it = 0, H = null;
  for (it = 0; it < maxIter; it++) {
    const g = new Array(k).fill(0);
    H = Array.from({ length: k }, () => new Array(k).fill(0));
    for (let i = 0; i < n; i++) {
      const z = A[i].reduce((s, v, j) => s + v * beta[j], 0);
      const pi = sigmoid(z), wi = Math.max(pi * (1 - pi), 1e-10), e = y[i] - pi;
      for (let a = 0; a < k; a++) { g[a] += A[i][a] * e; for (let b = a; b < k; b++) H[a][b] += wi * A[i][a] * A[i][b]; }
    }
    for (let a = 0; a < k; a++) for (let b = 0; b < a; b++) H[a][b] = H[b][a];
    for (let a = 1; a < k; a++) { H[a][a] += l2; g[a] -= l2 * beta[a]; }
    const step = solve(H, g);
    let mx = 0; for (let a = 0; a < k; a++) { const s = Math.max(-5, Math.min(5, step[a])); beta[a] += s; mx = Math.max(mx, Math.abs(s)); }
    if (mx < tol) { converged = true; break; }
  }
  const se = Array.from({ length: k }, (_, a) => { const e = new Array(k).fill(0); e[a] = 1; return Math.sqrt(Math.max(0, solve(H, e)[a])); });
  const big = beta.some((b) => Math.abs(b) > 25);
  return { b0: beta[0], w: beta.slice(1), se, loglik: logLik(X, y, beta[0], beta.slice(1)), converged: converged && !big, iters: it + 1 };
}

export const predictLogit = (row, b0, w) => b0 + row.reduce((s, v, j) => s + v * w[j], 0);
export const predictProb = (row, b0, w) => sigmoid(predictLogit(row, b0, w));

export function logLik(X, y, b0, w) {
  let s = 0;
  X.forEach((r, i) => { const q = Math.min(1 - 1e-12, Math.max(1e-12, predictProb(r, b0, w))); s += y[i] ? Math.log(q) : Math.log(1 - q); });
  return s;
}
/** Mean log-loss (binary cross-entropy) of probabilities p against labels y. */
export const logLoss = (p, y) => -p.reduce((s, q, i) => { q = Math.min(1 - 1e-12, Math.max(1e-12, q)); return s + (y[i] ? Math.log(q) : Math.log(1 - q)); }, 0) / p.length;

/** Rank-based AUC (probability a random positive scores above a random negative; ties count 1/2). */
export function auc(score, y) {
  const idx = score.map((s, i) => i).sort((a, b) => score[a] - score[b]);
  let rankSum = 0, nPos = 0, i = 0;
  while (i < idx.length) {
    let j = i; while (j + 1 < idx.length && score[idx[j + 1]] === score[idx[i]]) j++;
    const r = (i + j) / 2 + 1;
    for (let t = i; t <= j; t++) if (y[idx[t]]) { rankSum += r; nPos++; }
    i = j + 1;
  }
  const nNeg = y.length - nPos;
  return nPos && nNeg ? (rankSum - nPos * (nPos + 1) / 2) / (nPos * nNeg) : NaN;
}
/** ROC points [{fpr, tpr, thr}] from high threshold to low. */
export function roc(score, y) {
  const o = score.map((s, i) => [s, y[i]]).sort((a, b) => b[0] - a[0]);
  const P = y.filter(Boolean).length, N = y.length - P;
  const pts = [{ fpr: 0, tpr: 0, thr: Infinity }]; let tp = 0, fp = 0;
  o.forEach(([s, l], i) => { if (l) tp++; else fp++; if (i === o.length - 1 || o[i + 1][0] !== s) pts.push({ fpr: fp / N, tpr: tp / P, thr: s }); });
  return pts;
}
/** Confusion counts at a probability threshold. */
export function confusion(p, y, thr = 0.5) {
  let tp = 0, fp = 0, tn = 0, fn = 0;
  p.forEach((q, i) => { const yhat = q >= thr; if (yhat && y[i]) tp++; else if (yhat) fp++; else if (y[i]) fn++; else tn++; });
  return { tp, fp, tn, fn, acc: (tp + tn) / p.length, sens: tp / Math.max(1, tp + fn), spec: tn / Math.max(1, tn + fp), ppv: tp / Math.max(1, tp + fp) };
}
/** Stratified train/test split of indices 0..n-1. Returns {train, test}. */
export function stratifiedSplit(y, testFrac = 0.3, seed = 7) {
  const r = rng(seed), train = [], test = [];
  [0, 1].forEach((c) => {
    const ids = r.shuffle(y.map((v, i) => i).filter((i) => +y[i] === c));
    const nt = Math.round(ids.length * testFrac);
    ids.forEach((i, k) => (k < nt ? test : train).push(i));
  });
  return { train: train.sort((a, b) => a - b), test: test.sort((a, b) => a - b) };
}
/** Two-sided normal p-value for a z statistic. */
export function pNormal(z) {
  const t = 1 / (1 + 0.2316419 * Math.abs(z));
  const d = 0.3989423 * Math.exp(-z * z / 2);
  const q = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
  return 2 * q;
}

/**
 * Ordinary least squares (optional ridge penalty on the slopes) with classical standard errors.
 * X: rows of features (no intercept column). Returns {b0, w, beta, se:[se_b0,...], fitted, resid, rss, tss, r2, r2adj, sigma, df, n, lev, Ginv, cov(row)}.
 * `Ginv` is (AᵀA + penalty)⁻¹ with A = [1, X]; `varNew(row)` is the variance factor for a prediction at `row`.
 */
export function fitLinear(X, y, { l2 = 0 } = {}) {
  const n = X.length, p = X[0] ? X[0].length : 0, k = p + 1;
  const A = X.map((r) => [1, ...r]);
  const G = Array.from({ length: k }, () => new Array(k).fill(0)), g = new Array(k).fill(0);
  A.forEach((r, i) => { for (let a = 0; a < k; a++) { g[a] += r[a] * y[i]; for (let b = a; b < k; b++) G[a][b] += r[a] * r[b]; } });
  for (let a = 0; a < k; a++) for (let b = 0; b < a; b++) G[a][b] = G[b][a];
  for (let a = 1; a < k; a++) G[a][a] += l2;
  const beta = solve(G, g);
  const cols = Array.from({ length: k }, (_, a) => { const e = new Array(k).fill(0); e[a] = 1; return solve(G, e); });
  const Ginv = Array.from({ length: k }, (_, a) => Array.from({ length: k }, (_, b) => cols[b][a]));
  const fitted = A.map((r) => r.reduce((s, v, j) => s + v * beta[j], 0));
  const resid = y.map((v, i) => v - fitted[i]);
  const rss = resid.reduce((s, e) => s + e * e, 0), my = y.reduce((s, v) => s + v, 0) / n, tss = y.reduce((s, v) => s + (v - my) ** 2, 0);
  const df = Math.max(1, n - k), sigma2 = rss / df;
  const quad = (r) => r.reduce((s, v, a) => s + v * r.reduce((t, w2, b) => t + Ginv[a][b] * w2, 0), 0);
  return {
    b0: beta[0], w: beta.slice(1), beta, se: Ginv.map((row, a) => Math.sqrt(Math.max(0, sigma2 * row[a]))),
    fitted, resid, rss, tss, r2: 1 - rss / tss, r2adj: 1 - (rss / df) / (tss / (n - 1)), sigma: Math.sqrt(sigma2), df, n,
    lev: A.map(quad), Ginv, varNew: (row) => sigma2 * (1 + quad([1, ...row])), varMean: (row) => sigma2 * quad([1, ...row]),
  };
}
export const predictLinear = (row, b0, w) => b0 + row.reduce((s, v, j) => s + v * w[j], 0);
/** Approximate 97.5% quantile of Student's t with df degrees of freedom. */
export const t975 = (df) => 1.959964 + (1.959964 ** 3 + 1.959964) / (4 * df) + (5 * 1.959964 ** 5 + 16 * 1.959964 ** 3 + 3 * 1.959964) / (96 * df * df);
/** Inverse of the standard normal CDF (Acklam's rational approximation). */
export function qNorm(p) {
  const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239];
  const b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
  const c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
  const lo = 0.02425;
  if (p < lo) { const q = Math.sqrt(-2 * Math.log(p)); return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
  if (p > 1 - lo) return -qNorm(1 - p);
  const q = p - 0.5, r = q * q;
  return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}

// ---------------------------------------------------------------------------------------------
// Penalised logistic regression (ridge, lasso, elastic net), added for the regularisation lesson.
// Objective (glmnet convention): mean log-loss + lambda * ( alpha*|w|_1 + (1-alpha)/2 * |w|_2^2 ), intercept unpenalised.
// alpha = 0 uses Newton steps (exact ridge); alpha > 0 uses accelerated proximal gradient (FISTA) with soft-thresholding.
/** Largest eigenvalue of [1,X]ᵀ[1,X]/n by power iteration (used for the FISTA step size). */
export function lipschitz(X) {
  const n = X.length, p = X[0].length; let v = new Array(p + 1).fill(1), lam = 1;
  for (let it = 0; it < 30; it++) {
    const u = X.map((r) => v[0] + r.reduce((s, q, j) => s + q * v[j + 1], 0)), nv = new Array(p + 1).fill(0);
    for (let i = 0; i < n; i++) { nv[0] += u[i]; for (let j = 0; j < p; j++) nv[j + 1] += X[i][j] * u[i]; }
    lam = Math.hypot(...nv) / n; v = nv.map((q) => q / (lam * n));
  }
  return lam;
}
/** X: rows (should be standardised). Returns {b0, w}. opts: lambda, alpha, init:{b0,w}, iters, L (from lipschitz). */
export function fitPenalised(X, y, { lambda = 0, alpha = 0, init = null, iters = 150, L = null } = {}) {
  const n = X.length, p = X[0].length;
  let b0 = init ? init.b0 : 0, w = init ? init.w.slice() : new Array(p).fill(0);
  if (alpha === 0) {
    const k = p + 1;
    for (let it = 0; it < 25; it++) {
      const g = new Array(k).fill(0), H = Array.from({ length: k }, () => new Array(k).fill(0));
      for (let i = 0; i < n; i++) {
        const r = X[i]; let z = b0; for (let j = 0; j < p; j++) z += r[j] * w[j];
        const q = sigmoid(z), wi = Math.max(q * (1 - q), 1e-8), e = q - y[i];
        g[0] += e; H[0][0] += wi;
        for (let a = 0; a < p; a++) { const xa = r[a]; g[a + 1] += xa * e; H[0][a + 1] += wi * xa; for (let b = a; b < p; b++) H[a + 1][b + 1] += wi * xa * r[b]; }
      }
      for (let a = 0; a < k; a++) for (let b = a; b < k; b++) { H[a][b] /= n; if (b > a) H[b][a] = H[a][b]; }
      for (let a = 0; a < k; a++) g[a] /= n;
      for (let a = 1; a < k; a++) { H[a][a] += lambda; g[a] += lambda * w[a - 1]; }
      const st = solve(H, g); let mx = 0;
      b0 -= Math.max(-3, Math.min(3, st[0]));
      for (let j = 0; j < p; j++) { const s = Math.max(-3, Math.min(3, st[j + 1])); w[j] -= s; mx = Math.max(mx, Math.abs(s)); }
      if (mx < 1e-7) break;
    }
    return { b0, w };
  }
  const step = 1 / (0.25 * (L || lipschitz(X)) + lambda * (1 - alpha)), thr = step * lambda * alpha, sh = 1 / (1 + step * lambda * (1 - alpha));
  let yb = b0, yw = w.slice(), t = 1;
  for (let it = 0; it < iters; it++) {
    const gw = new Array(p).fill(0); let gb = 0;
    for (let i = 0; i < n; i++) {
      const r = X[i]; let z = yb; for (let j = 0; j < p; j++) z += r[j] * yw[j];
      const e = sigmoid(z) - y[i]; gb += e; for (let j = 0; j < p; j++) gw[j] += r[j] * e;
    }
    const nb = yb - step * gb / n, nw = new Array(p);
    for (let j = 0; j < p; j++) { const v = yw[j] - step * gw[j] / n; nw[j] = (v > thr ? v - thr : v < -thr ? v + thr : 0) * sh; }
    const nt = (1 + Math.sqrt(1 + 4 * t * t)) / 2, m = (t - 1) / nt;
    yb = nb + m * (nb - b0); for (let j = 0; j < p; j++) yw[j] = nw[j] + m * (nw[j] - w[j]);
    b0 = nb; w = nw; t = nt;
  }
  return { b0, w };
}
/** Smallest lambda at which the lasso solution is all zero: max_j |Xⱼᵀ(y − ȳ)| / n. */
export function lambdaMax(X, y) {
  const n = X.length, ym = y.reduce((s, v) => s + v, 0) / n;
  return Math.max(...X[0].map((_, j) => Math.abs(X.reduce((s, r, i) => s + r[j] * (y[i] - ym), 0)) / n));
}

// ---------------------------------------------------------------------------------------------
// Decision trees (binary classification, axis-aligned splits). Added for the decision-tree lesson.
// A node is {id, depth, n, c:[n0,n1], p (share of class 1), imp, f, thr, gain, left, right}.
// A sample goes left when row[f] < thr, right when row[f] >= thr.
/** Binary impurity of a node with class-1 share p. crit: 'gini' | 'entropy' | 'error'. */
export function impurity(p, crit = 'gini') {
  if (crit === 'entropy') return p <= 0 || p >= 1 ? 0 : -p * Math.log2(p) - (1 - p) * Math.log2(1 - p);
  if (crit === 'error') return Math.min(p, 1 - p);
  return 2 * p * (1 - p);
}
/** Weighted child impurity at every candidate threshold of feature f (midpoints between distinct sorted values). */
export function scanSplits(X, y, idx, f, { minLeaf = 1, crit = 'gini' } = {}) {
  const s = idx.slice().sort((a, b) => X[a][f] - X[b][f]), n = s.length, tot = s.reduce((a, i) => a + y[i], 0), out = [];
  let l1 = 0;
  for (let k = 0; k < n - 1; k++) {
    l1 += y[s[k]];
    const a = X[s[k]][f], b = X[s[k + 1]][f];
    if (a === b) continue;
    const nl = k + 1, nr = n - nl;
    if (nl < minLeaf || nr < minLeaf) continue;
    const il = impurity(l1 / nl, crit), ir = impurity((tot - l1) / nr, crit);
    out.push({ thr: (a + b) / 2, w: (nl * il + nr * ir) / n, nl, nr, il, ir, pl: l1 / nl, pr: (tot - l1) / nr });
  }
  return out;
}
/** Grow a tree greedily. Returns the root node with `leaves`, `size` helpers via treeLeaves(). */
export function fitTree(X, y, { maxDepth = 3, minLeaf = 1, minSplit = 2, crit = 'gini', feats = null, idx = null } = {}) {
  const F = feats || X[0].map((_, j) => j);
  let nid = 0;
  const grow = (ids, depth) => {
    const n1 = ids.reduce((a, i) => a + y[i], 0), n = ids.length, p = n ? n1 / n : 0;
    const node = { id: nid++, depth, n, c: [n - n1, n1], p, imp: impurity(p, crit) };
    if (depth >= maxDepth || n < minSplit || node.imp === 0) return node;
    let best = null;
    F.forEach((f) => scanSplits(X, y, ids, f, { minLeaf, crit }).forEach((s) => { if (!best || s.w < best.w - 1e-12) best = { ...s, f }; }));
    if (!best || best.w >= node.imp - 1e-12) return node;
    node.f = best.f; node.thr = best.thr; node.gain = node.imp - best.w; node.wgain = node.gain * n;
    node.left = grow(ids.filter((i) => X[i][best.f] < best.thr), depth + 1);
    node.right = grow(ids.filter((i) => X[i][best.f] >= best.thr), depth + 1);
    return node;
  };
  return grow(idx || X.map((_, i) => i), 0);
}
const isLeafT = (nd, o) => !nd.left || (o.maxDepth !== undefined && nd.depth >= o.maxDepth) || (o.cut && o.cut.has(nd.id));
/** The leaf a sample falls into, optionally truncated at maxDepth or at a set of collapsed node ids (`cut`). */
export function treeLeaf(nd, row, o = {}) { while (!isLeafT(nd, o)) nd = row[nd.f] < nd.thr ? nd.left : nd.right; return nd; }
export function treeLeaves(nd, o = {}, out = []) { if (isLeafT(nd, o)) out.push(nd); else { treeLeaves(nd.left, o, out); treeLeaves(nd.right, o, out); } return out; }
export const treeDepth = (nd) => (nd.left ? 1 + Math.max(treeDepth(nd.left), treeDepth(nd.right)) : 0);
/**
 * Cost-complexity pruning path: successively collapse the weakest link (smallest increase in training error
 * per leaf removed). Returns [{alpha, leaves, cut:Set}], the first entry being the unpruned tree.
 */
export function pruningPath(root) {
  const N = root.n, cut = new Set();
  const st = (nd) => { if (!nd.left || cut.has(nd.id)) return { R: Math.min(...nd.c) / N, L: 1 }; const a = st(nd.left), b = st(nd.right); return { R: a.R + b.R, L: a.L + b.L }; };
  const path = [{ alpha: 0, leaves: st(root).L, cut: new Set() }];
  while (root.left && !cut.has(root.id)) {
    const cand = [];
    (function walk(nd) { if (!nd.left || cut.has(nd.id)) return; const s = st(nd); cand.push({ id: nd.id, g: (Math.min(...nd.c) / N - s.R) / (s.L - 1) }); walk(nd.left); walk(nd.right); })(root);
    const gmin = Math.min(...cand.map((c) => c.g));
    cand.filter((c) => c.g <= gmin + 1e-12).forEach((c) => cut.add(c.id));
    path.push({ alpha: gmin, leaves: st(root).L, cut: new Set(cut) });
  }
  return path;
}

// ---------------------------------------------------------------------------------------------
// Tree ensembles (random forest and gradient boosting), added for the tree-ensembles lesson.
// A node is {n, depth, v (leaf value), f, thr, gain, left, right}; a sample goes left when row[f] < thr.
/**
 * Grow one binary tree. mode 'gini': y in {0,1}, leaf value = share of class 1, gain = weighted Gini decrease.
 * mode 'newton': per-sample gradient g and hessian h (log-loss boosting), leaf value = -G/(H+lambda), gain = XGBoost-style gain.
 * mtry < p draws a fresh random feature subset at every node (needs rand = rng(seed)). idx lists the training rows (repeats allowed).
 */
export function growTree(X, y, o = {}) {
  const { mode = 'gini', g = null, h = null, maxDepth = 3, minLeaf = 1, mtry = null, rand = null, lambda = 1, idx = null, order = null } = o;
  const p = X[0].length, all = Array.from({ length: p }, (_, j) => j), newton = mode === 'newton';
  // S[f] = this node's rows sorted by feature f (sorted once at the root, then partitioned stably down the tree)
  const grow = (S, depth) => {
    const ids = S[0], n = ids.length; let G = 0, H = 0, n1 = 0;
    if (newton) for (const i of ids) { G += g[i]; H += h[i]; } else for (const i of ids) n1 += y[i];
    const node = { n, depth, v: newton ? -G / (H + lambda) : (n ? n1 / n : 0) };
    if (depth >= maxDepth || n < 2 * minLeaf || (!newton && (n1 === 0 || n1 === n))) return node;
    const F = mtry && mtry < p ? rand.shuffle(all).slice(0, mtry) : all, par = newton ? G * G / (H + lambda) : 0, pp = n1 / n;
    let best = null;
    for (const f of F) {
      const s = S[f]; let gl = 0, hl = 0, l1 = 0;
      for (let k = 0; k < n - 1; k++) {
        const i = s[k]; if (newton) { gl += g[i]; hl += h[i]; } else l1 += y[i];
        const a = X[i][f], b = X[s[k + 1]][f], nl = k + 1, nr = n - nl;
        if (a === b || nl < minLeaf || nr < minLeaf) continue;
        const gain = newton ? gl * gl / (hl + lambda) + (G - gl) ** 2 / (H - hl + lambda) - par
          : n * 2 * pp * (1 - pp) - 2 * l1 * (1 - l1 / nl) - 2 * (n1 - l1) * (1 - (n1 - l1) / nr);
        if (!best || gain > best.gain + 1e-12) best = { f, thr: (a + b) / 2, gain };
      }
    }
    if (!best || best.gain <= 1e-12) return node;
    node.f = best.f; node.thr = best.thr; node.gain = best.gain;
    const bf = best.f, bt = best.thr, goL = (i) => X[i][bf] < bt, deep = depth + 1 < maxDepth;
    node.left = grow(deep ? S.map((l) => l.filter(goL)) : [S[0].filter(goL)], depth + 1);
    node.right = grow(deep ? S.map((l) => l.filter((i) => !goL(i))) : [S[0].filter((i) => !goL(i))], depth + 1);
    return node;
  };
  const ids0 = idx || X.map((_, i) => i);
  return grow(order && !idx ? order : all.map((f) => ids0.slice().sort((a, b) => X[a][f] - X[b][f])), 0);
}
export const treeValue = (nd, row) => { while (nd.left) nd = row[nd.f] < nd.thr ? nd.left : nd.right; return nd.v; };
/** Random forest: bootstrap resample + per-node random feature subset. Returns {trees:[{root, inbag}], importance (impurity, sums to 1), p}. */
export function fitForest(X, y, { nTrees = 100, maxDepth = 6, minLeaf = 1, mtry = null, seed = 1, bootstrap = true } = {}) {
  const r = rng(seed), n = X.length, p = X[0].length, trees = [], imp = new Array(p).fill(0);
  for (let t = 0; t < nTrees; t++) {
    const inbag = new Array(n).fill(0), idx = [];
    for (let i = 0; i < n; i++) { const j = bootstrap ? r.int(n) : i; inbag[j]++; idx.push(j); }
    const root = growTree(X, y, { idx, maxDepth, minLeaf, mtry: mtry || p, rand: r });
    (function walk(nd) { if (nd.left) { imp[nd.f] += nd.gain; walk(nd.left); walk(nd.right); } })(root);
    trees.push({ root, inbag });
  }
  const tot = imp.reduce((s, v) => s + v, 0) || 1;
  return { trees, importance: imp.map((v) => v / tot), p };
}
/** Mean of the first k trees' leaf values (probability of class 1). */
export const forestProb = (F, row, k = F.trees.length) => { let s = 0; for (let t = 0; t < k; t++) s += treeValue(F.trees[t].root, row); return s / k; };
/** Running forest score as trees are added. Returns S[k-1][i] = mean of the first k trees on row i; with oob = true only trees that did not see row i vote (NaN if none). */
export function forestStaged(F, X, oob = false) {
  const n = X.length, sum = new Array(n).fill(0), cnt = new Array(n).fill(0), S = [];
  F.trees.forEach((tr) => {
    for (let i = 0; i < n; i++) if (!oob || tr.inbag[i] === 0) { sum[i] += treeValue(tr.root, X[i]); cnt[i]++; }
    S.push(sum.map((s, i) => (cnt[i] ? s / cnt[i] : NaN)));
  });
  return S;
}
/** AUC / log-loss on the rows where the score is defined (not NaN). */
export function aucDefined(s, y) { const k = s.map((v, i) => i).filter((i) => !Number.isNaN(s[i])); return k.length ? auc(k.map((i) => s[i]), k.map((i) => y[i])) : NaN; }
/** Gradient boosting on log-loss with second-order (Newton) leaf values. Returns {f0, lr, trees:[root], p}. */
export function fitBoost(X, y, { nTrees = 100, lr = 0.1, maxDepth = 2, minLeaf = 3, lambda = 1, subsample = 1, seed = 1 } = {}) {
  const r = rng(seed), n = X.length, p0 = y.reduce((s, v) => s + v, 0) / n, f0 = Math.log(p0 / (1 - p0)), F = new Array(n).fill(f0), trees = [];
  const order = X[0].map((_, f) => X.map((q, i) => i).sort((a, b) => X[a][f] - X[b][f]));
  for (let t = 0; t < nTrees; t++) {
    const pr = F.map(sigmoid), g = pr.map((q, i) => q - y[i]), h = pr.map((q) => Math.max(q * (1 - q), 1e-6));
    const idx = subsample < 1 ? r.shuffle(X.map((_, i) => i)).slice(0, Math.round(subsample * n)) : null;
    const root = growTree(X, null, { mode: 'newton', g, h, maxDepth, minLeaf, lambda, idx, order });
    trees.push(root);
    for (let i = 0; i < n; i++) F[i] += lr * treeValue(root, X[i]);
  }
  return { f0, lr, trees, p: X[0].length };
}
export const boostRaw = (M, row, k = M.trees.length) => { let s = M.f0; for (let t = 0; t < k; t++) s += M.lr * treeValue(M.trees[t], row); return s; };
/** Raw boosted score after 0..T trees: S[k][i]. */
export function boostStaged(M, X) {
  const s = X.map(() => M.f0), out = [s.slice()];
  M.trees.forEach((root) => { X.forEach((r, i) => { s[i] += M.lr * treeValue(root, r); }); out.push(s.slice()); });
  return out;
}
/** Permutation importance: mean drop in AUC when one column is shuffled. scoreFn(matrix) -> scores. Returns [{drop, sd}] per column. */
export function permImportance(scoreFn, X, y, { reps = 5, seed = 3 } = {}) {
  const r = rng(seed), base = auc(scoreFn(X), y), p = X[0].length;
  return Array.from({ length: p }, (_, j) => {
    const d = [];
    for (let k = 0; k < reps; k++) { const col = r.shuffle(X.map((row) => row[j])); d.push(base - auc(scoreFn(X.map((row, i) => { const c = row.slice(); c[j] = col[i]; return c; })), y)); }
    const m = d.reduce((s, v) => s + v, 0) / reps;
    return { drop: m, sd: Math.sqrt(d.reduce((s, v) => s + (v - m) ** 2, 0) / Math.max(1, reps - 1)) };
  });
}
/** Exact interventional Shapley values for a few features: value of a coalition = mean prediction with those features fixed at x and the rest drawn from the background rows. */
export function exactShapley(predict, x, background) {
  const p = x.length, v = new Map();
  const val = (mask) => { if (!v.has(mask)) v.set(mask, background.reduce((s, b) => s + predict(b.map((q, j) => (mask >> j & 1 ? x[j] : q))), 0) / background.length); return v.get(mask); };
  const fact = [1]; for (let i = 1; i <= p; i++) fact.push(fact[i - 1] * i);
  const phi = Array.from({ length: p }, (_, j) => {
    let s = 0;
    for (let m = 0; m < (1 << p); m++) if (!(m >> j & 1)) { let k = 0; for (let q = 0; q < p; q++) k += m >> q & 1; s += fact[k] * fact[p - k - 1] / fact[p] * (val(m | (1 << j)) - val(m)); }
    return s;
  });
  return { phi, base: val(0), full: val((1 << p) - 1) };
}

// ---------- PCA (lane A helper, appended) ----------
/** Eigen-decomposition of a symmetric matrix (array of arrays) by cyclic Jacobi rotations. Returns {vals, vecs}: vals descending, vecs[m] = eigenvector m (array). */
export function symEig(A0) {
  const n = A0.length, A = A0.map((r) => r.slice()), V = A.map((_, i) => A.map((__, j) => (i === j ? 1 : 0)));
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += A[p][q] ** 2;
    if (off < 1e-22) break;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
      if (Math.abs(A[p][q]) < 1e-18) continue;
      const th = (A[q][q] - A[p][p]) / (2 * A[p][q]);
      const t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1)), c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < n; k++) { const a = A[k][p], b = A[k][q]; A[k][p] = c * a - s * b; A[k][q] = s * a + c * b; }
      for (let k = 0; k < n; k++) { const a = A[p][k], b = A[q][k]; A[p][k] = c * a - s * b; A[q][k] = s * a + c * b; }
      for (let k = 0; k < n; k++) { const a = V[k][p], b = V[k][q]; V[k][p] = c * a - s * b; V[k][q] = s * a + c * b; }
    }
  }
  const order = A.map((_, i) => i).sort((a, b) => A[b][b] - A[a][a]);
  return { vals: order.map((j) => Math.max(0, A[j][j])), vecs: order.map((j) => A.map((_, k) => V[k][j])) };
}

/**
 * Principal component analysis of rows X. opts.scale (default true): divide each column by its SD (correlation PCA); false = centre only (covariance PCA).
 * Returns {mu, sd, vals (variances), ratio, vecs[m][j] (loadings), Z (centred/scaled rows), S (scores), apply(row) -> centred/scaled row, project(row) -> scores}.
 * Sign convention: the largest-magnitude loading of each component is made positive.
 */
export function pca(X, { scale = true } = {}) {
  const n = X.length, p = X[0].length;
  const mu = Array.from({ length: p }, (_, j) => X.reduce((s, r) => s + r[j], 0) / n);
  const sd = mu.map((m, j) => (scale ? Math.sqrt(X.reduce((s, r) => s + (r[j] - m) ** 2, 0) / Math.max(1, n - 1)) || 1 : 1));
  const apply = (r) => r.map((v, j) => (v - mu[j]) / sd[j]);
  const Z = X.map(apply);
  const cov = Array.from({ length: p }, (_, a) => Array.from({ length: p }, (__, b) => Z.reduce((s, r) => s + r[a] * r[b], 0) / Math.max(1, n - 1)));
  const { vals, vecs } = symEig(cov);
  vecs.forEach((v) => { let big = 0; v.forEach((x, k) => { if (Math.abs(x) > Math.abs(v[big])) big = k; }); if (v[big] < 0) v.forEach((x, k) => { v[k] = -x; }); });
  const tot = vals.reduce((s, v) => s + v, 0);
  const project = (r) => vecs.map((v) => v.reduce((s, w, j) => s + w * r[j], 0));
  return { mu, sd, vals, ratio: vals.map((v) => v / tot), vecs, cov, Z, S: Z.map(project), apply, project: (row) => project(apply(row)) };
}

// ---------------------------------------------------------------------------------------------
// Clustering helpers (added for the clustering lesson). Rows are arrays of numbers; labels are integer arrays.
const d2 = (a, b) => { let s = 0; for (let j = 0; j < a.length; j++) s += (a[j] - b[j]) ** 2; return s; };
export const sqDist = d2;

/** Initial centroids: 'random' = k distinct rows; 'plusplus' = k-means++ (each new centre drawn with probability proportional to squared distance). */
export function kmeansInit(X, k, seed = 1, method = 'plusplus') {
  const r = rng(seed), n = X.length;
  if (method === 'random') return r.shuffle(Array.from({ length: n }, (_, i) => i)).slice(0, k).map((i) => X[i].slice());
  const C = [X[r.int(n)].slice()];
  while (C.length < k) {
    const w = X.map((x) => Math.min(...C.map((c) => d2(x, c)))), tot = w.reduce((s, v) => s + v, 0);
    let u = r() * tot, i = 0;
    while (i < n - 1 && u > w[i]) { u -= w[i]; i++; }
    C.push(X[i].slice());
  }
  return C;
}
export const kmeansAssign = (X, C) => X.map((x) => { let b = 0, bd = Infinity; C.forEach((c, j) => { const d = d2(x, c); if (d < bd) { bd = d; b = j; } }); return b; });
export function kmeansUpdate(X, lab, C) {
  const k = C.length, p = X[0].length, S = Array.from({ length: k }, () => new Array(p).fill(0)), N = new Array(k).fill(0);
  X.forEach((x, i) => { N[lab[i]]++; for (let j = 0; j < p; j++) S[lab[i]][j] += x[j]; });
  return S.map((s, c) => (N[c] ? s.map((v) => v / N[c]) : C[c].slice()));
}
export const inertia = (X, lab, C) => X.reduce((s, x, i) => s + d2(x, C[lab[i]]), 0);
/** Lloyd's algorithm with nInit restarts (best inertia kept). Returns {labels, C, inertia, iters}. */
export function kmeans(X, k, { nInit = 5, seed = 1, maxIter = 100, init = 'plusplus' } = {}) {
  let best = null;
  for (let t = 0; t < nInit; t++) {
    let C = kmeansInit(X, k, seed * 100 + t, init), lab = kmeansAssign(X, C), it = 0;
    for (; it < maxIter; it++) {
      C = kmeansUpdate(X, lab, C);
      const nl = kmeansAssign(X, C);
      if (nl.every((v, i) => v === lab[i])) break;
      lab = nl;
    }
    const J = inertia(X, lab, C);
    if (!best || J < best.inertia) best = { labels: lab, C, inertia: J, iters: it + 1 };
  }
  return best;
}
/** Silhouette of every row (and the mean). Rows in singleton clusters score 0. */
export function silhouette(X, lab) {
  const k = Math.max(...lab) + 1, n = X.length, s = new Array(n).fill(0), cnt = new Array(k).fill(0);
  lab.forEach((c) => { cnt[c]++; });
  for (let i = 0; i < n; i++) {
    const sum = new Array(k).fill(0);
    for (let j = 0; j < n; j++) if (j !== i) sum[lab[j]] += Math.sqrt(d2(X[i], X[j]));
    if (cnt[lab[i]] < 2) continue;
    const a = sum[lab[i]] / (cnt[lab[i]] - 1);
    let b = Infinity;
    for (let c = 0; c < k; c++) if (c !== lab[i] && cnt[c]) b = Math.min(b, sum[c] / cnt[c]);
    s[i] = b === Infinity ? 0 : (b - a) / Math.max(a, b);
  }
  return { s, mean: s.reduce((u, v) => u + v, 0) / n };
}
/** Adjusted Rand index between two labelings (any label values). 1 = identical partitions, about 0 = chance. */
export function ari(a, b) {
  const n = a.length, tab = new Map(), ra = new Map(), rb = new Map(), c2 = (x) => x * (x - 1) / 2;
  for (let i = 0; i < n; i++) { const key = a[i] + '|' + b[i]; tab.set(key, (tab.get(key) || 0) + 1); ra.set(a[i], (ra.get(a[i]) || 0) + 1); rb.set(b[i], (rb.get(b[i]) || 0) + 1); }
  let s = 0, sa = 0, sb = 0;
  for (const v of tab.values()) s += c2(v);
  for (const v of ra.values()) sa += c2(v);
  for (const v of rb.values()) sb += c2(v);
  const exp = sa * sb / c2(n), mx = (sa + sb) / 2;
  return mx === exp ? 1 : (s - exp) / (mx - exp);
}
/** Agglomerative clustering. linkage: 'ward' | 'average' | 'complete' | 'single'. Returns merges [{a, b, h, size}]; merge t creates node n + t. */
export function agglomerate(X, linkage = 'ward') {
  const n = X.length, D = new Float64Array(n * n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) D[i * n + j] = linkage === 'ward' ? d2(X[i], X[j]) : Math.sqrt(d2(X[i], X[j]));
  const active = Array.from({ length: n }, (_, i) => i), size = new Array(n).fill(1), id = active.slice(), merges = [];
  while (active.length > 1) {
    let bi = 0, bj = 1, bd = Infinity;
    for (let u = 0; u < active.length; u++) for (let v = u + 1; v < active.length; v++) { const d = D[active[u] * n + active[v]]; if (d < bd) { bd = d; bi = u; bj = v; } }
    const I = active[bi], J = active[bj], ni = size[I], nj = size[J];
    merges.push({ a: id[I], b: id[J], h: linkage === 'ward' ? Math.sqrt(2 * bd * ni * nj / (ni + nj)) : bd, size: ni + nj });
    for (const K of active) {
      if (K === I || K === J) continue;
      const dik = D[I * n + K], djk = D[J * n + K], nk = size[K];
      let d;
      if (linkage === 'single') d = Math.min(dik, djk);
      else if (linkage === 'complete') d = Math.max(dik, djk);
      else if (linkage === 'average') d = (ni * dik + nj * djk) / (ni + nj);
      else d = ((ni + nk) * dik + (nj + nk) * djk - nk * D[I * n + J]) / (ni + nj + nk);
      D[I * n + K] = D[K * n + I] = d;
    }
    size[I] = ni + nj; id[I] = n + merges.length - 1;
    active.splice(bj, 1);
  }
  return merges;
}
/** Labels (0..k-1, in order of first appearance) from applying all merges with height <= cut. */
export function cutTree(n, merges, cut) {
  const par = Array.from({ length: 2 * n - 1 }, (_, i) => i), find = (i) => { while (par[i] !== i) i = par[i] = par[par[i]]; return i; };
  merges.forEach((m, t) => { if (m.h <= cut) { par[find(m.a)] = n + t; par[find(m.b)] = n + t; } });
  const ids = new Map();
  return Array.from({ length: n }, (_, i) => { const r = find(i); if (!ids.has(r)) ids.set(r, ids.size); return ids.get(r); });
}
/** Height that yields exactly k clusters. */
export const cutForK = (merges, k) => (k <= 1 ? merges[merges.length - 1].h * 1.01 : (merges[merges.length - k].h + merges[merges.length - k + 1].h) / 2);
/** Symmetric (union) kNN graph: edge i-j if j is among the k nearest of i or vice versa. Returns {edges, order}. */
export function knnGraph(X, k) {
  const n = X.length, set = new Set();
  for (let i = 0; i < n; i++) {
    const o = Array.from({ length: n }, (_, j) => j).filter((j) => j !== i).map((j) => [j, d2(X[i], X[j])]).sort((a, b) => a[1] - b[1]);
    for (let q = 0; q < k; q++) { const j = o[q][0]; set.add(i < j ? i * n + j : j * n + i); }
  }
  return [...set].map((key) => [Math.floor(key / n), key % n]);
}
/** Modularity with resolution gamma. */
export function modularity(n, edges, part, gamma = 1) {
  const m2 = 2 * edges.length, deg = new Float64Array(n);
  edges.forEach(([i, j]) => { deg[i]++; deg[j]++; });
  const K = Math.max(...part) + 1, inW = new Float64Array(K), tot = new Float64Array(K);
  edges.forEach(([i, j]) => { if (part[i] === part[j]) inW[part[i]] += 2; });
  for (let i = 0; i < n; i++) tot[part[i]] += deg[i];
  let q = 0; for (let c = 0; c < K; c++) q += inW[c] / m2 - gamma * (tot[c] / m2) ** 2;
  return q;
}
/** Louvain community detection (local moving + aggregation) with resolution gamma. Returns labels ranked by size (0 = largest). */
export function louvain(n, edges, gamma = 1, seed = 1) {
  const r = rng(seed);
  let adj = Array.from({ length: n }, () => new Map());
  edges.forEach(([i, j]) => { adj[i].set(j, (adj[i].get(j) || 0) + 1); adj[j].set(i, (adj[j].get(i) || 0) + 1); });
  const m2 = 2 * edges.length;
  let member = Array.from({ length: n }, (_, i) => i);
  for (let level = 0; level < 20; level++) {
    const N = adj.length, deg = new Float64Array(N), comm = Int32Array.from({ length: N }, (_, i) => i), tot = new Float64Array(N);
    for (let u = 0; u < N; u++) { for (const w of adj[u].values()) deg[u] += w; tot[u] = deg[u]; }
    let movedAny = false;
    const nodes = r.shuffle(Array.from({ length: N }, (_, i) => i));
    for (let pass = 0; pass < 50; pass++) {
      let moved = false;
      for (const u of nodes) {
        const cu = comm[u], kin = new Map();
        for (const [v, w] of adj[u]) if (v !== u) kin.set(comm[v], (kin.get(comm[v]) || 0) + w);
        tot[cu] -= deg[u];
        let best = cu, bestGain = (kin.get(cu) || 0) - gamma * tot[cu] * deg[u] / m2;
        for (const [c, w] of kin) { const g = w - gamma * tot[c] * deg[u] / m2; if (g > bestGain + 1e-12) { bestGain = g; best = c; } }
        tot[best] += deg[u];
        if (best !== cu) { comm[u] = best; moved = true; movedAny = true; }
      }
      if (!moved) break;
    }
    if (!movedAny) break;
    const ids = new Map(); comm.forEach((c) => { if (!ids.has(c)) ids.set(c, ids.size); });
    const newAdj = Array.from({ length: ids.size }, () => new Map());
    for (let u = 0; u < N; u++) { const cu = ids.get(comm[u]); for (const [v, w] of adj[u]) { const cv = ids.get(comm[v]); newAdj[cu].set(cv, (newAdj[cu].get(cv) || 0) + w); } }
    member = member.map((c) => ids.get(comm[c]));
    adj = newAdj;
  }
  const cnt = new Map(); member.forEach((c) => cnt.set(c, (cnt.get(c) || 0) + 1));
  const rank = new Map([...cnt].sort((a, b) => b[1] - a[1] || a[0] - b[0]).map(([c], i) => [c, i]));
  return member.map((c) => rank.get(c));
}
/** Assign new rows to clusters by majority vote of their k nearest labelled rows (label transfer). */
export function knnTransfer(Xtrain, lab, Xnew, k = 5) {
  return Xnew.map((x) => {
    const nn = Xtrain.map((t, i) => [i, d2(x, t)]).sort((a, b) => a[1] - b[1]).slice(0, k), v = new Map();
    nn.forEach(([i]) => v.set(lab[i], (v.get(lab[i]) || 0) + 1));
    return [...v].sort((a, b) => b[1] - a[1] || a[0] - b[0])[0][0];
  });
}

// ---------------------------------------------------------------------------------------------
// Small multilayer perceptron for binary classification (neural-network lesson).
// Hidden layers use relu / tanh / sigmoid; the output unit is a sigmoid; the loss is binary cross-entropy.
// Training: mini-batch Adam with optional L2 weight decay and inverted dropout on the hidden units.
export const MLP_ACT = {
  relu: { f: (z) => (z > 0 ? z : 0), d: (z) => (z > 0 ? 1 : 0), name: 'ReLU' },
  tanh: { f: Math.tanh, d: (z, a) => 1 - a * a, name: 'tanh' },
  sigmoid: { f: sigmoid, d: (z, a) => a * (1 - a), name: 'sigmoid' },
};

/** sizes = [nIn, h1, ..., 1]. opts: {act, seed, l2, dropout} (l2 and dropout can be changed later through net.opts). */
export function makeMLP(sizes, { act = 'relu', seed = 1, l2 = 0, dropout = 0 } = {}) {
  const r = rng(seed), shuf = rng(seed + 1000), drop = rng(seed + 2000), A_ = MLP_ACT[act], L = sizes.length - 1;
  const opts = { l2, dropout };
  const W = [], b = [], gW = [], gb = [], mW = [], vW = [], mb = [], vb = [];
  for (let l = 0; l < L; l++) {
    const nin = sizes[l], nout = sizes[l + 1], s = Math.sqrt(2 / (nin + nout)) * (act === 'relu' && l < L - 1 ? 1.4 : 1);
    W.push(Float64Array.from({ length: nin * nout }, () => r.normal(0, s)));
    b.push(new Float64Array(nout));
    [gW, mW, vW].forEach((a) => a.push(new Float64Array(nin * nout)));
    [gb, mb, vb].forEach((a) => a.push(new Float64Array(nout)));
  }
  const Z = sizes.map((n) => new Float64Array(n)), P = sizes.map((n) => new Float64Array(n)), A = sizes.map((n) => new Float64Array(n));
  const M = sizes.map((n) => new Float64Array(n).fill(1)), D = sizes.map((n) => new Float64Array(n));
  let t = 0;
  function run(x, train) {
    A[0].set(x);
    for (let l = 0; l < L; l++) {
      const nin = sizes[l], nout = sizes[l + 1], w = W[l], last = l === L - 1;
      for (let i = 0; i < nout; i++) {
        let z = b[l][i];
        for (let j = 0; j < nin; j++) z += w[i * nin + j] * A[l][j];
        Z[l + 1][i] = z;
        if (last) { A[l + 1][i] = sigmoid(z); continue; }
        const a = A_.f(z); P[l + 1][i] = a;
        M[l + 1][i] = train && opts.dropout > 0 ? (drop() >= opts.dropout ? 1 / (1 - opts.dropout) : 0) : 1;
        A[l + 1][i] = a * M[l + 1][i];
      }
    }
    return A[L][0];
  }
  /** Accumulate the mean gradient over the sample indices idx into gW, gb. */
  function accumulate(X, Y, idx, train) {
    for (let l = 0; l < L; l++) { gW[l].fill(0); gb[l].fill(0); }
    for (const n of idx) {
      run(X[n], train);
      D[L][0] = A[L][0] - Y[n];
      for (let l = L - 1; l >= 0; l--) {
        const nin = sizes[l], nout = sizes[l + 1], w = W[l];
        for (let i = 0; i < nout; i++) { const d = D[l + 1][i]; gb[l][i] += d; for (let j = 0; j < nin; j++) gW[l][i * nin + j] += d * A[l][j]; }
        if (l > 0) for (let j = 0; j < nin; j++) { let s = 0; for (let i = 0; i < nout; i++) s += w[i * nin + j] * D[l + 1][i]; D[l][j] = s * M[l][j] * A_.d(Z[l][j], P[l][j]); }
      }
    }
    const m = idx.length;
    for (let l = 0; l < L; l++) { for (let k = 0; k < gW[l].length; k++) gW[l][k] /= m; for (let k = 0; k < gb[l].length; k++) gb[l][k] /= m; }
  }
  const order = [];
  function epoch(X, Y, lr = 0.01, batch = 32) {
    if (order.length !== X.length) { order.length = 0; for (let i = 0; i < X.length; i++) order.push(i); }
    for (let i = order.length - 1; i > 0; i--) { const j = Math.floor(shuf() * (i + 1)); const q = order[i]; order[i] = order[j]; order[j] = q; }
    for (let s = 0; s < order.length; s += batch) {
      accumulate(X, Y, order.slice(s, s + batch), true);
      t++;
      const c1 = 1 - 0.9 ** t, c2 = 1 - 0.999 ** t;
      for (let l = 0; l < L; l++) {
        const step = (g, m, v, p, k, decay) => { const gk = g[k] + decay * p[k]; m[k] = 0.9 * m[k] + 0.1 * gk; v[k] = 0.999 * v[k] + 0.001 * gk * gk; p[k] -= lr * (m[k] / c1) / (Math.sqrt(v[k] / c2) + 1e-8); };
        for (let k = 0; k < W[l].length; k++) step(gW[l], mW[l], vW[l], W[l], k, opts.l2);
        for (let k = 0; k < b[l].length; k++) step(gb[l], mb[l], vb[l], b[l], k, 0);
      }
    }
  }
  const forward = (x) => run(x, false);
  const predict = (X) => X.map(forward);
  const snapshot = () => ({ W: W.map((w) => w.slice()), b: b.map((v) => v.slice()) });
  const restore = (s) => { s.W.forEach((w, l) => W[l].set(w)); s.b.forEach((v, l) => b[l].set(v)); };
  const nParams = W.reduce((s, w) => s + w.length, 0) + b.reduce((s, v) => s + v.length, 0);
  return { sizes, act, opts, W, b, forward, predict, epoch, snapshot, restore, nParams, gradient: (X, Y, idx) => { accumulate(X, Y, idx, false); return { gW, gb }; } };
}
