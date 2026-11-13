// Small helpers shared by the first bulk RNA-seq pages of Tab 2 (overview .. bulk-classifier). Not a page.
import { cat, muted } from '../lib/plot.js';
const d3 = window.d3;

export const PHENOS = ['inflamed', 'excluded', 'desert'];
/** Fixed colour meanings used on these pages (always shown with a legend). */
export const phenoColor = (p) => cat(PHENOS.indexOf(p));
export const CENTRES = ['Centre A', 'Centre B'];
export const centreColor = (c) => cat(c === 'Centre A' ? 6 : 3);
export const respColor = (r) => (r ? cat(7) : muted());

/** Gene group colours (the planted structure each gene belongs to). */
export const GROUP_ORDER = ['IFNG module', 'CAF/TGF-b module', 'PROLIF module', 'other module', 'cell-type marker', 'housekeeping', 'filler'];
export function groupColor(g) {
  return { 'IFNG module': cat(7), 'CAF/TGF-b module': cat(3), 'PROLIF module': cat(6), 'other module': cat(4), 'cell-type marker': cat(2), housekeeping: cat(0) }[g] || muted();
}
export const groupLabel = (g) => (g === 'CAF/TGF-b module' ? 'CAF/TGF-β module' : g);

export const median = (a) => { const s = [...a].sort((x, y) => x - y), m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };
/** Median absolute deviation, scaled to match the SD for normal data. */
export function mad(a) { const m = median(a); return 1.4826 * median(a.map((v) => Math.abs(v - m))); }

/** Mann-Whitney AUC of scores s for binary labels y (ties count 1/2). */
export function auc(y, s) {
  let pos = 0, neg = 0, num = 0;
  const idx = d3.range(s.length).sort((a, b) => s[a] - s[b]);
  const ranks = new Array(s.length);
  for (let i = 0; i < idx.length;) {
    let j = i; while (j + 1 < idx.length && s[idx[j + 1]] === s[idx[i]]) j++;
    for (let k = i; k <= j; k++) ranks[idx[k]] = (i + j) / 2 + 1;
    i = j + 1;
  }
  y.forEach((v, i) => { if (v) { pos++; num += ranks[i]; } else neg++; });
  return (num - pos * (pos + 1) / 2) / (pos * neg);
}

export const pct = (v, d = 0) => (100 * v).toFixed(d) + '%';
export const f2 = (v) => (+v).toFixed(2);
export const f3 = (v) => (+v).toFixed(3);
