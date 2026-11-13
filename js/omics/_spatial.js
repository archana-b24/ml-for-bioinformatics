// Helpers shared by the spatial pages of Tab 2 (sp-*). Not a page (no registry id).
// Tissue maps are drawn on <canvas> (thousands of cells); colours are read from CSS tokens at draw time.
import { el, tip, untip } from '../lib/ui.js';
import { cat, css, muted } from '../lib/plot.js';
const d3 = window.d3;

export const CT_SHORT = ['Tumour', 'CD8 T', 'CD4 T', 'Treg', 'B', 'Macrophage', 'Fibroblast', 'Endothelial'];
/** Cell type t -> categorical slot t (fixed order, always shown with a legend). */
export const ctColor = (t) => cat(t);
/** Region r (Tumour nest, Margin, Stroma, TLS) -> slot. Nest and stroma reuse the tumour / fibroblast slots. */
export const regionColor = (r) => cat([0, 3, 6, 4][r]);
export const secShort = (s) => s.split(' ')[0];
export const fmt = (v, d = 2) => (+v).toFixed(d);
export const pct = (v, d = 0) => (100 * v).toFixed(d) + '%';

/**
 * Canvas tissue map. size = tissue width in µm (square). Returns { root, draw(n, colorFn, radiusFn?, alphaFn?), overlay(fn), hover(fn) }.
 * hover(fn): fn(i) -> html for the nearest cell within 12 µm (or null).
 */
export function tissueMap(parent, { size = 1000, px = 440, label = '', scaleBar = 100 } = {}) {
  const root = el('div', { style: 'position:relative;' });
  if (label) root.append(el('div', { style: 'font-size:13px;font-weight:600;color:var(--ink-2);margin:0 0 4px 2px;' }, label));
  const cv = el('canvas', { style: `width:100%;max-width:${px}px;aspect-ratio:1;border-radius:8px;background:var(--surface-2);cursor:crosshair;` });
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  cv.width = px * dpr; cv.height = px * dpr;
  root.append(cv);
  parent.append(root);
  const ctx = cv.getContext('2d');
  const k = (px * dpr) / size;
  let xs = [], ys = [], hoverFn = null, last = null, overlayFn = null;
  const grid = new Map(), G = 25;
  function index() {
    grid.clear();
    xs.forEach((x, i) => { const key = Math.floor(x / G) + ',' + Math.floor(ys[i] / G); (grid.get(key) || grid.set(key, []).get(key)).push(i); });
  }
  function nearest(x, y, rmax = 12) {
    let best = -1, bd = rmax * rmax;
    const gx = Math.floor(x / G), gy = Math.floor(y / G);
    for (let a = -1; a <= 1; a++) for (let b = -1; b <= 1; b++) for (const i of grid.get((gx + a) + ',' + (gy + b)) || []) {
      const d = (xs[i] - x) ** 2 + (ys[i] - y) ** 2; if (d < bd) { bd = d; best = i; }
    }
    return best;
  }
  function draw(x, y, colorFn, radiusFn = () => 5, alphaFn = null) {
    if (x !== xs) { xs = x; ys = y; index(); }
    last = [colorFn, radiusFn, alphaFn];
    ctx.clearRect(0, 0, cv.width, cv.height);
    for (let i = 0; i < xs.length; i++) {
      const c = colorFn(i); if (!c) continue;
      ctx.globalAlpha = alphaFn ? alphaFn(i) : 1;
      ctx.fillStyle = c;
      ctx.beginPath(); ctx.arc(xs[i] * k, ys[i] * k, radiusFn(i) * k, 0, 6.2832); ctx.fill();
    }
    ctx.globalAlpha = 1;
    if (overlayFn) overlayFn(ctx, k);
    // scale bar
    if (scaleBar) {
      const sb = scaleBar * k, x0 = cv.width - sb - 12 * dpr, y0 = cv.height - 12 * dpr;
      ctx.fillStyle = css('--ink'); ctx.fillRect(x0, y0, sb, 3 * dpr);
      ctx.font = `${11 * dpr}px system-ui, sans-serif`; ctx.textAlign = 'center';
      ctx.fillText(`${scaleBar} µm`, x0 + sb / 2, y0 - 5 * dpr);
    }
  }
  cv.addEventListener('mousemove', (e) => {
    if (!hoverFn) return;
    const r = cv.getBoundingClientRect();
    const x = (e.clientX - r.left) / r.width * size, y = (e.clientY - r.top) / r.height * size;
    const h = hoverFn(x, y, nearest);
    if (h) tip(e, h); else untip();
  });
  cv.addEventListener('mouseleave', untip);
  return {
    root, canvas: cv, ctx, k, draw,
    redraw: () => last && draw(xs, ys, ...last),
    overlay: (fn) => { overlayFn = fn; },
    hover: (fn) => { hoverFn = fn; },
    nearest,
  };
}

/** Expression colour: sequential surface-2 -> accent on log1p counts, scaled to the 99th percentile. */
export function exprScale(values) {
  const lv = values.map((v) => Math.log1p(v)).sort((a, b) => a - b);
  const hi = Math.max(lv[Math.floor(0.99 * (lv.length - 1))], 0.5);
  const s = d3.scaleSequential(d3.interpolateRgb(css('--surface-2'), css('--accent'))).domain([0, hi]).clamp(true);
  const lo = d3.interpolateRgb(css('--muted'), css('--accent'));
  return { color: (v) => (v > 0 ? lo(Math.min(1, 0.25 + 0.75 * Math.log1p(v) / hi)) : null), hi, base: s };
}

export const mutedColor = () => muted();

/** Delaunay edges (pairs [i, j], i < j) no longer than maxlen µm, as in the build script. */
export function delaunayEdges(x, y, maxlen = 35) {
  const n = x.length, del = d3.Delaunay.from(d3.range(n), (i) => x[i], (i) => y[i]);
  const t = del.triangles, seen = new Set(), e = [];
  for (let q = 0; q < t.length; q += 3) for (const [a, b] of [[t[q], t[q + 1]], [t[q + 1], t[q + 2]], [t[q], t[q + 2]]]) {
    const lo = Math.min(a, b), hi = Math.max(a, b), key = lo * n + hi;
    if (seen.has(key)) continue; seen.add(key);
    if (Math.hypot(x[a] - x[b], y[a] - y[b]) <= maxlen) e.push([lo, hi]);
  }
  return e;
}
/** Neighbour lists from edges. */
export function neighbours(n, edges) {
  const nb = Array.from({ length: n }, () => []);
  edges.forEach(([a, b]) => { nb[a].push(b); nb[b].push(a); });
  return nb;
}
/** log1p(counts / panel total * 100), the normalisation used by the build script. */
export const lognorm = (counts, lib) => counts.map((c, i) => Math.log1p(c / Math.max(lib[i], 1) * 100));
/** Moran's I with row-normalised weights. v: values, nb: neighbour lists. Returns { I, z, lag }. */
export function moran(v, nb) {
  const n = v.length, m = d3.mean(v), z = v.map((a) => a - m);
  const lag = nb.map((l) => (l.length ? d3.sum(l, (j) => z[j]) / l.length : 0));
  let num = 0, den = 0;
  for (let i = 0; i < n; i++) { num += z[i] * lag[i]; den += z[i] * z[i]; }
  return { I: den ? num / den : 0, z, lag };
}
