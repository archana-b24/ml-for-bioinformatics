// Shared helpers for the single-cell pages of Tab 2 (sc-*). Not a page (no registry id).
import { frame, cat, css, muted } from '../lib/plot.js';
import { tip, untip } from '../lib/ui.js';
const d3 = window.d3;

export const CT_COLOR = (t) => (t < 0 ? muted() : cat(t));          // cell type t -> categorical slot t (8 types, 8 slots)
export const BATCH_COLOR = (b) => cat(b === 0 ? 0 : 1);
export const f2 = (v) => (+v).toFixed(2);
export const f3 = (v) => (+v).toFixed(3);
export const pct = (v, d = 0) => (100 * v).toFixed(d) + '%';
export const fmtInt = (v) => d3.format(',')(Math.round(v));

/**
 * Scatter of thousands of points: SVG axes from frame(), points painted on a <canvas> inside a foreignObject,
 * hover via a quadtree (nearest point within 8 px).
 * opts: frame options + { hover(i) -> html }.
 * Returns { F, draw({ idx, x, y, color, r, alpha }) }.
 */
export function canvasScatter(parent, opts) {
  const F = frame(parent, opts);
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const fo = F.g.insert('foreignObject', ':first-child').attr('x', 0).attr('y', 0).attr('width', F.w).attr('height', F.h);
  const canvas = document.createElement('canvas');
  canvas.width = Math.round(F.w * dpr); canvas.height = Math.round(F.h * dpr);
  canvas.style.width = F.w + 'px'; canvas.style.height = F.h + 'px';
  fo.node().append(canvas);
  const ctx = canvas.getContext('2d');
  const ring = F.layer('hover').append('circle').attr('r', 6).attr('fill', 'none').attr('stroke', css('--ink')).attr('stroke-width', 1.5).style('display', 'none');
  let qt = null, cur = null;
  function draw({ idx, x, y, color, r = 2.2, alpha = 0.85 }) {
    cur = { idx, x, y };
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, F.w, F.h);
    ctx.globalAlpha = alpha;
    const rf = typeof r === 'function' ? r : () => r;
    for (const i of idx) {
      const px = F.x(x[i]), py = F.y(y[i]);
      if (!(px >= -4 && px <= F.w + 4 && py >= -4 && py <= F.h + 4)) continue;
      ctx.fillStyle = color(i);
      ctx.beginPath(); ctx.arc(px, py, rf(i), 0, 2 * Math.PI); ctx.fill();
    }
    ctx.globalAlpha = 1;
    qt = d3.quadtree().x((i) => F.x(x[i])).y((i) => F.y(y[i])).addAll(idx);
  }
  if (opts.hover) {
    F.svg.on('mousemove', (e) => {
      if (!qt) return;
      const [mx, my] = d3.pointer(e, F.g.node());
      const i = qt.find(mx, my, 8);
      if (i == null) { ring.style('display', 'none'); untip(); if (opts.onMiss) opts.onMiss(); return; }
      ring.style('display', null).attr('cx', F.x(cur.x[i])).attr('cy', F.y(cur.y[i]));
      tip(e, opts.hover(i));
    }).on('mouseleave', () => { ring.style('display', 'none'); untip(); if (opts.onMiss) opts.onMiss(); });
  }
  return { F, draw, canvas };
}

/** Cluster colour: 8 token slots, then darker / lighter variants of the same tokens (derived, so dark mode follows). */
export const clusterColor = (c) => {
  const base = d3.color(cat(c % 8));
  const v = Math.floor(c / 8);
  return (v === 0 ? base : v === 1 ? base.darker(0.9) : base.brighter(0.8)).formatHex();
};

/** Padded extent of an array (ignores non-finite). */
export function ext(a, pad = 0.04) {
  const [lo, hi] = d3.extent(a.filter(Number.isFinite));
  const d = (hi - lo) || 1;
  return [lo - pad * d, hi + pad * d];
}

/** Simple horizontal bar list: rows [{label, value, color, note}] scaled to [0, max]. */
export function bars(parent, rows, { width = 460, max = null, fmt = (v) => v, labelW = 150, rowH = 24 } = {}) {
  const h = rows.length * rowH + 8;
  const svg = d3.select(parent).append('svg').attr('viewBox', `0 0 ${width} ${h}`).attr('width', width);
  const mx = max ?? d3.max(rows, (r) => r.value);
  const x = d3.scaleLinear().domain([0, mx || 1]).range([0, width - labelW - 60]);
  rows.forEach((r, i) => {
    const g = svg.append('g').attr('transform', `translate(0,${4 + i * rowH})`);
    g.append('text').attr('x', labelW - 8).attr('y', rowH / 2 + 4).attr('text-anchor', 'end').attr('font-size', 12).attr('fill', css('--ink-2')).text(r.label);
    g.append('rect').attr('x', labelW).attr('y', 4).attr('height', rowH - 8).attr('rx', 3).attr('width', Math.max(0, x(r.value))).attr('fill', r.color || css('--accent'));
    g.append('text').attr('x', labelW + x(r.value) + 6).attr('y', rowH / 2 + 4).attr('font-size', 12).attr('fill', css('--ink')).text(fmt(r.value));
  });
  return svg;
}
