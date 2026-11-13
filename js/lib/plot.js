// d3 helpers. Colours always come from CSS tokens (read at draw time) so light/dark both work.
const d3 = window.d3;

export const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
/** Categorical colour by slot index (0-based), fixed order, never cycled past 8. */
export const cat = (i) => css(`--s${(i % 8) + 1}`);
export const CAT_N = 8;
export const ink = () => css('--ink');
export const ink2 = () => css('--ink-2');
export const muted = () => css('--muted');
export const surface = () => css('--surface');
/** Diverging scale (blue - grey - red) over [-m, m]. */
export function diverging(m = 1) {
  return d3.scaleLinear().domain([-m, 0, m]).range([css('--div-neg'), css('--div-mid'), css('--div-pos')]).clamp(true);
}
/** Sequential single-hue scale (surface -> accent). */
export function sequential(lo = 0, hi = 1) {
  return d3.scaleLinear().domain([lo, hi]).range([css('--surface-2'), css('--accent')]).clamp(true);
}

/**
 * Create a responsive SVG chart frame.
 * opts: { width=560, height=380, margin, x:[lo,hi], y:[lo,hi], xLabel, yLabel, grid=true, axes=true }
 * Returns { svg, g (plot group), x, y, w, h, layer(name) }
 */
export function frame(parent, opts = {}) {
  const width = opts.width || 560, height = opts.height || 380;
  const m = Object.assign({ top: 12, right: 14, bottom: opts.xLabel ? 42 : 28, left: opts.yLabel ? 50 : 38 }, opts.margin || {});
  const w = width - m.left - m.right, h = height - m.top - m.bottom;
  const svg = d3.select(parent).append('svg').attr('viewBox', `0 0 ${width} ${height}`).attr('width', width).attr('role', 'img');
  if (opts.ariaLabel) svg.attr('aria-label', opts.ariaLabel);
  const g = svg.append('g').attr('transform', `translate(${m.left},${m.top})`);
  const x = (opts.xScale || d3.scaleLinear()).domain(opts.x || [-1.2, 1.2]).range([0, w]);
  const y = (opts.yScale || d3.scaleLinear()).domain(opts.y || [-1.2, 1.2]).range([h, 0]);
  if (opts.grid !== false) {
    g.append('g').attr('class', 'gridline').call(d3.axisLeft(y).ticks(5).tickSize(-w).tickFormat(''));
  }
  if (opts.axes !== false) {
    g.append('g').attr('class', 'axis x').attr('transform', `translate(0,${h})`).call(d3.axisBottom(x).ticks(opts.xTicks || 5));
    g.append('g').attr('class', 'axis y').call(d3.axisLeft(y).ticks(opts.yTicks || 5));
  }
  if (opts.xLabel) g.append('text').attr('class', 'ax-label').attr('x', w / 2).attr('y', h + 36).attr('text-anchor', 'middle').text(opts.xLabel);
  if (opts.yLabel) g.append('text').attr('class', 'ax-label').attr('transform', 'rotate(-90)').attr('x', -h / 2).attr('y', -38).attr('text-anchor', 'middle').text(opts.yLabel);
  const layers = {};
  const layer = (name) => layers[name] || (layers[name] = g.append('g').attr('class', 'layer-' + name));
  return { svg, g, x, y, w, h, layer, margin: m };
}

/** Scatter points [{x,y,label}] into a layer. colorFn(d) optional. Returns selection. */
export function scatter(layer, data, x, y, { r = 4.5, color = (d) => cat(d.label || 0), stroke = true, onHover } = {}) {
  const sel = layer.selectAll('circle').data(data).join('circle')
    .attr('cx', (d) => x(d.x)).attr('cy', (d) => y(d.y)).attr('r', r)
    .attr('fill', color).attr('stroke', stroke ? surface() : 'none').attr('stroke-width', 1.2);
  if (onHover) sel.on('mousemove', onHover).on('mouseleave', () => window.__untip && window.__untip());
  return sel;
}

/**
 * Paint a decision surface on a <canvas> under an SVG plot.
 * predict(x, y) -> number in [0,1] (prob of class 1) or class index (if opts.classes).
 */
export function decisionCanvas(parentFrame, predict, { res = 4, classes = null, alpha = 0.28 } = {}) {
  const { x, y, w, h, svg, margin } = parentFrame;
  const fo = svg.insert('foreignObject', ':first-child').attr('x', margin.left).attr('y', margin.top).attr('width', w).attr('height', h);
  const canvas = document.createElement('canvas'); canvas.width = Math.ceil(w / res); canvas.height = Math.ceil(h / res);
  canvas.style.width = w + 'px'; canvas.style.height = h + 'px'; canvas.style.imageRendering = 'auto';
  fo.node().append(canvas);
  const ctx = canvas.getContext('2d');
  const draw = (fn = predict) => {
    const c0 = d3.color(cat(0)), c1 = d3.color(cat(1));
    const img = ctx.createImageData(canvas.width, canvas.height);
    for (let j = 0; j < canvas.height; j++) for (let i = 0; i < canvas.width; i++) {
      const px = x.invert(i * res + res / 2), py = y.invert(j * res + res / 2);
      let col, a = alpha;
      if (classes) { col = d3.color(cat(fn(px, py))); }
      else { const p = fn(px, py); col = p >= 0.5 ? c1 : c0; a = alpha * Math.min(1, Math.abs(p - 0.5) * 2.2 + 0.15); }
      const k = 4 * (j * canvas.width + i);
      img.data[k] = col.r; img.data[k + 1] = col.g; img.data[k + 2] = col.b; img.data[k + 3] = 255 * a;
    }
    ctx.putImageData(img, 0, 0);
  };
  draw();
  return { draw, canvas };
}

/** Simple line path helper. pts [{x,y}] */
export function line(layer, pts, x, y, { color = ink(), width = 2, dash = null, cls = '' } = {}) {
  const p = layer.append('path').datum(pts).attr('class', cls).attr('fill', 'none').attr('stroke', color).attr('stroke-width', width)
    .attr('d', d3.line().x((d) => x(d.x)).y((d) => y(d.y)));
  if (dash) p.attr('stroke-dasharray', dash);
  return p;
}

/** Heatmap of a 2D array M[rows][cols] into a frame-less svg. */
export function heatmap(parent, M, { cell = 18, color = null, rowLabels = null, colLabels = null, fmt = null, labelW = 70, labelH = 0, onHover = null } = {}) {
  const rows = M.length, cols = M[0].length;
  const top = colLabels ? (labelH || 60) : 4, left = rowLabels ? labelW : 4;
  const svg = d3.select(parent).append('svg').attr('viewBox', `0 0 ${left + cols * cell + 4} ${top + rows * cell + 4}`).attr('width', left + cols * cell + 4);
  const flat = M.flat();
  const sc = color || sequential(d3.min(flat), d3.max(flat));
  const g = svg.append('g').attr('transform', `translate(${left},${top})`);
  M.forEach((r, i) => r.forEach((v, j) => {
    const rect = g.append('rect').attr('x', j * cell).attr('y', i * cell).attr('width', cell - 1).attr('height', cell - 1).attr('rx', 2).attr('fill', sc(v));
    if (onHover) rect.on('mousemove', (e) => onHover(e, i, j, v)).on('mouseleave', () => window.__untip && window.__untip());
    if (fmt) g.append('text').attr('x', j * cell + cell / 2).attr('y', i * cell + cell / 2 + 4).attr('text-anchor', 'middle').attr('font-size', 10).attr('fill', ink()).text(fmt(v));
  }));
  if (rowLabels) rowLabels.forEach((t, i) => svg.append('text').attr('x', left - 6).attr('y', top + i * cell + cell / 2 + 4).attr('text-anchor', 'end').attr('font-size', 11).attr('fill', ink2()).text(t));
  if (colLabels) colLabels.forEach((t, j) => svg.append('text').attr('transform', `translate(${left + j * cell + cell / 2 + 4},${top - 6}) rotate(-50)`).attr('font-size', 11).attr('fill', ink2()).text(t));
  return { svg, g, scale: sc };
}
