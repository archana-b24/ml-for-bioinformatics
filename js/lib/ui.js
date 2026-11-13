// DOM + page-anatomy helpers shared by every page. Keep pages declarative: header(), figure(), slider(), explain blocks.

export function el(tag, attrs = {}, ...children) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === 'html') n.innerHTML = v;
    else if (k.startsWith('on') && typeof v === 'function') n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, v === true ? '' : v);
  }
  for (const c of children.flat()) if (c != null && c !== false) n.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return n;
}

/** Inline HTML string -> node (use for prose with <b>, <code>, <i> etc.). */
export const html = (s, tag = 'div', cls = '') => el(tag, { class: cls, html: s });

/** Page header: title + one-line intuition. */
export function header(main, title, lede) {
  main.append(el('h1', {}, title));
  if (lede) main.append(html(lede, 'p', 'lede'));
}

/** A figure card. Returns { root, body, controls, caption(text) }. */
export function figure(parent, title, captionHtml) {
  const root = el('div', { class: 'figure' });
  if (title) root.append(el('div', { class: 'fig-title' }, title));
  const controls = el('div', { class: 'controls' });
  const body = el('div', { class: 'fig-body' });
  const cap = html(captionHtml || '', 'div', 'fig-caption');
  root.append(controls, body, cap);
  parent.append(root);
  return { root, body, controls, caption: (h) => { cap.innerHTML = h; } };
}

/** Range slider. onInput(value) fires live. Returns { input, set(v), get() }. */
export function slider(parent, { label, min, max, step = 1, value, fmt = (v) => v, onInput }) {
  const input = el('input', { type: 'range', min, max, step, value, 'aria-label': label });
  const val = el('span', { class: 'val' }, fmt(+value));
  input.addEventListener('input', () => { val.textContent = fmt(+input.value); onInput && onInput(+input.value); });
  parent.append(el('label', { class: 'control' }, label, input, val));
  return { input, get: () => +input.value, set: (v) => { input.value = v; val.textContent = fmt(+v); } };
}

export function select(parent, { label, options, value, onChange }) {
  const s = el('select', { 'aria-label': label });
  options.forEach((o) => { const [v, t] = Array.isArray(o) ? o : [o, o]; s.append(el('option', { value: v, selected: v === value }, t)); });
  s.addEventListener('change', () => onChange && onChange(s.value));
  parent.append(el('label', { class: 'control' }, label, s));
  return s;
}

export function button(parent, label, onClick, primary = false) {
  const b = el('button', { class: 'btn' + (primary ? ' primary' : '') }, label);
  b.addEventListener('click', onClick);
  parent.append(b);
  return b;
}

export function checkbox(parent, { label, checked, onChange }) {
  const c = el('input', { type: 'checkbox', checked });
  c.addEventListener('change', () => onChange && onChange(c.checked));
  parent.append(el('label', { class: 'control' }, c, label));
  return c;
}

/** Row of live numeric readouts. stats(parent) -> set({label: value}). */
export function stats(parent) {
  const root = el('div', { class: 'stats' });
  parent.append(root);
  return {
    root,
    set(obj) {
      root.innerHTML = '';
      for (const [k, v] of Object.entries(obj)) root.append(el('div', { class: 'stat' }, el('b', {}, v), k));
    },
  };
}

/** Standard explanation blocks. Pass HTML strings. */
export function section(main, title, bodyHtml) {
  main.append(el('h2', {}, title));
  const d = html(bodyHtml, 'div', 'prose');
  main.append(d);
  return d;
}
export function steps(main, title, list) {
  if (title) main.append(el('h2', {}, title));
  main.append(el('ol', { class: 'steps prose' }, list.map((s) => el('li', { html: s }))));
}
export function callout(parent, kind, label, bodyHtml) {
  // kind: '' | 'warn' | 'bad' | 'good'
  const c = el('div', { class: 'callout ' + (kind || '') }, el('span', { class: 'lbl' }, label), html(bodyHtml, 'div'));
  parent.append(c);
  return c;
}
export function math(parent, text) { const m = el('div', { class: 'math' }, text); parent.append(m); return m; }
export function legend(parent, entries) {
  // entries: [[label, cssColor], ...]
  const l = el('div', { class: 'legend' }, entries.map(([t, c]) => el('span', {}, el('i', { style: `background:${c}` }), t)));
  parent.append(l);
  return l;
}

/** "See also" links: [['concepts','pca','PCA'], ...] */
export function seeAlso(main, links) {
  const p = el('p', { class: 'prose' }, el('b', {}, 'See also: '));
  links.forEach(([tab, id, label], i) => { if (i) p.append(' · '); p.append(el('a', { href: `#/${tab}/${id}` }, label)); });
  main.append(p);
}

// ---------- animation registry: pages register loops; router stops them on navigation ----------
const loops = new Set();
/** Run fn(dt, t) each frame until it returns false or the page changes. Returns stop(). */
export function animate(fn) {
  let last = performance.now(), alive = true;
  const h = { stop: () => { alive = false; loops.delete(h); } };
  loops.add(h);
  const tick = (now) => {
    if (!alive) return;
    const r = fn(Math.min(now - last, 50), now); last = now;
    if (r === false) { h.stop(); return; }
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
  return h.stop;
}
/** setInterval that is cleared on navigation. */
export function every(ms, fn) {
  const id = setInterval(fn, ms);
  const h = { stop: () => { clearInterval(id); loops.delete(h); } };
  loops.add(h);
  return h.stop;
}
export function stopAll() { [...loops].forEach((h) => h.stop()); }

// ---------- tooltip ----------
const tipEl = () => document.getElementById('tooltip');
export function tip(evt, htmlStr) {
  const t = tipEl(); if (!t) return;
  t.innerHTML = htmlStr; t.style.opacity = 1;
  const x = Math.min(evt.clientX + 14, window.innerWidth - t.offsetWidth - 8);
  t.style.left = x + 'px'; t.style.top = (evt.clientY + 14) + 'px';
}
export function untip() { const t = tipEl(); if (t) t.style.opacity = 0; }

/** Fetch JSON from /data with simple caching. */
const cache = new Map();
export async function loadJSON(path) {
  if (!cache.has(path)) cache.set(path, fetch(path).then((r) => { if (!r.ok) throw new Error(`${path}: ${r.status}`); return r.json(); }));
  return cache.get(path);
}
window.__untip = untip;
