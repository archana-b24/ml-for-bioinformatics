// Building blocks for the rewritten "lesson" pages. Every lesson follows one arc:
//   What is it → Why we need it → Where it is used → How it works (build-up steps, worked examples,
//   interactive lab and "what it means", all on the mock study) → Pitfalls → Recap
// See docs/AUTHORING.md ("Lesson pages").
import { el, html } from './ui.js';

export const ARC = [
  ['what', 'What is it'], ['why', 'Why we need it'], ['where', 'Where it is used'],
  ['how', 'How it works'], ['pitfalls', 'Pitfalls'], ['recap', 'Recap'],
];

/** Sticky chapter strip under the header; buttons scroll (hash routing owns the URL fragment). */
export function lessonNav(main, arc = ARC) {
  const nav = el('nav', { class: 'lesson-nav', 'aria-label': 'Sections on this page' });
  arc.forEach(([key, label], i) => {
    const b = el('button', { type: 'button' }, el('span', {}, String(i + 1)), label);
    b.onclick = () => { const t = document.getElementById('ch-' + key); if (t) t.scrollIntoView({ behavior: 'smooth', block: 'start' }); };
    nav.append(b);
  });
  main.append(nav);
  return nav;
}

/** Chapter heading with its number in the arc. Returns a container to append into. */
export function chapter(main, key, title, sub) {
  const i = ARC.findIndex(([k]) => k === key);
  const wrap = el('section', { class: 'chapter', id: 'ch-' + key });
  wrap.append(el('div', { class: 'ch-num' }, i >= 0 ? `${i + 1} · ${ARC[i][1]}` : ''));
  wrap.append(el('h2', {}, title));
  if (sub) wrap.append(html(sub, 'p', 'ch-sub'));
  main.append(wrap);
  return wrap;
}

/** Plain prose block. */
export const prose = (parent, h) => { const d = html(h, 'div', 'prose'); parent.append(d); return d; };

/** "Step N" rung of the build-up ladder. */
export function rung(parent, n, title, h) {
  const r = el('div', { class: 'rung' }, el('div', { class: 'rung-n' }, String(n)), el('div', { class: 'rung-b' }, el('h3', {}, title), html(h || '', 'div', 'prose')));
  parent.append(r);
  return r.querySelector('.rung-b');
}

/** Worked example box: a numbered, fully-computed example. */
export function worked(parent, title, h) {
  const w = el('div', { class: 'worked' }, el('div', { class: 'w-k' }, 'Worked example'), el('div', { class: 'w-t' }, title), html(h, 'div', 'prose'));
  parent.append(w);
  return w;
}

/** Self-check question with a hidden answer. */
export function quiz(parent, q, a) {
  const d = el('details', { class: 'quiz' }, el('summary', {}, el('span', { class: 'q-k' }, 'Check yourself'), ' ', html(q, 'span')), html(a, 'div', 'prose'));
  parent.append(d);
  return d;
}

/** Grid of example cards: items = [{tag, title, body}] */
export function cards(parent, items, cls = '') {
  const g = el('div', { class: 'ex-grid ' + cls });
  items.forEach((it) => g.append(el('div', { class: 'card ex-card' }, it.tag ? el('div', { class: 'ex-tag' }, it.tag) : null, el('div', { class: 'ex-t' }, it.title), html(it.body, 'div', 'ex-b'))));
  parent.append(g);
  return g;
}

/** Simple table. cols = [[key, label, fmt?]], rows = objects. opts.rowClass(row) for highlighting. */
export function table(parent, rows, cols, opts = {}) {
  const t = el('table', { class: 'simple ' + (opts.cls || '') });
  t.append(el('thead', {}, el('tr', {}, cols.map(([, label]) => el('th', { html: label })))));
  const tb = el('tbody');
  rows.forEach((r) => tb.append(el('tr', { class: opts.rowClass ? opts.rowClass(r) : null }, cols.map(([k, , f]) => el('td', { html: String(f ? f(r[k], r) : r[k]) })))));
  t.append(tb);
  const wrap = el('div', { class: 'table-wrap' }, t);
  parent.append(wrap);
  return { wrap, table: t, tbody: tb };
}

/** Code snippet with Python / R tabs. */
export function code(parent, variants) {
  const box = el('div', { class: 'code-box' });
  const bar = el('div', { class: 'code-tabs' });
  const pre = el('pre', { class: 'code' });
  const names = Object.keys(variants);
  const show = (k) => { pre.textContent = variants[k]; [...bar.children].forEach((b) => b.setAttribute('aria-selected', String(b.dataset.k === k))); };
  names.forEach((k) => { const b = el('button', { type: 'button', 'data-k': k }, k); b.onclick = () => show(k); bar.append(b); });
  box.append(bar, pre); parent.append(box); show(names[0]);
  return box;
}

/** The closing recap: one-line takeaways. */
export function recap(parent, items) {
  const r = el('div', { class: 'recap' }, el('ol', {}, items.map((s) => el('li', { html: s }))));
  parent.append(r);
  return r;
}
