// Hash router: #/<tab>/<topic>. Topic modules are lazy-imported so the first load stays small.
import { TABS, ARCHIVE, allItems } from './registry.js';
import { el, stopAll } from './lib/ui.js';

const $ = (s) => document.querySelector(s);
const tabsEl = $('#tabs'), navEl = $('#nav'), main = $('#main'), search = $('#search');

// theme toggle (per-viewer convenience only)
const THEME_KEY = 'mlatlas-theme';
try { const t = localStorage.getItem(THEME_KEY); if (t) document.documentElement.dataset.theme = t; } catch (e) { /* storage blocked */ }
$('#themeBtn').onclick = () => {
  const dark = getComputedStyle(document.documentElement).colorScheme.includes('dark');
  const next = dark ? 'light' : 'dark';
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem(THEME_KEY, next); } catch (e) { /* ignore */ }
  route(); // charts read CSS vars at draw time, so redraw
};
$('#menuBtn').onclick = () => document.body.classList.toggle('nav-open');

function parse() {
  const [, tabId, topicId] = location.hash.split('/');
  const tab = TABS.find((t) => t.id === tabId) || TABS[0];
  return { tab, topicId };
}

function drawTabs(active) {
  tabsEl.innerHTML = '';
  TABS.forEach((t) => {
    const b = el('button', { class: 'tab', role: 'tab', 'aria-selected': String(t === active) }, t.label);
    b.onclick = () => { location.hash = `#/${t.id}`; };
    tabsEl.append(b);
  });
}

function drawNav(tab, topicId) {
  navEl.innerHTML = '';
  const q = search.value.trim().toLowerCase();
  let n = 0;
  tab.sections.forEach((s) => {
    const items = s.items.filter(([id, title, blurb]) => !q || (title + ' ' + blurb).toLowerCase().includes(q));
    if (!items.length) { n += s.items.length; return; }
    navEl.append(el('div', { class: 'nav-section' }, s.name));
    s.items.forEach(([id, title, blurb]) => {
      n += 1;
      if (q && !(title + ' ' + blurb).toLowerCase().includes(q)) return;
      const a = el('a', { class: 'nav-item', href: `#/${tab.id}/${id}`, title: blurb },
        el('span', { class: 'num' }, String(n).padStart(2, '0')), title);
      if (id === topicId) a.setAttribute('aria-current', 'page');
      navEl.append(a);
    });
  });
  if (!tab.sections.length) navEl.append(el('p', { class: 'fig-caption', style: 'padding:0 8px' }, 'Topics are being added.'));
}

function home(tab) {
  main.innerHTML = '';
  main.append(el('div', { class: 'kicker' }, tab.label));
  main.append(el('h1', {}, { concepts: 'Machine learning, visually', omics: 'ML along an omics study', frontier: 'Frontier methods, rebuilt small' }[tab.id]));
  main.append(el('p', { class: 'lede' }, tab.intro));
  tab.sections.forEach((s) => {
    main.append(el('h2', {}, s.name));
    const g = el('div', { class: 'hero-grid' });
    s.items.forEach(([id, title, blurb]) => g.append(
      el('a', { class: 'card hero-card', href: `#/${tab.id}/${id}` }, el('div', { class: 't' }, title), el('div', { class: 'd' }, blurb))));
    main.append(g);
  });
}

let token = 0;
async function route() {
  stopAll(); // cancel animations from the previous page
  if (location.hash.startsWith('#/archive')) return routeArchive();
  const { tab, topicId } = parse();
  drawTabs(tab);
  drawNav(tab, topicId);
  document.body.classList.remove('nav-open');
  if (!topicId) { home(tab); document.title = `${tab.label.split('· ')[1]} · ML Visual Atlas`; return; }
  const items = allItems(tab);
  const idx = items.findIndex((i) => i.id === topicId);
  const item = items[idx];
  const my = ++token;
  main.innerHTML = '<p class="fig-caption">Loading…</p>';
  let mod;
  try { mod = await import(`./${tab.dir}/${topicId}.js`); } catch (e) { mod = null; if (!/Failed to fetch|Importing a module|error loading/i.test(String(e))) console.error(e); }
  if (my !== token) return;
  main.innerHTML = '';
  document.title = `${item ? item.title : topicId} · ML Visual Atlas`;
  if (item) main.append(el('div', { class: 'kicker' }, `${tab.label.split('· ')[1]} · ${item.section}`));
  if (!mod) {
    main.append(el('h1', {}, item ? item.title : topicId));
    main.append(el('p', { class: 'lede' }, item ? item.blurb : ''));
    main.append(el('div', { class: 'callout warn' }, 'This page is still being written.'));
  } else {
    try { await mod.default(main, { tab, item }); } catch (e) {
      console.error(e);
      main.append(el('div', { class: 'callout bad' }, 'This page hit an error while rendering: ' + e.message));
    }
  }
  const prev = items[idx - 1], next = items[idx + 1];
  main.append(el('nav', { class: 'pager' },
    prev ? el('a', { href: `#/${tab.id}/${prev.id}` }, '← ' + prev.title) : el('span'),
    next ? el('a', { href: `#/${tab.id}/${next.id}` }, next.title + ' →') : el('span')));
  main.focus({ preventScroll: true });
  window.scrollTo(0, 0);
}

// Retired pages: #/archive (index) and #/archive/<dir>/<id>. Not shown in the navigation.
async function routeArchive() {
  const [, , dir, id] = location.hash.split('/');
  drawTabs(null); navEl.innerHTML = ''; document.body.classList.remove('nav-open');
  const my = ++token;
  main.innerHTML = '';
  main.append(el('div', { class: 'kicker' }, 'Archive · earlier version of the site'));
  if (!dir || !id || !(ARCHIVE[dir] || []).some(([i]) => i === id)) {
    main.append(el('h1', {}, 'Archived pages'));
    main.append(el('p', { class: 'lede' }, 'Pages from the first version of the site, kept for reference. They are no longer maintained; the current pages are in the tabs above.'));
    Object.entries(ARCHIVE).forEach(([d, list]) => {
      main.append(el('h2', {}, { topics: 'ML concepts', omics: 'ML in omics', frontier: 'Frontier methods' }[d] || d));
      const g = el('div', { class: 'hero-grid' });
      list.forEach(([i, t]) => g.append(el('a', { class: 'card hero-card', href: `#/archive/${d}/${i}` }, el('div', { class: 't' }, t))));
      main.append(g);
    });
    document.title = 'Archive · ML Visual Atlas';
    return;
  }
  let mod = null;
  try { mod = await import(`./archive/${dir}/${id}.js`); } catch (e) { console.error(e); }
  if (my !== token) return;
  const title = ARCHIVE[dir].find(([i]) => i === id)[1];
  document.title = `${title} (archived) · ML Visual Atlas`;
  main.append(el('div', { class: 'callout warn' }, el('span', { class: 'lbl' }, 'Archived page'), 'This is an earlier version of the site, kept for reference. ', el('a', { href: '#/archive' }, 'All archived pages')));
  if (!mod) { main.append(el('h1', {}, title)); return; }
  try { await mod.default(main, { tab: null, item: { id, title } }); } catch (e) { console.error(e); main.append(el('div', { class: 'callout bad' }, 'This page hit an error while rendering: ' + e.message)); }
  window.scrollTo(0, 0);
}

search.addEventListener('input', () => { const { tab, topicId } = parse(); drawNav(tab, topicId); });
window.addEventListener('hashchange', route);
route();
