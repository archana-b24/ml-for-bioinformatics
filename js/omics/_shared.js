// Shared helpers for Tab 2 (ML in omics) pages. Not a page (no registry id); imported by js/omics/<id>.js.
import { el, html } from '../lib/ui.js';

/**
 * The step card every omics page shows right after header(): where we are in the study, the biological
 * question, the data, the method, and the verdict "ML, statistics, or both?" with a one-line why.
 *   stepCard(main, { stage: 'Bulk · step 4 of 11', question, data, method, verdict: 'stats'|'ml'|'both', why })
 */
export function stepCard(main, { stage, question, data, method, verdict, why }) {
  const label = { ml: 'Machine learning', stats: 'Statistics', both: 'ML + statistics' }[verdict] || verdict;
  const kind = { ml: 'good', stats: '', both: 'warn' }[verdict] || '';
  const row = (k, v) => v ? el('div', { class: 'step-row' }, el('b', {}, k + ' '), html(v, 'span')) : null;
  const card = el('div', { class: 'callout ' + kind },
    el('span', { class: 'lbl' }, stage || 'Study step'),
    row('Question:', question), row('Data:', data), row('Method:', method),
    el('div', { class: 'step-row' }, el('b', {}, 'ML or statistics? '), el('span', { class: 'tag' }, label), why ? html(why, 'span') : null));
  main.append(card);
  return card;
}

/** Load a data/omics JSON with a friendly in-page error if the build script has not been run. */
export async function omicsData(main, name) {
  try {
    const r = await fetch(`data/omics/${name}`);
    if (!r.ok) throw new Error(r.status);
    return await r.json();
  } catch (e) {
    main.append(html(`<p class="prose">Data file <code>data/omics/${name}</code> is missing. Run the build script in <code>scripts/</code>.</p>`));
    return null;
  }
}
