// Train / validation / test: why we hold data out, and how samples from the same patient leak across a random split.
import { header, figure, slider, select, button, stats, section, steps, callout, math, seeAlso, legend, tip, untip, el } from '../lib/ui.js';
import { frame, cat, css, ink, ink2, muted } from '../lib/plot.js';
import { rng, mean, sd, range } from '../lib/data.js';
const d3 = window.d3;

const SPLIT = { train: 0, test: 1, val: 2 };          // categorical slots: train = 0, test = 1, validation = 2
const splitColor = (s) => cat(SPLIT[s]);
const P = 30, MAXM = 6, K = 3;                        // patients, max samples per patient, k of the kNN classifier

// One cohort: each patient has a centre in a 2-feature space and ONE label (responder or not).
// The label depends only weakly on the features (feature 1 + noise), as in most real biomarker problems.
function cohort(seed, nPatients) {
  const r = rng(seed);
  return range(nPatients).map((p) => {
    const c = [r.normal(0, 1), r.normal(0, 1)];
    const label = c[0] + r.normal(0, 1) > 0 ? 1 : 0;
    const eps = range(MAXM).map(() => [r.normal(0, 1), r.normal(0, 1)]); // per-sample noise, scaled later
    return { id: p, c, label, eps };
  });
}
const samplesOf = (pts, m, w) => pts.flatMap((p) => range(m).map((s) => ({ pid: p.id, s, label: p.label, x: p.c[0] + w * p.eps[s][0], y: p.c[1] + w * p.eps[s][1] })));

/** k-NN prediction for one query; returns {pred, nbrs}. */
function knn(train, q, k = K) {
  const bd = new Array(k).fill(Infinity), bt = new Array(k).fill(null); // running top-k by insertion (k is tiny)
  for (const t of train) {
    const d = (t.x - q.x) ** 2 + (t.y - q.y) ** 2;
    if (d >= bd[k - 1]) continue;
    let j = k - 1; while (j > 0 && bd[j - 1] > d) { bd[j] = bd[j - 1]; bt[j] = bt[j - 1]; j--; }
    bd[j] = d; bt[j] = t;
  }
  const nbrs = bt.filter(Boolean), votes = d3.sum(nbrs, (n) => n.label);
  return { pred: votes * 2 > nbrs.length ? 1 : 0, nbrs };
}
/** Split samples: 'random' picks test samples; 'group' picks test patients. Returns array of 'train'|'test'. */
function split(samples, mode, frac, seed) {
  const r = rng(seed);
  if (mode === 'random') {
    const idx = r.shuffle(range(samples.length)), nt = Math.round(frac * samples.length), test = new Set(idx.slice(0, nt));
    return samples.map((_, i) => (test.has(i) ? 'test' : 'train'));
  }
  const pids = r.shuffle([...new Set(samples.map((s) => s.pid))]), nt = Math.round(frac * pids.length), test = new Set(pids.slice(0, nt));
  return samples.map((s) => (test.has(s.pid) ? 'test' : 'train'));
}
function evaluate(samples, assign) {
  const train = samples.filter((_, i) => assign[i] === 'train'), test = samples.filter((_, i) => assign[i] === 'test');
  const res = test.map((q) => { const { pred, nbrs } = knn(train, q); return { q, pred, nbrs, ok: pred === q.label }; });
  return { res, acc: mean(res.map((d) => +d.ok)), sib: mean(res.map((d) => +d.nbrs.some((n) => n.pid === d.q.pid))) };
}

export default async function render(main) {
  header(main, 'Train, validation, test',
    'A model is only useful if it works on data it has never seen, so we <b>hide some data from it and score it there</b>. The held-out data must be truly new: if a near-copy of a test sample sits in the training set, the score measures memory, not learning.');

  const patients = cohort(21, P);

  // ================= Figure 1: where does each sample go? =================
  const fig = figure(main, 'Where does each sample go?',
    'Each square is one sample; each column is one patient with 4 samples (e.g. 4 biopsies or 4 sequencing runs). The split puts 60% of samples in <b>train</b> (fit the model), 20% in <b>validation</b> (choose settings, e.g. k or λ) and 20% in <b>test</b> (touched once, at the very end). Hover a square for details.');
  let mode1 = 'random', seed1 = 1;
  select(fig.controls, { label: 'Split', value: 'random', options: [['random', 'Random by sample'], ['group', 'Grouped by patient']], onChange: (v) => { mode1 = v; draw1(); } });
  button(fig.controls, 'Reshuffle', () => { seed1++; draw1(); });
  const r1 = stats(fig.root);
  const lg1 = el('div'); fig.body.append(lg1);
  legend(lg1, [['train', splitColor('train')], ['validation', splitColor('val')], ['test', splitColor('test')]]);
  const M1 = 4, cell = 26, gx = 70, gy = 14;
  const svg1 = d3.select(fig.body).append('svg').attr('viewBox', `0 0 ${gx + P * cell + 10} ${gy + M1 * cell + 34}`).attr('width', gx + P * cell + 10);
  range(M1).forEach((s) => svg1.append('text').attr('x', gx - 8).attr('y', gy + s * cell + cell / 2 + 4).attr('text-anchor', 'end').attr('font-size', 11.5).attr('fill', ink2()).text(`sample ${s + 1}`));
  svg1.append('text').attr('x', gx + P * cell / 2).attr('y', gy + M1 * cell + 24).attr('text-anchor', 'middle').attr('font-size', 12).attr('fill', ink2()).text(`patients 1 … ${P}  (one column each)`);
  const cells1 = svg1.append('g');
  function draw1() {
    const r = rng(100 + seed1), samples = samplesOf(patients, M1, 0.15);
    let a;
    if (mode1 === 'random') {
      const idx = r.shuffle(range(samples.length)), n = samples.length;
      a = new Array(n); idx.forEach((k, i) => { a[k] = i < 0.2 * n ? 'test' : i < 0.4 * n ? 'val' : 'train'; });
    } else {
      const pids = r.shuffle(range(P)), role = {}; pids.forEach((p, i) => { role[p] = i < 0.2 * P ? 'test' : i < 0.4 * P ? 'val' : 'train'; });
      a = samples.map((s) => role[s.pid]);
    }
    cells1.selectAll('rect').data(samples).join('rect').attr('x', (d) => gx + d.pid * cell).attr('y', (d) => gy + d.s * cell)
      .attr('width', cell - 3).attr('height', cell - 3).attr('rx', 3).attr('fill', (d, i) => splitColor(a[i]))
      .on('mousemove', (e, d) => { const i = samples.indexOf(d); tip(e, `patient ${d.pid + 1}, sample ${d.s + 1}<br>split: <b>${a[i] === 'val' ? 'validation' : a[i]}</b>`); })
      .on('mouseleave', untip);
    const roles = range(P).map((p) => new Set(a.filter((_, i) => samples[i].pid === p)));
    const testIdx = a.map((v, i) => i).filter((i) => a[i] === 'test');
    const leakyTest = testIdx.filter((i) => roles[samples[i].pid].has('train')).length;
    r1.set({
      'train / val / test samples': `${a.filter((v) => v === 'train').length} / ${a.filter((v) => v === 'val').length} / ${testIdx.length}`,
      'patients in more than one split': `${roles.filter((s) => s.size > 1).length} of ${P}`,
      'test samples whose patient is also in train': `${leakyTest} of ${testIdx.length}`,
    });
  }
  draw1();

  // ================= Figure 2: the leak, and what it does to the score =================
  const fig2 = figure(main, 'Same patient on both sides: the score you get vs the score you deserve',
    'Each dot is a sample; a patient\'s samples form a tight clump (shaded outline). The classifier is <b>3-nearest-neighbours</b>: a test sample takes the majority label of the 3 closest training samples. Lines join each test sample to those 3 neighbours: <b>solid</b> when the neighbour is from the <i>same patient</i>, dashed otherwise. A dark ring marks a wrong prediction. The bars below average 50 different splits; the dot is the split shown.');
  let m = 4, w = 0.15, seed2 = 1;
  slider(fig2.controls, { label: 'Samples per patient', min: 1, max: MAXM, step: 1, value: m, onInput: (v) => { m = v; draw2(); } });
  slider(fig2.controls, { label: 'Within-patient spread', min: 0.05, max: 1.5, step: 0.05, value: w, fmt: (v) => v.toFixed(2), onInput: (v) => { w = v; draw2(); } });
  button(fig2.controls, 'New split', () => { seed2++; draw2(); }, true);
  const r2 = stats(fig2.root);
  const lg2 = el('div'); fig2.body.append(lg2);
  legend(lg2, [['train sample', splitColor('train')], ['test sample', splitColor('test')]]);
  const row = el('div', { class: 'fig-row' }); fig2.body.append(row);
  const left = el('div'), right = el('div'); row.append(left, right);
  const panel = (div, title) => {
    div.append(el('div', { class: 'fig-title', style: 'font-weight:500;font-size:14px;color:var(--ink-2)' }, title));
    const F = frame(div, { width: 460, height: 400, x: [-2.7, 2.7], y: [-2.7, 2.7], xLabel: 'feature 1 (e.g. gene A, z-score)', yLabel: 'feature 2 (gene B)' });
    const id = 'ttclip' + (++clipN);
    F.svg.append('defs').append('clipPath').attr('id', id).append('rect').attr('x', -6).attr('y', -6).attr('width', F.w + 12).attr('height', F.h + 12);
    ['hull', 'links', 'pts'].forEach((k) => F.layer(k).attr('clip-path', `url(#${id})`)); return F;
  };
  let clipN = 0;
  const FA = panel(left, 'Random split by sample (leaky)'), FB = panel(right, 'Grouped split by patient (honest)');
  const barDiv = el('div'); fig2.body.append(barDiv);
  const B = frame(barDiv, { width: 920, height: 190, x: [0.3, 1], y: [0, 3], xLabel: 'test accuracy of the 3-NN classifier', grid: false, margin: { left: 210, top: 10, right: 20 } });
  B.g.select('.axis.y').remove();
  B.g.append('g').attr('class', 'gridline').call(d3.axisBottom(B.x).ticks(7).tickSize(B.h).tickFormat('')).lower();
  const ROWS = [['random', 'Random split (leaky)'], ['group', 'Grouped split'], ['truth', 'Truth: 600 new patients']];
  const yb = (i) => B.h * (i + 0.5) / 3;
  ROWS.forEach(([, t], i) => B.g.append('text').attr('x', -12).attr('y', yb(i) + 4).attr('text-anchor', 'end').attr('font-size', 13).attr('fill', ink()).text(t));
  B.layer('bars'); B.layer('err'); B.layer('dot'); B.layer('chance');
  B.layer('chance').append('line').attr('x1', B.x(0.5)).attr('x2', B.x(0.5)).attr('y1', 0).attr('y2', B.h).attr('stroke', muted()).attr('stroke-dasharray', '4 3');
  B.layer('chance').append('text').attr('x', B.x(0.5) + 4).attr('y', 10).attr('font-size', 11).attr('fill', muted()).text('coin flip');
  const fresh = cohort(999, 600);

  function paintPanel(F, samples, assign, ev) {
    const test = ev.res;
    const byP = d3.groups(samples, (s) => s.pid).filter(([, g]) => g.length > 2);
    F.layer('hull').selectAll('path').data(byP).join('path')
      .attr('d', ([, g]) => { const h = d3.polygonHull(g.map((s) => [F.x(s.x), F.y(s.y)])); return h ? 'M' + h.join('L') + 'Z' : null; })
      .attr('fill', css('--accent-soft')).attr('fill-opacity', 0.45).attr('stroke', muted()).attr('stroke-width', 1).attr('stroke-opacity', 0.6).attr('stroke-linejoin', 'round');
    const links = test.flatMap((d) => d.nbrs.map((n) => ({ a: d.q, b: n, sib: n.pid === d.q.pid })));
    F.layer('links').selectAll('line').data(links).join('line').attr('x1', (l) => F.x(l.a.x)).attr('y1', (l) => F.y(l.a.y)).attr('x2', (l) => F.x(l.b.x)).attr('y2', (l) => F.y(l.b.y))
      .attr('stroke', (l) => (l.sib ? ink() : muted())).attr('stroke-width', (l) => (l.sib ? 1.6 : 1)).attr('stroke-dasharray', (l) => (l.sib ? null : '3 3'));
    const wrong = new Set(test.filter((d) => !d.ok).map((d) => d.q));
    const order = samples.map((s, i) => ({ s, sp: assign[i] })).sort((a, b) => (a.sp === 'test') - (b.sp === 'test'));
    F.layer('pts').selectAll('circle').data(order).join('circle').attr('cx', (d) => F.x(d.s.x)).attr('cy', (d) => F.y(d.s.y))
      .attr('r', (d) => (d.sp === 'test' ? 5 : 4)).attr('fill', (d) => splitColor(d.sp))
      .attr('stroke', (d) => (wrong.has(d.s) ? ink() : css('--surface'))).attr('stroke-width', (d) => (wrong.has(d.s) ? 2.2 : 1.2))
      .on('mousemove', (e, d) => {
        const t = test.find((q) => q.q === d.s);
        tip(e, `patient ${d.s.pid + 1}, sample ${d.s.s + 1}<br>true label: ${d.s.label ? 'responder' : 'non-responder'}<br>split: <b>${d.sp}</b>` +
          (t ? `<br>predicted: ${t.pred ? 'responder' : 'non-responder'} (${t.ok ? 'correct' : 'wrong'})<br>neighbours from same patient: ${t.nbrs.filter((n) => n.pid === d.s.pid).length} of ${K}` : ''));
      })
      .on('mouseleave', untip);
  }

  function draw2() {
    const samples = samplesOf(patients, m, w);
    const aR = split(samples, 'random', 0.25, 500 + seed2), aG = split(samples, 'group', 0.25, 500 + seed2);
    const eR = evaluate(samples, aR), eG = evaluate(samples, aG);
    paintPanel(FA, samples, aR, eR); paintPanel(FB, samples, aG, eG);
    // repeated splits for stable bars
    const reps = range(50).map((k) => [evaluate(samples, split(samples, 'random', 0.25, 9000 + k)).acc, evaluate(samples, split(samples, 'group', 0.25, 9000 + k)).acc]);
    const freshS = samplesOf(fresh, 1, w);
    const truth = mean(freshS.map((q) => +(knn(samples, q).pred === q.label)));
    const rows = [
      { m: mean(reps.map((r) => r[0])), s: sd(reps.map((r) => r[0])), cur: eR.acc },
      { m: mean(reps.map((r) => r[1])), s: sd(reps.map((r) => r[1])), cur: eG.acc },
      { m: truth, s: 0, cur: null },
    ];
    const bh = B.h / 3 * 0.5;
    B.layer('bars').selectAll('rect').data(rows).join('rect').attr('x', B.x(0.3)).attr('y', (d, i) => yb(i) - bh / 2).attr('height', bh)
      .attr('width', (d) => Math.max(0, B.x(Math.max(0.3, d.m)) - B.x(0.3))).attr('rx', 3).attr('fill', (d, i) => (i < 2 ? css('--accent-soft') : css('--surface-2'))).attr('stroke', (d, i) => (i < 2 ? css('--accent') : css('--axis')))
      .on('mousemove', (e, d) => tip(e, `mean accuracy ${d.m.toFixed(3)}${d.s ? ` ± ${d.s.toFixed(3)} (sd over 50 splits)<br>the split drawn above: ${d.cur.toFixed(3)}` : ''}`)).on('mouseleave', untip);
    B.layer('err').selectAll('path').data(rows.filter((d) => d.s)).join('path')
      .attr('d', (d, i) => { const y = yb(i); return `M${B.x(d.m - d.s)},${y}H${B.x(Math.min(1, d.m + d.s))}M${B.x(d.m - d.s)},${y - 6}V${y + 6}M${B.x(Math.min(1, d.m + d.s))},${y - 6}V${y + 6}`; })
      .attr('stroke', ink()).attr('stroke-width', 1.5).attr('fill', 'none');
    B.layer('dot').selectAll('circle').data(rows.filter((d) => d.cur != null)).join('circle').attr('cx', (d) => B.x(Math.max(0.3, d.cur))).attr('cy', (d, i) => yb(i))
      .attr('r', 5).attr('fill', ink()).attr('stroke', css('--surface')).attr('stroke-width', 1.5);
    B.layer('dot').selectAll('text').data(rows).join('text').attr('x', (d) => B.x(Math.min(1, d.m + d.s)) + 8).attr('y', (d, i) => yb(i) + 4).attr('font-size', 12).attr('fill', ink2())
      .text((d) => d.m.toFixed(2));
    r2.set({
      'random split: mean accuracy (50 splits)': rows[0].m.toFixed(2),
      'grouped split: mean accuracy (50 splits)': rows[1].m.toFixed(2),
      'true accuracy on new patients': truth.toFixed(2),
      'test samples with a same-patient neighbour (random / grouped)': `${Math.round(100 * eR.sib)}% / ${Math.round(100 * eG.sib)}%`,
    });
  }
  draw2();

  section(main, 'What to notice', `<ul>
    <li>In the first figure, a <b>random split by sample</b> leaves 27 of the 30 patients straddling two or more piles, and 21 of the 24 test samples have a sibling in train. Switch to <b>grouped by patient</b>: whole columns move together, and both numbers drop to 0.</li>
    <li>In the second figure, at the default settings the leaky random split scores <b>0.81</b> on average, the grouped split <b>0.60</b>, and the truth (the same model scored on 600 brand-new patients) is <b>0.67</b>. In 73% of the random-split test samples, at least one of the 3 neighbours is the same patient's own sample: the model looks the answer up rather than predicting it.</li>
    <li>Set <b>samples per patient = 1</b>: there are no siblings to leak, and the random and grouped splits agree (0.67 each). The bug needs related samples.</li>
    <li>Raise the <b>within-patient spread</b> to 1.0: a patient's samples are no longer near-copies, the solid lines mostly disappear, and the leak shrinks (0.57 random vs 0.51 grouped, truth 0.56). Technical replicates leak far more than independent biopsies do.</li>
    <li>The grouped estimate is a little <i>below</i> the truth, because each split trains on only 22 of the 30 patients. Honest estimates tend to be slightly pessimistic; leaky ones can be wildly optimistic.</li>
    <li>Press <b>New split</b> a few times: the single-split accuracy (black dot) jumps around by ±0.1. A single small test set is a noisy measuring device.</li>
  </ul>`);

  steps(main, 'How it works', [
    '<b>Decide what "new" means for your use case first.</b> New patients? New hospitals? New sequencing batches? The split must mimic that: whatever unit you will meet fresh in deployment must never appear on both sides.',
    'Set the <b>test set</b> aside before any analysis (including feature selection and normalisation choices). Look at it once, at the end.',
    'Split the rest into <b>train</b> (fit parameters) and <b>validation</b> (compare models and settings). With small cohorts, replace the single validation split with <a href="#/concepts/cross-validation">cross-validation</a>.',
    'If samples are related (same patient, same donor, same mouse, same batch), split by that <b>group</b>, so all of a patient\'s samples land on one side.',
    'Report the test score, ideally with an interval. If you go back and change the model after looking at the test score, it is no longer a test set.',
  ]);
  math(main, 'test error  ≈  expected error on new data   only if   test samples are independent of the training samples\nrandom split of n samples from P patients  ⇒  P(a test sample has a sibling in train) ≈ 1 − (test fraction)^(m−1)');

  callout(main, '', 'In biology', '<ul style="margin:4px 0"><li><b>Multiple samples per patient</b> (several biopsies, pre/post treatment, technical replicates): split by patient. In scRNA-seq, thousands of cells come from each donor; a cell-level random split lets a classifier recognise the donor rather than the disease. Split by donor.</li><li><b>Batch and site:</b> if every sample from hospital A is in train and hospital B is held out, you are measuring transfer to a new site, which is usually what reviewers actually care about.</li><li><b>Related sequences:</b> for protein or DNA sequence models, near-identical sequences across the split inflate scores; cluster by sequence identity (e.g. MMseqs2/CD-HIT) and split by cluster.</li><li><b>Time:</b> for a prognostic model, train on earlier patients and test on later ones.</li></ul>');
  callout(main, 'warn', 'Pitfalls', '<ul><li><b>Preprocessing on all the data</b> before splitting (selecting the top-variance or most differentially expressed genes, scaling, batch correction) leaks test information into training. Fit every step on the training data only.</li><li><b>Peeking:</b> trying 50 models and reporting the best test score turns the test set into a validation set.</li><li><b>Duplicates</b> you did not know about: the same cell line under two names, re-sequenced libraries, public datasets that overlap. Check sample identity (e.g. SNP fingerprints) before splitting.</li><li>A small test set gives a noisy score: with 30 test samples, one prediction moves accuracy by 3 points.</li></ul>');
  seeAlso(main, [['concepts', 'cross-validation', 'Cross-validation'], ['concepts', 'bias-variance', 'Under- vs overfitting'], ['omics', 'bulk-classifier', 'Response classifier & nested CV'], ['omics', 'sc-pseudobulk', 'Pseudobulk DE']]);
}
