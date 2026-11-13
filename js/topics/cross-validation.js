// Cross-validation: animated k-fold (plain / stratified / grouped), a nested-CV diagram, and a demo of
// how "tuning on the test folds" reports an optimistic score even on pure noise.
import { header, figure, slider, select, button, stats, section, steps, callout, math, seeAlso, legend, every, tip, untip, el } from '../lib/ui.js';
import { frame, cat, css, ink, ink2, muted } from '../lib/plot.js';
import { rng, mean, sd, range } from '../lib/data.js';
const d3 = window.d3;

const TRAIN = () => cat(0), TEST = () => cat(1), VAL = () => cat(2);
const P = 24, M = 3, K_NN = 3; // patients, samples per patient, k of the kNN classifier

// ---------- data: 24 patients x 3 samples, ~30% responders, label depends weakly on feature 1 ----------
function makeCohort(seed, nP) {
  const r = rng(seed);
  return range(nP).map((p) => {
    const c = [r.normal(0, 1), r.normal(0, 1)], label = c[0] + r.normal(0, 1) > 0.8 ? 1 : 0;
    return { id: p, c, label, eps: range(M).map(() => [r.normal(0, 1), r.normal(0, 1)]) };
  });
}
const toSamples = (pts, w = 0.15, m = M) => pts.flatMap((p) => range(m).map((s) => ({ pid: p.id, s, label: p.label, x: p.c[0] + w * p.eps[s][0], y: p.c[1] + w * p.eps[s][1] })));

function knnPredict(train, q, k = K_NN) {
  const bd = new Array(k).fill(Infinity), bl = new Array(k).fill(0);
  for (const t of train) {
    const d = (t.x - q.x) ** 2 + (t.y - q.y) ** 2;
    if (d >= bd[k - 1]) continue;
    let j = k - 1; while (j > 0 && bd[j - 1] > d) { bd[j] = bd[j - 1]; bl[j] = bl[j - 1]; j--; }
    bd[j] = d; bl[j] = t.label;
  }
  return d3.sum(bl) * 2 > k ? 1 : 0;
}

/** Assign each sample a fold 0..k-1. scheme: 'kfold' | 'strat' | 'group'. */
function assignFolds(samples, k, scheme, seed) {
  const r = rng(seed), fold = new Array(samples.length);
  if (scheme === 'kfold') r.shuffle(range(samples.length)).forEach((i, pos) => { fold[i] = pos % k; });
  else if (scheme === 'strat') { // deal each class round-robin so every fold gets its share of positives
    let pos = 0;
    [0, 1].forEach((c) => r.shuffle(range(samples.length).filter((i) => samples[i].label === c)).forEach((i) => { fold[i] = pos++ % k; }));
  } else { // grouped: whole patients go to one fold
    const pf = {}; r.shuffle([...new Set(samples.map((s) => s.pid))]).forEach((p, pos) => { pf[p] = pos % k; });
    samples.forEach((s, i) => { fold[i] = pf[s.pid]; });
  }
  return fold;
}
function cvScores(samples, fold, k) {
  return range(k).map((f) => {
    const tr = samples.filter((_, i) => fold[i] !== f), te = samples.filter((_, i) => fold[i] === f);
    return { acc: mean(te.map((q) => +(knnPredict(tr, q) === q.label))), n: te.length, pos: te.filter((q) => q.label).length };
  });
}

// ---------- optimism demo: pure-noise data, many candidate "settings" (random 3-gene subsets + nearest centroid) ----------
function noiseExperiment(nData = 10, nCfg = 100) {
  const n = 60, p = 200, kO = 5, kI = 4, sub = 3;
  const flat = [], nested = [];
  for (let dsi = 0; dsi < nData; dsi++) {
    const r = rng(300 + dsi);
    const X = range(n).map(() => Float64Array.from({ length: p }, () => r.normal()));
    const y = r.shuffle(range(n).map((i) => i % 2));                 // labels independent of X: true accuracy = 0.5
    const cfgs = range(nCfg).map(() => r.shuffle(range(p)).slice(0, sub));
    const ncScore = (cfg, trIdx, teIdx) => { // nearest-centroid accuracy using the chosen genes
      const cen = [0, 1].map((c) => { const m = trIdx.filter((i) => y[i] === c); return cfg.map((g) => mean(m.map((i) => X[i][g]))); });
      return mean(teIdx.map((i) => { const d = cen.map((ce) => d3.sum(cfg, (g, j) => (X[i][g] - ce[j]) ** 2)); return +((d[1] < d[0] ? 1 : 0) === y[i]); }));
    };
    const folds = (idx, k, seed) => { const rr = rng(seed), sh = rr.shuffle(idx); return range(k).map((f) => sh.filter((_, j) => j % k === f)); };
    const all = range(n), outer = folds(all, kO, 900 + dsi);
    // flat CV: score every config by 5-fold CV, then (wrongly) report the best of the first m
    const cvOf = cfgs.map((cfg) => mean(outer.map((te) => ncScore(cfg, all.filter((i) => !te.includes(i)), te))));
    flat.push(range(nCfg).map((m) => d3.max(cvOf.slice(0, m + 1))));
    // nested: inside each outer-training set, choose the config by inner CV; score that choice on the outer test fold
    const perOuter = outer.map((te, o) => {
      const tr = all.filter((i) => !te.includes(i)), inner = folds(tr, kI, 1900 + 10 * dsi + o);
      const inScore = cfgs.map((cfg) => mean(inner.map((ite) => ncScore(cfg, tr.filter((i) => !ite.includes(i)), ite))));
      const outScore = cfgs.map((cfg) => ncScore(cfg, tr, te));
      let best = 0; return range(nCfg).map((m) => { if (inScore[m] > inScore[best]) best = m; return outScore[best]; });
    });
    nested.push(range(nCfg).map((m) => mean(perOuter.map((po) => po[m]))));
  }
  return range(nCfg).map((m) => ({ m: m + 1, flat: mean(flat.map((f) => f[m])), nested: mean(nested.map((f) => f[m])) }));
}

export default async function render(main) {
  header(main, 'Cross-validation',
    'One train/test split gives one noisy score. <b>Cross-validation rotates the test role through every part of the data</b>, so every sample is tested exactly once and you get k scores whose spread tells you how much to trust the mean.');

  const patients = makeCohort(5, P), samples = toSamples(patients), N = samples.length;
  const freshS = toSamples(makeCohort(777, 600), 0.15, 1);
  const truth = mean(freshS.map((q) => +(knnPredict(samples, q) === q.label)));
  // display order: patients sorted by label, then id; each patient's samples adjacent
  const order = range(N).sort((a, b) => samples[a].label - samples[b].label || samples[a].pid - samples[b].pid || samples[a].s - samples[b].s);
  const col = new Array(N); order.forEach((i, c) => { col[i] = c; });

  // ================= Figure 1: animated k-fold =================
  const fig = figure(main, 'k-fold cross-validation, one fold at a time',
    `Top: each column is one sample (${P} patients × ${M} samples, ${d3.sum(patients, (p) => p.label)} of them responders), sorted so responders are on the right. Each row is one round: the <b>orange</b> block is held out and scored, the <b>blue</b> rest trains the model (3-nearest-neighbours). The two strips underneath mark responders and patients. Bottom left: the current round in feature space. Bottom right: each round's score; when all rounds are done, the band shows mean ± sd.`);
  let k = 5, scheme = 'kfold', seedF = 1, cur = -1, stop = null, folds, scores;
  select(fig.controls, { label: 'Scheme', value: scheme, options: [['kfold', 'k-fold (shuffled)'], ['strat', 'Stratified k-fold'], ['group', 'Grouped k-fold (by patient)']], onChange: (v) => { scheme = v; reset(); } });
  slider(fig.controls, { label: 'k', min: 2, max: 10, step: 1, value: k, onInput: (v) => { k = v; reset(); } });
  button(fig.controls, 'Run', () => run(), true);
  button(fig.controls, 'Step', () => { if (stop) { stop(); stop = null; } advance(); });
  button(fig.controls, 'Reshuffle', () => { seedF++; reset(); });
  const rd = stats(fig.root);
  const lg = el('div'); fig.body.append(lg);
  legend(lg, [['training samples', TRAIN()], ['test fold', TEST()]]);

  const cw = 12, lx = 74, top = 6;
  const mat = d3.select(fig.body).append('svg').attr('width', lx + N * cw + 70);
  const matG = mat.append('g'), stripG = mat.append('g');
  const row = el('div', { class: 'fig-row' }); fig.body.append(row);
  const left = el('div'), right = el('div'); row.append(left, right);
  const S = frame(left, { width: 460, height: 340, x: [-2.7, 2.7], y: [-2.7, 2.7], xLabel: 'feature 1', yLabel: 'feature 2' });
  S.layer('pts');
  const C = frame(right, { width: 460, height: 340, x: [0.5, 10.5], y: [0, 1], xLabel: 'fold (round)', yLabel: 'test accuracy in that fold' });
  C.layer('band'); C.layer('truth'); C.layer('dots');
  const tableDiv = el('div', { style: 'margin-top:10px' }); fig.body.append(tableDiv);

  function reset() {
    if (stop) { stop(); stop = null; }
    folds = assignFolds(samples, k, scheme, 40 + seedF);
    scores = cvScores(samples, folds, k);
    cur = -1; drawMatrix(); paint(); table();
  }
  function advance() { if (cur < k - 1) { cur++; paint(); } return cur < k - 1; }
  function run() { if (stop) stop(); cur = -1; paint(); stop = every(650, () => { if (!advance()) { stop(); stop = null; } }); }

  function drawMatrix() {
    const h = top + k * 16 + 46;
    mat.attr('viewBox', `0 0 ${lx + N * cw + 70} ${h}`);
    const cells = [];
    for (let f = 0; f < k; f++) samples.forEach((s, i) => cells.push({ f, i, test: folds[i] === f }));
    matG.selectAll('rect').data(cells).join('rect').attr('x', (d) => lx + col[d.i] * cw).attr('y', (d) => top + d.f * 16).attr('width', cw - 1).attr('height', 14)
      .attr('fill', (d) => (d.test ? TEST() : TRAIN()))
      .on('mousemove', (e, d) => { const s = samples[d.i]; tip(e, `round ${d.f + 1}: <b>${d.test ? 'test' : 'train'}</b><br>patient ${s.pid + 1}, sample ${s.s + 1}<br>${s.label ? 'responder' : 'non-responder'}`); })
      .on('mouseleave', untip);
    matG.selectAll('text.lab').data(range(k)).join('text').attr('class', 'lab').attr('x', lx - 8).attr('y', (f) => top + f * 16 + 11).attr('text-anchor', 'end').attr('font-size', 11).attr('fill', ink2()).text((f) => `round ${f + 1}`);
    matG.selectAll('text.pos').data(range(k)).join('text').attr('class', 'pos').attr('x', lx + N * cw + 6).attr('y', (f) => top + f * 16 + 11).attr('font-size', 11).attr('fill', ink2())
      .text((f) => `${scores[f].pos} resp.`);
    const ys = top + k * 16 + 6;
    stripG.selectAll('*').remove();
    stripG.append('text').attr('x', lx - 8).attr('y', ys + 9).attr('text-anchor', 'end').attr('font-size', 11).attr('fill', ink2()).text('responder');
    stripG.append('text').attr('x', lx - 8).attr('y', ys + 25).attr('text-anchor', 'end').attr('font-size', 11).attr('fill', ink2()).text('patient');
    samples.forEach((s, i) => {
      if (s.label) stripG.append('rect').attr('x', lx + col[i] * cw).attr('y', ys).attr('width', cw - 1).attr('height', 11).attr('fill', ink());
      stripG.append('rect').attr('x', lx + col[i] * cw).attr('y', ys + 16).attr('width', cw - 1).attr('height', 11)
        .attr('fill', d3.rank ? (s.pid % 2 ? css('--axis') : css('--surface-2')) : css('--axis'));
    });
    // patient boundaries
    order.forEach((i, c) => { if (c === 0 || samples[order[c - 1]].pid !== samples[i].pid) stripG.append('line').attr('x1', lx + c * cw - 0.5).attr('x2', lx + c * cw - 0.5).attr('y1', ys + 15).attr('y2', ys + 28).attr('stroke', ink2()).attr('stroke-width', 1); });
    C.x.domain([0.5, k + 0.5]); C.g.select('.axis.x').call(d3.axisBottom(C.x).ticks(k).tickFormat(d3.format('d')));
  }

  function paint() {
    matG.selectAll('rect').attr('opacity', (d) => (cur < 0 || d.f === cur ? 1 : 0.28));
    matG.selectAll('text.lab').attr('font-weight', (f) => (f === cur ? 700 : 400)).attr('fill', (f) => (f === cur ? ink() : ink2()));
    const sp = samples.map((s, i) => ({ s, test: cur >= 0 && folds[i] === cur })).sort((a, b) => a.test - b.test);
    S.layer('pts').selectAll('circle').data(sp).join('circle').attr('cx', (d) => S.x(d.s.x)).attr('cy', (d) => S.y(d.s.y))
      .attr('r', (d) => (d.test ? 5.5 : 4)).attr('fill', (d) => (cur < 0 ? muted() : d.test ? TEST() : TRAIN()))
      .attr('stroke', (d) => (d.s.label ? ink() : css('--surface'))).attr('stroke-width', (d) => (d.s.label ? 1.8 : 1.2))
      .on('mousemove', (e, d) => tip(e, `patient ${d.s.pid + 1}, sample ${d.s.s + 1}<br>${d.s.label ? 'responder (dark ring)' : 'non-responder'}`)).on('mouseleave', untip);
    const done = scores.slice(0, cur + 1).map((d, f) => ({ f, ...d }));
    C.layer('truth').selectAll('*').remove();
    C.layer('truth').append('line').attr('x1', 0).attr('x2', C.w).attr('y1', C.y(truth)).attr('y2', C.y(truth)).attr('stroke', muted()).attr('stroke-dasharray', '5 4').attr('stroke-width', 1.5);
    C.layer('truth').append('text').attr('x', C.w - 4).attr('y', C.y(truth) + 14).attr('text-anchor', 'end').attr('font-size', 11).attr('fill', muted()).text(`truth on new patients ${truth.toFixed(2)}`);
    const all = cur === k - 1, accs = scores.map((d) => d.acc), mu = mean(accs), s = sd(accs);
    C.layer('band').selectAll('*').remove();
    if (all) {
      C.layer('band').append('rect').attr('x', 0).attr('width', C.w).attr('y', C.y(Math.min(1, mu + s))).attr('height', C.y(Math.max(0, mu - s)) - C.y(Math.min(1, mu + s))).attr('fill', css('--accent-soft')).attr('opacity', 0.7);
      C.layer('band').append('line').attr('x1', 0).attr('x2', C.w).attr('y1', C.y(mu)).attr('y2', C.y(mu)).attr('stroke', css('--accent')).attr('stroke-width', 2);
      C.layer('band').append('text').attr('x', 4).attr('y', C.y(Math.min(1, mu + s)) - 5).attr('font-size', 11.5).attr('fill', ink2()).text(`mean ${mu.toFixed(2)} ± ${s.toFixed(2)} (sd)`);
    }
    C.layer('dots').selectAll('circle').data(done).join('circle').attr('cx', (d) => C.x(d.f + 1)).attr('cy', (d) => C.y(d.acc)).attr('r', 6)
      .attr('fill', TEST()).attr('stroke', css('--surface')).attr('stroke-width', 1.5)
      .on('mousemove', (e, d) => tip(e, `round ${d.f + 1}: accuracy ${d.acc.toFixed(3)}<br>${d.n} test samples, ${d.pos} responders`)).on('mouseleave', untip);
    const posR = d3.extent(scores, (d) => d.pos);
    rd.set({
      'round': cur < 0 ? `– / ${k}` : `${cur + 1} / ${k}`,
      'this round': cur < 0 ? '–' : scores[cur].acc.toFixed(2),
      'CV estimate (mean ± sd)': all ? `${mu.toFixed(2)} ± ${s.toFixed(2)}` : 'run all rounds',
      'responders per test fold': `${posR[0]}–${posR[1]}`,
      'truth on new patients': truth.toFixed(2),
    });
  }

  function table() {
    const rows = [['kfold', 'k-fold (shuffled)'], ['strat', 'Stratified k-fold'], ['group', 'Grouped k-fold (by patient)']].map(([sc, name]) => {
      // average over 20 reshuffles so the comparison is not hostage to one lucky split
      const reps = range(20).map((rep) => { const f = assignFolds(samples, k, sc, 1000 + rep), s = cvScores(samples, f, k); return { mu: mean(s.map((d) => d.acc)), sd: sd(s.map((d) => d.acc)), pos: d3.extent(s, (d) => d.pos) }; });
      return { sc, name, mu: mean(reps.map((r) => r.mu)), sd: mean(reps.map((r) => r.sd)), pmin: d3.min(reps, (r) => r.pos[0]), pmax: d3.max(reps, (r) => r.pos[1]) };
    });
    tableDiv.innerHTML = `<table class="simple"><tr><th>Scheme (k = ${k}, averaged over 20 reshuffles)</th><th>CV estimate</th><th>sd across folds</th><th>responders per test fold (range)</th></tr>` +
      rows.map((r) => `<tr${r.sc === scheme ? ' style="background:var(--surface-2)"' : ''}><td>${r.name}</td><td>${r.mu.toFixed(3)}</td><td>${r.sd.toFixed(3)}</td><td>${r.pmin}–${r.pmax}</td></tr>`).join('') +
      `<tr><td>Truth: same model on 600 new patients</td><td>${truth.toFixed(3)}</td><td></td><td></td></tr></table>`;
  }
  reset();

  // ================= Figure 2: nested CV =================
  const fig2 = figure(main, 'Nested cross-validation: tune inside, score outside',
    'When CV is also used to <b>choose</b> settings (k in kNN, λ in the lasso, which genes), the choice must be made without seeing the fold that will score it. The outer loop scores; an inner CV, run only on the outer-training data, does the choosing. Click an outer round to see its inner loop.');
  const NO = 5, NI = 4, W2 = 920;
  const ns = d3.select(fig2.body).append('svg').attr('viewBox', `0 0 ${W2} 250`).attr('width', W2);
  const lg2 = el('div'); fig2.body.prepend(lg2);
  legend(lg2, [['training', TRAIN()], ['inner validation (choose settings)', VAL()], ['outer test (score once)', TEST()]]);
  let outerSel = 0;
  const bx = 150, bw = 640;
  function drawNested() {
    ns.selectAll('*').remove();
    const txt = (x, y, t, o = {}) => ns.append('text').attr('x', x).attr('y', y).attr('font-size', o.size || 12).attr('text-anchor', o.anchor || 'start').attr('font-weight', o.bold ? 600 : 400).attr('fill', o.c || ink2()).text(t);
    txt(10, 20, 'Outer loop', { bold: true, c: ink() }); txt(10, 36, 'click a round');
    for (let o = 0; o < NO; o++) {
      const y = 8 + o * 17, g = ns.append('g').style('cursor', 'pointer').on('click', () => { outerSel = o; drawNested(); });
      for (let f = 0; f < NO; f++) g.append('rect').attr('x', bx + f * bw / NO).attr('y', y).attr('width', bw / NO - 2).attr('height', 14).attr('rx', 2)
        .attr('fill', f === o ? TEST() : TRAIN()).attr('opacity', o === outerSel ? 1 : 0.3);
      txt(bx + bw + 8, y + 11, `outer round ${o + 1}`, { c: o === outerSel ? ink() : muted(), bold: o === outerSel, size: 11 });
    }
    // inner loop: split the 4 outer-training blocks into NI inner folds
    const y0 = 118, trainW = bw * (NO - 1) / NO;
    ns.append('path').attr('d', `M${bx + 4},${8 + outerSel * 17 + 16} L${bx + 4},${y0 - 6}`).attr('stroke', ink2()).attr('stroke-dasharray', '3 3');
    txt(10, y0 + 12, 'Inner loop', { bold: true, c: ink() }); txt(10, y0 + 28, `on outer-train only`);
    for (let i = 0; i < NI; i++) {
      const y = y0 + i * 17;
      for (let f = 0; f < NI; f++) ns.append('rect').attr('x', bx + f * trainW / NI).attr('y', y).attr('width', trainW / NI - 2).attr('height', 14).attr('rx', 2).attr('fill', f === i ? VAL() : TRAIN());
      ns.append('rect').attr('x', bx + trainW + 6).attr('y', y).attr('width', bw / NO - 8).attr('height', 14).attr('rx', 2).attr('fill', 'none').attr('stroke', TEST()).attr('stroke-dasharray', '3 2');
      txt(bx + bw + 8, y + 11, `inner round ${i + 1}`, { size: 11 });
    }
    txt(bx + trainW + 6, y0 + NI * 17 + 12, 'outer test: not touched', { size: 11, c: muted() });
    const y1 = y0 + NI * 17 + 34;
    txt(bx, y1, `→ pick the setting with the best mean inner score, refit it on all outer-train data, score it ONCE on outer test fold ${outerSel + 1}.`, { c: ink() });
    txt(bx, y1 + 18, `Repeat for every outer round; the ${NO} outer scores are the honest estimate of the whole procedure.`, { c: ink() });
  }
  drawNested();

  // ================= Figure 3: optimism of tuning on the test folds =================
  const fig3 = figure(main, 'Tuning on the test folds is optimistic, even on pure noise',
    'Data: 60 samples × 200 random "genes" with labels assigned by coin flip, so <b>no model can truly beat 50%</b>. Each candidate setting is a different random 3-gene panel fed to a nearest-centroid classifier. <b>Flat CV</b>: score every candidate by 5-fold CV and report the best score. <b>Nested CV</b>: choose by inner CV, report the outer score. Curves average 10 such datasets.');
  const lg3 = el('div'); fig3.body.append(lg3);
  legend(lg3, [['flat CV: best score among candidates (what gets reported)', cat(1)], ['nested CV: honest estimate', cat(0)], ['truth (coin flip)', muted()]]);
  const res = noiseExperiment();
  let mSel = 30;
  slider(fig3.controls, { label: 'Candidate settings tried', min: 1, max: 100, step: 1, value: mSel, onInput: (v) => { mSel = v; paint3(); } });
  const rd3 = stats(fig3.root);
  const G = frame(fig3.body, { width: 920, height: 320, x: [1, 100], y: [0.35, 0.8], xLabel: 'number of candidate settings compared (log scale)', yLabel: 'reported accuracy', xScale: d3.scaleLog() });
  G.g.select('.axis.x').call(d3.axisBottom(G.x).tickValues([1, 2, 5, 10, 20, 50, 100]).tickFormat(d3.format('d')));
  G.layer('t').append('line').attr('x1', 0).attr('x2', G.w).attr('y1', G.y(0.5)).attr('y2', G.y(0.5)).attr('stroke', muted()).attr('stroke-dasharray', '5 4').attr('stroke-width', 1.5);
  [['flat', 1], ['nested', 0]].forEach(([key, c]) => G.layer('l').append('path').datum(res).attr('fill', 'none').attr('stroke', cat(c)).attr('stroke-width', 2.5)
    .attr('d', d3.line().x((d) => G.x(d.m)).y((d) => G.y(d[key]))));
  G.layer('mk');
  function paint3() {
    const d = res[mSel - 1];
    G.layer('mk').selectAll('*').remove();
    G.layer('mk').append('line').attr('x1', G.x(mSel)).attr('x2', G.x(mSel)).attr('y1', 0).attr('y2', G.h).attr('stroke', ink2()).attr('stroke-width', 1);
    [['flat', 1], ['nested', 0]].forEach(([key, c]) => G.layer('mk').append('circle').attr('cx', G.x(mSel)).attr('cy', G.y(d[key])).attr('r', 6).attr('fill', cat(c)).attr('stroke', css('--surface')).attr('stroke-width', 1.5)
      .on('mousemove', (e) => tip(e, `${mSel} candidates<br>${key === 'flat' ? 'flat CV best' : 'nested CV'}: ${d[key].toFixed(3)}`)).on('mouseleave', untip));
    rd3.set({ 'candidates tried': mSel, 'flat CV (reported)': d.flat.toFixed(3), 'nested CV': d.nested.toFixed(3), 'truth': '0.500', 'optimism of flat CV': '+' + (d.flat - 0.5).toFixed(3) });
  }
  paint3();

  section(main, 'What to notice', `<ul><li id="cv-notice"></li></ul>`);

  steps(main, 'How it works', [
    'Shuffle the samples and cut them into <b>k folds</b> of (nearly) equal size.',
    'For each fold in turn: train on the other k−1 folds, predict the held-out fold, record its score.',
    'Report the <b>mean</b> of the k scores, and their spread. Every sample is predicted exactly once, by a model that never saw it.',
    '<b>Stratified</b>: deal the samples of each class into folds separately, so every fold has the same class balance. Use it for classification, especially with a rare class.',
    '<b>Grouped</b>: assign whole groups (patients, donors, batches) to folds, so no group is on both sides. Use it whenever samples are related.',
    '<b>Nested</b>: if CV also chooses hyperparameters or features, put that whole choosing procedure inside an inner CV loop and score it on the outer folds.',
  ]);
  math(main, 'CV estimate = (1/k) Σ_f  score( model trained without fold f ,  fold f )\nsd across folds understates the true uncertainty: the k training sets overlap, so fold scores are not independent');

  callout(main, '', 'In biology', '<ul style="margin:4px 0"><li><b>Biomarker cohorts</b> of 50–200 patients are too small for a fixed test set, so CV is the norm. Put the whole pipeline inside each fold: gene filtering, DE-based feature selection, scaling and the lasso\'s λ (e.g. <code>cv.glmnet</code> run inside an outer loop, or scikit-learn <code>Pipeline</code> + <code>GridSearchCV</code> inside <code>cross_val_score</code>).</li><li><b>Multiple samples per patient</b> or cells per donor: use <code>GroupKFold</code> / <code>StratifiedGroupKFold</code> with the patient or donor id.</li><li><b>Multi-site studies:</b> leave-one-site-out CV estimates how a signature transfers to a new hospital or sequencing centre, which is usually the real question.</li><li>Rare responders (say 15%) with 5-fold CV can leave a fold with 1–2 responders. Stratify, or the AUC per fold becomes almost meaningless.</li></ul>');
  callout(main, 'warn', 'Pitfalls', '<ul><li><b>Feature selection before CV</b> (picking the top DE genes on all samples, then cross-validating a classifier on them) is the most common leak in omics. On pure noise it can produce AUCs above 0.9.</li><li>Reporting the best of many CV scores as the performance (the red curve above). Use nested CV, or a truly untouched test set.</li><li>The fold sd is not a confidence interval; it ignores the overlap between training sets. Repeated CV makes the mean more stable but does not fix this.</li><li>CV estimates the performance of the <i>procedure</i>, not of the single final model you fit on all the data, though the two are usually close.</li></ul>');
  seeAlso(main, [['concepts', 'train-test', 'Train, validation, test'], ['concepts', 'hyperparameter-tuning', 'Hyperparameter tuning'], ['concepts', 'metrics', 'Classification metrics'], ['omics', 'bulk-classifier', 'Response classifier & nested CV']]);
}
