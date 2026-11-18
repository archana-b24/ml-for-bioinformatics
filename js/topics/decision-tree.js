// Decision trees, lesson format.
// Arc: what → why → where → how it works (6 steps: one question → impurity and split scoring → growing the tree and reading rules
//      → interactions → depth, overfitting, pruning and instability → tree lab with try/see/means table) → pitfalls → recap.
// Every number in the prose is computed from data/study/cohort.json at render time.
import { header, figure, slider, select, button, checkbox, stats, legend, seeAlso, tip, untip, el, html } from '../lib/ui.js';
import { frame, cat, css, ink, ink2, muted } from '../lib/plot.js';
import { rng, sigmoid, mean } from '../lib/data.js';
import { fitLogistic, predictProb, auc, roc, standardize, impurity, scanSplits, fitTree, treeLeaf, treeLeaves, treeDepth, pruningPath } from '../lib/models.js';
import { loadCohort, FEATURES, studyCard } from '../lib/study.js';
import { lessonNav, chapter, prose, rung, worked, quiz, cards, table, code, recap } from '../lib/lesson.js';
const d3 = window.d3;

const f2 = (v) => (Math.abs(v) < 0.005 ? '0.00' : v.toFixed(2));
const f1 = (v) => v.toFixed(1);
const f3 = (v) => v.toFixed(3);
const pc = (v) => Math.round(100 * v) + '%';
const sum = (a) => a.reduce((s, v) => s + v, 0);
const mix = (p) => d3.interpolateRgb(cat(0), cat(1))(p);

const TF = ['ifng', 'log2tmb', 'caf', 'pdl1', 'cd8', 'prolif', 'age', 'smoker', 'centreB'];
const SHORT = { ifng: 'IFN-γ', log2tmb: 'TMB', caf: 'CAF', pdl1: 'PD-L1', cd8: 'CD8', prolif: 'Proliferation', age: 'Age', smoker: 'Ever smoker', centreB: 'Centre B' };
const BIN = new Set(['smoker', 'centreB']);
const fval = (k, v) => (k === 'log2tmb' ? (2 ** v).toFixed(1) : k === 'pdl1' || k === 'age' ? v.toFixed(0) : v.toFixed(2));
const cond = (keys, f, thr, yes) => { const k = keys[f]; return BIN.has(k) ? `${SHORT[k]}: ${yes ? 'yes' : 'no'}` : `${SHORT[k]} ${yes ? '≥' : '<'} ${fval(k, thr)}${k === 'log2tmb' ? ' mut/Mb' : k === 'pdl1' ? '%' : ''}`; };
const rowsK = (pts, keys) => pts.map((p) => keys.map((k) => FEATURES[k].get(p)));
const cut0 = (nd, o) => !nd.left || (o.maxDepth !== undefined && nd.depth >= o.maxDepth) || (o.cut && o.cut.has(nd.id));

/** Rules of the leaves: [{node, conds:[…], text}] in left-to-right order. */
function leafRules(root, keys, o = {}) {
  const out = [];
  (function walk(nd, cs) {
    if (cut0(nd, o)) { out.push({ node: nd, text: cs.length ? cs.join(' <b>and</b> ') : 'all patients' }); return; }
    walk(nd.right, [...cs, cond(keys, nd.f, nd.thr, true)]); walk(nd.left, [...cs, cond(keys, nd.f, nd.thr, false)]);
  })(root, []);
  return out;
}
const pathIds = (root, row, o = {}) => { const s = new Set(); let nd = root; s.add(nd.id); while (!cut0(nd, o)) { nd = row[nd.f] < nd.thr ? nd.left : nd.right; s.add(nd.id); } return s; };

/** Tree diagram. mode 'boxes' (readable, few leaves) or 'dots' (compact, whole tree with removed branches faded). Returns draw(state). */
function drawTree(host, keys0, { width = 820, height = 320, compact = false } = {}) {
  const svg = d3.select(host).append('svg').attr('viewBox', `0 0 ${width} ${height}`).attr('width', width).attr('role', 'img').attr('aria-label', 'decision tree diagram');
  const gl = svg.append('g'), gn = svg.append('g');
  return function draw(root, st = {}) {
    const o = { cut: st.cut, maxDepth: st.maxDepth }, path = st.path, keys = st.keys || keys0;
    const kids = compact ? (nd) => (nd.left ? [nd.left, nd.right] : null) : (nd) => (cut0(nd, o) ? null : [nd.left, nd.right]);
    const H = d3.hierarchy(root, kids), nL = H.leaves().length, dep = Math.max(1, d3.max(H.descendants(), (d) => d.depth));
    const bw = compact ? 0 : Math.max(76, Math.min(128, (width - 20) / nL - 8)), bh = 44, padX = compact ? 14 : bw / 2 + 6, top = compact ? 14 : bh / 2 + 6;
    if (!compact) svg.attr('viewBox', `0 0 ${width} ${Math.round(2 * top + dep * 76)}`);
    const hh = compact ? height : 2 * top + dep * 76;
    d3.tree().size([width - 2 * padX, hh - 2 * top]).separation(() => 1)(H);
    H.each((d) => { d.px = d.x + padX; d.py = d.y + top; d.dead = false; if (compact) { let a = d.parent; while (a) { if (cut0(a.data, o)) { d.dead = true; break; } a = a.parent; } } });
    gl.selectAll('path').data(H.links(), (l) => l.target.data.id).join('path').attr('fill', 'none').attr('stroke', muted()).attr('stroke-width', 1.3).attr('stroke-opacity', (l) => (l.target.dead ? 0.15 : 1))
      .attr('d', (l) => `M${l.source.px},${l.source.py} C${l.source.px},${(l.source.py + l.target.py) / 2} ${l.target.px},${(l.source.py + l.target.py) / 2} ${l.target.px},${l.target.py}`);
    gl.selectAll('text').data(compact ? [] : H.links(), (l) => l.target.data.id).join('text').attr('font-size', 11).attr('fill', muted()).attr('text-anchor', 'middle')
      .attr('x', (l) => (l.source.px + l.target.px) / 2 + (l.target.px > l.source.px ? 10 : -10)).attr('y', (l) => (l.source.py + l.target.py) / 2 + (bh / 2 - 6) * 0.4 + 6).text((l) => (l.target.data === l.source.data.right ? 'yes' : 'no'));
    const g = gn.selectAll('g.node').data(H.descendants(), (d) => d.data.id).join((en) => { const q = en.append('g').attr('class', 'node'); q.append(compact ? 'circle' : 'rect'); q.append('text').attr('class', 't1'); q.append('text').attr('class', 't2'); q.append('rect').attr('class', 'b0'); q.append('rect').attr('class', 'b1'); return q; });
    g.attr('transform', (d) => `translate(${d.px},${d.py})`).style('opacity', (d) => (d.dead ? 0.15 : 1))
      .on('mousemove', (e, d) => { const nd = d.data, leaf = cut0(nd, o); tip(e, `${leaf ? 'leaf' : 'split: ' + cond(keys, nd.f, nd.thr, true)}<br>n = ${nd.n} (${nd.c[1]} responded, ${nd.c[0]} did not)<br>impurity (Gini) ${f3(impurity(nd.p))}${leaf ? '' : `<br>impurity reduction ${f3(nd.gain)}`}`); }).on('mouseleave', untip);
    const onPath = (d) => (path ? path.has(d.data.id) : false);
    if (compact) {
      g.select('circle').attr('r', (d) => 2.6 + 0.42 * Math.sqrt(d.data.n)).attr('fill', (d) => mix(d.data.p)).attr('stroke', (d) => (cut0(d.data, o) && !d.dead ? ink() : css('--surface'))).attr('stroke-width', (d) => (cut0(d.data, o) && !d.dead ? 2 : 1));
      g.select('.t1').text(''); g.select('.t2').text(''); g.select('.b0').attr('width', 0); g.select('.b1').attr('width', 0);
      return H;
    }
    g.select('rect').attr('x', -bw / 2).attr('y', -bh / 2).attr('width', bw).attr('height', bh).attr('rx', 6).attr('fill', css('--surface')).attr('stroke', (d) => (onPath(d) ? ink() : css('--axis'))).attr('stroke-width', (d) => (onPath(d) ? 2.6 : 1));
    g.select('.t1').attr('text-anchor', 'middle').attr('y', -6).attr('font-size', 11.5).attr('font-weight', 600).attr('fill', ink()).text((d) => (cut0(d.data, o) ? `leaf: n = ${d.data.n}` : cond(keys, d.data.f, d.data.thr, true).replace(' mut/Mb', '')));
    g.select('.t2').attr('text-anchor', 'middle').attr('y', 8).attr('font-size', 11).attr('fill', ink2()).text((d) => (cut0(d.data, o) ? `${pc(d.data.p)} resp.` : `n = ${d.data.n} · ${pc(d.data.p)}`));
    g.select('.b0').attr('x', -bw / 2 + 5).attr('y', bh / 2 - 9).attr('height', 5).attr('width', (d) => (bw - 10) * d.data.c[0] / d.data.n).attr('fill', cat(0));
    g.select('.b1').attr('x', (d) => -bw / 2 + 5 + (bw - 10) * d.data.c[0] / d.data.n).attr('y', bh / 2 - 9).attr('height', 5).attr('width', (d) => (bw - 10) * d.data.c[1] / d.data.n).attr('fill', cat(1));
    return H;
  };
}

export default async function render(main) {
  const C = await loadCohort();
  const D = C.discovery, V = C.validation;
  const y = D.map((p) => p.response), yv = V.map((p) => p.response);
  const n = D.length, nR = sum(y), base = nR / n;
  const XD = rowsK(D, TF), XV = rowsK(V, TF);
  const acc = (t, X, yy, o) => mean(X.map((r, i) => ((treeLeaf(t, r, o).p >= 0.5 ? 1 : 0) === yy[i] ? 1 : 0)));
  const sc = (t, X, o) => X.map((r) => treeLeaf(t, r, o).p);
  const T3 = fitTree(XD, y, { maxDepth: 3, minLeaf: 5 });
  /** Fit on a chosen feature set; returns tree, accuracies, validation AUC and a logistic baseline on the same features. */
  const evalTree = (keys, { depth = 3, minLeaf = 5, crit = 'gini' } = {}) => {
    const X = rowsK(D, keys), Xv = rowsK(V, keys), t = fitTree(X, y, { maxDepth: depth, minLeaf, crit });
    const st = standardize(X), lf = fitLogistic(st.Z, y, { l2: 0.01 });
    const lp = Xv.map((r) => predictProb(st.apply(r), lf.b0, lf.w));
    return { t, X, Xv, keys, leaves: treeLeaves(t).length, tr: acc(t, X, y), va: acc(t, Xv, yv), aucV: auc(sc(t, Xv), yv), lpV: lp, aucL: auc(lp, yv), depth: treeDepth(t) };
  };

  header(main, 'Decision trees',
    `A model that classifies a sample by asking a short series of <b>yes/no questions</b> about its measurements, such as "is the IFN-γ score at least ${fval('ifng', T3.thr)}?". Each question is chosen to separate the outcomes as cleanly as possible. The result can be read as a flowchart or as a list of if-then rules.`);
  lessonNav(main);
  studyCard(main, C, '<b>Question for this page:</b> can a short series of yes/no questions on the biopsy measurements sort patients into groups with clearly different response rates, and how far can that be trusted?');

  // =====================================================================================
  // 1. WHAT
  // =====================================================================================
  let ch = chapter(main, 'what', 'What is a decision tree?', 'In one line: <b>a flowchart of threshold questions, learned from labelled examples</b>.');
  prose(ch, `<p>A decision tree is a <b>classifier</b> (it also exists in a regression form). Starting at the top, a sample answers one question about one measurement and follows the matching branch to the next question, until it reaches an end point. That end point predicts an outcome. The tree is learned from data: at each step the algorithm selects the measurement and cut-off that best separate responders from non-responders among the patients that reached that step.</p>
    <p>The diagram shows a tree of at most three levels, fitted to the ${n} discovery patients, and the route taken by one patient. The bar under each box shows the mix of non-responders (blue) and responders (orange).</p>`);
  const exP = D.find((p, i) => p.response && treeDepth(T3) >= 3 && (() => { const l = treeLeaf(T3, XD[i]); return l.depth === 3 && l.p >= 0.5; })());
  const exI = D.indexOf(exP), exPath = pathIds(T3, XD[exI]);
  const wrapT = el('div', { class: 'figure' }); ch.append(wrapT);
  wrapT.append(el('div', { class: 'fig-title' }, `One patient (${exP.id}) following the tree`));
  drawTree(wrapT, TF, { height: 300 })(T3, { path: exPath });
  wrapT.append(html('Left branch = answer "no", right branch = "yes". The outlined boxes are the route of this patient. Hover a box for its counts and impurity.', 'div', 'fig-caption'));
  const steps = []; { let nd = T3; while (nd.left && !cut0(nd, {})) { const yes = XD[exI][nd.f] >= nd.thr; steps.push(`${SHORT[TF[nd.f]]} = ${fval(TF[nd.f], XD[exI][nd.f])} → ${cond(TF, nd.f, nd.thr, yes)}: <b>${yes ? 'yes' : 'no'}</b>`); nd = yes ? nd.right : nd.left; } }
  const exLeaf = treeLeaf(T3, XD[exI]);
  prose(ch, `<p>The route of ${exP.id}: ${steps.join('; then ')}. The end point contains ${exLeaf.n} discovery patients, of whom ${pc(exLeaf.p)} responded, so the tree predicts a responder. ${exP.id} did respond. Key terms:</p>`);
  table(ch, [
    { t: 'Node', m: 'One question (or one end point) in the tree, together with the group of patients that reaches it', s: `root node: all ${n} patients, ${pc(base)} responded` },
    { t: 'Split (test)', m: 'A question of the form "feature ≥ threshold?"', s: `${cond(TF, T3.f, T3.thr, true)}` },
    { t: 'Threshold', m: 'The cut-off in a split, chosen from the observed values', s: `${fval(TF[T3.f], T3.thr)} for the root split` },
    { t: 'Leaf', m: 'An end point: no further question. It predicts the majority class and gives the share of the class in that group', s: `${treeLeaves(T3).length} leaves at depth ≤ 3` },
    { t: 'Depth', m: 'Number of questions on the longest route from the root to a leaf', s: `${treeDepth(T3)} here` },
    { t: 'Impurity', m: 'How mixed the outcomes in a node are: 0 when all patients have the same outcome (step 2)', s: `Gini of the root = ${f3(impurity(base))}` },
    { t: 'Rule', m: 'The conjunction of the answers along one route, read as "if … then …"', s: 'listed in step 3' },
  ], [['t', 'Term'], ['m', 'Meaning'], ['s', 'In our study']]);

  // =====================================================================================
  // 2. WHY
  // =====================================================================================
  ch = chapter(main, 'why', 'Why do we need it?', 'What a model for a clinical outcome can be asked to provide, and where the alternatives fall short.');
  prose(ch, `<p>Beyond predictive accuracy, a model is often required to:</p><ol>
    <li><b>Be readable.</b> A clinician or reviewer can check a short list of if-then rules directly, without software.</li>
    <li><b>Handle several marker types without preparation.</b> Thresholds work on any numeric or yes/no measurement, on any scale, without standardisation or log transformation.</li>
    <li><b>Allow the effect of one marker to depend on another</b> (an interaction) without specifying the interaction in advance.</li></ol>`);
  const ev1 = evalTree(['ifng'], { depth: 1 }), ev3 = evalTree(['ifng', 'log2tmb', 'caf'], { depth: 3 });
  const rule = D.map((p) => (p.ifng >= 6 ? 1 : 0)), ruleSens = sum(rule.map((r, i) => r && y[i])) / nR, ruleSpec = sum(rule.map((r, i) => !r && !y[i])) / (n - nR);
  worked(ch, 'Compared with one fixed cut-off and with logistic regression',
    `<p>A single hand-picked cut-off ("IFN-γ ≥ 6") calls ${sum(rule)} patients responders, with sensitivity <b>${pc(ruleSens)}</b> and specificity <b>${pc(ruleSpec)}</b>. A tree chooses the cut-off itself (a one-question tree on IFN-γ places it at ${fval('ifng', ev1.t.thr)}) and then keeps asking questions inside each group. A three-level tree on IFN-γ, TMB and CAF reaches a validation AUC of <b>${f3(ev3.aucV)}</b>, against <b>${f3(ev3.aucL)}</b> for logistic regression on the same three measurements. The tree's output takes only ${ev3.leaves} distinct values, one per leaf, whereas logistic regression returns a different probability for every patient.</p>`);
  table(ch, [
    { a: 'Single cut-off on one marker', r: 'yes (one rule)', i: 'no', s: 'n/a', d: 'tiny' },
    { a: '<b>Decision tree</b>', r: '<b>yes (a few rules)</b>', i: '<b>yes, found automatically</b>', s: '<b>piecewise constant</b>', d: '<b>small to moderate; unstable</b>' },
    { a: 'Logistic regression', r: 'weights, not rules', i: 'only if written in', s: 'smooth', d: 'small to moderate' },
    { a: 'Random forest / boosting', r: 'no (hundreds of trees)', i: 'yes', s: 'finer steps', d: 'moderate to large' },
  ], [['a', 'Approach'], ['r', 'Readable as rules?'], ['i', 'Interactions?'], ['s', 'Shape of the risk surface'], ['d', 'Data needed']]);

  // =====================================================================================
  // 3. WHERE
  // =====================================================================================
  ch = chapter(main, 'where', 'Where is it used?', 'Settings where a transparent sequence of threshold decisions is useful, directly or as a building block.');
  cards(ch, [
    { tag: 'Clinical decision rules', title: 'Bedside and triage flowcharts', body: 'Many published clinical rules are tree-shaped: a few yes/no questions on age, a laboratory value or a symptom lead to a risk group. Trees fitted with CART (R <code>rpart</code>, scikit-learn) are a standard way to derive such rules from cohort data.' },
    { tag: 'Flow and mass cytometry', title: 'Gating', body: 'Manual gating is a tree: a sequence of thresholds on marker intensities that isolates a cell population. Tree learners can reproduce or suggest gating strategies from labelled cells.' },
    { tag: 'Patient stratification', title: 'Subgroups with different outcomes', body: 'Survival trees and conditional-inference trees (R <code>ctree</code>) search for subgroups defined by a few markers with different response or survival, as in the groups in this page.' },
    { tag: 'Variant and read QC', title: 'Filtering rules', body: 'Hard filters on quality, depth and strand-bias metrics are hand-written trees. Learned filters for variant calls and for low-quality reads can use tree models, with the splits available for audit.' },
    { tag: 'Ensembles', title: 'Random forests and boosting', body: 'Random forests, XGBoost and LightGBM combine many trees and are among the strongest methods for tabular data such as expression panels or methylation sites. The single tree on this page is their building block.' },
    { tag: 'Interpretation', title: 'Surrogate models', body: 'A small tree fitted to the predictions of a complex model gives an approximate, readable summary of what that model does. The approximation should be checked against the model it explains.' },
    { tag: 'Biomarker cut-offs', title: 'Where to place a threshold', body: 'A one-split tree is a method for choosing a cut-off, for example for PD-L1 staining or a gene-signature score. The cut-off should be validated on independent patients, since it is selected to fit the training data.' },
    { tag: 'Outside biology', title: 'Credit, fraud, churn', body: 'Scorecards and rule engines in lending and fraud detection have long been derived from trees because the decisions have to be explained.' },
  ]);
  prose(ch, '<p><b>When not to use a single tree:</b> when the signal is smooth and additive across several markers (a weighted sum is then more efficient and more stable, as steps 4 and 5 show); when the cohort is small and stable estimates are needed; when probabilities must be finely graded (a tree returns one value per leaf); or when the best accuracy matters more than readability (use an ensemble of trees, or boosting).</p>');

  // =====================================================================================
  // 4. HOW
  // =====================================================================================
  ch = chapter(main, 'how', 'How it works', 'Six steps: one question, how a question is scored, growing a tree and reading its rules, interactions, depth, pruning and instability, and a tree lab.');

  // ---- Step 1: one question
  const scan1 = scanSplits(XD, y, XD.map((_, i) => i), 0, { minLeaf: 5 });
  const best1 = scan1.reduce((a, b) => (b.w < a.w ? b : a));
  let rb = rung(ch, 1, 'One yes/no question',
    `<p>Take one measurement, the IFN-γ score, and one cut-off <i>t</i>. The question "IFN-γ ≥ <i>t</i>?" divides the ${n} patients into two groups. A good question gives groups whose response rates differ strongly. The overall rate is ${pc(base)}; the figure shows how many patients in each IFN-γ bin responded (above the axis) and did not (below).</p>`);
  const f1g = figure(rb, 'Moving the cut-off', 'Bars: number of patients per IFN-γ bin, responders above the axis (orange) and non-responders below (blue). The line is the cut-off; the shaded side is the "no" group. <b>Gini impurity</b> of a group = 2 × p × (1 − p), where p is the share of responders in it: 0 for a pure group, 0.5 for an even mix.');
  let t1 = Math.round(best1.thr * 20) / 20;
  const sl1 = slider(f1g.controls, { label: 'cut-off t', min: 3, max: 9.5, step: 0.05, value: t1, fmt: f2, onInput: (v) => { t1 = v; paint1(); } });
  button(f1g.controls, 'Jump to the best cut-off', () => { t1 = Math.round(best1.thr * 20) / 20; sl1.set(t1); paint1(); }, true);
  const s1 = stats(f1g.root);
  const bins1 = d3.range(32).map((i) => ({ lo: 2.2 + i * 0.25, r: 0, nr: 0 }));
  D.forEach((p) => { const b = bins1[Math.min(31, Math.max(0, Math.floor((p.ifng - 2.2) / 0.25)))]; if (p.response) b.r++; else b.nr++; });
  const mc = d3.max(bins1, (b) => Math.max(b.r, b.nr)) + 1;
  const F1 = frame(f1g.body, { width: 820, height: 300, x: [2.2, 10.2], y: [-mc, mc], xLabel: 'IFN-γ score (log2 expression)', yLabel: 'patients per bin' });
  F1.g.select('.axis.y').call(d3.axisLeft(F1.y).ticks(6).tickFormat((v) => Math.abs(v)));
  const shade1 = F1.layer('shade').append('rect').attr('y', 0).attr('height', F1.h).attr('fill', css('--surface-2'));
  F1.layer('bars').selectAll('rect').data(bins1.flatMap((b) => [{ b, up: true }, { b, up: false }])).join('rect').attr('x', (d) => F1.x(d.b.lo) + 1).attr('width', F1.x(2.45) - F1.x(2.2) - 2)
    .attr('y', (d) => (d.up ? F1.y(d.b.r) : F1.y(0))).attr('height', (d) => Math.abs(F1.y(d.up ? d.b.r : -d.b.nr) - F1.y(0))).attr('fill', (d) => cat(d.up ? 1 : 0)).attr('fill-opacity', 0.85)
    .on('mousemove', (e, d) => tip(e, `IFN-γ ${f2(d.b.lo)}–${f2(d.b.lo + 0.25)}<br>${d.b.r} responded, ${d.b.nr} did not`)).on('mouseleave', untip);
  const L1 = F1.layer('line'), ann1 = F1.layer('ann');
  legend(f1g.body, [['responded', cat(1)], ['no response', cat(0)], ['cut-off t', ink()]]);
  function paint1() {
    const Lg = D.filter((p) => p.ifng < t1), Rg = D.filter((p) => p.ifng >= t1), pl = Lg.length ? mean(Lg.map((p) => p.response)) : 0, pr = Rg.length ? mean(Rg.map((p) => p.response)) : 0;
    const gl = impurity(pl), gr = impurity(pr), gw = (Lg.length * gl + Rg.length * gr) / n;
    shade1.attr('x', 0).attr('width', Math.max(0, F1.x(t1)));
    L1.selectAll('line').data([0]).join('line').attr('x1', F1.x(t1)).attr('x2', F1.x(t1)).attr('y1', 0).attr('y2', F1.h).attr('stroke', ink()).attr('stroke-width', 2.4);
    ann1.selectAll('text').data([{ x: Math.max(8, F1.x(t1) - 8), a: 'end', s: `"no": ${Lg.length} patients, ${pc(pl)} responded` }, { x: F1.x(t1) + 8, a: 'start', s: `"yes": ${Rg.length} patients, ${pc(pr)} responded` }]).join('text')
      .attr('x', (d) => d.x).attr('y', 14).attr('text-anchor', (d) => d.a).attr('font-size', 12).attr('fill', ink()).text((d) => d.s);
    s1.set({ 'responders in "no" group': pc(pl), 'responders in "yes" group': pc(pr), 'Gini "no" / "yes"': `${f3(gl)} / ${f3(gr)}`, 'weighted Gini of the two groups': f3(gw), 'Gini before the split': f3(impurity(base)), 'impurity reduction': f3(impurity(base) - gw) });
  }
  paint1();
  const lo1 = D.filter((p) => p.ifng < best1.thr), hi1 = D.filter((p) => p.ifng >= best1.thr);
  prose(rb, `<p>At the best cut-off (${f2(best1.thr)}) the "no" group has ${lo1.length} patients, ${pc(best1.pl)} of whom responded, and the "yes" group has ${hi1.length}, ${pc(best1.pr)} of whom responded. Moving the cut-off towards either end leaves one very large mixed group, and the weighted impurity approaches the starting value of ${f3(impurity(base))}.</p>`);

  // ---- Step 2: impurity and split scoring
  rb = rung(ch, 2, 'Scoring a question: impurity and the best cut-off',
    `<p>To compare questions, each group gets an <b>impurity</b> value, and a question is scored by the <b>size-weighted average impurity of the two groups it creates</b>. The split with the lowest score (largest reduction compared with the parent) is chosen. Two standard impurity measures are Gini, 2p(1 − p), and entropy, −p log₂ p − (1 − p) log₂(1 − p). Both are 0 for a pure group and largest for a 50:50 mix. The misclassification error, min(p, 1 − p), is also shown for comparison.</p>
     <div class="calc">weighted impurity = (n_no × impurity_no + n_yes × impurity_yes) / n
impurity reduction = impurity(parent) − weighted impurity</div>`);
  const f2g = figure(rb, 'Impurity curve and the search over cut-offs', 'Left: the impurity curve. The dashed chord joins the two children; the weighted average of the children lies on the chord directly below the parent, and the vertical gap is the impurity reduction. Right: weighted impurity for every possible cut-off of the chosen feature. Hover or click to select a cut-off; the cross marks the best one.');
  let fk = 0, crit2 = 'gini', thr2 = null;
  select(f2g.controls, { label: 'Feature', value: 'ifng', options: TF.map((k) => [k, SHORT[k]]), onChange: (v) => { fk = TF.indexOf(v); thr2 = null; paint2(); } });
  select(f2g.controls, { label: 'Impurity measure', value: crit2, options: [['gini', 'Gini'], ['entropy', 'Entropy'], ['error', 'Misclassification error']], onChange: (v) => { crit2 = v; paint2(); } });
  const s2 = stats(f2g.root);
  const row2 = el('div', { class: 'fig-row' }); f2g.body.append(row2);
  const c2a = el('div', { style: 'flex: 0 1 340px' }), c2b = el('div'); row2.append(c2a, c2b);
  const A2 = frame(c2a, { width: 340, height: 330, x: [0, 1], y: [0, 1.02], xLabel: 'share of responders in the group, p', yLabel: 'impurity', grid: false });
  const B2 = frame(c2b, { width: 560, height: 330, x: [0, 1], y: [0, 0.6], xLabel: 'cut-off', yLabel: 'weighted impurity of the two groups', grid: false });
  legend(f2g.body, [['parent group', ink()], ['"no" group', cat(0)], ['"yes" group', cat(1)], ['weighted average of the two', cat(2)]]);
  const curveL = A2.layer('curve'), ptsA = A2.layer('pts'), scanL = B2.layer('scan');
  const hit2 = B2.layer('hit').append('rect').attr('width', B2.w).attr('height', B2.h).attr('fill', 'none').style('pointer-events', 'all').style('cursor', 'crosshair');
  let sc2 = [];
  const nearest = (e) => { const [mx] = d3.pointer(e, hit2.node()); const v = B2.x.invert(mx); return sc2.reduce((a, b) => (Math.abs(b.thr - v) < Math.abs(a.thr - v) ? b : a)); };
  hit2.on('click', (e) => { if (sc2.length) { thr2 = nearest(e).thr; paint2(); } })
    .on('mousemove', (e) => { if (sc2.length) { const s = nearest(e); tip(e, `cut-off ${fval(TF[fk], s.thr)}<br>weighted impurity ${f3(s.w)}<br>${s.nl} patients "no", ${s.nr} "yes"`); } }).on('mouseleave', untip);
  function paint2() {
    const all = XD.map((_, i) => i), p0 = base, i0 = impurity(p0, crit2), ymax = crit2 === 'entropy' ? 1.02 : 0.52;
    sc2 = scanSplits(XD, y, all, fk, { minLeaf: 1, crit: crit2 });
    const bestS = sc2.reduce((a, b) => (b.w < a.w ? b : a), sc2[0]);
    const cur = thr2 === null ? bestS : sc2.reduce((a, b) => (Math.abs(b.thr - thr2) < Math.abs(a.thr - thr2) ? b : a));
    A2.y.domain([0, ymax]); A2.g.select('.axis.y').call(d3.axisLeft(A2.y).ticks(5));
    B2.y.domain([0, ymax * 1.02]); B2.g.select('.axis.y').call(d3.axisLeft(B2.y).ticks(5));
    const xs = d3.extent(XD, (r) => r[fk]); B2.x.domain([xs[0], xs[1]]); B2.g.select('.axis.x').call(d3.axisBottom(B2.x).ticks(6).tickFormat((v) => fval(TF[fk], v)));
    curveL.selectAll('path').data([d3.range(0, 1.001, 0.01)]).join('path').attr('fill', 'none').attr('stroke', ink2()).attr('stroke-width', 2).attr('d', d3.line().x((p) => A2.x(p)).y((p) => A2.y(impurity(p, crit2))));
    const pts = [{ p: p0, v: i0, c: ink(), l: 'parent', dy: -10 }, { p: cur.pl, v: impurity(cur.pl, crit2), c: cat(0), l: '"no"', dy: 4, dx: -11 }, { p: cur.pr, v: impurity(cur.pr, crit2), c: cat(1), l: '"yes"', dy: 4, dx: 11 }, { p: p0, v: cur.w, c: cat(2), l: 'weighted', dy: 20 }];
    ptsA.selectAll('line').data([0]).join('line').attr('x1', A2.x(cur.pl)).attr('y1', A2.y(pts[1].v)).attr('x2', A2.x(cur.pr)).attr('y2', A2.y(pts[2].v)).attr('stroke', muted()).attr('stroke-dasharray', '5 4');
    ptsA.selectAll('line.gap').data([0]).join('line').attr('class', 'gap').attr('x1', A2.x(p0)).attr('x2', A2.x(p0)).attr('y1', A2.y(i0)).attr('y2', A2.y(cur.w)).attr('stroke', ink()).attr('stroke-width', 2.2);
    ptsA.selectAll('circle').data(pts).join('circle').attr('cx', (d) => A2.x(d.p)).attr('cy', (d) => A2.y(d.v)).attr('r', 6).attr('fill', (d) => d.c).attr('stroke', css('--surface')).attr('stroke-width', 1.5);
    ptsA.selectAll('text').data(pts).join('text').attr('x', (d) => A2.x(d.p) + (d.dx || (d.l === 'weighted' ? 10 : 0))).attr('y', (d) => A2.y(d.v) + d.dy).attr('text-anchor', (d) => (d.l === 'weighted' || d.dx > 0 ? 'start' : d.dx < 0 ? 'end' : 'middle')).attr('font-size', 11.5).attr('fill', ink()).text((d) => d.l);
    scanL.selectAll('*').remove();
    scanL.append('line').attr('x1', 0).attr('x2', B2.w).attr('y1', B2.y(i0)).attr('y2', B2.y(i0)).attr('stroke', muted()).attr('stroke-dasharray', '5 4');
    scanL.append('text').attr('x', B2.w - 4).attr('y', B2.y(i0) - 6).attr('text-anchor', 'end').attr('font-size', 11.5).attr('fill', ink2()).text('impurity before splitting (parent)');
    if (sc2.length > 1) scanL.append('path').datum(sc2).attr('fill', 'none').attr('stroke', cat(6)).attr('stroke-width', 1.8).attr('d', d3.line().x((s) => B2.x(s.thr)).y((s) => B2.y(s.w)));
    else scanL.selectAll('circle.o').data(sc2).join('circle').attr('class', 'o').attr('cx', (s) => B2.x(s.thr)).attr('cy', (s) => B2.y(s.w)).attr('r', 5).attr('fill', cat(6));
    scanL.append('path').attr('d', d3.symbol(d3.symbolCross, 110)).attr('transform', `translate(${B2.x(bestS.thr)},${B2.y(bestS.w)})`).attr('fill', ink()).attr('stroke', css('--surface'));
    scanL.append('circle').attr('cx', B2.x(cur.thr)).attr('cy', B2.y(cur.w)).attr('r', 6).attr('fill', cat(2)).attr('stroke', css('--surface')).attr('stroke-width', 1.8);
    scanL.append('text').attr('x', B2.x(bestS.thr) + (B2.x(bestS.thr) > B2.w - 120 ? -10 : 10)).attr('y', B2.y(bestS.w) + 18).attr('text-anchor', B2.x(bestS.thr) > B2.w - 120 ? 'end' : 'start').attr('font-size', 11.5).attr('fill', ink()).text(`best: ${fval(TF[fk], bestS.thr)}`);
    s2.set({ 'impurity of parent': f3(i0), '"no" group': `${cur.nl} patients, ${f3(impurity(cur.pl, crit2))}`, '"yes" group': `${cur.nr} patients, ${f3(impurity(cur.pr, crit2))}`, 'weighted': f3(cur.w), 'reduction': f3(i0 - cur.w), 'best cut-off for this feature': `${fval(TF[fk], bestS.thr)} (reduction ${f3(i0 - bestS.w)})` });
  }
  paint2();
  const bestAll = TF.map((k, f) => { const s = scanSplits(XD, y, XD.map((_, i) => i), f, { minLeaf: 1 }); const b = s.reduce((a, c) => (c.w < a.w ? c : a)); return { k, f, w: b.w, thr: b.thr }; }).sort((a, b) => a.w - b.w);
  const gE = (crit) => { const s = scanSplits(XD, y, XD.map((_, i) => i), 0, { minLeaf: 1, crit }); return s.reduce((a, c) => (c.w < a.w ? c : a)).thr; };
  prose(rb, `<p>Across all ${TF.length} measurements, the best first question is <b>${cond(TF, bestAll[0].f, bestAll[0].thr, true)}</b> (weighted Gini ${f3(bestAll[0].w)}), followed by ${cond(TF, bestAll[1].f, bestAll[1].thr, true)} (${f3(bestAll[1].w)}) and ${cond(TF, bestAll[2].f, bestAll[2].thr, true)} (${f3(bestAll[2].w)}). For IFN-γ, Gini places the cut-off at ${f2(gE('gini'))} and entropy at ${f2(gE('entropy'))}; the choice of impurity measure rarely changes a tree much. The number of candidate cut-offs per feature is the number of distinct observed values minus one, so a continuous marker such as IFN-γ has ${scanSplits(XD, y, XD.map((_, i) => i), 0, { minLeaf: 1 }).length} candidates, and a yes/no marker such as Centre B has one.</p>`);
  quiz(rb, 'Why is the average of the two child impurities weighted by group size?', '<p>A cut-off that isolates two patients in a pure group and leaves the other patients mixed would otherwise appear as good as one that improves everyone. Weighting makes each patient count equally: the score is the average impurity per patient after the split.</p>');

  // ---- Step 3: growing the tree + rules
  rb = rung(ch, 3, 'Growing the tree and reading its rules',
    `<p>Once the best first question is chosen, the same search is repeated <b>separately inside each of the two groups</b>, then inside each of their subgroups, and so on. This is <b>greedy</b>: every split is the best available at that point, without looking ahead. Growth stops at a leaf when the group is pure, when it is too small to split (here, a leaf must keep at least 5 patients), or when the maximum depth is reached.</p>`);
  const f3g = figure(rb, 'Grow the tree split by split', 'Each press expands the leaf with the largest weighted impurity reduction. The table lists every current leaf as a rule, with the share of responders among the discovery patients (used to grow the tree) and among the independent validation patients (Centre C). Hover a box for its counts and impurity.');
  const expanded = new Set([T3.id]);
  const cut3 = () => { const s = new Set(); (function w(nd) { if (nd.left) { if (!expanded.has(nd.id)) s.add(nd.id); else { w(nd.left); w(nd.right); } } })(T3); return s; };
  const nextSplit = () => { const o = { cut: cut3() }, cand = treeLeaves(T3, o).filter((l) => l.left); return cand.length ? cand.reduce((a, b) => (b.wgain > a.wgain ? b : a)) : null; };
  const grow3 = () => { const nx = nextSplit(); if (nx) { expanded.add(nx.id); paint3(); } };
  button(f3g.controls, 'Grow the next split', grow3, true);
  button(f3g.controls, 'Grow to depth 3', () => { let g = 0; while (nextSplit() && g++ < 20) expanded.add(nextSplit().id); paint3(); });
  button(f3g.controls, 'Reset', () => { expanded.clear(); paint3(); });
  const s3 = stats(f3g.root);
  const tHost = el('div'); f3g.body.append(tHost);
  const draw3 = drawTree(tHost, TF, { height: 300 });
  const rulesBox = el('div'); f3g.body.append(rulesBox);
  function paint3() {
    const o = { cut: cut3() }; draw3(T3, o);
    const lv = leafRules(T3, TF, o);
    const vc = (nd) => { const m = V.filter((p, i) => treeLeaf(T3, XV[i], o) === nd); return { n: m.length, p: m.length ? mean(m.map((p) => p.response)) : NaN }; };
    rulesBox.innerHTML = '';
    table(rulesBox, lv.map((l) => ({ r: l.text, n: l.node.n, p: pc(l.node.p), v: (() => { const q = vc(l.node); return q.n ? `${pc(q.p)} (n = ${q.n})` : 'none'; })() })), [['r', 'Rule (leaf)'], ['n', 'Discovery n'], ['p', 'Discovery responded'], ['v', 'Validation responded']], { cls: 'compact' });
    s3.set({ leaves: lv.length, depth: d3.max(lv, (l) => l.node.depth), 'training accuracy': pc(acc(T3, XD, y, o)), 'validation accuracy': pc(acc(T3, XV, yv, o)), 'validation AUC': f3(auc(sc(T3, XV, o), yv)) });
  }
  paint3();
  const lv3 = leafRules(T3, TF), hiL = lv3.reduce((a, b) => (b.node.p > a.node.p ? b : a)), loL = lv3.reduce((a, b) => (b.node.p < a.node.p ? b : a));
  const usedAt2 = T3.left && T3.left.left ? `${SHORT[TF[T3.left.f]]} on the "no" side and ${SHORT[TF[T3.right.f]]} on the "yes" side` : '';
  prose(rb, `<p>Read the table as rules: the leaf with the highest response rate is "${hiL.text.replace(/<\/?b>/g, '')}" (${pc(hiL.node.p)} of ${hiL.node.n} discovery patients), and the lowest is "${loL.text.replace(/<\/?b>/g, '')}" (${pc(loL.node.p)} of ${loL.node.n}). Each branch is free to ask different follow-up questions: at the second level the tree uses ${usedAt2}. The depth-3 tree has a validation AUC of ${f3(auc(sc(T3, XV), yv))}.</p>`);

  // ---- Step 4: interactions
  const K2 = ['ifng', 'log2tmb'], X2 = rowsK(D, K2), X2v = rowsK(V, K2);
  const T2 = fitTree(X2, y, { maxDepth: 8, minLeaf: 3 });
  const st2 = standardize(X2), fitL2 = fitLogistic(st2.Z, y, { l2: 0.01 });
  const lw = (r) => { const z = st2.apply(r); return fitL2.b0 + fitL2.w[0] * z[0] + fitL2.w[1] * z[1]; };
  const aucL2 = auc(X2v.map((r) => sigmoid(lw(r))), yv), accL2 = mean(X2v.map((r, i) => ((lw(r) >= 0 ? 1 : 0) === yv[i] ? 1 : 0)));
  rb = rung(ch, 4, 'Interactions and the shape of the boundary',
    `<p>Because every branch is split on its own, a tree represents <b>interactions</b> without being told about them: "high TMB matters only when IFN-γ is also high" is a path of two questions. The price is the shape of its boundary. Every split is <b>axis-aligned</b> (a threshold on one measurement), so the regions are rectangles and a diagonal trade-off between two measurements is approximated by a staircase. The figure uses IFN-γ and TMB only.</p>`);
  const f4g = figure(rb, 'Rectangles from a tree, and the straight boundary of logistic regression', 'Each rectangle is one leaf, shaded by its share of responders (blue to orange). Dots are discovery patients. The dashed line is the p = 0.5 boundary of logistic regression on the same two measurements. Move the depth slider to see the partition refine.');
  let dep4 = 2, show4 = true;
  slider(f4g.controls, { label: 'maximum depth', min: 1, max: 8, step: 1, value: dep4, onInput: (v) => { dep4 = v; paint4(); } });
  checkbox(f4g.controls, { label: 'show logistic boundary', checked: show4, onChange: (c) => { show4 = c; paint4(); } });
  const s4 = stats(f4g.root);
  const F4 = frame(f4g.body, { width: 820, height: 380, x: [2.2, 10.2], y: [0.6, 5.5], xLabel: 'IFN-γ score', yLabel: 'log2 TMB (mut/Mb)', grid: false });
  F4.g.append('clipPath').attr('id', 'dt4-clip').append('rect').attr('width', F4.w).attr('height', F4.h);
  const rects4 = F4.layer('rects'), lg4 = F4.layer('lg').attr('clip-path', 'url(#dt4-clip)');
  F4.layer('pts').selectAll('circle').data(D).join('circle').attr('cx', (d) => F4.x(d.ifng)).attr('cy', (d) => F4.y(d.log2tmb)).attr('r', 3.8).attr('fill', (d) => cat(d.response)).attr('stroke', css('--surface')).attr('stroke-width', 1)
    .on('mousemove', (e, d) => tip(e, `${d.id}<br>IFN-γ ${d.ifng}, TMB ${d.tmb} mut/Mb<br>${d.response ? 'responded' : 'no response'}`)).on('mouseleave', untip);
  legend(f4g.body, [['no response', cat(0)], ['responded', cat(1)], ['logistic p = 0.5', ink()]]);
  function boxes(nd, b, o, out = []) {
    if (cut0(nd, o)) { out.push({ nd, b }); return out; }
    const lb = b.slice(), rb2 = b.slice(); lb[nd.f ? 3 : 1] = nd.thr; rb2[nd.f ? 2 : 0] = nd.thr;
    boxes(nd.left, lb, o, out); boxes(nd.right, rb2, o, out); return out;
  }
  function paint4() {
    const o = { maxDepth: dep4 }, bx = boxes(T2, [2.2, 10.2, 0.6, 5.5], o);
    rects4.selectAll('rect').data(bx).join('rect').attr('x', (d) => F4.x(Math.max(2.2, d.b[0]))).attr('width', (d) => F4.x(Math.min(10.2, d.b[1])) - F4.x(Math.max(2.2, d.b[0]))).attr('y', (d) => F4.y(Math.min(5.5, d.b[3]))).attr('height', (d) => F4.y(Math.max(0.6, d.b[2])) - F4.y(Math.min(5.5, d.b[3])))
      .attr('fill', (d) => mix(d.nd.p)).attr('fill-opacity', 0.45).attr('stroke', css('--surface')).attr('stroke-width', 1.2)
      .on('mousemove', (e, d) => tip(e, `leaf: ${d.nd.n} patients, ${pc(d.nd.p)} responded`)).on('mouseleave', untip);
    const xs = [2.2, 10.2], ln = xs.map((a) => { const z0 = (a - st2.mu[0]) / st2.sd[0]; const zt = -(fitL2.b0 + fitL2.w[0] * z0) / fitL2.w[1]; return [a, zt * st2.sd[1] + st2.mu[1]]; });
    lg4.selectAll('line').data(show4 ? [ln] : []).join('line').attr('x1', (d) => F4.x(d[0][0])).attr('y1', (d) => F4.y(d[0][1])).attr('x2', (d) => F4.x(d[1][0])).attr('y2', (d) => F4.y(d[1][1])).attr('stroke', ink()).attr('stroke-width', 2.4).attr('stroke-dasharray', '6 4');
    s4.set({ leaves: bx.length, 'tree: training accuracy': pc(acc(T2, X2, y, o)), 'tree: validation AUC': f3(auc(sc(T2, X2v, o), yv)), 'tree: validation accuracy': pc(acc(T2, X2v, yv, o)), 'logistic: validation AUC': f3(aucL2), 'logistic: validation accuracy': pc(accL2) });
  }
  paint4();
  const best4 = d3.range(1, 9).map((d) => ({ d, a: auc(sc(T2, X2v, { maxDepth: d }), yv) })).reduce((a, b) => (b.a > a.a ? b : a));
  prose(rb, `<p>In this cohort the true response model is an <b>additive</b> function of IFN-γ, log2 TMB and CAF, with no planted interaction. A straight boundary therefore suits the data, and logistic regression (validation AUC ${f3(aucL2)}) is at least as good as any depth of the tree (best tree AUC ${f3(best4.a)} at depth ${best4.d}). The tree has to approximate the diagonal with steps. Where a real interaction or threshold effect exists, the comparison can reverse; trees are not better in general, only able to represent interactions without being told about them.</p>`);

  // ---- Step 5: depth, overfitting, pruning
  const fullT = fitTree(XD, y, { maxDepth: 30, minLeaf: 1 }), dFull = treeDepth(fullT), path = pruningPath(fullT);
  const depthCurve = d3.range(0, dFull + 1).map((d) => { const o = { maxDepth: d }; return { s: d, leaves: treeLeaves(fullT, o).length, tr: acc(fullT, XD, y, o), va: acc(fullT, XV, yv, o), auc: auc(sc(fullT, XV, o), yv) }; });
  const pruneCurve = path.map((p, i) => { const o = { cut: p.cut }; return { s: i, leaves: p.leaves, alpha: p.alpha, tr: acc(fullT, XD, y, o), va: acc(fullT, XV, yv, o), auc: auc(sc(fullT, XV, o), yv) }; });
  // 5-fold cross-validation over the pruning path (candidate alphas are the path's own values)
  const folds = (() => { const r = rng(11), idx = r.shuffle(d3.range(n)); return d3.range(5).map((k) => idx.filter((_, i) => i % 5 === k)); })();
  const alphas = path.map((p) => p.alpha), cvAcc = alphas.map(() => 0);
  folds.forEach((te) => {
    const trI = d3.range(n).filter((i) => !te.includes(i)), t = fitTree(XD, y, { maxDepth: 30, minLeaf: 1, idx: trI }), pp = pruningPath(t);
    alphas.forEach((a, j) => { let q = pp[0]; pp.forEach((e) => { if (e.alpha <= a * 1.0000001 + 1e-12) q = e; }); cvAcc[j] += sum(te.map((i) => ((treeLeaf(t, XD[i], { cut: q.cut }).p >= 0.5 ? 1 : 0) === y[i] ? 1 : 0))) / n; });
  });
  const cvBest = cvAcc.reduce((bi, v, j) => (v >= cvAcc[bi] ? j : bi), 0), cvStep = pruneCurve[cvBest];
  rb = rung(ch, 5, 'Depth, overfitting, pruning and instability',
    `<p>If growth is not limited, the tree keeps splitting until every leaf is pure, which here takes ${treeLeaves(fullT).length} leaves and ${dFull} levels. Such a tree reproduces the training labels, noise included, and generalises worse. Two ways to limit it: stop early (a <b>maximum depth</b> or minimum leaf size), or grow fully and then <b>prune</b>. <b>Cost-complexity pruning</b> repeatedly removes the branch whose removal adds the least training error per leaf saved, which gives a sequence of smaller and smaller trees.</p>`);
  const f5g = figure(rb, 'Training and validation accuracy as the tree is made simpler', 'Left: accuracy against the number of leaves for the chosen method. Right: the fully grown tree; each dot is a node (area = patients, colour = share of responders); faded nodes are removed by the current setting and outlined dots are the resulting leaves. Validation = the independent Centre C patients.');
  let meth = 'depth', dep5 = 3, step5 = Math.max(0, path.length - 1 - 6);
  const selM = select(f5g.controls, { label: 'Limit complexity by', value: meth, options: [['depth', 'maximum depth'], ['prune', 'pruning']], onChange: (v) => { meth = v; paint5(); } });
  slider(f5g.controls, { label: 'maximum depth', min: 0, max: dFull, step: 1, value: dep5, onInput: (v) => { dep5 = v; meth = 'depth'; selM.value = 'depth'; paint5(); } });
  const sl5 = slider(f5g.controls, { label: 'pruning step', min: 0, max: path.length - 1, step: 1, value: step5, fmt: (v) => `${v} (${path[v].leaves} leaves)`, onInput: (v) => { step5 = v; meth = 'prune'; selM.value = 'prune'; paint5(); } });
  button(f5g.controls, 'Cross-validated choice', () => { step5 = cvBest; sl5.set(step5); meth = 'prune'; selM.value = 'prune'; paint5(); });
  const s5 = stats(f5g.root);
  const row5 = el('div', { class: 'fig-row' }); f5g.body.append(row5);
  const c5a = el('div', { style: 'flex: 0 1 440px' }), c5b = el('div'); row5.append(c5a, c5b);
  const maxLv = path[0].leaves;
  const A5 = frame(c5a, { width: 440, height: 340, x: [0, maxLv + 1], y: [0.45, 1.01], xLabel: 'number of leaves', yLabel: 'accuracy' });
  const curves5 = A5.layer('c'), mk5 = A5.layer('m');
  const draw5 = drawTree(c5b, TF, { width: 520, height: 340, compact: true });
  legend(f5g.body, [['training accuracy', cat(2)], ['validation accuracy', cat(6)]]);
  function paint5() {
    const cur = meth === 'depth' ? depthCurve[dep5] : pruneCurve[step5], cv = meth === 'depth' ? depthCurve : pruneCurve;
    curves5.selectAll('path').data([['tr', cat(2)], ['va', cat(6)]]).join('path').attr('fill', 'none').attr('stroke', (d) => d[1]).attr('stroke-width', 2.2).attr('d', (d) => d3.line().x((q) => A5.x(q.leaves)).y((q) => A5.y(q[d[0]])).curve(d3.curveStepAfter)(cv.slice().sort((a, b) => a.leaves - b.leaves)));
    mk5.selectAll('*').remove();
    mk5.append('line').attr('x1', A5.x(cur.leaves)).attr('x2', A5.x(cur.leaves)).attr('y1', 0).attr('y2', A5.h).attr('stroke', muted()).attr('stroke-dasharray', '4 4');
    [['tr', cat(2)], ['va', cat(6)]].forEach(([k, c]) => mk5.append('circle').attr('cx', A5.x(cur.leaves)).attr('cy', A5.y(cur[k])).attr('r', 6).attr('fill', c).attr('stroke', css('--surface')).attr('stroke-width', 1.8));
    mk5.append('text').attr('x', A5.x(maxLv * 0.5)).attr('y', A5.y(0.52)).attr('font-size', 11.5).attr('fill', ink2()).attr('text-anchor', 'middle').text(meth === 'depth' ? 'each step along the curve is one more level' : 'each step along the curve is one more pruned branch');
    draw5(fullT, meth === 'depth' ? { maxDepth: dep5 } : { cut: path[step5].cut });
    s5.set({ leaves: cur.leaves, 'training accuracy': pc(cur.tr), 'validation accuracy': pc(cur.va), 'validation AUC': f3(cur.auc), gap: `${Math.round(100 * (cur.tr - cur.va))} points` });
  }
  paint5();
  const bestVa = pruneCurve.reduce((a, b) => (b.va > a.va ? b : a)), fullRow = pruneCurve[0];
  prose(rb, `<p>The unpruned tree has ${pc(fullRow.tr)} training accuracy and ${pc(fullRow.va)} validation accuracy: a gap of ${Math.round(100 * (fullRow.tr - fullRow.va))} points. Pruning to the size chosen by 5-fold cross-validation on the discovery patients alone (${cvStep.leaves} leaves) gives ${pc(cvStep.tr)} training and ${pc(cvStep.va)} validation accuracy. The best validation accuracy anywhere on the pruning path is ${pc(bestVa.va)} at ${bestVa.leaves} leaves, but choosing the size by looking at the validation patients would leak them into the model. With ${V.length} validation patients, one patient changes accuracy by ${f1(100 / V.length)} points, so the curve is noisy and small differences between neighbouring sizes should not be interpreted.</p>`);
  quiz(rb, 'Why does training accuracy never decrease as the tree grows, while validation accuracy does?', '<p>Each extra split can only separate the training patients further, so training accuracy cannot fall. Late splits mostly fit noise in the training patients, which does not carry over to new patients; validation accuracy therefore flattens or drops.</p>');

  // ---- Step 6: instability
  prose(rb, `<p><b>Instability.</b> A tree built greedily is sensitive to the sample. A small change in the data can alter the root split, and everything below it changes with it. This can be examined by drawing <b>bootstrap resamples</b> (patients sampled with replacement, same size as the original) and refitting.</p>`);
  const f6g = figure(rb, 'Refitting on bootstrap resamples', 'Left: the feature chosen for the first question in each resample. Right: validation AUC of each refitted tree (grey bars) and of the tree fitted on the original discovery data (line). Press the button for a new set of resamples.');
  let seed6 = 100, depth6 = 3;
  select(f6g.controls, { label: 'Tree depth', value: '3', options: [['2', '2'], ['3', '3'], ['4', '4']], onChange: (v) => { depth6 = +v; paint6(); } });
  button(f6g.controls, 'Draw 100 new resamples', () => { seed6 += 1; paint6(); }, true);
  const s6 = stats(f6g.root);
  const row6 = el('div', { class: 'fig-row' }); f6g.body.append(row6);
  const c6a = el('div', { style: 'flex: 0 1 400px' }), c6b = el('div'); row6.append(c6a, c6b);
  const A6 = frame(c6a, { width: 400, height: 300, x: [0, 100], y: [0, 1], xLabel: 'resamples (of 100) with this first question', grid: false, axes: false, margin: { left: 100, right: 24 } });
  A6.g.append('g').attr('class', 'axis x').attr('transform', `translate(0,${A6.h})`).call(d3.axisBottom(A6.x).ticks(5));
  const B6 = frame(c6b, { width: 520, height: 300, x: [0.5, 0.95], y: [0, 1], xLabel: 'validation AUC of the refitted tree', yLabel: 'number of resamples' });
  legend(f6g.body, [['same feature as the original tree', cat(6)], ['a different feature', cat(7)], ['validation AUC, original data', ink()]]);
  const cnt6 = A6.layer('cnt'), hist6 = B6.layer('h');
  function paint6() {
    const r = rng(seed6), ref = evalTree(TF, { depth: depth6 }), roots = {}, aucs = [], thrs = [];
    for (let b = 0; b < 100; b++) {
      const idx = d3.range(n).map(() => r.int(n)), t = fitTree(XD, y, { maxDepth: depth6, minLeaf: 5, idx });
      const k = t.left ? TF[t.f] : 'none'; roots[k] = (roots[k] || 0) + 1; if (t.left && TF[t.f] === TF[ref.t.f]) thrs.push(t.thr);
      aucs.push(auc(sc(t, XV), yv));
    }
    const rows = Object.entries(roots).sort((a, b) => b[1] - a[1]), yb = d3.scaleBand().domain(rows.map((q) => q[0])).range([A6.h - Math.min(A6.h, 40 * rows.length), A6.h]).padding(0.3);   // bars sit on the axis
    cnt6.selectAll('*').remove();
    rows.forEach(([k, c]) => {
      cnt6.append('rect').attr('x', 0).attr('y', yb(k)).attr('width', A6.x(c)).attr('height', yb.bandwidth()).attr('fill', cat(k === TF[ref.t.f] ? 6 : 7)).on('mousemove', (e) => tip(e, `${SHORT[k] || k}: ${c} of 100 resamples`)).on('mouseleave', untip);
      cnt6.append('text').attr('x', -8).attr('y', yb(k) + yb.bandwidth() / 2 + 4).attr('text-anchor', 'end').attr('font-size', 12).attr('fill', ink2()).text(SHORT[k] || k);
      cnt6.append('text').attr('x', A6.x(c) + 5).attr('y', yb(k) + yb.bandwidth() / 2 + 4).attr('font-size', 11.5).attr('fill', ink()).text(c);
    });
    const bins = d3.bin().domain([0.5, 0.95]).thresholds(d3.range(0.5, 0.951, 0.025))(aucs), my = Math.max(5, d3.max(bins, (b) => b.length));
    B6.y.domain([0, my + 1]); B6.g.select('.axis.y').call(d3.axisLeft(B6.y).ticks(5));
    hist6.selectAll('*').remove();
    hist6.selectAll('rect').data(bins).join('rect').attr('x', (b) => B6.x(b.x0) + 1).attr('width', (b) => Math.max(0, B6.x(b.x1) - B6.x(b.x0) - 2)).attr('y', (b) => B6.y(b.length)).attr('height', (b) => B6.y(0) - B6.y(b.length)).attr('fill', muted()).attr('fill-opacity', 0.7)
      .on('mousemove', (e, b) => tip(e, `AUC ${f2(b.x0)}–${f2(b.x1)}: ${b.length} resamples`)).on('mouseleave', untip);
    hist6.append('line').attr('x1', B6.x(ref.aucV)).attr('x2', B6.x(ref.aucV)).attr('y1', 0).attr('y2', B6.h).attr('stroke', ink()).attr('stroke-width', 2.2);
    hist6.append('text').attr('x', B6.x(ref.aucV) + 6).attr('y', 14).attr('font-size', 11.5).attr('fill', ink()).text(`original data: ${f3(ref.aucV)}`);
    const q = d3.quantile(thrs.slice().sort(d3.ascending), 0.1), q2 = d3.quantile(thrs.slice().sort(d3.ascending), 0.9);
    s6.set({ 'first question on the original data': SHORT[TF[ref.t.f]], 'resamples with the same first feature': `${roots[TF[ref.t.f]] || 0} of 100`, 'different first features': rows.length, [`10–90% range of its cut-off`]: thrs.length > 3 ? `${fval(TF[ref.t.f], q)} to ${fval(TF[ref.t.f], q2)}` : '—', 'validation AUC, middle 80%': `${f2(d3.quantile(aucs.slice().sort(d3.ascending), 0.1))} to ${f2(d3.quantile(aucs.slice().sort(d3.ascending), 0.9))}` });
  }
  paint6();
  prose(rb, '<p>The first question changes in some resamples because several measurements (IFN-γ, CD8 and PD-L1 all reflect the same immune activity) give almost the same impurity reduction, and a few patients decide which one is marginally better. Even when the first question is the same, its cut-off and the questions below it vary, and the validation AUC of individual trees spans a wide range (see the readouts above). The rules read from a single tree should therefore not be treated as the only description of the data. Averaging many such trees (random forests) reduces this variance, which is the subject of the ensembles page.</p>');

  // ---- Step 7: lab
  rb = rung(ch, 6, 'Tree lab: choose measurements, depth and leaf size',
    `<p>Fitted on the ${n} discovery patients (Centres A, B) and evaluated on the ${V.length} validation patients (Centre C). Logistic regression on the same measurements is shown as a reference.</p>`);
  const fl = figure(rb, 'Tree lab', 'Top: the fitted tree (as boxes when it has at most 8 leaves, otherwise as dots). Below: ROC curves on the validation patients for the tree and for logistic regression with the same measurements; a tree has one point per distinct leaf probability, so its curve is a few straight steps.');
  const sel = new Set(['ifng', 'log2tmb', 'caf']), boxesC = {};
  const presets = el('div', { class: 'controls' }), featRow = el('div', { class: 'controls' });
  fl.controls.before(presets); fl.controls.before(featRow);
  presets.append(el('span', { class: 'fig-caption', style: 'margin:0' }, 'Presets:'));
  const PRE = [['IFN-γ only', ['ifng'], 1, 5], ['True three', ['ifng', 'log2tmb', 'caf'], 3, 5], ['PD-L1 only', ['pdl1'], 2, 5], ['Centre only', ['centreB'], 1, 5], ['Everything, deep', TF, 8, 1], ['Everything, shallow', TF, 3, 10]];
  let depthL = 3, minL = 5, critL = 'gini';
  const MINS = [1, 3, 5, 10, 20];
  const sD = slider(fl.controls, { label: 'maximum depth', min: 1, max: 8, step: 1, value: depthL, onInput: (v) => { depthL = v; paintL(); } });
  const sM = slider(fl.controls, { label: 'minimum leaf size', min: 0, max: MINS.length - 1, step: 1, value: 2, fmt: (i) => MINS[i], onInput: (i) => { minL = MINS[i]; paintL(); } });
  select(fl.controls, { label: 'Impurity', value: critL, options: [['gini', 'Gini'], ['entropy', 'Entropy']], onChange: (v) => { critL = v; paintL(); } });
  PRE.forEach(([lab, ks, d, m]) => button(presets, lab, () => { sel.clear(); ks.forEach((k) => sel.add(k)); TF.forEach((k) => { boxesC[k].checked = sel.has(k); }); depthL = d; minL = m; sD.set(d); sM.set(MINS.indexOf(m)); paintL(); }));
  TF.forEach((k) => { boxesC[k] = checkbox(featRow, { label: SHORT[k], checked: sel.has(k), onChange: (c) => { if (c) sel.add(k); else sel.delete(k); paintL(); } }); });
  const sL = stats(fl.root);
  const la = el('div'), lb = el('div', { style: 'max-width: 460px' }); fl.body.append(la, lb);
  const drawLB = drawTree(la, TF, { width: 820, height: 330 }), drawLD = drawTree(la, TF, { width: 820, height: 330, compact: true });
  const FR = frame(lb, { width: 460, height: 320, x: [0, 1], y: [0, 1], xLabel: 'false-positive rate', yLabel: 'true-positive rate (sensitivity)' });
  FR.layer('d').append('line').attr('x1', FR.x(0)).attr('y1', FR.y(0)).attr('x2', FR.x(1)).attr('y2', FR.y(1)).attr('stroke', muted()).attr('stroke-dasharray', '4 4');
  const rocL = FR.layer('roc'), rocT = FR.layer('t');
  legend(fl.body, [['tree', cat(2)], ['logistic regression', cat(6)], ['chance', muted()]]);
  const noteL = html('', 'p', 'fig-caption'); fl.body.append(noteL);
  function paintL() {
    const keys = TF.filter((k) => sel.has(k));
    if (!keys.length) { sL.set({ note: 'select at least one measurement' }); return; }
    const e = evalTree(keys, { depth: depthL, minLeaf: minL, crit: critL }), lvN = e.leaves;
    d3.select(la).selectAll('svg').style('display', 'none');
    const showBox = lvN <= 8; const svgs = d3.select(la).selectAll('svg').nodes(); d3.select(svgs[showBox ? 0 : 1]).style('display', null);
    (showBox ? drawLB : drawLD)(e.t, { keys: e.keys });
    // the draw functions use TF keys; temporary keys fix for subset models
    rocL.selectAll('path').data([[roc(e.lpV, yv), cat(6)], [roc(sc(e.t, e.Xv), yv), cat(2)]]).join('path').attr('fill', 'none').attr('stroke', (d) => d[1]).attr('stroke-width', 2.2).attr('d', (d) => d3.line().x((q) => FR.x(q.fpr)).y((q) => FR.y(q.tpr))(d[0]));
    sL.set({ leaves: lvN, depth: e.depth, 'training accuracy': pc(e.tr), 'validation accuracy': pc(e.va), 'tree validation AUC': f3(e.aucV), 'logistic validation AUC': f3(e.aucL) });
    noteL.innerHTML = showBox ? '' : 'More than 8 leaves: the tree is drawn as dots (area = patients, colour = share of responders); hover a dot for its counts.';
  }
  paintL();
  // try → see → means table (values computed with the same fitter)
  const e1 = evalTree(['ifng'], { depth: 1 }), eP = evalTree(['pdl1'], { depth: 2 }), eC = evalTree(['centreB'], { depth: 1 }), e3 = evalTree(['ifng', 'log2tmb', 'caf'], { depth: 3 });
  const eD = evalTree(TF, { depth: 8, minLeaf: 1 }), eS = evalTree(TF, { depth: 3, minLeaf: 10 }), eE = evalTree(['ifng', 'log2tmb', 'caf'], { depth: 3, crit: 'entropy' });
  prose(rb, '<p><b>Suggested comparisons</b>:</p>');
  table(rb, [
    { t: '<b>IFN-γ only</b>, depth 1', s: `One question, cut-off ${fval('ifng', e1.t.thr)}; validation AUC ${f3(e1.aucV)}, accuracy ${pc(e1.va)}`, m: 'A single split already captures most of the signal in a single marker; the tree outputs only two probabilities.' },
    { t: '<b>True three</b>, depth 3', s: `Validation AUC ${f3(e3.aucV)} vs ${f3(e3.aucL)} for logistic regression`, m: `The planted effect is additive, so logistic regression ${e3.aucL > e3.aucV ? 'ranks the validation patients better' : 'does not rank them better'}; the tree gives coarser, stepwise probabilities.` },
    { t: '<b>PD-L1 only</b>, depth 2', s: `Validation AUC ${f3(eP.aucV)} (IFN-γ only: ${f3(e1.aucV)})`, m: 'PD-L1 is induced by the same immune activity, so it predicts response but less sharply than the IFN-γ score.' },
    { t: '<b>Centre only</b>, depth 1', s: `Validation AUC ${f3(eC.aucV)}; training accuracy ${pc(eC.tr)}`, m: 'Centre is associated with response in the discovery data only through case mix (more inflamed tumours in Centre B); it is not a usable predictor in a new centre.' },
    { t: '<b>Everything</b>, depth 8, leaf size 1', s: `${eD.leaves} leaves; training accuracy ${pc(eD.tr)}, validation ${pc(eD.va)}; validation AUC ${f3(eD.aucV)}`, m: 'Overfitting: near-perfect on the discovery patients and clearly worse on new ones.' },
    { t: '<b>Everything</b>, depth 3, leaf size 10', s: `${eS.leaves} leaves; training ${pc(eS.tr)}, validation ${pc(eS.va)}; validation AUC ${f3(eS.aucV)}`, m: 'Restricting depth and leaf size narrows the gap between training and validation accuracy.' },
    { t: 'Gini → <b>Entropy</b> (true three, depth 3)', s: `Validation AUC ${f3(e3.aucV)} → ${f3(eE.aucV)}; leaves ${e3.leaves} → ${eE.leaves}`, m: 'The impurity measure usually has little effect compared with depth, leaf size and the sample.' },
  ], [['t', 'Try'], ['s', 'What you see'], ['m', 'What it means']], { cls: 'compact' });

  // =====================================================================================
  // 5. PITFALLS
  // =====================================================================================
  ch = chapter(main, 'pitfalls', 'Pitfalls', 'Common problems, how to recognise them, and how to address them.');
  const fl1 = treeLeaves(fullT), smallest = fl1.reduce((a, b) => (b.n < a.n ? b : a));
  const noiseK = ['prolif', 'age'].map((k) => evalTree([k], { depth: 1, minLeaf: 1 }));
  const allT = evalTree(TF, { depth: 3, minLeaf: 5 }), usedK = new Set(); (function w(nd) { if (nd.left) { usedK.add(TF[nd.f]); w(nd.left); w(nd.right); } })(allT.t);
  const oobAll = evalTree(TF, { depth: 6, minLeaf: 5 });
  const nz = (() => { const r = rng(5), nn = 20, Xn = XD.map((row) => [...row, ...d3.range(nn).map(() => r.normal())]), Xnv = XV.map((row) => [...row, ...d3.range(nn).map(() => r.normal())]), t = fitTree(Xn, y, { maxDepth: 5, minLeaf: 3 }); const u = new Set(); (function w(nd) { if (nd.left) { if (nd.f >= TF.length) u.add(nd.f); w(nd.left); w(nd.right); } })(t); return { used: u.size, tr: acc(t, Xn, y), va: acc(t, Xnv, yv), leaves: treeLeaves(t).length }; })();
  cards(ch, [
    { tag: 'Symptom: near-perfect training accuracy, poor validation', title: 'Overfitting from deep, unrestricted trees', body: `A tree grown to purity has ${treeLeaves(fullT).length} leaves for ${n} patients, and ${pc(fullRow.tr)} training against ${pc(fullRow.va)} validation accuracy. <b>Fix:</b> limit depth, set a minimum leaf size, or prune; choose the size by cross-validation on the training data (step 5).` },
    { tag: 'Symptom: different trees from similar data', title: 'Instability', body: `Resampling the same patients changes the first question and the rules below it (step 5). <b>Fix:</b> do not over-interpret one tree; report how stable the splits are, and use random forests or boosting when prediction matters more than readability.` },
    { tag: 'Symptom: noise measurements chosen as splits', title: 'Selection bias towards many candidate cut-offs', body: `A continuous noise variable offers about ${n - 1} possible cut-offs, so one of them separates the training patients well by chance. A one-question tree on proliferation, which has no effect on response, reaches ${pc(noiseK[0].tr)} training accuracy and ${pc(noiseK[0].va)} validation accuracy. With ${nz.used === 0 ? 'no' : nz.used} of 20 added random columns, a depth-5 tree fits ${pc(nz.tr)} of the training patients and ${pc(nz.va)} of the validation patients. <b>Fix:</b> restrict depth and leaf size, prune, validate on independent patients, and consider methods that correct for this (conditional-inference trees).` },
    { tag: 'Symptom: only roughly correct for diagonal effects', title: 'Axis-aligned boundaries', body: `When the signal is a weighted sum of several markers, a tree needs many splits to approximate it, and the leaves get small (step 4: logistic regression AUC ${f3(aucL2)} vs best tree ${f3(best4.a)} on the two measurements). <b>Fix:</b> compare against a linear model, or give the tree a derived feature such as a signature score.` },
    { tag: 'Symptom: leaf probabilities of 0% or 100%', title: 'Tiny leaves', body: `The smallest leaf of the full tree contains ${smallest.n} patient${smallest.n > 1 ? 's' : ''} and shows ${pc(smallest.p)} responders, a proportion with almost no information. <b>Fix:</b> set a minimum leaf size, report counts next to every percentage, and smooth or calibrate leaf probabilities.` },
    { tag: 'Symptom: the chosen markers change with small differences', title: 'Correlated measurements and confounders', body: `IFN-γ, CD8 and PD-L1 carry overlapping information, and Centre B reflects case mix. A tree on all ${TF.length} measurements uses ${[...usedK].map((k) => SHORT[k]).join(', ')} in its first three levels; which of several related markers is chosen depends on small differences. <b>Fix:</b> pre-specify the candidate measurements, do not read a split as evidence of mechanism, and validate in a new centre.` },
    { tag: 'Symptom: flat predictions at the extremes', title: 'No extrapolation', body: 'A tree predicts a constant in each region, so a patient with a value beyond the observed range gets the prediction of the nearest leaf, however extreme the value. <b>Fix:</b> use a model with a trend (linear, logistic) when extrapolation matters.' },
    { tag: 'Symptom: ties and coarse ranking', title: 'Few distinct probabilities', body: `A tree with ${e3.leaves} leaves gives only ${e3.leaves} distinct scores, so many patients tie and the ROC curve is a few straight segments. <b>Fix:</b> use probabilities from an ensemble, or choose thresholds on the validation data knowing that they are coarse.` },
  ]);

  // =====================================================================================
  // 6. RECAP
  // =====================================================================================
  ch = chapter(main, 'recap', 'Recap', 'Key points.');
  recap(ch, [
    'A decision tree classifies by a sequence of <b>threshold questions</b>; each end point (leaf) predicts the majority outcome and reports the share in that group.',
    'Each split is chosen to minimise the <b>size-weighted impurity</b> (Gini or entropy) of the two groups it creates. The choice is <b>greedy</b>: best now, no look-ahead.',
    'A tree can be read as <b>if-then rules</b>, handles any measurement scale without standardisation and represents interactions without being told about them.',
    'Boundaries are <b>axis-aligned</b>: a smooth, additive effect is approximated by steps, and a linear model can do as well or better.',
    'An unrestricted tree <b>overfits</b>: training accuracy rises to nearly 100% while validation accuracy does not. Limit depth, set a minimum leaf size, or prune, and choose by cross-validation.',
    'A single tree is <b>unstable</b>: resampling the patients changes the first question and the rules below. Do not over-interpret one tree.',
    'Leaf proportions from small leaves are unreliable; report counts, and validate on patients from a different centre.',
    'Ensembles of trees (random forests, boosting) trade readability for stability and accuracy.',
  ]);
  prose(ch, '<h3>In code</h3><p>Both examples use a minimum leaf size and cost-complexity pruning, with the pruning strength chosen by cross-validation on the training data, and report performance on held-out patients.</p>');
  code(ch, {
    Python: `import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

feats = ["ifng", "tmb", "caf", "pdl1", "cd8", "prolif", "age"]   # no scaling or log transform needed
# candidate pruning strengths come from the fully grown tree
path = DecisionTreeClassifier(min_samples_leaf=5, random_state=0) \\
    .cost_complexity_pruning_path(train[feats], train["response"])
grid = GridSearchCV(DecisionTreeClassifier(min_samples_leaf=5, random_state=0),
                    {"ccp_alpha": np.unique(path.ccp_alphas)}, cv=5, scoring="roc_auc")
grid.fit(train[feats], train["response"])
tree = grid.best_estimator_
print(export_text(tree, feature_names=feats))               # the if-then rules
print("validation AUC", roc_auc_score(test["response"], tree.predict_proba(test[feats])[:, 1]))`,
    R: `library(rpart)
# method = "class": Gini by default; cp is the cost-complexity parameter
fit <- rpart(response ~ ifng + tmb + caf + pdl1 + cd8 + prolif + age,
             data = train, method = "class", control = rpart.control(minbucket = 5, cp = 0.001))
printcp(fit)                                   # cross-validated error for each pruning level
best <- fit$cptable[which.min(fit$cptable[, "xerror"]), "CP"]
pruned <- prune(fit, cp = best)
print(pruned)                                  # the if-then rules
p <- predict(pruned, newdata = test, type = "prob")[, "1"]
pROC::auc(test$response, p)`,
  });
  seeAlso(main, [['concepts', 'logistic-regression', 'Logistic regression'], ['concepts', 'train-test', 'Train/test split and overfitting'], ['concepts', 'cross-validation', 'Cross-validation and data leakage'], ['concepts', 'tree-ensembles', 'Random forests and gradient boosting'], ['concepts', 'metrics', 'Evaluating a classifier'], ['concepts', 'regularization', 'Regularisation']]);
}
