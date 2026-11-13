// State (Arc Institute 2025): predict a perturbed *distribution* of cells, not one average cell.
// Mini-replication: the same tiny network trained with per-cell MSE on unpaired cells vs with a set-level MMD loss.
import { header, figure, select, button, stats, section, steps, math, legend, seeAlso, animate, tip, untip, el } from '../lib/ui.js';
import { frame, cat, css } from '../lib/plot.js';
import { rng } from '../lib/data.js';
import { sourceCard, toyBadge, coreIdea, realityCheck, useIt, cite } from './_shared.js';
const d3 = window.d3;

// ---------- a tiny tanh MLP with Adam (page-local) ----------
function mlp(sizes, r) {
  const L = sizes.slice(1).map((nout, k) => {
    const nin = sizes[k], s = Math.sqrt(1 / nin), nW = nin * nout;
    return { nin, nout, W: Float64Array.from({ length: nW }, () => r.normal(0, s)), b: new Float64Array(nout),
      gW: new Float64Array(nW), gb: new Float64Array(nout), mW: new Float64Array(nW), vW: new Float64Array(nW), mb: new Float64Array(nout), vb: new Float64Array(nout) };
  });
  let t = 0;
  function forward(x) {
    const acts = [x]; let a = x;
    L.forEach((l, k) => {
      const o = new Float64Array(l.nout);
      for (let i = 0; i < l.nout; i++) { let s = l.b[i]; for (let j = 0; j < l.nin; j++) s += l.W[i * l.nin + j] * a[j]; o[i] = k < L.length - 1 ? Math.tanh(s) : s; }
      acts.push(o); a = o;
    });
    return acts;
  }
  function backward(acts, gOut) {
    let g = gOut;
    for (let k = L.length - 1; k >= 0; k--) {
      const l = L[k], a = acts[k], gin = new Float64Array(l.nin);
      for (let i = 0; i < l.nout; i++) { const gi = g[i]; l.gb[i] += gi; for (let j = 0; j < l.nin; j++) { l.gW[i * l.nin + j] += gi * a[j]; gin[j] += l.W[i * l.nin + j] * gi; } }
      if (k > 0) for (let j = 0; j < l.nin; j++) gin[j] *= 1 - a[j] * a[j];
      g = gin;
    }
  }
  function step(lr) {
    t++; const b1 = 0.9, b2 = 0.999, c1 = 1 - b1 ** t, c2 = 1 - b2 ** t;
    const upd = (P, G, M, V) => { for (let i = 0; i < P.length; i++) { M[i] = b1 * M[i] + (1 - b1) * G[i]; V[i] = b2 * V[i] + (1 - b2) * G[i] * G[i]; P[i] -= lr * (M[i] / c1) / (Math.sqrt(V[i] / c2) + 1e-8); G[i] = 0; } };
    L.forEach((l) => { upd(l.W, l.gW, l.mW, l.vW); upd(l.b, l.gb, l.mb, l.vb); });
  }
  return { forward, backward, step, predict: (x) => { const a = forward(x); return a[a.length - 1]; } };
}

// ---------- toy data ----------
const N = 120;
const SCEN = {
  two: { name: 'Two responder states', gen: (r, i) => [r.normal(1.3, 0.3), r.normal(i % 2 ? 1.1 : -1.1, 0.3)] },
  partial: { name: 'Partial response (40% unaffected)', gen: (r, i) => (i % 5 < 2 ? [r.normal(-1.3, 0.35), r.normal(0, 0.35)] : [r.normal(1.3, 0.3), r.normal(0.7, 0.3)]) },
  shift: { name: 'One shifted, broader state', gen: (r) => [r.normal(1.2, 0.55), r.normal(0, 0.55)] },
};
const BW = [0.25, 0.6, 1.2];
const kern = (dx, dy) => { const q = dx * dx + dy * dy; let s = 0; for (const b of BW) s += Math.exp(-q / (2 * b * b)); return s / BW.length; };
function mmd2(A, B) {
  let aa = 0, bb = 0, ab = 0;
  for (const p of A) for (const q of A) aa += kern(p[0] - q[0], p[1] - q[1]);
  for (const p of B) for (const q of B) bb += kern(p[0] - q[0], p[1] - q[1]);
  for (const p of A) for (const q of B) ab += kern(p[0] - q[0], p[1] - q[1]);
  return aa / (A.length ** 2) + bb / (B.length ** 2) - 2 * ab / (A.length * B.length);
}
const spread = (P) => { const mx = d3.mean(P, (p) => p[0]), my = d3.mean(P, (p) => p[1]); return Math.sqrt(d3.mean(P, (p) => (p[0] - mx) ** 2 + (p[1] - my) ** 2)); };

export default async function render(main) {
  header(main, 'State: predicting distributions, not averages',
    'A perturbation does not move every cell the same way. Some cells respond, some do not, some go one of two ways. A model trained to predict <b>one output per cell with squared error</b> can only learn the average response. A model trained with a <b>distribution loss over a set of cells</b> can reproduce the whole perturbed population.');

  sourceCard(main, {
    org: 'Arc Institute',
    when: '2025',
    links: ['stateNews', 'stateGit'],
    claim: 'State embeds cells (SE), then a set transformer (ST) maps a set of control cells to a set of perturbed cells. Arc describes it as the "first to consistently beat linear baselines".',
  });

  coreIdea(main, `<p>In Perturb-seq you never see the same cell before and after a perturbation: sequencing destroys it. You only have two <b>unpaired populations</b>, control and perturbed. If you pair cells at random and train a regressor with mean squared error (MSE), the best it can do is predict the <i>mean</i> perturbed cell for everyone, because a random partner carries no information about which way an individual cell will go.</p>
  <p><b>Key ML ingredient:</b> a <b>set-level, distributional loss</b>. Instead of comparing cell i to partner i, compare the whole predicted set to the whole observed set. A common choice is <b>MMD</b> (maximum mean discrepancy), which is zero only when two point clouds have the same distribution. Our brief says State's loss is MMD according to secondary sources; <b>we have not verified this</b> against the paper. State also uses a transformer that attends across the cells of a set; this toy keeps only the loss idea.</p>`);

  const fig = figure(main, 'Same network, two losses',
    'Grey-blue dots: control cells (input). Orange rings: observed perturbed cells (target). Green dots: the model\'s predicted perturbed cells, one per control cell. Both models are the same 3→24→24→2 tanh network, given (control x, control y, a fixed random number z per cell). Only the training loss differs.');
  toyBadge(fig, 'two 2-D populations, 120 cells each');
  let scen = 'two', stop = null, it = 0;
  const MAX = 1500;
  select(fig.controls, { label: 'Perturbed population', value: scen, options: Object.entries(SCEN).map(([k, v]) => [k, v.name]), onChange: (v) => { scen = v; reset(); run(); } });
  const runBtn = button(fig.controls, 'Pause', () => { if (stop) { stop(); stop = null; runBtn.textContent = 'Train'; } else run(); }, true);
  button(fig.controls, 'Reset', () => { reset(); paint(); });
  legend(fig.root, [['control cells', cat(0)], ['observed perturbed cells', cat(1)], ['predicted cells', cat(2)]]);
  const readout = stats(fig.root);

  const row = el('div', { class: 'fig-row' }); fig.body.append(row);
  const panels = ['Per-cell MSE on random pairs', 'Set-level MMD loss'].map((t) => {
    const d = el('div'); row.append(d);
    d.append(el('div', { style: 'font-size:13px;font-weight:600;color:var(--ink-2);margin:0 0 2px 40px' }, t));
    return frame(d, { width: 460, height: 380, x: [-2.4, 2.6], y: [-2.2, 2.2], xLabel: 'expression axis 1', yLabel: 'expression axis 2' });
  });

  let X, Z, Y, models, r;
  function reset() {
    if (stop) { stop(); stop = null; }
    const rd = rng(11);
    X = d3.range(N).map(() => [rd.normal(-1.3, 0.35), rd.normal(0, 0.35)]);
    Z = d3.range(N).map(() => rd() * 2 - 1);
    const rt = rng(23);
    Y = d3.range(N).map((i) => SCEN[scen].gen(rt, i));
    r = rng(5);
    models = [mlp([3, 24, 24, 2], rng(3)), mlp([3, 24, 24, 2], rng(3))];
    it = 0;
  }
  const inputs = () => X.map((x, i) => Float64Array.from([x[0], x[1], Z[i]]));
  function trainStep() {
    const U = inputs();
    // model 0: MSE against a freshly shuffled partner (cells are unpaired in reality)
    const perm = r.shuffle(d3.range(N));
    U.forEach((u, i) => { const a = models[0].forward(u), o = a[a.length - 1], y = Y[perm[i]]; models[0].backward(a, [2 * (o[0] - y[0]) / N, 2 * (o[1] - y[1]) / N]); });
    models[0].step(0.01);
    // model 1: MMD between the predicted set and the observed set
    const acts = U.map((u) => models[1].forward(u)), O = acts.map((a) => a[a.length - 1]);
    O.forEach((o, i) => {
      let gx = 0, gy = 0;
      for (const [set, w] of [[O, 2 / (N * N)], [Y, -2 / (N * N)]]) {
        for (const q of set) {
          const dx = o[0] - q[0], dy = o[1] - q[1], qd = dx * dx + dy * dy;
          for (const b of BW) { const k = Math.exp(-qd / (2 * b * b)) / BW.length; gx += w * k * (-dx / (b * b)); gy += w * k * (-dy / (b * b)); }
        }
      }
      models[1].backward(acts[i], [gx, gy]);
    });
    models[1].step(0.01);
    it++;
  }
  const predict = (m) => inputs().map((u) => Array.from(models[m].predict(u)));

  function paint() {
    const preds = [predict(0), predict(1)];
    panels.forEach((F, m) => {
      const pt = (layer, data, style) => F.layer(layer).selectAll('circle').data(data).join('circle').attr('cx', (d) => F.x(d[0])).attr('cy', (d) => F.y(d[1])).call(style);
      pt('ctrl', X, (s) => s.attr('r', 4).attr('fill', cat(0)).attr('opacity', 0.35).attr('stroke', css('--surface')).attr('stroke-width', 1));
      pt('truth', Y, (s) => s.attr('r', 4.5).attr('fill', 'none').attr('stroke', cat(1)).attr('stroke-width', 1.5).attr('opacity', 0.8));
      pt('pred', preds[m], (s) => s.attr('r', 4.5).attr('fill', cat(2)).attr('stroke', css('--surface')).attr('stroke-width', 1.2)
        .on('mousemove', (e, d) => tip(e, `predicted cell<br>(${d[0].toFixed(2)}, ${d[1].toFixed(2)})`)).on('mouseleave', untip));
    });
    const sy = spread(Y);
    readout.set({
      'training step': `${it} / ${MAX}`,
      'MMD², MSE model': mmd2(preds[0], Y).toFixed(3),
      'MMD², MMD model': mmd2(preds[1], Y).toFixed(3),
      'spread: MSE / MMD / observed': `${spread(preds[0]).toFixed(2)} / ${spread(preds[1]).toFixed(2)} / ${sy.toFixed(2)}`,
    });
  }
  function run() {
    if (stop) stop();
    runBtn.textContent = 'Pause';
    let frameN = 0;
    stop = animate(() => {
      for (let k = 0; k < 6 && it < MAX; k++) trainStep();
      if (++frameN % 2 === 0 || it >= MAX) paint();
      if (it >= MAX) { runBtn.textContent = 'Train'; stop = null; return false; }
      return true;
    });
  }
  reset(); paint(); run();

  section(main, 'What to notice', `<ul>
    <li>With <b>two responder states</b>, the MSE model's green cloud shrinks onto the point <i>between</i> the two orange groups, where no real cell lives. Its "spread" readout heads towards zero, while the observed spread stays above 1. The MMD model splits its cells into both groups.</li>
    <li>The MSE model receives the same random number z per cell as the MMD model, and could use it to choose a branch. It learns to ignore z, because under squared error on random partners, hedging at the mean is optimal.</li>
    <li><b>Partial response</b> is the realistic case for a knockdown with incomplete penetrance. The MSE model predicts a modest shift for every cell, a phenotype that no cell actually has. Averaged into a pseudobulk profile, the two models can look similar; the difference only shows at the single-cell level.</li>
    <li>Even the <b>one-state</b> case shows the problem: the mean is right, but the MSE model collapses the cell-to-cell variability.</li>
  </ul>`);

  steps(main, 'How it works', [
    'Embed cells into a space where distances are meaningful (State has a dedicated cell-embedding model, SE; here the "embedding" is just 2-D).',
    'Take a <b>set</b> of control cells from one context and a perturbation label, and predict a set of perturbed cells, one per input cell.',
    'Compare the predicted set with the observed perturbed set using a <b>distribution distance</b> (here MMD with Gaussian kernels at three bandwidths), not cell-by-cell errors.',
    'Back-propagate: each predicted cell is pulled towards nearby observed cells and pushed away from other predicted cells. The push stops the predictions from collapsing to one point.',
    'At test time, feed in control cells and read out a whole perturbed population, from which you can compute means, fractions of responders, or DE.',
  ]);
  math(main, 'MMD²(P, Q) = E k(p,p\') + E k(q,q\') − 2 E k(p,q),    k(a,b) = exp(−‖a−b‖² / 2σ²)\nMSE on random pairs:  argmin_f E‖f(x) − y‖²  =  E[y]  when y is independent of x');

  realityCheck(main, `<ul>
    <li>"First to consistently beat linear baselines" is <b>Arc's own claim</b> (${cite('stateNews', 'Arc news post')}). Independent benchmarks are sobering: ${cite('ahlmann', 'Ahlmann-Eltze et al. 2025')} found nothing beat the training-mean baseline on unseen single perturbations, and ${cite('vcbench', 'VCBench 2026')} reports simple baselines matching or beating foundation models on 4 of 5 axes.</li>
    <li>At the ${cite('vcc2025', 'Virtual Cell Challenge 2025')} (run by Arc), almost nobody beat the mean baseline on MAE, and the winners were hybrids of statistics and deep learning. See the <a href="#/frontier/perturbation-arena">baselines arena</a> and <a href="#/frontier/metric-gaming">metric gaming</a> pages.</li>
    <li>Matching a distribution is not the same as predicting the right biology: a model can produce a realistic-looking cloud that is shifted in the wrong genes. Distribution losses fix collapse; they do not fix weak generalisation to unseen perturbations.</li>
    <li>That State's loss is MMD comes from secondary sources and is <b>unverified</b> in our brief.</li>
  </ul>`);

  useIt(main, `<ul>
    <li>When you evaluate any perturbation predictor (State, scGPT, a linear model), check the <b>predicted per-cell variance</b> and the responder fraction, not just the pseudobulk delta. A model that predicts only means will look fine on pseudobulk metrics.</li>
    <li>For your own data with heterogeneous responses (partial CRISPRi knockdown, drug resistance subclones), compare distributions with MMD or energy distance between predicted and held-out cells in PCA or scVI space.</li>
    <li>If you just need perturbation-level effects for prioritisation, a pseudobulk linear baseline is cheaper and usually as good; reach for set models when the <i>shape</i> of the response matters.</li>
  </ul>`);

  seeAlso(main, [['frontier', 'perturbation-arena', 'Perturbation baselines arena'], ['frontier', 'flow-matching', 'Flow matching'], ['frontier', 'metric-gaming', 'VCC metrics'],
    ['concepts', 'loss-functions', 'Loss functions'], ['concepts', 'transformer', 'Transformers'], ['concepts', 'gan', 'GANs (another distribution-matching idea)'], ['omics', 'sc-pseudobulk', 'Pseudobulk DE']]);
}
