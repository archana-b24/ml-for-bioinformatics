// Likelihood & Bayes: coin-flip likelihood as data accumulates; Beta prior x likelihood -> posterior; Gaussian MLE.
import { header, figure, slider, button, stats, section, steps, callout, math, seeAlso, legend, el, tip, untip, animate } from '../../lib/ui.js';
import { frame, cat, css, ink, ink2, muted } from '../../lib/plot.js';
import { rng, mean } from '../../lib/data.js';
const d3 = window.d3;

const TRUE_P = 0.7, NMAX = 200;
const PG = d3.range(0.0025, 1, 0.005); // grid over p (avoids the endpoints)

// Normalised Beta(a, b) density on the grid, computed in log space so large counts don't overflow.
function betaOnGrid(a, b) {
  const lg = PG.map((p) => (a - 1) * Math.log(p) + (b - 1) * Math.log(1 - p));
  const m = d3.max(lg), un = lg.map((v) => Math.exp(v - m));
  const z = d3.sum(un) * 0.005;
  return un.map((v) => v / z);
}
function quantile(dens, q) { let c = 0; for (let i = 0; i < PG.length; i++) { c += dens[i] * 0.005; if (c >= q) return PG[i]; } return PG[PG.length - 1]; }

export default async function render(main) {
  header(main, 'Likelihood & Bayes',
    'The <b>likelihood</b> asks: for each possible parameter value, how probable are the data I actually saw? The <b>maximum-likelihood estimate</b> (MLE) is the peak. Bayes adds a <b>prior</b> (what you believed before) and multiplies: <b>posterior ∝ prior × likelihood</b>. With little data the prior matters; with lots of data the likelihood wins.');

  // ================= Figure 1: coin flips =================
  const r = rng(7);
  const flips = Array.from({ length: NMAX }, () => (r() < TRUE_P ? 1 : 0));
  const cum = [0]; flips.forEach((f, i) => cum.push(cum[i] + f));
  let n = 10, s = 10, m0 = 0.5, stop = null;

  const fig = figure(main, 'Flip a biased coin, watch the estimate sharpen',
    `A coin with hidden heads-probability p (the dotted line shows the truth, p = ${TRUE_P}) is flipped up to ${NMAX} times. Left: prior, likelihood and posterior over p, each scaled to area 1 so they share an axis. Right: the MLE and the posterior mean as n grows, with the posterior 95% credible band.`);
  const nS = slider(fig.controls, { label: 'Flips seen n', min: 0, max: NMAX, step: 1, value: n, onInput: (v) => { halt(); n = v; update(); } });
  slider(fig.controls, { label: 'Prior strength (pseudo-flips)', min: 0, max: 100, step: 1, value: s, onInput: (v) => { s = v; update(); } });
  slider(fig.controls, { label: 'Prior mean', min: 0.05, max: 0.95, step: 0.05, value: m0, fmt: (v) => v.toFixed(2), onInput: (v) => { m0 = v; update(); } });
  button(fig.controls, 'Play', () => play(), true);
  button(fig.controls, 'Reset to n = 0', () => { halt(); n = 0; nS.set(0); update(); });
  const readout = stats(fig.root);

  // flip strip (neutral inks, so it doesn't clash with the series colours below)
  legend(fig.body, [['heads', ink2()], ['tails', muted()], ['outlined = not flipped yet', css('--axis')]]);
  const strip = d3.select(fig.body).append('svg').attr('viewBox', '0 0 900 58').attr('width', 900);
  const CW = 900 / 100;
  const cells = strip.selectAll('rect').data(flips).join('rect')
    .attr('x', (d, i) => (i % 100) * CW + 1).attr('y', (d, i) => Math.floor(i / 100) * 22 + 6).attr('width', CW - 2).attr('height', 18).attr('rx', 2)
    .on('mouseleave', untip);
  cells.on('mousemove', function (e, d) { const i = cells.nodes().indexOf(this); tip(e, `flip ${i + 1}: ${d ? 'heads' : 'tails'}${i >= n ? ' (not seen yet)' : ''}`); });

  const row = el('div', { class: 'fig-row' });
  fig.body.append(row);
  const lDiv = el('div'), rDiv = el('div');
  row.append(lDiv, rDiv);
  legend(lDiv, [['prior', cat(2)], ['likelihood (scaled)', cat(0)], ['posterior', cat(1)]]);
  legend(rDiv, [['MLE = heads / n', cat(0)], ['posterior mean', cat(1)], ['true p', muted()]]);
  const L = frame(lDiv, { width: 470, height: 340, x: [0, 1], y: [0, 5], xLabel: 'p (probability of heads)', yLabel: 'density' });
  L.layer('truth').append('line').attr('x1', L.x(TRUE_P)).attr('x2', L.x(TRUE_P)).attr('y1', 0).attr('y2', L.h).attr('stroke', muted()).attr('stroke-dasharray', '2 3');
  L.layer('curves'); L.layer('marks');
  const T = frame(rDiv, { width: 470, height: 340, x: [0, NMAX], y: [0, 1], xLabel: 'number of flips n', yLabel: 'estimate of p' });
  T.layer('truth').append('line').attr('x1', 0).attr('x2', T.w).attr('y1', T.y(TRUE_P)).attr('y2', T.y(TRUE_P)).attr('stroke', muted()).attr('stroke-dasharray', '2 3');
  T.layer('band'); T.layer('lines'); T.layer('now');

  // Prior Beta(α0, β0) = uniform Beta(1, 1) plus `s` pseudo-flips split by the prior mean. s = 0 is a flat prior.
  const A0 = () => 1 + s * m0, B0 = () => 1 + s * (1 - m0);
  const post = (k) => ({ a: A0() + cum[k], b: B0() + (k - cum[k]) });

  function update() {
    const h = cum[n], t = n - h;
    cells.attr('fill', (d, i) => (i < n ? (d ? ink2() : muted()) : css('--surface-2'))).attr('fill-opacity', (d, i) => (i < n && !d ? 0.45 : 1)).attr('stroke', (d, i) => (i < n ? 'none' : css('--axis')));
    // densities: prior Beta(α0, β0); likelihood p^h(1−p)^t scaled to area 1 = Beta(h+1, t+1); posterior Beta(α0+h, β0+t)
    const prior = betaOnGrid(A0(), B0());
    const lik = betaOnGrid(h + 1, t + 1);
    const pa = A0() + h, pb = B0() + t;
    const postD = betaOnGrid(pa, pb);
    const ymax = Math.min(16, Math.max(3, d3.max([...lik, ...postD, ...prior]) * 1.1));
    L.y.domain([0, ymax]); L.g.select('.axis.y').call(d3.axisLeft(L.y).ticks(5));
    L.g.select('.gridline').call(d3.axisLeft(L.y).ticks(5).tickSize(-L.w).tickFormat(''));
    const ln = (arr) => d3.line().x((p, i) => L.x(p)).y((p, i) => L.y(Math.min(ymax, arr[i])))(PG);
    const ser = [{ k: 'prior', c: 2, d: prior, dash: '5 4' }, { k: 'lik', c: 0, d: lik }, { k: 'post', c: 1, d: postD }];
    L.layer('curves').selectAll('path.area').data([postD]).join('path').attr('class', 'area').attr('fill', cat(1)).attr('fill-opacity', 0.15)
      .attr('d', d3.area().x((p) => L.x(p)).y0(L.y(0)).y1((p, i) => L.y(Math.min(ymax, postD[i])))(PG));
    L.layer('curves').selectAll('path.ln').data(ser).join('path').attr('class', 'ln').attr('fill', 'none').attr('stroke', (d) => cat(d.c)).attr('stroke-width', 2)
      .attr('stroke-dasharray', (d) => d.dash || null).attr('d', (d) => ln(d.d));
    const mle = n ? h / n : NaN, pm = pa / (pa + pb);
    const lo = quantile(postD, 0.025), hi = quantile(postD, 0.975);
    const marks = [
      ...(n ? [{ x: mle, c: 0, lbl: 'MLE' }] : []),
      { x: pm, c: 1, lbl: 'post. mean' },
    ];
    L.layer('marks').selectAll('line').data(marks).join('line').attr('x1', (d) => L.x(d.x)).attr('x2', (d) => L.x(d.x)).attr('y1', L.h).attr('y2', 0).attr('stroke', (d) => cat(d.c)).attr('stroke-width', 1.5).attr('stroke-dasharray', '1 3');
    L.layer('marks').selectAll('text').data(marks).join('text').attr('x', (d) => L.x(d.x) + (d.c ? -4 : 4)).attr('y', 10)
      .attr('text-anchor', (d) => (d.c ? 'end' : 'start')).attr('font-size', 11).attr('fill', ink2()).text((d) => d.lbl);
    // trajectories
    const traj = d3.range(0, NMAX + 1).map((k) => {
      const q = post(k), dens = betaOnGrid(q.a, q.b);
      return { k, mle: k ? cum[k] / k : null, pm: q.a / (q.a + q.b), lo: quantile(dens, 0.025), hi: quantile(dens, 0.975) };
    });
    T.layer('band').selectAll('path').data([traj]).join('path').attr('fill', cat(1)).attr('fill-opacity', 0.15)
      .attr('d', d3.area().x((d) => T.x(d.k)).y0((d) => T.y(d.lo)).y1((d) => T.y(d.hi)));
    T.layer('lines').selectAll('path.m').data([traj.filter((d) => d.k > 0)]).join('path').attr('class', 'm').attr('fill', 'none').attr('stroke', cat(0)).attr('stroke-width', 2)
      .attr('d', d3.line().x((d) => T.x(d.k)).y((d) => T.y(d.mle)));
    T.layer('lines').selectAll('path.p').data([traj]).join('path').attr('class', 'p').attr('fill', 'none').attr('stroke', cat(1)).attr('stroke-width', 2)
      .attr('d', d3.line().x((d) => T.x(d.k)).y((d) => T.y(d.pm)));
    T.layer('now').selectAll('line').data([0]).join('line').attr('x1', T.x(n)).attr('x2', T.x(n)).attr('y1', 0).attr('y2', T.h).attr('stroke', ink()).attr('stroke-width', 1).attr('stroke-dasharray', '3 3');
    T.layer('now').selectAll('circle').data([...(n ? [[mle, 0]] : []), [pm, 1]]).join('circle').attr('cx', T.x(n)).attr('cy', (d) => T.y(d[0])).attr('r', 5.5)
      .attr('fill', (d) => cat(d[1])).attr('stroke', css('--surface')).attr('stroke-width', 1.5);
    readout.set({ 'flips n': n, heads: h, MLE: n ? mle.toFixed(3) : '—', 'prior Beta(α, β)': `${A0().toFixed(1)}, ${B0().toFixed(1)}`, 'posterior mean': pm.toFixed(3), '95% credible interval': `${lo.toFixed(2)}–${hi.toFixed(2)}` });
  }
  function halt() { if (stop) { stop(); stop = null; } }
  function play() {
    halt(); if (n >= NMAX) n = 0;
    let acc = 0;
    stop = animate((dt) => { acc += dt; if (acc < 60) return true; acc = 0; n++; nS.set(n); update(); return n < NMAX; });
  }
  update();

  // ================= Figure 2: Gaussian likelihood =================
  const r2 = rng(23);
  const xs = Array.from({ length: 12 }, () => r2.normal(6, 1.2)).sort((a, b) => a - b);
  const muHat = mean(xs), sigHat = Math.sqrt(mean(xs.map((v) => (v - muHat) ** 2)));
  let mu = 4.5, sig = 2;
  const fig2 = figure(main, 'Likelihood is a function of the parameters',
    'Twelve log-expression values of one gene (dots). The curve is a normal distribution with your μ and σ. Each dot\'s likelihood is the height of the curve above it (vertical bars); the log-likelihood is the sum of their logs. Move the sliders to make the bars as tall as possible overall, then press <b>Jump to MLE</b>. Right: the log-likelihood as a function of μ, with σ held at its current value.');
  const muS = slider(fig2.controls, { label: 'μ (mean)', min: 2, max: 10, step: 0.05, value: mu, fmt: (v) => v.toFixed(2), onInput: (v) => { mu = v; update2(); } });
  const sgS = slider(fig2.controls, { label: 'σ (sd)', min: 0.3, max: 4, step: 0.05, value: sig, fmt: (v) => v.toFixed(2), onInput: (v) => { sig = v; update2(); } });
  button(fig2.controls, 'Jump to MLE', () => { mu = muHat; sig = sigHat; muS.set(+mu.toFixed(2)); sgS.set(+sig.toFixed(2)); update2(); }, true);
  const readout2 = stats(fig2.root);
  const row2 = el('div', { class: 'fig-row' });
  fig2.body.append(row2);
  const gDiv = el('div'), llDiv = el('div');
  row2.append(gDiv, llDiv);
  const G = frame(gDiv, { width: 470, height: 320, x: [1, 11], y: [0, 1], xLabel: 'log₂ expression', yLabel: 'density' });
  G.layer('curve'); G.layer('bars'); G.layer('dots');
  const LL = frame(llDiv, { width: 470, height: 320, x: [2, 10], y: [-60, -15], xLabel: 'μ', yLabel: 'log-likelihood' });
  LL.layer('c'); LL.layer('m');
  const npdf = (x, m, s2) => Math.exp(-0.5 * ((x - m) / s2) ** 2) / (s2 * Math.sqrt(2 * Math.PI));
  const loglik = (m, s2) => d3.sum(xs, (x) => Math.log(npdf(x, m, s2)));
  const maxLL = loglik(muHat, sigHat);

  function update2() {
    const ymax = Math.max(0.35, 1.15 / (sig * Math.sqrt(2 * Math.PI)));
    G.y.domain([0, ymax]); G.g.select('.axis.y').call(d3.axisLeft(G.y).ticks(5));
    G.g.select('.gridline').call(d3.axisLeft(G.y).ticks(5).tickSize(-G.w).tickFormat(''));
    const xg = d3.range(1, 11.001, 0.05);
    G.layer('curve').selectAll('path').data([xg]).join('path').attr('fill', cat(0)).attr('fill-opacity', 0.12).attr('stroke', cat(0)).attr('stroke-width', 2)
      .attr('d', d3.area().x((x) => G.x(x)).y0(G.y(0)).y1((x) => G.y(npdf(x, mu, sig))));
    G.layer('bars').selectAll('line').data(xs).join('line').attr('x1', (x) => G.x(x)).attr('x2', (x) => G.x(x)).attr('y1', G.y(0)).attr('y2', (x) => G.y(npdf(x, mu, sig)))
      .attr('stroke', cat(1)).attr('stroke-width', 2);
    G.layer('dots').selectAll('circle').data(xs).join('circle').attr('cx', (x) => G.x(x)).attr('cy', (x) => G.y(npdf(x, mu, sig))).attr('r', 4.5)
      .attr('fill', cat(1)).attr('stroke', css('--surface')).attr('stroke-width', 1.5)
      .on('mousemove', (e, x) => tip(e, `x = ${x.toFixed(2)}<br>density = ${npdf(x, mu, sig).toFixed(3)}<br>log density = ${Math.log(npdf(x, mu, sig)).toFixed(2)}`)).on('mouseleave', untip);
    const mg = d3.range(2, 10.001, 0.05).map((m) => ({ m, ll: loglik(m, sig) }));
    LL.layer('c').selectAll('path').data([mg]).join('path').attr('fill', 'none').attr('stroke', cat(0)).attr('stroke-width', 2)
      .attr('d', d3.line().defined((d) => d.ll >= -60).x((d) => LL.x(d.m)).y((d) => LL.y(d.ll)));
    const cur = loglik(mu, sig);
    LL.layer('m').selectAll('line').data([0]).join('line').attr('x1', LL.x(muHat)).attr('x2', LL.x(muHat)).attr('y1', 0).attr('y2', LL.h).attr('stroke', muted()).attr('stroke-dasharray', '2 3');
    LL.layer('m').selectAll('text').data([0]).join('text').attr('x', LL.x(muHat) + 4).attr('y', 10).attr('font-size', 11).attr('fill', ink2()).text(`μ̂ = sample mean = ${muHat.toFixed(2)}`);
    LL.layer('m').selectAll('circle').data([0]).join('circle').attr('cx', LL.x(mu)).attr('cy', LL.y(Math.max(-60, cur))).attr('r', 6).attr('fill', cat(1)).attr('stroke', css('--surface')).attr('stroke-width', 1.5);
    readout2.set({ 'log-likelihood': cur.toFixed(2), 'maximum possible': maxLL.toFixed(2), 'MLE μ̂': muHat.toFixed(2), 'MLE σ̂ (÷n)': sigHat.toFixed(2), 'sample sd (÷(n−1))': Math.sqrt(xs.reduce((a, v) => a + (v - muHat) ** 2, 0) / 11).toFixed(2) });
  }
  update2();

  section(main, 'What to notice', `<ul>
    <li><b>Reset to n = 0</b>, then step n up slowly. After 1 flip (heads) the MLE is 1.0, meaning "this coin always lands heads", which is absurd. The posterior mean is 0.54, because the prior holds it near 0.5.</li>
    <li>The likelihood gets <b>narrower</b> as n grows: its width shrinks roughly like 1/√n. Press <b>Play</b> and watch the credible band on the right pinch in.</li>
    <li><b>Set the prior mean to 0.2 and its strength to 100</b> (a confident, wrong prior). With 10 flips (8 heads) the posterior mean is still 0.26, close to the prior. By n = 200 it has moved to 0.52, about two-thirds of the way to the MLE (0.675), but it is still clearly biased low. A prior is worth "strength" extra flips, so 100 pseudo-flips need a few hundred real ones to overrule.</li>
    <li><b>Prior strength 0</b> (a flat prior over p): the posterior and the scaled likelihood are the same curve. Its peak is exactly the MLE; its mean is (heads + 1)/(n + 2), Laplace's rule of succession. Between MLE and Bayes, the only difference is the prior.</li>
    <li>In the Gaussian figure, the best μ is exactly the sample mean, and the best σ is the standard deviation with ÷n, slightly smaller than the usual ÷(n−1) sample sd. MLE variances are biased low in small samples.</li>
  </ul>`);

  steps(main, 'How it works', [
    'Write down a <b>model</b> for how data arise, with parameters θ: each flip is heads with probability p; each expression value is Normal(μ, σ).',
    'The <b>likelihood</b> L(θ) = P(data | θ) treats the data as fixed and θ as the variable. For independent samples it is a product, so we work with the sum of logs (the log-likelihood).',
    'The <b>MLE</b> is the θ that maximises L. For a coin, p̂ = heads/n. For a normal, μ̂ is the mean. Minimising MSE <i>is</i> MLE under Gaussian noise, and minimising log-loss <i>is</i> MLE for a Bernoulli outcome.',
    'Bayes: choose a <b>prior</b> P(θ), then posterior P(θ | data) ∝ P(data | θ) · P(θ). For a coin, a Beta(α, β) prior gives a Beta(α + heads, β + tails) posterior: the prior acts like α + β imaginary flips.',
    'Summarise the posterior by its mean or mode and a <b>credible interval</b>, an interval that contains θ with 95% posterior probability.',
  ]);
  math(main, 'coin:  L(p) = p^h (1 − p)^(n − h)      MLE p̂ = h / n\nprior Beta(α, β)  →  posterior Beta(α + h, β + n − h),   mean = (α + h)/(α + β + n)\n       here α = 1 + strength × prior mean,  β = 1 + strength × (1 − prior mean)   (strength 0 = flat Beta(1, 1))\nGaussian:  log L(μ, σ) = −n log σ − Σ (xᵢ − μ)² / (2σ²) + const   ⇒  μ̂ = x̄,  σ̂² = Σ (xᵢ − x̄)² / n');

  callout(main, '', 'In biology', '<b>DESeq2</b> and <b>edgeR</b> fit negative-binomial GLMs by maximum likelihood, and then shrink the gene-wise dispersions (and, with <code>lfcShrink</code>/apeglm, the fold changes) toward a prior fitted across all genes. That is the "weak prior rescues tiny n" effect above, applied to 20,000 genes with 3 replicates each. <b>limma</b>\'s empirical-Bayes moderated t-statistic does the same for variances. <b>scVI</b> is trained by maximising a lower bound on the likelihood of the counts. Allele-frequency and methylation-fraction estimates from low-coverage reads are exactly the coin-flip problem, which is why beta-binomial models are standard there.');
  callout(main, 'warn', 'Pitfalls', '<ul><li>With little data the MLE can sit on a boundary (p̂ = 0 after 3 tails, or a zero-count gene with a log fold change of ±∞). Priors and regularisation fix this.</li><li>A credible interval and a confidence interval answer different questions; they agree numerically only with a lot of data or a flat prior.</li><li>A strong prior that is wrong takes a lot of data to overturn. Check how sensitive conclusions are to the prior.</li><li>The likelihood is not a probability distribution over θ. It need not integrate to 1, which is why it is scaled here before plotting.</li></ul>');
  seeAlso(main, [['concepts', 'loss-functions', 'Loss functions'], ['concepts', 'regularization', 'Regularization (L1 / L2)'], ['concepts', 'naive-bayes', 'Naive Bayes'], ['concepts', 'multiple-testing', 'Multiple testing & FDR']]);
}
