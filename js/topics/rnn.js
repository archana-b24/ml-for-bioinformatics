// Recurrent networks: an unrolled RNN with its hidden state as colour bars, the vanishing-memory
// demo (Jacobian of h_t w.r.t. x_1, computed exactly), and a hand-wired LSTM whose gates are valves.
import { header, figure, slider, select, button, stats, section, steps, callout, math, seeAlso, animate, legend, tip, untip, el } from '../lib/ui.js';
import { frame, cat, css, ink, ink2, muted, diverging } from '../lib/plot.js';
import { rng, sigmoid } from '../lib/data.js';
const d3 = window.d3;

// ---------- tiny dense helpers (plain arrays; sizes are ≤ 8) ----------
const randMat = (r, rows, cols, s) => Array.from({ length: rows }, () => Array.from({ length: cols }, () => r.normal(0, s)));
const mv = (M, v) => M.map((row) => row.reduce((s, w, j) => s + w * v[j], 0));
const mm = (A, B) => A.map((row) => B[0].map((_, j) => row.reduce((s, a, k) => s + a * B[k][j], 0)));
const frob = (M) => Math.sqrt(M.reduce((s, row) => s + row.reduce((t, v) => t + v * v, 0), 0));
function orthogonal(r, n) { // Gram–Schmidt on a seeded Gaussian matrix -> random rotation
  const Q = [];
  for (let i = 0; i < n; i++) {
    let v = Array.from({ length: n }, () => r.normal());
    for (const q of Q) { const d = v.reduce((s, x, k) => s + x * q[k], 0); v = v.map((x, k) => x - d * q[k]); }
    const nv = Math.hypot(...v); Q.push(v.map((x) => x / nv));
  }
  return Q;
}

const H = 8; // hidden units
const SEQS = {
  dna: { name: 'DNA: a promoter fragment', tokens: 'GCTATAAAAGGC'.split(''), kind: 'dna' },
  time: { name: 'Time course: IFN-response gene after a stimulus', tokens: [0, 0.1, 0.9, 1.0, 0.8, 0.55, 0.35, 0.2, 0.12, 0.08, 0.05, 0.03], kind: 'time' },
};
const BASES = ['A', 'C', 'G', 'T'];

export default async function render(main) {
  header(main, 'Recurrent networks (RNNs)',
    'An RNN reads a sequence one step at a time and carries a <b>hidden state</b>: a small vector that summarises everything seen so far. The same weights are reused at every step, so it can read sequences of any length. The catch is that the summary is rewritten at every step, so <b>old information fades</b>. LSTMs add gates that decide what to keep.');

  // =============== Figure 1: unrolled RNN =================
  const f1 = figure(main, 'One cell, unrolled through time',
    'Each column is the hidden state h<sub>t</sub> (8 numbers, blue = negative, red = positive) after reading the token beneath it. Arrows to the right are the <i>same</i> recurrent weights W at every step; arrows up are the same input weights U. Hover a cell for its value.');
  let seqKey = 'dna', shown = SEQS.dna.tokens.length, stop1 = null;
  const r1 = rng(11);
  const W1 = randMat(r1, H, H, 0.9 / Math.sqrt(H));
  const U1 = { dna: randMat(r1, H, 4, 1.0), time: randMat(r1, H, 1, 1.6) };
  const b1 = Array.from({ length: H }, () => r1.normal(0, 0.1));
  select(f1.controls, { label: 'Sequence', value: 'dna', options: Object.entries(SEQS).map(([k, v]) => [k, v.name]), onChange: (k) => { seqKey = k; shown = SEQS[k].tokens.length; stepS.set(shown); draw1(); } });
  const stepS = slider(f1.controls, { label: 'Steps read', min: 1, max: 12, value: 12, onInput: (v) => { shown = v; draw1(); } });
  button(f1.controls, 'Play', () => {
    if (stop1) stop1(); shown = 1; stepS.set(1); draw1(); let acc = 0;
    stop1 = animate((dt) => { acc += dt; if (acc < 450) return true; acc = 0; shown++; stepS.set(shown); draw1(); return shown < SEQS[seqKey].tokens.length; });
  }, true);
  const ro1 = stats(f1.root);
  const svg1Host = el('div'); f1.body.append(svg1Host);

  function runRNN(seq) {
    let h = new Array(H).fill(0); const hs = [];
    for (const tok of seq.tokens) {
      const x = seq.kind === 'dna' ? BASES.map((b) => (b === tok ? 1 : 0)) : [tok];
      const a = mv(W1, h), u = mv(U1[seq.kind], x);
      h = a.map((v, i) => Math.tanh(v + u[i] + b1[i])); hs.push(h);
    }
    return hs;
  }

  function draw1() {
    svg1Host.innerHTML = '';
    const seq = SEQS[seqKey], hs = runRNN(seq), T = seq.tokens.length;
    const Wd = 1080, Hd = 300, cell = 16, colW = 66, x0 = 250, yTop = 40;
    const svg = d3.select(svg1Host).append('svg').attr('viewBox', `0 0 ${Wd} ${Hd}`).attr('width', Wd);
    const defs = svg.append('defs');
    defs.append('marker').attr('id', 'rnn-arr').attr('viewBox', '0 0 10 10').attr('refX', 9).attr('refY', 5).attr('markerWidth', 6).attr('markerHeight', 6).attr('orient', 'auto')
      .append('path').attr('d', 'M0,0 L10,5 L0,10 z').attr('fill', muted());
    const col = diverging(1);
    // rolled view
    const rx = 70, ry = yTop + 30;
    svg.append('rect').attr('x', rx - 30).attr('y', ry).attr('width', 60).attr('height', 70).attr('rx', 10).attr('fill', css('--surface-2')).attr('stroke', ink2());
    svg.append('text').attr('x', rx).attr('y', ry + 40).attr('text-anchor', 'middle').attr('fill', ink()).attr('font-size', 15).text('h');
    svg.append('path').attr('d', `M${rx + 30},${ry + 20} C${rx + 80},${ry - 20} ${rx - 80},${ry - 20} ${rx - 30},${ry + 20}`).attr('fill', 'none').attr('stroke', muted()).attr('stroke-width', 1.5).attr('marker-end', 'url(#rnn-arr)');
    svg.append('text').attr('x', rx).attr('y', ry - 18).attr('text-anchor', 'middle').attr('fill', ink2()).attr('font-size', 12).text('W (loop)');
    svg.append('line').attr('x1', rx).attr('x2', rx).attr('y1', ry + 130).attr('y2', ry + 72).attr('stroke', muted()).attr('stroke-width', 1.5).attr('marker-end', 'url(#rnn-arr)');
    svg.append('text').attr('x', rx).attr('y', ry + 146).attr('text-anchor', 'middle').attr('fill', ink2()).attr('font-size', 12).text('xₜ');
    svg.append('text').attr('x', rx + 8).attr('y', ry + 105).attr('fill', ink2()).attr('font-size', 12).text('U');
    svg.append('text').attr('x', 165).attr('y', ry + 40).attr('text-anchor', 'middle').attr('fill', ink2()).attr('font-size', 13).text('unroll →');
    // h0
    const hY = yTop + 10, tokY = hY + H * cell + 64;
    svg.append('text').attr('x', x0 - 26).attr('y', hY + H * cell / 2 + 4).attr('text-anchor', 'middle').attr('fill', muted()).attr('font-size', 12).text('h₀=0');
    for (let t = 0; t < T; t++) {
      const cx = x0 + t * colW, on = t < shown;
      const g = svg.append('g').attr('opacity', on ? 1 : 0.18);
      hs[t].forEach((v, i) => {
        g.append('rect').attr('x', cx).attr('y', hY + i * cell).attr('width', 26).attr('height', cell - 2).attr('rx', 2)
          .attr('fill', on ? col(v) : css('--surface-2'))
          .on('mousemove', (e) => on && tip(e, `h<sub>${t + 1}</sub>[${i + 1}] = <b>${v.toFixed(3)}</b>`)).on('mouseleave', untip);
      });
      if (t < T - 1) g.append('line').attr('x1', cx + 28).attr('x2', cx + colW - 4).attr('y1', hY + H * cell / 2).attr('y2', hY + H * cell / 2).attr('stroke', muted()).attr('stroke-width', 1.5).attr('marker-end', 'url(#rnn-arr)');
      g.append('line').attr('x1', cx + 13).attr('x2', cx + 13).attr('y1', tokY - 18).attr('y2', hY + H * cell + 4).attr('stroke', muted()).attr('stroke-width', 1.5).attr('marker-end', 'url(#rnn-arr)');
      const tok = seq.tokens[t];
      if (seq.kind === 'dna') g.append('text').attr('x', cx + 13).attr('y', tokY).attr('text-anchor', 'middle').attr('font-family', css('--mono')).attr('font-size', 18).attr('font-weight', 600).attr('fill', ink()).text(tok);
      else {
        g.append('rect').attr('x', cx + 4).attr('y', tokY - 4 - 34 * tok).attr('width', 18).attr('height', Math.max(1, 34 * tok)).attr('fill', cat(0));
        g.append('text').attr('x', cx + 13).attr('y', tokY + 14).attr('text-anchor', 'middle').attr('font-size', 11).attr('fill', ink2()).text(tok.toFixed(2));
      }
      g.append('text').attr('x', cx + 13).attr('y', tokY + 32).attr('text-anchor', 'middle').attr('font-size', 11).attr('fill', muted()).text(`t=${t + 1}`);
      if (t === shown - 1) svg.append('rect').attr('x', cx - 4).attr('y', hY - 6).attr('width', 34).attr('height', H * cell + 10).attr('rx', 5).attr('fill', 'none').attr('stroke', ink()).attr('stroke-width', 1.5);
    }
    svg.append('text').attr('x', x0 - 26).attr('y', tokY).attr('text-anchor', 'middle').attr('fill', muted()).attr('font-size', 12).text('xₜ');
    const h = hs[shown - 1], hp = shown > 1 ? hs[shown - 2] : new Array(H).fill(0);
    ro1.set({ 'step t': shown, '‖hₜ‖': Math.hypot(...h).toFixed(2), '‖hₜ − hₜ₋₁‖ (how much this token rewrote the state)': Math.hypot(...h.map((v, i) => v - hp[i])).toFixed(2), 'weights shared across steps': `${H * H + H * (seq.kind === 'dna' ? 4 : 1) + H} params` });
  }
  draw1();

  // =============== Figure 2: vanishing memory =================
  const f2 = figure(main, 'How much does step 1 still matter at step t?',
    'We feed 40 random inputs into an 8-unit tanh RNN whose recurrent matrix is W = ρ·Q (Q a fixed random rotation, so ρ is exactly its scale). The curve is the size of ∂h<sub>t</sub>/∂x<sub>1</sub>, the sensitivity of the state at step t to the very first input, relative to step 1. It is computed exactly by the chain rule. The same product of Jacobians is what backpropagation-through-time multiplies, so a vanishing curve also means vanishing gradients.');
  let rho = 0.9;
  slider(f2.controls, { label: 'Recurrent weight scale ρ', min: 0.3, max: 2.0, step: 0.05, value: 0.9, fmt: (v) => v.toFixed(2), onInput: (v) => { rho = v; draw2(); } });
  const ro2 = stats(f2.root);
  legend(f2.root, [['tanh RNN (actual)', cat(0)], ['linear RNN reference: ρ^(t−1)', cat(1)]]);
  const row2 = el('div', { class: 'fig-row' }); f2.body.append(row2);
  const stripHost = el('div'), plotHost = el('div'); row2.append(stripHost, plotHost);
  const r2 = rng(21), T2 = 40;
  const Q2 = orthogonal(r2, H), U2 = randMat(r2, H, 4, 0.5), X2 = randMat(r2, T2, 4, 1.0);

  function influence(rhoV) {
    const W = Q2.map((row) => row.map((v) => v * rhoV));
    let h = new Array(H).fill(0), J = null; const out = [];
    for (let t = 0; t < T2; t++) {
      const pre = mv(W, h).map((v, i) => v + mv(U2, X2[t])[i]);
      h = pre.map(Math.tanh);
      const D = h.map((v) => 1 - v * v);
      J = t === 0 ? U2.map((row, i) => row.map((u) => D[i] * u)) : mm(W, J).map((row, i) => row.map((v) => D[i] * v));
      out.push(frob(J));
    }
    return out.map((v) => v / out[0]);
  }

  function draw2() {
    stripHost.innerHTML = ''; plotHost.innerHTML = '';
    const inf = influence(rho);
    const lin = inf.map((_, t) => rho ** t);
    // strip: memory of step 1 as opacity over 40 steps
    const sw = 520, svg = d3.select(stripHost).append('svg').attr('viewBox', `0 0 ${sw} 360`).attr('width', sw);
    svg.append('text').attr('x', 10).attr('y', 20).attr('fill', ink2()).attr('font-size', 13).text('Memory of x₁ carried in hₜ (darker = stronger, log-scaled)');
    const shade = d3.scaleLinear().domain([-4, 0]).range([css('--surface-2'), css('--accent')]).clamp(true);
    const per = 8, cw = 60;
    inf.forEach((v, t) => {
      const rr = Math.floor(t / per), cc = t % per;
      const x = 10 + cc * cw, y = 36 + rr * 62;
      svg.append('rect').attr('x', x).attr('y', y).attr('width', cw - 6).attr('height', 44).attr('rx', 5).attr('fill', shade(Math.log10(Math.max(v, 1e-9))))
        .on('mousemove', (e) => tip(e, `t = ${t + 1}<br>relative influence of x₁: <b>${v.toExponential(2)}</b>`)).on('mouseleave', untip);
      svg.append('text').attr('x', x + (cw - 6) / 2).attr('y', y + 27).attr('text-anchor', 'middle').attr('font-size', 11).attr('fill', Math.log10(v) > -1.6 ? css('--surface') : ink2()).text(t + 1);
    });
    // log plot
    const lo = Math.min(1e-8, d3.min(inf)), hi = Math.max(10, d3.max(inf));
    const F = frame(plotHost, { width: 520, height: 360, x: [1, T2], y: [Math.max(lo, 1e-12), Math.min(hi, 1e8)], yScale: d3.scaleLog(), xLabel: 'step t', yLabel: '‖∂hₜ/∂x₁‖ relative to t=1' });
    F.g.select('.axis.y').call(d3.axisLeft(F.y).ticks(6, '~e'));
    const clampY = (v) => Math.min(Math.max(v, F.y.domain()[0]), F.y.domain()[1]);
    F.layer('ref').append('line').attr('x1', 0).attr('x2', F.w).attr('y1', F.y(0.01)).attr('y2', F.y(0.01)).attr('stroke', muted()).attr('stroke-dasharray', '4 4');
    F.layer('ref').append('text').attr('x', F.w - 4).attr('y', F.y(0.01) - 5).attr('text-anchor', 'end').attr('font-size', 11).attr('fill', muted()).text('1% of original');
    const ln = d3.line().x((d) => F.x(d.x)).y((d) => F.y(clampY(d.y)));
    F.layer('lin').append('path').datum(lin.map((y, i) => ({ x: i + 1, y }))).attr('d', ln).attr('fill', 'none').attr('stroke', cat(1)).attr('stroke-width', 2).attr('stroke-dasharray', '6 4');
    F.layer('rnn').append('path').datum(inf.map((y, i) => ({ x: i + 1, y }))).attr('d', ln).attr('fill', 'none').attr('stroke', cat(0)).attr('stroke-width', 2);
    F.layer('pts').selectAll('circle').data(inf).join('circle').attr('cx', (d, i) => F.x(i + 1)).attr('cy', (d) => F.y(clampY(d))).attr('r', 4)
      .attr('fill', cat(0)).attr('stroke', css('--surface')).attr('stroke-width', 1.2)
      .on('mousemove', (e, d) => tip(e, `t = ${inf.indexOf(d) + 1}: <b>${d.toExponential(2)}</b>`)).on('mouseleave', untip);
    const below = inf.findIndex((v) => v < 0.01);
    ro2.set({ 'influence at t=5': inf[4].toExponential(1), 'at t=20': inf[19].toExponential(1), 'at t=40': inf[39].toExponential(1), 'steps until < 1%': below < 0 ? 'never (in 40)' : below + 1 });
  }
  draw2();

  // =============== Figure 3: LSTM valves =================
  const f3 = figure(main, 'An LSTM remembering a stimulus pulse',
    'A one-cell LSTM with hand-set weights watches a time course. A stimulus pulse arrives at t=4; at t=18 a "query" cue asks "was there a pulse?"; at t=21 a "reset" cue says forget it. Left: the cell for the selected step, drawn as pipes and valves (a valve\'s fill = how open the gate is, pipe thickness = how much flows). Right: every gate and the memory over time, with a plain RNN (recurrent weight 0.9) for comparison.');
  let bf = 5, tSel = 18, stop3 = null;
  slider(f3.controls, { label: 'Forget-gate bias b_f', min: -2, max: 7, step: 0.5, value: 5, fmt: (v) => `${v.toFixed(1)} (f≈${sigmoid(v).toFixed(3)})`, onInput: (v) => { bf = v; draw3(); } });
  const tS = slider(f3.controls, { label: 'Selected step', min: 1, max: 24, value: 18, onInput: (v) => { tSel = v; draw3(); } });
  button(f3.controls, 'Play', () => {
    if (stop3) stop3(); tSel = 1; let acc = 0;
    stop3 = animate((dt) => { acc += dt; if (acc < 260) return true; acc = 0; tS.set(tSel); draw3(); tSel++; return tSel <= 24; });
  }, true);
  const ro3 = stats(f3.root);
  const row3 = el('div', { class: 'fig-row' }); f3.body.append(row3);
  const valveHost = el('div'), timeHost = el('div'); row3.append(valveHost, timeHost);
  legend(timeHost, [['forget f', cat(0)], ['input i', cat(1)], ['output o', cat(2)], ['cell memory c', cat(3)], ['plain RNN h', cat(4)]]);
  const timeSvgHost = el('div'); timeHost.append(timeSvgHost);
  const T3 = 24, r3 = rng(7);
  const S = Array.from({ length: T3 }, (_, t) => (t === 3 ? 1 : r3.normal(0, 0.015)));
  const QC = Array.from({ length: T3 }, (_, t) => (t === 17 ? 1 : 0));
  const RS = Array.from({ length: T3 }, (_, t) => (t === 20 ? 1 : 0));

  function runLSTM() {
    let c = 0, hr = 0; const out = [];
    for (let t = 0; t < T3; t++) {
      const f = sigmoid(bf - 10 * RS[t]), i = sigmoid(-5 + 10 * S[t]), g = Math.tanh(3 * S[t]), o = sigmoid(-4 + 8 * QC[t]);
      const cPrev = c; c = f * c + i * g; const h = o * Math.tanh(c);
      hr = Math.tanh(0.9 * hr + 3 * S[t]);
      out.push({ t: t + 1, f, i, g, o, cPrev, c, h, hr, s: S[t] });
    }
    return out;
  }

  function valve(svg, x, y, v, label) {
    const R = 22;
    svg.append('circle').attr('cx', x).attr('cy', y).attr('r', R).attr('fill', css('--surface')).attr('stroke', ink2()).attr('stroke-width', 1.5);
    // fill from bottom proportional to gate value
    const clipId = 'vclip' + label.replace(/\W/g, '') + Math.round(x);
    svg.append('clipPath').attr('id', clipId).append('circle').attr('cx', x).attr('cy', y).attr('r', R - 2);
    svg.append('rect').attr('x', x - R).attr('y', y + R - 2 * R * v).attr('width', 2 * R).attr('height', 2 * R * v).attr('fill', css('--accent')).attr('opacity', 0.75).attr('clip-path', `url(#${clipId})`);
    // valve handle rotates: horizontal = closed, vertical = open
    const ang = (1 - v) * 90;
    svg.append('line').attr('x1', x).attr('y1', y - R - 4).attr('x2', x).attr('y2', y - R - 16).attr('stroke', ink()).attr('stroke-width', 2.5).attr('transform', `rotate(${ang},${x},${y - R - 4})`);
    svg.append('text').attr('x', x).attr('y', y + 5).attr('text-anchor', 'middle').attr('font-size', 13).attr('font-weight', 700).attr('fill', ink()).text(label);
    svg.append('text').attr('x', x).attr('y', y + R + 16).attr('text-anchor', 'middle').attr('font-size', 12).attr('font-family', css('--mono')).attr('fill', ink()).text(v.toFixed(3));
  }

  function draw3() {
    valveHost.innerHTML = ''; timeSvgHost.innerHTML = '';
    const run = runLSTM(), d = run[tSel - 1];
    // ---- valve diagram ----
    const svg = d3.select(valveHost).append('svg').attr('viewBox', '0 0 520 330').attr('width', 520);
    const pw = (v) => 2 + 10 * Math.min(1, Math.abs(v));
    const yC = 70, yH = 250;
    const pipe = (x1, y1, x2, y2, v) => svg.append('line').attr('x1', x1).attr('y1', y1).attr('x2', x2).attr('y2', y2).attr('stroke', cat(3)).attr('stroke-width', pw(v)).attr('stroke-linecap', 'round').attr('opacity', 0.85);
    const lab = (x, y, t, anchor = 'middle', size = 12, col = ink2()) => svg.append('text').attr('x', x).attr('y', y).attr('text-anchor', anchor).attr('font-size', size).attr('fill', col).text(t);
    lab(10, 20, `Step t = ${tSel}   input sₜ = ${d.s.toFixed(2)}${QC[tSel - 1] ? '   (query cue)' : ''}${RS[tSel - 1] ? '   (reset cue)' : ''}`, 'start', 13, ink());
    // cell-state conveyor
    pipe(20, yC, 118, yC, d.cPrev); pipe(162, yC, 270, yC, d.f * d.cPrev); pipe(270, yC, 500, yC, d.c);
    lab(20, yC - 16, `c(t−1) = ${d.cPrev.toFixed(3)}`, 'start');
    lab(500, yC - 16, `c(t) = ${d.c.toFixed(3)}`, 'end', 12, ink());
    valve(svg, 140, yC, d.f, 'f');
    lab(140, yC + 56, 'forget: keep this', 'middle', 11);
    lab(140, yC + 70, 'fraction of memory', 'middle', 11);
    // input branch
    pipe(270, yC + 110, 270, yC + 22, d.i * d.g);
    svg.append('circle').attr('cx', 270).attr('cy', yC).attr('r', 12).attr('fill', css('--surface')).attr('stroke', ink2());
    lab(270, yC + 5, '+', 'middle', 16, ink());
    valve(svg, 270, yC + 132, d.i, 'i');
    lab(240, yC + 128, `candidate g = ${d.g.toFixed(3)}`, 'end', 11);
    lab(240, yC + 143, `adds i·g = ${(d.i * d.g).toFixed(3)}`, 'end', 11);
    // output branch
    pipe(420, yC, 420, yH - 22, Math.tanh(d.c));
    lab(428, yC + 50, `tanh(c) = ${Math.tanh(d.c).toFixed(3)}`, 'start', 11);
    valve(svg, 420, yH, d.o, 'o');
    pipe(442, yH, 505, yH, d.h);
    lab(505, yH + 28, `h(t) = ${d.h.toFixed(3)}`, 'end', 12, ink());
    lab(400, yH + 54, 'output: reveal memory?', 'middle', 11);
    // ---- time plots ----
    const F = frame(timeSvgHost, { width: 540, height: 300, x: [1, T3], y: [-0.05, 1.05], xLabel: 'time step', yLabel: 'value', xTicks: 12 });
    // input + cues as stems at bottom band
    [[3, 'pulse'], [17, 'query'], [20, 'reset']].forEach(([t, name]) => {
      F.layer('cues').append('line').attr('x1', F.x(t + 1)).attr('x2', F.x(t + 1)).attr('y1', 0).attr('y2', F.h).attr('stroke', css('--grid')).attr('stroke-width', 6).attr('opacity', 0.8);
      F.layer('cues').append('text').attr('x', F.x(t + 1)).attr('y', 10).attr('text-anchor', 'middle').attr('font-size', 11).attr('fill', muted()).text(name);
    });
    const series = [['f', 0], ['i', 1], ['o', 2], ['c', 3], ['hr', 4]];
    const ln = d3.line().x((p) => F.x(p.t)).y((p) => F.y(p.v));
    series.forEach(([k, ci]) => F.layer('l').append('path').datum(run.map((p) => ({ t: p.t, v: p[k] }))).attr('d', ln).attr('fill', 'none').attr('stroke', cat(ci)).attr('stroke-width', k === 'c' ? 3 : 2).attr('stroke-dasharray', k === 'hr' ? '5 3' : null));
    F.layer('sel').append('line').attr('x1', F.x(tSel)).attr('x2', F.x(tSel)).attr('y1', 0).attr('y2', F.h).attr('stroke', ink()).attr('stroke-width', 1);
    F.layer('sel').selectAll('circle').data(series).join('circle').attr('cx', F.x(tSel)).attr('cy', ([k]) => F.y(d[k])).attr('r', 4.5).attr('fill', ([, ci]) => cat(ci)).attr('stroke', css('--surface')).attr('stroke-width', 1.2);
    F.svg.on('mousemove', (e) => { const [mx] = d3.pointer(e, F.g.node()); const t = Math.round(F.x.invert(mx)); if (t < 1 || t > T3) return; const p = run[t - 1]; tip(e, `t=${t}<br>f=${p.f.toFixed(3)} i=${p.i.toFixed(3)} o=${p.o.toFixed(3)}<br>c=${p.c.toFixed(3)} h=${p.h.toFixed(3)}<br>plain RNN h=${p.hr.toFixed(3)}`); }).on('mouseleave', untip);
    const q = run[17];
    ro3.set({ 'LSTM answer at query (h, t=18)': q.h.toFixed(3), 'memory c at t=18': q.c.toFixed(3), 'plain RNN state at t=18': q.hr.toFixed(3), 'memory after reset (t=22)': run[21].c.toFixed(3) });
  }
  draw3();

  section(main, 'What to notice', `<ul>
    <li>In the unrolled view, slide <b>Steps read</b> back and forth: each new token rewrites every hidden unit. Earlier tokens only survive as whatever trace the rewrites leave behind.</li>
    <li>In the memory figure at the default <b>ρ = 0.9</b>, the influence of step 1 falls below 1% within a handful of steps and keeps falling exponentially. That is the <b>vanishing gradient</b>: the network cannot learn that step 1 mattered for an error at step 30, because almost no gradient reaches it.</li>
    <li>Push <b>ρ above 1</b>. The linear reference explodes (exploding gradients), but the tanh network does not follow it: units saturate near ±1, where tanh is flat, and the flat slope caps the growth. Around ρ = 1.3 memory lasts longer (14 steps to 1%) but still decays. At ρ = 2 the curve stops decaying and wobbles around 0.1: the network has become <b>chaotic</b>. Step 1 still changes the state at step 40, but in a scrambled way that is useless as memory and makes training unstable. A plain RNN has no setting that gives long, clean memory.</li>
    <li>In the LSTM, the default forget bias 5 gives f ≈ 0.993 per step, so 14 steps after the pulse the memory c is still about 0.9 and the query reads it out. The plain RNN's state has already fallen to about 0.2 and keeps shrinking geometrically.</li>
    <li>Lower the <b>forget-gate bias</b> to 1 (f ≈ 0.73): the memory is gone (c ≈ 0.01) long before the query. Memory is not magic; it is a learned forget gate close to 1. This is why LSTM implementations initialise the forget bias to 1 or more, so that the network starts out remembering.</li>
    <li>Watch the <b>output valve</b>: the memory is held in c the whole time, but h (what the rest of the network sees) stays near zero until the query opens o. Then the reset cue closes the forget valve and the memory is wiped in one step.</li>
  </ul>`);

  steps(main, 'How it works', [
    'Start with a hidden state <code>h₀ = 0</code>.',
    'At each step, combine the previous state and the new input: <code>hₜ = tanh(W·hₜ₋₁ + U·xₜ + b)</code>. The <b>same</b> W, U and b are used at every step (weight sharing, like a convolution sliding over time).',
    'Read predictions off the states: one per step (e.g. label each base) or from the last state (classify the whole sequence).',
    'Train with <b>backpropagation through time</b>: unroll the loop, backpropagate through all copies, and sum the gradients for the shared weights. The gradient for step 1 passes through every Jacobian in between, which is the product the memory figure plots.',
    'An <b>LSTM</b> adds a separate cell state <code>c</code> that is updated additively: <code>cₜ = f·cₜ₋₁ + i·g</code>. Three sigmoid gates, each a learned function of the input and previous state, decide what to forget (f), what to write (i) and what to reveal (o). When f ≈ 1 the gradient flows back along c almost unchanged. A <b>GRU</b> is a slimmer variant with two gates.',
  ]);
  math(main, 'RNN:   hₜ = tanh(W hₜ₋₁ + U xₜ + b)\n       ∂hₜ/∂x₁ = Dₜ W · Dₜ₋₁ W ··· D₂ W · D₁ U      (D = diag(1 − h²), the tanh slope)\n       ‖∂hₜ/∂x₁‖ shrinks ≈ like ρᵗ when ρ·slope < 1\nLSTM:  f, i, o = σ(·),  g = tanh(·)\n       cₜ = f ⊙ cₜ₋₁ + i ⊙ g        hₜ = o ⊙ tanh(cₜ)\n       ∂cₜ/∂cₜ₋₁ = f   (no squashing on the memory path)');

  callout(main, '', 'In biology', `<ul>
    <li><b>DNA and protein sequences:</b> before transformers, models such as DanQ (a CNN followed by a bidirectional LSTM) predicted chromatin accessibility and TF binding from sequence, with the LSTM capturing the arrangement of motifs that the CNN found. Bidirectional RNNs read the sequence both ways, which fits DNA, where there is no natural "past".</li>
    <li><b>Time courses:</b> RNNs and GRUs model longitudinal patient data (lab values, EHR visits) and dense time series such as live-cell imaging tracks. With the 4–10 time points of a typical omics time course, though, an RNN has far too many parameters; splines or ODE models are usually the better tool.</li>
    <li><b>Why the field moved on:</b> the vanishing-memory curve is the reason. A promoter's effect can depend on an enhancer tens of kilobases away, and a gene sentence in Geneformer or scGPT has thousands of tokens. <a href="#/concepts/attention">Attention</a> connects any two positions in one step instead of t steps, and processes all positions in parallel.</li>
  </ul>`);
  callout(main, 'warn', 'Pitfalls', `<ul>
    <li><b>Exploding gradients</b> cause sudden NaN losses. Clip the gradient norm (e.g. to 1.0).</li>
    <li>A plain RNN cannot learn dependencies longer than a few tens of steps, however long you train. Use an LSTM or GRU, or a transformer.</li>
    <li>RNNs are sequential: step t needs step t−1, so they cannot be parallelised across time on a GPU and are slow on long sequences.</li>
    <li>Irregularly sampled time points (common in clinical data) break the "one step = one tick" assumption. Add the time gap as an input, or use a continuous-time model.</li>
  </ul>`);
  seeAlso(main, [['concepts', 'backprop', 'Backpropagation'], ['concepts', 'attention', 'Attention'], ['concepts', 'transformer', 'Transformers'], ['concepts', 'cnn', 'Convolutions (CNNs)'], ['frontier', 'sequence-to-expression', 'Sequence → expression']]);
}
