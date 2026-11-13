// Backpropagation: forward pass fills values, backward pass carries dL/d(node) back through a computation graph,
// one chain-rule product at a time. Part 1: L = (w·x + b − y)². Part 2: a 2-2-1 sigmoid network.
import { header, figure, slider, select, button, stats, section, steps, callout, math, seeAlso, animate, legend, tip, untip, el } from '../lib/ui.js';
import { frame, cat, css, ink, ink2, muted, diverging } from '../lib/plot.js';
import { sigmoid } from '../lib/data.js';
const d3 = window.d3;

const f2 = (v) => (Math.abs(v) < 5e-3 ? 0 : v).toFixed(2).replace('-', '−');
const f3 = (v) => (Math.abs(v) < 5e-4 ? 0 : v).toFixed(3).replace('-', '−');

/** Tiny step player: calls show(stage) as stage advances from 0..max. Returns controls. */
function player(max, show, ms = 750) {
  let stage = max, stop = null;
  const halt = () => { if (stop) { stop(); stop = null; } };
  return {
    get stage() { return stage; },
    set(s) { halt(); stage = s; show(stage, false); },
    step() { halt(); stage = stage >= max ? 0 : stage + 1; show(stage, true); },
    play() {
      halt(); stage = 0; show(stage, true); let acc = 0;
      stop = animate((dt) => { acc += dt; if (acc < ms) return true; acc = 0; stage++; show(stage, true); return stage < max; });
    },
    halt,
  };
}

export default async function render(main) {
  header(main, 'Backpropagation',
    'A network is a chain of simple operations. Backpropagation computes how the loss changes with every parameter by applying the <b>chain rule backwards</b>: start from the loss, and at each node multiply the gradient arriving from above by that node\'s own local derivative.');

  /* ======================= Figure 1: scalar computation graph ======================= */
  const fig = figure(main, 'One neuron, one example: L = (w·x + b − y)²',
    'Each box shows the node\'s <b>value</b> (forward pass) and its <b>gradient ∂L/∂node</b> (backward pass). Edge labels are the <b>local derivatives</b>; the gradient reaching a node is the gradient of the node above it times that local derivative. With the starting values (w=0.5, x=2, b=0.5, y=3) the prediction is 1.5, the loss is 2.25, and ∂L/∂w = −6: increasing w would reduce the loss.');
  const P = { w: 0.5, x: 2, b: 0.5, y: 3 };
  const sl = {};
  [['w', 'weight w', -2, 2], ['x', 'input x', -3, 3], ['b', 'bias b', -2, 2], ['y', 'target y', -3, 3]].forEach(([k, label, lo, hi]) => {
    sl[k] = slider(fig.controls, { label, min: lo, max: hi, step: 0.1, value: P[k], fmt: (v) => f2(v), onInput: (v) => { P[k] = v; update(); } });
  });
  button(fig.controls, 'Play forward + backward', () => pl1.play(), true);
  button(fig.controls, 'Step', () => pl1.step());
  button(fig.controls, 'Gradient step (η = 0.05)', () => {
    const G = compute1();
    P.w = Math.max(-2, Math.min(2, P.w - 0.05 * G.g.w)); P.b = Math.max(-2, Math.min(2, P.b - 0.05 * G.g.b));
    sl.w.set(+P.w.toFixed(2)); sl.b.set(+P.b.toFixed(2)); P.w = +P.w.toFixed(2); P.b = +P.b.toFixed(2); update();
  });
  const read1 = stats(fig.root);

  // node definitions: parents, forward fn, local derivative wrt each parent [text, fn]
  const NODES = [
    { id: 'w', name: 'w', leaf: true, pos: [70, 48] },
    { id: 'x', name: 'x', leaf: true, pos: [70, 128] },
    { id: 'b', name: 'b', leaf: true, pos: [70, 208] },
    { id: 'y', name: 'y', leaf: true, pos: [70, 288] },
    { id: 'm', name: 'm = w·x', par: ['w', 'x'], f: (v) => v.w * v.x, d: { w: ['x', (v) => v.x], x: ['w', (v) => v.w] }, pos: [325, 88], fs: 1, op: '×' },
    { id: 'z', name: 'ŷ = m + b', par: ['m', 'b'], f: (v) => v.m + v.b, d: { m: ['1', () => 1], b: ['1', () => 1] }, pos: [510, 150], fs: 2, op: '+' },
    { id: 'e', name: 'e = ŷ − y', par: ['z', 'y'], f: (v) => v.z - v.y, d: { z: ['1', () => 1], y: ['−1', () => -1] }, pos: [720, 218], fs: 3, op: '−' },
    { id: 'L', name: 'L = e²', par: ['e'], f: (v) => v.e * v.e, d: { e: ['2e', (v) => 2 * v.e] }, pos: [890, 218], fs: 4, op: '²' },
  ];
  // backward stage of each node (when its gradient becomes known)
  const BST = { L: 5, e: 6, z: 7, y: 7, m: 8, b: 8, w: 9, x: 9 };
  const byId = Object.fromEntries(NODES.map((n) => [n.id, n]));
  const child = {}; NODES.forEach((n) => (n.par || []).forEach((p) => { child[p] = n.id; }));
  const MAX1 = 9;

  function compute1(p = P) {
    const v = { ...p };
    NODES.forEach((n) => { if (!n.leaf) v[n.id] = n.f(v); });
    const g = { L: 1 };
    [...NODES].reverse().forEach((n) => { if (n.leaf || g[n.id] == null) return; n.par.forEach((q) => { g[q] = (g[q] || 0) + g[n.id] * n.d[q][1](v); }); });
    return { v, g };
  }

  const row1 = el('div', { class: 'fig-row' });
  fig.body.append(row1);
  const gWrap = el('div', { style: 'flex: 1 1 100%' }), ledWrap = el('div', { style: 'flex: 1 1 100%' });
  row1.append(gWrap, ledWrap);
  const W1 = 960, H1 = 320, BW = 112, BH = 62;
  const svg1 = d3.select(gWrap).append('svg').attr('viewBox', `0 0 ${W1} ${H1}`).attr('width', W1).attr('role', 'img')
    .attr('aria-label', 'Computation graph for squared error of a linear neuron');
  const defs = svg1.append('defs');
  defs.append('marker').attr('id', 'bp-arrow').attr('viewBox', '0 0 10 10').attr('refX', 9).attr('refY', 5).attr('markerWidth', 7).attr('markerHeight', 7).attr('orient', 'auto')
    .append('path').attr('d', 'M0,0 L10,5 L0,10 z').attr('fill', css('--axis'));
  const edgeL = svg1.append('g'), nodeL = svg1.append('g'), flowL = svg1.append('g');
  const ledger1 = el('div', { style: 'font-size:13.5px; columns: 2 300px; column-gap: 28px' });
  ledWrap.append(el('div', { class: 'fig-title', style: 'font-size:14px' }, 'Chain-rule ledger'), ledger1);

  const edges1 = [];
  NODES.forEach((n) => (n.par || []).forEach((p) => edges1.push({ from: p, to: n.id })));
  const anchor = (id, side) => { const [cx, cy] = byId[id].pos; return side === 'r' ? [cx + BW / 2, cy] : [cx - BW / 2, cy]; };

  function show1(stage, live) {
    const { v, g } = compute1();
    // edges
    edgeL.selectAll('*').remove();
    edges1.forEach((e) => {
      const [x1, y1] = anchor(e.from, 'r'), [x2, y2] = anchor(e.to, 'l');
      const fwdOn = stage >= byId[e.to].fs, bwdOn = stage >= BST[e.from];
      const mx = (x1 + x2) / 2;
      edgeL.append('path').attr('d', `M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2 - 2},${y2}`).attr('fill', 'none')
        .attr('stroke', bwdOn ? cat(1) : fwdOn ? ink2() : css('--axis')).attr('stroke-width', bwdOn ? 2.5 : 1.5)
        .attr('marker-end', 'url(#bp-arrow)');
      if (bwdOn) {
        const [txt, fn] = byId[e.to].d[e.from];
        const label = `∂${short(e.to)}/∂${short(e.from)} = ${txt}${txt === '1' || txt === '−1' ? '' : ' = ' + f2(fn(v))}`;
        const lx = mx, ly = e.to === 'L' ? y1 - 42 : (y1 + y2) / 2 - 8;
        const t = edgeL.append('text').attr('x', lx).attr('y', ly).attr('text-anchor', 'middle').attr('font-size', 11.5).attr('fill', ink()).text(label);
        const bb = t.node().getBBox();
        edgeL.insert('rect', () => t.node()).attr('x', bb.x - 3).attr('y', bb.y - 1).attr('width', bb.width + 6).attr('height', bb.height + 2).attr('rx', 3)
          .attr('fill', css('--surface')).attr('opacity', 0.92);
      }
    });
    // nodes
    nodeL.selectAll('*').remove();
    NODES.forEach((n) => {
      const [cx, cy] = n.pos, gN = nodeL.append('g').attr('transform', `translate(${cx - BW / 2},${cy - BH / 2})`);
      const hasV = n.leaf || stage >= n.fs, hasG = stage >= BST[n.id];
      gN.append('rect').attr('width', BW).attr('height', BH).attr('rx', 8).attr('fill', hasV ? css('--surface-2') : css('--surface'))
        .attr('stroke', hasG ? cat(1) : css('--axis')).attr('stroke-width', hasG ? 2 : 1);
      gN.append('text').attr('x', 8).attr('y', 17).attr('font-size', 12.5).attr('font-weight', 600).attr('fill', ink()).text(n.name);
      gN.append('text').attr('x', 8).attr('y', 36).attr('font-size', 12).attr('fill', hasV ? ink() : muted()).text(hasV ? `value ${f2(v[n.id])}` : 'value ?');
      gN.append('text').attr('x', 8).attr('y', 53).attr('font-size', 12).attr('fill', hasG ? ink() : muted()).text(hasG ? `∂L/∂${short(n.id)} ${f2(g[n.id])}` : `∂L/∂${short(n.id)} ?`);
      if (hasG) gN.append('rect').attr('x', BW - 9).attr('y', 6).attr('width', 4).attr('height', BH - 12).attr('rx', 2).attr('fill', cat(1));
      gN.on('mousemove', (evt) => tip(evt, `<b>${n.name}</b><br>value = ${f3(v[n.id])}<br>∂L/∂${short(n.id)} = ${f3(g[n.id])}`)).on('mouseleave', untip);
    });
    // animated pulse travelling backwards along the edges that just got their gradient
    flowL.selectAll('*').remove();
    if (live && stage > 5) {
      const active = edges1.filter((e) => BST[e.from] === stage);
      active.forEach((e) => {
        const [x1, y1] = anchor(e.from, 'r'), [x2, y2] = anchor(e.to, 'l');
        flowL.append('circle').attr('r', 6).attr('fill', cat(1)).attr('stroke', css('--surface')).attr('stroke-width', 1.5)
          .attr('cx', x2).attr('cy', y2).transition().duration(550).ease(d3.easeCubicOut).attr('cx', x1).attr('cy', y1).remove();
      });
    }
    // ledger
    const lines = [];
    const fwd = [['m', `m = w·x = ${f2(v.w)}·${f2(v.x)} = ${f2(v.m)}`], ['z', `ŷ = m + b = ${f2(v.m)} + ${f2(v.b)} = ${f2(v.z)}`], ['e', `e = ŷ − y = ${f2(v.z)} − ${f2(v.y)} = ${f2(v.e)}`], ['L', `L = e² = <b>${f2(v.L)}</b>`]];
    fwd.forEach(([id, s]) => { if (stage >= byId[id].fs) lines.push(`<span style="color:${ink2()}">fwd</span> ${s}`); });
    if (stage >= 5) lines.push('<span style="color:' + ink2() + '">bwd</span> ∂L/∂L = 1');
    const bw = [['e', 'L', '2e'], ['z', 'e', '1'], ['y', 'e', '−1'], ['m', 'z', '1'], ['b', 'z', '1'], ['w', 'm', 'x'], ['x', 'm', 'w']];
    bw.forEach(([p, c, txt]) => {
      if (stage < BST[p]) return;
      const loc = byId[c].d[p][1](v);
      lines.push(`<span style="color:${ink2()}">bwd</span> ∂L/∂${short(p)} = ∂L/∂${short(c)} · ${txt} = ${f2(g[c])} · ${f2(loc)} = <b>${f2(g[p])}</b>`);
    });
    ledger1.innerHTML = lines.length ? lines.map((s) => `<div style="padding:2px 0;border-bottom:1px solid var(--border);break-inside:avoid">${s}</div>`).join('') : `<p style="color:${muted()}">Press <b>Play</b> or <b>Step</b>.</p>`;
    // readouts incl. a finite-difference check
    const h = 1e-4, Lp = compute1({ ...P, w: P.w + h }).v.L, Lm = compute1({ ...P, w: P.w - h }).v.L;
    read1.set({ 'prediction ŷ': f2(v.z), 'loss L': f3(v.L), '∂L/∂w (backprop)': f3(g.w), '∂L/∂w (finite difference)': f3((Lp - Lm) / (2 * h)), '∂L/∂b': f3(g.b) });
  }
  const short = (id) => ({ z: 'ŷ' }[id] || id);
  const pl1 = player(MAX1, show1, 800);
  function update() { pl1.set(MAX1); }
  update();

  /* ======================= Figure 2: a 2-2-1 network ======================= */
  const fig2 = figure(main, 'The same idea in a 2-2-1 network',
    'Two inputs, two sigmoid hidden units, one sigmoid output, binary cross-entropy loss. In the backward pass each neuron gets an <b>error signal δ</b> (∂L/∂pre-activation). A weight\'s gradient is δ of the neuron it feeds into times the activation flowing along it. Edge colour shows the sign of each weight\'s gradient (blue negative, red positive) and thickness its size; hover an edge for its numbers.');
  const X = { x1: 1.0, x2: 0.5, t: 1 };
  const NET0 = { W: [[0.8, -0.5], [0.3, 0.9]], b: [0.1, -0.2], v: [1.2, -0.7], c: 0.05 };
  let net = clone(NET0), lossHist = [];
  const s2 = {};
  s2.x1 = slider(fig2.controls, { label: 'x₁', min: -2, max: 2, step: 0.1, value: X.x1, fmt: f2, onInput: (v) => { X.x1 = v; lossHist = []; update2(); } });
  s2.x2 = slider(fig2.controls, { label: 'x₂', min: -2, max: 2, step: 0.1, value: X.x2, fmt: f2, onInput: (v) => { X.x2 = v; lossHist = []; update2(); } });
  select(fig2.controls, { label: 'target t', value: '1', options: [['1', '1'], ['0', '0']], onChange: (v) => { X.t = +v; lossHist = []; update2(); } });
  button(fig2.controls, 'Play forward + backward', () => pl2.play(), true);
  button(fig2.controls, 'Step', () => pl2.step());
  button(fig2.controls, 'Train 1 step (η = 1)', () => train(1));
  button(fig2.controls, 'Train 20 steps', () => train(20));
  button(fig2.controls, 'Reset weights', () => { net = clone(NET0); lossHist = []; update2(); });
  const read2 = stats(fig2.root);

  function fb(n = net) {
    const x = [X.x1, X.x2];
    const a = [0, 1].map((j) => n.W[j][0] * x[0] + n.W[j][1] * x[1] + n.b[j]);
    const h = a.map(sigmoid);
    const o = n.v[0] * h[0] + n.v[1] * h[1] + n.c, yhat = sigmoid(o);
    const eps = 1e-12, L = -(X.t * Math.log(yhat + eps) + (1 - X.t) * Math.log(1 - yhat + eps));
    const dOut = yhat - X.t;
    const dv = [dOut * h[0], dOut * h[1]], dc = dOut;
    const dH = [0, 1].map((j) => dOut * n.v[j] * h[j] * (1 - h[j]));
    const dW = [0, 1].map((j) => [dH[j] * x[0], dH[j] * x[1]]), db = dH.slice();
    return { x, a, h, o, yhat, L, dOut, dv, dc, dH, dW, db };
  }
  function train(k) {
    pl2.halt();
    for (let i = 0; i < k; i++) {
      const r = fb(); if (!lossHist.length) lossHist.push(r.L);
      net.v = net.v.map((v, j) => v - r.dv[j]); net.c -= r.dc;
      net.W = net.W.map((row, j) => row.map((w, i2) => w - r.dW[j][i2])); net.b = net.b.map((b, j) => b - r.db[j]);
      lossHist.push(fb().L);
    }
    update2();
  }

  const row2 = el('div', { class: 'fig-row' });
  fig2.body.append(row2);
  const nWrap = el('div', { style: 'flex: 2 1 480px' }), rWrap = el('div', { style: 'flex: 1 1 300px' });
  row2.append(nWrap, rWrap);
  const W2 = 640, H2 = 330;
  const svg2 = d3.select(nWrap).append('svg').attr('viewBox', `0 0 ${W2} ${H2}`).attr('width', W2).attr('role', 'img').attr('aria-label', 'A 2-2-1 neural network');
  const e2L = svg2.append('g'), n2L = svg2.append('g');
  legend(nWrap, [['∂L/∂w < 0 (increase w to lower L)', diverging(1)(-1)], ['∂L/∂w > 0', diverging(1)(1)]]);
  const ledger2 = el('div', { class: 'prose', style: 'font-size:13.5px' });
  rWrap.append(el('div', { class: 'fig-title', style: 'font-size:14px' }, 'Chain-rule ledger'), ledger2);
  const lossDiv = el('div');
  rWrap.append(lossDiv);

  const NP = { x1: [70, 95], x2: [70, 245], h1: [320, 95], h2: [320, 245], o: [550, 170] };
  const R = 34;
  // stages: 0 inputs; 1 hidden; 2 output; 3 loss; 4 δ_out + grads of v,c; 5 δ_hidden; 6 grads of W
  const MAX2 = 6;
  function show2(stage) {
    const r = fb();
    const gmax = Math.max(1e-6, ...r.dW.flat().map(Math.abs), ...r.dv.map(Math.abs));
    const div = diverging(gmax);
    const E = [];
    [0, 1].forEach((j) => [0, 1].forEach((i) => E.push({ from: 'x' + (i + 1), to: 'h' + (j + 1), w: net.W[j][i], g: r.dW[j][i], name: `w${j + 1}${i + 1}`, bs: 6, fs: 1, act: r.x[i], actName: `x${i + 1}`, delta: r.dH[j], dName: `δh${j + 1}` })));
    [0, 1].forEach((j) => E.push({ from: 'h' + (j + 1), to: 'o', w: net.v[j], g: r.dv[j], name: `v${j + 1}`, bs: 4, fs: 2, act: r.h[j], actName: `h${j + 1}`, delta: r.dOut, dName: 'δout' }));
    e2L.selectAll('*').remove();
    E.forEach((e) => {
      const [x1, y1] = NP[e.from], [x2, y2] = NP[e.to];
      const on = stage >= e.bs;
      e2L.append('line').attr('x1', x1 + R).attr('y1', y1).attr('x2', x2 - R).attr('y2', y2)
        .attr('stroke', on ? div(e.g) : stage >= e.fs ? ink2() : css('--axis'))
        .attr('stroke-width', on ? 2 + 7 * Math.abs(e.g) / gmax : 1.6).attr('stroke-linecap', 'round');
      // weight label near the source end
      const fr = e.to === 'o' ? 0.5 : (y1 === y2 ? 0.5 : 0.24);
      const tx = x1 + R + (x2 - x1 - 2 * R) * fr, ty = y1 + (y2 - y1) * fr - 7;
      const t = e2L.append('text').attr('x', tx).attr('y', ty).attr('text-anchor', 'middle').attr('font-size', 11.5).attr('fill', ink())
        .text(`${e.name}=${f2(e.w)}` + (on ? `  ∂=${f2(e.g)}` : ''));
      const bb = t.node().getBBox();
      e2L.insert('rect', () => t.node()).attr('x', bb.x - 3).attr('y', bb.y - 1).attr('width', bb.width + 6).attr('height', bb.height + 2).attr('rx', 3).attr('fill', css('--surface')).attr('opacity', 0.9);
      // fat invisible hit line for tooltips
      e2L.append('line').attr('x1', x1 + R).attr('y1', y1).attr('x2', x2 - R).attr('y2', y2).attr('stroke', 'transparent').attr('stroke-width', 14)
        .on('mousemove', (evt) => tip(evt, `<b>${e.name}</b> = ${f3(e.w)}<br>∂L/∂${e.name} = ${e.dName} · ${e.actName} = ${f3(e.delta)} · ${f3(e.act)} = <b>${f3(e.g)}</b>`)).on('mouseleave', untip);
    });
    n2L.selectAll('*').remove();
    const neuron = (id, label, val, has, delta, hasD, sub) => {
      const [cx, cy] = NP[id], g = n2L.append('g');
      g.append('circle').attr('cx', cx).attr('cy', cy).attr('r', R).attr('fill', has ? css('--surface-2') : css('--surface'))
        .attr('stroke', hasD ? cat(1) : css('--axis')).attr('stroke-width', hasD ? 2.2 : 1.2);
      g.append('text').attr('x', cx).attr('y', cy - 6).attr('text-anchor', 'middle').attr('font-size', 12).attr('fill', ink2()).text(label);
      g.append('text').attr('x', cx).attr('y', cy + 12).attr('text-anchor', 'middle').attr('font-size', 14).attr('font-weight', 600).attr('fill', has ? ink() : muted()).text(has ? f2(val) : '?');
      if (sub) g.append('text').attr('x', cx).attr('y', cy + R + 16).attr('text-anchor', 'middle').attr('font-size', 11.5).attr('fill', ink2()).text(sub);
      if (hasD) g.append('text').attr('x', cx).attr('y', cy - R - 8).attr('text-anchor', 'middle').attr('font-size', 12).attr('fill', ink()).text(`δ = ${f3(delta)}`);
    };
    neuron('x1', 'x₁', r.x[0], true, 0, false);
    neuron('x2', 'x₂', r.x[1], true, 0, false);
    neuron('h1', 'h₁', r.h[0], stage >= 1, r.dH[0], stage >= 5, `b₁=${f2(net.b[0])}`);
    neuron('h2', 'h₂', r.h[1], stage >= 1, r.dH[1], stage >= 5, `b₂=${f2(net.b[1])}`);
    neuron('o', 'ŷ', r.yhat, stage >= 2, r.dOut, stage >= 4, `c=${f2(net.c)}`);
    if (stage >= 3) {
      n2L.append('text').attr('x', NP.o[0]).attr('y', NP.o[1] + R + 36).attr('text-anchor', 'middle').attr('font-size', 12.5).attr('fill', ink())
        .text(`L = ${f3(r.L)} (t = ${X.t})`);
    }
    // ledger
    const lines = [];
    if (stage >= 1) lines.push(`a₁ = ${f2(net.W[0][0])}·${f2(r.x[0])} + ${f2(net.W[0][1])}·${f2(r.x[1])} + ${f2(net.b[0])} = ${f2(r.a[0])}, h₁ = σ(a₁) = ${f3(r.h[0])}`, `a₂ = ${f2(r.a[1])}, h₂ = ${f3(r.h[1])}`);
    if (stage >= 2) lines.push(`o = ${f2(net.v[0])}·${f3(r.h[0])} + ${f2(net.v[1])}·${f3(r.h[1])} + ${f2(net.c)} = ${f3(r.o)}, ŷ = ${f3(r.yhat)}`);
    if (stage >= 3) lines.push(`L = −[t·log ŷ + (1−t)·log(1−ŷ)] = <b>${f3(r.L)}</b>`);
    if (stage >= 4) lines.push(`δout = ∂L/∂o = ŷ − t = <b>${f3(r.dOut)}</b> (sigmoid + cross-entropy simplify to this)`, `∂L/∂v₁ = δout·h₁ = ${f3(r.dv[0])}, ∂L/∂v₂ = ${f3(r.dv[1])}, ∂L/∂c = ${f3(r.dc)}`);
    if (stage >= 5) lines.push(`δh₁ = δout · v₁ · h₁(1−h₁) = ${f3(r.dOut)}·${f2(net.v[0])}·${f3(r.h[0] * (1 - r.h[0]))} = <b>${f3(r.dH[0])}</b>`, `δh₂ = δout · v₂ · h₂(1−h₂) = <b>${f3(r.dH[1])}</b>`);
    if (stage >= 6) lines.push(`∂L/∂w₁₁ = δh₁·x₁ = ${f3(r.dW[0][0])}, ∂L/∂w₁₂ = ${f3(r.dW[0][1])}`, `∂L/∂w₂₁ = δh₂·x₁ = ${f3(r.dW[1][0])}, ∂L/∂w₂₂ = ${f3(r.dW[1][1])}`);
    ledger2.innerHTML = lines.length ? lines.map((s) => `<div style="padding:2px 0;border-bottom:1px solid var(--border)">${s}</div>`).join('') : `<p style="color:${muted()}">Press <b>Play</b> or <b>Step</b>.</p>`;
    // loss curve
    lossDiv.innerHTML = '';
    if (lossHist.length > 1) {
      const F = frame(lossDiv, { width: 340, height: 170, x: [0, Math.max(10, lossHist.length - 1)], y: [0, Math.max(0.1, d3.max(lossHist) * 1.1)], xLabel: 'training step', yLabel: 'loss', yTicks: 3 });
      F.layer('c').append('path').datum(lossHist).attr('fill', 'none').attr('stroke', cat(0)).attr('stroke-width', 2).attr('d', d3.line().x((d, i) => F.x(i)).y((d) => F.y(d)));
    }
    read2.set({ 'prediction ŷ': f3(r.yhat), 'loss L': f3(r.L), 'δout': f3(r.dOut), 'training steps': Math.max(0, lossHist.length - 1) });
  }
  const pl2 = player(MAX2, show2, 900);
  function update2() { pl2.set(MAX2); }
  update2();

  section(main, 'What to notice', `<ul>
    <li>In the first graph every gradient on the way back is a <b>running product</b>: ∂L/∂w = 2e · 1 · 1 · x. The upstream part (2e) is shared by every parameter; only the last local factor differs. Backprop computes the shared part once, which is why it costs about as much as one forward pass.</li>
    <li>Set <b>x = 0</b>: ∂L/∂w becomes 0 however wrong the prediction is. A weight only learns when its input is active. This is the same reason dead ReLUs stop learning.</li>
    <li>Set <b>y</b> equal to the prediction: e = 0, so every gradient is 0. No error, no update.</li>
    <li>Click <b>Gradient step</b> a few times. w and b move opposite to their gradients and the loss falls. The finite-difference readout matches backprop to three decimals, a quick sanity check that the chain rule was applied correctly.</li>
    <li>In the network, the hidden δ is multiplied by <b>h(1−h)</b>, the sigmoid's slope, which is never larger than 0.25. Stack many sigmoid layers and these factors multiply towards zero: the <a href="#/concepts/activations">vanishing gradient</a>.</li>
    <li>With the default inputs and t = 1, v₁ gets a negative gradient (raising it raises ŷ towards 1), while v₂ also has a negative gradient that pushes the negative weight back towards zero. Press <b>Train 20 steps</b> and watch the loss curve fall.</li>
  </ul>`);

  steps(main, 'How it works', [
    '<b>Forward pass.</b> Feed the input through the graph, computing and <b>storing</b> every intermediate value (these are needed for the local derivatives).',
    'Compute the loss L. Its gradient with respect to itself is 1. That is the seed of the backward pass.',
    '<b>Backward pass.</b> Visit nodes in reverse order. For each node, multiply the gradient arriving from its output by the local derivative with respect to each input, and pass the product back.',
    'If a value feeds several nodes, <b>add</b> the gradients coming back from each of them (the multivariable chain rule).',
    'When you reach the parameters, you have ∂L/∂θ for all of them. Hand these to <a href="#/concepts/gradient-descent">gradient descent</a> or Adam to update θ.',
    'Frameworks like PyTorch and JAX do exactly this automatically ("reverse-mode automatic differentiation"): each operation records its local derivative rule during the forward pass.',
  ]);
  math(main, 'chain rule:   ∂L/∂w = ∂L/∂e · ∂e/∂ŷ · ∂ŷ/∂m · ∂m/∂w = 2e · 1 · 1 · x\n' +
    'layer form:   δ_out = ŷ − t   (sigmoid + cross-entropy)\n' +
    '              δ_hidden = (Wᵀ δ_next) ⊙ σ\'(a),   ∂L/∂W = δ · (input activation)ᵀ');

  callout(main, '', 'In biology', 'Every deep model in single-cell and spatial omics is trained this way. <b>scVI</b> backpropagates a negative-binomial reconstruction loss through its decoder and encoder; <b>Geneformer</b> and <b>scGPT</b> backpropagate a masked-gene prediction loss through dozens of transformer layers; a histology CNN predicting spatial expression from H&E backpropagates a regression loss to its convolution kernels. The same chain-rule products also power <b>gradient-based attributions</b> ("saliency", integrated gradients), which ask which input genes or pixels the output is most sensitive to.');
  callout(main, 'warn', 'Pitfalls', `<ul>
    <li><b>Vanishing / exploding gradients:</b> long products of local derivatives shrink or blow up. ReLU-family activations, careful initialisation, normalisation layers and residual connections are the standard fixes.</li>
    <li><b>Forgetting to zero gradients</b> between steps in PyTorch accumulates them (step 4 above, applied across batches by accident).</li>
    <li>Backprop gives the gradient of the loss <i>on this batch</i>. It says nothing about whether the model generalises.</li>
    <li>Gradient-based attributions are local: a large ∂output/∂gene says the output is sensitive to small changes in that gene, not that the gene is causal.</li>
  </ul>`);
  seeAlso(main, [['concepts', 'gradient-descent', 'Gradient descent'], ['concepts', 'activations', 'Activation functions'], ['concepts', 'neural-network', 'Neural network playground'], ['concepts', 'optimizers', 'Optimizers race']]);
}

function clone(n) { return { W: n.W.map((r) => r.slice()), b: n.b.slice(), v: n.v.slice(), c: n.c }; }
