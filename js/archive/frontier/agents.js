// Agents & AI co-scientists: a scripted, deterministic replay of a Biomni-style plan -> tool call -> observe -> revise loop
// over a mock cohort (no LLM is called), plus an Elo tournament of hypotheses judged by a simulated, possibly biased judge.
import { header, figure, select, button, checkbox, stats, section, steps, math, legend, seeAlso, every, tip, untip, el, html } from '../../lib/ui.js';
import { frame, cat, css, ink, ink2, muted } from '../../lib/plot.js';
import { rng, range } from '../../lib/data.js';
import { sourceCard, toyBadge, coreIdea, realityCheck, useIt, cite } from '../../frontier/_shared.js';
const d3 = window.d3;

// ---------------------------------------------------------------- mock cohort + real (mock-data) tools
const GENES = ['CXCL9', 'GZMB', 'IFNG', 'CD8A', 'PDCD1', 'TGFB1', 'MKI67', 'VEGFA', 'KRT5', 'SOX2', 'MYC', 'EGFR'];
const EFFECT = { CXCL9: 1.1, GZMB: 1.0, IFNG: 0.9, CD8A: 0.5, PDCD1: 0.4 };   // planted truth: an IFN-γ / cytotoxic program
const N = 40, NP = 2000, SEED = 43;
const REGISTRY = ['load_cohort', 'de_test', 'de_all', 'gene_set_score', 'lookup_gene'];
const MECH = { MKI67: 'proliferation', MYC: 'MYC-driven', EGFR: 'EGFR-signalling', SOX2: 'squamous-lineage', VEGFA: 'angiogenic', TGFB1: 'T-cell-exclusion', KRT5: 'squamous-differentiation' };
const KB = { MKI67: 'proliferation marker', MYC: 'oncogenic transcription factor', EGFR: 'receptor tyrosine kinase', SOX2: 'squamous lineage factor', VEGFA: 'angiogenesis', TGFB1: 'TGF-β signalling, T-cell exclusion', KRT5: 'basal/squamous keratin' };

function makeCohort() {
  const y = range(N).map((i) => (i < N / 2 ? 1 : 0));
  const r = rng(SEED);
  const X = {}; GENES.forEach((g) => { X[g] = y.map((yi) => r.normal(yi * (EFFECT[g] || 0), 1)); });
  const pr = rng(99); const perms = range(NP).map(() => pr.shuffle(y));
  const diff = (v, lab) => { let a = 0, b = 0; for (let i = 0; i < N; i++) { if (lab[i]) a += v[i]; else b += v[i]; } return (a - b) / (N / 2); };
  const test = (v) => {
    const o = diff(v, y); let c = 0;
    for (const p of perms) if (Math.abs(diff(v, p)) >= Math.abs(o) - 1e-12) c++;
    return { diff: o, p: (1 + c) / (1 + NP) };
  };
  const score = (genes) => y.map((_, i) => genes.reduce((s, g) => s + X[g][i], 0) / genes.length);
  return { y, X, test, score };
}
function bh(ps) {
  const m = ps.length, idx = range(m).sort((a, b) => ps[a] - ps[b]), q = new Array(m);
  let run = 1;
  for (let k = m - 1; k >= 0; k--) { run = Math.min(run, (ps[idx[k]] * m) / (k + 1)); q[idx[k]] = run; }
  return q;
}
const fp = (p) => (p < 0.001 ? '&lt;0.001' : p.toFixed(3));
const dir = (d) => (d > 0 ? 'higher' : 'lower');

/** Build the three replay scripts from real tool outputs on the mock cohort. Each step: plan, call, obs, note, flag, tested[]. */
function buildScripts(D) {
  const T = {}; GENES.forEach((g) => { T[g] = D.test(D.X[g]); });
  const load = { plan: 'Load the cohort and check the groups before testing anything.', call: 'load_cohort("mock_IO_cohort")', obs: `${N} tumours: ${N / 2} responders, ${N / 2} non-responders; ${GENES.length} genes, log-scale expression.`, tested: [] };

  // A: careful loop
  const allP = GENES.map((g) => T[g].p), allQ = bh(allP);
  const hits = GENES.filter((g, i) => allQ[i] < 0.05);
  const ifn = ['CXCL9', 'GZMB', 'IFNG'], ifnS = D.test(D.score(ifn));
  const ctrlSet = ['KRT5', 'SOX2', 'VEGFA'], ctrlS = D.test(D.score(ctrlSet));
  const careful = [
    load,
    { plan: 'Pre-register the analysis: test all 12 genes in one call with Benjamini-Hochberg FDR at 5%, chosen before looking.', call: 'de_all(genes="all", test="permutation", correction="BH")',
      obs: `${hits.length} genes pass q &lt; 0.05: ${hits.map((g) => `${g} (q ${fp(allQ[GENES.indexOf(g)])})`).join(', ')}. All ${dir(T[hits[0]].diff)} in responders.`, tested: GENES.map((g, i) => ({ g, p: allP[i], q: allQ[i] })) },
    { plan: `Revise: ${hits.join(', ')} are one interferon-γ / cytotoxic program. Score the program instead of reading single genes.`, call: `gene_set_score(["${ifn.join('", "')}"])`,
      obs: `Program score ${dir(ifnS.diff)} in responders by ${Math.abs(ifnS.diff).toFixed(2)} units, permutation p ${fp(ifnS.p)}.`, tested: [{ g: 'IFN set', p: ifnS.p }] },
    { plan: 'Negative control: a 3-gene set with no expected link to response, scored the same way.', call: `gene_set_score(["${ctrlSet.join('", "')}"])`,
      obs: `Control set differs by ${Math.abs(ctrlS.diff).toFixed(2)} units, p ${fp(ctrlS.p)}: no signal, so the set score is not trivially significant.`, tested: [{ g: 'control set', p: ctrlS.p }] },
    { plan: 'Conclude, with every number traced to a tool output above.', call: '(final answer)', flag: 'good',
      obs: `Responders show a higher IFN-γ program (p ${fp(ifnS.p)}). CD8A and PDCD1 did not pass FDR (q ${fp(allQ[3])}, ${fp(allQ[4])}); in this mock cohort they carry a smaller planted effect, so n = ${N} is underpowered, not proof of no effect. Needs an independent cohort.`, tested: [] },
  ];

  // B: hallucinated tool
  const hall = [
    load,
    { plan: 'Run a pathway enrichment on the ranked genes.', call: 'run_gsea_pro(cohort, collection="hallmark")', flag: 'warn',
      obs: `ERROR: no tool named "run_gsea_pro". Registered tools: ${REGISTRY.join(', ')}.`, tested: [] },
    { plan: 'Unguarded agent continues as if the call had worked.', call: '(writes the result anyway)', flag: 'bad',
      audit: 'claim without tool output', obs: '"GSEA shows strong enrichment of HALLMARK_INTERFERON_GAMMA_RESPONSE (NES 2.4, FDR 0.001)." None of these numbers appears in any tool output: it is a fabricated observation that happens to sound right.', tested: [] },
    { plan: 'Guard: a trace check flags any number in the answer that no tool returned. Fall back to a registered tool.', call: `gene_set_score(["${ifn.join('", "')}"])`, flag: 'good',
      obs: `Program score ${dir(ifnS.diff)} in responders by ${Math.abs(ifnS.diff).toFixed(2)} units, permutation p ${fp(ifnS.p)}. Same direction as the fabricated claim, but this time it is evidence.`, tested: [{ g: 'IFN set', p: ifnS.p }] },
  ];

  // C: confirmation bias: test favourite genes one at a time, stop at the first p < 0.05
  const order = ['MYC', 'EGFR', 'SOX2', 'VEGFA', 'TGFB1', 'MKI67', 'KRT5'];
  const bias = [load]; const done = [];
  for (const g of order) {
    done.push(g);
    const hit = T[g].p < 0.05;
    bias.push({ plan: done.length === 1 ? `Hypothesis: ${g} (${KB[g]}) drives response. Test it.` : `No luck. Maybe ${g} (${KB[g]}) instead.`, call: `de_test("${g}")`,
      obs: `${g} ${dir(T[g].diff)} in responders, p ${fp(T[g].p)}.${hit ? ' Below 0.05: stop and write it up.' : ''}`, tested: [{ g, p: T[g].p }], flag: hit ? 'warn' : null });
    if (hit) break;
  }
  const last = done[done.length - 1], qs = bh(done.map((g) => T[g].p));
  bias.push({ plan: 'Biased agent reports its finding.', call: '(final answer)', flag: 'bad', audit: 'selective reporting',
    obs: `"${last} (${KB[last]}) is ${dir(T[last].diff)} in responders (p = ${fp(T[last].p)}), supporting a ${MECH[last]} mechanism." The ${done.length - 1} failed tests before it are not mentioned.`, tested: [] });
  bias.push({ plan: 'Audit: correct for every test that was actually run, and check the genes the agent never looked at.', call: `de_all(genes=[${done.length} tested], correction="BH")`, flag: 'good',
    obs: `${last}: BH q = ${fp(qs[done.length - 1])} over the ${done.length} tests run, not significant. Meanwhile the untested CXCL9, GZMB and IFNG have p ${fp(T.CXCL9.p)}, ${fp(T.GZMB.p)}, ${fp(T.IFNG.p)}: the real signal was never on the agent's list.`,
    tested: done.map((g, i) => ({ g, p: T[g].p, q: qs[i] })) });
  return { careful, hall, bias, T, firstBias: { gene: last, n: done.length, p: T[last].p, q: qs[done.length - 1] }, hits, ifnS };
}

// ---------------------------------------------------------------- Elo tournament
const HYP = [
  { id: 'H1', text: 'IFN-γ / cytotoxic program (CXCL9, GZMB, IFNG) is higher in responders.', q: 0.9, v: 0.3 },
  { id: 'H2', text: 'CD8 T-cell infiltration, not the IFN program itself, drives response.', q: 0.7, v: 0.4 },
  { id: 'H3', text: 'PD-1 (PDCD1) expression alone predicts response.', q: 0.45, v: 0.2 },
  { id: 'H4', text: 'Squamous differentiation (KRT5) is unrelated to response.', q: 0.5, v: 0.1 },
  { id: 'H5', text: 'TGF-β excludes T cells from non-responding tumours.', q: 0.35, v: 0.6 },
  { id: 'H6', text: 'Proliferation (MKI67) marks non-responders.', q: 0.25, v: 0.3 },
  { id: 'H7', text: 'VEGFA-driven angiogenesis suppresses response through multiple converging immune, metabolic and stromal mechanisms.', q: 0.15, v: 0.85 },
  { id: 'H8', text: 'A master-regulator network of MYC, SOX2 and EGFR rewires immunity across every tumour compartment, explaining response, resistance and relapse.', q: 0.1, v: 1.0 },
];
const MATCHES = 300, KELO = 24;
function tournament(biasOn) {
  const r = rng(biasOn ? 71 : 70);
  const elo = HYP.map(() => 1200), hist = [elo.slice()], log = [];
  for (let t = 0; t < MATCHES; t++) {
    const a = r.int(HYP.length); let b = r.int(HYP.length - 1); if (b >= a) b++;
    const z = 3 * (HYP[a].q - HYP[b].q) + (biasOn ? 4 * (HYP[a].v - HYP[b].v) : 0);
    const pA = 1 / (1 + Math.exp(-z)), aWins = r() < pA;
    const eA = 1 / (1 + 10 ** ((elo[b] - elo[a]) / 400)), sA = aWins ? 1 : 0;
    elo[a] += KELO * (sA - eA); elo[b] -= KELO * (sA - eA);
    hist.push(elo.slice()); log.push({ a, b, aWins, pA });
  }
  return { hist, log };
}
function spearman(x, y) {
  const rk = (v) => { const o = range(v.length).sort((i, j) => v[i] - v[j]); const R = new Array(v.length); o.forEach((i, k) => { R[i] = k; }); return R; };
  const a = rk(x), b = rk(y), n = x.length; let s = 0; for (let i = 0; i < n; i++) s += (a[i] - b[i]) ** 2;
  return 1 - (6 * s) / (n * (n * n - 1));
}

export default async function render(main) {
  header(main, 'Agents & AI co-scientists',
    'An <b>agent</b> is a language model in a loop: it plans, calls a tool (run a test, query a database), reads the result and revises its plan. A <b>co-scientist</b> adds a tournament in which hypotheses compete and are ranked. Both are only as trustworthy as the <b>tools, the judge and the audit trail</b>, which this page replays on mock data.');

  sourceCard(main, {
    org: 'Stanford (Biomni); Google Research (AI co-scientist); CellAgent; GPTCelltype',
    when: '2024–2025',
    links: ['biomni', 'coscientist', 'cellagent', 'gptcelltype'],
    claim: 'Biomni: a ReAct-style agent with about 150 biomedical tools. Google\'s AI co-scientist: a Gemini multi-agent system that ranks hypotheses in an Elo tournament. Published venues for the Science version of Biomni and a Nature version of the co-scientist are unverified in our brief.',
  });

  coreIdea(main, `<p><b>ReAct</b> ("reason + act") is the loop behind most science agents: the model writes a thought, emits a structured tool call, the system runs the tool and pastes the output back into the context, and the model continues. The model never touches the data directly; it only sees what tools return. That makes the <b>tool registry</b> and the <b>trace</b> (the log of every call and output) the things to inspect.</p>
  <p>A co-scientist generates many hypotheses and needs to rank them without running every experiment. Google's system uses an <b>Elo tournament</b>: pairs of hypotheses are compared (by model-run debates), and each comparison nudges their ratings, as in chess. Elo converts many noisy pairwise preferences into one ranking. The catch: the ranking measures <i>what the judge prefers</i>, which is only as good as the judge.</p>
  <p><b>Key ML ingredients:</b> tool-augmented generation in a loop, and pairwise-preference ranking (Elo). <b>No language model runs on this page</b>: the traces below are scripted, and every number in them is computed live from a mock cohort by real (tiny) tools.</p>`);

  // ---------------------------------------------------------------- figure 1: tool loop replay
  const D = makeCohort(), SC = buildScripts(D);
  const fig = figure(main, 'Replay an agent\'s tool loop',
    `Mock cohort: ${N} tumours, 12 genes, with a planted interferon-γ program (CXCL9, GZMB, IFNG strong; CD8A, PDCD1 weak). Tools are real functions on this data: permutation tests (${NP} label shuffles) and Benjamini-Hochberg FDR. The agent's text is scripted. Right: −log10 p of everything tested so far.`);
  toyBadge(fig, 'Scripted replay; no LLM is called.');
  const RUNS = { careful: 'Careful loop (pre-registered, FDR)', hall: 'Failure: hallucinated tool', bias: 'Failure: confirmation bias' };
  let run = 'careful', idx = 1, stopPlay = null;
  select(fig.controls, { label: 'Run', value: run, options: Object.entries(RUNS), onChange: (v) => { run = v; idx = 0; halt(); paint(); } });
  button(fig.controls, '◂ Back', () => { halt(); idx = Math.max(0, idx - 1); paint(); });
  button(fig.controls, 'Next step ▸', () => { halt(); idx = Math.min(SC[run].length - 1, idx + 1); paint(); }, true);
  const playBtn = button(fig.controls, 'Play', () => {
    if (stopPlay) { halt(); return; }
    if (idx >= SC[run].length - 1) idx = 0;
    playBtn.textContent = 'Pause';
    stopPlay = every(1100, () => { if (idx >= SC[run].length - 1) { halt(); return; } idx++; paint(); });
    paint();
  });
  function halt() { if (stopPlay) { stopPlay(); stopPlay = null; } playBtn.textContent = 'Play'; }
  const readout = stats(fig.root);
  const row = el('div', { class: 'fig-row' }); fig.body.append(row);
  const trace = el('div', { style: 'flex:1 1 460px;min-width:0;max-width:600px;max-height:440px;overflow:auto;display:flex;flex-direction:column;gap:8px;padding-right:4px' });
  const right = el('div'); row.append(trace, right);
  right.append(html(`<div style="font-size:13px;color:var(--ink-2);margin:0 0 4px 50px">Tool registry: ${REGISTRY.map((t) => `<code>${t}</code>`).join(' ')}</div>`));
  legend(right, [['raw p (single test)', cat(0)], ['FDR-adjusted q', cat(1)]]);
  const B = frame(right, { width: 470, height: 360, x: [0, 1], y: [0, 3.6], xScale: d3.scaleBand().padding(0.2), yLabel: '−log10 p  (higher = stronger)', margin: { bottom: 70 } });
  const thr = B.layer('thr');
  thr.append('line').attr('x1', 0).attr('x2', B.w).attr('y1', B.y(-Math.log10(0.05))).attr('y2', B.y(-Math.log10(0.05))).attr('stroke', muted()).attr('stroke-dasharray', '5 4').attr('stroke-width', 1.5);
  thr.append('text').attr('x', B.w - 4).attr('y', B.y(-Math.log10(0.05)) - 5).attr('text-anchor', 'end').attr('font-size', 11).attr('fill', ink2()).text('0.05');

  const FLAG = { good: ['var(--good)', 'grounded'], warn: ['var(--warn)', 'warning'], bad: ['var(--bad)', 'failure'] };
  function paint() {
    const S = SC[run];
    trace.innerHTML = '';
    S.slice(0, idx + 1).forEach((s, i) => {
      const f = s.flag ? FLAG[s.flag] : null;
      const card = el('div', { class: 'card', style: `padding:8px 10px;font-size:13.5px;line-height:1.45;border-left:4px solid ${f ? f[0] : 'var(--border)'};${i === idx ? '' : 'opacity:0.72'}` });
      card.append(html(`<div style="display:flex;justify-content:space-between;gap:8px"><b>Step ${i + 1}</b>${f ? `<span class="tag">${f[1]}</span>` : ''}</div>
        <div><span style="color:var(--muted)">Plan</span> ${s.plan}</div>
        <div><span style="color:var(--muted)">Call</span> <code>${s.call}</code></div>
        <div><span style="color:var(--muted)">Observe</span> ${s.obs}</div>`));
      trace.append(card);
    });
    trace.scrollTop = trace.scrollHeight;
    // evidence board: everything tested up to this step (later entries for the same name replace earlier ones)
    const seen = new Map();
    S.slice(0, idx + 1).forEach((s) => s.tested.forEach((t) => seen.set(t.g, t)));
    const data = [...seen.values()];
    B.x.domain(data.map((d) => d.g));
    B.svg.select('.axis.x').call(d3.axisBottom(B.x)).selectAll('text').attr('transform', 'rotate(-40)').attr('text-anchor', 'end').attr('dx', '-0.4em').attr('dy', '0.5em');
    const bw = B.x.bandwidth() || 1, cl = (p) => Math.min(3.6, -Math.log10(p));
    B.layer('bars').selectAll('rect.p').data(data, (d) => d.g).join('rect').attr('class', 'p')
      .attr('x', (d) => B.x(d.g)).attr('width', (d) => (d.q != null ? bw / 2 : bw)).attr('y', (d) => B.y(cl(d.p))).attr('height', (d) => B.h - B.y(cl(d.p))).attr('rx', 2).attr('fill', cat(0))
      .on('mousemove', (e, d) => tip(e, `<b>${d.g}</b><br>p = ${fp(d.p)}${d.q != null ? `<br>BH q = ${fp(d.q)}` : ''}`)).on('mouseleave', untip);
    B.layer('bars').selectAll('rect.q').data(data.filter((d) => d.q != null), (d) => d.g).join('rect').attr('class', 'q')
      .attr('x', (d) => B.x(d.g) + bw / 2).attr('width', bw / 2).attr('y', (d) => B.y(cl(d.q))).attr('height', (d) => B.h - B.y(cl(d.q))).attr('rx', 2).attr('fill', cat(1))
      .on('mousemove', (e, d) => tip(e, `<b>${d.g}</b><br>p = ${fp(d.p)}<br>BH q = ${fp(d.q)}`)).on('mouseleave', untip);
    const calls = S.slice(0, idx + 1).filter((s) => !s.call.startsWith('(')).length;
    const nTests = data.length;
    const badStep = S.slice(0, idx + 1).find((s) => s.flag === 'bad');
    readout.set({
      step: `${idx + 1} / ${S.length}`,
      'tool calls so far': calls,
      'distinct hypotheses tested': nTests,
      'audit': badStep ? badStep.audit : 'no problem flagged',
    });
  }
  paint();

  // ---------------------------------------------------------------- figure 2: Elo tournament
  const fig2 = figure(main, 'Elo tournament of hypotheses',
    `Eight mock hypotheses about the cohort above, each with a hidden "true quality" set by the planted biology. A simulated judge compares random pairs: it prefers the better hypothesis with probability logistic(3·Δquality). With the bias switched on it also rewards long, sweeping claims (+4·Δverbosity). Elo update K = ${KELO}, ${MATCHES} matches.`);
  toyBadge(fig2, 'Simulated judge; not Google\'s debate procedure.');
  let biasOn = false, T = tournament(false), t = 0, stop2 = null;
  checkbox(fig2.controls, { label: 'Judge prefers long, confident hypotheses', checked: false, onChange: (v) => { biasOn = v; T = tournament(v); halt2(); paint2(); } });
  button(fig2.controls, 'Next match ▸', () => { halt2(); t = Math.min(MATCHES, t + 1); paint2(); });
  const play2 = button(fig2.controls, 'Run tournament', () => {
    if (stop2) { halt2(); return; }
    if (t >= MATCHES) t = 0;
    play2.textContent = 'Pause';
    stop2 = every(40, () => { t = Math.min(MATCHES, t + 3); paint2(); if (t >= MATCHES) halt2(); });
  }, true);
  button(fig2.controls, 'Reset', () => { halt2(); t = 0; paint2(); });
  function halt2() { if (stop2) { stop2(); stop2 = null; } play2.textContent = 'Run tournament'; }
  const readout2 = stats(fig2.root);
  const row2 = el('div', { class: 'fig-row' }); fig2.body.append(row2);
  const l2 = el('div'), r2 = el('div'); row2.append(l2, r2);
  const order = range(HYP.length);   // already sorted by true quality, best first (H1..H8)
  const E = frame(l2, { width: 560, height: 340, x: [900, 1500], y: [0, 1], yScale: d3.scaleBand().padding(0.25), margin: { left: 40, bottom: 42, right: 200 }, xTicks: 4, xLabel: 'Elo rating (start 1200)', grid: false });
  E.y.domain(order.map((i) => HYP[i].id).reverse());
  E.svg.select('.axis.y').call(d3.axisLeft(E.y));
  E.g.append('line').attr('x1', E.x(1200)).attr('x2', E.x(1200)).attr('y1', 0).attr('y2', E.h).attr('stroke', muted()).attr('stroke-dasharray', '4 3');
  const R = frame(r2, { width: 440, height: 340, x: [0, MATCHES], y: [900, 1500], xLabel: 'match', yLabel: 'Elo' });
  legend(r2, [['H1 (best by truth)', cat(0)], ['H8 (worst, most verbose)', cat(1)], ['H7 (verbose)', cat(2)]]);
  const lastBox = html('', 'div'); lastBox.style.cssText = 'font-size:13.5px;margin:6px 0;min-height:42px;color:var(--ink-2)'; fig2.body.append(lastBox);

  function paint2() {
    const elo = T.hist[t];
    E.layer('bars').selectAll('rect').data(order).join('rect')
      .attr('y', (i) => E.y(HYP[i].id)).attr('height', E.y.bandwidth()).attr('x', E.x(900)).attr('width', (i) => Math.max(0, E.x(Math.max(900, elo[i])) - E.x(900))).attr('rx', 2).attr('fill', cat(0))
      .on('mousemove', (e, i) => tip(e, `<b>${HYP[i].id}</b>: ${HYP[i].text}<br>Elo ${elo[i].toFixed(0)} · true quality ${HYP[i].q} · verbosity ${HYP[i].v}`)).on('mouseleave', untip);
    const rankElo = range(HYP.length).sort((a, b) => elo[b] - elo[a]);
    E.layer('lbl').selectAll('text').data(order).join('text').attr('x', E.w + 10).attr('y', (i) => E.y(HYP[i].id) + E.y.bandwidth() / 2 + 4).attr('text-anchor', 'start').attr('font-size', 11).attr('fill', ink())
      .text((i) => `Elo rank ${t ? rankElo.indexOf(i) + 1 : '–'} · true rank ${i + 1}${HYP[i].v >= 0.8 ? ' · verbose' : ''}`);
    const series = [[0, cat(0)], [7, cat(1)], [6, cat(2)]];
    R.layer('lines').selectAll('path').data(series).join('path').attr('fill', 'none').attr('stroke-width', 2).attr('stroke', (d) => d[1])
      .attr('d', (d) => d3.line().x((_, k) => R.x(k)).y((h) => R.y(Math.max(900, Math.min(1500, h[d[0]]))))(T.hist.slice(0, t + 1)));
    const rho = spearman(elo, HYP.map((h) => h.q));
    readout2.set({ 'matches played': `${t} / ${MATCHES}`, 'Spearman ρ (Elo vs true quality)': t ? rho.toFixed(2) : '–', 'top-ranked by Elo': t ? HYP[rankElo[0]].id : '–' });
    if (t) {
      const m = T.log[t - 1], w = m.aWins ? m.a : m.b, lo = m.aWins ? m.b : m.a;
      lastBox.innerHTML = `<b>Match ${t}:</b> judge picks <b>${HYP[w].id}</b> over ${HYP[lo].id} (it preferred ${HYP[m.a].id} with probability ${m.pA.toFixed(2)}). <i>${HYP[w].text}</i>`;
    } else lastBox.innerHTML = 'Press <b>Run tournament</b> or step one match at a time.';
  }
  paint2();

  const fb = SC.firstBias;
  const fin = (bOn) => { const e = tournament(bOn).hist[MATCHES]; const rk = range(HYP.length).sort((x, y) => e[y] - e[x]); return { rho: spearman(e, HYP.map((h) => h.q)), rank: (i) => rk.indexOf(i) + 1, top: HYP[rk[0]].id }; };
  const F0 = fin(false), F1 = fin(true);
  section(main, 'What to notice', `<ul>
    <li><b>Careful loop:</b> step 2 tests all 12 genes at once and corrects for it. ${SC.hits.join(', ')} pass FDR; the agent then scores them as one program (p ${fp(SC.ifnS.p)}) and runs a negative-control set. Each claim in the final answer points back to a tool output.</li>
    <li><b>Hallucinated tool:</b> the agent calls <code>run_gsea_pro</code>, which is not in the registry. The error is returned, but the unguarded agent writes an enrichment result anyway. The fabricated claim even points in the right direction, which is what makes this failure dangerous: you cannot spot it by plausibility, only by checking the trace.</li>
    <li><b>Confirmation bias:</b> the agent tests its favourite genes one at a time and stops at the first p &lt; 0.05: ${fb.gene} after ${fb.n} tests (p = ${fp(fb.p)}). Corrected for the ${fb.n} tests it ran, q = ${fp(fb.q)}. The planted signal (CXCL9, GZMB, IFNG) was never tested. Watch the bars on the right: the raw-p bar crosses the dashed line, the q bar does not.</li>
    <li><b>Elo, honest judge:</b> run the tournament. After ${MATCHES} matches ${F0.top} is ranked first (H1 is the planted truth) and Spearman ρ between Elo and true quality is ${F0.rho.toFixed(2)}. Step one match at a time to see how noisy single comparisons are; the judge sometimes picks the worse hypothesis, and only many matches average that out.</li>
    <li><b>Elo, biased judge:</b> tick the bias box and rerun. The two sweeping, low-quality hypotheses climb: H7 ends at Elo rank ${F1.rank(6)} and H8 at rank ${F1.rank(7)} (true ranks 7 and 8), and ρ falls to ${F1.rho.toFixed(2)}. The tournament worked exactly as designed; it faithfully ranked the judge's taste.</li>
  </ul>`);

  steps(main, 'How it works', [
    '<b>Plan:</b> the model writes its next intention in text.',
    '<b>Act:</b> it emits a structured tool call (name + arguments). The system checks the name against the registry and runs it; the model never runs code or reads raw data itself.',
    '<b>Observe:</b> the tool output is appended to the context. Errors are outputs too, and a good loop treats them as information.',
    '<b>Revise:</b> the model updates the plan and repeats until it writes a final answer. A trace check verifies that each claimed number appears in some tool output.',
    '<b>Rank hypotheses:</b> generate many, compare random pairs with a judge, update Elo ratings after each comparison, and send the top-rated ones on to (real) experiments.',
  ]);
  math(main, 'Elo expected score   E_A = 1 / (1 + 10^((R_B − R_A)/400))\nupdate               R_A ← R_A + K (S_A − E_A),   S_A ∈ {0, 1}\nBH FDR               q_(k) = min_{j ≥ k}  p_(j) · m / j');

  realityCheck(main, `<ul>
    <li>Capability claims for ${cite('biomni', 'Biomni')} and the ${cite('coscientist', 'AI co-scientist')} come from their developers. Our brief lists no independent head-to-head benchmark of either, and the Science (Biomni) and Nature (co-scientist) versions are <b>unverified</b> in our brief.</li>
    <li><b>Evaluation gap:</b> an agent that finishes a task and an agent that finishes it <i>correctly</i> look the same unless someone checks the trace. Elo rankings measure agreement with a model judge, not experimental truth; only experiments can close that gap.</li>
    <li>Narrower LLM tools report more modest numbers: the ${cite('gptcelltype', 'GPTCelltype')} authors report &gt;75% concordance with manual annotation in most tissues, which still means a quarter of labels need a human (see <a href="#/archive/frontier/llm-annotation">LLM cell annotation</a>).</li>
    <li>Agents inherit every statistical sin a human can commit, faster: forking paths, optional stopping, untracked multiple testing. Speed multiplies the number of tests, so FDR control matters more, not less.</li>
  </ul>`);

  useIt(main, `<ul>
    <li>Keep the full trace (calls, arguments, outputs) and make every number in a report link to the call that produced it. Reject answers containing numbers with no source.</li>
    <li>Restrict agents to a registry of tested tools, and make unknown-tool calls fail loudly rather than return something plausible.</li>
    <li>Pre-register the analysis plan (genes, tests, correction) before the agent sees outcomes, exactly as you would for a human analyst.</li>
    <li>Treat an agent's ranked hypotheses as a reading list, not as evidence. Check the judge for length and confidence bias by swapping in deliberately verbose decoys.</li>
  </ul>`);

  seeAlso(main, [['frontier', 'llm-annotation', 'LLM cell annotation'], ['frontier', 'metric-gaming', 'Metric gaming'], ['concepts', 'multiple-testing', 'Multiple testing & FDR'],
    ['omics', 'bulk-de', 'Differential expression'], ['omics', 'bulk-validation', 'Locking & validating the signature']]);
}
