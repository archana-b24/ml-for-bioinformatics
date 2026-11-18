// Embeddings, attention and transformers, lesson format.
// Arc: what → why → where → how it works (6 steps: gene/cell vectors → query-key-value → heads and position → the block
//      → masked pretraining → from pretrained representations to patient prediction) → pitfalls → recap.
// Gene-level toy corpus: 400 simulated cells built from the study's programmes (IFN-γ, stroma/CAF, proliferation, myeloid).
// Patient-level lab: the shared cohort. Every number in the prose is computed at render time.
import { header, figure, slider, select, button, checkbox, stats, legend, seeAlso, animate, tip, untip, el, html } from '../lib/ui.js';
import { frame, cat, css, ink, ink2, muted, diverging, sequential } from '../lib/plot.js';
import { rng, mean, sd, range } from '../lib/data.js';
import { fitLogistic, predictProb, auc, pca, standardize } from '../lib/models.js';
import { loadCohort, FEATURES, studyCard } from '../lib/study.js';
import { lessonNav, chapter, prose, rung, worked, quiz, cards, table, code, recap } from '../lib/lesson.js';
const d3 = window.d3;

const f2 = (v) => (Math.abs(v) < 0.005 ? '0.00' : v.toFixed(2));
const f1 = (v) => v.toFixed(1);
const f3 = (v) => v.toFixed(3);
const pc = (v) => Math.round(100 * v) + '%';
const sum = (a) => a.reduce((s, v) => s + v, 0);
const dotp = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
const cosine = (a, b) => dotp(a, b) / (Math.hypot(...a) * Math.hypot(...b) || 1);
const softmaxRow = (r) => { const m = Math.max(...r), e = r.map((v) => Math.exp(v - m)), s = sum(e); return e.map((v) => v / s); };
const mm = (A, B) => A.map((r) => B[0].map((_, j) => sum(r.map((v, k) => v * B[k][j]))));
const tr = (A) => A[0].map((_, j) => A.map((r) => r[j]));

// ---------------------------------------------------------------------------------------------
// Toy single-cell corpus and masked-gene pretraining (a CBOW-style stand-in for a masked language model)
const PROGS = [
  { name: 'IFN-γ response', genes: ['CD8A', 'GZMB', 'IFNG', 'CXCL9', 'CD274'] },
  { name: 'Stroma (CAF)', genes: ['COL1A1', 'COL1A2', 'FAP', 'TGFB1', 'POSTN'] },
  { name: 'Proliferation', genes: ['MKI67', 'TOP2A', 'PCNA', 'MCM2', 'CCNB1'] },
  { name: 'Myeloid', genes: ['CD68', 'LYZ', 'CD14', 'C1QA', 'S100A8'] },
  { name: 'Housekeeping', genes: ['ACTB', 'GAPDH', 'B2M', 'RPL13A'] },
];
const GENES = PROGS.flatMap((p, k) => p.genes.map((name) => ({ name, prog: k })));
const G = GENES.length, GI = Object.fromEntries(GENES.map((g, i) => [g.name, i]));
const CTYPES = [{ name: 'T cell', on: 0 }, { name: 'Fibroblast', on: 1 }, { name: 'Proliferating tumour', on: 2 }, { name: 'Macrophage', on: 3 }, { name: 'Quiescent tumour', on: -1 }];
const SCALE = 4, MASK = 0.3, EPOCHS = 100, LR = 1.5, D0 = 2;

/** Each cell is the set of genes detected in it (a "sentence" of gene tokens). */
function makeCells(perType = 80, seed = 5) {
  const r = rng(seed), out = [];
  CTYPES.forEach((t, ti) => {
    for (let c = 0; c < perType; c++) {
      let toks;
      do {
        toks = [];
        GENES.forEach((g, gi) => {
          const pr = g.prog === 4 ? 0.35 : g.prog === t.on ? 0.75 : (t.on === 3 && g.name === 'CD274') ? 0.45 : 0.04;
          if (r() < pr) toks.push(gi);
        });
      } while (toks.length < 4);
      out.push({ type: ti, toks: r.shuffle(toks) });
    }
  });
  return r.shuffle(out);
}
function newModel(seed = 11, d = 2) {
  const r = rng(seed), m = () => Array.from({ length: G }, () => Array.from({ length: d }, () => r.normal(0, 0.3)));
  return { E: m(), U: m(), B: new Array(G).fill(0), d, r, epoch: 0 };
}
function pick(toks, frac, r) {
  const mk = toks.filter(() => r() < frac);
  let masked = mk.length ? mk : [toks[r.int(toks.length)]];
  if (masked.length === toks.length) masked = masked.slice(1);
  const set = new Set(masked);
  return { masked, vis: toks.filter((t) => !set.has(t)) };
}
function ctxOf(M, vis) { const c = new Array(M.d).fill(0); vis.forEach((t) => M.E[t].forEach((v, k) => { c[k] += v / vis.length; })); return c; }
function probsOf(M, vis) { const c = ctxOf(M, vis); return softmaxRow(M.U.map((u, g) => M.B[g] + SCALE * dotp(u, c))); }
function trainEpoch(M, cells, frac, lr = LR, batch = 16) {
  const idx = M.r.shuffle(cells.map((_, i) => i)), d = M.d;
  lr *= 1 - 0.8 * M.epoch / EPOCHS; // learning rate decays linearly to 20% of its start
  for (let s0 = 0; s0 < idx.length; s0 += batch) {
    const dU = M.U.map((u) => u.map(() => 0)), dB = new Array(G).fill(0), dE = M.E.map((u) => u.map(() => 0));
    const part = idx.slice(s0, s0 + batch);
    part.forEach((i) => {
      const { masked, vis } = pick(cells[i].toks, frac, M.r), c = ctxOf(M, vis), p = probsOf(M, vis);
      const dl = p.slice(); masked.forEach((m) => { dl[m] -= 1 / masked.length; });
      const dc = new Array(d).fill(0);
      for (let g = 0; g < G; g++) { for (let k = 0; k < d; k++) { dc[k] += SCALE * dl[g] * M.U[g][k]; dU[g][k] += SCALE * dl[g] * c[k]; } dB[g] += dl[g]; }
      vis.forEach((t) => { for (let k = 0; k < d; k++) dE[t][k] += dc[k] / vis.length; });
    });
    const sc = lr / part.length;
    for (let g = 0; g < G; g++) { M.B[g] -= sc * dB[g]; for (let k = 0; k < d; k++) { M.U[g][k] -= sc * dU[g][k]; M.E[g][k] -= sc * dE[g][k]; } }
  }
  M.epoch++;
}
/** Held-out masked-gene prediction with fixed masks: loss, top-1 accuracy, and how often the top guess is in the right programme. */
function evaluate(M, cells, frac, seed = 3) {
  const r = rng(seed); let L = 0, n = 0, top1 = 0, prog = 0, np = 0;
  cells.forEach((cl) => {
    const { masked, vis } = pick(cl.toks, frac, r), p = probsOf(M, vis), am = p.indexOf(Math.max(...p));
    masked.forEach((m) => { L -= Math.log(p[m]); n++; if (am === m) top1++; if (GENES[m].prog < 4) { np++; if (GENES[am].prog === GENES[m].prog) prog++; } });
  });
  return { loss: L / n, top1: top1 / n, prog: prog / np };
}
function unigramLoss(train, test, frac, seed = 3) {
  const cnt = new Array(G).fill(1); train.forEach((c) => c.toks.forEach((t) => cnt[t]++));
  const tot = sum(cnt), r = rng(seed); let L = 0, n = 0;
  test.forEach((cl) => { pick(cl.toks, frac, r).masked.forEach((m) => { L -= Math.log(cnt[m] / tot); n++; }); });
  return L / n;
}

export default async function render(main) {
  const C = await loadCohort();
  const D = C.discovery, V = C.validation, y = D.map((p) => p.response), yv = V.map((p) => p.response);
  const cells = makeCells(), TRN = cells.slice(0, 320), VAL = cells.slice(320);
  const cache = {};
  const pretrain = (d, frac) => { const k = d + '/' + frac; if (!cache[k]) { const M = newModel(11, d); for (let e = 0; e < EPOCHS; e++) trainEpoch(M, TRN, frac); cache[k] = M; } return cache[k]; };
  const base = pretrain(D0, MASK);
  const baseEval = evaluate(base, VAL, MASK), uni = unigramLoss(TRN, VAL, MASK), chance = Math.log(G);
  // 2-d view of the learned vectors: the first two principal components of the gene vectors, scaled so PC1 has unit sd
  const P2 = pca(base.E, { scale: false });
  const sc1 = sd(P2.S.map((r) => r[0]));
  const XY = P2.S.map((r) => [r[0] / sc1, r[1] / sc1]);
  const gname = (i) => GENES[i].name;
  const cellVec = (c) => range(D0).map((k) => mean(c.toks.map((t) => base.E[t][k])));
  const cellVecs = cells.map(cellVec);
  const nearestGenes = (i, k = 5) => range(G).filter((j) => j !== i).map((j) => ({ j, s: cosine(base.E[i], base.E[j]) })).sort((a, b) => b.s - a.s).slice(0, k);
  const nbHit = (() => { // share of each gene's 4 nearest neighbours that sit in its own programme, and the value expected by chance
    let h = 0, ch = 0, tot = 0;
    GENES.forEach((g, i) => { const same = GENES.filter((x) => x.prog === g.prog).length - 1, k = Math.min(4, same); h += nearestGenes(i, k).filter((q) => GENES[q.j].prog === g.prog).length; tot += k; ch += k * same / (G - 1); });
    return { obs: h / tot, chance: ch / tot };
  })();
  const cellNN = (() => { let ok = 0; cellVecs.forEach((v, i) => { let b = -2, bj = -1; cellVecs.forEach((w, j) => { if (j !== i) { const s = cosine(v, w); if (s > b) { b = s; bj = j; } } }); if (cells[bj].type === cells[i].type) ok++; }); return ok / cells.length; })();
  const nnBy = CTYPES.map((_, t) => { let ok = 0, n = 0; cellVecs.forEach((v, i) => { if (cells[i].type !== t) return; let b = -2, bj = -1; cellVecs.forEach((w, j) => { if (j !== i) { const sv = cosine(v, w); if (sv > b) { b = sv; bj = j; } } }); n++; if (cells[bj].type === t) ok++; }); return ok / n; });
  const worst = nnBy.indexOf(Math.min(...nnBy));
  const typeShare = (t) => cells.filter((c) => c.type === t).length / cells.length;

  header(main, 'Embeddings, attention and transformers',
    'A way to represent genes, cells and patients as <b>lists of numbers in which similar things sit close together</b>, and a mechanism, <b>attention</b>, that lets each item update its numbers using the other items present. Stacked, and trained to fill in hidden values, these parts form the transformer, the architecture behind single-cell foundation models.');
  lessonNav(main);
  studyCard(main, C, '<b>Question for this page:</b> how can genes, cells and patients be turned into vectors that capture biology, and how does attention combine them so that a model trained on unlabelled data helps when few labels are available?');
  prose(main, `<p class="fig-caption">Two data sources are used. Chapters 1–4 (steps 1–5) work on a <b>toy single-cell corpus of ${cells.length} simulated cells</b> that reuses the study's biological programmes (IFN-γ response, stroma/CAF, proliferation, myeloid). Step 6 returns to the <b>patient cohort</b> above, with validation on Centre C.</p>`);

  // =====================================================================================
  // 1. WHAT
  // =====================================================================================
  let ch = chapter(main, 'what', 'What are embeddings, attention and transformers?', 'In one line: <b>vectors that place similar items close together, plus a rule that lets items borrow information from each other</b>.');
  prose(ch, `<p>Three ideas build on each other.</p><ol>
    <li>An <b>embedding</b> is a vector (a short list of numbers) assigned to each item, such as a gene, a cell or a patient. The numbers are chosen, usually by training, so that items with similar behaviour receive similar vectors. "Similar" is then measured by distance or by the angle between vectors (cosine similarity).</li>
    <li><b>Attention</b> is an operation on a set of vectors. Each item compares itself with every other item, converts the comparisons into weights that sum to 1, and replaces its vector by a weighted average of the others' contributions. The model learns what to compare, so the same gene can attend to different partners in different cells.</li>
    <li>A <b>transformer</b> stacks attention with small per-item networks. It is usually trained by <b>self-supervised pretraining</b>: parts of the input are hidden (<b>masked</b>) and the model must predict them, which needs no labels.</li></ol>
    <p>The diagram follows one T cell from the toy corpus through these parts.</p>`);
  const exCell = cells.find((c) => c.type === 0 && c.toks.length >= 6) || cells[0];
  ch.append(pipelineSvg(exCell, base, XY));
  prose(ch, '<p>Key terms:</p>');
  table(ch, [
    { t: 'Token', d: 'One input item. For a cell, a token is a detected gene (some models also attach its expression level or rank).', e: `${gname(exCell.toks[0])}, ${gname(exCell.toks[1])}, …` },
    { t: 'Embedding', d: 'The learned vector for a token. All tokens of the same gene share one vector (a lookup table).', e: `${gname(exCell.toks[0])} → [${XY[exCell.toks[0]].map(f2).join(', ')}] (${D0} numbers, rescaled as in the diagram)` },
    { t: 'Query, key, value', d: 'Three vectors derived from each token. A query is matched against every key; the matching strengths weight the values.', e: 'worked through in step 2' },
    { t: 'Attention head', d: 'One query/key/value set. Several heads run in parallel and can learn different relationships.', e: 'step 3' },
    { t: 'Transformer block', d: 'Multi-head attention plus a small feed-forward network, each with a residual connection and normalisation. Blocks are stacked.', e: 'step 4' },
    { t: 'Masked pretraining', d: 'Hide some tokens and train the model to predict them from the rest. No labels are needed, so very large unlabelled collections can be used.', e: 'step 5' },
  ], [['t', 'Term'], ['d', 'Meaning'], ['e', 'In this page']], { cls: 'compact' });

  // =====================================================================================
  // 2. WHY
  // =====================================================================================
  ch = chapter(main, 'why', 'Why do we need them?', 'Tables of numbers treat every gene as an unrelated column. Embeddings and attention supply the missing notion of relatedness and context.');
  prose(ch, `<p>A conventional model sees a patient or cell as a row of columns. This has three limitations.</p><ol>
    <li><b>No notion of similarity between features.</b> A one-hot encoding of a gene (a vector with a single 1) makes every pair of genes equally unrelated: the cosine similarity between CD8A and GZMB is exactly 0, the same as between CD8A and COL1A1. A learned embedding can place CD8A near GZMB.</li>
    <li><b>Fixed-width input.</b> A table needs the same columns for every sample. Cells express different genes, studies measure different panels, and a set of detected genes has no natural order.</li>
    <li><b>No context.</b> A fixed weight for a gene ignores the other genes present. CD274 (PD-L1) means something different in a cell with IFN-γ response genes than in a quiescent tumour cell.</li></ol>`);
  const fw = figure(ch, 'One-hot vectors versus learned embeddings: cosine similarity between all gene pairs',
    `Each cell of a grid is the cosine similarity between two genes (rows and columns in the same order; colour bars mark programmes). Left: one-hot vectors, where only the diagonal is non-zero. Right: embedding vectors learned by masked pretraining on the ${cells.length} toy cells (step 5). Hover a cell for the pair and its value.`);
  const rowW = el('div', { class: 'fig-row' }); fw.body.append(rowW);
  const wa = el('div'), wb = el('div'); rowW.append(wa, wb);
  const simAll = (f) => range(G).map((i) => range(G).map((j) => f(i, j)));
  const Soh = simAll((i, j) => (i === j ? 1 : 0)), Semb = simAll((i, j) => cosine(base.E[i], base.E[j]));
  simGrid(wa, Soh, 'One-hot'); simGrid(wb, Semb, `Learned embedding (${D0} dimensions)`);
  legend(fw.body, PROGS.map((p, i) => [p.name, cat(i)]));
  const within = [], between = [];
  GENES.forEach((a, i) => GENES.forEach((b, j) => { if (j > i && a.prog < 4 && b.prog < 4) (a.prog === b.prog ? within : between).push(Semb[i][j]); }));
  prose(ch, `<p>Among the 20 programme genes, the mean cosine similarity is <b>${f2(mean(within))}</b> within a programme and <b>${f2(mean(between))}</b> between programmes. The vectors were never told the programmes: they emerge from which genes occur together. Housekeeping genes occur in every cell type, so they carry no programme information.</p>`);
  worked(ch, 'The same gene in two contexts',
    `<p>CD274 is detected in ${pc(cells.filter((c) => c.toks.includes(GI.CD274)).length / cells.length)} of the toy cells: in T cells (IFN-γ response) and in macrophages. A fixed column cannot represent that CD274 co-occurs with CXCL9 in one cell type and with CD68 in another. An attention layer can: the vector for CD274 is updated using whichever neighbours are present (step 2).</p>`);

  // =====================================================================================
  // 3. WHERE
  // =====================================================================================
  ch = chapter(main, 'where', 'Where are they used?', 'Wherever the input is a set or sequence of items with relationships among them, and unlabelled data are plentiful.');
  cards(ch, [
    { tag: 'Single-cell', title: 'Foundation models for cells', body: 'Geneformer ranks genes by expression within each cell and masks some; scGPT combines gene tokens with binned expression values. Both are pretrained on millions of cells and used to embed new cells. See <a href="#/frontier/foundation-models">Single-cell foundation models</a> for what these models do and do not improve over simple baselines.' },
    { tag: 'Proteins', title: 'Protein language models', body: 'Amino acids are tokens; masked pretraining on protein sequence databases yields embeddings used to predict structure, variant effects and function (ESM-2, with ESMFold for structure).' },
    { tag: 'Genomics', title: 'DNA sequence to function', body: 'Models such as Enformer place attention layers over long DNA windows so that a promoter can use information from distant enhancers.' },
    { tag: 'Pathology', title: 'Vision transformers on slides', body: 'Image patches are the tokens. Slide-level models attend over thousands of tiles, and the resulting embeddings are used to predict molecular status or outcome.' },
    { tag: 'Spatial omics', title: 'Cells as tokens', body: 'A cell, its neighbours or a spatial niche can be a token, so attention integrates the local tissue context.' },
    { tag: 'Embeddings alone', title: 'Cell and patient maps', body: 'PCA, scVI and similar methods produce embeddings used for clustering, annotation, integration and nearest-neighbour search (see <a href="#/concepts/pca">PCA</a> and <a href="#/concepts/tsne-umap">t-SNE and UMAP</a>).' },
    { tag: 'Clinical text', title: 'Language models for reports', body: 'Pathology and clinical notes are processed by transformers, for example to extract diagnoses or match patients to trials.' },
    { tag: 'Retrieval', title: 'Similarity search', body: 'Cells, papers or variants are embedded once and then searched by cosine similarity, for example to find reference cells similar to a query cell.' },
    { tag: 'Outside biology', title: 'Language and images', body: 'Transformers were introduced for machine translation and are now the standard architecture for text, image and multi-modal models.' },
  ]);
  prose(ch, `<p><b>When not to use them:</b> small tabular datasets with a few well-understood predictors (a penalised regression or tree ensemble is more stable and easier to interpret, as step 6 illustrates); settings where the number of labelled samples is large and a simple model already saturates; and problems where interpretability of each weight is required. Pretrained models also require checking that the pretraining data resemble the new data (tissue, platform, species).</p>`);

  // =====================================================================================
  // 4. HOW
  // =====================================================================================
  ch = chapter(main, 'how', 'How it works', 'Six steps: vectors, query–key–value attention, heads and position, the block, masked pretraining, and use with few labels.');

  // ---- Step 1: vectors
  let rb = rung(ch, 1, 'Genes and cells as vectors',
    `<p>An embedding table has one row of ${D0} numbers per gene. In step 5 these rows are learned so that genes that occur together in cells end up with similar rows. A <b>cell</b> can then be described by the average of the vectors of the genes detected in it. Similarity is measured by <b>cosine similarity</b>: the cosine of the angle between two vectors (1 = same direction, 0 = unrelated, −1 = opposite).</p>
     <p>The figure shows the vectors from the end of the step-5 training run (${D0 === 2 ? 'the vectors have only two numbers, so they are plotted as they are, with the axes rotated to the principal components' : 'projected to their first two principal components'}). Select a gene to list its nearest neighbours.</p>`);
  const f1g = figure(rb, 'Learned gene and cell vectors',
    'Large points are genes, coloured by programme. Small points (optional) are cells, each placed at the mean of its genes\' vectors and coloured by cell type (same colour order as the programmes; quiescent tumour cells use the housekeeping colour). Right: the selected gene\'s five most similar genes by cosine similarity in the full vector.');
  let gSel = GI.CD274, showCells = false;
  const gPick = select(f1g.controls, { label: 'Gene', value: String(gSel), options: GENES.map((g, i) => [String(i), `${g.name} (${PROGS[g.prog].name})`]), onChange: (v) => { gSel = +v; paint1(); } });
  select(f1g.controls, { label: 'Show', value: 'genes', options: [['genes', 'Genes'], ['both', 'Genes and cells']], onChange: (v) => { showCells = v === 'both'; paint1(); } });
  const s1 = stats(f1g.root);
  const row1 = el('div', { class: 'fig-row' }); f1g.body.append(row1);
  const c1a = el('div'), c1b = el('div'); row1.append(c1a, c1b);
  const ext1 = [d3.min(XY, (r) => r[0]), d3.max(XY, (r) => r[0])], ext2 = [d3.min(XY, (r) => r[1]), d3.max(XY, (r) => r[1])];
  const F1 = frame(c1a, { width: 500, height: 400, x: [ext1[0] - 0.4, ext1[1] + 0.4], y: [ext2[0] - 0.4, ext2[1] + 0.4], xLabel: 'principal component 1', yLabel: 'principal component 2' });
  const cellXY = cells.map((c) => [mean(c.toks.map((t) => XY[t][0])), mean(c.toks.map((t) => XY[t][1]))]);
  const cellL = F1.layer('cells'), geneL = F1.layer('genes'), lblL = F1.layer('lbl');
  const NB = frame(c1b, { width: 360, height: 400, x: [0, 1], y: [0, 1], axes: false, grid: false, margin: { left: 80, right: 20, top: 34, bottom: 30 } });
  legend(f1g.body, PROGS.map((p, i) => [p.name, cat(i)]));
  function paint1() {
    cellL.selectAll('circle').data(showCells ? cells : []).join('circle').attr('cx', (d, i) => F1.x(cellXY[cells.indexOf(d)][0])).attr('cy', (d) => F1.y(cellXY[cells.indexOf(d)][1])).attr('r', 2.6)
      .attr('fill', (d) => cat(d.type)).attr('fill-opacity', 0.35)
      .on('mousemove', (e, d) => tip(e, `${CTYPES[d.type].name}<br>${d.toks.length} genes detected`)).on('mouseleave', untip);
    const nb = nearestGenes(gSel, 5), nbSet = new Set(nb.map((q) => q.j));
    geneL.selectAll('circle').data(GENES).join('circle').attr('cx', (d, i) => F1.x(XY[i][0])).attr('cy', (d, i) => F1.y(XY[i][1]))
      .attr('r', (d, i) => (i === gSel ? 10 : 7)).attr('fill', (d) => cat(d.prog)).attr('stroke', (d, i) => (i === gSel ? ink() : css('--surface'))).attr('stroke-width', (d, i) => (i === gSel ? 3 : 1.5))
      .on('mousemove', (e, d) => tip(e, `<b>${d.name}</b> (${PROGS[d.prog].name})<br>cosine with ${gname(gSel)}: ${f2(cosine(base.E[GI[d.name]], base.E[gSel]))}`)).on('mouseleave', untip)
      .on('click', (e, d) => { gSel = GI[d.name]; gPick.value = String(gSel); paint1(); });
    lblL.selectAll('text').data(range(G).filter((i) => i === gSel || nbSet.has(i))).join('text').attr('x', (i) => F1.x(XY[i][0]) + 11).attr('y', (i) => F1.y(XY[i][1]) - 9)
      .attr('font-size', 11.5).attr('font-weight', (i) => (i === gSel ? 700 : 400)).attr('fill', ink()).attr('paint-order', 'stroke').attr('stroke', css('--surface')).attr('stroke-width', 3).text(gname);
    NB.g.selectAll('*').remove();
    NB.g.append('text').attr('x', -76).attr('y', -16).attr('font-size', 12).attr('fill', ink2()).text(`Most similar to ${gname(gSel)} (cosine)`);
    const yb = d3.scaleBand().domain(nb.map((q) => q.j)).range([0, NB.h - 20]).padding(0.3), xs = d3.scaleLinear().domain([-1, 1]).range([0, NB.w - 30]);
    NB.g.append('line').attr('x1', xs(0)).attr('x2', xs(0)).attr('y1', 0).attr('y2', NB.h - 20).attr('stroke', muted()).attr('stroke-dasharray', '3 3');
    nb.forEach((q) => {
      const yy = yb(q.j);
      NB.g.append('rect').attr('x', xs(Math.min(0, q.s))).attr('width', Math.abs(xs(q.s) - xs(0))).attr('y', yy).attr('height', yb.bandwidth()).attr('fill', cat(GENES[q.j].prog))
        .on('mousemove', (e) => tip(e, `${gname(q.j)} (${PROGS[GENES[q.j].prog].name})<br>cosine ${f2(q.s)}`)).on('mouseleave', untip);
      NB.g.append('text').attr('x', -8).attr('y', yy + yb.bandwidth() / 2 + 4).attr('text-anchor', 'end').attr('font-size', 12).attr('fill', ink2()).text(gname(q.j));
      NB.g.append('text').attr('x', xs(q.s) + (q.s >= 0 ? 4 : -4)).attr('text-anchor', q.s >= 0 ? 'start' : 'end').attr('y', yy + yb.bandwidth() / 2 + 4).attr('font-size', 11).attr('fill', ink()).text(f2(q.s));
    });
    NB.g.append('text').attr('x', xs(0)).attr('y', NB.h + 2).attr('text-anchor', 'middle').attr('font-size', 11).attr('fill', muted()).text('cosine similarity (−1 to 1)');
    const sameN = nb.filter((q) => GENES[q.j].prog === GENES[gSel].prog).length;
    s1.set({ 'selected gene': gname(gSel), 'of its 5 nearest, same programme': `${sameN} / 5`, 'all genes: nearest 4 in own programme': pc(nbHit.obs), 'expected by chance': pc(nbHit.chance), 'cells whose nearest cell has the same type': pc(cellNN) });
  }
  paint1();
  prose(rb, `<p>Across all genes, <b>${pc(nbHit.obs)}</b> of the nearest neighbours lie in the same programme, against ${pc(nbHit.chance)} expected if vectors were random. Among cells, <b>${pc(cellNN)}</b> have a nearest neighbour of the same type (the most common type accounts for ${pc(Math.max(...CTYPES.map((_, t) => typeShare(t))))} of cells). The type placed least reliably is ${CTYPES[worst].name} (${pc(nnBy[worst])} of its cells have a same-type nearest neighbour).</p>`);

  // ---- Step 2: attention worked example
  const T4 = ['CD8A', 'GZMB', 'COL1A1', 'ACTB'], X4 = T4.map((g) => XY[GI[g]]);
  const WV = [[0.8, 0.3], [-0.3, 0.8]];
  rb = rung(ch, 2, 'Attention: queries, keys and values on four genes',
    `<p>Take one cell in which four genes are detected: ${T4.join(', ')}. Each gene has a 2-number vector <b>x</b> (its two principal-component coordinates from step 1, so every matrix below is small enough to read). Three matrices turn each x into three vectors:</p>
     <ul><li><b>query</b> q = x·W<sub>Q</sub>: what this gene is looking for;</li><li><b>key</b> k = x·W<sub>K</sub>: what this gene offers to others;</li><li><b>value</b> v = x·W<sub>V</sub>: what this gene passes on if it is chosen.</li></ul>
     <p>The <b>score</b> of gene <i>i</i> for gene <i>j</i> is the dot product q<sub>i</sub>·k<sub>j</sub> (divided by √2, the square root of the vector length, to keep scores moderate). A <b>softmax</b> turns each row of scores into positive weights that sum to 1. The new vector for gene <i>i</i> is the weighted sum of all values. W<sub>K</sub> is the identity here and W<sub>V</sub> is fixed; the rotation angle of W<sub>Q</sub> is adjustable. In a real model these matrices are learned in step 5.</p>`);
  const f2g = figure(rb, 'One attention head, with every number shown',
    'Top: the embedding matrix X and the query, key and value matrices (rows = genes). Bottom: scores, attention weights (each row sums to 1) and outputs. The outlined row is the selected query gene. Hover a cell for details.');
  let qSel = 0, theta = 0, useRoot = true;
  select(f2g.controls, { label: 'Query gene', value: '0', options: T4.map((g, i) => [String(i), g]), onChange: (v) => { qSel = +v; paint2(); } });
  slider(f2g.controls, { label: 'W_Q rotation', min: 0, max: 180, step: 5, value: 0, fmt: (v) => v + '°', onInput: (v) => { theta = v; paint2(); } });
  checkbox(f2g.controls, { label: 'divide scores by √2', checked: true, onChange: (v) => { useRoot = v; paint2(); } });
  const s2 = stats(f2g.root);
  const svg2 = d3.select(f2g.body).append('svg').attr('viewBox', '0 0 820 352').attr('width', 820).attr('role', 'img').attr('aria-label', 'attention worked example');
  const calc2 = worked(rb, 'Following the query gene through the head', '');
  const att4 = () => {
    const a = theta * Math.PI / 180, WQ = [[Math.cos(a), Math.sin(a)], [-Math.sin(a), Math.cos(a)]];
    const Q = mm(X4, WQ), K = X4, Vv = mm(X4, WV);
    const S = mm(Q, tr(K)).map((r) => r.map((v) => (useRoot ? v / Math.SQRT2 : v)));
    const A = S.map(softmaxRow), O = mm(A, Vv);
    return { WQ, Q, K, V: Vv, S, A, O };
  };
  function paint2() {
    const R = att4(), div = diverging(2.2);
    svg2.selectAll('*').remove();
    const gs = ['x', 'q', 'k', 'v'], mats = [X4, R.Q, R.K, R.V], ttl = ['X (embeddings)', 'Q = X·W_Q', 'K = X·W_K', 'V = X·W_V (W_V fixed)'];
    mats.forEach((M, i) => matrix(svg2, M, 90 + i * 190, 44, { cw: 54, chh: 28, color: div, rows: i === 0 ? T4 : null, cols: ['d1', 'd2'], title: ttl[i], hi: qSel, tipf: (r, c, v) => `${ttl[i]}<br>${T4[r]}, ${['d1', 'd2'][c]}: <b>${f2(v)}</b>` }));
    matrix(svg2, R.S, 90, 230, { cw: 54, chh: 28, color: diverging(4), rows: T4, cols: T4, title: 'scores q·k' + (useRoot ? ' / √2' : ''), hi: qSel, tipf: (r, c, v) => `score ${T4[r]} → ${T4[c]}<br>${f2(v)}` });
    matrix(svg2, R.A, 90 + 280, 230, { cw: 54, chh: 28, color: sequential(0, 1), rows: null, cols: T4, title: 'weights = softmax of each row', hi: qSel, tipf: (r, c, v) => `weight ${T4[r]} → ${T4[c]}<br><b>${f2(v)}</b>` });
    matrix(svg2, R.O, 90 + 560, 230, { cw: 54, chh: 28, color: div, rows: null, cols: ['d1', 'd2'], title: 'output = weights · V', hi: qSel, tipf: (r, c, v) => `output for ${T4[r]}, ${['d1', 'd2'][c]}: <b>${f2(v)}</b>` });
    svg2.append('text').attr('x', 90 + 4 * 54 + 16).attr('y', 230 + 2 * 28 + 4).attr('font-size', 18).attr('fill', ink2()).text('→');
    svg2.append('text').attr('x', 90 + 280 + 4 * 54 + 16).attr('y', 230 + 2 * 28 + 4).attr('font-size', 13).attr('fill', ink2()).text('× V →');
    const w = R.A[qSel], q = R.Q[qSel], top = w.indexOf(Math.max(...w));
    const self = w[qSel], uni = 1 / T4.length;
    s2.set({ 'query': T4[qSel], 'largest weight': `${T4[top]} (${f2(w[top])})`, 'weight on itself': f2(self), 'weight if all equal': f2(uni), 'output vs own value': `[${R.O[qSel].map(f2).join(', ')}] vs [${R.V[qSel].map(f2).join(', ')}]` });
    const j = (top === qSel ? (qSel + 1) % 4 : top);
    calc2.querySelector('.prose').innerHTML = `<p>Query <b>${T4[qSel]}</b>: q = [${q.map(f2).join(', ')}]. Score against <b>${T4[j]}</b>: k = [${R.K[j].map(f2).join(', ')}], so q·k${useRoot ? ' / √2' : ''} = (${f2(q[0])}×${f2(R.K[j][0])} + ${f2(q[1])}×${f2(R.K[j][1])})${useRoot ? ' / 1.41' : ''} = <b>${f2(R.S[qSel][j])}</b>.</p>
      <p>Softmax of the four scores [${R.S[qSel].map(f2).join(', ')}] gives weights [${w.map(f2).join(', ')}] (sum ${f2(sum(w))}). Output = ${w.map((v, i) => `${f2(v)}·v<sub>${T4[i]}</sub>`).join(' + ')} = [${R.O[qSel].map(f2).join(', ')}].</p>`;
  }
  paint2();
  const R0 = (() => { const sv = theta; theta = 0; const r = att4(); theta = sv; return r; })();
  prose(rb, `<p>At 0° the score is the (scaled) dot product of the two embeddings, so attention resembles similarity: CD8A gives ${f2(R0.A[0][1])} of its weight to GZMB, ${f2(R0.A[0][0])} to itself and ${f2(R0.A[0][2])} to COL1A1. Rotating W<sub>Q</sub> breaks this equivalence: queries and keys are different projections, so a gene can attend to partners that are <i>not</i> like itself. Dot products also grow with vector length, so genes with large vectors tend to receive more attention regardless of direction. Without the √2 division the weights become sharper; with long vectors this saturates the softmax and slows training, which is the reason for the division.</p>`);
  quiz(rb, 'If all four scores in a row were equal, what would the output vector be?', '<p>The weights would be ¼ each, so the output is the plain average of the four value vectors. Attention interpolates between "copy one token" (one weight near 1) and "average everything" (equal weights).</p>');

  // ---- Step 3: heads and position
  const SENT = ['CD8A', 'COL1A1', 'GZMB', 'MKI67', 'IFNG', 'ACTB', 'CXCL9', 'TOP2A'].map((g) => GI[g]);
  rb = rung(ch, 3, 'Several heads, and where position enters',
    `<p>A transformer layer runs several <b>heads</b> in parallel, each with its own W<sub>Q</sub>, W<sub>K</sub> and W<sub>V</sub>. Their outputs are concatenated and mixed by a further matrix W<sub>O</sub>. Three heads are shown on an eight-gene cell. Heads 1 and 2 have hand-set weights, to show two relationships a trained head can encode; head 3 has random weights, as an untrained head would.</p>
     <ul><li><b>Head 1 (content):</b> compares gene embeddings, so genes of the same programme attend to each other.</li>
     <li><b>Head 2 (previous token):</b> uses a <b>positional code</b>, a vector of sines and cosines of the position that is added to each token. Without it attention sees an unordered set.</li>
     <li><b>Head 3 (random):</b> a random projection, usually diffuse.</li></ul>`);
  const f3g = figure(rb, 'Three heads on the same eight genes',
    'Each panel is an 8 × 8 attention matrix (row = the gene that is updated, column = the gene it reads from; rows sum to 1). Tick marks beside names show programme colours. Click a row label to outline that query. Shuffling the gene order moves the content head\'s pattern with the genes but leaves the previous-token stripe in place.');
  let order = SENT.slice(), usePos = true, q3 = 0;
  const shuf = rng(99);
  checkbox(f3g.controls, { label: 'add positional code', checked: true, onChange: (v) => { usePos = v; paint3(); } });
  button(f3g.controls, 'Shuffle gene order', () => { order = shuf.shuffle(order); paint3(); });
  button(f3g.controls, 'Original order', () => { order = SENT.slice(); paint3(); });
  const s3 = stats(f3g.root);
  const svg3 = d3.select(f3g.body).append('svg').attr('viewBox', '0 0 830 330').attr('width', 830).attr('role', 'img').attr('aria-label', 'three attention heads');
  const rr3 = rng(17), RQ = range(6).map(() => [rr3.normal(0, 0.7), rr3.normal(0, 0.7)]), RK = range(6).map(() => [rr3.normal(0, 0.7), rr3.normal(0, 0.7)]);
  const posCode = (i) => [Math.sin(0.9 * i), Math.cos(0.9 * i), Math.sin(0.35 * i), Math.cos(0.35 * i)];
  const heads3 = () => {
    const n = order.length, c = order.map((g) => XY[g]), P = range(n).map((i) => (usePos ? posCode(i) : [0, 0, 0, 0]));
    const h1 = c.map((a) => c.map((b) => 1.8 * 1.8 * dotp(a, b) / Math.SQRT2));
    const h2 = range(n).map((i) => range(n).map((j) => 10 * dotp(usePos ? posCode(i - 1) : [0, 0, 0, 0], P[j]) / 2));
    const x6 = c.map((a, i) => [...a, ...P[i]]);
    const h3 = x6.map((a) => x6.map((b) => dotp(mm([a], RQ)[0], mm([b], RK)[0]) / Math.SQRT2));
    return [h1, h2, h3].map((S) => S.map(softmaxRow));
  };
  function paint3() {
    svg3.selectAll('*').remove();
    const Hs = heads3(), names = ['Head 1: content', 'Head 2: previous token', 'Head 3: random'], cell = 26, n = order.length;
    Hs.forEach((A, h) => matrix(svg3, A, 62 + h * 268, 76, { cw: cell, chh: cell, color: sequential(0, 1), fmt: (v) => (v >= 0.1 ? v.toFixed(1).replace(/^0/, '') : ''), rows: h === 0 ? order.map(gname) : order.map(gname), cols: order.map(gname), title: names[h], hi: q3, rowClick: (i) => { q3 = i; paint3(); }, tipf: (r, c, v) => `${names[h]}<br>${gname(order[r])} reads from ${gname(order[c])}: <b>${f3(v)}</b>`, rowColour: (i) => cat(GENES[order[i]].prog), colColour: (j) => cat(GENES[order[j]].prog), colLabelsRot: true }));
    const same = (A) => mean(A.map((r, i) => sum(r.map((v, j) => (GENES[order[i]].prog === GENES[order[j]].prog ? v : 0)))));
    const prev = (A) => mean(A.slice(1).map((r, i) => r[i]));
    const o = {};
    Hs.forEach((A, h) => { o[`head ${h + 1}: weight on same-programme genes`] = f2(same(A)); o[`head ${h + 1}: weight on previous gene`] = f2(prev(A)); });
    s3.set(o);
  }
  paint3();
  const H3 = heads3(); const sameH = (A) => mean(A.map((r, i) => sum(r.map((v, j) => (GENES[order[i]].prog === GENES[order[j]].prog ? v : 0)))));
  const prevH = (A) => mean(A.slice(1).map((r, i) => r[i]));
  const sameNull = mean(order.map((g, i) => order.filter((h) => GENES[h].prog === GENES[g].prog).length / order.length));
  prose(rb, `<p>With the original order and positional code, head 1 puts <b>${f2(sameH(H3[0]))}</b> of its weight on same-programme genes (uniform weights would give ${f2(sameNull)}), and head 2 puts <b>${f2(prevH(H3[1]))}</b> on the preceding gene (uniform: ${f2(1 / 8)}). The random head 3 gives ${f2(sameH(H3[2]))} and ${f2(prevH(H3[2]))}: near the uniform values, with no clear pattern. Training is what turns diffuse heads into sharp ones. Unticking the positional code makes head 2 uniform, because the model then has no information about order; <b>attention on its own treats the input as an unordered set</b>. That suits a set of detected genes, but not a DNA or protein sequence, where position must be added.</p>`);

  // ---- Step 4: the block
  rb = rung(ch, 4, 'The transformer block and its size',
    `<p>One <b>block</b> applies, in order: multi-head attention (tokens exchange information), a <b>residual connection</b> (the input is added back to the output, so a block only has to learn a change) with normalisation, a small <b>feed-forward network</b> applied to each token separately, and another residual connection. Blocks are stacked, and every token keeps its own vector throughout. The last vectors feed a task head, for example masked-gene prediction, or are pooled into a cell embedding.</p>`);
  const f4g = figure(rb, 'Block diagram and parameter count',
    'Left: one pass through the model. Right: parameters by component for a chosen size. A block has about 4d² attention parameters (W_Q, W_K, W_V, W_O, each d × d) and 8d² feed-forward parameters (d → 4d → d); the embedding table has V × d. Biases and normalisation weights are ignored. The vocabulary is assumed to be 20,000 genes.');
  const DS = [64, 128, 256, 512, 768, 1024], HS = [1, 2, 4, 8, 16], NS = [256, 512, 1024, 2048, 4096];
  const cfg = { d: 256, L: 6, h: 8, n: 2048 }, VOC = 20000;
  select(f4g.controls, { label: 'Vector length d', value: String(cfg.d), options: DS.map((v) => [String(v), String(v)]), onChange: (v) => { cfg.d = +v; paint4(); } });
  slider(f4g.controls, { label: 'Blocks L', min: 1, max: 24, step: 1, value: cfg.L, fmt: (v) => v, onInput: (v) => { cfg.L = v; paint4(); } });
  select(f4g.controls, { label: 'Heads h', value: String(cfg.h), options: HS.map((v) => [String(v), String(v)]), onChange: (v) => { cfg.h = +v; paint4(); } });
  select(f4g.controls, { label: 'Genes per cell n', value: String(cfg.n), options: NS.map((v) => [String(v), String(v)]), onChange: (v) => { cfg.n = +v; paint4(); } });
  const s4 = stats(f4g.root);
  const row4 = el('div', { class: 'fig-row' }); f4g.body.append(row4);
  const c4a = el('div'), c4b = el('div'); row4.append(c4a, c4b);
  blockDiagram(c4a);
  const B4 = frame(c4b, { width: 400, height: 300, x: [0, 1], y: [0, 1], axes: false, grid: false, margin: { left: 20, right: 20, top: 40, bottom: 20 } });
  function paint4() {
    const { d, L, h, n } = cfg, emb = VOC * d, att = L * 4 * d * d, ffn = L * 8 * d * d, tot = emb + att + ffn, hd = d / h;
    const parts = [['Embedding table', emb, cat(2)], ['Attention', att, cat(0)], ['Feed-forward', ffn, cat(1)]];
    B4.g.selectAll('*').remove();
    B4.g.append('text').attr('x', 0).attr('y', -18).attr('font-size', 12).attr('fill', ink2()).text('Parameters by component');
    const xs = d3.scaleLinear().domain([0, tot]).range([0, B4.w]);
    let acc = 0;
    parts.forEach(([nm, v, col], i) => {
      B4.g.append('rect').attr('x', xs(acc)).attr('width', Math.max(1, xs(v))).attr('y', 0).attr('height', 40).attr('fill', col).attr('stroke', css('--surface'))
        .on('mousemove', (e) => tip(e, `${nm}: ${fmtN(v)} (${pc(v / tot)})`)).on('mouseleave', untip);
      B4.g.append('text').attr('x', 0).attr('y', 74 + i * 22).attr('font-size', 12).attr('fill', ink2()).text(`${nm}: ${fmtN(v)} (${pc(v / tot)})`);
      B4.g.append('rect').attr('x', -16).attr('y', 63 + i * 22).attr('width', 10).attr('height', 10).attr('fill', col);
      acc += v;
    });
    [`Attention scores per cell: h × L × n²`, `= ${h} × ${L} × ${n.toLocaleString()}² = ${fmtN(h * L * n * n)}`, `Doubling n multiplies this by 4.`, `Each head works in ${hd} dimensions (d / h).`].forEach((t, i) => B4.g.append('text').attr('x', 0).attr('y', 160 + i * 20).attr('font-size', 12).attr('fill', ink2()).text(t));
    s4.set({ 'total parameters': fmtN(tot), 'in blocks': pc((att + ffn) / tot), 'per-head dimension': hd, 'scores per head per block': fmtN(n * n) });
  }
  paint4();
  prose(rb, `<p>Two consequences follow. First, the cost of attention grows with the <b>square of the number of tokens</b>: scoring every pair of ${NS[3].toLocaleString()} genes needs ${fmtN(NS[3] ** 2)} scores per head per block, which is why single-cell models cap the input length (for example by keeping the top-ranked genes). Second, the same block is reused for every task. Only the input encoding and the output head change.</p>`);
  quiz(rb, 'Why is a residual connection useful in a deep stack of blocks?', '<p>Each block outputs "input + change". Information and gradients can pass through unchanged, so many blocks can be stacked and trained, and a block that has nothing useful to add can leave the vectors almost untouched.</p>');

  // ---- Step 5: masked pretraining
  rb = rung(ch, 5, 'Self-supervised pretraining by masking',
    `<p>Hide a fraction of the detected genes in each cell and train the model to predict them from the visible genes. The loss is the cross-entropy over all ${G} genes, so a model that guesses uniformly scores ln ${G} = <b>${f2(chance)}</b>. Predicting the hidden gene needs knowledge of which genes co-occur, so the embedding table is shaped by the task. No clinical or cell-type labels are used.</p>
     <p><b>Teaching toy.</b> To keep training live in the browser, the model replaces attention blocks with a single mean over the visible genes' vectors followed by a linear softmax. The objective is the one used by masked-gene transformers; the architecture is not. It is trained on ${TRN.length} cells and evaluated on ${VAL.length} held-out cells with fixed masks.</p>`);
  const f5g = figure(rb, 'Masked-gene pretraining, live',
    'Left: gene vectors during training, projected onto the first two principal components of the final model so the map stays fixed. Right: held-out loss against the baselines. Below: one held-out cell with some genes masked and the model\'s top three guesses for the first masked gene.');
  let d5 = D0, frac5 = MASK, M5 = null, hist5 = [], stop5 = null, ex5 = 0, basis5 = null;
  select(f5g.controls, { label: 'Vector length', value: String(d5), options: [['2', '2'], ['4', '4'], ['8', '8']], onChange: (v) => { d5 = +v; reset5(); } });
  slider(f5g.controls, { label: 'Fraction masked', min: 0.1, max: 0.6, step: 0.1, value: frac5, fmt: (v) => pc(v), onInput: (v) => { frac5 = v; reset5(); } });
  button(f5g.controls, 'Train', () => run5(), true);
  button(f5g.controls, 'Pause', () => halt5());
  button(f5g.controls, 'Reset', () => reset5());
  button(f5g.controls, 'Next example cell', () => { ex5 = (ex5 + 1) % VAL.length; paintEx(); });
  const s5 = stats(f5g.root);
  const row5 = el('div', { class: 'fig-row' }); f5g.body.append(row5);
  const c5a = el('div'), c5b = el('div'); row5.append(c5a, c5b);
  const F5 = frame(c5a, { width: 460, height: 340, x: [-2.8, 2.8], y: [-2.8, 2.8], xLabel: 'component 1 (final-model axes)', yLabel: 'component 2' });
  const L5 = frame(c5b, { width: 460, height: 340, x: [0, EPOCHS], y: [2.6, 3.3], xLabel: 'epoch', yLabel: 'held-out masked-gene loss' });
  const pts5 = F5.layer('pts');
  [['uniform guess (ln 24)', chance, ink2()], ['gene-frequency baseline', uni, muted()]].forEach(([t, v, c], i) => {
    L5.layer('b').append('line').attr('x1', 0).attr('x2', L5.w).attr('y1', L5.y(v)).attr('y2', L5.y(v)).attr('stroke', c).attr('stroke-dasharray', '5 4');
    L5.layer('b').append('text').attr('x', L5.w - 4).attr('text-anchor', 'end').attr('y', L5.y(v) + (i ? 14 : -5)).attr('font-size', 11).attr('fill', ink2()).text(`${t} ${f2(v)}`);
  });
  const lossPath = L5.layer('l').append('path').attr('fill', 'none').attr('stroke', cat(6)).attr('stroke-width', 2.5);
  const lossDot = L5.layer('l').append('circle').attr('r', 5).attr('fill', cat(6)).attr('stroke', css('--surface')).attr('stroke-width', 1.5);
  legend(f5g.body, PROGS.map((p, i) => [p.name, cat(i)]));
  const exBox = el('div', { class: 'prose' }); f5g.body.append(exBox);
  function reset5() {
    halt5(); M5 = newModel(11, d5); basis5 = pca(pretrain(d5, frac5).E, { scale: false });
    const sc = sd(basis5.S.map((r) => r[0])); basis5.sc = sc;
    hist5 = [{ ep: 0, ...evaluate(M5, VAL, frac5) }]; paint5();
  }
  function halt5() { if (stop5) { stop5(); stop5 = null; } }
  function run5() {
    if (stop5) return; if (M5.epoch >= EPOCHS) reset5();
    let acc = 0;
    stop5 = animate((dt) => {
      acc += dt;
      while (acc >= 60 && M5.epoch < EPOCHS) { acc -= 60; trainEpoch(M5, TRN, frac5); hist5.push({ ep: M5.epoch, ...evaluate(M5, VAL, frac5) }); }
      paint5();
      if (M5.epoch >= EPOCHS) { stop5 = null; return false; }
    });
  }
  function paint5() {
    const pos = M5.E.map((r) => { const s = basis5.project(r); return [s[0] / basis5.sc, s[1] / basis5.sc]; });
    pts5.selectAll('circle').data(GENES).join('circle').attr('cx', (g, i) => F5.x(Math.max(-2.7, Math.min(2.7, pos[i][0])))).attr('cy', (g, i) => F5.y(Math.max(-2.7, Math.min(2.7, pos[i][1])))).attr('r', 6.5).attr('fill', (g) => cat(g.prog)).attr('stroke', css('--surface')).attr('stroke-width', 1.5)
      .on('mousemove', (e, g) => tip(e, `<b>${g.name}</b> (${PROGS[g.prog].name})`)).on('mouseleave', untip);
    lossPath.attr('d', d3.line().x((h) => L5.x(h.ep)).y((h) => L5.y(Math.min(3.3, h.loss)))(hist5));
    const last = hist5[hist5.length - 1];
    lossDot.attr('cx', L5.x(last.ep)).attr('cy', L5.y(Math.min(3.3, last.loss)));
    s5.set({ epoch: `${M5.epoch} / ${EPOCHS}`, 'held-out loss': f3(last.loss), 'top-1 accuracy': pc(last.top1), 'top guess in correct programme': pc(last.prog), 'parameters (E and U)': fmtN(2 * G * d5 + G) });
    paintEx();
  }
  function paintEx() {
    const cl = VAL[ex5], r = rng(500 + ex5), { masked, vis } = pick(cl.toks, frac5, r), p = probsOf(M5, vis);
    const top = range(G).sort((a, b) => p[b] - p[a]).slice(0, 3), m0 = masked[0];
    const chip = (t, m) => `<span style="display:inline-block;margin:2px 3px;padding:1px 7px;border-radius:6px;border:1px solid ${m ? ink() : css('--axis')};background:${m ? css('--surface-2') : 'transparent'};font-family:${css('--mono')};font-size:12px">${m ? '[MASK]' : gname(t)}</span>`;
    exBox.innerHTML = `<p><b>Example</b> (${CTYPES[cl.type].name}): ${cl.toks.map((t) => chip(t, masked.includes(t))).join('')}<br>Hidden gene${masked.length > 1 ? 's' : ''}: ${masked.map(gname).join(', ')}. Top guesses for the first: ${top.map((g) => `<b>${gname(g)}</b> ${pc(p[g])}${g === m0 ? ' ✓' : ''}`).join(', ')}; the true gene ${gname(m0)} is given ${pc(p[m0])}.</p>`;
  }
  reset5();
  const vl = [2, 4, 8].map((d) => { const M = pretrain(d, MASK), ws = []; GENES.forEach((a, i) => GENES.forEach((b, j) => { if (j > i && a.prog < 4 && a.prog === b.prog) ws.push(cosine(M.E[i], M.E[j])); })); return { loss: evaluate(M, VAL, MASK).loss, w: mean(ws) }; });
  prose(rb, `<p>After ${EPOCHS} epochs (default settings) the held-out loss is <b>${f2(baseEval.loss)}</b>, against ${f2(uni)} for a model that only knows how common each gene is and ${f2(chance)} for a uniform guess. The top guess falls in the correct programme for <b>${pc(baseEval.prog)}</b> of hidden programme genes (about ${pc(1 / 4)} if the guess were random among four programmes), but it names the exact gene only ${pc(baseEval.top1)} of the time. This is expected: genes within a programme are interchangeable in this corpus, so the best possible model cannot tell them apart. Vector length changes the loss and the structure of the vectors in different directions: held-out loss is ${f2(vl[0].loss)}, ${f2(vl[1].loss)} and ${f2(vl[2].loss)} at lengths 2, 4 and 8, while the mean within-programme cosine similarity of the gene vectors is ${f2(vl[0].w)}, ${f2(vl[1].w)} and ${f2(vl[2].w)}. A lower loss therefore does not imply vectors that are easier to interpret. A real masked model is judged by what its embeddings enable downstream (step 6), not by this loss.</p>`);
  table(rb, [
    { t: 'Press <b>Train</b> and watch the map', s: 'Genes start as one blob; programmes separate as the loss falls.', m: 'The embedding structure is a by-product of predicting masked tokens.' },
    { t: 'Vector length <b>2</b> versus <b>8</b>', s: `Final held-out loss ${f2(vl[0].loss)} versus ${f2(vl[2].loss)}; within-programme cosine ${f2(vl[0].w)} versus ${f2(vl[2].w)}`, m: 'A lower loss and clearly grouped vectors need not go together: the loss measures prediction, not how the vectors are organised.' },
    { t: 'Fraction masked <b>10%</b> versus <b>60%</b>', s: `Loss ${f2(evaluate(pretrain(D0, 0.1), VAL, 0.1).loss)} versus ${f2(evaluate(pretrain(D0, 0.6), VAL, 0.6).loss)} (different tasks, so not strictly comparable)`, m: 'Heavier masking is a harder task with less context; the choice is a design decision.' },
  ], [['t', 'Try'], ['s', 'What you see'], ['m', 'What it means']], { cls: 'compact' });

  // ---- Step 6: lab
  const FEAT7 = ['ifng', 'log2tmb', 'caf', 'pdl1', 'cd8', 'prolif', 'age'];
  const rowsOf = (pts, ks) => pts.map((p) => ks.map((k) => FEATURES[k].get(p)));
  const allP = C.all, allX = rowsOf(allP, FEAT7), PC7 = pca(allX, { scale: true });
  const whiten = (r) => PC7.project(r).map((v, j) => v / Math.sqrt(PC7.vals[j]));
  const stD = standardize(rowsOf(D, FEAT7)), stT = standardize(rowsOf(D, ['ifng', 'log2tmb', 'caf']));
  const rep = { raw: { name: 'all 7 measurements', f: (p) => stD.apply(rowsOf([p], FEAT7)[0]) }, true3: { name: 'true three (IFN-γ, TMB, CAF)', f: (p) => stT.apply(rowsOf([p], ['ifng', 'log2tmb', 'caf'])[0]) } };
  const Dz = {}, Vz = {};
  const prep = (key, f) => { Dz[key] = D.map(f); Vz[key] = V.map(f); };
  prep('raw', rep.raw.f); prep('true3', rep.true3.f);
  const Dw = D.map((p) => whiten(rowsOf([p], FEAT7)[0])), Vw = V.map((p) => whiten(rowsOf([p], FEAT7)[0]));
  const NL = [10, 20, 40, 80, 120, 180], REPS = 12;
  const perms = range(REPS).map((i) => rng(100 + i).shuffle(range(D.length)));
  const score = (ZD, ZV, n, rp) => {
    const idx = perms[rp].slice(0, n), yy = idx.map((i) => y[i]), np = sum(yy);
    if (np < 2 || n - np < 2) return NaN;
    const fit = fitLogistic(idx.map((i) => ZD[i]), yy, { l2: 1 });
    return auc(ZV.map((r) => predictProb(r, fit.b0, fit.w)), yv);
  };
  const curve = (ZD, ZV) => NL.map((n) => { const v = range(REPS).map((r) => score(ZD, ZV, n, r)).filter((a) => !Number.isNaN(a)); return { n, m: mean(v), s: sd(v), k: v.length }; });
  const cRaw = curve(Dz.raw, Vz.raw), cTrue = curve(Dz.true3, Vz.true3);
  const cPC = {}; const pcCurve = (k) => cPC[k] || (cPC[k] = curve(Dw.map((r) => r.slice(0, k)), Vw.map((r) => r.slice(0, k))));
  const atN = (c, n) => c.find((q) => q.n === n).m;

  rb = rung(ch, 6, 'Lab: a representation learned without labels, then few labelled patients',
    `<p>Foundation models follow a two-stage recipe: learn a representation from many unlabelled samples, then train a small model on the few labelled ones. The lab reproduces the recipe on the patient cohort. The representation is <b>principal components of the 7 measurements, computed on all ${allP.length} patients without using response</b>. This is a linear stand-in for a pretrained encoder: it is the simplest embedding that needs no labels. A logistic regression with a ridge penalty is then fitted on <i>n</i> labelled discovery patients (average over ${REPS} random draws) and evaluated on the ${V.length} Centre C patients.</p>
     <p class="fig-caption">The principal components use the measurements of the Centre C patients too (not their response). This is acceptable for a pretraining corpus, but it is transductive: pretraining data that overlap the evaluation set can inflate results when their labels are related to the features.</p>`);
  const flab = figure(rb, 'Lab: embedding dimensions versus number of labelled patients',
    'Left: all patients in the first two principal components (filled = discovery, open ring = Centre C), coloured by tumour phenotype. Right: validation AUC on Centre C against the number of labelled patients, for the raw measurements, the first k components, and the three truly informative measurements (a reference that is not available in practice). Bands are ± 1 sd over the random draws.');
  let kDim = 2, nSel = 20;
  slider(flab.controls, { label: 'Embedding dimensions k', min: 1, max: 7, step: 1, value: kDim, fmt: (v) => v, onInput: (v) => { kDim = v; paintL(); } });
  slider(flab.controls, { label: 'Labelled patients n (readout)', min: 0, max: NL.length - 1, step: 1, value: 1, fmt: (i) => NL[i], onInput: (i) => { nSel = NL[i]; paintL(); } });
  const sL = stats(flab.root);
  const rowL = el('div', { class: 'fig-row' }); flab.body.append(rowL);
  const la = el('div'), lb = el('div'); rowL.append(la, lb);
  const PCs = PC7.S, e1 = d3.extent(PCs, (r) => r[0]), e2 = d3.extent(PCs, (r) => r[1]);
  const FA = frame(la, { width: 440, height: 360, x: [e1[0] - 0.3, e1[1] + 0.3], y: [e2[0] - 0.3, e2[1] + 0.3], xLabel: `PC1 (${pc(PC7.ratio[0])} of variance)`, yLabel: `PC2 (${pc(PC7.ratio[1])})` });
  const PH = ['inflamed', 'excluded', 'desert'];
  FA.layer('p').selectAll('circle').data(allP).join('circle').attr('cx', (p, i) => FA.x(PCs[i][0])).attr('cy', (p, i) => FA.y(PCs[i][1])).attr('r', 4)
    .attr('fill', (p) => (p.cohort === 'discovery' ? cat(PH.indexOf(p.phenotype)) : 'none')).attr('fill-opacity', 0.75).attr('stroke', (p) => (p.cohort === 'discovery' ? css('--surface') : cat(PH.indexOf(p.phenotype)))).attr('stroke-width', (p) => (p.cohort === 'discovery' ? 0.8 : 1.6))
    .on('mousemove', (e, p) => tip(e, `${p.id} (${p.cohort}, ${p.centre})<br>${p.phenotype}, ${p.response ? 'responded' : 'no response'}<br>IFN-γ ${p.ifng}, CD8 ${p.cd8}%`)).on('mouseleave', untip);
  legend(la, PH.map((p, i) => [p, cat(i)]));
  const FB = frame(lb, { width: 500, height: 360, x: [0, 185], y: [0.6, 0.93], xLabel: 'labelled patients n', yLabel: 'validation AUC (Centre C)' });
  const lineG = FB.layer('lines');
  legend(lb, [['all 7 measurements', cat(0)], ['first k components', cat(6)], ['true three (reference)', cat(2)]]);
  const msg = html('', 'p', 'fig-caption'); flab.body.append(msg);
  function paintL() {
    const cP = pcCurve(kDim), series = [[cRaw, cat(0), '5 4'], [cP, cat(6), null], [cTrue, cat(2), '2 3']];
    lineG.selectAll('*').remove();
    series.forEach(([c, col, dash]) => {
      lineG.append('path').datum(c).attr('fill', col).attr('fill-opacity', 0.12).attr('d', d3.area().x((q) => FB.x(q.n)).y0((q) => FB.y(q.m - q.s)).y1((q) => FB.y(q.m + q.s)));
      const p = lineG.append('path').datum(c).attr('fill', 'none').attr('stroke', col).attr('stroke-width', 2.4).attr('d', d3.line().x((q) => FB.x(q.n)).y((q) => FB.y(q.m)));
      if (dash) p.attr('stroke-dasharray', dash);
      lineG.selectAll(null).data(c).join('circle').attr('cx', (q) => FB.x(q.n)).attr('cy', (q) => FB.y(q.m)).attr('r', 4).attr('fill', col).attr('stroke', css('--surface')).attr('stroke-width', 1.2)
        .on('mousemove', (e, q) => tip(e, `n = ${q.n}<br>AUC ${f3(q.m)} ± ${f3(q.s)} (${q.k} draws)`)).on('mouseleave', untip);
    });
    lineG.append('line').attr('x1', FB.x(nSel)).attr('x2', FB.x(nSel)).attr('y1', 0).attr('y2', FB.h).attr('stroke', muted()).attr('stroke-dasharray', '3 3');
    sL.set({ [`n = ${nSel}: all 7 measurements`]: f3(atN(cRaw, nSel)), [`first ${kDim} component${kDim > 1 ? 's' : ''}`]: f3(atN(cP, nSel)), 'true three': f3(atN(cTrue, nSel)), 'variance explained by k': pc(sum(PC7.ratio.slice(0, kDim))) });
    msg.innerHTML = '';
  }
  paintL();
  const nnPh = (() => { const Z = PC7.Z; let ok = 0; Z.forEach((v, i) => { let b = Infinity, bj = -1; Z.forEach((w, j) => { if (j !== i) { const dd = sum(v.map((a, k) => (a - w[k]) ** 2)); if (dd < b) { b = dd; bj = j; } } }); if (allP[bj].phenotype === allP[i].phenotype) ok++; }); return ok / Z.length; })();
  const phShare = PH.map((p) => allP.filter((q) => q.phenotype === p).length / allP.length), phChance = sum(phShare.map((s) => s * s));
  const cmp = (a, b) => (a > b + 0.005 ? 'higher than' : a < b - 0.005 ? 'lower than' : 'about the same as');
  const pc2 = pcCurve(2), pc3 = pcCurve(3), pc7c = pcCurve(7), pc1 = pcCurve(1);
  table(rb, [
    { t: 'Look at the <b>PC1–PC2 map</b>', s: `Patients with the same phenotype lie close: ${pc(nnPh)} have a nearest neighbour (in all 7 standardised measurements) of the same phenotype, against ${pc(phChance)} expected by chance. PC1 carries ${pc(PC7.ratio[0])} of the variance.`, m: 'Similar profiles are close in the embedding, the property exploited by cell and patient maps. Phenotype is a summary of immune state, not the response itself.' },
    { t: '<b>n = 10</b>: raw 7 measurements versus first 2 components', s: `AUC ${f3(atN(cRaw, 10))} versus <b>${f3(atN(pc2, 10))}</b> (${cmp(atN(pc2, 10), atN(cRaw, 10))} the raw measurements); true three ${f3(atN(cTrue, 10))}`, m: 'With few labels a model with fewer inputs has less variance. A compressed representation helps only if the retained directions are the informative ones.' },
    { t: '<b>n = 180</b>: raw versus first 2 components', s: `AUC ${f3(atN(cRaw, 180))} versus <b>${f3(atN(pc2, 180))}</b> (${cmp(atN(pc2, 180), atN(cRaw, 180))} the raw measurements); true three ${f3(atN(cTrue, 180))}`, m: 'With ample labels the full measurements catch up or surpass the compressed ones. The unsupervised components discard variation that happens to matter for response.' },
    { t: 'Set <b>k = 1</b>, then <b>k = 3</b>, then <b>k = 7</b> (n = 180)', s: `AUC ${f3(atN(pc1, 180))}, ${f3(atN(pc3, 180))}, ${f3(atN(pc7c, 180))}`, m: 'k = 7 contains the same information as the raw measurements (a rotation); the difference to the raw curve shows only the penalty acting on rotated inputs. Variance explained is not the same as relevance to the outcome.' },
  ], [['t', 'Try'], ['s', 'What you see'], ['m', 'What it means']], { cls: 'compact' });
  prose(rb, `<p><b>Interpreting the result.</b> The reference model that uses only the three truly informative measurements reaches ${f3(atN(cTrue, 180))} at n = 180, which is ${cmp(atN(cTrue, 180), atN(cRaw, 180))} the 7-measurement model (${f3(atN(cRaw, 180))}). PCA is not a pretrained transformer, and these seven measurements are already low-dimensional and informative. Pretraining pays off when the raw input is large (thousands of genes), labels are scarce and the unlabelled corpus is much larger than the labelled set. Whether a pretrained single-cell model beats PCA or a simple baseline is an empirical question, discussed on the <a href="#/frontier/foundation-models">foundation models</a> page.</p>`);

  // =====================================================================================
  // 5. PITFALLS
  // =====================================================================================
  ch = chapter(main, 'pitfalls', 'Pitfalls', 'Common problems, how to recognise them, and how to address them.');
  cards(ch, [
    { tag: 'Symptom: a large weight is read as "this gene regulates that gene"', title: 'Attention is not explanation', body: `A large weight shows that a value vector was mixed in, not that it caused the prediction. In step 2 the weights change with W<sub>Q</sub> alone (CD8A → GZMB ${f2(R0.A[0][1])} at 0°) with no change in biology. <b>Fix:</b> use attribution methods, perturbation experiments (mask or delete a gene and measure the change) and independent evidence.` },
    { tag: 'Symptom: results change with gene order', title: 'Position and set structure', body: 'Attention alone is permutation-invariant (step 3). Sequence models need positional codes; set models should not use an arbitrary order such as file order. <b>Fix:</b> use a meaningful order (for example expression rank) or none, and test by shuffling.' },
    { tag: 'Symptom: out-of-memory with long inputs', title: 'Quadratic cost', body: `Scores grow as n²: ${fmtN(NS[3] ** 2)} per head per block at n = ${NS[3].toLocaleString()}. <b>Fix:</b> truncate to informative genes, use windowed or sparse attention, or work on pseudobulks or gene programmes.` },
    { tag: 'Symptom: cells cluster by batch or donor', title: 'Embeddings encode technical variation', body: 'An embedding captures whatever varies most or helps the training task, including batch, depth and ambient RNA. <b>Fix:</b> inspect embeddings coloured by batch and donor, integrate or correct where needed, and evaluate on held-out batches.' },
    { tag: 'Symptom: good pretraining loss, no downstream gain', title: 'Objective mismatch', body: `A low masked-gene loss does not guarantee useful features. In step 5 the loss is bounded because genes within a programme are interchangeable, and in step 6 compression ${atN(pc2, 180) < atN(cRaw, 180) ? 'lowered' : 'did not raise'} the AUC at n = 180. <b>Fix:</b> compare against PCA and a plain model, zero-shot and fine-tuned, on the real task.` },
    { tag: 'Symptom: suspiciously high benchmark scores', title: 'Overlap between pretraining and test data', body: 'Public atlases are reused across studies. If the evaluation cells, donors or labels were in the pretraining corpus, the benchmark is optimistic. <b>Fix:</b> document corpus composition and hold out entire studies.' },
    { tag: 'Symptom: expression levels ignored', title: 'Tokenisation discards information', body: 'Here a gene is a token that is present or absent. Geneformer-style rank encoding and scGPT-style value bins each keep different information about abundance, and the choice affects what can be learned. <b>Fix:</b> know the encoding of the model in use and test whether it preserves the signal of interest.' },
    { tag: 'Symptom: a small cohort and a very large model', title: 'Fitting a transformer to a small table', body: `With ${D.length} patients and ${FEAT7.length} measurements (step 6) there is nothing for a large model to learn that a penalised regression cannot. <b>Fix:</b> match model size to data size and keep a simple baseline in every comparison.` },
  ]);

  // =====================================================================================
  // 6. RECAP
  // =====================================================================================
  ch = chapter(main, 'recap', 'Recap', 'Key points.');
  recap(ch, [
    'An <b>embedding</b> assigns each token (gene, cell, patient) a vector; similar items receive similar vectors, measured by distance or cosine similarity. Unlike one-hot vectors, embeddings encode relatedness.',
    'A cell can be summarised by pooling its genes\' vectors, so a variable set of detected genes becomes one fixed-length vector.',
    '<b>Attention</b> gives each token a query, a key and a value. Weights = softmax(q·k / √d<sub>k</sub>), where d<sub>k</sub> is the length of the key vectors; the output is the weighted sum of values, so each token\'s new vector depends on the other tokens present.',
    'Queries and keys are different projections, so attention is not the same as similarity. Dividing by √d<sub>k</sub> keeps the softmax from saturating.',
    '<b>Heads</b> run in parallel and learn different relationships. Attention alone ignores order; <b>positional codes</b> must be added for sequences.',
    'A <b>transformer block</b> is multi-head attention plus a feed-forward network, each with a residual connection and normalisation. Attention cost grows with n².',
    '<b>Masked pretraining</b> hides tokens and predicts them from context. It needs no labels, and the useful by-product is the embeddings.',
    'Pretrained representations help most when labels are scarce and raw input is high-dimensional. Always compare with PCA and a simple supervised baseline on held-out data, and do not read attention weights as causal evidence.',
  ]);
  prose(ch, '<h3>In code</h3><p>The first block implements one attention head from scratch (the computation of step 2). The second sketches masked pretraining with a standard transformer encoder; single-cell tools wrap the same pieces.</p>');
  code(ch, {
    Python: `import numpy as np, torch, torch.nn as nn

# Step 2: one attention head. X is (n_tokens, d); W_* are (d, d_k)
def attention(X, Wq, Wk, Wv):
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    S = Q @ K.T / np.sqrt(K.shape[1])                    # scores
    A = np.exp(S - S.max(1, keepdims=True)); A /= A.sum(1, keepdims=True)
    return A @ V, A                                      # new vectors, weights

# Steps 4-5: masked-gene pretraining with a small transformer encoder
n_genes, d = 20000, 128
MASK_ID = n_genes                                        # extra token id for [MASK]
embed = nn.Embedding(n_genes + 1, d)
pos = nn.Embedding(2048, d)                              # positional code (learned)
block = nn.TransformerEncoderLayer(d, nhead=8, dim_feedforward=4 * d, batch_first=True)
encoder = nn.TransformerEncoder(block, num_layers=6)
head = nn.Linear(d, n_genes)

def step(tokens, frac=0.15):                             # tokens: (batch, n) gene ids
    mask = torch.rand(tokens.shape) < frac
    x = tokens.masked_fill(mask, MASK_ID)
    h = encoder(embed(x) + pos(torch.arange(tokens.shape[1])))
    loss = nn.functional.cross_entropy(head(h[mask]), tokens[mask])
    return loss                                          # backpropagate, as in any network

# Using a representation: PCA on unlabelled data, then a small supervised model
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
Z = make_pipeline(StandardScaler(), PCA(n_components=3)).fit(X_all)   # no labels used
clf = LogisticRegression(C=1.0).fit(Z.transform(X_labelled), y)
print("validation AUC", roc_auc_score(y_val, clf.predict_proba(Z.transform(X_val))[:, 1]))`,
    R: `# Step 2: one attention head in base R. X is (n_tokens x d)
softmax_rows <- function(S) { E <- exp(S - apply(S, 1, max)); E / rowSums(E) }
attention <- function(X, Wq, Wk, Wv) {
  Q <- X %*% Wq; K <- X %*% Wk; V <- X %*% Wv
  A <- softmax_rows(Q %*% t(K) / sqrt(ncol(K)))          # weights, rows sum to 1
  list(output = A %*% V, weights = A)
}

# Cosine similarity between embedding rows (genes x dims)
cosine <- function(E) { En <- E / sqrt(rowSums(E^2)); En %*% t(En) }

# Using a representation: PCA on unlabelled samples, then a penalised model on labelled ones
pc <- prcomp(X_all, scale. = TRUE)                       # no labels used
Z  <- predict(pc, X_labelled)[, 1:3]
fit <- glmnet::cv.glmnet(Z, y, family = "binomial", alpha = 0)
p <- predict(fit, predict(pc, X_val)[, 1:3], s = "lambda.min", type = "response")
pROC::auc(y_val, as.numeric(p))

# Pretrained single-cell transformers (Geneformer, scGPT) are used from Python;
# their cell embeddings can be exported and analysed in R with the code above.`,
  });
  seeAlso(main, [['frontier', 'foundation-models', 'Single-cell foundation models'], ['frontier', 'virtual-cells', 'Virtual cells and perturbation prediction'], ['concepts', 'neural-network', 'Neural networks and backpropagation'], ['concepts', 'pca', 'PCA'], ['concepts', 'tsne-umap', 't-SNE and UMAP'], ['concepts', 'logistic-regression', 'Logistic regression']]);
}

// ---------------------------------------------------------------------------------------------
function fmtN(v) { return v >= 1e9 ? (v / 1e9).toFixed(2) + ' billion' : v >= 1e6 ? (v / 1e6).toFixed(1) + ' million' : v >= 1e3 ? Math.round(v).toLocaleString() : String(Math.round(v)); }

/** Matrix of numbers drawn as a grid. o: {cw, chh, color, fmt, rows, cols, title, hi, tipf, rowClick, rowColour, colColour, colLabelsRot} */
function matrix(svg, M, x, y, o) {
  const g = svg.append('g').attr('transform', `translate(${x},${y})`), nR = M.length, nC = M[0].length;
  if (o.title) g.append('text').attr('x', 0).attr('y', -(o.cols ? (o.colLabelsRot ? 52 : 24) : 10)).attr('font-size', 12).attr('font-weight', 600).attr('fill', ink2()).text(o.title);
  M.forEach((r, i) => r.forEach((v, j) => {
    const fill = o.color(v), L = d3.hsl(fill).l;
    g.append('rect').attr('x', j * o.cw).attr('y', i * o.chh).attr('width', o.cw - 2).attr('height', o.chh - 2).attr('rx', 2).attr('fill', fill)
      .on('mousemove', (e) => o.tipf && tip(e, o.tipf(i, j, v))).on('mouseleave', untip);
    const t = o.fmt ? o.fmt(v) : f2(v);
    if (t) g.append('text').attr('x', j * o.cw + o.cw / 2 - 1).attr('y', i * o.chh + o.chh / 2 + 3.5).attr('text-anchor', 'middle').attr('font-size', o.cw < 30 ? 9.5 : 11).attr('pointer-events', 'none').attr('fill', L < 0.5 ? css('--surface') : ink()).text(t);
  }));
  if (o.rows) o.rows.forEach((t, i) => {
    const tx = g.append('text').attr('x', -6).attr('y', i * o.chh + o.chh / 2 + 4).attr('text-anchor', 'end').attr('font-size', 10.5).attr('font-weight', i === o.hi ? 700 : 400).attr('fill', ink2()).text(t);
    if (o.rowColour) g.append('rect').attr('x', -o.cw * 0 - 4).attr('y', i * o.chh + 3).attr('width', 2.5).attr('height', o.chh - 8).attr('fill', o.rowColour(i)).attr('transform', 'translate(-0,0)');
    if (o.rowClick) tx.style('cursor', 'pointer').on('click', () => o.rowClick(i));
  });
  if (o.cols) o.cols.forEach((t, j) => {
    const tx = g.append('text').attr('font-size', 10.5).attr('fill', ink2()).text(t);
    if (o.colLabelsRot) tx.attr('transform', `translate(${j * o.cw + o.cw / 2 + 3},-6) rotate(-55)`); else tx.attr('x', j * o.cw + o.cw / 2 - 1).attr('y', -6).attr('text-anchor', 'middle');
  });
  if (o.hi != null && o.hi >= 0) g.append('rect').attr('x', -1).attr('y', o.hi * o.chh - 1).attr('width', nC * o.cw).attr('height', o.chh).attr('fill', 'none').attr('stroke', ink()).attr('stroke-width', 2).attr('rx', 3);
  return g;
}

/** 24 x 24 similarity grid with programme colour bars. */
function simGrid(parent, S, title) {
  const cell = 13, lab = 28, W = lab + G * cell + 6;
  const svg = d3.select(parent).append('svg').attr('viewBox', `0 0 ${W} ${lab + G * cell + 24}`).attr('width', W).attr('role', 'img').attr('aria-label', title);
  svg.append('text').attr('x', lab).attr('y', 14).attr('font-size', 12).attr('font-weight', 600).attr('fill', ink2()).text(title);
  const col = diverging(1), g = svg.append('g').attr('transform', `translate(${lab},${22})`);
  S.forEach((r, i) => r.forEach((v, j) => g.append('rect').attr('x', j * cell).attr('y', i * cell).attr('width', cell - 1).attr('height', cell - 1).attr('fill', col(v))
    .on('mousemove', (e) => tip(e, `${GENES[i].name} vs ${GENES[j].name}<br>cosine <b>${f2(v)}</b>`)).on('mouseleave', untip)));
  GENES.forEach((gn, i) => {
    svg.append('rect').attr('x', lab - 8).attr('y', 22 + i * cell).attr('width', 5).attr('height', cell - 1).attr('fill', cat(gn.prog));
    svg.append('rect').attr('x', lab + i * cell).attr('y', 22 + G * cell + 3).attr('width', cell - 1).attr('height', 5).attr('fill', cat(gn.prog));
  });
}

/** Diagram of a single transformer block. */
function blockDiagram(parent) {
  const W = 400, H = 400, svg = d3.select(parent).append('svg').attr('viewBox', `0 0 ${W} ${H}`).attr('width', W).attr('role', 'img').attr('aria-label', 'transformer block');
  const box = (x, y, w, h, t, fill, sub) => {
    svg.append('rect').attr('x', x).attr('y', y).attr('width', w).attr('height', h).attr('rx', 8).attr('fill', fill).attr('stroke', css('--axis'));
    svg.append('text').attr('x', x + w / 2).attr('y', y + h / 2 + (sub ? -2 : 4)).attr('text-anchor', 'middle').attr('font-size', 12).attr('font-weight', 600).attr('fill', ink()).text(t);
    if (sub) svg.append('text').attr('x', x + w / 2).attr('y', y + h / 2 + 12).attr('text-anchor', 'middle').attr('font-size', 10.5).attr('fill', ink()).text(sub);
  };
  const arrow = (y1, y2, x = 200) => svg.append('path').attr('d', `M${x},${y1} L${x},${y2} M${x - 4},${y2 - 6} L${x},${y2} L${x + 4},${y2 - 6}`).attr('fill', 'none').attr('stroke', ink2()).attr('stroke-width', 1.5);
  const sf = css('--surface-2');
  box(90, 8, 220, 32, 'Gene tokens (with masks)', sf);
  arrow(40, 56);
  box(90, 56, 220, 38, 'Embedding + position', sf, 'lookup table, V × d');
  arrow(94, 112);
  svg.append('rect').attr('x', 50).attr('y', 112).attr('width', 300).attr('height', 214).attr('rx', 10).attr('fill', 'none').attr('stroke', muted()).attr('stroke-dasharray', '5 4');
  svg.append('text').attr('x', 60).attr('y', 128).attr('font-size', 11).attr('fill', muted()).text('Block, repeated L times');
  box(90, 138, 220, 40, 'Multi-head attention', cat(0), 'tokens exchange information');
  arrow(178, 190);
  box(90, 190, 220, 26, 'Add (residual) + normalise', sf);
  arrow(216, 228);
  box(90, 228, 220, 40, 'Feed-forward network', cat(1), 'applied to each token separately');
  arrow(268, 280);
  box(90, 280, 220, 26, 'Add (residual) + normalise', sf);
  [[138, 190], [228, 280]].forEach(([a, b]) => svg.append('path').attr('d', `M76,${a - 6} L68,${a - 6} L68,${b + 13} L88,${b + 13}`).attr('fill', 'none').attr('stroke', ink2()).attr('stroke-dasharray', '3 3'));
  svg.append('text').attr('x', 60).attr('y', 224).attr('font-size', 10).attr('fill', muted()).attr('transform', 'rotate(-90 60 224)').text('skip');
  arrow(326, 342);
  box(40, 342, 320, 40, 'Output: masked-gene prediction or cell vector', sf);
}

/** The "parts" diagram for one cell. */
function pipelineSvg(cl, M, XY) {
  const wrap = el('div', { class: 'figure' });
  wrap.append(el('div', { class: 'fig-title' }, 'From a cell to contextual vectors'));
  const svg = d3.select(wrap).append('svg').attr('viewBox', '0 0 980 290').attr('width', 980).attr('role', 'img').attr('aria-label', 'tokens, embeddings, attention, outputs');
  const toks = cl.toks.slice(0, 6), RH = 36, y0 = 58;
  const cols = [{ x: 10, w: 190, t: '① Tokens' }, { x: 250, w: 220, t: '② Embedding (2-d view)' }, { x: 520, w: 200, t: '③ Attention mixes tokens' }, { x: 770, w: 200, t: '④ Output uses' }];
  cols.forEach((c) => { svg.append('rect').attr('x', c.x).attr('y', 14).attr('width', c.w).attr('height', 256).attr('rx', 10).attr('fill', css('--surface-2')).attr('stroke', css('--axis')); svg.append('text').attr('x', c.x + 12).attr('y', 36).attr('font-size', 13.5).attr('font-weight', 700).attr('fill', ink()).text(c.t); });
  const cy = (i) => y0 + i * RH;
  toks.forEach((t, i) => {
    svg.append('circle').attr('cx', 30).attr('cy', cy(i)).attr('r', 6).attr('fill', cat(GENES[t].prog));
    svg.append('text').attr('x', 44).attr('y', cy(i) + 4).attr('font-family', css('--mono')).attr('font-size', 13).attr('fill', ink2()).text(GENES[t].name);
    svg.append('text').attr('x', 262).attr('y', cy(i) + 4).attr('font-family', css('--mono')).attr('font-size', 12.5).attr('fill', ink2()).text(`[${XY[t].map(f2).join(', ')}]`);
    svg.append('path').attr('d', `M200,${cy(i)} L250,${cy(i)}`).attr('stroke', muted()).attr('stroke-width', 1.2);
    svg.append('circle').attr('cx', 540).attr('cy', cy(i)).attr('r', 6).attr('fill', cat(GENES[t].prog)).attr('stroke', css('--surface'));
    svg.append('circle').attr('cx', 700).attr('cy', cy(i)).attr('r', 6).attr('fill', cat(GENES[t].prog)).attr('stroke', css('--surface')).attr('stroke-width', 2);
    svg.append('path').attr('d', `M470,${cy(i)} L534,${cy(i)}`).attr('stroke', muted()).attr('stroke-width', 1.2);
  });
  toks.forEach((a, i) => toks.forEach((b, j) => { if (i < j && GENES[a].prog === GENES[b].prog) svg.append('path').attr('d', `M546,${cy(i)} C610,${cy(i)} 630,${cy(j)} 694,${cy(j)}`).attr('fill', 'none').attr('stroke', ink2()).attr('stroke-width', 1.6).attr('opacity', 0.7); }));
  toks.forEach((a, i) => svg.append('path').attr('d', `M546,${cy(i)} L694,${cy(i)}`).attr('stroke', muted()).attr('stroke-width', 1).attr('opacity', 0.6));
  svg.append('text').attr('x', 530).attr('y', 258).attr('font-size', 11).attr('fill', muted()).text('curves: same-programme links');
  [['masked-gene prediction', 'predict hidden genes'], ['cell vector', 'mean of the token vectors'], ['task heads', 'cell type, response, …']].forEach(([a, b], i) => {
    svg.append('text').attr('x', 782).attr('y', 70 + i * 62).attr('font-size', 13).attr('font-weight', 600).attr('fill', ink()).text(a);
    svg.append('text').attr('x', 782).attr('y', 88 + i * 62).attr('font-size', 11.5).attr('fill', ink2()).text(b);
  });
  svg.append('text').attr('x', 262).attr('y', 258).attr('font-size', 11).attr('fill', muted()).text(`cell of type ${CTYPES[cl.type].name}, first ${toks.length} of ${cl.toks.length} genes`);
  return wrap;
}
