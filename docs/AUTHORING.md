# Page authoring contract

Every page is one ES module at `js/<dir>/<id>.js`, where `dir` is `topics`, `omics` or `frontier`. The id comes from `js/registry.js`, and the router lazy-imports the module:

```js
import { header, figure, slider, select, button, checkbox, stats, section, steps, callout, math, legend, seeAlso, animate, every, tip, untip, loadJSON, el, html } from '../lib/ui.js';
import { frame, scatter, decisionCanvas, line, heatmap, cat, css, ink, ink2, muted, diverging, sequential } from '../lib/plot.js';
import { rng, blobs, moons, circles, spiral, xor, regression1d, ridge, solve, mean, sd, sigmoid, linspace, range, eig2, clamp } from '../lib/data.js';
const d3 = window.d3;               // d3 v7.9.0 is loaded globally (vendored)
export default async function render(main, ctx) { ... }
```

**Reference implementation:** `js/topics/gradient-descent.js`. Copy its structure.

## Anatomy, in this order
1. `header(main, title, lede)`. The lede is one or two sentences that give the intuition in plain words. Bold the key idea.
2. **One or more interactive figures** made with `figure(main, title, caption)`. Each has controls (`slider`, `select`, `button`, `checkbox`), a visual, and live readouts (`stats`). The figure is the point of the page: the learner should *see the mechanism happen*. That means animating iterative algorithms step by step, letting the learner drag or click data where it helps, and showing two views side by side when a comparison is the lesson.
3. `section(main, 'What to notice', '<ul>…')`: concrete things to try ("set k=1 and…").
4. `steps(main, 'How it works', [...])`: the algorithm as numbered plain-language steps.
5. Optionally `math(main, '…')`: the key equation(s) as plain-text monospace. Keep it short.
6. `callout(main, '', 'In biology', …)`: where this shows up in omics. Be specific: tools and data types.
7. `callout(main, 'warn', 'Pitfalls', …)`: the classic ways it goes wrong.
8. `seeAlso(main, [[tab, id, label], …])`: links to related pages, using ids from the registry.

## Rules
- **Charts:**
  - Colours come only from CSS tokens: `cat(i)` for categorical slot i, in fixed order (slot 0 = class 0, slot 1 = class 1); `diverging()`; `sequential()`; `ink()`/`ink2()`/`muted()` for text.
  - No hard-coded hex, so dark mode works.
  - Text is never series-coloured.
  - Use 2px lines, markers ≥ 4px radius with a surface-colour ring, and recessive grids.
  - Use a `legend()` when there are 2+ series.
  - Add hover tooltips (`tip(evt, html)` / `untip()`) on points and cells where values matter.
- **Sizes:** `frame(parent, {width, height, …})` uses a viewBox, so the chart scales down. Design for about 460–560 px wide per panel, and put two panels side by side inside `el('div', {class:'fig-row'})`.
- **Animation:** use `animate(fn)` / `every(ms, fn)` only. The router stops them on navigation. Never use bare `setInterval`/`requestAnimationFrame`.
- **Determinism:** use `rng(seed)` instead of `Math.random`. That way the page looks the same on every visit, and the text can describe what the learner will see.
- **Performance:** each interaction should finish in under 100 ms. Use small n (100–400 points). If a computation is heavy, precompute it in Python into `data/…json` and load it with `loadJSON('data/…')`.
- **No external network** and no new libraries. Use d3 plus vanilla JS.
- **Tone:**
  - Plain English. Define each term the first time it appears. Use British or American spelling, but be consistent within a page.
  - Explain *why*, not just *what*.
  - Never pad.
  - Every number you state in prose must be true for the figure as rendered (it is seeded).
- **Honesty:** for frontier methods, cite the source (a real URL from `docs/frontier_research.md`), separate the authors' claims from independent benchmarks, and label each mini-replication as a teaching toy, not a reproduction of the paper's numbers.
- **Testing:** with the local server running, run `~/my_env/bin/python scripts/check_pages.py --only <id1,id2>`. It must print `OK`, with no console errors. Use `--shots <dir>` to screenshot pages, then look at the screenshots.
