# MEMORY.md: what to remember when picking this project back up

**Goal:** a free, hosted dashboard for understanding ML *visually*, for the owner to revisit. It has three tabs:
1. **ML concepts.** Every core ML idea as an interactive picture.
2. **ML in omics.** One mock anti-PD-1 study, walked end to end: bulk biomarker discovery → single-cell → spatial. Each step says where ML is used and where plain statistics is the right tool.
3. **Frontier methods.** Recent (2024–26) industry and institute methods, each rebuilt as a toy on mock data.

**Intent:** keep it *very intentional*. Explain clearly, show the mechanism, be honest about pitfalls and hype.

## Run it
- tmux session **`ml-dash`**:
  - window `serve` runs `python3 -m http.server 8765 --bind 127.0.0.1`, so the site is at http://127.0.0.1:8765/
  - from a laptop: `ssh -L 8765:localhost:8765 <vm>`
  - window `setup` is for builds
- Before running Python: `source ~/storage/sc_spatial/storage_env.sh`. The root disk is full, so everything must live on /storage.
- Regenerate omics data with `.venv/bin/python scripts/build_bulk.py`, `build_bulk_extra.py` (model comparison ~11 min; run single steps with e.g. `survival validation`) and `build_sc_spatial.py`, which write `data/omics/*.json`. All are seeded, so the prose numbers (computed from JSON at render time) stay true.
- QA: `~/my_env/bin/python scripts/check_pages.py [--tab omics] [--shots DIR]`.

## Where things are
- `js/registry.js` is the navigation and the list of every page id.
- `js/{topics,omics,frontier}/<id>.js` hold one page each.
- `js/lib/` holds the shared helpers: `ui.js` (page anatomy), `plot.js` (d3 + colour tokens), `data.js` (rng, toy datasets, linalg).
- Tab helpers (not pages; `_` prefix): `js/omics/_shared.js` (`stepCard`, `omicsData`), `_bulk_common.js`, `_sc_common.js`, `_spatial.js`; `js/frontier/_shared.js` (REFS, source cards). `js/topics/autoencoder.js` also exports the tiny MLP used by vae/contrastive/foundation-models.
- `docs/AUTHORING.md` is the page contract. `docs/frontier_research.md` is the literature brief.
- `DEV.md` records the key decisions.

## Owner preferences (carried over from the lung atlas)
- Zero cost.
- No Claude or AI attribution in commits or repo files. Commit as the owner's own git identity.
- Discuss big changes first.
- Long jobs go in tmux.
- The owner pushes to GitHub themselves.

## Status
- 2026-11-12: scaffold, shared library, gradient-descent exemplar, mock biology, and research brief done.
- 2026-11-13: **all 91 pages complete** (49 concepts, 25 omics, 17 frontier); `check_pages.py` passes in light and dark. First commit made.
- Known soft spots, stated on the pages: sp-biomarker inflamed-vs-excluded AUC 1.0 is near-circular by simulation; sc-pseudobulk finds ~nothing on real labels (6 vs 6 patients); responders show *more* CD8 exhaustion in the simulation; AlphaGenome has no URL in the brief.
