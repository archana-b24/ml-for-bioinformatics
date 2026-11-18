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
- `js/registry.js` holds the navigation and the list of every page id. Retired pages are under `js/archive/<dir>/` and reachable at `#/archive/...`, but are not in the navigation.
- `js/{topics,omics,frontier}/<id>.js` hold one lesson page each: 16 concepts, 8 omics, 4 frontier.
- `js/lib/` holds the shared helpers: `lesson.js` (lesson arc: chapter, rung, figure, worked, quiz, pitfalls, recap), `study.js` (loads `data/study/cohort.json`), `models.js` (logistic, ridge, PCA, trees, AUC…), `ui.js`, `plot.js` (d3 + colour tokens) and `data.js` (rng, linalg).
- Tab helpers are not pages and carry a `_` prefix: `js/omics/_shared.js` (`stepCard`), `_bulk_common.js`, `_sc_common.js`, `_spatial.js`; `js/frontier/_shared.js` (REFS) and `_fm_toy.js`.
- Data comes from `scripts/build_study.py` (the patient-level cohort used by every lesson) and the per-lesson `scripts/build_*_lesson.py` / `build_*.py` scripts, which write `data/study/*.json` and `data/omics/*_lab.json`.
- `docs/AUTHORING.md` is the page contract (see "Lesson pages"). `docs/frontier_research.md` is the literature brief.
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
- 2026-11-16: **rewrite started.** Logistic regression rebuilt as the lesson template (`js/lib/lesson.js`, `study.js`, `models.js`, `data/study/cohort.json`). Waiting for owner feedback before rolling out. Planned list: concepts (16) what-is-ML, train/test+overfitting, CV+leakage, gradient descent, linear reg, logistic reg, regularisation, judging a classifier, trees, RF+boosting, PCA, clustering, t-SNE/UMAP, neural nets, embeddings/attention/transformers, multiple testing; omics (8); frontier (4).
- 2026-11-16 to 18: the 27 remaining lesson pages were drafted in batch from per-page briefs (`scripts/rollout/`, git-ignored; see DEV #17), then reviewed by hand. The landscape and he-to-expression drafts were never reviewed, because the batch stopped before their review step. Both were fixed on 2026-11-18, including the H&E simulation, whose per-patient noise was swamping the image signal. All 28 pages pass `check_pages.py` in light and dark. A backup from before the rewrite is at `~/storage/ml_dash_backup_20261116_pre_rollout.tgz`.
- Known soft spots, stated on the pages: the H&E new-centre test shows a level shift (bias), not a drop in r, because Pearson r ignores a uniform shift; pseudobulk finds 1 gene at 5% FDR across six cell types with 6 vs 6 patients.
