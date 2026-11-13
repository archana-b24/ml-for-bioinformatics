# ML Visual Atlas

Machine learning, explained by pictures you can poke. The site has three tabs:

1. **ML concepts.** About 50 core ideas, from gradient descent to diffusion models. Each one has an interactive figure, what to notice, how it works, and its pitfalls.
2. **ML in omics.** One mock anti-PD-1 lung-cancer study, followed from bulk RNA-seq biomarker discovery through single-cell to spatial. Every step states whether ML or plain statistics is the right tool.
3. **Frontier methods.** Methods from 2024–26 industry and institute papers (Geneformer, C2S-Scale, Arc's State and Stack, perturbation baselines, Novae, H&E→expression, agents), each rebuilt as a small teaching toy, with an honest reality check.

All data is synthetic. The pages are a static site with no build step and no server.

## Run locally
```bash
python3 -m http.server 8765        # then open http://localhost:8765
```

## Regenerate the mock omics data
```bash
source ~/storage/sc_spatial/storage_env.sh
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python numpy scipy scikit-learn umap-learn leidenalg igraph
.venv/bin/python scripts/build_bulk.py         # bulk steps 1-6 (~2 min)
.venv/bin/python scripts/build_bulk_extra.py   # bulk steps 7-11 (~12 min; or name steps: survival validation)
.venv/bin/python scripts/build_sc_spatial.py   # single-cell + spatial (~3.5 min)
```

## Deploy free on GitHub Pages
1. Create a public GitHub repo (e.g. `ml-visual-atlas`) and push this folder to `main`.
2. In the repo, go to **Settings → Pages → Build and deployment → Source: Deploy from a branch**, and choose **Branch: `main` / root**.
3. The site appears at `https://<user>.github.io/ml-visual-atlas/` within about a minute. Every push redeploys it.

`.nojekyll` is included so GitHub serves the files as they are.

## Layout
| Path | What |
|---|---|
| `index.html`, `css/style.css` | The page shell and the design tokens for light and dark themes. |
| `js/registry.js` | Every tab, section and page id. This is the navigation. |
| `js/{topics,omics,frontier}/<id>.js` | One page per file, lazy-loaded. |
| `js/lib/` | Shared helpers: page anatomy, d3 plotting, seeded RNG, and toy datasets. |
| `scripts/` | The mock-data generators and the page smoke test (`check_pages.py`). |
| `data/` | Precomputed JSON for the omics pages. |
| `docs/` | The authoring contract and the frontier literature brief. |
| `DEV.md` / `MEMORY.md` | Key decisions, and how to pick the project back up. |
