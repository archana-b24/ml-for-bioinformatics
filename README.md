# ML for Bioinformatics

**View it here: https://archana-b24.github.io/ml-for-bioinformatics/**

I made this to understand machine learning properly, by seeing it rather than only reading about it, and to see where these methods actually fit in omics work.

It's also my notebook of fundamentals. When I forget how cross-validation leaks, why PCA picks the axes it does, or when a plain t-test beats a fancy model, I come back here.

## What's inside

There are three tabs, and every page is carefully picked and written to be read slowly:

1. **ML concepts.** The basics, one topic per page: train/test splits, cross-validation, gradient descent, regression, trees, PCA, clustering, UMAP, neural networks, transformers, multiple testing, and a few more.
2. **ML in omics.** One made-up immunotherapy lung-cancer study, followed from bulk RNA-seq to single-cell to spatial data. Each step asks a simple question: is this a job for machine learning, or for plain statistics?
3. **Frontier methods.** Some of the newer ideas (single-cell foundation models, virtual cells, predicting gene expression from H&E slides), each rebuilt as a small toy, with an honest look at what holds up.

Every page follows the same flow: what it is, why it matters, where it's used, how it works step by step, common mistakes, and a short recap.

## Why simple pictures

No dashboard and nothing fancy. Each idea gets a small interactive picture: drag a slider, move a point, and watch what changes. A simple picture you can poke at usually teaches more than a big dashboard.

All the data is made up, so nothing here is a real result. It's built to make the ideas clear.

## Run it on your own computer

```bash
python3 -m http.server 8765
# then open http://localhost:8765
```

It's plain HTML and JavaScript, so there's nothing to install or build.
