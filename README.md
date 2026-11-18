# ML for Bioinformatics

**Check it out here:** https://archana-b24.github.io/ml-for-bioinformatics/

I've been spending a lot of time trying to get a better understanding of machine learning, not just how to use different models, but how they actually work and when they make sense to use in bioinformatics.

I ended up building this little interactive website as a way to learn things visually instead of just reading through explanations. It's also become a place I can go back to whenever I need to brush up on something, whether that's cross-validation, PCA, or why a simple statistical test might be a better choice than a complicated ML model.

## What's inside?

I've split it into three sections:

1. **ML Concepts:** Covers the fundamentals like regression, decision trees, PCA, clustering, gradient descent, neural networks, transformers, cross-validation, and more. Each topic has its own page with interactive examples.

2. **ML in Omics:** Walks through a fictional lung cancer immunotherapy study using bulk RNA-seq, single-cell, and spatial transcriptomics. The idea is to understand where ML is useful, where traditional statistics works better, and how to decide between the two.

3. **Frontier Methods:** A space to explore some of the newer approaches I've been curious about, like single-cell foundation models, virtual cells, and predicting gene expression from histology images. I've broken these down into smaller examples to make the underlying ideas easier to follow.

## How does it work?

I wanted to keep everything simple and visual. Instead of long explanations or complicated dashboards, most pages have small interactive examples where you can move things around, adjust parameters, and see what changes.

Each topic walks through what the method does, why it's useful, how it works, where it shows up in bioinformatics, and some common pitfalls to watch out for.

All the datasets are synthetic, so the examples are meant for learning rather than drawing any real biological conclusions.

## Want to run it locally?

It's built with plain HTML and JavaScript, so there's no complicated setup.

```bash
python3 -m http.server 8765
```

Then open http://localhost:8765 in your browser.

This is very much a learning project for me, and something I plan to keep adding to as I explore more methods. If you're also learning ML or working with omics data, hopefully you'll find some of it useful too!
