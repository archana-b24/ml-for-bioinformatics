// Single source of truth for navigation. Each entry -> module at js/<tab.dir>/<id>.js
// exporting `default async function render(main, ctx)`. Missing module => "being written" placeholder.
// Pages follow the lesson arc in docs/AUTHORING.md ("Lesson pages"). Retired pages live in js/archive/<dir>/
// and stay reachable at #/archive/<dir>/<id> (listed in ARCHIVE below), but are not in the navigation.

export const TABS = [
  {
    id: 'concepts', label: '1 · ML concepts', dir: 'topics',
    intro: 'The core ideas of machine learning, each explained from the basics on one shared mock study: 180 lung-cancer patients treated with anti-PD-1 immunotherapy. Every page answers what the method is, why it is needed and where it is used, then builds it up step by step with worked examples and an interactive model.',
    sections: [
      { name: 'Foundations', items: [
        ['what-is-ml', 'What is machine learning?', 'Learning a rule from examples; supervised vs unsupervised; ML vs classical statistics'],
        ['train-test', 'Train/test split and overfitting', 'Why models are judged on data they have not seen, and what overfitting looks like'],
        ['cross-validation', 'Cross-validation and data leakage', 'k-fold and nested CV, and how leakage inflates performance'],
        ['gradient-descent', 'Gradient descent', 'How a model learns: loss functions and stepping downhill'],
      ]},
      { name: 'Supervised learning', items: [
        ['linear-regression', 'Linear regression', 'Predicting a number with a weighted sum; least squares and coefficients'],
        ['logistic-regression', 'Logistic regression', 'Will this patient respond? Odds, the S-curve, odds ratios, and a model builder on the study'],
        ['regularization', 'Regularisation (ridge and lasso)', 'Shrinking coefficients to control overfitting and select features'],
        ['metrics', 'Evaluating a classifier', 'Confusion matrix, ROC and PR curves, calibration and class imbalance'],
        ['decision-tree', 'Decision trees', 'Splitting patients into groups with simple yes/no questions'],
        ['tree-ensembles', 'Random forests and gradient boosting', 'Combining many trees: bagging vs boosting'],
      ]},
      { name: 'Unsupervised learning', items: [
        ['pca', 'Principal component analysis', 'Summarising many correlated measurements with a few directions'],
        ['clustering', 'Clustering', 'From k-means to graph-based (Leiden) clustering of cells'],
        ['tsne-umap', 't-SNE and UMAP', 'Two-dimensional maps of high-dimensional data, and what they distort'],
      ]},
      { name: 'Deep learning', items: [
        ['neural-network', 'Neural networks and backpropagation', 'Layers of weighted sums and non-linearities, trained by the chain rule'],
        ['transformers', 'Embeddings, attention and transformers', 'Representing genes and cells as vectors, and how attention combines them'],
      ]},
      { name: 'Statistics for omics', items: [
        ['multiple-testing', 'Multiple testing and FDR', 'Testing thousands of genes: p-values, Bonferroni and Benjamini–Hochberg'],
      ]},
    ],
  },
  {
    id: 'omics', label: '2 · ML in omics', dir: 'omics',
    intro: 'One mock anti-PD-1 study followed from raw data to a biomarker: bulk RNA-seq, then single-cell, then spatial. Each step states the biological question, whether machine learning or classical statistics is the right tool, and the common pitfalls.',
    sections: [
      { name: 'The study', items: [
        ['overview', 'Study overview and analysis map', 'Cohort design, the three data layers, and where ML is and is not used'],
      ]},
      { name: 'Bulk RNA-seq', items: [
        ['bulk-qc-batch', 'QC, normalisation and batch effects', 'Library size, log-CPM, outliers, and correcting batch without removing biology'],
        ['bulk-de', 'Differential expression', 'Linear models, shrinkage and FDR: why this step is statistics, not ML'],
        ['bulk-classifier', 'Biomarker classifier and nested CV', 'Selecting genes and building a response classifier without leakage'],
        ['bulk-survival', 'Survival analysis and risk scores', 'Kaplan–Meier, Cox regression, and prognostic vs predictive markers'],
      ]},
      { name: 'Single-cell RNA-seq', items: [
        ['sc-clustering', 'Clustering and cell-type annotation', 'From counts to clusters to labelled cell types'],
        ['sc-pseudobulk', 'Pseudobulk differential expression', 'Why cells are not independent replicates'],
      ]},
      { name: 'Spatial omics', items: [
        ['sp-biomarker', 'Spatial biomarker', 'Inflamed vs excluded tumours: where the T cells are'],
      ]},
    ],
  },
  {
    id: 'frontier', label: '3 · Frontier methods', dir: 'frontier',
    intro: 'Recent (2024–2026) methods from industry and research institutes, each rebuilt as a small model on mock data to show the core mechanism. Every page cites its source and separates the authors\' claims from independent benchmarks.',
    sections: [
      { name: 'Overview', items: [
        ['landscape', 'The 2024–26 landscape', 'Who is building what, at what scale, and how the methods compare'],
      ]},
      { name: 'Methods', items: [
        ['foundation-models', 'Single-cell foundation models', 'Pretraining on millions of cells, and whether embeddings beat PCA'],
        ['virtual-cells', 'Virtual cells and perturbation prediction', 'Predicting the effect of a perturbation, and the baselines that are hard to beat'],
        ['he-to-expression', 'H&E to gene expression', 'Predicting spatial expression from histology images'],
      ]},
    ],
  },
];

/** Retired pages (pre-2026-11 rewrite): id + title per dir, served from js/archive/<dir>/<id>.js at #/archive/<dir>/<id>. */
export const ARCHIVE = {"topics": [["what-is-ml", "What \"learning\" means"], ["train-test", "Train, validation, test"], ["bias-variance", "Under- vs overfitting"], ["loss-functions", "Loss functions"], ["gradient-descent", "Gradient descent"], ["optimizers", "Optimizers race"], ["regularization", "Regularization (L1 / L2)"], ["cross-validation", "Cross-validation"], ["metrics", "Classification metrics"], ["imbalance", "Class imbalance"], ["scaling", "Feature scaling"], ["curse-dim", "Curse of dimensionality"], ["probability-mle", "Likelihood & Bayes"], ["linear-regression", "Linear regression"], ["logistic-regression", "Logistic regression"], ["knn", "k-nearest neighbours"], ["naive-bayes", "Naive Bayes"], ["decision-tree", "Decision trees"], ["random-forest", "Random forests & bagging"], ["gradient-boosting", "Gradient boosting"], ["svm", "Support vector machines"], ["kmeans", "k-means"], ["hierarchical", "Hierarchical clustering"], ["gmm", "Gaussian mixtures & EM"], ["dbscan", "DBSCAN"], ["pca", "PCA"], ["tsne-umap", "t-SNE & UMAP"], ["graph-clustering", "kNN graphs & Leiden"], ["neural-network", "Neural network playground"], ["activations", "Activation functions"], ["backprop", "Backpropagation"], ["cnn", "Convolutions (CNNs)"], ["rnn", "Recurrent networks"], ["attention", "Attention"], ["transformer", "Transformers"], ["embeddings", "Embeddings"], ["autoencoder", "Autoencoders"], ["vae", "Variational autoencoders"], ["gan", "GANs"], ["diffusion", "Diffusion models"], ["contrastive", "Contrastive learning"], ["gnn", "Graph neural networks"], ["training-tricks", "Dropout, batch norm, early stopping"], ["hyperparameter-tuning", "Hyperparameter tuning"], ["interpretability", "Feature importance & SHAP"], ["calibration", "Calibration"], ["multiple-testing", "Multiple testing & FDR"], ["foundation-models", "Self-supervision & foundation models"], ["double-descent", "Double descent"]], "omics": [["overview", "The mock study & ML map"], ["bulk-qc", "QC, normalisation & outliers"], ["bulk-batch", "Exploration & batch effects"], ["bulk-subtypes", "Unsupervised subtypes"], ["bulk-de", "Differential expression"], ["bulk-feature-selection", "Biomarker feature selection"], ["bulk-classifier", "Response classifier & nested CV"], ["bulk-model-compare", "Model comparison & calibration"], ["bulk-explain", "Explaining predictions"], ["bulk-survival", "Survival & risk scores"], ["bulk-deconvolution", "Cell-type deconvolution"], ["bulk-validation", "Locking & validating the signature"], ["sc-qc", "QC & doublet detection"], ["sc-embedding", "Normalise, HVGs, PCA"], ["sc-clustering", "Graph clustering & UMAP"], ["sc-integration", "Batch integration"], ["sc-annotation", "Cell-type annotation"], ["sc-trajectory", "Trajectories & pseudotime"], ["sc-pseudobulk", "Pseudobulk DE"], ["sp-data", "Spatial data & neighbourhoods"], ["sp-svg", "Spatially variable genes"], ["sp-domains", "Spatial domains"], ["sp-deconvolution", "Spot deconvolution"], ["sp-niches", "Niches & cell–cell communication"], ["sp-biomarker", "Back to the biomarker"]], "frontier": [["landscape", "The 2024–26 landscape"], ["geneformer", "Geneformer: rank-value encoding"], ["cell2sentence", "Cell2Sentence / C2S-Scale"], ["llm-annotation", "LLM cell annotation"], ["zero-shot-check", "Do foundation embeddings beat PCA?"], ["in-context", "In-context learning on cells (Stack)"], ["perturbation-arena", "Perturbation baselines arena"], ["state-sets", "State: predicting distributions"], ["flow-matching", "Flow matching (CellFlow / PRiMeFlow)"], ["metric-gaming", "Virtual Cell Challenge metrics"], ["perturbation-tensor", "Large Perturbation Model"], ["spatial-niches", "Novae & Nicheformer: niches"], ["masked-cells", "CellPLM: impute cells from neighbours"], ["he-to-expression", "H&E → gene expression"], ["clinical-signature", "Small signature + clinical score"], ["agents", "Agents & AI co-scientists"], ["sequence-to-expression", "Sequence → expression (AlphaGenome)"]]};

export function allItems(tab) {
  const out = [];
  tab.sections.forEach((s) => s.items.forEach(([id, title, blurb]) => out.push({ id, title, blurb, section: s.name })));
  return out;
}
