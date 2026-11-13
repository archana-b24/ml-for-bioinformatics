// Single source of truth for navigation. Each entry -> module at js/<tab.dir>/<id>.js
// exporting `default async function render(main, ctx)`. Missing module => "being written" placeholder.

export const TABS = [
  {
    id: 'concepts', label: '1 · ML concepts', dir: 'topics',
    intro: 'Every core idea in machine learning as a picture you can poke. Each page has the same parts: the intuition in one line, a figure you can play with, what to notice, how it works step by step, and the classic ways it goes wrong.',
    sections: [
      { name: 'Foundations', items: [
        ['what-is-ml', 'What "learning" means', 'Fitting a function from examples; the supervised / unsupervised / self-supervised map'],
        ['train-test', 'Train, validation, test', 'Why we hold data out, and how leakage quietly ruins it'],
        ['bias-variance', 'Under- vs overfitting', 'Drag model complexity and watch train and test error split apart'],
        ['loss-functions', 'Loss functions', 'MSE, MAE, Huber, cross-entropy, hinge: what each one punishes'],
        ['gradient-descent', 'Gradient descent', 'Roll a ball down a loss surface; learning rate decides everything'],
        ['optimizers', 'Optimizers race', 'SGD vs momentum vs RMSProp vs Adam on the same landscape'],
        ['regularization', 'Regularization (L1 / L2)', 'Diamonds vs circles: why the lasso zeroes coefficients'],
        ['cross-validation', 'Cross-validation', 'k-fold, stratified, grouped, and nested CV, animated'],
        ['metrics', 'Classification metrics', 'Confusion matrix, ROC, PR curves driven by one threshold slider'],
        ['imbalance', 'Class imbalance', 'Why 99% accuracy can mean nothing, and what to do about it'],
        ['scaling', 'Feature scaling', 'How unscaled features distort distances and gradients'],
        ['curse-dim', 'Curse of dimensionality', 'In high dimensions, every point is equally far from every other'],
        ['probability-mle', 'Likelihood & Bayes', 'MLE vs posterior: how data and priors combine'],
      ]},
      { name: 'Supervised learning', items: [
        ['linear-regression', 'Linear regression', 'Least squares as literally minimising squared areas'],
        ['logistic-regression', 'Logistic regression', 'A line, a sigmoid, and a probability'],
        ['knn', 'k-nearest neighbours', 'Vote of neighbours; k controls smoothness'],
        ['naive-bayes', 'Naive Bayes', 'Multiply per-feature evidence; surprisingly strong baseline'],
        ['decision-tree', 'Decision trees', 'Axis-aligned splits that greedily purify groups'],
        ['random-forest', 'Random forests & bagging', 'Many noisy trees averaged into a smooth boundary'],
        ['gradient-boosting', 'Gradient boosting', 'Each new tree fits the residuals of the last'],
        ['svm', 'Support vector machines', 'Maximum margin and the kernel trick'],
      ]},
      { name: 'Unsupervised learning', items: [
        ['kmeans', 'k-means', 'Assign, move centroids, repeat'],
        ['hierarchical', 'Hierarchical clustering', 'Build a dendrogram by merging nearest groups'],
        ['gmm', 'Gaussian mixtures & EM', 'Soft clustering with ellipses that learn their shape'],
        ['dbscan', 'DBSCAN', 'Density-based clusters and noise points'],
        ['pca', 'PCA', 'Find the direction of maximum variance; project onto it'],
        ['tsne-umap', 't-SNE & UMAP', 'Neighbour-preserving maps, and what they distort'],
        ['graph-clustering', 'kNN graphs & Leiden', 'The single-cell way to cluster: build a graph, find communities'],
      ]},
      { name: 'Deep learning', items: [
        ['neural-network', 'Neural network playground', 'Train a small network in your browser and watch it bend space'],
        ['activations', 'Activation functions', 'ReLU, sigmoid, tanh, GELU and their gradients'],
        ['backprop', 'Backpropagation', 'The chain rule flowing backwards through a computation graph'],
        ['cnn', 'Convolutions (CNNs)', 'A sliding filter that detects patterns anywhere'],
        ['rnn', 'Recurrent networks', 'Memory carried step by step, and why it fades'],
        ['attention', 'Attention', 'Queries, keys and values: every token looks at every other'],
        ['transformer', 'Transformers', 'Attention + MLP blocks stacked; positions, masks, heads'],
        ['embeddings', 'Embeddings', 'Turning discrete things (words, genes, cells) into geometry'],
        ['autoencoder', 'Autoencoders', 'Squeeze through a bottleneck, reconstruct, learn structure'],
        ['vae', 'Variational autoencoders', 'A smooth latent space you can sample from'],
        ['gan', 'GANs', 'A generator and a critic in an arms race'],
        ['diffusion', 'Diffusion models', 'Learn to undo noise one step at a time'],
        ['contrastive', 'Contrastive learning', 'Pull augmented pairs together, push others apart'],
        ['gnn', 'Graph neural networks', 'Nodes update by passing messages to neighbours'],
        ['training-tricks', 'Dropout, batch norm, early stopping', 'The practical tricks that make deep nets train'],
      ]},
      { name: 'Practice & interpretation', items: [
        ['hyperparameter-tuning', 'Hyperparameter tuning', 'Grid vs random vs Bayesian optimisation'],
        ['interpretability', 'Feature importance & SHAP', 'Which features drove this prediction, and how much'],
        ['calibration', 'Calibration', 'When the model says 80%, is it right 80% of the time?'],
        ['multiple-testing', 'Multiple testing & FDR', 'Testing 20,000 genes: p-values, Bonferroni, Benjamini–Hochberg'],
        ['foundation-models', 'Self-supervision & foundation models', 'Pretrain on unlabelled data, fine-tune on your task'],
        ['double-descent', 'Double descent', 'Why huge models can overfit less, not more'],
      ]},
    ],
  },
  {
    id: 'omics', label: '2 · ML in omics', dir: 'omics',
    intro: 'One mock immunotherapy study, followed from raw counts to a biomarker. At each step: what the biological question is, where ML is used (and where plain statistics is the better tool), which method, why, and the pitfall that catches people.',
    sections: [
      { name: 'The study', items: [
        ['overview', 'The mock study & ML map', 'Cohort design, the three data layers, and every place ML enters'],
      ]},
      { name: 'Bulk RNA-seq → biomarker', items: [
        ['bulk-qc', 'QC, normalisation & outliers', 'Library size, log-CPM, and outlier detection'],
        ['bulk-batch', 'Exploration & batch effects', 'PCA reveals the batch; correct it, without erasing biology'],
        ['bulk-subtypes', 'Unsupervised subtypes', 'Clustering patients and checking it is real (consensus, silhouette)'],
        ['bulk-de', 'Differential expression', 'Linear models, shrinkage, FDR: statistics, not ML, and why'],
        ['bulk-feature-selection', 'Biomarker feature selection', 'Lasso / elastic-net paths and stability selection'],
        ['bulk-classifier', 'Response classifier & nested CV', 'The leakage trap: selecting genes outside the CV loop turns random labels into AUC 0.96'],
        ['bulk-model-compare', 'Model comparison & calibration', 'Logistic vs random forest vs boosting, honestly compared'],
        ['bulk-explain', 'Explaining predictions', 'Per-patient contributions and what they do or do not mean'],
        ['bulk-survival', 'Survival & risk scores', 'Cox-lasso risk score and Kaplan–Meier curves'],
        ['bulk-deconvolution', 'Cell-type deconvolution', 'Which cells drive the bulk signal? NNLS against a reference'],
        ['bulk-validation', 'Locking & validating the signature', 'External cohort, frozen model, and what "validated" means'],
      ]},
      { name: 'Single-cell RNA-seq', items: [
        ['sc-qc', 'QC & doublet detection', 'MAD thresholds and a classifier trained on simulated doublets'],
        ['sc-embedding', 'Normalise, HVGs, PCA', 'From sparse counts to a denoised low-dimensional space'],
        ['sc-clustering', 'Graph clustering & UMAP', 'kNN graph, Leiden, and the resolution question'],
        ['sc-integration', 'Batch integration', 'Mixing batches while keeping cell types apart'],
        ['sc-annotation', 'Cell-type annotation', 'Marker scores vs reference label transfer vs foundation models'],
        ['sc-trajectory', 'Trajectories & pseudotime', 'Ordering T cells along an exhaustion path'],
        ['sc-pseudobulk', 'Pseudobulk DE', 'Cells are not independent replicates: the pseudoreplication trap'],
      ]},
      { name: 'Spatial omics', items: [
        ['sp-data', 'Spatial data & neighbourhoods', 'Spots vs single cells, and building a spatial graph'],
        ['sp-svg', 'Spatially variable genes', "Moran's I: is expression patterned in space?"],
        ['sp-domains', 'Spatial domains', 'Clustering with neighbour smoothing to find tissue regions'],
        ['sp-deconvolution', 'Spot deconvolution', 'Using the single-cell reference to unmix each spot'],
        ['sp-niches', 'Niches & cell–cell communication', 'Neighbourhood enrichment and ligand–receptor scoring'],
        ['sp-biomarker', 'Back to the biomarker', 'Inflamed vs excluded tumours: a spatial biomarker'],
      ]},
    ],
  },
  {
    id: 'frontier', label: '3 · Frontier methods', dir: 'frontier',
    intro: 'What large pharma, tech companies and research institutes are publishing now (2024–2026), each rebuilt in miniature on mock data so you can see the core mechanism. Every page cites the source and is honest about benchmark caveats.',
    sections: [ // chosen from docs/frontier_research.md as a learning path
      { name: 'Start here', items: [
        ['landscape', 'The 2024–26 landscape', 'Who is building what, the scale of the data, and the honest scoreboard'],
      ]},
      { name: 'Cells as language', items: [
        ['geneformer', 'Geneformer: rank-value encoding', 'Turn a cell into a sentence of genes, then mask and predict'],
        ['cell2sentence', 'Cell2Sentence / C2S-Scale', 'Google & Yale: write cells as plain text and train an LLM on them'],
        ['llm-annotation', 'LLM cell annotation', 'GPTCelltype-style labelling from marker lists, and where it hallucinates'],
        ['zero-shot-check', 'Do foundation embeddings beat PCA?', 'The zero-shot reality check (Kedzierska et al. 2025)'],
        ['in-context', 'In-context learning on cells (Stack)', 'Arc 2026: predict a new condition from a prompt of example cells'],
      ]},
      { name: 'Virtual cells & perturbations', items: [
        ['perturbation-arena', 'Perturbation baselines arena', 'Mean vs additive vs linear vs deep: the benchmark that humbled foundation models'],
        ['state-sets', 'State: predicting distributions', 'Arc 2025: set models and distribution losses vs per-cell regression'],
        ['flow-matching', 'Flow matching (CellFlow / PRiMeFlow)', 'Learn a velocity field that carries control cells to perturbed ones'],
        ['metric-gaming', 'Virtual Cell Challenge metrics', 'How a single scale factor games one metric and fails another'],
        ['perturbation-tensor', 'Large Perturbation Model', 'Perturbation × gene × context as a factorised tensor'],
      ]},
      { name: 'Spatial & pathology', items: [
        ['spatial-niches', 'Novae & Nicheformer: niches', 'Message passing over the tissue graph, then self-supervised prototypes'],
        ['masked-cells', 'CellPLM: impute cells from neighbours', 'Treat tissue as a sentence of cells; mask and reconstruct'],
        ['he-to-expression', 'H&E → gene expression', 'Pathology foundation models and the HEST benchmark protocol'],
      ]},
      { name: 'Clinical & agents', items: [
        ['clinical-signature', 'Small signature + clinical score', 'Tempus-style immune profile score with honest nested CV'],
        ['agents', 'Agents & AI co-scientists', 'Biomni-style tool loops and Elo-ranked hypotheses, replayed'],
        ['sequence-to-expression', 'Sequence → expression (AlphaGenome)', 'A tiny CNN reads DNA motifs; in-silico mutagenesis'],
      ]},
    ],
  },
];

export function allItems(tab) {
  const out = [];
  tab.sections.forEach((s) => s.items.forEach(([id, title, blurb]) => out.push({ id, title, blurb, section: s.name })));
  return out;
}
