# Frontier methods: research brief (compiled 2026-11-12)

Compiled by web search for Tab 3. Only URLs that were actually seen are listed. Performance claims are the developers' own unless they say otherwise. The **Unverified** list is at the bottom.

## Single-cell foundation models
| Method | Org / venue | Core ML ingredient | Caveat |
|---|---|---|---|
| **Geneformer** | Theodoris lab (Gladstone/Broad); Nature 2023, V2 104M cells (Dec 2024); NVIDIA BioNeMo recipe. https://huggingface.co/ctheodoris/Geneformer | Rank-value encoding: divide by the gene's corpus non-zero median, then rank. BERT masked-LM. In-silico deletion = drop a token and measure the embedding shift. | Kedzierska et al. 2025 (Genome Biol): zero-shot is worse than HVG+PCA / scVI / Harmony. https://link.springer.com/article/10.1186/s13059-025-03574-x |
| **scGPT** | Bo Wang lab; Nat Methods 2024. https://www.nature.com/articles/s41592-024-02201-0.pdf | Gene token + binned-value token; masked generative pretraining on 33M cells | Same zero-shot critique |
| **scFoundation** | BioMap/Tsinghua; Nat Methods 2024. https://github.com/biomap-research/scFoundation | Asymmetric encoder-decoder; read-depth-aware masked regression | Basis of the 2025 VCC winner |
| **UCE** | Stanford (Rosen, Roohani, Leskovec, Quake); bioRxiv 2023. https://www.github.com/snap-stanford/UCE | Genes represented by ESM2 protein embeddings, so models work across species | |
| **TranscriptFormer** | CZI/Biohub; bioRxiv 2025. https://www.biorxiv.org/content/10.1101/2025.04.25.650731.full.pdf | Autoregressive next-gene + count; 112M cells, 12 species | Preprint |
| **rbio1** | CZI; bioRxiv 2025. https://www.biorxiv.org/content/10.1101/2025.08.18.670981.full.pdf | LLM post-trained with RL, using a virtual-cell model as a soft verifier | |
| **C2S-Scale (Cell2Sentence)** | Google Research/DeepMind + Yale; bioRxiv 2025. https://doi.org/10.1101/2025.04.14.648850 | A cell becomes text: its gene names ordered by expression. Gemma-2 causal LM, 2B/27B | The silmitasertib discovery rests on a single in-vitro validation; preprint |
| **Stack** | Arc Institute, Jan 2026. https://arcinstitute.org/news/foundation-model-stack, https://github.com/arcinstitute/stack | Tabular transformer that attends across genes and across cells; in-context learning from a prompt set of cells | Self-reported wins |

## Perturbation and virtual cells
- **State** (Arc 2025; https://arcinstitute.org/news/virtual-cell-model-state, https://github.com/arcinstitute/state). An SE embedding feeds an ST set transformer that maps a control-cell *set* to a perturbed set. The loss is distributional (MMD per secondary sources, unverified). Arc calls it "first to consistently beat linear baselines", which is self-reported.
- **Baseline critiques**:
  - Ahlmann-Eltze, Huber, Anders, Nat Methods 22:1657 (2025), https://github.com/const-ae/linear_perturbation_prediction-Paper. The additive baseline beats 5 foundation models and 2 other deep learning models on double perturbations. Nothing beats the training mean on unseen single perturbations. A linear model with perturbation-data-pretrained embeddings does best.
  - **Systema** (Brbić lab, EPFL; https://brbiclab.epfl.ch/projects/systema): systematic perturbed-vs-control differences inflate metrics, so it evaluates against the perturbed centroid instead.
  - Benchmarks: PertEval-scFM (ICML 2025, https://proceedings.mlr.press/v267/wenteler25a.html), PerturBench (https://arxiv.org/html/2408.10609v4), VCBench 2026 (https://www.biorxiv.org/content/10.64898/2026.06.18.733146v1.full.pdf), where simple baselines match or beat foundation models on 4 of 5 axes.
- **Virtual Cell Challenge 2025** (https://arcinstitute.org/news/virtual-cell-challenge-2025-wrap-up). H1 hESC, 300 CRISPRi perturbations. Metrics: PDS, DES, MAE.
  - Winners: BioMap (scFoundation-style), XLearning (MLP on pseudobulk deltas), Outlier (statistical reweighting).
  - Generalist prize: Altos (flow matching, PRiMeFlow https://arxiv.org/html/2604.13986v1).
  - Lessons: almost nobody beat the mean baseline on MAE; L1-based PDS rewards inflating prediction magnitude; hybrid statistical + deep learning approaches won.
  - The 2026 round (https://arcinstitute.org/news/virtual-cell-challenge-2026) is zero-shot on 6 new cell lines. Winners have not been announced as of 2026-11-12.
- **CellFlow** (Theis lab, bioRxiv 2025, https://www.biorxiv.org/content/10.1101/2025.04.11.648220.full.pdf): conditional flow matching from the control to the perturbed distribution.
- **LPM** (Large Perturbation Model; arXiv 2503.23535, https://arxiv.org/abs/2503.23535; Nat Comput Sci 2025): disentangles perturbation × readout × context. Genentech affiliation is inferred from the GitHub org.
- **TxPert** (Valence Labs / Recursion 2025, https://www.valencelabs.com/txpert-predicting-cellular-responses-to-unseen-genetic-perturbations/): a GNN over knowledge graphs for unseen perturbations.
- **Data at scale**:
  - Tahoe-100M (https://www.biorxiv.org/content/10.1101/2025.02.20.639398.full.pdf)
  - Xaira X-Atlas/Orion, 8M-cell Perturb-seq (https://www.biorxiv.org/content/10.1101/2025.06.11.659105.full.pdf)

## Spatial and histology
- **Nicheformer** (Theis lab, Nat Methods 2025, DOI 10.1038/s41592-025-02814-z): pretrained on SpatialCorpus-110M. Spatial context labels learned from spatial data are transferred onto dissociated cells.
- **Novae** (Nat Methods 2025, DOI 10.1038/s41592-025-02899-6, https://pypi.org/project/novae/): graph attention over neighbourhoods plus SwAV self-supervision, giving zero-shot spatial domains.
- **CellPLM** (ICLR 2024): cells are tokens and a tissue is the sentence; masked cells are reconstructed from their spatial neighbours.
- **H&E → expression**:
  - Pathology foundation models: UNI, Prov-GigaPath (Microsoft/Providence, Nature 2024, https://pubmed.ncbi.nlm.nih.gov/38778098/), Virchow (Paige), H-optimus (Bioptimus).
  - HEST-1k benchmark (https://arxiv.org/html/2406.16192v2): frozen embeddings → PCA → ridge → Pearson r on 50 HVGs, with the best at about 0.41.
  - Bioptimus M-Optimus-1 (2026, https://docs.bioptimus.com/documentation/models/m-optimus) is multimodal; its numbers are vendor claims.
- **Industry data**:
  - Owkin MOSAIC (https://docs.owkin.com/user-guide/what-data-is-available-in-k/overview-of-mosaic-data)
  - Noetik OCTO-vc (https://www.businesswire.com/news/home/20241218973612/en/)

## Bulk and biomarkers
- **BulkFormer** (bioRxiv 2025, https://www.biorxiv.org/content/10.1101/2025.06.11.659222.full.pdf): masked-expression foundation model for bulk RNA.
- **Tempus Immune Profile Score** (https://medrxiv.org/content/10.1101/2024.11.05.24316583v2.full.pdf): an ML model on TMB + 8 RNA genes; development n=1707, validation n=1600.
- **AstraZeneca × SOPHiA** multimodal stratification (POSEIDON) is announcement-level only.

## Agents and LLMs
- **Biomni** (Stanford, https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12157518/): ReAct agent with about 150 tools.
- **Google AI co-scientist** (https://research.google/blog/accelerating-scientific-breakthroughs-with-an-ai-co-scientist/): Gemini multi-agent system with Elo tournament ranking of hypotheses.
- **CellAgent** (https://arxiv.org/abs/2407.09811v1).
- **GPTCelltype** (Nat Methods 2024, DOI 10.1038/s41592-024-02235-4): >75% concordance in most tissues.
- **Owkin K Navigator** (2025).
- **AlphaGenome** (DeepMind, June 2025): DNA → RNA tracks, up to 1 Mb.

## Unverified
- State's loss (MMD).
- Venues: Systema; the Science version of Biomni; BulkFormer in Cell Systems; AI co-scientist in Nature; AlphaGenome in Nature.
- LPM's Genentech link.
- The Noetik–GSK deal.
- 2026 VCC winners (not yet announced).

No 2024–26 transcriptomics foundation-model papers were found from Novartis, Sanofi, J&J, Bayer, Insitro, Illumina, Isomorphic or Meta.
