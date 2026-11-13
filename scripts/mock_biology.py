"""Shared ground truth for the mock immunotherapy study (Tab 2). Every omics generator imports this,
so bulk, single-cell and spatial data tell ONE consistent biological story.

The story (fictional, but shaped like real anti-PD-1 NSCLC data):
  * Tumours fall into three immune phenotypes: inflamed, excluded, desert.
  * Response to anti-PD-1 is mostly driven by CD8 T-cell infiltration *inside* the tumour (inflamed),
    which shows up as an IFN-gamma / cytotoxic gene module (CXCL9, GZMB, ...). This is the planted
    predictive biomarker.
  * Excluded tumours have T cells trapped in TGF-beta-rich fibroblast stroma: the CAF/TGF-b module is
    high and response is low. Bulk sees T cells; only spatial data sees that they are outside the tumour.
  * A proliferation module (MKI67, TOP2A, ...) is PROGNOSTIC (shorter survival) but NOT predictive of
    response. This is a deliberate trap for the survival vs response pages.
  * Bulk expression = sum over cell types of (fraction x reference profile) x noise, so deconvolution works.
"""
import numpy as np

SEED = 2026

CELL_TYPES = ["Tumour", "CD8 T", "CD4 T", "Treg", "B", "Macrophage", "Fibroblast", "Endothelial"]

# Canonical markers (real symbols, for teaching). Expression is high in the listed cell type.
MARKERS = {
    "Tumour": ["EPCAM", "KRT8", "KRT18", "KRT19", "NKX2-1", "SFTPB", "MUC1", "CEACAM5"],
    "CD8 T": ["CD3E", "CD3D", "CD8A", "CD8B", "NKG7", "CCL5"],
    "CD4 T": ["CD3E", "CD3D", "CD4", "IL7R", "CCR7", "TCF7"],
    "Treg": ["CD3E", "CD4", "FOXP3", "IL2RA", "CTLA4", "TNFRSF18"],
    "B": ["MS4A1", "CD79A", "CD79B", "CD19", "BANK1", "CXCR5"],
    "Macrophage": ["CD68", "CD163", "C1QA", "C1QB", "LYZ", "APOE"],
    "Fibroblast": ["COL1A1", "COL1A2", "DCN", "LUM", "PDGFRB", "FAP"],
    "Endothelial": ["PECAM1", "VWF", "CDH5", "CLDN5", "PLVAP", "KDR"],
}

# Modules: genes whose expression depends on a latent program (on top of cell-type identity).
MODULES = {
    # predictive biomarker: interferon-gamma / cytotoxic (expressed by CD8 T + IFN-responding tumour/myeloid)
    "IFNG": ["IFNG", "CXCL9", "CXCL10", "CXCL11", "IDO1", "GZMA", "GZMB", "PRF1", "STAT1", "HLA-DRA", "LAG3", "CD274"],
    # exclusion: TGF-beta / cancer-associated fibroblast program (fibroblasts)
    "TGFB_CAF": ["TGFB1", "POSTN", "ACTA2", "FN1", "COL11A1", "LRRC15", "MMP11", "TAGLN"],
    # prognostic, not predictive: proliferation (tumour)
    "PROLIF": ["MKI67", "TOP2A", "CCNB1", "CDK1", "BIRC5", "UBE2C", "TPX2", "MCM2"],
    # CD8 exhaustion (single-cell trajectory endpoint)
    "EXHAUSTION": ["PDCD1", "HAVCR2", "TOX", "TIGIT", "CXCL13", "ENTPD1"],
    # TLS / B-cell follicle (spatial niche in responders)
    "TLS": ["CXCL13", "CCL19", "CCL21", "LTB", "CR2", "LAMP3"],
}

HOUSEKEEPING = ["ACTB", "GAPDH", "B2M", "RPLP0", "EEF1A1", "TPT1", "MALAT1", "RPS27A"]

N_FILLER = 1840  # anonymous genes "G0001"... carrying noise + weak cell-type structure


def gene_universe():
    """Ordered unique gene list (~2000) and a dict gene -> annotation string."""
    genes, ann = [], {}
    for ct, gs in MARKERS.items():
        for g in gs:
            if g not in ann: genes.append(g); ann[g] = f"marker:{ct}"
    for m, gs in MODULES.items():
        for g in gs:
            if g not in ann: genes.append(g); ann[g] = f"module:{m}"
    for g in HOUSEKEEPING:
        genes.append(g); ann[g] = "housekeeping"
    for i in range(N_FILLER):
        g = f"G{i + 1:04d}"; genes.append(g); ann[g] = "filler"
    return genes, ann


def reference_profiles(seed=SEED):
    """Mean expression (counts per 10k, i.e. CP10k) for each cell type x gene, in the IFNG-OFF / CAF-OFF state.
    Returns (profiles [n_types x n_genes], genes). Programs are added on top via program_effects()."""
    rng = np.random.default_rng(seed)
    genes, ann = gene_universe()
    G = len(genes); idx = {g: i for i, g in enumerate(genes)}
    base = rng.lognormal(mean=-0.5, sigma=1.3, size=G)               # gene-wise baseline abundance
    P = np.tile(base, (len(CELL_TYPES), 1)) * rng.lognormal(0, 0.35, size=(len(CELL_TYPES), G))
    # weak cell-type structure in filler genes: 15% of fillers are 3x up in one random type
    fill = [i for i, g in enumerate(genes) if ann[g] == "filler"]
    for i in rng.choice(fill, size=int(0.15 * len(fill)), replace=False):
        P[rng.integers(len(CELL_TYPES)), i] *= 3.0
    for t, ct in enumerate(CELL_TYPES):
        for g in MARKERS[ct]:
            P[:, idx[g]] = np.minimum(P[:, idx[g]], 0.3)
            P[t, idx[g]] = rng.uniform(25, 80)
    for g in HOUSEKEEPING:
        P[:, idx[g]] = rng.uniform(150, 400) * rng.lognormal(0, 0.05, len(CELL_TYPES))
    for m in MODULES.values():
        for g in m:
            if ann[g].startswith("module"): P[:, idx[g]] = np.minimum(P[:, idx[g]], 0.5)
    P = P / P.sum(1, keepdims=True) * 1e4
    return P, genes


def program_effects(genes):
    """Which cell types express each program, and the fold-change at program activity 1.0."""
    idx = {g: i for i, g in enumerate(genes)}
    eff = {
        "IFNG": {"cell_types": ["CD8 T", "Macrophage", "Tumour"], "fold": 12.0},
        "TGFB_CAF": {"cell_types": ["Fibroblast"], "fold": 10.0},
        "PROLIF": {"cell_types": ["Tumour"], "fold": 15.0},
        "EXHAUSTION": {"cell_types": ["CD8 T"], "fold": 20.0},
        "TLS": {"cell_types": ["B", "CD4 T"], "fold": 15.0},
    }
    for k in eff: eff[k]["gene_idx"] = [idx[g] for g in MODULES[k]]
    return eff


# Immune phenotypes: Dirichlet concentration over CELL_TYPES, program activity means, response probability.
PHENOTYPES = {
    #            Tum  CD8  CD4  Treg  B   Mac  Fib  Endo
    "inflamed": {"alpha": [30, 14, 8, 3, 5, 10, 8, 4], "IFNG": 0.9, "TGFB_CAF": 0.2, "p_response": 0.65},
    "excluded": {"alpha": [30, 6, 6, 3, 3, 10, 22, 5], "IFNG": 0.25, "TGFB_CAF": 0.9, "p_response": 0.15},
    "desert":   {"alpha": [50, 2, 4, 2, 2, 8, 8, 4], "IFNG": 0.05, "TGFB_CAF": 0.3, "p_response": 0.05},
}
PHENOTYPE_MIX = {"inflamed": 0.38, "excluded": 0.32, "desert": 0.30}

# Study design constants (bulk discovery cohort, external validation, single-cell, spatial).
DESIGN = {
    "bulk_n": 180, "bulk_batches": {"Centre A": 100, "Centre B": 80},
    "batch_note": "Centre B enrolled more inflamed tumours (partial confounding of batch and biology).",
    "validation_n": 120, "validation_centre": "Centre C",
    "sc_patients": 12, "sc_cells_target": 6000, "sc_batches": ["3' v2", "3' v3"], "sc_doublet_rate": 0.06,
    "spatial_sections": {"S1 inflamed (R)": "inflamed", "S2 inflamed (R)": "inflamed",
                         "S3 excluded (NR)": "excluded", "S4 desert (NR)": "desert"},
}
