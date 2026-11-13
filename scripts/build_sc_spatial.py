"""Single-cell + spatial data for Tab 2 (ML in omics): writes data/omics/sc_*.json and data/omics/sp_*.json.

Imports the shared ground truth from mock_biology.py (never edits it) and simulates:
  * a 12-patient scRNA-seq study (6 R / 6 NR, two 3' chemistries) with low-quality cells, ~6% heterotypic
    doublets, a chemistry batch effect, donor-to-donor variation and a CD8 naive -> effector -> exhausted continuum;
  * four single-cell-resolution spatial sections (Xenium/CosMx-like, 150-gene panel) with tumour nests, stroma,
    vessels, a CAF-rich margin and TLS aggregates, plus Visium-like 55 um spots aggregated from the same cells;
  * a 36-section "spatial cohort" (cell types and positions only) for the spatial biomarker page.
Then it runs the analysis each page shows (QC, doublets, HVG/PCA, Leiden, UMAP, integration, annotation,
pseudotime, pseudobulk DE, Moran's I, domains, deconvolution, neighbourhood enrichment, LR scores, biomarker).

Run:  source ~/storage/sc_spatial/storage_env.sh && .venv/bin/python scripts/build_sc_spatial.py
Extra packages (installed into .venv with uv): umap-learn, leidenalg, igraph.
"""
import json, sys, time, pathlib, itertools
import numpy as np
import scipy.sparse as sp
from scipy import stats, special
from scipy.optimize import nnls, linear_sum_assignment
from scipy.spatial import Delaunay, cKDTree
from scipy.ndimage import gaussian_filter, distance_transform_edt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import adjusted_rand_score, roc_curve, roc_auc_score, silhouette_score

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import mock_biology as mb  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "omics"
OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


def rnd(x, d=2):
    """Round arrays / scalars for compact JSON."""
    if isinstance(x, (list, tuple)):
        x = np.asarray(x)
    if isinstance(x, np.ndarray):
        if x.dtype.kind in "iub":
            return x.astype(int).tolist()
        y = np.round(x.astype(float), d)
        y[~np.isfinite(y)] = 0
        if d == 0:
            return y.astype(int).tolist()
        return y.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    return round(float(x), d)


def dump(name, obj):
    p = OUT / name
    s = json.dumps(obj, separators=(",", ":"))
    p.write_text(s)
    log(f"wrote {name}: {len(s) / 1e3:.0f} kB")


# ------------------------------------------------------------------------------------------------
# Gene universe: mock_biology genes + 2 receptors added here (CXCR3, TGFBR2) + 13 mitochondrial genes
# ------------------------------------------------------------------------------------------------
CT = mb.CELL_TYPES
NT = len(CT)
P0, GENES0 = mb.reference_profiles()
_, ANN = mb.gene_universe()
EXTRA = {"CXCR3": {"CD8 T": 30, "CD4 T": 10, "Treg": 6},
         "TGFBR2": {"Fibroblast": 25, "CD8 T": 8, "CD4 T": 8, "Treg": 8, "Endothelial": 12, "Macrophage": 6}}
MT = ["MT-ND1", "MT-ND2", "MT-CO1", "MT-CO2", "MT-ATP8", "MT-ATP6", "MT-CO3", "MT-ND3", "MT-ND4L", "MT-ND4",
      "MT-ND5", "MT-ND6", "MT-CYB"]
GENES = GENES0 + list(EXTRA)
for g in EXTRA:
    ANN[g] = "receptor (added)"
G = len(GENES)
GI = {g: i for i, g in enumerate(GENES)}
P = np.zeros((NT, G))
P[:, :len(GENES0)] = P0
for g, d in EXTRA.items():
    P[:, GI[g]] = 0.2
    for c, v in d.items():
        P[CT.index(c), GI[g]] = v
P = P / P.sum(1, keepdims=True)  # fractions
EFF = mb.program_effects(GENES)
SIZE = np.array([1.6, 0.6, 0.6, 0.6, 0.6, 1.3, 1.1, 1.0])  # relative RNA content per cell type
MEM_GENES = [GI[g] for g in ["TCF7", "IL7R", "CCR7"]]
EFF_GENES = [GI[g] for g in ["GZMA", "GZMB", "PRF1"]]


def category(g):
    a = ANN.get(g, "")
    if a.startswith("marker"): return "marker"
    if a.startswith("module"): return "module:" + a.split(":")[1]
    if a == "housekeeping": return "housekeeping"
    if a.startswith("receptor"): return "receptor"
    if g.startswith("MT-"): return "mito"
    return "filler"


def cell_mu(types, prog, t=None):
    """Expected expression fractions (n x G) for cells of given types with program activities.
    prog: dict program -> activity array (n,). t: CD8 pseudotime (n,) or None."""
    M = P[types].copy()
    for name, act in prog.items():
        e = EFF[name]
        tids = [CT.index(c) for c in e["cell_types"]]
        m = np.isin(types, tids)
        if not m.any(): continue
        fac = 1 + (e["fold"] - 1) * np.clip(act[m], 0, None)[:, None]
        rows = np.where(m)[0]
        M[np.ix_(rows, e["gene_idx"])] *= fac
    if t is not None:
        m = (types == CT.index("CD8 T"))
        rows = np.where(m)[0]
        tt = t[m]
        M[np.ix_(rows, MEM_GENES)] *= (1 + 70 * np.clip(1 - 1.7 * tt, 0, 1) ** 1.5)[:, None]
        M[np.ix_(rows, EFF_GENES)] *= (1 + 30 * np.exp(-((tt - 0.55) / 0.2) ** 2))[:, None]
        ex = 1 / (1 + np.exp(-(tt - 0.66) / 0.07))
        M[np.ix_(rows, EFF["EXHAUSTION"]["gene_idx"])] *= (1 + 19 * ex)[:, None]
    return M


# =================================================================================================
# SINGLE CELL
# =================================================================================================
log("single-cell simulation")
rng = np.random.default_rng(mb.SEED + 11)
PAT = [  # id, response, phenotype, batch
    ("P01", "R", "inflamed", 0), ("P02", "R", "inflamed", 0), ("P03", "R", "excluded", 0),
    ("P04", "NR", "excluded", 0), ("P05", "NR", "desert", 0), ("P06", "NR", "inflamed", 0),
    ("P07", "R", "inflamed", 1), ("P08", "R", "inflamed", 1), ("P09", "R", "inflamed", 1),
    ("P10", "NR", "excluded", 1), ("P11", "NR", "excluded", 1), ("P12", "NR", "desert", 1),
]
NP = len(PAT)
BATCHES = mb.DESIGN["sc_batches"]
N_TARGET = mb.DESIGN["sc_cells_target"]
DBL_RATE = mb.DESIGN["sc_doublet_rate"]
N_DBL = int(round(N_TARGET * DBL_RATE))
N_RAW = N_TARGET + N_DBL  # parents of doublets are consumed

w = rng.lognormal(0, 0.25, NP)
n_per = rng.multinomial(N_RAW, w / w.sum())
T_BETA = {"inflamed": (2.4, 1.5), "excluded": (1.6, 2.2), "desert": (1.2, 3.0)}
pat_eff = rng.lognormal(0, 0.10, (NP, G))           # donor-to-donor variation (biological, not response).
# NB: sigma 0.25 on every gene made each donor its own cluster (85% of kNN neighbours from the same donor), which
# real tumour scRNA-seq does not show and which stopped any batch correction from mixing the two chemistries.
batch_eff = np.ones((2, G)); batch_eff[1] = rng.lognormal(0, 0.45, G)   # chemistry effect, gene-specific
batch_lib = np.array([1.0, 1.6])
mt_w = rng.dirichlet(np.ones(len(MT)) * 3)


def simulate_cells(pat_rows, n_per, pat_effect, batch_of_pat, batch_effect, batch_libsize, rng, lowq_rate=0.05):
    """Return dict with counts (csr, n x (G+MT)) and truth vectors."""
    types, pids, tt, ifn, tgf, pro, tls = [], [], [], [], [], [], []
    for p, (pid, resp, ph, b) in enumerate(pat_rows):
        n = n_per[p]
        alpha = np.array(mb.PHENOTYPES[ph]["alpha"], float)
        comp = rng.dirichlet(alpha * 2)
        ty = rng.choice(NT, size=n, p=comp)
        types.append(ty); pids.append(np.full(n, p))
        tt.append(rng.beta(*T_BETA[ph], size=n))
        a_ifn = np.clip(rng.normal(mb.PHENOTYPES[ph]["IFNG"], 0.08), 0, 1)
        a_tgf = np.clip(rng.normal(mb.PHENOTYPES[ph]["TGFB_CAF"], 0.08), 0, 1)
        ifn.append(np.clip(a_ifn * rng.lognormal(0, 0.25, n), 0, 1.3))
        tgf.append(np.clip(a_tgf * rng.lognormal(0, 0.25, n), 0, 1.3))
        pro.append(np.full(n, rng.uniform(0.1, 0.9)) * rng.lognormal(0, 0.3, n))
        tls.append(np.full(n, 0.6 if ph == "inflamed" else 0.1) * rng.lognormal(0, 0.3, n))
    types = np.concatenate(types); pids = np.concatenate(pids); tt = np.concatenate(tt)
    prog = {"IFNG": np.concatenate(ifn), "TGFB_CAF": np.concatenate(tgf), "PROLIF": np.concatenate(pro),
            "TLS": np.concatenate(tls)}
    n = len(types)
    M = cell_mu(types, prog, tt)
    M *= pat_effect[pids]
    b = batch_of_pat[pids]
    M *= batch_effect[b]
    M /= M.sum(1, keepdims=True)
    lib = rng.lognormal(np.log(2600), 0.45, n) * SIZE[types] * batch_libsize[b]
    fmt = np.where(types == 0, rng.beta(9, 90, n), rng.beta(4, 90, n))
    lowq = rng.random(n) < lowq_rate
    fmt[lowq] = rng.uniform(0.2, 0.6, lowq.sum())
    lib[lowq] *= rng.uniform(0.12, 0.4, lowq.sum())
    lam = M * (lib * (1 - fmt))[:, None] * rng.gamma(8, 1 / 8, M.shape)
    X = rng.poisson(lam).astype(np.int32)
    XM = rng.poisson(np.outer(lib * fmt, mt_w)).astype(np.int32)
    C = sp.csr_matrix(np.hstack([X, XM]))
    return dict(C=C, types=types, pid=pids, t=tt, lowq=lowq, batch=b, ifn=prog["IFNG"])


pat_batch = np.array([p[3] for p in PAT])
S = simulate_cells(PAT, n_per, pat_eff, pat_batch, batch_eff, batch_lib, rng)
GALL = GENES + MT
NG_ALL = len(GALL)
MT_IDX = np.arange(G, NG_ALL)

# --- doublets: merge pairs of healthy cells of different types from the same patient (same channel)
healthy = np.where(~S["lowq"])[0]
used = np.zeros(len(S["types"]), bool)
pairs = []
order = rng.permutation(healthy)
for i in order:
    if len(pairs) >= N_DBL: break
    if used[i]: continue
    cand = healthy[(S["pid"][healthy] == S["pid"][i]) & (S["types"][healthy] != S["types"][i]) & ~used[healthy]]
    cand = cand[cand != i]
    if not len(cand): continue
    j = rng.choice(cand)
    used[i] = used[j] = True
    pairs.append((i, j))
pairs = np.array(pairs)
single = np.where(~used)[0]
C = sp.vstack([S["C"][single], S["C"][pairs[:, 0]] + S["C"][pairs[:, 1]]]).tocsr()
n = C.shape[0]
truth_type = np.concatenate([S["types"][single], np.full(len(pairs), -1)])
truth_type2 = np.concatenate([np.full(len(single), -1), S["types"][pairs[:, 1]]])
truth_type1 = np.concatenate([S["types"][single], S["types"][pairs[:, 0]]])
status = np.concatenate([np.where(S["lowq"][single], 1, 0), np.full(len(pairs), 2)])  # 0 ok, 1 low quality, 2 doublet
pid = np.concatenate([S["pid"][single], S["pid"][pairs[:, 0]]])
batch = pat_batch[pid]
tcd8 = np.concatenate([S["t"][single], np.full(len(pairs), np.nan)])
perm = rng.permutation(n)  # shuffle barcode order
C = C[perm]; truth_type = truth_type[perm]; truth_type2 = truth_type2[perm]; truth_type1 = truth_type1[perm]
status = status[perm]; pid = pid[perm]; batch = batch[perm]; tcd8 = tcd8[perm]
resp = np.array([1 if PAT[p][1] == "R" else 0 for p in pid])
log(f"cells {n}: ok {np.sum(status == 0)}, lowq {np.sum(status == 1)}, doublets {np.sum(status == 2)}")

# ------------------------------------------------------------------- QC (MAD thresholds per batch)
tot = np.asarray(C.sum(1)).ravel().astype(float)
ngen = np.asarray((C > 0).sum(1)).ravel().astype(float)
mtc = np.asarray(C[:, MT_IDX].sum(1)).ravel()
pct_mt = 100 * mtc / tot


def mad(x):
    med = np.median(x)
    return med, 1.4826 * np.median(np.abs(x - med))


NMAD = 3
qc_thr = []
fail_counts = np.zeros(n, bool); fail_genes = np.zeros(n, bool); fail_mt = np.zeros(n, bool)
for b in range(2):
    m = batch == b
    mc, sc_ = mad(np.log10(tot[m])); mg, sg = mad(np.log10(ngen[m])); mm, sm = mad(pct_mt[m])
    thr = dict(batch=BATCHES[b], min_counts=10 ** (mc - NMAD * sc_), min_genes=10 ** (mg - NMAD * sg),
               max_mt=mm + NMAD * sm)
    qc_thr.append(thr)
    fail_counts[m] = tot[m] < thr["min_counts"]
    fail_genes[m] = ngen[m] < thr["min_genes"]
    fail_mt[m] = pct_mt[m] > thr["max_mt"]
qc_fail = fail_counts | fail_genes | fail_mt
# naive fixed thresholds (common copy-paste defaults) for comparison
fixed_fail = (ngen < 200) | (pct_mt > 10)
log(f"QC fail {qc_fail.sum()} (true lowq caught {np.sum(qc_fail & (status == 1))}/{np.sum(status == 1)})")

# ------------------------------------------------------------------- doublets (Scrublet-like and classifier)
passed = np.where(~qc_fail)[0]
Cp = C[passed][:, :G].astype(float)


def lognorm(Cm, target=1e4):
    s = np.asarray(Cm.sum(1)).ravel()
    s[s == 0] = 1
    X = sp.diags(target / s) @ Cm
    return X.log1p() if sp.issparse(X) else np.log1p(X)


def hvg_seurat(Cm, n_top=1000, exclude=None):
    """scanpy 'seurat' flavour: mean & dispersion on CP10k, z-scored within 20 mean bins."""
    s = np.asarray(Cm.sum(1)).ravel(); s[s == 0] = 1
    X = sp.diags(1e4 / s) @ Cm
    X = X.tocsc() if sp.issparse(X) else X
    mean = np.asarray(X.mean(0)).ravel()
    sq = np.asarray(X.multiply(X).mean(0)).ravel() if sp.issparse(X) else (X ** 2).mean(0)
    var = sq - mean ** 2
    mean = np.maximum(mean, 1e-12)
    disp = np.log(np.maximum(var / mean, 1e-12))
    lm = np.log1p(mean)
    bins = np.digitize(lm, np.linspace(lm.min(), lm.max(), 21)[1:-1])
    z = np.zeros_like(disp)
    for k in np.unique(bins):
        m = bins == k
        sd = disp[m].std() if m.sum() > 1 else 1
        z[m] = (disp[m] - disp[m].mean()) / (sd if sd > 0 else 1)
    ok = (mean > 0.0125 * 0 + 1e-3)
    if exclude is not None: ok &= ~exclude
    zz = np.where(ok, z, -np.inf)
    top = np.argsort(-zz)[:n_top]
    hv = np.zeros(len(z), bool); hv[top] = True
    return hv, lm, disp, z


hv_d, *_ = hvg_seurat(Cp, 1000)
Xp = lognorm(Cp)[:, hv_d].toarray()
mu_, sd_ = Xp.mean(0), Xp.std(0) + 1e-6
pca_d = PCA(30, random_state=0).fit((Xp - mu_) / sd_)
Zobs = pca_d.transform((Xp - mu_) / sd_)
NSIM = 2 * len(passed)
ia, ib = rng.integers(0, len(passed), NSIM), rng.integers(0, len(passed), NSIM)
Csim = Cp[ia] + Cp[ib]
Xs = lognorm(Csim)[:, hv_d].toarray()
Zsim = pca_d.transform((Xs - mu_) / sd_)
Zall = np.vstack([Zobs, Zsim])
kd = int(round(0.5 * np.sqrt(len(passed))))
nn = NearestNeighbors(n_neighbors=kd + 1).fit(Zall)
_, idx = nn.kneighbors(Zobs)
idx = idx[:, 1:]
q = (idx >= len(passed)).mean(1)
rsim = NSIM / len(passed)
score_knn = q / (q + (1 - q) * rsim + 1e-9) * 1  # Scrublet-style adjustment for simulation ratio
score_knn = q * 1.0
# classifier (scDblFinder-like): obs (0) vs simulated doublets (1), out-of-fold scores for observed cells
lt_obs = np.log10(tot[passed]); ng_obs = np.log10(ngen[passed])
tot_sim = np.asarray(Csim.sum(1)).ravel(); ng_sim = np.asarray((Csim > 0).sum(1)).ravel()
F = np.vstack([np.column_stack([Zobs, lt_obs, ng_obs, q]),
               np.column_stack([Zsim, np.log10(tot_sim), np.log10(ng_sim), np.zeros(NSIM)])])
# kNN feature for simulated cells too
_, idx_s = nn.kneighbors(Zsim)
F[len(passed):, -1] = (idx_s[:, 1:] >= len(passed)).mean(1)
yF = np.r_[np.zeros(len(passed)), np.ones(NSIM)]
clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, random_state=0)
oof = cross_val_predict(clf, F, yF, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
score_clf = oof[:len(passed)]
is_dbl = status[passed] == 2


def roc_pts(y, s, npts=60):
    fpr, tpr, _ = roc_curve(y, s)
    k = np.unique(np.linspace(0, len(fpr) - 1, min(npts, len(fpr))).astype(int))
    return dict(fpr=rnd(fpr[k], 3), tpr=rnd(tpr[k], 3), auc=rnd(roc_auc_score(y, s), 3))


roc = {"Total counts only": roc_pts(is_dbl, lt_obs), "kNN score (Scrublet-like)": roc_pts(is_dbl, score_knn),
       "Classifier (scDblFinder-like)": roc_pts(is_dbl, score_clf)}
exp_rate = DBL_RATE
n_call = int(round(exp_rate * len(passed)))
thr_clf = np.sort(score_clf)[::-1][n_call - 1]
dbl_call = score_clf >= thr_clf
log("doublet AUCs", {k: v["auc"] for k, v in roc.items()}, "calls", dbl_call.sum(), "TP", np.sum(dbl_call & is_dbl))

kept = passed[~dbl_call]
nk = len(kept)
log(f"kept {nk} cells; residual doublets {np.sum(status[kept] == 2)}, residual lowq {np.sum(status[kept] == 1)}")

# save QC json (all cells; doublet scores only for QC-passed cells)
dscore_c = np.full(n, -1.0); dscore_c[passed] = score_clf
dscore_k = np.full(n, -1.0); dscore_k[passed] = score_knn
dump("sc_qc.json", dict(
    n=int(n), batches=BATCHES, nmad=NMAD, thresholds=[{k: (rnd(v, 2) if not isinstance(v, str) else v) for k, v in t.items()} for t in qc_thr],
    total=rnd(tot, 0), ngenes=rnd(ngen, 0), pct_mt=rnd(pct_mt, 1), batch=rnd(batch), status=rnd(status),
    type=rnd(truth_type1), fail=rnd(qc_fail.astype(int)), fixed_fail=rnd(fixed_fail.astype(int)),
    dbl_clf=rnd(dscore_c, 3), dbl_knn=rnd(dscore_k, 3), dbl_threshold=rnd(thr_clf, 3), dbl_call_n=int(dbl_call.sum()),
    n_passed=int(len(passed)), roc=roc, cell_types=CT, n_sim=int(NSIM), k_knn=kd,
    summary=dict(
        n_lowq=int(np.sum(status == 1)), n_dbl=int(np.sum(status == 2)),
        qc_fail=int(qc_fail.sum()), qc_tp=int(np.sum(qc_fail & (status == 1))),
        qc_fp=int(np.sum(qc_fail & (status != 1))),
        qc_fp_by_type={CT[t]: int(np.sum(qc_fail & (status == 0) & (truth_type1 == t))) for t in range(NT)},
        fixed_fail=int(fixed_fail.sum()), fixed_tp=int(np.sum(fixed_fail & (status == 1))),
        fixed_fp_by_type={CT[t]: int(np.sum(fixed_fail & (status == 0) & (truth_type1 == t))) for t in range(NT)},
        dbl_tp=int(np.sum(dbl_call & is_dbl)), dbl_fp=int(np.sum(dbl_call & ~is_dbl)), dbl_in_passed=int(is_dbl.sum()),
        kept=int(nk), resid_dbl=int(np.sum(status[kept] == 2)), resid_lowq=int(np.sum(status[kept] == 1)))))

# ------------------------------------------------------------------- normalise, HVG, PCA
Ck = C[kept][:, :G].astype(float).tocsr()
tk = truth_type[kept]; tk1 = truth_type1[kept]; bk = batch[kept]; pk = pid[kept]; rk = resp[kept]; tck = tcd8[kept]
stk = status[kept]
lib_k = np.asarray(Ck.sum(1)).ravel()
Lk = lognorm(Ck).toarray()  # nk x G log CP10k
hv, lmean, ldisp, zdisp = hvg_seurat(Ck, 1000)
cats = np.array([category(g) for g in GENES])
planted = np.isin(cats, ["marker"]) | np.char.startswith(cats.astype(str), "module")
Xs_ = Lk[:, hv]
Xs_ = np.clip((Xs_ - Xs_.mean(0)) / (Xs_.std(0) + 1e-6), -10, 10)
pca = PCA(50, random_state=0).fit(Xs_)
Z = pca.transform(Xs_)
NPC = 20
Z20 = Z[:, :NPC]
# pitfall: PCA on unnormalised counts (log1p of raw counts, no size factor)
Xraw = np.log1p(Ck[:, hv].toarray())
Xraw = (Xraw - Xraw.mean(0)) / (Xraw.std(0) + 1e-6)
Zraw = PCA(2, random_state=0).fit_transform(Xraw)
r_raw = abs(np.corrcoef(Zraw[:, 0], np.log10(lib_k))[0, 1])
r_norm = max(abs(np.corrcoef(Z[:, i], np.log10(lib_k))[0, 1]) for i in range(2))
log(f"HVG planted recall {np.sum(hv & planted)}/{planted.sum()}, PC1~lib raw {r_raw:.2f} norm {r_norm:.2f}")

# ------------------------------------------------------------------- integration (Harmony-like)
def harmony(Z, key, K=20, theta=2.0, sigma=0.1, lam=1.0, n_iter=10, n_clus_iter=20, block=0.05, seed=0):
    """Harmony (Korsunsky et al. 2019), faithful compact re-implementation: soft k-means with a batch-diversity
    penalty (block-wise R updates, as in the reference code) alternating with mixture-of-experts ridge correction."""
    rs = np.random.default_rng(seed)
    Z = np.asarray(Z, float); n = len(Z)
    B = int(key.max() + 1)
    Phi = np.eye(B)[key].T                      # B x n
    Pr = Phi.sum(1) / n
    Phi_moe = np.vstack([np.ones(n), Phi])      # (B+1) x n, row 0 = intercept
    Zc = Z.copy()
    cos = lambda X: X / np.linalg.norm(X, axis=1, keepdims=True)
    Zn = cos(Zc)
    Y = cos(KMeans(K, n_init=4, random_state=seed).fit(Zn).cluster_centers_)
    dist = 2 * (1 - Zn @ Y.T)                   # n x K
    R = np.exp(-dist / sigma); R /= R.sum(1, keepdims=True)
    E = np.outer(R.sum(0), Pr); O = R.T @ Phi.T  # K x B
    for it in range(n_iter):
        for _ in range(n_clus_iter):
            Y = cos(R.T @ Zn)
            dist = 2 * (1 - Zn @ Y.T)
            order = rs.permutation(n)
            for blk in np.array_split(order, int(1 / block)):
                E -= np.outer(R[blk].sum(0), Pr); O -= R[blk].T @ Phi[:, blk].T
                Rb = np.exp(-dist[blk] / sigma) * (((E + 1) / (O + 1)) ** theta)[:, key[blk]].T
                Rb /= Rb.sum(1, keepdims=True)
                R[blk] = Rb
                E += np.outer(Rb.sum(0), Pr); O += Rb.T @ Phi[:, blk].T
        Zc = Z.copy()
        for k in range(K):
            PhiR = Phi_moe * R[:, k][None, :]
            A = PhiR @ Phi_moe.T + np.diag([0] + [lam] * B)
            W = np.linalg.solve(A, PhiR @ Z)    # (B+1) x d
            W[0] = 0                            # keep the intercept (cell-type position)
            Zc -= (W.T @ PhiR).T
        Zn = cos(Zc)
    return Zc


log("integration")
Zh = harmony(Z20, bk, K=20, theta=2.0)
# over-correction: patient as the batch key, strong diversity penalty, almost no ridge shrinkage -> patient-level
# biology (the IFN-g program differs by patient phenotype) is regressed out along with the technical effect
Zo = harmony(Z20, pk, K=8, theta=6.0, sigma=0.2, lam=0.01, n_iter=20)
EMB = {"uncorrected": Z20, "harmony": Zh, "over": Zo}


def knn_idx(Zm, k=30):
    nn_ = NearestNeighbors(n_neighbors=k + 1).fit(Zm)
    return nn_.kneighbors(Zm)[1][:, 1:]


singlet_k = tk >= 0
sub_sil = rng.choice(np.where(singlet_k)[0], 2500, replace=False)
# per-cell IFN-g program score (from expression, independent of the embedding correction)
IFN_IDX = [GI[g] for g in mb.MODULES["IFNG"]]
ifn_score = Lk[:, IFN_IDX].mean(1)
ifn_cells = np.isin(tk, [CT.index("CD8 T"), CT.index("Macrophage"), CT.index("Tumour")])


def r2_linear(Xf, y):
    Xf = np.column_stack([np.ones(len(Xf)), Xf])
    beta, *_ = np.linalg.lstsq(Xf, y, rcond=None)
    return 1 - np.sum((y - Xf @ beta) ** 2) / np.sum((y - y.mean()) ** 2)


def knn_graph(Zm, k=15):
    import igraph as ig
    I = knn_idx(Zm, k)
    rows = np.repeat(np.arange(len(Zm)), k)
    edges = np.unique(np.sort(np.column_stack([rows, I.ravel()]), 1), axis=0)
    return ig.Graph(n=len(Zm), edges=edges.tolist())


def leiden(g, res, seed=1):
    import leidenalg
    part = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, resolution_parameter=res, seed=seed)
    lab = np.array(part.membership)
    # relabel by size
    _, inv, cnt = np.unique(lab, return_inverse=True, return_counts=True)
    order = np.argsort(-cnt); remap = np.empty_like(order); remap[order] = np.arange(len(order))
    return remap[inv]


RES = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0]
clus = {}
metrics = {}
for name, E in EMB.items():
    I = knn_idx(E, 30)
    bfrac = bk[I].mean(1)
    pglob = bk.mean()
    ilisi = 1 / (bfrac ** 2 + (1 - bfrac) ** 2)
    # kBET-like: chi-square of neighbourhood batch counts vs global proportion
    obs1 = bk[I].sum(1); exp1 = 30 * pglob
    chi = (obs1 - exp1) ** 2 / exp1 + ((30 - obs1) - (30 - exp1)) ** 2 / (30 - exp1)
    kbet = np.mean(stats.chi2.sf(chi, 1) > 0.05)
    tcount = np.stack([(tk[I] == t).mean(1) for t in range(NT)], 1)
    clisi = 1 / np.maximum((tcount ** 2).sum(1), 1e-9)
    pcount = np.stack([(pk[I] == p).mean(1) for p in range(NP)], 1)
    plisi = 1 / (pcount ** 2).sum(1)
    asw = silhouette_score(E[sub_sil], tk[sub_sil])
    asw_b = silhouette_score(E[sub_sil], bk[sub_sil])
    r2 = r2_linear(E[ifn_cells], ifn_score[ifn_cells])
    g = knn_graph(E, 15)
    labs = {}
    aris = []
    ncl = []
    for r in RES:
        lab = leiden(g, r)
        labs[r] = lab
        aris.append(adjusted_rand_score(tk[singlet_k], lab[singlet_k]))
        ncl.append(int(lab.max() + 1))
    clus[name] = labs
    metrics[name] = dict(ilisi=float(np.mean(ilisi)), ilisi_norm=float(np.mean(ilisi) - 1), kbet=float(kbet),
                         clisi_norm=float(1 - (np.median(clisi[singlet_k]) - 1) / (NT - 1)), asw_type=float((asw + 1) / 2),
                         asw_batch=float(1 - abs(asw_b)), plisi=float(np.median(plisi)), ifn_r2=float(r2),
                         ari=[float(a) for a in aris], n_clusters=ncl,
                         ari_best=float(max(aris)), res_best=RES[int(np.argmax(aris))])
    log(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in metrics[name].items()})

# batch purity per cluster (uncorrected, chosen resolution) -> how many clusters are >80% one batch
def batch_split(lab):
    out = []
    for c in range(lab.max() + 1):
        m = lab == c
        f = bk[m].mean()
        tt = tk[m]; tt = tt[tt >= 0]
        maj = int(np.bincount(tt, minlength=NT).argmax()) if len(tt) else -1
        out.append(dict(n=int(m.sum()), frac_v3=round(float(f), 3), major=maj,
                        purity=round(float(np.mean(tt == maj)) if len(tt) else 0, 3)))
    return out


cluster_info = {name: {str(r): batch_split(clus[name][r]) for r in RES} for name in ["uncorrected", "harmony", "over"]}

log("UMAP")
import umap  # noqa: E402
UM = {}
for name, E in EMB.items():
    UM[name] = umap.UMAP(n_neighbors=15, min_dist=0.35, random_state=0).fit_transform(E)

# ------------------------------------------------------------------- annotation
log("annotation")
PAT_REF = [("A1", "R", "inflamed", 0), ("A2", "R", "inflamed", 0), ("A3", "NR", "excluded", 0),
           ("A4", "NR", "excluded", 0), ("A5", "NR", "desert", 0), ("A6", "R", "inflamed", 0)]
rng_ref = np.random.default_rng(mb.SEED + 99)
nref = rng_ref.multinomial(3000, np.ones(6) / 6)
ref_beff = rng_ref.lognormal(0, 0.3, (1, G))
Rsim = simulate_cells(PAT_REF, nref, rng_ref.lognormal(0, 0.25, (6, G)), np.zeros(6, int), ref_beff,
                      np.array([1.3]), rng_ref, lowq_rate=0.0)
Cref = Rsim["C"][:, :G].astype(float)
Lref = lognorm(Cref).toarray()
yref = Rsim["types"]
feat = np.where(hv)[0]


def zs(X):
    return (X - X.mean(0)) / (X.std(0) + 1e-3)


Xr = zs(Lref[:, feat]); Xq = zs(Lk[:, feat])
ann = {}
for nm, keep_t in [("lt_full", np.arange(NT)), ("lt_notreg", np.array([t for t in range(NT) if CT[t] != "Treg"]))]:
    m = np.isin(yref, keep_t)
    lr = LogisticRegression(C=0.05, max_iter=3000)
    lr.fit(Xr[m], yref[m])
    pr = lr.predict_proba(Xq)
    ann[nm] = dict(pred=lr.classes_[pr.argmax(1)], conf=pr.max(1))
# marker-score annotation
zall = zs(Lk)
msc = np.column_stack([zall[:, [GI[g] for g in mb.MARKERS[c]]].mean(1) for c in CT])
ann["marker"] = dict(pred=msc.argmax(1), conf=np.exp(msc * 3).max(1) / np.exp(msc * 3).sum(1))
truth_rows = tk.copy(); truth_rows[truth_rows < 0] = NT  # row NT = doublet
ann_out = {}
for nm, a in ann.items():
    cm = np.zeros((NT + 1, NT), int)
    np.add.at(cm, (truth_rows, a["pred"]), 1)
    acc = np.mean(a["pred"][singlet_k] == tk[singlet_k])
    ann_out[nm] = dict(confusion=cm.tolist(), acc=round(float(acc), 4),
                       conf_singlet=round(float(np.median(a["conf"][singlet_k])), 3),
                       conf_doublet=round(float(np.median(a["conf"][~singlet_k])), 3) if (~singlet_k).any() else None,
                       conf_treg=round(float(np.median(a["conf"][tk == CT.index("Treg")])), 3),
                       recall={CT[t]: round(float(np.mean(a["pred"][tk == t] == t)), 3) for t in range(NT)})
    log(nm, "acc", round(acc, 3), "conf treg", ann_out[nm]["conf_treg"], "conf dbl", ann_out[nm]["conf_doublet"])
# confidence histograms (singlet vs doublet vs Treg)
def hist(v, bins):
    return np.histogram(v, bins=bins)[0].tolist()


cbins = np.linspace(0, 1, 21)
for nm in ann:
    a = ann[nm]
    ann_out[nm]["hist_singlet"] = hist(a["conf"][singlet_k & (tk != CT.index("Treg"))], cbins)
    ann_out[nm]["hist_treg"] = hist(a["conf"][tk == CT.index("Treg")], cbins)
    ann_out[nm]["hist_doublet"] = hist(a["conf"][~singlet_k], cbins)

# ------------------------------------------------------------------- trajectory on CD8 T cells (label-transfer CD8)
log("trajectory")
cd8 = np.where(ann["lt_full"]["pred"] == CT.index("CD8 T"))[0]
Lc = Lk[cd8]
hv8, *_ = hvg_seurat(Ck[cd8], 300)


def diffmap(center_batch, npc=10, kn=30):
    """Diffusion map (Haghverdi 2016, alpha=1) on PCA of the CD8 HVGs. center_batch: remove per-chemistry gene means
    first (the CD8 subset is re-embedded from expression, so it must be batch-corrected again; without it DC1 = chemistry)."""
    X = Lc[:, hv8].copy()
    if center_batch:
        for b in (0, 1):
            X[bk[cd8] == b] -= X[bk[cd8] == b].mean(0)
    Xc_ = np.clip(zs(X), -8, 8)
    Z_ = PCA(npc, random_state=0).fit_transform(Xc_)
    dist8, ind8 = NearestNeighbors(n_neighbors=kn).fit(Z_).kneighbors(Z_)
    sig = dist8[:, 7]
    rows = np.repeat(np.arange(len(cd8)), kn)
    cols = ind8.ravel()
    vals = np.exp(-dist8.ravel() ** 2 / (sig[rows] * sig[cols]))
    Km = sp.csr_matrix((vals, (rows, cols)), shape=(len(cd8), len(cd8)))
    Km = Km.maximum(Km.T)
    q8 = np.asarray(Km.sum(1)).ravel()
    Km = sp.diags(1 / q8) @ Km @ sp.diags(1 / q8)  # alpha=1 density normalisation
    dsum = np.asarray(Km.sum(1)).ravel()
    Asym = sp.diags(1 / np.sqrt(dsum)) @ Km @ sp.diags(1 / np.sqrt(dsum))
    ev, evecs = np.linalg.eigh(Asym.toarray())
    o = np.argsort(-ev)[:11]
    ev, evecs = ev[o], evecs[:, o]
    psi = evecs / np.sqrt(dsum)[:, None]
    psi = psi / np.linalg.norm(psi, axis=0)
    return psi[:, 1:] * (ev[1:] / (1 - ev[1:]))[None, :], ev, Z_


DC_raw, _, _ = diffmap(False)
DC, evals, Zc8 = diffmap(True)


def dpt(root, D_=None):
    D_ = DC if D_ is None else D_
    d = np.linalg.norm(D_ - D_[root], axis=1)
    return (d - d.min()) / (d.max() - d.min())


naive_sc = Lc[:, MEM_GENES].mean(1)
effe_sc = Lc[:, EFF_GENES].mean(1) - Lc[:, EFF["EXHAUSTION"]["gene_idx"]].mean(1)
root_good = int(np.argmax(naive_sc))
root_bad = int(np.argmax(effe_sc))
pt_good, pt_bad = dpt(root_good), dpt(root_bad)
pt_nobatch = dpt(root_good, DC_raw)
pc1 = Zc8[:, 0]
tt8 = tck[cd8]
okt = np.isfinite(tt8) & (tk[cd8] == CT.index("CD8 T"))
rho_good = stats.spearmanr(pt_good[okt], tt8[okt])[0]
rho_bad = stats.spearmanr(pt_bad[okt], tt8[okt])[0]
rho_pc1 = abs(stats.spearmanr(pc1[okt], tt8[okt])[0])
rho_nobatch = stats.spearmanr(pt_nobatch[okt], tt8[okt])[0]
dc1_batch = abs(stats.spearmanr(DC_raw[:, 0], bk[cd8])[0])
log(f"CD8 n={len(cd8)} rho good {rho_good:.3f} bad {rho_bad:.3f} pc1 {rho_pc1:.3f} no-batch-fix {rho_nobatch:.3f} (DC1~batch {dc1_batch:.2f})")
TRAJ_GENES = ["TCF7", "IL7R", "CCR7", "GZMA", "GZMB", "PRF1", "NKG7", "IFNG", "PDCD1", "HAVCR2", "TOX", "TIGIT",
              "CXCL13", "ENTPD1", "CD8A", "CXCR3"]
nb = 20


def trend(pt):
    qb = np.quantile(pt, np.linspace(0, 1, nb + 1))
    bi = np.clip(np.searchsorted(qb, pt, side="right") - 1, 0, nb - 1)
    out = {}
    for g in TRAJ_GENES:
        v = Lc[:, GI[g]]
        out[g] = rnd([v[bi == b].mean() if np.any(bi == b) else 0 for b in range(nb)], 3)
    centres = [float(pt[bi == b].mean()) if np.any(bi == b) else 0 for b in range(nb)]
    return dict(x=rnd(centres, 3), y=out)


# gene-vs-pseudotime correlations for module genes
ex_rho = {g: rnd(stats.spearmanr(pt_good, Lc[:, GI[g]])[0], 3) for g in TRAJ_GENES}
dump("sc_trajectory.json", dict(
    n=int(len(cd8)), genes=TRAJ_GENES, dc1=rnd(DC[:, 0], 4), dc2=rnd(DC[:, 1], 4),
    pt=rnd(pt_good, 3), pt_bad=rnd(pt_bad, 3), pt_nobatch=rnd(pt_nobatch, 3), batch=rnd(bk[cd8]),
    dc1_raw=rnd(DC_raw[:, 0], 4), dc2_raw=rnd(DC_raw[:, 1], 4), rho_nobatch=rnd(rho_nobatch, 3), dc1_batch=rnd(dc1_batch, 3), true_t=rnd(np.nan_to_num(tt8, nan=-1), 3),
    true_cd8=rnd(okt.astype(int)), patient=rnd(pk[cd8]), resp=rnd(rk[cd8]),
    expr={g: rnd(Lc[:, GI[g]], 2) for g in TRAJ_GENES},
    root=root_good, root_bad=root_bad, rho_good=rnd(rho_good, 3), rho_bad=rnd(rho_bad, 3), rho_pc1=rnd(rho_pc1, 3),
    trend_good=trend(pt_good), trend_bad=trend(pt_bad), gene_rho=ex_rho,
    exhaustion=mb.MODULES["EXHAUSTION"], patients=[p[0] for p in PAT], phen=[p[2] for p in PAT],
    pt_by_resp=dict(R=hist(pt_good[rk[cd8] == 1], np.linspace(0, 1, 21)), NR=hist(pt_good[rk[cd8] == 0], np.linspace(0, 1, 21))),
    n_by_resp=dict(R=int(np.sum(rk[cd8] == 1)), NR=int(np.sum(rk[cd8] == 0))),
    frac_exh=dict(R=rnd(np.mean(pt_good[rk[cd8] == 1] > 0.6), 3), NR=rnd(np.mean(pt_good[rk[cd8] == 0] > 0.6), 3)),
    evals=rnd(evals[1:], 4)))

# ------------------------------------------------------------------- pseudobulk vs naive DE
log("pseudobulk DE")
annot = ann["lt_full"]["pred"]
DE_TYPES = ["CD8 T", "CD4 T", "B", "Macrophage", "Tumour", "Fibroblast"]
PROG_TYPES = {name: EFF[name]["cell_types"] for name in EFF}


def planted_set(ct):
    s = set()
    for name in ["IFNG", "TGFB_CAF", "TLS"]:
        if ct in PROG_TYPES[name]: s |= set(mb.MODULES[name])
    if ct == "CD8 T":
        s |= set(mb.MODULES["EXHAUSTION"]) | {"TCF7", "IL7R", "CCR7", "GZMA", "GZMB", "PRF1"}
    return s


def bh(p):
    p = np.asarray(p); o = np.argsort(p); r = np.empty_like(o); r[o] = np.arange(1, len(p) + 1)
    q = p * len(p) / r
    qs = np.minimum.accumulate(q[o][::-1])[::-1]
    out = np.empty_like(q); out[o] = np.minimum(qs, 1)
    return out


def limma_lite(Y, grp):
    """Moderated t on log2-CPM pseudobulk (limma-style empirical Bayes with fitted prior)."""
    a, b = Y[grp == 1], Y[grp == 0]
    n1, n2 = len(a), len(b)
    diff = a.mean(0) - b.mean(0)
    df = n1 + n2 - 2
    s2 = (((a - a.mean(0)) ** 2).sum(0) + ((b - b.mean(0)) ** 2).sum(0)) / df
    s2 = np.maximum(s2, 1e-8)
    # method-of-moments fit of scaled inverse chi-square prior on log s2 (limma fitFDist, simplified)
    z = np.log(s2) - (special.digamma(df / 2) - np.log(df / 2))
    zbar = z.mean()
    ev = np.var(z, ddof=1) - special.polygamma(1, df / 2)
    d0 = 2 * 50.0 if ev <= 0 else max(2 * _inv_trigamma(ev), 1e-3)
    s02 = np.exp(zbar + special.digamma(d0 / 2) - np.log(d0 / 2))
    s2m = (d0 * s02 + df * s2) / (d0 + df)
    t = diff / np.sqrt(s2m * (1 / n1 + 1 / n2))
    p = 2 * stats.t.sf(np.abs(t), df + d0)
    return diff, p


def _inv_trigamma(x):
    y = 0.5 + 1 / x
    for _ in range(50):
        tri = special.polygamma(1, y)
        dif = tri * (1 - tri / x) / special.polygamma(2, y)
        y = y + dif
        if np.all(np.abs(dif / y) < 1e-8): break
    return float(y)


# balanced permutations of patient labels: each fake group has 3 true R + 3 true NR (true signal cancels)
Ridx = [p for p in range(NP) if PAT[p][1] == "R"]
NRidx = [p for p in range(NP) if PAT[p][1] == "NR"]
combos = [(a, b) for a in itertools.combinations(Ridx, 3) for b in itertools.combinations(NRidx, 3)]
prng = np.random.default_rng(7)
perm_sets = [combos[i] for i in prng.choice(len(combos), 20, replace=False)]
pat_resp = np.array([1 if p[1] == "R" else 0 for p in PAT])

pb_out = {}
Ck_csr = Ck.tocsr()
for ct in DE_TYPES:
    t = CT.index(ct)
    cells = np.where(annot == t)[0]
    pc = pk[cells]
    pats = [p for p in range(NP) if np.sum(pc == p) >= 10]
    cells = cells[np.isin(pc, pats)]
    pc = pk[cells]
    Lct = Lk[cells]
    expr_frac = (Ck_csr[cells] > 0).mean(0).A1
    gsel = np.where((expr_frac >= 0.05) & np.array([not g.startswith("MT-") for g in GENES]))[0]
    # pseudobulk
    PBc = np.vstack([np.asarray(Ck_csr[cells[pc == p]][:, gsel].sum(0)).ravel() for p in pats])
    cpm = PBc / PBc.sum(1, keepdims=True) * 1e6
    Y = np.log2(cpm + 1)
    grp_pat = pat_resp[pats]

    def run(grp_pat_, grp_cell_):
        lfc_pb, p_pb = limma_lite(Y, grp_pat_)
        p_nv = stats.mannwhitneyu(Lct[grp_cell_ == 1][:, gsel], Lct[grp_cell_ == 0][:, gsel], axis=0).pvalue
        p_nv = np.where(np.isfinite(p_nv), p_nv, 1.0)
        lfc_nv = (np.log2(np.expm1(Lct[grp_cell_ == 1][:, gsel]).mean(0) + 0.1) -
                  np.log2(np.expm1(Lct[grp_cell_ == 0][:, gsel]).mean(0) + 0.1))
        return lfc_pb, p_pb, lfc_nv, p_nv

    lfc_pb, p_pb, lfc_nv, p_nv = run(grp_pat, resp[kept][cells])
    q_pb, q_nv = bh(p_pb), bh(p_nv)
    pl = planted_set(ct)
    is_pl = np.array([GENES[g] in pl for g in gsel])
    perm_res = []
    for (aR, aN) in perm_sets:
        fake = np.array([1 if p in aR or p in aN else 0 for p in range(NP)])
        _, pp, _, pn = run(fake[pats], fake[pc])
        perm_res.append(dict(pb=int(np.sum(bh(pp) < 0.05)), naive=int(np.sum(bh(pn) < 0.05))))
    # p-value histograms for the first permutation
    fake = np.array([1 if p in perm_sets[0][0] or p in perm_sets[0][1] else 0 for p in range(NP)])
    _, pp0, _, pn0 = run(fake[pats], fake[pc])
    pb_out[ct] = dict(
        n_cells=int(len(cells)), n_patients=len(pats), n_R=int(grp_pat.sum()), n_NR=int(len(pats) - grp_pat.sum()),
        n_cells_R=int(np.sum(resp[kept][cells] == 1)), n_cells_NR=int(np.sum(resp[kept][cells] == 0)),
        genes=[GENES[g] for g in gsel], planted=rnd(is_pl.astype(int)), cat=[category(GENES[g]) for g in gsel],
        lfc_pb=rnd(lfc_pb, 2), lfc_nv=rnd(lfc_nv, 2),
        nlp_pb=rnd(-np.log10(np.maximum(p_pb, 1e-300)), 2), nlp_nv=rnd(-np.log10(np.maximum(p_nv, 1e-300)), 2),
        sig_pb=rnd((q_pb < 0.05).astype(int)), sig_nv=rnd((q_nv < 0.05).astype(int)),
        n_planted=int(is_pl.sum()),
        tp_pb=int(np.sum((q_pb < 0.05) & is_pl)), fp_pb=int(np.sum((q_pb < 0.05) & ~is_pl)),
        tp_nv=int(np.sum((q_nv < 0.05) & is_pl)), fp_nv=int(np.sum((q_nv < 0.05) & ~is_pl)),
        perm=perm_res, perm_hist_pb=hist(pp0, np.linspace(0, 1, 21)), perm_hist_nv=hist(pn0, np.linspace(0, 1, 21)),
        perm_groups=[[PAT[p][0] for p in perm_sets[0][0] + perm_sets[0][1]]],
        pb_sample_mean=rnd(Y.mean(1), 2))
    log(ct, f"cells {len(cells)} pats {len(pats)} genes {len(gsel)} | real: PB tp {pb_out[ct]['tp_pb']} fp {pb_out[ct]['fp_pb']}, "
            f"naive tp {pb_out[ct]['tp_nv']} fp {pb_out[ct]['fp_nv']} | perm mean PB {np.mean([r['pb'] for r in perm_res]):.1f} "
            f"naive {np.mean([r['naive'] for r in perm_res]):.1f}")
dump("sc_pseudobulk.json", dict(types=DE_TYPES, patients=[p[0] for p in PAT], resp=[p[1] for p in PAT],
                                phen=[p[2] for p in PAT], de=pb_out, n_perm=len(perm_sets)))

# ------------------------------------------------------------------- per-cell plotting file + summaries
NSUB = 3000
sub = np.sort(rng.choice(nk, NSUB, replace=False))
SHOW_GENES = ["EPCAM", "CD3E", "CD8A", "CD4", "FOXP3", "MS4A1", "CD68", "COL1A1", "PECAM1", "MKI67", "CXCL9",
              "GZMB", "PDCD1", "TCF7", "CXCL13", "POSTN"]
dump("sc_cells.json", dict(
    n=NSUB, n_kept=int(nk), cell_types=CT, batches=BATCHES, patients=[p[0] for p in PAT],
    resp=[p[1] for p in PAT], phen=[p[2] for p in PAT],
    type=rnd(tk[sub]), batch=rnd(bk[sub]), patient=rnd(pk[sub]), lib=rnd(lib_k[sub], 0),
    umap={k: [rnd(UM[k][sub, 0], 2), rnd(UM[k][sub, 1], 2)] for k in UM},
    pca=[rnd(Z[sub, 0], 2), rnd(Z[sub, 1], 2)], pca_raw=[rnd(Zraw[sub, 0], 2), rnd(Zraw[sub, 1], 2)],
    res=RES, leiden={k: {str(r): rnd(clus[k][r][sub]) for r in RES} for k in ["uncorrected", "harmony", "over"]},
    ann={k: rnd(v["pred"][sub]) for k, v in ann.items()}, conf={k: rnd(v["conf"][sub], 2) for k, v in ann.items()},
    genes=SHOW_GENES, expr={g: rnd(Lk[sub, GI[g]], 1) for g in SHOW_GENES},
    # 10 nearest neighbours of each plotted cell among the plotted cells, in the 20-PC space (for drawing the kNN graph)
    knn={k: rnd(knn_idx(EMB[k][sub], 10)) for k in ["uncorrected", "harmony"]}))

dump("sc_embedding.json", dict(
    n_genes=G, n_hvg=int(hv.sum()), genes=GENES, cat=[category(g) for g in GENES],
    log_mean=rnd(lmean, 3), log_disp=rnd(ldisp, 2), z_disp=rnd(zdisp, 2), hvg=rnd(hv.astype(int)),
    planted_total=int(planted.sum()), planted_hvg=int(np.sum(hv & planted)),
    filler_total=int(np.sum(cats == "filler")), filler_hvg=int(np.sum(hv & (cats == "filler"))),
    var_ratio=rnd(pca.explained_variance_ratio_, 4), npc=NPC,
    r_lib_raw=rnd(r_raw, 3), r_lib_norm=rnd(r_norm, 3),
    lib_median=rnd(np.median(lib_k), 0), lib_q=rnd(np.quantile(lib_k, [0.05, 0.95]), 0)))

dump("sc_integration.json", dict(metrics={k: {kk: (rnd(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()}
                                          for k, v in metrics.items()}, res=RES, clusters=cluster_info,
                                 n_patients=NP, n_types=NT))
dump("sc_annotation.json", dict(methods=ann_out, cell_types=CT, rows=CT + ["Doublet (missed)"], n_ref=int(len(yref)),
                                n_ref_treg=int(np.sum(yref == CT.index("Treg"))), conf_bins=rnd(cbins, 2),
                                n_features=int(len(feat))))

# =================================================================================================
# SPATIAL
# =================================================================================================
log("spatial simulation")
srng = np.random.default_rng(mb.SEED + 202)
REGIONS = ["Tumour nest", "Margin", "Stroma", "TLS"]
#            Tum   CD8   CD4   Treg  B     Mac   Fib   Endo
TP = {
    "inflamed": {0: [.78, .11, .02, .01, .00, .06, .015, .005], 1: [.08, .14, .10, .03, .02, .15, .40, .08],
                 2: [.01, .05, .07, .02, .03, .13, .57, .12]},
    "excluded": {0: [.93, .008, .005, .002, .00, .04, .01, .005], 1: [.04, .20, .12, .05, .02, .12, .40, .05],
                 2: [.01, .06, .06, .02, .02, .11, .62, .10]},
    "desert": {0: [.95, .004, .004, .002, .00, .03, .006, .004], 1: [.06, .02, .04, .01, .01, .14, .62, .10],
               2: [.01, .01, .03, .01, .01, .12, .68, .13]},
}
TLS_P = [0, .08, .28, .04, .52, .03, .02, .03]
NEST_FRAC = {"inflamed": 0.45, "excluded": 0.42, "desert": 0.55}


def layout(ph, size, rng_, cd8_mult=1.0, nest_frac=None, n_tls=None, spacing=17.0):
    px = 5.0
    npx = int(size / px)
    f = gaussian_filter(rng_.normal(size=(npx, npx)), 18)
    nf = nest_frac if nest_frac is not None else NEST_FRAC[ph]
    nest = f > np.quantile(f, 1 - nf)
    dist_out = distance_transform_edt(~nest) * px  # distance from stroma pixel to nest
    # cells on a jittered grid
    g = np.arange(spacing / 2, size, spacing)
    xx, yy = np.meshgrid(g, g)
    xy = np.column_stack([xx.ravel(), yy.ravel()]) + rng_.uniform(-6, 6, (xx.size, 2))
    xy = np.clip(xy, 0, size - 1e-3)
    pi_ = (xy / px).astype(int)
    in_nest = nest[pi_[:, 1], pi_[:, 0]]
    d = dist_out[pi_[:, 1], pi_[:, 0]]
    region = np.where(in_nest, 0, np.where(d <= 45, 1, 2))
    # TLS aggregates in stroma far from nests (inflamed only)
    ntl = n_tls if n_tls is not None else (3 if ph == "inflamed" else 0)
    tls_centres = []
    cand = np.where(dist_out.ravel() > 110)[0]
    for _ in range(ntl):
        if not len(cand): break
        c = cand[rng_.integers(len(cand))]
        cy, cx = np.unravel_index(c, dist_out.shape)
        cxy = np.array([cx, cy]) * px
        if any(np.linalg.norm(cxy - t) < 200 for t in tls_centres): continue
        tls_centres.append(cxy)
    for c in tls_centres:
        rad = rng_.uniform(55, 75)
        region[(np.linalg.norm(xy - c, axis=1) < rad) & (region == 2)] = 3
    # vessels
    stroma_pts = np.where(region == 2)[0]
    vessel = np.zeros(len(xy), bool)
    for _ in range(int(size / 1000 * 14)):
        if not len(stroma_pts): break
        c = xy[stroma_pts[rng_.integers(len(stroma_pts))]]
        vessel |= (np.linalg.norm(xy - c, axis=1) < rng_.uniform(14, 22)) & (region == 2)
    types = np.zeros(len(xy), int)
    for r in range(4):
        m = region == r
        pr = np.array(TLS_P if r == 3 else TP[ph][r], float)
        pr[1] *= cd8_mult
        pr /= pr.sum()
        types[m] = rng_.choice(NT, size=m.sum(), p=pr)
    vm = vessel & (rng_.random(len(xy)) < 0.8)
    types[vm] = CT.index("Endothelial")
    return dict(xy=xy, region=region, types=types, vessel=vessel, size=size)


def programs_spatial(L, ph, rng_):
    xy, types, region = L["xy"], L["types"], L["region"]
    tree = cKDTree(xy)
    nb_ = tree.query_ball_point(xy, 50)
    is8 = types == CT.index("CD8 T")
    f8 = np.array([is8[n_].mean() for n_ in nb_])
    base = {"inflamed": 0.15, "excluded": 0.05, "desert": 0.0}[ph]
    ifn = np.clip(base + 3.0 * f8 + rng_.normal(0, 0.05, len(xy)), 0, 1)
    # excluded: CAFs are TGF-b-activated in a band around the nests (the barrier), less so deeper in stroma
    tgf_base = {"inflamed": 0.2, "excluded": 0.25, "desert": 0.3}[ph]
    tgf = np.full(len(xy), tgf_base)
    if ph == "excluded": tgf[region == 1] = 1.0
    tgf = np.clip(tgf + rng_.normal(0, 0.08, len(xy)), 0, 1.2)
    pro = np.clip(rng_.uniform(0.3, 0.6) + rng_.normal(0, 0.2, len(xy)), 0, 1)
    tls = np.where(region == 3, 1.0, 0.05)
    t = np.where(region == 0, rng_.beta(5, 2, len(xy)), np.where(region == 1, rng_.beta(2.5, 2.5, len(xy)),
                                                                  rng_.beta(1.5, 3, len(xy))))
    return {"IFNG": ifn, "TGFB_CAF": tgf, "PROLIF": pro, "TLS": tls}, t, f8


# gene panel (150)
cats_all = {g: category(g) for g in GENES}
panel = [g for g in GENES if cats_all[g] in ("marker", "receptor") or cats_all[g].startswith("module")]
panel += ["ACTB", "B2M"]
fillers = [g for g in GENES if cats_all[g] == "filler"]
fi = np.array([GI[g] for g in fillers])
ratio = P[:, fi].max(0) / np.median(P[:, fi], 0)
expr_ok = P[:, fi].mean(0) * 1e4 > 0.8
struct = [fillers[i] for i in np.where((ratio > 2.5) & expr_ok)[0]]
flat = [fillers[i] for i in np.where((ratio < 1.6) & expr_ok)[0]]
n_left = 150 - len(panel)
n_struct = n_left // 2
panel += list(srng.choice(struct, n_struct, replace=False)) + list(srng.choice(flat, n_left - n_struct, replace=False))
PANEL = panel
PI = np.array([GI[g] for g in PANEL])
NPAN = len(PANEL)
log("panel", NPAN, "struct fillers", n_struct)

SECTIONS = list(mb.DESIGN["spatial_sections"].items())
platform = srng.lognormal(0, 0.3, G)  # Visium vs scRNA gene-wise capture differences
secs = []
for si, (sname, ph) in enumerate(SECTIONS):
    L = layout(ph, 1000.0, srng)
    prog, t, f8 = programs_spatial(L, ph, srng)
    M = cell_mu(L["types"], prog, t)
    M /= M.sum(1, keepdims=True)
    Mp = M[:, PI]; Mp /= Mp.sum(1, keepdims=True)
    lib = srng.lognormal(np.log(230), 0.35, len(Mp)) * SIZE[L["types"]]
    X = srng.poisson(Mp * lib[:, None])
    L.update(name=sname, ph=ph, prog=prog, t=t, f8=f8, M=M, X=X, lib=lib)
    secs.append(L)
    log(sname, len(L["types"]), "cells; types", np.bincount(L["types"], minlength=NT).tolist(),
        "regions", np.bincount(L["region"], minlength=4).tolist())


def delaunay_edges(xy, maxlen=35.0):
    tri = Delaunay(xy)
    s = tri.simplices
    e = np.vstack([s[:, [0, 1]], s[:, [1, 2]], s[:, [0, 2]]])
    e = np.unique(np.sort(e, 1), axis=0)
    ln = np.linalg.norm(xy[e[:, 0]] - xy[e[:, 1]], axis=1)
    return e[ln <= maxlen]


def adj_from_edges(e, n_):
    A = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n_, n_)).tocsr()
    return A + A.T


for L in secs:
    L["edges"] = delaunay_edges(L["xy"])
    L["A"] = adj_from_edges(L["edges"], len(L["xy"]))
    Xn = L["X"] / np.maximum(L["X"].sum(1, keepdims=True), 1) * 100
    L["logn"] = np.log1p(Xn)

# ---------------------------------------------------------------- sp_data.json: per-cell maps
MAP_GENES = ["EPCAM", "CD8A", "CXCR3", "CXCL9", "CXCL10", "GZMB", "PDCD1", "CXCL13", "MS4A1", "CCL19",
             "COL1A1", "POSTN", "TGFB1", "TGFBR2", "PECAM1", "CD68", "MKI67"]
spi = {g: i for i, g in enumerate(PANEL)}
flat_in = [g for g in PANEL if g in flat]
struct_in = [g for g in PANEL if g in struct]
MAP_GENES += [flat_in[0], struct_in[0]]
FLAT_EX, STRUCT_EX = flat_in[0], struct_in[0]

# ---------------------------------------------------------------- Moran's I
log("Moran's I")


# p from a normal approximation to the permutation null (z = (I - mean_null) / sd_null, as Squidpy's pval_z_sim):
# the raw permutation p floors at 1/(nperm+1), which Benjamini-Hochberg over 150 genes cannot pass.
def morans_z(Xv, A, nperm=199, rng_=None):
    W = sp.diags(1 / np.maximum(np.asarray(A.sum(1)).ravel(), 1)) @ A
    Zm = Xv - Xv.mean(0)
    den = (Zm ** 2).sum(0)
    den[den == 0] = 1
    I = (Zm * (W @ Zm)).sum(0) / den
    nul = np.zeros((nperm, Xv.shape[1]))
    for b in range(nperm):
        pz = Zm[rng_.permutation(len(Zm))]
        nul[b] = (pz * (W @ pz)).sum(0) / den
    ge = (nul >= I).sum(0)
    z = (I - nul.mean(0)) / (nul.std(0) + 1e-12)
    return I, (ge + 1) / (nperm + 1), z, stats.norm.sf(z)


svg = []
for L in secs:
    Xl = L["logn"]
    I_raw, ps_raw, z_raw, p_raw = morans_z(Xl, L["A"], 199, srng)
    # residual after removing cell-type means: spatial pattern beyond cell-type composition
    R_ = Xl.copy()
    for t in range(NT):
        m = L["types"] == t
        if m.any(): R_[m] -= Xl[m].mean(0)
    I_res, ps_res, z_res, p_res = morans_z(R_, L["A"], 199, srng)
    svg.append(dict(I=rnd(I_raw, 3), p_sim=rnd(ps_raw, 3), z=rnd(z_raw, 1), p=rnd(p_raw, 6), q=rnd(bh(p_raw), 4),
                    I_res=rnd(I_res, 3), p_sim_res=rnd(ps_res, 3), z_res=rnd(z_res, 1), p_res=rnd(p_res, 6), q_res=rnd(bh(p_res), 4)))
    log(L["name"], "n SVG raw", int(np.sum(bh(p_raw) < 0.05)), "resid", int(np.sum(bh(p_res) < 0.05)),
        "CXCL9 I", round(float(I_raw[spi["CXCL9"]]), 3), round(float(I_res[spi["CXCL9"]]), 3))
pan_cat = [category(g) for g in PANEL]
for g in PANEL:
    if g in flat: pan_cat[spi[g]] = "filler (flat)"
    elif g in struct: pan_cat[spi[g]] = "filler (cell-type)"
dump("sp_svg.json", dict(genes=PANEL, cat=pan_cat, sections=[L["name"] for L in secs], res=svg, nperm=199,
                         flat_example=FLAT_EX, struct_example=STRUCT_EX))

# ---------------------------------------------------------------- spatial domains (expression-only vs neighbour-smoothed)
log("domains")
allX = np.vstack([L["logn"] for L in secs])
allX = (allX - allX.mean(0)) / (allX.std(0) + 1e-6)
pcs = PCA(20, random_state=0).fit_transform(allX)
offs = np.cumsum([0] + [len(L["xy"]) for L in secs])
nbmean = np.zeros_like(pcs)
for si, L in enumerate(secs):
    tree = cKDTree(L["xy"])
    d_, i_ = tree.query(L["xy"], k=16)
    wts = np.exp(-(d_[:, 1:] ** 2) / (2 * 30.0 ** 2))
    wts /= wts.sum(1, keepdims=True)
    P_ = pcs[offs[si]:offs[si + 1]]
    nbmean[offs[si]:offs[si + 1]] = (P_[i_[:, 1:]] * wts[:, :, None]).sum(1)
LAMS = [0.0, 0.2, 0.4, 0.6, 0.8, 0.95]
true_reg = np.concatenate([L["region"] for L in secs])
true_ty = np.concatenate([L["types"] for L in secs])
dom = {}
dom_ari = []
dom_ari_type = []
KDOM = 4
for lam in LAMS:
    Fd = np.hstack([np.sqrt(1 - lam) * pcs, np.sqrt(lam) * nbmean])
    lab = KMeans(KDOM, n_init=10, random_state=0).fit_predict(Fd)
    # match clusters to true regions (Hungarian) for consistent colours
    ctab = np.zeros((KDOM, 4))
    np.add.at(ctab, (lab, true_reg), 1)
    r_, c_ = linear_sum_assignment(-ctab)
    mp = np.empty(KDOM, int); mp[r_] = c_
    lab = mp[lab]
    dom[lam] = lab
    dom_ari.append(adjusted_rand_score(true_reg, lab))
    dom_ari_type.append(adjusted_rand_score(true_ty, lab))
    per_sec = [adjusted_rand_score(true_reg[offs[i]:offs[i + 1]], lab[offs[i]:offs[i + 1]]) for i in range(len(secs))]
    log(f"lambda {lam}: ARI region {dom_ari[-1]:.3f} ARI celltype {dom_ari_type[-1]:.3f} per-section {np.round(per_sec, 2)}")
dom_sec_ari = {str(lam): [rnd(adjusted_rand_score(true_reg[offs[i]:offs[i + 1]], dom[lam][offs[i]:offs[i + 1]]), 3)
                          for i in range(len(secs))] for lam in LAMS}
dom_comp = {}
for lam in LAMS:
    ctab = np.zeros((KDOM, NT))
    np.add.at(ctab, (dom[lam], true_ty), 1)
    dom_comp[str(lam)] = rnd(ctab / np.maximum(ctab.sum(1, keepdims=True), 1), 3)
dump("sp_domains.json", dict(lambdas=LAMS, k=KDOM, regions=REGIONS, ari=rnd(dom_ari, 3), ari_type=rnd(dom_ari_type, 3),
                             ari_section=dom_sec_ari, composition=dom_comp, cell_types=CT,
                             labels={str(l): [rnd(dom[l][offs[i]:offs[i + 1]]) for i in range(len(secs))] for l in LAMS}))

# ---------------------------------------------------------------- neighbourhood enrichment + LR
log("niches")
NE = []
for L in secs:
    A = L["A"]
    oh = np.eye(NT)[L["types"]]
    obs = oh.T @ (A @ oh)
    nulls = []
    for _ in range(300):
        ohp = oh[srng.permutation(len(oh))]
        nulls.append(ohp.T @ (A @ ohp))
    nulls = np.array(nulls)
    z = (obs - nulls.mean(0)) / (nulls.std(0) + 1e-9)
    present = np.bincount(L["types"], minlength=NT)
    NE.append(dict(z=rnd(z, 2), count=rnd(present)))
LR_PAIRS = [("CXCL9", "CXCR3"), ("CXCL10", "CXCR3"), ("CXCL13", "CXCR5"), ("TGFB1", "TGFBR2")]
LR = []
for L in secs:
    A = L["A"]
    Xl = L["logn"]
    res = {}
    for lg, rc in LR_PAIRS:
        l_ = Xl[:, spi[lg]]; r_ = Xl[:, spi[rc]]
        # score = mean over neighbouring pairs of ligand(i) * receptor(j)
        e = L["edges"]
        obs = (np.sum(l_[e[:, 0]] * r_[e[:, 1]]) + np.sum(l_[e[:, 1]] * r_[e[:, 0]])) / (2 * len(e))
        null = []
        for _ in range(300):
            pp = srng.permutation(len(l_))
            lp, rp = l_[pp], r_[pp]
            null.append((np.sum(lp[e[:, 0]] * rp[e[:, 1]]) + np.sum(lp[e[:, 1]] * rp[e[:, 0]])) / (2 * len(e)))
        null = np.array(null)
        # per-cell receiver score: receptor(j) * mean ligand of neighbours
        deg = np.maximum(np.asarray(A.sum(1)).ravel(), 1)
        recv = r_ * (A @ l_) / deg
        res[f"{lg}→{rc}"] = dict(score=rnd(obs, 4), z=rnd((obs - null.mean()) / (null.std() + 1e-12), 2),
                                 recv=rnd(recv, 2), recv_by_region=[rnd(recv[L["region"] == k].mean() if np.any(L["region"] == k) else 0, 3) for k in range(4)],
                                 recv_by_type=[rnd(recv[L["types"] == t].mean() if np.any(L["types"] == t) else 0, 4) for t in range(NT)])
    LR.append(res)
    log(L["name"], {k: v["z"] for k, v in res.items()})
dump("sp_niches.json", dict(sections=[L["name"] for L in secs], cell_types=CT, enrichment=NE, lr=LR,
                            pairs=[f"{a}→{b}" for a, b in LR_PAIRS], regions=REGIONS, nperm=300))

# ---------------------------------------------------------------- Visium-like spots + deconvolution
log("spots + deconvolution")
ref_types = ann["lt_full"]["pred"]
REFM = np.zeros((NT, G)); REF_LIB = np.zeros(NT)
for t in range(NT):
    m = ref_types == t
    cpm_ = np.asarray(Ck[m].sum(0)).ravel()
    REFM[t] = cpm_ / cpm_.sum()
    REF_LIB[t] = np.median(lib_k[m])
REF_LIB_REL = REF_LIB / REF_LIB.mean()
# marker genes for NNLS: top 40 by specificity per type
spec = np.log((REFM + 1e-7) / (REFM.mean(0, keepdims=True) + 1e-7))
# cell-state programme genes (IFN-g, CAF, proliferation, ...) vary between reference and tissue, so they are left out
state_g = np.array([category(g).startswith("module") for g in GENES])
spec[:, state_g] = -np.inf
dg = np.unique(np.concatenate([np.argsort(-spec[t])[:40] for t in range(NT)]))
spots = []
for si, L in enumerate(secs):
    xy = L["xy"]
    pitch, rad = 100.0, 27.5
    ys = np.arange(60, 1000 - 40, pitch * np.sqrt(3) / 2)
    tree = cKDTree(xy)
    lib_full = srng.lognormal(np.log(1500), 0.3, len(xy)) * SIZE[L["types"]]
    for ri, y in enumerate(ys):
        for x in np.arange(60 + (pitch / 2 if ri % 2 else 0), 1000 - 40, pitch):
            ids = tree.query_ball_point([x, y], rad)
            if not ids: continue
            ids = np.array(ids)
            lam_ = (L["M"][ids] * lib_full[ids, None]).sum(0) * platform
            yv = srng.poisson(lam_ * srng.gamma(10, 0.1, G))
            comp = np.bincount(L["types"][ids], minlength=NT)
            spots.append(dict(sec=si, x=x, y=y, y_counts=yv, comp=comp, ids=ids,
                              region=int(np.bincount(L["region"][ids], minlength=4).argmax())))
YS = np.array([s["y_counts"] for s in spots], float)
COMP = np.array([s["comp"] for s in spots], float)
TRUEF = COMP / COMP.sum(1, keepdims=True)
# NNLS on marker genes, proportions scale
W_nnls = np.zeros((len(spots), NT))
Rn = REFM[:, dg].T
for i in range(len(spots)):
    yy = YS[i, dg] / max(YS[i].sum(), 1)
    w_, _ = nnls(Rn, yy)
    W_nnls[i] = w_
# Poisson model with gene-wise platform factor (Cell2location-like, simplified): y_sg ~ Pois(sum_t w_st * ref_tg * m_g)
Yg = YS[:, dg]
Rg = REFM[:, dg]
Wp = np.full((len(spots), NT), Yg.sum(1, keepdims=True) / NT)
mg = np.ones(len(dg))
for it in range(400):
    mu_ = Wp @ (Rg * mg) + 1e-12
    Wp *= ((Yg / mu_) @ (Rg * mg).T) / ((Rg * mg).sum(1)[None, :])
    if it % 10 == 9:
        mu_ = Wp @ Rg + 1e-12
        mg = Yg.sum(0) / mu_.sum(0)
        mg /= np.exp(np.mean(np.log(mg)))


def to_frac(W, size_correct):
    W = W / np.maximum(W.sum(1, keepdims=True), 1e-12)
    if size_correct:
        W = W / REF_LIB_REL[None, :]
        W = W / W.sum(1, keepdims=True)
    return W


DEC = {"nnls": {False: to_frac(W_nnls, False), True: to_frac(W_nnls, True)},
       "poisson": {False: to_frac(Wp, False), True: to_frac(Wp, True)}}
dec_metrics = {}
for mname in DEC:
    for sc_ in (False, True):
        Wf = DEC[mname][sc_]
        rs = [float(np.corrcoef(Wf[:, t], TRUEF[:, t])[0, 1]) for t in range(NT)]
        rmse = float(np.sqrt(np.mean((Wf - TRUEF) ** 2)))
        bias = [float(np.mean(Wf[:, t] - TRUEF[:, t])) for t in range(NT)]
        dec_metrics[f"{mname}_{int(sc_)}"] = dict(r=rnd(rs, 3), rmse=rnd(rmse, 4), bias=rnd(bias, 4))
        log(mname, "size-corrected" if sc_ else "raw", "rmse", round(rmse, 4), "r", np.round(rs, 2), "bias", np.round(bias, 3))
dump("sp_deconvolution.json", dict(
    cell_types=CT, sections=[L["name"] for L in secs], n_spots=len(spots), n_genes=int(len(dg)),
    sec=[s["sec"] for s in spots], x=[rnd(s["x"], 0) for s in spots], y=[rnd(s["y"], 0) for s in spots],
    region=[s["region"] for s in spots], n_cells=rnd(COMP.sum(1)), true=rnd(TRUEF, 3),
    est={f"{m}_{int(c)}": rnd(DEC[m][c], 3) for m in DEC for c in (False, True)}, metrics=dec_metrics,
    ref_lib=rnd(REF_LIB_REL, 2), spot_diam=55, pitch=100,
    cells_per_spot=dict(median=rnd(np.median(COMP.sum(1)), 1), min=int(COMP.sum(1).min()), max=int(COMP.sum(1).max()))))

dump("sp_data.json", dict(
    sections=[L["name"] for L in secs], phen=[L["ph"] for L in secs], size=1000, cell_types=CT, regions=REGIONS,
    panel=PANEL, n_panel=NPAN, genes=MAP_GENES, flat_example=FLAT_EX, struct_example=STRUCT_EX,
    cells=[dict(x=rnd(L["xy"][:, 0], 0), y=rnd(L["xy"][:, 1], 0), type=rnd(L["types"]), region=rnd(L["region"]),
                expr={g: rnd(L["X"][:, spi[g]]) for g in MAP_GENES}, lib=rnd(L["X"].sum(1)),
                n=int(len(L["xy"])), n_edges=int(len(L["edges"])),
                mean_deg=rnd(2 * len(L["edges"]) / len(L["xy"]), 2)) for L in secs]))

# ---------------------------------------------------------------- spatial biomarker (4 sections + 36-section cohort)
log("biomarker cohort")
brng = np.random.default_rng(mb.SEED + 303)


def section_metrics(L, prog):
    xy, types = L["xy"], L["types"]
    tree = cKDTree(xy)
    _, nbi = tree.query(xy, k=11)
    tum_frac = (types[nbi] == 0).mean(1)
    tumour_comp = tum_frac >= 0.5  # data-derived tumour compartment (no ground-truth regions used)
    is8 = types == CT.index("CD8 T")
    area_t = tumour_comp.sum(); area_s = (~tumour_comp).sum()
    d_t = (is8 & tumour_comp).sum() / max(area_t, 1) * 100
    d_s = (is8 & ~tumour_comp).sum() / max(area_s, 1) * 100
    infil = np.log2((d_t + 0.2) / (d_s + 0.2))
    cd8_frac = is8.mean()
    tt = cKDTree(xy[types == 0])
    dmin = tt.query(xy[is8])[0] if is8.any() else np.array([0.0])
    cx = P[:, GI["CXCL9"]][types] * np.where(np.isin(types, [0, 1, 5]), 1 + 11 * prog["IFNG"], 1)
    ifn_bulk = np.log2(cx.mean() * 1e4 + 1)
    return dict(cd8_frac=float(cd8_frac), d_tumour=float(d_t), d_stroma=float(d_s), infil=float(infil),
                ifn_bulk=float(ifn_bulk), cd8_dist_med=float(np.median(dmin)), frac_inside=float(np.mean(tumour_comp[is8])) if is8.any() else 0.0)


cohort = []
for i, L in enumerate(secs):
    m = section_metrics(L, L["prog"])
    m.update(name=L["name"], ph=L["ph"], resp=1 if "(R)" in L["name"] else 0, main=1)
    cohort.append(m)
for ph in ["inflamed", "excluded", "desert"]:
    for j in range(12):
        mult = brng.lognormal(0, 0.5)
        L = layout(ph, 600.0, brng, cd8_mult=mult, nest_frac=brng.uniform(0.35, 0.55), n_tls=brng.integers(0, 3) if ph == "inflamed" else 0)
        prog, _, _ = programs_spatial(L, ph, brng)
        m = section_metrics(L, prog)
        m["ifn_bulk"] += brng.normal(0, 0.35)  # bulk assay noise
        m.update(name=f"C{len(cohort) - 3:02d}", ph=ph, resp=int(brng.random() < mb.PHENOTYPES[ph]["p_response"]), main=0)
        cohort.append(m)
ph_arr = np.array([c["ph"] for c in cohort]); resp_arr = np.array([c["resp"] for c in cohort])
aucs = {}
for k in ["cd8_frac", "ifn_bulk", "infil", "frac_inside", "d_tumour"]:
    v = np.array([c[k] for c in cohort])
    mie = np.isin(ph_arr, ["inflamed", "excluded"])
    aucs[k] = dict(inf_vs_exc=rnd(roc_auc_score(ph_arr[mie] == "inflamed", v[mie]), 3),
                   resp=rnd(roc_auc_score(resp_arr, v), 3))
log("biomarker AUCs", aucs, "responders", resp_arr.sum(), "/", len(resp_arr))
dump("sp_biomarker.json", dict(cohort=[{k: (rnd(v, 3) if isinstance(v, float) else v) for k, v in c.items()} for c in cohort],
                               auc=aucs, n=len(cohort), n_resp=int(resp_arr.sum()),
                               maps=[dict(name=L["name"], cd8=rnd(np.where(L["types"] == 1)[0])) for L in secs]))
total = sum(p.stat().st_size for p in OUT.glob("s[cp]_*.json"))
log(f"done; sc_/sp_ JSON total {total / 1e6:.2f} MB")
