"""Build every data/omics/bulk_*.json used by the Tab 2 bulk RNA-seq pages.

One mock anti-PD-1 NSCLC study, simulated from the shared ground truth in scripts/mock_biology.py:
  * discovery cohort: 180 patients from Centre A (100) and Centre B (80); Centre B enrolled more inflamed
    tumours, so batch and biology are partially confounded.
  * external validation cohort: 120 patients from Centre C (different technical shift and case mix),
    plus 12 "bridging" RNA aliquots sequenced at both Centre A and Centre C.

Every analysis below is a real, correctly implemented method (numpy / scipy / scikit-learn); the JS pages only
draw what is precomputed here. Run:
  source ~/storage/sc_spatial/storage_env.sh && cd /home/abalasubramanian/storage/ml_dash && .venv/bin/python scripts/build_bulk.py
"""
import json, sys, time, pathlib, warnings
import numpy as np
from scipy import stats, optimize, special
from sklearn.decomposition import PCA
from sklearn.covariance import MinCovDet
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, silhouette_score, adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.inspection import permutation_importance
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import mock_biology as mb  # noqa: E402  (shared ground truth; never edited here)

OUT = ROOT / "data" / "omics"
OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


# ----------------------------------------------------------------------------------------------- JSON helpers
def sig(x, s=4):
    if x is None or not np.isfinite(x): return None
    if x == 0: return 0
    return float(f"{x:.{s}g}")


def clean(o, s=4):
    if isinstance(o, dict): return {str(k): clean(v, s) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v, s) for v in o]
    if isinstance(o, np.ndarray): return clean(o.tolist(), s)
    if isinstance(o, (np.bool_, bool)): return bool(o)
    if isinstance(o, (np.integer, int)): return int(o)
    if isinstance(o, (np.floating, float)): return sig(float(o), s)
    return o


SIZES = {}


def dump(name, obj, s=4):
    p = OUT / f"bulk_{name}.json"
    txt = json.dumps(clean(obj, s), separators=(",", ":"))
    p.write_text(txt)
    SIZES[p.name] = len(txt)
    log(f"wrote {p.name}: {len(txt) / 1024:.0f} KB")


# ----------------------------------------------------------------------------------------------- ground truth
P_REF, GENES = mb.reference_profiles()
_, ANN = mb.gene_universe()
G = len(GENES)
GI = {g: i for i, g in enumerate(GENES)}
EFF = mb.program_effects(GENES)
CT = mb.CELL_TYPES
PH = list(mb.PHENOTYPES)
PROGS = ["IFNG", "TGFB_CAF", "PROLIF", "EXHAUSTION", "TLS"]


def group_of(g):
    """Colour group used on every page: which planted structure a gene belongs to."""
    a = ANN[g]
    if g in mb.MODULES["IFNG"]: return "IFNG module"
    if g in mb.MODULES["TGFB_CAF"]: return "CAF/TGF-b module"
    if g in mb.MODULES["PROLIF"]: return "PROLIF module"
    if a.startswith("module"): return "other module"
    if a.startswith("marker"): return "cell-type marker"
    if a == "housekeeping": return "housekeeping"
    return "filler"


GROUPS = ["IFNG module", "CAF/TGF-b module", "PROLIF module", "other module", "cell-type marker", "housekeeping", "filler"]

rng = np.random.default_rng(mb.SEED + 7)
clip = lambda v: float(np.clip(v, 0, 1))


def draw_programs(ph):
    p = mb.PHENOTYPES[ph]
    ifng = clip(rng.normal(p["IFNG"], 0.15))
    return {
        "IFNG": ifng,
        "TGFB_CAF": clip(rng.normal(p["TGFB_CAF"], 0.15)),
        "PROLIF": float(rng.beta(2, 2)),                 # independent of phenotype: prognostic, not predictive
        "EXHAUSTION": clip(0.7 * ifng + rng.normal(0, 0.1)),
        "TLS": clip(rng.normal(0.45 if ph == "inflamed" else 0.1, 0.15)),
    }


def patient_mean(frac, prog):
    """Bulk expected expression = sum_t fraction_t x (reference_t with program fold-changes applied)."""
    Pt = P_REF.copy()
    for k, a in prog.items():
        e = EFF[k]
        for ct in e["cell_types"]:
            Pt[CT.index(ct), e["gene_idx"]] *= e["fold"] ** a
    mu = frac @ Pt
    return mu / mu.sum()


PHI = 0.05 * rng.lognormal(0, 0.4, G)          # gene-wise NB dispersion (biological over-dispersion)
logit = lambda p: np.log(p / (1 - p))


def nb_counts(prop, depth, phi=PHI):
    m = prop * depth[:, None]
    return rng.poisson(rng.gamma(1 / phi, m * phi)).astype(np.int64)


def simulate(n_by_centre_pheno, shifts, depth_mu, depth_sd, prefix):
    rows = []
    for centre, mix in n_by_centre_pheno.items():
        for ph, n in mix.items():
            rows += [(centre, ph)] * n
    order = rng.permutation(len(rows))
    rows = [rows[i] for i in order]
    n = len(rows)
    frac = np.zeros((n, len(CT))); prog = []; prop = np.zeros((n, G))
    resp = np.zeros(n, int); time_ = np.zeros(n); event = np.zeros(n, int)
    for i, (c, ph) in enumerate(rows):
        frac[i] = rng.dirichlet(mb.PHENOTYPES[ph]["alpha"])
        pr = draw_programs(ph); prog.append(pr)
        prop[i] = patient_mean(frac[i], pr)
        p0, m0 = mb.PHENOTYPES[ph]["p_response"], mb.PHENOTYPES[ph]["IFNG"]
        p_resp = special.expit(logit(p0) + 4.0 * (pr["IFNG"] - m0))       # response: phenotype, modulated by IFNG
        resp[i] = rng.random() < p_resp
        lam = np.log(2) / 5.0 * np.exp(-1.5 * resp[i] + 0.6 * (pr["PROLIF"] - 0.5) / 0.2236)  # PFS hazard (months)
        T = rng.exponential(1 / lam)
        C = min(rng.uniform(12, 48), rng.exponential(120))
        time_[i] = min(T, C); event[i] = int(T <= C)
    centre = np.array([r[0] for r in rows]); pheno = np.array([r[1] for r in rows])
    obs = prop * np.stack([shifts[c] for c in centre])            # technical (centre) shift, gene-wise multiplicative
    obs /= obs.sum(1, keepdims=True)
    depth = rng.lognormal(np.log(depth_mu), depth_sd, n)
    return dict(ids=[f"{prefix}{i + 1:03d}" for i in range(n)], centre=centre, pheno=pheno, frac=frac,
                prog={k: np.array([p[k] for p in prog]) for k in PROGS}, prop=prop, obs=obs, depth=depth,
                resp=resp, time=time_, event=event)


SHIFT = {"Centre A": np.ones(G), "Centre B": rng.lognormal(0, 0.4, G), "Centre C": rng.lognormal(0, 0.35, G)}
log("simulating discovery cohort")
DISC_MIX = {"Centre A": {"inflamed": 19, "excluded": 41, "desert": 40}, "Centre B": {"inflamed": 52, "excluded": 16, "desert": 12}}
D = simulate(DISC_MIX, SHIFT, 15e6, 0.3, "D")
# planted QC failures
QC_PLANT = {"low depth": 23, "degraded (strong)": 131, "degraded (mild)": 64}
D["depth"][QC_PLANT["low depth"]] = 0.45e6
for k, s in (("degraded (strong)", 0.9), ("degraded (mild)", 0.5)):
    i = QC_PLANT[k]
    D["obs"][i] *= rng.lognormal(0, s, G); D["obs"][i] /= D["obs"][i].sum()
D["counts"] = nb_counts(D["obs"], D["depth"])

log("simulating validation cohort")
VAL_MIX = {"Centre C": {"inflamed": 34, "excluded": 38, "desert": 48}}
V = simulate(VAL_MIX, SHIFT, 8e6, 0.4, "V")
V["counts"] = nb_counts(V["obs"], V["depth"], PHI * 1.3)


def logcpm(counts):
    lib = counts.sum(1, keepdims=True)
    return np.log2(counts / lib * 1e6 + 1)


# ============================================================================================== 1. QC
log("QC")
L_all = logcpm(D["counts"])
lib = D["counts"].sum(1)
detected = (D["counts"] >= 5).sum(1)
C_ = np.corrcoef(L_all)
np.fill_diagonal(C_, np.nan)
medcor = np.nanmedian(C_, 1)
hv = np.argsort(-L_all.var(0))[:1000]
pca_qc = PCA(10, random_state=0).fit(L_all[:, hv])
S_qc = pca_qc.transform(L_all[:, hv])
mcd = MinCovDet(random_state=0).fit(S_qc[:, :5])
rdist = np.sqrt(mcd.mahalanobis(S_qc[:, :5]))
iso = IsolationForest(n_estimators=500, random_state=0).fit(S_qc[:, :5])
iso_score = -iso.score_samples(S_qc[:, :5])          # higher = more anomalous


def mad_flag(v, k, side):
    med = np.median(v); mad = 1.4826 * np.median(np.abs(v - med))
    return (v < med - k * mad) if side == "low" else (v > med + k * mad)


K_QC = 3.0
flag_cor = mad_flag(medcor, K_QC, "low")
flag_dist = mad_flag(np.log(rdist), K_QC, "high")
flag_lib = mad_flag(np.log10(lib), K_QC, "low")
EXCLUDE = np.where(flag_cor | flag_dist)[0]          # expression-based failures are excluded; low depth alone is not
log("QC flagged lib", np.where(flag_lib)[0], "cor", np.where(flag_cor)[0], "dist", np.where(flag_dist)[0])
keep = np.setdiff1d(np.arange(len(lib)), EXCLUDE)

# normalisation demo: per-sample quantiles under 3 transforms, for 40 samples ordered by depth
demo = np.argsort(lib)[np.linspace(0, len(lib) - 1, 40).astype(int)]
qs = [0.1, 0.25, 0.5, 0.75, 0.9]
cnt = D["counts"].astype(float)
trans = {"raw": np.log10(cnt + 1), "cpm": np.log10(cnt / lib[:, None] * 1e6 + 1), "logcpm": L_all}
norm_demo = {k: [np.quantile(v[i], qs).tolist() for i in demo] for k, v in trans.items()}

# gene filter (filterByExpr-like): CPM >= 1 in >= 10% of kept samples
cpm_keep = D["counts"][keep] / lib[keep, None] * 1e6
gkeep = np.where((cpm_keep >= 1).mean(0) >= 0.10)[0]
log("genes kept", len(gkeep), "of", G)

dump("qc", {
    "ids": D["ids"], "centre": D["centre"], "pheno": D["pheno"],
    "lib": lib, "detected": detected, "medcor": medcor, "rdist": rdist, "iso": iso_score,
    "pc": S_qc[:, :2], "pcvar": pca_qc.explained_variance_ratio_[:2],
    "planted": {k: D["ids"][i] for k, i in QC_PLANT.items()},
    "excluded": [D["ids"][i] for i in EXCLUDE], "k_default": K_QC,
    "norm_demo": {"ids": [D["ids"][i] for i in demo], "lib": lib[demo], "q": qs, **norm_demo},
    "n_genes": G, "n_genes_kept": len(gkeep), "n_planted_degraded_counts": 0,
})

# ============================================================================================== downstream matrices
ids = [D["ids"][i] for i in keep]
n = len(keep)
centre = D["centre"][keep]; pheno = D["pheno"][keep]; y = D["resp"][keep]
bB = (centre == "Centre B").astype(float)
GK = [GENES[i] for i in gkeep]
GGROUP = [group_of(g) for g in GK]
X_raw = L_all[keep][:, gkeep]
TRUE_FRAC = D["frac"][keep]
PROG = {k: v[keep] for k, v in D["prog"].items()}
T_pfs = D["time"][keep]; E_pfs = D["event"][keep]
X_truth = np.log2(D["prop"][keep][:, gkeep] * 1e6 + 1)       # noise-free, batch-free expected logCPM
ph_dum = np.stack([(pheno == "excluded"), (pheno == "desert")], 1).astype(float)


def mod_idx(name):
    return [GK.index(g) for g in mb.MODULES[name] if g in GK]


IFNG_IDX, CAF_IDX, PROLIF_IDX = mod_idx("IFNG"), mod_idx("TGFB_CAF"), mod_idx("PROLIF")

# ============================================================================================== 2. batch
log("batch correction")


def ls_correct(Y, b, covars=None):
    """Location/scale batch correction toward the reference batch (b==0), ComBat-style but without the
    empirical-Bayes shrinkage. covars = biology to PROTECT (kept in the model so it is not regressed away)."""
    Dm = np.column_stack([np.ones(len(b)), b] + ([covars] if covars is not None else []))
    beta, *_ = np.linalg.lstsq(Dm, Y, rcond=None)
    fit = Dm @ beta
    R = Y - fit
    sA = R[b == 0].std(0, ddof=1); sB = R[b == 1].std(0, ddof=1)
    Yc = Y.copy()
    base = fit - np.outer(b, beta[1])                  # fitted value with the batch term removed
    Yc[b == 1] = base[b == 1] + R[b == 1] * (sA / sB)
    return Yc


X_naive = ls_correct(X_raw, bB)
X_prot = ls_correct(X_raw, bB, ph_dum)
VERS = {"raw": X_raw, "naive": X_naive, "protected": X_prot}
hv_b = np.argsort(-X_raw.var(0))[:1000]


def ifng_score(X):
    return X[:, IFNG_IDX].mean(1)


def effect(X):
    s = ifng_score(X)
    return s[pheno == "inflamed"].mean() - s[pheno != "inflamed"].mean()


truth_eff = effect(X_truth)
batch_out = {"ids": ids, "centre": centre, "pheno": pheno, "resp": y, "truth_effect": truth_eff, "versions": {}}
for k, X in VERS.items():
    p = PCA(10, random_state=0).fit(X[:, hv_b]); S = p.transform(X[:, hv_b])
    r2 = [np.corrcoef(S[:, j], bB)[0, 1] ** 2 for j in range(5)]
    batch_out["versions"][k] = {
        "pc": S[:, :2], "var": p.explained_variance_ratio_[:2], "r2_batch": r2,
        "sil_batch": silhouette_score(S, centre), "sil_pheno": silhouette_score(S, pheno),
        "ifng": ifng_score(X), "effect": effect(X),
        "auc_ifng": roc_auc_score(y, ifng_score(X)),
    }
    log(k, "r2 PC1/2", np.round(r2[:2], 3), "effect", round(effect(X), 3), "truth", round(truth_eff, 3))
batch_out["counts"] = {c: {p_: int(((centre == c) & (pheno == p_)).sum()) for p_ in PH} for c in ["Centre A", "Centre B"]}
dump("batch", batch_out)


# ============================================================================================== 3. subtypes
log("subtypes (consensus clustering)")


def consensus(S, k, reps=100, frac=0.8, seed=0):
    r = np.random.default_rng(seed); m = len(S)
    M = np.zeros((m, m)); I = np.zeros((m, m))
    for b in range(reps):
        sub = r.choice(m, int(frac * m), replace=False)
        lab = KMeans(k, n_init=5, random_state=b).fit_predict(S[sub])
        same = lab[:, None] == lab[None, :]
        I[np.ix_(sub, sub)] += 1; M[np.ix_(sub, sub)] += same
    C = M / np.maximum(I, 1); np.fill_diagonal(C, 1)
    Z = linkage(squareform(1 - C, checks=False), "average")
    cl = fcluster(Z, k, "maxclust") - 1
    tri = C[np.triu_indices(m, 1)]
    pac = float(((tri > 0.1) & (tri < 0.9)).mean())
    return C, cl, pac


sub_out = {"ks": [2, 3, 4, 5, 6], "data": {}}
for vname, X in (("protected", X_prot), ("raw", X_raw)):
    hv = np.argsort(-X.var(0))[:1000]
    S = PCA(10, random_state=0).fit_transform(X[:, hv] - X[:, hv].mean(0))
    res = {"pc": S[:, :2], "k": {}}
    for k in sub_out["ks"]:
        C, cl, pac = consensus(S, k)
        res["k"][k] = {"cl": cl, "pac": pac, "sil": silhouette_score(S, cl),
                       "ari_pheno": adjusted_rand_score(pheno, cl), "ari_batch": adjusted_rand_score(centre, cl)}
        if vname == "protected":
            # consensus matrix for a 90-patient subsample, as 0..99 ints (keeps the JSON small)
            sel = np.sort(np.random.default_rng(1).choice(n, 90, replace=False))
            o = sel[np.lexsort((pheno[sel], cl[sel]))]
            res["k"][k]["cm_idx"] = o
            res["k"][k]["cm"] = np.round(C[np.ix_(o, o)] * 99).astype(int)
        log(vname, k, {kk: round(float(v), 3) for kk, v in res["k"][k].items() if kk in ("pac", "sil", "ari_pheno", "ari_batch")})
    sub_out["data"][vname] = res
sub_out.update({"ids": ids, "pheno": pheno, "centre": centre, "resp": y})
dump("subtypes", sub_out)

# ============================================================================================== 4. DE
log("differential expression")


def bh(p):
    m = len(p); o = np.argsort(p); q = p[o] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]; out = np.empty(m); out[o] = np.minimum(q, 1); return out


def welch(X, g):
    a, b = X[g == 1], X[g == 0]
    ma, mb_, va, vb = a.mean(0), b.mean(0), a.var(0, ddof=1), b.var(0, ddof=1)
    se = np.sqrt(va / len(a) + vb / len(b)); t = (ma - mb_) / se
    df = se ** 4 / ((va / len(a)) ** 2 / (len(a) - 1) + (vb / len(b)) ** 2 / (len(b) - 1))
    return ma - mb_, 2 * stats.t.sf(np.abs(t), df), t


def trigamma_inv(x):
    yv = 0.5 + 1 / x
    for _ in range(50):
        tri = special.polygamma(1, yv); d = tri * (1 - tri / x) / special.polygamma(2, yv); yv = yv + d
        if np.all(np.abs(d / yv) < 1e-8): break
    return yv


def linmod(X, Dm, j=1, moderate=False):
    """Per-gene OLS of logCPM on design Dm; test coefficient j. moderate=True applies limma's empirical-Bayes
    variance shrinkage (squeezeVar, Smyth 2004)."""
    beta, *_ = np.linalg.lstsq(Dm, X, rcond=None)
    R = X - Dm @ beta; d = len(Dm) - Dm.shape[1]
    s2 = (R ** 2).sum(0) / d
    v = np.linalg.inv(Dm.T @ Dm)[j, j]
    d0, s02 = 0, 0
    if moderate:
        z = np.log(s2); e = z - special.digamma(d / 2) + np.log(d / 2)
        ev = e.var(ddof=1) - special.polygamma(1, d / 2).mean() if np.ndim(d) else e.var(ddof=1) - special.polygamma(1, d / 2)
        d0 = 2 * trigamma_inv(ev) if ev > 0 else np.inf
        s02 = np.exp(e.mean() + special.digamma(d0 / 2) - np.log(d0 / 2)) if np.isfinite(d0) else np.exp(e.mean())
        s2 = (d0 * s02 + d * s2) / (d0 + d) if np.isfinite(d0) else np.full_like(s2, s02)
    t = beta[j] / np.sqrt(s2 * v); dft = d + (d0 if np.isfinite(d0) else 1e6)
    return beta[j], 2 * stats.t.sf(np.abs(t), dft), t, d0


one = np.ones(n)
de_out = {"genes": GK, "group": GGROUP, "groups": GROUPS, "methods": {}}
yperm = np.random.default_rng(5).permutation(y)
runs = {
    "welch": ("Welch t-test, no batch term", lambda: welch(X_raw, y)[:2]),
    "lm": ("Linear model ~ response + centre", lambda: linmod(X_raw, np.column_stack([one, y, bB]))[:2]),
    "limma": ("Moderated t (limma-style eBayes) ~ response + centre", lambda: linmod(X_raw, np.column_stack([one, y, bB]), moderate=True)[:2]),
    "perm": ("Same linear model, response labels permuted", lambda: linmod(X_raw, np.column_stack([one, yperm, bB]))[:2]),
}
for k, (lab, f) in runs.items():
    lfc, p = f(); q = bh(p)
    hits = q < 0.05
    by = {g: [int((hits & (np.array(GGROUP) == g)).sum()), int((np.array(GGROUP) == g).sum())] for g in GROUPS}
    de_out["methods"][k] = {"label": lab, "lfc": lfc, "p": p, "q": q, "n_hits": int(hits.sum()), "by_group": by}
    log("DE", k, int(hits.sum()), by)
_, _, _, d0 = linmod(X_raw, np.column_stack([one, y, bB]), moderate=True)
de_out["limma_d0"] = d0
de_out["n"] = n; de_out["n_resp"] = int(y.sum())
dump("de", de_out, s=3)

# ============================================================================================== 5. feature selection
log("feature selection: lasso / elastic-net paths, stability selection")
MU_D, SD_D = X_prot.mean(0), X_prot.std(0, ddof=0)
Z = (X_prot - MU_D) / SD_D
CS = np.logspace(-2.1, -0.3, 30)


def l1fit(Xs, ys, C, ratio=1.0, warm=None):
    if ratio == 1.0:
        m = LogisticRegression(penalty="l1", solver="liblinear", C=C, max_iter=2000, tol=1e-5)
    else:
        m = LogisticRegression(penalty="elasticnet", l1_ratio=ratio, solver="saga", C=C, max_iter=4000, tol=1e-4)
    return m.fit(Xs, ys)


def path(Xs, ys, ratio=1.0):
    coefs, ints = [], []
    for C in CS:
        m = l1fit(Xs, ys, C, ratio); coefs.append(m.coef_[0]); ints.append(m.intercept_[0])
    return np.array(coefs), np.array(ints)


def cv_path_auc(Xs, ys, ratio=1.0, reps=3, folds=5, seed=0):
    out = np.zeros((reps, len(CS)))
    for r in range(reps):
        oof = np.zeros((len(ys), len(CS)))
        for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed + r).split(Xs, ys):
            mu, sd = Xs[tr].mean(0), Xs[tr].std(0) + 1e-9
            for j, C in enumerate(CS):
                m = l1fit((Xs[tr] - mu) / sd, ys[tr], C, ratio)
                oof[te, j] = m.decision_function((Xs[te] - mu) / sd)
        out[r] = [roc_auc_score(ys, oof[:, j]) if np.ptp(oof[:, j]) > 0 else 0.5 for j in range(len(CS))]
    return out


def one_se(aucs):
    m = aucs.mean(0); se = aucs.std(0, ddof=1) / np.sqrt(aucs.shape[0]) if aucs.shape[0] > 1 else np.zeros_like(m)
    best = int(np.argmax(m)); thr = m[best] - max(se[best], 0.01)
    return best, int(np.argmax(m >= thr))       # smallest C (most regularised) within 1 SE of the best


fs_out = {"Cs": CS, "lambda": 1 / (CS * n), "paths": {}}
for name, ratio in (("lasso", 1.0), ("enet", 0.5)):
    coefs, ints = path(Z, y, ratio)
    aucs = cv_path_auc(X_prot, y, ratio)
    best, ise = one_se(aucs)
    nz = np.where((coefs != 0).any(0))[0]
    entry = [int(np.argmax(coefs[:, j] != 0)) for j in nz]
    order = nz[np.lexsort((-np.abs(coefs[-1, nz]), entry))][:70]
    fs_out["paths"][name] = {
        "genes": [GK[j] for j in order], "group": [GGROUP[j] for j in order],
        "coef": coefs[:, order].T, "nnz": (coefs != 0).sum(1), "auc_mean": aucs.mean(0), "auc_sd": aucs.std(0, ddof=1),
        "best": best, "one_se": ise,
    }
    log(name, "nnz", (coefs != 0).sum(1).tolist(), "auc", np.round(aucs.mean(0), 3).tolist(), "best", best, "1se", ise)
FINAL_C = CS[fs_out["paths"]["lasso"]["one_se"]]
FINAL = l1fit(Z, y, FINAL_C)
log("final lasso C", FINAL_C, "genes", [GK[j] for j in np.where(FINAL.coef_[0] != 0)[0]])

# stability selection (Meinshausen & Buhlmann 2010): lasso on many half-samples, record selection frequency
stab = {}
r_s = np.random.default_rng(11)
for lab, C in (("strong", CS[8]), ("medium", CS[13]), ("weak", CS[18])):
    freq = np.zeros(len(GK)); qn = []
    for b in range(100):
        idx = r_s.choice(n, n // 2, replace=False)
        while y[idx].sum() < 5: idx = r_s.choice(n, n // 2, replace=False)
        Zb = (X_prot[idx] - X_prot[idx].mean(0)) / (X_prot[idx].std(0) + 1e-9)
        sel = l1fit(Zb, y[idx], C).coef_[0] != 0
        freq += sel; qn.append(sel.sum())
    freq /= 100
    top = np.argsort(-freq)[:40]
    stab[lab] = {"C": C, "genes": [GK[j] for j in top], "group": [GGROUP[j] for j in top], "freq": freq[top], "q": float(np.mean(qn))}
    log("stability", lab, "q", np.mean(qn), list(zip([GK[j] for j in top[:10]], np.round(freq[top[:10]], 2))))
fs_out["stability"] = stab
fs_out["p"] = len(GK); fs_out["n"] = n; fs_out["n_resp"] = int(y.sum())
fs_out["final"] = {"C": FINAL_C, "genes": [GK[j] for j in np.where(FINAL.coef_[0] != 0)[0]]}
dump("fs", fs_out)

# ============================================================================================== 6. classifier: leaky vs nested
log("classifier: leaky vs nested CV")
KS = [5, 10, 20, 50, 100, 200]
C_GRID = [0.01, 0.1, 1.0]


def ftest(Xs, ys):
    return np.abs(welch(Xs, ys)[2])


def fit_tuned(Xtr, ytr, seed):
    """Inner CV picks the L2 strength (the 'nested' part of hyperparameter tuning)."""
    best, bestC = -1, C_GRID[0]
    for C in C_GRID:
        sc = []
        for tr, te in StratifiedKFold(3, shuffle=True, random_state=seed).split(Xtr, ytr):
            m = LogisticRegression(C=C, max_iter=2000).fit(Xtr[tr], ytr[tr])
            sc.append(roc_auc_score(ytr[te], m.decision_function(Xtr[te])))
        if np.mean(sc) > best: best, bestC = np.mean(sc), C
    return LogisticRegression(C=bestC, max_iter=2000).fit(Xtr, ytr)


def run_cv(X, ys, k, leaky, seed):
    oof = np.zeros(len(ys))
    if leaky: feats_all = np.argsort(-ftest(X, ys))[:k]       # selection sees ALL samples, incl. future test folds
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, ys):
        feats = feats_all if leaky else np.argsort(-ftest(X[tr], ys[tr]))[:k]
        mu, sd = X[tr][:, feats].mean(0), X[tr][:, feats].std(0) + 1e-9
        m = fit_tuned((X[tr][:, feats] - mu) / sd, ys[tr], seed)
        oof[te] = m.predict_proba((X[te][:, feats] - mu) / sd)[:, 1]
    return oof


def roc_pts(yv, s, m=50):
    f, t, _ = roc_curve(yv, s)
    if len(f) > m:
        i = np.unique(np.r_[0, np.linspace(0, len(f) - 1, m).astype(int), len(f) - 1]); f, t = f[i], t[i]
    return np.column_stack([f, t])


REPS = 8
cls_out = {"ks": KS, "reps": REPS, "res": {}}
for lab_kind in ("real", "perm"):
    for leaky in (True, False):
        for k in KS:
            aucs, roc0 = [], None
            for r in range(REPS):
                ys = y if lab_kind == "real" else np.random.default_rng(100 + r).permutation(y)
                oof = run_cv(X_prot, ys, k, leaky, seed=r)
                aucs.append(roc_auc_score(ys, oof))
                if r == 0: roc0 = roc_pts(ys, oof)
            key = f"{lab_kind}|{'leaky' if leaky else 'nested'}|{k}"
            cls_out["res"][key] = {"auc": aucs, "mean": np.mean(aucs), "sd": np.std(aucs, ddof=1), "roc": roc0}
            log(key, round(np.mean(aucs), 3), "+-", round(np.std(aucs, ddof=1), 3))
cls_out["n"] = n; cls_out["n_resp"] = int(y.sum()); cls_out["p"] = len(GK)
dump("classifier", cls_out, s=3)
