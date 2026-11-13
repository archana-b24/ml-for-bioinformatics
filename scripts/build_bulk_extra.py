"""Build data/omics/bulk_{compare,explain,survival,deconv,validation}.json (Tab 2, bulk steps 7-11).

The simulation is NOT re-implemented: this script executes scripts/build_bulk.py up to (not including) its
"3. subtypes" section, with file writing disabled, so the discovery cohort D, the validation cohort V, the QC
exclusions and the batch-protected matrix X_prot are bit-for-bit the same objects the earlier pages use. It then
checks that the regenerated bulk_qc / bulk_batch JSON text equals the files on disk (proof of identity) and never
writes those files. build_bulk.py itself is not modified.

New simulated data added here (consistent with mock_biology.py; build_bulk.py already simulates PFS for the
anti-PD-1 cohorts with hazard  lambda = ln2/5 * exp(-1.5*response + 0.6*z(PROLIF))):
  * a randomised docetaxel CONTROL arm (n=160, Centre A) drawn from the same phenotype mix and programs, with the
    same hazard formula and response = 0 (no anti-PD-1 benefit). It is what makes "prognostic vs predictive"
    testable at all.
  * 12 bridging RNA aliquots: 12 Centre C validation samples re-sequenced at Centre A.

Run:
  source ~/storage/sc_spatial/storage_env.sh && cd /home/abalasubramanian/storage/ml_dash && .venv/bin/python scripts/build_bulk_extra.py
"""
import json, sys, time, pathlib, warnings
import numpy as np
from scipy import stats, optimize, special
from scipy.optimize import nnls
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss, log_loss

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[1]
BB = ROOT / "scripts" / "build_bulk.py"
OUT = ROOT / "data" / "omics"
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


# ============================================================================== 0. replay build_bulk.py (read-only)
src = BB.read_text()
cut = src.index("# ==============================================================================================" " 3. subtypes")
part = src[:cut]
assert part.count("    p.write_text(txt)\n") == 1
part = part.replace("    p.write_text(txt)\n", "    _CAPTURE[name] = txt\n")
ns = {"__file__": str(BB), "__name__": "build_bulk_replay", "_CAPTURE": {}}
log("replaying build_bulk.py up to the subtypes step (no files written)")
exec(compile(part, str(BB), "exec"), ns)
for name in ("qc", "batch"):
    on_disk = (OUT / f"bulk_{name}.json").read_text()
    assert ns["_CAPTURE"][name] == on_disk, f"replayed bulk_{name}.json differs from disk: simulation drifted"
log("replay verified: bulk_qc.json and bulk_batch.json regenerate byte-identically")

clean, GENES, GI, CT, P_REF = ns["clean"], ns["GENES"], ns["GI"], ns["CT"], ns["P_REF"]
D, V, keep, gkeep, GK = ns["D"], ns["V"], ns["keep"], ns["gkeep"], ns["GK"]
X = ns["X_prot"]; y = ns["y"]; n = len(y); pheno = ns["pheno"]; centre = ns["centre"]; ids = ns["ids"]
IFNG_IDX, PROLIF_IDX, CAF_IDX = ns["IFNG_IDX"], ns["PROLIF_IDX"], ns["CAF_IDX"]
T_pfs, E_pfs, TRUE_FRAC, PROG = ns["T_pfs"], ns["E_pfs"], ns["TRUE_FRAC"], ns["PROG"]
logcpm, group_of = ns["logcpm"], ns["group_of"]
FS = json.loads((OUT / "bulk_fs.json").read_text())
CS = np.logspace(-2.1, -0.3, 30)                       # identical to build_bulk.py step 5
FINAL_C = CS[FS["paths"]["lasso"]["one_se"]]
MU_D, SD_D = X.mean(0), X.std(0, ddof=0)
Z = (X - MU_D) / SD_D
log("n", n, "responders", int(y.sum()), "genes", len(GK), "final lasso C", FINAL_C)


STEPS = set(sys.argv[1:]) or {"compare", "explain", "survival", "deconv", "validation"}
RUN = lambda k: k in STEPS
ALPHA = float(__import__("os").environ.get("COX_ALPHA", "0.5"))   # Cox elastic-net mixing (1 = lasso)
LAM_RULE = __import__("os").environ.get("COX_RULE", "min")
ifng_io, prolif_io = X[:, IFNG_IDX].mean(1), X[:, PROLIF_IDX].mean(1)


def dump(name, obj, s=4):
    p = OUT / f"bulk_{name}.json"
    assert name not in ("qc", "batch", "subtypes", "de", "fs", "classifier")
    txt = json.dumps(clean(obj, s), separators=(",", ":"))
    p.write_text(txt)
    log(f"wrote {p.name}: {len(txt) / 1024:.0f} KB")


def roc_pts(yv, s, m=60):
    f, t, _ = roc_curve(yv, s)
    if len(f) > m:
        i = np.unique(np.r_[0, np.linspace(0, len(f) - 1, m).astype(int), len(f) - 1]); f, t = f[i], t[i]
    return np.column_stack([f, t])


def cal_slope(yv, p):
    """Logistic recalibration y ~ a + b*logit(p): returns (intercept a, slope b). Ideal: (0, 1)."""
    lp = special.logit(np.clip(p, 1e-6, 1 - 1e-6))
    m = LogisticRegression(C=1e6, max_iter=1000).fit(lp[:, None], yv)
    return float(m.intercept_[0]), float(m.coef_[0, 0])


def cal_intercept(yv, p):
    """Calibration-in-the-large: intercept a in logit P(y) = a + logit(p) (offset). Ideal 0."""
    lp = special.logit(np.clip(p, 1e-6, 1 - 1e-6))
    f = lambda a: -np.sum(yv * (a + lp) - np.log1p(np.exp(a + lp)))
    return float(optimize.minimize_scalar(f, bounds=(-5, 5), method="bounded").x)


def metrics(yv, p):
    a, b = cal_slope(yv, p)
    return {"auc": roc_auc_score(yv, p), "brier": brier_score_loss(yv, p), "logloss": log_loss(yv, np.clip(p, 1e-6, 1 - 1e-6)),
            "citl": cal_intercept(yv, p), "slope": b, "mean_p": p.mean(), "obs": yv.mean()}


# ============================================================================== 7. model comparison (nested CV)
if RUN("compare"):
    log("step 7: model comparison under nested CV")
    GRIDS = {
        "lasso": [dict(C=c) for c in (0.02, 0.05, 0.1, 0.3, 1.0)],
        "rf": [dict(max_features=mf, min_samples_leaf=ml) for mf in ("sqrt", 0.1) for ml in (1, 5)],
        "gb": [dict(learning_rate=lr, max_depth=md) for lr in (0.03, 0.1) for md in (1, 3)],
    }
    LABELS = {"ifng": "IFN-γ score (1 feature, prespecified)", "lasso": "Lasso logistic", "rf": "Random forest", "gb": "Gradient boosting"}


    def make(kind, hp, seed):
        if kind == "lasso": return LogisticRegression(penalty="l1", solver="liblinear", max_iter=2000, intercept_scaling=100, **hp)  # large scaling = intercept effectively unpenalised
        if kind == "rf": return RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=16, **hp)
        if kind == "gb": return GradientBoostingClassifier(n_estimators=100, subsample=0.8, max_features=0.2, random_state=seed, **hp)


    def fit_predict(kind, hp, Xtr, ytr, Xte, seed):
        if kind == "ifng":
            s_tr, s_te = Xtr[:, IFNG_IDX].mean(1, keepdims=True), Xte[:, IFNG_IDX].mean(1, keepdims=True)
            return LogisticRegression(C=1e6, max_iter=1000).fit(s_tr, ytr).predict_proba(s_te)[:, 1]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9           # standardise with TRAINING statistics only
        return make(kind, hp, seed).fit((Xtr - mu) / sd, ytr).predict_proba((Xte - mu) / sd)[:, 1]


    def inner_pick(kind, Xtr, ytr, seed):
        best, bhp = -1, None
        for hp in GRIDS[kind]:
            oof = np.zeros(len(ytr))
            for a, b in StratifiedKFold(3, shuffle=True, random_state=100 + seed).split(Xtr, ytr):
                oof[b] = fit_predict(kind, hp, Xtr[a], ytr[a], Xtr[b], seed)
            sc = roc_auc_score(ytr, oof)
            if sc > best: best, bhp = sc, hp
        return bhp


    REPS = 5
    cmp_res = {k: {"auc": [], "brier": [], "citl": [], "slope": [], "picked": []} for k in LABELS}
    oof0 = {}
    plain = {k: np.zeros((REPS, len(GRIDS[k]))) for k in GRIDS}   # non-nested: every config scored on the outer folds
    for r in range(REPS):
        folds = list(StratifiedKFold(5, shuffle=True, random_state=r).split(X, y))
        for kind in LABELS:
            oof = np.zeros(n)
            for tr, te in folds:
                hp = inner_pick(kind, X[tr], y[tr], r) if kind != "ifng" else None
                if hp is not None: cmp_res[kind]["picked"].append(GRIDS[kind].index(hp))
                oof[te] = fit_predict(kind, hp, X[tr], y[tr], X[te], r)
            m = metrics(y, oof)
            for k in ("auc", "brier", "citl", "slope"): cmp_res[kind][k].append(m[k])
            if r == 0: oof0[kind] = oof
            if kind in GRIDS:
                for j, hp in enumerate(GRIDS[kind]):
                    o2 = np.zeros(n)
                    for tr, te in folds: o2[te] = fit_predict(kind, hp, X[tr], y[tr], X[te], r)
                    plain[kind][r, j] = roc_auc_score(y, o2)
            log(f"rep {r} {kind}: AUC {m['auc']:.3f} Brier {m['brier']:.3f} slope {m['slope']:.2f}")

    cmp_out = {"n": n, "n_resp": int(y.sum()), "p": len(GK), "reps": REPS, "y": y, "pheno": pheno, "models": {}}
    for kind, lab in LABELS.items():
        rr = cmp_res[kind]
        entry = {"label": lab, "auc": rr["auc"], "brier": rr["brier"], "citl": rr["citl"], "slope": rr["slope"],
                 "p": oof0[kind], "rep0": metrics(y, oof0[kind])}
        if kind in GRIDS:
            entry["grid"] = [", ".join(f"{k}={v}" for k, v in hp.items()) for hp in GRIDS[kind]]
            entry["picked_counts"] = np.bincount(rr["picked"], minlength=len(GRIDS[kind]))
            entry["plain_mean"] = plain[kind].mean(0)          # per config, mean over reps (non-nested)
        cmp_out["models"][kind] = entry
    # paired differences per repeat vs the lasso (same folds within a repeat, so the difference is paired)
    cmp_out["diff_vs_lasso"] = {k: (np.array(cmp_res[k]["auc"]) - np.array(cmp_res["lasso"]["auc"])) for k in LABELS}
    dump("compare", cmp_out)

# ============================================================================== 8. explaining predictions
if RUN("explain"):
    log("step 8: explanations (linear contributions, exact interventional Shapley for boosting)")
    PANEL = ["CXCL9", "CXCL10", "GZMB", "CD8A", "CD274", "FAP", "POSTN", "MKI67"]
    pidx = [GK.index(g) for g in PANEL]
    Xp = Z[:, pidx]                                       # standardised (z) expression of the panel
    lin = LogisticRegression(C=0.5, max_iter=2000).fit(Xp, y)
    gbm = GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=2, subsample=0.8, random_state=0).fit(Xp, y)
    rbg = np.random.default_rng(42)
    BG = Xp[rbg.choice(n, 60, replace=False)]           # background (reference) patients
    pp = len(PANEL)
    masks = np.array([[(m >> j) & 1 for j in range(pp)] for m in range(2 ** pp)], bool)


    def shapley_exact(f, Xq, bg):
        """Exact interventional Shapley values: v(S) = mean_b f(x_S, bg_b,-S); phi_j = sum_S w(|S|)[v(S+j)-v(S)]."""
        nb, M = len(bg), len(masks)
        V = np.zeros((len(Xq), M))
        for i, x in enumerate(Xq):
            rows = np.where(masks[:, None, :], x[None, None, :], bg[None, :, :]).reshape(-1, pp)
            V[i] = f(rows).reshape(M, nb).mean(1)
        fact = special.factorial
        phi = np.zeros((len(Xq), pp))
        msize = masks.sum(1)
        for j in range(pp):
            without = np.where(~masks[:, j])[0]
            with_j = without + (1 << j)
            w = fact(msize[without]) * fact(pp - msize[without] - 1) / fact(pp)
            phi[:, j] = ((V[:, with_j] - V[:, without]) * w).sum(1)
        return phi, V[:, 0]


    phi_gb, base_gb = shapley_exact(gbm.decision_function, Xp, BG)
    f_gb = gbm.decision_function(Xp)
    assert np.allclose(phi_gb.sum(1) + base_gb, f_gb, atol=1e-8), "Shapley efficiency failed"
    bgm = BG.mean(0)
    phi_lin = lin.coef_[0] * (Xp - bgm)                 # exact Shapley for a linear model with independent features
    base_lin = float(lin.decision_function(bgm[None])[0])
    f_lin = lin.decision_function(Xp)
    # out-of-fold AUC of both panel models, so the page can say how good the explained models are
    oof_l, oof_g = np.zeros(n), np.zeros(n)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Xp, y):
        oof_l[te] = LogisticRegression(C=0.5, max_iter=2000).fit(Xp[tr], y[tr]).decision_function(Xp[te])
        oof_g[te] = GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=2, subsample=0.8, random_state=0).fit(Xp[tr], y[tr]).decision_function(Xp[te])
    # drop-one-gene refit: what happens to CXCL9 credit when its correlated partner CXCL10 is removed
    lin_drop = LogisticRegression(C=0.5, max_iter=2000).fit(np.delete(Xp, 1, 1), y)
    exp_out = {
        "genes": PANEL, "group": [group_of(g) for g in PANEL], "y": y, "pheno": pheno, "ids": ids,
        "x": Xp, "corr": np.corrcoef(Xp.T),
        "lin": {"coef": lin.coef_[0], "intercept": lin.intercept_[0], "base": base_lin, "f": f_lin, "phi": phi_lin,
                "auc_oof": roc_auc_score(y, oof_l), "coef_drop_cxcl10": np.insert(lin_drop.coef_[0], 1, np.nan)},
        "gb": {"base": base_gb.mean(), "f": f_gb, "phi": phi_gb, "auc_oof": roc_auc_score(y, oof_g)},
        "n_bg": len(BG),
    }
    log("explain OOF AUC lin", round(exp_out["lin"]["auc_oof"], 3), "gb", round(exp_out["gb"]["auc_oof"], 3))
    dump("explain", exp_out, s=3)

# ============================================================================== 9. survival
if RUN("survival"):
    log("step 9: survival (Cox-lasso), control arm for prognostic vs predictive")
    ns["rng"] = np.random.default_rng(mb_seed := 2026 + 909)
    mbm = ns["mb"]
    N_CTRL = 160
    mix = {k: int(round(v * N_CTRL)) for k, v in mbm.PHENOTYPE_MIX.items()}
    mix["desert"] += N_CTRL - sum(mix.values())
    rowsC = [ph for ph, k in mix.items() for _ in range(k)]
    rowsC = [rowsC[i] for i in ns["rng"].permutation(len(rowsC))]
    propC = np.zeros((N_CTRL, len(GENES))); progC = []; tC = np.zeros(N_CTRL); eC = np.zeros(N_CTRL, int)
    fracC = np.zeros((N_CTRL, len(CT)))
    for i, ph in enumerate(rowsC):
        fracC[i] = ns["rng"].dirichlet(mbm.PHENOTYPES[ph]["alpha"])
        pr = ns["draw_programs"](ph); progC.append(pr)
        propC[i] = ns["patient_mean"](fracC[i], pr)
        lam = np.log(2) / 5.0 * np.exp(-1.5 * 0 + 0.6 * (pr["PROLIF"] - 0.5) / 0.2236)   # same formula, response = 0
        T = ns["rng"].exponential(1 / lam); C = min(ns["rng"].uniform(12, 48), ns["rng"].exponential(120))
        tC[i] = min(T, C); eC[i] = int(T <= C)
    depthC = ns["rng"].lognormal(np.log(15e6), 0.3, N_CTRL)
    countsC = ns["nb_counts"](propC, depthC)              # Centre A chemistry: no technical shift
    XC = logcpm(countsC)[:, gkeep]


    def cox_nll_grad(beta, Xs, t, e):
        o = np.argsort(-t, kind="stable"); Xo, eo = Xs[o], e[o]
        eta = Xo @ beta; mx = eta.max(); w = np.exp(eta - mx); R = np.cumsum(w)
        ll = np.sum(eo * (eta - mx - np.log(R)))
        g_eta = -eo + w * np.cumsum((eo / R)[::-1])[::-1]
        return -ll / len(t), Xo.T @ g_eta / len(t)


    def cox_lasso(Xs, t, e, lam, beta0=None, iters=400, alpha=None):
        """FISTA proximal gradient on the Breslow negative log partial likelihood + lam*||beta||_1."""
        p = Xs.shape[1]; b = np.zeros(p) if beta0 is None else beta0.copy(); zv = b.copy(); tk = 1.0; L = 1.0
        f_old = np.inf
        for it in range(iters):
            fz, gz = cox_nll_grad(zv, Xs, t, e)
            while True:
                a_ = ALPHA if alpha is None else alpha
                bn = zv - gz / L; bn = np.sign(bn) * np.maximum(np.abs(bn) - a_ * lam / L, 0) / (1 + (1 - a_) * lam / L)
                fn, _ = cox_nll_grad(bn, Xs, t, e)
                if fn <= fz + gz @ (bn - zv) + 0.5 * L * np.sum((bn - zv) ** 2) + 1e-12: break
                L *= 2
            tn = (1 + np.sqrt(1 + 4 * tk * tk)) / 2
            zv = bn + (tk - 1) / tn * (bn - b); b, tk = bn, tn
            obj = fn + lam * (a_ * np.abs(b).sum() + 0.5 * (1 - a_) * (b ** 2).sum())
            if abs(f_old - obj) < 1e-7: break
            f_old = obj
        return b


    def cox_path(Xs, t, e, lams):
        out, b = [], None
        for lam in lams:
            b = cox_lasso(Xs, t, e, lam, b); out.append(b.copy())
        return np.array(out)


    def partial_ll(beta, Xs, t, e):
        return -cox_nll_grad(beta, Xs, t, e)[0] * len(t)


    def cv_lambda(Xs, t, e, lams, seed, k=5):
        """Cross-validated partial likelihood (Verweij & van Houwelingen 1993), as in glmnet's type.measure='deviance'."""
        cvl = np.zeros((k, len(lams)))
        for f, (tr, te) in enumerate(KFold(k, shuffle=True, random_state=seed).split(Xs)):
            mu, sd = Xs[tr].mean(0), Xs[tr].std(0) + 1e-9
            B = cox_path((Xs[tr] - mu) / sd, t[tr], e[tr], lams)
            Zall = (Xs - mu) / sd
            for j, b in enumerate(B):
                cvl[f, j] = partial_ll(b, Zall, t, e) - partial_ll(b, Zall[tr], t[tr], e[tr])
        dev = -2 * cvl.sum(0)
        se = (-2 * cvl * k).std(0, ddof=1) / np.sqrt(k)
        return dev, se


    def cindex(t, e, risk):
        num = den = 0.0
        for i in np.where(e == 1)[0]:
            m = t > t[i]
            den += m.sum(); num += (risk[i] > risk[m]).sum() + 0.5 * (risk[i] == risk[m]).sum()
        return num / den


    def cox_newton(Xd, t, e, iters=50):
        b = np.zeros(Xd.shape[1])
        for _ in range(iters):
            o = np.argsort(-t, kind="stable"); Xo, eo = Xd[o], e[o]
            w = np.exp(Xo @ b); R = np.cumsum(w); S1 = np.cumsum(w[:, None] * Xo, 0)
            S2 = np.cumsum(w[:, None, None] * Xo[:, :, None] * Xo[:, None, :], 0)
            g = (eo[:, None] * (Xo - S1 / R[:, None])).sum(0)
            H = -(eo[:, None, None] * (S2 / R[:, None, None] - (S1 / R[:, None])[:, :, None] * (S1 / R[:, None])[:, None, :])).sum(0)
            step = np.linalg.solve(H, g); b = b - step
            if np.abs(step).max() < 1e-9: break
        se = np.sqrt(np.diag(np.linalg.inv(-H)))
        return b, se


    Zs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    _, g0 = cox_nll_grad(np.zeros(Zs.shape[1]), Zs, T_pfs, E_pfs)
    lam_max = np.abs(g0).max() / ALPHA
    LAMS = lam_max * np.logspace(0, -1.3, 24)
    dev, dse = cv_lambda(X, T_pfs, E_pfs, LAMS, seed=0)
    jmin = int(np.argmin(dev)); j1se = int(np.argmax(dev <= dev[jmin] + dse[jmin]))
    Bfull = cox_path(Zs, T_pfs, E_pfs, LAMS)
    jsel = jmin if LAM_RULE == "min" else j1se
    beta_sel = Bfull[jsel]
    nz = np.where(beta_sel != 0)[0]
    log("cox-lasso lambda idx min", jmin, "1se", j1se, "genes", [(GK[j], round(beta_sel[j], 3)) for j in nz])
    risk_in = Zs @ beta_sel
    # honest (nested) out-of-fold risk: lambda chosen by inner CV inside each outer fold
    risk_cv = np.zeros(n); grp_cv = np.zeros(n, int); outer_genes = []
    for f, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=1).split(X)):
        d_in, s_in = cv_lambda(X[tr], T_pfs[tr], E_pfs[tr], LAMS, seed=10 + f)
        jm = int(np.argmin(d_in)); j1 = int(np.argmax(d_in <= d_in[jm] + s_in[jm]))
        j1 = jm if LAM_RULE == "min" else j1
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        b = cox_path((X[tr] - mu) / sd, T_pfs[tr], E_pfs[tr], LAMS[: j1 + 1])[-1]
        risk_cv[te] = ((X[te] - mu) / sd) @ b
        grp_cv[te] = risk_cv[te] > np.median(((X[tr] - mu) / sd) @ b)     # split at the TRAINING median
        outer_genes.append([GK[j] for j in np.where(b != 0)[0]])
    log("outer-fold genes", outer_genes)
    risk_ctrl = ((XC - X.mean(0)) / (X.std(0) + 1e-9)) @ beta_sel     # locked IO risk score applied to control arm
    ifng_io, prolif_io = X[:, IFNG_IDX].mean(1), X[:, PROLIF_IDX].mean(1)
    ifng_c, prolif_c = XC[:, IFNG_IDX].mean(1), XC[:, PROLIF_IDX].mean(1)


    def zs(a, ref):
        return (a - ref.mean()) / ref.std()


    arm = np.r_[np.ones(n), np.zeros(N_CTRL)]; tt = np.r_[T_pfs, tC]; ee = np.r_[E_pfs, eC]
    inter = {}
    for nm, sio, sc in (("ifng", ifng_io, ifng_c), ("prolif", prolif_io, prolif_c)):
        ref = np.r_[sio, sc]; s = zs(ref, ref)
        Xd = np.column_stack([arm, s, arm * s])
        b, se = cox_newton(Xd, tt, ee)
        pv = 2 * stats.norm.sf(np.abs(b / se))
        hr_io, se_io = cox_newton(zs(sio, ref)[:, None], T_pfs, E_pfs)
        hr_c, se_c = cox_newton(zs(sc, ref)[:, None], tC, eC.astype(float))
        inter[nm] = {"coef": b, "se": se, "p": pv,
                     "hr_io": np.exp(hr_io[0]), "p_io": 2 * stats.norm.sf(abs(hr_io[0] / se_io[0])),
                     "hr_ctrl": np.exp(hr_c[0]), "p_ctrl": 2 * stats.norm.sf(abs(hr_c[0] / se_c[0])),
                     "auc_resp": roc_auc_score(y, sio)}
        log(nm, "interaction p", pv[2], "HR/SD io", np.exp(hr_io[0]), "ctrl", np.exp(hr_c[0]), "AUC resp", inter[nm]["auc_resp"])
    surv_out = {
        "io": {"t": T_pfs, "e": E_pfs, "resp": y, "pheno": pheno, "risk_in": risk_in, "risk_cv": risk_cv, "grp_cv": grp_cv,
               "ifng": ifng_io, "prolif": prolif_io},
        "ctrl": {"t": tC, "e": eC, "pheno": np.array(rowsC), "risk": risk_ctrl, "ifng": ifng_c, "prolif": prolif_c},
        "path": {"lam": LAMS, "dev": dev, "se": dse, "nnz": (Bfull != 0).sum(1), "jmin": jmin, "j1se": j1se, "jsel": jsel, "rule": LAM_RULE, "alpha": ALPHA},
        "model": {"genes": [GK[j] for j in nz], "group": [group_of(GK[j]) for j in nz], "coef": beta_sel[nz]},
        "outer_genes": outer_genes,
        "cindex": {"in": cindex(T_pfs, E_pfs, risk_in), "cv": cindex(T_pfs, E_pfs, risk_cv), "ctrl": cindex(tC, eC, risk_ctrl)},
        "inter": inter, "n_io": n, "n_ctrl": N_CTRL,
    }
    log("c-index", surv_out["cindex"])
    dump("survival", surv_out)

# ============================================================================== 10. deconvolution
if RUN("deconv"):
    log("step 10: deconvolution (NNLS)")
    cnt = D["counts"][keep].astype(float)
    cpm = cnt / cnt.sum(1, keepdims=True) * 1e6
    REF = P_REF / P_REF.sum(1, keepdims=True) * 1e6       # pure-cell-type CPM, programs OFF
    mk = sorted({GI[g] for gs in mbm.MARKERS.values() for g in gs})
    allg = np.arange(len(GENES))


    def deconv(B, A):
        est = np.array([nnls(A, b)[0] for b in B])
        return est / est.sum(1, keepdims=True)


    fib = CT.index("Fibroblast")
    SCEN = {
        "markers": ("Marker genes, linear CPM", lambda: deconv(cpm[:, mk], REF[:, mk].T)),
        "all": ("All 2,000 genes, linear CPM", lambda: deconv(cpm, REF.T)),
        "log": ("Marker genes, log2 CPM (wrong scale)", lambda: deconv(np.log2(cpm[:, mk] + 1), np.log2(REF[:, mk].T + 1))),
        "missing": ("Markers, reference lacks fibroblasts", lambda: np.insert(deconv(cpm[:, mk], np.delete(REF, fib, 0)[:, mk].T), fib, 0, axis=1)),
    }
    dec_out = {"cell_types": CT, "true": TRUE_FRAC, "pheno": pheno, "centre": centre, "resp": y, "ids": ids,
               "n_markers": len(mk), "scen": {}}
    for k, (lab, f) in SCEN.items():
        E = f()
        r = [np.corrcoef(TRUE_FRAC[:, t], E[:, t])[0, 1] if E[:, t].std() > 0 else None for t in range(len(CT))]
        rmse = np.sqrt(((E - TRUE_FRAC) ** 2).mean(0))
        bias_c = {c: (E[centre == c] - TRUE_FRAC[centre == c]).mean(0) for c in ("Centre A", "Centre B")}
        dec_out["scen"][k] = {"label": lab, "est": E, "r": r, "rmse": rmse, "bias": bias_c}
        log(k, "r", np.round([v if v is not None else np.nan for v in r], 2), "rmse", np.round(rmse, 3))
    cd8 = CT.index("CD8 T")
    dec_out["auc_resp"] = {"true_cd8": roc_auc_score(y, TRUE_FRAC[:, cd8]), "est_cd8": roc_auc_score(y, dec_out["scen"]["markers"]["est"][:, cd8]),
                           "ifng": roc_auc_score(y, ifng_io)}
    dec_out["cd8_by_pheno"] = {ph: float(dec_out["scen"]["markers"]["est"][pheno == ph, cd8].mean()) for ph in ("inflamed", "excluded", "desert")}
    dec_out["resp_by_pheno"] = {ph: float(y[pheno == ph].mean()) for ph in ("inflamed", "excluded", "desert")}
    log("deconv AUCs", dec_out["auc_resp"], dec_out["cd8_by_pheno"])
    dump("deconv", dec_out, s=3)

# ============================================================================== 11. locking & external validation
if RUN("validation"):
    log("step 11: lock the lasso, validate in Centre C")
    lockm = LogisticRegression(penalty="l1", solver="liblinear", C=FINAL_C, max_iter=2000, tol=1e-5).fit(Z, y)
    lg = np.where(lockm.coef_[0] != 0)[0]
    # relaxed lasso: the lasso picks the genes, an unpenalised logistic refit sets the weights (the lasso's own
    # coefficients are shrunk toward 0, and liblinear also shrinks the intercept, which ruins calibration)
    refit = LogisticRegression(C=1e6, max_iter=2000).fit(Z[:, lg], y)
    LOCK = {"genes": [GK[j] for j in lg], "coef": refit.coef_[0], "intercept": refit.intercept_[0],
            "lasso_coef": lockm.coef_[0][lg], "mu": MU_D[lg], "sd": SD_D[lg], "C": FINAL_C}
    assert LOCK["genes"] == FS["final"]["genes"], (LOCK["genes"], FS["final"]["genes"])
    log("locked model", LOCK["genes"], LOCK["coef"], LOCK["intercept"])


    def lock_predict(Xg, mu=None, sd=None):
        mu = LOCK["mu"] if mu is None else mu; sd = LOCK["sd"] if sd is None else sd
        return special.expit(LOCK["intercept"] + ((Xg - mu) / sd) @ LOCK["coef"])


    p_app = lock_predict(X[:, lg])
    # internal validation: same pipeline (standardise, lasso at the locked C) re-fit in each CV training fold
    p_int = np.zeros(n)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        mu, sd = X[tr].mean(0), X[tr].std(0)
        Ztr, Zte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        sel = np.where(LogisticRegression(penalty="l1", solver="liblinear", C=FINAL_C, max_iter=2000, tol=1e-5).fit(Ztr, y[tr]).coef_[0] != 0)[0]
        if len(sel) == 0: p_int[te] = y[tr].mean(); continue
        m = LogisticRegression(C=1e6, max_iter=2000).fit(Ztr[:, sel], y[tr])
        p_int[te] = m.predict_proba(Zte[:, sel])[:, 1]
    fpr, tpr, thr = roc_curve(y, p_int)
    THR = float(thr[np.argmax(tpr - fpr)])               # Youden cut-off chosen on INTERNAL predictions, then frozen

    gl = [GI[g] for g in LOCK["genes"]]
    LV = logcpm(V["counts"])
    XV = LV[:, gl]; yV = V["resp"]
    # bridging: 12 Centre C RNA aliquots re-sequenced at Centre A (same biology, Centre A chemistry)
    rb = np.random.default_rng(2026 + 1111)
    ns["rng"] = rb
    bri = np.sort(rb.choice(len(yV), 12, replace=False))
    cntA = ns["nb_counts"](V["prop"][bri], rb.lognormal(np.log(15e6), 0.3, 12))
    offset = LV[bri][:, gl].mean(0) - logcpm(cntA)[:, gl].mean(0)
    p_frozen = lock_predict(XV)
    p_bridge = lock_predict(XV - offset)
    p_rez = lock_predict(XV, XV.mean(0), XV.std(0))
    # model updating: logistic recalibration (intercept + slope) estimated on Centre C outcomes, out-of-fold
    lpV = special.logit(p_frozen); p_recal = np.zeros(len(yV))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(XV, yV):
        m = LogisticRegression(C=1e6).fit(lpV[tr, None], yV[tr]); p_recal[te] = m.predict_proba(lpV[te, None])[:, 1]


    def boot_auc(yv, p, B=1000, seed=0):
        r = np.random.default_rng(seed); a = []
        while len(a) < B:
            i = r.integers(0, len(yv), len(yv))
            if 0 < yv[i].sum() < len(i): a.append(roc_auc_score(yv[i], p[i]))
        return np.percentile(a, [2.5, 97.5])


    STRAT = {
        "apparent": ("Discovery, in-sample (apparent)", y, p_app),
        "internal": ("Discovery, 5-fold CV (internal)", y, p_int),
        "frozen": ("Centre C, frozen model as-is", yV, p_frozen),
        "bridge": ("Centre C, bridging-sample offset", yV, p_bridge),
        "rez": ("Centre C, re-standardised within cohort", yV, p_rez),
        "recal": ("Centre C, recalibrated on Centre C outcomes", yV, p_recal),
    }
    val_out = {"lock": LOCK, "thr": THR, "strategies": {}, "pheno_v": V["pheno"], "pheno_d": pheno,
               "n_v": len(yV), "n_d": n, "bridge_idx": bri, "offset": offset}
    for k, (lab, yy, pp_) in STRAT.items():
        m = metrics(yy, pp_); m["ci"] = boot_auc(yy, pp_)
        m["sens"] = float((pp_[yy == 1] >= THR).mean()); m["spec"] = float((pp_[yy == 0] < THR).mean())
        val_out["strategies"][k] = {"label": lab, "y": yy, "p": pp_, **m}
        log(f"{k:9s} AUC {m['auc']:.3f} [{m['ci'][0]:.2f},{m['ci'][1]:.2f}] CITL {m['citl']:+.2f} slope {m['slope']:.2f} "
            f"mean p {m['mean_p']:.2f} obs {m['obs']:.2f} sens {m['sens']:.2f} spec {m['spec']:.2f}")
    # gene-wise technical shift: Centre A discovery vs Centre C (raw logCPM means), and the truth
    LA = ns["L_all"][keep][centre == "Centre A"]
    shift_true = np.log2(ns["SHIFT"]["Centre C"])
    hv = np.argsort(-LA.mean(0))[:600]
    val_out["genes_shift"] = {"meanA": LA.mean(0)[hv], "meanC": LV.mean(0)[hv], "true": shift_true[hv],
                              "lock_meanA": LA.mean(0)[gl], "lock_meanC": LV.mean(0)[gl], "lock_true": shift_true[gl],
                              "lock_offset_est": offset}
    val_out["resp_rate"] = {"d": y.mean(), "v": yV.mean()}
    val_out["xv"] = XV; val_out["yv"] = yV               # raw Centre C log2 CPM of the locked genes (for the stress test)
    val_out["mix_v"] = {ph: int((V["pheno"] == ph).sum()) for ph in ("inflamed", "excluded", "desert")}
    val_out["mix_d"] = {ph: int((pheno == ph).sum()) for ph in ("inflamed", "excluded", "desert")}
    dump("validation", val_out)
    log("done")
