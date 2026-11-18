"""Build data/study/regularization.json for the regularisation lesson.

Contents
  * 120 simulated "noise genes" for all 300 study patients (independent of response; blocks of 10 genes share
    a weak common factor so that neighbouring genes are mildly correlated, as real expression data are).
  * Stability-selection results: for 50 random half-samples of the discovery patients, an L1 logistic regression
    (the 10 cohort measurements + the first 40 noise genes, standardised within each half-sample) is fitted at
    12 values of lambda; which features received a non-zero weight is stored as 0/1 strings.
Objective (same as the page's JS solver): mean log-loss + lambda * |w|_1, intercept unpenalised.
Run:  source ~/storage/sc_spatial/storage_env.sh && .venv/bin/python scripts/build_regularization.py
"""
import json, pathlib, warnings
import numpy as np
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[1]
d = json.load(open(ROOT / "data/study/cohort.json"))
cols = d["columns"]; rows = d["rows"]
ci = {c: i for i, c in enumerate(cols)}
ids = [r[ci["id"]] for r in rows]
rng = np.random.default_rng(20261117)
NG = 120
X = np.zeros((len(rows), NG))
for b in range(NG // 10):
    f = rng.normal(size=len(rows))
    for j in range(10):
        X[:, b * 10 + j] = np.sqrt(0.3) * f + np.sqrt(0.7) * rng.normal(size=len(rows))
genes = [f"G{j + 1:03d}" for j in range(NG)]

disc = [i for i, r in enumerate(rows) if r[ci["cohort"]] == "discovery"]
def feat(r):
    return [r[ci["ifng"]], np.log2(r[ci["tmb"]]), r[ci["caf"]], r[ci["pdl1"]], r[ci["cd8"]], r[ci["prolif"]],
            r[ci["age"]], 0 if r[ci["smoking"]] == "never" else 1, 1 if r[ci["sex"]] == "M" else 0, 1 if r[ci["centre"]] == "Centre B" else 0]
A = np.array([feat(rows[i]) + list(X[i, :40]) for i in disc], float)
y = np.array([rows[i][ci["response"]] for i in disc])
B, M = 50, 90
lams = np.round(10 ** np.linspace(-2.2, -0.7, 12), 5)
sel = {str(k): [] for k in range(len(lams))}
for b in range(B):
    idx = rng.choice(len(disc), M, replace=False)
    Z = A[idx]; Z = (Z - Z.mean(0)) / Z.std(0, ddof=1)
    for k, lam in enumerate(lams):
        m = LogisticRegression(penalty="l1", solver="saga", C=1 / (M * lam), tol=1e-5, max_iter=5000).fit(Z, y[idx])
        sel[str(k)].append("".join("1" if abs(w) > 1e-8 else "0" for w in m.coef_[0]))
out = {"ids": ids, "genes": genes, "X": np.round(X, 2).tolist(),
       "stab": {"lambdas": lams.tolist(), "B": B, "m": M, "n_noise": 40, "sel": sel}}
(ROOT / "data/study/regularization.json").write_text(json.dumps(out, separators=(",", ":")))
print("ok", len(ids), NG, B)
