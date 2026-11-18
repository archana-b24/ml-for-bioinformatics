"""Build data/omics/bulk_survival_lab.json for the survival lesson (js/omics/bulk-survival.js).

Replays scripts/build_bulk_extra.py's set-up (which itself replays build_bulk.py read-only) with no steps
enabled, so the discovery matrix X, PFS times and the Centre C validation cohort are the same objects the other
bulk pages use. It regenerates the docetaxel control arm with the same seed and code as build_bulk_extra.py
(step 9) and checks it against data/omics/bulk_survival.json. No existing JSON file is written.

For every penalty value on the elastic-net Cox path (same lambda grid and alpha as bulk_survival.json) it stores:
  * the genes kept and their coefficients (per SD of log-CPM, discovery scaling)
  * C-index in-sample, out-of-fold (5-fold CV at that fixed lambda), in Centre C (genes z-scored within
    Centre C, which removes the centre's gene-wise technical shift) and in the docetaxel control arm
  * the Centre C risk score, so the page can draw Kaplan-Meier curves for the locked score.

Run:
  source ~/storage/sc_spatial/storage_env.sh && cd /home/abalasubramanian/storage/ml_dash && \
    .venv/bin/python scripts/build_survival_lesson.py
"""
import json, sys, pathlib
import numpy as np
from sklearn.model_selection import KFold

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "omics"
SRC = (ROOT / "scripts" / "build_bulk_extra.py").read_text()

# ---- replay build_bulk_extra.py with no steps (set-up only) and grab its survival helpers
sys_argv = sys.argv
sys.argv = ["replay", "__none__"]
ns = {"__file__": str(ROOT / "scripts" / "build_bulk_extra.py"), "__name__": "bbx_replay"}
exec(compile(SRC, "build_bulk_extra.py", "exec"), ns)
sys.argv = sys_argv
start = SRC.index('if RUN("survival"):')
end = SRC.index("    Zs = (X - X.mean(0))", start)
body = SRC[start:end].split("\n", 1)[1]
body = "\n".join(l[4:] if l.startswith("    ") else l for l in body.split("\n"))
exec(compile(body, "survival_setup", "exec"), ns)   # control arm (same seed) + cox helpers

X, T, E, n = ns["X"], ns["T_pfs"], ns["E_pfs"], ns["n"]
XC, tC, eC = ns["XC"], ns["tC"], ns["eC"]
GK, group_of, ALPHA = ns["GK"], ns["group_of"], ns["ALPHA"]
cox_path, cindex = ns["cox_path"], ns["cindex"]
V, gkeep, logcpm = ns["V"], ns["gkeep"], ns["logcpm"]

ref = json.loads((OUT / "bulk_survival.json").read_text())
assert np.allclose(np.array(ref["ctrl"]["t"]), tC, rtol=2e-3, atol=2e-3), "control arm drifted"
assert np.allclose(np.array(ref["io"]["t"]), T, rtol=2e-3, atol=2e-3), "discovery PFS drifted"
print("replay verified against bulk_survival.json")

mu, sd = X.mean(0), X.std(0) + 1e-9
Zs = (X - mu) / sd
_, g0 = ns["cox_nll_grad"](np.zeros(Zs.shape[1]), Zs, T, E)
lam_max = np.abs(g0).max() / ALPHA
LAMS = lam_max * np.logspace(0, -1.3, 24)
assert np.allclose(LAMS, ref["path"]["lam"], rtol=1e-3)
B = cox_path(Zs, T, E, LAMS)
print("path nnz", (B != 0).sum(1))

# out-of-fold risk at each fixed lambda
oof = np.zeros((len(LAMS), n))
for tr, te in KFold(5, shuffle=True, random_state=1).split(X):
    m_, s_ = X[tr].mean(0), X[tr].std(0) + 1e-9
    Bf = cox_path((X[tr] - m_) / s_, T[tr], E[tr], LAMS)
    oof[:, te] = (((X[te] - m_) / s_) @ Bf.T).T

LV = logcpm(V["counts"])[:, gkeep]
ZV = (LV - LV.mean(0)) / (LV.std(0) + 1e-9)          # within-cohort scaling for Centre C
ZC = (XC - mu) / sd                                    # control arm: same chemistry as Centre A
tV, eV = V["time"], V["event"]

used = sorted(set(np.where((B != 0).any(0))[0]))
steps = []
for j, b in enumerate(B):
    nz = np.where(b != 0)[0]
    rin, rv, rc = Zs @ b, ZV @ b, ZC @ b
    flat = len(nz) == 0
    steps.append({
        "g": [used.index(k) for k in nz], "c": [round(float(b[k]), 4) for k in nz],
        "c_in": 0.5 if flat else round(cindex(T, E, rin), 4),
        "c_cv": round(cindex(T, E, oof[j]), 4) if np.ptp(oof[j]) > 0 else 0.5,
        "c_val": 0.5 if flat else round(cindex(tV, eV, rv), 4),
        "c_ctrl": 0.5 if flat else round(cindex(tC, eC.astype(float), rc), 4),
        "thr": round(float(np.median(rin)), 4),
        "rv": [round(float(v), 3) for v in rv],
    })
    print(j, len(nz), steps[-1]["c_in"], steps[-1]["c_cv"], steps[-1]["c_val"], steps[-1]["c_ctrl"])

out = {
    "about": "Elastic-net Cox path (alpha=%.2f) on discovery PFS; Centre C validation; docetaxel control arm. Built by scripts/build_survival_lesson.py." % ALPHA,
    "alpha": ALPHA, "lam": [float(v) for v in LAMS],
    "genes": [GK[k] for k in used], "group": [group_of(GK[k]) for k in used],
    "steps": steps,
    "val": {"t": [round(float(v), 2) for v in tV], "e": [int(v) for v in eV], "resp": [int(v) for v in V["resp"]], "pheno": [str(p) for p in V["pheno"]]},
}
p = OUT / "bulk_survival_lab.json"
assert not p.exists() or "--force" in sys.argv, f"{p} exists; pass --force to rebuild"
p.write_text(json.dumps(out, separators=(",", ":")))
print("wrote", p, p.stat().st_size, "bytes")
