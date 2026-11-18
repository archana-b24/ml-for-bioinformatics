"""Build data/omics/bulk_de_lab.json for the lesson page js/omics/bulk-de.js.

The simulation is NOT re-implemented: like build_qc_batch_lesson.py, this script executes scripts/build_bulk.py
sections 0-2 and 4 (QC, batch, DE; the subtypes section is skipped) with file writing disabled, checks that the
regenerated bulk_qc / bulk_batch / bulk_de JSON text equals the files on disk, and never writes those files.

It exports what the page needs to run per-gene linear models, limma-style variance moderation and BH live:
  * the 177 discovery patients that passed QC: response, Centre B indicator, phenotype, true tumour-cell fraction
  * uncorrected log-CPM for all 1,933 genes (rounded to 0.01, packed as base64 little-endian uint16 = 100 * value)
  * truth_lfc: the response coefficient of (noise-free, batch-free expected log-CPM) ~ response + centre,
    i.e. the bulk difference the linear model estimates, without measurement noise (known only in a simulation)
  * val: an independent replication in the 120 Centre C validation patients (one centre, so ~ response only)

Run:
  source ~/storage/sc_spatial/storage_env.sh && cd /home/abalasubramanian/storage/ml_dash && .venv/bin/python scripts/build_de_lesson.py
"""
import base64, json, pathlib
import numpy as np
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parents[1]
BB = ROOT / "scripts" / "build_bulk.py"
OUT = ROOT / "data" / "omics"
SEP = "# ============================================================================================== "

src = BB.read_text()
c3, c4, c5 = (src.index(SEP + s) for s in ("3. subtypes", "4. DE", "5. feature selection"))
part = src[:c3] + src[c4:c5]
assert part.count("    p.write_text(txt)\n") == 1
part = part.replace("    p.write_text(txt)\n", "    _CAPTURE[name] = txt\n")
ns = {"__file__": str(BB), "__name__": "build_bulk_replay", "_CAPTURE": {}}
exec(compile(part, str(BB), "exec"), ns)
for name in ("qc", "batch", "de"):
    assert ns["_CAPTURE"][name] == (OUT / f"bulk_{name}.json").read_text(), f"bulk_{name}.json drifted"
print("replay verified: bulk_qc / bulk_batch / bulk_de.json regenerate byte-identically")

X, XT, y, bB = ns["X_raw"], ns["X_truth"], ns["y"].astype(int), ns["bB"]
n, G = X.shape
Dm = np.column_stack([np.ones(n), y, bB])
truth = np.linalg.lstsq(Dm, XT, rcond=None)[0][1]

# Centre C replication: log-CPM of the validation cohort on the same genes, plain linear model ~ response.
V = ns["V"]
LV = ns["logcpm"](V["counts"])[:, ns["gkeep"]]
yv = V["resp"].astype(int)
Dv = np.column_stack([np.ones(len(yv)), yv])
bv, *_ = np.linalg.lstsq(Dv, LV, rcond=None)
dv = len(yv) - 2
s2v = ((LV - Dv @ bv) ** 2).sum(0) / dv
tv = bv[1] / np.sqrt(s2v * np.linalg.inv(Dv.T @ Dv)[1, 1])
pv = 2 * stats.t.sf(np.abs(tv), dv)

q16 = np.clip(np.round(X * 100), 0, 65535).astype("<u2")
r3 = lambda a: [float(f"{v:.3g}") for v in a]
out = {
    "ids": ns["ids"], "genes": ns["GK"], "group": ns["GGROUP"],
    "y": y.tolist(), "centreB": bB.astype(int).tolist(), "pheno": ns["pheno"].tolist(),
    "tumour_frac": [round(float(v), 3) for v in ns["TRUE_FRAC"][:, ns["CT"].index("Tumour")]],
    "x_scale": 100, "x_b64": base64.b64encode(q16.tobytes()).decode(),
    "truth_lfc": [round(float(v), 3) for v in truth],
    "val": {"n": int(len(yv)), "n_resp": int(yv.sum()), "lfc": [round(float(v), 3) for v in bv[1]], "p": r3(pv)},
}
p = OUT / "bulk_de_lab.json"
assert not p.exists() or json.loads(p.read_text()).get("x_b64") is not None, "refusing to overwrite a different file"
txt = json.dumps(out, separators=(",", ":"))
p.write_text(txt)
print(f"wrote {p.name}: {len(txt) / 1024:.0f} KB, {n} patients x {G} genes")
