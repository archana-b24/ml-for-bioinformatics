"""Build data/omics/bulk_qcbatch_lab.json for the lesson page js/omics/bulk-qc-batch.js.

The simulation is NOT re-implemented: like build_bulk_extra.py, this script executes scripts/build_bulk.py up to
(not including) its "3. subtypes" section with file writing disabled, checks that the regenerated bulk_qc /
bulk_batch JSON text equals the files on disk, and never writes those files.

It exports a compact gene matrix so the page can run batch correction and PCA live in the browser:
  * the 177 patients that passed QC (same order as bulk_batch.json)
  * the 150 most variable genes of the uncorrected log-CPM matrix, plus every IFNG-module gene
  * uncorrected log-CPM (X) and the noise-free, batch-free expected log-CPM (T, known only in a simulation)

Run:
  source ~/storage/sc_spatial/storage_env.sh && cd /home/abalasubramanian/storage/ml_dash && .venv/bin/python scripts/build_qc_batch_lesson.py
"""
import json, pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
BB = ROOT / "scripts" / "build_bulk.py"
OUT = ROOT / "data" / "omics"

src = BB.read_text()
cut = src.index("# ==============================================================================================" " 3. subtypes")
part = src[:cut]
assert part.count("    p.write_text(txt)\n") == 1
part = part.replace("    p.write_text(txt)\n", "    _CAPTURE[name] = txt\n")
ns = {"__file__": str(BB), "__name__": "build_bulk_replay", "_CAPTURE": {}}
exec(compile(part, str(BB), "exec"), ns)
for name in ("qc", "batch"):
    assert ns["_CAPTURE"][name] == (OUT / f"bulk_{name}.json").read_text(), f"bulk_{name}.json drifted"
print("replay verified: bulk_qc.json and bulk_batch.json regenerate byte-identically")

X_raw, X_truth, GK, IFNG_IDX = ns["X_raw"], ns["X_truth"], ns["GK"], ns["IFNG_IDX"]
group_of = ns["group_of"]
order = list(np.argsort(-X_raw.var(0))[:150])
for j in IFNG_IDX:
    if j not in order: order.append(j)
order = sorted(order)
sel = np.array(order)
out = {
    "ids": ns["ids"],
    "genes": [GK[j] for j in sel],
    "group": [group_of(GK[j]) for j in sel],
    "ifng_idx": [order.index(j) for j in IFNG_IDX],
    "X": np.round(X_raw[:, sel], 2).tolist(),
    "T": np.round(X_truth[:, sel], 2).tolist(),
}
p = OUT / "bulk_qcbatch_lab.json"
assert not p.exists() or json.loads(p.read_text()).get("genes") is not None
txt = json.dumps(out, separators=(",", ":"))
p.write_text(txt)
print(f"wrote {p.name}: {len(txt) / 1024:.0f} KB, {len(sel)} genes")
