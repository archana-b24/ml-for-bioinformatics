"""Build data/omics/bulk_clf_lab.json for the lesson page js/omics/bulk-classifier.js.

Like build_de_lesson.py, the simulation is NOT re-implemented: this script executes scripts/build_bulk.py
sections 0-2 and 4 with file writing disabled, checks that the regenerated bulk_qc / bulk_batch / bulk_de JSON
text equals the files on disk, and never writes those files.

It exports the one thing the classifier lab needs that bulk_de_lab.json does not have: the log-CPM of the
120 Centre C validation patients on the same 1,933 genes (rounded to 0.01, base64 little-endian uint16 =
100 * value), with their response and phenotype. The discovery matrix is read from bulk_de_lab.json.

Run:
  source ~/storage/sc_spatial/storage_env.sh && cd /home/abalasubramanian/storage/ml_dash && .venv/bin/python scripts/build_classifier_lesson.py
"""
import base64, json, pathlib
import numpy as np

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

lab = json.loads((OUT / "bulk_de_lab.json").read_text())
assert lab["genes"] == ns["GK"], "gene order differs from bulk_de_lab.json"
V = ns["V"]
LV = ns["logcpm"](V["counts"])[:, ns["gkeep"]]
yv = V["resp"].astype(int)
q16 = np.clip(np.round(LV * 100), 0, 65535).astype("<u2")
out = {
    "n": int(len(yv)), "y": yv.tolist(), "pheno": [str(p) for p in V["pheno"]],
    "x_scale": 100, "x_b64": base64.b64encode(q16.tobytes()).decode(),
}
p = OUT / "bulk_clf_lab.json"
assert not p.exists(), "refusing to overwrite an existing file"
txt = json.dumps(out, separators=(",", ":"))
p.write_text(txt)
print(f"wrote {p.name}: {len(txt) / 1024:.0f} KB, {len(yv)} patients x {LV.shape[1]} genes")
