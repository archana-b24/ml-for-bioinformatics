"""Extra data for the lesson page sc-pseudobulk: writes data/omics/sc_pseudobulk_lab.json (new file; nothing else is touched).

Re-runs the single-cell simulation of build_sc_spatial.py up to the pseudobulk section (same seeds, so the same cells),
with the output directory redirected to a scratch folder, then adds per patient x cell type counts, per-gene cell-level
values with patient pseudobulk values, effect sizes / SDs for a power curve, and random patient subsets (k vs k).
Run: source ~/storage/sc_spatial/storage_env.sh && .venv/bin/python scripts/build_sc_pseudobulk_lab.py
"""
import pathlib, json, sys
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
src = (HERE / "build_sc_spatial.py").read_text().split("\npb_out = {}")[0]
src = src.replace('OUT = ROOT / "data" / "omics"', 'OUT = pathlib.Path("/home/abalasubramanian/storage/ml_dash_rollout_scratch/pb_scratch")')
g = {"__file__": str(HERE / "build_sc_spatial.py"), "__name__": "bsc"}
exec(compile(src, "build_sc_spatial_head", "exec"), g)
np = g["np"]; itertools = g["itertools"]
GENES, PAT, NP, CT, annot, pk, Ck, Lk, resp, kept = (g[k] for k in "GENES PAT NP CT annot pk Ck Lk resp kept".split())
Ck_csr = Ck.tocsr()
planted_set, limma_lite, bh, DE_TYPES = g["planted_set"], g["limma_lite"], g["bh"], g["DE_TYPES"]
pat_resp = g["pat_resp"]
old = json.load(open(ROOT / "data/omics/sc_pseudobulk.json"))
rnd = lambda a, d=2: np.round(np.asarray(a, float), d).tolist()
out = dict(types=DE_TYPES, batch=[int(q[3]) for q in PAT], ncells={}, genes={}, subsets={}, power={})
rs = np.random.default_rng(11)
for ct in DE_TYPES:
    t = CT.index(ct)
    cells_all = np.where(annot == t)[0]
    out["ncells"][ct] = [int(np.sum(pk[cells_all] == p)) for p in range(NP)]
    pc = pk[cells_all]
    pats = [p for p in range(NP) if np.sum(pc == p) >= 10]
    cells = cells_all[np.isin(pc, pats)]; pc = pk[cells]
    expr_frac = (Ck_csr[cells] > 0).mean(0).A1
    gsel = np.where((expr_frac >= 0.05) & np.array([not x.startswith("MT-") for x in GENES]))[0]
    genes = [GENES[i] for i in gsel]
    assert genes == old["de"][ct]["genes"], ct
    PB = np.vstack([np.asarray(Ck_csr[cells[pc == p]][:, gsel].sum(0)).ravel() for p in pats])
    Y = np.log2(PB / PB.sum(1, keepdims=True) * 1e6 + 1)
    grp = pat_resp[pats]
    e = old["de"][ct]
    pl = np.array(e["planted"]).astype(bool)
    nlp_pb, nlp_nv = np.array(e["nlp_pb"]), np.array(e["nlp_nv"])
    picks = []
    ip = np.where(pl)[0]; picks.append(("strongest planted gene (pseudobulk)", int(ip[np.argmax(nlp_pb[ip])])))
    inn = np.where(~pl)[0]; picks.append(("top cell-level hit with no planted difference", int(inn[np.argmax(nlp_nv[inn])])))
    if "B2M" in genes: picks.append(("B2M (housekeeping-like)", genes.index("B2M")))
    gl = []
    for lab, j in picks:
        pcell = pc
        gl.append(dict(label=lab, gene=genes[j], planted=int(pl[j]), nlp_nv=e["nlp_nv"][j], nlp_pb=e["nlp_pb"][j],
                       lfc_pb=e["lfc_pb"][j], lfc_nv=e["lfc_nv"][j],
                       cell_pat=[int(x) for x in pcell], cell_val=rnd(Lk[cells][:, gsel[j]]),
                       pb_pat=[int(x) for x in pats], pb_val=rnd(Y[:, j])))
    out["genes"][ct] = gl
    # effect size (lfc / residual SD) for planted genes -> power curve
    a, b = Y[grp == 1], Y[grp == 0]
    df = len(a) + len(b) - 2
    sd = np.sqrt(np.maximum((((a - a.mean(0)) ** 2).sum(0) + ((b - b.mean(0)) ** 2).sum(0)) / df, 1e-8))
    out["power"][ct] = dict(n_R=int(grp.sum()), n_NR=int(len(grp) - grp.sum()), d_planted=rnd(np.abs(a.mean(0) - b.mean(0))[pl] / sd[pl], 3),
                            lfc_planted=rnd((a.mean(0) - b.mean(0))[pl], 3), sd_planted=rnd(sd[pl], 3), sd_all_median=float(np.median(sd)))
    # random k-vs-k patient subsets
    Ri = [i for i, p in enumerate(pats) if pat_resp[p] == 1]; Ni = [i for i, p in enumerate(pats) if pat_resp[p] == 0]
    sub = {}
    for k in range(2, min(len(Ri), len(Ni)) + 1):
        combos = [(r, n) for r in itertools.combinations(Ri, k) for n in itertools.combinations(Ni, k)]
        pick = combos if len(combos) <= 40 else [combos[i] for i in rs.choice(len(combos), 40, replace=False)]
        res = []
        for r, n in pick:
            idx = list(r) + list(n); gk = np.array([1] * k + [0] * k)
            _, p = limma_lite(Y[idx], gk)
            q = bh(p); top = np.argsort(p)[:20]
            res.append([int(np.sum(q < 0.05)), int(np.sum(pl[top])), round(float(-np.log10(p[pl].min())), 2), [int(pats[i]) for i in idx]])
        sub[str(k)] = res
    out["subsets"][ct] = sub
    out["power"][ct]["n_genes"] = len(genes); out["power"][ct]["n_planted"] = int(pl.sum())
    print(ct, len(pats), {k: len(v) for k, v in sub.items()}, flush=True)
s = json.dumps(out, separators=(",", ":"))
(ROOT / "data/omics/sc_pseudobulk_lab.json").write_text(s)
print("wrote", len(s) / 1e3, "kB")
