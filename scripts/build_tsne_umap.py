"""Build data/study/tsne_umap.json for the t-SNE / UMAP lesson.

Simulated single-cell-like data from the same fictional tumour biopsies: 600 cells in a 20-dimensional
"principal-component" space, 8 cell types with known geometry (different widths, a CD8 T-cell
trajectory, Treg close to CD4 T, endothelial far from everything), plus a 2-batch shift.
A second dataset of pure Gaussian noise (no structure) is embedded the same way.
Embeddings are precomputed with scikit-learn TSNE and umap-learn.
Run:  source ~/storage/sc_spatial/storage_env.sh && .venv/bin/python scripts/build_tsne_umap.py
"""
import json, pathlib, warnings
import numpy as np
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import adjusted_rand_score
import umap

warnings.filterwarnings("ignore")
OUT = pathlib.Path(__file__).resolve().parents[1] / "data" / "study" / "tsne_umap.json"
rng = np.random.default_rng(2026)
D = 20
TYPES = ["Tumour", "CD8 T", "CD4 T", "Treg", "B", "Macrophage", "Fibroblast", "Endothelial"]
N = [140, 120, 80, 40, 60, 70, 60, 30]
SD = [2.0, 0.6, 1.0, 0.7, 0.9, 1.4, 0.8, 0.7]
def vec(*v):
    a = np.zeros(D); a[:len(v)] = v; return a
CEN = {0: vec(0, 0, 0, 0), 2: vec(-8, 10, 0, 0), 3: vec(-8, 14, 0, 0), 4: vec(10, 18, 0, 0),
       5: vec(20, -8, 6, 0), 6: vec(14, 0, 0, 0), 7: vec(30, 30, -20, 0)}
X, lab, pt = [], [], []
for t in range(8):
    for i in range(N[t]):
        if t == 1:   # CD8 T: a curved path from naive-like (near CD4) to exhausted
            s = rng.random(); pt.append(s)
            c = vec(-11, 6, 0) + s * vec(-11, 6, 14) + 4 * np.sin(np.pi * s) * vec(0, 0, 0, 1)
        else:
            pt.append(-1.0); c = CEN[t]
        X.append(c + rng.normal(0, SD[t], D)); lab.append(t)
X = np.array(X); lab = np.array(lab); pt = np.array(pt)
batch = rng.integers(0, 2, len(X))
shift = np.zeros(D); shift[6:10] = 2.2
X = X + batch[:, None] * shift + rng.normal(0, 0.4, X.shape)
perm = rng.permutation(len(X)); X, lab, pt, batch = X[perm], lab[perm], pt[perm], batch[perm]
Xn = rng.normal(0, 2.0, X.shape)   # structure-free control

def knn(Z, k=15):
    return NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(Z)[1][:, 1:]
def pres(nn0, Y, k=15):
    nn1 = knn(Y, k)
    return float(np.mean([len(set(a) & set(b)) / k for a, b in zip(nn0, nn1)]))
def norm(Y):
    Y = Y - Y.mean(0); return Y / np.abs(Y).max() * 50

def run(Z, labels, nn0, tag):
    outs = []
    def add(m, p, md, seed, init, Y):
        Y = norm(Y)
        ari = adjusted_rand_score(labels, KMeans(8, n_init=5, random_state=0).fit_predict(Y)) if labels is not None else None
        db = DBSCAN(eps=2.2, min_samples=6).fit_predict(Y)   # Y spans about [-50, 50]
        ndb = int(db.max() + 1)
        outs.append(dict(ds=tag, ndb=ndb, m=m, p=p, md=md, seed=seed, init=init, knn=round(pres(nn0, Y), 3),
                         ari=None if ari is None else round(float(ari), 3), xy=np.round(Y, 1).ravel().tolist()))
    seeds = [1, 2, 3] if tag == "cells" else [1]
    for perp in [5, 15, 30, 60, 120]:
        for s in seeds:
            add("tsne", perp, None, s, "random", TSNE(2, perplexity=perp, init="random", random_state=s, max_iter=1000).fit_transform(Z))
    add("tsne", 30, None, 1, "pca", TSNE(2, perplexity=30, init="pca", random_state=1, max_iter=1000).fit_transform(Z))
    for nb in [5, 15, 50, 150]:
        for md in [0.05, 0.5]:
            for s in seeds:
                add("umap", nb, md, s, "spectral", umap.UMAP(n_neighbors=nb, min_dist=md, random_state=s).fit_transform(Z))
    return outs

nn_c, nn_n = knn(X), knn(Xn)
pca = lambda Z: (lambda c: c @ np.linalg.svd(c, full_matrices=False)[2][:2].T)(Z - Z.mean(0))
em = run(X, lab, nn_c, "cells") + run(Xn, None, nn_n, "noise")
res = dict(types=TYPES, lab=lab.tolist(), batch=batch.tolist(), pt=np.round(pt, 3).tolist(), d=D,
           X=np.round(X, 2).tolist(), nn=nn_c.tolist(), nn_noise=nn_n.tolist(),
           pca=np.round(norm(pca(X)), 1).ravel().tolist(), pca_noise=np.round(norm(pca(Xn)), 1).ravel().tolist(),
           ari_orig=round(float(adjusted_rand_score(lab, KMeans(8, n_init=5, random_state=0).fit_predict(X))), 3),
           ari_pca=round(float(adjusted_rand_score(lab, KMeans(8, n_init=5, random_state=0).fit_predict(norm(pca(X))))), 3),
           knn_pca=round(pres(nn_c, norm(pca(X))), 3), embeds=em)
OUT.write_text(json.dumps(res, separators=(",", ":")))
print(OUT.stat().st_size // 1024, "KB", len(em), "embeddings", res["ari_orig"], res["ari_pca"], res["knn_pca"])
for e in em[:40]: print(e["ndb"], e["ds"], e["m"], e["p"], e["md"], e["seed"], e["init"], e["knn"], e["ari"])
