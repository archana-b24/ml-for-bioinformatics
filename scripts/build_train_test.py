"""Build data/study/train_test.json for the "Train/test split and overfitting" lesson (js/topics/train-test.js).

Uses the same generator as scripts/build_study.py (imported, not re-run; cohort.json is NOT touched) to simulate
extra patients from the same populations, which the real 180-patient cohort cannot provide:
  * truth: the true P(response | IFN-gamma) curve for the Centre A+B mix and for Centre C (kernel-smoothed p_true
    over a large simulated population), plus the IFN-gamma density in the A+B mix (weights for bias/variance).
  * reps: 40 fresh discovery cohorts (100 Centre A + 80 Centre B, as in the real study). For each, a logistic
    regression on a Legendre polynomial of IFN-gamma (degree 0-10) is fitted exactly as on the page (IRLS, L2 = 1e-3
    on non-intercept weights); the fitted probability curves are stored on a grid.
    Inputs are clipped to IFN-gamma 2.5-10 and log2 TMB 0.5-5.5 before the polynomial is formed (as on the page),
    so the polynomials do not explode outside the range they were fitted on. The page computes bias and
    variance from them.
  * learning: learning curves for logistic regression on polynomial features of (IFN-gamma, log2 TMB), degree 1 and
    6: n training patients drawn from a large simulated A+B population, scored on the real 120 Centre C patients.
Run:  source ~/storage/sc_spatial/storage_env.sh && .venv/bin/python scripts/build_train_test.py
"""
import json, pathlib, sys
import numpy as np
from numpy.polynomial import legendre as L

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_study import simulate  # noqa: E402

SEED = 4242
GRID = np.round(np.arange(2.5, 10.0001, 0.125), 3)
DEG1 = list(range(0, 11))
L2 = 1e-3


def leg(u, k):
    return L.legval(u, np.eye(k + 1)[k])


def basis1(x, deg):  # same as polyRow() in the page
    u = np.clip((np.asarray(x) - 6.25) / 3.75, -1, 1)
    return np.stack([leg(u, k) for k in range(1, deg + 1)], 1) if deg > 0 else np.zeros((len(u), 0))


def basis2(x, t, deg):  # same as poly2Row() in the page: P_i(u) P_j(v), 1 <= i + j <= deg
    u, v = np.clip((np.asarray(x) - 6.25) / 3.75, -1, 1), np.clip((np.asarray(t) - 3.0) / 2.5, -1, 1)
    return np.stack([leg(u, s - j) * leg(v, j) for s in range(1, deg + 1) for j in range(s + 1)], 1)


def irls(X, y, l2=L2, it=100):  # mirrors fitLR() in js/topics/train-test.js: Newton steps with step halving
    A = np.c_[np.ones(len(X)), X]
    b = np.zeros(A.shape[1])
    P = np.eye(len(b)) * l2
    P[0, 0] = 0

    def obj(bb):
        z = A @ bb
        return float(np.sum(np.logaddexp(0, z) - y * z) + 0.5 * bb @ P @ bb)
    f = obj(b)
    for _ in range(it):
        p = 1 / (1 + np.exp(-A @ b))
        W = np.maximum(p * (1 - p), 1e-10)
        H = A.T @ (A * W[:, None]) + P
        g = A.T @ (y - p) - P @ b
        s = np.linalg.solve(H, g)
        t = 1.0
        while t > 1e-6:
            fn = obj(b + t * s)
            if fn <= f + 1e-12:
                break
            t /= 2
        b = b + t * s
        conv = abs(f - fn) < 1e-10 * (1 + abs(f))
        f = fn
        if conv:
            break
    return b


def prob(A, b):
    return 1 / (1 + np.exp(-(b[0] + A @ b[1:])))


def logloss(p, y):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def auc(s, y):
    from scipy.stats import rankdata
    r = rankdata(s)
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def arr(rows, k):
    return np.array([r[k] for r in rows], float)


def smooth(x, v, grid, bw=0.15):
    w = np.exp(-0.5 * ((grid[:, None] - x[None]) / bw) ** 2)
    return (w @ v) / w.sum(1), w.sum(1)


def main():
    rng = np.random.default_rng(SEED)
    pop = lambda nA, nB: simulate(rng, "Centre A", nA, 1) + simulate(rng, "Centre B", nB, 1)
    big = pop(50000, 40000)
    bigC = simulate(rng, "Centre C", 50000, 1)
    tAB, dens = smooth(arr(big, "ifng"), arr(big, "p_true"), GRID)
    tC, _ = smooth(arr(bigC, "ifng"), arr(bigC, "p_true"), GRID)
    dens = dens / dens.sum()

    reps = []
    for r in range(40):
        c = pop(100, 80)
        x, y = arr(c, "ifng"), arr(c, "response")
        reps.append([np.round(prob(basis1(GRID, d), irls(basis1(x, d), y)), 3).tolist() for d in DEG1])
        print("rep", r, end="\r")

    cohort = json.loads((ROOT / "data/study/cohort.json").read_text())
    cols = cohort["columns"]
    V = [dict(zip(cols, row)) for row in cohort["rows"] if row[cols.index("cohort")] == "validation"]
    xv, tv, yv = arr(V, "ifng"), np.log2(arr(V, "tmb")), arr(V, "response")
    pool = pop(15000, 12000)
    px, pt, py = arr(pool, "ifng"), np.log2(arr(pool, "tmb")), arr(pool, "response")
    NS = [30, 45, 60, 90, 120, 180, 270, 360, 540, 720, 1080, 1440]
    learning = {}
    for deg in (1, 6):
        Av = basis2(xv, tv, deg)
        out = []
        for n in NS:
            m = {"tr_ll": [], "va_ll": [], "tr_auc": [], "va_auc": []}
            for dr in range(30):
                idx = rng.choice(len(pool), n, replace=False)
                y = py[idx]
                if y.min() == y.max():
                    continue
                A = basis2(px[idx], pt[idx], deg)
                b = irls(A, y)
                ptr, pva = prob(A, b), prob(Av, b)
                m["tr_ll"].append(logloss(ptr, y)); m["va_ll"].append(logloss(pva, yv))
                m["tr_auc"].append(auc(ptr, y)); m["va_auc"].append(auc(pva, yv))
            out.append({"n": n, **{k: [round(float(np.mean(v)), 4), round(float(np.quantile(v, 0.1)), 4), round(float(np.quantile(v, 0.9)), 4)] for k, v in m.items()}})
            print("deg", deg, "n", n, {k: round(float(np.mean(v)), 3) for k, v in m.items()})
        # large-sample limit: fit on 20,000 simulated patients
        idx = rng.choice(len(pool), 20000, replace=False)
        b = irls(basis2(px[idx], pt[idx], deg), py[idx])
        pva = prob(Av, b)
        learning[str(deg)] = {"curve": out, "limit": {"va_ll": round(logloss(pva, yv), 4), "va_auc": round(auc(pva, yv), 4)}}

    out = {
        "about": "Extra simulated patients from the generator in scripts/build_study.py (seed %d). See scripts/build_train_test.py." % SEED,
        "grid": GRID.tolist(), "deg1": DEG1, "l2": L2,
        "truthAB": np.round(tAB, 4).tolist(), "truthC": np.round(tC, 4).tolist(), "densAB": np.round(dens, 5).tolist(),
        "reps": reps, "learning": learning, "ns": NS,
    }
    (ROOT / "data/study/train_test.json").write_text(json.dumps(out, separators=(",", ":")))
    print("\nwrote data/study/train_test.json")


if __name__ == "__main__":
    main()
