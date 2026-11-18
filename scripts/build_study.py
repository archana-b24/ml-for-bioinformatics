"""Build data/study/cohort.json: the patient-level table every rewritten concept page uses.

One mock anti-PD-1 NSCLC study (same story as scripts/mock_biology.py), reduced to the handful of
per-patient measurements a clinician or analyst would actually look at. Because we simulate it, we
KNOW the true rules, and pages can show "what the model found" next to "what is really true".

Ground truth (planted):
  * Response (yes/no) depends on IFN-gamma score (+), tumour mutational burden (+), CAF/TGF-b score (-).
    Nothing else has a direct effect.
  * PD-L1 and CD8 % are driven by the same immune activity as IFN-gamma, so they are correlated with
    response but carry (almost) no extra information once IFN-gamma is known.
  * Proliferation shortens survival (prognostic) but does NOT change response (not predictive).
  * Centre B enrolled more inflamed tumours: centre looks predictive on its own, but is not, once
    IFN-gamma is in the model (confounding).
Run:  source ~/storage/sc_spatial/storage_env.sh && .venv/bin/python scripts/build_study.py
"""
import json, pathlib
import numpy as np

OUT = pathlib.Path(__file__).resolve().parents[1] / "data" / "study"
SEED = 2026

# True response model, in natural units (log-odds scale).
TRUTH = {"intercept": -0.75, "ifng": 0.70, "log2_tmb": 0.60, "caf": -0.45}
REF = {"ifng": 6.0, "tmb": 8.0, "caf": 4.0}  # centring points used in the truth formula

PHENO_MIX = {
    "Centre A": {"inflamed": 0.25, "excluded": 0.40, "desert": 0.35},
    "Centre B": {"inflamed": 0.55, "excluded": 0.25, "desert": 0.20},
    "Centre C": {"inflamed": 0.38, "excluded": 0.32, "desert": 0.30},
}
ACT = {"inflamed": (0.85, 0.12), "excluded": (0.35, 0.12), "desert": (0.08, 0.06)}   # immune activity
CAFT = {"inflamed": (0.20, 0.12), "excluded": (0.85, 0.12), "desert": (0.35, 0.12)}  # stroma programme
CD8 = {"inflamed": 14.0, "excluded": 6.0, "desert": 2.0}


def simulate(rng, centre, n, start):
    rows = []
    phen = list(PHENO_MIX[centre])
    for i in range(n):
        ph = rng.choice(phen, p=[PHENO_MIX[centre][k] for k in phen])
        a = float(np.clip(rng.normal(*ACT[ph]), 0, 1))
        t = float(np.clip(rng.normal(*CAFT[ph]), 0, 1))
        ifng = 3.5 + 5.0 * a + rng.normal(0, 0.55)
        caf = 2.0 + 4.0 * t + rng.normal(0, 0.5)
        cd8 = max(0.2, CD8[ph] * (0.6 + 0.8 * a) * np.exp(rng.normal(0, 0.45)))
        tps = 100 / (1 + np.exp(-(-3.2 + 5.0 * a + rng.normal(0, 1.3))))
        tps = 0.0 if tps < 1 else float(round(tps))
        smoke = rng.choice(["never", "former", "current"], p=[0.15, 0.50, 0.35])
        tmb = float(np.exp(np.log(5.0) + {"never": 0, "former": 0.45, "current": 0.75}[smoke] + rng.normal(0, 0.6)))
        age = int(np.clip(round(rng.normal(66, 9)), 38, 88))
        sex = "M" if rng.random() < 0.55 else "F"
        prolif = rng.normal(5.0, 1.0)
        logit = (TRUTH["intercept"] + TRUTH["ifng"] * (ifng - REF["ifng"])
                 + TRUTH["log2_tmb"] * np.log2(tmb / REF["tmb"]) + TRUTH["caf"] * (caf - REF["caf"]))
        p = 1 / (1 + np.exp(-logit))
        resp = int(rng.random() < p)
        # survival: proliferation is prognostic, response protective, age mildly harmful (Weibull, shape 1.2)
        lp = 0.45 * (prolif - 5) - 1.1 * resp + 0.02 * (age - 66)
        scale = 20.0 * np.exp(-lp / 1.2)
        os_m = scale * rng.weibull(1.2)
        cens = rng.uniform(8, 48)
        rows.append(dict(id=f"P{start + i:03d}", centre=centre, phenotype=ph, age=age, sex=sex, smoking=smoke,
                         tmb=round(tmb, 1), pdl1=tps, ifng=round(float(ifng), 2), cd8=round(float(cd8), 1),
                         caf=round(float(caf), 2), prolif=round(float(prolif), 2), p_true=round(float(p), 3),
                         response=resp, os_months=round(float(min(os_m, cens)), 1), os_event=int(os_m <= cens)))
    return rows


def main():
    rng = np.random.default_rng(SEED)
    disc = simulate(rng, "Centre A", 100, 1) + simulate(rng, "Centre B", 80, 101)
    val = simulate(rng, "Centre C", 120, 181)
    for r in disc: r["cohort"] = "discovery"
    for r in val: r["cohort"] = "validation"
    rows = disc + val
    cols = list(rows[0].keys())
    meta = {
        "ifng": {"label": "IFN-γ score", "unit": "log2 expr.", "desc": "Average expression of 12 interferon-gamma / cytotoxic genes (CXCL9, GZMB, ...). Measures how 'hot' the immune response in the tumour is."},
        "tmb": {"label": "TMB", "unit": "mut/Mb", "desc": "Tumour mutational burden: mutations per megabase of DNA. More mutations → more neoantigens the immune system can see."},
        "caf": {"label": "CAF / TGF-β score", "unit": "log2 expr.", "desc": "Expression of the cancer-associated-fibroblast / TGF-β programme. High = dense stroma that keeps T cells out of the tumour."},
        "pdl1": {"label": "PD-L1 TPS", "unit": "%", "desc": "Tumour proportion score: % of tumour cells staining for PD-L1 (the drug's target pathway) by IHC."},
        "cd8": {"label": "CD8 T cells", "unit": "% of cells", "desc": "Share of all cells in the sample that are cytotoxic CD8 T cells."},
        "prolif": {"label": "Proliferation score", "unit": "log2 expr.", "desc": "Expression of cell-cycle genes (MKI67, TOP2A, ...). How fast the tumour is dividing."},
        "age": {"label": "Age", "unit": "years", "desc": "Age at start of treatment."},
        "smoking": {"label": "Smoking", "unit": "", "desc": "never / former / current smoker."},
        "sex": {"label": "Sex", "unit": "", "desc": "M / F."},
        "centre": {"label": "Centre", "unit": "", "desc": "Hospital that enrolled the patient (A and B = discovery; C = external validation)."},
        "phenotype": {"label": "Immune phenotype", "unit": "", "desc": "Inflamed (T cells inside tumour), excluded (T cells stuck in stroma) or desert (few T cells). Known from pathology review; hidden from most models."},
        "response": {"label": "Responded", "unit": "", "desc": "1 = tumour shrank on anti-PD-1 (RECIST partial/complete response), 0 = did not."},
        "p_true": {"label": "True response probability", "unit": "", "desc": "The probability the simulation actually used. Never available in real life."},
        "os_months": {"label": "Overall survival", "unit": "months", "desc": "Time from treatment start to death or last follow-up."},
        "os_event": {"label": "Death observed", "unit": "", "desc": "1 = died at os_months, 0 = still alive at last follow-up (censored)."},
    }
    out = {
        "about": "Mock anti-PD-1 NSCLC study: 180 discovery patients (Centres A, B) + 120 validation (Centre C). Simulated; seed %d." % SEED,
        "truth": {"response_logit": TRUTH, "centred_at": REF,
                  "formula": "logit P(response) = %.2f + %.2f·(IFN-γ − 6) + %.2f·log2(TMB / 8) %+.2f·(CAF − 4)" % (
                      TRUTH["intercept"], TRUTH["ifng"], TRUTH["log2_tmb"], TRUTH["caf"]),
                  "not_predictive": ["pdl1", "cd8", "prolif", "age", "sex", "centre", "smoking (except via TMB)"]},
        "meta": meta, "columns": cols, "rows": [[r[c] for c in cols] for r in rows],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cohort.json").write_text(json.dumps(out, separators=(",", ":"), ensure_ascii=False))
    d = [r for r in rows if r["cohort"] == "discovery"]
    print("discovery n=%d responders=%d (%.0f%%)" % (len(d), sum(r["response"] for r in d), 100 * np.mean([r["response"] for r in d])))
    for ph in ACT:
        s = [r["response"] for r in d if r["phenotype"] == ph]
        print(f"  {ph:9s} n={len(s):3d} resp={np.mean(s):.2f}")
    for c in ["Centre A", "Centre B"]:
        s = [r["response"] for r in d if r["centre"] == c]
        print(f"  {c} resp={np.mean(s):.2f}")
    v = [r["response"] for r in val]
    print("validation n=%d resp=%.2f" % (len(v), np.mean(v)))
    print("ifng range", min(r["ifng"] for r in d), max(r["ifng"] for r in d), "tmb", min(r["tmb"] for r in d), max(r["tmb"] for r in d))
    print("corr ifng~pdl1 %.2f ifng~cd8 %.2f" % (np.corrcoef([r["ifng"] for r in d], [r["pdl1"] for r in d])[0, 1],
                                               np.corrcoef([r["ifng"] for r in d], [r["cd8"] for r in d])[0, 1]))


if __name__ == "__main__":
    main()
