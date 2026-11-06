"""Audit v2 — (a) convergence fix + 10-seed control, (c) short-separation ablation.

Runs the same LOSO protocol on three preprocessing variants of Luke 2021:
    raw  = 33 pairs, SSCs included as model channels   (reproduces v0/v1)
    long = 25 long pairs only                          (SSCs dropped)
    wss  = 25 long pairs, SSC regressed out            (Fischer et al. 2026 method)

Arms per variant: MtReNIRS pretrained, 10 independently re-initialised encoders,
classical features. Total dimensionality matched at 64 (block-wise PCA).
Convergence: features are standardised after PCA and lbfgs gets 3000 iterations;
convergence status is recorded per arm.

Run: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 audit/run_audit3.py
"""
import copy
import json
import sys
import time
import warnings

import numpy as np
import torch
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "third_party/MtReNIRS/alignment/src")
from mtrenirs.alignment import align  # noqa: E402
from mtrenirs.models import MtReNIRS  # noqa: E402

CKPT = "third_party/MtReNIRS/models/weights/mtrenirs_10s.pt"
OUT = "results/results_audit3.json"
VARIANTS = ["raw", "long", "wss"]
if len(sys.argv) > 1:
    VARIANTS = sys.argv[1].split(",")
    OUT = "results/results_audit3_" + "_".join(VARIANTS) + ".json"
SEEDS = list(range(10))
N_PCA_TOTAL = 64


def log(m):
    print(m, flush=True)


def load_variant(v):
    d = np.load(f"data/luke_{v}.npz", allow_pickle=True)
    sig = d["signal"].astype(np.float32)
    y = d["labels"].astype(int)
    subs = np.array([str(s) for s in d["subjects"]])
    cht = np.array([str(c) for c in d["ch_types"]])
    pos = d["ch_pos"]
    r = align(positions=pos, names=[f"ch{i}" for i in range(len(pos))],
              unit="m", modality="fnirs")
    return sig, y, subs, cht, r.ch_pos_1005.astype(np.float32), r.valid.astype(bool)


def embeddings(model, signal, batch=256):
    out = []
    with torch.no_grad():
        for i in range(0, len(signal), batch):
            out.append(model.embed(torch.from_numpy(signal[i:i + batch]),
                                   pool="flatten").cpu().numpy())
    E = np.concatenate(out, 0)
    return E.reshape(len(E), -1)


def reinit(model, seed):
    c = copy.deepcopy(model)
    torch.manual_seed(seed)
    with torch.no_grad():
        for _, m in c.encoder.named_modules():
            r = getattr(m, "reset_parameters", None)
            if r is not None:
                r()
    return c


def classical(signal):
    t = np.linspace(-1, 1, signal.shape[2])
    return np.concatenate([signal.mean(2), (signal @ t) / (t @ t)], 1)


def loso(blocks, y, subs):
    """Strict LOSO. Returns (fold accuracies, n_fits, n_nonconverged)."""
    accs, n_fit, n_bad = [], 0, 0
    n_pca = max(1, N_PCA_TOTAL // len(blocks))
    for s in np.unique(subs):
        tr, te = subs != s, subs == s
        trs, tes = [], []
        for X in blocks:
            sc = StandardScaler().fit(X[tr])
            Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
            k = min(n_pca, Xtr.shape[0] - 1, Xtr.shape[1])
            p = PCA(n_components=k, random_state=0).fit(Xtr)
            t2 = p.transform(Xtr)
            sc2 = StandardScaler().fit(t2)          # re-standardise -> fast convergence
            trs.append(sc2.transform(t2))
            tes.append(sc2.transform(p.transform(Xte)))
        Xtr, Xte = np.concatenate(trs, 1), np.concatenate(tes, 1)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", ConvergenceWarning)
            clf = LogisticRegression(max_iter=3000, C=1.0).fit(Xtr, y[tr])
            n_bad += sum(issubclass(x.category, ConvergenceWarning) for x in w)
        n_fit += 1
        accs.append((clf.predict(Xte) == y[te]).mean())
    return np.array(accs), n_fit, n_bad


def paired(a, b, label, store):
    diff = a - b
    t, p = stats.ttest_rel(a, b)
    try:
        _, pw = stats.wilcoxon(a, b)
    except ValueError:
        pw = float("nan")
    rng = np.random.default_rng(0)
    bs = np.array([rng.choice(diff, len(diff), replace=True).mean() for _ in range(5000)])
    ci = np.percentile(bs, [2.5, 97.5])
    log(f"  {label:36s} {diff.mean()*100:+6.2f} pp  p={p:.4f}  wilcoxon={pw:.4f}  "
        f"CI=[{ci[0]*100:+.2f},{ci[1]*100:+.2f}]")
    store[label] = {"diff_pp": float(diff.mean() * 100), "p": float(p),
                    "wilcoxon_p": float(pw), "ci95_pp": [float(ci[0] * 100), float(ci[1] * 100)]}


def main():
    t0 = time.time()
    results = {"variants": {}, "meta": {"seeds": SEEDS, "n_pca_total": N_PCA_TOTAL,
                                        "chance": 1 / 3}}
    arm_accs = {}
    for v in VARIANTS:
        log(f"\n{'='*70}\nVARIANT: {v}")
        sig, y, subs, cht, ch_pos, valid = load_variant(v)
        log(f"  signal {sig.shape}, {len(set(subs))} subjects")
        model = MtReNIRS.from_pretrained(checkpoint=CKPT, device="cpu", ch_types=list(cht),
                                         ch_pos_1005=ch_pos, ch_pos_1005_valid=valid)
        arms = {"pretrained": [embeddings(model, sig)], "classical": [classical(sig)]}
        for s in SEEDS:
            arms[f"random_seed{s}"] = [embeddings(reinit(model, s), sig)]
        accs = {}
        for name, blocks in arms.items():
            a, nf, nb = loso(blocks, y, subs)
            accs[name] = a
            log(f"  {name:22s} {a.mean():.4f} +/- {a.std():.4f}  (non-converged {nb}/{nf})")
        seeds_acc = np.array([accs[f"random_seed{s}"] for s in SEEDS])
        seed_means = seeds_acc.mean(1)
        results["variants"][v] = {
            "accuracies": {k: {"mean": float(a.mean()), "sd": float(a.std()), "folds": a.tolist()}
                           for k, a in accs.items()},
            "seed_distribution": {"mean": float(seed_means.mean()), "sd": float(seed_means.std()),
                                  "min": float(seed_means.min()), "max": float(seed_means.max()),
                                  "pretrained_percentile": float((seed_means < accs["pretrained"].mean()).mean() * 100)},
            "paired": {}}
        log(f"  seed distribution: {seed_means.mean():.4f} +/- {seed_means.std():.4f} "
            f"[{seed_means.min():.4f},{seed_means.max():.4f}]")
        log("  paired tests:")
        paired(accs["pretrained"], seeds_acc.mean(0), "pretrained - mean(10 seeds)",
               results["variants"][v]["paired"])
        paired(accs["pretrained"], accs["classical"], "pretrained - classical",
               results["variants"][v]["paired"])
        paired(seeds_acc.mean(0), accs["classical"], "mean(10 seeds) - classical",
               results["variants"][v]["paired"])
        arm_accs[v] = accs
        json.dump(results, open(OUT, "w"), indent=2)

    # cross-variant comparisons (identifical folds and labels)
    log(f"\n{'='*70}\nCROSS-VARIANT (same folds, same labels)")
    results["cross_variant"] = {}
    for a_name, b_name in [("raw", "long"), ("raw", "wss"), ("long", "wss")]:
        for arm in ("pretrained", "classical"):
            paired(arm_accs[a_name][arm], arm_accs[b_name][arm],
                   f"{arm}: {a_name} - {b_name}", results["cross_variant"])
    results["meta"]["runtime_s"] = round(time.time() - t0, 1)
    json.dump(results, open(OUT, "w"), indent=2)
    log(f"\nDONE {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
