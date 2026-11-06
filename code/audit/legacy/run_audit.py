"""MtReNIRS audit — first independent LOSO evaluation on a clean held-out dataset.

Dataset: Luke et al. 2021 passive auditory fNIRS (local BIDS, 17 subjects, 3 balanced
classes, 1020 epochs). Verified absent from MtReNIRS's 12-dataset pretraining corpus.

One protocol, identical folds for every model:
  A. MtReNIRS pretrained embeddings      + logistic probe
  B. randomly re-initialised encoder     + logistic probe      (from-scratch control)
  C. classical features (mean + slope)   + logistic probe
  D. permutation null, labels shuffled; PCA re-used per fold (unsupervised, no leakage)

Run:  python3 audit/run_audit.py
"""
import json
import sys
import time

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "third_party/MtReNIRS/alignment/src")
from mtrenirs.models import MtReNIRS  # noqa: E402

DATA, ALIGN = "data/luke_auditory.npz", "data/luke_aligned.npz"
OUT = "results/results_audit.json"
SEED, N_PCA, N_PERM = 0, 64, 100
CKPT = "third_party/MtReNIRS/models/weights/mtrenirs_10s.pt"


def log(msg):
    print(msg, flush=True)


def load():
    d = np.load(DATA, allow_pickle=True)
    a = np.load(ALIGN)
    return dict(signal=d["signal"].astype(np.float32),
                y=d["labels"].astype(int),
                subjects=np.array([str(s) for s in d["subjects"]]),
                ch_types=np.array([str(c) for c in d["ch_types"]]),
                ch_pos=a["ch_pos_1005"].astype(np.float32),
                valid=a["valid"].astype(bool))


def embeddings(model, signal, pool="flatten", batch=256):
    out = []
    with torch.no_grad():
        for i in range(0, len(signal), batch):
            x = torch.from_numpy(signal[i:i + batch])
            out.append(model.embed(x, pool=pool).cpu().numpy())
    E = np.concatenate(out, 0)
    return E.reshape(len(E), -1)


def precompute_folds(X, y, subjects):
    """Per-fold PCA on training data only (unsupervised, so reusable across permutations)."""
    folds = []
    for s in np.unique(subjects):
        tr, te = subjects != s, subjects == s
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        pca = PCA(n_components=min(N_PCA, Xtr.shape[0] - 1, Xtr.shape[1]), random_state=SEED)
        pca.fit(Xtr)
        folds.append(dict(tr=tr, te=te, Xtr=pca.transform(Xtr), Xte=pca.transform(Xte),
                          ytr=y[tr], yte=y[te]))
    return folds


def loso_acc(folds, y_override=None):
    accs = []
    for f in folds:
        clf = LogisticRegression(max_iter=300, C=1.0)
        clf.fit(f["Xtr"], y_override[f["tr"]] if y_override is not None else f["ytr"])
        accs.append((clf.predict(f["Xte"]) == f["yte"]).mean())
    return float(np.mean(accs)), float(np.std(accs))


def classical(d):
    x = d["signal"]
    t = np.linspace(-1, 1, x.shape[2])
    return np.concatenate([x.mean(2), (x @ t) / (t @ t)], 1)


def main():
    t0 = time.time()
    torch.manual_seed(SEED)
    d = load()
    log(f"data: signal={d['signal'].shape} subjects={len(set(d['subjects']))} "
        f"classes={np.bincount(d['y']).tolist()}")

    model = MtReNIRS.from_pretrained(checkpoint=CKPT, device="cpu",
                                     ch_types=list(d["ch_types"]),
                                     ch_pos_1005=d["ch_pos"],
                                     ch_pos_1005_valid=d["valid"])
    n_par = sum(p.numel() for p in model.encoder.parameters())
    log(f"MtReNIRS loaded: {n_par:,} params ({time.time()-t0:.1f}s)")

    feats = {}
    log("[A] MtReNIRS pretrained embeddings ...")
    feats["mtreNIRS_flatten"] = embeddings(model, d["signal"], "flatten")
    feats["mtreNIRS_tmean"] = embeddings(model, d["signal"], "temporal_mean").reshape(len(d["y"]), -1)
    log(f"    flatten {feats['mtreNIRS_flatten'].shape}, tmean {feats['mtreNIRS_tmean'].shape}")

    log("[B] random-init control ...")
    import copy
    rnd = copy.deepcopy(model)
    with torch.no_grad():
        for _, m in rnd.encoder.named_modules():
            reset = getattr(m, "reset_parameters", None)
            if reset is not None:
                reset()
    feats["random_flatten"] = embeddings(rnd, d["signal"], "flatten")

    log("[C] classical features ...")
    feats["classical"] = classical(d)

    results = {"meta": {"n_epochs": int(len(d["y"])), "n_subjects": int(len(set(d["subjects"]))),
                        "n_channels": int(d["signal"].shape[1]), "classes": np.bincount(d["y"]).tolist(),
                        "params": int(n_par), "chance": float(1 / 3),
                        "dataset": "Luke2021 passive auditory (clean, not in MtReNIRS pretraining)"}}
    rng = np.random.default_rng(SEED)
    for name, X in feats.items():
        log(f"\n>>> {name} ({X.shape[1]} dims)")
        folds = precompute_folds(X, d["y"], d["subjects"])
        acc, sd = loso_acc(folds)
        log(f"    LOSO acc = {acc:.4f} +/- {sd:.4f}")
        entry = {"acc": acc, "sd": sd, "dims": int(X.shape[1])}
        if name in ("mtreNIRS_flatten", "mtreNIRS_tmean", "random_flatten"):
            log(f"    permutation null ({N_PERM}) ...")
            null = np.array([loso_acc(folds, rng.permutation(d["y"]))[0] for _ in range(N_PERM)])
            entry.update(null_mean=float(null.mean()), null_p95=float(np.percentile(null, 95)),
                         p_value=float((null >= acc).mean()))
            log(f"    null mean={null.mean():.4f} p95={np.percentile(null,95):.4f} "
                f"p={(null>=acc).mean():.4f}")
        results[name] = entry
        json.dump(results, open(OUT, "w"), indent=2)

    results["meta"]["runtime_s"] = round(time.time() - t0, 1)
    json.dump(results, open(OUT, "w"), indent=2)
    log(f"\nDONE in {time.time()-t0:.0f}s -> {OUT}")
    log("\n=== SUMMARY ===")
    for k, v in results.items():
        if isinstance(v, dict) and "acc" in v:
            p = v.get("p_value")
            log(f"{k:22s} acc={v['acc']:.4f} sd={v['sd']:.4f}"
                + (f" p={p:.4f}" if p is not None else ""))


if __name__ == "__main__":
    main()
