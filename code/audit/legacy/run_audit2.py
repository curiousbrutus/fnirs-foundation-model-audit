"""Audit v1 — harden the central contrast and test whether combining representations helps.

Adds to v0 (audit/RESULTS_v0.md):
  (1) MULTI-SEED random-init control: 5 independently re-initialised encoders, so the
      pretrained-vs-random contrast is judged against the seed distribution, not one draw.
  (4) FEATURE-BANK STACKING: pretrained embeddings + classical features (block-wise PCA,
      fitted on training folds only), to test whether combining representations beats any
      single one. Also stacks random-init + classical as the matched control.

Protocol identical to v0: strict LOSO (17 folds), standardise on training subjects only,
block-wise PCA-32 per block -> concatenate, logistic regression. No permutation null here
(already established in v0 that both embedding arms beat chance); this run is about the
contrast between arms, so paired tests across the 17 folds are the relevant statistic.

Run:  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 audit/run_audit2.py
"""
import copy
import json
import sys
import time

import numpy as np
import torch
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "third_party/MtReNIRS/alignment/src")
from mtrenirs.models import MtReNIRS  # noqa: E402

DATA, ALIGN = "data/luke_auditory.npz", "data/luke_aligned.npz"
OUT = "results/results_audit2.json"
CKPT = "third_party/MtReNIRS/models/weights/mtrenirs_10s.pt"
SEEDS = [0, 1, 2, 3, 4]
N_PCA_TOTAL = 64   # matched total dimensionality across arms


def log(m):
    print(m, flush=True)


def load():
    d = np.load(DATA, allow_pickle=True)
    a = np.load(ALIGN)
    return (d["signal"].astype(np.float32), d["labels"].astype(int),
            np.array([str(s) for s in d["subjects"]]),
            np.array([str(c) for c in d["ch_types"]]),
            a["ch_pos_1005"].astype(np.float32), a["valid"].astype(bool))


def embeddings(model, signal, batch=256):
    out = []
    with torch.no_grad():
        for i in range(0, len(signal), batch):
            out.append(model.embed(torch.from_numpy(signal[i:i + batch]),
                                   pool="flatten").cpu().numpy())
    E = np.concatenate(out, 0)
    return E.reshape(len(E), -1)


def reinit(model, seed):
    clone = copy.deepcopy(model)
    torch.manual_seed(seed)
    with torch.no_grad():
        for _, m in clone.encoder.named_modules():
            r = getattr(m, "reset_parameters", None)
            if r is not None:
                r()
    return clone


def classical(signal):
    t = np.linspace(-1, 1, signal.shape[2])
    return np.concatenate([signal.mean(2), (signal @ t) / (t @ t)], 1)


def block_features(blocks, tr, te, n_pca=None):
    """Standardise + PCA each block on TRAIN ONLY, then concatenate.

    Total dimensionality is held constant at N_PCA_TOTAL so a two-block stack is not
    advantaged over a one-block arm simply by having more features."""
    n_pca = n_pca or max(1, N_PCA_TOTAL // len(blocks))
    out_tr, out_te = [], []
    for X in blocks:
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        k = min(n_pca, Xtr.shape[0] - 1, Xtr.shape[1])
        p = PCA(n_components=k, random_state=0).fit(Xtr)
        out_tr.append(p.transform(Xtr))
        out_te.append(p.transform(Xte))
    return np.concatenate(out_tr, 1), np.concatenate(out_te, 1)


def loso(blocks, y, subjects):
    accs = []
    for s in np.unique(subjects):
        tr, te = subjects != s, subjects == s
        Xtr, Xte = block_features(blocks, tr, te)
        clf = LogisticRegression(max_iter=300, C=1.0).fit(Xtr, y[tr])
        accs.append((clf.predict(Xte) == y[te]).mean())
    return np.array(accs)


def paired(a, b, label, res):
    diff = a - b
    t, p = stats.ttest_rel(a, b)
    try:
        _, pw = stats.wilcoxon(a, b)
    except ValueError:
        pw = float("nan")
    rng = np.random.default_rng(0)
    bs = np.array([rng.choice(diff, len(diff), replace=True).mean() for _ in range(5000)])
    ci = np.percentile(bs, [2.5, 97.5])
    log(f"{label:34s} diff={diff.mean()*100:+6.2f} pp  t-p={p:.4f}  "
        f"wilcoxon-p={pw:.4f}  95%CI=[{ci[0]*100:+.2f},{ci[1]*100:+.2f}] pp")
    res[label] = {"diff_pp": float(diff.mean() * 100), "t_p": float(p),
                  "wilcoxon_p": float(pw), "ci95_pp": [float(ci[0] * 100), float(ci[1] * 100)]}
    return res


def main():
    t0 = time.time()
    sig, y, subs, cht, ch_pos, valid = load()
    log(f"data {sig.shape}, {len(set(subs))} subjects, classes {np.bincount(y).tolist()}")

    model = MtReNIRS.from_pretrained(checkpoint=CKPT, device="cpu", ch_types=list(cht),
                                     ch_pos_1005=ch_pos, ch_pos_1005_valid=valid)
    E_pre = embeddings(model, sig)
    E_rnd = {s: embeddings(reinit(model, s), sig) for s in SEEDS}
    F_cls = classical(sig)
    log(f"embeddings {E_pre.shape}; {len(SEEDS)} random seeds; classical {F_cls.shape}")

    arms = {"pretrained": [E_pre], "classical": [F_cls],
            "stack_pretrained_classical": [E_pre, F_cls]}
    for s in SEEDS:
        arms[f"random_seed{s}"] = [E_rnd[s]]
    arms["stack_random0_classical"] = [E_rnd[0], F_cls]

    accs = {}
    for name, blocks in arms.items():
        a = loso(blocks, y, subs)
        accs[name] = a
        log(f"{name:30s} LOSO={a.mean():.4f} +/- {a.std():.4f}")

    seeds = np.array([accs[f"random_seed{s}"].mean() for s in SEEDS])
    log(f"\nrandom-init seeds: mean={seeds.mean():.4f} sd={seeds.std():.4f} "
        f"range=[{seeds.min():.4f},{seeds.max():.4f}]")
    pre = accs["pretrained"].mean()
    pct = float((seeds < pre).mean() * 100)
    log(f"pretrained={pre:.4f} -> percentile within seed distribution: {pct:.0f}%")

    res = {"accuracies": {k: {"mean": float(v.mean()), "sd": float(v.std()),
                              "folds": v.tolist()} for k, v in accs.items()},
           "random_seed_distribution": {"mean": float(seeds.mean()), "sd": float(seeds.std()),
                                        "min": float(seeds.min()), "max": float(seeds.max()),
                                        "pretrained_percentile": pct},
           "paired": {}}
    log("\n=== paired tests across the 17 folds ===")
    paired(accs["pretrained"], accs["random_seed0"], "pretrained - random_seed0", res["paired"])
    paired(accs["pretrained"], np.mean([accs[f"random_seed{s}"] for s in SEEDS], 0),
           "pretrained - mean(random seeds)", res["paired"])
    paired(accs["pretrained"], accs["classical"], "pretrained - classical", res["paired"])
    paired(accs["stack_pretrained_classical"], accs["pretrained"],
           "stack - pretrained", res["paired"])
    paired(accs["stack_pretrained_classical"], accs["classical"],
           "stack - classical", res["paired"])
    paired(accs["stack_random0_classical"], accs["random_seed0"],
           "stack_random - random0", res["paired"])

    res["meta"] = {"n_folds": int(len(set(subs))), "chance": 1 / 3,
                   "n_epochs": int(len(y)), "seeds": SEEDS,
                   "runtime_s": round(time.time() - t0, 1)}
    json.dump(res, open(OUT, "w"), indent=2)
    log(f"\nDONE {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
