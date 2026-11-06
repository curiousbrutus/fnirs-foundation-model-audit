"""Blind structure recovery (item 2) + null calibration (item 3) + cross-session
reproducibility (item 4) for task-5.

Three unsupervised segmenters are applied to the cleaned 16-channel HbO (no protocol
information used by the methods themselves):

  A. Binary-segmentation change-point detection with an L2 cost on the first 3 PCs
     (PELT-equivalent greedy version; Killick et al. 2012), BIC penalty.
  B. 2-state diagonal-Gaussian HMM on the first 2 PCs (Baum-Welch EM + Viterbi).
  C. Sliding-window variance/amplitude event detector on the global-mean HbO.

Scoring against the protocol PRIOR (pre-registered, see report section 2):
  R1  N segments in [25,35]
  R2  median segment duration in [18,22] s
  R3  >= 50 % of inter-onset intervals in [20,30] s
  R4  exactly 2 inter-onset gaps in [25,35] s
  R5  protocol-match score M >= 0.80 and p < 0.01 vs BOTH surrogate families

M = fraction of the 30 nominal onsets (best free t0, +-3 s tolerance) that are matched by a
recovered onset.

Outputs: results_blind.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import nulls  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FS = 1.0  # analysis rate for the segmenters


# ------------------------------------------------------------------ helpers
def get_series(key, variant="P2", roi="all16"):
    d = common.preprocess(key, variant)
    fs = float(d["fs"])
    hb = d["hbo_all"][:, common.LONG]
    hb = (hb - hb.mean(0)) / (hb.std(0) + 1e-12)
    yd, fsd = nulls.decimate_to(hb, fs, FS)
    if roi == "posterior":
        yd = yd[:, [common.POSTERIOR.index(c) for c in common.POSTERIOR]]
    return yd, fsd


def pcs(X, k=3):
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:k].T, S


def binseg_l2(X, penalty, min_len=8, max_cp=80):
    """Greedy binary segmentation with an L2 (piecewise-constant) cost."""
    n, d = X.shape
    cps = []

    def cost(a, b):
        seg = X[a:b]
        return float(((seg - seg.mean(0)) ** 2).sum())

    stack = [(0, n)]
    while stack and len(cps) < max_cp:
        a, b = stack.pop()
        if b - a < 2 * min_len:
            continue
        seg = X[a:b]
        m = b - a
        cs = np.vstack([np.zeros(d), np.cumsum(seg, 0)])
        cs2 = np.concatenate([[0.0], np.cumsum((seg ** 2).sum(1))])
        k = np.arange(min_len, m - min_len + 1)
        n1 = k.astype(float)
        n2 = m - n1
        c1 = cs2[k] - (cs[k] ** 2).sum(1) / n1
        c2 = (cs2[-1] - cs2[k]) - ((cs[-1] - cs[k]) ** 2).sum(1) / n2
        tot = c1 + c2
        i = int(np.argmin(tot))
        if cost(a, b) - tot[i] > penalty:
            cp = a + int(k[i])
            cps.append(cp)
            stack.append((a, cp))
            stack.append((cp, b))
    return sorted(cps)


def hmm_2state(X, n_iter=30, seed=0):
    """2-state diagonal-Gaussian HMM, Baum-Welch EM, returns Viterbi state sequence."""
    n, d = X.shape
    rng = np.random.default_rng(seed)
    # k-means init (sklearn) on the feature matrix -- robust against the single-state collapse
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=2, n_init=10, random_state=seed).fit(X)
    st = km.labels_
    mu = np.array([X[st == k].mean(0) for k in range(2)])
    var = np.array([X[st == k].var(0) + 1e-3 for k in range(2)])
    A = np.full((2, 2), 0.2) + np.eye(2) * 0.4
    A = A / A.sum(1, keepdims=True)
    pi = np.array([0.5, 0.5])
    from scipy.stats import norm
    for _ in range(n_iter):
        # emission log-probs
        B = np.zeros((n, 2))
        for k in range(2):
            B[:, k] = norm.logpdf(X, mu[k], np.sqrt(var[k])).sum(1)
        # forward-backward in log space
        logA = np.log(A + 1e-300)
        al = np.zeros((n, 2))
        al[0] = B[0] + np.log(pi)
        for t in range(1, n):
            a0 = B[t, 0] + np.logaddexp(al[t - 1, 0] + logA[0, 0], al[t - 1, 1] + logA[1, 0])
            a1 = B[t, 1] + np.logaddexp(al[t - 1, 0] + logA[0, 1], al[t - 1, 1] + logA[1, 1])
            al[t, 0] = a0
            al[t, 1] = a1
        be = np.zeros((n, 2))
        for t in range(n - 2, -1, -1):
            be[t, 0] = np.logaddexp(logA[0, 0] + B[t + 1, 0] + be[t + 1, 0],
                                    logA[0, 1] + B[t + 1, 1] + be[t + 1, 1])
            be[t, 1] = np.logaddexp(logA[1, 0] + B[t + 1, 0] + be[t + 1, 0],
                                    logA[1, 1] + B[t + 1, 1] + be[t + 1, 1])
        ll = np.logaddexp(al[-1, 0], al[-1, 1])
        g = np.exp(al + be - ll)
        g = np.maximum(g, 1e-12)
        g /= g.sum(1, keepdims=True)
        # transitions
        xi = np.zeros((2, 2))
        for t in range(n - 1):
            for i in range(2):
                for j in range(2):
                    xi[i, j] += np.exp(al[t, i] + logA[i, j] + B[t + 1, j] + be[t + 1, j] - ll)
        A = xi / np.maximum(xi.sum(1, keepdims=True), 1e-12)
        A = np.clip(A, 1e-4, 1 - 1e-4)
        A /= A.sum(1, keepdims=True)
        pi = g[0] / g[0].sum()
        for k in range(2):
            w = g[:, k]
            mu[k] = (w[:, None] * X).sum(0) / w.sum()
            var[k] = (w[:, None] * (X - mu[k]) ** 2).sum(0) / w.sum() + 1e-6
    # viterbi
    logA = np.log(A)
    B = np.zeros((n, 2))
    for k in range(2):
        B[:, k] = norm.logpdf(X, mu[k], np.sqrt(var[k])).sum(1)
    delta = np.zeros((n, 2))
    psi = np.zeros((n, 2), dtype=int)
    delta[0] = B[0] + np.log(pi)
    for t in range(1, n):
        for k in range(2):
            j = int(np.argmax(delta[t - 1] + logA[:, k]))
            psi[t, k] = j
            delta[t, k] = delta[t - 1, j] + logA[j, k] + B[t, k]
    path = np.zeros(n, dtype=int)
    path[-1] = int(np.argmax(delta[-1]))
    for t in range(n - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]
    return path, mu, var


def var_event_detector(y, fs, win=5.0, zthr=2.0, min_gap=8.0):
    """Sliding-window variance detector on a 1-D signal -> event onsets."""
    w = max(3, int(win * fs))
    k = np.ones(w) / w
    m = np.convolve(y, k, mode="same")
    v = np.convolve((y - m) ** 2, k, mode="same")
    v = np.sqrt(np.maximum(v, 0))
    med = np.median(v)
    mad = 1.4826 * np.median(np.abs(v - med)) + 1e-12
    z = (v - med) / mad
    above = z > zthr
    onsets, durs = [], []
    last = -1e9
    i = 1
    while i < len(above):
        if above[i] and not above[i - 1]:
            j = i
            while j < len(above) and above[j]:
                j += 1
            if i / fs - last > min_gap:
                onsets.append(i / fs)
                durs.append((j - i) / fs)
                last = i / fs
            i = j
        else:
            i += 1
    return onsets, durs, z


def segments_from_onsets(onsets, dur):
    onsets = np.array(onsets)
    if len(onsets) < 2:
        return dict(n=int(len(onsets)), med_dur=float(dur), ioi_frac=0.0, gaps_25_35=0,
                    io_is=[])
    io = np.diff(onsets)
    return dict(n=int(len(onsets)), med_dur=float(dur),
                ioi_frac=float(np.mean((io >= 20) & (io <= 30))),
                gaps_25_35=int(np.sum((io >= 25) & (io <= 35))),
                io_is=[round(float(x), 2) for x in io])


def runs_from_states(path, fs):
    """Runs of the state with the larger mean (the 'task' state)."""
    mu = [path.mean() if False else 0, 0]
    # choose state with the higher mean of the PC1 (state 1 is arbitrary)
    return path


def onsets_from_state(path, X, fs):
    # the "task" state is the minority-occupancy state (stimulus epochs are shorter than rest)
    occ0 = float(np.mean(path == 0))
    task = int(occ0 > 0.5)
    if min(occ0, 1 - occ0) < 0.02:
        # degenerate: pick the state with larger variance instead
        v0 = X[path == 0].var() if (path == 0).any() else 0
        v1 = X[path == 1].var() if (path == 1).any() else 0
        task = int(v1 > v0)
    onsets, durs = [], []
    t = 0
    n = len(path)
    while t < n:
        if path[t] == task:
            s = t
            while t < n and path[t] == task:
                t += 1
            onsets.append(s / fs)
            durs.append((t - s) / fs)
        else:
            t += 1
    return onsets, durs


def protocol_match(onsets, t0_grid):
    """Max over t0 of the fraction of the 30 nominal onsets matched within +-3 s."""
    import importlib
    d2 = importlib.import_module("d02_injection")
    nominal = d2.nominal_onsets(0.0)
    onsets = np.array(onsets) if len(onsets) else np.array([np.nan])
    best = 0.0
    for t0 in t0_grid:
        nom = nominal + t0
        # fraction of nominal onsets with a recovered onset within +-3 s
        if len(onsets) == 0:
            frac = 0.0
        else:
            d = np.abs(nom[:, None] - onsets[None, :])
            frac = float((d.min(1) <= 3.0).mean())
        best = max(best, frac)
    return best


def run_methods(X, fs):
    """Return dict method -> (onsets, median_duration, n_segments)."""
    Xp, S = pcs(X, 3)
    out = {}
    # A: change points (BIC penalty)
    d = Xp.shape[1]
    pen = d * np.log(len(Xp))
    for mult in (0.5, 1.0, 2.0, 4.0):
        cps = binseg_l2(Xp, pen * mult)
        onsets = [c / fs for c in cps]
        durs = np.diff([0] + list(cps) + [len(Xp)]) / fs
        out[f"A_cp_x{mult:g}"] = (onsets, durs)
    # B: HMM on 5-s windowed features (mean + std of PC1/PC2) -- stable for autocorrelated data
    def roll(x, w):
        k = np.ones(w) / w
        m = np.convolve(x, k, mode="same")
        v = np.convolve((x - m) ** 2, k, mode="same")
        return m, np.sqrt(np.maximum(v, 0))
    win = max(3, int(5 * fs))
    m1, s1 = roll(Xp[:, 0], win)
    m2, _ = roll(Xp[:, 1], win)
    Feat = np.column_stack([m1, s1, m2])
    Feat = (Feat - Feat.mean(0)) / (Feat.std(0) + 1e-12)
    path, mu, var = hmm_2state(Feat, seed=0)
    onsets, durs = onsets_from_state(path, Feat, fs)
    out["B_hmm"] = (onsets, durs)
    # C: variance events on global mean
    y = X.mean(1)
    for zthr in (2.0, 3.0):
        onsets, durs, _ = var_event_detector(y, fs, zthr=zthr)
        out[f"C_var_z{zthr:g}"] = (onsets, durs)
    return out


def score(onsets, fs, n_expected=30):
    onsets = np.array(onsets)
    if len(onsets) < 2:
        return dict(n=len(onsets), med_dur=np.nan, ioi_frac=np.nan, gaps=0, match=0.0)
    io = np.diff(onsets)
    return dict(n=len(onsets), ioi_frac=float(np.mean((io >= 20) & (io <= 30))),
                gaps=int(np.sum((io >= 25) & (io <= 35))),
                match=protocol_match(onsets, np.arange(0, 120, 1.0)))


def main():
    rng_seed = 4242
    results = {}
    for key in ("session_a", "session_b", "pilot"):
        X, fs = get_series(key)
        rng = np.random.default_rng(rng_seed)
        obs = run_methods(X, fs)
        # null families
        nulls_runs = {}
        for fam in ("phase", "smooth"):
            nulls_runs[fam] = []
            for _ in range(50):
                y = X.mean(1)
                if fam == "phase":
                    ysur = nulls.phase_randomise(y, rng)
                else:
                    ysur = nulls.smoothed_spectrum_surrogate(y, rng)
                # rebuild a pseudo-multichannel surrogate by adding the surrogate residual
                Xsur = X - X.mean(1, keepdims=True) + (ysur - ysur.mean())[:, None]
                nulls_runs[fam].append(run_methods(Xsur, fs))
        sessions = {}
        for mname, (ons, durs) in obs.items():
            sc = score(ons, fs)
            dur = float(np.median(durs)) if len(durs) else np.nan
            # median duration from the state/segment structure: approximate by IOI/2 for
            # variance events; for A/B use the supplied duration
            rec = dict(n_segments=sc["n"], median_dur_s=dur,
                       ioi_frac_20_30=round(float(sc["ioi_frac"]), 3),
                       gaps_25_35=sc["gaps"],
                       protocol_match=round(float(sc["match"]), 3),
                       onsets_s=[round(float(o), 2) for o in ons][:60])
            for fam in ("phase", "smooth"):
                vals = np.array([len(r[mname][0]) for r in nulls_runs[fam]], dtype=float)
                ms = np.array([score(r[mname][0], fs)["match"] for r in nulls_runs[fam]])
                rec[f"null_{fam}_n_mean"] = round(float(vals.mean()), 2)
                rec[f"null_{fam}_n_p95"] = round(float(np.percentile(vals, 95)), 2)
                rec[f"p_n_{fam}"] = round(float(np.mean(vals >= sc["n"])), 3)
                rec[f"null_{fam}_match_mean"] = round(float(ms.mean()), 3)
                rec[f"null_{fam}_match_p95"] = round(float(np.percentile(ms, 95)), 3)
                rec[f"p_match_{fam}"] = round(float(np.mean(ms >= sc["match"])), 3)
            # pre-registered criteria
            rec["R1_n_25_35"] = bool(25 <= sc["n"] <= 35)
            rec["R2_meddur_18_22"] = bool(np.isfinite(dur) and 18 <= dur <= 22)
            rec["R3_ioi"] = bool(sc["ioi_frac"] >= 0.5)
            rec["R4_gaps"] = bool(sc["gaps"] == 2)
            rec["R5_stats"] = bool(rec["p_match_phase"] < 0.01 and rec["p_match_smooth"] < 0.01
                                   and sc["match"] >= 0.8)
            rec["RESCUE"] = bool(rec["R1_n_25_35"] and rec["R2_meddur_18_22"] and rec["R3_ioi"]
                                 and rec["R4_gaps"] and rec["R5_stats"])
            sessions[mname] = rec
            print(f"{key:6s} {mname:12s} N={sc['n']:3d} ioi={sc['ioi_frac']:.2f} gaps={sc['gaps']} "
                  f"match={sc['match']:.2f} p_phase={rec['p_match_phase']:.3f} "
                  f"p_smooth={rec['p_match_smooth']:.3f} RESCUE={rec['RESCUE']}", flush=True)
        results[key] = sessions
    json.dump(results, open(os.path.join(HERE, "results_blind.json"), "w"), indent=1)
    print("wrote results_blind.json")


if __name__ == "__main__":
    main()
