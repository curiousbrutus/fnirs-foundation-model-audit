"""Montage-free channel grouping (task-5 item 5).

Question: without a digitised montage, can the 16 long channels be split into an
anterior/prefrontal and a posterior/occipital group from the signals alone?

Approach (uses NO geometry):
  1. clean the data (P2: detrend + spike repair + SSC regression + band-pass)
  2. build the 16x16 inter-channel correlation matrix in the haemodynamic band (0.01-0.10 Hz)
     and in the cardiac band (0.8-1.5 Hz)
  3. three independent 2-group splits:
       (a) Ward hierarchical clustering on correlation distance (1 - r)
       (b) spectral clustering on the (positive) correlation affinity
       (c) k-means on the loadings of the first 4 ICA components (FastICA)
  4. validation:
       (i)  SSC adjacency: both short-separation channels are physically posterior, so the
            group whose channels couple more strongly to the SSC traces is the posterior group
       (ii) cross-session stability: Adjusted Rand Index between the Session A and Session B splits
       (iii) recovery of the geometry hypothesis (posterior = pairs 0-3,12-15)
       (iv) bootstrap stability over 20 time-segment resamples

Outputs: results_channels.json
"""
import json
import os
import sys

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import FastICA
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
GEOM_POSTERIOR = common.POSTERIOR       # [0,1,2,3,12,13,14,15]


def corr_matrix(key, band=(0.01, 0.10), variant="P2"):
    d = common.preprocess(key, variant)
    fs = float(d["fs"])
    x = d["hbo_all"][:, common.LONG]
    from scipy import signal
    b, a = signal.butter(2, [band[0] / (fs / 2), band[1] / (fs / 2)], btype="band")
    xf = signal.filtfilt(b, a, x, axis=0)
    C = np.corrcoef(xf.T)
    return C, xf, fs, d


def ssc_coupling(key, variant="P2"):
    """|corr| of each long channel with the two SSC traces (low band) -> posterior evidence."""
    d = common.preprocess(key, variant)
    fs = float(d["fs"])
    from scipy import signal
    b, a = signal.butter(2, [0.01 / (fs / 2), 0.10 / (fs / 2)], btype="band")
    xf = signal.filtfilt(b, a, d["hbo_all"], axis=0)
    ssc = xf[:, common.SSC]           # (n,2)
    out = []
    for j in common.LONG:
        c = [abs(np.corrcoef(xf[:, j], ssc[:, k])[0, 1]) for k in range(2)]
        out.append(float(np.mean(c)))
    return np.array(out)


def split_ward(C):
    D = 1.0 - C
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2
    Z = linkage(squareform(D, checks=False), method="ward")
    lab = fcluster(Z, t=2, criterion="maxclust")
    return np.unique(lab, return_inverse=True)[1]


def split_spectral(C):
    A = np.clip(C, 0, None)
    np.fill_diagonal(A, 0)
    sc = SpectralClustering(n_clusters=2, affinity="precomputed",
                            random_state=0).fit(A)
    return np.unique(sc.labels_, return_inverse=True)[1]


def split_ica(xf, fs, n_comp=4):
    X = (xf - xf.mean(0)) / (xf.std(0) + 1e-12)
    ica = FastICA(n_components=n_comp, random_state=0, max_iter=800, tol=1e-3)
    S = ica.fit_transform(X)          # (n, n_comp)
    A = ica.mixing_                    # (16, n_comp) channel loadings
    lab = KMeans(n_clusters=2, n_init=20, random_state=0).fit_predict(A)
    return lab, A


def bootstrap_stability(key, band=(0.01, 0.10), n_boot=30, seed=0, block_s=30.0):
    """Proper block bootstrap: resample 30-s time blocks with replacement, recompute the
    correlation matrix, recluster, and compare with the full-data split (Adjusted Rand)."""
    d = common.preprocess(key, "P2")
    fs = float(d["fs"])
    from scipy import signal
    x = d["hbo_all"][:, common.LONG]
    b, a = signal.butter(2, [band[0] / (fs / 2), band[1] / (fs / 2)], btype="band")
    xf = signal.filtfilt(b, a, x, axis=0)
    ref = split_ward(np.corrcoef(xf.T))
    n = len(xf)
    blk = int(block_s * fs)
    nb = n // blk
    rng = np.random.default_rng(seed)
    aris = []
    for _ in range(n_boot):
        idx = rng.integers(0, nb, nb)
        parts = [xf[i * blk:(i + 1) * blk] for i in idx]
        xb = np.vstack(parts)
        try:
            aris.append(adjusted_rand_score(ref, split_ward(np.corrcoef(xb.T))))
        except Exception:
            pass
    return float(np.mean(aris)) if aris else float("nan")


def align_to_geometry(lab):
    """Return 1 if the cluster containing channels 0..3 is the same as that of 12..15."""
    a, b = lab[0], lab[12]
    return a == b


def posterior_cluster(lab, ssc):
    m0 = ssc[lab == 0].mean()
    m1 = ssc[lab == 1].mean()
    return 0 if m0 > m1 else 1, m0, m1


def main():
    out = {}
    for key in ("session_a", "session_b"):
        C, xf, fs, d = corr_matrix(key)
        ssc = ssc_coupling(key)
        lab_ward = split_ward(C)
        lab_spec = split_spectral(C)
        lab_ica, A = split_ica(xf, fs)
        geom = np.array([1 if j in GEOM_POSTERIOR else 0 for j in range(16)])
        pcl_w, m0w, m1w = posterior_cluster(lab_ward, ssc)
        pcl_s, m0s, m1s = posterior_cluster(lab_spec, ssc)
        pcl_i, m0i, m1i = posterior_cluster(lab_ica, ssc)
        rec = dict(
            session=key,
            ward=lab_ward.tolist(), spectral=lab_spec.tolist(), ica=lab_ica.tolist(),
            geometry=geom.tolist(),
            ssc_coupling=[round(float(v), 3) for v in ssc],
            ward_posterior_cluster=int(pcl_w), ward_ssc_mean=[round(m0w, 3), round(m1w, 3)],
            spectral_posterior_cluster=int(pcl_s), spectral_ssc_mean=[round(m0s, 3), round(m1s, 3)],
            ica_posterior_cluster=int(pcl_i), ica_ssc_mean=[round(m0i, 3), round(m1i, 3)],
            ari_ward_geom=round(float(adjusted_rand_score(geom, lab_ward)), 3),
            ari_spec_geom=round(float(adjusted_rand_score(geom, lab_spec)), 3),
            ari_ica_geom=round(float(adjusted_rand_score(geom, lab_ica)), 3),
            ari_ward_spec=round(float(adjusted_rand_score(lab_ward, lab_spec)), 3),
            ari_ward_ica=round(float(adjusted_rand_score(lab_ward, lab_ica)), 3),
            bootstrap_stability=round(bootstrap_stability(key), 3),
            k4_ward=fcluster(linkage(squareform(np.clip(1 - C, 0, 2) * (1 - np.eye(16)), checks=False),
                                     method="ward"), t=4, criterion="maxclust").tolist(),
            n_same_subtemplate_kept=bool(align_to_geometry(lab_ward)),
        )
        out[key] = rec
        print(f"== {key}")
        print("  ward    :", lab_ward.tolist(), " ARI vs geometry:", rec["ari_ward_geom"],
              " posterior cluster:", pcl_w, rec["ward_ssc_mean"])
        print("  spectral:", lab_spec.tolist(), " ARI vs geometry:", rec["ari_spec_geom"],
              " posterior cluster:", pcl_s)
        print("  ica     :", lab_ica.tolist(), " ARI vs geometry:", rec["ari_ica_geom"],
              " posterior cluster:", pcl_i)
        print("  SSC coupling per channel:", np.round(ssc, 3))
        print("  bootstrap stability (ward):", rec["bootstrap_stability"])
    # cross-session agreement
    w = np.array([out["session_a"]["ward"], out["session_b"]["ward"]])
    out["cross_session"] = dict(
        ari_ward_session_a_session_b=round(float(adjusted_rand_score(w[0], w[1])), 3),
        ari_spec_session_a_session_b=round(float(adjusted_rand_score(out["session_a"]["spectral"],
                                                              out["session_b"]["spectral"])), 3),
        ari_ica_session_a_session_b=round(float(adjusted_rand_score(out["session_a"]["ica"],
                                                             out["session_b"]["ica"])), 3),
    )
    print("cross-session:", out["cross_session"])
    json.dump(out, open(os.path.join(HERE, "results_channels.json"), "w"), indent=1)
    print("wrote results_channels.json")


if __name__ == "__main__":
    main()
