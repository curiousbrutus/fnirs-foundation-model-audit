"""Build three preprocessing variants of the Luke 2021 passive-auditory dataset.

The montage (identical in all 17 subjects) has:
    8 short-separation pairs, 7.6-8.4 mm   (systemic physiology, not brain)
   25 long pairs, 30.4-41.9 mm            (cortical signal)

v0/v1 used ALL 33 pairs, i.e. the short channels entered the model as if they were
cortical channels. This script produces the clean ablation:

  luke_raw.npz   33 pairs, no regression      (reproduces v0/v1 exactly)
  luke_long.npz  25 long pairs only           (drops SSC from the model input)
  luke_wss.npz   25 long pairs, SSC regressed out per subject on the continuous
                 signal before epoching      (Fischer et al. 2026 method)

SS regression: for each long channel y(t) and chromophore, fit
    y ~ 1 + SSC_1..SSC_8   (ordinary least squares, whole run)
and subtract the fitted SSC component (intercept retained). Done on continuous
data, never per epoch, so no epoch-level information leaks into the regression.

Run: python3 audit/prep_variants.py
"""
import glob
import json
import os

import h5py
import numpy as np
import pandas as pd
from scipy.signal import butter, detrend, filtfilt, resample

import os as _os
# Point LUKE_BIDS_ROOT at the BIDS root of Luke et al. (2021), which contains sub-01..sub-17.
BASE = _os.environ.get("LUKE_BIDS_ROOT", "data/bids/2021-fNIRS-Analysis-Methods-Passive-Auditory")
OUTDIR = "data"
EPS = {760.0: {"hbo": 1486.6, "hbr": 1075.5},
       850.0: {"hbo": 1157.6, "hbr": 1913.0}}
DPF = 6.0
EPOCH_S, TARGET_FS = 10.0, 10.0
SHORT_MM = 12.0
LABELS = {"Control": 0, "Speech": 1, "Noise": 2}


def load_snirf(path):
    with h5py.File(path, "r") as f:
        n_ch = f["nirs/data1/dataTimeSeries"].shape[1]
        meta = []
        for i in range(1, n_ch + 1):
            m = f["nirs/data1"][f"measurementList{i}"]
            meta.append((int(m["sourceIndex"][()]), int(m["detectorIndex"][()]),
                         float(f["nirs/probe/wavelengths"][int(m["wavelengthIndex"][()]) - 1])))
        return (f["nirs/data1/dataTimeSeries"][:], f["nirs/data1/time"][:],
                meta, f["nirs/probe/sourcePos3D"][:], f["nirs/probe/detectorPos3D"][:])


def to_concentration(ts, meta, src, det):
    """MBLL -> list of (pair, distance_mm, conc(T,2)); sorted by (source, detector)."""
    by_pair = {}
    for j, (s, d, wl) in enumerate(meta):
        by_pair.setdefault((s, d), {})[wl] = j
    pairs = []
    for (s, d) in sorted(by_pair):
        wls = by_pair[(s, d)]
        if len(wls) != 2:
            continue
        lams = sorted(wls)
        od = []
        for wl in lams:
            raw = ts[:, wls[wl]]
            od.append(-np.log(np.maximum(raw, 1e-12) / np.nanmean(raw)))
        od = np.vstack(od)
        dist_cm = np.linalg.norm(src[s - 1] - det[d - 1]) * 100.0
        e = np.array([[EPS[lams[0]]["hbo"], EPS[lams[0]]["hbr"]],
                      [EPS[lams[1]]["hbo"], EPS[lams[1]]["hbr"]]]) * dist_cm * DPF
        conc, *_ = np.linalg.lstsq(e, od, rcond=None)
        pairs.append(dict(pair=(s, d), dist_mm=dist_cm * 10.0, conc=conc.T,
                          pos=(src[s - 1] + det[d - 1]) / 2.0))
    return pairs


def bandpass(x, fs):
    x = detrend(x, axis=0, type="linear")
    nyq = fs / 2.0
    b, a = butter(3, [0.01 / nyq, min(0.2, 0.95 * nyq) / nyq], btype="band")
    return filtfilt(b, a, x, axis=0)


def ss_regress(conc, ssc_idx):
    """Subtract the SSC-predicted component from every long channel (per chromophore)."""
    out = conc.copy()
    if not len(ssc_idx):
        return out
    X = np.concatenate([np.ones((conc.shape[0], 1)), conc[:, ssc_idx, :].reshape(conc.shape[0], -1)],
                       axis=1)
    for p in range(conc.shape[1]):
        if p in ssc_idx:
            continue
        beta, *_ = np.linalg.lstsq(X, conc[:, p, :], rcond=None)   # (k, 2)
        out[:, p, :] = conc[:, p, :] - X[:, 1:] @ beta[1:]
    return out


def epoch(conc, events, fs, pairs_used):
    X, Y = [], []
    n_new = int(round(EPOCH_S * TARGET_FS))
    for _, r in events.iterrows():
        lab = LABELS.get(str(r["trial_type"]), -1)
        if lab < 0:
            continue
        i0, i1 = int(round(r["onset"] * fs)), int(round((r["onset"] + EPOCH_S) * fs))
        if i1 > conc.shape[0]:
            continue
        seg = resample(conc[i0:i1], n_new, axis=0)
        seg = (seg - seg.mean(0)) / (seg.std(0) + 1e-9)          # per-epoch z, as in v0
        X.append(np.concatenate([seg[:, :, 0], seg[:, :, 1]], 1).astype(np.float32))  # HbO then HbR
        Y.append(lab)
    return np.stack(X), np.array(Y), n_new


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    files = sorted(glob.glob(f"{BASE}/sub-*/ses-01/nirs/*_nirs.snirf"))
    variants = {k: dict(X=[], Y=[], S=[], pos=None) for k in ("raw", "long", "wss")}
    ref_pairs = None

    for path in files:
        sub = path.split("/")[-4]
        ts, t, meta, src, det = load_snirf(path)
        fs = 1.0 / np.median(np.diff(t))
        pairs = to_concentration(ts, meta, src, det)
        names = [p["pair"] for p in pairs]
        dist = np.array([p["dist_mm"] for p in pairs])
        ssc_idx = [i for i, d in enumerate(dist) if d < SHORT_MM]
        long_idx = [i for i, d in enumerate(dist) if d >= SHORT_MM]
        assert ref_pairs is None or ref_pairs == names, "montage differs across subjects"
        ref_pairs = names
        conc_all = np.stack([p["conc"] for p in pairs], 1)        # (T, pairs, 2)
        conc_all = bandpass(conc_all, fs)
        ev = pd.read_csv(os.path.join(os.path.dirname(path),
                                      f"{sub}_ses-01_task-AudioSpeechNoise_events.tsv"), sep="\t")

        for key, sel in (("raw", list(range(len(pairs)))), ("long", long_idx), ("wss", long_idx)):
            if key == "wss":
                # keep SSCs in the array so they can act as regressors, then drop them
                c = ss_regress(conc_all, ssc_idx)[:, long_idx, :]
            else:
                c = conc_all[:, sel, :]
            X, Y, n_new = epoch(c, ev, fs, sel)
            variants[key]["X"].append(X)
            variants[key]["Y"].append(Y)
            variants[key]["S"].append(np.repeat(sub, len(Y)))
            if variants[key]["pos"] is None:
                variants[key]["pos"] = np.array([pairs[i]["pos"] for i in sel])
            k = len(Y)
            # repeat the position block per chromophore, matching [HbO block, HbR block]
            if variants[key].get("_n_prev", 0) == 0:
                variants[key]["_n_prev"] = 1

    for key, v in variants.items():
        X = np.transpose(np.concatenate(v["X"], 0), (0, 2, 1))    # (N, C, T)
        Y = np.concatenate(v["Y"])
        S = np.concatenate(v["S"])
        pos = np.concatenate([v["pos"], v["pos"]], 0)             # HbO block then HbR block
        cht = np.array(["hbo"] * len(v["pos"]) + ["hbr"] * len(v["pos"]))
        np.savez_compressed(f"{OUTDIR}/luke_{key}.npz", signal=X, labels=Y, subjects=S,
                            ch_pos=pos, ch_types=cht, fs=TARGET_FS)
        print(f"luke_{key}: signal={X.shape} classes={np.bincount(Y).tolist()} C={len(cht)}")

    json.dump({"variants": ["raw", "long", "wss"],
               "n_short_separation": 8, "short_mm_range": "7.6-8.4",
               "n_long": 25, "long_mm_range": "30.4-41.9"},
              open(f"{OUTDIR}/variants_meta.json", "w"), indent=2)


if __name__ == "__main__":
    main()
