"""Build a MtReNIRS-ready dataset from the local BIDS passive-auditory fNIRS dataset
(Luke et al. 2021) — verified NOT in MtReNIRS's pretraining corpus, so it is a clean
held-out test set for the audit.

Pipeline: raw NIRSCWAMPLITUDE -> optical density -> modified Beer-Lambert -> HbO/HbR
         -> bandpass -> 10 s epochs at 10 Hz (MtReNIRS contract).

Outputs data/luke_auditory.npz with
  signal   (N, C, 100) float32   # C = 33 pairs x 2 chromophores (all HbO, then all HbR)
  labels   (N,)   int            # 0=Control 1=Speech 2=Noise
  subjects (N,)   str
  ch_pos   (C, 3) float64        # channel midpoints (metres), same order as signal
  ch_types (C,)   str            # 'hbo' / 'hbr'
  fs       10.0
"""
import glob
import json
import os
import sys

import h5py
import numpy as np
from scipy.signal import butter, detrend, filtfilt, resample

BASE = ("arch/Data/2021-fNIRS-Analysis-Methods-Passive-Auditory/"
        "2021-fNIRS-Analysis-Methods-Passive-Auditory")
OUT = "data/luke_auditory.npz"

# Extinction coefficients (M^-1 cm^-1) at 760 / 850 nm (Prahl tabulation).
EPS = {760.0: {"hbo": 1486.6, "hbr": 1075.5},
       850.0: {"hbo": 1157.6, "hbr": 1913.0}}
DPF = 6.0
EPOCH_S = 10.0
TARGET_FS = 10.0
BASELINE_S = 2.0          # pre-stimulus baseline for OD reference
LABEL_MAP = {"Control": 0, "Speech": 1, "Noise": 2}


def load_groups(path):
    """Return per-channel (source, detector, wavelength) and probe geometry."""
    with h5py.File(path, "r") as f:
        n_ch = f["nirs/data1/dataTimeSeries"].shape[1]
        ml = f["nirs/data1"]
        meta = []
        for i in range(1, n_ch + 1):
            m = ml[f"measurementList{i}"]
            meta.append((int(m["sourceIndex"][()]),
                         int(m["detectorIndex"][()]),
                         float(f["nirs/probe/wavelengths"][int(m["wavelengthIndex"][()]) - 1])))
        src = f["nirs/probe/sourcePos3D"][:]
        det = f["nirs/probe/detectorPos3D"][:]
        ts = f["nirs/data1/dataTimeSeries"][:]
        t = f["nirs/data1/time"][:]
    return ts, t, meta, src, det


def mbll(ts, meta, src, det):
    """Modified Beer-Lambert: returns (pairs, 2, T) array ordered [HbO, HbR]."""
    pairs = {}
    for j, (s, d, wl) in enumerate(meta):
        pairs.setdefault((s, d), {})[wl] = j
    out, pos = [], []
    for (s, d), wls in sorted(pairs.items()):
        if len(wls) != 2:
            continue
        lams = sorted(wls)
        od = []
        for wl in lams:
            raw = ts[:, wls[wl]]
            # optical density relative to the run mean intensity
            i0 = np.nanmean(raw)
            od.append(-np.log(np.maximum(raw, 1e-12) / i0))
        od = np.vstack(od)                                    # (2, T)
        dist = np.linalg.norm(src[s - 1] - det[d - 1]) * 100.0  # metres -> cm
        e = np.array([[EPS[lams[0]]["hbo"], EPS[lams[0]]["hbr"]],
                      [EPS[lams[1]]["hbo"], EPS[lams[1]]["hbr"]]])
        scale = e * dist * DPF                                  # (2, 2) per-chromophore
        conc, *_ = np.linalg.lstsq(scale, od, rcond=None)        # (2, T) [HbO, HbR]
        out.append(conc)
        pos.append((src[s - 1] + det[d - 1]) / 2.0)
    conc = np.stack(out, axis=-1).transpose(1, 2, 0)          # (2,pairs,T) -> (T,pairs,2)
    return conc, np.array(pos)                                  # (T, pairs, 2), (pairs, 3)


def preprocess(x, fs):
    """x: (T, pairs, 2) -> bandpassed, detrended."""
    x = detrend(x, axis=0, type="linear")
    nyq = fs / 2.0
    b, a = butter(3, [0.01 / nyq, min(0.2, 0.95 * nyq) / nyq], btype="band")
    return filtfilt(b, a, x, axis=0)


def main():
    rows = sorted(glob.glob(f"{BASE}/sub-*/ses-01/nirs/*_nirs.snirf"))
    if not rows:
        sys.exit(f"no SNIRF files under {BASE}")
    X, Y, S, POS = [], [], [], []
    for path in rows:
        sub = path.split("/")[-4]
        ts, t, meta, src, det = load_groups(path)
        fs = 1.0 / np.median(np.diff(t))
        conc, pos = mbll(ts, meta, src, det)                    # (T, pairs, 2)
        conc = preprocess(conc, fs)
        ev = os.path.join(os.path.dirname(path), sub + "_ses-01_task-AudioSpeechNoise_events.tsv")
        import pandas as pd
        ev = pd.read_csv(ev, sep="\t")
        n_pairs = conc.shape[1]
        for _, r in ev.iterrows():
            onset, label = float(r["onset"]), LABEL_MAP.get(str(r["trial_type"]), -1)
            if label < 0:
                continue
            i0, i1 = int(round(onset * fs)), int(round((onset + EPOCH_S) * fs))
            if i1 > conc.shape[0]:
                continue
            seg = conc[i0:i1]                                   # (T', pairs, 2)
            n_new = int(round(EPOCH_S * TARGET_FS))
            seg = resample(seg, n_new, axis=0)
            # z-score each channel within the epoch (MtReNIRS wants run-normalised input)
            seg = (seg - seg.mean(0)) / (seg.std(0) + 1e-9)
            arr = np.concatenate([seg[:, :, 0], seg[:, :, 1]], axis=1)  # (n_new, C)
            X.append(arr.astype(np.float32))
            Y.append(label)
            S.append(sub)
        pos_all = np.concatenate([pos, pos], axis=0)             # HbO block then HbR block
        if len(POS) == 0:
            POS = pos_all
    X = np.transpose(np.stack(X), (0, 2, 1))                     # (N, C, 100)
    Y = np.array(Y)
    S = np.array(S)
    ch_types = np.array(["hbo"] * (POS.shape[0] // 2) + ["hbr"] * (POS.shape[0] // 2))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(OUT, signal=X, labels=Y, subjects=S,
                        ch_pos=POS, ch_types=ch_types, fs=TARGET_FS)
    print(f"subjects={len(set(S))} epochs={len(Y)} C={X.shape[2] if X.ndim==3 else '?'}")
    print("signal", X.shape, "labels", np.bincount(Y))
    json.dump({"n_epochs": int(len(Y)), "n_subjects": int(len(set(S))),
               "channels": int(POS.shape[0]), "fs": TARGET_FS},
              open("data/luke_auditory_meta.json", "w"), indent=2)


if __name__ == "__main__":
    main()
