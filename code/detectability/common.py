"""Shared data loading + preprocessing for the synthetic-HRF detectability study (task-5).

Data provenance (all verified in docs/DATA_INVENTORY.md):
  session_a  50 Hz  .../fNIRS_Mind-Vis/session_a_file.txt      (59 809 rows)
  session_b 50 Hz  .../fNIRS_Mind-Vis/session_b.xlsx         (59 289 valid rows)
  init   25 Hz  .../fNIRS_Mind-Vis/trial_for pilot.txt (29 157 rows)

Channel order (export legend / .oxy5 template 114), 18 HbO/HbR pairs:
  0 Rx1-Tx1, 1 Rx1-Tx3, 2 Rx2-Tx1, 3 Rx2-Tx3,
  4 Rx3-Tx4, 5 Rx3-Tx5, 6 Rx4-Tx4, 7 Rx4-Tx5,
  8 Rx7-Tx9, 9 Rx7-Tx10, 10 Rx8-Tx9, 11 Rx8-Tx10,
  12 Rx5-Tx6, 13 Rx5-Tx8, 14 Rx6-Tx6, 15 Rx6-Tx8,
  16 Rx1-Tx2 (SSC, 10 mm), 17 Rx5-Tx7 (SSC, 10 mm)

Geometry-based region hypothesis (from docs/DATA_INVENTORY.md §3.2, medium confidence):
  POSTERIOR (occipital) = pairs 0,1,2,3,12,13,14,15
  ANTERIOR (prefrontal) = pairs 4,5,6,7,8,9,10,11
  SSC = 16,17 (both physically posterior)
"""
import os
import numpy as np
from scipy import signal

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(ROOT, "scripts", "detectability", "cache")
os.makedirs(CACHE, exist_ok=True)

FILES = {
    "session_a": ("fNIRS-Vise/fNIRS_Mind-Vis/session_a_file.txt", 50.0),
    "session_b": ("fNIRS-Vise/fNIRS_Mind-Vis/session_b.xlsx", 50.0),
    "pilot": ("fNIRS-Vise/fNIRS_Mind-Vis/trial_for pilot.txt", 25.0),
}

PAIRS = ["Rx1-Tx1", "Rx1-Tx3", "Rx2-Tx1", "Rx2-Tx3",
         "Rx3-Tx4", "Rx3-Tx5", "Rx4-Tx4", "Rx4-Tx5",
         "Rx7-Tx9", "Rx7-Tx10", "Rx8-Tx9", "Rx8-Tx10",
         "Rx5-Tx6", "Rx5-Tx8", "Rx6-Tx6", "Rx6-Tx8",
         "Rx1-Tx2(SSC)", "Rx5-Tx7(SSC)"]
POSTERIOR = [0, 1, 2, 3, 12, 13, 14, 15]
ANTERIOR = [4, 5, 6, 7, 8, 9, 10, 11]
LONG = POSTERIOR + ANTERIOR          # 16 long channels
SSC = [16, 17]

# sub-template groups (for reference / clustering comparison)
SUB = {1: [0, 1, 2, 3], 2: [4, 5, 6, 7], 3: [8, 9, 10, 11], 4: [12, 13, 14, 15]}


# ----------------------------------------------------------------------------- IO
def _read_txt_subset(fn):
    """Read an OxySoft tab-separated export, skipping the header block."""
    with open(fn, "r", errors="replace") as f:
        lines = f.readlines()
    hdr = None
    for i, l in enumerate(lines):
        p = l.rstrip("\r\n").split("\t")
        if len(p) > 30 and p[0].strip() == "1" and p[1].strip() == "2":
            hdr = i
            break
    rows = []
    for l in lines[hdr + 1:]:
        p = l.rstrip("\r\n").split("\t")
        if len(p) < 38:
            continue
        try:
            rows.append([float(x) for x in p[1:37]])
        except ValueError:
            continue
    return np.asarray(rows, dtype=float)


def _read_xlsx_subset(fn):
    import openpyxl
    wb = openpyxl.load_workbook(fn, read_only=True, data_only=True)
    ws = wb["Export 1"]
    rows = []
    for i, r in enumerate(ws.iter_rows(values_only=True)):
        if i < 89:
            continue
        vals = r[1:37]
        if all(isinstance(v, (int, float)) for v in vals):
            rows.append([float(v) for v in vals])
    wb.close()
    return np.asarray(rows, dtype=float)


def load_raw(key):
    """Return (X, fs) with X[:, 0:36] = 36 traces interleaved (HbO_0, HbR_0, HbO_1, ...)."""
    cache = os.path.join(CACHE, f"raw_{key}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return z["X"], float(z["fs"])
    fn, fs = FILES[key]
    path = os.path.join(ROOT, fn)
    if path.endswith(".xlsx"):
        X = _read_xlsx_subset(path)
    else:
        X = _read_txt_subset(path)
    X = X[~np.isnan(X).any(axis=1)]
    np.savez_compressed(cache, X=X, fs=fs)
    return X, fs


def get_hb(key):
    X, fs = load_raw(key)
    hbo = X[:, 0::2]
    hbr = X[:, 1::2]
    return hbo, hbr, fs


# ---------------------------------------------------------------- preprocessing
def repair_spikes(data, fs=50.0, thresh=8.0, pad_s=0.2, vel_s=0.2, n_iter=2):
    """Motion-artefact repair: detect windows whose smoothed temporal derivative exceeds
    `thresh` robust standard deviations (1.4826*MAD) and linearly interpolate across them.

    Transparent alternative to TDDR (Fishburn et al. 2019). It never redistributes signal
    globally, so it cannot manufacture low-frequency stimulus-locked structure.
    """
    x = np.array(data, dtype=float)
    pad = int(pad_s * fs)
    w = max(1, int(vel_s * fs))
    idx = np.arange(x.shape[0])
    for _ in range(n_iter):
        d = (x[w:] - x[:-w]) / (w / fs)
        scale = 1.4826 * np.median(np.abs(d - np.median(d, axis=0)), axis=0)
        scale = np.maximum(scale, 1e-9)
        bad_d = (np.abs(d) > thresh * scale[None, :]).any(axis=1)
        if not bad_d.any():
            break
        bad = np.zeros(x.shape[0], dtype=bool)
        bad[:-w] |= bad_d
        bad[w:] |= bad_d
        bad = np.convolve(bad.astype(float), np.ones(2 * pad + 1), mode="same") > 0
        good = ~bad
        if good.sum() < 10:
            break
        for j in range(x.shape[1]):
            x[bad, j] = np.interp(idx[bad], idx[good], x[good, j])
    return x


def ss_regress(hbo, hbr, ssc=SSC, fs=50.0, lo=0.01, hi=0.15):
    """Remove the short-separation (superficial) component: linear regression of every long
    channel's HbO/HbR on the two SSC HbO/HbR traces (within the task band)."""
    x = hbo.copy()
    reg = np.column_stack([hbr[:, ssc[0]], hbr[:, ssc[1]], hbo[:, ssc[0]], hbo[:, ssc[1]]])
    reg = signal.detrend(reg, axis=0)
    b, a = signal.butter(2, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    regf = signal.filtfilt(b, a, reg, axis=0)
    # standardise + regularise
    mu = regf.mean(0)
    sd = regf.std(0)
    sd[sd < 1e-9] = 1.0
    R = np.column_stack([(regf[:, j] - mu[j]) / sd[j] for j in range(regf.shape[1])])
    R = np.column_stack([np.ones(len(R)), R])
    out = x.copy()
    for j in range(hbo.shape[1]):
        if j in ssc:
            continue
        y = x[:, j] - x[:, j].mean()
        beta, *_ = np.linalg.lstsq(R, y, rcond=None)
        out[:, j] = y - R @ beta + x[:, j].mean()
    return out


def bandpass(x, fs, lo=0.01, hi=0.15, order=2):
    b, a = signal.butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return signal.filtfilt(b, a, x, axis=0)


def preprocess(key, variant="P1", lo=0.01, hi=0.15):
    """Return dict(fs, t, hbo_long (n,16), ssc (n,2), all (n,18), sample_numbers).

    variants:
      P0 = detrend + bandpass                        (minimal, no SS correction)
      P1 = detrend + SS regression + bandpass        (the pipeline that failed on real data)
      P2 = detrend + spike repair + SS regression + bandpass (full motion + SS correction)
    """
    cache = os.path.join(CACHE, f"prep_{key}_{variant}_{lo}_{hi}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    hbo, hbr, fs = get_hb(key)
    x = signal.detrend(hbo, axis=0)
    if variant == "P2":
        x = repair_spikes(x, fs=fs)
    if variant in ("P1", "P2"):
        x = ss_regress(x, hbr, fs=fs, lo=lo, hi=hi)
    xf = bandpass(x, fs, lo=lo, hi=hi)
    d = dict(fs=fs, t=np.arange(len(xf)) / fs, hbo_all=xf, ssc=xf[:, SSC])
    np.savez_compressed(cache, **d)
    return d


if __name__ == "__main__":
    for k in FILES:
        hbo, hbr, fs = get_hb(k)
        print(f"{k}: {hbo.shape} fs={fs} dur={len(hbo)/fs:.1f}s")
        for v in ("P0", "P1", "P2"):
            d = preprocess(k, v)
            print(f"   {v}: global-mean std={d['hbo_all'][:, LONG].mean(1).std():.4f}  "
                  f"posterior std={d['hbo_all'][:, POSTERIOR].mean(1).std():.4f}  "
                  f"anterior std={d['hbo_all'][:, ANTERIOR].mean(1).std():.4f}")
