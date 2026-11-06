"""Synthetic-HRF injection detectability study (task-5, part 1).

Reimplements the channel-space synthetic-HRF augmentation of Fischer, Middell, Moradi &
von Luehmann, J. Neural Eng. 23 (2026) 036042, sec. 2.2.1 (DOI 10.1088/1741-2552/ae782f):
a boxcar stimulus train is convolved with an HRF basis, weighted by a spatial activation
pattern, and added to resting-state data.

Implementation choices (all stated in docs/SYNTHETIC_DETECTABILITY.md):
  * protocol: 3 blocks x 10 images, 20 s stimulus + 5 s rest, 30 s between blocks (span 810 s)
  * HRF: double-gamma, peak 6 s, undershoot 0.35
  * injection is done in the ANALYSIS domain: the evoked train is band-passed exactly as the
    data are, then scaled so that the *unfiltered* (physiological) peak HbO equals A (uM)
  * HbR is not modelled (analysis uses HbO only)
  * scenario M: identical HRF/duration per event (upper-bound power)
  * scenario R: per-event latency jitter N(0,1 s), duration 20+-2 s, amplitude lognormal(0,0.25)
  * amplitude axis = peak HbO of the evoked response in uM, before the analysis band-pass

Outputs: results_injection.json
"""
import json
import os
import sys

import numpy as np
from scipy import signal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import nulls  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

STIM, REST, GAP = 20.0, 5.0, 30.0
NPER, NBLOCK = 10, 3


def nominal_onsets(t0, stim=STIM, rest=REST, gap=GAP, nper=NPER, nblock=NBLOCK):
    ons = []
    t = t0
    for b in range(nblock):
        for k in range(nper):
            ons.append(t)
            t += stim + rest
        t += gap
    return np.array(ons)


def hrf_kernel(fs, peak_s=6.0, undershoot=0.35):
    from scipy.stats import gamma as _g
    t = np.arange(0, 40, 1.0 / fs)
    scale = peak_s / 5.0
    h = _g.pdf(t, 6, scale=scale) - undershoot * _g.pdf(t, 16, scale=scale)
    h = np.maximum(h - h[-1] * np.exp(-t / 4.0), 0)
    return h / h.max(), t


def response_train(onsets, n, fs, h, stims=None, amps=None):
    u = np.zeros(n)
    for k, o in enumerate(onsets):
        i0 = int(round(o * fs))
        d = STIM if stims is None else stims[k]
        i1 = min(int(round((o + d) * fs)), n)
        if i0 >= n:
            continue
        u[i0:i1] = 1.0 if amps is None else amps[k]
    return signal.fftconvolve(u, h)[:n]


def build_template(key, variant, fs, n):
    """Pipeline-consistent protocol template (30 events) for one session."""
    h, _ = hrf_kernel(fs)
    train = response_train(nominal_onsets(0.0), n, fs, h)
    raw_peak = float(np.abs(train).max())
    filt = common.bandpass(train[:, None], fs)[:, 0]
    filt_peak = float(np.abs(filt).max())
    g = filt_peak / raw_peak
    tmpl = filt / filt_peak
    tmd, fsd = nulls.decimate_to(tmpl, fs, 4.0)
    tmd = tmd / np.abs(tmd).max()
    return tmd, fsd, g


def make_injection_eff(tmd, fsd, t0, amp_uM, g, pattern, scenario="M", rng=None,
                       stim=STIM, rest=REST):
    """Analysis-domain injected HbO for all 18 pairs, shape (18, len)."""
    n = len(tmd)
    if scenario == "R" and rng is not None:
        ons = nominal_onsets(t0, stim, rest) + rng.normal(0, 1.0, 30)
        durs = stim + rng.normal(0, 2.0, 30)
        amps = np.clip(rng.lognormal(0.0, 0.25, 30), 0.4, 2.0)
        acc = np.zeros(n)
        for k in range(30):
            shift = int(round((ons[k] - t0) * fsd))
            acc += amps[k] * (min(max(durs[k], 4.0), 40.0) / stim) * np.roll(tmd, shift)
        r = acc / max(1e-9, np.abs(acc).max())
    else:
        r = np.roll(tmd, int(round(t0 * fsd)))
    return np.outer(pattern, r) * (amp_uM * g), r


AMP_GRID = [0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 0.90, 1.50]
PATTERNS = {
    "posterior": np.array([1.0 if j in common.POSTERIOR else (0.3 if j in common.ANTERIOR else 0.0)
                           for j in range(18)]),
    "uniform": np.array([1.0 if j in common.LONG else 0.0 for j in range(18)]),
}
ROIS = {"posterior": common.POSTERIOR, "global": common.LONG, "anterior": common.ANTERIOR}


def run(sessions, variants, n_rep=60, n_null=300, pattern_names=("posterior",),
        roi_names=("posterior", "global", "anterior"), scenarios=("M", "R"),
        seed_base=20261106, out="results_injection.json", amp_grid=None):
    rng0 = np.random.default_rng(seed_base)
    rows = []
    amp_grid = AMP_GRID if amp_grid is None else amp_grid
    for key in sessions:
        for var in variants:
            d = common.preprocess(key, var)
            fs = float(d["fs"])
            hb = d["hbo_all"]
            n = hb.shape[0]
            tmd, fsd, g = build_template(key, var, fs, n)
            for pat_name in pattern_names:
                pat = PATTERNS[pat_name]
                for roi_name in roi_names:
                    roi = ROIS[roi_name]
                    y0 = hb[:, roi].mean(1)
                    yd, fsd2 = nulls.decimate_to(y0, fs, fsd)
                    rng = np.random.default_rng(seed_base + (abs(hash((key, var, roi_name))) % 100000))
                    circ, pr, ar = [], [], []
                    for _ in range(n_null):
                        circ.append(nulls.xcorr_max_fast(
                            np.roll(yd, int(rng.integers(int(60 * fsd2), len(yd) - int(120 * fsd2)))),
                            tmd, fsd2, 90.0)[0])
                        pr.append(nulls.xcorr_max_fast(nulls.phase_randomise(yd, rng), tmd, fsd2, 90.0)[0])
                        ar.append(nulls.xcorr_max_fast(nulls.ar1_surrogate(yd, rng), tmd, fsd2, 90.0)[0])
                    circ, pr, ar = np.array(circ), np.array(pr), np.array(ar)
                    r_obs, t_obs, _ = nulls.xcorr_max_fast(yd, tmd, fsd2, 90.0)
                    base = dict(session=key, prep=var, pattern=pat_name, roi=roi_name,
                                n_samples=int(n), dur_s=round(n / fs, 2),
                                sigma_roi_uM=round(float(yd.std()), 4),
                                bandpass_gain=round(g, 4),
                                observed_r=round(float(r_obs), 4),
                                observed_t0=round(float(t_obs), 2),
                                null_phase_mean=round(float(pr.mean()), 4),
                                null_phase_p95=round(float(np.percentile(pr, 95)), 4),
                                null_phase_p99=round(float(np.percentile(pr, 99)), 4),
                                null_phase_p=round(float((pr >= r_obs).mean()), 4),
                                null_ar1_mean=round(float(ar.mean()), 4),
                                null_ar1_p95=round(float(np.percentile(ar, 95)), 4),
                                null_ar1_p=round(float((ar >= r_obs).mean()), 4),
                                null_circ_mean=round(float(circ.mean()), 4),
                                null_circ_p95=round(float(np.percentile(circ, 95)), 4),
                                null_circ_p=round(float((circ >= r_obs).mean()), 4),
                                n_null=int(n_null))
                    print(f"NULL {key} {var} {pat_name} {roi_name}: obs={r_obs:.3f} "
                          f"phase p={base['null_phase_p']:.3f} p95={base['null_phase_p95']:.3f} | "
                          f"ar1 p={base['null_ar1_p']:.3f} | circ p={base['null_circ_p']:.3f}", flush=True)
                    thr_phase = float(np.percentile(pr, 95))
                    for scen in scenarios:
                        for amp in amp_grid:
                            det, hit, rr = [], [], []
                            for rep in range(n_rep):
                                t0 = float(rng0.uniform(0.0, 60.0))
                                inj, r_eff = make_injection_eff(
                                    tmd, fsd2, t0, amp, g, pat, scenario=scen,
                                    rng=np.random.default_rng(int(rng0.integers(2 ** 31))))
                                y = yd + inj[roi, :].mean(0)
                                rm, th, _ = nulls.xcorr_max_fast(y, tmd, fsd2, 90.0)
                                det.append(rm > thr_phase)
                                err_abs = abs(th - t0)
                                err_mod = min(abs((th - t0) % 25.0), 25.0 - abs((th - t0) % 25.0))
                                hit.append(err_mod <= 2.5 or err_abs <= 5.0)
                                rr.append(rm)
                            row = dict(base)
                            row.update(scenario=scen, amp_uM=amp,
                                       amp_z=round(float(amp / max(base["sigma_roi_uM"], 1e-9)), 3),
                                       det_prob=round(float(np.mean(det)), 3),
                                       onset_hit_prob=round(float(np.mean(hit)), 3),
                                       r_mean=round(float(np.mean(rr)), 4),
                                       r_p05=round(float(np.percentile(rr, 5)), 4),
                                       r_p95=round(float(np.percentile(rr, 95)), 4))
                            rows.append(row)
                            print(f"{key:6s} {var} {pat_name[:3]} {roi_name:9s} {scen} "
                                  f"A={amp:5.2f}uM ({row['amp_z']:5.2f}z) "
                                  f"det={row['det_prob']:.2f} hit={row['onset_hit_prob']:.2f} "
                                  f"r={row['r_mean']:.3f}", flush=True)
    out_path = os.path.join(HERE, out)
    json.dump(rows, open(out_path, "w"), indent=1)
    print("wrote", out_path, len(rows), "rows")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        run(["session_a"], ["P1"], n_rep=20, n_null=100, roi_names=("posterior",),
            scenarios=("M",), out="results_injection_quick.json")
    else:
        run(["session_a", "session_b", "pilot"], ["P1", "P0", "P2"], n_rep=60, n_null=300)
