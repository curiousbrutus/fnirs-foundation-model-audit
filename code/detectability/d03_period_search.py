"""Free-period search (item 5) and 392.26 s block-hypothesis test (item 6) for task-5.

STATISTICAL DESIGN (pre-registered; see docs/SYNTHETIC_DETECTABILITY.md section 2):

  * Item 5 (free-period search, "is there ANY periodicity at 20-45 s?").
    Template: band-passed square wave of period p and duty d. Statistic: max over all lags of
    the Pearson r with the ROI-mean HbO. Because the template is periodic, max-over-lag is to
    first order a function of the spectral MAGNITUDE at the harmonics k/p -- so a
    spectrum-preserving null (phase randomisation) is degenerate for this question. We therefore
    use a SMOOTHED-SPECTRUM surrogate as the primary null: it keeps the 1/f background but
    removes narrow-band peaks, and randomises phases. p-value is computed over the FULL search
    space (all p, all d, all lags), i.e. it is multiple-comparison corrected by construction.
    AR(1) and raw phase-randomised nulls are also reported as diagnostics.

  * Item 6 (392.26 s block hypothesis, "is anything phase-locked to this design?").
    Template: 3 blocks spaced exactly 392.26 s apart (the measured separation of the two
    Oxysoft user events), 10 stimuli per block, free per-image spacing q in {39.23, 38.03,
    36.23} s, free stimulus duration in {15,20,25,30} s, free t0 in [0,60] s (the recording
    start must precede the experiment, so t0 cannot be arbitrarily large).
    Null: phase-randomised (valid here because the block/gap structure is NOT periodic with the
    search range) + smoothed-spectrum.

Outputs: results_period.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import nulls  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FS_ANALYSIS = 4.0

P_GRID = np.arange(20.0, 45.01, 0.5)
D_GRID = np.arange(0.50, 0.851, 0.05)

P_BLOCK = 392.26
Q_GRID = [39.226, 38.03, 36.23]
STIM_GRID = [15.0, 20.0, 25.0, 30.0]


def periodic_square(n, fs, p, duty):
    t = np.arange(n) / fs
    ph = np.mod(t, p) / p
    return (ph < duty).astype(float)


def block_design(n, fs, t0, q, stim, p_block=P_BLOCK, nper=10, nblock=3):
    u = np.zeros(n)
    for k in range(nblock):
        for j in range(nper):
            a = t0 + k * p_block + j * q
            i0 = int(round(a * fs))
            i1 = int(round((a + stim) * fs))
            if i0 >= n or i1 <= 0:
                continue
            u[max(i0, 0):min(i1, n)] = 1.0
    return u


def bandpassed(u, fs, lo=0.01, hi=0.15):
    v = common.bandpass(u[:, None], fs, lo, hi)[:, 0]
    m = np.abs(v).max()
    return v / (m if m > 0 else 1.0)


def prep_roi(key, variant="P1", roi_name="global"):
    d = common.preprocess(key, variant)
    fs = float(d["fs"])
    roi = common.LONG if roi_name == "global" else common.POSTERIOR
    y = d["hbo_all"][:, roi].mean(1)
    yd, fsd = nulls.decimate_to(y, fs, FS_ANALYSIS)
    return yd, fsd


# ---------------------------------------------------------------- item 5
def _folded_templates(n, fsd):
    """Pre-compute folded (phase-averaged) periodic templates for the whole (p,d) grid."""
    out = []
    for p in P_GRID:
        u = periodic_square(n, fsd, p, 1.0)
        L = max(4, int(round(p * fsd)))
        nper = n // L
        if nper < 4:
            continue
        for duty in D_GRID:
            t = bandpassed(periodic_square(n, fsd, p, duty), fsd)[: nper * L]
            tf = t.reshape(nper, L).mean(0)
            tf = tf - tf.mean()
            nrm = np.linalg.norm(tf)
            if nrm < 1e-9:
                continue
            out.append((float(p), float(duty), L, nper, tf / nrm))
    return out


def _max_stat_folded(y, templates):
    """Max over lags of the FULL-signal Pearson r between y and each periodic template.

    For a template of period p repeated nper times, <y,t> = nper * <y_fold, t_fold> and
    ||t|| = sqrt(nper)*||t_fold|| (t_fold normalised to unit norm), so
        r(lag) = sqrt(nper) * <y_fold, roll(t_fold,lag)> / ||y||.
    """
    best = (-9.0, None)
    ny = np.linalg.norm(y - y.mean())
    cache = {}
    for p, duty, L, nper, tf in templates:
        if L not in cache:
            seg = y[: nper * L].reshape(nper, L).mean(0)
            seg = seg - seg.mean()
            cache[L] = seg
        seg = cache[L]
        if seg is None or ny <= 0:
            continue
        c = np.fft.irfft(np.fft.rfft(seg, 2 * L) * np.conj(np.fft.rfft(tf, 2 * L)), 2 * L)[:L]
        i = int(np.argmax(c))
        r = np.sqrt(nper) * c[i] / ny
        if r > best[0]:
            best = (float(r), (p, duty, i / FS_ANALYSIS))
    return best


def period_search(key, roi_name="global", variant="P1", n_null=200, seed=7):
    yd, fsd = prep_roi(key, variant, roi_name)
    n = len(yd)
    rng = np.random.default_rng(seed)
    templates = _folded_templates(n, fsd)
    r_obs, par = _max_stat_folded(yd, templates)
    res = {}
    for name, gen in (
            ("smooth", lambda r: nulls.smoothed_spectrum_surrogate(yd, r)),
            ("ar1", lambda r: nulls.ar1_surrogate(yd, r)),
            ("phase", lambda r: nulls.phase_randomise(yd, r))):
        nsim = n_null if name == "smooth" else min(n_null, 120)
        vals = []
        for _ in range(nsim):
            vals.append(_max_stat_folded(gen(rng), templates)[0])
        vals = np.array(vals)
        res[name] = dict(mean=round(float(vals.mean()), 4),
                         p95=round(float(np.percentile(vals, 95)), 4),
                         max=round(float(vals.max()), 4),
                         p=round(float((vals >= r_obs).mean()), 4), n=int(nsim))
    out = dict(session=key, roi=roi_name, prep=variant, n=int(n),
               r_max=round(r_obs, 4), best_period_s=par[0], best_duty=round(par[1], 2),
               best_lag_s=round(par[2], 2), nulls=res)
    print(f"PERIOD {key:6s} {roi_name:9s}: Rmax={r_obs:.3f} at p={par[0]:.1f}s d={par[1]:.2f} "
          f"lag={par[2]:.1f}s | smooth p95={res['smooth']['p95']:.3f} p={res['smooth']['p']:.3f} "
          f"| ar1 p={res['ar1']['p']:.3f} | phase p={res['phase']['p']:.3f}", flush=True)
    return out


# ---------------------------------------------------------------- item 6
def block_test(key, roi_name="global", variant="P1", n_null=300, seed=11, t0max=60.0):
    yd, fsd = prep_roi(key, variant, roi_name)
    n = len(yd)
    rng = np.random.default_rng(seed)
    kk = int(round(t0max * fsd))

    def max_over_models(y):
        best = (-9.0, None)
        for q in Q_GRID:
            for stim in STIM_GRID:
                t = bandpassed(block_design(n, fsd, 0.0, q, stim), fsd)
                rc = nulls.xcorr_all_lags(y, t)[:kk]
                i = int(np.argmax(rc))
                if rc[i] > best[0]:
                    best = (float(rc[i]), (float(q), float(stim), i / fsd))
        return best

    r_obs, par = max_over_models(yd)
    res = {}
    for name, gen in (("phase", lambda r: nulls.phase_randomise(yd, r)),
                      ("smooth", lambda r: nulls.smoothed_spectrum_surrogate(yd, r))):
        vals = np.array([max_over_models(gen(rng))[0] for _ in range(n_null)])
        res[name] = dict(mean=round(float(vals.mean()), 4),
                         p95=round(float(np.percentile(vals, 95)), 4),
                         max=round(float(vals.max()), 4),
                         p=round(float((vals >= r_obs).mean()), 4), n=int(n_null))
    out = dict(session=key, roi=roi_name, prep=variant, r_max=round(r_obs, 4),
               best_q_s=par[0], best_stim_s=par[1], best_t0_s=round(par[2], 2),
               nulls=res, block_period_s=P_BLOCK)
    print(f"BLOCK  {key:6s} {roi_name:9s}: Rmax={r_obs:.3f} q={par[0]:.2f}s stim={par[1]:.0f}s "
          f"t0={par[2]:.1f}s | phase p={res['phase']['p']:.3f} p95={res['phase']['p95']:.3f} "
          f"| smooth p={res['smooth']['p']:.3f}", flush=True)
    return out


if __name__ == "__main__":
    rows_p, rows_b = [], []
    for key in ("session_a", "session_b", "pilot"):
        for roi in ("global", "posterior"):
            rows_p.append(period_search(key, roi, n_null=150))
            rows_b.append(block_test(key, roi, n_null=250))
    json.dump(dict(period_search=rows_p, block_test=rows_b),
              open(os.path.join(HERE, "results_period.json"), "w"), indent=1)
    print("wrote results_period.json")
