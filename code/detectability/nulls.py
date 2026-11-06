"""Surrogate generators and the detection statistic for task-5.

WHY THREE DIFFERENT NULLS (documented in docs/SYNTHETIC_DETECTABILITY.md):
  * circular shift  : INVALID for a max-over-lag statistic (shift of y == shift of the design
                      family), so the test has ~no power. Replaced; kept only to demonstrate the
                      degeneracy.
  * phase randomised: preserves |FFT| exactly (so spectral peaks are NOT evidence), randomises
                      phases with phase[0]=0 and Nyquist phase=0. Valid null for "is the signal
                      aligned with THIS design".
  * AR(1)           : smooth noise with no oscillation. Valid null for "is there more periodic
                      power than a non-oscillating autocorrelated process produces".
"""
import numpy as np
from scipy import signal


def phase_randomise(x, rng):
    """Exact power-spectrum preserving surrogate (DC and Nyquist phase set to 0/pi)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    X = np.fft.rfft(x)
    ph = rng.uniform(0, 2 * np.pi, len(X))
    ph[0] = 0.0
    if n % 2 == 0:
        ph[-1] = 0.0
    return np.fft.irfft(np.abs(X) * np.exp(1j * ph), n)


def ar1_surrogate(x, rng):
    """Amplitude-matched AR(1) surrogate: same length, same std, lag-1 coefficient fitted
    to the signal, driven by Gaussian innovations."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    den = np.dot(x[:-1], x[:-1])
    a = float(np.dot(x[1:], x[:-1]) / den) if den > 0 else 0.0
    a = float(np.clip(a, 0.0, 0.995))
    sig = float(np.std(x) * np.sqrt(max(1e-12, 1 - a ** 2)))
    e = rng.normal(0.0, sig, n)
    y = np.empty(n)
    y[0] = e[0]
    for i in range(1, n):
        y[i] = a * y[i - 1] + e[i]
    y -= y.mean()
    s = y.std()
    if s > 0:
        y *= (x.std() / s)
    return y


def decimate_to(x, fs, target=4.0):
    """Anti-alias lowpass then integer decimation. Returns (x_dec, fs_dec)."""
    if fs <= target:
        return np.asarray(x, dtype=float), fs
    q = int(round(fs / target))
    fs2 = fs / q
    b, a = signal.butter(4, 0.45 * fs2 / (fs / 2), btype="low")
    y = signal.filtfilt(b, a, np.asarray(x, dtype=float), axis=0)
    return y[::q], fs2


def xcorr_max_fast(y, tmpl, fs, t0max_s=90.0, prepad=True):
    """max Pearson r of y with tmpl over circular shifts t0 in [0, t0max_s].

    Both zero-meaned; FFT so all lags at once. `prepad` pads y with zeros on the left so the
    first model sample can be placed at t0>0 without circular wrap.
    """
    y = np.asarray(y, dtype=float)
    y = y - y.mean()
    tmpl = np.asarray(tmpl, dtype=float)
    tmpl = tmpl - tmpl.mean()
    n = len(y)
    N = 1 << int(np.ceil(np.log2(2 * n)))
    c = np.fft.irfft(np.fft.rfft(y, N) * np.conj(np.fft.rfft(tmpl, N)), N)[:n]
    den = np.linalg.norm(y) * np.linalg.norm(tmpl)
    r = c / max(den, 1e-12)
    k = max(1, int(round(t0max_s * fs)))
    sub = r[:k]
    i = int(np.argmax(sub))
    return float(sub[i]), i / fs, r


def xcorr_all_lags(y, tmpl):
    """Pearson r of y with tmpl at every circular lag; returns (r_curve, n)."""
    y = np.asarray(y, float)
    y = y - y.mean()
    tmpl = np.asarray(tmpl, float)
    tmpl = tmpl - tmpl.mean()
    n = len(y)
    N = 1 << int(np.ceil(np.log2(2 * n)))
    c = np.fft.irfft(np.fft.rfft(y, N) * np.conj(np.fft.rfft(tmpl, N)), N)[:n]
    return c / max(np.linalg.norm(y) * np.linalg.norm(tmpl), 1e-12)


def smoothed_spectrum_surrogate(x, rng, smooth_bins=15, match_std=True):
    """AAFT-style surrogate that preserves the *aperiodic* spectral background only.

    The Fourier amplitudes are median-smoothed across frequency (removing narrow-band peaks such
    as a genuine 1/30 Hz oscillation) and the phases are randomised. This is the correct null for
    the question "is there narrow-band periodic power at a candidate period, above what the
    aperiodic 1/f-like background produces?" -- it does NOT have the degeneracy of a raw
    phase-randomised null when the template is strictly periodic.
    """
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    X = np.fft.rfft(x)
    amp = np.abs(X)
    k = int(max(3, smooth_bins | 1))
    if len(amp) > k:
        pad = k // 2
        ap = np.pad(amp, pad, mode="edge")
        sm = np.median(np.lib.stride_tricks.sliding_window_view(ap, k), axis=-1)
        amp = sm[: len(amp)]
    ph = rng.uniform(0, 2 * np.pi, len(X))
    ph[0] = 0.0
    if n % 2 == 0:
        ph[-1] = 0.0
    y = np.fft.irfft(amp * np.exp(1j * ph), n)
    y = y - y.mean()
    if match_std and y.std() > 1e-12:
        y *= x.std() / y.std()
    return y
