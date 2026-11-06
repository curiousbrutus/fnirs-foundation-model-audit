"""Figures for docs/SYNTHETIC_DETECTABILITY.md (task-5)."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
FIG = os.path.join(ROOT, "docs", "figures")
# Participant names are replaced by neutral session codes in every published figure.
LABEL = {"session_a": "Session A", "session_b": "Session B", "pilot": "Pilot"}
os.makedirs(FIG, exist_ok=True)


def fig_detectability():
    rows = json.load(open(os.path.join(HERE, "results_injection.json")))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, key in zip(axes, ("session_a", "session_b")):
        for scen, col in (("M", "C0"), ("R", "C3")):
            sel = sorted([r for r in rows if r["session"] == key and r["prep"] == "P1"
                          and r["pattern"] == "posterior" and r["roi"] == "posterior"
                          and r["scenario"] == scen], key=lambda r: r["amp_uM"])
            a = [r["amp_uM"] for r in sel]
            p = [r["det_prob"] for r in sel]
            ax.plot(a, p, "o-", color=col,
                    label=("matched template" if scen == "M" else "realistic jitter (R)"))
        ax.axhline(0.8, ls="--", c="grey", lw=0.8)
        ax.axhline(0.5, ls=":", c="grey", lw=0.8)
        ax.axvspan(0.1, 0.5, color="gold", alpha=0.18,
                   label="plausible visual HbO range (0.1-0.5 uM)")
        sig = [r for r in rows if r["session"] == key and r["prep"] == "P1"][0]["sigma_roi_uM"]
        ax.axvline(sig, color="k", lw=0.8, ls="-.")
        ax.text(sig * 1.03, 0.05, f"noise SD\n{sig:.2f} uM", fontsize=7)
        ax.set_xscale("log")
        ax.set_xlabel("injected peak HbO amplitude (uM)")
        ax.set_title(LABEL.get(key, key))
        ax.grid(alpha=0.25)
        ax.set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("detection probability (phase-randomised p95 threshold)")
    axes[0].legend(fontsize=7, loc="lower right")
    fig.suptitle("Synthetic-HRF detectability: P1 preprocessing, posterior ROI, 20 s/5 s/30 s protocol")
    fig.tight_layout()
    out = os.path.join(FIG, "fig_detectability.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("wrote", out)


def fig_nulls():
    rows = json.load(open(os.path.join(HERE, "results_injection.json")))
    cfgs = [(k, roi) for k in ("session_a", "session_b", "pilot") for roi in ("posterior", "global", "anterior")]
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    x = np.arange(len(cfgs))
    obs, p95, mean = [], [], []
    for k, roi in cfgs:
        r = [q for q in rows if q["session"] == k and q["roi"] == roi and q["prep"] == "P1"][0]
        obs.append(r["observed_r"])
        p95.append(r["null_phase_p95"])
        mean.append(r["null_phase_mean"])
    ax.bar(x, p95, color="lightgrey", label="phase-randomised null 95th pct")
    ax.plot(x, mean, "k_", ms=14, label="phase-randomised null mean")
    ax.plot(x, obs, "ro", label="observed max r")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{k}\n{roi}" for k, roi in cfgs], fontsize=7)
    ax.set_ylabel("max Pearson r")
    ax.legend(fontsize=8)
    ax.set_title("Observed protocol correlation vs. the valid (phase-randomised) null")
    fig.tight_layout()
    out = os.path.join(FIG, "fig_nulls.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("wrote", out)


def fig_channels():
    import d06_channels as d6
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0),
                             gridspec_kw={"width_ratios": [1.25, 1.25, 1]})
    for ax, key in zip(axes[:2], ("session_a", "session_b")):
        C, xf, fs, d = d6.corr_matrix(key)
        lab = d6.split_ward(C)
        order = np.argsort(lab, kind="stable")
        im = ax.imshow(C[np.ix_(order, order)], vmin=-1, vmax=1, cmap="RdBu_r")
        # cluster divider
        ncl = np.sum(lab[order] == lab[order][0])
        ax.axhline(ncl - 0.5, color="k", lw=1.2)
        ax.axvline(ncl - 0.5, color="k", lw=1.2)
        ax.set_title(f"{LABEL.get(key, key)}: HbO correlation, Ward k=2\n(split after channel index "
                     f"{order[ncl-1] if ncl>0 else '?'})", fontsize=9)
        ax.set_xticks(range(16))
        ax.set_yticks(range(16))
        ax.set_xticklabels([common.PAIRS[common.LONG[i]] for i in order], rotation=90, fontsize=5)
        ax.set_yticklabels([common.PAIRS[common.LONG[i]] for i in order], fontsize=5)
        fig.colorbar(im, ax=ax, fraction=0.046)
    ax = axes[2]
    for key, col in (("session_a", "C0"), ("session_b", "C3")):
        name = LABEL.get(key, key)
        ssc = d6.ssc_coupling(key)
        ax.plot(range(16), ssc, "o-", color=col, label=name, ms=4)
    ax.axvline(7.5, color="k", ls="--", lw=1)
    ax.set_xlabel("long-channel index (0-7 = receivers 1-4, 8-15 = receivers 5-8)")
    ax.set_ylabel("|corr| with SSC channels")
    ax.set_title("SSC adjacency", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(FIG, "fig_channels.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    fig_detectability()
    fig_nulls()
    fig_channels()
