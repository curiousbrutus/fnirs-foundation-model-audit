"""Align the channel midpoints of each preprocessing variant to standard 10-05 space.

Requires MtReNIRS's alignment package (clone microsoft/MtReNIRS into third_party/):
    git clone https://github.com/microsoft/MtReNIRS third_party/MtReNIRS
Then:  python3 code/audit/align_variants.py
Writes data/luke_<variant>_aligned.npz with ch_pos_1005 and a validity mask.
"""
import os
import sys

import numpy as np

sys.path.insert(0, "third_party/MtReNIRS/alignment/src")
from mtrenirs.alignment import align  # noqa: E402

VARIANTS = ["raw", "long", "wss"]
for v in VARIANTS:
    d = np.load(f"data/luke_{v}.npz", allow_pickle=True)
    pos = d["ch_pos"]
    r = align(positions=pos, names=[f"ch{i}" for i in range(len(pos))],
              unit="m", modality="fnirs")
    out = f"data/luke_{v}_aligned.npz"
    np.savez(out, ch_pos_1005=r.ch_pos_1005, valid=r.valid)
    print(f"{v}: {int(np.sum(r.valid))}/{len(pos)} channels mapped "
          f"(median projection {np.nanmedian(r.distances)*1000:.1f} mm) -> {out}")
