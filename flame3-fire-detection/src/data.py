"""Index, load, clean, normalize, split and cache the FLAME 3 thermal TIFFs."""
import os
import numpy as np
import cv2
import tifffile as tiff
from . import config


def build_index(data_dir):
    """Return [(tiff_path, label)] with No Fire = 0, Fire = 1, in sorted (flight) order."""
    root = os.path.join(data_dir, "FLAME 3 CV Dataset")
    records = []
    for cls, label in [("No Fire", 0), ("Fire", 1)]:
        d = os.path.join(root, cls, "Thermal", "Celsius TIFF")
        for f in sorted(os.listdir(d)):
            if f.lower().endswith((".tif", ".tiff")):
                records.append((os.path.join(d, f), label))
    return records


def load(path):
    """Celsius TIFF -> float32 array (512, 640) in degrees C."""
    return tiff.imread(path).astype(np.float32)


def clean(t):
    """Replace NaN / inf pixels with the frame median."""
    bad = ~np.isfinite(t)
    if bad.any():
        t = t.copy()
        t[bad] = np.median(t[~bad])
    return t


def normalize(t):
    """Clip to [T_MIN, T_MAX] and scale to [0, 1] with a fixed global range."""
    t = np.clip(t, config.T_MIN, config.T_MAX)
    return (t - config.T_MIN) / (config.T_MAX - config.T_MIN)


def prep(t):
    """Clean, normalize, resize, add channel axis -> (H, W, 1)."""
    x = normalize(clean(t))
    x = cv2.resize(x, (config.IMG_W, config.IMG_H), interpolation=cv2.INTER_AREA)
    return x[..., None]


def make_mask(t, thr=config.THRESHOLD_C):
    """Threshold mask from raw degrees C (for future segmentation work)."""
    m = (clean(t) > thr).astype(np.float32)
    m = cv2.resize(m, (config.IMG_W, config.IMG_H), interpolation=cv2.INTER_NEAREST)
    return m[..., None]


def _block_split(recs, tr=0.7, va=0.15):
    n = len(recs)
    a, b = int(n * tr), int(n * (tr + va))
    return recs[:a], recs[a:b], recs[b:]


def split_records(records):
    """Contiguous-block split per class (avoids near-duplicate frame leakage)."""
    fire = [r for r in records if r[1] == 1]
    nofire = [r for r in records if r[1] == 0]
    f_tr, f_va, f_te = _block_split(fire)
    n_tr, n_va, n_te = _block_split(nofire)
    return f_tr + n_tr, f_va + n_va, f_te + n_te


def build_arrays(recs):
    X = np.stack([prep(load(p)) for p, _ in recs]).astype(np.float32)
    y = np.array([l for _, l in recs], dtype=np.float32)
    M = np.stack([make_mask(load(p)) for p, _ in recs]).astype(np.float32)
    return X, y, M


def load_or_build_cache(splits, cache_dir):
    """splits: {"train": recs, "val": recs, "test": recs} -> {name: (X, y, M)}."""
    os.makedirs(cache_dir, exist_ok=True)
    out = {}
    for name, recs in splits.items():
        f = os.path.join(cache_dir, f"{name}.npz")
        if not os.path.exists(f):
            X, y, M = build_arrays(recs)
            np.savez_compressed(f, X=X, y=y, M=M)
        d = np.load(f)
        out[name] = (d["X"], d["y"], d["M"])
    return out
