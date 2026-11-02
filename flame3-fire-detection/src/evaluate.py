"""Evaluate on the test split, compare with the 80 C rule, save figures.
Usage: python -m src.evaluate --data_dir <dataset folder>"""
import argparse
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, ConfusionMatrixDisplay)
from . import config
from .data import build_index, split_records, load_or_build_cache, load


def rgb_for(tiff_path):
    """Matching RGB frame (same number, .JPG) in RGB/Corrected FOV."""
    cls_dir = os.path.dirname(os.path.dirname(os.path.dirname(tiff_path)))
    stem = os.path.splitext(os.path.basename(tiff_path))[0]
    hits = glob.glob(os.path.join(cls_dir, "RGB", "Corrected FOV", stem + ".*"))
    return Image.open(hits[0]) if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--cache_dir", default="cache")
    ap.add_argument("--out_dir", default="outputs")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    records = build_index(args.data_dir)
    train, val, test = split_records(records)
    data = load_or_build_cache({"train": train, "val": val, "test": test}, args.cache_dir)
    Xte, y_true, _ = data["test"]
    y_true = y_true.astype(int)

    model = tf.keras.models.load_model(os.path.join(args.out_dir, "model.keras"))
    probs = model.predict(Xte, batch_size=config.BATCH).ravel()
    y_pred = (probs > 0.5).astype(int)

    max_t = np.array([load(p).max() for p, _ in test])
    y_rule = (max_t > config.THRESHOLD_C).astype(int)

    print(classification_report(y_true, y_pred, target_names=["No Fire", "Fire"]))
    print(confusion_matrix(y_true, y_pred))
    print(f"Rule accuracy: {accuracy_score(y_true, y_rule):.4f}")
    print(f"CNN accuracy:  {accuracy_score(y_true, y_pred):.4f}")

    # --- confusion matrices: CNN vs rule ---
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.5))
    for a, pred, title in zip(ax, [y_pred, y_rule], ["CNN", f"{config.THRESHOLD_C:.0f}°C rule"]):
        ConfusionMatrixDisplay.from_predictions(
            y_true, pred, display_labels=["No Fire", "Fire"], ax=a, colorbar=False)
        a.set_title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "confusion_matrices.png"), dpi=150)
    plt.show()

    # --- samples: every error first, then 3 Fire + 3 No Fire ---
    rng = np.random.default_rng(0)
    miss = list(np.where(y_true != y_pred)[0])
    fire = [i for i in rng.permutation(np.where(y_true == 1)[0]) if i not in miss][:3]
    nofi = [i for i in rng.permutation(np.where(y_true == 0)[0]) if i not in miss][:3]
    sel = miss + fire + nofi

    per_row = 2
    rows = int(np.ceil(len(sel) / per_row))
    fig, axes = plt.subplots(rows, per_row * 2, figsize=(6 * per_row, 3.2 * rows), squeeze=False)
    for a in axes.ravel():
        a.axis("off")
    for n, i in enumerate(sel):
        r, c = divmod(n, per_row)
        a_rgb, a_th = axes[r, 2 * c], axes[r, 2 * c + 1]
        rgb = rgb_for(test[i][0])
        if rgb is None:
            a_rgb.text(0.5, 0.5, "RGB not found", ha="center", va="center")
        else:
            a_rgb.imshow(rgb)
        a_rgb.set_title("RGB (what a person sees)", fontsize=8)
        a_th.imshow(Xte[i, ..., 0], cmap="inferno", vmin=0, vmax=1)
        ok = y_true[i] == y_pred[i]
        a_th.set_title(
            f"True: {'Fire' if y_true[i] else 'No Fire'} | Pred: {'Fire' if y_pred[i] else 'No Fire'}\n"
            f"p={probs[i]:.2f} | max {max_t[i]:.0f}°C",
            color="green" if ok else "red", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "sample_predictions.png"), dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
