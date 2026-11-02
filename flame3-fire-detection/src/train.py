"""Train the classifier.  Usage: python -m src.train --data_dir <dataset folder>"""
import argparse
import os
import tensorflow as tf
from . import config
from .data import build_index, split_records, load_or_build_cache
from .model import build_model

AUTOTUNE = tf.data.AUTOTUNE
aug = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.05),
    # no brightness/contrast changes: pixel values encode real temperature
])


def make_ds(X, y, train=False):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if train:
        ds = ds.shuffle(len(X)).map(lambda x, y: (aug(x, training=True), y),
                                    num_parallel_calls=AUTOTUNE)
    return ds.batch(config.BATCH).prefetch(AUTOTUNE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="folder containing 'FLAME 3 CV Dataset'")
    ap.add_argument("--cache_dir", default="cache")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--epochs", type=int, default=15)
    args = ap.parse_args()

    records = build_index(args.data_dir)
    train, val, test = split_records(records)
    data = load_or_build_cache({"train": train, "val": val, "test": test}, args.cache_dir)
    (Xtr, ytr, _), (Xva, yva, _) = data["train"], data["val"]

    n0, n1 = (ytr == 0).sum(), (ytr == 1).sum()
    class_weight = {0: len(ytr) / (2 * n0), 1: len(ytr) / (2 * n1)}
    print("class_weight:", class_weight)

    model = build_model()
    early = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True)
    model.fit(make_ds(Xtr, ytr, train=True), validation_data=make_ds(Xva, yva),
              epochs=args.epochs, class_weight=class_weight, callbacks=[early])

    os.makedirs(args.out_dir, exist_ok=True)
    model.save(os.path.join(args.out_dir, "model.keras"))


if __name__ == "__main__":
    main()
