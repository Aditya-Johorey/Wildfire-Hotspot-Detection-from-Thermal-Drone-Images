# Wildfire / Hotspot Detection from Thermal Drone Images

A quick prototype that classifies thermal drone frames as **Fire** or **No Fire** using the
[FLAME 3 computer-vision subset (Sycan Marsh)](https://www.kaggle.com/datasets/brycehopkins/flame-3-computer-vision-subset-sycan-marsh).
A MobileNetV2 transfer-learning model is compared against a one-line baseline: `max temperature > 80 °C`.

> **Key finding:** the 80 °C rule scores **100%** on the test split, the CNN scores **99.1%**.
> The two classes do not overlap on per-image max temperature, so this dataset is too easy to show
> that the CNN generalizes. See [Limitations](#limitations).

## Dataset

- 738 radiometric thermal frames (512 × 640, float32, **already in °C**): 622 Fire, 116 No Fire (about 5.4 : 1).
- No Fire max temps: about 7 to 47 °C. Fire max temps: about 108 to 608 °C. There is a gap between ~47 and ~108 °C.
- The sensor appears to saturate near ~600 °C, so values at the top are "at least this hot".
- Filenames repeat across classes, so the index stores full paths with labels.

Expected layout (download from Kaggle):

```
<data_dir>/FLAME 3 CV Dataset/
    Fire/     Thermal/Celsius TIFF/*.TIFF   RGB/Corrected FOV/*.JPG
    No Fire/  Thermal/Celsius TIFF/*.TIFF   RGB/Corrected FOV/*.JPG
```

## Pipeline

1. **File index**: (path, label) pairs in sorted, flight order.
2. **Load** TIFFs as float32 without converting to 8-bit.
3. **Clean** NaN/inf pixels (frame median).
4. **Normalize**: clip to [-20, 600] °C, then scale to [0, 1] with a fixed global range (not per image).
5. **Resize** to 256 × 320 and add a channel axis.
6. **Split** by contiguous blocks per class (70/15/15), since consecutive frames are near-duplicates.
7. **Class weights** for the 5.4 : 1 imbalance (train only).
8. **Augment** (train only): flips and small rotations. No brightness/contrast changes, because pixel values are real temperatures.
9. **Model**: frozen MobileNetV2 (ImageNet), thermal channel repeated 3x and rescaled to [-1, 1], plus a sigmoid head.
10. **Cache** processed arrays to disk so reruns are fast.

## Usage

```bash
pip install -r requirements.txt

python -m src.train    --data_dir /path/to/flame-3-computer-vision-subset-sycan-marsh
python -m src.evaluate --data_dir /path/to/flame-3-computer-vision-subset-sycan-marsh
```

Outputs go to `outputs/` (`model.keras`, `confusion_matrices.png`, `sample_predictions.png`).
Processed arrays are cached in `cache/`. Delete it if you change preprocessing settings.

In Google Colab, mount Drive first and point `--cache_dir` at a Drive folder so the cache survives restarts.

## Results (test split, 112 images)

| Model | Accuracy | Notes |
|---|---|---|
| 80 °C rule | 1.000 | 18/18 No Fire, 94/94 Fire |
| MobileNetV2 | 0.991 | 18/18 No Fire, 93/94 Fire |

The CNN's single error was the dimmest fire frame (max ~108 °C), classified as No Fire. The threshold rule gets it right.

## Limitations

- **Trivially separable data:** a plain threshold already classifies every image, so a high CNN score says little about generalization.
- **Small No Fire sample:** 116 images in total and 18 in the test split, so one error moves its metrics a lot.
- **No hard negatives:** no sun-heated ground, rocks or vehicles in the data, which is where real false alarms come from.
- **Apparent temperatures:** the DJI conversion assumes fixed emissivity (~0.95) and distance, so values over flames are approximate, and small flames can be under-reported.
- **Single location and campaign:** results may not transfer to other terrain or sensors.

## Next steps

- Pixel-level segmentation (`make_mask` in `src/data.py` already builds threshold masks). Prefer the dataset's own fire masks if it ships them.
- Evaluate on hard negatives and on other flights or sites.
- Add a minimum hot-pixel count to the baseline, e.g. `(t > 80).sum() >= 10`.

## Acknowledgements

Data: FLAME 3 computer-vision subset (Sycan Marsh), via Kaggle. Please check the dataset page for its license and citation requirements.
Code is released under the MIT License.
