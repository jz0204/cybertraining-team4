# Stage 2A native-three-class LA run — for Xihan

Run the frozen new Stage 2A model on the existing LA v3 pre-fire crops and SAM 3
masks. This produces building-type probabilities and a population exposure proxy
once per building. No new training, Stage 1 run, or Stage 2B run is needed.

## 1. Files and environment

**Distribution status:** inference code and this guide are available in the fork.
The archives are prepared locally; public Release attachment upload is pending
the owner's approval. The Release link below will become available after publication.

The code is committed in
[jz0204/cybertraining-team4](https://github.com/jz0204/cybertraining-team4/tree/main/handoff/stage2a_la).
Download the two archives and `SHA256SUMS` from the
[Stage 2A release](https://github.com/jz0204/cybertraining-team4/releases/tag/stage2a-la-v1).
The archives extract into the same `stage2a_la/` directory:

- `stage2a_la_code.tar.gz`: this guide, runner, original inference implementation,
  dependencies, provenance and model checksums; suitable for a Git branch.
- `stage2a_la_weights.tar.gz`: five ConvNeXt-Tiny checkpoints (seeds 42, 123, 2025,
  3407, 7777), the residual-HGB population pickle and its training-validation
  summary. Weights are Release attachments and are ignored by Git.

From the folder containing both downloaded archives and `SHA256SUMS`:

```bash
sha256sum -c SHA256SUMS
tar -xzf stage2a_la_code.tar.gz
tar -xzf stage2a_la_weights.tar.gz
cd stage2a_la
```

If you prefer cloning the fork, clone
`https://github.com/jz0204/cybertraining-team4.git`, download the weights archive
into `cybertraining-team4/handoff/`, and extract it there. Then run from
`cybertraining-team4/handoff/stage2a_la/`. The code is already in the clone; weights
must still be downloaded from the Release.

Use a separate environment to avoid changing the existing SAM 3 environment:

```bash
conda create -n stage2a_la python=3.11 pip -y
conda activate stage2a_la
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python run_la.py --verify-only
```

The CUDA pair above is the pair used in the existing deployment runtime; confirm
the server driver supports CUDA 12.6. A driver-compatible matching PyTorch /
torchvision pair can be used instead and will be recorded in the run manifest.
The local CPU check used torch 2.0.0 / torchvision 0.15.1; see `VALIDATION.md`.
**Keep scikit-learn at 1.9.0:** that is the population artifact's serialized version.
Inference loads all weights locally and does not download pretrained backbones.

## 2. Inputs already on the server

Set these two absolute paths. The first is from `COLLABORATION.md`; update it if
the v3 directory has moved. The second can use the committed product from the
repository or its identical server copy.

```bash
export LA_STAGE2A_RUN_ROOT=/media/data/building_instance_tamu/la_fire_2025/stage2_damage/multidate_full_run_v3
export LA_STAGE2A_PRODUCT="$LA_STAGE2A_RUN_ROOT/building_damage_all_cells.csv"
```

If the server CSV is elsewhere, use the committed
`cybertraining-team4/results/final_product/building_damage_all_cells.csv` instead.
The product SHA-256 in `SOURCE.json` identifies the exact local v3 copy checked
during packaging; compare it if the server copy may have changed.

For each product `(cell_id, bldg_uid)`, the runner reads:

```text
<run-root>/<cell_id>/shared_base/shared_instance_samples.csv
<run-root>/<cell_id>/shared_base/crops_pre/<pre_crop filename>
<run-root>/<cell_id>/shared_base/masks_M/<mask_M filename>
```

The shared CSV must have `bldg_uid`, `pre_crop` and `mask_M`. The runner uses the
existing aligned 256×256 RGB crop and binary 0/255 mask. The model resizes both
to 224×224, normalizes RGB with the original ImageNet mean/std and derives mask
geometry using the original implementation. Crops must retain their existing
pixel scale; do not resize them before supplying them to the runner.

`footprint_m2` comes from the **full** `polygon_wkt_utm` in the product CSV, whose
coordinates are in EPSG:32611 (metres), and `log_footprint_m2 = log1p(area)`.
There is no training-data pixel-area fallback and no area estimate from a
potentially clipped mask. These are SAM 3 footprint areas, not ground-truth areas.

Only product rows are selected: the committed v3 CSV has 22,024 rows in 126 cells.
The runner preserves the original `cell_id` and `bldg_uid` and uses a composite
`building_uid` internally to prevent accidental collisions across cells.
Old product `stage2a_type` / `stage2a_population` values are not model inputs.
Post-fire quality flags do not define this run's cohort: Stage 2A uses pre-fire
imagery, including buildings with no valid post-fire coverage.

Absolute paths left over from an older run are resolved to the selected v3
`shared_base` directory first. For a different layout, add an explicit prefix map:

```bash
# Append to a command below, substituting the actual prefixes:
--path-map /old/absolute/prefix=/new/absolute/prefix
```

## 3. Smoke run, then full run

From `stage2a_la/`, first run 100 product rows on one GPU:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python run_la.py \
  --run-root "$LA_STAGE2A_RUN_ROOT" \
  --product-csv "$LA_STAGE2A_PRODUCT" \
  --out-dir outputs/la_v3_smoke_100 \
  --limit 100 --batch-size 32 --num-workers 4 --device cuda
```

Check `input_audit.json`, `prediction_summary.json`, and visually inspect several
crop/mask pairs named in `stage2a_input.csv`. Automated checks validate file
existence, image sizes, mask values, positive polygon area, and output ID/probability
consistency; spatial alignment and radiometric suitability still need visual inspection.
The smoke cohort is the first 100 product rows and is not an accuracy sample.

If the smoke run passes:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python run_la.py \
  --run-root "$LA_STAGE2A_RUN_ROOT" \
  --product-csv "$LA_STAGE2A_PRODUCT" \
  --out-dir outputs/la_v3_full \
  --batch-size 64 --num-workers 4 --device cuda
```

Five ensemble members run sequentially on one GPU. If GPU memory is insufficient,
use a smaller batch size and a new output directory. A CPU run is available with
`--device cpu --num-workers 0`, but the full LA run is intended for the A6000.
To follow inference progress in a second terminal:

```bash
tail -f outputs/la_v3_full/inference.log
```

Output directories must be new; the runner refuses to overwrite an existing run.
Malformed/missing inputs, empty masks and entirely zero pre-fire crops are listed
in `excluded_inputs.csv` and stop inference by default. Fix the inputs and rerun
into a new directory. If the excluded buildings are intentionally unavailable,
rerun with `--allow-exclusions` and report the excluded count and reasons; do not
describe that result as covering all 22,024 buildings.
Partly zero pre-fire crops are retained and flagged by `pre_zero_fraction`.
Use `--prepare-only` with a new output directory to inspect inputs without inference.

## 4. Return these files

Please return the whole full-run output folder (small relative to the imagery):

- `stage2a_predictions.csv`: final predictions with `cell_id`, `bldg_uid`,
  `pred_type_class`, three `prob_*` columns, `pred_population`, footprint area,
  pre-fire zero fraction and member diagnostics. Join back to the product using
  **both `cell_id` and `bldg_uid`**.
- `stage2a_input.csv`, `input_audit.json`, `excluded_inputs.csv`: cohort and source paths.
- `prediction_summary.json`: class counts and population quantiles, without accuracy metrics.
- `run_manifest.json`, `inference.log`, `raw_predictions.csv`: environment, command,
  source-code/weight checksums and inference record. Success is `status: complete`.

These outputs are separate from the existing v3 product and its old Stage 2A columns.
Please also report any input-layout changes or visually suspicious crop/mask pairs.

## 5. Model interpretation

The model has three classes: `residential_small`, `residential_multi`, `commercial`.
It was trained after dropping `institutional` and `other`; it has no unknown-class
head and will force any supplied building into these three classes. Predictions
outside that taxonomy should not be interpreted as validated building-use labels.

`pred_population` predicts the training dataset's estimated-population exposure
proxy, including a commercial-building proxy. It is not measured occupancy or a
count of residents. Do not sum every class to report residents affected by the fire.
Ensemble member spread is a diagnostic, not a calibrated confidence interval.

This run applies the frozen model to LA without fine-tuning or temperature scaling.
Assessor use labels, DINS labels and Census data are not needed to run it. Building
type accuracy requires a separate matched assessor-label evaluation; actual
building population accuracy requires independent population reference data.
