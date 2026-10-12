# Handoff validation

Checked locally on 2026-10-11 (America/Chicago).

- All seven packaged model files pass their recorded SHA-256 checks: five native
  three-class checkpoints, one population model, and its summary (568,303,926 bytes).
- Eight input/output contract tests pass: area from full polygons, holes and
  multipolygons, unique IDs across cells, invalid geometry reporting, duplicate
  product rejection, incorrect mask scale, nodata crops, explicit path-prefix
  mapping, and prediction-to-source ID recovery.
- The committed LA v3 product contains 22,024 unique `(cell_id, bldg_uid)` pairs in
  126 cells; all 22,024 EPSG:32611 footprint polygons are valid with positive area.
- A CPU run through `run_la.py` loaded all five checkpoints and the population
  model, produced two prediction rows, and passed output identity/probability/
  population checks. Both predictions have all three class probabilities and the
  original building keys. The resulting run manifest has `status: complete`.
- Python compilation and whitespace checks pass. Copied inference files match
  the source hashes in `SOURCE.json`.

The two-row execution check regenerated 256×256 crops and masks from the local
April LA demo `II_package/la_fire_results/demo_cell_00507/cell_00507_pre.tif`,
using rows 2 and 3 (zero-based) of its existing 20250110 shared table and the
GeoTIFF transform to create metric footprints. Building IDs:

```text
fc77ca51-6f5d-45a7-af9d-66f98239a258  footprint 149.642209 m²
af6fde37-1115-424b-a689-cc2a4738d5c8  footprint 321.857495 m²
```

This verifies execution using actual LA imagery; it is not a run on the remote
v3 crop tree, nor a type/population accuracy evaluation. The remote A6000/CUDA
environment and all 22,024 crop/mask pairs remain to be checked by Xihan's
100-row smoke run and full run.

The local CPU environment used Python 3.11.9, torch 2.0.0, torchvision 0.15.1,
NumPy 1.26.4, Pillow 11.3.0, scikit-learn 1.9.0, SciPy 1.15.3 and Shapely 2.0.2.
The handoff pins Shapely 2.1.2 to match the existing deployment runtime.

To repeat the contract tests from this package directory:

```bash
python -m unittest discover -s tests -v
python run_la.py --verify-only
```
