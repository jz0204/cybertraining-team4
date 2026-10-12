#!/usr/bin/env python3
"""Run frozen native-three-class Stage 2A on the existing LA v3 product."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
from shapely import wkt
from shapely.errors import GEOSException

PACKAGE_ROOT = Path(__file__).resolve().parent
CLASS_NAMES = ["residential_small", "residential_multi", "commercial"]
INPUT_FIELDS = [
    "building_uid", "cell_id", "bldg_uid", "crop_path", "mask_path",
    "footprint_m2", "log_footprint_m2", "footprint_m2_source", "tile_id",
    "event_id", "hazard_type", "sam3_confidence", "pre_zero_fraction",
]


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def write_csv(path, fields, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def footprint_area(text):
    geometry = wkt.loads(text)
    if geometry.geom_type not in {"Polygon", "MultiPolygon"}:
        raise ValueError("Footprint must be a Polygon or MultiPolygon")
    area = geometry.area
    if geometry.is_empty or not geometry.is_valid or not math.isfinite(area) or area <= 0:
        raise ValueError("Footprint must be valid and have positive finite area")
    return float(area)


def parse_path_maps(values):
    maps = []
    for value in values:
        old, separator, new = value.partition("=")
        if not separator or not old or not new:
            raise ValueError("--path-map must be OLD_PREFIX=NEW_PREFIX")
        maps.append((Path(old).expanduser(), Path(new).expanduser().resolve()))
    return sorted(maps, key=lambda pair: len(str(pair[0])), reverse=True)


def resolve_asset(value, shared_csv, run_root, folder, path_maps):
    if not value:
        raise ValueError(f"Empty {folder} path")
    original = Path(value).expanduser()
    # Prefer the selected v3 tree, even if an old v2 absolute path still exists.
    canonical = shared_csv.parent / folder / original.name
    if canonical.is_file():
        return canonical.resolve()
    for old, new in path_maps:
        try:
            mapped = new / original.relative_to(old)
        except ValueError:
            continue
        if mapped.is_file():
            return mapped.resolve()
    candidate = original if original.is_absolute() else shared_csv.parent / original
    candidate = candidate.resolve()
    if candidate.is_relative_to(run_root) and candidate.is_file():
        return candidate
    raise FileNotFoundError(
        f"Cannot find {value!r} in the selected run tree. Expected {canonical}; "
        "use --path-map for nonstandard locations."
    )


def validate_images(crop_path, mask_path):
    with Image.open(crop_path) as crop, Image.open(mask_path) as mask:
        if crop.size != (256, 256) or mask.size != crop.size:
            raise ValueError("Expected aligned 256x256 RGB crop and mask")
        if crop.mode != "RGB":
            raise ValueError(f"Expected RGB PNG crop, found mode {crop.mode}")
        if mask.mode not in {"L", "1"}:
            raise ValueError(f"Expected single-channel binary mask, found mode {mask.mode}")
        rgb = np.asarray(crop)
        values = np.asarray(mask.convert("L"))
    unique = set(np.unique(values).tolist())
    if not unique <= {0, 255} or not np.any(values):
        raise ValueError("Mask must contain foreground and use 0/255 values")
    zero_fraction = float(np.mean(np.all(rgb == 0, axis=2)))
    if zero_fraction == 1.0:
        raise ValueError("Pre-fire crop is entirely zero/nodata")
    return zero_fraction


def prepare_inputs(product_csv, run_root, out_dir, limit=0, path_maps=()):
    product = read_csv(product_csv)
    if not product:
        raise ValueError("Product CSV is empty")
    required = {"cell_id", "bldg_uid", "polygon_wkt_utm"}
    if not required <= product[0].keys():
        raise ValueError(f"Product CSV requires {sorted(required)}")
    keys = [(row["cell_id"], row["bldg_uid"]) for row in product]
    if len(keys) != len(set(keys)) or any(not all(key) for key in keys):
        raise ValueError("Product has empty or duplicate (cell_id, bldg_uid) keys")
    selected = product[:limit] if limit else product
    shared_tables = {}
    source_hashes = {}
    inputs, excluded = [], []
    for row in selected:
        cell_id, uid = row["cell_id"], row["bldg_uid"]
        try:
            if Path(cell_id).name != cell_id or cell_id in {".", ".."}:
                raise ValueError("Invalid cell_id")
            shared_csv = run_root / cell_id / "shared_base/shared_instance_samples.csv"
            if cell_id not in shared_tables:
                shared_rows = read_csv(shared_csv)
                shared = {item["bldg_uid"]: item for item in shared_rows}
                if len(shared) != len(shared_rows):
                    raise ValueError(f"Duplicate bldg_uid in {shared_csv}")
                shared_tables[cell_id] = shared
                source_hashes[str(shared_csv)] = sha256(shared_csv)
            shared = shared_tables[cell_id][uid]
            crop = resolve_asset(shared["pre_crop"], shared_csv, run_root, "crops_pre", path_maps)
            mask = resolve_asset(shared["mask_M"], shared_csv, run_root, "masks_M", path_maps)
            zero_fraction = validate_images(crop, mask)
            area = footprint_area(row["polygon_wkt_utm"])
            inputs.append({
                "building_uid": json.dumps([cell_id, uid], separators=(",", ":")),
                "cell_id": cell_id, "bldg_uid": uid,
                "crop_path": str(crop), "mask_path": str(mask),
                "footprint_m2": area, "log_footprint_m2": math.log1p(area),
                "footprint_m2_source": "LA_v3_polygon_wkt_utm_EPSG32611",
                "tile_id": cell_id, "event_id": "la_fire_2025", "hazard_type": "fire",
                "sam3_confidence": row.get("sam3_confidence", ""),
                "pre_zero_fraction": zero_fraction,
            })
        except (KeyError, ValueError, OSError, GEOSException) as error:
            excluded.append({"cell_id": cell_id, "bldg_uid": uid, "reason": str(error)})
        if (len(inputs) + len(excluded)) % 1000 == 0:
            print(f"[prepare] valid={len(inputs)} excluded={len(excluded)}", flush=True)
    write_csv(out_dir / "stage2a_input.csv", INPUT_FIELDS, inputs)
    write_csv(out_dir / "excluded_inputs.csv", ["cell_id", "bldg_uid", "reason"], excluded)
    audit = {
        "product_csv": str(product_csv), "product_sha256": sha256(product_csv),
        "run_root": str(run_root), "product_rows": len(product), "selected_rows": len(selected),
        "valid_rows": len(inputs), "excluded_rows": len(excluded),
        "limit": limit, "shared_csv_sha256": source_hashes,
        "area_crs": "EPSG:32611", "area_source": "product polygon_wkt_utm",
        "path_maps": [[str(old), str(new)] for old, new in path_maps],
        "partly_zero_crops": sum(row["pre_zero_fraction"] > 0 for row in inputs),
    }
    (out_dir / "input_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return inputs, audit


def verify_weights(weights_dir):
    manifest = json.loads((PACKAGE_ROOT / "MODEL_ASSETS.json").read_text())
    for item in manifest["files"]:
        path = weights_dir / item["path"]
        if not path.is_file() or path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise ValueError(f"Missing or altered model artifact: {path}")
    return manifest


def finalize_predictions(raw_csv, inputs, output_csv):
    raw = read_csv(raw_csv)
    expected = {row["building_uid"]: row for row in inputs}
    ids = [row["building_uid"] for row in raw]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError("Prediction IDs do not match prepared inputs exactly")
    for row in raw:
        source = expected[row["building_uid"]]
        probs = [float(row[f"prob_{name}"]) for name in CLASS_NAMES]
        population = float(row["pred_population"])
        if not all(math.isfinite(p) and 0 <= p <= 1 for p in probs) or not math.isclose(sum(probs), 1.0, abs_tol=1e-5):
            raise ValueError("Invalid type probabilities")
        if not math.isfinite(population) or population < 0 or row["pred_type_class"] not in CLASS_NAMES:
            raise ValueError("Invalid population/type prediction")
        row.update({key: source[key] for key in ["cell_id", "bldg_uid", "footprint_m2", "pre_zero_fraction"]})
    fields = ["cell_id", "bldg_uid", "footprint_m2", "pre_zero_fraction", *raw[0].keys()]
    write_csv(output_csv, list(dict.fromkeys(fields)), raw)
    return {
        "prediction_rows": len(raw), "class_counts": dict(Counter(row["pred_type_class"] for row in raw)),
        "population_quantiles": dict(zip(["p0", "p25", "p50", "p75", "p95", "p99", "p100"],
            np.quantile([float(row["pred_population"]) for row in raw], [0, .25, .5, .75, .95, .99, 1]).tolist())),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--product-csv", type=Path, help="LA v3 CSV with EPSG:32611 polygon_wkt_utm")
    parser.add_argument("--out-dir", type=Path, help="New directory; existing paths are never overwritten")
    parser.add_argument("--weights-dir", type=Path, default=PACKAGE_ROOT / "weights")
    parser.add_argument("--limit", type=int, default=0, help="0=all product rows; 100 is recommended for smoke testing")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--path-map", action="append", default=[], help="OLD_PREFIX=NEW_PREFIX (repeatable)")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--verify-only", action="store_true", help="Verify weights without reading LA inputs")
    parser.add_argument("--allow-exclusions", action="store_true", help="Continue with valid rows and report every excluded product row")
    args = parser.parse_args()
    if args.limit < 0 or args.batch_size < 1 or args.num_workers < 0:
        parser.error("limit/workers must be nonnegative and batch size must be positive")
    weights = args.weights_dir.expanduser().resolve()
    if args.verify_only:
        verified = verify_weights(weights)
        print(f"Verified {len(verified['files'])} model artifacts")
        return
    if not all([args.run_root, args.product_csv, args.out_dir]):
        parser.error("--run-root, --product-csv and --out-dir are required for an LA run")
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=False)
    inputs, audit = prepare_inputs(
        args.product_csv.expanduser().resolve(), args.run_root.expanduser().resolve(), out_dir,
        args.limit, parse_path_maps(args.path_map),
    )
    if audit["excluded_rows"] and not args.allow_exclusions:
        raise ValueError(f"{audit['excluded_rows']} invalid inputs; inspect {out_dir / 'excluded_inputs.csv'}. "
                         "Fix the inputs or explicitly use --allow-exclusions with a new output directory.")
    if not inputs:
        raise ValueError("No valid inputs")
    if args.prepare_only:
        print(json.dumps(audit, indent=2))
        return
    manifest = verify_weights(weights)
    import torch
    import sklearn
    if sklearn.__version__ != "1.9.0":
        raise ValueError("The serialized population model requires scikit-learn==1.9.0; see RUN.md")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; check the environment or explicitly use --device cpu")
    ckpts = [str(weights / f"seed{seed}/stage2a_best_model.pt") for seed in [42, 123, 2025, 3407, 7777]]
    command = [
        sys.executable, str(PACKAGE_ROOT / "scripts/stage2a/infer/infer_stage2a_native3_ensemble.py"),
        "--input_csv", str(out_dir / "stage2a_input.csv"), "--ckpts", ",".join(ckpts),
        "--population_model_pkl", str(weights / "population_logfootprint_soft_type_residual_hgb.pkl"),
        "--out_csv", str(out_dir / "raw_predictions.csv"), "--img_size", "224",
        "--batch_size", str(args.batch_size), "--num_workers", str(args.num_workers),
        "--device", args.device, "--print_examples", "0",
    ]
    versions = {name: importlib.metadata.version(name) for name in
                ["torch", "torchvision", "numpy", "pillow", "scikit-learn", "scipy", "shapely"]}
    run_manifest = {
        "started_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
        "command": command, "python": sys.version, "versions": versions,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "torch_cuda_version": torch.version.cuda,
        "device": args.device, "gpu_name": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "weights": manifest, "input_audit": audit,
        "input_sha256": sha256(out_dir / "stage2a_input.csv"),
        "source_snapshot": json.loads((PACKAGE_ROOT / "SOURCE.json").read_text()),
        "package_code_sha256": {str(path.relative_to(PACKAGE_ROOT)): sha256(path)
                                for path in [Path(__file__).resolve(), *sorted((PACKAGE_ROOT / "scripts").rglob("*.py"))]},
    }
    manifest_path = out_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(run_manifest, indent=2) + "\n")
    print(f"[infer] buildings={len(inputs)} device={args.device}; log: {out_dir / 'inference.log'}", flush=True)
    with (out_dir / "inference.log").open("w") as log:
        result = subprocess.run(command, cwd=PACKAGE_ROOT, stdout=log, stderr=subprocess.STDOUT)
    run_manifest.update({"finished_utc": datetime.now(timezone.utc).isoformat(), "returncode": result.returncode})
    if result.returncode:
        run_manifest["status"] = "failed"
        manifest_path.write_text(json.dumps(run_manifest, indent=2) + "\n")
        raise RuntimeError(f"Inference failed; see {out_dir / 'inference.log'}")
    summary = finalize_predictions(out_dir / "raw_predictions.csv", inputs, out_dir / "stage2a_predictions.csv")
    run_manifest.update({"status": "complete", "summary": summary,
                         "predictions_sha256": sha256(out_dir / "stage2a_predictions.csv")})
    manifest_path.write_text(json.dumps(run_manifest, indent=2) + "\n")
    (out_dir / "prediction_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"[done] {out_dir / 'stage2a_predictions.csv'}")


if __name__ == "__main__":
    main()
