"""Measure frozen-model sensitivity; interventions are diagnostics, not fixes."""

import csv
import io
import json
import math
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader

from audit_sources import OUT
from audit_training import LEGACY, save_json, quantiles
from scripts.stage2a.common import Stage2aInferenceDataset, collate_stage2a, build_model_from_config, load_torch_checkpoint, geometry_vector
from scripts.stage2a.infer.infer_stage2a_native3_ensemble import _load_population_artifact, _predict_population
from run_la import verify_weights


PACKAGE = Path(__file__).resolve().parent
CLASSES = ["residential_small", "residential_multi", "commercial"]


def population_diagnostics():
    pred = pd.read_csv(PACKAGE / "results/la_v3_full/stage2a_predictions.csv", dtype={"cell_id": str, "bldg_uid": str})
    rows = pred.to_dict("records")
    probs = pred[[f"prob_{c}" for c in CLASSES]].to_numpy()
    artifact = _load_population_artifact(PACKAGE / "weights/population_logfootprint_soft_type_residual_hgb.pkl")
    base = np.maximum(0, np.expm1(_predict_population(artifact, rows, probs)))
    assert np.allclose(base, pred.pred_population, atol=1e-7, rtol=1e-6)
    # Deliberately restore the old training scale only to measure sensitivity, never as a correction.
    scale = json.loads((OUT / "training_summary.json").read_text())["area_scale_ratio"]["p50"]
    inflated = [dict(r, footprint_m2=r["footprint_m2"] * scale) for r in rows]
    cf = np.maximum(0, np.expm1(_predict_population(artifact, inflated, probs)))
    result = pred[["cell_id", "bldg_uid", "footprint_m2", "pred_type_class", "pred_population"]].copy()
    result["diagnostic_population_old_training_area_scale"] = cf
    summary = {"n": len(rows), "baseline_reproduction_max_absolute_error": float(np.max(abs(base - pred.pred_population))), "area_multiplier_diagnostic": scale, "baseline_population_quantiles": quantiles(base), "inflated_area_population_quantiles": quantiles(cf), "baseline_population_sum": float(base.sum()), "inflated_area_population_sum": float(cf.sum()), "note": "Type probabilities held fixed; inflated area is an intentional sensitivity intervention, not a deployment correction or population ground truth."}
    for i, c in enumerate(CLASSES):
        one_hot = np.zeros_like(probs)
        one_hot[:, i] = 1
        v = np.maximum(0, np.expm1(_predict_population(artifact, rows, one_hot)))
        result[f"diagnostic_population_type_forced_{c}"] = v
        summary[f"type_forced_{c}_quantiles"] = quantiles(v)
    result.to_csv(OUT / "la_population_model_sensitivity.csv", index=False)
    save_json("population_model_sensitivity.json", summary)
    print("[population] reproduced all", len(rows), "outputs", flush=True)


def cnn_diagnostics():
    audit = pd.read_csv(OUT / "training_area_label_audit.csv")
    with zipfile.ZipFile(LEGACY / "Harris_CBG_Building_Population.zip") as archive:
        labels = pd.read_csv(archive.open("ml_dataset/labels.csv"))
        selected = pd.concat([labels[labels.type_class_name == c].sort_values("building_uid").sample(n=100, random_state=2025) for c in CLASSES], ignore_index=True)
        selected = selected.merge(audit[["building_uid", "area_equal_area_m2", "counterfactual_type"]], on="building_uid", validate="one_to_one")
        sample_dir = OUT / "model_sample"
        sample_dir.mkdir(exist_ok=True)
        for col in ["crop_path", "mask_path"]:
            local_paths = []
            for i, row in selected.iterrows():
                target = sample_dir / f"{i:03d}_{col}.png"
                target.write_bytes(archive.read(row[col]))
                local_paths.append(str(target))
            selected[col] = local_paths
    selected.to_csv(OUT / "model_sample_manifest.csv", index=False)
    records = selected.to_dict("records")
    member_probs = {"original_area": [], "physical_area": [], "zero_geometry": []}
    config_summaries = []
    torch.set_num_threads(4)
    for seed in [42, 123, 2025, 3407, 7777]:
        checkpoint = load_torch_checkpoint(PACKAGE / f"weights/seed{seed}/stage2a_best_model.pt")
        config = checkpoint["config"]
        assert config["type_geometry_mode"] == "concat" and config["pooling_mode"] == "mask_m"
        cols = config["geometry_cols"].split(",")
        config_summaries.append({"seed": seed, "geometry_cols": cols, "type_geometry_mode": config["type_geometry_mode"], "input_mode": config["input_mode"]})
        model = build_model_from_config(config, pretrained=False)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        ds = Stage2aInferenceDataset(records, img_size=224, input_mode=config["input_mode"], geometry_cols=cols, require_ring_mask=False)
        loader = DataLoader(ds, batch_size=8, num_workers=0, collate_fn=collate_stage2a)
        captured = []
        handle = model.shared_fc.register_forward_hook(lambda module, args, output: captured.append(output))
        variant_probs = {k: [] for k in member_probs}
        with torch.inference_mode():
            start = 0
            for batch in loader:
                captured.clear()
                output = model(batch["x"], geometry=batch["geometry"], mask=batch["mask"], return_dict=True)
                z = captured[0]
                n = len(z)
                actual_geom = batch["geometry"].clone()
                area_index = cols.index("log_footprint_m2")
                actual_geom[:, area_index] = torch.tensor(np.log1p(selected.area_equal_area_m2.iloc[start:start + n].to_numpy()), dtype=actual_geom.dtype)
                old_logits = model.type_head(torch.cat([z, model.geometry_proj(batch["geometry"].float())], dim=1))
                assert torch.allclose(old_logits, output["type_logits"], atol=1e-6)
                variant_probs["original_area"].append(output["type_probs"].numpy())
                for name, geometry in [("physical_area", actual_geom), ("zero_geometry", torch.zeros_like(actual_geom))]:
                    logits = model.type_head(torch.cat([z, model.geometry_proj(geometry)], dim=1))
                    variant_probs[name].append(torch.softmax(logits, dim=1).numpy())
                start += n
        handle.remove()
        for name in member_probs:
            member_probs[name].append(np.concatenate(variant_probs[name]))
        print("[cnn] completed seed", seed, flush=True)
    summary = {"n": len(records), "sampling": "100 per archived active class, deterministic random_state=2025; diagnostic training-package sample, not held-out accuracy", "checkpoint_configs": config_summaries, "intervention": "Same RGB, same mask, same extracted image features; replace only numeric log area, or zero all geometry inputs. No model training."}
    original = np.mean(member_probs["original_area"], axis=0)
    for name in member_probs:
        probs = np.mean(member_probs[name], axis=0)
        predicted = np.array(CLASSES)[probs.argmax(axis=1)]
        selected[f"prediction_{name}"] = predicted
        for i, c in enumerate(CLASSES):
            selected[f"{name}_prob_{c}"] = probs[:, i]
        summary[name] = {"class_counts": pd.Series(predicted).value_counts().to_dict(), "argmax_changed_vs_original": int((probs.argmax(axis=1) != original.argmax(axis=1)).sum()), "mean_probability_L1_change": float(np.abs(probs - original).sum(axis=1).mean()), "agreement_with_archived_proxy_labels": float((predicted == selected.type_class_name).mean()), "agreement_with_area_counterfactual_labels": float((predicted == selected.counterfactual_type).mean())}
    selected.to_csv(OUT / "cnn_area_sensitivity_predictions.csv", index=False)
    save_json("cnn_area_sensitivity.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    verify_weights(PACKAGE / "weights")
    population_diagnostics()
    cnn_diagnostics()
