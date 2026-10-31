# Paired visible-target close-up experiment

The close-up ensemble did not improve the target outcome. Diagnostic macro-F1 fell from 0.610370 to 0.580358; multifamily recall stayed at 3/14 with the same 11 misses, while false positives increased from 15 to 21. Weighted validation macro-F1 also fell, from 0.925022 to 0.902005. Retain C as the research baseline; do not promote this close-up candidate.

Fixed uses the frozen C two-seed CNN. Close-up changes only the image/mask view, with the same source pixels, labels, split, initial weights, sampling draws and training budget. All four geometry features retain their original augmented-mask values. Each fit uses six epochs and 1,536 optimizer updates; checkpoint selection includes epoch zero and uses only the original weighted validation cohort.

|Model|LA diagnostic macro-F1|Multifamily TP|FP|FN|Precision|Recall|
|---|---:|---:|---:|---:|---:|---:|
|fixed_42|0.570692|2|14|12|0.1250|0.1429|
|fixed_43|0.568049|3|26|11|0.1034|0.2143|
|fixed|0.610370|3|15|11|0.1667|0.2143|
|closeup_42|0.575965|3|20|11|0.1304|0.2143|
|closeup_43|0.550620|3|31|11|0.0882|0.2143|
|closeup|0.580358|3|21|11|0.1250|0.2143|

|Model|Weighted validation macro-F1|Weighted validation multifamily F1|
|---|---:|---:|
|fixed_42|0.870473|0.818182|
|fixed_43|0.918572|0.866667|
|fixed|0.925022|0.864629|
|closeup_42|0.889406|0.811060|
|closeup_43|0.860428|0.792000|
|closeup|0.902005|0.839286|

Paired spatial-cell bootstrap, 2,000 draws, conditional on these fitted checkpoints (does not include training variability):

{
  "macro_f1": {
    "mean": -0.030605056812554963,
    "percentile95": [
      -0.073069900528696,
      0.002399544122151487
    ],
    "n": 2000
  },
  "multifamily_recall": {
    "mean": 0.0,
    "percentile95": [
      0.0,
      0.0
    ],
    "n": 2000
  }
}

Input audit: 18,480 records (14,710 training, 438 validation, 3,332 diagnostic), 36,960 original file hashes checked. Fixed-view tensors match the prior runtime; augmented geometry is identical across views. Every visible target-mask pixel is retained. 3,204 diagnostic inputs zoom; 121 edge-touching masks retain the original crop. Larger non-edge targets can also retain the original crop.

This is a same-pixel scale/context experiment, not recovery of complete roofs from wider source imagery. It jointly enlarges the visible target and reduces surrounding context. It cannot identify which of those changes causes a result. Original fixed windows have different physical widths across source domains; the minimum close-up width is 40 m.

Seed directions are mixed, and the paired macro-F1 interval includes zero; this is no demonstrated improvement, not proof of universal degradation. The conditional context-extension run was not started because the close-up gate did not show a benefit. This does not rule out a separately designed joint-view model, higher native resolution, or full-roof source recovery. The next exposure prerequisite is independently verified per-building unit supervision, with a separate spatial test set. The prepared 51-case unit review package supplies evidence and review fields, not verified unit truth.

The fixed fits are reused, with source/runtime parity and exact sampling-draw checks, rather than rerun. LA has already been used for development; the cohort has only 14 multifamily references. No unseen-city generalization, dwelling-unit error, observed population, updated population head, or damage-model benefit is measured. Reserved LA inputs remain untouched. No production model is promoted.

Training seconds (fixed42/43, close-up42/43): [178.8, 178.3, 177.0, 175.0]. GPU peak reserved GiB: [2.52, 2.52, 2.52, 2.52].
