# Artifact manifest

- EXPERIMENT_PLAN.20261029.md and EXPERIMENT_PLAN.md: fixed pretraining protocol.
- REVIEW_SOURCE.json: explicit per-parcel visual decisions; source of frozen roles.
- REVIEW.md, BUILDING_REVIEW.csv, contact_sheets/:31-parcel/62-outline review evidence.
- DATA_AUDIT.json: additions, unchanged cohort hashes and spatial isolation checks.
- EXPERIMENT_TRACKER.md: execution state.
- RESULTS.json, METRICS.csv, MINORITY_COMPARISON.csv, PREDICTIONS.json: produced after both seeds finish.
- External la_fourplex_20261029/: raw official queries,62 image sets, frozen dataset/protocol and execution/checkpoints.
- Reproduce: prepare_la_fourplex.py freeze (once before freezing); analyze_la_fourplex.py after collecting both fits.
- REPORT.md: completed analysis and next-step recommendation.
- SOURCE_SHA256.json and VERIFICATION.json: code/review/results provenance and completed checks.
- CHANGED_PREDICTIONS.json: all16 changed predictions versus ordinary LA CNN.
