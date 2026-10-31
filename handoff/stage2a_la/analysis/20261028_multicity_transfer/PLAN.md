# NYC-only versus multicity DINO data preparation

## Scope and status

This prepares data only. The parent task owns LA scripts, inference, all GPU
orchestration, and any eventual model training. No training, service operations,
final-city acquisition, Git commits, or edits to the old experiment are included.

Owned outputs are `prepare_multicity_transfer.py`, this plan, and
`/Volumes/Crucial X9/Instance-Impact-IGUIDE-SpatialAIChallenge2026/II_package/outputs/stage2a/multicity_transfer_20261028`.
The final `ACQUISITION_REPORT.json` is authoritative for readiness; a selected
manifest or an image file alone does not establish successful preparation.

Preparation completed at **2026-10-28 05:31:31 UTC** with status
`ready_for_parent_preflight`. All 177 missing image sets were acquired in one
round, with no failures or replacements. All 1,354 unique records passed native
image checks: minimum near-window target retention was 1.0, maximum missing
context fraction was 0.000005, and target missing-pixel fraction was zero.
Actual train/validation context intersections and shared parcels were both zero.
The 15 recorded original-source hashes remained unchanged.

Both written arm datasets passed the existing `reliable_transfer_common.load`
reader, contain 720 train + 174 identical validation records, and have 7,154
resolvable entries in their respective input hash manifests. Their protocols
record fixed 15-epoch / 675-update fine-tuning, but no training was launched.

| Arm | Dataset SHA-256 |
|---|---|
| A_nyc | `5ef37441ce921650f6f2d1ea115aa11d5f4365ab4e4cb77915e2780a4cbcf67e` |
| B_multicity | `c1c839948df7bf6af1db64d8d3cfccde5bb24cf17fabc0d208df330a10f46b49` |

## Fixed comparison

| Arm / city | Each of five uses | Auxiliary role only | Total |
|---|---:|---:|---:|
| A / NYC | 120 | 120 | 720 |
| B / NYC | 40 | 60 | 260 |
| B / Cambridge | 40 | 60 | 260 |
| B / Seattle | 40 | 0 | 200 |

Both arms use exactly the same original 174 NYC validation records, in their
original order, with use counts 30/30/30/30/29 and 25 auxiliary records. All
validation labels, evidence, metadata and image bytes are preserved. Each arm
has 720 distinct training parcels. B's 260 NYC training records are a subset
of A; 460 records differ between arms.

The previously inspected Cambridge 300 / Seattle 244 external tests are
explicitly retired into development/training pools. Their old artifacts remain
unchanged. Any subsequent scores on those cities, including unused or adjacent
records, are development results and must not be presented as heldout evidence.

| Coarse subtypes, in order | A NYC | B NYC | B Cambridge | B Seattle |
|---|---:|---:|---:|---:|
| Small: 1/2/3 dwelling units | 60/30/30 | 10/15/15 | 12/14/14 | 38/1/1 |
| Multi: 4-8 / >=9 units | 60/60 | 20/20 | 20/20 | 20/20 |
| Commercial: retail-service / office | 60/60 | 20/20 | 20/20 | 20/20 |
| Institutional: religious / education / civic-cultural-health | 40/40/40 | 10/10/20 | 5/20/15 | 25/10/5 |
| Other: warehouse / industrial | 60/60 | 8/32 | 26/14 | 26/14 |

Primary coarse-subtype totals match between arms. These broad categories do not
establish equivalence of every municipal source code. Seattle small-residential
is predominantly one-unit. The Cambridge industrial quota uses all 14 eligible
records; failure of any one blocks this exact design. Cambridge utility records
are not relabeled industrial to fill a shortfall.

A auxiliary: 40 NYC 1004 / 80 NYC 5110. B auxiliary: 20 NYC 1004 / 40 NYC 5110 /
60 Cambridge OUTBLDG. NYC has only 43 eligible distinct 1004 training parcels.
OUTBLDG does not establish garage versus other auxiliary subtype; that difference
is explicit and is not claimed to be matched. Auxiliary use stays missing.

## Selection, acquisition and verification

1. Read the frozen eligible pools and verify the original dataset SHA-256
   `0a9fe81ed273067a7a01f962409ce779f9055a501211a2aa33faa6af0161411f`.
   Retain the original NYC train/validation cell assignment. Exclude validation
   parcels, overlapping candidate contexts (wide half-width plus 2 m), and the
   two already documented imagery failures.
2. Persist seed-20261027 queues by city/use/coarse subtype. Order cached original
   imagery first, then the existing `prepare_reliable_transfer.digest(sample_id)`.
   Reserve A's scarce auxiliary parcels first, select B, then fill A around its
   shared NYC subset. Use one parcel across all strata within each arm.
3. Link existing image/metadata/source-window files from the original experiment
   into the new root; do not rewrite them. Copy only the relevant raw NAIP STAC
   catalogs and their saved pagination/provenance files. The initial union has
   1,354 records: 1,177 cached and 177 missing image sets (151 NYC, 24 Cambridge,
   2 Seattle), including the unchanged validation set.
4. Rebind `prepare_reliable_transfer.ROOT` to the new root and call its existing
   `images(city)`. Acquire only missing near/wide RGB, masks, marked images and
   native source windows through public COG APIs. Record every acquisition round.
   An image-quality failure advances the same frozen subtype queue; network or
   other unexpected errors stop acquisition rather than excluding labels.
5. Recheck actual native support against the fixed validation contexts after
   acquisition. An overlapping training context requires a same-subtype
   replacement. No quota, use label, role label, or validation row is relaxed.
6. Reuse `freeze()` for actual context/parcel isolation, then check native
   source hashes, coverage, missing pixels, >99% target retention, and nonempty
   512-pixel masks. Verify reused image hashes against the old input manifest.
   Write per-arm `dataset.json`, `MANIFEST.json`, `INPUT_SHA256.json` and
   `PROTOCOL.json`, with relative links to the shared prepared image cache.
7. Real-row assert checks cover exact counts, all five city/use allocations,
   primary subtype equality, the B-NYC subset, parcel uniqueness, unchanged
   validation, deterministic replacement, and a deliberate exhausted-Cambridge-
   industrial case that must fail. The acquisition report records failures,
   readiness, final hashes, source preservation and image-audit results.

Run these data-only stages with the existing environment, always disabling
bytecode writes so the script remains within its owned paths:

```sh
/private/tmp/la_stage2a_eval_20261023/bin/python -B handoff/stage2a_la/prepare_multicity_transfer.py self-check
/private/tmp/la_stage2a_eval_20261023/bin/python -B handoff/stage2a_la/prepare_multicity_transfer.py prepare
/private/tmp/la_stage2a_eval_20261023/bin/python -u -B handoff/stage2a_la/prepare_multicity_transfer.py acquire
/private/tmp/la_stage2a_eval_20261023/bin/python -u -B handoff/stage2a_la/prepare_multicity_transfer.py freeze
```

The parent updated the repository DINO trainer with `--fixed-epochs`, keeping
its default behavior unchanged. For this comparison, use `--mode finetune
--fixed-epochs` and the existing `--directory` argument for `A_nyc` or
`B_multicity`. Each arm and each seed (42/43) runs exactly 15 epochs: 720 records,
batch size 8, accumulation 2, 45 optimizer steps per epoch, and 675 updates total.
There is no early stopping; the same original NYC validation panel still selects
the best checkpoint. Both arms retain the same model initialization recipe, role
weights and validation score. This task neither changes nor runs the trainer.
The old analysis wrapper assumes NYC-only training and a fixed set of models;
it cannot be used unchanged for the new comparison.

Parent LA handoff: LA continues to use its already frozen earlier trainer
snapshot. Its fixed Maxar cohort has 3,157 usable records and 175 context
failures, all small-residential. The primary all-3,332 protocol retains those
failures as unknown; the common 3,157 subset is reported separately. These are
parent-supplied constraints, not newly audited LA results from this preparation.
No LA scripts, images, labels, or inference state are modified here.

## Candidate final cities: no acquisition authorized here

- Boston: [official assessor parcels](https://gisportal.boston.gov/arcgis/rest/services/Assessing/properties_boston_gov/MapServer/0)
  expose RES_UNITS, COM_UNITS, NUM_BLDGS and use codes;
  [Buildings](https://gis.bostonplans.org/hosting/rest/services/Boston_Buildings/FeatureServer/9)
  supplies outlines. The public
  [outbuildings table](https://gisportal.boston.gov/arcgis/rest/services/Assessing/properties_boston_gov/MapServer/10)
  returned Garage SF and Aux Storage AREA codes during the earlier source check.
  Its tot_units includes area quantities, not dwelling counts. Link role evidence
  to an unambiguous individual outline. Boston is adjacent to Cambridge in the
  same metro: purge all overlap with Cambridge development image support and do
  not describe it as a new metropolitan domain.
- San Francisco: [assessor rolls](https://data.sf.gov/Housing-and-Buildings/Assessor-Historical-Secured-Property-Tax-Rolls/wv5m-vpq2)
  provide parcel IDs, property classes and number_of_units;
  [footprints](https://data.sf.gov/Geographic-Locations-and-Boundaries/Building-Footprints/ynuv-fyni)
  provide building IDs and parcel keys but originate from 2010 geometry. Neither
  supplies explicit auxiliary roles.
  [Building permits](https://data.sf.gov/Housing-and-Buildings/Building-Permits/i98e-djp9)
  can corroborate completed work and existing/proposed site units. They require
  temporal and footprint linkage; commercial garage classes are not automatically
  auxiliaries. Independent auxiliary adjudication remains a real gap.

Both are candidates, not ready final tests. Later acquisition needs date-aligned
imagery, verified parcel/building joins, dwelling-count interpretation, independent
label/visibility review, and a frozen untouched protocol before final scoring.

## Handoff constraints within this owned scope

- Keep the original frozen directory immutable. In particular, do not call
  `images()` or `freeze()` until its module ROOT is rebound to the new directory.
- Missing use is not an additional class. Preserve source codes and unknown
  auxiliary subtypes; no inferred occupancy or population truth is introduced.
- A failed source request is not evidence that a label is ineligible. Preserve
  the selection and report infrastructure blockers.
- Relative arm links depend on the shared cache; cached files also link to the
  preserved original experiment. A later transfer must dereference links or copy
  their targets, then verify the per-arm hashes.
- No changes to the root HANDOFF.MD are made because the user limited owned paths
  to this script, plan, and the new X9 experiment directory.
