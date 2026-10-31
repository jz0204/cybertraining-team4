# New-city readiness: Denver and San Francisco

Verified 2026-10-28. **Prioritize Denver.** These are official-reference candidates,
not actual-building truth, human-adjudicated gold, or exact-date occupancy labels.
The current A/B training datasets are frozen; NYC/Cambridge/Seattle are development,
as are LA/Harris/SD. No Denver/SF use was found in the narrowly inspected Stage2a
preparation/training/configuration/protocol paths; SF appeared only as a proposed
candidate. This is a scoped repository check, not an exhaustive historical audit.

[Aggregate receipts](NEW_CITY_READINESS_RECEIPTS.json) preserve exact endpoints,
query parameters/SoQL and observed response snippets from the preceding read-only
inspection. They are transcribed receipts, not raw HTTP captures. No further
research or imagery acquisition was performed to save this note.

## Readiness and obtainable candidate counts

| City / denominator | Small | Multi | Commercial | Institutional | Other | Readiness |
|---|---:|---:|---:|---:|---:|---|
| Denver: screened assessor parcels, before footprint joins | 132,389 | 3,049 | 3,172 | 717 | 2,466 | Strongest candidate; building attribution and temporal checks pending |
| SF: FY2025 roll, positive property area, one matched footprint, recorded construction year 1700–2010 | 98,671 | 9,334 | 3,035 | 379 | 1,289 | Five-use screening feasible; auxiliary references unresolved |

These are candidate denominators, **not guaranteed final eligible building counts**.
Denver screening requires positive relevant floor area and construction year;
multi accepts residential or commercial fields. SF's corresponding counts before
the footprint/year filters are 111,493 / 10,681 / 3,640 / 569 / 1,588.
A singleton footprint key does not establish a primary role, current geometry,
per-building dwelling allocation, or actual use.

## Official sources and structural linkage

- **Denver parcels:** [Socrata dataset/metadata](https://data.colorado.gov/api/views/tsdg-z9uy.json);
  [authoritative polygon service, layer 245](https://services1.arcgis.com/zdB7qR0BtYrg0Xpl/arcgis/rest/services/ODC_PROP_PARCELS_A/FeatureServer/245).
  The supplied [envj-hfws map](https://data.colorado.gov/Government/Map-of-Denver-Parcels/envj-hfws)
  points to `tsdg-z9uy`. Fields include `SCHEDNUM`, `D_CLASS`, `D_CLASS_CN`,
  `TOT_UNITS`, `RES_ORIG_YEAR_BUILT`, `COM_ORIG_YEAR_BUILT`,
  `RES_ABOVE_GRADE_AREA`, and `COM_GROSS_AREA`. No building count/foreign key
  is exposed. Preserve text identifiers from ArcGIS, including leading zeros.
- **Denver structures:** [Building Outlines 2022, layer 111](https://services1.arcgis.com/zdB7qR0BtYrg0Xpl/ArcGIS/rest/services/ODC_PROP_BUILDINGOUTLINES_A/FeatureServer/111),
  published by The_City_and_County_of_Denver. Union roof-level polygons by
  `BUILDING_I`; `OBJECTID` is a roof-record ID. Fields include `BLDG_TYPE`
  and `SOURCE`; no parcel key is exposed. The whole service has 390,104 polygons
  and 379,481 distinct building IDs and extends outside Denver. Use the
  [county boundary layer](https://services1.arcgis.com/zdB7qR0BtYrg0Xpl/arcgis/rest/services/ODC_ENG_SRVCCDBOUNDARY_A/FeatureServer/40)
  with `COUNTY='Denver'`, not its other counties.
- **SF:** [assessor rolls](https://data.sf.gov/d/wv5m-vpq2),
  [official class dictionary](https://data.sf.gov/d/pa56-ek2h),
  [footprint polygon API](https://data.sf.gov/resource/ynuv-fyni.json),
  [footprint schema](https://data.sf.gov/api/views/ynuv-fyni.json), and
  [active/retired parcel polygons](https://data.sf.gov/d/acdm-wktn).
  Footprints expose `shape`, `sf16_bldgid` (177,023 unique IDs) and `mblr`.
  Verified join: `mblr = 'SF' || parcel_number`; exclude `SM` and `CN`
  identifiers. Preserve letter suffixes. `blklot` identifies a parcel;
  `mapblklot` can be shared by condominium lots. Neither roll nor parcel
  schema supplies building counts. Inspect overlap and parcel history after
  key matching; the original footprint linkage used centroids.

## Exact screening mappings

`U` is Denver `tot_units` or SF `number_of_units`. Both are property-level
fields; neither inspected dictionary certifies dwelling allocation to a single
physical building. Never divide a parcel total among multiple primary buildings.

| Target | Denver | SF property_class_code |
|---|---|---|
| Small (1–3 dwellings) | 112–119 with U=1; 122–124 with U=2; 132–134 with U=3 | D,DBM with U=1; F,A,AG with U in {1,2,3} |
| Multi (at least 4) | Exact descriptions RESIDENTIAL-4 TO 8 UNITS, RESIDENTIAL-APARTMENT, RESIDENTIAL-MULTI UNIT APTS; U>=4 | A,AG with U=4; A5,A5G,F5 with U=5–14; A15,A15G,F15 with U>=15 |
| Commercial | Exact descriptions COMMERCIAL-RETAIL, COMMERCIAL-OFFICE | B,C,C1,C1G,CD,CG,O,OAH,OAL,OAM,OAT,OBH,OBL,OBM,OCH,OCL,OCM,OG |
| Institutional | 309,331,332,333,394,395 | E,EG,W,WG,N1,N1G,N2,N2G,FIRG,POLG,LIBG |
| Other | INDUSTRIAL- plus WAREHOUSE, FACTORY, PRINTING PLANT, MEAT PACKING, FOOD PROCESSING, BRICK PLANT | I,IG,IW,IDC |

Full predicates and response counts are in the receipts. Exclude mixed-use,
condominium, vacant, missing/ambiguous and explicit multiple-building parcel classes.
Do not map Denver's INDUSTRIAL-SCHOOL/CHURCH to other. SF A means <=4 units, so
it crosses the small/multi boundary. Misc/unknown is not other; explicit utility
subtype coverage remains unverified.

## Role and date limitations

- Denver has **126,037 distinct Garage/Shed building IDs** fully contained within
  the Denver boundary. `SOURCE='Visual'` applies to the roofprint data.
  This is official structural evidence, not 126,037 verified auxiliary labels.
  Require an unambiguous residential parent and companion primary; investigate
  conflicting roof-group types and conversions. Size alone and commercial
  parking classifications do not establish auxiliary role.
- SF footprints have no role field. [Building permits](https://data.sf.gov/d/i98e-djp9)
  expose scope, status, completion dates, existing/proposed use and **site** units.
  Completed permits mentioning DETACHED GARAGE by 2022-05-18 comprise 116 rows /
  103 distinct permits, not 103 standing auxiliary buildings. Descriptions may
  concern conversion/demolition; one permit can repeat across addresses.
- Denver outlines are from **2022**; a catalog-only NAIP center-point check found
  **2023-09-25, 0.3 m**. Assessor attributes are current, not a verified 2023
  historical snapshot. SF geometry originates in **2010**; its catalog spot check
  found **2022-05-18, 0.6 m**, whereas the counted roll is **2025**. Historical
  SF rolls are available, but alignment has not been established. The exact
  NAIP requests and metadata snippets are in the receipts; no image was selected
  for evaluation. Refresh dates and construction year do not establish unchanged
  use, footprint, units, or role. Center-point coverage is not case-level coverage.

## Minimal next steps

1. **Freeze the Denver selection and score protocol before candidate selection or
   fresh-city predictions.** Preserve frozen A/B training data. Lock the seed/hash
   ordering, 50 per use plus 50 auxiliary targets, bounded candidate budget,
   deterministic replacements, exclusions/unknowns, temporal policy, case evidence,
   role/use denominators, source-only checkpoint selection and spatial-cluster
   bootstrap. Fix model/checkpoint hashes before any eventual predictions.
2. Prepare only Denver using official records and bounded ID-based retrieval.
   Union BUILDING_I groups; require exactly one non-Garage/Shed primary group,
   unambiguous >=90% footprint containment and consistent source code/units.
   Auxiliary records need a residential parent and companion primary. Enforce
   one parcel per selected record across all strata. Preserve failed candidates
   and ordered replacements; do not infer unit splits or generate VLM labels.
3. Under the frozen protocol, obtain real near/wide views through the existing
   pipeline, verify target/context coverage and alignment, and retain case-level
   source/version/date provenance. This note itself performs no acquisition.
   Stop with actual eligibility/quality counts if any stratum cannot reach 50;
   do not relax definitions to meet quotas.
4. Freeze the eligible cohort, references, inputs and exclusions before predictions.
   Keep Denver out of training/tuning. Report official-reference performance and
   the 2022/2023/current-record mismatch; do not claim human or exact-date gold.
   Keep SF acquisition deferred until its role/date gaps have an agreed protocol.

This document does not amend Locke's PLAN.md or the frozen A/B datasets.

