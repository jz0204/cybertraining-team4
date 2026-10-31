# Best-C LA fire visual preview

[Interactive local viewer](/Volumes/Crucial X9/Instance-Impact-IGUIDE-SpatialAIChallenge2026/II_package/outputs/stage2a/la_best_preview_20261031/viewer/index.html) · [Preview image](/Volumes/Crucial X9/Instance-Impact-IGUIDE-SpatialAIChallenge2026/II_package/outputs/stage2a/la_best_preview_20261031/viewer/preview.jpg) · [All predictions](/Volumes/Crucial X9/Instance-Impact-IGUIDE-SpatialAIChallenge2026/II_package/outputs/stage2a/la_best_preview_20261031/viewer/predictions.csv)

Fresh CPU inference of both frozen C_positive_negative checkpoints covers all 22,024 records in the unchanged full-LA manifest. Mean use probabilities yield 20,993 small residential, 187 multifamily, and 844 commercial predictions. Both jobs completed; all 44,048 image hashes passed per seed. The full 3,332 reference cohort matches the saved C class predictions exactly (maximum probability difference 0.000001818), preserving macro-F1 0.610370 and multifamily TP3/FP15/FN11.

The viewer uses the downloaded LA_fire_results_page_2026-10.html as its reference. All original point coordinates and damage votes were matched by row and coordinate. The saved v3 damage counts remain 14,486 no damage, 122 minor, 370 major, 5,748 destroyed and 1,298 unknown. They are existing multi-date Stage2b outputs, not outputs of the new Stage2a use inference. Exposure/population was not recomputed.

The page includes 126-cell Palisades/Eaton maps, use/damage category filters, per-building probability bars and crops, interactive use outlines for cells00365/00471/00507, and all14multifamily reference cases. Scene00365 includes405records with3repaired geometries; its saved post-fire illustration shows402footprints. The original HTML and imagery were preserved.

The HTML embeds map data and scene pictures; crops load lazily via two links to the existing external-drive datasets. Keep the drive mounted for crop inspection. Syntax, DOM target IDs, all22,024crop/mask paths and point-in-polygon courtyard behavior were checked.

With the user's explicit browser permission, interaction checks passed in the Codex browser at the observed narrow and desktop layouts: all three scenes, enlarged-image selection and Close/Escape, crop/probability navigation, category filters, map-point selection, and reference-case selection. All 14 reference thumbnails loaded, including the three recognized cases and eleven misses. The CSV download was retrieved; browser logs showed no warnings or errors during the check. Screenshots are saved beside this report.

Browser inspection exposed a source-coordinate issue: the old report's centroids for the 136 previously repaired geometries lay near the equator, stretching the Palisades map almost blank. The generator now derives those centroids from their already-recovered EPSG:32611 geometries. All 21,888 other coordinates and every use prediction/damage vote remain unchanged. The repaired locations place 2,870 records in Palisades and 19,154 in Eaton. The generator checks all locations against the LA bounds; a separate verification checked that exactly the 136 repair records changed and that their coordinates transform back to the recovered geometry centroids.

No models were trained or promoted. CPU inference left the GPU services running and ready. Source: build_la_best_preview.py and la_best_preview.html; tests/check_la_preview.js exercises the actual polygon hit-test.

## Exposure update, 31 October 2026

Supersedes earlier exposure-not-recomputed statements: current C probabilities now feed the frozen B_semantics_mean3 OOF exposure head. All 22,024 estimates appear in the inspector and CSV. Total proxy: 69,023.955; minor/major/destroyed-building proxy: 17,522.957. Unknown damage is excluded from the damaged subtotal. These are population proxies, not occupancy/casualty/displacement truth; this head with C probabilities on LA remains unvalidated. Saved test log predictions reproduced exactly (max error 0); CSV/HTML values and aggregates agree. No models retrained.
