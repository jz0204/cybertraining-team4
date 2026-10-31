"""Compare frozen LA exposure outputs on spatially complete Census areas."""
import csv, hashlib, json, re
from collections import defaultdict
from pathlib import Path
import fiona
import numpy as np
from pyproj import Transformer
from shapely import wkt
from shapely.geometry import Point, box, shape
from shapely.ops import transform, unary_union
from shapely.strtree import STRtree

HERE=Path(__file__).resolve().parent
STAGE=Path('/Volumes/Crucial X9/Instance-Impact-IGUIDE-SpatialAIChallenge2026/II_package/outputs/stage2a')
PREVIEW=STAGE/'la_best_preview_20261031'
CENSUS=STAGE/'la_eval_20261023/census/la_census'
OUT=HERE/'analysis/20261031_exposure_validation'


def main():
    OUT.mkdir(exist_ok=True)
    raw=read_csv(STAGE/'la_eval_20261023/inputs/building_damage_all_cells.csv')
    old_path=HERE/'results/la_v3_full/stage2a_predictions.csv'
    old={ (r['cell_id'],r['bldg_uid']):r for r in read_csv(old_path)}
    html=(PREVIEW/'viewer/index.html').read_text()
    data=json.loads(re.search(r'<script id="la-data" type="application/json">(.*?)</script>',html,re.S).group(1))
    current=data['records'];assert len(current)==22024 and len(old)==21888
    bycell=defaultdict(list)
    for r in raw:bycell[r['cell_id']].append(r)
    grids={}
    for cell,rows in bycell.items():
        samples={r['bldg_uid']:r for r in read_csv(STAGE/f'la_eval_20261023/inputs/{cell}/shared_base/shared_instance_samples.csv')}
        origins=[]
        for r in [r for r in rows if (cell,r['bldg_uid']) in old][:3]:
            p=samples[r['bldg_uid']];g=np.array(wkt.loads(r['polygon_wkt_utm']).exterior.coords);xy=np.array(wkt.loads(p['polygon_wkt_xy_pre']).exterior.coords)
            assert g.shape==xy.shape
            offsets=g-xy*[600/int(p['width']),-600/int(p['width'])];origin=np.round(np.median(offsets,axis=0)/600)*600
            assert np.max(np.abs(offsets-origin))<.002
            origins.append(origin)
        assert origins and np.allclose(origins,origins[0]), (cell,len(origins))
        grids[cell]=origins[0].tolist()
    footprint=unary_union([box(x,y-600,x+600,y) for x,y in grids.values()])
    project=Transformer.from_crs(4326,32611,always_xy=True).transform
    inverse=Transformer.from_crs(32611,4326,always_xy=True).transform
    bounds=transform(inverse,footprint).bounds
    summaries={};assignments=[]
    for level in ['blocks_2020','block_groups']:
        geoms=[];areas=[]
        with fiona.open(CENSUS/f'la_county_{level}.gpkg') as source:
            for feature in source.filter(bbox=bounds):
                g=transform(project,shape(feature['geometry']))
                intersection=g.intersection(footprint).area
                if intersection<=0:continue
                prop=dict(feature['properties']);coverage=intersection/g.area
                geoms.append(g);areas.append(dict(geoid=prop['geoid'],coverage=coverage,population=prop.get('pop20'),housing=prop.get('housing20'),n_current=0,n_shared=0,n_missing_old=0,old=0.,current_shared=0.,current_all=0.,old_residential=0.,current_residential=0.))
        if level=='block_groups':
            acs=json.loads((OUT/'ACS_REFERENCE.json').read_text())
            assert acs['source_sha256']==hashlib.sha256((CENSUS/'la_county_acs_block_group_2022.parquet').read_bytes()).hexdigest()
            lookup={r['geoid']:r for r in acs['rows']}
            for a in areas:
                a['population']=float(lookup[a['geoid']]['B01003_001E'])
                a['population_moe90']=float(lookup[a['geoid']]['B01003_001M'])
        tree=STRtree(geoms);unmatched=0;ambiguous=0
        for r in current:
            key=(data['cells'][r[1]],r[0]);p=Point(*project(r[2],r[3]));hits=tree.query(p,predicate='intersects')
            if len(hits)!=1:
                unmatched+=len(hits)==0;ambiguous+=len(hits)>1;continue
            a=areas[int(hits[0])];v=r[10];a['n_current']+=1;a['current_all']+=v
            if level=='blocks_2020':assignments.append(dict(cell_id=key[0],building_uid=key[1],geoid=a['geoid'],current=v,original=float(old[key]['pred_population']) if key in old else ''))
            if key not in old:a['n_missing_old']+=1;continue
            o=old[key];ov=float(o['pred_population']);assert np.isfinite(ov) and ov>=0
            a['n_shared']+=1;a['old']+=ov;a['current_shared']+=v
            if o['pred_type_class']!='commercial':a['old_residential']+=ov
            if r[5]!=2:a['current_residential']+=v
        assert unmatched==ambiguous==0,(level,unmatched,ambiguous)
        assert sum(a['n_current'] for a in areas)==22024 and sum(a['n_shared'] for a in areas)==21888
        write_csv(OUT/f'{level}.csv',areas)
        cohorts={'fully_covered_no_missing_original':[a for a in areas if a['coverage']>=.999999 and a['n_missing_old']==0], 'fully_covered_all':[a for a in areas if a['coverage']>=.999999], 'coverage_99pct_sensitivity':[a for a in areas if a['coverage']>=.99]}
        summaries[level]=dict(intersecting=len(areas),partial=sum(a['coverage']<.999999 for a in areas),unmatched=unmatched,ambiguous=ambiguous,cohorts={name:metrics(rows) for name,rows in cohorts.items()})
    write_csv(OUT/'building_comparison.csv',assignments)
    sources=[old_path,PREVIEW/'EXPOSURE.json',CENSUS/'la_county_blocks_2020.gpkg',CENSUS/'la_county_block_groups.gpkg',CENSUS/'la_county_acs_block_group_2022.parquet']
    result=dict(summary=summaries,full_inventory=dict(n_current=22024,n_shared=21888,current_all=sum(r[10] for r in current),current_shared=sum(r[10] for r in current if (data['cells'][r[1]],r[0]) in old),original=sum(float(r['pred_population']) for r in old.values())),source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},cell_origins=grids,limits=['2020 population/2022 ACS versus pre-2025 fire imagery; not fire-time occupancy.','Provided boundaries are TIGER2025 with older population joined by GEOID; vintage crosswalk not independently verified.','Full tile coverage does not establish complete building detection or populated pixels.','Primary comparison excludes blocks with 136 previously unavailable original-model inputs; zero-prediction blocks retained.','Current-versus-original compares whole pipelines, including a different exposure head; no classifier-only causal claim.','Commercial/daytime occupancy differs from Census usual residence; residential-only sums are sensitivity checks.','No population rescaling, fitting or tuning to LA Census values.'])
    (OUT/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
    write_report(result)
    print(json.dumps(dict(summary=summaries,full_inventory=result['full_inventory']),indent=2))


def write_report(result):
    inv=result['full_inventory']
    lines=['# LA exposure validation and original-model comparison', '',
        'The current pipeline improves agreement with 2020 Census blocks, but still substantially overestimates population. The 2022 ACS block-group check does not consistently improve. These are aggregate consistency checks, not validated building occupancy or fire-time exposure.', '',
        '## Same-building comparison', '',
        f"The original saved v3 output contains {inv['n_shared']:,} buildings. Their original total is {inv['original']:,.1f}, versus {inv['current_shared']:,.1f} for the current pipeline. The current complete inventory adds 136 repaired inputs and totals {inv['current_all']:,.1f}. Do not compare these different inventories as model-only gains.", '',
        'The original comparator is the saved `pred_population` field, not a reconstructed approximation. The current pipeline uses C42/43 mean probabilities and the frozen B_semantics_mean3 exposure head. Both classifier and exposure estimator changed; this comparison cannot attribute improvement to the classifier alone.', '',
        '## Spatial comparison', '',
        'Coverage is computed from the union of 126 recovered 600 m image cells in EPSG:32611. Fully covered means at least 99.9999% polygon area coverage. The primary cohort additionally excludes areas containing any of the 136 records absent from the original output. Zero-prediction and zero-population areas are retained. No Census population is allocated by area to partial polygons.', '',
        '| Reference and cohort | Areas | Census population | Original total | Current shared total | Original MAE | Current MAE | Original bias | Current bias |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for level,title in [('blocks_2020','2020 Census blocks'),('block_groups','2022 ACS block groups')]:
        for cohort,label in [('fully_covered_no_missing_original','complete, no missing old inputs'),('fully_covered_all','complete, shared inputs'),('coverage_99pct_sensitivity','99% coverage sensitivity')]:
            m=result['summary'][level]['cohorts'][cohort]
            if not m['n']:continue
            o,c=m['old'],m['current_shared']
            lines.append(f"| {title}: {label} | {m['n']} | {m['population']:,.0f} | {o['total']:,.0f} | {c['total']:,.0f} | {o['mae']:.2f} | {c['mae']:.2f} | {o['bias']:+.1%} | {c['bias']:+.1%} |")
    primary=result['summary']['blocks_2020']['cohorts']['fully_covered_no_missing_original']
    lines += ['',f"Primary block cohort: {primary['n_buildings']:,} buildings; current MAE reduction {1-primary['current_shared']['mae']/primary['old']['mae']:.1%}. MAE is people per area; bias is (predicted total − reference total)/reference total. This is descriptive; no spatially independent significance or confidence claim is made.", '',
        '## Residential-only sensitivity', '',
        'Census population measures usual residence, not commercial/daytime occupants. Restricting each model to its own predicted residential buildings changes both the population sum and the included building set; it is a sensitivity analysis, not verified residential selection.', '',
        '| Cohort | Original MAE | Current MAE | Original total bias | Current total bias |', '|---|---:|---:|---:|---:|']
    for level,title in [('blocks_2020','323 primary Census blocks'),('block_groups','10 fully covered ACS groups')]:
        cohort='fully_covered_no_missing_original' if level=='blocks_2020' else 'fully_covered_all'
        m=result['summary'][level]['cohorts'][cohort];o,c=m['old_residential'],m['current_residential']
        lines.append(f"| {title} | {o['mae']:.2f} | {c['mae']:.2f} | {o['bias']:+.1%} | {c['bias']:+.1%} |")
    lines += ['', '## Interpretation and next step', '',
        'The preview should continue to label exposure as a proxy. Define resident population separately from commercial/daytime occupancy, audit accessory/duplicate/missed buildings, and validate any population calibration on geographically held-out blocks. Do not rescale to these Census totals and then call the same totals validation. The existing 17,523 damaged-building proxy has no observed displaced-person or casualty reference here.', '',
        '## Limits', '', *['- '+v for v in result['limits']], '',
        'ACS sampling margins of error are retained per group in block_groups.csv. Point-estimate errors do not incorporate ACS sampling uncertainty; the 10-group sensitivity cohort is small and the two-group complete-input cohort is insufficient for a strong generalization claim.', '',
        '## Provenance and checks', '',
        'Used the already-downloaded LA County Census package, not a new population model. Its README identifies 2020 Decennial counts, 2018–2022 ACS estimates, and TIGER2025 boundaries. All 22,024 buildings matched exactly one block and one block group. The old/current shared join has 21,888 records. Grid alignment is checked against available original polygon vertices; joins, valid population values, counts and source SHA256 are asserted or recorded in RESULTS.json. No calibration or model changes were made.', '',
        '[Census usual-residence definition](https://www.census.gov/programs-surveys/decennial-census/decade/2020/about/residence-rule.html) · [ACS five-year geography and data](https://www.census.gov/data/developers/data-sets/acs-5year.html)', '',
        'Files: RESULTS.json; blocks_2020.csv; block_groups.csv (with ACS 90% margins of error); building_comparison.csv; ACS_REFERENCE.json (source-hashed projection of geoid/B01003_001E/B01003_001M).', '',
        'Reproduce: run validate_la_exposure.py with the existing py311 GIS environment. ACS_REFERENCE.json was extracted from the supplied Parquet using pyarrow; its original Parquet hash is checked at every run. Temporary reader packages are isolated in /private/tmp/la-census-parquet, and NumPy2 in /private/tmp/la-exposure-runtime; normal GIS runtime packages were not changed.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')


def metrics(rows):
    if not rows:return dict(n=0)
    y=np.array([a['population'] for a in rows],float);assert np.isfinite(y).all() and (y>=0).all()
    output=dict(n=len(rows),population=float(y.sum()),n_buildings=sum(a['n_current'] for a in rows),missing_original=sum(a['n_missing_old'] for a in rows),zero_population_areas=int((y==0).sum()))
    for name in ['old','current_shared','current_all','old_residential','current_residential']:
        p=np.array([a[name] for a in rows]);e=p-y
        output[name]=dict(total=float(p.sum()),mae=float(np.abs(e).mean()),rmse=float(np.sqrt((e*e).mean())),wape=float(np.abs(e).sum()/y.sum()) if y.sum() else None,bias=float(e.sum()/y.sum()) if y.sum() else None)
    return output


def read_csv(path):
    with path.open() as f:return list(csv.DictReader(f))


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


if __name__=='__main__':main()
