"""Freeze all LA product targets, preserving the historical reference cohort."""

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.features import rasterize
from rasterio.transform import Affine
from rasterio.windows import Window
from shapely import affinity, wkt

import prepare_la_transfer as transfer
from reliable_transfer_common import load, save
from run_la import read_csv, sha256, validate_images

STAGE=transfer.STAGE
ROOT=STAGE/'la_full_20261030'
REMOTE='/stage3/stage2a_la_full_20261030'


def main():
    assert not (ROOT/'PROTOCOL.json').exists(), 'Full inference inputs already frozen'
    for name in ['images','source_cells','alignment','execution']:(ROOT/name).mkdir(exist_ok=True)
    product_path=STAGE/'la_eval_20261023/inputs/building_damage_all_cells.csv'
    input_path=STAGE/'la_explore_20261023/data/la_input.csv'
    products=read_csv(product_path);existing=read_csv(input_path)
    keys=[(r['cell_id'],r['bldg_uid']) for r in products]
    assert len(keys)==len(set(keys))==22024
    existing={r['building_uid']:r for r in existing};assert len(existing)==21888
    references,reference_protocol=load(STAGE/'la_transfer_20261027')
    ref_ids={r['building_uid'] for r in references}
    shared={};grids={};repairs=[];records=[];images={};recovery_sources={}
    for cell in sorted({r['cell_id'] for r in products if uid(r) not in existing}):
        path=STAGE/'la_eval_20261023/inputs'/cell/'shared_base/shared_instance_samples.csv'
        shared[cell]={r['bldg_uid']:r for r in read_csv(path)}
        normal=[r for r in products if r['cell_id']==cell and uid(r) in existing]
        probes=[]
        for r in sorted(normal,key=lambda r:float(existing[uid(r)]['pre_zero_fraction']))[:3]:
            old=shared[cell][r['bldg_uid']];g=wkt.loads(r['polygon_wkt_utm']);px=wkt.loads(old['polygon_wkt_xy_pre'])
            gsd=600/int(old['width']);origin=transfer.cell_origin(np.asarray(px.exterior.coords),np.asarray(g.exterior.coords),gsd)
            probes.append(dict(sample_id='la_'+cell+'_'+r['bldg_uid'],original_gsd=gsd,origin=origin.tolist(),
                geometry_wkt=g.wkt,wide_m=80,original_crop=str(local(existing[uid(r)]['crop_path'])),
                crop_x0=int(old['crop_x0']),crop_y0=int(old['crop_y0'])))
        assert len(probes)==3 and all(p['origin']==probes[0]['origin'] and p['original_gsd']==probes[0]['original_gsd'] for p in probes)
        grids[cell]=probes
    transfer.ROOT=ROOT
    items=json.loads((STAGE/'la_transfer_20261027/maxar_prefire_items.json').read_text())
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',GDAL_HTTP_TIMEOUT=40,GDAL_HTTP_MAX_RETRY=2,GDAL_CACHEMAX=256*1024**2):
        for i,r in enumerate(products):
            key=uid(r);cell=r['cell_id'];sid='la_'+cell+'_'+r['bldg_uid']
            if key in existing:
                row=dict(existing[key],sample_id=sid,input_status='original_valid')
                validate_images(local(row['crop_path']),local(row['mask_path']))
            else:
                assert key not in ref_ids, 'Never change a frozen test input'
                pixel=wkt.loads(r['polygon_wkt_utm']);grid=grids[cell][0];gsd=grid['original_gsd'];origin=grid['origin']
                assert pixel.is_valid and pixel.area>0 and min(pixel.bounds)>-64 and max(pixel.bounds)<2030
                geometry=affinity.affine_transform(pixel,[gsd,0,0,-gsd,*origin])
                source=STAGE/'la_transfer_20261027/source_cells'/(cell+'.tif')
                receipt=STAGE/'la_transfer_20261027/alignment'/(cell+'.json')
                if not source.exists():
                    transfer.recover(cell,grids[cell],items)
                    source=ROOT/'source_cells'/(cell+'.tif');receipt=ROOT/'alignment'/(cell+'.json')
                assert json.loads(receipt.read_text())['passed']
                if cell not in recovery_sources:
                    recovery_sources[cell]=dict(source=str(source),source_sha256=sha256(source),
                        receipt=str(receipt),receipt_sha256=sha256(receipt),grid=grid)
                center=pixel.centroid
                if not pixel.contains(center):center=pixel.representative_point()
                size=round(600/gsd)
                x=max(0,min(size-256,int(round(center.x))-128));y=max(0,min(size-256,int(round(center.y))-128))
                crop_transform=Affine(gsd,0,origin[0],0,-gsd,origin[1])*Affine.translation(x,y)
                with rasterio.open(source) as src:
                    col,rowoff=(~src.transform)*(crop_transform.c,crop_transform.f)
                    assert abs(col-round(col))<1e-5 and abs(rowoff-round(rowoff))<1e-5
                    rgb=src.read([1,2,3],window=Window(round(col),round(rowoff),256,256),boundless=True,fill_value=0)
                mask=rasterize([(geometry,255)],out_shape=(256,256),transform=crop_transform,dtype='uint8')
                assert mask.any() and (rgb.max(axis=0)[mask>0]>0).mean()>.95, sid
                crop=ROOT/'images'/(sid+'.png');mp=ROOT/'images'/(sid+'_mask.png')
                Image.fromarray(rgb.transpose(1,2,0)).save(crop);Image.fromarray(mask).save(mp)
                zero=validate_images(crop,mp)
                row=dict(building_uid=key,cell_id=cell,bldg_uid=r['bldg_uid'],sample_id=sid,input_status='recovered_pixel_geometry',
                    crop_path=REMOTE+'/images/'+crop.name,mask_path=REMOTE+'/images/'+mp.name,
                    footprint_m2=geometry.area,log_footprint_m2=math.log1p(geometry.area),pre_zero_fraction=zero,gsd=gsd,
                    footprint_m2_source='original_pixel_coordinates_georeferenced_using_three_unchanged_cell_anchors',
                    tile_id=cell,event_id='la_fire_2025',hazard_type='fire',sam3_confidence=r.get('sam3_confidence',''))
                repairs.append(dict(sample_id=sid,cell_id=cell,raw_mislabeled_utm_wkt=r['polygon_wkt_utm'],
                    recovered_geometry_wkt=geometry.wkt,crs='EPSG:32611',gsd=gsd,origin=origin,
                    crop_x0=x,crop_y0=y,mask_pixels=int((mask>0).sum()),
                    target_observed_fraction=float((rgb.max(axis=0)[mask>0]>0).mean()),
                    note='Pixel convention inferred from original geometry; separate input repair, not a reference label.'))
            records.append(row)
            for field in ['crop_path','mask_path']:
                path=row[field];images[path]=sha256(local(path))
            if (i+1)%4000==0:print('PREPARE',i+1,'/',len(products),flush=True)
    assert len(records)==22024 and len(repairs)==136 and len(images)==44048
    previous=json.loads((STAGE/'la_adaptation_20261029/DATASET.json').read_text())
    lookup={r['sample_id']:r for r in records}
    for r in previous:
        if r['split']=='test':
            for k in ['crop_path','mask_path','building_uid','footprint_m2','log_footprint_m2']:
                assert lookup[r['sample_id']][k]==r[k],(r['sample_id'],k)
    old_paths={str(seed):STAGE/('la_data_20261024' if seed==42 else 'la_data_20261024/followup_20261025')/'predictions'/f'N2_nyc_natural_seed{seed}'/'la.csv' for seed in [42,43]}
    old_predictions={}
    for seed,path in old_paths.items():
        rr=read_csv(path);old_predictions[seed]={r['building_uid']:[float(r['prob_'+n]) for n in ['residential_small','residential_multi','commercial']] for r in rr}
        assert set(old_predictions[seed])==set(existing)
    baseline=json.loads((Path(__file__).parent/'analysis/20261029_la_adaptation/PREDICTIONS.json').read_text())
    probe_ids=[r['sample_id'] for label in ['residential_small','residential_multi','commercial'] for r in [q for q in references if q['reference_use']==label][:12]]
    expected={str(seed):{r['sample_id']:r['probabilities'] for r in baseline[f'cnn_lowrise_s{seed}'] if r['sample_id'] in probe_ids} for seed in [42,43]}
    adaptation=json.loads((STAGE/'la_adaptation_20261029/PROTOCOL.json').read_text());checkpoints={}
    for seed in [42,43]:
        base=adaptation['initial_checkpoints'][str(seed)];new='/stage3/stage2a_la_adaptation_20261029/runs/cnn_lowrise_s'+str(seed)+'/best.pt'
        model=STAGE/'la_adaptation_20261029/execution/collected/runs'/f'cnn_lowrise_s{seed}'
        result=json.loads((model/'RESULT.json').read_text());assert sha256(model/'best.pt')==result['checkpoint_sha256']
        checkpoints[str(seed)]=dict(initial=base,initial_sha256=sha256(local(base)),adapted=new,adapted_sha256=result['checkpoint_sha256'],selected_epoch=result['best_epoch'])
    save(ROOT/'DATASET.json',records);save(ROOT/'REPAIRS.json',repairs);save(ROOT/'RECOVERY_SOURCES.json',recovery_sources)
    save(ROOT/'INPUT_SHA256.json',images);save(ROOT/'OLD_PREDICTIONS.json',old_predictions);save(ROOT/'EXPECTED_PROBES.json',expected)
    sources=[product_path,input_path,STAGE/'la_explore_20261023/data/la_reference.csv',STAGE/'la_eval_20261023/reference/la_assessor_matched.csv',STAGE/'la_transfer_20261027/dataset.json',*old_paths.values()]
    save(ROOT/'PROTOCOL.json',dict(dataset_sha256=sha256(ROOT/'DATASET.json'),original_test_sha256=reference_protocol['dataset_sha256'],
        n=22024,classes=['residential_small','residential_multi','commercial'],input_status_counts=dict(Counter(r['input_status'] for r in records)),
        cnn_config=adaptation['cnn_config'],checkpoints=checkpoints,probe_ids=probe_ids,common_sha256=sha256(ROOT/'execution/common.py'),
        source_sha256={str(p):sha256(p) for p in sources},
        frozen_files={name:sha256(ROOT/name) for name in ['INPUT_SHA256.json','OLD_PREDICTIONS.json','EXPECTED_PROBES.json','REPAIRS.json']},
        primary='Unchanged primary_single_outline_iou50 3332; strict subset secondary; broader parcel labels exploratory',
        old_comparison='Original N2 seed42/43 probability mean; reuse frozen valid-input scores after parity check; recompute repaired inputs',
        limits=['Parcel references are not independent building gold','Repaired136 have no newly assigned ground truth','LA is historical development, not untouched generalization']))
    print('FULL_INPUT_FREEZE_PASS',len(records),'repaired',len(repairs),flush=True)


def uid(row):
    return json.dumps([row['cell_id'],row['bldg_uid']],separators=(',',':'))


def local(path):
    assert path.startswith('/stage3/')
    return STAGE/path[len('/stage3/'):].removeprefix('stage2a_')


if __name__=='__main__':main()
