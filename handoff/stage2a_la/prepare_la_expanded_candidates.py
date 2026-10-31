"""Screen isolated county references and acquire dated NAIP review images."""
import argparse
import json
import math
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from shapely import wkt
from shapely.geometry import box,shape
from shapely.ops import transform

import prepare_la_adaptation as base
from vlm_multiclass_data import view_sizes

OLD=base.ROOT
ROOT=base.STAGE/'la_hard_expansion_20261030'
NAIP='https://imagery.geoplatform.gov/iipp/rest/services/NAIP/NAIP2022_CONUS/ImageServer'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['footprints','images','sheets'])
    a=p.parse_args()
    assert not (ROOT/'PROTOCOL.json').exists(),'Preserve frozen inputs'
    for name in ['raw','images','source_windows','image_metadata','acquired_images','contact_sheets']:
        (ROOT/name).mkdir(parents=True,exist_ok=True)
    base.ROOT=ROOT
    {'footprints':footprints,'images':images,'sheets':sheets}[a.action]()


def footprints():
    queue=[]
    for use in ['residential_multi','residential_small']:
        pool=json.loads((ROOT/(use+'_EXPANDED_POOL.json')).read_text());groups=defaultdict(list)
        for r in pool:groups[r['spatial_group']].append(r)
        ranked=[sorted(groups[g],key=lambda r:base.digest(r['parcel'])) for g in sorted(groups,key=base.digest)]
        balanced=[group[i] for i in range(12 if use=='residential_multi' else 4) for group in ranked if i<len(group)]
        queue.extend(balanced[:240])
    assert len({r['parcel'] for r in queue})==len(queue)
    base.save(ROOT/'PARCEL_QUEUE.json',queue)
    def fetch_old(batch):
        return base.fetch('footprints_'+base.digest(','.join(batch))[:20],base.FOOTPRINT+'/query',
            dict(f='geojson',where='AIN IN ('+','.join("'"+x+"'" for x in batch)+')',
                 outFields='OBJECTID,BLD_ID,AIN,HEIGHT,AREA,DATE_,STATUS',returnGeometry='true',outSR=4326))
    ains=[r['parcel'] for r in queue]
    replies=list(ThreadPoolExecutor(max_workers=4).map(fetch_old,[ains[i:i+30] for i in range(0,len(ains),30)]))
    by=defaultdict(list)
    for d in replies:
        assert not d.get('exceededTransferLimit')
        for f in d['features']:by[f['properties']['AIN']].append(f)
    chosen=[];rejected=Counter()
    forbidden=shape(json.loads((OLD/'TEST_EXCLUSION.geojson').read_text())['features'][0]['geometry'])
    for r in queue:
        ff=by[r['parcel']]
        if len(ff)!=1:rejected['not_one_countywide_outline']+=1;continue
        f=ff[0];g=transform(base.UTM,shape(f['geometry']));near,wide=view_sizes(g.bounds,1)
        if not g.is_valid or g.geom_type!='Polygon' or not 120<=g.area<=700:rejected['geometry_or_area']+=1;continue
        ratio=r['evidence']['SQFTmain']*.09290304/g.area
        if not .65<=ratio<=3.5:rejected['floor_roof_disagreement']+=1;continue
        c=g.centroid;gx,gy=math.floor(c.x/2000),math.floor(c.y/2000)
        support=box(c.x-wide/2-10,c.y-wide/2-10,c.x+wide/2+10,c.y+wide/2+10)
        if r['spatial_group']!=f'{gx}_{gy}' or not box(gx*2000,gy*2000,(gx+1)*2000,(gy+1)*2000).contains(support) or forbidden.intersects(support):
            rejected['actual_context_overlap']+=1;continue
        chosen.append(dict(r,sample_id='la_expand_'+r['parcel'],geometry_wkt=g.wkt,geometry_crs='EPSG:32611',
                           footprint_m2=g.area,near_m=near,wide_m=wide,footprint_evidence=f['properties'],
                           parcel_outline_count=1,floor_area_to_footprint=ratio,reference_role=None,
                           reference_grade='official_single_record_outline_pending_visual_review',test_distance_m=g.distance(forbidden)+2000))
    ids=[r['footprint_evidence']['BLD_ID'] for r in chosen]
    def fetch_modern(batch):
        return base.fetch('modern_'+base.digest(','.join(batch))[:20],base.MODERN+'/query',
            dict(f='geojson',where='BLD_ID IN ('+','.join("'"+x+"'" for x in batch)+')',
                 outFields='OBJECTID,BLD_ID,DATE_,HEIGHT,AREA,STATUS',returnGeometry='true',outSR=4326))
    modern=defaultdict(list)
    for d in ThreadPoolExecutor(max_workers=4).map(fetch_modern,[ids[i:i+30] for i in range(0,len(ids),30)]):
        assert not d.get('exceededTransferLimit')
        for f in d['features']:modern[f['properties']['BLD_ID']].append(f)
    accepted=[]
    for r in chosen:
        ff=modern[r['footprint_evidence']['BLD_ID']]
        if len(ff)!=1:rejected['modern_id_missing_or_nonunique']+=1;continue
        g=transform(base.UTM,shape(ff[0]['geometry']));old=wkt.loads(r['geometry_wkt'])
        iou=g.intersection(old).area/g.union(old).area
        if iou<.8:rejected['outline_changed']+=1;continue
        accepted.append(dict(r,modern_footprint_evidence=ff[0]['properties'],outline_iou_2014_2020=iou))
    base.save(ROOT/'CANDIDATES.json',accepted)
    base.save(ROOT/'FOOTPRINT_AUDIT.json',dict(queued=len(queue),accepted=len(accepted),rejected=dict(rejected),
              counts=dict(Counter(r['reference_use'] for r in accepted)),groups=dict(Counter(r['spatial_group'] for r in accepted)),
              no_new_gold=True,labels_pending_visual_review=True))
    print('FOOTPRINTS',len(accepted),dict(Counter(r['reference_use'] for r in accepted)),dict(rejected),flush=True)


def images():
    import requests
    import rasterio
    rows=json.loads((ROOT/'CANDIDATES.json').read_text())
    def acquire(r):
        sid=r['sample_id'];g=wkt.loads(r['geometry_wkt']);c=g.centroid
        h=r['wide_m']/2+10;bounds=[c.x-h,c.y-h,c.x+h,c.y+h]
        query=dict(f='json',geometry=','.join(map(str,bounds)),geometryType='esriGeometryEnvelope',inSR=32611,
                   spatialRel='esriSpatialRelContains',where="category=1 AND year=2022 AND state='CA'",
                   outFields='objectid,name,year,state,resolution,public',returnGeometry='true',outSR=32611)
        # ArcGIS spatial predicates are query-geometry relative: use intersects then test full support locally.
        query['spatialRel']='esriSpatialRelIntersects'
        d=base.fetch(sid+'_naip_catalog',NAIP+'/query',query)
        assert not d.get('exceededTransferLimit')
        from shapely.geometry import Polygon
        options=[f for f in d['features'] if Polygon(f['geometry']['rings'][0]).contains(box(*bounds))]
        assert options,('No complete primary NAIP tile',sid)
        tile=min(options,key=lambda f:f['attributes']['objectid'])['attributes']
        date=datetime.strptime(tile['name'].split('_')[-2],'%Y%m%d').date().isoformat()
        assert date.startswith('2022-') and tile['year']==2022 and tile['public']=='yes'
        # Match the prior 256-pixel crop's physical extent; native NAIP detail remains 0.6m.
        size=math.ceil(2*h/.3051882487)
        params=dict(f='json',bbox=','.join(map(str,bounds)),bboxSR=32611,imageSR=32611,size=f'{size},{size}',
                    format='tiff',pixelType='U8',bandIds='0,1,2',interpolation='RSP_BilinearInterpolation',
                    mosaicRule=json.dumps(dict(mosaicMethod='esriMosaicLockRaster',lockRasterIds=[tile['objectid']],ascending=True)))
        path=ROOT/'acquired_images'/(sid+'.tif')
        if not path.exists():
            result=base.fetch(sid+'_naip_export',NAIP+'/exportImage',params)
            response=requests.get(result['href'],timeout=60);response.raise_for_status();path.write_bytes(response.content)
        with rasterio.open(path) as src:
            assert src.crs.to_epsg()==32611 and src.count==3 and src.dtypes==('uint8',)*3
            assert max(abs(a-b) for a,b in zip(src.bounds,bounds))<.01
        return dict(r,imagery_provider='NAIP2022',native_gsd=tile['resolution'],naip_tile=tile,
                    scene=dict(id=tile['name'],assets=dict(visual=dict(href=str(path))),properties=dict(datetime=date)))
    acquired=list(ThreadPoolExecutor(max_workers=3).map(acquire,rows))
    base.save(ROOT/'STABLE_CANDIDATES.json',acquired)
    base.images()


def sheets():
    from PIL import Image,ImageDraw
    rows=json.loads((ROOT/'STABLE_CANDIDATES.json').read_text())
    for start in range(0,len(rows),6):
        canvas=Image.new('RGB',(1100,6*310),'white');draw=ImageDraw.Draw(canvas)
        for j,r in enumerate(rows[start:start+6]):
            y=j*310
            draw.text((5,y+2),f"#{start+j+1} {r['sample_id']} {r['reference_use']} units={r['evidence']['Units']} roof={r['footprint_m2']:.0f}m2 floor/roof={r['floor_area_to_footprint']:.2f} grid={r['spatial_group']}",fill='black')
            for k,n in enumerate(['near_marked','wide_marked','legacy_rgb']):
                with Image.open(ROOT/'images'/(r['sample_id']+'_'+n+'.png')) as im:canvas.paste(im.resize((280,280)),(k*365,y+25))
        canvas.save(ROOT/'contact_sheets'/f'sheet_{start//6+1:02d}.jpg',quality=92)
    print('SHEETS',len(rows),flush=True)


if __name__=='__main__':main()
