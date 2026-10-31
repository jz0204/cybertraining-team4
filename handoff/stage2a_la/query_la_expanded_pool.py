import sys,json,math
from collections import Counter
from shapely.geometry import shape,box
sys.path.insert(0,'/Users/zhou/Documents/GitHub/cybertraining-team4/handoff/stage2a_la')
import prepare_la_adaptation as b
from evaluate_assessor import map_use
old=b.ROOT;b.ROOT=b.STAGE/'la_hard_expansion_20261030'
forbidden=shape(json.loads((old/'TEST_EXCLUSION.geojson').read_text())['features'][0]['geometry'])
used={r['parcel'] for r in json.loads((old/'PARCEL_POOL.json').read_text())}
base="RollYear='2024' AND totBuildingDataLines=1 AND YearBuilt>='1850' AND YearBuilt<='2014' AND CENTER_LON BETWEEN -118.65 AND -117.9 AND CENTER_LAT BETWEEN 33.78 AND 34.30"
clauses={'residential_multi':"(UseCode LIKE '04%' OR UseCode LIKE '05%') AND Units BETWEEN 4 AND 8 AND SQFTmain BETWEEN 1000 AND 8000",'residential_small':"UseCode LIKE '01%' AND Units=1 AND SQFTmain BETWEEN 2500 AND 8500 AND MOD(OBJECTID,10)=0"}
for use,clause in clauses.items():
 where=base+' AND '+clause;count=b.fetch(use+'_expanded_count',b.ASSESSOR+'/query',dict(f='json',where=where,returnCountOnly='true'))['count'];print(use,'query',count,flush=True);allrows=[]
 for offset in range(0,count,2000):
  d=b.fetch(use+f'_expanded_{offset:06}',b.ASSESSOR+'/query',dict(f='json',where=where,outFields=b.FIELDS+',PropertyLocation,Bedrooms,Bathrooms',returnGeometry='false',orderByFields='AIN',resultOffset=offset,resultRecordCount=2000));allrows.extend(f['attributes'] for f in d['features'])
 assert len(allrows)==count and len({r['AIN'] for r in allrows})==count
 rows=[];rejected=Counter()
 for r in allrows:
  if r['AIN'] in used or map_use(r)!=use:rejected['used_or_unmapped']+=1;continue
  x,y=b.UTM(r['CENTER_LON'],r['CENTER_LAT']);gx,gy=math.floor(x/2000),math.floor(y/2000)
  if min(x%2000,y%2000,2000-x%2000,2000-y%2000)<250:rejected['boundary']+=1;continue
  if int(b.digest((gx,gy))[:8],16)%5 in [0,1]:rejected['val_or_reserved_grid']+=1;continue
  if forbidden.intersects(box(x-250,y-250,x+250,y+250)):rejected['test_buffer']+=1;continue
  rows.append(dict(parcel=r['AIN'],evidence=r,spatial_group=f'{gx}_{gy}',split='train',reference_use=use))
 b.save(b.ROOT/(use+'_EXPANDED_POOL.json'),rows);print(use,'eligible',len(rows),'grids',len({r['spatial_group'] for r in rows}),'reject',dict(rejected),flush=True)
