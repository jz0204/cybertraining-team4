"""Build the LA viewer from frozen C predictions and the downloaded fire report."""
import base64
import csv
import hashlib
import io
import json
import re
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFont
from rasterio.warp import transform
from rasterio.windows import Window
from shapely import wkt

STAGE=Path('/Volumes/Crucial X9/Instance-Impact-IGUIDE-SpatialAIChallenge2026/II_package/outputs/stage2a')
ROOT=STAGE/'la_best_preview_20261031'
OUT=ROOT/'viewer'
HERE=Path(__file__).resolve().parent
ORIGINAL=Path('/Users/zhou/Downloads/LA_fire_results_page_2026-10.html')
CELLS=['cell_00365','cell_00471','cell_00507']
COLORS=['#3e9ed8','#a879e9','#edaa45']


def main():
    OUT.mkdir(exist_ok=True); (OUT/'assets').mkdir(exist_ok=True)
    data=read(ROOT/'DATASET.json'); protocol=read(ROOT/'PROTOCOL.json')
    assert digest(ROOT/'DATASET.json')==protocol['dataset_sha256']
    products=list(csv.DictReader((STAGE/'la_eval_20261023/inputs/building_damage_all_cells.csv').open()))
    assert ['la_'+r['cell_id']+'_'+r['bldg_uid'] for r in products]==[r['sample_id'] for r in data]
    original_text=ORIGINAL.read_text(); original_data=json.loads(re.search(r'window.__DATA__=(.*?);\s*</script>',original_text,re.S).group(1))
    assert len(original_data['la_points'])==len(products)==22024
    assert all(abs(float(r['centroid_lon'])-point[0])<.00001 and abs(float(r['centroid_lat'])-point[1])<.00001
               and int(r['m2b_damage_class'])==point[2] for r,point in zip(products,original_data['la_points']))
    references={r['sample_id']:r for r in read(STAGE/'la_transfer_20261027/dataset.json')}
    repairs={r['sample_id']:r for r in read(ROOT/'REPAIRS.json')}
    cells=sorted({r['cell_id'] for r in products}); members=[]; receipts={}
    for seed in [42,43]:
        folder=ROOT/f'execution/collected/runs/seed{seed}'
        result=read(folder/'RESULT.json'); predictions=read(folder/'predictions.json'); witness=read(folder/'WITNESS.json')
        assert result['dataset_sha256']==protocol['dataset_sha256'] and result['n']==22024
        assert digest(folder/'predictions.json')==result['predictions_sha256']
        assert result['checkpoint_sha256']==protocol['checkpoints'][str(seed)]['adapted_sha256']
        assert [r['sample_id'] for r in predictions]==[r['sample_id'] for r in data]
        members.append(np.asarray([r['probabilities'] for r in predictions]));receipts[str(seed)]=dict(result=result,witness=witness)
    probabilities=np.mean(members,axis=0); assert probabilities.shape==(22024,3) and np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(1),1,atol=1e-5)
    saved={r['sample_id']:r for r in read(HERE/'analysis/20261030_hard_ablation/PREDICTIONS.json')['C']}
    error=max(float(np.max(np.abs(probabilities[i]-saved[r['sample_id']]['probabilities']))) for i,r in enumerate(data) if r['sample_id'] in saved)
    assert error<3e-4
    assert all(int(probabilities[i].argmax())==['residential_small','residential_multi','commercial'].index(saved[r['sample_id']]['use'])
               for i,r in enumerate(data) if r['sample_id'] in saved)
    exposure=read(ROOT/'EXPOSURE.json'); estimates=exposure['records']
    assert [r['sample_id'] for r in estimates]==[r['sample_id'] for r in data]
    records=[]; geometry={}; table=[]
    for i,(r,p) in enumerate(zip(products,probabilities)):
        sid=data[i]['sample_id']; ref=references.get(sid)
        estimate=estimates[i]
        assert np.allclose(estimate['probabilities'],p,atol=1e-12)
        assert estimate['footprint_m2']==float(data[i]['footprint_m2'])
        value=estimate['estimated_exposure']
        assert np.isfinite(value) and value>=0
        label=int(p.argmax()); d=int(r['m2b_damage_class']); repaired=sid in repairs
        reference=['residential_small','residential_multi','commercial'].index(ref['reference_use']) if ref else -1
        geometry[i]=wkt.loads(repairs[sid]['recovered_geometry_wkt'] if repaired else r['polygon_wkt_utm'])
        assert geometry[i].area>0
        lon,lat=float(r['centroid_lon']),float(r['centroid_lat'])
        if repaired:
            center=geometry[i].centroid
            xs,ys=transform(repairs[sid]['crs'],'EPSG:4326',[center.x],[center.y])
            lon,lat=xs[0],ys[0]
        records.append([r['bldg_uid'],cells.index(r['cell_id']),lon,lat,
                        [round(float(v),7) for v in p],label,d,round(float(data[i]['footprint_m2']),2),reference,int(repaired),value])
        table.append(dict(sample_id=sid,cell_id=r['cell_id'],building_uid=r['bldg_uid'],
            predicted_use=['residential_small','residential_multi','commercial'][label],
            prob_residential_small=float(p[0]),prob_residential_multi=float(p[1]),prob_commercial=float(p[2]),
            estimated_exposure=value,exposure_in_damaged_building=value if d in [1,2,3] else 0,existing_damage_vote=d,footprint_m2=data[i]['footprint_m2'],input_status=data[i]['input_status']))
        local=STAGE/data[i]['crop_path'].removeprefix('/stage3/').removeprefix('stage2a_')
        assert local.is_file(),local
    # The old report transformed 136 pixel centroids as UTM, placing them near the equator.
    assert all(-118.7<r[2]<-118 and 34<r[3]<34.3 for r in records)
    for name,target in [('inputs',STAGE/'la_eval_20261023/inputs'),('recovered',STAGE/'la_full_20261030/images')]:
        link=OUT/'assets'/name
        if link.is_symlink(): assert link.resolve()==target
        else: link.symlink_to(target,target_is_directory=True)
    images=ReportImages();images.feed(original_text)
    scenes=[]
    for cell in CELLS:
        ref=next(r for r in references.values() if r['cell_id']==cell);origin=ref['origin'];gsd=ref['original_gsd']
        src_path=STAGE/'la_transfer_20261027/source_cells'/(cell+'.tif')
        with rasterio.open(src_path) as src:
            col,row=(~src.transform)*(origin[0],origin[1]);assert abs(col-round(col))<1e-5 and abs(row-round(row))<1e-5
            rgb=src.read([1,2,3],window=Window(round(col),round(row),1966,1966))
            assert rgb.shape==(3,1966,1966)
        pre=Image.fromarray(rgb.transpose(1,2,0)); shapes=[]
        for i,r in enumerate(products):
            if r['cell_id']!=cell:continue
            g=geometry[i];parts=list(g.geoms) if g.geom_type=='MultiPolygon' else [g]
            rings=[[[round((x-origin[0])/600,6),round((origin[1]-y)/600,6)] for x,y in ring.coords]
                   for part in parts for ring in [part.exterior,*part.interiors]]
            shapes.append([i,rings])
        assert shapes and all(-.05<=v<=1.05 for _,rings in shapes for ring in rings for xy in ring for v in xy)
        if cell!='cell_00507':
            post=images.values[f'Cell {cell[-5:]} damage overlay on the post-fire image']
            date='19 Jan 2025' if cell=='cell_00365' else 'Saved v3 post-fire image'
            raw_post=False
        else:
            path=STAGE.parents[1]/'la_fire_results/demo_cell_00507/cell_00507_post_20250115.tif'
            with rasterio.open(path) as src:
                assert src.width==src.height==1966 and abs(src.transform.c-origin[0])<.001 and abs(src.transform.f-origin[1])<.001
                post=jpeg(Image.fromarray(src.read([1,2,3]).transpose(1,2,0)))
            date='15 Jan 2025';raw_post=True
        scenes.append(dict(cell=cell,pre=jpeg(pre),post=post,postDate=date,rawPost=raw_post,shapes=shapes))
        preview=pre.resize((1000,1000));draw=ImageDraw.Draw(preview,'RGBA')
        for i,rings in shapes:
            color=COLORS[records[i][5]]
            for ring in rings:draw.polygon([(x*1000,y*1000) for x,y in ring],fill=color+'55',outline=color)
        preview.save(OUT/'assets'/(cell+'_current_use.jpg'),quality=92)
    counts=np.bincount(probabilities.argmax(1),minlength=3).tolist(); damage=Counter(int(r['m2b_damage_class']) for r in products)
    cm=np.zeros((3,3),dtype=int)
    for r in records:
        if r[8]>=0:cm[r[8],r[5]]+=1
    assert cm.tolist()==[[3242,15,19],[10,3,1],[12,0,30]]
    payload=dict(records=records,cells=cells,scenes=scenes,useCounts=counts,damageCounts=dict(damage),
                 sourceDate='31 Oct 2026',referenceConfusion=cm.tolist(),exposure={k:v for k,v in exposure.items() if k!='records'})
    template=(HERE/'la_best_preview.html').read_text()
    assert template.count('__LA_DATA__')==1
    (OUT/'index.html').write_text(template.replace('__LA_DATA__',json.dumps(payload,separators=(',',':')).replace('</','<\\/')))
    first=Image.open(OUT/'assets/cell_00365_current_use.jpg').resize((800,800))
    second=Image.open(io.BytesIO(base64.b64decode(scenes[0]['post'].split(',',1)[1]))).convert('RGB').resize((800,800))
    figure=Image.new('RGB',(1632,900),'#f3f5f4');draw=ImageDraw.Draw(figure)
    font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',22)
    small=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',17)
    draw.text((12,13),'Pre-fire: current building use (C ensemble)',font=font,fill='#17201d')
    draw.text((820,13),'Post-fire: saved v3 damage',font=font,fill='#17201d')
    figure.paste(first,(12,50));figure.paste(second,(820,50))
    for x,color,label in [(12,COLORS[0],'Small residential'),(215,COLORS[1],'Multifamily'),(385,COLORS[2],'Commercial'),
                          (820,'#3f9e6e','No damage'),(980,'#d99a2b','Minor'),(1095,'#d8642a','Major'),(1210,'#b8322a','Destroyed')]:
        draw.rectangle((x,867,x+12,879),fill=color);draw.text((x+20,862),label,font=small,fill='#17201d')
    figure.save(OUT/'preview.jpg',quality=92)
    with (OUT/'predictions.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    report=dict(n=len(records),cells=len(cells),use_counts=dict(zip(['residential_small','residential_multi','commercial'],counts)),
        damage_counts=dict(damage),reference_n=len(saved),reference_confusion=cm.tolist(),max_reference_probability_error=error,
        original_html_sha256=digest(ORIGINAL),input_dataset_sha256=protocol['dataset_sha256'],inference=receipts,
        damage_source='Unchanged v3 M2b product, matched to downloaded HTML on every record',
        exposure_reestimated=True,exposure={k:v for k,v in exposure.items() if k!='records'},examples=CELLS,geometry_repairs=len(repairs),
        map_centroid_repairs=len(repairs),map_coordinates_checked=len(records),
        source_sha256={n:digest(HERE/n) for n in ['build_la_best_preview.py','la_best_preview.html']})
    (ROOT/'VIEWER_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n')
    (OUT/'README.md').write_text('''# LA fire: current Stage2a preview

Open index.html. It embeds all 22,024 C-ensemble predictions, three pre-fire scene images, and the saved post-fire illustrations. Building crops load lazily through two local asset links; keep the external data drive mounted. Predictions can be downloaded as CSV from the page.

Stage2a is the C_positive_negative two-seed ensemble. The original SAM3 outlines and v3 damage votes are reused. The current Stage2a model does not produce the damage layer. Exposure is recomputed with current C probabilities and the frozen B_semantics_mean3 OOF exposure head. It is an area/parcel-use/ACS population proxy, not observed occupancy. This head has not been validated with C probabilities on LA. Damaged-building exposure sums minor/major/destroyed records and is not a casualty or displacement estimate. Reference labels are the existing assessor-derived development cohort, not independent building-use gold. The 136 previously repaired input geometries retain their audit flag.

The geographic maps use centroids of the recovered geometries for 136 repaired records; the old report's coordinates placed these records near the equator. The other 21,888 centroids are unchanged. Scene outlines use the same recovered geometries. No class predictions or damage votes were changed by this map correction. Cell 00365 includes 405 records, including three previously repaired input geometries; the original post-fire illustration shows 402 footprints. The saved post-fire illustrations predate these geometry repairs. Probability bars are model scores, not calibrated confidence guarantees.
''')
    print('VIEWER_READY',OUT/'index.html',json.dumps({k:report[k] for k in ['n','cells','use_counts','damage_counts','max_reference_probability_error']}))


class ReportImages(HTMLParser):
    def __init__(self):super().__init__();self.values={}
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if tag=='img' and a.get('alt') and a.get('src','').startswith('data:image/'):
            self.values[a['alt']]=a['src']


def jpeg(image):
    image=image.resize((1100,1100),Image.Resampling.LANCZOS);stream=io.BytesIO();image.save(stream,format='JPEG',quality=88)
    return 'data:image/jpeg;base64,'+base64.b64encode(stream.getvalue()).decode()


def read(path):return json.loads(path.read_text())
def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


if __name__=='__main__':main()
