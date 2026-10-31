"""Download public, pre-fire Assessor labels and Census variable definitions."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "analysis/20261015_gsd_assessor"
ASSESSOR = "https://services.arcgis.com/RmCCgQtiZLDCtblq/ArcGIS/rest/services/Parcel_Data_2021_Table/FeatureServer/0"


def get_json(url, params=None):
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])))
    response = session.get(url, params=params, timeout=60)
    response.raise_for_status()
    data = response.json()
    if "error" in data:
        raise RuntimeError(data["error"])
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2024)
    args = parser.parse_args()
    raw = OUT / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    target = raw / f"assessor_{args.year}.json"
    if target.exists():
        raise FileExistsError(target)
    with sqlite3.connect(f"file:{ROOT / 'evaluation/la_fire_validation/data/lariac_dins_in_region.gpkg'}?mode=ro", uri=True) as con:
        ains = sorted({row[0] for row in con.execute('SELECT AIN FROM lariac_dins_in_region') if row[0]})
    assert all(len(ain) == 10 and ain.isdigit() for ain in ains)
    schema = get_json(ASSESSOR, {"f": "json"})
    (raw / "assessor_schema.json").write_text(json.dumps(schema, indent=2))
    fields = "AIN,RollYear,UseType,UseCode,UseCodeDescChar1,UseCodeDescChar2,UseCodeDescChar3,UseCodeDescChar4,Units,totBuildingDataLines,YearBuilt,SQFTmain,OBJECTID"
    batches = [ains[i:i + 40] for i in range(0, len(ains), 40)]

    def fetch(batch):
        where = f"RollYear='{args.year}' AND AIN IN (" + ",".join(f"'{ain}'" for ain in batch) + ")"
        params = {"f": "json", "where": where, "outFields": fields, "returnGeometry": "false"}
        result = get_json(ASSESSOR + "/query", params)
        assert not result.get("exceededTransferLimit"), "Unexpected truncation"
        assert all(f["attributes"]["AIN"] in batch and str(f["attributes"]["RollYear"]) == str(args.year) for f in result["features"])
        return {"params": params, "response": result}

    responses = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, response in enumerate(pool.map(fetch, batches)):
            responses.append(response)
            if i % 10 == 0:
                print(f"[assessor] {i + 1}/{len(batches)} batches", flush=True)
    snapshot = {"source": ASSESSOR, "retrieved_utc": datetime.now(timezone.utc).isoformat(), "roll_year": args.year, "requested_ains": ains, "requests": responses}
    target.write_text(json.dumps(snapshot))
    print("[saved]", target, "sha256", hashlib.sha256(target.read_bytes()).hexdigest(), flush=True)
    for group in ["B25032", "B25033", "B25002"]:
        target = raw / f"acs2022_{group}_metadata.json"
        if not target.exists():
            result = get_json(f"https://api.census.gov/data/2022/acs/acs5/groups/{group}.json")
            target.write_text(json.dumps(result))
            print("[saved]", target, flush=True)


if __name__ == "__main__":
    main()
