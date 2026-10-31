"""Retain Harris County block-group rows from official ACS bulk tables."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json

import requests

from audit_sources import OUT


def download(group):
    url = f"https://www2.census.gov/programs-surveys/acs/summary_file/2022/table-based-SF/data/5YRData/acsdt5y2022-{group.lower()}.dat"
    target = OUT / f"raw/acs2022_{group}_harris.dat"
    if target.exists():
        return
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    digest, pending, header, retained, total = hashlib.sha256(), b"", None, [], 0
    for block in response.iter_content(chunk_size=1024 * 1024):
        digest.update(block)
        total += len(block)
        lines = (pending + block).split(b"\n")
        pending = lines.pop()
        for line in lines:
            if header is None:
                header = line
            elif line.startswith(b"1500000US48201"):
                retained.append(line)
    if pending.startswith(b"1500000US48201"):
        retained.append(pending)
    assert len(retained) == 2830, (group, len(retained))
    target.write_bytes(b"\n".join([header] + retained) + b"\n")
    (OUT / f"raw/acs2022_{group}_bulk_provenance.json").write_text(json.dumps({"url": url, "retrieved_utc": datetime.now(timezone.utc).isoformat(), "filter": "GEO_ID starts 1500000US48201 (Harris County block groups)", "retained_rows": len(retained), "downloaded_decoded_bytes": total, "downloaded_decoded_sha256": digest.hexdigest(), "retained_file_sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "etag": response.headers.get("ETag"), "last_modified": response.headers.get("Last-Modified")}, indent=2))
    print("[saved]", group, len(retained), "rows", total, "source bytes", flush=True)


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(download, ["B25033", "B25032", "B25002"]))
