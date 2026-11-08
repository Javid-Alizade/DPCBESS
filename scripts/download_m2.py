"""Milestone M2: download the open datasets listed in configs/m2.json into data/raw/.

Run:  python scripts/download_m2.py
Writes results/m2_download.json with the size, MD5 and SHA-256 of every file.

If a file cannot be downloaded or its published MD5 does not match, the script stops with an error.
It never creates a replacement for missing data. A finished file is checked and not downloaded again.
"""
import hashlib
import http.client
import json
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "m2.json"
RAW = ROOT / "data" / "raw"
REPORT = ROOT / "results" / "m2_download.json"
CHUNK = 1024 * 1024  # 1 MB


def hashes(path):
    """MD5 and SHA-256 of a file, read in pieces so a 3 GB file does not fill the memory."""
    md5, sha = hashlib.md5(), hashlib.sha256()
    with open(path, "rb") as f:
        while piece := f.read(CHUNK):
            md5.update(piece)
            sha.update(piece)
    return md5.hexdigest(), sha.hexdigest()


def download(url, target, max_attempts=20):
    """Download to a .part file and continue from where it stopped if the connection drops.

    The file gets its final name only when its size equals the size announced by the server, so an
    interrupted download never looks like a finished file. The MD5 check in main() comes after this.
    """
    part = target.with_name(target.name + ".part")
    for attempt in range(1, max_attempts + 1):
        done = part.stat().st_size if part.exists() else 0
        headers = {"User-Agent": "dpc-grid-data-download"}
        if done:
            headers["Range"] = f"bytes={done}-"
        total, finished = None, False
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as response:
                resumed = response.status == 206
                length = response.headers["Content-Length"]  # some servers do not announce the size
                if length is not None:
                    total = int(length) + (done if resumed else 0)
                with open(part, "ab" if resumed else "wb") as out:  # a server without resume support restarts the file
                    shutil.copyfileobj(response, out, CHUNK)
            finished = True
        except (OSError, http.client.HTTPException) as error:
            print(f"  attempt {attempt}: {error}", flush=True)
        size = part.stat().st_size if part.exists() else 0
        print(f"  attempt {attempt}: {size / 1e6:.0f} MB received, server announced "
              f"{'no size' if total is None else f'{total / 1e6:.0f} MB'}", flush=True)
        # Without an announced size, only a transfer that ended without an error counts. main() then tests the zip.
        if size == total or (total is None and finished):
            part.replace(target)
            return
        time.sleep(5)
    raise RuntimeError(f"{target.name}: not complete after {max_attempts} attempts. Partial file kept as {part}.")


def main():
    config = json.loads(CONFIG.read_text())
    report = {}
    for dataset, source in config["sources"].items():
        folder = RAW / dataset
        folder.mkdir(parents=True, exist_ok=True)
        for name, spec in source["files"].items():
            target = folder / name
            if target.exists():
                print(f"{name}: already here, checking it", flush=True)
            else:
                print(f"{name}: downloading from {spec['url']}", flush=True)
                download(spec["url"], target)
            md5, sha = hashes(target)
            if spec["md5"] is not None and md5 != spec["md5"]:
                raise RuntimeError(f"{name}: MD5 is {md5}, the source publishes {spec['md5']}. "
                                   f"The file is left in {folder} for inspection and must not be used.")
            verified = spec["md5"] is not None
            print(f"{name}: {target.stat().st_size / 1e6:.1f} MB, "
                  f"{'MD5 matches the source' if verified else 'the source publishes no checksum'}", flush=True)
            report[name] = {"dataset": dataset, "url": spec["url"], "bytes": target.stat().st_size,
                            "md5": md5, "sha256": sha, "md5_checked_against_source": verified}
            if target.suffix == ".zip":  # the zip stores a CRC of every member, which detects a damaged file
                with zipfile.ZipFile(target) as archive:
                    bad = archive.testzip()
                    if bad is not None:
                        raise RuntimeError(f"{name}: member {bad} fails its CRC check. The file must not be used.")
                    report[name]["zip_members"] = {i.filename: i.file_size for i in archive.infolist()}
                print(f"{name}: zip CRC check passed, members {report[name]['zip_members']}", flush=True)
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2))
    print(f"Saved {REPORT}", flush=True)


if __name__ == "__main__":
    main()
