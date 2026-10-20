"""Download a pinned asset subset for three local seed-0 smoke tasks."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time
from urllib.parse import quote

import requests
from huggingface_hub.file_download import xet_get
from huggingface_hub.utils._xet import parse_xet_file_data_from_response

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "third_party/RoboDojo"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "sim/setup/robodojo/assets-smoke3-manifest.json").read_text())
    revision = manifest["revision"]
    selected = manifest["files"]
    print(f"Selected {len(selected)} files, {manifest['total_bytes']/1e6:.1f} MB", flush=True)
    if not args.download:
        return
    lock = threading.Lock()
    stop = threading.Event()
    counters = {"bytes": 0, "done": 0}
    children = []

    def abort(code, message):
        stop.set()
        print(message, flush=True)
        with lock:
            running = list(children)
        for process in running:
            if process.poll() is None:
                process.terminate()
        os._exit(code)  # Stop native Xet threads as well; retain partial files.

    class Progress:
        def update(self, size):
            with lock:
                counters["bytes"] += size

    def valid(path, row):
        if not path.is_file() or path.stat().st_size != row["size"]:
            return False
        data = path.read_bytes()
        lfs = row.get("lfs", {})
        digest = hashlib.sha256(data).hexdigest() if lfs else hashlib.sha1(
            f"blob {len(data)}\0".encode() + data).hexdigest()
        return digest == (lfs.get("oid") or row["oid"])

    def download(row):
        if stop.is_set():
            return
        target = NATIVE / row["path"]
        if not valid(target, row):
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + ".download-partial")
            url = f"https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo/resolve/{revision}/{quote(row['path'])}"
            if row.get("lfs"):
                metadata = requests.head(url, allow_redirects=False, timeout=30)
                metadata.raise_for_status()
                xet = parse_xet_file_data_from_response(metadata)
                if xet is None:
                    raise RuntimeError(f"Missing Xet metadata for {row['path']}")
                xet_get(incomplete_path=partial, xet_file_data=xet, headers={},
                        expected_size=row["size"], displayed_filename=target.name,
                        _tqdm_bar=Progress())
            else:
                with subprocess.Popen(
                    ["curl", "--http1.1", "-sSfL", "--connect-timeout", "20", "--max-time", "180", url],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                ) as response:
                    with lock:
                        children.append(response)
                    with partial.open("wb") as stream:
                        while chunk := response.stdout.read(1024 * 1024):
                            if stop.is_set():
                                response.terminate()
                                return
                            stream.write(chunk)
                            with lock:
                                counters["bytes"] += len(chunk)
                    error = response.stderr.read().decode(errors="replace")
                    if response.wait():
                        raise RuntimeError(f"curl failed for {row['path']}: {error[:300]}")
            if not valid(partial, row):
                raise RuntimeError(f"Asset checksum mismatch: {row['path']}")
            partial.replace(target)
        with lock:
            counters["done"] += 1

    started = time.monotonic()
    checkpoint, previous = started, 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(download, row) for row in selected]
        while not all(f.done() for f in futures):
            time.sleep(5)
            errors = [f.exception() for f in futures if f.done() and f.exception()]
            if errors:
                abort(2, f"Download stopped: {errors[0]}")
            now = time.monotonic()
            with lock:
                count, size = counters["done"], counters["bytes"]
            print(f"{count}/{len(selected)} files, {size/1e6:.1f} MB, avg {size/1e6/(now-started):.2f} MB/s", flush=True)
            if now - checkpoint >= 60:
                rate = (size - previous) / (now - checkpoint)
                if rate < 1e6:
                    abort(3, f"Download paused: sustained throughput {rate/1e6:.2f} MB/s is below workspace 1 MB/s rule")
                checkpoint, previous = now, size
        for future in futures:
            future.result()
    print("ASSET_SUBSET_VERIFIED", flush=True)


if __name__ == "__main__":
    main()
