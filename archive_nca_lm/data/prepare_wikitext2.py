"""
Data Ingestion Script for WikiText-2 (Raw).
Downloads train, valid, and test splits and validates their integrity.
"""

import os
import sys
import zipfile
import urllib.request
from pathlib import Path

WIKITEXT2_ZIP_URL = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip"
GITHUB_FALLBACK_BASE = "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/"


def download_url(url: str, dest_path: Path) -> bool:
    try:
        print(f"Downloading {url} -> {dest_path}...")
        headers = {'User-Agent': 'Mozilla/5.0'}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as response, open(dest_path, 'wb') as out_file:
            out_file.write(response.read())
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}", file=sys.stderr)
        if dest_path.exists():
            dest_path.unlink()
        return False


def prepare_wikitext2(raw_dir: str = "data/raw") -> dict:
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": raw_path / "train.txt",
        "valid": raw_path / "valid.txt",
        "test": raw_path / "test.txt",
    }

    # Check if all files already exist and are non-empty
    if all(p.exists() and p.stat().st_size > 100_000 for p in splits.values()):
        print("WikiText-2 raw files already present and verified.")
        return {k: str(v) for k, v in splits.items()}

    # Try downloading official zip archive first
    zip_path = raw_path / "wikitext-2-raw-v1.zip"
    if download_url(WIKITEXT2_ZIP_URL, zip_path):
        try:
            print(f"Extracting {zip_path}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                for member in zip_ref.namelist():
                    if "wiki.train.raw" in member:
                        with open(splits["train"], "wb") as f:
                            f.write(zip_ref.read(member))
                    elif "wiki.valid.raw" in member:
                        with open(splits["valid"], "wb") as f:
                            f.write(zip_ref.read(member))
                    elif "wiki.test.raw" in member:
                        with open(splits["test"], "wb") as f:
                            f.write(zip_ref.read(member))
            zip_path.unlink()
        except Exception as e:
            print(f"Failed to extract zip: {e}", file=sys.stderr)

    # Fallback to direct raw file downloads if zip extraction did not produce all files
    if not all(p.exists() and p.stat().st_size > 100_000 for p in splits.values()):
        print("Falling back to direct split downloads...")
        for name, file_path in splits.items():
            if not (file_path.exists() and file_path.stat().st_size > 100_000):
                url = f"{GITHUB_FALLBACK_BASE}{name}.txt"
                if not download_url(url, file_path):
                    raise RuntimeError(f"Could not download split {name} from {url}")

    # Validate line counts and sizes
    stats = {}
    for name, p in splits.items():
        size = p.stat().st_size
        with open(p, "r", encoding="utf-8") as f:
            lines = sum(1 for _ in f)
        stats[name] = {"path": str(p), "size_bytes": size, "lines": lines}
        print(f"Split [{name}]: {lines:,} lines, {size / 1024 / 1024:.2f} MB")
        if size == 0 or lines == 0:
            raise ValueError(f"Corrupted or empty file for split {name} at {p}")

    return {k: str(v) for k, v in splits.items()}


if __name__ == "__main__":
    prepare_wikitext2()
