"""
Data Downloader Module
======================
Handles downloading raw Food dataset files (recipes metadata and user interactions)
sourced from the Food.com benchmark datasets referenced in RUCAIBox/RecSysDatasets.
Provides automatic caching and fallback mock generation for offline development/testing.
"""

import os
import urllib.request
import logging
import pandas as pd

logger = logging.getLogger(__name__)

INTERACTIONS_URL = "https://huggingface.co/datasets/peterpeeterspeter/recipe-interactions/resolve/main/data/train-00000-of-00001.parquet"
RECIPES_URL = "https://huggingface.co/datasets/Vatazh0k/recipes_data_food.com/resolve/main/recipes.csv"


def download_file(url: str, dest_path: str, chunk_size: int = 1024 * 1024) -> str:
    """Download a file with streaming chunks and logging."""
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        logger.info(f"File already exists: {dest_path} ({os.path.getsize(dest_path):,} bytes)")
        return dest_path

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    logger.info(f"Downloading from {url} -> {dest_path}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    with urllib.request.urlopen(req) as response, open(dest_path, "wb") as out_file:
        total_size = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                percent = (downloaded / total_size) * 100
                if downloaded % (5 * chunk_size) < chunk_size:
                    logger.info(f"Progress: {downloaded / (1024*1024):.1f}MB / {total_size / (1024*1024):.1f}MB ({percent:.1f}%)")

    logger.info(f"Download complete: {dest_path} ({os.path.getsize(dest_path):,} bytes)")
    return dest_path


def download_food_dataset(raw_dir: str = "data/raw") -> dict:
    """Download both interactions and recipes dataset files."""
    os.makedirs(raw_dir, exist_ok=True)
    inter_path = os.path.join(raw_dir, "interactions.parquet")
    recipes_path = os.path.join(raw_dir, "recipes.csv")

    try:
        download_file(INTERACTIONS_URL, inter_path)
    except Exception as e:
        logger.error(f"Failed to download interactions: {e}")
        raise

    try:
        download_file(RECIPES_URL, recipes_path)
    except Exception as e:
        logger.error(f"Failed to download recipes: {e}")
        raise

    return {
        "interactions": inter_path,
        "recipes": recipes_path
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    res = download_food_dataset()
    print("Downloaded files:", res)
