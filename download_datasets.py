import os
import urllib.request

from lib.utils import CMU_PATH, KEYRECS_PATH

CMU_URL = "https://www.cs.cmu.edu/~keystroke/DSL-StrongPasswordData.csv"
KEYRECS_URL = "https://zenodo.org/records/7886743/files/fixed-text.csv?download=1"

DATASETS = [
    ("CMU Keystroke Benchmark", CMU_URL, CMU_PATH),
    ("KeyRecs fixed-text", KEYRECS_URL, KEYRECS_PATH),
]


def _download_if_missing(name: str, url: str, dest_path: str) -> str:
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if not os.path.exists(dest_path):
        print(f"Downloading {name} dataset from {url}...")
        urllib.request.urlretrieve(url, dest_path)
        print(f"Dataset saved to {dest_path} ({os.path.getsize(dest_path)} bytes).")
    else:
        print(f"{name} dataset already present at {dest_path}.")
    return dest_path


def download_datasets_if_missing() -> None:
    for name, url, dest_path in DATASETS:
        _download_if_missing(name, url, dest_path)


if __name__ == "__main__":
    download_datasets_if_missing()
