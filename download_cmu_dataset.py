import os
import urllib.request

CMU_URL = "https://www.cs.cmu.edu/~keystroke/DSL-StrongPasswordData.csv"
DEST_PATH = "datasets/cmu/dataset.csv"


def download_cmu_if_missing() -> str:
    os.makedirs(os.path.dirname(DEST_PATH), exist_ok=True)
    if not os.path.exists(DEST_PATH):
        print(f"Downloading CMU Keystroke Benchmark dataset from {CMU_URL}...")
        urllib.request.urlretrieve(CMU_URL, DEST_PATH)
        print(f"Dataset saved to {DEST_PATH} ({os.path.getsize(DEST_PATH)} bytes).")
    else:
        print(f"CMU dataset already present at {DEST_PATH}.")
    return DEST_PATH


if __name__ == "__main__":
    download_cmu_if_missing()
