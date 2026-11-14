import os
import urllib.request

from lib.utils import KEYRECS_PATH

KEYRECS_URL = "https://zenodo.org/records/7886743/files/fixed-text.csv?download=1"


def download_keyrecs_if_missing() -> str:
    os.makedirs(os.path.dirname(KEYRECS_PATH), exist_ok=True)
    if not os.path.exists(KEYRECS_PATH):
        print(f"Downloading KeyRecs fixed-text dataset from {KEYRECS_URL}...")
        urllib.request.urlretrieve(KEYRECS_URL, KEYRECS_PATH)
        print(f"Dataset saved to {KEYRECS_PATH} ({os.path.getsize(KEYRECS_PATH)} bytes).")
    else:
        print(f"KeyRecs dataset already present at {KEYRECS_PATH}.")
    return KEYRECS_PATH


if __name__ == "__main__":
    download_keyrecs_if_missing()
