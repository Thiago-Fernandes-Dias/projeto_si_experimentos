import logging
import os
import re
from datetime import datetime
from typing import Callable, TypeVar

import numpy as np
import pandas as pd

T = TypeVar('T')
S = TypeVar('S')

exclude_hold_times_pt: str = "((D|U)D)(.[a-zA-Z]+)+"


def n_copies(element: T, n: int) -> list[T]:
    return [element] * n


def data_frame_length(df: pd.DataFrame) -> int:
    return df.shape[0]


def sorted_nicely(l: list[str]):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def create_labels(df: pd.DataFrame, label: T) -> list[T]:
    return n_copies(label, data_frame_length(df))


def select(iterable: list[T], f: Callable[[T], S]) -> list[S]:
    return [f(x) for x in iterable]


def item_with_max_value(d: dict[T, float], comp: Callable[[float, float], int]) -> tuple[T, float]:
    max_item, max_value = d.popitem()
    for (item, value) in d.items():
        if comp(value, max_value) == 1:
            max_value = value
            max_item = item
    return max_item, max_value


def bigger_comp(a: float, b: float) -> int:
    if a > b:
        return 1
    elif a == b:
        return 0
    return -1


def lower_comp(a: float, b: float) -> int:
    if a > b:
        return -1
    elif a == b:
        return 0
    return 1


def create_dir_if_not_exists(name: str):
    if not os.path.exists(name):
        os.makedirs(name)


def log_completion(start_time: datetime, msg: str, logger: logging.Logger) -> None:
    time_elapsed = datetime.now() - start_time
    seconds_elapsed = time_elapsed.total_seconds()
    logger.info(f"{msg} Time elapsed: {seconds_elapsed} seconds")


def cmu_default_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return df[(df['sessionIndex'] == 1)], df[(df['sessionIndex'] != 1)]


def cmu_test_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return df[(df['sessionIndex'] == 1) & (df['rep'] <= 5)], df[(df['sessionIndex'] != 1) & (df['rep'] <= 5)]


def keyrecs_default_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return df[(df['session'] == 1) & (df['repetition'] <= 50)], df[(df['session'] == 2) | (df['repetition'] > 50)]


def keyrecs_test_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return df[(df['session'] == 1) & (df['repetition'] <= 5)], df[(df['session'] == 2) & (df['repetition'] <= 5)]


def interleave_preserving_chronology(
    genuine_df: pd.DataFrame,
    impostor_df: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Interleaves genuine and impostor rows at random positions while strictly
    preserving the internal chronological sequence of each group.
    """
    genuine_count = len(genuine_df)
    impostor_count = len(impostor_df)
    total_count = genuine_count + impostor_count
    if total_count == 0:
        return pd.DataFrame()
    if impostor_count == 0:
        return genuine_df.reset_index(drop=True)
    if genuine_count == 0:
        return impostor_df.reset_index(drop=True)

    genuine_positions = set(rng.choice(total_count, size=genuine_count, replace=False).tolist())
    genuine_rows = [row for _, row in genuine_df.iterrows()]
    impostor_rows = [row for _, row in impostor_df.iterrows()]

    genuine_idx = 0
    impostor_idx = 0
    ordered_rows: list[pd.Series] = []
    for pos in range(total_count):
        if pos in genuine_positions:
            ordered_rows.append(genuine_rows[genuine_idx])
            genuine_idx += 1
        else:
            ordered_rows.append(impostor_rows[impostor_idx])
            impostor_idx += 1

    return pd.DataFrame(ordered_rows).reset_index(drop=True)


default_seeds_range = range(0, 5)
cmu_split = cmu_default_split
keyrecs_split = keyrecs_default_split
IMPOSTOR_LABEL = -1
GENUINE_LABEL = 1
CMU_PATH = "datasets/cmu/dataset.csv"
KEYRECS_PATH = "datasets/keyrecs/dataset.csv"
