"""
Handles the generation and caching of preprocessing data specificlaly with fulldays datasets.
Can be used for creating preprocessing layers, etc.
"""

import os
import hashlib
from typing import Tuple

import numpy as np

from hympi_ml.data.fulldays import FullDaysLoader, DKey

os.umask(0o007)  # correctly sets the umask to force proper file permissions


def get_save_path(loader: FullDaysLoader, key: DKey, suffix: str) -> str:
    days_str = f"{sorted(loader.days)}"

    hash_obj = hashlib.sha256(days_str.encode())
    digest = hash_obj.hexdigest()[0:8]

    return f"{loader.data_dir}/{key.name}_{digest}_{suffix}.npz"


def get_minmax(loader: FullDaysLoader, key: DKey) -> Tuple[float, float]:
    """
    Loads the cached minimum and maximum values of the dataset loaded with the provided key.
    Creates the min and max data at the "loader.data_dir" path if cached data is not found.

    Args:
        loader (FullDaysLoader): The loader being used to load the dataset
        key (DKey): The key for the dataset to create / load the min max data from

    Returns:
        Tuple[float, float]: The tuple that represents the minimuim and maximum values in the form of (min, max)
    """
    save_path = get_save_path(loader, key, "minmax")

    try:
        minmax = np.load(save_path)
        return (minmax["min"], minmax["max"])

    except OSError:
        data = loader.get_data(key).to_ndarray()
        mins = data.min(axis=0)
        maxs = data.max(axis=0)

        np.savez(save_path, min=mins, max=maxs)

        return (mins, maxs)
