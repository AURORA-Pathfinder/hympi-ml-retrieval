from typing import List
from enum import Enum, auto
import math

import numpy as np
import tensorflow as tf

from hympi_ml.data.memmap import MemmapSequence


class DKey(str, Enum):
    """
    An Enum that contains the list of datasets that could be loaded.
    """

    # ensures values are all equal to the string version of the enum name
    def _generate_next_value_(name, *_):  # noqa: N805
        return name

    HSEL = auto()
    H1 = auto()
    HA = auto()
    HB = auto()
    HC = auto()
    HD = auto()
    HW = auto()

    ATMS = auto()

    CPL = auto()

    NATURE_SCALAR = auto()
    LATITUDE = auto()
    LONGITUDE = auto()
    PBLH = auto()
    LAND_FRACTION = auto()
    SURFACE_PRESSURE = auto()
    SURFACE_TEMPERATURE = auto()

    NATURE_TABLE = auto()
    PRESSURE = auto()
    TEMPERATURE = auto()
    WATER_VAPOR = auto()


class DPath(Enum):
    """
    An Enum that contains the list of. paths where data can be loaded from
    """

    ALL_06 = "/data/nature_run/fulldays"
    "The entire 2006 dataset."

    CPL_06 = "/data/nature_run/fulldays_cpl"
    "A subset of ALL_06 with CPL_Flag = 1."

    CPL_06_REDUCED = "/data/nature_run/fulldays_reduced"
    "A subset of ALL_06 with CPL_Flag = 1 and cloud fraction <= 0.1."


class FullDaysLoader:
    """
    A base class for loading days worth of data initialized with a given number of days in the YYYYMMDD format.
    """

    def __init__(self, days: List[str], data_path: DPath) -> None:
        """Initialize a loader with a list of days

        Args:
            days (List[str]): the list of days to read from (format: YYYYMMDD)
        """
        self.days = days
        self.dpath = data_path

    @property
    def data_dir(self) -> str:
        return self.dpath.value

    def get_data(self, key: DKey | str) -> MemmapSequence:
        """
        Returns a MemmapSequence from a given DKey on the days initialized in this loader.
        """
        # convert dataName from a string to enum (if applicable)
        if isinstance(key, str):
            key = DKey[key.upper()]

        memmaps = [self._find_memmap(day, key) for day in self.days]
        return MemmapSequence(memmaps)

    def get_tf_dataset(self, key: DKey, load_batch_size: int = 1024) -> tf.data.Dataset:
        """
        Creates a tensorflow.data.Dataset for a specific DKey of data.

        Args:
            key (DKey): The DKey of the data to load and create the dataset
            batch_size (int, optional): The size of the batches to load (purely for loading, the data does not come batched)
                Defaults to 1024.

        Returns:
            tf.data.Dataset: The generated dataset.
        """
        data = self.get_data(key)

        shape = (load_batch_size,) + data.data_shape

        if shape == (load_batch_size, 1):
            shape = (load_batch_size,)

        batches = math.floor(len(data) / load_batch_size)

        def gen():
            for i in range(batches):
                start = i * load_batch_size
                stop = start + load_batch_size
                yield data[start:stop]

        return (
            tf.data.Dataset.from_generator(
                generator=gen,
                output_signature=tf.TensorSpec(shape=shape, dtype=tf.float64, name=key),
            )
            .unbatch()
            .apply(tf.data.experimental.assert_cardinality(batches * load_batch_size))
        )

    def _find_memmap(self, day: str, key: DKey) -> np.memmap:
        """Matches a given DataName to a specific memmap in a file on disk.

        Args:
            day (str): The day of data to pull from (format: YYYYMMDD)
            key (DKey): The DKey of the dataset to load

        Raises:
            Exception: If the day or key provided does not match any existing stored data.

        Returns:
            np.memmap: A numpy memmap of the found dataset for the given day
        """
        day_path = f"{self.data_dir}/{day}"

        if self.dpath == DPath.CPL_06_REDUCED:
            nature_scalar = np.load(f"{day_path}/scalar.npy", mmap_mode="r")
            nature_table = np.load(f"{day_path}/table.npy", mmap_mode="r")

            hsel = np.load(f"{self.data_dir}/{day}/hsel.npy", mmap_mode="r")

            match key:
                case DKey.HSEL:
                    return hsel
                case DKey.HA:
                    return hsel[:, 0:471]
                case DKey.HB:
                    return hsel[:, 471:932]
                case DKey.HC:
                    return hsel[:, 932:1433]
                case DKey.HD:
                    return hsel[:, 1433:1934]
                case DKey.HW:
                    return hsel[:, 1433:1957]
        else:
            match key:
                case DKey.H1:
                    return np.load(f"{day_path}/h1.npy", mmap_mode="r")
                case DKey.HA:
                    return np.load(f"{day_path}/ha.npy", mmap_mode="r")
                case DKey.HB:
                    return np.load(f"{day_path}/hb.npy", mmap_mode="r")
                case DKey.HC:
                    return np.load(f"{day_path}/hc.npy", mmap_mode="r")
                case DKey.HD:
                    return np.load(f"{day_path}/hd.npy", mmap_mode="r")
                case DKey.HW:
                    return np.load(f"{day_path}/hw.npy", mmap_mode="r")

            nature_scalar = np.load(f"{day_path}/nature_scalar.npy", mmap_mode="r")
            nature_table = np.load(f"{day_path}/nature_table.npy", mmap_mode="r")

        if self.dpath == DPath.CPL_06 or self.dpath == DPath.CPL_06_REDUCED:
            if key == DKey.CPL:
                return np.load(f"{self.data_dir}/{day}/cpl.npy", mmap_mode="r")

        match key:
            case DKey.ATMS:
                return np.load(f"{day_path}/mh.npy", mmap_mode="r")

            case DKey.NATURE_SCALAR:
                return nature_scalar
            case DKey.LATITUDE:
                return nature_scalar[:, 0]
            case DKey.LONGITUDE:
                return nature_scalar[:, 1]
            case DKey.LAND_FRACTION:
                return nature_scalar[:, 2]
            case DKey.SURFACE_PRESSURE:
                return nature_scalar[:, 3]
            case DKey.SURFACE_TEMPERATURE:
                return nature_scalar[:, 4]
            case DKey.PBLH:
                return nature_scalar[:, 13]

            case DKey.NATURE_TABLE:
                return nature_table
            case DKey.PRESSURE:
                return nature_table[:, :, 0]
            case DKey.TEMPERATURE:
                return nature_table[:, :, 1]
            case DKey.WATER_VAPOR:
                return nature_table[:, :, 2]

            case _:
                raise KeyError(f"No match for DKey {str(key)} found in {self.dpath.name}")
