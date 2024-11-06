from typing import List
from enum import Enum, auto

import numpy as np

from hympi_ml.data.memmap import MemmapSequence


class DKey(str, Enum):
    """
    An Enum that contains the list of datasets that could be loaded.
    """

    # ensures values are all equal to the string version of the enum name
    def _generate_next_value_(name, *_):  # noqa: N805
        return name

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
    TEMPERATURE_14 = auto()
    WATER_VAPOR = auto()
    WATER_VAPOR_14 = auto()


class FullDaysLoader:
    """
    A class for loading Fulldays data initialized with a given number of days in the YYYYMMDD format.
    """

    def __init__(self, days: List[str]) -> None:
        """Initialize a FullDaysLoader with a list of days

        Args:
            days (List[str]): the list of days to read from (format: YYYYMMDD)
        """
        self.days = days

    @property
    def data_dir(self) -> str:
        """
        Returns the directory where the fulldays data is being pulled from.
        """
        return "/data/nature_run/fulldays_cpl"

    def get_data(self, key: DKey | str) -> MemmapSequence:
        """
        Returns a MemmapSequence from a given DKey on the days initialized in this loader.
        """
        # convert dataName from a string to enum (if applicable)
        if isinstance(key, str):
            key = DKey[key.upper()]

        memmaps = [self.find_memmap(day, key) for day in self.days]
        return MemmapSequence(memmaps)

    def find_memmap(self, day: str, key: DKey) -> np.memmap:
        """Matches a given DataName to a specific memmap in a file on disk.

        Args:
            day (str): The day of data to pull from (format: YYYYMMDD)
            key (DKey): The DKey of the dataset to load

        Raises:
            Exception: If the day or key provided does not match any existing stored data.

        Returns:
            np.memmap: _description_
        """
        dir_path = f"{self.data_dir}/{day}/"

        nature_scalar = np.load(dir_path + "nature_scalar.npy", mmap_mode="r")
        nature_table = np.load(dir_path + "nature_table.npy", mmap_mode="r")

        match key:
            case DKey.H1:
                return np.load(f"{dir_path}/h1.npy", mmap_mode="r")
            case DKey.HA:
                return np.load(f"{dir_path}/ha.npy", mmap_mode="r")
            case DKey.HB:
                return np.load(f"{dir_path}/hb.npy", mmap_mode="r")
            case DKey.HC:
                return np.load(f"{dir_path}/hc.npy", mmap_mode="r")
            case DKey.HD:
                return np.load(f"{dir_path}/hd.npy", mmap_mode="r")
            case DKey.HW:
                return np.load(f"{dir_path}/hw.npy", mmap_mode="r")

            case DKey.ATMS:
                return np.load(dir_path + "mh.npy", mmap_mode="r")

            case DKey.CPL:
                return np.load(dir_path + "cpl.npy", mmap_mode="r")

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
            case DKey.TEMPERATURE_14:
                return nature_table[:, -14:, 1]
            case DKey.WATER_VAPOR:
                return nature_table[:, :, 2]
            case DKey.WATER_VAPOR_14:
                return nature_table[:, -14:, 2]

            case _:
                raise Exception(f"No match for {str(key)} found in {dir_path}")
