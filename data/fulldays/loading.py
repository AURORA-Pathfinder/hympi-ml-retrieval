from typing import List
from enum import Enum, auto

import numpy as np

from data.memmap import MemmapSequence


class DKey(str, Enum):
    """
    An Enum that contains the list of datasets that could be loaded.
    """

    def _generate_next_value_(name, *_):
        return name

    HSEL = auto()
    HA = auto()
    HB = auto()
    HC = auto()
    HD = auto()
    HW = auto()
    HMW = auto()

    ATMS = auto()

    LABELS_SCALAR = auto()
    LATITUDE = auto()
    LONGITUDE = auto()
    PBLH = auto()
    LAND_FRACTION = auto()
    SURFACE_PRESSURE = auto()
    SURFACE_TEMPERATURE = auto()

    LABELS_TABLE = auto()
    PRESSURE = auto()
    TEMPERATURE = auto()
    WATER_VAPOR = auto()


class FullDaysLoader:
    """
    A utility class for loading Fulldays data initialized with a given number of days in the YYYYMMDD format.
    """

    def __init__(self, days: List[str]) -> None:
        self.days = days

    @property
    def data_dir(self):
        """
        Returns the directory where the fulldays data is being pulled from.
        """
        return "/data/nature_run/fulldays_reduced_evenmore"

    def get_data(self, key: DKey | str) -> MemmapSequence:
        """
        Returns a MemmapSequence from a given DataName on the days initialized in this loader.
        """
        # convert dataName from a string to enum (if applicable)
        if isinstance(key, str):
            key = DKey[key.upper()]

        memmaps = [self.find_memmap(day, key) for day in self.days]
        return MemmapSequence(memmaps)

    def find_memmap(self, day: str, dataName: DKey) -> np.memmap:
        """
        Matches a given DataName to a specific memmap in a file on disk.
        """
        dir_path = f"{self.data_dir}/{day}/"

        hsel = np.load(dir_path + "hsel.npy", mmap_mode="r")
        hw = np.load(dir_path + "hw.npy", mmap_mode="r")
        atms = np.load(dir_path + "mh.npy", mmap_mode="r")
        scalar = np.load(dir_path + "scalar.npy", mmap_mode="r")
        table = np.load(dir_path + "table.npy", mmap_mode="r")

        match dataName:
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
            case DKey.HMW:
                return hsel[:, -1]
            case DKey.HW:
                return hw

            case DKey.ATMS:
                return atms

            case DKey.LABELS_SCALAR:
                return scalar
            case DKey.LATITUDE:
                return scalar[:, 0]
            case DKey.LONGITUDE:
                return scalar[:, 1]
            case DKey.LAND_FRACTION:
                return scalar[:, 2]
            case DKey.SURFACE_PRESSURE:
                return scalar[:, 3]
            case DKey.SURFACE_TEMPERATURE:
                return scalar[:, 4]
            case DKey.PBLH:
                return scalar[:, 13]

            case DKey.LABELS_TABLE:
                return table
            case DKey.PRESSURE:
                return table[:, :, 0]
            case DKey.TEMPERATURE:
                return table[:, :, 1]
            case DKey.WATER_VAPOR:
                return table[:, :, 2]

            case _:
                raise Exception(f"No match for {str(dataName)} found in {dir_path}")
