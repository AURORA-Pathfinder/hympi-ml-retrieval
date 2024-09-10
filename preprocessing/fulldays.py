from enum import Enum, auto
from typing import List

import numpy as np
from numpy import ndarray

from preprocessing.memmap import MemmapSequence

# TODO: Make the Data names be ones that WILL exist if you provide a day (otherwise what's the point of an enum!)
class DataName(Enum):
    '''
    An Enum that contains the list of datasets that could be loaded.
    '''
    hsel = auto()
    hwd = auto()
    ha = auto()
    hb = auto()
    hc = auto()
    hd = auto()
    hw = auto()
    cosmir_h = auto()
    bsl = auto()
    bsl_depth = auto()
    atms = auto()

    labels_scalar = auto()
    latitude = auto()
    longitude = auto()
    pblh = auto()
    surface_pressure = auto()
    surface_temperature = auto()
    date_time = auto()

    labels_table = auto()
    pressure = auto()
    height = auto()
    temperature = auto()
    water_vapor = auto()
    ozone_density = auto()

class FullDaysLoader:
    def __init__(self, days: List[str]) -> None:
        self.days = days

    def get_data(self, dataName: DataName | str) -> MemmapSequence:
        # convert dataName from a string to enum (if applicable)
        if isinstance(dataName, str):
            dataName = DataName[dataName]

        memmaps = [self.find_memmap(day, dataName) for day in self.days]
        return MemmapSequence(memmaps)
    
    # matches a given dataName to a specific set of data in a file 
    def find_memmap(self, day: str, dataName: DataName) -> np.memmap:
        dir_path = f"/data/nature_run/fulldays_reduced_evenmore/{day}/"

        hsel = np.load(dir_path + "hsel.npy", mmap_mode='r')
        atms = np.load(dir_path + "mh.npy", mmap_mode='r')
        scalar = np.load(dir_path + "scalar.npy", mmap_mode='r')
        table = np.load(dir_path + "table.npy", mmap_mode='r')

        match dataName:
            case DataName.hsel: return hsel
            case DataName.hwd: return hsel[0]
            case DataName.ha: return hsel[:, 1:473]
            case DataName.hb: return hsel[:, 473:934]
            case DataName.hc: return hsel[:, 934:1435]
            case DataName.hd: return hsel[:, 1435:1934]
            case DataName.hw: return hsel[:, 1934:1957]

            case DataName.atms: return atms

            case DataName.labels_scalar: return scalar
            case DataName.latitude: return scalar[:, 0]
            case DataName.longitude: return scalar[:, 1]
            # case DataName.what_is_this_one?: return scalar[:, 2]
            case DataName.surface_pressure: return scalar[:, 3]

            case DataName.labels_table: return table
            case DataName.pressure: return table[:, :, 0]
            case DataName.temperature: return table[:, :, 1]
            case DataName.water_vapor: return table[:, :, 2]
            
            case _: raise Exception(f"No match for {str(dataName)} found in {dir_path}")