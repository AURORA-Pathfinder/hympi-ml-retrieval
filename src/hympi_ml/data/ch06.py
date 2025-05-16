import glob

import numpy as np

from hympi_ml.data import cosmirh, atms, ampr, nature_run
from hympi_ml.data.memmap import MemmapSequence

CH06_PATH = "/explore/nobackup/people/dgershm1/pbl_fusion/ch06"
"""
The path to the data CH06 dataset itself. This is used directly in Ch06Source loading.
"""


class Ch06Source(
    cosmirh.CosmirhSource, 
    atms.ATMSSource, 
    ampr.AMPRSource, 
    nature_run.NRSource,
):
    """
    The data source for the 12 days of worth of simulated CRTM CosMIR-H, ATMS, AMPR, and Nature Run data.
    The available days come in the format: 2006##15 where "##" is the month with a leading zero (as needed).
    """

    days: list[str]

    @property
    def data_dir(self) -> str:
        return CH06_PATH
    
    @property
    def sample_count(self) -> int:
        return len(self.ampr)

    def _load_memmaps(self, name: str):
        files = []

        for day in self.days:
            files += glob.glob(f"{self.data_dir}/{day}/*/{name}.npy")

        return [np.load(f, mmap_mode="r") for f in files]

    @property
    def c1(self) -> MemmapSequence:
        return MemmapSequence(self._load_memmaps("C1"))
    
    @property
    def c2(self) -> MemmapSequence:
        return MemmapSequence(self._load_memmaps("C2"))
    
    @property
    def c3(self) -> MemmapSequence:
        return MemmapSequence(self._load_memmaps("C3"))
    
    @property
    def c4(self) -> MemmapSequence:
        return MemmapSequence(self._load_memmaps("C4"))
    
    @property
    def c5(self) -> MemmapSequence:
        return MemmapSequence(self._load_memmaps("C5"))
    
    @property
    def c6(self) -> MemmapSequence:
        return MemmapSequence(self._load_memmaps("C6"))

    @property
    def atms(self) -> MemmapSequence:
        return MemmapSequence(self._load_memmaps("MH"))

    @property
    def ampr(self) -> MemmapSequence:
        return MemmapSequence(self._load_memmaps("AP"))

    @property
    def nr_temperature(self) -> MemmapSequence:
        seq = []
        for table in self._load_memmaps("nature_table"):
            seq.append(table[:, :, 1])

        return MemmapSequence(seq)

    @property
    def nr_water_vapor(self) -> MemmapSequence:
        seq = []
        for table in self._load_memmaps("nature_table"):
            seq.append(table[:, :, 2])

        return MemmapSequence(seq)

    @property
    def nr_pressure(self) -> MemmapSequence:
        seq = []
        for table in self._load_memmaps("nature_table"):
            seq.append(table[:, :, 0])

        return MemmapSequence(seq)

    @property
    def nr_pblh(self) -> MemmapSequence:
        seq = []
        for scalar in self._load_memmaps("nature_scalar"):
            seq.append(scalar[:, 13])

        return MemmapSequence(seq)

    @property
    def nr_latlon(self) -> tuple[MemmapSequence, MemmapSequence]:
        lats = []
        lons = []
        for scalar in self._load_memmaps("nature_scalar"):
            lats.append(scalar[:, 0])
            lons.append(scalar[:, 1])

        lat_ds = MemmapSequence(lats)
        lon_ds = MemmapSequence(lons)

        return (lat_ds, lon_ds)
