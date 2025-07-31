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
    The data source for several days of worth of simulated CRTM CosMIR-H, ATMS, AMPR, and Nature Run data.
    The available days come in the 2006MMDD format..
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
            files = sorted(files)

        return [np.load(f, mmap_mode="r") for f in files]

    @property
    def ch(self) -> MemmapSequence:
        return MemmapSequence(self._load_memmaps("CH"))

    @property
    def atms(self) -> MemmapSequence:
        return MemmapSequence(self._load_memmaps("MH"))

    @property
    def ampr(self) -> MemmapSequence:
        return MemmapSequence(self._load_memmaps("AP"))

    def _load_nr_table(self, index: int) -> MemmapSequence:
        seq = []
        for table in self._load_memmaps("nature_table"):
            seq.append(table[:, :, index])

        return MemmapSequence(seq)

    def _load_nr_scalar(self, index: int) -> MemmapSequence:
        seq = []
        for table in self._load_memmaps("nature_scalar"):
            seq.append(table[:, index])

        return MemmapSequence(seq)

    @property
    def nr_pressure(self) -> MemmapSequence:
        return self._load_nr_table(0)

    @property
    def nr_temperature(self) -> MemmapSequence:
        return self._load_nr_table(1)

    @property
    def nr_water_vapor(self) -> MemmapSequence:
        return self._load_nr_table(2)

    @property
    def nr_pblh(self) -> MemmapSequence:
        return self._load_nr_scalar(13)

    @property
    def nr_latitude(self) -> MemmapSequence:
        return self._load_nr_scalar(0)

    @property
    def nr_longitude(self) -> MemmapSequence:
        return self._load_nr_scalar(1)
