from abc import abstractmethod
from collections.abc import Sequence

import torch

from hympi_ml.data import DataSpec, DataSource, RFBand

C50_BAND = RFBand(low=50.0019, high=57.99799375, channel_width=0.00390625)
"""The 50 GHz band of CoSMIR-H from ~50-58 GHz containing 2048 individual channels."""

C183_BAND = RFBand(low=175.3120, high=191.30809375, channel_width=0.00390625)
"""The 183 GHz band of CoSMIR-H from ~175-192 GHz containing 4096 individual channels."""

WINDOW_CHANNELS = [89.00, 165.30]
"""The 4 window channels of CoSMIR-H data. Note that this list only contains the pure frequency values (two floats)
as each frequency is horizontally and vertically polarized"""


class CosmirhSource(DataSource):
    """
    An abstract base class for loading CoSMIR-H data.
    """

    @property
    @abstractmethod
    def ch(self) -> Sequence:
        """
        Loads the entire set of cosmir-h data as a sequence of samples each with a size of 6148 channels the order:
        1. 2048 channel 50-58 GHz band
        2. 4096 channel 175-192 GHz band
        3. 4 window channels at (89 GHz and 165.3 GHz, horizontally and vertically polarized)
        """
        pass


class CosmirhSpec(DataSpec):
    frequencies: list[RFBand | float]
    ignore_frequencies: list[RFBand | float] | None = None
    _indices = None

    def _calculate_indices(self) -> list[int]:
        all_freqs = set()

        for f in self.frequencies:
            if isinstance(f, RFBand):
                all_freqs.update(C50_BAND.intersection(f))
                all_freqs.update(C183_BAND.intersection(f))
            elif isinstance(f, float):
                all_freqs.add(f)

        indices: list[int] = []

        for freq in all_freqs:
            ignore = False

            if self.ignore_frequencies is not None:
                for ignore_freq in self.ignore_frequencies:
                    if isinstance(ignore_freq, RFBand) and freq in ignore_freq:
                        ignore = True
                    elif isinstance(ignore_freq, float) and freq == ignore_freq:
                        ignore = True

            if ignore:
                continue

            if freq in C50_BAND:
                indices.append(C50_BAND.index(freq))
            elif freq in C183_BAND:
                indices.append(C183_BAND.index(freq) + 2048)
            elif freq in WINDOW_CHANNELS:
                indices.append(WINDOW_CHANNELS.index(freq) + 2048 + 4096)

            indices = sorted(indices)

        if indices == []:
            raise Exception("No data found to match the provided set of frequencies.")

        return indices

    @property
    def indices(self) -> list[int]:
        if self._indices is None:
            self._indices = self._calculate_indices()

        return self._indices

    @property
    def shape(self) -> tuple[int, ...]:
        size = len(self.indices)
        return (size,)

    @property
    def units(self) -> str:
        return "Brightness Temperature (K)"

    def load_raw_slice(self, source: CosmirhSource, start: int, end: int) -> Sequence:
        if not isinstance(source, CosmirhSource):
            raise Exception(
                "The provided loader does not have support for loading CoSMIR-H data."
            )

        return source.ch[start:end]

    def apply_batch(self, batch) -> torch.Tensor:
        indices_tensor = torch.tensor(self.indices, device=batch.device)
        batch = torch.index_select(batch, dim=1, index=indices_tensor)
        return super().apply_batch(batch)
