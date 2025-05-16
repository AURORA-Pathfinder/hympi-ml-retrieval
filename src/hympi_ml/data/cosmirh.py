from abc import abstractmethod
import collections.abc

import numpy as np
import torch

from hympi_ml.data import DataSpec, DataSource, RFBand

C1_band = RFBand(low=50.0019, high=53.91986875, resolution=0.00390625)
C2_band = RFBand(low=53.923775, high=57.99799375, resolution=0.00390625)
C3_band = [89.00, 165.30]
C4_band = RFBand(low=175.3120, high=180.7729375, resolution=0.00390625)
C5_band = RFBand(low=180.77684375, high=186.23778125, resolution=0.00390625)
C6_band = RFBand(low=186.2416875, high=191.30809375, resolution=0.00390625)


class CosmirhSource(DataSource):
    """
    An abstract base class for loading CoSMIR-H data.
    Requires six functions for loading each of the six parts of cosmir-h data.
    """

    @property
    @abstractmethod
    def c1(self) -> collections.abc.Sequence:
        pass

    @property
    @abstractmethod
    def c2(self) -> collections.abc.Sequence:
        pass

    @property
    @abstractmethod
    def c3(self) -> collections.abc.Sequence:
        pass

    @property
    @abstractmethod
    def c4(self) -> collections.abc.Sequence:
        pass

    @property
    @abstractmethod
    def c5(self) -> collections.abc.Sequence:
        pass

    @property
    @abstractmethod
    def c6(self) -> collections.abc.Sequence:
        pass


class CosmirhSpec(DataSpec):
    frequencies: list[RFBand | float]
    ignore_frequencies: list[RFBand | float] | None = None
    _indices = None

    def _calculate_indices(self) -> dict[str, list[int]]:
        subsets = [C1_band, C2_band, C3_band, C4_band, C5_band, C6_band]

        all_freqs = set()

        for f in self.frequencies:
            if isinstance(f, RFBand):
                for subset in subsets:
                    if isinstance(subset, RFBand):
                        all_freqs.update(subset.intersection(f))
            elif isinstance(f, float):
                all_freqs.add(f)

        indices: dict[str, list[int]] = {
            "C1": [],
            "C2": [],
            "C3": [],
            "C4": [],
            "C5": [],
            "C6": [],
        }

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

            for i, subset in enumerate(subsets):
                key = list(indices.keys())[i]
                if freq in subset:
                    indices[key].append(subset.index(freq))

                indices[key] = sorted(indices[key])

        if indices == []:
            raise Exception("No data found to match the provided set of frequencies.")

        return indices

    @property
    def shape(self) -> tuple[int, ...]:
        if self._indices is None:
            self._indices = self._calculate_indices()

        size = sum([len(index_set) for index_set in self._indices.values()])
        return (size,)

    @property
    def units(self) -> str:
        return "Brightness Temperature (K)"

    def load_raw_slice(
        self, source: CosmirhSource, start: int, end: int
    ) -> collections.abc.Sequence:
        if not isinstance(source, CosmirhSource):
            raise Exception(
                "The provided loader does not have support for loading CoSMIR-H data."
            )

        if self._indices is None:
            self._indices = self._calculate_indices()

        bt = []

        for subset, indices in self._indices.items():
            if indices == []:
                continue

            seq = []
            match subset:
                case "C1":
                    seq = source.c1
                case "C2":
                    seq = source.c2
                case "C3":
                    seq = source.c3
                case "C4":
                    seq = source.c4
                case "C5":
                    seq = source.c5
                case "C6":
                    seq = source.c6

            bt.append(seq[start:end][:, indices])

        return np.hstack(bt)

    def apply_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Applies the data spec transformations to a single batch of data.
        """
        batch = super().apply_batch(batch)
        return batch
