from collections.abc import Sequence

import numpy as np


class MemmapSequence(Sequence):
    """
    A wrapper class for a list of memmaps that represent a single, combined dataset.

    Once initialized, allows for quickly indexing and slicing into the set of memmaps
    as if it were one concatenated array.
    """

    def __init__(self, memmaps: list[np.memmap]):
        ref_shape = memmaps[0].shape[1:]

        if not all(mm.shape[1:] == ref_shape for mm in memmaps):
            raise Exception(
                f"A memmap has the wrong shape! Expected {ref_shape} but a file did not match!"
            )

        self.memmaps = memmaps

    @classmethod
    def from_files(cls, file_paths: list[str]):
        """
        Creates a new Memmaps from a set of '.npy' file paths by loading each file in "read" mode.
        """
        mmaps = [np.load(f, mmap_mode="r") for f in file_paths]
        return cls(mmaps)

    def __len__(self):
        """
        Returns the total size by taking the sum of the first shape value in each memmaps.
        """
        return sum((m.shape[0] for m in self.memmaps))

    def __getitem__(self, val: int | slice | np.ndarray) -> np.memmap | np.ndarray:
        """
        Returns the value at a specific index or slice (or set of indices) from the list of memmaps as if it were
        one concatenated list.

        Note that, if the value is an index or slice within a single file, then
        the output will be a numpy memmap. However, if a slice traverses multiple files, then an ndarray
        will be returned due to limitations when concatenating numpy memmaps.
        """

        if isinstance(val, (int, np.int_)):
            index = val
            file_index = 0

            for m in self.memmaps:
                m_size = m.shape[0]

                if index >= 0 and index < m_size:
                    break

                file_index += 1
                index -= m_size

            return self.memmaps[file_index][index]

        if isinstance(val, slice):
            start = val.start or 0
            stop = val.stop or self.__len__()

            slices = []

            for memmap in self.memmaps:
                size = memmap.shape[0]

                if start < size and stop > 0:
                    clamp_stop = min(size, stop)
                    clamp_start = max(start, 0)

                    data = memmap[clamp_start : clamp_stop : val.step]

                    slices.append(data)

                start -= size
                stop -= size

            if len(slices) == 1:
                return slices[0]
            else:
                return np.concatenate(slices)

        if isinstance(val, np.ndarray):
            samples = []
            for index in val:
                samples.append(self[index])

            return np.array(samples)

    @property
    def data_shape(self) -> tuple[int, ...]:
        """
        Returns the shape of the first index of the dataset
        """
        shape = self.memmaps[0][0].shape

        if shape == ():
            return (1,)

        return shape

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Returns the shape of the combined set of the memmaps
        """
        return (self.__len__(),) + self.data_shape

    def to_ndarray(self) -> np.ndarray:
        """
        Converts the entire sequence into one, large, concatenated ndarray.

        Note: This may take a while if this sequence lots of data. Use wisely.
        """
        return np.concatenate(self.memmaps)
