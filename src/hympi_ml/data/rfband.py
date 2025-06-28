import math

import numpy as np
from pydantic import BaseModel


class RFBand(BaseModel):
    """A class that defines a band of radiofrequency (RF) at an optional resolution.
    This is meant to be paired with other data to define Radiometer specifications.

    Note that there is no checks for order of magnitude. Ensure that the RFBand you implement
    are all using the same units (normally it's GHz for Hyperspectral data).
    """

    low: float
    high: float
    resolution: float | None = None

    @property
    def _order(self) -> int:
        return len(str(self.resolution).split(".")[1])

    def __contains__(self, f: float) -> bool:
        """Checks if the provided frequency is contained in this frequency band and resolution.

        Args:
            f (float): The frequency to check.

        Returns:
            bool: Whether this band contains the frequency.
        """
        # check range
        if not (f >= self.low and f <= self.high):
            return False

        if self.resolution is None:
            return True

        diff = round(f - self.low, self._order)

        return round(diff / self.resolution, self._order) % 1 == 0

    def to_array(self) -> list[float]:
        """Expands the frequency band into an array containing all possible values in the range
            at the resolution.

        Raises:
            ValueError: If no resolution was defined for this band.

        Returns:
            list[float]: An array containing all frequencies in this band.
        """
        if self.resolution is None:
            raise ValueError(
                "No resolution provided for this band. Cannot convert to any array without a resolution."
            )

        order = len(str(self.resolution).split(".")[1])

        stop = round(self.high + self.resolution, order)
        ha_freqs = np.arange(self.low, stop, step=self.resolution)

        return [float(round(f, order)) for f in ha_freqs]

    def index(self, f: float) -> int:
        """Finds the index where the provided frequency would be if the band was an array.

        Raises:
            ValueError: If the frequency was not found or no resolution exists for the band.

        Returns:
            int: The index of the frequency if the band was an array.
        """
        if f not in self:
            raise ValueError(f"Frequency {f} not found in band.")

        if self.resolution is None:
            raise ValueError(
                f"No resolution provided for this band. Cannot find index of {f} without a resolution."
            )

        return round((f - self.low) / self.resolution)

    def intersection(self, other: "RFBand") -> list[float]:
        """
        Returns a list of frequencies that exist in this band and the provided band.
        """
        if self.resolution is None and other.resolution is None:
            raise ValueError(
                "Neither band has a resolution, no intersection can be found."
            )

        if self.resolution is not None and other.resolution is not None:
            self_set = set(self.to_array())
            other_set = set(other.to_array())
            return list(self_set.intersection(other_set))

        if self.resolution is None:
            no_res = self
            with_res = other
        else:
            no_res = other
            with_res = self

        res_list = with_res.to_array()

        if no_res.low > with_res.high or no_res.high < with_res.low:
            return []

        low_index = None
        high_index = None

        for i, f in enumerate(res_list):
            if low_index is None and f >= no_res.low:
                low_index = i

            if high_index is None and f > no_res.high:
                high_index = i - 1

        return res_list[low_index:high_index]

    def scale(self, factor: float):
        """
        Returns a new RFBand with the resolution scaled by the provided factor.

        Raises:
            ValueError: If no resolution is defined for this band.
        """
        if self.resolution is not None:
            return RFBand(
                low=self.low, high=self.high, resolution=self.resolution * factor
            )

        raise ValueError("Cannot scale RFBand with no resolution!")
