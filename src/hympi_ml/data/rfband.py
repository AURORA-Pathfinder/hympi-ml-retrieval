import numpy as np
from pydantic import BaseModel


class RFBand(BaseModel):
    """A class that defines a band of radiofrequency (RF) at an optional channel width.
    This is meant to be paired with other data to define Radiometer specifications.

    Note that there is no checks for order of magnitude. Ensure that the RFBand you implement
    are all using the same units (normally it's GHz/MHz for Hyperspectral data).
    """

    low: float
    high: float
    channel_width: float | None = None

    @property
    def _order(self) -> int:
        return len(str(self.channel_width).split(".")[1])

    def __contains__(self, f: float) -> bool:
        """Checks if the provided frequency is contained in this frequency band and channel width.

        Args:
            f (float): The frequency to check.

        Returns:
            bool: Whether this band contains the frequency.
        """
        # check range
        if not (f >= self.low and f <= self.high):
            return False

        if self.channel_width is None:
            return True

        diff = round(f - self.low, self._order)

        return round(diff / self.channel_width, self._order) % 1 == 0

    def to_array(self) -> list[float]:
        """Expands the frequency band into an array containing all possible values in the range
            at the channel width.

        Raises:
            ValueError: If no channel width was defined for this band.

        Returns:
            list[float]: An array containing all frequencies in this band.
        """
        if self.channel_width is None:
            raise ValueError(
                "No channel width provided for this band. Cannot convert to any array without a channel width."
            )

        order = len(str(self.channel_width).split(".")[1])

        stop = round(self.high + self.channel_width, order)
        ha_freqs = np.arange(self.low, stop, step=self.channel_width)

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

        if self.channel_width is None:
            raise ValueError(
                f"No channel width provided for this band. Cannot find index of {f} without a channel width."
            )

        return round((f - self.low) / self.channel_width)

    def intersection(self, other: "RFBand") -> list[float]:
        """
        Returns a list of frequencies that exist in this band and the provided band.
        """
        if self.channel_width is None and other.channel_width is None:
            raise ValueError(
                "Neither band has a channel width, no intersection can be found."
            )

        if self.channel_width is not None and other.channel_width is not None:
            self_set = set(self.to_array())
            other_set = set(other.to_array())
            return list(self_set.intersection(other_set))

        if self.channel_width is None:
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
        Returns a new RFBand with the channel width scaled by the provided factor.

        Raises:
            ValueError: If no channel width is defined for this band.
        """
        if self.channel_width is not None:
            return RFBand(
                low=self.low, high=self.high, channel_width=self.channel_width * factor
            )

        raise ValueError("Cannot scale RFBand with no channel width!")
