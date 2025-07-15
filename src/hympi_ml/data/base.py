from abc import ABC, abstractmethod
import collections.abc

import torch
import numpy as np
from pydantic import BaseModel


class NamedBaseModel(BaseModel):
    @classmethod
    def from_dump(cls, dump: dict):
        if cls.__name__ == dump["class_name"]:
            return cls(**dump)

        for sub_class in cls.__subclasses__():
            if sub_class.__name__ == dump["class_name"]:
                return sub_class(**dump)
            else:
                try:
                    return sub_class.from_dump(
                        dump
                    )  # recursively read the dump in sub classes
                except LookupError:  # hapens when the class name doesn't match
                    continue

        raise LookupError(
            f"Class '{dump['class_name']}' not found. You may need to import it."
        )

    def model_dump(self, *args, **kwargs):
        dump = super().model_dump(*args, **kwargs)
        dump["class_name"] = self.__class__.__name__
        return dump


class DataSource(NamedBaseModel, ABC):
    """The base class for a source of data"""

    @property
    @abstractmethod
    def sample_count(self) -> int:
        """The total number of sample data points from this DataSource instance."""
        pass


class DataSpec(NamedBaseModel, ABC):
    """A base class for the specification for some kind of data."""

    scale_range: tuple[float, float] | None = None
    filter_range: tuple[float, float] | None = None

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        pass

    @property
    @abstractmethod
    def units(self) -> str:
        pass

    @abstractmethod
    def load_raw_slice(
        self, source: DataSource, start: int, end: int
    ) -> collections.abc.Sequence:
        """
        Load a slice of raw data from a DataSource with the provided 'start' and 'end' indices.
        """
        pass

    def apply_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Applies the data spec transformations to a single batch of raw data.
        """
        if self.scale_range is not None:
            minimum = torch.from_numpy(np.array(self.scale_range[0], dtype=np.float64))
            maximum = torch.from_numpy(np.array(self.scale_range[1], dtype=np.float64))
            return torch.div(torch.sub(batch, minimum), torch.sub(maximum, minimum))

        return batch

    def get_filter_mask(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Given a batch of data, creates a mask based on the filter_range for this spec.
        Apply this mask or combine it with other masks to filter values for a batch of data.

        NOTE: This mask is a one-dimensional vector for whether a single sample in a batch
        matches the filter.
        """
        mask = torch.ones(size=batch.shape, device=batch.device) == 1

        if self.filter_range is not None:
            low, high = self.filter_range
            mask &= (batch >= low) & (batch <= high)

        # flattens multi-dimensional masks (if any value in a sample is false, the entire sample is filtered out)
        if len(mask.shape) > 1:
            return mask.all(dim=1)
        else:
            return mask
