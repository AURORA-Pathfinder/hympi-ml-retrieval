from abc import ABC, abstractmethod

import torch
from pydantic import BaseModel


class Filter(BaseModel, ABC):
    @abstractmethod
    def get_batch_mask(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Given a batch of data, creates a mask based on the filter_range for this spec.
        Apply this mask or combine it with other masks to filter values for a batch of data.

        NOTE: This mask is a one-dimensional vector for whether a single sample in a batch
        matches the filter.
        """
        pass

    def _flatten_dims(self, mask: torch.Tensor) -> torch.Tensor:
        """
        A helpful method to flatten any masks using multi-dimensional data to ignore all values
        in that sample if any one fails to meet the mask.
        """
        if len(mask.shape) > 1:
            return mask.all(dim=1)
        else:
            return mask


class SimpleRangeFilter(Filter):
    """
    A filter that returns a mask that ignores any values outside of the provided minimum and maximim values (inclusive).
    """

    minimum: float
    maximum: float

    def get_batch_mask(self, batch: torch.Tensor) -> torch.Tensor:
        mask = (batch >= self.minimum) & (batch <= self.maximum)
        return self._flatten_dims(mask)


class ExactValueFilter(Filter):
    """
    A filter that returns a mask that ignores any values not exactly equal to the defined value.
    """

    value: float

    def get_batch_mask(self, batch: torch.Tensor) -> torch.Tensor:
        mask = batch == self.value
        return self._flatten_dims(mask)


class ResolutionFilter(Filter):
    """
    A filter that returns a mask that will ignore any values that are not a multiple of the defined resolution.
    """

    resolution: float

    def get_batch_mask(self, batch: torch.Tensor) -> torch.Tensor:
        mask = batch % self.resolution == 0
        return self._flatten_dims(mask)
