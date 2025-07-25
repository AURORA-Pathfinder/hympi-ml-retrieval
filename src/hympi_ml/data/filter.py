from abc import ABC, abstractmethod

import torch
from pydantic import BaseModel


class Filter(BaseModel, ABC):
    @abstractmethod
    def get_batch_mask(self, batch: torch.Tensor) -> torch.Tensor:
        pass


class SimpleRangeFilter(Filter):
    minimum: float
    maximum: float

    def get_batch_mask(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Given a batch of data, creates a mask based on the filter_range for this spec.
        Apply this mask or combine it with other masks to filter values for a batch of data.

        NOTE: This mask is a one-dimensional vector for whether a single sample in a batch
        matches the filter.
        """
        mask = (batch >= self.minimum) & (batch <= self.maximum)

        # flattens multi-dimensional masks (if any value in a sample is false, the entire sample is filtered out)
        if len(mask.shape) > 1:
            return mask.all(dim=1)
        else:
            return mask
