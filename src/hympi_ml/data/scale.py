from abc import ABC, abstractmethod

import torch
from torch import Tensor
import numpy as np
from pydantic import BaseModel


class Scaler(BaseModel, ABC):
    @abstractmethod
    def scale_batch(self, batch: Tensor) -> Tensor:
        """
        Takes a batch of data and applies this scaler's transformation to all values in the batch.
        """
        pass

    @abstractmethod
    def unscale_batch(self, batch: Tensor) -> Tensor:
        """
        Takes a batch of previously scaled data and reverses this scaler's transformation to all values in the batch.
        """
        pass


class MinMaxScaler(Scaler):
    minimum: float
    maximum: float

    def _min_max_to_tensor(self) -> tuple[Tensor, Tensor]:
        minimum = torch.from_numpy(np.array(self.minimum, dtype=np.float64))
        maximum = torch.from_numpy(np.array(self.maximum, dtype=np.float64))

        return minimum, maximum

    def scale_batch(self, batch: Tensor) -> Tensor:
        minimum, maximum = self._min_max_to_tensor()
        return torch.div(torch.sub(batch, minimum), torch.sub(maximum, minimum))

    def unscale_batch(self, batch: Tensor) -> Tensor:
        # minimum, maximum = self._min_max_to_tensor()
        # range = torch.sub(maximum, minimum)
        return torch.mul(batch, self.maximum - self.minimum) + self.minimum
