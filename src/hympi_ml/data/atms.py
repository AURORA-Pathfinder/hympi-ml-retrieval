from abc import abstractmethod
import collections.abc

import torch

from hympi_ml.data.base import DataSpec, DataSource

NEDT = [
    0.7,
    0.8,
    0.9,
    0.7,
    0.7,
    0.7,
    0.7,
    0.7,
    0.7,
    0.75,
    1.2,
    1.2,
    1.5,
    2.4,
    3.6,
    0.5,
    0.6,
    0.8,
    0.8,
    0.8,
    0.8,
    0.9,
]


class ATMSSource(DataSource):
    @property
    @abstractmethod
    def atms(self) -> collections.abc.Sequence:
        pass


class ATMSSpec(DataSpec):
    nedt: bool = False

    @property
    def shape(self) -> tuple[int, ...]:
        return (22,)

    @property
    def units(self) -> str:
        return "Brightness Temperature (K)"

    def load_raw_slice(
        self, source: ATMSSource, start: int, end: int
    ) -> collections.abc.Sequence:
        if not isinstance(source, ATMSSource):
            raise Exception(
                "The provided source does not have support for loading ATMS data."
            )

        return source.atms[start:end]

    def _apply_nedt(self, raw_batch: torch.Tensor) -> torch.Tensor:
        # save the random state before applying a seeded NEDT
        rand_state = torch.random.get_rng_state()

        # set the seed based on the sum of the batch values
        torch.manual_seed(raw_batch.sum())

        nedt = torch.tensor(NEDT, device=raw_batch.device)
        batch_nedt = torch.normal(raw_batch, nedt)

        # reapply old random state to preserve the same randomness regardless of whether this method runs
        torch.random.set_rng_state(rand_state)

        return batch_nedt

    def transform_batch(self, batch: torch.Tensor) -> torch.Tensor:
        if self.nedt:
            batch = self._apply_nedt(batch)

        return super().transform_batch(batch)
