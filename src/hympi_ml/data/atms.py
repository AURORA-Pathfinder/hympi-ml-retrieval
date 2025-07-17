from abc import abstractmethod
import collections.abc

from hympi_ml.data.base import DataSpec, DataSource

ATMS_NEDT = [
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
