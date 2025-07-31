from abc import abstractmethod, ABC
import collections.abc

from hympi_ml.data.base import DataSpec, DataSource


class AMPRSource(DataSource, ABC):
    @property
    @abstractmethod
    def ampr(self) -> collections.abc.Sequence:
        pass


class AMPRSpec(DataSpec):
    @property
    def shape(self) -> tuple[int, ...]:
        return (8,)

    @property
    def units(self) -> str:
        return "Brightness Temperature (K)"

    def load_raw_slice(
        self, source: AMPRSource, start: int, end: int
    ) -> collections.abc.Sequence:
        if not isinstance(source, AMPRSource):
            raise Exception(
                "The provided source does not have support for loading AMPR data."
            )

        return source.ampr[start:end]
