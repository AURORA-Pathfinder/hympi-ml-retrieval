from abc import abstractmethod
import collections.abc
from typing import Literal

from hympi_ml.data.base import DataSpec, DataSource


class NRSource(DataSource):
    @property
    @abstractmethod
    def nr_temperature(self) -> collections.abc.Sequence:
        pass

    @property
    @abstractmethod
    def nr_water_vapor(self) -> collections.abc.Sequence:
        pass

    @property
    @abstractmethod
    def nr_pressure(self) -> collections.abc.Sequence:
        pass

    @property
    @abstractmethod
    def nr_pblh(self) -> collections.abc.Sequence:
        pass

    @property
    @abstractmethod
    def nr_latlon(self) -> collections.abc.Sequence:
        pass


class NRSpec(DataSpec):
    dataset: Literal["TEMPERATURE", "WATER_VAPOR", "PRESSURE", "PBLH", "LATLON"]

    @property
    def shape(self) -> tuple[int, ...]:
        match self.dataset:
            case "LATLON":
                return (2,)
            case "PBLH":
                return (1,)

            case "PRESSURE" | "TEMPERATURE" | "WATER_VAPOR":
                return (72,)
            case _:
                raise KeyError(f"{self.dataset} is not handled by nature run!")

    @property
    def units(self) -> str:
        match self.dataset:
            case "LATLON":
                return "Degrees"
            case "PBLH":
                return "Height (m)"

            case "PRESSURE":
                return "Presure (mb)"
            case "TEMPERATURE":
                return "Temperature (K)"
            case "WATER_VAPOR":
                return "Specific Humidity (q)"
            case _:
                raise KeyError(f"{self.dataset} is not handled by nature run!")

    def load_raw_slice(
        self, source: NRSource, start: int, end: int
    ) -> collections.abc.Sequence:
        if not isinstance(source, NRSource):
            raise Exception(
                "The provided data source does not have support for loading nature run data."
            )

        match self.dataset:
            case "LATLON":
                return source.nr_latlon[start:end]
            case "PBLH":
                return source.nr_pblh[start:end]

            case "PRESSURE":
                return source.nr_pressure[start:end]
            case "TEMPERATURE":
                return source.nr_temperature[start:end]
            case "WATER_VAPOR":
                return source.nr_water_vapor[start:end]
            case _:
                raise KeyError(f"{self.dataset} is not handled by nature run!")
