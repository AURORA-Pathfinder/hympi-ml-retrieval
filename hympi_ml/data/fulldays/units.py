"""
Provides useful functions for related Dkey to units
"""

from hympi_ml.data.fulldays import DKey


def get_formatted_units(key: DKey) -> str | None:
    match key:
        case DKey.H1 | DKey.HA | DKey.HB | DKey.HC | DKey.HD | DKey.HW | DKey.ATMS | DKey.CPL:
            return "Brightness Temperature (K)"

        case DKey.PBLH:
            return "Height (m)"

        case DKey.TEMPERATURE | DKey.SURFACE_TEMPERATURE:
            return "Temperature (K)"

        case DKey.PRESSURE | DKey.SURFACE_PRESSURE:
            return "Pressure (mb)"

        case DKey.WATER_VAPOR:
            return "Specific Humidity (q)"

    return None
