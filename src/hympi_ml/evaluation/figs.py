import numpy as np
import matplotlib.pyplot as plt


def plot_profiles(
    profiles: dict[str, np.ndarray],
    value_axis: str | None = None,
    levels_axis: str = "Levels",
):
    """
    Given a dictionary of arrays that each represent a single profile, plots them into a common format used for
    profile figures. This includes a vertical, inverted x-axis with proper labels.

    Note: This should be used to replace a matplotlib `plt.plot()` method for any profile.
    This way, all profile plots have a consistent look.

    Args:
        profiles (Dict[str, np.ndarray]):
            A dictionary with keys as the label and values as an ndarray representing a profile
        value_axis (str): The label for the values of the profile (example: "Temperature (K)")
        levels_axis (str): The label for the number of levels in this profile. Defaults to "Levels".
    """
    for label, profile in profiles.items():
        plt.plot(profile, range(len(profile)), ".-", label=label)

    ax = plt.gca()

    if value_axis is None:
        value_axis = "Values"

    ax.set_xlabel(value_axis)
    ax.set_ylabel(levels_axis)
    ax.invert_yaxis()
