"""
Generates the .npy files for the fulldays datasets as well as spatially joining CPL data to the other datasets.

It does this by pulling from the .npz already located in /data/nature_run/fulldays
"""

import os
from glob import glob
from typing import Tuple

import numpy as np
import pandas as pd

import geopandas as gpd

from shapely.geometry import Point


def convert_to_reduced_npy():
    """
    Takes .npz data generated in files prepended with `all_cpl_` in `/data/nature_run/fulldays_reduced/` and
    converts them into separate .npy files in directories named by day.
    """

    source = "/data/nature_run/fulldays_reduced"

    for i in glob(f"{source}/all_cpl_*.npz"):
        data = np.load(i)

        print("Loading from " + i)
        mh = data["mh"]
        hsel = data["hsel"]
        table = data["table"]
        scalar = data["scalar"]
        cpl = data["cpl"]

        dir_name = i.strip(".npz").replace("fulldays_reduced/all_cpl_", "fulldays_reduced/") + "/"

        print("Saving data to " + dir_name + " directory...", end=" ")

        os.makedirs(dir_name, exist_ok=True)

        np.save(dir_name + "mh.npy", mh)
        np.save(dir_name + "hsel.npy", hsel)
        np.save(dir_name + "scalar.npy", scalar)
        np.save(dir_name + "table.npy", table)
        np.save(dir_name + "cpl.npy", cpl)

        print("Done!")


def load_old_fulldays(day: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads the complete fulldays datasets from the old .npz files.
    This may take quite a while considering the hympi files alone are several hundreds of gigabytes.
    """
    base_path = "/data/nature_run/fulldays"

    # HSEL
    hsel_file = np.load(f"{base_path}/hympi_{day}.npz")
    hsel_keys = sorted(hsel_file.keys())
    hsel_data = [hsel_file[x] for x in hsel_keys]
    hsel = np.hstack(hsel_data)

    # MH
    mh = np.load(f"{base_path}/MH_{day}.npz")["table"]

    # Nature Run (scalar and table)
    nrun_data = np.load(f"{base_path}/Nature_{day}.npz")
    scalar = nrun_data["scalar"]
    table = nrun_data["table"]

    return (hsel, mh, scalar, table)


def get_joined_cpl_data(day: str, scalar: np.ndarray, max_distance: int = 100_000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get's the CPL data that is spatially joined to the other datasets in Fulldays.

    Returns a tuple in the form of (CPL data, indices) where the indices are the indices
    where CPL data and fulldays data both exist.
    """
    # First NPZ
    geometry = [Point(lon, lat) for lon, lat in zip(scalar[:, 1], scalar[:, 0])]
    mh_times = scalar[:, -1].astype(int).astype(str)
    mh_times = [x[-4:] for x in mh_times]

    gdf_npz_uncorrected = gpd.GeoDataFrame(geometry=geometry)
    gdf_npz_uncorrected["times_npz"] = mh_times
    gdf_npz_uncorrected["index_npz"] = list(range(len(mh_times)))
    gdf_npz_uncorrected.set_crs(epsg=4326, inplace=True)

    cpl_path = f"data/nature_run/fulldays/cpl_merged/cpl_all_{day}.pkl"
    gdf_cpl_uncorrected = pd.read_pickle(cpl_path)

    # Project, this seems weird, fix this to be better
    gdf_npz = gdf_npz_uncorrected.to_crs(epsg=6933)
    gdf_cpl: gpd.GeoDataFrame = gdf_cpl_uncorrected.to_crs(epsg=6933)

    print(f"gdf cpl len before {len(gdf_cpl)}")
    print(f"gdf npz len before {len(gdf_npz)}")

    merged_df = []

    all_times = sorted(set(mh_times))
    for i in all_times:
        # Stack by times since points could be close spatially but not
        #  temporally
        cpl_at_time: gpd.GeoDataFrame = gdf_cpl[gdf_cpl["time"] == i]
        npz_at_time = gdf_npz[gdf_npz["times_npz"] == i]
        cpl_merged = cpl_at_time.sjoin_nearest(
            npz_at_time, how="left", distance_col="distances", max_distance=max_distance
        )

        # Fix crap
        cpl_merged = cpl_merged.dropna(subset=["distances"])
        cpl_merged["index_npz"] = cpl_merged["index_npz"].astype(int)

        merged_df.append(cpl_merged)
    merged_df = pd.concat(merged_df, ignore_index=True)

    print(f"combined after {len(merged_df)}")

    # Get the data index's and grab from then write out
    indices = merged_df["index_npz"].to_numpy()
    cpl = merged_df.iloc[:, 1:734].to_numpy()

    return (cpl, indices)


def generate_fulldays(day: str, max_distance: int = 100000):
    """
    Generates the fulldays dataset with all cloud fractions for a specific day.

    Note that this loads the entirety of the old fulldays data for joining with CPL data later, this can take lots of
    time as the full hsel data is several hundred gigabytes.
    """
    (hsel, mh, scalar, table) = load_old_fulldays(day)

    (cpl, indices) = get_joined_cpl_data(day, scalar, max_distance)

    dir_name = f"/data/nature_run/fulldays/{day}"

    np.save(f"{dir_name}/mh.npy", mh[indices])
    np.save(f"{dir_name}/hsel.npy", hsel[indices])
    np.save(f"{dir_name}/scalar.npy", scalar[indices])
    np.save(f"{dir_name}/table.npy", table[indices])
    np.save(f"{dir_name}/cpl.npy", cpl)
