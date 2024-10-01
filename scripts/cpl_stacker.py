import numpy as np

import pandas as pd
import geopandas as gpd
from glob import glob

from shapely.geometry import Point

from tqdm import tqdm


def stack_data(index, max_distance=100000):

    paths = sorted(glob("/data/nature_run/fulldays_reduced/cpl_merged/*.pkl"))
    npz_paths = sorted(glob("/data/nature_run/fulldays_reduced/*.npz"))

    # First NPZ
    data = np.load(npz_paths[index])
    mh_data = data['scalar']

    geometry = [Point(lon, lat) for lon, lat in zip(mh_data[:, 1], mh_data[:, 0])]
    mh_times = mh_data[:, -1].astype(int).astype(str)
    mh_times = [x[-4:] for x in mh_times]

    gdf_npz_uncorrected = gpd.GeoDataFrame(geometry=geometry)
    gdf_npz_uncorrected['times_npz'] = mh_times
    gdf_npz_uncorrected['index_npz'] = list(range(len(mh_times)))
    gdf_npz_uncorrected.set_crs(epsg=4326, inplace=True)

    gdf_cpl_uncorrected = pd.read_pickle(paths[index])

    # Project, this seems weird, fix this to be better
    gdf_npz = gdf_npz_uncorrected.to_crs(epsg=6933)
    gdf_cpl = gdf_cpl_uncorrected.to_crs(epsg=6933)

    print(f"gdf cpl len before {len(gdf_cpl)}")
    print(f"gdf npz len before {len(gdf_npz)}")

    merged_df = []

    all_times = sorted(list(set(mh_times)))
    for i in all_times:
        # Stack by times since points could be close spatially but not
        #  temporally
        cpl_at_time = gdf_cpl[gdf_cpl['time'] == i]
        npz_at_time = gdf_npz[gdf_npz['times_npz'] == i]
        cpl_merged = cpl_at_time.sjoin_nearest(npz_at_time, how='left',
                                               distance_col='distances',
                                               max_distance=max_distance)

        # Fix crap
        cpl_merged = cpl_merged.dropna(subset=['distances'])
        cpl_merged['index_npz'] = cpl_merged['index_npz'].astype(int)

        merged_df.append(cpl_merged)
    merged_df = pd.concat(merged_df, ignore_index=True)

    print(f"combined after {len(merged_df)}")

    # Get the data index's and grab from then write out
    good_data = merged_df['index_npz'].to_numpy()
    cpl_data = merged_df.iloc[:, 1:734].to_numpy()

    mh = data["mh"][good_data]
    hsel = data["hsel"][good_data]
    scalar = data["scalar"][good_data]
    table = data["table"][good_data]

    day = paths[index].split("_")[-1].split(".")[0]

    np.savez(f"/data/nature_run/fulldays_reduced/all_cpl_{day}.npz",
             cpl=cpl_data, mh=mh, hsel=hsel, scalar=scalar, table=table)

    return


if __name__ == "__main__":
    # 10 days currently
    for i in tqdm(range(10)):
        stack_data(i)
