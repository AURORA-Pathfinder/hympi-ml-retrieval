

# cpl1064_allsky_200608150000_01.nc
# /discover/nobackup/projects/hympi/kechris1/allsky/2006MMDD/cpl/
#base = /discover/nobackup/projects/hympi/kechris1/allsky/
import numpy as np
import glob
import sys
import os
from random import randint
import netCDF4 as nc
import matplotlib.pyplot as plt
from shapely.geometry import Point
import geopandas as gpd
import pandas as pd
from multiprocessing import Pool


def load_single_timestep(fname):
    base_out = "/home/jmackin1/hympi/jmackin1/cpl"

    print(f"Processing {fname}")

    dataset = nc.Dataset(fname, mode='r')

    flag = "LIDAR_FLAG"
    data = "INSTRUMENT_ATB_NOISE"
    lons = "trjLon"
    lats = "trjLat"

    lidar_flag_data = dataset.variables[flag][:]
    cpl_data = dataset.variables[data][:]
    lond = dataset.variables[lons][:]
    latd = dataset.variables[lats][:]

    flagv = 1
    indices = np.where(lidar_flag_data == flagv)

    df = pd.DataFrame(cpl_data[indices], columns=[f'cpl_{i}' for i in range(734)])

    geometry = [Point(lon, lat) for lon, lat in zip(lond[indices], latd[indices])]

    # Step 4: Create a GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=geometry)
    gdf.set_crs(epsg=4326, inplace=True)

    fname_out = f"{base_out}/{fname.split('/')[-1].replace('nc', 'pkl')}"
    gdf.to_pickle(fname_out)

    return


if __name__ == "__main__":
    base = "/discover/nobackup/projects/hympi/kechris1/allsky"

    # fpw = open("done_files.txt", 'a')

    all_paths = []
    for day in ["20060315", "20060515", "20060615", "20060715",  "20060803",
                "20060815",  "20060915",  "20061015",  "20061115",  "20061215"]:
        paths = glob.glob(f"{base}/{day}/cpl/cpl532*.nc")
        all_paths += paths

    # print(all_paths)
    # print(len(all_paths))
    with Pool(48) as p:
        p.map(load_single_timestep, all_paths)

    #for fname in paths:
    #    print(f"Processing {fname}")
    #    gdf = load_single_timestep(fname)
    #    fname_out = f"{base_out}/{fname.split('/')[-1].replace('nc', 'pkl')}"
    #    gdf.to_pickle(fname_out)
    #    fpw.write(f"{fname_out}\n")`
