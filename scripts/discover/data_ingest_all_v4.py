"""
This will parse naturerun level 2 and level1 data
Made to work with v41 level 2 and new radiances, do
not use on older data sets
Updated: 9/25/2023
Updated for NCCS 5/31/2024


Info:

v42 is the sigma

Store location: /discover/nobackup/projects/hympi
Level_1: /discover/nobackup/projects/hympi/rirosenb/crtm/clear_sky/output/forjames_mar2024/G5NR_strips/level1
Level_2: /discover/nobackup/projects/hympi/rirosenb/crtm/clear_sky/output/forjames_mar2024/G5NR_strips/level2/v41
Level_2_aero: /discover/nobackup/projects/hympi/rirosenb/crtm/all_sky/output/forjames_aerosols/G5NR_strips/level2/v42/

Difference between all_sky and clear_sky

"""
import os
import re
import sys
import glob
from multiprocessing import Pool

import numpy as np

# General config

# number of files to process at once
num_workers = int(sys.argv[1])

# Set to where you want your npy saved
base_dir = "/discover/nobackup/projects/hympi/jmackin1/data"


def get_meta(data):
    """
    metadata format: ('extra values')
    lat,lon,land_pct,surface_pressure,skin,mw_se,topog,smw_surfclas, \
      sat_height,view_angl,sol_zen,co2
    """
    meta_d = {"plevels": 72, "head": 4}

    firstline = data[0].split()
    coord = [float(firstline[1]), float(firstline[2])]
    meta = [float(x) for x in data[1].split()]
    meta2 = [float(x) for x in data[2].split()]
    meta3 = [float(x) for x in data[3].split()]

    # num of cloud layers
    meta_d["ncld"] = int(meta[-1])
    meta_d["cld_freq"] = meta_d["ncld"]*2
    meta = meta[:-1]

    meta_d["n_smw"] = int(meta2[1])
    meta_d["nemis"] = int(meta2[2])
    meta_d["ncemis"] = int(meta2[3])
    meta2 = [meta2[0]]

    #  magic offset to get past header and column
    skip = meta_d["plevels"] + 4
    skip += meta_d["ncld"]+meta_d["n_smw"]+meta_d["nemis"]+meta_d["cld_freq"]
    meta_d["extra"] = [int(x) for x in data[skip].split()]

    final = coord+meta+meta2+meta3

    return meta_d, final


def get_cloud(meta, chunk):
    cloud = []
    for i in range(meta["ncld"]):
        cloud.append([float(x) for x in chunk[meta["head"]+i].split()])

    return np.array(cloud)


def get_table(meta, chunk):
    """
    primary nature run data, format:
    OLD!!! DO NOT USE THESE LABELS, LEAVING FOR REFERENCE TO OLD VERSION
    temp,water column density,ozone density,water mix ratio,ice/water flag, \
      co column density,CH4 mixin,C02 mixing ratio,ice mixing, snow mixing, \
      cloud fraction, falling rain, falling snow
    """
    star = meta["head"] + meta["ncld"]
    skip = star + meta["plevels"]
    table_out = [[float(i) for i in x.split()] for x in chunk[star:skip]]

    return np.array(table_out)


def get_nsmw(meta, chunk):
    star = meta["head"] + meta["ncld"] + meta["plevels"]
    skip = star + meta["n_smw"]
    nsmw_out = [[float(i) for i in x.split()] for x in chunk[star:skip]]

    return np.array(nsmw_out)


def get_nemis(meta, chunk):
    star = meta["head"] + meta["ncld"] + meta["plevels"] + meta["n_smw"]
    skip = star + meta["nemis"]
    nemis_out = [[float(i) for i in x.split()] for x in chunk[star:skip]]

    return np.array(nemis_out)


def get_cldfreq(meta, chunk):
    star = (meta["head"] + meta["ncld"] +
            meta["plevels"] + meta["n_smw"] + meta["nemis"])
    skip = star + meta["cld_freq"]
    cldfreq_out = [[float(i) for i in x.split()] for x in chunk[star:skip]]

    return np.array(cldfreq_out)


def get_extra(meta, chunk):
    start = (meta["head"] + meta["ncld"] + meta["cld_freq"] +
             meta["plevels"] + meta["n_smw"] + meta["nemis"] + 1)
    n_ints = meta["extra"][0]
    n_real = meta["extra"][1]

    # we don't know how many lines this data is encoded into
    ints = []
    reals = []
    cnt = 0
    while True:
        line = chunk[start+cnt]
        if n_ints > 0:
            curr = [int(x) for x in line.split()]
            ints += curr
            n_ints -= len(curr)
        else:
            curr = [float(x) for x in line.split()]
            reals += curr
            n_real -= len(curr)
            if n_real == 0:
                break

        cnt += 1

    # ints not needed, but need to get past to the reals
    return np.array(reals)


def process_chunk_nature(chunk):
    meta, data = get_meta(chunk)
    cloud = get_cloud(meta, chunk)
    table = get_table(meta, chunk)
    nsmw = get_nsmw(meta, chunk)
    nemis = get_nemis(meta, chunk)
    cldfreq = get_cldfreq(meta, chunk)
    extra = get_extra(meta, chunk)

    return (np.array(data), cloud, table, nsmw, nemis, cldfreq, extra)


def process_chunk_simple(chunk):
    # For level 1 data

    meta = []
    data = []

    # Collects lat and long as well now
    for n, i in enumerate(chunk):
        if n == 0:
            line = [x for x in i.split()]
            meta.append(float(line[1]))
            meta.append(float(line[2]))
        elif n >= 5 and n < 14:
            [meta.append(float(x)) for x in i.split()]
        elif n >= 14:
            [data.append(float(x)) for x in i.split()]

    return (np.array(meta), np.array(data))


def process_file(fpath):
    with open(fpath, 'r') as fp:
        data = [f.strip() for f in fp.readlines()]

    # Grab header data first
    if len(data) == 0:
        return (0, 0, ""), []

    firstline = data[0].split()
    try:
        base = fpath.split("/")[-1]
        fname = base[:-3]+"npy"
        if "Nature" in base:
            header = (int(firstline[1]), int(firstline[4]), fname)
        else:
            header = (int(firstline[2]), int(firstline[3]), fname)
    except ValueError as e:
        print(f"Unsupported Header {e}")
        return (0, 0, ""), []

    # Initial parser
    # all_data contains lists containing all Lxxx data
    all_data = []
    start = False  # skip pressure stack
    for j in data:
        if j and j[:2].lower() == "l0":
            if start:
                all_data.append(spot)
            spot = []
            start = True
        if start:
            spot.append(j)
    else:
        # do this to catch the last one
        all_data.append(spot)

    return header, all_data


def calc_cloud(chunk):
    cloud_index = 10
    cloud_vector = chunk[:, cloud_index]
    return np.average(cloud_vector)


def process_naturerun(args):
    # Example Nature_ver26_200609152100_177_label.npy

    try:
        # Nature Run
        nat_file = args[1]
        header, all_data = process_file(nat_file)

        if header == (0,0,""):
            raise FileNotFoundError("Empty nature run file")

        tempx = np.zeros((header[0], header[1], 16))
        # P, height above sea level, temp, water vapor, ozone density, water mixing, co,
        #  ch4, co2, ice mix, cloud fract, falling rain, falling snow, optical depth,
        #  optical depth2, ir optical depth
        # We care about 10 for calculating cloud fractions

        # Extra values
        # actual extras + cloud freq + header info
        num_extras = 35 + 1 + 7

        tempy = np.zeros((header[0], num_extras))
        # Lat, lon, land_fraction, pressure, surfance_temp, emmisivity, topog, cloud, extra1,
        #  extra2, extra3, extra4, extra5, PBLH, extra...

        for m, chunk in enumerate(all_data):
            chunk_data = process_chunk_nature(chunk)

            tempx[m, :, :] = np.array(chunk_data[2])
            cloud_fraction = calc_cloud(chunk_data[2])
            tempy[m, :] = np.hstack((chunk_data[0][:7], cloud_fraction, chunk_data[-1]))

        np.save(f"{base_dir}/{header[2][:12]}_{args[0]}.npy", tempx)
        np.save(f"{base_dir}/{header[2][:12]}_{args[0]}_label.npy", tempy)

    except Exception as e:
        print(f"Failed on {i} with error {str(e)}")
        with open("bad_radfiles.txt", 'a') as fp:
            fp.write(f"Failed on {i} with {str(e)}\n\n")

    return


def process_rad(args):
    # Example output HC_200609151200_081.npy

    try:
        temp_rad = args[1]

        #print(f"Starting {temp_rad}...")
        rad_files = []
        for i in ["H1", "HA", "HB", "HC", "HD", "HW", "MH"]:
            rad_files.append(temp_rad.replace("H1", i))

        for i in rad_files:
            header, all_data = process_file(i)

            if header == (0,0,""):
                raise FileNotFoundError("Empty radiances run file")

            tempx = np.zeros((header[0], header[1]))

            # extra 2 is for lat long
            tempy = np.zeros((header[0], 37))
            for m, chunk in enumerate(all_data):
                chunk_data = process_chunk_simple(chunk)


                tempx[m, :] = chunk_data[1]
                tempy[m, :] = chunk_data[0]

            np.save(f"{base_dir}/{header[2][:2]}_{args[0]}.npy", tempx)
            np.save(f"{base_dir}/{header[2][:2]}_{args[0]}_label.npy", tempy)

    except Exception as e:
        print(f"Failed on {i} with error {str(e)}")
        with open("bad_radfiles.txt", 'a') as fp:
            fp.write(f"Failed on {i} with {str(e)}\n\n")

    return


if __name__ == "__main__":
    """
    Can be used to process all H* data
    Modified to work with ATMS
    Also generates labels, but these are redundant since labels
    are the same for all H*

    TODO: Skip over NPY that already exist
    """

    #level2_path = "/data/nature_run/level2/v41/"
    level2_path = "/discover/nobackup/projects/hympi/rirosenb/crtm/clear_sky/output/forjames_mar2024/G5NR_strips/level2/v41"

    level2_files_all = glob.glob(f"{level2_path}/**/*.asc", recursive=True)
    level2_files = [x for x in level2_files_all if os.path.getsize(x) > 0]
    #print(len(level2_files))
    #print(level2_files[:10])

    combined = []
    for i in list(level2_files):
        dtime = re.search('\d{12}_\d{3}', i).group()
        combined.append([dtime, i])

    with Pool(num_workers) as p:
        p.map(process_naturerun, combined)


    #level1_path = "/data/nature_run/level1"
    #level1_path = "/discover/nobackup/projects/hympi/rirosenb/crtm/clear_sky/output/forjames_mar2024/G5NR_strips/level1"
    #level1_files_all = glob.glob(f"{level1_path}/**/*_L1_*[0-9]",
                                 recursive=True)

    #for i in ["H1", "HA", "HB", "HC", "HD", "HW", "MH"]:
    #    print(len([x for x in level1_files_all if i in x]))

    #level1_files_h1 = [x for x in level1_files_all if "H1" in x]
    #print(len(level1_files_h1))
    #print(level1_files_h1[:10])

    #combined = []
    #for i in list(level1_files_h1):
    #    dtime = re.search('\d{12}_\d{3}', i).group()
    #    combined.append([dtime, i])

    #with Pool(num_workers) as p:
    #    p.map(process_rad, combined)
