import os
import sys
import numpy as np

print("Starting..")

base_path = "/data/nature_run/fulldays"
out_path = "/data/nature_run/fulldays_reduced"
allowed_cf = float(sys.argv[1])

all_files = os.listdir(base_path)
times = set()
for i in all_files:
    times.add(i.split("_")[-1].split(".")[0])

for i in sorted(list(times)):
    print(f"Processing {i}")

    # HSEL
    hsel_file = np.load(f"{base_path}/hympi_{i}.npz")
    hsel_keys = sorted(list(hsel_file.keys()))
    hsel_data = [hsel_file[x] for x in hsel_keys]
    hsel_all = np.hstack(hsel_data)

    # MH
    mh_data = np.load(f"{base_path}/MH_{i}.npz")["table"]

    # Natty
    nrun_data = np.load(f"{base_path}/Nature_{i}.npz")
    nrun_tab = nrun_data["table"]
    nrun_sca = nrun_data["scalar"]

    print(hsel_all.shape,
          hsel_data[0].shape,
          mh_data.shape,
          nrun_tab.shape,
          nrun_sca.shape)

    cf = nrun_sca[:, 24]
    indices = cf <= allowed_cf
    print(np.sum(indices)*100/len(indices), "% remaining after culling")

    np.savez(f"{out_path}/all_{i}.npz",
             hsel=hsel_all[indices, :],
             mh=mh_data[indices, :],
             scalar=nrun_sca[indices, :],
             table=nrun_tab[indices, :, :])
