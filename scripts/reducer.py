import glob
import numpy as np
import os

source = "/data/nature_run/fulldays_reduced"
sink = "/mnt/fastdata0/nature_run/fulldays_reduced_evenmore"

scale = 0.20

for i in glob.glob(f"{source}/*.npz"):
    data = np.load(i)
    # ['hsel', 'mh', 'scalar', 'table']
    keys = list(data.keys())

    print("Loading from " + i)
    mh = data["mh"]
    hsel = data["hsel"]
    table = data["table"]
    scalar = data["scalar"]

    num_rows = int(mh.shape[0] * scale)

    indices = np.array(
        sorted(np.random.choice(mh.shape[0], size=num_rows, replace=False))
    )

    dir_name = (
        i.strip(".npz").replace("fulldays_reduced/all_", "fulldays_reduced_evenmore/")
        + "/"
    )

    print("Saving data to " + dir_name + " directory...", end=" ")

    os.makedirs(dir_name, exist_ok=True)

    np.save(dir_name + "mh.npy", mh[indices, :])
    np.save(dir_name + "hsel.npy", hsel[indices, :])
    np.save(dir_name + "scalar.npy", scalar[indices, :])
    np.save(dir_name + "table.npy", table[indices, :, :])
    print("Done!")
