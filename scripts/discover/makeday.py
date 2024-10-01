import re
import pickle
from os.path import isfile
import numpy as np
from glob import glob


def getmem(x):
    return x.nbytes/2**30


def get_files_for_day(day, base="../processed"):
    return sorted(glob(f"{base}/*{day}*"))


def process_day(day, band="Nature"):
    files = get_files_for_day(day)

    table_ind = np.r_[0, 2, 3]
    scalr_ind = np.r_[0, 1, 3, 4, 5, 7, 13]

    nat_scalar = []
    nat_table = []
    for i in files:
        delta = int(re.search("\d{12}", i).group())

        if band in i and band == "Nature":
            print(i)
            nature = np.load(i)
            nat_table.append(nature['x'][:, :, table_ind])
            scale_tmp = nature['y'][:, :]
            nat_scalar.append(np.hstack([scale_tmp,
                                         np.repeat(delta,
                                                   scale_tmp.shape[0]).reshape(-1, 1)]))

        elif band in i:
            print(i)
            data = np.load(i)
            nat_table.append(data['x'])

    if band == "Nature":
        np.savez(f"/home/jmackin1/hympi/jmackin1/fulldays/{band}_{day}.npz",
                 table=np.vstack(nat_table),
                 scalar=np.vstack(nat_scalar))
    else:
        np.savez(f"/home/jmackin1/hympi/jmackin1/fulldays/{band}_{day}.npz",
                 table=np.vstack(nat_table))


if __name__ == "__main__":
    days = ["20060315", "20060515", "20060615", "20060715",
            "20060803", "20060815", "20060915", "20061015",
            "20061115", "20061215"]

    #bands = ["Nature", "H1", "HA", "HB", "HC",
    #         "HD", "HW", "MH"]
    bands = ["Nature"]

    if isfile("done.pkl"):
        done = pickle.load(open("done.pkl", 'rb'))
    else:
        done = []

    for day in days:
        for band in bands:
            if day+band in done:
                print(f"{day+band} Skipped")
                continue

            try:
                process_day(day, band)
                done.append(day+band)
                pickle.dump(done, open("done.pkl", 'wb'))

            except Exception as e:
                with open("log.txt", 'a') as fp:
                    fp.write(f"Bad write on {day} {band}\n {e}")
