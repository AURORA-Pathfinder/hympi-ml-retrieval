import os
import numpy as np
import glob
from multiprocessing import Pool

os.umask(0o007)

in_path = "/explore/nobackup/people/dgershm1/pbl_fusion/ch06"
out_path = "/explore/nobackup/people/dgershm1/pbl_fusion/ch06"


def time_merge(args: tuple[str, str, str]):
    day, time, key = args

    out_file = f"{out_path}/{day}/{time}/{key}.npy"

    if not os.path.exists(out_file):
        files = glob.glob(f"{in_path}/{day}/{time}/*/{key}.npy")
        files = sorted(files)

        memmaps = [np.load(file, mmap_mode="r") for file in files]
        stack = np.vstack(memmaps)

        np.save(out_file, stack)


def ch_merge(args: tuple[str, str]):
    day, time = args

    try:
        out_file = f"{out_path}/{day}/{time}/CH.npy"
        print(f"Working on {day} at {time}!")

        if not os.path.exists(out_file):
            memmaps = []

            for key in ["C1", "C2", "C4", "C5", "C6", "C3"]:
                file = f"{in_path}/{day}/{time}/{key}.npy"

                if not os.path.exists(file):
                    print(f"Couldn't find {key} in {day} at {time}!")
                    return

                memmaps.append(np.load(file, mmap_mode="r"))

            ch = np.hstack(memmaps)

            print(f"Final save of {out_file}!")
            np.save(out_file, ch)
        else:
            print(f"Output file already exists for {day} at {time}!")

        # check to see if new file (load with memmap) contains all previous subsets
        # delete subset file if all parts are in correct place
        if os.path.exists(f"{out_path}/{day}/{time}/C1.npy"):
            ch_memmap = np.load(out_file, mmap_mode="r")

            c1 = np.load(f"{out_path}/{day}/{time}/C1.npy", mmap_mode="r")
            c2 = np.load(f"{out_path}/{day}/{time}/C2.npy", mmap_mode="r")
            c3 = np.load(f"{out_path}/{day}/{time}/C3.npy", mmap_mode="r")
            c4 = np.load(f"{out_path}/{day}/{time}/C4.npy", mmap_mode="r")
            c5 = np.load(f"{out_path}/{day}/{time}/C5.npy", mmap_mode="r")
            c6 = np.load(f"{out_path}/{day}/{time}/C6.npy", mmap_mode="r")

            c50_match = ch_memmap[0][:2048] == np.hstack((c1[0], c2[0]))
            c183_match = ch_memmap[0][2048:6144] == np.hstack((c4[0], c5[0], c6[0]))
            window_match = ch_memmap[0][6144:] == c3[0]

            if c50_match.all() and c183_match.all() and window_match.all():
                print(f"Sucessful check for {day} at {time}! Deleting subsets...")
                os.remove(f"{out_path}/{day}/{time}/C1.npy")
                os.remove(f"{out_path}/{day}/{time}/C2.npy")
                os.remove(f"{out_path}/{day}/{time}/C3.npy")
                os.remove(f"{out_path}/{day}/{time}/C4.npy")
                os.remove(f"{out_path}/{day}/{time}/C5.npy")
                os.remove(f"{out_path}/{day}/{time}/C6.npy")
            else:
                print(f"BAD MERGE for {day} at {time}!")
    except Exception as e:
        print(f"ERROR! for {day} at {time}: {e}")


days = [
    "20060115",
    "20060215",
    "20060315",
    "20060415",
    "20060515",
    "20060615",
    "20060715",
    "20060815",
    # "20060915",
    # "20061015",
    # "20061115",
    # "20061215",
]

times = [
    "0000",
    "0300",
    "0600",
    "0900",
    "1200",
    "1500",
    "1800",
    "2100",
]


# ch_merge(("20060115", "0000"))

proc_list = [(day, time) for day in days for time in times]

for proc in proc_list:
    ch_merge(proc)

# with Pool(processes=2) as p:
#     p.map(ch_merge, proc_list)

print("done!")
