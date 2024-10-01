import pickle
import numpy as np
from multiprocessing import Pool, Lock

# Mutex for writing problematic file
mutex = Lock()


def process_singleband_daydelta(args):
    # Load useful info
    base = "/home/jmackin1/hympi/jmackin1/data"
    processed = "/home/jmackin1/hympi/jmackin1/processed"
    good_files = pickle.load(open("good_files.pkl", 'rb'))

    # Args
    day = args[0]
    delta = args[1]
    band = args[2]

    try:
        # Load all npys
        tmpx = []
        tmpy = []
        for i in range(1, 181):
            current_delta = f"{day}{delta}_{i:03}"
            if current_delta in good_files:
                tmpx.append(np.load(f"{base}/{band}_{current_delta}.npy"))
                tmpy.append(np.load(f"{base}/{band}_{current_delta}_label.npy"))

        x = np.vstack(tmpx)
        y = np.vstack(tmpy)

        np.savez(f"{processed}/{band}_{day}{delta}.npz", x=x, y=y)
        print(f"Wrote {processed}/{band}_{day}{delta}.npz")

        return True

    except Exception as e:
        print(f"Failed to write {processed}/{band}_{day}{delta}.npz")
        print(f"Failed with error: {e}")
        with mutex:
            with open("redo.pkl", 'rb') as f:
                redo_list = pickle.load(f)

            redo_list.append((day, delta, band))

            with open("redo.pkl", 'wb') as f:
                pickle.dump(redo_list, f)

        return False


if __name__ == "__main__":
    #prefix = ["H1", "HA", "HB", "HC", "HD", "HW", "MH", "Nature_ver41"]
    prefix = ["Nature_ver41"]
    days = ['20060815', '20060803', '20060615', '20061015', '20060315',
            '20060515', '20061115', '20060715', '20061215', '20060915']
    times = ['0600', '0300', '1200', '0000', '0900', '2100', '1500', '1800']

    # Initial Run
    args = []
    for day in days:
        for delta in times:
            for band in prefix:
                args.append((day, delta, band))

    print("Starting first attempt.")
    with Pool(14) as p:
        outdata = p.map(process_singleband_daydelta, args)

    print(f"Succesfully ran {sum(outdata)} deltas out of {len(outdata)}.")

    # Redo run
    core_count = 10
    attempt = 0
    while True:

        print(f"Starting attempt {attempt+1} on {core_count} cores.")
        attempt += 1

        # get list to redo
        with open("redo.pkl", 'rb') as f:
            redo_list = pickle.load(f)

        if len(redo_list) == 0:
            print("Done... Flawless Victory!")
            break

        # Purge list, hopefully this gets shorter each time
        with open("redo.pkl", 'wb') as f:
            pickle.dump([], f)

        with Pool(core_count) as p:
            outdata = p.map(process_singleband_daydelta, redo_list)

        if sum(outdata) == len(redo_list) or len(outdata) <= 0:
            print("DONE!")
            break
        else:
            # Adjust core count if fail
            core_count -= 2
            if core_count < 1:
                core_count = 1
