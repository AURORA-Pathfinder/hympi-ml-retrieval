import pandas as pd
import glob


pickle_location = "/home/jmackin1/hympi/jmackin1/cpl"

pickle_out = "/home/jmackin1/hympi/jmackin1/cpl_merged"


paths = sorted(glob.glob(f"{pickle_location}/*.pkl"))
#for i in paths:
#    print(i)

for day in ["20060315", "20060515", "20060615", "20060715",  "20060803",
            "20060815",  "20060915",  "20061015",  "20061115",  "20061215"]:
    print(day)
    current_day_paths = [x for x in paths if day in x]
    dfs = []
    for i in current_day_paths:
        time = i.split("_")[-2][-4:]
        df = pd.read_pickle(i)
        df['time'] = time
        dfs.append(df)
    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df.to_pickle(f"{pickle_out}/cpl_all_{day}.pkl")
