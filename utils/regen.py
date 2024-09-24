import numpy as np
from preprocessing.fulldays import FullDaysLoader, DataName
from rich.progress import track

days = [
    "20060315", "20060515", "20060615",
    "20060715", "20060915", "20061015",
    "20061115", "20061215", "20060815",
    "20060803"
];

loader = FullDaysLoader(days)

for day in track(days):
    dir_path = f"/data/nature_run/fulldays_reduced_evenmore/{day}"
    hsel = loader.find_memmap(day, DataName.hsel)
    hsel_indices = ~np.in1d(range(1957), range(1934, 1937))
    np.save(f"{dir_path}/hsel_new.npy", hsel[:, hsel_indices])
