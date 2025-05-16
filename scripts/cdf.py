import sys

sys.path.insert(0, "..")  # add parent folder path where lib folder is

import mlflow
import hympi_ml.evaluation.nc as nc

# sets the tracking uri from where the mlflow run will be read from
mlflow.set_tracking_uri("/home/dgershm1/mlruns")

day = "20060315"
prefix = "NOISY"
dir_path = "/home/dgershm1/nc"

run_ids = [
    "a9770cc8d2754554999ac97ba658227e",
]

# generates the netcdf automatically with the given run_id (note that this run_id is specific to the tracking_uri data)
for run_id in run_ids:
    nc.generate_netcdf_from_run(
        run_id=run_id, day=day, dir_path=dir_path, name_prefix=prefix
    )
