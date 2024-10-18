from typing import Optional

import numpy as np
from h5netcdf.legacyapi import Dataset
from keras.models import Model

import hympi_ml.data.fulldays as fd
from hympi_ml.data.fulldays.dataset import FullDaysDataset
from hympi_ml.utils.gpu import set_gpus
import hympi_ml.utils.mlflow_log as mlflow_log

set_gpus(count=1)


def generate_netcdf(
    day: str,
    model: Model,
    train: FullDaysDataset,
    test: FullDaysDataset,
    validation: Optional[FullDaysDataset],
    cloud_fraction: float,
    name_prefix: str,
):
    """
    Generates a NetCDF file represnting the prediction for a single day worth of data.

    Note that the files will always generate at the /data/nature_run/nc directory.

    Args:
        day (str): The day in the string format YYYYMMDD
        model (Model): The keras model to run predictions
        train (FullDaysDataset): The training dataset for metadata
        test (FullDaysDataset): The test dataset for metadata
        validation (FullDaysDataset): The validation dataset for metadata
        cloud_fraction (float): The cloud fraciton of the train, test, and validation datasets
        name_prefix (str): The prefix applied to the output file name
    """

    train_days = train.days
    day_context = "train"

    if day in test.days:
        day_context = "test"
    elif validation is not None and day in validation.days:
        day_context = "val"

    day_dataset = train
    day_dataset.set_days([day])

    instrument = list(day_dataset.features.keys())[0]
    target_name = day_dataset.target_name

    nc_path = "/data/nature_run/nc"
    file_path = f"{nc_path}/{name_prefix}_{day_context}_{target_name}_{day}_{instrument}.nc"

    with Dataset(file_path, "w") as nc:
        nc.title = "Truth vs. Predicted Profiles"
        nc.run_config = name_prefix

        nc.training_days = ",".join(train_days)
        if validation is not None:
            nc.validation_days = ", ".join(validation.days)
        nc.test_days = ",".join(test.days)

        nc.cf = str(cloud_fraction)  # TODO: Include CF in fulldays?

        nc.model_type = f"{instrument} {target_name}"

        num_rows = day_dataset.count
        nc.createDimension("row", num_rows)

        nc.createDimension("z", day_dataset.target_shape[0])

        z_true = nc.createVariable("profile_true", np.float64, ("row", "z"))
        z_pred = nc.createVariable("profile_pred", np.float64, ("row", "z"))
        spr = nc.createVariable("surface_pressure", np.float64, ("row",))
        lat = nc.createVariable("lat", np.float64, ("row",))
        lon = nc.createVariable("lon", np.float64, ("row",))

        (data_lat, data_lon) = day_dataset.get_latlon()
        data_spress = day_dataset.loader.get_data(fd.DKey.SURFACE_PRESSURE)

        lat[:] = data_lat[:]
        lon[:] = data_lon[:]
        spr[:] = data_spress[:]

        z_true[:, :] = day_dataset.target[:]

        pred = day_dataset.predict(model)
        z_pred[:, :] = pred


def generate_netcdf_from_run(run_id: str, day: str, cloud_fraction: float = 0.1):
    loaded_model = mlflow_log.get_autolog_model(run_id)

    datasets = fd.get_datasets_from_run(run_id)
    test = datasets["test"]

    if "validation" in datasets.keys():
        validation = datasets["validation"]
    else:
        validation = None

    train = datasets["train"]

    generate_netcdf(
        day=day,
        model=loaded_model,
        train=train,
        test=test,
        validation=validation,
        cloud_fraction=cloud_fraction,
        name_prefix=f"{run_id}",
    )
