from typing import Optional

import numpy as np
from h5netcdf.legacyapi import Dataset
from keras.models import Model

from hympi_ml.data.fulldays.dataset import FullDaysDataset, get_datasets_from_run
from hympi_ml.data.fulldays.loading import DPath, DKey
from hympi_ml.utils.gpu import set_gpus
import hympi_ml.utils.mlflow_log as mlflow_log

set_gpus(min_free=0.8)


def generate_netcdf(
    day: str,
    model: Model,
    train: FullDaysDataset,
    test: FullDaysDataset,
    validation: Optional[FullDaysDataset],
    cloud_fraction: float,
    name_prefix: str,
    path: str = "/data/nature_run/nc",
):
    """
    Generates a NetCDF file represnting the prediction for a single day worth of data.

    Args:
        day (str): The day in the string format YYYYMMDD
        model (Model): The keras model to run predictions
        train (FullDaysDataset): The training dataset for metadata
        test (FullDaysDataset): The test dataset for metadata
        validation (FullDaysDataset): The validation dataset for metadata
        cloud_fraction (float): The cloud fraciton of the train, test, and validation datasets
        name_prefix (str): The prefix applied to the output file name
        path (str): The path where the netcdf file will be generated. Defaults to "/data/nature_run/nc".
    """

    train_days = train.days
    day_context = "train"

    if day in test.days:
        day_context = "test"
    elif validation is not None and day in validation.days:
        day_context = "val"

    day_dataset = train
    day_dataset.set_days([day])

    feature_names = "+".join(list(day_dataset.feature_specs.keys()))
    target_names = "+".join(list(day_dataset.target_shapes.keys()))

    file_path = f"{path}/{name_prefix}_{day_context}_{day}_{feature_names}=={target_names}.nc"

    with Dataset(file_path, "w") as nc:
        nc.title = "Truth vs. Predicted Profiles"
        nc.run_config = name_prefix

        nc.training_days = ",".join(train_days)
        if validation is not None:
            nc.validation_days = ", ".join(validation.days)
        nc.test_days = ",".join(test.days)

        nc.cf = str(cloud_fraction)

        nc.model_type = f"{feature_names}=>{target_names}"

        preds = day_dataset.predict(model, unscale=True)
        truths = day_dataset.get_targets(scaling=False)

        num_rows = list(truths.values())[0].shape[0]
        nc.createDimension("row", num_rows)
        nc.createDimension("z", 72)  # the 72-sigma level dimension

        for name in truths.keys():
            dims = ("row",)

            if truths[name].ndim == 2:
                dims = ("row", "z")

            z_true = nc.createVariable(f"{name}_true", np.float64, dims)
            z_pred = nc.createVariable(f"{name}_pred", np.float64, dims)

            if len(dims) == 2:
                z_true[:, :] = truths[name][:]
                z_pred[:, :] = preds[name][:]
            elif len(dims) == 1:
                z_true[:] = truths[name][:]
                z_pred[:] = preds[name][:]

        spr = nc.createVariable("surface_pressure", np.float64, ("row",))
        lat = nc.createVariable("lat", np.float64, ("row",))
        lon = nc.createVariable("lon", np.float64, ("row",))

        lat_lon_spress = np.array(
            list(
                day_dataset.as_tf_dataset(keys=[DKey.LATITUDE, DKey.LONGITUDE, DKey.SURFACE_PRESSURE])
                .map(lambda x: list(x.values()))
                .as_numpy_iterator()
            )
        )

        lat[:] = lat_lon_spress[:, 0]
        lon[:] = lat_lon_spress[:, 1]
        spr[:] = lat_lon_spress[:, 2]


def generate_netcdf_from_run(run_id: str, day: str, name_prefix: str | None = None):
    loaded_model = mlflow_log.get_autolog_model(run_id)

    datasets = get_datasets_from_run(run_id)
    test = datasets["test"]

    if "validation" in datasets.keys():
        validation = datasets["validation"]
    else:
        validation = None

    train = datasets["train"]

    # NOTE This is a temporary solution for getting cloud_fraction, this should be cleaner
    cloud_fraction = 1.0

    if train._data_path == DPath.CPL_06_REDUCED:
        cloud_fraction = 0.1

    generate_netcdf(
        day=day,
        model=loaded_model,
        train=train,
        test=test,
        validation=validation,
        cloud_fraction=cloud_fraction,
        name_prefix=f"{run_id}_{name_prefix}",
    )
