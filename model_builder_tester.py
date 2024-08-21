import hydra
from mlflow_logging import start_run, log_config, log_dataset, end_run
from omegaconf import DictConfig
from conf.config import Config

import os
import sys

# TODO get best gpu by function
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

config_path = "conf/Temperature Profile"

if len(sys.argv) > 1:
    config_name = sys.argv[1]
else:
    config_name = "MODELONLY_TEST"


@hydra.main(config_path=config_path, config_name=config_name, version_base=None)
def main(cfg: DictConfig):

    config: Config = hydra.utils.instantiate(cfg.config)

    # netcdf evaluator now built into the system
    # Dry run mode... able to test various configuration quickly
    # built dataset, 8640 nature run, 3619 level 1 (per)
    #  Probably only generated rad data for naturerun files that have data in them

    # TODO Surface Pressure
    # TODO Migrate to new dataset
    # TODO Loss Function - related to dry run modes
    # TODO Data science

    config.model.summary()

    # Investigate modelIOs
    print(type(config.modelIOs), len(config.modelIOs))

    feature_processor = config.modelIOs[0]

    # datarray
    print(feature_processor.features[-1].shape)

    # config.model.compile(**config.compile_args)

    # trainIO.fit_model(config.model, config.fit_args)


if __name__ == "__main__":
    main()
