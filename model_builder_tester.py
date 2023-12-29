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

    config.model.summary()

    # config.model.compile(**config.compile_args)

    # trainIO.fit_model(config.model, config.fit_args)


if __name__ == "__main__":
    main()
