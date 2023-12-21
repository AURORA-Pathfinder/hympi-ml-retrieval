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
    config_name = "ATMS+BSL"


@hydra.main(config_path=config_path, config_name=config_name, version_base=None)
def main(cfg: DictConfig):

    print(f"Starting overall run {config_name}, from {config_path}")

    print("Preprocessing...")
    config: Config = hydra.utils.instantiate(cfg.config)

    current_run = start_run(config.mlflow_experiment_name, log_datasets=False)
    print(f"Starting MLFlow Run {current_run} ...")

    log_config(config_path, config_name)

    (trainIO, testIO) = config.modelIOs

    print(trainIO.latlon.shape)

    print("Logging datasets...")
    log_dataset(config.dataset_name, "train", trainIO)

    config.model.summary()

    config.model.compile(**config.compile_args)

    trainIO.fit_model(config.model, config.fit_args)

    print("Evaluating model...")
    config_dict = {"name": config_name}
    config.evaluator.set_current_run(current_run, config_dict)

    # test eval
    (test_pred, test_truth) = testIO.predict_model(config.model)
    config.evaluator.evaluate(test_pred, test_truth, testIO.latlon,
                              "test", config.evaluator.show_plot)

    # train eval
    (train_pred, train_truth) = trainIO.predict_model(config.model)
    config.evaluator.evaluate(train_pred, train_truth, trainIO.latlon,
                              "train", config.evaluator.show_plot)

    print("Run complete!")

    end_run()


if __name__ == "__main__":
    main()
