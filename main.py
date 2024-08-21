import hydra
from hydra.core.hydra_config import HydraConfig
from mlflow_logging import start_run, log_config, log_dataset, end_run
from omegaconf import DictConfig
from conf.config import Config
from model_creation.loss import weighted_mae

import os

# TODO get best gpu by function
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

config_path = "conf/WaterV"
config_name = ""


@hydra.main(config_path=config_path, config_name=config_name, version_base=None)
def main(cfg: DictConfig):
    hydra_cfg = HydraConfig.get()
    config_name = hydra_cfg['job']['config_name']

    print(cfg)

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

    # TODO Figure this out for config
    # config.model.compile(**config.compile_args)
    #config.model.compile(optimizer='adam', loss=weighted_mae)
    config.model.compile(optimizer='adam', loss="mae")
    config.model.summary()

    trainIO.fit_model(config.model, config.fit_args)

    print("Evaluating model...")
    config_dict = {"name": config_name}
    config.evaluator.set_current_run(current_run, config_dict)

    # test eval

    (test_pred, test_truth, test_surf) = testIO.predict_model(config.model)
    print(dir(test_surf))
    config.evaluator.evaluate(test_pred, test_truth, testIO.latlon,
                              test_surf.data, "test", config.evaluator.show_plot)

    # train eval
    (train_pred, train_truth, train_surf) = trainIO.predict_model(config.model)
    config.evaluator.evaluate(train_pred, train_truth, trainIO.latlon,
                              train_surf.data, "train", config.evaluator.show_plot)

    print("Run complete!")

    # ml_flow method
    end_run()


if __name__ == "__main__":
    main()
