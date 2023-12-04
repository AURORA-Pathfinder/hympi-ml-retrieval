import hydra
from mlflow_logging import start_run, log_config, log_dataset, end_run
from omegaconf import DictConfig
from conf.config import Config

import os
os.environ["CUDA_VISIBLE_DEVICES"]="6"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

config_path = "conf/Temperature Profile"
config_name = "HSEL+BSL"

@hydra.main(config_path=config_path, config_name=config_name, version_base=None)
def main(cfg: DictConfig):
    print("Preprocessing...")
    config: Config = hydra.utils.instantiate(cfg.config)

    print("Starting MLFlow Run...")
    current_run = start_run(config.mlflow_experiment_name, log_datasets = False)
    log_config(config_path, config_name)
    
    (trainIO, testIO) = config.modelIOs
    
    print("Logging datasets...")
    log_dataset(config.dataset_name, "train", trainIO)

    config.model.summary()

    config.model.compile(**config.compile_args)

    trainIO.fit_model(config.model, config.fit_args)
    
    print("Evaluating model...")
    config.evaluator.set_current_run(current_run)

    # test eval
    (test_pred, test_truth) = testIO.predict_model(config.model)
    config.evaluator.evaluate(test_pred, test_truth, "test")

    # train eval
    (train_pred, train_truth) = trainIO.predict_model(config.model)
    config.evaluator.evaluate(train_pred, train_truth, "train")

    print("Run complete!")

    end_run()

if __name__ == "__main__":
    main()
