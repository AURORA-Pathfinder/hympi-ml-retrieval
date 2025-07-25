import mlflow
import lightning as L
from torchmetrics import MetricCollection
import torchmetrics.regression as re
import matplotlib.pyplot as plt

from hympi_ml.utils import mlf
from hympi_ml.model import MLPModel
from hympi_ml.data import NRSpec
from hympi_ml.data.filter import SimpleRangeFilter
from hympi_ml.data.batches import RawDataModule
import hympi_ml.evaluation.figs as figs

# MLFlow setup
tracking_uri = "/explore/nobackup/people/dgershm1/mlruns"
mlflow.set_tracking_uri(tracking_uri)

# set the id of the run we want and get the path for the best checkpoint
run_id = "166a8b9bb76840769b63be777a0eae30"
cpath = mlf.get_checkpoint_path(run_id)

# get the mod and training data from the checkpoint
model = MLPModel.load_from_checkpoint(cpath)
raw_data = RawDataModule.load_from_checkpoint(cpath)

extras = {
    "PBLH": NRSpec(
        dataset="PBLH",
        filter=SimpleRangeFilter(minimum=200, maximum=500),
    ),
}

raw_data.spec.extras = extras
model.spec.extras = extras

# test metrics setup
model.log_metrics = False  # important to avoid logging tensor metrics
model.unscale_metrics = True  # automatically unscales

model.test_metrics = {
    "TEMPERATURE": MetricCollection(
        {
            "mae_profile": re.MeanAbsoluteError(num_outputs=72),
        }
    )
}

# set up trainer and run test!
trainer = L.Trainer(enable_progress_bar=True)
trainer.test(model, dataloaders=raw_data.test_dataloader())

# compute the metrics after aggregating on test dataset
computed_metrics = model.test_metrics["TEMPERATURE"].compute()

# create figure for each metric and log it to them mlflow run artifacts
for k, metric in computed_metrics.items():
    figs.plot_profiles({k: metric.cpu()}, value_axis=k)
    local_path = f"/tmp/{k}_pblh.png"
    plt.savefig(local_path)
    mlflow.log_artifact(local_path, "metric_figures", run_id=run_id)
