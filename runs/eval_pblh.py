import mlflow
import lightning as L
from torchmetrics import MetricCollection
import matplotlib.pyplot as plt

from hympi_ml.utils import mlf
from hympi_ml.model import MLPModel
from hympi_ml.data.batches import RawDataModule
from hympi_ml.evaluation.metrics import ErrorHistogram
from hympi_ml.evaluation import figs

from torchmetrics import Metric
import torch


tracking_uri = "/explore/nobackup/people/dgershm1/mlruns"
mlflow.set_tracking_uri(tracking_uri)

run_id = "70e70dc1e33f4df9b593cea9f5b7822b"
cpath = mlf.get_checkpoint_path(run_id)

model = MLPModel.load_from_checkpoint(cpath)
raw_data = RawDataModule.load_from_checkpoint(cpath)

model.test_metrics = {
    "PBLH": MetricCollection(
        {
            "error_histogram": ErrorHistogram(num_bins=100, min=-1000, max=1000),
        }
    )
}

trainer = L.Trainer(enable_progress_bar=True)

loader = raw_data.test_dataloader()

model.log_metrics = False
model.unscale_metrics = True

trainer.test(model, dataloaders=loader)

metric: ErrorHistogram = model.test_metrics["PBLH"]["error_histogram"]

metric.plot()
local_path = "/tmp/error_histogram_pblh.png"
plt.savefig(local_path)
mlflow.log_artifact(local_path, "metric_figures", run_id=run_id)
