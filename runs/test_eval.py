import mlflow
import lightning as L
from torch.utils.data import DataLoader
import torchmetrics.regression

from hympi_ml.utils import mlf
import hympi_ml.data.model_dataset as md
from hympi_ml.model import MLPModel

# # from hympi_ml.data import RFBand
# # import hympi_ml.data.cosmirh as c
from hympi_ml.data.ch06 import Ch06Source
import torchmetrics

mlflow.set_tracking_uri("/explore/nobackup/people/dgershm1/mlruns")

run_id = "c45536491e8840429abc77e9764d0eed"
datasets = md.get_datasets_from_run(run_id)
model = MLPModel.load_from_checkpoint(mlf.get_checkpoint_path(run_id))

print(model)

val = datasets["val"]

loader = DataLoader(val, batch_size=None, num_workers=9)


metric = torchmetrics.regression.MeanAbsoluteError(num_outputs=72)
model.test_metrics["mae_prof"] = metric

trainer = L.Trainer(
    enable_progress_bar=True,
    # max_epochs=1,
    enable_model_summary=True,
    # logger=mlf_logger,
)

trainer.test(model, dataloaders=loader)

print(metric.compute())
