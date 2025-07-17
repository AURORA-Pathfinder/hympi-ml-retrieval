import mlflow
import lightning as L
from torch.utils.data import DataLoader
import torchmetrics.regression
import numpy as np
from lightning.pytorch.loggers import MLFlowLogger


from hympi_ml.utils import mlf
import hympi_ml.data.model_dataset as md
from hympi_ml.model import MLPModel

# # from hympi_ml.data import RFBand
# # import hympi_ml.data.cosmirh as c
from hympi_ml.data.ch06 import Ch06Source
import torchmetrics

tracking_uri = "/explore/nobackup/people/dgershm1/mlruns"
mlflow.set_tracking_uri(tracking_uri)

run_id = "c45536491e8840429abc77e9764d0eed"
datasets = md.get_datasets_from_run(run_id)
model = MLPModel.load_from_checkpoint(mlf.get_checkpoint_path(run_id))

print(model)

val = datasets["val"]

loader = DataLoader(val, batch_size=None, num_workers=9)

metric = torchmetrics.regression.MeanAbsoluteError(num_outputs=72)
model.test_metrics["mae_prof"] = metric

mlf_logger = MLFlowLogger(
    tracking_uri=tracking_uri,
    run_id=run_id,
)

trainer = L.Trainer(
    enable_progress_bar=True,
    # max_epochs=1,
    enable_model_summary=True,
    logger=mlf_logger,
)

model.eval()

trainer.test(model, dataloaders=loader)


local_path = f"/tmp/TEMPERATURE.npy"
np.save(local_path, v.cpu().numpy())
mlflow.log_artifact(local_path, f"{context}_metrics", run_id=self.logger.run_id)
