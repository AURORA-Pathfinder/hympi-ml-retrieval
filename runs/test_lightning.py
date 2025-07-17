import lightning as L
from lightning.pytorch.loggers import MLFlowLogger

from torch import nn
from torch.utils.data import DataLoader

from torchmetrics import MetricCollection
import torchmetrics.regression as re

from hympi_ml.data import CosmirhSpec, AMPRSpec, NRSpec, ATMSSpec

from hympi_ml.data import cosmirh
from hympi_ml.data.ch06 import Ch06Source
from hympi_ml.data.model_dataset import get_split_datasets
from hympi_ml.model import MLPModel
from hympi_ml.utils import mlf

# data setup
features = {
    "CH": CosmirhSpec(
        frequencies=[
            cosmirh.C50_BAND,
            cosmirh.C183_BAND,
        ],
        ignore_frequencies=[  # problematic CRTM frequencies
            56.96674375,
            57.60736875,
            57.611275,
            57.61518125,
        ],
    ),
    # "ATMS": ATMSSpec(),
    # "AMPR": AMPRSpec(),
}

targets = {
    # "PBLH": NRSpec(
    #     dataset="PBLH",
    #     scale_range=(200, 2000),
    #     filter_range=(200, 2000),
    # )
    "TEMPERATURE": NRSpec(
        dataset="TEMPERATURE",
        scale_range=(175.0, 325.0),
    ),
    # "WATER_VAPOR": NRSpec(
    #     dataset="WATER_VAPOR",
    #     # scale_range=(1.11e-7, 3.08e-2),
    # ),
}

# metrics setup
train_metrics = {
    "TEMPERATURE": MetricCollection(
        {
            "mae": re.MeanAbsoluteError(),
        },
        prefix="train_",
    )
}

val_metrics = {
    "TEMPERATURE": MetricCollection(
        {
            "mae": re.MeanAbsoluteError(),
            "mse": re.MeanSquaredError(),
            "rmse": re.NormalizedRootMeanSquaredError(),
        },
        prefix="val_",
    )
}

test_metrics = {
    "TEMPERATURE": MetricCollection(
        {
            "mae": re.MeanAbsoluteError(),
            "mse": re.MeanSquaredError(),
            "rmse": re.NormalizedRootMeanSquaredError(),
        },
        prefix="test_",
    )
}

# define model
model = MLPModel(
    features=features,
    targets=targets,
    train_metrics=train_metrics,
    val_metrics=val_metrics,
    test_metrics=test_metrics,
    feature_paths={
        "CH": nn.Sequential(
            nn.LazyLinear(512),
            nn.GELU(),
            nn.LazyLinear(256),
            nn.GELU(),
            nn.LazyLinear(128),
            nn.GELU(),
        ),
        # "ATMS": nn.Sequential(
        #     nn.Linear(22, 32),
        #     nn.GELU(),
        #     nn.Linear(32, 32),
        #     nn.GELU(),
        # ),
        # "AMPR": nn.Sequential(
        #     nn.Linear(8, 8),
        #     nn.GELU(),
        # ),
    },
    output_path=nn.Sequential(
        nn.LazyLinear(128),
        nn.GELU(),
        nn.LazyLinear(72),
    ),
)

# define datasets
train, val, test = get_split_datasets(
    features=features,
    targets=targets,
    train_source=Ch06Source(
        days=[
            "20060115",
            "20060215",
            # "20060315", # removed due to space constraints
            "20060415",
            "20060515",
            "20060615",
            "20060715",
            "20060815",
            "20061015",
            # "20061115", # test
        ]
    ),
    validation_source=Ch06Source(days=["20061115"]),
    test_source=Ch06Source(days=["20061115"]),
    batch_size=8192,
)

# create loaders
loader_params = {
    "batch_size": None,
    "num_workers": 20,
    "pin_memory": True,
    "prefetch_factor": 1,
}

train_loader = DataLoader(train, shuffle=True, **loader_params)
val_loader = DataLoader(val, **loader_params)
test_loader = DataLoader(test, **loader_params)

# train!
tracking_uri = "/explore/nobackup/people/dgershm1/mlruns"

with mlf.start_run(
    tracking_uri=tracking_uri,
    experiment_name="+".join(targets.keys()),
    log_datasets=False,
) as run:
    mlf_logger = MLFlowLogger(
        tracking_uri=tracking_uri,
        run_id=run.info.run_id,
        log_model=True,
    )

    train.log("train")
    val.log("val")
    test.log("test")

    trainer = L.Trainer(
        enable_progress_bar=True,
        max_epochs=1,
        enable_model_summary=True,
        logger=mlf_logger,
    )

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
