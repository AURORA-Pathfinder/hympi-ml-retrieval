import os
from typing import Mapping

os.environ["KERAS_BACKEND"] = "torch"

import keras

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from hympi_ml.data import cosmirh, nature_run, model_dataset
from hympi_ml.data.ch06 import Ch06Source
from hympi_ml.data.base import DataSpec

train_source = Ch06Source(days=[
    "20060115",
    "20060215",
    "20060315",
    "20060415",
    "20060515",
    "20060615",
    "20060715",
    # "20060815",
    
    # The rest is not transferred! I ran out of quota :(
    # "20060915",
    # "20061015",
    # "20061115",
    # "20061215",
])

test_source = Ch06Source(days = ["20060815"])

features = {
    "C1" : cosmirh.CosmirhSpec(subset=1),
    "C2" : cosmirh.CosmirhSpec(subset=2),
    "C3" : cosmirh.CosmirhSpec(subset=3),
    "C4" : cosmirh.CosmirhSpec(subset=4),
    "C5" : cosmirh.CosmirhSpec(subset=5),
    "C6" : cosmirh.CosmirhSpec(subset=6),
}

targets = {
    "TEMPERATURE" : nature_run.NRSpec(dataset="TEMPERATURE", scale_range=(175.0, 325.0)), 
    "WATER_VAPOR" : nature_run.NRSpec(dataset="WATER_VAPOR", scale_range=(1.11e-7, 3.08e-2)), 
}

train_ds = model_dataset.ModelDataset(train_source, features, targets, batch_size = 8192)
train_dataloader = DataLoader(train_ds, batch_size=None, shuffle=True, num_workers=10, pin_memory=True)

test_ds = model_dataset.ModelDataset(test_source, features,targets, batch_size = 16384)
test_dataloader = DataLoader(test_ds, batch_size=None, shuffle=False, num_workers=10, pin_memory=True)

path_outputs = []

inputs = train_ds.get_input_layers()

for name, layer in inputs.items():
    path_out = layer
    
    # path_out = keras.layers.LayerNormalization()(path_out)
    # path_out = keras.layers.Dense(512, "relu")(path_out)
    path_out = keras.layers.Dense(100, "relu")(path_out)
    
    path_outputs.append(path_out)
    
out = keras.layers.Concatenate()(path_outputs)
out = keras.layers.Dense(100, "relu")(out)
out = keras.layers.Dense(100, "relu")(out)

outputs = {k : (v)(out) for k, v in train_ds.get_output_layers().items()}

model = keras.Model(inputs, outputs)

model.summary()
model.compile(optimizer="adam", loss="mae")

model.fit(train_dataloader, epochs=2, validation_data=test_dataloader)