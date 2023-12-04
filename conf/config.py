from typing import Dict, Any, Tuple
from dataclasses import dataclass
from evaluation.Evaluator import Evaluator
from preprocessing.ModelIO import ModelIO

from tensorflow.keras.models import Model

@dataclass
class Config:
    mlflow_experiment_name: str
    dataset_name: str
    modelIOs: Tuple[ModelIO]
    model: Model
    compile_args: Dict[str, Any]
    fit_args: Dict[str, Any]
    evaluator: Evaluator
