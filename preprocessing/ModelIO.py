from typing import List, Dict, Any
from pydantic import BaseModel
import numpy as np

from tensorflow import convert_to_tensor
from tensorflow.keras.models import Model

from preprocessing.DataArray import DataArray


# A basic representation of input and output data in machine learning application
class ModelIO(BaseModel):
    features: List[DataArray]
    target: DataArray
    latlon: np.ndarray

    # pydantic config, allows use of ndarray as a type hint
    class Config:
        arbitrary_types_allowed = True

    # returns the features (transformed) as an ndarray (will turn the list into a single ndarray )
    def get_transformed_features(self) -> np.ndarray:
        x = [feature.transformed_data for feature in self.features]

        if len(x) == 1:
            x = x[0]

        return x

    def get_transformed_target(self) -> np.ndarray:
        return self.target.transformed_data

    # Runs the fit method on the given model and arguments
    def fit_model(self, model: Model, fit_args: Dict[str, Any]):
        x = self.get_transformed_features()
        y = self.get_transformed_target()

        model.fit(x, y, **fit_args)

    # Given a model and batch_size, runs a model.predict and returns a tuple in the form of (pred, truth)
    def predict_model(self, model: Model, batch_size: int = 10000) -> (np.ndarray, np.ndarray):
        x = self.get_transformed_features()

        pred = model.predict(x, batch_size=batch_size, verbose=False)
        pred = self.target.transformer.inverse_transform(pred) # transforms predicted target based on target transformer

        truth = self.target.data # note this is the original data (not transformed)

        return (pred, truth)
