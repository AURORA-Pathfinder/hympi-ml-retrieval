from __future__ import annotations

from numpy import ndarray

from sklearn.preprocessing import FunctionTransformer


# an array of data associated with a transformer
class DataArray:

    # Initializes a new DataArray and creates the transformed data
    def __init__(self, data: ndarray,
                 transformer: FunctionTransformer) -> None:

        # Reshapes the data if it is not 2D (required for transformation)
        if data.shape[0] == 1:
            # one sample with multiple values
            data = data.reshape(1, -1)
        elif len(data.shape) > 1 and data.shape[1] == 1:
            # multiple samples with one value
            data = data.reshape(-1, 1)

        self.data = data
        self.transformed_data = transformer.fit_transform(data)

        self.transformer = transformer

    def set_datasource(self, source: str):
        self.source = source
