from typing import Dict, Any

from keras import Input, Model, layers
import numpy as np


def create_identity_model(input_shapes: Dict[str, tuple]) -> Model:
    """
    Creates a model that outputs its inputs. This is meant to be a model that can quickly
    take the inputs and output them using the GPU by using this model's predict function.

    Args:
        input_shapes (Dict[str, tuple]): A set of input shapes with a string as the name of the input (often a DKey).

    Returns:
        Model: The created identity model
    """
    inputs = [Input(shape, name=name) for name, shape in input_shapes.items()]
    outs = [layers.Lambda(lambda x: x)(inp) for inp in inputs]

    return Model(inputs, outs)


def predict_dict(model: Model, dataset: Any) -> Dict[str, np.ndarray]:
    """
    Predicts on the given model with the provided input but organizes the output
    as a dictionary based on the names of the outputs in the model.

    This also organizes the outputs to ensure scalar outputs are flattened properly.

    Args:
        model (Model): The model to calculate predictions.
        dataset (Any): The prediction input (either a tf.data.Dataset or a keras Sequence or some other format)

    Returns:
        Dict[str, np.ndarray]: The dictionary of outputs with keys as target names.
    """
    preds = model.predict(dataset)

    if len(model.outputs) == 1:
        preds = [preds]

    # generate the dictionary of target ndarray data
    preds_dict = {}

    for i, name in enumerate(model.output_names):
        # flatten for any scalar targets
        if preds[i].ndim > 1 and len(preds[i][0]) == 1:
            preds[i] = preds[i].flatten()

        preds_dict[name] = preds[i]

    return preds_dict
