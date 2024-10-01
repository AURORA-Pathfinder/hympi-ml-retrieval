from typing import Any, List

from keras.layers import Layer, Dense, Dropout


def get_dense_layers(
    input_layer: Any | list,
    sizes: List[int],
    activation: str | Any,
    dropout_rate: float,
    **dense_args,
) -> Layer:
    """
    Creates a chain of dense layers for each size in the sizes list with the same activation, and dropout_rate.

    Additional kwargs are available for more other dense layer arguments.

    Args:
        sizes (List[int]): The list of sizes for each dense layer
        activation (str | Any): the shared activation used for all dense layers
        dropout_rate (float): the shared dropout rate for all dense layers (if 0, no dropout layer is added)
        dense_args (**kwargs):
            Additional keyword arguments that are applied to all dense layers (ex. kernal_regularizer, etc)

    Returns:
        Layer: The final layer in the chain
    """

    layers = []

    for size in sizes:
        dense = Dense(units=size, activation=activation, **dense_args)
        layers.append(dense)

        if dropout_rate > 0:
            layers.append(Dropout(dropout_rate))

    final_layer = input_layer

    for layer in layers:
        final_layer = (layer)(final_layer)

    return final_layer
