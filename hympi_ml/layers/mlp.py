from typing import Any, List, Tuple

from keras.layers import Layer, Dense, Dropout


def get_dense_layers(
    input_layer: Any,
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


def get_autoencoder_layers(
    input_layer: Any,
    input_size: int,
    encode_sizes: List[int],
    latent_size: int,
    activation: str | Any,
    dropout_rate: float,
    **dense_args,
) -> Tuple[Layer, Layer]:
    """
    Generates a chain of layers representing an Autoencoder with a given latent size.


    Args:
        input_layer (Any): _description_
        encode_sizes (List[int]): _description_
        latent_size (int): _description_
        activation (str | Any): _description_
        dropout_rate (float): _description_

    Returns:
        Tuple[Layer, Layer]: A tuple with the autoencoder and encoder layers respectively.
            Specifically in the form of (autoencoder, encoder).
    """

    encoder = get_dense_layers(input_layer, encode_sizes, activation, dropout_rate, **dense_args)
    encoder = (Dense(latent_size, activation=activation, **dense_args))(encoder)
    encode_sizes.reverse()
    autoencoder = get_dense_layers(encoder, encode_sizes, activation, dropout_rate, **dense_args)
    autoencoder = (Dense(input_size))(autoencoder)

    return (autoencoder, encoder)
