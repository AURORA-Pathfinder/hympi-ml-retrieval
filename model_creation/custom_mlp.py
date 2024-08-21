from __future__ import annotations
from typing import List

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate, Flatten


# data for a multi-layer perceptron
# note: dropout_rate is has a default value that is low enough that it is the same as not having a dropout at all
class MLPData:
    def __init__(self, layer_sizes: List[int], activation: str, output_activation: str = None, dropout_rate: float = 0.000001) -> None:
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.output_activation = output_activation
        self.dropout_rate = dropout_rate


# builds the layers for an MLP in the functional model and returns a tuple in
#  form of (input layer, final layer)
# If an output_activation is defined in the data, the output layer will not
#  have dropout and will use that activation
# otherwise, all layers will use the activation with the dropout_rate
def build_mlp_layers(mlp_data: MLPData):
    is_output = mlp_data.output_activation != None

    input_layer = Input(shape=(mlp_data.layer_sizes[0],))
    final_layer = input_layer

    if is_output:
        sizes = mlp_data.layer_sizes[1:-1]
    else:
        sizes = mlp_data.layer_sizes[1:]

    for size in sizes:
        final_layer = Dropout(mlp_data.dropout_rate)(Dense(size, mlp_data.activation, kernel_regularizer='l1')(final_layer))

    if is_output:
        output_size = mlp_data.layer_sizes[-1]
        final_layer = Dense(output_size, mlp_data.output_activation)(final_layer)

    return (input_layer, final_layer)


# builds a model from MLPData
def build_mlp_model(mlp_data: MLPData):
    (input_layer, output) = build_mlp_layers(mlp_data)
    return Model(input_layer, output)


# Constructs an mlp consisting of multiple paths with a final MLP for the output
def build_multipath_mlp_model(path_mlp_datas: List[MLPData], output_mlp_data: MLPData):
    paths_layers = [build_mlp_layers(path_data) for path_data in path_mlp_datas]

    input_layers = [path_layers[0] for path_layers in paths_layers]
    path_final_layers = [path_layers[1] for path_layers in paths_layers]

    output = Concatenate(axis=1)(path_final_layers)
    output = Flatten()(output)

    for size in output_mlp_data.layer_sizes[0:-1]:
        output = Dropout(output_mlp_data.dropout_rate)(Dense(size, output_mlp_data.activation, kernel_regularizer='l1')(output))

    output_size = output_mlp_data.layer_sizes[-1]
    output = Dense(output_size, output_mlp_data.output_activation)(output)

    return Model(input_layers, output)
