# from __future__ import annotations
from dataclasses import dataclass
from typing import List

from keras.models import Model
import keras.layers
from keras.layers import Dense, Dropout, Concatenate, Flatten

@dataclass
class MLPData:
    '''
    Data that can build a Multi-Layer Perceptron (MLP) model
    '''
    input_layer: keras.layers.Layer
    dense_sizes: List[int]
    activation: str
    dropout_rate: int | None
    output_activation: str | None


def build_mlp_layers(mlp_data: MLPData):
    '''
    Builds the layers of a Multi-Layer Perceptron (MLP) model and returns a tuple in the form of (input layer, output layer).
    '''

    sizes = mlp_data.dense_sizes

    is_output = mlp_data.output_activation != None

    if is_output:
        sizes = mlp_data.dense_sizes[0:-1]

    final_layer = mlp_data.input_layer

    for size in sizes:
        final_layer = Dense(size, mlp_data.activation, kernel_regularizer='l1')(final_layer)

        if mlp_data.dropout_rate is not None:
            final_layer = Dropout(mlp_data.dropout_rate)(final_layer)

    if is_output:
        output_size = mlp_data.dense_sizes[-1]
        final_layer = Dense(output_size, mlp_data.output_activation)(final_layer)

    return (mlp_data.input_layer, final_layer)


def build_mlp_model(mlp_data: MLPData):
    '''
    Builds a single-path Multi-Layer Perceptron (MLP) model from MLP data
    '''
    (input_layer, output) = build_mlp_layers(mlp_data)
    return Model(input_layer, output)


def build_multipath_mlp_model(path_mlp_datas: List[MLPData], output_mlp_data: MLPData):
    '''
    Constructs a Multi-Layer Perception (MLP) model consisting of multiple paths with a final
    MLP for the output.
    '''

    paths_layers = [build_mlp_layers(path_data) for path_data in path_mlp_datas]

    input_layers = [path_layers[0] for path_layers in paths_layers]
    path_final_layers = [path_layers[1] for path_layers in paths_layers]

    output = Concatenate(axis=1)(path_final_layers)
    output = Flatten()(output)

    for size in output_mlp_data.dense_sizes[0:-1]:
        output = Dense(size, output_mlp_data.activation, kernel_regularizer='l1')(output)

        if output_mlp_data.dropout_rate is not None:
            output = Dropout(output_mlp_data.dropout_rate)(output)

    output_size = output_mlp_data.dense_sizes[-1]
    output = Dense(output_size, output_mlp_data.output_activation)(output)

    return Model(input_layers, output)
