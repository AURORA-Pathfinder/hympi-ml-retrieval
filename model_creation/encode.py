import keras.layers
import keras.models
from keras.saving.experimental.saving_lib import serialize_keras_object, deserialize_keras_object

import mlflow

class Encode(keras.layers.Layer):
    '''
    A Keras model layer that encodes the input with a provided encoder model
    '''
    def __init__(self, encoder_path: str):
        '''        
        Creates a new encoder layer from a path to a Keras model
        '''
        super().__init__(trainable=False)
        self.encoder_path = encoder_path
        self.encoder = keras.models.load_model(encoder_path)
    
    @classmethod
    def from_mlflow_artifact(cls, experiment_name: str, run_name: str, artifact_path: str):
        '''
        Creates a new encoder layer from an MLFlow artifact path given an experiment and run name.
        '''
        experiment = mlflow.get_experiment_by_name(experiment_name)
        exp_path = experiment.artifact_location

        search = mlflow.search_runs(
            experiment_names=[experiment_name], 
            filter_string=f"run_name='{run_name}'"
        )

        run_id = search['run_id'][0]

        path = f"{exp_path}/{run_id}/artifacts/{artifact_path}"
        return cls(path)

    def call(self, inputs):
        return self.encoder(inputs)
    
    def get_config(self):
        base_config = super().get_config()
        config = {
            "encoder_path": serialize_keras_object(self.encoder_path),
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        config = config.pop("encoder_path")
        encoder_path = deserialize_keras_object(config)
        return cls(encoder_path, **config)
    
    