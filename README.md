# HyMPI DSI/IRAD ML Retrieval

This code contains everything necessary to build neural networks for retrieving atmospheric parameters such as temperature and humidity profiles or planetary boundary layer height.

#### The Config Object
This contains all of the parts that are used in the `main.py` file. The idea is that any config that you make will create one of these objects.

All of the configs and the `config.py` file can be found in the `src/conf` directory.

The items in the Config object are in order of their usage:
- `mlflow_experiment_name`: this one is a bit self explanatory
    - Note: Make sure this matches the name of an existing MLFlow experiment or else it will make a new one (unless that is intended)
- `dataset_name`: refers to the name of the dataset that will be logged in MLFlow
	- Think `ATMS` or `ATMS+BSL` or `HSEL`
- `modelIOs`: this is a tuple in the form of (train, test) each being an instance of a `ModelIO`
- `model`: a Keras model
- `compile_args`: the arguments used for `model.compile`
- `fit_args`: the arguments used for `model.fit`
- `evaluator`: the evaluator that will be used after fitting the model

*Note: All config files must have an item called `config` that must target the Config object. This is noted at the top of any currently existing config files.*

#### ModelIO
Model IO represents the inputs and outputs of a given model. It mainly works with two things:
- A list of `DataArray` called `features`
- A single `DataArray` called `target`

It contains a few helpful functions for running `model.fit` or `model.evaluate` a given model based on the features and target.

#### DataArray
A very simple object that simply packages together some `ndarray` with any kind of SciKit Learn transformer.
It contains three variables:
- `data`
- `transformer`
- `transformed_data` (populated when `DataArray` is instantiated)

#### Evaluator
Another simple object that contains an `evaluate` function that takes in a set of `truth` and `predicted` data and calculates some metrics or data or whatever you'd like!

The idea is that the base class is something to build on for more specific evaluations you'd like to do! 
You can have one for evaluating profiles and making NetCDF files and such.

Currently, the base class calculates MAE and MSE and that's about it.

#### Splitter
A `Splitter` object is something that creates the train / test datasets. It's one of the first steps of data preprocessing.

It must contain a function called `get_train_test_split` which, given a `DataName` will return a tuple of `ndarray` in the form of (train, test).

There are many kinds of `Splitter` and because they can now load, they even support the ability to load data for each of the train and test sets separately.

#### DataLoader
Just like the old `data_loader.py` file, this object effectively acts as a wrapper for a function that can convert a given `DataName` into an `ndarray` of that specific set of data.

#### DataPreprocessor
This, although seemingly complex, is quite simple as all it does is take a `DataName`, a transformer, and a `Splitter` and creates a tuple of `DataArray` in the form of (train, test). Effectively it splits a dataset but also links the transformers thanks to the `DataArray` object.

The big part comes from the function `create_modelIOs` which is located in the same `DataPreprocessor.py` file. This one can take a list of `DataPreprocessor` for features and a single `DataPreprocessor` for the target data and convert it into train and test `ModelIO`. This is effectively the big central function that does all of the final preprocessing steps and organizes the data to be worked with for model training and evaluating.