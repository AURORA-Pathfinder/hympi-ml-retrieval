# HyMPI DSI/IRAD ML Retrieval

This code contains everything necessary to build neural networks for retrieving atmospheric parameters such as temperature and humidity profiles or planetary boundary layer height.

## Project Structure
### Preprocessing

#### Data Loading
Loading our data involves the `FullDaysLoader` which is capapble of taking in a reference to a subset of the "fulldays" dataset for a given number of days. One can create a different loader for each set of days. Such as, for a training, testing, or validation dataset.

#### Memory Maps and Memmap Sequence
As our data grows in size, there is a need to be able to load large amounts of data quickly and efficiently. This is why the data loading process returns a `MemmapSequence` which is an abstraction for a list of numpy memory maps or "memmap". Each memmap can be sliced or indexed without needing to load the entire set of data. Despite this, our data is fragmented along many days of data and thus, many different memmaps. `MemmapSequence` provides an abstraction layer above that list of memmaps and allows a user to treat it as a single larger, concatenated dataset.

### Model Creation
Part of the "model_creation" directory is a set of modules for supporting the model creation process. Everything from creating input layers, normalization layers, or even custom layers and loss functions.

### Evaluation
In the "evaluation" directory is another set of modules that contain useful plotting of figures and even easy means of working with MLFlow for logging various model metrics after training.

## For Contributors
This project uses [Poetry](https://python-poetry.org/) to handle dependencies. Installation instructions can be found [here](https://python-poetry.org/docs/). 
Also, an import note directly from the Poetry docs: "Poetry should always be installed in a dedicated virtual environment to isolate it from the rest of your system."

### Install Dependencies
Use 'poetry install' to install all dependencies into a new virtual environment. If you'd like to install the dependencies directly within the project directory (which is reccommended), execute the following config command:
'''
poetry config virtualenvs.in-project true
'''