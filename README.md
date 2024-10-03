# HyMPI ML Retrieval (DSI/IRAD)

This code contains everything necessary to build neural networks for retrieving atmospheric parameters such as temperature and humidity profiles or planetary boundary layer height.

Features include:
- Comprehensive data loading system that works with large-scale all sky per-day data
- Deep integration with MLFlow for expressive logging of model runs and data reproducability
- Custom layers and model creation function specific to our use cases
- Temperature / Humidity Profile evaluation plots (consistent w/ scientific figures)
- NetCDF file generation
- much more!

The primary module that is built with this project is located in the `hympi_ml` directory.

## For Contributors
This project uses [Poetry](https://python-poetry.org/) to manage project dependencies. Installation instructions can be found [here](https://python-poetry.org/docs/). 

Also, an important note directly from the Poetry docs: "Poetry should always be installed in a dedicated virtual environment to isolate it from the rest of your system."

### Install Dependencies
Use `poetry install` to install all dependencies into a new virtual environment. If you'd like to install the dependencies directly within the project directory (which is reccommended), execute the following config command:
```
poetry config virtualenvs.in-project true
```

### Linting
This project uses `ruff` for linting / formatting. The config is is located in the `pyproject.toml` file.
To use ruff directly, run the following command
```
poetry run ruff check
```

If using Visual Studio Code simply install the extension also called "Ruff" to use it in the IDE.