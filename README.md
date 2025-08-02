# HyMPI ML Retrieval (DSI/IRAD)

This code contains everything necessary to build neural networks for retrieving atmospheric parameters such as temperature and humidity profiles or planetary boundary layer height.

#### *Submitted to IEEE TRANSACTIONS ON GEOSCIENCE AND REMOTE SENSING: Improved Planetary Boundary Layer Sounding Using Hyperspectral Microwave and Backscatter Lidar Data Fusion*

Features include:
- Comprehensive and expandable data specifications system for defining model variations.
- Deep integration with PyTorch and PyTorch Lightning for training and serialization with checkpoints.
- Productive metrics system for deep analysis of model outputs and data.
- much more!

The primary module that is built with this project is located in the `src` directory.

## For Contributors
This project uses [uv](https://docs.astral.sh/uv/) to manage project dependencies. Installation instructions can be found [here](https://docs.astral.sh/uv/getting-started/installation/). 

### Install Dependencies
Use `uv sync` to install all dependencies into a **new** virtual environment.

Note: uv is not python dependent and thus can create virtual environments on it's own.
