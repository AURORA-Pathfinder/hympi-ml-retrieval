#!/bin/bash
#SBATCH -n1 -G1 -t1440 -c20
uv run python water_vapor.py