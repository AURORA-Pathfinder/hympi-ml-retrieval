#!/bin/bash
#SBATCH -n1 -G1 -t1000 -c20
uv run python test_lightning.py