#!/bin/sh
# Reproduce every number, table and figure of the PlantMONT-Risk paper.
# Full campaign: about 2.5 hours on 2 CPU cores. Delete results/*.json to rerun from scratch.
set -e
cd "$(dirname "$0")"
python -m pytest -q test_sim.py
python run_experiments.py all
python gap.py
python scaling.py
python make_outputs.py
