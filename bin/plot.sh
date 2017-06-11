#!/bin/bash

source .env/bin/activate
export PYTHONPATH=.
python tools/sigmoid_plotter.py
deactivate
