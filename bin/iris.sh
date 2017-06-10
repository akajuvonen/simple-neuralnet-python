#!/bin/bash

source .env/bin/activate
export PYTHONPATH=.
python tests/iris_test.py
deactivate
