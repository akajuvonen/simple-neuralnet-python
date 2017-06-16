#!/bin/bash

source .env/bin/activate
nosetests --with-coverage --cover-package=neuralnet --verbosity=2
deactivate
