#!/bin/bash

source .env/bin/activate
nosetests -v
deactivate
