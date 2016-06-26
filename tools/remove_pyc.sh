#!/bin/sh

# Removes all pyc-files from directory and all subdirectories.
# Run from /tools directory.

find .. -name \*.pyc -type f -delete
