#!/bin/bash

git pull
./venv/bin/python -m pip install --upgrade -r requirements.txt -r requirements-cuda.txt
./venv/bin/python -m pip install --upgrade -r requirements-secondary.txt
git submodule init
git submodule update
