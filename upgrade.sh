#!/bin/bash

git pull
./venv/bin/python -m pip install --upgrade -r requirements/requirements.txt -r requirements/requirements-cuda.txt
./venv/bin/python -m pip install --upgrade -r requirements/requirements-secondary.txt
git submodule init
git submodule update
