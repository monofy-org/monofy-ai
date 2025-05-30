#!/bin/bash
source venv/bin/activate
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda

git pull
python3 -m pip install --upgrade pip
python3 -m pip install -U -r requirements/requirements.txt -r requirements/requirements-wheels.txt
python3 -m pip install -r requirements/requirements.txt -r requirements/requirements-secondary.txt
git submodule init
git submodule update
