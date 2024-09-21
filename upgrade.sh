#!/bin/bash
source venv/bin/activate
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda

git pull
python3 -m pip install --upgrade pip
python3 -m pip install --U -r requirements/requirements-wsl.txt
python3 -m pip install -r requirements/requirements-secondary.txt
git submodule init
git submodule update
