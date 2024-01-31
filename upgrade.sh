#!/bin/bash

git pull && ./venv/bin/python -m pip install --upgrade -r requirements.txt -r requirements-linux.txt -r requirements-cuda.txt && git submodule init && git submodule update