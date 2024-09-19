#!/bin/bash
source venv/bin/activate

git pull
python3 -m pip install --upgrade -r requirements/requirements-wsl.txt
python3 -m pip install --upgrade -r requirements/requirements-secondary.txt
git submodule init
git submodule update
