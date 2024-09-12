#!/bin/bash

export CUDA_LAUNCH_BLOCKING=1

# Check if NVIDIA GPU driver is installed
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU driver detected."
else
    echo "NVIDIA GPU driver not found."
    exit 1
fi

TORCH_INDEX_URL=https://download.pytorch.org/whl/nightly/cu121

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    python3 -m pip install --upgrade pip    
    python3 -m pip install -r requirements/requirements-cuda.txt -r requirements/requirements.txt -r requirements/requirements-secondary.txt -r requirements/requirements-tts.txt
    git submodule init
    git submodule update
    if [ "$USE_CUDA" = "False" ]; then
        ./venv/bin/python3 run.py "$@"
        exit
    else
        echo "Running accelerate config..."
        accelerate config
    fi
else
    source venv/bin/activate
fi

#./venv/bin/python3 run.py "$@"
accelerate launch --num_processes=1 --num_machines=1 --mixed_precision=no --dynamo_backend=no run.py "$@"

deactivate
