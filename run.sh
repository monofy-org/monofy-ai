#!/bin/bash

# export CUDA_LAUNCH_BLOCKING=1

nvidia-smi --version

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    export PATH=/usr/local/cuda/bin:$PATH
    python3 -m pip install --upgrade pip    
    python3 -m pip install -r requirements/requirements-wsl.txt
    python3 -m pip install -r requirements/requirements-secondary.txt
    python3 -m pip install git+https://github.com/facebookresearch/detectron2@main#subdirectory=projects/DensePose
    mkdir ./models/mediapipe
    wget https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task -o ./models/mediapipe/face_landmarker_v2_with_blendshapes.task
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
