#!/bin/bash

export CUDA_HOME=/usr/local/cuda

# export CUDA_LAUNCH_BLOCKING=1

nvidia-smi --version

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    export PATH=/usr/local/cuda/bin:$PATH
    python3 -m pip install --upgrade pip
    python3 -m pip install torch==2.2.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    python3 -m pip install -r requirements/requirements.txt -r requirements/requirements-wheels.txt
    
    git submodule init
    git submodule update

    mkdir ./models/mediapipe
    wget https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task -o ./models/mediapipe/face_landmarker_v2_with_blendshapes.task

    python3 -m pip install -r requirements/requirements.txt -r requirements/requirements-secondary.txt

    python3 -m pip install -r requirements/requirements.txt git+https://github.com/facebookresearch/detectron2
    python3 -m pip install -r requirements/requirements.txt git+https://github.com/facebookresearch/detectron2@main#subdirectory=projects/DensePose
else
    source venv/bin/activate
fi

#./venv/bin/python3 run.py "$@"
accelerate launch --num_processes=1 --num_machines=1 --mixed_precision=bf16 --dynamo_backend=no run.py "$@"

deactivate
