#!/bin/bash

export CUDA_HOME=/usr/local/cuda

# export CUDA_LAUNCH_BLOCKING=1

nvidia-smi --version

if [ ! -d "venv" ]; then
    export TORCH_VERSION=torch==2.6.0+cu124
    export EXTRA_INDEX_URL=--extra-index-url https://download.pytorch.org/whl/cu124
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    export PATH=/usr/local/cuda/bin:$PATH
    python3 -m pip install --upgrade pip
    python3 -m pip install $TORCH_VERSION torch==2.6.0+cu124 torchvision torchaudio wheel $EXTRA_INDEX_URL
    python3 -m pip install $TORCH_VERSION -r requirements/requirements.txt -r requirements/requirements-wheels.txt $EXTRA_INDEX_URL
    
    git submodule init
    git submodule update

    mkdir ./models/mediapipe
    wget https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task -o ./models/mediapipe/face_landmarker_v2_with_blendshapes.task

    python3 -m pip install $TORCH_VERSION -r requirements/requirements-secondary.txt $EXTRA_INDEX_URL

    python3 -m pip install $TORCH_VERSION git+https://github.com/facebookresearch/detectron2 $EXTRA_INDEX_URL
    python3 -m pip install $TORCH_VERSION git+https://github.com/facebookresearch/detectron2@main#subdirectory=projects/DensePose $EXTRA_INDEX_URL
else
    source venv/bin/activate
fi

if [ ! -f "venv/Lib/site-packages/google/protobuf/internal/builder.py" ]; then
    echo Downloading builder.py...
    wget https://raw.githubusercontent.com/protocolbuffers/protobuf/main/python/google/protobuf/internal/builder.py -O venv/Lib/site-packages/google/protobuf/internal/builder.py
fi

#./venv/bin/python3 run.py "$@"
accelerate launch --num_processes=1 --num_machines=1 --mixed_precision=bf16 --dynamo_backend=no run.py "$@"

deactivate
