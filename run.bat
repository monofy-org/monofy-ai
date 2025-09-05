@echo off

setlocal

set CUDA_HOME=%CUDA_PATH%

set CUDA_LAUNCH_BLOCKING=1

nvidia-smi --version
if errorlevel 1 goto missing
goto next

:missing
echo Make sure CUDA is installed.
exit /b %errorlevel%

:next
if not exist "venv\" (    
    echo Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat    
    python.exe -m pip install --upgrade pip
    python.exe -m pip install torch==2.6.0+cu124 torchvision torchaudio wheel ninja --extra-index-url https://download.pytorch.org/whl/cu124
    python.exe -m pip install -r requirements\requirements.txt -r requirements\requirements-wheels.txt

    git submodule init
    git submodule update

    md models\mediapipe
    powershell wget https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task -o models\mediapipe\face_landmarker_v2_with_blendshapes.task
    
    python.exe -m pip install -r requirements\requirements.txt -r requirements\requirements-secondary.txt --extra-index-url https://download.pytorch.org/whl/cu121

    python.exe -m pip install -r requirements\requirements.txt git+https://github.com/facebookresearch/detectron2
    python.exe -m pip install -r requirements\requirements.txt git+https://github.com/facebookresearch/detectron2@main#subdirectory=projects/DensePose

    echo Running accelerate config...
    accelerate config
    
) else (
    call venv\Scripts\activate.bat
)

if not exist venv\Lib\site-packages\google\protobuf\internal\builder.py (
    echo Downloading builder.py...
    powershell -C wget https://raw.githubusercontent.com/protocolbuffers/protobuf/main/python/google/protobuf/internal/builder.py -o venv\Lib\site-packages\google\protobuf\internal\builder.py
)

set ACCELERATE="venv\Scripts\accelerate.exe"
if EXIST %ACCELERATE% goto :accelerate_launch

:launch
python run.py %*
exit /b

:accelerate_launch
echo Using Accelerate
%ACCELERATE% launch --num_cpu_threads_per_process=6 run.py %*
exit /b

call venv\Scripts\deactivate.bat

endlocal