@echo off

setlocal

rem set CUDA_LAUNCH_BLOCKING=1

set CUDA_PATH > nul
if "%errorlevel%" equ "0" goto found
goto notfound

:found
echo Using CUDA
set USE_CUDA=True
set TORCH_REQ=requirements\requirements-cuda.txt
goto next

:notfound
echo CUDA device not found. Assuming ROCm.
set USE_CUDA=False
set TORCH_REQ=requirements\requirements-rocm.txt

:next
if not exist "venv\" (    
    echo Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat 
    python.exe -m pip install --upgrade pip
    python.exe -m pip install -r requirements\requirements.txt -r %TORCH_REQ%
    python.exe -m pip install -r requirements\requirements-secondary.txt
    git submodule init
    git submodule update    
    if "%USE_CUDA%" equ "False" goto launch

    echo Running accelerate config...
    accelerate config
    
    if not exist venv\Lib\site-packages\google\protobuf\internal\builder.py (
        echo Downloading builder.py...
        wget https://raw.githubusercontent.com/protocolbuffers/protobuf/main/python/google/protobuf/internal/builder.py -o venv\Lib\site-packages\google\protobuf\internal\builder.py
    )
    
    md models\mediapipe
    wget https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task -o models\mediapipe\face_landmarker_v2_with_blendshapes.task
) else (
    call venv\Scripts\activate.bat
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