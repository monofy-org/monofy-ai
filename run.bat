@echo off

setlocal

rem set CUDA_LAUNCH_BLOCKING=1

set CUDA_PATH > nul
if "%errorlevel%" equ "0" goto found
goto notfound

:found
nvidia-smi --version
if errorlevel 1 goto missing
goto next
:missing
echo nvidia-smi.exe was not found. Make sure CUDA is installed.
exit /b %errorlevel%

:notfound
echo CUDA device not found.
exit

:next
if not exist "venv\" (    
    echo Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat 
    python.exe -m pip install --upgrade pip
    python.exe -m pip install -r requirements\requirements-torch.txt -r requirements\requirements-windows.txt    
    python.exe -m pip install -r requirements\requirements-secondary.txt
    python.exe -m pip install git+https://github.com/facebookresearch/detectron2@main#subdirectory=projects/DensePose
    git submodule init
    git submodule update

    md models\mediapipe
    powershell wget https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task -o models\mediapipe\face_landmarker_v2_with_blendshapes.task

    echo Running accelerate config...
    accelerate config
    
    @REM if not exist venv\Lib\site-packages\google\protobuf\internal\builder.py (
    @REM     echo Downloading builder.py...
    @REM     wget https://raw.githubusercontent.com/protocolbuffers/protobuf/main/python/google/protobuf/internal/builder.py -o venv\Lib\site-packages\google\protobuf\internal\builder.py
    @REM )
    
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