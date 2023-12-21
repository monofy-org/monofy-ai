@echo off

setlocal

set CUDA_LAUNCH_BLOCKING=1

set CUDA_PATH > nul
if "%errorlevel%" equ "0" goto found
goto notfound

:found
echo Using CUDA.
set TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
goto next

:notfound
echo Using ROCm.
set TORCH_INDEX_URL=https://download.pytorch.org/whl/nightly/rocm5.7

:next
if not exist "venv\" (    
    echo Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat
    python.exe -m pip install --upgrade pip
    python.exe -m pip install -r requirements.txt --extra-index-url %TORCH_INDEX_URL%
    accelerate config
) else (
    call venv\Scripts\activate.bat
)

rem python run.py %*

accelerate launch --num_processes=1 --num_machines=1 --mixed_precision=no --dynamo_backend=no run.py %* 

call venv\Scripts\deactivate.bat

endlocal