@echo off

setlocal

rem set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1

if not exist "venv\" (    
    echo Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat
    python.exe -m pip install --upgrade pip
    python.exe -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
) else (
    call venv\Scripts\activate.bat
)

rem python main.py %*

python run.py %*

call venv\Scripts\deactivate.bat

endlocal