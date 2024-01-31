@echo off
setlocal enabledelayedexpansion

call venv\scripts\activate.bat

git pull

REM Get current date and time
for /f "delims=" %%a in ('wmic OS Get localdatetime ^| find "."') do set datetime=%%a
set "YYYY=!datetime:~0,4!"
set "MM=!datetime:~4,2!"
set "DD=!datetime:~6,2!"
set "HH=!datetime:~8,2!"
set "MIN=!datetime:~10,2!"

REM Set the rollback directory
set "rollback_dir=.rollback"

REM Create the rollback directory if it doesn't exist
if not exist "%rollback_dir%" (
    mkdir "%rollback_dir%"
)

REM Generate a unique file name
set "filename=requirements-%YYYY%%MM%%DD%-%HH%%MIN%.txt"

REM Pipe the output of pip freeze to the unique file
pip freeze > "%rollback_dir%\%filename%"

echo Requirements exported to %rollback_dir%\%filename%

set CUDA_PATH
if "%errorlevel%" equ "0" goto found
goto notfound

:found
echo Using CUDA
set USE_CUDA = 1
set TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
goto next

:notfound
set USE_CUDA = 0
echo Using ROCm
set TORCH_INDEX_URL=https://download.pytorch.org/whl/nightly/rocm5.7

:next
python.exe -m pip install -r requirements.txt --upgrade --extra-index-url %TORCH_INDEX_URL%
python.exe -m pip install -r requirements-win.txt --upgrade
if "%USE_CUDA%" equ "0" goto skip_cuad
python.exe -m pip install -r requirements-cuda.txt --upgrade

git submodule init
git submodule update
