@echo off
setlocal enabledelayedexpansion

call venv\scripts\activate.bat

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

pip install -r requirements.txt --upgrade
