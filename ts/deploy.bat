rem change to directory of bat file
cd %~dp0

rem make sure folder for arg exists

if not exist %1 (
    echo Folder %1 does not exist
    exit /b
)

rem copy %1\dist contents to ..\public_html\%1
xcopy %1\dist ..\public_html\%1 /s /e /y
