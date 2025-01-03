@echo off
setlocal enabledelayedexpansion

:: Check if the source file exists
if "%~1"=="" (
    echo No file selected. Drag and drop an .rgb file onto this script.
    pause
    exit /b
)

:: Get the source file path
set SOURCE=%~1

:: Check if the source file is an .rgb file
if /i not "%SOURCE:~-4%"==".rgb" (
    echo The file is not a .rgb file. Please drag and drop a .rgb file.
    pause
    exit /b
)

:: Create the target directory if it doesn't exist
set TARGET_DIR=Skate 1 RGBs
if not exist "%TARGET_DIR%" (
    mkdir "%TARGET_DIR%"
)

:: List of filenames to rename and copy to the target directory
set FILENAMES=cc_danny.rgb cc_financial.rgb cc_oldtown.rgb cc_outskirts.rgb cc_park.rgb cc_planb.rgb cc_suburbs.rgb cc_urbanrez.rgb cc_xgames.rgb

:: Loop through each filename and copy the source file with the new name
for %%f in (%FILENAMES%) do (
    copy "%SOURCE%" "%TARGET_DIR%\%%f"
)

echo Files have been copied and renamed successfully.
pause
