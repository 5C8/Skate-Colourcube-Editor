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
set TARGET_DIR=Skate 2 RGBs
if not exist "%TARGET_DIR%" (
    mkdir "%TARGET_DIR%"
)

:: List of filenames to rename and copy to the target directory
set FILENAMES=cc_70s.rgb cc_bigevent.rgb cc_bkup1.rgb cc_bkup2.rgb cc_bkup3.rgb cc_bkup4.rgb cc_creature.rgb cc_danny.rgb cc_financial.rgb cc_neutral.rgb cc_oldtown.rgb cc_park.rgb cc_petes.rgb cc_projects.rgb cc_pure.rgb cc_slappy.rgb cc_slappy_warehouse.rgb cc_soho.rgb cc_soho_skatepark.rgb cc_suburbs.rgb cc_svm.rgb cc_urbanrez.rgb cc_waterfront.rgb

:: Loop through each filename and copy the source file with the new name
for %%f in (%FILENAMES%) do (
    copy "%SOURCE%" "%TARGET_DIR%\%%f"
)

echo Files have been copied and renamed successfully.
pause
