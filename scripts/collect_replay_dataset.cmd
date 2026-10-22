@echo off
rem Fully automatic dataset collection from TMX replays:
rem   1. download keyboard A01-Race replays from TMNF Exchange (resumes),
rem   2. open TrackMania via TMLoader if it is not already running,
rem   3. drive every replay in the game and record frames + telemetry.
rem Usage: collect_replay_dataset.cmd [laps]   (default 300)
rem Safe to stop with Ctrl+C and run again; finished laps are skipped.
setlocal
set "PROJECT_ROOT=%~dp0.."
set "PYTHON=%PROJECT_ROOT%\.venv\Scripts\python.exe"
set "LAPS=%~1"
if "%LAPS%"=="" set "LAPS=300"

if not exist "%PYTHON%" (
    echo Project Python was not found at "%PYTHON%" 1>&2
    exit /b 1
)

echo === Step 1/2: downloading up to %LAPS% keyboard replays from TMNF Exchange ===
"%PYTHON%" "%~dp0fetch_tmx_replays.py" --keyboard-count %LAPS%
if errorlevel 1 (
    echo Replay download failed; nothing was recorded. 1>&2
    exit /b %errorlevel%
)

echo === Step 2/2: opening TrackMania if needed and recording every replay ===
"%PYTHON%" "%~dp0record_replays.py"
exit /b %errorlevel%
