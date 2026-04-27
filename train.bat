@echo off
echo.
echo ==================================================
echo   Retraining AI Sequence Predictor (GPU)
echo ==================================================
echo.

set PY_EXE=python312\python.exe
set PYTHONPATH=.

if not exist %PY_EXE% (
    echo [ERROR] Python 3.12 not found. Run install.bat first.
    pause
    exit /b
)

echo [INFO] Starting 1,000,000 word training session on GPU...
%PY_EXE% src/train.py

echo.
echo [INFO] Training finished. You can now use start.bat.
pause
