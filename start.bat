@echo off
echo.
echo ==================================================
echo   Starting AI Sequence Predictor (GPU Mode)
echo ==================================================
echo.

set PY_EXE=python312\python.exe
set PYTHONPATH=.

if not exist %PY_EXE% (
    echo [ERROR] Python 3.12 not found. Run install.bat first.
    pause
    exit /b
)

echo [INFO] Launching FastAPI server...
echo [INFO] Application will be available at http://127.0.0.1:8000
echo.

%PY_EXE% src/app.py

pause
