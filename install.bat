@echo off
echo.
echo ==================================================
echo   AI Sequence Predictor - Local Installer (3.12)
echo ==================================================
echo.

set PY_EXE=python312\python.exe

if not exist %PY_EXE% (
    echo [ERROR] Python 3.12 not found in python312 folder.
    echo Please make sure the python312 folder exists.
    pause
    exit /b
)

echo [1/2] Upgrading pip...
%PY_EXE% -m pip install --upgrade pip --no-cache-dir

echo [2/2] Installing requirements (GPU enabled)...
%PY_EXE% -m pip install torch --index-url https://download.pytorch.org/whl/cu121 --no-cache-dir
%PY_EXE% -m pip install fastapi uvicorn numpy pandas scikit-learn pydantic python-multipart wikipedia-api --no-cache-dir

echo.
echo ==================================================
echo   Installation Complete! Use start.bat to run.
echo ==================================================
pause
